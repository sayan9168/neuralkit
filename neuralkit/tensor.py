"""Core Tensor with reverse-mode automatic differentiation.

A NumPy-backed dynamic computation graph, similar in spirit to micrograd /
tinyautograd but production-hardened:

* Topologically-sorted backprop (each node's backward runs exactly once).
* Correct gradient accumulation for reused nodes.
* Broadcasting-aware gradient reduction.
* ``no_grad`` context manager / decorator for inference.
* Gradient checkpointing helper (:func:`checkpoint`).
"""

from __future__ import annotations

import numpy as np

from neuralkit.exceptions import GradientError, ShapeError, TensorError

__all__ = ["Tensor", "no_grad", "is_grad_enabled", "checkpoint"]

_GRAD_ENABLED = True


class _NoGrad:
    """Context manager / decorator that disables gradient tracking."""

    def __enter__(self) -> "_NoGrad":
        global _GRAD_ENABLED
        self._prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev

    def __call__(self, func=None):
        # Usable both as ``with no_grad():`` sibling decorator form:
        #   @no_grad()          -> returns context manager (this object)
        #   @no_grad            -> wraps function directly
        if func is None:
            return self

        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        wrapper.__name__ = getattr(func, "__name__", "wrapper")
        return wrapper


no_grad = _NoGrad()


def is_grad_enabled() -> bool:
    """Return True when gradient tracking is currently enabled."""
    return _GRAD_ENABLED


def _as_array(data, dtype) -> np.ndarray:
    if isinstance(data, Tensor):
        data = data.data
    try:
        return np.asarray(data, dtype=dtype)
    except (TypeError, ValueError) as e:
        raise TensorError(f"Cannot convert data to array: {e}") from e


def _reduce_broadcast(g: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum gradient down to a broadcasted operand's original shape."""
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


class Tensor:
    """An n-dimensional array with automatic differentiation support."""

    def __init__(self, data, requires_grad: bool = False, dtype=np.float32):
        self.data: np.ndarray = _as_array(data, dtype)
        if self.data.ndim == 0:
            self.data = self.data.reshape(1)
        self.requires_grad: bool = bool(requires_grad)
        self.grad: np.ndarray | None = np.zeros_like(self.data) if requires_grad else None
        self.parents: tuple = ()
        self._op: str | None = None
        self._meta: dict = {}

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def array(data, requires_grad: bool = False, dtype=np.float32) -> "Tensor":
        return Tensor(data, requires_grad=requires_grad, dtype=dtype)

    @staticmethod
    def zeros(*shape, requires_grad: bool = False, dtype=np.float32) -> "Tensor":
        return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad, dtype=dtype)

    @staticmethod
    def ones(*shape, requires_grad: bool = False, dtype=np.float32) -> "Tensor":
        return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad, dtype=dtype)

    @staticmethod
    def randn(*shape, scale: float = 1.0, seed: int | None = None,
              requires_grad: bool = False, dtype=np.float32) -> "Tensor":
        rng = np.random.default_rng(seed)
        return Tensor(rng.standard_normal(shape) * scale, requires_grad=requires_grad, dtype=dtype)

    @staticmethod
    def rand(*shape, seed: int | None = None, requires_grad: bool = False,
             dtype=np.float32) -> "Tensor":
        rng = np.random.default_rng(seed)
        return Tensor(rng.random(shape), requires_grad=requires_grad, dtype=dtype)

    @staticmethod
    def arange(*args, requires_grad: bool = False, dtype=np.float32) -> "Tensor":
        return Tensor(np.arange(*args, dtype=dtype), requires_grad=requires_grad, dtype=dtype)

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return int(self.data.size)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def value(self) -> float:
        """Scalar value (for loss tensors)."""
        if self.data.size != 1:
            raise TensorError("value requires a scalar tensor")
        return float(self.data.reshape(-1)[0])

    def item(self) -> float:
        return self.value

    def detach(self) -> "Tensor":
        """Return a new leaf Tensor with copied data, detached from the graph."""
        return Tensor(self.data.copy(), requires_grad=False, dtype=self.dtype)

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def tolist(self):
        return self.data.tolist()

    def clone(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=self.requires_grad, dtype=self.dtype)

    def copy_(self, other) -> "Tensor":
        """In-place copy of data (used by optimizers / weight loading)."""
        src = other.data if isinstance(other, Tensor) else np.asarray(other)
        if src.shape != self.data.shape:
            raise ShapeError(f"copy_ shape mismatch {src.shape} vs {self.data.shape}")
        self.data = src.astype(self.dtype, copy=False)
        return self

    def cast(self, dtype) -> "Tensor":
        return Tensor(self.data.astype(dtype), requires_grad=self.requires_grad, dtype=dtype)

    # ------------------------------------------------------------------ #
    # Graph plumbing
    # ------------------------------------------------------------------ #
    def _register(self, parents, op: str) -> "Tensor":
        if is_grad_enabled():
            tracked = tuple(p for p in parents
                            if isinstance(p, Tensor) and p.requires_grad)
            if tracked:
                # Gradient tracking propagates through the graph.  Only keep
                # differentiable parents; backward functions receive exactly
                # these tensors (values needed from constants are stored in
                # ``_meta``).
                self.requires_grad = True
                self.parents = tracked
                self._op = op
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
        return self

    def zero_grad(self) -> None:
        """Reset this tensor's gradient to zero (in place)."""
        if self.grad is not None:
            self.grad[...] = 0.0

    def backward(self, gradient=None) -> None:
        """Reverse-mode autodiff from this (scalar by default) tensor."""
        if not is_grad_enabled():
            raise GradientError("backward() called inside no_grad context")
        if not self.requires_grad:
            raise GradientError("Calling backward on a tensor that doesn't require grad")

        if gradient is None:
            if self.data.size != 1:
                raise GradientError("grad must be provided for non-scalar tensors")
            seed = np.ones_like(self.data)
        else:
            seed = _as_array(gradient, self.dtype)

        # Iterative DFS post-order -> topological order (root first)
        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, processed = stack.pop()
            nid = id(node)
            if processed:
                order.append(node)
                continue
            if nid in visited:
                continue
            visited.add(nid)
            stack.append((node, True))
            for p in node.parents:
                if id(p) not in visited:
                    stack.append((p, False))
        order.reverse()

        acc: dict[int, np.ndarray] = {id(self): seed}

        for node in order:
            g = acc.pop(id(node), None)
            if g is None:
                continue
            if node.grad is not None:
                node.grad += g
            if not node.parents:  # leaf: nothing to propagate further
                continue
            fn = _BACKWARD_FUNCS.get(node._op)
            if fn is None:
                continue
            pg = fn(node, g)
            if pg is None:
                continue
            for p, gp in zip(node.parents, pg):
                if not p.requires_grad:  # untracked parent: drop its gradient
                    continue
                gp = _reduce_broadcast(gp, p.shape)
                pid = id(p)
                if pid in acc:
                    acc[pid] = acc[pid] + gp
                else:
                    acc[pid] = gp

    # ------------------------------------------------------------------ #
    # Elementwise binary ops (broadcasting aware)
    # ------------------------------------------------------------------ #
    def _binary(self, other, op: str) -> "Tensor":
        other_t = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        a, b = self.data, other_t.data
        meta: dict = {}
        if op == "add":
            out = a + b
        elif op == "sub":
            out = a - b
        elif op == "mul":
            out = a * b
            if not other_t.requires_grad and self.requires_grad:
                meta["other_value"] = b  # constant multiplier
        elif op == "div":
            out = a / b
            if not other_t.requires_grad and self.requires_grad:
                meta["single"] = "den"
                meta["other_value"] = b
            elif other_t.requires_grad and not self.requires_grad:
                meta["single"] = "num"
                meta["other_value"] = a
        elif op == "pow":
            out = np.power(a, b)
            if not other_t.requires_grad and self.requires_grad:
                meta["exponent"] = b
        else:  # pragma: no cover
            raise TensorError(f"unknown op {op}")
        t = Tensor(out, dtype=self.dtype)
        t._meta = meta
        return t._register((self, other_t), op)

    def __add__(self, other):
        return self._binary(other, "add")

    def __radd__(self, other):
        return self._binary(other, "add")

    def __sub__(self, other):
        return self._binary(other, "sub")

    def __rsub__(self, other):
        return (-self)._binary(other, "add")

    def __mul__(self, other):
        return self._binary(other, "mul")

    def __rmul__(self, other):
        return self._binary(other, "mul")

    def __truediv__(self, other):
        return self._binary(other, "div")

    def __rtruediv__(self, other):
        return Tensor(other, dtype=self.dtype)._binary(self, "div")

    def __pow__(self, other):
        return self._binary(other, "pow")

    def __neg__(self):
        return self._unary("neg")

    def __pos__(self):
        return self

    def __abs__(self):
        return self._unary("abs")

    def _unary(self, op: str) -> "Tensor":
        x = self.data
        if op == "neg":
            out = -x
        elif op == "abs":
            out = np.abs(x)
        elif op == "exp":
            out = np.exp(np.clip(x, -60, 60))
        elif op == "log":
            out = np.log(np.maximum(x, 1e-12))
        elif op == "sqrt":
            out = np.sqrt(np.maximum(x, 0.0))
        elif op == "tanh":
            out = np.tanh(x)
        elif op == "sigmoid":
            out = 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
        elif op == "sin":
            out = np.sin(x)
        elif op == "cos":
            out = np.cos(x)
        else:  # pragma: no cover
            raise TensorError(f"unknown unary op {op}")
        t = Tensor(out, dtype=self.dtype)
        return t._register((self,), op)

    def exp(self):
        return self._unary("exp")

    def abs(self):
        return self._unary("abs")

    def log(self):
        return self._unary("log")

    def sqrt(self):
        return self._unary("sqrt")

    def tanh(self):
        return self._unary("tanh")

    def sigmoid(self):
        return self._unary("sigmoid")

    def sin(self):
        return self._unary("sin")

    def cos(self):
        return self._unary("cos")

    # ------------------------------------------------------------------ #
    # Reductions & reshaping
    # ------------------------------------------------------------------ #
    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        t = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), dtype=self.dtype)
        t._meta = {"axis": axis, "keepdims": keepdims}
        return t._register((self,), "sum")

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        t = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), dtype=self.dtype)
        t._meta = {"axis": axis, "keepdims": keepdims}
        return t._register((self,), "mean")

    def max(self, axis=None, keepdims: bool = False) -> "Tensor":
        t = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), dtype=self.dtype)
        t._meta = {"axis": axis, "keepdims": keepdims}
        return t._register((self,), "max")

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        t = Tensor(self.data.reshape(shape), dtype=self.dtype)
        return t._register((self,), "reshape")

    def flatten(self, start_dim: int = 1) -> "Tensor":
        return self.reshape(self.shape[:start_dim] + (-1,))

    def transpose(self, dim0: int = -2, dim1: int = -1) -> "Tensor":
        t = Tensor(np.swapaxes(self.data, dim0, dim1), dtype=self.dtype)
        t._meta = {"dims": (dim0, dim1)}
        return t._register((self,), "transpose")

    def permute(self, *dims) -> "Tensor":
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        t = Tensor(np.transpose(self.data, dims), dtype=self.dtype)
        t._meta = {"dims": dims}
        return t._register((self,), "permute")

    def expand_dims(self, axis: int = 0) -> "Tensor":
        t = Tensor(np.expand_dims(self.data, axis), dtype=self.dtype)
        t._meta = {"axis": axis}
        return t._register((self,), "expand")

    def matmul(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(other, dtype=self.dtype)
        try:
            out = self.data @ other.data
        except ValueError as e:
            raise ShapeError(f"matmul {self.shape} @ {other.shape}: {e}") from e
        t = Tensor(out, dtype=self.dtype)
        return t._register((self, other), "matmul")

    def __matmul__(self, other):
        return self.matmul(other)

    def dot(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    @staticmethod
    def stack(tensors, axis: int = 0) -> "Tensor":
        arrs = [t.data for t in tensors]
        out = np.stack(arrs, axis=axis)
        t = Tensor(out, dtype=tensors[0].dtype)
        t._meta = {"axis": axis, "shapes": tuple(x.shape for x in tensors)}
        return t._register(tuple(tensors), "stack")

    def clip(self, lo: float, hi: float) -> "Tensor":
        t = Tensor(np.clip(self.data, lo, hi), dtype=self.dtype)
        t._meta = {"clip": (lo, hi)}
        return t._register((self,), "clip")

    # ------------------------------------------------------------------ #
    # Non-differentiable helpers
    # ------------------------------------------------------------------ #
    def argmax(self, axis: int = -1):
        return np.argmax(self.data, axis=axis)

    def max_value(self) -> float:
        return float(self.data.max())

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"


# ---------------------------------------------------------------------- #
# Backward functions registry
# ---------------------------------------------------------------------- #
def _bw_add(node, g):
    return (g,) * len(node.parents)


def _bw_sub(node, g):
    if len(node.parents) == 2:
        return (g, -g)
    # reversed subtraction fallback (rsub keeps both parents tracked)
    return (g,) * len(node.parents)


def _bw_mul(node, g):
    # Multiply is commutative: gradient wrt each tracked parent is
    # g * product(values of the other parents).
    if len(node.parents) == 1:
        # scalar constant multiplication; value stored in meta by _binary
        c = node._meta.get("other_value")
        return (g * c,)
    a, b = node.parents
    return (g * b.data, g * a.data)


def _bw_div(node, g):
    m = node._meta
    if len(node.parents) == 1 and m.get("single") == "num":
        # out = c / x  ->  d/dx = -c / x^2
        c = m["other_value"]
        x = node.parents[0].data
        return (-g * c / (x * x),)
    if len(node.parents) == 1 and m.get("single") == "den":
        # out = x / c  ->  d/dx = 1/c
        c = m["other_value"]
        return (g / c,)
    a, b = node.parents
    return (g / b.data, -g * a.data / (b.data * b.data))


def _bw_pow(node, g):
    m = node._meta
    if len(node.parents) == 1:
        p = node.parents[0]
        e = m.get("exponent", m.get("other_value"))
        safe = np.where(np.abs(p.data) < 1e-12, 1e-12, p.data)
        return (g * e * np.power(safe, e - 1),)
    a, b = node.parents
    safe = np.where(np.abs(a.data) < 1e-12, 1e-12, a.data)
    ga = g * b.data * np.power(np.abs(safe), b.data - 1)
    gb = g * node.data * np.log(np.maximum(np.abs(a.data), 1e-12))
    return (ga, gb)


def _bw_neg(node, g):
    return (-g,)


def _bw_abs(node, g):
    return (g * np.sign(node.parents[0].data),)


def _bw_exp(node, g):
    return (g * node.data,)


def _bw_log(node, g):
    return (g / np.maximum(node.parents[0].data, 1e-12),)


def _bw_sqrt(node, g):
    return (g / (2.0 * np.maximum(node.data, 1e-12)),)


def _bw_tanh(node, g):
    return (g * (1.0 - node.data * node.data),)


def _bw_sigmoid(node, g):
    s = node.data
    return (g * s * (1.0 - s),)


def _bw_sin(node, g):
    return (g * np.cos(node.parents[0].data),)


def _bw_cos(node, g):
    return (-g * np.sin(node.parents[0].data),)


def _reduction_backward(node, g, factor_fn):
    m = node._meta
    axis, keepdims = m.get("axis"), m.get("keepdims", False)
    x = node.parents[0]
    g = factor_fn(g, x, axis, m)
    if not keepdims:
        if axis is None:
            g = g.reshape([1] * x.ndim)
        else:
            g = np.expand_dims(g, axis=axis)
    return (np.broadcast_to(g, x.shape).copy(),)


def _bw_sum(node, g):
    return _reduction_backward(node, g, lambda g, x, ax, m: g)


def _bw_mean(node, g):
    def factor(g, x, ax, m):
        if ax is None:
            n = x.size
        elif isinstance(ax, int):
            n = x.shape[ax]
        else:
            n = int(np.prod([x.shape[a] for a in ax]))
        return g / n
    return _reduction_backward(node, g, factor)


def _bw_max(node, g):
    m = node._meta
    axis, keepdims = m.get("axis"), m.get("keepdims", False)
    x = node.parents[0]
    xd, md = x.data, node.data
    if axis is None:
        mask = (xd == md).astype(xd.dtype)
        mask /= max(mask.sum(), 1)
        gg = mask * g.reshape([1] * x.ndim)
    else:
        md_e = md if keepdims else np.expand_dims(md, axis=axis)
        g_e = g if keepdims else np.expand_dims(g, axis=axis)
        md_b = np.broadcast_to(md_e, xd.shape)
        mask = (xd == md_b).astype(xd.dtype)
        cnt = mask.sum(axis=axis, keepdims=True)
        mask = mask / np.maximum(cnt, 1)
        gg = mask * np.broadcast_to(g_e, xd.shape)
    return (gg,)


def _bw_reshape(node, g):
    return (g.reshape(node.parents[0].shape),)


def _bw_transpose(node, g):
    d0, d1 = node._meta["dims"]
    return (np.swapaxes(g, d0, d1),)


def _bw_permute(node, g):
    dims = node._meta["dims"]
    inv = tuple(int(i) for i in np.argsort(dims))
    return (np.transpose(g, inv),)


def _bw_expand(node, g):
    axis = node._meta["axis"]
    return (g.sum(axis=axis),)


def _bw_matmul(node, g):
    a, b = node.parents
    if a.ndim == 2 and b.ndim == 1:
        ga = np.outer(g, b.data)
        gb = a.data.T @ g
        return (ga, gb)
    ga = g @ np.swapaxes(b.data, -1, -2)
    gb = np.swapaxes(a.data, -1, -2) @ g
    ga = _reduce_broadcast(ga, a.shape)
    gb = _reduce_broadcast(gb, b.shape)
    return (ga, gb)


def _bw_stack(node, g):
    axis = node._meta["axis"]
    outs, idx = [], 0
    for shp in node._meta["shapes"]:
        n = shp[axis]
        sl = [slice(None)] * g.ndim
        sl[axis] = slice(idx, idx + n)
        outs.append(g[tuple(sl)])
        idx += n
    return tuple(outs)


def _bw_clip(node, g):
    lo, hi = node._meta["clip"]
    x = node.parents[0].data
    mask = ((x >= lo) & (x <= hi)).astype(x.dtype)
    return (g * mask,)


_BACKWARD_FUNCS = {
    "add": _bw_add, "sub": _bw_sub, "mul": _bw_mul, "div": _bw_div, "pow": _bw_pow,
    "neg": _bw_neg, "abs": _bw_abs, "exp": _bw_exp, "log": _bw_log, "sqrt": _bw_sqrt,
    "tanh": _bw_tanh, "sigmoid": _bw_sigmoid, "sin": _bw_sin, "cos": _bw_cos,
    "sum": _bw_sum, "mean": _bw_mean, "max": _bw_max, "reshape": _bw_reshape,
    "transpose": _bw_transpose, "permute": _bw_permute, "expand": _bw_expand,
    "matmul": _bw_matmul, "stack": _bw_stack, "clip": _bw_clip,
}


def checkpoint(fn, *inputs):
    """Recompute activations during backward instead of storing them.

    Trades compute for memory, like PyTorch's ``torch.utils.checkpoint``.
    ``fn`` must accept plain :class:`Tensor` args and return a single Tensor.
    """
    inputs = [t if isinstance(t, Tensor) else Tensor(t) for t in inputs]
    if not any(t.requires_grad for t in inputs):
        return fn(*inputs)
    with no_grad():
        out_val = fn(*inputs).data.copy()
    out = Tensor(out_val, dtype=inputs[0].dtype)
    op_name = f"ckpt_{id(out)}"

    def _ckpt_backward(node, g):
        for t in inputs:
            t.zero_grad()
        y = fn(*inputs)
        y.backward(g)
        grads = []
        for t in inputs:
            grads.append(t.grad.copy() if t.grad is not None else np.zeros_like(t.data))
            t.zero_grad()
        return grads

    _BACKWARD_FUNCS[op_name] = _ckpt_backward
    out._register(tuple(inputs), op_name)
    return out
