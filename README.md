# neuralkit

**Production-oriented deep learning framework written from scratch in pure NumPy**

Autodiff tensor engine · Neural networks · Optimizers · Metrics — zero heavy dependencies.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Early%20Development-orange)](#)

> Learn how deep learning frameworks actually work by reading (and extending) a clean, educational implementation.

---

## Why neuralkit?

Most people treat PyTorch / TensorFlow as black boxes.  
**neuralkit** is deliberately small and readable so you can:

- Understand automatic differentiation
- See how tensors, gradients, and optimizers are implemented
- Experiment without pulling in massive dependencies
- Use it as a teaching / research base

---

## Current Status

| Component | Status |
|-----------|--------|
| Tensor + Autodiff engine | ✅ Implemented |
| Basic neural net layers | 🔄 In progress |
| Optimizers (SGD, Adam…) | 🔄 Planned |
| Metrics & training loop | 🔄 Planned |
| Examples / notebooks | 🔄 Planned |

Core autodiff lives in `neuralkit/tensor.py`.

---

## Quick Start

```bash
git clone https://github.com/sayan9168/neuralkit.git
cd neuralkit
pip install numpy
```

```python
from neuralkit.tensor import Tensor

# Example usage once the public API is stabilized
# x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
# y = x * 2 + 1
# y.backward()
```

---

## Project Structure

```text
neuralkit/
├── neuralkit/
│   ├── tensor.py      # Autodiff Tensor engine
│   └── exceptions.py
└── README.md
```

---

## Roadmap

- [x] Core Tensor + reverse-mode autodiff
- [ ] Linear, ReLU, Softmax layers
- [ ] SGD / Adam optimizers
- [ ] Cross-entropy & MSE losses
- [ ] Simple training loop + examples
- [ ] More comprehensive tests

---

## Contributing

This is an educational + research project. Issues and PRs are welcome, especially:

- Cleaner autodiff graph visualization
- Additional operators
- Unit tests
- Example notebooks

---

## License

MIT License © [Sayan Mahata](https://github.com/sayan9168)

---

<div align="center">

Built by [Sayan the researcher](https://github.com/sayan9168)

</div>
