"""NeuralKit exception hierarchy.

All NeuralKit errors derive from :class:`NeuralKitError` so that callers can
catch library errors with a single ``except`` clause.
"""

from __future__ import annotations


class NeuralKitError(Exception):
    """Base class for every error raised by NeuralKit."""


class TensorError(NeuralKitError):
    """Raised for invalid tensor operations (shape/dtype/grad misuse)."""


class ShapeError(TensorError):
    """Raised when operand shapes are incompatible."""


class GradientError(NeuralKitError):
    """Raised when the autodiff engine is used incorrectly."""


class NotFittedError(NeuralKitError):
    """Raised when an estimator/model is used before fitting."""


class ValidationError(NeuralKitError):
    """Raised when metrics/validators detect an unusable model state."""


class SerializationError(NeuralKitError):
    """Raised when saving/loading artifacts fails or format is unknown."""


class ConfigurationError(NeuralKitError):
    """Raised when user-supplied configuration is inconsistent."""
