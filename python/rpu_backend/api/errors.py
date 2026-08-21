"""Public error hierarchy for ``rpu_backend``.

Runtime-owned errors are re-exported here so runtime modules do not need to
import the higher-level API package. ``RPUConfigError`` inherits
``RPUBackendError``, so callers may catch either the specific error or the
backend-wide base class.
"""

from rpu_backend.runtime import (  # noqa: F401
    RPUBackendError,
    RPUConfigError,
    UnsupportedModelError,
)


class RPUUnsupportedDtypeError(RPUBackendError):
    """The requested dtype is unsupported by the selected RPU path."""


class RPUSingleHandleError(RPUBackendError):
    """A second concurrent RPU model or policy instance was requested."""


class SPMExhaustionError(RPUBackendError):
    """The requested execution plan exceeds the available SPM budget."""


class WeightShapeMismatchError(RPUBackendError):
    """A weight tensor shape violates the adapter's layout constraints."""


__all__ = [
    "RPUBackendError",
    "RPUConfigError",
    "UnsupportedModelError",
    "RPUUnsupportedDtypeError",
    "RPUSingleHandleError",
    "SPMExhaustionError",
    "WeightShapeMismatchError",
]
