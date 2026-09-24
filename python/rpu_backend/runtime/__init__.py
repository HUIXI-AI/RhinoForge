"""rpu_backend.runtime — internal framework layer (debug, weights, registry, decoder, hw_attrs, control).

This subpackage MUST NOT import from rpu_backend.adapters or rpu_backend.api
(CI: scripts/check_import_graph_v5.py).

Error classes owned here:
``RPUBackendError`` (base), ``RPUConfigError``, ``UnsupportedModelError``, and
``PlannerRejectError`` are the canonical runtime-owned error classes raised by
the hardware validator, adapter registry, and execution planner.
``rpu_backend.api.errors`` re-exports them so the user-facing surface remains
``from rpu_backend.api.errors import RPUConfigError`` (ADR §10 #1 unchanged).
The other 4 error classes (``RPUUnsupportedDtypeError``, ``RPUSingleHandleError``,
``SPMExhaustionError``, ``WeightShapeMismatchError``) stay owned in ``api/errors.py``
because ``runtime/`` does not raise them.
"""


from rpu_backend.runtime._errors import RPUBackendError


class RPUConfigError(RPUBackendError):
    """Invalid `_rpu_*` per-instance hardware-attribute configuration (A9).

    Raised by the A9 validator in ``rpu_backend.runtime.hw_attrs`` when:
      - A pre-existing ``_rpu_*`` not in ``PUBLIC_HW_ATTRS`` /
        ``INTERNAL_HW_ATTRS_TRANSITIONAL`` is detected during pre-stamp scan
        (``validate_preinstall``).
      - A write to an ``INTERNAL_HW_ATTRS_TRANSITIONAL`` monotonic attr
        AFTER the validator has stamped the module tree.
      - A write to a ``COLD_ONLY_ATTRS`` member AFTER first forward.
      - ``__class__`` reassignment fails on a ``__slots__``-bearing or C-backed
        module.

    Inherits ``RPUBackendError`` so ``except RPUBackendError:`` still catches
    config errors.
    """


class UnsupportedModelError(RPUBackendError):
    """HF architecture not in adapter registry, or size above Phase 2 envelope.

    Raised by ``rpu_backend.runtime.registry.get_adapter``. Re-exported from
    ``rpu_backend.api.errors``.
    """


from rpu_backend.runtime import (  # noqa: F401, E402
    control,
    debug,
    decoder,
    execution_planner,
    hw_attrs,
    registry,
    weights,
)

# Re-exported here (not only from `control`) because it is imported by every
# adapter: `from rpu_backend.runtime import rpu_env_bool` resolves against the
# already-imported subpackage, so it also works in the host-only unit tests that
# load one adapter source file with `rpu_backend.runtime` stubbed in sys.modules.
from rpu_backend.runtime.control import rpu_env_bool  # noqa: F401, E402
from rpu_backend.runtime.execution_planner import PlannerRejectError  # noqa: F401, E402

__all__ = [
    "control",
    "debug",
    "decoder",
    "execution_planner",
    "hw_attrs",
    "registry",
    "weights",
    "rpu_env_bool",
    "RPUBackendError",
    "RPUConfigError",
    "UnsupportedModelError",
    "PlannerRejectError",
]
