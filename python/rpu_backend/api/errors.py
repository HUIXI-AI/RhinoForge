"""v5.0 owning module for the rpu_backend public error hierarchy.

Per ADR §2.2 + REQ SKEL-04 + DELETION-LEDGER §N0a: this file surfaces the
8 user-facing error classes. Predecessor: ``core/errors.py`` (deleted in
v5-04 / D3 ledger).

**v5-04 DAG inversion (codex Stage-3 G8 fix):** ``RPUBackendError`` (base) +
``RPUConfigError`` + ``UnsupportedModelError`` + ``PlannerRejectError`` are
owned by the ``runtime`` package (the planner error specifically lives in
``runtime/execution_planner.py``)
because ``runtime/registry.get_adapter`` and ``runtime/hw_attrs.validate_pre/postinstall``
raise them, and ADR §3.1 + §3.2 forbid ``runtime/`` from importing ``api/``.
``api/errors.py`` re-exports those 4 so the user-facing import surface
``from rpu_backend.api.errors import RPUConfigError`` is unchanged.

Per-class ``__module__`` resolves to ``rpu_backend.api.errors`` for the 4
api-owned classes (RPUUnsupportedDtypeError, RPUSingleHandleError,
SPMExhaustionError, WeightShapeMismatchError) so SKEL-04 acceptance is
satisfied for the api-owned subset. The base, config, and unsupported-model
classes have ``__module__ == 'rpu_backend.runtime'``; the planner error has
``__module__ == 'rpu_backend.runtime.execution_planner'``. Both are acceptable
per the DAG fix (public access still works via ``rpu_backend.api.errors``).

Inheritance invariant (NM1 iter-3 carried forward): ``RPUConfigError``
inherits ``RPUBackendError`` so ``except RPUBackendError:`` still catches
config errors.
"""

from rpu_backend.runtime import (  # noqa: F401
    PlannerRejectError,
    RPUBackendError,
    RPUConfigError,
    UnsupportedModelError,
)


class RPUUnsupportedDtypeError(RPUBackendError):
    """dtype other than torch.float16.

    NM1 reviews iter-3: relocated from `transformers/errors.py`. Re-exported
    from `transformers.errors` for user-code back-compat.
    """


class RPUSingleHandleError(RPUBackendError):
    """Second concurrent RPU model or policy instance attempted.

    NM1 reviews iter-3: relocated from `transformers/errors.py`. Re-exported
    from `transformers.errors` for user-code back-compat.
    """


class SPMExhaustionError(RPUBackendError):
    """Measured peak SPM allocation exceeds the 8,303,708 B/core planner budget — typically from chunk/seq mismatch.

    NM1 reviews iter-3: relocated from `transformers/errors.py`. Re-exported
    from `transformers.errors` for user-code back-compat.
    """


class WeightShapeMismatchError(RPUBackendError):
    """Weight tensor shape does not match adapter expected partition constraints.

    NM1 reviews iter-3: relocated from `transformers/errors.py`. Re-exported
    from `transformers.errors` for user-code back-compat.
    """


__all__ = [
    "RPUBackendError",
    "RPUConfigError",
    "UnsupportedModelError",
    "PlannerRejectError",
    "RPUUnsupportedDtypeError",
    "RPUSingleHandleError",
    "SPMExhaustionError",
    "WeightShapeMismatchError",
]
