"""Public lifecycle contract for externally assembled RhinoVLA runtimes.

The backend owns the RPU components, but the separate RhinoVLA model repository
owns checkpoint composition, preprocessing, and the live-input contract. This
wrapper gives delivery executors one stable API without pretending the backend
can infer those model-repository choices.
"""
from __future__ import annotations

import copy
import importlib
from collections.abc import Callable, Mapping
from os import PathLike, fspath
from typing import Any


RuntimeFactory = str | Callable[..., Any]
_MISSING = object()


def _normalize_execution(value, *, entry_point):
    from rpu_backend.adapters.rhinovla.runtime import (
        RHINOVLA_EXECUTION_COMPONENTS,
    )
    from rpu_backend.api._execution import normalize_rpu_execution

    return normalize_rpu_execution(
        value,
        entry_point=entry_point,
        supported_components=RHINOVLA_EXECUTION_COMPONENTS,
    )


def _resolve_runtime_factory(target: RuntimeFactory) -> Callable[..., Any]:
    if callable(target):
        return target
    if not isinstance(target, str):
        raise TypeError(
            "RhinoVLA runtime_factory must be callable or 'module:callable', "
            f"got {type(target).__name__}"
        )
    module_name, separator, attr = target.partition(":")
    if not separator or not module_name or not attr:
        raise ValueError(
            "RhinoVLA runtime_factory must use 'module:callable' syntax, "
            f"got {target!r}"
        )
    factory = getattr(importlib.import_module(module_name), attr)
    if not callable(factory):
        raise TypeError(f"RhinoVLA runtime factory is not callable: {target}")
    return factory


def _bound_execution_request(runtime, value):
    """Fill only cold defaults captured by this runtime, never re-read env."""
    normalized = _normalize_execution(value, entry_point="RhinoVLAPolicy.from_runtime")
    snapshot = getattr(runtime, "_pipeline_cold_config", None)
    return normalized if snapshot is None else snapshot.bind_defaults(normalized)


class RhinoVLAPolicy:
    """Stable policy facade around a model-repository-owned RhinoVLA runtime."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "RhinoVLAPolicy() is not a public constructor. Use from_runtime(), "
            "from_factory(), or from_pretrained()."
        )

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        rpu_execution=None,
    ) -> "RhinoVLAPolicy":
        """Wrap an already assembled, RPU-ready runtime with ``predict``."""
        if isinstance(runtime, cls):
            if rpu_execution is None:
                return runtime
            requested = _bound_execution_request(runtime._runtime, rpu_execution)
            existing = _normalize_execution(
                getattr(runtime, "_rpu_execution", None),
                entry_point="RhinoVLAPolicy.from_runtime",
            )
            if requested != existing:
                raise ValueError(
                    "RhinoVLAPolicy.from_runtime: rpu_execution conflicts with "
                    "the existing policy configuration"
                )
            return runtime
        if not callable(getattr(runtime, "predict", None)):
            raise TypeError(
                "RhinoVLA runtime must provide callable predict(**request)"
            )
        policy = cls.__new__(cls)
        policy._runtime = runtime
        controller = getattr(runtime, "_rhinovla_execution_controller", None)
        existing_raw = getattr(runtime, "_rpu_execution", _MISSING)
        requested = _bound_execution_request(
            runtime,
            rpu_execution if rpu_execution is not None else (
                existing_raw if existing_raw is not _MISSING else None
            ),
        )
        if controller is None:
            raise ValueError(
                "RhinoVLAPolicy.from_runtime: the runtime must call "
                "bind_rhinovla_execution_runtime() with its explicit text, "
                "vision, and action children"
            )
        if existing_raw is not _MISSING:
            existing = _normalize_execution(
                existing_raw,
                entry_point="RhinoVLAPolicy.from_runtime",
            )
            if rpu_execution is not None and requested != existing:
                raise ValueError(
                    "RhinoVLAPolicy.from_runtime: rpu_execution conflicts with "
                    "the runtime's effective configuration"
                )
            requested = existing
        policy._rpu_execution = requested
        policy._rhinovla_execution_controller = controller
        if controller is not None:
            if controller.runtime is not runtime:
                raise ValueError(
                    "RhinoVLAPolicy.from_runtime: execution controller owns a "
                    "different runtime"
                )
            policy._execution_session = controller.session
            controller.session.register_config_view(policy)
        policy._last_rpu_execution_plan = {}
        return policy

    @classmethod
    def _from_factory_runtime(cls, runtime, *, rpu_execution):
        try:
            return cls.from_runtime(runtime, rpu_execution=rpu_execution)
        except BaseException as error:
            from rpu_backend.adapters.rhinovla.runtime import (
                _RhinoVLAExecutionController,
                close_rhinovla_runtime,
            )

            controller = getattr(runtime, "_rhinovla_execution_controller", None)
            owns_children = bool(vars(runtime).get("_rhinovla_retirement_children"))
            if ((isinstance(controller, _RhinoVLAExecutionController)
                 and getattr(controller, "runtime", None) is runtime)
                    or owns_children):
                try:
                    if isinstance(controller, _RhinoVLAExecutionController):
                        controller.close()
                    else:
                        close_rhinovla_runtime(runtime)
                except BaseException as cleanup_error:
                    error.add_note(f"RhinoVLA factory cleanup failed: {cleanup_error!r}")
            raise

    @classmethod
    def from_factory(
        cls,
        runtime_factory: RuntimeFactory,
        config: Mapping[str, Any],
        *,
        rpu_execution=None,
    ) -> "RhinoVLAPolicy":
        """Build through a factory that binds all three physical children.

        The factory always receives canonical ``rpu_execution`` (including
        the empty mapping for AUTO) and must call
        ``bind_rhinovla_execution_runtime`` before returning.
        """
        if not isinstance(config, Mapping):
            raise TypeError(
                f"RhinoVLA factory config must be a mapping, got {type(config).__name__}"
            )
        execution_config = _normalize_execution(
            rpu_execution,
            entry_point="RhinoVLAPolicy.from_factory",
        )
        factory = _resolve_runtime_factory(runtime_factory)
        runtime = factory(config, rpu_execution=execution_config)
        return cls._from_factory_runtime(
            runtime,
            rpu_execution=execution_config if rpu_execution is not None else None,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | PathLike[str],
        *,
        runtime_factory: RuntimeFactory,
        config: Mapping[str, Any] | None = None,
        rpu_execution=None,
        **factory_kwargs: Any,
    ) -> "RhinoVLAPolicy":
        """Delegate checkpoint construction to an explicit model runtime factory.

        The factory receives ``pretrained_name_or_path`` as its first argument.
        When supplied, ``config`` is forwarded as a keyword argument. Loading,
        irreversible RPU conversion, and input-envelope validation remain the
        factory's responsibility. It always receives canonical
        ``rpu_execution`` and must bind the explicit text, vision, and action
        children before returning.
        """
        if isinstance(pretrained_name_or_path, str):
            path_value = pretrained_name_or_path
        elif isinstance(pretrained_name_or_path, PathLike):
            path_value = fspath(pretrained_name_or_path)
            if not isinstance(path_value, str):
                raise TypeError(
                    "RhinoVLA pretrained_name_or_path PathLike must resolve "
                    "to str"
                )
        else:
            raise TypeError(
                "RhinoVLA pretrained_name_or_path must be str or PathLike, "
                f"got {type(pretrained_name_or_path).__name__}"
            )
        if not path_value:
            raise ValueError("RhinoVLA pretrained_name_or_path must not be empty")
        execution_config = _normalize_execution(
            rpu_execution,
            entry_point="RhinoVLAPolicy.from_pretrained",
        )
        factory = _resolve_runtime_factory(runtime_factory)
        if config is not None:
            if not isinstance(config, Mapping):
                raise TypeError(
                    "RhinoVLA config must be a mapping, "
                    f"got {type(config).__name__}"
                )
            if "config" in factory_kwargs:
                raise TypeError("RhinoVLA factory received duplicate config")
            factory_kwargs["config"] = config
        factory_kwargs["rpu_execution"] = execution_config
        runtime = factory(pretrained_name_or_path, **factory_kwargs)
        return cls._from_factory_runtime(
            runtime,
            rpu_execution=execution_config if rpu_execution is not None else None,
        )

    @property
    def runtime(self) -> Any:
        """Return the wrapped runtime for model-specific diagnostics."""
        return self._runtime

    def close(self) -> None:
        """Delegate native retirement to the actual repository-owned parent.

        An external runtime's optional hook retains its historical contract;
        a facade-only Session close is not evidence of physical device release.
        """
        from rpu_backend.adapters.rhinovla.runtime import (
            _RhinoVLAExecutionController, _retirement_failure,
        )

        controller = self._rhinovla_execution_controller
        if isinstance(controller, _RhinoVLAExecutionController):
            if (getattr(controller, "runtime", None) is not self._runtime
                    or getattr(controller, "session", None) is not self._execution_session):
                error = RuntimeError("RhinoVLA facade retirement authority changed")
                _retirement_failure(self._runtime, error)
                raise error
            controller.close()
            return
        retire = getattr(self._runtime, "close", None)
        session = self._execution_session
        with session._lock:
            runtime_state = vars(self._runtime)
            failed = runtime_state.get("_rhinovla_facade_retirement_failed")
            if failed is not None:
                raise RuntimeError(
                    "RhinoVLA external retirement already failed; restart the process"
                ) from failed
            if runtime_state.get("_rhinovla_facade_retired"):
                session.shutdown()
                return
            if session._active:
                raise RuntimeError("cannot close RhinoVLA during a forward")
            from rpu_backend.api._execution import _require_execution_process_safe
            _require_execution_process_safe()
            try:
                if callable(retire):
                    retire()
                runtime_state["_rhinovla_facade_retired"] = True
                # The external hook may own this same Session. A second,
                # idempotent shutdown is safe; nesting the hook inside shutdown
                # would skip its physical retirement.
                session.shutdown()
            except BaseException as error:
                runtime_state.setdefault(
                    "_rhinovla_facade_retirement_failed", error
                )
                _retirement_failure(self._runtime, error)
                raise

    @property
    def last_rpu_execution_plan(self) -> dict[str, dict[str, Any]]:
        """Return a detached copy of the last validated per-stage plan."""
        return copy.deepcopy(self._last_rpu_execution_plan)

    def _collect_execution_receipts(self) -> None:
        controller = self._rhinovla_execution_controller
        if controller is not None:
            self._last_rpu_execution_plan = controller.collect_receipts()

    def prepare_graphs(self, **request: Any) -> Any:
        """Prepare the runtime's finite graph profile for a representative request."""
        prepare = getattr(self._runtime, "prepare_graphs", None)
        if not callable(prepare):
            raise TypeError(
                "RhinoVLA runtime does not provide prepare_graphs(**request)"
            )
        from rpu_backend.api._execution import execution_guard

        with execution_guard(self._runtime):
            if self._rhinovla_execution_controller is not None:
                self._rhinovla_execution_controller.clear_receipts()
            result = prepare(**request)
            self._collect_execution_receipts()
        return result

    def predict(self, **request: Any) -> Any:
        """Run one complete RhinoVLA request and return its action chunk."""
        from rpu_backend.api._execution import execution_guard

        with execution_guard(self._runtime):
            if self._rhinovla_execution_controller is not None:
                self._rhinovla_execution_controller.clear_receipts()
            result = self._runtime.predict(**request)
            self._collect_execution_receipts()
        return result

    def reconfigure(self, rpu_execution) -> Mapping[str, Any]:
        """Atomically update one or more RhinoVLA physical children."""
        if self._rhinovla_execution_controller is None:
            raise RuntimeError(
                "RhinoVLAPolicy.reconfigure requires a bound RhinoVLA runtime"
            )
        from rpu_backend.api._execution import reconfigure_rpu_execution

        return reconfigure_rpu_execution(self._runtime, rpu_execution)

    def predict_action_chunk(self, **request: Any) -> Any:
        """VLA-policy spelling of :meth:`predict`."""
        return self.predict(**request)


__all__ = ["RhinoVLAPolicy"]
