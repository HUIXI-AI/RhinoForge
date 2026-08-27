"""Public lifecycle contract for externally assembled RhinoVLA runtimes.

The backend owns the RPU components, but the separate RhinoVLA model repository
owns checkpoint composition, preprocessing, and the live-input contract. This
wrapper gives runtime integrators one stable API without pretending the backend
can infer those model-repository choices.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable, Collection, Mapping
from os import PathLike, fspath
from typing import Any


RuntimeFactory = str | Callable[..., Any]
_MISSING = object()
_EXECUTION_STAGES = ("prefill", "vision", "action")
_EXECUTION_FIELDS = ("chunk_size", "padding_rows", "padding_budget")
_RESOLVED_FIELDS = ("logical_len", "execution_len", "chunk_size", "padding_rows")
_RESOLVED_OPTIONAL_FIELDS = ("execution_alignment",)


def _execution_capabilities(owner: Any, *, entry_point: str):
    """Return the external runtime's declared stage/field support."""
    raw = getattr(owner, "_rpu_execution_capabilities", _MISSING)
    if raw is _MISSING:
        raise ValueError(
            f"{entry_point}: non-empty rpu_execution requires "
            "_rpu_execution_capabilities on the runtime factory and returned "
            "runtime"
        )
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"{entry_point}: _rpu_execution_capabilities must be a mapping"
        )
    unknown_stages = set(raw) - set(_EXECUTION_STAGES)
    if unknown_stages:
        raise ValueError(
            f"{entry_point}: capability has unknown stage(s) "
            f"{sorted(map(str, unknown_stages))}"
        )
    supported: dict[str, tuple[str, ...]] = {}
    for stage, fields in raw.items():
        if (
            isinstance(fields, (str, bytes))
            or not isinstance(fields, Collection)
        ):
            raise TypeError(
                f"{entry_point}: capability for {stage!r} must be a "
                "collection of field names"
            )
        unknown_fields = set(fields) - set(_EXECUTION_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"{entry_point}: capability for {stage!r} has unknown "
                f"field(s) {sorted(map(str, unknown_fields))}"
            )
        supported[stage] = tuple(
            field for field in _EXECUTION_FIELDS if field in fields
        )
    return supported


def _resolved_execution_generation(runtime: Any, *, entry_point: str) -> int:
    raw = getattr(runtime, "_rpu_execution_resolved_generation", _MISSING)
    if raw is _MISSING:
        return -1
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise TypeError(
            f"{entry_point}: _rpu_execution_resolved_generation must be a "
            "non-negative integer"
        )
    return raw


def _validate_resolved_execution(
    runtime: Any,
    requested,
    *,
    previous_generation: int,
    entry_point: str,
):
    """Validate post-dispatch telemetry against the cold request."""
    generation = _resolved_execution_generation(
        runtime, entry_point=entry_point
    )
    if generation <= previous_generation:
        raise RuntimeError(
            f"{entry_point}: runtime did not publish a fresh resolved execution "
            "plan; _rpu_execution_resolved_generation must strictly increase "
            "during this dispatch"
        )
    raw = getattr(runtime, "_rpu_execution_resolved", _MISSING)
    if raw is _MISSING:
        raise RuntimeError(
            f"{entry_point}: runtime did not publish "
            "_rpu_execution_resolved after execution"
        )
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"{entry_point}: _rpu_execution_resolved must be a mapping"
        )
    resolved: dict[str, dict[str, int]] = {}
    for stage, config in requested.items():
        plan = raw.get(stage)
        if not isinstance(plan, Mapping):
            raise RuntimeError(
                f"{entry_point}: resolved plan is missing stage {stage!r}"
            )
        missing = set(_RESOLVED_FIELDS) - set(plan)
        unknown = set(plan) - set(_RESOLVED_FIELDS) - set(
            _RESOLVED_OPTIONAL_FIELDS
        )
        if missing or unknown:
            raise RuntimeError(
                f"{entry_point}: resolved {stage!r} plan has "
                f"missing={sorted(missing)} unknown={sorted(map(str, unknown))}"
            )
        values = {field: plan[field] for field in _RESOLVED_FIELDS}
        for field in _RESOLVED_OPTIONAL_FIELDS:
            if field in plan:
                values[field] = plan[field]
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in values.values()):
            raise TypeError(
                f"{entry_point}: resolved {stage!r} plan values must be integers"
            )
        logical = values["logical_len"]
        execution = values["execution_len"]
        chunk = values["chunk_size"]
        padding = values["padding_rows"]
        execution_alignment = values.get("execution_alignment", 1)
        if (
            logical < 0
            or execution < logical
            or padding < 0
            or execution != logical + padding
            or execution_alignment <= 0
            or execution % execution_alignment != 0
            or chunk <= 0
            or chunk % 16 != 0
        ):
            raise RuntimeError(
                f"{entry_point}: invalid resolved {stage!r} plan {values}"
            )
        requested_chunk = config.get("chunk_size")
        requested_padding = config.get("padding_rows")
        padding_budget = config.get("padding_budget")
        mandatory_padding = (-logical) % execution_alignment
        optional_padding = padding - mandatory_padding
        if isinstance(requested_chunk, int) and chunk != requested_chunk:
            raise RuntimeError(
                f"{entry_point}: resolved {stage!r} chunk_size={chunk} does "
                f"not match exact request {requested_chunk}"
            )
        if isinstance(requested_padding, int) and padding != requested_padding:
            raise RuntimeError(
                f"{entry_point}: resolved {stage!r} padding_rows={padding} "
                f"does not match exact request {requested_padding}"
            )
        if (
            isinstance(padding_budget, int)
            and optional_padding > padding_budget
        ):
            raise RuntimeError(
                f"{entry_point}: resolved {stage!r} optional padding="
                f"{optional_padding} (total={padding}, mandatory="
                f"{mandatory_padding}, alignment={execution_alignment}) "
                f"exceeds budget {padding_budget}"
            )
        resolved[stage] = values
    return resolved


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
            from rpu_backend.api._execution import normalize_rpu_execution
            requested = normalize_rpu_execution(
                rpu_execution,
                entry_point="RhinoVLAPolicy.from_runtime",
            )
            if requested:
                supported = _execution_capabilities(
                    runtime._runtime,
                    entry_point="RhinoVLAPolicy.from_runtime",
                )
                requested = normalize_rpu_execution(
                    requested,
                    entry_point="RhinoVLAPolicy.from_runtime",
                    supported=supported,
                )
            existing = normalize_rpu_execution(
                getattr(runtime, "_rpu_execution", None),
                entry_point="RhinoVLAPolicy.from_runtime",
            )
            if requested != existing:
                raise ValueError(
                    "RhinoVLAPolicy.from_runtime: rpu_execution conflicts with "
                    "the existing policy configuration"
                )
            runtime._rpu_execution_handshake_required = bool(requested)
            return runtime
        if not callable(getattr(runtime, "predict", None)):
            raise TypeError(
                "RhinoVLA runtime must provide callable predict(**request)"
            )
        policy = cls.__new__(cls)
        policy._runtime = runtime
        from rpu_backend.api._execution import normalize_rpu_execution
        existing_raw = getattr(runtime, "_rpu_execution", _MISSING)
        requested = normalize_rpu_execution(
            rpu_execution if rpu_execution is not None else (
                existing_raw if existing_raw is not _MISSING else None
            ),
            entry_point="RhinoVLAPolicy.from_runtime",
        )
        if requested:
            supported = _execution_capabilities(
                runtime,
                entry_point="RhinoVLAPolicy.from_runtime",
            )
            requested = normalize_rpu_execution(
                requested,
                entry_point="RhinoVLAPolicy.from_runtime",
                supported=supported,
            )
        if rpu_execution is not None and existing_raw is _MISSING and requested:
            raise ValueError(
                "RhinoVLAPolicy.from_runtime: non-empty rpu_execution was not "
                "consumed by the runtime; the runtime must expose its effective "
                "configuration as _rpu_execution"
            )
        if existing_raw is not _MISSING:
            existing = normalize_rpu_execution(
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
        policy._rpu_execution_handshake_required = (
            rpu_execution is not None and bool(requested)
        )
        policy._last_rpu_execution_plan = {}
        return policy

    @classmethod
    def from_factory(
        cls,
        runtime_factory: RuntimeFactory,
        config: Mapping[str, Any],
        *,
        rpu_execution=None,
    ) -> "RhinoVLAPolicy":
        """Build from the runtime contract ``factory(config) -> runtime``."""
        if not isinstance(config, Mapping):
            raise TypeError(
                f"RhinoVLA factory config must be a mapping, got {type(config).__name__}"
            )
        from rpu_backend.api._execution import normalize_rpu_execution
        execution_config = normalize_rpu_execution(
            rpu_execution,
            entry_point="RhinoVLAPolicy.from_factory",
        )
        factory = _resolve_runtime_factory(runtime_factory)
        if execution_config:
            execution_config = normalize_rpu_execution(
                execution_config,
                entry_point="RhinoVLAPolicy.from_factory",
                supported=_execution_capabilities(
                    factory,
                    entry_point="RhinoVLAPolicy.from_factory",
                ),
            )
        runtime = (
            factory(config, rpu_execution=execution_config)
            if rpu_execution is not None
            else factory(config)
        )
        return cls.from_runtime(
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
        factory's responsibility.
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
        from rpu_backend.api._execution import normalize_rpu_execution
        execution_config = normalize_rpu_execution(
            rpu_execution,
            entry_point="RhinoVLAPolicy.from_pretrained",
        )
        factory = _resolve_runtime_factory(runtime_factory)
        if execution_config:
            execution_config = normalize_rpu_execution(
                execution_config,
                entry_point="RhinoVLAPolicy.from_pretrained",
                supported=_execution_capabilities(
                    factory,
                    entry_point="RhinoVLAPolicy.from_pretrained",
                ),
            )
        if config is not None:
            if not isinstance(config, Mapping):
                raise TypeError(
                    "RhinoVLA config must be a mapping, "
                    f"got {type(config).__name__}"
                )
            if "config" in factory_kwargs:
                raise TypeError("RhinoVLA factory received duplicate config")
            factory_kwargs["config"] = config
        if rpu_execution is not None:
            factory_kwargs["rpu_execution"] = execution_config
        runtime = factory(pretrained_name_or_path, **factory_kwargs)
        return cls.from_runtime(
            runtime,
            rpu_execution=execution_config if rpu_execution is not None else None,
        )

    @property
    def runtime(self) -> Any:
        """Return the wrapped runtime for model-specific diagnostics."""
        return self._runtime

    @property
    def last_rpu_execution_plan(self) -> dict[str, dict[str, int]]:
        """Return a detached copy of the last validated per-stage plan."""
        return {
            stage: dict(fields)
            for stage, fields in self._last_rpu_execution_plan.items()
        }

    def prepare_graphs(self, **request: Any) -> Any:
        """Prepare the runtime's finite graph profile for a representative request."""
        prepare = getattr(self._runtime, "prepare_graphs", None)
        if not callable(prepare):
            raise TypeError(
                "RhinoVLA runtime does not provide prepare_graphs(**request)"
            )
        if self._rpu_execution_handshake_required:
            previous_generation = _resolved_execution_generation(
                self._runtime,
                entry_point="RhinoVLAPolicy.prepare_graphs",
            )
        result = prepare(**request)
        if self._rpu_execution_handshake_required:
            self._last_rpu_execution_plan = _validate_resolved_execution(
                self._runtime,
                self._rpu_execution,
                previous_generation=previous_generation,
                entry_point="RhinoVLAPolicy.prepare_graphs",
            )
        return result

    def predict(self, **request: Any) -> Any:
        """Run one complete RhinoVLA request and return its action chunk."""
        if self._rpu_execution_handshake_required:
            previous_generation = _resolved_execution_generation(
                self._runtime,
                entry_point="RhinoVLAPolicy.predict",
            )
        result = self._runtime.predict(**request)
        if self._rpu_execution_handshake_required:
            self._last_rpu_execution_plan = _validate_resolved_execution(
                self._runtime,
                self._rpu_execution,
                previous_generation=previous_generation,
                entry_point="RhinoVLAPolicy.predict",
            )
        return result

    def predict_action_chunk(self, **request: Any) -> Any:
        """VLA-policy spelling of :meth:`predict`."""
        return self.predict(**request)


__all__ = ["RhinoVLAPolicy"]
