"""Validation shared by public RPU execution-configuration entry points."""
from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any


_STAGES = ("prefill", "vision", "action")
_FIELDS = ("chunk_size", "padding_rows", "padding_budget")
_MISSING = object()


class _FrozenExecutionMapping(Mapping[str, Any]):
    """Small immutable mapping that remains deepcopy- and pickle-safe."""

    __slots__ = ("_items",)

    def __init__(self, value: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_items", tuple(value.items()))

    def __getitem__(self, key: str) -> Any:
        for item_key, item_value in self._items:
            if item_key == key:
                return item_value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("rpu_execution is read-only")

    def __delattr__(self, name: str) -> None:
        raise TypeError("rpu_execution is read-only")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __reduce__(self):
        return type(self), (dict(self.items()),)


def _freeze_rpu_execution(
    value: Mapping[str, Mapping[str, str | int]],
) -> Mapping[str, Mapping[str, str | int]]:
    """Return a detached, deeply read-only execution mapping."""
    return _FrozenExecutionMapping({
        stage: _FrozenExecutionMapping(dict(fields))
        for stage, fields in value.items()
    })


def normalize_rpu_execution(
    value: Mapping[str, Mapping[str, Any]] | None,
    *,
    entry_point: str,
    vlm_chunk_size: int | None = None,
    supported: Mapping[str, Collection[str]] | None = None,
) -> Mapping[str, Mapping[str, str | int]]:
    """Return a detached, deeply read-only config or fail before loading."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{entry_point}: rpu_execution must be a mapping, got "
            f"{type(value).__name__}"
        )

    unknown_stages = set(value) - set(_STAGES)
    if unknown_stages:
        raise ValueError(
            f"{entry_point}: rpu_execution has unknown stage(s) "
            f"{sorted(map(str, unknown_stages))}; supported stages are {list(_STAGES)}"
        )
    if supported is not None:
        unsupported_stages = set(value) - set(supported)
        if unsupported_stages:
            raise ValueError(
                f"{entry_point}: rpu_execution stage(s) "
                f"{sorted(map(str, unsupported_stages))} are not supported by "
                "this model entry point"
            )

    normalized: dict[str, dict[str, str | int]] = {}
    for stage in _STAGES:
        if stage not in value:
            continue
        stage_value = value[stage]
        if not isinstance(stage_value, Mapping):
            raise TypeError(
                f"{entry_point}: rpu_execution[{stage!r}] must be a mapping, "
                f"got {type(stage_value).__name__}"
            )
        unknown_fields = set(stage_value) - set(_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"{entry_point}: rpu_execution[{stage!r}] has unknown field(s) "
                f"{sorted(map(str, unknown_fields))}; supported fields are {list(_FIELDS)}"
            )
        if supported is not None:
            unsupported_fields = set(stage_value) - set(supported[stage])
            if unsupported_fields:
                raise ValueError(
                    f"{entry_point}: rpu_execution[{stage!r}] field(s) "
                    f"{sorted(map(str, unsupported_fields))} are not supported "
                    "by this model entry point"
                )

        stage_config: dict[str, str | int] = {}
        for field in _FIELDS:
            if field not in stage_value:
                continue
            field_value = stage_value[field]
            if field == "chunk_size":
                if isinstance(field_value, str) and field_value == "auto":
                    stage_config[field] = "auto"
                elif (
                    isinstance(field_value, bool)
                    or not isinstance(field_value, int)
                    or field_value <= 0
                    or field_value % 16 != 0
                ):
                    raise ValueError(
                        f"{entry_point}: rpu_execution[{stage!r}]['chunk_size'] "
                        "must be 'auto' or a positive multiple of 16, got "
                        f"{field_value!r}"
                    )
                else:
                    stage_config[field] = field_value
            elif field == "padding_rows":
                if isinstance(field_value, str) and field_value == "auto":
                    stage_config[field] = "auto"
                elif (
                    isinstance(field_value, bool)
                    or not isinstance(field_value, int)
                    or field_value < 0
                ):
                    raise ValueError(
                        f"{entry_point}: rpu_execution[{stage!r}]['padding_rows'] "
                        "must be 'auto' or a non-negative integer, got "
                        f"{field_value!r}"
                    )
                else:
                    stage_config[field] = field_value
            elif (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or field_value < 0
            ):
                raise ValueError(
                    f"{entry_point}: rpu_execution[{stage!r}]['padding_budget'] "
                    f"must be a non-negative integer, got {field_value!r}"
                )
            else:
                stage_config[field] = field_value
        if (
            isinstance(stage_config.get("padding_rows"), int)
            and "padding_budget" in stage_config
        ):
            raise ValueError(
                f"{entry_point}: rpu_execution[{stage!r}]['padding_rows'] "
                "is exact and conflicts with 'padding_budget'"
            )
        if stage_config:
            normalized[stage] = stage_config

    if vlm_chunk_size is not None:
        if (
            isinstance(vlm_chunk_size, bool)
            or not isinstance(vlm_chunk_size, int)
            or vlm_chunk_size < 0
            or (vlm_chunk_size and vlm_chunk_size % 16 != 0)
        ):
            raise ValueError(
                f"{entry_point}: vlm_chunk_size must be 0 or a positive "
                f"multiple of 16, got {vlm_chunk_size!r}"
            )
        if "chunk_size" in normalized.get("prefill", {}):
            raise ValueError(
                f"{entry_point}: vlm_chunk_size conflicts with "
                "rpu_execution['prefill']['chunk_size']; pass only one"
            )
        normalized.setdefault("prefill", {})["chunk_size"] = (
            "auto" if vlm_chunk_size == 0 else vlm_chunk_size
        )

    return _freeze_rpu_execution({
        stage: {
            field: normalized[stage][field]
            for field in _FIELDS
            if field in normalized[stage]
        }
        for stage in _STAGES
        if stage in normalized
    })


def bind_rpu_execution(
    owner: Any,
    value: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    entry_point: str,
    supported: Mapping[str, Collection[str]] | None = None,
) -> Mapping[str, Mapping[str, str | int]]:
    """Validate and attach one cold config, preserving an equal existing one."""
    existing = getattr(owner, "_rpu_execution", _MISSING)
    normalized = normalize_rpu_execution(
        existing if value is None and existing is not _MISSING else value,
        entry_point=entry_point,
        supported=supported,
    )
    if existing is not _MISSING:
        existing_normalized = normalize_rpu_execution(
            existing,
            entry_point=entry_point,
            supported=supported,
        )
        if value is not None and existing_normalized != normalized:
            raise ValueError(
                f"{entry_point}: rpu_execution conflicts with the existing "
                "model configuration"
            )
        if (
            isinstance(existing, _FrozenExecutionMapping)
            and all(
                isinstance(fields, _FrozenExecutionMapping)
                for fields in existing.values()
            )
            and existing == existing_normalized
        ):
            return existing
        normalized = existing_normalized
    owner._rpu_execution = normalized
    return normalized


__all__ = ["bind_rpu_execution", "normalize_rpu_execution"]
