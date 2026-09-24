"""Validation shared by public RPU execution-configuration entry points."""
from __future__ import annotations

import functools
import os
import re
import threading
import weakref
from collections.abc import Collection, Mapping
from contextlib import contextmanager
from itertools import count
from numbers import Integral
from types import MappingProxyType
from typing import Any


_STAGES = ("prefill", "vision", "action")
# Physical cold controls are legal only when a component/stage declares them.
# They are booleans, not AUTO/search selectors; changing a bound value requires
# that owner's cold reconstruction contract.
_COLD_BOOL_FIELDS = (
    "rpu_patch_embed", "rpu_mergers", "rpu_mergers_streaming", "fast_replay",
    "preload_replay_skip", "bake_merger", "fused_merger", "prefix_prep_cache",
    "prefix_prep_rpu", "prefix_no_clone", "partial_mrope", "merger_scatter",
    "shared_cache", "prefix_alias", "denoise_unroll", "linear_acc32", "gelu_erf_ultra",
)
_FIELDS = ("chunk_size", "padding_rows", "padding_budget", *_COLD_BOOL_FIELDS)
_COMPONENTS = "components"
_MISSING = object()
_SESSION_ATTR = "_execution_session"
_NATIVE_RECONFIGURE_ATTEMPTS = count(1)
_UNSAFE_PROCESS_REASON: str | None = None

# Pi's public three-child normalization is also used by the board-free v5
# source pin. Keep one resolver; importing the adapter would initialize torch.
PI05_VISION_COMPONENT = "vision_encoder"
PI05_TEXT_COMPONENT = "language_model"
PI05_ACTION_COMPONENT = "action_expert"
PI05_EXECUTION_COMPONENTS = {
    PI05_VISION_COMPONENT: {"vision": ("chunk_size", "linear_acc32")},
    PI05_TEXT_COMPONENT: {"prefill": ("chunk_size", "padding_rows", "padding_budget", "linear_acc32")},
    PI05_ACTION_COMPONENT: {"action": ("chunk_size", "linear_acc32")},
}

PI05_EXECUTION_SUPPORTED = {
    "model": ("num_cores",),
    "vision": ("chunk_size", "linear_acc32"),
    "prefill": ("chunk_size", "padding_rows", "padding_budget", "linear_acc32"),
    "action": ("chunk_size", "linear_acc32"),
}


def resolve_pi05_execution_components(value, *, entry_point: str, legacy_prefill_chunk: int = 0):
    """Resolve one root config into Pi0.5's three stable child views."""
    if (isinstance(legacy_prefill_chunk, bool) or not isinstance(legacy_prefill_chunk, int)
            or legacy_prefill_chunk < 0 or (legacy_prefill_chunk and legacy_prefill_chunk % 16)):
        raise ValueError(f"{entry_point}: legacy Pi0.5 prefill chunk must be 0 or a "
                         f"positive multiple of 16, got {legacy_prefill_chunk!r}")
    root = normalize_rpu_execution(value, entry_point=entry_point,
                                   supported=PI05_EXECUTION_SUPPORTED,
                                   supported_components=PI05_EXECUTION_COMPONENTS)
    views = [resolve_component_rpu_execution(root, component, entry_point=entry_point,
        supported=PI05_EXECUTION_SUPPORTED,
        supported_components=PI05_EXECUTION_COMPONENTS, profile_auto={stage: {"chunk_size": chunk}})
        for component, stage, chunk in ((PI05_VISION_COMPONENT, "vision", "auto"),
            (PI05_TEXT_COMPONENT, "prefill", legacy_prefill_chunk or "auto"),
            (PI05_ACTION_COMPONENT, "action", "auto"))]
    return root, *views


def _mark_execution_process_unsafe(reason: str) -> None:
    global _UNSAFE_PROCESS_REASON
    if _UNSAFE_PROCESS_REASON is None:
        _UNSAFE_PROCESS_REASON = str(reason)


def _require_execution_process_safe() -> None:
    # Unlike a normal process-terminal owner (e.g. HY), uncertain native
    # cleanup forbids all subsequent execution, including already-bound owners.
    if _UNSAFE_PROCESS_REASON is not None:
        raise RuntimeError(f"RPU process is unsafe: {_UNSAFE_PROCESS_REASON}; restart the process")


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
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return a detached, deeply read-only execution mapping."""
    return _FrozenExecutionMapping({
        key: _freeze_rpu_execution(item) if isinstance(item, Mapping) else item
        for key, item in value.items()
    })


def _normalize_stage_scope(
    value: Mapping[str, Any],
    *,
    entry_point: str,
    path: str,
    supported: Mapping[str, Collection[str]] | None,
) -> dict[str, dict[str, str | int]]:
    """Validate one top-level or component-local stage mapping."""
    unknown_stages = set(value) - set(_STAGES)
    if unknown_stages:
        raise ValueError(
            f"{entry_point}: {path} has unknown stage(s) "
            f"{sorted(map(str, unknown_stages))}; supported stages are {list(_STAGES)}"
        )
    if supported is not None:
        unsupported_stages = set(value) - set(supported)
        if unsupported_stages:
            raise ValueError(
                f"{entry_point}: {path} stage(s) "
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
                f"{entry_point}: {path}[{stage!r}] must be a mapping, "
                f"got {type(stage_value).__name__}"
            )
        unknown_fields = set(stage_value) - set(_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"{entry_point}: {path}[{stage!r}] has unknown field(s) "
                f"{sorted(map(str, unknown_fields))}; supported fields are {list(_FIELDS)}"
            )
        if supported is not None:
            unsupported_fields = set(stage_value) - set(supported[stage])
            if unsupported_fields:
                raise ValueError(
                    f"{entry_point}: {path}[{stage!r}] field(s) "
                    f"{sorted(map(str, unsupported_fields))} are not supported "
                    "by this model entry point"
                )
        elif set(stage_value).intersection(_COLD_BOOL_FIELDS):
            raise ValueError(
                f"{entry_point}: {path}[{stage!r}] cold boolean controls "
                "require an explicit supported component/stage declaration"
            )

        stage_config: dict[str, str | int] = {}
        for field in _FIELDS:
            if field not in stage_value:
                continue
            field_value = stage_value[field]
            if field in _COLD_BOOL_FIELDS:
                if not isinstance(field_value, bool):
                    raise TypeError(
                        f"{entry_point}: {path}[{stage!r}][{field!r}] "
                        f"must be bool, got {field_value!r}"
                    )
                stage_config[field] = field_value
            elif field == "chunk_size":
                if isinstance(field_value, str) and field_value == "auto":
                    stage_config[field] = "auto"
                elif (
                    isinstance(field_value, bool)
                    or not isinstance(field_value, int)
                    or field_value <= 0
                    or field_value % 16 != 0
                ):
                    raise ValueError(
                        f"{entry_point}: {path}[{stage!r}]['chunk_size'] "
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
                        f"{entry_point}: {path}[{stage!r}]['padding_rows'] "
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
                    f"{entry_point}: {path}[{stage!r}]['padding_budget'] "
                    f"must be a non-negative integer, got {field_value!r}"
                )
            else:
                stage_config[field] = field_value
        if (
            isinstance(stage_config.get("padding_rows"), int)
            and "padding_budget" in stage_config
        ):
            raise ValueError(
                f"{entry_point}: {path}[{stage!r}]['padding_rows'] "
                "is exact and conflicts with 'padding_budget'"
            )
        if stage_config:
            normalized[stage] = stage_config
    return normalized


def _component_capabilities(
    value: Mapping[str, Mapping[str, Collection[str]]] | None,
    *,
    entry_point: str,
) -> tuple[
    Mapping[str, Mapping[str, Collection[str]]] | None,
    Mapping[str, Collection[str]] | None,
]:
    """Validate registered stable IDs and derive their top-level stage union."""
    if value is None:
        return None, None
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{entry_point}: supported_components must be a mapping, got "
            f"{type(value).__name__}"
        )
    normalized: dict[str, dict[str, tuple[str, ...]]] = {}
    union: dict[str, set[str]] = {}
    for component_id, stages in value.items():
        if not isinstance(component_id, str) or not component_id:
            raise TypeError(
                f"{entry_point}: registered component IDs must be non-empty strings, "
                f"got {component_id!r}"
            )
        if not isinstance(stages, Mapping):
            raise TypeError(
                f"{entry_point}: capability for component {component_id!r} must "
                f"be a mapping, got {type(stages).__name__}"
            )
        unknown_stages = set(stages) - set(_STAGES) - {"decode"}
        if unknown_stages:
            raise ValueError(
                f"{entry_point}: component {component_id!r} capability has unknown "
                f"stage(s) {sorted(map(str, unknown_stages))}"
            )
        normalized_stages = {}
        for stage, fields in stages.items():
            if isinstance(fields, (str, bytes)) or not isinstance(fields, Collection):
                raise TypeError(
                    f"{entry_point}: capability for component {component_id!r} "
                    f"stage {stage!r} must be a field collection"
                )
            if stage == "decode" and fields:
                raise ValueError(f"{entry_point}: fixed decode capability has no public fields")
            unknown_fields = set(fields) - set(_FIELDS)
            if unknown_fields:
                raise ValueError(
                    f"{entry_point}: component {component_id!r} capability stage "
                    f"{stage!r} has unknown field(s) "
                    f"{sorted(map(str, unknown_fields))}"
                )
            normalized_fields = tuple(field for field in _FIELDS if field in fields)
            normalized_stages[stage] = normalized_fields
            # A registered fixed physical stage is not a public parser stage.
            if stage in _STAGES:
                union.setdefault(stage, set()).update(normalized_fields)
        normalized[component_id] = normalized_stages
    return _freeze_rpu_execution(normalized), union


def normalize_rpu_execution(
    value: Mapping[str, Any] | None,
    *,
    entry_point: str,
    vlm_chunk_size: int | None = None,
    supported: Mapping[str, Collection[str]] | None = None,
    supported_components: Mapping[
        str, Mapping[str, Collection[str]]
    ] | None = None,
) -> Mapping[str, Any]:
    """Return a detached, deeply read-only config or fail before loading."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{entry_point}: rpu_execution must be a mapping, got "
            f"{type(value).__name__}"
        )

    component_capabilities, component_union = _component_capabilities(
        supported_components, entry_point=entry_point
    )
    allowed_top_level = set(_STAGES)
    if supported is not None and "model" in supported:
        allowed_top_level.add("model")
    if component_capabilities is not None:
        allowed_top_level.add(_COMPONENTS)
    unknown_stages = set(value) - allowed_top_level
    if unknown_stages:
        raise ValueError(
            f"{entry_point}: rpu_execution has unknown stage(s) "
            f"{sorted(map(str, unknown_stages))}; supported stages are {list(_STAGES)}"
        )
    top_level = {stage: value[stage] for stage in _STAGES if stage in value}
    normalized: dict[str, Any] = _normalize_stage_scope(
        top_level,
        entry_point=entry_point,
        path="rpu_execution",
        supported=supported if supported is not None else component_union,
    )
    if "model" in value:
        model_scope = value["model"]
        if not isinstance(model_scope, Mapping):
            raise TypeError(f"{entry_point}: rpu_execution['model'] must be a mapping")
        unknown = set(model_scope) - {"num_cores"}
        if unknown:
            raise ValueError(f"{entry_point}: rpu_execution['model'] has unknown field(s) {sorted(unknown)}")
        if "num_cores" in model_scope:
            if "num_cores" not in supported["model"]:
                raise ValueError(f"{entry_point}: model.num_cores is not supported by this model entry point")
            # Keep the public parser board-free; architecture admission follows
            # AutoConfig and precedes full weight loading.
            value_cores = model_scope["num_cores"]
            if (isinstance(value_cores, bool) or not isinstance(value_cores, int)
                    or value_cores not in (4, 6, 8)):
                raise ValueError(f"{entry_point}: model.num_cores must be one of 4, 6 or 8")
            normalized["model"] = {"num_cores": value_cores}

    if _COMPONENTS in value:
        raw_components = value[_COMPONENTS]
        if not isinstance(raw_components, Mapping):
            raise TypeError(
                f"{entry_point}: rpu_execution['components'] must be a mapping, "
                f"got {type(raw_components).__name__}"
            )
        invalid_ids = [
            component_id for component_id in raw_components
            if not isinstance(component_id, str) or not component_id
        ]
        if invalid_ids:
            raise TypeError(
                f"{entry_point}: rpu_execution component IDs must be non-empty "
                f"strings, got {invalid_ids!r}"
            )
        unknown_components = set(raw_components) - set(component_capabilities)
        if unknown_components:
            raise ValueError(
                f"{entry_point}: rpu_execution has unknown component(s) "
                f"{sorted(unknown_components)}"
            )
        normalized_components = {}
        for component_id in sorted(raw_components):
            component_value = raw_components[component_id]
            if not isinstance(component_value, Mapping):
                raise TypeError(
                    f"{entry_point}: rpu_execution['components']"
                    f"[{component_id!r}] must be a mapping, got "
                    f"{type(component_value).__name__}"
                )
            component_config = _normalize_stage_scope(
                component_value,
                entry_point=entry_point,
                path=f"rpu_execution['components'][{component_id!r}]",
                supported=component_capabilities[component_id],
            )
            if component_config:
                normalized_components[component_id] = component_config
        if normalized_components:
            normalized[_COMPONENTS] = normalized_components

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

    ordered = {
        stage: {
            field: normalized[stage][field]
            for field in _FIELDS
            if field in normalized[stage]
        }
        for stage in _STAGES
        if stage in normalized
    }
    if _COMPONENTS in normalized:
        ordered[_COMPONENTS] = normalized[_COMPONENTS]
    if "model" in normalized:
        ordered["model"] = normalized["model"]
    return _freeze_rpu_execution(ordered)


# Cold text configuration is shared with board-free CI input inspection.
# Keep these translators identical to the production import/re-export paths.
_CAUSAL_DECODER_EXECUTION_SUPPORTED = {
    "prefill": ("chunk_size", "padding_rows", "padding_budget"),
}
_QWEN3_EXECUTION_SUPPORTED = {
    **_CAUSAL_DECODER_EXECUTION_SUPPORTED,
    "prefill": ("chunk_size", "padding_rows", "padding_budget", "linear_acc32"),
    "model": ("num_cores",),
}
_QWEN3_5_EXECUTION_SUPPORTED = {
    "model": ("num_cores",),
    "prefill": ("chunk_size", "padding_rows", "padding_budget", "linear_acc32"),
    # The adapter preflight allows only zero/auto Vision padding.
    "vision": ("chunk_size", "padding_rows", "padding_budget", "linear_acc32"),
}
QWEN3_REDUCED_CORE_ASSET_ALIASES = frozenset({
    "qwen3-0.6b", "qwen3-0.6b-instruct", "qwen3-1.7b",
    "qwen3-4b", "qwen3-4b-instruct", "qwen3-8b",
})
QWEN3_VL_REDUCED_CORE_ASSET_ALIASES = frozenset({
    "qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b",
})



def is_qwen3_17b_w8a16_core_config(config):
    """The exact 1.7B W8 decoder/head format; no other reduced W8 profile."""
    return (
        tuple(getattr(config, "architectures", ()) or ()) == ("Qwen3ForCausalLM",)
        and tuple(getattr(config, key, None) for key in (
            "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size"))
            == ("qwen3", 2048, 6144, 28, 16, 8, 128, 151936)
        and getattr(config, "tie_word_embeddings", None) is False
        and getattr(config, "rms_norm_eps", None) == 1e-6
        and getattr(config, "quantization_config", None) is None
        and getattr(config, "quant_config", None) == {
            "method": "w8a16", "mode": "per_channel_symmetric", "qaxis": 0,
            "skip_modules": [], "quantized_lm_head": True,
            "quantized_embed_tokens": False, "lm_head_untied": True,
            "embed_tokens_untied": False,
        }
    )


def qwen3_core_profile(config, num_cores):
    """Resolve exact plain geometry before model loading or weight mutation.

    Keep this import lazy: public example and harness syntax checks load this
    stdlib leaf without initializing the backend or inspecting model assets.
    """
    from rpu_backend.runtime.topology import decoder_geometry_profile

    profile = decoder_geometry_profile(
        hidden_size=getattr(config, "hidden_size", None),
        intermediate_size=getattr(config, "intermediate_size", None),
        num_q_heads=getattr(config, "num_attention_heads", None),
        num_kv_heads=getattr(config, "num_key_value_heads", None),
        head_dim=getattr(config, "head_dim", None),
        vocab_size=getattr(config, "vocab_size", None))
    layers = getattr(config, "num_hidden_layers", None)
    plain = (profile is not None and getattr(config, "model_type", None) == "qwen3"
             and not isinstance(layers, bool) and isinstance(layers, Integral)
             and layers == profile.num_layers)
    quantized_17b = num_cores in (4, 6) and is_qwen3_17b_w8a16_core_config(config)
    plain = plain and (quantized_17b or not (getattr(config, "quant_config", None)
                           or getattr(config, "quantization_config", None)))
    plain = plain and not (getattr(config, "attention_bias", False)
                           or getattr(config, "mlp_bias", False)
                           or getattr(config, "use_sliding_window", False))
    plain = plain and getattr(config, "hidden_act", "silu") == "silu"
    layer_types = getattr(config, "layer_types", None)
    plain = plain and (layer_types is None or (len(layer_types) == profile.num_layers
                        and all(kind == "full_attention" for kind in layer_types)))
    plain = plain and getattr(config, "partial_rotary_factor", 1.0) == 1.0
    rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    plain = plain and isinstance(rope, Mapping) and rope.get("rope_type", rope.get("type", "default")) == "default"
    if not plain and num_cores != 8:
        raise ValueError(
            "model.num_cores=4/6 requires an exact plain FP16 Qwen3 "
            "0.6B/1.7B/4B/8B or 1.7B W8A16 decoder/head profile with full attention and default RoPE")
    return profile if plain else None


def validate_qwen3_core_profile(config, num_cores):
    """Config-only reduced-core admission, shared by loader and cache callers."""
    return qwen3_core_profile(config, num_cores) is not None


def qwen3_vl_text_core_profile(config, num_cores):
    """Exact dense M-RoPE text admission, separate from plain Qwen3."""
    from rpu_backend.runtime.topology import decoder_geometry_profile

    profile = decoder_geometry_profile(
        hidden_size=getattr(config, "hidden_size", None),
        intermediate_size=getattr(config, "intermediate_size", None),
        num_q_heads=getattr(config, "num_attention_heads", None),
        num_kv_heads=getattr(config, "num_key_value_heads", None),
        head_dim=getattr(config, "head_dim", None),
        vocab_size=getattr(config, "vocab_size", None))
    rope = (getattr(config, "rope_parameters", None)
            or getattr(config, "rope_scaling", None) or {})
    layers = getattr(config, "num_hidden_layers", None)
    layer_types = getattr(config, "layer_types", None)
    valid = (
        profile is not None and profile.hidden_size in (2048, 2560, 4096)
        and getattr(config, "model_type", None) == "qwen3_vl_text"
        and isinstance(layers, Integral) and not isinstance(layers, bool)
        and layers == profile.num_layers
        and getattr(config, "hidden_act", None) == "silu"
        and getattr(config, "rms_norm_eps", None) == 1e-6
        and getattr(config, "attention_bias", None) is False
        and getattr(config, "use_cache", None) is True
        and not getattr(config, "mlp_bias", False)
        and not getattr(config, "use_sliding_window", False)
        and getattr(config, "partial_rotary_factor", 1.0) == 1.0
        and (layer_types is None or (
            len(layer_types) == layers and all(t == "full_attention" for t in layer_types)))
        and all(getattr(config, k, None) is None for k in ("quant_config", "quantization_config"))
        and isinstance(rope, Mapping)
        and rope.get("rope_type", rope.get("type")) == "default"
        and rope.get("rope_theta", getattr(config, "rope_theta", None)) == 5_000_000.0
        and rope.get("mrope_interleaved") is True
        and isinstance(rope.get("mrope_section"), (list, tuple))
        and tuple(rope["mrope_section"]) == (24, 20, 20)
    )
    if not valid and num_cores != 8:
        raise ValueError("model.num_cores=4/6 requires exact dense FP16 Qwen3-VL "
                         "2B/4B/8B text semantics with interleaved M-RoPE")
    return profile if valid else None


# Extra plain-text rows may buy a better chunk plan. Translate the legacy
# environment once during install; 0 disables optional padding entirely.
_PREFILL_PADDING_BUDGET_ENV = "RPU_CAUSAL_PREFILL_PADDING_BUDGET"
_PREFILL_PADDING_BUDGET_DEFAULT = 64
QWEN3_5_OPTIONAL_PADDING_CAP = 64
QWEN3_5_TOTAL_PADDING_CAP = 63 + QWEN3_5_OPTIONAL_PADDING_CAP


def _validate_qwen35_text_padding(prefill) -> None:
    """The public text padding admission boundary, also used by cold queries."""
    for field, limit in (("padding_budget", QWEN3_5_OPTIONAL_PADDING_CAP),
                         ("padding_rows", QWEN3_5_TOTAL_PADDING_CAP)):
        value = prefill.get(field)
        if isinstance(value, int) and value > limit:
            raise ValueError(
                f"Qwen3.5 prefill {field} exceeds the cache-backed maximum "
                f"{limit}: got {value}. Refusing before loading model weights.")


def _cold_causal_decoder_execution(execution_config):
    """Translate legacy padding once, before the canonical Session is bound."""
    execution = dict(execution_config or {})
    prefill = dict(execution.get("prefill", {}))
    if not isinstance(prefill.get("padding_rows", "auto"), int) and "padding_budget" not in prefill:
        try:
            budget = int(os.environ.get(
                _PREFILL_PADDING_BUDGET_ENV, _PREFILL_PADDING_BUDGET_DEFAULT))
        except ValueError as exc:
            raise ValueError(f"{_PREFILL_PADDING_BUDGET_ENV} must be an integer") from exc
        if budget < 0:
            raise ValueError(f"{_PREFILL_PADDING_BUDGET_ENV} must be non-negative")
        prefill["padding_budget"] = budget
    execution["prefill"] = prefill
    return execution


def _resolve_text_install_options(
    max_seq_len, *, execution_config=None, resolved_options=None,
    cold_snapshot=None,
) -> tuple[int, int, int, dict[str, object]]:
    """Validate one immutable cold snapshot of text planner controls.

    ``QWEN3_5_TEXT_CHUNK`` is retained as a legacy alias only when the
    canonical request omits ``prefill.chunk_size``.  A present canonical value
    therefore cannot be silently narrowed by a stale CI environment.
    """
    if resolved_options is not None:
        if (
            not isinstance(resolved_options, tuple)
            or len(resolved_options) != 4
        ):
            raise TypeError("resolved_options must be a four-item tuple")
        # The adapter resolves this snapshot before any dtype/device move.  A
        # direct installer call may omit it and follows the normal cold path.
        return (
            int(resolved_options[0]),
            int(resolved_options[1]),
            int(resolved_options[2]),
            resolved_options[3],
        )
    if (
        isinstance(max_seq_len, bool)
        or not isinstance(max_seq_len, Integral)
        or max_seq_len < 1
    ):
        raise ValueError(
            f"Qwen3.5 max_seq_len must be a positive integer, got {max_seq_len!r}"
        )
    if cold_snapshot is not None and not isinstance(cold_snapshot, Mapping):
        raise TypeError("cold_snapshot must be a mapping")
    if cold_snapshot is None:
        env_chunk_present = "QWEN3_5_TEXT_CHUNK" in os.environ
        env_padding_present = "QWEN3_5_TEXT_PADDING_BUDGET" in os.environ
        env_chunk_raw = os.environ.get("QWEN3_5_TEXT_CHUNK", "0")
        env_padding_raw = os.environ.get("QWEN3_5_TEXT_PADDING_BUDGET", "64")
    else:
        env_chunk_present = bool(cold_snapshot.get("chunk_present", False))
        env_padding_present = bool(cold_snapshot.get(
            "padding_present",
            cold_snapshot.get("padding_source") == "legacy_env",
        ))
        env_chunk_raw = str(cold_snapshot.get("chunk_raw", "0"))
        env_padding_raw = str(cold_snapshot.get("padding_raw", "64"))
    try:
        env_chunk = int(env_chunk_raw)
        env_padding = int(env_padding_raw)
    except ValueError as exc:
        raise ValueError(
            "QWEN3_5_TEXT_CHUNK and QWEN3_5_TEXT_PADDING_BUDGET must be integers"
        ) from exc
    if 0 < env_chunk < 64:
        raise ValueError("QWEN3_5_TEXT_CHUNK must be 0 or at least 64")
    if env_padding < 0:
        raise ValueError("QWEN3_5_TEXT_PADDING_BUDGET must be non-negative")
    if env_padding > QWEN3_5_OPTIONAL_PADDING_CAP:
        raise ValueError(
            "QWEN3_5_TEXT_PADDING_BUDGET exceeds the cache-backed maximum "
            f"{QWEN3_5_OPTIONAL_PADDING_CAP}: got {env_padding}"
        )
    if execution_config is None:
        execution_config = {}
    if not isinstance(execution_config, Mapping):
        raise TypeError(
            "Qwen3.5 execution_config must be a mapping, got "
            f"{type(execution_config).__name__}"
        )
    prefill = execution_config.get("prefill", {})
    if not isinstance(prefill, Mapping):
        raise TypeError("Qwen3.5 execution_config['prefill'] must be a mapping")
    canonical_chunk_present = "chunk_size" in prefill
    canonical_chunk = prefill.get("chunk_size", "auto")
    if canonical_chunk_present:
        if canonical_chunk == "auto":
            chunk_size_cap = 0
        elif isinstance(canonical_chunk, Integral) and not isinstance(canonical_chunk, bool):
            if canonical_chunk <= 0 or canonical_chunk % 64:
                raise ValueError(
                    "Qwen3.5 prefill chunk_size must be a positive multiple of 64, got "
                    f"{canonical_chunk}"
                )
            if env_chunk > 0 and env_chunk != canonical_chunk:
                raise ValueError(
                    "QWEN3_5_TEXT_CHUNK conflicts with canonical "
                    f"rpu_execution prefill chunk_size: {env_chunk} != "
                    f"{canonical_chunk}"
                )
            chunk_size_cap = 0
        else:
            # normalize_rpu_execution normally catches this; keep this helper
            # safe for direct/host callers too.
            raise ValueError(
                "Qwen3.5 prefill chunk_size must be 'auto' or an integer"
            )
    else:
        chunk_size_cap = max(0, env_chunk)

    if "padding_budget" in prefill:
        raw_budget = prefill["padding_budget"]
        if (
            isinstance(raw_budget, bool)
            or not isinstance(raw_budget, Integral)
        ):
            raise ValueError(
                "Qwen3.5 prefill padding_budget must be a non-negative integer"
            )
        padding_budget = int(raw_budget)
    else:
        padding_budget = env_padding
    if padding_budget < 0:
        raise ValueError("QWEN3_5_TEXT_PADDING_BUDGET must be non-negative")
    if padding_budget > QWEN3_5_OPTIONAL_PADDING_CAP:
        raise ValueError(
            "QWEN3_5_TEXT_PADDING_BUDGET exceeds the cache-backed maximum "
            f"{QWEN3_5_OPTIONAL_PADDING_CAP}: got {padding_budget}"
        )
    snapshot = MappingProxyType({
        "chunk_present": env_chunk_present,
        "padding_present": env_padding_present,
        "canonical_chunk_present": canonical_chunk_present,
        "canonical_chunk": canonical_chunk,
        "chunk_raw": env_chunk_raw,
        "padding_raw": env_padding_raw,
        "chunk_source": (
            "canonical" if canonical_chunk_present
            else "legacy_env" if env_chunk_present else "default"
        ),
        "chunk_effective_cap": chunk_size_cap,
        "padding_source": (
            "canonical" if "padding_budget" in prefill
            else "legacy_env" if env_padding_present
            else "default"
        ),
        "padding_effective_budget": padding_budget,
        "canonical_padding_present": "padding_budget" in prefill,
    })
    return int(max_seq_len), chunk_size_cap, padding_budget, snapshot


def resolve_component_rpu_execution(
    value: Mapping[str, Any] | None,
    component_id: str,
    *,
    entry_point: str,
    supported_components: Mapping[
        str, Mapping[str, Collection[str]]
    ],
    profile_auto: Mapping[str, Mapping[str, Any]],
    supported: Mapping[str, Collection[str]] | None = None,
) -> Mapping[str, Mapping[str, str | int]]:
    """Resolve one registered child: override, stage default, profile AUTO."""
    component_capabilities, _union = _component_capabilities(
        supported_components, entry_point=entry_point
    )
    if not isinstance(component_id, str) or not component_id:
        raise TypeError(
            f"{entry_point}: component_id must be a non-empty string, got "
            f"{component_id!r}"
        )
    if component_capabilities is None:
        raise ValueError(f"{entry_point}: no rpu_execution components are registered")
    if component_id not in component_capabilities:
        raise ValueError(
            f"{entry_point}: unknown rpu_execution component {component_id!r}"
        )
    normalized = normalize_rpu_execution(
        value,
        entry_point=entry_point,
        supported=supported,
        supported_components=component_capabilities,
    )
    component_supported = component_capabilities[component_id]
    resolved = {
        stage: dict(fields)
        for stage, fields in normalize_rpu_execution(
            profile_auto,
            entry_point=f"{entry_point} profile AUTO for {component_id!r}",
            supported=component_supported,
        ).items()
    }
    component_overrides = normalized.get(_COMPONENTS, {}).get(component_id, {})
    for source in (normalized, component_overrides):
        for stage, allowed_fields in component_supported.items():
            if stage not in source:
                continue
            patch = {
                field: source[stage][field]
                for field in allowed_fields
                if field in source[stage]
            }
            stage_config = resolved.setdefault(stage, {})
            if isinstance(patch.get("padding_rows"), int):
                stage_config.pop("padding_budget", None)
            if "padding_budget" in patch and isinstance(
                stage_config.get("padding_rows"), int
            ):
                stage_config.pop("padding_rows")
            stage_config.update(patch)
    return normalize_rpu_execution(
        resolved,
        entry_point=entry_point,
        supported=component_supported,
    )


def bind_rpu_execution(
    owner: Any,
    value: Mapping[str, Any] | None = None,
    *,
    entry_point: str,
    supported: Mapping[str, Collection[str]] | None = None,
    supported_components: Mapping[
        str, Mapping[str, Collection[str]]
    ] | None = None,
) -> Mapping[str, Any]:
    """Validate and attach one cold config, preserving an equal existing one."""
    existing = getattr(owner, "_rpu_execution", _MISSING)
    normalized = normalize_rpu_execution(
        existing if value is None and existing is not _MISSING else value,
        entry_point=entry_point,
        supported=supported,
        supported_components=supported_components,
    )
    if existing is not _MISSING:
        existing_normalized = normalize_rpu_execution(
            existing,
            entry_point=entry_point,
            supported=supported,
            supported_components=supported_components,
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


class ExecutionSession:
    """One stop-the-world generation gate for a live execution owner.

    Physical owners keep their existing graph lifecycle.  The session only
    serializes forward/reconfigure, validates one canonical mapping, and
    publishes a new generation after the owner callback commits.
    """

    def __init__(
        self,
        owner: Any,
        value: Mapping[str, Any] | None,
        *,
        entry_point: str,
        supported: Mapping[str, Collection[str]] | None = None,
        supported_components: Mapping[
            str, Mapping[str, Collection[str]]
        ] | None = None,
        validate=None,
        apply=None,
        rollback=None,
        cold_config_only=None,
        graph_mode: str,
    ) -> None:
        _require_execution_process_safe()
        self._owner = owner
        self._entry_point = entry_point
        self._supported = supported
        self._supported_components, _component_union = _component_capabilities(
            supported_components, entry_point=entry_point
        )
        self._validate = validate
        self._apply = apply
        self._rollback = rollback
        if cold_config_only is not None and not callable(cold_config_only):
            raise TypeError("cold_config_only must be an owner callback")
        self._cold_config_only = cold_config_only
        self._graph_mode = str(graph_mode)
        self._lock = threading.RLock()
        self._active = 0
        self._started = False
        self._closed = False
        self._shutting_down = False
        self._poisoned = False
        self._generation = 0
        self._commits = 0
        self._failures = 0
        self._config = normalize_rpu_execution(
            value,
            entry_point=entry_point,
            supported=supported,
            supported_components=self._supported_components,
        )
        self._config_views = weakref.WeakSet()
        self._planner_costs = {}
        self._planner_native = {}
        self._planner_native_catalogs = {}
        self._planner_native_revalidators = {}
        self._planner_revalidating_native = None
        self._planner_cost_observer = None
        self._planner_calibration = None
        self._prepared_execution_plans = {}

    @property
    def config(self) -> Mapping[str, Any]:
        return self._config

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def graph_mode(self) -> str:
        return self._graph_mode

    def register_config_view(self, view: Any) -> None:
        """Keep another live facade's read-only config mirror synchronized."""
        with self._lock:
            if view is self._owner:
                return
            self._config_views.add(view)
            vars(view)["_rpu_execution"] = self._config

    def require_cold(self) -> None:
        """Reject replacing model state after this session has been used."""
        with self._lock:
            _require_execution_process_safe()
            if (self._started or self._closed or self._poisoned
                    or self._planner_revalidating_native is not None):
                raise RuntimeError(
                    f"{self._entry_point}: model replacement requires a cold session "
                    "before the first forward"
                )

    def _install_planner_costs(self, component: str, stage: str, install, *,
                               native_prefix="", allow_singleton=False) -> None:
        """Internal trusted-artifact hook; never a configuration field.

        The verifier installs the native parent and returns a validated scope
        and immutable certificates. Run under the same cold session lock, so a
        forward cannot race partial native/Python installation. A failed install
        may have changed native state and therefore poisons this session.
        """
        with self._lock:
            _require_execution_process_safe()
            if self._started or self._closed or self._poisoned:
                raise RuntimeError("planner costs must be installed before the first forward")
            if self._supported_components:
                supported = self._supported_components.get(component, {})
            else:
                if component:
                    raise ValueError("a leaf cost binding must use the empty component ID")
                supported = self._supported if self._supported is not None else _STAGES
            # Decode has a fixed public ABI, not a configurable parser stage.
            # Its physical planner evidence still belongs to the text owner.
            if stage not in supported and not (stage == "decode" and "prefill" in supported):
                raise ValueError("planner cost binding names an undeclared component/stage")
            if not isinstance(native_prefix, str) or re.fullmatch(r"[a-z0-9_]*", native_prefix) is None:
                raise ValueError("planner costs require an internal native prefix")
            key = (component, stage, native_prefix)
            if key in self._planner_costs:
                raise RuntimeError("planner costs are immutable after installation")
            if not callable(install):
                raise TypeError("planner costs require the trusted installation callback")
            if type(allow_singleton) is not bool:
                raise TypeError("trusted singleton admission must be bool")
            from rpu_backend.runtime.execution_planner import PlannerCostCertificate, PlannerCostScope

            try:
                scope, certificates = install()
                if self._started:
                    raise RuntimeError("planner cost installation must not execute the owner")
                if (not isinstance(scope, PlannerCostScope)
                        or not isinstance(certificates, tuple)
                        or (not certificates and not allow_singleton)
                        or any(not isinstance(item, PlannerCostCertificate)
                               or item.scope != scope or item.sealed is not True
                               or not isinstance(item.candidate_costs, tuple)
                               or any(not isinstance(row, tuple) for row in item.candidate_costs)
                               for item in certificates)):
                    raise ValueError("planner cost installer must return sealed owner-local certificates")
                self._planner_costs[key] = (scope, certificates)
                self._clear_prepared_execution_plans()
            except BaseException:
                self.poison()
                raise

    def planner_costs(self, component: str, stage: str, native_prefix=""):
        """Read the immutable binding for this exact child, never a global one."""
        with self._lock:
            return self._planner_costs_locked(component, stage, native_prefix)

    def _planner_costs_locked(self, component, stage, native_prefix):
        if self._closed or self._poisoned:
            raise RuntimeError("cannot plan with a closed or poisoned execution session")
        revalidating = self._planner_revalidating_native
        if revalidating is not None and (component, native_prefix) == revalidating[1:3]:
            return None, ()  # The new native parent is not yet verified.
        return self._planner_costs.get((component, stage, native_prefix), (None, ()))

    @contextmanager
    def _collect_planner_costs(self, observer, *, read_only=False):
        """Observe cold planning, or read an unchanged live cost domain at rest."""
        with self._lock:
            _require_execution_process_safe()
            if type(read_only) is not bool:
                raise TypeError("read_only cost collection must be bool")
            def live_identity():
                _require_execution_process_safe()
                if (not self._started or self._active or self._closed or self._poisoned
                        or self._shutting_down or self._planner_revalidating_native is not None
                        or self._planner_calibration is not None):
                    raise RuntimeError("read-only cost collection requires an idle live FINAL session")
                return (self.stats(), dict(self._planner_costs),
                        dict(self._planner_native_catalogs), dict(self._planner_native_revalidators),
                        tuple(sorted((key, id(value[0]), value[1])
                                     for key, value in self._planner_native.items())))
            if read_only:
                before = live_identity()
            elif self._planner_revalidating_native is None:
                self.require_cold()
            elif self._closed or self._poisoned:
                raise RuntimeError("cannot revalidate a closed or poisoned session")
            if self._planner_cost_observer is not None or not callable(observer):
                raise ValueError("cost collection requires one non-nested observer")
            self._planner_cost_observer = observer
            try:
                yield
                if read_only:
                    if live_identity() != before:
                        raise RuntimeError("read-only cost collection changed the live planner authority")
                elif self._planner_revalidating_native is None:
                    self.require_cold()
                elif self._closed or self._poisoned:
                    raise RuntimeError("native revalidation lost its live session")
            finally:
                self._planner_cost_observer = None

    def _bind_planner_owner(self, owner: Any, component: str = "") -> None:
        """Bind a real child to this session, without a process-wide handle map."""
        with self._lock:
            if self._closed or self._poisoned:
                raise RuntimeError("planner owners must be bound before the first forward")
            if self._supported_components:
                if component not in self._supported_components:
                    raise ValueError("planner owner names an undeclared component")
            elif component:
                raise ValueError("a leaf planner owner must use the empty component ID")
            existing = getattr(owner, "_planner_cost_session", None)
            if existing is not None:
                if (not isinstance(existing, tuple) or len(existing) != 2 or
                        not isinstance(existing[0], weakref.ReferenceType) or
                        existing[0]() is not self or not isinstance(existing[1], frozenset)):
                    raise ValueError("planner owner is already bound to a different session")
                if component in existing[1]:
                    return  # An existing facade binding is read-only, even after forward.
                if self._started:
                    raise RuntimeError("planner owners must be bound before the first forward")
                vars(owner)["_planner_cost_session"] = (
                    existing[0], existing[1] | {component})
                return
            if self._started:
                raise RuntimeError("planner owners must be bound before the first forward")
            vars(owner)["_planner_cost_session"] = (weakref.ref(self), frozenset({component}))

    def _record_planner_native(self, owner, component, stage, native) -> None:
        """Record the actual production planner's native leaf, never artifact labels."""
        with self._lock:
            self._record_planner_native_locked(owner, component, stage, native)

    def _record_planner_native_locked(self, owner, component, stage, native):
        _require_execution_process_safe()
        if self._closed or self._poisoned:
            raise RuntimeError("cannot register native planning on a closed or poisoned session")
        if (not isinstance(native, tuple) or len(native) != 2 or
                not isinstance(native[0], str) or
                re.fullmatch(r"[a-z][a-z0-9_]*", native[0]) is None or
                type(native[1]) is not int or not 0 < native[1] < 2**63):
            raise ValueError("native planner binding requires its literal prefix and actual handle")
        key = (component, stage, native[0])
        previous = self._planner_native.get(key)
        same_previous = (previous is not None and previous[0] is owner
                         and previous[1] == native[1])
        target = (owner, component, native[0], native[1])
        revalidating = self._planner_revalidating_native
        if revalidating is not None:
            same_target = owner is revalidating[0] and target[1:] == revalidating[1:]
            if not same_target and not same_previous:
                raise RuntimeError("native revalidation cannot rebuild another owner or handle")
            self._planner_native[key] = (owner, native[1])
            return
        if self._planner_cost_observer is not None and self._started and not same_previous:
            raise RuntimeError("read-only cost collection cannot replace a native owner")
        proof_key = (component, native[0])
        proof = self._planner_native_catalogs.get(proof_key)
        if same_previous and (proof is None or proof[0] == native[1]):
            return  # Already validated; do not scan costs or rewrite the binding.
        changed = (previous is not None and previous[0] is not owner) or (
            proof[0] != native[1] if proof is not None
            else previous is not None and not same_previous)
        has_costs = proof is not None or any(
            bound_component == component and prefix == native[0]
            for bound_component, _stage, prefix in self._planner_costs)
        if changed and has_costs:
            revalidate = self._planner_native_revalidators.get(proof_key)
            if proof is None or not callable(revalidate):
                raise RuntimeError("a changed native owner requires trusted cost identity revalidation")
            saved_native = dict(self._planner_native)
            self._planner_revalidating_native = target
            try:
                self._planner_native[key] = (owner, native[1])
                replacement = revalidate(native)
                actual = self._planner_native.get(key)
                if (not isinstance(replacement, tuple) or len(replacement) != len(proof)
                        or replacement[0] != native[1] or replacement[1:] != proof[1:]
                        or actual is None or actual[0] is not owner or actual[1] != native[1]):
                    raise ValueError("native revalidation did not prove the actual replacement")
                self._planner_native_catalogs[proof_key] = replacement
            except BaseException:
                self._planner_native = saved_native
                self.poison()
                raise
            finally:
                self._planner_revalidating_native = None
        self._planner_native[key] = (owner, native[1])

    def configure_callbacks(self, *, validate=None, apply=None, rollback=None,
                            cold_config_only=None) -> None:
        """Fill previously empty owner hooks before the session is used."""
        with self._lock:
            if cold_config_only is not None and not callable(cold_config_only):
                raise TypeError("cold_config_only must be an owner callback")
            updates = tuple(
                (name, current, replacement)
                for name, current, replacement in (
                    ("validate", self._validate, validate),
                    ("apply", self._apply, apply),
                    ("rollback", self._rollback, rollback),
                    ("cold_config_only", self._cold_config_only, cold_config_only),
                )
                if replacement is not None and replacement is not current
            )
            if not updates:
                return
            if self._started or self._closed or self._poisoned:
                raise RuntimeError(
                    f"{self._entry_point}: execution callbacks are immutable "
                    "after the session is used"
                )
            rebound = [name for name, current, _new in updates if current is not None]
            if rebound:
                raise RuntimeError(
                    f"{self._entry_point}: execution callback(s) already bound: "
                    f"{rebound}"
                )
            for name, _current, replacement in updates:
                setattr(self, f"_{name}", replacement)

    @contextmanager
    def execute(self):
        """Serialize one physical execution against generation changes."""
        with self._lock:
            _require_execution_process_safe()
            if self._planner_revalidating_native is not None:
                raise RuntimeError("native cost revalidation must not execute a nested forward")
            if self._closed:
                raise RuntimeError(f"{self._entry_point}: execution session is closed")
            if self._poisoned:
                raise RuntimeError(
                    f"{self._entry_point}: execution session is poisoned; "
                    "restart the process"
                )
            if self._planner_cost_observer is not None:
                raise RuntimeError("cost collection must not execute a forward")
            self._started = True
            self._active += 1
            try:
                yield self._generation
            except BaseException:
                self._clear_prepared_execution_plans()
                raise
            finally:
                self._active -= 1

    def _clear_prepared_execution_plans(self):
        for cache in self._prepared_execution_plans.values():
            cache.clear()

    def _prepared_plan_cache(self, component, stage, native_prefix):
        """Preparation for owners whose contract uses a raw one-shot Graph."""
        from rpu_backend.runtime.execution_planner import PreparedExecutionPlans

        with self._lock:
            _require_execution_process_safe()
            if self._closed or self._poisoned:
                raise RuntimeError("cannot prepare plans on a closed or poisoned session")
            key = (component, stage, native_prefix)
            if key not in self._prepared_execution_plans:
                self._prepared_execution_plans[key] = PreparedExecutionPlans()
            return self._prepared_execution_plans[key]

    def poison(self) -> None:
        """Fail closed after an owner cannot restore execution invariants."""
        with self._lock:
            self._started = True
            self._poisoned = True
            self._failures += 1
            self._clear_prepared_execution_plans()

    def _merge_patch(
        self, value: Mapping[str, Any] | None
    ) -> Mapping[str, Any]:
        if value is None:
            value = {}
        # Validate the patch itself first.  A present stage replaces that
        # stage's hot overrides, so ``{"prefill": {}}`` removes those without
        # a second deletion sentinel/parser. Omitted bound cold bools survive;
        # an explicit cold change is validated by the owner's controller.
        patch = normalize_rpu_execution(
            value,
            entry_point=self._entry_point,
            supported=self._supported,
            supported_components=self._supported_components,
        )
        merged = {
            stage: dict(fields) for stage, fields in self._config.items()
        }
        def stage_patch(old, new):
            return {
                **{key: item for key, item in old.items()
                   if key in _COLD_BOOL_FIELDS},
                **new,
            }

        for stage in set(value) - {_COMPONENTS}:
            fields = stage_patch(self._config.get(stage, {}), patch.get(stage, {}))
            if fields:
                merged[stage] = dict(fields)
            else:
                merged.pop(stage, None)
        if _COMPONENTS in value:
            components = {
                component_id: dict(stages)
                for component_id, stages in self._config.get(
                    _COMPONENTS, {}
                ).items()
            }
            patch_components = patch.get(_COMPONENTS, {})
            for component_id, raw_stage_patch in value[_COMPONENTS].items():
                if not raw_stage_patch:
                    cold = {
                        stage: stage_patch(fields, {})
                        for stage, fields in components.get(component_id, {}).items()
                        if set(fields).intersection(_COLD_BOOL_FIELDS)
                    }
                    if cold:
                        components[component_id] = cold
                    else:
                        components.pop(component_id, None)
                    continue
                component = dict(components.get(component_id, {}))
                normalized_component = patch_components.get(component_id, {})
                for stage in raw_stage_patch:
                    fields = stage_patch(
                        component.get(stage, {}), normalized_component.get(stage, {})
                    )
                    if fields:
                        component[stage] = dict(fields)
                    else:
                        component.pop(stage, None)
                if component:
                    components[component_id] = component
                else:
                    components.pop(component_id, None)
            if components:
                merged[_COMPONENTS] = components
            else:
                merged.pop(_COMPONENTS, None)
        return normalize_rpu_execution(
            merged,
            entry_point=self._entry_point,
            supported=self._supported,
            supported_components=self._supported_components,
        )

    def _select_planner_calibration(
        self, owner, stage, selection, *, component=None, native_prefix,
    ) -> None:
        """Retire through the existing transaction before selecting a P7 row.

        This private hook is called by the pinned calibration runner, not by
        the execution-config parser. Production inputs remain unchanged.
        """
        from rpu_backend.runtime.execution_planner import PlannerCalibrationSelection

        with self._lock:
            _require_execution_process_safe()
            bound, component = _planner_owner_binding(owner, stage, component)
            if bound is not self:
                raise ValueError("calibration requires the actual bound execution owner")
            native = self._planner_native.get((component, stage, native_prefix))
            if native is None or native[0] is not owner:
                raise ValueError("calibration requires the actual production native producer")
            if selection is not None and not isinstance(selection, PlannerCalibrationSelection):
                raise TypeError("calibration selection must be the private typed P7 request")
            if (self._active or self._planner_cost_observer is not None
                    or self._planner_revalidating_native is not None):
                raise RuntimeError("calibration requires a quiescent execution session")
            if self._apply is None:
                raise RuntimeError("calibration requires the owner's graph retirement transaction")
            previous = self._planner_calibration
            self._planner_calibration = (
                None if selection is None else (owner, component, stage, native_prefix, selection))
            try:
                self._reconfigure({}, force_rebuild=True)
            except BaseException:
                self._planner_calibration = previous
                raise

    def reconfigure(
        self, value: Mapping[str, Any] | None
    ) -> Mapping[str, Any]:
        """Atomically publish one quiescent execution-config generation."""
        with self._lock:
            if self._planner_calibration is not None:
                raise RuntimeError("clear P7 calibration before public reconfigure")
            return self._reconfigure(value)

    def _reconfigure(self, value, *, force_rebuild=False):
        with self._lock:
            _require_execution_process_safe()
            if self._planner_revalidating_native is not None:
                raise RuntimeError("native cost revalidation must not reconfigure the owner")
            if self._closed:
                raise RuntimeError(f"{self._entry_point}: execution session is closed")
            if self._poisoned:
                raise RuntimeError(
                    f"{self._entry_point}: execution session is poisoned; "
                    "restart the process"
                )
            if self._planner_cost_observer is not None:
                raise RuntimeError("cost collection must not reconfigure the owner")
            if self._active:
                raise RuntimeError(
                    f"{self._entry_point}: cannot reconfigure from inside a forward"
                )
            config_only = not force_rebuild and self._is_cold_config_only()
            if not config_only:
                self._started = True
            try:
                new_config = self._merge_patch(value)
                if config_only and not self._is_cold_config_only():
                    raise RuntimeError("cold configuration crossed into runtime execution")
                if new_config == self._config and not force_rebuild:
                    return self._config
                if self._validate is not None:
                    self._validate(new_config)
                if config_only and not self._is_cold_config_only():
                    raise RuntimeError("cold configuration crossed into runtime execution")
            except BaseException:
                if config_only and not self._is_cold_config_only():
                    self.poison()
                raise

            old_config = self._config
            old_generation = self._generation
            old_commits = self._commits
            next_generation = self._generation + 1
            views = tuple(self._config_views)
            owner_state = None
            view_states = []
            apply_started = False
            try:
                state = vars(self._owner)
                old_owner_config = state.get("_rpu_execution", _MISSING)
                owner_state = state
                for view in views:
                    view_state = vars(view)
                    view_states.append(
                        (view_state, view_state.get("_rpu_execution", _MISSING))
                    )
                if config_only and not self._is_cold_config_only(views):
                    raise RuntimeError("cold configuration crossed into runtime execution")
                if self._apply is not None:
                    apply_started = True
                    if force_rebuild:
                        # A same-config generation alone does not retire child
                        # graphs. Owners must explicitly implement this signal;
                        # an unadopted callback fails closed instead of no-oping.
                        self._apply(old_config, new_config, next_generation, force_rebuild=True)
                    else:
                        self._apply(old_config, new_config, next_generation)
                if config_only and not self._is_cold_config_only(views):
                    raise RuntimeError("cold configuration crossed into runtime execution")

                # Controlled publication deliberately bypasses the A9 cold-write
                # interceptor; arbitrary ``owner._rpu_execution = ...`` remains
                # rejected after first forward.  Keep publication inside the
                # compensation boundary: native apply may already have committed.
                # Owner undo snapshots must survive apply's return. Retaining one
                # until the next apply also covers publication interruptions.
                owner_state["_rpu_execution"] = new_config
                for view_state, _old_view_config in view_states:
                    view_state["_rpu_execution"] = new_config
                self._config = new_config
                self._generation = next_generation
                self._commits = old_commits + 1
                self._clear_prepared_execution_plans()
                if config_only and not self._is_cold_config_only(views):
                    raise RuntimeError("cold configuration crossed into runtime execution")
                return new_config
            except BaseException as apply_error:
                self._failures += 1
                self._clear_prepared_execution_plans()
                crossed_runtime = config_only and not self._is_cold_config_only(views)
                if crossed_runtime:
                    self._poisoned = True
                elif apply_started and self._rollback is None:
                    self._poisoned = True
                elif apply_started:
                    try:
                        self._rollback(old_config, new_config, old_generation)
                    except BaseException as rollback_error:
                        self._poisoned = True
                        if hasattr(apply_error, "add_note"):
                            apply_error.add_note(
                                "execution reconfigure rollback failed: "
                                f"{rollback_error!r}"
                            )
                try:
                    if owner_state is not None:
                        if old_owner_config is _MISSING:
                            owner_state.pop("_rpu_execution", None)
                        else:
                            owner_state["_rpu_execution"] = old_owner_config
                    for view_state, old_view_config in view_states:
                        if old_view_config is _MISSING:
                            view_state.pop("_rpu_execution", None)
                        else:
                            view_state["_rpu_execution"] = old_view_config
                    self._config = old_config
                    self._generation = old_generation
                    self._commits = old_commits
                except BaseException as restore_error:
                    self._poisoned = True
                    if hasattr(apply_error, "add_note"):
                        apply_error.add_note(
                            "execution config mirror restoration failed: "
                            f"{restore_error!r}"
                        )
                if config_only and (self._poisoned or not self._is_cold_config_only(views)):
                    self._started = True
                    self._poisoned = True
                raise

    def _is_cold_config_only(self, views=None) -> bool:
        """Private owner proof; never rewind a used session back to cold."""
        if self._cold_config_only is None or self._cold_config_only() is not True:
            return False
        if views is not None and {id(view) for view in views} != {
                id(view) for view in self._config_views}:
            return False
        return not (
            self._started or self._active or self._closed or self._poisoned
            or self._shutting_down
            or self._planner_costs or self._planner_native
            or self._planner_native_catalogs or self._planner_native_revalidators
            or self._planner_revalidating_native is not None
            or self._planner_cost_observer is not None
            or self._planner_calibration is not None
            or _UNSAFE_PROCESS_REASON is not None
        )

    def shutdown(self, retire=None):
        """Run one exclusive retirement after in-flight execution drains."""
        with self._lock:
            if self._planner_cost_observer is not None:
                raise RuntimeError("cost collection must not close the owner")
            if self._closed:
                if self._poisoned:
                    raise RuntimeError(
                        f"{self._entry_point}: execution session shutdown failed"
                    )
                return None
            _require_execution_process_safe()
            if self._shutting_down:
                return None
            if self._active:
                raise RuntimeError(
                    f"{self._entry_point}: cannot close during a forward"
                )
            self._started = True
            self._shutting_down = True
            try:
                result = None if retire is None else retire()
            except BaseException:
                self._poisoned = True
                self._closed = True
                raise
            else:
                self._closed = True
                self._poisoned = False
                return result
            finally:
                self._shutting_down = False
                self._clear_prepared_execution_plans()

    def close(self) -> None:
        self.shutdown()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "generation": self._generation,
                "graph_mode": self._graph_mode,
                "state": (
                    "POISONED" if self._poisoned else
                    "CLOSED" if self._closed else
                    "EXECUTING" if self._active else "QUIESCENT"
                ),
                "active": self._active,
                "commit_count": self._commits,
                "failure_count": self._failures,
                "rpu_execution": self._config,
            }


def bind_execution_session(
    owner: Any,
    value: Mapping[str, Any] | None,
    *,
    entry_point: str,
    supported: Mapping[str, Collection[str]] | None = None,
    supported_components: Mapping[
        str, Mapping[str, Collection[str]]
    ] | None = None,
    validate=None,
    apply=None,
    rollback=None,
    cold_config_only=None,
    graph_mode: str,
) -> ExecutionSession:
    """Attach or refresh the one execution session owned by ``owner``."""
    _require_execution_process_safe()
    component_capabilities, _component_union = _component_capabilities(
        supported_components, entry_point=entry_point
    )
    session = getattr(owner, _SESSION_ATTR, None)
    if isinstance(session, ExecutionSession) and (
        session._supported_components != component_capabilities
    ):
        raise ValueError(
            f"{entry_point}: registered rpu_execution components changed"
        )
    normalized = normalize_rpu_execution(
        value,
        entry_point=entry_point,
        supported=supported,
        supported_components=component_capabilities,
    )
    if session is None:
        session = ExecutionSession(
            owner,
            normalized,
            entry_point=entry_point,
            supported=supported,
            supported_components=component_capabilities,
            validate=validate,
            apply=apply,
            rollback=rollback,
            cold_config_only=cold_config_only,
            graph_mode=graph_mode,
        )
        setattr(owner, _SESSION_ATTR, session)
    else:
        if not isinstance(session, ExecutionSession):
            raise TypeError(f"{entry_point}: {_SESSION_ATTR} has an invalid type")
        if session.config != normalized:
            raise ValueError(
                f"{entry_point}: execution session conflicts with the existing "
                "model configuration"
            )
        if session.graph_mode != graph_mode:
            raise ValueError(
                f"{entry_point}: execution session graph mode changed from "
                f"{session.graph_mode} to {graph_mode}"
            )
        session.configure_callbacks(
            validate=validate, apply=apply, rollback=rollback,
            cold_config_only=cold_config_only,
        )
    return session


@contextmanager
def execution_guard(owner: Any):
    """Enter an owner's session when it has adopted the common control plane."""
    session = getattr(owner, _SESSION_ATTR, None)
    if session is None:
        yield 0
        return
    with session.execute() as generation:
        yield generation


def _planner_owner_binding(owner: Any, stage: str, component: str | None = None):
    bound = getattr(owner, "_planner_cost_session", None)
    if bound is None:
        session = getattr(owner, _SESSION_ATTR, None)
        if session is None:
            return None, component
        if not isinstance(session, ExecutionSession):
            raise TypeError("planner owner has an invalid execution session")
        if session._supported_components:
            raise ValueError("composite planning requires an explicitly bound child")
        if component not in (None, ""):
            raise ValueError("a leaf planner cannot select another component")
        component = ""
    else:
        if (not isinstance(bound, tuple) or len(bound) != 2 or
                not isinstance(bound[0], weakref.ReferenceType) or
                not isinstance(bound[1], frozenset) or not bound[1]):
            raise TypeError("planner owner has an invalid cost binding")
        session, components = bound[0](), bound[1]
        if not isinstance(session, ExecutionSession):
            raise RuntimeError("planner owner has lost its execution session")
        if component is None:
            if len(components) != 1:
                raise ValueError("multi-component planning requires an explicit component")
            component = next(iter(components))
        if component not in components:
            raise ValueError("planner owner is not bound to the selected component")
    supported = (session._supported_components[component] if session._supported_components
                 else session._supported if session._supported is not None else _STAGES)
    if stage not in supported and not (stage == "decode" and "prefill" in supported):
        raise ValueError("planner cost lookup names an undeclared stage")
    return session, component


def planner_execution_context(owner, stage, component, native):
    """Read one current owner context without caching session authority.

    The native producer admission, costs and active observer/calibration are
    resolved together under the same session lock for each planning request.
    """
    session, component = _planner_owner_binding(owner, stage, component)
    if session is None:
        return None, (), None, None
    with session._lock:
        if native is not None:
            session._record_planner_native_locked(owner, component, stage, native)
        scope, certificates = session._planner_costs_locked(
            component, stage, native[0] if native else "")
        observer = None
        if session._planner_cost_observer is not None:
            if native is None:
                raise ValueError("cost collection requires the actual planner's native binding")
            def observer(kind, value):
                session._planner_cost_observer(owner, component, stage, native, kind, value)
        target = session._planner_calibration
        calibration = None
        if (target is not None and target[0] is owner and target[1:3] == (component, stage)
                and native is not None and native[0] == target[3]):
            calibration = target[4]
        return scope, certificates, observer, calibration


def planner_cost_binding(owner: Any, stage: str, component: str | None = None, *, native_prefix=""):
    """Read only the exact production owner's cold-bound cost context."""
    session, component = _planner_owner_binding(owner, stage, component)
    return (None, ()) if session is None else session.planner_costs(component, stage, native_prefix)


def prepared_execution_plan_cache(owner, stage, component, native_prefix):
    """Keep one-shot preparation with its real owner, never a global handle map."""
    session, component = _planner_owner_binding(owner, stage, component)
    if session is not None:
        return session._prepared_plan_cache(component, stage, native_prefix)
    from rpu_backend.runtime.execution_planner import PreparedExecutionPlans

    caches = vars(owner).setdefault("_prepared_execution_plans", {})
    key = (component, stage, native_prefix)
    if key not in caches:
        caches[key] = PreparedExecutionPlans()
    return caches[key]


def record_planner_native(owner: Any, stage: str, component, native) -> None:
    session, component = _planner_owner_binding(owner, stage, component)
    if session is not None:
        session._record_planner_native(owner, component, stage, native)


def planner_cost_observer(owner: Any, stage: str, component, native):
    session, component = _planner_owner_binding(owner, stage, component)
    if session is None or session._planner_cost_observer is None:
        return None
    if native is None:
        raise ValueError("cost collection requires the actual planner's native binding")
    def observe(kind, value):
        session._planner_cost_observer(owner, component, stage, native, kind, value)
    return observe


def planner_calibration_selection(owner, stage, component, native):
    """Read only this actual producer's active internal P7 selection."""
    session, component = _planner_owner_binding(owner, stage, component)
    if session is None:
        return None
    with session._lock:
        target = session._planner_calibration
        if target is None or target[0] is not owner or target[1:3] != (component, stage):
            return None
        if native is None or native[0] != target[3]:
            return None
        return target[4]


def execution_serialized(method):
    """Guard a bound production entry point without changing its signature."""
    @functools.wraps(method)
    def wrapped(owner, *args, **kwargs):
        with execution_guard(owner):
            return method(owner, *args, **kwargs)
    return wrapped


@contextmanager
def native_execution_reconfigure(ops=None, *, token=None, journal=None):
    """Hold the process-wide native STW claim while controls are staged."""
    _require_execution_process_safe()
    if ops is None:
        import torch
        ops = torch.ops.rpu
    required = (
        "execution_reconfigure_begin",
        "execution_reconfigure_commit",
        "execution_reconfigure_abort",
        "execution_reconfigure_abort_attempt",
    )
    missing = [name for name in required if not hasattr(ops, name)]
    if missing:
        raise RuntimeError(
            "RPU binary lacks native execution-reconfigure op(s): "
            + ", ".join(missing)
        )
    if token is None:
        attempt_token = next(_NATIVE_RECONFIGURE_ATTEMPTS)
        try:
            token = int(ops.execution_reconfigure_begin(attempt_token))
        except BaseException as error:
            try:
                ops.execution_reconfigure_abort_attempt(attempt_token)
            except BaseException as abort_error:
                if hasattr(error, "add_note"):
                    error.add_note(
                        "native reconfigure attempt recovery failed: "
                        f"{abort_error!r}"
                    )
            raise
    else:
        token = int(token)
    try:
        yield token
    except BaseException as error:
        try:
            ops.execution_reconfigure_abort(token)
        except BaseException as abort_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    f"native reconfigure abort failed: {abort_error!r}"
                )
        raise
    # The native call may commit and then raise at the Python boundary. Mark
    # possible mutation BEFORE entering it so the owner compensates or poisons.
    if journal is not None:
        journal["mutation_started"] = True
    try:
        ops.execution_reconfigure_commit(token)
    except BaseException as error:
        try:
            ops.execution_reconfigure_abort(token)
        except BaseException as abort_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    "native reconfigure commit recovery failed: "
                    f"{abort_error!r}"
                )
        raise


def reconfigure_rpu_execution(
    owner: Any,
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Public hot path; construction-time owners without a session fail closed."""
    session = getattr(owner, _SESSION_ATTR, None)
    if not isinstance(session, ExecutionSession):
        raise RuntimeError(
            f"{type(owner).__name__} has no live ExecutionSession; configure "
            "rpu_execution at construction time"
        )
    return session.reconfigure(value)


def rpu_execution_stats(owner: Any) -> dict[str, Any]:
    session = getattr(owner, _SESSION_ATTR, None)
    if not isinstance(session, ExecutionSession):
        raise RuntimeError(f"{type(owner).__name__} has no live ExecutionSession")
    return session.stats()


__all__ = [
    "ExecutionSession",
    "bind_execution_session",
    "bind_rpu_execution",
    "execution_guard",
    "execution_serialized",
    "native_execution_reconfigure",
    "normalize_rpu_execution",
    "reconfigure_rpu_execution",
    "resolve_component_rpu_execution",
    "rpu_execution_stats",
]
