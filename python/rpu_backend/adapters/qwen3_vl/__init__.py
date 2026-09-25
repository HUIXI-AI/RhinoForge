"""Qwen3-VL → rpu_backend adapter.

Public surface (R-Phase 4 complete):
  - `Qwen3VLAdapter` — top-level adapter dispatched by `RPUModelForConditionalGeneration`.
  - `install_qwen3_vl_text_for_rpu` / `install_qwen3_vl_vision_for_rpu` — used
    by the adapter; exported for direct test access (golden harnesses).
  - DeepStack helpers (zero keepalive, dense scatter).

Internal:
  - `patches.py` — idempotent class-level patches for `Qwen3VLTextRMSNorm` and
    `Qwen3VLTextRotaryEmbedding` (loaded at import time).
  - `text.py` / `vision.py` — R-Phase 1-3 installers.

R3 resolution: F6 Dynamo-safety stamping is applied on `vision_model`
(R-Phase 3, vision.py:450-461) only. The top-level
`Qwen3VLForConditionalGeneration` wrapper is NOT stamped here — its forward
is replaced wholesale by `_rpu_qwen3vl_forward` (no HF `hasattr` guards left
inside), mirroring the existing qwen3.py text-only adapter pattern. Vision
+ text decoder + lm_head are each separately Dynamo-safe via their own
install paths; this surface only orchestrates them.
"""

from __future__ import annotations

import gc
import os
import threading
import types
from typing import Any

import torch
import torch.nn as nn

import rpu_backend
from rpu_backend.runtime.log import _LOG
from rpu_backend.api.causal_lm import _claim_live_instance, _release_live_instance
from rpu_backend.api._execution import (
    bind_execution_session,
    bind_rpu_execution,
    execution_serialized,
    native_execution_reconfigure,
    normalize_rpu_execution,
    resolve_component_rpu_execution,
    qwen3_vl_text_core_profile,
)
from rpu_backend.runtime.device import extract_to_device_target, is_rpu_device_target
from rpu_backend.runtime.control import rpu_env_bool
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError
from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime.decoder import (
    _run_causal_decoder_forward,
    _text_decode_execution_plan,
    chunk_policy_key,
    plan_bounded_prefill_execution,
    validate_qwen3_vl_text_semantics,
)
from rpu_backend.runtime.causal_append import continuation_segments, run_continuation_segments
from rpu_backend.runtime.execution_planner import GRAPH_COMPOSITE_CHILD
from rpu_backend.runtime.rope_partial import build_interleaved_mrope_cos_sin
from rpu_backend.runtime.weights import (
    convert_linear_weights_inplace,
    transform_linear_weight,
    validate_decoder_weight_structure, validate_decoder_mlp_padding,
    pad_decoder_mlp_weights,
)
from rpu_backend.runtime.topology import (
    execution_core_count, resolve_decoder_topology, decoder_mlp_intermediate_size,
    require_same_decoder_topology, validate_decoder_cache_topology,
)

from . import patches  # noqa: F401 — idempotent class swaps at import time
from .text import (
    QWEN3_VL_TEXT_ARCH,
    build_mrope_cos_sin_tables,
    get_or_create_zero_keepalive,
    install_qwen3_vl_text_for_rpu,
    make_zero_visual_embeds,
    lookup_causal_decoder,
    scatter_visual_embeds_to_dense,
)
from .vision import (
    QWEN3_VL_VISION_ARCH,
    _is_qwen3_vl_8b_vision_config,
    build_vision_rope_tables,
    install_qwen3_vl_vision_for_rpu,
    enable_qwen3_vl_vision_merger_on_device,
    register_qwen3vl_vision_fused_merger,
    validate_qwen3_vl_vision_core_profile,
)


__all__ = [
    "Qwen3VLAdapter",
    "build_mrope_cos_sin_tables",
    "install_qwen3_vl_text_for_rpu",
    "get_or_create_zero_keepalive",
    "make_zero_visual_embeds",
    "scatter_visual_embeds_to_dense",
    "QWEN3_VL_TEXT_ARCH",
    "build_vision_rope_tables",
    "install_qwen3_vl_vision_for_rpu",
    "enable_qwen3_vl_vision_merger_on_device",
    "register_qwen3vl_vision_fused_merger",
    "QWEN3_VL_VISION_ARCH",
    "uninstall_qwen3_vl_runtime",
]


# Supported text profiles (hidden_size, intermediate_size, num_hidden_layers,
# num_key_value_heads). All Qwen3-VL Instruct text decoders re-use Qwen3
# topologies; the 30B-A3B MoE variant is out-of-envelope.
_QWEN3_VL_4B_TEXT_PROFILE = (2560, 9728, 36, 8)
_QWEN3_VL_8B_TEXT_PROFILE = (4096, 12288, 36, 8)
_SUPPORTED_TEXT_PROFILES: frozenset[tuple[int, int, int, int]] = frozenset({
    (2048,  6144, 28,  8),  # Qwen3-VL-2B-Instruct
    _QWEN3_VL_4B_TEXT_PROFILE,
    _QWEN3_VL_8B_TEXT_PROFILE,
})

_SUPPORTED_TEXT_DETAILS = {
    (2048, 6144, 28, 8): (16, 128),
    (2560, 9728, 36, 8): (32, 128),
    _QWEN3_VL_8B_TEXT_PROFILE: (32, 128),
}
_MROPE_STABLE_SLOT_LIMIT = 128
_QWEN3_VL_8B_TEXT_SEMANTICS = (
    "qwen3_vl_text", "silu", 1e-6, 151936, False, True,
    "default", 5_000_000.0, True, (24, 20, 20), False,
)
_SUPPORTED_VISION_PROFILE = (
    1024,  # hidden_size
    4096,  # intermediate_size
    24,    # depth
    16,    # num_heads
    16,    # patch_size
    2,     # temporal_patch_size
    2,     # spatial_merge_size
    (5, 11, 17),  # deepstack_visual_indexes
    "gelu_pytorch_tanh",
)
_QWEN3_VL_8B_VISION_PROFILE = (
    1152,  # hidden_size
    4304,  # logical intermediate_size; physically padded to 4352
    27,    # depth
    16,    # num_heads; logical head_dim 72 is physically padded to 80
    16,    # patch_size
    2,     # temporal_patch_size
    2,     # spatial_merge_size
    (8, 16, 24),  # deepstack_visual_indexes
    "gelu_pytorch_tanh",
)
_QWEN3_VL_8B_VISION_SEMANTICS = ("qwen3_vl", 3, 2304)
# Qwen3-VL-32B W8A16 is integrated only as a controlled-evaluation path.  The
# source MR records a retained-Graph defect where the third and later decode
# logits become exactly zero.  Keep it outside the supported profile registry
# and require an exact opt-in before model loading or any irreversible mutation.
_QWEN3_VL_32B_GRAPH_BLOCKED_ENV = "QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED"
_QWEN3_VL_32B_TEXT_PROFILE = (5120, 25600, 64, 8)
# The full runtime-W8 weights exceed the SDK's instruction-address window.
# Reserve low-address arenas before weights, bounded by all three caches plus
# one serialized transient queue. Eight entries per cache keeps the owner plan
# within the board's mapping budget; capacity misses retain fail-closed behavior.
_QWEN3_VL_32B_RUNTIME_W8_GRAPH_ENTRIES = 8
_QWEN3_VL_TEXT_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


# Module-level swizzle lock (mirrors Qwen3Adapter pattern).
_SWIZZLE_LOCK = threading.Lock()

_QWEN3_VL_PREFILL_PADDING_BUDGET = 64
_QWEN3_VL_MROPE_KEEPALIVE_ROWS = 8192
_QWEN3_VL_TEXT_COMPONENT = "language_model"
_QWEN3_VL_VISION_COMPONENT = "vision_encoder"
_QWEN3_VL_EXECUTION_COMPONENTS = {
    _QWEN3_VL_TEXT_COMPONENT: {
        "prefill": (
            "chunk_size", "padding_rows", "padding_budget", "linear_acc32",
            "fast_replay",
        ),
    },
    _QWEN3_VL_VISION_COMPONENT: {
        "vision": ("chunk_size", "linear_acc32"),
    },
}
_QWEN3_VL_EXECUTION_SUPPORTED = {
    "model": ("num_cores",),
    "prefill": (
        "chunk_size", "padding_rows", "padding_budget", "linear_acc32",
        "fast_replay",
    ),
    "vision": ("chunk_size", "linear_acc32"),
}


def _qwen3_vl_linear_acc32(execution, stage: str, default: bool = False) -> bool:
    """Resolve the cold per-stage Linear accumulation policy.

    The public schema keeps this as an explicit boolean when callers want to
    choose ACC16 (``False``) or ACC32 (``True``).  An omitted field selects ACC16 for every weight precision.
    Historical ACC32 numerical profiles must request it explicitly.
    """
    value = execution.get(stage, {}).get("linear_acc32", default)
    if not isinstance(value, bool):
        raise TypeError(
            f"Qwen3-VL {stage}.linear_acc32 must be bool, got {value!r}"
        )
    return value


def _resolve_qwen3_vl_execution(value, *, entry_point: str):
    root = normalize_rpu_execution(
        value,
        entry_point=entry_point,
        supported_components=_QWEN3_VL_EXECUTION_COMPONENTS,
        supported=_QWEN3_VL_EXECUTION_SUPPORTED,
    )
    text = resolve_component_rpu_execution(
        root,
        _QWEN3_VL_TEXT_COMPONENT,
        entry_point=entry_point,
        supported_components=_QWEN3_VL_EXECUTION_COMPONENTS,
        supported=_QWEN3_VL_EXECUTION_SUPPORTED,
        profile_auto={"prefill": {"chunk_size": "auto"}},
    )
    vision = resolve_component_rpu_execution(
        root,
        _QWEN3_VL_VISION_COMPONENT,
        entry_point=entry_point,
        supported_components=_QWEN3_VL_EXECUTION_COMPONENTS,
        supported=_QWEN3_VL_EXECUTION_SUPPORTED,
        profile_auto={"vision": {"chunk_size": "auto"}},
    )
    cores = execution_core_count(root)
    if cores != 8:
        # The public budget belongs to the composite. Its finite child layouts
        # are derived here; component overrides cannot choose different cores.
        text = normalize_rpu_execution(
            {**text, "model": {"num_cores": cores}}, entry_point=entry_point,
            supported=_QWEN3_VL_EXECUTION_SUPPORTED)
        vision = normalize_rpu_execution(
            {**vision, "model": {"num_cores": 4}}, entry_point=entry_point,
            supported=_QWEN3_VL_EXECUTION_SUPPORTED)
    return root, text, vision
_QWEN3_VL_2B_TEXT_PROFILE = (2048, 6144, 28, 8)


def _quantize_pack_lm_head_w4_groupwise(
    weight: torch.Tensor, group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack the VL head with the shared bounded TP8/G32 physical ABI."""
    from rpu_backend.quant.int4_pgrp_pack import quantize_pack_int4_pgrp_col_bounded

    return quantize_pack_int4_pgrp_col_bounded(weight, group_size)


_QWEN3_VL_DELIVERY_PROFILE = "three-frame-hybrid-w8-w4-top2-v1"
_QWEN3_VL_DELIVERY_BLOCKED_ENV = "QWEN3_VL_DELIVERY_ALLOW_NUMERIC_BLOCKED"


def _require_delivery_opt_in() -> None:
    if os.environ.get(_QWEN3_VL_DELIVERY_BLOCKED_ENV) != "1":
        raise UnsupportedModelError(
            "Qwen3-VL fixed delivery is numeric-blocked and hardware-unverified "
            "on the current runtime. Controlled evaluation requires exact "
            f"{_QWEN3_VL_DELIVERY_BLOCKED_ENV}=1 before loading or installation."
        )


def _validate_delivery_config(config) -> None:
    """The historical workload has its own semantics, never the FP16 profile."""
    _require_delivery_opt_in()
    _check_profile(config)
    text, vision = config.text_config, config.vision_config
    rope = getattr(text, "rope_parameters", None) or getattr(text, "rope_scaling", None) or {}
    semantics = (
        getattr(text, "model_type", None), getattr(text, "hidden_act", None),
        getattr(text, "rms_norm_eps", None), getattr(text, "attention_bias", None),
        getattr(text, "use_cache", None), getattr(text, "vocab_size", None),
        rope.get("rope_type"), rope.get("rope_theta", getattr(text, "rope_theta", None)),
        rope.get("mrope_interleaved"), tuple(rope.get("mrope_section", ())),
        getattr(config, "tie_word_embeddings", None),
        getattr(text, "tie_word_embeddings", None),
        getattr(vision, "in_channels", None), getattr(vision, "num_position_embeddings", None),
        getattr(config, "image_token_id", None), getattr(config, "video_token_id", None),
        getattr(config, "vision_start_token_id", None), getattr(config, "vision_end_token_id", None),
    )
    if (config.architectures != ["Qwen3VLForConditionalGeneration"]
            or _text_config_profile(text) != _QWEN3_VL_2B_TEXT_PROFILE
            or semantics != ("qwen3_vl_text", "silu", 1e-6, False, True, 151936,
                             "default", 5_000_000.0, True, (24, 20, 20), True, True,
                             3, 2304, 151655, 151656, 151652, 151653)
            or any(getattr(cfg, name, None) is not None
                   for cfg in (config, text, vision)
                   for name in ("quantization_config", "quant_config"))):
        raise UnsupportedModelError("fixed delivery requires exact unquantized Qwen3-VL-2B config and semantics")


def _delivery_profile_active(model) -> bool:
    stamp = getattr(model, "_qwen3_vl_delivery_profile", None)
    if stamp is None:
        return False
    if stamp != _QWEN3_VL_DELIVERY_PROFILE:
        raise UnsupportedModelError("unknown Qwen3-VL delivery profile")
    _validate_delivery_config(model.config)
    return True


def _normalize_delivery_execution(value=None):
    fixed = {"prefill": {"chunk_size": 192, "padding_rows": 0},
             "vision": {"chunk_size": 576}}
    root, text, vision = _resolve_qwen3_vl_execution(
        fixed if value is None else value, entry_point="Qwen3VL fixed delivery")
    _, expected_text, expected_vision = _resolve_qwen3_vl_execution(
        fixed, entry_point="Qwen3VL fixed delivery")
    if text != expected_text or vision != expected_vision:
        raise UnsupportedModelError("fixed delivery requires text C192 without padding and vision C576")
    return root


def _validate_delivery_weights(model) -> bool:
    if not _delivery_profile_active(model):
        return False
    text = model.model.language_model
    if len(text.layers) != 28:
        raise RPUBackendError("fixed delivery requires all 28 text layers")
    expected_shapes = {"self_attn.q_proj": (2048, 2048), "self_attn.k_proj": (1024, 2048),
                       "self_attn.v_proj": (1024, 2048), "self_attn.o_proj": (2048, 2048),
                       "mlp.gate_proj": (6144, 2048), "mlp.up_proj": (6144, 2048),
                       "mlp.down_proj": (2048, 6144)}
    for index, layer in enumerate(text.layers):
        for name in _QWEN3_VL_TEXT_PROJECTIONS:
            projection = layer.get_submodule(name)
            weight, scale = projection.weight, getattr(projection, "weight_scale", None)
            quantized = index < 27 or name.startswith("mlp.")
            expected_dtype = torch.int8 if quantized else torch.float16
            if (weight.dtype != expected_dtype or weight.device.type != "cpu"
                    or not weight.is_contiguous() or weight.ndim != 2
                    or tuple(weight.shape) != expected_shapes[name]
                    or not isinstance(scale, torch.Tensor) or scale.dtype != torch.float16
                    or scale.device.type != "cpu" or not scale.is_contiguous()
                    or tuple(scale.shape) != (weight.shape[0],)
                    or not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all())):
                raise RPUBackendError(f"fixed delivery weight/scale mismatch: layer {index}.{name}")
    return True


def _lm_head_optimization_config(model) -> tuple[bool, int, bool, int]:
    """A typed loader profile fixes these choices once before swizzle."""
    retired = ("RPU_QWEN3VL_DECODE_LM_HEAD_W4A16", "RPU_QWEN3VL_DECODE_LM_HEAD_EXACT_TOPK",
               "RPU_QWEN3VL_PREFILL_LAST_LM_HEAD_W4A16", "RPU_QWEN3VL_REUSE_LM_HEAD_OUTPUT")
    if any(os.environ.get(name, "0").lower() not in ("", "0", "false", "off") for name in retired):
        raise UnsupportedModelError("retired Qwen3-VL route environment settings must be removed; use load_delivery_model with its explicit controlled opt-in")
    if _delivery_profile_active(model):
        return True, 2, True, 32
    return False, 0, False, 32


def _text_config_profile(text_config) -> tuple[int, int, int, int]:
    return (
        int(text_config.hidden_size),
        int(text_config.intermediate_size),
        int(text_config.num_hidden_layers),
        int(text_config.num_key_value_heads),
    )


def _qwen3_vl_has_quant_metadata(config) -> bool:
    """Reject nested quant metadata as well as the top-level aliases."""
    parts = (
        config,
        getattr(config, "text_config", None),
        getattr(config, "vision_config", None),
    )
    return any(
        getattr(part, name, None) is not None
        for part in parts
        if part is not None
        for name in ("quant_config", "quantization_config")
    )


def _is_qwen3_vl_dense_eager_profile(model) -> bool:
    """Match the validated ordinary dense FP16 2B/4B eager profiles.

    Existing profile admission still runs before any irreversible mutation.
    This predicate only selects acceleration; it adds no new support envelope.
    """
    if getattr(model, "_qwen3_vl_delivery_profile", None) is not None:
        return False
    try:
        config = model.config
        text, vision = config.text_config, config.vision_config
        profile = _text_config_profile(text)
        if (config.model_type != "qwen3_vl"
                or config.architectures != ["Qwen3VLForConditionalGeneration"]
                or profile not in (
                    _QWEN3_VL_2B_TEXT_PROFILE,
                    _QWEN3_VL_4B_TEXT_PROFILE,
                )
                or _qwen3_vl_has_quant_metadata(config)):
            return False
        rope = getattr(text, "rope_parameters", None) or getattr(text, "rope_scaling", None) or {}
        return (
            (text.num_attention_heads, text.head_dim)
            == _SUPPORTED_TEXT_DETAILS[profile]
            and (text.model_type, text.hidden_act, text.rms_norm_eps, text.vocab_size,
             text.attention_bias, text.use_cache, config.tie_word_embeddings,
             text.tie_word_embeddings, rope.get("rope_type"),
             rope.get("rope_theta", getattr(text, "rope_theta", None)),
             rope.get("mrope_interleaved"), tuple(rope.get("mrope_section", ())))
            == ("qwen3_vl_text", "silu", 1e-6, 151936, False, True,
                True, True, "default", 5_000_000, True, (24, 20, 20))
            and (vision.hidden_size, vision.intermediate_size, vision.depth,
                 vision.num_heads, vision.patch_size, vision.temporal_patch_size,
                 vision.spatial_merge_size, tuple(vision.deepstack_visual_indexes),
                 vision.hidden_act) == _SUPPORTED_VISION_PROFILE
            and (vision.model_type, vision.in_channels, vision.num_position_embeddings,
                 vision.out_hidden_size) == ("qwen3_vl", 3, 2304, profile[0])
            and (config.image_token_id, config.video_token_id,
                 config.vision_start_token_id, config.vision_end_token_id)
            == (151655, 151656, 151652, 151653)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _is_qwen3_vl_2b_eager_profile(model) -> bool:
    """Compatibility helper retained for existing internal diagnostics."""
    try:
        return (
            _text_config_profile(model.config.text_config)
            == _QWEN3_VL_2B_TEXT_PROFILE
            and _is_qwen3_vl_dense_eager_profile(model)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _is_qwen3_vl_32b_w8a16_config(config) -> bool:
    """Return whether *config* is the one admitted 32B controlled profile."""
    # Keep config admission single-sourced with the CPU-only loader so API
    # preflight, direct adapter construction, and checkpoint loading cannot
    # drift on quant metadata or nested geometry.
    from rpu_backend.quant.load import is_qwen3_vl_32b_w8a16_config

    return is_qwen3_vl_32b_w8a16_config(config)


def _is_qwen3_vl_4b_w8a16_config(config) -> bool:
    from rpu_backend.quant.load import is_qwen3_vl_4b_w8a16_config

    return is_qwen3_vl_4b_w8a16_config(config)


def _is_qwen3_vl_awq_config(config) -> bool:
    from rpu_backend.quant.load import is_qwen3_vl_awq_config

    return is_qwen3_vl_awq_config(config)


def _is_qwen3_vl_runtime_quant_config(config) -> bool:
    from rpu_backend.quant.load_qwen3_vl_runtime import is_qwen3_vl_runtime_quant_config

    return is_qwen3_vl_runtime_quant_config(config)


def _check_profile(config) -> None:
    """Validate the admitted Qwen3-VL profiles before weight mutation."""
    if getattr(config, "model_type", None) != "qwen3_vl":
        raise UnsupportedModelError(
            f"Qwen3VLAdapter: expected config.model_type='qwen3_vl', got "
            f"{getattr(config, 'model_type', None)!r}."
        )
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is None:
        raise UnsupportedModelError(
            "Qwen3VLAdapter: config.text_config missing."
        )
    try:
        profile = _text_config_profile(text_cfg)
    except (AttributeError, TypeError, ValueError) as exc:
        raise UnsupportedModelError(
            "Qwen3VLAdapter: incomplete or malformed Qwen3-VL text config."
        ) from exc
    is_graph_blocked_32b = _is_qwen3_vl_32b_w8a16_config(config)
    runtime_quant = _is_qwen3_vl_runtime_quant_config(config)
    runtime_quant_32b = runtime_quant and profile == _QWEN3_VL_32B_TEXT_PROFILE
    awq_config = getattr(config, "quantization_config", None)
    if (isinstance(awq_config, dict)
            and awq_config.get("quant_method") == "compressed-tensors"
            and not _is_qwen3_vl_awq_config(config)):
        raise UnsupportedModelError(
            "Qwen3VLAdapter: compressed-tensors requires the exact 2B/4B "
            "Instruct symmetric Text-W4 G32 packed checkpoint profile."
        )
    quant_config = getattr(config, "quant_config", None)
    if (isinstance(quant_config, dict)
            and quant_config.get("method") == "w4a16"
            and not _is_qwen3_vl_runtime_quant_config(config)):
        raise UnsupportedModelError(
            "Qwen3VLAdapter: W4A16 requires an exact registered runtime-alignment "
            "Text7/head signed-W4 G32 checkpoint profile."
        )
    if (isinstance(quant_config, dict)
            and str(quant_config.get("profile", "")).startswith("qwen3_vl_")
            and str(quant_config.get("profile", "")).endswith("_runtime_align_v1")
            and not (_is_qwen3_vl_4b_w8a16_config(config)
                     or _is_qwen3_vl_runtime_quant_config(config))):
        raise UnsupportedModelError(
            "Qwen3VLAdapter: quantized config must match an explicit controlled "
            "runtime-alignment or legacy 32B W8A16 profile."
        )
    if is_graph_blocked_32b:
        if os.environ.get(_QWEN3_VL_32B_GRAPH_BLOCKED_ENV) != "1":
            raise UnsupportedModelError(
                "Qwen3VLAdapter: the exact Qwen3-VL-32B W8A16 profile is "
                "graph-blocked: retained Graph replay produces all-zero logits "
                "from the third decode step in the source MR. It is not a "
                "supported profile. Set exact "
                f"{_QWEN3_VL_32B_GRAPH_BLOCKED_ENV}=1 only for controlled "
                "evaluation; clearing the graph per token is not an accepted "
                "runtime workaround."
            )
    elif profile == _QWEN3_VL_32B_TEXT_PROFILE and not runtime_quant_32b:
        raise UnsupportedModelError(
            "Qwen3VLAdapter: only the exact Qwen3-VL-32B W8A16 config may enter "
            "the graph-blocked controlled-evaluation path; architecture, text "
            "geometry, logical vision geometry, and quant method must all match."
        )
    elif profile not in _SUPPORTED_TEXT_PROFILES and not runtime_quant_32b:
        raise UnsupportedModelError(
            f"Qwen3VLAdapter: text profile {profile} not supported. "
            f"Supported: {sorted(_SUPPORTED_TEXT_PROFILES)} "
            "(dense Instruct profiles only; 30B-A3B MoE is out-of-envelope)."
        )
    if (
        profile == _QWEN3_VL_8B_TEXT_PROFILE
        and _qwen3_vl_has_quant_metadata(config)
        and not runtime_quant
    ):
        raise UnsupportedModelError(
            "Qwen3VLAdapter: the exact Qwen3-VL-8B padded profile requires "
            "an FP16 checkpoint or a registered runtime quantization profile."
        )
    vision_cfg = getattr(config, "vision_config", None)
    if vision_cfg is None:
        raise UnsupportedModelError(
            "Qwen3VLAdapter: config.vision_config missing."
        )
    try:
        text_details = (
            int(text_cfg.num_attention_heads),
            int(text_cfg.head_dim),
        )
        text_semantics = None
        if profile == _QWEN3_VL_8B_TEXT_PROFILE:
            rope_params = getattr(text_cfg, "rope_parameters", None)
            if rope_params is None:
                rope_params = getattr(text_cfg, "rope_scaling", None)
            rope_params = rope_params or {}
            text_semantics = (
                text_cfg.model_type,
                text_cfg.hidden_act,
                float(text_cfg.rms_norm_eps),
                int(text_cfg.vocab_size),
                text_cfg.attention_bias,
                text_cfg.use_cache,
                rope_params.get("rope_type"),
                float(rope_params.get("rope_theta")),
                rope_params.get("mrope_interleaved"),
                tuple(int(x) for x in rope_params.get("mrope_section", ())),
                getattr(config, "tie_word_embeddings", None),
            )
        vision_profile = (
            int(vision_cfg.hidden_size),
            int(vision_cfg.intermediate_size),
            int(vision_cfg.depth),
            int(vision_cfg.num_heads),
            int(vision_cfg.patch_size),
            int(vision_cfg.temporal_patch_size),
            int(vision_cfg.spatial_merge_size),
            tuple(int(x) for x in vision_cfg.deepstack_visual_indexes),
            vision_cfg.hidden_act,
        )
        vision_semantics = None
        if profile == _QWEN3_VL_8B_TEXT_PROFILE:
            vision_semantics = (
                vision_cfg.model_type,
                int(vision_cfg.in_channels),
                int(vision_cfg.num_position_embeddings),
            )
        vision_out_hidden = int(vision_cfg.out_hidden_size)
    except (AttributeError, TypeError, ValueError) as exc:
        raise UnsupportedModelError(
            "Qwen3VLAdapter: incomplete or malformed Qwen3-VL config."
        ) from exc
    if is_graph_blocked_32b or runtime_quant_32b:
        # The exact nested geometry was checked by the controlled-profile
        # predicate; it intentionally stays outside the certified 2B/4B table.
        return
    expected_text_details = _SUPPORTED_TEXT_DETAILS[profile]
    if text_details != expected_text_details:
        raise UnsupportedModelError(
            "Qwen3VLAdapter: unsupported text attention geometry: "
            f"expected {expected_text_details}, got {text_details}."
        )
    if (
        profile == _QWEN3_VL_8B_TEXT_PROFILE
        and text_semantics != _QWEN3_VL_8B_TEXT_SEMANTICS
    ):
        raise UnsupportedModelError(
            "Qwen3VLAdapter: unsupported Qwen3-VL-8B text semantics; "
            f"expected {_QWEN3_VL_8B_TEXT_SEMANTICS}, got {text_semantics}."
        )
    expected_vision_profile = (
        _QWEN3_VL_8B_VISION_PROFILE
        if profile == _QWEN3_VL_8B_TEXT_PROFILE
        else _SUPPORTED_VISION_PROFILE
    )
    if (vision_profile != expected_vision_profile
            or vision_out_hidden != profile[0]):
        raise UnsupportedModelError(
            "Qwen3VLAdapter: unsupported vision profile for the admitted "
            f"text profile {profile}; expected {expected_vision_profile} "
            f"with out_hidden_size={profile[0]}, "
            f"got {vision_profile} with out_hidden_size={vision_out_hidden}."
        )
    if (
        profile == _QWEN3_VL_8B_TEXT_PROFILE
        and vision_semantics != _QWEN3_VL_8B_VISION_SEMANTICS
    ):
        raise UnsupportedModelError(
            "Qwen3VLAdapter: unsupported Qwen3-VL-8B vision semantics; "
            f"expected {_QWEN3_VL_8B_VISION_SEMANTICS}, got "
            f"{vision_semantics}."
        )


def _qwen3_vl_core_topology(config, execution_config):
    cores = execution_core_count(execution_config)
    if cores == 8:
        return None
    _check_profile(config)
    profile = qwen3_vl_text_core_profile(config.text_config, cores)
    vision = config.vision_config
    from rpu_backend.quant.load import is_qwen3_vl_2b_w8a16_config
    text_w8 = is_qwen3_vl_2b_w8a16_config(config)
    if (getattr(config, "architectures", None) != ["Qwen3VLForConditionalGeneration"]
            or (not text_w8 and any(getattr(cfg, key, None) is not None
                   for cfg in (config, config.text_config, vision)
                   for key in ("quant_config", "quantization_config")))
            or (vision.model_type, vision.in_channels, vision.num_position_embeddings)
                != ("qwen3_vl", 3, 2304)
            or tuple(getattr(config, key, None) for key in (
                "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"))
                != (151655, 151656, 151652, 151653)
            or getattr(config, "tie_word_embeddings", None) is not (profile.hidden_size != 4096)):
        raise UnsupportedModelError("reduced-core Qwen3-VL requires exact FP16 Instruct or 2B text-only W8A16 config")
    return resolve_decoder_topology(num_cores=cores, **profile.geometry())


def _qwen3_vl_decoder_weight_view(model):
    """Reuse the decoder validators/converter without traversing Vision."""
    view = nn.Module()
    view.model = model.model.language_model
    view.lm_head = model.lm_head
    view.config = model.config.text_config
    return view


def _validate_qwen3_vl_reduced_weights(model, topology):
    if topology is None:
        return
    if getattr(model, "_qwen3_vl_delivery_profile", None) is not None:
        raise UnsupportedModelError("Qwen3-VL delivery profiles require 8 cores")
    profile = qwen3_vl_text_core_profile(model.config.text_config, topology.num_cores)
    actual = qwen3_vl_text_core_profile(model.model.language_model.config, topology.num_cores)
    if actual != profile:
        raise ValueError("Qwen3-VL loaded text geometry differs from the root config")
    if _validate_qwen3_vl_2b_w8a16_model(model):
        return
    view = _qwen3_vl_decoder_weight_view(model)
    validate_decoder_weight_structure(view, **profile.weight_geometry())
    validate_qwen3_vl_text_semantics(model.model.language_model)
    validate_decoder_mlp_padding(view, logical_size=profile.intermediate_size,
        physical_size=decoder_mlp_intermediate_size(profile.intermediate_size, topology.mlp_tp))
    for name, parameter in model.named_parameters():
        if (parameter.device.type != "cpu" or parameter.dtype != torch.float16
                or not parameter.is_contiguous()):
            raise ValueError(f"reduced-core Qwen3-VL requires original contiguous CPU FP16 weights: {name}")
    for name, buffer in model.named_buffers():
        if buffer.device.type != "cpu":
            raise ValueError(f"reduced-core Qwen3-VL requires materialized CPU buffers: {name}")
    for name, module in view.named_modules():
        if any(getattr(module, attr, None) is not None for attr in (
                "weight_scale", "weight_scale_inv", "_rpu_linear_partition", "qweight")):
            raise ValueError(f"reduced-core Qwen3-VL requires unswizzled FP16 text: {name}")
    validate_qwen3_vl_vision_core_profile(model.model.visual,
        num_cores=4, vision_config=model.config.vision_config)


def _validate_qwen3_vl_2b_w8a16_model(model) -> bool:
    """Validate every original text/Vision leaf before ownership or swizzle."""
    from rpu_backend.quant.load import is_qwen3_vl_2b_w8a16_config

    if not is_qwen3_vl_2b_w8a16_config(model.config):
        return False
    text = model.model.language_model
    profile = qwen3_vl_text_core_profile(model.config.text_config, 4)
    if qwen3_vl_text_core_profile(text.config, 4) != profile:
        raise ValueError("Qwen3-VL W8A16 loaded text geometry differs from root config")
    view = _qwen3_vl_decoder_weight_view(model)
    validate_decoder_weight_structure(
        view, **profile.weight_geometry(), projection_dtype=torch.int8)
    validate_qwen3_vl_text_semantics(text)
    projections = {
        f"model.language_model.layers.{index}.{name}"
        for index in range(profile.num_layers)
        for name in _QWEN3_VL_TEXT_PROJECTIONS
    }
    for name, parameter in model.named_parameters():
        expected = torch.int8 if name.removesuffix(".weight") in projections else torch.float16
        if (parameter.device.type != "cpu" or parameter.dtype != expected
                or not parameter.is_contiguous()):
            raise ValueError(f"Qwen3-VL-2B W8A16 requires original contiguous CPU {expected} tensor: {name}")
    for name, module in model.named_modules():
        scale = getattr(module, "weight_scale", None)
        if name in projections:
            if (not isinstance(scale, torch.Tensor) or scale.dtype != torch.float16
                    or scale.device.type != "cpu" or not scale.is_contiguous()
                    or tuple(scale.shape) != (module.out_features,)):
                raise ValueError(f"Qwen3-VL-2B W8A16 requires CPU FP16 per-channel scale: {name}")
        elif scale is not None:
            raise ValueError(f"Qwen3-VL-2B W8A16 does not quantize this module: {name}")
        if any(getattr(module, attr, None) is not None for attr in (
                "weight_scale_inv", "_rpu_linear_partition", "qweight")):
            raise ValueError(f"Qwen3-VL-2B W8A16 requires original unswizzled text/Vision: {name}")
    for name, buffer in model.named_buffers():
        if buffer.device.type != "cpu":
            raise ValueError(f"Qwen3-VL-2B W8A16 requires materialized CPU buffer: {name}")
    if model.lm_head.weight is not text.embed_tokens.weight:
        raise ValueError("Qwen3-VL-2B W8A16 requires its original tied FP16 embedding/head")
    # This validator checks the same complete unpadded 2B Vision inventory for
    # either budget; it does not install or choose the actual Vision topology.
    validate_qwen3_vl_vision_core_profile(model.model.visual,
        num_cores=4, vision_config=model.config.vision_config)
    return True


def _validate_qwen3_vl_32b_w8a16_model(model) -> bool:
    """Validate the loaded 32B tensor inventory before RPU ownership/mutation."""
    if not _is_qwen3_vl_32b_w8a16_config(model.config):
        return False

    try:
        text_model = model.model.language_model
        layers = text_model.layers
        embed_tokens = text_model.embed_tokens
        lm_head = model.lm_head
    except AttributeError as exc:
        raise RPUBackendError(
            "Qwen3-VL-32B W8A16 validation failed: incomplete model hierarchy."
        ) from exc
    if len(layers) != 64:
        raise RPUBackendError(
            f"Qwen3-VL-32B W8A16 validation failed: layers={len(layers)}, expected 64."
        )

    expected_shapes = {
        "self_attn.q_proj": (64 * 128, 5120),
        "self_attn.k_proj": (8 * 128, 5120),
        "self_attn.v_proj": (8 * 128, 5120),
        "self_attn.o_proj": (5120, 64 * 128),
        "mlp.gate_proj": (25600, 5120),
        "mlp.up_proj": (25600, 5120),
        "mlp.down_proj": (5120, 25600),
    }
    for layer_idx, layer in enumerate(layers):
        for projection_name in _QWEN3_VL_TEXT_PROJECTIONS:
            try:
                projection = layer.get_submodule(projection_name)
            except (AttributeError, KeyError) as exc:
                raise RPUBackendError(
                    "Qwen3-VL-32B W8A16 validation failed: missing projection "
                    f"layer {layer_idx}.{projection_name}."
                ) from exc
            weight = getattr(projection, "weight", None)
            scale = getattr(projection, "weight_scale", None)
            expected = expected_shapes[projection_name]
            if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != expected:
                raise RPUBackendError(
                    "Qwen3-VL-32B W8A16 validation failed: "
                    f"layer {layer_idx}.{projection_name}.weight must be {expected}."
                )
            if (
                weight.dtype != torch.int8
                or weight.device.type != "cpu"
                or not weight.is_contiguous()
            ):
                raise RPUBackendError(
                    "Qwen3-VL-32B W8A16 validation failed: "
                    f"layer {layer_idx}.{projection_name}.weight must be contiguous CPU int8."
                )
            if (
                not isinstance(scale, torch.Tensor)
                or scale.dtype != torch.float16
                or scale.device.type != "cpu"
                or not scale.is_contiguous()
                or tuple(scale.shape) != (expected[0],)
            ):
                raise RPUBackendError(
                    "Qwen3-VL-32B W8A16 validation failed: "
                    f"layer {layer_idx}.{projection_name}.weight_scale must be "
                    f"contiguous CPU fp16 [{expected[0]}]."
                )

    for name, module in (("model.embed_tokens", embed_tokens), ("lm_head", lm_head)):
        weight = getattr(module, "weight", None)
        if (
            not isinstance(weight, torch.Tensor)
            or tuple(weight.shape) != (151936, 5120)
            or not weight.is_floating_point()
            or weight.dtype != torch.float16
            or weight.device.type != "cpu"
            or not weight.is_contiguous()
        ):
            raise RPUBackendError(
                "Qwen3-VL-32B W8A16 validation failed: "
                f"{name}.weight must be contiguous CPU fp16."
            )

    for name, parameter in model.named_parameters():
        if parameter.is_meta or parameter.device.type != "cpu":
            raise RPUBackendError(
                "Qwen3-VL-32B W8A16 validation failed before RPU ownership: "
                f"parameter {name!r} must be materialized on CPU."
            )
        is_projection = name.endswith(tuple(
            f".{projection}.weight"
            for projection in _QWEN3_VL_TEXT_PROJECTIONS
        ))
        expected_dtype = torch.int8 if is_projection else torch.float16
        if parameter.dtype != expected_dtype:
            raise RPUBackendError(
                "Qwen3-VL-32B W8A16 validation failed: parameter "
                f"{name!r} must be {expected_dtype}, got {parameter.dtype}."
            )
    for name, buffer in model.named_buffers():
        if buffer.is_meta or buffer.device.type != "cpu":
            raise RPUBackendError(
                "Qwen3-VL-32B W8A16 validation failed before RPU ownership: "
                f"buffer {name!r} must be materialized on CPU."
            )
    return True


def _qwen3_vl_text_scale_lists(text_model):
    layers = text_model.layers
    return tuple(
        [layer.get_submodule(name).weight_scale for layer in layers]
        for name in _QWEN3_VL_TEXT_PROJECTIONS
    )


def _apply_prefill_lm_head(model, hidden, slice_indices):
    """Preserve requested-position logits at the existing eager head boundary."""
    head_input = hidden[:, slice_indices, :]
    head = model.lm_head
    weight = getattr(head, "weight", None)
    # Small host tests and adapter embeddings may provide a callable head
    # without a materialized module weight; retain the eager boundary there.
    if weight is None:
        return head(head_input)
    text = model.model.language_model
    cores = getattr(head, "_rpu_linear_num_cores", 8)
    scale = getattr(head, "weight_scale", None) if weight.dtype in (torch.int8, torch.uint8) else None
    return torch.ops.rpu.linear_with_accumulation(
        head_input.contiguous(), weight, getattr(head, "bias", None), 1, cores,
        getattr(text, "_rpu_execution", {}).get("prefill", {}).get("linear_acc32", False), scale,
    )


def _validate_qwen3_vl_4b_w8a16_model(model) -> bool:
    """Check the materialized controlled profile before any weight mutation."""
    if not _is_qwen3_vl_4b_w8a16_config(model.config):
        return False
    text = model.model.language_model
    if len(text.layers) != 36:
        raise RPUBackendError("4B runtime-alignment W8 requires all 36 text layers")
    shapes = ((4096, 2560), (1024, 2560), (1024, 2560), (2560, 4096),
              (9728, 2560), (9728, 2560), (2560, 9728))
    quant_modules = [(f"layer {i}.{name}", layer.get_submodule(name), shape)
                     for i, layer in enumerate(text.layers)
                     for name, shape in zip(_QWEN3_VL_TEXT_PROJECTIONS, shapes, strict=True)]
    quant_modules.append(("lm_head", model.lm_head, (151936, 2560)))
    for name, module, shape in quant_modules:
        weight = module.weight
        scale = getattr(module, "weight_scale", None)
        if (weight.dtype != torch.int8 or tuple(weight.shape) != shape
                or weight.device.type != "cpu" or not weight.is_contiguous()
                or not isinstance(scale, torch.Tensor)
                or scale.dtype != torch.float16 or tuple(scale.shape) != (shape[0],)
                or scale.device.type != "cpu" or not scale.is_contiguous()
                or not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all())):
            raise RPUBackendError(f"4B runtime-alignment W8 invalid weight/scale: {name}")
    quant_ids = {id(module.weight) for _, module, _ in quant_modules}
    for name, parameter in model.named_parameters():
        expected = torch.int8 if id(parameter) in quant_ids else torch.float16
        if parameter.device.type != "cpu" or parameter.dtype != expected or not parameter.is_contiguous():
            raise RPUBackendError(f"4B runtime-alignment W8 invalid CPU tensor: {name}")
    if tuple(text.embed_tokens.weight.shape) != (151936, 2560):
        raise RPUBackendError("4B runtime-alignment W8 requires raw FP16 embedding [151936,2560]")
    if text.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr():
        raise RPUBackendError("4B W8 head must be independent of the FP16 embedding")
    for name, buffer in model.named_buffers():
        if buffer.device.type != "cpu":
            raise RPUBackendError(f"4B runtime-alignment W8 non-CPU buffer: {name}")
    return True


_FAILED_RETIREMENTS = []
def _qwen3_vl_text_projection_weights_are_fp16(text_model) -> bool:
    """Whether every installed decoder projection can use ACC32 kernels."""
    layers = getattr(text_model, "layers", ())
    return all(
        layer.get_submodule(name).weight.dtype == torch.float16
        for layer in layers
        for name in _QWEN3_VL_TEXT_PROJECTIONS
    )




def _closed_qwen3_vl_forward(*args, **kwargs):
    raise RPUBackendError("Qwen3-VL is closed; reload from_pretrained before forward.")


def _is_retirement_owner(parent, model):
    from rpu_backend.api._execution import ExecutionSession

    session = getattr(parent, "_execution_session", None)
    return (isinstance(parent, Qwen3VLAdapter) and getattr(parent, "model", None) is model
            and getattr(parent, "_gc_retirement_enabled", False)
            and isinstance(session, ExecutionSession)
            and session is getattr(model, "_execution_session", None)
            and getattr(session, "_owner", None) is model)


def _retain_failed_retirement(owner, error):
    from rpu_backend.api._execution import _mark_execution_process_unsafe

    if not any(value is owner for value in _FAILED_RETIREMENTS):
        _FAILED_RETIREMENTS.append(owner)
    vars(owner)["_retirement_failed"] = error
    session = getattr(owner, "_execution_session", None)
    if session is not None:
        session.poison()
    _mark_execution_process_unsafe(f"Qwen3-VL retirement failed: {error}")


def _take_component_retirement(adapter, child, finalizer_name):
    """The actual composite owns graphs and weights, not integer callbacks."""
    if not _is_retirement_owner(adapter, adapter.model):
        raise RuntimeError("Qwen3-VL component requires its actual retirement Session owner")
    parent = getattr(child, "_qwen3_vl_retirement_owner", None)
    if parent is not None and parent is not adapter:
        raise RuntimeError("Qwen3-VL component already has a different retirement owner")
    resource = getattr(child, finalizer_name.replace("handle_finalizer", "retirement_state"), None)
    if resource is not None:
        if resource.owner() is not child:
            raise RuntimeError("Qwen3-VL component has a different native resource owner")
        resource.take_ownership(adapter)
    finalizer = getattr(child, finalizer_name, None)
    if finalizer is not None:
        if not finalizer.alive and parent is None and resource is None:
            raise RuntimeError("Qwen3-VL cannot adopt a consumed native finalizer")
        if finalizer.alive:
            finalizer.detach()
    vars(child)["_qwen3_vl_retirement_owner"] = adapter


def _qwen3_vl_runtime_complete(model) -> bool:
    try:
        text_model = model.model.language_model
        vision_model = model.model.visual
    except AttributeError:
        return False
    parent = getattr(model, "_qwen3_vl_retirement_owner", None)
    return bool(
        _is_retirement_owner(parent, model)
        and not getattr(parent, "_closed", False)
        and getattr(parent, "_retirement_failed", None) is None
        and not parent._execution_session._closed
        and not parent._execution_session._poisoned
        and getattr(text_model, "_qwen3_vl_retirement_owner", None) is parent
        and getattr(vision_model, "_qwen3_vl_retirement_owner", None) is parent
        and not getattr(getattr(text_model, "_rpu_decoder_handle_finalizer", None), "alive", False)
        and not getattr(getattr(vision_model, "_rpu_vision_handle_finalizer", None), "alive", False)
        and getattr(model, "_rpu_swizzled", False)
        and hasattr(model, "_rpu_lm_head_w_keepalive")
        and "forward" in vars(model)
        and getattr(text_model, "_rpu_decoder_handle", None) is not None
        and hasattr(text_model, "_rpu_text_graph_cache")
        and hasattr(text_model, "_rpu_decoder_graph_cache")
        and "forward" in vars(text_model)
        and getattr(vision_model, "_rpu_vision_handle", None) is not None
        and hasattr(vision_model, "_rpu_vision_graph_cache")
        and "forward" in vars(vision_model)
    )


def _restore_instance_forward(owner, had_forward: bool, forward) -> None:
    owner_state = vars(owner)
    if had_forward:
        owner_state["forward"] = forward
    else:
        owner_state.pop("forward", None)


def uninstall_qwen3_vl_runtime(
    model,
    text_model,
    vision_model,
    forward_snapshots=(),
) -> None:
    """Retire a Qwen3-VL text+vision install: caches, handles, attrs, forwards.

    Split out of `_cleanup_failed_qwen3_vl_install` so that the OTHER builders
    reusing the Qwen3-VL INSTALLERS reuse its UNINSTALLER too, instead of each
    shipping a partial one. `adapters/gr00t/runtime.py` is the caller that
    motivated the split: it runs the same `install_qwen3_vl_vision_for_rpu` /
    `install_qwen3_vl_text_for_rpu`, and its rollback used to clear two graph
    caches and fire two finalizers while deleting NO `_rpu_*` attribute and
    restoring NO `forward` — so a failure after those installs left a module
    tree that still looked installed but pointed at a destroyed handle.

    `forward_snapshots` is a sequence of `(owner, had_forward, forward)` taken
    BEFORE the install; the default `()` means "leave instance forwards alone",
    so callers that install one MUST pass their snapshots.

    Weight swizzling is irreversible and is deliberately NOT undone; the caller
    keeps its own poison marker (`_rpu_swizzle_started`) in place.
    """
    from rpu_backend.api._execution import _require_execution_process_safe

    try:
        _require_execution_process_safe()
        seen = set()
        for child, attrs in (
            (text_model, ("_rpu_text_graph_cache", "_rpu_decoder_graph_cache")),
            (vision_model, ("_rpu_vision_graph_cache",)),
        ):
            for attr in attrs:
                cache = getattr(child, attr, None)
                if cache is not None and id(cache) not in seen:
                    seen.add(id(cache))
                    cache.clear()
                    if not cache.cache_invariant_ok():
                        raise RuntimeError("Qwen3-VL GraphCache invariant failed during close")
        for child, handle_name, finalizer_name, destroy in (
            (text_model, "_rpu_decoder_handle", "_rpu_decoder_handle_finalizer", "causal_decoder_destroy"),
            (vision_model, "_rpu_vision_handle", "_rpu_vision_handle_finalizer", "qwen3vl_vision_destroy"),
        ):
            handle = getattr(child, handle_name, None)
            if handle is None:
                continue
            finalizer = getattr(child, finalizer_name, None)
            if finalizer is not None and not finalizer.alive:
                parent = getattr(child, "_qwen3_vl_retirement_owner", None)
                if (not _is_retirement_owner(parent, model)
                        or parent is not getattr(model, "_qwen3_vl_retirement_owner", None)):
                    raise RuntimeError("Qwen3-VL live handle has a consumed finalizer")
            resource = getattr(child, finalizer_name.replace("handle_finalizer", "retirement_state"), None)
            # Cyclic GC clears child weakrefs before the actual composite's
            # __del__. Its strong child/Session ownership was checked above.
            if resource is not None and (
                    (resource.owner() is not None and resource.owner() is not child)
                    or resource.handle != handle):
                raise RuntimeError("Qwen3-VL native retirement handle identity changed")
            getattr(torch.ops.rpu, destroy)(handle)
            if resource is not None:
                resource._native_destroyed(handle)
            vars(child)[handle_name] = None
            if finalizer is not None and finalizer.alive:
                finalizer.detach()
    except BaseException as error:
        for child, attr in ((text_model, "_rpu_decoder_retirement_state"),
                            (vision_model, "_rpu_vision_retirement_state")):
            resource = getattr(child, attr, None)
            if resource is not None and resource.handle is not None:
                resource.retain_failure(error, model)
        _retain_failed_retirement(model, error)
        raise

    if text_model is not None:
        text_state = vars(text_model)
        for name in tuple(text_state):
            if (name.startswith("_rpu_decoder_")
                    or name.startswith("_rpu_text_")
                    or name.startswith("_rpu_qwen3vl_")
                    or name.startswith("_rpu_deepstack_")
                    or name == "_rpu_prefill_execution_plan"):
                text_state.pop(name, None)
    if vision_model is not None:
        # Eager merger copies live on child modules, outside the root's
        # _rpu_vision_* namespace. Keep them until every native destroy above
        # succeeded, including on rollback or failed retirement.
        from .vision import _clear_vision_merger_aux
        _clear_vision_merger_aux(vision_model)
        vision_state = vars(vision_model)
        vision_exact_attrs = {
            "_rpu_lazy_init_checked",
            "_rpu_required_attrs",
            "_rpu_patch_embed_on_device",
            "_rpu_patch_embed_w_rpu",
            "_rpu_patch_embed_b_rpu",
            "_rpu_patch_embed_output_slot",
        }
        for name in tuple(vision_state):
            if name.startswith("_rpu_vision_") or name in vision_exact_attrs:
                vision_state.pop(name, None)

    for owner, had_forward, forward in forward_snapshots:
        _restore_instance_forward(owner, had_forward, forward)
    model_state = vars(model)
    model_state.pop("_rpu_lm_head_w_keepalive", None)
    model_state.pop("_rpu_lm_head_scale_keepalive", None)
    model_state.pop("_rpu_lm_head_raw_fp16_cpu", None)
    model_state.pop("_rpu_last_decode_hidden", None)
    model_state.pop("_rpu_visual_cached_prefix", None)
    model_state["_rpu_swizzled"] = False


def _cleanup_failed_qwen3_vl_install(
    adapter,
    model,
    text_model,
    vision_model,
    forward_snapshots,
) -> None:
    """Retire a partial composite install without unpoisoning swizzled weights."""
    adapter._forward_snapshots = tuple(forward_snapshots)
    adapter._execution_session.shutdown(lambda: adapter._close_without_session(release=False))
    adapter._rpu_is_ready = False


def _expand_video_grid_per_frame(video_grid_thw: torch.Tensor) -> torch.Tensor:
    """Expand `[[grid_t, h, w], ...]` → `[[1, h, w], [1, h, w], ...]`.

    HF Qwen3-VL `get_rope_index` walks `mm_token_type_ids` group-by-group, and
    the processor wraps each fold-frame in its own `<|vision_start|>...
    <|vision_end|>` block, so `get_rope_index` sees `grid_t` separate video
    groups per video and needs the same count of grid_thw rows.  The vision
    tower keeps the T-folded form (`[[grid_t, h, w]]`) — these two shapes
    co-exist for the same video forward.
    """
    rows = []
    for row in video_grid_thw:
        for _ in range(int(row[0].item())):
            rows.append(torch.tensor(
                [1, int(row[1].item()), int(row[2].item())],
                dtype=video_grid_thw.dtype,
            ))
    return torch.stack(rows, dim=0)


def _deepstack_text_layer_indices(vision_config) -> list[int]:
    """Map ordered vision features to the first text layers, as HF does."""
    return list(range(len(getattr(vision_config, "deepstack_visual_indexes", ()) or ())))


class Qwen3VLAdapter:
    """Per-instance adapter for `Qwen3VLForConditionalGeneration`.

    Lifecycle (D-01 two-step + RPU shortcut):
      1. `__init__(model)` — preflight, install `.to('rpu')` interceptor.
      2. `to_rpu()` — irreversible swizzle + install vision/text C++ handles +
         patch top-level forward. Returns the same model object.

    Internal handle map (after `to_rpu()`):
      - `model.model.visual._rpu_vision_handle`     (Qwen3VLVisionModel)
      - `model.model.language_model._rpu_decoder_handle`  (Qwen3VLTextModel)
      - `model._rpu_lm_head_w_keepalive`            (col-swizzled lm_head)
    """

    _rpu_execution_components = _QWEN3_VL_EXECUTION_COMPONENTS
    _EXECUTION_SUPPORTED = _QWEN3_VL_EXECUTION_SUPPORTED

    @classmethod
    def preflight(cls, config) -> None:
        """Config-only check; called from RPUModelForConditionalGeneration."""
        _check_profile(config)

    @classmethod
    def preflight_execution(cls, config, execution_config) -> None:
        _root, text_execution, vision_execution = _resolve_qwen3_vl_execution(
            execution_config,
            entry_point="Qwen3VLAdapter.preflight_execution",
        )
        runtime_quant_large = (
            _is_qwen3_vl_runtime_quant_config(config)
            and config.text_config.hidden_size in (4096, 5120)
        )
        _qwen3_vl_core_topology(config, _root)
        requested = text_execution.get("prefill", {}).get(
            "chunk_size", "auto"
        )
        # Auto planning needs an admitted envelope too. Reject before the
        # public loader stages weights, rather than during handle installation.
        text = config.text_config
        if _is_qwen3_vl_32b_w8a16_config(config):
            # The exact controlled legacy recipe owns a separate narrow envelope.
            # _check_profile retains its existing explicit opt-in and precision gate.
            _check_profile(config)
            from .text import _LEGACY_32B_CONTROLLED_ENVELOPE
            env = _LEGACY_32B_CONTROLLED_ENVELOPE
        elif runtime_quant_large and text.hidden_size == 5120:
            from .text import _RUNTIME_QUANTIZED_32B_ENVELOPE
            env = _RUNTIME_QUANTIZED_32B_ENVELOPE
        else:
            try:
                env = lookup_causal_decoder(
                    QWEN3_VL_TEXT_ARCH,
                    int(text.num_hidden_layers),
                    int(text.hidden_size),
                )
            except RuntimeError as exc:
                raise UnsupportedModelError(
                    "Qwen3-VL has no certified chunk envelope for text "
                    f"geometry ({text.num_hidden_layers}, {text.hidden_size}). "
                    "A controlled-evaluation opt-in does not supply an envelope. "
                    "Refusing before loading model weights."
                ) from exc
        if isinstance(requested, int):
            if requested > env.chunk:
                raise UnsupportedModelError(
                    f"Qwen3-VL prefill chunk_size={requested} exceeds this "
                    f"profile's planning ceiling {env.chunk}; maximum prefill "
                    f"execution length is {env.max_kv_len}. Refusing before "
                    "loading model weights."
                )
        vision_chunk = vision_execution.get("vision", {}).get(
            "chunk_size", "auto"
        )
        if isinstance(vision_chunk, int) and vision_chunk > 4096:
            raise UnsupportedModelError(
                "Qwen3-VL vision chunk_size exceeds the native 4096-patch "
                f"planning horizon: got {vision_chunk}."
            )

    def __init__(self, model) -> None:
        self._gc_retirement_enabled = False
        self._closed = False
        _check_profile(model.config)
        self.model = model
        self._rpu_execution = bind_rpu_execution(
            model,
            getattr(model, "_rpu_execution", None),
            entry_point="Qwen3VLAdapter",
            supported_components=_QWEN3_VL_EXECUTION_COMPONENTS,
            supported=_QWEN3_VL_EXECUTION_SUPPORTED,
        )
        self.preflight_execution(model.config, self._rpu_execution)
        self._topology = _qwen3_vl_core_topology(model.config, self._rpu_execution)
        self._linear_acc32_defaults = None
        self._linear_acc32_policy = None
        self._text_fast_replay_policy = None
        self._core_profile = (qwen3_vl_text_core_profile(model.config.text_config,
            self._topology.num_cores) if self._topology is not None else None)
        existing_session = getattr(model, "_execution_session", None)
        inherited_generation = int(getattr(existing_session, "generation", 0))
        inherited_component_generations = getattr(
            model, "_qwen3_vl_execution_component_generations", None
        )
        self._execution_generation = inherited_generation
        self._component_generations = (
            {
                _QWEN3_VL_TEXT_COMPONENT: inherited_generation,
                _QWEN3_VL_VISION_COMPONENT: inherited_generation,
            }
            if inherited_component_generations is None
            else {
                component: int(inherited_component_generations[component])
                for component in _QWEN3_VL_EXECUTION_COMPONENTS
            }
        )
        self._execution_reconfigure_journal = None
        self._forward_snapshots = tuple(
            getattr(model, "_qwen3_vl_forward_snapshots", ())
        )
        self._publish_execution_views(
            _resolve_qwen3_vl_execution(
                self._rpu_execution, entry_point="Qwen3VLAdapter"
            ),
            generation=inherited_generation,
            component_generations=self._component_generations,
        )
        session_callbacks = {}
        if existing_session is None:
            session_callbacks = {
                "validate": self.validate_execution_reconfigure,
                "apply": self.apply_execution_reconfigure,
                "rollback": self.rollback_execution_reconfigure,
            }
        self._execution_session = bind_execution_session(
            model,
            self._rpu_execution,
            entry_point="Qwen3VLAdapter",
            supported_components=_QWEN3_VL_EXECUTION_COMPONENTS,
            supported=_QWEN3_VL_EXECUTION_SUPPORTED,
            graph_mode=GRAPH_COMPOSITE_CHILD,
            **session_callbacks,
        )
        self._execution_session.register_config_view(self)
        self._execution_session._bind_planner_owner(
            model.model.language_model, _QWEN3_VL_TEXT_COMPONENT
        )
        self._execution_session._bind_planner_owner(
            model.model.visual, _QWEN3_VL_VISION_COMPONENT
        )
        self._rpu_is_ready: bool = _qwen3_vl_runtime_complete(model)
        self._rpu_w8a16_staged_plan = None
        if (_is_qwen3_vl_32b_w8a16_config(model.config)
                or _is_qwen3_vl_4b_w8a16_config(model.config)) and not self._rpu_is_ready:
            from rpu_backend.quant.load import (
                _W8A16_IMAGETEXT_STAGE_ATTR,
                _validate_staged_w8a16_imagetext,
            )

            if _W8A16_IMAGETEXT_STAGE_ATTR in vars(model):
                self._rpu_w8a16_staged_plan = (
                    _validate_staged_w8a16_imagetext(model)
                )
            else:
                _validate_qwen3_vl_32b_w8a16_model(model)
                _validate_qwen3_vl_4b_w8a16_model(model)
        self._is_graph_blocked_32b_w8a16 = bool(
            _is_qwen3_vl_32b_w8a16_config(model.config)
        )
        self._is_runtime_align_4b_w8a16 = _is_qwen3_vl_4b_w8a16_config(model.config)
        from rpu_backend.quant.load import is_qwen3_vl_2b_w8a16_config
        self._is_text_w8a16_2b = is_qwen3_vl_2b_w8a16_config(model.config)
        if self._is_text_w8a16_2b and not self._rpu_is_ready:
            _validate_qwen3_vl_2b_w8a16_model(model)
        self._is_runtime_quant = _is_qwen3_vl_runtime_quant_config(model.config)
        if self._is_runtime_quant and not self._rpu_is_ready:
            from rpu_backend.quant.load_qwen3_vl_runtime import _validate_qwen3_vl_runtime_model

            _validate_qwen3_vl_runtime_model(model)
        self._is_text_awq = _is_qwen3_vl_awq_config(model.config)
        if self._is_text_awq and not self._rpu_is_ready:
            from rpu_backend.quant.load import _validate_qwen3_vl_awq_model

            _validate_qwen3_vl_awq_model(model)

        # Idempotent `.to('rpu')` interceptor (matches Qwen3Adapter pattern).
        if getattr(model.to, "__rpu_wrapped__", False):
            return

        original_to = model.to

        def rpu_aware_to(*args, **kwargs):
            target = extract_to_device_target(args, kwargs)
            if target is not None and is_rpu_device_target(
                target, entry_point="Qwen3VLAdapter.model.to"
            ):
                rejected = set(kwargs) - {"device"}
                if rejected:
                    raise ValueError(
                        f"model.to('rpu', ...) does not accept extra kwargs "
                        f"{sorted(rejected)}; pass execution controls through "
                        "the loader's cold `rpu_execution` argument."
                    )
                if len(args) > 1:
                    raise ValueError(
                        f"model.to('rpu', *args) does not accept extra positional args ({args[1:]!r})."
                    )
                return self.to_rpu()
            if (
                self._rpu_is_ready
                or getattr(self.model, "_rpu_swizzled", False)
                or getattr(self.model, "_rpu_swizzle_started", False)
            ):
                raise RPUBackendError(
                    f"model.to({target!r}) rejected: weights swizzled for RPU "
                    "(irreversible). Reload via RPUModelForConditionalGeneration.from_pretrained(...)."
                )
            return original_to(*args, **kwargs)

        rpu_aware_to.__rpu_wrapped__ = True
        model.to = rpu_aware_to

    def _publish_execution_views(
        self,
        resolved,
        *,
        generation: int,
        component_generations=None,
    ) -> None:
        root, text, vision = resolved
        if component_generations is not None:
            self._component_generations = dict(component_generations)
        self._rpu_execution = root
        self._text_execution = text
        self._vision_execution = vision
        self._execution_generation = int(generation)

        text_model = self.model.model.language_model
        vision_model = self.model.model.visual
        children = (
            (text_model, _QWEN3_VL_TEXT_COMPONENT, text),
            (vision_model, _QWEN3_VL_VISION_COMPONENT, vision),
        )
        vars(self.model)["_qwen3_vl_execution_components"] = {
            _QWEN3_VL_TEXT_COMPONENT: text,
            _QWEN3_VL_VISION_COMPONENT: vision,
        }
        vars(self.model)["_qwen3_vl_execution_generation"] = int(generation)
        vars(self.model)["_qwen3_vl_execution_component_generations"] = dict(
            self._component_generations
        )
        for child, component, config in children:
            child_state = vars(child)
            child_state["_rpu_execution"] = config
            child_state["_fmb_execution_component_id"] = component
            child_state["_fmb_execution_generation"] = int(
                self._component_generations[component]
            )
        vars(vision_model)["_rpu_execution_graph_key_words"] = (
            int(self._component_generations[_QWEN3_VL_VISION_COMPONENT]),
        )

    @staticmethod
    def _reset_graph_owner(cache, *, label: str) -> None:
        if cache is None:
            return
        cache.begin_warmup()
        cache.clear()
        if not cache.cache_invariant_ok():
            raise RuntimeError(
                f"Qwen3-VL {label} GraphCache invariant failed during "
                "execution reconfigure"
            )

    def _reset_component_graphs(self, changed) -> None:
        if _QWEN3_VL_TEXT_COMPONENT in changed:
            text_model = self.model.model.language_model
            seen = set()
            for attr in ("_rpu_text_graph_cache", "_rpu_decoder_graph_cache"):
                cache = getattr(text_model, attr, None)
                if cache is not None and id(cache) not in seen:
                    seen.add(id(cache))
                    self._reset_graph_owner(cache, label="text")
            vars(text_model).pop("_rpu_last_execution_plan", None)
        if _QWEN3_VL_VISION_COMPONENT in changed:
            vision_model = self.model.model.visual
            self._reset_graph_owner(
                getattr(vision_model, "_rpu_vision_graph_cache", None),
                label="vision",
            )
            vars(vision_model).pop("_rpu_last_execution_plan", None)

    def validate_execution_reconfigure(self, execution_config) -> None:
        if not self._rpu_is_ready:
            raise RuntimeError(
                "Qwen3VLAdapter.reconfigure requires to_rpu() to complete first"
            )
        require_same_decoder_topology(self._rpu_execution, execution_config)
        self.preflight_execution(self.model.config, execution_config)
        _root, text_execution, vision_execution = _resolve_qwen3_vl_execution(
            execution_config,
            entry_point="Qwen3VLAdapter.validate_execution_reconfigure",
        )
        if self._linear_acc32_policy is not None:
            defaults = self._linear_acc32_defaults
            requested = {
                "prefill": _qwen3_vl_linear_acc32(
                    text_execution, "prefill", defaults["prefill"]),
                "vision": _qwen3_vl_linear_acc32(
                    vision_execution, "vision", defaults["vision"]),
            }
            if requested != self._linear_acc32_policy:
                raise ValueError(
                    "Qwen3-VL linear_acc32 is immutable after installation; "
                    "reload the model with the desired cold configuration"
                )
        delivery_profile = _delivery_profile_active(self.model)
        if delivery_profile:
            _normalize_delivery_execution(execution_config)
        requested_fast_replay = delivery_profile or bool(
            text_execution.get("prefill", {}).get("fast_replay", False)
        )
        if (self._text_fast_replay_policy is not None
                and requested_fast_replay != self._text_fast_replay_policy):
            raise ValueError(
                "Qwen3-VL fast_replay is immutable after installation; "
                "reload the model with the desired cold configuration"
            )
        required = (
            "execution_reconfigure_begin",
            "execution_reconfigure_commit",
            "execution_reconfigure_abort",
            "execution_reconfigure_abort_attempt",
            "causal_decoder_stage_chunk_size_override",
            "qwen3vl_vision_stage_chunk_size",
        )
        missing = [name for name in required if not hasattr(torch.ops.rpu, name)]
        if missing:
            raise RuntimeError(
                "Qwen3-VL binary lacks native execution-reconfigure op(s): "
                + ", ".join(missing)
            )

    @staticmethod
    def _native_chunk(config, stage: str) -> int:
        value = config.get(stage, {}).get("chunk_size", "auto")
        return 0 if value == "auto" else int(value)

    def _apply_execution_state(
        self,
        resolved,
        *,
        generation: int,
        component_generations=None,
        force_components=(),
    ) -> None:
        root, text, vision = resolved
        changed = {
            component
            for component, old, new in (
                (_QWEN3_VL_TEXT_COMPONENT, self._text_execution, text),
                (_QWEN3_VL_VISION_COMPONENT, self._vision_execution, vision),
            )
            if old != new
        }
        changed.update(force_components)
        if component_generations is None:
            component_generations = dict(self._component_generations)
            for component in changed:
                component_generations[component] += 1

        if changed:
            text_model = self.model.model.language_model
            vision_model = self.model.model.visual
            with native_execution_reconfigure(
                torch.ops.rpu, journal=self._execution_reconfigure_journal,
            ) as token:
                if _QWEN3_VL_TEXT_COMPONENT in changed:
                    torch.ops.rpu.causal_decoder_stage_chunk_size_override(
                        text_model._rpu_decoder_handle,
                        token,
                        self._native_chunk(text, "prefill"),
                    )
                if _QWEN3_VL_VISION_COMPONENT in changed:
                    torch.ops.rpu.qwen3vl_vision_stage_chunk_size(
                        vision_model._rpu_vision_handle,
                        token,
                        self._native_chunk(vision, "vision"),
                    )

            if _QWEN3_VL_VISION_COMPONENT in changed:
                vars(vision_model)["_rpu_vision_execution_chunk_size"] = (
                    vision.get("vision", {}).get("chunk_size", "auto")
                )
            self._reset_component_graphs(changed)
        if self._execution_reconfigure_journal is not None:
            self._execution_reconfigure_journal["mutation_started"] = True
        self._publish_execution_views(
            resolved,
            generation=generation,
            component_generations=component_generations,
        )

    def apply_execution_reconfigure(
        self, old_config, new_config, generation: int, *, force_rebuild=False
    ) -> None:
        self._execution_reconfigure_journal = {"mutation_started": False}
        old_resolved = _resolve_qwen3_vl_execution(
            old_config, entry_point="Qwen3VLAdapter.reconfigure rollback"
        )
        new_resolved = _resolve_qwen3_vl_execution(
            new_config, entry_point="Qwen3VLAdapter.reconfigure"
        )
        changed = {
            component
            for component, old, new in (
                (_QWEN3_VL_TEXT_COMPONENT, old_resolved[1], new_resolved[1]),
                (_QWEN3_VL_VISION_COMPONENT, old_resolved[2], new_resolved[2]),
            )
            if old != new
        }
        if force_rebuild:
            changed.update((_QWEN3_VL_TEXT_COMPONENT, _QWEN3_VL_VISION_COMPONENT))
        self._execution_reconfigure_journal.update({
            "resolved": old_resolved,
            "generation": self._execution_generation,
            "component_generations": dict(self._component_generations),
            "changed": changed,
        })
        self._apply_execution_state(
            new_resolved, generation=generation, force_components=changed,
        )
        # Retain undo state through the shared session's config publication.

    def rollback_execution_reconfigure(
        self, old_config, _new_config, generation: int
    ) -> None:
        journal = self._execution_reconfigure_journal
        if journal is not None and not journal.get("mutation_started", False):
            self._execution_reconfigure_journal = None
            return
        if journal is None:
            resolved = _resolve_qwen3_vl_execution(
                old_config, entry_point="Qwen3VLAdapter.reconfigure rollback"
            )
            force_components = {
                component
                for component, current, old in (
                    (
                        _QWEN3_VL_TEXT_COMPONENT,
                        self._text_execution,
                        resolved[1],
                    ),
                    (
                        _QWEN3_VL_VISION_COMPONENT,
                        self._vision_execution,
                        resolved[2],
                    ),
                )
                if current != old
            }
            component_generations = dict(self._component_generations)
        else:
            resolved = journal["resolved"]
            generation = journal["generation"]
            force_components = journal["changed"]
            component_generations = journal["component_generations"]
        self._apply_execution_state(
            resolved,
            generation=generation,
            component_generations=component_generations,
            force_components=force_components,
        )
        self._execution_reconfigure_journal = None

    def _close_without_session(self, *, release=True) -> None:
        if self._closed:
            return
        text_model = self.model.model.language_model
        vision_model = self.model.model.visual
        uninstall_qwen3_vl_runtime(
            self.model,
            text_model,
            vision_model,
            self._forward_snapshots if not release else (),
        )
        vars(vision_model).pop("_rpu_execution_graph_key_words", None)
        self._rpu_is_ready = False
        self._closed = True
        if release:
            for child in (self.model, text_model, vision_model):
                vars(child)["forward"] = _closed_qwen3_vl_forward
            _release_live_instance(self.model)

    def close(self) -> None:
        parent = getattr(self.model, "_qwen3_vl_retirement_owner", None)
        if parent is not None and parent is not self:
            if not _is_retirement_owner(parent, self.model):
                raise RuntimeError("Qwen3-VL close lost its actual composite retirement owner")
            parent.close()
            self._rpu_is_ready = False
            self._closed = True
            return
        try:
            self._execution_session.shutdown(self._close_without_session)
        except BaseException as error:
            from rpu_backend.api import _execution
            if _execution._UNSAFE_PROCESS_REASON is not None:
                _retain_failed_retirement(self, error)
            raise
        if not self._closed:
            error = RuntimeError("Qwen3-VL closed Session still owns native resources")
            _retain_failed_retirement(self, error)
            raise error

    def __del__(self):
        if (not getattr(self, "_gc_retirement_enabled", False)
                or getattr(self, "_closed", False)
                or getattr(self, "_retirement_failed", None) is not None):
            return
        try:
            with self._execution_session._lock:
                if self._execution_session._active:
                    raise RuntimeError("Qwen3-VL GC during an active forward")
                from rpu_backend.api import causal_lm
                with causal_lm._LIVE_LOCK:
                    live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                    if live is not None and live is not self.model:
                        raise RuntimeError("Qwen3-VL GC after live-owner handoff")
                    self.close()
        except BaseException as error:
            _retain_failed_retirement(self, error)

    # ------------------------------------------------------------------ #
    # to_rpu
    # ------------------------------------------------------------------ #

    def _validate_staged_checkpoint_plan(self, staged_plan):
        """Recheck the same metadata object before the irreversible claim."""
        if staged_plan is not None:
            from rpu_backend.quant.load import _validate_staged_w8a16_imagetext

            if _validate_staged_w8a16_imagetext(self.model) is not staged_plan:
                raise RPUBackendError(
                    "Qwen3VLAdapter.to_rpu(): staged checkpoint plan changed; reload."
                )
        return staged_plan

    def to_rpu(self):
        """Run the full RPU enablement sequence. Idempotent for the same model.

        Order (intentional — see findings F31 P1):
          A. For ordinary checkpoints, swizzle and transfer each text decoder
             layer in turn. Staged W8A16 uses its dedicated materializer.
          B. Untie + col-swizzle `lm_head` on CPU.
          C. Move the remaining text_model modules + lm_head to RPU.
          D. Install vision encoder (does its own fused-QKV split + swizzle).
          E. Install text decoder C++ handle with DeepStack outputs injected
             into the leading text layers in list order.
          F. Push lm_head weight to the text-decoder C++ handle.
          G. Patch top-level Qwen3VLForConditionalGeneration.forward.
        """
        from rpu_backend.api._execution import _require_execution_process_safe
        _require_execution_process_safe()
        if self._closed:
            raise RPUBackendError("Qwen3-VL prior swizzle attempt or close ended this instance; reload from_pretrained.")
        if self._rpu_is_ready or getattr(self.model, "_rpu_swizzled", False):
            if not _qwen3_vl_runtime_complete(self.model):
                raise RPUBackendError(
                    "Qwen3VLAdapter.to_rpu(): model is marked swizzled but its "
                    "RPU runtime is incomplete; reload from_pretrained."
                )
            self._rpu_is_ready = True
            return self.model

        if (
            getattr(self.model, "_rpu_swizzle_started", False)
            and not getattr(self.model, "_rpu_swizzled", False)
        ):
            raise RPUBackendError(
                "Qwen3VLAdapter.to_rpu(): prior swizzle attempt left the model "
                "in undefined state. Reload from_pretrained."
            )

        # The adapter may have been constructed long before this irreversible
        # boundary. Re-check nested config objects before claiming ownership or
        # transforming any weight.
        _check_profile(self.model.config)
        staged_plan = self._rpu_w8a16_staged_plan
        self._validate_staged_checkpoint_plan(staged_plan)
        is_runtime_align_4b_w8a16 = _is_qwen3_vl_4b_w8a16_config(self.model.config)
        if staged_plan is not None:
            is_graph_blocked_32b_w8a16 = _is_qwen3_vl_32b_w8a16_config(self.model.config)
        else:
            is_graph_blocked_32b_w8a16 = (
                _validate_qwen3_vl_32b_w8a16_model(self.model)
            )
            _validate_qwen3_vl_4b_w8a16_model(self.model)
        if (is_graph_blocked_32b_w8a16 != self._is_graph_blocked_32b_w8a16
                or is_runtime_align_4b_w8a16 != getattr(self, "_is_runtime_align_4b_w8a16", False)):
            raise RPUBackendError(
                "Qwen3VLAdapter.to_rpu(): config or quantized tensor inventory "
                "changed after adapter construction; reload from_pretrained."
            )
        text_w8_2b = _validate_qwen3_vl_2b_w8a16_model(self.model)
        if text_w8_2b != getattr(self, "_is_text_w8a16_2b", False):
            raise RPUBackendError("Qwen3-VL-2B W8A16 config changed after adapter construction; reload")
        from rpu_backend.quant.load_qwen3_vl_runtime import _validate_qwen3_vl_runtime_model
        runtime_quant = _validate_qwen3_vl_runtime_model(self.model)
        if runtime_quant != getattr(self, "_is_runtime_quant", False):
            raise RPUBackendError("Qwen3-VL runtime quantized config changed after adapter construction; reload")
        runtime_align_quantized = is_runtime_align_4b_w8a16 or runtime_quant
        runtime_w4 = runtime_quant and self.model.lm_head.weight.dtype == torch.uint8
        from rpu_backend.quant.load import _validate_qwen3_vl_awq_model
        text_awq = _validate_qwen3_vl_awq_model(self.model)
        if text_awq != getattr(self, "_is_text_awq", False):
            raise RPUBackendError("Qwen3-VL AWQ config changed after adapter construction; reload")

        if not _SWIZZLE_LOCK.acquire(blocking=False):
            raise RPUBackendError(
                "Qwen3VLAdapter.to_rpu(): another swizzle in progress."
            )
        text_model = None
        vision_model = None
        forward_snapshots = []
        claim_acquired = False
        mutation_started = False
        try:
            if self._rpu_is_ready or getattr(self.model, "_rpu_swizzled", False):
                if not _qwen3_vl_runtime_complete(self.model):
                    raise RPUBackendError(
                        "Qwen3VLAdapter.to_rpu(): model is marked swizzled but "
                        "its RPU runtime is incomplete; reload from_pretrained."
                    )
                self._rpu_is_ready = True
                return self.model

            _check_profile(self.model.config)
            if staged_plan is not None:
                self._validate_staged_checkpoint_plan(staged_plan)
                is_graph_blocked_32b_w8a16 = _is_qwen3_vl_32b_w8a16_config(self.model.config)
            else:
                is_graph_blocked_32b_w8a16 = (
                    _validate_qwen3_vl_32b_w8a16_model(self.model)
                )
                _validate_qwen3_vl_4b_w8a16_model(self.model)
            cfg = self.model.config
            text_cfg = cfg.text_config
            vision_cfg = cfg.vision_config
            is_padded_8b = (
                _text_config_profile(text_cfg) == _QWEN3_VL_8B_TEXT_PROFILE
            )
            runtime_quant_large = runtime_quant and text_cfg.hidden_size in (4096, 5120)
            graph_cache_max_entries = (
                _QWEN3_VL_32B_RUNTIME_W8_GRAPH_ENTRIES
                if is_graph_blocked_32b_w8a16 or (runtime_quant
                and text_cfg.hidden_size == 5120
                and cfg.quant_config["method"] == "w8a16") else None
            )
            text_model = self.model.model.language_model
            vision_model = self.model.model.visual
            (
                lm_head_w4a16,
                lm_head_exact_candidates,
                fuse_prefill_last_lm_head,
                lm_head_w4_group_size,
            ) = _lm_head_optimization_config(self.model)
            delivery_w8_text = _validate_delivery_weights(self.model)
            if delivery_w8_text:
                _normalize_delivery_execution(self._rpu_execution)
            topology = _qwen3_vl_core_topology(cfg, self._rpu_execution)
            if topology != self._topology or (topology is not None and
                    qwen3_vl_text_core_profile(text_cfg, topology.num_cores) != self._core_profile):
                raise ValueError("Qwen3-VL geometry changed its bound cold topology")
            _validate_qwen3_vl_reduced_weights(self.model, topology)
            if _validate_qwen3_vl_2b_w8a16_model(self.model) != text_w8_2b:
                raise RPUBackendError("Qwen3-VL-2B W8A16 config changed before ownership")
            if _validate_qwen3_vl_runtime_model(self.model) != runtime_quant:
                raise RPUBackendError("Qwen3-VL runtime quantized config changed before ownership")
            if _validate_qwen3_vl_awq_model(self.model) != text_awq:
                raise RPUBackendError("Qwen3-VL AWQ config changed before ownership")
            use_2b_weight_banks = (
                topology is None and not delivery_w8_text and not lm_head_w4a16
                and getattr(self.model, "_qwen3_vl_delivery_profile", None) is None
                and (text_w8_2b or (
                    _is_qwen3_vl_2b_eager_profile(self.model)
                    and _qwen3_vl_text_projection_weights_are_fp16(text_model)))
            )
            use_4b_fp16_weight_banks = (
                topology is None and not delivery_w8_text and not lm_head_w4a16
                and getattr(self.model, "_qwen3_vl_delivery_profile", None) is None
                and _text_config_profile(text_cfg) == _QWEN3_VL_4B_TEXT_PROFILE
                and _is_qwen3_vl_dense_eager_profile(self.model)
                and _qwen3_vl_text_projection_weights_are_fp16(text_model)
            )
            # Exact padded 8B reuses storage only. Keep its accumulation
            # policy, C128 ceiling and Vision schedule unchanged.
            use_8b_fp16_weight_banks = (
                topology is None and is_padded_8b and not delivery_w8_text
                and not lm_head_w4a16
                and getattr(self.model, "_qwen3_vl_delivery_profile", None) is None
                and _qwen3_vl_text_projection_weights_are_fp16(text_model)
            )
            use_fp16_vision_weight_banks = (
                use_2b_weight_banks or use_4b_fp16_weight_banks or (
                    text_awq and topology is None and not delivery_w8_text
                    and not lm_head_w4a16
                    and getattr(self.model, "_qwen3_vl_delivery_profile", None) is None)
            )
            forward_snapshots = [
                (self.model, "forward" in vars(self.model), vars(self.model).get("forward")),
                (text_model, "forward" in vars(text_model), vars(text_model).get("forward")),
                (vision_model, "forward" in vars(vision_model), vars(vision_model).get("forward")),
            ]

            _claim_live_instance(self.model)
            claim_acquired = True
            try:
                # Reuse HostDDR mappings for weights and request temporaries.
                # This process policy must precede the first RPU allocation,
                # but follow the exclusive-model claim so a rejected install
                # cannot alter the live owner's allocator.
                torch.rpu.set_caching_allocator(True)
                if graph_cache_max_entries is not None:
                    arena_policy = rpu_backend.graph.GraphRuntimePolicy.from_environment(
                        graph_arena_count=3 * graph_cache_max_entries + 1)
                    if not arena_policy.prepare_arenas():
                        raise RPUBackendError(
                            "Qwen3-VL-32B W8 requires cold SDK graph arenas")
                    self._graph_arena_policy = arena_policy
            except BaseException:
                _release_live_instance(self.model)
                claim_acquired = False
                raise
            vars(self.model)["_qwen3_vl_retirement_owner"] = self
            self._gc_retirement_enabled = True
            vars(self.model)["_rpu_swizzle_started"] = True
            mutation_started = True

            # ---------- Step A: stream text decoder layers ------------------
            if runtime_quant:
                from rpu_backend.quant.load_qwen3_vl_runtime import _move_materialized_runtime_decoder_for_rpu

                _move_materialized_runtime_decoder_for_rpu(self.model)
            elif text_awq:
                from rpu_backend.quant.load import _move_materialized_awq_decoder_for_rpu

                _move_materialized_awq_decoder_for_rpu(self.model)
            elif staged_plan is not None:
                from rpu_backend.quant.load import (
                    _materialize_staged_w8a16_imagetext_for_rpu,
                )

                _materialize_staged_w8a16_imagetext_for_rpu(self.model)
            elif is_graph_blocked_32b_w8a16:
                from rpu_backend.quant.load import _move_materialized_decoder_for_rpu
                _move_materialized_decoder_for_rpu(
                    self.model, dtype=torch.int8, per_layer=True)
            elif is_runtime_align_4b_w8a16:
                from rpu_backend.quant.load import (
                    _move_materialized_w8a16_decoder_for_rpu,
                )

                _move_materialized_w8a16_decoder_for_rpu(self.model)
            elif use_2b_weight_banks or use_4b_fp16_weight_banks or use_8b_fp16_weight_banks:
                from rpu_backend.quant.load import _move_materialized_decoder_for_rpu

                _move_materialized_decoder_for_rpu(
                    self.model, dtype=torch.int8 if text_w8_2b else torch.float16,
                    **({"per_layer": True}
                       if use_4b_fp16_weight_banks or use_8b_fp16_weight_banks else {}))
            else:
                if topology is not None:
                    view = _qwen3_vl_decoder_weight_view(self.model)
                    pad_decoder_mlp_weights(
                        view, logical_size=text_cfg.intermediate_size,
                        physical_size=decoder_mlp_intermediate_size(
                            text_cfg.intermediate_size, topology.mlp_tp))
                    del view
                # Swizzling an entire decoder first retains all transformed CPU
                # weights while RPU copies are allocated. Bound that overlap for
                # every admitted non-staged profile by converting and moving one
                # layer at a time. The final text_model.to() below moves only the
                # embedding, final norm, and any other remaining weights.
                for layer in text_model.layers:
                    convert_linear_weights_inplace(
                        layer, skip_names=set(),
                        **({"attn_num_cores": topology.attn_tp,
                            "mlp_num_cores": topology.mlp_tp,
                            "lm_head_num_cores": topology.lm_head_tp,
                            "execution_core_count": topology.num_cores}
                           if topology is not None else {}),
                    )
                    layer.to("rpu")
                    gc.collect()

            # ---------- Step B: lm_head untie + col-swizzle (CPU) ------------
            # If lm_head.weight is tied with text_model.embed_tokens, clone
            # first so the embedding row layout stays raw.
            embed_w_ptr = text_model.embed_tokens.weight.data_ptr()
            lm_head_w = self.model.lm_head.weight.data
            if lm_head_w.data_ptr() == embed_w_ptr:
                lm_head_w = lm_head_w.clone()
                self.model.lm_head.weight = nn.Parameter(lm_head_w, requires_grad=False)
            # Runtime-alignment heads own quantized storage independently of embeddings.
            lm_head_cpu = (
                self.model.lm_head.weight.data.contiguous()
                if runtime_align_quantized else
                self.model.lm_head.weight.data.to(torch.float16).contiguous()
            )
            lm_head_w4_packed = None
            lm_head_w4_scale = None
            if lm_head_exact_candidates:
                self.model._rpu_lm_head_raw_fp16_cpu = lm_head_cpu
            if lm_head_w4a16:
                lm_head_w4_packed, lm_head_w4_scale = (
                    _quantize_pack_lm_head_w4_groupwise(
                        lm_head_cpu, group_size=lm_head_w4_group_size,
                    )
                )
            if runtime_w4:
                # The strict loader already performed the sole physical pgrp pack.
                swizzled = lm_head_cpu
            elif topology is None:
                swizzled = transform_linear_weight(lm_head_cpu, partition=1)
                self.model.lm_head.weight = nn.Parameter(swizzled, requires_grad=False)
            else:
                # Reduced text layers were streamed above; convert the
                # untied vocabulary head separately with its own cold layout.
                head_view = nn.Module()
                head_view.add_module("lm_head", self.model.lm_head)
                convert_linear_weights_inplace(
                    head_view, skip_names=set(),
                    attn_num_cores=topology.attn_tp,
                    mlp_num_cores=topology.mlp_tp,
                    lm_head_num_cores=topology.lm_head_tp,
                    execution_core_count=topology.num_cores,
                )
                swizzled = self.model.lm_head.weight.data
                del head_view

            # ---------- Step C: move text + lm_head to RPU -------------------
            text_model.to("rpu")
            self.model.lm_head.to("rpu")
            # These locals still own the old CPU storages after Module.to()
            # replaces its Parameter data. Keep only explicitly requested raw
            # candidate weights; release ordinary lm-head staging before the
            # vision tower and decoder handle allocate their own resources.
            del lm_head_w, lm_head_cpu, swizzled
            gc.collect()

            # ---------- Step D: install vision encoder -----------------------
            # The vision adapter handles its own per-block fused-QKV split +
            # swizzle. patch_embed / pos_embed / merger / deepstack_merger_list
            # remain on CPU per the adapter contract (findings F31/F32).
            install_qwen3_vl_vision_for_rpu(
                vision_model, vision_config=vision_cfg,
                execution_chunk_size=self._vision_execution.get(
                    "vision", {}
                ).get("chunk_size", "auto"),
                _allow_graph_blocked_32b=is_graph_blocked_32b_w8a16,
                _allow_padded_8b=is_padded_8b and not runtime_quant_large,
                **({"_allow_runtime_quantized_large": True} if runtime_quant_large else {}),
                **({"w8a16": True} if runtime_align_quantized else {}),
                **({"_pack_fp16_block_weights": True}
                   if use_fp16_vision_weight_banks or use_8b_fp16_weight_banks or is_graph_blocked_32b_w8a16 else {}),
                **({"num_cores": 4} if topology is not None else {}),
                **({"_graph_cache_max_entries": graph_cache_max_entries}
                   if graph_cache_max_entries is not None else {}),
            )
            _take_component_retirement(self, vision_model, "_rpu_vision_handle_finalizer")

            # ---------- Step E: install text decoder C++ handle --------------
            deepstack_visual_indexes = list(
                getattr(vision_cfg, "deepstack_visual_indexes", [])
            )
            deepstack_lang_layers = (
                deepstack_visual_indexes
                if delivery_w8_text
                else _deepstack_text_layer_indices(vision_cfg)
            )
            install_qwen3_vl_text_for_rpu(
                text_model,
                text_config=text_cfg,
                vision_config=vision_cfg,
                deepstack_lang_layers=deepstack_lang_layers,
                enable_deepstack=True,
                execution_config=self._text_execution,
                **({"runtime_align_w8a16": True} if is_runtime_align_4b_w8a16 else {}),
                **({"_legacy_32b_w8a16": True} if is_graph_blocked_32b_w8a16 else {}),
                **({"_runtime_quantized_32b": True}
                   if runtime_quant_large and text_cfg.hidden_size == 5120 else {}),
                **({"_graph_cache_max_entries": graph_cache_max_entries}
                   if graph_cache_max_entries is not None else {}),
                **({"topology": topology} if topology is not None else {}),
                scale_lists=(
                    _qwen3_vl_text_scale_lists(text_model)
                    if is_graph_blocked_32b_w8a16 or text_w8_2b or text_awq or delivery_w8_text or runtime_align_quantized else None
                ),
            )
            _take_component_retirement(self, text_model, "_rpu_decoder_handle_finalizer")
            # Exact-profile fast replay also skips ordinary position-register
            # refresh. Disabling an individual fusion must not collapse these
            # position-specific graphs into one stale graph.
            text_model._rpu_text_fused_decode_position = delivery_w8_text
            text_model._qwen3_vl_delivery_profile = getattr(self.model, "_qwen3_vl_delivery_profile", None)

            # ---------- Step F: push lm_head weight to C++ -------------------
            handle = text_model._rpu_decoder_handle
            enable_reconfigure = getattr(
                torch.ops.rpu,
                "causal_decoder_enable_execution_reconfigure",
                None,
            )
            if enable_reconfigure is None:
                raise RuntimeError(
                    "Qwen3-VL binary lacks causal-decoder hot-reconfigure "
                    "support"
                )
            enable_reconfigure(handle)
            # The P576/C192 candidate is delivery-only and unverified here. Keep the default
            # 2B envelope at 336, including FP16 and component-only callers.
            if delivery_w8_text:
                torch.ops.rpu.causal_decoder_set_chunk_envelope(handle, 608, 192)
            # Bind both policies before configuring any precision-specific route.
            vision_linear_acc32 = _qwen3_vl_linear_acc32(self._vision_execution, "vision")
            text_linear_acc32 = _qwen3_vl_linear_acc32(self._text_execution, "prefill")
            self._linear_acc32_defaults = {"prefill": False, "vision": False}
            self._linear_acc32_policy = {
                "prefill": text_linear_acc32, "vision": vision_linear_acc32,
            }
            torch.ops.rpu.qwen3vl_vision_set_linear_acc32(
                vision_model._rpu_vision_handle, vision_linear_acc32)
            vision_model._rpu_vision_linear_acc32 = vision_linear_acc32
            torch.ops.rpu.causal_decoder_set_linear_acc32(handle, text_linear_acc32)
            if not is_graph_blocked_32b_w8a16:
                # Text fast replay is an explicit cold opt-in. It is bounded
                # in native code to initial B1 causal prefill with an owner
                # witness; omitted config keeps the historical default off.
                text_fast_replay = bool(
                    self._text_execution.get("prefill", {}).get(
                        "fast_replay", False
                    )
                )
                self._text_fast_replay_policy = delivery_w8_text or text_fast_replay
                if text_fast_replay and not delivery_w8_text:
                    torch.ops.rpu.causal_decoder_set_fast_replay(handle, True)
                    torch.ops.rpu.causal_decoder_set_preload_replay_skip(handle, True)
                    text_model._rpu_text_fast_replay = True
                if use_fp16_vision_weight_banks:
                    # Storage banks also serve explicit ACC16. The selected
                    # large-image schedule is admitted only for ACC32.
                    if vision_linear_acc32:
                        torch.ops.rpu.qwen3vl_vision_set_large_image_auto_chunk(
                            vision_model._rpu_vision_handle, True,
                        )
                    # Cold ordinary 2B/4B FP16 Vision: replay the recorded
                    # encoder/preloads; post_fn still emits any admitted merger.
                    torch.ops.rpu.qwen3vl_vision_set_fast_replay(
                        vision_model._rpu_vision_handle, True,
                    )
                    torch.ops.rpu.qwen3vl_vision_set_preload_replay_skip(
                        vision_model._rpu_vision_handle, True,
                    )
                if runtime_align_quantized:
                    enable_qwen3_vl_vision_merger_on_device(vision_model, w8a16=True)
                    if not runtime_quant_large and not vision_linear_acc32:
                        register_qwen3vl_vision_fused_merger(vision_model, w8a16=True)
                        vision_model._rpu_vision_chunked_merger_w8a16 = True
                        torch.ops.rpu.qwen3vl_vision_set_n1200_tm160(
                            vision_model._rpu_vision_handle, True,
                        )
                        torch.ops.rpu.qwen3vl_vision_set_w8_compact_encoder(
                            vision_model._rpu_vision_handle, True,
                        )
                        vision_model._rpu_vision_compact_encoder_w8a16 = True
                    # Cold ordinary Vision only: retain post_fn re-emission for
                    # fused W8 mergers, including their post-encoder chunks.
                    torch.ops.rpu.qwen3vl_vision_set_fast_replay(vision_model._rpu_vision_handle, True)
                    torch.ops.rpu.qwen3vl_vision_set_preload_replay_skip(vision_model._rpu_vision_handle, True)
                elif delivery_w8_text:
                    torch.ops.rpu.qwen3vl_vision_set_delivery_compatibility(
                        vision_model._rpu_vision_handle, True
                    )
                    register_qwen3vl_vision_fused_merger(vision_model)
                    enable_qwen3_vl_vision_merger_on_device(vision_model)
                    torch.ops.rpu.qwen3vl_vision_set_fast_replay(vision_model._rpu_vision_handle, True)
                    torch.ops.rpu.qwen3vl_vision_set_preload_replay_skip(vision_model._rpu_vision_handle, True)
                    torch.ops.rpu.qwen3vl_vision_set_bake_merger(vision_model._rpu_vision_handle, True)
                    torch.ops.rpu.causal_decoder_set_fast_replay(handle, True)
                    torch.ops.rpu.causal_decoder_set_preload_replay_skip(handle, True)
                    text_model._rpu_text_fast_replay = True
                elif (
                    topology is None and (
                        text_w8_2b or text_awq
                        or (
                            ((is_padded_8b and _is_qwen3_vl_8b_vision_config(vision_cfg))
                             or _is_qwen3_vl_dense_eager_profile(self.model))
                            and _qwen3_vl_text_projection_weights_are_fp16(text_model)
                        )
                    )
                    and not lm_head_w4a16
                    and not getattr(vision_model, "_rpu_vision_host_fp32_patch", False)
                ):
                    # Dense profiles, exact 2B Text-W8 and Text-AWQ retain FP16 Vision.
                    # Install eager auxiliaries for general grids. Ordinary
                    # 2B/4B additionally admit the N1200 chunked device merger;
                    # an explicit CPU FP32 patch profile remains unchanged.
                    enable_qwen3_vl_vision_merger_on_device(vision_model)
                    if (use_fp16_vision_weight_banks and vision_linear_acc32
                            and _is_qwen3_vl_dense_eager_profile(self.model)
                            and _qwen3_vl_text_projection_weights_are_fp16(text_model)):
                        register_qwen3vl_vision_fused_merger(vision_model)
                        torch.ops.rpu.qwen3vl_vision_set_fp16_chunked_merger(
                            vision_model._rpu_vision_handle, True,
                        )
                        vision_model._rpu_vision_chunked_merger_fp16 = True
                        torch.ops.rpu.qwen3vl_vision_set_fp16_compact_encoder(
                            vision_model._rpu_vision_handle, True,
                        )
                        vision_model._rpu_vision_compact_encoder_fp16 = True
            if runtime_align_quantized:
                lm_w = self.model.lm_head.weight.detach().contiguous()
                lm_scale = self.model.lm_head.weight_scale.detach().contiguous()
                self.model._rpu_lm_head_w_keepalive = lm_w
                self.model._rpu_lm_head_scale_keepalive = lm_scale
                torch.ops.rpu.causal_decoder_set_lm_head(handle, lm_w, lm_scale)
            elif lm_head_w4a16:
                assert lm_head_w4_packed is not None
                assert lm_head_w4_scale is not None
                lm_w = lm_head_w4_packed.to("rpu").contiguous()
                lm_scale = lm_head_w4_scale.to(
                    device="rpu", dtype=torch.float16,
                ).contiguous()
                self.model._rpu_lm_head_w_keepalive = lm_w
                self.model._rpu_lm_head_scale_keepalive = lm_scale
                torch.ops.rpu.causal_decoder_set_lm_head(
                    handle,
                    lm_w,
                    lm_scale,
                    fuse_prefill_last_lm_head,
                    lm_head_exact_candidates,
                )
            else:
                lm_w = self.model.lm_head.weight.detach().to(
                    torch.float16
                ).contiguous()
                self.model._rpu_lm_head_w_keepalive = lm_w
                torch.ops.rpu.causal_decoder_set_lm_head(handle, lm_w)

            if delivery_w8_text:
                text_model._rpu_text_delivery_decode_descriptors = {
                    position: tuple(_text_decode_execution_plan(
                        text_model, handle, position=position,
                        graph_cache=text_model._rpu_text_graph_cache,
                    ).selected.stage_tuple.physical_descriptor)
                    for position in range(576, 580)
                }
            else:
                # Step E planned the decoder before lm_head changed its
                # physical topology. The ordinary composite forward dispatches
                # this generic descriptor directly; refresh it before ready.
                _text_decode_execution_plan(
                    text_model, handle,
                    graph_cache=text_model._rpu_text_graph_cache,
                )

            # ---------- Step G: top-level forward replacement ---------------
            self.model.forward = types.MethodType(_rpu_qwen3vl_forward, self.model)

            self._publish_execution_views(
                (
                    self._rpu_execution,
                    self._text_execution,
                    self._vision_execution,
                ),
                generation=self._execution_generation,
                component_generations=self._component_generations,
            )

            self.model._rpu_swizzled = True
            self._rpu_is_ready = True
            self._rpu_w8a16_staged_plan = None
            self._forward_snapshots = tuple(forward_snapshots)
            vars(self.model)["_qwen3_vl_forward_snapshots"] = (
                self._forward_snapshots
            )

            _LOG.info(
                "Qwen3VLAdapter ready: text_handle=%d, vision_handle=%d, "
                "deepstack_layers=%s, lm_head=%s, exact_candidates=%d",
                handle,
                vision_model._rpu_vision_handle,
                deepstack_lang_layers,
                "w4a16" if lm_head_w4a16 or runtime_w4 else
                ("w8a16" if runtime_align_quantized else "fp16"),
                lm_head_exact_candidates,
            )

            return self.model
        except BaseException as error:
            if mutation_started:
                try:
                    _cleanup_failed_qwen3_vl_install(
                        self, self.model, text_model, vision_model, forward_snapshots)
                except BaseException as cleanup_error:
                    _retain_failed_retirement(self, cleanup_error)
                    error.add_note(f"Qwen3-VL cleanup also failed: {cleanup_error}")
            elif claim_acquired:
                vars(self.model).pop("_qwen3_vl_retirement_owner", None)
                vars(self.model).pop("_rpu_swizzle_started", None)
                self._gc_retirement_enabled = False
                _release_live_instance(self.model)
                claim_acquired = False
            raise
        finally:
            _SWIZZLE_LOCK.release()


# ─────────────────────────────────────────────────────────────────────────────
# Top-level forward replacement
# ─────────────────────────────────────────────────────────────────────────────

def _split_image_embeds_per_image(
    pooler_output: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Validate HF per-image row counts without split/re-concatenate copies."""
    split_sizes = (image_grid_thw.prod(-1) // (spatial_merge_size ** 2)).tolist()
    expected_rows = sum(int(size) for size in split_sizes)
    if pooler_output.shape[0] != expected_rows:
        raise RuntimeError(
            f"Qwen3VL RPU visual rows {pooler_output.shape[0]} "
            f"!= grid-derived rows {expected_rows}"
        )
    return pooler_output


def _tensor_content_key(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    cpu = tensor.detach().cpu().contiguous()
    return (tuple(cpu.shape), str(cpu.dtype), cpu.numpy().tobytes())


def _mrope_position_ids_cpu_il(position_ids: torch.Tensor) -> torch.Tensor:
    """Normalize HF M-RoPE positions to the C++ ``[seq, 3]`` contract."""
    position_ids_cpu = position_ids.detach().to("cpu")
    if position_ids_cpu.dim() == 3:
        if position_ids_cpu.size(0) != 3 or position_ids_cpu.size(1) != 1:
            raise ValueError(
                "Qwen3VL RPU position_ids must be [3, batch=1, seq_len], "
                f"got {tuple(position_ids_cpu.shape)}"
            )
        position_ids_cpu = position_ids_cpu.squeeze(1).transpose(0, 1)
    elif position_ids_cpu.dim() != 2 or position_ids_cpu.size(1) != 3:
        raise ValueError(
            "Qwen3VL RPU position_ids must normalize to [seq_len, 3], "
            f"got {tuple(position_ids_cpu.shape)}"
        )
    return position_ids_cpu.to(torch.int32).contiguous()


# The planner moved to runtime/decoder.py when the plain text decoder started
# padding too — ONE implementation, so the VL and text paths cannot drift.
# Kept under the old name because it is this adapter's internal vocabulary.
_plan_causal_prefill_execution = plan_bounded_prefill_execution


# Shared with multi-component policies using the same M-RoPE text decoder.
from rpu_backend.runtime.decoder import pad_mrope_prefill_inputs as _pad_causal_prefill_inputs


def _restore_logical_prefill(
    raw: torch.Tensor,
    cache: RPUCache,
    start_position: int,
    logical_len: int,
    execution_len: int,
) -> torch.Tensor:
    if execution_len == logical_len:
        return raw
    cache.reset_to_position(start_position + logical_len)
    return raw[:, :logical_len]


def _qwen3_vl_text_graph_signature(
    text_model,
    logical_len: int,
    execution_len: int,
    position: int,
    planned_chunk_size: int,
    plan_result=None,
):
    """Key text decode by layout, while allowing position-stable replay."""
    dyn_dims = [
        text_model._rpu_text_num_layers,
        text_model._rpu_text_deepstack_hash,
        chunk_policy_key(text_model._rpu_decoder_handle),
        int(planned_chunk_size),
        int(getattr(text_model, "_fmb_execution_generation", 0)),
    ]
    topology = getattr(text_model, "_rpu_decoder_topology", None)
    if topology is not None:
        dyn_dims.extend(topology.identity())
    fused_decode_position = getattr(
        text_model, "_rpu_text_fused_decode_position", False
    )
    if logical_len != 1 or fused_decode_position:
        # Multi-token continuation planning depends on its absolute cache base.
        # The ordinary single-token route refreshes position-derived state on
        # REPLAY. The exact delivery's fast replay retains position registers,
        # even when its individual operator fusions are disabled.
        dyn_dims.append(int(position))
    if plan_result is not None:
        dyn_dims.extend(plan_result.graph_key_words())
    return rpu_backend.graph.GraphSignature(
        op_id="qwen3vl_text",
        shapes=[
            int(logical_len),
            int(execution_len),
            text_model._rpu_text_hidden_size,
        ],
        dyn_dims=dyn_dims,
        dtypes=[torch.float16],
    )


def _qwen3_vl_prefill_plan(text_model, cache, logical_len, position):
    """Plan initial or continuing M-RoPE prefill without advancing the cache."""
    stage = getattr(text_model, "_rpu_execution", {}).get("prefill", {})
    physical_limit = min(
        int(cache.max_seq_len) - int(position),
        int(cache.sKeyVx) * int(cache.sKeyChunk),
        int(cache.sValVx) * int(cache.sValChunk),
        _QWEN3_VL_MROPE_KEEPALIVE_ROWS,
    )
    handle = text_model._rpu_decoder_handle
    rope_mode = int(getattr(text_model, "_qwen3_vl_delivery_profile", None)
                    == _QWEN3_VL_DELIVERY_PROFILE)
    plan_box = {}
    execution_len, chunk = _plan_causal_prefill_execution(
        logical_len, physical_limit,
        int(stage.get("padding_budget", _QWEN3_VL_PREFILL_PADDING_BUDGET)),
        resolve_stage_domain=lambda n: (
            torch.ops.rpu.causal_decoder_resolve_prefill_stage_domain(
                handle, n, position, True, 0, rope_mode, 1, logical_len)),
        position=position,
        # Initial multimodal input keeps its existing physical alignment.
        # Continuations search from their actual length, with optional padding.
        alignment=16 if position == 0 else 1,
        padding_rows=stage.get("padding_rows", "auto"),
        exact_chunk_size=(stage.get("chunk_size")
                          if isinstance(stage.get("chunk_size"), int) else None),
        queue_owner_id=int(handle), execution_owner=text_model,
        execution_stage="prefill", execution_native=("causal_decoder", int(handle)),
        plan_result_sink=lambda result: plan_box.__setitem__("result", result),
        plan_signature=(True, 0, rope_mode, 1),
        graph_cache=text_model._rpu_text_graph_cache,
    )
    return execution_len, chunk, plan_box["result"]


@execution_serialized
def _rpu_qwen3vl_forward(
    self,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: Any | None = None,
    inputs_embeds: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    cache_position: torch.Tensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    **kwargs,
):
    """RPU-dispatching forward for Qwen3VLForConditionalGeneration.

    Mirrors HF semantics:
      - prefill with `pixel_values` → run vision encoder, scatter image embeds
        into inputs_embeds (CPU fallback for masked_scatter), build dense
        DeepStack visual embeds, run text decoder with DeepStack injection,
        apply lm_head.
      - decode / text-only → zero keepalive views for DeepStack, plain text
        decoder, apply lm_head.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLCausalLMOutputWithPast,
    )

    precomputed_vision = kwargs.pop("rpu_precomputed_vision", None)
    if getattr(self, "_qwen3_vl_delivery_profile", None) == _QWEN3_VL_DELIVERY_PROFILE:
        length = (int(input_ids.shape[1]) if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2
                  else int(inputs_embeds.shape[1]) if isinstance(inputs_embeds, torch.Tensor) and inputs_embeds.ndim == 3 else -1)
        position = int(getattr(past_key_values, "position", -1))
        if (pixel_values_videos is not None or video_grid_thw is not None
                or not isinstance(logits_to_keep, int) or logits_to_keep != 1
                or getattr(past_key_values, "max_seq_len", None) != 608
                or not ((length == 576 and position == 0) or
                        (length == 1 and 576 <= position <= 579))):
            raise UnsupportedModelError("fixed delivery requires P576, four decode positions 576..579 and logits_to_keep=1; video is unsupported")
        if length == 576:
            expected_grid = torch.tensor([[1, 24, 24]] * 3, dtype=torch.int64)
            if (not isinstance(image_grid_thw, torch.Tensor) or
                    not torch.equal(image_grid_thw.detach().cpu().to(torch.int64), expected_grid)):
                raise UnsupportedModelError("fixed delivery prefill requires exactly three [1,24,24] image grids")

    if labels is not None:
        raise NotImplementedError(
            "Qwen3VL RPU forward: `labels` (loss) not supported."
        )
    if output_attentions:
        raise NotImplementedError(
            "Qwen3VL RPU forward: output_attentions=True not supported."
        )
    if output_hidden_states:
        raise NotImplementedError(
            "Qwen3VL RPU forward: output_hidden_states=True not supported."
        )
    if use_cache is False:
        raise NotImplementedError(
            "Qwen3VL RPU forward requires use_cache=True."
        )
    if return_dict is False:
        raise NotImplementedError(
            "Qwen3VL RPU forward requires return_dict=True."
        )

    integer_dtypes = {
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    }
    if input_ids is not None:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                "Qwen3VL RPU forward: input_ids must be a torch.Tensor."
            )
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError(
                "Qwen3VL RPU forward: input_ids must have shape [1, seq] "
                "with seq >= 1."
            )
        if input_ids.dtype not in integer_dtypes:
            raise TypeError(
                "Qwen3VL RPU forward: input_ids must use an integer dtype, "
                f"got {input_ids.dtype}."
            )
    if inputs_embeds is not None:
        if not isinstance(inputs_embeds, torch.Tensor):
            raise TypeError(
                "Qwen3VL RPU forward: inputs_embeds must be a torch.Tensor."
            )
        if (inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1
                or inputs_embeds.shape[1] < 1):
            raise ValueError(
                "Qwen3VL RPU forward: inputs_embeds must have shape "
                "[1, seq, hidden] with seq >= 1."
            )

    # The fused text runner is unconditionally causal. Its native entry point
    # deliberately ignores a supplied mask in that mode, so accepting a mask
    # with zeros would let HF use it for M-RoPE positions while attention uses
    # different semantics. Internal execution padding is appended later and is
    # protected by causality; caller-visible masking is not supported here.
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError(
                "Qwen3VL RPU forward: attention_mask must be a torch.Tensor."
            )
        if attention_mask.ndim != 2:
            raise NotImplementedError(
                "Qwen3VL RPU forward supports only a 2D all-ones "
                "attention_mask; additive/4D masks are not consumed by the "
                "causal fused decoder."
            )
        if input_ids is not None and tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError(
                "Qwen3VL RPU forward: attention_mask shape "
                f"{tuple(attention_mask.shape)} != input_ids shape "
                f"{tuple(input_ids.shape)}."
            )
        if not bool(torch.all(attention_mask != 0).item()):
            raise NotImplementedError(
                "Qwen3VL RPU forward supports only an all-ones caller "
                "attention_mask; masked/padded caller tokens would be "
                "silently ignored by the causal fused decoder."
            )

    if not isinstance(past_key_values, RPUCache):
        raise TypeError(
            "Qwen3VL RPU forward: past_key_values must be an RPUCache instance. "
            "Use RPUCache.from_model(model.model.language_model, ...) or build manually."
        )
    if pixel_values is not None and pixel_values_videos is not None:
        raise NotImplementedError(
            "Qwen3VL RPU forward: simultaneous image + video inputs not "
            "supported.  Pass only one modality per forward."
        )
    if precomputed_vision is not None and (
        pixel_values is not None or pixel_values_videos is not None
    ):
        raise ValueError(
            "Qwen3VL RPU forward: rpu_precomputed_vision is mutually exclusive "
            "with pixel_values and pixel_values_videos."
        )

    text_model = self.model.language_model
    vision_model = self.model.visual
    text_cfg = self.config.text_config
    topology = getattr(text_model, "_rpu_decoder_topology", None)
    if topology is not None:
        if _qwen3_vl_core_topology(self.config, self._rpu_execution) != topology:
            raise ValueError("Qwen3-VL forward topology differs from its cold model config")
        request = inputs_embeds if inputs_embeds is not None else input_ids
        validate_decoder_cache_topology(topology, past_key_values,
            batch_size=request.shape[0] if request is not None else None)
        if pixel_values_videos is not None:
            raise NotImplementedError("reduced-core Qwen3-VL admits still images and text only")
        if (getattr(self.lm_head, "_rpu_linear_partition", None),
                getattr(self.lm_head, "_rpu_linear_num_cores", None)) != (1, topology.lm_head_tp):
            raise ValueError("Qwen3-VL lm_head weight layout differs from its cold topology")
    hidden_size = text_cfg.hidden_size
    spatial_merge_size = vision_cfg_or(self).spatial_merge_size

    # ----- Vision encoder pass (image OR video) ----------------------------
    # HF `get_video_features` is literally `get_image_features` — same vision
    # tower, same per-row encoder loop.  The grid_thw form differs: vision
    # tower takes T-folded form for both modalities (`[[1,h,w]]` for image,
    # `[[grid_t,h,w]]` for video; per-frame expansion is rope-only).
    deepstack_features: list[Any] | None = None
    visual_embeds_flat: Any | None = None
    visual_token_id: int | None = None
    cached_prefix_parts = 0
    if pixel_values is not None:
        if image_grid_thw is None:
            raise ValueError(
                "Qwen3VL RPU forward: image_grid_thw must accompany pixel_values."
            )
        pix_in = pixel_values
        grid_in = image_grid_thw
        visual_token_id = self.config.image_token_id
    elif pixel_values_videos is not None:
        if video_grid_thw is None:
            raise ValueError(
                "Qwen3VL RPU forward: video_grid_thw must accompany pixel_values_videos."
            )
        pix_in = pixel_values_videos
        grid_in = video_grid_thw
        visual_token_id = self.config.video_token_id
    else:
        pix_in = None
        grid_in = None

    if pix_in is not None:
        with torch.no_grad():
            vision_kwargs = ({"_rpu_keep_merged_fp16": True}
                             if getattr(vision_model, "_rpu_vision_merger_on_device", False)
                             else {})
            vision_output = vision_model(pix_in, grid_thw=grid_in, **vision_kwargs)
        # RPU mergers keep their FP16 values for the following text scatter;
        # CPU-only mergers retain their original FP32 output until that cast.
        pooler = vision_output.pooler_output
        deepstack_features = list(vision_output.deepstack_features)
        # Per-item split (single image / single video → identity).
        visual_embeds_flat = _split_image_embeds_per_image(
            pooler, grid_in.detach().cpu(), spatial_merge_size,
        )
    elif precomputed_vision is not None:
        if image_grid_thw is None:
            raise ValueError(
                "Qwen3VL RPU forward: image_grid_thw must accompany "
                "rpu_precomputed_vision."
            )
        if not isinstance(precomputed_vision, dict):
            raise TypeError(
                "Qwen3VL RPU forward: rpu_precomputed_vision must be a dict."
            )
        standard_keys = {"pooler_output", "deepstack_features"}
        parts_keys = {"pooler_output_parts", "deepstack_feature_parts"}
        has_standard = standard_keys <= set(precomputed_vision)
        has_parts = parts_keys <= set(precomputed_vision)
        if has_standard == has_parts:
            raise ValueError(
                "Qwen3VL RPU forward: rpu_precomputed_vision requires exactly "
                "one of pooler_output/deepstack_features or "
                "pooler_output_parts/deepstack_feature_parts."
            )
        if has_parts:
            cached_prefix_parts = int(
                precomputed_vision.get("cached_prefix_parts", 0)
            )
            visual_embeds_flat = list(
                precomputed_vision["pooler_output_parts"]
            )
            deepstack_features = [
                list(parts)
                for parts in precomputed_vision["deepstack_feature_parts"]
            ]
            if not visual_embeds_flat:
                raise ValueError(
                    "Qwen3VL RPU pooler_output_parts must be non-empty."
                )
            if not 0 <= cached_prefix_parts < len(visual_embeds_flat):
                raise ValueError(
                    "Qwen3VL RPU cached_prefix_parts must be in "
                    "[0, len(pooler_output_parts))"
                )
            expected_rows = int((
                image_grid_thw.detach().cpu().prod(-1)
                // (spatial_merge_size ** 2)
            ).sum().item())
            actual_rows = sum(
                int(part.shape[0]) for part in visual_embeds_flat
            )
            if actual_rows != expected_rows:
                raise RuntimeError(
                    f"Qwen3VL RPU visual part rows {actual_rows} "
                    f"!= grid-derived rows {expected_rows}"
                )
        else:
            visual_embeds_flat = _split_image_embeds_per_image(
                precomputed_vision["pooler_output"],
                image_grid_thw.detach().cpu(),
                spatial_merge_size,
            )
            deepstack_features = list(
                precomputed_vision["deepstack_features"]
            )
        visual_token_id = self.config.image_token_id

    # ----- Build inputs_embeds + scatter image features --------------------
    input_ids_cpu = None
    inputs_embeds_in_stable_slot = False
    input_slots = getattr(text_model, "_rpu_qwen3vl_input_slots", None)
    if input_slots is None:
        input_slots = {}
        text_model._rpu_qwen3vl_input_slots = input_slots

    if inputs_embeds is None:
        if input_ids is None:
            raise ValueError(
                "Qwen3VL RPU forward: must provide input_ids or inputs_embeds."
            )
        embed_tokens = self.get_input_embeddings()
        input_slot_key = (
            tuple(input_ids.shape) + (embed_tokens.embedding_dim,),
            torch.float16,
        )
        input_slot = input_slots.get(input_slot_key)
        if input_slot is None:
            input_slot = torch.empty(
                input_slot_key[0], device="rpu", dtype=torch.float16
            ).contiguous()
            input_slots[input_slot_key] = input_slot
        direct_decode_ids = (
            input_ids.device.type == "rpu" and input_ids.dtype == torch.int64
            and input_ids.is_contiguous() and input_ids.numel() == 1
            and visual_embeds_flat is None
            and hasattr(self, "_rpu_lm_head_raw_fp16_cpu")
        )
        if direct_decode_ids:
            gather_ids = input_ids.view(-1)
        else:
            input_ids_cpu = input_ids.to(
                "cpu", dtype=torch.int64
            ).contiguous()
            gather_ids = input_ids_cpu.view(-1)
        skip_token_id = visual_token_id if visual_embeds_flat is not None else -1
        torch.ops.rpu.qwen3vl_gather_embedding_from_rpu_unflushed_(
            input_slot.view(-1, embed_tokens.embedding_dim),
            embed_tokens.weight,
            gather_ids,
            skip_token_id,
        )
        inputs_embeds = input_slot
        inputs_embeds_in_stable_slot = True

    if inputs_embeds.shape[-1] != hidden_size:
        raise ValueError(
            "Qwen3VL RPU forward: inputs_embeds hidden dimension "
            f"{inputs_embeds.shape[-1]} != model hidden_size {hidden_size}."
        )

    inputs_embeds = inputs_embeds.to(dtype=torch.float16)
    if not inputs_embeds.is_contiguous():
        inputs_embeds = inputs_embeds.contiguous()

    if inputs_embeds.device.type != "rpu":
        inputs_embeds = inputs_embeds.to(
            device="rpu", dtype=torch.float16
        ).contiguous()

    # Graph BUILD records the decoder input address. Keep one stable slot per
    # shape and refresh caller-supplied embeddings before entering capture.
    if not inputs_embeds_in_stable_slot:
        input_slot_key = (tuple(inputs_embeds.shape), inputs_embeds.dtype)
        input_slot = input_slots.get(input_slot_key)
        if input_slot is None:
            input_slot = inputs_embeds.clone().contiguous()
            input_slots[input_slot_key] = input_slot
        else:
            input_slot.copy_(inputs_embeds)
        inputs_embeds = input_slot

    if visual_embeds_flat is None:
        previous_prefix = getattr(self, "_rpu_visual_cached_prefix", None)
        if previous_prefix is not None and previous_prefix[0][0] == inputs_embeds.data_ptr():
            self._rpu_visual_cached_prefix = None
    if visual_embeds_flat is not None:
        if input_ids is None:
            raise ValueError(
                "Qwen3VL RPU forward: visual scatter requires input_ids "
                "(needed to locate image/video token positions)."
            )
        if input_ids_cpu is None:
            input_ids_cpu = input_ids.to("cpu", dtype=torch.int64).contiguous()
        visual_mask = input_ids_cpu == visual_token_id
        row_indices_cpu = (
            visual_mask.squeeze(0).nonzero().flatten().to(dtype=torch.int64)
        ).contiguous()
        n_visual_tokens = row_indices_cpu.numel()
        if isinstance(visual_embeds_flat, (list, tuple)):
            visual_parts = [
                part.detach().to(dtype=torch.float16).contiguous()
                for part in visual_embeds_flat
            ]
            visual_rows = sum(int(part.shape[0]) for part in visual_parts)
            if visual_rows != n_visual_tokens:
                raise RuntimeError(
                    f"Qwen3VL RPU visual part rows {visual_rows} "
                    f"!= visual token count {n_visual_tokens}"
                )
            scatter_parts = visual_parts
            scatter_indices = row_indices_cpu
            previous_prefix = getattr(self, "_rpu_visual_cached_prefix", None)
            self._rpu_visual_cached_prefix = None
            pending_prefix = None
            if cached_prefix_parts:
                prefix_rows = sum(
                    int(part.shape[0])
                    for part in visual_parts[:cached_prefix_parts]
                )
                prefix_sources = visual_parts[:cached_prefix_parts]
                cacheable_prefix = all(not torch.is_inference(part) for part in prefix_sources)
                prefix_signature = (
                    int(inputs_embeds.data_ptr()),
                    row_indices_cpu.numpy().tobytes(),
                    tuple(
                        (int(part.data_ptr()), tuple(part.shape), str(part.dtype),
                         int(part._version) if cacheable_prefix else None)
                        for part in prefix_sources
                    ),
                )
                if (cacheable_prefix and inputs_embeds_in_stable_slot
                        and previous_prefix is not None
                        and previous_prefix[0] == prefix_signature):
                    scatter_parts = visual_parts[cached_prefix_parts:]
                    scatter_indices = row_indices_cpu[prefix_rows:].contiguous()
                if cacheable_prefix:
                    # Retain source storage as well as its version: a recycled
                    # data_ptr must not look like an unchanged cached frame.
                    pending_prefix = (prefix_signature, prefix_sources)
            torch.ops.rpu.qwen3vl_scatter_row_parts_unflushed_(
                inputs_embeds.squeeze(0), scatter_indices, scatter_parts
            )
            # A rejected suffix must not certify a prefix that was never copied.
            self._rpu_visual_cached_prefix = pending_prefix
        else:
            self._rpu_visual_cached_prefix = None
            visual_embeds_cpu = visual_embeds_flat.to(
                "cpu", dtype=torch.float16
            ).contiguous()
            if visual_embeds_cpu.shape[0] != n_visual_tokens:
                raise RuntimeError(
                    f"Qwen3VL RPU forward: visual_embeds rows "
                    f"{visual_embeds_cpu.shape[0]} != visual token count "
                    f"{n_visual_tokens}"
                )
            torch.ops.rpu.qwen3vl_scatter_row_parts_unflushed_(
                inputs_embeds.squeeze(0), row_indices_cpu, [visual_embeds_cpu]
            )

    seq_len = inputs_embeds.shape[1]
    logical_seq_len = seq_len
    start_position = int(past_key_values.position)
    if cache_position is not None:
        if (not isinstance(cache_position, torch.Tensor)
                or cache_position.dtype not in integer_dtypes):
            raise TypeError(
                "Qwen3VL RPU forward: cache_position must be an integer tensor."
            )
        got_cache_position = cache_position.reshape(-1).to(
            "cpu", dtype=torch.long
        )
        expected_cache_position = torch.arange(
            start_position,
            start_position + logical_seq_len,
            dtype=torch.long,
        )
        if not torch.equal(got_cache_position, expected_cache_position):
            raise NotImplementedError(
                "Qwen3VL RPU forward: cache_position disagrees with the RPU "
                f"cache; expected {expected_cache_position.tolist()}, got "
                f"{got_cache_position.tolist()}."
            )
    if start_position + logical_seq_len > int(past_key_values.max_seq_len):
        raise ValueError(
            "Qwen3VL RPU forward: logical sequence exceeds KV-cache horizon: "
            f"position={start_position}, seq_len={logical_seq_len}, "
            f"max_seq_len={past_key_values.max_seq_len}"
        )

    execution_seq_len = logical_seq_len
    planned_chunk_size: int | None = None
    prefill_plan_result = None
    prefill_cfg = getattr(
        text_model, "_rpu_execution", {}
    ).get("prefill", {})
    spans = continuation_segments(
        start_position, logical_seq_len, prefill_cfg,
        capacity=past_key_values.max_seq_len,
    )
    if spans is not None and all(value is None for value in (
        pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw,
        mm_token_type_ids, precomputed_vision,
    )):
        if attention_mask is not None and tuple(attention_mask.shape) != (1, logical_seq_len):
            raise ValueError("Qwen3VL continuation attention_mask must cover every logical token")
        if position_ids is not None:
            if not isinstance(position_ids, torch.Tensor) or position_ids.dtype not in integer_dtypes:
                raise TypeError("Qwen3VL continuation position_ids must be an integer tensor")
            normalized_positions = _mrope_position_ids_cpu_il(position_ids)
            if tuple(normalized_positions.shape) != (logical_seq_len, 3):
                raise ValueError("Qwen3VL continuation position_ids must cover every logical token")
            if bool((normalized_positions < 0).any().item()):
                raise ValueError("Qwen3VL continuation position_ids must be non-negative")
        # Preflight every physical body before the first single-token write.
        for offset, count in spans:
            if count > 1:
                _qwen3_vl_prefill_plan(text_model, past_key_values, count,
                                      start_position + offset)
        _text_decode_execution_plan(text_model, text_model._rpu_decoder_handle,
                                    graph_cache=text_model._rpu_text_graph_cache)
        return run_continuation_segments(
            lambda **call: _rpu_qwen3vl_forward(self, **call),
            past_key_values, text_model, spans,
            dict(input_ids=input_ids,
                 inputs_embeds=inputs_embeds,
                 attention_mask=attention_mask, position_ids=position_ids,
                 cache_position=cache_position, past_key_values=past_key_values,
                 logits_to_keep=logits_to_keep, use_cache=use_cache,
                 output_attentions=output_attentions,
                 output_hidden_states=output_hidden_states, return_dict=return_dict,
                 **kwargs),
        )
    if logical_seq_len > 1:
        execution_seq_len, planned_chunk_size, prefill_plan_result = (
            _qwen3_vl_prefill_plan(text_model, past_key_values,
                                  logical_seq_len, start_position)
        )

    # ----- position_ids: compute via get_rope_index when needed ------------
    prefill_position_cache = None
    decode_position_slot_key = None
    decode_position_content_key = None
    decode_position_prepared = None
    if position_ids is None:
        if (
            input_ids is not None
            and mm_token_type_ids is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
        ):
            # HF Qwen3-VL get_rope_index needs per-frame expansion of
            # video_grid_thw (see _expand_video_grid_per_frame docstring).
            input_ids_position_cpu = (
                input_ids_cpu
                if input_ids_cpu is not None else input_ids.to("cpu")
            )
            mm_token_type_ids_cpu = mm_token_type_ids.to("cpu")
            image_grid_thw_cpu = (
                image_grid_thw.detach().cpu()
                if image_grid_thw is not None else None
            )
            attention_mask_cpu = (
                attention_mask.to("cpu")
                if attention_mask is not None else None
            )
            vgt_rope = (
                _expand_video_grid_per_frame(video_grid_thw.detach().cpu())
                if video_grid_thw is not None else None
            )
            position_key = (
                _tensor_content_key(input_ids_position_cpu),
                _tensor_content_key(mm_token_type_ids_cpu),
                _tensor_content_key(image_grid_thw_cpu),
                _tensor_content_key(vgt_rope),
                _tensor_content_key(attention_mask_cpu),
            )
            prefill_position_cache = getattr(
                text_model, "_rpu_qwen3vl_prefill_position_cache", None
            )
            if prefill_position_cache is None \
                    or prefill_position_cache["key"] != position_key:
                pos_ids_cpu, rope_deltas = self.model.get_rope_index(
                    input_ids_position_cpu,
                    mm_token_type_ids=mm_token_type_ids_cpu,
                    image_grid_thw=image_grid_thw_cpu,
                    video_grid_thw=vgt_rope,
                    attention_mask=attention_mask_cpu,
                )
                prefill_position_cache = {
                    "key": position_key,
                    "position_ids": pos_ids_cpu.contiguous(),
                    "rope_deltas": rope_deltas,
                }
                text_model._rpu_qwen3vl_prefill_position_cache = (
                    prefill_position_cache
                )
            else:
                pos_ids_cpu = prefill_position_cache["position_ids"]
                rope_deltas = prefill_position_cache["rope_deltas"]
            self.model.rope_deltas = rope_deltas
            position_ids = pos_ids_cpu  # [3, batch, seq_len] CPU
        else:
            # text-only or pre-cached image continuation: arange-expanded.
            batch_size = inputs_embeds.shape[0]
            delta_key = _tensor_content_key(self.model.rope_deltas)
            decode_position_cache = getattr(
                text_model, "_rpu_qwen3vl_decode_position_prepared", None
            )
            if decode_position_cache is None:
                decode_position_cache = {}
                text_model._rpu_qwen3vl_decode_position_prepared = (
                    decode_position_cache
                )
            prepared_key = (
                int(past_key_values.position), int(seq_len), int(batch_size),
                delta_key,
            )
            decode_position_prepared = decode_position_cache.get(prepared_key)
            if decode_position_prepared is None:
                if len(decode_position_cache) >= _MROPE_STABLE_SLOT_LIMIT:
                    decode_position_cache.clear()
                arange = torch.arange(
                    past_key_values.position,
                    past_key_values.position + seq_len,
                    dtype=torch.long,
                )
                position_ids = arange.view(1, 1, -1).expand(
                    3, batch_size, -1
                )
                if self.model.rope_deltas is not None:
                    position_ids = position_ids + self.model.rope_deltas
                position_ids_cpu_il = _mrope_position_ids_cpu_il(position_ids)
                partial_position_ids_cpu_il = position_ids_cpu_il.to(torch.int64)
                decode_position_prepared = {
                    "position_ids": position_ids,
                    "position_ids_cpu_il": position_ids_cpu_il,
                    "partial_position_ids_cpu_il": partial_position_ids_cpu_il,
                    "rope_key": _tensor_content_key(partial_position_ids_cpu_il),
                    "content_key": _tensor_content_key(position_ids),
                }
                decode_position_cache[prepared_key] = decode_position_prepared
            else:
                position_ids = decode_position_prepared["position_ids"]
            decode_position_slot_key = (
                int(past_key_values.position), tuple(position_ids.shape)
            )
            decode_position_content_key = decode_position_prepared["content_key"]

    inputs_embeds, attention_mask, position_ids = _pad_causal_prefill_inputs(
        inputs_embeds, attention_mask, position_ids, execution_seq_len,
    )

    # ----- Build dense_visual_embeds (DeepStack injection) -----------------
    if deepstack_features is not None and visual_token_id is not None:
        dense_visual_embeds = scatter_visual_embeds_to_dense(
            deepstack_features,
            visual_mask.squeeze(0),
            logical_seq_len,
            hidden_size,
            cache_owner=text_model,
            row_indices_cpu=row_indices_cpu,
            cached_prefix_parts=cached_prefix_parts,
            execution_len=execution_seq_len,
        )
    else:
        # Zero-copy views of the persistent 8192-row keepalive: request the
        # final execution shape directly instead of concatenating three zeros.
        dense_visual_embeds = make_zero_visual_embeds(
            text_model, execution_seq_len,
        )
    seq_len = execution_seq_len

    # ----- Run text decoder (RPU) ------------------------------------------
    # Bypass `rpu_decoder_model_forward` (the base patch installed by
    # `_install_causal_decoder_forward`) and call the runner directly. The
    # base patch asserts `raw.last_dim == hidden_size`, but with fused
    # lm_head enabled (causal_decoder_set_lm_head, step F), the C++ side
    # returns `[B, 1, vocab_size]` logits for decode (seq_len=1). Pattern
    # matches qwen3.py:`_apply_fused_lm_head_for_rpu`.
    #
    # R-Phase 7 (perf): wrap in graph_cache.capture(sig) so all RPU
    # kernel dispatches inside `_run_causal_decoder_forward` batch into
    # one queue execution. Position is deliberately absent only for single-token
    # decode: the generic decoder refreshes every position-derived register/DMA
    # field on REPLAY, so successive `seq_len == 1` steps share one entry.
    # Multi-token continuation keeps position because its chunk/SDPA plan can
    # change with the absolute start. Position preparation stays outside capture
    # so each forward refreshes its source values. The caller mask was validated
    # as all-valid above and remains available to CPU M-RoPE/padding preparation.
    # The native consumer is always causal and ignores an explicit mask, so do
    # not upload it (including for ordinary, non-delivery profiles).
    attn_rpu = None
    if decode_position_slot_key is not None and logical_seq_len == execution_seq_len:
        position_ids_cpu_il = decode_position_prepared["position_ids_cpu_il"]
    else:
        position_ids_cpu_il = (
            _mrope_position_ids_cpu_il(position_ids)
            if position_ids is not None else None
        )
    if position_ids_cpu_il is None:
        pos_rpu = None
    elif prefill_position_cache is not None:
        position_slots = getattr(
            text_model, "_rpu_qwen3vl_prefill_position_slots", None
        )
        if position_slots is None:
            position_slots = {}
            text_model._rpu_qwen3vl_prefill_position_slots = position_slots
        position_slot_key = (
            tuple(position_ids_cpu_il.shape), position_ids_cpu_il.dtype
        )
        pos_rpu = position_slots.get(position_slot_key)
        if pos_rpu is None:
            pos_rpu = position_ids_cpu_il.to("rpu").contiguous()
            position_slots[position_slot_key] = pos_rpu
        elif prefill_position_cache.get("position_ids_rpu") is not pos_rpu:
            pos_rpu.copy_(position_ids_cpu_il)
        prefill_position_cache["position_ids_rpu"] = pos_rpu
    elif decode_position_slot_key is not None and logical_seq_len == execution_seq_len:
        # Prepared text positions contain P rows. Padded requests retain the
        # E-row positions computed above in the general stable slots.
        decode_position_slots = getattr(
            text_model, "_rpu_qwen3vl_decode_position_slots", None
        )
        if decode_position_slots is None:
            decode_position_slots = {}
            text_model._rpu_qwen3vl_decode_position_slots = decode_position_slots
        decode_position_entry = decode_position_slots.get(
            decode_position_slot_key
        )
        position_ids_cpu_il = decode_position_prepared["position_ids_cpu_il"]
        if decode_position_entry is None:
            if len(decode_position_slots) >= _MROPE_STABLE_SLOT_LIMIT:
                decode_position_slots.clear()
            pos_rpu = position_ids_cpu_il.to("rpu").contiguous()
            decode_position_slots[decode_position_slot_key] = {
                "content_key": decode_position_content_key,
                "position_ids_rpu": pos_rpu,
            }
        else:
            pos_rpu = decode_position_entry["position_ids_rpu"]
            if decode_position_entry["content_key"] != decode_position_content_key:
                pos_rpu.copy_(position_ids_cpu_il)
                decode_position_entry["content_key"] = decode_position_content_key
    else:
        position_slots = getattr(
            text_model, "_rpu_qwen3vl_position_slots", None
        )
        if position_slots is None:
            position_slots = {}
            text_model._rpu_qwen3vl_position_slots = position_slots
        position_slot_key = (
            tuple(position_ids_cpu_il.shape), position_ids_cpu_il.dtype
        )
        position_content_key = _tensor_content_key(position_ids_cpu_il)
        position_entry = position_slots.get(position_slot_key)
        if position_entry is None:
            pos_rpu = position_ids_cpu_il.to("rpu").contiguous()
            position_slots[position_slot_key] = {
                "content_key": position_content_key,
                "tensor": pos_rpu,
            }
        else:
            pos_rpu = position_entry["tensor"]
            if position_entry["content_key"] != position_content_key:
                pos_rpu.copy_(position_ids_cpu_il)
                position_entry["content_key"] = position_content_key

    rope_cos_il = None
    rope_sin_il = None
    partial_mrope_enabled = getattr(text_model, "_qwen3_vl_delivery_profile", None) == _QWEN3_VL_DELIVERY_PROFILE

    if position_ids_cpu_il is not None and partial_mrope_enabled:
        if decode_position_prepared is not None:
            pos_il_cpu = decode_position_prepared[
                "partial_position_ids_cpu_il"
            ]
            rope_key = decode_position_prepared["rope_key"]
        else:
            pos_il_cpu = position_ids_cpu_il.to(torch.int64)
            rope_key = _tensor_content_key(pos_il_cpu)
        rope_cache = getattr(text_model, "_rpu_qwen3vl_rope_cache", None)
        if rope_cache is None:
            rope_cache = {}
            text_model._rpu_qwen3vl_rope_cache = rope_cache
        cached_rope = rope_cache.get(rope_key)
        if cached_rope is None:
            if len(rope_cache) >= _MROPE_STABLE_SLOT_LIMIT:
                rope_cache.clear()
            head_dim = getattr(text_cfg, "head_dim", None)
            if head_dim is None:
                head_dim = text_cfg.hidden_size // text_cfg.num_attention_heads
            rope_params = getattr(text_cfg, "rope_parameters", None) or {}
            cos_cpu, sin_cpu = build_interleaved_mrope_cos_sin(
                pos_il_cpu,
                head_dim=int(head_dim),
                rope_theta=float(rope_params.get(
                    "rope_theta", getattr(text_cfg, "rope_theta", 10000.0)
                )),
                mrope_section=[
                    int(value) for value in rope_params["mrope_section"]
                ],
            )
            cached_rope = (cos_cpu.contiguous(), sin_cpu.contiguous())
            rope_cache[rope_key] = cached_rope
        rope_slots = getattr(text_model, "_rpu_qwen3vl_rope_slots", None)
        if rope_slots is None:
            rope_slots = {}
            text_model._rpu_qwen3vl_rope_slots = rope_slots
        rope_slot_key = (tuple(cached_rope[0].shape), rope_key)
        rope_entry = rope_slots.get(rope_slot_key)
        if rope_entry is None:
            if len(rope_slots) >= _MROPE_STABLE_SLOT_LIMIT:
                rope_slots.clear()
            rope_cos_il = cached_rope[0].to("rpu").contiguous()
            rope_sin_il = cached_rope[1].to("rpu").contiguous()
            rope_slots[rope_slot_key] = {
                "cos": rope_cos_il,
                "sin": rope_sin_il,
            }
        else:
            rope_cos_il = rope_entry["cos"]
            rope_sin_il = rope_entry["sin"]
    text_graph_cache = text_model._rpu_text_graph_cache
    decode_plan_result = None
    decode_descriptor = None
    if logical_seq_len == 1 and partial_mrope_enabled:
        decode_plan_result = _text_decode_execution_plan(
            text_model, text_model._rpu_decoder_handle, position=start_position,
            graph_cache=text_graph_cache)
        decode_descriptor = tuple(decode_plan_result.selected.stage_tuple.physical_descriptor)
        if decode_descriptor != text_model._rpu_text_delivery_decode_descriptors.get(start_position):
            raise RuntimeError("fixed delivery decode descriptor identity changed; reload a fresh model")
    sig = _qwen3_vl_text_graph_signature(
        text_model,
        logical_seq_len,
        seq_len,
        int(past_key_values.position),
        int(planned_chunk_size or 0),
        prefill_plan_result if prefill_plan_result is not None else decode_plan_result,
    )
    with text_graph_cache.capture(sig):
        raw, pkv = _run_causal_decoder_forward(
            text_model,
            text_model._rpu_decoder_handle,
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attn_rpu,
            position_ids=pos_rpu,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            deepstack_dense_visual_embeds=dense_visual_embeds,
            prefill_plan=(
                (execution_seq_len, planned_chunk_size, prefill_plan_result)
                if planned_chunk_size is not None else None
            ),
            rope_cos_il=rope_cos_il,
            rope_sin_il=rope_sin_il,
            decode_descriptor=decode_descriptor,
        )

    # The shared runner publishes the result of this successful native call.
    # Reuse that receipt instead of querying the same handle a second time.
    native_execution_receipt = getattr(
        text_model, "_rpu_last_execution_plan", None
    )
    resolved_chunk_size = (
        native_execution_receipt.get("chunk_size")
        if isinstance(native_execution_receipt, dict) else None
    )
    if (type(resolved_chunk_size) is not int or resolved_chunk_size <= 0
            or (planned_chunk_size is not None
                and resolved_chunk_size != planned_chunk_size)):
        raise RuntimeError(
            "Qwen3VL causal forward has no matching dispatched chunk receipt: "
            f"planned={planned_chunk_size}, observed={resolved_chunk_size}"
        )
    # M-RoPE inputs arrive padded to E, but the shared runner uses the A6
    # winner's logical P to slice outputs and advance the cache. Preserve that
    # P/E contract when validating and publishing the root receipt.
    if (
        not isinstance(native_execution_receipt, dict)
        or native_execution_receipt.get("logical_len") != logical_seq_len
        or native_execution_receipt.get("execution_len") != seq_len
        or native_execution_receipt.get("chunk_size") != resolved_chunk_size
        or native_execution_receipt.get("position") != start_position
        or not native_execution_receipt.get("physical_descriptor")
    ):
        observed = (
            {key: native_execution_receipt.get(key) for key in (
                "logical_len", "execution_len", "chunk_size", "position", "padding_rows"
            )} if isinstance(native_execution_receipt, dict) else None
        )
        descriptor_present = bool(
            isinstance(native_execution_receipt, dict)
            and native_execution_receipt.get("physical_descriptor")
        )
        raise RuntimeError(
            "Qwen3VL text forward did not retain its dispatched physical "
            "descriptor: "
            f"expected logical_len={logical_seq_len}, execution_len={seq_len}, "
            f"chunk_size={resolved_chunk_size}, position={start_position}; "
            f"observed metadata={observed}, descriptor_present={descriptor_present}"
        )
    dispatched_descriptor = tuple(
        native_execution_receipt["physical_descriptor"]
    )
    if (
        prefill_plan_result is not None
        and dispatched_descriptor
        != tuple(prefill_plan_result.selected.stage_tuple.physical_descriptor)
    ):
        raise RuntimeError(
            "Qwen3VL text forward descriptor disagrees with its A6 winner"
        )
    if decode_descriptor is not None and dispatched_descriptor != decode_descriptor:
        raise RuntimeError("fixed delivery decode receipt disagrees with its position-specific A6 winner")
    receipt_graph_mode = native_execution_receipt.get("graph_mode")
    if prefill_plan_result is not None:
        if (
            receipt_graph_mode is not None
            and receipt_graph_mode != prefill_plan_result.graph_mode
        ):
            raise RuntimeError(
                "Qwen3VL text forward lifecycle disagrees with its A6 winner"
            )
        receipt_graph_mode = prefill_plan_result.graph_mode

    raw = _restore_logical_prefill(
        raw, pkv, start_position, logical_seq_len, seq_len,
    )
    execution_receipt = {
        "stage": "prefill" if logical_seq_len > 1 else "decode",
        "component": _QWEN3_VL_TEXT_COMPONENT,
        "generation": int(getattr(
            text_model, "_fmb_execution_generation", 0
        )),
        "logical_len": int(logical_seq_len),
        "execution_len": int(seq_len),
        "chunk_size": resolved_chunk_size,
        "padding_rows": int(seq_len - logical_seq_len),
        "position": start_position,
        "physical_descriptor": dispatched_descriptor,
        "dry_forward_agreement": (
            prefill_plan_result is not None
            and resolved_chunk_size == planned_chunk_size
        ),
    }
    if prefill_plan_result is None:
        execution_receipt.update({
            "authority": "NATIVE_RUNTIME_RESOLVED",
            "selection_scope": "DECODE_OR_UNPADDED_CONTINUATION",
        })
    else:
        execution_receipt.update({
            "authority": "NATIVE_A6_STAGE_DESCRIPTOR",
            "selection_scope": prefill_plan_result.selection_scope,
            "physical_plan_digest": (
                prefill_plan_result.physical_plan_digest
            ),
            "plan_digest": prefill_plan_result.plan_digest,
        })
    if receipt_graph_mode is not None:
        execution_receipt["graph_mode"] = receipt_graph_mode
    vars(text_model)["_rpu_last_execution_plan"] = execution_receipt

    vocab_size = self.config.text_config.vocab_size
    hidden_dim = self.config.text_config.hidden_size
    if isinstance(logits_to_keep, int):
        slice_indices = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
    else:
        slice_indices = logits_to_keep

    if raw.size(-1) == vocab_size + hidden_dim:
        # Approximate W4 logits plus final normalized hidden state. The
        # explicit exact-rerank helper recomputes only the best two rows with
        # the retained raw FP16 weight.
        self._rpu_last_decode_hidden = raw[..., vocab_size:].clone()
        logits = raw[..., :vocab_size]
    elif raw.size(-1) == vocab_size:
        # Decode-fused lm_head — C++ already produced [B, 1, vocab] logits.
        self._rpu_last_decode_hidden = None
        logits = raw
    elif raw.size(-1) == hidden_dim:
        # Prefill (or non-fused decode) — apply Python-side lm_head.
        self._rpu_last_decode_hidden = None
        logits = _apply_prefill_lm_head(self, raw, slice_indices)
    else:
        raise RuntimeError(
            f"Qwen3VL RPU forward: unexpected raw last-dim {raw.size(-1)}, "
            f"expected hidden_size={hidden_dim}, vocab_size={vocab_size}, or "
            f"vocab_size+hidden_size={vocab_size + hidden_dim}"
        )

    return Qwen3VLCausalLMOutputWithPast(
        loss=None,
        logits=logits,
        past_key_values=pkv,
        hidden_states=None,
        attentions=None,
        rope_deltas=self.model.rope_deltas,
    )


def vision_cfg_or(model_or_cfg) -> Any:
    """Resolve vision_config from a Qwen3VLForConditionalGeneration instance."""
    if hasattr(model_or_cfg, "config"):
        return model_or_cfg.config.vision_config
    return model_or_cfg.vision_config


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

from rpu_backend.runtime.registry import register_adapter

register_adapter("Qwen3VLForConditionalGeneration", Qwen3VLAdapter)
