"""Qwen3-VL → rpu_backend adapter.

Public surface:
  - `Qwen3VLAdapter` — top-level adapter dispatched by `RPUModelForConditionalGeneration`.
  - `install_qwen3_vl_text_for_rpu` / `install_qwen3_vl_vision_for_rpu` — used
    by the adapter and exported for direct integration.
  - DeepStack helpers (zero keepalive, dense scatter).

Internal:
  - `patches.py` — idempotent class-level patches for `Qwen3VLTextRMSNorm` and
    `Qwen3VLTextRotaryEmbedding` (loaded at import time).
  - `text.py` / `vision.py` — text and vision installers.

Dynamo-safety stamping is applied on `vision_model` only. The top-level
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
from rpu_backend.api.causal_lm import _claim_live_instance
from rpu_backend.runtime.device import extract_to_device_target, is_rpu_device_target
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError
from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime.decoder import (
    _run_causal_decoder_forward,
    chunk_policy_key,
    plan_bounded_prefill_execution,
)
from rpu_backend.runtime.weights import (
    convert_linear_weights_inplace,
    transform_linear_weight,
)

from . import patches  # noqa: F401 — idempotent class swaps at import time
from .text import (
    QWEN3_VL_TEXT_ARCH,
    _deepstack_text_layer_indices,
    build_mrope_cos_sin_tables,
    get_or_create_zero_keepalive,
    install_qwen3_vl_text_for_rpu,
    make_zero_visual_embeds,
    lookup_causal_decoder,
    scatter_visual_embeds_to_dense,
)
from .vision import (
    QWEN3_VL_VISION_ARCH,
    build_vision_rope_tables,
    install_qwen3_vl_vision_for_rpu,
    enable_qwen3_vl_vision_merger_on_device,
    register_qwen3vl_vision_fused_merger,
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
_SUPPORTED_TEXT_PROFILES: frozenset[tuple[int, int, int, int]] = frozenset({
    (2048,  6144, 28,  8),  # Qwen3-VL-2B-Instruct
    _QWEN3_VL_4B_TEXT_PROFILE,
})

_SUPPORTED_TEXT_DETAILS = {
    (2048, 6144, 28, 8): (16, 128),
    (2560, 9728, 36, 8): (32, 128),
}
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

# Known unsupported 8B profile. Keep the shape here only for preflight so the
# shared loader rejects it before moving or transforming weights.
_KNOWN_8B_TEXT_PROFILE = (4096, 12288, 36, 8)

# Qwen3-VL-32B W8A16 is integrated only as a controlled-evaluation path.  The
# retained-Graph path produces exactly zero logits from the third decode step.
# Keep it outside the supported profile registry
# and require an exact opt-in before model loading or any irreversible mutation.
_QWEN3_VL_32B_GRAPH_BLOCKED_ENV = "QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED"
_QWEN3_VL_32B_TEXT_PROFILE = (5120, 25600, 64, 8)
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


def _text_config_profile(text_config) -> tuple[int, int, int, int]:
    return (
        int(text_config.hidden_size),
        int(text_config.intermediate_size),
        int(text_config.num_hidden_layers),
        int(text_config.num_key_value_heads),
    )


def _is_qwen3_vl_32b_w8a16_config(config) -> bool:
    """Return whether *config* is the one admitted 32B controlled profile."""
    # Keep config admission single-sourced with the CPU-only loader so API
    # preflight, direct adapter construction, and checkpoint loading cannot
    # drift on quant metadata or nested geometry.
    from rpu_backend.quant.load import is_qwen3_vl_32b_w8a16_config

    return is_qwen3_vl_32b_w8a16_config(config)


def _check_profile(config) -> None:
    """Validate the public Qwen3-VL profiles before model mutation."""
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
    if is_graph_blocked_32b:
        if os.environ.get(_QWEN3_VL_32B_GRAPH_BLOCKED_ENV) != "1":
            raise UnsupportedModelError(
                "Qwen3VLAdapter: the exact Qwen3-VL-32B W8A16 profile is "
                "graph-blocked: retained Graph replay produces all-zero logits "
                "from the third decode step. It is not a "
                "supported profile. Set exact "
                f"{_QWEN3_VL_32B_GRAPH_BLOCKED_ENV}=1 only for controlled "
                "evaluation; clearing the graph per token is not an accepted "
                "runtime workaround."
            )
    elif profile == _QWEN3_VL_32B_TEXT_PROFILE:
        raise UnsupportedModelError(
            "Qwen3VLAdapter: only the exact Qwen3-VL-32B W8A16 config may enter "
            "the graph-blocked controlled-evaluation path; architecture, text "
            "geometry, logical vision geometry, and quant method must all match."
        )
    elif profile == _KNOWN_8B_TEXT_PROFILE:
        raise UnsupportedModelError(
            "Qwen3VLAdapter: Qwen3-VL-8B is unsupported by the current "
            "operator/runtime profile."
        )
    elif profile not in _SUPPORTED_TEXT_PROFILES:
        raise UnsupportedModelError(
            f"Qwen3VLAdapter: text profile {profile} not supported. "
            f"Supported: {sorted(_SUPPORTED_TEXT_PROFILES)}."
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
        vision_out_hidden = int(vision_cfg.out_hidden_size)
    except (AttributeError, TypeError, ValueError) as exc:
        raise UnsupportedModelError(
            "Qwen3VLAdapter: incomplete or malformed Qwen3-VL config."
        ) from exc
    if is_graph_blocked_32b:
        # The exact nested geometry was checked by the controlled-profile
        # predicate; it intentionally stays outside the certified 2B/4B table.
        return
    expected_text_details = _SUPPORTED_TEXT_DETAILS[profile]
    if text_details != expected_text_details:
        raise UnsupportedModelError(
            "Qwen3VLAdapter: unsupported text attention geometry: "
            f"expected {expected_text_details}, got {text_details}."
        )
    if (vision_profile != _SUPPORTED_VISION_PROFILE
            or vision_out_hidden != profile[0]):
        raise UnsupportedModelError(
            "Qwen3VLAdapter: unsupported vision profile. Only the certified "
            "2B/4B-Instruct vision tower is supported; expected "
            f"{_SUPPORTED_VISION_PROFILE} with out_hidden_size={profile[0]}, "
            f"got {vision_profile} with out_hidden_size={vision_out_hidden}."
        )


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


def _clear_graph_cache(owner, attr: str) -> None:
    cache = getattr(owner, attr, None)
    if cache is not None and hasattr(cache, "clear"):
        try:
            cache.clear()
        except Exception:
            pass


def _invoke_finalizer(owner, attr: str) -> bool:
    finalizer = getattr(owner, attr, None)
    if finalizer is None:
        return False
    if getattr(finalizer, "alive", False):
        try:
            finalizer()
        except Exception:
            pass
    return True


def _destroy_untracked_handle(owner, attr: str, destroy_op: str) -> None:
    handle = getattr(owner, attr, None)
    if handle is not None:
        try:
            getattr(torch.ops.rpu, destroy_op)(handle)
        except Exception:
            pass


def _qwen3_vl_runtime_complete(model) -> bool:
    try:
        text_model = model.model.language_model
        vision_model = model.model.visual
    except AttributeError:
        return False
    text_finalizer = getattr(
        text_model, "_rpu_decoder_handle_finalizer", None
    )
    vision_finalizer = getattr(
        vision_model, "_rpu_vision_handle_finalizer", None
    )
    return bool(
        getattr(model, "_rpu_swizzled", False)
        and hasattr(model, "_rpu_lm_head_w_keepalive")
        and "forward" in vars(model)
        and hasattr(text_model, "_rpu_decoder_handle")
        and text_finalizer is not None
        and getattr(text_finalizer, "alive", False)
        and hasattr(text_model, "_rpu_text_graph_cache")
        and hasattr(text_model, "_rpu_decoder_graph_cache")
        and "forward" in vars(text_model)
        and hasattr(vision_model, "_rpu_vision_handle")
        and vision_finalizer is not None
        and getattr(vision_finalizer, "alive", False)
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

    Builders that reuse the Qwen3-VL installers call this same uninstaller so
    graph caches, finalizers, ``_rpu_*`` attributes, and bound forwards are
    retired as one transaction.

    `forward_snapshots` is a sequence of `(owner, had_forward, forward)` taken
    BEFORE the install; the default `()` means "leave instance forwards alone",
    so callers that install one MUST pass their snapshots.

    Weight swizzling is irreversible and is deliberately NOT undone; the caller
    keeps its own poison marker (`_rpu_swizzle_started`) in place.
    """
    if text_model is not None:
        _clear_graph_cache(text_model, "_rpu_text_graph_cache")
        _clear_graph_cache(text_model, "_rpu_decoder_graph_cache")
    if vision_model is not None:
        _clear_graph_cache(vision_model, "_rpu_vision_graph_cache")

    if text_model is not None:
        tracked = _invoke_finalizer(
            text_model, "_rpu_decoder_handle_finalizer"
        )
        if not tracked:
            _destroy_untracked_handle(
                text_model,
                "_rpu_decoder_handle",
                "causal_decoder_destroy",
            )
    if vision_model is not None:
        tracked = _invoke_finalizer(
            vision_model, "_rpu_vision_handle_finalizer"
        )
        if not tracked:
            _destroy_untracked_handle(
                vision_model,
                "_rpu_vision_handle",
                "qwen3vl_vision_destroy",
            )

    if text_model is not None:
        text_state = vars(text_model)
        for name in tuple(text_state):
            if (name.startswith("_rpu_decoder_")
                    or name.startswith("_rpu_text_")
                    or name == "_rpu_deepstack_lang_layers"):
                text_state.pop(name, None)
    if vision_model is not None:
        vision_state = vars(vision_model)
        vision_exact_attrs = {
            "_rpu_lazy_init_checked",
            "_rpu_required_attrs",
            "_rpu_patch_embed_on_device",
            "_rpu_patch_embed_w_rpu",
            "_rpu_patch_embed_b_rpu",
        }
        for name in tuple(vision_state):
            if name.startswith("_rpu_vision_") or name in vision_exact_attrs:
                vision_state.pop(name, None)

    for owner, had_forward, forward in forward_snapshots:
        _restore_instance_forward(owner, had_forward, forward)
    model_state = vars(model)
    model_state.pop("_rpu_lm_head_w_keepalive", None)
    model_state["_rpu_swizzled"] = False


def _cleanup_failed_qwen3_vl_install(
    adapter,
    model,
    text_model,
    vision_model,
    forward_snapshots,
) -> None:
    """Retire a partial composite install without unpoisoning swizzled weights."""
    uninstall_qwen3_vl_runtime(
        model, text_model, vision_model, forward_snapshots
    )
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


class Qwen3VLAdapter:
    """Per-instance adapter for `Qwen3VLForConditionalGeneration`.

    Lifecycle:
      1. `__init__(model)` — preflight, install `.to('rpu')` interceptor.
      2. `to_rpu()` — irreversible swizzle + install vision/text C++ handles +
         patch top-level forward. Returns the same model object.

    Internal handle map (after `to_rpu()`):
      - `model.model.visual._rpu_vision_handle`     (Qwen3VLVisionModel)
      - `model.model.language_model._rpu_decoder_handle`  (Qwen3VLTextModel)
      - `model._rpu_lm_head_w_keepalive`            (col-swizzled lm_head)
    """

    @classmethod
    def preflight(cls, config) -> None:
        """Config-only check; called from RPUModelForConditionalGeneration."""
        _check_profile(config)

    @classmethod
    def preflight_execution(cls, config, execution_config) -> None:
        requested = execution_config.get("prefill", {}).get(
            "chunk_size", "auto"
        )
        if not isinstance(requested, int):
            return
        text = config.text_config
        try:
            known_8b = _text_config_profile(text) == _KNOWN_8B_TEXT_PROFILE
        except (AttributeError, TypeError, ValueError):
            known_8b = False
        if known_8b:
            raise UnsupportedModelError(
                "Qwen3VLAdapter: Qwen3-VL-8B is unsupported by the current "
                "operator/runtime profile."
            )
        env = lookup_causal_decoder(
            QWEN3_VL_TEXT_ARCH,
            int(text.num_hidden_layers),
            int(text.hidden_size),
        )
        if requested > env.chunk:
            raise UnsupportedModelError(
                f"Qwen3-VL prefill chunk_size={requested} exceeds this "
                f"profile's certified ceiling {env.chunk}; max certified "
                f"execution length is {env.max_kv_len}. Refusing before "
                "loading model weights."
            )

    def __init__(self, model) -> None:
        _check_profile(model.config)
        self.model = model
        self._rpu_is_ready: bool = _qwen3_vl_runtime_complete(model)
        self._rpu_w8a16_staged_plan = None
        if _is_qwen3_vl_32b_w8a16_config(model.config) and not self._rpu_is_ready:
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
        self._is_graph_blocked_32b_w8a16 = bool(
            _is_qwen3_vl_32b_w8a16_config(model.config)
        )

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
            if self._rpu_is_ready or getattr(self.model, "_rpu_swizzled", False):
                raise RPUBackendError(
                    f"model.to({target!r}) rejected: weights swizzled for RPU "
                    "(irreversible). Reload via RPUModelForConditionalGeneration.from_pretrained(...)."
                )
            return original_to(*args, **kwargs)

        rpu_aware_to.__rpu_wrapped__ = True
        model.to = rpu_aware_to

    # ------------------------------------------------------------------ #
    # to_rpu
    # ------------------------------------------------------------------ #

    def to_rpu(self):
        """Run the full RPU enablement sequence. Idempotent for the same model.

        Required order:
          A. Swizzle text decoder linears on CPU (`convert_linear_weights_inplace`).
          B. Untie + col-swizzle `lm_head` on CPU.
          C. Move text_model + lm_head to RPU.
          D. Install vision encoder (does its own fused-QKV split + swizzle).
          E. Install the text decoder with one DeepStack injection in each of
             the first N text layers for N ordered vision features.
          F. Push lm_head weight to the text-decoder C++ handle.
          G. Patch top-level Qwen3VLForConditionalGeneration.forward.
        """
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
        if staged_plan is not None:
            from rpu_backend.quant.load import _validate_staged_w8a16_imagetext

            if _validate_staged_w8a16_imagetext(self.model) is not staged_plan:
                raise RPUBackendError(
                    "Qwen3VLAdapter.to_rpu(): staged checkpoint plan changed; reload."
                )
            is_graph_blocked_32b_w8a16 = True
        else:
            is_graph_blocked_32b_w8a16 = (
                _validate_qwen3_vl_32b_w8a16_model(self.model)
            )
        if is_graph_blocked_32b_w8a16 != self._is_graph_blocked_32b_w8a16:
            raise RPUBackendError(
                "Qwen3VLAdapter.to_rpu(): config or quantized tensor inventory "
                "changed after adapter construction; reload from_pretrained."
            )

        if not _SWIZZLE_LOCK.acquire(blocking=False):
            raise RPUBackendError(
                "Qwen3VLAdapter.to_rpu(): another swizzle in progress."
            )
        text_model = None
        vision_model = None
        forward_snapshots = []
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
                from rpu_backend.quant.load import _validate_staged_w8a16_imagetext

                if _validate_staged_w8a16_imagetext(self.model) is not staged_plan:
                    raise RPUBackendError(
                        "Qwen3VLAdapter.to_rpu(): staged checkpoint plan changed; reload."
                    )
                is_graph_blocked_32b_w8a16 = True
            else:
                is_graph_blocked_32b_w8a16 = (
                    _validate_qwen3_vl_32b_w8a16_model(self.model)
                )
            cfg = self.model.config
            text_cfg = cfg.text_config
            vision_cfg = cfg.vision_config
            text_model = self.model.model.language_model
            vision_model = self.model.model.visual
            forward_snapshots = [
                (self.model, "forward" in vars(self.model), vars(self.model).get("forward")),
                (text_model, "forward" in vars(text_model), vars(text_model).get("forward")),
                (vision_model, "forward" in vars(vision_model), vars(vision_model).get("forward")),
            ]

            _claim_live_instance(self.model)
            self.model._rpu_swizzle_started = True
            mutation_started = True

            # ---------- Step A: swizzle text decoder linears (CPU) -----------
            if staged_plan is not None:
                from rpu_backend.quant.load import (
                    _materialize_staged_w8a16_imagetext_for_rpu,
                )

                _materialize_staged_w8a16_imagetext_for_rpu(self.model)
            elif is_graph_blocked_32b_w8a16:
                # The 32B checkpoint is large enough that retaining every CPU
                # layer while allocating a second RPU copy can exhaust the
                # board host. Convert and migrate one layer at a time.
                for layer in text_model.layers:
                    convert_linear_weights_inplace(layer, skip_names=set())
                    layer.to("rpu")
                    gc.collect()
            else:
                convert_linear_weights_inplace(text_model, skip_names=set())

            # ---------- Step B: lm_head untie + col-swizzle (CPU) ------------
            # If lm_head.weight is tied with text_model.embed_tokens, clone
            # first so the embedding row layout stays raw.
            embed_w_ptr = text_model.embed_tokens.weight.data_ptr()
            lm_head_w = self.model.lm_head.weight.data
            if lm_head_w.data_ptr() == embed_w_ptr:
                lm_head_w = lm_head_w.clone()
                self.model.lm_head.weight = nn.Parameter(lm_head_w, requires_grad=False)
            swizzled = transform_linear_weight(
                self.model.lm_head.weight.data.to(torch.float16).contiguous(),
                partition=1,
            )
            self.model.lm_head.weight = nn.Parameter(swizzled, requires_grad=False)

            # ---------- Step C: move text + lm_head to RPU -------------------
            text_model.to("rpu")
            self.model.lm_head.to("rpu")

            # ---------- Step D: install vision encoder -----------------------
            # The vision adapter handles its own per-block fused-QKV split +
            # swizzle. patch_embed / pos_embed / merger / deepstack_merger_list
            # remain on CPU per the adapter contract.
            install_qwen3_vl_vision_for_rpu(
                vision_model, vision_config=vision_cfg,
                execution_chunk_size=getattr(
                    self.model, "_rpu_execution", {}
                ).get("vision", {}).get("chunk_size", "auto"),
                _allow_graph_blocked_32b=is_graph_blocked_32b_w8a16,
            )

            # ---------- Step E: install text decoder C++ handle --------------
            deepstack_lang_layers = _deepstack_text_layer_indices(vision_cfg)
            install_qwen3_vl_text_for_rpu(
                text_model,
                text_config=text_cfg,
                vision_config=vision_cfg,
                deepstack_lang_layers=deepstack_lang_layers,
                enable_deepstack=True,
                execution_config=getattr(
                    self.model, "_rpu_execution", None
                ),
                scale_lists=(
                    _qwen3_vl_text_scale_lists(text_model)
                    if is_graph_blocked_32b_w8a16 else None
                ),
            )

            # ---------- Step F: push lm_head weight to C++ -------------------
            handle = text_model._rpu_decoder_handle
            lm_w = self.model.lm_head.weight.detach().to(torch.float16).contiguous()
            self.model._rpu_lm_head_w_keepalive = lm_w
            torch.ops.rpu.causal_decoder_set_lm_head(handle, lm_w)

            # ---------- Step G: top-level forward replacement ---------------
            self.model.forward = types.MethodType(_rpu_qwen3vl_forward, self.model)

            self.model._rpu_swizzled = True
            self._rpu_is_ready = True
            self._rpu_w8a16_staged_plan = None

            _LOG.info("Qwen3VLAdapter ready: text_handle=%d, vision_handle=%d, "
                      "deepstack_layers=%s",
                      handle, vision_model._rpu_vision_handle, deepstack_lang_layers)

            return self.model
        except BaseException:
            if mutation_started:
                _cleanup_failed_qwen3_vl_install(
                    self,
                    self.model,
                    text_model,
                    vision_model,
                    forward_snapshots,
                )
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
    """Mirror HF `get_image_features` post-processing: split per image then
    re-concatenate. For single image this is identity but keeps the path
    consistent for multi-image.
    """
    split_sizes = (image_grid_thw.prod(-1) // (spatial_merge_size ** 2)).tolist()
    parts = torch.split(pooler_output, split_sizes)
    return torch.cat(list(parts), dim=0)


# The planner moved to runtime/decoder.py when the plain text decoder started
# padding too — ONE implementation, so the VL and text paths cannot drift.
# Kept under the old name because it is this adapter's internal vocabulary.
_plan_causal_prefill_execution = plan_bounded_prefill_execution


def _pad_causal_prefill_inputs(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    execution_len: int,
):
    """Right-pad token-indexed inputs to the selected execution length.

    Padded rows cannot affect the logical prefix under the causal
    lower-triangular mask (LTM).
    """
    logical_len = inputs_embeds.size(1)
    pad = execution_len - logical_len
    if pad < 0:
        raise ValueError(
            f"execution_len={execution_len} is smaller than logical_len={logical_len}"
        )
    if pad == 0:
        return inputs_embeds, attention_mask, position_ids

    inputs_embeds = torch.cat(
        [inputs_embeds,
         inputs_embeds.new_zeros(inputs_embeds.size(0), pad, inputs_embeds.size(2))],
        dim=1,
    )
    if attention_mask is not None and attention_mask.dim() == 2:
        if attention_mask.size(-1) != logical_len:
            raise ValueError(
                "Qwen3VL RPU forward: 2D attention_mask length "
                f"{attention_mask.size(-1)} != input length {logical_len}"
            )
        attention_mask = torch.cat(
            [attention_mask,
             attention_mask.new_zeros(attention_mask.size(0), pad)],
            dim=-1,
        )
    if position_ids.dim() == 3 and position_ids.size(-1) == logical_len:
        position_ids = torch.cat(
            [position_ids,
             position_ids[..., -1:].expand(*position_ids.shape[:-1], pad)],
            dim=-1,
        ).contiguous()
    elif (
        position_ids.dim() == 2
        and position_ids.size(0) == logical_len
        and position_ids.size(1) == 3
    ):
        position_ids = torch.cat(
            [position_ids, position_ids[-1:].expand(pad, 3)],
            dim=0,
        ).contiguous()
    else:
        raise ValueError(
            "Qwen3VL RPU forward: position_ids must be [3, batch, seq] "
            f"or [seq, 3] with seq={logical_len}, got "
            f"{tuple(position_ids.shape)}"
        )
    return inputs_embeds, attention_mask, position_ids


def _pad_visual_embeds_to_execution(
    dense_visual_embeds: list[torch.Tensor],
    logical_len: int,
    execution_len: int,
) -> list[torch.Tensor]:
    """Append zero rows only to real-image DeepStack dense tensors."""
    pad = execution_len - logical_len
    if pad < 0:
        raise ValueError(
            f"execution_len={execution_len} is smaller than logical_len={logical_len}"
        )
    if pad == 0:
        return dense_visual_embeds
    padded_visual_embeds = []
    for dense in dense_visual_embeds:
        if dense.size(0) != logical_len:
            raise ValueError(
                "Qwen3VL RPU forward: DeepStack rows "
                f"{dense.size(0)} != input length {logical_len}"
            )
        padded_visual_embeds.append(torch.cat(
            [dense, dense.new_zeros(pad, dense.size(1))], dim=0,
        ))
    return padded_visual_embeds


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
):
    """Key text decode by layout, while allowing position-stable replay."""
    dyn_dims = [
        text_model._rpu_text_num_layers,
        text_model._rpu_text_deepstack_hash,
        chunk_policy_key(text_model._rpu_decoder_handle),
        int(planned_chunk_size),
    ]
    if logical_len != 1:
        # Multi-token continuation planning depends on its absolute cache base.
        # Single-token decode refreshes every position-derived register/DMA on
        # REPLAY and must reuse one graph across successive positions.
        dyn_dims.append(int(position))
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


def _reject_continuation_padding(
    prefill_config: Any,
    logical_len: int,
    start_position: int,
) -> None:
    """Keep Qwen3-VL padding confined to the certified initial prefill."""
    padding_rows = prefill_config.get("padding_rows")
    requests_padding = (
        padding_rows == "auto"
        or (isinstance(padding_rows, int) and padding_rows > 0)
        or int(prefill_config.get("padding_budget", 0)) > 0
    )
    if logical_len > 1 and start_position > 0 and requests_padding:
        raise ValueError(
            "Qwen3-VL prefill padding is supported only for the initial "
            "multi-token prefill at cache position 0; continuation padding "
            "would be ignored. Use padding_rows=0 with no padding_budget."
        )


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

    text_model = self.model.language_model
    vision_model = self.model.visual
    text_cfg = self.config.text_config
    hidden_size = text_cfg.hidden_size
    spatial_merge_size = vision_cfg_or(self).spatial_merge_size

    # ----- Vision encoder pass (image OR video) ----------------------------
    # HF `get_video_features` is literally `get_image_features` — same vision
    # tower, same per-row encoder loop.  The grid_thw form differs: vision
    # tower takes T-folded form for both modalities (`[[1,h,w]]` for image,
    # `[[grid_t,h,w]]` for video; per-frame expansion is rope-only).
    deepstack_features: list[torch.Tensor] | None = None
    visual_embeds_flat: torch.Tensor | None = None
    visual_token_id: int | None = None
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
            vision_output = vision_model(pix_in, grid_thw=grid_in)
        # `pooler_output` is the merger output, CPU fp32 [N/sm², 2048].
        pooler = vision_output.pooler_output
        deepstack_features = list(vision_output.deepstack_features)
        # Per-item split (single image / single video → identity).
        visual_embeds_flat = _split_image_embeds_per_image(
            pooler, grid_in.detach().cpu(), spatial_merge_size,
        )

    # ----- Build inputs_embeds + scatter image features --------------------
    if inputs_embeds is None:
        if input_ids is None:
            raise ValueError(
                "Qwen3VL RPU forward: must provide input_ids or inputs_embeds."
            )
        # `embed_tokens` lives on RPU but aten::embedding falls back to CPU;
        # output is on the input_ids device (RPU when caller passes RPU input_ids).
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if inputs_embeds.shape[-1] != hidden_size:
        raise ValueError(
            "Qwen3VL RPU forward: inputs_embeds hidden dimension "
            f"{inputs_embeds.shape[-1]} != model hidden_size {hidden_size}."
        )

    inputs_embeds = inputs_embeds.to(dtype=torch.float16)
    if not inputs_embeds.is_contiguous():
        inputs_embeds = inputs_embeds.contiguous()

    if visual_embeds_flat is not None:
        # masked_scatter on RPU falls back to CPU via custom_cpu_fallback.
        # Do it explicitly on CPU to avoid device-mixing surprises.
        if input_ids is None:
            raise ValueError(
                "Qwen3VL RPU forward: visual scatter requires input_ids "
                "(needed to locate image/video token positions)."
            )
        input_ids_cpu = input_ids.to("cpu")
        inputs_embeds_cpu = inputs_embeds.to("cpu", dtype=inputs_embeds.dtype)
        visual_mask = (input_ids_cpu == visual_token_id)
        visual_mask_expanded = visual_mask.unsqueeze(-1).expand_as(inputs_embeds_cpu)
        visual_embeds_cpu = visual_embeds_flat.to("cpu", dtype=inputs_embeds_cpu.dtype)
        n_visual_tokens = int(visual_mask.sum())
        if visual_embeds_cpu.shape[0] != n_visual_tokens:
            raise RuntimeError(
                f"Qwen3VL RPU forward: visual_embeds rows {visual_embeds_cpu.shape[0]} "
                f"!= visual token count {n_visual_tokens}"
            )
        inputs_embeds_cpu = inputs_embeds_cpu.masked_scatter(
            visual_mask_expanded, visual_embeds_cpu
        )
        inputs_embeds = inputs_embeds_cpu.to(device="rpu", dtype=torch.float16).contiguous()

    if inputs_embeds.device.type != "rpu":
        inputs_embeds = inputs_embeds.to(device="rpu", dtype=torch.float16).contiguous()

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
    prefill_cfg = getattr(
        text_model, "_rpu_execution", {}
    ).get("prefill", {})
    _reject_continuation_padding(
        prefill_cfg, logical_seq_len, start_position
    )
    if logical_seq_len > 1 and start_position == 0:
        physical_limit = min(
            int(past_key_values.max_seq_len) - start_position,
            int(past_key_values.sKeyVx) * int(past_key_values.sKeyChunk),
            int(past_key_values.sValVx) * int(past_key_values.sValChunk),
            _QWEN3_VL_MROPE_KEEPALIVE_ROWS,
        )
        handle = text_model._rpu_decoder_handle
        execution_seq_len, planned_chunk_size = (
            _plan_causal_prefill_execution(
                logical_seq_len,
                physical_limit,
                int(prefill_cfg.get(
                    "padding_budget", _QWEN3_VL_PREFILL_PADDING_BUDGET
                )),
                lambda n: (
                    torch.ops.rpu.causal_decoder_resolve_prefill_chunk_size(
                        handle, n, start_position
                    )
                ),
                position=start_position,
                alignment=16,
                padding_rows=prefill_cfg.get("padding_rows", "auto"),
                exact_chunk_size=(
                    prefill_cfg.get("chunk_size")
                    if isinstance(prefill_cfg.get("chunk_size"), int)
                    else None
                ),
            )
        )

    # ----- position_ids: compute via get_rope_index when needed ------------
    if position_ids is None:
        if (
            input_ids is not None
            and mm_token_type_ids is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
        ):
            # HF Qwen3-VL get_rope_index needs per-frame expansion of
            # video_grid_thw (see _expand_video_grid_per_frame docstring).
            vgt_rope = (
                _expand_video_grid_per_frame(video_grid_thw.detach().cpu())
                if video_grid_thw is not None else None
            )
            pos_ids_cpu, rope_deltas = self.model.get_rope_index(
                input_ids.to("cpu"),
                mm_token_type_ids=mm_token_type_ids.to("cpu"),
                image_grid_thw=image_grid_thw.detach().cpu()
                    if image_grid_thw is not None else None,
                video_grid_thw=vgt_rope,
                attention_mask=attention_mask.to("cpu")
                    if attention_mask is not None else None,
            )
            self.model.rope_deltas = rope_deltas
            position_ids = pos_ids_cpu  # [3, batch, seq_len] CPU
        else:
            # text-only or pre-cached image continuation: arange-expanded.
            batch_size = inputs_embeds.shape[0]
            arange = torch.arange(
                past_key_values.position,
                past_key_values.position + seq_len,
                dtype=torch.long,
            )
            position_ids = arange.view(1, 1, -1).expand(3, batch_size, -1)
            # If we have cached rope_deltas (decode continuation after image
            # prefill), add the delta — matches HF compute_3d_position_ids.
            if self.model.rope_deltas is not None:
                position_ids = position_ids + self.model.rope_deltas

    inputs_embeds, attention_mask, position_ids = _pad_causal_prefill_inputs(
        inputs_embeds, attention_mask, position_ids, execution_seq_len,
    )

    # ----- Build dense_visual_embeds (DeepStack injection) -----------------
    if deepstack_features is not None and visual_token_id is not None:
        visual_pos_mask = (input_ids == visual_token_id).squeeze(0)
        dense_visual_embeds = scatter_visual_embeds_to_dense(
            deepstack_features,
            visual_pos_mask,
            logical_seq_len,
            hidden_size,
        )
        dense_visual_embeds = _pad_visual_embeds_to_execution(
            dense_visual_embeds, logical_seq_len, execution_seq_len,
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
    # Wrap in graph_cache.capture(sig) so all RPU
    # kernel dispatches inside `_run_causal_decoder_forward` batch into
    # one queue execution. Position is deliberately absent only for single-token
    # decode: the generic decoder refreshes every position-derived register/DMA
    # field on REPLAY, so successive `seq_len == 1` steps share one entry.
    # Multi-token continuation keeps position because its chunk/SDPA plan can
    # change with the absolute start. CPU→RPU prep for
    # attention_mask/position_ids stays OUTSIDE capture so each forward
    # picks up fresh source tensor addresses (the framework re-binds
    # tensor inputs of recorded ops but a CPU→RPU DMA inside capture
    # would freeze the first call's CPU pointer).
    attn_rpu = attention_mask.to("rpu") if attention_mask is not None else None
    pos_rpu = position_ids.to("rpu") if position_ids is not None else None
    text_graph_cache = text_model._rpu_text_graph_cache
    sig = _qwen3_vl_text_graph_signature(
        text_model,
        logical_seq_len,
        seq_len,
        int(past_key_values.position),
        int(planned_chunk_size or 0),
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
        )

    if planned_chunk_size is not None:
        resolved_chunk_size = int(
            torch.ops.rpu.causal_decoder_get_resolved_chunk_size(
                text_model._rpu_decoder_handle
            )
        )
        if resolved_chunk_size != planned_chunk_size:
            raise RuntimeError(
                "Qwen3VL causal prefill dry/forward chunk plan drift: "
                f"dry={planned_chunk_size}, forward={resolved_chunk_size}, "
                f"execution_len={seq_len}"
            )
    else:
        resolved_chunk_size = int(
            torch.ops.rpu.causal_decoder_get_resolved_chunk_size(
                text_model._rpu_decoder_handle
            )
        )

    raw = _restore_logical_prefill(
        raw, pkv, start_position, logical_seq_len, seq_len,
    )
    vars(text_model)["_rpu_last_execution_plan"] = {
        "stage": "prefill" if logical_seq_len > 1 else "decode",
        "logical_len": int(logical_seq_len),
        "execution_len": int(seq_len),
        "chunk_size": resolved_chunk_size,
        "padding_rows": int(seq_len - logical_seq_len),
        "position": start_position,
    }

    vocab_size = self.config.text_config.vocab_size
    hidden_dim = self.config.text_config.hidden_size
    if isinstance(logits_to_keep, int):
        slice_indices = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
    else:
        slice_indices = logits_to_keep

    if raw.size(-1) == vocab_size:
        # Decode-fused lm_head — C++ already produced [B, 1, vocab] logits.
        logits = raw
    elif raw.size(-1) == hidden_dim:
        # Prefill (or non-fused decode) — apply Python-side lm_head.
        logits = self.lm_head(raw[:, slice_indices, :])
    else:
        raise RuntimeError(
            f"Qwen3VL RPU forward: unexpected raw last-dim {raw.size(-1)}, "
            f"expected hidden_size={hidden_dim} or vocab_size={vocab_size}"
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
