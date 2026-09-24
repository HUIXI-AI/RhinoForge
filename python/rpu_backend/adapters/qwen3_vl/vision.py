"""Qwen3-VL vision encoder → rpu_backend (R-Phase 3).

Wires `Qwen3VLVisionModel` into the C++ vision subsystem registered as
`torch.ops.rpu.qwen3vl_vision_*`. Mirrors SigLIP's `patch_siglip_model_for_rpu`
pattern: handle lifecycle, weight conversion, forward replacement.

R-Phase 3 deliverable surface:
  - install_qwen3_vl_vision_for_rpu(vision_model)
  - build_vision_rope_tables(...)              # FreqCos / FreqSin static tables
  - The forward replacement returns
    BaseModelOutputWithDeepstackFeatures(last_hidden_state, pooler_output,
                                          deepstack_features) matching the HF
    reference (so callers and R-Phase 4 e2e can use it transparently).

Architecture decisions (carried over from save branch findings F7 / F13 /
F31 / F32):
  - Patch embed runs OUTSIDE the fused graph on **CPU fp16**. `aten::linear`
    on RPU expects the weight to be pre-swizzled (col/row partition); the
    folded Conv3d weight is plain `[embed_dim, cin*tp*ps*ps]` row-major and
    silently produces garbage if run on RPU without swizzle. (F31)
  - `fast_pos_embed_interpolate` runs on **CPU fp16**. The HF
    `nn.Embedding` + `view` + `permute(0,1,3,2,4,5)` + `flatten(0,4)` path
    has a permute-then-flatten contiguity bug on RPU (empirically cos_sim
    mean ~0.1 with bimodal distribution). (F32)
  - Merger + DeepStack mergers run on **CPU fp32** — saves ~192 MB RPU DDR
    and runs once per image post-encoder on small `[N/sm², 2048]` tensors.
  - 2D RoPE: position_idx built per forward, copy_in to model-owned
    keepalive `[MAX_KEEPALIVE_SEQ, 2] int16`.

R-Phase 3 migration vs save branch:
  - Per-image vision call wrapped in `with cache.capture(sig)` (F8) so each
    unique `num_patches` produces one BUILD + many REPLAYs. Replaces the
    deleted C++ `cfg.main_graph_id / weights_graph_id` admission.
  - Dynamo-safety stamping (F6): eager RPUCache init, `_rpu_lazy_init_checked`,
    `_rpu_required_attrs`, `_verify_lazy_init(...)` — mirrors
    `siglip.py:481`.
"""
from __future__ import annotations

import os
import types
from typing import Any

import torch
import torch.nn as nn

import rpu_backend
from rpu_backend.runtime import rpu_env_bool
from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
from rpu_backend.runtime.execution_planner import GRAPH_COMPOSITE_CHILD
from rpu_backend.runtime.log import _LOG
from rpu_backend.runtime._native_retirement import _InstalledNativeResource
from rpu_backend.runtime.weights import (
    tp_col_swizzle_mc_weight,
    tp_row_swizzle_mc_weight,
)
from rpu_backend.api.cache import RPUCache


def _qwen3vl_rope_route_request() -> int:
    """Cold legacy env translator: absent=AUTO, explicit 1/0=SPM/DDR."""
    name = "RPU_QWEN3VL_VISION_ROPE_SPM"
    if name not in os.environ:
        return 0
    return 1 if rpu_env_bool(name) else 2


def _deepstack_output_cls():
    """Return the transformers 5.x output class or a 4.57-compatible equivalent."""
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            BaseModelOutputWithDeepstackFeatures,
        )
        return BaseModelOutputWithDeepstackFeatures
    except ImportError:
        pass

    global _DEEPSTACK_OUTPUT_FALLBACK
    if _DEEPSTACK_OUTPUT_FALLBACK is None:
        import dataclasses
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        @dataclasses.dataclass
        class BaseModelOutputWithDeepstackFeatures(BaseModelOutputWithPooling):
            deepstack_features: list[torch.FloatTensor] | None = None

        _DEEPSTACK_OUTPUT_FALLBACK = BaseModelOutputWithDeepstackFeatures
    return _DEEPSTACK_OUTPUT_FALLBACK


_DEEPSTACK_OUTPUT_FALLBACK = None


QWEN3_VL_VISION_ARCH = "qwen3_vl_vision"
_QWEN3_VL_32B_GRAPH_BLOCKED_ENV = "QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED"
_QWEN3_VL_32B_LOGICAL_VISION_PROFILE = (
    1152,
    4304,
    27,
    16,
    16,
    2,
    2,
    (8, 16, 24),
    5120,
    "gelu_pytorch_tanh",
)
_QWEN3_VL_8B_LOGICAL_VISION_PROFILE = (
    1152,
    4304,
    27,
    16,
    16,
    2,
    2,
    (8, 16, 24),
    4096,
    "gelu_pytorch_tanh",
)
_QWEN3_VL_32B_PHYSICAL_HEAD_DIM = 80
_QWEN3_VL_32B_PHYSICAL_INTERMEDIATE_SIZE = 4352


def _validate_vision_core_count(num_cores: int) -> None:
    if type(num_cores) is not int or num_cores not in (4, 8):
        raise ValueError("Qwen3-VL Vision num_cores must be 4 or 8")


def validate_qwen3_vl_vision_core_profile(
    vision_model, *, num_cores: int = 8, vision_config=None,
) -> None:
    """Read-only admission of the complete fresh four-core FP16 Vision tree.

    The composite adapter calls this before claiming or converting either
    tower. Eight-core callers retain their existing profile admission.
    """
    _validate_vision_core_count(num_cores)
    if num_cores == 8:
        return
    if rpu_env_bool("RPU_QWEN3VL_VISION_BATCH"):
        raise ValueError("four-core Vision requires per-image execution; packed image batches require 8 cores")
    cfg = vision_config or vision_model.config
    fields = ("hidden_size", "intermediate_size", "depth", "num_heads",
              "patch_size", "temporal_patch_size", "spatial_merge_size",
              "out_hidden_size", "in_channels", "num_position_embeddings")
    geometry = tuple(getattr(cfg, field, None) for field in fields)
    allowed = {
        (1024, 4096, 24, 16, 16, 2, 2, 2048, 3, 2304): (5, 11, 17),
        (1024, 4096, 24, 16, 16, 2, 2, 2560, 3, 2304): (5, 11, 17),
        (1152, 4304, 27, 16, 16, 2, 2, 4096, 3, 2304): (8, 16, 24),
    }
    if (any(type(value) is not int for value in geometry)
            or geometry not in allowed
            or getattr(cfg, "model_type", None) != "qwen3_vl"
            or getattr(cfg, "hidden_act", None) != "gelu_pytorch_tanh"
            or tuple(getattr(cfg, "deepstack_visual_indexes", ())) != allowed[geometry]):
        raise ValueError("four-core Vision requires exact Qwen3-VL 2B/4B/8B FP16 geometry")
    if not isinstance(vision_model, nn.Module):
        raise ValueError("four-core Vision requires the complete CPU module tree")
    for module in vision_model.modules():
        if any(getattr(owner, name, None) is not None
               for owner in (module, getattr(module, "config", None))
               for name in ("quantization_config", "quant_config")):
            raise ValueError("four-core Vision does not admit quantization metadata")
        # The composite constructor publishes cold configuration before its
        # to_rpu preclaim check. These views own no weights or device storage.
        cold_views = {"_rpu_execution", "_rpu_execution_graph_key_words"} if module is vision_model else set()
        if any((name.startswith("_rpu_") and name not in cold_views)
               or name == "_qwen3_vl_delivery_profile"
               for name in vars(module)):
            raise ValueError("four-core Vision requires fresh unconverted CPU weights")
    hidden, intermediate, depth, heads, _, _, _, out_hidden, _, _ = geometry
    if (vision_model.spatial_merge_size != 2 or vision_model.patch_size != 16
            or vision_model.spatial_merge_unit != 4 or vision_model.num_grid_per_side != 48
            or tuple(vision_model.deepstack_visual_indexes) != allowed[geometry]
            or vision_model.rotary_pos_emb.dim != hidden // heads // 2
            or vision_model.rotary_pos_emb.theta != 10000.0):
        raise ValueError("four-core Vision runtime geometry differs from its config")
    expected = {}

    def tensor(name, value, shape):
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != torch.float16 or not value.is_contiguous()
                or tuple(value.shape) != shape):
            raise ValueError(f"four-core Vision {name} must be contiguous CPU FP16 {shape}")
        expected[name] = value

    def linear(name, module, in_features, out_features):
        if (type(module) is not nn.Linear or module.in_features != in_features
                or module.out_features != out_features):
            raise ValueError(f"four-core Vision {name} has invalid Linear geometry")
        tensor(name + ".weight", module.weight, (out_features, in_features))
        tensor(name + ".bias", module.bias, (out_features,))

    def norm(name, module, width):
        if (type(module) is not nn.LayerNorm or tuple(module.normalized_shape) != (width,)
                or module.eps != 1e-6):
            raise ValueError(f"four-core Vision {name} has invalid LayerNorm semantics")
        tensor(name + ".weight", module.weight, (width,))
        tensor(name + ".bias", module.bias, (width,))

    if len(vision_model.blocks) != depth or len(vision_model.deepstack_merger_list) != 3:
        raise ValueError("four-core Vision requires every encoder and DeepStack layer")
    from transformers.activations import GELUTanh
    for index, block in enumerate(vision_model.blocks):
        prefix = f"blocks.{index}"
        norm(prefix + ".norm1", block.norm1, hidden)
        norm(prefix + ".norm2", block.norm2, hidden)
        if (block.attn.num_heads != heads or block.attn.head_dim != hidden // heads
                or block.attn.attention_dropout != 0.0 or block.attn.is_causal is not False):
            raise ValueError(f"four-core Vision {prefix} has invalid attention semantics")
        linear(prefix + ".attn.qkv", block.attn.qkv, hidden, 3 * hidden)
        linear(prefix + ".attn.proj", block.attn.proj, hidden, hidden)
        linear(prefix + ".mlp.linear_fc1", block.mlp.linear_fc1, hidden, intermediate)
        linear(prefix + ".mlp.linear_fc2", block.mlp.linear_fc2, intermediate, hidden)
        if type(block.mlp.act_fn) is not GELUTanh:
            raise ValueError(f"four-core Vision {prefix} requires tanh GELU")
    patch = vision_model.patch_embed.proj
    if (type(patch) is not nn.Conv3d or patch.in_channels != 3 or patch.out_channels != hidden
            or patch.kernel_size != (2, 16, 16) or patch.stride != (2, 16, 16)
            or patch.padding != (0, 0, 0) or patch.dilation != (1, 1, 1) or patch.groups != 1):
        raise ValueError("four-core Vision patch embedding has invalid Conv3d geometry")
    tensor("patch_embed.proj.weight", patch.weight, (hidden, 3, 2, 16, 16))
    tensor("patch_embed.proj.bias", patch.bias, (hidden,))
    tensor("pos_embed.weight", vision_model.pos_embed.weight, (2304, hidden))
    for name, merger, postshuffle in [
        ("merger", vision_model.merger, False),
        *((f"deepstack_merger_list.{index}", merger, True)
          for index, merger in enumerate(vision_model.deepstack_merger_list)),
    ]:
        if merger.use_postshuffle_norm is not postshuffle or merger.hidden_size != hidden * 4:
            raise ValueError(f"four-core Vision {name} has invalid merger semantics")
        if type(merger.act_fn) is not nn.GELU or merger.act_fn.approximate != "none":
            raise ValueError(f"four-core Vision {name} requires exact CPU GELU")
        norm(name + ".norm", merger.norm, hidden * (4 if postshuffle else 1))
        linear(name + ".linear_fc1", merger.linear_fc1, hidden * 4, hidden * 4)
        linear(name + ".linear_fc2", merger.linear_fc2, hidden * 4, out_hidden)
    actual = dict(vision_model.named_parameters())
    if actual.keys() != expected.keys() or any(actual[name] is not value for name, value in expected.items()):
        raise ValueError("four-core Vision parameter inventory differs from the admitted tower")
    buffers = dict(vision_model.named_buffers())
    if set(buffers) != {"rotary_pos_emb.inv_freq"}:
        raise ValueError("four-core Vision buffer inventory differs from the admitted tower")
    for name, value in buffers.items():
        if (name != "rotary_pos_emb.inv_freq" or value.device.type != "cpu"
                or value.dtype not in (torch.float16, torch.float32) or not value.is_contiguous()
                or tuple(value.shape) != (hidden // heads // 4,)):
            raise ValueError(f"four-core Vision has an unsupported buffer {name}")


def _execution_graph_key_words(model) -> tuple[int, ...]:
    """Validate the optional parent-composite physical identity hook."""
    value = getattr(model, "_rpu_execution_graph_key_words", ())
    if not isinstance(value, tuple) or any(
        isinstance(word, bool) or not isinstance(word, int) or word < 0
        for word in value
    ):
        raise RuntimeError(
            "Qwen3-VL vision _rpu_execution_graph_key_words must be a "
            "tuple of non-negative integers"
        )
    return value


def _plan_qwen3_vl_vision_execution(
    handle: int,
    num_patches: int,
    image_batch_count: int,
    exact_chunk_size: int | None,
    *, execution_owner=None,
    graph_cache=None, plan_signature=(),
):
    """Resolve one unpadded native Vision stage descriptor before BUILD."""
    plan_box = {}
    execution_len, chunk_size = plan_bounded_prefill_execution(
        int(num_patches),
        int(num_patches),
        0,
        resolve_stage_domain=lambda length: (
            torch.ops.rpu.qwen3vl_vision_resolve_stage_domain(
                handle, int(length), int(image_batch_count)
            )
        ),
        position=0,
        alignment=1,
        padding_rows=0,
        exact_chunk_size=exact_chunk_size,
        request_id="qwen3-vl:vision_encoder:vision",
        execution_owner=execution_owner,
        execution_stage="vision",
        execution_native=("qwen3vl_vision", int(handle)) if execution_owner is not None else None,
        queue_owner_id=int(handle),
        plan_result_sink=lambda result: plan_box.__setitem__("result", result),
        graph_mode=GRAPH_COMPOSITE_CHILD,
        plan_signature=(int(image_batch_count), *plan_signature),
        graph_cache=graph_cache,
    )
    if execution_len != int(num_patches):
        raise RuntimeError(
            "Qwen3-VL vision dry planner changed the unpadded execution "
            f"length: patches={num_patches}, execution={execution_len}"
        )
    result = plan_box["result"]
    if (
        result.selected is None
        or chunk_size != result.selected.stage_tuple.compute_chunk
        or not result.selected.stage_tuple.physical_descriptor
    ):
        raise RuntimeError(
            "Qwen3-VL vision dry planner returned no consumable native "
            "stage descriptor"
        )
    return result


def _vision_execution_receipt(
    model, total_patches, plans, resolved_chunks, physical_descriptors,
):
    if (
        not plans
        or len(plans) != len(resolved_chunks)
        or len(plans) != len(physical_descriptors)
        or any(
            plan.selected is None
            or plan.selected.stage_tuple.compute_chunk != chunk
            or tuple(plan.selected.stage_tuple.physical_descriptor)
            != tuple(descriptor)
            for plan, chunk, descriptor in zip(
                plans, resolved_chunks, physical_descriptors
            )
        )
    ):
        raise RuntimeError(
            "Qwen3-VL vision cannot publish a descriptor without "
            "dry/forward agreement"
        )
    graph_mode = plans[0].graph_mode
    if any(plan.graph_mode != graph_mode for plan in plans[1:]):
        raise RuntimeError(
            "Qwen3-VL vision cannot publish inconsistent graph lifecycles"
        )
    unique_chunks = set(resolved_chunks)
    receipt = {
        "stage": "vision",
        "component": getattr(
            model, "_fmb_execution_component_id", "vision_encoder"
        ),
        "generation": int(getattr(model, "_fmb_execution_generation", 0)),
        "logical_len": int(total_patches),
        "execution_len": int(total_patches),
        "chunk_size": (
            next(iter(unique_chunks)) if len(unique_chunks) == 1 else 0
        ),
        "padding_rows": 0,
        "position": 0,
        "graph_mode": graph_mode,
        "dispatch_chunk_sizes": tuple(resolved_chunks),
        "physical_descriptors": tuple(
            tuple(descriptor) for descriptor in physical_descriptors
        ),
        "authority": "NATIVE_A6_STAGE_DESCRIPTOR",
        "selection_scopes": tuple(plan.selection_scope for plan in plans),
        "physical_plan_digests": tuple(
            plan.physical_plan_digest for plan in plans
        ),
        "plan_digests": tuple(plan.plan_digest for plan in plans),
        "dry_forward_agreement": True,
    }
    if getattr(model, "_rpu_vision_fused_merger_w8a16", False):
        receipt["dispatch_merger_routes"] = tuple(
            "chunked_fused_w8a16" if (
                getattr(model, "_rpu_vision_compact_encoder_w8a16", False) and
                getattr(model, "_rpu_vision_execution_chunk_size", "auto") == "auto" and
                plan.selected.execution_len == 1200 and
                plan.selected.stage_tuple.compute_chunk == 1200)
            else "fused_w8a16" if plan.selected.stage_tuple.compute_chunk >= plan.selected.execution_len
            else ("chunked_fused_w8a16" if getattr(model, "_rpu_vision_chunked_merger_w8a16", False)
                  else "eager_w8a16") for plan in plans)
    elif getattr(model, "_rpu_vision_chunked_merger_fp16", False):
        route = ("chunked_fused_fp16" if all(
            plan.selected.execution_len == 1200 and
            (plan.selected.stage_tuple.compute_chunk == 608 or
             (getattr(model, "_rpu_vision_compact_encoder_fp16", False) and
              getattr(model, "_rpu_vision_execution_chunk_size", "auto") == "auto" and
              plan.selected.stage_tuple.compute_chunk == 1200)) for plan in plans)
            else "eager_fp16")
        receipt["dispatch_merger_routes"] = (route,) * len(plans)
    return receipt


def _is_qwen3_vl_32b_vision_config(cfg: Any) -> bool:
    try:
        return (
            int(cfg.hidden_size),
            int(cfg.intermediate_size),
            int(cfg.depth),
            int(cfg.num_heads),
            int(cfg.patch_size),
            int(cfg.temporal_patch_size),
            int(cfg.spatial_merge_size),
            tuple(int(x) for x in cfg.deepstack_visual_indexes),
            int(cfg.out_hidden_size),
            cfg.hidden_act,
        ) == _QWEN3_VL_32B_LOGICAL_VISION_PROFILE
    except (AttributeError, TypeError, ValueError):
        return False


def _is_qwen3_vl_8b_vision_config(cfg: Any) -> bool:
    try:
        return (
            int(cfg.hidden_size),
            int(cfg.intermediate_size),
            int(cfg.depth),
            int(cfg.num_heads),
            int(cfg.patch_size),
            int(cfg.temporal_patch_size),
            int(cfg.spatial_merge_size),
            tuple(int(x) for x in cfg.deepstack_visual_indexes),
            int(cfg.out_hidden_size),
            cfg.hidden_act,
        ) == _QWEN3_VL_8B_LOGICAL_VISION_PROFILE
    except (AttributeError, TypeError, ValueError):
        return False


def build_vision_rope_tables(
    head_dim: int,
    max_hw: int,
    *,
    device: str | torch.device = "rpu",
    dtype: torch.dtype = torch.float16,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FreqCos / FreqSin `[max_hw, head_dim/4]` fp16 tables for the
    `rope_2d_ddr` kernel.

    Mirrors `Qwen3VLVisionRotaryEmbedding(head_dim // 2).forward(max_hw)`
    followed by element-wise cos/sin. The kernel uses the same table for
    BOTH row and col axes — it indexes once per axis via the per-token
    position_idx int16 pair.

    Args:
        head_dim: attention head dim (full, NOT half). For Qwen3-VL-2B/4B
            vision this is 64. The freqs use head_dim/2 lanes; the kernel
            splits into row/col halves of head_dim/4 each.
        max_hw: largest of max(h, w) across the image grids the model will
            see. 48 covers any image with patch-grid ≤ 48 (e.g., 768×768
            with patch_size=16 → 48×48). Larger values are safe but waste
            DDR.
    """
    if head_dim <= 0 or (head_dim % 4) != 0:
        raise ValueError(
            f"build_vision_rope_tables: head_dim must be positive multiple of 4, "
            f"got {head_dim}"
        )
    if max_hw <= 0:
        raise ValueError(f"build_vision_rope_tables: max_hw must be positive, got {max_hw}")

    half_axis = head_dim // 4
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float64) / (head_dim // 2))
    )  # [head_dim/4]
    positions = torch.arange(max_hw, dtype=torch.float64)  # [max_hw]
    freqs = positions[:, None] * inv_freq[None, :]  # [max_hw, head_dim/4]
    expected_shape = (max_hw, half_axis)
    if tuple(freqs.shape) != expected_shape:
        raise RuntimeError(
            "build_vision_rope_tables produced an invalid frequency shape: "
            f"expected {expected_shape}, got {tuple(freqs.shape)}"
        )

    cos = freqs.cos().to(dtype=dtype).contiguous()
    sin = freqs.sin().to(dtype=dtype).contiguous()
    return cos.to(device=device), sin.to(device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Weight conversion: split fused QKV, swizzle per partition
# ─────────────────────────────────────────────────────────────────────────────

_VISION_CONVERSION_IN_PROGRESS = "qwen3vl-vision-conversion-in-progress"


def _quantize_vision_w8_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cold Vision-only quantization boundary; eager arithmetic does not scan scales."""
    from rpu_backend.quant._common import quantize_linear_per_channel

    quantized, scale = quantize_linear_per_channel(weight)
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("Qwen3-VL Vision W8 quantization requires finite positive scales")
    return quantized, scale


def _validate_vision_w8_block_source(block) -> None:
    # Inspect the complete block before its first irreversible replacement.
    # FP32 inputs must remain finite after the FP16 conversion used by quantization.
    for layer in (block.attn.qkv, block.attn.proj,
                  block.mlp.linear_fc1, block.mlp.linear_fc2):
        if not bool(torch.isfinite(layer.weight.detach().half()).all()):
            raise ValueError("Qwen3-VL Vision W8 source must be finite in FP16")


def _convert_vision_block_weights_for_rpu(
    block,
    num_heads: int,
    hidden_size: int,
    w8a16: bool = False,
    *,
    num_cores: int = 8,
    physical_head_dim: int | None = None,
    physical_intermediate_size: int | None = None,
) -> None:
    """In-place swizzle of a single Qwen3VLVisionBlock's weights.

    Splits the fused HF `attn.qkv` Linear `[3*dim, dim]` into three Linears
    q/k/v `[dim, dim]` (stashed as `_rpu_q_w`, `_rpu_q_b`, etc. on the block),
    then col-swizzles each for the 8-core layout. The o_proj and fc1/fc2 get
    swizzled in place via direct `tp_*_swizzle_mc_weight` calls.

    W8A16 (opt-in, gr00t only): int8 the 6 GEMMs (q/k/v/o/fc1/fc2) — per-output-channel
    symmetric int8 + fp16 scale, int8 swizzle (dwidth=1). The fp16 scales are stashed as
    `_rpu_*_ws`. Default off → fp16 (byte-identical for the qwen3_vl E2E path).

    Idempotent via `_rpu_qwen3vl_vision_weights_converted` marker.
    """
    _validate_vision_core_count(num_cores)
    if num_cores != 8 and w8a16:
        raise ValueError("four-core Vision does not admit W8A16 conversion")
    if not isinstance(w8a16, bool):
        raise TypeError(
            "qwen3_vl vision w8a16 must be a bool, "
            f"got {type(w8a16).__name__}."
        )

    # Validate fused QKV first so malformed public inputs fail with the
    # established diagnostic before we inspect any sibling submodule.
    qkv_w = block.attn.qkv.weight.data  # [3*hidden, hidden]
    qkv_b = block.attn.qkv.bias.data    # [3*hidden]
    expected_weight = (3 * hidden_size, hidden_size)
    if tuple(qkv_w.shape) != expected_weight:
        raise ValueError(
            "Qwen3-VL vision fused QKV weight must have shape "
            f"{expected_weight}, got {tuple(qkv_w.shape)}"
        )
    expected_bias = (3 * hidden_size,)
    if tuple(qkv_b.shape) != expected_bias:
        raise ValueError(
            "Qwen3-VL vision fused QKV bias must have shape "
            f"{expected_bias}, got {tuple(qkv_b.shape)}"
        )

    _prev = getattr(block, "_rpu_qwen3vl_vision_weights_converted", None)
    if _prev == _VISION_CONVERSION_IN_PROGRESS:
        raise RuntimeError(
            "qwen3_vl vision block has a partial prior weight conversion; "
            "reload the model before retrying."
        )

    if num_heads <= 0 or hidden_size % num_heads:
        raise ValueError(
            "qwen3_vl vision hidden_size must be divisible by positive "
            f"num_heads, got hidden_size={hidden_size}, num_heads={num_heads}."
        )
    derived_head_dim = hidden_size // num_heads
    derived_intermediate_size = int(block.mlp.linear_fc1.weight.size(0))
    logical_head_dim, logical_intermediate_size = getattr(
        block,
        "_rpu_qwen3vl_vision_logical_geometry",
        (derived_head_dim, derived_intermediate_size),
    )
    physical_head_dim = (
        logical_head_dim if physical_head_dim is None
        else int(physical_head_dim)
    )
    physical_intermediate_size = (
        logical_intermediate_size
        if physical_intermediate_size is None
        else int(physical_intermediate_size)
    )
    physical_geometry = (physical_head_dim, physical_intermediate_size)
    padded_vision = physical_geometry != (
        logical_head_dim,
        logical_intermediate_size,
    )
    if padded_vision and (
        (
            hidden_size,
            logical_intermediate_size,
            num_heads,
            logical_head_dim,
            physical_head_dim,
            physical_intermediate_size,
        ) != (1152, 4304, 16, 72, 80, 4352)
    ):
        raise ValueError(
            "qwen3_vl vision physical padding is private to the exact 8B/32B "
            "vision tower (head 72->80, MLP 4304->4352)."
        )

    # The marker records the quant MODE: re-installing the same block in the same mode is a
    # no-op, but switching mode (fp16 <-> w8a16) after the weights are already swizzled would
    # double-swizzle / mismatch dtype — fail loud rather than silently corrupt.
    if _prev is not None:
        if getattr(block, "_rpu_qwen3vl_vision_num_cores", 8) != num_cores:
            raise RuntimeError("Vision core count cannot change after weight conversion")
        if _prev != bool(w8a16):
            raise RuntimeError(
                f"qwen3_vl vision block already converted with w8a16={_prev}; cannot re-install "
                f"with w8a16={bool(w8a16)} (weights already swizzled). Rebuild from a fresh model.")
        previous_geometry = getattr(
            block,
            "_rpu_qwen3vl_vision_physical_geometry",
            (logical_head_dim, logical_intermediate_size),
        )
        if tuple(previous_geometry) != physical_geometry:
            raise RuntimeError(
                "qwen3_vl vision block already converted with physical "
                f"geometry={tuple(previous_geometry)}, cannot reinstall with "
                f"geometry={physical_geometry}. Reload the model."
            )
        return

    def _swz(w_unswz, swz_fn):       # -> (weight [int8 dwidth=1 if w8a16 else fp16], fp16 scale or None)
        if w8a16:
            wq, ws = _quantize_vision_w8_weight(w_unswz.half())
            return swz_fn(wq, num_cores, dwidth=1).contiguous(), ws.to(torch.float16)
        # The public installer accepts a CPU model and owns the dtype conversion.
        # LingBot-VLA-V2 checkpoints are fp32, while the default swizzler's
        # dwidth=2 contract requires fp16 input before it rearranges bytes.
        return (swz_fn(w_unswz.half()) if num_cores == 8
                else swz_fn(w_unswz.half(), num_cores)), None

    with torch.no_grad():
        o_w_raw = block.attn.proj.weight.data
        o_b = block.attn.proj.bias.data
        fc1_w_raw = block.mlp.linear_fc1.weight.data
        fc1_b_raw = block.mlp.linear_fc1.bias.data
        fc2_w_raw = block.mlp.linear_fc2.weight.data
        fc2_b = block.mlp.linear_fc2.bias.data
        expected_shapes = {
            "o weight": ((hidden_size, hidden_size), tuple(o_w_raw.shape)),
            "o bias": ((hidden_size,), tuple(o_b.shape)),
            "fc1 weight": (
                (logical_intermediate_size, hidden_size),
                tuple(fc1_w_raw.shape),
            ),
            "fc1 bias": ((logical_intermediate_size,), tuple(fc1_b_raw.shape)),
            "fc2 weight": (
                (hidden_size, logical_intermediate_size),
                tuple(fc2_w_raw.shape),
            ),
            "fc2 bias": ((hidden_size,), tuple(fc2_b.shape)),
        }
        for name, (expected, actual) in expected_shapes.items():
            if actual != expected:
                raise ValueError(
                    f"Qwen3-VL vision {name} must have shape {expected}, got {actual}"
                )

        if w8a16:
            _validate_vision_w8_block_source(block)

        # From this point onward the conversion mutates published Parameters.
        # Leave a poison marker on BaseException so retry cannot double-swizzle
        # a partially converted block.
        block._rpu_qwen3vl_vision_weights_converted = (
            _VISION_CONVERSION_IN_PROGRESS
        )

        q_w = qkv_w[0:hidden_size].contiguous()
        k_w = qkv_w[hidden_size : 2 * hidden_size].contiguous()
        v_w = qkv_w[2 * hidden_size : 3 * hidden_size].contiguous()
        q_b = qkv_b[0:hidden_size].contiguous()
        k_b = qkv_b[hidden_size : 2 * hidden_size].contiguous()
        v_b = qkv_b[2 * hidden_size : 3 * hidden_size].contiguous()

        if padded_vision:
            def _pad_attention_rows(weight):
                padded = weight.new_zeros(
                    num_heads, physical_head_dim, hidden_size)
                padded[:, :logical_head_dim, :] = weight.view(
                    num_heads, logical_head_dim, hidden_size)
                return padded.reshape(
                    num_heads * physical_head_dim, hidden_size).contiguous()

            def _pad_attention_bias(bias):
                padded = bias.new_zeros(num_heads, physical_head_dim)
                padded[:, :logical_head_dim] = bias.view(
                    num_heads, logical_head_dim)
                return padded.reshape(num_heads * physical_head_dim).contiguous()

            def _pad_attention_columns(weight):
                padded = weight.new_zeros(
                    hidden_size, num_heads, physical_head_dim)
                padded[:, :, :logical_head_dim] = weight.view(
                    hidden_size, num_heads, logical_head_dim)
                return padded.reshape(
                    hidden_size, num_heads * physical_head_dim).contiguous()

            q_w = _pad_attention_rows(q_w)
            k_w = _pad_attention_rows(k_w)
            v_w = _pad_attention_rows(v_w)
            q_b = _pad_attention_bias(q_b)
            k_b = _pad_attention_bias(k_b)
            v_b = _pad_attention_bias(v_b)
            o_w_raw = _pad_attention_columns(o_w_raw.contiguous())

            fc1_w_padded = fc1_w_raw.new_zeros(
                physical_intermediate_size, hidden_size)
            fc1_w_padded[:logical_intermediate_size, :] = fc1_w_raw
            fc1_w_raw = fc1_w_padded.contiguous()
            fc1_b_padded = fc1_b_raw.new_zeros(physical_intermediate_size)
            fc1_b_padded[:logical_intermediate_size] = fc1_b_raw
            fc1_b_raw = fc1_b_padded.contiguous()
            fc2_w_padded = fc2_w_raw.new_zeros(
                hidden_size, physical_intermediate_size)
            fc2_w_padded[:, :logical_intermediate_size] = fc2_w_raw
            fc2_w_raw = fc2_w_padded.contiguous()

        q_w, block._rpu_q_ws = _swz(q_w, tp_col_swizzle_mc_weight)
        k_w, block._rpu_k_ws = _swz(k_w, tp_col_swizzle_mc_weight)
        v_w, block._rpu_v_ws = _swz(v_w, tp_col_swizzle_mc_weight)

        block._rpu_q_w, block._rpu_k_w, block._rpu_v_w = q_w, k_w, v_w
        block._rpu_q_b, block._rpu_k_b, block._rpu_v_b = q_b, k_b, v_b

        o_w, block._rpu_o_ws = _swz(
            o_w_raw.contiguous(), tp_row_swizzle_mc_weight)
        o_b = o_b.contiguous()
        block.attn.proj.weight = nn.Parameter(o_w, requires_grad=False)
        block._rpu_o_b = o_b

        fc1_w, block._rpu_fc1_ws = _swz(
            fc1_w_raw.contiguous(), tp_col_swizzle_mc_weight)
        fc1_b = fc1_b_raw.contiguous()
        block.mlp.linear_fc1.weight = nn.Parameter(fc1_w, requires_grad=False)
        block._rpu_fc1_b = fc1_b

        fc2_w, block._rpu_fc2_ws = _swz(
            fc2_w_raw.contiguous(), tp_row_swizzle_mc_weight)
        fc2_b = fc2_b.contiguous()
        block.mlp.linear_fc2.weight = nn.Parameter(fc2_w, requires_grad=False)
        block._rpu_fc2_b = fc2_b

    block._rpu_qwen3vl_vision_logical_geometry = (
        logical_head_dim, logical_intermediate_size)
    block._rpu_qwen3vl_vision_physical_geometry = physical_geometry
    block._rpu_qwen3vl_vision_num_cores = num_cores
    block._rpu_qwen3vl_vision_weights_converted = bool(w8a16)


def _use_w8_vision_weight_banks(config, *, w8a16: bool,
                                    num_cores: int, num_layers: int) -> bool:
    return (w8a16 and num_cores == 8 and num_layers == 24
            and getattr(config, "depth", None) == 24
            and getattr(config, "hidden_size", None) == 1024
            and getattr(config, "intermediate_size", None) == 4096
            and getattr(config, "num_heads", None) == 16
            and getattr(config, "spatial_merge_size", None) == 2
            and getattr(config, "out_hidden_size", None) in (2048, 2560))


def _use_fp16_vision_weight_banks(config, *, w8a16: bool,
                                num_cores: int, num_layers: int) -> bool:
    return (not w8a16 and num_cores == 8 and num_layers == 24
            and getattr(config, "out_hidden_size", None) in (2048, 2560)
            and tuple(getattr(config, name, None) for name in (
                "model_type", "depth", "hidden_size", "intermediate_size", "num_heads",
                "patch_size", "temporal_patch_size", "spatial_merge_size",
                "in_channels", "num_position_embeddings", "hidden_act"))
            == ("qwen3_vl", 24, 1024, 4096, 16, 16, 2, 2, 3, 2304, "gelu_pytorch_tanh")
            and tuple(getattr(config, "deepstack_visual_indexes", ())) == (5, 11, 17))


def _vision_block_weights_to_rpu(weights, *, w8a16: bool, pack_weights: bool):
    """Keep one cold storage owner per admitted block; views preserve ABI."""
    if not pack_weights:
        return tuple(weight.to(device="rpu") if w8a16 else
                     weight.to(dtype=torch.float16, device="rpu") for weight in weights)
    dtype = torch.int8 if w8a16 else torch.float16
    element_size = 1 if w8a16 else 2
    if len(weights) != 6 or any(
            weight.device.type != "cpu" or weight.dtype != dtype
            or weight.dim() != 2 or not weight.is_contiguous() for weight in weights):
        raise ValueError(f"Vision block bank requires six contiguous CPU {dtype} matrices")
    offsets, end = [], 0
    for weight in weights:
        offset = (end + 255) // 256 * 256
        offsets.append(offset // element_size)
        end = offset + weight.numel() * element_size
    flat = torch.zeros((end + 255) // 256 * 256 // element_size, dtype=dtype, device="cpu")
    for weight, offset in zip(weights, offsets, strict=True):
        flat.narrow(0, offset, weight.numel()).copy_(weight.reshape(-1))
    bank = flat.to(device="rpu")
    return tuple(bank.narrow(0, offset, weight.numel()).view(weight.shape)
                 for weight, offset in zip(weights, offsets, strict=True))


def _fold_conv3d_to_linear_weight(conv3d_weight: torch.Tensor) -> torch.Tensor:
    """`[embed_dim, cin, tp, ps, ps]` Conv3d weight → `[embed_dim, cin*tp*ps*ps]`
    Linear weight.

    HF Qwen3VLVisionPatchEmbed processes flattened input `[N, cin*tp*ps*ps]` by
    `view(-1, cin, tp, ps, ps)` then Conv3d with kernel=stride=[tp, ps, ps]
    yielding `[N, embed_dim, 1, 1, 1]`. This is mathematically equivalent to a
    Linear `[embed_dim, cin*tp*ps*ps]` applied to `[N, cin*tp*ps*ps]` because
    the reshape `[N, cin*tp*ps*ps] → [N, cin, tp, ps, ps]` preserves memory
    order (contiguous strides match).
    """
    if conv3d_weight.dim() != 5:
        raise ValueError(
            f"_fold_conv3d_to_linear_weight: expected 5D weight, got {conv3d_weight.dim()}D"
        )
    embed_dim, cin, tp, ps_h, ps_w = conv3d_weight.shape
    return conv3d_weight.reshape(embed_dim, cin * tp * ps_h * ps_w).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Opt-in: run the patch-merger GEMMs on RPU (default keeps them CPU fp32)
# ─────────────────────────────────────────────────────────────────────────────

_VISION_MERGER_AUX_ATTRS = (
    "_rpu_merge_hidden", "_rpu_fc1_w", "_rpu_fc1_b", "_rpu_fc2_w", "_rpu_fc2_b",
    "_rpu_fc1_scale", "_rpu_fc2_scale", "_rpu_fc1_quantized", "_rpu_fc2_quantized",
    "_rpu_merger_w8a16", "_rpu_merger_on_device",
)


def _clear_vision_merger_aux(vision_model) -> None:
    """Release eager weights only after the composite's native retirement."""
    mergers = (getattr(vision_model, "merger", None),
               *getattr(vision_model, "deepstack_merger_list", ()))
    for merger in mergers:
        if merger is not None:
            for name in _VISION_MERGER_AUX_ATTRS:
                vars(merger).pop(name, None)


def _convert_vision_merger_for_rpu(merger, cores: int = 8, *, w8a16: bool = False) -> None:
    """In-place: stash RPU-resident, col-swizzled FP16 or W8 weights for a
    `Qwen3VLVisionPatchMerger`'s two GEMMs (`linear_fc1` → GELU → `linear_fc2`).

    The LayerNorm (`merger.norm`) is left on CPU fp32 — it is cheap and keeping
    its variance in fp32 matches the golden (wall_oss merger precedent,
    `adapters/wall_oss/vision.py`). Both GEMM shapes (`[merge_hidden, merge_hidden]`,
    `[out_hidden, merge_hidden]`) satisfy can_row && can_col so `rpu_linear`
    picks the col partition — hence `tp_col_swizzle_mc_weight`. Idempotent.
    """
    if getattr(merger, "_rpu_merger_on_device", False):
        if getattr(merger, "_rpu_merger_w8a16", False) != w8a16:
            raise ValueError("Vision merger weight precision cannot change after installation")
        return
    # Publish only complete copies; failed transfers leave CPU weights intact.
    with torch.no_grad():
        quantized = []
        scales = []
        weights = []
        for layer in (merger.linear_fc1, merger.linear_fc2):
            raw = layer.weight.data.half()
            if w8a16:
                raw, scale = _quantize_vision_w8_weight(raw)
                quantized.append(raw)
                scales.append(scale.to(device="rpu").contiguous())
                swizzled = tp_col_swizzle_mc_weight(raw, cores, dwidth=1)
            else:
                quantized.append(None)
                scales.append(None)
                swizzled = tp_col_swizzle_mc_weight(raw, cores)
            weights.append(swizzled.to(device="rpu").contiguous())
        fc1_w, fc2_w = weights
        fc1_b = merger.linear_fc1.bias.data.half().to(device="rpu").contiguous()
        fc2_b = merger.linear_fc2.bias.data.half().to(device="rpu").contiguous()
    vars(merger).update(
        _rpu_merge_hidden=int(merger.linear_fc1.in_features),
        _rpu_fc1_w=fc1_w, _rpu_fc1_b=fc1_b, _rpu_fc2_w=fc2_w, _rpu_fc2_b=fc2_b,
        _rpu_fc1_scale=scales[0], _rpu_fc2_scale=scales[1],
        _rpu_fc1_quantized=quantized[0], _rpu_fc2_quantized=quantized[1],
        _rpu_merger_w8a16=w8a16, _rpu_merger_on_device=True,
    )


def _merger_forward_on_device(
    merger, x_cpu_f32: torch.Tensor, *, keep_fp16: bool = False,
    linear_acc32: bool = False,
) -> torch.Tensor:
    """Mirror `Qwen3VLVisionPatchMerger.forward` with its GEMMs on RPU.

    `merger.norm` runs on CPU fp32 (as installed); the post-norm `[N/sm², merge_hidden]`
    activation hops to RPU fp16 for both Linear calls; eager exact-ERF GELU follows the
    registered CPU fallback. The result is an independent CPU FP32 snapshot by
    default, or FP16 for the internal text consumer. Requires
    `_convert_vision_merger_for_rpu`.
    """
    x = x_cpu_f32.view(-1, merger._rpu_merge_hidden) if merger.use_postshuffle_norm else x_cpu_f32
    xn = merger.norm(x).view(-1, merger._rpu_merge_hidden)          # CPU fp32 [N/sm², merge_hidden]
    xr = xn.to(device="rpu", dtype=torch.float16).contiguous()
    w8a16 = getattr(merger, "_rpu_merger_w8a16", False)
    xr = torch.ops.rpu.linear_with_accumulation(
        xr, merger._rpu_fc1_w, merger._rpu_fc1_b, 1, 8, linear_acc32,
        merger._rpu_fc1_scale if w8a16 else None)
    xr = torch.nn.functional.gelu(xr)  # graph-aware CPU exact-erf fallback
    xr = torch.ops.rpu.linear_with_accumulation(
        xr, merger._rpu_fc2_w, merger._rpu_fc2_b, 1, 8, linear_acc32,
        merger._rpu_fc2_scale if w8a16 else None)
    # RPU .cpu() may expose shared DDR. Both consumers need an independent
    # snapshot; the text consumer can keep the existing FP16 values instead
    # of expanding them to FP32 only to round back before its scatter.
    host = xr.cpu()
    return host.clone() if keep_fp16 else host.float()


def enable_qwen3_vl_vision_merger_on_device(vision_model, cores: int = 8, *, w8a16: bool = False) -> None:
    """Move patch and merger GEMMs to RPU before the first Vision dispatch.

    An already enabled install is a no-op. CPU FP32 LayerNorm is retained;
    patch/merger activations and GELU intermediates use FP16, changing rounding relative
    to CPU FP32 mergers. Validate this precision path for the exact model profile.
    """
    if type(w8a16) is not bool:
        raise TypeError("Vision auxiliary w8a16 must be a bool")
    if cores != 8 or getattr(vision_model, "_rpu_vision_num_cores", 8) != 8:
        raise ValueError("reduced-core Vision keeps patch embedding and mergers on CPU")
    if (getattr(vision_model, "_rpu_patch_embed_on_device", False)
            and getattr(vision_model, "_rpu_vision_aux_w8a16", False) != w8a16):
        raise ValueError("Vision auxiliary weight precision cannot change after installation")
    mergers = (vision_model.merger, *vision_model.deepstack_merger_list)
    if (getattr(vision_model, "_rpu_vision_merger_on_device", False)
            and getattr(vision_model, "_rpu_patch_embed_on_device", False)
            and all(getattr(merger, "_rpu_merger_on_device", False) for merger in mergers)):
        return
    if getattr(vision_model, "_rpu_vision_handle", None) is None:
        raise RuntimeError("Vision merger acceleration requires a live installed Vision handle")
    if getattr(vision_model, "_rpu_vision_has_dispatched", False):
        raise RuntimeError("Vision merger acceleration must be enabled before the first Vision dispatch")
    if getattr(vision_model, "_rpu_vision_host_fp32_patch", False):
        raise ValueError("Vision merger acceleration cannot override the CPU FP32 patch profile")
    patch_attrs = ("_rpu_patch_embed_w_rpu", "_rpu_patch_embed_b_rpu",
                   "_rpu_patch_embed_on_device", "_rpu_vision_merger_on_device",
                   "_rpu_vision_patch_embed_scale", "_rpu_vision_aux_w8a16")
    scopes = [(merger, _VISION_MERGER_AUX_ATTRS) for merger in mergers]
    scopes.append((vision_model, patch_attrs))
    snapshots = [(owner, attrs, {name: vars(owner)[name] for name in attrs if name in vars(owner)})
                 for owner, attrs in scopes]
    try:
        for merger in mergers:
            _convert_vision_merger_for_rpu(merger, cores, w8a16=w8a16)
        # Folded Conv3d weights MUST be col-swizzled for rpu_linear.
        if not getattr(vision_model, "_rpu_patch_embed_on_device", False):
            pe_w = vision_model._rpu_vision_patch_embed_w
            pe_b = vision_model._rpu_vision_patch_embed_b
            if w8a16:
                pe_w, pe_scale = _quantize_vision_w8_weight(pe_w)
                pe_w_rpu = tp_col_swizzle_mc_weight(pe_w, cores, dwidth=1).to(device="rpu").contiguous()
                pe_scale_rpu = pe_scale.to(device="rpu").contiguous()
            else:
                pe_w_rpu = tp_col_swizzle_mc_weight(pe_w.half(), cores).to(device="rpu").contiguous()
                pe_scale_rpu = None
            pe_b_rpu = pe_b.half().to(device="rpu").contiguous() if pe_b is not None else None
            vision_model._rpu_patch_embed_w_rpu = pe_w_rpu
            vision_model._rpu_patch_embed_b_rpu = pe_b_rpu
            vision_model._rpu_vision_patch_embed_scale = pe_scale_rpu
            vision_model._rpu_patch_embed_on_device = True
        vision_model._rpu_vision_aux_w8a16 = w8a16
        vision_model._rpu_vision_merger_on_device = True
    except BaseException:
        for owner, attrs, snapshot in snapshots:
            for name in attrs:
                vars(owner).pop(name, None)
            vars(owner).update(snapshot)
        raise


def register_qwen3vl_vision_fused_merger(vision_model, cores: int = 8, *, w8a16: bool = False) -> None:
    """Register the patch-merger weights for the in-graph fused merger (ROUND-3 opt #1, gr00t).

    Call AFTER `install_qwen3_vl_vision_for_rpu` and BEFORE the first forward. m0
    (linear_fc1) is col-swizzled, m2
    (linear_fc2) is row-swizzled, ln_q (LayerNorm γ/β) raw [HID]. Registering
    these weights is the native post_fn authority; the post_fn then runs
    the patch merger AND (inc-2) the N deepstack mergers inside the vision graph;
    `qwen3vl_vision_forward` returns the merged pooler `[seq/sm², out_hidden]` and the merged
    deepstack features are popped via `qwen3vl_vision_pop_deepstack_merged`. The forward
    replacement reads `_rpu_vision_fused_merger` to consume both. W8 installs
    can use a post-encoder merger over the resolved compute chunks under the
    controlled 4B profile; other multi-chunk installs retain eager mergers.
    Idempotent.
    """
    if type(w8a16) is not bool:
        raise TypeError("Vision fused merger w8a16 must be a bool")
    if cores != 8 or getattr(vision_model, "_rpu_vision_num_cores", 8) != 8:
        raise ValueError("reduced-core Vision keeps patch embedding and mergers on CPU")
    if getattr(vision_model, "_rpu_vision_fused_merger", False):
        if getattr(vision_model, "_rpu_vision_fused_merger_w8a16", False) != w8a16:
            raise ValueError("Vision fused merger precision cannot change after installation")
        return

    if getattr(vision_model, "_rpu_vision_has_dispatched", False):
        raise RuntimeError(
            "register_qwen3vl_vision_fused_merger must be called before the first "
            "vision forward because it changes the persistent SPM layout")
    if w8a16 and not getattr(vision_model, "_rpu_vision_aux_w8a16", False):
        raise RuntimeError("Enable W8 Vision patch and eager mergers before registering W8 fusion")

    def _merger_rpu_weights(mg):
        if w8a16:
            _convert_vision_merger_for_rpu(mg, cores, w8a16=True)
            return (
                mg.norm.weight.data.detach().half().to(device="rpu").contiguous(),
                mg.norm.bias.data.detach().half().to(device="rpu").contiguous(),
                mg._rpu_fc1_w, mg._rpu_fc1_b,
                tp_row_swizzle_mc_weight(mg._rpu_fc2_quantized, cores, dwidth=1).to(device="rpu").contiguous(),
                mg._rpu_fc2_b, mg._rpu_fc1_scale, mg._rpu_fc2_scale,
            )
        # (ln_q_w, ln_q_b, m0_w[col], m0_b, m2_w[row], m2_b) — all fp16 RPU.
        return (
            mg.norm.weight.data.detach().half().to(device="rpu").contiguous(),
            mg.norm.bias.data.detach().half().to(device="rpu").contiguous(),
            tp_col_swizzle_mc_weight(mg.linear_fc1.weight.data.detach().half(), cores).to(device="rpu").contiguous(),
            mg.linear_fc1.bias.data.detach().half().to(device="rpu").contiguous(),
            tp_row_swizzle_mc_weight(mg.linear_fc2.weight.data.detach().half(), cores).to(device="rpu").contiguous(),
            mg.linear_fc2.bias.data.detach().half().to(device="rpu").contiguous(),
        )

    with torch.no_grad():
        m = vision_model.merger
        pw = _merger_rpu_weights(m)
        dsw = [_merger_rpu_weights(dm) for dm in vision_model.deepstack_merger_list]
        torch.ops.rpu.qwen3vl_vision_set_merger_weights(
            vision_model._rpu_vision_handle, pw[0], pw[1], pw[2], pw[3], pw[4], pw[5],
            int(m.linear_fc2.out_features), float(m.norm.eps),
            *((pw[6], pw[7]) if w8a16 else ()))
        # inc-2: the N deepstack mergers (same structure, applied to the layer-{5,11,17} snapshots).
        if dsw:
            torch.ops.rpu.qwen3vl_vision_set_deepstack_merger_weights(
                vision_model._rpu_vision_handle,
                [d[0] for d in dsw], [d[1] for d in dsw], [d[2] for d in dsw],
                [d[3] for d in dsw], [d[4] for d in dsw], [d[5] for d in dsw],
                *(([d[6] for d in dsw], [d[7] for d in dsw]) if w8a16 else ()))
    vision_model._rpu_vision_fused_merger = True
    vision_model._rpu_vision_fused_merger_w8a16 = w8a16
    vision_model._rpu_vision_graph_cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Per-forward CPU compute: position_idx
# ─────────────────────────────────────────────────────────────────────────────

def _compute_vision_position_idx_cpu(
    grid_thw_cpu: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Build `[num_patches, 2]` int16 (row_idx, col_idx) on CPU from grid_thw.

    Mirrors HF `Qwen3VLVisionModel.rot_pos_emb` body — same `coords` layout
    (row, col stacked along the trailing dim of size 2). The RPU rope_2d
    kernel does the `freq_table[pos_ids]` lookup internally via position_idx
    + FreqCos/Sin table indexing.
    """
    if grid_thw_cpu.dim() != 2 or grid_thw_cpu.size(-1) != 3:
        raise ValueError(
            f"_compute_vision_position_idx_cpu: grid_thw must be [n, 3], got {tuple(grid_thw_cpu.shape)}"
        )
    grid_thw_list = grid_thw_cpu.tolist()
    merge_size = int(spatial_merge_size)

    total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.int64)

    offset = 0
    for num_frames, height, width in grid_thw_list:
        merged_h = height // merge_size
        merged_w = width // merge_size

        block_rows = torch.arange(merged_h)
        block_cols = torch.arange(merged_w)
        intra_row = torch.arange(merge_size)
        intra_col = torch.arange(merge_size)

        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)
        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    if pos_ids.max().item() >= 2**15:
        raise ValueError(
            f"_compute_vision_position_idx_cpu: max idx {pos_ids.max().item()} "
            f"exceeds int16 range"
        )
    return pos_ids.to(torch.int16).contiguous()


def _prepare_qwen3vl_vision_input_for_spm_pipeline(
    vision_model,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    device_patch_embed: bool = False,
) -> torch.Tensor:
    """Prepare exact equal-grid Vision inputs for an outer SPM pipeline.

    This helper is deliberately narrower than :func:`_rpu_vision_forward`:
    it accepts either one image or the LingBot2 three-camera pack, all with
    ``grid_thw=[1,16,16]`` (256 patches/image).  It runs one installed folded
    patch embedding through either the default CPU FP16 path or an explicitly
    selected RPU device path, stages the model-owned RoPE position keepalive
    once, and returns contiguous RPU FP16
    ``[image_count,256,1024]`` storage.  Packing here does not batch the native
    Vision encoder: the outer owner still consumes three sequential N=256
    views, so its Graph/SPM topology is unchanged.

    The caller must invoke it before opening the composite Graph/SPM
    lifecycle.  It neither enters the Vision GraphCache nor dispatches the
    native Vision forward, pops/clones outputs, resets caches, or touches SPM
    ownership.
    """
    error_prefix = "Qwen3-VL SPM pipeline Vision input"

    handle = getattr(vision_model, "_rpu_vision_handle", None)
    if type(handle) is not int or handle <= 0:
        raise RuntimeError(
            f"{error_prefix}: install_qwen3_vl_vision_for_rpu must publish "
            "a live handle before input preparation"
        )

    if not isinstance(grid_thw, torch.Tensor):
        raise TypeError(f"{error_prefix}: grid_thw must be a torch.Tensor")
    if grid_thw.device.type != "cpu":
        raise ValueError(f"{error_prefix}: grid_thw must remain on CPU")
    if (
        grid_thw.dim() != 2
        or grid_thw.size(1) != 3
        or grid_thw.size(0) not in (1, 3)
    ):
        raise ValueError(
            f"{error_prefix}: grid_thw must describe exactly one or three "
            f"images, got shape={tuple(grid_thw.shape)}"
        )
    image_count = int(grid_thw.size(0))
    if grid_thw.tolist() != [[1, 16, 16]] * image_count:
        raise ValueError(
            f"{error_prefix}: every image must use grid_thw=[1,16,16], "
            f"got value={grid_thw.tolist()}"
        )

    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError(f"{error_prefix}: hidden_states must be a torch.Tensor")
    if hidden_states.device.type != "cpu" or hidden_states.dtype != torch.float16:
        raise ValueError(
            f"{error_prefix}: hidden_states must be CPU FP16 raw patches"
        )
    if (
        hidden_states.dim() != 2
        or hidden_states.size(0) != image_count * 256
    ):
        raise ValueError(
            f"{error_prefix}: hidden_states must be flat "
            f"[{image_count * 256},patch_dim], got {tuple(hidden_states.shape)}"
        )

    hidden_size = getattr(vision_model, "_rpu_vision_hidden_size", None)
    spatial_merge_size = getattr(
        vision_model, "_rpu_vision_spatial_merge_size", None
    )
    if hidden_size != 1024 or spatial_merge_size != 2:
        raise RuntimeError(
            f"{error_prefix}: exact contract requires hidden_size=1024 and "
            f"spatial_merge_size=2, got {hidden_size!r} and {spatial_merge_size!r}"
        )
    if int(getattr(vision_model, "_rpu_vision_max_hw", 0)) < 16:
        raise RuntimeError(f"{error_prefix}: installed max_hw must cover 16")
    if bool(getattr(vision_model, "_rpu_vision_rope_disable", False)):
        raise RuntimeError(f"{error_prefix}: RoPE-disabled debug mode is unsupported")

    patch_weight = getattr(vision_model, "_rpu_vision_patch_embed_w", None)
    patch_bias = getattr(vision_model, "_rpu_vision_patch_embed_b", None)
    if (
        not isinstance(patch_weight, torch.Tensor)
        or patch_weight.device.type != "cpu"
        or patch_weight.dtype != torch.float16
        or patch_weight.dim() != 2
        or tuple(patch_weight.shape[:1]) != (1024,)
        or patch_weight.size(1) != hidden_states.size(1)
        or not patch_weight.is_contiguous()
    ):
        raise RuntimeError(
            f"{error_prefix}: installed folded patch weight must be contiguous CPU "
            "FP16 [1024, patch_dim] and match hidden_states"
        )
    if patch_bias is not None and (
        not isinstance(patch_bias, torch.Tensor)
        or patch_bias.device.type != "cpu"
        or patch_bias.dtype != torch.float16
        or tuple(patch_bias.shape) != (1024,)
        or not patch_bias.is_contiguous()
    ):
        raise RuntimeError(
            f"{error_prefix}: installed patch bias must be contiguous CPU FP16 [1024]"
        )

    patch_weight_rpu = None
    patch_bias_rpu = None
    if device_patch_embed:
        patch_weight_rpu = getattr(
            vision_model, "_rpu_patch_embed_w_rpu", None
        )
        patch_bias_rpu = getattr(
            vision_model, "_rpu_patch_embed_b_rpu", None
        )
        if (
            not isinstance(patch_weight_rpu, torch.Tensor)
            or patch_weight_rpu.device.type != "rpu"
            or patch_weight_rpu.dtype != torch.float16
            or tuple(patch_weight_rpu.shape)
            != (1024, hidden_states.size(1))
            or not patch_weight_rpu.is_contiguous()
        ):
            raise RuntimeError(
                f"{error_prefix}: installed device patch weight must be contiguous "
                "RPU FP16 [1024, patch_dim] and match hidden_states"
            )
        if (patch_bias is None) != (patch_bias_rpu is None):
            raise RuntimeError(
                f"{error_prefix}: installed CPU and device patch bias presence "
                "must match"
            )
        if patch_bias_rpu is not None and (
            not isinstance(patch_bias_rpu, torch.Tensor)
            or patch_bias_rpu.device.type != "rpu"
            or patch_bias_rpu.dtype != torch.float16
            or tuple(patch_bias_rpu.shape) != (1024,)
            or not patch_bias_rpu.is_contiguous()
        ):
            raise RuntimeError(
                f"{error_prefix}: installed device patch bias must be contiguous "
                "RPU FP16 [1024]"
            )

    keepalive = getattr(
        vision_model, "_rpu_vision_position_idx_keepalive", None
    )
    if (
        not isinstance(keepalive, torch.Tensor)
        or keepalive.device.type != "rpu"
        or keepalive.dtype != torch.int16
        or keepalive.dim() != 2
        or keepalive.size(0) < 256
        or keepalive.size(1) != 2
        or not keepalive.is_contiguous()
    ):
        raise RuntimeError(
            f"{error_prefix}: installed position_idx keepalive must be contiguous "
            "RPU int16 [capacity>=256,2]"
        )

    # Use the same memo fields as the ordinary forward without calling it or
    # its GraphCache.  A canonical owned key avoids retaining caller metadata.
    canonical_grid = torch.tensor(
        [[1, 16, 16]] * image_count, dtype=torch.long
    )
    position_key = getattr(vision_model, "_rpu_vision_pos_embeds_key", None)
    if (
        not isinstance(position_key, torch.Tensor)
        or not torch.equal(position_key, canonical_grid)
        or not hasattr(vision_model, "_rpu_vision_pos_embeds")
    ):
        position_cpu = type(vision_model).fast_pos_embed_interpolate(
            vision_model, canonical_grid
        )
        if (
            not isinstance(position_cpu, torch.Tensor)
            or position_cpu.device.type != "cpu"
            or position_cpu.dtype != torch.float16
            or tuple(position_cpu.shape) != (image_count * 256, 1024)
        ):
            raise RuntimeError(
                f"{error_prefix}: fast_pos_embed_interpolate must return CPU "
                f"FP16 [{image_count * 256},1024]"
            )
        vision_model._rpu_vision_pos_embeds = position_cpu.to(
            device="rpu", dtype=torch.float16
        ).contiguous()
        vision_model._rpu_vision_pos_embeds_cpu32 = (
            position_cpu.float().contiguous()
        )
        vision_model._rpu_vision_pos_embeds_key = canonical_grid

    position_rpu = vision_model._rpu_vision_pos_embeds
    if (
        not isinstance(position_rpu, torch.Tensor)
        or position_rpu.device.type != "rpu"
        or position_rpu.dtype != torch.float16
        or tuple(position_rpu.shape) != (image_count * 256, 1024)
        or not position_rpu.is_contiguous()
    ):
        raise RuntimeError(
            f"{error_prefix}: memoized position embedding must be contiguous "
            f"RPU FP16 [{image_count * 256},1024]"
        )

    position_idx_cpu = _compute_vision_position_idx_cpu(
        canonical_grid.narrow(0, 0, 1), spatial_merge_size
    )
    if tuple(position_idx_cpu.shape) != (256, 2):
        raise RuntimeError(
            f"{error_prefix}: position_idx must be [256,2], got "
            f"{tuple(position_idx_cpu.shape)}"
        )
    keepalive.narrow(0, 0, 256).copy_(position_idx_cpu.to(device="rpu"))
    vision_model._rpu_vision_pos_idx_key = (1, 16, 16, 1, 2)
    vision_model._rpu_vision_pos_idx_val = position_idx_cpu
    vision_model._rpu_vision_keepalive_key = ((1, 16, 16, 1, 2), 256, False)

    with torch.no_grad():
        if device_patch_embed:
            patch_input_rpu = hidden_states.contiguous().to(
                device="rpu", dtype=torch.float16, non_blocking=False
            ).contiguous()
            patch_rpu = torch.nn.functional.linear(
                patch_input_rpu, patch_weight_rpu, patch_bias_rpu
            ).contiguous()
        else:
            patch_cpu = torch.nn.functional.linear(
                hidden_states.contiguous(), patch_weight, patch_bias
            )
            if (
                patch_cpu.dtype != torch.float16
                or tuple(patch_cpu.shape) != (image_count * 256, 1024)
            ):
                raise RuntimeError(
                    f"{error_prefix}: CPU patch embedding must produce FP16 "
                    f"[{image_count * 256},1024]"
                )
            patch_rpu = patch_cpu.to(
                device="rpu", dtype=torch.float16, non_blocking=False
            ).contiguous()
        if (
            patch_rpu.device.type != "rpu"
            or patch_rpu.dtype != torch.float16
            or tuple(patch_rpu.shape) != (image_count * 256, 1024)
            or not patch_rpu.is_contiguous()
        ):
            raise RuntimeError(
                f"{error_prefix}: patch embedding must produce contiguous RPU "
                f"FP16 [{image_count * 256},1024]"
            )
        prepared = (patch_rpu + position_rpu).view(
            image_count, 256, 1024
        ).contiguous()

    if (
        prepared.device.type != "rpu"
        or prepared.dtype != torch.float16
        or tuple(prepared.shape) != (image_count, 256, 1024)
        or not prepared.is_contiguous()
    ):
        raise RuntimeError(
            f"{error_prefix}: prepared input must be contiguous RPU FP16 "
            f"[{image_count},256,1024]"
        )
    return prepared


# ─────────────────────────────────────────────────────────────────────────────
# Main install function
# ─────────────────────────────────────────────────────────────────────────────

_VISION_INSTALL_EXACT_ATTRS = frozenset({
    "_rpu_lazy_init_checked",
    "_rpu_required_attrs",
    "_rpu_patch_embed_on_device",
    "_rpu_patch_embed_w_rpu",
    "_rpu_patch_embed_b_rpu",
    "_rpu_patch_embed_output_slot",
})


def _is_vision_install_attr(name: str) -> bool:
    return name.startswith("_rpu_vision_") or name in _VISION_INSTALL_EXACT_ATTRS


def _clear_vision_install_attrs(vision_model) -> None:
    state = vars(vision_model)
    for name in tuple(state):
        if _is_vision_install_attr(name):
            state.pop(name, None)


def install_qwen3_vl_vision_for_rpu(
    vision_model,
    *,
    num_cores: int = 8,
    vision_config: Any | None = None,
    max_hw: int = 48,
    max_seq_len: int = 2048,
    w8a16: bool = False,
    execution_chunk_size: str | int = "auto",
    _allow_graph_blocked_32b: bool = False,
    _allow_padded_8b: bool = False,
    _allow_runtime_quantized_large: bool = False,
    _pack_fp16_block_weights: bool = False,
    _graph_cache_max_entries: int | None = None,
) -> int:
    """Install or replace the Qwen3-VL vision runtime transactionally.

    A replacement is published only after its Python state and native handle
    are fully configured. Any failure destroys the pending handle immediately
    and restores the previously published install.
    """
    if _graph_cache_max_entries is not None and (
        type(_graph_cache_max_entries) is not int or _graph_cache_max_entries < 1
    ):
        raise ValueError("_graph_cache_max_entries must be a positive integer or None")
    _validate_vision_core_count(num_cores)
    installed_cores = getattr(vision_model, "_rpu_vision_num_cores", 8)
    if (hasattr(vision_model, "_rpu_vision_handle") and installed_cores != num_cores):
        raise ValueError("Vision core count cannot change after installation")
    if num_cores == 4:
        if w8a16 or _allow_graph_blocked_32b or _allow_runtime_quantized_large:
            raise ValueError("four-core Vision only admits unquantized 2B/4B/8B")
        if not hasattr(vision_model, "_rpu_vision_handle"):
            validate_qwen3_vl_vision_core_profile(
                vision_model, num_cores=num_cores, vision_config=vision_config)
        elif tuple(torch.ops.rpu.qwen3vl_vision_topology(vision_model._rpu_vision_handle)) != (4, 4, 4, 8):
            raise ValueError("installed four-core Vision native topology drifted")
    if not isinstance(w8a16, bool):
        raise TypeError(
            "install_qwen3_vl_vision_for_rpu: w8a16 must be a bool, "
            f"got {type(w8a16).__name__}."
        )
    if execution_chunk_size != "auto" and (
        isinstance(execution_chunk_size, bool)
        or not isinstance(execution_chunk_size, int)
        or execution_chunk_size <= 0
        or execution_chunk_size % 16
    ):
        raise ValueError(
            "install_qwen3_vl_vision_for_rpu: execution_chunk_size must "
            "be 'auto' or a positive multiple of 16, got "
            f"{execution_chunk_size!r}"
        )
    if not isinstance(_allow_graph_blocked_32b, bool):
        raise TypeError(
            "install_qwen3_vl_vision_for_rpu: "
            "_allow_graph_blocked_32b must be a bool."
        )
    if not isinstance(_allow_padded_8b, bool):
        raise TypeError(
            "install_qwen3_vl_vision_for_rpu: "
            "_allow_padded_8b must be a bool."
        )
    if not isinstance(_allow_runtime_quantized_large, bool):
        raise TypeError("_allow_runtime_quantized_large must be a bool")
    cfg = vision_config or vision_model.config
    if not isinstance(_pack_fp16_block_weights, bool):
        raise TypeError("_pack_fp16_block_weights must be a bool")
    if _pack_fp16_block_weights and not (
        _use_fp16_vision_weight_banks(
            cfg, w8a16=w8a16, num_cores=num_cores, num_layers=len(vision_model.blocks))
        or (_allow_padded_8b and not w8a16 and num_cores == 8
            and len(vision_model.blocks) == 27 and _is_qwen3_vl_8b_vision_config(cfg)
            and tuple(getattr(cfg, name, None) for name in (
                "model_type", "in_channels", "num_position_embeddings"))
            == ("qwen3_vl", 3, 2304))
    ):
        raise ValueError(
            "Vision weight banks require the exact eight-core 2B/4B FP16 tower "
            "or gated padded 8B FP16 tower")
    is_32b_vision = _is_qwen3_vl_32b_vision_config(cfg)
    is_8b_vision = _is_qwen3_vl_8b_vision_config(cfg)
    if is_8b_vision and not (_allow_padded_8b or _allow_runtime_quantized_large):
        raise ValueError(
            "install_qwen3_vl_vision_for_rpu: the exact Qwen3-VL-8B vision "
            "tower may only be installed by the gated top-level adapter."
        )
    if is_32b_vision and not (_allow_graph_blocked_32b or _allow_runtime_quantized_large):
        raise ValueError(
            "install_qwen3_vl_vision_for_rpu: the Qwen3-VL-32B vision tower "
            "may only be installed by the gated top-level adapter."
        )
    if _allow_runtime_quantized_large and not (
        w8a16 and num_cores == 8 and len(vision_model.blocks) == 27
        and (is_8b_vision or is_32b_vision)
        and all(type(getattr(cfg, name, None)) is int for name in (
            "hidden_size", "intermediate_size", "depth", "num_heads", "patch_size",
            "temporal_patch_size", "spatial_merge_size", "out_hidden_size",
            "in_channels", "num_position_embeddings"))
        and all(type(index) is int for index in cfg.deepstack_visual_indexes)
        and not _allow_padded_8b and not _allow_graph_blocked_32b
        and tuple(getattr(cfg, name, None) for name in (
            "model_type", "in_channels", "num_position_embeddings"))
        == ("qwen3_vl", 3, 2304)
    ):
        raise ValueError("runtime quantized large Vision requires the exact TP8 8B/32B W8 tower")
    if _allow_graph_blocked_32b and (
        not is_32b_vision
        or os.environ.get(_QWEN3_VL_32B_GRAPH_BLOCKED_ENV) != "1"
        or w8a16
    ):
        raise ValueError(
            "install_qwen3_vl_vision_for_rpu: the private 32B path requires "
            "the exact logical profile, exact opt-in value '1', and an FP16 "
            "vision tower."
        )
    if _allow_padded_8b and (
        not is_8b_vision or _allow_graph_blocked_32b or w8a16
    ):
        raise ValueError(
            "install_qwen3_vl_vision_for_rpu: the private 8B padded path "
            "requires the exact logical profile and an FP16 vision tower."
        )
    # The top-level Qwen3-VL profile checks this too, but GR00T and LingBot
    # call the shared installer directly.  Fail before irreversible swizzling
    # because the native Vision block implements tanh-approximate GELU.
    if getattr(cfg, "hidden_act", None) != "gelu_pytorch_tanh":
        raise ValueError(
            "install_qwen3_vl_vision_for_rpu: hidden_act must be "
            f"'gelu_pytorch_tanh', got {getattr(cfg, 'hidden_act', None)!r}."
        )
    host_fp32_patch = rpu_env_bool("RPU_QWEN3VL_VISION_HOST_FP32_PATCH")
    vision_batch = rpu_env_bool("RPU_QWEN3VL_VISION_BATCH")
    if num_cores != 8 and vision_batch:
        raise ValueError("four-core Vision requires per-image execution; packed image batches require 8 cores")
    try:
        vision_batch_cap = int(os.environ.get("RPU_QWEN3VL_VISION_BATCH_CAP", "3"))
    except ValueError as exc:
        raise ValueError(
            "RPU_QWEN3VL_VISION_BATCH_CAP must be a positive integer"
        ) from exc
    if vision_batch_cap <= 0:
        raise ValueError("RPU_QWEN3VL_VISION_BATCH_CAP must be a positive integer")
    model_vars = vars(vision_model)
    snapshot = {
        name: value
        for name, value in model_vars.items()
        if _is_vision_install_attr(name)
    }
    had_instance_forward = "forward" in model_vars
    old_instance_forward = model_vars.get("forward")
    install_state = {
        "old_handle": snapshot.get("_rpu_vision_handle"),
        "had_old_handle": "_rpu_vision_handle" in snapshot,
        "old_finalizer": snapshot.get("_rpu_vision_handle_finalizer"),
        "old_resource": snapshot.get("_rpu_vision_retirement_state"),
    }
    if install_state["old_resource"] is not None:
        install_state["old_resource"].require_replaceable()
    else:
        from rpu_backend.api._execution import _require_execution_process_safe
        _require_execution_process_safe()
    try:
        return _install_qwen3_vl_vision_for_rpu_impl(
            vision_model,
            num_cores=num_cores,
            vision_config=vision_config,
            max_hw=max_hw,
            max_seq_len=max_seq_len,
            w8a16=w8a16,
            execution_chunk_size=execution_chunk_size,
            host_fp32_patch=host_fp32_patch,
            vision_batch=vision_batch,
            vision_batch_cap=vision_batch_cap,
            _allow_graph_blocked_32b=_allow_graph_blocked_32b,
            _allow_padded_8b=_allow_padded_8b,
            _allow_runtime_quantized_large=_allow_runtime_quantized_large,
            _pack_fp16_block_weights=_pack_fp16_block_weights,
            _graph_cache_max_entries=_graph_cache_max_entries,
            _install_state=install_state,
        )
    except BaseException as error:
        if not install_state.get("committed", False):
            pending = install_state.get("pending_resource")
            if pending is not None:
                pending.cleanup_failure(error, vision_model, snapshot)

            state = vars(vision_model)
            for name in tuple(state):
                if _is_vision_install_attr(name):
                    state.pop(name, None)
            state.update(snapshot)
            if had_instance_forward:
                state["forward"] = old_instance_forward
            else:
                state.pop("forward", None)
        raise


def _make_dummy_vision_kv_caches(
    num_layers: int,
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
    *, num_cores: int = 8,
) -> RPUCache:
    """Create an RPUCache sized for bidirectional vision SDPA.

    Vision encoder always inserts at position=0 and reads all tokens once —
    no autoregressive growth. We size the cache to `max_seq_len` (worst-case
    num_patches) so any forward shape fits. The cache stays alive across
    forwards; `reset_to_position(0)` is called per forward.
    """
    return RPUCache(
        num_layers=num_layers,
        batch_size=1,
        max_seq_len=max_seq_len,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        attn_tp=min(num_cores, num_heads),
    )


def _install_qwen3_vl_vision_for_rpu_impl(
    vision_model,
    *,
    num_cores: int = 8,
    vision_config: Any | None = None,
    max_hw: int = 48,
    max_seq_len: int = 2048,
    w8a16: bool = False,
    execution_chunk_size: str | int = "auto",
    host_fp32_patch: bool,
    vision_batch: bool,
    vision_batch_cap: int,
    _allow_graph_blocked_32b: bool = False,
    _allow_padded_8b: bool = False,
    _allow_runtime_quantized_large: bool = False,
    _pack_fp16_block_weights: bool = False,
    _graph_cache_max_entries: int | None = None,
    _install_state: dict[str, Any],
) -> int:
    """Install RPU all-layers-once forward on a `Qwen3VLVisionModel` instance.

    Prerequisites:
        - vision_model.to('rpu') NOT required (this function moves weights).
        - vision_model.eval() recommended (avoids dropout / autograd state).

    Args:
        vision_model: Qwen3VLVisionModel (NOT the wrapping
            Qwen3VLForConditionalGeneration). Pass `model.visual` from the HF
            model after `from_pretrained`.
        vision_config: optional, defaults to `vision_model.config`.
        max_hw: max(h, w) bound for FreqCos/Sin table size. Default 48 covers
            up to 768×768 px with patch_size=16.
        max_seq_len: RPUCache capacity. Default 2048. Equal-size image batches
            are split into groups that fit this capacity; a single image that
            exceeds it fails before dispatch. The C++ side also caps position
            keepalive at QWEN3VL_VISION_MAX_KEEPALIVE_SEQ (4096).

    Returns the C++ handle (also stashed at `vision_model._rpu_vision_handle`).
    """
    cfg = vision_config or vision_model.config

    num_layers = len(vision_model.blocks)
    num_heads = cfg.num_heads
    hidden_size = cfg.hidden_size
    logical_intermediate_size = cfg.intermediate_size
    logical_head_dim = hidden_size // num_heads
    use_physical_padding = (_allow_graph_blocked_32b or _allow_padded_8b
                            or _allow_runtime_quantized_large)
    physical_head_dim = (
        _QWEN3_VL_32B_PHYSICAL_HEAD_DIM
        if use_physical_padding else logical_head_dim
    )
    physical_intermediate_size = (
        _QWEN3_VL_32B_PHYSICAL_INTERMEDIATE_SIZE
        if use_physical_padding else logical_intermediate_size
    )
    spatial_merge_size = cfg.spatial_merge_size
    eps = 1e-6  # Qwen3VLVisionBlock LayerNorm eps (hardcoded in HF)
    deepstack_visual_indexes = list(cfg.deepstack_visual_indexes)

    # ------------------------------------------------------------------ #
    # Step 1: prepare encoder weights while the old install remains published.
    # ------------------------------------------------------------------ #
    if w8a16:
        # Reject a corrupt later layer before an earlier block is swizzled.
        for block in vision_model.blocks:
            if getattr(block, "_rpu_qwen3vl_vision_weights_converted", None) is None:
                _validate_vision_w8_block_source(block)
    for block in vision_model.blocks:
        _convert_vision_block_weights_for_rpu(
            block,
            num_heads,
            hidden_size,
            w8a16=w8a16,
            physical_head_dim=physical_head_dim,
            physical_intermediate_size=physical_intermediate_size,
            **({"num_cores": num_cores} if num_cores != 8 else {}),
        )

    # ------------------------------------------------------------------ #
    # Step 2: gather per-layer weights into 16 lists (one per arg)
    # ------------------------------------------------------------------ #
    q_w_list, k_w_list, v_w_list, o_w_list = [], [], [], []
    fc1_w_list, fc2_w_list = [], []
    ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list = [], [], [], []
    q_b_list, k_b_list, v_b_list, o_b_list = [], [], [], []
    fc1_b_list, fc2_b_list = [], []
    # W8A16: parallel int8 weights stay int8 (no fp16 cast) + per-output-channel fp16 scales.
    q_s_list, k_s_list, v_s_list, o_s_list, fc1_s_list, fc2_s_list = [], [], [], [], [], []

    pack_w8_blocks = (_allow_runtime_quantized_large or _use_w8_vision_weight_banks(
        cfg, w8a16=w8a16, num_cores=num_cores, num_layers=num_layers))
    for block in vision_model.blocks:
        block_weights = _vision_block_weights_to_rpu(
            (block._rpu_q_w, block._rpu_k_w, block._rpu_v_w,
             block.attn.proj.weight.data, block.mlp.linear_fc1.weight.data,
             block.mlp.linear_fc2.weight.data),
            w8a16=w8a16, pack_weights=pack_w8_blocks or _pack_fp16_block_weights)
        for weights, weight in zip(
                (q_w_list, k_w_list, v_w_list, o_w_list, fc1_w_list, fc2_w_list),
                block_weights, strict=True):
            weights.append(weight)

        if w8a16:
            q_s_list.append(block._rpu_q_ws.to(device="rpu"))
            k_s_list.append(block._rpu_k_ws.to(device="rpu"))
            v_s_list.append(block._rpu_v_ws.to(device="rpu"))
            o_s_list.append(block._rpu_o_ws.to(device="rpu"))
            fc1_s_list.append(block._rpu_fc1_ws.to(device="rpu"))
            fc2_s_list.append(block._rpu_fc2_ws.to(device="rpu"))

        ln1_w_list.append(block.norm1.weight.to(dtype=torch.float16, device="rpu"))
        ln1_b_list.append(block.norm1.bias.to(dtype=torch.float16, device="rpu"))
        ln2_w_list.append(block.norm2.weight.to(dtype=torch.float16, device="rpu"))
        ln2_b_list.append(block.norm2.bias.to(dtype=torch.float16, device="rpu"))

        q_b_list.append(block._rpu_q_b.to(dtype=torch.float16, device="rpu"))
        k_b_list.append(block._rpu_k_b.to(dtype=torch.float16, device="rpu"))
        v_b_list.append(block._rpu_v_b.to(dtype=torch.float16, device="rpu"))
        o_b_list.append(block._rpu_o_b.to(dtype=torch.float16, device="rpu"))

        fc1_b_list.append(block._rpu_fc1_b.to(dtype=torch.float16, device="rpu"))
        fc2_b_list.append(block._rpu_fc2_b.to(dtype=torch.float16, device="rpu"))

    # ------------------------------------------------------------------ #
    # Step 3: prepare the native set_weights arguments.
    # ------------------------------------------------------------------ #
    _vision_args = (
        q_w_list, k_w_list, v_w_list, o_w_list,
        fc1_w_list, fc2_w_list,
        ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list,
        q_b_list, k_b_list, v_b_list, o_b_list,
        fc1_b_list, fc2_b_list,
        num_heads, physical_head_dim, hidden_size, physical_intermediate_size,
        float(eps),
        deepstack_visual_indexes,
    )
    # ------------------------------------------------------------------ #
    # Step 4: prepare rope tables. The keepalive view is handle-owned and is
    # retrieved only after the pending handle has been configured below.
    # ------------------------------------------------------------------ #
    freq_cos, freq_sin = build_vision_rope_tables(
        logical_head_dim, max_hw, device="rpu", dtype=torch.float16)

    # ------------------------------------------------------------------ #
    # Step 5: fold patch_embed Conv3d → Linear weight, KEEP ON CPU fp16. (F31)
    #
    # `aten::linear` on RPU expects the weight to be pre-swizzled (col- or
    # row-partition layout for the 8-core GEMM). Passing an unswizzled weight
    # silently produces garbage (each core reads the wrong slice). The fold
    # output is plain `[embed_dim=1024, cin*tp*ps*ps=1536]` row-major — not
    # the kernel's expected layout. The CPU reference path consumes this
    # ordinary row-major weight directly.
    # ------------------------------------------------------------------ #
    pe = vision_model.patch_embed
    pe_w_folded = _fold_conv3d_to_linear_weight(pe.proj.weight.data)
    pe_b = pe.proj.bias.data if pe.proj.bias is not None else None

    patch_embed_w = pe_w_folded.to(dtype=torch.float16, device="cpu").contiguous()
    patch_embed_b = (
        pe_b.to(dtype=torch.float16, device="cpu").contiguous() if pe_b is not None else None
    )

    # ------------------------------------------------------------------ #
    # Step 6: keep pos_embed + mergers on CPU. (F32 + scratch perf)
    # ------------------------------------------------------------------ #
    vision_model.pos_embed.to(device="cpu", dtype=torch.float16)
    vision_model.merger.to(device="cpu", dtype=torch.float32)
    for ds_merger in vision_model.deepstack_merger_list:
        ds_merger.to(device="cpu", dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Step 7: prepare dummy KV caches + GraphCache before creating a handle.
    # ------------------------------------------------------------------ #
    vision_kv_cache = _make_dummy_vision_kv_caches(
        num_layers, max_seq_len, num_heads, physical_head_dim,
        **({"num_cores": num_cores} if num_cores != 8 else {}),
    )
    vision_graph_cache = rpu_backend.graph.GraphCache(
        **({"max_entries": _graph_cache_max_entries}
           if _graph_cache_max_entries is not None else {}),
        **({"runtime_policy": rpu_backend.graph.GraphRuntimePolicy.from_environment(
            execution_core_count=num_cores)} if num_cores != 8 else {}))

    # FNV1a-style hash of the deepstack indexes list — used as a tiebreaker
    # in the GraphSignature so two models with different deepstack layouts
    # but same num_patches don't collide on the same BUILT entry.
    _ds_hash = 0
    for idx in deepstack_visual_indexes:
        _ds_hash = (_ds_hash * 1099511628211) ^ int(idx)
        _ds_hash &= (1 << 63) - 1
    required_attrs = (
        "_rpu_vision_handle",
        "_rpu_vision_num_cores",
        "_rpu_vision_kv_cache",
        "_rpu_vision_graph_cache",
        "_rpu_vision_max_hw",
        "_rpu_vision_has_dispatched",
        "_rpu_vision_freq_cos",
        "_rpu_vision_freq_sin",
        "_rpu_vision_position_idx_keepalive",
        "_rpu_vision_patch_embed_w",
        "_rpu_vision_host_fp32_patch",
        "_rpu_vision_batch_enabled",
        "_rpu_vision_batch_cap",
    )

    # ------------------------------------------------------------------ #
    # Step 8: configure a pending handle, publish all Python state, then
    # retire the old handle. The public wrapper owns rollback until commit.
    # ------------------------------------------------------------------ #
    handle = (torch.ops.rpu.qwen3vl_vision_create() if num_cores == 8
              else torch.ops.rpu.qwen3vl_vision_create(num_cores))
    _install_state["pending_handle"] = handle
    resource = _InstalledNativeResource(
        vision_model, handle, torch.ops.rpu.qwen3vl_vision_destroy,
        graphs=(vision_graph_cache,),
        keepalive=(_vision_args, q_s_list, k_s_list, v_s_list, o_s_list,
                   fc1_s_list, fc2_s_list, freq_cos, freq_sin,
                   vision_kv_cache, patch_embed_w, patch_embed_b),
        label="Vision", handle_name="_rpu_vision_handle")
    handle_finalizer = resource.finalizer
    _install_state["pending_resource"] = resource

    if w8a16:  # the _w8a16 op only exists on the w8a16-built extension
        torch.ops.rpu.qwen3vl_vision_set_weights_w8a16(
            handle, *_vision_args,
            q_s_list, k_s_list, v_s_list, o_s_list, fc1_s_list, fc2_s_list)
    else:
        torch.ops.rpu.qwen3vl_vision_set_weights(handle, *_vision_args)
    if num_cores != 8 and tuple(torch.ops.rpu.qwen3vl_vision_topology(handle)) != (4, 4, 4, 8):
        raise RuntimeError("four-core Vision native topology differs from the prepared weights")
    torch.ops.rpu.qwen3vl_vision_set_chunk_envelope(handle, int(vision_kv_cache.max_seq_len), 0)
    torch.ops.rpu.qwen3vl_vision_set_rope_route(
        handle, freq_cos, freq_sin,
        _qwen3vl_rope_route_request(),
    )
    torch.ops.rpu.qwen3vl_vision_set_chunk_size(
        handle,
        0 if execution_chunk_size == "auto" else int(execution_chunk_size),
    )
    position_idx_keepalive = torch.ops.rpu.qwen3vl_vision_position_idx_keepalive(handle)
    resource.keepalive = (resource.keepalive, position_idx_keepalive)

    _clear_vision_install_attrs(vision_model)
    vision_model._rpu_vision_freq_cos = freq_cos
    vision_model._rpu_vision_freq_sin = freq_sin
    vision_model._rpu_vision_position_idx_keepalive = position_idx_keepalive
    vision_model._rpu_vision_patch_embed_w = patch_embed_w
    vision_model._rpu_vision_patch_embed_b = patch_embed_b
    vision_model._rpu_vision_kv_cache = vision_kv_cache
    vision_model._rpu_vision_graph_cache = vision_graph_cache
    vision_model._rpu_vision_spatial_merge_size = spatial_merge_size
    vision_model._rpu_vision_max_hw = int(max_hw)
    vision_model._rpu_vision_has_dispatched = False
    vision_model._rpu_vision_num_layers = num_layers
    vision_model._rpu_vision_num_cores = num_cores
    vision_model._rpu_vision_hidden_size = hidden_size
    vision_model._rpu_vision_logical_head_dim = logical_head_dim
    vision_model._rpu_vision_physical_head_dim = physical_head_dim
    vision_model._rpu_vision_logical_intermediate_size = logical_intermediate_size
    vision_model._rpu_vision_physical_intermediate_size = physical_intermediate_size
    vision_model._rpu_vision_deepstack_indexes = deepstack_visual_indexes
    vision_model._rpu_vision_deepstack_hash = _ds_hash
    vision_model._rpu_vision_execution_chunk_size = execution_chunk_size
    vision_model._rpu_vision_host_fp32_patch = host_fp32_patch
    vision_model._rpu_vision_batch_enabled = vision_batch
    vision_model._rpu_vision_batch_cap = vision_batch_cap
    # The controlled 4B W8 tower admits only B2/S256 by default. Preserve
    # explicit legacy BATCH on/off and its broader opt-in grouping semantics.
    vision_model._rpu_vision_default_pair_n256 = (
        pack_w8_blocks and not _allow_runtime_quantized_large
        and "RPU_QWEN3VL_VISION_BATCH" not in os.environ)
    vision_model._rpu_vision_handle = handle
    vision_model._rpu_vision_retirement_state = resource
    vision_model._rpu_vision_handle_finalizer = handle_finalizer
    vision_model._rpu_lazy_init_checked = True
    vision_model._rpu_required_attrs = required_attrs
    vision_model.forward = types.MethodType(_rpu_vision_forward, vision_model)

    from rpu_backend.graph.lazy_init_guard import _verify_lazy_init
    _verify_lazy_init(vision_model)

    old_resource = _install_state["old_resource"]
    if old_resource is not None:
        old_resource.retire()
        if old_resource.parent is not None:
            resource.take_ownership(old_resource.parent())
    elif _install_state["old_handle"] is not None:
        raise RuntimeError("Vision replacement requires its actual retirement state")
    _install_state["committed"] = True
    old_finalizer = _install_state["old_finalizer"]
    if old_finalizer is not None and getattr(old_finalizer, "alive", False):
        old_finalizer.detach()

    _LOG.info("Patched Qwen3VLVisionModel: handle=%d, num_layers=%d, "
              "deepstack_layers=%s, hidden=%d, num_heads=%d, "
              "head_dim=%d->%d, intermediate=%d->%d, max_hw=%d",
              handle, num_layers, deepstack_visual_indexes,
              hidden_size, num_heads, logical_head_dim, physical_head_dim,
              logical_intermediate_size, physical_intermediate_size, max_hw)

    return handle


# ─────────────────────────────────────────────────────────────────────────────
# Forward replacement (instance method bound by install_qwen3_vl_vision_for_rpu)
# ─────────────────────────────────────────────────────────────────────────────

def _vision_image_groups(grid_thw_cpu, patches_per_image, *, batch_cap, cache_capacity,
                         exact_chunk, fused_merger, allow_chunked_single=False,
                         default_pair_n256=False):
    """Original forward grouping, also inspected before cost installation."""
    def _minibatch_sdpa_supported(per_image_ctx_len: int, image_batch_count: int) -> bool:
        # Keep the native tile/grid predicate: at least two whole 128-row
        # tiles per image, with <=8 blocks or a multiple-of-eight sync group.
        blocks_per_image, remainder = divmod(per_image_ctx_len, 128)
        if remainder or blocks_per_image < 2:
            return False
        grid_dim_x = blocks_per_image * image_batch_count
        return grid_dim_x <= 8 or grid_dim_x % 8 == 0

    groups = []
    gi = 0
    while gi < len(patches_per_image):
        if exact_chunk is None:
            g = 1
            group_cap = batch_cap
            if default_pair_n256:
                group_cap = (min(batch_cap, 2) if patches_per_image[gi] == 256
                             and int(grid_thw_cpu[gi, 0]) == 1 else 1)
            while (g < group_cap and gi + g < len(patches_per_image)
                   and torch.equal(grid_thw_cpu[gi + g], grid_thw_cpu[gi])
                   and (g + 1) * patches_per_image[gi] <= cache_capacity
                   and _minibatch_sdpa_supported(patches_per_image[gi], g + 1)):
                g += 1
        else:
            n_i = int(patches_per_image[gi])
            run = 1
            while (gi + run < len(patches_per_image)
                   and torch.equal(grid_thw_cpu[gi + run], grid_thw_cpu[gi])):
                run += 1
            candidates = []
            for candidate in range(1, min(run, 3) + 1):
                if run % candidate or candidate * n_i > int(cache_capacity):
                    continue
                if candidate > 1 and not _minibatch_sdpa_supported(n_i, candidate):
                    continue
                full_chunk = ((candidate * n_i + 15) // 16) * 16
                if allow_chunked_single and candidate == 1:
                    candidates.append(candidate)
                elif exact_chunk == 144:
                    # Legacy one-image split is incompatible with fused merger.
                    if candidate == 1 and not fused_merger:
                        candidates.append(candidate)
                elif full_chunk == exact_chunk:
                    candidates.append(candidate)
                elif candidate == 1 and not fused_merger and exact_chunk < full_chunk:
                    # Ordinary single-image execution may split QKV/compute
                    # while retaining the whole image as its input group.
                    # The native dry planner must still admit the exact size.
                    candidates.append(candidate)
            if not candidates:
                raise ValueError(
                    "Qwen3-VL vision cannot realize exact chunk_size="
                    f"{exact_chunk} before dispatch: image_{gi} has {n_i} "
                    f"patches, equal-grid run={run}, fused_merger="
                    f"{bool(fused_merger)}, cache_capacity={cache_capacity}"
                )
            g = max(candidates)
        groups.append((gi, g))
        gi += g
    return groups


def _rpu_vision_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
    """RPU-dispatching forward for Qwen3VLVisionModel.

    Args (mirror HF):
        hidden_states: `[seq_len, cin*tp*ps*ps]` fp16/fp32 pixel features
            (pre-flattened patches). Will be moved to CPU + cast to fp16 for
            patch_embed (F31).
        grid_thw: `[n_images, 3]` int (T, H, W) per image. CPU OR RPU; will
            be moved to CPU for position_idx + pos_emb_interpolate compute.

    Returns BaseModelOutputWithDeepstackFeatures matching HF semantics.

    Multi-image: SDPA must not cross image boundaries. By default we call the
    C++ encoder ONCE PER IMAGE and concatenate (same-shape images reuse the
    same GraphCache BUILT entry — one BUILD + N REPLAYs). The
    controlled 4B W8 tower pairs consecutive equal-grid N256 images by default.
    With the RPU_QWEN3VL_VISION_BATCH gate, consecutive EQUAL-size images are packed
    into one forward and per-image isolation is preserved by the minibatch
    SDPA kernel (image i attends only its own per_image_ctx K/V) — 1 REPLAY +
    GEMM MFU over the pack instead of g separate REPLAYs. Sig keyed on
    (op_id, packed_num_patches, hidden, depth, deepstack_hash, fused_merger
    [, image_batch_count]).
    """
    # LingBot2-private consumer mode.  Its fused merger writes pooler +
    # DeepStack rows directly into persistent prefix buffers, so the generic
    # stable output slots are deliberately not refreshed and must not be
    # popped/cloned.  The normal/public path below is unchanged.
    _prefix_scatter_consume_only = bool(
        kwargs.pop("_rpu_prefix_scatter_consume_only", False))
    # The composite text path consumes FP16 merged features. Standalone
    # Vision retains its public CPU FP32 merger outputs by default.
    _keep_merged_fp16 = kwargs.pop("_rpu_keep_merged_fp16", False)
    if type(_keep_merged_fp16) is not bool:
        raise TypeError("Qwen3-VL internal merged-output request must be a bool")
    if _keep_merged_fp16 and not getattr(self, "_rpu_vision_merger_on_device", False):
        raise RuntimeError("FP16 merged snapshots require installed RPU mergers")
    if (_prefix_scatter_consume_only
            and not getattr(
                self, "_rpu_prefix_scatter_consume_only_allowed", False)):
        raise RuntimeError(
            "Qwen3-VL prefix-scatter consume-only is an internal direct-prefix "
            "path and was not enabled for this Vision instance.")

    BaseModelOutputWithDeepstackFeatures = (
        None if _prefix_scatter_consume_only else _deepstack_output_cls())

    handle = self._rpu_vision_handle
    spatial_merge_size = self._rpu_vision_spatial_merge_size
    num_layers = self._rpu_vision_num_layers
    hidden_size = self._rpu_vision_hidden_size
    cache = self._rpu_vision_kv_cache
    graph_cache = self._rpu_vision_graph_cache
    deepstack_hash = self._rpu_vision_deepstack_hash
    num_cores = getattr(self, "_rpu_vision_num_cores", 8)
    if num_cores != 8 and any(getattr(self, name, False) for name in (
            "_rpu_patch_embed_on_device", "_rpu_vision_merger_on_device",
            "_rpu_vision_fused_merger", "_rpu_prefix_scatter_consume_only_allowed")):
        raise RuntimeError("reduced-core Vision keeps patch embedding and mergers on CPU")

    grid_thw_cpu = grid_thw.detach().cpu() if grid_thw.device.type != "cpu" else grid_thw
    grid_hw_max = int(grid_thw_cpu[:, 1:].max().item()) if grid_thw_cpu.numel() else 0
    if grid_hw_max > self._rpu_vision_max_hw:
        raise ValueError(
            f"Qwen3VL vision grid H/W max {grid_hw_max} exceeds installed max_hw "
            f"{self._rpu_vision_max_hw}; reinstall with a larger max_hw")

    # Position embeddings are prompt-geometry constants. Cache both the RPU
    # fp16 form and a CPU fp32 form used by LingBot2's host patch-embed path.
    _pe_key = getattr(self, "_rpu_vision_pos_embeds_key", None)
    if _pe_key is None or not torch.equal(grid_thw_cpu, _pe_key):
        pos_embeds_cpu = type(self).fast_pos_embed_interpolate(self, grid_thw_cpu)
        self._rpu_vision_pos_embeds = pos_embeds_cpu.to(
            device="rpu", dtype=torch.float16).contiguous()
        self._rpu_vision_pos_embeds_cpu32 = pos_embeds_cpu.float().contiguous()
        self._rpu_vision_pos_embeds_key = grid_thw_cpu.clone()

    # ----- Input prep + patch_embed (Conv3d→Linear folded) ----------------
    _host_fp32_patch = self._rpu_vision_host_fp32_patch
    _pos_folded = False
    if getattr(self, "_rpu_patch_embed_on_device", False):
        # On-device patch_embed uses the column-swizzled folded weight and
        # uploads raw patches instead of CPU-computed embeddings.
        if hidden_states.device.type == "cpu":
            source = hidden_states.to(dtype=torch.float16).contiguous()
            input_slot = getattr(self, "_rpu_vision_input_slot", None)
            if (
                input_slot is None
                or input_slot.shape != source.shape
                or input_slot.dtype != source.dtype
            ):
                input_slot = source.to(device="rpu").contiguous()
                self._rpu_vision_input_slot = input_slot
            else:
                input_slot.copy_(source)
            hs = input_slot
        else:
            hs = hidden_states.to(
                device="rpu", dtype=torch.float16
            ).contiguous()
        output_shape = (
            *hs.shape[:-1], self._rpu_patch_embed_w_rpu.shape[0]
        )
        output_slot = getattr(self, "_rpu_patch_embed_output_slot", None)
        if (
            output_slot is None
            or output_slot.shape != output_shape
            or output_slot.dtype != hs.dtype
        ):
            output_slot = torch.empty(
                output_shape, device=hs.device, dtype=hs.dtype
            )
            self._rpu_patch_embed_output_slot = output_slot
        if getattr(self, "_rpu_vision_aux_w8a16", False):
            torch.ops.rpu.linear_w8a16_into(
                hs, self._rpu_patch_embed_w_rpu, self._rpu_vision_patch_embed_scale,
                self._rpu_patch_embed_b_rpu, output_slot,
                getattr(self, "_rpu_vision_linear_acc32", False),
            )
        else:
            torch.ops.rpu.linear_into(
                hs, self._rpu_patch_embed_w_rpu,
                self._rpu_patch_embed_b_rpu, output_slot,
                getattr(self, "_rpu_vision_linear_acc32", False),
            )
        embed_packed = output_slot
    elif _host_fp32_patch:
        pe_w32 = getattr(self, "_rpu_vision_patch_embed_w32", None)
        if pe_w32 is None:
            pe_w32 = self._rpu_vision_patch_embed_w.float().contiguous()
            self._rpu_vision_patch_embed_w32 = pe_w32
            self._rpu_vision_patch_embed_b32 = (
                self._rpu_vision_patch_embed_b.float().contiguous()
                if self._rpu_vision_patch_embed_b is not None else None
            )
        hs32 = hidden_states if hidden_states.dtype == torch.float32 else hidden_states.float()
        if hs32.device.type != "cpu":
            hs32 = hs32.cpu()
        embed32 = torch.nn.functional.linear(
            hs32, pe_w32, self._rpu_vision_patch_embed_b32)
        embed32 = embed32 + self._rpu_vision_pos_embeds_cpu32
        embed_packed = embed32.to(
            device="rpu", dtype=torch.float16).contiguous()
        _pos_folded = True
    else:
        if hidden_states.device.type != "cpu":
            hidden_states_cpu = hidden_states.to(device="cpu", dtype=torch.float16).contiguous()
        elif hidden_states.dtype != torch.float16:
            hidden_states_cpu = hidden_states.to(dtype=torch.float16).contiguous()
        else:
            hidden_states_cpu = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
        pe_w = self._rpu_vision_patch_embed_w  # CPU fp16
        pe_b = self._rpu_vision_patch_embed_b  # CPU fp16
        embed_cpu = torch.nn.functional.linear(hidden_states_cpu, pe_w, pe_b)  # CPU fp16 [N_total, hidden]
        embed_packed = embed_cpu.to(device="rpu", dtype=torch.float16, non_blocking=False).contiguous()

    if not _pos_folded:
        embed_packed.add_(self._rpu_vision_pos_embeds)

    # ----- Per-image encoder loop -----------------------------------------
    keepalive = self._rpu_vision_position_idx_keepalive
    k_caches = [cache.k_caches[i] for i in range(num_layers)]
    v_caches = [cache.v_caches[i] for i in range(num_layers)]

    patches_per_image = grid_thw_cpu.prod(-1).tolist()  # [n_0, n_1, ...]
    total_patches = sum(patches_per_image)
    if embed_packed.size(0) != total_patches:
        raise RuntimeError(
            f"Qwen3VL vision forward: patch_embed produced {embed_packed.size(0)} tokens "
            f"but grid_thw implies {total_patches}"
        )

    n_deepstack = len(self._rpu_vision_deepstack_indexes)
    # Fused patch merger (ROUND-3 opt #1, gr00t): the C++ post_fn returns the merged pooler
    # [n_i/sm², out_hidden] directly from qwen3vl_vision_forward — accumulate that instead of
    # the raw encoder hidden, and skip the Python patch merger below. Deepstack stays eager.
    _fused_merger = getattr(self, "_rpu_vision_fused_merger", False)
    _fp16_chunked_merger = getattr(self, "_rpu_vision_chunked_merger_fp16", False)
    _requested_chunk = getattr(
        self, "_rpu_vision_execution_chunk_size", "auto"
    )
    if _fp16_chunked_merger and (
            any(n != 1200 for n in patches_per_image)
            or _requested_chunk not in ("auto", 608)):
        # Preserve the original whole-request eager math for other grids,
        # including mixed requests. The cold native capability is N1200/B1.
        _fused_merger = False
    _adaptive_merger = _fused_merger and (
        _fp16_chunked_merger or getattr(self, "_rpu_vision_fused_merger_w8a16", False))
    _exact_chunk = (
        None if _requested_chunk == "auto" else int(_requested_chunk)
    )
    if _prefix_scatter_consume_only and not _fused_merger:
        raise RuntimeError(
            "Qwen3-VL prefix-scatter consume-only requires the fused merger.")
    last_hidden_per_image: list[torch.Tensor] = []
    merged_per_image: list[torch.Tensor] = []
    deepstack_snapshots_per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_deepstack)]
    # inc-2: per-image in-graph merged deepstack features (when _fused_merger).
    deepstack_merged_per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_deepstack)]

    # Multi-image BATCH grouping. Gate RPU_QWEN3VL_VISION_BATCH packs consecutive
    # EQUAL-size images into ONE forward (per-image minibatch SDPA → 1 BUILD/REPLAY
    # + GEMM MFU over g·n_i rows, vs g separate REPLAYs). The controlled 4B W8
    # default additionally pairs N256 only; all other defaults keep size 1. The cap bounds the
    # packed seq for SPM headroom + the minibatch grid constraint (override via
    # RPU_QWEN3VL_VISION_BATCH_CAP, default 3). Groups are also capped by the
    # allocated KV-cache capacity. Different-size neighbours split the group, so
    # mixed-resolution inputs fall back to g=1 (current path).
    _batch_on = self._rpu_vision_batch_enabled
    _default_pair_n256 = (_adaptive_merger and _exact_chunk is None and
                         getattr(self, "_rpu_vision_default_pair_n256", False))
    _batch_cap = self._rpu_vision_batch_cap if (_batch_on or _default_pair_n256) else 1
    if _fp16_chunked_merger and _fused_merger:
        _batch_cap = 1

    groups = _vision_image_groups(grid_thw_cpu, patches_per_image,
        batch_cap=_batch_cap, cache_capacity=cache.max_seq_len,
        exact_chunk=_exact_chunk, fused_merger=_fused_merger and not _adaptive_merger,
        allow_chunked_single=_adaptive_merger, default_pair_n256=_default_pair_n256)
    if _prefix_scatter_consume_only and len(groups) != 1:
        raise RuntimeError(
            "Qwen3-VL prefix-scatter consume-only requires exactly one packed "
            f"Vision group; planner produced {groups}.")

    # Snapshot each stable native slot directly into its final request-owned
    # rows. Ordinary N1200 multi-image callers otherwise clone every group and
    # then copy those snapshots again through four cats.
    _pack_merged = (
        _adaptive_merger and _keep_merged_fp16 and not _prefix_scatter_consume_only
        and n_deepstack == 3 and len(groups) > 1
        and (_fp16_chunked_merger
             or getattr(self, "_rpu_vision_chunked_merger_w8a16", False))
        and all(g == 1 and patches_per_image[i] == 1200 for i, g in groups)
    )
    packed_merged = None
    packed_raw = None
    merged_rows_written = 0

    _batch_signature = _batch_on or any(g > 1 for _, g in groups)
    _resolved_chunks: list[int] = []
    _vision_plans = []
    _dispatched_descriptors: list[tuple[int, ...]] = []
    _parent_graph_key_words = _execution_graph_key_words(self)

    offset = 0
    for first_idx, g in groups:
        n_i = patches_per_image[first_idx]   # patches/image (equal across the group)
        g_patches = g * n_i
        if g_patches > cache.max_seq_len:
            raise ValueError(
                f"Qwen3VL vision group@{first_idx} packs {g} image(s) into "
                f"{g_patches} patches, exceeding KV-cache capacity "
                f"{cache.max_seq_len}; increase max_seq_len or reduce image size")
        vision_plan = _plan_qwen3_vl_vision_execution(
            handle, g_patches, g, _exact_chunk, execution_owner=self,
            graph_cache=None if getattr(self, "_rpu_vision_graph_disable", False) else graph_cache,
            plan_signature=(tuple(int(v) for v in grid_thw_cpu[first_idx].tolist()), int(spatial_merge_size), bool(_fused_merger), bool(getattr(self, "_rpu_vision_rope_disable", False)), *((num_cores,) if num_cores != 8 else ())),
        )
        selected = vision_plan.selected
        if selected is None:
            raise RuntimeError("Qwen3-VL vision dry planner selected no plan")
        # Native COMPLETE manifests cover each merger invocation, including
        # the post-encoder chunks of an ordinary W8 group.
        _group_fused_merger = _fused_merger and (
            not _adaptive_merger or selected.stage_tuple.compute_chunk >= g_patches
            or _fp16_chunked_merger
            or getattr(self, "_rpu_vision_chunked_merger_w8a16", False))
        planned_stage_descriptor = list(
            selected.stage_tuple.physical_descriptor
        )
        _vision_plans.append(vision_plan)
        embed_group = embed_packed.narrow(0, offset, g_patches).contiguous()

        # position_idx is identical for every equal-size image → compute once, tile g×
        # (the minibatch kernel reads per-row positions, so each image-block in the pack
        # gets its own image-local (row,col) indices).
        _pi_key = (
            int(grid_thw_cpu[first_idx][0]),
            int(grid_thw_cpu[first_idx][1]),
            int(grid_thw_cpu[first_idx][2]),
            g,
            spatial_merge_size,
        )
        if getattr(self, "_rpu_vision_pos_idx_key", None) != _pi_key:
            pos_idx_cpu = _compute_vision_position_idx_cpu(
                grid_thw_cpu[first_idx : first_idx + 1], spatial_merge_size)
            if pos_idx_cpu.size(0) != n_i:
                raise RuntimeError(
                    f"position_idx rows {pos_idx_cpu.size(0)} != "
                    f"image_{first_idx} patches {n_i}"
                )
            if g > 1:
                pos_idx_cpu = pos_idx_cpu.repeat(g, 1)
            self._rpu_vision_pos_idx_val = pos_idx_cpu
            self._rpu_vision_pos_idx_key = _pi_key
        pos_idx_cpu = self._rpu_vision_pos_idx_val

        _rope_disabled = bool(getattr(self, "_rpu_vision_rope_disable", False))
        _ka_key = (_pi_key, g_patches, _rope_disabled)
        if getattr(self, "_rpu_vision_keepalive_key", None) != _ka_key:
            if _rope_disabled:
                keepalive.narrow(0, 0, g_patches).zero_()
            else:
                keepalive.narrow(0, 0, g_patches).copy_(
                    pos_idx_cpu.to(keepalive.device))
            self._rpu_vision_keepalive_key = _ka_key

        cache.reset_to_position(0)

        embed_group_3d = embed_group.unsqueeze(0).contiguous()

        self._rpu_vision_has_dispatched = True

        # F8: Wrap in Graph.capture(sig). The signature is keyed on
        # (op_id, num_patches, hidden_size, num_layers, deepstack_hash,
        # fused_merger [, g]) so
        # each unique (packed) num_patches produces one BUILD + N REPLAYs across
        # same-shape groups. `g` (image_batch_count) is appended only when batching
        # is engaged so the batch-off / qwen3_vl path keeps its exact legacy sig.
        # Skip the wrap when caller opts out (`_rpu_vision_graph_disable=True`) —
        # used for numerical-isolation debug (PASSTHROUGH vs RECORDING/REPLAYING).
        if getattr(self, "_rpu_vision_graph_disable", False):
            out_3d = torch.ops.rpu.qwen3vl_vision_forward(
                handle, embed_group_3d, k_caches, v_caches, g_patches, g,
                planned_stage_descriptor,
            )
        else:
            _dyn = (
                [num_layers, deepstack_hash, int(_group_fused_merger)]
                + ([g] if _batch_signature else [])
                + list(vision_plan.graph_key_words())
                + list(_parent_graph_key_words)
                + ([num_cores, num_cores, num_cores, 8] if num_cores != 8 else [])
            )
            sig = rpu_backend.graph.GraphSignature(
                op_id="qwen3vl_vision",
                shapes=[g_patches, hidden_size],
                dyn_dims=_dyn,
                dtypes=[torch.float16],
            )
            with graph_cache.capture(sig):
                out_3d = torch.ops.rpu.qwen3vl_vision_forward(
                    handle, embed_group_3d, k_caches, v_caches, g_patches, g,
                    planned_stage_descriptor,
                )

        resolved_chunk = int(
            torch.ops.rpu.qwen3vl_vision_get_resolved_chunk_size(handle)
        )
        _resolved_chunks.append(resolved_chunk)
        if resolved_chunk != selected.stage_tuple.compute_chunk:
            raise RuntimeError(
                "Qwen3-VL vision dry/forward chunk plan drift: dry="
                f"{selected.stage_tuple.compute_chunk}, forward="
                f"{resolved_chunk}, group@{first_idx} has images={g}, "
                f"patches={g_patches}"
            )
        if _exact_chunk is not None and resolved_chunk != _exact_chunk:
            raise RuntimeError(
                "Qwen3-VL vision exact chunk drift: requested="
                f"{_exact_chunk}, resolved={resolved_chunk}, group@{first_idx} "
                f"has images={g}, patches={g_patches}"
            )
        _dispatched_descriptors.append(tuple(planned_stage_descriptor))

        if _adaptive_merger:
            # Public ordinary Vision continues returning raw [N,H] hidden,
            # including when native post_fn additionally produces pooler/DS.
            if _pack_merged:
                # capture has finished: copy before the next group reuses the
                # native slot. Fresh request storage also avoids the final cat.
                if (tuple(out_3d.shape) != (1, g_patches, hidden_size)
                        or out_3d.dtype != torch.float16
                        or out_3d.device != embed_group.device
                        or offset + g_patches > total_patches):
                    raise RuntimeError(
                        f"group@{first_idx}: invalid FP16 raw snapshot shape/dtype/device")
                if packed_raw is None:
                    packed_raw = out_3d.new_empty((total_patches, hidden_size))
                packed_raw.narrow(0, offset, g_patches).copy_(out_3d.squeeze(0))
            else:
                last_hidden_per_image.append(out_3d.squeeze(0).clone())
        if _group_fused_merger:
            if not _prefix_scatter_consume_only:
                # ONE pop returns ALL in-graph merged outputs from their STABLE slots (fixed DMA →
                # bit-identical): [pooler_merged, ds_merged_0, ...], each [g_patches/sm², out_hidden]
                # and g image-blocks in order (out_3d is ignored encoder output, not the pooler).
                # Snapshot before the next group overwrites these stable slots.
                # A CPU view alone does not establish independent ownership.
                merged_all = torch.ops.rpu.qwen3vl_vision_pop_merged(handle)
                expected_merged = 1 + n_deepstack
                if len(merged_all) != expected_merged:
                    raise RuntimeError(
                        f"group@{first_idx}: merged count {len(merged_all)} != "
                        f"{expected_merged}"
                    )
                if _pack_merged:
                    rows = g_patches // (spatial_merge_size ** 2)
                    total_rows = total_patches // (spatial_merge_size ** 2)
                    sources = [value.cpu() for value in merged_all]
                    width = sources[0].size(1) if sources[0].ndim == 2 else 0
                    if (width <= 0 or any(
                            value.dtype != torch.float16 or value.ndim != 2
                            or tuple(value.shape) != (rows, width)
                            for value in sources)
                            or merged_rows_written + rows > total_rows
                            or (packed_merged is not None
                                and packed_merged[0].size(1) != width)):
                        raise RuntimeError(
                            f"group@{first_idx}: invalid FP16 merged snapshot shape/dtype")
                    if packed_merged is None:
                        packed_merged = [torch.empty(
                            (total_rows, width), dtype=torch.float16, device="cpu")
                            for _ in sources]
                    for destination, source in zip(packed_merged, sources):
                        destination.narrow(0, merged_rows_written, rows).copy_(source)
                    merged_rows_written += rows
                else:
                    merged_per_image.append(
                        (merged_all[0].cpu().clone() if _keep_merged_fp16
                         else merged_all[0].cpu().float()) if _adaptive_merger
                        else merged_all[0].clone())
                    for layer_k in range(n_deepstack):
                        deepstack_merged_per_layer[layer_k].append(
                            (merged_all[1 + layer_k].cpu().clone() if _keep_merged_fp16
                             else merged_all[1 + layer_k].cpu().float()) if _adaptive_merger
                            else merged_all[1 + layer_k].clone())
        else:
            if not _adaptive_merger:
                last_hidden_per_image.append(out_3d.squeeze(0).clone())  # [g_patches, hidden] (clone: P4)
            snapshots_i = torch.ops.rpu.qwen3vl_vision_pop_deepstack_snapshots(handle)
            if len(snapshots_i) != n_deepstack:
                raise RuntimeError(
                    f"group@{first_idx}: deepstack snapshot count "
                    f"{len(snapshots_i)} != {n_deepstack}"
                )
            if _adaptive_merger:
                merged_per_image.append(_merger_forward_on_device(
                    self.merger, last_hidden_per_image[-1].detach().cpu().float(),
                    keep_fp16=_keep_merged_fp16,
                    linear_acc32=getattr(self, "_rpu_vision_linear_acc32", False)))
                for layer_k, snap in enumerate(snapshots_i):
                    deepstack_merged_per_layer[layer_k].append(_merger_forward_on_device(
                        self.deepstack_merger_list[layer_k], snap.detach().cpu().float(),
                        keep_fp16=_keep_merged_fp16,
                    linear_acc32=getattr(self, "_rpu_vision_linear_acc32", False)))
            else:
                for layer_k, snap in enumerate(snapshots_i):
                    deepstack_snapshots_per_layer[layer_k].append(snap.detach().cpu().float())

        offset += g_patches

    if _prefix_scatter_consume_only:
        torch.ops.rpu.spm_alloc_reset_temporary()
        vars(self)["_rpu_last_execution_plan"] = _vision_execution_receipt(
            self, total_patches, _vision_plans, _resolved_chunks,
            _dispatched_descriptors,
        )
        return None

    # ----- Concatenate per-image outputs ----------------------------------
    # Merger + deepstack mergers: CPU fp32 by default; on RPU when opted in
    # (enable_qwen3_vl_vision_merger_on_device) — the GEMMs are the dominant per-image CPU
    # cost (e.g. GR00T VLA). When the fused in-graph merger is active the C++ op already
    # returned the merged pooler; we just concat. norm stays CPU fp32 in the eager path.
    merger_on_device = getattr(self, "_rpu_vision_merger_on_device", False)
    if _pack_merged and (packed_merged is None or merged_rows_written
                        != total_patches // (spatial_merge_size ** 2)):
        raise RuntimeError("Qwen3-VL merged snapshots did not cover all image rows")
    if _pack_merged and (packed_raw is None or offset != total_patches):
        raise RuntimeError("Qwen3-VL raw snapshots did not cover all image rows")
    if _fused_merger:
        # Single group (the common gr00t 双图/三图 batched case) → the clone IS the full
        # [N_total/sm², out_hidden] already; skip the degenerate 1-element torch.cat (which
        # would materialize a redundant copy). >1 group (mixed/over-cap) → concat in order.
        merged = (packed_merged[0] if _pack_merged else
                  merged_per_image[0] if len(merged_per_image) == 1
                  else torch.cat(merged_per_image, dim=0))     # [N_total/sm², out_hidden] (in-graph)
        # Raw snapshots own their storage before another group reuses the slot.
        last_hidden = (packed_raw if _pack_merged else
                       (last_hidden_per_image[0] if len(last_hidden_per_image) == 1
                        else torch.cat(last_hidden_per_image, dim=0)) if _adaptive_merger
                       else merged)  # Legacy composite consumers ignore raw hidden.
    else:
        last_hidden = torch.cat(last_hidden_per_image, dim=0)  # [N_total, hidden] RPU
        last_hidden_cpu = last_hidden.detach().cpu().float()
        if merger_on_device:
            merged = _merger_forward_on_device(
                self.merger, last_hidden_cpu, keep_fp16=_keep_merged_fp16,
                    linear_acc32=getattr(self, "_rpu_vision_linear_acc32", False))
        else:
            merged = self.merger(last_hidden_cpu)              # [N_total/sm², out_hidden]

    deepstack_features: list[torch.Tensor] = []
    if _pack_merged:
        deepstack_features = packed_merged[1:]
    elif _fused_merger:
        # inc-2: deepstack mergers ran in-graph — concat per-group merged features (single
        # group → use the clone directly, skip the redundant 1-element cat).
        for layer_k in range(n_deepstack):
            lst = deepstack_merged_per_layer[layer_k]
            deepstack_features.append(lst[0] if len(lst) == 1 else torch.cat(lst, dim=0))
    else:
        for layer_k, ds_merger in enumerate(self.deepstack_merger_list):
            snap_packed_cpu = torch.cat(deepstack_snapshots_per_layer[layer_k], dim=0)
            if merger_on_device:
                deepstack_features.append(_merger_forward_on_device(
                    ds_merger, snap_packed_cpu, keep_fp16=_keep_merged_fp16,
                    linear_acc32=getattr(self, "_rpu_vision_linear_acc32", False)))
            else:
                deepstack_features.append(ds_merger(snap_packed_cpu))

    torch.ops.rpu.spm_alloc_reset_temporary()

    vars(self)["_rpu_last_execution_plan"] = _vision_execution_receipt(
        self, total_patches, _vision_plans, _resolved_chunks,
        _dispatched_descriptors,
    )

    return BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=last_hidden,
        pooler_output=merged,
        deepstack_features=deepstack_features,
    )
