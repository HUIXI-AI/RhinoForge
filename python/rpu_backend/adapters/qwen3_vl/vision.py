"""Qwen3-VL vision encoder for rpu_backend.

Wires `Qwen3VLVisionModel` into the C++ vision subsystem registered as
`torch.ops.rpu.qwen3vl_vision_*`. Mirrors SigLIP's `patch_siglip_model_for_rpu`
pattern: handle lifecycle, weight conversion, forward replacement.

Public surface:
  - install_qwen3_vl_vision_for_rpu(vision_model)
  - build_vision_rope_tables(...)              # FreqCos / FreqSin static tables
  - The forward replacement returns
    BaseModelOutputWithDeepstackFeatures(last_hidden_state, pooler_output,
                                          deepstack_features) matching the HF
    reference.

Execution design:
  - Patch embed runs OUTSIDE the fused graph on **CPU fp16**. `aten::linear`
    on RPU expects the weight to be pre-swizzled (col/row partition); the
    folded Conv3d weight is plain `[embed_dim, cin*tp*ps*ps]` row-major and
    produces invalid output if run on RPU without swizzle.
  - `fast_pos_embed_interpolate` runs on **CPU fp16**. The HF
    `nn.Embedding` + `view` + `permute(0,1,3,2,4,5)` + `flatten(0,4)` path
    is not layout-compatible with the RPU path.
  - Merger + DeepStack mergers run on **CPU fp32** by default, with an explicit
    RPU opt-in.
  - 2D RoPE: position_idx built per forward, copy_in to model-owned
    keepalive `[MAX_KEEPALIVE_SEQ, 2] int16`.

Graph lifecycle:
  - Each per-image vision call uses `cache.capture(sig)`; each unique
    `num_patches` has one retained graph entry.
  - Dynamo safety uses eager RPUCache initialization, `_rpu_lazy_init_checked`,
    `_rpu_required_attrs`, and `_verify_lazy_init(...)`.
"""
from __future__ import annotations

import os
import types
import weakref
from typing import Any

import torch
import torch.nn as nn

import rpu_backend
from rpu_backend.runtime import rpu_env_bool
from rpu_backend.runtime.log import _LOG
from rpu_backend.runtime.weights import (
    tp_col_swizzle_mc_weight,
    tp_row_swizzle_mc_weight,
)
from rpu_backend.api.cache import RPUCache


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
_SUPPORTED_VISION_GEOMETRY = (
    1024, 4096, 24, 16, 16, 2, 2, (5, 11, 17), "gelu_pytorch_tanh",
)
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
_QWEN3_VL_32B_PHYSICAL_HEAD_DIM = 80
_QWEN3_VL_32B_PHYSICAL_INTERMEDIATE_SIZE = 4352


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


def _convert_vision_block_weights_for_rpu(
    block,
    num_heads: int,
    hidden_size: int,
    w8a16: bool = False,
    *,
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
    padded_32b = physical_geometry != (
        logical_head_dim,
        logical_intermediate_size,
    )
    if padded_32b and (
        w8a16
        or (
            hidden_size,
            logical_intermediate_size,
            num_heads,
            logical_head_dim,
            physical_head_dim,
            physical_intermediate_size,
        ) != (1152, 4304, 16, 72, 80, 4352)
    ):
        raise ValueError(
            "qwen3_vl vision physical padding is private to the exact 32B "
            "FP16 vision tower (head 72->80, MLP 4304->4352)."
        )

    # The marker records the quant MODE: re-installing the same block in the same mode is a
    # no-op, but switching mode (fp16 <-> w8a16) after the weights are already swizzled would
    # double-swizzle / mismatch dtype — fail loud rather than silently corrupt.
    if _prev is not None:
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

    if w8a16:
        from rpu_backend.quant._common import quantize_linear_per_channel

    def _swz(w_unswz, swz_fn):       # -> (weight [int8 dwidth=1 if w8a16 else fp16], fp16 scale or None)
        if w8a16:
            wq, ws = quantize_linear_per_channel(w_unswz.half())
            return swz_fn(wq, 8, dwidth=1).contiguous(), ws.to(torch.float16)
        # The public installer accepts a CPU model and owns the dtype conversion.
        # LingBot-VLA-V2 checkpoints are fp32, while the default swizzler's
        # dwidth=2 contract requires fp16 input before it rearranges bytes.
        return swz_fn(w_unswz.half()), None

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

        if padded_32b:
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
    block._rpu_qwen3vl_vision_weights_converted = bool(w8a16)


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

def _convert_vision_merger_for_rpu(merger, cores: int = 8) -> None:
    """In-place: stash RPU-resident, col-swizzled fp16 weights for a
    `Qwen3VLVisionPatchMerger`'s two GEMMs (`linear_fc1` → GELU → `linear_fc2`).

    The LayerNorm (`merger.norm`) is left on CPU fp32 to preserve its numerical
    behavior. Both GEMM shapes (`[merge_hidden, merge_hidden]`,
    `[out_hidden, merge_hidden]`) satisfy can_row && can_col so `rpu_linear`
    picks the col partition — hence `tp_col_swizzle_mc_weight`. Idempotent.
    """
    if getattr(merger, "_rpu_merger_on_device", False):
        return
    with torch.no_grad():
        merger._rpu_merge_hidden = int(merger.linear_fc1.in_features)
        merger._rpu_fc1_w = tp_col_swizzle_mc_weight(
            merger.linear_fc1.weight.data.half(), cores).to(device="rpu").contiguous()
        merger._rpu_fc1_b = merger.linear_fc1.bias.data.half().to(device="rpu").contiguous()
        merger._rpu_fc2_w = tp_col_swizzle_mc_weight(
            merger.linear_fc2.weight.data.half(), cores).to(device="rpu").contiguous()
        merger._rpu_fc2_b = merger.linear_fc2.bias.data.half().to(device="rpu").contiguous()
    merger._rpu_merger_on_device = True


def _merger_forward_on_device(merger, x_cpu_f32: torch.Tensor) -> torch.Tensor:
    """Mirror `Qwen3VLVisionPatchMerger.forward` with its GEMMs on RPU.

    `merger.norm` runs on CPU fp32 (as installed); the post-norm `[N/sm², merge_hidden]`
    activation hops to RPU fp16 for both Linear calls; eager exact-ERF GELU follows the
    registered CPU fallback, then the result returns to CPU fp32 (the existing merger
    output contract). Requires `_convert_vision_merger_for_rpu`.
    """
    x = x_cpu_f32.view(-1, merger._rpu_merge_hidden) if merger.use_postshuffle_norm else x_cpu_f32
    xn = merger.norm(x).view(-1, merger._rpu_merge_hidden)          # CPU fp32 [N/sm², merge_hidden]
    xr = xn.to(device="rpu", dtype=torch.float16).contiguous()
    xr = torch.nn.functional.linear(xr, merger._rpu_fc1_w, merger._rpu_fc1_b)  # rpu_linear (col)
    xr = torch.nn.functional.gelu(xr)              # graph-aware CPU exact-erf fallback
    xr = torch.nn.functional.linear(xr, merger._rpu_fc2_w, merger._rpu_fc2_b)
    return xr.float().cpu()


def enable_qwen3_vl_vision_merger_on_device(vision_model, cores: int = 8) -> None:
    """Move patch and DeepStack merger GEMMs from CPU fp32 to RPU fp16.

    Call after ``install_qwen3_vl_vision_for_rpu``. Idempotent; sets
    ``_rpu_vision_merger_on_device=True`` for the forward replacement.
    """
    _convert_vision_merger_for_rpu(vision_model.merger, cores)
    for ds_merger in vision_model.deepstack_merger_list:
        _convert_vision_merger_for_rpu(ds_merger, cores)
    # Also move the patch_embed Linear onto RPU.
    # the folded Conv3d weight is plain row-major [embed_dim, cin*tp*ps*ps] and MUST be col-swizzled
    # for rpu_linear, else it silently produces garbage.
    if not getattr(vision_model, "_rpu_patch_embed_on_device", False):
        pe_w = vision_model._rpu_vision_patch_embed_w
        pe_b = vision_model._rpu_vision_patch_embed_b
        vision_model._rpu_patch_embed_w_rpu = tp_col_swizzle_mc_weight(
            pe_w.half(), cores).to(device="rpu").contiguous()
        vision_model._rpu_patch_embed_b_rpu = (
            pe_b.half().to(device="rpu").contiguous() if pe_b is not None else None)
        vision_model._rpu_patch_embed_on_device = True
    vision_model._rpu_vision_merger_on_device = True


def register_qwen3vl_vision_fused_merger(vision_model, cores: int = 8) -> None:
    """Register patch-merger weights for the in-graph fused merger.

    Call AFTER `install_qwen3_vl_vision_for_rpu` and BEFORE the first forward. m0
    (linear_fc1) is col-swizzled, m2
    (linear_fc2) is row-swizzled, ln_q (LayerNorm γ/β) raw [HID]. The C++ post_fn — gated on
    `RPU_QWEN3VL_VISION_FUSED_MERGER` (set this env before the first forward/BUILD) — then runs
    the patch merger and the N deepstack mergers inside the vision graph;
    `qwen3vl_vision_forward` returns the merged pooler `[seq/sm², out_hidden]` and the merged
    deepstack features are popped via `qwen3vl_vision_pop_deepstack_merged`. The forward
    replacement reads `_rpu_vision_fused_merger` to consume both. Idempotent.
    """
    if getattr(vision_model, "_rpu_vision_fused_merger", False):
        return

    if getattr(vision_model, "_rpu_vision_has_dispatched", False):
        raise RuntimeError(
            "register_qwen3vl_vision_fused_merger must be called before the first "
            "vision forward because it changes the persistent SPM layout")

    def _merger_rpu_weights(mg):
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
        torch.ops.rpu.qwen3vl_vision_set_merger_weights(
            vision_model._rpu_vision_handle, pw[0], pw[1], pw[2], pw[3], pw[4], pw[5],
            int(m.linear_fc2.out_features), float(m.norm.eps))
        # The N deepstack mergers apply the same structure to their layer snapshots.
        dsw = [_merger_rpu_weights(dm) for dm in vision_model.deepstack_merger_list]
        if dsw:
            torch.ops.rpu.qwen3vl_vision_set_deepstack_merger_weights(
                vision_model._rpu_vision_handle,
                [d[0] for d in dsw], [d[1] for d in dsw], [d[2] for d in dsw],
                [d[3] for d in dsw], [d[4] for d in dsw], [d[5] for d in dsw])
    vision_model._rpu_vision_fused_merger = True
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

def _qwen3vl_vision_destroy_handle(h: int) -> None:
    try:
        torch.ops.rpu.qwen3vl_vision_destroy(h)
    except Exception:
        pass


_VISION_INSTALL_EXACT_ATTRS = frozenset({
    "_rpu_lazy_init_checked",
    "_rpu_required_attrs",
    "_rpu_patch_embed_on_device",
    "_rpu_patch_embed_w_rpu",
    "_rpu_patch_embed_b_rpu",
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
    vision_config: Any | None = None,
    max_hw: int = 48,
    max_seq_len: int = 2048,
    w8a16: bool = False,
    execution_chunk_size: str | int = "auto",
    _allow_graph_blocked_32b: bool = False,
) -> int:
    """Install or replace the Qwen3-VL vision runtime transactionally.

    A replacement is published only after its Python state and native handle
    are fully configured. Any failure destroys the pending handle immediately
    and restores the previously published install.
    """
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
    cfg = vision_config or vision_model.config
    is_32b_vision = _is_qwen3_vl_32b_vision_config(cfg)
    try:
        geometry = (
            int(cfg.hidden_size),
            int(cfg.intermediate_size),
            int(cfg.depth),
            int(cfg.num_heads),
            int(cfg.patch_size),
            int(cfg.temporal_patch_size),
            int(cfg.spatial_merge_size),
            tuple(int(x) for x in cfg.deepstack_visual_indexes),
            cfg.hidden_act,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "install_qwen3_vl_vision_for_rpu: incomplete vision config."
        ) from exc
    if not is_32b_vision and geometry != _SUPPORTED_VISION_GEOMETRY:
        raise ValueError(
            "install_qwen3_vl_vision_for_rpu: unsupported vision profile."
        )
    if is_32b_vision and not _allow_graph_blocked_32b:
        raise ValueError(
            "install_qwen3_vl_vision_for_rpu: the Qwen3-VL-32B vision tower "
            "may only be installed by the gated top-level adapter."
        )
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
    # The top-level Qwen3-VL profile checks this too, but GR00T and LingBot
    # call the shared installer directly.  Fail before irreversible swizzling
    # because the native Vision block implements tanh-approximate GELU.
    if getattr(cfg, "hidden_act", None) != "gelu_pytorch_tanh":
        raise ValueError(
            "install_qwen3_vl_vision_for_rpu: hidden_act must be "
            f"'gelu_pytorch_tanh', got {getattr(cfg, 'hidden_act', None)!r}."
        )
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
    }
    try:
        return _install_qwen3_vl_vision_for_rpu_impl(
            vision_model,
            vision_config=vision_config,
            max_hw=max_hw,
            max_seq_len=max_seq_len,
            w8a16=w8a16,
            execution_chunk_size=execution_chunk_size,
            _allow_graph_blocked_32b=_allow_graph_blocked_32b,
            _install_state=install_state,
        )
    except BaseException:
        if not install_state.get("committed", False):
            pending_finalizer = install_state.get("pending_finalizer")
            pending_handle = install_state.get("pending_handle")
            if pending_finalizer is not None:
                if pending_finalizer.alive:
                    pending_finalizer()
            elif pending_handle is not None:
                _qwen3vl_vision_destroy_handle(pending_handle)

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
        attn_tp=min(8, num_heads),
    )


def _install_qwen3_vl_vision_for_rpu_impl(
    vision_model,
    *,
    vision_config: Any | None = None,
    max_hw: int = 48,
    max_seq_len: int = 2048,
    w8a16: bool = False,
    execution_chunk_size: str | int = "auto",
    _allow_graph_blocked_32b: bool = False,
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
    physical_head_dim = (
        _QWEN3_VL_32B_PHYSICAL_HEAD_DIM
        if _allow_graph_blocked_32b else logical_head_dim
    )
    physical_intermediate_size = (
        _QWEN3_VL_32B_PHYSICAL_INTERMEDIATE_SIZE
        if _allow_graph_blocked_32b else logical_intermediate_size
    )
    spatial_merge_size = cfg.spatial_merge_size
    eps = 1e-6  # Qwen3VLVisionBlock LayerNorm eps (hardcoded in HF)
    deepstack_visual_indexes = list(cfg.deepstack_visual_indexes)

    # ------------------------------------------------------------------ #
    # Step 1: prepare encoder weights while the old install remains published.
    # ------------------------------------------------------------------ #
    for block in vision_model.blocks:
        _convert_vision_block_weights_for_rpu(
            block,
            num_heads,
            hidden_size,
            w8a16=w8a16,
            physical_head_dim=physical_head_dim,
            physical_intermediate_size=physical_intermediate_size,
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

    def _w_rpu(w):  # int8 weight -> keep dtype; fp16 weight -> cast (guards non-half models)
        return w.to(device="rpu") if w8a16 else w.to(dtype=torch.float16, device="rpu")

    for block in vision_model.blocks:
        q_w_list.append(_w_rpu(block._rpu_q_w))
        k_w_list.append(_w_rpu(block._rpu_k_w))
        v_w_list.append(_w_rpu(block._rpu_v_w))
        o_w_list.append(_w_rpu(block.attn.proj.weight.data))

        fc1_w_list.append(_w_rpu(block.mlp.linear_fc1.weight.data))
        fc2_w_list.append(_w_rpu(block.mlp.linear_fc2.weight.data))

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

    # Step 5: fold patch_embed Conv3d to a row-major CPU-fp16 Linear weight.
    # The optional RPU path must convert it to the required partition layout.
    pe = vision_model.patch_embed
    pe_w_folded = _fold_conv3d_to_linear_weight(pe.proj.weight.data)
    pe_b = pe.proj.bias.data if pe.proj.bias is not None else None

    patch_embed_w = pe_w_folded.to(dtype=torch.float16, device="cpu").contiguous()
    patch_embed_b = (
        pe_b.to(dtype=torch.float16, device="cpu").contiguous() if pe_b is not None else None
    )

    # ------------------------------------------------------------------ #
    # Step 6: keep pos_embed + mergers on CPU.
    # ------------------------------------------------------------------ #
    vision_model.pos_embed.to(device="cpu", dtype=torch.float16)
    vision_model.merger.to(device="cpu", dtype=torch.float32)
    for ds_merger in vision_model.deepstack_merger_list:
        ds_merger.to(device="cpu", dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Step 7: prepare dummy KV caches + GraphCache before creating a handle.
    # ------------------------------------------------------------------ #
    vision_kv_cache = _make_dummy_vision_kv_caches(
        num_layers, max_seq_len, num_heads, physical_head_dim
    )
    vision_graph_cache = rpu_backend.graph.GraphCache()

    # FNV1a-style hash of the deepstack indexes list — used as a tiebreaker
    # in the GraphSignature so two models with different deepstack layouts
    # but same num_patches don't collide on the same BUILT entry.
    _ds_hash = 0
    for idx in deepstack_visual_indexes:
        _ds_hash = (_ds_hash * 1099511628211) ^ int(idx)
        _ds_hash &= (1 << 63) - 1
    required_attrs = (
        "_rpu_vision_handle",
        "_rpu_vision_kv_cache",
        "_rpu_vision_graph_cache",
        "_rpu_vision_max_hw",
        "_rpu_vision_has_dispatched",
        "_rpu_vision_freq_cos",
        "_rpu_vision_freq_sin",
        "_rpu_vision_position_idx_keepalive",
        "_rpu_vision_patch_embed_w",
    )

    # ------------------------------------------------------------------ #
    # Step 8: configure a pending handle, publish all Python state, then
    # retire the old handle. The public wrapper owns rollback until commit.
    # ------------------------------------------------------------------ #
    handle = torch.ops.rpu.qwen3vl_vision_create()
    _install_state["pending_handle"] = handle
    handle_finalizer = weakref.finalize(
        vision_model, _qwen3vl_vision_destroy_handle, h=handle)
    _install_state["pending_finalizer"] = handle_finalizer

    if w8a16:  # the _w8a16 op only exists on the w8a16-built extension
        torch.ops.rpu.qwen3vl_vision_set_weights_w8a16(
            handle, *_vision_args,
            q_s_list, k_s_list, v_s_list, o_s_list, fc1_s_list, fc2_s_list)
    else:
        torch.ops.rpu.qwen3vl_vision_set_weights(handle, *_vision_args)
    torch.ops.rpu.qwen3vl_vision_set_rope(handle, freq_cos, freq_sin)
    torch.ops.rpu.qwen3vl_vision_set_chunk_size(
        handle,
        0 if execution_chunk_size == "auto" else int(execution_chunk_size),
    )
    position_idx_keepalive = torch.ops.rpu.qwen3vl_vision_position_idx_keepalive(handle)

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
    vision_model._rpu_vision_hidden_size = hidden_size
    vision_model._rpu_vision_logical_head_dim = logical_head_dim
    vision_model._rpu_vision_physical_head_dim = physical_head_dim
    vision_model._rpu_vision_logical_intermediate_size = logical_intermediate_size
    vision_model._rpu_vision_physical_intermediate_size = physical_intermediate_size
    vision_model._rpu_vision_deepstack_indexes = deepstack_visual_indexes
    vision_model._rpu_vision_deepstack_hash = _ds_hash
    vision_model._rpu_vision_execution_chunk_size = execution_chunk_size
    vision_model._rpu_vision_handle = handle
    vision_model._rpu_vision_handle_finalizer = handle_finalizer
    vision_model._rpu_lazy_init_checked = True
    vision_model._rpu_required_attrs = required_attrs
    vision_model.forward = types.MethodType(_rpu_vision_forward, vision_model)

    from rpu_backend.graph.lazy_init_guard import _verify_lazy_init
    _verify_lazy_init(vision_model)

    if (
        _install_state["had_old_handle"]
        and (
            _install_state["old_finalizer"] is None
            or getattr(_install_state["old_finalizer"], "alive", False)
        )
    ):
        torch.ops.rpu.qwen3vl_vision_destroy(_install_state["old_handle"])
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

def _rpu_vision_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
    """RPU-dispatching forward for Qwen3VLVisionModel.

    Args (mirror HF):
        hidden_states: `[seq_len, cin*tp*ps*ps]` fp16/fp32 pixel features
            (pre-flattened patches). Will be moved to CPU + cast to fp16 for
            patch_embed.
        grid_thw: `[n_images, 3]` int (T, H, W) per image. CPU OR RPU; will
            be moved to CPU for position_idx + pos_emb_interpolate compute.

    Returns BaseModelOutputWithDeepstackFeatures matching HF semantics.

    Multi-image SDPA must not cross image boundaries. The default path encodes
    each image separately. With RPU_QWEN3VL_VISION_BATCH, consecutive equal-size
    images are packed and minibatch SDPA preserves per-image K/V isolation.
    The signature is keyed on
    (op_id, packed_num_patches, hidden, depth, deepstack_hash, fused_merger
    [, image_batch_count]).
    """
    # LingBot2-private consumer mode.  Its fused merger writes pooler +
    # DeepStack rows directly into persistent prefix buffers, so the generic
    # stable output slots are deliberately not refreshed and must not be
    # popped/cloned.  The normal/public path below is unchanged.
    _prefix_scatter_consume_only = bool(
        kwargs.pop("_rpu_prefix_scatter_consume_only", False))
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
    _host_fp32_patch = os.environ.get(
        "RPU_QWEN3VL_VISION_HOST_FP32_PATCH", "0") == "1"
    _pos_folded = False
    if getattr(self, "_rpu_patch_embed_on_device", False):
        # Opt-in: patch_embed Linear on RPU with a col-swizzled folded weight.
        # This skips the CPU GEMM and embed[N,hidden] CPU→RPU transfer.
        hs = hidden_states.to(device="rpu", dtype=torch.float16).contiguous()
        embed_packed = torch.nn.functional.linear(
            hs, self._rpu_patch_embed_w_rpu, self._rpu_patch_embed_b_rpu).contiguous()  # [N_total, hidden] RPU
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
        embed_packed = embed_packed + self._rpu_vision_pos_embeds

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
    # The fused patch merger's C++ post_fn returns the merged pooler
    # [n_i/sm², out_hidden] directly from qwen3vl_vision_forward — accumulate that instead of
    # the raw encoder hidden, and skip the Python patch merger below. Deepstack stays eager.
    _fused_merger = getattr(self, "_rpu_vision_fused_merger", False)
    _requested_chunk = getattr(
        self, "_rpu_vision_execution_chunk_size", "auto"
    )
    _exact_chunk = (
        None if _requested_chunk == "auto" else int(_requested_chunk)
    )
    if _prefix_scatter_consume_only and not _fused_merger:
        raise RuntimeError(
            "Qwen3-VL prefix-scatter consume-only requires the fused merger.")
    last_hidden_per_image: list[torch.Tensor] = []
    merged_per_image: list[torch.Tensor] = []
    deepstack_snapshots_per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_deepstack)]
    # Per-image in-graph merged DeepStack features (when _fused_merger).
    deepstack_merged_per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_deepstack)]

    # RPU_QWEN3VL_VISION_BATCH groups consecutive equal-size images. The cap
    # bounds SPM, minibatch-grid, and KV-cache use; mixed resolutions split groups.
    _batch_on = rpu_env_bool("RPU_QWEN3VL_VISION_BATCH")
    _batch_cap = max(1, int(os.environ.get("RPU_QWEN3VL_VISION_BATCH_CAP", "3"))) if _batch_on else 1

    def _minibatch_sdpa_supported(per_image_ctx_len: int, image_batch_count: int) -> bool:
        # rpu_sdpa_minibatch uses tile_m=128 for Qwen3-VL. It requires at least
        # two whole tiles per image, and its sync group requires the total block
        # count to be <=8 or a multiple of 8.
        blocks_per_image, remainder = divmod(per_image_ctx_len, 128)
        if remainder or blocks_per_image < 2:
            return False
        grid_dim_x = blocks_per_image * image_batch_count
        return grid_dim_x <= 8 or grid_dim_x % 8 == 0

    groups: list[tuple[int, int]] = []  # (first_image_idx, group_size)
    gi = 0
    while gi < len(patches_per_image):
        if _exact_chunk is None:
            g = 1
            while (g < _batch_cap and gi + g < len(patches_per_image)
                   # full grid_thw, not just patch count: [1,16,16] and [1,8,32] are both 256
                   # patches but need different RoPE/window layout — must NOT be co-batched.
                   and torch.equal(grid_thw_cpu[gi + g], grid_thw_cpu[gi])
                   and (g + 1) * patches_per_image[gi] <= cache.max_seq_len
                   and _minibatch_sdpa_supported(patches_per_image[gi], g + 1)):
                g += 1
        else:
            n_i = int(patches_per_image[gi])
            run = 1
            while (
                gi + run < len(patches_per_image)
                and torch.equal(grid_thw_cpu[gi + run], grid_thw_cpu[gi])
            ):
                run += 1
            candidates = []
            for candidate in range(1, min(run, 3) + 1):
                if run % candidate:
                    continue
                if candidate * n_i > int(cache.max_seq_len):
                    continue
                if candidate > 1 and not _minibatch_sdpa_supported(
                    n_i, candidate
                ):
                    continue
                full_chunk = ((candidate * n_i + 15) // 16) * 16
                if _exact_chunk == 144:
                    # 144 is the supported fixed split for one 256-patch
                    # image. It is incompatible with fused-merger/minibatch
                    # paths, both of which require one full-sequence chunk.
                    if candidate == 1 and not _fused_merger:
                        candidates.append(candidate)
                elif full_chunk == _exact_chunk:
                    candidates.append(candidate)
            if not candidates:
                raise ValueError(
                    "Qwen3-VL vision cannot realize exact chunk_size="
                    f"{_exact_chunk} before dispatch: image_{gi} has {n_i} "
                    f"patches, equal-grid run={run}, fused_merger="
                    f"{bool(_fused_merger)}, cache_capacity="
                    f"{cache.max_seq_len}"
                )
            g = max(candidates)
        groups.append((gi, g))
        gi += g
    if _prefix_scatter_consume_only and len(groups) != 1:
        raise RuntimeError(
            "Qwen3-VL prefix-scatter consume-only requires exactly one packed "
            f"Vision group; planner produced {groups}.")

    _batch_signature = _batch_on or any(g > 1 for _, g in groups)
    _resolved_chunks: list[int] = []

    offset = 0
    for first_idx, g in groups:
        n_i = patches_per_image[first_idx]   # patches/image (equal across the group)
        g_patches = g * n_i
        if g_patches > cache.max_seq_len:
            raise ValueError(
                f"Qwen3VL vision group@{first_idx} packs {g} image(s) into "
                f"{g_patches} patches, exceeding KV-cache capacity "
                f"{cache.max_seq_len}; increase max_seq_len or reduce image size")
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

        # Wrap in Graph.capture(sig). The signature is keyed on
        # (op_id, num_patches, hidden_size, num_layers, deepstack_hash,
        # fused_merger [, g]) so
        # each unique (packed) num_patches produces one BUILD + N REPLAYs across
        # same-shape groups. `g` (image_batch_count) is appended only when batching
        # is engaged so the batch-off path keeps its existing signature.
        # Skip the wrap when caller opts out (`_rpu_vision_graph_disable=True`) —
        # used for numerical-isolation debug (PASSTHROUGH vs RECORDING/REPLAYING).
        if getattr(self, "_rpu_vision_graph_disable", False):
            out_3d = torch.ops.rpu.qwen3vl_vision_forward(
                handle, embed_group_3d, k_caches, v_caches, g_patches, g,
            )
        else:
            _dyn = [num_layers, deepstack_hash, int(_fused_merger)] + ([g] if _batch_signature else [])
            sig = rpu_backend.graph.GraphSignature(
                op_id="qwen3vl_vision",
                shapes=[g_patches, hidden_size],
                dyn_dims=_dyn,
                dtypes=[torch.float16],
            )
            with graph_cache.capture(sig):
                out_3d = torch.ops.rpu.qwen3vl_vision_forward(
                    handle, embed_group_3d, k_caches, v_caches, g_patches, g,
                )

        resolved_chunk = int(
            torch.ops.rpu.qwen3vl_vision_get_resolved_chunk_size(handle)
        )
        _resolved_chunks.append(resolved_chunk)
        if _exact_chunk is not None and resolved_chunk != _exact_chunk:
            raise RuntimeError(
                "Qwen3-VL vision exact chunk drift: requested="
                f"{_exact_chunk}, resolved={resolved_chunk}, group@{first_idx} "
                f"has images={g}, patches={g_patches}"
            )

        if _fused_merger:
            if not _prefix_scatter_consume_only:
                # ONE pop returns ALL in-graph merged outputs from their STABLE slots (fixed DMA →
                # bit-identical): [pooler_merged, ds_merged_0, ...], each [g_patches/sm², out_hidden]
                # and g image-blocks in order (out_3d is ignored encoder output, not the pooler).
                # ⚠️ MUST .clone() each: pop returns the stable per-(num_patches) slot tensor *by
                # reference*; same-size groups share one slot, so without the clone every appended
                # entry would alias the LAST group's data. Clone snapshots this group before the
                # next forward overwrites the slot.
                merged_all = torch.ops.rpu.qwen3vl_vision_pop_merged(handle)
                expected_merged = 1 + n_deepstack
                if len(merged_all) != expected_merged:
                    raise RuntimeError(
                        f"group@{first_idx}: merged count {len(merged_all)} != "
                        f"{expected_merged}"
                    )
                merged_per_image.append(merged_all[0].clone())
                for layer_k in range(n_deepstack):
                    deepstack_merged_per_layer[layer_k].append(
                        merged_all[1 + layer_k].clone())
        else:
            # Return fresh storage because callers may retain outputs.
            last_hidden_per_image.append(out_3d.squeeze(0).clone())  # [g_patches, hidden]
            snapshots_i = torch.ops.rpu.qwen3vl_vision_pop_deepstack_snapshots(handle)
            if len(snapshots_i) != n_deepstack:
                raise RuntimeError(
                    f"group@{first_idx}: deepstack snapshot count "
                    f"{len(snapshots_i)} != {n_deepstack}"
                )
            for layer_k, snap in enumerate(snapshots_i):
                deepstack_snapshots_per_layer[layer_k].append(snap.detach().cpu().float())

        offset += g_patches

    if _prefix_scatter_consume_only:
        torch.ops.rpu.spm_alloc_reset_temporary()
        return None

    # Merger and DeepStack mergers use CPU fp32 by default or RPU when opted in.
    # The fused path already returns merged outputs; eager LayerNorm remains CPU fp32.
    merger_on_device = getattr(self, "_rpu_vision_merger_on_device", False)
    if _fused_merger:
        # Use the sole group directly or concatenate multiple groups in order.
        merged = (merged_per_image[0] if len(merged_per_image) == 1
                  else torch.cat(merged_per_image, dim=0))     # [N_total/sm², out_hidden] (in-graph)
        last_hidden = merged                                   # gr00t ignores last_hidden_state
    else:
        last_hidden = torch.cat(last_hidden_per_image, dim=0)  # [N_total, hidden] RPU
        last_hidden_cpu = last_hidden.detach().cpu().float()
        if merger_on_device:
            merged = _merger_forward_on_device(self.merger, last_hidden_cpu)
        else:
            merged = self.merger(last_hidden_cpu)              # [N_total/sm², out_hidden]

    deepstack_features: list[torch.Tensor] = []
    if _fused_merger:
        # Deepstack mergers ran in-graph — concat per-group merged features (single
        # group → use the clone directly, skip the redundant 1-element cat).
        for layer_k in range(n_deepstack):
            lst = deepstack_merged_per_layer[layer_k]
            deepstack_features.append(lst[0] if len(lst) == 1 else torch.cat(lst, dim=0))
    else:
        for layer_k, ds_merger in enumerate(self.deepstack_merger_list):
            snap_packed_cpu = torch.cat(deepstack_snapshots_per_layer[layer_k], dim=0)
            if merger_on_device:
                deepstack_features.append(_merger_forward_on_device(ds_merger, snap_packed_cpu))
            else:
                deepstack_features.append(ds_merger(snap_packed_cpu))

    torch.ops.rpu.spm_alloc_reset_temporary()

    unique_chunks = set(_resolved_chunks)
    vars(self)["_rpu_last_execution_plan"] = {
        "stage": "vision",
        "logical_len": int(total_patches),
        "execution_len": int(total_patches),
        "chunk_size": (
            next(iter(unique_chunks)) if len(unique_chunks) == 1 else 0
        ),
        "padding_rows": 0,
        "position": 0,
        "dispatch_chunk_sizes": tuple(_resolved_chunks),
    }

    return BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=last_hidden,
        pooler_output=merged,
        deepstack_features=deepstack_features,
    )
