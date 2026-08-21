"""Qwen3-VL vision encoder for RhinoVLA.

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
  - Patch embed runs OUTSIDE the fused graph on **CPU fp16** by default.
    RhinoVLA can opt in to an RPU path that pre-swizzles the folded Conv3d
    weight for the 8-core col partition before calling `aten::linear`. Plain
    row-major folded weights produce invalid output on RPU.
  - `fast_pos_embed_interpolate` runs on **CPU fp16**. The HF
    `nn.Embedding` + `view` + `permute(0,1,3,2,4,5)` + `flatten(0,4)` path
    is not layout-compatible with the RPU path.
  - Merger + DeepStack mergers run on **CPU fp32** by default. RhinoVLA can
    opt in to RPU merger MLPs: LayerNorm stays on CPU by default (fp32 or
    fp16, selected by env) for numerical stability and lower temporary
    pressure, the two Linear projections execute on RPU fp16 with swizzled
    8-core weights, and RhinoVLA may keep the output on RPU for prefix prep.
  - 2D RoPE: position_idx built per forward, copy_in to model-owned
    keepalive `[MAX_KEEPALIVE_SEQ, 2] int16`.

Graph lifecycle:
  - Each per-image vision call uses `cache.capture(sig)`; each unique
    `num_patches` has one retained graph entry.
  - Dynamo safety uses eager RPUCache initialization, `_rpu_lazy_init_checked`,
    `_rpu_required_attrs`, and `_verify_lazy_init(...)`.
"""
from __future__ import annotations

import gc
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
    transform_linear_weight,
)
from rpu_backend.api.cache import RPUCache


QWEN3_VL_VISION_ARCH = "qwen3_vl_vision"
_RPU_VISION_PREP_CACHE_MAX_ENTRIES = 16


def _grid_thw_cache_key(grid_thw_cpu: torch.Tensor, spatial_merge_size: int) -> tuple[int, ...]:
    return (int(spatial_merge_size), *[int(x) for x in grid_thw_cpu.reshape(-1).tolist()])


def _bounded_cache_put(cache: dict[Any, Any], key: Any, value: Any, max_entries: int) -> None:
    if key in cache:
        cache[key] = value
        return
    if len(cache) >= max_entries:
        oldest_key = next(iter(cache))
        del cache[oldest_key]
    cache[key] = value


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
    if tuple(freqs.shape) != (max_hw, half_axis):
        raise RuntimeError(
            "build_vision_rope_tables produced an invalid frequency shape: "
            f"expected {(max_hw, half_axis)}, got {tuple(freqs.shape)}"
        )

    cos = freqs.cos().to(dtype=dtype).contiguous()
    sin = freqs.sin().to(dtype=dtype).contiguous()
    return cos.to(device=device), sin.to(device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Weight conversion: split fused QKV, swizzle per partition
# ─────────────────────────────────────────────────────────────────────────────

_VISION_CONVERSION_IN_PROGRESS = "rhinovla-vision-conversion-in-progress"


def _convert_vision_block_weights_for_rpu(
    block, num_heads: int, hidden_size: int
) -> None:
    """In-place swizzle of a single Qwen3VLVisionBlock's weights.

    Splits the fused HF `attn.qkv` Linear `[3*dim, dim]` into three Linears
    q/k/v `[dim, dim]` (stashed as `_rpu_q_w`, `_rpu_q_b`, etc. on the block),
    then col-swizzles each for the 8-core layout. The o_proj and fc1/fc2 get
    swizzled in place via direct `tp_*_swizzle_mc_weight` calls.

    Idempotent via `_rpu_qwen3vl_vision_weights_converted` marker.
    """
    conversion_state = getattr(
        block, "_rpu_qwen3vl_vision_weights_converted", None
    )
    if conversion_state is True:
        return
    if conversion_state is not None:
        raise RuntimeError(
            "RhinoVLA vision block has a partial prior weight conversion; "
            "reload the model before retrying."
        )

    with torch.no_grad():
        # RPU linear swizzle is a byte-layout transform for fp16 weights.
        # Always cast before swizzling so callers cannot accidentally build a
        # wrong fp32 layout by installing from a CPU-fp32 reference model.
        qkv_w = block.attn.qkv.weight.detach().to(dtype=torch.float16, device="cpu").contiguous()  # [3*hidden, hidden]
        qkv_b = block.attn.qkv.bias.detach().to(dtype=torch.float16, device="cpu").contiguous()    # [3*hidden]
        expected_weight = (3 * hidden_size, hidden_size)
        if tuple(qkv_w.shape) != expected_weight:
            raise ValueError(
                "RhinoVLA vision fused QKV weight must have shape "
                f"{expected_weight}, got {tuple(qkv_w.shape)}"
            )
        expected_bias = (3 * hidden_size,)
        if tuple(qkv_b.shape) != expected_bias:
            raise ValueError(
                "RhinoVLA vision fused QKV bias must have shape "
                f"{expected_bias}, got {tuple(qkv_b.shape)}"
            )

        block._rpu_qwen3vl_vision_weights_converted = (
            _VISION_CONVERSION_IN_PROGRESS
        )

        q_w = qkv_w[0:hidden_size].contiguous()
        k_w = qkv_w[hidden_size : 2 * hidden_size].contiguous()
        v_w = qkv_w[2 * hidden_size : 3 * hidden_size].contiguous()
        q_b = qkv_b[0:hidden_size].contiguous()
        k_b = qkv_b[hidden_size : 2 * hidden_size].contiguous()
        v_b = qkv_b[2 * hidden_size : 3 * hidden_size].contiguous()

        q_w = tp_col_swizzle_mc_weight(q_w)
        k_w = tp_col_swizzle_mc_weight(k_w)
        v_w = tp_col_swizzle_mc_weight(v_w)

        block._rpu_q_w, block._rpu_k_w, block._rpu_v_w = q_w, k_w, v_w
        block._rpu_q_b, block._rpu_k_b, block._rpu_v_b = q_b, k_b, v_b

        o_w = block.attn.proj.weight.detach().to(dtype=torch.float16, device="cpu").contiguous()  # [hidden, hidden]
        o_w = tp_row_swizzle_mc_weight(o_w)
        o_b = block.attn.proj.bias.detach().to(dtype=torch.float16, device="cpu").contiguous()
        block.attn.proj.weight = nn.Parameter(o_w, requires_grad=False)
        block._rpu_o_b = o_b

        fc1_w = block.mlp.linear_fc1.weight.detach().to(dtype=torch.float16, device="cpu").contiguous()  # [intermediate, hidden]
        fc1_w = tp_col_swizzle_mc_weight(fc1_w)
        fc1_b = block.mlp.linear_fc1.bias.detach().to(dtype=torch.float16, device="cpu").contiguous()
        block.mlp.linear_fc1.weight = nn.Parameter(fc1_w, requires_grad=False)
        block._rpu_fc1_b = fc1_b

        fc2_w = block.mlp.linear_fc2.weight.detach().to(dtype=torch.float16, device="cpu").contiguous()  # [hidden, intermediate]
        fc2_w = tp_row_swizzle_mc_weight(fc2_w)
        fc2_b = block.mlp.linear_fc2.bias.detach().to(dtype=torch.float16, device="cpu").contiguous()
        block.mlp.linear_fc2.weight = nn.Parameter(fc2_w, requires_grad=False)
        block._rpu_fc2_b = fc2_b

    block._rpu_qwen3vl_vision_weights_converted = True


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


def _rpu_empty_cache_if_available() -> None:
    gc.collect()
    if hasattr(torch, "rpu") and hasattr(torch.rpu, "empty_cache"):
        torch.rpu.empty_cache()


def _rpu_memory_stats_summary() -> str:
    if not hasattr(torch, "rpu") or not hasattr(torch.rpu, "get_memory_stats"):
        return "unavailable"
    stats = torch.rpu.get_memory_stats()
    if not stats:
        return "unavailable"

    def cur(name: str) -> int:
        item = stats.get(name, {})
        return int(item.get("current", 0)) if isinstance(item, dict) else 0

    return (
        f"reserved={cur('reserved_bytes') // (1024 * 1024)}MiB "
        f"allocated={cur('allocated_bytes') // (1024 * 1024)}MiB "
        f"active={cur('active_bytes') // (1024 * 1024)}MiB "
        f"inactive_split={cur('inactive_split_bytes') // (1024 * 1024)}MiB "
        f"cached={int(stats.get('total_cached_memory', 0)) // (1024 * 1024)}MiB "
        f"largest_cached={int(stats.get('largest_available_block', 0)) // (1024 * 1024)}MiB "
        f"ooms={int(stats.get('num_ooms', 0))}"
    )


def _rpu_merger_mem_debug_enabled() -> bool:
    return rpu_env_bool("RPU_RHINOVLA_VISION_RPU_MERGERS_MEM_DEBUG")


def _to_rpu_contiguous_fp16(tensor: torch.Tensor) -> torch.Tensor:
    out = tensor.to(dtype=torch.float16, device="rpu")
    return out if out.is_contiguous() else out.contiguous()


def _materialize_vision_patch_merger_resident(
    merger: nn.Module, *, include_norm: bool = False
) -> None:
    if getattr(merger, "_rpu_merger_resident", False):
        if include_norm and not hasattr(merger, "_rpu_merger_norm_w_rpu"):
            merger._rpu_merger_norm_w_rpu = _to_rpu_contiguous_fp16(merger._rpu_merger_norm_w_cpu)
            merger._rpu_merger_norm_b_rpu = _to_rpu_contiguous_fp16(merger._rpu_merger_norm_b_cpu)
        return

    if _rpu_merger_mem_debug_enabled():
        _LOG.info("merger resident materialize before: %s", _rpu_memory_stats_summary())

    _rpu_empty_cache_if_available()
    with torch.no_grad():
        if include_norm:
            merger._rpu_merger_norm_w_rpu = _to_rpu_contiguous_fp16(merger._rpu_merger_norm_w_cpu)
            merger._rpu_merger_norm_b_rpu = _to_rpu_contiguous_fp16(merger._rpu_merger_norm_b_cpu)
        merger._rpu_merger_fc1_w_rpu = _to_rpu_contiguous_fp16(merger._rpu_merger_fc1_w_cpu)
        merger._rpu_merger_fc1_b_rpu = _to_rpu_contiguous_fp16(merger._rpu_merger_fc1_b_cpu)
        merger._rpu_merger_fc2_w_rpu = _to_rpu_contiguous_fp16(merger._rpu_merger_fc2_w_cpu)
        merger._rpu_merger_fc2_b_rpu = _to_rpu_contiguous_fp16(merger._rpu_merger_fc2_b_cpu)
    _rpu_empty_cache_if_available()
    merger._rpu_merger_resident = True

    if _rpu_merger_mem_debug_enabled():
        _LOG.info("merger resident materialize after: %s", _rpu_memory_stats_summary())


def _drop_vision_patch_merger_resident(merger: nn.Module) -> None:
    if not getattr(merger, "_rpu_merger_resident", False):
        return
    for name in (
        "_rpu_merger_norm_w_rpu",
        "_rpu_merger_norm_b_rpu",
        "_rpu_merger_fc1_w_rpu",
        "_rpu_merger_fc1_b_rpu",
        "_rpu_merger_fc2_w_rpu",
        "_rpu_merger_fc2_b_rpu",
    ):
        if hasattr(merger, name):
            delattr(merger, name)
    merger._rpu_merger_resident = False
    _rpu_empty_cache_if_available()


def _prepare_vision_patch_merger_for_rpu(
    merger: nn.Module,
    *,
    resident: bool = True,
    resident_norm: bool = False,
) -> None:
    """Stash RPU-ready weights for `Qwen3VLVisionPatchMerger`.

    The HF module is:
        norm(x or x.view(-1, 4H)) -> view(-1, 4H) -> fc1 -> GELU -> fc2

    We keep the original module intact for CPU fallback/debug and store
    swizzled fp16 copies for the opt-in RPU path. By default the norm stays
    CPU fp32, matching the WallOSS merger pattern and avoiding extra RPU
    temporaries; RhinoVLA can opt in to RPU fp16 merger norm for host-bubble
    experiments.
    """
    if getattr(merger, "_rpu_qwen3vl_merger_converted", False):
        if resident:
            _materialize_vision_patch_merger_resident(
                merger, include_norm=resident_norm
            )
        else:
            _drop_vision_patch_merger_resident(merger)
        return

    with torch.no_grad():
        hidden_size = int(merger.hidden_size)
        use_postshuffle_norm = bool(getattr(merger, "use_postshuffle_norm", False))

        merger._rpu_merger_hidden_size = hidden_size
        merger._rpu_merger_use_postshuffle_norm = use_postshuffle_norm
        merger._rpu_merger_norm_eps = float(merger.norm.eps)
        merger._rpu_merger_norm_w_cpu = (
            merger.norm.weight.detach().to(dtype=torch.float32, device="cpu").contiguous()
        )
        merger._rpu_merger_norm_b_cpu = (
            merger.norm.bias.detach().to(dtype=torch.float32, device="cpu").contiguous()
        )
        merger._rpu_merger_norm_w_cpu_fp16 = merger._rpu_merger_norm_w_cpu.to(
            dtype=torch.float16
        ).contiguous()
        merger._rpu_merger_norm_b_cpu_fp16 = merger._rpu_merger_norm_b_cpu.to(
            dtype=torch.float16
        ).contiguous()

        fc1_w = (
            merger.linear_fc1.weight.detach().to(dtype=torch.float16, device="cpu").contiguous()
        )
        fc2_w = (
            merger.linear_fc2.weight.detach().to(dtype=torch.float16, device="cpu").contiguous()
        )
        merger._rpu_merger_fc1_w_cpu = transform_linear_weight(
            fc1_w, partition=1, num_cores=8
        ).contiguous()
        merger._rpu_merger_fc2_w_cpu = transform_linear_weight(
            fc2_w, partition=1, num_cores=8
        ).contiguous()
        merger._rpu_merger_fc1_b_cpu = (
            merger.linear_fc1.bias.detach().to(dtype=torch.float16, device="cpu").contiguous()
        )
        merger._rpu_merger_fc2_b_cpu = (
            merger.linear_fc2.bias.detach().to(dtype=torch.float16, device="cpu").contiguous()
        )

    merger._rpu_qwen3vl_merger_converted = True
    merger._rpu_merger_resident = False
    if resident:
        _materialize_vision_patch_merger_resident(
            merger, include_norm=resident_norm
        )


def _run_vision_patch_merger_on_rpu(
    merger: nn.Module,
    x: torch.Tensor,
    *,
    norm_on_rpu: bool = False,
    return_rpu: bool = False,
    cpu_norm_fp16: bool = False,
) -> torch.Tensor:
    """Run a prepared Qwen3-VL patch-merger MLP on RPU."""
    hidden_size = int(merger._rpu_merger_hidden_size)
    use_postshuffle_norm = bool(merger._rpu_merger_use_postshuffle_norm)
    if cpu_norm_fp16:
        norm_w = merger._rpu_merger_norm_w_cpu_fp16
        norm_b = merger._rpu_merger_norm_b_cpu_fp16
        cpu_norm_dtype = torch.float16
    else:
        norm_w = merger._rpu_merger_norm_w_cpu
        norm_b = merger._rpu_merger_norm_b_cpu
        cpu_norm_dtype = torch.float32
    eps = float(merger._rpu_merger_norm_eps)
    resident = bool(getattr(merger, "_rpu_merger_resident", False))

    if norm_on_rpu:
        if x.device.type != "rpu":
            x_rpu = _to_rpu_contiguous_fp16(x)
        elif x.dtype != torch.float16:
            x_rpu = x.to(dtype=torch.float16).contiguous()
        else:
            x_rpu = x if x.is_contiguous() else x.contiguous()
        if use_postshuffle_norm:
            x_norm_in = x_rpu.view(-1, hidden_size)
            x_norm_in = x_norm_in if x_norm_in.is_contiguous() else x_norm_in.contiguous()
        else:
            x_norm_in = x_rpu
        if resident:
            norm_w_rpu = merger._rpu_merger_norm_w_rpu
            norm_b_rpu = merger._rpu_merger_norm_b_rpu
        else:
            norm_w_rpu = _to_rpu_contiguous_fp16(norm_w)
            norm_b_rpu = _to_rpu_contiguous_fp16(norm_b)
        x_norm = torch.nn.functional.layer_norm(
            x_norm_in,
            (int(norm_w.numel()),),
            norm_w_rpu,
            norm_b_rpu,
            eps,
        )
        if not resident:
            del norm_w_rpu, norm_b_rpu
        x_proj = x_norm.view(-1, hidden_size)
        x_proj = x_proj if x_proj.is_contiguous() else x_proj.contiguous()
        del x_norm, x_norm_in, x_rpu
    else:
        x_cpu = x.detach().to(device="cpu", dtype=cpu_norm_dtype).contiguous()
        if use_postshuffle_norm:
            x_norm_in = x_cpu.view(-1, hidden_size).contiguous()
        else:
            x_norm_in = x_cpu

        x_norm = torch.nn.functional.layer_norm(
            x_norm_in,
            (int(norm_w.numel()),),
            norm_w,
            norm_b,
            eps,
        )
        x_proj = x_norm.view(-1, hidden_size).to(
            device="rpu", dtype=torch.float16
        ).contiguous()
        del x_cpu, x_norm, x_norm_in
    torch.ops.rpu.spm_alloc_reset_temporary()

    if resident:
        fc1_w = merger._rpu_merger_fc1_w_rpu
        fc1_b = merger._rpu_merger_fc1_b_rpu
    else:
        fc1_w = _to_rpu_contiguous_fp16(merger._rpu_merger_fc1_w_cpu)
        fc1_b = _to_rpu_contiguous_fp16(merger._rpu_merger_fc1_b_cpu)
    x_proj = torch.nn.functional.linear(x_proj, fc1_w, fc1_b)
    if not resident:
        del fc1_w, fc1_b
    x_proj = torch.nn.functional.gelu(x_proj)  # eager exact-ERF CPU fallback
    torch.ops.rpu.spm_alloc_reset_temporary()

    if resident:
        fc2_w = merger._rpu_merger_fc2_w_rpu
        fc2_b = merger._rpu_merger_fc2_b_rpu
    else:
        fc2_w = _to_rpu_contiguous_fp16(merger._rpu_merger_fc2_w_cpu)
        fc2_b = _to_rpu_contiguous_fp16(merger._rpu_merger_fc2_b_cpu)
    x_proj = torch.nn.functional.linear(x_proj, fc2_w, fc2_b)
    if not resident:
        del fc2_w, fc2_b
    if return_rpu:
        out = x_proj if x_proj.is_contiguous() else x_proj.contiguous()
        torch.ops.rpu.spm_alloc_reset_temporary()
        return out
    out = x_proj.to(device="cpu", dtype=torch.float32).contiguous()
    del x_proj
    torch.ops.rpu.spm_alloc_reset_temporary()
    return out


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


def _validate_vision_grid_bounds(
    grid_thw_cpu: torch.Tensor,
    max_hw: int,
) -> None:
    """Reject a grid that cannot index the installed 2D-RoPE tables."""
    grid_hw_max = (
        int(grid_thw_cpu[:, 1:].max().item())
        if grid_thw_cpu.numel()
        else 0
    )
    if grid_hw_max > int(max_hw):
        raise ValueError(
            f"RhinoVLA vision grid H/W max {grid_hw_max} exceeds installed "
            f"max_hw {int(max_hw)}; reinstall with a larger max_hw"
        )


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
    max_seq_len: int = 1024,
    rpu_patch_embed: bool = False,
    rpu_mergers: bool = False,
    rpu_mergers_streaming: bool = False,
) -> int:
    """Install or replace the RhinoVLA vision runtime transactionally.

    A replacement is published only after its Python state and native handle
    are fully configured. Any failure destroys the pending handle immediately
    and restores the previously published install.
    """
    flags = {
        "rpu_patch_embed": rpu_patch_embed,
        "rpu_mergers": rpu_mergers,
        "rpu_mergers_streaming": rpu_mergers_streaming,
    }
    invalid_flags = [
        name for name, value in flags.items() if not isinstance(value, bool)
    ]
    if invalid_flags:
        name = invalid_flags[0]
        raise TypeError(
            f"RhinoVLA vision {name} must be a bool, "
            f"got {type(flags[name]).__name__}."
        )
    cfg = vision_config or vision_model.config
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
            rpu_patch_embed=rpu_patch_embed,
            rpu_mergers=rpu_mergers,
            rpu_mergers_streaming=rpu_mergers_streaming,
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


def register_qwen3vl_vision_fused_merger(vision_model, cores: int = 8) -> None:
    """Register patch-merger weights for the in-graph fused merger.

    Call AFTER `install_qwen3_vl_vision_for_rpu`. m0 (linear_fc1) is col-swizzled, m2
    (linear_fc2) is row-swizzled, ln_q (LayerNorm gamma/beta) raw [HID]. The C++ post_fn —
    gated on `RPU_QWEN3VL_VISION_FUSED_MERGER` (set this env before the first forward/BUILD)
    — then runs the patch merger AND the N deepstack mergers inside the vision graph;
    the merged outputs are popped via `qwen3vl_vision_pop_merged`. The forward replacement
    reads `_rpu_vision_fused_merger` to consume them. Idempotent. Default off (this function
    is only invoked from the RhinoVLA opt-in path; plain qwen3_vl never calls it).
    """
    if getattr(vision_model, "_rpu_vision_fused_merger", False):
        return

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
        # the N deepstack mergers (same structure, applied to the layer-{5,11,17} snapshots).
        dsw = [_merger_rpu_weights(dm) for dm in vision_model.deepstack_merger_list]
        if dsw:
            torch.ops.rpu.qwen3vl_vision_set_deepstack_merger_weights(
                vision_model._rpu_vision_handle,
                [d[0] for d in dsw], [d[1] for d in dsw], [d[2] for d in dsw],
                [d[3] for d in dsw], [d[4] for d in dsw], [d[5] for d in dsw])
    vision_model._rpu_vision_fused_merger = True


def _install_qwen3_vl_vision_for_rpu_impl(
    vision_model,
    *,
    vision_config: Any | None = None,
    max_hw: int = 48,
    max_seq_len: int = 1024,
    rpu_patch_embed: bool = False,
    rpu_mergers: bool = False,
    rpu_mergers_streaming: bool = False,
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
        max_seq_len: bound for RPUCache + position_idx keepalive. Default 1024
            covers any image up to 32×32 grid pre-merger. The C++ side caps
            at QWEN3VL_VISION_MAX_KEEPALIVE_SEQ (4096) — raising this also
            requires raising the C++ constant.
        rpu_patch_embed: run folded patch projection on RPU when true; otherwise
            use the shared Qwen3-VL CPU-fp16 path.
        rpu_mergers: run merger and DeepStack merger MLPs on RPU when true.
        rpu_mergers_streaming: stream merger weights on each forward instead of
            retaining RPU copies.

    Returns the C++ handle (also stashed at `vision_model._rpu_vision_handle`).
    """
    cfg = vision_config or vision_model.config

    num_layers = len(vision_model.blocks)
    num_heads = cfg.num_heads
    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size
    head_dim = hidden_size // num_heads
    spatial_merge_size = cfg.spatial_merge_size
    eps = 1e-6  # Qwen3VLVisionBlock LayerNorm eps (hardcoded in HF)
    deepstack_visual_indexes = list(cfg.deepstack_visual_indexes)

    # ------------------------------------------------------------------ #
    # Step 1: prepare encoder weights while the old install remains published.
    # ------------------------------------------------------------------ #
    for block in vision_model.blocks:
        _convert_vision_block_weights_for_rpu(block, num_heads, hidden_size)

    # ------------------------------------------------------------------ #
    # Step 2: gather per-layer weights into 16 lists (one per arg)
    # ------------------------------------------------------------------ #
    q_w_list, k_w_list, v_w_list, o_w_list = [], [], [], []
    fc1_w_list, fc2_w_list = [], []
    ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list = [], [], [], []
    q_b_list, k_b_list, v_b_list, o_b_list = [], [], [], []
    fc1_b_list, fc2_b_list = [], []

    for block in vision_model.blocks:
        q_w_list.append(block._rpu_q_w.to(dtype=torch.float16, device="rpu"))
        k_w_list.append(block._rpu_k_w.to(dtype=torch.float16, device="rpu"))
        v_w_list.append(block._rpu_v_w.to(dtype=torch.float16, device="rpu"))
        o_w_list.append(block.attn.proj.weight.to(dtype=torch.float16, device="rpu"))

        fc1_w_list.append(block.mlp.linear_fc1.weight.to(dtype=torch.float16, device="rpu"))
        fc2_w_list.append(block.mlp.linear_fc2.weight.to(dtype=torch.float16, device="rpu"))

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
    vision_args = (
        q_w_list, k_w_list, v_w_list, o_w_list,
        fc1_w_list, fc2_w_list,
        ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list,
        q_b_list, k_b_list, v_b_list, o_b_list,
        fc1_b_list, fc2_b_list,
        num_heads, head_dim, hidden_size, intermediate_size,
        float(eps),
        deepstack_visual_indexes,
    )

    # ------------------------------------------------------------------ #
    # Step 4: prepare rope tables. The keepalive view is handle-owned and is
    # retrieved only after the pending handle has been configured below.
    # ------------------------------------------------------------------ #
    freq_cos, freq_sin = build_vision_rope_tables(head_dim, max_hw, device="rpu", dtype=torch.float16)

    # ------------------------------------------------------------------ #
    # Step 5: fold patch_embed Conv3d → Linear weight. Default keeps CPU fp16
    # for shared Qwen3-VL behavior; RhinoVLA may opt in to a pre-swizzled RPU
    # path.
    #
    # `aten::linear` on RPU expects the weight to be pre-swizzled (col- or
    # row-partition layout for the 8-core GEMM). Passing an unswizzled weight
    # silently produces garbage (each core reads the wrong slice). The fold
    # output is plain `[embed_dim=1024, cin*tp*ps*ps=1536]` row-major. The RPU
    # opt-in stores a separate col-partition copy; CPU fallback keeps the raw
    # folded tensor.
    # ------------------------------------------------------------------ #
    pe = vision_model.patch_embed
    pe_w_folded = _fold_conv3d_to_linear_weight(pe.proj.weight.data)
    pe_b = pe.proj.bias.data if pe.proj.bias is not None else None

    pe_w_raw = pe_w_folded.to(dtype=torch.float16, device="cpu").contiguous()
    pe_b_raw = pe_b.to(dtype=torch.float16, device="cpu").contiguous() if pe_b is not None else None

    patch_embed_on_rpu = bool(rpu_patch_embed)
    patch_embed_w_rpu = None
    patch_embed_b_rpu = None
    if rpu_patch_embed:
        patch_embed_w_rpu = transform_linear_weight(
            pe_w_raw, partition=1, num_cores=8
        ).to(dtype=torch.float16, device="rpu").contiguous()
        patch_embed_b_rpu = (
            pe_b_raw.to(dtype=torch.float16, device="rpu").contiguous()
            if pe_b_raw is not None
            else None
        )

    # ------------------------------------------------------------------ #
    # Step 6: keep pos_embed on CPU. Default keeps mergers on CPU fp32; RhinoVLA
    # may opt in to RPU mergers while preserving CPU fp32
    # outputs for the caller contract.
    # ------------------------------------------------------------------ #
    vision_model.pos_embed.to(device="cpu", dtype=torch.float16)
    mergers_on_rpu = bool(rpu_mergers)
    mergers_resident = mergers_on_rpu and not bool(rpu_mergers_streaming)
    prep_cache_enabled = rpu_env_bool("RPU_RHINOVLA_VISION_PREP_CACHE")
    batch_mergers = mergers_on_rpu and rpu_env_bool(
        "RPU_RHINOVLA_VISION_BATCH_MERGERS")
    merger_norm_on_rpu = mergers_on_rpu and rpu_env_bool(
        "RPU_RHINOVLA_VISION_RPU_MERGER_NORM")
    merger_output_rpu = mergers_on_rpu and rpu_env_bool(
        "RPU_RHINOVLA_VISION_RPU_MERGER_OUTPUT_RPU")
    cpu_merger_norm_fp16 = mergers_on_rpu and rpu_env_bool(
        "RPU_RHINOVLA_VISION_CPU_MERGER_NORM_FP16")
    collect_raw_snapshots = not rpu_env_bool(
        "RPU_RHINOVLA_VISION_SKIP_RAW_SNAPSHOTS")
    if rpu_mergers:
        _prepare_vision_patch_merger_for_rpu(
            vision_model.merger,
            resident=mergers_resident,
            resident_norm=merger_norm_on_rpu,
        )
        for ds_merger in vision_model.deepstack_merger_list:
            _prepare_vision_patch_merger_for_rpu(
                ds_merger,
                resident=mergers_resident,
                resident_norm=merger_norm_on_rpu,
            )
    else:
        vision_model.merger.to(device="cpu", dtype=torch.float32)
        for ds_merger in vision_model.deepstack_merger_list:
            ds_merger.to(device="cpu", dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Step 7: prepare dummy KV caches + GraphCache before creating a handle.
    # ------------------------------------------------------------------ #
    vision_kv_cache = _make_dummy_vision_kv_caches(
        num_layers, max_seq_len, num_heads, head_dim
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
        "_rpu_vision_freq_cos",
        "_rpu_vision_freq_sin",
        "_rpu_vision_position_idx_keepalive",
        "_rpu_vision_patch_embed_w",
        "_rpu_vision_patch_embed_on_rpu",
        "_rpu_vision_mergers_on_rpu",
        "_rpu_vision_mergers_resident",
        "_rpu_vision_prep_cache_enabled",
        "_rpu_vision_batch_mergers",
        "_rpu_vision_merger_norm_on_rpu",
        "_rpu_vision_merger_output_rpu",
        "_rpu_vision_cpu_merger_norm_fp16",
        "_rpu_vision_collect_raw_snapshots",
        "_rpu_vision_pos_embed_cache",
        "_rpu_vision_pos_idx_cache",
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

    torch.ops.rpu.qwen3vl_vision_set_weights(handle, *vision_args)
    torch.ops.rpu.qwen3vl_vision_set_rope(handle, freq_cos, freq_sin)
    position_idx_keepalive = torch.ops.rpu.qwen3vl_vision_position_idx_keepalive(handle)

    _clear_vision_install_attrs(vision_model)
    vision_model._rpu_vision_freq_cos = freq_cos
    vision_model._rpu_vision_freq_sin = freq_sin
    vision_model._rpu_vision_position_idx_keepalive = position_idx_keepalive
    vision_model._rpu_vision_patch_embed_w = pe_w_raw
    vision_model._rpu_vision_patch_embed_b = pe_b_raw
    vision_model._rpu_vision_patch_embed_on_rpu = patch_embed_on_rpu
    if patch_embed_on_rpu:
        vision_model._rpu_vision_patch_embed_w_rpu = patch_embed_w_rpu
        vision_model._rpu_vision_patch_embed_b_rpu = patch_embed_b_rpu
    vision_model._rpu_vision_mergers_on_rpu = mergers_on_rpu
    vision_model._rpu_vision_mergers_resident = mergers_resident
    vision_model._rpu_vision_prep_cache_enabled = prep_cache_enabled
    vision_model._rpu_vision_batch_mergers = batch_mergers
    vision_model._rpu_vision_merger_norm_on_rpu = merger_norm_on_rpu
    vision_model._rpu_vision_merger_output_rpu = merger_output_rpu
    vision_model._rpu_vision_cpu_merger_norm_fp16 = cpu_merger_norm_fp16
    vision_model._rpu_vision_collect_raw_snapshots = collect_raw_snapshots
    vision_model._rpu_vision_pos_embed_cache = {}
    vision_model._rpu_vision_pos_idx_cache = {}
    vision_model._rpu_vision_kv_cache = vision_kv_cache
    vision_model._rpu_vision_graph_cache = vision_graph_cache
    vision_model._rpu_vision_spatial_merge_size = spatial_merge_size
    vision_model._rpu_vision_max_hw = int(max_hw)
    vision_model._rpu_vision_num_layers = num_layers
    vision_model._rpu_vision_hidden_size = hidden_size
    vision_model._rpu_vision_deepstack_indexes = deepstack_visual_indexes
    vision_model._rpu_vision_deepstack_hash = _ds_hash
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
              "deepstack_layers=%s, hidden=%d, num_heads=%d, head_dim=%d, "
              "max_hw=%d, rpu_patch_embed=%s, rpu_mergers=%s, "
              "rpu_mergers_resident=%s",
              handle, num_layers, deepstack_visual_indexes,
              hidden_size, num_heads, head_dim, max_hw, bool(rpu_patch_embed),
              bool(rpu_mergers), mergers_resident)

    return handle


# ─────────────────────────────────────────────────────────────────────────────
# Forward replacement (instance method bound by install_qwen3_vl_vision_for_rpu)
# ─────────────────────────────────────────────────────────────────────────────

def _rpu_vision_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
    """RPU-dispatching forward for Qwen3VLVisionModel.

    Args (mirror HF):
        hidden_states: `[seq_len, cin*tp*ps*ps]` fp16/fp32 pixel features
            (pre-flattened patches). Will be moved to CPU fp16 for the default
            patch_embed path, or RPU fp16 when RhinoVLA opts in.
        grid_thw: `[n_images, 3]` int (T, H, W) per image. CPU OR RPU; will
            be moved to CPU for position_idx + pos_emb_interpolate compute.

    Returns BaseModelOutputWithDeepstackFeatures matching HF semantics.

    Multi-image: SDPA must not cross image boundaries. The C++ vision
    encoder runs bidirectional SDPA over the packed sequence — to preserve
    per-image isolation we call it ONCE PER IMAGE here and concatenate.
    Same-shape images reuse the same GraphCache BUILT entry (one BUILD + N
    REPLAYs in steady state), keyed on (op_id, num_patches, hidden, depth,
    deepstack_hash).
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        BaseModelOutputWithDeepstackFeatures,
    )

    handle = self._rpu_vision_handle
    spatial_merge_size = self._rpu_vision_spatial_merge_size
    num_layers = self._rpu_vision_num_layers
    hidden_size = self._rpu_vision_hidden_size
    cache = self._rpu_vision_kv_cache
    graph_cache = self._rpu_vision_graph_cache
    deepstack_hash = self._rpu_vision_deepstack_hash
    mergers_on_rpu = bool(getattr(self, "_rpu_vision_mergers_on_rpu", False))
    prep_cache_enabled = bool(getattr(self, "_rpu_vision_prep_cache_enabled", False))
    batch_mergers = mergers_on_rpu and bool(getattr(self, "_rpu_vision_batch_mergers", False))
    merger_norm_on_rpu = mergers_on_rpu and bool(
        getattr(self, "_rpu_vision_merger_norm_on_rpu", False)
    )
    merger_output_rpu = mergers_on_rpu and bool(
        getattr(self, "_rpu_vision_merger_output_rpu", False)
    )
    cpu_merger_norm_fp16 = mergers_on_rpu and bool(
        getattr(self, "_rpu_vision_cpu_merger_norm_fp16", False)
    )
    # In-graph fused merger (RPU_QWEN3VL_VISION_FUSED_MERGER): the C++ post_fn ran the patch +
    # deepstack mergers inside the vision graph → consume pop_merged() and skip the eager merger.
    # Set only by register_qwen3vl_vision_fused_merger (RhinoVLA opt-in); default off.
    _fused_merger = bool(getattr(self, "_rpu_vision_fused_merger", False))
    collect_raw_snapshots = bool(
        getattr(self, "_rpu_vision_collect_raw_snapshots", True)
    )

    # Validate before patch-embed, RPU copies, native planning, or keepalive
    # writes. The native op checks token capacity but cannot prove that each
    # position_idx value is inside the installed [max_hw, ...] RoPE table.
    grid_thw_cpu = (
        grid_thw.detach().cpu()
        if grid_thw.device.type != "cpu"
        else grid_thw
    )
    _validate_vision_grid_bounds(
        grid_thw_cpu,
        self._rpu_vision_max_hw,
    )

    # ----- Input prep + patch_embed (Conv3d→Linear folded) ----------------
    if getattr(self, "_rpu_vision_patch_embed_on_rpu", False):
        if hidden_states.device.type != "rpu":
            hidden_states_rpu = hidden_states.to(device="rpu", dtype=torch.float16).contiguous()
        elif hidden_states.dtype != torch.float16:
            hidden_states_rpu = hidden_states.to(dtype=torch.float16).contiguous()
        else:
            hidden_states_rpu = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()

        pe_w = self._rpu_vision_patch_embed_w_rpu  # RPU fp16, col-swizzled
        pe_b = self._rpu_vision_patch_embed_b_rpu  # RPU fp16
        embed_packed = torch.nn.functional.linear(hidden_states_rpu, pe_w, pe_b).contiguous()
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

    # ----- fast_pos_embed_interpolate (CPU fp16 → RPU + add) -------------
    grid_cache_key = _grid_thw_cache_key(grid_thw_cpu, spatial_merge_size)
    pos_embeds = None
    if prep_cache_enabled:
        pos_embeds = self._rpu_vision_pos_embed_cache.get(grid_cache_key)
        if (
            pos_embeds is not None
            and (pos_embeds.device != embed_packed.device or pos_embeds.dtype != embed_packed.dtype)
        ):
            pos_embeds = None
    if pos_embeds is None:
        pos_embeds_cpu = type(self).fast_pos_embed_interpolate(self, grid_thw_cpu)  # [N_total, hidden] CPU fp16
        pos_embeds = pos_embeds_cpu.to(device=embed_packed.device, dtype=embed_packed.dtype).contiguous()
        if prep_cache_enabled:
            _bounded_cache_put(
                self._rpu_vision_pos_embed_cache,
                grid_cache_key,
                pos_embeds,
                _RPU_VISION_PREP_CACHE_MAX_ENTRIES,
            )
    embed_packed = embed_packed + pos_embeds  # rpu_add

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
    pos_idx_rpu_by_image = None
    if prep_cache_enabled and not getattr(self, "_rpu_vision_rope_disable", False):
        pos_idx_rpu_by_image = self._rpu_vision_pos_idx_cache.get(grid_cache_key)
        if pos_idx_rpu_by_image is None:
            pos_idx_tensors: list[torch.Tensor] = []
            for j, n_j in enumerate(patches_per_image):
                pos_idx_cpu = _compute_vision_position_idx_cpu(
                    grid_thw_cpu[j : j + 1], spatial_merge_size
                )
                if pos_idx_cpu.size(0) != n_j:
                    raise RuntimeError(
                        f"position_idx rows {pos_idx_cpu.size(0)} != image_{j} patches {n_j}"
                    )
                pos_idx_tensors.append(
                    pos_idx_cpu.to(keepalive.device).contiguous()
                )
            pos_idx_rpu_by_image = tuple(pos_idx_tensors)
            _bounded_cache_put(
                self._rpu_vision_pos_idx_cache,
                grid_cache_key,
                pos_idx_rpu_by_image,
                _RPU_VISION_PREP_CACHE_MAX_ENTRIES,
            )

    n_deepstack = len(self._rpu_vision_deepstack_indexes)
    last_hidden_per_image: list[torch.Tensor] = []
    deepstack_snapshots_per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_deepstack)]
    merged_per_image: list[torch.Tensor] = []
    deepstack_features_per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_deepstack)]
    raw_snapshots_cpu_per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_deepstack)]

    # Equal-view batching uses block-diagonal minibatch SDPA. Disabled,
    # unequal-view, and graph-disabled inputs use the per-view path.
    _batch_views = (
        rpu_env_bool("RPU_RHINOVLA_VISION_BATCH_VIEWS")
        and len(patches_per_image) > 1
        and all(n == patches_per_image[0] for n in patches_per_image)
        and not getattr(self, "_rpu_vision_graph_disable", False)
    )
    batched_out = None
    batched_snaps = None
    if _batch_views:
        if getattr(self, "_rpu_vision_rope_disable", False):
            keepalive.narrow(0, 0, total_patches).zero_()
        elif pos_idx_rpu_by_image is not None:
            keepalive.narrow(0, 0, total_patches).copy_(
                torch.cat(list(pos_idx_rpu_by_image), dim=0)
            )
        else:
            cat_idx = torch.cat(
                [
                    _compute_vision_position_idx_cpu(grid_thw_cpu[i : i + 1], spatial_merge_size)
                    for i in range(len(patches_per_image))
                ],
                dim=0,
            ).to(keepalive.device)
            keepalive.narrow(0, 0, total_patches).copy_(cat_idx)
        cache.reset_to_position(0)
        embed_all_3d = embed_packed.unsqueeze(0)
        embed_all_3d = embed_all_3d if embed_all_3d.is_contiguous() else embed_all_3d.contiguous()
        sig = rpu_backend.graph.GraphSignature(
            op_id="qwen3vl_vision",
            shapes=[total_patches, hidden_size],
            dyn_dims=[num_layers, deepstack_hash, len(patches_per_image), int(_fused_merger)],
            dtypes=[torch.float16],
        )
        # RhinoVLA-on-main: the shared qwen3vl_vision_forward op takes an int
        # `image_batch_count` (N equal-size views → minibatch SDPA), NOT the
        # rhino-vla branch's int[] view_lens. Equal-size views (all 256x256 here)
        # map cleanly to image_batch_count = len(views); keep gr00t's shared op
        # signature untouched. Require equal-size (main's forward requires it).
        _ppi = list(patches_per_image)
        if len(set(_ppi)) != 1:
            raise RuntimeError(
                "RhinoVLA batched vision on main needs equal-size views "
                f"(int image_batch_count); got patches_per_image={_ppi}"
            )
        with graph_cache.capture(sig):
            out_all_3d = torch.ops.rpu.qwen3vl_vision_forward(
                handle, embed_all_3d, k_caches, v_caches, total_patches,
                len(_ppi),
            )
        batched_out = out_all_3d.squeeze(0)  # [total_patches, hidden]
        batched_snaps = torch.ops.rpu.qwen3vl_vision_pop_deepstack_snapshots(handle)
        if len(batched_snaps) != n_deepstack:
            raise RuntimeError(
                "batched: deepstack snapshot count "
                f"{len(batched_snaps)} != {n_deepstack}"
            )

    # When the fused merger runs on-device, the per-image last_hidden and
    # deepstack-snapshot clones are dead work — last_hidden_state is ignored downstream
    # and the merger consumed the snapshots in-graph (pop_merged below). Skip the whole
    # slice loop; last_hidden is set to the batched output (== cat of the per-view slices).
    fused_batched = _fused_merger and batched_out is not None
    # Guard: the in-graph fused merger writes stable per-forward slots that the
    # single post-loop pop_merged() reads once. On the non-batched per-image path
    # (>1 image) each forward overwrites the previous view's slots, so only the
    # last view would survive — the batched path (batched_out) pops all views from
    # one forward. Fail loud rather than silently drop views.
    if _fused_merger and not fused_batched and len(patches_per_image) > 1:
        raise RuntimeError(
            "RhinoVLA fused vision merger with >1 image needs the batched-view path "
            f"(RPU_RHINOVLA_VISION_BATCH_VIEWS=1 + equal-size views); got "
            f"{len(patches_per_image)} images with batching off — the per-image merger "
            "slots would be overwritten. Enable BATCH_VIEWS or disable the fused merger."
        )
    offset = 0
    for i, n_i in enumerate(patches_per_image):
        if fused_batched:
            continue
        if batched_out is not None:
            # Slice this view out of the single batched forward.
            last_hidden_i = batched_out.narrow(0, offset, n_i).detach().clone()
            snapshots_i = [s.narrow(0, offset, n_i).detach().clone() for s in batched_snaps]
        else:
            embed_i = embed_packed.narrow(0, offset, n_i)
            embed_i = embed_i if embed_i.is_contiguous() else embed_i.contiguous()

            if getattr(self, "_rpu_vision_rope_disable", False):
                keepalive.narrow(0, 0, n_i).zero_()
            elif pos_idx_rpu_by_image is not None:
                pos_idx_rpu = pos_idx_rpu_by_image[i]
                if pos_idx_rpu.size(0) != n_i:
                    raise RuntimeError(
                        f"position_idx rows {pos_idx_rpu.size(0)} != "
                        f"image_{i} patches {n_i}"
                    )
                keepalive.narrow(0, 0, n_i).copy_(pos_idx_rpu)
            else:
                pos_idx_cpu = _compute_vision_position_idx_cpu(
                    grid_thw_cpu[i : i + 1], spatial_merge_size
                )
                if pos_idx_cpu.size(0) != n_i:
                    raise RuntimeError(
                        f"position_idx rows {pos_idx_cpu.size(0)} != "
                        f"image_{i} patches {n_i}"
                    )
                keepalive.narrow(0, 0, n_i).copy_(pos_idx_cpu.to(keepalive.device))

            cache.reset_to_position(0)

            embed_i_3d = embed_i.unsqueeze(0)
            embed_i_3d = embed_i_3d if embed_i_3d.is_contiguous() else embed_i_3d.contiguous()

            # Wrap in Graph.capture(sig). The signature is keyed on
            # (op_id, num_patches, hidden_size, num_layers, deepstack_hash) so
            # each unique num_patches produces one BUILD + N REPLAYs across
            # same-shape images. Skip the wrap when caller opts out
            # (`_rpu_vision_graph_disable=True`) — used for numerical-isolation
            # debug to compare PASSTHROUGH vs RECORDING/REPLAYING semantics.
            if getattr(self, "_rpu_vision_graph_disable", False):
                out_3d = torch.ops.rpu.qwen3vl_vision_forward(
                    handle, embed_i_3d, k_caches, v_caches, n_i,
                )
            else:
                sig = rpu_backend.graph.GraphSignature(
                    op_id="qwen3vl_vision",
                    shapes=[n_i, hidden_size],
                    dyn_dims=[num_layers, deepstack_hash, int(_fused_merger)],
                    dtypes=[torch.float16],
                )
                with graph_cache.capture(sig):
                    out_3d = torch.ops.rpu.qwen3vl_vision_forward(
                        handle, embed_i_3d, k_caches, v_caches, n_i,
                    )

            # GraphCache Path-B returns the model's shape-stable output buffer.
            # Same-shape multi-image forwards reuse that buffer, so keep a
            # per-image clone before the next image replay overwrites it.
            last_hidden_i = out_3d.squeeze(0).detach().clone()  # [n_i, hidden]

            snapshots_i = torch.ops.rpu.qwen3vl_vision_pop_deepstack_snapshots(handle)
            if len(snapshots_i) != n_deepstack:
                raise RuntimeError(
                    f"image_{i}: deepstack snapshot count "
                    f"{len(snapshots_i)} != {n_deepstack}"
                )

        last_hidden_per_image.append(last_hidden_i)
        if _fused_merger:
            # C++ merger_post_fn computes the patch + deepstack mergers in-graph;
            # Python collects neither the RPU merger nor CPU snapshots here.
            pass
        elif mergers_on_rpu:
            if batch_mergers:
                for layer_k, snap in enumerate(snapshots_i):
                    # Graph replay may reuse the snapshot output buffer on the
                    # next same-shape image. Preserve the per-image snapshot
                    # before deferring merger execution until after the loop.
                    deepstack_snapshots_per_layer[layer_k].append(
                        snap.detach().clone()
                    )
            else:
                merged_per_image.append(
                    _run_vision_patch_merger_on_rpu(
                        self.merger,
                        last_hidden_i,
                        norm_on_rpu=merger_norm_on_rpu,
                        return_rpu=merger_output_rpu,
                        cpu_norm_fp16=cpu_merger_norm_fp16,
                    )
                )
                for layer_k, snap in enumerate(snapshots_i):
                    if collect_raw_snapshots:
                        raw_snapshots_cpu_per_layer[layer_k].append(
                            snap.detach().cpu().float()
                        )
                    deepstack_features_per_layer[layer_k].append(
                        _run_vision_patch_merger_on_rpu(
                            self.deepstack_merger_list[layer_k],
                            snap,
                            norm_on_rpu=merger_norm_on_rpu,
                            return_rpu=merger_output_rpu,
                            cpu_norm_fp16=cpu_merger_norm_fp16,
                        )
                    )
        else:
            for layer_k, snap in enumerate(snapshots_i):
                deepstack_snapshots_per_layer[layer_k].append(snap.detach().cpu().float())

        offset += n_i

    if fused_batched:
        # The batched output is the concatenation of its per-view slices.
        last_hidden = batched_out  # [N_total, hidden] RPU (stable buffer; not consumed downstream)
    else:
        last_hidden = torch.cat(last_hidden_per_image, dim=0)  # [N_total, hidden] RPU

    if _fused_merger:
        # The fused merger writes pooler [0] and DeepStack [1+k] into stable DDR slots.
        merged_all = torch.ops.rpu.qwen3vl_vision_pop_merged(handle)
        # Clone unless the caller guarantees consumption before slot reuse.
        _mnc = bool(getattr(self, "_rpu_vision_merged_no_clone", False))
        merged = merged_all[0] if _mnc else merged_all[0].clone()
        deepstack_features = [
            (merged_all[1 + k] if _mnc else merged_all[1 + k].clone())
            for k in range(n_deepstack)
        ]
        raw_snapshots_cpu = []
    elif mergers_on_rpu:
        if batch_mergers:
            merged = _run_vision_patch_merger_on_rpu(
                self.merger,
                last_hidden,
                norm_on_rpu=merger_norm_on_rpu,
                return_rpu=merger_output_rpu,
                cpu_norm_fp16=cpu_merger_norm_fp16,
            )
            raw_snapshots_cpu = []
            deepstack_features = []
            for layer_k, snaps in enumerate(deepstack_snapshots_per_layer):
                snap_packed = torch.cat(snaps, dim=0).contiguous()
                if collect_raw_snapshots:
                    raw_snapshots_cpu.append(snap_packed.detach().cpu().float())
                deepstack_features.append(
                    _run_vision_patch_merger_on_rpu(
                        self.deepstack_merger_list[layer_k],
                        snap_packed,
                        norm_on_rpu=merger_norm_on_rpu,
                        return_rpu=merger_output_rpu,
                        cpu_norm_fp16=cpu_merger_norm_fp16,
                    )
                )
        else:
            merged = torch.cat(merged_per_image, dim=0).contiguous()
            raw_snapshots_cpu = [
                torch.cat(snaps, dim=0).float()
                for snaps in raw_snapshots_cpu_per_layer
            ] if collect_raw_snapshots else []
            deepstack_features = [
                torch.cat(features, dim=0).contiguous()
                for features in deepstack_features_per_layer
            ]
    else:
        last_hidden_cpu = last_hidden.detach().cpu().float()
        merged = self.merger(last_hidden_cpu)  # [N_total/sm², out_hidden]
        raw_snapshots_cpu = [
            torch.cat(snaps, dim=0).float()
            for snaps in deepstack_snapshots_per_layer
        ]
        deepstack_features = []
        for layer_k, ds_merger in enumerate(self.deepstack_merger_list):
            snap_packed_cpu = raw_snapshots_cpu[layer_k]
            deepstack_features.append(ds_merger(snap_packed_cpu))

    # Debug-only public breadcrumb for RhinoVLA / qwen3_vl layer-localization
    # harnesses. Normal callers keep using BaseModelOutputWithDeepstackFeatures.
    self._rpu_vision_last_raw_snapshot_indexes = (
        list(self._rpu_vision_deepstack_indexes) if raw_snapshots_cpu else []
    )
    self._rpu_vision_last_raw_snapshots = raw_snapshots_cpu

    torch.ops.rpu.spm_alloc_reset_temporary()

    return BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=last_hidden,
        pooler_output=merged,
        deepstack_features=deepstack_features,
    )
