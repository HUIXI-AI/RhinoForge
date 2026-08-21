"""Qwen3.5 vision tower (== HF ``Qwen3_5VisionModel``) → rpu_backend.

This module adapts the Qwen3-VL vision path (``qwen3_vl/vision.py``) with
DeepStack physically removed. Qwen3.5's vision config carries ``deepstack_visual_indexes=[]``
and the HF ``Qwen3_5VisionModel`` deletes the deepstack members, so the encoder is
byte-for-byte the qwen3vl tower minus deepstack — same 24-block ViT (fused QKV +
2D RoPE + GELU MLP), output ``BaseModelOutputWithPooling(last_hidden_state,
pooler_output)`` (no ``deepstack_features``).

Wires ``Qwen3_5VisionModel`` into the C++ vision subsystem registered as
``torch.ops.rpu.qwen3_5_vision_*`` (see ``src/fused/rpu_qwen3_5_vision_model.cpp``).

Public surface:
  - ``install_qwen3_5_vision_for_rpu(model)`` — swizzle vision blocks, create the
    C++ handle on ``model.model.visual``, set weights + rope tables, replace the
    vision-tower forward. The numeric-blocked path refuses installation unless
    controlled evaluation is explicitly enabled. Idempotent; called lazily on
    the first image forward.
  - ``fuse_visual_embeds(model, hidden, input_ids, pixel_values, image_grid_thw, ...)``
    — vision→text fusion: run the tower and merger on RPU, DMA merged rows directly
    into ``hidden``, compute 3D M-RoPE ``position_ids``, return
    ``(hidden, position_ids[3, N])``.

Architecture decisions:
  - During controlled evaluation, patch embed + addition of an HF-exact position
    tensor run as in-graph STEP0 by default. HF computes that tensor once per
    whole grid; the graph uploads it through a mutable-source DMA.
  - The patch merger runs in-graph and uses mutable-destination DMA for direct
    text fusion; standalone vision still returns a normal pooler tensor.
  - 2D RoPE: position_idx built per forward, copy_in to the model-owned keepalive
    ``[MAX_KEEPALIVE_SEQ, 2] int16``.
"""
from __future__ import annotations

import os
import threading
import types
import weakref
from typing import Any

import torch
import torch.nn as nn

import rpu_backend
from rpu_backend.adapters.qwen3_5.text import _oneshot_scope
from rpu_backend.runtime import rpu_env_bool
from rpu_backend.runtime.log import _LOG
from rpu_backend.runtime.weights import (
    tp_col_swizzle_mc_weight,
    tp_row_swizzle_mc_weight,
)
from rpu_backend.api.cache import RPUCache


QWEN3_5_VISION_ARCH = "qwen3_5_vision"
_VISION_INSTALL_LOCK = threading.RLock()


def build_vision_rope_tables(
    head_dim: int,
    max_hw: int,
    *,
    device: str | torch.device = "rpu",
    dtype: torch.dtype = torch.float16,
    theta: float = 10000.0,
    inv_freq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FreqCos / FreqSin ``[max_hw, head_dim/4]`` fp16 tables for the
    ``rope_2d_spm`` kernel.

    Mirrors ``Qwen3_5VisionRotaryEmbedding(head_dim // 2).forward(max_hw)``
    followed by element-wise cos/sin. Pass the model's ``inv_freq`` buffer when
    available: loading the model in fp16 rounds that buffer, and HF then performs
    the position outer-product in that same dtype. Upcasting it before the
    multiply changes many table entries. The kernel uses the same table for BOTH
    row and col axes — it indexes once per axis via the per-token position_idx
    int16 pair. ``head_dim`` is the full attention head dim (64 for Qwen3.5-2B/4B
    vision).
    """
    if head_dim <= 0 or (head_dim % 4) != 0:
        raise ValueError(
            f"build_vision_rope_tables: head_dim must be positive multiple of 4, "
            f"got {head_dim}"
        )
    if max_hw <= 0:
        raise ValueError(f"build_vision_rope_tables: max_hw must be positive, got {max_hw}")

    half_axis = head_dim // 4
    if inv_freq is None:
        inv_freq_cpu = 1.0 / (
            theta ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float64) / (head_dim // 2))
        )
    else:
        inv_freq_cpu = inv_freq.detach().to(device="cpu").flatten()
        if inv_freq_cpu.numel() != half_axis:
            raise ValueError(
                f"build_vision_rope_tables: inv_freq must have {half_axis} elements, "
                f"got {inv_freq_cpu.numel()}"
            )
    positions = torch.arange(max_hw, dtype=inv_freq_cpu.dtype)
    freqs = positions[:, None] * inv_freq_cpu[None, :]  # [max_hw, head_dim/4]
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

def _convert_vision_block_weights_for_rpu(
    block, num_heads: int, hidden_size: int
) -> None:
    """In-place swizzle of a single Qwen3_5VisionBlock's weights.

    Splits the fused HF ``attn.qkv`` Linear ``[3*dim, dim]`` into three Linears
    q/k/v ``[dim, dim]`` (stashed as ``_rpu_q_w``, ``_rpu_q_b``, etc. on the block),
    then col-swizzles each for the 8-core layout. o_proj and fc1/fc2 get swizzled
    in place. Idempotent via ``_rpu_qwen3_5_vision_weights_converted`` marker.
    """
    if getattr(block, "_rpu_qwen3_5_vision_weights_converted", False):
        return
    if getattr(block, "_rpu_qwen3_5_vision_conversion_started", False):
        raise RuntimeError(
            "Qwen3.5 vision weight conversion previously failed after mutation; "
            "reload the model before retrying."
        )

    with torch.no_grad():
        qkv_w = block.attn.qkv.weight.data  # [3*hidden, hidden]
        qkv_b = block.attn.qkv.bias.data    # [3*hidden]
        expected_weight = (3 * hidden_size, hidden_size)
        if tuple(qkv_w.shape) != expected_weight:
            raise ValueError(
                "Qwen3.5 vision fused QKV weight must have shape "
                f"{expected_weight}, got {tuple(qkv_w.shape)}"
            )
        expected_bias = (3 * hidden_size,)
        if tuple(qkv_b.shape) != expected_bias:
            raise ValueError(
                "Qwen3.5 vision fused QKV bias must have shape "
                f"{expected_bias}, got {tuple(qkv_b.shape)}"
            )

        # Everything below is irreversible. Keep a poison marker until the
        # complete block is published so a failed partial assignment can never
        # be followed by a double swizzle.
        block._rpu_qwen3_5_vision_conversion_started = True

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

        o_w = block.attn.proj.weight.data.contiguous()  # [hidden, hidden]
        o_w = tp_row_swizzle_mc_weight(o_w)
        o_b = block.attn.proj.bias.data.contiguous()
        block.attn.proj.weight = nn.Parameter(o_w, requires_grad=False)
        block._rpu_o_b = o_b

        fc1_w = block.mlp.linear_fc1.weight.data.contiguous()  # [intermediate, hidden]
        fc1_w = tp_col_swizzle_mc_weight(fc1_w)
        fc1_b = block.mlp.linear_fc1.bias.data.contiguous()
        block.mlp.linear_fc1.weight = nn.Parameter(fc1_w, requires_grad=False)
        block._rpu_fc1_b = fc1_b

        fc2_w = block.mlp.linear_fc2.weight.data.contiguous()  # [hidden, intermediate]
        fc2_w = tp_row_swizzle_mc_weight(fc2_w)
        fc2_b = block.mlp.linear_fc2.bias.data.contiguous()
        block.mlp.linear_fc2.weight = nn.Parameter(fc2_w, requires_grad=False)
        block._rpu_fc2_b = fc2_b

    block._rpu_qwen3_5_vision_weights_converted = True
    del block._rpu_qwen3_5_vision_conversion_started


def _fold_conv3d_to_linear_weight(conv3d_weight: torch.Tensor) -> torch.Tensor:
    """``[embed_dim, cin, tp, ps, ps]`` Conv3d weight → ``[embed_dim, cin*tp*ps*ps]``
    Linear weight.

    HF Qwen3_5VisionPatchEmbed processes flattened input ``[N, cin*tp*ps*ps]`` by
    ``view(-1, cin, tp, ps, ps)`` then Conv3d with kernel=stride=[tp, ps, ps]. This
    is mathematically equivalent to a Linear ``[embed_dim, cin*tp*ps*ps]`` because
    the reshape preserves memory order (contiguous strides match).
    """
    if conv3d_weight.dim() != 5:
        raise ValueError(
            f"_fold_conv3d_to_linear_weight: expected 5D weight, got {conv3d_weight.dim()}D"
        )
    embed_dim, cin, tp, ps_h, ps_w = conv3d_weight.shape
    return conv3d_weight.reshape(embed_dim, cin * tp * ps_h * ps_w).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Per-forward CPU compute: position_idx
# ─────────────────────────────────────────────────────────────────────────────

def _compute_vision_position_idx_cpu(
    grid_thw_cpu: torch.Tensor,
    spatial_merge_size: int,
    max_hw: int | None = None,
) -> torch.Tensor:
    """Build ``[num_patches, 2]`` int16 (row_idx, col_idx) on CPU from grid_thw.

    Mirrors HF ``Qwen3_5VisionModel.rot_pos_emb`` body — same ``coords`` layout
    (row, col stacked along the trailing dim of size 2). The RPU rope_2d kernel
    does the ``freq_table[pos_ids]`` lookup internally via position_idx +
    FreqCos/Sin table indexing. ``max_hw`` rejects indices outside that table.
    """
    grid_thw_list, merge_size = _validate_vision_grid(
        grid_thw_cpu, spatial_merge_size, "_compute_vision_position_idx_cpu")

    total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.int64)

    offset = 0
    for _, height, width in grid_thw_list:
        if max_hw is not None and (height > max_hw or width > max_hw):
            raise ValueError(
                f"_compute_vision_position_idx_cpu: image grid HxW={height}x{width} "
                f"exceeds 2D RoPE table max_hw={max_hw}"
            )
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

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    if pos_ids.max().item() >= 2**15:
        raise ValueError(
            f"_compute_vision_position_idx_cpu: max idx {pos_ids.max().item()} "
            f"exceeds int16 range"
        )
    return pos_ids.to(torch.int16).contiguous()


def _cached_vision_position_idx_cpu(
    vision_model,
    grid_thw_cpu: torch.Tensor,
    spatial_merge_size: int,
    max_hw: int | None = None,
) -> torch.Tensor:
    """Keep one exact, shape-only position layout per vision instance."""
    cache_key = (int(spatial_merge_size), None if max_hw is None else int(max_hw))
    cached_grid = getattr(vision_model, "_rpu_vision_position_idx_grid", None)
    if (
        getattr(vision_model, "_rpu_vision_position_idx_cache_key", None) != cache_key
        or cached_grid is None
        or not torch.equal(cached_grid, grid_thw_cpu)
    ):
        vision_model._rpu_vision_position_idx_cpu = _compute_vision_position_idx_cpu(
            grid_thw_cpu,
            spatial_merge_size,
            max_hw=max_hw,
        )
        vision_model._rpu_vision_position_idx_grid = grid_thw_cpu.clone()
        vision_model._rpu_vision_position_idx_cache_key = cache_key
    return vision_model._rpu_vision_position_idx_cpu


def _pack_temporal_step0_pos(pos: torch.Tensor, num_cores: int = 8) -> torch.Tensor:
    """Pack [N,H] position rows into col-parallel [core,N,H/core] order."""
    if pos.ndim != 2 or pos.size(1) % num_cores:
        raise ValueError("temporal STEP0 position width must divide num_cores")
    return pos.view(pos.size(0), num_cores, pos.size(1) // num_cores).permute(
        1, 0, 2
    ).contiguous()


def _validate_vision_grid(grid_thw_cpu, spatial_merge_size, caller):
    """Return a non-empty positive grid compatible with spatial merging."""
    if grid_thw_cpu.dim() != 2 or grid_thw_cpu.size(-1) != 3:
        raise ValueError(
            f"{caller}: grid_thw must be [n, 3], got {tuple(grid_thw_cpu.shape)}")
    if grid_thw_cpu.size(0) == 0:
        raise ValueError(f"{caller}: grid_thw must contain at least one image")
    merge_size = int(spatial_merge_size)
    if merge_size <= 0:
        raise ValueError(f"{caller}: spatial_merge_size must be positive")
    grid = [[int(x) for x in row] for row in grid_thw_cpu.tolist()]
    for t, h, w in grid:
        if min(t, h, w) <= 0:
            raise ValueError(f"{caller}: T/H/W must be positive, got {(t, h, w)}")
        if t != 1:
            raise ValueError(f"{caller}: video grids are not supported, got T={t}")
        if h % merge_size or w % merge_size:
            raise ValueError(
                f"{caller}: H/W must be divisible by spatial_merge_size="
                f"{merge_size}, got {(t, h, w)}")
    return grid, merge_size


def _visual_run_starts(input_ids_cpu, image_token_id, split_sizes):
    """Return one contiguous image-token start per image, or ``None`` if fragmented."""
    if input_ids_cpu.dim() != 2 or input_ids_cpu.size(0) != 1:
        raise ValueError("Qwen3.5 vision fusion supports batch_size == 1 only")
    positions = (input_ids_cpu[0] == image_token_id).nonzero(as_tuple=True)[0].tolist()
    if len(positions) != sum(split_sizes):
        raise RuntimeError(
            f"Qwen3.5 vision fuse: image token count {len(positions)} "
            f"!= image embeds rows {sum(split_sizes)}")
    starts = []
    offset = 0
    for size in split_sizes:
        run = positions[offset:offset + size]
        if not run or run != list(range(run[0], run[0] + size)):
            return None
        starts.append(run[0])
        offset += size
    return starts


def _qwen3_5_vision_destroy_handle(h: int) -> None:
    try:
        torch.ops.rpu.qwen3_5_vision_destroy(h)
    except Exception:
        pass


def _make_dummy_vision_kv_caches(
    num_layers: int,
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
) -> RPUCache:
    """Create an RPUCache sized for bidirectional vision SDPA.

    Vision encoder always inserts at position=0 and reads all tokens once — no
    autoregressive growth. Size the cache to ``max_seq_len`` (worst-case
    num_patches) so any forward shape fits. ``reset_to_position(0)`` per forward.
    """
    return RPUCache(
        num_layers=num_layers,
        batch_size=1,
        max_seq_len=max_seq_len,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        attn_tp=min(8, num_heads),
    )


def _resolve_vision_model(model_or_vision):
    """Accept either the top ``Qwen3_5ForConditionalGeneration`` / ``Qwen3_5Model``
    or the ``Qwen3_5VisionModel`` directly and return the vision tower."""
    if hasattr(model_or_vision, "blocks") and hasattr(model_or_vision, "merger"):
        return model_or_vision                       # already the vision tower
    inner = getattr(model_or_vision, "model", model_or_vision)  # ForCondGen → Qwen3_5Model
    return inner.visual


_QWEN3_5_VISION_READY_ATTRS = (
    "_rpu_vision_handle",
    "_rpu_vision_handle_finalizer",
    "_rpu_vision_freq_cos",
    "_rpu_vision_freq_sin",
    "_rpu_vision_position_idx_keepalive",
    "_rpu_vision_step0",
    "_rpu_vision_patch_embed_w",
    "_rpu_vision_patch_embed_b",
    "_rpu_vision_has_merger",
    "_rpu_vision_kv_cache",
    "_rpu_vision_graph_disable",
    "_rpu_vision_graph_cache",
    "_rpu_vision_graph_key",
    "_rpu_vision_graph_sig",
    "_rpu_vision_debug_graph",
    "_rpu_vision_spatial_merge_size",
    "_rpu_vision_num_layers",
    "_rpu_vision_hidden_size",
    "_rpu_qwen3_5_had_instance_forward",
    "_rpu_qwen3_5_original_forward",
)

_QWEN3_5_VISION_PREINSTALL_ATTRS = frozenset({
    "_rpu_vision_graph_disable",
    "_rpu_vision_rope_disable",
})


def _qwen3_5_vision_runtime_complete(
    vision_model, *, installed_forward=None
) -> bool:
    """Return whether the controlled-evaluation Vision runtime is complete.

    ``installed_forward`` lets a model-specific wrapper validate the shared
    forward it retained underneath its own instance-level wrapper. Normal
    Qwen3.5 callers continue to validate the currently published forward.
    """
    if getattr(vision_model, "_rpu_vision_installing", False):
        return False
    if any(not hasattr(vision_model, name)
           for name in _QWEN3_5_VISION_READY_ATTRS):
        return False
    handle = getattr(vision_model, "_rpu_vision_handle", None)
    finalizer = getattr(
        vision_model, "_rpu_vision_handle_finalizer", None
    )
    if handle is None or finalizer is None:
        return False
    if not getattr(finalizer, "alive", False):
        return False
    if getattr(vision_model, "_rpu_vision_graph_cache", None) is None:
        return False
    if getattr(vision_model, "_rpu_vision_debug_graph", None) is None:
        return False
    for name in (
        "_rpu_vision_freq_cos",
        "_rpu_vision_freq_sin",
        "_rpu_vision_position_idx_keepalive",
        "_rpu_vision_patch_embed_w",
        "_rpu_vision_kv_cache",
    ):
        if getattr(vision_model, name, None) is None:
            return False
    blocks = getattr(vision_model, "blocks", None)
    if not blocks:
        return False
    if getattr(vision_model, "_rpu_vision_num_layers", None) != len(blocks):
        return False
    if getattr(vision_model, "_rpu_vision_step0", False):
        for name in (
            "_rpu_vision_pe_w_swz",
            "_rpu_vision_pe_b_rpu",
            "_rpu_vision_step0_pos_grid",
            "_rpu_vision_step0_pos",
            "_rpu_vision_step0_pos_temporal_num_frames",
            "_rpu_vision_step0_pos_camera_batch_count",
        ):
            if not hasattr(vision_model, name):
                return False
        if getattr(vision_model, "_rpu_vision_pe_w_swz", None) is None:
            return False
        if getattr(vision_model, "_rpu_vision_pe_b_rpu", None) is None:
            return False
    if getattr(vision_model, "_rpu_vision_has_merger", False):
        if getattr(vision_model, "_rpu_vision_merger_refs", None) is None:
            return False
    for block in blocks:
        if getattr(
            block, "_rpu_qwen3_5_vision_weights_converted", False
        ) is not True:
            return False
        if getattr(block, "_rpu_qwen3_5_vision_conversion_started", False):
            return False
    if installed_forward is None:
        installed_forward = vars(vision_model).get("forward")
    return bool(
        getattr(installed_forward, "__self__", None) is vision_model
        and getattr(installed_forward, "__func__", None)
        is _rpu_vision_forward
    )


def _rollback_qwen3_5_vision_install(vision_model) -> None:
    """Destroy an in-flight handle and leave no callable half-install."""
    graph_cache = getattr(
        vision_model, "_rpu_vision_graph_cache", None)
    if graph_cache is not None:
        try:
            graph_cache.clear()
        except Exception:
            pass

    finalizer = getattr(
        vision_model, "_rpu_vision_installing_finalizer", None)
    if finalizer is None:
        finalizer = getattr(
            vision_model, "_rpu_vision_handle_finalizer", None)
    handle = getattr(vision_model, "_rpu_vision_installing_handle", None)
    if handle is None:
        handle = getattr(vision_model, "_rpu_vision_handle", None)
    if finalizer is not None and getattr(finalizer, "alive", False):
        finalizer()
    elif finalizer is None and handle is not None:
        _qwen3_5_vision_destroy_handle(handle)

    model_state = vars(vision_model)
    for name in tuple(model_state):
        if name.startswith("_rpu_vision_"):
            model_state.pop(name, None)
    had_instance_forward = getattr(
        vision_model, "_rpu_qwen3_5_had_instance_forward", None)
    original_forward = getattr(
        vision_model, "_rpu_qwen3_5_original_forward", None)
    if had_instance_forward is True:
        model_state["forward"] = original_forward
    elif had_instance_forward is False:
        model_state.pop("forward", None)
    elif had_instance_forward is None and original_forward is not None:
        # Compatibility with an in-process install started by older code.
        model_state["forward"] = original_forward


def install_qwen3_5_vision_for_rpu(
    model,
    *,
    vision_config: Any | None = None,
    max_hw: int = 128,        # 2D RoPE table rows; supports up to 2048 px per side and about 4:1 aspect ratio
    max_seq_len: int = 4096,  # generic vision KV-cache cap; G0.5 uses isolated 768-row spatial minibatches
    _allow_certified_g05_k1: bool = False,
) -> int:
    """Install the experimental vision path transactionally.

    A live install is an idempotent no-op. Failures before weight mutation can
    be retried; failures after an irreversible block transform retain a poison
    marker and require a model reload. In either case the in-flight handle and
    every ``_rpu_vision_*`` owner are removed and the pre-install forward is
    restored. ``_allow_certified_g05_k1`` is private to the exact
    module/class/config-selected G0.5 adapter; its caller separately binds
    checkpoint provenance, and generic Qwen3.5 Vision remains fail-closed.
    """
    vision_model = _resolve_vision_model(model)
    with _VISION_INSTALL_LOCK:
        if getattr(vision_model, "_rpu_vision_installing", False):
            raise RuntimeError(
                "Qwen3.5 vision installation re-entered before commit"
            )
        if _qwen3_5_vision_runtime_complete(vision_model):
            return vision_model._rpu_vision_handle
        stale_runtime_attrs = {
            name for name in vars(vision_model)
            if name.startswith("_rpu_vision_")
            and name not in _QWEN3_5_VISION_PREINSTALL_ATTRS
        }
        if stale_runtime_attrs:
            raise RuntimeError(
                "Qwen3.5 vision runtime state is incomplete; reload the "
                "model instead of reinstalling mutated weights."
            )
        try:
            return _install_qwen3_5_vision_for_rpu_impl(
                model,
                vision_config=vision_config,
                max_hw=max_hw,
                max_seq_len=max_seq_len,
                _allow_certified_g05_k1=_allow_certified_g05_k1,
            )
        except BaseException:
            if any(
                hasattr(vision_model, name)
                for name in (
                    "_rpu_vision_installing",
                    "_rpu_vision_installing_handle",
                    "_rpu_vision_handle",
                )
            ):
                _rollback_qwen3_5_vision_install(vision_model)
            raise


def _install_qwen3_5_vision_for_rpu_impl(
    model,
    *,
    vision_config: Any | None = None,
    max_hw: int = 128,
    max_seq_len: int = 4096,
    _allow_certified_g05_k1: bool = False,
) -> int:
    """Install RPU all-layers-once forward on the ``Qwen3_5VisionModel``.

    ``model`` may be the top ``Qwen3_5ForConditionalGeneration``, the middle
    ``Qwen3_5Model`` fusion, or the ``Qwen3_5VisionModel`` itself — the vision
    tower is resolved via ``model.model.visual`` (or used directly). Idempotence
    is handled by the public wrapper; this implementation performs one install.

    Returns the C++ handle (also stashed at ``vision_model._rpu_vision_handle``).
    """
    vision_model = _resolve_vision_model(model)
    cfg = vision_config or vision_model.config

    num_layers = len(vision_model.blocks)
    num_heads = cfg.num_heads
    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size
    head_dim = hidden_size // num_heads
    spatial_merge_size = cfg.spatial_merge_size
    eps = 1e-6  # Qwen3_5VisionBlock LayerNorm eps (hardcoded in HF)

    # The 2B/4B checkpoints share this exact vision tower. The 0.8B tower has
    # 12 heads; the 9B tower has head_dim=72 and 27 layers. Neither matches the
    # supported kernel/SPM envelope.
    vision_profile = (num_layers, num_heads, hidden_size, intermediate_size)
    supported_profile = (24, 16, 1024, 4096)
    if vision_profile != supported_profile:
        raise ValueError(
            f"Qwen3.5 vision profile {vision_profile} is unsupported; only the "
            f"shared 2B/4B tower {supported_profile} is supported."
        )
    if (
        not _allow_certified_g05_k1
        and os.environ.get("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED") != "1"
    ):
        raise NotImplementedError(
            "Qwen3.5 Vision 2B/4B is numerically blocked and disabled by "
            "default. Controlled evaluation is limited to the shared 2B/4B "
            "tower; results remain profile-, image-, and chunk-dependent, and "
            "may fail numerical or memory limits. Qwen3.5 text remains supported. "
            "To opt into the controlled path, set "
            "QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1 before the first image forward."
        )

    if not hasattr(vision_model, "_rpu_qwen3_5_original_forward"):
        model_vars = vars(vision_model)
        vision_model._rpu_qwen3_5_had_instance_forward = "forward" in model_vars
        vision_model._rpu_qwen3_5_original_forward = model_vars.get("forward")
    vision_model._rpu_vision_installing = True

    # ------------------------------------------------------------------ #
    # Step 1: create handle + GC finalizer
    # ------------------------------------------------------------------ #
    handle = torch.ops.rpu.qwen3_5_vision_create()
    vision_model._rpu_vision_installing_handle = handle
    handle_finalizer = weakref.finalize(
        vision_model, _qwen3_5_vision_destroy_handle, h=handle)
    vision_model._rpu_vision_installing_finalizer = handle_finalizer

    # ------------------------------------------------------------------ #
    # Step 2: convert per-block encoder weights (split fused QKV + swizzle)
    # ------------------------------------------------------------------ #
    for block in vision_model.blocks:
        _convert_vision_block_weights_for_rpu(block, num_heads, hidden_size)

    # ------------------------------------------------------------------ #
    # Step 3: gather per-layer weights into 16 lists (one per arg)
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
    # Step 4: set_weights (no deepstack_visual_indexes — Qwen3.5 has no DeepStack)
    # ------------------------------------------------------------------ #
    torch.ops.rpu.qwen3_5_vision_set_weights(
        handle,
        q_w_list, k_w_list, v_w_list, o_w_list,
        fc1_w_list, fc2_w_list,
        ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list,
        q_b_list, k_b_list, v_b_list, o_b_list,
        fc1_b_list, fc2_b_list,
        num_heads, head_dim, hidden_size, intermediate_size,
        float(eps),
    )

    # ------------------------------------------------------------------ #
    # Step 5: build + set rope tables, retrieve keepalive view
    # ------------------------------------------------------------------ #
    freq_cos, freq_sin = build_vision_rope_tables(
        head_dim,
        max_hw,
        device="rpu",
        dtype=torch.float16,
        inv_freq=vision_model.rotary_pos_emb.inv_freq,
    )
    vision_model._rpu_vision_freq_cos = freq_cos
    vision_model._rpu_vision_freq_sin = freq_sin
    torch.ops.rpu.qwen3_5_vision_set_rope(handle, freq_cos, freq_sin)

    position_idx_keepalive = torch.ops.rpu.qwen3_5_vision_position_idx_keepalive(handle)
    vision_model._rpu_vision_position_idx_keepalive = position_idx_keepalive

    # ------------------------------------------------------------------ #
    # Step 6: fold patch_embed Conv3d → Linear weight.
    #
    # STEP 0 (patch_embed + exact HF position upload) runs in the fused graph unless
    # RPU_VISION_STEP0=0. The CPU path remains available as a fallback. The folded
    # patch_embed weight is col-swizzled here, and HF computes the exact position
    # tensor once for the complete grid before graph capture. Both weight layout and
    # token order affect values without necessarily causing a runtime error.
    # ------------------------------------------------------------------ #
    pe = vision_model.patch_embed
    pe_w_folded = _fold_conv3d_to_linear_weight(pe.proj.weight.data)
    pe_b = pe.proj.bias.data if pe.proj.bias is not None else None

    use_rpu_step0 = rpu_env_bool("RPU_VISION_STEP0", default=True)
    vision_model._rpu_vision_step0 = use_rpu_step0

    # CPU copies — the fallback path reads these, and they cost ~5MB.
    vision_model._rpu_vision_patch_embed_w = pe_w_folded.to(dtype=torch.float16, device="cpu").contiguous()
    vision_model._rpu_vision_patch_embed_b = (
        pe_b.to(dtype=torch.float16, device="cpu").contiguous() if pe_b is not None else None
    )

    if use_rpu_step0:
        if pe_b is None:
            raise RuntimeError(
                "Qwen3.5 vision STEP 0: patch_embed has no bias, but the RPU path "
                "assumes one (set RPU_VISION_STEP0=0 to fall back to CPU)"
            )
        # COL-parallel swizzle: patch_embed splits the output hidden dimension.
        #
        # Deliberately the col primitive rather than transform_linear_weight(w,
        # partition): hidden=1024 satisfies BOTH the col (N%128) and row (N%16)
        # asserts, so a wrong `partition` argument would raise nothing and just
        # return wrong numbers. Naming the primitive makes that unrepresentable.
        pe_w_swz = tp_col_swizzle_mc_weight(
            pe_w_folded.to(dtype=torch.float16, device="cpu").contiguous()
        ).to("rpu")
        pe_b_rpu = pe_b.to(dtype=torch.float16, device="cpu").contiguous().to("rpu")
        # Hold refs: these back in-graph DMAs, so they must outlive every forward.
        vision_model._rpu_vision_pe_w_swz = pe_w_swz
        vision_model._rpu_vision_pe_b_rpu = pe_b_rpu

        torch.ops.rpu.qwen3_5_vision_set_patch_embed(handle, pe_w_swz, pe_b_rpu)
        vision_model._rpu_vision_step0_pos_grid = None
        vision_model._rpu_vision_step0_pos_temporal_num_frames = None
        vision_model._rpu_vision_step0_pos_camera_batch_count = None
        vision_model._rpu_vision_step0_pos = None

    # ------------------------------------------------------------------ #
    # Step 7: pos_embed stays on CPU for HF-exact whole-grid interpolation and
    # for the full CPU fallback. The patch merger goes IN-GRAPH as a
    # post_layers_fn when RPU_VISION_MERGER != 0 (the tower's last CPU math).
    # RPU_VISION_MERGER=0 keeps it CPU fp32 for A/B. The CPU module stays
    # resident either way.
    # No deepstack_merger_list on Qwen3.5.
    # ------------------------------------------------------------------ #
    vision_model.pos_embed.to(device="cpu", dtype=torch.float16)
    vision_model.merger.to(device="cpu", dtype=torch.float32)

    use_rpu_merger = rpu_env_bool("RPU_VISION_MERGER", default=True)
    if use_rpu_merger:
        merger = vision_model.merger
        # HF Qwen3_5VisionPatchMerger: norm=LayerNorm(hidden, eps=1e-6) BEFORE the
        # 2x2 merge (use_postshuffle_norm=False), then fc1 [4h,4h] / GELU / fc2
        # [out_hidden,4h]. fc1 is COL-parallel (splits the 4h output), fc2 is
        # ROW-parallel (splits the 4h K then all_reduce).
        #
        # ⚠️ Deliberately the col/row primitives, NOT transform_linear_weight(w,
        # partition): 4h and out_hidden satisfy BOTH the col (N%(16*tp)) and row
        # (N%16 / K%128) asserts, so a wrong partition would raise nothing and just
        # produce a wrong result. fc1=col, fc2=row is load-bearing (see emit_merger).
        fc1_w = merger.linear_fc1.weight.data.to(dtype=torch.float16, device="cpu").contiguous()  # [4h, 4h]
        fc1_w_swz = tp_col_swizzle_mc_weight(fc1_w).to("rpu")
        fc1_b = merger.linear_fc1.bias.data.to(dtype=torch.float16, device="cpu").contiguous().to("rpu")
        fc2_w = merger.linear_fc2.weight.data.to(dtype=torch.float16, device="cpu").contiguous()  # [out_hidden, 4h]
        fc2_w_swz = tp_row_swizzle_mc_weight(fc2_w).to("rpu")
        fc2_b = merger.linear_fc2.bias.data.to(dtype=torch.float16, device="cpu").contiguous().to("rpu")
        norm_w = merger.norm.weight.data.to(dtype=torch.float16, device="cpu").contiguous().to("rpu")
        norm_b = merger.norm.bias.data.to(dtype=torch.float16, device="cpu").contiguous().to("rpu")
        out_hidden_size = int(fc2_w.size(0))
        # Hold refs — these back in-graph DMAs, so they must outlive every forward
        # (same contract as the STEP 0 pe_w_swz refs above).
        vision_model._rpu_vision_merger_refs = (
            fc1_w_swz, fc1_b, fc2_w_swz, fc2_b, norm_w, norm_b)
        torch.ops.rpu.qwen3_5_vision_set_merger(
            handle, fc1_w_swz, fc1_b, fc2_w_swz, fc2_b, norm_w, norm_b, out_hidden_size)
    vision_model._rpu_vision_has_merger = use_rpu_merger

    # ------------------------------------------------------------------ #
    # Step 8: allocate dummy KV caches and the optional debug graph.
    # ------------------------------------------------------------------ #
    vision_model._rpu_vision_kv_cache = _make_dummy_vision_kv_caches(
        num_layers, max_seq_len, num_heads, head_dim
    )
    # One cached shape is enough for same-size images/repeated requests and bounds the
    # private Queue_t cost. A shape change evicts this entry before building the next.
    vision_model._rpu_vision_graph_disable = (
        os.environ.get("QWEN3_5_VISION_GRAPH_DISABLE", "0") != "0"
    )
    vision_model._rpu_vision_graph_cache = rpu_backend.graph.GraphCache(max_entries=1)
    vision_model._rpu_vision_graph_key = None
    vision_model._rpu_vision_graph_sig = None
    vision_model._rpu_vision_debug_graph = rpu_backend.graph.Graph()

    # Stash config for forward replacement.
    vision_model._rpu_vision_spatial_merge_size = spatial_merge_size
    vision_model._rpu_vision_num_layers = num_layers
    vision_model._rpu_vision_hidden_size = hidden_size

    # ------------------------------------------------------------------ #
    # Step 9: forward replacement
    # ------------------------------------------------------------------ #
    vision_model.forward = types.MethodType(_rpu_vision_forward, vision_model)

    # Publish the handle only after every dependent field and the replacement
    # forward are ready. Before this point rollback owns the temporary handle.
    vision_model._rpu_vision_handle = handle
    vision_model._rpu_vision_handle_finalizer = handle_finalizer
    del vision_model._rpu_vision_installing_handle
    del vision_model._rpu_vision_installing_finalizer
    del vision_model._rpu_vision_installing

    _LOG.info("Patched Qwen3_5VisionModel: handle=%d, num_layers=%d, "
              "hidden=%d, num_heads=%d, head_dim=%d, max_hw=%d",
              handle, num_layers, hidden_size, num_heads, head_dim, max_hw)

    return handle


# ─────────────────────────────────────────────────────────────────────────────
# Forward replacement (instance method bound by install_qwen3_5_vision_for_rpu)
# ─────────────────────────────────────────────────────────────────────────────

def _rpu_vision_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
    """RPU-dispatching forward for Qwen3_5VisionModel.

    Args (mirror HF):
        hidden_states: Normally ``[seq_len, cin*tp*ps*ps]`` pre-flattened pixel
            patches. Canonical G0.5 B3/K6 may pass compact
            ``[18,3,256,256]`` normalized FP16 frames instead.
        grid_thw: ``[n_images, 3]`` int (T, H, W) per image. Read on CPU to
            build position indices and the exact whole-grid position tensor.

    Returns ``BaseModelOutputWithPooling(last_hidden_state, pooler_output)`` (no
    deepstack). Generic multi-image input runs once per image. G0.5's exact
    B_camera=3/K=6/P=256 profile may instead use one camera-major packed call.
    """
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    handle = self._rpu_vision_handle
    spatial_merge_size = self._rpu_vision_spatial_merge_size
    num_layers = self._rpu_vision_num_layers
    cache = self._rpu_vision_kv_cache
    fusion_target = kwargs.pop("_rpu_fusion_target", None)
    fusion_run_starts = kwargs.pop("_rpu_fusion_run_starts", None)
    temporal_num_frames = int(kwargs.pop("_rpu_temporal_num_frames", 1))
    camera_batch_count = int(kwargs.pop("_rpu_camera_batch_count", 1))
    if temporal_num_frames < 1:
        raise ValueError("_rpu_temporal_num_frames must be positive")
    if camera_batch_count not in (1, 3):
        raise ValueError("_rpu_camera_batch_count must be 1 or 3")
    grid_thw_cpu = grid_thw.detach().cpu() if grid_thw.device.type != "cpu" else grid_thw
    _validate_vision_grid(
        grid_thw_cpu, spatial_merge_size, "Qwen3.5 vision forward")
    step0_on = getattr(self, "_rpu_vision_step0", False)
    compact_input = hidden_states.dim() == 4
    if compact_input and not step0_on:
        raise RuntimeError("compact G0.5 vision input requires RPU STEP0")

    if step0_on:
        # ----- STEP 0 runs in the fused graph: hand the tower RAW folded patches.
        # No CPU patch-embedding GEMM and no RPU→CPU→RPU round trip. HF computes
        # the small position tensor once per grid; pre_layers_fn uploads and adds
        # it after patch_embed + all_gather.
        embed_packed = (
            hidden_states.to(device="rpu", dtype=torch.float16).contiguous()
            if (hidden_states.device.type != "rpu" or hidden_states.dtype != torch.float16)
            else hidden_states.contiguous()
        )
    else:
        # ----- Fallback: patch_embed + pos_embed on CPU (the pre-STEP0 path).
        # Kept for A/B against the in-graph path — set RPU_VISION_STEP0=0.
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
        pos_embeds_cpu = type(self).fast_pos_embed_interpolate(self, grid_thw_cpu)  # [N_total, hidden] CPU fp16
        pos_embeds = pos_embeds_cpu.to(device=embed_packed.device, dtype=embed_packed.dtype).contiguous()
        embed_packed = embed_packed + pos_embeds  # rpu_add

    # ----- Per-image encoder loop -----------------------------------------
    keepalive = self._rpu_vision_position_idx_keepalive
    k_caches = [cache.k_caches[i] for i in range(num_layers)]
    v_caches = [cache.v_caches[i] for i in range(num_layers)]

    patches_per_image = grid_thw_cpu.prod(-1).tolist()  # [n_0, n_1, ...]
    total_patches = sum(patches_per_image)
    if compact_input:
        if (
            tuple(embed_packed.shape) != (18, 3, 256, 256)
            or temporal_num_frames != 6
            or camera_batch_count != 3
            or total_patches != 4608
        ):
            raise NotImplementedError(
                "compact G0.5 vision supports only B_camera=3,K=6,C=3,"
                "H=W=256,P=256"
            )
    elif embed_packed.size(0) != total_patches:
        raise RuntimeError(
            f"Qwen3.5 vision forward: patch_embed produced {embed_packed.size(0)} tokens "
            f"but grid_thw implies {total_patches}"
        )
    if len(patches_per_image) % temporal_num_frames:
        raise ValueError(
            "Qwen3.5 temporal vision requires a whole number of frame groups"
        )
    if camera_batch_count == 3:
        if (
            temporal_num_frames != 6
            or len(patches_per_image) != 18
            or set(patches_per_image) != {256}
        ):
            raise NotImplementedError(
                "packed Qwen3.5 vision supports only B_camera=3,K=6,P=256"
            )
        groups = [(0, total_patches, 3 * 256)]
    else:
        groups = []
        for grid_start in range(0, len(patches_per_image), temporal_num_frames):
            counts = patches_per_image[
                grid_start : grid_start + temporal_num_frames
            ]
            if temporal_num_frames > 1 and len(set(counts)) != 1:
                raise NotImplementedError(
                    "Qwen3.5 temporal vision requires equal patch counts across frames"
                )
            groups.append((grid_start, sum(counts), counts[-1]))

    last_hidden_per_image: list[torch.Tensor] = []
    # In-graph merger runs as a post_layers_fn inside each image's forward; we pull
    # its per-image output via the getter and cat (merge-then-cat). RPU_VISION_MERGER=0
    # leaves this off and falls back to the CPU merger over the cat'd last_hidden.
    merger_on = getattr(self, "_rpu_vision_has_merger", False)
    step0_pos_all = None
    if step0_on:
        cached_grid = self._rpu_vision_step0_pos_grid
        cached_frames = self._rpu_vision_step0_pos_temporal_num_frames
        cached_camera_batch = self._rpu_vision_step0_pos_camera_batch_count
        if (
            cached_grid is None
            or cached_frames != temporal_num_frames
            or cached_camera_batch != camera_batch_count
            or not torch.equal(cached_grid, grid_thw_cpu)
        ):
            pos_cpu = type(self).fast_pos_embed_interpolate(self, grid_thw_cpu)
            if temporal_num_frames > 1:
                packed = []
                pos_offset = 0
                for _, n_i, _ in groups:
                    packed.append(
                        _pack_temporal_step0_pos(
                            pos_cpu.narrow(0, pos_offset, n_i)
                        ).to(device="rpu", dtype=torch.float16)
                    )
                    pos_offset += n_i
                self._rpu_vision_step0_pos = tuple(packed)
            else:
                self._rpu_vision_step0_pos = pos_cpu.to(
                    device="rpu", dtype=torch.float16
                ).contiguous()
            self._rpu_vision_step0_pos_grid = grid_thw_cpu.clone()
            self._rpu_vision_step0_pos_temporal_num_frames = temporal_num_frames
            self._rpu_vision_step0_pos_camera_batch_count = camera_batch_count
        step0_pos_all = self._rpu_vision_step0_pos

    direct_fusion = fusion_target is not None
    if direct_fusion:
        if not merger_on:
            raise RuntimeError("Qwen3.5 direct vision fusion requires the RPU merger")
        expected_starts = camera_batch_count if camera_batch_count == 3 else len(groups)
        if fusion_run_starts is None or len(fusion_run_starts) != expected_starts:
            raise RuntimeError(
                "Qwen3.5 direct vision fusion requires one run start per image")
    merged_per_image: list[torch.Tensor] = []

    offset = 0
    for i, (grid_start, n_i, output_patches) in enumerate(groups):
        embed_i = (
            embed_packed
            if compact_input
            else embed_packed.narrow(0, offset, n_i).contiguous()
        )

        pos_idx_cpu = _cached_vision_position_idx_cpu(
            self,
            grid_thw_cpu[
                grid_start : grid_start + temporal_num_frames * camera_batch_count
            ],
            spatial_merge_size,
            max_hw=int(self._rpu_vision_freq_cos.size(0)),
        )
        if pos_idx_cpu.size(0) != n_i:
            raise RuntimeError(
                f"position_idx rows {pos_idx_cpu.size(0)} != "
                f"image_{i} patches {n_i}"
            )
        if getattr(self, "_rpu_vision_rope_disable", False):
            keepalive.narrow(0, 0, n_i).zero_()
        else:
            keepalive.narrow(0, 0, n_i).copy_(pos_idx_cpu.to(keepalive.device))

        cache.reset_to_position(0)

        embed_i_3d = embed_i.unsqueeze(0).contiguous()
        if step0_pos_all is None:
            step0_pos_i = None
        elif temporal_num_frames > 1:
            step0_pos_i = step0_pos_all[i]
        else:
            step0_pos_i = step0_pos_all.narrow(0, offset, n_i)
        op_args = (
            handle, embed_i_3d, k_caches, v_caches, n_i,
            step0_pos_i,
            fusion_target if direct_fusion else None,
            (
                [int(x) for x in fusion_run_starts]
                if direct_fusion and camera_batch_count == 3
                else [int(fusion_run_starts[i])] if direct_fusion else []
            ),
            temporal_num_frames,
            camera_batch_count,
        )

        if getattr(self, "_rpu_vision_graph_disable", False):
            # Diagnostic fallback; every kernel incurs one host↔RPU round trip.
            out_3d = torch.ops.rpu.qwen3_5_vision_forward(*op_args)
        elif torch.rpu.get_debug_export():
            # Debug DMAs point at per-forward buffers, so never retain their addresses.
            with _oneshot_scope(self._rpu_vision_debug_graph):
                out_3d = torch.ops.rpu.qwen3_5_vision_forward(*op_args)
        else:
            key = (
                n_i, self._rpu_vision_hidden_size, num_layers,
                temporal_num_frames, camera_batch_count, direct_fusion,
                compact_input, tuple(embed_i.shape),
            )
            if self._rpu_vision_graph_key != key:
                if self._rpu_vision_graph_sig is not None:
                    self._rpu_vision_graph_cache.evict(self._rpu_vision_graph_sig)
                self._rpu_vision_graph_sig = rpu_backend.graph.GraphSignature(
                    op_id="qwen3_5_vision",
                    shapes=[
                        n_i, self._rpu_vision_hidden_size,
                        *[int(dim) for dim in embed_i.shape],
                    ],
                    dyn_dims=[
                        num_layers, temporal_num_frames, camera_batch_count,
                        int(direct_fusion), int(compact_input),
                    ],
                    dtypes=[torch.float16],
                )
                self._rpu_vision_graph_key = key
            with self._rpu_vision_graph_cache.capture(self._rpu_vision_graph_sig):
                out_3d = torch.ops.rpu.qwen3_5_vision_forward(*op_args)

        if not direct_fusion:
            # .clone() is REQUIRED, not defensive. Graph mode returns a per-shape
            # model buffer, so a second same-size image overwrites the first view.
            # The direct path does not consume last_hidden_state and skips this copy.
            live = out_3d.squeeze(0)
            if temporal_num_frames > 1:
                patches_per_frame = output_patches // camera_batch_count
                if camera_batch_count == 1:
                    live = live.narrow(
                        0, n_i - patches_per_frame, patches_per_frame
                    )
                else:
                    live = torch.cat(
                        [
                            live.narrow(
                                0,
                                (camera * temporal_num_frames
                                 + temporal_num_frames - 1)
                                * patches_per_frame,
                                patches_per_frame,
                            )
                            for camera in range(camera_batch_count)
                        ],
                        dim=0,
                    )
            last_hidden_per_image.append(live.clone())

        if merger_on and not direct_fusion:
            # The in-graph merger already ran (post_layers_fn); pull this image's
            # merged rows. .clone() is REQUIRED for the SAME reason as out_3d above —
            # merged_buf_ is a stable per-model buffer overwritten by the next image's
            # forward, so appending a view would let image i+1 clobber image i.
            merged_i = torch.ops.rpu.qwen3_5_vision_get_merger_out(
                handle)  # [1, n_i/sm², out_hidden] RPU fp16
            merged_per_image.append(merged_i.squeeze(0).clone())
        offset += n_i

    # ----- Concatenate per-image outputs ----------------------------------
    if direct_fusion:
        last_hidden = None
        merged = None
    else:
        last_hidden = (last_hidden_per_image[0] if len(last_hidden_per_image) == 1
                       else torch.cat(last_hidden_per_image, dim=0))
        if merger_on:
            # merge-then-cat: every n_i is a multiple of sm², so groups never
            # straddle images. Keep the standalone pooler output on CPU fp32.
            merged = (merged_per_image[0] if len(merged_per_image) == 1
                      else torch.cat(merged_per_image, dim=0)).detach().cpu().float()
        else:
            merged = self.merger(last_hidden.detach().cpu().float())

    torch.ops.rpu.spm_alloc_reset_temporary()

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden,
        pooler_output=merged,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vision → text fusion, called by _rpu_qwen3_5_forward when pixel_values
# is not None. Mirrors HF Qwen3_5Model.forward's fusion semantics + 3D M-RoPE,
# minus DeepStack; contiguous image spans use direct-to-final-DDR merger output.
# ─────────────────────────────────────────────────────────────────────────────

def fuse_visual_embeds(model, hidden, input_ids, pixel_values, image_grid_thw,
                       *, attention_mask=None, video_grid_thw=None,
                       mm_token_type_ids=None, past_key_values=None, **kw):
    """Run the vision tower + fuse into ``hidden``; return ``(hidden, position_ids)``.

    ``model`` is the top ``Qwen3_5ForConditionalGeneration``; ``model.model`` is the
    ``Qwen3_5Model`` fusion (owns ``visual`` + ``get_image_features`` +
    ``compute_3d_position_ids``). ``hidden`` is ``[1, seq, hidden]`` embeds (fp16,
    RPU). It follows HF ``Qwen3_5Model.forward`` fusion semantics without
    DeepStack:
      1. vision encode and write the merger output directly into the contiguous
         image-token runs of ``hidden`` on RPU.
      2. Fall back to CPU ``masked_scatter`` only for fragmented token layouts or
         when the RPU merger is disabled.
      3. 3D ``position_ids`` via ``compute_3d_position_ids`` (get_rope_index) →
         return ``position_ids[:, 0, :]`` = ``[3, seq]`` for the prefill M-RoPE.
    """
    fusion = model.model  # Qwen3_5Model
    vision_model = fusion.visual

    # Lazy install: first image forward swizzles + installs the vision tower.
    # Vision touches the
    # RPU only when an image actually arrives.
    # The installer is a cheap locked no-op after commit and also validates
    # that an apparent existing handle still has a live finalizer.
    install_qwen3_5_vision_for_rpu(model)

    if image_grid_thw is None:
        raise ValueError("Qwen3.5 vision fuse: image_grid_thw must accompany pixel_values.")
    if video_grid_thw is not None or kw.get("pixel_values_videos") is not None:
        raise NotImplementedError("Qwen3.5 vision fuse: video inputs not supported yet.")

    image_token_id = int(model.config.image_token_id)

    if input_ids is None:
        raise ValueError("Qwen3.5 vision fuse: input_ids required to locate image tokens.")
    input_ids_cpu = input_ids.to("cpu")
    visual_mask = (input_ids_cpu == image_token_id)
    n_visual_tokens = int(visual_mask.sum())
    split_sizes = (image_grid_thw.detach().cpu().prod(-1) //
                   int(vision_model.config.spatial_merge_size) ** 2).tolist()
    if sum(split_sizes) != n_visual_tokens:
        raise RuntimeError(
            f"Qwen3.5 vision fuse: image embeds rows {sum(split_sizes)} "
            f"!= image token count {n_visual_tokens}")

    run_starts = _visual_run_starts(input_ids_cpu, image_token_id, split_sizes)
    direct_fusion = getattr(vision_model, "_rpu_vision_has_merger", False) and run_starts is not None
    if direct_fusion:
        hidden = hidden.to(device="rpu", dtype=torch.float16).contiguous()
        with torch.no_grad():
            vision_model(
                pixel_values.type(vision_model.dtype), grid_thw=image_grid_thw,
                return_dict=True, _rpu_fusion_target=hidden,
                _rpu_fusion_run_starts=run_starts)
    else:
        with torch.no_grad():
            image_outputs = fusion.get_image_features(
                pixel_values, image_grid_thw, return_dict=True)
        image_embeds = torch.cat(list(image_outputs.pooler_output), dim=0)
        hidden_cpu = hidden.to("cpu", dtype=torch.float16)
        visual_mask_expanded = visual_mask.unsqueeze(-1).expand_as(hidden_cpu)
        hidden_cpu = hidden_cpu.masked_scatter(
            visual_mask_expanded, image_embeds.to("cpu", dtype=torch.float16))
        hidden = hidden_cpu.to(device="rpu", dtype=torch.float16).contiguous()

    # ----- 3. 3D position_ids via compute_3d_position_ids (get_rope_index) ----
    # Image input is admitted only at cache position 0, so this takes the
    # get_rope_index path and stashes fusion.rope_deltas for decode.
    if mm_token_type_ids is None:
        raise ValueError(
            "Qwen3.5 vision fuse: mm_token_type_ids is required for M-RoPE "
            "(returned by the processor alongside input_ids).")
    position_ids = fusion.compute_3d_position_ids(
        input_ids=input_ids_cpu,
        inputs_embeds=hidden,
        image_grid_thw=image_grid_thw.detach().cpu(),
        video_grid_thw=None,
        attention_mask=attention_mask.to("cpu") if attention_mask is not None else None,
        past_key_values=past_key_values,
        mm_token_type_ids=mm_token_type_ids.to("cpu"),
    )  # [3, batch, seq]
    # batch=1 ⇒ [3, seq] token-indexed T/H/W for the prefill partial M-RoPE.
    position_ids = position_ids[:, 0, :].contiguous()

    # Push the decode M-RoPE offset (HF rope_deltas, set by compute_3d_position_ids)
    # to the C++ text model. DECODE after this image indexes the static cos_ at
    # position + delta (prefill uses the per-token prefill_cos_ table, no delta). Set
    # once here during the image prefill; it persists on the C++ model for the decodes.
    inner = getattr(fusion, "language_model", fusion)
    st = getattr(inner, "_rpu_qwen3_5", None)
    if st is not None and getattr(fusion, "rope_deltas", None) is not None:
        delta = int(fusion.rope_deltas.flatten()[0].item())
        torch.ops.rpu.qwen3_5_set_mrope_position_delta(st.handle, delta)

    return hidden, position_ids
