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
  - Patch embed runs OUTSIDE the fused graph on **CPU fp16** by default.
    RhinoVLA can opt in to an RPU path that pre-swizzles the folded Conv3d
    weight for the 8-core col partition before calling `aten::linear`. Plain
    row-major folded weights silently produce garbage on RPU. (F31)
  - `fast_pos_embed_interpolate` runs on **CPU fp16**. The HF
    `nn.Embedding` + `view` + `permute(0,1,3,2,4,5)` + `flatten(0,4)` path
    has a permute-then-flatten contiguity bug on RPU (empirically cos_sim
    mean ~0.1 with bimodal distribution). (F32)
  - Merger + DeepStack mergers run on **CPU fp32** by default. RhinoVLA can
    opt in to RPU merger MLPs: LayerNorm stays on CPU by default (fp32 or
    fp16, selected by env) for numerical stability and lower temporary
    pressure, the two Linear projections execute on RPU with swizzled
    8-core FP16 or W8A16 weights. RhinoVLA may keep output on RPU for prefix prep.
  - Full W8A16 is a cold install option: encoder, folded patch embedding and
    every merger use actual INT8 projections with FP16 channel scales.
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

import gc
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
from rpu_backend.runtime.weights import (
    tp_col_swizzle_mc_weight,
    tp_row_swizzle_mc_weight,
    transform_linear_weight,
)
from rpu_backend.api.cache import RPUCache


QWEN3_VL_VISION_ARCH = "qwen3_vl_vision"
_RPU_VISION_PREP_CACHE_MAX_ENTRIES = 16
_RPU_VISION_INPUT_CACHE_MAX_ENTRIES = 4


def _qwen3vl_rope_route_request() -> int:
    """Cold legacy env translator: absent=AUTO, explicit 1/0=SPM/DDR."""
    name = "RPU_QWEN3VL_VISION_ROPE_SPM"
    if name not in os.environ:
        return 0
    return 1 if rpu_env_bool(name) else 2


def _execution_graph_key_words(model) -> tuple[int, ...]:
    value = getattr(model, "_rpu_execution_graph_key_words", ())
    if not isinstance(value, tuple) or any(
        isinstance(word, bool) or not isinstance(word, int) or word < 0
        for word in value
    ):
        raise RuntimeError(
            "RhinoVLA vision _rpu_execution_graph_key_words must be a "
            "tuple of non-negative integers"
        )
    return value


def _plan_rhinovla_vision_execution(
    model, num_patches: int, image_batch_count: int,
):
    requested = getattr(
        model, "_rpu_vision_execution_chunk_size", "auto"
    )
    exact_chunk = requested if isinstance(requested, int) else None
    handle = int(model._rpu_vision_handle)
    generation = int(getattr(model, "_fmb_execution_generation", 0))
    component = str(getattr(
        model, "_fmb_execution_component_id", "vision_encoder"
    ))
    raw_forward = bool(getattr(model, "_rpu_vision_graph_disable", False))
    plan_box = {}
    execution_len, chunk_size = plan_bounded_prefill_execution(
        int(num_patches),
        int(num_patches),
        0,
        execution_owner=model,
        execution_native=("qwen3vl_vision", handle),
        execution_stage="vision",
        plan_signature=(int(image_batch_count), raw_forward),
        graph_cache=None if raw_forward else getattr(model, "_rpu_vision_graph_cache", None),
        resolve_stage_domain=lambda length: (
            torch.ops.rpu.qwen3vl_vision_resolve_stage_domain(
                handle, int(length), int(image_batch_count)
            )
        ),
        position=0,
        alignment=1,
        padding_rows=0,
        exact_chunk_size=exact_chunk,
        request_id=f"rhinovla:{component}:vision",
        plan_result_sink=lambda result: plan_box.__setitem__("result", result),
        graph_mode=GRAPH_COMPOSITE_CHILD,
        queue_owner_id=handle,
        physical_metadata=(
            (f"component:{component}", 1),
            ("execution_generation", generation),
        ),
    )
    result = plan_box["result"]
    if (
        result.selected is None
        or execution_len != int(num_patches)
        or chunk_size != result.selected.stage_tuple.compute_chunk
        or not result.selected.stage_tuple.physical_descriptor
    ):
        raise RuntimeError(
            "RhinoVLA vision dry planner returned no consumable native "
            "stage descriptor"
        )
    return result


def _vision_execution_receipt(
    model, total_patches, plans, chunks, physical_descriptors,
):
    if (
        not plans
        or len(plans) != len(chunks)
        or len(plans) != len(physical_descriptors)
        or any(
            plan.selected is None
            or plan.selected.stage_tuple.compute_chunk != chunk
            or tuple(plan.selected.stage_tuple.physical_descriptor)
            != tuple(descriptor)
            for plan, chunk, descriptor in zip(
                plans, chunks, physical_descriptors
            )
        )
    ):
        raise RuntimeError(
            "RhinoVLA vision cannot publish a descriptor without "
            "dry/forward agreement"
        )
    graph_mode = plans[0].graph_mode
    if any(plan.graph_mode != graph_mode for plan in plans[1:]):
        raise RuntimeError(
            "RhinoVLA vision cannot publish inconsistent graph lifecycles"
        )
    unique_chunks = set(chunks)
    return {
        "stage": "vision",
        "component": getattr(
            model, "_fmb_execution_component_id", "vision_encoder"
        ),
        "generation": int(getattr(model, "_fmb_execution_generation", 0)),
        "logical_len": int(total_patches),
        "execution_len": int(total_patches),
        "chunk_size": next(iter(unique_chunks)) if len(unique_chunks) == 1 else 0,
        "padding_rows": 0,
        "position": 0,
        "graph_mode": graph_mode,
        "authority": "NATIVE_A6_STAGE_DESCRIPTOR",
        "selection_scopes": tuple(plan.selection_scope for plan in plans),
        "physical_plan_digests": tuple(
            plan.physical_plan_digest for plan in plans
        ),
        "plan_digests": tuple(plan.plan_digest for plan in plans),
        "dispatch_chunk_sizes": tuple(chunks),
        "physical_descriptors": tuple(
            tuple(descriptor) for descriptor in physical_descriptors
        ),
        "dry_forward_agreement": True,
    }


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


def _prepare_vision_cached_patch_input(model, pixels: torch.Tensor, rows: int) -> torch.Tensor:
    """Reuse allocation only: every invocation uploads the current pixel values."""
    weight = model._rpu_vision_patch_embed_w_rpu
    if (not isinstance(pixels, torch.Tensor) or pixels.ndim != 2
            or tuple(pixels.shape) != (rows, int(weight.shape[1]))
            or rows <= 0 or not pixels.is_floating_point()
            or pixels.device.type not in ("cpu", "rpu")):
        raise ValueError(
            "RhinoVLA cached patch input must be floating CPU/RPU "
            f"[{rows}, {int(weight.shape[1])}] pixels"
        )
    if pixels.device.type != "cpu":
        # Already-resident callers keep the original direct path. The cache
        # addresses the CPU cast allocation, not the unavoidable input upload.
        return pixels.to(device=weight.device, dtype=torch.float16).contiguous()
    key = (tuple(pixels.shape), pixels.device, weight.device)
    cache = model._rpu_vision_patch_input_cache
    entry = cache.get(key)
    if entry is None:
        # A first inference_mode caller must not make the reusable slabs
        # unwritable for a later ordinary no_grad caller.
        with torch.inference_mode(False), torch.no_grad():
            host = torch.empty(pixels.shape, dtype=torch.float16, device="cpu")
            resident = torch.empty(pixels.shape, dtype=torch.float16, device=weight.device)
        entry = (host, resident)
        _bounded_cache_put(cache, key, entry, _RPU_VISION_INPUT_CACHE_MAX_ENTRIES)
    host, resident = entry
    with torch.no_grad():
        if pixels.dtype == torch.float16 and pixels.is_contiguous():
            resident.copy_(pixels)
        else:
            # RPU cross-dtype copy allocates src_cpu.to(dtype) internally.
            # Cast in a persistent CPU slab first to avoid that allocation.
            host.copy_(pixels)
            resident.copy_(host)
    return resident


def _cached_vision_patch_projection(model, pixels: torch.Tensor) -> torch.Tensor:
    """Reuse private output storage while recomputing the complete projection."""
    weight = model._rpu_vision_patch_embed_w_rpu
    shape = (*pixels.shape[:-1], int(weight.shape[0]))
    cache = model._rpu_vision_patch_input_cache
    key = ("projection_output", shape, pixels.device, pixels.dtype)
    output = cache.get(key)
    if output is None:
        with torch.inference_mode(False), torch.no_grad():
            output = torch.empty(shape, device=pixels.device, dtype=pixels.dtype)
        _bounded_cache_put(cache, key, output, _RPU_VISION_INPUT_CACHE_MAX_ENTRIES)
    bias = model._rpu_vision_patch_embed_b_rpu
    if model._rpu_vision_w8a16:
        torch.ops.rpu.linear_w8a16_into(
            pixels, weight, model._rpu_vision_patch_embed_scale, bias, output, False
        )
    else:
        torch.ops.rpu.linear_into(pixels, weight, bias, output, True)
    return output


def _vision_cached_grid_key(model, grid: torch.Tensor) -> tuple[int, ...]:
    merge = int(model._rpu_vision_spatial_merge_size)
    if (not isinstance(grid, torch.Tensor) or grid.device.type != "cpu"
            or grid.ndim != 2 or grid.shape[1] != 3 or not grid.shape[0]
            or grid.dtype not in (torch.int16, torch.int32, torch.int64)
            or merge <= 0 or bool((grid <= 0).any())
            or bool((grid[:, 1:] % merge != 0).any())):
        raise ValueError("RhinoVLA cached vision grid must contain positive integer [T,H,W] rows with merge-aligned H/W")
    _validate_vision_grid_bounds(grid, model._rpu_vision_max_hw)
    return _grid_thw_cache_key(grid, merge)


def _vision_tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors have no mutation counter. They remain usable, but
        # cannot prove an unchanged position owner and must be refreshed.
        return None


def _prime_vision_cached_batch_positions(model, grid: torch.Tensor, rows: int) -> None:
    """Prime the complete batch once per grid and unchanged native keepalive."""
    grid_key = _vision_cached_grid_key(model, grid)
    keepalive = model._rpu_vision_position_idx_keepalive
    if (rows != int(grid.prod(-1).sum()) or keepalive.ndim != 2
            or keepalive.shape[1] != 2 or keepalive.shape[0] < rows
            or keepalive.dtype != torch.int16 or not keepalive.is_contiguous()):
        raise ValueError("RhinoVLA cached position rows exceed or disagree with the native INT16 keepalive")
    cache = model._rpu_vision_batched_pos_idx_cache
    key = (grid_key, keepalive.device)
    entry = cache.get(key)
    if entry is None or entry[1] is None or _vision_tensor_version(entry[0]) != entry[1]:
        with torch.inference_mode(False), torch.no_grad():
            positions = _compute_vision_position_idx_cpu(
                grid, model._rpu_vision_spatial_merge_size
            ).to(device=keepalive.device).contiguous()
        entry = (positions, _vision_tensor_version(positions))
        _bounded_cache_put(cache, key, entry, _RPU_VISION_INPUT_CACHE_MAX_ENTRIES)
    positions, source_version = entry
    witness = model._rpu_vision_batch_position_witness
    version = _vision_tensor_version(keepalive)
    if (version is not None and source_version is not None and witness is not None
            and witness[0] == key and witness[1] is keepalive
            and witness[2] == version and witness[3] is positions
            and witness[4] == source_version):
        return
    with torch.no_grad():
        keepalive.narrow(0, 0, rows).copy_(positions)
    model._rpu_vision_batch_position_witness = (
        key, keepalive, _vision_tensor_version(keepalive), positions, source_version,
    )


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
    block, num_heads: int, hidden_size: int, *, w8a16: bool = False
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
    if type(w8a16) is not bool:
        raise TypeError("RhinoVLA vision w8a16 must be a bool")
    if conversion_state is not None and (
        bool(getattr(block, "_rpu_rhinovla_vision_w8a16", False)) != w8a16
    ):
        raise RuntimeError(
            "RhinoVLA vision precision cannot change after weight conversion; "
            "reload the model before retrying."
        )
    if w8a16:
        from rpu_backend.adapters.qwen3_vl.vision import (
            _convert_vision_block_weights_for_rpu as convert_w8_block,
        )
        convert_w8_block(block, num_heads, hidden_size, w8a16=True)
        block._rpu_rhinovla_vision_w8a16 = True
        return
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
        merger._rpu_merger_fc1_w_rpu = merger._rpu_merger_fc1_w_cpu.to(device="rpu").contiguous()
        merger._rpu_merger_fc1_b_rpu = _to_rpu_contiguous_fp16(merger._rpu_merger_fc1_b_cpu)
        merger._rpu_merger_fc2_w_rpu = merger._rpu_merger_fc2_w_cpu.to(device="rpu").contiguous()
        merger._rpu_merger_fc2_b_rpu = _to_rpu_contiguous_fp16(merger._rpu_merger_fc2_b_cpu)
        if getattr(merger, "_rpu_merger_w8a16", False):
            for projection in ("fc1", "fc2"):
                setattr(merger, f"_rpu_merger_{projection}_scale_rpu",
                        getattr(merger, f"_rpu_merger_{projection}_scale_cpu")
                        .to(device="rpu").contiguous())
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
        "_rpu_merger_fc1_scale_rpu",
        "_rpu_merger_fc2_scale_rpu",
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
    w8a16: bool = False,
) -> None:
    """Stash RPU-ready weights for `Qwen3VLVisionPatchMerger`.

    The HF module is:
        norm(x or x.view(-1, 4H)) -> view(-1, 4H) -> fc1 -> GELU -> fc2

    We keep the original module intact for CPU fallback/debug and store
    swizzled FP16 or W8 copies for the opt-in RPU path. By default the norm stays
    CPU fp32, matching the WallOSS merger pattern and avoiding extra RPU
    temporaries; RhinoVLA can opt in to RPU fp16 merger norm for host-bubble
    experiments.
    """
    if type(w8a16) is not bool:
        raise TypeError("RhinoVLA merger w8a16 must be a bool")
    if getattr(merger, "_rpu_qwen3vl_merger_converted", False):
        if bool(getattr(merger, "_rpu_merger_w8a16", False)) != w8a16:
            raise RuntimeError("RhinoVLA merger precision cannot change after preparation")
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
        if w8a16:
            from rpu_backend.adapters.qwen3_vl.vision import _quantize_vision_w8_weight
            fc1_w, fc1_scale = _quantize_vision_w8_weight(fc1_w)
            fc2_w, fc2_scale = _quantize_vision_w8_weight(fc2_w)
            merger._rpu_merger_fc1_scale_cpu = fc1_scale
            merger._rpu_merger_fc2_scale_cpu = fc2_scale
            # Eager fc2 uses col partition; the fused residual path uses row.
            merger._rpu_merger_fc2_quantized_cpu = fc2_w
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
    merger._rpu_merger_w8a16 = w8a16
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
    w8a16 = bool(getattr(merger, "_rpu_merger_w8a16", False))

    def project(value, weight, bias, name):
        if not w8a16:
            return torch.nn.functional.linear(value, weight, bias)
        scale = getattr(merger, f"_rpu_merger_{name}_scale_rpu") if resident else (
            getattr(merger, f"_rpu_merger_{name}_scale_cpu").to(device="rpu").contiguous()
        )
        return torch.ops.rpu.linear_w8a16(value, weight, scale, bias)

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
        fc1_w = merger._rpu_merger_fc1_w_cpu.to(device="rpu").contiguous()
        fc1_b = _to_rpu_contiguous_fp16(merger._rpu_merger_fc1_b_cpu)
    x_proj = project(x_proj, fc1_w, fc1_b, "fc1")
    if not resident:
        del fc1_w, fc1_b
    x_proj = torch.nn.functional.gelu(x_proj)  # eager exact-ERF CPU fallback
    torch.ops.rpu.spm_alloc_reset_temporary()

    if resident:
        fc2_w = merger._rpu_merger_fc2_w_rpu
        fc2_b = merger._rpu_merger_fc2_b_rpu
    else:
        fc2_w = merger._rpu_merger_fc2_w_cpu.to(device="rpu").contiguous()
        fc2_b = _to_rpu_contiguous_fp16(merger._rpu_merger_fc2_b_cpu)
    x_proj = project(x_proj, fc2_w, fc2_b, "fc2")
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
    execution_chunk_size: str | int = "auto",
    w8a16: bool = False,
) -> int:
    """Install or replace the RhinoVLA vision runtime transactionally.

    A replacement is published only after its Python state and native handle
    are fully configured. Failure retires the pending handle and restores the
    old install; uncertain retirement is retained and forbids native retries.
    """
    flags = {
        "rpu_patch_embed": rpu_patch_embed,
        "rpu_mergers": rpu_mergers,
        "rpu_mergers_streaming": rpu_mergers_streaming,
        "w8a16": w8a16,
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
    if w8a16 and (not rpu_patch_embed or not rpu_mergers):
        raise ValueError("RhinoVLA full W8 vision requires RPU patch embedding and mergers")
    if hasattr(vision_model, "_rpu_vision_w8a16") and (
        vision_model._rpu_vision_w8a16 != w8a16
    ):
        raise RuntimeError("RhinoVLA vision precision cannot change after installation")
    if execution_chunk_size != "auto" and (
        isinstance(execution_chunk_size, bool)
        or not isinstance(execution_chunk_size, int)
        or execution_chunk_size <= 0
        or execution_chunk_size % 16
    ):
        raise ValueError(
            "RhinoVLA vision execution_chunk_size must be 'auto' or a "
            f"positive multiple of 16, got {execution_chunk_size!r}"
        )
    cfg = vision_config or vision_model.config
    if getattr(cfg, "hidden_act", None) != "gelu_pytorch_tanh":
        raise ValueError(
            "install_qwen3_vl_vision_for_rpu: hidden_act must be "
            f"'gelu_pytorch_tanh', got {getattr(cfg, 'hidden_act', None)!r}."
        )
    batch_views = rpu_env_bool("RPU_RHINOVLA_VISION_BATCH_VIEWS")
    input_cache_enabled = rpu_env_bool("RPU_RHINOVLA_VISION_INPUT_CACHE")
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
            vision_config=vision_config,
            max_hw=max_hw,
            max_seq_len=max_seq_len,
            rpu_patch_embed=rpu_patch_embed,
            rpu_mergers=rpu_mergers,
            rpu_mergers_streaming=rpu_mergers_streaming,
            execution_chunk_size=execution_chunk_size,
            w8a16=w8a16,
            batch_views=batch_views,
            input_cache_enabled=input_cache_enabled,
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


def register_qwen3vl_vision_fused_merger(
    vision_model, cores: int = 8, *, w8a16: bool = False,
) -> None:
    """Register the patch-merger weights for the in-graph fused merger (ROUND-3 opt #1).

    Call AFTER `install_qwen3_vl_vision_for_rpu`. m0 (linear_fc1) is col-swizzled, m2
    (linear_fc2) is row-swizzled, ln_q (LayerNorm gamma/beta) raw [HID]. Registering
    these weights is the native post_fn authority; it then runs the patch merger
    AND the N deepstack mergers inside the vision graph;
    the merged outputs are popped via `qwen3vl_vision_pop_merged`. The forward replacement
    reads `_rpu_vision_fused_merger` to consume them. Idempotent. Default off (this function
    is only invoked from the RhinoVLA opt-in path; plain qwen3_vl never calls it).
    """
    if type(w8a16) is not bool:
        raise TypeError("RhinoVLA fused merger w8a16 must be a bool")
    if cores != 8:
        raise ValueError("RhinoVLA vision merger requires eight cores")
    if bool(getattr(vision_model, "_rpu_vision_w8a16", False)) != w8a16:
        raise ValueError("RhinoVLA fused merger precision must match the vision install")
    if getattr(vision_model, "_rpu_vision_fused_merger", False):
        return

    def _merger_rpu_weights(mg):
        if w8a16:
            _prepare_vision_patch_merger_for_rpu(mg, w8a16=True)
            mg._rpu_merger_fc2_fused_w_rpu = tp_row_swizzle_mc_weight(
                mg._rpu_merger_fc2_quantized_cpu, cores, dwidth=1
            ).to(device="rpu").contiguous()
            return (
                mg.norm.weight.detach().half().to(device="rpu").contiguous(),
                mg.norm.bias.detach().half().to(device="rpu").contiguous(),
                mg._rpu_merger_fc1_w_rpu, mg._rpu_merger_fc1_b_rpu,
                mg._rpu_merger_fc2_fused_w_rpu,
                mg._rpu_merger_fc2_b_rpu,
                mg._rpu_merger_fc1_scale_rpu, mg._rpu_merger_fc2_scale_rpu,
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
        # the N deepstack mergers (same structure, applied to the layer-{5,11,17} snapshots).
        if dsw:
            torch.ops.rpu.qwen3vl_vision_set_deepstack_merger_weights(
                vision_model._rpu_vision_handle,
                [d[0] for d in dsw], [d[1] for d in dsw], [d[2] for d in dsw],
                [d[3] for d in dsw], [d[4] for d in dsw], [d[5] for d in dsw],
                *(([d[6] for d in dsw], [d[7] for d in dsw]) if w8a16 else ()))
    # Retain scale and row-layout owners with the same native retirement lifetime.
    resource = getattr(vision_model, "_rpu_vision_retirement_state", None)
    if resource is not None:
        resource.keepalive = (resource.keepalive, pw, dsw)
    vision_model._rpu_vision_fused_merger = True
    vision_model._rpu_vision_fused_merger_w8a16 = w8a16


def _install_qwen3_vl_vision_for_rpu_impl(
    vision_model,
    *,
    vision_config: Any | None = None,
    max_hw: int = 48,
    max_seq_len: int = 1024,
    rpu_patch_embed: bool = False,
    rpu_mergers: bool = False,
    rpu_mergers_streaming: bool = False,
    execution_chunk_size: str | int = "auto",
    w8a16: bool = False,
    batch_views: bool,
    input_cache_enabled: bool = False,
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
        rpu_patch_embed: optional RhinoVLA perf path. Default False preserves
            the shared Qwen3-VL CPU-fp16 patch_embed behavior.
        rpu_mergers: optional RhinoVLA perf path for merger and DeepStack
            merger MLPs. Default False preserves shared Qwen3-VL behavior.
        rpu_mergers_streaming: force the older debug path that streams merger
            weights on every forward instead of keeping resident RPU copies.
        w8a16: use INT8 weights and FP16 per-output-channel scales for all
            encoder, folded patch and merger projections; norms stay floating point.

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
        _convert_vision_block_weights_for_rpu(
            block, num_heads, hidden_size, **({"w8a16": True} if w8a16 else {})
        )

    # ------------------------------------------------------------------ #
    # Step 2: gather per-layer weights into 16 lists (one per arg)
    # ------------------------------------------------------------------ #
    q_w_list, k_w_list, v_w_list, o_w_list = [], [], [], []
    fc1_w_list, fc2_w_list = [], []
    ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list = [], [], [], []
    q_b_list, k_b_list, v_b_list, o_b_list = [], [], [], []
    fc1_b_list, fc2_b_list = [], []
    scale_lists = tuple([] for _ in range(6))
    projection_dtype = torch.int8 if w8a16 else torch.float16

    for block in vision_model.blocks:
        block_weights = (
            block._rpu_q_w, block._rpu_k_w, block._rpu_v_w,
            block.attn.proj.weight, block.mlp.linear_fc1.weight,
            block.mlp.linear_fc2.weight,
        )
        if w8a16 and any(weight.dtype != torch.int8 for weight in block_weights):
            raise ValueError("RhinoVLA vision W8 requires all six INT8 projections")
        q_w_list.append(block._rpu_q_w.to(dtype=projection_dtype, device="rpu"))
        k_w_list.append(block._rpu_k_w.to(dtype=projection_dtype, device="rpu"))
        v_w_list.append(block._rpu_v_w.to(dtype=projection_dtype, device="rpu"))
        o_w_list.append(block.attn.proj.weight.to(dtype=projection_dtype, device="rpu"))

        fc1_w_list.append(block.mlp.linear_fc1.weight.to(dtype=projection_dtype, device="rpu"))
        fc2_w_list.append(block.mlp.linear_fc2.weight.to(dtype=projection_dtype, device="rpu"))
        if w8a16:
            for scales, name, weight in zip(
                scale_lists, ("q", "k", "v", "o", "fc1", "fc2"), block_weights,
            ):
                scale = getattr(block, f"_rpu_{name}_ws")
                if (scale.dtype != torch.float16 or scale.ndim != 1
                        or scale.numel() != weight.shape[0]
                        or not scale.is_contiguous()
                        or not bool(torch.isfinite(scale).all())
                        or not bool((scale > 0).all())):
                    raise ValueError("RhinoVLA vision W8 requires contiguous finite positive FP16 scales")
                scales.append(scale.to(device="rpu").contiguous())

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
    # path for Phase 4 perf. (F31)
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
    patch_embed_scale_rpu = None
    if rpu_patch_embed:
        pe_projection = pe_w_raw
        if w8a16:
            from rpu_backend.adapters.qwen3_vl.vision import _quantize_vision_w8_weight
            pe_projection, pe_scale = _quantize_vision_w8_weight(pe_w_raw)
            patch_embed_scale_rpu = pe_scale.to(device="rpu").contiguous()
        patch_embed_w_rpu = transform_linear_weight(
            pe_projection, partition=1, num_cores=8
        ).to(device="rpu").contiguous()
        patch_embed_b_rpu = (
            pe_b_raw.to(dtype=torch.float16, device="rpu").contiguous()
            if pe_b_raw is not None
            else None
        )

    # ------------------------------------------------------------------ #
    # Step 6: keep pos_embed on CPU. Default keeps mergers on CPU fp32; RhinoVLA
    # may opt in to RPU mergers for Phase 4 perf while preserving CPU fp32
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
            **({"w8a16": True} if w8a16 else {}),
        )
        for ds_merger in vision_model.deepstack_merger_list:
            _prepare_vision_patch_merger_for_rpu(
                ds_merger,
                resident=mergers_resident,
                resident_norm=merger_norm_on_rpu,
                **({"w8a16": True} if w8a16 else {}),
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
    patch_input_cache = {}
    batched_pos_idx_cache = {}

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
        "_rpu_vision_execution_chunk_size",
        "_rpu_vision_batch_views",
        "_rpu_vision_w8a16",
        "_rpu_vision_input_cache_enabled",
        "_rpu_vision_patch_input_cache",
        "_rpu_vision_batched_pos_idx_cache",
        "_rpu_vision_batch_position_witness",
    )

    # ------------------------------------------------------------------ #
    # Step 8: configure a pending handle, publish all Python state, then
    # retire the old handle. The public wrapper owns rollback until commit.
    # ------------------------------------------------------------------ #
    handle = torch.ops.rpu.qwen3vl_vision_create()
    _install_state["pending_handle"] = handle
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource
    resource = _InstalledNativeResource(
        vision_model, handle, torch.ops.rpu.qwen3vl_vision_destroy,
        graphs=(vision_graph_cache,),
        keepalive=(vision_args, scale_lists, freq_cos, freq_sin, vision_kv_cache,
                   pe_w_raw, pe_b_raw, patch_embed_w_rpu, patch_embed_b_rpu,
                   patch_embed_scale_rpu, patch_input_cache, batched_pos_idx_cache),
        label="RhinoVLA Vision", handle_name="_rpu_vision_handle")
    handle_finalizer = resource.finalizer
    _install_state["pending_resource"] = resource

    if w8a16:
        torch.ops.rpu.qwen3vl_vision_set_weights_w8a16(handle, *vision_args, *scale_lists)
    else:
        torch.ops.rpu.qwen3vl_vision_set_weights(handle, *vision_args)
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
    vision_model._rpu_vision_patch_embed_w = pe_w_raw
    vision_model._rpu_vision_patch_embed_b = pe_b_raw
    vision_model._rpu_vision_patch_embed_on_rpu = patch_embed_on_rpu
    vision_model._rpu_vision_w8a16 = w8a16
    vision_model._rpu_vision_input_cache_enabled = input_cache_enabled
    vision_model._rpu_vision_patch_input_cache = patch_input_cache
    vision_model._rpu_vision_batched_pos_idx_cache = batched_pos_idx_cache
    vision_model._rpu_vision_batch_position_witness = None
    # The actual uploaded owners, independent of the untouched CPU checkpoint
    # Parameters, are exposed for precision accounting and retained retirement.
    vision_model._rpu_vision_projection_weights = vision_args[:6]
    vision_model._rpu_vision_projection_scales = scale_lists
    if patch_embed_on_rpu:
        vision_model._rpu_vision_patch_embed_w_rpu = patch_embed_w_rpu
        vision_model._rpu_vision_patch_embed_b_rpu = patch_embed_b_rpu
        vision_model._rpu_vision_patch_embed_scale = patch_embed_scale_rpu
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
    vision_model._rpu_vision_execution_chunk_size = execution_chunk_size
    vision_model._rpu_vision_batch_views = batch_views
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
    elif _install_state["had_old_handle"] and _install_state["old_handle"] is not None:
        raise RuntimeError("RhinoVLA Vision replacement requires its actual retirement state")
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
            patch_embed path, or RPU fp16 when RhinoVLA opts in. (F31).
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
    input_cache_enabled = bool(getattr(self, "_rpu_vision_input_cache_enabled", False))
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
    # In-graph fused merger: the registered native post_fn ran the patch +
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
    if input_cache_enabled:
        _vision_cached_grid_key(self, grid_thw_cpu)
    _validate_vision_grid_bounds(
        grid_thw_cpu,
        self._rpu_vision_max_hw,
    )
    patches_per_image = grid_thw_cpu.prod(-1).tolist()
    total_patches = sum(patches_per_image)
    # Equal views can share one native pass without crossing image boundaries.
    _batch_views = (
        self._rpu_vision_batch_views
        and len(patches_per_image) > 1
        and all(n == patches_per_image[0] for n in patches_per_image)
        and not getattr(self, "_rpu_vision_graph_disable", False)
    )
    if input_cache_enabled:
        position_rows = total_patches if _batch_views else max(patches_per_image)
        if position_rows > self._rpu_vision_position_idx_keepalive.shape[0]:
            raise ValueError("RhinoVLA cached vision grid exceeds the native position keepalive")

    # ----- Input prep + patch_embed (Conv3d→Linear folded) ----------------
    private_patch_output = False
    if getattr(self, "_rpu_vision_patch_embed_on_rpu", False):
        if input_cache_enabled:
            hidden_states_rpu = _prepare_vision_cached_patch_input(
                self, hidden_states, total_patches
            )
        elif hidden_states.device.type != "rpu":
            hidden_states_rpu = hidden_states.to(device="rpu", dtype=torch.float16).contiguous()
        elif hidden_states.dtype != torch.float16:
            hidden_states_rpu = hidden_states.to(dtype=torch.float16).contiguous()
        else:
            hidden_states_rpu = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()

        pe_w = self._rpu_vision_patch_embed_w_rpu  # RPU fp16, col-swizzled
        pe_b = self._rpu_vision_patch_embed_b_rpu  # RPU fp16
        if prep_cache_enabled:
            # Keep the eager projection's original arithmetic: FP16 uses
            # ACC32, W8 uses ACC16. This slot is private to input preparation.
            embed_packed = _cached_vision_patch_projection(self, hidden_states_rpu)
            private_patch_output = True
        elif getattr(self, "_rpu_vision_w8a16", False):
            embed_packed = torch.ops.rpu.linear_w8a16(
                hidden_states_rpu, pe_w, self._rpu_vision_patch_embed_scale, pe_b
            ).contiguous()
        else:
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

    # ----- fast_pos_embed_interpolate (CPU fp16 → RPU + add) (F32) --------
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
    if private_patch_output:
        # The next projection overwrites every element before this addition.
        embed_packed.add_(pos_embeds)
    else:
        embed_packed = embed_packed + pos_embeds

    # ----- Per-image encoder loop -----------------------------------------
    keepalive = self._rpu_vision_position_idx_keepalive
    k_caches = [cache.k_caches[i] for i in range(num_layers)]
    v_caches = [cache.v_caches[i] for i in range(num_layers)]

    if embed_packed.size(0) != total_patches:
        raise RuntimeError(
            f"Qwen3VL vision forward: patch_embed produced {embed_packed.size(0)} tokens "
            f"but grid_thw implies {total_patches}"
        )
    pos_idx_rpu_by_image = None
    if (prep_cache_enabled and not (input_cache_enabled and _batch_views)
            and not getattr(self, "_rpu_vision_rope_disable", False)):
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

    # ----- 3-view batched encode (env-gated, equal-view) ------------------
    # One ViT pass over ALL views' patches; attention is block-diagonal per view
    # in C++ (view_lens -> minibatch SDPA). Linears/LN/rope/reduce load each
    # layer's weights ONCE instead of per view (vision is weight-VLD-bound).
    # Falls back to the per-view loop when off / unequal-view / graph-disabled.
    batched_out = None
    batched_snaps = None
    vision_plans = []
    resolved_chunks = []
    dispatched_descriptors = []
    parent_graph_key_words = _execution_graph_key_words(self)
    if _batch_views:
        if getattr(self, "_rpu_vision_rope_disable", False):
            keepalive.narrow(0, 0, total_patches).zero_()
        elif input_cache_enabled:
            _prime_vision_cached_batch_positions(self, grid_thw_cpu, total_patches)
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
        vision_plan = _plan_rhinovla_vision_execution(
            self, total_patches, len(patches_per_image)
        )
        selected = vision_plan.selected
        if selected is None:
            raise RuntimeError("RhinoVLA vision planner selected no plan")
        vision_plans.append(vision_plan)
        sig = rpu_backend.graph.GraphSignature(
            op_id="qwen3vl_vision",
            shapes=[total_patches, hidden_size],
            dyn_dims=[
                num_layers, deepstack_hash, len(patches_per_image),
                int(_fused_merger), *vision_plan.graph_key_words(),
                *parent_graph_key_words,
            ],
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
        planned_stage_descriptor = tuple(
            selected.stage_tuple.physical_descriptor
        )
        with graph_cache.capture(sig):
            out_all_3d = torch.ops.rpu.qwen3vl_vision_forward(
                handle, embed_all_3d, k_caches, v_caches, total_patches,
                len(_ppi),
                planned_stage_descriptor,
            )
        resolved_chunk = int(
            torch.ops.rpu.qwen3vl_vision_get_resolved_chunk_size(handle)
        )
        if resolved_chunk != selected.stage_tuple.compute_chunk:
            raise RuntimeError(
                "RhinoVLA vision dry/forward chunk drift: dry="
                f"{selected.stage_tuple.compute_chunk}, forward="
                f"{resolved_chunk}"
            )
        resolved_chunks.append(resolved_chunk)
        dispatched_descriptors.append(planned_stage_descriptor)
        batched_out = out_all_3d.squeeze(0)  # [total_patches, hidden]
        batched_snaps = torch.ops.rpu.qwen3vl_vision_pop_deepstack_snapshots(handle)
        if len(batched_snaps) != n_deepstack:
            raise RuntimeError(
                "batched: deepstack snapshot count "
                f"{len(batched_snaps)} != {n_deepstack}"
            )

    # T5b: when the fused merger ran on-device (RhinoVLA), the per-image last_hidden +
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
            vision_plan = _plan_rhinovla_vision_execution(self, n_i, 1)
            selected = vision_plan.selected
            if selected is None:
                raise RuntimeError("RhinoVLA vision planner selected no plan")
            vision_plans.append(vision_plan)
            planned_stage_descriptor = list(
                selected.stage_tuple.physical_descriptor
            )

            # F8: Wrap in Graph.capture(sig). The signature is keyed on
            # (op_id, num_patches, hidden_size, num_layers, deepstack_hash) so
            # each unique num_patches produces one BUILD + N REPLAYs across
            # same-shape images. Skip the wrap when caller opts out
            # (`_rpu_vision_graph_disable=True`) — used for numerical-isolation
            # debug to compare PASSTHROUGH vs RECORDING/REPLAYING semantics.
            if getattr(self, "_rpu_vision_graph_disable", False):
                out_3d = torch.ops.rpu.qwen3vl_vision_forward(
                    handle, embed_i_3d, k_caches, v_caches, n_i,
                    1, planned_stage_descriptor,
                )
            else:
                sig = rpu_backend.graph.GraphSignature(
                    op_id="qwen3vl_vision",
                    shapes=[n_i, hidden_size],
                    dyn_dims=[
                        num_layers, deepstack_hash, int(_fused_merger),
                        *vision_plan.graph_key_words(),
                        *parent_graph_key_words,
                    ],
                    dtypes=[torch.float16],
                )
                with graph_cache.capture(sig):
                    out_3d = torch.ops.rpu.qwen3vl_vision_forward(
                        handle, embed_i_3d, k_caches, v_caches, n_i,
                        1, planned_stage_descriptor,
                    )

            resolved_chunk = int(
                torch.ops.rpu.qwen3vl_vision_get_resolved_chunk_size(handle)
            )
            if resolved_chunk != selected.stage_tuple.compute_chunk:
                raise RuntimeError(
                    "RhinoVLA vision dry/forward chunk drift: dry="
                    f"{selected.stage_tuple.compute_chunk}, forward="
                    f"{resolved_chunk}"
                )
            resolved_chunks.append(resolved_chunk)
            dispatched_descriptors.append(
                tuple(planned_stage_descriptor)
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

    # ----- Concatenate per-image outputs ----------------------------------
    if fused_batched:
        # T5b: cat of the per-view slices == the batched output; RhinoVLA ignores
        # last_hidden_state, so skip the per-view clones + cat entirely.
        last_hidden = batched_out  # [N_total, hidden] RPU (stable buffer; not consumed downstream)
    else:
        last_hidden = torch.cat(last_hidden_per_image, dim=0)  # [N_total, hidden] RPU

    if _fused_merger:
        # In-graph fused merger: the registered native post_fn already ran
        # the patch + N deepstack mergers into stable DDR slots. Pop all in one call; clone each
        # (stable slots are reused on the next same-shape forward — P4). pooler = [0], deepstack
        # = [1+k]. Skips the eager merger; RPU fp16 outputs feed the on-device prefix assembly.
        merged_all = torch.ops.rpu.qwen3vl_vision_pop_merged(handle)
        # P4 clone of the stable pop_merged slots, unless the caller guarantees the
        # outputs are consumed before the next same-shape forward overwrites the slots
        # (RhinoVLA: vision runs once per predict + prefix assembly copies them out
        # immediately) — then skip the clone (RPU_RHINOVLA_PREFIX_NO_CLONE).
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

    vars(self)["_rpu_last_execution_plan"] = _vision_execution_receipt(
        self, total_patches, vision_plans, resolved_chunks,
        dispatched_descriptors,
    )

    return BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=last_hidden,
        pooler_output=merged,
        deepstack_features=deepstack_features,
    )
