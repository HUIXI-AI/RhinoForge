"""Wall-OSS-0.5 Qwen2.5-VL ViT vision tower → RPU (P3).

The VLM backbone's vision tower is a stock Qwen2.5-VL ViT (32 blocks, hidden
1280, head_dim 80, RMSNorm, biased SwiGLU, 2D RoPE, window attention). This
adapter drives the forked C++ model `qwen25vl_vision_*` (raw-tensor style, like
`wall_oss/llm.py` — no HF module).

Weight layout (real ckpt): `visual.blocks.{L}.attn.qkv.{weight[3840,1280],bias}`
(fused, split q/k/v at 1280 each); `attn.proj.{weight[1280,1280],bias}`;
`mlp.gate_up_proj.{weight[6840,1280],bias}` (FUSED → split gate/up and
zero-pad each half to 3456 since 3420 is not swizzle-representable).  The v728
checkpoint stores the same tensors already zero-padded per half as 6912 rows;
both layouts are accepted.  `mlp.down_proj` similarly accepts K=3420 or the
pre-padded K=3456; `norm1/norm2.weight[1280]` (RMSNorm, no bias);
`patch_embed.proj.weight[1280,3,2,14,14]` (Conv3d, foldable to Linear, no bias);
`merger.{ln_q.weight, mlp.0[5120,5120]+b, mlp.2[2048,5120]+b}`.
Unquantized stock Qwen2.5-VL checkpoints may store separate `gate_proj` and
`up_proj`; these are combined only while loading, before the same padding.

P3a: full-attention only (MASK_NONE), no window reorder. P3b adds the window
mask + CPU reorder/reverse.

The default adapter runs patch_embed on RPU and folds the merger into the C++
graph; diagnostic fallbacks retain the CPU implementations. With
``w8a16=True``, the 32 RPU ViT blocks use int8 weights + fp16 per-row scales,
while patch_embed/merger values remain sourced from the fp16 checkpoint. RTC
mixed precision keeps block 17's down projection in paired-source FP16 for the
BF16 residual transition.
"""
from __future__ import annotations

import json
import os
import time
from typing import Sequence

import torch
import torch.nn.functional as F

import rpu_backend
from rpu_backend.runtime import rpu_env_bool
from rpu_backend.adapters.wall_oss._sorting import sorting_config_from_env
from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
from rpu_backend.runtime.execution_planner import GRAPH_COMPOSITE_CHILD
from rpu_backend.runtime.weights import (
    tp_col_swizzle_mc_weight,
    tp_row_swizzle_mc_weight,
)
from rpu_backend.adapters.wall_oss.llm import (
    _resolve_ckpt,
    _SafeTensorStore,
    _to_rpu_int8,
)
from rpu_backend.adapters.wall_oss.checkpoint_validation import (
    validate_w8_nvfp4_checkpoint_pair,
)

_VISION_CORES = 8           # MHA: attn_tp = min(8, num_heads=16) = 8 (all cores)
_MAX_BATCH_SEQ = 768        # batched-vision ceiling = 3×256 (one forward for the 3-camera
                            # case). Fits the full-VLA SPM after the vision temp-peak work
                            # (per-image sdpa_tmp + up→residual1 alias). Override via
                            # RPU_WALL_OSS_BATCH_MAX_SEQ.
_LAYER_GROUP_TOTAL_SEQ = 3 * 576  # Exact 3-camera experiment; default remains off.
_NATIVE_WINDOW_BOUNDARIES = (
    0, 64, 128, 192, 256, 320, 384, 448, 512, 528, 544, 560, 576,
)
_SMALL_NATIVE_WINDOW_BOUNDARIES = (0, 64, 128, 192, 256)
_INTER_PAD = 3456           # fp16: 3420 -> next multiple of 128
_INTER_PAD_W8A16 = 3584     # int8 row-swizzle needs K divisible by 32B*8 = 256
_PROCESS_EXPERIMENTAL_SCHEDULE: tuple[int, bool] | None = None
_RMSNORM_CAP_BASE = 1
_RMSNORM_CAP_FIXED32 = 1 | 4


def _experimental_schedule_from_env() -> tuple[int, bool]:
    raw_layer_group = os.environ.get(
        "RPU_WALL_OSS_VISION_LAYER_GROUP", "0"
    ).strip() or "0"
    if raw_layer_group not in ("0", "32"):
        raise ValueError(
            "RPU_WALL_OSS_VISION_LAYER_GROUP accepts only 0 (off) or 32 "
            f"(exact 3x576 single-capture experiment), got {raw_layer_group!r}"
        )
    layer_group = int(raw_layer_group)
    weight_outer = rpu_env_bool("RPU_WALL_OSS_VISION_WEIGHT_OUTER")
    if weight_outer and layer_group:
        raise ValueError(
            "RPU_WALL_OSS_VISION_WEIGHT_OUTER and "
            "RPU_WALL_OSS_VISION_LAYER_GROUP are mutually exclusive"
        )
    return layer_group, weight_outer


def _claim_experimental_schedule() -> tuple[int, bool]:
    """Validate and pin the process-wide experimental Vision schedule."""
    schedule = _experimental_schedule_from_env()
    global _PROCESS_EXPERIMENTAL_SCHEDULE
    if _PROCESS_EXPERIMENTAL_SCHEDULE is None:
        _PROCESS_EXPERIMENTAL_SCHEDULE = schedule
    if _PROCESS_EXPERIMENTAL_SCHEDULE != schedule:
        raise RuntimeError(
            "Wall-OSS Vision experimental scheduling is process-startup-only; "
            "use a fresh process after changing "
            "RPU_WALL_OSS_VISION_LAYER_GROUP or "
            "RPU_WALL_OSS_VISION_WEIGHT_OUTER"
        )
    return schedule


def _to_rpu_half(t: torch.Tensor) -> torch.Tensor:
    return t.to(dtype=torch.float16, device="rpu").contiguous()


def _pad_rows(w: torch.Tensor, target: int) -> torch.Tensor:
    """[N, K] → [target, K], zero-pad rows at the bottom."""
    return w if w.size(0) == target else F.pad(w, (0, 0, 0, target - w.size(0)))


def _pad_cols(w: torch.Tensor, target: int) -> torch.Tensor:
    """[N, K] → [N, target], zero-pad columns at the right."""
    return w if w.size(1) == target else F.pad(w, (0, target - w.size(1)))


def _pad_1d(b: torch.Tensor, target: int) -> torch.Tensor:
    return b if b.size(0) == target else F.pad(b, (0, target - b.size(0)))


def _split_gate_up(
    t: torch.Tensor,
    logical: int,
    padded: int,
    name: str,
    *,
    check_padding: bool = True,
):
    """Split raw or per-half padded gate/up tensors at their stored midpoint."""
    rows = t.size(0)
    stored = rows // 2
    if rows % 2 or not logical <= stored <= padded:
        raise ValueError(
            f"{name} rows must contain two halves padded from {logical} through "
            f"{padded}, got {rows}"
        )
    gate, up = t[:stored], t[stored:]
    if check_padding and stored > logical and (
        torch.count_nonzero(gate[logical:]).item()
        or torch.count_nonzero(up[logical:]).item()
    ):
        raise ValueError(f"{name} padded rows must be zero")
    return gate, up


def _load_vision_gate_up(store, prefix, hidden_size, intermediate_size, *, w8a16):
    """Read fused Wall weights or the original floating Qwen2.5-VL layout."""
    fused = f"{prefix}.gate_up_proj"
    try:
        weight = store.get_tensor(f"{fused}.weight")
    except KeyError:
        if w8a16:
            raise
        weights = [store.get_tensor(f"{prefix}.{name}_proj.weight")
                   for name in ("gate", "up")]
        biases = [store.get_tensor(f"{prefix}.{name}_proj.bias")
                  for name in ("gate", "up")]
        if (any(tuple(value.shape) != (intermediate_size, hidden_size)
                or not value.is_floating_point() for value in weights)
                or weights[0].dtype != weights[1].dtype
                or any(tuple(value.shape) != (intermediate_size,)
                       or not value.is_floating_point() for value in biases)):
            raise ValueError(f"{prefix} has invalid separate gate/up weights or biases")
        return torch.cat(weights).contiguous(), torch.cat(biases).float()
    return weight.contiguous(), store.get_tensor(f"{fused}.bias").float()


def _rmsnorm_cpu(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (w.float() * (xf * torch.rsqrt(var + eps)))


def build_vision_rope_tables(
    head_dim: int,
    max_hw: int,
    *,
    device: str | torch.device = "rpu",
    dtype: torch.dtype = torch.float16,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FreqCos/FreqSin `[max_hw, head_dim/4]` for the 2D RoPE kernel.

    Wall-OSS uses the same Qwen2.5/3-VL vision rotary layout: each attention
    head splits the rotary half into row and col axes, so the lookup table has
    `head_dim / 4` lanes per axis.
    """
    if head_dim <= 0 or (head_dim % 4) != 0:
        raise ValueError(
            f"build_vision_rope_tables: head_dim must be positive multiple of 4, "
            f"got {head_dim}")
    if max_hw <= 0:
        raise ValueError(f"build_vision_rope_tables: max_hw must be positive, got {max_hw}")

    half_axis = head_dim // 4
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float64)
                  / (head_dim // 2))
    )
    positions = torch.arange(max_hw, dtype=torch.float64)
    freqs = positions[:, None] * inv_freq[None, :]
    if tuple(freqs.shape) != (max_hw, half_axis):
        raise RuntimeError(
            "Wall-OSS vision RoPE frequency shape mismatch: "
            f"got {tuple(freqs.shape)}, expected {(max_hw, half_axis)}"
        )

    cos = freqs.cos().to(dtype=dtype).contiguous()
    sin = freqs.sin().to(dtype=dtype).contiguous()
    return cos.to(device=device), sin.to(device=device)


def _ragged_mask_plan_identity(mask, seq, image_batch_count, valid_patches_per_image):
    target = ((seq == 672 and image_batch_count == 2 and valid_patches_per_image == 324)
              or (seq == 324 and image_batch_count == 1 and valid_patches_per_image == 0))
    rows = seq // image_batch_count if image_batch_count else 0
    if (not target or mask is None or mask.device.type != "cpu"
            or mask.dtype != torch.float16 or mask.layout != torch.strided
            or not mask.is_contiguous() or tuple(mask.shape) != (rows, rows)):
        return None
    try:
        version = int(mask._version)
    except RuntimeError:  # Inference tensors have no mutation counter: dense only.
        return None
    return ("wall_base_ragged_mask_v1", int(mask._cdata), int(mask.data_ptr()), version,
            rows, int(mask.stride(0)), int(mask.stride(1)), int(mask.storage_offset()))


def _compute_vision_position_idx_cpu(
    grid_thw_cpu: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Build `[num_patches, 2]` int16 row/col position ids on CPU."""
    if grid_thw_cpu.dim() != 2 or grid_thw_cpu.size(-1) != 3:
        raise ValueError(
            f"_compute_vision_position_idx_cpu: grid_thw must be [n, 3], "
            f"got {tuple(grid_thw_cpu.shape)}")
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
            "exceeds int16 range")
    return pos_ids.to(torch.int16).contiguous()


def _get_window_index(grid_thw, window_size, spatial_merge_size, patch_size):
    """CPU port of HF `Qwen2_5_VisionTransformer.get_window_index`
    (modeling_qwen2_5_vl.py:404-444). Returns (window_index, cu_window_seqlens,
    reverse_index) — all on CPU. window_index / reverse_index are merge-unit
    length (seq // spatial_merge_unit); cu_window_seqlens is patch-unit."""
    vit_win = window_size // spatial_merge_size // patch_size       # 4 merge-units
    smu = spatial_merge_size * spatial_merge_size                    # 4
    window_index, cu = [], [0]
    wid = 0
    for gt, gh, gw in grid_thw.tolist():
        lh, lw = gh // spatial_merge_size, gw // spatial_merge_size
        index = torch.arange(gt * lh * lw).reshape(gt, lh, lw)
        # HF verbatim: when lh divisible by vit_win this pads a full empty window
        # (all -100 → filtered out; its 0-length cu entry collapses below).
        pad_h = vit_win - lh % vit_win
        pad_w = vit_win - lw % vit_win
        nwh, nww = (lh + pad_h) // vit_win, (lw + pad_w) // vit_win
        ip = F.pad(index, (0, pad_w, 0, pad_h), value=-100)
        ip = ip.reshape(gt, nwh, vit_win, nww, vit_win)
        ip = ip.permute(0, 1, 3, 2, 4).reshape(gt, nwh * nww, vit_win, vit_win)
        seqlens = (ip != -100).sum([2, 3]).reshape(-1)
        ip = ip.reshape(-1)
        window_index.append(ip[ip != -100] + wid)
        cu_tmp = seqlens.cumsum(0) * smu + cu[-1]
        cu.extend(cu_tmp.tolist())
        wid += gt * lh * lw
    window_index = torch.cat(window_index, dim=0)
    cu = torch.unique_consecutive(torch.tensor(cu, dtype=torch.long))
    reverse_index = torch.argsort(window_index)
    return window_index, cu, reverse_index


def _block_diag_mask(cu_window_seqlens: torch.Tensor, seq: int) -> torch.Tensor:
    """Dense [seq, seq] fp16 additive mask: 0 inside each window block (attend),
    real -inf elsewhere. Block boundaries = cu_window_seqlens (patch units)."""
    mask = torch.full((seq, seq), float("-inf"), dtype=torch.float16)
    cu = cu_window_seqlens.tolist()
    for a, b in zip(cu[:-1], cu[1:]):
        if b > a:
            mask[a:b, a:b] = 0.0
    return mask


def _vision_graph_dyn_dims(
    base: Sequence[int], window_boundaries: Sequence[int] | None
) -> list[int]:
    """Keep the ordinary Wall key unchanged; controlled window mode keys every boundary."""
    dims = list(base)
    if window_boundaries is not None:
        dims.extend((1, *window_boundaries))
    return dims


def _pack_multi_pos(grid_group: torch.Tensor, sms: int) -> torch.Tensor:
    """Concatenate per-image 2D-RoPE position ids → ``[sum patches, 2]`` int16.

    Each image's row/col coordinates restart at its own grid (vision RoPE is
    content-position, applied with pos_offset=0), so a plain concat is correct.
    """
    return torch.cat(
        [_compute_vision_position_idx_cpu(grid_group[i:i + 1], sms)
         for i in range(grid_group.size(0))],
        dim=0,
    )


def _pack_multi_image(grid_group: torch.Tensor, window_size: int, sms: int,
                      patch_size: int):
    """Multi-image window reorder + shared mask (all images SAME size).

    Returns ``(window_index_multi, reverse_indices, mask, cu_window_seqlens)``:
    - ``window_index_multi`` — merge-unit reorder, each image's indices offset by
      ``i * (n_i // smu)`` so the concatenated per-image blocks stay disjoint.
    - ``reverse_indices`` — list of per-image merge-unit argsorts (applied to each
      image's merged-output slice AFTER the merger).
    - ``mask`` — a single ``[n_i, n_i]`` block-diagonal window MASK_2D, reused for
      every per-image window-SDPA call (image isolation comes from the per-image
      KV-cache slice, not the mask).
    """
    smu = sms * sms
    wins, reverses = [], []
    off_mu = 0
    for i in range(grid_group.size(0)):
        win, _cu, rev = _get_window_index(grid_group[i:i + 1], window_size, sms, patch_size)
        wins.append(win + off_mu)
        reverses.append(rev)
        off_mu += win.numel()
    window_index_multi = torch.cat(wins, dim=0)
    n_i = int(grid_group[0].prod().item())
    _, cu0, _ = _get_window_index(grid_group[0:1], window_size, sms, patch_size)
    mask = _block_diag_mask(cu0, n_i)
    return window_index_multi, reverses, mask, cu0


def _pair_252_layout(grid: torch.Tensor):
    """Two actual 18x18 images in independent 336-row physical spans.

    Each image owns 81 real merge units followed by three dummy units. Both
    masks isolate the dummy keys and give dummy queries a finite local block.
    The final reverse index simultaneously restores raster order and drops pads.
    """
    if tuple(grid.shape) != (2, 3) or not bool(
            (grid == grid.new_tensor([1, 18, 18])).all()):
        raise ValueError("paired Vision layout requires two actual 18x18 grids")
    one = grid[:1]
    win, cu, rev = _get_window_index(one, 112, 2, 14)
    pos = _compute_vision_position_idx_cpu(one, 2).reshape(81, 4, 2)[win]
    padded_pos = torch.zeros(336, 2, dtype=torch.int16)
    padded_pos[:324] = pos.reshape(324, 2)
    padded_win = torch.cat((win, torch.arange(81, 84)))
    window = torch.full((336, 336), -float("inf"), dtype=torch.float16)
    window[:324, :324] = _block_diag_mask(cu, 324)
    window[324:, 324:] = 0
    full = torch.full_like(window, -float("inf"))
    full[:324, :324] = 0
    full[324:, 324:] = 0
    return (padded_pos.repeat(2, 1),
            torch.cat((padded_win, padded_win + 84)),
            torch.cat((rev, rev + 84)), window, full)


def _destroy_qwen25vl_vision_handle(h):
    """Raw destroy; graph retirement belongs to the actual component owner."""
    torch.ops.rpu.qwen25vl_vision_destroy(h)


def _configure_qwen25vl_vision_handle(
    *, set_weights, weight_args, freq_cos, freq_sin, merger_args, max_seq_len,
    execution_routes=(False, False, False, False, False, _RMSNORM_CAP_BASE),
    chunk_size=0,
    per_window_sdpa=False, _owned_vision=None, sorting=None, linear_acc32=False,
):
    """Configure a pending handle; owned diagnostics retire via their parent."""
    from rpu_backend.api._execution import _require_execution_process_safe
    _require_execution_process_safe()
    handle = torch.ops.rpu.qwen25vl_vision_create()
    if _owned_vision is not None:
        _owned_vision._handle = handle
    try:
        if sorting is not None:
            torch.ops.rpu.qwen25vl_vision_set_wall_sorting_routes(
                handle, *sorting.vision_args()
            )
        torch.ops.rpu.qwen25vl_vision_set_execution_routes(
            handle, *execution_routes
        )
        if linear_acc32:
            torch.ops.rpu.qwen25vl_vision_set_linear_acc32(handle, True)
        set_weights(handle, *weight_args)
        torch.ops.rpu.qwen25vl_vision_set_chunk_envelope(handle, int(max_seq_len), 0)
        torch.ops.rpu.qwen25vl_vision_set_chunk_size(handle, chunk_size)
        if per_window_sdpa:
            torch.ops.rpu.qwen25vl_vision_set_per_window_sdpa(handle, True)
        torch.ops.rpu.qwen25vl_vision_set_rope(handle, freq_cos, freq_sin)
        keepalive = torch.ops.rpu.qwen25vl_vision_position_idx_keepalive(handle)
        torch.ops.rpu.qwen25vl_vision_set_merger_weights(
            handle, *merger_args
        )
        return handle, keepalive
    except BaseException as error:
        if _owned_vision is None:
            from rpu_backend.adapters.wall_oss._retirement import _cleanup_build_failure
            _cleanup_build_failure(error, (handle, weight_args, freq_cos, freq_sin, merger_args),
                                   lambda: _destroy_qwen25vl_vision_handle(handle))
        raise


class WallOssVision:
    """Qwen2.5-VL ViT vision tower on RPU.

    Construct via :func:`build_wall_oss_vision`. Call :meth:`forward` with CPU
    `pixel_values [num_patches, 1176]` + `grid_thw [1, 3]`; returns the merged
    vision features `[num_patches/4, 2048]` on CPU (fp32).
    """

    def __init__(self, *, handle, cache, graph_cache, keepalive, patch_w, merger,
                 hidden_size, num_layers, spatial_merge_size, window,
                 window_size, patch_size, experimental_schedule=None,
                 execution_chunk_size="auto", per_window_sdpa=False,
                 fused_merger=False, mixed_precision=False, linear_acc32=False):
        self._gc_retirement_enabled = False
        schedule = (
            _claim_experimental_schedule()
            if experimental_schedule is None else experimental_schedule
        )
        self._handle = handle
        self.cache = cache
        self._graph_cache = graph_cache
        self._keepalive = keepalive
        self._patch_w = patch_w               # [hidden, 1176] CPU fp32
        self._merger = merger                 # callable: [seq,1280]fp32 -> [seq/4,2048]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._sms = spatial_merge_size
        self._window = window                 # P3b: window attention on/off
        self._window_size = window_size
        self._patch_size = patch_size
        self._execution_chunk_size = execution_chunk_size
        self._linear_acc32 = linear_acc32
        self._per_window_sdpa = per_window_sdpa
        self._batch_vision = rpu_env_bool(
            "RPU_WALL_OSS_BATCH_VISION", default=True
        )
        try:
            self._batch_max_seq = int(os.environ.get(
                "RPU_WALL_OSS_BATCH_MAX_SEQ", str(_MAX_BATCH_SEQ)
            ))
        except ValueError as exc:
            raise ValueError(
                "RPU_WALL_OSS_BATCH_MAX_SEQ must be a positive integer"
            ) from exc
        if self._batch_max_seq <= 0:
            raise ValueError(
                "RPU_WALL_OSS_BATCH_MAX_SEQ must be a positive integer"
            )
        # Keep the merged vision tokens on the RPU device (merger RMSNorm on-device +
        # window reverse-reorder via the gather op) so the VLA's on-device embed
        # assembly needs no merged readback/upload. DEFAULT OFF: this legacy eager
        # path is superseded by the default fused-merger post_fn. Eager RMSNorm is
        # correctness-safe as of 2026-08-01 but uses a host fallback, so this switch
        # is diagnostic rather than a performance alternative.
        self._device_merged = rpu_env_bool("RPU_WALL_OSS_VISION_DEVICE_MERGED")
        # The builder resolves these once and passes the same cold values to
        # this Python owner and the native handle.
        self._fused_merger = bool(fused_merger)
        self._mixed_precision = bool(mixed_precision)
        if self._mixed_precision and not self._fused_merger:
            raise ValueError(
                "RPU_WALL_OSS_VISION_MIXED_PRECISION=1 requires "
                "RPU_WALL_OSS_VISION_FUSED_MERGER=1"
            )
        if self._mixed_precision and schedule[0] != 0:
            raise ValueError(
                "Wall-OSS Vision mixed precision is incompatible with "
                "RPU_WALL_OSS_VISION_LAYER_GROUP=32"
            )
        # On-device patch_embed: run the folded Conv3d→Linear as an eager rpu linear
        # instead of the ~5.6 ms/frame CPU fp32 GEMM (torch profile aten::mm). _patch_w_rpu
        # (padded+col-swizzled weight) is set by build_wall_oss_vision; None disables.
        # RPU_WALL_OSS_DEVICE_PATCH_EMBED=0 forces the CPU GEMM.
        self._device_patch_embed = rpu_env_bool(
            "RPU_WALL_OSS_DEVICE_PATCH_EMBED", default=True)
        self._patch_w_rpu = None
        self._pe_k = None
        # Normalize-fold patch_embed (set by enable_normalize_fold): the per-channel
        # image normalize (rescale/mean/std) is baked into the patch_embed weight so the
        # host preproc can patchify RAW uint8 (no fp32 cast, no normalize passes — the
        # imgproc host block) and the device GEMM does normalize+project in one pass. The
        # bias (-W·mean/std) rides a homogeneous activation column (input col K = 1.0,
        # weight col K = bias) so it stays a single GEMM. dtype-dispatched: uint8 input →
        # folded weight; fp32 input (HF-processor fallback) → the plain _patch_w_rpu.
        self._patch_w_fold_rpu = None    # [hidden, _pe_k] col-swizzled, normalize+bias baked
        self._patch_w_fold_cpu = None    # [hidden, _pe_k] CPU fp32 (device-patch-embed off)
        # Memoized per-grid CPU window/position layout (frame-invariant for a fixed
        # camera). RPU_WALL_OSS_HOST_CACHE=0 disables (bit-exact A/B / escape hatch).
        self._layout_cache = {}          # single-image (_forward_one / _window_layout)
        self._group_layout_cache = {}    # batched multi-image (_forward_group / _group_layout)
        self._index_rpu_cache = {}       # content-checked patch/reverse indices
        self._position_keepalive_snapshot = None
        self._pe_buf_cache = {}          # persistent [seq, _pe_k] fp16 patch_embed input buffers
                                         # (bias/pad cols set once; data region refilled per frame)
        # HOST_CACHE=0 still keeps the immediately previous CPU window mask alive while
        # allocating the next one. This prevents allocator address reuse from fooling
        # the C++ source-pointer memo into skipping an upload for changed contents.
        self._last_window_mask_ref = None
        # Fused embed-assembly: fold the window-REVERSE into the runtime's assemble op
        # (so the two gather_embedding — vision reverse + token gather — collapse and the
        # merged DDR round-trip is gone). When on, forward() returns the RAW window-order
        # merged for a single-subgroup forward and stashes `_pending_rev = (rev_idx_rpu,
        # [merged_tokens_per_image])`; the runtime reads it and calls
        # assemble_inputs_embeds_rev. Multi-subgroup (mixed-size) forwards fall back (no
        # fuse). Only meaningful with the fused merger. RPU_WALL_OSS_FUSED_ASSEMBLE=1.
        self._fused_assemble = rpu_env_bool("RPU_WALL_OSS_FUSED_ASSEMBLE")
        self._pending_rev = None         # (rev_idx_rpu int32, [mu_per_image]) or None
        self._pair_252 = rpu_env_bool("RPU_WALL_OSS_VISION_PAIR_252")
        self._host_cache = rpu_env_bool("RPU_WALL_OSS_HOST_CACHE", default=True)
        # Experimental exact-shape scheduling A/B. Treat this as an
        # instance/startup setting: the C++ side snapshots the same env value.
        self._layer_group, self._weight_outer = schedule
        # The native-window SDPA route changes FP16 reduction order enough to
        # miss the controlled W8 action floor. Keep it for record-only perf A/B,
        # but use the established dense-mask route unless explicitly requested.
        self._native_window_sdpa = rpu_env_bool(
            "RPU_WALL_OSS_VISION_NATIVE_WINDOW_SDPA")
        self._closed = False
        # Diagnostic children already belong to their parent's graph-aware close.
        self._gc_retirement_enabled = getattr(self, "_retirement_session", None) is None

    def _retire_native(self):
        from rpu_backend.adapters.wall_oss._retirement import _retire_native
        _retire_native(self, _destroy_qwen25vl_vision_handle)

    def close(self, *, _graphs_retired=False) -> None:
        """Release graph/native resources. Safe to call more than once."""
        if self._closed:
            return
        session = getattr(self, "_retirement_session", None)
        if session is not None:
            from rpu_backend.api._execution import ExecutionSession

            parent = getattr(session, "_owner", None)
            if (not _graphs_retired or not isinstance(session, ExecutionSession)
                    or getattr(parent, "_execution_session", None) is not session
                    or not session._shutting_down
                    or session._active or getattr(parent, "_diagnostic_vision", None) is not self):
                raise RuntimeError("owned Wall-OSS Vision must close through its parent Session shutdown")
            if getattr(self, "_handle_finalizer", None) is not None:
                raise RuntimeError("owned Wall-OSS Vision cannot have an independent raw finalizer")
            if self._handle is not None:
                torch.ops.rpu.qwen25vl_vision_destroy(self._handle)
                self._handle = None
            self._closed = True
            return
        if _graphs_retired:
            raise RuntimeError("precleared Wall-OSS Vision requires its actual retirement parent")
        from rpu_backend.adapters.wall_oss._retirement import _close
        _close(self, (self,))

    def __del__(self):
        from rpu_backend.adapters.wall_oss._retirement import _gc_close
        _gc_close(self)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Return one merged tensor in image order; subgroup outputs stay fresh."""
        return self._forward(pixel_values, grid_thw, return_parts=False)

    @torch.no_grad()
    def _forward_parts(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        """Keep separate subgroup outputs for the internal device assembler."""
        return self._forward(pixel_values, grid_thw, return_parts=True)

    def _forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, *,
                 return_parts: bool):
        """pixel_values [sum patches, 1176] + grid_thw [N, 3] → merged [sum patches/4, 2048].

        ③ multi-image: Qwen2.5-VL ViT attention is block-diagonal per image (image i
        NEVER attends image j). Consecutive equal-size images are batched into ONE
        packed forward (≤3 same-size images, ``_forward_group``) to amortize dispatch +
        weight-DMA; a size change or the cap starts a new group, and odd singletons fall
        back to ``_forward_one``. The generic route keeps attention image-local
        with per-image SDPA. The exact RTC weight-outer route admits only 3x576
        and uses one dedicated B=3 by-MHA launch per layer with SPM-resident K/V.
        """
        from rpu_backend.adapters.wall_oss._retirement import _require_open
        _require_open(self)
        if not isinstance(grid_thw, torch.Tensor):
            raise TypeError(
                "WallOssVision.forward grid_thw must be a torch.Tensor, got "
                f"{type(grid_thw).__name__}"
            )
        if (
            grid_thw.dim() != 2
            or grid_thw.size(0) < 1
            or grid_thw.size(1) != 3
        ):
            raise ValueError(
                "WallOssVision.forward grid_thw must be [N, 3], got "
                f"{tuple(grid_thw.shape)}"
            )
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(
                "WallOssVision.forward pixel_values must be a torch.Tensor, got "
                f"{type(pixel_values).__name__}"
            )
        grid_cpu = grid_thw.detach().cpu()
        seq = int(grid_cpu.prod(-1).sum().item())
        pixel_rows = int(pixel_values.size(0)) if pixel_values.dim() >= 1 else None
        if pixel_rows != seq:
            raise ValueError(
                f"pixel_values rows {pixel_rows} != grid patches {seq}"
            )
        self._pending_rev = None
        self._last_forward_plans = []
        counts = [int(grid_cpu[i].prod().item()) for i in range(grid_cpu.size(0))]
        N = grid_cpu.size(0)
        batch_on = self._batch_vision
        layer_group = self._layer_group
        weight_outer = self._weight_outer
        mixed_precision = getattr(self, "_mixed_precision", False)
        per_window_sdpa = getattr(self, "_per_window_sdpa", False)
        if mixed_precision and any(n != 576 for n in counts):
            raise ValueError(
                "Wall-OSS Vision mixed precision is validated only for "
                f"576 patches per image, got {counts}"
            )
        if per_window_sdpa and (
            not self._window
            or N != 1
            or mixed_precision
            or layer_group != 0
            or weight_outer
        ):
            raise ValueError(
                "per-window Vision SDPA is restricted to the controlled "
                "single-image FP16 normal schedule"
            )
        if layer_group == 32:
            if not batch_on:
                raise ValueError(
                    "RPU_WALL_OSS_VISION_LAYER_GROUP=32 requires "
                    "RPU_WALL_OSS_BATCH_VISION=1"
                )
            if N != 3 or counts != [576, 576, 576]:
                raise ValueError(
                    "experimental vision layer grouping is validated only for "
                    f"3x576 patches, got {counts}"
                )
        if weight_outer:
            if not batch_on:
                raise ValueError(
                    "RPU_WALL_OSS_VISION_WEIGHT_OUTER requires "
                    "RPU_WALL_OSS_BATCH_VISION=1"
                )
            if N != 3 or counts != [576, 576, 576]:
                raise ValueError(
                    "vision weight-outer mode is validated only for 3x576 patches, "
                    f"got {counts}"
                )
        if N == 1:
            self._validate_execution_chunks(
                (self._native_chunk_size(grid_cpu),)
            )
            return self._forward_one(
                pixel_values, grid_cpu, fused=self._fused_assemble)

        offs = [0]
        for n in counts:
            offs.append(offs[-1] + n)
        i = 0
        max_seq = self._batch_max_seq
        if mixed_precision and not weight_outer:
            # The normal C++ path is fixed at M=576. Keep multiple cameras as
            # separate GraphCache signatures unless exact 3x576 weight-outer is
            # selected.
            max_seq = min(max_seq, 576)
        if layer_group == 32 or weight_outer:
            max_seq = max(max_seq, _LAYER_GROUP_TOTAL_SEQ)
        if (getattr(self, "_pair_252", False)
                and tuple(grid_cpu.shape) in ((2, 3), (3, 3))
                and bool((grid_cpu == grid_cpu.new_tensor([1, 18, 18])).all())
                and not torch.rpu.get_debug_export()):
            if not batch_on or max_seq < 672:
                raise ValueError("paired Vision requires batch mode and max_seq >= 672")
            self._validate_execution_chunks(tuple([672] if N == 2 else [672, 336]))
            pair = self._forward_pair_252(pixel_values[:648], grid_cpu[:2])
            if N == 2:
                return pair
            last = self._forward_one(pixel_values[648:], grid_cpu[2:])
            return (pair, last) if return_parts else torch.cat((pair, last), dim=0)
        subgroups = []   # (sub_pixel_values, grid_slice, is_group)
        while i < N:
            n_i = counts[i]
            # Batchable only when per-image patches keep the minibatch valid
            # (multiple of 128, ≥256 ⇒ per_image_ctx/tile_m≥2) AND g·n_i ≤
            # max_seq (E2E-SPM ceiling; RPU_WALL_OSS_BATCH_MAX_SEQ to override);
            # else run the image solo.
            packed_ok = n_i >= 256 and n_i % 128 == 0
            exact_3img_ok = (layer_group == 32 or weight_outer) and n_i == 576
            cap = (min(3, max_seq // n_i)
                   if (batch_on and (packed_ok or exact_3img_ok)) else 1)
            j = i + 1
            while j < N and torch.equal(grid_cpu[j], grid_cpu[i]) and (j - i) < cap:  # full grid, not just patch count (same count / different H·W must split)
                j += 1
            subgroups.append((pixel_values[offs[i]:offs[j]], grid_cpu[i:j], (j - i) > 1))
            i = j
        self._validate_execution_chunks(
            self._native_chunk_size(group) for _, group, _ in subgroups
        )
        # Fused embed-assembly needs a single window-order output (the runtime reverses
        # via assemble_inputs_embeds_rev). Only when the whole forward is ONE subgroup
        # (1 cam, or N same-size cams ≤ cap) — the cat'd mixed-size case keeps the reverse.
        if (self._fused_assemble and len(subgroups) == 1
                and layer_group == 0 and not weight_outer):
            sub, gslice, is_group = subgroups[0]
            return (self._forward_group(sub, gslice, fused=True) if is_group
                    else self._forward_one(sub, gslice, fused=True))
        if len(subgroups) == 1:
            sub, gslice, is_group = subgroups[0]
            return (self._forward_group(sub, gslice) if is_group
                    else self._forward_one(sub, gslice))
        # Upload the same padded pixels once for separate camera graphs. Row
        # views preserve their original Linear inputs and stay alive until all
        # synchronous subgroup forwards have consumed the request-owned slab.
        patch_rows = [0]
        for sub, _, _ in subgroups:
            patch_rows.append(patch_rows[-1] + sub.size(0))
        prepared_pixels = None
        patch_weight = (self._patch_w_fold_rpu if pixel_values.dtype == torch.uint8
                        else self._patch_w_rpu)
        if (pixel_values.device.type == "cpu" and self._device_patch_embed
                and patch_weight is not None
                and all(row * self._pe_k * 2 % 256 == 0 for row in patch_rows[:-1])):
            prepared_pixels = self._prepare_patch_pixels(pixel_values).to(device="rpu")
        merged = []
        for index, (sub, group, is_group) in enumerate(subgroups):
            forward = self._forward_group if is_group else self._forward_one
            if prepared_pixels is None:
                merged.append(forward(sub, group))
            else:
                merged.append(forward(
                    sub, group, prepared_pixels=prepared_pixels.narrow(
                        0, patch_rows[index], sub.size(0))))
        return tuple(merged) if return_parts else torch.cat(merged, dim=0)

    def _native_chunk_size(self, grid_group: torch.Tensor) -> int:
        """Return the one chunk size the native forward will actually use."""
        seq = int(grid_group.prod(-1).sum().item())
        if (
            grid_group.size(0) > 1
            and (self._layer_group == 32 or self._weight_outer)
            and seq == _LAYER_GROUP_TOTAL_SEQ
        ):
            seq = int(grid_group[0].prod().item())
        return ((seq + 15) // 16) * 16

    def _plan_native_execution(
        self, seq: int, image_batch_count: int,
        window_mask: torch.Tensor | None,
        cu_window_seqlens: torch.Tensor | None,
        *, valid_patches_per_image: int = 0,
    ):
        requested = getattr(self, "_rpu_execution", {}).get("vision", {}).get(
            "chunk_size", getattr(self, "_execution_chunk_size", "auto"))
        # Read this invocation's CPU boundary contents once. The native op still
        # receives the tensor; planner and Graph keys share the same snapshot.
        window_boundaries = (
            None if cu_window_seqlens is None
            else tuple(int(row) for row in cu_window_seqlens.tolist())
        )
        plan_box = {}
        logical_len = (int(valid_patches_per_image) * int(image_batch_count)
                       if valid_patches_per_image else int(seq))
        extra_native_args = (int(valid_patches_per_image),) if valid_patches_per_image else ()
        mask_identity = _ragged_mask_plan_identity(
            window_mask, int(seq), int(image_batch_count), int(valid_patches_per_image))
        native_args = ((int(valid_patches_per_image), window_mask)
                       if mask_identity is not None else extra_native_args)
        _, chunk_size = plan_bounded_prefill_execution(
            logical_len, int(seq), 0,
            execution_owner=self,
            execution_stage="vision",
            execution_native=("qwen25vl_vision", int(self._handle)),
            position=0,
            alignment=1,
            padding_rows=int(seq) - logical_len,
            exact_chunk_size=(
                int(requested) if requested != "auto" else None
            ),
            resolve_stage_domain=lambda length: (
                torch.ops.rpu.qwen25vl_vision_resolve_stage_domain(
                    self._handle, int(length), int(image_batch_count),
                    window_mask is not None, cu_window_seqlens,
                    int(requested) if requested != "auto" else 0,
                    *native_args,
                )
            ),
            request_id="qwen25vl:vision_encoder:vision",
            plan_result_sink=lambda result: plan_box.__setitem__(
                "result", result),
            graph_mode=GRAPH_COMPOSITE_CHILD,
            queue_owner_id=int(self._handle),
            physical_metadata=(
                ("component:vision_encoder", 1),
                ("execution_generation", int(getattr(
                    self, "_fmb_execution_generation", 0))),
                ("semantic_span_count", int(image_batch_count)),
                ("semantic_span_rows", int(seq) // int(image_batch_count)),
                ("window_mask_present", int(window_mask is not None)),
            ) + ((("valid_patches_per_image", int(valid_patches_per_image)),)
                 if valid_patches_per_image else ()),
            plan_signature=(int(image_batch_count), window_mask is not None, window_boundaries)
                + extra_native_args + (() if mask_identity is None else (mask_identity,)),
            graph_cache=self._graph_cache,
        )
        plan = plan_box["result"]
        descriptor = plan.selected.stage_tuple.physical_descriptor
        if not descriptor:
            raise RuntimeError(
                "Qwen2.5-VL Vision A6 winner has no native descriptor")
        return int(chunk_size), plan, descriptor, window_boundaries

    def _record_native_execution(
        self, plan, *, seq: int, image_batch_count: int, chunk_size: int,
        valid_patches_per_image: int = 0,
    ) -> None:
        selected = plan.selected
        if (
            selected is None
            or int(selected.execution_len) != int(seq)
            or int(selected.stage_tuple.compute_chunk) != int(chunk_size)
        ):
            raise RuntimeError("Qwen2.5-VL Vision dry/forward plan disagreement")
        self._rpu_last_execution_plan = {
            "component": getattr(
                self, "_fmb_execution_component_id", "vision_encoder"),
            "stage": "vision",
            "generation": int(getattr(
                self, "_fmb_execution_generation", 0)),
            "logical_len": (int(valid_patches_per_image) * int(image_batch_count)
                            if valid_patches_per_image else int(seq)),
            "execution_len": int(selected.execution_len),
            "chunk_size": int(chunk_size),
            "padding_rows": int(selected.padding_rows),
            "image_batch_count": int(image_batch_count),
            "graph_mode": plan.graph_mode,
            "semantic_span_owner": "INPUT",
            "semantic_span_count": int(image_batch_count),
            "authority": "NATIVE_A6_STAGE_DESCRIPTOR",
            "selection_scope": plan.selection_scope,
            "physical_plan_digest": plan.physical_plan_digest,
            "plan_digest": plan.plan_digest,
            "descriptor_words": len(
                selected.stage_tuple.physical_descriptor),
            "physical_descriptor": tuple(
                selected.stage_tuple.physical_descriptor),
            "dry_forward_agreement": True,
        }
        if getattr(self, "_pair_252", False):
            self._last_forward_plans.append(dict(self._rpu_last_execution_plan))

    def _validate_execution_chunks(self, chunks) -> None:
        requested = getattr(self, "_rpu_execution", {}).get("vision", {}).get(
            "chunk_size", getattr(self, "_execution_chunk_size", "auto"))
        if requested == "auto":
            return
        native = tuple(int(chunk) for chunk in chunks)
        if any(chunk != int(requested) for chunk in native):
            raise ValueError(
                "Wall-OSS vision chunk_size must equal every native Vision "
                f"forward chunk for this request: requested={requested}, "
                f"native={native}"
            )

    def _select_native_window_boundaries(
        self, grouped_chunks: bool, boundaries: torch.Tensor,
        grid_group: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self._native_window_sdpa:
            return None
        actual = (() if boundaries is None else
                  tuple(int(value) for value in boundaries.tolist()))
        exact = grouped_chunks and actual == _NATIVE_WINDOW_BOUNDARIES
        small = (
            not grouped_chunks and grid_group is not None
            and tuple(grid_group.shape) in ((2, 3), (3, 3))
            and bool((grid_group == grid_group.new_tensor([1, 16, 16])).all())
            and self._sms == 2 and self._window_size == 112 and self._patch_size == 14
            and actual == _SMALL_NATIVE_WINDOW_BOUNDARIES
        )
        if not (exact or small):
            raise ValueError(
                "native-window SDPA requires exact 3x576 canonical boundaries "
                "or two/three 16x16 grids with four 64-patch windows each; "
                f"got grouped_chunks={grouped_chunks}, boundaries={actual}"
            )
        return boundaries

    def _window_layout(self, grid_cpu: torch.Tensor, seq: int):
        """Memoized CPU window/position layout `(pos_idx, window_index, reverse_index,
        window_mask)`. Pure function of the grid + (instance-fixed) merge/window config,
        so a fixed-camera rollout reuses one entry and skips the ~1 ms/frame recompute.
        All four are CPU tensors (the pos_idx→keepalive copy checks content) and
        are only read downstream, so sharing them across frames is safe."""
        key = (int(seq),) + tuple(grid_cpu.flatten().tolist())
        if self._host_cache:
            hit = self._layout_cache.get(key)
            if hit is not None:
                return hit
        smu = self._sms * self._sms
        pos_idx = _compute_vision_position_idx_cpu(grid_cpu, self._sms)   # [seq, 2] int16
        if pos_idx.size(0) != seq:
            raise RuntimeError(
                f"vision position rows {pos_idx.size(0)} != sequence {seq}"
            )
        window_index = reverse_index = window_mask = cu_window_seqlens = None
        if self._window:
            # P3b: reorder by merge-unit window_index ON CPU.
            window_index, cu_window_seqlens, reverse_index = _get_window_index(
                grid_cpu, self._window_size, self._sms, self._patch_size)
            pos_idx = pos_idx.reshape(seq // smu, smu, 2)[window_index].reshape(seq, 2)
            # Dense block-diagonal additive mask (CPU fp16). Passing it CPU-side
            # avoids the RPU→CPU readback staleness in sdpa_prepare_mask.
            if not self._per_window_sdpa:
                window_mask = _block_diag_mask(cu_window_seqlens, seq)
        out = (pos_idx, window_index, reverse_index, window_mask)
        if self._per_window_sdpa:
            out += (cu_window_seqlens,)
        if self._host_cache:
            self._layout_cache[key] = out
        else:
            self._last_window_mask_ref = window_mask
        return out

    def _group_layout(self, grid_group: torch.Tensor, seq: int, smu: int):
        """Memoized batched (multi-image) window/position layout `(pos_idx,
        window_index_multi, reverses, window_mask, cu_window_seqlens)` — the
        group analog of _window_layout.
        Pure function of grid_group + (instance-fixed) window/merge/patch config, so a
        fixed-camera rollout reuses one entry and skips the ~2.6 ms/frame _pack_multi_pos
        (~0.8) + _pack_multi_image (~1.8, incl the [n_i,n_i] block-diag mask). All CPU
        tensors, read-only downstream → safe to share. RPU_WALL_OSS_HOST_CACHE=0 disables."""
        key = tuple(grid_group.flatten().tolist())
        if self._host_cache:
            hit = self._group_layout_cache.get(key)
            if hit is not None:
                return hit
        pos_idx = _pack_multi_pos(grid_group, self._sms)                # [seq, 2] int16
        window_mask = reverses = window_index_multi = cu_window_seqlens = None
        if self._window:
            # Per-image window reorder (offset window_index keeps each image's block
            # disjoint) + ONE shared [n_i, n_i] mask reused for every per-image SDPA.
            (window_index_multi, reverses, window_mask,
             cu_window_seqlens) = _pack_multi_image(
                grid_group, self._window_size, self._sms, self._patch_size)
            pos_idx = pos_idx.reshape(seq // smu, smu, 2)[window_index_multi].reshape(seq, 2)
        out = (pos_idx, window_index_multi, reverses, window_mask,
               cu_window_seqlens)
        if self._host_cache:
            self._group_layout_cache[key] = out
        else:
            self._last_window_mask_ref = window_mask
        return out

    def enable_normalize_fold(self, rescale: float, image_mean, image_std) -> bool:
        """Bake the image normalize (x*rescale - mean)/std into the patch_embed weight so
        the host preproc can patchify RAW uint8 (no fp32 cast / no normalize passes). Built
        from self._patch_w [hidden, K] in the Qwen2.5-VL [ch, t, ps, ps] column layout.
        Returns True on success (the caller then emits uint8 pixel_values).

        Identity: F.linear(normalize(u8), W) == F.linear([u8 | 1 | 0…], W_fold), where
          W_fold[:, j] = W[:, j] * rescale/std[ch(j)]   (j in [0, K))
          W_fold[:, K] = -Σ_j W[:, j] * mean[ch(j)]/std[ch(j)]   (homogeneous bias column)
        normalize(u8)[j] = u8[j]*rescale/std[ch] - mean[ch]/std[ch], so the per-row dot
        product reproduces the normalized GEMM exactly (validated fp32-identity in the
        bench; fp16 cos 1.0002)."""
        if self._patch_w is None or self._pe_k is None:
            return False
        W = self._patch_w.float()                                  # [hidden, K]
        O, K = W.shape
        if self._pe_k <= K:
            raise RuntimeError(
                "normalize-fold patch embedding has no pad column for the "
                f"homogeneous bias: padded={self._pe_k}, input={K}"
            )
        ps = self._patch_size
        tp = K // (3 * ps * ps)                                    # temporal_patch_size
        mean = torch.tensor(image_mean, dtype=torch.float32)
        std = torch.tensor(image_std, dtype=torch.float32)
        scale = (rescale / std).view(3, 1, 1, 1).expand(3, tp, ps, ps).reshape(K)
        shift = (mean / std).view(3, 1, 1, 1).expand(3, tp, ps, ps).reshape(K)
        fold = torch.zeros(O, self._pe_k, dtype=torch.float32)
        fold[:, :K] = W * scale
        fold[:, K] = -(W * shift).sum(dim=1)                       # bias at homogeneous col K
        self._patch_w_fold_cpu = fold
        self._patch_w_fold_rpu = _to_rpu_half(
            tp_col_swizzle_mc_weight(fold.to(torch.float16), _VISION_CORES))
        return True

    def _index_rpu(self, kind: str, indices: torch.Tensor) -> torch.Tensor:
        """Publish private layout indices once while their full contents agree."""
        indices = indices.to(dtype=torch.int32).contiguous()
        key = (kind, tuple(indices.shape))
        cached = self._index_rpu_cache.get(key) if self._host_cache else None
        if cached is not None and torch.equal(cached[0], indices):
            return cached[1]
        out = indices.to(device="rpu").contiguous()
        if self._host_cache:
            self._index_rpu_cache[key] = (indices.clone(), out)
        return out

    def _refresh_position_keepalive(self, pos_idx: torch.Tensor) -> None:
        """Only this adapter writes the native position owner's live prefix."""
        cached = self._position_keepalive_snapshot if self._host_cache else None
        if (cached is not None and cached[0] is self._keepalive
                and torch.equal(cached[1], pos_idx)):
            return
        # copy_ publishes the CPU rows directly; a temporary RPU source would
        # add another upload plus an RPU-to-host invalidation before this copy.
        self._keepalive.narrow(0, 0, pos_idx.size(0)).copy_(pos_idx)
        if self._host_cache:
            self._position_keepalive_snapshot = (self._keepalive, pos_idx.clone())

    def _patch_perm_ids(self, window_index: torch.Tensor, seq: int, smu: int) -> torch.Tensor:
        """Per-row spatial→window gather indices [seq] int32 on RPU. window_index
        [seq/smu] permutes merge-units (out unit j = in unit window_index[j]); expand
        to per-row: perm[j*smu+r] = window_index[j]*smu + r. The CPU layout cache
        owns the source, and the device index cache checks every value before reuse."""
        wi = window_index.to(torch.int64)                          # [seq/smu]
        perm = (wi.unsqueeze(1) * smu + torch.arange(smu)).reshape(-1)  # [seq]
        return self._index_rpu("patch", perm)

    def _prepare_patch_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Preserve the existing FP16 cast, pad and homogeneous-bias order."""
        seq, width = pixels.shape
        folded = pixels.dtype == torch.uint8
        if self._host_cache:
            key = (seq, folded)
            buf = self._pe_buf_cache.get(key)
            if buf is None:
                buf = torch.zeros(seq, self._pe_k, dtype=torch.float16)
                if folded:
                    buf[:, width] = 1.0
                self._pe_buf_cache[key] = buf
            buf[:, :width].copy_(pixels)
            return buf
        out = F.pad(pixels.to(torch.float16), (0, self._pe_k - width))
        if folded:
            out[:, width] = 1.0
        return out.contiguous()

    def _patch_embed(self, pixel_values: torch.Tensor, window_index, seq: int,
                     smu: int, prepared_pixels: torch.Tensor | None = None) -> torch.Tensor:
        """pixel_values [seq, K] (raster order) → embed [1, seq, hidden] fp16 on RPU.
        uint8 → the normalize-folded weight (+ homogeneous bias column); fp32
        (HF-processor fallback) → the plain weight. patch_embed is per-row, so the
        window reorder commutes: GEMM in raster order, then gather the OUTPUT rows into
        window order ON-DEVICE (llama_gather_embedding), replacing the ~2 ms/frame CPU
        fancy-index gather on the [seq, K] pixels. The fp32 fallback keeps the CPU
        pre-GEMM reorder (no device gather on that path)."""
        pv = pixel_values                                          # raster order
        folded = pv.dtype == torch.uint8
        w_rpu = self._patch_w_fold_rpu if folded else self._patch_w_rpu
        if self._device_patch_embed and w_rpu is not None:
            x = (self._prepare_patch_pixels(pv)
                 if prepared_pixels is None else prepared_pixels)
            return self._consume_patch_embed_input(x, w_rpu, window_index, seq, smu)
        # CPU fp32 GEMM fallback — reorder the pixels here (no device gather on this path).
        if window_index is not None:
            pv = pv.reshape(seq // smu, smu, -1)[window_index].reshape(seq, -1)
        if folded:
            x = F.pad(pv.to(torch.float32), (0, self._pe_k - pv.size(1)))
            x[:, pv.size(1)] = 1.0
            embed = F.linear(x.contiguous(), self._patch_w_fold_cpu)
        else:
            embed = F.linear(pv.float(), self._patch_w)
        return embed.to(dtype=torch.float16, device="rpu").unsqueeze(0).contiguous()

    def _consume_patch_embed_input(self, x: torch.Tensor, weight: torch.Tensor,
                                   window_index, seq: int, smu: int) -> torch.Tensor:
        """Consume a prepared FP16 raster input through the ordinary device path."""
        device_input = x.to("rpu")
        embed = (torch.ops.rpu.linear_with_accumulation(
            device_input, weight, None, 1, _VISION_CORES, True)
            if self._linear_acc32 else F.linear(device_input, weight))
        if window_index is not None:
            perm = self._patch_perm_ids(window_index, seq, smu)
            embed = torch.ops.rpu.gather_embedding(embed.contiguous(), perm)
        return embed.unsqueeze(0).contiguous()

    def _patch_embed_pair_252(self, pixel_values: torch.Tensor,
                              window_index) -> torch.Tensor:
        """Stage the admitted 2x324 real rows into their two 336-row spans."""
        if (self._host_cache and self._device_patch_embed
                and self._patch_w_fold_rpu is not None
                and pixel_values.device.type == "cpu"
                and pixel_values.dtype == torch.uint8):
            K = pixel_values.size(1)
            # A normal 672-row frame can write the pair's dummy rows. Keep this
            # layout separate from the ordinary (seq, folded) cache namespace.
            key = ("pair252", 2, 324, 336, K, self._pe_k, True)
            buf = self._pe_buf_cache.get(key)
            if buf is None:
                buf = torch.zeros(2, 336, self._pe_k, dtype=torch.float16)
                buf[:, :, K] = 1.0  # Includes dummy rows: raw zero still has bias.
                self._pe_buf_cache[key] = buf
            buf[:, :324, :K].copy_(pixel_values.reshape(2, 324, K))
            return self._consume_patch_embed_input(
                buf.reshape(672, self._pe_k), self._patch_w_fold_rpu,
                window_index, 672, 4)
        # CPU projection, nonfolded inputs and HOST_CACHE=0 retain the original
        # padding and patch-embedding arithmetic.
        padded = pixel_values.new_zeros((2, 336, pixel_values.size(1)))
        padded[:, :324].copy_(pixel_values.reshape(2, 324, -1))
        return self._patch_embed(padded.reshape(672, -1), window_index, 672, 4)

    def _forward_one(self, pixel_values: torch.Tensor, grid_cpu: torch.Tensor,
                     fused: bool = False,
                     prepared_pixels: torch.Tensor | None = None) -> torch.Tensor:
        """Single-image ViT (grid_cpu [1, 3]) → merged [patches/4, 2048]. The validated
        P3a/P3b path (full or window attention, ≤256 patches). When `fused`, return the
        RAW window-order merged + stash `_pending_rev` (the runtime applies the reverse
        inside assemble_inputs_embeds_rev)."""
        seq = int(grid_cpu.prod(-1).sum().item())
        self._validate_execution_chunks(
            (self._native_chunk_size(grid_cpu),)
        )
        smu = self._sms * self._sms

        layout = self._window_layout(grid_cpu, seq)
        if self._per_window_sdpa:
            pos_idx, window_index, reverse_index, window_mask, cu_window_seqlens = layout
        else:
            pos_idx, window_index, reverse_index, window_mask = layout
            cu_window_seqlens = None

        embed_rpu = self._patch_embed(
            pixel_values, window_index, seq, smu, prepared_pixels)
        # copy_ publishes this private 2D RoPE position owner before capture.
        self._refresh_position_keepalive(pos_idx)

        self.cache.reset_to_position(0)
        k_caches = [self.cache.k_caches[i] for i in range(self.num_layers)]
        v_caches = [self.cache.v_caches[i] for i in range(self.num_layers)]

        # Eager patch_embed/gather above uses the shared temporary arena after
        # WallOssVLA entered the Vision subsystem. Reset again at the exact graph
        # boundary so BUILD and every REPLAY allocate the fused Vision layout from
        # the same t_end=0. embed_rpu/keepalive/KV live in DDR and survive this.
        torch.ops.rpu.spm_alloc_reset_temporary()
        native_boundaries = (
            cu_window_seqlens if self._per_window_sdpa else None
        )
        planned_chunk_size, a6_plan, planned_stage_descriptor, window_boundaries = (
            self._plan_native_execution(
                seq, 1, window_mask, native_boundaries)
        )
        dyn_dims = _vision_graph_dyn_dims(
            [
                self.num_layers,
                self._native_chunk_size(grid_cpu),
                int(self._window),
                int(self._mixed_precision),
                int(torch.rpu.get_debug_export()),
            ],
            window_boundaries,
        )
        dyn_dims.extend(a6_plan.graph_key_words())
        sig = rpu_backend.graph.GraphSignature(
            op_id="wall_oss_vision",
            shapes=[seq, self.hidden_size],
            dyn_dims=dyn_dims,
            dtypes=[torch.float16],
        )
        with self._graph_cache.capture(sig):
            out = torch.ops.rpu.qwen25vl_vision_forward(
                self._handle, embed_rpu, k_caches, v_caches, seq, window_mask,
                1, native_boundaries, planned_stage_descriptor, position_published=True,
            )
        # Successful native forward consumed this exact descriptor and checked
        # its compute/qkv capacities against the live execution plan.
        self._record_native_execution(
            a6_plan, seq=seq, image_batch_count=1,
            chunk_size=planned_chunk_size,
        )

        if self._fused_merger:
            # C++ post_fn already ran the merger; `out` is [seq/4, out_hidden]
            # window-order (RPU fp16) — NOT [1, seq, hidden]. Apply only the
            # window reverse-gather (merge-unit granularity), on-device.
            merged = out
            if reverse_index is not None:
                ri = self._index_rpu("reverse", reverse_index)
                if fused:
                    # Defer the reverse to the runtime's fused assemble op.
                    self._pending_rev = (ri, [seq // smu])
                    return merged                                       # window order
                merged = torch.ops.rpu.gather_embedding(merged, ri)
            return merged

        hidden = out.squeeze(0)                                         # [seq, hidden] RPU fp16
        if self._device_merged:
            merged = self._merger(hidden, device_out=True)              # [seq/4, out_hidden] RPU fp16
            if reverse_index is not None:
                # Undo the window permutation on-device (row gather via llama_gather_embedding).
                ri = self._index_rpu("reverse", reverse_index)
                merged = torch.ops.rpu.gather_embedding(merged, ri)     # [seq/4, out_hidden] RPU
            return merged
        merged = self._merger(hidden)                                   # [seq/4, out_hidden] CPU fp32
        if reverse_index is not None:
            # Undo the window permutation AFTER the merger (merge-unit granularity).
            merged = merged[reverse_index]
        return merged

    def _forward_pair_252(self, pixel_values: torch.Tensor,
                          grid: torch.Tensor) -> torch.Tensor:
        """Encode two images with shared Linear calls, then remove physical pads."""
        if (self.hidden_size != 1280 or not self._window or not self._fused_merger
                or self._mixed_precision or self._per_window_sdpa
                or self._layer_group or self._weight_outer
                or self._native_window_sdpa
                or (self._sms, self._window_size, self._patch_size) != (2, 112, 14)
                or pixel_values.dim() != 2 or pixel_values.size(0) != 648):
            raise ValueError("paired Vision requires the exact uniform window/merger layout")
        if self.cache.max_seq_len < 672:
            raise ValueError("paired Vision requires at least 672 cache rows")
        key = ("pair252", *grid.flatten().tolist())
        layout = self._group_layout_cache.get(key) if self._host_cache else None
        if layout is None:
            layout = _pair_252_layout(grid)
            if self._host_cache:
                self._group_layout_cache[key] = layout
        pos, win, reverse, window_mask, full_mask = layout
        embed = self._patch_embed_pair_252(pixel_values, win)
        # The following ordinary camera must refresh after this distinct layout.
        self._position_keepalive_snapshot = None
        self._keepalive[:672].copy_(pos)
        self.cache.reset_to_position(0)
        k = [self.cache.k_caches[i] for i in range(self.num_layers)]
        v = [self.cache.v_caches[i] for i in range(self.num_layers)]
        torch.ops.rpu.spm_alloc_reset_temporary()
        chunk, plan, descriptor, _ = self._plan_native_execution(
            672, 2, window_mask, None, valid_patches_per_image=324)
        dyn = [self.num_layers, 672, 1, 2, 324]
        dyn.extend(plan.graph_key_words())
        sig = rpu_backend.graph.GraphSignature(
            op_id="wall_oss_vision", shapes=[672, self.hidden_size],
            dyn_dims=dyn, dtypes=[torch.float16])
        with self._graph_cache.capture(sig):
            out = torch.ops.rpu.qwen25vl_vision_forward(
                self._handle, embed, k, v, 672, window_mask, 2, None,
                descriptor, 324, full_mask, position_published=True)
        self._record_native_execution(
            plan, seq=672, image_batch_count=2, chunk_size=chunk,
            valid_patches_per_image=324)
        # Fresh merger DDR survives releasing the shared temporary arena.
        torch.ops.rpu.spm_alloc_reset_temporary()
        return torch.ops.rpu.gather_embedding(
            out, reverse.to(torch.int32).to("rpu").contiguous())

    def _forward_group(self, pixel_values: torch.Tensor,
                       grid_group: torch.Tensor, fused: bool = False,
                       prepared_pixels: torch.Tensor | None = None) -> torch.Tensor:
        """Batched ViT over ``g`` equal-size images packed into ONE forward (g≤3).

        Mirrors :meth:`_forward_one` but runs all g images' patches as one
        ``[1, g·n_i, hidden]`` sequence, passing ``image_batch_count=g`` so the C++
        routes the generic schedule to image-local SDPA. The exact 3x576
        weight-outer schedule instead uses a dedicated batched by-MHA kernel;
        both preserve block-diagonal image isolation."""
        g = grid_group.size(0)
        if not bool(
            (grid_group.prod(-1) == grid_group[0].prod()).all()
        ):
            raise ValueError("batched group requires equal-size images")
        n_i = int(grid_group[0].prod().item())
        seq = g * n_i
        smu = self._sms * self._sms
        layer_group = self._layer_group
        weight_outer = self._weight_outer
        grouped_chunks = (
            (layer_group == 32 or weight_outer)
            and seq == _LAYER_GROUP_TOTAL_SEQ)
        if grouped_chunks:
            if g != 3 or n_i != 576:
                raise ValueError(
                    "experimental 3-image vision schedule requires 3x576 "
                    f"patches, got {g}x{n_i}"
                )
        cache_seq = n_i if grouped_chunks else seq
        self._validate_execution_chunks(
            (self._native_chunk_size(grid_group),)
        )
        if cache_seq > self.cache.max_seq_len:
            raise ValueError(
                f"vision attention seq {cache_seq} exceeds cache capacity "
                f"{self.cache.max_seq_len}"
            )

        _vi = rpu_env_bool("RPU_WALL_OSS_INSTRUMENT")
        _vt = [time.perf_counter()]; _vp = {}
        def _vm(n):
            if _vi:
                x = time.perf_counter(); _vp[n] = (x - _vt[0]) * 1000; _vt[0] = x
        # Memoized batched window/position layout (see _group_layout) — ~2.6 ms/frame of
        # _pack_multi_pos + _pack_multi_image, frame-invariant for a fixed camera set.
        (pos_idx, window_index_multi, reverses, window_mask,
         cu_window_seqlens) = self._group_layout(grid_group, seq, smu)
        if pos_idx.size(0) != seq:
            raise RuntimeError(
                f"batched vision position rows {pos_idx.size(0)} != sequence {seq}"
            )
        _vm("a_group_layout")

        # Batched patch_embed: pack all g images' patches into one [seq, K] slab and run
        # ONE GEMM (per-row, so the window reorder commutes; bit-exact vs the per-image
        # loop). uint8 input → normalize-folded weight; fp32 (HF fallback) → plain weight.
        embed_rpu = self._patch_embed(
            pixel_values, window_index_multi, seq, smu, prepared_pixels)
        _vm("b_patch_embed")
        self._refresh_position_keepalive(pos_idx)

        self.cache.reset_to_position(0)
        k_caches = [self.cache.k_caches[i] for i in range(self.num_layers)]
        v_caches = [self.cache.v_caches[i] for i in range(self.num_layers)]
        _vm("c_keepalive+reset")

        # _patch_embed may run eager RPU Linear + gather, both before capture.
        # Re-establish the fused graph's deterministic temporary-SPM entry state.
        torch.ops.rpu.spm_alloc_reset_temporary()
        native_window_boundaries = self._select_native_window_boundaries(
            grouped_chunks, cu_window_seqlens, grid_group
        )
        planned_chunk_size, a6_plan, planned_stage_descriptor, window_boundaries = (
            self._plan_native_execution(
                seq, int(g), window_mask, native_window_boundaries)
        )
        graph_dyn_dims = _vision_graph_dyn_dims(
            [self.num_layers, self._native_chunk_size(grid_group),
             int(self._window), g,
             layer_group if grouped_chunks else 0,
             int(weight_outer), int(self._mixed_precision),
             int(torch.rpu.get_debug_export())],
            window_boundaries,
        )
        graph_dyn_dims.extend(a6_plan.graph_key_words())
        sig = rpu_backend.graph.GraphSignature(
            op_id="wall_oss_vision",
            shapes=[seq, self.hidden_size],
            # g distinguishes 1×768 (single big image) from g×256 (batched) — they
            # take different SDPA branches and must NOT share a cached graph.
            dyn_dims=graph_dyn_dims,
            dtypes=[torch.float16],
        )
        with self._graph_cache.capture(sig):
            out = torch.ops.rpu.qwen25vl_vision_forward(
                self._handle, embed_rpu, k_caches, v_caches, seq, window_mask, g,
                native_window_boundaries, planned_stage_descriptor, position_published=True,
            )
        self._record_native_execution(
            a6_plan, seq=seq, image_batch_count=int(g),
            chunk_size=planned_chunk_size,
        )
        _vm("d_vision_graph(wrap+hw)")

        if self._fused_merger:
            # C++ post_fn ran either one packed merger or the exact
            # weight-outer schedule's image-local merger passes. Both return
            # [seq/4, out_hidden] window-order with g contiguous per-image
            # blocks of `mu` rows — NOT [1, seq, hidden]. Apply the per-image
            # reverse via a COMPOSED global index (reverses[i] is
            # image-local 0..mu-1; +i*mu lifts it to the packed [seq/4] block) +
            # one on-device gather.
            merged = out                                                # [seq/4, out_hidden] RPU fp16
            if reverses is not None:
                mu = n_i // smu                                         # merged tokens/image
                global_rev = torch.cat(
                    [reverses[i] + i * mu for i in range(g)], dim=0)    # [seq/4]
                gri = self._index_rpu("reverse", global_rev)
                if fused:
                    # Defer the reverse to the runtime's fused assemble op (folds the
                    # window-reverse gather + the per-run scatter into one op).
                    self._pending_rev = (gri, [mu] * g)
                    return merged                                       # window order
                # Keep the reverse ON-DEVICE (vs the old CPU round-trip: download 786 KB +
                # CPU gather + re-upload). The on-device gather needs ~seq/4·out_hidden of
                # SPM staging, which OOMs while the vision graph still holds the arena —
                # so free it FIRST. We are OUTSIDE the capture, and the cached graph keeps
                # its baked [0,peak] offsets for the next replay, so this plain reset is
                # safe (it is the same reset predict() does at the vision→prefill boundary,
                # moved one step earlier; `out`/`merged` live in DDR and survive it).
                torch.ops.rpu.spm_alloc_reset_temporary()
                merged = torch.ops.rpu.gather_embedding(merged, gri)   # [seq/4, out_hidden] RPU
            _vm("e_merger+reverse")
            if _vi:
                print("[INSTR vision] " + "  ".join(f"{k}={v:.2f}" for k, v in _vp.items()))
            return merged

        hidden = out.squeeze(0)                                         # [seq, hidden] RPU fp16
        merged = self._merger(hidden)                                   # [seq/4, out_hidden] CPU fp32
        if reverses is not None:
            mu = n_i // smu                                             # merged tokens/image
            merged = torch.cat(
                [merged[i * mu:(i + 1) * mu][reverses[i]] for i in range(g)], dim=0)
        return merged


def build_wall_oss_vision(
    ckpt_dir: str | None = None,
    *,
    layer_indices: Sequence[int] | None = None,
    window: bool = False,
    max_hw: int = 128,
    max_seq_len: int = 1024,
    w8a16: bool = False,
    fp16_ckpt_dir: str | None = None,
    execution_config=None,
    graph_runtime_policy=None,
    per_window_sdpa: bool = False,
    rmsnorm_fixed32: bool = False,
    _owned_vision: WallOssVision | None = None,
) -> WallOssVision:
    """Load `visual.*` weights and install the RPU Qwen2.5-VL ViT vision tower.

    window=False (P3a): every layer is full attention (MASK_NONE), no reorder.
    window=True  (P3b): block-diagonal window attention with fullatt_block_indexes
    as full-attn layers + CPU get_window_index reorder/reverse.

    ``ckpt_dir``/``fp16_ckpt_dir`` default to ``None`` → resolved at call time
    under the current ``$RPU_MODEL_CACHE`` (W8A16 vs fp16 registry path).
    """
    if graph_runtime_policy is not None:
        from rpu_backend.graph import GraphRuntimePolicy
        if not isinstance(graph_runtime_policy, GraphRuntimePolicy):
            raise TypeError("graph_runtime_policy must be GraphRuntimePolicy or None")
    from rpu_backend.api._execution import ExecutionSession, normalize_rpu_execution
    from rpu_backend.api._execution import _require_execution_process_safe
    _require_execution_process_safe()
    if _owned_vision is not None:
        session = getattr(_owned_vision, "_retirement_session", None)
        parent = getattr(session, "_owner", None)
        if (not isinstance(_owned_vision, WallOssVision) or not isinstance(session, ExecutionSession)
                or getattr(parent, "_execution_session", None) is not session
                or getattr(parent, "_diagnostic_vision", None) is not _owned_vision
                or _owned_vision._handle is not None or _owned_vision._closed):
            raise RuntimeError("owned Vision build requires its cold parent and pending native owner")
        session.require_cold()
    vision_execution = normalize_rpu_execution(
        {
            "vision": ({} if execution_config is None else
                       execution_config.get("vision", {}))
        },
        entry_point="build_wall_oss_vision",
        supported={"vision": ("chunk_size", "linear_acc32")},
    ).get("vision", {})
    execution_chunk_size = vision_execution.get("chunk_size", "auto")
    linear_acc32 = vision_execution.get("linear_acc32", False)
    for name, value in (("window", window), ("w8a16", w8a16),
                        ("per_window_sdpa", per_window_sdpa),
                        ("rmsnorm_fixed32", rmsnorm_fixed32)):
        if not isinstance(value, bool):
            raise ValueError(
                f"Wall-OSS vision {name} must be bool, got {value!r}"
            )
    if per_window_sdpa and not window:
        raise ValueError("per_window_sdpa=True requires window=True")
    experimental_schedule = _claim_experimental_schedule()
    mixed_precision = rpu_env_bool(
        "RPU_WALL_OSS_VISION_MIXED_PRECISION"
    )
    fused_merger = rpu_env_bool(
        "RPU_WALL_OSS_VISION_FUSED_MERGER",
    )
    if linear_acc32 and (mixed_precision or experimental_schedule != (0, False)
                         or per_window_sdpa or rmsnorm_fixed32):
        raise ValueError("Wall Vision ACC32 supports ordinary base only; RTC/exact/per-window modes retain their fixed accumulation")
    rope_spm = rpu_env_bool("RPU_WALL_OSS_VISION_ROPE_SPM")
    native_window_sdpa = rpu_env_bool(
        "RPU_WALL_OSS_VISION_NATIVE_WINDOW_SDPA")
    sorting = sorting_config_from_env()
    fp16_transfer = not w8a16 and bool(sorting.fp16_mask & 1)
    if rpu_env_bool("RPU_WALL_OSS_VISION_PAIR_252") and not (
            experimental_schedule == (0, False) and window and fused_merger
            and sorting.silu_mul and not mixed_precision and not per_window_sdpa
            and not native_window_sdpa and not rmsnorm_fixed32):
        raise ValueError("paired Vision requires uniform window/fused-merger/sorting SiLU")
    if rmsnorm_fixed32 and not (
        (w8a16 or fp16_transfer)
        and experimental_schedule == (0, True)
        and window
        and mixed_precision
        and fused_merger
    ):
        raise ValueError(
            "Wall-OSS Vision FIXED32 RMSNorm requires the exact RTC W8A16 "
            "or opted-in FP16 "
            "weight-outer/window/mixed/fused-merger profile"
        )
    exact_native_window = (
        (w8a16 or fp16_transfer)
        and experimental_schedule == (0, True)
        and window
        and mixed_precision
        and fused_merger
    )
    small_native_window = (
        (w8a16 or fp16_transfer)
        and sorting.vision_row
        and experimental_schedule == (0, False)
        and window
        and not mixed_precision
        and not per_window_sdpa
        and fused_merger
    )
    if native_window_sdpa and not (exact_native_window or small_native_window):
        raise ValueError(
            "RPU_WALL_OSS_VISION_NATIVE_WINDOW_SDPA=1 requires the exact "
            "W8A16 or opted-in FP16 weight-outer/window/mixed/fused-merger profile, "
            "or uniform FP16/W8 window/fused-merger with row reuse for three16x16 grids"
        )
    if per_window_sdpa and (
        experimental_schedule != (0, False)
        or mixed_precision
    ):
        raise ValueError(
            "per-window Vision SDPA is incompatible with mixed precision, "
            "layer grouping, and weight-outer scheduling"
        )
    ckpt_dir = _resolve_ckpt(ckpt_dir, w8a16)
    fp16_ckpt_dir = _resolve_ckpt(fp16_ckpt_dir, False)
    mixed_w8a16 = w8a16 and mixed_precision
    if mixed_w8a16:
        validate_w8_nvfp4_checkpoint_pair(ckpt_dir, fp16_ckpt_dir)
    cfg = json.load(open(f"{ckpt_dir}/config.json"))
    vc = cfg["vision_config"]
    HID = vc["hidden_size"]            # 1280
    NHEADS = vc["num_heads"]           # 16
    HD = HID // NHEADS                 # 80
    INTER = vc["intermediate_size"]    # 3420
    SMS = vc["spatial_merge_size"]     # 2
    OUT_HID = vc["out_hidden_size"]    # 2048
    WIN_SIZE = vc["window_size"]       # 112
    PATCH = vc["patch_size"]           # 14
    EPS = 1e-6                         # Qwen2_5_VLRMSNorm eps
    DIM = NHEADS * HD                  # 1280 (= HID; qkv split unit)
    INTER_PAD = _INTER_PAD_W8A16 if w8a16 or fp16_transfer else _INTER_PAD

    if layer_indices is None:
        layer_indices = list(range(vc["depth"]))
    else:
        layer_indices = list(layer_indices)
    if mixed_w8a16 and layer_indices != list(range(32)):
        raise ValueError(
            "Wall-OSS Vision mixed W8A16 requires the full block range 0-31 "
            "so the block-17 FP16 boundary stays at its validated layer"
        )
    num_layers = len(layer_indices)

    # fullatt indices remapped to local layer positions (full-attn layers).
    fullatt_local = [layer_indices.index(i) for i in vc["fullatt_block_indexes"]
                     if i in layer_indices] if window else []

    st = _SafeTensorStore(ckpt_dir)
    cpu_st = _SafeTensorStore(fp16_ckpt_dir) if w8a16 else st

    def gf(key: str) -> torch.Tensor:
        return st.get_tensor(key).float()

    def gr(key: str) -> torch.Tensor:
        return st.get_tensor(key).contiguous()

    def g_cpu(key: str) -> torch.Tensor:
        return cpu_st.get_tensor(key).float()

    q_w, k_w, v_w, o_w = [], [], [], []
    gate_w, up_w, down_w = [], [], []
    q_s, k_s, v_s, o_s = [], [], [], []
    gate_s, up_s, down_s = [], [], []
    q_b, k_b, v_b, o_b = [], [], [], []
    gate_b, up_b, down_b = [], [], []
    norm1_w, norm2_w = [], []

    for L in layer_indices:
        p = f"visual.blocks.{L}"
        qkv_w = gr(f"{p}.attn.qkv.weight")  # [3*DIM, HID]
        qkv_b = gf(f"{p}.attn.qkv.bias")    # [3*DIM]
        qw, kw, vw = qkv_w[:DIM], qkv_w[DIM:2 * DIM], qkv_w[2 * DIM:3 * DIM]
        qb, kb, vb = qkv_b[:DIM], qkv_b[DIM:2 * DIM], qkv_b[2 * DIM:3 * DIM]
        ow = gr(f"{p}.attn.proj.weight")    # [HID, DIM]
        ob = gf(f"{p}.attn.proj.bias")      # [HID]
        if w8a16:
            qkv_scale = gr(f"{p}.attn.qkv.weight_scale")
            qs, ks, vs = (qkv_scale[:DIM].contiguous(),
                          qkv_scale[DIM:2 * DIM].contiguous(),
                          qkv_scale[2 * DIM:3 * DIM].contiguous())
            o_scale = gr(f"{p}.attn.proj.weight_scale")

        gu_w, gu_b = _load_vision_gate_up(
            st, f"{p}.mlp", HID, INTER, w8a16=w8a16,
        )
        gw, uw = _split_gate_up(
            gu_w, INTER, _INTER_PAD, f"{p}.mlp.gate_up_proj.weight"
        )
        gb, ub = _split_gate_up(
            gu_b, INTER, _INTER_PAD, f"{p}.mlp.gate_up_proj.bias"
        )
        gw = _pad_rows(gw.contiguous(), INTER_PAD)
        uw = _pad_rows(uw.contiguous(), INTER_PAD)
        gb = _pad_1d(gb.contiguous(), INTER_PAD)
        ub = _pad_1d(ub.contiguous(), INTER_PAD)
        down_key = f"{p}.mlp.down_proj.weight"
        dw = _pad_cols(
            (
                cpu_st.get_tensor(down_key).contiguous()
                if mixed_w8a16 and L == 17
                else gr(down_key)
            ),
            INTER_PAD,
        )
        db = gf(f"{p}.mlp.down_proj.bias")  # [HID]
        if w8a16:
            gu_scale = gr(f"{p}.mlp.gate_up_proj.weight_scale")
            gate_scale, up_scale = _split_gate_up(
                gu_scale,
                INTER,
                _INTER_PAD,
                f"{p}.mlp.gate_up_proj.weight_scale",
                check_padding=False,
            )
            gate_scale = _pad_1d(gate_scale.contiguous(), INTER_PAD)
            up_scale = _pad_1d(up_scale.contiguous(), INTER_PAD)
            down_scale = gr(f"{p}.mlp.down_proj.weight_scale")

        # Attention Q/K/V col-partition + O row-partition; MLP gate/up col + down
        # row — all 8-core (MHA, attn_tp=8). Bias NOT swizzled (col-swizzle keeps
        # each core's output channels contiguous, matching the C++ bias scatter).
        if w8a16:
            q_w.append(_to_rpu_int8(tp_col_swizzle_mc_weight(qw, _VISION_CORES, dwidth=1)))
            k_w.append(_to_rpu_int8(tp_col_swizzle_mc_weight(kw, _VISION_CORES, dwidth=1)))
            v_w.append(_to_rpu_int8(tp_col_swizzle_mc_weight(vw, _VISION_CORES, dwidth=1)))
            o_w.append(_to_rpu_int8(tp_row_swizzle_mc_weight(ow, _VISION_CORES, dwidth=1)))
            gate_w.append(_to_rpu_int8(tp_col_swizzle_mc_weight(gw, _VISION_CORES, dwidth=1)))
            up_w.append(_to_rpu_int8(tp_col_swizzle_mc_weight(uw, _VISION_CORES, dwidth=1)))
            down_w.append(
                _to_rpu_half(tp_row_swizzle_mc_weight(dw.half(), _VISION_CORES))
                if mixed_w8a16 and L == 17
                else _to_rpu_int8(
                    tp_row_swizzle_mc_weight(dw, _VISION_CORES, dwidth=1)
                )
            )
            q_s.append(_to_rpu_half(qs)); k_s.append(_to_rpu_half(ks)); v_s.append(_to_rpu_half(vs))
            o_s.append(_to_rpu_half(o_scale))
            gate_s.append(_to_rpu_half(gate_scale))
            up_s.append(_to_rpu_half(up_scale))
            down_s.append(_to_rpu_half(down_scale))
        else:
            q_w.append(_to_rpu_half(tp_col_swizzle_mc_weight(qw.half(), _VISION_CORES)))
            k_w.append(_to_rpu_half(tp_col_swizzle_mc_weight(kw.half(), _VISION_CORES)))
            v_w.append(_to_rpu_half(tp_col_swizzle_mc_weight(vw.half(), _VISION_CORES)))
            o_w.append(_to_rpu_half(tp_row_swizzle_mc_weight(ow.half(), _VISION_CORES)))
            gate_w.append(_to_rpu_half(tp_col_swizzle_mc_weight(gw.half(), _VISION_CORES)))
            up_w.append(_to_rpu_half(tp_col_swizzle_mc_weight(uw.half(), _VISION_CORES)))
            down_w.append(_to_rpu_half(tp_row_swizzle_mc_weight(dw.half(), _VISION_CORES)))

        q_b.append(_to_rpu_half(qb)); k_b.append(_to_rpu_half(kb)); v_b.append(_to_rpu_half(vb))
        o_b.append(_to_rpu_half(ob))
        gate_b.append(_to_rpu_half(gb)); up_b.append(_to_rpu_half(ub)); down_b.append(_to_rpu_half(db))
        norm1_w.append(_to_rpu_half(gf(f"{p}.norm1.weight")))
        norm2_w.append(_to_rpu_half(gf(f"{p}.norm2.weight")))

    set_weights = (
        torch.ops.rpu.qwen25vl_vision_set_weights_w8a16
        if w8a16 else torch.ops.rpu.qwen25vl_vision_set_weights
    )
    weight_args = [
        q_w, k_w, v_w, o_w,
        gate_w, up_w, down_w,
        norm1_w, norm2_w,
        q_b, k_b, v_b, o_b,
        gate_b, up_b, down_b,
        NHEADS, HD, HID, INTER_PAD, float(EPS),
        fullatt_local,
    ]
    if w8a16:
        weight_args.extend([q_s, k_s, v_s, o_s, gate_s, up_s, down_s])

    freq_cos, freq_sin = build_vision_rope_tables(HD, max_hw, device="rpu",
                                                  dtype=torch.float16, theta=10000.0)

    cache = RPUCache(
        num_layers=num_layers, batch_size=1, max_seq_len=max_seq_len,
        num_kv_heads=NHEADS, head_dim=HD, attn_tp=min(8, NHEADS),
    )
    if _owned_vision is not None:
        _owned_vision.cache = cache
        _owned_vision._keepalive = (weight_args, freq_cos, freq_sin)
    graph_cache = rpu_backend.graph.GraphCache(
        **({"runtime_policy": graph_runtime_policy}
           if graph_runtime_policy is not None else {}))
    if _owned_vision is not None:
        _owned_vision._graph_cache = graph_cache

    # patch_embed: fold Conv3d [HID,3,2,14,14] → Linear [HID, 1176], CPU fp32, no bias.
    patch_w = g_cpu("visual.patch_embed.proj.weight").reshape(HID, -1).contiguous()

    # merger: RMSNorm (CPU fp32) → reshape(-1, 4*HID) → Linear → GELU → Linear.
    # The two Linears dominate vision wall (~40ms CPU fp32: m0 5120x5120 + m2
    # 2048x5120), so they run on RPU fp16. Weights are col-partition swizzled
    # (8-core) to match rpu_linear's rpu_get_linear_partition choice (both shapes
    # satisfy can_row && can_col → returns col). RMSNorm stays CPU fp32 (cheap,
    # keeps the variance fp32 to match the golden); the [N,5120] hop to RPU is ~1MB.
    merge_hidden = HID * SMS * SMS    # 5120
    ln_q_w = g_cpu("visual.merger.ln_q.weight")
    ln_q_w_rpu = _to_rpu_half(ln_q_w)               # device weight for the on-device rms_norm
    m0_b_rpu = _to_rpu_half(g_cpu("visual.merger.mlp.0.bias"))
    m2_b_rpu = _to_rpu_half(g_cpu("visual.merger.mlp.2.bias"))
    m0_w_rpu = _to_rpu_half(tp_col_swizzle_mc_weight(
        g_cpu("visual.merger.mlp.0.weight").half(), _VISION_CORES))   # [5120,5120]
    m2_w_rpu = _to_rpu_half(tp_col_swizzle_mc_weight(
        g_cpu("visual.merger.mlp.2.weight").half(), _VISION_CORES))   # [2048,5120]

    # Plumb merger weights into the C++ fused subsystem. The fused post_fn runs
    # m0 as a COL-partition GEMM (matches m0_w_rpu's col-swizzle) and m2 as a
    # ROW-partition GEMM + all-reduce (mirrors the ViT down_proj). The eager
    # `merger` closure below dispatches m2 via F.linear → rpu_get_linear_partition
    # → COL, so it needs the COL-swizzled m2_w_rpu. The C++ ROW GEMM needs a
    # ROW-swizzled m2 — register that separately; the eager path keeps m2_w_rpu.
    m2_w_rpu_row = _to_rpu_half(tp_row_swizzle_mc_weight(
        g_cpu("visual.merger.mlp.2.weight").half(), _VISION_CORES))   # [2048,5120] row
    merger_args = (
        ln_q_w_rpu, m0_w_rpu, m0_b_rpu, m2_w_rpu_row, m2_b_rpu, OUT_HID
    )
    if _owned_vision is not None:
        _owned_vision._keepalive += (merger_args,)

    def merger(x: torch.Tensor, device_out: bool = False) -> torch.Tensor:
        # x [seq, HID] RPU fp16 → [seq/4, OUT_HID]. device_out=True keeps the result on
        # the RPU (no x readback, no merged upload downstream): the RMSNorm runs via the
        # public eager RMSNorm fallback (FP32 host math over zero-copy DDR views).
        # The GEMMs run on RPU; eager exact-ERF GELU uses the registered CPU fallback.
        if device_out:
            xr = torch.ops.rpu.rms_norm(x, [HID], ln_q_w_rpu, EPS).reshape(-1, merge_hidden)
        else:
            xr = _rmsnorm_cpu(x.cpu().float(), ln_q_w, EPS).reshape(-1, merge_hidden)
            xr = _to_rpu_half(xr)                    # [N, 5120] RPU fp16
        xr = (torch.ops.rpu.linear_with_accumulation(
            xr, m0_w_rpu, m0_b_rpu, 1, _VISION_CORES, True)
            if linear_acc32 else F.linear(xr, m0_w_rpu, m0_b_rpu))
        xr = F.gelu(xr)                              # graph-aware CPU exact-erf fallback
        xr = (torch.ops.rpu.linear_with_accumulation(
            xr, m2_w_rpu, m2_b_rpu, 1, _VISION_CORES, True)
            if linear_acc32 else F.linear(xr, m2_w_rpu, m2_b_rpu))
        return xr if device_out else xr.float().cpu()   # device fp16 OR CPU fp32 (legacy)

    # On-device patch_embed weight: pad K (1176) to a multiple of 16, col-swizzle, upload
    # once. F.linear on rpu then replaces the per-frame CPU fp32 GEMM.
    _pe_k = ((patch_w.size(1) + 15) // 16) * 16
    patch_w_rpu = _to_rpu_half(tp_col_swizzle_mc_weight(
        F.pad(patch_w.to(torch.float16), (0, _pe_k - patch_w.size(1))), _VISION_CORES))

    # Standalone handles stay builder-owned until their tracked finalizer exists;
    # diagnostic children publish the handle immediately to the parent instead.
    handle, keepalive = _configure_qwen25vl_vision_handle(
        set_weights=set_weights,
        max_seq_len=cache.max_seq_len,
        weight_args=weight_args,
        freq_cos=freq_cos,
        freq_sin=freq_sin,
        merger_args=merger_args,
        execution_routes=(
            rope_spm,
            fused_merger,
            mixed_precision,
            experimental_schedule[0] == 32,
            experimental_schedule[1],
            (_RMSNORM_CAP_FIXED32
             if rmsnorm_fixed32 else _RMSNORM_CAP_BASE),
        ),
        chunk_size=(
            0 if execution_chunk_size == "auto"
            else int(execution_chunk_size)
        ),
        per_window_sdpa=per_window_sdpa,
        _owned_vision=_owned_vision, sorting=sorting, linear_acc32=linear_acc32,
    )
    vis = None
    try:
        arguments = dict(
            handle=handle, cache=cache, graph_cache=graph_cache,
            keepalive=(keepalive if _owned_vision is None else
                       (keepalive, _owned_vision._keepalive)), patch_w=patch_w, merger=merger,
            hidden_size=HID, num_layers=num_layers, spatial_merge_size=SMS,
            window=window, window_size=WIN_SIZE, patch_size=PATCH,
            experimental_schedule=experimental_schedule,
            execution_chunk_size=execution_chunk_size,
            per_window_sdpa=per_window_sdpa,
            fused_merger=fused_merger,
            mixed_precision=mixed_precision, linear_acc32=linear_acc32,
        )
        if _owned_vision is None:
            vis = WallOssVision.__new__(WallOssVision)
            vis._handle, vis._graph_cache, vis._closed = handle, graph_cache, False
            vis._gc_retirement_enabled = False
        else:
            vis = _owned_vision
        WallOssVision.__init__(vis, **arguments)
        vis._patch_w_rpu = patch_w_rpu
        vis._pe_k = _pe_k
        return vis
    except BaseException as error:
        if _owned_vision is None:
            from rpu_backend.adapters.wall_oss._retirement import _cleanup_build_failure
            pending = (weight_args, cache, graph_cache, freq_cos, freq_sin, merger_args, keepalive)
            if vis is not None:
                vis._gc_retirement_enabled = False
                vis._retirement_keepalive = pending
                _cleanup_build_failure(error, vis, vis.close)
            else:
                _cleanup_build_failure(error, (handle, pending),
                                       lambda: _destroy_qwen25vl_vision_handle(handle))
        raise
