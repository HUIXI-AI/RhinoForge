"""Wall-OSS-0.5 Qwen2.5-VL ViT vision tower for RPU.

The VLM backbone's vision tower is a stock Qwen2.5-VL ViT (32 blocks, hidden
1280, head_dim 80, RMSNorm, biased SwiGLU, 2D RoPE, window attention). This
adapter drives the native C++ model `qwen25vl_vision_*` (raw-tensor style, like
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

The full-attention path uses MASK_NONE with no window reorder. The window path
adds the window mask and CPU reorder/reverse.

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
import weakref
from typing import Sequence

import torch
import torch.nn.functional as F

import rpu_backend
from rpu_backend.runtime import rpu_env_bool
from rpu_backend.api.cache import RPUCache
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
_LAYER_GROUP_TOTAL_SEQ = 3 * 576  # Exact three-camera layer-group sequence length.
_INTER_PAD = 3456           # fp16: 3420 -> next multiple of 128
_INTER_PAD_W8A16 = 3584     # int8 row-swizzle needs K divisible by 32B*8 = 256
_PROCESS_EXPERIMENTAL_SCHEDULE: tuple[int, bool] | None = None


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
    # C++ reads the same value, so only 0/1 are accepted.
    weight_outer = rpu_env_bool(
        "RPU_WALL_OSS_VISION_WEIGHT_OUTER",
        cpp_mirror="src/fused/rpu_qwen25vl_vision_model.cpp")
    if weight_outer and layer_group:
        raise ValueError(
            "RPU_WALL_OSS_VISION_WEIGHT_OUTER and "
            "RPU_WALL_OSS_VISION_LAYER_GROUP are mutually exclusive"
        )
    return layer_group, weight_outer


def _claim_experimental_schedule() -> tuple[int, bool]:
    """Validate and pin the process-wide Vision schedule."""
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
    """Build the Qwen2.5-VL window permutation on CPU.

    Returns (window_index, cu_window_seqlens,
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
    base: Sequence[int], cu_window_seqlens: torch.Tensor | None
) -> list[int]:
    """Keep the ordinary Wall key unchanged; controlled window mode keys every boundary."""
    dims = list(base)
    if cu_window_seqlens is not None:
        dims.extend((1, *(int(value) for value in cu_window_seqlens.tolist())))
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

    Returns ``(window_index_multi, reverse_indices, mask)``:
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
    return window_index_multi, reverses, mask


def _destroy_qwen25vl_vision_handle(h):
    """Release the C++ Qwen25VLVisionModel handle. Called by weakref.finalize on GC."""
    try:
        torch.ops.rpu.qwen25vl_vision_destroy(h)
    except Exception:
        pass


def _configure_qwen25vl_vision_handle(
    *, set_weights, weight_args, freq_cos, freq_sin, merger_args,
    chunk_size=0,
    per_window_sdpa=False,
):
    """Create/configure a Vision handle, destroying it on any failed gate."""
    handle = torch.ops.rpu.qwen25vl_vision_create()
    configured = False
    try:
        set_weights(handle, *weight_args)
        torch.ops.rpu.qwen25vl_vision_set_chunk_size(handle, chunk_size)
        if per_window_sdpa:
            torch.ops.rpu.qwen25vl_vision_set_per_window_sdpa(handle, True)
        torch.ops.rpu.qwen25vl_vision_set_rope(handle, freq_cos, freq_sin)
        keepalive = torch.ops.rpu.qwen25vl_vision_position_idx_keepalive(handle)
        torch.ops.rpu.qwen25vl_vision_set_merger_weights(
            handle, *merger_args
        )
        configured = True
        return handle, keepalive
    finally:
        if not configured:
            _destroy_qwen25vl_vision_handle(handle)


class WallOssVision:
    """Qwen2.5-VL ViT vision tower on RPU.

    Construct via :func:`build_wall_oss_vision`. Call :meth:`forward` with CPU
    `pixel_values [num_patches, 1176]` + `grid_thw [1, 3]`; returns the merged
    vision features `[num_patches/4, 2048]` on CPU (fp32).
    """

    def __init__(self, *, handle, cache, graph_cache, keepalive, patch_w, merger,
                 hidden_size, num_layers, spatial_merge_size, window,
                 window_size, patch_size, experimental_schedule=None,
                 execution_chunk_size="auto", per_window_sdpa=False):
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
        self._window = window                 # window attention on/off
        self._window_size = window_size
        self._patch_size = patch_size
        self._execution_chunk_size = execution_chunk_size
        self._per_window_sdpa = per_window_sdpa
        # Keep the merged vision tokens on the RPU device (merger RMSNorm on-device +
        # window reverse-reorder via the gather op) so the VLA's on-device embed
        # assembly needs no merged readback/upload. Default off: this eager
        # path is superseded by the default fused-merger post_fn. Eager RMSNorm is
        # implemented through a host fallback, so this switch is diagnostic rather
        # than a performance alternative.
        self._device_merged = rpu_env_bool("RPU_WALL_OSS_VISION_DEVICE_MERGED")
        # RPU_WALL_OSS_VISION_FUSED_MERGER (default OFF): fold the merger
        # (RMSNorm -> m0 GEMM -> GELU -> m2 GEMM -> all-reduce) into the C++
        # qwen25vl_vision_forward graph via a post_fn. The normal route merges
        # one packed sequence; exact 3x576 weight-outer merges three image-local
        # chunks. The forward op returns the merged [seq/4, out_hidden]
        # window-order tensor directly; Python only applies the reverse-gather.
        #
        # ⚠️ SHARED SWITCH — MUST agree with the C++
        # `vision_fused_merger_enabled()` parser, which reads the same
        # variable as `e && s != "0" && s != "false"` (byte-exact). If C++ reads
        # ON and this reads OFF, the merger runs TWICE — folded into the graph
        # and again here — and the vision embedding is silently wrong, with no
        # exception. The two rules agree only on `1` and `0`; `VAR=` (what
        # `export VAR=$UNSET` yields), `VAR=False` and `VAR=off` are ON in C++
        # and OFF here, so `cpp_mirror` refuses every spelling but `1`/`0` —
        # the only two on which the Python and C++ parsers agree.
        self._fused_merger = rpu_env_bool(
            "RPU_WALL_OSS_VISION_FUSED_MERGER",
            cpp_mirror="src/fused/rpu_qwen25vl_vision_model.cpp")
        self._mixed_precision = rpu_env_bool(
            "RPU_WALL_OSS_VISION_MIXED_PRECISION",
            cpp_mirror="src/fused/rpu_qwen25vl_vision_model.cpp")
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
        if self._mixed_precision and schedule[1]:
            if rpu_env_bool("RPU_ALLREDUCE_TWOSTAGE"):
                raise ValueError(
                    "Wall-OSS Vision mixed weight-outer requires "
                    "RPU_ALLREDUCE_TWOSTAGE=0"
                )
            if os.environ.get("RPU_ALLREDUCE_RING", "").strip() != "profile":
                raise ValueError(
                    "Wall-OSS Vision mixed weight-outer requires "
                    "RPU_ALLREDUCE_RING=profile"
                )
        # On-device patch_embed: run the folded Conv3d→Linear as an eager RPU linear. _patch_w_rpu
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
        # camera). RPU_WALL_OSS_HOST_CACHE=0 disables this cache.
        self._layout_cache = {}          # single-image (_forward_one / _window_layout)
        self._group_layout_cache = {}    # batched multi-image (_forward_group / _group_layout)
        self._pe_buf_cache = {}          # persistent [seq, _pe_k] fp16 patch_embed input buffers
                                         # (bias/pad cols set once; data region refilled per frame)
        # HOST_CACHE=0 still keeps the immediately previous CPU window mask alive while
        # allocating the next one. This prevents allocator address reuse from fooling
        # the C++ source-pointer memo into skipping an upload for changed contents.
        self._last_window_mask_ref = None
        # Fused embed assembly returns raw window-order rows and stores the reverse
        # index for assemble_inputs_embeds_rev. Mixed-size subgroups use the fallback.
        self._fused_assemble = rpu_env_bool("RPU_WALL_OSS_FUSED_ASSEMBLE")
        self._pending_rev = None         # (rev_idx_rpu int32, [mu_per_image]) or None
        self._host_cache = rpu_env_bool("RPU_WALL_OSS_HOST_CACHE", default=True)
        # Controlled exact-shape scheduling. Treat this as an
        # instance/startup setting: the C++ side snapshots the same env value.
        self._layer_group, self._weight_outer = schedule
        self._closed = False
        # Register ownership only after every fallible constructor step. The
        # builder owns and destroys the raw handle until this assignment.
        self._handle_finalizer = weakref.finalize(
            self, _destroy_qwen25vl_vision_handle, handle
        )

    def close(self) -> None:
        """Release graph/native resources. Safe to call more than once."""
        if self._closed:
            return
        graph_cache = getattr(self, "_graph_cache", None)
        if graph_cache is not None:
            graph_cache.clear()
        finalizer = getattr(self, "_handle_finalizer", None)
        if finalizer is not None and getattr(finalizer, "alive", False):
            # Explicit close is transactional: unlike the GC callback, a Z1
            # prepared/active destroy rejection must stay visible and retryable.
            torch.ops.rpu.qwen25vl_vision_destroy(self._handle)
            finalizer.detach()
        self._closed = True

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """pixel_values [sum patches, 1176] + grid_thw [N, 3] → merged [sum patches/4, 2048].

        ③ multi-image: Qwen2.5-VL ViT attention is block-diagonal per image (image i
        NEVER attends image j). Consecutive equal-size images are batched into ONE
        packed forward (≤3 same-size images, ``_forward_group``) to amortize dispatch +
        weight-DMA; a size change or the cap starts a new group, and odd singletons fall
        back to ``_forward_one``. The generic route keeps attention image-local
        with per-image SDPA. The exact RTC weight-outer route admits only 3x576
        and uses one dedicated B=3 by-MHA launch per layer with SPM-resident K/V.
        """
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
        counts = [int(grid_cpu[i].prod().item()) for i in range(grid_cpu.size(0))]
        N = grid_cpu.size(0)
        batch_on = rpu_env_bool("RPU_WALL_OSS_BATCH_VISION", default=True)
        layer_group = self._layer_group
        weight_outer = self._weight_outer
        mixed_precision = getattr(self, "_mixed_precision", False)
        per_window_sdpa = getattr(self, "_per_window_sdpa", False)
        if _experimental_schedule_from_env() != (layer_group, weight_outer):
            raise RuntimeError(
                "Wall-OSS Vision experimental schedule is startup-only; "
                "rebuild the vision instance in a fresh process after changing it"
            )
        if rpu_env_bool(
            "RPU_WALL_OSS_VISION_MIXED_PRECISION",
            cpp_mirror="src/fused/rpu_qwen25vl_vision_model.cpp",
        ) != mixed_precision:
            raise RuntimeError(
                "Wall-OSS Vision mixed precision is startup-only; rebuild the "
                "vision instance in a fresh process after changing it"
            )
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
        max_seq = int(os.environ.get("RPU_WALL_OSS_BATCH_MAX_SEQ", str(_MAX_BATCH_SEQ)))
        if mixed_precision and not weight_outer:
            # The normal C++ path is fixed at M=576. Keep multiple cameras as
            # separate GraphCache signatures unless exact 3x576 weight-outer is
            # selected.
            max_seq = min(max_seq, 576)
        if layer_group == 32 or weight_outer:
            max_seq = max(max_seq, _LAYER_GROUP_TOTAL_SEQ)
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
        merged = [(self._forward_group(s, g) if is_g else self._forward_one(s, g))
                  for s, g, is_g in subgroups]
        return torch.cat(merged, dim=0)

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

    def _validate_execution_chunks(self, chunks) -> None:
        requested = getattr(self, "_execution_chunk_size", "auto")
        if requested == "auto":
            return
        native = tuple(int(chunk) for chunk in chunks)
        if any(chunk != int(requested) for chunk in native):
            raise ValueError(
                "Wall-OSS vision chunk_size must equal every native Vision "
                f"forward chunk for this request: requested={requested}, "
                f"native={native}"
            )

    def _window_layout(self, grid_cpu: torch.Tensor, seq: int):
        """Memoized CPU window/position layout `(pos_idx, window_index, reverse_index,
        window_mask)`. Pure function of the grid + (instance-fixed) merge/window config,
        so a fixed-camera rollout reuses one entry.
        All four are CPU tensors (the pos_idx→keepalive device copy stays per-forward) and
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
            # Reorder by merge-unit window_index on CPU.
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
        window_index_multi, reverses, window_mask)` — the group analog of _window_layout.
        Pure function of grid_group + (instance-fixed) window/merge/patch config, so a
        fixed-camera rollout reuses one entry. All CPU
        tensors, read-only downstream → safe to share. RPU_WALL_OSS_HOST_CACHE=0 disables."""
        key = tuple(grid_group.flatten().tolist())
        if self._host_cache:
            hit = self._group_layout_cache.get(key)
            if hit is not None:
                return hit
        pos_idx = _pack_multi_pos(grid_group, self._sms)                # [seq, 2] int16
        window_mask = reverses = window_index_multi = None
        if self._window:
            # Per-image window reorder (offset window_index keeps each image's block
            # disjoint) + ONE shared [n_i, n_i] mask reused for every per-image SDPA.
            window_index_multi, reverses, window_mask = _pack_multi_image(
                grid_group, self._window_size, self._sms, self._patch_size)
            pos_idx = pos_idx.reshape(seq // smu, smu, 2)[window_index_multi].reshape(seq, 2)
        out = (pos_idx, window_index_multi, reverses, window_mask)
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
        product reproduces the normalized GEMM exactly."""
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

    def _patch_perm_ids(self, window_index: torch.Tensor, seq: int, smu: int) -> torch.Tensor:
        """Per-row spatial→window gather indices [seq] int32 on RPU. window_index
        [seq/smu] permutes merge-units (out unit j = in unit window_index[j]); expand
        to per-row: perm[j*smu+r] = window_index[j]*smu + r. Frame-invariant math but
        recomputed+uploaded per frame (a small [seq] index array matching the existing
        reverse_index upload pattern)."""
        wi = window_index.to(torch.int64)                          # [seq/smu]
        perm = (wi.unsqueeze(1) * smu + torch.arange(smu)).reshape(-1)  # [seq]
        return perm.to(torch.int32).to("rpu").contiguous()

    def _patch_embed(self, pixel_values: torch.Tensor, window_index, seq: int,
                     smu: int) -> torch.Tensor:
        """pixel_values [seq, K] (raster order) → embed [1, seq, hidden] fp16 on RPU.
        uint8 → the normalize-folded weight (+ homogeneous bias column); fp32
        (HF-processor fallback) → the plain weight. patch_embed is per-row, so the
        window reorder commutes: GEMM in raster order, then gather the OUTPUT rows into
        window order on device (llama_gather_embedding), replacing the CPU
        fancy-index gather on the [seq, K] pixels. The fp32 fallback keeps the CPU
        pre-GEMM reorder (no device gather on that path)."""
        pv = pixel_values                                          # raster order
        folded = pv.dtype == torch.uint8
        w_rpu = self._patch_w_fold_rpu if folded else self._patch_w_rpu
        if self._device_patch_embed and w_rpu is not None:
            K = pv.size(1)
            # Persistent pre-initialized input buffer: the pad/bias columns are
            # frame-invariant (zeros, + the homogeneous bias col=1.0 when folded), so
            # allocate once per seq and refill only the data region each frame with a
            # single cast-copy. Gated on RPU_WALL_OSS_HOST_CACHE.
            if self._host_cache:
                # Key on (seq, folded): the folded path needs the homogeneous bias col
                # buf[:, K]=1.0, the fp32-fallback path does not — a seq-only key would
                # reuse a non-folded buffer for a folded frame and drop the normalize bias.
                ck = (seq, folded)
                buf = self._pe_buf_cache.get(ck)
                if buf is None:
                    buf = torch.zeros(seq, self._pe_k, dtype=torch.float16)
                    if folded:
                        buf[:, K] = 1.0                           # homogeneous bias col (frame-invariant)
                    self._pe_buf_cache[ck] = buf
                buf[:, :K].copy_(pv)                              # cast uint8/fp32 → fp16, data region only
                x = buf
            else:
                x = F.pad(pv.to(torch.float16), (0, self._pe_k - K))
                if folded:
                    x[:, K] = 1.0                                 # homogeneous bias activation
                x = x.contiguous()
            embed = F.linear(x.to("rpu"), w_rpu)                  # [seq, hidden] RASTER order
            if window_index is not None:
                perm = self._patch_perm_ids(window_index, seq, smu)
                embed = torch.ops.rpu.gather_embedding(embed.contiguous(), perm)  # → window order
            return embed.unsqueeze(0).contiguous()
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

    def _forward_one(self, pixel_values: torch.Tensor, grid_cpu: torch.Tensor,
                     fused: bool = False) -> torch.Tensor:
        """Single-image ViT (grid_cpu [1, 3]) → merged [patches/4, 2048].

        Supports full or window attention with at most 256 patches. When `fused`, return the
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

        embed_rpu = self._patch_embed(pixel_values, window_index, seq, smu)
        # 2D RoPE position_idx → model keepalive (C++ flushes before reading).
        self._keepalive.narrow(0, 0, seq).copy_(pos_idx.to(self._keepalive.device))

        self.cache.reset_to_position(0)
        k_caches = [self.cache.k_caches[i] for i in range(self.num_layers)]
        v_caches = [self.cache.v_caches[i] for i in range(self.num_layers)]

        # Eager patch_embed/gather above uses the shared temporary arena after
        # WallOssVLA entered the Vision subsystem. Reset again at the exact graph
        # boundary so BUILD and every REPLAY allocate the fused Vision layout from
        # the same t_end=0. embed_rpu/keepalive/KV live in DDR and survive this.
        torch.ops.rpu.spm_alloc_reset_temporary()
        dyn_dims = _vision_graph_dyn_dims(
            [
                self.num_layers,
                self._native_chunk_size(grid_cpu),
                int(self._window),
                int(self._mixed_precision),
                int(torch.rpu.get_debug_export()),
            ],
            cu_window_seqlens if self._per_window_sdpa else None,
        )
        sig = rpu_backend.graph.GraphSignature(
            op_id="wall_oss_vision",
            shapes=[seq, self.hidden_size],
            dyn_dims=dyn_dims,
            dtypes=[torch.float16],
        )
        with self._graph_cache.capture(sig):
            out = torch.ops.rpu.qwen25vl_vision_forward(
                self._handle, embed_rpu, k_caches, v_caches, seq, window_mask,
                1, cu_window_seqlens if self._per_window_sdpa else None,
            )

        if self._fused_merger:
            # C++ post_fn already ran the merger; `out` is [seq/4, out_hidden]
            # window-order (RPU fp16) — NOT [1, seq, hidden]. Apply only the
            # window reverse-gather (merge-unit granularity), on-device.
            merged = out
            if reverse_index is not None:
                ri = reverse_index.to(torch.int32).to("rpu").contiguous()
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
                ri = reverse_index.to(torch.int32).to("rpu").contiguous()
                merged = torch.ops.rpu.gather_embedding(merged, ri)     # [seq/4, out_hidden] RPU
            return merged
        merged = self._merger(hidden)                                   # [seq/4, out_hidden] CPU fp32
        if reverse_index is not None:
            # Undo the window permutation AFTER the merger (merge-unit granularity).
            merged = merged[reverse_index]
        return merged

    def _forward_group(self, pixel_values: torch.Tensor,
                       grid_group: torch.Tensor, fused: bool = False) -> torch.Tensor:
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
        # Memoized batched window/position layout, frame-invariant for a fixed camera set.
        pos_idx, window_index_multi, reverses, window_mask = self._group_layout(grid_group, seq, smu)
        if pos_idx.size(0) != seq:
            raise RuntimeError(
                f"batched vision position rows {pos_idx.size(0)} != sequence {seq}"
            )
        _vm("a_group_layout")

        # Pack the group's patches into one [seq, K] slab. Per-row projection
        # commutes with window reorder; weight selection follows the input dtype.
        embed_rpu = self._patch_embed(pixel_values, window_index_multi, seq, smu)
        _vm("b_patch_embed")
        self._keepalive.narrow(0, 0, seq).copy_(pos_idx.to(self._keepalive.device))

        self.cache.reset_to_position(0)
        k_caches = [self.cache.k_caches[i] for i in range(self.num_layers)]
        v_caches = [self.cache.v_caches[i] for i in range(self.num_layers)]
        _vm("c_keepalive+reset")

        # _patch_embed may run eager RPU Linear + gather, both before capture.
        # Re-establish the fused graph's deterministic temporary-SPM entry state.
        torch.ops.rpu.spm_alloc_reset_temporary()
        sig = rpu_backend.graph.GraphSignature(
            op_id="wall_oss_vision",
            shapes=[seq, self.hidden_size],
            # g distinguishes 1×768 (single big image) from g×256 (batched) — they
            # take different SDPA branches and must NOT share a cached graph.
            dyn_dims=[self.num_layers, self._native_chunk_size(grid_group),
                      int(self._window), g,
                      layer_group if grouped_chunks else 0,
                      int(weight_outer), int(self._mixed_precision),
                      int(torch.rpu.get_debug_export())],
            dtypes=[torch.float16],
        )
        with self._graph_cache.capture(sig):
            out = torch.ops.rpu.qwen25vl_vision_forward(
                self._handle, embed_rpu, k_caches, v_caches, seq, window_mask, g,
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
                gri = global_rev.to(torch.int32).to("rpu").contiguous()  # tiny (seq/4 int32)
                if fused:
                    # Defer the reverse to the runtime's fused assemble op (folds the
                    # window-reverse gather + the per-run scatter into one op).
                    self._pending_rev = (gri, [mu] * g)
                    return merged                                       # window order
                # Keep the reverse on device. The gather needs ~seq/4·out_hidden of
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
    per_window_sdpa: bool = False,
) -> WallOssVision:
    """Load `visual.*` weights and install the RPU Qwen2.5-VL ViT vision tower.

    window=False: every layer is full attention (MASK_NONE), no reorder.
    window=True: block-diagonal window attention with fullatt_block_indexes
    as full-attn layers + CPU get_window_index reorder/reverse.

    ``ckpt_dir``/``fp16_ckpt_dir`` default to ``None`` → resolved at call time
    under the current ``$RPU_MODEL_CACHE`` (W8A16 vs fp16 registry path).
    """
    from rpu_backend.api._execution import normalize_rpu_execution
    vision_execution = normalize_rpu_execution(
        {
            "vision": ({} if execution_config is None else
                       execution_config.get("vision", {}))
        },
        entry_point="build_wall_oss_vision",
        supported={"vision": ("chunk_size",)},
    ).get("vision", {})
    execution_chunk_size = vision_execution.get("chunk_size", "auto")
    for name, value in (("window", window), ("w8a16", w8a16),
                        ("per_window_sdpa", per_window_sdpa)):
        if not isinstance(value, bool):
            raise ValueError(
                f"Wall-OSS vision {name} must be bool, got {value!r}"
            )
    if per_window_sdpa and not window:
        raise ValueError("per_window_sdpa=True requires window=True")
    experimental_schedule = _claim_experimental_schedule()
    if per_window_sdpa and (
        experimental_schedule != (0, False)
        or rpu_env_bool(
            "RPU_WALL_OSS_VISION_MIXED_PRECISION",
            cpp_mirror="src/fused/rpu_qwen25vl_vision_model.cpp",
        )
    ):
        raise ValueError(
            "per-window Vision SDPA is incompatible with mixed precision, "
            "layer grouping, and weight-outer scheduling"
        )
    ckpt_dir = _resolve_ckpt(ckpt_dir, w8a16)
    fp16_ckpt_dir = _resolve_ckpt(fp16_ckpt_dir, False)
    mixed_w8a16 = w8a16 and rpu_env_bool(
        "RPU_WALL_OSS_VISION_MIXED_PRECISION",
        cpp_mirror="src/fused/rpu_qwen25vl_vision_model.cpp",
    )
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
    INTER_PAD = _INTER_PAD_W8A16 if w8a16 else _INTER_PAD

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
            os = gr(f"{p}.attn.proj.weight_scale")

        gu_w = gr(f"{p}.mlp.gate_up_proj.weight")
        gu_b = gf(f"{p}.mlp.gate_up_proj.bias")
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
            o_s.append(_to_rpu_half(os))
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
    graph_cache = rpu_backend.graph.GraphCache()

    # patch_embed: fold Conv3d [HID,3,2,14,14] → Linear [HID, 1176], CPU fp32, no bias.
    patch_w = g_cpu("visual.patch_embed.proj.weight").reshape(HID, -1).contiguous()

    # merger: RMSNorm (CPU fp32) → reshape(-1, 4*HID) → Linear → GELU → Linear.
    # The two large Linears run on RPU fp16. Weights are col-partition swizzled
    # (8-core) to match rpu_linear's rpu_get_linear_partition choice (both shapes
    # satisfy can_row && can_col → returns col). RMSNorm stays CPU fp32 so
    # variance is accumulated in fp32.
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
        xr = F.linear(xr, m0_w_rpu, m0_b_rpu)        # rpu_linear (col) → [N, 5120]
        xr = F.gelu(xr)                              # graph-aware CPU exact-erf fallback
        xr = F.linear(xr, m2_w_rpu, m2_b_rpu)        # → [N, 2048]
        return xr if device_out else xr.float().cpu()   # device fp16 OR CPU fp32 (legacy)

    # Prepare the col-swizzled patch_embed weight with a 16-aligned K dimension.
    _pe_k = ((patch_w.size(1) + 15) // 16) * 16
    patch_w_rpu = _to_rpu_half(tp_col_swizzle_mc_weight(
        F.pad(patch_w.to(torch.float16), (0, _pe_k - patch_w.size(1))), _VISION_CORES))

    # The raw handle remains builder-owned until WallOssVision installs its
    # tracked finalizer. Any native configuration or object-construction error
    # therefore has one deterministic cleanup path instead of waiting for GC.
    handle, keepalive = _configure_qwen25vl_vision_handle(
        set_weights=set_weights,
        weight_args=weight_args,
        freq_cos=freq_cos,
        freq_sin=freq_sin,
        merger_args=merger_args,
        chunk_size=(
            0 if execution_chunk_size == "auto"
            else int(execution_chunk_size)
        ),
        per_window_sdpa=per_window_sdpa,
    )
    transferred = False
    try:
        vis = WallOssVision(
            handle=handle, cache=cache, graph_cache=graph_cache,
            keepalive=keepalive, patch_w=patch_w, merger=merger,
            hidden_size=HID, num_layers=num_layers, spatial_merge_size=SMS,
            window=window, window_size=WIN_SIZE, patch_size=PATCH,
            experimental_schedule=experimental_schedule,
            execution_chunk_size=execution_chunk_size,
            per_window_sdpa=per_window_sdpa,
        )
        transferred = True
        vis._patch_w_rpu = patch_w_rpu
        vis._pe_k = _pe_k
        return vis
    finally:
        if not transferred:
            _destroy_qwen25vl_vision_handle(handle)
