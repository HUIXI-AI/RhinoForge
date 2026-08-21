"""Wall-OSS-0.5 expert-0 (Qwen2.5 text decoder) for RPU.

A text-only token sequence routes every token to expert-0, so the Wall-OSS MoT
decoder reduces to a standard Qwen2.5-VL text decoder. This adapter drives that decoder on
RPU through the extended `causal_decoder_*` op family:

  * Qwen2.5 == Qwen3 decoder **minus QK-norm, plus QKV bias** → we pass empty
    q/k/v_norm lists (`has_qk_norm_=false`) and populated q/k/v_bias lists
    (`has_qkv_bias_=true`; fused for FP16/W8/W4, separate SPM add for NVFP4).
  * `num_kv_heads=2` → the C++ side derives `attn_tp()=min(8,2)=2` and runs the
    whole attention phase tensor-parallel over 2 cores; the KV cache must be
    declared with the matching `attn_tp=2`.
  * mRoPE (`mrope_section=[16,24,24]`). For text-only the three position axes are
    equal, so mRoPE degenerates numerically to standard 1D RoPE.

Checkpoint weight layout: fused `qkv_proj_experts.0` `[2560,2048]` (+bias)
splits into q`[2048]`/k`[256]`/v`[256]`; `o_proj_experts.0` `[2048,2048]`; fused
`moe.experts.0.gate_up_proj` `[22016,2048]` → gate/up `[11008,2048]`; `down_proj`
`[2048,11008]`. Per-layer norms `input_layernorms.0` / `post_attention_layernorms.0`,
final `norms.0`, `embed_tokens` `[155765,2048]`.
"""
from __future__ import annotations

import json
import os
import struct
import weakref
from numbers import Integral
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open

import rpu_backend
from rpu_backend.runtime import rpu_env_bool
from rpu_backend.runtime.rope_partial import build_chunked_mrope_cos_sin
from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime.weights import (
    tp_col_swizzle_mc_weight,
    tp_row_swizzle_mc_weight,
)
from rpu_backend.adapters.wall_oss.checkpoint_validation import (
    validate_rtc_w4_checkpoint_pair,
    validate_w8_nvfp4_checkpoint_pair,
)
from rpu_backend.quant.int4_pack import swizzle_pack_int4
from rpu_backend.quant.int4_pgrp_pack import (
    swizzle_int4_pgrp_scale,
    swizzle_pack_int4_pgrp,
)
from rpu_backend.quant.nvfp4_pack import quantize_nvfp4, swizzle_pack_nvfp4

# attn_tp for the action/text expert (num_kv_heads=2). Q/K/V/O swizzle + KV cache
# + the C++ attention phase all run at this tensor-parallel factor; MLP stays 8.
_ATTN_TP = 2
_MLP_CORES = 8
_PREFILL_CHUNK_SIZE_CAP_SAFE = 256
_PREFILL_CHUNK_SIZE_CAP_TP8_PAD16 = 384
_PREFILL_CHUNK_POLICY_VERSION = 4


def _kvinsert_pad16_enabled() -> bool:
    return rpu_env_bool("RPU_WALL_OSS_KVINSERT_PAD16", default=True)


def _prefill_mixed_precision_enabled() -> bool:
    return rpu_env_bool(
        "RPU_WALL_OSS_PREFILL_MIXED_PRECISION",
        cpp_mirror="src/fused/rpu_qwen3_model.h",
    )


def _eff_attn_tp(nkv_real: int) -> int:
    """Effective attention tensor-parallel factor (= number of attention cores).

    Default = ``nkv_real`` (``attn_tp=min(8,nkv)`` gives 2 cores for
    Wall-OSS). With ``RPU_WALL_OSS_ATTN_TP8=1`` it returns 8: the K/V heads are
    replicated ``8//nkv_real``× (see :func:`_replicate_kv`) so the whole attention
    phase uses the ``num_cores=8`` col-swizzle / ``nkv=8`` SDPA path. ``8 %
    nkv_real`` must be 0."""
    if (
        isinstance(nkv_real, bool)
        or not isinstance(nkv_real, Integral)
        or nkv_real < 1
    ):
        raise ValueError(
            f"num_key_value_heads must be a positive integer, got {nkv_real!r}"
        )
    nkv_real = int(nkv_real)
    if not rpu_env_bool("RPU_WALL_OSS_ATTN_TP8"):
        return nkv_real
    if _MLP_CORES % nkv_real != 0:
        raise ValueError(
            f"ATTN_TP8 needs {_MLP_CORES} % nkv({nkv_real}) == 0"
        )
    return _MLP_CORES


def _prefill_chunk_size_cap(attn_tp: int) -> int:
    """Return the build-time auto-planner ceiling for text prefill.

    The 384-token layout is supported by the SPM envelope only for TP8, and
    padding keeps the equal-two long-prefix plan legal. Keep the established
    256-token ceiling for TP2 or an explicitly unpadded prefix.
    """
    if attn_tp == _MLP_CORES and _kvinsert_pad16_enabled():
        return _PREFILL_CHUNK_SIZE_CAP_TP8_PAD16
    return _PREFILL_CHUNK_SIZE_CAP_SAFE


def _equal_two_chunk_supported(
    chunk_size: int, execution_len: int, geom: dict
) -> bool:
    """Whether Wall TP8 LTM can safely execute two equal FP16 chunks.

    This delegates to ``sdpa_helpers.is_valid_chunk_size``, the same native
    predicate used by the auto planner and ``compute_chunks_impl``. Feasibility
    is a property of ``(chunk_size, execution_len)`` because accumulated calls
    have different grid geometry. C288 stays excluded separately because its
    ACC16 first chunk is outside the strict prefix contract represented here.
    """
    if chunk_size == 288:
        return False
    from rpu_backend._cpp_ext import sdpa_helpers as h
    return bool(h.is_valid_chunk_size(
        **geom, cs=chunk_size, seq_len=execution_len, position=0))


def _prefill_execution_plan(
    real_len: int,
    chunk_size_cap: int,
    *,
    pad16: bool,
    equal_two: bool,
    geom: dict,
) -> tuple[int, int]:
    """Return ``(execution_len, planned_chunk_size)`` before embedding assembly.

    A prefix that fits the supported single-chunk cap keeps the existing 16-token
    KV-insert alignment. Longer Wall TP8 prefixes use the smallest 32-aligned
    execution length, so the two chunks are both 16-aligned and exactly equal.

    This helper owns the Wall-specific execution-length policy.  A live model
    must still ask the native dry resolver for the actual chunk size because
    the full decoder's SPM and kernel-validity gates can tighten it below this
    policy ceiling.
    """
    if real_len <= 0:
        raise ValueError(f"prefill length must be positive, got {real_len}")
    if not pad16:
        return real_len, min(real_len, chunk_size_cap)

    padded16 = ((real_len + 15) // 16) * 16
    if not equal_two or padded16 <= chunk_size_cap:
        return padded16, min(padded16, chunk_size_cap)

    execution_len = ((real_len + 31) // 32) * 32
    while execution_len <= 2 * chunk_size_cap:
        chunk_size = execution_len // 2
        if _equal_two_chunk_supported(chunk_size, execution_len, geom):
            return execution_len, chunk_size
        execution_len += 32
    raise ValueError(
        "Wall-OSS equal-two prefill has no safe two-chunk execution profile: "
        f"real_len={real_len}, chunk_size_cap={chunk_size_cap}, "
        f"max_execution_len={2 * chunk_size_cap}"
    )


def _prefill_graph_signature(
    op_id: str,
    seq: int,
    hidden_size: int,
    planned_chunk_size: int,
    num_layers: int,
    mixed_precision: bool = False,
):
    """Build the prefill cache key, including the execution-policy version."""
    return rpu_backend.graph.GraphSignature(
        op_id=op_id,
        shapes=[seq, hidden_size, planned_chunk_size],
        dyn_dims=[
            num_layers,
            _PREFILL_CHUNK_POLICY_VERSION,
            int(mixed_precision),
        ],
        dtypes=[torch.float16],
    )


def _partial_mrope_enabled() -> bool:
    """Require the Qwen2.5-VL chunked host-table M-RoPE path.

    The legacy C++ launcher implements Qwen3-style THW interleaving, which is
    only equivalent for text positions where T == H == W. It is incorrect for
    Wall-OSS image positions, so disabling the host-table path is not a valid
    fallback.
    """
    if not rpu_env_bool("RPU_WALL_OSS_PARTIAL_MROPE", default=True):
        raise RuntimeError(
            "RPU_WALL_OSS_PARTIAL_MROPE=0 is unsupported: Wall-OSS/Qwen2.5-VL "
            "requires contiguous T/H/W M-RoPE tables; the legacy C++ path uses "
            "Qwen3-style interleaving"
        )
    return True


def _replicate_kv(t: torch.Tensor, nkv_real: int, hd: int, rep: int) -> torch.Tensor:
    """Replicate each of ``nkv_real`` KV heads ``rep``× along the head axis (eff head
    e = real head e//rep). A real KV head maps onto a contiguous group of ``rep``
    effective heads, so the C++ SDPA GQA routing (Q head q → eff KV q//(NQ/NKV_eff))
    still reaches the correct real head. Accepts a 2D weight ``[nkv_real*hd, K]`` and
    a 1D bias/scale ``[nkv_real*hd]``; ``rep==1`` is a no-op."""
    if rep == 1:
        return t
    if t.dim() == 2:
        return (t.view(nkv_real, hd, -1).repeat_interleave(rep, dim=0)
                .reshape(nkv_real * rep * hd, -1).contiguous())
    return (t.view(nkv_real, hd).repeat_interleave(rep, dim=0)
            .reshape(nkv_real * rep * hd).contiguous())
# Checkpoint dirs are resolved at CALL time (not import time) under the *current*
# $RPU_MODEL_CACHE via the model registry — never a hardcoded filesystem root and
# never frozen at import. Builders default ckpt_dir=None and call these resolvers,
# so `import → set RPU_MODEL_CACHE → build(...)` picks up the new root.
from rpu_backend.model_registry import model_path as _model_path


def _default_ckpt() -> str:
    return str(_model_path("wall-oss-0.5"))


def _default_w8a16_ckpt() -> str:
    return str(_model_path("wall-oss-0.5-w8a16"))


def _resolve_ckpt(ckpt_dir: str | None, w8a16: bool) -> str:
    """Resolve a wall_oss checkpoint dir under the current $RPU_MODEL_CACHE.

    Explicit ``ckpt_dir`` always wins. Otherwise pick the W8A16 or fp16 registry
    path for the requested precision at call time.
    """
    if ckpt_dir is not None:
        return ckpt_dir
    return _default_w8a16_ckpt() if w8a16 else _default_ckpt()


_ST_DTYPE = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


class _SafeTensorStore:
    """Small safetensors reader that supports single-file and indexed shards.

    ``mmap=True`` uses safetensors ``safe_open``. ``mmap=False`` uses buffered
    reads with a transient per-tensor buffer. Both modes return identical tensors.
    """

    def __init__(self, ckpt_dir: str, mmap: bool = True):
        self.root = Path(ckpt_dir)
        self.mmap = mmap
        index_path = self.root / "model.safetensors.index.json"
        if index_path.is_file():
            self.weight_map = dict(json.loads(index_path.read_text())["weight_map"])
        elif (self.root / "model.safetensors").is_file():
            with safe_open(self.root / "model.safetensors", framework="pt", device="cpu") as f:
                self.weight_map = {name: "model.safetensors" for name in f.keys()}
        else:
            raise FileNotFoundError(f"no safetensors checkpoint found in {self.root}")
        self._handles = {}
        self._headers = {}  # shard -> (header_dict, data_base_offset) for mmap=False

    def _header(self, shard: str):
        hdr = self._headers.get(shard)
        if hdr is None:
            with open(self.root / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                meta = json.loads(f.read(n))
            hdr = (meta, 8 + n)
            self._headers[shard] = hdr
        return hdr

    def get_tensor(self, name: str) -> torch.Tensor:
        try:
            shard = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"missing tensor {name}") from exc
        if self.mmap:
            handle = self._handles.get(shard)
            if handle is None:
                handle = safe_open(self.root / shard, framework="pt", device="cpu")
                self._handles[shard] = handle
            return handle.get_tensor(name)
        # Non-mmap: plain read of the tensor's byte range into an anonymous
        # buffer. Page cache used here is reclaimable and not charged to RSS.
        header, base = self._header(shard)
        meta = header[name]
        begin, end = meta["data_offsets"]
        dtype = _ST_DTYPE[meta["dtype"]]
        buf = bytearray(end - begin)
        with open(self.root / shard, "rb") as f:
            f.seek(base + begin)
            f.readinto(buf)
        return torch.frombuffer(buf, dtype=dtype).reshape(meta["shape"])


def _build_rope_tables(head_dim: int, rope_theta: float, max_seq_len: int):
    """Kernel-format 1D RoPE cos/sin `[max_seq_len, head_dim/2]` (fp16, RPU).

    Identical formula to transformers' `Qwen2_5_VLRotaryEmbedding` (default rope,
    attention_scaling=1.0). The per-axis (T/H/W) mRoPE selection is applied by
    the kernel via strobe masks derived from `mrope_section`; this table is the
    plain rotary basis. It is built in fp64 and cast to the fp16 kernel format.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    )  # [half]
    positions = torch.arange(max_seq_len, dtype=torch.float64)  # [max_seq]
    freqs = positions[:, None] * inv_freq[None, :]  # [max_seq, half]
    cos = freqs.cos().to(torch.float16).to("rpu").contiguous()
    sin = freqs.sin().to(torch.float16).to("rpu").contiguous()
    expected_shape = (max_seq_len, half)
    if tuple(cos.shape) != expected_shape or tuple(sin.shape) != expected_shape:
        raise RuntimeError(
            "Wall-OSS RoPE table shape mismatch: "
            f"cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, "
            f"expected={expected_shape}"
        )
    return cos, sin


def _to_rpu_half(t: torch.Tensor) -> torch.Tensor:
    return t.to(dtype=torch.float16, device="rpu").contiguous()


def _to_rpu_int8(t: torch.Tensor) -> torch.Tensor:
    return t.to(dtype=torch.int8, device="rpu").contiguous()


def _pack_int4_rpu(w_int4: torch.Tensor, partition: int, num_cores: int,
                   group_wise: bool = False) -> torch.Tensor:
    # int4 values [N,K] (int8 container) -> packed uint8 [N, K/2] on RPU. The packer
    # returns a flat [N*K/2] buffer; the fused set_weights + wINT4 kernel both require
    # the 2D [N, K/2] shape (mirrors adapters/pi05/w4pack.py). group_wise selects the
    # pgrp swizzle (deinterleaved K-group) for the tiled wint4a16_pgrp kernels.
    N, K = w_int4.shape
    pack = swizzle_pack_int4_pgrp if group_wise else swizzle_pack_int4
    packed = pack(w_int4, partition, num_cores).reshape(N, K // 2)
    return packed.to(dtype=torch.uint8, device="rpu").contiguous()


def _pack_nvfp4_rpu(
    weight: torch.Tensor, partition: int, num_cores: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one logical [N,K] projection and return packed weight/FP8 scale/TS."""
    n, k = weight.shape
    codes, block_scale, tensor_scale = quantize_nvfp4(weight)
    packed = swizzle_pack_nvfp4(codes, partition, num_cores).reshape(n, k // 2)
    return (
        packed.to(dtype=torch.uint8, device="rpu").contiguous(),
        block_scale.to(dtype=torch.uint8, device="rpu").contiguous(),
        tensor_scale.cpu(),
    )


def _nvfp4_tensor_scale_array(values: list[torch.Tensor]) -> torch.Tensor:
    # Canonical Rhino NVFP4 ACC16 reads one FP32 value by layer_id from a
    # projection-family array through one aligned 32-byte vector load. Pad the
    # source to a whole eight-fp32 block so the persistent preload initializes
    # every byte fetched by the kernel (and remains legal for 1..7-layer probes).
    scales = torch.stack(values).to(dtype=torch.float32)
    padded_layers = ((scales.numel() + 7) // 8) * 8
    if padded_layers != scales.numel():
        scales = F.pad(scales, (0, padded_layers - scales.numel()))
    return scales.to(device="rpu").contiguous()


def _int8_to_int4_perchannel(w_int8: torch.Tensor, scale_1d: torch.Tensor):
    """Re-quantize per-output-channel INT8 weights (±127) to symmetric per-channel
    INT4 (±7) on the fly. Returns (int4 values in int8 container [N,K], fp16 scale [N]).
    LOSSY (int8->int4). Used for the pgrp ckpt's expert-0 down_proj, which is stored
    per-channel int8 while the other projections are true int4 — enabling int4 down
    for the optional prefill weight-DMA path
    (RPU_WALL_OSS_EXPERT0_DOWN_INT4=1). Double-quant
    (fp->int8->int4): precision degrades to int4."""
    w = w_int8.to(torch.float32)                                 # int8 integer values
    amax = w.abs().amax(dim=1, keepdim=True).clamp_min_(1.0)     # [N,1] per output channel
    q = torch.round(w * (7.0 / amax)).clamp_(-7, 7).to(torch.int8)
    new_scale = (scale_1d.to(torch.float32).view(-1, 1) * (amax / 7.0)).squeeze(1).to(torch.float16)
    return q, new_scale


def _w4_is_group_wise(ckpt_dir: str) -> bool:
    """True if the w4a16 checkpoint is group-wise (pgrp/AWQ) vs per-channel, read
    from rpu_quant_config.json (mode == 'group_wise_symmetric')."""
    qc = Path(ckpt_dir) / "rpu_quant_config.json"
    return qc.is_file() and json.loads(qc.read_text()).get("mode") == "group_wise_symmetric"


def _expert0_down_int4() -> bool:
    """Pack expert-0 down_proj as per-channel int4 (INTER padded to a %512 multiple, half
    weight DMA) vs keep it int8 (full DMA). RPU_WALL_OSS_EXPERT0_DOWN_INT4, default ON.
    per-channel w4a16 ckpt: down is already int4 VALUES → lossless pack toggle.
    pgrp ckpt: down is stored per-channel INT8 (±127; other proj are group-wise int4) →
    _int8_to_int4_perchannel re-quantizes int8→int4 on the fly (LOSSY double-quant).
    The switch keeps this conversion optional."""
    return rpu_env_bool("RPU_WALL_OSS_EXPERT0_DOWN_INT4", default=True)


def _chan_first_scale(s: torch.Tensor) -> torch.Tensor:
    """group-wise pgrp scale [G, N] -> [N, G] (channels on dim 0) so the same
    split/slice/replicate code as the per-channel [N] path applies. 1D unchanged."""
    return s.t().contiguous() if s.dim() == 2 else s


def _scale_to_rpu(
    s: torch.Tensor, *, in_features: int, partition: int, num_cores: int
) -> torch.Tensor:
    """Upload one per-channel scale or controller-swizzle a pgrp ``[N,G]`` scale."""
    if s.dim() != 2:
        return _to_rpu_half(s)
    logical = s.t().contiguous()
    groups = logical.size(0)
    if in_features % groups:
        raise ValueError(
            f"pgrp scale G={groups} does not divide K={in_features}"
        )
    physical = swizzle_int4_pgrp_scale(
        logical, in_features // groups, partition, num_cores
    )
    return _to_rpu_half(physical)


def _destroy_causal_decoder_handle(h):
    """Release the C++ CausalDecoderModel handle. Called by weakref.finalize on GC."""
    try:
        torch.ops.rpu.causal_decoder_destroy(h)
    except Exception:
        pass


# Certified chunk envelope for the Wall-OSS decoders.
#
# The chunk half is the cap this adapter computes in _prefill_chunk_size_cap().
# The envelope also enforces it for explicit chunk-size overrides.
#
# The length half is the RPUCache capacity these handles are built against. This
# is the 2D-rect-mask path, so kv length is genuinely SPM-load-bearing here —
# declare_buffers adds sdpa_mask = comp_cs * ceil16(kv) * 32, and the cache
# bound is the limit the adapter already enforces.
_CHUNK_ENVELOPE_MAX_KV = 8192


def _configure_causal_decoder_handle(
    *, set_weights, weight_args, chunk_size_cap, equal_two_prefill,
    chunk_size_override=0
):
    """Create/configure a decoder handle, destroying it on any failed gate."""
    handle = torch.ops.rpu.causal_decoder_create()
    configured = False
    try:
        set_weights(handle, *weight_args)
        torch.ops.rpu.causal_decoder_set_chunk_size_cap(
            handle, chunk_size_cap
        )
        torch.ops.rpu.causal_decoder_set_chunk_envelope(
            handle, _CHUNK_ENVELOPE_MAX_KV, chunk_size_cap
        )
        torch.ops.rpu.causal_decoder_set_equal_two_prefill(
            handle, equal_two_prefill
        )
        # The cap/equal-two setters reset the override, so exact public control
        # must be installed last.  Zero preserves the native auto planner.
        torch.ops.rpu.causal_decoder_set_chunk_size_override(
            handle, chunk_size_override
        )
        configured = True
        return handle
    finally:
        if not configured:
            _destroy_causal_decoder_handle(handle)


class WallOssLLM:
    """Expert-0 Qwen2.5 text decoder on RPU.

    Construct via :func:`build_wall_oss_llm`. Call :meth:`forward` with CPU
    `input_ids` `[1, seq]`; returns the final-normed `last_hidden` `[1, seq, H]`
    on the RPU device.
    """

    def __init__(self, *, handle, cache, graph_cache, embed_w, hidden_size,
                 num_layers, head_dim, rope_theta, mrope_section,
                 prefill_chunk_size_cap, prefill_pad16, prefill_equal_two,
                 attn_geometry, partial_mrope=None, execution_config=None,
                 prefill_mixed_precision=False):
        partial_mrope = (
            _partial_mrope_enabled()
            if partial_mrope is None else bool(partial_mrope)
        )
        self._handle = handle
        # partial_mrope: precompute Qwen2.5-VL chunked cos/sin on the host.
        self._head_dim = head_dim
        self._rope_theta = rope_theta
        self._mrope_section = mrope_section
        self._partial_mrope = partial_mrope
        self.cache = cache
        self._graph_cache = graph_cache
        self._embed_w = embed_w          # [vocab, hidden] fp32 CPU
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._prefill_chunk_size_cap = prefill_chunk_size_cap
        self._prefill_pad16 = prefill_pad16
        self._prefill_equal_two = prefill_equal_two
        # Attention geometry AS THE C++ DECODER SEES IT (post KV replication),
        # so the host prefill planner can ask the real chunk predicate rather
        # than re-deriving the kernel's tiling rules. See
        # :func:`_equal_two_chunk_supported`.
        self._attn_geometry = dict(attn_geometry)
        self._rpu_execution = (
            {} if execution_config is None else execution_config
        )
        self._prefill_mixed_precision = bool(prefill_mixed_precision)
        # Memoized prefill chunked cos/sin (partial_mrope), keyed on seq. cos/sin
        # are a pure function of position_ids, which for a fixed prompt structure is
        # f(seq) only. Recurring prefill lengths reuse one entry;
        # RPU_WALL_OSS_HOST_CACHE=0 disables the cache.
        self._prefill_rope_cache = {}
        # Stable per-seq RPU position tensors. CausalDecoderModel memoizes the
        # source pointer used to refresh its keepalive; retaining the source
        # prevents allocator address reuse from making changed contents look
        # like a pointer cache hit (notably synthetic prewarm -> first real
        # request). The CPU copy validates same-seq content before reuse.
        self._prefill_pos_cache = {}
        self._last_prefill_pos_ref = None
        self._last_prefill_rope_ref = None
        self._host_cache = rpu_env_bool("RPU_WALL_OSS_HOST_CACHE", default=True)
        self._closed = False
        self._handle_finalizer = weakref.finalize(
            self, _destroy_causal_decoder_handle, handle
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
            torch.ops.rpu.causal_decoder_destroy(self._handle)
            finalizer.detach()
        self._closed = True

    def prefill_execution_plan(
        self, real_len: int, *, reserve_rows: int = 0
    ) -> tuple[int, int]:
        if (
            isinstance(reserve_rows, bool)
            or not isinstance(reserve_rows, int)
            or reserve_rows < 0
        ):
            raise ValueError(
                f"Wall-OSS reserve_rows must be a non-negative integer, got "
                f"{reserve_rows!r}"
            )
        physical_limit = min(
            int(self.cache.max_seq_len), _CHUNK_ENVELOPE_MAX_KV
        ) - reserve_rows
        if physical_limit <= 0:
            raise ValueError(
                "Wall-OSS cache has no room for prefill after reserving the "
                f"action suffix: max_seq_len={self.cache.max_seq_len}, "
                f"reserve_rows={reserve_rows}"
            )
        stage = self._rpu_execution.get("prefill", {})
        if self._prefill_mixed_precision and real_len == 672:
            requested = stage.get("chunk_size", "auto")
            if requested not in ("auto", 336):
                raise ValueError(
                    "Wall-OSS mixed P672 prefill requires chunk_size='auto' "
                    f"or 336, got {requested!r}"
                )
            if 672 > physical_limit:
                raise ValueError(
                    "Wall-OSS mixed P672 prefill plus reserved action rows "
                    "exceeds the shared KV cache: "
                    f"reserve_rows={reserve_rows}, "
                    f"max_seq_len={self.cache.max_seq_len}"
                )
            resolved = int(
                torch.ops.rpu.causal_decoder_resolve_prefill_chunk_size(
                    self._handle, 672, 0
                )
            )
            if resolved != 336:
                raise RuntimeError(
                    "Wall-OSS mixed P672 native planner must resolve C336, "
                    f"got C{resolved}"
                )
            return 672, resolved
        if (
            stage.get("chunk_size", "auto") == "auto"
            and "padding_rows" not in stage
            and "padding_budget" not in stage
        ):
            # Preserve the Wall default execution length byte-for-byte
            # when no public padding policy was requested.  The policy helper's
            # chunk is only a ceiling/equal-two target: the full native planner
            # can choose a smaller value once it applies the model's real SPM
            # and kernel-validity gates (TP2 P192 resolves to C96, for example).
            execution_len, _ = _prefill_execution_plan(
                real_len,
                self._prefill_chunk_size_cap,
                pad16=self._prefill_pad16,
                equal_two=self._prefill_equal_two,
                geom=self._attn_geometry,
            )
            if execution_len > physical_limit:
                raise ValueError(
                    "Wall-OSS prefill execution plus reserved action rows "
                    "exceeds the shared KV cache: "
                    f"real_len={real_len}, execution_len={execution_len}, "
                    f"reserve_rows={reserve_rows}, "
                    f"max_seq_len={self.cache.max_seq_len}"
                )
            chunk_size = int(
                torch.ops.rpu.causal_decoder_resolve_prefill_chunk_size(
                    self._handle, execution_len, 0
                )
            )
            return execution_len, chunk_size

        from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
        return plan_bounded_prefill_execution(
            real_len,
            physical_limit,
            int(stage.get("padding_budget", 64)),
            lambda execution_len: (
                torch.ops.rpu.causal_decoder_resolve_prefill_chunk_size(
                    self._handle, execution_len, 0
                )
            ),
            alignment=16 if self._prefill_pad16 else 1,
            padding_rows=stage.get("padding_rows", "auto"),
            exact_chunk_size=(
                stage.get("chunk_size")
                if isinstance(stage.get("chunk_size"), int)
                else None
            ),
        )

    def _bake_prefill_cos_sin(self, position_ids: torch.Tensor):
        """Host-bake Qwen2.5-VL chunked cos/sin [seq, hd/2] → RPU fp16."""
        cos_cpu, sin_cpu = build_chunked_mrope_cos_sin(
            position_ids.to(torch.int64), head_dim=self._head_dim,
            rope_theta=self._rope_theta, mrope_section=self._mrope_section)
        return cos_cpu.to("rpu").contiguous(), sin_cpu.to("rpu").contiguous()

    def _prefill_position_ids_rpu(
        self, position_ids: torch.Tensor, seq: int
    ) -> torch.Tensor:
        """Return a stable RPU int32 position tensor, validated by full content."""
        cached = self._prefill_pos_cache.get(seq) if self._host_cache else None
        if cached is not None and torch.equal(cached[0], position_ids):
            return cached[1]
        # Allocate while ``cached`` (or _last_prefill_pos_ref when caching is
        # disabled) still owns the previous tensor, so the RPU allocator cannot
        # recycle its address and fool the C++ source-pointer upload memo.
        pos = position_ids.to(dtype=torch.int32, device="rpu").contiguous()
        if self._host_cache:
            self._prefill_pos_cache[seq] = (position_ids.clone(), pos)
        else:
            self._last_prefill_pos_ref = pos
        return pos

    def _prefill_cos_sin(self, position_ids: torch.Tensor, seq: int):
        """Memoized Qwen2.5-VL chunked cos/sin tables in RPU fp16.

        The tables are a pure function of position_ids: cache by sequence length,
        then validate with ``torch.equal`` so a same-length frame with genuinely
        different positions cannot serve stale data.
        """
        c = self._prefill_rope_cache.get(seq) if self._host_cache else None
        if c is not None and torch.equal(c[0], position_ids):
            return c[1], c[2]
        cos_il, sin_il = self._bake_prefill_cos_sin(position_ids)
        if self._host_cache:
            self._prefill_rope_cache[seq] = (position_ids.clone(), cos_il, sin_il)
        else:
            # Keep the previous sources alive until the newly baked pair has
            # been allocated. Otherwise an allocator address reuse can fool the
            # C++ keepalive's source-pointer memo for same-shape A→B replays.
            self._last_prefill_rope_ref = (cos_il, sin_il)
        return cos_il, sin_il

    def _run_prefill(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor,
                     op_id: str, *, logical_len: int | None = None,
                     reserve_rows: int = 0) -> torch.Tensor:
        """Shared LTM prefill body: inputs_embeds [1, seq, H] + mRoPE position_ids
        [seq, 3] → final-normed last_hidden, filling the cache prefix K/V at [0, seq)."""
        seq = int(inputs_embeds.size(1))
        if not isinstance(position_ids, torch.Tensor):
            raise TypeError(
                "position_ids must be a torch.Tensor, got "
                f"{type(position_ids).__name__}"
            )
        if tuple(position_ids.shape) != (seq, 3):
            raise ValueError(
                f"position_ids must be [seq, 3], got {tuple(position_ids.shape)}"
            )
        logical_len = seq if logical_len is None else int(logical_len)
        if logical_len <= 0 or logical_len > seq:
            raise ValueError(
                f"logical_len must be in [1, {seq}], got {logical_len}"
            )
        if _prefill_mixed_precision_enabled() != self._prefill_mixed_precision:
            raise RuntimeError(
                "Wall-OSS prefill mixed precision is fixed at LLM construction; "
                "rebuild the LLM instance after changing it"
            )
        # RTC maps every logical REAL599..671 prompt to the exact physical
        # P672/C336 profile. The stock/main planner still owns all other cases.
        plan_len = (
            seq
            if self._prefill_mixed_precision and seq == 672
            else logical_len
        )
        execution_len, planned_chunk_size = self.prefill_execution_plan(
            plan_len, reserve_rows=reserve_rows
        )
        supported_mixed_profile = (
            (seq == 640 and execution_len == 640
             and planned_chunk_size == 320)
            or (seq == 672 and execution_len == 672
                and planned_chunk_size == 336)
        )
        if self._prefill_mixed_precision and (
            op_id != "wall_oss_llm_embeds"
            or not supported_mixed_profile
        ):
            raise ValueError(
                "Wall-OSS prefill mixed precision is validated only for the RTC "
                "embedded P640/C320 or P672/C336 profile; got "
                f"op_id={op_id!r}, seq={seq}, execution_len={execution_len}, "
                f"planned_chunk_size={planned_chunk_size}"
            )
        if execution_len != seq:
            raise ValueError(
                "Wall-OSS prefill must be padded exactly once before the native "
                f"forward: logical_len={logical_len}, got execution_len={seq}, "
                f"planned_execution_len={execution_len}"
            )
        ie = _to_rpu_half(inputs_embeds)
        pos = self._prefill_position_ids_rpu(position_ids, seq)
        if tuple(pos.shape) != (seq, 3):
            raise RuntimeError(
                f"RPU position_ids must be [seq, 3], got {tuple(pos.shape)}"
            )

        # Host-bake Qwen2.5-VL's contiguous T/H/W frequency chunks [seq, hd/2].
        # The non-partial in-kernel strobe gather is Qwen3-style and is not equivalent
        # for mixed vision positions.
        # Memoized on seq (see _prefill_cos_sin) — frame-invariant for a fixed prefix length.
        cos_il, sin_il = self._prefill_cos_sin(position_ids, seq)

        # Fresh prefill from position 0. Use reset_to_position (no zero-fill) instead
        # of reset(): the LTM insert kernel overwrites K/V at [0, seq) before SDPA
        # reads it (same overwrite-before-read invariant reset_to_position documents
        # and the denoise loop already relies on), so re-zeroing all 36×2 cache
        # tensors every prefill is unnecessary.
        self.cache.reset_to_position(0)
        sig = _prefill_graph_signature(
            op_id,
            seq,
            self.hidden_size,
            planned_chunk_size,
            self.num_layers,
            self._prefill_mixed_precision,
        )
        with self._graph_cache.capture(sig):
            raw = torch.ops.rpu.causal_decoder_forward(
                self._handle, ie, self.cache.k_caches, self.cache.v_caches,
                None,                  # attention_mask=None → LTM (is_causal)
                self.cache.position,   # 0
                True,                  # is_causal
                pos,                   # mRoPE [seq, 3]
                [],                    # deepstack_dense_visual_embeds (none)
                cos_il, sin_il,        # partial_mrope Qwen2.5-VL chunked tables
            )
        resolved_chunk_size = int(
            torch.ops.rpu.causal_decoder_get_resolved_chunk_size(self._handle)
        )
        if resolved_chunk_size != planned_chunk_size:
            raise RuntimeError(
                "Wall-OSS prefill dry/forward chunk mismatch: "
                f"planned={planned_chunk_size}, resolved={resolved_chunk_size}, "
                f"logical_len={logical_len}, execution_len={seq}"
            )
        vars(self)["_rpu_last_execution_plan"] = {
            "stage": "prefill",
            "logical_len": logical_len,
            "execution_len": seq,
            "chunk_size": resolved_chunk_size,
            "padding_rows": seq - logical_len,
            "position": 0,
        }
        self.cache.update_position(seq)
        return raw

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, *, reserve_rows: int = 0
    ) -> torch.Tensor:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                "WallOssLLM.forward input_ids must be a torch.Tensor, got "
                f"{type(input_ids).__name__}"
            )
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError(
                "WallOssLLM.forward input_ids must be [1, seq], got "
                f"{tuple(input_ids.shape)}"
            )
        seq = int(input_ids.size(1))
        execution_len, _ = self.prefill_execution_plan(
            seq, reserve_rows=reserve_rows
        )
        # Embed on CPU in fp32, then move to RPU fp16.
        inputs_embeds = F.embedding(input_ids.cpu(), self._embed_w)  # [1, seq, H] fp32
        # Text-only → mRoPE 3 axes equal (T=H=W=index) → standard 1D RoPE.
        pos = torch.arange(seq, dtype=torch.int32)[:, None].expand(seq, 3).contiguous()
        if execution_len != seq:
            inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, execution_len - seq))
            pos = torch.cat(
                [pos, pos[-1:].expand(execution_len - seq, 3)], dim=0
            ).contiguous()
        raw = self._run_prefill(
            inputs_embeds,
            pos,
            "wall_oss_llm",
            logical_len=seq,
            reserve_rows=reserve_rows,
        )
        if execution_len != seq:
            self.cache.reset_to_position(seq)
            raw = raw[:, :seq]
        return raw

    @torch.no_grad()
    def forward_embeds(self, inputs_embeds: torch.Tensor,
                       position_ids: torch.Tensor, *,
                       logical_len: int | None = None,
                       reserve_rows: int = 0) -> torch.Tensor:
        """Prefill from pre-assembled inputs_embeds [1, seq, H] (vision scattered into
        image-token positions) + real 3D mRoPE position_ids [seq, 3] (vision grid +
        text). Used by the VLA orchestrator and fills the shared cache prefix K/V."""
        if not isinstance(inputs_embeds, torch.Tensor):
            raise TypeError(
                "forward_embeds inputs_embeds must be a torch.Tensor, got "
                f"{type(inputs_embeds).__name__}"
            )
        if inputs_embeds.dim() != 3 or inputs_embeds.size(0) != 1:
            raise ValueError(
                "forward_embeds inputs_embeds must be [1, seq, H], got "
                f"{tuple(inputs_embeds.shape)}"
            )
        raw = self._run_prefill(
            inputs_embeds,
            position_ids,
            "wall_oss_llm_embeds",
            logical_len=logical_len,
            reserve_rows=reserve_rows,
        )
        return raw


def _split_qkv(qkv: torch.Tensor, nq: int, nkv: int, hd: int):
    """Split fused `[(nq+2*nkv)*hd, ...]` into (q, k, v) along dim 0."""
    qd, kd = nq * hd, nkv * hd
    q = qkv[:qd].contiguous()
    k = qkv[qd:qd + kd].contiguous()
    v = qkv[qd + kd:qd + 2 * kd].contiguous()
    return q, k, v


def build_wall_oss_llm(
    ckpt_dir: str | None = None,
    *,
    expert: int = 0,
    layer_indices: Sequence[int] | None = None,
    max_seq_len: int = 2048,
    w8a16: bool = False,
    fp16_ckpt_dir: str | None = None,
    w4a16: bool = False,
    nvfp4a16: bool = False,
    w8_down_ckpt_dir: str | None = None,
    execution_config=None,
) -> WallOssLLM:
    """Load expert-`expert` weights and install the RPU text decoder.

    Args:
        ckpt_dir: Wall-OSS-0.5 checkpoint directory (model.safetensors + config.json).
            ``None`` (default) resolves at call time under the current
            ``$RPU_MODEL_CACHE``: the W8A16 registry path when ``w8a16=True``,
            else the fp16 path.
        expert: which MoT/MoE expert (0 = VL/text).
        layer_indices: layers to include, in order. Default = all
            `num_hidden_layers`. Pass e.g. `[0]` for isolated single-layer
            diagnostics (the C++ final-norm fuse applies after the last layer).
        max_seq_len: RoPE table + KV cache capacity.
        w8a16: load int8 weights + fp16 per-row scales from the W8A16 checkpoint
            and call `causal_decoder_set_weights_w8a16`.
        fp16_ckpt_dir: paired source checkpoint for the layer-2 FP16 down projection
            when RTC prefill mixed precision and W8A16/W4A16 are enabled.
        w8_down_ckpt_dir: W8A16 source for expert-0 ``down_proj`` only. Valid
            exclusively with ``expert=0, nvfp4a16=True`` so NVFP4 covers the
            same projection set as the retained WINT4A16 pgrp composition.
    """
    precision_modes = {
        "w8a16": w8a16,
        "w4a16": w4a16,
        "nvfp4a16": nvfp4a16,
    }
    invalid_modes = {
        name: value
        for name, value in precision_modes.items()
        if not isinstance(value, bool)
    }
    if invalid_modes:
        raise ValueError(
            f"Wall-OSS precision flags must be bool, got {invalid_modes}"
        )
    selected_modes = [name for name, enabled in precision_modes.items() if enabled]
    if len(selected_modes) > 1:
        raise ValueError(
            "pick one of w8a16 / w4a16 / nvfp4a16, got "
            f"{selected_modes}"
        )
    if w8_down_ckpt_dir is not None and (not nvfp4a16 or expert != 0):
        raise ValueError(
            "w8_down_ckpt_dir is valid only for expert=0 with nvfp4a16=True")
    if w4a16 and ckpt_dir is None:
        ckpt_dir = str(_model_path("wall-oss-0.5-w4a16"))
    else:
        ckpt_dir = _resolve_ckpt(ckpt_dir, w8a16)
    if nvfp4a16 and w8_down_ckpt_dir is not None:
        validate_w8_nvfp4_checkpoint_pair(w8_down_ckpt_dir, ckpt_dir)
    cfg = json.load(open(f"{ckpt_dir}/config.json"))
    H = cfg["hidden_size"]
    NQ = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    HD = cfg.get("head_dim") or (H // NQ)
    INTER = cfg["intermediate_size"]
    # W4A16 int4 down_proj: down is row-partition (K=INTER), and int4 row swizzle needs
    # K % (64*cores)==512==0. expert-1 INTER=2048 is aligned; expert-0 INTER=11008 is not
    # (11008%512=256) -> pad INTER to 11264 (+256 zero channels) so the int4 swizzle is
    # legal. Pad is bit-exact (gate/up pad rows -> silu(0)*0=0, down pad cols=0) and the
    # The fused decoder sizes the MLP SPM + down K from this padded INTER.
    # RPU_WALL_OSS_EXPERT0_DOWN_INT4=0 keeps expert-0 down in the int8 container
    # without padding.
    down_int4 = w4a16 and (expert != 0 or _expert0_down_int4())
    nvfp4_down_w8 = nvfp4a16 and w8_down_ckpt_dir is not None
    down_4bit = down_int4 or (nvfp4a16 and not nvfp4_down_w8)
    INTER_PAD = (((INTER + 511) // 512) * 512) if down_4bit else INTER
    EPS = cfg["rms_norm_eps"]
    THETA = cfg["rope_theta"]
    MROPE = list(cfg["rope_scaling"]["mrope_section"])  # [16, 24, 24]
    ATTN_TP = _eff_attn_tp(NKV)     # 2 (default) or 8 via RPU_WALL_OSS_ATTN_TP8
    # Snapshot the cap with the build-time attention/padding configuration. The
    # C++ setter rejects changes after the first forward, so existing handles and
    # their GraphCache entries cannot be hot-switched by later env changes.
    PREFILL_PAD16 = _kvinsert_pad16_enabled()
    PREFILL_CHUNK_SIZE_CAP = _prefill_chunk_size_cap(ATTN_TP)
    PREFILL_MIXED_PRECISION = _prefill_mixed_precision_enabled()
    execution_config = {} if execution_config is None else execution_config
    requested_chunk = execution_config.get("prefill", {}).get(
        "chunk_size", "auto"
    )
    CHUNK_SIZE_OVERRIDE = (
        0 if requested_chunk == "auto" else int(requested_chunk)
    )
    if CHUNK_SIZE_OVERRIDE > PREFILL_CHUNK_SIZE_CAP:
        raise ValueError(
            "Wall-OSS prefill chunk_size exceeds the certified ceiling: "
            f"requested={CHUNK_SIZE_OVERRIDE}, "
            f"ceiling={PREFILL_CHUNK_SIZE_CAP}"
        )
    PREFILL_EQUAL_TWO = (
        ATTN_TP == _MLP_CORES
        and PREFILL_PAD16
        and (
            CHUNK_SIZE_OVERRIDE == 0
            or (
                PREFILL_MIXED_PRECISION
                and CHUNK_SIZE_OVERRIDE in (320, 336)
            )
        )
    )
    REP = ATTN_TP // NKV            # KV-head replication factor (1 or 4)
    NKV_EFF = NKV * REP             # effective KV heads the C++ decoder sees (== ATTN_TP)

    if layer_indices is None:
        layer_indices = list(range(cfg["num_hidden_layers"]))
    else:
        layer_indices = list(layer_indices)
    num_layers = len(layer_indices)
    if PREFILL_MIXED_PRECISION:
        exact_profile = (
            expert == 0
            and not nvfp4a16
            and (not w4a16 or not down_int4)
            and layer_indices == list(range(36))
            and H == 2048 and NQ == 16 and NKV == 2 and HD == 128
            and INTER_PAD == 11008 and MROPE == [16, 24, 24]
            and ATTN_TP == 8 and NKV_EFF == 8
            and PREFILL_PAD16 and PREFILL_EQUAL_TWO
            and PREFILL_CHUNK_SIZE_CAP == 384
            and max_seq_len >= 640
        )
        if not exact_profile:
            raise ValueError(
                "RPU_WALL_OSS_PREFILL_MIXED_PRECISION=1 supports only the "
                "full 36-layer FP16/W8A16/W4A16 RTC expert-0 profile with logical "
                "H2048/I11008/NQ16/NKV2/HD128, TP8, pad16, and max_seq_len>=640"
            )

    mixed_quantized = PREFILL_MIXED_PRECISION and (w8a16 or w4a16)
    if mixed_quantized:
        fp16_ckpt_dir = _resolve_ckpt(fp16_ckpt_dir, False)
        if w4a16:
            validate_rtc_w4_checkpoint_pair(ckpt_dir, fp16_ckpt_dir)
        else:
            validate_w8_nvfp4_checkpoint_pair(ckpt_dir, fp16_ckpt_dir)

    st = _SafeTensorStore(ckpt_dir)
    fp16_st = _SafeTensorStore(fp16_ckpt_dir) if mixed_quantized else None
    down_w8_st = (
        _SafeTensorStore(w8_down_ckpt_dir)
        if w8_down_ckpt_dir is not None
        else None
    )

    # group-wise (pgrp/AWQ) int4 vs per-channel int4 — group-wise emits a 2D [G, N]
    # scale and packs with the pgrp swizzle for the tiled wint4a16_pgrp kernels.
    w4_group_wise = w4a16 and _w4_is_group_wise(ckpt_dir)

    def gf(key: str) -> torch.Tensor:
        return st.get_tensor(key).float()

    def gr(key: str) -> torch.Tensor:
        return st.get_tensor(key).contiguous()

    q_w_list, k_w_list, v_w_list, o_w_list = [], [], [], []
    q_b_list, k_b_list, v_b_list = [], [], []
    gate_list, up_list, down_list = [], [], []
    q_s_list, k_s_list, v_s_list, o_s_list = [], [], [], []
    gate_s_list, up_s_list, down_s_list = [], [], []
    q_ts_list, k_ts_list, v_ts_list, o_ts_list = [], [], [], []
    gate_ts_list, up_ts_list, down_ts_list = [], [], []
    in_norm_list, post_norm_list = [], []

    for L in layer_indices:
        p = f"model.layers.{L}"
        qkv_w = gr(f"{p}.self_attn.qkv_proj_experts.{expert}.weight")  # [2560, H]
        qkv_b = gf(f"{p}.self_attn.qkv_proj_experts.{expert}.bias")    # [2560]
        qw, kw, vw = _split_qkv(qkv_w, NQ, NKV, HD)
        qb, kb, vb = _split_qkv(qkv_b, NQ, NKV, HD)
        # ATTN_TP8: present REP copies of each real KV head so the col-swizzle sees
        # NKV_EFF==ATTN_TP heads (1 whole head/core). No-op (REP==1) by default.
        kw, vw = _replicate_kv(kw, NKV, HD, REP), _replicate_kv(vw, NKV, HD, REP)
        kb, vb = _replicate_kv(kb, NKV, HD, REP), _replicate_kv(vb, NKV, HD, REP)
        ow = gr(f"{p}.self_attn.o_proj_experts.{expert}.weight")       # [H, nq*hd]
        if w8a16 or w4a16:
            qkv_s = _chan_first_scale(gr(f"{p}.self_attn.qkv_proj_experts.{expert}.weight_scale"))
            qs, ks, vs = _split_qkv(qkv_s, NQ, NKV, HD)
            ks, vs = _replicate_kv(ks, NKV, HD, REP), _replicate_kv(vs, NKV, HD, REP)
            os = _chan_first_scale(gr(f"{p}.self_attn.o_proj_experts.{expert}.weight_scale"))

        # Attention: Q/K/V col-partition + O row-partition, swizzled for attn_tp
        # (matches the C++ num_cores=tp). Bias is NOT swizzled — the col-swizzle
        # keeps core c's output channels in the contiguous slice the C++ bias
        # scatter expects.
        if w8a16:
            q_w_list.append(_to_rpu_int8(tp_col_swizzle_mc_weight(qw, ATTN_TP, dwidth=1)))
            k_w_list.append(_to_rpu_int8(tp_col_swizzle_mc_weight(kw, ATTN_TP, dwidth=1)))
            v_w_list.append(_to_rpu_int8(tp_col_swizzle_mc_weight(vw, ATTN_TP, dwidth=1)))
            o_w_list.append(_to_rpu_int8(tp_row_swizzle_mc_weight(ow, ATTN_TP, dwidth=1)))
            q_s_list.append(_to_rpu_half(qs))
            k_s_list.append(_to_rpu_half(ks))
            v_s_list.append(_to_rpu_half(vs))
            o_s_list.append(_to_rpu_half(os))
        elif w4a16:
            # int4 values (int8 container) from the w4a16 ckpt -> packed uint8 [N,K/2].
            # col(q/k/v)+row(o); qkv carries bias in the int4 path.
            q_w_list.append(_pack_int4_rpu(qw, 1, ATTN_TP, w4_group_wise))
            k_w_list.append(_pack_int4_rpu(kw, 1, ATTN_TP, w4_group_wise))
            v_w_list.append(_pack_int4_rpu(vw, 1, ATTN_TP, w4_group_wise))
            o_w_list.append(_pack_int4_rpu(ow, 0, ATTN_TP, w4_group_wise))
            q_s_list.append(_scale_to_rpu(
                qs, in_features=H, partition=1, num_cores=ATTN_TP))
            k_s_list.append(_scale_to_rpu(
                ks, in_features=H, partition=1, num_cores=ATTN_TP))
            v_s_list.append(_scale_to_rpu(
                vs, in_features=H, partition=1, num_cores=ATTN_TP))
            o_s_list.append(_scale_to_rpu(
                os, in_features=NQ * HD, partition=0, num_cores=ATTN_TP))
        elif nvfp4a16:
            q_w, q_s, q_ts = _pack_nvfp4_rpu(qw, 1, ATTN_TP)
            k_w, k_s, k_ts = _pack_nvfp4_rpu(kw, 1, ATTN_TP)
            v_w, v_s, v_ts = _pack_nvfp4_rpu(vw, 1, ATTN_TP)
            o_w, o_s, o_ts = _pack_nvfp4_rpu(ow, 0, ATTN_TP)
            q_w_list.append(q_w); q_s_list.append(q_s); q_ts_list.append(q_ts)
            k_w_list.append(k_w); k_s_list.append(k_s); k_ts_list.append(k_ts)
            v_w_list.append(v_w); v_s_list.append(v_s); v_ts_list.append(v_ts)
            o_w_list.append(o_w); o_s_list.append(o_s); o_ts_list.append(o_ts)
        else:
            q_w_list.append(_to_rpu_half(tp_col_swizzle_mc_weight(qw.half(), ATTN_TP)))
            k_w_list.append(_to_rpu_half(tp_col_swizzle_mc_weight(kw.half(), ATTN_TP)))
            v_w_list.append(_to_rpu_half(tp_col_swizzle_mc_weight(vw.half(), ATTN_TP)))
            o_w_list.append(_to_rpu_half(tp_row_swizzle_mc_weight(ow.half(), ATTN_TP)))
        q_b_list.append(_to_rpu_half(qb))
        k_b_list.append(_to_rpu_half(kb))
        v_b_list.append(_to_rpu_half(vb))

        gu = gr(f"{p}.moe.experts.{expert}.gate_up_proj.weight")       # [2*INTER, H]
        gate_w = gu[:INTER].contiguous()
        up_w = gu[INTER:2 * INTER].contiguous()
        down_key = f"{p}.moe.experts.{expert}.down_proj.weight"
        down_w = (
            down_w8_st.get_tensor(down_key).contiguous()
            if down_w8_st is not None
            else (
                fp16_st.get_tensor(down_key).contiguous()
                if mixed_quantized and L == 2
                else gr(down_key)
            )
        )                                                              # [H, INTER]
        if down_w8_st is not None:
            down_s = down_w8_st.get_tensor(
                f"{p}.moe.experts.{expert}.down_proj.weight_scale"
            ).contiguous()
        if nvfp4a16 and INTER_PAD != INTER:
            npad = INTER_PAD - INTER
            gate_w = F.pad(gate_w, (0, 0, 0, npad))
            up_w = F.pad(up_w, (0, 0, 0, npad))
            down_w = F.pad(down_w, (0, npad))
        if w8a16 or w4a16:
            gu_s = _chan_first_scale(gr(f"{p}.moe.experts.{expert}.gate_up_proj.weight_scale"))
            gate_s = gu_s[:INTER].contiguous()
            up_s = gu_s[INTER:2 * INTER].contiguous()
            down_s = gr(f"{p}.moe.experts.{expert}.down_proj.weight_scale")
        # MLP: gate/up col-partition + down row-partition, full 8-core.
        if w8a16:
            gate_list.append(_to_rpu_int8(tp_col_swizzle_mc_weight(gate_w, _MLP_CORES, dwidth=1)))
            up_list.append(_to_rpu_int8(tp_col_swizzle_mc_weight(up_w, _MLP_CORES, dwidth=1)))
            down_list.append(
                _to_rpu_half(tp_row_swizzle_mc_weight(down_w.half(), _MLP_CORES))
                if mixed_quantized and L == 2
                else _to_rpu_int8(
                    tp_row_swizzle_mc_weight(down_w, _MLP_CORES, dwidth=1)
                )
            )
            gate_s_list.append(_to_rpu_half(gate_s))
            up_s_list.append(_to_rpu_half(up_s))
            down_s_list.append(_to_rpu_half(down_s))
        elif w4a16:
            # gate/up always int4 (col). down int4 (default, INTER-pad above) OR int8
            # container (RPU_WALL_OSS_EXPERT0_DOWN_INT4=0; full DMA, same int4 numerics).
            # When down_int4: pad gate/up output + down input to INTER_PAD with zero channels
            # (bit-exact: silu(0)*0=0, down pad cols=0; col(gate/up)+row(down) pad both land
            # on core 7's tail -> aligned). gate/up scale per-INTER-channel (padded, 1D);
            # down scale per-H-channel (unaffected by INTER pad).
            npad = INTER_PAD - INTER
            if down_int4 and npad:
                gate_w = F.pad(gate_w, (0, 0, 0, npad))     # [INTER,H] -> [INTER_PAD,H]
                up_w = F.pad(up_w, (0, 0, 0, npad))
                down_w = F.pad(down_w, (0, npad))            # [H,INTER] -> [H,INTER_PAD]
                # per-INTER-channel scale: 1D [INTER] pads the channel dim; group-wise pgrp
                # scale [INTER, G] must pad the OUTPUT-channel dim (0), NOT the group dim.
                _spad = (0, 0, 0, npad) if gate_s.dim() == 2 else (0, npad)
                gate_s = F.pad(gate_s, _spad, value=1.0)
                up_s = F.pad(up_s, _spad, value=1.0)
            gate_list.append(_pack_int4_rpu(gate_w, 1, _MLP_CORES, w4_group_wise))
            up_list.append(_pack_int4_rpu(up_w, 1, _MLP_CORES, w4_group_wise))
            if mixed_quantized and L == 2:
                down_list.append(
                    _to_rpu_half(
                        tp_row_swizzle_mc_weight(down_w.half(), _MLP_CORES)
                    )
                )
            elif down_int4:
                # down is per-channel (1D scale) in BOTH ckpts -> per-channel int4 pack
                # (group_wise=False), NOT the pgrp group swizzle. The pgrp ckpt stores
                # expert-0 down as per-channel INT8 (±127) -> re-quant to int4 on the fly
                # (LOSSY int8->int4) when values exceed the int4 range.
                if down_w.dtype == torch.int8 and int(down_w.abs().max()) > 7:
                    down_w, down_s = _int8_to_int4_perchannel(down_w, down_s)
                down_list.append(_pack_int4_rpu(down_w, 0, _MLP_CORES, group_wise=False))
            else:
                down_list.append(_to_rpu_int8(tp_row_swizzle_mc_weight(down_w, _MLP_CORES, dwidth=1)))
            gate_s_list.append(_scale_to_rpu(
                gate_s, in_features=H, partition=1, num_cores=_MLP_CORES))
            up_s_list.append(_scale_to_rpu(
                up_s, in_features=H, partition=1, num_cores=_MLP_CORES))
            down_s_list.append(_to_rpu_half(down_s))
        elif nvfp4a16:
            gate_wq, gate_s, gate_ts = _pack_nvfp4_rpu(gate_w, 1, _MLP_CORES)
            up_wq, up_s, up_ts = _pack_nvfp4_rpu(up_w, 1, _MLP_CORES)
            gate_list.append(gate_wq); gate_s_list.append(gate_s); gate_ts_list.append(gate_ts)
            up_list.append(up_wq); up_s_list.append(up_s); up_ts_list.append(up_ts)
            if down_w8_st is not None:
                down_list.append(_to_rpu_int8(
                    tp_row_swizzle_mc_weight(
                        down_w, _MLP_CORES, dwidth=1)))
                down_s_list.append(_to_rpu_half(down_s))
                # The shared NVFP4 setter carries one tensor-scale array per
                # projection family. W8 dispatch keys off the int8 weight dtype
                # and ignores this slot, but keep a complete layer array.
                down_ts_list.append(torch.tensor(0.0, dtype=torch.float32))
            else:
                down_wq, down_s, down_ts = _pack_nvfp4_rpu(
                    down_w, 0, _MLP_CORES)
                down_list.append(down_wq)
                down_s_list.append(down_s)
                down_ts_list.append(down_ts)
        else:
            gate_list.append(_to_rpu_half(tp_col_swizzle_mc_weight(gate_w.half(), _MLP_CORES)))
            up_list.append(_to_rpu_half(tp_col_swizzle_mc_weight(up_w.half(), _MLP_CORES)))
            down_list.append(_to_rpu_half(tp_row_swizzle_mc_weight(down_w.half(), _MLP_CORES)))

        in_norm_list.append(_to_rpu_half(gf(f"{p}.input_layernorms.{expert}.weight")))
        post_norm_list.append(_to_rpu_half(gf(f"{p}.post_attention_layernorms.{expert}.weight")))

    final_norm_w = _to_rpu_half(gf(f"model.norms.{expert}.weight"))
    cos, sin = _build_rope_tables(HD, THETA, max_seq_len)
    embed_w = gf("model.embed_tokens.weight")  # [vocab, H] fp32 CPU (kept for lookup)
    partial_mrope = _partial_mrope_enabled()
    cache = RPUCache(
        num_layers=num_layers, batch_size=1, max_seq_len=max_seq_len,
        num_kv_heads=NKV_EFF, head_dim=HD, attn_tp=ATTN_TP,
    )
    graph_cache = rpu_backend.graph.GraphCache()

    set_weights = (
        torch.ops.rpu.causal_decoder_set_weights_nvfp4
        if nvfp4a16 else (
            torch.ops.rpu.causal_decoder_set_weights_w8a16
            if (w8a16 or w4a16) else torch.ops.rpu.causal_decoder_set_weights
        )
    )
    weight_args = [
        q_w_list, k_w_list, v_w_list, o_w_list,
        [], [],                       # q_norm / k_norm — Qwen2.5 has none
        in_norm_list, post_norm_list,
        gate_list, up_list, down_list,
        cos, sin, final_norm_w,
        NQ, NKV_EFF, HD, H, INTER_PAD,   # padded INTER (w4a16 down int4); == INTER otherwise
        EPS, True,                    # use_silu (SwiGLU)
        MROPE, [],                    # mrope_section, deepstack_lang_layers
    ]
    if nvfp4a16:
        weight_args.extend([
            q_s_list, k_s_list, v_s_list, o_s_list,
            gate_s_list, up_s_list, down_s_list,
            _nvfp4_tensor_scale_array(q_ts_list),
            _nvfp4_tensor_scale_array(k_ts_list),
            _nvfp4_tensor_scale_array(v_ts_list),
            _nvfp4_tensor_scale_array(o_ts_list),
            _nvfp4_tensor_scale_array(gate_ts_list),
            _nvfp4_tensor_scale_array(up_ts_list),
            _nvfp4_tensor_scale_array(down_ts_list),
            q_b_list, k_b_list, v_b_list,
        ])
    elif w8a16 or w4a16:
        weight_args.extend([
            q_s_list, k_s_list, v_s_list, o_s_list,
            gate_s_list, up_s_list, down_s_list,
            q_b_list, k_b_list, v_b_list,
        ])
    else:
        weight_args.extend([q_b_list, k_b_list, v_b_list])

    handle = _configure_causal_decoder_handle(
        set_weights=set_weights,
        weight_args=weight_args,
        chunk_size_cap=PREFILL_CHUNK_SIZE_CAP,
        equal_two_prefill=PREFILL_EQUAL_TWO,
        chunk_size_override=CHUNK_SIZE_OVERRIDE,
    )
    transferred = False
    try:
        model = WallOssLLM(
            handle=handle, cache=cache, graph_cache=graph_cache,
            embed_w=embed_w, hidden_size=H, num_layers=num_layers,
            head_dim=HD, rope_theta=THETA, mrope_section=MROPE,
            prefill_chunk_size_cap=PREFILL_CHUNK_SIZE_CAP,
            prefill_pad16=PREFILL_PAD16,
            prefill_equal_two=PREFILL_EQUAL_TWO,
            # NKV_EFF / ATTN_TP, not the checkpoint's NKV: the C++ decoder sees
            # the REPLICATED KV heads, and the chunk predicate keys on what the
            # kernel is launched with.
            attn_geometry=dict(num_q_heads=NQ, num_kv_heads=NKV_EFF,
                               num_cores=ATTN_TP, head_dim=HD,
                               attn_mask_type=1),   # 1 = LTM (causal prefill)
            partial_mrope=partial_mrope,
            execution_config=execution_config,
            prefill_mixed_precision=PREFILL_MIXED_PRECISION,
        )
        transferred = True
        return model
    finally:
        if not transferred:
            _destroy_causal_decoder_handle(handle)
