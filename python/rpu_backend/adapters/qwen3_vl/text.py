"""Qwen3-VL text decoder → rpu_backend (R-Phase 1).

Wires `Qwen3VLTextModel` into the existing `causal_decoder_set_weights` /
`causal_decoder_forward` op family with M-RoPE enabled. Reuses 90 % of the
Qwen3 adapter wiring (same QKV / RMSNorm / SwiGLU / qk-norm layout) and
adds three pieces:

1. `mrope_section` is read from `text_config.rope_parameters["mrope_section"]`
   and passed through the new append-only kwarg.
2. cos/sin static tables are built in the kernel format
   `[max_seq, head_dim/2]` (NOT the HF-expanded `[max_seq, head_dim]` form —
   the C++ side rejects the latter on M-RoPE).
3. position_ids is built at forward time by HF's `get_rope_index` and
   normalized to `[seq_len, 3]` int32 RPU by `_run_causal_decoder_forward`.

R-Phase 1 verification uses `install_qwen3_vl_text_for_rpu` directly from
the verify_mrope_text_only.py golden test; the user-facing entry
(`RPUModelForConditionalGeneration`) lands in R-Phase 4.

DeepStack helpers + injection wiring arrive in R-Phase 2.
"""

from __future__ import annotations

from typing import Any

import torch

from rpu_backend.runtime.decoder import (
    build_mrope_cos_sin_tables, install_mrope_text_decoder_for_rpu,
)
from rpu_backend.runtime.chunk_envelope import ChunkEnvelope, make_lookup
from rpu_backend.runtime.topology import execution_core_count

# Declared execution-length and chunk ceilings for each text geometry.
_CHUNK_ENVELOPE = {
    # Bound the padded execution length, not the logical token count.
    # P4097 requires E4112; optional padding can extend the search to E4176.
    # Per-chunk ceilings do not bypass native SPM/SDPA feasibility checks.
    ("qwen3_vl_text", 28, 2048): ChunkEnvelope(4176, 320),  # 2b
    # 4B long-context admission retains C256 and every native SPM/SDPA check.
    # 4176 bounds padded prefill execution; P4096D128 separately needs KV4224.
    ("qwen3_vl_text", 36, 2560): ChunkEnvelope(4176, 256),  # 4b
    # Exact padded 8B retains C128; the extended length window does not
    # enlarge native SPM/SDPA candidates or the separately allocated KV cache.
    ("qwen3_vl_text", 36, 4096): ChunkEnvelope(4176, 128),  # padded 8b
    # GR00T-N1.7-3B backbone — a TRUNCATED 16-layer Qwen3-VL-2B (select_layer=16)
    # installed through install_qwen3_vl_text_for_rpu, so it lands on this arch
    # key with its own geometry. Auto is independently certified to length 512;
    # no exact chunk is certified for this truncated geometry.
    ("qwen3_vl_text", 16, 2048): ChunkEnvelope(512),
}
lookup_causal_decoder = make_lookup(
    _CHUNK_ENVELOPE, "rpu_backend/adapters/qwen3_vl/text.py::_CHUNK_ENVELOPE")

QWEN3_VL_TEXT_ARCH = "qwen3_vl_text"
_GR00T_TEXT_GEOMETRY = (QWEN3_VL_TEXT_ARCH, 16, 2048)


def _chunk_envelope_for_execution(
    arch: str,
    num_layers: int,
    hidden_size: int,
    execution_config=None,
) -> ChunkEnvelope:
    """Return the certified envelope for this handle's requested policy.

    GR00T's truncated decoder is certified only with the native AUTO policy.
    Keep this second guard at the installer boundary so a caller that bypasses
    ``build_gr00t_vla`` still cannot turn an auto-only envelope into an exact
    claim.
    """
    envelope = lookup_causal_decoder(arch, num_layers, hidden_size)
    key = (arch, int(num_layers), int(hidden_size))
    if key != _GR00T_TEXT_GEOMETRY:
        return envelope

    requested = (execution_config or {}).get("prefill", {}).get(
        "chunk_size", "auto"
    )
    if requested == "auto":
        return envelope
    raise ValueError(
        "GR00T truncated text prefill supports chunk_size='auto' only; "
        f"got {requested!r}"
    )


_RUNTIME_QUANTIZED_32B_ENVELOPE = ChunkEnvelope(4176, 64)


def _validate_runtime_quantized_32b_text(text_model, cfg, scale_lists):
    """Bind the exact 32B envelope to the live quantized projection owners."""
    fields = ("num_hidden_layers", "hidden_size", "intermediate_size",
              "num_attention_heads", "num_key_value_heads", "head_dim")
    for config in (cfg, getattr(text_model, "config", None)):
        values = tuple(getattr(config, field, None) for field in fields)
        if (getattr(config, "model_type", None) != QWEN3_VL_TEXT_ARCH
                or any(type(value) is not int for value in values)
                or values != (64, 5120, 25600, 64, 8, 128)):
            raise ValueError("runtime quantized 32B text requires exact live geometry")
    layers = getattr(text_model, "layers", ())
    if (len(layers) != 64 or not isinstance(scale_lists, (tuple, list))
            or len(scale_lists) != 7
            or any(not isinstance(scales, (tuple, list)) or len(scales) != 64
                   for scales in scale_lists)):
        raise ValueError("runtime quantized 32B text requires seven complete scale lists")
    shapes = (("self_attn.q_proj", 8192, 5120),
              ("self_attn.k_proj", 1024, 5120),
              ("self_attn.v_proj", 1024, 5120),
              ("self_attn.o_proj", 5120, 8192),
              ("mlp.gate_proj", 25600, 5120),
              ("mlp.up_proj", 25600, 5120),
              ("mlp.down_proj", 5120, 25600))
    dtype = None
    for (role, n, k), scales in zip(shapes, scale_lists, strict=True):
        for layer, scale in zip(layers, scales, strict=True):
            try:
                projection = layer.get_submodule(role)
            except (AttributeError, KeyError) as exc:
                raise ValueError("runtime quantized 32B text requires all seven projections") from exc
            weight = getattr(projection, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.dtype not in (torch.int8, torch.uint8):
                raise ValueError("runtime quantized 32B text requires uniform INT8 or packed W4 weights")
            if dtype is None:
                dtype = weight.dtype
            packed = dtype == torch.uint8
            if (weight.dtype != dtype or not weight.is_contiguous()
                    or (projection.out_features, projection.in_features) != (n, k)
                    or tuple(weight.shape) != (n, k // 2 if packed else k)
                    or getattr(projection, "bias", None) is not None
                    or not isinstance(scale, torch.Tensor)
                    or scale is not projection._buffers.get("weight_scale")
                    or scale.dtype != torch.float16 or not scale.is_contiguous()
                    or scale.device != weight.device
                    or tuple(scale.shape) != ((32, n * k // 1024) if packed else (n,))):
                raise ValueError("runtime quantized 32B text requires matching projection weights and actual scale buffers")


def install_qwen3_vl_text_for_rpu(
    text_model, *, text_config=None, max_seq_len=None, vision_config=None,
    deepstack_lang_layers=None, enable_deepstack=True, scale_lists=None,
    execution_config=None, topology=None, runtime_align_w8a16=False,
    _runtime_quantized_32b=False,
    _graph_cache_max_entries=None,
) -> int:
    """Bind this adapter's certified geometry to the shared M-RoPE installer."""
    if type(_runtime_quantized_32b) is not bool:
        raise ValueError("_runtime_quantized_32b must be a bool")
    if _runtime_quantized_32b:
        if (runtime_align_w8a16 or topology is not None
                or execution_core_count(execution_config) != 8):
            raise ValueError("runtime quantized 32B text requires TP8 without the legacy W8 flag")
        cfg = text_config if text_config is not None else getattr(text_model, "config", None)
        _validate_runtime_quantized_32b_text(text_model, cfg, scale_lists)
    if runtime_align_w8a16:
        cfg = text_config if text_config is not None else text_model.config
        if (topology is not None or scale_lists is None or
                (cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size,
                 cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim)
                != (36, 2560, 9728, 32, 8, 128)):
            raise ValueError("runtime-alignment W8 text requires exact 4B TP8 geometry and scales")
    return install_mrope_text_decoder_for_rpu(
        text_model, arch=QWEN3_VL_TEXT_ARCH,
        chunk_envelope_for=lambda arch, num_layers, hidden_size: (
            # Explicit candidate only; native planning/SPM checks still apply.
            # This is not an ordinary FP16 envelope or hardware certification.
            _RUNTIME_QUANTIZED_32B_ENVELOPE if _runtime_quantized_32b
            and (arch, num_layers, hidden_size) == (QWEN3_VL_TEXT_ARCH, 64, 5120) else
            ChunkEnvelope(4176, 256) if runtime_align_w8a16 else
            _chunk_envelope_for_execution(arch, num_layers, hidden_size, execution_config)
        ),
        text_config=text_config, max_seq_len=max_seq_len,
        vision_config=vision_config, deepstack_lang_layers=deepstack_lang_layers,
        enable_deepstack=enable_deepstack, scale_lists=scale_lists,
        execution_config=execution_config, topology=topology,
        _kv_cache_layer_bank_size=8 if _runtime_quantized_32b else 1,
        **({"_graph_cache_max_entries": _graph_cache_max_entries}
           if _graph_cache_max_entries is not None else {}),
    )


# ─────────────────────────────────────────────────────────────────────────
# DeepStack helpers (R-Phase 2)
# ─────────────────────────────────────────────────────────────────────────


def get_or_create_zero_keepalive(
    text_model,
    *,
    text_config: Any | None = None,
    max_seq_len: int = 8192,
) -> torch.Tensor:
    """Return a per-model `[max_seq_len, hidden_size]` fp16 RPU zero buffer.

    The buffer is created on first call and cached on the text_model. Its
    purpose is to serve as the "zero" dense visual_embeds tensor that the
    DeepStack fused-graph injection reads when there are no actual visual
    tokens to add (text-only prefill, decode, multi-chunk text prefill).

    Sized to MAX_KEEPALIVE_SEQ rather than max_chunk per findings F27 — the
    mutable DMA's `src_offset_bytes = chunk.offset × hidden × 2` is baked at
    BUILD time, so chunk N > 0 reads beyond the per-forward seq_len region.
    """
    existing = getattr(text_model, "_rpu_deepstack_zero_keepalive", None)
    if existing is not None and existing.shape[0] >= max_seq_len:
        return existing

    cfg = text_config if text_config is not None else text_model.config
    hidden_size = cfg.hidden_size
    z = torch.zeros(
        (max_seq_len, hidden_size),
        dtype=torch.float16,
        device="rpu",
    ).contiguous()
    text_model._rpu_deepstack_zero_keepalive = z
    text_model._rpu_deepstack_zero_views = {}
    return z


def make_zero_visual_embeds(
    text_model,
    seq_len: int,
    *,
    n_mergers: int = 3,
) -> list[torch.Tensor]:
    """Return `n_mergers` views into the per-model zero keepalive, each
    shaped `[seq_len, hidden]`. Use for text-only / decode forwards.

    All views share the same underlying storage — the C++ side flushes them
    once and reads the (all-zero) memory in the mutable DMA. Allocating once
    per model (not per forward) keeps the RPU caching allocator from churning.
    """
    z = get_or_create_zero_keepalive(text_model)
    if seq_len > z.shape[0]:
        raise ValueError(
            f"make_zero_visual_embeds: seq_len={seq_len} exceeds "
            f"keepalive capacity={z.shape[0]}; bump max_seq_len or split chunks."
        )
    zero_views = getattr(text_model, "_rpu_deepstack_zero_views", None)
    if zero_views is None:
        zero_views = {}
        text_model._rpu_deepstack_zero_views = zero_views
    view = zero_views.get(seq_len)
    if view is None:
        view = z.narrow(0, 0, seq_len)
        zero_views[seq_len] = view
    return [view for _ in range(n_mergers)]


def scatter_visual_embeds_to_dense(
    visual_embeds: list[torch.Tensor | list[torch.Tensor]],
    visual_pos_mask: torch.Tensor,
    seq_len: int,
    hidden_size: int,
    *,
    device: str = "rpu",
    cache_owner: Any | None = None,
    row_indices_cpu: torch.Tensor | None = None,
    cached_prefix_parts: int = 0,
    execution_len: int | None = None,
) -> list[torch.Tensor]:
    """Build dense FP16 DeepStack inputs with zero execution-padding rows.

    A cache owner enables stable internal DDR buffers. Per-image source parts
    may mix CPU and RPU tensors; unchanged leading cached parts are retained
    while only fresh suffix rows are overwritten. Scatter indices remain in
    the logical `seq_len` even when the buffer has `execution_len` rows.
    """
    if execution_len is None:
        execution_len = seq_len
    if execution_len < seq_len:
        raise ValueError(
            f"execution_len={execution_len} is smaller than logical seq_len={seq_len}"
        )
    if visual_pos_mask.numel() != seq_len:
        raise ValueError(
            f"scatter_visual_embeds_to_dense: visual_pos_mask length "
            f"{visual_pos_mask.numel()} != seq_len {seq_len}"
        )

    mask_unchanged = False
    if cache_owner is None:
        mask = visual_pos_mask.to(device=device, dtype=torch.bool)
        dense_list = [
            torch.zeros(
                (execution_len, hidden_size), dtype=torch.float16, device=device
            )
            for _ in visual_embeds
        ]
    else:
        signature = (seq_len, execution_len, hidden_size, len(visual_embeds), str(device))
        previous_signature = getattr(
            cache_owner, "_rpu_deepstack_dense_signature", None
        )
        if previous_signature != signature:
            dense_list = [
                torch.zeros(
                    (execution_len, hidden_size), dtype=torch.float16, device=device
                ).contiguous()
                for _ in visual_embeds
            ]
            cache_owner._rpu_deepstack_dense_buffers = dense_list
            cache_owner._rpu_deepstack_dense_signature = signature
        else:
            dense_list = cache_owner._rpu_deepstack_dense_buffers

        mask_cpu = visual_pos_mask.detach().to(
            device="cpu", dtype=torch.bool
        ).contiguous()
        if row_indices_cpu is None:
            row_indices_cpu = (
                mask_cpu.nonzero().flatten().to(torch.int64).contiguous()
            )
        else:
            row_indices_cpu = row_indices_cpu.detach().to(
                device="cpu", dtype=torch.int64
            ).contiguous()
        if execution_len > seq_len and row_indices_cpu.numel() \
                and int(row_indices_cpu.max()) >= seq_len:
            raise ValueError("DeepStack scatter indices must stay within logical seq_len")
        previous_mask = getattr(
            cache_owner, "_rpu_deepstack_dense_mask_cpu", None
        )
        mask_unchanged = (
            previous_signature == signature
            and previous_mask is not None
            and torch.equal(previous_mask, mask_cpu)
        )
        if previous_signature == signature and previous_mask is not None \
                and not mask_unchanged:
            for dense in dense_list:
                dense.zero_()
        if not mask_unchanged:
            # A changed mask clears every destination. Invalidate all merger
            # prefix stamps before any one native scatter can fail.
            cache_owner._rpu_deepstack_cached_prefix_signatures = {}
        cache_owner._rpu_deepstack_dense_mask_cpu = mask_cpu.clone()

    expected_visual_rows = (
        row_indices_cpu.numel()
        if row_indices_cpu is not None else int(mask.sum().item())
    )
    for i, embed in enumerate(visual_embeds):
        embed_parts = list(embed) if isinstance(embed, (list, tuple)) else None
        tensors = embed_parts if embed_parts is not None else [embed]
        for part in tensors:
            if part.dim() != 2 or part.shape[1] != hidden_size:
                raise ValueError(
                    f"scatter_visual_embeds_to_dense[{i}]: shape "
                    f"{tuple(part.shape)} must be "
                    f"[num_visual_tokens, {hidden_size}]"
                )
        source_rows = sum(int(part.shape[0]) for part in tensors)
        if source_rows != expected_visual_rows:
            raise ValueError(
                f"scatter_visual_embeds_to_dense[{i}]: source rows "
                f"{source_rows} != visual rows {expected_visual_rows}"
            )

        dense = dense_list[i]
        if cache_owner is None:
            merged = torch.cat(tensors, dim=0) if embed_parts is not None else embed
            dense[:seq_len][mask] = merged.to(dtype=torch.float16, device=device)
            continue

        prefix_signatures = getattr(
            cache_owner, "_rpu_deepstack_cached_prefix_signatures", {}
        )
        previous_prefix = prefix_signatures.pop(i, None)
        if embed_parts is None:
            embed_cpu = embed.detach().to(
                device="cpu", dtype=torch.float16
            ).contiguous()
            torch.ops.rpu.qwen3vl_scatter_row_parts_unflushed_(
                dense, row_indices_cpu, [embed_cpu]
            )
            continue

        normalized_parts = [
            part.detach().to(dtype=torch.float16).contiguous()
            for part in embed_parts
        ]
        scatter_parts = normalized_parts
        scatter_indices = row_indices_cpu
        pending_prefix = None
        if cached_prefix_parts:
            if not 0 <= cached_prefix_parts < len(normalized_parts):
                raise ValueError(
                    "cached_prefix_parts must be in "
                    "[0, len(deepstack parts))"
                )
            prefix_rows = sum(
                int(part.shape[0])
                for part in normalized_parts[:cached_prefix_parts]
            )
            prefix_sources = normalized_parts[:cached_prefix_parts]
            cacheable_prefix = all(not torch.is_inference(part) for part in prefix_sources)
            prefix_signature = (
                int(dense.data_ptr()),
                tuple(
                    (int(part.data_ptr()), tuple(part.shape), str(part.dtype),
                     int(part._version) if cacheable_prefix else None)
                    for part in prefix_sources
                ),
            )
            if (mask_unchanged and cacheable_prefix and previous_prefix is not None
                    and previous_prefix[0] == prefix_signature):
                scatter_parts = normalized_parts[cached_prefix_parts:]
                scatter_indices = row_indices_cpu[prefix_rows:].contiguous()
            if cacheable_prefix:
                pending_prefix = (prefix_signature, prefix_sources)
        torch.ops.rpu.qwen3vl_scatter_row_parts_unflushed_(
            dense, scatter_indices, scatter_parts
        )
        if pending_prefix is not None:
            prefix_signatures[i] = pending_prefix
            cache_owner._rpu_deepstack_cached_prefix_signatures = prefix_signatures
    return dense_list
