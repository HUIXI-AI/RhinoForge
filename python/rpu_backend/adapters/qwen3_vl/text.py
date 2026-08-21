"""Qwen3-VL text decoder integration.

Wires `Qwen3VLTextModel` into the existing `causal_decoder_set_weights` /
`causal_decoder_forward` op family with M-RoPE enabled. Reuses the Qwen3
adapter wiring (same QKV / RMSNorm / SwiGLU / qk-norm layout) and
adds three pieces:

1. `mrope_section` is read from `text_config.rope_parameters["mrope_section"]`
   and passed through the new append-only kwarg.
2. cos/sin static tables are built in the kernel format
   `[max_seq, head_dim/2]` (NOT the HF-expanded `[max_seq, head_dim]` form —
   the C++ side rejects the latter on M-RoPE).
3. position_ids is built at forward time by HF's `get_rope_index` and
   normalized to `[seq_len, 3]` int32 RPU by `_run_causal_decoder_forward`.

The same installer serves the public
``RPUModelForConditionalGeneration`` entry point and DeepStack path.
"""

from __future__ import annotations

from typing import Any

import torch

import rpu_backend
from rpu_backend.runtime.decoder import _install_causal_decoder_forward
from rpu_backend.runtime.chunk_envelope import ChunkEnvelope, make_lookup

# Certified chunk envelopes. Qwen3-VL-8B has no row because its exact padded
# attention, KV, MLP, RoPE, and vision-language profile is not supported.
_CHUNK_ENVELOPE = {
    # The envelope bounds executed, 16-aligned length rather than logical token
    # count. The 2B and 4B profiles intentionally retain different certified
    # chunk ceilings; an envelope may reject a request but must not silently
    # change its resolved plan.
    ("qwen3_vl_text", 28, 2048): ChunkEnvelope(336, 320),   # 2b
    ("qwen3_vl_text", 36, 2560): ChunkEnvelope(320, 256),   # 4b
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


def _deepstack_text_layer_indices(vision_config: Any) -> list[int]:
    """Map N ordered vision features to the first N text layers, as HF does."""
    visual_layers = getattr(vision_config, "deepstack_visual_indexes", ()) or ()
    return list(range(len(visual_layers)))


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


def build_mrope_cos_sin_tables(
    text_config: Any,
    *,
    max_seq_len: int | None = None,
    device: str | torch.device = "rpu",
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build kernel-format M-RoPE cos/sin tables.

    Shape: `[max_seq_len, head_dim / 2]` (the rotary half-dim — the kernel
    consumes the standard 1D RoPE table; the per-axis (T/H/W) selection is
    performed at runtime via strobe masks set by the launcher).

    Args:
        text_config: Qwen3VLTextConfig (or anything quack-compatible with
            head_dim / num_attention_heads / hidden_size / rope_parameters).
        max_seq_len: Defaults to `text_config.max_position_embeddings` and
            sizes the table to match the runtime keepalive
            (QWEN3_MROPE_MAX_KEEPALIVE_SEQ = 8192). Larger values are safe
            but waste DDR.
        device: target device (default "rpu").
        dtype: target dtype (default torch.float16 — matches the kernel
            register format).
    """
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None:
        head_dim = text_config.hidden_size // text_config.num_attention_heads
    if head_dim <= 0 or (head_dim % 2) != 0:
        raise ValueError(
            f"build_mrope_cos_sin_tables: head_dim must be positive and even, "
            f"got {head_dim}"
        )

    # rope_parameters (transformers >=5.x) replaces rope_scaling. Try both for
    # forward-compat with mid-flight transformers versions.
    rope_params = getattr(text_config, "rope_parameters", None)
    if rope_params is None:
        rope_params = getattr(text_config, "rope_scaling", None) or {}
    rope_type = rope_params.get("rope_type", "default")
    if rope_type != "default":
        # YaRN / dynamic / linear scaling cases need the corresponding HF
        # rope_init_fn. Only the default branch is wired, so fail loudly
        # rather than silently produce wrong cos/sin.
        raise NotImplementedError(
            f"build_mrope_cos_sin_tables: rope_type={rope_type!r} not yet "
            "supported; only 'default' for Qwen3-VL 2B/4B-Instruct is wired."
        )
    rope_theta = rope_params.get("rope_theta", None)
    if rope_theta is None:
        rope_theta = getattr(text_config, "rope_theta", 10000.0)

    if max_seq_len is None:
        max_seq_len = int(getattr(text_config, "max_position_embeddings", 8192))

    half = head_dim // 2
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    )  # [half]
    positions = torch.arange(max_seq_len, dtype=torch.float64)  # [max_seq]
    freqs = positions[:, None] * inv_freq[None, :]  # [max_seq, half]
    cos = freqs.cos().to(dtype=dtype)
    sin = freqs.sin().to(dtype=dtype)
    cos = cos.to(device=device).contiguous()
    sin = sin.to(device=device).contiguous()
    expected_shape = (max_seq_len, half)
    if tuple(cos.shape) != expected_shape or tuple(sin.shape) != expected_shape:
        raise RuntimeError(
            "build_mrope_cos_sin_tables produced invalid table shapes: "
            f"cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, "
            f"expected={expected_shape}"
        )
    return cos, sin


def install_qwen3_vl_text_for_rpu(
    text_model,
    *,
    text_config: Any | None = None,
    max_seq_len: int | None = None,
    vision_config: Any | None = None,
    deepstack_lang_layers: list[int] | None = None,
    enable_deepstack: bool = True,
    scale_lists=None,
    execution_config=None,
) -> int:
    """Install RPU all-layers-once forward on a `Qwen3VLTextModel` instance.

    Prerequisites (same as the Qwen3 path):
      - `text_model.to('rpu')` has been called.
      - `convert_linear_weights_inplace(text_model)` has been called.

    Text path with M-RoPE and optional DeepStack injection wired through
    `causal_decoder_set_weights`. The supported Qwen3-VL ConditionalGeneration
    adapter composes this with the Vision tower; component/text-only callers can
    opt out of DeepStack via `enable_deepstack=False` or an empty layer list.

    Args:
        text_model: Qwen3VLTextModel instance (already moved to RPU).
        text_config: optional; defaults to `text_model.config`.
        max_seq_len: cos/sin table length; defaults to
            text_config.max_position_embeddings.
        vision_config: optional; used only to auto-detect
            `deepstack_visual_indexes` when `deepstack_lang_layers` is None.
            Pass `model.config.vision_config` for end-to-end Qwen3-VL flows.
        deepstack_lang_layers: explicit list of text-decoder layer indices that
            receive DeepStack injection. Overrides vision_config detection.
        enable_deepstack: false → bypass DeepStack injection regardless of
            config (text-only fallback).
        scale_lists: optional W8A16 per-output-channel fp16 scale tuple
            (q,k,v,o,gate,up,down), one fp16 [out] tensor per layer per
            projection. When provided, the decoder's 7 projection weights must
            already be int8 (swizzled dwidth=1) and the install routes to
            `causal_decoder_set_weights_w8a16`. None → fp16 (default).

    Returns the C++ handle (also stashed at `text_model._rpu_decoder_handle`).
    """
    cfg = text_config if text_config is not None else getattr(text_model, "config", None)
    if cfg is None:
        raise ValueError(
            "install_qwen3_vl_text_for_rpu: text_config not provided and "
            "text_model has no .config attribute."
        )

    rope_params = getattr(cfg, "rope_parameters", None)
    if rope_params is None:
        rope_params = getattr(cfg, "rope_scaling", None) or {}
    mrope_section = rope_params.get("mrope_section", None)
    if mrope_section is None:
        raise ValueError(
            "install_qwen3_vl_text_for_rpu: text_config.rope_parameters['mrope_section'] "
            "is required (expected [T_dim_half, H_dim_half, W_dim_half])."
        )
    mrope_section = [int(x) for x in mrope_section]

    cos, sin = build_mrope_cos_sin_tables(
        cfg, max_seq_len=max_seq_len, device="rpu", dtype=torch.float16,
    )

    # Resolve deepstack layer list. Caller > vision_config > [].
    if not enable_deepstack:
        ds_layers: list[int] = []
    elif deepstack_lang_layers is not None:
        ds_layers = [int(x) for x in deepstack_lang_layers]
    elif vision_config is not None:
        ds_layers = _deepstack_text_layer_indices(vision_config)
    else:
        ds_layers = []

    # Eager-initialize the text-decoder GraphCache so the
    # Qwen3-VL adapter can batch all kernel dispatches inside one
    # `with graph_cache.capture(sig):` block per forward. Without the
    # wrap, the graph runtime dispatches each kernel in immediate mode.
    text_graph_cache = rpu_backend.graph.GraphCache()
    text_num_layers = int(cfg.num_hidden_layers)
    text_hidden_size = int(cfg.hidden_size)
    # FNV1a-style hash of the deepstack lang-layer list — used as a
    # tiebreaker in the GraphSignature so two models with different
    # deepstack layouts but the same sequence shape don't alias.
    _ds_hash = 0
    for idx in ds_layers:
        _ds_hash = (_ds_hash * 1099511628211) ^ int(idx)
        _ds_hash &= (1 << 63) - 1
    runtime_attrs = {
        "_rpu_text_graph_cache": text_graph_cache,
        "_rpu_text_num_layers": text_num_layers,
        "_rpu_text_hidden_size": text_hidden_size,
        "_rpu_text_deepstack_hash": _ds_hash,
    }
    model_vars = vars(text_model)
    snapshot = {
        name: (name in model_vars, model_vars.get(name))
        for name in runtime_attrs
    }

    # Publish only pre-built, non-native wrapper state before entering the
    # common decoder transaction. If that transaction fails, its old handle
    # remains published and this wrapper state is restored exactly.
    try:
        for name, value in runtime_attrs.items():
            setattr(text_model, name, value)
        handle = _install_causal_decoder_forward(
            text_model,
            arch=QWEN3_VL_TEXT_ARCH,
            mrope_section=mrope_section,
            cos_sin=(cos, sin),
            deepstack_lang_layers=ds_layers,
            scale_lists=scale_lists,
            chunk_envelope_for=lambda arch, num_layers, hidden_size: (
                _chunk_envelope_for_execution(
                    arch,
                    num_layers,
                    hidden_size,
                    execution_config,
                )
            ),
            execution_config=execution_config,
        )
    except BaseException:
        try:
            text_graph_cache.clear()
        except Exception:
            pass
        # Rollback must not re-enter model-defined attribute hooks: the
        # publication failure may have come from those hooks in the first
        # place, and they may keep rejecting subsequent writes/deletes.
        model_vars = vars(text_model)
        for name in runtime_attrs:
            model_vars.pop(name, None)
        for name, (had_attr, value) in snapshot.items():
            if had_attr:
                model_vars[name] = value
        raise

    old_graph_cache = snapshot["_rpu_text_graph_cache"][1]
    if old_graph_cache is not None and old_graph_cache is not text_graph_cache:
        try:
            old_graph_cache.clear()
        except Exception:
            pass

    return handle


# ─────────────────────────────────────────────────────────────────────────
# DeepStack helpers
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

    Sized to MAX_KEEPALIVE_SEQ rather than max_chunk because the
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
    view = z.narrow(0, 0, seq_len)
    return [view for _ in range(n_mergers)]


def scatter_visual_embeds_to_dense(
    visual_embeds: list[torch.Tensor],
    visual_pos_mask: torch.Tensor,
    seq_len: int,
    hidden_size: int,
    *,
    device: str = "rpu",
) -> list[torch.Tensor]:
    """Build dense `[seq_len, hidden]` fp16 tensors from sparse visual_embeds.

    HF Qwen3-VL `_deepstack_process` does `hidden[mask] += visual_embeds[i]`.
    Equivalent: dense = zeros[seq_len, hidden]; dense[mask] = visual_embeds[i];
    then `hidden += dense`. The fused-graph injection consumes the dense form.

    Args:
        visual_embeds: list of N `[num_visual_tokens, hidden]` tensors (one per
            DeepStack merger). All on the same device, fp16.
        visual_pos_mask: 1D bool/long tensor of shape `[seq_len]` marking
            which positions hold visual tokens.
        seq_len: full prompt length (text + visual tokens).
        hidden_size: text-decoder hidden size.
        device: target device (default 'rpu').

    Returns N dense fp16 tensors of shape `[seq_len, hidden_size]`.
    """
    mask = visual_pos_mask.to(device=device)
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    if mask.numel() != seq_len:
        raise ValueError(
            f"scatter_visual_embeds_to_dense: visual_pos_mask length "
            f"{mask.numel()} != seq_len {seq_len}"
        )

    dense_list: list[torch.Tensor] = []
    for i, embed in enumerate(visual_embeds):
        if embed.shape[1] != hidden_size:
            raise ValueError(
                f"scatter_visual_embeds_to_dense[{i}]: hidden {embed.shape[1]} "
                f"!= text hidden_size {hidden_size}"
            )
        dense = torch.zeros((seq_len, hidden_size), dtype=torch.float16, device=device)
        dense[mask] = embed.to(dtype=torch.float16, device=device)
        dense_list.append(dense.contiguous())
    return dense_list
