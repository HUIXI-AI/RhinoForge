"""DINOv3 ViT → rpu_backend adapter.

Public surface:
  - `DINOv3Adapter` — direct per-instance adapter. Run config preflight before
    loading weights, then call `DINOv3Adapter(model).to_rpu()`.
  - `build_dinov3_rope_tables` / `build_position_idx` — RoPE-2D helpers.

Install and forward path:
  - Install path: config preflight → patch_embed Conv2d→Linear CPU fp16
    fold + ViT-B 12→16 head zero-pad + linear swizzle → move to RPU →
    install RPU keepalives (CLS+reg / γ_attn / γ_mlp stacks) → install
    fused C++ vision handle (set_weights / set_rope / KV cache) → set
    hardware attrs → replace forward.
  - Forward dispatch: `_rpu_dinov3_forward`:
      1. patch_embed via CPU fp16 folded Linear (Conv2d k=16 s=16 → Linear).
      2. CLS + 4 register tokens prepend from RPU keepalive.
      3. Per-forward `build_position_idx(num_h, num_w) → [num_patches, 2]
         int16` copied into the C++ position-idx keepalive's first rows.
      4. `torch.ops.rpu.dinov3_vision_forward` runs the all-layers-once
         fused graph: per layer LN1 → Q/K/V Linear (col-partition) → 2D
         RoPE on patch tokens only (CLS+4 reg unrotated via SPM byte
         offset) → bidirectional SDPA → O Linear (row-partition) +
         γ_attn elementwise mul on per-core partial + AllReduce-sum-residual
         → LN2 → up_proj + GELU → down_proj + γ_mlp mul + AllReduce-residual.
      5. Final LayerNorm (`model.norm`, aten op on RPU) + CLS slice as
         `pooler_output`.

"""
from __future__ import annotations

import logging
import math
import threading
import types
import weakref
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import rpu_backend
from rpu_backend.api.cache import RPUCache
from rpu_backend.api.causal_lm import _claim_live_instance
from rpu_backend.runtime.device import extract_to_device_target, is_rpu_device_target
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError
from rpu_backend.runtime.registry import register_adapter
from rpu_backend.runtime.weights import convert_linear_weights_inplace


__all__ = [
    "DINOv3Adapter",
    "DINOV3_VIT_ARCH",
    "build_dinov3_rope_tables",
    "build_position_idx",
    "install_dinov3_vision_for_rpu",
]


DINOV3_VIT_ARCH = "DINOv3ViTModel"

# Mirrors C++ NUM_CORES. Attention head count must be a multiple of this; the
# Python adapter pads ViT-B (12 heads) → 16 heads with zero rows/cols before
# the linear swizzle (see `_pad_attn_heads_to_multiple_of_cores`).
_NUM_CORES = 8
_LOG = logging.getLogger(__name__)


# =============================================================================
# Section 1 — Loader: HF config validation
# =============================================================================
#
# Supported variants. Profile tuple is:
#   (hidden_size, num_attention_heads, num_hidden_layers, patch_size,
#    image_size, num_register_tokens)
# ViT-7B/16 (4096/32/40 + head_dim 128) is outside the supported envelope.
_SUPPORTED_PROFILES: frozenset[tuple[int, int, int, int, int, int]] = frozenset({
    # ViT-B/16 (86 M) — head_dim 64
    ( 768, 12, 12, 16, 224, 4),
})

_NUMERIC_BLOCKED_PROFILES: dict[
    tuple[int, int, int, int, int, int], str
] = {
    # ViT-S/16: q·k^T reaches ~1.8e7, far beyond the fp16 finite range.
    (384, 6, 12, 16, 224, 4): "attention q·k^T overflows FP16",
    # ViT-L/16: LayerScale × block output exceeds the fp16 finite range.
    (1024, 16, 24, 16, 224, 4): "LayerScale output overflows FP16",
}


def _profile(config: Any) -> tuple[int, int, int, int, int, int]:
    return (
        int(config.hidden_size),
        int(config.num_attention_heads),
        int(config.num_hidden_layers),
        int(config.patch_size),
        int(config.image_size),
        int(getattr(config, "num_register_tokens", 0)),
    )


def _check_profile(config: Any) -> None:
    """Raise UnsupportedModelError if the HF config isn't in
    `_SUPPORTED_PROFILES`. Reads only config — never touches weights.
    """
    if getattr(config, "model_type", None) != "dinov3_vit":
        raise UnsupportedModelError(
            f"DINOv3Adapter: expected config.model_type='dinov3_vit', got "
            f"{getattr(config, 'model_type', None)!r}."
        )
    if config.use_gated_mlp:
        raise UnsupportedModelError(
            "DINOv3Adapter: SwiGLU variant (use_gated_mlp=True, i.e. ViT-S+/"
            "H+/7B) is not supported. The supported profile uses a GELU MLP."
        )
    if config.num_register_tokens != 4:
        raise UnsupportedModelError(
            f"DINOv3Adapter: expected num_register_tokens=4, got "
            f"{config.num_register_tokens}. Other counts are not supported."
        )
    profile = _profile(config)
    blocked_reason = _NUMERIC_BLOCKED_PROFILES.get(profile)
    if blocked_reason is not None:
        raise UnsupportedModelError(
            f"DINOv3Adapter: profile {profile} is unsupported/numerical-blocked: "
            f"{blocked_reason}. Only DINOv3 ViT-B is currently supported; "
            "a model-registry path does not imply runtime support."
        )
    if profile not in _SUPPORTED_PROFILES:
        raise UnsupportedModelError(
            f"DINOv3Adapter: profile {profile} not supported. "
            f"Supported: {sorted(_SUPPORTED_PROFILES)} "
            "(ViT-B at 224×224 with 4 register tokens; ViT-S/L are "
            "numerical-blocked and ViT-7B is outside the supported envelope)."
        )


# =============================================================================
# Section 2 — Weights: pre-move CPU conversion (linear swizzle + patch_embed
#                       Conv2d→Linear fold). LayerScale γ is captured by the
#                       post-move keepalive installer (Section 2b).
# =============================================================================
#
# Double-swizzle guard: convert_linear_weights_inplace must
# run EXACTLY once per model. The adapter `_rpu_swizzled` marker enforces
# this; the global swizzle lock prevents concurrent calls.
#
# DINOv3 attention has bias=None on k_proj (HF config.key_bias=False).
# `convert_linear_weights_inplace` only operates on `child.weight`, never
# touches bias.
#
# Patch embed (Conv2d k=16 s=16) is folded to a row-major CPU-fp16 Linear.
# RPU `aten::linear` requires a pre-swizzled weight.

SKIP_LINEAR_NAMES: set[str] = set()


def _padded_num_heads(num_heads: int) -> int:
    """Round `num_heads` up to the next multiple of NUM_CORES.

    DINOv3 ViT-B (12 heads) → 16; ViT-L (16 heads) → 16; ViT-S (6 heads) → 8.
    The framework hard-assumes `nq % NUM_CORES == 0` (it computes per-core
    head slices via `nq // NUM_CORES`; non-divisible counts silently drop the
    remainder). Padding with zero attention weights yields zero contribution
    from the phantom heads while keeping the kernel sequence unchanged.
    """
    if num_heads <= 0:
        raise ValueError(f"_padded_num_heads: num_heads must be positive, got {num_heads}")
    return ((num_heads + _NUM_CORES - 1) // _NUM_CORES) * _NUM_CORES


def _pad_attn_heads_to_multiple_of_cores(model: nn.Module) -> None:
    """Zero-pad q/k/v/o weights (and q/v biases) so `num_heads` becomes a
    multiple of NUM_CORES.

    HF DINOv3 attention layout:
      - q_proj / k_proj / v_proj: `nn.Linear(hidden, nq * head_dim)` with
        bias=True for q/v and bias=None for k.
      - o_proj: `nn.Linear(nq * head_dim, hidden)` with bias=True.

    For ViT-B (nq=12, head_dim=64) we pad to nq=16:
      q/k/v_proj.weight: `[12·64=768, hidden] → [16·64=1024, hidden]`
        with the last 4·64=256 output rows zero (silent zero contribution
        from those heads in SDPA).
      q/v_proj.bias: `[768] → [1024]` with the last 256 entries zero.
      o_proj.weight: `[hidden, 768] → [hidden, 1024]` with the last 256
        input columns zero (silent zero contribution to o_proj sum).
      o_proj.bias and k_proj.bias are unchanged.

    No-op if `nq` is already divisible by NUM_CORES (e.g., ViT-L).

    Must run BEFORE `convert_linear_weights_inplace` — the swizzle interleaves
    bytes assuming the canonical `[out, in]` shape, so the pad-then-swizzle
    order matters. After padding, the Linear's `.in_features` / `.out_features`
    metadata is stale; the converter only reads `.weight.shape` and
    `.bias.shape`, so this is safe.
    """
    cfg = model.config
    num_heads = int(cfg.num_attention_heads)
    target = _padded_num_heads(num_heads)
    if target == num_heads:
        return
    head_dim = int(cfg.hidden_size) // num_heads
    pad_rows = (target - num_heads) * head_dim

    with torch.no_grad():
        for layer in model.layer:
            attn = layer.attention
            for name in ("q_proj", "k_proj", "v_proj"):
                lin = getattr(attn, name)
                w = lin.weight.data
                pad = torch.zeros(
                    pad_rows, w.shape[1], dtype=w.dtype, device=w.device,
                )
                lin.weight = nn.Parameter(
                    torch.cat([w, pad], dim=0).contiguous(), requires_grad=False,
                )
            for name in ("q_proj", "v_proj"):
                lin = getattr(attn, name)
                b = lin.bias.data
                pad = torch.zeros(pad_rows, dtype=b.dtype, device=b.device)
                lin.bias = nn.Parameter(
                    torch.cat([b, pad], dim=0).contiguous(), requires_grad=False,
                )
            o = attn.o_proj
            w = o.weight.data
            pad = torch.zeros(
                w.shape[0], pad_rows, dtype=w.dtype, device=w.device,
            )
            o.weight = nn.Parameter(
                torch.cat([w, pad], dim=1).contiguous(), requires_grad=False,
            )


def _convert_dinov3_weights_for_rpu(model: nn.Module) -> None:
    """Pre-move CPU weight transforms.

    1. Capture HF `embeddings.patch_embeddings` Conv2d weight `[H, 3, k, k]`
       folded to `[H, 3·k·k]` plus its bias, both CPU fp16 contiguous,
       stashed on `model._rpu_dinov3_patch_embed_w` / `_b`.
    2. Pad attention heads up to a multiple of NUM_CORES (no-op for ViT-L).
    3. Swizzle every leaf `nn.Linear` weight in-place (attention q/k/v/o,
       MLP up_proj/down_proj). k_proj has bias=None — preserved because
       `convert_linear_weights_inplace` only touches `.weight`.
    """
    with torch.no_grad():
        pe = model.embeddings.patch_embeddings
        pe_w = pe.weight.data  # [hidden, in_channels, k, k]
        if pe_w.dim() != 4:
            raise RPUBackendError(
                f"DINOv3 patch_embeddings.weight: expected 4D Conv2d, got "
                f"{pe_w.dim()}D shape={tuple(pe_w.shape)}"
            )
        hidden, c_in, k_h, k_w = pe_w.shape
        model._rpu_dinov3_patch_embed_w = (
            pe_w.reshape(hidden, c_in * k_h * k_w)
            .to(dtype=torch.float16, device="cpu")
            .contiguous()
        )
        model._rpu_dinov3_patch_embed_b = (
            pe.bias.data.to(dtype=torch.float16, device="cpu").contiguous()
        )

    _pad_attn_heads_to_multiple_of_cores(model)
    convert_linear_weights_inplace(model, skip_names=SKIP_LINEAR_NAMES)


# =============================================================================
# Section 2b — Post-move RPU keepalive buffers: CLS+reg tokens + LayerScale γ
# =============================================================================
#
# Called AFTER `model.to('rpu')` so the source tensors (cls_token,
# register_tokens, layer_scale*.lambda1) are already RPU fp16 and we can
# copy into freshly-allocated RPU keepalives without device-bouncing.
#
# Keep LayerScale γ as a runtime multiply: folding it into projection weights
# can push fp16 values into the subnormal range.


def _install_dinov3_rpu_keepalives(model: nn.Module) -> None:
    """Install RPU fp16 keepalive buffers required by the fused forward.

      - `_rpu_dinov3_cls_reg`   shape `[1+num_register, hidden]`
      - `_rpu_dinov3_gamma_attn` shape `[num_layers, hidden]`
      - `_rpu_dinov3_gamma_mlp`  shape `[num_layers, hidden]`

    All contiguous fp16 on `'rpu'`. Stacked per branch so the fused forward
    can index by layer id without per-layer attribute walks.
    """
    cfg = model.config
    hidden = int(cfg.hidden_size)
    num_reg = int(cfg.num_register_tokens)
    num_layers = len(model.layer)

    with torch.no_grad():
        cls = model.embeddings.cls_token.data.view(1, hidden)
        reg = model.embeddings.register_tokens.data.view(num_reg, hidden)
        model._rpu_dinov3_cls_reg = (
            torch.cat([cls, reg], dim=0)
            .to(dtype=torch.float16, device="rpu")
            .contiguous()
        )

        gamma_attn = torch.empty(
            (num_layers, hidden), dtype=torch.float16, device="rpu"
        )
        gamma_mlp = torch.empty(
            (num_layers, hidden), dtype=torch.float16, device="rpu"
        )
        for layer_idx, layer in enumerate(model.layer):
            gamma_attn[layer_idx].copy_(
                layer.layer_scale1.lambda1.data.to(torch.float16)
            )
            gamma_mlp[layer_idx].copy_(
                layer.layer_scale2.lambda1.data.to(torch.float16)
            )
        model._rpu_dinov3_gamma_attn = gamma_attn
        model._rpu_dinov3_gamma_mlp = gamma_mlp


# =============================================================================
# Section 3 — Patches: NONE needed for DINOv3
# =============================================================================
#
# DINOv3 uses LayerNorm (already aten op, no class swap), GELU (already
# aten op), and a function-based RoPE (`apply_rotary_pos_emb`), not a
# class. The 2D RoPE is invoked from the forward dispatch — no class
# patching required.
#
# No (a) layout patches, no (b) hw attr injection patches.


# =============================================================================
# Section 4 — Runtime: hardware attribute defaults
# =============================================================================

def _dinov3_single_chunk_size(num_tokens: int) -> int:
    """Return the 16-aligned capacity for one physical vision chunk."""
    return ((int(num_tokens) + 15) // 16) * 16


def _configured_vision_chunk(model: nn.Module) -> int:
    requested = getattr(model, "_rpu_execution", {}).get(
        "vision", {}).get("chunk_size", "auto")
    return 0 if requested == "auto" else int(requested)


def _apply_runtime_attrs(model: nn.Module) -> None:
    """Cold-set the per-instance hw attribute defaults."""
    model._rpu_chunk_size = _configured_vision_chunk(model)
    if not hasattr(model, "_rpu_spm_mode"):
        model._rpu_spm_mode = False
    if not hasattr(model, "_rpu_warmup"):
        model._rpu_warmup = 0
    if not hasattr(model, "_rpu_debug_export"):
        model._rpu_debug_export = False


# =============================================================================
# Section 2c — Fused C++ vision handle install + forward dispatch
# =============================================================================
#
# install_dinov3_vision_for_rpu wires the post-swizzle, post-keepalive model
# into the `torch.ops.rpu.dinov3_vision_*` C++ handle:
#   1. Gather per-layer weights/biases/γ and prepare RoPE/KV/GraphCache state.
#   2. Create and configure a pending handle (ModelHandleRegistry slot).
#   3. Pin the position_idx keepalive (Python writes per forward, C++ flushes).
#   4. Publish the handle and its tracked GC finalizer.
#   5. Retire a prior handle only after the replacement is ready.
#
# `_rpu_dinov3_forward` replaces the model's `forward` with a CPU patch_embed
# + RPU CLS+reg prepend + handle.forward + final-LN + CLS-slice dispatch.


def _destroy_dinov3_vision_handle(handle: int) -> None:
    try:
        torch.ops.rpu.dinov3_vision_destroy(handle)
    except Exception:
        pass


_DINOV3_HANDLE_INSTALL_ATTRS = frozenset({
    "_rpu_dinov3_handle",
    "_rpu_dinov3_handle_finalizer",
    "_rpu_dinov3_freq_cos",
    "_rpu_dinov3_freq_sin",
    "_rpu_dinov3_position_idx_keepalive",
    "_rpu_dinov3_kv_cache",
    "_rpu_dinov3_num_layers",
    "_rpu_dinov3_num_heads_padded",
    "_rpu_dinov3_head_dim",
    "_rpu_dinov3_num_special_tokens",
    "_rpu_dinov3_patch_size",
    "_rpu_dinov3_max_hw",
    "_rpu_dinov3_max_seq_len",
    "_rpu_dinov3_hidden_size",
    "_rpu_dinov3_graph_cache",
})


def _clear_dinov3_handle_install(vision_model: nn.Module) -> None:
    model_state = vars(vision_model)
    for name in _DINOV3_HANDLE_INSTALL_ATTRS:
        model_state.pop(name, None)


def _cleanup_dinov3_partial_runtime(vision_model: nn.Module) -> None:
    """Best-effort cleanup after the irreversible outer install has failed."""
    graph_cache = getattr(vision_model, "_rpu_dinov3_graph_cache", None)
    if graph_cache is not None and hasattr(graph_cache, "clear"):
        try:
            graph_cache.clear()
        except Exception:
            pass

    finalizer = getattr(
        vision_model, "_rpu_dinov3_handle_finalizer", None)
    handle = getattr(vision_model, "_rpu_dinov3_handle", None)
    if finalizer is not None and getattr(finalizer, "alive", False):
        finalizer()
    elif finalizer is None and handle is not None:
        _destroy_dinov3_vision_handle(handle)
    _clear_dinov3_handle_install(vision_model)


def install_dinov3_vision_for_rpu(
    vision_model: nn.Module,
    *,
    max_hw: int | None = None,
    max_seq_len: int | None = None,
) -> int:
    """Install or replace the DINOv3 native vision runtime transactionally."""
    snapshot = {
        name: getattr(vision_model, name)
        for name in _DINOV3_HANDLE_INSTALL_ATTRS
        if hasattr(vision_model, name)
    }
    install_state = {
        "old_handle": snapshot.get("_rpu_dinov3_handle"),
        "had_old_handle": "_rpu_dinov3_handle" in snapshot,
        "old_finalizer": snapshot.get("_rpu_dinov3_handle_finalizer"),
    }
    try:
        return _install_dinov3_vision_for_rpu_impl(
            vision_model,
            max_hw=max_hw,
            max_seq_len=max_seq_len,
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
                _destroy_dinov3_vision_handle(pending_handle)

            # Bypass a custom __setattr__ that may itself have caused the
            # interrupted tentative publication.
            state = vars(vision_model)
            for name in _DINOV3_HANDLE_INSTALL_ATTRS:
                state.pop(name, None)
            for name, value in snapshot.items():
                state[name] = value
        raise


def _make_dummy_dinov3_kv_caches(
    num_layers: int, max_seq_len: int, num_heads: int, head_dim: int,
) -> RPUCache:
    """Per-layer RPUCache sized for bidirectional vision SDPA.

    Vision encoder always inserts at position=0 and reads all tokens once —
    no autoregressive growth. We size to `max_seq_len` (worst-case num_tokens)
    so any forward shape fits.
    """
    return RPUCache(
        num_layers=num_layers,
        batch_size=1,
        max_seq_len=max_seq_len,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        attn_tp=min(_NUM_CORES, num_heads),
    )


def _install_dinov3_vision_for_rpu_impl(
    vision_model: nn.Module,
    *,
    max_hw: int | None = None,
    max_seq_len: int | None = None,
    _install_state: dict[str, Any],
) -> int:
    """Install the RPU C++ vision handle on a swizzled-and-moved DINOv3 model.

    Preconditions (all enforced by the adapter `to_rpu()` sequence):
      - `_convert_dinov3_weights_for_rpu` has run (patch_embed folded to CPU
        fp16, attention heads padded to multiple of NUM_CORES, all Linear
        weights swizzled).
      - `model.to('rpu')` has moved all params + tensors to fp16 RPU.
      - `_install_dinov3_rpu_keepalives` has stashed `_rpu_dinov3_cls_reg`,
        `_rpu_dinov3_gamma_attn`, `_rpu_dinov3_gamma_mlp`.

    Args:
        max_hw: max patch-grid side for the RoPE-2D table. Defaults to
            `image_size // patch_size` (e.g., 14 for 224×224, k=16).
        max_seq_len: bound for the dummy KV cache. Defaults to next-256
            multiple of `1 + num_register + (image_size/patch_size)²`.

    Returns the C++ handle (also stashed at `model._rpu_dinov3_handle`).
    """
    cfg = vision_model.config
    hidden = int(cfg.hidden_size)
    intermediate = int(cfg.intermediate_size)
    head_dim = hidden // int(cfg.num_attention_heads)
    num_heads_padded = _padded_num_heads(int(cfg.num_attention_heads))
    num_layers = len(vision_model.layer)
    num_register = int(cfg.num_register_tokens)
    num_special_tokens = 1 + num_register
    eps = float(cfg.layer_norm_eps)
    theta = float(getattr(cfg, "rope_theta", 100.0))
    image_size = int(cfg.image_size)
    patch_size = int(cfg.patch_size)
    grid_side = image_size // patch_size

    if max_hw is None:
        max_hw = grid_side
    if max_seq_len is None:
        max_seq_len = ((num_special_tokens + grid_side * grid_side + 255) // 256) * 256

    # Gather per-layer tensors. Weights are post-swizzle nn.Parameters; we
    # pass `.weight` directly so the C++ side holds the same storage without a
    # copy; the Python model object owns its lifetime.
    layers = list(vision_model.layer)
    q_w  = [L.attention.q_proj.weight for L in layers]
    k_w  = [L.attention.k_proj.weight for L in layers]
    v_w  = [L.attention.v_proj.weight for L in layers]
    o_w  = [L.attention.o_proj.weight for L in layers]
    up_w = [L.mlp.up_proj.weight  for L in layers]
    dn_w = [L.mlp.down_proj.weight for L in layers]

    ln1_w = [L.norm1.weight for L in layers]
    ln1_b = [L.norm1.bias   for L in layers]
    ln2_w = [L.norm2.weight for L in layers]
    ln2_b = [L.norm2.bias   for L in layers]

    q_b  = [L.attention.q_proj.bias for L in layers]
    v_b  = [L.attention.v_proj.bias for L in layers]
    o_b  = [L.attention.o_proj.bias for L in layers]
    up_b = [L.mlp.up_proj.bias      for L in layers]
    dn_b = [L.mlp.down_proj.bias    for L in layers]

    # γ keepalives are stacked `[num_layers, hidden]`. Unbind into
    # per-layer 1D rows so the C++ schema (TensorList[hidden]) is happy.
    ga_stack = vision_model._rpu_dinov3_gamma_attn
    gm_stack = vision_model._rpu_dinov3_gamma_mlp
    ga = [ga_stack[L].contiguous() for L in range(num_layers)]
    gm = [gm_stack[L].contiguous() for L in range(num_layers)]

    weight_args = (
        q_w, k_w, v_w, o_w, up_w, dn_w,
        ln1_w, ln1_b, ln2_w, ln2_b,
        q_b, v_b, o_b, up_b, dn_b,
        ga, gm,
        num_heads_padded, head_dim, hidden, intermediate, eps,
    )

    freq_cos, freq_sin = build_dinov3_rope_tables(
        head_dim=head_dim, max_hw=max_hw,
        device="rpu", dtype=torch.float16, theta=theta,
    )
    kv_cache = _make_dummy_dinov3_kv_caches(
        num_layers, max_seq_len, num_heads_padded, head_dim,
    )

    # Retain one fused encoder graph per admitted signature for stable replay.
    graph_cache = rpu_backend.graph.GraphCache()

    # Configure the pending native entry only after all Python-side state is
    # ready. The public wrapper owns immediate cleanup until commit.
    handle = torch.ops.rpu.dinov3_vision_create()
    _install_state["pending_handle"] = handle
    handle_finalizer = weakref.finalize(
        vision_model, _destroy_dinov3_vision_handle, handle)
    _install_state["pending_finalizer"] = handle_finalizer
    torch.ops.rpu.dinov3_vision_set_weights(handle, *weight_args)
    torch.ops.rpu.dinov3_vision_set_rope(handle, freq_cos, freq_sin)
    configured_chunk = _configured_vision_chunk(vision_model)
    if configured_chunk:
        torch.ops.rpu.dinov3_vision_set_chunk_size(handle, configured_chunk)
    keepalive = torch.ops.rpu.dinov3_vision_position_idx_keepalive(handle)

    _clear_dinov3_handle_install(vision_model)
    vision_model._rpu_dinov3_handle = handle
    vision_model._rpu_dinov3_handle_finalizer = handle_finalizer
    vision_model._rpu_dinov3_freq_cos = freq_cos
    vision_model._rpu_dinov3_freq_sin = freq_sin
    vision_model._rpu_dinov3_position_idx_keepalive = keepalive
    vision_model._rpu_dinov3_kv_cache = kv_cache
    vision_model._rpu_dinov3_num_layers = num_layers
    vision_model._rpu_dinov3_num_heads_padded = num_heads_padded
    vision_model._rpu_dinov3_head_dim = head_dim
    vision_model._rpu_dinov3_num_special_tokens = num_special_tokens
    vision_model._rpu_dinov3_patch_size = patch_size
    vision_model._rpu_dinov3_max_hw = int(max_hw)
    vision_model._rpu_dinov3_max_seq_len = int(max_seq_len)
    vision_model._rpu_dinov3_hidden_size = hidden
    vision_model._rpu_dinov3_graph_cache = graph_cache

    if (
        _install_state["had_old_handle"]
        and (
            _install_state["old_finalizer"] is None
            or getattr(_install_state["old_finalizer"], "alive", False)
        )
    ):
        torch.ops.rpu.dinov3_vision_destroy(_install_state["old_handle"])
    _install_state["committed"] = True
    old_finalizer = _install_state["old_finalizer"]
    if old_finalizer is not None and getattr(old_finalizer, "alive", False):
        old_finalizer.detach()

    return handle


def _validate_dinov3_input_bounds(
    num_patches_h: int,
    num_patches_w: int,
    num_tokens: int,
    *,
    max_hw: int,
    max_seq_len: int,
) -> None:
    """Fail before allocation/native dispatch when an image exceeds install bounds."""
    if num_patches_h > max_hw or num_patches_w > max_hw:
        raise RPUBackendError(
            "DINOv3 forward: patch grid "
            f"({num_patches_h},{num_patches_w}) exceeds the installed "
            f"2D-RoPE max_hw={max_hw}"
        )
    if num_tokens > max_seq_len:
        raise RPUBackendError(
            f"DINOv3 forward: num_tokens={num_tokens} exceeds the installed "
            f"KV-cache max_seq_len={max_seq_len}"
        )



def _rpu_dinov3_forward(self, pixel_values: torch.Tensor, **_kwargs: Any):
    """RPU-dispatching forward for DINOv3ViTModel.

    Pipeline:
      1. Move pixel_values to CPU fp16 (patch_embed runs CPU).
      2. Patch embed via folded `_rpu_dinov3_patch_embed_w` Linear.
      3. Prepend CLS + register tokens from `_rpu_dinov3_cls_reg`.
      4. Build per-forward [num_patches, 2] int16 position_idx, write to
         the C++ keepalive's first `num_patches` rows.
      5. Reset KV cache to position=0 (bidir vision is single-chunk).
      6. `torch.ops.rpu.dinov3_vision_forward(handle, input, k, v, num_tokens)`
         returns `[1, num_tokens, hidden]` fp16 RPU (pre final-LN).
      7. Apply `model.norm` (final LayerNorm, still an aten op on RPU).
      8. CLS slice as `pooler_output`.

    Returns `transformers.modeling_outputs.BaseModelOutputWithPooling` to mirror
    HF's `DINOv3ViTModel.forward` return signature.
    """
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    handle = self._rpu_dinov3_handle
    num_special_tokens = self._rpu_dinov3_num_special_tokens
    patch_size = self._rpu_dinov3_patch_size

    if pixel_values.dim() != 4:
        raise RPUBackendError(
            f"DINOv3 forward: pixel_values must be [B, 3, H, W], got "
            f"{tuple(pixel_values.shape)}"
        )
    if pixel_values.size(0) != 1:
        raise RPUBackendError(
            "DINOv3 forward: only batch=1 is supported "
            f"(got batch={pixel_values.size(0)})"
        )
    H, W = pixel_values.shape[-2], pixel_values.shape[-1]
    if H % patch_size != 0 or W % patch_size != 0:
        raise RPUBackendError(
            f"DINOv3 forward: image (H,W)=({H},{W}) must be divisible by "
            f"patch_size={patch_size}"
        )
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    num_tokens = num_special_tokens + num_patches
    _validate_dinov3_input_bounds(
        num_patches_h,
        num_patches_w,
        num_tokens,
        max_hw=self._rpu_dinov3_max_hw,
        max_seq_len=self._rpu_dinov3_max_seq_len,
    )
    configured_chunk = _configured_vision_chunk(self)
    required_chunk = _dinov3_single_chunk_size(num_tokens)
    if configured_chunk and configured_chunk != required_chunk:
        raise RPUBackendError(
            "DINOv3 vision chunk_size must equal this input's 16-aligned "
            f"single-chunk capacity {required_chunk} "
            f"for {num_tokens} logical tokens, got {configured_chunk}"
        )
    planned_chunk = int(
        torch.ops.rpu.dinov3_vision_resolve_chunk_size(handle, num_tokens)
    )
    if planned_chunk != required_chunk:
        raise RPUBackendError(
            "DINOv3 vision requires one physical chunk: native planner "
            f"resolved {planned_chunk}, expected {required_chunk} for "
            f"{num_tokens} logical tokens"
        )

    # Step 1-2: CPU fp16 patch embed (Conv2d k=ps s=ps == Linear on unfolded
    # patches). `_rpu_dinov3_patch_embed_w` is row-major `[hidden, 3·ps·ps]`
    # (output rows == flattened Conv2d kernel order: (c, kh, kw)). F.unfold
    # yields `[B, 3·ps·ps, num_patches]` in the same (c, kh, kw) inner order.
    px_cpu = pixel_values.to(device="cpu", dtype=torch.float16).contiguous()
    unfolded = F.unfold(px_cpu, kernel_size=patch_size, stride=patch_size)
    unfolded = unfolded.transpose(1, 2).contiguous()   # [B, num_patches, 3·ps·ps]
    patch_tokens_cpu = F.linear(
        unfolded,
        self._rpu_dinov3_patch_embed_w,
        self._rpu_dinov3_patch_embed_b,
    )  # [B, num_patches, hidden] CPU fp16

    # Step 3: prepend CLS + register tokens (RPU keepalive expanded to batch=1).
    patch_tokens = patch_tokens_cpu.to(device="rpu").contiguous()
    cls_reg = self._rpu_dinov3_cls_reg.unsqueeze(0)        # [1, R+1, hidden]
    sequence = torch.cat([cls_reg, patch_tokens], dim=1)   # [1, num_tokens, hidden]
    sequence = sequence.contiguous()

    # Step 4: position_idx → keepalive. CPU build is cheap; the C++ side
    # `rpu_ddr_flush_force`s the keepalive on every forward, picking up the
    # fresh values regardless of caching-allocator buffering.
    pos_idx = build_position_idx(num_patches_h, num_patches_w)
    keepalive = self._rpu_dinov3_position_idx_keepalive
    keepalive[:num_patches].copy_(pos_idx)

    # Step 5: per-layer KV cache reset (vision SDPA always reads from pos=0).
    cache = self._rpu_dinov3_kv_cache
    cache.reset_to_position(0)
    num_layers = self._rpu_dinov3_num_layers
    k_caches = [cache.k_caches[i] for i in range(num_layers)]
    v_caches = [cache.v_caches[i] for i in range(num_layers)]

    # Step 6: fused-graph forward (all layers, single REPLAY).
    # GraphSignature keyed on (num_tokens, hidden_size, num_layers) — each
    # unique num_tokens shape produces one BUILD + N REPLAYs.
    graph_cache = self._rpu_dinov3_graph_cache
    sig = rpu_backend.graph.GraphSignature(
        op_id="dinov3_vision",
        shapes=[num_tokens, self._rpu_dinov3_hidden_size],
        dyn_dims=[num_layers],
        dtypes=[torch.float16],
    )
    with graph_cache.capture(sig):
        raw = torch.ops.rpu.dinov3_vision_forward(
            handle, sequence, k_caches, v_caches, num_tokens,
        )
    resolved_chunk = int(
        torch.ops.rpu.dinov3_vision_get_resolved_chunk_size(handle))
    if resolved_chunk != planned_chunk:
        raise RPUBackendError(
            "DINOv3 vision dry/forward chunk plan drift: "
            f"dry={planned_chunk}, forward={resolved_chunk}, "
            f"logical_tokens={num_tokens}"
        )
    vars(self)["_rpu_last_execution_plan"] = {
        "vision": {
            "stage": "vision",
            "logical_len": num_tokens,
            "execution_len": num_tokens,
            "chunk_size": resolved_chunk,
            "padding_rows": 0,
            "position": 0,
        }
    }

    # Reset SPM temporary slots so subsequent RPU ops receive fresh workspace.
    torch.ops.rpu.spm_alloc_reset_temporary()

    # Step 8: final LayerNorm + CLS slice on CPU.
    # Running an eager LN on RPU immediately after the fused encoder competes
    # with the graph's DDR allocator reserve. The final LN is small, so it runs
    # on CPU after materializing the fused output there.
    raw_cpu = raw.detach().cpu()
    last_hidden_cpu = F.layer_norm(
        raw_cpu, normalized_shape=(self.config.hidden_size,),
        weight=self.norm.weight.detach().cpu(),
        bias=self.norm.bias.detach().cpu(),
        eps=self.config.layer_norm_eps,
    )
    pooler_output_cpu = last_hidden_cpu[:, 0, :].contiguous()

    # Move back to RPU to match HF's device contract. Callers may accumulate
    # these tensors across forwards, so each result needs independent storage;
    # `.to(device)` produces a fresh RPU tensor rather than a graph-buffer view.
    last_hidden = last_hidden_cpu.to(device="rpu")
    pooler_output = pooler_output_cpu.to(device="rpu")

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden,
        pooler_output=pooler_output,
    )


# =============================================================================
# RoPE-2D helpers
# =============================================================================


def build_dinov3_rope_tables(
    head_dim: int,
    max_hw: int,
    *,
    theta: float = 100.0,
    device: str | torch.device = "rpu",
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos / sin tables (`[max_hw, head_dim/4]` each) for the RPU
    `rope_2d_ddr` kernel using DINOv3's frequency convention.

    DINOv3 frequency math (from HF DINOv3ViTRopePositionEmbedding):
        cos_table[r, k] = cos(2π · norm_coord(r) · inv_freq[k])
        where:
          norm_coord(r) = 2·(r + 0.5)/max_hw − 1   ∈ [-1, +1]
          inv_freq[k]   = 1 / theta^(4k / head_dim)
          theta = 100   (DINOv3 config default; cf Qwen3-VL's 10000)

    The same table is indexed by both row and col axes by the kernel
    (DINOv3 uses identical math for both axes since num_patches_h ==
    num_patches_w).

    Args:
        head_dim: attention head dim (DINOv3-B/L: 64). Must be divisible
            by 4 (kernel splits into row/col halves of head_dim/4 each).
        max_hw: largest patch-grid side, e.g. 14 for 224×224, patch=16.
        theta: rope base; DINOv3 config default = 100.
        device: where to place the tables. Use "rpu" for live forward.
        dtype: fp16 for the kernel; fp32 when checking the math on CPU.

    Returns:
        (cos_table, sin_table), each shape `[max_hw, head_dim/4]`.
    """
    if head_dim % 4 != 0:
        raise ValueError(
            f"build_dinov3_rope_tables: head_dim must be divisible by 4, "
            f"got {head_dim}"
        )
    if max_hw <= 0:
        raise ValueError(
            f"build_dinov3_rope_tables: max_hw must be positive, got {max_hw}"
        )

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, 1, 4 / head_dim, dtype=torch.float64))
    )
    norm_coords = 2.0 * (
        torch.arange(0.5, max_hw, dtype=torch.float64) / max_hw
    ) - 1.0
    angles = 2 * math.pi * norm_coords[:, None] * inv_freq[None, :]
    cos = torch.cos(angles).to(dtype).contiguous()
    sin = torch.sin(angles).to(dtype).contiguous()
    return cos.to(device=device), sin.to(device=device)


def build_position_idx(
    num_patches_h: int, num_patches_w: int
) -> torch.Tensor:
    """Build `[num_patches_h*num_patches_w, 2]` int16 position_idx with
    (row_idx, col_idx) per patch in row-major order.

    Matches HF DINOv3's `get_patches_center_coordinates` ordering
    (`meshgrid(..., indexing='ij')` → flatten(0,1)).
    """
    row_idx = (
        torch.arange(num_patches_h, dtype=torch.long)[:, None]
        .expand(num_patches_h, num_patches_w).reshape(-1)
    )
    col_idx = (
        torch.arange(num_patches_w, dtype=torch.long)[None, :]
        .expand(num_patches_h, num_patches_w).reshape(-1)
    )
    return torch.stack([row_idx, col_idx], dim=-1).to(torch.int16)


_DINOV3_READY_ATTRS = _DINOV3_HANDLE_INSTALL_ATTRS | frozenset({
    "_rpu_dinov3_patch_embed_w",
    "_rpu_dinov3_patch_embed_b",
    "_rpu_dinov3_cls_reg",
    "_rpu_dinov3_gamma_attn",
    "_rpu_dinov3_gamma_mlp",
    "_rpu_chunk_size",
    "_rpu_spm_mode",
    "_rpu_warmup",
    "_rpu_debug_export",
})


def _dinov3_runtime_complete(model: nn.Module) -> bool:
    """Return whether the model owns a complete live DINOv3 runtime."""
    if getattr(model, "_rpu_swizzled", False) is not True:
        return False
    if getattr(model, "_rpu_swizzle_started", False) is not True:
        return False
    if any(not hasattr(model, name) for name in _DINOV3_READY_ATTRS):
        return False
    finalizer = getattr(model, "_rpu_dinov3_handle_finalizer", None)
    if finalizer is None or not getattr(finalizer, "alive", False):
        return False
    if getattr(model, "_rpu_dinov3_handle", None) is None:
        return False
    installed_forward = vars(model).get("forward")
    return bool(
        getattr(installed_forward, "__self__", None) is model
        and getattr(installed_forward, "__func__", None)
        is _rpu_dinov3_forward
    )


# =============================================================================
# Section 5 — Adapter class + registry binding
# =============================================================================


_SWIZZLE_LOCK = threading.Lock()


class DINOv3Adapter:
    """Per-instance adapter for DINOv3ViTModel.

    Lifecycle (two-step with an RPU shortcut):
      1. `__init__(model)` — preflight, install `.to('rpu')` interceptor.
      2. `to_rpu()` — irreversible swizzle + move to RPU + apply hw attrs.
         Returns the same model object.

    Forward dispatch uses `torch.ops.rpu.dinov3_vision_forward`. Calling
    `model(pixel_values)` after `to_rpu()` runs the fused-graph pipeline and
    returns `BaseModelOutputWithPooling(last_hidden_state, pooler_output)`.
    """

    @classmethod
    def preflight(cls, config: Any) -> None:
        """Fail-fast config-only check. Called BEFORE the HF weight load."""
        _check_profile(config)

    @classmethod
    def preflight_execution(cls, config: Any, execution_config) -> None:
        requested = execution_config.get("vision", {}).get(
            "chunk_size", "auto")
        if not isinstance(requested, int):
            return
        patches = (int(config.image_size) // int(config.patch_size)) ** 2
        logical_tokens = 1 + int(config.num_register_tokens) + patches
        required = _dinov3_single_chunk_size(logical_tokens)
        if requested != required:
            raise ValueError(
                "DINOv3Adapter: vision chunk_size must equal the profile's "
                f"16-aligned single-chunk capacity {required} for "
                f"{logical_tokens} logical tokens, got {requested}"
            )

    def __init__(self, model: nn.Module, *, rpu_execution=None) -> None:
        from rpu_backend.api._execution import (
            bind_rpu_execution,
            normalize_rpu_execution,
        )

        supported_execution = {"vision": ("chunk_size",)}
        normalized_execution = None
        if rpu_execution is not None:
            normalized_execution = normalize_rpu_execution(
                rpu_execution,
                entry_point="DINOv3Adapter",
                supported=supported_execution,
            )
        _check_profile(model.config)
        self._rpu_execution = bind_rpu_execution(
            model,
            normalized_execution if rpu_execution is not None else None,
            entry_point="DINOv3Adapter",
            supported=supported_execution,
        )
        self.preflight_execution(model.config, self._rpu_execution)
        self.model = model
        self._rpu_is_ready: bool = _dinov3_runtime_complete(model)

        # Idempotent `.to('rpu')` interceptor.
        if getattr(model.to, "__rpu_wrapped__", False):
            return

        original_to = model.to

        def rpu_aware_to(*args: Any, **kwargs: Any):
            target = extract_to_device_target(args, kwargs)

            if target is not None and is_rpu_device_target(
                target, entry_point="DINOv3Adapter.model.to"
            ):
                rejected = set(kwargs) - {"device"}
                if rejected:
                    raise ValueError(
                        f"model.to('rpu', ...) does not accept extra kwargs "
                        f"{sorted(rejected)}. Set per-instance _rpu_* attrs "
                        "after .to('rpu') instead."
                    )
                if len(args) > 1:
                    raise ValueError(
                        f"model.to('rpu', *args) does not accept extra "
                        f"positional args ({args[1:]!r})."
                    )
                return self.to_rpu()

            if (
                self._rpu_is_ready
                or getattr(self.model, "_rpu_swizzled", False)
                or getattr(self.model, "_rpu_swizzle_started", False)
            ):
                raise RPUBackendError(
                    f"model.to({target!r}) rejected: weights have been "
                    "swizzled for RPU and cannot be moved back safely. "
                    "Reload via from_pretrained() if you need a CPU copy."
                )
            return original_to(*args, **kwargs)

        rpu_aware_to.__rpu_wrapped__ = True
        model.to = rpu_aware_to

    def to_rpu(self) -> nn.Module:
        """Run the install sequence. Idempotent for the same model.

        Install sequence:
          A. Pre-move: fold patch_embed Conv2d to CPU fp16 Linear, zero-pad
             attention heads up to a multiple of NUM_CORES (no-op for ViT-L),
             swizzle every leaf Linear.
          B. Move model to RPU device.
          C. Install RPU fp16 keepalives (CLS+reg stacked, γ_attn / γ_mlp
             per-layer stacks).
          C.5. Transactionally install the fused C++ vision handle:
               set_weights / set_rope / position_idx_keepalive / dummy KV cache.
          D. Apply runtime hardware attrs.
          E. Replace forward with `_rpu_dinov3_forward` (CPU patch_embed +
             CLS+reg prepend + handle.forward + final LN + CLS pool).
        """
        if _dinov3_runtime_complete(self.model):
            self._rpu_is_ready = True
            return self.model
        if self._rpu_is_ready or getattr(self.model, "_rpu_swizzled", False):
            self._rpu_is_ready = False
            raise RPUBackendError(
                "DINOv3Adapter.to_rpu(): the ready marker exists but the "
                "vision runtime is incomplete. Reload from_pretrained()."
            )

        if getattr(self.model, "_rpu_swizzle_started", False):
            raise RPUBackendError(
                "DINOv3Adapter.to_rpu(): prior swizzle attempt left the "
                "model in undefined state. Reload from_pretrained()."
            )

        if not _SWIZZLE_LOCK.acquire(blocking=False):
            raise RPUBackendError(
                "DINOv3Adapter.to_rpu(): another swizzle in progress."
            )
        model_vars = vars(self.model)
        had_instance_forward = "forward" in model_vars
        old_instance_forward = model_vars.get("forward")
        install_started = False
        try:
            _claim_live_instance(self.model)
            self.model._rpu_swizzle_started = True
            install_started = True

            # ---------- Step A: pre-move CPU conversion --------------------
            # Linear swizzle + patch_embed Conv2d→Linear CPU fp16 fold.
            _convert_dinov3_weights_for_rpu(self.model)

            # ---------- Step B: move to RPU --------------------------------
            # Use the original .to to bypass our own interceptor.
            type(self.model).to(self.model, "rpu")

            # ---------- Step C: install RPU keepalives ---------------------
            # CLS+reg stacked buffer + per-layer γ stacks. Source params are
            # already on RPU after Step B.
            _install_dinov3_rpu_keepalives(self.model)

            # ---------- Step C.5: install fused C++ vision handle ----------
            install_dinov3_vision_for_rpu(self.model)

            # ---------- Step D: hw attrs -----------------------------------
            _apply_runtime_attrs(self.model)

            # ---------- Step E: forward replacement (real dispatch) -------
            self.model.forward = types.MethodType(
                _rpu_dinov3_forward, self.model,
            )

            _LOG.info(
                "DINOv3Adapter installed: profile=%s, "
                "num_heads_padded=%d, num_layers=%d",
                _profile(self.model.config),
                self.model._rpu_dinov3_num_heads_padded,
                self.model._rpu_dinov3_num_layers,
            )

            self._rpu_is_ready = True
            self.model._rpu_swizzled = True
            return self.model
        except BaseException:
            if install_started:
                _cleanup_dinov3_partial_runtime(self.model)
            model_state = vars(self.model)
            if had_instance_forward:
                model_state["forward"] = old_instance_forward
            else:
                model_state.pop("forward", None)
            self._rpu_is_ready = False
            raise
        finally:
            _SWIZZLE_LOCK.release()


register_adapter(DINOV3_VIT_ARCH, DINOv3Adapter)
