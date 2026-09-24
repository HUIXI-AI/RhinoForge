"""SigLIP all-layers-once instance patch (canonical home).

v5-03 B4: relocated from _internal/patches/pi05_all_layers_once.py L773-1013
per ADR §6.2. Pi05Adapter uses this direct-handle path as production
(v5-05 cutover SHIPPED; v2 shim deleted).

v5-cleanup: lifted from `adapters/pi05/siglip.py` to `adapters/siglip.py` as
a peer module so non-Pi0.5 ViT-using ports can reuse it directly; the
projector-wrapper polymorphism on L73 / L180 already supports both Pi0.5's
`projector.linear` wrapper and plain `nn.Linear` callers. ADR §3.3 forbids
reintroducing a ComponentBase abstraction; this file remains a flat
function module.

v5-05 D-02 + D-03: weight-prep helpers (`_convert_siglip_weights_for_rpu`,
`_convert_siglip_projector_for_rpu`, `patch_siglip_embeddings_for_rpu`) were
lifted byte-equal from `_internal/patches/siglip_pi05.py` (now deleted) so this
module is fully self-contained. The `patch_siglip_embeddings_for_rpu` helper's
forward-replacement (the old `rpu_embeddings_forward` → `fused_patch_embedding`
op call) was removed during the lift; the direct-handle path bypasses
`embeddings.forward` entirely (it routes pixel_values through
`siglip_forward(handle, 4D, ...)` via the C++ `siglip_model_set_patch_emb`).
"""
from __future__ import annotations
from dataclasses import replace
import os
import types

import torch
import torch.nn as nn

import rpu_backend  # GraphCache / GraphSignature
from rpu_backend.quant._common import quantize_linear_per_channel
from rpu_backend.runtime import rpu_env_bool
from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
from rpu_backend.runtime.log import _LOG
from rpu_backend.runtime.weights import (
    transform_linear_weight, convert_linear_weights_inplace,
    NUM_CORES,
)
from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime.execution_planner import GRAPH_COMPOSITE_CHILD


def _siglip_graph_enabled() -> bool:
    return rpu_env_bool("RPU_PI05_SIGLIP_GRAPH", default=True)


def _siglip_component_execution_plan(
    vision_model, handle: int, total_seq: int, image_batch_count: int,
    *, external_patch_prologue: bool, _cost_request=None,
    graph_cache=None,
):
    """Resolve and publish one native A6 plan for either SigLIP input path."""
    component_id = getattr(
        vision_model, "_fmb_execution_component_id", "vision_encoder"
    )
    component_config = getattr(
        vision_model, "_fmb_execution_component_config", {}
    )
    stage_config = component_config.get("vision", component_config)
    requested_chunk = stage_config.get("chunk_size", "auto")
    if _cost_request is not None:
        from rpu_backend.runtime.decoder import _cold_text_cost_request

        if type(handle) is not int or handle != vision_model._rpu_vision_handle:
            raise ValueError("cold SigLIP planning requires its actual handle")
        request, _ = _cold_text_cost_request(
            vision_model, ("siglip", handle), None, _cost_request, 0,
            stage="vision")
        if request.padding_rows != 0 or request.padding_budget != 0:
            raise ValueError("cold SigLIP planning cannot add execution padding")
        requested_chunk = request.chunk_size or "auto"
    generation = int(getattr(vision_model, "_fmb_execution_generation", 0))
    plan_box = {}
    execution_len, chunk_size = plan_bounded_prefill_execution(
        int(total_seq), int(total_seq), 0,
        resolve_stage_domain=lambda length: (
            torch.ops.rpu.siglip_resolve_stage_domain(
                handle, int(length), int(image_batch_count),
                bool(external_patch_prologue),
            )
        ),
        position=0,
        alignment=1,
        padding_rows=0,
        exact_chunk_size=(
            None if requested_chunk == "auto" else int(requested_chunk)
        ),
        request_id=f"{component_id}:vision",
        execution_owner=vision_model,
        execution_stage="vision",
        execution_native=("siglip", int(handle)),
        plan_result_sink=lambda result: plan_box.__setitem__("result", result),
        graph_mode=GRAPH_COMPOSITE_CHILD,
        physical_metadata=(
            (f"component:{component_id}", 1),
            ("execution_generation", generation),
            ("image_batch_count", int(image_batch_count)),
            ("position", 0),
            ("stage:vision", 1),
        ),
        queue_owner_id=int(handle),
        plan_signature=(int(image_batch_count), bool(external_patch_prologue)),
        graph_cache=graph_cache,
    )
    plan = plan_box["result"]
    if (
        execution_len != int(total_seq)
        or plan.selected is None
        or chunk_size != plan.selected.stage_tuple.compute_chunk
        or not plan.selected.stage_tuple.physical_descriptor
    ):
        raise RuntimeError(
            "SigLIP planner returned no consumable native stage descriptor"
        )
    if _cost_request is None:
        vars(vision_model)["_fmb_last_execution_plan"] = plan
    return plan


# GraphSignature branch discriminator for the "pi05_siglip_compute" capture.
#
# The C++ `siglip_forward` auto-detects input rank (D-507) and records TWO
# STRUCTURALLY DIFFERENT graphs on one handle:
#   - 4D [N,C,H,W] pixel input → patch-embedding prologue first (node 0 is the
#     im2col input DMA), then the encoder.
#   - 3D [1,S,hidden] pre-embedded input → straight into the encoder (node 0 is
#     a kernel).
# Every other signature term is identical for the canonical pair
# ([1,3,224,224] vs [1,256,1152]): shapes[0] = _n_packed*256 = 256 either way,
# because `_n_packed` is `input.shape[0]` and equals 1 for both. Without this
# discriminator the two recordings collide on ONE cache entry and whichever
# runs second REPLAYs the other's node list — the graph runtime then aborts at
# cursor 0 ("kind mismatch ... expected Dma got 0" / "kernel_id mismatch in
# replay at node 1"). `branch_key` is the framework's designated control-flow
# discriminator (graph_infra.h GraphSignature; it participates in operator==
# and the hash, and drives MISS_BRANCH_KEY attribution).
_SIGLIP_BRANCH_PATCH_EMBED = 1   # 4D pixels → patch-embed prologue + encoder
_SIGLIP_BRANCH_PRE_EMBEDDED = 2  # 3D hidden → encoder only


# MR-B / T2 — the 3D pre-embedded branch is pinned to exactly SIGLIP_SEQ_LEN rows.
#
# The trap this closes: the GraphSignature for this path keys its row count as
# `_n_packed * _seq_len`, and for a 3D [1, S, hidden] input `_n_packed` is the
# BATCH dim (1), not S. So the signature reads 256 for EVERY 3D input whatever S
# is — two different row counts would share one cache entry and the second would
# replay the first's node list, silently, with correct shapes and wrong numbers.
#
# Why a guard rather than a generalised key: no caller needs non-256 HERE. HALO
# does run the encoder at n_vit=1564, but through the NATIVE path with
# `shapes=[n_vit, ...]` — its real row count — under its own op_id and its own
# GraphCache (adapters/halo/vit_cache.py:342). HALO is already correct, so
# generalising this key would have zero consumers; what is actually left is a
# trap for the next caller. Generalise only when a real non-256 caller appears on
# THIS path (followups_plan_20260811.md §1 B4).
SIGLIP_SEQ_LEN = 256


def check_pre_embedded_rows(shape, seq_len=SIGLIP_SEQ_LEN):
    """Raise unless a 3D pre-embedded SigLIP input has exactly `seq_len` rows.

    No-op for non-3D shapes (the 4D patch-embed branch keys on the real N).
    """
    if len(shape) != 3 or shape[1] == seq_len:
        return
    raise ValueError(
        f"SigLIP pre-embedded (3D) input must have exactly {seq_len} rows, got "
        f"{shape[1]} (shape {tuple(shape)}). This path keys its GraphCache entry "
        f"on _n_packed * {seq_len}, which ignores the actual row count, so a "
        f"different S would silently REPLAY the {seq_len}-row graph. If you need "
        f"another length, follow HALO: drive the native handle with "
        f"shapes=[rows, ...] under your own op_id and GraphCache "
        f"(adapters/halo/vit_cache.py:342).")


def _siglip_w8a16_enabled(default_enabled: bool = False) -> bool:
    # Unset → caller default. The Pi05 w8a16 path passes default_enabled=True so
    # SigLIP W8A16 is on by default for w8a16 models (fp16 models pass False).
    # An explicit RPU_PI05_SIGLIP_W8A16 value always overrides — including "0"
    # to force SigLIP back to fp16 on a w8a16 model.
    return rpu_env_bool("RPU_PI05_SIGLIP_W8A16", default=default_enabled)


def _siglip_w8a16_projection_names(default_enabled: bool = False) -> set[str]:
    if not _siglip_w8a16_enabled(default_enabled=default_enabled):
        return set()
    raw = os.environ.get("RPU_PI05_SIGLIP_W8A16_SCOPE", "all").strip().lower()
    if raw in ("", "0", "none", "off", "false"):
        return set()
    all_names = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.out_proj", "mlp.fc1", "mlp.fc2",
    }
    if raw in ("all", "full"):
        return all_names
    if raw in ("row", "safe"):
        return {"self_attn.out_proj", "mlp.fc2"}
    if raw in ("out", "out_proj", "o"):
        return {"self_attn.out_proj"}
    if raw in ("fc2", "mlp.fc2"):
        return {"mlp.fc2"}
    raise RuntimeError(
        "RPU_PI05_SIGLIP_W8A16_SCOPE must be one of row,out_proj,fc2,all,none; "
        f"got {raw!r}"
    )


def _set_siglip_weight_scale(module: nn.Module, scale: torch.Tensor) -> None:
    scale = scale.to(torch.float16).contiguous()
    if "weight_scale" in module._buffers:
        module._buffers["weight_scale"] = scale
    else:
        if hasattr(module, "weight_scale"):
            delattr(module, "weight_scale")
        module.register_buffer("weight_scale", scale)


def _quantize_and_swizzle_siglip_weight(
    weight: torch.Tensor,
    *,
    partition: int,
    num_cores: int = NUM_CORES,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.device.type != "cpu":
        raise RuntimeError(
            "RPU_PI05_SIGLIP_W8A16 requires SigLIP encoder conversion before "
            "moving weights to RPU"
        )
    w_int8, scale = quantize_linear_per_channel(weight.contiguous())
    w_swizzled = transform_linear_weight(
        w_int8.contiguous(), partition=partition, num_cores=num_cores)
    return w_swizzled.contiguous(), scale.contiguous()


def _install_siglip_linear_weight(
    module: nn.Linear,
    weight: torch.Tensor,
    *,
    partition: int,
    w8a16: bool,
    num_cores: int = NUM_CORES,
) -> None:
    if w8a16:
        weight, scale = _quantize_and_swizzle_siglip_weight(
            weight, partition=partition, num_cores=num_cores)
        module.weight = nn.Parameter(weight, requires_grad=False)
        _set_siglip_weight_scale(module, scale)
    else:
        weight = transform_linear_weight(
            weight.contiguous(), partition=partition, num_cores=num_cores)
        module.weight = nn.Parameter(weight.contiguous(), requires_grad=False)


def _iter_siglip_encoder_projections(encoder):
    for layer_idx, layer in enumerate(encoder.layers):
        sa = layer.self_attn
        mlp = layer.mlp
        yield layer_idx, "self_attn.q_proj", sa.q_proj
        yield layer_idx, "self_attn.k_proj", sa.k_proj
        yield layer_idx, "self_attn.v_proj", sa.v_proj
        yield layer_idx, "self_attn.out_proj", sa.out_proj
        yield layer_idx, "mlp.fc1", mlp.fc1
        yield layer_idx, "mlp.fc2", mlp.fc2


def _detect_and_validate_siglip_encoder_w8a16(encoder) -> bool:
    projections = list(_iter_siglip_encoder_projections(encoder))
    allowed_int8 = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.out_proj", "mlp.fc1", "mlp.fc2",
    }
    int8_modules = [
        (layer_idx, name, module)
        for layer_idx, name, module in projections
        if module.weight.dtype == torch.int8
    ]
    if not int8_modules:
        return False

    unexpected = [
        f"encoder.layers[{layer_idx}].{name}"
        for layer_idx, name, _ in int8_modules
        if name not in allowed_int8
    ]
    if unexpected:
        raise RuntimeError(
            "SigLIP W8A16 validation failed: unsupported int8 projection(s): "
            + ", ".join(unexpected)
        )

    for layer_idx, name, module in int8_modules:
        scale = getattr(module, "weight_scale", None)
        fq_name = f"encoder.layers[{layer_idx}].{name}"
        if scale is None or scale.dtype != torch.float16:
            raise RuntimeError(
                f"SigLIP W8A16 validation failed: int8 {fq_name}.weight "
                "requires fp16 weight_scale"
            )
        if scale.dim() != 1 or scale.numel() != module.weight.size(0):
            raise RuntimeError(
                f"SigLIP W8A16 validation failed: {fq_name}.weight_scale "
                f"shape {tuple(scale.shape)} incompatible with output dim "
                f"{module.weight.size(0)}"
            )
    return True


def _siglip_weight_to_rpu(module: nn.Linear) -> torch.Tensor:
    if module.weight.dtype == torch.int8:
        return module.weight.to(device="rpu").contiguous()
    return module.weight.to(dtype=torch.float16, device="rpu").contiguous()


def _siglip_scale_to_rpu(module: nn.Linear) -> torch.Tensor:
    scale = getattr(module, "weight_scale", None)
    if scale is None:
        raise RuntimeError("SigLIP W8A16 weight_scale missing")
    return scale.to(dtype=torch.float16, device="rpu").contiguous()


# patch-reason: (a) SigLIP all-layers-once C++ handle lifecycle helper — §3a (a)
def _siglip_destroy_handle(h):
    """Raw destroy; the installed resource owns retirement failures."""
    torch.ops.rpu.siglip_destroy(h)


# =============================================================================
# v5-05 D-02: SigLIP weight-prep helpers (lifted byte-equal from
# `_internal/patches/siglip_pi05.py`; that file is now deleted).
# =============================================================================
# These were the substantive weight-prep bodies from siglip_converter.py
# through v4.0 (relocated to siglip_pi05.py in Phase 12 D-B1 and again to here
# in v5-05 D-02 to keep the direct-handle path self-contained). Idempotence
# guards on `_rpu_siglip_weights_converted` and `_rpu_weights_converted`
# preserved verbatim; they short-circuit double-call from a pre-move CPU
# swizzle (Pi05Adapter._init_:154) followed by the inner call inside
# `patch_siglip_model_for_rpu_all_layers_once` (line 108 below).


def _convert_siglip_weights_for_rpu(vision_tower, *, w8a16_default: bool = False, num_cores: int = 8):
    """Convert SigLIP encoder weights: pad head_dim/intermediate, swizzle for RPU.

    Modifications:
      - Q/K/V weights: pad head_dim 72->80, col swizzle
      - Q/K/V biases: pad head_dim 72->80
      - O_proj weight: pad head_dim 72->80, row swizzle
      - fc1 weight: pad intermediate 4304->4352, col swizzle
      - fc1 bias: pad intermediate 4304->4352
      - fc2 weight: pad intermediate 4304->4352, row swizzle
      - LayerNorm weight/bias: no change (replicated on all cores by C++)

    Idempotent: guards on the normalized vit object to prevent double-swizzle
    regardless of whether caller passes wrapper or inner vit.
    """
    # Normalize wrapper vs inner
    vit = vision_tower.vision_model if hasattr(vision_tower, 'vision_model') else vision_tower
    if type(num_cores) is not int or num_cores not in (4, 8):
        raise ValueError("SigLIP execution core count must be 4 or 8")
    siglip_w8a16_names = _siglip_w8a16_projection_names(default_enabled=w8a16_default)
    siglip_w8a16 = bool(siglip_w8a16_names)

    # Idempotence guard on the NORMALIZED vit object
    if getattr(vit, '_rpu_siglip_weights_converted', False):
        if getattr(vit, '_rpu_vision_num_cores', 8) != num_cores:
            raise ValueError('SigLIP core count is cold-only; reload before changing its weight layout')
        if siglip_w8a16 and not getattr(vit, '_siglip_w8a16', False):
            _LOG.warning(
                "SigLIP W8A16 requested after encoder was already converted as fp16; "
                "reload the model to apply RPU_PI05_SIGLIP_W8A16=1")
        return

    encoder = vit.encoder
    if num_cores == 4 and (len(encoder.layers) != 27 or
            (vit.config.hidden_size, vit.config.intermediate_size, vit.config.num_attention_heads) != (1152, 4304, 16)):
        raise ValueError("SigLIP4 requires the exact Pi0.5 SigLIP27 geometry")

    layer0 = encoder.layers[0]
    hidden_size = layer0.self_attn.embed_dim
    num_heads = layer0.self_attn.num_heads
    orig_head_dim = hidden_size // num_heads
    orig_intermediate = layer0.mlp.fc1.out_features
    num_layers = len(encoder.layers)

    # Pad to satisfy RPU kernel alignment: head_dim % 16 == 0, intermediate % 128 == 0
    padded_head_dim = ((orig_head_dim + 15) // 16) * 16       # 72 -> 80
    padded_intermediate = ((orig_intermediate + 127) // 128) * 128  # 4304 -> 4352
    padded_qkv_out = num_heads * padded_head_dim               # 16 * 80 = 1280

    _LOG.info("SigLIP: %d layers, hidden=%d, heads=%d",
              num_layers, hidden_size, num_heads)
    _LOG.info("  head_dim: %d -> %d, intermediate: %d -> %d",
              orig_head_dim, padded_head_dim, orig_intermediate, padded_intermediate)
    if siglip_w8a16:
        _LOG.info("  SigLIP encoder Linear weights: W8A16 enabled for %s",
                  ",".join(sorted(siglip_w8a16_names)))

    for layer_idx, layer in enumerate(encoder.layers):
        attn = layer.self_attn
        mlp = layer.mlp

        with torch.no_grad():
            # === Q/K/V weights: [orig_out, hidden] -> pad -> [padded_out, hidden] -> col swizzle ===
            for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                proj = getattr(attn, proj_name)
                w = proj.weight.data  # [num_heads*orig_hd, hidden_size]
                w = w.view(num_heads, orig_head_dim, hidden_size)
                if padded_head_dim > orig_head_dim:
                    pad = torch.zeros(num_heads, padded_head_dim - orig_head_dim, hidden_size,
                                      dtype=w.dtype, device=w.device)
                    w = torch.cat([w, pad], dim=1)
                w = w.reshape(padded_qkv_out, hidden_size).contiguous()
                _install_siglip_linear_weight(
                    proj, w, partition=1,
                    w8a16=(f"self_attn.{proj_name}" in siglip_w8a16_names), num_cores=num_cores)
                proj.out_features = padded_qkv_out

                # Bias: [num_heads*orig_hd] -> pad -> [num_heads*padded_hd]
                b = proj.bias.data.view(num_heads, orig_head_dim)
                if padded_head_dim > orig_head_dim:
                    bpad = torch.zeros(num_heads, padded_head_dim - orig_head_dim,
                                       dtype=b.dtype, device=b.device)
                    b = torch.cat([b, bpad], dim=1)
                proj.bias = nn.Parameter(b.reshape(-1).contiguous(), requires_grad=False)

            # === O_proj (out_proj): [hidden, orig_in] -> pad -> [hidden, padded_in] -> row swizzle ===
            w = attn.out_proj.weight.data  # [hidden_size, num_heads*orig_hd]
            w = w.view(hidden_size, num_heads, orig_head_dim)
            if padded_head_dim > orig_head_dim:
                pad = torch.zeros(hidden_size, num_heads, padded_head_dim - orig_head_dim,
                                  dtype=w.dtype, device=w.device)
                w = torch.cat([w, pad], dim=2)
            w = w.reshape(hidden_size, padded_qkv_out).contiguous()
            _install_siglip_linear_weight(
                attn.out_proj, w, partition=0,
                w8a16=("self_attn.out_proj" in siglip_w8a16_names), num_cores=num_cores)
            attn.out_proj.in_features = padded_qkv_out
            # O_proj bias: [hidden_size] -- no padding needed

            # === fc1: [orig_inter, hidden] -> pad -> [padded_inter, hidden] -> col swizzle ===
            w = mlp.fc1.weight.data
            if padded_intermediate > orig_intermediate:
                pad = torch.zeros(padded_intermediate - orig_intermediate, hidden_size,
                                  dtype=w.dtype, device=w.device)
                w = torch.cat([w, pad], dim=0)
            _install_siglip_linear_weight(
                mlp.fc1, w.contiguous(), partition=1,
                w8a16=("mlp.fc1" in siglip_w8a16_names), num_cores=num_cores)
            mlp.fc1.out_features = padded_intermediate

            # fc1 bias: [orig_inter] -> pad -> [padded_inter]
            b = mlp.fc1.bias.data
            if padded_intermediate > orig_intermediate:
                bpad = torch.zeros(padded_intermediate - orig_intermediate,
                                   dtype=b.dtype, device=b.device)
                b = torch.cat([b, bpad], dim=0)
            mlp.fc1.bias = nn.Parameter(b.contiguous(), requires_grad=False)

            # === fc2: [hidden, orig_inter] -> pad -> [hidden, padded_inter] -> row swizzle ===
            w = mlp.fc2.weight.data
            if padded_intermediate > orig_intermediate:
                pad = torch.zeros(hidden_size, padded_intermediate - orig_intermediate,
                                  dtype=w.dtype, device=w.device)
                w = torch.cat([w, pad], dim=1)
            _install_siglip_linear_weight(
                mlp.fc2, w.contiguous(), partition=0,
                w8a16=("mlp.fc2" in siglip_w8a16_names), num_cores=num_cores)
            mlp.fc2.in_features = padded_intermediate
            # fc2 bias: [hidden_size] -- no padding needed

        # Store metadata for fused forward
        layer._rpu_layer_idx = layer_idx
        layer._rpu_num_layers = num_layers
        layer._rpu_num_heads = num_heads
        layer._rpu_head_dim = padded_head_dim
        layer._rpu_intermediate_size = padded_intermediate

    vit._rpu_vision_num_cores = num_cores
    vit._rpu_siglip_weights_converted = True
    vit._siglip_w8a16 = siglip_w8a16
    suffix = " (W8A16)" if siglip_w8a16 else ""
    _LOG.info("Converted %d SigLIP encoder layers%s", num_layers, suffix)


def _convert_siglip_projector_for_rpu(projector, *, num_cores: int = 8):
    """Convert the actual leaf once, with the native projector's column layout."""
    if type(num_cores) is not int or num_cores not in (4, 8):
        raise ValueError("SigLIP projector core count must be 4 or 8")
    leaf = projector.linear if hasattr(projector, 'linear') else projector
    if getattr(projector, '_rpu_weights_converted', False) or getattr(leaf, '_rpu_weights_converted', False):
        if getattr(leaf, '_rpu_linear_num_cores', 8) != num_cores:
            raise ValueError("SigLIP projector core count is cold-only; reload the model")
        return
    leaf.weight.data = transform_linear_weight(leaf.weight.data, partition=1, num_cores=num_cores)
    leaf._rpu_linear_num_cores = num_cores
    leaf._rpu_linear_partition = 1
    projector._rpu_weights_converted = True
    leaf._rpu_weights_converted = True


# patch-reason: (a) Pi0.5 SigLIP embedding GEMM packing for RPU patch-emb kernel — §3a (a) hardware-constraint attribute injection
# v5-05 D-03: forward replacement removed; direct handle path calls siglip_forward(handle, 4D, ...) directly.
def patch_siglip_embeddings_for_rpu(embeddings_module, num_cores=8):
    """Patch SiglipVisionEmbeddings for RPU fused patch embedding.

    Converts Conv2d(cin=3, cout=1152, k=14, s=14) into im2col + GEMM pipeline:
      - Weight: pad cin 3→16, reshape [cout, cin*kh*kw] = [1152, 3136], col-swizzle
      - Pre-fuse bias + position embedding: pos_emb_fused = pos_emb + conv.bias

    The direct-handle path consumes the resulting `_rpu_gemm_weight` and
    `_rpu_pos_emb_fused` buffers via `siglip_model_set_patch_emb` (called by
    `patch_siglip_model_for_rpu_all_layers_once` Step 7); the legacy forward
    replacement that called `torch.ops.rpu.fused_patch_embedding` directly was
    removed in v5-05 D-03 because the direct-handle C++ pipeline routes 4D
    pixel_values through `siglip_forward(handle, 4D, ...)`.
    """
    conv = embeddings_module.patch_embedding
    cin_orig = conv.in_channels  # 3
    cin_padded = 16
    cout = conv.out_channels     # 1152
    kh = conv.kernel_size[0]     # 14
    kw = conv.kernel_size[1]     # 14
    K = cin_padded * kh * kw     # 3136

    with torch.no_grad():
        w = conv.weight.data.half()  # [cout, cin_orig, kh, kw]
        # Pad cin: [cout, 3, 14, 14] → [cout, 16, 14, 14]
        if cin_padded > cin_orig:
            pad_w = torch.zeros(cout, cin_padded - cin_orig, kh, kw,
                                dtype=w.dtype, device=w.device)
            w = torch.cat([w, pad_w], dim=1)

        # --- GEMM weight: reshape to [cout, K] and apply col-swizzle (single-core) ---
        # Conv2d weight [cout, cin_padded, kh, kw] → im2col-compatible [cout, kh*kw*cin_padded]
        # im2col output layout is [outhw, flth*fltw*cin] where inner order is (fh, fw, c)
        # So weight must match: reshape as [cout, kh, kw, cin_padded] then flatten last 3 dims
        # Single-core because K=3136 not divisible by 128 (row-partition alignment req)
        w_gemm = w.permute(0, 2, 3, 1).reshape(cout, K).contiguous()  # [1152, 3136]
        w_gemm_swizzled = transform_linear_weight(w_gemm, partition=1, num_cores=1)
        embeddings_module.register_buffer('_rpu_gemm_weight',
                                          w_gemm_swizzled.to("rpu"), persistent=False)

        # --- Pre-fuse bias + position embedding ---
        # all_reduce_sum_residual(GEMM_partial, pos_emb_fused) = sum(partials) + pos_emb + bias
        pos_emb = embeddings_module.position_embedding(
            embeddings_module.position_ids).half()  # [1, num_patches, cout]
        if conv.bias is not None:
            # Fuse bias into pos_emb: [1, num_patches, cout] + [1, 1, cout]
            pos_emb_fused = pos_emb + conv.bias.data.half().view(1, 1, -1)
        else:
            pos_emb_fused = pos_emb
        embeddings_module.register_buffer(
            '_rpu_pos_emb_fused', pos_emb_fused.contiguous().to("rpu"), persistent=False)

    # Store params for fused path
    embeddings_module._rpu_cin_orig = cin_orig
    embeddings_module._rpu_cin_padded = cin_padded
    embeddings_module._rpu_stride = conv.stride[0]
    embeddings_module._rpu_cout = cout
    embeddings_module._rpu_kh = kh
    embeddings_module._rpu_kw = kw

    _LOG.info("Patched SigLIP embeddings: cin %d→%d, GEMM weight [%d, %d] "
              "single-core (direct-handle path consumes via siglip_model_set_patch_emb)",
              cin_orig, cin_padded, cout, K)


# patch-reason: (a) SigLIP all-layers-once instance patch — §3a (a) run-different-op-on-RPU
_SIGLIP_RUNTIME_INSTALL_ATTRS = (
    "_test_siglip_weight_args_tuple",
    "_siglip_w8a16",
    "_rpu_cache",
    "_rpu_siglip_graph_cache",
    "_rpu_lazy_init_checked",
    "_rpu_required_attrs",
    "_rpu_vision_handle",
    "_rpu_vision_handle_finalizer",
    "_rpu_vision_retirement_state",
    "_siglip_graph_enabled",
    "forward",
)


def patch_siglip_model_for_rpu_all_layers_once(vision_model, projector,
                                               num_images: int = 1,
                                               w8a16_default: bool = False,
                                               num_cores: int = 8,
                                               *, runtime_policy=None, linear_acc32=None) -> int:
    """
    Patch a Pi0.5 SigLIP vision model instance to use all-layers-once
    C++ execution via SigLIPModel (FusedModelBase).

    SigLIP-batch (2026-05-31): with num_images > 1, the patched forward expects
    pixel_values to be [num_images, 3, H, W] — N camera images packed into ONE
    seq=N*256 fused encoder forward (per-image attention isolation via the
    minibatch SDPA). The KV cache is sized for up to num_images (N*256 rows);
    the per-forward GraphSignature is keyed on the ACTUAL packed N so single-
    and multi-image graphs never collide. num_images = 1 (default) is the legacy
    single-image path, byte-identical to before, used by standalone SigLIP +
    non-Pi0.5 ViT ports.

    Steps:
      1. Keep the currently published handle alive while preparing replacement state
      2. Convert weights (encoder + projector) -- idempotence guards inside each
      3. Gather per-layer weights (16 per layer + 4 global tensors)
      4. Prepare patch embedding, cache, and replacement forward state
      5. Create and fully configure a pending C++ handle
      6. Tentatively publish and validate the replacement state
      7. Retire the old handle and commit the replacement

    Args:
        vision_model: SigLIP vision model (either wrapper with .vision_model or
                      the inner vit directly)
        projector: Projector module (Pi0.5 wrapper with .linear, or plain nn.Linear)

    Returns:
        int: handle for this model. Also stored as vision_model._rpu_vision_handle.
    """
    # Keep the published install intact until its replacement is completely
    # configured. This matters for direct re-patching (for example lifecycle
    # tests and adapter recovery): a failed set_weights must not leave a model
    # pointing at a half-initialized handle.
    if linear_acc32 is not None and type(linear_acc32) is not bool:
        raise TypeError("SigLIP linear_acc32 must be bool or None")

    from rpu_backend.api._execution import _require_execution_process_safe
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource

    _require_execution_process_safe()
    if type(num_cores) is not int or num_cores not in (4, 8) or (num_cores == 4 and num_images != 1):
        raise ValueError("SigLIP4 requires the fixed serial one-image profile")
    if runtime_policy is not None:
        if runtime_policy.execution_core_count < num_cores:
            raise ValueError("SigLIP Graph policy cannot run fewer cores than its weights")
        if runtime_policy.execution_core_count != num_cores:
            # Pi0.5 MLP6 has a six-core root but its serial SigLIP child uses
            # four. Keep the root's replay choices while narrowing this queue.
            runtime_policy = replace(
                runtime_policy, execution_core_count=num_cores,
                provenance=runtime_policy.provenance +
                (f"owner:siglip:execution_core_count={num_cores}",),
            )
    old_resource = getattr(vision_model, "_rpu_vision_retirement_state", None)
    if old_resource is not None:
        old_resource.require_replaceable()
    elif getattr(vision_model, "_rpu_vision_handle", None) is not None:
        raise RuntimeError("SigLIP replacement requires its actual retirement resource")
    model_state = vars(vision_model)
    install_snapshot = {
        name: model_state[name]
        for name in _SIGLIP_RUNTIME_INSTALL_ATTRS
        if name in model_state
    }
    old_finalizer = install_snapshot.get(
        "_rpu_vision_handle_finalizer")

    # ------------------------------------------------------------------ #
    # Step 1: extract config from vision_model
    # ------------------------------------------------------------------ #
    encoder = vision_model.encoder
    config = vision_model.config
    num_layers = len(encoder.layers)
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads  # 72 original (will be 80 after pad)
    hidden_size = config.hidden_size  # 1152
    intermediate_size = config.intermediate_size  # 4304 original (4352 after pad)
    # Pi0.5 projector is a wrapper with .linear attribute (pi05_converter.py:1039-1041)
    # Support both nn.Linear (has .weight directly) and wrapper (has .linear.weight)
    if hasattr(projector, 'linear'):
        _proj_linear = projector.linear  # Pi0.5 wrapper: projector_module.linear
    else:
        _proj_linear = projector  # Plain nn.Linear
    projection_dim = _proj_linear.out_features  # 2048
    eps = config.layer_norm_eps

    # ------------------------------------------------------------------ #
    # Step 2: weight conversion (sole owner) -- idempotence guards inside
    # ------------------------------------------------------------------ #
    # v5-05 D-02: helpers are now module-private (defined above).
    # Encoder conversion
    _convert_siglip_weights_for_rpu(vision_model, w8a16_default=w8a16_default, num_cores=num_cores)

    # Projector conversion
    _convert_siglip_projector_for_rpu(projector, num_cores=num_cores)
    siglip_w8a16 = _detect_and_validate_siglip_encoder_w8a16(encoder)

    # ------------------------------------------------------------------ #
    # Step 3: gather per-layer weights
    # ------------------------------------------------------------------ #
    q_w_list, k_w_list, v_w_list, o_w_list = [], [], [], []
    q_ws_list, k_ws_list, v_ws_list, o_ws_list = [], [], [], []
    fc1_w_list, fc2_w_list = [], []
    fc1_ws_list, fc2_ws_list = [], []
    ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list = [], [], [], []
    q_b_list, k_b_list, v_b_list, o_b_list = [], [], [], []
    fc1_b_list, fc2_b_list = [], []

    for layer in encoder.layers:
        sa = layer.self_attn
        mlp = layer.mlp

        q_w_list.append(_siglip_weight_to_rpu(sa.q_proj))
        k_w_list.append(_siglip_weight_to_rpu(sa.k_proj))
        v_w_list.append(_siglip_weight_to_rpu(sa.v_proj))
        o_w_list.append(_siglip_weight_to_rpu(sa.out_proj))

        fc1_w_list.append(_siglip_weight_to_rpu(mlp.fc1))
        fc2_w_list.append(_siglip_weight_to_rpu(mlp.fc2))
        if sa.q_proj.weight.dtype == torch.int8:
            q_ws_list.append(_siglip_scale_to_rpu(sa.q_proj))
        if sa.k_proj.weight.dtype == torch.int8:
            k_ws_list.append(_siglip_scale_to_rpu(sa.k_proj))
        if sa.v_proj.weight.dtype == torch.int8:
            v_ws_list.append(_siglip_scale_to_rpu(sa.v_proj))
        if sa.out_proj.weight.dtype == torch.int8:
            o_ws_list.append(_siglip_scale_to_rpu(sa.out_proj))
        if mlp.fc1.weight.dtype == torch.int8:
            fc1_ws_list.append(_siglip_scale_to_rpu(mlp.fc1))
        if mlp.fc2.weight.dtype == torch.int8:
            fc2_ws_list.append(_siglip_scale_to_rpu(mlp.fc2))

        ln1_w_list.append(layer.layer_norm1.weight.to(dtype=torch.float16, device="rpu"))
        ln1_b_list.append(layer.layer_norm1.bias.to(dtype=torch.float16, device="rpu"))
        ln2_w_list.append(layer.layer_norm2.weight.to(dtype=torch.float16, device="rpu"))
        ln2_b_list.append(layer.layer_norm2.bias.to(dtype=torch.float16, device="rpu"))

        q_b_list.append(sa.q_proj.bias.to(dtype=torch.float16, device="rpu"))
        k_b_list.append(sa.k_proj.bias.to(dtype=torch.float16, device="rpu"))
        v_b_list.append(sa.v_proj.bias.to(dtype=torch.float16, device="rpu"))
        o_b_list.append(sa.out_proj.bias.to(dtype=torch.float16, device="rpu"))

        fc1_b_list.append(mlp.fc1.bias.to(dtype=torch.float16, device="rpu"))
        fc2_b_list.append(mlp.fc2.bias.to(dtype=torch.float16, device="rpu"))

    # Global tensors
    post_ln_w = vision_model.post_layernorm.weight.to(dtype=torch.float16, device="rpu")
    post_ln_b = vision_model.post_layernorm.bias.to(dtype=torch.float16, device="rpu")
    proj_w = _proj_linear.weight.to(dtype=torch.float16, device="rpu")  # already swizzled
    proj_b = (_proj_linear.bias.to(dtype=torch.float16, device="rpu")
              if _proj_linear.bias is not None
              else torch.zeros(projection_dim, dtype=torch.float16, device="rpu"))

    # ------------------------------------------------------------------ #
    # Step 4: prepare set_weights arguments
    # ------------------------------------------------------------------ #
    padded_head_dim = q_w_list[0].shape[0] // num_heads  # 80 after pad
    padded_intermediate = fc1_w_list[0].shape[0]  # 4352 after pad

    # Build ordered positional args tuple (matches C++ schema order, 26 args after handle)
    weight_args_tuple = (
        q_w_list, k_w_list, v_w_list, o_w_list,
        fc1_w_list, fc2_w_list,
        ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list,
        q_b_list, k_b_list, v_b_list, o_b_list,
        fc1_b_list, fc2_b_list,
        post_ln_w, post_ln_b, proj_w, proj_b,
        num_heads, padded_head_dim, hidden_size, padded_intermediate, projection_dim, eps,
    )

    # ------------------------------------------------------------------ #
    # Step 5: patch embedding setup (D-507, MANDATORY)
    # ------------------------------------------------------------------ #
    embeddings = vision_model.embeddings
    patch_projection = embeddings.patch_embedding  # nn.Conv2d

    # If patch_siglip_embeddings_for_rpu already prepared the fused weight, reuse
    if hasattr(embeddings, '_rpu_gemm_weight'):
        gemm_weight = embeddings._rpu_gemm_weight
        pos_emb = embeddings._rpu_pos_emb_fused
    else:
        # Prepare it ourselves (same logic as patch_siglip_embeddings_for_rpu).
        # v5-05 D-03: helper is now module-private (defined above).
        patch_siglip_embeddings_for_rpu(embeddings)
        gemm_weight = embeddings._rpu_gemm_weight
        pos_emb = embeddings._rpu_pos_emb_fused

    kernel_size = patch_projection.kernel_size[0]  # 14
    stride = patch_projection.stride[0]  # 14
    # ------------------------------------------------------------------ #
    # Step 6: prepare replacement forward and eager runtime state
    # ------------------------------------------------------------------ #
    # Capture config in closure
    _seq_len = 256  # SigLIP per-image fixed seq_len
    # SigLIP-batch: num_images>1 packs N camera images into one seq=N*256 fused
    # encoder forward; num_images=1 is the legacy single-image path.
    _num_images = num_images
    _total_seq = _seq_len * _num_images
    _num_layers = num_layers
    _num_heads = num_heads
    _padded_head_dim = padded_head_dim
    _attn_tp = min(num_cores, num_heads)
    _siglip_sig_w8a16 = int(siglip_w8a16)
    _split_siglip_graph = _siglip_graph_enabled()

    def rpu_siglip_forward(self, pixel_values=None, **kwargs):
        h = self._rpu_vision_handle

        # D-502: RPUCache 已经在 patch 函数末尾 eager init (见 line ~492 下方),
        # forward 内只 reset_to_position(0)。**lazy init 会破坏 Dynamo cache**:
        # `hasattr(self, '_rpu_cache')` 第一次 trace 时 False,被烘成 guard;
        # 后续 call 时变 True → guard fail → recompile (P7.1g 已知问题)。
        cache = self._rpu_cache
        cache.reset_to_position(0)  # SigLIP re-inserts all tokens every forward

        # Input preparation
        if pixel_values is None:
            raise ValueError("pixel_values is required for SigLIP forward")

        input_tensor = pixel_values

        if input_tensor.device.type != 'rpu':
            input_tensor = input_tensor.to(dtype=torch.float16, device='rpu')
        elif input_tensor.dtype != torch.float16:
            input_tensor = input_tensor.to(dtype=torch.float16)

        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        # SigLIP-batch: pixel_values is [N,3,H,W] with N packed camera images.
        # The C++ forward derives image_batch_count from input.size(0): N>1 →
        # per-image minibatch SDPA, N==1 → legacy unified SDPA. The KV cache is
        # sized for up to _num_images; the GraphSignature is keyed on the ACTUAL
        # N so the batched (N=_num_images) and any single-image fallback (N=1 —
        # training / RPU_PI05_EMBED_PREFIX_PATCH=0 → upstream per-image
        # embed_prefix) build DISTINCT cached graphs rather than colliding.
        # N never exceeds _num_images in practice (patched embed_prefix stacks
        # exactly len(images)==_num_images); guard the cache-capacity bound.
        _n_packed = input_tensor.shape[0]
        if _n_packed > _num_images:
            raise ValueError(
                f"SigLIP forward got {_n_packed} packed images but the KV cache "
                f"is sized for at most {_num_images}")
        _this_total_seq = _n_packed * _seq_len

        # Composite owners may publish a component-scoped fixed-ABI contract.
        # Standalone SigLIP instances have no such attribute and retain their
        # existing signature unchanged.
        _component_plan = _siglip_component_execution_plan(
            self, h, _this_total_seq, _n_packed,
            external_patch_prologue=input_tensor.dim() == 4,
            graph_cache=self._rpu_siglip_graph_cache,
        )
        _component_plan_words = (
            () if _component_plan is None else _component_plan.graph_key_words()
        )
        _planned_stage_descriptor = list(
            _component_plan.selected.stage_tuple.physical_descriptor
        )

        # MR-B / T2 — see check_pre_embedded_rows near the branch constants for
        # why this is a hard refusal and not a generalised cache key.
        check_pre_embedded_rows(input_tensor.shape, _seq_len)

        # Call C++ forward (handles 4D->patch_emb and 3D->direct via D-507)
        k_caches = [cache.k_caches[i] for i in range(_num_layers)]
        v_caches = [cache.v_caches[i] for i in range(_num_layers)]

        # SigLIP-batch: a [N,3,H,W] input is patch-embedded + PACKED into
        # [1,N*256,1152] entirely in C++ (run_packed_patch_embed) inside this one
        # encoder capture — no Python torch.cat (which on RPU goes through the
        # CPU fallback and leaves the result un-flushed → the downstream mutable
        # DMA reads stale DDR). The C++ patch_emb flushes each input image slice
        # (fused_patch_embedding step ①); the packed hidden stays in the graph's
        # coherent DDR. N==1 is the same path with one image (byte-identical).
        encoder_input = input_tensor

        # Wrap the fused C++ op in GraphCache.capture(sig). SigLIP has no position
        # concept (bidirectional + fixed per-image seq_len=256 + position=0
        # always), and the encoder uses MASK_NONE → no
        # mask DMA risk. SigLIP-batch: the N images now ride ONE forward
        # (1 BUILD, then cross-forward all REPLAY) instead of 3 calls.
        # R-2 (1-core patch_emb + 8-core encoder mix) is handled by the
        # graph runtime auto-segmenting by num_cores.
        # branch_key: the 4D and 3D inputs record different node lists on the
        # SAME handle (D-507 auto-detection) and every other term above is
        # identical for [1,3,224,224] vs [1,256,1152]. See the constants near
        # the top of this module.
        _sig = rpu_backend.graph.GraphSignature(
            op_id="pi05_siglip_compute",
            shapes=[_this_total_seq, _num_layers, _padded_head_dim],
            dyn_dims=[
                _num_heads, _attn_tp, _n_packed, _siglip_sig_w8a16,
                *_component_plan_words,
            ],
            dtypes=[torch.float16],
            branch_key=(_SIGLIP_BRANCH_PATCH_EMBED if input_tensor.dim() == 4
                        else _SIGLIP_BRANCH_PRE_EMBEDDED),
        )
        if _split_siglip_graph:
            # Route B restore (2026-06-04): wrap the COMBINED siglip_forward
            # (patch-embed + encoder, handles 4D internally) in ONE capture like
            # the fused encoder does, so patch-embed rides graph mode
            # instead of the immediate/PASSTHROUGH shared queue (which trips the
            # enqueu-after-build_batch WARN). main split the 4D path citing a
            # RECORDING-NaN; verified the merged C++ no longer NaNs (siglip +
            # pi05 multi both 0 WARN, finite/MSE OK). num_cores mix (1-core
            # patch-emb + 8-core encoder) is auto-segmented by the graph runtime.
            with self._rpu_siglip_graph_cache.capture(_sig):
                output = torch.ops.rpu.siglip_forward(
                    h, encoder_input, k_caches, v_caches,
                    _planned_stage_descriptor)
            # FMB fast replay skips the layer body that allocates the native
            # projector output. Copy only after capture executes its kernels:
            # callers may retain several camera outputs before concatenating.
            output = output.clone()
        else:
            output = torch.ops.rpu.siglip_forward(
                h, encoder_input, k_caches, v_caches,
                _planned_stage_descriptor)

        # Release temporary SPM (subsystem boundary).
        # R-4: MUST stay OUTSIDE the capture scope — graph-aware
        # spm_alloc_reset_temporary marks the graph non-replayable
        # (src/graph/graph_spm_reset_guard.cpp + docs/graph_rules.md §12.4).
        torch.ops.rpu.spm_alloc_reset_temporary()

        # Return in HuggingFace format
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        return BaseModelOutputWithPooling(
            last_hidden_state=output,
            pooler_output=None,
        )

    # P7.1g D-502: 在 patch 阶段 eager init _rpu_cache,**不能 lazy init 在
    # forward 内**。理由:Dynamo trace 第一次 forward 时把 `hasattr(self,
    # '_rpu_cache')` 当 guard 烘进 cache,Call 2 时 attr 已存在 guard fail
    # → 强制 recompile (frontend cache 命中失败,replay 永不命中)。
    rpu_cache = RPUCache(
        num_layers=_num_layers,
        batch_size=1,
        max_seq_len=_total_seq,  # SigLIP-batch: N*256 holds all packed images' KV
        num_kv_heads=_num_heads,
        head_dim=_padded_head_dim,
        attn_tp=_attn_tp,
    )
    # S2: per-instance GraphCache for the SigLIP forward wrap. Same Dynamo
    # guard-stability rationale as `_rpu_cache` above — eager init.
    siglip_graph_cache = (
        rpu_backend.graph.GraphCache(runtime_policy=runtime_policy)
        if runtime_policy is not None
        else (rpu_backend.graph.GraphCache(runtime_policy=
              rpu_backend.graph.GraphRuntimePolicy.from_environment(
                  execution_core_count=num_cores))
              if num_cores != 8 else rpu_backend.graph.GraphCache())
    )

    # P7.1h L2 防线 — stamp marker + declare required attrs + auto freeze.
    # `_rpu_required_attrs` enumerates persistent `_rpu_*` attrs that forward()
    # reads — if any is missing at preflight time,DynamoUnsafeLazyStateError
    # raises. forward 内 grep `self._rpu_*` 即枚举:
    #   - _rpu_cache (this RPUCache instance,line above)
    #   - _rpu_vision_handle (line 305,via rpu_siglip_forward at line 436)
    #   - _rpu_siglip_graph_cache (S2 wrap site, line ~470)
    required_attrs = (
        '_rpu_cache',
        '_rpu_vision_handle',
        '_rpu_siglip_graph_cache',
    )

    # ------------------------------------------------------------------ #
    # Step 7: configure a pending native handle, tentatively publish all
    # Python-visible state, validate it, then retire the old install as the
    # commit gate. Successful cleanup restores the exact prior instance state;
    # uncertain retirement retains both installs and poisons the process.
    # The finalizer is retained on the model so later re-patches can detach
    # the retired handle's callback.
    # ------------------------------------------------------------------ #
    resource = _InstalledNativeResource(
        vision_model, None, _siglip_destroy_handle, graphs=(siglip_graph_cache,),
        keepalive=(weight_args_tuple, q_ws_list, k_ws_list, v_ws_list, o_ws_list,
                   fc1_ws_list, fc2_ws_list, gemm_weight, pos_emb, rpu_cache),
        label="SigLIP", handle_name="_rpu_vision_handle")
    handle_finalizer = resource.finalizer
    committed = False
    try:
        resource.handle = handle = torch.ops.rpu.siglip_create(linear_acc32)
        if num_cores != 8:
            torch.ops.rpu.siglip_set_execution_core_count(handle, num_cores)
        if siglip_w8a16:
            torch.ops.rpu.siglip_set_weights_w8a16(
                handle,
                *weight_args_tuple,
                q_ws_list, k_ws_list, v_ws_list, o_ws_list,
                fc1_ws_list, fc2_ws_list,
            )
        else:
            torch.ops.rpu.siglip_set_weights(handle, *weight_args_tuple)
        if num_cores != 8:
            actual = tuple(torch.ops.rpu.siglip_get_execution_topology(handle))
            expected = (1, 4, 4, 4, 1, 8, 4304, 4352)
            if actual != expected:
                raise RuntimeError(f"SigLIP native topology mismatch: {actual} != {expected}")
        # Bind the installed profile to this cache, before any native planning.
        torch.ops.rpu.siglip_set_chunk_envelope(handle, int(_total_seq), 0)
        torch.ops.rpu.siglip_model_set_patch_emb(
            handle, gemm_weight, pos_emb, kernel_size, stride)

        # v5-05 R1 HIGH-2: test-only state for Section 4 same-handle
        # invalidation test. Renamed from `_rpu_siglip_weight_args_tuple` to
        # drop the `_rpu_` prefix (A9 audits only `_rpu_*` attrs).
        vision_model._test_siglip_weight_args_tuple = weight_args_tuple
        vision_model._siglip_w8a16 = _siglip_sig_w8a16
        vision_model._siglip_graph_enabled = _split_siglip_graph
        vision_model._rpu_cache = rpu_cache
        vision_model._rpu_siglip_graph_cache = siglip_graph_cache
        vision_model._rpu_lazy_init_checked = True
        vision_model._rpu_required_attrs = required_attrs
        vision_model._rpu_vision_handle = handle
        vision_model._rpu_vision_retirement_state = resource
        vision_model._rpu_vision_handle_finalizer = handle_finalizer
        vision_model.forward = types.MethodType(
            rpu_siglip_forward, vision_model)

        from rpu_backend.graph.lazy_init_guard import _verify_lazy_init
        _verify_lazy_init(vision_model)

        if old_resource is not None:
            old_resource.retire()
        committed = True
    except BaseException as error:
        if not committed:
            cleanup_ok = resource.cleanup_failure(error, vision_model, install_snapshot)
            if resource.handle is None and handle_finalizer.alive:
                handle_finalizer.detach()
            # Bypass a custom __setattr__ that may itself have interrupted
            # tentative publication.
            model_state = vars(vision_model)
            if cleanup_ok:
                for name in _SIGLIP_RUNTIME_INSTALL_ATTRS:
                    model_state.pop(name, None)
                model_state.update(install_snapshot)
            else:
                model_state.update(
                    _rpu_vision_handle=resource.handle,
                    _rpu_vision_handle_finalizer=handle_finalizer,
                    _rpu_vision_retirement_state=resource,
                    _rpu_siglip_graph_cache=siglip_graph_cache, _rpu_cache=rpu_cache)
        raise

    if old_finalizer is not None and getattr(old_finalizer, "alive", False):
        old_finalizer.detach()

    _LOG.info("Patched SigLIPModel (instance) with all-layers-once fused forward, "
              "handle=%d, num_layers=%d, w8a16=%s",
              handle, num_layers, siglip_w8a16)

    return handle


def _shared_siglip_image_slab(images_list):
    """Return the marked camera slab only when every dim-0 view is intact."""
    if not images_list:
        return None
    shared_slab = getattr(images_list[0], "_pi05_batch_slab", None)
    shared_slab_ok = (
        isinstance(shared_slab, torch.Tensor)
        and shared_slab.ndim == 4
        and shared_slab.shape[0] == len(images_list)
        and all(
            getattr(image, "_pi05_batch_slab", None) is shared_slab
            and getattr(image, "_pi05_batch_index", None) == index
            and image.shape == shared_slab[index:index + 1].shape
            and image.stride() == shared_slab[index:index + 1].stride()
            and image.storage_offset()
            == shared_slab[index:index + 1].storage_offset()
            and image.data_ptr() == shared_slab[index:index + 1].data_ptr()
            for index, image in enumerate(images_list)
        )
    )
    return shared_slab if shared_slab_ok else None


def _prepare_siglip_images(images_list):
    """Convert camera inputs to fp16 RPU tensors, coalescing a marked slab."""
    shared_slab = _shared_siglip_image_slab(images_list)
    if shared_slab is not None:
        packed = shared_slab.to(
            dtype=torch.float16, device="rpu"
        ).contiguous()
        return [
            packed.narrow(0, index, 1)
            for index in range(len(images_list))
        ]
    return [
        image.to(dtype=torch.float16, device="rpu").contiguous()
        for image in images_list
    ]


# patch-reason: (a) SigLIP-batch all-RPU forward — bypass torch.cat + embed_image
def _siglip_forward_images(vision_model, images_list):
    """Run the RPU SigLIP encoder over a LIST of N camera images in ONE fused
    forward, returning the packed projector output [1, N*256, projection_dim].

    All-RPU variant of the patched `rpu_siglip_forward` (the bound method): it
    calls `torch.ops.rpu.siglip_forward_multi`, which patch-embeds + packs the N
    images in C++ (run_packed_patch_embed_list). So the Pi0.5 batched
    `embed_prefix` no longer needs `torch.cat(images)` (a CPU fallback that
    leaves its result un-flushed to DDR → the downstream mutable patch-emb DMA
    reads stale bytes), nor the C++ a per-image slice of a re-cat'd tensor.
    Bit-exact vs the `embed_image(torch.cat(images))` path it replaces
    (test_siglip_batch_3image.py + the E2E A/B in the resume doc).

    `vision_model` is the patched SigLIP inner vit (carries `_rpu_vision_handle`,
    `_rpu_cache`, `_rpu_siglip_graph_cache`). `images_list` is the per-camera
    pixel_values list (each [1,3,H,W]); the returned tensor matches what
    `embed_image` returns for the same images (pooler_output == last_hidden_state,
    already fp16 — so `embed_image`'s trailing `.half()` was a no-op).
    """
    h = vision_model._rpu_vision_handle
    cache = vision_model._rpu_cache
    cache.reset_to_position(0)  # SigLIP re-inserts all tokens every forward

    # Per-image fixed seq_len (So400m: 224px / 14 patch → 16×16 = 256). Same
    # constant as the single-image rpu_siglip_forward.
    _seq_len = 256
    _n_packed = len(images_list)
    _this_total_seq = _n_packed * _seq_len
    if _this_total_seq > cache.max_seq_len:
        raise ValueError(
            f"SigLIP forward_multi got {_n_packed} images ({_this_total_seq} "
            f"tokens) but the KV cache holds at most {cache.max_seq_len}")

    # GraphSignature fields recovered from the eager-init cache (SigLIP is MHA:
    # num_kv_heads == num_q_heads == num_heads). op_id stays "pi05_siglip_compute"
    # — same trace block as the single-image path. (The old note here claimed
    # "_n_packed in dyn_dims differs anyway"; it does NOT for N=1. Sharing an
    # entry with the N=1 4D path is fine because the recordings are identical —
    # see the branch_key comment below — but it is not what _n_packed buys.)
    _num_layers = cache.num_layers
    _num_heads = cache.num_kv_heads
    _padded_head_dim = cache.head_dim
    _attn_tp = cache.attn_tp
    _siglip_sig_w8a16 = int(getattr(vision_model, "_siglip_w8a16", 0))

    component_plan = _siglip_component_execution_plan(
        vision_model, h, _this_total_seq, _n_packed,
        external_patch_prologue=True,
        graph_cache=vision_model._rpu_siglip_graph_cache,
    )
    component_plan_words = (
        () if component_plan is None else component_plan.graph_key_words()
    )
    planned_stage_descriptor = list(
        component_plan.selected.stage_tuple.physical_descriptor
    )

    # The Pi0.5 batch-preprocessing fast path returns dim-0 views of one CPU
    # slab. Upload that slab once, then pass contiguous views to the existing
    # mutable per-image DMAs. Dynamic/missing-camera reference inputs have no
    # marker and preserve the established one-upload-per-image path.
    imgs = _prepare_siglip_images(images_list)

    k_caches = [cache.k_caches[i] for i in range(_num_layers)]
    v_caches = [cache.v_caches[i] for i in range(_num_layers)]

    # branch_key = PATCH_EMBED: forward_multi always patch-embeds, and for N=1
    # it is node-identical to the 4D `siglip_forward` path (both funnel into
    # run_packed_patch_embed_core), so the two intentionally SHARE one entry.
    # What they must never share is the 3D pre-embedded recording.
    _sig = rpu_backend.graph.GraphSignature(
        op_id="pi05_siglip_compute",
        shapes=[_this_total_seq, _num_layers, _padded_head_dim],
        dyn_dims=[
            _num_heads, _attn_tp, _n_packed, _siglip_sig_w8a16,
            *component_plan_words,
        ],
        dtypes=[torch.float16],
        branch_key=_SIGLIP_BRANCH_PATCH_EMBED,
    )
    if bool(getattr(vision_model, "_siglip_graph_enabled", True)):
        # Route B restore (2026-06-04 experiment): wrap the COMBINED
        # siglip_forward_multi (patch-embed + encoder) in ONE capture, like the
        # fused encoder does, so the patch-embed DMA rides graph
        # mode instead of the immediate/PASSTHROUGH shared queue (which trips
        # the enqueu-after-build_batch WARN). main split it to dodge a claimed
        # 4D-in-graph NaN — testing whether the merged C++ still NaNs.
        with vision_model._rpu_siglip_graph_cache.capture(_sig):
            out = torch.ops.rpu.siglip_forward_multi(
                h, imgs, k_caches, v_caches, planned_stage_descriptor)
    else:
        out = torch.ops.rpu.siglip_forward_multi(
            h, imgs, k_caches, v_caches, planned_stage_descriptor)

    # Subsystem boundary — MUST stay OUTSIDE the capture scope (graph-aware
    # spm_alloc_reset_temporary marks the graph non-replayable). Mirrors
    # rpu_siglip_forward.
    torch.ops.rpu.spm_alloc_reset_temporary()

    return out  # [1, N*256, projection_dim], fp16
