"""SigLIP all-layers-once instance patch.

Pi05Adapter uses this direct-handle path, and other ViT adapters can reuse
the same flat helper module. Weight preparation supports both Pi0.5's
``projector.linear`` wrapper and plain ``nn.Linear`` callers.

The direct-handle path bypasses ``embeddings.forward`` and routes 4-D pixel
inputs through ``siglip_forward`` after configuring patch-embedding weights
with ``siglip_model_set_patch_emb``.
"""
from __future__ import annotations
import os
import types
import weakref

import torch
import torch.nn as nn

import rpu_backend  # GraphCache / GraphSignature
from rpu_backend.quant._common import quantize_linear_per_channel
from rpu_backend.runtime import rpu_env_bool
from rpu_backend.runtime.log import _LOG
from rpu_backend.runtime.weights import (
    transform_linear_weight, convert_linear_weights_inplace,
    NUM_CORES,
)
from rpu_backend.api.cache import RPUCache


def _siglip_graph_enabled() -> bool:
    return rpu_env_bool("RPU_PI05_SIGLIP_GRAPH", default=True)


# siglip_forward records structurally different graphs for 4D pixel input
# (patch embedding plus encoder) and 3D pre-embedded input (encoder only).
# Their canonical shape terms overlap, so branch_key is part of the cache key.
_SIGLIP_BRANCH_PATCH_EMBED = 1   # 4D pixels → patch-embed prologue + encoder
_SIGLIP_BRANCH_PRE_EMBEDDED = 2  # 3D hidden → encoder only


# The 3D branch is fixed to SIGLIP_SEQ_LEN because its signature uses the packed
# image count. Other sequence lengths require a handle keyed by the actual rows.
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
        f"another length, use a native handle whose GraphSignature keys the "
        f"actual row count under a distinct op_id and GraphCache.")


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
) -> None:
    if w8a16:
        weight, scale = _quantize_and_swizzle_siglip_weight(
            weight, partition=partition)
        module.weight = nn.Parameter(weight, requires_grad=False)
        _set_siglip_weight_scale(module, scale)
    else:
        weight = transform_linear_weight(
            weight.contiguous(), partition=partition, num_cores=NUM_CORES)
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


def _siglip_destroy_handle(h):
    """Release C++ SigLIPModel handle. Called by weakref.finalize on GC."""
    try:
        torch.ops.rpu.siglip_destroy(h)
    except Exception:
        # Swallow -- during interpreter shutdown the op may be gone
        pass


# =============================================================================
# SigLIP weight-preparation helpers.
# =============================================================================
# Idempotence guards short-circuit a second conversion when callers prepare
# weights before entering `patch_siglip_model_for_rpu_all_layers_once`.


def _convert_siglip_weights_for_rpu(vision_tower, *, w8a16_default: bool = False):
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
    siglip_w8a16_names = _siglip_w8a16_projection_names(default_enabled=w8a16_default)
    siglip_w8a16 = bool(siglip_w8a16_names)

    # Idempotence guard on the NORMALIZED vit object
    if getattr(vit, '_rpu_siglip_weights_converted', False):
        if siglip_w8a16 and not getattr(vit, '_siglip_w8a16', False):
            _LOG.warning(
                "SigLIP W8A16 requested after encoder was already converted as fp16; "
                "reload the model to apply RPU_PI05_SIGLIP_W8A16=1")
        return

    encoder = vit.encoder

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
                    w8a16=(f"self_attn.{proj_name}" in siglip_w8a16_names))
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
                w8a16=("self_attn.out_proj" in siglip_w8a16_names))
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
                w8a16=("mlp.fc1" in siglip_w8a16_names))
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
                w8a16=("mlp.fc2" in siglip_w8a16_names))
            mlp.fc2.in_features = padded_intermediate
            # fc2 bias: [hidden_size] -- no padding needed

        # Store metadata for fused forward
        layer._rpu_layer_idx = layer_idx
        layer._rpu_num_layers = num_layers
        layer._rpu_num_heads = num_heads
        layer._rpu_head_dim = padded_head_dim
        layer._rpu_intermediate_size = padded_intermediate

    vit._rpu_siglip_weights_converted = True
    vit._siglip_w8a16 = siglip_w8a16
    suffix = " (W8A16)" if siglip_w8a16 else ""
    _LOG.info("Converted %d SigLIP encoder layers%s", num_layers, suffix)


def _convert_siglip_projector_for_rpu(projector):
    """Idempotent projector weight conversion for RPU col-partition.
    Handles both Pi0.5 wrapper (has .linear) and plain nn.Linear.
    Guards on BOTH wrapper and leaf to prevent double-swizzle from mixed calls."""
    # Normalize to leaf for guard check -- if either has the marker, skip
    _leaf = projector.linear if hasattr(projector, 'linear') else projector
    if getattr(projector, '_rpu_weights_converted', False) or \
       getattr(_leaf, '_rpu_weights_converted', False):
        return
    if hasattr(projector, 'linear'):
        # Pi0.5 wrapper -- recurse into children to find the nn.Linear
        convert_linear_weights_inplace(projector)
    else:
        # Plain nn.Linear -- transform directly (convert_linear_weights_inplace
        # on a leaf module is a no-op since it only recurses named_children).
        # Hardcode partition=1 (col-partition) -- matches C++ projector kernel.
        projector.weight.data = transform_linear_weight(
            projector.weight.data, partition=1, num_cores=8)
    # Mark BOTH wrapper and leaf to handle mixed caller patterns
    projector._rpu_weights_converted = True
    _leaf._rpu_weights_converted = True


def patch_siglip_embeddings_for_rpu(embeddings_module, num_cores=8):
    """Patch SiglipVisionEmbeddings for RPU fused patch embedding.

    Converts Conv2d(cin=3, cout=1152, k=14, s=14) into im2col + GEMM pipeline:
      - Weight: pad cin 3→16, reshape [cout, cin*kh*kw] = [1152, 3136], col-swizzle
      - Pre-fuse bias + position embedding: pos_emb_fused = pos_emb + conv.bias

    The direct-handle path consumes the resulting `_rpu_gemm_weight` and
    `_rpu_pos_emb_fused` buffers via `siglip_model_set_patch_emb` (called by
    `patch_siglip_model_for_rpu_all_layers_once`); the direct-handle C++
    pipeline routes 4-D pixel values through `siglip_forward`.
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


_SIGLIP_RUNTIME_INSTALL_ATTRS = (
    "_test_siglip_weight_args_tuple",
    "_siglip_w8a16",
    "_rpu_cache",
    "_rpu_siglip_graph_cache",
    "_rpu_lazy_init_checked",
    "_rpu_required_attrs",
    "_rpu_vision_handle",
    "_rpu_vision_handle_finalizer",
    "forward",
)


def patch_siglip_model_for_rpu_all_layers_once(vision_model, projector,
                                               num_images: int = 1,
                                               w8a16_default: bool = False) -> int:
    """
    Patch a Pi0.5 SigLIP vision model instance to use all-layers-once
    C++ execution via SigLIPModel (FusedModelBase).

    With num_images > 1, the patched forward expects
    pixel_values to be [num_images, 3, H, W] — N camera images packed into ONE
    seq=N*256 fused encoder forward (per-image attention isolation via the
    minibatch SDPA). The KV cache is sized for up to num_images (N*256 rows);
    the per-forward GraphSignature is keyed on the ACTUAL packed N so single-
    and multi-image graphs never collide. num_images = 1 is the default
    single-image path used by standalone SigLIP and other ViT ports.

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
    model_state = vars(vision_model)
    install_snapshot = {
        name: model_state[name]
        for name in _SIGLIP_RUNTIME_INSTALL_ATTRS
        if name in model_state
    }
    had_old_handle = "_rpu_vision_handle" in install_snapshot
    old_handle = install_snapshot.get("_rpu_vision_handle")
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
    # Pi0.5 projector may be a wrapper with a .linear attribute.
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
    # Encoder conversion
    _convert_siglip_weights_for_rpu(vision_model, w8a16_default=w8a16_default)

    # Projector conversion
    _convert_siglip_projector_for_rpu(projector)
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
    # Step 5: mandatory patch-embedding setup
    # ------------------------------------------------------------------ #
    embeddings = vision_model.embeddings
    patch_projection = embeddings.patch_embedding  # nn.Conv2d

    # If patch_siglip_embeddings_for_rpu already prepared the fused weight, reuse
    if hasattr(embeddings, '_rpu_gemm_weight'):
        gemm_weight = embeddings._rpu_gemm_weight
        pos_emb = embeddings._rpu_pos_emb_fused
    else:
        # Prepare it ourselves using the same helper.
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
    _attn_tp = min(8, num_heads)
    _siglip_sig_w8a16 = int(siglip_w8a16)

    def rpu_siglip_forward(self, pixel_values=None, **kwargs):
        h = self._rpu_vision_handle

        # RPUCache is eagerly initialized below; forward only resets it to
        # position zero. Lazy initialization would break the Dynamo cache:
        # `hasattr(self, '_rpu_cache')` 第一次 trace 时 False,被烘成 guard;
        # later calls see True, fail the guard, and recompile.
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

        # See check_pre_embedded_rows near the branch constants for
        # why this is a hard refusal and not a generalised cache key.
        check_pre_embedded_rows(input_tensor.shape, _seq_len)

        # Call C++ forward (4-D patch embedding or 3-D pre-embedded input).
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
        # Mixed core counts are handled by graph-runtime auto-segmentation.
        # branch_key: the 4D and 3D inputs record different node lists on the
        # same handle, and every other term above is
        # identical for [1,3,224,224] vs [1,256,1152]. See the constants near
        # the top of this module.
        _sig = rpu_backend.graph.GraphSignature(
            op_id="pi05_siglip_compute",
            shapes=[_this_total_seq, _num_layers, _padded_head_dim],
            dyn_dims=[_num_heads, _attn_tp, _n_packed, _siglip_sig_w8a16],
            dtypes=[torch.float16],
            branch_key=(_SIGLIP_BRANCH_PATCH_EMBED if input_tensor.dim() == 4
                        else _SIGLIP_BRANCH_PRE_EMBEDDED),
        )
        split_siglip_graph = _siglip_graph_enabled()
        if split_siglip_graph:
            # Capture combined patch embedding and encoder execution once. The
            # graph runtime segments the path by its core selections.
            with self._rpu_siglip_graph_cache.capture(_sig):
                output = torch.ops.rpu.siglip_forward(
                    h, encoder_input, k_caches, v_caches)
        else:
            output = torch.ops.rpu.siglip_forward(
                h, encoder_input, k_caches, v_caches)

        # Release temporary SPM (subsystem boundary).
        # This must stay outside capture: graph-aware temporary-SPM reset
        # marks the graph non-replayable.
        torch.ops.rpu.spm_alloc_reset_temporary()

        # Return in HuggingFace format
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        return BaseModelOutputWithPooling(
            last_hidden_state=output,
            pooler_output=None,
        )

    # Eager-initialize `_rpu_cache` during patching rather than in forward.
    # Dynamo would otherwise record an attribute-existence guard that changes
    # after the first call and forces recompilation.
    rpu_cache = RPUCache(
        num_layers=_num_layers,
        batch_size=1,
        max_seq_len=_total_seq,  # SigLIP-batch: N*256 holds all packed images' KV
        num_kv_heads=_num_heads,
        head_dim=_padded_head_dim,
        attn_tp=_attn_tp,
    )
    # Per-instance GraphCache for the SigLIP forward wrap. Same Dynamo
    # guard-stability rationale as `_rpu_cache` above — eager init.
    siglip_graph_cache = rpu_backend.graph.GraphCache()

    # Stamp the marker, declare required attributes, and auto-freeze.
    # `_rpu_required_attrs` enumerates persistent `_rpu_*` attrs that forward()
    # reads — if any is missing at preflight time,DynamoUnsafeLazyStateError
    # raises. Forward reads the following persistent attributes:
    #   - _rpu_cache
    #   - _rpu_vision_handle
    #   - _rpu_siglip_graph_cache
    required_attrs = (
        '_rpu_cache',
        '_rpu_vision_handle',
        '_rpu_siglip_graph_cache',
    )

    # ------------------------------------------------------------------ #
    # Step 7: configure a pending native handle, tentatively publish all
    # Python-visible state, validate it, then retire the old install as the
    # commit gate. Until that gate succeeds, any BaseException restores the
    # exact prior instance state and immediately destroys the pending handle.
    # The finalizer is retained on the model so later re-patches can detach
    # the retired handle's callback.
    # ------------------------------------------------------------------ #
    handle = torch.ops.rpu.siglip_create()
    handle_finalizer = None
    committed = False
    try:
        handle_finalizer = weakref.finalize(
            vision_model, _siglip_destroy_handle, h=handle)
        if siglip_w8a16:
            torch.ops.rpu.siglip_set_weights_w8a16(
                handle,
                *weight_args_tuple,
                q_ws_list, k_ws_list, v_ws_list, o_ws_list,
                fc1_ws_list, fc2_ws_list,
            )
        else:
            torch.ops.rpu.siglip_set_weights(handle, *weight_args_tuple)
        torch.ops.rpu.siglip_model_set_patch_emb(
            handle, gemm_weight, pos_emb, kernel_size, stride)

        # Diagnostic state for same-handle invalidation checks. The name avoids
        # the `_rpu_` prefix reserved for validated runtime attributes.
        vision_model._test_siglip_weight_args_tuple = weight_args_tuple
        vision_model._siglip_w8a16 = _siglip_sig_w8a16
        vision_model._rpu_cache = rpu_cache
        vision_model._rpu_siglip_graph_cache = siglip_graph_cache
        vision_model._rpu_lazy_init_checked = True
        vision_model._rpu_required_attrs = required_attrs
        vision_model._rpu_vision_handle = handle
        vision_model._rpu_vision_handle_finalizer = handle_finalizer
        vision_model.forward = types.MethodType(
            rpu_siglip_forward, vision_model)

        from rpu_backend.graph.lazy_init_guard import _verify_lazy_init
        _verify_lazy_init(vision_model)

        if (
            had_old_handle
            and (
                old_finalizer is None
                or getattr(old_finalizer, "alive", False)
            )
        ):
            torch.ops.rpu.siglip_destroy(old_handle)
        committed = True
    except BaseException:
        if not committed:
            # Bypass a custom __setattr__ that may itself have interrupted
            # tentative publication.
            model_state = vars(vision_model)
            for name in _SIGLIP_RUNTIME_INSTALL_ATTRS:
                model_state.pop(name, None)
            model_state.update(install_snapshot)

            if handle_finalizer is None:
                _siglip_destroy_handle(handle)
            elif handle_finalizer.alive:
                handle_finalizer()
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


def _siglip_forward_images(vision_model, images_list):
    """Run the RPU SigLIP encoder over a LIST of N camera images in ONE fused
    forward, returning the packed projector output [1, N*256, projection_dim].

    All-RPU variant of the patched `rpu_siglip_forward` (the bound method): it
    calls `torch.ops.rpu.siglip_forward_multi`, which patch-embeds + packs the N
    images in C++ (run_packed_patch_embed_list). So the Pi0.5 batched
    `embed_prefix` no longer needs `torch.cat(images)` (a CPU fallback that
    leaves its result un-flushed to DDR → the downstream mutable patch-emb DMA
    reads stale bytes), nor the C++ a per-image slice of a re-cat'd tensor.
    The output contract matches the per-image `embed_image` path.

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
    # — same trace block as the single-image path. For N=1 the recordings are
    # identical; ``branch_key`` distinguishes branches when their recordings differ.
    _num_layers = cache.num_layers
    _num_heads = cache.num_kv_heads
    _padded_head_dim = cache.head_dim
    _attn_tp = min(8, _num_heads)
    _siglip_sig_w8a16 = int(getattr(vision_model, "_siglip_w8a16", 0))

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
        dyn_dims=[_num_heads, _attn_tp, _n_packed, _siglip_sig_w8a16],
        dtypes=[torch.float16],
        branch_key=_SIGLIP_BRANCH_PATCH_EMBED,
    )
    split_siglip_graph = _siglip_graph_enabled()
    if split_siglip_graph:
        # Capture combined patch embedding and encoder execution once. The
        # graph runtime segments the path by its core selections.
        with vision_model._rpu_siglip_graph_cache.capture(_sig):
            out = torch.ops.rpu.siglip_forward_multi(h, imgs, k_caches, v_caches)
    else:
        out = torch.ops.rpu.siglip_forward_multi(h, imgs, k_caches, v_caches)

    # Subsystem boundary — MUST stay OUTSIDE the capture scope (graph-aware
    # spm_alloc_reset_temporary marks the graph non-replayable). Mirrors
    # rpu_siglip_forward.
    torch.ops.rpu.spm_alloc_reset_temporary()

    return out  # [1, N*256, projection_dim], fp16
