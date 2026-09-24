"""RhinoVLA RPU weight prep (P1).

Self-contained weight-conversion helpers for the RhinoVLA action expert. The
expert math is identical to QwenPI05's, so these helpers are ported from
adapters/qwenpi05/fused.py (config-agnostic — depth/dims read from `expert`)
and reused via the qwenpi05 C++ ops. RhinoVLA-specific:
  - `convert_expert_for_rpu` owns the required order: half() -> swizzle -> rpu.
  - `_prefill_kv_cache` supports arbitrary prefix_len (no %8 restriction) and
    handles prefix_len == 0.
"""

from hashlib import sha256

import torch


_EXPERT_PROJECTIONS = (
    ("self_attn", "q_proj"), ("self_attn", "k_proj"),
    ("self_attn", "v_proj"), ("self_attn", "o_proj"),
    ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
)


def _quantize_expert_projections(expert):
    """Quantize the seven main projections in natural CPU layout, once.

    Stage every result before changing the model, so malformed or nonfinite
    weights cannot leave a partially quantized expert. AdaRMS and IO stay FP16.
    """
    from rpu_backend.quant._common import quantize_linear_per_channel

    prepared = []
    for index, layer in enumerate(expert.layers):
        for parent, name in _EXPERT_PROJECTIONS:
            linear = getattr(getattr(layer, parent), name)
            weight = linear.weight
            if (weight.device.type != "cpu" or weight.dtype != torch.float16
                    or weight.ndim != 2 or hasattr(linear, "weight_scale")
                    or linear.bias is not None):
                raise ValueError(
                    f"RhinoVLA W8A16 requires fresh bias-free FP16 CPU weights: "
                    f"layer {index}.{parent}.{name}")
            if not torch.isfinite(weight).all():
                raise ValueError(f"RhinoVLA W8A16 nonfinite weight: layer {index}.{name}")
            quantized, scale = quantize_linear_per_channel(weight)
            if not (torch.isfinite(scale).all() and (scale > 0).all()):
                raise ValueError(f"RhinoVLA W8A16 invalid scale: layer {index}.{name}")
            prepared.append((linear, quantized.contiguous(), scale.contiguous()))
    for linear, quantized, scale in prepared:
        linear.weight = torch.nn.Parameter(quantized, requires_grad=False)
        linear.register_buffer("weight_scale", scale)


def _gather_expert_scales(expert):
    """Admit uniform FP16 or W8A16 storage and return natural [N] scales."""
    groups = [
        [getattr(getattr(layer, parent), name) for layer in expert.layers]
        for parent, name in _EXPERT_PROJECTIONS
    ]
    modules = [linear for group in groups for linear in group]
    if all(linear.weight.dtype == torch.float16
           and not hasattr(linear, "weight_scale") for linear in modules):
        return ()
    for linear in modules:
        scale = getattr(linear, "weight_scale", None)
        if (linear.weight.dtype != torch.int8 or scale is None
                or scale.dtype != torch.float16 or scale.ndim != 1
                or scale.numel() != linear.weight.shape[0]
                or scale.device != linear.weight.device or not scale.is_contiguous()):
            raise ValueError("RhinoVLA requires uniform W8A16 weights with FP16 [N] scales")
    # Native admission also checks actual RPU storage, finite positive scales,
    # exact v3 geometry and all retained FP16 weights before committing owners.
    return tuple([linear.weight_scale for linear in group] for group in groups)


def convert_expert_for_rpu(expert, *, w8a16=False):
    """Move a RhinoVLA action expert to RPU in the CORRECT order.

    convert_linear_weights_inplace swizzles by weight.element_size();
    swizzling fp32 then casting to fp16 produces a wrong-dwidth byte layout.
    Order MUST be: half() -> optional expert-only W8 quantization -> swizzle
    (skip cond) -> to('rpu'). Scales stay in natural output-channel order.
    """
    from rpu_backend.runtime.weights import convert_linear_weights_inplace
    # convert_linear_weights_inplace mutates weights irreversibly (row/col swizzle);
    # a second call double-swizzles → right L2 norm, wrong direction (Pitfall P1).
    # Guard like qwenpi05.convert: refuse a repeat, and stamp _started before the
    # mutation so a failure mid-swizzle can't be re-run silently.
    if getattr(expert, "_rpu_swizzled", False):
        raise RuntimeError(
            "convert_expert_for_rpu: expert already has _rpu_swizzled=True from a "
            "prior call. Re-swizzling corrupts the weights (Pitfall P1). Rebuild the "
            "expert from the checkpoint instead of converting twice."
        )
    if getattr(expert, "_rpu_swizzle_started", False):
        raise RuntimeError(
            "convert_expert_for_rpu: a prior conversion started but did not finish "
            "(_rpu_swizzle_started=True). Weights may be partially swizzled — rebuild "
            "the expert from the checkpoint."
        )
    expert = expert.half()
    if w8a16:
        _quantize_expert_projections(expert)
    expert._rpu_swizzle_started = True
    convert_linear_weights_inplace(expert, skip_names={"cond"})
    expert = expert.to("rpu")
    expert._rpu_swizzled = True
    return expert


def _fold_adarms_cond_dense(norm, layer_idx, src_norm_name):
    """Fold AdaRMS gamma into one cond projection in CPU fp32."""
    if not hasattr(norm, 'cond') or norm.cond is None:
        raise RuntimeError(
            f"rhinovla.convert: layer {layer_idx}.{src_norm_name} has no "
            f".cond -- is this an AdaRMS norm? (Expected AdaRMSNorm with cond Linear.)"
        )
    if norm.cond.bias is None:
        raise RuntimeError(
            f"rhinovla.convert: layer {layer_idx}.{src_norm_name}.cond has "
            f"no bias -- AdaRMS cond must include a bias vector."
        )
    if not hasattr(norm, 'norm') or not hasattr(norm.norm, 'weight') or norm.norm.weight is None:
        raise RuntimeError(
            f"rhinovla.convert: layer {layer_idx}.{src_norm_name}.norm has "
            f"no learnable weight (gamma). Expected Qwen3VLTextRMSNorm."
        )

    gamma = norm.norm.weight.detach().cpu().to(dtype=torch.float32)  # [H]
    W_f32 = norm.cond.weight.detach().cpu().to(dtype=torch.float32)  # [3H, H]
    b_f32 = norm.cond.bias.detach().cpu().to(dtype=torch.float32)    # [3H]

    H = gamma.shape[0]
    if W_f32.shape[0] != 3 * H or W_f32.shape[1] != H:
        raise RuntimeError(
            f"rhinovla.convert: layer {layer_idx}.{src_norm_name}.cond.weight "
            f"has shape {tuple(W_f32.shape)}, expected [{3*H}, {H}]."
        )
    if b_f32.shape[0] != 3 * H:
        raise RuntimeError(
            f"rhinovla.convert: layer {layer_idx}.{src_norm_name}.cond.bias "
            f"has shape {tuple(b_f32.shape)}, expected [{3*H}]."
        )

    # CPU AdaRMSNorm:
    #   out = (x/rms(x)) * gamma * (1 + scale) + shift
    # C++ fused kernel:
    #   out = (x/rms(x)) * scale_fused + shift
    # Fold gamma into the scale block only. The kernel adds +1 to block 0.
    W_s, W_sh, W_g = W_f32[0:H], W_f32[H:2*H], W_f32[2*H:3*H]
    b_s, b_sh, b_g = b_f32[0:H], b_f32[H:2*H], b_f32[2*H:3*H]

    W_folded = torch.cat([gamma[:, None] * W_s, W_sh, W_g], dim=0).contiguous()
    b_folded = torch.cat([gamma * b_s + (gamma - 1.0), b_sh, b_g],
                         dim=0).contiguous()
    return W_folded, b_folded


def _quantize_full_w8_weight(weight):
    from rpu_backend.quant._common import quantize_linear_per_channel

    weight = weight.detach().cpu().to(torch.float16).contiguous()
    if weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError("RhinoVLA full W8 requires finite 2D projection weights")
    quantized, scale = quantize_linear_per_channel(weight)
    if not (torch.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError("RhinoVLA full W8 requires finite positive FP16 scales")
    return quantized.contiguous(), scale.contiguous()


def _prepare_full_w8_cond_weights(expert):
    """Retain row weights for the owner and col weights for cold RPU tables."""
    from rpu_backend.runtime.weights import transform_linear_weight

    cached = getattr(expert, "_rpu_full_w8_cond_owners", None)
    if cached is not None:
        return cached["weights"]
    if len(expert.layers) != 18 or int(expert.config.width) != 1024:
        raise ValueError("RhinoVLA full W8 AdaRMS requires 18 layers/H1024")
    attn_w, attn_b, mlp_w, mlp_b, pair_w, pair_b = ([] for _ in range(6))
    attn_s, mlp_s, pair_s, cold_w = ([] for _ in range(4))
    for index, layer in enumerate(expert.layers):
        folded = [_fold_adarms_cond_dense(getattr(layer, name), index, name)
                  for name in ("input_layernorm", "post_attention_layernorm")]
        quantized, scale = _quantize_full_w8_weight(torch.cat([x[0] for x in folded]))
        row = transform_linear_weight(quantized, partition=0, num_cores=8).to("rpu")
        col = transform_linear_weight(quantized, partition=1, num_cores=8).to("rpu")
        scale = scale.to("rpu")
        bias = torch.cat([x[1] for x in folded]).to(dtype=torch.float16, device="rpu")
        # Row swizzle's leading dimension is N/16, so these views are exactly
        # the independent attn/MLP layouts and share the paired allocation.
        attn_w.append(row[:3072]); mlp_w.append(row[3072:]); pair_w.append(row)
        attn_b.append(bias[:3072]); mlp_b.append(bias[3072:]); pair_b.append(bias)
        attn_s.append(scale[:3072]); mlp_s.append(scale[3072:]); pair_s.append(scale)
        cold_w.append(col)
    owners = {
        "weights": (attn_w, attn_b, mlp_w, mlp_b, pair_w, pair_b),
        "scales": (attn_s, mlp_s, pair_s),
        "cold_weights": cold_w, "cold_scales": pair_s, "cold_biases": pair_b,
    }
    expert._rpu_full_w8_cond_owners = owners
    return owners["weights"]


def _prepare_cond_dense_weights(expert, *, full_w8a16=False):
    """Swizzle per-layer AdaRMS cond weights for RPU row-partition GEMV.

    For each layer x each of {input_layernorm, post_attention_layernorm}:
      norm.cond.weight [3*width, width] (γ-folded on scale block)
      -> transform_linear_weight(partition=0, num_cores=8)   # row partition (K split)
      -> .to('rpu')

    Returns six parallel lists (one entry per layer):
      attn_dense_w_list, attn_dense_b_list, mlp_dense_w_list, mlp_dense_b_list,
      pair_dense_w_list, pair_dense_b_list
    """
    if full_w8a16:
        return _prepare_full_w8_cond_weights(expert)
    from rpu_backend.runtime.weights import transform_linear_weight, NUM_CORES

    attn_w, attn_b, mlp_w, mlp_b, pair_w, pair_b = [], [], [], [], [], []
    gamma_fold_sites = 0

    for layer_idx, layer in enumerate(expert.layers):
        layer_pair_w = []
        layer_pair_b = []
        for src_norm_name, (w_out, b_out) in (
            ('input_layernorm', (attn_w, attn_b)),
            ('post_attention_layernorm', (mlp_w, mlp_b)),
        ):
            norm = getattr(layer, src_norm_name)
            W_folded, b_folded = _fold_adarms_cond_dense(norm, layer_idx, src_norm_name)
            layer_pair_w.append(W_folded)
            layer_pair_b.append(b_folded)

            # Idempotent re-patch: reuse cached row-partition buffers if present.
            if hasattr(norm, '_rpu_cond_w_rp') and hasattr(norm, '_rpu_cond_b_rp'):
                w_out.append(norm._rpu_cond_w_rp)
                b_out.append(norm._rpu_cond_b_rp)
                continue

            # Cast to fp16 and apply row-partition swizzle (K split across 8 cores).
            folded_w_fp16 = W_folded.to(dtype=torch.float16).contiguous()  # [3H, H]
            transformed_w = transform_linear_weight(
                folded_w_fp16, partition=0, num_cores=NUM_CORES
            ).to('rpu').contiguous()
            folded_b_rpu = b_folded.to(dtype=torch.float16).to('rpu').contiguous()

            norm.register_buffer('_rpu_cond_w_rp', transformed_w)
            norm.register_buffer('_rpu_cond_b_rp', folded_b_rpu)

            w_out.append(transformed_w)
            b_out.append(folded_b_rpu)
            gamma_fold_sites += 1

        pair_w_folded = torch.cat(layer_pair_w, dim=0).contiguous()  # [6H, H]
        pair_b_folded = torch.cat(layer_pair_b, dim=0).contiguous()  # [6H]
        pair_w.append(transform_linear_weight(
            pair_w_folded.to(dtype=torch.float16).contiguous(),
            partition=0, num_cores=NUM_CORES).to('rpu').contiguous())
        pair_b.append(pair_b_folded.to(dtype=torch.float16, device='rpu').contiguous())

    print(f"[Rhino] cond gamma folded into row-partition GEMV weights at {gamma_fold_sites} sites")
    return attn_w, attn_b, mlp_w, mlp_b, pair_w, pair_b


def _prepare_final_norm_dense_weights(final_norm):
    """Prepare the final AdaRMSNorm dense for the fused denoise-loop post hook.

    Mirrors ``_prepare_cond_dense_weights`` for the final expert norm, including
    the gamma fold on the scale block. The CPU fallback keeps ``final_norm`` in
    fp32; these buffers are additive RPU-only copies.
    """
    from rpu_backend.runtime.weights import transform_linear_weight, NUM_CORES

    if not hasattr(final_norm, "cond") or final_norm.cond is None:
        raise RuntimeError("RhinoVLA final norm has no AdaRMS cond Linear")
    if final_norm.cond.bias is None:
        raise RuntimeError("RhinoVLA final norm cond Linear has no bias")
    if not hasattr(final_norm, "norm") or final_norm.norm.weight is None:
        raise RuntimeError("RhinoVLA final norm has no RMS gamma weight")

    gamma = final_norm.norm.weight.detach().cpu().to(dtype=torch.float32)
    W_f32 = final_norm.cond.weight.detach().cpu().to(dtype=torch.float32)
    b_f32 = final_norm.cond.bias.detach().cpu().to(dtype=torch.float32)

    H = gamma.shape[0]
    if W_f32.shape != (3 * H, H):
        raise RuntimeError(
            f"RhinoVLA final norm cond.weight shape {tuple(W_f32.shape)}, "
            f"expected {(3 * H, H)}")
    if b_f32.shape != (3 * H,):
        raise RuntimeError(
            f"RhinoVLA final norm cond.bias shape {tuple(b_f32.shape)}, "
            f"expected {(3 * H,)}")

    W_s, W_sh, W_g = W_f32[0:H], W_f32[H:2 * H], W_f32[2 * H:3 * H]
    b_s, b_sh, b_g = b_f32[0:H], b_f32[H:2 * H], b_f32[2 * H:3 * H]
    W_folded = torch.cat([gamma[:, None] * W_s, W_sh, W_g], dim=0).contiguous()
    b_folded = torch.cat([gamma * b_s + (gamma - 1.0), b_sh, b_g],
                         dim=0).contiguous()

    w_rpu = transform_linear_weight(
        W_folded.to(dtype=torch.float16).contiguous(),
        partition=0, num_cores=NUM_CORES).to("rpu").contiguous()
    b_rpu = b_folded.to(dtype=torch.float16, device="rpu").contiguous()
    return w_rpu, b_rpu


def _fold_final_norm_dense(final_norm):
    """Return final AdaRMS cond dense folded like _prepare_final_norm_dense_weights.

    This CPU fp32 helper is used by the denoise-loop AdaRMS precompute path.
    The C++ path adds +1 to the scale block after the dense, so callers should
    apply that addition to the first H outputs of the returned projection.
    """
    if not hasattr(final_norm, "cond") or final_norm.cond is None:
        raise RuntimeError("RhinoVLA final norm has no AdaRMS cond Linear")
    if final_norm.cond.bias is None:
        raise RuntimeError("RhinoVLA final norm cond Linear has no bias")
    if not hasattr(final_norm, "norm") or final_norm.norm.weight is None:
        raise RuntimeError("RhinoVLA final norm has no RMS gamma weight")

    gamma = final_norm.norm.weight.detach().cpu().to(dtype=torch.float32)
    W_f32 = final_norm.cond.weight.detach().cpu().to(dtype=torch.float32)
    b_f32 = final_norm.cond.bias.detach().cpu().to(dtype=torch.float32)

    H = gamma.shape[0]
    if W_f32.shape != (3 * H, H):
        raise RuntimeError(
            f"RhinoVLA final norm cond.weight shape {tuple(W_f32.shape)}, "
            f"expected {(3 * H, H)}")
    if b_f32.shape != (3 * H,):
        raise RuntimeError(
            f"RhinoVLA final norm cond.bias shape {tuple(b_f32.shape)}, "
            f"expected {(3 * H,)}")

    W_s, W_sh, W_g = W_f32[0:H], W_f32[H:2 * H], W_f32[2 * H:3 * H]
    b_s, b_sh, b_g = b_f32[0:H], b_f32[H:2 * H], b_f32[2 * H:3 * H]
    W_folded = torch.cat([gamma[:, None] * W_s, W_sh, W_g], dim=0).contiguous()
    b_folded = torch.cat([gamma * b_s + (gamma - 1.0), b_sh, b_g],
                         dim=0).contiguous()
    return W_folded, b_folded


def _prepare_denoise_adarms_tables(expert, cond_all_cpu, *, full_w8a16=False,
                                   loop_weights=None, precompute_gate_tanh=False,
                                   high_precision=False):
    """Precompute denoise-loop AdaRMS dense outputs for fixed timestep conds.

    RhinoVLA's denoise schedule uses the same timestep embedding for every
    inference with the same ``steps`` value. The per-layer AdaRMS cond dense
    output therefore does not depend on image/text/state/action inputs. This
    helper materializes:

      * pair_table:  [steps, layers, 6H]  = attn + mlp AdaRMS params
      * final_table: [steps, 3H]          = final norm AdaRMS params

    Both tensors are fp16 RPU tensors consumed by a C++ mutable-DMA fast path.
    """
    if high_precision and not full_w8a16:
        raise ValueError("RhinoVLA high precision requires full W8 AdaRMS tables")
    if precompute_gate_tanh and not full_w8a16:
        raise ValueError("RhinoVLA gate TANH precompute requires full W8 AdaRMS tables")
    if full_w8a16:
        return _prepare_full_w8_adarms_tables(
            expert, cond_all_cpu, loop_weights, precompute_gate_tanh=precompute_gate_tanh,
            high_precision=high_precision)
    cond = cond_all_cpu.detach().cpu().to(dtype=torch.float32).contiguous()
    if cond.dim() != 2:
        raise RuntimeError(
            f"RhinoVLA AdaRMS precompute expects cond_all [steps,H], got {tuple(cond.shape)}")
    steps, H = int(cond.shape[0]), int(cond.shape[1])
    if steps <= 0 or H <= 0:
        raise RuntimeError("RhinoVLA AdaRMS precompute got empty cond table")

    pair_layers = []
    for layer_idx, layer in enumerate(expert.layers):
        layer_outs = []
        for src_norm_name in ("input_layernorm", "post_attention_layernorm"):
            W, b = _fold_adarms_cond_dense(
                getattr(layer, src_norm_name), layer_idx, src_norm_name)
            out = torch.nn.functional.linear(cond, W, b).contiguous()
            out[:, :H] += 1.0
            layer_outs.append(out)
        pair_layers.append(torch.cat(layer_outs, dim=1).contiguous())

    pair_table = torch.stack(pair_layers, dim=1).to(
        dtype=torch.float16, device="rpu").contiguous()

    final_w, final_b = _fold_final_norm_dense(expert.norm)
    final_table = torch.nn.functional.linear(cond, final_w, final_b).contiguous()
    final_table[:, :H] += 1.0
    final_table = final_table.to(dtype=torch.float16, device="rpu").contiguous()
    print(
        f"[Rhino] AdaRMS precomputed: pair_table={tuple(pair_table.shape)} "
        f"final_table={tuple(final_table.shape)}")
    return pair_table, final_table


def _prepare_full_w8_adarms_tables(expert, cond_all_cpu, loop_weights, *,
                                  precompute_gate_tanh=False, high_precision=False):
    """Run actual W8A16 RPU Linear once for each fixed-timestep projection."""
    cond = cond_all_cpu.detach().to(dtype=torch.float16, device="cpu").contiguous()
    if tuple(cond.shape) != (10, 1024) or not torch.isfinite(cond).all():
        raise ValueError("RhinoVLA full W8 AdaRMS requires finite [10,1024] conditions")
    owners = getattr(expert, "_rpu_full_w8_cond_owners", None)
    if owners is None or loop_weights is None or "final_norm_w_col" not in loop_weights:
        raise ValueError("RhinoVLA full W8 AdaRMS requires installed quantized cond/IO owners")
    weights = [*owners["cold_weights"], loop_weights["final_norm_w_col"]]
    scales = [*owners["cold_scales"], loop_weights["io_scales"][4]]
    biases = [*owners["cold_biases"], loop_weights["final_norm_b"]]
    cond_rpu = cond.to("rpu")
    results = []
    for index, (weight, scale, bias) in enumerate(zip(weights, scales, biases)):
        # No dequantized CPU Linear: this dispatched op executes shipped W8A16
        # ACC16 hardware and returns fresh FP16 DDR output at the cold boundary.
        output = torch.ops.rpu.linear_w8a16(cond_rpu, weight, scale, bias).cpu()
        output[:, :1024] += 1.0
        if index < 18:
            output[:, 3072:4096] += 1.0
            if precompute_gate_tanh:
                # Use the owner's actual RPU precision for both fixed gates.
                # Final-norm's gate is never read.
                gates = torch.cat((output[:, 2048:3072], output[:, 5120:6144]),
                                  dim=1).contiguous().to("rpu")
                gates = (torch.ops.rpu.rhino_vla_tanh_high_precision(gates)
                         if high_precision else torch.tanh(gates)).cpu()
                output[:, 2048:3072] = gates[:, :1024]
                output[:, 5120:6144] = gates[:, 1024:]
        results.append(output)
    pair_table = torch.stack(results[:18], dim=1).to("rpu").contiguous()
    final_table = results[18].to("rpu").contiguous()
    loop_weights["adarms_cold_owners"] = (weights, scales, biases)
    return pair_table, final_table, weights, scales, biases


def _prepare_denoise_time_proj_table(action_io, cond_all_cpu):
    """Precompute the timestep-only half of action_time_mlp_in.

    action_time_mlp_in([action_embeds, time_emb]) is split by linearity in the
    C++ loop. The time half depends only on the fixed denoise timestep schedule,
    so it can be materialized once as [steps,H] and DMA'd during replay.
    """
    cond = cond_all_cpu.detach().cpu().to(dtype=torch.float32).contiguous()
    H = int(action_io.config.width)
    if cond.dim() != 2 or int(cond.shape[1]) != H:
        raise RuntimeError(
            f"RhinoVLA time-proj precompute expects cond_all [steps,{H}], "
            f"got {tuple(cond.shape)}")
    linear = action_io.action_time_mlp_in
    W = linear.weight.detach().cpu().to(dtype=torch.float32)
    b = linear.bias.detach().cpu().to(dtype=torch.float32)
    if W.shape != (H, 2 * H):
        raise RuntimeError(
            f"action_time_mlp_in.weight shape {tuple(W.shape)}, expected {(H, 2 * H)}")
    if b.shape != (H,):
        raise RuntimeError(
            f"action_time_mlp_in.bias shape {tuple(b.shape)}, expected {(H,)}")
    table = torch.nn.functional.linear(cond, W[:, H:].contiguous(), b).contiguous()
    table = table.to(dtype=torch.float16, device="rpu").contiguous()
    print(f"[Rhino] time-proj precomputed: table={tuple(table.shape)}")
    return table


def _direct_action_input(action_io):
    """v3 projects actions directly; earlier IO has a paired time MLP."""
    has_in = hasattr(action_io, "action_time_mlp_in")
    has_out = hasattr(action_io, "action_time_mlp_out")
    if has_in != has_out:
        raise ValueError("RhinoVLA action/time MLP requires both input and output layers")
    return not has_in


def _prepare_denoise_loop_weights(expert, action_io, mask_proj, *, full_w8a16=False):
    """Prepare RhinoVLA action/final-norm weights for the Phase-3 loop op.

    The C++ loop pre/post hooks use single-core SPM GEMMs for the action IO
    path. Inputs with non-16-aligned K (72-D action, 8-D state) are padded in
    Python; the C++ op receives padded x/action-mask/state tensors and slices
    the final padded action back on the Python side.
    """
    import torch.nn.functional as F
    from rpu_backend.runtime.weights import transform_linear_weight

    cfg = action_io.config
    H = int(cfg.width)
    AD = int(cfg.action_dim)
    SD = int(cfg.state_dim)
    AH = int(cfg.action_horizon)
    ADP = ((AD + 15) // 16) * 16
    SDP = ((max(SD, 1) + 15) // 16) * 16
    if full_w8a16 and ((H, AD, SD, AH) != (1024, 96, 96, 30)
                       or not _direct_action_input(action_io)):
        raise ValueError("RhinoVLA full W8 IO requires ordinary v3 H1024/96D/H30")
    io_scales = {}

    def col_w(weight, *, pad_cols=0, pad_rows=0, scale_name=None):
        w = weight.detach().cpu().to(dtype=torch.float16).contiguous()
        if pad_cols:
            w = F.pad(w, (0, pad_cols))
        if pad_rows:
            w = F.pad(w, (0, 0, 0, pad_rows))
        if full_w8a16:
            w, scale = _quantize_full_w8_weight(w)
            io_scales[scale_name] = scale.to("rpu")
        return transform_linear_weight(
            w.contiguous(), partition=1, num_cores=1).to("rpu").contiguous()

    def bias(bias, *, pad=0):
        if bias is None:
            raise RuntimeError("RhinoVLA denoise-loop expected Linear bias")
        b = bias.detach().cpu().to(dtype=torch.float16).contiguous()
        if pad:
            b = F.pad(b, (0, pad))
        return b.to("rpu").contiguous()

    def folded_action_time_in():
        # Fold time_in_action(action_in_proj(x)) into one small action-dim GEMM.
        # Use fp16-rounded weights/biases because the old RPU path consumed fp16
        # parameters before the second linear.
        action_w = action_io.action_in_proj.weight.detach().cpu().to(dtype=torch.float16)
        action_b = action_io.action_in_proj.bias.detach().cpu().to(dtype=torch.float16)
        time_action_w = mlp_in_w[:, :H].detach().cpu().to(dtype=torch.float16)
        if action_w.shape != (H, AD):
            raise RuntimeError(
                f"action_in_proj.weight shape {tuple(action_w.shape)}, expected {(H, AD)}")
        if action_b.shape != (H,):
            raise RuntimeError(
                f"action_in_proj.bias shape {tuple(action_b.shape)}, expected {(H,)}")
        action_w = action_w.to(dtype=torch.float32)
        if ADP != AD:
            action_w = F.pad(action_w, (0, ADP - AD))
        folded_w = torch.matmul(
            time_action_w.to(dtype=torch.float32),
            action_w).to(dtype=torch.float16).contiguous()
        folded_b = torch.matmul(
            time_action_w.to(dtype=torch.float32),
            action_b.to(dtype=torch.float32)).to(dtype=torch.float16).contiguous()
        return (
            transform_linear_weight(
                folded_w, partition=1, num_cores=1).to("rpu").contiguous(),
            folded_b.to("rpu").contiguous(),
        )

    if action_io.state_proj is None or mask_proj is None:
        raise RuntimeError("RhinoVLA Phase3 loop currently requires state token + mask_proj")

    direct = _direct_action_input(action_io)
    if direct:
        # Keep the legacy op's tensor ABI, without allocating dummy matrices.
        # Native direct-input routes never read these empty tensors.
        empty = torch.empty(0, dtype=torch.float16, device="rpu")
        time_weights = dict.fromkeys((
            "action_time_in_w", "action_time_in_b", "time_in_action_w",
            "time_in_time_w", "time_in_b", "time_out_w", "time_out_b",
        ), empty)
    else:
        mlp_in_w = action_io.action_time_mlp_in.weight.detach().cpu()
        if mlp_in_w.shape != (H, 2 * H):
            raise RuntimeError(
                f"action_time_mlp_in.weight shape {tuple(mlp_in_w.shape)}, "
                f"expected {(H, 2 * H)}")
        action_time_in_w, action_time_in_b = folded_action_time_in()
        time_weights = {
            "action_time_in_w": action_time_in_w,
            "action_time_in_b": action_time_in_b,
            "time_in_action_w": col_w(mlp_in_w[:, :H]),
            "time_in_time_w": col_w(mlp_in_w[:, H:]),
            "time_in_b": bias(action_io.action_time_mlp_in.bias),
            "time_out_w": col_w(action_io.action_time_mlp_out.weight),
            "time_out_b": bias(action_io.action_time_mlp_out.bias),
        }
    final_col = None
    if full_w8a16:
        final_w, final_b = _fold_final_norm_dense(expert.norm)
        final_q, final_scale = _quantize_full_w8_weight(final_w)
        final_w = transform_linear_weight(final_q, partition=0, num_cores=8).to("rpu")
        final_col = transform_linear_weight(final_q, partition=1, num_cores=8).to("rpu")
        final_b = final_b.to(dtype=torch.float16, device="rpu").contiguous()
        io_scales["final_norm"] = final_scale.to("rpu")
    else:
        final_w, final_b = _prepare_final_norm_dense_weights(expert.norm)
    weights = {
        "action_in_w": col_w(action_io.action_in_proj.weight,
                             pad_cols=ADP - AD, scale_name="action_in"),
        "action_in_b": bias(action_io.action_in_proj.bias),
        **time_weights,
        "direct_action_input": direct,
        "state_w": col_w(action_io.state_proj.weight, pad_cols=SDP - SD, scale_name="state"),
        "state_b": bias(action_io.state_proj.bias),
        "state_mask_w": col_w(mask_proj["state_mask_proj"].weight,
                              pad_cols=SDP - SD, scale_name="state_mask"),
        "state_mask_b": bias(mask_proj["state_mask_proj"].bias),
        "action_mask_w": col_w(mask_proj["action_mask_proj"].weight,
                               pad_cols=ADP - AD, scale_name="action_mask"),
        "action_mask_b": bias(mask_proj["action_mask_proj"].bias),
        "final_norm_w": final_w,
        "final_norm_b": final_b,
        "action_out_w": col_w(action_io.action_out_proj.weight,
                              pad_rows=ADP - AD, scale_name="action_out"),
        "action_out_b": bias(action_io.action_out_proj.bias, pad=ADP - AD),
        "action_dim": AD,
        "action_dim_pad": ADP,
        "state_dim": SD,
        "state_dim_pad": SDP,
        "action_horizon": AH,
        "suffix_len": AH + 1,
    }
    if full_w8a16:
        weights["io_scales"] = [io_scales[name] for name in (
            "action_in", "state", "state_mask", "action_mask", "final_norm", "action_out")]
        weights["final_norm_w_col"] = final_col
    return weights


def _gather_expert_weights(expert, *, full_w8a16=False):
    """Gather all per-layer weight lists for the RhinoVLA set_weights op.

    Returns a tuple of 15 lists:
      (q_w, k_w, v_w, o_w, gate, up, down,
       attn_dw, attn_db, mlp_dw, mlp_db, pair_dw, pair_db, q_norm, k_norm)

    Standard attention+MLP weights are assumed to already be swizzled by
    convert_linear_weights_inplace (with skip_names={'cond'}). QK norm + cond
    dense weights are prepared here.
    """
    num_layers = len(expert.layers)

    q_w = [expert.layers[i].self_attn.q_proj.weight for i in range(num_layers)]
    k_w = [expert.layers[i].self_attn.k_proj.weight for i in range(num_layers)]
    v_w = [expert.layers[i].self_attn.v_proj.weight for i in range(num_layers)]
    o_w = [expert.layers[i].self_attn.o_proj.weight for i in range(num_layers)]
    gate = [expert.layers[i].mlp.gate_proj.weight for i in range(num_layers)]
    up = [expert.layers[i].mlp.up_proj.weight for i in range(num_layers)]
    down = [expert.layers[i].mlp.down_proj.weight for i in range(num_layers)]

    q_norm = [
        expert.layers[i].self_attn.q_norm.weight
        .detach().to(dtype=torch.float16, device='rpu').contiguous()
        for i in range(num_layers)
    ]
    k_norm = [
        expert.layers[i].self_attn.k_norm.weight
        .detach().to(dtype=torch.float16, device='rpu').contiguous()
        for i in range(num_layers)
    ]

    attn_dw, attn_db, mlp_dw, mlp_db, pair_dw, pair_db = _prepare_cond_dense_weights(
        expert, full_w8a16=full_w8a16)

    return (q_w, k_w, v_w, o_w, gate, up, down,
            attn_dw, attn_db, mlp_dw, mlp_db,
            pair_dw, pair_db,
            q_norm, k_norm)


def _precompute_mrope_cos_sin(rotary_emb, prefix_len, suffix_len):
    """Pre-compute M-RoPE cos/sin on CPU for suffix positions [prefix_len, prefix_len+suffix_len).

    Suffix tokens share all 3 M-RoPE axes -> degenerates to standard 1D RoPE.
    Qwen3-VL cos/sin are concat-format [seq, head_dim] with cos[:, :D/2] ==
    cos[:, D/2:]; the SPM rope kernel expects the HALF-dim [seq, head_dim/2]
    (slicing the duplicate half is lossless and required for correct addressing).

    Returns (cos, sin): each [suffix_len, head_dim/2] fp16 on RPU.
    """
    suffix_positions = torch.arange(
        prefix_len, prefix_len + suffix_len, device="cpu"
    ).unsqueeze(0)  # [1, S]
    position_ids_3d = suffix_positions.unsqueeze(0).expand(3, -1, -1)  # [3, 1, S]

    dummy_x = torch.zeros(1, dtype=torch.float32, device="cpu")
    cos, sin = rotary_emb(dummy_x, position_ids_3d)  # [1, S, head_dim]
    cos = cos.squeeze(0)
    sin = sin.squeeze(0)
    head_dim = cos.shape[-1]
    cos = cos[..., : head_dim // 2]
    sin = sin[..., : head_dim // 2]
    cos = cos.to(dtype=torch.float16, device='rpu').contiguous()  # [suffix_len, head_dim/2]
    sin = sin.to(dtype=torch.float16, device='rpu').contiguous()
    return cos, sin


def _create_kv_cache(config, max_seq_len):
    """Create RPUCache for KV storage in 7D swizzled format."""
    from rpu_backend.api.cache import RPUCache
    from rpu_backend.runtime.weights import NUM_CORES

    return RPUCache(
        num_layers=config.depth,
        batch_size=1,
        max_seq_len=max_seq_len,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        attn_tp=min(NUM_CORES, config.num_key_value_heads),
    )


def _plan_rhino_prefix_kv_execution(
    prefix_pairs,
    *,
    component,
    generation,
    queue_owner_id,
):
    """Resolve one exact typed arena route before any prefix cache mutation."""
    if not prefix_pairs:
        raise ValueError("RhinoVLA prefix planner requires at least one KV layer")
    first_k, first_v = prefix_pairs[0]
    if (
        not isinstance(first_k, torch.Tensor)
        or not isinstance(first_v, torch.Tensor)
        or first_k.dim() != 4
        or tuple(first_k.shape) != tuple(first_v.shape)
    ):
        raise ValueError(
            "RhinoVLA prefix planner requires identical 4D K/V geometry"
        )
    expected_shape = tuple(first_k.shape)
    for layer_idx, (prefix_k, prefix_v) in enumerate(prefix_pairs):
        if (
            not isinstance(prefix_k, torch.Tensor)
            or not isinstance(prefix_v, torch.Tensor)
            or prefix_k.dim() != 4
            or tuple(prefix_k.shape) != tuple(prefix_v.shape)
            or tuple(prefix_k.shape) != expected_shape
        ):
            raise ValueError(
                "RhinoVLA prefix planner requires identical 4D K/V geometry "
                f"across layers; layer {layer_idx} has K={getattr(prefix_k, 'shape', None)} "
                f"V={getattr(prefix_v, 'shape', None)}"
            )
    batch, num_kv_heads, logical_rows, head_dim = expected_shape
    if batch != 1 or num_kv_heads <= 0 or head_dim <= 0:
        raise ValueError(
            "RhinoVLA prefix planner requires batch=1 and positive KV geometry"
        )
    return _plan_rhino_prefix_kv_geometry(
        num_layers=len(prefix_pairs), num_kv_heads=num_kv_heads,
        logical_rows=logical_rows, head_dim=head_dim, component=component,
        generation=generation, queue_owner_id=queue_owner_id,
    )


def _plan_rhino_prefix_kv_geometry(*, num_layers, num_kv_heads, logical_rows,
                                  head_dim, component, generation, queue_owner_id):
    """The original typed arena query, shared with cold cost inspection.

    Forward derives these dimensions from every actual exported K/V pair.
    Cold inspection derives them from the same installed text/expert geometry;
    neither path uploads or substitutes prefix values while planning.
    """
    if any(type(value) is not int or value <= 0
           for value in (num_layers, num_kv_heads, head_dim)) or (
            type(logical_rows) is not int or logical_rows < 0):
        raise ValueError("RhinoVLA prefix planner requires bounded positive KV geometry")
    if logical_rows == 0:
        return None, (), ""

    physical_rows = ((logical_rows + 15) // 16) * 16
    route_arguments = tuple(int(value) for value in (
        torch.rpu.resolve_kvcache_arena_plan(
            0, physical_rows, num_kv_heads, head_dim
        )
    ))
    if len(route_arguments) != 22:
        raise RuntimeError(
            "RhinoVLA prefix planner returned a non-canonical KV route"
        )
    route_argument_digest = sha256(
        ("RPU-RHINOVLA-PREFIX-KV-V1\0" + ",".join(
            str(value) for value in route_arguments
        )).encode("ascii")
    ).hexdigest()

    from rpu_backend.runtime.execution_planner import (
        GRAPH_COMPOSITE_CHILD,
        plan_fixed_component_execution,
    )

    plan = plan_fixed_component_execution(
        logical_rows,
        execution_len=physical_rows,
        chunk_size=physical_rows,
        component_id=str(component),
        stage="prefix_kv",
        generation=int(generation),
        position=0,
        stage_config={"chunk_size": "auto"},
        graph_mode=GRAPH_COMPOSITE_CHILD,
        physical_metadata=(
            ("num_layers", num_layers),
            ("num_kv_heads", num_kv_heads),
            ("head_dim", head_dim),
            ("source_logical_rows", logical_rows),
            *(tuple(
                (f"kv_route_arg_{index:02d}", value)
                for index, value in enumerate(route_arguments)
            )),
        ),
        queue_owner_id=int(queue_owner_id),
    )
    if plan.selected is None:
        raise RuntimeError("RhinoVLA prefix planner selected no typed KV route")
    return plan, route_arguments, route_argument_digest


def _prefill_kv_cache(cache, prefix_pairs, planned_route_arguments):
    """Pre-fill prefix KV into RPUCache via the RPU insert_kcache/vcache kernels.

    The insert kernels are the source of truth for the 7D swizzled DDR cache
    layout SDPA reads. We write into a prefix-sized temporary 7D cache then
    copy the swizzled bytes into the first slots of the real cache.

    RhinoVLA deltas vs qwenpi05: NO %8 restriction (the temp insert tensors are
    padded to the 16-token cache chunk, which is always a multiple of 8), and
    `prefix_len == 0` is an early return (suffix-only Stage-1 path).
    """
    import math as _math

    # prefix_len == 0 (suffix-only): nothing to insert.
    if not prefix_pairs:
        raise ValueError("RhinoVLA prefix fill requires at least one KV layer")
    first_k, _first_v = prefix_pairs[0]
    if first_k.shape[2] != 0 and not planned_route_arguments:
        raise RuntimeError("RhinoVLA prefix fill requires an exact typed KV route")

    cache.reset_to_position(0)
    if first_k.shape[2] == 0:
        cache.update_position(0)
        return

    sKeyChunk = cache.sKeyChunk
    sValChunk = cache.sValChunk
    headDimChunk = cache.headDimChunk
    nKVHeadChunk = cache.nKVHeadChunk
    nKVHeadVx = cache.nKVHeadVx
    headDimVx = cache.headDimVx
    B = cache.batch_size

    prefix_len_layer = None
    for layer_idx, (prefix_k, prefix_v) in enumerate(prefix_pairs):
        # prefix_k: [B, num_kv_heads, prefix_len, head_dim]
        prefix_len_layer = prefix_k.shape[2]

        # insert_kcache/vcache expect [B, seq_len, num_kv_heads, head_dim].
        pk = prefix_k.transpose(1, 2).to(dtype=torch.float16, device='rpu').contiguous()
        pv = prefix_v.transpose(1, 2).to(dtype=torch.float16, device='rpu').contiguous()

        # Pad prefix to a multiple of sKeyChunk (16) for the 7D cache layout.
        # 16 is always a multiple of 8, so the inserted seq length is 8-aligned
        # regardless of the (arbitrary) real prefix_len -> no %8 restriction.
        sKeyVx_prefix = _math.ceil(prefix_len_layer / sKeyChunk)
        sValVx_prefix = _math.ceil(prefix_len_layer / sValChunk)
        padded_seq_k = sKeyVx_prefix * sKeyChunk
        padded_seq_v = sValVx_prefix * sValChunk
        if padded_seq_k > prefix_len_layer:
            pk = torch.nn.functional.pad(pk, (0, 0, 0, 0, 0, padded_seq_k - prefix_len_layer))
        if padded_seq_v > prefix_len_layer:
            pv = torch.nn.functional.pad(pv, (0, 0, 0, 0, 0, padded_seq_v - prefix_len_layer))

        k_tmp = torch.zeros(
            (B, sKeyVx_prefix, nKVHeadVx, headDimVx,
             nKVHeadChunk, sKeyChunk, headDimChunk),
            dtype=torch.float16, device='rpu',
        )
        v_tmp = torch.zeros(
            (B, sValVx_prefix, nKVHeadVx, headDimVx,
             nKVHeadChunk, headDimChunk, sValChunk),
            dtype=torch.float16, device='rpu',
        )

        # Temporary offsets accumulate across layers and are reset below.
        torch.rpu.insert_kvcache_arena(
            pk, pv, k_tmp, v_tmp, 0, list(planned_route_arguments)
        )

        cache.k_caches[layer_idx][:, :sKeyVx_prefix, ...].copy_(k_tmp)
        cache.v_caches[layer_idx][:, :sValVx_prefix, ...].copy_(v_tmp)

    # Free the arena staging buffers accumulated across the layer loop.
    torch.ops.rpu.spm_alloc_reset_temporary()
    # Set position to prefix_len (unpadded) so suffix starts at the correct offset.
    cache.update_position(prefix_len_layer)


def copy_prefix_cache_from_text_cache(
    expert_cache,
    text_cache,
    *,
    source_layer_offset: int,
    num_layers: int,
    prefix_len: int,
):
    """Copy a swizzled on-device text prefix KV into the RhinoVLA expert cache.

    Both caches use ``RPUCache``'s byte-identical 7D K/V layout. This copies the
    already-swizzled prefix chunks from text layers ``[source_layer_offset,
    source_layer_offset + num_layers)`` into expert layers ``[0, num_layers)``
    and sets the expert cache position to the unpadded ``prefix_len``.
    """
    import math as _math

    prefix_len = int(prefix_len)
    source_layer_offset = int(source_layer_offset)
    num_layers = int(num_layers)
    if prefix_len < 0:
        raise ValueError(f"prefix_len must be non-negative, got {prefix_len}")
    if source_layer_offset < 0:
        raise ValueError(
            f"source_layer_offset must be non-negative, got {source_layer_offset}")
    if source_layer_offset + num_layers > text_cache.num_layers:
        raise ValueError(
            "text cache does not have enough layers: "
            f"offset={source_layer_offset} num_layers={num_layers} "
            f"text_layers={text_cache.num_layers}")
    if num_layers > expert_cache.num_layers:
        raise ValueError(
            f"expert cache layers {expert_cache.num_layers} < requested {num_layers}")
    if prefix_len > text_cache.max_seq_len or prefix_len > expert_cache.max_seq_len:
        raise ValueError(
            f"prefix_len={prefix_len} exceeds cache capacity "
            f"text={text_cache.max_seq_len} expert={expert_cache.max_seq_len}")

    fields = (
        "batch_size", "num_kv_heads", "head_dim", "dtype",
        "sKeyChunk", "sValChunk", "headDimChunk",
        "nKVHeadChunk", "nKVHeadVx", "headDimVx",
    )
    for field in fields:
        src_v = getattr(text_cache, field)
        dst_v = getattr(expert_cache, field)
        if src_v != dst_v:
            raise ValueError(
                f"RPUCache layout mismatch for {field}: text={src_v} expert={dst_v}")

    expert_cache.reset_to_position(0)
    if prefix_len == 0:
        return

    s_key_vx = _math.ceil(prefix_len / expert_cache.sKeyChunk)
    s_val_vx = _math.ceil(prefix_len / expert_cache.sValChunk)
    if s_key_vx > text_cache.sKeyVx or s_key_vx > expert_cache.sKeyVx:
        raise ValueError(
            f"K prefix chunks {s_key_vx} exceed cache chunks "
            f"text={text_cache.sKeyVx} expert={expert_cache.sKeyVx}")
    if s_val_vx > text_cache.sValVx or s_val_vx > expert_cache.sValVx:
        raise ValueError(
            f"V prefix chunks {s_val_vx} exceed cache chunks "
            f"text={text_cache.sValVx} expert={expert_cache.sValVx}")

    for layer_idx in range(num_layers):
        src_idx = source_layer_offset + layer_idx
        expert_cache.k_caches[layer_idx].narrow(1, 0, s_key_vx).copy_(
            text_cache.k_caches[src_idx].narrow(1, 0, s_key_vx))
        expert_cache.v_caches[layer_idx].narrow(1, 0, s_val_vx).copy_(
            text_cache.v_caches[src_idx].narrow(1, 0, s_val_vx))

    expert_cache.reset_to_position(prefix_len)
