"""RhinoVLA RPU weight preparation.

Self-contained weight-conversion helpers for the RhinoVLA action expert. The
expert math is identical to QwenPI05's, so these helpers are ported from
adapters/qwenpi05/fused.py (config-agnostic — depth/dims read from `expert`)
and reused via the qwenpi05 C++ ops. RhinoVLA-specific:
  - `convert_expert_for_rpu` enforces half() -> swizzle -> rpu.
  - `_prefill_kv_cache` supports arbitrary prefix_len (no %8 restriction) and
    handles prefix_len == 0.
"""

import torch


def convert_expert_for_rpu(expert):
    """Move a RhinoVLA action expert to RPU in the CORRECT order.

    convert_linear_weights_inplace swizzles by weight.element_size();
    swizzling fp32 then casting to fp16 produces a wrong-dwidth byte layout.
    Order MUST be: half() -> swizzle (skip cond) -> to('rpu').
    """
    from rpu_backend.runtime.weights import convert_linear_weights_inplace
    # convert_linear_weights_inplace mutates weights irreversibly (row/col swizzle);
    # a second call double-swizzles → right L2 norm, wrong direction.
    # Guard like qwenpi05.convert: refuse a repeat, and stamp _started before the
    # mutation so a failure mid-swizzle can't be re-run silently.
    if getattr(expert, "_rpu_swizzled", False):
        raise RuntimeError(
            "convert_expert_for_rpu: expert already has _rpu_swizzled=True from a "
            "prior call. Re-swizzling corrupts the weights. Rebuild the "
            "expert from the checkpoint instead of converting twice."
        )
    if getattr(expert, "_rpu_swizzle_started", False):
        raise RuntimeError(
            "convert_expert_for_rpu: a prior conversion started but did not finish "
            "(_rpu_swizzle_started=True). Weights may be partially swizzled — rebuild "
            "the expert from the checkpoint."
        )
    expert = expert.half()
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


def _prepare_cond_dense_weights(expert):
    """Swizzle per-layer AdaRMS cond weights for RPU row-partition GEMV.

    For each layer x each of {input_layernorm, post_attention_layernorm}:
      norm.cond.weight [3*width, width] (γ-folded on scale block)
      -> transform_linear_weight(partition=0, num_cores=8)   # row partition (K split)
      -> .to('rpu')

    Returns six parallel lists (one entry per layer):
      attn_dense_w_list, attn_dense_b_list, mlp_dense_w_list, mlp_dense_b_list,
      pair_dense_w_list, pair_dense_b_list
    """
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


def _prepare_denoise_adarms_tables(expert, cond_all_cpu):
    """Precompute denoise-loop AdaRMS dense outputs for fixed timestep conds.

    RhinoVLA's denoise schedule uses the same timestep embedding for every
    inference with the same ``steps`` value. The per-layer AdaRMS cond dense
    output therefore does not depend on image/text/state/action inputs. This
    helper materializes:

      * pair_table:  [steps, layers, 6H]  = attn + mlp AdaRMS params
      * final_table: [steps, 3H]          = final norm AdaRMS params

    Both tensors are fp16 RPU tensors consumed by a C++ mutable-DMA fast path.
    """
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


def _prepare_denoise_loop_weights(expert, action_io, mask_proj):
    """Prepare RhinoVLA action/final-norm weights for the denoise-loop op.

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

    def col_w(weight, *, pad_cols=0, pad_rows=0):
        w = weight.detach().cpu().to(dtype=torch.float16).contiguous()
        if pad_cols:
            w = F.pad(w, (0, pad_cols))
        if pad_rows:
            w = F.pad(w, (0, 0, 0, pad_rows))
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
        # Use fp16-rounded weights/biases because the denoise op consumes fp16
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
        raise RuntimeError("RhinoVLA denoise loop requires state token + mask_proj")

    mlp_in_w = action_io.action_time_mlp_in.weight.detach().cpu()
    if mlp_in_w.shape != (H, 2 * H):
        raise RuntimeError(
            f"action_time_mlp_in.weight shape {tuple(mlp_in_w.shape)}, "
            f"expected {(H, 2 * H)}")

    final_w, final_b = _prepare_final_norm_dense_weights(expert.norm)
    action_time_in_w, action_time_in_b = folded_action_time_in()
    weights = {
        "action_in_w": col_w(action_io.action_in_proj.weight,
                             pad_cols=ADP - AD),
        "action_in_b": bias(action_io.action_in_proj.bias),
        "action_time_in_w": action_time_in_w,
        "action_time_in_b": action_time_in_b,
        "time_in_action_w": col_w(mlp_in_w[:, :H]),
        "time_in_time_w": col_w(mlp_in_w[:, H:]),
        "time_in_b": bias(action_io.action_time_mlp_in.bias),
        "time_out_w": col_w(action_io.action_time_mlp_out.weight),
        "time_out_b": bias(action_io.action_time_mlp_out.bias),
        "state_w": col_w(action_io.state_proj.weight, pad_cols=SDP - SD),
        "state_b": bias(action_io.state_proj.bias),
        "state_mask_w": col_w(mask_proj["state_mask_proj"].weight,
                              pad_cols=SDP - SD),
        "state_mask_b": bias(mask_proj["state_mask_proj"].bias),
        "action_mask_w": col_w(mask_proj["action_mask_proj"].weight,
                               pad_cols=ADP - AD),
        "action_mask_b": bias(mask_proj["action_mask_proj"].bias),
        "final_norm_w": final_w,
        "final_norm_b": final_b,
        "action_out_w": col_w(action_io.action_out_proj.weight,
                              pad_rows=ADP - AD),
        "action_out_b": bias(action_io.action_out_proj.bias, pad=ADP - AD),
        "action_dim": AD,
        "action_dim_pad": ADP,
        "state_dim": SD,
        "state_dim_pad": SDP,
        "action_horizon": AH,
        "suffix_len": AH + 1,
    }
    return weights


def _gather_expert_weights(expert):
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

    attn_dw, attn_db, mlp_dw, mlp_db, pair_dw, pair_db = _prepare_cond_dense_weights(expert)

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


def _prefill_kv_cache(cache, prefix_key_values, prefix_layer_offset, num_layers):
    """Pre-fill prefix KV into RPUCache via the RPU insert_kcache/vcache kernels.

    The insert kernels are the source of truth for the 7D swizzled DDR cache
    layout SDPA reads. We write into a prefix-sized temporary 7D cache then
    copy the swizzled bytes into the first slots of the real cache.

    RhinoVLA deltas vs qwenpi05: NO %8 restriction (the temp insert tensors are
    padded to the 16-token cache chunk, which is always a multiple of 8), and
    `prefix_len == 0` is an early return for suffix-only execution.
    """
    from vla.model.modules.action_model.qwen3_pi05_expert import get_cache_layer
    import math as _math

    cache.reset_to_position(0)

    # prefix_len == 0 (suffix-only): nothing to insert.
    first_k, _first_v = get_cache_layer(prefix_key_values, prefix_layer_offset)
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
    for layer_idx in range(num_layers):
        prefix_k, prefix_v = get_cache_layer(
            prefix_key_values, prefix_layer_offset + layer_idx
        )
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

        # Arena-staged insert (not the basic insert_kcache, which stages through
        # SDK LocalSPM and collides with the cached denoise graph's SPM pool).
        # The arena variant stages
        # through the repo SPM_ALLOC bump arena (free here); numerically identical
        # (same unified swizzle kernel). Offsets accumulate across the layer loop;
        # one reset_temporary below frees them all.
        torch.rpu.insert_kcache_arena(pk, k_tmp, 0)
        torch.rpu.insert_vcache_arena(pv, v_tmp, 0)

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
