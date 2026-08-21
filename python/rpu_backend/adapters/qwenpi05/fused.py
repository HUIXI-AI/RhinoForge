"""
QwenPI05 Fused Converter

Patches a QwenPI05ActionExpert model instance to use the FusedModelBase
all-layers-once C++ execution path (QwenPI05Model).

Execution contract:
  - Uses 'cond' (not 'dense') for AdaRMS conditioning Linear
  - Extracts QK norm weights per-layer [head_dim]
  - cos/sin passed per-forward (not at set_weights time)
  - Manual KV cache with RPUCache for correct 7D swizzled format

Usage:
    from rpu_backend.runtime.weights import convert_linear_weights_inplace
    from rpu_backend.adapters.qwenpi05.fused import patch_qwenpi05_for_rpu_fused
    expert = expert.half()
    convert_linear_weights_inplace(expert, skip_names={"cond"})
    expert = expert.to("rpu")
    handle = patch_qwenpi05_for_rpu_fused(expert)
"""

import copy
import types
import weakref

import torch


def _patch_adarms_norm_for_rpu():
    """Monkey-patch AdaRMSNorm.forward so cond matches the Linear weight's
    dtype/device before the projection.

    The unpatched forward feeds `adarms_cond` (float32 from time_emb) directly
    into `self.cond`, which on RPU has fp16 weights → `rpu_linear: only half
    supported`. This wrapper casts/transfers `cond` as needed while leaving
    CPU/CUDA behavior identical (the `.to()` is a no-op when already matching).

    Idempotent via the `_rpu_patched` flag on the class.
    """
    try:
        from vla.model.modules.action_model.qwen3_pi05_expert import AdaRMSNorm
    except ImportError:
        print("[QwenPI05-RPU] Warning: AdaRMSNorm import failed, skipping patch")
        return
    if getattr(AdaRMSNorm, '_rpu_patched', False):
        return

    original_forward = AdaRMSNorm.forward

    def patched_forward(self, hidden_states, cond=None):
        hidden_states = self.norm(hidden_states)
        if self.cond is None or cond is None:
            return hidden_states, None
        # Cast cond to match self.cond weight (fixes fp32 time_emb vs fp16 RPU
        # Linear weight). No-op for matching dtype/device.
        cond = cond.to(device=self.cond.weight.device, dtype=self.cond.weight.dtype)
        scale, shift, gate = self.cond(cond).chunk(3, dim=-1)
        hidden_states = hidden_states * (1 + scale[:, None, :]) + shift[:, None, :]
        return hidden_states, gate[:, None, :]

    AdaRMSNorm.forward = patched_forward
    AdaRMSNorm._rpu_patched = True
    print("[QwenPI05-RPU] Patched AdaRMSNorm.forward for dtype/device matching")


def _patch_action_io_decode_actions_for_rpu():
    """Monkey-patch QwenPI05ActionIO.decode_actions to avoid fp32 → fp16
    Linear on RPU.

    The unpatched decode_actions does
        self.action_out_proj(action_hidden.to(torch.float32))
    which forces fp32 input into a potentially fp16 RPU Linear. The patched
    version matches the Linear weight's dtype/device, runs the projection,
    then casts the final output to fp32 for downstream DDIM math.

    Idempotent via the `_rpu_patched` flag on the class.
    """
    try:
        from vla.model.modules.action_model.qwen3_pi05_expert import QwenPI05ActionIO
    except ImportError:
        print("[QwenPI05-RPU] Warning: QwenPI05ActionIO import failed, skipping patch")
        return
    if getattr(QwenPI05ActionIO, '_rpu_patched', False):
        return

    def patched_decode_actions(self, suffix_hidden):
        action_hidden = suffix_hidden[:, -self.config.action_horizon:]
        w = self.action_out_proj.weight
        action_hidden = action_hidden.to(device=w.device, dtype=w.dtype)
        out = self.action_out_proj(action_hidden)
        return out.to(torch.float32)

    QwenPI05ActionIO.decode_actions = patched_decode_actions
    QwenPI05ActionIO._rpu_patched = True
    print("[QwenPI05-RPU] Patched QwenPI05ActionIO.decode_actions for RPU fp16")


def _qwenpi05_destroy_handle(h):
    """Release C++ QwenPI05Model handle. Called by weakref.finalize on GC."""
    try:
        torch.ops.rpu.qwenpi05_destroy(h)
    except Exception:
        # Swallow -- during interpreter shutdown the op may be gone
        pass


_QWENPI05_RUNTIME_INSTALL_ATTRS = (
    "_rpu_qwenpi05_graph_cache",
    "_rpu_cos_cached",
    "_rpu_sin_cached",
    "_rpu_kv_cache",
    "_rpu_qwenpi05_handle",
    "_rpu_qwenpi05_handle_finalizer",
    "forward",
)


def _prepare_cond_dense_weights(expert):
    """Swizzle per-layer AdaRMS cond weights for RPU row-partition GEMV.

    For each layer x each of {input_layernorm, post_attention_layernorm}:
      norm.cond.weight [3*width, width] (γ-folded on scale block)
      -> transform_linear_weight(partition=0, num_cores=8)   # row partition (K split)
      -> .to('rpu')

    The C++ `adarms_gemv_to_spm` runs GEMV with partition=0, num_cores=8:
    each core holds local_k=H/8 columns of the full [3H, H] weight and
    produces a partial [3H]; a fused all_reduce_sum_residual then folds in
    the bias and broadcasts the full [3H] back to every core.

    Returns four parallel lists (one entry per layer):
      attn_dense_w_list, attn_dense_b_list, mlp_dense_w_list, mlp_dense_b_list

    The helper is self-contained: if `norm._rpu_cond_w_rp` is already
    registered (idempotent re-patch), it is reused; otherwise the weight is
    recomputed from the live `norm.cond` Linear.
    """
    from rpu_backend.runtime.weights import transform_linear_weight, NUM_CORES

    attn_w, attn_b, mlp_w, mlp_b = [], [], [], []
    gamma_fold_sites = 0

    for layer_idx, layer in enumerate(expert.layers):
        for src_norm_name, (w_out, b_out) in (
            ('input_layernorm', (attn_w, attn_b)),
            ('post_attention_layernorm', (mlp_w, mlp_b)),
        ):
            norm = getattr(layer, src_norm_name)
            if not hasattr(norm, 'cond') or norm.cond is None:
                raise RuntimeError(
                    f"patch_qwenpi05_for_rpu_fused: "
                    f"layer {layer_idx}.{src_norm_name} has no .cond -- is this "
                    f"an AdaRMS norm? (Expected AdaRMSNorm with cond Linear.)"
                )

            # Reuse an existing row-partitioned, gamma-folded buffer. Other
            # buffer layouts are intentionally not accepted here.
            if hasattr(norm, '_rpu_cond_w_rp') and hasattr(norm, '_rpu_cond_b_rp'):
                w_out.append(norm._rpu_cond_w_rp)
                b_out.append(norm._rpu_cond_b_rp)
                continue

            if norm.cond.bias is None:
                raise RuntimeError(
                    f"patch_qwenpi05_for_rpu_fused: "
                    f"layer {layer_idx}.{src_norm_name}.cond has no bias -- "
                    f"AdaRMS cond must include a bias vector."
                )

            # ---------------------------------------------------------- #
            # γ-fold: the fused C++ kernel assumes the only multiplicative
            # weight in the RMSNorm is (1+scale). QwenPI05's AdaRMSNorm is:
            #     out_cpu = (x/rms(x)) * γ * (1+scale) + shift
            # The kernel computes:
            #     out_rpu = (x/rms(x)) * attn_scale + attn_shift
            # So to match CPU exactly:
            #     attn_scale = γ*(1+scale)   →  fold γ into row-block 0
            #     attn_shift = shift         →  row-block 1 UNCHANGED (γ is NOT
            #                                   applied to shift in CPU math)
            #     gate       unchanged       →  row-block 2
            #
            # Block 0 fold (since post-GEMV the kernel adds +1 to block 0):
            #     W_s_new = γ * W_s
            #     B_s_new = γ*b_s + (γ-1)
            # ---------------------------------------------------------- #
            if not hasattr(norm.norm, 'weight') or norm.norm.weight is None:
                raise RuntimeError(
                    f"patch_qwenpi05_for_rpu_fused: "
                    f"layer {layer_idx}.{src_norm_name}.norm has no learnable "
                    f"weight (γ). Expected Qwen3VLTextRMSNorm."
                )

            # All math on CPU in fp32 for precision, then cast to fp16.
            gamma = norm.norm.weight.detach().cpu().to(dtype=torch.float32)  # [H]
            W_f32 = norm.cond.weight.detach().cpu().to(dtype=torch.float32)  # [3H, H]
            b_f32 = norm.cond.bias.detach().cpu().to(dtype=torch.float32)    # [3H]

            H = gamma.shape[0]
            if W_f32.shape[0] != 3 * H or W_f32.shape[1] != H:
                raise RuntimeError(
                    f"patch_qwenpi05_for_rpu_fused: "
                    f"layer {layer_idx}.{src_norm_name}.cond.weight has shape "
                    f"{tuple(W_f32.shape)}, expected [{3*H}, {H}]."
                )
            if b_f32.shape[0] != 3 * H:
                raise RuntimeError(
                    f"patch_qwenpi05_for_rpu_fused: "
                    f"layer {layer_idx}.{src_norm_name}.cond.bias has shape "
                    f"{tuple(b_f32.shape)}, expected [{3*H}]."
                )

            # Row-block split: scale=[0:H], shift=[H:2H], gate=[2H:3H]
            W_s, W_sh, W_g = W_f32[0:H], W_f32[H:2*H], W_f32[2*H:3*H]
            b_s, b_sh, b_g = b_f32[0:H], b_f32[H:2*H], b_f32[2*H:3*H]

            gamma_col = gamma.unsqueeze(1)  # [H, 1] for row-wise scale
            # Block 0 (scale): fold γ so kernel's (1+scale_new) = γ*(1+scale_cpu)
            W_s_new  = gamma_col * W_s
            B_s_new  = gamma * b_s + (gamma - 1.0)
            # Block 1 (shift): CPU applies shift AFTER (x/rms*γ)*(1+scale), i.e.
            # shift is NOT multiplied by γ. Leave untouched.
            W_sh_new = W_sh
            B_sh_new = b_sh
            # Block 2 (gate): applied in gated residual AFTER the norm; untouched.
            W_g_new  = W_g
            B_g_new  = b_g

            W_folded = torch.cat([W_s_new, W_sh_new, W_g_new], dim=0).contiguous()  # [3H, H]
            b_folded = torch.cat([B_s_new, B_sh_new, B_g_new], dim=0).contiguous()  # [3H]

            # Cast to fp16 and apply row-partition swizzle (K split across 8 cores).
            # The natural [3H, H] weight is retained, and the per-core partial
            # reduction plus bias is fused through all_reduce_sum_residual.
            folded_w_fp16 = W_folded.to(dtype=torch.float16).contiguous()  # [3H, H]
            transformed_w = transform_linear_weight(
                folded_w_fp16, partition=0, num_cores=NUM_CORES
            ).to('rpu').contiguous()
            folded_b_rpu = b_folded.to(dtype=torch.float16).to('rpu').contiguous()

            # Cache the row-partitioned, gamma-folded representation on the module.
            norm.register_buffer('_rpu_cond_w_rp', transformed_w)
            norm.register_buffer('_rpu_cond_b_rp', folded_b_rpu)

            w_out.append(transformed_w)
            b_out.append(folded_b_rpu)
            gamma_fold_sites += 1

    print(f"[RPU] QwenPI05 fused converter: γ folded into cond weights at {gamma_fold_sites} sites")
    return attn_w, attn_b, mlp_w, mlp_b


def _gather_qwenpi05_weights(expert):
    """Gather all 13 per-layer weight lists for qwenpi05_set_weights.

    Returns a tuple of 13 lists:
      (q_w, k_w, v_w, o_w, gate, up, down,
       attn_dw, attn_db, mlp_dw, mlp_db,
       q_norm, k_norm)

    Standard attention+MLP weights are assumed to already be swizzled by
    convert_linear_weights_inplace (with skip_names={'cond'}).
    QK norm weights and cond dense weights are prepared here.
    """
    num_layers = len(expert.layers)

    # Standard attention+MLP weights (already swizzled by convert_linear_weights_inplace)
    q_w = [expert.layers[i].self_attn.q_proj.weight for i in range(num_layers)]
    k_w = [expert.layers[i].self_attn.k_proj.weight for i in range(num_layers)]
    v_w = [expert.layers[i].self_attn.v_proj.weight for i in range(num_layers)]
    o_w = [expert.layers[i].self_attn.o_proj.weight for i in range(num_layers)]
    gate = [expert.layers[i].mlp.gate_proj.weight for i in range(num_layers)]
    up = [expert.layers[i].mlp.up_proj.weight for i in range(num_layers)]
    down = [expert.layers[i].mlp.down_proj.weight for i in range(num_layers)]

    # QK norm weights (QwenPI05-specific, per-layer [head_dim])
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

    # AdaRMS cond dense weights (expand+swizzle)
    attn_dw, attn_db, mlp_dw, mlp_db = _prepare_cond_dense_weights(expert)

    return (q_w, k_w, v_w, o_w, gate, up, down,
            attn_dw, attn_db, mlp_dw, mlp_db,
            q_norm, k_norm)


def _precompute_mrope_cos_sin(rotary_emb, prefix_len, suffix_len):
    """Pre-compute M-RoPE cos/sin on CPU for suffix positions.

    For suffix tokens, all 3 M-RoPE axes share the same positions:
    [prefix_len, prefix_len+1, ..., prefix_len+suffix_len-1].
    This makes M-RoPE degenerate to standard interleaved 1D RoPE.

    The Qwen3-VL rotary returns cos/sin in concat format [seq, head_dim]
    where the first and second halves are identical duplicates
    (cos[..., :D/2] == cos[..., D/2:]). The RPU SPM rope kernel expects
    the HALF-dim format [seq, head_dim/2] — slicing off the duplicate
    second half is mathematically lossless and required for the kernel
    to compute the correct per-position cos/sin addresses.

    Returns:
        (cos, sin): each [suffix_len, head_dim/2] fp16 on RPU
    """
    suffix_positions = torch.arange(
        prefix_len, prefix_len + suffix_len, device="cpu"
    ).unsqueeze(0)  # [1, S]
    position_ids_3d = suffix_positions.unsqueeze(0).expand(3, -1, -1)  # [3, 1, S]

    # Use CPU for M-RoPE computation (autocast issues on RPU)
    dummy_x = torch.zeros(1, dtype=torch.float32, device="cpu")
    cos, sin = rotary_emb(dummy_x, position_ids_3d)
    # Output: [1, suffix_len, head_dim] after interleaved M-RoPE + cat
    cos = cos.squeeze(0)  # [suffix_len, head_dim]
    sin = sin.squeeze(0)
    # Slice to first half — Qwen3-VL concat format has cos[:, :D/2] == cos[:, D/2:]
    head_dim = cos.shape[-1]
    cos = cos[..., : head_dim // 2]
    sin = sin[..., : head_dim // 2]
    cos = cos.to(dtype=torch.float16, device='rpu').contiguous()  # [suffix_len, head_dim/2]
    sin = sin.to(dtype=torch.float16, device='rpu').contiguous()
    return cos, sin


def _create_kv_cache(config, max_seq_len):
    """Create RPUCache for KV storage in 7D swizzled format.

    RPUCache produces the 7D swizzled layout required by C++
    insert_kcache/insert_vcache and SDPA kernels. Position management and
    prefix pre-fill are manual.
    """
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


def _pad_to_multiple_of_8(tensor, dim=1):
    """Pad tensor along `dim` to make that dimension a multiple of 8.

    Required because rpu_backend.insert_kcache/insert_vcache kernels
    need seq_len divisible by 8 (8 cores).

    Returns (padded_tensor, original_len).
    """
    size = tensor.shape[dim]
    remainder = size % 8
    if remainder == 0:
        return tensor, size
    pad_amount = 8 - remainder
    # Build padding spec: F.pad expects pairs from last dim backward
    ndim = tensor.dim()
    pad_spec = [0] * (2 * ndim)
    # dim index from the end: ndim - 1 - dim
    pad_idx = 2 * (ndim - 1 - dim) + 1  # upper pad for this dim
    pad_spec[pad_idx] = pad_amount
    import torch.nn.functional as F
    padded = F.pad(tensor, pad_spec, value=0.0)
    return padded, size


def _prefill_kv_cache_python_swizzle(cache, prefix_key_values, prefix_layer_offset, num_layers):
    """Pre-fill prefix KV by directly writing RPUCache's 7D swizzled format.

    The DDR insert_kcache kernel validates cache sKeyVx == ceil(input_seq/16),
    which prevents inserting a short prefix into a large cache. Instead, we
    manually swizzle prefix K/V from [B, heads, seq, head_dim] into the 7D
    cache layout and write directly into the cache tensors.

    K-cache layout: [B, sKeyVx, nKVHeadVx, headDimVx, nKVHeadChunk, sKeyChunk, headDimChunk]
    V-cache layout: [B, sValVx, nKVHeadVx, headDimVx, nKVHeadChunk, headDimChunk, sValChunk]

    Swizzle is the inverse of the unswizzle permutation in RPUCache.to_dynamic_cache():
      K unswizzle: permute(0,4,2,1,5,3,6) → so swizzle: permute(0,4,2,5,1,6,3) on 7D intermediate
      V unswizzle: permute(0,4,2,1,6,3,5) → so swizzle: permute(0,4,2,5,1,3,6) on 7D intermediate
    """
    from vla.model.modules.action_model.qwen3_pi05_expert import get_cache_layer
    import math

    cache.reset_to_position(0)

    sKeyChunk = cache.sKeyChunk   # 16
    sValChunk = cache.sValChunk   # 16
    headDimChunk = cache.headDimChunk  # 16
    nKVHeadChunk = cache.nKVHeadChunk
    nKVHeadVx = cache.nKVHeadVx
    headDimVx = cache.headDimVx

    for layer_idx in range(num_layers):
        prefix_k, prefix_v = get_cache_layer(
            prefix_key_values, prefix_layer_offset + layer_idx
        )
        # prefix_k: [B, num_kv_heads, prefix_len, head_dim]
        prefix_len = prefix_k.shape[2]
        B = prefix_k.shape[0]

        pk = prefix_k.to(dtype=torch.float16, device='rpu').contiguous()
        pv = prefix_v.to(dtype=torch.float16, device='rpu').contiguous()

        # Pad heads to effective_kv_slots (nKVHeadChunk * nKVHeadVx)
        effective_heads = nKVHeadChunk * nKVHeadVx
        num_kv_heads = pk.shape[1]
        if num_kv_heads < effective_heads:
            pad_h = effective_heads - num_kv_heads
            pk = torch.nn.functional.pad(pk, (0, 0, 0, 0, 0, pad_h))
            pv = torch.nn.functional.pad(pv, (0, 0, 0, 0, 0, pad_h))

        # Pad seq_len to multiple of sKeyChunk (16)
        padded_seq = math.ceil(prefix_len / sKeyChunk) * sKeyChunk
        if prefix_len < padded_seq:
            pad_s = padded_seq - prefix_len
            pk = torch.nn.functional.pad(pk, (0, 0, 0, pad_s))
            pv = torch.nn.functional.pad(pv, (0, 0, 0, pad_s))

        sKeyVx_prefix = padded_seq // sKeyChunk
        sValVx_prefix = padded_seq // sValChunk

        # K swizzle: [B, heads, seq, head_dim]
        #   → [B, nKVHeadChunk, nKVHeadVx, sKeyVx, sKeyChunk, headDimVx, headDimChunk]
        #   → permute to [B, sKeyVx, nKVHeadVx, headDimVx, nKVHeadChunk, sKeyChunk, headDimChunk]
        k_reshaped = pk.reshape(B, nKVHeadChunk, nKVHeadVx, sKeyVx_prefix, sKeyChunk, headDimVx, headDimChunk)
        k_swizzled = k_reshaped.permute(0, 3, 2, 5, 1, 4, 6).contiguous()

        # V swizzle: [B, heads, seq, head_dim]
        #   → [B, nKVHeadChunk, nKVHeadVx, sValVx, sValChunk, headDimVx, headDimChunk]
        #   → permute to [B, sValVx, nKVHeadVx, headDimVx, nKVHeadChunk, headDimChunk, sValChunk]
        v_reshaped = pv.reshape(B, nKVHeadChunk, nKVHeadVx, sValVx_prefix, sValChunk, headDimVx, headDimChunk)
        v_swizzled = v_reshaped.permute(0, 3, 2, 5, 1, 6, 4).contiguous()

        # Write into the first sKeyVx_prefix / sValVx_prefix slots of the cache
        cache.k_caches[layer_idx][:, :sKeyVx_prefix, ...] = k_swizzled
        cache.v_caches[layer_idx][:, :sValVx_prefix, ...] = v_swizzled

    # Set position to prefix_len (unpadded) so suffix starts at correct offset
    cache.update_position(prefix_len)


def _prefill_kv_cache(cache, prefix_key_values, prefix_layer_offset, num_layers):
    """Pre-fill prefix KV into RPUCache using the RPU insert_kcache/vcache kernels.

    The insert kernels are THE source of truth for the 7D swizzled DDR cache
    layout that SDPA reads. Using them (instead of a Python-side transcription
    of the swizzle permutations) guarantees the prefix layout is byte-compatible
    with suffix layout — which is ALWAYS written by the same kernels from the
    fused forward path.

    Caveat: the DDR insert kernel validates `k_cache.size(1) == ceil(seq_len/16)`,
    so we cannot write directly into the big max_seq cache. We allocate a
    prefix-sized temporary 7D cache, write prefix KV into it via the kernel,
    then copy the swizzled bytes into the first `sKeyVx_prefix` slots of the
    real cache. The leading-slot layout of a 7D cache is independent of
    `sKeyVx` (only the outer stride changes), so the copy is a plain slice
    assignment.
    """
    from vla.model.modules.action_model.qwen3_pi05_expert import get_cache_layer
    import rpu_backend

    cache.reset_to_position(0)

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

        # insert_kcache/vcache expect [B, seq_len, num_kv_heads, head_dim] (unified layout).
        pk = prefix_k.transpose(1, 2).to(dtype=torch.float16, device='rpu').contiguous()
        pv = prefix_v.transpose(1, 2).to(dtype=torch.float16, device='rpu').contiguous()

        # Kernels require seq_len % 8 == 0 (8 cores).
        if pk.shape[1] % 8 != 0:
            raise RuntimeError(
                f"_prefill_kv_cache: prefix_len={pk.shape[1]} must be a multiple of 8"
            )

        # Pad prefix to a multiple of sKeyChunk (16) for the 7D cache layout.
        import math as _math
        sKeyVx_prefix = _math.ceil(prefix_len_layer / sKeyChunk)
        sValVx_prefix = _math.ceil(prefix_len_layer / sValChunk)
        padded_seq_k = sKeyVx_prefix * sKeyChunk
        padded_seq_v = sValVx_prefix * sValChunk
        if padded_seq_k > prefix_len_layer:
            pk = torch.nn.functional.pad(pk, (0, 0, 0, 0, 0, padded_seq_k - prefix_len_layer))
        if padded_seq_v > prefix_len_layer:
            pv = torch.nn.functional.pad(pv, (0, 0, 0, 0, 0, padded_seq_v - prefix_len_layer))

        # Allocate prefix-sized temporary 7D caches matching the DDR kernel's
        # expected shape for this specific seq_len. Zero-initialized to match
        # RPUCache's zero-fill (padding remains 0).
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

        torch.rpu.insert_kcache(pk, k_tmp, 0)   # Operator is registered on torch.rpu.
        torch.rpu.insert_vcache(pv, v_tmp, 0)

        # Copy swizzled bytes into the first slots of the real cache. Slice
        # along `sKeyVx` (axis 1): the inner 6 dims form the "page" at a given
        # sKeyVx position and are identical across cache sizes.
        cache.k_caches[layer_idx][:, :sKeyVx_prefix, ...].copy_(k_tmp)
        cache.v_caches[layer_idx][:, :sValVx_prefix, ...].copy_(v_tmp)

    # Set position to prefix_len (unpadded) so suffix starts at correct offset
    cache.update_position(prefix_len_layer)


def patch_qwenpi05_for_rpu_fused(expert, *, prefix_len=0, suffix_len=31, max_seq_len=2048, cpu_rotary_emb=None):
    """Patch a QwenPI05ActionExpert to use fused C++ execution.

    Steps:
      1. Preserve the currently published install
      2. Gather weights and pre-compute M-RoPE/KV/GraphCache state
      3. Prepare the replacement forward with prefix KV pre-fill
      4. Configure and tentatively publish a pending handle
      5. Retire the old handle to commit the replacement

    Prerequisites:
      - expert.half() must be called before swizzling weights
      - convert_linear_weights_inplace(expert, skip_names={'cond'}) must have been called
        on the fp16 model; swizzling fp32 then casting to fp16 corrupts the
        layout consumed by the fp16 RPU linear kernels.
      - expert.to('rpu') must have been called after swizzling

    Args:
        expert: A QwenPI05ActionExpert model instance.
        prefix_len: Length of prefix sequence (for KV cache).
        suffix_len: Length of suffix sequence (for KV cache).
        max_seq_len: Maximum sequence length for KV cache allocation.

    Returns:
        int: handle for this model. Also stored as expert._rpu_qwenpi05_handle.
    """
    # ------------------------------------------------------------------ #
    # Step 0: class-level monkey patches for RPU compatibility
    #   - AdaRMSNorm.forward: cast cond to Linear weight dtype/device
    #   - QwenPI05ActionIO.decode_actions: avoid fp32→fp16 Linear on RPU
    # Both patches are idempotent and leave CPU behavior unchanged.
    # ------------------------------------------------------------------ #
    _patch_adarms_norm_for_rpu()
    _patch_action_io_decode_actions_for_rpu()

    if not hasattr(expert, 'layers'):
        raise RuntimeError(
            "patch_qwenpi05_for_rpu_fused: expert has no .layers"
        )
    if not hasattr(expert, 'config'):
        raise RuntimeError(
            "patch_qwenpi05_for_rpu_fused: expert has no .config"
        )
    expert_state = vars(expert)
    install_snapshot = {
        name: expert_state[name]
        for name in _QWENPI05_RUNTIME_INSTALL_ATTRS
        if name in expert_state
    }
    had_old_handle = "_rpu_qwenpi05_handle" in install_snapshot
    old_handle = install_snapshot.get("_rpu_qwenpi05_handle")
    old_finalizer = install_snapshot.get(
        "_rpu_qwenpi05_handle_finalizer")

    import rpu_backend as _rb
    graph_cache = _rb.graph.GraphCache()
    _GraphSignature = _rb.graph.GraphSignature

    # ------------------------------------------------------------------ #
    # Step 3: gather per-layer weights (all 13 lists)
    # ------------------------------------------------------------------ #
    num_layers = len(expert.layers)
    if num_layers == 0:
        raise RuntimeError(
            "patch_qwenpi05_for_rpu_fused: expert has no layers"
        )

    # Optional monotonic timestamps for correlating runtime events.
    import time as _diag_time
    def _dtlog(label):
        print(f"[PY-DIAG] {label} t={_diag_time.monotonic():.6f}")

    _dtlog("before _gather_qwenpi05_weights (incl. _prepare_cond_dense_weights: 10L × 2sites × (w+b) = 40 H2D)")
    weights = _gather_qwenpi05_weights(expert)
    _dtlog("after  _gather_qwenpi05_weights")

    # ------------------------------------------------------------------ #
    # Step 4: M-RoPE cos/sin pre-computation
    # ------------------------------------------------------------------ #
    # Use caller-provided CPU rotary_emb clone (cloned before .to('rpu')),
    # or fall back to cloning from expert (only works if still on CPU).
    from rpu_backend.adapters.qwenpi05.convert import _RPURotaryEmbWrapper
    if cpu_rotary_emb is not None:
        cpu_rotary = _RPURotaryEmbWrapper(cpu_rotary_emb.cpu())
    else:
        # Fallback: try deepcopy (only works if expert is still on CPU)
        rotary_src = expert.qwen_rotary_emb
        if hasattr(rotary_src, '_cpu_emb'):
            rotary_src = rotary_src._cpu_emb
        cpu_rotary = _RPURotaryEmbWrapper(copy.deepcopy(rotary_src).cpu())
    _dtlog("before _precompute_mrope_cos_sin (CPU-only)")
    cos_cached, sin_cached = _precompute_mrope_cos_sin(cpu_rotary, prefix_len, suffix_len)
    _dtlog("after  _precompute_mrope_cos_sin")

    # Create KV cache (RPUCache for correct 7D swizzled format)
    _dtlog("before _create_kv_cache  (EXPECT: 10L × 2 torch.zeros(device=rpu) = 20 DMAs)")
    kv_cache = _create_kv_cache(expert.config, max_seq_len)
    _dtlog("after  _create_kv_cache")

    print(
        f"[RPU] QwenPI05 M-RoPE pre-computed: cos/sin [{suffix_len}, {cos_cached.shape[-1]}], "
        f"KV cache max_seq={max_seq_len}"
    )

    # ------------------------------------------------------------------ #
    # Step 5: prepare set_weights args
    # ------------------------------------------------------------------ #
    config = expert.config
    set_weights_args = (
        *weights,  # 13 lists: q,k,v,o,gate,up,down, attn_dw,attn_db, mlp_dw,mlp_db, q_norm,k_norm
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.width,           # hidden_size
        config.mlp_dim,         # intermediate_size
        config.rms_norm_eps,
    )

    # ------------------------------------------------------------------ #
    # Step 5b: move final AdaRMSNorm back to CPU.
    #   Rationale: the final `expert.norm` is an AdaRMSNorm that runs AFTER
    #   the fused C++ layer stack. Its `cond` Linear is a normal nn.Linear
    #   with un-swizzled weights (we pass `skip_names={'cond'}` to
    #   convert_linear_weights_inplace). When `.to('rpu')` is applied to the
    #   whole expert, the norm's Linear goes to RPU in a layout that the
    #   RPU `rpu_linear` op cannot consume (no column/row partition swizzle),
    #   producing garbage output.
    #
    #   Keep the final norm on CPU for Python-side computation. The
    #   hidden-state DMA back to CPU in `rpu_qwenpi05_fused_forward` already
    #   materializes the tensor on CPU just before the final norm — CPU
    #   execution happens after the transformer stack.
    # ------------------------------------------------------------------ #
    if getattr(expert, 'norm', None) is not None:
        expert.norm = expert.norm.to('cpu').to(dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Step 6: forward replacement
    # ------------------------------------------------------------------ #
    def rpu_qwenpi05_fused_forward(
        self,
        suffix_embeds,
        *,
        prefix_key_values=None,
        prefix_mask=None,
        suffix_mask=None,
        position_ids=None,
        adarms_cond=None,
        prefix_layer_offset=0,
    ):
        """Fused forward: routes through C++ qwenpi05_forward op."""
        from vla.model.modules.action_model.qwen3_pi05_expert import (
            make_suffix_attn_mask, bool_mask_to_attention_bias,
        )

        # -- Device capture for restoration --
        orig_device = suffix_embeds.device

        # -- Route hidden_states to RPU fp16 --
        hidden_states = suffix_embeds
        if hidden_states.device.type != 'rpu':
            hidden_states = hidden_states.to(dtype=torch.float16, device='rpu')
        if hidden_states.dtype != torch.float16:
            hidden_states = hidden_states.to(torch.float16)

        # -- Pad suffix to target suffix_len for RPU SDPA alignment --
        actual_suffix_len = hidden_states.shape[1]
        target_suffix_len = self._rpu_cos_cached.shape[0]  # from M-RoPE pre-compute
        if actual_suffix_len > target_suffix_len:
            raise RuntimeError(
                f"QwenPI05 fused: runtime suffix_len {actual_suffix_len} exceeds "
                f"the precomputed cos/sin table length {target_suffix_len}; "
                "re-patch with a larger suffix_len."
            )
        if actual_suffix_len < target_suffix_len:
            pad_len = target_suffix_len - actual_suffix_len
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_len))
            # Also pad suffix_mask
            if suffix_mask is not None:
                suffix_mask = torch.nn.functional.pad(
                    suffix_mask, (0, pad_len), value=False
                )
        hidden_states = hidden_states.contiguous()

        if hidden_states.shape[0] != 1:
            raise AssertionError(
                f"QwenPI05 fused: batch_size must be 1, got {hidden_states.shape[0]}"
            )

        # -- Normalize cond to [hidden_size] fp16 RPU contiguous --
        hidden_size = self.config.width
        if adarms_cond is None:
            raise AssertionError("QwenPI05 fused: adarms_cond is required")
        cond_rpu = adarms_cond.to(dtype=torch.float16, device='rpu')
        if cond_rpu.dim() == 2 and cond_rpu.shape[0] == 1:
            cond_rpu = cond_rpu.squeeze(0)
        if cond_rpu.dim() != 1 or cond_rpu.shape[0] != hidden_size:
            raise AssertionError(
                f"QwenPI05 fused: cond must be [{hidden_size}] or [1,{hidden_size}], "
                f"got {tuple(cond_rpu.shape)}"
            )
        cond_rpu = cond_rpu.contiguous()

        # -- KV cache: get or create --
        cache = self._rpu_kv_cache

        # -- Prefix KV pre-fill (first denoising step or after cache reset) --
        if prefix_key_values is not None and cache.position == 0:
            _prefill_kv_cache(
                cache, prefix_key_values, prefix_layer_offset,
                len(self.layers)
            )
        prefix_len_actual = cache.position if prefix_key_values is not None else 0

        # -- Attention mask --
        if prefix_mask is not None and suffix_mask is not None:
            bool_mask = make_suffix_attn_mask(prefix_mask, suffix_mask)
            attention_mask = bool_mask_to_attention_bias(bool_mask, torch.float16)
            attention_mask = attention_mask.cpu().contiguous()
            is_causal = False
        else:
            attention_mask = None
            is_causal = True

        # -- Fused C++ forward --
        sig = _GraphSignature(
            op_id="qwenpi05_forward",
            shapes=[int(hidden_states.shape[1]), int(hidden_states.shape[2])],
            dyn_dims=[
                int(len(self.layers)),
                int(self.config.num_attention_heads),
                int(self.config.num_key_value_heads),
                int(self.config.head_dim),
                int(prefix_len_actual),
                int(is_causal),
                1 if attention_mask is not None else 0,
            ],
            dtypes=[torch.float16],
        )
        try:
            with self._rpu_qwenpi05_graph_cache.capture(sig):
                output = torch.ops.rpu.qwenpi05_forward(
                    self._rpu_qwenpi05_handle,
                    hidden_states,
                    cond_rpu,
                    cache.k_caches,
                    cache.v_caches,
                    self._rpu_cos_cached,
                    self._rpu_sin_cached,
                    attention_mask,
                    prefix_len_actual,   # position: suffix starts after prefix
                    is_causal,
                )
        finally:
            # Always release temporary SPM allocations, including on errors.
            torch.ops.rpu.spm_alloc_reset_temporary()

        # -- Strip suffix padding before returning --
        if actual_suffix_len < target_suffix_len:
            output = output[:, :actual_suffix_len, :]

        # -- Final norm in Python: norm is on CPU fp32 to match the CPU path.
        #    Move output to CPU + cast to fp32 first.
        if getattr(self, "norm", None) is not None:
            norm_w = getattr(self.norm.norm, 'weight', None)
            norm_device = norm_w.device if norm_w is not None else torch.device('cpu')
            norm_dtype = norm_w.dtype if norm_w is not None else torch.float32
            output_for_norm = output.to(device=norm_device, dtype=norm_dtype)
            output, _ = self.norm(output_for_norm, adarms_cond)

        # -- Restore device if caller expected non-cpu device --
        if orig_device.type != output.device.type:
            output = output.to(orig_device)

        return output

    handle = torch.ops.rpu.qwenpi05_create()
    handle_finalizer = None
    committed = False
    try:
        handle_finalizer = weakref.finalize(
            expert, _qwenpi05_destroy_handle, h=handle)
        _dtlog("before qwenpi05_set_weights  (C++ side: only stores at::Tensor refs -- should be 0 DMA)")
        torch.ops.rpu.qwenpi05_set_weights(handle, *set_weights_args)
        _dtlog("after  qwenpi05_set_weights")

        expert._rpu_qwenpi05_graph_cache = graph_cache
        expert._rpu_cos_cached = cos_cached
        expert._rpu_sin_cached = sin_cached
        expert._rpu_kv_cache = kv_cache
        expert._rpu_qwenpi05_handle = handle
        expert._rpu_qwenpi05_handle_finalizer = handle_finalizer
        expert.forward = types.MethodType(
            rpu_qwenpi05_fused_forward, expert)

        if (
            had_old_handle
            and (
                old_finalizer is None
                or getattr(old_finalizer, "alive", False)
            )
        ):
            torch.ops.rpu.qwenpi05_destroy(old_handle)
        committed = True
    except BaseException:
        if not committed:
            expert_state = vars(expert)
            for name in _QWENPI05_RUNTIME_INSTALL_ATTRS:
                expert_state.pop(name, None)
            expert_state.update(install_snapshot)

            if handle_finalizer is None:
                _qwenpi05_destroy_handle(handle)
            elif handle_finalizer.alive:
                handle_finalizer()
        raise

    if old_finalizer is not None and getattr(old_finalizer, "alive", False):
        old_finalizer.detach()
    print(
        f"[RPU] Patched QwenPI05ActionExpert with fused forward, "
        f"handle={handle}, layers={len(expert.layers)}"
    )

    return handle
