"""Pi0.5 AdaRMS all-layers-once instance patch.

``Pi05Adapter`` installs this flat helper module lazily from ``to_rpu``.
"""
from __future__ import annotations
import types
import weakref

import torch
import torch.nn as nn

import rpu_backend  # GraphCache / GraphSignature
from rpu_backend.runtime import rpu_env_bool
from rpu_backend.runtime.log import _LOG
from rpu_backend.runtime.weights import (
    transform_linear_weight, convert_linear_weights_inplace,
    NUM_CORES, tp_row_swizzle_mc_weight, tp_col_swizzle_mc_weight,
)
from rpu_backend.api.cache import RPUCache


def _adarms_graph_enabled() -> bool:
    return rpu_env_bool("RPU_PI05_ADARMS_GRAPH", default=True)


def _adarms_w8a16_graph_enabled() -> bool:
    # W8A16 AdaRMS graph capture avoids the "enqueu_kernel called after
    # build_batch" warnings emitted by PASSTHROUGH: the preceding
    # gemma_prefill graph build leaves batch_built_=true on the shared
    # Queue_t, so an uncaptured AdaRMS layer cannot be enqueued on that queue.
    # Default ON; set RPU_PI05_ADARMS_W8A16_GRAPH=0 to force PASSTHROUGH for
    # debugging.
    return rpu_env_bool("RPU_PI05_ADARMS_W8A16_GRAPH", default=True)


# AdaRMS all-layers-once C++ handle lifecycle helper.
def _adarms_destroy_handle(h):
    """Release C++ AdaRMSModel handle. Called by weakref.finalize on GC."""
    try:
        torch.ops.rpu.adarms_destroy(h)
    except Exception:
        # During interpreter shutdown the op may already be gone.
        pass


_ADARMS_RUNTIME_INSTALL_ATTRS = (
    "_gemma_rope_cos",
    "_gemma_rope_sin",
    "_rpu_cache",
    "_rpu_adarms_graph_cache",
    "_rpu_lazy_init_checked",
    "_rpu_required_attrs",
    "_rpu_action_handle",
    "_rpu_action_handle_finalizer",
    "forward",
)


def _prepare_adarms_dense_weights(model):
    """Swizzle per-layer AdaRMS dense weights for RPU row-partition GEMM.

    For each layer x each of {input_layernorm, post_attention_layernorm}:
      [3H, H] -> transform_linear_weight(partition=0, num_cores=8) -> row-partition swizzle
      -> .to('rpu')

    The C++ `AdaRMSModel::adarms_gemv_to_spm` runs row-partition GEMM at M=1
    with `force_gemm=true`: each core holds local_k=H/NUM_CORES columns of the
    full [3H, H] weight and produces a partial [3H]; a fused
    all_reduce_sum_residual folds in the bias and broadcasts the full [3H]
    back to every core.

    Returns four parallel lists (one entry per layer):
      attn_dense_w_list, attn_dense_b_list, mlp_dense_w_list, mlp_dense_b_list

    The helper is self-contained: if `norm._rpu_dense_w_rp` is already
    registered, it is
    reused; otherwise the weight is recomputed from the live `norm.dense`
    Linear. `_rpu_dense_w_rp` distinguishes row-partitioned data from any
    incompatible `_rpu_dense_w` buffer.
    """
    attn_w, attn_b, mlp_w, mlp_b = [], [], [], []

    for layer_idx, layer in enumerate(model.layers):
        for src_norm_name, (w_out, b_out) in (
            ('input_layernorm', (attn_w, attn_b)),
            ('post_attention_layernorm', (mlp_w, mlp_b)),
        ):
            norm = getattr(layer, src_norm_name)
            if not hasattr(norm, 'dense') or norm.dense is None:
                raise RuntimeError(
                    f"patch_adarms_model_for_rpu_all_layers_once: "
                    f"layer {layer_idx}.{src_norm_name} has no .dense -- is this "
                    f"an AdaRMS norm? (Expected PiGemmaRMSNorm with AdaRMS.)"
                )

            # If pi05_converter already prepared these buffers, reuse them.
            if hasattr(norm, '_rpu_dense_w_rp') and hasattr(norm, '_rpu_dense_b_rp'):
                w_out.append(norm._rpu_dense_w_rp)
                b_out.append(norm._rpu_dense_b_rp)
                continue

            # Self-contained path: compute from the live dense Linear.
            # Force CPU for the swizzle pipeline so this works whether or not
            # `expert.to("rpu")` has already been applied (the production path
            # calls .to("rpu") before the patch; some test paths may not).
            orig_w = norm.dense.weight.detach().cpu().to(dtype=torch.float16).contiguous()  # CPU [3H, H]
            transformed_w = transform_linear_weight(
                orig_w, partition=0, num_cores=NUM_CORES
            ).to('rpu').contiguous()
            if norm.dense.bias is None:
                raise RuntimeError(
                    f"patch_adarms_model_for_rpu_all_layers_once: "
                    f"layer {layer_idx}.{src_norm_name}.dense has no bias -- "
                    f"AdaRMS dense must include a bias vector."
                )
            orig_b = norm.dense.bias.detach().cpu().to(dtype=torch.float16).to('rpu').contiguous()

            # Cache on the module for subsequent re-patches + inspection
            # (version-bumped name `_rpu_dense_w_rp` -- `_rp` = row partition --
            # to avoid colliding with stale col-partition-replicated `_rpu_dense_w`
            # buffers left over from older converter versions).
            norm.register_buffer('_rpu_dense_w_rp', transformed_w)
            norm.register_buffer('_rpu_dense_b_rp', orig_b)

            w_out.append(transformed_w)
            b_out.append(orig_b)

    return attn_w, attn_b, mlp_w, mlp_b


# AdaRMS all-layers-once instance patch.
def patch_adarms_model_for_rpu_all_layers_once(model) -> int:
    """
    Patch a Pi0.5 Action Expert (Gemma + AdaRMS) instance to use all-layers-once
    C++ execution via AdaRMSModel (FusedModelBase).

    Steps:
      1. Preserve the published install while preparing replacement state
      2. Gather per-layer attention / MLP / AdaRMS dense weights
      3. Extract cos/sin from rotary_emb on RPU device (kernel format)
      4. Prepare cache and replacement forward state
      5. Configure and tentatively publish a pending handle, then retire old
      6. Replace model.forward with a method that:
         - validates preconditions (use_cache, batch, adarms_cond, devices)
         - resolves inputs_embeds / input_ids
         - normalizes adarms_cond to [hidden_size] fp16 RPU contiguous
         - selects RPUCache when none is supplied
         - computes the is_causal contract
         - calls torch.ops.rpu.adarms_forward(handle, ...)
         - calls torch.ops.rpu.spm_alloc_reset_temporary() at exit
         - returns BaseModelOutputWithPast without applying final_norm in C++

    Prerequisites:
      - convert_linear_weights_inplace(model) must have been called
      - model.to("rpu") must have been called
      - model.norm may stay on CPU (PiGemmaRMSNorm runs in Python in fp32)

    **Idempotent**: a failed replacement preserves the currently published
    `_rpu_action_handle` and its forward/cache/RoPE state.

    Args:
        model: A Pi0.5 Gemma Action Expert model instance (base decoder, not
               the outer conditional generation wrapper).

    Returns:
        int: handle for this model. Also stored as model._rpu_action_handle.
    """
    try:
        from transformers.modeling_outputs import BaseModelOutputWithPast
    except ImportError as e:
        raise RuntimeError(
            "patch_adarms_model_for_rpu_all_layers_once requires transformers "
            "with BaseModelOutputWithPast"
        ) from e

    model_state = vars(model)
    install_snapshot = {
        name: model_state[name]
        for name in _ADARMS_RUNTIME_INSTALL_ATTRS
        if name in model_state
    }
    had_old_handle = "_rpu_action_handle" in install_snapshot
    old_handle = install_snapshot.get("_rpu_action_handle")
    old_finalizer = install_snapshot.get(
        "_rpu_action_handle_finalizer")

    # ------------------------------------------------------------------ #
    # Step 1: gather per-layer weights
    # ------------------------------------------------------------------ #
    if not hasattr(model, 'layers'):
        raise RuntimeError(
            "patch_adarms_model_for_rpu_all_layers_once: model has no .layers"
        )
    if not hasattr(model, 'config'):
        raise RuntimeError(
            "patch_adarms_model_for_rpu_all_layers_once: model has no .config"
        )
    num_layers = len(model.layers)
    if num_layers == 0:
        raise RuntimeError(
            "patch_adarms_model_for_rpu_all_layers_once: model has no layers"
        )

    q_w_list    = [model.layers[i].self_attn.q_proj.weight for i in range(num_layers)]
    k_w_list    = [model.layers[i].self_attn.k_proj.weight for i in range(num_layers)]
    v_w_list    = [model.layers[i].self_attn.v_proj.weight for i in range(num_layers)]
    o_w_list    = [model.layers[i].self_attn.o_proj.weight for i in range(num_layers)]
    gate_w_list = [model.layers[i].mlp.gate_proj.weight    for i in range(num_layers)]
    up_w_list   = [model.layers[i].mlp.up_proj.weight      for i in range(num_layers)]
    down_w_list = [model.layers[i].mlp.down_proj.weight    for i in range(num_layers)]

    attn_dense_w_list, attn_dense_b_list, mlp_dense_w_list, mlp_dense_b_list = \
        _prepare_adarms_dense_weights(model)

    # ------------------------------------------------------------------ #
    # Step 2: cos/sin on RPU (kernel format [max_pos, head_dim/2])
    # ------------------------------------------------------------------ #
    config = model.config
    head_dim = config.head_dim
    num_q_heads = config.num_attention_heads
    num_kv_heads = int(getattr(
        model, "_rpu_effective_num_kv_heads", config.num_key_value_heads
    ))
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    eps = config.rms_norm_eps

    rotary = getattr(model, "rotary_emb", None)
    if rotary is None:
        rotary = getattr(model.layers[0].self_attn, "rotary_emb", None)
    if rotary is None:
        raise RuntimeError(
            "patch_adarms_model_for_rpu_all_layers_once: could not locate "
            "rotary embedding module (tried model.rotary_emb and "
            "layers[0].self_attn.rotary_emb)"
        )

    max_pos = getattr(rotary, "max_position_embeddings", 4096)
    dummy = torch.zeros(1, 1, hidden_size, dtype=torch.float16, device="rpu")
    dummy_pos = torch.arange(max_pos, device="rpu").unsqueeze(0)
    cos_cached, sin_cached = rotary(dummy, dummy_pos)
    if cos_cached.dim() == 3:
        cos_cached = cos_cached.squeeze(0)
        sin_cached = sin_cached.squeeze(0)
    cos_cached = cos_cached.to(dtype=torch.float16, device="rpu").contiguous()
    sin_cached = sin_cached.to(dtype=torch.float16, device="rpu").contiguous()

    # ------------------------------------------------------------------ #
    # Step 3: prepare set_weights arguments (no final_norm)
    # ------------------------------------------------------------------ #
    from rpu_backend.adapters.pi05.w8a16 import pi05_expert_scale_lists
    scale_lists = pi05_expert_scale_lists(model, require_rpu=True)
    adarms_w8a16 = bool(scale_lists[0])
    set_weights_args = (
        q_w_list, k_w_list, v_w_list, o_w_list,
        gate_w_list, up_w_list, down_w_list,
        attn_dense_w_list, attn_dense_b_list,
        mlp_dense_w_list,  mlp_dense_b_list,
        cos_cached, sin_cached,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size, eps,
        *scale_lists,
    )

    # ------------------------------------------------------------------ #
    # Step 6: replace forward
    #
    # Capture config in the closure for the GraphCache signature. The signature
    # omits position because all position-derived kernel regs go
    # through set_regs + add_kernel_mutable, and the only baked DMA
    # (sdpa_dma_mask_to_spm) uses a shape-keyed stable slot. Mutable parameters
    # are synchronized before a same-signature replay at a new position.
    # ------------------------------------------------------------------ #
    _adarms_sig_hidden_size = int(hidden_size)
    _adarms_sig_num_layers = int(num_layers)
    _adarms_sig_num_q_heads = int(num_q_heads)
    _adarms_sig_w8a16 = bool(adarms_w8a16)

    def rpu_adarms_model_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        adarms_cond=None,
        **kwargs,
    ):
        # -- Preconditions before the native op --
        if output_attentions:
            raise AssertionError(
                "AdaRMS (RPU all-layers-once): output_attentions is not supported"
            )
        if output_hidden_states:
            raise AssertionError(
                "AdaRMS (RPU all-layers-once): output_hidden_states is not supported"
            )
        # use_cache=False is valid for Pi0.5 denoise_step. Cache reset and
        # `_prefix_len` publication are conditional, while the fused op still
        # uses RPUCache for SDPA reads and writes. The type guard below is the
        # correctness check.
        if adarms_cond is None:
            raise AssertionError(
                "AdaRMS (RPU all-layers-once): adarms_cond is required"
            )

        # -- Input resolution (mutual exclusion) --
        if input_ids is not None and inputs_embeds is not None:
            raise AssertionError(
                "AdaRMS: must provide exactly one of input_ids / inputs_embeds"
            )
        if inputs_embeds is None:
            if input_ids is None:
                raise AssertionError(
                    "AdaRMS: must provide exactly one of input_ids / inputs_embeds"
                )
            if self.embed_tokens is None:
                raise AssertionError(
                    "AdaRMS: input_ids given but model.embed_tokens is None "
                    "(Action Expert typically uses inputs_embeds, not ids)"
                )
            inputs_embeds = self.embed_tokens(input_ids).to(
                dtype=torch.float16, device='rpu')

        hidden_states = inputs_embeds
        # Capture the caller's original device so it can be restored before the
        # final norm.
        # Pi0.5 denoise_step delivers suffix_embs on CPU; downstream code
        # needs the suffix hidden on CPU so action_out_proj's CPU weights
        # dispatch correctly and v_t lands beside x_t for `x_t + dt * v_t`.
        orig_device = hidden_states.device
        # Auto-route CPU inputs to the RPU in fp16.
        if hidden_states.device.type != 'rpu':
            hidden_states = hidden_states.to(dtype=torch.float16, device='rpu')
        if hidden_states.dtype != torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        if hidden_states.shape[0] != 1:
            raise AssertionError(
                f"AdaRMS: batch_size must be 1, got {hidden_states.shape[0]}"
            )
        seq_len = hidden_states.shape[1]

        # -- cond normalization ([hidden_size] contiguous FP16 on RPU) --
        cond_rpu = adarms_cond.to(dtype=torch.float16, device='rpu')
        if cond_rpu.dim() == 2 and cond_rpu.shape[0] == 1:
            cond_rpu = cond_rpu.squeeze(0)
        if cond_rpu.dim() != 1 or cond_rpu.shape[0] != hidden_size:
            raise AssertionError(
                f"AdaRMS: cond must be [hidden_size={hidden_size}] or "
                f"[1, hidden_size], got {tuple(cond_rpu.shape)}"
            )
        cond_rpu = cond_rpu.contiguous()

        # -- RPUCache auto-creation --
        if past_key_values is None:
            # `_rpu_cache` is eagerly initialized during patch installation.
            if use_cache:
                self._rpu_cache.reset_to_position(0)
            past_key_values = self._rpu_cache

        if not isinstance(past_key_values, RPUCache):
            raise AssertionError(
                f"AdaRMS: past_key_values must be RPUCache, got "
                f"{type(past_key_values).__name__}"
            )

        # -- is_causal contract + mask normalization --
        is_causal = (attention_mask is None)
        if attention_mask is not None:
            # Cache the converted mask keyed on input Python identity. The
            # Pi0.5 denoise loop builds the mask once outside the loop, so
            # subsequent steps pass the
            # SAME Python object → cache hit → reuse the converted tensor →
            # downstream C++ AdaRMSModel cache also hits (same data_ptr).
            # Avoid `_rpu_*` prefix — Validated_PiGemmaModel rejects unlisted
            # `_rpu_*` setattr at runtime. Use private adapter prefix instead.
            cached_in  = getattr(self, "__adarms_mask_in",  None)
            cached_out = getattr(self, "__adarms_mask_out", None)
            if cached_in is attention_mask and cached_out is not None:
                attention_mask = cached_out
            else:
                converted = attention_mask.to(
                    dtype=torch.float16).cpu().contiguous()
                object.__setattr__(self, "__adarms_mask_in",  attention_mask)
                object.__setattr__(self, "__adarms_mask_out", converted)
                attention_mask = converted

        # Wrap the fused C++ op in GraphCache.capture(sig).
        # sig keys on `position` (== prefix_len here), which is what the SDPA
        # kv length and the [1,1,suffix,prefix+suffix] 2D mask both derive
        # from. It is NOT a per-step split: the denoise loop calls
        # reset_to_position(prefix_len) before every step (runtime.py), so all
        # steps of one profile share a position and therefore one cache entry.
        # Pi0.5 denoise passes a 2D mask, so the prefix position must remain in
        # the signature.
        # RoPE start for the suffix. The expert's `position` is the PHYSICAL
        # cache row (the denoise loop resets the shared cache to
        # prefix_pad_masks.shape[1]); its RoPE position is the LOGICAL one the
        # reference uses, sum(prefix_pad_masks). They differ by the prefix's pad
            # rows. -1 keeps physical-row behavior for callers that pass no
        # position_ids. Baked at graph BUILD, hence also in the signature below.
        _rope_pos = -1
        if position_ids is not None:
            _p = position_ids[0] if position_ids.dim() == 2 else position_ids
            _rope_pos = int(_p.reshape(-1)[0])
        torch.ops.rpu.adarms_set_rope_position(
            self._rpu_action_handle, _rope_pos)

        _sig = rpu_backend.graph.GraphSignature(
            op_id="pi05_adarms",
            shapes=[int(seq_len), _adarms_sig_hidden_size],
            dyn_dims=[_adarms_sig_num_layers, _adarms_sig_num_q_heads,
                      int(past_key_values.position), _rope_pos],
            dtypes=[torch.float16],
        )
        # -- Call the fused C++ op (schema: handle, hidden, cond, k_caches,
        #    v_caches, attention_mask, position, is_causal) --
        if _adarms_graph_enabled() and (not _adarms_sig_w8a16 or _adarms_w8a16_graph_enabled()):
            with self._rpu_adarms_graph_cache.capture(_sig):
                output = torch.ops.rpu.adarms_forward(
                    self._rpu_action_handle,
                    hidden_states,
                    cond_rpu,
                    past_key_values.k_caches,
                    past_key_values.v_caches,
                    attention_mask,
                    past_key_values.position,
                    is_causal,
                )
        else:
            output = torch.ops.rpu.adarms_forward(
                self._rpu_action_handle,
                hidden_states,
                cond_rpu,
                past_key_values.k_caches,
                past_key_values.v_caches,
                attention_mask,
                past_key_values.position,
                is_causal,
            )

        past_key_values.update_position(seq_len)

        # Subsystem boundary: release temporary SPM for the next subsystem.
        # In Pi0.5 this is the next denoise step re-entering Gemma VLM -> AdaRMS.
        # This must stay outside the capture scope: graph-aware temporary-SPM
        # reset marks a graph non-replayable.
        torch.ops.rpu.spm_alloc_reset_temporary()

        if use_cache and isinstance(past_key_values, RPUCache):
            past_key_values._prefix_len = past_key_values.position

        # Restore the caller's original device before the final norm. Pi0.5
        # denoise_step supplies suffix_embs on CPU and expects the suffix
        # output on CPU so that `x_t = x_t + dt * v_t` stays on CPU, matching the noise tensor
        # and action_out_proj's CPU weights).
        if orig_device.type != 'rpu':
            output = output.to(orig_device)

        # Final PiGemmaRMSNorm stays in this Python wrapper rather than the
        # decoder-layer graph, so `last_hidden_state` is post-norm for the
        # suffix-only consumer.
        if getattr(self, "norm", None) is not None:
            output, _ = self.norm(output, adarms_cond)

        return BaseModelOutputWithPast(
            last_hidden_state=output,
            past_key_values=past_key_values if use_cache is not False else None,
        )

    # Eagerly initialize the cache, stamp the installed state, and freeze it.
    _effective_num_kv_heads = int(getattr(
        model, "_rpu_effective_num_kv_heads", config.num_key_value_heads
    ))
    _attn_tp_default = int(
        getattr(model, '_rpu_attn_tp',
                min(8, _effective_num_kv_heads))
    )
    _max_seq = int(getattr(model, '_rpu_max_seq_len', 2048))
    rpu_cache = RPUCache(
        num_layers=int(config.num_hidden_layers),
        batch_size=1,
        max_seq_len=_max_seq,
        num_kv_heads=_effective_num_kv_heads,
        head_dim=int(config.head_dim),
        attn_tp=_attn_tp_default,
    )
    # Per-instance GraphCache for the AdaRMS forward wrap. Eager initialization
    # mirrors `_rpu_cache` to keep Dynamo guard-set stable across calls
    # because lazy initialization would invalidate the frontend cache on first use.
    adarms_graph_cache = rpu_backend.graph.GraphCache()
    required_attrs = (
        '_rpu_cache',
        '_rpu_action_handle',
        '_rpu_adarms_graph_cache',
    )

    handle = torch.ops.rpu.adarms_create()
    handle_finalizer = None
    committed = False
    try:
        handle_finalizer = weakref.finalize(
            model, _adarms_destroy_handle, h=handle)
        torch.ops.rpu.adarms_set_weights(handle, *set_weights_args)

        # Stash cos/sin on the model so
        # _install_fused_denoise_handle can reuse the already-allocated RPU
        # tensors without a second allocation.
        model._gemma_rope_cos = cos_cached
        model._gemma_rope_sin = sin_cached
        model._rpu_cache = rpu_cache
        model._rpu_adarms_graph_cache = adarms_graph_cache
        model._rpu_lazy_init_checked = True
        model._rpu_required_attrs = required_attrs
        model._rpu_action_handle = handle
        model._rpu_action_handle_finalizer = handle_finalizer
        model.forward = types.MethodType(rpu_adarms_model_forward, model)

        from rpu_backend.graph.lazy_init_guard import _verify_lazy_init
        _verify_lazy_init(model)

        if (
            had_old_handle
            and (
                old_finalizer is None
                or getattr(old_finalizer, "alive", False)
            )
        ):
            torch.ops.rpu.adarms_destroy(old_handle)
        committed = True
    except BaseException:
        if not committed:
            model_state = vars(model)
            for name in _ADARMS_RUNTIME_INSTALL_ATTRS:
                model_state.pop(name, None)
            model_state.update(install_snapshot)

            if handle_finalizer is None:
                _adarms_destroy_handle(handle)
            elif handle_finalizer.alive:
                handle_finalizer()
        raise

    if old_finalizer is not None and getattr(old_finalizer, "alive", False):
        old_finalizer.detach()

    graph_note = ""
    if adarms_w8a16:
        graph_note = (
            ", W8A16 graph enabled"
            if _adarms_w8a16_graph_enabled()
            else ", W8A16 graph disabled by RPU_PI05_ADARMS_W8A16_GRAPH=0")
    _LOG.info("Patched AdaRMSModel (instance) with all-layers-once fused forward, "
              "handle=%d, num_layers=%d%s", handle, num_layers,
              graph_note)

    return handle
