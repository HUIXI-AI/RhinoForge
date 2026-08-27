"""Pi0.5 Gemma all-layers-once instance patch.

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


def _gemma_graph_enabled() -> bool:
    return rpu_env_bool("RPU_PI05_GEMMA_GRAPH", default=True)


# Native Gemma handle lifecycle helper.
def _gemma_destroy_handle(h):
    """Release C++ GemmaModel handle. Called by weakref.finalize on GC."""
    try:
        torch.ops.rpu.gemma_destroy(h)
    except Exception:
        # Swallow -- during interpreter shutdown the op may be gone
        pass


_GEMMA_RUNTIME_INSTALL_ATTRS = (
    "_rpu_cache",
    "_rpu_gemma_graph_cache",
    "_rpu_lazy_init_checked",
    "_rpu_required_attrs",
    "_rpu_vlm_decoder_handle",
    "_rpu_vlm_decoder_handle_finalizer",
    "forward",
)


# Install the all-layers-once runtime on one Gemma model instance.
def patch_gemma_model_for_rpu_all_layers_once(model) -> int:
    """
    Patch a specific Gemma/PiGemma model instance to use all-layers-once C++
    execution via GemmaModel (FusedModelBase).

    This function prepares all Python-side state before creating a pending
    C++ GemmaModel handle. The pending handle is configured, Python state is
    tentatively published and frozen, and old-handle retirement commits it.

    Differences from the Qwen3 all-layers-once patch:
      - No QK head norms (Gemma does not have q_norm / k_norm)
      - Gemma uses GELU activation (not SiLU)
      - Supports bidirectional attention via attention_mask (is_causal contract)
      - RPUCache is selected automatically when past_key_values is None

    **Idempotent**: a failed replacement preserves the currently published
    ``_rpu_vlm_decoder_handle`` and its forward/cache state.

    Prerequisites:
      - model.to("rpu") must have been called (weights on RPU device)
      - convert_linear_weights_inplace(model) must have been called

    Args:
        model: A PiGemmaModel / GemmaModel instance (base decoder, not the
               outer conditional generation wrapper).

    Returns:
        int: The handle for this model. Also stored as model._rpu_vlm_decoder_handle.
    """
    try:
        from transformers.modeling_outputs import BaseModelOutputWithPast
    except ImportError as e:
        raise RuntimeError(
            "patch_gemma_model_for_rpu_all_layers_once requires transformers "
            "with BaseModelOutputWithPast"
        ) from e

    model_state = vars(model)
    install_snapshot = {
        name: model_state[name]
        for name in _GEMMA_RUNTIME_INSTALL_ATTRS
        if name in model_state
    }
    had_old_handle = "_rpu_vlm_decoder_handle" in install_snapshot
    old_handle = install_snapshot.get("_rpu_vlm_decoder_handle")
    old_finalizer = install_snapshot.get(
        "_rpu_vlm_decoder_handle_finalizer")

    # ------------------------------------------------------------------ #
    # 2. Gather per-layer weights + global params
    # ------------------------------------------------------------------ #
    if not hasattr(model, "layers"):
        raise RuntimeError(
            "patch_gemma_model_for_rpu_all_layers_once: model has no .layers "
            "attribute"
        )
    if not hasattr(model, "config"):
        raise RuntimeError(
            "patch_gemma_model_for_rpu_all_layers_once: model has no .config "
            "attribute"
        )
    num_layers = len(model.layers)
    if num_layers == 0:
        raise RuntimeError(
            "patch_gemma_model_for_rpu_all_layers_once: model has no layers"
        )

    q_w_list = [model.layers[i].self_attn.q_proj.weight for i in range(num_layers)]
    k_w_list = [model.layers[i].self_attn.k_proj.weight for i in range(num_layers)]
    v_w_list = [model.layers[i].self_attn.v_proj.weight for i in range(num_layers)]
    o_w_list = [model.layers[i].self_attn.o_proj.weight for i in range(num_layers)]
    input_norm_list = [model.layers[i].input_layernorm.weight for i in range(num_layers)]
    post_norm_list = [model.layers[i].post_attention_layernorm.weight for i in range(num_layers)]
    gate_list = [model.layers[i].mlp.gate_proj.weight for i in range(num_layers)]
    up_list = [model.layers[i].mlp.up_proj.weight for i in range(num_layers)]
    down_list = [model.layers[i].mlp.down_proj.weight for i in range(num_layers)]
    from rpu_backend.adapters.pi05.w8a16 import gemma_decoder_scale_lists
    scale_lists = gemma_decoder_scale_lists(model, require_rpu=True, label="vlm")
    gemma_w8a16 = bool(scale_lists[0])

    # ------------------------------------------------------------------ #
    # 3. Derive model dimension params from config
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

    # ------------------------------------------------------------------ #
    # 4. cos/sin extraction from rotary embedding
    # ------------------------------------------------------------------ #
    rotary = getattr(model, "rotary_emb", None)
    if rotary is None:
        rotary = getattr(model.layers[0].self_attn, "rotary_emb", None)
    if rotary is None:
        raise RuntimeError(
            "patch_gemma_model_for_rpu_all_layers_once: could not locate "
            "rotary embedding module (tried model.rotary_emb and "
            "layers[0].self_attn.rotary_emb)"
        )

    # Always compute cos/sin through the patched rotary forward on RPU device
    # to get kernel format [max_pos, head_dim/2]. The patched rotary returns
    # [head_dim] on CPU but [head_dim/2] on RPU.
    # NOTE: next(model.parameters()).device may be CPU (embed_tokens kept on CPU),
    # so we must explicitly use "rpu" device.
    max_pos = getattr(rotary, "max_position_embeddings", 4096)
    dummy = torch.zeros(1, 1, hidden_size, dtype=torch.float16, device="rpu")
    dummy_pos = torch.arange(max_pos, device="rpu").unsqueeze(0)
    cos_cached, sin_cached = rotary(dummy, dummy_pos)
    if cos_cached.dim() == 3:
        cos_cached = cos_cached.squeeze(0)
        sin_cached = sin_cached.squeeze(0)

    cos_cached = cos_cached.to(dtype=torch.float16, device="rpu").contiguous()
    sin_cached = sin_cached.to(dtype=torch.float16, device="rpu").contiguous()

    # CPU mirror of the same tables, kept for the per-forward position_ids
    # gather below. Indexing the RPU copies would CPU-fallback and round-trip
    # every forward; these are ~1 MB total and read-only.
    _cos_cpu = cos_cached.cpu()
    _sin_cpu = sin_cached.cpu()

    # ------------------------------------------------------------------ #
    # 5. Final norm weight
    # ------------------------------------------------------------------ #
    # norm might be on CPU (for CPU FP32 final_norm). Ensure weight is on RPU for C++ set_weights.
    final_norm_w = model.norm.weight.to(dtype=torch.float16, device="rpu")

    # ------------------------------------------------------------------ #
    # 6. Prepare set_weights arguments
    # ------------------------------------------------------------------ #
    set_weights_args = (
        q_w_list, k_w_list, v_w_list, o_w_list,
        input_norm_list, post_norm_list,
        gate_list, up_list, down_list,
        cos_cached, sin_cached, final_norm_w,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size, eps,
        *scale_lists,
    )
    # ------------------------------------------------------------------ #
    # 7. Replace forward with all-layers-once implementation
    #
    # Capture signature constants in the closure. The signature drops position
    # (Pi05 VLM prefill always has position=0; chunk_masks_ uses
    # sdpa_prepare_mask's shape-keyed stable slot, NOT gemma2's baked
    # chunk_masks_ vector). 1 prefill call per forward → cross-forward
    # same-shape prompts all REPLAY.
    # ------------------------------------------------------------------ #
    _gemma_sig_hidden_size = int(hidden_size)
    _gemma_sig_num_layers = int(num_layers)
    _gemma_sig_num_q_heads = int(num_q_heads)
    _gemma_sig_w8a16 = bool(gemma_w8a16)

    def rpu_gemma_model_forward(
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
        # ---------------------------------------------------------- #
        # Precondition guards (same pattern as Qwen3)
        # ---------------------------------------------------------- #
        if output_attentions:
            raise AssertionError(
                "Gemma (RPU all-layers-once): output_attentions is not supported"
            )
        if output_hidden_states:
            raise AssertionError(
                "Gemma (RPU all-layers-once): output_hidden_states is not supported"
            )
        if use_cache is False:
            raise AssertionError(
                "Gemma (RPU all-layers-once): use_cache=False is not supported "
                "(caching path only; no-cache training scenario deferred)"
            )

        # ---------------------------------------------------------- #
        # Input embedding resolution uses the same mutual-exclusion guard as
        # the shared causal-decoder path.
        # ---------------------------------------------------------- #
        if input_ids is not None and inputs_embeds is not None:
            raise AssertionError(
                "Gemma (RPU all-layers-once): must provide exactly one of "
                "input_ids or inputs_embeds, got both"
            )
        if inputs_embeds is None:
            if input_ids is None:
                raise AssertionError(
                    "Gemma (RPU all-layers-once): must provide exactly one of "
                    "input_ids or inputs_embeds, got neither"
                )
            # embed_tokens lives on CPU; embed then move to RPU
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds.to(dtype=torch.float16, device="rpu")

        hidden_states = inputs_embeds

        # Move to RPU + fp16 if needed (Pi0.5 pipeline passes CPU embeds)
        if hidden_states.device.type != "rpu":
            hidden_states = hidden_states.to(dtype=torch.float16, device="rpu")
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        if hidden_states.dtype != torch.float16:
            hidden_states = hidden_states.to(torch.float16)

        seq_len = hidden_states.shape[1]

        # ---------------------------------------------------------- #
        # RPUCache selection.
        # ---------------------------------------------------------- #
        if past_key_values is None:
            # `_rpu_cache` is eagerly initialized during installation so
            # Dynamo does not record a changing attribute-existence guard.
            if use_cache:
                self._rpu_cache.reset_to_position(0)
            past_key_values = self._rpu_cache

        if not isinstance(past_key_values, RPUCache):
            raise AssertionError(
                f"Gemma (RPU all-layers-once): past_key_values must be an "
                f"RPUCache instance, got {type(past_key_values).__name__}"
            )

        # ---------------------------------------------------------- #
        # is_causal contract.
        # ---------------------------------------------------------- #
        is_causal = (attention_mask is None)

        # Mask normalization.
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.float16).cpu().contiguous()

        # ---------------------------------------------------------- #
        # RoPE positions must follow `position_ids` when masked rows can precede
        # real tokens; row indexes alone would renumber later tokens.
        #
        # Sent on EVERY forward, including when the positions ARE the row index:
        # which table the kernel reads is decided at graph BUILD, so a handle
        # that only sometimes sets it would replay one forward's choice against
        # another's positions. Falling back to `arange + position` rather than
        # skipping the call keeps that choice constant for the handle's whole
        # life. The default arange path preserves physical-row semantics.
        if position_ids is None:
            _pos = torch.arange(seq_len) + int(past_key_values.position)
        else:
            _pos = position_ids[0] if position_ids.dim() == 2 else position_ids
            _pos = _pos.to(device="cpu", dtype=torch.long)
        torch.ops.rpu.gemma_set_prefill_rope(
            self._rpu_vlm_decoder_handle,
            _cos_cpu[_pos].to("rpu"), _sin_cpu[_pos].to("rpu"))

        # ---------------------------------------------------------- #
        # Call fused all-layers-once C++ op
        # ---------------------------------------------------------- #
        chunk_size = getattr(
            self, '_rpu_planned_prefill_chunk_size',
            getattr(self, '_rpu_chunk_size', 0),
        )
        # A single chunk is valid because the layer-input DMA uses the same
        # `!ctx().input_in_spm` guard as the other fused models, so a
        # one-chunk prefill (which resolves InterLayerIO::AUTO to SPM_RESIDENT,
        # where the DDR ping-pong buffers are deliberately absent) died with
        # "ddr_broadcast_spm_dma: null ddr_ptr" at layer 1.
        import os as _os_cs
        if _os_cs.environ.get("RPU_PI05_LOG_CONVERSION"):
            _LOG.debug(
                "[iter-10-probe] GemmaModel.forward handle=%s seq_len=%d chunk_size=%s "
                "(_rpu_chunk_size attr present: %s)",
                self._rpu_vlm_decoder_handle, hidden_states.shape[1], chunk_size,
                hasattr(self, '_rpu_chunk_size'))
        # Wrap the fused C++ op in GraphCache.capture(sig). Pi05 VLM
        # prefill is invoked ONCE per inference with position=0 stable
        # with the same prompt shape across inferences can REPLAY. `chunk_masks_`
        # is rebuilt per-forward via dynamic_config lazy
        # prepare; each PreparedMask uses sdpa_prepare_mask's shape-keyed
        # stable slot → cross-forward same-shape ptr stability).
        _sig = rpu_backend.graph.GraphSignature(
            op_id="pi05_gemma_prefill",
            shapes=[int(seq_len), _gemma_sig_num_layers, _gemma_sig_hidden_size],
            dyn_dims=[_gemma_sig_num_q_heads, int(chunk_size)],
            dtypes=[torch.float16],
        )
        if _gemma_graph_enabled():
            with self._rpu_gemma_graph_cache.capture(_sig):
                output = torch.ops.rpu.gemma_forward(
                    self._rpu_vlm_decoder_handle,
                    hidden_states,
                    past_key_values.k_caches,
                    past_key_values.v_caches,
                    attention_mask,
                    past_key_values.position,
                    is_causal,
                    chunk_size,
                )
        else:
            output = torch.ops.rpu.gemma_forward(
                self._rpu_vlm_decoder_handle,
                hidden_states,
                past_key_values.k_caches,
                past_key_values.v_caches,
                attention_mask,
                past_key_values.position,
                is_causal,
                chunk_size,
            )

        # Advance global position (the fused kernel does not touch RPUCache state)
        past_key_values.update_position(seq_len)

        # Phase transition: release Gemma VLM's temporary SPM before the next
        # subsystem (AdaRMS Action Expert) allocates. Without this, AdaRMS's bump
        # allocation lands at different absolute offsets depending on Gemma's
        # lifecycle-aliasing config — causing action corruption even though VLM
        # output is bit-identical. Gemma's next call re-allocates via ensure_allocated
        # Path 2 (gen_changed), which safely re-runs preprocess_all_norms().
        # This must stay outside the capture scope above: graph-aware
        # spm_alloc_reset_temporary marks the graph non-replayable.
        torch.ops.rpu.spm_alloc_reset_temporary()

        # Final norm is done in C++ (fused into last layer)

        # Record prefix length for shared cache (used by denoise_step to
        # reset position between denoising iterations.
        if use_cache and isinstance(past_key_values, RPUCache):
            past_key_values._prefix_len = past_key_values.position

        # Optional KV cache probe after prefill.
        import os as _os
        _probe_dir = _os.environ.get("RPU_PI05_PROBE_DIR")
        if use_cache and _probe_dir and isinstance(past_key_values, RPUCache):
            try:
                import torch as _torch
                _os.makedirs(_probe_dir, exist_ok=True)
                for _L in (0, 8, 17):
                    _k = past_key_values.k_caches[_L].cpu().float()
                    _v = past_key_values.v_caches[_L].cpu().float()
                    _torch.save(_k, _os.path.join(_probe_dir, f"kv_post_prefill_L{_L}_k.pt"))
                    _torch.save(_v, _os.path.join(_probe_dir, f"kv_post_prefill_L{_L}_v.pt"))
            except Exception as _exc:
                _LOG.warning("KV probe dump failed: %s", _exc)

        return BaseModelOutputWithPast(
            last_hidden_state=output,
            past_key_values=past_key_values if use_cache is not False else None,
        )

    # Eagerly initialize _rpu_cache, stamp the installed state, and freeze it.
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
    # Per-instance GraphCache for the Gemma VLM prefill wrapper. It uses the same
    # Dynamo guard-stability rationale as `_rpu_cache` above — eager init.
    gemma_graph_cache = rpu_backend.graph.GraphCache()
    required_attrs = (
        '_rpu_cache',
        '_rpu_vlm_decoder_handle',
        '_rpu_gemma_graph_cache',
    )

    handle = torch.ops.rpu.gemma_create()
    handle_finalizer = None
    committed = False
    try:
        handle_finalizer = weakref.finalize(
            model, _gemma_destroy_handle, h=handle)
        torch.ops.rpu.gemma_set_weights(handle, *set_weights_args)

        model._rpu_cache = rpu_cache
        model._rpu_gemma_graph_cache = gemma_graph_cache
        model._rpu_lazy_init_checked = True
        model._rpu_required_attrs = required_attrs
        model._rpu_vlm_decoder_handle = handle
        model._rpu_vlm_decoder_handle_finalizer = handle_finalizer
        model.forward = types.MethodType(rpu_gemma_model_forward, model)

        from rpu_backend.graph.lazy_init_guard import _verify_lazy_init
        _verify_lazy_init(model)

        if (
            had_old_handle
            and (
                old_finalizer is None
                or getattr(old_finalizer, "alive", False)
            )
        ):
            torch.ops.rpu.gemma_destroy(old_handle)
        committed = True
    except BaseException:
        if not committed:
            model_state = vars(model)
            for name in _GEMMA_RUNTIME_INSTALL_ATTRS:
                model_state.pop(name, None)
            model_state.update(install_snapshot)

            if handle_finalizer is None:
                _gemma_destroy_handle(handle)
            elif handle_finalizer.alive:
                handle_finalizer()
        raise

    if old_finalizer is not None and getattr(old_finalizer, "alive", False):
        old_finalizer.detach()

    _LOG.info("Patched GemmaModel (instance) with all-layers-once fused forward, "
              "handle=%d, num_layers=%d%s", handle, num_layers,
              ", W8A16 graph enabled" if gemma_w8a16 else "")

    return handle
