"""Pi0.5 Gemma all-layers-once instance patch (canonical home).

v5-03 B4: relocated from _internal/patches/pi05_all_layers_once.py L41-369
per ADR §6.2. Pi05Adapter is the only in-tree caller (lazy import inside
to_rpu()); test consumers live under tests/model/test_gemma_*.py.

ADR §3.3 forbids reintroducing a ComponentBase abstraction; this file is
intentionally a flat function module sharing the adapters/pi05/ namespace
with adarms.py and siglip.py.
"""
from __future__ import annotations
import types

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
from rpu_backend.runtime._native_retirement import _InstalledNativeResource


def _gemma_graph_enabled() -> bool:
    return rpu_env_bool("RPU_PI05_GEMMA_GRAPH", default=True)


# patch-reason: (a) Gemma all-layers-once C++ handle lifecycle helper — §3a (a) run-different-op-on-RPU
def _gemma_destroy_handle(h):
    """Raw destroy; the installed resource owns retirement failures."""
    torch.ops.rpu.gemma_destroy(h)


_GEMMA_RUNTIME_INSTALL_ATTRS = (
    "_gemma_graph_enabled",
    "_rpu_cache",
    "_rpu_gemma_graph_cache",
    "_rpu_prefill_kv_only",
    "_rpu_lazy_init_checked",
    "_rpu_required_attrs",
    "_rpu_vlm_decoder_handle",
    "_rpu_vlm_decoder_handle_finalizer",
    "_rpu_vlm_decoder_retirement_state",
    "forward",
)


# patch-reason: (a) Gemma all-layers-once instance patch — §3a (a) run-different-op-on-RPU
def patch_gemma_model_for_rpu_all_layers_once(
    model, *, runtime_policy=None, prefill_pair_rows=400,
) -> int:
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
      - RPUCache is auto-created when past_key_values is None (D-14)

    Successful rollback preserves the published handle and forward/cache state.
    Uncertain retirement retains both installs and poisons the process.

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

    from rpu_backend.api._execution import _require_execution_process_safe

    _require_execution_process_safe()
    from .cores import model_topology, graph_cache_options, validate_native_topology
    topology = model_topology(model)
    old_resource = getattr(model, "_rpu_vlm_decoder_retirement_state", None)
    if old_resource is not None:
        old_resource.require_replaceable()
    elif getattr(model, "_rpu_vlm_decoder_handle", None) is not None:
        raise RuntimeError("Pi0.5 Gemma replacement requires its actual retirement resource")
    model_state = vars(model)
    graph_enabled = (
        bool(model_state["_gemma_graph_enabled"])
        if "_gemma_graph_enabled" in model_state
        else bool(_gemma_graph_enabled())
    )
    install_snapshot = {
        name: model_state[name]
        for name in _GEMMA_RUNTIME_INSTALL_ATTRS
        if name in model_state
    }
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
    # Cold, per-install opt-in. The forward closure and its bounded cache retire
    # together; a replacement native handle cannot inherit another's upload.
    _prefill_rope_cache = None
    if rpu_env_bool("RPU_PI05_PREFILL_ROPE_CACHE", default=False):
        from ._rope_cache import PrefillRopeUploadCache
        with torch.inference_mode(False):
            _cos_cpu = _cos_cpu.clone()
            _sin_cpu = _sin_cpu.clone()
        _prefill_rope_cache = PrefillRopeUploadCache(_cos_cpu, _sin_cpu)

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
    # Capture signature constants in the closure. VLM prefill resets position
    # to zero, and prepared masks use stable shape-keyed storage. Matching
    # prompt shapes can therefore reuse the prefill graph.
    # ------------------------------------------------------------------ #
    _gemma_sig_hidden_size = int(hidden_size)
    _gemma_sig_num_layers = int(num_layers)
    _gemma_sig_num_q_heads = int(num_q_heads)
    _gemma_sig_w8a16 = bool(gemma_w8a16)
    _prefill_kv_only = not bool(getattr(model, "_fmb_execution_component_config", {})
                                .get("prefill", {}).get("linear_acc32", False)) and rpu_env_bool(
        "RPU_PI05_PREFILL_KV_ONLY", default=False,
        cpp_mirror="src/fused/rpu_gemma_model.cpp",
    )

    def _run_gemma_prefill(
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
        _kv_only=False,
        **kwargs,
    ):
        if bool(_kv_only) != _prefill_kv_only:
            raise RuntimeError(
                "Pi0.5 Gemma entry does not match the installed KV-only policy; "
                "use the dedicated cache-only entry for a KV-only owner"
            )
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
        # Input embedding resolution (Bug 4 fix: require inputs_embeds,
        # same mutual-exclusion guard as Qwen3 all-layers-once)
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
        # D-14: RPUCache auto-creation (matching v2 pi05_converter.py:794-816)
        # ---------------------------------------------------------- #
        if past_key_values is None:
            # Installation creates the cache eagerly, keeping its ownership
            # stable across tracing and replay.
            if use_cache:
                self._rpu_cache.reset_to_position(0)
            past_key_values = self._rpu_cache

        if not isinstance(past_key_values, RPUCache):
            raise AssertionError(
                f"Gemma (RPU all-layers-once): past_key_values must be an "
                f"RPUCache instance, got {type(past_key_values).__name__}"
            )

        # ---------------------------------------------------------- #
        # D-13: is_causal contract (exact v2 rule, no ambiguity)
        # ---------------------------------------------------------- #
        is_causal = (attention_mask is None)

        # D-13: Mask normalization (matching model_converter.py:1188-1195)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.float16).cpu().contiguous()

        # ---------------------------------------------------------- #
        # RoPE positions. The fused prefill used to derive RoPE from the ROW
        # INDEX because this op never forwarded `position_ids` — correct only
        # while no masked row precedes a real one. Violating that renumbered
        # every token after an absent camera and held the pi05 camera legs red
        # at mse 0.696 until 2026-08-06, which was worked around by DROPPING
        # absent cameras from the prefix (adapters/pi05/patches.py). Gathering
        # the table by `position_ids` removes the constraint instead.
        #
        # Sent on EVERY forward, including when the positions ARE the row index:
        # which table the kernel reads is decided at graph BUILD, so a handle
        # that only sometimes sets it would replay one forward's choice against
        # another's positions. Falling back to `arange + position` rather than
        # skipping the call keeps that choice constant for the handle's whole
        # life, and reproduces the old row-index behaviour exactly. Cost is a
        # [seq, head_dim/2] gather + H2D — ~78 KB at pi05's prefix lengths,
        # against a whole VLM prefill.
        if position_ids is None:
            _pos = torch.arange(seq_len) + int(past_key_values.position)
        else:
            _pos = position_ids[0] if position_ids.dim() == 2 else position_ids
            _pos = _pos.to(device="cpu", dtype=torch.long)
        if _prefill_rope_cache is None:
            _cos_rows, _sin_rows = _cos_cpu[_pos].to("rpu"), _sin_cpu[_pos].to("rpu")
        else:
            _cos_rows, _sin_rows = _prefill_rope_cache.tables(_pos)
        # Always refresh the native stable slots, including a cache hit. BUILD
        # and REPLAY keep the same table source and dynamic position semantics.
        torch.ops.rpu.gemma_set_prefill_rope(
            self._rpu_vlm_decoder_handle, _cos_rows, _sin_rows)

        # ---------------------------------------------------------- #
        # Call fused all-layers-once C++ op
        # ---------------------------------------------------------- #
        a6_plan = self._rpu_planned_prefill_plan
        selected = a6_plan.selected
        if selected is None or not selected.stage_tuple.physical_descriptor:
            raise RuntimeError("Pi0.5 prefill A6 winner has no native descriptor")
        chunk_size = int(selected.stage_tuple.compute_chunk)
        planned_stage_descriptor = selected.stage_tuple.physical_descriptor
        # A single chunk is fine again. The "split a would-be-single chunk in
        # two" workaround that lived here was removed once the real defect was
        # found: rpu_gemma_model.cpp emitted the layer-input DMA without the
        # `!ctx().input_in_spm` guard every other fused model has, so a
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
        # Prefill starts at cache position zero on every inference. Masks are
        # refreshed in stable shape-keyed slots before the cached graph runs.
        _sig = rpu_backend.graph.GraphSignature(
            op_id="pi05_gemma_prefill",
            shapes=[int(seq_len), _gemma_sig_num_layers, _gemma_sig_hidden_size],
            dyn_dims=[_gemma_sig_num_q_heads, int(chunk_size),
                      *a6_plan.graph_key_words()],
            dtypes=[torch.float16],
        )
        native_forward = (
            torch.ops.rpu.gemma_prefill_kv_only
            if _kv_only else torch.ops.rpu.gemma_forward
        )
        if graph_enabled:
            with self._rpu_gemma_graph_cache.capture(_sig):
                output = native_forward(
                    self._rpu_vlm_decoder_handle,
                    hidden_states,
                    past_key_values.k_caches,
                    past_key_values.v_caches,
                    attention_mask,
                    past_key_values.position,
                    is_causal,
                    0,
                    planned_stage_descriptor,
                )
        else:
            output = native_forward(
                self._rpu_vlm_decoder_handle,
                hidden_states,
                past_key_values.k_caches,
                past_key_values.v_caches,
                attention_mask,
                past_key_values.position,
                is_causal,
                0,
                planned_stage_descriptor,
            )

        # Advance global position (the fused kernel does not touch RPUCache state)
        past_key_values.update_position(seq_len)

        # Phase transition: release Gemma VLM's temporary SPM before the next
        # subsystem (AdaRMS Action Expert) allocates. Without this, AdaRMS's bump
        # allocation lands at different absolute offsets depending on Gemma's
        # lifecycle-aliasing config — causing action corruption even though VLM
        # output is bit-identical. Gemma's next call re-allocates via ensure_allocated
        # Path 2 (gen_changed), which safely re-runs preprocess_all_norms().
        # R-4: MUST stay OUTSIDE the capture scope above — graph-aware
        # spm_alloc_reset_temporary marks the graph non-replayable.
        torch.ops.rpu.spm_alloc_reset_temporary()

        # Final norm is done in C++ (fused into last layer)

        # Record prefix length for shared cache (used by denoise_step to
        # reset position between denoising iterations — matches v2 behavior
        # at pi05_converter.py:901)
        if use_cache and isinstance(past_key_values, RPUCache):
            past_key_values._prefix_len = past_key_values.position

        if _kv_only:
            return past_key_values
        return BaseModelOutputWithPast(
            last_hidden_state=output,
            past_key_values=past_key_values if use_cache is not False else None,
        )

    def rpu_gemma_model_forward(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, use_cache=None,
        cache_position=None, output_attentions=None, output_hidden_states=None,
        return_dict=None, adarms_cond=None, **kwargs,
    ):
        # Preserve the ordinary hidden-state return contract. A KV-only owner
        # rejects this entry before input/cache mutation in the shared body.
        return _run_gemma_prefill(
            self, input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            cache_position=cache_position, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
            adarms_cond=adarms_cond, _kv_only=False, **kwargs,
        )

    def rpu_gemma_prefill_kv_only(
        self, *, inputs_embeds, attention_mask, position_ids,
    ):
        """Build the full prefix cache without promising final hidden states."""
        return _run_gemma_prefill(
            self, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=None, use_cache=True,
            _kv_only=True,
        )

    # P7.1h L1+L2:eager init _rpu_cache + stamp marker + freeze.
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
    # S3: per-instance GraphCache for the Gemma VLM prefill wrap. Same
    # Dynamo guard-stability rationale as `_rpu_cache` above — eager init.
    gemma_graph_cache = (
        rpu_backend.graph.GraphCache(**graph_cache_options(topology.num_cores))
        if runtime_policy is None
        else rpu_backend.graph.GraphCache(runtime_policy=runtime_policy)
    )
    required_attrs = (
        '_rpu_cache',
        '_rpu_vlm_decoder_handle',
        '_rpu_gemma_graph_cache',
    )

    resource = _InstalledNativeResource(
        model, None, _gemma_destroy_handle, graphs=(gemma_graph_cache,),
        keepalive=(set_weights_args, rpu_cache), label="Pi0.5 Gemma",
        handle_name="_rpu_vlm_decoder_handle")
    handle_finalizer = resource.finalizer
    committed = False
    try:
        resource.handle = handle = torch.ops.rpu.gemma_create(
            bool(getattr(model, "_fmb_execution_component_config", {})
                 .get("prefill", {}).get("linear_acc32", False)))
        if topology.num_cores != 8:
            torch.ops.rpu.gemma_set_execution_core_count(handle, topology.num_cores)
        torch.ops.rpu.gemma_set_prefill_pair_rows(handle, prefill_pair_rows)
        torch.ops.rpu.gemma_set_weights(handle, *set_weights_args)
        validate_native_topology("gemma", handle, topology, intermediate_size, gate_list[0].dtype)
        # Physical KV capacity is independent of gathered logical RoPE rows.
        torch.ops.rpu.gemma_set_chunk_envelope(handle, _max_seq, 0)

        model._rpu_cache = rpu_cache
        model._gemma_graph_enabled = graph_enabled
        model._rpu_gemma_graph_cache = gemma_graph_cache
        model._rpu_prefill_kv_only = (
            types.MethodType(rpu_gemma_prefill_kv_only, model)
            if _prefill_kv_only else None
        )
        model._rpu_lazy_init_checked = True
        model._rpu_required_attrs = required_attrs
        model._rpu_vlm_decoder_handle = handle
        model._rpu_vlm_decoder_retirement_state = resource
        model._rpu_vlm_decoder_handle_finalizer = handle_finalizer
        model.forward = types.MethodType(rpu_gemma_model_forward, model)

        from rpu_backend.graph.lazy_init_guard import _verify_lazy_init
        _verify_lazy_init(model)

        if old_resource is not None:
            old_resource.retire()
        committed = True
    except BaseException as error:
        if not committed:
            cleanup_ok = resource.cleanup_failure(error, model, install_snapshot)
            if resource.handle is None and handle_finalizer.alive:
                handle_finalizer.detach()
            model_state = vars(model)
            if cleanup_ok:
                for name in _GEMMA_RUNTIME_INSTALL_ATTRS:
                    model_state.pop(name, None)
                model_state.update(install_snapshot)
            else:
                model_state.update(
                    _rpu_vlm_decoder_handle=resource.handle,
                    _rpu_vlm_decoder_handle_finalizer=handle_finalizer,
                    _rpu_vlm_decoder_retirement_state=resource,
                    _rpu_gemma_graph_cache=gemma_graph_cache, _rpu_cache=rpu_cache)
        raise

    if old_finalizer is not None and getattr(old_finalizer, "alive", False):
        old_finalizer.detach()

    _LOG.info("Patched GemmaModel (instance) with all-layers-once fused forward, "
              "handle=%d, num_layers=%d%s", handle, num_layers,
              ", W8A16 graph enabled" if gemma_w8a16 else "")

    return handle
