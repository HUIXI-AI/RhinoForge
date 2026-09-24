"""Pi0.5 AdaRMS all-layers-once instance patch (canonical home).

v5-03 B4: relocated from _internal/patches/pi05_all_layers_once.py L372-770
per ADR §6.2. Pi05Adapter is the only in-tree caller (lazy import inside
to_rpu()); no test consumers reference this module directly.

ADR §3.3 forbids reintroducing a ComponentBase abstraction; this file is
intentionally a flat function module sharing the adapters/pi05/ namespace
with gemma.py and siglip.py.
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
from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
from rpu_backend.runtime.execution_planner import GRAPH_COMPOSITE_CHILD


_ADARMS_ACTION_COMPONENT = "action_expert"


def _adarms_graph_enabled() -> bool:
    return rpu_env_bool("RPU_PI05_ADARMS_GRAPH", default=True)


def _adarms_w8a16_graph_enabled() -> bool:

    return rpu_env_bool("RPU_PI05_ADARMS_W8A16_GRAPH", default=True)


def _plan_adarms_action_execution(
    model,
    *,
    logical_len: int,
    execution_len: int,
    position: int,
    kv_len: int,
    use_attention_mask: bool,
    is_causal: bool,
    _cost_request=None,
    graph_cache=None,
):
    """Resolve the native one-chunk action descriptor before Graph admission."""
    config = getattr(
        model, "_fmb_execution_component_config",
        {"action": {"chunk_size": "auto"}},
    )
    requested = config.get("action", {}).get("chunk_size", "auto")
    if _cost_request is not None:
        from rpu_backend.runtime.decoder import _cold_text_cost_request

        request, _ = _cold_text_cost_request(
            model, ("adarms", int(model._rpu_action_handle)), model._rpu_cache,
            _cost_request, position, component=_ADARMS_ACTION_COMPONENT, stage="action")
        if request.padding_rows != 0 or request.padding_budget != 0:
            raise ValueError("Pi0.5 AdaRMS action cannot pad or split its suffix")
        requested = request.chunk_size or "auto"
    exact_chunk = requested if isinstance(requested, int) else None
    result_box = {}
    resolved_execution, resolved_chunk = plan_bounded_prefill_execution(
        logical_len,
        execution_len,
        0,
        execution_owner=model,
        execution_component="action_expert",
        execution_stage="action",
        execution_native=("adarms", int(model._rpu_action_handle)),
        position=position,
        alignment=1,
        padding_rows=0,
        exact_chunk_size=exact_chunk,
        resolve_stage_domain=lambda length: (
            torch.ops.rpu.adarms_resolve_action_stage_domain(
                int(model._rpu_action_handle),
                int(length),
                logical_len,
                position,
                kv_len,
                use_attention_mask,
                is_causal,
                0 if exact_chunk is None else exact_chunk,
            )
        ),
        request_id="pi05:action_expert:action",
        graph_mode=GRAPH_COMPOSITE_CHILD,
        queue_owner_id=int(model._rpu_action_handle),
        physical_metadata=(
            ("component:action_expert", 1),
            ("execution_generation", int(getattr(
                model, "_fmb_execution_generation", 0
            ))),
            ("prefix_len", position),
            ("kv_len", kv_len),
        ),
        plan_result_sink=lambda result: result_box.__setitem__("result", result),
        plan_signature=(int(kv_len), bool(use_attention_mask), bool(is_causal)),
        graph_cache=graph_cache,
    )
    result = result_box["result"]
    selected = result.selected
    if (
        selected is None
        or resolved_execution != execution_len
        or resolved_chunk != selected.stage_tuple.compute_chunk
        or not selected.stage_tuple.physical_descriptor
    ):
        raise RuntimeError(
            "Pi0.5 AdaRMS planner returned no consumable native descriptor"
        )
    return result


def _publish_adarms_action_plan(model, plan, *, logical_len: int) -> None:
    selected = plan.selected
    if selected is None:  # pragma: no cover - planner rejects first
        raise RuntimeError("Pi0.5 AdaRMS planner selected no action plan")
    resolved = int(torch.ops.rpu.adarms_get_resolved_chunk_size(
        model._rpu_action_handle
    ))
    if resolved != selected.stage_tuple.compute_chunk:
        raise RuntimeError(
            "Pi0.5 AdaRMS dry/forward chunk plan drift: "
            f"dry={selected.stage_tuple.compute_chunk}, forward={resolved}"
        )
    vars(model)["_fmb_last_execution_plan"] = plan
    vars(model)["_rpu_last_action_execution_plan"] = {
        **plan.as_dict(include_candidates=False),
        "component": _ADARMS_ACTION_COMPONENT,
        "stage": "action",
        "logical_len": logical_len,
        "execution_len": selected.execution_len,
        "chunk_size": resolved,
        "padding_rows": selected.padding_rows,
        "authority": "NATIVE_A6_STAGE_DESCRIPTOR",
        "attention_policy": "DDR_REQUIRED",
        "attention_reason": "PREFIX_HISTORY_DDR_REQUIRED_NO_RAW_RESIDENCY_ABI",
        "dry_forward_agreement": True,
    }


# patch-reason: (a) AdaRMS all-layers-once C++ handle lifecycle helper — §3a (a)
def _adarms_destroy_handle(h):
    """Raw destroy; the installed resource owns retirement failures."""
    torch.ops.rpu.adarms_destroy(h)


_ADARMS_RUNTIME_INSTALL_ATTRS = (
    "_adarms_graph_enabled",
    "_adarms_w8a16_graph_enabled",
    "_adarms_graph_capture_enabled",
    "_gemma_rope_cos",
    "_gemma_rope_sin",
    "_rpu_cache",
    "_rpu_adarms_graph_cache",
    "_rpu_lazy_init_checked",
    "_rpu_required_attrs",
    "_rpu_action_handle",
    "_rpu_action_handle_finalizer",
    "_rpu_action_retirement_state",
    "forward",
)


def _prepare_adarms_dense_weights(model):
    """Swizzle per-layer AdaRMS dense weights for RPU row-partition GEMM.

    For each layer x each of {input_layernorm, post_attention_layernorm}:
      [3H, H] -> transform_linear_weight(partition=0, num_cores=8) -> row-partition swizzle
      -> .to('rpu')

    The C++ `AdaRMSModel::adarms_gemv_to_spm` runs the generated row-partition
    auto-tile family at M=1: each core holds local_k=H/NUM_CORES columns of the
    full [3H, H] weight and produces a partial [3H]; a fused
    all_reduce_sum_residual folds in the bias and broadcasts the full [3H]
    back to every core.

    Returns four parallel lists (one entry per layer):
      attn_dense_w_list, attn_dense_b_list, mlp_dense_w_list, mlp_dense_b_list

    The helper is self-contained: if `norm._rpu_dense_w_rp` is already
    registered (pi05_converter v2 pre-registration under the new name), it is
    reused; otherwise the weight is recomputed from the live `norm.dense`
    Linear. The cache buffer is version-bumped from `_rpu_dense_w` to
    `_rpu_dense_w_rp` so stale col-partition-replicated tensors from older
    converter versions are ignored.
    """
    from .cores import model_topology
    condition_cores = model_topology(model).attn_tp
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
                orig_w, partition=0, num_cores=condition_cores
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


# patch-reason: (a) AdaRMS all-layers-once instance patch — §3a (a) run-different-op-on-RPU
def patch_adarms_model_for_rpu_all_layers_once(
    model, *, runtime_policy=None,
) -> int:
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
         - auto-creates RPUCache on first call (D-14)
         - computes is_causal per D-13 contract
         - calls torch.ops.rpu.adarms_forward(handle, ...)
         - calls torch.ops.rpu.spm_alloc_reset_temporary() at exit
         - returns BaseModelOutputWithPast (no final_norm applied -- D-402)

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

    from rpu_backend.api._execution import _require_execution_process_safe
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource

    _require_execution_process_safe()
    from .cores import model_topology, graph_cache_options, validate_native_topology
    topology = model_topology(model)
    old_resource = getattr(model, "_rpu_action_retirement_state", None)
    if old_resource is not None:
        old_resource.require_replaceable()
    elif getattr(model, "_rpu_action_handle", None) is not None:
        raise RuntimeError("Pi0.5 AdaRMS replacement requires its actual retirement resource")
    model_state = vars(model)
    graph_enabled = (
        bool(model_state["_adarms_graph_enabled"])
        if "_adarms_graph_enabled" in model_state
        else bool(_adarms_graph_enabled())
    )
    install_snapshot = {
        name: model_state[name]
        for name in _ADARMS_RUNTIME_INSTALL_ATTRS
        if name in model_state
    }
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
    # Step 3: prepare set_weights arguments (NO final_norm -- D-402)
    # ------------------------------------------------------------------ #
    from rpu_backend.adapters.pi05.w8a16 import pi05_expert_scale_lists, pi05_nvfp4_tensor_scale_tables
    scale_lists = pi05_expert_scale_lists(model, require_rpu=True)
    adarms_w8a16 = bool(scale_lists[0])
    w8a16_graph_enabled = (
        bool(model_state["_adarms_w8a16_graph_enabled"])
        if "_adarms_w8a16_graph_enabled" in model_state
        else (
            bool(_adarms_w8a16_graph_enabled())
            if adarms_w8a16
            else True
        )
    )
    graph_capture_enabled = bool(
        graph_enabled and (not adarms_w8a16 or w8a16_graph_enabled)
    )
    set_weights_args = (
        q_w_list, k_w_list, v_w_list, o_w_list,
        gate_w_list, up_w_list, down_w_list,
        attn_dense_w_list, attn_dense_b_list,
        mlp_dense_w_list,  mlp_dense_b_list,
        cos_cached, sin_cached,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size, eps,
        *scale_lists, pi05_nvfp4_tensor_scale_tables(model),
    )

    # ------------------------------------------------------------------ #
    # Step 6: replace forward
    #
    # Capture model constants in the closure for GraphCache signatures.
    # Mutable kernel registers are refreshed on replay; mask DMA uses stable
    # prepared storage. The signature below also binds prefix and RoPE positions.
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
        # -- Preconditions (Python-side guards before the C++ op, T-04-06) --
        if output_attentions:
            raise AssertionError(
                "AdaRMS (RPU all-layers-once): output_attentions is not supported"
            )
        if output_hidden_states:
            raise AssertionError(
                "AdaRMS (RPU all-layers-once): output_hidden_states is not supported"
            )
        # D-14 relaxed: use_cache=False is legitimate for Pi0.5 denoise_step
        # (see pi05_converter._patched_denoise_step -> paligemma_with_expert.forward
        # -> gemma_expert.model.forward with use_cache=False). Matches v2 pattern in
        # pi05_converter.py:802/900 which gates cache-reset and _prefix_len setter
        # on use_cache, but still uses the cache for SDPA KV reads/writes in the
        # C++ op regardless. The guard below on past_key_values/RPUCache is the
        # actual correctness check.
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
        # Capture the caller's original device so we can restore it before the
        # final norm (matches v2 rpu_gemma_model_forward at pi05_converter.py:792
        # / 894 which does `orig_device = inputs_embeds.device` then
        # `hidden_states.to(orig_device)` before `self.norm(...)`).
        # Pi0.5 denoise_step delivers suffix_embs on CPU; downstream code
        # (modeling_pi05.py:901-902) needs the suffix hidden on CPU so that
        # action_out_proj's CPU weights dispatch correctly and v_t lands on
        # the same device as x_t (CPU) for the `x_t + dt * v_t` update at :858.
        orig_device = hidden_states.device
        # Auto-route CPU inputs to RPU (matches v2 pattern in pi05_converter.py:826
        # where rpu_gemma_model_forward does `inputs_embeds.to(dtype=fp16, device=model_device)`
        # unconditionally).
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

        # -- cond normalization ([hidden_size] fp16 RPU contiguous) (T-04-05) --
        cond_rpu = adarms_cond.to(dtype=torch.float16, device='rpu')
        if cond_rpu.dim() == 2 and cond_rpu.shape[0] == 1:
            cond_rpu = cond_rpu.squeeze(0)
        if cond_rpu.dim() != 1 or cond_rpu.shape[0] != hidden_size:
            raise AssertionError(
                f"AdaRMS: cond must be [hidden_size={hidden_size}] or "
                f"[1, hidden_size], got {tuple(cond_rpu.shape)}"
            )
        cond_rpu = cond_rpu.contiguous()

        # -- D-14: RPUCache auto-creation --
        if past_key_values is None:
            # Installation creates the cache eagerly for stable ownership.
            if use_cache:
                self._rpu_cache.reset_to_position(0)
            past_key_values = self._rpu_cache

        if not isinstance(past_key_values, RPUCache):
            raise AssertionError(
                f"AdaRMS: past_key_values must be RPUCache, got "
                f"{type(past_key_values).__name__}"
            )

        # -- D-13: is_causal contract + mask normalization --
        is_causal = (attention_mask is None)
        if attention_mask is not None:
            # PERF L2 (2026-05-07): cache the converted mask keyed on input
            # Python identity. The Pi0.5 denoise loop builds the mask ONCE
            # outside the loop (runtime.py L2 hoist), so steps 2-5 pass the
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

        # -- S1: wrap the fused C++ op in GraphCache.capture(sig) --
        # sig keys on `position` (== prefix_len here), which is what the SDPA
        # kv length and the [1,1,suffix,prefix+suffix] 2D mask both derive
        # from. It is NOT a per-step split: the denoise loop calls
        # reset_to_position(prefix_len) before every step (runtime.py), so all
        # steps of one profile share a position and therefore one cache entry
        # -- the original reason for dropping it still holds.
        #
        # Denoise passes an explicit attention mask, so a graph prepared for
        # one prefix length cannot be reused with another prefix geometry.
        # RoPE start for the suffix. The expert's `position` is the PHYSICAL
        # cache row (the denoise loop resets the shared cache to
        # prefix_pad_masks.shape[1]); its RoPE position is the LOGICAL one the
        # reference uses, sum(prefix_pad_masks). They differ by the prefix's pad
        # rows. -1 keeps the old physical-row behaviour for callers that pass no
        # position_ids. Baked at graph BUILD, hence also in the signature below.
        _rope_pos = -1
        if position_ids is not None:
            _p = position_ids[0] if position_ids.dim() == 2 else position_ids
            _rope_pos = int(_p.reshape(-1)[0])
        torch.ops.rpu.adarms_set_rope_position(
            self._rpu_action_handle, _rope_pos)

        _position = int(past_key_values.position)
        _kv_len = (
            int(attention_mask.shape[-1])
            if attention_mask is not None
            else _position + int(seq_len)
        )
        _component_plan = _plan_adarms_action_execution(
            self,
            logical_len=int(seq_len),
            execution_len=int(seq_len),
            position=_position,
            kv_len=_kv_len,
            use_attention_mask=attention_mask is not None,
            is_causal=is_causal,
            graph_cache=self._rpu_adarms_graph_cache,
        )
        _selected_component_plan = _component_plan.selected
        if _selected_component_plan is None:
            raise RuntimeError("Pi0.5 AdaRMS planner selected no action plan")

        _sig = rpu_backend.graph.GraphSignature(
            op_id="pi05_adarms",
            shapes=[int(seq_len), _adarms_sig_hidden_size],
            dyn_dims=[
                _adarms_sig_num_layers, _adarms_sig_num_q_heads,
                int(past_key_values.position), _rope_pos,
                *_component_plan.graph_key_words(),
            ],
            dtypes=[torch.float16],
        )
        # -- Call the fused C++ op (schema: handle, hidden, cond, k_caches,
        #    v_caches, attention_mask, position, is_causal) --
        if graph_capture_enabled:
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
                    list(
                        _selected_component_plan.stage_tuple.physical_descriptor
                    ),
                )
        else:
            # DIAGNOSTIC_ONLY: without an active Graph, FMB cannot bind a
            # COMPLETE descriptor. The same native AUTO resolver still runs,
            # and the dry/forward agreement check below fails on any drift.
            output = torch.ops.rpu.adarms_forward(
                self._rpu_action_handle,
                hidden_states,
                cond_rpu,
                past_key_values.k_caches,
                past_key_values.v_caches,
                attention_mask,
                past_key_values.position,
                is_causal,
                [],
            )

        _publish_adarms_action_plan(
            self, _component_plan, logical_len=int(seq_len)
        )

        past_key_values.update_position(seq_len)

        # Subsystem boundary: release temporary SPM for the next subsystem.
        # Keep the reset outside capture: resetting temporary SPM inside a
        # graph would mark that graph non-replayable.
        torch.ops.rpu.spm_alloc_reset_temporary()

        if use_cache and isinstance(past_key_values, RPUCache):
            past_key_values._prefix_len = past_key_values.position

        # Restore caller's original device before the final norm — mirrors v2
        # rpu_gemma_model_forward at pi05_converter.py:894 `hidden_states.to(orig_device)`.
        # Pi0.5 denoise_step delivers suffix_embs on CPU and expects the suffix
        # output on CPU so that modeling_pi05.py:901-902 / :858
        # (`x_t = x_t + dt * v_t`) stays on CPU (matches x_t's CPU noise tensor
        # and action_out_proj's CPU weights).
        if orig_device.type != 'rpu':
            output = output.to(orig_device)

        # Final PiGemmaRMSNorm — matches v2 pi05_converter.py:894-895 which
        # runs `hidden_states, _ = self.norm(hidden_states, adarms_cond)`
        # inside the patched `rpu_gemma_model_forward`. D-402 (no C++ fusion)
        # only prohibits baking final_norm into the decoder-layer graph; the
        # Python-level final norm still needs to be applied inside this
        # `model.forward` wrapper so `last_hidden_state` is post-norm, matching
        # both v2 behavior AND the lerobot PaliGemmaWithExpertModel.forward
        # suffix-only branch (modeling_pi05.py:475-483) that consumes
        # `suffix_output.last_hidden_state` directly.
        if getattr(self, "norm", None) is not None:
            output, _ = self.norm(output, adarms_cond)

        return BaseModelOutputWithPast(
            last_hidden_state=output,
            past_key_values=past_key_values if use_cache is not False else None,
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
    # S1: per-instance GraphCache for the AdaRMS forward wrap. Eager-init
    # mirrors `_rpu_cache` to keep Dynamo guard-set stable across calls
    # (lazy init would invalidate the frontend cache on first hit;
    # see siglip.py line ~477-481 for the same pattern + reasoning).
    adarms_graph_cache = (
        rpu_backend.graph.GraphCache(**graph_cache_options(topology.num_cores))
        if runtime_policy is None
        else rpu_backend.graph.GraphCache(runtime_policy=runtime_policy)
    )
    required_attrs = (
        '_rpu_cache',
        '_rpu_action_handle',
        '_rpu_adarms_graph_cache',
    )

    resource = _InstalledNativeResource(
        model, None, _adarms_destroy_handle, graphs=(adarms_graph_cache,),
        keepalive=(set_weights_args, rpu_cache), label="Pi0.5 AdaRMS",
        handle_name="_rpu_action_handle")
    handle_finalizer = resource.finalizer
    committed = False
    try:
        resource.handle = handle = torch.ops.rpu.adarms_create(
            bool(getattr(model, "_fmb_execution_component_config", {})
                 .get("action", {}).get("linear_acc32", False)))
        if topology.num_cores != 8:
            torch.ops.rpu.adarms_set_execution_core_count(handle, topology.num_cores)
        torch.ops.rpu.adarms_set_weights(handle, *set_weights_args)
        validate_native_topology("adarms", handle, topology, intermediate_size, gate_w_list[0].dtype)
        # Shared physical KV offsets need not equal logical/suffix RoPE rows.
        torch.ops.rpu.adarms_set_chunk_envelope(handle, _max_seq, 0)

        # Task 4.3: stash cos/sin on model so
        # _install_fused_denoise_handle can reuse the already-allocated RPU
        # tensors without a second allocation.
        model._gemma_rope_cos = cos_cached
        model._gemma_rope_sin = sin_cached
        model._adarms_graph_enabled = graph_enabled
        model._adarms_w8a16_graph_enabled = w8a16_graph_enabled
        model._adarms_graph_capture_enabled = graph_capture_enabled
        model._rpu_cache = rpu_cache
        model._rpu_adarms_graph_cache = adarms_graph_cache
        model._rpu_lazy_init_checked = True
        model._rpu_required_attrs = required_attrs
        model._rpu_action_handle = handle
        model._rpu_action_retirement_state = resource
        model._rpu_action_handle_finalizer = handle_finalizer
        model.forward = types.MethodType(rpu_adarms_model_forward, model)

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
                for name in _ADARMS_RUNTIME_INSTALL_ATTRS:
                    model_state.pop(name, None)
                model_state.update(install_snapshot)
            else:
                model_state.update(
                    _rpu_action_handle=resource.handle,
                    _rpu_action_handle_finalizer=handle_finalizer,
                    _rpu_action_retirement_state=resource,
                    _rpu_adarms_graph_cache=adarms_graph_cache, _rpu_cache=rpu_cache)
        raise

    if old_finalizer is not None and getattr(old_finalizer, "alive", False):
        old_finalizer.detach()

    graph_note = ""
    if adarms_w8a16:
        graph_note = (
            ", W8A16 graph enabled"
            if w8a16_graph_enabled
            else ", W8A16 graph disabled by RPU_PI05_ADARMS_W8A16_GRAPH=0")
    _LOG.info("Patched AdaRMSModel (instance) with all-layers-once fused forward, "
              "handle=%d, num_layers=%d%s", handle, num_layers,
              graph_note)

    return handle
