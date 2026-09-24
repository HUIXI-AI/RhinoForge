"""Gemma2 all-layers-once patch for PaliGemma2."""
from __future__ import annotations

import types
import weakref
from typing import Any

import torch

from rpu_backend.api.cache import RPUCache
from rpu_backend.api.errors import RPUBackendError
from rpu_backend.api._execution import (
    bind_execution_session, execution_serialized, normalize_rpu_execution,
)
from rpu_backend.runtime.weights import NUM_CORES, convert_linear_weights_inplace

_MISSING = object()
_GEMMA2_EXECUTION_COMPONENTS = {"language_model": {"prefill": ("chunk_size",)}}
_INSTALL_STATE_ATTRS = (
    "_rpu_effective_num_kv_heads",
    "_rpu_gemma2_handle",
    "_rpu_gemma2_handle_finalizer",
    "_rpu_cache",
    "_rpu_lazy_init_checked",
    "_rpu_required_attrs",
    "_rpu_gemma2_had_instance_forward",
    "_rpu_gemma2_original_instance_forward",
    "_rpu_gemma2_install_ready",
    "_rpu_execution",
    "_rpu_last_execution_plan",
    "_execution_session",
    "_planner_cost_session",
    "_rpu_gemma2_graph_cache",
    "_fmb_execution_generation",
    "reconfigure_rpu_execution",
    "close_rpu_execution",
)


def _normalize_gemma2_execution(value):
    return normalize_rpu_execution(
        value, entry_point="PaliGemma2 diagnostic language component",
        supported_components=_GEMMA2_EXECUTION_COMPONENTS,
    )


def _apply_gemma2_execution(model, old, new, generation, *, force_rebuild=False):
    cache = model._rpu_gemma2_graph_cache
    cache.begin_warmup()
    cache.clear()
    model._rpu_cache.reset_to_position(0)
    vars(model).pop("_rpu_last_execution_plan", None)
    vars(model)["_fmb_execution_generation"] = int(generation)


def _reconfigure_gemma2_execution(model, value):
    return model._execution_session.reconfigure(value)


def _close_gemma2_execution(model):
    from rpu_backend.adapters.paligemma import _cleanup_gemma2_handle

    _cleanup_gemma2_handle(model)


def _gemma2_execution_stage(seq_len, position):
    return "decode" if int(seq_len) == 1 and int(position) > 0 else "prefill"


def _plan_gemma2_execution(model, *, seq_len, position, max_seq_len):
    from rpu_backend.runtime.decoder import plan_bounded_prefill_execution

    from rpu_backend.api._execution import resolve_component_rpu_execution

    component = resolve_component_rpu_execution(
        model._rpu_execution, "language_model", entry_point="Gemma2 diagnostic planner",
        supported_components=_GEMMA2_EXECUTION_COMPONENTS,
        profile_auto={"prefill": {"chunk_size": "auto"}},
    )
    stage = _gemma2_execution_stage(seq_len, position)
    requested = component.get("prefill", {}).get("chunk_size", "auto") if stage == "prefill" else "auto"
    exact = None if requested == "auto" else int(requested)
    handle = int(model._rpu_gemma2_handle)
    plans = []
    plan_bounded_prefill_execution(
        int(seq_len), int(max_seq_len) - int(position), 0,
        execution_owner=model, execution_component="language_model",
        execution_stage=stage, execution_native=("gemma2", handle),
        position=int(position), padding_rows=0, exact_chunk_size=exact,
        resolve_stage_domain=lambda length: torch.ops.rpu.gemma2_resolve_prefill_stage_domain(
            handle, int(length), int(position), 0 if exact is None else exact
        ),
        request_id=f"paligemma2:language_model:{stage}:diagnostic",
        graph_mode="COMPOSITE_CHILD", queue_owner_id=handle,
        physical_metadata=(("execution_generation", int(model._fmb_execution_generation)),),
        plan_result_sink=plans.append,
        plan_signature=(),
        graph_cache=model._rpu_gemma2_graph_cache,
    )
    planned = plans[0]
    if planned.selected is None or not planned.selected.stage_tuple.physical_descriptor:
        raise RuntimeError("Gemma2 diagnostic planner returned no physical descriptor")
    return planned


def _gemma2_destroy_handle(handle: int) -> None:
    try:
        torch.ops.rpu.gemma2_destroy(handle)
    except Exception:
        pass


def _configure_gemma2_handle(weight_args: tuple[Any, ...], max_kv_len: int) -> int:
    """Create/configure one raw handle; destroy it on every failed setter."""
    handle = int(torch.ops.rpu.gemma2_create())
    configured = False
    try:
        torch.ops.rpu.gemma2_set_weights(handle, *weight_args)
        torch.ops.rpu.gemma2_set_chunk_envelope(handle, max_kv_len, 0)
        configured = True
        return handle
    finally:
        if not configured:
            _gemma2_destroy_handle(handle)


def _restore_attrs(model: Any, snapshot: dict[str, Any]) -> None:
    # Rollback may follow a failure in model-defined attribute hooks. Restore
    # plain runtime metadata directly so those hooks cannot mask the original
    # exception or leave a half-published native owner behind.
    model_vars = vars(model)
    for attr, value in snapshot.items():
        if value is _MISSING:
            model_vars.pop(attr, None)
        else:
            model_vars[attr] = value


def _restore_instance_forward(
    model: Any, *, had_instance_forward: bool, original_forward: Any
) -> None:
    model_vars = vars(model)
    if had_instance_forward:
        model_vars["forward"] = original_forward
    else:
        model_vars.pop("forward", None)


def _layer_type_ids(layer_types: list[str]) -> list[int]:
    mapping = {
        "sliding_attention": 0,
        "full_attention": 1,
    }
    ids: list[int] = []
    for value in layer_types:
        try:
            ids.append(mapping[value])
        except KeyError as exc:
            raise ValueError(
                f"unknown Gemma2 layer type {value!r}; expected one of "
                f"{sorted(mapping)}"
            ) from exc
    return ids


def _require_attr(obj: Any, attr: str, owner: str) -> Any:
    if not hasattr(obj, attr):
        raise RPUBackendError(f"Gemma2 RPU patch requires {owner}.{attr}")
    return getattr(obj, attr)


def _gemma2_norm_gamma(norm: torch.nn.Module) -> torch.Tensor:
    """Translate the real norm's stored parameter into native multiplicative gamma."""
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm

    if type(norm) is Gemma2RMSNorm:
        gamma = norm.weight.detach().float() + 1.0
    elif type(norm) is torch.nn.RMSNorm:
        gamma = norm.weight.detach().float().clone()
    else:
        raise RPUBackendError(
            "Gemma2 norm requires actual Gemma2RMSNorm offset weights or torch.nn.RMSNorm gamma weights"
        )
    if gamma.ndim != 1 or not torch.isfinite(gamma).all():
        raise RPUBackendError("Gemma2 norm gamma must be one-dimensional and finite")
    gamma = gamma.to(dtype=torch.float16).contiguous()
    if not torch.isfinite(gamma).all():
        raise RPUBackendError("Gemma2 norm gamma must remain finite in FP16")
    return gamma


def _validate_gemma2_layer(layer: Any, layer_idx: int) -> None:
    prefix = f"model.layers[{layer_idx}]"
    attn = _require_attr(layer, "self_attn", prefix)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        proj = _require_attr(attn, name, f"{prefix}.self_attn")
        _require_attr(proj, "weight", f"{prefix}.self_attn.{name}")
    for name in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ):
        norm = _require_attr(layer, name, prefix)
        _require_attr(norm, "weight", f"{prefix}.{name}")
    mlp = _require_attr(layer, "mlp", prefix)
    for name in ("gate_proj", "up_proj", "down_proj"):
        proj = _require_attr(mlp, name, f"{prefix}.mlp")
        _require_attr(proj, "weight", f"{prefix}.mlp.{name}")


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _validate_gemma2_cache(
    cache: Any,
    *,
    config: Any,
    batch_size: int,
    attn_tp: int,
    effective_num_kv_heads: int | None = None,
) -> RPUCache:
    if not isinstance(cache, RPUCache):
        raise RPUBackendError(
            "Gemma2 RPU path requires past_key_values to be an RPUCache "
            f"instance, got {type(cache).__name__}"
        )

    expected_num_layers = int(config.num_hidden_layers)
    expected_num_kv_heads = (
        int(effective_num_kv_heads)
        if effective_num_kv_heads is not None
        else int(config.num_key_value_heads)
    )
    expected_head_dim = int(config.head_dim)
    expected_batch_size = int(batch_size)
    expected_attn_tp = int(attn_tp)
    if expected_attn_tp <= 0:
        raise RPUBackendError(
            "Gemma2 RPU cache layout requires positive attn_tp; got "
            f"{expected_attn_tp}"
        )

    effective_kv_slots = NUM_CORES * expected_num_kv_heads // expected_attn_tp
    if effective_kv_slots <= 0:
        raise RPUBackendError(
            "Gemma2 RPU cache layout requires attn_tp to leave at least one "
            f"effective KV slot; got attn_tp={expected_attn_tp}, "
            f"num_kv_heads={expected_num_kv_heads}, NUM_CORES={NUM_CORES}"
        )
    expected_nKVHeadChunk = min(NUM_CORES, effective_kv_slots)
    expected_nKVHeadVx = _ceil_div(effective_kv_slots, expected_nKVHeadChunk)

    mismatches: list[str] = []
    if int(cache.num_layers) != expected_num_layers:
        mismatches.append(
            f"num_layers={cache.num_layers} expected {expected_num_layers}"
        )
    if int(cache.batch_size) != expected_batch_size:
        mismatches.append(
            f"batch_size={cache.batch_size} expected {expected_batch_size}"
        )
    if int(cache.num_kv_heads) != expected_num_kv_heads:
        mismatches.append(
            f"num_kv_heads={cache.num_kv_heads} expected {expected_num_kv_heads}"
        )
    if int(cache.head_dim) != expected_head_dim:
        mismatches.append(f"head_dim={cache.head_dim} expected {expected_head_dim}")
    if int(cache.nKVHeadChunk) != expected_nKVHeadChunk:
        mismatches.append(
            f"nKVHeadChunk={cache.nKVHeadChunk} expected {expected_nKVHeadChunk}"
        )
    if int(cache.nKVHeadVx) != expected_nKVHeadVx:
        mismatches.append(
            f"nKVHeadVx={cache.nKVHeadVx} expected {expected_nKVHeadVx}"
        )
    if mismatches:
        raise RPUBackendError(
            "Gemma2 RPU cache layout mismatch for attn_tp="
            f"{expected_attn_tp}; create RPUCache with the adapter's "
            f"_rpu_attn_tp. Mismatches: {', '.join(mismatches)}"
        )
    return cache


def _select_gemma2_attention_mask(
    attention_mask: Any,
    *,
    query_len: int,
    key_value_len: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None

    selected = attention_mask
    if isinstance(attention_mask, dict):
        selected = None
        for key in ("full_attention", "sliding_attention"):
            value = attention_mask.get(key)
            if torch.is_tensor(value):
                selected = value
                break
        if selected is None:
            for value in attention_mask.values():
                if torch.is_tensor(value):
                    selected = value
                    break
        if selected is None:
            raise RPUBackendError(
                "Gemma2 RPU requires additive attention mask tensor; "
                "attention_mask dict contains no tensor values"
            )

    if not torch.is_tensor(selected):
        raise RPUBackendError(
            "Gemma2 RPU requires additive attention mask tensor or dict of "
            f"tensors, got {type(selected).__name__}"
        )

    dims = selected.dim()
    shape = tuple(selected.shape)
    expected_tail = (int(query_len), int(key_value_len))
    if dims not in (2, 3, 4):
        raise RPUBackendError(
            "Gemma2 RPU requires additive attention mask with rank 2, 3, or "
            f"4 and trailing shape {expected_tail}; got shape {shape}"
        )
    if tuple(shape[-2:]) != expected_tail:
        raise RPUBackendError(
            "Gemma2 RPU requires additive attention mask with trailing shape "
            f"{expected_tail}; got shape {shape}. Batch padding masks such "
            "as [batch, seq] must be converted before the Gemma2 RPU path."
        )

    selected = selected.to(dtype=torch.float16).cpu().contiguous()
    if bool((selected > 0).any().item()):
        raise RPUBackendError(
            "Gemma2 RPU requires additive attention mask values <= 0. "
            "Batch padding masks such as [batch, seq] with positive keep "
            "values must be converted before the Gemma2 RPU path."
        )
    return selected


def patch_gemma2_model_for_rpu_all_layers_once(model: Any) -> int:
    existing_handle = getattr(model, "_rpu_gemma2_handle", None)
    install_ready = getattr(model, "_rpu_gemma2_install_ready", False) is True
    if install_ready:
        finalizer = getattr(model, "_rpu_gemma2_handle_finalizer", None)
        installed_forward = vars(model).get("forward")
        forward_function = getattr(installed_forward, "__func__", None)
        if (
            existing_handle is not None
            and finalizer is not None
            and getattr(finalizer, "alive", False)
            and getattr(model, "_rpu_cache", None) is not None
            and getattr(model, "_rpu_gemma2_graph_cache", None) is not None
            and getattr(model, "_execution_session", None) is not None
            and getattr(model, "_rpu_lazy_init_checked", False) is True
            and getattr(installed_forward, "__self__", None) is model
            and getattr(forward_function, "__name__", None)
            == "rpu_gemma2_model_forward"
        ):
            return int(existing_handle)
        raise RPUBackendError(
            "Gemma2 RPU ready marker is incomplete; reload the model instead "
            "of retrying a broken install."
        )
    if getattr(model, "_rpu_gemma2_install_started", False) or existing_handle is not None:
        raise RPUBackendError(
            "Gemma2 RPU installation previously started but did not complete; "
            "reload the model to avoid double-swizzling weights."
        )

    from transformers.modeling_outputs import BaseModelOutputWithPast
    execution_config = _normalize_gemma2_execution(getattr(model, "_rpu_execution", None))

    config = _require_attr(model, "config", "model")
    layers = list(_require_attr(model, "layers", "model"))
    if not layers:
        raise RPUBackendError("Gemma2 RPU patch requires at least one layer")
    _require_attr(model, "embed_tokens", "model")
    norm = _require_attr(model, "norm", "model")
    _require_attr(norm, "weight", "model.norm")
    for name in (
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_size",
        "intermediate_size",
        "rms_norm_eps",
        "query_pre_attn_scalar",
        "attn_logit_softcapping",
        "max_position_embeddings",
        "layer_types",
    ):
        _require_attr(config, name, "model.config")
    layer_type_ids = _layer_type_ids(list(config.layer_types))
    for layer_idx, layer in enumerate(layers):
        _validate_gemma2_layer(layer, layer_idx)
    # Inspect the actual source classes before irreversible conversion. HF
    # stores offsets, while native RMSNorm consumes gamma; never mutate the
    # source Parameter or add one again when an owner is revisited.
    norm_gammas = {id(norm): _gemma2_norm_gamma(norm)}
    for layer in layers:
        for name in (
            "input_layernorm", "post_attention_layernorm",
            "pre_feedforward_layernorm", "post_feedforward_layernorm",
        ):
            layer_norm = getattr(layer, name)
            norm_gammas[id(layer_norm)] = _gemma2_norm_gamma(layer_norm)

    rotary = getattr(model, "rotary_emb", None)
    if rotary is None:
        rotary = getattr(layers[0].self_attn, "rotary_emb", None)
    if rotary is None:
        raise RPUBackendError("Gemma2 RPU patch could not locate rotary embedding")

    cfg_num_kv_heads = int(config.num_key_value_heads)
    # PaliGemma2 remains a downgraded recertification scaffold. Keep its native
    # GQA geometry unchanged; the abandoned KV-replication experiment must not
    # mutate checkpoint Parameters behind an unsupported public entry point.
    effective_num_kv_heads = cfg_num_kv_heads
    # Everything below may replace Parameters or allocate RPU tensors. A
    # partial failure is not retryable on the same model instance.
    model._rpu_gemma2_install_started = True

    convert_linear_weights_inplace(
        model,
        attn_num_cores=min(NUM_CORES, effective_num_kv_heads),
    )

    # H-NEW7 fix: explicit .to('rpu') when building per-layer weight lists.
    # Otherwise the C++ handle captures at::Tensor refs to the current device
    # (CPU at patch-time inside .to('rpu')), and the later decoder.to('rpu') in
    # the parent caller swaps Parameter.data to new RPU tensors -- leaving the
    # C++ refs pointing at the now-stale CPU tensors. Mirrors siglip.py:350-369.
    _w_to_rpu = lambda w: w.to(dtype=torch.float16, device="rpu")
    q_w_list = [_w_to_rpu(layer.self_attn.q_proj.weight) for layer in layers]
    k_w_list = [_w_to_rpu(layer.self_attn.k_proj.weight) for layer in layers]
    v_w_list = [_w_to_rpu(layer.self_attn.v_proj.weight) for layer in layers]
    o_w_list = [_w_to_rpu(layer.self_attn.o_proj.weight) for layer in layers]
    input_norm_list = [_w_to_rpu(norm_gammas[id(layer.input_layernorm)]) for layer in layers]
    post_attention_norm_list = [
        _w_to_rpu(norm_gammas[id(layer.post_attention_layernorm)]) for layer in layers
    ]
    pre_feedforward_norm_list = [
        _w_to_rpu(norm_gammas[id(layer.pre_feedforward_layernorm)]) for layer in layers
    ]
    post_feedforward_norm_list = [
        _w_to_rpu(norm_gammas[id(layer.post_feedforward_layernorm)]) for layer in layers
    ]
    gate_list = [_w_to_rpu(layer.mlp.gate_proj.weight) for layer in layers]
    up_list = [_w_to_rpu(layer.mlp.up_proj.weight) for layer in layers]
    down_list = [_w_to_rpu(layer.mlp.down_proj.weight) for layer in layers]

    max_position_embeddings = int(config.max_position_embeddings)
    dummy = torch.zeros(
        1,
        1,
        int(config.hidden_size),
        dtype=torch.float16,
        device="rpu",
    )
    positions = torch.arange(max_position_embeddings, device="rpu").unsqueeze(0)
    cos, sin = rotary(dummy, positions)
    if cos.dim() == 3:
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    cos = cos.to(dtype=torch.float16, device="rpu").contiguous()
    sin = sin.to(dtype=torch.float16, device="rpu").contiguous()
    final_norm_w = _w_to_rpu(norm_gammas[id(norm)]).contiguous()

    # Preserve instance-level forward state so any later publication/freeze
    # failure can restore the Python object exactly. Class-level forward is
    # restored by deleting the temporary instance override.
    instance_dict = vars(model)
    had_instance_forward = "forward" in instance_dict
    original_instance_forward = instance_dict.get("forward")

    @execution_serialized
    def rpu_gemma2_model_forward(
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
        **kwargs,
    ):
        if output_attentions:
            raise AssertionError("Gemma2 RPU path does not support output_attentions")
        if output_hidden_states:
            raise AssertionError(
                "Gemma2 RPU path does not support output_hidden_states"
            )
        if use_cache is False:
            raise AssertionError("Gemma2 RPU path requires use_cache=True")
        if input_ids is not None and inputs_embeds is not None:
            raise AssertionError(
                "Gemma2 RPU path requires exactly one of input_ids or inputs_embeds"
            )
        if input_ids is None and inputs_embeds is None:
            raise AssertionError(
                "Gemma2 RPU path requires exactly one of input_ids or inputs_embeds"
            )
        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids.cpu())
            inputs_embeds = inputs_embeds.to(dtype=torch.float16, device="rpu")

        hidden_states = inputs_embeds.to(dtype=torch.float16, device="rpu").contiguous()
        hidden_states = hidden_states * torch.tensor(
            float(self.config.hidden_size) ** 0.5,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        seq_len = hidden_states.shape[1]
        eff_num_kv_heads = int(
            getattr(
                self,
                "_rpu_effective_num_kv_heads",
                int(self.config.num_key_value_heads),
            )
        )
        attn_tp = int(
            getattr(
                self,
                "_rpu_attn_tp",
                min(NUM_CORES, eff_num_kv_heads),
            )
        )

        if past_key_values is None:
            # P7.1h L1: _rpu_cache eager init in patch 末尾 (见 line ~554 下方),
            # 不再 forward 内 lazy。Dynamo trace 时 attr 已存在,不会烘 hasattr
            # guard,Call 2+ 才能 replay 命中。详见 docs/graph_rules.md §6.6。
            self._rpu_cache.reset_to_position(0)
            past_key_values = self._rpu_cache
        past_key_values = _validate_gemma2_cache(
            past_key_values,
            config=self.config,
            batch_size=hidden_states.shape[0],
            attn_tp=attn_tp,
            effective_num_kv_heads=eff_num_kv_heads,
        )

        if cache_position is not None:
            position = int(cache_position.reshape(-1)[0].item())
        else:
            position = int(past_key_values.position)
        selected_attention_mask = _select_gemma2_attention_mask(
            attention_mask,
            query_len=seq_len,
            key_value_len=position + seq_len,
        )

        planned = _plan_gemma2_execution(
            self, seq_len=seq_len, position=position,
            max_seq_len=past_key_values.max_seq_len,
        )
        selected = planned.selected
        from rpu_backend.graph import GraphSignature

        signature = GraphSignature(
            op_id="gemma2:diagnostic:forward", shapes=[int(value) for value in hidden_states.shape],
            dtypes=[hidden_states.dtype],
            dyn_dims=[int(position), int(selected_attention_mask is not None), *planned.graph_key_words()],
        )
        try:
            with self._rpu_gemma2_graph_cache.capture(signature):
                output = torch.ops.rpu.gemma2_forward(
                    self._rpu_gemma2_handle, hidden_states,
                    past_key_values.k_caches, past_key_values.v_caches,
                    selected_attention_mask, position, True, 0,
                    selected.stage_tuple.physical_descriptor,
                )
        finally:
            torch.ops.rpu.spm_alloc_reset_temporary()
        past_key_values.update_position(seq_len)
        receipt = planned.as_dict(include_candidates=False)
        receipt.update(
            stage=_gemma2_execution_stage(seq_len, position),
            component="language_model", graph_mode="COMPOSITE_CHILD",
            generation=int(self._fmb_execution_generation),
            physical_descriptor=list(selected.stage_tuple.physical_descriptor),
        )
        vars(self)["_rpu_last_execution_plan"] = receipt

        return BaseModelOutputWithPast(
            last_hidden_state=output,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    # Prepare every Python owner before native creation. This avoids a live
    # handle with no cache/forward owner when allocation fails.
    max_seq = int(getattr(model, "_rpu_max_seq_len", 4096))
    attn_tp_default = int(
        getattr(model, "_rpu_attn_tp", min(NUM_CORES, effective_num_kv_heads))
    )
    cache = RPUCache(
        num_layers=int(config.num_hidden_layers),
        batch_size=1,
        max_seq_len=max_seq,
        num_kv_heads=effective_num_kv_heads,
        head_dim=int(config.head_dim),
        attn_tp=attn_tp_default,
    )
    weight_args = (
        q_w_list,
        k_w_list,
        v_w_list,
        o_w_list,
        input_norm_list,
        post_attention_norm_list,
        pre_feedforward_norm_list,
        post_feedforward_norm_list,
        gate_list,
        up_list,
        down_list,
        cos,
        sin,
        final_norm_w,
        layer_type_ids,
        int(config.num_attention_heads),
        int(effective_num_kv_heads),
        int(config.head_dim),
        int(config.hidden_size),
        int(config.intermediate_size),
        float(config.rms_norm_eps),
        float(config.query_pre_attn_scalar),
    )
    snapshot = {
        attr: getattr(model, attr, _MISSING)
        for attr in _INSTALL_STATE_ATTRS
    }

    from rpu_backend.graph import GraphCache

    graph_cache = GraphCache()
    handle: int | None = None
    finalizer = None
    session = None
    try:
        # Gemma2's one-based RoPE requires position + length < table rows.
        handle = _configure_gemma2_handle(weight_args, min(max_seq, int(cos.shape[0]) - 1))
        finalizer = weakref.finalize(model, _gemma2_destroy_handle, handle)
        model._rpu_effective_num_kv_heads = int(effective_num_kv_heads)
        model._rpu_gemma2_handle = handle
        model._rpu_gemma2_handle_finalizer = finalizer
        model._rpu_cache = cache
        model._rpu_lazy_init_checked = True
        model._rpu_required_attrs = (
            "_rpu_cache",
            "_rpu_gemma2_handle",
            "_rpu_gemma2_handle_finalizer",
            "_rpu_gemma2_graph_cache",
        )
        model._rpu_gemma2_had_instance_forward = had_instance_forward
        model._rpu_gemma2_original_instance_forward = original_instance_forward
        model._rpu_execution = execution_config
        model._rpu_gemma2_graph_cache = graph_cache
        model._fmb_execution_generation = 0
        model.forward = types.MethodType(rpu_gemma2_model_forward, model)
        from rpu_backend.graph.lazy_init_guard import _verify_lazy_init
        _verify_lazy_init(model)
        session = bind_execution_session(
            model, execution_config, entry_point="PaliGemma2 diagnostic language component",
            supported_components=_GEMMA2_EXECUTION_COMPONENTS, validate=_normalize_gemma2_execution,
            apply=lambda old, new, generation, *, force_rebuild=False: _apply_gemma2_execution(
                model, old, new, generation, force_rebuild=force_rebuild
            ),
            rollback=lambda old, new, generation: _apply_gemma2_execution(model, new, old, generation),
            graph_mode="COMPOSITE_CHILD",
        )
        session._bind_planner_owner(model, "language_model")
        model.reconfigure_rpu_execution = types.MethodType(_reconfigure_gemma2_execution, model)
        model.close_rpu_execution = types.MethodType(_close_gemma2_execution, model)
        model._rpu_gemma2_install_ready = True
    except BaseException:
        if session is not None:
            session.close()
        if finalizer is not None and getattr(finalizer, "alive", False):
            finalizer()
        elif finalizer is None and handle is not None:
            _gemma2_destroy_handle(handle)
        _restore_instance_forward(
            model,
            had_instance_forward=had_instance_forward,
            original_forward=original_instance_forward,
        )
        _restore_attrs(model, snapshot)
        raise
    return handle
