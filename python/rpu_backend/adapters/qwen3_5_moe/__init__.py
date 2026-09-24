"""Exact Qwen3.5-35B-A3B mixed-E4M3 fixed-core adapter."""
from __future__ import annotations

import threading
import types

import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeCausalLMOutputWithPast,
)

from rpu_backend.api._execution import (
    QWEN3_5_OPTIONAL_PADDING_CAP,
    QWEN3_5_TOTAL_PADDING_CAP,
    _resolve_text_install_options,
    bind_execution_session,
    bind_rpu_execution,
    execution_serialized,
    normalize_rpu_execution,
)
from rpu_backend.api.causal_lm import _claim_live_instance
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError
from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache
from rpu_backend.runtime.device import extract_to_device_target, is_rpu_device_target
from rpu_backend.runtime.registry import register_adapter
from rpu_backend.runtime.weights import transform_linear_weight

from rpu_backend.adapters.qwen3_5.vision import (
    QWEN3_5_MOE_35B_VISION_PROFILE,
    QWEN3_5_MOE_35B_TP4_VISION_PROFILE,
    QWEN3_5_MOE_35B_TP6_VISION_PROFILE,
    _prepare_qwen3_5_multimodal_spm,
    _resolve_vision_install_options,
    _rollback_qwen3_5_vision_install,
    make_qwen3_5_moe_vision_text_bridge,
)
from .cores import core_topology
from .loader import load_qwen3_5_moe_model
from .quant_scope import EXACT_PROFILE_ID, validate_exact_profile
from .text import (
    CERTIFIED_CHUNK_ENVELOPE,
    _bind_qwen35_moe_retirement,
    _clear_qwen35_moe_graphs,
    _preflight_qwen3_5_moe_text,
    _qwen35_moe_retirement_ready,
    _retire_failed_qwen3_5_moe_text_install,
    _retire_qwen3_5_moe_text_state,
    _text_config,
    _validate_layer_types,
    _validate_qwen35_moe_retirement_children,
    install_qwen3_5_moe_text_for_rpu,
    layer_types_to_is_full,
    run_qwen3_5_moe_text,
)
from .vision import fuse_visual_embeds


_SUPPORTED_PROFILES = frozenset({
    (2048, 512, 512, 40, 16, 2, 256, 256, 8),
})
_SWIZZLE_LOCK = threading.Lock()
_FAILED_MODEL_RETIREMENTS = []


def _config_profile(config):
    tc = _text_config(config)
    return (
        int(tc.hidden_size), int(tc.moe_intermediate_size),
        int(tc.shared_expert_intermediate_size), int(tc.num_hidden_layers),
        int(tc.num_attention_heads), int(tc.num_key_value_heads),
        int(tc.head_dim), int(tc.num_experts), int(tc.num_experts_per_tok),
    )


def _check_profile(config) -> None:
    try:
        validate_exact_profile(config, require_quant=True)
    except (TypeError, ValueError) as exc:
        raise UnsupportedModelError(str(exc)) from exc
    profile = _config_profile(config)
    if profile not in _SUPPORTED_PROFILES:
        raise UnsupportedModelError(
            f"Qwen3.5-MoE profile {profile} is not the exact 35B-A3B geometry"
        )


def _model_parts(model):
    fusion = getattr(model, "model", model)
    inner = getattr(fusion, "language_model", fusion)
    vision = getattr(fusion, "visual", None)
    return fusion, inner, vision


def _close_qwen3_5_moe_model(model, *, primary_error=None) -> None:
    """Retire Vision then text under the exact top-level Session."""
    from rpu_backend.api import causal_lm
    from rpu_backend.api._execution import ExecutionSession, _mark_execution_process_unsafe

    _fusion, inner, vision = _model_parts(model)
    state = getattr(inner, "_rpu_qwen3_5_moe", None)
    session = getattr(model, "_execution_session", None)

    def has_handles():
        adopted = any(
            item[2].handle is not None
            for item in vars(model).get("_qwen35_retirement_children", ())
        )
        text_live = getattr(state, "handle", None) is not None
        vision_live = vision is not None and any(
            getattr(vision, name, None) is not None
            for name in ("_rpu_vision_handle", "_rpu_vision_installing_handle")
        )
        return adopted or text_live or vision_live

    def poison(error):
        if not getattr(model, "_rpu_qwen3_5_moe_retirement_failed", None):
            vars(model)["_rpu_qwen3_5_moe_retirement_failed"] = repr(error)
            _FAILED_MODEL_RETIREMENTS.append(model)
            if state is not None:
                state.retirement_failed = repr(error)
            if isinstance(session, ExecutionSession):
                session.poison()
            with causal_lm._LIVE_LOCK:
                if causal_lm._LIVE_TERMINAL_REASON is None:
                    live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                    if live is None:
                        live = model
                        causal_lm._claim_live_instance(live)
                    causal_lm._poison_live_instance(
                        live,
                        f"Qwen3.5-MoE model cleanup failed: {error!r}",
                        unsafe=True,
                    )
                _mark_execution_process_unsafe(
                    f"Qwen3.5-MoE model cleanup failed: {error!r}"
                )
            for item in (
                state,
                getattr(vision, "_rpu_vision_retirement_state", None),
            ):
                resource = getattr(item, "retirement", None)
                if resource is not None and resource.handle is not None:
                    resource.retain_failure(error, model)

    def retire():
        try:
            _clear_qwen35_moe_graphs((state,), vision)
            if vision is not None and any(
                name.startswith("_rpu_vision_") for name in vars(vision)
            ):
                _rollback_qwen3_5_vision_install(vision, _graphs_retired=True)
            _retire_qwen3_5_moe_text_state(
                inner, state=state, _graphs_retired=True
            )
        except BaseException as error:
            poison(error)
            raise

    try:
        if getattr(model, "_rpu_qwen3_5_moe_retirement_failed", None):
            raise RuntimeError(
                "Qwen3.5-MoE model retirement failed; restart the process"
            )
        if getattr(model, "_rpu_qwen3_5_moe_retired", False):
            return
        if not isinstance(session, ExecutionSession) or session._owner is not model:
            if not has_handles() and state is None:
                return
            error = RuntimeError(
                "Qwen3.5-MoE model close requires its actual top-level Session"
            )
            poison(error)
            raise error
        with session._lock, causal_lm._LIVE_LOCK:
            if session._active or session._shutting_down:
                raise RuntimeError(
                    "Qwen3.5-MoE model cannot close during active execution"
                )
            live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
            if live is not None and live is not model:
                raise RuntimeError("Qwen3.5-MoE retirement lost the live-owner slot")
            vision_state = getattr(vision, "_rpu_vision_retirement_state", None)
            _validate_qwen35_moe_retirement_children(
                model, ((inner, state), (vision, vision_state))
            )
            for child_state in (state, vision_state):
                if child_state is None or getattr(child_state, "handle", None) is None:
                    continue
                resource = getattr(child_state, "retirement", None)
                if (
                    resource is None
                    or resource.failed is not None
                    or resource.handle != child_state.handle
                    or resource.owner() is not child_state
                    or resource.parent is None
                    or resource.parent() is not model
                    or getattr(child_state, "_execution_session", None) is not session
                    or getattr(child_state, "_retirement_close", None)
                    is not _close_qwen3_5_moe_model
                ):
                    raise RuntimeError(
                        "Qwen3.5-MoE retirement child belongs to a different parent"
                    )
            session.shutdown(retire)
            if has_handles():
                error = RuntimeError(
                    "Qwen3.5-MoE closed Session still owns native resources"
                )
                poison(error)
                raise error
            vars(model)["_rpu_qwen3_5_moe_retired"] = True
            vars(model).pop("_qwen35_retirement_children", None)
            causal_lm._release_live_instance(model)
    except BaseException as cleanup_error:
        if primary_error is None:
            raise
        primary_error.add_note(
            f"Qwen3.5-MoE model cleanup also failed: {cleanup_error!r}"
        )


_QWEN3_5_MOE_TEXT_STATE_ATTRS = (
    "handle", "handle_finalizer", "retirement", "graph_cache",
    "prefill_graph", "max_seq_len", "num_layers", "hidden_size",
    "rotary_dim", "rope_theta", "mrope_section", "padding_budget",
    "padding_rows", "execution_config", "control_snapshot",
    "exact_chunk_size", "chunk_envelope", "execution_topology",
    "vision_control_options", "vision_graph_policy", "vision_arena_admitted",
    "execution_generation", "prepared_profile",
)


def _qwen3_5_moe_text_runtime_state(model):
    if getattr(model, "_rpu_swizzled", False) is not True:
        return None
    if getattr(model, "_rpu_swizzle_started", False) is not True:
        return None
    _fusion, inner, _vision = _model_parts(model)
    state = getattr(inner, "_rpu_qwen3_5_moe", None)
    if state is None or any(
        not hasattr(state, name) for name in _QWEN3_5_MOE_TEXT_STATE_ATTRS
    ):
        return None
    if not _qwen35_moe_retirement_ready(state):
        return None
    if getattr(model, "_lm_head_w_rpu_keepalive", None) is None:
        return None
    installed_forward = vars(model).get("forward")
    if not (
        getattr(installed_forward, "__self__", None) is model
        and getattr(installed_forward, "__func__", None)
        is _rpu_qwen3_5_moe_forward
    ):
        return None
    return state


def _sync_adapter_text_state(adapter, state) -> None:
    adapter._handle = state.handle
    adapter._graph_cache = state.graph_cache
    adapter._rpu_is_ready = True


def _cleanup_failed_install(
    adapter, model, inner, had_instance_forward, original_forward, claimed
) -> None:
    state = getattr(inner, "_rpu_qwen3_5_moe", None) if inner is not None else None
    if state is not None:
        _retire_failed_qwen3_5_moe_text_install(
            inner, state=state, session=getattr(model, "_execution_session", None)
        )
    model_state = vars(model)
    model_state.pop("_lm_head_w_rpu_keepalive", None)
    if had_instance_forward:
        model_state["forward"] = original_forward
    else:
        model_state.pop("forward", None)
    adapter._handle = None
    adapter._graph_cache = None
    adapter._rpu_is_ready = False
    if claimed and state is None:
        from rpu_backend.api.causal_lm import _release_live_instance
        _release_live_instance(model)


def _qwen3_5_moe_cold_config_only(adapter) -> bool:
    """Prove that changing the public config cannot race native state."""
    model = adapter.model
    _fusion, inner, _vision = _model_parts(model)
    return (
        getattr(model, "_rpu_swizzle_started", False) is not True
        and getattr(model, "_rpu_swizzled", False) is not True
        and getattr(inner, "_rpu_qwen3_5_moe", None) is None
    )


def _validate_qwen3_5_moe_execution_reconfigure(
    adapter, execution_config
) -> None:
    """Admit config changes only before this exact native model is installed."""
    adapter.preflight_execution(adapter.model.config, execution_config)
    if not _qwen3_5_moe_cold_config_only(adapter):
        raise ValueError(
            "Qwen3.5-35B-A3B execution controls are immutable after RPU "
            "installation; reload the model to use another configuration"
        )


class Qwen3_5MoeAdapter:
    """Public adapter for one exact mixed-E4M3 35B-A3B checkpoint."""

    _EXECUTION_SUPPORTED = {
        "model": ("num_cores",),
        "prefill": ("chunk_size", "padding_rows", "padding_budget", "linear_acc32"),
        "vision": (
            "chunk_size", "padding_rows", "padding_budget",
            "linear_acc32", "gelu_erf_ultra",
        ),
    }

    def __init__(self, model, *, rpu_execution=None):
        normalized = None
        if rpu_execution is not None:
            normalized = normalize_rpu_execution(
                rpu_execution,
                entry_point="Qwen3_5MoeAdapter",
                supported=self._EXECUTION_SUPPORTED,
            )
        self.model = model
        _check_profile(model.config)
        _validate_layer_types(model.config)
        if getattr(model, "_qwen3_5_moe_profile", None) != EXACT_PROFILE_ID:
            raise UnsupportedModelError(
                "Qwen3.5-35B-A3B must be loaded by "
                "Qwen3_5MoeAdapter.load_pretrained so raw E4M3 tensors retain "
                "their typed mode-5 storage"
            )
        self._rpu_execution = bind_rpu_execution(
            model,
            normalized if rpu_execution is not None else None,
            entry_point="Qwen3_5MoeAdapter",
            supported=self._EXECUTION_SUPPORTED,
        )
        self.preflight_execution(model.config, self._rpu_execution)
        session_callbacks = {}
        if getattr(model, "_execution_session", None) is None:
            session_callbacks = {
                "validate": lambda config: (
                    _validate_qwen3_5_moe_execution_reconfigure(self, config)
                ),
                "cold_config_only": lambda: (
                    _qwen3_5_moe_cold_config_only(self)
                ),
            }
        self._execution_session = bind_execution_session(
            model,
            self._rpu_execution,
            entry_point="Qwen3_5MoeAdapter",
            supported=self._EXECUTION_SUPPORTED,
            graph_mode="BOUNDED_ONESHOT",
            **session_callbacks,
        )
        self._execution_session.register_config_view(self)
        fusion, inner, vision = _model_parts(model)
        self._execution_session._bind_planner_owner(inner)
        if vision is not None:
            self._execution_session._bind_planner_owner(vision)
        state = _qwen3_5_moe_text_runtime_state(model)
        self._handle = getattr(state, "handle", None)
        self._graph_cache = getattr(state, "graph_cache", None)
        self._rpu_is_ready = state is not None

        if getattr(model.to, "__rpu_wrapped__", False):
            return
        original_to = model.to

        def rpu_aware_to(*args, **kwargs):
            target = extract_to_device_target(args, kwargs)
            if target is not None and is_rpu_device_target(
                target, entry_point="Qwen3_5MoeAdapter.model.to"
            ):
                if len(args) > 1 or set(kwargs) - {"device"}:
                    raise ValueError(
                        "model.to('rpu') takes no extra arguments; use "
                        "Qwen3_5MoeAdapter(model).to_rpu(max_seq_len=...)"
                    )
                return self.to_rpu()
            if (
                self._rpu_is_ready
                or getattr(model, "_rpu_swizzled", False)
                or getattr(model, "_rpu_swizzle_started", False)
            ):
                raise RPUBackendError(
                    "Qwen3.5-MoE weights are irreversibly swizzled for RPU; "
                    "reload the checkpoint to select another device"
                )
            return original_to(*args, **kwargs)

        rpu_aware_to.__rpu_wrapped__ = True
        model.to = rpu_aware_to

    @classmethod
    def preflight(cls, config) -> None:
        _check_profile(config)
        _validate_layer_types(config)

    @classmethod
    def preflight_execution(cls, config, execution_config) -> None:
        _check_profile(config)
        core_topology(execution_config)
        prefill = execution_config.get("prefill", {})
        for field, limit in (
            ("padding_budget", QWEN3_5_OPTIONAL_PADDING_CAP),
            ("padding_rows", QWEN3_5_TOTAL_PADDING_CAP),
        ):
            value = prefill.get(field)
            if isinstance(value, int) and value > limit:
                raise ValueError(
                    f"Qwen3.5-MoE prefill {field} exceeds {limit}: got {value}"
                )
        requested = prefill.get("chunk_size", "auto")
        if isinstance(requested, int):
            if requested % 64:
                raise ValueError(
                    "Qwen3.5-MoE prefill chunk_size must be a multiple of 64"
                )
            tc = _text_config(config)
            ceiling = CERTIFIED_CHUNK_ENVELOPE[
                (int(tc.num_hidden_layers), int(tc.hidden_size))
            ][1]
            if requested > ceiling:
                raise UnsupportedModelError(
                    f"Qwen3.5-MoE prefill chunk_size={requested} exceeds {ceiling}"
                )
        vision = execution_config.get("vision", {})
        if (
            isinstance(vision.get("padding_rows"), int)
            and vision["padding_rows"] != 0
        ) or int(vision.get("padding_budget", 0)) != 0:
            raise ValueError("Qwen3.5-MoE Vision does not admit padding")

    @classmethod
    def load_pretrained(cls, path, *, config=None, dtype=torch.float16, **hf_kwargs):
        return load_qwen3_5_moe_model(
            path, config=config, dtype=dtype, **hf_kwargs
        )

    def to_rpu(self, max_seq_len=8192):
        if type(max_seq_len) is not int or not 0 < max_seq_len <= 8192:
            raise ValueError(
                "Qwen3.5-35B-A3B max_seq_len must be an integer in [1, 8192]"
            )
        model = self.model
        state = _qwen3_5_moe_text_runtime_state(model)
        if state is not None:
            _sync_adapter_text_state(self, state)
            return model
        if self._rpu_is_ready or getattr(model, "_rpu_swizzled", False):
            self._handle = None
            self._graph_cache = None
            self._rpu_is_ready = False
            raise RPUBackendError(
                "Qwen3_5MoeAdapter ready marker exists without complete ownership; "
                "reload the model"
            )
        if getattr(model, "_rpu_swizzle_started", False):
            raise RPUBackendError(
                "a prior Qwen3.5-MoE install mutated weights; reload the model"
            )
        topology = core_topology(model._rpu_execution)
        resolved_text = _resolve_text_install_options(
            max_seq_len, execution_config=model._rpu_execution
        )
        resolved_vision = _resolve_vision_install_options(
            execution_config=model._rpu_execution
        )
        max_seq_len = resolved_text[0]
        if not _SWIZZLE_LOCK.acquire(blocking=False):
            raise RPUBackendError("another Qwen3.5-MoE swizzle is in progress")

        inner = None
        claimed = False
        had_instance_forward = "forward" in vars(model)
        original_forward = vars(model).get("forward")
        try:
            from rpu_backend.graph import GraphRuntimePolicy
            from rpu_backend.runtime.hw_attrs import (
                install_hw_attr_validator,
                validate_postinstall,
                validate_preinstall,
            )

            validate_preinstall(model)
            _claim_live_instance(model)
            claimed = True
            # Reuse released tensor storage for decode inputs and outputs instead
            # of creating new SDK mappings on every token. Claim the immutable
            # process policy before any graph arenas or weights are materialized.
            torch.rpu.set_caching_allocator(True)
            text_policy = GraphRuntimePolicy.from_environment(
                execution_core_count=topology.num_cores, graph_arena_count=4
            )
            if not text_policy.prepare_arenas():
                raise RPUBackendError(
                    "Qwen3.5-35B-A3B requires accessible hardware to prepare "
                    "its four-arena Graph policy before weight upload"
                )
            from dataclasses import replace
            vision_policy = (replace(text_policy, execution_core_count=topology.attn_tp)
                             if topology.attn_tp != topology.num_cores else text_policy)
            model._rpu_swizzle_started = True
            _fusion, inner, _vision = _model_parts(model)
            output = model.get_output_embeddings()
            if output is None:
                raise UnsupportedModelError("Qwen3.5-MoE requires an untied lm_head")
            if output.weight is inner.embed_tokens.weight:
                raise UnsupportedModelError("Qwen3.5-35B-A3B lm_head must be untied")
            for name, weight in (
                ("embed_tokens", inner.embed_tokens.weight),
                ("lm_head", output.weight),
            ):
                if weight.dtype != torch.float16 or tuple(weight.shape) != (248320, 2048):
                    raise UnsupportedModelError(
                        f"Qwen3.5-35B-A3B {name} must be FP16 [248320, 2048], "
                        f"got {weight.dtype} {tuple(weight.shape)}"
                    )

            handle = install_qwen3_5_moe_text_for_rpu(
                inner,
                model.config,
                max_seq_len,
                execution_config=model._rpu_execution,
                _resolved_options=resolved_text,
                _cpu_stage_weights=True,
                _graph_runtime_policy=text_policy,
            )
            self._handle = handle
            state = inner._rpu_qwen3_5_moe
            _bind_qwen35_moe_retirement(
                state, model, _close_qwen3_5_moe_model, inner
            )
            state.execution_generation = self._execution_session.generation
            state.vision_control_options = resolved_vision
            state.vision_graph_policy = vision_policy
            state.vision_arena_admitted = True
            state.prepared_profile = {
                4: QWEN3_5_MOE_35B_TP4_VISION_PROFILE,
                6: QWEN3_5_MOE_35B_TP6_VISION_PROFILE,
                8: QWEN3_5_MOE_35B_VISION_PROFILE,
            }[topology.num_cores]
            self._graph_cache = state.graph_cache

            model._lm_head_w_rpu_keepalive = transform_linear_weight(
                output.weight.detach().to("cpu").contiguous(), 1, topology.lm_head_tp
            ).to("rpu")
            inner.embed_tokens.to("rpu")
            model.forward = types.MethodType(_rpu_qwen3_5_moe_forward, model)
            validate_postinstall(model)
            install_hw_attr_validator(model)
            self._rpu_is_ready = True
            model._rpu_swizzled = True
            return model
        except BaseException as error:
            try:
                _cleanup_failed_install(
                    self, model, inner, had_instance_forward, original_forward, claimed
                )
            except BaseException as cleanup_error:
                error.add_note(
                    f"Qwen3.5-MoE outer install cleanup failed: {cleanup_error!r}"
                )
            raise
        finally:
            _SWIZZLE_LOCK.release()


@execution_serialized
def _rpu_qwen3_5_moe_forward(
    self,
    input_ids=None,
    inputs_embeds=None,
    past_key_values=None,
    attention_mask=None,
    position_ids=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    mm_token_type_ids=None,
    logits_to_keep=0,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    **kw,
):
    from rpu_backend.runtime.hw_attrs import mark_first_forward_done
    mark_first_forward_done(self)

    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Qwen3.5-MoE requires exactly one input source")
    if not isinstance(past_key_values, Qwen3_5MoeCache):
        raise TypeError(
            "Qwen3.5-MoE requires Qwen3_5MoeCache.from_model(model, ...)"
        )
    if labels is not None:
        raise NotImplementedError("Qwen3.5-MoE RPU is inference-only")
    if use_cache is False or return_dict is False:
        raise NotImplementedError("Qwen3.5-MoE requires cache and return_dict")
    if output_attentions or output_hidden_states:
        raise NotImplementedError(
            "Qwen3.5-MoE does not expose intermediate attention/hidden tensors"
        )
    if pixel_values_videos is not None or video_grid_thw is not None:
        raise NotImplementedError("Qwen3.5-MoE video inputs are not supported")
    if kw:
        raise TypeError(
            "unsupported Qwen3.5-MoE forward arguments: " + ", ".join(sorted(kw))
        )
    if type(logits_to_keep) is not int or logits_to_keep < 0:
        raise ValueError("logits_to_keep must be a non-negative integer")

    cache = past_key_values
    _fusion, inner, _vision = _model_parts(self)
    if pixel_values is not None:
        if input_ids is None or image_grid_thw is None or mm_token_type_ids is None:
            raise ValueError(
                "image prefill requires input_ids, image_grid_thw, and mm_token_type_ids"
            )
        if int(cache.position) != 0:
            raise NotImplementedError("images are accepted only at prefill position 0")
    if attention_mask is not None:
        mask_cpu = attention_mask.detach().to("cpu")
        if not bool(mask_cpu.bool().all().item()):
            raise NotImplementedError("Qwen3.5-MoE does not support padded attention")
        attention_mask = None

    request = inputs_embeds if inputs_embeds is not None else input_ids
    request_len = int(request.shape[1])
    if cache_position is not None:
        actual = cache_position.detach().to("cpu").reshape(-1).long()
        expected = torch.arange(
            int(cache.position), int(cache.position) + request_len
        )
        if not torch.equal(actual, expected):
            raise ValueError("cache_position does not match the live MoE cache")
    preflight_plan = _preflight_qwen3_5_moe_text(
        inner,
        cache,
        request_len,
        multimodal_prefill=pixel_values is not None,
    )
    if pixel_values is not None and _prepare_qwen3_5_multimodal_spm(
        self, input_ids, pixel_values, image_grid_thw, preflight_plan[0],
        _text_bridge=make_qwen3_5_moe_vision_text_bridge(self),
    ):
        # Vision's persistent buffers lower the text owner's available SPM.
        # Bind the final descriptor only after both owners reserve their floor.
        preflight_plan = _preflight_qwen3_5_moe_text(
            inner, cache, request_len, multimodal_prefill=True)

    if inputs_embeds is not None:
        hidden = inputs_embeds.to(device="rpu", dtype=torch.float16).contiguous()
    else:
        ids_rpu = input_ids if input_ids.device.type == "rpu" else input_ids.to("rpu")
        hidden = inner.embed_tokens(ids_rpu).to(dtype=torch.float16).contiguous()

    if pixel_values is None and int(cache.position) == 0:
        self.model.rope_deltas = None
        make_qwen3_5_moe_vision_text_bridge(self).set_mrope_position_delta(0)
    if pixel_values is not None:
        hidden, position_ids = fuse_visual_embeds(
            self,
            hidden,
            input_ids,
            pixel_values,
            image_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            past_key_values=cache,
        )

    out = run_qwen3_5_moe_text(
        inner,
        hidden,
        cache,
        attention_mask=attention_mask,
        position_ids=position_ids,
        multimodal_prefill=pixel_values is not None,
        preflight_plan=preflight_plan,
    )
    head_in = (
        out[:, -logits_to_keep:, :].contiguous()
        if logits_to_keep > 0
        else out
    )
    state = inner._rpu_qwen3_5_moe
    flat = head_in.reshape(-1, head_in.shape[-1]).contiguous()
    logits = torch.ops.rpu.linear_with_accumulation(
        flat, self._lm_head_w_rpu_keepalive, None, 1,
        state.execution_topology.lm_head_tp, state.linear_acc32)
    logits = logits.reshape(*head_in.shape[:-1], logits.shape[-1])
    logits = logits.to("cpu").to(torch.float32)
    return Qwen3_5MoeCausalLMOutputWithPast(
        loss=None,
        aux_loss=None,
        logits=logits,
        past_key_values=cache,
        hidden_states=None,
        attentions=None,
        router_logits=None,
        rope_deltas=getattr(self.model, "rope_deltas", None),
    )


register_adapter("Qwen3_5MoeForConditionalGeneration", Qwen3_5MoeAdapter)


__all__ = [
    "Qwen3_5MoeAdapter",
    "Qwen3_5MoeCache",
    "layer_types_to_is_full",
    "load_qwen3_5_moe_model",
]
