"""Qwen3.5 adapter — orchestration entry (== HF ``Qwen3_5ForConditionalGeneration``
+ the ``Qwen3_5Model`` multimodal-fusion level).

Qwen3.5 (``Qwen3_5ForConditionalGeneration``) is a VL model whose text backbone
interleaves full-attention and Gated-DeltaNet (GDN) linear layers per
``text_config.layer_types``. This package mirrors the shipping ``qwen3_vl/``
adapter: the HF hierarchy gets a home per level.

  ``Qwen3_5ForConditionalGeneration``   ← ``__init__.py``: top forward + lm_head + register
          ├── ``Qwen3_5VisionModel``      ← ``vision.py`` (controlled 2B/4B path)
          └── ``Qwen3_5Model`` (fusion)   ← ``__init__.py``: ``_rpu_qwen3_5_forward``
                    └── ``Qwen3_5TextModel``   ← ``text.py`` (text backbone)

Package layout (``adapters/qwen3_5/``, mirroring ``adapters/qwen3_vl/``):
``__init__.py`` (orchestration) + ``text.py`` (backbone) + ``vision.py`` (vision).

Backward-compat: this module keeps exporting ``Qwen3_5Adapter`` and
``layer_types_to_is_full`` (re-export), keeps ``_SUPPORTED_PROFILES`` mutable at
module scope (tests monkeypatch it), and ``Qwen3_5Adapter.to_rpu()`` keeps
``self._handle`` / ``self._graph_cache`` on the adapter instance. The text-only
forward is byte-for-byte identical to the pre-restructure single-file version.
"""
from __future__ import annotations

import os
import threading
import types
import weakref

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
)

from rpu_backend.api.causal_lm import _claim_live_instance, _release_live_instance
from rpu_backend.api._execution import (
    _QWEN3_5_EXECUTION_SUPPORTED,
    _validate_qwen35_text_padding,
    bind_execution_session,
    execution_serialized,
    native_execution_reconfigure,
)
from rpu_backend.runtime.device import extract_to_device_target, is_rpu_device_target
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError
from rpu_backend.api.qwen3_5_cache import (
    QWEN3_5_OPTIONAL_PADDING_CAP,
    QWEN3_5_TOTAL_PADDING_CAP,
    Qwen3_5Cache,
)
from rpu_backend.runtime.registry import register_adapter
from rpu_backend.runtime.weights import transform_linear_weight
from rpu_backend.adapters.qwen3_5.quant_scope import validate_legacy_text_profile

# Text backbone (== HF Qwen3_5TextModel). `layer_types_to_is_full` is re-exported
# here for backward-compat: ~8 tests import it from `adapters.qwen3_5`.
from .text import (  # noqa: F401
    _text_config,
    _resolve_text_install_options,
    _preflight_qwen3_5_text,
    _validate_text_install_options,
    _validate_layer_types,
    _retire_failed_qwen3_5_text_install,
    layer_types_to_is_full,
    install_qwen3_5_text_for_rpu,
    run_qwen3_5_text,
)
# Vision tower (== HF Qwen3_5VisionModel).
from .vision import (
    _prepare_qwen3_5_multimodal_spm,
    _resolve_vision_install_options,
    install_qwen3_5_vision_for_rpu,
    fuse_visual_embeds,
)


# (hidden_size, intermediate_size, num_hidden_layers, num_key_value_heads).
# Qwen3.5-0.8B uses head_dim=256 and num_attention_heads=8.
# Dense small models only — MoE rejected.
#
# NOTE: kept at module scope (NOT in text.py) — several tests widen it via
# `import rpu_backend.adapters.qwen3_5 as _adp; _adp._SUPPORTED_PROFILES |= {_p}`.
# `_check_profile` (below, same module) reads it as a global so the rebind is seen.
_SUPPORTED_PROFILES = frozenset({
    (1024, 3584, 24, 2),   # Qwen3.5-0.8B
    (2048, 6144, 24, 2),   # Qwen3.5-2B
    (2560, 9216, 32, 4),   # Qwen3.5-4B
    (4096, 12288, 32, 4),  # Qwen3.5-9B (untied lm_head)
    (5120, 17408, 64, 4),  # Controlled legacy W8A16 text-only; checked below.
})


_SWIZZLE_LOCK = threading.Lock()

# Failed explicit cleanup must keep every outstanding graph/tensor address
# alive, even if a caller drops its model after the propagated exception.
_FAILED_MODEL_RETIREMENTS = []


def _close_qwen3_5_model(model, *, primary_error=None) -> None:
    """Retire this complete model under its existing explicit/GC parent Session.

    A caller in a finally block can pass its active exception: retirement then
    annotates that error instead of hiding the original forward/build failure.
    """
    from rpu_backend.api import causal_lm
    from rpu_backend.api._execution import ExecutionSession
    from .text import _retire_qwen3_5_text_state
    from .vision import _rollback_qwen3_5_vision_install

    fusion = getattr(model, "model", model)
    inner = getattr(fusion, "language_model", fusion)
    state = getattr(inner, "_rpu_qwen3_5", None)
    vision = getattr(fusion, "visual", None)
    session = getattr(model, "_execution_session", None)

    def has_handles():
        return any(item[2].handle is not None for item in vars(model).get("_qwen35_retirement_children", ())) or getattr(state, "handle", None) is not None or any(
            getattr(vision, name, None) is not None
            for name in ("_rpu_vision_handle", "_rpu_vision_installing_handle"))

    def retire():
        try:
            from .text import _clear_qwen35_graphs

            _clear_qwen35_graphs((state,), vision)
            # Reverse installation order, only after every sibling Graph has
            # released its references to shared text/fusion buffers.
            if vision is not None and any(name.startswith("_rpu_vision_") for name in vars(vision)):
                _rollback_qwen3_5_vision_install(vision, _graphs_retired=True)
            _retire_qwen3_5_text_state(inner, state=state, _graphs_retired=True)
        except BaseException as error:
            poison(error)
            raise

    def poison(error):
        if not getattr(model, "_rpu_qwen3_5_retirement_failed", None):
            vars(model)["_rpu_qwen3_5_retirement_failed"] = repr(error)
            _FAILED_MODEL_RETIREMENTS.append(model)
            if state is not None:
                state.retirement_failed = repr(error)  # Disable the text raw finalizer too.
            if isinstance(session, ExecutionSession):
                session.poison()
            with causal_lm._LIVE_LOCK:
                if causal_lm._LIVE_TERMINAL_REASON is None:
                    live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                    if live is None:
                        live = model
                        causal_lm._claim_live_instance(live)
                    causal_lm._poison_live_instance(
                        live, f"Qwen3.5 model cleanup failed: {error!r}", unsafe=True)
                from rpu_backend.api._execution import _mark_execution_process_unsafe

                _mark_execution_process_unsafe(f"Qwen3.5 model cleanup failed: {error!r}")
            for item in (state, getattr(vision, "_rpu_vision_retirement_state", None)):
                resource = getattr(item, "retirement", None)
                if resource is not None and resource.handle is not None:
                    resource.retain_failure(error, model)
            for _child, _state, resource, _session in vars(model).get("_qwen35_retirement_children", ()):
                if resource.handle is not None:
                    resource.retain_failure(error, model)

    try:
        if getattr(model, "_rpu_qwen3_5_retirement_failed", None):
            raise RuntimeError("Qwen3.5 model retirement failed; restart the process")
        if getattr(model, "_rpu_qwen3_5_retired", False):
            return
        if getattr(state, "retirement_failed", None) or (
                causal_lm._LIVE_TERMINAL_REASON is not None and has_handles()):
            error = RuntimeError("Qwen3.5 model retirement blocked by an earlier unsafe cleanup")
            poison(error)
            raise error
        if not isinstance(session, ExecutionSession) or session._owner is not model:
            if not has_handles() and state is None:
                return  # CPU-only failure before the adapter bound its Session.
            error = RuntimeError("Qwen3.5 model close requires its actual top-level Session")
            poison(error)
            raise error
        with session._lock, causal_lm._LIVE_LOCK:
            # Reject re-entrant close without poisoning an otherwise live
            # forward. Every child we are about to mutate must belong to this
            # exact top-level Session, not an independently bound owner.
            if session._active or session._shutting_down:
                raise RuntimeError("Qwen3.5 model cannot close during active execution or retirement")
            try:
                live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                if live is not None and live is not model:
                    raise RuntimeError("Qwen3.5 retirement lost the live-owner slot")
                from .text import _validate_qwen35_retirement_children

                _validate_qwen35_retirement_children(model, (
                    (inner, state), (vision, getattr(vision, "_rpu_vision_retirement_state", None))))
                for child, present in ((inner, state is not None), (vision, vision is not None and any(
                        name.startswith("_rpu_vision_") for name in vars(vision)))):
                    if not present:
                        continue
                    binding = getattr(child, "_planner_cost_session", None)
                    child_state = (state if child is inner else
                                   getattr(child, "_rpu_vision_retirement_state", None))
                    resource = getattr(child_state, "retirement", None)
                    if getattr(child_state, "handle", None) is not None and (
                            resource is None or resource.failed is not None
                            or resource.handle != child_state.handle
                            or (resource.owner() is not None and resource.owner() is not child_state)
                            or resource.parent is None
                            or (resource.parent() is not None and resource.parent() is not model)
                            or getattr(child_state, "_execution_session", None) is not session
                            or getattr(child_state, "_retirement_close", None) is not _close_qwen3_5_model):
                        raise RuntimeError("Qwen3.5 retirement child belongs to a different parent")
                    if child is vision and getattr(vision, "_rpu_vision_handle", None) is not None and child_state is None:
                        raise RuntimeError("Qwen3.5 Vision retirement has no resource owner")
                    cleared_gc_binding = (
                        resource is not None and resource.owner() is None
                        and resource.parent is not None and resource.parent() is None
                        and getattr(child_state, "_execution_session", None) is session
                        and resource.handle == getattr(child_state, "handle", None))
                    if (not isinstance(binding, tuple) or len(binding) != 2
                            or not isinstance(binding[0], weakref.ReferenceType)
                            or (binding[0]() is not session and not (binding[0]() is None and cleared_gc_binding))
                            or getattr(child, "_execution_session", session) is not session):
                        raise RuntimeError("Qwen3.5 retirement child belongs to a different Session")
                session.shutdown(retire)
            except BaseException as error:
                poison(error)
                raise
            if has_handles():
                error = RuntimeError("Qwen3.5 closed Session still owns native resources")
                poison(error)
                raise error
            vars(model)["_rpu_qwen3_5_retired"] = True
            vars(model).pop("_qwen35_retirement_children", None)
            causal_lm._release_live_instance(model)
    except BaseException as cleanup_error:
        if primary_error is None:
            raise
        primary_error.add_note(f"Qwen3.5 model cleanup also failed: {cleanup_error!r}")


_QWEN3_5_TEXT_STATE_ATTRS = (
    "handle",
    "handle_finalizer",
    "graph_cache",
    "max_seq_len",
    "num_layers",
    "hidden_size",
    "rotary_dim",
    "rope_theta",
    "mrope_section",
    "padding_budget",
    "padding_rows",
    "execution_config",
    "control_snapshot",
    "exact_chunk_size",
    "chunk_envelope",
    "vision_control_options",
    "last_a6_plan",
    "prefill_graph",
)


def _qwen3_5_text_runtime_state(model):
    """Return the complete live text state, or ``None`` for partial state."""
    if getattr(model, "_rpu_swizzled", False) is not True:
        return None
    if getattr(model, "_rpu_swizzle_started", False) is not True:
        return None
    fusion = getattr(model, "model", None)
    if fusion is None:
        return None
    inner = getattr(fusion, "language_model", fusion)
    state = getattr(inner, "_rpu_qwen3_5", None)
    if state is None or getattr(state, "retirement_failed", None) or any(
        not hasattr(state, name) for name in _QWEN3_5_TEXT_STATE_ATTRS
    ):
        return None
    finalizer = state.handle_finalizer
    if state.handle is None or finalizer is None:
        return None
    from .text import _qwen35_retirement_ready

    if not _qwen35_retirement_ready(state):
        return None
    if state.graph_cache is None or state.prefill_graph is None:
        return None
    if getattr(inner, "_rpu_qwen3_5_text_install_started", False) is not True:
        return None
    if getattr(model, "_lm_head_w_rpu_keepalive", None) is None:
        return None
    installed_forward = vars(model).get("forward")
    if not (
        getattr(installed_forward, "__self__", None) is model
        and getattr(installed_forward, "__func__", None)
        is _rpu_qwen3_5_forward
    ):
        return None
    return state


def _sync_adapter_text_state(adapter, state) -> None:
    adapter._handle = state.handle
    adapter._graph_cache = state.graph_cache
    adapter._rpu_is_ready = True


def _clear_execution_graph(cache) -> None:
    if cache is None:
        return
    cache.begin_warmup()
    cache.clear()
    if not cache.cache_invariant_ok():
        raise RuntimeError("Qwen3.5 GraphCache invariant failed during reconfigure")


def _validate_qwen3_5_execution_reconfigure(adapter, execution_config) -> None:
    """Reject native cold-control changes before any live state is mutated."""
    from rpu_backend.adapters.qwen3_5.cores import core_topology, topology_tuple
    model = adapter.model
    adapter.preflight_execution(model.config, execution_config)
    state = _qwen3_5_text_runtime_state(model)
    if state is None:
        from .text import _public_retained_prefill
        session = getattr(adapter, "_execution_session", None)
        mode = ("RETAINED_CACHE" if _public_retained_prefill(
            model.config, execution_config, public_owner=True,
            vision_admitted=os.environ.get("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED") == "1")
            else "BOUNDED_ONESHOT")
        if session is not None and session.graph_mode != mode:
            raise ValueError("Qwen3.5 cold prefill lifecycle is immutable; reload the model")
        return
    installed = getattr(state, "execution_topology", None) or core_topology()
    if topology_tuple(installed) != topology_tuple(core_topology(execution_config)):
        raise ValueError("Qwen3.5 model.num_cores is immutable after installation; reload the model")
    for stage in ("prefill", "vision"):
        if (execution_config.get(stage, {}).get("linear_acc32", False)
                != adapter._rpu_execution.get(stage, {}).get("linear_acc32", False)):
            raise ValueError(
                f"Qwen3.5 {stage}.linear_acc32 is immutable after installation; "
                "reload the model")
    required = (
        "execution_reconfigure_begin",
        "execution_reconfigure_commit",
        "execution_reconfigure_abort",
        "execution_reconfigure_abort_attempt",
        "qwen3_5_stage_prefill_execution_controls",
    )
    missing = [name for name in required if not hasattr(torch.ops.rpu, name)]
    if missing:
        raise RuntimeError(
            "Qwen3.5 binary lacks native execution-reconfigure op(s): "
            + ", ".join(missing)
        )

    vision_model = getattr(model.model, "visual", None)
    if getattr(vision_model, "_rpu_vision_handle", None) is None:
        return
    if not hasattr(
        torch.ops.rpu, "qwen3_5_vision_stage_prefill_execution_controls"
    ):
        raise RuntimeError(
            "Qwen3.5 vision binary lacks "
            "qwen3_5_vision_stage_prefill_execution_controls"
        )


def _apply_qwen3_5_execution_config(
    adapter, execution_config, generation, *, force_native=False
) -> None:
    """Apply one admitted generation to the live Qwen3.5 handles."""
    adapter._execution_reconfigure_journal = {"mutation_started": False}
    model = adapter.model
    state = _qwen3_5_text_runtime_state(model)
    if state is None:
        adapter._execution_reconfigure_journal["mutation_started"] = True
        adapter._rpu_execution = execution_config
        return

    text_options = _resolve_text_install_options(
        state.max_seq_len,
        execution_config=execution_config,
        cold_snapshot=state.control_snapshot,
    )
    vision_cold_snapshot = None
    vision_control_options = getattr(state, "vision_control_options", None)
    if (
        isinstance(vision_control_options, tuple)
        and len(vision_control_options) == 5
    ):
        vision_cold_snapshot = vision_control_options[4]
    vision_options = _resolve_vision_install_options(
        execution_config=execution_config,
        cold_snapshot=vision_cold_snapshot,
    )
    prefill = execution_config.get("prefill", {})
    requested_chunk = prefill.get("chunk_size", "auto")
    exact_chunk = 0 if requested_chunk == "auto" else int(requested_chunk)
    padding_rows = prefill.get("padding_rows", "auto")

    fusion = model.model
    vision_model = getattr(fusion, "visual", None)
    vision_handle = getattr(vision_model, "_rpu_vision_handle", None)
    current_text_cap = int(
        state.control_snapshot.get("chunk_effective_cap", 0)
    )
    current_text_exact = int(state.exact_chunk_size or 0)
    text_controls_changed = force_native or (
        int(text_options[1]) != current_text_cap
        or exact_chunk != current_text_exact
    )
    vision_controls_changed = vision_handle is not None and (
        force_native
        or int(vision_options[0])
            != int(vision_model._rpu_vision_chunk_size_cap)
        or int(vision_options[1] or 0)
            != int(vision_model._rpu_vision_exact_chunk_size or 0)
    )
    with native_execution_reconfigure(torch.ops.rpu) as token:
        # A failed begin has not touched native state and needs no compensating
        # transaction.  Once begin succeeds, fail closed by restoring the old
        # controls after any later interruption, including an ambiguous commit.
        adapter._execution_reconfigure_journal["mutation_started"] = True
        if text_controls_changed:
            max_kv_len, envelope_chunk = state.chunk_envelope
            torch.ops.rpu.qwen3_5_stage_prefill_execution_controls(
                state.handle,
                token,
                int(text_options[1]),
                exact_chunk,
                int(max_kv_len),
                int(envelope_chunk),
            )
        if vision_controls_changed:
            torch.ops.rpu.qwen3_5_vision_stage_prefill_execution_controls(
                vision_handle,
                token,
                int(vision_options[0]),
                int(vision_options[1] or 0),
            )

    # Publish the native controls to the private state before fallible graph
    # cleanup so the session rollback can see and reverse a committed change.
    state.padding_budget = text_options[2]
    state.padding_rows = padding_rows
    state.execution_config = execution_config
    state.control_snapshot = text_options[3]
    state.exact_chunk_size = exact_chunk or None
    state.vision_control_options = vision_options
    if vision_handle is not None:
        (
            vision_model._rpu_vision_chunk_size_cap,
            vision_model._rpu_vision_exact_chunk_size,
            vision_model._rpu_vision_padding_rows,
            vision_model._rpu_vision_padding_budget,
            vision_model._rpu_vision_control_snapshot,
        ) = vision_options

    # Native commit invalidates each staged model. Drop Python graph owners only
    # after that commit; a later failure is restored by the session rollback.
    _clear_execution_graph(state.graph_cache)
    _clear_execution_graph(getattr(state, "prefill_graph_cache", None))
    _clear_execution_graph(getattr(state, "prefill_debug_graph_cache", None))
    state.prefill_graph.invalidate()
    if vision_handle is not None:
        _clear_execution_graph(vision_model._rpu_vision_graph_cache)
        vision_model._rpu_vision_debug_graph.invalidate()

    state.prefill_graph_sig = None
    state.prefill_debug_graph_sig = None
    state.prefill_rope_shape = None
    state.prefill_rope_cache = None
    state.execution_generation = int(generation)
    state.prefill_plan_key = None
    state.prefill_plan = None
    state.decode_stage_descriptor = None
    state.decode_plan = None
    state.last_a6_plan = None
    if vision_handle is not None:
        vision_model._rpu_vision_graph_key = None
        vision_model._rpu_vision_graph_sig = None
        vision_model._rpu_vision_last_a6_plan = None
        vision_model._rpu_vision_a6_plans = {}
        vision_model._execution_generation = int(generation)
    adapter._rpu_execution = execution_config
    # Retain undo state through the shared session's config publication.


def _rollback_qwen3_5_execution_config(
    adapter, old_config, generation
) -> None:
    journal = getattr(adapter, "_execution_reconfigure_journal", None)
    if journal is not None and not journal.get("mutation_started", False):
        adapter._execution_reconfigure_journal = None
        return
    _apply_qwen3_5_execution_config(
        adapter, old_config, generation, force_native=True
    )
    adapter._execution_reconfigure_journal = None


def _cleanup_failed_text_install(
    adapter, model, inner, had_instance_forward, original_forward
) -> None:
    """Retire a failed text install without unpoisoning mutated weights."""
    if inner is not None:
        _retire_failed_qwen3_5_text_install(
            inner, session=getattr(adapter, "_execution_session", None))
    model_state = vars(model)
    model_state.pop("_lm_head_w_rpu_keepalive", None)
    if had_instance_forward:
        model_state["forward"] = original_forward
    else:
        model_state.pop("forward", None)
    adapter._handle = None
    adapter._graph_cache = None
    adapter._rpu_is_ready = False


def _config_profile(config):
    tc = _text_config(config)
    return (int(tc.hidden_size), int(tc.intermediate_size),
            int(tc.num_hidden_layers), int(tc.num_key_value_heads))


def _check_quant_checkpoint_loaded(model) -> None:
    """Reject quantized tensors that went through the generic HF loader.

    Generic HF loading can cast int8 projection tensors into the skeleton's
    fp16 Parameters and produce a runnable but silently wrong model. The
    Qwen3.5 loader stamps the model after installing the checkpoint's declared
    projection layouts.
    """
    quant = getattr(model.config, "quant_config", None)
    if not isinstance(quant, dict) or not quant:
        return
    if getattr(model, "_qwen3_5_quant_config", None) is not None:
        return
    raise RPUBackendError(
        "This Qwen3.5 checkpoint declares quant_config="
        f"{quant!r}, but it was loaded by a generic HF loader "
        "that cannot preserve int8 weights. Load it with "
        "rpu_backend.api.RPUModelForConditionalGeneration.from_pretrained(...)."
    )


def _check_quant_policy(config) -> None:
    quant = getattr(config, "quant_config", None)
    if not quant or os.environ.get("QWEN3_5_QUANT_ALLOW_UNCERTIFIED") == "1":
        return
    raise UnsupportedModelError(
        "Qwen3.5 W8A16/W4A16 is available only for controlled, uncertified "
        "evaluation; Dense FP16 is the supported path. Set exact "
        "QWEN3_5_QUANT_ALLOW_UNCERTIFIED=1 to opt in before loading weights."
    )


def _check_profile(config) -> None:
    """Raise UnsupportedModelError for MoE or out-of-envelope profiles."""
    tc = _text_config(config)
    n_experts = int(getattr(tc, "num_experts", 0) or 0)
    if n_experts > 0:
        raise UnsupportedModelError(
            f"Qwen3.5 MoE (num_experts={n_experts}) is out of scope; only dense "
            "small models (0.8B/2B/4B/9B) are supported by the RPU graph-building port."
        )
    profile = _config_profile(config)
    if profile not in _SUPPORTED_PROFILES:
        raise UnsupportedModelError(
            f"Qwen3.5 profile {profile} is not supported (SPM envelope / not yet "
            f"validated). Supported profiles: {sorted(_SUPPORTED_PROFILES)}."
        )
    try:
        legacy = validate_legacy_text_profile(config)
    except ValueError as exc:
        raise UnsupportedModelError(str(exc)) from exc
    if profile == (5120, 17408, 64, 4) and not legacy:
        raise UnsupportedModelError(
            "Qwen3.8-27B requires the exact legacy W8A16 text-only bundle; "
            "BF16, uniform W8, W4, Vision and MTP are not admitted.")


class Qwen3_5Adapter:
    """Orchestration adapter for Qwen3.5 (== ForConditionalGeneration + fusion).

    Lifecycle:
      1. ``__init__(model)`` — profile + layer_types fail-fast guards.
      2. ``to_rpu()`` — install the text backbone C++ handle, prep lm_head,
         leave the vision tower for lazy first-image install, and replace the
         top-level forward. Returns the model.

    The C++ handle stays on the adapter (``self._handle``) and the per-model
    GraphCache is mirrored to ``self._graph_cache`` (both read by tests / the
    ``qwen3_5_get_resolved_chunk_size`` bench path).
    """

    _EXECUTION_SUPPORTED = _QWEN3_5_EXECUTION_SUPPORTED

    def __init__(self, model, *, rpu_execution=None):
        from rpu_backend.api._execution import (
            bind_rpu_execution,
            normalize_rpu_execution,
        )

        supported_execution = self._EXECUTION_SUPPORTED
        normalized_execution = None
        if rpu_execution is not None:
            normalized_execution = normalize_rpu_execution(
                rpu_execution,
                entry_point="Qwen3_5Adapter",
                supported=supported_execution,
            )
            fixed_chunk = normalized_execution.get("prefill", {}).get(
                "chunk_size"
            )
            if isinstance(fixed_chunk, int) and fixed_chunk % 64:
                raise ValueError(
                    "Qwen3_5Adapter: prefill chunk_size must be a multiple of "
                    f"64, got {fixed_chunk}"
                )

        self.model = model
        _check_quant_policy(model.config)
        _check_profile(model.config)
        _validate_layer_types(model.config)
        _check_quant_checkpoint_loaded(model)
        if validate_legacy_text_profile(model.config) and not getattr(
            model, "_qwen3_5_legacy_text_only", False
        ):
            raise UnsupportedModelError(
                "Qwen3.8-27B requires the scope-aware legacy text-only loader")
        self._rpu_execution = bind_rpu_execution(
            model,
            normalized_execution if rpu_execution is not None else None,
            entry_point="Qwen3_5Adapter",
            supported=supported_execution,
        )
        self.preflight_execution(model.config, self._rpu_execution)
        session_callbacks = {}
        if getattr(model, "_execution_session", None) is None:
            session_callbacks = {
                "validate": lambda config: (
                    _validate_qwen3_5_execution_reconfigure(self, config)
                ),
                "apply": lambda _old, new, generation, *, force_rebuild=False: (
                    _apply_qwen3_5_execution_config(
                        self, new, generation, force_native=force_rebuild,
                    )
                ),
                "rollback": lambda old, _new, generation: (
                    _rollback_qwen3_5_execution_config(self, old, generation)
                ),
            }
        from .text import _public_retained_prefill

        self._execution_session = bind_execution_session(
            model,
            self._rpu_execution,
            entry_point="Qwen3_5Adapter",
            supported=supported_execution,
            graph_mode=("RETAINED_CACHE" if _public_retained_prefill(
                model.config, self._rpu_execution, public_owner=True,
                vision_admitted=os.environ.get("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED") == "1")
                        else "BOUNDED_ONESHOT"),
            **session_callbacks,
        )
        self._execution_session.register_config_view(self)
        fusion = getattr(model, "model", model)
        self._execution_session._bind_planner_owner(
            getattr(fusion, "language_model", fusion)
        )
        vision = getattr(fusion, "visual", None)
        if vision is not None:
            self._execution_session._bind_planner_owner(vision)
        state = _qwen3_5_text_runtime_state(model)
        self._handle = getattr(state, "handle", None)
        self._graph_cache = getattr(state, "graph_cache", None)
        self._rpu_is_ready = state is not None

        if getattr(model.to, "__rpu_wrapped__", False):
            return

        original_to = model.to

        def rpu_aware_to(*args, **kwargs):
            target = extract_to_device_target(args, kwargs)
            if target is not None and is_rpu_device_target(
                target, entry_point="Qwen3_5Adapter.model.to"
            ):
                if len(args) > 1 or set(kwargs) - {"device"}:
                    raise ValueError(
                        "model.to('rpu') takes no extra args/kwargs; call "
                        "Qwen3_5Adapter(model).to_rpu(max_seq_len=...) for a "
                        "non-default sequence horizon."
                    )
                return self.to_rpu()
            if (
                self._rpu_is_ready
                or getattr(self.model, "_rpu_swizzled", False)
                or getattr(self.model, "_rpu_swizzle_started", False)
            ):
                raise RPUBackendError(
                    f"model.to({target!r}) rejected: weights are swizzled for RPU "
                    "(irreversible). Reload the model to use another device."
                )
            return original_to(*args, **kwargs)

        rpu_aware_to.__rpu_wrapped__ = True
        model.to = rpu_aware_to

    @classmethod
    def preflight(cls, config) -> None:
        _check_quant_policy(config)
        _check_profile(config)
        _validate_layer_types(config)

    @classmethod
    def preflight_execution(cls, config, execution_config) -> None:
        from rpu_backend.adapters.qwen3_5.cores import (
            validate_core_profile, is_dense_9b_vl_candidate, validate_dense_9b_vl_profile,
        )
        validate_core_profile(config, execution_config)
        from .text import _validate_qwen35_linear_accumulation
        _validate_qwen35_linear_accumulation(config, execution_config)
        if (is_dense_9b_vl_candidate(config)
                and (not getattr(config, "quant_config", None)
                     or os.environ.get("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED") == "1")):
            validate_dense_9b_vl_profile(config)
        if validate_legacy_text_profile(config) and execution_config.get("vision"):
            raise UnsupportedModelError(
                "Qwen3.8-27B legacy W8A16 supports only text execution")
        prefill = execution_config.get("prefill", {})
        _validate_qwen35_text_padding(prefill)
        requested = prefill.get("chunk_size", "auto")
        if isinstance(requested, int):
            if requested % 64:
                raise ValueError(
                    "Qwen3.5 prefill chunk_size must be a multiple of 64, got "
                    f"{requested}. Refusing before loading model weights."
                )
            tc = _text_config(config)
            from .text import CERTIFIED_CHUNK_ENVELOPE
            env = CERTIFIED_CHUNK_ENVELOPE.get(
                (int(tc.num_hidden_layers), int(tc.hidden_size))
            )
            if env is not None and env[1] > 0 and requested > env[1]:
                raise UnsupportedModelError(
                    f"Qwen3.5 prefill chunk_size={requested} exceeds this profile's "
                    f"declared ceiling {env[1]}; max declared KV length is "
                    f"{env[0]}. Refusing before loading model weights."
                )
        vision = execution_config.get("vision", {})
        vision_chunk = vision.get("chunk_size", "auto")
        if isinstance(vision_chunk, int):
            if vision_chunk <= 0 or vision_chunk % 16:
                raise ValueError(
                    "Qwen3.5 vision chunk_size must be a positive multiple of 16, "
                    f"got {vision_chunk}. Refusing before loading model weights."
                )
        for field, limit in (
            ("padding_budget", QWEN3_5_OPTIONAL_PADDING_CAP),
            ("padding_rows", QWEN3_5_TOTAL_PADDING_CAP),
        ):
            value = vision.get(field)
            if isinstance(value, int) and value > limit:
                raise ValueError(
                    f"Qwen3.5 vision {field} exceeds the cache-backed maximum "
                    f"{limit}: got {value}. Refusing before loading model weights."
                )
        if (
            isinstance(vision.get("padding_rows"), int)
            and vision["padding_rows"] != 0
        ) or int(vision.get("padding_budget", 0)) != 0:
            raise ValueError(
                "Qwen3.5 vision padding is unsupported because bidirectional "
                "attention has no padding-mask route."
            )

    @classmethod
    def load_pretrained(cls, path, *, config, dtype=torch.float16, **hf_kwargs):
        """Load FP16 or projection-quantized Qwen3.5 checkpoints on CPU."""
        from rpu_backend.adapters.qwen3_5.loader import load_qwen3_5_model

        return load_qwen3_5_model(
            path, config=config, dtype=dtype, **hf_kwargs)

    def to_rpu(self, max_seq_len=8192):
        """Install text backbone + lm_head and patch the top-level forward.

        Order (A→B→C→D matters): text first (creates the handle the rest read),
        lm_head after (uses ``get_output_embeddings``), lazy vision state next,
        and forward replacement last.
        """
        model = self.model
        from rpu_backend.adapters.qwen3_5.cores import (
            validate_core_profile, validate_text_weight_geometry, topology_tuple,
            is_dense_9b_vl_candidate, validate_dense_9b_vl_weights, validate_cold_weight_dtype,
            validate_text_mlp_padding_storage,
        )
        topology = validate_core_profile(getattr(model, "config", None), getattr(model, "_rpu_execution", None))
        state = _qwen3_5_text_runtime_state(model)
        if state is not None:
            installed = getattr(state, "execution_topology", None)
            if installed is not None and topology_tuple(installed) != topology_tuple(topology):
                raise ValueError("Qwen3.5 model.num_cores is immutable after installation; reload the model")
            _sync_adapter_text_state(self, state)
            return model
        if self._rpu_is_ready or getattr(model, "_rpu_swizzled", False):
            self._handle = None
            self._graph_cache = None
            self._rpu_is_ready = False
            raise RPUBackendError(
                "Qwen3_5Adapter.to_rpu(): the ready marker exists but text "
                "runtime ownership is incomplete; reload the model."
            )
        if getattr(model, "_rpu_swizzle_started", False):
            raise RPUBackendError(
                "Qwen3_5Adapter.to_rpu(): a prior install failed after mutation; "
                "reload the model before retrying."
            )
        arena_candidate = is_dense_9b_vl_candidate(getattr(model, "config", None))
        arena_opt_in = (arena_candidate
                        and os.environ.get("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED") == "1")
        # The Session chooses this lifecycle before installation. Recheck cold
        # scope drift before ownership or weight mutation; never silently switch
        # the native descriptor mode or add a queue to a different arena plan.
        session = getattr(self, "_execution_session", None)
        public_prefill_graph_mode = None
        if session is not None:
            from .text import _public_retained_prefill
            public_prefill_graph_mode = ("RETAINED_CACHE" if _public_retained_prefill(
                model.config, getattr(model, "_rpu_execution", None), public_owner=True,
                vision_admitted=arena_opt_in) else "BOUNDED_ONESHOT")
            if session.graph_mode != public_prefill_graph_mode:
                raise ValueError("Qwen3.5 cold prefill lifecycle changed; reload the model "
                                 "with its final core, quantization and Vision scope")
        # Pure text also needs instruction storage reserved before the 9B
        # weights consume the PC-addressable allocation range. This exact
        # FP16 resource plan is independent of the controlled Vision gate.
        arena_required = (arena_candidate
                          and (arena_opt_in or not getattr(model.config, "quant_config", None)))
        if arena_required:
            validate_dense_9b_vl_weights(model)
        if topology.num_cores != 8:
            for name in ("qwen3_5_set_execution_cores", "qwen3_5_get_execution_topology",
                         "qwen3_5_vision_set_execution_cores", "qwen3_5_vision_get_execution_topology",
                         "linear_with_accumulation"):
                if not hasattr(torch.ops.rpu, name):
                    raise RuntimeError(f"Qwen3.5 binary lacks reduced-core op {name}; rebuild before installing")
            inner = getattr(model.model, "language_model", model.model)
            validate_core_profile(inner.config, getattr(model, "_rpu_execution", None))
            validate_text_weight_geometry(inner, model.config, topology)
            validate_text_mlp_padding_storage(inner, model.config, topology, cold_cpu=True)
            output = model.get_output_embeddings()
            if tuple(output.weight.shape) != (inner.config.vocab_size, inner.config.hidden_size):
                raise ValueError("Qwen3.5 reduced LM-head weight shape disagrees with config")

        # Cold handle/environment settings must fail before live ownership is
        # claimed and before dtype conversion/weight swizzling can mutate the model.
        resolved_text_options = _resolve_text_install_options(
            max_seq_len,
            execution_config=getattr(model, "_rpu_execution", None),
        )
        resolved_vision_options = _resolve_vision_install_options(
            execution_config=getattr(model, "_rpu_execution", None),
        )
        max_seq_len = resolved_text_options[0]
        if not _SWIZZLE_LOCK.acquire(blocking=False):
            raise RPUBackendError(
                "Qwen3_5Adapter.to_rpu(): another Qwen3.5 swizzle is in progress."
            )

        inner = None
        claimed = False
        had_instance_forward = "forward" in vars(model)
        original_forward = vars(model).get("forward")
        try:
            from rpu_backend.runtime.hw_attrs import (
                install_hw_attr_validator,
                validate_postinstall,
                validate_preinstall,
            )
            validate_preinstall(model)

            state = _qwen3_5_text_runtime_state(model)
            if state is not None:
                _sync_adapter_text_state(self, state)
                return model
            if getattr(model, "_rpu_swizzled", False):
                raise RPUBackendError(
                    "Qwen3_5Adapter.to_rpu(): the ready marker exists but text "
                    "runtime ownership is incomplete; reload the model."
                )
            if getattr(model, "_rpu_swizzle_started", False):
                raise RPUBackendError(
                    "Qwen3_5Adapter.to_rpu(): a prior install failed after "
                    "mutation; reload the model before retrying."
                )

            is_legacy_text = validate_legacy_text_profile(model.config)
            graph_options = {"_public_prefill_graph_mode": public_prefill_graph_mode}
            if is_legacy_text:
                from rpu_backend.graph import GraphRuntimePolicy

                graph_policy = GraphRuntimePolicy.from_environment(
                    execution_core_count=topology.num_cores,
                    qwen35_legacy_27b_sdk_budget=True)
                graph_options["_graph_runtime_policy"] = graph_policy
            elif arena_required:
                from rpu_backend.graph import GraphRuntimePolicy

                graph_policy = GraphRuntimePolicy.from_environment(
                    execution_core_count=topology.num_cores, graph_arena_count=4)
                graph_options["_graph_runtime_policy"] = graph_policy
            _claim_live_instance(model)
            claimed = True
            try:
                # The legacy 27B path requires direct allocations: recycled
                # mappings can corrupt full-prefill rows. Claim the
                # immutable native policy at the exclusive-owner boundary;
                # reading it first leaves a check-to-allocation race.
                if is_legacy_text:
                    torch.rpu.set_caching_allocator(False)
                    if not graph_policy.prepare_arenas():
                        raise RPUBackendError(
                            "Qwen3.8-27B requires accessible RPU hardware to "
                            "prepare its SDK graph arenas before weight upload"
                        )
                elif not arena_required:
                    # Ordinary dense owners, including quantized 9B text, reuse
                    # transient mappings. A 9B geometry alone does not require
                    # the separate FP16 arena plan or direct tensor allocations.
                    torch.rpu.set_caching_allocator(True)
                elif arena_required:
                    # Dense 9B reuses transient tensors for FP16 and W8 alike;
                    # its cold graph arenas have an independent lifecycle.
                    torch.rpu.set_caching_allocator(True)
                    if not graph_policy.prepare_arenas():
                        raise RPUBackendError(
                            "Qwen3.5 9B text requires accessible RPU hardware to prepare "
                            "its cold graph arenas before weight upload"
                        )
            except BaseException:
                _release_live_instance(model)
                claimed = False
                raise
            model._rpu_swizzle_started = True

            # Some Qwen3.5 checkpoints keep nested weights in bf16 despite
            # from_pretrained(dtype=fp16). Cast once, but keep the lazy vision
            # tower on CPU; only the text stack moves now. The output weight is
            # copied into the swizzled keepalive below.
            model.half()
            if arena_required:
                validate_cold_weight_dtype(model, fp16=True)
            inner = getattr(model.model, "language_model", model.model)
            _out_emb = model.get_output_embeddings()
            inner.to("rpu")

            # ── Step A: text backbone (swizzle + create + set_weights) ──
            handle = install_qwen3_5_text_for_rpu(
                inner,
                model.config,
                max_seq_len,
                execution_config=getattr(model, "_rpu_execution", None),
                _resolved_options=resolved_text_options,
                **graph_options,
            )
            self._handle = handle
            from .text import _bind_qwen35_retirement

            _bind_qwen35_retirement(inner._rpu_qwen3_5, model, _close_qwen3_5_model, inner)
            # `self._graph_cache` is read by bench_qwen3_5_decode; mirror it from the
            # per-model state stashed by install_qwen3_5_text_for_rpu.
            self._graph_cache = inner._rpu_qwen3_5.graph_cache
            inner._rpu_qwen3_5.execution_generation = (
                getattr(self, "_execution_session", None).generation
                if getattr(self, "_execution_session", None) is not None
                else 0
            )
            inner._rpu_qwen3_5.vision_control_options = (
                resolved_vision_options
            )
            if arena_candidate:
                inner._rpu_qwen3_5.vision_arena_admitted = arena_opt_in

            # ── Step B: lm_head (ForConditionalGeneration concern) ──
            embed = inner.embed_tokens

            # lm_head uses the explicit per-model Linear accumulation policy.
            # tie_word_embeddings=True → lm_head weight is the tied
            # embedding matrix. embed_tokens is an nn.Embedding so it is NOT swizzled
            # by convert_linear_weights_inplace; the RPU GEMM kernel expects a
            # col-partition swizzled weight (get_linear_partition(hidden,vocab)=1).
            # Feeding the raw weight to rpu_linear yields garbage (cos≈0) — that was
            # an earlier misdiagnosis of "RPU matmul is broken". Swizzle once here;
            # the matmul then runs on-device for both prefill and (non-fused) decode.
            # get_output_embeddings() returns the tied embedding when tie_word_embeddings
            # (0.8B/2B/4B) and the SEPARATE lm_head weight when NOT tied (9B). Using
            # embed.weight directly is wrong for untied models (garbage logits).
            _lm_head_w = _out_emb.weight if _out_emb is not None else embed.weight
            model._lm_head_w_rpu_keepalive = transform_linear_weight(
                _lm_head_w.detach().to("cpu").to(torch.float16).contiguous(),
                1, topology.lm_head_tp).to("rpu")

            # ── Step C: vision tower — LAZY (installed on first image forward) ──
            # Not installed here on purpose: a Qwen3.5 checkpoint is used for BOTH
            # text-only chat and vision, so eagerly swizzling the 297-tensor vision
            # tower would burden every text-only run and require the vision C++ ops
            # in the .so even when no image is ever passed. `fuse_visual_embeds`
            # (vision.py) installs it idempotently the first time pixel_values arrive,
            # keeping text-only to_rpu byte-identical (BC — spec §9.1).

            # ── Step D: replace top-level forward ──
            model.forward = types.MethodType(_rpu_qwen3_5_forward, model)
            validate_postinstall(model)
            install_hw_attr_validator(model)
            self._rpu_is_ready = True
            model._rpu_swizzled = True
            return model
        except BaseException as error:
            try:
                _cleanup_failed_text_install(
                    self, model, inner, had_instance_forward, original_forward)
                if claimed and not getattr(model, "_rpu_swizzle_started", False):
                    _release_live_instance(model)
            except BaseException as cleanup_error:
                error.add_note(f"Qwen3.5 outer install cleanup failed: {cleanup_error!r}")
            raise
        finally:
            _SWIZZLE_LOCK.release()


@execution_serialized
def _rpu_qwen3_5_forward(self, input_ids=None, inputs_embeds=None,
                         past_key_values=None, attention_mask=None,
                         position_ids=None, pixel_values=None,
                         pixel_values_videos=None, image_grid_thw=None,
                         video_grid_thw=None, mm_token_type_ids=None,
                         logits_to_keep=0, labels=None, use_cache=None,
                         output_attentions=None, output_hidden_states=None,
                         return_dict=None, cache_position=None, **kw):
    """Top forward for Qwen3_5ForConditionalGeneration (fusion-shaped).

    embed → optional vision fuse (direct merger DMA + 3D M-RoPE) → text
    decoder (RPU) → lm_head. Fragmented image-token layouts fall back to a
    host scatter; text-only skips the vision branch entirely.
    """
    if getattr(self, "_qwen3_5_legacy_text_only", False) and any(
        value is not None for value in
        (pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw)
    ):
        raise NotImplementedError("Qwen3.8-27B legacy W8A16 is text-only")
    from rpu_backend.runtime.hw_attrs import mark_first_forward_done
    mark_first_forward_done(self)

    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError(
            "Qwen3.5 RPU forward requires exactly one of input_ids or "
            "inputs_embeds.")
    if not isinstance(past_key_values, Qwen3_5Cache):
        raise TypeError(
            "Qwen3.5 RPU forward requires Qwen3_5Cache.from_config(...); "
            "past_key_values=None and generic HF caches are not supported.")
    if labels is not None:
        raise NotImplementedError(
            "Qwen3.5 RPU inference does not support labels/loss computation.")
    if use_cache is False:
        raise NotImplementedError(
            "Qwen3.5 RPU inference requires use_cache=True.")
    if return_dict is False:
        raise NotImplementedError(
            "Qwen3.5 RPU inference requires return_dict=True.")
    if output_attentions:
        raise NotImplementedError(
            "Qwen3.5 RPU inference does not support output_attentions=True.")
    if output_hidden_states:
        raise NotImplementedError(
            "Qwen3.5 RPU inference does not support output_hidden_states=True; "
            "the fused output buffer is reused across forwards.")
    if pixel_values_videos is not None or video_grid_thw is not None:
        raise NotImplementedError(
            "Qwen3.5 RPU inference does not support video inputs.")
    if kw:
        raise TypeError(
            "Qwen3.5 RPU forward received unsupported keyword argument(s): "
            + ", ".join(sorted(kw)))
    if not isinstance(logits_to_keep, int) or logits_to_keep < 0:
        raise NotImplementedError(
            "Qwen3.5 RPU forward supports only a non-negative integer "
            "logits_to_keep.")

    cache = past_key_values
    fusion = getattr(self, "model", self)
    language = getattr(fusion, "language_model", fusion)
    installed = getattr(getattr(language, "_rpu_qwen3_5", None), "execution_topology", None)
    if installed is not None and installed.num_cores != 8:
        request = input_ids if input_ids is not None else inputs_embeds
        expected_rank = 2 if input_ids is not None else 3
        if request.ndim != expected_rank or request.shape[0] != 1:
            raise NotImplementedError("reduced Qwen3.5 inference requires batch size 1")
    if pixel_values is not None:
        if input_ids is None:
            raise ValueError(
                "Qwen3.5 image inference requires input_ids to locate image tokens.")
        if image_grid_thw is None or mm_token_type_ids is None:
            raise ValueError(
                "Qwen3.5 image inference requires image_grid_thw and "
                "mm_token_type_ids.")
        if int(cache.position) != 0:
            raise NotImplementedError(
                "Qwen3.5 image inputs are accepted under controlled evaluation "
                "only at prefill position 0.")

    # The fused text path implements causal, unpadded attention. Accepting a
    # padding mask and then dropping it would silently change model semantics.
    if attention_mask is not None:
        mask_cpu = attention_mask.detach().to("cpu")
        if not bool(mask_cpu.bool().all().item()):
            raise NotImplementedError(
                "Qwen3.5 RPU forward does not support padding/zero entries in "
                "attention_mask.")
        attention_mask = None

    inner = getattr(self.model, "language_model", self.model)  # Qwen3_5TextModel

    request_len = int(
        (inputs_embeds if inputs_embeds is not None else input_ids).shape[1]
    )
    if cache_position is not None:
        actual = cache_position.detach().to("cpu").reshape(-1).long()
        expected = torch.arange(
            int(cache.position), int(cache.position) + request_len
        )
        if not torch.equal(actual, expected):
            raise ValueError(
                f"Qwen3.5 cache_position={actual.tolist()} does not match "
                f"cache position {int(cache.position)} and input length "
                f"{request_len}.")
    text_preflight_plan = _preflight_qwen3_5_text(
        inner, cache, request_len,
        multimodal_prefill=pixel_values is not None,
    )
    if pixel_values is not None and _prepare_qwen3_5_multimodal_spm(
        self, input_ids, pixel_values, image_grid_thw, text_preflight_plan[0]
    ):
        # Preserve early rejection before installing vision, then select the
        # actual winner against both owners' fixed SPM regions.
        text_preflight_plan = _preflight_qwen3_5_text(
            inner, cache, request_len, multimodal_prefill=True)

    # 1. embed (CPU→RPU prep stays OUTSIDE any capture scope).
    caller_embeds = inputs_embeds is not None
    if caller_embeds:
        hidden = inputs_embeds.to(device="rpu", dtype=torch.float16).contiguous()
    else:
        ids_rpu = (input_ids if input_ids.device.type == "rpu"
                   else input_ids.to("rpu"))
        hidden = inner.embed_tokens(ids_rpu).to(dtype=torch.float16).contiguous()

    # A new text-only conversation must not inherit the decode offset installed
    # by an earlier image prefill on the same C++ model. Decode-after-image keeps
    # it because its cache position is non-zero; reset()+refeed starts at zero.
    if pixel_values is None and cache is not None and int(cache.position) == 0:
        st = getattr(inner, "_rpu_qwen3_5", None)
        if st is not None:
            torch.ops.rpu.qwen3_5_set_mrope_position_delta(st.handle, 0)

    # 2. vision fusion; text-only skips it entirely.
    #    默认路径由 in-graph merger 直接 DMA 到 final hidden DDR；非连续
    #    image-token layout 才回退 host scatter。M-RoPE 位置计算也在本段。
    if pixel_values is not None:
        hidden, position_ids = fuse_visual_embeds(
            self, hidden, input_ids, pixel_values, image_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            past_key_values=cache)

    # 3. text decoder (RPU): returns [B, real_len, hidden].
    out = run_qwen3_5_text(
        inner, hidden, cache, attention_mask=attention_mask,
        position_ids=position_ids,
        multimodal_prefill=pixel_values is not None,
        preflight_plan=text_preflight_plan)

    # 4. lm_head runs on RPU with the swizzled weight. Only project the last
    # `logits_to_keep` positions; 0 keeps every prefill position for parity tests.
    if isinstance(logits_to_keep, int) and logits_to_keep > 0:
        head_in = out[:, -logits_to_keep:, :].contiguous()
    else:
        head_in = out
    state = inner._rpu_qwen3_5
    topology = getattr(state, "execution_topology", None)
    flat = head_in.reshape(-1, head_in.shape[-1]).contiguous()
    logits = torch.ops.rpu.linear_with_accumulation(
        flat, self._lm_head_w_rpu_keepalive, None, 1,
        topology.lm_head_tp if topology is not None else 8, state.linear_acc32)
    logits = logits.reshape(*head_in.shape[:-1], logits.shape[-1]).to("cpu").to(torch.float32)
    return Qwen3_5CausalLMOutputWithPast(
        loss=None,
        logits=logits,
        past_key_values=cache,
        hidden_states=None,
        attentions=None,
        rope_deltas=getattr(self.model, "rope_deltas", None),
    )


register_adapter("Qwen3_5ForConditionalGeneration", Qwen3_5Adapter)
