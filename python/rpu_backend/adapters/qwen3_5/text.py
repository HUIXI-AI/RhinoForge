"""Qwen3.5 text backbone (== HF ``Qwen3_5TextModel``) → rpu_backend.

Sibling module of ``__init__.py`` (the orchestration entry) in the
``adapters/qwen3_5/`` package. This module owns the text-decoder half of the
adapter: config-driven layer dispatch, weight swizzle + GDN mixer prep, the C++
``qwen3_5_*`` handle install, and the per-forward RPU decoder run (route-B pad +
partial M-RoPE + capture). It is a VERBATIM move of the text algorithm that used
to live inline in ``qwen3_5.py::to_rpu`` — the only boundary change is that
embedding + lm_head are lifted to the orchestration top forward (``__init__.py``),
so ``run_qwen3_5_text`` takes an already-embedded ``hidden`` and returns the raw
decoder output (no lm_head, no output wrapping).

Public surface:
  - ``layer_types_to_is_full(config)`` — per-layer full/linear dispatch (re-exported
    by ``__init__.py`` for backward-compat; ~8 tests import it from ``adapters.qwen3_5``).
  - ``install_qwen3_5_text_for_rpu(text_model, config) -> handle`` — swizzle text
    weights, create the C++ model, ``set_weights``; stashes per-model forward state
    on ``text_model._rpu_qwen3_5`` (a non-``*_handle`` namespace) and returns the
    C++ handle (the ADAPTER keeps it as ``self._handle``).
  - ``run_qwen3_5_text(text_model, hidden, cache, ...) -> raw`` — drive the RPU
    decoder for one forward and return the raw output ([B, real_len, hidden]).

The profile guard (``_SUPPORTED_PROFILES`` / ``_check_profile``) intentionally
stays in ``__init__.py`` — tests monkeypatch ``qwen3_5._SUPPORTED_PROFILES`` and
the reader must live in the same module.
"""
from __future__ import annotations

import os
import types
from contextlib import contextmanager
from collections.abc import Mapping

from rpu_backend.runtime import rpu_env_bool
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError
from rpu_backend.api.qwen3_5_cache import qwen3_5_kv_replication
from rpu_backend.api._execution import (
    QWEN3_5_OPTIONAL_PADDING_CAP,
    _resolve_text_install_options,
)
from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
from rpu_backend.runtime.execution_planner import GRAPH_RETAINED_CACHE
from rpu_backend.adapters.qwen3_5.quant_scope import (
    FP16,
    W4A16_PGRP,
    W8A16,
    normalize_quant_config,
    validate_legacy_text_profile,
    quant_relpaths,
    resolve_quant_specs,
)


class _Qwen35NativeState(types.SimpleNamespace):
    """Actual handle state; parent GC uses the same exclusive close as callers.

    Only this ordinary object holds the Session. The globally registered leaf
    resource finalizer holds graphs/weights, never this state or the HF owner.
    Thus cyclic GC can still inspect the actual Session before releasing it.
    """

    def __del__(self):
        resource = getattr(self, "retirement", None)
        close = getattr(self, "_retirement_close", None)
        session = getattr(self, "_execution_session", None)
        if (resource is None or resource.handle is None or resource.failed is not None
                or close is None or session is None):
            return
        try:
            from rpu_backend.api import causal_lm

            parent = session._owner
            with session._lock, causal_lm._LIVE_LOCK:
                live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                if live is not None and live is not parent and live is not getattr(parent, "model", None):
                    raise RuntimeError("Qwen3.5 retirement lost the live-owner slot")
                close(parent)
                if resource.handle is not None:
                    raise RuntimeError("Qwen3.5 parent GC left a live native resource")
        except BaseException as error:
            from rpu_backend.api._execution import ExecutionSession

            if isinstance(session, ExecutionSession):
                session.poison()
            resource.retain_failure(error, self, getattr(session, "_owner", None))


def _bind_qwen35_retirement(state, parent, close, child):
    session = getattr(parent, "_execution_session", None)
    resource = getattr(state, "retirement", None)
    if (session is None or session._owner is not parent or resource is None
            or resource.owner() is not state or resource.handle != state.handle
            or not any(getattr(child, name, None) is state for name in
                       ("_rpu_qwen3_5", "_rpu_vision_retirement_state"))):
        raise RuntimeError("Qwen3.5 retirement parent identity changed")
    # Ordinary parent-owned references remain cyclic-GC collectable. Allocate
    # and publish the original owners before detaching their leaf callbacks.
    owned = vars(parent).setdefault("_qwen35_retirement_children", [])
    existing = [item for item in owned if item[2] is resource]
    if existing:
        if len(existing) != 1 or not all(left is right for left, right in
                                        zip(existing[0], (child, state, resource, session))):
            raise RuntimeError("Qwen3.5 retirement child identity changed")
    else:
        if resource.parent is not None:
            raise RuntimeError("Qwen3.5 retirement lost its original child record")
        owned.append((child, state, resource, session))
    vars(state).update(_execution_session=session, _retirement_close=close)
    resource.take_ownership(parent)


def _validate_qwen35_retirement_children(parent, children):
    """A mutable model tree cannot erase or replace an already adopted child."""
    session = getattr(parent, "_execution_session", None)
    owned = vars(parent).get("_qwen35_retirement_children", ())
    live = [item for item in owned if item[2].handle is not None]
    current = [(child, state) for child, state in children if state is not None
               and (getattr(state, "handle", None) is not None
                    or getattr(getattr(state, "retirement", None), "handle", None) is not None)]
    if len(live) != len(current) or any(
            not any(child is present and state is actual for present, actual in current)
            or resource is not getattr(state, "retirement", None)
            or original_session is not session
            or getattr(state, "_execution_session", None) is not session
            for child, state, resource, original_session in live):
        raise RuntimeError("Qwen3.5 retirement lost an original child/state/resource/Session")


def _qwen35_retirement_ready(state):
    resource = getattr(state, "retirement", None)
    if (resource is None or resource.owner() is not state or resource.failed is not None
            or resource.handle is None or resource.handle != getattr(state, "handle", None)
            or resource.finalizer is not getattr(state, "handle_finalizer", None)):
        return False
    if resource.parent is None:
        return resource.finalizer.alive
    session = getattr(state, "_execution_session", None)
    return (not resource.finalizer.alive and session is not None
            and resource.parent() is session._owner
            and any(item[1] is state and item[2] is resource and item[3] is session
                    for item in vars(session._owner).get("_qwen35_retirement_children", ()))
            and callable(getattr(state, "_retirement_close", None))
            and not session._closed and not session._poisoned)


def _refresh_qwen35_text_retirement(state):
    resource = getattr(state, "retirement", None)
    if resource is not None:
        resource.graphs = tuple(value for name in (
            "graph_cache", "prefill_graph_cache", "prefill_debug_graph_cache",
            "action_graph_cache", "action_trace_graph_cache")
            if (value := getattr(state, name, None)) is not None)


def _clear_qwen35_graphs(states, vision=None):
    """One complete graph phase shared by the Q35 and G05 parent owners."""
    caches, raw_graphs = [], []
    for state in states:
        child_caches = tuple(getattr(state, name, None) for name in (
            "graph_cache", "prefill_graph_cache", "prefill_debug_graph_cache",
            "action_graph_cache", "action_trace_graph_cache"))
        child_raw = (getattr(state, "prefill_graph", None),)
        resource = getattr(state, "retirement", None)
        if resource is not None and (
                resource.handle != getattr(state, "handle", None)
                or resource.finalizer is not getattr(state, "handle_finalizer", None)):
            raise RuntimeError("Qwen3.5 text retirement handle/finalizer identity changed")
        if resource is not None and (
                {id(graph) for graph in child_caches if graph is not None} != {id(graph) for graph in resource.graphs}
                or {id(graph) for graph in child_raw if graph is not None} != {id(graph) for graph in resource.raw_graphs}):
            raise RuntimeError("Qwen3.5 text retirement graph identity changed")
        caches.extend(child_caches)
        raw_graphs.extend(child_raw)
    if vision is not None:
        cache = getattr(vision, "_rpu_vision_graph_cache", None)
        raw = getattr(vision, "_rpu_vision_debug_graph", None)
        state = getattr(vision, "_rpu_vision_retirement_state", None)
        resource = getattr(state, "retirement", None)
        handle = getattr(vision, "_rpu_vision_installing_handle", None)
        if handle is None:
            handle = getattr(vision, "_rpu_vision_handle", None)
        finalizer = getattr(vision, "_rpu_vision_installing_finalizer", None)
        if finalizer is None:
            finalizer = getattr(vision, "_rpu_vision_handle_finalizer", None)
        if resource is not None and (
                handle != resource.handle or handle != getattr(state, "handle", None)
                or finalizer is not resource.finalizer
                or finalizer is not getattr(state, "handle_finalizer", None)):
            raise RuntimeError("Qwen3.5 Vision retirement handle/finalizer identity changed")
        if resource is not None and (
                {id(cache)} - {id(None)} != {id(graph) for graph in resource.graphs}
                or {id(raw)} - {id(None)} != {id(graph) for graph in resource.raw_graphs}):
            raise RuntimeError("Qwen3.5 Vision retirement graph identity changed")
        caches.append(cache)
        raw_graphs.append(raw)
    seen = set()
    for cache in caches:
        if cache is not None and id(cache) not in seen:
            seen.add(id(cache))
            cache.clear()
            if not cache.cache_invariant_ok():
                raise RuntimeError("Qwen3.5 GraphCache invariant failed during retirement")
    for graph in raw_graphs:
        if graph is not None and id(graph) not in seen:
            seen.add(id(graph))
            graph.invalidate()


def _destroy_qwen3_5_handle(handle, state=None):
    """Compatibility callback: only the actual resource may retire a handle."""
    if state is None or getattr(state, "retirement_failed", None):
        return
    resource = getattr(state, "retirement", None)
    if resource is not None and resource.handle == handle:
        resource.retire()


def _poison_qwen3_5_text_retirement(inner, state, cleanup_error, session=None):
    """Keep failed resources discoverable and latch the unsafe process."""
    state.retirement_failed = repr(cleanup_error)
    vars(inner)["_rpu_qwen3_5"] = state
    resource = getattr(state, "retirement", None)
    if session is None:
        session = getattr(inner, "_execution_session", None)
        binding = getattr(inner, "_planner_cost_session", None)
        if session is None and binding is not None:
            session = binding[0]()
    if session is not None:
        session.poison()
    from rpu_backend.api import causal_lm

    # A failed release makes the *process* unsafe. This current claim is
    # not evidence that its model owns this particular pending handle.
    with causal_lm._LIVE_LOCK:
        if causal_lm._LIVE_TERMINAL_REASON is None:
            owner = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
            if owner is None:
                owner = inner
                causal_lm._claim_live_instance(owner)
            causal_lm._poison_live_instance(
                owner, f"Qwen3.5 text cleanup failed: {cleanup_error!r}", unsafe=True)
        from rpu_backend.api._execution import _mark_execution_process_unsafe

        _mark_execution_process_unsafe(f"Qwen3.5 text cleanup failed: {cleanup_error!r}")
    if resource is not None:
        resource.retain_failure(cleanup_error, inner)


def _retire_qwen3_5_text_state(inner, *, state=None, _graphs_retired=False):
    """Retire resources only; the caller owns the exclusive Session boundary."""
    if state is None:
        state = getattr(inner, "_rpu_qwen3_5", None)
    if state is None:
        return
    if getattr(state, "retirement_failed", None):
        raise RuntimeError(f"Qwen3.5 text retirement already failed: {state.retirement_failed}")
    if _graphs_retired:
        from rpu_backend.api._execution import ExecutionSession

        binding = getattr(inner, "_planner_cost_session", None)
        session = getattr(state, "_execution_session", None)
        if session is None:
            session = getattr(inner, "_execution_session", None) or (binding[0]() if binding else None)
        if not isinstance(session, ExecutionSession) or not session._shutting_down or session._active:
            raise RuntimeError("Qwen3.5 precleared text retirement requires its exclusive Session shutdown")
    try:
        import torch

        handle = getattr(state, "handle", None)
        finalizer = getattr(state, "handle_finalizer", None)
        resource = getattr(state, "retirement", None)
        if resource is not None and (
                (resource.owner() is not None and resource.owner() is not state) or resource.handle != handle
                or resource.failed is not None):
            raise RuntimeError("Qwen3.5 text retirement handle identity changed")
        if handle is None and finalizer is not None and finalizer.alive:
            raise RuntimeError("Qwen3.5 text live finalizer has no recorded handle")
        if handle is not None and resource is None and finalizer is not None and not finalizer.alive:
            raise RuntimeError("Qwen3.5 text handle has a dead finalizer without a confirmed retirement")
        if not _graphs_retired:
            for name in ("graph_cache", "prefill_graph_cache", "prefill_debug_graph_cache",
                         "action_graph_cache", "action_trace_graph_cache"):
                cache = getattr(state, name, None)
                if cache is not None:
                    cache.clear()
                    if not cache.cache_invariant_ok():
                        raise RuntimeError("Qwen3.5 text GraphCache invariant failed")
            graph = getattr(state, "prefill_graph", None)
            if graph is not None:
                graph.invalidate()
        if handle is not None:
            torch.ops.rpu.qwen3_5_destroy(handle)
            if resource is not None:
                resource._native_destroyed(handle)
            elif finalizer is not None:
                finalizer.detach()
        state.handle = None
        if vars(inner).get("_rpu_qwen3_5") is state:
            vars(inner).pop("_rpu_qwen3_5")
    except BaseException as cleanup_error:
        _poison_qwen3_5_text_retirement(inner, state, cleanup_error)
        raise


def _retire_failed_qwen3_5_text_install(inner, *, state=None, session=None):
    """Installation entry: retire the pending state under its actual Session."""
    if state is None:
        state = getattr(inner, "_rpu_qwen3_5", None)
    if state is None:
        return
    if getattr(state, "retirement_failed", None):
        raise RuntimeError(f"Qwen3.5 text retirement already failed: {state.retirement_failed}")
    if session is None:
        session = getattr(inner, "_execution_session", None)
        binding = getattr(inner, "_planner_cost_session", None)
        if session is None and binding is not None:
            session = binding[0]()
    try:
        if session is not None and getattr(state, "handle", None) is not None:
            session.shutdown(lambda: _retire_qwen3_5_text_state(inner, state=state))
            if state.handle is not None:
                raise RuntimeError("closed execution session still owns a Qwen3.5 text handle")
        else:
            _retire_qwen3_5_text_state(inner, state=state)
        if session is not None:
            owned = vars(session._owner).get("_qwen35_retirement_children", ())
            if all(item[2].handle is None for item in owned):
                vars(session._owner).pop("_qwen35_retirement_children", None)
    except BaseException as cleanup_error:
        if not getattr(state, "retirement_failed", None):
            _poison_qwen3_5_text_retirement(inner, state, cleanup_error, session)
        raise


def _validate_text_install_options(
    max_seq_len, *, execution_config=None
) -> tuple[int, int, int]:
    """Compatibility wrapper returning the three historical scalar values."""
    return _resolve_text_install_options(
        max_seq_len, execution_config=execution_config
    )[:3]


@contextmanager
def _oneshot_scope(g):
    """Record, batch, execute once, then return to PASSTHROUGH.

    The remaining reduced/Vision 9B and internal profiles keep this bounded
    lifecycle. Public retained prefill and decode use their separate caches.
    """
    g.begin()
    try:
        yield
    except BaseException:
        g.abort()
        raise
    else:
        g.end()


def _public_retained_prefill(config, execution_config=None, *, public_owner=False,
                             vision_admitted=False):
    """Select the cold public lifecycle, independently of certification.

    Exact TP8 9B FP16/uniform W8/W4-group32 text additionally requires the
    public owner's immutable Session binding. Internal installs, reduced 9B,
    Vision, legacy 27B, MoE and G0.5 keep their separate lifecycle contracts.
    """
    tc = getattr(config, "text_config", config)
    if getattr(tc, "adaptive_mode", None) is not None:
        return False
    geometry = (getattr(tc, "num_hidden_layers", None), getattr(tc, "hidden_size", None))
    if geometry in {(24, 1024), (24, 2048), (32, 2560)}:
        return True
    if geometry != (32, 4096) or not public_owner or vision_admitted:
        return False
    from .cores import core_topology, validate_dense_9b_retained_text_profile
    if core_topology(execution_config).num_cores != 8:
        return False
    try:
        validate_dense_9b_retained_text_profile(config)
    except (TypeError, ValueError):
        return False
    return True


def _text_config(config):
    """Qwen3.5 nests text fields under ``config.text_config``; unit-test
    namespaces may be flat. Return the text config (fallback: config itself)."""
    return getattr(config, "text_config", config)


def _plan_prefill_execution(real_len, physical_limit, padding_budget,
                            *, resolve_stage_domain, padding_rows="auto",
                            exact_chunk_size=None, max_stage_chunk_size=None,
                            plan_result_sink=None, queue_owner_id=0,
                            graph_mode="BOUNDED_ONESHOT", execution_owner=None,
                            execution_native=None, plan_signature=None, graph_cache=None):
    """Qwen3.5 wrapper around the shared planner's mandatory 64-row grid."""
    return plan_bounded_prefill_execution(
        real_len,
        physical_limit,
        padding_budget,
        resolve_stage_domain=resolve_stage_domain,
        alignment=64,
        padding_rows=padding_rows,
        exact_chunk_size=exact_chunk_size,
        max_stage_chunk_size=max_stage_chunk_size,
        plan_result_sink=plan_result_sink,
        graph_mode=graph_mode,
        queue_owner_id=queue_owner_id,
        execution_owner=execution_owner,
        execution_stage="prefill",
        execution_native=execution_native,
        plan_signature=plan_signature,
        graph_cache=graph_cache,
    )


_VALID_LAYER_TYPES = {"linear_attention", "full_attention"}

# Declared chunk envelope.
#
# Keyed by (num_hidden_layers, hidden_size) -> (max_kv_len, safe chunk ceiling).
# Public exact requests are allowed only at-or-below a positive ceiling; a zero
# row admits auto only. Deny-by-default: a size with no row here
# cannot prefill at all, because the C++ planner refuses an undeclared handle.
#
# Long-prefill rows describe the implemented language capability. They are
# ceilings, not pins: the shared planner still selects only native-feasible
# SDPA/SPM stage tuples at or below each row.  The historical constant name is
# retained for compatibility, but the current-ref hardware status of these
# enlarged rows remains unverified on hardware until profile validation completes.
CERTIFIED_CHUNK_ENVELOPE: dict[tuple[int, int], tuple[int, int]] = {
    (24, 1024): (8192, 512),  # qwen3_5-0.8b
    (24, 2048): (8192, 640),  # qwen3_5-2b
    (32, 2560): (8192, 448),  # qwen3_5-4b
    (32, 4096): (8192, 256),  # qwen3_5-9b
    # Legacy W8A16 text-only candidate; current-runtime HW unverified.
    # Historical C192 numerics failed. AUTO/exact remain inside C128 and the
    # actual native domain; P8192 is a capacity declaration, not certification.
    (64, 5120): (8192, 128),
}

def _qwen3_5_prefill_envelope(state) -> tuple[int, int]:
    """Return this Language owner's declared native envelope, not a Vision gate.

    A specialized owner may have re-declared a narrower native envelope.
    Legacy cold chunk caps remain independent constraints in the native domain.
    """
    return getattr(state, "prefill_capability_envelope", state.chunk_envelope)


def _declare_qwen3_5_chunk_envelope(
    handle, num_layers, hidden_size
) -> tuple[int, int]:
    import torch as _torch
    key = (int(num_layers), int(hidden_size))
    env = CERTIFIED_CHUNK_ENVELOPE.get(key)
    if env is None:
        raise UnsupportedModelError(
            f"Qwen3.5: no declared chunk envelope for (num_hidden_layers, "
            f"hidden_size)={key}. This profile has no implemented/qualified "
            f"hardware envelope; running it on the auto chunk planner risks "
            f"busting SPM and WEDGING the board. Measure it on a resettable "
            f"board, then add "
            f"a row here AND to docs/roadmap/chunk_certified_envelope.md."
        )
    _torch.ops.rpu.qwen3_5_set_chunk_envelope(handle, env[0], env[1])
    return env


def _validate_layer_types(config) -> None:
    """Fail-fast on a missing / wrong-length / unknown-valued layer_types."""
    tc = _text_config(config)
    lt = list(getattr(tc, "layer_types", []) or [])
    n = int(tc.num_hidden_layers)
    if len(lt) != n:
        raise UnsupportedModelError(
            f"Qwen3.5 layer_types length {len(lt)} != num_hidden_layers {n}."
        )
    bad = sorted(set(lt) - _VALID_LAYER_TYPES)
    if bad:
        raise UnsupportedModelError(
            f"Qwen3.5 unsupported layer_types {bad}; "
            f"expected only {sorted(_VALID_LAYER_TYPES)}."
        )


def layer_types_to_is_full(config):
    """Return a per-layer list[int] (1=full_attention, 0=linear_attention)."""
    _validate_layer_types(config)
    lt = list(_text_config(config).layer_types)
    return [1 if t == "full_attention" else 0 for t in lt]


def split_q_gate(qg_weight, num_heads, head_dim):
    """Split a fused Qwen3.5 q_proj weight [num_heads*head_dim*2, in] into
    (query, gate), each [num_heads*head_dim, in], matching HF's per-head
    interleaved layout (q_proj(h).view(*, num_heads, head_dim*2).chunk(2, -1)).
    Rows are [h0_query h0_gate h1_query h1_gate ...], NOT [all_query | all_gate]."""
    inn = qg_weight.shape[1]
    w = qg_weight.view(num_heads, 2 * head_dim, inn)
    query = w[:, :head_dim, :].reshape(num_heads * head_dim, inn).contiguous()
    gate = w[:, head_dim:, :].reshape(num_heads * head_dim, inn).contiguous()
    return query, gate


def split_q_gate_scale(qg_scale, num_heads, head_dim):
    """Apply :func:`split_q_gate` to a channel-first quant scale."""
    tail = qg_scale.shape[1:]
    scale = qg_scale.reshape(num_heads, 2 * head_dim, *tail)
    query = scale[:, :head_dim].reshape(num_heads * head_dim, *tail).contiguous()
    gate = scale[:, head_dim:].reshape(num_heads * head_dim, *tail).contiguous()
    return query, gate


def _replicate_kv_heads(w, nkv, head_dim, rep):
    """GQA KV replication for FULL-ATTENTION only: [nkv*head_dim, hidden] →
    [nkv*rep*head_dim, hidden], each KV head copied `rep` times consecutively
    (repeat_interleave). With rep = NUM_CORES//nkv this makes nkv_eff == NUM_CORES so
    every core owns one complete KV head (per-core norm/rope path, no cross-core GQA
    kernel). Core i gets KV head i//rep, matching the GQA grouping. GDN is untouched —
    it keeps its own dims + runtime GQA tile; only full-attn's num_kv_heads changes."""
    if rep == 1:
        return w
    hidden = w.shape[1]
    return (w.view(nkv, head_dim, hidden)
             .repeat_interleave(rep, dim=0)
             .reshape(nkv * rep * head_dim, hidden)
             .contiguous())


def _replicate_kv_head_scale(scale, nkv, head_dim, rep):
    """Replicate a channel-first KV scale exactly like its weight rows."""
    if rep == 1:
        return scale
    tail = scale.shape[1:]
    return (scale.reshape(nkv, head_dim, *tail)
            .repeat_interleave(rep, dim=0)
            .reshape(nkv * rep * head_dim, *tail)
            .contiguous())


def _as_channel_first(scale):
    """Convert group-wise ``[G,N]`` to ``[N,G]``; keep W8 ``[N]``."""
    return scale if scale.dim() == 1 else scale.t().contiguous()


def _scale_to_rpu(scale, partition, num_cores, group_size):
    """Finalize a channel-first scale for current main's Linear ABI."""
    if scale.dim() <= 1:
        out = scale
    else:
        from rpu_backend.quant.int4_pgrp_pack import swizzle_int4_pgrp_scale

        out = swizzle_int4_pgrp_scale(
            scale.t().contiguous(), group_size, partition, num_cores)
    return out.detach().to("rpu").contiguous()


def _projection_scale(linear, name):
    """Return a validated channel-first scale, or an empty tensor for FP16."""
    import torch

    weight = linear.weight
    scale = getattr(linear, "weight_scale", None)
    if weight.dtype == torch.int8:
        if not isinstance(scale, torch.Tensor):
            raise RPUBackendError(
                f"Qwen3.5 quantized projection {name} has int8 weight but no "
                "weight_scale")
        if scale.dtype != torch.float16 or scale.dim() not in (1, 2):
            raise RPUBackendError(
                f"Qwen3.5 quantized projection {name} scale must be 1D or 2D "
                f"fp16, got dtype={scale.dtype}, shape={tuple(scale.shape)}")
        n_out = weight.size(0)
        if scale.dim() == 1:
            if scale.numel() != n_out:
                raise RPUBackendError(
                    f"Qwen3.5 W8A16 projection {name} scale numel="
                    f"{scale.numel()} != out_features={n_out}")
        else:
            groups, channels = scale.shape
            k_in = weight.size(1)
            if channels != n_out or groups <= 0 or k_in % groups != 0:
                raise RPUBackendError(
                    f"Qwen3.5 W4A16 projection {name} scale shape "
                    f"{tuple(scale.shape)} incompatible with weight "
                    f"{tuple(weight.shape)}; expected [K/group_size, {n_out}]")
        # Installation starts after the text stack is moved to RPU.  Scale
        # swizzling is a controller-layout transform and must stay on CPU;
        # only its finalized payload is uploaded below.
        return _as_channel_first(scale.detach().to("cpu").contiguous())
    if weight.dtype != torch.float16:
        raise UnsupportedModelError(
            f"Qwen3.5 projection {name} must be fp16 or quantized int8, got "
            f"{weight.dtype}")
    if isinstance(scale, torch.Tensor) and scale.numel() != 0:
        raise RPUBackendError(
            f"Qwen3.5 FP16 projection {name} unexpectedly carries weight_scale")
    return torch.empty(0)


def _resolve_relpath(inner, relpath):
    obj = inner
    for part in relpath.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _iter_text_projections(inner, is_full):
    for relpath in quant_relpaths(is_full):
        yield _resolve_relpath(inner, relpath), relpath


def _validate_projection_specs(inner, is_full, quant_config):
    """Validate every scoped projection against its config-driven QuantSpec."""
    import torch

    try:
        specs = resolve_quant_specs(
            is_full, normalize_quant_config(is_full, quant_config))
    except ValueError as exc:
        raise UnsupportedModelError(str(exc)) from exc
    for linear, name in _iter_text_projections(inner, is_full):
        scale = _projection_scale(linear, name)
        if linear.weight.dtype == torch.float16:
            actual_method = FP16
        else:
            actual_method = W8A16 if scale.dim() == 1 else W4A16_PGRP
        spec = specs[name]
        if actual_method != spec.method:
            raise RPUBackendError(
                f"Qwen3.5 projection {name} declares method={spec.method} but "
                f"its tensors encode {actual_method}")
        if spec.method == W4A16_PGRP:
            actual_group_size = linear.weight.size(1) // scale.size(1)
            if actual_group_size != spec.group_size:
                raise RPUBackendError(
                    f"Qwen3.5 projection {name} declares group_size="
                    f"{spec.group_size} but its scale encodes "
                    f"group_size={actual_group_size}")
    return specs


def _build_partial_mrope_cos_sin(head_dim, rotary_dim, rope_theta, max_seq,
                                 dtype=None, device="rpu"):
    """Build the [max_seq, rotary_dim/2] cos/sin tables for the `partial_mrope`
    kernel. The kernel reads only rotary_dim/2 columns per token and rotates the
    first rotary_dim dims (rotate-half WITHIN rotary_dim). Indexed by token ==
    position (text-only M-RoPE degenerates to 1D partial RoPE since T==H==W);
    pos_offset selects the chunk start. inv_freq uses the rotary_dim base
    (Qwen3.5 partial_rotary_factor)."""
    import torch
    if dtype is None:
        dtype = torch.float16
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    )  # [rotary_dim/2]
    pos = torch.arange(max_seq, dtype=torch.float64)
    freqs = pos[:, None] * inv_freq[None, :]            # [max_seq, rotary_dim/2]
    return (freqs.cos().to(dtype).to(device).contiguous(),
            freqs.sin().to(dtype).to(device).contiguous())


def _apply_interleaved_mrope(freqs, mrope_section):
    """M-RoPE T/H/W interleave.
    freqs: [3, 1, N, rd/2] (T/H/W). Returns [1, N, rd/2] with H/W columns strobed
    in. Text-only (T==H==W) ⇒ no-op, degenerates to arange×inv_freq."""
    import torch  # noqa: F401
    freqs_t = freqs[0].clone()
    for dim_idx, offset in enumerate((1, 2), start=1):        # H=1, W=2
        length = mrope_section[dim_idx] * 3
        freqs_t[..., slice(offset, length, 3)] = freqs[dim_idx, ..., slice(offset, length, 3)]
    return freqs_t


def _build_interleaved_mrope_cos_sin(position_ids, rotary_dim, rope_theta,
                                     mrope_section, dtype=None, device="rpu"):
    """PER-FORWARD prefill cos/sin for the `partial_mrope` kernel (B channel).
    position_ids: [3, N] int (T/H/W per TOKEN — token-indexed, NOT position-lookup).
    Returns cos/sin [N, rotary_dim/2] fp16; row t = token t's interleaved M-RoPE
    freq. Initial prefill runs at position 0, so N covers its padded token span.
    Text-only ⇒ position_ids = arange (interleave degenerates to the arange table
    === set_weights' static cos_)."""
    import torch
    if dtype is None:
        dtype = torch.float16
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    )                                                             # [rd/2]
    pos_3s = position_ids.to(torch.float64)                      # [3, N]
    inv_freq_exp = inv_freq[None, None, :, None].expand(3, 1, -1, 1)  # [3,1,rd/2,1]
    pos_exp = pos_3s[:, None, None, :]                           # [3,1,1,N]
    freqs = (inv_freq_exp @ pos_exp).transpose(2, 3)            # [3,1,N,rd/2]
    freqs_merged = _apply_interleaved_mrope(freqs, mrope_section)    # [1,N,rd/2]
    cos = freqs_merged.cos().squeeze(0).to(dtype).to(device).contiguous()
    sin = freqs_merged.sin().squeeze(0).to(dtype).to(device).contiguous()
    return cos, sin


def _get_prefill_mrope_tables(state, position_ids, rotary_dim, rope_theta,
                              mrope_section):
    """Build M-RoPE tables, or reuse an adapter-opted-in exact one-entry cache."""
    import torch

    cached = getattr(state, "prefill_rope_cache", None)
    if cached is not None and torch.equal(cached[0], position_ids):
        return cached[1], cached[2]
    cos, sin = _build_interleaved_mrope_cos_sin(
        position_ids, rotary_dim, rope_theta, mrope_section)
    if hasattr(state, "prefill_rope_cache"):
        state.prefill_rope_cache = (position_ids.clone(), cos, sin)
    return cos, sin


def _normalize_prefill_position_ids(position_ids, *, real_len, padded_len):
    """Return token-indexed T/H/W positions covering an initial prefill.

    HF generation supplies ``[4, batch, seq]`` (text + T/H/W), while direct
    forwards commonly supply only the three M-RoPE lanes.
    """
    import torch

    pid = position_ids.detach().to("cpu")
    if pid.dim() == 3:
        if pid.size(1) != 1:
            raise ValueError(
                "Qwen3.5 RPU fused prefill supports batch_size == 1 for "
                "position_ids.")
        pid = pid[:, 0, :]
    if pid.dim() != 2 or pid.size(0) not in (1, 3, 4):
        raise ValueError(
            "Qwen3.5 position_ids must be [1, seq], [3, seq], "
            "[3, 1, seq], or HF generation form [4, 1, seq].")
    if pid.size(0) == 4:
        pid = pid[1:]  # HF lane 0 is text position; lanes 1..3 are T/H/W.
    elif pid.size(0) == 1:
        pid = pid.expand(3, -1)

    real_len = int(real_len)
    padded_len = int(padded_len)
    supplied_len = int(pid.size(1))
    if supplied_len < real_len:
        raise ValueError(
            f"Qwen3.5 position_ids length {supplied_len} is shorter than "
            f"the current input length {real_len}.")

    table = pid[:, :padded_len].clone()

    # Route-B pad rows are discarded and excluded from GDN carried state, but
    # the kernel still indexes them. Continue each lane to provide in-bounds,
    # deterministic values without changing the supplied real-token positions.
    if table.size(1) < padded_len:
        count = padded_len - table.size(1)
        base = table[:, -1:] if table.size(1) else torch.zeros(
            3, 1, dtype=pid.dtype)
        steps = torch.arange(1, count + 1, dtype=pid.dtype).view(1, -1)
        table = torch.cat([table, base + steps], dim=1)
    return table.contiguous()


def _drop_tensor_data(owner, name):
    tensor = getattr(owner, name, None)
    if tensor is not None and tensor.numel() > 0:
        tensor.data = tensor.data.new_empty(0)


def _drop_linear_data(linear):
    _drop_tensor_data(linear, "weight")
    _drop_tensor_data(linear, "weight_scale")


def _drop_linears_data(owner, names):
    for name in names:
        linear = getattr(owner, name, None)
        if linear is not None:
            _drop_linear_data(linear)


def _drop_raw_hf_text_layer_weights(layer, is_full):
    """Drop one layer after every derived install tensor owns its storage."""
    _drop_linears_data(layer.mlp, ("gate_proj", "up_proj", "down_proj"))
    _drop_tensor_data(layer.input_layernorm, "weight")
    _drop_tensor_data(layer.post_attention_layernorm, "weight")
    if is_full:
        _drop_linears_data(
            layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj"))
        _drop_tensor_data(layer.self_attn.q_norm, "weight")
        _drop_tensor_data(layer.self_attn.k_norm, "weight")
        return
    linear_attn = layer.linear_attn
    _drop_linears_data(
        linear_attn,
        ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
         "out_proj", "conv1d"),
    )
    _drop_tensor_data(linear_attn.norm, "weight")
    _drop_tensor_data(linear_attn, "A_log")
    _drop_tensor_data(linear_attn, "dt_bias")


def _drop_raw_hf_text_weights(inner, is_full):
    """Drop all decoder sources while preserving tied ``embed_tokens``."""
    if not rpu_env_bool("RPU_QWEN3_5_FREE_HF_WEIGHTS", default=True):
        return
    for i, layer in enumerate(inner.layers):
        _drop_raw_hf_text_layer_weights(layer, bool(is_full[i]))
    _drop_tensor_data(inner.norm, "weight")


def _validate_qwen35_linear_accumulation(config, execution_config):
    """Resolve the cold Linear policy for FP16, W8A16 and W4A16 weights."""
    value = (execution_config or {}).get("prefill", {}).get("linear_acc32", False)
    if not isinstance(value, bool):
        raise TypeError("Qwen3.5 prefill.linear_acc32 must be bool")
    return value


def install_qwen3_5_text_for_rpu(
    text_model,
    config,
    max_seq_len=8192,
    *,
    execution_config=None,
    _resolved_options=None,
    _cpu_stage_weights: bool = False,
    _graph_runtime_policy=None,
    _public_prefill_graph_mode=None,
):
    """Swizzle the text-decoder weights, create the C++ Qwen3_5 model, and
    ``set_weights``. Returns the C++ handle.

    ``text_model`` is the HF ``Qwen3_5TextModel`` backbone (``model.model`` for the
    text-only causal-LM, or ``model.model.language_model`` for the VL wrapper).
    ``config`` is the TOP model config (text fields under ``config.text_config``).

    Per-model forward state (graph cache, dims, rope params, handle finalizer,
    and the handle itself) is stashed on ``text_model._rpu_qwen3_5`` — a
    non-``*_handle`` namespace, so it does not trip the HND-02 model-instance
    handle invariants. lm_head is a ForConditionalGeneration concern and is handled
    by the orchestration ``to_rpu`` (``qwen3_5.py``), NOT here.

    Full-attention layers use the Qwen3 causal-decoder path; the fused
    q_proj [query;gate] is split (query → q linear; gate stored, deferred). GDN
    layers carry their decomposed mixer weights (decode recurrent + prefill chunk).
    """
    import rpu_backend as _rb
    import torch

    if validate_legacy_text_profile(config):
        if (_graph_runtime_policy is None or not
                _graph_runtime_policy.qwen35_legacy_27b_sdk_budget or _cpu_stage_weights):
            raise UnsupportedModelError(
                "Qwen3.8-27B must use the public text-only adapter with its "
                "prepared cold graph policy")

    if hasattr(text_model, "_rpu_qwen3_5"):
        raise RPUBackendError(
            "Qwen3.5 text runtime is already installed; repeated installation "
            "would swizzle the same weights twice."
        )
    if getattr(text_model, "_rpu_qwen3_5_text_install_started", False):
        raise RPUBackendError(
            "Qwen3.5 text installation previously started an irreversible "
            "weight transform; reload the model before retrying."
        )

    execution_config = {} if execution_config is None else execution_config
    _validate_qwen35_linear_accumulation(config, execution_config)
    if not hasattr(torch.ops.rpu, "qwen3_5_set_linear_acc32"):
        raise RuntimeError("Qwen3.5 binary lacks qwen3_5_set_linear_acc32; rebuild before installing")
    from rpu_backend.adapters.qwen3_5.cores import (
        validate_core_profile, validate_text_weight_geometry, validate_text_mlp_padding_storage,
    )
    topology = validate_core_profile(config, execution_config)
    validate_text_weight_geometry(text_model, config, topology)
    validate_text_mlp_padding_storage(text_model, config, topology)
    if _public_prefill_graph_mode not in (None, "RETAINED_CACHE", "BOUNDED_ONESHOT"):
        raise ValueError("Qwen3.5 public prefill lifecycle is invalid")
    retained_prefill = (not _cpu_stage_weights and
        (_public_retained_prefill(config) if _public_prefill_graph_mode is None
         else _public_prefill_graph_mode == "RETAINED_CACHE"))
    if _public_prefill_graph_mode == "RETAINED_CACHE" and (
            _cpu_stage_weights or not _public_retained_prefill(
                config, execution_config, public_owner=True)):
        raise ValueError("Qwen3.5 public retained prefill conflicts with the cold owner scope")
    graph_cache_options = {"max_entries": 1} if _graph_runtime_policy is not None else {}
    if _graph_runtime_policy is not None:
        if not isinstance(_graph_runtime_policy, _rb.graph.GraphRuntimePolicy):
            raise TypeError("Qwen3.5 text graph policy must be a GraphRuntimePolicy")
        if _graph_runtime_policy.execution_core_count != topology.num_cores:
            raise ValueError("Qwen3.5 text graph policy conflicts with execution topology")
        if _graph_runtime_policy.graph_arena_count:
            from rpu_backend.adapters.qwen3_5.cores import validate_dense_9b_vl_profile

            validate_dense_9b_vl_profile(config)
            if _graph_runtime_policy.graph_arena_count != 4:
                raise ValueError("Qwen3.5 9B text requires exactly four cold graph arenas")
    elif topology.num_cores != 8:
        _graph_runtime_policy = _rb.graph.GraphRuntimePolicy.from_environment(
            execution_core_count=topology.num_cores)
    if retained_prefill and getattr(_text_config(config), "hidden_size", None) == 4096:
        expected_arenas = 0 if getattr(config, "quant_config", None) else 4
        actual_arenas = getattr(_graph_runtime_policy, "graph_arena_count", 0)
        if actual_arenas != expected_arenas:
            raise ValueError("Qwen3.5 retained 9B text conflicts with its cold graph arena policy")
        if not hasattr(torch.ops.rpu, "qwen3_5_set_retained_prefill_graph"):
            raise RuntimeError("Qwen3.5 binary lacks retained prefill binding; rebuild before installing")
    if topology.num_cores != 8:
        for name in ("qwen3_5_set_execution_cores", "qwen3_5_get_execution_topology"):
            if not hasattr(torch.ops.rpu, name):
                raise RuntimeError(f"Qwen3.5 binary lacks reduced-core op {name}; rebuild before installing")
    (
        max_seq_len,
        chunk_size_cap,
        padding_budget,
        control_snapshot,
    ) = _resolve_text_install_options(
        max_seq_len,
        execution_config=execution_config,
        resolved_options=_resolved_options,
    )
    prefill_config = execution_config.get("prefill", {})
    requested_chunk = prefill_config.get("chunk_size", "auto")
    exact_chunk_size = (
        0 if requested_chunk == "auto" else int(requested_chunk)
    )
    if exact_chunk_size and exact_chunk_size % 64:
        raise ValueError(
            "Qwen3.5 prefill chunk_size must be a multiple of 64, got "
            f"{exact_chunk_size}"
        )
    if exact_chunk_size:
        # The new exact control supersedes the legacy cap environment knob.
        chunk_size_cap = 0
    padding_budget = int(
        prefill_config.get("padding_budget", padding_budget)
    )
    padding_rows = prefill_config.get("padding_rows", "auto")
    # These objects own native graph/queue resources. Prepare them before the
    # irreversible weight transform, then explicitly clear them on every failed
    # path instead of relying on Python GC.
    pending_state = _Qwen35NativeState(
        handle=None, handle_finalizer=None, graph_cache=None, prefill_graph=None)
    try:
        if _graph_runtime_policy is None:
            graph_cache = pending_state.graph_cache = _rb.graph.GraphCache()
            prefill_graph = pending_state.prefill_graph = _rb.graph.Graph()
        else:
            graph_cache = pending_state.graph_cache = _rb.graph.GraphCache(
                **graph_cache_options, runtime_policy=_graph_runtime_policy)
            prefill_graph = pending_state.prefill_graph = _rb.graph.Graph(
                runtime_policy=_graph_runtime_policy)
        pending_state.public_retained_prefill = retained_prefill
        if pending_state.public_retained_prefill:
            pending_state.prefill_graph_cache = _rb.graph.GraphCache(
                max_entries=1, runtime_policy=graph_cache.runtime_policy)
            pending_state.prefill_graph_sig = None
            pending_state.prefill_rope_cache = None
            pending_state.prefill_rope_shape = None
        return _install_qwen3_5_text_for_rpu_impl(
            text_model,
            config,
            max_seq_len=max_seq_len,
            chunk_size_cap=chunk_size_cap,
            exact_chunk_size=exact_chunk_size,
            padding_budget=padding_budget,
            padding_rows=padding_rows,
            execution_config=execution_config,
            control_snapshot=control_snapshot,
            graph_cache=graph_cache,
            prefill_graph=prefill_graph,
            cpu_stage_weights=bool(_cpu_stage_weights),
            pending_state=pending_state,
        )
    except BaseException as error:
        try:
            _retire_failed_qwen3_5_text_install(text_model, state=pending_state)
        except BaseException as cleanup_error:
            error.add_note(f"Qwen3.5 text install cleanup failed: {cleanup_error!r}")
        raise


def _install_qwen3_5_text_for_rpu_impl(
    text_model,
    config,
    *,
    max_seq_len,
    chunk_size_cap,
    exact_chunk_size,
    padding_budget,
    padding_rows,
    execution_config,
    control_snapshot,
    graph_cache,
    prefill_graph,
    cpu_stage_weights,
    pending_state,
):
    """Implementation for :func:`install_qwen3_5_text_for_rpu`.

    The public wrapper owns graph cleanup until this function publishes the
    complete ``_rpu_qwen3_5`` namespace.
    """
    import torch
    from rpu_backend.runtime.weights import (
        convert_linear_weights_inplace, transform_linear_weight, pad_decoder_mlp_projection,
    )

    inner = text_model
    linear_acc32 = _validate_qwen35_linear_accumulation(config, execution_config)
    # RPU kernels are fp16-only. Like every rpu_backend adapter, the swizzle
    # (convert_linear_weights_inplace / transform_linear_weight) PRESERVES weight
    # dtype (so int8/w8a16 quant paths stay intact) and relies on the CALLER having
    # loaded the model as fp16. Fail fast with a clear message here instead of a
    # cryptic "expected Half but found BFloat16" deep in qwen3_5_forward. Check an
    # ACTUAL weight, not config.dtype (the latter stays bfloat16 even after .half()):
    # some Qwen3.5 ckpts (e.g. 4B) carry text_config.dtype=bfloat16 which overrides
    # from_pretrained(dtype=torch.float16), so the caller must .half() the model.
    tc = _text_config(config)
    adaptive_mode = getattr(tc, "adaptive_mode", None)
    if adaptive_mode not in (None, "adaLN"):
        raise UnsupportedModelError(
            f"Qwen3.5 RPU adapter does not support adaptive_mode={adaptive_mode!r}."
        )
    is_adaptive = adaptive_mode == "adaLN"
    _wdt = (
        inner.layers[0].mlp.gate_proj.weight.dtype
        if is_adaptive
        else inner.norm.weight.dtype
    )
    if _wdt != torch.float16:
        raise UnsupportedModelError(
            f"Qwen3.5 RPU adapter needs fp16 weights (RPU kernels are fp16-only), "
            f"but the model loaded as {_wdt}. Call .half() before to_rpu(), e.g. "
            f"AutoModelForCausalLM.from_pretrained(..., dtype=torch.float16).half(). "
            f"(Some Qwen3.5 ckpts set text_config.dtype=bfloat16, which overrides "
            f"from_pretrained(dtype=fp16) — an explicit .half() is required.)"
        )
    if cpu_stage_weights and any(
        parameter.device.type != "cpu" for parameter in inner.parameters()
    ):
        raise UnsupportedModelError(
            "Qwen3.5 CPU-staged installation requires CPU FP16 source weights"
        )
    is_full = layer_types_to_is_full(config)
    if is_adaptive and not all(is_full):
        raise UnsupportedModelError(
            "Qwen3.5 adaptive_mode='adaLN' currently requires all layers to "
            "use full_attention."
        )
    if not all(is_full):
        dk = int(tc.linear_key_head_dim)
        dv = int(tc.linear_value_head_dim)
        if dk != 128 or dv != 128:
            raise UnsupportedModelError(
                f"Qwen3.5 GDN requires key/value head dimensions 128/128, got {dk}/{dv}"
            )
    quant_config = getattr(config, "quant_config", None) or {}
    projection_specs = _validate_projection_specs(
        inner, is_full, quant_config or None)
    if cpu_stage_weights and quant_config:
        raise UnsupportedModelError(
            "Qwen3.5 quantized CPU-staged installation is not enabled; "
            "Vision currently supports FP16 text projections only."
        )
    head_dim = int(tc.head_dim)
    nq = int(tc.num_attention_heads)
    nkv = int(tc.num_key_value_heads)
    # Full-attention GQA (nkv < NUM_CORES): replicate each KV head so nkv_eff==NUM_CORES
    # and full-attn runs on all 8 cores, ONE whole KV head per core (per-core norm/rope
    # path, no cross-core GQA kernel). attn_tp() in C++ = min(NUM_CORES, num_kv_heads())
    # then resolves to 8 automatically. Costs KV-cache size (nkv_eff/nkv×) vs the
    # no-replication attn_tp path, but matches the phase1-decode known-good decode parity.
    from rpu_backend.adapters.qwen3_5.cores import validate_core_profile, topology_tuple
    topology = validate_core_profile(config, execution_config)
    NUM_CORES = topology.attn_tp
    mlp_cores = topology.mlp_tp
    from rpu_backend.runtime.topology import decoder_mlp_intermediate_size
    mlp_physical_size = decoder_mlp_intermediate_size(tc.intermediate_size, mlp_cores)
    legacy_text = validate_legacy_text_profile(config)
    # REVERT #1 (attn_tp → 8-core + KV replication): full-attn runs on all 8 cores
    # like the phase1-decode known-good. Replicate each KV head rep=8//nkv times so
    # nkv_eff==NUM_CORES (one whole KV head/core, per-core norm/rope path). GDN is NOT
    # affected — it uses its own dims + a runtime GQA tile; only full-attn's num_kv_heads
    # (and thus the KV cache) changes here.
    kv_rep = qwen3_5_kv_replication(config, NUM_CORES)
    nkv_eff = nkv * kv_rep

    # Every quant-scope Linear uses the per-QuantSpec path below: q/k/v and GDN
    # need Qwen3.5-specific derivations, while W4 needs nibble packing.  Each
    # layer's source fields are dropped immediately after its final tensors have
    # been appended; retaining every raw source until set_weights doubles 9B's
    # live weight set.
    # The marker intentionally survives both success and failure. A successful
    # install is guarded by the published namespace; if an outer orchestration
    # step later fails and removes that namespace, this marker still prevents a
    # direct second install from double-swizzling the same model.
    inner._rpu_qwen3_5_text_install_started = True
    convert_linear_weights_inplace(
        inner.layers,
        skip_names={"q_proj", "k_proj", "v_proj", "o_proj", "lm_head",
                    "gate_proj", "up_proj", "down_proj",
                    "in_proj_qkv", "in_proj_z", "in_proj_b",
                    "in_proj_a", "out_proj"})
    free_raw_hf_weights = rpu_env_bool(
        "RPU_QWEN3_5_FREE_HF_WEIGHTS", default=True)

    # ── GDN mixer weight prep for the 8-core (2-head/core) fused mixer ──
    # in_proj is 8-core col-partition: each core's contiguous output slice must
    # be [2 q-heads | 2 k-heads | 2 v-heads] for heads {2c,2c+1}. So reorder
    # the conv_dim channels of in_proj_qkv (+ the depthwise conv weight + the
    # conv_state cache layout) by this permutation, THEN col-swizzle. z/b/a are
    # naturally head-ordered (col-part gives 2/core). out_proj is ROW-partition
    # (each core owns the K-slice for its 2 heads) + all_reduce.
    _gnvh = int(tc.linear_num_value_heads)
    _gnkh = int(tc.linear_num_key_heads)
    _gdk = int(tc.linear_key_head_dim)
    _gdv = int(tc.linear_value_head_dim)
    _gHc = _gnvh // NUM_CORES

    _gksz = _gnkh * _gdk                   # q proj size = k proj size (2048, no rep)

    def _prep_weight(w, partition, spec, *, num_cores=NUM_CORES):
        """Swizzle FP16/W8, or swizzle+nibble-pack W4, for one ``[N,K]``."""
        wc = w.detach().to("cpu").contiguous()
        if spec.method != W4A16_PGRP:
            out = transform_linear_weight(wc, partition, num_cores)
        else:
            from rpu_backend.quant.int4_pgrp_pack import swizzle_pack_int4_pgrp

            n, k = wc.shape
            out = (swizzle_pack_int4_pgrp(wc, partition, num_cores)
                   .reshape(n, k // 2).to(dtype=torch.uint8).contiguous())
        return out if cpu_stage_weights else out.to("rpu")

    def _dense_weight(linear, name, partition):
        weight = linear.weight.data
        if ".mlp." in name and mlp_physical_size != tc.intermediate_size:
            pad = pad_decoder_mlp_projection
            if legacy_text:
                from rpu_backend.adapters.qwen3_5.weights import pad_legacy_mlp_projection

                pad = pad_legacy_mlp_projection
            weight = pad(
                weight.detach().to("cpu").contiguous(),
                projection_name=name.rsplit(".", 1)[-1],
                logical_size=tc.intermediate_size, physical_size=mlp_physical_size)
        return _prep_weight(
            weight, partition, projection_specs[name],
            num_cores=mlp_cores if ".mlp." in name else NUM_CORES)

    def _sw_col(w, spec):
        return _prep_weight(w, 1, spec)

    # Keep an aligned scalar row/core. Legacy GDN4 needs16 slots for12 heads;
    # all existing GDN8 and small dense GDN4 layouts retain8 slots/core.
    def _pad_scalar(w):
        from rpu_backend.adapters.qwen3_5.weights import pad_gdn_scalar

        return pad_gdn_scalar(w, num_heads=_gnvh, num_cores=NUM_CORES).to("rpu")

    def _sw_row(w, spec):
        return _prep_weight(w, 0, spec)

    # ===== GDN per-path weights — SEPARATE q/k/v proj + per-path conv + N_bg b/a =====
    # Shared by decode (build_gdn recurrent) and prefill (chunk); matches rhino's
    # decomposed prep. in_proj_qkv = [all-q(_gksz) | all-k(_gksz) | all-v(nvh·dv)].
    _gvsz = _gnvh * _gdv                       # v proj width (4096)
    def _qkv_path(w, lo, hi, spec):
        return _prep_weight(w.detach().to("cpu")[lo:hi].contiguous(), 1, spec)
    def _conv_path(la, lo, hi):  # per-path conv slice -> [nc, Kc, (hi-lo)/nc]
        cw = la.conv1d.weight.detach().to("cpu").squeeze(1)[lo:hi]   # [hi-lo, Kc]
        Kc = cw.shape[1]
        wd = (hi - lo) // NUM_CORES
        return cw.reshape(NUM_CORES, wd, Kc).permute(0, 2, 1).contiguous().to("rpu")
    _vg_c = _gnvh // NUM_CORES                 # v heads/core (4)
    _n_bg = ((_vg_c + 15) // 16) * 16          # padded width/core (16)
    def _ba_nbg(w, spec):
        hid = w.shape[1]
        out = torch.zeros(_n_bg * NUM_CORES, hid, dtype=w.dtype)
        wc = w.detach().to("cpu")
        for c in range(NUM_CORES):
            out[c * _n_bg:c * _n_bg + _vg_c] = wc[c * _vg_c:(c + 1) * _vg_c]
        return _prep_weight(out.contiguous(), 1, spec)

    def _scale_rpu(linear, name, partition):
        scale = _projection_scale(linear, name)
        if scale.numel() == 0:
            return empty
        spec = projection_specs[name]
        if legacy_text and ".mlp." in name and mlp_physical_size != tc.intermediate_size:
            from rpu_backend.adapters.qwen3_5.weights import pad_legacy_mlp_scale

            scale = pad_legacy_mlp_scale(scale, projection_name=name.rsplit(".", 1)[-1])
        return _scale_to_rpu(
            scale, partition, mlp_cores if ".mlp." in name else NUM_CORES,
            spec.group_size or 0)

    def _qkv_scale_path(scale, lo, hi, spec):
        if scale.numel() == 0:
            return empty
        return _scale_to_rpu(
            scale.detach().to("cpu")[lo:hi].contiguous(),
            1, NUM_CORES, spec.group_size or 0)

    def _ba_nbg_scale(scale, spec):
        if scale.numel() == 0:
            return empty
        sc = scale.detach().to("cpu")
        out = torch.zeros(_n_bg * NUM_CORES, *sc.shape[1:], dtype=torch.float16)
        for c in range(NUM_CORES):
            out[c * _n_bg:c * _n_bg + _vg_c] = sc[c * _vg_c:(c + 1) * _vg_c]
        return _scale_to_rpu(
            out.contiguous(), 1, NUM_CORES, spec.group_size or 0)

    empty = torch.empty(0)
    q, k, v, o, qn, kn, ag, inrm, postrm, g, u, d = ([] for _ in range(12))
    q_s, k_s, v_s, o_s, ag_s, g_s, u_s, d_s = ([] for _ in range(8))
    # GDN mixer weight lists (parallel to num_layers; full slots = empty).
    g_z, g_out, g_alog, g_dt, g_nrm = ([] for _ in range(5))
    # prefill (chunk) per-path weights: separate q/k/v proj + conv, N_bg b/a
    g_q, g_k, g_v, g_cq, g_ck, g_cv, g_b_bg, g_a_bg = ([] for _ in range(8))
    g_z_s, g_out_s, g_q_s, g_k_s, g_v_s, g_b_bg_s, g_a_bg_s = (
        [] for _ in range(7))
    unit_norm = None
    if is_adaptive:
        if cpu_stage_weights:
            unit_norm = torch.ones(
                int(tc.hidden_size), dtype=torch.float16, device="cpu"
            )
        else:
            unit_norm = inner.layers[0].mlp.gate_proj.weight.new_ones(
                int(tc.hidden_size)
            )
    for i, layer in enumerate(inner.layers):
        if not is_full[i]:
            # GDN layer: standard post_norm + MLP half collected like every
            # layer; the token mixer weights go into the GDN lists below.
            for lst in (q, k, v, o, qn, kn, ag):   # inrm filled below (unified channel)
                lst.append(empty)
            for lst in (q_s, k_s, v_s, o_s, ag_s):
                lst.append(empty)
            postrm.append(layer.post_attention_layernorm.weight.data + 1.0)
            gate_name = f"layers.{i}.mlp.gate_proj"
            up_name = f"layers.{i}.mlp.up_proj"
            down_name = f"layers.{i}.mlp.down_proj"
            g.append(_dense_weight(layer.mlp.gate_proj, gate_name, 1))
            u.append(_dense_weight(layer.mlp.up_proj, up_name, 1))
            d.append(_dense_weight(layer.mlp.down_proj, down_name, 0))
            g_s.append(_scale_rpu(layer.mlp.gate_proj, gate_name, 1))
            u_s.append(_scale_rpu(layer.mlp.up_proj, up_name, 1))
            d_s.append(_scale_rpu(layer.mlp.down_proj, down_name, 0))
            la = layer.linear_attn
            # input_layernorm goes into the unified input_norm channel (inrm); the
            # C++ layer-level rmsnorm reads persistent "norm_w" for ALL layers.
            inrm.append(layer.input_layernorm.weight.data + 1.0)   # Qwen3_5RMSNorm 1+w
            z_name = f"layers.{i}.linear_attn.in_proj_z"
            out_name = f"layers.{i}.linear_attn.out_proj"
            z_spec = projection_specs[z_name]
            out_spec = projection_specs[out_name]
            g_z.append(_sw_col(la.in_proj_z.weight, z_spec))
            g_out.append(_sw_row(la.out_proj.weight, out_spec))
            g_z_s.append(_scale_rpu(la.in_proj_z, z_name, 1))
            g_out_s.append(_scale_rpu(la.out_proj, out_name, 0))
            g_alog.append(_pad_scalar(la.A_log.data))
            g_dt.append(_pad_scalar(la.dt_bias.data))
            g_nrm.append(la.norm.weight.data)   # gated norm: plain weight
            # prefill per-path (separate q/k/v proj + conv, N_bg b/a)
            _qkvw = la.in_proj_qkv.weight
            qkv_name = f"layers.{i}.linear_attn.in_proj_qkv"
            qkv_spec = projection_specs[qkv_name]
            _qkvs = _projection_scale(la.in_proj_qkv, qkv_name)
            g_q.append(_qkv_path(_qkvw, 0, _gksz, qkv_spec))
            g_k.append(_qkv_path(_qkvw, _gksz, 2 * _gksz, qkv_spec))
            g_v.append(_qkv_path(
                _qkvw, 2 * _gksz, 2 * _gksz + _gvsz, qkv_spec))
            g_q_s.append(_qkv_scale_path(_qkvs, 0, _gksz, qkv_spec))
            g_k_s.append(_qkv_scale_path(
                _qkvs, _gksz, 2 * _gksz, qkv_spec))
            g_v_s.append(_qkv_scale_path(
                _qkvs, 2 * _gksz, 2 * _gksz + _gvsz, qkv_spec))
            g_cq.append(_conv_path(la, 0, _gksz))
            g_ck.append(_conv_path(la, _gksz, 2 * _gksz))
            g_cv.append(_conv_path(la, 2 * _gksz, 2 * _gksz + _gvsz))
            b_name = f"layers.{i}.linear_attn.in_proj_b"
            a_name = f"layers.{i}.linear_attn.in_proj_a"
            b_spec = projection_specs[b_name]
            a_spec = projection_specs[a_name]
            g_b_bg.append(_ba_nbg(la.in_proj_b.weight, b_spec))
            g_a_bg.append(_ba_nbg(la.in_proj_a.weight, a_spec))
            g_b_bg_s.append(_ba_nbg_scale(
                _projection_scale(la.in_proj_b, b_name), b_spec))
            g_a_bg_s.append(_ba_nbg_scale(
                _projection_scale(la.in_proj_a, a_name), a_spec))
            del _qkvw, _qkvs
            if free_raw_hf_weights:
                _drop_raw_hf_text_layer_weights(layer, False)
            continue
        for lst in (g_z, g_out, g_alog, g_dt, g_nrm,
                    g_q, g_k, g_v, g_cq, g_ck, g_cv, g_b_bg, g_a_bg):
            lst.append(empty)
        for lst in (g_z_s, g_out_s, g_q_s, g_k_s, g_v_s,
                    g_b_bg_s, g_a_bg_s):
            lst.append(empty)
        a = layer.self_attn
        # q_proj.weight is [num_heads*head_dim*2, hidden] = per-head interleaved
        # [h0_query h0_gate h1_query h1_gate ...] (skipped by the swizzle pass →
        # still raw). Split per-head, then col-swizzle each over all NUM_CORES cores
        # (each core owns nq/NUM_CORES query heads, matching the 8-core GEMM).
        query_raw, gate_raw = split_q_gate(a.q_proj.weight.data, nq, head_dim)
        q_name = f"layers.{i}.self_attn.q_proj"
        q_spec = projection_specs[q_name]
        q.append(_prep_weight(query_raw, 1, q_spec))
        ag.append(_prep_weight(gate_raw, 1, q_spec))
        qg_scale = _projection_scale(a.q_proj, q_name)
        if qg_scale.numel() == 0:
            q_s.append(empty)
            ag_s.append(empty)
        else:
            query_scale, gate_scale = split_q_gate_scale(
                qg_scale.detach().to("cpu"), nq, head_dim)
            q_s.append(_scale_to_rpu(
                query_scale, 1, NUM_CORES, q_spec.group_size or 0))
            ag_s.append(_scale_to_rpu(
                gate_scale, 1, NUM_CORES, q_spec.group_size or 0))
            del query_scale, gate_scale
        # k/v_proj skipped by the swizzle pass → raw; replicate each KV head kv_rep
        # times (nkv_eff==NUM_CORES) then col-swizzle over all 8 cores, so every core
        # owns one whole KV head (per-core norm/rope path, no cross-core GQA kernel).
        k_name = f"layers.{i}.self_attn.k_proj"
        v_name = f"layers.{i}.self_attn.v_proj"
        k_spec = projection_specs[k_name]
        v_spec = projection_specs[v_name]
        k.append(_prep_weight(
            _replicate_kv_heads(a.k_proj.weight.data, nkv, head_dim, kv_rep),
            1, k_spec))
        v.append(_prep_weight(
            _replicate_kv_heads(a.v_proj.weight.data, nkv, head_dim, kv_rep),
            1, v_spec))
        k_scale = _projection_scale(a.k_proj, k_name)
        v_scale = _projection_scale(a.v_proj, v_name)
        k_s.append(empty if k_scale.numel() == 0 else _scale_to_rpu(
            _replicate_kv_head_scale(
                k_scale.detach().to("cpu"), nkv, head_dim, kv_rep),
            1, NUM_CORES, k_spec.group_size or 0))
        v_s.append(empty if v_scale.numel() == 0 else _scale_to_rpu(
            _replicate_kv_head_scale(
                v_scale.detach().to("cpu"), nkv, head_dim, kv_rep),
            1, NUM_CORES, v_spec.group_size or 0))
        o_name = f"layers.{i}.self_attn.o_proj"
        o.append(_dense_weight(a.o_proj, o_name, 0))
        o_s.append(_scale_rpu(a.o_proj, o_name, 0))
        qn.append(a.q_norm.weight.data + 1.0); kn.append(a.k_norm.weight.data + 1.0)
        if is_adaptive:
            # G0.5 action AdaLN supplies its per-step (1+scale) tensors through
            # qwen3_5_action_forward. Identity weights still satisfy the shared
            # handle's norm-weight contract during cold installation.
            inrm.append(unit_norm)
            postrm.append(unit_norm)
        else:
            inrm.append(layer.input_layernorm.weight.data + 1.0)
            postrm.append(layer.post_attention_layernorm.weight.data + 1.0)
        gate_name = f"layers.{i}.mlp.gate_proj"
        up_name = f"layers.{i}.mlp.up_proj"
        down_name = f"layers.{i}.mlp.down_proj"
        g.append(_dense_weight(layer.mlp.gate_proj, gate_name, 1))
        u.append(_dense_weight(layer.mlp.up_proj, up_name, 1))
        d.append(_dense_weight(layer.mlp.down_proj, down_name, 0))
        g_s.append(_scale_rpu(layer.mlp.gate_proj, gate_name, 1))
        u_s.append(_scale_rpu(layer.mlp.up_proj, up_name, 1))
        d_s.append(_scale_rpu(layer.mlp.down_proj, down_name, 0))
        del query_raw, gate_raw, qg_scale, k_scale, v_scale
        if free_raw_hf_weights:
            _drop_raw_hf_text_layer_weights(layer, True)

    # M-RoPE interleaved + partial rotary. The cos/sin tables contain exactly
    # rotary_dim/2 columns; the partial kernels receive rotary_dim explicitly.
    rp = getattr(tc, "rope_parameters", None) or {}
    rotary_dim = int(round(rp.get("partial_rotary_factor", 1.0) * head_dim))
    rope_theta = float(rp.get("rope_theta", getattr(tc, "rope_theta", 1e4)))
    mrope_section = [int(x) for x in rp.get("mrope_section", [])]
    # Decode cos/sin table MUST be as long as the KV cache (max_seq_len): decode
    # indexes cos_[position] with position up to max_seq_len-1, and there is NO
    # bounds check — a table shorter than the cache reads past it (garbage) once
    # decode passes the table length. Threaded from to_rpu(max_seq_len=...); the
    # 8192 default matches the historical hardcode for callers that don't pass it.
    cos, sin = _build_partial_mrope_cos_sin(
        head_dim, rotary_dim, rope_theta, max_seq=max_seq_len)

    if cpu_stage_weights:
        # G0.5 installs Vision before text. Performing swizzle or scalar norm
        # arithmetic after moving the text modules to RPU can enqueue hundreds
        # of eager kernels behind that live runtime and stall installation.
        # Keep every transformation above on CPU and upload only the finalized
        # contiguous FP16 tensors used by set_weights().
        for weights in (
            q, k, v, o, qn, kn, ag, inrm, postrm, g, u, d,
            g_z, g_out, g_alog, g_dt, g_nrm,
            g_q, g_k, g_v, g_cq, g_ck, g_cv, g_b_bg, g_a_bg,
            q_s, k_s, v_s, o_s, ag_s, g_s, u_s, d_s,
            g_z_s, g_out_s, g_q_s, g_k_s, g_v_s, g_b_bg_s, g_a_bg_s,
        ):
            for index, weight in enumerate(weights):
                if weight.numel() and weight.device.type != "rpu":
                    weights[index] = weight.detach().to(
                        device="cpu", dtype=torch.float16
                    ).contiguous().to("rpu")

    gdn_nvh = int(tc.linear_num_value_heads)
    gdn_nkh = int(tc.linear_num_key_heads)
    gdn_dk = int(tc.linear_key_head_dim)
    gdn_dv = int(tc.linear_value_head_dim)
    # No GQA weight replication: conv_dim uses TRUE nkh (q/k stay nkh heads); the
    # nkh→nvh expansion is a runtime tile in build_gdn (12288 → 8192 for 4B).
    gdn_conv_dim = 2 * (gdn_nkh * gdn_dk) + gdn_nvh * gdn_dv   # 8192 for 4B
    gdn_conv_kernel = int(tc.linear_conv_kernel_dim)
    final_norm = (
        unit_norm
        if is_adaptive
        else inner.norm.weight.data + 1.0
    )
    if free_raw_hf_weights:
        _drop_tensor_data(inner.norm, "weight")

    handle = torch.ops.rpu.qwen3_5_create()
    pending_state.handle = handle
    handle_finalizer = None
    try:
        from rpu_backend.runtime._native_retirement import _InstalledNativeResource

        pending_state.retirement = _InstalledNativeResource(
            pending_state, handle, torch.ops.rpu.qwen3_5_destroy,
            graphs=(graph_cache,), raw_graphs=(prefill_graph,),
            keepalive=(q, k, v, o, qn, kn, ag, inrm, postrm, g, u, d, cos, sin,
                       final_norm, g_z, g_out, g_alog, g_dt, g_nrm, g_q, g_k, g_v,
                       g_cq, g_ck, g_cv, g_b_bg, g_a_bg, q_s, k_s, v_s, o_s,
                       ag_s, g_s, u_s, d_s, g_z_s, g_out_s, g_q_s, g_k_s,
                       g_v_s, g_b_bg_s, g_a_bg_s),
            label="Qwen3.5 text", handle_name="handle")
        handle_finalizer = pending_state.retirement.finalizer
        _refresh_qwen35_text_retirement(pending_state)
        if cpu_stage_weights and final_norm.device.type != "rpu":
            final_norm = final_norm.detach().to(
                device="cpu", dtype=torch.float16
            ).contiguous().to("rpu")
            pending_state.retirement.keepalive += (final_norm,)
        if topology.num_cores != 8:
            torch.ops.rpu.qwen3_5_set_execution_cores(handle, topology.num_cores)
        torch.ops.rpu.qwen3_5_set_linear_acc32(handle, linear_acc32)
        torch.ops.rpu.qwen3_5_set_weights(
            handle, q, k, v, o, qn, kn, ag, inrm, postrm, g, u, d,
            cos, sin, final_norm, is_full,
            nq, nkv_eff, head_dim,
            int(tc.hidden_size), int(tc.intermediate_size),
            float(tc.rms_norm_eps), True, mrope_section,
            g_z, g_out, g_alog, g_dt, g_nrm,
            gdn_nvh, gdn_dk, gdn_dv, gdn_conv_dim, gdn_conv_kernel,
            g_q, g_k, g_v, g_cq, g_ck, g_cv, g_b_bg, g_a_bg,
            q_s, k_s, v_s, o_s, ag_s, g_s, u_s, d_s,
            g_z_s, g_out_s, g_q_s, g_k_s, g_v_s, g_b_bg_s, g_a_bg_s)
        if topology.num_cores != 8:
            actual = tuple(torch.ops.rpu.qwen3_5_get_execution_topology(handle))
            if actual != topology_tuple(topology):
                raise RuntimeError(f"Qwen3.5 native/Python topology mismatch: {actual}")
        torch.ops.rpu.qwen3_5_set_chunk_size_cap(handle, chunk_size_cap)
        torch.ops.rpu.qwen3_5_set_prefill_chunk_size(
            handle, exact_chunk_size
        )
        # MR-A: declared chunk envelope, deny-by-default. Must precede the first
        # forward. Rows and current evidence status live by the table above.
        chunk_envelope = _declare_qwen3_5_chunk_envelope(
            handle,
            int(tc.num_hidden_layers),
            int(tc.hidden_size),
        )
        if getattr(pending_state, "public_retained_prefill", False):
            torch.ops.rpu.qwen3_5_set_retained_prefill_graph(handle, True)
        # Decode and eligible public prefill own separate bounded graph caches.
        # Reduced/Vision 9B and internal owners retain their raw-Graph exception.
        #
        # Stash all per-forward state on `text_model._rpu_qwen3_5` (a namespace whose
        # name does NOT end in `_handle`, so it is invisible to the HND-02 model
        # handle-invariant scan). `run_qwen3_5_text` reads it; the ADAPTER mirrors
        # `.handle`/`.graph_cache` onto `self._handle`/`self._graph_cache` (BC2).
        vars(pending_state).update(
            execution_topology=topology,
            handle=handle,
            handle_finalizer=handle_finalizer,
            graph_cache=graph_cache,
            max_seq_len=max_seq_len,
            num_layers=len(is_full),
            hidden_size=int(tc.hidden_size),
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            padding_budget=padding_budget,
            padding_rows=padding_rows,
            execution_config=execution_config,
            linear_acc32=linear_acc32,
            control_snapshot=control_snapshot,
            exact_chunk_size=exact_chunk_size or None,
            chunk_envelope=chunk_envelope,
            adaptive_mode=adaptive_mode,
            has_dispatched=False,
            prefill_plan_key=None,
            prefill_plan=None,
            decode_stage_descriptor=None,
            decode_plan=None,
            last_a6_plan=None,
            # Kept for the reduced/Vision 9B and internal one-shot lifecycle. It
            # remains unused for the bounded public retained prefill path.
            prefill_graph=prefill_graph,
        )
        inner._rpu_qwen3_5 = pending_state
    finally:
        pending_state.handle_finalizer = handle_finalizer
    return handle


def _prepare_qwen35_prefill_capture(st, *, seq_len, real_len, planned_chunk_size, plan):
    """Validate READY and retire stale RoPE users before any native mutation.

    A physical shape change can reallocate native prefill cos/sin backing,
    so its retained graph must be evicted first. Equal shapes retain their
    addresses and refresh values through native copy_.
    """
    import torch
    import rpu_backend as _rb

    prefill_cache = getattr(st, "prefill_graph_cache", None)
    if prefill_cache is None:
        return _oneshot_scope(st.prefill_graph)
    active = prefill_cache
    sig_attr = "prefill_graph_sig"
    family = "qwen3_5" if getattr(st, "public_retained_prefill", False) else "g05_qwen3_5"
    sig = _rb.graph.GraphSignature(
        op_id=f"rpu_{family}_prefill",
        shapes=[seq_len, st.hidden_size],
        dyn_dims=[st.num_layers, real_len, planned_chunk_size,
                  int(getattr(st, "execution_generation", 0)), *plan.graph_key_words()],
        dtypes=[torch.float16])
    if active.is_frozen():
        # A lookup-only READY miss must not change valid_len, RoPE or a sibling
        # cache. capture().__enter__ would otherwise check only after setters.
        active.get_or_create(sig)
    rope_shape = (seq_len, st.rotary_dim // 2) if st.mrope_section else None
    old_shape = getattr(st, "prefill_rope_shape", None)
    if old_shape is not None and rope_shape != old_shape:
        siblings = [(prefill_cache, "prefill_graph_sig")]
        if any(cache is not None and cache.is_frozen() for cache, _ in siblings):
            raise RuntimeError("Qwen3.5 READY cannot replace shared prefill RoPE backing; begin warmup first")
        for cache, attr in siblings:
            old_sig = getattr(st, attr, None)
            if cache is not None and old_sig is not None:
                cache.evict(old_sig)
                setattr(st, attr, None)
    old_sig = getattr(st, sig_attr, None)
    if old_sig is not None and old_sig != sig:
        active.evict(old_sig)
    setattr(st, sig_attr, sig)
    return active.capture(sig)


def _preflight_qwen3_5_text(
    text_model, cache, real_len, *, multimodal_prefill=False, _cost_request=None,
    _cost_position=None
):
    """Resolve the text envelope and A6 plan before any mutating RPU work."""
    import torch

    st = text_model._rpu_qwen3_5
    from rpu_backend.adapters.qwen3_5.cores import core_topology, topology_tuple
    topology = getattr(st, "execution_topology", None) or core_topology()
    expected_topology = topology_tuple(topology)
    actual_topology = getattr(cache, "execution_topology", (8, 8, 8, 8, 8))
    if tuple(actual_topology) != expected_topology:
        raise ValueError("Qwen3.5 cache core layout does not match the installed model; use Qwen3_5Cache.from_model(model, ...)")
    if _cost_position is not None and (type(real_len) is not int or real_len <= 1):
        raise ValueError("cold text prefill length must be an integer greater than one")
    real_len = int(real_len)
    inspection = None
    if _cost_position is not None:
        from rpu_backend.runtime.decoder import _cold_text_cost_request
        from rpu_backend.api._execution import _validate_qwen35_text_padding

        inspection, position = _cold_text_cost_request(
            text_model, ("qwen3_5", int(st.handle)), cache, _cost_request, _cost_position)
        _validate_qwen35_text_padding(inspection.as_dict())
        if position != 0:
            raise NotImplementedError("Qwen3.5 cold prefill planning is supported only at position 0")
        if (multimodal_prefill or getattr(st, "action_mode", False)
                or (getattr(st, "prefill_graph_cache", None) is not None
                    and not getattr(st, "public_retained_prefill", False))):
            raise ValueError("cold text leg inspection requires the original public text owner")
        if inspection.chunk_size is not None and inspection.chunk_size % 64:
            raise ValueError("Qwen3.5 prefill chunk_size must be a multiple of 64")
    profile = (int(st.num_layers), int(st.hidden_size))
    envelope = _qwen3_5_prefill_envelope(st)
    if envelope is None or real_len > envelope[0]:
        limit = None if envelope is None else envelope[0]
        raise UnsupportedModelError(
            f"Qwen3.5 {'multimodal' if multimodal_prefill else 'text'} prefill "
            f"length {real_len} exceeds the declared limit {limit} for "
            f"profile {profile}"
        )
    if cache.position + real_len > cache.max_seq_len:
        raise ValueError(
            f"Qwen3.5 cache capacity exceeded: position={cache.position}, "
            f"input={real_len}, max_seq_len={cache.max_seq_len}."
        )
    if real_len > 1 and cache.position != 0:
        raise NotImplementedError(
            "Qwen3.5 RPU multi-token prefill is supported only at cache "
            "position 0; reset the cache and re-feed the full prefix."
        )
    if real_len <= 1:
        if _cost_request is not None:
            raise ValueError("Qwen3.5 AUTO twin requires a multi-token prefill")
        return real_len, 0, None

    exact_chunk_size = st.exact_chunk_size
    if _cost_request is not None and inspection is None:
        # Only cold cost collection may inspect the AUTO twin of a canonical
        # exact text request. The native oracle already returns the complete
        # domain; its cap and validity ceiling coincide for these cap=0 rows.
        # Never change the installed request, native controls or forward memo.
        from dataclasses import replace
        from rpu_backend.api._execution import _planner_owner_binding, planner_cost_observer
        from rpu_backend.runtime.execution_planner import PlannerRequest

        collecting_costs = planner_cost_observer(
            text_model, "prefill", None, ("qwen3_5", int(st.handle))) is not None
        session, _component = _planner_owner_binding(text_model, "prefill", "")
        if session is None or not collecting_costs:
            raise RuntimeError("Qwen3.5 AUTO twin requires cold cost collection")
        session.require_cold()
        fields = {
            "chunk_size": st.exact_chunk_size or "auto",
            "padding_rows": st.padding_rows,
        }
        if st.padding_rows == "auto":
            fields["padding_budget"] = st.padding_budget
        original = PlannerRequest.from_mapping(fields)
        if (not isinstance(_cost_request, PlannerRequest)
                or original.mode != "EXACT"
                or _cost_request != replace(original, mode="AUTO", chunk_size=None)
                or "chunk_size" not in st.execution_config.get("prefill", {})
                or st.control_snapshot.get("chunk_effective_cap") != 0
                or not envelope or envelope[1] <= 0
                or getattr(st, "action_mode", False)
                or (getattr(st, "prefill_graph_cache", None) is not None
                    and not getattr(st, "public_retained_prefill", False))):
            raise ValueError("Qwen3.5 AUTO twin requires the unchanged canonical cap-zero text request")
        exact_chunk_size = None
    padding_rows, padding_budget = st.padding_rows, st.padding_budget
    planning_override = ()
    if inspection is not None:
        exact_chunk_size = inspection.chunk_size
        padding_rows = "auto" if inspection.padding_rows is None else inspection.padding_rows
        padding_budget = inspection.padding_budget
        planning_override = (inspection.chunk_size or 0,)
    # The shared lifecycle owns reuse, including READY and native identity.
    # Keep the last receipt below for its existing readers, never as a cache.
    plan_graph_cache = getattr(st, "prefill_graph_cache", None)

    plan_box = {}
    pad_len, planned_chunk_size = _plan_prefill_execution(
        real_len,
        cache.allocated_max_seq_len,
        padding_budget,
        execution_owner=text_model,
        execution_native=("qwen3_5", int(st.handle)),
        plan_signature=(real_len, bool(multimodal_prefill), tuple(planning_override)),
        graph_cache=plan_graph_cache,
        resolve_stage_domain=lambda n: (
            torch.ops.rpu.qwen3_5_resolve_prefill_stage_domain(
                st.handle, n, real_len, *planning_override
            )
        ),
        padding_rows=padding_rows,
        exact_chunk_size=exact_chunk_size,
        max_stage_chunk_size=envelope[1],
        plan_result_sink=lambda result: plan_box.__setitem__("result", result),
        queue_owner_id=int(st.handle),
        graph_mode=(
            "RETAINED_CACHE"
            if getattr(st, "prefill_graph_cache", None) is not None
            else "BOUNDED_ONESHOT"
        ),
    )
    result = plan_box["result"]
    if _cost_request is not None:
        return pad_len, planned_chunk_size, result
    st.last_a6_plan = result.as_dict(include_candidates=False)
    st.prefill_plan_key = None
    st.prefill_plan = (pad_len, planned_chunk_size, result)
    return st.prefill_plan


def _plan_qwen3_5_decode(text_model):
    """Observe the real fixed decode ABI; keep public configuration prefill-only."""
    import torch
    from rpu_backend.runtime.decoder import plan_native_component_execution

    state = text_model._rpu_qwen3_5
    native = ("qwen3_5", int(state.handle))
    generation = int(getattr(state, "execution_generation", 0))

    def domain(_length):
        descriptor = tuple(torch.ops.rpu.qwen3_5_resolve_decode_stage_descriptor(state.handle))
        return [1, 1, len(descriptor), *descriptor]

    _, result = plan_native_component_execution(
        1, component_id="language_model", stage="decode", generation=generation,
        execution={}, resolve_stage_domain=domain, graph_mode=GRAPH_RETAINED_CACHE,
        queue_owner_id=int(state.handle), execution_owner=text_model, execution_native=native,
        plan_signature=(), graph_cache=getattr(state, "graph_cache", None))
    state.decode_stage_descriptor = result.selected.stage_tuple.physical_descriptor
    state.decode_plan = (None, result)
    return result


def run_qwen3_5_text(text_model, hidden, cache, *, attention_mask=None,
                     position_ids=None, multimodal_prefill=False,
                     preflight_plan=None):
    """Drive the RPU text decoder for one forward and return the RAW output.

    ``hidden`` is the already-embedded [B, seq, hidden] input (the orchestration
    top forward does the embed + any vision scatter). Returns the decoder output
    ([B, real_len, hidden]). lm_head + output wrapping happen in the orchestration
    top forward.

    ``position_ids`` (optional [3, N] token-indexed T/H/W) drives the prefill
    partial M-RoPE; None ⇒ text-only arange (identical to the static table).
    """
    import torch
    import rpu_backend as _rb

    st = text_model._rpu_qwen3_5
    handle = st.handle
    graph_cache = st.graph_cache
    num_layers = st.num_layers
    hidden_size = st.hidden_size
    rotary_dim = st.rotary_dim
    rope_theta = st.rope_theta
    mrope_section = st.mrope_section
    # CPU→RPU prep MUST stay OUTSIDE capture (a DMA recorded inside would
    # freeze the first call's pointer across replays).
    if hidden.dtype != torch.float16:
        hidden = hidden.to(torch.float16)
    if not hidden.is_contiguous():
        hidden = hidden.contiguous()
    if attention_mask is not None and attention_mask.device.type != "rpu":
        attention_mask = attention_mask.to("rpu")
    seq_len = hidden.shape[1]
    real_len = seq_len
    if preflight_plan is None:
        preflight_plan = _preflight_qwen3_5_text(
            text_model, cache, real_len,
            multimodal_prefill=multimodal_prefill,
        )
    # Route B: first satisfy the mandatory 64-row GDN alignment, then optionally
    # spend the cold padding budget only when the exact C++ planner can reduce
    # the chunk count. Padding remains text-only and bounded by physical KV capacity.
    planned_chunk_size = 0
    planned_stage_descriptor = ()
    prefill_plan_result = None
    if seq_len > 1:
        pad_len, planned_chunk_size, prefill_plan_result = preflight_plan
        assert prefill_plan_result.selected is not None
        planned_stage_descriptor = (
            prefill_plan_result.selected.stage_tuple.physical_descriptor
        )
        if not planned_stage_descriptor:
            raise RuntimeError("Qwen3.5 A6 winner has no native stage descriptor")
        if pad_len > seq_len:
            hidden = torch.cat(
                [hidden, hidden.new_zeros(hidden.shape[0], pad_len - seq_len, hidden.shape[2])],
                dim=1)
            seq_len = pad_len
    is_prefill = seq_len > 1
    if is_prefill:
        prefill_ctx = _prepare_qwen35_prefill_capture(
            st, seq_len=seq_len, real_len=real_len,
            planned_chunk_size=planned_chunk_size, plan=prefill_plan_result)
        torch.ops.rpu.qwen3_5_set_valid_prefill_len(handle, real_len)
    # PREFILL (seq_len>1) partial M-RoPE: build the per-forward interleaved cos/sin
    # (B channel) from THIS forward's 3D position_ids and stash them in the C++ model
    # (build_full_attention reads them for prefill). Token-indexed over the padded
    # initial sequence; text-only ⇒ arange. MUST stay OUTSIDE any capture scope
    # (a DMA recorded inside would freeze the first table's pointer across replays).
    if seq_len > 1 and mrope_section:
        pid = position_ids
        if pid is None:                      # text-only: T==H==W == absolute token index
            rng = torch.arange(0, cache.position + seq_len, dtype=torch.long)
            pid = rng.view(1, -1).expand(3, -1)          # [3, position+seq_len]
        else:
            pid = _normalize_prefill_position_ids(
                pid, real_len=real_len, padded_len=seq_len)
        pcos, psin = _get_prefill_mrope_tables(
            st, pid, rotary_dim, rope_theta, mrope_section)
        torch.ops.rpu.qwen3_5_set_prefill_rope(handle, pcos, psin)
        st.prefill_rope_shape = (seq_len, rotary_dim // 2)
    if is_prefill:
        # READY lookup and every eviction happened before the native setters.
        ctx = prefill_ctx
    else:
        decode_plan_result = _plan_qwen3_5_decode(text_model)
        planned_stage_descriptor = decode_plan_result.selected.stage_tuple.physical_descriptor
        # → 留档复用。position 不进 sig,所以 decode 各步 sig 相同、命中 REPLAYING
        # (KV slot 与 M-RoPE 的 position 都走 set_regs、replay 每次刷新:
        #  rpu_llama_kvcache.cpp:467 / rpu_mrope.cpp:86)。
        # Decode always has real_len == 1. Prefill signatures above retain
        # real_len because GDN valid-tail fills are baked into their graphs.
        ctx = graph_cache.capture(_rb.graph.GraphSignature(
            op_id="rpu_qwen3_5",
            shapes=[seq_len, hidden_size],
            dyn_dims=[
                num_layers, int(getattr(st, "execution_generation", 0)),
                *decode_plan_result.graph_key_words(),
            ],
            dtypes=[torch.float16],
        ))
    with ctx:
        out = torch.ops.rpu.qwen3_5_forward(
            handle, hidden, cache.k_caches, cache.v_caches,
            cache.gdn_states, cache.conv_states,
            attention_mask, cache.position, True,
            0, planned_stage_descriptor)
    st.has_dispatched = True
    if is_prefill:
        resolved = int(torch.ops.rpu.qwen3_5_get_resolved_chunk_size(handle))
        if resolved != planned_chunk_size:
            raise RuntimeError(
                "Qwen3.5 prefill dry/dispatch chunk plan drift: "
                f"planned={planned_chunk_size}, resolved={resolved}."
            )
    else:
        resolved = int(torch.ops.rpu.qwen3_5_get_resolved_chunk_size(handle))
    last_plan = {
        "stage": "prefill" if is_prefill else "decode",
        "component": getattr(
            text_model, "_rpu_execution_component_id", "language_model"
        ),
        "logical_len": int(real_len),
        "execution_len": int(seq_len),
        "chunk_size": resolved,
        "padding_rows": int(seq_len - real_len),
        "position": int(cache.position),
        "graph_mode": (
            prefill_plan_result.graph_mode
            if is_prefill else GRAPH_RETAINED_CACHE
        ),
        "physical_descriptor": tuple(planned_stage_descriptor),
    }
    a6 = (getattr(st, "last_a6_plan", None) if is_prefill
          else decode_plan_result.as_dict(include_candidates=False))
    if a6 is not None:
        last_plan.update({
            "a6_plan_digest": a6["plan_digest"],
            "a6_domain_digest": a6["domain_digest"],
            "a6_optimality": a6["optimality"],
            "a6_search_complete": a6["search_complete"],
        })
    vars(text_model)["_rpu_last_execution_plan"] = last_plan
    # Route B: advance the cache by the REAL length (not the padded seq_len) so decode
    # continues from the real end and overwrites the padded KV; slice the padded tail rows
    # off `out` (batch=1 ⇒ the narrow view is contiguous). The GDN carried states already
    # reflect real_len (build_gdn zeroed the pad tokens via set_valid_prefill_len).
    cache.update_position(real_len)
    if real_len != seq_len:
        out = out[:, :real_len]
    return out
