"""Qwen3.5 text backbone (== HF ``Qwen3_5MoeTextModel``) → rpu_backend.

Sibling module of ``__init__.py`` (the orchestration entry) in the
``adapters/qwen3_5_moe/`` package. This module owns the text-decoder half of the
adapter: config-driven layer dispatch, weight swizzle + GDN mixer prep, the C++
``qwen3_5_moe_*`` handle install, and the per-forward RPU decoder run (route-B pad +
partial M-RoPE + capture). Attention/GDN and forward lifecycle follow the dense
Qwen3.5 implementation, while weight collection and the entire runtime state are
independent and sparse-MoE specific.

Public surface:
  - ``layer_types_to_is_full(config)`` — per-layer full/linear dispatch (re-exported
    by ``__init__.py`` for backward-compat; ~8 tests import it from ``adapters.qwen3_5_moe``).
  - ``install_qwen3_5_moe_text_for_rpu(text_model, config) -> handle`` — swizzle text
    weights, create the C++ model, ``set_weights``; stashes per-model forward state
    on ``text_model._rpu_qwen3_5_moe`` (a non-``*_handle`` namespace) and returns the
    C++ handle (the ADAPTER keeps it as ``self._handle``).
  - ``run_qwen3_5_moe_text(text_model, hidden, cache, ...) -> raw`` — drive the RPU
    decoder for one forward and return the raw output ([B, real_len, hidden]).

The profile guard (``_SUPPORTED_PROFILES`` / ``_check_profile``) intentionally
stays in ``__init__.py`` — tests monkeypatch ``qwen3_5_moe._SUPPORTED_PROFILES`` and
the reader must live in the same module.
"""
from __future__ import annotations

import types
from contextlib import contextmanager

from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError
from rpu_backend.api._execution import _resolve_text_install_options
from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
from rpu_backend.runtime.execution_planner import GRAPH_RETAINED_CACHE


class _Qwen35MoeNativeState(types.SimpleNamespace):
    """Native MoE state whose parent Session owns the exclusive close path."""

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
                    raise RuntimeError("Qwen3.5-MoE retirement lost the live-owner slot")
                close(parent)
                if resource.handle is not None:
                    raise RuntimeError("Qwen3.5-MoE parent GC left a live native resource")
        except BaseException as error:
            from rpu_backend.api._execution import ExecutionSession

            if isinstance(session, ExecutionSession):
                session.poison()
            resource.retain_failure(error, self, getattr(session, "_owner", None))


def _bind_qwen35_moe_retirement(state, parent, close, child):
    session = getattr(parent, "_execution_session", None)
    resource = getattr(state, "retirement", None)
    if (session is None or session._owner is not parent or resource is None
            or resource.owner() is not state or resource.handle != state.handle
            or not any(getattr(child, name, None) is state for name in
                       ("_rpu_qwen3_5_moe", "_rpu_vision_retirement_state"))):
        raise RuntimeError("Qwen3.5-MoE retirement parent identity changed")
    owned = vars(parent).setdefault("_qwen35_retirement_children", [])
    existing = [item for item in owned if item[2] is resource]
    if existing:
        if len(existing) != 1 or not all(left is right for left, right in
                                        zip(existing[0], (child, state, resource, session))):
            raise RuntimeError("Qwen3.5-MoE retirement child identity changed")
    else:
        if resource.parent is not None:
            raise RuntimeError("Qwen3.5-MoE retirement lost its original child record")
        owned.append((child, state, resource, session))
    vars(state).update(_execution_session=session, _retirement_close=close)
    resource.take_ownership(parent)


def _validate_qwen35_moe_retirement_children(parent, children):
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
        raise RuntimeError("Qwen3.5-MoE retirement lost an original child/state/resource/Session")


def _qwen35_moe_retirement_ready(state):
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


def _refresh_qwen35_moe_text_retirement(state):
    resource = getattr(state, "retirement", None)
    if resource is not None:
        resource.graphs = tuple(value for name in ("graph_cache", "prefill_graph_cache")
                                if (value := getattr(state, name, None)) is not None)


def _clear_qwen35_moe_graphs(states, vision=None):
    caches, raw_graphs = [], []
    for state in states:
        child_caches = tuple(getattr(state, name, None)
                             for name in ("graph_cache", "prefill_graph_cache"))
        child_raw = (getattr(state, "prefill_graph", None),)
        resource = getattr(state, "retirement", None)
        if resource is not None and (
                resource.handle != getattr(state, "handle", None)
                or resource.finalizer is not getattr(state, "handle_finalizer", None)):
            raise RuntimeError("Qwen3.5-MoE text retirement handle/finalizer identity changed")
        if resource is not None and (
                {id(graph) for graph in child_caches if graph is not None}
                != {id(graph) for graph in resource.graphs}
                or {id(graph) for graph in child_raw if graph is not None}
                != {id(graph) for graph in resource.raw_graphs}):
            raise RuntimeError("Qwen3.5-MoE text retirement graph identity changed")
        caches.extend(child_caches)
        raw_graphs.extend(child_raw)
    if vision is not None:
        cache = getattr(vision, "_rpu_vision_graph_cache", None)
        raw = getattr(vision, "_rpu_vision_debug_graph", None)
        vision_state = getattr(vision, "_rpu_vision_retirement_state", None)
        resource = getattr(vision_state, "retirement", None)
        handle = (getattr(vision, "_rpu_vision_installing_handle", None)
                  or getattr(vision, "_rpu_vision_handle", None))
        finalizer = (getattr(vision, "_rpu_vision_installing_finalizer", None)
                     or getattr(vision, "_rpu_vision_handle_finalizer", None))
        if resource is not None and (
                handle != resource.handle or handle != getattr(vision_state, "handle", None)
                or finalizer is not resource.finalizer
                or finalizer is not getattr(vision_state, "handle_finalizer", None)):
            raise RuntimeError("Qwen3.5-MoE Vision retirement handle/finalizer identity changed")
        if resource is not None and (
                {id(cache)} - {id(None)} != {id(graph) for graph in resource.graphs}
                or {id(raw)} - {id(None)} != {id(graph) for graph in resource.raw_graphs}):
            raise RuntimeError("Qwen3.5-MoE Vision retirement graph identity changed")
        caches.append(cache)
        raw_graphs.append(raw)
    seen = set()
    for cache in caches:
        if cache is not None and id(cache) not in seen:
            seen.add(id(cache))
            cache.clear()
            if not cache.cache_invariant_ok():
                raise RuntimeError("Qwen3.5-MoE GraphCache invariant failed during retirement")
    for graph in raw_graphs:
        if graph is not None and id(graph) not in seen:
            seen.add(id(graph))
            graph.invalidate()


def _destroy_qwen3_5_moe_handle(handle, state=None):
    """Compatibility callback: only the installed resource retires a handle."""
    if state is None or getattr(state, "retirement_failed", None):
        return
    resource = getattr(state, "retirement", None)
    if resource is not None and resource.handle == handle:
        resource.retire()


def _poison_qwen3_5_moe_text_retirement(inner, state, cleanup_error, session=None):
    state.retirement_failed = repr(cleanup_error)
    vars(inner)["_rpu_qwen3_5_moe"] = state
    resource = getattr(state, "retirement", None)
    if session is None:
        session = getattr(state, "_execution_session", None)
    if session is not None:
        session.poison()
    from rpu_backend.api import causal_lm
    with causal_lm._LIVE_LOCK:
        if causal_lm._LIVE_TERMINAL_REASON is None:
            owner = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
            if owner is None:
                owner = inner
                causal_lm._claim_live_instance(owner)
            causal_lm._poison_live_instance(
                owner, f"Qwen3.5-MoE text cleanup failed: {cleanup_error!r}", unsafe=True)
        from rpu_backend.api._execution import _mark_execution_process_unsafe
        _mark_execution_process_unsafe(
            f"Qwen3.5-MoE text cleanup failed: {cleanup_error!r}")
    if resource is not None:
        resource.retain_failure(cleanup_error, inner)


def _retire_qwen3_5_moe_text_state(inner, *, state=None, _graphs_retired=False):
    if state is None:
        state = getattr(inner, "_rpu_qwen3_5_moe", None)
    if state is None:
        return
    if getattr(state, "retirement_failed", None):
        raise RuntimeError(
            f"Qwen3.5-MoE text retirement already failed: {state.retirement_failed}")
    if _graphs_retired:
        from rpu_backend.api._execution import ExecutionSession
        session = getattr(state, "_execution_session", None)
        if not isinstance(session, ExecutionSession) or not session._shutting_down or session._active:
            raise RuntimeError(
                "Qwen3.5-MoE precleared text retirement requires its exclusive Session shutdown")
    try:
        import torch
        handle = getattr(state, "handle", None)
        finalizer = getattr(state, "handle_finalizer", None)
        resource = getattr(state, "retirement", None)
        if resource is not None and (
                (resource.owner() is not None and resource.owner() is not state)
                or resource.handle != handle or resource.failed is not None):
            raise RuntimeError("Qwen3.5-MoE text retirement handle identity changed")
        if handle is None and finalizer is not None and finalizer.alive:
            raise RuntimeError("Qwen3.5-MoE text live finalizer has no recorded handle")
        if handle is not None and resource is None and finalizer is not None and not finalizer.alive:
            raise RuntimeError(
                "Qwen3.5-MoE text handle has a dead finalizer without confirmed retirement")
        if not _graphs_retired:
            _clear_qwen35_moe_graphs((state,))
        if handle is not None:
            torch.ops.rpu.qwen3_5_moe_destroy(handle)
            if resource is not None:
                resource._native_destroyed(handle)
            elif finalizer is not None:
                finalizer.detach()
        state.handle = None
        if vars(inner).get("_rpu_qwen3_5_moe") is state:
            vars(inner).pop("_rpu_qwen3_5_moe")
    except BaseException as cleanup_error:
        _poison_qwen3_5_moe_text_retirement(inner, state, cleanup_error)
        raise


def _retire_failed_qwen3_5_moe_text_install(inner, *, state=None, session=None):
    if state is None:
        state = getattr(inner, "_rpu_qwen3_5_moe", None)
    if state is None:
        return
    if getattr(state, "retirement_failed", None):
        raise RuntimeError(
            f"Qwen3.5-MoE text retirement already failed: {state.retirement_failed}")
    if session is None:
        session = getattr(state, "_execution_session", None)
    try:
        if session is not None and getattr(state, "handle", None) is not None:
            session.shutdown(lambda: _retire_qwen3_5_moe_text_state(inner, state=state))
            if state.handle is not None:
                raise RuntimeError(
                    "closed execution session still owns a Qwen3.5-MoE text handle")
        else:
            _retire_qwen3_5_moe_text_state(inner, state=state)
        if session is not None:
            owned = vars(session._owner).get("_qwen35_retirement_children", ())
            if all(item[2].handle is None for item in owned):
                vars(session._owner).pop("_qwen35_retirement_children", None)
    except BaseException as cleanup_error:
        if not getattr(state, "retirement_failed", None):
            _poison_qwen3_5_moe_text_retirement(
                inner, state, cleanup_error, session)
        raise


def _validate_text_install_options(
    max_seq_len, *, execution_config=None
) -> tuple[int, int, int]:
    """Compatibility wrapper for the shared controlled text options."""
    return _resolve_text_install_options(
        max_seq_len, execution_config=execution_config
    )[:3]


@contextmanager
def _oneshot_scope(g):
    """Record, batch, execute once, then return to PASSTHROUGH.

    Qwen3.5 prefill shapes are effectively unbounded and rarely repeat. Keeping
    each large graph would consume one cache slot and one private queue without
    creating a replay hit; decode uses GraphCache below because its shape is stable.
    """
    g.begin()
    try:
        yield
    except BaseException:
        g.abort()
        raise
    else:
        g.end()


def _text_config(config):
    """Qwen3.5 nests text fields under ``config.text_config``; unit-test
    namespaces may be flat. Return the text config (fallback: config itself)."""
    return getattr(config, "text_config", config)


def _plan_prefill_execution(
    real_len,
    physical_limit,
    padding_budget,
    *,
    resolve_stage_domain,
    padding_rows="auto",
    exact_chunk_size=None,
    max_stage_chunk_size=None,
    plan_result_sink=None,
    queue_owner_id=0,
    graph_mode="BOUNDED_ONESHOT",
    execution_owner=None,
    execution_native=None,
    plan_signature=None,
    graph_cache=None,
):
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

# ── Development chunk admission. ───────────────────────────────────────────
#
# The no-merge KV sort in Qwen3_5MoeModel::subclass_chunk_size_valid accepts at
# most 4096 routed rows (chunk * top_k). The exact profile uses top_k=8.
# This kernel bound is not an SPM-fit claim: the framework planner checks the
# actual declarations, allocator budget, and kernel validity for every candidate.
# Keep a positive admission bound so public exact requests retain the same SPM
# checks. Auto has no additional 64/128-row ceiling. Native set_weights binds
# this same development admission before any forward; its historical name
# does not imply hardware certification (IMPLEMENTED_UNVERIFIED_HW).
_KV_SORT_MAX_ROWS = 4096
CERTIFIED_CHUNK_ENVELOPE: dict[tuple[int, int], tuple[int, int]] = {
    (40, 2048): (8192, _KV_SORT_MAX_ROWS // 8),
}


def _qwen3_5_moe_prefill_envelope(num_layers, hidden_size) -> tuple[int, int]:
    key = (int(num_layers), int(hidden_size))
    env = CERTIFIED_CHUNK_ENVELOPE.get(key)
    if env is None:
        raise UnsupportedModelError(
            f"Qwen3.5-MoE: no chunk admission for (num_hidden_layers, "
            f"hidden_size)={key}. This profile has never been measured on "
            f"hardware; running it on the auto chunk planner risks busting SPM "
            f"and WEDGING the board. Measure it on a resettable board first."
        )
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
    """Drop one MoE layer after every derived install tensor owns its storage."""
    mlp = layer.mlp
    _drop_linear_data(mlp.gate)
    _drop_tensor_data(mlp.experts, "gate_up_proj")
    _drop_tensor_data(mlp.experts, "gate_up_proj_scale")
    _drop_tensor_data(mlp.experts, "down_proj")
    _drop_tensor_data(mlp.experts, "down_proj_scale")
    _drop_linears_data(
        mlp.shared_expert, ("gate_proj", "up_proj", "down_proj"))
    _drop_linear_data(mlp.shared_expert_gate)
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
    for i, layer in enumerate(inner.layers):
        _drop_raw_hf_text_layer_weights(layer, bool(is_full[i]))
    _drop_tensor_data(inner.norm, "weight")


def install_qwen3_5_moe_text_for_rpu(
    text_model,
    config,
    max_seq_len=8192,
    *,
    execution_config=None,
    _resolved_options=None,
    _cpu_stage_weights: bool = True,
    _graph_runtime_policy=None,
):
    """Install the exact 35B-A3B mixed-E4M3 text owner transactionally."""
    if type(max_seq_len) is not int or not 0 < max_seq_len <= 8192:
        raise ValueError(
            "Qwen3.5-35B-A3B max_seq_len must be an integer in [1, 8192]"
        )
    import rpu_backend as _rb
    import torch

    from .cores import core_topology
    from .quant_scope import validate_exact_profile

    try:
        validate_exact_profile(config, require_quant=True)
    except (TypeError, ValueError) as exc:
        raise UnsupportedModelError(str(exc)) from exc
    execution_config = {} if execution_config is None else execution_config
    topology = core_topology(execution_config)
    for name in ("qwen3_5_moe_set_execution_cores", "qwen3_5_moe_get_execution_topology",
                 "qwen3_5_moe_get_execution_topology_v2"):
        if not callable(getattr(torch.ops.rpu, name, None)):
            raise RuntimeError(f"Qwen3.5-MoE binary lacks core topology op {name}; rebuild before installing")
    if _cpu_stage_weights is not True:
        raise UnsupportedModelError(
            "Qwen3.5-35B-A3B requires bounded CPU-staged weight installation"
        )
    if hasattr(text_model, "_rpu_qwen3_5_moe"):
        raise RPUBackendError(
            "Qwen3.5-MoE text runtime is already installed; repeated installation "
            "would swizzle the same weights twice."
        )
    if getattr(text_model, "_rpu_qwen3_5_moe_text_install_started", False):
        raise RPUBackendError(
            "Qwen3.5-MoE text installation previously started an irreversible "
            "weight transform; reload the model before retrying."
        )
    if _graph_runtime_policy is not None:
        if not isinstance(_graph_runtime_policy, _rb.graph.GraphRuntimePolicy):
            raise TypeError("Qwen3.5-MoE text graph policy must be a GraphRuntimePolicy")
        if _graph_runtime_policy.execution_core_count != topology.num_cores:
            raise ValueError("Qwen3.5-MoE graph policy conflicts with execution topology")
    elif topology.num_cores != 8:
        _graph_runtime_policy = _rb.graph.GraphRuntimePolicy.from_environment(
            execution_core_count=topology.num_cores)

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
    linear_acc32 = prefill_config.get("linear_acc32", False)
    if type(linear_acc32) is not bool:
        raise TypeError("Qwen3.5-MoE prefill.linear_acc32 must be bool")
    requested_chunk = prefill_config.get("chunk_size", "auto")
    exact_chunk_size = 0 if requested_chunk == "auto" else int(requested_chunk)
    if exact_chunk_size and exact_chunk_size % 64:
        raise ValueError(
            "Qwen3.5-MoE prefill chunk_size must be a multiple of 64, got "
            f"{exact_chunk_size}"
        )
    envelope = _qwen3_5_moe_prefill_envelope(
        _text_config(config).num_hidden_layers, _text_config(config).hidden_size
    )
    if exact_chunk_size and exact_chunk_size > envelope[1]:
        raise UnsupportedModelError(
            f"Qwen3.5-MoE prefill chunk_size={exact_chunk_size} exceeds the "
            f"exact profile ceiling {envelope[1]}"
        )
    if exact_chunk_size:
        chunk_size_cap = 0
    padding_budget = int(prefill_config.get("padding_budget", padding_budget))
    padding_rows = prefill_config.get("padding_rows", "auto")
    pending_state = _Qwen35MoeNativeState(
        handle=None,
        handle_finalizer=None,
        graph_cache=None,
        prefill_graph=None,
        execution_topology=topology,
        linear_acc32=linear_acc32,
    )
    try:
        graph_options = ({"runtime_policy": _graph_runtime_policy}
                         if _graph_runtime_policy is not None else {})
        cache_options = {"max_entries": 1} if _graph_runtime_policy is not None else {}
        graph_cache = pending_state.graph_cache = _rb.graph.GraphCache(
            **cache_options, **graph_options
        )
        prefill_graph = pending_state.prefill_graph = _rb.graph.Graph(**graph_options)
        return _install_qwen3_5_moe_text_for_rpu_impl(
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
            cpu_stage_weights=True,
            pending_state=pending_state,
        )
    except BaseException as error:
        try:
            _retire_failed_qwen3_5_moe_text_install(
                text_model, state=pending_state
            )
        except BaseException as cleanup_error:
            error.add_note(
                f"Qwen3.5-MoE text install cleanup failed: {cleanup_error!r}"
            )
        raise


def _install_qwen3_5_moe_text_for_rpu_impl(
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
    """Implementation for :func:`install_qwen3_5_moe_text_for_rpu`.

    The public wrapper owns graph cleanup until this function publishes the
    complete ``_rpu_qwen3_5_moe`` namespace.
    """
    import torch
    from rpu_backend.runtime.weights import (
        convert_linear_weights_inplace, transform_linear_weight,
    )
    from .weights import build_route_tables, prepare_moe_layer_weights

    inner = text_model
    # RPU kernels are fp16-only. Like every rpu_backend adapter, the swizzle
    # (convert_linear_weights_inplace / transform_linear_weight) PRESERVES weight
    # dtype (so int8/w8a16 quant paths stay intact) and relies on the CALLER having
    # loaded the model as fp16. Fail fast with a clear message here instead of a
    # cryptic "expected Half but found BFloat16" deep in qwen3_5_moe_forward. Check an
    # ACTUAL weight, not config.dtype (the latter stays bfloat16 even after .half()):
    # some Qwen3.5 ckpts (e.g. 4B) carry text_config.dtype=bfloat16 which overrides
    # from_pretrained(dtype=torch.float16), so the caller must .half() the model.
    tc = _text_config(config)
    adaptive_mode = getattr(tc, "adaptive_mode", None)
    if adaptive_mode is not None:
        raise UnsupportedModelError(
            f"Qwen3.5-MoE text adapter does not support adaptive_mode={adaptive_mode!r}."
        )
    _wdt = inner.norm.weight.dtype
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
            "Qwen3.5 CPU-staged installation requires CPU source weights"
        )
    is_full = layer_types_to_is_full(config)
    head_dim = int(tc.head_dim)
    nq = int(tc.num_attention_heads)
    nkv = int(tc.num_key_value_heads)
    # Replicate logical KV heads to one complete head per active attention
    # core. The cache retains its physical eight-stripe DDR representation.
    topology = pending_state.execution_topology
    NUM_CORES = topology.attn_tp
    kv_rep = NUM_CORES // nkv if (nkv < NUM_CORES and NUM_CORES % nkv == 0) else 1
    nkv_eff = nkv * kv_rep

    # The outer adapter owns the untied LM head. This pass prepares ordinary
    # attention output projections with attention TP; q/k/v, GDN and expert
    # projections remain raw until their specialized cold preparation below.
    # The marker intentionally survives both success and failure. A successful
    # install is guarded by the published namespace; if an outer orchestration
    # step later fails and removes that namespace, this marker still prevents a
    # direct second install from double-swizzling the same model.
    inner._rpu_qwen3_5_moe_text_install_started = True
    convert_linear_weights_inplace(
        inner.layers,
        attn_num_cores=topology.attn_tp,
        mlp_num_cores=topology.mlp_tp,
        lm_head_num_cores=topology.lm_head_tp,
        execution_core_count=topology.num_cores,
        skip_names={"q_proj", "k_proj", "v_proj", "lm_head",
                    "in_proj_qkv", "in_proj_z", "in_proj_b",
                    "in_proj_a", "out_proj", "gate_proj", "up_proj",
                    "down_proj", "gate", "shared_expert_gate"})
    # The public 35B profile always releases transformed CPU sources as each
    # layer is uploaded. Keeping a debug copy would violate the bounded loader.
    free_raw_hf_weights = True

    # GDN projections and convolution histories use active-core head slices;
    # scalar DMA storage retains eight FP16 slots per core.
    _gnvh = int(tc.linear_num_value_heads)
    _gnkh = int(tc.linear_num_key_heads)
    _gdk = int(tc.linear_key_head_dim)
    _gdv = int(tc.linear_value_head_dim)
    _gHc = _gnvh // NUM_CORES

    _gksz = _gnkh * _gdk                   # q proj size = k proj size (2048, no rep)

    empty = torch.empty(0, dtype=torch.float16)

    def _projection_scale(projection, fq_name):
        """Return a CPU FP16 per-output scale for raw E4M3, else empty.

        The ordinary ParallelLinear API stays unchanged: raw uint8 [N,K]
        selects E4M3 in the existing wrapper and this tensor uses its existing
        scale argument.  FP16 projections therefore carry an empty scale.
        """
        weight = projection.weight
        scale = getattr(projection, "weight_scale", None)
        if weight.dtype != torch.uint8:
            raise UnsupportedModelError(
                f"{fq_name}: exact mixer projection must be raw E4M3 uint8, "
                f"got {weight.dtype}"
            )
        if (
            getattr(projection, "rpu_fp8_format", None) != "fp8_e4m3"
            or getattr(projection, "rpu_weight_mode", None) != 5
            or getattr(projection, "rpu_fp8_scale_axis", None) != -1
        ):
            raise UnsupportedModelError(
                f"{fq_name}: raw uint8 weight is missing the typed E4M3 mode-5 markers"
            )
        expected = (int(weight.shape[0]),)
        if scale is None or scale.dtype != torch.float16 or tuple(scale.shape) != expected:
            actual = None if scale is None else (scale.dtype, tuple(scale.shape))
            raise UnsupportedModelError(
                f"{fq_name}: raw FP8 weight requires FP16 weight_scale {expected}, "
                f"got {actual}"
            )
        return scale.detach().to("cpu").contiguous()

    def _sw_col(w):   # Active-core col partition; z remains head-ordered
        return transform_linear_weight(
            w.detach().to("cpu").contiguous(), 1, NUM_CORES).to("rpu")

    # Scalar scatter requires 16 bytes/core: cold-pad to eight FP16 slots.
    # TP4 uses all eight slots; TP8 uses the first four.
    def _pad_scalar(w):  # [H] → [nc*8] core-major (first Hc/core)
        out = torch.zeros(NUM_CORES * 8, dtype=w.dtype)
        wc = w.detach().to("cpu")
        for c in range(NUM_CORES):
            out[c * 8:c * 8 + _gHc] = wc[c * _gHc:(c + 1) * _gHc]
        return out.contiguous().to("rpu")

    def _sw_row(w):   # Active-core row partition (out_proj)
        return transform_linear_weight(
            w.detach().to("cpu").contiguous(), 0, NUM_CORES).to("rpu")

    # ===== GDN per-path weights — SEPARATE q/k/v proj + per-path conv + N_bg b/a =====
    # Shared by decode (build_gdn recurrent) and prefill (chunk); matches rhino's
    # decomposed prep. in_proj_qkv = [all-q(_gksz) | all-k(_gksz) | all-v(nvh·dv)].
    _gvsz = _gnvh * _gdv                       # v proj width (4096)
    def _qkv_path(w, lo, hi):  # col-parallel proj of one path (head-ordered, no reorder)
        return transform_linear_weight(
            w.detach().to("cpu")[lo:hi].contiguous(), 1, NUM_CORES).to("rpu")
    def _conv_path(la, lo, hi):  # per-path conv slice -> [nc, Kc, (hi-lo)/nc]
        cw = la.conv1d.weight.detach().to("cpu").squeeze(1)[lo:hi]   # [hi-lo, Kc]
        Kc = cw.shape[1]
        wd = (hi - lo) // NUM_CORES
        return cw.reshape(NUM_CORES, wd, Kc).permute(0, 2, 1).contiguous().to("rpu")
    _vg_c = _gnvh // NUM_CORES                 # v heads/core (4)
    _n_bg = ((_vg_c + 15) // 16) * 16          # padded width/core (16)
    def _ba_nbg(w):  # [nvh, hidden] -> col-parallel [n_bg·nc, hidden] (first vg_c/core valid)
        hid = w.shape[1]
        out = torch.zeros(_n_bg * NUM_CORES, hid, dtype=w.dtype)
        wc = w.detach().to("cpu")
        for c in range(NUM_CORES):
            out[c * _n_bg:c * _n_bg + _vg_c] = wc[c * _vg_c:(c + 1) * _vg_c]
        return transform_linear_weight(out.contiguous(), 1, NUM_CORES).to("rpu")

    q, k, v, o, qn, kn, ag, inrm, postrm = ([] for _ in range(9))
    q_s, k_s, v_s, o_s, ag_s = ([] for _ in range(5))
    router, routed_g, routed_u, routed_d = ([] for _ in range(4))
    routed_gs, routed_us, routed_ds = ([] for _ in range(3))
    shared_g, shared_u, shared_d, shared_scalar = ([] for _ in range(4))
    # GDN mixer weight lists (parallel to num_layers; full slots = empty).
    g_z, g_out, g_alog, g_dt, g_nrm = ([] for _ in range(5))
    g_z_s, g_out_s = ([] for _ in range(2))
    # prefill (chunk) per-path weights: separate q/k/v proj + conv, N_bg b/a
    g_q, g_k, g_v, g_cq, g_ck, g_cv, g_b_bg, g_a_bg = ([] for _ in range(8))
    g_q_s, g_k_s, g_v_s = ([] for _ in range(3))
    unit_norm = None
    routed_weight_mode = None
    for i, layer in enumerate(inner.layers):
        packed_moe = prepare_moe_layer_weights(
            layer.mlp, num_cores=topology.num_cores, device="rpu"
        )
        if routed_weight_mode is None:
            routed_weight_mode = packed_moe.routed_weight_mode
        elif routed_weight_mode != packed_moe.routed_weight_mode:
            raise UnsupportedModelError(
                "Qwen3.5-MoE requires one routed weight mode across all layers; "
                f"layer {i} uses {packed_moe.routed_weight_mode}, expected "
                f"{routed_weight_mode}"
            )
        router.append(packed_moe.router)
        routed_g.append(packed_moe.routed_gate)
        routed_u.append(packed_moe.routed_up)
        routed_d.append(packed_moe.routed_down)
        routed_gs.append(packed_moe.routed_gate_scale)
        routed_us.append(packed_moe.routed_up_scale)
        routed_ds.append(packed_moe.routed_down_scale)
        shared_g.append(packed_moe.shared_gate)
        shared_u.append(packed_moe.shared_up)
        shared_d.append(packed_moe.shared_down)
        shared_scalar.append(packed_moe.shared_scalar_gate)
        if not is_full[i]:
            # GDN layer: standard post_norm + MLP half collected like every
            # layer; the token mixer weights go into the GDN lists below.
            for lst in (q, k, v, o, qn, kn, ag,
                        q_s, k_s, v_s, o_s, ag_s):  # inrm filled below
                lst.append(empty)
            postrm.append(layer.post_attention_layernorm.weight.data + 1.0)
            la = layer.linear_attn
            # input_layernorm goes into the unified input_norm channel (inrm); the
            # C++ layer-level rmsnorm reads persistent "norm_w" for ALL layers.
            inrm.append(layer.input_layernorm.weight.data + 1.0)   # Qwen3_5MoeRMSNorm 1+w
            g_z.append(_sw_col(la.in_proj_z.weight))
            g_out.append(_sw_row(la.out_proj.weight))       # row-partition
            g_z_s.append(_projection_scale(
                la.in_proj_z, f"layer {i}.linear_attn.in_proj_z"))
            g_out_s.append(_projection_scale(
                la.out_proj, f"layer {i}.linear_attn.out_proj"))
            g_alog.append(_pad_scalar(la.A_log.data))
            g_dt.append(_pad_scalar(la.dt_bias.data))
            g_nrm.append(la.norm.weight.data)   # gated norm: plain weight
            # prefill per-path (separate q/k/v proj + conv, N_bg b/a)
            _qkvw = la.in_proj_qkv.weight
            _qkvs = _projection_scale(
                la.in_proj_qkv, f"layer {i}.linear_attn.in_proj_qkv")
            g_q.append(_qkv_path(_qkvw, 0, _gksz))
            g_k.append(_qkv_path(_qkvw, _gksz, 2 * _gksz))
            g_v.append(_qkv_path(_qkvw, 2 * _gksz, 2 * _gksz + _gvsz))
            if _qkvs.numel():
                g_q_s.append(_qkvs[0:_gksz].contiguous())
                g_k_s.append(_qkvs[_gksz:2 * _gksz].contiguous())
                g_v_s.append(_qkvs[2 * _gksz:2 * _gksz + _gvsz].contiguous())
            else:
                g_q_s.append(empty); g_k_s.append(empty); g_v_s.append(empty)
            g_cq.append(_conv_path(la, 0, _gksz))
            g_ck.append(_conv_path(la, _gksz, 2 * _gksz))
            g_cv.append(_conv_path(la, 2 * _gksz, 2 * _gksz + _gvsz))
            g_b_bg.append(_ba_nbg(la.in_proj_b.weight))
            g_a_bg.append(_ba_nbg(la.in_proj_a.weight))
            del _qkvw, _qkvs
            if free_raw_hf_weights:
                _drop_raw_hf_text_layer_weights(layer, False)
            continue
        for lst in (g_z, g_out, g_z_s, g_out_s, g_alog, g_dt, g_nrm,
                    g_q, g_k, g_v, g_q_s, g_k_s, g_v_s,
                    g_cq, g_ck, g_cv, g_b_bg, g_a_bg):
            lst.append(empty)
        a = layer.self_attn
        # q_proj.weight is [num_heads*head_dim*2, hidden] = per-head interleaved
        # [h0_query h0_gate h1_query h1_gate ...] (skipped by the swizzle pass →
        # still raw). Split per-head, then col-swizzle each over all NUM_CORES cores
        # (each core owns nq/NUM_CORES query heads, matching the installed GEMM topology).
        query_raw, gate_raw = split_q_gate(a.q_proj.weight.data, nq, head_dim)
        q.append(transform_linear_weight(query_raw, 1, NUM_CORES))
        ag.append(transform_linear_weight(gate_raw, 1, NUM_CORES))  # gate; deferred
        qg_scale = _projection_scale(a.q_proj, f"layer {i}.self_attn.q_proj")
        if qg_scale.numel():
            qg_scale = qg_scale.view(nq, 2 * head_dim)
            q_s.append(qg_scale[:, :head_dim].reshape(-1).contiguous())
            ag_s.append(qg_scale[:, head_dim:].reshape(-1).contiguous())
        else:
            q_s.append(empty); ag_s.append(empty)
        # k/v_proj skipped by the swizzle pass → raw; replicate each KV head kv_rep
        # times (nkv_eff==NUM_CORES) then col-swizzle over all active cores, so every core
        # owns one whole KV head (per-core norm/rope path, no cross-core GQA kernel).
        k.append(transform_linear_weight(
            _replicate_kv_heads(a.k_proj.weight.data, nkv, head_dim, kv_rep), 1, NUM_CORES))
        v.append(transform_linear_weight(
            _replicate_kv_heads(a.v_proj.weight.data, nkv, head_dim, kv_rep), 1, NUM_CORES))
        k_scale = _projection_scale(a.k_proj, f"layer {i}.self_attn.k_proj")
        v_scale = _projection_scale(a.v_proj, f"layer {i}.self_attn.v_proj")
        if k_scale.numel():
            k_s.append(_replicate_kv_heads(
                k_scale.view(-1, 1), nkv, head_dim, kv_rep).reshape(-1).contiguous())
            v_s.append(_replicate_kv_heads(
                v_scale.view(-1, 1), nkv, head_dim, kv_rep).reshape(-1).contiguous())
        else:
            k_s.append(empty); v_s.append(empty)
        o.append(a.o_proj.weight.data)
        o_s.append(_projection_scale(a.o_proj, f"layer {i}.self_attn.o_proj"))
        qn.append(a.q_norm.weight.data + 1.0); kn.append(a.k_norm.weight.data + 1.0)
        inrm.append(layer.input_layernorm.weight.data + 1.0)
        postrm.append(layer.post_attention_layernorm.weight.data + 1.0)
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
            q, k, v, o, qn, kn, ag, q_s, k_s, v_s, o_s, ag_s,
            inrm, postrm,
            g_z, g_out, g_z_s, g_out_s, g_alog, g_dt, g_nrm,
            g_q, g_k, g_v, g_q_s, g_k_s, g_v_s,
            g_cq, g_ck, g_cv, g_b_bg, g_a_bg,
        ):
            for index, weight in enumerate(weights):
                if weight.numel() and weight.device.type != "rpu":
                    # Preserve raw uint8 E4M3 weights; scales and auxiliary
                    # tensors are already FP16 by construction.
                    weights[index] = weight.detach().to(
                        device="cpu"
                    ).contiguous().to("rpu")

    gdn_nvh = int(tc.linear_num_value_heads)
    gdn_nkh = int(tc.linear_num_key_heads)
    gdn_dk = int(tc.linear_key_head_dim)
    gdn_dv = int(tc.linear_value_head_dim)
    # No GQA weight replication: conv_dim uses TRUE nkh (q/k stay nkh heads); the
    # nkh→nvh expansion is a runtime tile in build_gdn (12288 → 8192 for 4B).
    gdn_conv_dim = 2 * (gdn_nkh * gdn_dk) + gdn_nvh * gdn_dv   # 8192 for 4B
    gdn_conv_kernel = int(tc.linear_conv_kernel_dim)
    final_norm = inner.norm.weight.data + 1.0
    if routed_weight_mode != 5:
        raise UnsupportedModelError(
            f"Qwen3.5-35B-A3B routed experts require E4M3 mode 5, got {routed_weight_mode}"
        )
    if free_raw_hf_weights:
        _drop_tensor_data(inner.norm, "weight")

    handle = torch.ops.rpu.qwen3_5_moe_create()
    pending_state.handle = handle
    handle_finalizer = None
    try:
        from rpu_backend.runtime._native_retirement import _InstalledNativeResource

        pending_state.retirement = _InstalledNativeResource(
            pending_state,
            handle,
            torch.ops.rpu.qwen3_5_moe_destroy,
            graphs=(graph_cache,),
            raw_graphs=(prefill_graph,),
            keepalive=(
                q, k, v, o, qn, kn, ag, inrm, postrm,
                router, routed_g, routed_u, routed_d,
                shared_g, shared_u, shared_d, shared_scalar,
                cos, sin, final_norm,
                g_z, g_out, g_alog, g_dt, g_nrm,
                g_q, g_k, g_v, g_cq, g_ck, g_cv, g_b_bg, g_a_bg,
            ),
            label="Qwen3.5-MoE text",
            handle_name="handle",
        )
        handle_finalizer = pending_state.retirement.finalizer
        if final_norm.device.type != "rpu":
            final_norm = final_norm.detach().to(
                device="cpu", dtype=torch.float16
            ).contiguous().to("rpu")
            pending_state.retirement.keepalive += (final_norm,)
        max_admitted_kv, max_chunk = _qwen3_5_moe_prefill_envelope(
            tc.num_hidden_layers, tc.hidden_size
        )
        route_table_chunk = min(
            max_admitted_kv, _KV_SORT_MAX_ROWS // int(tc.num_experts_per_tok)
        )
        if route_table_chunk != max_chunk or route_table_chunk != 512:
            raise RuntimeError("Qwen3.5-MoE route-table and planner ceilings diverged")
        expert_ids, token_ids = build_route_tables(
            route_table_chunk,
            int(tc.num_experts),
            int(tc.num_experts_per_tok),
            device="rpu",
        )
        packed_scales = [
            scale
            for group in (
                q_s, k_s, v_s, o_s, ag_s,
                routed_gs, routed_us, routed_ds,
                g_z_s, g_out_s, g_q_s, g_k_s, g_v_s,
            )
            for scale in group
        ]
        pending_state.retirement.keepalive += (expert_ids, token_ids, packed_scales)
        torch.ops.rpu.qwen3_5_moe_set_execution_cores(handle, topology.num_cores)
        torch.ops.rpu.qwen3_5_moe_set_weights(
            handle, q, k, v, o, qn, kn, ag,
            packed_scales, inrm, postrm,
            router, routed_g, routed_u, routed_d,
            shared_g, shared_u, shared_d, shared_scalar,
            cos, sin, final_norm, expert_ids, token_ids, is_full,
            nq, nkv_eff, head_dim, int(tc.hidden_size),
            int(tc.num_experts), int(tc.num_experts_per_tok),
            int(tc.moe_intermediate_size),
            int(tc.shared_expert_intermediate_size), topology.mlp_tp,
            int(routed_weight_mode), float(tc.rms_norm_eps), True, mrope_section,
            g_z, g_out, g_alog, g_dt, g_nrm,
            gdn_nvh, gdn_dk, gdn_dv, gdn_conv_dim, gdn_conv_kernel,
            g_q, g_k, g_v,
            g_cq, g_ck, g_cv, g_b_bg, g_a_bg,
        )
        from .cores import validate_native_topology

        validate_native_topology(handle, topology)
        torch.ops.rpu.qwen3_5_moe_set_chunk_size_cap(handle, chunk_size_cap)
        torch.ops.rpu.qwen3_5_moe_set_prefill_chunk_size(handle, exact_chunk_size)
        torch.ops.rpu.qwen3_5_moe_set_linear_acc32(handle, pending_state.linear_acc32)
        torch.ops.rpu.qwen3_5_moe_set_retained_prefill_graph(handle, False)
        vars(pending_state).update(
            handle=handle,
            handle_finalizer=handle_finalizer,
            graph_cache=graph_cache,
            prefill_graph=prefill_graph,
            max_seq_len=max_seq_len,
            num_layers=len(is_full),
            hidden_size=int(tc.hidden_size),
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
            padding_budget=padding_budget,
            padding_rows=padding_rows,
            execution_config=execution_config,
            control_snapshot=control_snapshot,
            exact_chunk_size=exact_chunk_size or None,
            chunk_envelope=(max_admitted_kv, max_chunk),
            adaptive_mode=None,
            routed_weight_mode=int(routed_weight_mode),
            route_tables=(expert_ids, token_ids),
            has_dispatched=False,
            prefill_plan_key=None,
            prefill_plan=None,
            decode_stage_descriptor=None,
            decode_plan=None,
            last_a6_plan=None,
            execution_generation=0,
        )
        vars(inner)["_rpu_qwen3_5_moe"] = pending_state
    finally:
        pending_state.handle_finalizer = handle_finalizer
    return handle


def _preflight_qwen3_5_moe_text(
    text_model, cache, real_len, *, multimodal_prefill=False
):
    """Resolve one complete native stage domain before mutating RPU state."""
    import torch

    from rpu_backend.adapters.qwen3_5.cores import topology_tuple

    st = text_model._rpu_qwen3_5_moe
    if not _qwen35_moe_retirement_ready(st):
        raise RuntimeError("Qwen3.5-MoE text resource is not live and parent-owned")
    expected_topology = topology_tuple(st.execution_topology)
    if tuple(getattr(cache, "execution_topology", ())) != expected_topology:
        raise ValueError(
            "Qwen3.5-MoE cache core layout does not match the installed model; "
            "use Qwen3_5MoeCache.from_model(model, ...)"
        )
    real_len = int(real_len)
    envelope = st.chunk_envelope
    if cache.position + real_len > min(st.max_seq_len, envelope[0]):
        raise ValueError(
            "Qwen3.5-MoE request exceeds the installed KV/rotary capacity: "
            f"position={cache.position}, input={real_len}, "
            f"limit={min(st.max_seq_len, envelope[0])}"
        )
    if real_len > envelope[0]:
        raise UnsupportedModelError(
            f"Qwen3.5-MoE {'multimodal' if multimodal_prefill else 'text'} "
            f"prefill length {real_len} exceeds the declared limit {envelope[0]}"
        )
    if cache.position + real_len > cache.max_seq_len:
        raise ValueError(
            f"Qwen3.5-MoE cache capacity exceeded: position={cache.position}, "
            f"input={real_len}, max_seq_len={cache.max_seq_len}"
        )
    if real_len > 1 and cache.position != 0:
        raise NotImplementedError(
            "Qwen3.5-MoE multi-token prefill is supported only at cache position 0"
        )
    if real_len <= 1:
        return real_len, 0, None

    plan_box = {}
    pad_len, planned_chunk_size = _plan_prefill_execution(
        real_len,
        cache.allocated_max_seq_len,
        st.padding_budget,
        execution_owner=text_model,
        execution_native=("qwen3_5_moe", int(st.handle)),
        plan_signature=(real_len, bool(multimodal_prefill)),
        graph_cache=None,
        resolve_stage_domain=lambda execution_len: (
            torch.ops.rpu.qwen3_5_moe_resolve_prefill_stage_domain(
                st.handle, execution_len, real_len
            )
        ),
        padding_rows=st.padding_rows,
        exact_chunk_size=st.exact_chunk_size,
        max_stage_chunk_size=envelope[1],
        plan_result_sink=lambda result: plan_box.__setitem__("result", result),
        queue_owner_id=int(st.handle),
        graph_mode="BOUNDED_ONESHOT",
    )
    result = plan_box["result"]
    st.last_a6_plan = result.as_dict(include_candidates=False)
    st.prefill_plan_key = None
    st.prefill_plan = (pad_len, planned_chunk_size, result)
    return st.prefill_plan


def _plan_qwen3_5_moe_decode(text_model):
    """Plan the real fixed decode descriptor through the shared planner."""
    import torch
    from rpu_backend.runtime.decoder import plan_native_component_execution

    state = text_model._rpu_qwen3_5_moe
    native = ("qwen3_5_moe", int(state.handle))
    generation = int(getattr(state, "execution_generation", 0))

    def domain(_length):
        descriptor = tuple(
            torch.ops.rpu.qwen3_5_moe_resolve_decode_stage_descriptor(state.handle)
        )
        return [1, 1, len(descriptor), *descriptor]

    _, result = plan_native_component_execution(
        1,
        component_id="language_model",
        stage="decode",
        generation=generation,
        execution={},
        resolve_stage_domain=domain,
        graph_mode=GRAPH_RETAINED_CACHE,
        queue_owner_id=int(state.handle),
        execution_owner=text_model,
        execution_native=native,
        plan_signature=(),
        graph_cache=state.graph_cache,
    )
    state.decode_stage_descriptor = result.selected.stage_tuple.physical_descriptor
    state.decode_plan = (None, result)
    return result


def run_qwen3_5_moe_text(
    text_model,
    hidden,
    cache,
    *,
    attention_mask=None,
    position_ids=None,
    multimodal_prefill=False,
    preflight_plan=None,
):
    """Drive exact-profile prefill/decode using planner-selected descriptors."""
    import torch
    import rpu_backend as _rb

    st = text_model._rpu_qwen3_5_moe
    handle = st.handle
    if hidden.dtype != torch.float16:
        hidden = hidden.to(torch.float16)
    if not hidden.is_contiguous():
        hidden = hidden.contiguous()
    if attention_mask is not None and attention_mask.device.type != "rpu":
        attention_mask = attention_mask.to("rpu")
    seq_len = int(hidden.shape[1])
    real_len = seq_len
    if preflight_plan is None:
        preflight_plan = _preflight_qwen3_5_moe_text(
            text_model,
            cache,
            real_len,
            multimodal_prefill=multimodal_prefill,
        )

    planned_chunk_size = 0
    planned_stage_descriptor = ()
    prefill_plan_result = None
    if seq_len > 1:
        pad_len, planned_chunk_size, prefill_plan_result = preflight_plan
        if prefill_plan_result.selected is None:
            raise RuntimeError("Qwen3.5-MoE planner returned no prefill winner")
        planned_stage_descriptor = (
            prefill_plan_result.selected.stage_tuple.physical_descriptor
        )
        if not planned_stage_descriptor:
            raise RuntimeError("Qwen3.5-MoE A6 winner has no native stage descriptor")
        if pad_len > seq_len:
            hidden = torch.cat(
                [hidden, hidden.new_zeros(
                    hidden.shape[0], pad_len - seq_len, hidden.shape[2]
                )],
                dim=1,
            )
            seq_len = int(pad_len)
        torch.ops.rpu.qwen3_5_moe_set_valid_prefill_len(handle, real_len)

    if seq_len > 1 and st.mrope_section:
        pid = position_ids
        if pid is None:
            rng = torch.arange(0, cache.position + seq_len, dtype=torch.long)
            pid = rng.view(1, -1).expand(3, -1)
        else:
            pid = _normalize_prefill_position_ids(
                pid, real_len=real_len, padded_len=seq_len
            )
        pcos, psin = _get_prefill_mrope_tables(
            st, pid, st.rotary_dim, st.rope_theta, st.mrope_section
        )
        torch.ops.rpu.qwen3_5_moe_set_prefill_rope(handle, pcos, psin)

    is_prefill = seq_len > 1
    if is_prefill:
        ctx = _oneshot_scope(st.prefill_graph)
    else:
        decode_plan_result = _plan_qwen3_5_moe_decode(text_model)
        planned_stage_descriptor = (
            decode_plan_result.selected.stage_tuple.physical_descriptor
        )
        ctx = st.graph_cache.capture(_rb.graph.GraphSignature(
            op_id="rpu_qwen3_5_moe",
            shapes=[seq_len, st.hidden_size],
            dyn_dims=[
                st.num_layers,
                int(getattr(st, "execution_generation", 0)),
                *decode_plan_result.graph_key_words(),
            ],
            dtypes=[torch.float16],
        ))

    with ctx:
        out = torch.ops.rpu.qwen3_5_moe_forward(
            handle,
            hidden,
            cache.k_caches,
            cache.v_caches,
            cache.gdn_states,
            cache.conv_states,
            attention_mask,
            cache.position,
            True,
            0,  # The complete descriptor owns the native chunk decision.
            planned_stage_descriptor,
        )
    st.has_dispatched = True
    last_plan = {
        "stage": "prefill" if is_prefill else "decode",
        "component": getattr(
            text_model, "_rpu_execution_component_id", "language_model"
        ),
        "logical_len": real_len,
        "execution_len": seq_len,
        "chunk_size": int(planned_chunk_size),
        "padding_rows": seq_len - real_len,
        "position": int(cache.position),
        "graph_mode": (
            prefill_plan_result.graph_mode if is_prefill else GRAPH_RETAINED_CACHE
        ),
        "physical_descriptor": tuple(planned_stage_descriptor),
    }
    receipt = (
        st.last_a6_plan
        if is_prefill
        else decode_plan_result.as_dict(include_candidates=False)
    )
    if receipt is not None:
        last_plan.update({
            "a6_plan_digest": receipt["plan_digest"],
            "a6_domain_digest": receipt["domain_digest"],
            "a6_optimality": receipt["optimality"],
            "a6_search_complete": receipt["search_complete"],
        })
    vars(text_model)["_rpu_last_execution_plan"] = last_plan
    cache.update_position(real_len)
    if real_len != seq_len:
        out = out[:, :real_len]
    return out
