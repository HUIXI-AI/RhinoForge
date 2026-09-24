"""Actual Pi adapter, public close, Session and GC; only native calls are doubled."""
import gc
import weakref

import pytest

from rpu_backend.adapters import pi05
from rpu_backend.api import _execution, causal_lm
from rpu_backend.api.policy import Pi05Policy
from rpu_backend.runtime import _native_retirement
from rpu_backend.tests.acceptance.test_pi05_adapter_install_lifecycle import _complete_policy
from test_graph_cache_freeze import _cache, _sig


@pytest.fixture(autouse=True)
def isolated_process(monkeypatch):
    gc.collect()
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])


def _owner(monkeypatch, *, fault=None, error_type=RuntimeError, fused=True, shared=False):
    policy, model, vlm, expert, vision = _complete_policy(fused=fused)
    events = []
    for child, handle_name, graph_name, op in (
        (model, "_rpu_fused_denoise_handle", "_rpu_fused_denoise_graph_cache", "pi05_denoise_step_destroy"),
        (vision, "_rpu_vision_handle", "_rpu_siglip_graph_cache", "siglip_destroy"),
        (expert, "_rpu_action_handle", "_rpu_adarms_graph_cache", "adarms_destroy"),
        (vlm, "_rpu_vlm_decoder_handle", "_rpu_gemma_graph_cache", "gemma_destroy"),
    ):
        handle = getattr(child, handle_name)
        if handle is None:
            continue
        graph = _cache()
        graph.begin_warmup()
        with graph.capture(_sig(graph_name)):
            pass
        graph.freeze()
        clear = graph._impl.clear

        def raw_clear(graph_name=graph_name, clear=clear):
            events.append(("clear", graph_name))
            if fault == graph_name:
                raise error_type("injected graph failure")
            clear()

        graph._impl.clear = raw_clear
        graph._impl.cache_invariant_ok = lambda name=graph_name: fault != name + "-invariant"
        setattr(child, graph_name, graph)

        def raw_destroy(handle):
            events.append(("destroy", handle))
            if fault == handle:
                raise error_type("injected native failure")

        monkeypatch.setattr(pi05.torch.ops.rpu, op, raw_destroy, raising=False)

        def legacy_finalizer(handle, destroy=raw_destroy):
            events.append(("legacy_finalizer", handle))
            try:
                destroy(handle)
            except Exception:
                pass

        if shared:
            resource = _native_retirement._InstalledNativeResource(
                child, handle, raw_destroy, graphs=(graph,),
                keepalive=(getattr(child, "_rpu_cache", vlm._rpu_cache), pi05.torch.ones(1)),
                label=op, handle_name=handle_name)
            setattr(child, handle_name.replace("_handle", "_retirement_state"), resource)
            finalizer = resource.finalizer
        else:
            finalizer = weakref.finalize(child, legacy_finalizer, handle)
        setattr(child, handle_name + "_finalizer", finalizer)

    adapter = pi05.Pi05Adapter(policy)
    causal_lm._claim_live_instance(policy)
    # This is the production final publication step after the four installers.
    adapter._take_retirement_ownership()
    public = object.__new__(Pi05Policy)
    public._adapter = adapter
    return public, events


@pytest.mark.parametrize("fused", [True, False])
def test_success_and_mirror_close_retire_graphs_before_native_once(monkeypatch, fused):
    public, events = _owner(monkeypatch, fused=fused)
    adapter = public._adapter
    model = adapter._lerobot_policy.model
    mirror = pi05.Pi05Adapter(adapter._lerobot_policy)
    assert mirror.to_rpu() is adapter._lerobot_policy
    mirror.__del__()
    assert events == []
    mirror.close()
    public.close()
    adapter.__del__()
    assert [event[0] for event in events] == ["clear"] * (4 if fused else 3) + ["destroy"] * (4 if fused else 3)
    assert [handle for op, handle in events if op == "destroy"] == ([19] if fused else []) + [17, 13, 11]
    assert adapter._execution_session.stats()["state"] == "CLOSED"
    assert adapter._closed and not adapter._rpu_is_ready
    assert causal_lm._LIVE_REF is None and not _native_retirement._FAILED_RETIREMENTS
    assert model not in pi05.runtime._PI05_HANDLES
    for child, *_ in adapter._retirement_inventory():
        with pytest.raises(pi05.RPUBackendError, match="closed.*reload"):
            child.forward()
    with pytest.raises(pi05.RPUBackendError, match="closed.*reload"):
        model.sample_actions()
    replacement = type("Next", (), {})()
    causal_lm._claim_live_instance(replacement)
    public.close()
    mirror.close()
    assert causal_lm._LIVE_REF() is replacement
    assert _execution._UNSAFE_PROCESS_REASON is None


@pytest.mark.parametrize("fault", [
    "_rpu_fused_denoise_graph_cache", "_rpu_siglip_graph_cache",
    "_rpu_adarms_graph_cache", "_rpu_gemma_graph_cache",
    "_rpu_siglip_graph_cache-invariant", 19, 17, 13, 11,
])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failure_keeps_actual_resources_and_never_retries(monkeypatch, fault, error_type):
    public, events = _owner(monkeypatch, fault=fault, error_type=error_type)
    adapter = public._adapter
    resources = adapter._retirement_components
    policy, model = adapter._lerobot_policy, adapter._lerobot_policy.model
    error_type = RuntimeError if isinstance(fault, str) and fault.endswith("-invariant") else error_type
    with pytest.raises(error_type):
        public.close()
    assert not adapter._closed and adapter._rpu_is_ready
    assert adapter._execution_session.stats()["state"] == "POISONED"
    assert _native_retirement._FAILED_RETIREMENTS == [adapter]
    assert _execution._UNSAFE_PROCESS_REASON
    assert causal_lm._LIVE_REF() is policy and model in pi05.runtime._PI05_HANDLES
    destroyed = [handle for op, handle in events if op == "destroy"]
    assert destroyed == ([] if isinstance(fault, str) else [19, 17, 13, 11][:[19, 17, 13, 11].index(fault) + 1])
    for owner, name, graph_name, _destroy, _attrs, handle, finalizer, graph, _resource in resources:
        assert getattr(owner, graph_name) is graph
        assert getattr(owner, name) == (None if handle in destroyed and handle != fault else handle)
        assert not finalizer.alive
    before = list(events)
    with pytest.raises(RuntimeError):
        public.close()
    adapter.__del__()
    del public, adapter, resources, policy, model
    gc.collect()
    assert events == before
    with pytest.raises(RuntimeError, match="unsafe"):
        causal_lm._claim_live_instance(type("Next", (), {})())


@pytest.mark.parametrize("fault", [None, "_rpu_adarms_graph_cache", 13])
def test_actual_cyclic_gc_uses_the_parent_retirement(monkeypatch, fault):
    public, events = _owner(monkeypatch, fault=fault)
    ref = weakref.ref(public._adapter._lerobot_policy)
    del public
    gc.collect()
    assert all(event[0] != "legacy_finalizer" for event in events)
    destroys = [value for op, value in events if op == "destroy"]
    assert destroys == ([19, 17, 13, 11] if fault is None else [] if isinstance(fault, str) else [19, 17, 13])
    if fault is None:
        assert ref() is None and not _native_retirement._FAILED_RETIREMENTS
    else:
        assert len(_native_retirement._FAILED_RETIREMENTS) == 1
        assert _execution._UNSAFE_PROCESS_REASON
    before = list(events)
    gc.collect()
    assert events == before


def test_gc_after_weakref_release_cannot_touch_a_new_owner(monkeypatch):
    public, events = _owner(monkeypatch)
    replacement = type("Next", (), {})()
    release = causal_lm._release_live_instance

    def handoff(model=None):
        release(model)
        if model is None:
            causal_lm._claim_live_instance(replacement)

    monkeypatch.setattr(causal_lm, "_release_live_instance", handoff)
    del public
    gc.collect()
    assert events == []
    assert causal_lm._LIVE_REF() is replacement
    assert len(_native_retirement._FAILED_RETIREMENTS) == 1
    with pytest.raises(RuntimeError, match="unsafe"):
        _execution.bind_execution_session(
            replacement, {}, entry_point="replacement", graph_mode="COMPOSITE_CHILD")


def test_active_close_rejects_without_poison_then_closes(monkeypatch):
    public, events = _owner(monkeypatch)
    session = public._adapter._execution_session
    with session.execute():
        with pytest.raises(RuntimeError, match="during a forward"):
            public.close()
        assert events == [] and not _native_retirement._FAILED_RETIREMENTS
    assert session.stats()["state"] == "QUIESCENT"
    public.close()


@pytest.mark.parametrize("fault", [None, 13])
def test_parent_gc_synchronizes_actual_shared_leaf_resources(monkeypatch, fault):
    public, events = _owner(monkeypatch, fault=fault, shared=True)
    resources = [getattr(row[0], row[1].replace("_handle", "_retirement_state"))
                 for row in public._adapter._retirement_components]
    del public
    gc.collect()
    assert [value for op, value in events if op == "destroy"] == (
        [19, 17, 13, 11] if fault is None else [19, 17, 13])
    for resource in resources:
        assert not resource.finalizer.alive
        if fault is not None and resource.handle in (13, 11):
            assert resource.failed and resource.keepalive and resource.graphs
            assert resource in _native_retirement._FAILED_RETIREMENTS
        else:
            assert resource.handle is None and resource.keepalive == () and resource.graphs == ()
    before = list(events)
    for resource in resources:
        resource._gc_retire()
    gc.collect()
    assert events == before


@pytest.mark.parametrize("replacement", ["missing", "same_handle"])
def test_parent_retains_original_resource_when_publication_changes(monkeypatch, replacement):
    public, events = _owner(monkeypatch, shared=True)
    adapter = public._adapter
    model = adapter._lerobot_policy.model
    original = model._rpu_fused_denoise_retirement_state
    replaced = None
    if replacement == "same_handle":
        replaced = _native_retirement._InstalledNativeResource(
            model, original.handle, original.destroy, graphs=original.graphs,
            keepalive=original.keepalive, label=original.label, handle_name=original.handle_name)
    model._rpu_fused_denoise_retirement_state = replaced
    with pytest.raises(RuntimeError, match="identity changed"):
        public.close()
    assert events == []
    assert adapter._execution_session.stats()["state"] == "POISONED"
    assert _execution._UNSAFE_PROCESS_REASON
    for resource in (original, replaced):
        if resource is not None:
            assert resource.handle == 19 and resource.keepalive and resource.graphs
            assert resource in _native_retirement._FAILED_RETIREMENTS
            assert not resource.finalizer.alive
            resource._gc_retire()
    adapter.__del__()
    gc.collect()
    assert events == []


@pytest.mark.parametrize("fault", ["session_closed", "session_poisoned", "handle", "graph", "parent", "finalizer", "handoff"])
def test_lost_retirement_authority_preserves_native_ownership(monkeypatch, fault):
    public, events = _owner(monkeypatch)
    adapter = public._adapter
    model = adapter._lerobot_policy.model
    if fault == "session_closed":
        adapter._execution_session.close()
    elif fault == "session_poisoned":
        adapter._execution_session.poison()
    elif fault == "handle":
        model._rpu_fused_denoise_handle += 1
    elif fault == "graph":
        model._rpu_fused_denoise_graph_cache = _cache()
    elif fault == "finalizer":
        model._rpu_fused_denoise_handle_finalizer = weakref.finalize(
            model, lambda: events.append(("unsafe_finalizer", 19)))
    elif fault == "handoff":
        replacement = type("Next", (), {})()
        causal_lm._release_live_instance(adapter._lerobot_policy)
        causal_lm._claim_live_instance(replacement)
    else:
        model._pi05_retirement_owner = object()
    with pytest.raises(RuntimeError):
        public.close()
    assert events == [] and _native_retirement._FAILED_RETIREMENTS == [adapter]
    assert adapter._execution_session.stats()["state"] == "POISONED"
    assert _execution._UNSAFE_PROCESS_REASON
    adapter.__del__()
    assert events == []
