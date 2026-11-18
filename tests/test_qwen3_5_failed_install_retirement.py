"""Failed-install retirement uses real Sessions and CPU-only native doubles."""
from types import SimpleNamespace
import gc
import weakref

import pytest
import torch

import rpu_backend
from rpu_backend.api import causal_lm
from rpu_backend.api._execution import bind_execution_session
from rpu_backend.adapters import qwen3_5
from rpu_backend.adapters.qwen3_5 import text
from rpu_backend.runtime import _native_retirement


class Owner:
    pass


@pytest.fixture
def resources(monkeypatch, request):
    events, faults = [], set()

    def event(name):
        events.append(name)
        if name in faults:
            raise RuntimeError(name + " failed")

    class Cache:
        def __init__(self, **kwargs):
            self.runtime_policy = kwargs.get("runtime_policy", rpu_backend.graph.GraphRuntimePolicy())

        def clear(self):
            event("clear")

        def cache_invariant_ok(self):
            return True

    class Graph:
        def invalidate(self):
            event("discard")

    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    from rpu_backend.api import _execution
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])
    monkeypatch.setattr(qwen3_5, "_FAILED_MODEL_RETIREMENTS", [])
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_destroy", lambda _: event("destroy"), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_linear_acc32",
                        lambda _handle, _enabled: None, raising=False)
    monkeypatch.setattr(rpu_backend.graph, "GraphCache", Cache)
    monkeypatch.setattr(rpu_backend.graph, "Graph", Graph)
    root, inner = Owner(), Owner()
    root.model = inner
    session = bind_execution_session(root, {}, entry_point="failed-install-test", graph_mode="BOUNDED_ONESHOT")
    session._bind_planner_owner(inner)
    causal_lm._claim_live_instance(root)
    state = text._Qwen35NativeState(handle=7, graph_cache=Cache(), prefill_graph=Graph())
    state.retirement = _native_retirement._InstalledNativeResource(
        state, 7, torch.ops.rpu.qwen3_5_destroy, graphs=(state.graph_cache,),
        raw_graphs=(state.prefill_graph,), keepalive=(), label="Q35 pending fixture", handle_name="handle")
    state.handle_finalizer = state.retirement.finalizer
    request.addfinalizer(state.handle_finalizer.detach)
    request.addfinalizer(lambda: vars(state).pop("_retirement_close", None))
    inner._rpu_qwen3_5 = state
    root.forward = object()
    root._lm_head_w_rpu_keepalive = object()
    adapter = SimpleNamespace(_handle=7, _graph_cache=state.graph_cache, _rpu_is_ready=True, _execution_session=session)
    return SimpleNamespace(events=events, faults=faults, root=root, inner=inner,
                           session=session, state=state, adapter=adapter)


@pytest.mark.parametrize("failure", ["clear", "discard", "destroy"])
def test_outer_failed_cleanup_retains_state_and_blocks_next_process_owner(resources, failure):
    c = resources
    c.faults.add(failure)
    with pytest.raises(RuntimeError, match=failure + " failed"):
        qwen3_5._cleanup_failed_text_install(c.adapter, c.root, c.inner, False, None)
    assert c.inner._rpu_qwen3_5 is c.state
    assert c.state.handle == 7 and c.state.handle_finalizer.alive
    assert c.adapter._handle == 7
    assert c.session.stats()["state"] == "POISONED"
    assert c.root._lm_head_w_rpu_keepalive is not None
    calls = list(c.events)
    with pytest.raises(RuntimeError):
        qwen3_5._cleanup_failed_text_install(c.adapter, c.root, c.inner, False, None)
    assert c.events == calls
    # Explicit release and GC must not undo the process-terminal latch.
    causal_lm._release_live_instance(c.root)
    gc.collect()
    with pytest.raises(RuntimeError, match="process is unsafe"):
        causal_lm._claim_live_instance(Owner())


def test_outer_success_uses_raw_destroy_then_detach(resources):
    c = resources
    original = object()
    qwen3_5._cleanup_failed_text_install(c.adapter, c.root, c.inner, True, original)
    assert c.events == ["clear", "discard", "destroy"]
    assert c.session.stats()["state"] == "CLOSED"
    assert not c.state.handle_finalizer.alive
    assert not hasattr(c.inner, "_rpu_qwen3_5")
    assert c.root.forward is original
    assert causal_lm._LIVE_TERMINAL_REASON is None


@pytest.mark.parametrize("failure", [None, "clear", "discard", "destroy"])
def test_actual_outer_installer_keeps_primary_exception(resources, monkeypatch, failure):
    c = resources
    del c.inner._rpu_qwen3_5
    if failure:
        c.faults.add(failure)
    c.root.config = object()
    c.root.half = lambda: c.root
    c.root.get_output_embeddings = lambda: SimpleNamespace(weight=torch.ones(1))
    c.inner.to = lambda _device: c.inner
    c.inner.embed_tokens = object()
    adapter = object.__new__(qwen3_5.Qwen3_5Adapter)
    vars(adapter).update(model=c.root, _rpu_is_ready=False, _execution_session=c.session)
    def install(*_args, **_kwargs):
        c.inner._rpu_qwen3_5 = c.state
        return 7
    primary = RuntimeError("lm head failed")
    def fail_head(*_args):
        raise primary
    monkeypatch.setattr(qwen3_5, "install_qwen3_5_text_for_rpu", install)
    monkeypatch.setattr(qwen3_5, "transform_linear_weight", fail_head)
    with pytest.raises(RuntimeError, match="lm head failed") as caught:
        adapter.to_rpu(max_seq_len=64)
    assert caught.value is primary
    assert c.root._rpu_swizzle_started
    if failure:
        assert c.inner._rpu_qwen3_5 is c.state and c.state.retirement.failed
        assert any(failure + " failed" in note for note in primary.__notes__)
        assert c.session.stats()["state"] == "POISONED"
    else:
        assert c.events == ["clear", "discard", "destroy"]
        assert not hasattr(c.inner, "_rpu_qwen3_5")


@pytest.mark.parametrize("kind", ["vlm", "action"])
@pytest.mark.parametrize("already_failed", [False, True])
def test_g05_callers_preserve_failed_shared_text_retirement(resources, monkeypatch, kind, already_failed):
    from test_g05_adapter_contracts import _load_adapter, _load_action_adapter

    c = resources
    del c.inner._rpu_qwen3_5
    c.faults.add("destroy")
    class Castable(list):
        def to(self, *_args, **_kwargs):
            return self
        def float(self):
            return self
    c.inner.layers, c.inner.norm = Castable(), Castable()
    c.inner.config, c.inner.training = object(), False
    c.inner.forward, c.inner.decode = object(), object()
    c.root.vlm = c.inner
    module = _load_adapter() if kind == "vlm" else _load_action_adapter()
    if kind == "vlm":
        monkeypatch.setattr(module, "_preflight_g05_vlm", lambda model: model.vlm)
        monkeypatch.setattr(module, "_claim_live_instance", causal_lm._claim_live_instance)
        call = lambda: module.patch_g05_vlm_for_rpu(c.root)
    else:
        monkeypatch.setattr(module, "_check_g05_action_profile", lambda _config: None)
        call = lambda: module.patch_g05_action_expert_for_rpu(c.inner)
    primary = RuntimeError("shared pending failed" if already_failed else "post install failed")
    def fail_post(*_args):
        raise primary
    def install(*_args, **_kwargs):
        c.inner._rpu_qwen3_5 = c.state
        if already_failed:
            try:
                text._retire_failed_qwen3_5_text_install(c.inner)
            except BaseException as cleanup_error:
                primary.add_note(repr(cleanup_error))
            raise primary
        return 7
    monkeypatch.setattr(text, "install_qwen3_5_text_for_rpu", install)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_linear_acc32", fail_post, raising=False)
    with pytest.raises(RuntimeError, match=str(primary)) as caught:
        call()
    assert caught.value is primary
    assert c.inner._rpu_qwen3_5 is c.state and c.state.handle_finalizer.alive
    assert c.events == ["clear", "discard", "destroy"]
    assert c.session.stats()["state"] == "POISONED"
    assert any("cleanup" in note for note in primary.__notes__)
    assert not getattr(c.root, "_rpu_g05_vlm_ready", False)
    assert not getattr(c.inner, "_rpu_g05_action_ready", False)


@pytest.mark.parametrize("failure", [None, "clear", "discard", "destroy"])
@pytest.mark.parametrize("install_failure", ["weights", "finalizer", "publish", "retained"])
def test_real_pending_failure_preserves_primary_and_cleanup_identity(resources, monkeypatch, request, failure, install_failure):
    c = resources
    del c.inner._rpu_qwen3_5
    c.state.handle_finalizer.detach()
    if failure:
        c.faults.add(failure)
    # Empty CPU weight lists isolate the actual create/set/publish transaction;
    # neither weight swizzle nor a PrivateUse1 tensor/kernel is executed.
    c.inner.layers = []
    c.inner.norm = SimpleNamespace(weight=torch.ones(16, dtype=torch.float16))
    config = SimpleNamespace(layer_types=[], head_dim=16, num_attention_heads=8,
        num_key_value_heads=8, linear_num_value_heads=8, linear_num_key_heads=8,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        hidden_size=16, intermediate_size=16, rms_norm_eps=1e-6, num_hidden_layers=0)
    from rpu_backend.runtime import weights
    monkeypatch.setattr(weights, "convert_linear_weights_inplace", lambda *_a, **_k: None)
    monkeypatch.setattr(text, "_build_partial_mrope_cos_sin", lambda *_a, **_k: (torch.ones(1), torch.ones(1)))
    monkeypatch.setenv("RPU_QWEN3_5_FREE_HF_WEIGHTS", "0")
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_create", lambda: 17, raising=False)
    primary = RuntimeError(install_failure + " install failed")
    def fail_install(*_args, **_kwargs):
        raise primary
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights", fail_install if install_failure == "weights" else lambda *_a: None, raising=False)
    for name in ("qwen3_5_set_chunk_size_cap", "qwen3_5_set_prefill_chunk_size"):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *_a: None, raising=False)
    monkeypatch.setattr(text, "_declare_qwen3_5_chunk_envelope", lambda *_a, **_k: (64, 64))
    if install_failure == "retained":
        # Exact cold scope/cache binding is covered by the real 9B Session
        # tests. Empty weight lists isolate the actual native flag transaction.
        monkeypatch.setattr(text, "_public_retained_prefill", lambda *_a, **_k: True)
        def fail_retained(handle, retained):
            assert handle == 17 and retained is True
            assert not hasattr(c.inner, "_rpu_qwen3_5")
            raise primary
        monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_retained_prefill_graph", fail_retained, raising=False)
    if install_failure == "finalizer":
        monkeypatch.setattr(_native_retirement.weakref, "finalize", fail_install)
    elif install_failure == "publish":
        class RejectPublication(Owner):
            def __setattr__(self, name, value):
                if name == "_rpu_qwen3_5":
                    raise primary
                super().__setattr__(name, value)
        c.inner.__class__ = RejectPublication
    with pytest.raises(RuntimeError, match=install_failure + " install failed") as caught:
        text.install_qwen3_5_text_for_rpu(c.inner, config, max_seq_len=64)
    assert caught.value is primary
    if failure:
        pending = c.inner._rpu_qwen3_5
        assert pending.handle == 17
        if install_failure == "finalizer":
            assert pending.handle_finalizer is None
        else:
            request.addfinalizer(pending.handle_finalizer.detach)
            assert pending.handle_finalizer.alive
        assert any(failure + " failed" in note for note in primary.__notes__)
        assert c.session.stats()["state"] == "POISONED"
        assert causal_lm._LIVE_TERMINAL_REASON is not None
        calls = list(c.events)
        if pending.handle_finalizer is not None:
            pending.handle_finalizer()
            assert c.events == calls  # GC does not retry a known-unsafe retirement.
    else:
        assert not hasattr(c.inner, "_rpu_qwen3_5")
        assert c.events == ["clear"] * (2 if install_failure == "retained" else 1) + ["discard", "destroy"]
        assert not getattr(primary, "__notes__", ())


def test_unclaimed_low_level_cleanup_failure_installs_terminal_latch(resources):
    c = resources
    causal_lm._release_live_instance(c.root)
    del c.inner._planner_cost_session
    c.faults.add("destroy")
    with pytest.raises(RuntimeError, match="destroy failed"):
        text._retire_failed_qwen3_5_text_install(c.inner)
    assert causal_lm._LIVE_REF() is c.inner
    causal_lm._release_live_instance(c.inner)
    with pytest.raises(RuntimeError, match="process is unsafe"):
        causal_lm._claim_live_instance(Owner())


def test_failed_retirement_is_not_a_ready_runtime(resources, request):
    from test_qwen3_5_install_lifecycle import _complete_text_model
    model, state = _complete_text_model(request)
    assert qwen3_5._qwen3_5_text_runtime_state(model) is state
    state.retirement_failed = "destroy failed"
    assert qwen3_5._qwen3_5_text_runtime_state(model) is None


def test_dead_finalizer_does_not_prove_a_live_handle_retired(resources):
    c = resources
    c.state.handle_finalizer.detach()
    del c.state.retirement
    with pytest.raises(RuntimeError, match="dead finalizer"):
        text._retire_failed_qwen3_5_text_install(c.inner)
    assert c.inner._rpu_qwen3_5 is c.state and c.state.handle == 7
    assert c.events == []
    assert c.session.stats()["state"] == "POISONED"
    assert causal_lm._LIVE_TERMINAL_REASON is not None


@pytest.mark.parametrize("failure", [None, "destroy", "vision-destroy", "vision-clear", "dead-text", "dead-vision", "active"])
def test_actual_g05_explicit_close_uses_one_parent_shutdown(resources, monkeypatch, request, failure):
    from test_g05_adapter_contracts import _load_adapter
    from rpu_backend.adapters.qwen3_5.vision import _qwen3_5_vision_destroy_handle

    c = resources
    module = _load_adapter()
    policy, vision = Owner(), Owner()
    policy.model = c.root
    policy._rpu_g05_policy_ready = True
    c.root.vlm = c.inner
    c.root.vision_tower = vision
    session = bind_execution_session(policy, {}, entry_point="g05-close-test", graph_mode="NATIVE_COMPOSITE1")
    vars(c.inner).pop("_planner_cost_session")
    session._bind_planner_owner(c.inner)
    def event(name):
        c.events.append(name)
        if name == failure:
            raise RuntimeError(name + " failed")
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_destroy", lambda _h: event("vision-destroy"), raising=False)
    vision._rpu_vision_handle = 8
    vision._rpu_vision_graph_cache = SimpleNamespace(
        clear=lambda: event("vision-clear"), cache_invariant_ok=lambda: True)
    visual = text._Qwen35NativeState(handle=8)
    visual.retirement = _native_retirement._InstalledNativeResource(visual, 8,
        torch.ops.rpu.qwen3_5_vision_destroy, graphs=(vision._rpu_vision_graph_cache,),
        keepalive=(), label="G05 close Vision fixture", handle_name="handle")
    finalizer = visual.handle_finalizer = visual.retirement.finalizer
    vision._rpu_vision_handle_finalizer = finalizer
    vision._rpu_vision_retirement_state = visual
    for child, state in ((c.inner, c.state), (vision, visual)):
        child._execution_session = weakref.proxy(session)
        text._bind_qwen35_retirement(state, policy, module._rpu_g05_close_execution, child)
        request.addfinalizer(lambda state=state: vars(state).pop("_retirement_close", None))
    if failure == "dead-text":
        del c.state.retirement
    elif failure == "dead-vision":
        del visual.retirement
    elif failure == "destroy":
        c.faults.add("destroy")
    if failure == "active":
        with session.execute():
            with pytest.raises(RuntimeError, match="during a forward"):
                module._rpu_g05_close_execution(policy)
            assert not c.events
            assert causal_lm._LIVE_TERMINAL_REASON is None
        return
    if failure is not None:
        with pytest.raises(RuntimeError):
            module._rpu_g05_close_execution(policy)
        assert session.stats()["state"] == "POISONED"
        assert causal_lm._LIVE_REF() is c.root
        assert causal_lm._LIVE_TERMINAL_REASON is not None
        assert policy._rpu_g05_policy_ready is True
        assert vision._rpu_vision_handle == 8
        calls = list(c.events)
        with pytest.raises(RuntimeError, match="already failed"):
            module._rpu_g05_close_execution(policy)
        assert c.events == calls
    else:
        module._rpu_g05_close_execution(policy)
        assert c.events == ["clear", "vision-clear", "discard", "destroy", "vision-destroy"]
        assert session.stats()["state"] == "CLOSED"
        assert not c.state.handle_finalizer.alive and not finalizer.alive
        assert not hasattr(c.inner, "_rpu_qwen3_5")
        assert not hasattr(vision, "_rpu_vision_handle")
        assert causal_lm._LIVE_REF is None
        assert policy._rpu_g05_policy_ready is False
        module._rpu_g05_close_execution(policy)
        assert c.events.count("destroy") == c.events.count("vision-destroy") == 1


@pytest.mark.parametrize("with_session", [False, True])
def test_g05_parent_install_transaction_keeps_original_and_failed_child(resources, monkeypatch, with_session):
    from test_g05_adapter_contracts import _load_adapter

    c = resources
    module = _load_adapter()
    policy = Owner()
    policy.model = c.root
    c.root.action_expert = c.inner
    del c.inner._rpu_qwen3_5
    c.faults.add("destroy")
    primary = RuntimeError("policy publish failed")
    @module._transactional_g05_policy_install
    def fail_publish(owner):
        owner._rpu_g05_policy_install_started = True
        c.inner._rpu_qwen3_5 = c.state
        if with_session:
            owner._execution_session = c.session
        raise primary
    with pytest.raises(RuntimeError, match="policy publish failed") as caught:
        fail_publish(policy)
    assert caught.value is primary
    assert any("destroy failed" in note for note in primary.__notes__)
    assert c.inner._rpu_qwen3_5 is c.state and c.state.handle_finalizer.alive
    assert c.events == ["clear", "discard", "clear", "discard", "destroy"]
    assert causal_lm._LIVE_TERMINAL_REASON is not None
    assert policy._rpu_g05_policy_ready is False


@pytest.mark.parametrize("phase", ["pending", "published"])
@pytest.mark.parametrize("failure", [None, "vision-clear", "vision-discard", "vision-destroy"])
def test_g05_parent_install_strict_vision_retirement(resources, monkeypatch, request, phase, failure):
    from test_g05_adapter_contracts import _load_adapter
    from rpu_backend.adapters.qwen3_5.vision import _qwen3_5_vision_destroy_handle

    c = resources
    module = _load_adapter()
    policy, vision = Owner(), Owner()
    policy.model = c.root
    c.root.vision_tower = vision
    primary = RuntimeError("policy publish failed")
    def event(name):
        c.events.append(name)
        if name == failure:
            raise RuntimeError(name + " failed")
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_destroy", lambda _h: event("vision-destroy"), raising=False)
    handle_field = "_rpu_vision_installing_handle" if phase == "pending" else "_rpu_vision_handle"
    finalizer_field = "_rpu_vision_installing_finalizer" if phase == "pending" else "_rpu_vision_handle_finalizer"
    finalizer = weakref.finalize(vision, _qwen3_5_vision_destroy_handle, 9)
    request.addfinalizer(finalizer.detach)
    @module._transactional_g05_policy_install
    def publish(owner):
        owner._rpu_g05_policy_install_started = True
        owner._execution_session = c.session
        setattr(vision, handle_field, 9)
        setattr(vision, finalizer_field, finalizer)
        vision._rpu_vision_graph_cache = SimpleNamespace(
            clear=lambda: event("vision-clear"), cache_invariant_ok=lambda: True)
        vision._rpu_vision_debug_graph = SimpleNamespace(invalidate=lambda: event("vision-discard"))
        raise primary
    with pytest.raises(RuntimeError, match="policy publish failed") as caught:
        publish(policy)
    assert caught.value is primary
    if failure is None:
        assert c.session.stats()["state"] == "CLOSED"
        assert c.events == ["vision-clear", "vision-discard", "vision-clear", "vision-discard", "vision-destroy"]
        assert not any(name.startswith("_rpu_vision_") for name in vars(vision))
        assert not finalizer.alive
    else:
        assert c.session.stats()["state"] == "POISONED"
        assert causal_lm._LIVE_TERMINAL_REASON is not None
        assert getattr(vision, handle_field) == 9 and finalizer.alive
        assert any(failure + " failed" in note for note in primary.__notes__)
        before = list(c.events)
        finalizer()  # The real GC callback must not retry a failed retirement.
        assert c.events == before
        causal_lm._release_live_instance(c.root)
        with pytest.raises(RuntimeError, match="process is unsafe"):
            causal_lm._claim_live_instance(Owner())


@pytest.mark.parametrize("has_finalizer", [False, True])
def test_actual_vision_installer_preserves_pending_failure(resources, monkeypatch, request, has_finalizer):
    from rpu_backend.adapters.qwen3_5 import vision as adapter

    c = resources
    vision = Owner()
    vision.config = SimpleNamespace(
        num_heads=16, hidden_size=1024, intermediate_size=4096, spatial_merge_size=2)
    primary = RuntimeError("vision weight upload failed")
    monkeypatch.setattr(adapter, "_resolve_vision_model", lambda _model: vision)
    monkeypatch.setattr(adapter, "_resolve_vision_install_options", lambda **_kwargs: ())
    def destroy(_handle):
        c.events.append("vision-destroy")
        raise RuntimeError("vision destroy failed")
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_destroy", destroy, raising=False)
    def install(*_args, **_kwargs):
        vision._rpu_vision_installing = True
        vision._rpu_vision_installing_handle = 11
        if has_finalizer:
            vision._rpu_vision_installing_finalizer = weakref.finalize(vision, adapter._qwen3_5_vision_destroy_handle, 11)
            request.addfinalizer(vision._rpu_vision_installing_finalizer.detach)
        raise primary
    monkeypatch.setattr(adapter, "_install_qwen3_5_vision_for_rpu_impl", install)
    with pytest.raises(RuntimeError, match="vision weight upload failed") as caught:
        adapter.install_qwen3_5_vision_for_rpu(vision)
    assert caught.value is primary
    assert any("vision destroy failed" in note for note in primary.__notes__)
    assert vision._rpu_vision_installing_handle == 11
    assert causal_lm._LIVE_TERMINAL_REASON is not None
    with pytest.raises(RuntimeError, match="unsafe process"):
        adapter._rollback_qwen3_5_vision_install(vision)
    assert c.events == ["vision-destroy"]
    if has_finalizer:
        assert vision._rpu_vision_installing_finalizer.alive
        vision._rpu_vision_installing_finalizer()
        assert c.events == ["vision-destroy"]


@pytest.mark.parametrize("shared_returned", [False, True])
def test_g05_vision_wrapper_keeps_failed_shared_resources(resources, monkeypatch, request, shared_returned):
    from test_g05_adapter_contracts import _load_adapter
    from rpu_backend.adapters.qwen3_5 import vision as adapter

    c = resources
    module = _load_adapter()
    primary = RuntimeError("vision wrapper publish failed")
    class Model(Owner):
        def __setattr__(self, name, value):
            if name == "_rpu_g05_vision_ready" and value is True:
                raise primary
            super().__setattr__(name, value)
    model, vision = Model(), Owner()
    model.vision_tower = vision
    original = vision.forward = object()
    vision.half = lambda: vision
    monkeypatch.setattr(module, "_preflight_g05_vision", lambda _model: (vision, SimpleNamespace(), (0, "factorized", 24, None, False)))
    def destroy(_handle):
        c.events.append("vision-destroy")
        raise RuntimeError("vision destroy failed")
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_destroy", destroy, raising=False)
    def install(_vision, **_kwargs):
        vision._rpu_vision_handle = 13
        vision._rpu_vision_handle_finalizer = weakref.finalize(vision, adapter._qwen3_5_vision_destroy_handle, h=13)
        request.addfinalizer(vision._rpu_vision_handle_finalizer.detach)
        vision.forward = lambda *_args: None
        if not shared_returned:
            raise primary
        return 13
    monkeypatch.setattr(adapter, "install_qwen3_5_vision_for_rpu", install)
    with pytest.raises(RuntimeError, match="vision wrapper publish failed") as caught:
        module.patch_g05_vision_for_rpu(model, _allow_certified_g05_k1=True)
    assert caught.value is primary
    assert any("vision destroy failed" in note for note in primary.__notes__)
    assert vision._rpu_vision_handle == 13 and vision._rpu_vision_handle_finalizer.alive
    assert vision.forward is not original
    assert c.events == ["vision-destroy"]
    assert causal_lm._LIVE_TERMINAL_REASON is not None
    vision._rpu_vision_handle_finalizer()
    assert c.events == ["vision-destroy"]
