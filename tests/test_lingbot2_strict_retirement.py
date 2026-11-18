"""Real parent Session/Graph/shared-resource retirement; native ops are CPU doubles."""
import gc
import threading
import weakref
from types import SimpleNamespace

import pytest

from rpu_backend.adapters.lingbot_vla_v2 import runtime, convert
from rpu_backend.api import _execution, causal_lm, lingbot2
from rpu_backend.runtime import _native_retirement
from test_graph_cache_freeze import _cache, _sig


class Child:
    pass


@pytest.fixture(autouse=True)
def isolated_process(monkeypatch):
    gc.collect()
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(runtime, "_FAILED_RETIREMENTS", [])
    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])


def _owner(monkeypatch, *, fault=None, error_type=RuntimeError, owner=None, claim=True):
    owner = owner or runtime.LingbotVlaV2Policy.__new__(runtime.LingbotVlaV2Policy)
    owner._closed = owner._live_slot_released = False
    owner._cleanup_ok = None
    owner._gc_retirement_enabled = True
    owner._exp, owner._exp_keep = 13, Child()
    owner._cache = Child()
    owner._text, owner._visual = Child(), Child()
    owner._vlm = SimpleNamespace(language_model=owner._text, visual=owner._visual)
    owner._z2_state = {}
    events = []

    def graph(name):
        cache = _cache()
        cache.begin_warmup()
        with cache.capture(_sig(name)):
            pass
        cache.freeze()
        clear = cache._impl.clear

        def do_clear():
            events.append(("clear", name))
            if fault == name:
                raise error_type("graph failure")
            clear()

        cache._impl.clear = do_clear
        cache._impl.cache_invariant_ok = lambda: fault != name + "-invariant"
        return cache

    def destroy(handle):
        events.append(("destroy", handle))
        if fault == handle:
            raise error_type("native failure")

    for attr in ("_dgc", "_egc", "_vgc"):
        setattr(owner, attr, graph(attr))
    for child, name, handle, op in (
        (owner._text, "decoder", 12, "causal_decoder_destroy"),
        (owner._visual, "vision", 11, "qwen3vl_vision_destroy"),
    ):
        prefix = f"_rpu_{name}_"
        setattr(child, prefix + "handle", handle)
        cache = graph(name)
        setattr(child, prefix + "graph_cache", cache)
        if name == "decoder":
            child._rpu_text_graph_cache = graph("text")
        child.weight = Child()
        state = _native_retirement._InstalledNativeResource(
            child, handle, destroy, graphs=(cache,), keepalive=(child.weight,),
            label=name, handle_name=prefix + "handle")
        setattr(child, prefix + "retirement_state", state)
        setattr(child, prefix + "handle_finalizer", state.finalizer)
        runtime._take_lingbot2_component_retirement(owner, child, name)
        monkeypatch.setattr(getattr(runtime.torch.ops, "rpu"), op, destroy, raising=False)
    monkeypatch.setattr(runtime.torch.ops.rpu, "lingbot_v2_moe_destroy", destroy, raising=False)
    if not hasattr(owner, "_execution_session"):
        _execution.bind_execution_session(owner, {}, entry_point="Ling2 retire",
                                          supported_components=runtime.LINGBOT2_EXECUTION_COMPONENTS,
                                          graph_mode="COMPOSITE_CHILD")
    owner._execution_session._bind_planner_owner(owner, runtime.LINGBOT2_ACTION_COMPONENT)
    if claim:
        causal_lm._claim_live_instance(owner)
    return owner, events


def test_success_clears_all_graphs_before_any_native_and_updates_owned_states(monkeypatch):
    owner, events = _owner(monkeypatch)
    text, vision = owner._text, owner._visual
    session = owner._execution_session
    owner.close()
    owner.close()
    owner.__del__()
    assert events == [("clear", n) for n in ("_dgc", "_egc", "_vgc", "text", "decoder", "vision")] + [
        ("destroy", 13), ("destroy", 12), ("destroy", 11)]
    assert session.stats()["state"] == "CLOSED"
    assert owner._closed and owner._cleanup_ok and owner._exp_keep is None
    for child, name in ((text, "decoder"), (vision, "vision")):
        state = getattr(child, f"_rpu_{name}_retirement_state")
        assert state.handle is None and not state.keepalive and not state.finalizer.alive
        assert getattr(child, f"_rpu_{name}_handle") is None
        assert not hasattr(child, "_lingbot2_retirement_owner")
    assert causal_lm._LIVE_REF is None


@pytest.mark.parametrize("fault", ["_dgc", "_egc", "_vgc", "text", "decoder", "vision", "vision-invariant", 13, 12, 11])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, ReferenceError])
def test_failure_preserves_unretired_resources_and_never_retries(monkeypatch, fault, error_type):
    owner, events = _owner(monkeypatch, fault=fault, error_type=error_type)
    keep, text, vision = owner._exp_keep, owner._text, owner._visual
    expected = RuntimeError if fault == "vision-invariant" else error_type
    with pytest.raises(expected):
        owner.close()
    before = list(events)
    assert owner._cleanup_ok is False and not owner._closed
    assert owner._exp_keep is keep and owner._text is text and owner._visual is vision
    assert owner._execution_session.stats()["state"] == "POISONED"
    assert runtime._FAILED_RETIREMENTS == [owner] and _execution._UNSAFE_PROCESS_REASON
    destroyed = [n for op, n in events if op == "destroy"]
    assert destroyed == ([] if isinstance(fault, str) else [13, 12, 11][: [13, 12, 11].index(fault) + 1])
    assert owner._exp == (None if fault in (12, 11) else 13)
    assert text._rpu_decoder_retirement_state.handle == (None if fault == 11 else 12)
    assert vision._rpu_vision_retirement_state.handle == 11
    with pytest.raises(RuntimeError):
        owner.close()
    owner.__del__()
    text._rpu_decoder_handle_finalizer()
    vision._rpu_vision_handle_finalizer()
    gc.collect()
    assert events == before


@pytest.mark.parametrize("fault", [None, "vision", 12])
def test_actual_last_owner_cyclic_gc_keeps_failed_weights_without_retry(monkeypatch, fault):
    owner, events = _owner(monkeypatch, fault=fault)
    weight = weakref.ref(owner._exp_keep)
    vision_weight = weakref.ref(owner._visual.weight)
    weight_id, vision_weight_id = id(owner._exp_keep), id(owner._visual.weight)
    owner_ref = weakref.ref(owner)
    del owner
    gc.collect()
    before = list(events)
    gc.collect()
    assert events == before
    assert owner_ref() is None  # weakrefs clear before cyclic __del__.
    if fault is None:
        assert weight() is None and vision_weight() is None
        assert [n for op, n in events if op == "destroy"] == [13, 12, 11]
    else:
        retained, = runtime._FAILED_RETIREMENTS
        assert id(retained._exp_keep) == weight_id
        assert id(retained._visual.weight) == vision_weight_id
        # GC clears cyclic weakrefs before __del__ can retain the actual owner.
        weight_after = weakref.ref(retained._exp_keep)
        del retained
        gc.collect()
        assert weight_after() is not None and events == before
        assert _execution._UNSAFE_PROCESS_REASON


def test_old_cyclic_gc_after_live_handoff_performs_no_device_work(monkeypatch):
    owner, events = _owner(monkeypatch)
    replacement = Child()
    def handoff(ref):
        causal_lm._claim_live_instance(replacement)
    causal_lm._LIVE_REF = weakref.ref(owner, handoff)
    del owner
    gc.collect()
    assert events == [] and runtime._FAILED_RETIREMENTS
    assert _execution._UNSAFE_PROCESS_REASON
    with pytest.raises(RuntimeError, match="unsafe|restart|Restart"):
        _execution.bind_execution_session(replacement, {}, entry_point="successor", graph_mode="COMPOSITE_CHILD")


def test_closed_session_noop_preserves_live_handles(monkeypatch):
    owner, events = _owner(monkeypatch)
    owner._execution_session.close()
    with pytest.raises(RuntimeError, match="closed Session still owns"):
        owner.close()
    assert events == [] and owner._exp == 13 and owner._cleanup_ok is False
    assert runtime._FAILED_RETIREMENTS == [owner]


def test_active_explicit_close_rejects_without_mutation_or_poison(monkeypatch):
    owner, events = _owner(monkeypatch)
    with owner._execution_session.execute():
        with pytest.raises(RuntimeError, match="during a forward"):
            owner.close()
        assert events == [] and owner._cleanup_ok is None
    owner.close()


@pytest.mark.parametrize("drift", ["handle", "parent", "owner", "marker"])
def test_child_ownership_drift_rejects_before_graph_or_native(monkeypatch, drift):
    owner, events = _owner(monkeypatch)
    state = owner._text._rpu_decoder_retirement_state
    foreign = Child()
    if drift == "handle":
        owner._text._rpu_decoder_handle = 88
    elif drift == "parent":
        state.parent = weakref.ref(foreign)
    elif drift == "owner":
        state.owner = weakref.ref(foreign)
    else:
        owner._text._lingbot2_retirement_owner = foreign
    with pytest.raises(RuntimeError, match="ownership changed"):
        owner.close()
    assert events == [] and owner._exp == 13 and state.handle == 12
    assert runtime._FAILED_RETIREMENTS == [owner]


def test_held_original_graph_also_clears_when_public_graph_reference_drifts(monkeypatch):
    owner, events = _owner(monkeypatch, fault="vision")
    owner._visual._rpu_vision_graph_cache = None
    with pytest.raises(RuntimeError, match="graph failure"):
        owner.close()
    assert events[-1] == ("clear", "vision")
    assert not any(e[0] == "destroy" for e in events)


@pytest.mark.parametrize("prepared", [True, False])
def test_physical_z2_failure_is_first_and_retirement_is_not_retried(monkeypatch, prepared):
    owner, events = _owner(monkeypatch)
    owner._z2_state = dict(prepare_attempted=True, prepared=prepared,
                          owner_thread=threading.get_ident(), vision_handle=11,
                          text_handle=12, expert_handle=13, plan_hash=42,
                          prepare_attempt_token=3)
    error = KeyboardInterrupt("physical Z2 failure")

    def fail(*args):
        events.append(("z2", args[3]))
        raise error

    for op in ("lingbot2_multiview_spm_z2_close", "lingbot2_multiview_spm_z2_cancel_prepare"):
        monkeypatch.setattr(runtime.torch.ops.rpu, op, fail, raising=False)
    with pytest.raises(KeyboardInterrupt) as caught:
        owner.close()
    assert caught.value is error
    assert events == [("z2", 42 if prepared else 3)]
    assert owner._z2_state["retirement_error"] is error
    owner.__del__()
    assert not runtime._close_multiview_spm_z2_state(owner._z2_state)
    assert len(events) == 1


def test_z2_cancel_recovery_failure_keeps_first_cleanup_exception(monkeypatch):
    owner, events = _owner(monkeypatch)
    owner._z2_state = dict(prepare_attempted=True, prepared=False,
                          owner_thread=threading.get_ident(), vision_handle=11,
                          text_handle=12, expert_handle=13, prepare_attempt_token=3)
    original = ValueError("cancel failure")
    def cancel(*args):
        raise original
    def close(*args):
        raise RuntimeError("poisoned close failure")
    monkeypatch.setattr(runtime.torch.ops.rpu, "lingbot2_multiview_spm_z2_cancel_prepare", cancel, raising=False)
    monkeypatch.setattr(runtime.torch.ops.rpu, "lingbot2_multiview_spm_z2_close", close, raising=False)
    with pytest.raises(ValueError) as caught:
        owner.close()
    assert caught.value is original and "poisoned close failure" in " ".join(original.__notes__)
    assert events == []


@pytest.mark.parametrize("fault", [None, 12])
def test_partial_builder_preserves_original_exception_and_claim(monkeypatch, fault):
    saved = {}
    original = ValueError("original builder failure")

    def build(policy, *args, **kwargs):
        policy._rpu_materialization_started = True
        owner, events = _owner(monkeypatch, owner=policy, claim=False, fault=fault)
        saved.update(owner=owner, events=events)
        raise original

    monkeypatch.setattr(runtime, "_build_lingbot_vla_v2_impl", build)
    with pytest.raises(ValueError) as caught:
        runtime.build_lingbot_vla_v2("unused")
    assert caught.value is original and causal_lm._LIVE_REF() is saved["owner"]
    if fault:
        assert "native failure" in " ".join(original.__notes__)
        assert saved["owner"]._text._rpu_decoder_handle == 12
        assert saved["owner"]._cleanup_ok is False
    else:
        assert saved["owner"]._closed


@pytest.mark.parametrize("failure", [False, True])
def test_facade_explicit_close_does_not_consume_gc_callback_on_failure(monkeypatch, failure):
    policy, events = _owner(monkeypatch, fault=12 if failure else None)
    facade = lingbot2.Lingbot2Policy.__new__(lingbot2.Lingbot2Policy)
    facade._closed = False
    facade._policy = policy
    facade._rpu_build_lock = threading.RLock()
    facade._env_snapshot = {}
    facade._execution_session = policy._execution_session
    facade._policy_finalizer = weakref.finalize(facade, lingbot2._finalize_runtime_policy, policy, {})
    finalizer = facade._policy_finalizer
    if failure:
        with pytest.raises(RuntimeError, match="native failure"):
            facade.close()
        assert finalizer.alive and facade._policy is policy
        before = list(events)
        del facade
        gc.collect()
        assert events == before
    else:
        facade.close()
        assert not finalizer.alive and facade._policy is None and facade._closed


@pytest.mark.parametrize("fault", [None, "vision", 12])
def test_actual_facade_gc_with_expired_runtime_session_proxy(monkeypatch, fault):
    policy, events = _owner(monkeypatch, fault=fault)
    facade = lingbot2.Lingbot2Policy.__new__(lingbot2.Lingbot2Policy)
    facade._execution_session = policy._execution_session
    policy._execution_session = weakref.proxy(facade._execution_session)
    facade._policy = policy
    facade._cycle = facade
    facade._policy_finalizer = weakref.finalize(facade, lingbot2._finalize_runtime_policy, policy, {})
    facade_ref = weakref.ref(facade)
    keep_id = id(policy._exp_keep)
    del policy, facade
    gc.collect()
    before = list(events)
    gc.collect()
    assert facade_ref() is None and events == before
    if fault is None:
        assert [n for op, n in events if op == "destroy"] == [13, 12, 11]
        assert not runtime._FAILED_RETIREMENTS
    else:
        retained, = runtime._FAILED_RETIREMENTS
        assert id(retained._exp_keep) == keep_id and retained._visual is not None
        assert _execution._UNSAFE_PROCESS_REASON


@pytest.mark.parametrize("failing_op", ["set_weights", "set_base_scales", "set_packed_weights", "set_packed_scales"])
def test_actual_expert_builder_publishes_keep_before_each_fallible_bind(monkeypatch, failing_op):
    import torch
    from rpu_backend.runtime import weights

    owner, _ = _owner(monkeypatch, fault=13)
    monkeypatch.setenv("RPU_LINGBOT2_BASE_W8A16", "1")
    monkeypatch.setenv("RPU_LINGBOT2_EXPERT_W8A16", "1")
    monkeypatch.setenv("RPU_LINGBOT2_EXPERT_W4A16", "0")
    monkeypatch.setenv("RPU_LINGBOT2_GROUPED_EXPERTS", "1")
    geom = dict(EXP_LAYERS=1, EXP_HID=2, NQ=1, NKV=1, HD=2, RMS_EPS=1e-6,
                NUM_EXPERTS=1, TOP_K=1, ROUTED_INTER=2, SHARED_INTER=2,
                SHARED_INTER_PAD=2, ROUTED_SCALING=1., N_ACTION=1,
                ATTN_TP=1, MLP_CORES=1, ROUTER_CORES=1)
    monkeypatch.setattr(convert, "load_router_correction_biases", lambda *a, **k: [torch.zeros(1)])
    monkeypatch.setattr(convert, "_h", lambda t: t.half())
    monkeypatch.setattr(convert, "_q", lambda t: t)
    monkeypatch.setattr(convert, "_build_rope_tables", lambda *a: (torch.ones(2, 2), torch.ones(2, 2)))
    for name in ("tp_col_swizzle_mc_weight", "tp_row_swizzle_mc_weight"):
        monkeypatch.setattr(weights, name, lambda t, *a, **k: t)
    monkeypatch.setattr(torch.ops.rpu, "lingbot_v2_moe_create", lambda *a: 13, raising=False)
    observed = []

    def setter(name):
        def call(handle, *args):
            assert owner._exp == handle == 13
            assert len(owner._exp_keep) == 33
            if name == "set_weights":
                assert owner._exp_keep[0] is args[0]
                assert owner._exp_keep[17] is args[9]  # original RoPE tuple contract
            if name == "set_packed_weights":
                assert owner._exp_keep[-6] is args[0] and args[0]
            if name == "set_packed_scales":
                assert owner._exp_keep[-3] is args[0] and args[0]
            observed.append(name)
            if name == failing_op:
                raise ValueError(f"original {name} failure")
        return call

    for name in ("set_weights", "set_base_scales", "set_packed_weights", "set_packed_scales"):
        monkeypatch.setattr(torch.ops.rpu, "lingbot_v2_moe_" + name, setter(name), raising=False)

    def gw(name):
        return torch.ones((1, 2, 2) if "mlp.experts." in name else (2, 2))

    def build_and_retire():
        try:
            convert.build_expert_moe(
                gw, max_seq=2, rope_theta=1., geom=geom,
                runtime_config=runtime._Lingbot2MoeColdConfig.from_env(),
                on_handle_created=lambda handle, keep: vars(owner).update(_exp=handle, _exp_keep=keep))
        except ValueError as error:
            try:
                owner.close()
            except RuntimeError as cleanup_error:
                error.add_note(repr(cleanup_error))
            return str(error)

    assert build_and_retire() == f"original {failing_op} failure"
    assert observed[-1] == failing_op
    weight_ref = weakref.ref(owner._exp_keep[0][0])
    del owner
    gc.collect()
    gc.collect()
    assert weight_ref() is not None and runtime._FAILED_RETIREMENTS
