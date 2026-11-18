"""Actual NextDiT constructor/NavDP builder, with CPU-only native doubles."""
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

import rpu_backend
from rpu_backend.adapters.internvla_n1 import nextdit, execution
from rpu_backend.adapters.navdp import runtime as navdp
from rpu_backend.api import _execution, causal_lm
from rpu_backend.runtime import _native_retirement
from test_graph_cache_freeze import _cache, _sig


def _tensor(*shape):
    return torch.ones(1).expand(shape)


def _nextdit_model():
    def linear(out, inp):
        return SimpleNamespace(in_features=inp, out_features=out,
            weight=_tensor(out, inp), bias=_tensor(out))
    def norm():
        return SimpleNamespace(weight=_tensor(384), bias=_tensor(384), eps=1e-5)
    def attention():
        return SimpleNamespace(heads=6, to_q=linear(384, 384),
            to_k=linear(384, 384), to_v=linear(384, 384),
            to_out=[linear(384, 384)], norm_q=norm(), norm_k=norm())
    layer = SimpleNamespace(attn1=attention(), attn2=attention(), head_dim=64,
        feed_forward=SimpleNamespace(linear_1=linear(1024, 384),
            linear_2=linear(384, 1024), linear_3=linear(1024, 384)),
        norm1=SimpleNamespace(norm=norm(), linear=linear(1536, 384)),
        norm2=norm(), ffn_norm1=norm(), ffn_norm2=norm(),
        norm1_context=norm(), gate=_tensor(6))
    return SimpleNamespace(layers=[layer] * 12, caption_projection=object(),
        time_caption_embed=object(), parameters=lambda: iter([_tensor(1)]),
        norm_out=SimpleNamespace(linear_1=linear(384, 384),
            linear_2=linear(384, 384), norm=norm()))


def _navdp_weights():
    weights = {}
    for layer in range(16):
        prefix = f"decoder.layers.{layer}."
        for name in ("self_attn", "multihead_attn"):
            for suffix, shape in (("in_proj_weight", (1152, 384)), ("in_proj_bias", (1152,)),
                                  ("out_proj.weight", (384, 384)), ("out_proj.bias", (384,))):
                weights[prefix + name + "." + suffix] = _tensor(*shape)
        for name, out, inp in (("linear1", 1536, 384), ("linear2", 384, 1536)):
            weights[prefix + name + ".weight"] = _tensor(out, inp)
            weights[prefix + name + ".bias"] = _tensor(out)
        for name in ("norm1", "norm2", "norm3"):
            for slot in ("weight", "bias"):
                weights[prefix + name + "." + slot] = _tensor(384)
    for key, shape in (("input_embed.weight", (384, 3)), ("input_embed.bias", (384,)),
            ("out_pos_embed", (1, 32, 384)), ("cond_pos_embed", (1, 34, 384)),
            ("layernorm.weight", (384,)), ("layernorm.bias", (384,)),
            ("action_head.weight", (3, 384)), ("action_head.bias", (3,))):
        weights[key] = _tensor(*shape)
    return weights


class Cache:
    def __init__(self, **kwargs):
        self.max_seq_len = kwargs["max_seq_len"]
        self.k_caches, self.v_caches = [], []


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensors(child)


@pytest.fixture
def native(monkeypatch):
    gc.collect()
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    state = dict(events=[], resources=[], fault=None, destroy_error=None, next_handle=41, envelopes=[])
    for module in (nextdit, navdp):
        monkeypatch.setattr(module, "require_controlled_evaluation", lambda _: None)
        monkeypatch.setattr(module, "verify_asset_manifest", lambda *args: None)
        monkeypatch.setattr(module, "_col", lambda value, *args: value)
        monkeypatch.setattr(module, "_row", lambda value, *args: value)
    monkeypatch.setattr(nextdit, "_to_rpu_half", lambda value: value)
    monkeypatch.setattr(navdp, "_hr", lambda value: value)
    monkeypatch.setattr(navdp, "tp_col_swizzle_mc_weight", lambda value, **kwargs: value)
    for key in ("RPU_INTERNVLA_NEXTDIT_FOLD_NORM_MOD", "RPU_INTERNVLA_NEXTDIT_STEP_PREP_RPU",
                "RPU_INTERNVLA_NEXTDIT_FUSED_QKQ"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(rpu_backend.api, "RPUCache", Cache)
    monkeypatch.setattr(navdp, "RPUCache", Cache)
    def graph():
        cache = _cache()
        cache._impl.cache_invariant_ok = lambda: True
        return cache
    monkeypatch.setattr(rpu_backend.graph, "GraphCache", graph)
    for name in ("zeros", "empty"):
        original = getattr(torch, name)
        def allocate(*args, _original=original, **kwargs):
            if kwargs.get("device") == "rpu":
                kwargs["device"] = "cpu"
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, allocate)
    resource_type = _native_retirement._InstalledNativeResource
    def resource(*args, **kwargs):
        value = resource_type(*args, **kwargs)
        state["resources"].append(value)
        return value
    for module in (nextdit, navdp):
        monkeypatch.setattr(module, "_InstalledNativeResource", resource)
    def create(*args):
        handle = state["next_handle"]
        state["next_handle"] += 1
        state["events"].append(("create", handle))
        return handle
    def destroy(handle):
        state["events"].append(("destroy", handle))
        if state["destroy_error"] is not None:
            raise state["destroy_error"]
    for prefix in ("internvla_nextdit", "navdp"):
        monkeypatch.setattr(torch.ops.rpu, prefix + "_create", create, raising=False)
        monkeypatch.setattr(torch.ops.rpu, prefix + "_destroy", destroy, raising=False)
    state["debug_selections"] = {}
    def debug_export(handle, enabled):
        if state["fault"] == "internvla_nextdit_set_debug_export":
            raise KeyboardInterrupt("original debug setter failure")
        assert handle not in state["debug_selections"]
        state["debug_selections"][handle] = enabled
    monkeypatch.setattr(torch.ops.rpu, "internvla_nextdit_set_debug_export", debug_export, raising=False)
    def setter(name):
        def invoke(handle, *args):
            resource = next(item for item in state["resources"] if item.handle == handle)
            kept = {id(t) for t in _tensors(resource.keepalive)}
            assert all(id(t) in kept for t in _tensors(args)), "setter exposed an unowned address"
            state["events"].append((name, handle))
            if state["fault"] == name:
                raise KeyboardInterrupt("original setter failure")
        return invoke
    for name in ("internvla_nextdit_set_weights", "internvla_nextdit_set_fused_qkq_weights",
                 "internvla_nextdit_set_step_prepare_weights", "navdp_set_weights",
                 "navdp_set_denoise_glue"):
        monkeypatch.setattr(torch.ops.rpu, name, setter(name), raising=False)
    def envelope(name):
        def invoke(handle, *capacities):
            state["envelopes"].append((name, handle, *capacities))
            if state["fault"] == name:
                raise KeyboardInterrupt("original envelope failure")
        return invoke
    for prefix in ("internvla_nextdit", "navdp"):
        name = prefix + "_set_chunk_envelope"
        monkeypatch.setattr(torch.ops.rpu, name, envelope(name), raising=False)
    import safetensors.torch
    monkeypatch.setattr(safetensors.torch, "load_file", lambda _: _navdp_weights())
    yield state
    for resource in state["resources"] + _native_retirement._FAILED_RETIREMENTS:
        resource.finalizer.detach()
    torch.set_num_threads(threads)


def _build(kind, *, deferred=False):
    if kind == "nextdit":
        return nextdit.build_nextdit_runtime(_nextdit_model(), asset_paths=("unused",))
    return navdp.build_navdp("unused", _defer_execution_session=deferred)


KINDS = ("nextdit", "navdp")


@pytest.mark.parametrize("fold,prepare", [(False, True), (True, False), (True, True)])
def test_nextdit_fold_prepare_original_cold_dependency_and_unroll_preparation(native, monkeypatch, fold, prepare):
    """Actual cold builder and prepare_steps/denoise caller, without a numeric claim."""
    # Retirement-only doubles normally skip conversion; this preparation check
    # needs the actual FP32->FP16 boundary while keeping storage on CPU.
    monkeypatch.setattr(nextdit, "_to_rpu_half", lambda value: value.half().contiguous())
    monkeypatch.setenv("RPU_INTERNVLA_NEXTDIT_FUSED_QKQ", "0")
    monkeypatch.setenv("RPU_INTERNVLA_NEXTDIT_FOLD_NORM_MOD", str(int(fold)))
    monkeypatch.setenv("RPU_INTERNVLA_NEXTDIT_STEP_PREP_RPU", str(int(prepare)))
    model = _nextdit_model()
    layer = model.layers[0]
    for value, norm in enumerate((layer.norm1.norm, layer.norm2, layer.ffn_norm1, layer.ffn_norm2), 2):
        norm.weight = torch.full((384,), float(value))
    owner = nextdit.build_nextdit_runtime(model, asset_paths=("unused",),
        action_encoder=torch.nn.Linear(3, 384), action_decoder=torch.nn.Linear(384, 3),
        action_position=torch.zeros(32, 384))
    try:
        active = fold and prepare
        assert owner._rpu_step_prepare is active
        assert (owner._fold_norm_weights is not None) is fold
        assert ("internvla_nextdit_set_step_prepare_weights", owner._handle) in native["events"] if active else (
            "internvla_nextdit_set_step_prepare_weights", owner._handle) not in native["events"]
        assert ("internvla_nextdit_set_fused_qkq_weights", owner._handle) not in native["events"]
        timesteps = torch.arange(900, -100, -100)
        timestep_key = tuple(timesteps.tolist())
        time_embedding = torch.full((10, 384), .001)
        caption_embedding = torch.full((1, 384), .002)
        owner._step_time_embeddings[timestep_key] = time_embedding
        condition = torch.zeros(1, 260, 768)
        state = SimpleNamespace(caption_embedding=caption_embedding)
        monkeypatch.setattr(owner, "_condition_state", lambda *a, **k: (0, state))
        ordinary, final_scale = owner._step_prepare_jit(time_embedding, caption_embedding)
        # Ordinary preparation remains host math even when both cold flags are on.
        owner.prepare_steps(timesteps, condition)
        ordinary_state = owner._step_batches[(0, timestep_key, False)]
        assert torch.equal(ordinary_state.modulation_rpu, ordinary.half())
        assert torch.equal(ordinary_state.final_scale_rpu, final_scale.half())
        owner.prepare_steps(timesteps, condition, _fold_for_unroll=fold)
        unroll_state = owner._step_batches[(0, timestep_key, fold)]
        if active:
            expected = torch.nn.functional.silu(time_embedding + caption_embedding).half()
            assert unroll_state.modulation_rpu.shape == (10, 384)
            assert torch.equal(unroll_state.modulation_rpu, expected)
            assert unroll_state.final_scale_rpu.data_ptr() == owner._step_final_scale_slots[0].data_ptr()
        elif fold:
            # The existing host fold changes all four modulation planes.
            scales = torch.tensor([2., 3., 4., 5.], dtype=torch.float16).view(1, 1, 4, 1)
            assert torch.equal(unroll_state.modulation_rpu, ordinary.half() * scales)
            assert torch.equal(unroll_state.final_scale_rpu, final_scale.half())
        class BeforePlan(RuntimeError):
            pass
        invocations = []
        def stop_before_plan(**invocation):
            invocations.append(invocation)
            raise BeforePlan("numeric/Graph unroll gate is not certified by this preparation test")
        monkeypatch.setattr(owner, "_execution_plan", stop_before_plan)
        with pytest.raises(BeforePlan):
            owner.denoise_actions(torch.zeros(1, 32, 3), timesteps, torch.full((10,), -.1), condition)
        assert invocations == [dict(action_rows=32, cross_seq=260, use_action_encoder=True,
            populate_cross_cache=True, action_batch=1, denoise_steps=10, prepare_modulation_on_rpu=active)]
        before = dict(owner._step_batches)
        for steps in (0, 11):
            with pytest.raises(ValueError, match="length in"):
                owner.denoise_actions(torch.zeros(1, 32, 3), torch.zeros(steps), torch.zeros(steps), condition)
        assert owner._step_batches == before and len(invocations) == 1
    finally:
        owner.destroy()


@pytest.mark.parametrize("enabled,failure", [(False, False), (True, False), (True, True)])
def test_nextdit_qkq_cold_constructor_installs_exact_three_projections(native, monkeypatch, enabled, failure):
    """Actual constructor/set_weights; only swizzle/device endpoints are CPU doubles."""
    monkeypatch.setenv("RPU_INTERNVLA_NEXTDIT_FUSED_QKQ", str(int(enabled)))
    monkeypatch.setenv("RPU_INTERNVLA_NEXTDIT_FOLD_NORM_MOD", "0")
    monkeypatch.setenv("RPU_INTERNVLA_NEXTDIT_STEP_PREP_RPU", "0")
    model = _nextdit_model()
    layer = model.layers[0]
    for index, projection in enumerate((layer.attn1.to_q, layer.attn1.to_k, layer.attn2.to_q), 1):
        projection.weight = torch.full((384, 384), float(index))
    original = torch.ops.rpu.internvla_nextdit_set_fused_qkq_weights
    installed = []
    def install(handle, weights):
        assert len(weights) == 12
        for weight in weights:
            assert weight.shape == (1152, 384)
            for index, chunk in enumerate(weight.chunk(3, dim=0), 1):
                assert torch.equal(chunk, torch.full_like(chunk, float(index)))
        installed.append(handle)
        original(handle, weights)
    monkeypatch.setattr(torch.ops.rpu, "internvla_nextdit_set_fused_qkq_weights", install)
    if failure:
        native["fault"] = "internvla_nextdit_set_fused_qkq_weights"
        with pytest.raises(KeyboardInterrupt, match="original setter failure"):
            nextdit.build_nextdit_runtime(model, asset_paths=("unused",))
        assert installed == [41] and ("destroy", 41) in native["events"]
        assert ("internvla_nextdit_set_step_prepare_weights", 41) not in native["events"]
        return
    owner = nextdit.build_nextdit_runtime(model, asset_paths=("unused",))
    try:
        assert owner._fused_qkq is enabled and owner._fold_norm_weights is None
        assert owner._rpu_step_prepare is False
        assert installed == ([owner._handle] if enabled else [])
    finally:
        owner.destroy()


@pytest.mark.parametrize("cross_capacity", [64, 256, 640])
def test_nextdit_cold_envelope_preserves_actual_self_and_cross_roles(native, cross_capacity):
    owner = nextdit.build_nextdit_runtime(_nextdit_model(), asset_paths=("unused",),
                                        max_cross_seq_len=cross_capacity)
    try:
        assert owner._self_cache.max_seq_len == 32
        assert native["envelopes"] == [("internvla_nextdit_set_chunk_envelope", owner._handle,
            owner._packed_self_cache.max_seq_len, owner._cross_cache.max_seq_len, 0)]
        assert owner._packed_self_cache.max_seq_len == 128
        assert owner._cross_cache.max_seq_len == cross_capacity
    finally:
        owner.destroy()


def test_navdp_cold_envelope_uses_actual_base_cache(native):
    owner = _build("navdp")
    try:
        assert owner._cache.max_seq_len == 64
        assert native["envelopes"] == [("navdp_set_chunk_envelope", owner._handle, 64, 0)]
    finally:
        owner.destroy()


@pytest.mark.parametrize("kind", KINDS)
def test_action_cold_envelope_failure_cleans_pending_resource_without_publish(native, kind):
    prefix = "internvla_nextdit" if kind == "nextdit" else "navdp"
    native["fault"] = prefix + "_set_chunk_envelope"
    with pytest.raises(KeyboardInterrupt, match="original envelope failure"):
        _build(kind)
    resource, = native["resources"]
    assert resource.handle is None and not resource.keepalive and not resource.finalizer.alive
    assert native["events"].count(("destroy", 41)) == 1
    assert _execution._UNSAFE_PROCESS_REASON is None


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fault", (None, "clear", "invariant", "destroy"))
def test_actual_build_retirement_is_graph_first_once_and_terminal_on_failure(native, kind, fault):
    owner = _build(kind)
    resource = owner._native_resource
    graph, = resource.graphs
    with graph.capture(_sig(kind)):
        pass
    graph.freeze()
    native["events"].clear()
    clear = graph._impl.clear
    def clear_graph():
        native["events"].append("clear")
        if fault == "clear":
            raise ReferenceError("graph clear failure")
        clear()
    graph._impl.clear = clear_graph
    graph._impl.cache_invariant_ok = lambda: fault != "invariant"
    if fault == "destroy":
        native["destroy_error"] = KeyboardInterrupt("native destroy failure")
    if fault is None:
        owner.destroy()
        owner.destroy()
        assert native["events"] == ["clear", ("destroy", 41)]
        assert resource.handle is None and not resource.keepalive and not resource.finalizer.alive
        if kind == "nextdit":
            assert owner._owner._rpu_nextdit_runtime is None
            for method, args in ((owner.forward, (None, None, None)),
                    (owner.forward_actions, (None, None, None)),
                    (owner.denoise_actions, (None, None, None, None)),
                    (owner.prepare_steps, (None, None))):
                with pytest.raises(RuntimeError, match="destroyed"):
                    method(*args)
        else:
            with pytest.raises(RuntimeError, match="closed"):
                owner.predict_noise(None, None, None, None)
    else:
        with pytest.raises((ReferenceError, RuntimeError, KeyboardInterrupt)):
            owner.destroy()
        assert resource.failed and resource.keepalive and owner._handle == 41
        before = list(native["events"])
        with pytest.raises(RuntimeError):
            owner.destroy()
        resource.finalizer()
        gc.collect()
        assert native["events"] == before and _execution._UNSAFE_PROCESS_REASON
        if kind == "nextdit":
            assert owner._owner._rpu_nextdit_runtime is owner


@pytest.mark.parametrize("kind,op", [("nextdit", "internvla_nextdit_set_weights"),
    ("nextdit", "internvla_nextdit_set_fused_qkq_weights"),
    ("nextdit", "internvla_nextdit_set_step_prepare_weights"), ("navdp", "navdp_set_weights")])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_actual_pending_setter_keeps_every_address_and_original_error(native, kind, op, cleanup_fails):
    native["fault"] = op
    if cleanup_fails:
        native["destroy_error"] = ReferenceError("cleanup failure")
    with pytest.raises(KeyboardInterrupt, match="original setter failure") as caught:
        _build(kind)
    resource, = native["resources"]
    assert (resource.handle is not None) is cleanup_fails
    assert bool(resource.keepalive) is cleanup_fails
    assert bool(getattr(caught.value, "__notes__", ())) is cleanup_fails
    assert [event for event in native["events"] if event[0] == "destroy"] == [("destroy", 41)]
    before = list(native["events"])
    resource.finalizer()
    assert native["events"] == before


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("failed", [False, True])
def test_actual_cyclic_gc_retains_failed_weights_without_external_resource_reference(native, kind, failed):
    owner = _build(kind)
    cached = weakref.ref(owner._self_cache if kind == "nextdit" else owner._cache)
    owner.cycle = owner
    reference = weakref.ref(owner)
    if failed:
        native["destroy_error"] = RuntimeError("GC destroy failure")
    native["events"].clear()
    native["resources"].clear()
    del owner
    gc.collect()
    assert reference() is None
    assert native["events"] == [("destroy", 41)]
    assert (cached() is not None) is failed
    assert bool(_native_retirement._FAILED_RETIREMENTS) is failed
    gc.collect()
    assert native["events"] == [("destroy", 41)]


@pytest.mark.parametrize("kind", KINDS)
def test_actual_parent_takeover_rejects_leaf_destroy_and_keeps_sibling_ready(native, kind):
    owner = _build(kind, deferred=True)
    sibling = SimpleNamespace(_graph_cache=_cache())
    sibling._graph_cache._impl.cache_invariant_ok = lambda: True
    with sibling._graph_cache.capture(_sig("sibling")):
        pass
    sibling._graph_cache.freeze()
    component = execution.NEXTDIT_COMPONENT if kind == "nextdit" else execution.NAVDP_COMPONENT
    controller = execution.InternVLAExecutionController(SimpleNamespace(), {
        component: owner, execution.BACKBONE_COMPONENT: sibling})
    native["events"].clear()
    with controller.execute():
        with pytest.raises(RuntimeError, match="through its parent"):
            owner.destroy()
    assert not native["events"] and _execution._UNSAFE_PROCESS_REASON is None
    with pytest.raises(RuntimeError, match="through its parent"):
        owner.destroy()
    assert controller.session.stats()["state"] == "QUIESCENT"
    assert sibling._graph_cache.phase == "READY" and sibling._graph_cache.size() == 1
    with controller.execute():
        with sibling._graph_cache.capture(_sig("sibling")):
            pass
    controller.close()
    assert owner._handle is None


@pytest.mark.parametrize("kind", KINDS)
def test_closed_own_session_with_live_handle_is_not_successful_retirement(native, kind):
    owner = _build(kind)
    if kind == "nextdit":
        _execution.bind_execution_session(owner, {}, entry_point="test-nextdit", graph_mode="COMPOSITE_CHILD")
    owner._execution_session.shutdown()
    native["events"].clear()
    with pytest.raises(RuntimeError, match="still owns native"):
        owner.destroy()
    assert not native["events"] and owner._native_resource.failed
    assert owner._handle == 41 and _execution._UNSAFE_PROCESS_REASON


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_actual_navdp_constructor_parent_binding_failure_has_one_cleanup(native, monkeypatch, cleanup_fails):
    error = ValueError("actual constructor bind failure")
    def fail(*args):
        raise error
    monkeypatch.setattr(_execution.ExecutionSession, "_bind_planner_owner", fail)
    if cleanup_fails:
        native["destroy_error"] = ReferenceError("cleanup failure")
    with pytest.raises(ValueError) as caught:
        _build("navdp")
    assert caught.value is error
    assert [event for event in native["events"] if event[0] == "destroy"] == [("destroy", 41)]
    resource, = native["resources"]
    assert bool(resource.keepalive) is cleanup_fails
    assert bool(getattr(error, "__notes__", ())) is cleanup_fails


def test_actual_navdp_active_glue_setter_failure_retains_addresses_without_clearing_or_destroy(native, monkeypatch):
    owner = _build("navdp")
    native["fault"] = "navdp_set_denoise_glue"
    native["events"].clear()
    monkeypatch.setattr(owner, "_denoise_unroll", True)
    monkeypatch.setattr(owner, "_predict_action_unroll", lambda *args: owner._ensure_glue_installed())
    with pytest.raises(KeyboardInterrupt, match="original setter failure"):
        owner.predict_action(None, None)
    assert native["events"] == [("navdp_set_denoise_glue", 41)]
    assert not getattr(owner, "_glue_installed", False)
    assert owner._native_resource.failed and owner._native_resource.keepalive
    before = list(native["events"])
    for call in (lambda: owner.predict_action(None, None), owner.destroy, owner._ensure_glue_installed):
        with pytest.raises(RuntimeError):
            call()
    owner._handle_finalizer()
    assert native["events"] == before and owner._execution_session.stats()["state"] == "POISONED"


@pytest.mark.parametrize("kind", KINDS)
def test_failed_process_rejects_new_builder_before_loading_weights_or_allocating(native, monkeypatch, kind):
    _execution._mark_execution_process_unsafe("previous native retirement failure")
    for module in (nextdit, navdp):
        monkeypatch.setattr(module, "verify_asset_manifest", lambda *args: pytest.fail("asset load after unsafe"))
    with pytest.raises(RuntimeError, match="unsafe"):
        _build(kind)
    assert not native["events"] and not native["resources"]


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_actual_nextdit_publication_failure_retires_before_removing_owned_marker(native, cleanup_fails):
    error = KeyboardInterrupt("original marker publication failure")
    class Model(SimpleNamespace):
        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if name == "_rpu_nextdit_runtime":
                raise error
    model = Model(**vars(_nextdit_model()))
    if cleanup_fails:
        native["destroy_error"] = ReferenceError("cleanup failure")
    with pytest.raises(KeyboardInterrupt) as caught:
        nextdit.build_nextdit_runtime(model, asset_paths=("unused",))
    assert caught.value is error
    resource, = native["resources"]
    assert (model._rpu_nextdit_runtime is not None) is cleanup_fails
    assert bool(resource.keepalive) is cleanup_fails
    before = list(native["events"])
    resource.finalizer()
    assert native["events"] == before


def test_actual_nextdit_consumed_finalizer_then_explicit_destroy_clears_cpu_marker(native):
    owner = _build("nextdit")
    model = owner._owner
    owner._handle_finalizer()
    assert owner._handle is None and model._rpu_nextdit_runtime is owner
    before = list(native["events"])
    owner.destroy()
    owner.destroy()
    assert model._rpu_nextdit_runtime is None and native["events"] == before
    replacement = nextdit.build_nextdit_runtime(model, asset_paths=("unused",))
    assert model._rpu_nextdit_runtime is replacement and replacement._handle == 42
    replacement.destroy()


@pytest.mark.parametrize("failed", [False, True])
def test_actual_navdp_first_batched_forward_cache_survives_failed_gc(native, monkeypatch, failed):
    owner = _build("navdp")
    owner._denoise_unroll = owner._batch_unroll = True
    original_to = torch.Tensor.to
    def to(value, *args, **kwargs):
        if args and args[0] == "rpu":
            args = ("cpu", *args[1:])
        return original_to(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", to)
    plan = SimpleNamespace(graph_key_words=lambda: (),
        selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=())))
    monkeypatch.setattr(navdp, "_plan_navdp_execution", lambda *args, **kwargs: (64, plan))
    monkeypatch.setattr(navdp, "_publish_navdp_execution_receipt", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.ops.rpu, "navdp_get_resolved_chunk_size", lambda _: 64, raising=False)
    admissions = []
    def admit(handle, rows, steps):
        assert handle == owner._handle and rows == 64 and steps == navdp.DDPM_STEPS
        admissions.append((handle, rows, steps))
    monkeypatch.setattr(torch.ops.rpu, "navdp_validate_execution_admission", admit, raising=False)
    def forward(handle, x0, memory, noise, coeff, k, v, out, descriptor):
        assert any(value is owner._batch_cache for value in owner._native_resource.keepalive)
        out.zero_()
        return out
    monkeypatch.setattr(torch.ops.rpu, "navdp_denoise_loop_forward", forward, raising=False)
    result = owner.predict_action(torch.zeros(1, 1, 384), torch.zeros(1, 32, 384), sample_num=2)
    assert result.shape == (2, 32, 3)
    assert admissions == [(owner._handle, 64, navdp.DDPM_STEPS)]
    # The real first-use setter is installed once; replay does not rebind it.
    owner._ensure_glue_installed()
    assert native["events"].count(("navdp_set_denoise_glue", 41)) == 1
    cached = weakref.ref(owner._batch_cache)
    reference = weakref.ref(owner)
    if failed:
        native["destroy_error"] = RuntimeError("GC native destroy failure")
    native["resources"].clear()
    native["events"].clear()
    del owner
    gc.collect()
    assert reference() is None and (cached() is not None) is failed
    assert native["events"] == [("destroy", 41)]
    gc.collect()
    assert native["events"] == [("destroy", 41)]


@pytest.mark.parametrize("fused_qkq,debug", [(False, False), (False, True), (True, True)])
def test_nextdit_actual_debug_caller_reads_three_slots_and_rejects_mode_drift(native, monkeypatch, fused_qkq, debug):
    """CPU endpoint checks original caller/readback; no native Graph/model claim."""
    from contextlib import nullcontext
    mode = [debug]
    monkeypatch.setattr(torch.rpu, "get_debug_export", lambda: mode[0], raising=False)
    for name in ("FOLD_NORM_MOD", "STEP_PREP_RPU"):
        monkeypatch.setenv("RPU_INTERNVLA_NEXTDIT_" + name, "0")
    monkeypatch.setenv("RPU_INTERNVLA_NEXTDIT_FUSED_QKQ", str(int(fused_qkq)))
    owner = nextdit.build_nextdit_runtime(_nextdit_model(), collect_layer_outputs=True, asset_paths=("unused",))
    try:
        assert owner._debug_export is debug and owner._fused_qkq is fused_qkq
        assert native["debug_selections"][owner._handle] is debug
        state = SimpleNamespace(cross_context_rpu=torch.zeros(1, 260, 384, dtype=torch.float16))
        step = SimpleNamespace(modulation_rpu=torch.zeros(12, 4, 384, dtype=torch.float16),
                               final_scale_rpu=torch.zeros(384, dtype=torch.float16))
        monkeypatch.setattr(owner, "_condition_bindings", lambda *a, **k: [(0, state)])
        monkeypatch.setattr(owner, "_x_batch_to_rpu", lambda x: x.half())
        monkeypatch.setattr(owner, "_step_state", lambda *a: step)
        plan = SimpleNamespace(selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(1, 2, 3))),
                               graph_key_words=lambda: (7, 8, 9))
        monkeypatch.setattr(owner, "_execution_plan", lambda **kw: (32, plan))
        monkeypatch.setattr(owner._graph_cache, "capture", lambda signature: nullcontext())
        monkeypatch.setattr(torch.ops.rpu, "internvla_nextdit_get_resolved_chunk_size", lambda h: 32, raising=False)
        receipts, calls = [], []
        monkeypatch.setattr(nextdit, "publish_execution_receipt", lambda owner, actual, **kw: receipts.append((actual, kw)))
        def forward(handle, x, modulation, scale, raw, action, populate, unroll, prepare,
                    euler, cross, sk, sv, ck, cv, probes, collect, descriptor):
            assert handle == owner._handle and collect and not unroll and not prepare and not euler
            assert probes is owner._layer_outputs_rpu and tuple(descriptor) == (1, 2, 3)
            assert mode[0] is debug and owner._fused_qkq is fused_qkq
            calls.append((handle, action, populate))
            for slot in range(15):
                probes[slot].fill_(slot + len(calls))
            return x.clone()
        monkeypatch.setattr(torch.ops.rpu, "internvla_nextdit_forward", forward, raising=False)
        x, time, condition = torch.ones(1, 32, 384), torch.tensor([900]), torch.ones(1, 260, 768)
        result = owner.forward(x, time, condition)
        assert torch.equal(result, x) and len(calls) == len(receipts) == 1
        assert receipts[0][0] is plan and receipts[0][1]["logical_len"] == 32
        probes = owner.last_debug_outputs
        if debug:
            assert set(probes) == {"internvla_nextdit_L0_q_" + suffix for suffix in ("projected", "normalized", "headmajor")}
            for key, value in probes.items():
                slot = ("projected", "normalized", "headmajor").index(key.removeprefix("internvla_nextdit_L0_q_"))
                assert value.shape == ((6, 32, 64) if slot == 2 else (32, 384))
                assert value.dtype == torch.float32 and torch.all(value == 13 + slot)
        else:
            assert probes == {}
        previous = {key: value.clone() for key, value in probes.items()}
        signature = owner._graph_signature(260)
        owner.forward(x, time, condition)
        assert owner._graph_signature(260) is signature and len(calls) == 2
        for key, value in previous.items():
            assert torch.equal(probes[key], value)
            assert torch.equal(owner.last_debug_outputs[key], value + 1)
        mode[0] = not debug
        before = len(calls), len(receipts), dict(owner._graph_signatures)
        with pytest.raises(RuntimeError, match="debug export is cold"):
            owner.forward(x, time, condition)
        assert (len(calls), len(receipts), owner._graph_signatures) == before
        mode[0] = debug
        owner.forward(x, time, condition)
        assert len(calls) == 3
    finally:
        owner.destroy()
