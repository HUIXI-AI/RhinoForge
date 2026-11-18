"""Host contracts for the exact 35B-A3B shared Vision bridge."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_5 import vision
from rpu_backend.graph import GraphRuntimePolicy


@pytest.fixture(autouse=True)
def _installed_tp8_native_topology(monkeypatch):
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_get_execution_topology_v2",
                        lambda _: (2, 8, 8, 8, 8, 8, 8, 8, 8, 8), raising=False)
    """Keep bridge tests host-only while modelling the installed native owner."""
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_moe_get_execution_topology",
        lambda handle: (8, 8, 8, 8, 8) if handle == 41 else (),
        raising=False,
    )


def _vision_config(**overrides):
    values = dict(
        depth=27,
        num_heads=16,
        hidden_size=1152,
        intermediate_size=4304,
        out_hidden_size=2048,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=2304,
        model_type="qwen3_5_moe",
        hidden_act="gelu_pytorch_tanh",
        deepstack_visual_indexes=[],
        quant_config=None,
        quantization_config=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _visual(config=None):
    return SimpleNamespace(
        config=config or _vision_config(), blocks=[object() for _ in range(27)]
    )


def _moe_owner(policy=None):
    from rpu_backend.adapters import qwen3_5_moe
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource

    class Owner:
        pass

    class State:
        pass

    policy = policy or GraphRuntimePolicy(
        execution_core_count=8, graph_arena_count=4
    )
    visual = _visual()
    language = SimpleNamespace()
    fusion = SimpleNamespace(language_model=language, visual=visual)
    model = Owner()
    model.model = fusion

    close = qwen3_5_moe._close_qwen3_5_moe_model

    session = SimpleNamespace(
        _owner=model, _closed=False, _poisoned=False, generation=3
    )
    model._execution_session = session
    state = State()
    vars(state).update(
        handle=41,
        _execution_session=session,
        _retirement_close=close,
        graph_cache=SimpleNamespace(runtime_policy=policy),
        prefill_graph=SimpleNamespace(runtime_policy=policy),
        vision_graph_policy=policy,
        vision_arena_admitted=True,
        vision_control_options=vision._resolve_vision_install_options(),
        execution_topology=SimpleNamespace(
            num_cores=8,
            attn_tp=8,
            mlp_tp=8,
            routed_tp=8,
            lm_head_tp=8,
            physical_kv_cores=8,
        ),
        execution_generation=3,
        prepared_profile=vision.QWEN3_5_MOE_35B_VISION_PROFILE,
    )
    state.retirement = resource = _InstalledNativeResource(
        state, 41, lambda _handle: None, graphs=(), keepalive=(),
        label="MoE Vision bridge fixture", handle_name="handle",
    )
    state.handle_finalizer = resource.finalizer
    model._qwen35_retirement_children = [
        (language, state, resource, session)
    ]
    resource.take_ownership(model)
    language._rpu_qwen3_5_moe = state
    return model, visual, fusion, state, close


def test_exact_owner8_out2048_profile_preserves_cold_padding_contract():
    visual = _visual()
    vision._validate_qwen3_5_moe_35b_vision_profile(
        visual, visual.config, 8
    )
    assert vision._qwen3_5_vision_padded_head_dim(1152 // 16) == 80
    assert vision._qwen3_5_vision_padded_intermediate_size(4304) == 4352


@pytest.mark.parametrize(
    "config,cores",
    [
        (_vision_config(), 4),
        (_vision_config(out_hidden_size=4096), 8),
        (_vision_config(intermediate_size=4352), 8),
        (_vision_config(model_type="qwen3_5"), 8),
    ],
)
def test_moe_profile_rejects_reduced_owner_and_near_neighbours(config, cores):
    with pytest.raises(ValueError, match="exact TP8 35B-A3B"):
        vision._validate_qwen3_5_moe_35b_vision_profile(
            _visual(config), config, cores
        )


def test_moe_policy_is_explicit_and_not_derived_from_dense_text_policy():
    model, _visual_model, _fusion, state, _close = _moe_owner()
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    options = vision._lazy_vision_graph_options(
        state, prepared_profile=bridge.prepared_profile
    )
    assert options == {
        "_graph_runtime_policy": state.vision_graph_policy,
        "_cold_numeric_opt_in": True,
    }
    assert options["_graph_runtime_policy"] is state.vision_graph_policy

    state.execution_topology = SimpleNamespace(num_cores=4, attn_tp=4)
    with pytest.raises(RuntimeError, match="root8/Vision8 topology"):
        vision.make_qwen3_5_moe_vision_text_bridge(model)


def test_moe_bridge_rejects_policy_detached_from_installed_text_graphs():
    model, _visual_model, _fusion, state, _close = _moe_owner()
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    state.vision_graph_policy = GraphRuntimePolicy(
        execution_core_count=8, graph_arena_count=4
    )
    assert state.vision_graph_policy == state.graph_cache.runtime_policy
    assert state.vision_graph_policy is not state.graph_cache.runtime_policy
    with pytest.raises(RuntimeError, match="Vision8/root8 four-arena policy pair"):
        bridge.set_mrope_position_delta(1)


def test_moe_installer_rejects_an_equivalent_borrowed_policy():
    model, _visual_model, _fusion, state, _close = _moe_owner()
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    borrowed = GraphRuntimePolicy(
        execution_core_count=8, graph_arena_count=4
    )
    assert borrowed == state.vision_graph_policy
    assert borrowed is not state.vision_graph_policy
    with pytest.raises(RuntimeError, match="differs from its cold owner"):
        vision.install_qwen3_5_vision_for_rpu(
            model,
            execution_config={"model": {"num_cores": 8}},
            _graph_runtime_policy=borrowed,
            _cold_numeric_opt_in=True,
            _text_bridge=bridge,
        )


def test_installer_routes_exact_moe_profile_policy_and_parent_close(
    monkeypatch,
):
    model, visual, _fusion, state, close = _moe_owner()
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    captured = {}

    def install(*_args, **kwargs):
        captured["install"] = kwargs
        visual._rpu_vision_retirement_state = object()
        return 77

    def bind(child_state, parent, callback, child):
        captured["bind"] = (child_state, parent, callback, child)

    from rpu_backend.adapters.qwen3_5 import text

    monkeypatch.setattr(
        vision, "_validate_qwen3_5_moe_35b_vision_weights", lambda *_a: None
    )
    monkeypatch.setattr(vision, "_install_qwen3_5_vision_for_rpu_impl", install)
    monkeypatch.setattr(text, "_bind_qwen35_retirement", bind)

    assert vision.install_qwen3_5_vision_for_rpu(
        model,
        execution_config={"model": {"num_cores": 8}},
        _resolved_options=state.vision_control_options,
        _graph_runtime_policy=state.vision_graph_policy,
        _cold_numeric_opt_in=True,
        _text_bridge=bridge,
    ) == 77
    assert (
        captured["install"]["_prepared_profile"]
        == vision.QWEN3_5_MOE_35B_VISION_PROFILE
    )
    assert captured["install"]["_graph_runtime_policy"] is state.vision_graph_policy
    assert captured["bind"] == (
        visual._rpu_vision_retirement_state,
        model,
        close,
        visual,
    )


def test_idempotent_moe_install_rejects_vision_bound_to_another_parent(
    monkeypatch,
):
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource

    model, visual, _fusion, state, close = _moe_owner()
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    visual.merger = object()
    visual._rpu_vision_handle = 55
    visual._rpu_vision_graph_cache = SimpleNamespace(
        runtime_policy=state.vision_graph_policy
    )
    visual._rpu_vision_num_cores = 8
    visual._rpu_vision_linear_acc32 = False
    for name, value in zip(
        (
            "_rpu_vision_chunk_size_cap",
            "_rpu_vision_exact_chunk_size",
            "_rpu_vision_padding_rows",
            "_rpu_vision_padding_budget",
            "_rpu_vision_control_snapshot",
        ),
        state.vision_control_options,
    ):
        setattr(visual, name, value)
    monkeypatch.setattr(
        vision, "_qwen3_5_vision_runtime_complete", lambda *_a, **_kw: True
    )
    borrowed = GraphRuntimePolicy(
        execution_core_count=8, graph_arena_count=4
    )
    visual._rpu_vision_graph_cache.runtime_policy = borrowed
    with pytest.raises(RuntimeError, match="policy is immutable"):
        vision.install_qwen3_5_vision_for_rpu(
            model,
            execution_config={"model": {"num_cores": 8}},
            _resolved_options=state.vision_control_options,
            _graph_runtime_policy=state.vision_graph_policy,
            _cold_numeric_opt_in=True,
            _text_bridge=bridge,
        )
    visual._rpu_vision_graph_cache.runtime_policy = state.vision_graph_policy

    class WrongParent:
        pass

    class VisionState:
        pass

    wrong_parent = WrongParent()
    wrong_session = SimpleNamespace(
        _owner=wrong_parent, _closed=False, _poisoned=False
    )
    vision_state = VisionState()
    vision_state.handle = 55
    vision_state._execution_session = wrong_session
    vision_state._retirement_close = close
    vision_state.retirement = resource = _InstalledNativeResource(
        vision_state,
        55,
        lambda _handle: None,
        graphs=(),
        keepalive=(),
        label="foreign MoE Vision fixture",
        handle_name="handle",
    )
    vision_state.handle_finalizer = resource.finalizer
    wrong_parent._qwen35_retirement_children = [
        (visual, vision_state, resource, wrong_session)
    ]
    resource.take_ownership(wrong_parent)
    visual._rpu_vision_retirement_state = vision_state
    try:
        with pytest.raises(RuntimeError, match="different owner"):
            vision.install_qwen3_5_vision_for_rpu(
                model,
                execution_config={"model": {"num_cores": 8}},
                _resolved_options=state.vision_control_options,
                _graph_runtime_policy=state.vision_graph_policy,
                _cold_numeric_opt_in=True,
                _text_bridge=bridge,
            )
    finally:
        vars(vision_state).pop("_retirement_close", None)
        resource.finalizer.detach()


@pytest.mark.parametrize("certified_g05", [False, True])
def test_moe_profile_cannot_enter_shared_installer_without_bridge(
    monkeypatch, certified_g05
):
    visual = _visual()
    visual.merger = object()
    create = lambda: pytest.fail("bridge rejection must precede native create")
    monkeypatch.setattr(
        torch.ops.rpu, "qwen3_5_vision_create", create, raising=False
    )
    with pytest.raises(ValueError, match="verified text bridge"):
        vision.install_qwen3_5_vision_for_rpu(
            visual, _allow_certified_g05_k1=certified_g05
        )


def test_certified_g05_geometry_keeps_its_existing_shared_install_path(
    monkeypatch,
):
    config = _vision_config(
        depth=24,
        hidden_size=1024,
        intermediate_size=4096,
        out_hidden_size=2048,
    )
    visual = SimpleNamespace(
        config=config,
        blocks=[object() for _ in range(24)],
        merger=object(),
    )
    monkeypatch.setattr(
        vision, "_install_qwen3_5_vision_for_rpu_impl", lambda *_a, **_kw: 23
    )
    assert vision.install_qwen3_5_vision_for_rpu(
        visual, _allow_certified_g05_k1=True
    ) == 23


def test_moe_35b_geometry_cannot_borrow_g05_flag_after_model_type_drift():
    visual = _visual(_vision_config(model_type="qwen3_5"))
    visual.merger = object()
    with pytest.raises(ValueError, match="verified text bridge"):
        vision.install_qwen3_5_vision_for_rpu(
            visual, _allow_certified_g05_k1=True
        )


def test_image_delta_text_reset_and_next_image_restore_use_moe_handle(
    monkeypatch,
):
    model, _visual_model, fusion, _state, _close = _moe_owner()
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    calls = []
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_moe_set_mrope_position_delta",
        lambda handle, delta: calls.append((handle, delta)),
        raising=False,
    )

    fusion.rope_deltas = torch.tensor([[9]])
    vision._publish_qwen3_5_vision_mrope_delta(
        fusion, _text_bridge=bridge
    )
    bridge.set_mrope_position_delta(0)
    fusion.rope_deltas = torch.tensor([[13]])
    vision._publish_qwen3_5_vision_mrope_delta(
        fusion, _text_bridge=bridge
    )
    assert calls == [(41, 9), (41, 0), (41, 13)]


def test_moe_image_without_rope_delta_fails_before_reusing_old_offset(
    monkeypatch,
):
    model, _visual_model, fusion, _state, _close = _moe_owner()
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_moe_set_mrope_position_delta",
        lambda *_args: pytest.fail("missing image delta must not be published"),
        raising=False,
    )
    fusion.rope_deltas = None
    with pytest.raises(RuntimeError, match="produced no M-RoPE delta"):
        vision._publish_qwen3_5_vision_mrope_delta(
            fusion, _text_bridge=bridge
        )


def test_moe_bridge_fails_closed_after_state_replacement(monkeypatch):
    model, _visual_model, _fusion, state, _close = _moe_owner()
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    setter = lambda *_args: pytest.fail("stale handle must not be called")
    monkeypatch.setattr(
        torch.ops.rpu, "qwen3_5_moe_set_mrope_position_delta", setter,
        raising=False,
    )
    model.model.language_model._rpu_qwen3_5_moe = SimpleNamespace(
        handle=state.handle
    )
    with pytest.raises(RuntimeError, match="removed or replaced"):
        bridge.set_mrope_position_delta(1)


def test_moe_bridge_fails_closed_after_top_session_replacement(monkeypatch):
    model, _visual_model, _fusion, _state, _close = _moe_owner()
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_moe_set_mrope_position_delta",
        lambda *_args: pytest.fail("detached Session must not use native state"),
        raising=False,
    )
    model._execution_session = SimpleNamespace(_owner=model, generation=3)
    with pytest.raises(RuntimeError, match="Session owner changed"):
        bridge.set_mrope_position_delta(1)


def test_moe_bridge_fails_closed_after_profile_admission_is_lost(monkeypatch):
    model, _visual_model, _fusion, state, _close = _moe_owner()
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    setter = lambda *_args: pytest.fail("unadmitted handle must not be called")
    monkeypatch.setattr(
        torch.ops.rpu, "qwen3_5_moe_set_mrope_position_delta", setter,
        raising=False,
    )
    state.prepared_profile = None
    with pytest.raises(RuntimeError, match="lacks its prepared profile"):
        bridge.set_mrope_position_delta(1)


def test_moe_bridge_fails_closed_after_execution_generation_drift(monkeypatch):
    model, _visual_model, _fusion, state, _close = _moe_owner()
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    setter = lambda *_args: pytest.fail("stale generation must not be called")
    monkeypatch.setattr(
        torch.ops.rpu, "qwen3_5_moe_set_mrope_position_delta", setter,
        raising=False,
    )
    state.execution_generation += 1
    with pytest.raises(RuntimeError, match="execution generation changed"):
        bridge.set_mrope_position_delta(1)


def test_moe_bridge_rejects_wrong_initial_close_and_child_identity():
    model, _visual_model, _fusion, state, _close = _moe_owner()
    state._retirement_close = lambda _model: None
    with pytest.raises(RuntimeError, match="close callback changed"):
        vision.make_qwen3_5_moe_vision_text_bridge(model)

    model, _visual_model, _fusion, state, _close = _moe_owner()
    _child, recorded_state, resource, session = (
        model._qwen35_retirement_children[0]
    )
    model._qwen35_retirement_children[0] = (
        object(), recorded_state, resource, session
    )
    with pytest.raises(RuntimeError, match="retirement child identity changed"):
        vision.make_qwen3_5_moe_vision_text_bridge(model)


def test_moe_bridge_fails_closed_after_retirement(monkeypatch):
    model, _visual_model, _fusion, state, _close = _moe_owner()
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    setter = lambda *_args: pytest.fail("retired handle must not be called")
    monkeypatch.setattr(
        torch.ops.rpu, "qwen3_5_moe_set_mrope_position_delta", setter,
        raising=False,
    )
    state.retirement._native_destroyed(41)
    with pytest.raises(RuntimeError, match="no live handle"):
        bridge.set_mrope_position_delta(1)


def test_legacy_callback_keyword_cannot_be_silently_swallowed():
    with pytest.raises(TypeError, match="_mrope_setter"):
        vision.fuse_visual_embeds(
            None, None, None, None, None, _mrope_setter=lambda _delta: None
        )


def test_moe_bridge_rejects_a_setter_overriding_subclass():
    model, _visual_model, fusion, state, close = _moe_owner()

    class SilentBridge(vision.Qwen3_5MoeVisionTextBridge):
        def set_mrope_position_delta(self, _delta):
            pass

    bridge = SilentBridge(
        model=model, fusion=fusion, state=state, close_callback=close
    )
    with pytest.raises(TypeError, match="Qwen3_5MoeVisionTextBridge"):
        vision._validate_qwen3_5_moe_vision_text_bridge(bridge, model)


def test_dense_mrope_publish_path_is_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_set_mrope_position_delta",
        lambda handle, delta: calls.append((handle, delta)),
        raising=False,
    )
    fusion = SimpleNamespace(
        language_model=SimpleNamespace(
            _rpu_qwen3_5=SimpleNamespace(handle=17)
        ),
        rope_deltas=torch.tensor([[6]]),
    )
    vision._publish_qwen3_5_vision_mrope_delta(fusion)
    assert calls == [(17, 6)]


def test_dense_lazy_policy_behavior_is_unchanged():
    assert vision._lazy_vision_graph_options(SimpleNamespace()) == {}
    policy = GraphRuntimePolicy(execution_core_count=8, graph_arena_count=4)
    state = SimpleNamespace(
        vision_arena_admitted=True,
        graph_cache=SimpleNamespace(runtime_policy=policy),
        execution_topology=SimpleNamespace(num_cores=8, attn_tp=8),
    )
    options = vision._lazy_vision_graph_options(state)
    assert options["_graph_runtime_policy"] is policy
    assert options["_cold_numeric_opt_in"] is True


@pytest.mark.parametrize("missing", ["execution_generation", "prepared_profile"])
def test_moe_idempotent_to_rpu_rejects_incomplete_bridge_state(missing):
    from types import MethodType

    from rpu_backend.adapters import qwen3_5_moe as moe
    from rpu_backend.api.errors import RPUBackendError

    model, _visual_model, _fusion, state, _close = _moe_owner()
    for name in moe._QWEN3_5_MOE_TEXT_STATE_ATTRS:
        if not hasattr(state, name):
            setattr(state, name, None)
    model._rpu_swizzle_started = True
    model._rpu_swizzled = True
    model._lm_head_w_rpu_keepalive = object()
    model.forward = MethodType(moe._rpu_qwen3_5_moe_forward, model)

    adapter = object.__new__(moe.Qwen3_5MoeAdapter)
    adapter.model = model
    adapter._handle = None
    adapter._graph_cache = None
    adapter._rpu_is_ready = False
    assert adapter.to_rpu() is model
    assert adapter._handle == state.handle

    delattr(state, missing)
    with pytest.raises(RPUBackendError, match="without complete ownership"):
        adapter.to_rpu()
    assert adapter._handle is None
    assert adapter._graph_cache is None
    assert adapter._rpu_is_ready is False


@pytest.mark.parametrize(
    "argument",
    [
        "execution_config",
        "_resolved_options",
        "_text_bridge",
        "_graph_runtime_policy",
        "_cold_numeric_opt_in",
    ],
)
def test_moe_vision_wrapper_rejects_owned_cold_arguments(argument):
    from rpu_backend.adapters.qwen3_5_moe import vision as moe_vision

    model, _visual_model, _fusion, _state, _close = _moe_owner()
    with pytest.raises(TypeError, match="owns its cold install controls"):
        moe_vision.install_qwen3_5_moe_vision_for_rpu(
            model, **{argument: object()}
        )


def test_moe_fuse_defers_install_until_shared_validation(monkeypatch):
    from rpu_backend.adapters.qwen3_5_moe import vision as moe_vision

    model, _visual_model, _fusion, state, _close = _moe_owner()
    captured = {}

    def shared_fuse(*args, **kwargs):
        captured["args"] = args
        captured["bridge"] = kwargs["_text_bridge"]
        return "fused"

    monkeypatch.setattr(
        moe_vision, "_fuse_shared_visual_embeds", shared_fuse
    )
    monkeypatch.setattr(
        moe_vision,
        "install_qwen3_5_moe_vision_for_rpu",
        lambda *_a, **_kw: pytest.fail("wrapper must not pre-install Vision"),
    )
    assert moe_vision.fuse_visual_embeds(
        model, object(), object(), object(), object()
    ) == "fused"
    assert captured["args"][0] is model
    assert captured["bridge"].state is state


@pytest.mark.parametrize("text_started", [False, True])
@pytest.mark.parametrize("exact", [False, True])
def test_moe_image_prefill_replans_after_both_persistent_floors(
    monkeypatch, text_started, exact,
):
    from rpu_backend.adapters.qwen3_5_moe import text
    from rpu_backend.graph import GraphCache
    from rpu_backend.runtime.execution_planner import PlannerRejectError
    from test_execution_planner import _native_candidate_v2

    model, visual, fusion, state, _ = _moe_owner()
    model.config = SimpleNamespace(image_token_id=99)
    policy = state.vision_graph_policy
    events = []
    state.graph_cache = GraphCache(runtime_policy=policy)
    state.prefill_graph = SimpleNamespace(
        runtime_policy=policy, invalidate=lambda: events.append("clear-text"))
    state.retirement.graphs = (state.graph_cache,)
    state.retirement.raw_graphs = (state.prefill_graph,)
    vars(state).update(
        max_seq_len=8192, chunk_envelope=(8192, 512), padding_budget=0,
        padding_rows="auto", exact_chunk_size=256 if exact else None,
        has_dispatched=text_started,
    )
    cache = SimpleNamespace(
        position=0, max_seq_len=8192, allocated_max_seq_len=8192,
        execution_topology=(8, 8, 8, 8, 8),
        moe_execution_topology=(2, 8, 8, 8, 8, 8, 8, 8, 8, 8))
    ids = torch.ones(1, 512, dtype=torch.long)
    ids[:, :196] = 99
    pixels, grid = torch.zeros(784, 1536), torch.tensor([[1, 28, 28]])
    reserved = {"vision": False, "text": text_started}

    def install(owner, native_state, **kwargs):
        assert owner is model and native_state is state
        assert kwargs["prepared_profile"] == state.prepared_profile
        assert kwargs["_text_bridge"].state is state
        events.append("install")
        visual._rpu_vision_handle = 77
        return 77

    def reserve_vision(handle, length):
        assert (handle, length) == (77, 784)
        reserved["vision"] = True
        events.append("vision-fixed")

    def reserve_text(handle, length):
        assert (handle, length) == (41, 512)
        reserved["text"] = True
        events.append("text-fixed")

    def domain(handle, execution, logical):
        assert (handle, execution, logical) == (41, 512, 512)
        available = 7161344 if reserved["vision"] else 7600000
        rows = [_native_candidate_v2(execution, (chunk,) * 3,
                descriptor_version=3, graph_lifecycle=2)
                for chunk, cost in ((128, 6291456), (256, 7364608))
                if cost <= available]
        return (1, len(rows), *(v for row in rows for v in (len(row), *row)))

    monkeypatch.setattr(vision, "_install_lazy_qwen3_5_vision", install)
    for name, fn in {
        "qwen3_5_vision_get_resolved_chunk_size": lambda _: 0,
        "qwen3_5_vision_prepare_persistent_spm": reserve_vision,
        "qwen3_5_moe_prepare_persistent_spm": reserve_text,
        "qwen3_5_moe_planner_cache_identity": lambda handle: (handle, 1),
        "qwen3_5_moe_resolve_prefill_stage_domain": domain,
    }.items():
        monkeypatch.setattr(torch.ops.rpu, name, fn, raising=False)
    early = text._preflight_qwen3_5_moe_text(
        fusion.language_model, cache, 512, multimodal_prefill=True)
    assert early[1] == 256
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    assert vision._prepare_qwen3_5_multimodal_spm(
        model, ids, pixels, grid, early[0], _text_bridge=bridge)
    assert events == ["install", "clear-text", "vision-fixed"] + (
        [] if text_started else ["text-fixed"])
    assert all(reserved.values())
    if exact:
        with pytest.raises(PlannerRejectError):
            text._preflight_qwen3_5_moe_text(
                fusion.language_model, cache, 512, multimodal_prefill=True)
    else:
        final = text._preflight_qwen3_5_moe_text(
            fusion.language_model, cache, 512, multimodal_prefill=True)
        assert final[1] == 128 and final[2] is not early[2]
    assert not vision._prepare_qwen3_5_multimodal_spm(
        model, ids, pixels, grid, early[0], _text_bridge=bridge)
    assert state.multimodal_spm_identity == (41, 77)
