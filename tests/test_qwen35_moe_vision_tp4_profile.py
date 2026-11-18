"""Pure-host contracts for the fixed TP4 35B-A3B Vision profile."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_5 import vision
from rpu_backend.graph import GraphRuntimePolicy


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
        config=config or _vision_config(),
        blocks=[object() for _ in range(27)],
        merger=object(),
    )


def _tp4_owner(policy=None):
    from rpu_backend.adapters import qwen3_5_moe
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource

    class Owner:
        pass

    class State:
        pass

    policy = policy or GraphRuntimePolicy(
        execution_core_count=4, graph_arena_count=4
    )
    visual = _visual()
    language = SimpleNamespace()
    fusion = SimpleNamespace(language_model=language, visual=visual)
    model = Owner()
    model.model = fusion
    model._rpu_execution = {"model": {"num_cores": 4}}

    close = qwen3_5_moe._close_qwen3_5_moe_model
    session = SimpleNamespace(
        _owner=model, _closed=False, _poisoned=False, generation=7
    )
    model._execution_session = session
    state = State()
    vars(state).update(
        handle=43,
        _execution_session=session,
        _retirement_close=close,
        graph_cache=SimpleNamespace(runtime_policy=policy),
        prefill_graph=SimpleNamespace(runtime_policy=policy),
        vision_graph_policy=policy,
        vision_arena_admitted=True,
        vision_control_options=vision._resolve_vision_install_options(),
        execution_topology=SimpleNamespace(
            num_cores=4,
            attn_tp=4,
            mlp_tp=4,
            routed_tp=4,
            lm_head_tp=4,
            physical_kv_cores=8,
        ),
        execution_generation=7,
        prepared_profile=vision.QWEN3_5_MOE_35B_TP4_VISION_PROFILE,
    )
    state.retirement = resource = _InstalledNativeResource(
        state,
        43,
        lambda _handle: None,
        graphs=(),
        keepalive=(),
        label="MoE TP4 Vision bridge fixture",
        handle_name="handle",
    )
    state.handle_finalizer = resource.finalizer
    model._qwen35_retirement_children = [
        (language, state, resource, session)
    ]
    resource.take_ownership(model)
    language._rpu_qwen3_5_moe = state
    return model, visual, state, close


@pytest.fixture(autouse=True)
def _installed_tp4_native_topology(monkeypatch):
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_get_execution_topology_v2",
                        lambda _: (2, 4, 4, 4, 4, 4, 4, 4, 4, 8), raising=False)
    calls = []

    def get_topology(handle):
        calls.append(handle)
        return (4, 4, 4, 4, 8)

    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_moe_get_execution_topology",
        get_topology,
        raising=False,
    )
    return calls


def test_tp4_and_tp8_prepared_profiles_are_distinct_and_fixed():
    assert (
        vision.QWEN3_5_MOE_35B_TP4_VISION_PROFILE
        == "qwen3_5_moe_35b_multimodal_tp4"
    )
    assert (
        vision._qwen3_5_moe_vision_profile_cores(
            vision.QWEN3_5_MOE_35B_TP4_VISION_PROFILE
        )
        == 4
    )
    assert (
        vision._qwen3_5_moe_vision_profile_cores(
            vision.QWEN3_5_MOE_35B_VISION_PROFILE
        )
        == 8
    )
    with pytest.raises(ValueError, match="unknown .* prepared profile"):
        vision._qwen3_5_moe_vision_profile_cores(
            "qwen3_5_moe_35b_multimodal_tp5"
        )


def test_exact_tp4_profile_preserves_cold_padding_contract():
    visual = _visual()
    vision._validate_qwen3_5_moe_35b_vision_profile(
        visual,
        visual.config,
        4,
        vision.QWEN3_5_MOE_35B_TP4_VISION_PROFILE,
    )
    assert vision._qwen3_5_vision_padded_head_dim(1152 // 16) == 80
    assert vision._qwen3_5_vision_padded_intermediate_size(4304) == 4352


@pytest.mark.parametrize(
    "num_cores,prepared_profile",
    [
        (8, vision.QWEN3_5_MOE_35B_TP4_VISION_PROFILE),
        (4, vision.QWEN3_5_MOE_35B_VISION_PROFILE),
    ],
)
def test_prepared_profile_rejects_the_other_owner_count(
    num_cores, prepared_profile
):
    visual = _visual()
    expected = 4 if "tp4" in prepared_profile else 8
    with pytest.raises(ValueError, match=rf"exact TP{expected} 35B-A3B"):
        vision._validate_qwen3_5_moe_35b_vision_profile(
            visual, visual.config, num_cores, prepared_profile
        )


def test_tp4_bridge_attests_python_and_native_topology(
    _installed_tp4_native_topology,
):
    model, _visual_model, state, _close = _tp4_owner()
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    assert bridge.prepared_profile == vision.QWEN3_5_MOE_35B_TP4_VISION_PROFILE
    assert _installed_tp4_native_topology == [43]

    options = vision._lazy_vision_graph_options(
        state, prepared_profile=bridge.prepared_profile
    )
    assert options == {
        "_graph_runtime_policy": state.vision_graph_policy,
        "_cold_numeric_opt_in": True,
    }
    assert _installed_tp4_native_topology == [43, 43]


def test_tp4_bridge_rejects_native_topology_mismatch(monkeypatch):
    model, _visual_model, _state, _close = _tp4_owner()
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_moe_get_execution_topology",
        lambda _handle: (8, 8, 8, 8, 8),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="native/Python topology mismatch"):
        vision.make_qwen3_5_moe_vision_text_bridge(model)


def test_tp4_bridge_requires_native_topology_api(monkeypatch):
    model, _visual_model, _state, _close = _tp4_owner()
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_moe_get_execution_topology",
        None,
        raising=False,
    )
    with pytest.raises(RuntimeError, match="lacks .*get_execution_topology"):
        vision.make_qwen3_5_moe_vision_text_bridge(model)


def test_tp4_bridge_rejects_an_eight_core_vision_policy():
    policy = GraphRuntimePolicy(execution_core_count=8, graph_arena_count=4)
    model, _visual_model, _state, _close = _tp4_owner(policy)
    with pytest.raises(RuntimeError, match="prepared root4 four-arena text policy"):
        vision.make_qwen3_5_moe_vision_text_bridge(model)


def test_shared_installer_routes_tp4_profile_without_dense_admission(
    monkeypatch,
):
    from rpu_backend.adapters.qwen3_5 import cores, text

    model, visual, state, close = _tp4_owner()
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    captured = {}

    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_vision_set_execution_cores",
        lambda *_args: None,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_vision_get_execution_topology",
        lambda _handle: (4, 4, 4, 8),
        raising=False,
    )
    monkeypatch.setattr(
        cores,
        "validate_vision_core_profile",
        lambda *_args: pytest.fail("MoE prepared profile must bypass dense admission"),
    )
    monkeypatch.setattr(
        vision, "_validate_qwen3_5_moe_35b_vision_weights", lambda *_args: None
    )

    def install(*_args, **kwargs):
        captured["install"] = kwargs
        visual._rpu_vision_retirement_state = object()
        return 79

    def bind(child_state, parent, callback, child):
        captured["bind"] = (child_state, parent, callback, child)

    monkeypatch.setattr(vision, "_install_qwen3_5_vision_for_rpu_impl", install)
    monkeypatch.setattr(text, "_bind_qwen35_retirement", bind)

    assert vision.install_qwen3_5_vision_for_rpu(
        model,
        execution_config=model._rpu_execution,
        _resolved_options=state.vision_control_options,
        _graph_runtime_policy=state.vision_graph_policy,
        _cold_numeric_opt_in=True,
        _text_bridge=bridge,
    ) == 79
    assert captured["install"]["num_cores"] == 4
    assert (
        captured["install"]["_prepared_profile"]
        == vision.QWEN3_5_MOE_35B_TP4_VISION_PROFILE
    )
    assert captured["install"]["_graph_runtime_policy"] is state.vision_graph_policy
    assert captured["bind"] == (
        visual._rpu_vision_retirement_state,
        model,
        close,
        visual,
    )


def test_moe_wrapper_forwards_tp4_owner_controls(monkeypatch):
    from rpu_backend.adapters.qwen3_5_moe import vision as moe_vision

    model, _visual_model, state, _close = _tp4_owner()
    captured = {}

    def install(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return 83

    monkeypatch.setattr(moe_vision, "_install_shared_vision_for_rpu", install)

    assert moe_vision.install_qwen3_5_moe_vision_for_rpu(model) == 83
    assert captured["args"] == (model,)
    assert captured["kwargs"]["execution_config"] is model._rpu_execution
    assert captured["kwargs"]["_resolved_options"] is state.vision_control_options
    assert captured["kwargs"]["_graph_runtime_policy"] is state.vision_graph_policy
    assert captured["kwargs"]["_cold_numeric_opt_in"] is True
    assert (
        captured["kwargs"]["_text_bridge"].prepared_profile
        == vision.QWEN3_5_MOE_35B_TP4_VISION_PROFILE
    )
