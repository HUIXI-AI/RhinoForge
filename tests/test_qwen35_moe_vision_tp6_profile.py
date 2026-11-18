"""Pure-host contracts for the fixed mixed-TP6 35B-A3B Vision draft."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from rpu_backend.graph import GraphRuntimePolicy


DRAFT = Path(__file__).resolve().parents[1]
SHARED_VISION = (
    DRAFT / "python/rpu_backend/adapters/qwen3_5/vision.py"
)
MOE_VISION = (
    DRAFT / "python/rpu_backend/adapters/qwen3_5_moe/vision.py"
)
MOE_CORES = DRAFT / "python/rpu_backend/adapters/qwen3_5_moe/cores.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vision():
    # Exercise the shared Vision draft against the root-owned TP6 cores draft,
    # while leaving the frozen SOURCE module restored for the surrounding run.
    import rpu_backend.adapters.qwen3_5_moe.cores  # noqa: F401

    key = "rpu_backend.adapters.qwen3_5_moe.cores"
    original = sys.modules[key]
    _load_module(key, MOE_CORES)
    try:
        yield _load_module("qwen35_tp6_draft_shared_vision", SHARED_VISION)
    finally:
        sys.modules[key] = original


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


def _owner(vision, *, profile=None, topology=None, root_policy=None,
           vision_policy=None):
    from rpu_backend.adapters import qwen3_5_moe
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource

    class Owner:
        pass

    class State:
        pass

    profile = profile or vision.QWEN3_5_MOE_35B_TP6_VISION_PROFILE
    topology = topology or (6, 4, 6, 6, 8)
    root_policy = root_policy or GraphRuntimePolicy(
        execution_core_count=topology[0], graph_arena_count=4
    )
    vision_policy = vision_policy or replace(
        root_policy,
        execution_core_count=vision._qwen3_5_moe_vision_profile_cores(profile),
    )
    visual = _visual()
    language = SimpleNamespace()
    fusion = SimpleNamespace(language_model=language, visual=visual)
    model = Owner()
    model.model = fusion
    model._rpu_execution = {"model": {"num_cores": topology[0]}}

    close = qwen3_5_moe._close_qwen3_5_moe_model
    session = SimpleNamespace(
        _owner=model, _closed=False, _poisoned=False, generation=11
    )
    model._execution_session = session
    state = State()
    vars(state).update(
        handle=61,
        _execution_session=session,
        _retirement_close=close,
        graph_cache=SimpleNamespace(runtime_policy=root_policy),
        prefill_graph=SimpleNamespace(runtime_policy=root_policy),
        vision_graph_policy=vision_policy,
        vision_arena_admitted=True,
        vision_control_options=vision._resolve_vision_install_options(),
        execution_topology=SimpleNamespace(
            num_cores=topology[0],
            attn_tp=topology[1],
            mlp_tp=topology[2],
            routed_tp=topology[2],
            lm_head_tp=topology[3],
            physical_kv_cores=topology[4],
        ),
        execution_generation=11,
        prepared_profile=profile,
    )
    state.retirement = resource = _InstalledNativeResource(
        state,
        61,
        lambda _handle: None,
        graphs=(),
        keepalive=(),
        label="MoE mixed-TP6 Vision bridge draft fixture",
        handle_name="handle",
    )
    state.handle_finalizer = resource.finalizer
    model._qwen35_retirement_children = [
        (language, state, resource, session)
    ]
    resource.take_ownership(model)
    language._rpu_qwen3_5_moe = state
    return model, visual, state, close


def _install_native_topology(monkeypatch, topology, *, full_topology=None):
    legacy_calls = []
    full_calls = []

    if full_topology is None:
        root, attention, mlp, head, physical_kv = topology
        full_topology = (
            2,
            root,
            attention,
            attention,
            attention,
            mlp,
            mlp,
            mlp,
            head,
            physical_kv,
        )

    def get_topology(handle):
        legacy_calls.append(handle)
        return topology

    def get_full_topology(handle):
        full_calls.append(handle)
        return full_topology

    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_moe_get_execution_topology",
        get_topology,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_moe_get_execution_topology_v2",
        get_full_topology,
        raising=False,
    )
    return SimpleNamespace(legacy=legacy_calls, full=full_calls)


def _install_native_vision_topology(monkeypatch, topology=(4, 4, 4, 8)):
    set_calls = []

    def set_topology(handle, num_cores):
        set_calls.append((handle, num_cores))

    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_vision_set_execution_cores",
        set_topology,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_vision_get_execution_topology",
        lambda _handle: topology,
        raising=False,
    )
    return set_calls


def test_profile_layouts_keep_root_and_vision_axes_distinct(vision):
    assert vision._qwen3_5_moe_vision_profile_layout(
        vision.QWEN3_5_MOE_35B_VISION_PROFILE
    ) == ((8, 8, 8, 8, 8), 8)
    assert vision._qwen3_5_moe_vision_profile_layout(
        vision.QWEN3_5_MOE_35B_TP4_VISION_PROFILE
    ) == ((4, 4, 4, 4, 8), 4)
    assert vision._qwen3_5_moe_vision_profile_layout(
        vision.QWEN3_5_MOE_35B_TP6_VISION_PROFILE
    ) == ((6, 4, 6, 6, 8), 4)
    assert vision._qwen3_5_moe_vision_profile_root_cores(
        vision.QWEN3_5_MOE_35B_TP6_VISION_PROFILE
    ) == 6
    assert vision._qwen3_5_moe_vision_profile_cores(
        vision.QWEN3_5_MOE_35B_TP6_VISION_PROFILE
    ) == 4
    with pytest.raises(ValueError, match="unknown .* prepared profile"):
        vision._qwen3_5_moe_vision_profile_layout("uniform_tp6")


def test_tp6_bridge_attests_text6_vision4_and_policy_pair(
    monkeypatch, vision
):
    calls = _install_native_topology(monkeypatch, (6, 4, 6, 6, 8))
    model, _visual_model, state, _close = _owner(vision)

    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    assert bridge.prepared_profile == (
        vision.QWEN3_5_MOE_35B_TP6_VISION_PROFILE
    )
    assert calls.legacy == [61]
    assert calls.full == [61]
    assert state.graph_cache.runtime_policy.execution_core_count == 6
    assert state.prefill_graph.runtime_policy is state.graph_cache.runtime_policy
    assert state.vision_graph_policy.execution_core_count == 4
    assert state.vision_graph_policy is not state.graph_cache.runtime_policy
    assert state.graph_cache.runtime_policy.graph_arena_count == 4
    assert state.vision_graph_policy.graph_arena_count == 4

    options = vision._lazy_vision_graph_options(
        state, prepared_profile=bridge.prepared_profile
    )
    assert options == {
        "_graph_runtime_policy": state.vision_graph_policy,
        "_cold_numeric_opt_in": True,
    }
    assert calls.legacy == [61, 61]
    assert calls.full == [61, 61]


@pytest.mark.parametrize(
    "topology",
    [
        (6, 6, 6, 6, 8),
        (4, 4, 4, 4, 8),
        (6, 4, 4, 4, 8),
        (6, 4, 6, 4, 8),
        (6, 4, 6, 6, 6),
    ],
)
def test_tp6_bridge_rejects_uniform_or_partial_python_topology(
    monkeypatch, vision, topology
):
    _install_native_topology(monkeypatch, (6, 4, 6, 6, 8))
    model, _visual_model, _state, _close = _owner(
        vision, topology=topology,
        root_policy=GraphRuntimePolicy(
            execution_core_count=6, graph_arena_count=4
        ),
    )
    with pytest.raises(RuntimeError, match="root6/Vision4 topology"):
        vision.make_qwen3_5_moe_vision_text_bridge(model)


def test_tp6_bridge_rejects_uniform_native_topology(monkeypatch, vision):
    _install_native_topology(monkeypatch, (6, 6, 6, 6, 8))
    model, _visual_model, _state, _close = _owner(vision)
    with pytest.raises(RuntimeError, match="native/Python topology mismatch"):
        vision.make_qwen3_5_moe_vision_text_bridge(model)


@pytest.mark.parametrize(
    "full_topology",
    [
        (2, 6, 4, 6, 4, 6, 6, 6, 6, 8),  # GDN follows root incorrectly.
        (2, 6, 4, 4, 6, 6, 6, 6, 6, 8),  # Router follows root incorrectly.
        (2, 6, 4, 4, 4, 4, 6, 6, 6, 8),  # Routing owner is not route6.
        (2, 6, 4, 4, 4, 6, 4, 6, 6, 8),  # Routed experts are not route6.
        (2, 6, 4, 4, 4, 6, 6, 4, 6, 8),  # Shared expert is not route6.
        (2, 6, 4, 4, 4, 6, 6, 6, 4, 8),  # Head is not head6.
    ],
)
def test_tp6_bridge_rejects_component_owner_mismatch(
    monkeypatch, vision, full_topology
):
    _install_native_topology(
        monkeypatch,
        (6, 4, 6, 6, 8),
        full_topology=full_topology,
    )
    model, _visual_model, _state, _close = _owner(vision)
    with pytest.raises(RuntimeError, match="component topology mismatch"):
        vision.make_qwen3_5_moe_vision_text_bridge(model)


@pytest.mark.parametrize(
    "root_cores,vision_cores,arenas,error",
    [
        (4, 4, 4, "root6 four-arena text policy"),
        (6, 6, 4, "Vision4/root6 four-arena policy pair"),
        (6, 4, 3, "root6 four-arena text policy"),
    ],
)
def test_tp6_policy_pair_is_exact(
    monkeypatch, vision, root_cores, vision_cores, arenas, error
):
    _install_native_topology(monkeypatch, (6, 4, 6, 6, 8))
    root_policy = GraphRuntimePolicy(
        execution_core_count=root_cores, graph_arena_count=arenas
    )
    vision_policy = replace(
        root_policy, execution_core_count=vision_cores
    )
    model, _visual_model, _state, _close = _owner(
        vision,
        root_policy=root_policy,
        vision_policy=vision_policy,
    )
    with pytest.raises(RuntimeError, match=error):
        vision.make_qwen3_5_moe_vision_text_bridge(model)


def test_tp6_bridge_rejects_vision_policy_used_as_text_policy(
    monkeypatch, vision
):
    _install_native_topology(monkeypatch, (6, 4, 6, 6, 8))
    model, _visual_model, state, _close = _owner(vision)
    state.prefill_graph.runtime_policy = state.vision_graph_policy
    with pytest.raises(RuntimeError, match="root6 four-arena text policy"):
        vision.make_qwen3_5_moe_vision_text_bridge(model)


def test_exact_tp6_profile_runs_the_existing_vision4_geometry(vision):
    visual = _visual()
    vision._validate_qwen3_5_moe_35b_vision_profile(
        visual,
        visual.config,
        4,
        vision.QWEN3_5_MOE_35B_TP6_VISION_PROFILE,
    )
    assert vision._qwen3_5_vision_padded_head_dim(1152 // 16) == 80
    assert vision._qwen3_5_vision_padded_intermediate_size(4304) == 4352
    with pytest.raises(ValueError, match="exact TP4 35B-A3B"):
        vision._validate_qwen3_5_moe_35b_vision_profile(
            visual,
            visual.config,
            6,
            vision.QWEN3_5_MOE_35B_TP6_VISION_PROFILE,
        )


def test_shared_installer_routes_root6_config_to_vision4(
    monkeypatch, vision
):
    from rpu_backend.adapters.qwen3_5 import cores, text

    _install_native_topology(monkeypatch, (6, 4, 6, 6, 8))
    vision_topology_calls = _install_native_vision_topology(monkeypatch)
    model, visual, state, close = _owner(vision)
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    captured = {}

    monkeypatch.setattr(
        cores,
        "validate_vision_core_profile",
        lambda *_args: pytest.fail(
            "MoE prepared profile must bypass dense admission"
        ),
    )
    monkeypatch.setattr(
        vision, "_validate_qwen3_5_moe_35b_vision_weights", lambda *_args: None
    )

    def install(*_args, **kwargs):
        captured["install"] = kwargs
        visual._rpu_vision_retirement_state = object()
        return 97

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
    ) == 97
    assert model._rpu_execution == {"model": {"num_cores": 6}}
    assert captured["install"]["num_cores"] == 4
    assert captured["install"]["_graph_runtime_policy"] is (
        state.vision_graph_policy
    )
    assert captured["install"]["_prepared_profile"] == (
        vision.QWEN3_5_MOE_35B_TP6_VISION_PROFILE
    )
    # The delegate owns handle creation and native Vision topology mutation;
    # this host-only seam proves the shared bridge selected Vision4 without
    # pretending that the mocked delegate installed a real native handle.
    assert vision_topology_calls == []
    assert captured["bind"] == (
        visual._rpu_vision_retirement_state,
        model,
        close,
        visual,
    )


def test_shared_installer_rejects_root_config_drift_before_delegate(
    monkeypatch, vision
):
    _install_native_topology(monkeypatch, (6, 4, 6, 6, 8))
    model, _visual_model, state, _close = _owner(vision)
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    delegated = []
    monkeypatch.setattr(
        vision,
        "_install_qwen3_5_vision_for_rpu_impl",
        lambda *_args, **_kwargs: delegated.append(True),
    )
    with pytest.raises(ValueError, match="prepared root6 profile"):
        vision.install_qwen3_5_vision_for_rpu(
            model,
            execution_config={"model": {"num_cores": 4}},
            _resolved_options=state.vision_control_options,
            _graph_runtime_policy=state.vision_graph_policy,
            _cold_numeric_opt_in=True,
            _text_bridge=bridge,
        )
    assert delegated == []


@pytest.mark.parametrize(
    "profile,topology,cores",
    [
        ("QWEN3_5_MOE_35B_TP4_VISION_PROFILE", (4, 4, 4, 4, 8), 4),
        ("QWEN3_5_MOE_35B_VISION_PROFILE", (8, 8, 8, 8, 8), 8),
    ],
)
def test_tp4_and_tp8_keep_one_policy_identity(
    monkeypatch, vision, profile, topology, cores
):
    _install_native_topology(monkeypatch, topology)
    policy = GraphRuntimePolicy(
        execution_core_count=cores, graph_arena_count=4
    )
    model, _visual_model, state, _close = _owner(
        vision,
        profile=getattr(vision, profile),
        topology=topology,
        root_policy=policy,
        vision_policy=policy,
    )
    bridge = vision.make_qwen3_5_moe_vision_text_bridge(model)
    assert vision._lazy_vision_graph_options(
        state, prepared_profile=bridge.prepared_profile
    )["_graph_runtime_policy"] is policy


def test_moe_wrapper_forwards_root6_and_vision4_controls(
    monkeypatch, vision
):
    _install_native_topology(monkeypatch, (6, 4, 6, 6, 8))
    model, _visual_model, state, _close = _owner(vision)
    key = "rpu_backend.adapters.qwen3_5.vision"
    original = sys.modules.get(key)
    sys.modules[key] = vision
    try:
        wrapper = _load_module("qwen35_tp6_draft_moe_vision", MOE_VISION)
    finally:
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original

    captured = {}

    def install(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return 101

    monkeypatch.setattr(wrapper, "_install_shared_vision_for_rpu", install)
    assert wrapper.install_qwen3_5_moe_vision_for_rpu(model) == 101
    assert captured["args"] == (model,)
    assert captured["kwargs"]["execution_config"] == {
        "model": {"num_cores": 6}
    }
    assert captured["kwargs"]["_graph_runtime_policy"] is (
        state.vision_graph_policy
    )
    assert captured["kwargs"]["_graph_runtime_policy"].execution_core_count == 4
    assert captured["kwargs"]["_text_bridge"].prepared_profile == (
        vision.QWEN3_5_MOE_35B_TP6_VISION_PROFILE
    )
