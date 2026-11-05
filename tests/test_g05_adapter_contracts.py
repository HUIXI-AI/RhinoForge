"""Board-free contracts for the first G0.5 Qwen3.5 VLM port stage."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import types
import weakref
from copy import deepcopy

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_ADAPTER = _ROOT / "python" / "rpu_backend" / "adapters" / "g05" / "runtime.py"
_ACTION_ADAPTER = _ROOT / "python" / "rpu_backend" / "adapters" / "g05" / "action.py"
_RELEASE_REPLAY = (
    _ROOT
    / "python"
    / "rpu_backend"
    / "adapters"
    / "g05"
    / "release_replay.py"
)
_QWEN_TEXT = _ROOT / "python" / "rpu_backend" / "adapters" / "qwen3_5" / "text.py"
_QWEN_VISION = _ROOT / "python" / "rpu_backend" / "adapters" / "qwen3_5" / "vision.py"
_QWEN_CPP = _ROOT / "src" / "fused" / "rpu_qwen3_5_model.cpp"
_QWEN_HEADER = _ROOT / "src" / "fused" / "rpu_qwen3_5_model.h"
_QWEN_VISION_CPP = _ROOT / "src" / "fused" / "rpu_qwen3_5_vision_model.cpp"
_RPU_DISPATCH = _ROOT / "src" / "core" / "rpu_dispatch_registrations.inc"
_RPU_KERNEL_DECLS = _ROOT / "src" / "core" / "rpu_kernel_decls.h"
_RPU_KERNEL_CACHE = _ROOT / "src" / "core" / "rpu_kernel_cache.inc"
_RTC_RELEASE_GATE = (
    _ROOT
    / "python"
    / "rpu_backend"
    / "tests"
    / "model"
    / "g05"
    / "test_g05_rtc_release_e2e.py"
)
_RTC_RELEASE_VISION_GATE = (
    _ROOT
    / "python"
    / "rpu_backend"
    / "tests"
    / "model"
    / "g05"
    / "test_g05_rtc_release_vision_e2e.py"
)
_RTC_RELEASE_VISION_PERF = (
    _ROOT
    / "python"
    / "rpu_backend"
    / "tests"
    / "model"
    / "g05"
    / "test_g05_rtc_release_vision_perf.py"
)


def _load_adapter():
    __import__("rpu_backend.api._execution")
    action_name = "rpu_backend.adapters.g05.action"
    action_stub = types.ModuleType(action_name)
    action_stub._check_g05_action_profile = lambda _config: None
    action_stub._g05_action_runtime_ready = lambda _expert: False
    action_stub._rpu_g05_inference_fm = lambda *_args, **_kwargs: None
    action_stub.patch_g05_action_expert_for_rpu = lambda *_args, **_kwargs: None
    causal_name = "rpu_backend.api.causal_lm"
    causal_stub = types.ModuleType(causal_name)
    causal_stub._claim_live_instance = lambda _owner: None
    modules = __import__("sys").modules
    previous = {
        action_name: modules.get(action_name),
        causal_name: modules.get(causal_name),
    }
    modules[action_name] = action_stub
    modules[causal_name] = causal_stub
    spec = importlib.util.spec_from_file_location("g05_adapter_under_test", _ADAPTER)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                modules.pop(name, None)
            else:
                modules[name] = old_module
    return module


def _load_action_adapter():
    spec = importlib.util.spec_from_file_location(
        "g05_action_adapter_under_test", _ACTION_ADAPTER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_g05_private_same_config_rebuild_retires_each_child_graph(monkeypatch, request):
    from contextlib import nullcontext
    import torch
    from rpu_backend.api import _execution
    from rpu_backend.adapters.qwen3_5 import text, vision
    from test_qwen3_5_install_lifecycle import _native_fixture_state

    module = _load_adapter()
    events, graphs = [], []

    class Graph:
        def __init__(self):
            self.clears = 0
            self.warmups = 0
            graphs.append(self)
        def begin_warmup(self):
            self.warmups += 1
        def clear(self):
            self.clears += 1
        def invalidate(self):
            self.clears += 1
        def cache_invariant_ok(self):
            return True

    text_state = _native_fixture_state(request,
        handle=11, control_snapshot={}, chunk_envelope=(8192, 640),
        graph_cache=Graph(), prefill_graph_cache=Graph(),
        prefill_graph=Graph(),
    )
    action_state = _native_fixture_state(request,
        handle=33, graph_cache=Graph(), action_graph_cache=Graph(),
        action_trace_graph_cache=Graph(),
    )
    class Policy:
        pass
    policy = Policy()
    policy.model = Policy()
    vars(policy.model).update(
        vlm=types.SimpleNamespace(_rpu_g05_max_seq_len=8192, _rpu_qwen3_5=text_state),
        vision_tower=types.SimpleNamespace(
            _rpu_vision_handle=22, _rpu_vision_control_snapshot={},
            _rpu_vision_graph_cache=Graph(), _rpu_vision_debug_graph=Graph(),
        ),
        action_expert=types.SimpleNamespace(_rpu_qwen3_5=action_state),
    )
    monkeypatch.setattr(text, "_resolve_text_install_options", lambda *a, **k: (8192, 640, 64, {}))
    monkeypatch.setattr(vision, "_resolve_vision_install_options", lambda **k: (256, None, "auto", 0, {}))
    monkeypatch.setattr(_execution, "native_execution_reconfigure", lambda *a, **k: nullcontext(71))
    for name in (
        "execution_reconfigure_begin", "execution_reconfigure_commit",
        "execution_reconfigure_abort", "execution_reconfigure_abort_attempt",
        "qwen3_5_stage_prefill_execution_controls",
        "qwen3_5_vision_stage_prefill_execution_controls",
        "qwen3_5_stage_action_chunk_size",
    ):
        monkeypatch.setattr(torch.ops.rpu, name,
                            lambda *args, _name=name: events.append((_name, args)), raising=False)
    resolved = module._resolve_g05_execution({}, max_seq_len=8192, entry_point="test")
    module._install_g05_execution_session(policy, resolved)
    for component, child, stage in (
        ("language_model", policy.model.vlm, "prefill"),
        ("vision_encoder", policy.model.vision_tower, "vision"),
        ("action_expert", policy.model.action_expert, "action"),
    ):
        assert _execution._planner_owner_binding(child, stage) == (policy._execution_session, component)
    policy._execution_session.reconfigure({})
    assert all(graph.clears == 0 for graph in graphs)
    policy._execution_session._reconfigure({}, force_rebuild=True)
    assert all(graph.clears == 1 for graph in graphs)
    assert set(policy._rpu_g05_execution_component_generations.values()) == {1}
    assert text_state.execution_generation == action_state.execution_generation == 1
    assert len(events) == 3


def _load_release_replay():
    spec = importlib.util.spec_from_file_location(
        "g05_release_replay_under_test", _RELEASE_REPLAY
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_rtc_release_gate():
    spec = importlib.util.spec_from_file_location(
        "g05_rtc_release_gate_under_test", _RTC_RELEASE_GATE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_rtc_release_vision_gate():
    spec = importlib.util.spec_from_file_location(
        "g05_rtc_release_vision_gate_under_test", _RTC_RELEASE_VISION_GATE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_rtc_release_vision_perf():
    spec = importlib.util.spec_from_file_location(
        "g05_rtc_release_vision_perf_under_test", _RTC_RELEASE_VISION_PERF
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_fm_helper():
    return types.SimpleNamespace(
        action_causal=False,
        horizon_steps=32,
        action_dim=27,
        num_inference_steps=10,
        time_convention="pi_convention",
        _sample_noise=lambda *_args, **_kwargs: None,
    )


def test_g05_action_cold_prime_publishes_bounded_oneshot_evidence():
    action = _load_action_adapter()

    class Graph:
        def __init__(self):
            self._state = 0

        def begin(self):
            self._state = 1

        def end(self):
            self._state = 0

        def abort(self):
            self._state = 0

        def state(self):
            return self._state

        def graph_size(self):
            return 4

    cache = types.SimpleNamespace(
        size=lambda: 0,
        cache_invariant_ok=lambda: True,
    )
    plan = types.SimpleNamespace(
        as_dict=lambda **_kwargs: {"graph_mode": "BOUNDED_ONESHOT"}
    )
    calls = []
    receipt = action._run_action_cold_prime(
        types.SimpleNamespace(Graph=Graph), cache, plan,
        lambda: calls.append("dispatch"),
    )

    assert calls == ["dispatch"]
    assert receipt["graph_mode"] == "BOUNDED_ONESHOT"
    assert receipt["graph_nodes"] == 4
    assert receipt["post_state"] == "PASSTHROUGH"
    assert receipt["retained_cache_size_before"] == 0
    assert receipt["retained_cache_size_after"] == 0
    assert receipt["retained_cache_invariant"] is True


def test_g05_adaptive_modulation_cache_is_exact(monkeypatch):
    action = _load_action_adapter()
    torch = __import__("torch")
    torch.manual_seed(0)

    def norm():
        return types.SimpleNamespace(dense=torch.nn.Linear(2, 6))

    expert = types.SimpleNamespace(
        config=types.SimpleNamespace(hidden_size=2),
        layers=[types.SimpleNamespace(
            input_layernorm=norm(), post_attention_layernorm=norm()
        )],
        norm=norm(),
        _rpu_g05_adaptive_mod_cache={},
        _rpu_g05_adaptive_mod_stack_cache=None,
    )
    original_to = torch.Tensor.to

    def fake_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs = dict(kwargs, device="cpu")
        return original_to(tensor, *args, **kwargs)

    calls = 0
    original_linear = action.F.linear

    def counting_linear(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_linear(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    monkeypatch.setattr(action.F, "linear", counting_linear)
    cond = torch.tensor([[0.25, -0.5]])
    first = action._adaptive_modulation(expert, cond)
    second = action._adaptive_modulation(expert, cond.clone())

    assert calls == 3
    assert not first.requires_grad
    assert first.data_ptr() == second.data_ptr()
    assert len(expert._rpu_g05_adaptive_mod_cache) == 1

    third = action._adaptive_modulation(expert, cond + 1.0)
    assert calls == 6
    assert len(expert._rpu_g05_adaptive_mod_cache) == 2
    assert not torch.equal(first, third)

    for offset in range(2, 10):
        action._adaptive_modulation(expert, cond + offset)
    action._adaptive_modulation(expert, cond + 10)
    action._adaptive_modulation(expert, cond + 10)
    assert calls == 36
    assert len(expert._rpu_g05_adaptive_mod_cache) == 10

    stack = action._adaptive_modulation_stack(expert, [cond, cond + 1.0])
    cached_stack = action._adaptive_modulation_stack(
        expert, [cond.clone(), cond + 1.0]
    )
    assert calls == 42
    assert tuple(stack.shape) == (2, 3, 6)
    assert stack.data_ptr() == cached_stack.data_ptr()


def test_g05_time_condition_cache_is_exact():
    action = _load_action_adapter()
    torch = __import__("torch")
    weight = torch.nn.Parameter(torch.tensor([[1.0], [2.0]]))
    calls = 0

    def encode_time(timestep):
        nonlocal calls
        calls += 1
        return torch.nn.functional.linear(timestep.reshape(1, 1), weight)

    expert = types.SimpleNamespace(
        _rpu_g05_original_encode_time=encode_time,
        _rpu_g05_time_cond_cache={},
    )
    first = action._rpu_g05_encode_time(expert, torch.tensor([0.25]))
    second = action._rpu_g05_encode_time(expert, torch.tensor([0.25]))
    assert calls == 1
    assert not first.requires_grad
    assert first.data_ptr() == second.data_ptr()

    third = action._rpu_g05_encode_time(expert, torch.tensor([0.5]))
    assert calls == 2
    assert not torch.equal(first, third)

    for value in range(2, 10):
        action._rpu_g05_encode_time(expert, torch.tensor([float(value)]))
    action._rpu_g05_encode_time(expert, torch.tensor([10.0]))
    action._rpu_g05_encode_time(expert, torch.tensor([10.0]))
    assert calls == 12
    assert len(expert._rpu_g05_time_cond_cache) == 10


def test_g05_fixed_action_time_schedule_builds_once(monkeypatch):
    action = _load_action_adapter()
    torch = __import__("torch")
    calls = {"encode": 0, "stack": 0}
    modulation = torch.arange(3)

    def encode_time(timestep):
        calls["encode"] += 1
        return timestep.reshape(1, 1) * 2

    def build_stack(_expert, conditions):
        calls["stack"] += 1
        assert len(conditions) == 10
        return modulation

    expert = types.SimpleNamespace(
        encode_time=encode_time,
        _rpu_g05_fixed_time_schedule=None,
    )
    monkeypatch.setattr(action, "_adaptive_modulation_stack", build_stack)

    first = action._fixed_action_time_schedule(expert, 10, torch.float32)
    second = action._fixed_action_time_schedule(expert, 10, torch.float32)

    assert calls == {"encode": 10, "stack": 1}
    assert first is second
    assert isinstance(first[0], tuple)
    assert first[1] is modulation
    assert len(first[0]) == 10
    assert first[0][0].item() == 2.0
    assert torch.isclose(first[0][-1], torch.tensor([[0.2]]), atol=1e-6).all()
    with pytest.raises(ValueError, match="10 FP32 timesteps"):
        action._fixed_action_time_schedule(expert, 5, torch.float32)
    with pytest.raises(ValueError, match="10 FP32 timesteps"):
        action._fixed_action_time_schedule(expert, 10, torch.float16)


def test_g05_fixed_rtc_schedule_is_compact_exact_and_cached(monkeypatch):
    action = _load_action_adapter()
    torch = __import__("torch")
    original_to = torch.Tensor.to

    def fake_to(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            args = ("cpu", *args[1:])
        if kwargs.get("device") == "rpu":
            kwargs = dict(kwargs, device="cpu")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_to)

    weight = torch.tensor([
        [1.0, 0.0], [0.0, 1.0],
        [2.0, 0.0], [0.0, 2.0],
        [3.0, 0.0], [0.0, 3.0],
    ])

    def norm():
        dense = torch.nn.Linear(2, 6)
        with torch.no_grad():
            dense.weight.copy_(weight)
            dense.bias.zero_()
        return types.SimpleNamespace(dense=dense)

    encode_calls = []

    def encode_time(timestep):
        encode_calls.append(timestep.clone())
        return torch.stack((timestep, timestep + 10.0), dim=-1)

    expert = types.SimpleNamespace(
        config=types.SimpleNamespace(hidden_size=2),
        layers=[
            types.SimpleNamespace(
                input_layernorm=norm(), post_attention_layernorm=norm()
            )
            for _ in range(24)
        ],
        norm=norm(),
        _rpu_g05_original_encode_time=encode_time,
        _rpu_g05_rtc_time_schedules={},
        _rpu_g05_adaptive_mod_stack_cache=None,
    )

    first = action._fixed_action_rtc_time_schedule(
        expert, 10, torch.float32, inference_delay=5
    )
    cached = action._fixed_action_rtc_time_schedule(
        expert, 10, torch.float32, inference_delay=5
    )

    conditions, modulation = first
    assert first is cached
    assert len(encode_calls) == 10
    assert len(conditions) == 10
    assert tuple(conditions[0].shape) == (2, 2)
    assert tuple(modulation.shape) == (10, 49, 2, 6)
    assert modulation.data_ptr() == cached[1].data_ptr()
    assert torch.equal(conditions[0][0], torch.tensor([0.0, 10.0]))
    assert torch.equal(conditions[0][1], torch.tensor([1.0, 11.0]))
    assert torch.allclose(
        conditions[-1][1], torch.tensor([0.1, 10.1]), atol=1e-6
    )
    assert torch.equal(
        modulation[0, 0, 0],
        torch.tensor([1.0, 11.0, 0.0, 20.0, 0.0, 30.0]).half(),
    )
    assert torch.equal(
        modulation[0, -1, 1],
        torch.tensor([2.0, 12.0, 2.0, 22.0, 3.0, 33.0]).half(),
    )

    other_delay = action._fixed_action_rtc_time_schedule(
        expert, 10, torch.float32, inference_delay=7
    )
    assert len(encode_calls) == 20
    assert set(expert._rpu_g05_rtc_time_schedules) == {5, 7}
    assert other_delay is expert._rpu_g05_rtc_time_schedules[7]


def test_g05_action_rope_cache_is_exact(monkeypatch):
    action = _load_action_adapter()
    torch = __import__("torch")
    calls = {"normalize": 0, "build": 0, "set": 0}
    text_module = types.ModuleType("rpu_backend.adapters.qwen3_5.text")

    def normalize(position_ids, *, real_len, padded_len):
        assert real_len == padded_len == 3
        calls["normalize"] += 1
        return position_ids.detach().cpu()[1:, 0, :].clone()

    def build(pid, *_args):
        calls["build"] += 1
        return pid.float(), -pid.float()

    text_module._normalize_prefill_position_ids = normalize
    text_module._build_interleaved_mrope_cos_sin = build
    monkeypatch.setitem(__import__("sys").modules, text_module.__name__, text_module)
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_set_prefill_rope",
        lambda *_args: calls.__setitem__("set", calls["set"] + 1),
        raising=False,
    )
    state = types.SimpleNamespace(
        action_rope_cache=None,
        rotary_dim=64,
        rope_theta=10_000_000.0,
        mrope_section=(11, 11, 10),
        handle=1,
    )
    positions = torch.arange(12).reshape(4, 1, 3)

    action._ensure_action_rope(state, positions, 3)
    action._ensure_action_rope(state, positions, 3)
    assert calls == {"normalize": 1, "build": 1, "set": 1}

    equivalent = positions.clone()
    equivalent[0] += 100
    action._ensure_action_rope(state, equivalent, 3)
    equivalent[0, 0, 0] += 1
    action._ensure_action_rope(state, equivalent, 3)
    assert calls == {"normalize": 3, "build": 1, "set": 1}

    changed = equivalent.clone()
    changed[1] += 7
    action._ensure_action_rope(state, changed, 3)
    assert calls == {"normalize": 4, "build": 2, "set": 2}


def test_g05_action_layout_cache_is_exact():
    action = _load_action_adapter()
    torch = __import__("torch")
    calls = 0

    class Model:
        mask_helper = types.SimpleNamespace(action_position_offset=None)

        def build_action_mask_and_position_ids(
            self,
            attention_mask,
            action_len,
            position_ids_prefix,
            split_index,
            dtype,
            action_causal,
        ):
            nonlocal calls
            calls += 1
            assert split_index == attention_mask.size(1)
            assert not action_causal
            prefix = (attention_mask == 0).to(dtype) * torch.finfo(dtype).min
            prefix = prefix[:, None, None].expand(-1, 1, action_len, -1)
            action_part = torch.zeros(
                1, 1, action_len, action_len, dtype=dtype
            )
            mask = torch.cat((prefix, action_part), dim=-1)
            position = torch.full((3, 1, action_len), calls, dtype=torch.long)
            return mask, position

    model = Model()
    state = types.SimpleNamespace(action_layout_cache=None)
    attention = torch.ones(1, 4, dtype=torch.long)
    positions = torch.arange(12).reshape(3, 1, 4)

    def run(mask=attention, position_ids=positions):
        return action._cached_action_position(
            model,
            state,
            mask,
            position_ids,
            action_len=3,
            prefix_len=4,
            dtype=torch.float32,
            action_causal=False,
        )

    first = run()
    assert run(attention.clone(), positions.clone()).data_ptr() == first.data_ptr()
    assert calls == 1

    changed_positions = positions.clone()
    changed_positions[0, 0, 0] += 1
    assert run(position_ids=changed_positions).data_ptr() != first.data_ptr()
    assert calls == 2
    restored = run()
    assert calls == 3

    invalid_attention = attention.clone()
    invalid_attention[0, 0] = 0
    with pytest.raises(NotImplementedError, match="unpadded"):
        run(mask=invalid_attention)
    assert calls == 4
    assert run().data_ptr() == restored.data_ptr()
    assert calls == 4

    model.mask_helper.action_position_offset = 7
    run()
    run()
    assert calls == 6


def test_g05_action_keep_mask_cache_is_exact(monkeypatch):
    action = _load_action_adapter()
    torch = __import__("torch")
    original_to = torch.Tensor.to

    def fake_to(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            args = ("cpu", *args[1:])
        if kwargs.get("device") == "rpu":
            kwargs = dict(kwargs, device="cpu")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    state = types.SimpleNamespace(
        action_dim=3,
        action_dim_pad=4,
        action_keep_mask_cache=None,
    )
    unpadded = action._cached_action_keep_mask(state, None)
    assert torch.equal(unpadded, torch.tensor([1, 1, 1, 0], dtype=torch.float16))
    assert action._cached_action_keep_mask(state, None).data_ptr() == unpadded.data_ptr()

    pad_mask = torch.tensor([[[False, True, False]]])
    padded = action._cached_action_keep_mask(state, pad_mask)
    assert torch.equal(padded, torch.tensor([1, 0, 1, 0], dtype=torch.float16))
    assert (
        action._cached_action_keep_mask(state, pad_mask.clone()).data_ptr()
        == padded.data_ptr()
    )
    assert action._cached_action_keep_mask(state, None).data_ptr() != unpadded.data_ptr()

    with pytest.raises(ValueError, match="feature pad mask"):
        action._cached_action_keep_mask(state, torch.zeros(1, 3, dtype=torch.bool))


def test_g05_action_physical_prefix_alias_lifecycle():
    action = _load_action_adapter()

    class FakeTensor:
        ndim = 7

        def __init__(self, ptr):
            self.ptr = ptr

        def data_ptr(self):
            return self.ptr

    class FakeGraphCache:
        clears = 0
        frozen = False

        def is_frozen(self):
            return self.frozen

        def clear(self):
            self.clears += 1

    dst_k = [object() for _ in range(24)]
    dst_v = [object() for _ in range(24)]
    physical = types.SimpleNamespace(
        k_caches=[FakeTensor(1000 + i) for i in range(24)],
        v_caches=[FakeTensor(2000 + i) for i in range(24)],
    )
    graph_cache = FakeGraphCache()
    trace_graph_cache = FakeGraphCache()
    state = types.SimpleNamespace(
        action_cache=types.SimpleNamespace(k_caches=dst_k, v_caches=dst_v),
        action_graph_cache=graph_cache,
        action_trace_graph_cache=trace_graph_cache,
        action_prefix_binding=None,
        action_prefix_key=None,
        action_prefix_source=None,
        action_prefix_lens=None,
        num_layers=24,
    )
    expert = types.SimpleNamespace(_rpu_qwen3_5=state)
    prefix = types.SimpleNamespace(
        _rpu_qwen3_5_cache=physical,
        _rpu_qwen3_5_generation=1,
    )

    prefix_lens, changed = action._prepare_prefix(expert, prefix, 64)
    assert changed
    assert graph_cache.clears == 0
    assert trace_graph_cache.clears == 0
    for layer_idx in range(24):
        if layer_idx in action._SHARED_PREFIX_LAYERS:
            assert state.action_k_caches[layer_idx] is physical.k_caches[layer_idx]
            assert state.action_v_caches[layer_idx] is physical.v_caches[layer_idx]
            assert prefix_lens[layer_idx] == 64
        else:
            assert state.action_k_caches[layer_idx] is dst_k[layer_idx]
            assert state.action_v_caches[layer_idx] is dst_v[layer_idx]
            assert prefix_lens[layer_idx] == 0

    _, changed = action._prepare_prefix(expert, prefix, 64)
    assert not changed
    assert graph_cache.clears == 0
    assert trace_graph_cache.clears == 0

    prefix._rpu_qwen3_5_generation = 2
    _, changed = action._prepare_prefix(expert, prefix, 64)
    assert changed
    assert graph_cache.clears == 0
    assert trace_graph_cache.clears == 0

    drift_layer = action._SHARED_PREFIX_LAYERS[2]
    physical.k_caches[drift_layer] = FakeTensor(9000)
    _, changed = action._prepare_prefix(expert, prefix, 64)
    assert changed
    assert graph_cache.clears == 1
    assert trace_graph_cache.clears == 1
    assert state.action_k_caches[drift_layer] is physical.k_caches[drift_layer]

    graph_cache.frozen = trace_graph_cache.frozen = True
    before = dict(vars(state))
    physical.k_caches[drift_layer] = FakeTensor(10000)
    with pytest.raises(RuntimeError, match="READY"):
        action._prepare_prefix(expert, prefix, 64)
    assert vars(state) == before
    assert graph_cache.clears == trace_graph_cache.clears == 1
    physical.k_caches[drift_layer] = state.action_k_caches[drift_layer]
    with pytest.raises(RuntimeError, match="READY"):
        action._prepare_prefix(expert, prefix, 80)
    assert vars(state) == before
    assert graph_cache.clears == trace_graph_cache.clears == 1


def test_g05_action_logical_prefix_falls_back_to_copy(monkeypatch):
    action = _load_action_adapter()

    class FakeTensor:
        def __init__(self, ptr):
            self.ptr = ptr

        def data_ptr(self):
            return self.ptr

    dst_k = [FakeTensor(3000 + i) for i in range(24)]
    dst_v = [FakeTensor(4000 + i) for i in range(24)]
    state = types.SimpleNamespace(
        action_cache=types.SimpleNamespace(k_caches=dst_k, v_caches=dst_v),
        action_graph_cache=types.SimpleNamespace(clear=lambda: None),
        action_prefix_binding=None,
        action_prefix_key=None,
        action_prefix_source=None,
        action_prefix_lens=None,
        num_layers=24,
    )
    expert = types.SimpleNamespace(_rpu_qwen3_5=state)
    prefix = types.SimpleNamespace(
        _rpu_qwen3_5_generation=1,
        key_cache={layer: object() for layer in action._SHARED_PREFIX_LAYERS},
        value_cache={layer: object() for layer in action._SHARED_PREFIX_LAYERS},
    )
    copies = []
    monkeypatch.setattr(
        action,
        "_copy_logical_prefix",
        lambda dst, layer, key, value, length: copies.append(
            (dst, layer, key, value, length)
        ),
    )

    _, changed = action._prepare_prefix(expert, prefix, 64)
    assert changed
    assert [copy[1] for copy in copies] == list(action._SHARED_PREFIX_LAYERS)
    assert all(copy[0] is state.action_cache and copy[4] == 64 for copy in copies)
    assert state.action_k_caches is dst_k
    assert state.action_v_caches is dst_v
    assert state.action_prefix_binding[0] == "copy"

    _, changed = action._prepare_prefix(expert, prefix, 64)
    assert not changed
    assert len(copies) == len(action._SHARED_PREFIX_LAYERS)


def _config():
    layer_types = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
        for i in range(24)
    ]
    return types.SimpleNamespace(
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_value_heads=16,
        linear_num_key_heads=16,
        linear_value_head_dim=128,
        linear_key_head_dim=128,
        linear_conv_kernel_dim=4,
        attention_bias=False,
        hidden_act="silu",
        input_type="embedding",
        output_type="lm_head",
        adaptive_mode=None,
        use_final_norm=True,
        rms_norm_eps=1.0e-6,
        rope_parameters={
            "rope_theta": 10_000_000.0,
            "rope_type": "default",
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
            "partial_rotary_factor": 0.25,
        },
        layer_types=layer_types,
    )


def test_g05_vlm_profile_is_exact(monkeypatch):
    module = _load_adapter()
    errors = types.ModuleType("rpu_backend.api.errors")
    errors.UnsupportedModelError = ValueError
    monkeypatch.setitem(__import__("sys").modules, "rpu_backend.api.errors", errors)

    module._check_g05_vlm_profile(_config())
    bad = _config()
    bad.linear_conv_kernel_dim = 3
    with pytest.raises(ValueError, match="canonical Qwen3.5-2B VLM"):
        module._check_g05_vlm_profile(bad)


def test_g05_reuses_decoder_only_qwen35_install_contract():
    adapter = _ADAPTER.read_text()
    qwen_text = _QWEN_TEXT.read_text()
    qwen_cpp = _QWEN_CPP.read_text()
    vlm_install = adapter.split("def patch_g05_vlm_for_rpu(", 1)[1].split(
        "\ndef _rpu_g05_encode_inference(", 1
    )[0]
    compile(adapter, str(_ADAPTER), "exec")
    assert "_cpu_stage_weights=True" in adapter
    assert 'vlm.layers.to("cpu").to("rpu")' not in adapter
    assert "if cpu_stage_weights:" in qwen_text
    assert "weights[index] = weight.detach().to(" in qwen_text
    assert "run_qwen3_5_text(" in adapter
    assert "Qwen3_5Cache.from_config(" in adapter
    assert 'handle, linear_accumulation == "acc32"' in adapter
    assert "qwen3_5_set_mrope_position_delta" in adapter
    assert 'inputs_embeds.to(device="cpu", dtype=torch.float16)' in adapter
    assert 'hidden.to(device="rpu")' in adapter
    assert 'hidden = hidden.to("cpu")' in adapter
    assert "RPUCache.reset_to_position(cache, 0)" in adapter
    assert "for state in cache.recurrent_states:" in adapter
    assert "for state in cache.conv_states:" in adapter
    assert "BAR/block masks are not ported" in adapter
    assert "interleaved InferenceState " in adapter
    assert "objects are not supported" in adapter
    assert "_rpu_g05_build_prefix_action_kv" in adapter
    assert "patch_g05_action_expert_for_rpu(" in adapter
    assert "torch.ops.rpu.gather_embedding(" in adapter
    assert "_rpu_fusion_target=hidden" in adapter
    assert 'object.__setattr__(\n            model,\n            "_rpu_g05_proprio_mlp"' in adapter
    assert "copy.deepcopy(source_mlp)" in adapter
    assert "self._rpu_g05_proprio_mlp(" in adapter
    assert "self.proprio_embedder(proprio)" not in adapter
    assert "hidden = hidden.clone()" in adapter
    assert "_rpu_g05_vlm_handle" not in adapter
    assert "prefill_graph_cache = _rb.graph.GraphCache(max_entries=1)" in adapter
    assert "prefill_rope_cache = None" in adapter
    assert "if inputs_embeds.shape[1] == 1:" in adapter
    assert 'op_id=f"rpu_{family}_prefill"' in qwen_text
    assert "_get_prefill_mrope_tables(" in qwen_text
    assert "qwen3_5_set_prefill_rope(handle, pcos, psin)" in qwen_text
    assert "*plan.graph_key_words()" in qwen_text
    assert "0, planned_stage_descriptor)" in qwen_text
    assert "!decode && ctx().position == 0 && chunk.offset == 0" in qwen_cpp
    assert "if (initial_prefill_chunk)" in qwen_cpp
    assert 'D("gdn_state"), Hc * Dk * Dv, c10::Half(0.0f), nc' in qwen_cpp
    assert 'D("gdn_cs"), Kc * lconv, c10::Half(0.0f), nc' in qwen_cpp
    assert "fast_replay_active_ = fast_replay_enabled_ && seq_len > 1;" in qwen_cpp
    assert "cfg.fast_replay_skip_layer_loop = fast_replay_active_;" in qwen_cpp
    assert "convert_linear_weights_inplace(\n        inner.layers," in qwen_text
    chunk_call = "torch.ops.rpu.qwen3_5_set_chunk_envelope(handle, physical_capacity, 256)"
    assert "physical_capacity = vlm._rpu_qwen3_5.chunk_envelope[0]" in vlm_install
    assert vlm_install.count(chunk_call) == 1
    chunk_index = vlm_install.index(chunk_call)
    assert vlm_install.rfind(
        "if _allow_certified_g05_k1:", 0, chunk_index
    ) < chunk_index < vlm_install.index(
        "torch.ops.rpu.qwen3_5_set_linear_acc32("
    )
    assert "execution_config=install_execution" in vlm_install
    assert "os.environ" not in vlm_install


def test_g05_failed_install_cleanup_preserves_irreversible_owner_contract():
    adapter = _ADAPTER.read_text()
    action = _ACTION_ADAPTER.read_text()
    vlm_install = adapter.split("def patch_g05_vlm_for_rpu(", 1)[1].split(
        "\ndef _rpu_g05_encode_inference(", 1
    )[0]
    action_install = action.split("def patch_g05_action_expert_for_rpu(", 1)[1]

    assert "except BaseException as error:" in vlm_install
    assert "_release_live_instance" not in vlm_install
    assert "_retire_failed_qwen3_5_text_install(vlm, state=state)" in vlm_install
    assert "state is not state_before" in vlm_install
    assert "reversed(snapshots)" in vlm_install
    assert '"_rpu_g05_vlm_ready"' in vlm_install
    assert '"build_causal_mask_and_position_ids"' in vlm_install
    assert '"_build_prefix_action_kv"' in vlm_install
    assert "error.add_note(" in vlm_install

    assert "except BaseException as error:" in action_install
    assert "_retire_failed_qwen3_5_text_install(expert, state=state)" in action_install
    assert "state is not state_before" in action_install
    assert "reversed(snapshots)" in action_install
    assert '"_rpu_g05_action_ready"' in action_install
    assert '"encode_time"' in action_install
    assert '"forward"' in action_install
    retirement = _QWEN_TEXT.read_text().split("def _retire_qwen3_5_text_state(", 1)[1].split("def _retire_failed_qwen3_5_text_install(", 1)[0]
    for cache_name in (
        "graph_cache",
        "prefill_graph_cache",
        "action_graph_cache",
        "action_trace_graph_cache",
    ):
        assert f'"{cache_name}"' in retirement
    assert retirement.index("cache.clear()") < retirement.index("qwen3_5_destroy(handle)") < retirement.index("finalizer.detach()")
    assert "error.add_note(" in action_install
    assert "_rpu_g05_action_install_started = True" in action_install


def test_g05_direct_installers_require_cpu_before_irreversible_mutation():
    adapter = _ADAPTER.read_text()
    action = _ACTION_ADAPTER.read_text()
    vlm_install = adapter.split("def patch_g05_vlm_for_rpu(", 1)[1].split(
        "\ndef _rpu_g05_encode_inference(", 1
    )[0]
    action_install = action.split("def patch_g05_action_expert_for_rpu(", 1)[1]

    for install in (vlm_install, action_install):
        assert 'parameter.device.type != "cpu"' in install
        assert "CPU-hosted checkpoint" in install
        assert install.index('parameter.device.type != "cpu"') < install.index(
            "install_started = True"
        )


def test_g05_claims_single_live_owner_before_any_vision_mutation():
    adapter = _ADAPTER.read_text()
    vision_install = adapter.split("def patch_g05_vision_for_rpu(", 1)[1].split(
        "\ndef _pack_g05_vision(", 1
    )[0]
    vlm_install = adapter.split("def patch_g05_vlm_for_rpu(", 1)[1].split(
        "\ndef _rpu_g05_encode_inference(", 1
    )[0]
    policy_install = adapter.split("def patch_g05_policy_for_rpu(", 1)[1].split(
        "\n\n__all__ =", 1
    )[0]

    assert vision_install.index("_claim_live_instance(model)") < (
        vision_install.index("vision.half()")
    )
    assert vlm_install.index("_claim_live_instance(model)") < (
        vlm_install.index('vlm.layers.to(device="cpu", dtype=torch.float16)')
    )
    assert policy_install.index("_claim_live_instance(model)") < (
        policy_install.index("patch_g05_vision_for_rpu(")
    )
    assert "_VISION_PATCH_LOCK.acquire(blocking=False)" in vision_install
    assert "_restore_instance_attrs(snapshots)" in vision_install
    assert "_rollback_qwen3_5_vision_install" in vision_install
    assert "@_transactional_g05_policy_install" in adapter
    assert "_g05_vision_runtime_ready(model)" in vision_install
    assert "_g05_vlm_runtime_ready(model)" in adapter
    assert "_g05_policy_runtime_ready(policy)" in adapter
    assert "_g05_action_runtime_ready(expert)" in adapter
    assert "installed_forward=implementation" in adapter
    qwen_vision = _QWEN_VISION.read_text()
    assert "installed_forward=None" in qwen_vision
    assert '"_rpu_vision_step0_pos_temporal_num_frames"' in qwen_vision
    assert '"_rpu_vision_step0_pos_camera_batch_count"' in qwen_vision
    assert "_rpu_g05_policy_forward" in adapter


def test_g05_exact_policy_reaches_install_without_numeric_opt_in(monkeypatch):
    module = _load_adapter()
    monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    perf_env = ("RPU_FASTREPLAY_SKIP_SYNC",)
    for name in perf_env:
        monkeypatch.delenv(name, raising=False)

    events = []
    monkeypatch.setattr(module, "_is_vla_rtc_fm_release", lambda *_args: True)
    monkeypatch.setattr(module, "_check_g05_action_profile", lambda _cfg: None)
    monkeypatch.setattr(
        module,
        "_preflight_g05_vlm",
        lambda model: events.append("vlm-preflight") or model.vlm,
    )
    monkeypatch.setattr(
        module,
        "_preflight_g05_vision",
        lambda _model: events.append("vision-preflight")
        or (object(), object(), (0, "factorized", 24, None, False)),
    )
    monkeypatch.setattr(
        module.copy,
        "deepcopy",
        lambda value: events.append("deepcopy") or value,
    )
    def stop_at_claim(_owner):
        events.append("claim")
        raise RuntimeError("reached-claim")

    monkeypatch.setattr(module, "_claim_live_instance", stop_at_claim)
    for name in (
        "patch_g05_vision_for_rpu",
        "patch_g05_vlm_for_rpu",
        "patch_g05_action_expert_for_rpu",
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(
                f"claim must stop before {_name}"
            ),
        )

    policy_module_name = "g05_numeric_gate_test_policy"
    policy_module = types.ModuleType(policy_module_name)
    policy_module._ensure_sample_list = lambda samples: samples
    monkeypatch.setitem(__import__("sys").modules, policy_module_name, policy_module)
    policy_type = type(
        "Policy",
        (),
        {
            "forward": lambda self, *_args, **_kwargs: None,
            "predict_action": lambda self, batch: batch,
            "forward_inference": lambda self, **_kwargs: {},
        },
    )
    policy_type.__module__ = policy_module_name
    model = types.SimpleNamespace(
        action_expert=types.SimpleNamespace(config=object(), training=False),
        fm_helper=_canonical_fm_helper(),
        cfg=types.SimpleNamespace(ae_vlm_condition_mode="cross_attn_only"),
        vlm=types.SimpleNamespace(input_proj=__import__("torch").nn.Linear(2, 2)),
        proprio_embedder=types.SimpleNamespace(
            mlp=__import__("torch").nn.Linear(2, 2)
        ),
        _forward_embed=lambda *_args, **_kwargs: None,
        inference_fm=lambda *_args, **_kwargs: None,
    )
    policy = policy_type()
    policy.training = False
    policy.continuous_action = True
    policy.discrete_action = False
    policy.predict_cot = False
    policy.model = model

    with pytest.raises(RuntimeError, match="reached-claim"):
        module.patch_g05_policy_for_rpu(policy)

    assert events == ["vlm-preflight", "vision-preflight", "deepcopy", "claim"]
    assert not hasattr(policy, "_rpu_g05_policy_install_started")
    assert all(name not in os.environ for name in perf_env)


def test_g05_exact_vision_reaches_process_claim_without_numeric_opt_in(monkeypatch):
    module = _load_adapter()
    monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)

    def stop_at_claim(_owner):
        raise RuntimeError("reached-claim")

    monkeypatch.setattr(module, "_claim_live_instance", stop_at_claim)
    real_import = module.importlib.import_module
    vision_adapter = types.SimpleNamespace(
        install_qwen3_5_vision_for_rpu=lambda *_args, **_kwargs: pytest.fail(
            "claim must stop before shared Vision install"
        ),
        _qwen3_5_vision_runtime_complete=lambda _vision: False,
    )
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: vision_adapter
        if name == "rpu_backend.adapters.qwen3_5.vision"
        else real_import(name),
    )
    config = types.SimpleNamespace(
        depth=24,
        num_heads=16,
        hidden_size=1024,
        intermediate_size=4096,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
        out_hidden_size=2048,
        hidden_act="gelu_pytorch_tanh",
        temporal_freq=0,
        spacetime_mode="factorized",
        token_drop_layer=None,
        temporal_pe_pretrain_frames=None,
        batch_all_cameras=False,
    )
    vision = types.SimpleNamespace(
        config=config,
        parameters=lambda: iter(()),
        half=lambda: pytest.fail("claim must stop before FP16 mutation"),
    )
    model = types.SimpleNamespace(training=False, vision_tower=vision)

    with pytest.raises(RuntimeError, match="reached-claim"):
        module.patch_g05_vision_for_rpu(
            model, _allow_certified_g05_k1=True
        )

    assert not hasattr(model, "_rpu_g05_vision_install_started")


def test_qwen35_shared_vision_validates_profile_before_mutation():
    source = _QWEN_VISION.read_text()
    install = source.split(
        "def _install_qwen3_5_vision_for_rpu_impl(", 1
    )[1].split("\ndef ", 1)[0]
    profile_check = install.index("if vision_profile not in controlled_profiles:")
    assert (
        "_QWEN3_5_VISION_PROFILES | _QWEN3_5_VISION_CANDIDATE_PROFILES"
        in install
    )
    numeric_gate = install.index(
        'os.environ.get("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED")'
    )
    marker = install.index("vision_model._rpu_vision_installing = True")
    handle = install.index("torch.ops.rpu.qwen3_5_vision_create()")
    swizzle = install.index("_convert_vision_block_weights_for_rpu(")

    assert profile_check < numeric_gate < marker < handle < swizzle
    assert "not _allow_certified_g05_k1" in install


def test_g05_policy_transaction_restores_composite_hooks_on_base_exception(
    monkeypatch,
):
    module = _load_adapter()

    class InstallInterrupted(BaseException):
        pass

    original_execution = {"prefill": {"chunk_size": 64}}
    original_embed = object()
    original_inference = object()
    original_encode = object()
    original_predict = object()
    original_policy_forward = object()
    model = types.SimpleNamespace(
        _rpu_execution=original_execution,
        _forward_embed=original_embed,
        inference_fm=original_inference,
    )
    processor = types.SimpleNamespace(encode_inference=original_encode)
    policy = types.SimpleNamespace(
        model=model,
        processor=processor,
        discrete_action=True,
        forward=original_policy_forward,
        predict_action=original_predict,
    )
    perf_env_names = ("RPU_FASTREPLAY_SKIP_SYNC",)
    monkeypatch.delenv(perf_env_names[0], raising=False)
    monkeypatch.setenv("RPU_LINEAR_ACC32", "inert-sentinel")

    @module._transactional_g05_policy_install
    def fail_during_publish(owner):
        owner._rpu_g05_policy_install_started = True
        model._rpu_execution = {"prefill": {"chunk_size": 128}}
        owner._rpu_execution = {"components": {"language_model": model._rpu_execution}}
        model._rpu_g05_embed_w_rpu = object()
        model._forward_embed = "wrapped-embed"
        model.inference_fm = "wrapped-inference"
        processor._rpu_g05_original_encode_inference = original_encode
        processor.encode_inference = "wrapped-encode"
        owner._rpu_g05_original_predict_action = original_predict
        owner._rpu_g05_original_policy_forward = original_policy_forward
        owner._rpu_g05_ensure_sample_list = lambda samples: samples
        owner.discrete_action = False
        owner.forward = "wrapped-forward"
        owner.predict_action = "wrapped-predict"
        owner._rpu_g05_policy_ready = True
        for name in perf_env_names:
            os.environ[name] = "1"
        raise InstallInterrupted

    with pytest.raises(InstallInterrupted):
        fail_during_publish(policy)

    assert model._rpu_execution is original_execution
    assert not hasattr(policy, "_rpu_execution")
    assert model._forward_embed is original_embed
    assert model.inference_fm is original_inference
    assert not hasattr(model, "_rpu_g05_embed_w_rpu")
    assert processor.encode_inference is original_encode
    assert not hasattr(processor, "_rpu_g05_original_encode_inference")
    assert policy.discrete_action is True
    assert policy.forward is original_policy_forward
    assert policy.predict_action is original_predict
    assert not hasattr(policy, "_rpu_g05_policy_ready")
    assert not hasattr(policy, "_rpu_g05_original_predict_action")
    assert not hasattr(policy, "_rpu_g05_original_policy_forward")
    assert not hasattr(policy, "_rpu_g05_ensure_sample_list")
    assert all(name not in os.environ for name in perf_env_names)
    assert os.environ["RPU_LINEAR_ACC32"] == "inert-sentinel"
    assert policy._rpu_g05_policy_install_started is True
    with pytest.raises(RuntimeError, match="already attempted"):
        fail_during_publish(policy)


def test_g05_policy_transaction_retires_components_in_reverse_order(monkeypatch):
    module = _load_adapter()
    events = []

    class Cache:
        def __init__(self, name):
            self.name = name

        def clear(self):
            events.append(f"{self.name}-clear")

        def cache_invariant_ok(self):
            return True

    class Finalizer:
        def __init__(self, name):
            self.name = name
            self.alive = True

        def __call__(self):
            events.append(f"{self.name}-finalize")
            self.alive = False

        def detach(self):
            events.append(f"{self.name}-detach")
            self.alive = False

    monkeypatch.setattr(__import__("torch").ops.rpu, "qwen3_5_destroy", lambda handle: events.append(f"{handle}-destroy"), raising=False)

    originals = {name: object() for name in (
        "vision_forward", "vlm_forward", "vlm_decode", "mask", "prefix",
        "action_forward", "encode_time",
    )}
    vision = types.SimpleNamespace(forward=originals["vision_forward"])
    vlm = types.SimpleNamespace(
        forward=originals["vlm_forward"],
        decode=originals["vlm_decode"],
    )
    expert = types.SimpleNamespace(
        forward=originals["action_forward"],
        encode_time=originals["encode_time"],
    )
    model = types.SimpleNamespace(
        vision_tower=vision,
        vlm=vlm,
        action_expert=expert,
        build_causal_mask_and_position_ids=originals["mask"],
        _build_prefix_action_kv=originals["prefix"],
    )
    policy = types.SimpleNamespace(model=model)

    vision_module = types.ModuleType("rpu_backend.adapters.qwen3_5.vision")

    def rollback_vision(owner):
        owner._rpu_vision_graph_cache.clear()
        owner._rpu_vision_handle_finalizer()
        for name in tuple(vars(owner)):
            if name.startswith("_rpu_vision_"):
                vars(owner).pop(name)

    vision_module._rollback_qwen3_5_vision_install = rollback_vision
    monkeypatch.setitem(
        __import__("sys").modules, vision_module.__name__, vision_module
    )

    @module._transactional_g05_policy_install
    def fail_after_components_install(owner):
        owner._rpu_g05_policy_install_started = True

        model._rpu_g05_vision_install_started = True
        model._rpu_g05_vision_ready = True
        vision._rpu_vision_handle = 11
        vision._rpu_vision_graph_cache = Cache("vision")
        vision._rpu_vision_handle_finalizer = Finalizer("vision")
        vision._rpu_g05_vision_forward_impl = object()
        vision.forward = object()

        model._rpu_g05_vlm_install_started = True
        model._rpu_g05_vlm_ready = True
        vlm._rpu_qwen3_5_text_install_started = True
        vlm._rpu_qwen3_5 = types.SimpleNamespace(
            handle=23,
            graph_cache=Cache("vlm"),
            handle_finalizer=Finalizer("vlm"),
        )
        vlm.forward = object()
        vlm.decode = object()
        model.build_causal_mask_and_position_ids = object()
        model._build_prefix_action_kv = object()

        expert._rpu_g05_action_install_started = True
        expert._rpu_qwen3_5_text_install_started = True
        expert._rpu_g05_action_ready = True
        expert._rpu_qwen3_5 = types.SimpleNamespace(
            handle=24,
            action_graph_cache=Cache("action"),
            handle_finalizer=Finalizer("action"),
        )
        expert.forward = object()
        expert.encode_time = object()
        raise RuntimeError("publish failed")

    with pytest.raises(RuntimeError, match="publish failed"):
        fail_after_components_install(policy)

    assert events == [
        "vlm-clear", "action-clear", "vision-clear",
        "action-clear", "24-destroy", "action-detach",
        "vlm-clear", "23-destroy", "vlm-detach",
        "vision-clear", "vision-finalize",
    ]
    assert vision.forward is originals["vision_forward"]
    assert vlm.forward is originals["vlm_forward"]
    assert vlm.decode is originals["vlm_decode"]
    assert model.build_causal_mask_and_position_ids is originals["mask"]
    assert model._build_prefix_action_kv is originals["prefix"]
    assert expert.forward is originals["action_forward"]
    assert expert.encode_time is originals["encode_time"]
    assert not hasattr(vlm, "_rpu_qwen3_5")
    assert not hasattr(expert, "_rpu_qwen3_5")
    assert not hasattr(model, "_rpu_g05_vision_ready")
    assert not hasattr(model, "_rpu_g05_vlm_ready")
    assert not hasattr(expert, "_rpu_g05_action_ready")
    assert policy._rpu_g05_policy_install_started is True
    assert model._rpu_g05_vision_install_started is True
    assert model._rpu_g05_vlm_install_started is True
    assert expert._rpu_g05_action_install_started is True
    assert vlm._rpu_qwen3_5_text_install_started is True
    assert expert._rpu_qwen3_5_text_install_started is True


def test_g05_action_runtime_ready_requires_live_state_and_bound_hooks(request):
    module = _load_action_adapter()
    from test_qwen3_5_install_lifecycle import _native_fixture_state

    class Expert:
        pass

    expert = Expert()
    expert._rpu_g05_action_ready = True
    expert._rpu_g05_action_install_started = True
    expert._rpu_qwen3_5_text_install_started = True
    state = _native_fixture_state(request,
        handle=object(),
        handle_finalizer=types.SimpleNamespace(alive=True),
        graph_cache=object(),
        action_graph_cache=object(),
        action_trace_graph_cache=object(),
        action_cache=object(),
        action_k_caches=object(),
        action_v_caches=object(),
        action_prefix_binding=None,
        action_prefix_key=None,
        action_prefix_source=None,
        action_prefix_lens=None,
        action_rope_cache=None,
        action_layout_cache=None,
        action_keep_mask_cache=None,
        action_dim=27,
        action_dim_pad=32,
    )
    expert._rpu_qwen3_5 = state
    expert._rpu_execution_component_id = "action_expert"
    expert._rpu_execution_generation = 0
    expert._rpu_g05_adaptive_mod_cache = {}
    expert._rpu_g05_adaptive_mod_stack_cache = None
    expert._rpu_g05_time_cond_cache = {}
    expert._rpu_g05_fixed_time_schedule = None
    expert._rpu_g05_rtc_time_schedules = {}
    expert._rpu_g05_original_encode_time = lambda timestep: timestep
    expert._rpu_g05_original_forward = lambda *args, **kwargs: (args, kwargs)
    expert.encode_time = types.MethodType(module._rpu_g05_encode_time, expert)
    expert.forward = types.MethodType(module._rpu_g05_action_forward, expert)

    assert module._g05_action_runtime_ready(expert)
    original_encode_time = expert._rpu_g05_original_encode_time
    del expert._rpu_g05_original_encode_time
    assert not module._g05_action_runtime_ready(expert)
    expert._rpu_g05_original_encode_time = original_encode_time
    state.handle_finalizer.detach()
    assert not module._g05_action_runtime_ready(expert)
    with pytest.raises(RuntimeError, match="incomplete runtime state"):
        module.patch_g05_action_expert_for_rpu(expert)


def test_g05_vision_runtime_ready_requires_shared_wrapper_and_temporal_keepalive(
    monkeypatch,
):
    module = _load_adapter()
    vision_module = types.ModuleType("rpu_backend.adapters.qwen3_5.vision")

    class Vision:
        pass

    class Model:
        pass

    vision = Vision()
    vision.config = types.SimpleNamespace(temporal_freq=4)
    vision._shared_runtime_live = True

    def shared_forward(self, *_args, **_kwargs):
        return self

    implementation = types.MethodType(shared_forward, vision)
    vision._rpu_g05_vision_forward_impl = implementation
    vision._rpu_g05_temporal_pe = object()
    vision.forward = types.MethodType(module._rpu_g05_vision_forward, vision)

    def runtime_complete(owner, *, installed_forward=None):
        return bool(
            owner._shared_runtime_live and installed_forward is implementation
        )

    vision_module._qwen3_5_vision_runtime_complete = runtime_complete
    monkeypatch.setitem(
        __import__("sys").modules, vision_module.__name__, vision_module
    )
    model = Model()
    model.vision_tower = vision
    model._rpu_g05_vision_install_started = True
    model._rpu_g05_vision_ready = True

    assert module._g05_vision_runtime_ready(model)
    vision._rpu_g05_temporal_pe = None
    assert not module._g05_vision_runtime_ready(model)
    with pytest.raises(RuntimeError, match="incomplete runtime state"):
        module.patch_g05_vision_for_rpu(model)


@pytest.mark.parametrize("cameras", [2, 3])
@pytest.mark.parametrize("accumulation", ["acc16", "acc32"])
def test_g05_vision_runtime_ready_validates_real_camera_group_closure(
    monkeypatch, cameras, accumulation,
):
    from test_g05_camera_profile import source_fixture

    module = _load_adapter()
    policy, processor = source_fixture()
    names = module._G05_RELEASE_CAMERA_NAMES[:cameras]
    module.configure_g05_camera_profile(policy, processor, camera_names=names)
    model, vision = policy.model, policy.model.vision_tower
    model._rpu_g05_vision_ready = model._rpu_g05_vision_install_started = True
    vision.forward = types.MethodType(module._rpu_g05_vision_forward, vision)
    vision._rpu_vision_w8a16 = True
    vision._rpu_g05_weight_dtype = "w8a16"
    vision._rpu_g05_linear_accumulation = accumulation
    vision._rpu_vision_linear_acc32 = accumulation == "acc32"
    vision._rpu_vision_num_cores = 8
    vision._rpu_vision_k1_camera_group_available = (True, True)
    vision._rpu_vision_exact_chunk_size = None
    vision._rpu_vision_chunk_size_cap = 0

    def shared_forward(self, *_args, **_kwargs):
        return self

    original = types.MethodType(shared_forward, vision)
    vision._rpu_g05_vision_forward_impl = original
    seen = []
    shared = types.ModuleType("rpu_backend.adapters.qwen3_5.vision")
    def runtime_complete(owner, *, installed_forward=None):
        seen.append(installed_forward)
        return owner is vision and installed_forward is original
    shared._qwen3_5_vision_runtime_complete = runtime_complete
    monkeypatch.setitem(__import__("sys").modules, shared.__name__, shared)

    class Session:
        hot = False
        checks = 0
        def require_cold(self):
            self.checks += 1
            if self.hot:
                raise RuntimeError("already warm")
    session = Session()
    policy._execution_session = session
    policy._rpu_g05_fm_only_customer_release = True
    assert module._g05_vision_runtime_ready(model) and seen[-1] is original
    module.configure_g05_vision_camera_batch(policy, cameras)
    wrapper = vision._rpu_g05_vision_forward_impl
    assert module._g05_vision_runtime_ready(model) and seen[-1] is original
    assert module.patch_g05_vision_for_rpu(
        model, _allow_certified_g05_k1=True, weight_dtype="w8a16",
        linear_accumulation=accumulation,
    ) is model
    assert vision._rpu_g05_vision_forward_impl is wrapper
    module.configure_g05_vision_camera_batch(policy, cameras)
    assert vision._rpu_g05_vision_forward_impl is wrapper
    session.hot = True
    checks = session.checks
    assert module._g05_vision_runtime_ready(model)
    assert session.checks == checks  # Readiness is read-only even after warmup.
    for requested in (1, cameras):
        with pytest.raises(RuntimeError, match="already warm"):
            module.configure_g05_vision_camera_batch(policy, requested)
        assert vision._rpu_g05_vision_forward_impl is wrapper

    def rejects(owner, name, value):
        previous = getattr(owner, name)
        setattr(owner, name, value)
        try:
            assert not module._g05_vision_runtime_ready(model), name
        finally:
            setattr(owner, name, previous)
        assert module._g05_vision_runtime_ready(model)

    rejects(wrapper, "_g05_k1_camera_group_wrapper", False)
    rejects(wrapper, "_g05_k1_camera_batch", True)
    rejects(wrapper, "_g05_k1_b3_wrapper", cameras != 3)
    rejects(wrapper, "_g05_k1_original", lambda *a, **k: None)
    rejects(wrapper, "_g05_k1_original", types.MethodType(shared_forward, object()))
    rejects(vision, "_rpu_vision_k1_camera_group_available", (False, False))
    rejects(vision, "_rpu_vision_exact_chunk_size", 256)
    rejects(vision, "_rpu_vision_chunk_size_cap", 256)
    rejects(vision, "_rpu_vision_w8a16", False)
    rejects(vision, "_rpu_g05_weight_dtype", "fp16")
    rejects(vision, "_rpu_vision_linear_acc32", accumulation != "acc32")
    rejects(vision, "_rpu_vision_num_cores", 4)
    rejects(model.cfg, "num_input_images", 5 - cameras)
    if cameras == 2:
        rejects(model, "_rpu_g05_camera_names", tuple(reversed(names)))

    counterfeit = lambda *a, **k: None
    vars(counterfeit).update(vars(wrapper))
    rejects(vision, "_rpu_g05_vision_forward_impl", counterfeit)
    # Use the real selector's exact code with a foreign captured owner. This
    # catches validating only its markers or the retained method's __self__.
    def cell(value):
        return (lambda: value).__closure__[0]
    closure = tuple(cell(object()) if name == "vision" else captured
                    for name, captured in zip(wrapper.__code__.co_freevars,
                                              wrapper.__closure__))
    foreign = types.FunctionType(wrapper.__code__, wrapper.__globals__,
                                 closure=closure)
    vars(foreign).update(vars(wrapper))
    rejects(vision, "_rpu_g05_vision_forward_impl", foreign)
    # The actual shared readiness result remains authoritative after unwrap.
    shared._qwen3_5_vision_runtime_complete = lambda *a, **k: False
    assert not module._g05_vision_runtime_ready(model)


def test_g05_vlm_runtime_ready_requires_live_graphs_owner_and_bound_hooks(request):
    module = _load_adapter()
    from test_qwen3_5_install_lifecycle import _native_fixture_state

    class Model:
        pass

    class VLM:
        pass

    model = Model()
    vlm = VLM()
    model.vlm = vlm
    model._rpu_g05_vlm_install_started = True
    model._rpu_g05_vlm_ready = True
    model._rpu_g05_prefix_bridge_installed = True
    vlm._rpu_qwen3_5_text_install_started = True
    vlm._rpu_g05_max_seq_len = 2048
    vlm._rpu_g05_owner = weakref.ref(model)
    vlm._rpu_g05_original_decode = lambda hidden: hidden
    vlm._rpu_qwen3_5 = _native_fixture_state(request,
        handle=object(),
        handle_finalizer=types.SimpleNamespace(alive=True),
        graph_cache=object(),
        prefill_graph=object(),
        prefill_graph_cache=object(),
        prefill_graph_sig=None,
        prefill_debug_graph_cache=None,
        prefill_debug_graph_sig=None,
        prefill_rope_cache=None,
        max_seq_len=2048,
        num_layers=24,
        hidden_size=2048,
        rotary_dim=128,
        rope_theta=10_000_000.0,
        mrope_section=(11, 11, 10),
        padding_budget=64,
    )
    vlm.forward = types.MethodType(module._rpu_g05_vlm_forward, vlm)
    vlm.decode = types.MethodType(module._rpu_g05_vlm_decode, vlm)
    model._rpu_g05_original_build_causal_mask_and_position_ids = lambda *args: args
    model.build_causal_mask_and_position_ids = types.MethodType(
        module._rpu_g05_build_causal_mask_and_position_ids, model
    )
    model._build_prefix_action_kv = types.MethodType(
        module._rpu_g05_build_prefix_action_kv, model
    )

    assert module._g05_vlm_runtime_ready(model)
    model._rpu_g05_prefix_bridge_installed = False
    del model._build_prefix_action_kv
    assert module._g05_vlm_runtime_ready(model)
    model._rpu_g05_prefix_bridge_installed = True
    model._build_prefix_action_kv = types.MethodType(
        module._rpu_g05_build_prefix_action_kv, model
    )
    vlm._rpu_qwen3_5.handle_finalizer.detach()
    assert not module._g05_vlm_runtime_ready(model)
    with pytest.raises(RuntimeError, match="incomplete runtime state"):
        module.patch_g05_vlm_for_rpu(model)


def _g05_vlm_installer_fixture(monkeypatch, module, events, request, *, linear_acc32=True):
    from rpu_backend.adapters.qwen3_5 import text as text_module
    from test_qwen3_5_install_lifecycle import _native_fixture_state

    class CastTarget:
        def to(self, *, device, dtype):
            events.append(("cast", device, dtype))
            return self

    class Cache:
        def clear(self):
            events.append("clear")

        def cache_invariant_ok(self):
            return True

    class VLM:
        def parameters(self):
            return ()

    class Model:
        pass

    vlm = VLM()
    vlm.layers = CastTarget()
    vlm.norm = CastTarget()
    vlm.config = object()
    vlm.forward = lambda *_args, **_kwargs: None
    vlm.decode = lambda *_args, **_kwargs: None
    model = Model()
    model.vlm = vlm
    model.build_causal_mask_and_position_ids = lambda *_args: None

    def install(
        owner, config, max_seq_len, *, execution_config=None,
        _cpu_stage_weights, _g05_w8a16=False, _g05_math_precision="base",
    ):
        assert execution_config == {"prefill": {"linear_acc32": linear_acc32}}
        assert (owner, config, max_seq_len, _cpu_stage_weights) == (
            vlm,
            vlm.config,
            2048,
            True,
        )
        events.append("install")
        owner._rpu_qwen3_5_text_install_started = True
        owner._rpu_qwen3_5 = _native_fixture_state(request,
            handle=23,
            graph_cache=Cache(),
            prefill_graph=types.SimpleNamespace(invalidate=lambda: events.append("discard")),
            max_seq_len=2048,
            chunk_envelope=(2176, 640),
            num_layers=24,
            hidden_size=2048,
            linear_acc32=linear_acc32,
            g05_math_precision=_g05_math_precision,
            rotary_dim=128,
            rope_theta=10_000_000.0,
            mrope_section=(11, 11, 10),
            padding_budget=64,
        )
        return 23

    monkeypatch.setattr(text_module, "install_qwen3_5_text_for_rpu", install)
    monkeypatch.setattr(__import__("torch").ops.rpu, "qwen3_5_destroy", lambda _h: events.append("destroy"), raising=False)
    monkeypatch.setattr(module, "_preflight_g05_vlm", lambda owner: owner.vlm)
    monkeypatch.setattr(module, "_claim_live_instance", lambda _owner: None)
    monkeypatch.setattr(
        __import__("rpu_backend"),
        "graph",
        types.SimpleNamespace(
            GraphCache=lambda *, max_entries: events.append(
                ("graph_cache", max_entries)
            )
            or Cache()
        ),
    )
    monkeypatch.setattr(
        __import__("torch").ops.rpu,
        "qwen3_5_set_chunk_envelope",
        lambda handle, max_kv_len, chunk: events.append(
            ("chunk_envelope", handle, max_kv_len, chunk)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        __import__("torch").ops.rpu,
        "qwen3_5_set_linear_acc32",
        lambda handle, enabled: events.append(("linear_acc32", handle, enabled)),
        raising=False,
    )
    monkeypatch.setattr(
        __import__("torch").ops.rpu,
        "qwen3_5_set_fast_replay",
        lambda handle, enabled: events.append(("fast_replay", handle, enabled)),
        raising=False,
    )
    monkeypatch.setattr(
        __import__("torch").ops.rpu,
        "qwen3_5_set_retained_prefill_graph",
        lambda handle, enabled: events.append(
            ("retained_prefill_graph", handle, enabled)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        __import__("torch").ops.rpu,
        "qwen3_5_enable_execution_reconfigure",
        lambda handle: events.append(("execution_reconfigure", handle)),
        raising=False,
    )
    return model, vlm


def test_g05_exact_vlm_installer_publishes_chunk_envelope_before_ready(monkeypatch, request):
    module = _load_adapter()
    events = []
    model, vlm = _g05_vlm_installer_fixture(monkeypatch, module, events, request)

    assert module.patch_g05_vlm_for_rpu(
        model, _allow_certified_g05_k1=True
    ) is model
    assert model._rpu_g05_vlm_ready is True
    assert events[2:9] == [
        "install",
        ("chunk_envelope", 23, 2176, 256),
        ("linear_acc32", 23, True),
        ("fast_replay", 23, True),
        ("retained_prefill_graph", 23, True),
        ("execution_reconfigure", 23),
        ("graph_cache", 1),
    ]
    assert [event for event in events if event[0] == "chunk_envelope"] == [
        ("chunk_envelope", 23, 2176, 256)
    ]
    assert vlm._rpu_qwen3_5.prefill_capability_envelope == (2176, 256)

    previous = list(events)
    assert module.patch_g05_vlm_for_rpu(
        model, _allow_certified_g05_k1=True
    ) is model
    assert events == previous


def test_g05_generic_vlm_installer_does_not_override_chunk_envelope(monkeypatch, request):
    module = _load_adapter()
    events = []
    model, vlm = _g05_vlm_installer_fixture(monkeypatch, module, events, request)

    assert module.patch_g05_vlm_for_rpu(model) is model
    assert not [event for event in events if event[0] == "chunk_envelope"]


def test_g05_exact_vlm_chunk_envelope_failure_rolls_back_and_poison(monkeypatch, request):
    module = _load_adapter()
    events = []
    model, vlm = _g05_vlm_installer_fixture(monkeypatch, module, events, request)

    def fail_chunk_envelope(_handle, _max_kv_len, _chunk):
        events.append("chunk_failure")
        raise RuntimeError("chunk envelope setter failed")

    monkeypatch.setattr(
        __import__("torch").ops.rpu,
        "qwen3_5_set_chunk_envelope",
        fail_chunk_envelope,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="chunk envelope setter failed"):
        module.patch_g05_vlm_for_rpu(
            model, _allow_certified_g05_k1=True
        )
    assert events.index("clear") < events.index("discard") < events.index("destroy")
    assert not hasattr(vlm, "_rpu_qwen3_5")
    assert not hasattr(model, "_rpu_g05_vlm_ready")
    assert model._rpu_g05_vlm_install_started is True
    with pytest.raises(RuntimeError, match="already attempted"):
        module.patch_g05_vlm_for_rpu(
            model, _allow_certified_g05_k1=True
        )

def test_g05_policy_runtime_ready_aggregates_components_and_exact_hooks(
    monkeypatch,
):
    module = _load_adapter()

    class Model:
        pass

    class Policy:
        pass

    model = Model()
    session = types.SimpleNamespace(
        graph_mode="COMPOSITE_CHILD",
        stats=lambda: {"state": "QUIESCENT"},
    )
    expert = types.SimpleNamespace(
        _rpu_execution_component_id="action_expert",
        _execution_session=session,
    )
    model.action_expert = expert
    model.vision_tower = types.SimpleNamespace(
        _rpu_execution_component_id="vision_encoder",
        _execution_session=session,
    )
    model.vlm = types.SimpleNamespace(
        _rpu_execution_component_id="language_model",
        _execution_session=session,
    )
    model._rpu_g05_prefix_bridge_installed = True
    model._rpu_g05_original_inference_fm = lambda *_args, **_kwargs: None
    model.inference_fm = types.MethodType(module._rpu_g05_inference_fm, model)
    policy = Policy()
    policy.model = model
    policy.processor = types.SimpleNamespace()
    policy._rpu_g05_policy_install_started = True
    policy._rpu_g05_policy_ready = True
    policy._rpu_g05_original_predict_action = lambda batch: batch
    policy.predict_action = types.MethodType(module._rpu_g05_predict_action, policy)
    policy.forward_inference = lambda **_kwargs: {}
    policy._execution_session = session
    policy.reconfigure_rpu_execution = types.MethodType(
        module._rpu_g05_reconfigure_execution, policy
    )
    policy.close_rpu_execution = types.MethodType(
        module._rpu_g05_close_execution, policy
    )
    monkeypatch.setattr(module, "_g05_vlm_runtime_ready", lambda owner: owner is model)
    monkeypatch.setattr(
        module, "_g05_action_runtime_ready", lambda owner: owner is expert
    )

    assert module._g05_policy_runtime_ready(policy)
    model._rpu_g05_prefix_bridge_installed = False
    assert not module._g05_policy_runtime_ready(policy)
    model._rpu_g05_prefix_bridge_installed = True
    model._rpu_g05_vision_install_started = True
    assert not module._g05_policy_runtime_ready(policy)
    del model._rpu_g05_vision_install_started
    policy.predict_action = lambda batch: batch
    assert not module._g05_policy_runtime_ready(policy)
    with pytest.raises(RuntimeError, match="incomplete runtime state"):
        module.patch_g05_policy_for_rpu(policy)

    policy.predict_action = types.MethodType(module._rpu_g05_predict_action, policy)
    policy._rpu_g05_fm_only_customer_release = True
    policy._rpu_g05_original_discrete_action = True
    policy._rpu_g05_original_policy_forward = lambda *_args, **_kwargs: None
    policy._rpu_g05_ensure_sample_list = lambda samples: samples
    policy.forward = types.MethodType(module._rpu_g05_policy_forward, policy)
    policy.discrete_action = False
    model._rpu_g05_vision_ready = True
    model._rpu_g05_certified_k1_vision = True
    model.vlm = types.SimpleNamespace(
        _rpu_execution_component_id="language_model",
        _execution_session=session,
    )
    model._rpu_g05_embed_w_rpu = object()
    model._rpu_g05_proprio_mlp = object()
    model.vision_tower = types.SimpleNamespace(
        _rpu_vision_has_merger=True,
        _rpu_vision_step0=True,
        _rpu_vision_graph_disable=False,
        _rpu_execution_component_id="vision_encoder",
        _execution_session=session,
    )
    model._forward_embed = types.MethodType(module._rpu_g05_forward_embed, model)
    monkeypatch.setattr(module, "_g05_vision_runtime_ready", lambda owner: owner is model)
    assert module._g05_policy_runtime_ready(policy)
    model.vision_tower._rpu_vision_has_merger = False
    assert not module._g05_policy_runtime_ready(policy)
    model.vision_tower._rpu_vision_has_merger = True
    model._rpu_g05_certified_k1_vision = False
    assert not module._g05_policy_runtime_ready(policy)
    policy._rpu_g05_linear_accumulation = "acc16"
    policy._rpu_g05_weight_dtype = "w8a16"
    model._rpu_g05_linear_accumulation = "acc16"
    model._rpu_g05_acc16_experiment = True
    for child in (model.vision_tower, model.vlm, model.action_expert):
        child._rpu_g05_linear_accumulation = "acc16"
    model.vision_tower._rpu_vision_linear_acc32 = False
    for child in (model.vlm, model.action_expert):
        child._rpu_qwen3_5 = types.SimpleNamespace(linear_acc32=False)
    assert module._g05_policy_runtime_ready(policy)
    policy._rpu_g05_weight_dtype = "fp16"
    assert module._g05_policy_runtime_ready(policy)
    policy._rpu_g05_weight_dtype = "invalid"
    assert not module._g05_policy_runtime_ready(policy)
    policy._rpu_g05_weight_dtype = "fp16"
    model.vlm._rpu_qwen3_5.linear_acc32 = True
    assert not module._g05_policy_runtime_ready(policy)
    model.vlm._rpu_qwen3_5.linear_acc32 = False
    model._rpu_g05_certified_k1_vision = True
    assert not module._g05_policy_runtime_ready(policy)


def test_g05_qwen35_prefill_layer_export_splits_last_decoder_and_final_norm():
    qwen_cpp = _QWEN_CPP.read_text()
    qwen_header = _QWEN_HEADER.read_text()
    forward = qwen_cpp[
        qwen_cpp.index("at::Tensor Qwen3_5Model::forward("):
        qwen_cpp.index("void Qwen3_5Model::build_layer_subgraph(")
    ]
    tail = qwen_cpp[
        qwen_cpp.index("void Qwen3_5Model::emit_mlp_and_output("):
        qwen_cpp.index("std::vector<BufferDecl> Qwen3_5Model::declare_buffers(")
    ]

    assert '#include "rpu_runtime_state.h"' in qwen_cpp
    assert "at::Tensor                 per_layer_debug_buf_;" in qwen_header
    assert "at::Tensor                 last_layer_decoder_debug_buf_;" in qwen_header
    assert "if (get_debug_export() && !action_mode_ && seq_len > 1)" in forward
    assert "{num_layers(), batch_size, seq_len, h}" in forward
    assert ".device(at::kPrivateUse1)" in forward
    assert forward.index("per_layer_debug_buf_ = at::empty(") < forward.index(
        "run_all_layers("
    )
    assert forward.index(
        "last_layer_decoder_debug_buf_ = at::empty("
    ) < forward.index("run_all_layers(")
    assert '"_layer_output"' in forward
    assert "g_debug_tensors[key] = per_layer_debug_buf_[layer_idx];" in forward
    assert 'last_prefix + "_decoder_output"' in forward
    assert 'last_prefix + "_final_norm_output"' in forward
    assert ".clone()" not in forward

    decoder_dma = tail.index("// Split the last decoder-layer output")
    final_norm = tail.index("// final_norm fuse at the last layer")
    debug_dma = tail.index("// Export the post-layer residual stream")
    output_dma = tail.index("emit_layer_output_dma(layer_idx, chunk)")
    assert decoder_dma < final_norm < debug_dma < output_dma
    assert "ctx().seq_len > 1" in tail
    assert "+ chunk.offset * h" in tail
    assert 'addr(0, "residual1"), decoder_base, chunk.len * h' in tail
    assert 'addr(0, "residual1"), layer_base, chunk.len * h' in tail


def test_g05_installers_enable_fast_replay_once_per_handle():
    call = "torch.ops.rpu.qwen3_5_set_fast_replay(handle,True)"
    for path in (_ADAPTER, _ACTION_ADAPTER):
        source = path.read_text()
        compile(source, str(path), "exec")
        assert "".join(source.split()).count(call) == 1


def test_g05_packed_vision_op_contract_is_complete():
    adapter = _ADAPTER.read_text()
    vision = _QWEN_VISION.read_text()
    vision_cpp = _QWEN_VISION_CPP.read_text()
    vision_header = (
        _ROOT / "src" / "fused" / "rpu_qwen3_5_vision_model.h"
    ).read_text()
    backend = _RPU_DISPATCH.read_text()

    compile(adapter, str(_ADAPTER), "exec")
    compile(vision, str(_QWEN_VISION), "exec")
    assert "int[] fusion_row_starts=[]" in backend
    assert "int camera_batch_count=1" in backend
    assert "temporal_num_frames, camera_batch_count, direct_fusion" in vision
    assert "[int(x) for x in fusion_run_starts]" in vision
    assert "camera * camera_hidden_rows" in vision_cpp
    assert "frames, group, h" in vision_cpp
    assert "const int64_t composite_heads = group * nq;" in vision_cpp
    assert "std::array<uint64_t, 3> merger_dst_bases_" in vision_header

    temporal_body = vision_cpp.split(
        "void Qwen3_5VisionModel::emit_temporal_layer", 1
    )[1].split("void Qwen3_5VisionModel::emit_spatial_group", 1)[0]
    assert temporal_body.count("emit_layer_input_dma(") == 1
    assert temporal_body.count("rpu_launch_permute4d_0213_spm_kernel(") == 1
    assert 'layer_idx, camera_slice, "t_frame"' in temporal_body
    assert 'addr(0, "t_frame"), addr(0, "t_grouped")' in temporal_body
    assert temporal_body.count("rpu_launch_permute3d_spm_kernel(") == 3
    assert "patch_groups, frames, permute_inner, {1, 0, 2}" in temporal_body
    assert "rpu_launch_spm_scatter_ddr_dma(" not in temporal_body
    assert "rpu_launch_ddr_scatter_spm_dma(" not in temporal_body
    for name in ("temporal_q_stage_", "temporal_k_stage_", "temporal_v_stage_"):
        assert name not in vision_cpp
        assert name not in vision_header
    for name in ('"t_q_stage"', '"t_k_stage"', '"t_out_stage"'):
        assert name in vision_cpp
    decl_order = [
        vision_cpp.index(f'{{"{name}"')
        for name in (
            "t_grouped", "t_frame", "t_q_stage", "t_k_stage",
            "t_sdpa_tmp", "t_out_stage", "t_v",
        )
    ]
    assert decl_order == sorted(decl_order)
    for name in ("q", "k", "v"):
        assert f'{{"{name}",         temporal_stage, 3, 9,' in vision_cpp


def test_g05_temporal_spm_permute_matches_frame_major_stage_oracle():
    import torch

    patch_groups, frames, group, local_dim = 4, 6, 2, 3
    frame_major = torch.arange(
        frames * patch_groups * group * local_dim
    ).view(frames, patch_groups, group * local_dim)
    grouped = frame_major.view(
        1, frames, patch_groups, group * local_dim
    ).permute(0, 2, 1, 3).contiguous().view(
        patch_groups, frames, group, local_dim
    )
    assert torch.equal(
        grouped,
        frame_major.view(frames, patch_groups, group, local_dim).permute(1, 0, 2, 3),
    )

    patch_major = torch.arange(
        patch_groups * frames * group * local_dim
    ).view(patch_groups, frames, group, local_dim)
    permuted = patch_major.reshape(
        patch_groups, frames, group * local_dim
    ).permute(1, 0, 2).contiguous().view(frames, patch_groups * group, local_dim)

    oracle = torch.empty_like(permuted)
    for patch_group in range(patch_groups):
        start = patch_group * group
        oracle[:, start : start + group] = patch_major[patch_group]
    assert torch.equal(permuted, oracle)


def test_g05_spatial_followers_stay_spm_resident_between_owner_callbacks():
    vision_cpp = _QWEN_VISION_CPP.read_text()
    vision_header = (
        _ROOT / "src" / "fused" / "rpu_qwen3_5_vision_model.h"
    ).read_text()
    temporal_body = vision_cpp.split(
        "void Qwen3_5VisionModel::emit_temporal_layer", 1
    )[1].split("void Qwen3_5VisionModel::emit_spatial_group", 1)[0]
    spatial_body = vision_cpp.split(
        "void Qwen3_5VisionModel::emit_spatial_group", 1
    )[1].split("void Qwen3_5VisionModel::emit_kv_first_body", 1)[0]

    owners = [layer for layer in range(24) if layer == 0 or layer % 4 == 3]
    noops = [layer for layer in range(24) if layer not in owners]
    blocks = [
        (owner, min(23, 2 if owner == 0 else owner + 3))
        for owner in owners
    ]
    assert owners == [0, 3, 7, 11, 15, 19, 23]
    assert noops == [
        1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,
        16, 17, 18, 20, 21, 22,
    ]
    assert blocks == [
        (0, 2), (3, 6), (7, 10), (11, 14),
        (15, 18), (19, 22), (23, 23),
    ]

    assert "if (layer_idx != 0 && layer_idx % 4 != 3) return;" in temporal_body
    assert "layer_idx == 0 ? 2 : layer_idx + 3" in temporal_body
    assert temporal_body.count("emit_spatial_group(") == 2
    assert "int first_layer_idx, int last_layer_idx" in vision_header

    assert spatial_body.count("emit_layer_input_dma(") == 1
    assert "emit_layer_input_dma(first_layer_idx, group);" in spatial_body
    assert "for (int layer_idx = first_layer_idx;" in spatial_body
    assert "staged_qkv && layer_idx == first_layer_idx" in spatial_body
    assert spatial_body.count("emit_layer_output_dma(") == 1
    assert "if (layer_idx == last_layer_idx)" in spatial_body
    assert (
        'emit_layer_output_dma(last_layer_idx, group, "residual1");'
        in spatial_body
    )
    assert "((int64_t)layer_idx * current_num_patches_ + offset) * h" in spatial_body
    assert spatial_body.index("emit_layer_output_dma(") < spatial_body.index(
        "if (get_debug_export())"
    )


def test_g05_action_loop_fuses_io_and_fp16_euler_in_one_graph():
    action = _ACTION_ADAPTER.read_text()
    qwen_cpp = _QWEN_CPP.read_text()
    qwen_header = _QWEN_HEADER.read_text()
    backend = _RPU_DISPATCH.read_text()

    assert "torch.ops.rpu.qwen3_5_set_action_io_weights(" in action
    install = action.split("def patch_g05_action_expert_for_rpu(", 1)[1]
    assert '.half().to("rpu")' not in install
    assert "_cpu_stage_weights=True" in install
    assert "torch.ops.rpu.qwen3_5_action_step_forward(" in action
    assert 'op_id="rpu_g05_action_euler_step"' in action
    assert '"rpu_g05_action_euler_loop"' in action
    assert "_rpu_g05_action_loop(" in action
    assert 'velocity.to("cpu").float()' in action
    assert "action -= delta_t * velocity" not in action
    assert "action_input, action_output = action_output, action_input" not in action
    assert "&action_input_live_base_" in qwen_cpp
    assert "&action_output_live_base_" in qwen_cpp
    assert "rpu_launch_ddr_broadcast_spm_dma_mutable(" in qwen_cpp
    assert "rpu_launch_spm_copy_ddr_dma_mutable(" in qwen_cpp
    assert "ValuOpType::ADD, action_euler_scale_" in qwen_cpp
    assert "&Qwen3_5Model::emit_action_input_projection" in qwen_cpp
    assert "&Qwen3_5Model::emit_action_output_projection" in qwen_cpp
    assert "cfg.body_iterations = action_loop_active_ ? action_num_steps_ : 1;" in qwen_cpp
    assert "(ctx().body_iter * rows + row) * variants" in qwen_cpp
    assert "ctx().body_iter == 0" in qwen_cpp
    assert "ctx().body_iter + 1 == action_num_steps_" in qwen_cpp
    assert "action_step_active_ && is_last_layer" in qwen_cpp
    assert "_plan_g05_action_execution(" in action
    assert "*plan.graph_key_words()" in action
    assert "stage_tuple.physical_descriptor" in action
    assert "int(action_route_flags)," in action
    assert "_ACTION_ROUTE_IO | (_ACTION_ROUTE_RTC if rtc_active else 0)" in action
    assert "_ACTION_ROUTE_IO | _ACTION_ROUTE_RTC | _ACTION_ROUTE_TRACE" in action
    assert "_run_action_cold_prime(" in action
    assert "cold-prime one-shot did not return to PASSTHROUGH" in action
    assert 'graph_mode=_GRAPH_BOUNDED_ONESHOT' in action
    assert '"cold_prime"' in action
    assert "action_descriptor_graph_lifecycle_" in qwen_cpp
    assert "FmbGraphLifecycle::BOUNDED_ONESHOT" in qwen_cpp
    assert "action_descriptor_route_flags_" in qwen_cpp
    assert "kFullAttentionSite = 9137328812466618373LL" in qwen_header
    assert "kFullAttentionKvInsertSite = 8503568375190929130LL" in qwen_header
    assert "kActionLocalKvInsertSite" not in qwen_header
    assert '"kv_insert_call_site": 8503568375190929130' in action
    assert '"local_kv_insert_invocation": 0' in action
    assert '"prefix_kv_insert_invocation": 1' in action
    assert "restore_kvinsert_plan(" in qwen_cpp
    assert "kFullAttentionKvInsertSite, manifest.graph_lifecycle," in qwen_cpp
    assert "rpu_kvinsert_segment_plan_from_route_arguments(" not in qwen_cpp
    assert "rpu_launch_insert_kvcache_spm_unified_with_plan(" in qwen_cpp
    assert "set_chunk_size_override(action_len_)" not in qwen_cpp
    assert (
        "int[] prefix_lens, int num_steps, "
        "int[] planned_stage_descriptor) -> Tensor"
    ) in backend
    for op in ("qwen3_5_set_action_io_weights", "qwen3_5_action_step_forward"):
        assert f'm.def("{op}' in backend
        assert f'TORCH_FN(rpu_{op})' in backend
        assert f'm.impl("{op}", torch::CppFunction::makeFallthrough())' in backend


def test_g05_rtc_action_op_keeps_prefix_mutable_and_modulation_segmented():
    action = _ACTION_ADAPTER.read_text()
    qwen_cpp = _QWEN_CPP.read_text()
    qwen_header = (
        _ROOT / "src" / "fused" / "rpu_qwen3_5_model.h"
    ).read_text()
    backend = _RPU_DISPATCH.read_text()
    declarations = _RPU_KERNEL_DECLS.read_text()
    op = "qwen3_5_action_rtc_step_forward"

    assert f"torch.ops.rpu.{op}(" in action
    assert '"rpu_g05_action_rtc_euler_loop"' in action
    schema = next(line for line in backend.splitlines() if f'm.def("{op}' in line)
    for argument in (
        "Tensor action_prefix",
        "int action_prefix_len",
        "Tensor adaptive_mod",
        "int num_steps",
    ):
        assert argument in schema
    assert f'TORCH_FN(rpu_{op})' in backend
    assert f'm.impl("{op}", torch::CppFunction::makeFallthrough())' in backend
    assert f"at::Tensor rpu_{op}(" in declarations
    assert "forward_action_rtc_step(" in qwen_header
    assert "forward_action_step_impl(" in qwen_header
    assert f"at::Tensor rpu_{op}(" in qwen_cpp

    step_impl = qwen_cpp.split(
        "at::Tensor Qwen3_5Model::forward_action_step_impl", 1
    )[1].split("// ── forward:", 1)[0]
    assert "adaptive_mod.dim() == 4" in step_impl
    assert "adaptive_mod.size(2) == 2" in step_impl
    assert "adaptive_mod.size(3) == modulation_width" in step_impl
    assert "action_prefix->data_ptr() != action.data_ptr()" in step_impl
    load_modulation = qwen_cpp.split(
        "void Qwen3_5Model::load_adaptive_mod_row", 1
    )[1].split("void Qwen3_5Model::apply_adaptive_norm", 1)[0]
    assert "action_rtc_active_ ? 2 : 1" in load_modulation
    assert "rpu_launch_ddr_broadcast_spm_dma_mutable(" in load_modulation
    assert "variants * width" in load_modulation

    adaptive_norm = qwen_cpp.split(
        "void Qwen3_5Model::apply_adaptive_norm", 1
    )[1].split("void Qwen3_5Model::apply_adaptive_residual_gate", 1)[0]
    adaptive_gate = qwen_cpp.split(
        "void Qwen3_5Model::apply_adaptive_residual_gate", 1
    )[1].split("void Qwen3_5Model::apply_raw_residual_gate", 1)[0]
    for segmented_body in (adaptive_norm, adaptive_gate):
        assert "action_rtc_prefix_len_ - chunk.offset" in segmented_body
        assert "clean_rows, chunk.len - clean_rows" in segmented_body
        assert "3 * h * DWIDTH" in segmented_body

    action_input = qwen_cpp.split(
        "void Qwen3_5Model::emit_action_input_projection", 1
    )[1].split("void Qwen3_5Model::emit_action_output_projection", 1)[0]
    rtc_reload = action_input.split("if (action_rtc_active_) {", 1)[1].split("}", 1)[0]
    assert "action_rtc_prefix_live_base_" in rtc_reload
    assert "rpu_launch_ddr_broadcast_spm_dma_mutable(" in rtc_reload
    assert "ctx().body_iter" not in rtc_reload
    # Allocation and row offsets are exercised by the actual-native body-table
    # test, including the larger ten-step route and the one-row fallback.


def test_g05_policy_perf_defaults_are_scoped_to_policy_patch():
    source = _ADAPTER.read_text()
    marker = "def patch_g05_policy_for_rpu"
    before, body = source.split(marker, 1)
    compact = "".join(body.split())
    assert 'os.environ.setdefault("RPU_LINEAR_ACC32"' not in before
    assert 'os.environ.setdefault("RPU_LINEAR_ACC32"' not in body
    assert "_allow_certified_g05_k1=rtc_release_fm_only" in compact
    assert 'os.environ.setdefault("RPU_FASTREPLAY_SKIP_SYNC"' not in before
    assert compact.count(
        'os.environ.setdefault("RPU_FASTREPLAY_SKIP_SYNC","1")'
    ) == 1


def test_g05_vision_reuses_qwen35_and_accepts_canonical_temporal(monkeypatch):
    module = _load_adapter()
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    calls = []
    install_order = []
    forward_calls = []
    vision_module = types.ModuleType("rpu_backend.adapters.qwen3_5.vision")

    def install(vision, *, vision_config):
        install_order.append("install")
        calls.append((vision, vision_config))
        def forward(hidden, grid, **kwargs):
            forward_calls.append(kwargs)
            return types.SimpleNamespace(
                last_hidden_state=(hidden, grid),
                pooler_output="pooler",
            )

        vision.forward = forward
        vision._fake_shared_runtime_live = True
        return 7

    vision_module.install_qwen3_5_vision_for_rpu = install
    vision_module._qwen3_5_vision_runtime_complete = (
        lambda owner, *, installed_forward=None: bool(
            getattr(owner, "_fake_shared_runtime_live", False)
            and installed_forward is owner._rpu_g05_vision_forward_impl
        )
    )
    monkeypatch.setitem(__import__("sys").modules, vision_module.__name__, vision_module)

    class FakePE:
        def to(self, *args, **kwargs):
            return self

        def contiguous(self):
            return self

    temporal_module = types.ModuleType("g05.models.g05.qwen35.vision")
    temporal_module._sinusoidal_temporal_pe = lambda *args: FakePE()
    sys_modules = __import__("sys").modules
    for name in ("g05", "g05.models", "g05.models.g05", "g05.models.g05.qwen35"):
        monkeypatch.setitem(sys_modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys_modules, temporal_module.__name__, temporal_module)
    temporal_calls = []
    monkeypatch.setattr(
        __import__("torch").ops.rpu,
        "qwen3_5_vision_set_temporal",
        lambda *args: temporal_calls.append(args),
        raising=False,
    )

    vision = types.SimpleNamespace(
        config=types.SimpleNamespace(
            depth=24,
            num_heads=16,
            hidden_size=1024,
            intermediate_size=4096,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            in_channels=3,
            out_hidden_size=2048,
            hidden_act="gelu_pytorch_tanh",
            temporal_freq=4,
            spacetime_mode="factorized",
            token_drop_layer=None,
            temporal_pe_pretrain_frames=None,
            batch_all_cameras=False,
        ),
        forward=lambda *args: None,
    )
    vision.half = lambda: install_order.append("half") or vision
    model = types.SimpleNamespace(training=False, vision_tower=vision)
    assert module.patch_g05_vision_for_rpu(model) is model
    assert calls == [(vision, vision.config)]
    assert install_order == ["half", "install"]
    assert temporal_calls and temporal_calls[0][0] == 7
    assert vision.forward("hidden", "grid") == (("hidden", "grid"), "pooler")
    assert vision.forward("hidden", "grid", num_frames=6, bsz=1) == (
        ("hidden", "grid"), "pooler"
    )
    assert forward_calls == [
        {"_rpu_temporal_num_frames": 1, "_rpu_camera_batch_count": 1},
        {"_rpu_temporal_num_frames": 6, "_rpu_camera_batch_count": 1},
    ]
    with pytest.raises(NotImplementedError, match="B_camera=1/3, K=1/6"):
        vision.forward("hidden", "grid", num_frames=2, bsz=1)
    assert module.patch_g05_vision_for_rpu(model) is model
    assert len(calls) == 1


@pytest.mark.parametrize("acc16", [False, True])
@pytest.mark.parametrize("weight_dtype", ["fp16", "w8a16"])
def test_g05_release_k1_vision_skips_temporal_programming(monkeypatch, acc16, weight_dtype):
    module = _load_adapter()
    options = {"weight_dtype": weight_dtype, "linear_accumulation": "acc16" if acc16 else "acc32"}
    install_calls = []
    install_order = []
    vision_module = types.ModuleType("rpu_backend.adapters.qwen3_5.vision")

    def install(vision, *, vision_config, _allow_certified_g05_k1=False,
                execution_config=None, _g05_w8a16=False, _g05_acc16_experiment=False):
        assert execution_config == {"vision": {"linear_acc32": not acc16}}
        assert _g05_w8a16 is (weight_dtype == "w8a16")
        assert _g05_acc16_experiment is acc16
        install_order.append("install")
        install_calls.append((vision, vision_config, _allow_certified_g05_k1))
        vision.forward = lambda hidden, grid, **_kwargs: types.SimpleNamespace(
            last_hidden_state=hidden,
            pooler_output=grid,
        )
        vision._fake_shared_runtime_live = True
        return 11

    vision_module.install_qwen3_5_vision_for_rpu = install
    vision_module._qwen3_5_vision_runtime_complete = (
        lambda owner, *, installed_forward=None: bool(
            getattr(owner, "_fake_shared_runtime_live", False)
            and installed_forward is owner._rpu_g05_vision_forward_impl
        )
    )
    monkeypatch.setitem(__import__("sys").modules, vision_module.__name__, vision_module)
    monkeypatch.setattr(
        __import__("torch").ops.rpu,
        "qwen3_5_vision_set_temporal",
        lambda *_args: pytest.fail("K1 temporal_freq=0 must not program temporal PE"),
        raising=False,
    )

    vision = types.SimpleNamespace(
        config=types.SimpleNamespace(
            depth=24,
            num_heads=16,
            hidden_size=1024,
            intermediate_size=4096,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            in_channels=3,
            out_hidden_size=2048,
            hidden_act="gelu_pytorch_tanh",
            temporal_freq=0,
            spacetime_mode="factorized",
            token_drop_layer=None,
            temporal_pe_pretrain_frames=None,
            batch_all_cameras=False,
        ),
        forward=lambda *_args, **_kwargs: None,
    )
    vision.half = lambda: install_order.append("half") or vision
    model = types.SimpleNamespace(training=False, vision_tower=vision)

    vision.config.temporal_freq = 4
    with pytest.raises(NotImplementedError, match="private customer K1 Vision"):
        module.patch_g05_vision_for_rpu(
            model, _allow_certified_g05_k1=True, **options
        )
    assert install_calls == []
    assert not hasattr(model, "_rpu_g05_vision_install_started")
    vision.config.temporal_freq = 0

    vision.config.spatial_merge_size = 4
    with pytest.raises(NotImplementedError, match="canonical Qwen3.5 tower profile"):
        module.patch_g05_vision_for_rpu(
            model, _allow_certified_g05_k1=True, **options
        )
    assert install_calls == []
    assert not hasattr(model, "_rpu_g05_vision_install_started")
    vision.config.spatial_merge_size = 2

    assert module.patch_g05_vision_for_rpu(
        model, _allow_certified_g05_k1=True, **options
    ) is model
    assert vision._rpu_g05_linear_accumulation == ("acc16" if acc16 else "acc32")
    assert install_calls == [(vision, vision.config, True)]
    assert install_order == ["half", "install"]
    assert not hasattr(vision, "_rpu_g05_temporal_pe")
    assert vision.forward("hidden", "grid") == ("hidden", "grid")


def test_g05_vision_publish_failure_rolls_back_wrapper_and_poison(monkeypatch):
    module = _load_adapter()
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    events = []
    vision_module = types.ModuleType("rpu_backend.adapters.qwen3_5.vision")

    def install(vision, *, vision_config, execution_config=None, _allow_certified_g05_k1=False):
        del vision_config
        assert _allow_certified_g05_k1 is True
        events.append("install")
        vision.forward = lambda *_args, **_kwargs: "shared-forward"
        return 13

    vision_module.install_qwen3_5_vision_for_rpu = install
    vision_module._qwen3_5_vision_runtime_complete = lambda _vision: False
    vision_module._rollback_qwen3_5_vision_install = (
        lambda _vision: events.append("rollback")
    )
    monkeypatch.setitem(
        __import__("sys").modules, vision_module.__name__, vision_module
    )
    class InstallInterrupted(BaseException):
        pass

    class Model:
        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if name == "_rpu_g05_vision_ready" and value is True:
                raise InstallInterrupted

    original_forward = lambda *_args, **_kwargs: "original-forward"
    vision = types.SimpleNamespace(
        config=types.SimpleNamespace(
            depth=24,
            num_heads=16,
            hidden_size=1024,
            intermediate_size=4096,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            in_channels=3,
            out_hidden_size=2048,
            hidden_act="gelu_pytorch_tanh",
            temporal_freq=0,
            spacetime_mode="factorized",
            token_drop_layer=None,
            temporal_pe_pretrain_frames=None,
            batch_all_cameras=False,
        ),
        forward=original_forward,
    )
    vision.half = lambda: events.append("half") or vision
    model = Model()
    model.training = False
    model.vision_tower = vision

    with pytest.raises(InstallInterrupted):
        module.patch_g05_vision_for_rpu(
            model, _allow_certified_g05_k1=True
        )

    assert events == [
        "half",
        "install",
        "rollback",
    ]
    assert vision.forward is original_forward
    assert not hasattr(vision, "_rpu_g05_vision_forward_impl")
    assert not hasattr(model, "_rpu_g05_vision_ready")
    assert model._rpu_g05_vision_install_started is True
    with pytest.raises(RuntimeError, match="already attempted"):
        module.patch_g05_vision_for_rpu(model)


def _enable_packed_vision(monkeypatch, module):
    config = (262_144, 32, 256)
    monkeypatch.setattr(module, "_get_lkn_batch_config", lambda: (config, config))


def test_g05_packed_vision_capacity_rejects_post_import_env_drift(monkeypatch):
    module = _load_adapter()
    small = (65_536, 8, 64)
    large = (262_144, 32, 256)

    monkeypatch.setattr(module, "_get_lkn_batch_config", lambda: (small, large))
    assert not module._g05_packed_vision_capacity_enabled()
    monkeypatch.setattr(module, "_get_lkn_batch_config", lambda: (large, small))
    assert not module._g05_packed_vision_capacity_enabled()
    monkeypatch.setattr(module, "_get_lkn_batch_config", lambda: (large, large))
    assert module._g05_packed_vision_capacity_enabled()


def test_g05_packed_vision_wrapper_preserves_camera_major_and_passes_batch(monkeypatch):
    module = _load_adapter()
    torch = __import__("torch")
    calls = []
    monkeypatch.setattr(module, "_get_lkn_batch_config", lambda: None)

    def forward(hidden, grid, **kwargs):
        calls.append((hidden.clone(), grid.clone(), kwargs))
        return types.SimpleNamespace(last_hidden_state=hidden, pooler_output=grid)

    vision = types.SimpleNamespace(_rpu_g05_vision_forward_impl=forward)
    hidden = torch.arange(4608).view(4608, 1)
    grid = torch.tensor([[1, 16, 16]] * 18)

    module._rpu_g05_vision_forward(vision, hidden, grid, num_frames=6, bsz=3)
    fallback_hidden, _, fallback_kwargs = calls.pop()
    assert torch.equal(fallback_hidden, hidden)
    assert fallback_kwargs == {
        "_rpu_temporal_num_frames": 6,
        "_rpu_camera_batch_count": 1,
    }

    _enable_packed_vision(monkeypatch, module)
    module._rpu_g05_vision_forward(vision, hidden, grid, num_frames=6, bsz=3)
    packed_hidden, packed_grid, packed_kwargs = calls.pop()
    assert torch.equal(packed_hidden, hidden)
    assert torch.equal(packed_grid, grid)
    assert packed_kwargs == {
        "_rpu_temporal_num_frames": 6,
        "_rpu_camera_batch_count": 3,
    }


def test_g05_single_frame_vision_pack_is_camera_ordered(monkeypatch):
    module = _load_adapter()
    torch = __import__("torch")
    vision = types.SimpleNamespace(
        patch_size=2,
        spatial_merge_size=2,
        config=types.SimpleNamespace(temporal_patch_size=2),
    )
    first = torch.arange(16).reshape(1, 1, 1, 4, 4)
    patches, grid, splits, frames = module._pack_g05_vision(
        {"head": first, "wrist": first + 100}, vision
    )

    assert patches.shape == (8, 8)
    assert grid.tolist() == [[1, 2, 2], [1, 2, 2]]
    assert splits == [1, 1]
    assert frames == 1
    assert torch.equal(patches[4:] - patches[:4], torch.full((4, 8), 100))
    video = __import__("torch").cat([first + 10 * i for i in range(6)], dim=1)
    video_patches, video_grid, video_splits, video_frames = module._pack_g05_vision(
        {"head": video}, vision
    )
    assert video_patches.shape == (24, 8)
    assert video_grid.shape == (6, 3)
    assert video_splits == [1]
    assert video_frames == 6
    for frame in range(6):
        assert torch.equal(
            video_patches[frame * 4 : (frame + 1) * 4] - video_patches[:4],
            torch.full((4, 8), 10 * frame),
        )

    rgb_video = torch.arange(6 * 3 * 4 * 4).reshape(1, 6, 3, 4, 4)
    rgb_patches, _, _, _ = module._pack_g05_vision(
        {"head": rgb_video.float(), "wrist": rgb_video.float() + 1000}, vision
    )
    rgb_images = torch.cat((rgb_video[0], rgb_video[0] + 1000), dim=0)
    rgb_expected = (
        rgb_images.unsqueeze(2)
        .expand(-1, -1, 2, -1, -1)
        .reshape(12, 2, 3, 1, 2, 2, 1, 2, 2)
        .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        .reshape(48, 24)
    )
    assert rgb_patches.is_contiguous() and rgb_patches.dtype == torch.float16
    assert torch.equal(rgb_patches, rgb_expected.half())

    wide = torch.arange(3 * 4 * 8).reshape(1, 1, 3, 4, 8)
    mixed_patches, mixed_grid, mixed_splits, _ = module._pack_g05_vision(
        {"head": rgb_video[:, :1], "wide": wide}, vision
    )
    assert mixed_patches.shape == (12, 24)
    assert mixed_grid.tolist() == [[1, 2, 2], [1, 2, 4]]
    assert mixed_splits == [1, 2]
    with pytest.raises(NotImplementedError, match="K=1 or 6"):
        module._pack_g05_vision(
            {"head": first.expand(1, 2, 1, 4, 4)}, vision
        )

    _enable_packed_vision(monkeypatch, module)
    canonical_vision = types.SimpleNamespace(
        patch_size=1,
        spatial_merge_size=2,
        config=types.SimpleNamespace(temporal_patch_size=1),
    )
    cameras = {
        f"camera_{camera}": torch.stack(
            [torch.full((1, 16, 16), 100 * camera + 10 * frame)
             for frame in range(6)]
        ).unsqueeze(0).float()
        for camera in range(3)
    }
    packed, packed_grid, packed_splits, packed_frames = module._pack_g05_vision(
        cameras, canonical_vision
    )
    assert packed.shape == (4608, 1)
    assert packed.view(18, 256)[:, 0].tolist() == [
        0, 10, 20, 30, 40, 50,
        100, 110, 120, 130, 140, 150,
        200, 210, 220, 230, 240, 250,
    ]
    assert packed_grid.tolist() == [[1, 16, 16]] * 18
    assert packed_splits == [64, 64, 64]
    assert packed_frames == 6
    assert module._g05_packed_camera_batch_count(
        packed_grid, packed_splits, packed_frames
    ) == 3


def test_g05_canonical_compact_vision_reconstructs_packed_bytes(monkeypatch):
    module = _load_adapter()
    torch = __import__("torch")
    torch.manual_seed(20260802)
    vision = types.SimpleNamespace(
        patch_size=16,
        spatial_merge_size=2,
        config=types.SimpleNamespace(temporal_patch_size=2),
    )
    images = torch.randn(18, 3, 256, 256, dtype=torch.float32)
    cameras = {
        f"camera_{camera}": images[camera * 6 : (camera + 1) * 6].unsqueeze(0)
        for camera in range(3)
    }

    real_cat = torch.cat
    frame_cat_calls = []

    def tracked_cat(tensors, *args, **kwargs):
        tensors = tuple(tensors)
        if len(tensors) == 3 and tuple(tensors[0].shape) == (6, 3, 256, 256):
            frame_cat_calls.append(tensors)
        return real_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", tracked_cat)

    packed, grid, splits, frames = module._pack_g05_vision(
        cameras, vision, compact=False
    )
    compact, compact_grid, compact_splits, compact_frames = (
        module._pack_g05_vision(cameras, vision, compact=True)
    )

    assert compact.shape == (18, 3, 256, 256)
    assert compact.dtype == torch.float16 and compact.is_contiguous()
    assert compact.nbytes * 2 == packed.nbytes
    assert torch.equal(compact_grid, grid)
    assert compact_splits == splits == [64, 64, 64]
    assert compact_frames == frames == 6
    assert frame_cat_calls == []

    independent = {key: value.clone() for key, value in cameras.items()}
    independent_compact, *_ = module._pack_g05_vision(
        independent, vision, compact=True
    )
    assert len(frame_cat_calls) == 1
    assert torch.equal(independent_compact, compact)

    # CPU oracle for the production STEP0: tile temporal planes, patchify,
    # switch plane/patch axes, reproduce the official C/T reinterpretation,
    # then put raster patches into 2x2 merge-group row order.
    reconstructed = []
    for frame_start in range(0, 18, 3):
        chunk = compact[frame_start : frame_start + 3]
        tiled = chunk.reshape(9, 256, 256).repeat(1, 2, 1).reshape(
            18, 256, 256
        )
        patchified = (
            tiled.reshape(288, 16, 16, 16)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(18, 256, 256)
        )
        patch_major = (
            patchified.reshape(3, 6, 256, 256)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        channel_temporal = (
            patch_major.reshape(768, 2, 3, 256)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        merge_rows = (
            channel_temporal.reshape(3, 8, 2, 8, 2, 1536)
            .reshape(24, 2, 8, 3072)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(768, 1536)
        )
        reconstructed.append(merge_rows)
    reconstructed = torch.cat(reconstructed)

    assert torch.equal(reconstructed, packed)
    assert reconstructed[0, ::256][:6].tolist() == [
        compact[0, 0, 0, 0].item(),
        compact[0, 1, 0, 0].item(),
        compact[0, 0, 0, 0].item(),
        compact[0, 2, 0, 0].item(),
        compact[0, 1, 0, 0].item(),
        compact[0, 2, 0, 0].item(),
    ]


def test_g05_compact_vision_rejects_noncanonical_inputs():
    module = _load_adapter()
    torch = __import__("torch")
    vision = types.SimpleNamespace(
        patch_size=16,
        spatial_merge_size=2,
        config=types.SimpleNamespace(temporal_patch_size=2),
    )

    single_frame = {
        f"camera_{camera}": torch.zeros(1, 1, 3, 256, 256)
        for camera in range(3)
    }
    single_packed, *_ = module._pack_g05_vision(
        single_frame, vision, compact=True
    )
    assert single_packed.shape == (3 * 256, 1536)

    two_cameras = {
        f"camera_{camera}": torch.zeros(1, 6, 3, 256, 256)
        for camera in range(2)
    }
    two_camera_packed, *_ = module._pack_g05_vision(
        two_cameras, vision, compact=True
    )
    assert two_camera_packed.shape == (2 * 6 * 256, 1536)


def test_g05_compact_vision_production_contract_is_scoped():
    adapter = _ADAPTER.read_text()
    vision = _QWEN_VISION.read_text()
    vision_cpp = _QWEN_VISION_CPP.read_text()
    kernel_cache = _RPU_KERNEL_CACHE.read_text()
    vision_header = (
        _ROOT / "src" / "fused" / "rpu_qwen3_5_vision_model.h"
    ).read_text()

    assert 'getattr(self.vision_tower, "_rpu_vision_step0", False)' in adapter
    assert "and _g05_packed_vision_capacity_enabled()" in adapter
    assert "compact_input = hidden_states.dim() == 4" in vision
    assert "current_compact_step0_" in vision_cpp
    assert "current_compact_step0_" in vision_header
    emit_step0 = vision_cpp.split(
        "void Qwen3_5VisionModel::emit_step0()", 1
    )[1].split("void Qwen3_5VisionModel::emit_merger()", 1)[0]
    compact_step0 = emit_step0.split(
        "if (current_compact_step0_)", 1
    )[1].split("} else {", 1)[0]
    assert compact_step0.count("rpu_launch_permute4d_0213_spm_kernel(") == 4
    assert compact_step0.count("rpu_launch_ddr_broadcast_spm_dma_mutable(") == 1
    assert "G05_COMPACT_STEP0_FRAMES" in compact_step0
    step0_body = vision_cpp.split(
        "void Qwen3_5VisionModel::emit_step0()", 1
    )[1].split("void Qwen3_5VisionModel::emit_merger()", 1)[0]
    assert step0_body.count("rpu_launch_linear_spm_to_spm_acc16_kernel(") == 1
    assert "patch_input_addr, pe_w_" in step0_body
    assert '"s0_layout_a"' in vision_cpp
    assert '"s0_layout_b"' in vision_cpp
    assert "Persistent" not in compact_step0
    assert '{"transpose_qnbc_c16", RHINO_OP_LIB_PATH}' in kernel_cache
    assert "configure_cache_capacity(int64_t max_seq_len)" in vision_cpp
    geometry = vision_cpp.split(
        "void Qwen3_5VisionModel::configure_execution_geometry(", 1
    )[1].split(
        "int64_t Qwen3_5VisionModel::input_chunk_size_for_geometry(", 1
    )[0]
    generic_branch = geometry.split("    } else {", 1)[1].split(
        "    if (compact_input)", 1
    )[0]
    assert "execution_len <= cache_capacity_" in generic_branch
    assert "current_output_patches_ = execution_len" in generic_branch


def test_g05_data_processor_fast_path_is_exact_and_instance_local():
    module = _load_adapter()
    torch = __import__("torch")

    def make_transform(name, module_name, **attributes):
        cls = type(name, (), {
            "__call__": lambda self, image: self.forward(image),
        })
        cls.__module__ = module_name
        value = cls()
        for key, attribute in attributes.items():
            setattr(value, key, attribute)
        return value

    processor_cls = type("GalaxeaCoTProcessor", (), {})
    processor_cls.__module__ = "g05.data_processor.processor.galaxea_cot_processor"

    def make_processor():
        processor = processor_cls()
        processor.is_train = False
        processor.num_obs_steps = 6
        processor.num_output_cameras = 18
        processor.action_horizon = 32
        processor.shape_meta = {
            "images": [
                {"key": key, "camera_type": camera_type, "shape": [3, 4, 4]}
                for key, camera_type in (
                    ("head", "exterior"),
                    ("left", "wrist_left"),
                    ("right", "wrist_right"),
                )
            ]
        }
        processor.val_transforms = {}
        processor.original_process_calls = 0
        for key in ("head", "left", "right"):
            chain = [
                make_transform(
                    "Resize",
                    "torchvision.transforms.transforms",
                    size=[4, 4],
                    interpolation="bilinear",
                    max_size=None,
                    antialias=True,
                ),
                make_transform(
                    "ToTensor",
                    "g05.data_processor.transforms.image",
                ),
                make_transform(
                    "Normalize",
                    "torchvision.transforms.transforms",
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    inplace=False,
                ),
            ]
            chain[0].forward = types.MethodType(
                lambda self, image: image[..., :4, :4],
                chain[0],
            )
            chain[1].forward = types.MethodType(
                lambda self, image: image.to(torch.float32) / 255.0,
                chain[1],
            )
            chain[2].forward = types.MethodType(
                lambda self, image: (
                    image if self.inplace else image.clone()
                ).sub_(0.5).div_(0.5),
                chain[2],
            )
            processor.val_transforms[key] = chain

        def process_images(self, data):
            self.original_process_calls += 1
            result = {}
            for meta in self.shape_meta["images"]:
                image = data["images"][meta["key"]]
                for transform in self.val_transforms[meta["key"]]:
                    image = transform(image)
                result[meta["camera_type"]] = image
            return result

        def preprocess(self, data):
            copied = deepcopy(data)
            self.gt_image_ptr = copied["images"]["head"].data_ptr()
            return {
                "pixel_values": self.process_images(data),
                "action": copied["action"].clone(),
                "samples": {"task": data["task"]},
            }

        processor.process_images = types.MethodType(process_images, processor)
        processor.preprocess = types.MethodType(preprocess, processor)
        return processor

    def make_data():
        return {
            "images": {
                key: torch.arange(6 * 3 * 8 * 8, dtype=torch.int32)
                .remainder(256)
                .to(torch.uint8)
                .reshape(6, 3, 8, 8)
                for key in ("head", "left", "right")
            },
            "action": torch.arange(4),
            "task": "pick",
        }

    reference = make_processor()
    actual = make_processor()
    reference_data = make_data()
    actual_data = make_data()
    reference_output = reference.preprocess(reference_data)
    original_images = actual_data["images"]
    assert module.patch_g05_data_processor_for_rpu(actual) is actual
    original_preprocess = actual._rpu_g05_original_preprocess
    actual_output = actual.preprocess(actual_data)

    assert reference.gt_image_ptr != reference_data["images"]["head"].data_ptr()
    assert actual.gt_image_ptr == actual_data["images"]["head"].data_ptr()
    assert actual_data["images"] is original_images
    assert torch.equal(reference_output["action"], actual_output["action"])
    assert reference_output["samples"] == actual_output["samples"]
    assert all(
        torch.equal(reference_output["pixel_values"][key], actual_output["pixel_values"][key])
        for key in reference_output["pixel_values"]
    )
    storages = [
        image.untyped_storage().data_ptr()
        for image in actual_output["pixel_values"].values()
    ]
    assert len(set(storages)) == 1
    assert actual.val_transforms["head"][-1].inplace
    assert not actual.val_transforms["left"][-1].inplace
    assert not actual.val_transforms["right"][-1].inplace
    assert actual.original_process_calls == 0
    actual.is_train = True
    actual.process_images(actual_data)
    assert actual.original_process_calls == 1
    actual.is_train = False
    probe = torch.arange(256, dtype=torch.uint8)
    assert torch.equal(
        actual.val_transforms["head"][1].forward(probe),
        probe.to(torch.float32) / 255.0,
    )
    assert module.patch_g05_data_processor_for_rpu(actual) is actual
    assert actual._rpu_g05_original_preprocess is original_preprocess

    mismatched = make_processor()
    mismatched.val_transforms["head"][-1].mean[0] = 0.4
    with pytest.raises(NotImplementedError, match="canonical resize and normalization"):
        module.patch_g05_data_processor_for_rpu(mismatched)
    assert not hasattr(mismatched, "_rpu_g05_original_preprocess")
    assert not any(
        chain[-1].inplace for chain in mismatched.val_transforms.values()
    )


def test_g05_exact_release_processor_and_runner_preserve_bytes_rng_and_views(
    monkeypatch,
):
    module = _load_adapter()
    torch = __import__("torch")
    np = __import__("numpy")

    class Resize:
        def __init__(self):
            self.size = [256, 256]
            self.interpolation = "bilinear"
            self.max_size = None
            self.antialias = True

        def __call__(self, image):
            return image[..., :256, :256].contiguous()

    Resize.__module__ = "torchvision.transforms.transforms"

    class ToTensor:
        def __call__(self, image):
            return image.to(torch.float32) / 255.0

    ToTensor.__module__ = "vla_rtc.transforms.image"

    class Normalize:
        def __init__(self):
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
            self.inplace = False

        def __call__(self, image):
            output = image if self.inplace else image.clone()
            return output.sub_(0.5).div_(0.5)

    Normalize.__module__ = "torchvision.transforms.transforms"

    def preprocess(self, data):
        pixel_values = {}
        for meta in self.shape_meta["images"]:
            key = meta["key"]
            image = data["images"][key]
            for transform in self.val_transforms[key]:
                image = transform(image)
            pixel_values[key] = image
        return {
            "pixel_values": pixel_values,
            "samples": {
                "token": data["token"],
                "task": "pick",
                "frame_shape": (256, 256),
                "metadata": {
                    "frequency": 15,
                    "proprio": data["state"][0],
                    "action": {"dim": 27},
                },
            },
            "proprio": data["state"],
            "idx": data["idx"],
        }

    processor_cls = type("BaseProcessor", (), {"preprocess": preprocess})
    processor_cls.__module__ = "vla_rtc.processors.base_processor"
    processor = processor_cls()
    processor.is_train = False
    processor.num_obs_steps = 1
    processor.num_output_cameras = 3
    processor.action_output_dim = 27
    processor.proprio_output_dim = 27
    processor.pixel_values_format = "dict"
    processor.shape_meta = {
        "images": [
            {"key": key, "raw_shape": raw_shape, "shape": (3, 256, 256)}
            for key, raw_shape in module._G05_RELEASE_IMAGE_SPECS
        ]
    }
    processor.val_transforms = {
        key: [Resize(), ToTensor(), Normalize()]
        for key, _shape in module._G05_RELEASE_IMAGE_SPECS
    }

    def make_torch_obs():
        return {
            "images": {
                key: torch.arange(
                    3 * raw_shape[1] * raw_shape[2], dtype=torch.int32
                ).remainder(256).to(torch.uint8).reshape(1, *raw_shape)
                for key, raw_shape in module._G05_RELEASE_IMAGE_SPECS
            },
            "state": torch.arange(27, dtype=torch.float32).reshape(1, 27),
            "token": torch.tensor([7, 9]),
            "idx": 3,
        }

    def assert_tree_equal(expected, actual):
        if isinstance(expected, torch.Tensor):
            assert isinstance(actual, torch.Tensor)
            assert torch.equal(expected, actual)
            return
        if isinstance(expected, dict):
            assert expected.keys() == actual.keys()
            for key in expected:
                assert_tree_equal(expected[key], actual[key])
            return
        if isinstance(expected, (list, tuple)):
            assert isinstance(actual, type(expected))
            assert len(expected) == len(actual)
            for expected_item, actual_item in zip(expected, actual):
                assert_tree_equal(expected_item, actual_item)
            return
        assert expected == actual

    reference_obs = make_torch_obs()
    actual_obs = make_torch_obs()
    raw_snapshots = {
        key: image.clone() for key, image in actual_obs["images"].items()
    }
    reference = processor.preprocess(reference_obs)
    assert module.patch_g05_release_processor_for_rpu(processor) is processor
    actual = processor.preprocess(actual_obs)
    assert_tree_equal(reference, actual)
    assert module._g05_release_processor_runtime_ready(processor)
    assert module.patch_g05_release_processor_for_rpu(processor) is processor
    for key, snapshot in raw_snapshots.items():
        assert torch.equal(actual_obs["images"][key], snapshot)
    pixels = list(actual["pixel_values"].values())
    storage = pixels[0].untyped_storage().data_ptr()
    assert all(image.untyped_storage().data_ptr() == storage for image in pixels)
    assert [image.storage_offset() for image in pixels] == [
        0,
        pixels[0].numel(),
        2 * pixels[0].numel(),
    ]

    def dict_apply(value, function):
        if isinstance(value, dict):
            return {key: dict_apply(item, function) for key, item in value.items()}
        return function(value)

    def infer_collated_batch_size(value):
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return value.shape[0]
        if isinstance(value, dict):
            for child in value.values():
                batch_size = infer_collated_batch_size(child)
                if batch_size is not None:
                    return batch_size
        if isinstance(value, (list, tuple)):
            if value and all(isinstance(item, str) for item in value):
                return len(value)
            for child in value:
                batch_size = infer_collated_batch_size(child)
                if batch_size is not None:
                    return batch_size
        return None

    def uncollate_value(value, index, batch_size):
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and value.shape[0] == batch_size:
                item = value[index]
                return item.item() if item.ndim == 0 else item
            return value.item() if value.ndim == 0 else value
        if isinstance(value, dict):
            return {
                key: uncollate_value(child, index, batch_size)
                for key, child in value.items()
            }
        if isinstance(value, tuple):
            return tuple(
                uncollate_value(child, index, batch_size) for child in value
            )
        if isinstance(value, list):
            if value and all(isinstance(item, str) for item in value):
                return value[index]
            if value and all(isinstance(item, torch.Tensor) for item in value):
                return tuple(
                    uncollate_value(item, index, batch_size) for item in value
                )
            if len(value) == batch_size:
                return value[index]
            return [
                uncollate_value(child, index, batch_size) for child in value
            ]
        return value

    def ensure_sample_list(samples):
        if isinstance(samples, list) and all(
            isinstance(sample, dict) for sample in samples
        ):
            return samples
        if not isinstance(samples, dict):
            return [samples]
        batch_size = infer_collated_batch_size(samples)
        if batch_size is None:
            return [dict(samples)]
        return [
            uncollate_value(samples, index, batch_size)
            for index in range(batch_size)
        ]

    def original_preprocess_batch(self, obs):
        converted = dict_apply(
            obs,
            lambda value: (
                torch.from_numpy(value.copy())
                if isinstance(value, np.ndarray)
                else value
            ),
        )
        batch = torch.utils.data.default_collate(
            [self.processor.preprocess(converted)]
        )
        return dict_apply(
            batch,
            lambda value: (
                value.to(self.device) if isinstance(value, torch.Tensor) else value
            ),
        )

    runner_module_name = "vla_rtc.replay"
    runner_module = types.ModuleType(runner_module_name)
    runner_module.dict_apply = dict_apply
    runner_module.np = np
    monkeypatch.setitem(__import__("sys").modules, runner_module_name, runner_module)
    original_preprocess_batch.__module__ = runner_module_name
    runner_cls = type(
        "ReplayRunner", (), {"_preprocess_batch": original_preprocess_batch}
    )
    runner_cls.__module__ = runner_module_name
    policy = object()
    monkeypatch.setattr(
        module, "_g05_policy_runtime_ready", lambda owner: owner is policy
    )
    runner = runner_cls()
    runner.policy = policy
    runner.processor = processor
    runner.device = "cpu"
    numpy_obs = {
        "images": {
            key: image.numpy().copy()
            for key, image in make_torch_obs()["images"].items()
        },
        "state": np.arange(27, dtype=np.float32).reshape(1, 27),
        "token": np.asarray([7, 9], dtype=np.int64),
        "idx": 3,
    }
    torch.manual_seed(123)
    reference_batch = runner._preprocess_batch(numpy_obs)
    reference_rng = torch.get_rng_state()
    torch.manual_seed(123)
    assert module.patch_g05_release_runner_for_rpu(runner) is runner
    actual_batch = runner._preprocess_batch(numpy_obs)
    actual_rng = torch.get_rng_state()
    assert torch.equal(reference_rng, actual_rng)
    for key in reference_batch:
        if key != "samples":
            assert_tree_equal(reference_batch[key], actual_batch[key])
    assert_tree_equal(
        ensure_sample_list(reference_batch["samples"]), actual_batch["samples"]
    )
    batched_pixels = list(actual_batch["pixel_values"].values())
    batched_storage = batched_pixels[0].untyped_storage().data_ptr()
    assert all(
        image.untyped_storage().data_ptr() == batched_storage
        for image in batched_pixels
    )
    assert all(image.is_contiguous() for image in batched_pixels)
    assert [image.storage_offset() for image in batched_pixels] == [
        0,
        batched_pixels[0].numel(),
        2 * batched_pixels[0].numel(),
    ]
    assert module._g05_release_runner_runtime_ready(runner)
    assert module.patch_g05_release_runner_for_rpu(runner) is runner


def test_g05_batch_one_collate_uses_views_and_falls_back():
    module = _load_adapter()
    torch = __import__("torch")
    inferencer_cls = type("PolicyInferencer", (), {})
    inferencer_cls.__module__ = "g05.models.g05.inferencer"
    inferencer = inferencer_cls()
    inferencer.device = "cpu"
    inferencer.policy = types.SimpleNamespace(_rpu_g05_policy_ready=False)
    inferencer.processor = types.SimpleNamespace(
        processors={
            "galaxea_r1lite": types.SimpleNamespace(
                _rpu_g05_data_processor_ready=True
            )
        }
    )
    fallback_calls = []

    def original_collate(samples, padding_input_id):
        fallback_calls.append((samples, padding_input_id))
        return {"fallback": True}

    inferencer._collate = original_collate
    inferencer._prepare = lambda obs: types.SimpleNamespace(obs_dict=obs)
    inferencer.infer_with_timing = lambda obs: ({"fallback_timing": obs}, {})
    inferencer.forward_batch = lambda batch: {"fallback_batch": batch}
    pixels = torch.arange(6 * 3 * 4 * 4).reshape(6, 3, 4, 4)
    proprio = torch.arange(12).reshape(6, 2)
    sample = {
        "pixel_values": {"exterior": pixels},
        "proprio": proprio,
        "action_dim_is_pad": torch.zeros(4, dtype=torch.bool),
        "samples": {"template": "<image>", "action": torch.arange(2)},
        "idx": 7,
        "task": "pick",
        "embodiment": "galaxea_r1lite",
        "frequency": 15.0,
    }

    assert module.patch_g05_inferencer_for_rpu(inferencer) is inferencer
    original = inferencer._rpu_g05_original_collate
    assert inferencer.forward_batch({"x": 1}) == {"fallback_batch": {"x": 1}}
    inferencer.policy = types.SimpleNamespace(
        _rpu_g05_policy_ready=True,
        predict_action=lambda batch: {"fast_batch": batch},
    )
    assert inferencer.forward_batch({"x": 2}) == {"fast_batch": {"x": 2}}
    batch = inferencer._collate([sample], padding_input_id=0)
    assert not fallback_calls
    assert batch["pixel_values"]["exterior"].shape == (1, 6, 3, 4, 4)
    assert batch["pixel_values"]["exterior"].data_ptr() == pixels.data_ptr()
    assert batch["proprio"].shape == (1, 6, 2)
    assert batch["proprio"].data_ptr() == proprio.data_ptr()
    assert batch["samples"] == [sample["samples"]]
    assert batch["idx"].tolist() == [7]
    assert batch["task"] == ["pick"]
    assert batch["sample_meta"] == [{
        "idx": 7,
        "task": "pick",
        "embodiment": "galaxea_r1lite",
        "frequency": 15.0,
    }]

    assert inferencer._collate([sample, sample], 0) == {"fallback": True}
    assert inferencer._collate([{
        **sample,
        "input_ids": torch.ones(1),
        "labels": torch.ones(1),
        "attention_mask": torch.ones(1),
    }], 0) == {"fallback": True}
    assert inferencer._collate([{
        **sample,
        "gt_action": {"left": torch.ones(2)},
    }], 0) == {"fallback": True}
    assert len(fallback_calls) == 3
    assert module.patch_g05_inferencer_for_rpu(inferencer) is inferencer
    assert inferencer._rpu_g05_original_collate is original


def test_g05_prepare_reuses_only_dead_padded_action_metadata(monkeypatch):
    module = _load_adapter()
    torch = __import__("torch")

    def exact_type(module_name, name):
        cls = type(name, (), {})
        cls.__module__ = module_name
        return cls

    action_filter = exact_type(
        "g05.data_processor.transforms.action_filter", "BaseActionFilter"
    )()
    transform = exact_type(
        "g05.data_processor.transforms.relative_action", "RelativeJointTransform"
    )()
    transform.fast_forward = True
    normalizer = exact_type("g05.utils.data.normalizer", "LinearNormalizer")()
    merger = exact_type(
        "g05.data_processor.transforms.action_state_merger",
        "GroupedPaddingMerger",
    )()
    merger._raw_max_action_shape_meta = {"arm": 5}
    merger.merge_spec = {"control": ["arm"]}
    merger.merge = True
    active_builder = exact_type(
        "g05.data_processor.processor.samples_builder", "BaseSamplesBuilder"
    )()
    active_builder.template = "State: <proprio_proprio_!>;<EOC><action_action>"
    builder = exact_type(
        "g05.data_processor.processor.samples_builder", "MixedSamplesBuilder"
    )()
    builder._training = False
    builder._eval_builder = active_builder
    sub_processor = types.SimpleNamespace(
        _rpu_g05_data_processor_ready=True,
        is_train=False,
        action_filter=action_filter,
        action_state_transforms=[transform],
        _normalizer=normalizer,
        action_state_merger=merger,
        samples_builder=builder,
        _vlm_input_action_normalizer=None,
    )

    model_processor = types.SimpleNamespace()
    model_processor.batchify_action = True
    model_processor._rpu_g05_original_encode_inference = lambda *_args, **_kwargs: None
    model_processor._prepare_segments = lambda _samples, control_flag: [[]]
    model_processor.encode_inference = types.MethodType(
        module._rpu_g05_encode_inference, model_processor
    )
    policy = types.SimpleNamespace(
        _rpu_g05_policy_ready=True,
        processor=model_processor,
    )

    calls = []

    def original_prepare(obs):
        calls.append("action" in obs)
        sample = {
            "pixel_values": obs["frame"].clone(),
            "task": obs["task"],
            "samples": {
                "template": active_builder.template,
                "command": obs["task"],
            },
        }
        if "action" in obs:
            action_dim_is_pad = torch.zeros(27, dtype=torch.bool)
            action_op_mask = torch.ones(27, dtype=torch.bool)
            sample.update({
                "action_dim_is_pad": action_dim_is_pad,
                "action_op_mask": action_op_mask,
                "action_parts_meta": {"arm": 5},
            })
            sample["samples"]["action"] = {
                "value": torch.zeros(32, 27),
                "action_dim_is_pad": action_dim_is_pad,
                "action_op_mask": action_op_mask,
            }
        return types.SimpleNamespace(
            obs_dict=obs,
            sample=sample,
            sub_processor=sub_processor,
        )

    inferencer_cls = exact_type("g05.models.g05.inferencer", "PolicyInferencer")
    inferencer = inferencer_cls()
    inferencer.device = "cpu"
    inferencer.policy = policy
    inferencer.processor = sub_processor
    inferencer._prepare = original_prepare
    inferencer.infer_with_timing = lambda obs_dicts: (
        [inferencer._prepare(obs) for obs in obs_dicts],
        {},
    )
    inferencer._collate = lambda samples, padding_input_id: samples
    inferencer.forward_batch = lambda batch: batch

    inferencer_module = types.ModuleType("g05.models.g05.inferencer")
    inferencer_module.resolve_processor = lambda processor, _obs: processor
    monkeypatch.setitem(__import__("sys").modules, inferencer_module.__name__, inferencer_module)
    module.patch_g05_inferencer_for_rpu(inferencer)

    def observation(value, *, padded=True, width=5):
        return {
            "frame": torch.tensor([value]),
            "task": f"task-{value}",
            "action": {"arm": torch.full((32, width), float(value))},
            "action_is_pad": torch.full((32,), padded, dtype=torch.bool),
        }

    first_obs = observation(1)
    first = inferencer._prepare(first_obs)
    second_obs = observation(2)
    second = inferencer._prepare(second_obs)
    assert calls == [True, False]
    assert second.obs_dict is second_obs
    assert "action" in second_obs
    assert second.sample["task"] == "task-2"
    assert second.sample["samples"]["command"] == "task-2"
    assert torch.equal(second.sample["action_dim_is_pad"], first.sample["action_dim_is_pad"])
    assert (
        second.sample["action_dim_is_pad"].data_ptr()
        != first.sample["action_dim_is_pad"].data_ptr()
    )

    second.sample["action_op_mask"].zero_()
    third = inferencer._prepare(observation(3))
    assert calls == [True, False, False]
    assert bool(third.sample["action_op_mask"].all())

    inferencer._prepare(observation(4, padded=False))
    inferencer._prepare(observation(5, width=6))
    assert calls[-2:] == [True, True]

    inferencer.infer_with_timing([observation(6), observation(7)])
    assert calls[-2:] == [True, True]

    transform.fast_forward = False
    inferencer._prepare(observation(8))
    assert calls[-1] is True


def test_g05_embedding_cache_is_exact_and_keeps_fusion_targets_fresh(monkeypatch):
    module = _load_adapter()
    torch = __import__("torch")
    token_module = types.ModuleType("vla_rtc.models.g05.mask_helper")
    token_module.TOKEN_INDEX = types.SimpleNamespace(
        IMAGE_TOKEN_INDEX=7,
        PROPRIO_TOKEN_INDEX=8,
    )
    vision_module = types.ModuleType("rpu_backend.adapters.qwen3_5.vision")
    vision_module._visual_run_starts = lambda *_args: [1]
    modules = __import__("sys").modules
    for name in ("vla_rtc", "vla_rtc.models", "vla_rtc.models.g05"):
        monkeypatch.setitem(modules, name, types.ModuleType(name))
    monkeypatch.setitem(modules, token_module.__name__, token_module)
    monkeypatch.setitem(modules, vision_module.__name__, vision_module)

    original_to = torch.Tensor.to

    def fake_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs = dict(kwargs, device="cpu")
        return original_to(tensor, *args, **kwargs)

    gather_calls = []

    def gather(weight, ids):
        gather_calls.append(ids.clone())
        return weight[ids.long()]

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    monkeypatch.setattr(
        torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False
    )
    monkeypatch.setattr(torch.ops.rpu, "gather_embedding", gather, raising=False)
    monkeypatch.setattr(
        module,
        "_pack_g05_vision",
        lambda *_args, **_kwargs: (
            torch.zeros(1), torch.tensor([[1, 1, 1]]), [1], 1
        ),
    )
    monkeypatch.setattr(module, "_g05_packed_camera_batch_count", lambda *_args: 1)

    def vision_forward(*_args, **kwargs):
        kwargs["_rpu_fusion_target"][0, 1].fill_(99)

    mask_helper_cls = type("MaskHelper", (), {})
    mask_helper_cls.__module__ = token_module.__name__
    model = types.SimpleNamespace(
        image_token_index=9,
        mask_helper=mask_helper_cls(),
        proprio_embedder=types.SimpleNamespace(state_token_id=None),
        vision_tower=types.SimpleNamespace(
            _rpu_g05_vision_forward_impl=vision_forward
        ),
        _rpu_g05_embed_w_rpu=torch.arange(60).reshape(10, 6).half(),
    )
    input_ids = torch.tensor([[2, 9, 3]])
    token_types = torch.tensor([[1, 7, 1]])
    first = module._rpu_g05_forward_embed(
        model, input_ids, token_types, {"camera": torch.zeros(1)}
    )
    second = module._rpu_g05_forward_embed(
        model, input_ids.clone(), token_types, {"camera": torch.ones(1)}
    )
    changed = input_ids.clone()
    changed[0, 0] = 4
    third = module._rpu_g05_forward_embed(
        model, changed, token_types, {"camera": torch.ones(1)}
    )

    assert len(gather_calls) == 2
    assert torch.equal(gather_calls[0], input_ids.reshape(-1).to(torch.int32))
    assert torch.equal(gather_calls[1], changed.reshape(-1).to(torch.int32))
    assert torch.equal(first, second)
    assert len({first.data_ptr(), second.data_ptr(), third.data_ptr()}) == 3
    first.zero_()
    assert not torch.equal(first, second)


def test_g05_reuses_one_rpu_cache_across_turns(monkeypatch):
    module = _load_adapter()
    torch = __import__("torch")
    reset_inputs = []

    class FakeRPUCache:
        created = []

        def __init__(self):
            self.position = 0
            self._seen_tokens = 0
            self.max_seq_len = 32
            self.k_caches = [torch.zeros(1)]
            self.v_caches = [torch.zeros(1)]
            self.recurrent_states = [torch.zeros(1)]
            self.conv_states = [torch.zeros(1)]
            self.__class__.created.append(self)

        @classmethod
        def from_config(cls, config, max_seq_len):
            del config, max_seq_len
            return cls()

        @staticmethod
        def reset_to_position(cache, position):
            cache.position = position

    cache_module = types.ModuleType("rpu_backend.api.qwen3_5_cache")
    cache_module.Qwen3_5Cache = FakeRPUCache
    monkeypatch.setitem(__import__("sys").modules, cache_module.__name__, cache_module)
    base_cache_module = types.ModuleType("rpu_backend.api.cache")
    base_cache_module.RPUCache = FakeRPUCache
    monkeypatch.setitem(
        __import__("sys").modules, base_cache_module.__name__, base_cache_module
    )

    text_module = types.ModuleType("rpu_backend.adapters.qwen3_5.text")

    def run_qwen3_5_text(vlm, hidden, cache, *, position_ids):
        del vlm, position_ids
        reset_inputs.append(tuple(
            values[0].clone()
            for values in (
                cache.k_caches,
                cache.v_caches,
                cache.recurrent_states,
                cache.conv_states,
            )
        ))
        cache.k_caches[0].fill_(3)
        cache.v_caches[0].fill_(4)
        cache.recurrent_states[0].fill_(5)
        cache.conv_states[0].fill_(6)
        cache.position += hidden.shape[1]
        return hidden

    text_module.run_qwen3_5_text = run_qwen3_5_text
    text_module._normalize_prefill_position_ids = lambda *args, **kwargs: None
    monkeypatch.setitem(__import__("sys").modules, text_module.__name__, text_module)
    monkeypatch.setattr(
        __import__("torch").ops.rpu,
        "qwen3_5_set_mrope_position_delta",
        lambda *args: None,
        raising=False,
    )

    class FakeTensor:
        device = types.SimpleNamespace(type="cpu")

        def __init__(self, seq_len):
            self.shape = (1, seq_len, 2048)

        def to(self, *args, **kwargs):
            return self

        def contiguous(self):
            return self

    class Facade:
        def num_items(self):
            return 0

    owner = types.SimpleNamespace(
        training=False,
        ar_helper=types.SimpleNamespace(block_wise_autoregressive=False),
    )
    vlm = types.SimpleNamespace(
        config=object(),
        _rpu_g05_max_seq_len=32,
        _rpu_g05_owner=lambda: owner,
        _rpu_qwen3_5=types.SimpleNamespace(handle=1),
    )
    first, second, third = Facade(), Facade(), Facade()
    kwargs = dict(
        attention_mask=None,
        position_ids=None,
        return_kv_cache=True,
    )

    module._rpu_g05_vlm_forward(
        vlm, inputs_embeds=FakeTensor(3), kv_cache=first, **kwargs
    )
    module._rpu_g05_vlm_forward(
        vlm, inputs_embeds=FakeTensor(1), kv_cache=second, **kwargs
    )
    module._rpu_g05_vlm_forward(
        vlm, inputs_embeds=FakeTensor(3), kv_cache=third, **kwargs
    )

    assert len(FakeRPUCache.created) == 1
    assert first._rpu_qwen3_5_cache is second._rpu_qwen3_5_cache
    assert second._rpu_qwen3_5_cache is third._rpu_qwen3_5_cache
    assert [float(value) for value in reset_inputs[1]] == [3.0, 4.0, 0.0, 0.0]
    assert [float(value) for value in reset_inputs[2]] == [3.0, 4.0, 5.0, 6.0]
    assert third.num_items() == 3
    owner.training = True
    with pytest.raises(NotImplementedError, match="inference only"):
        module._rpu_g05_vlm_forward(
            vlm, inputs_embeds=FakeTensor(1), kv_cache=third, **kwargs
        )
    owner.training = False
    with pytest.raises(RuntimeError, match="older turn"):
        module._rpu_g05_vlm_forward(
            vlm, inputs_embeds=FakeTensor(1), kv_cache=first, **kwargs
        )


def test_g05_rejects_noncausal_block_mask():
    module = _load_adapter()
    torch = __import__("torch")
    causal = torch.triu(torch.full((1, 1, 3, 3), -1.0), diagonal=1)
    module._check_causal_mask(causal, query_len=3, past_len=0)
    with pytest.raises(NotImplementedError, match="BAR/block masks"):
        module._check_causal_mask(torch.zeros_like(causal), query_len=3, past_len=0)


def test_g05_mask_fallback_matches_customer_release_signature():
    module = _load_adapter()
    torch = __import__("torch")
    calls = []

    def original(input_ids, attention_mask, kv_len=0, dtype=torch.float32):
        calls.append((input_ids, attention_mask, kv_len, dtype))
        return "mask", "positions"

    model = types.SimpleNamespace(
        training=False,
        _rpu_g05_original_build_causal_mask_and_position_ids=original,
    )
    input_ids = torch.tensor([[4]])
    attention_mask = torch.tensor([[1]])

    assert module._rpu_g05_build_causal_mask_and_position_ids(
        model,
        input_ids,
        attention_mask,
        kv_len=7,
        dtype=torch.float16,
    ) == ("mask", "positions")
    assert calls == [(input_ids, attention_mask, 7, torch.float16)]


def test_g05_prefill_skips_dense_mask_and_caches_exact_positions():
    module = _load_adapter()
    torch = __import__("torch")
    calls = []
    fallbacks = []

    class Helper:
        _mrope_position_deltas = None
        _last_T_pos = None

        def _build_mrope_position_ids(self, attention_mask, device):
            calls.append(attention_mask.clone())
            self._mrope_position_deltas = torch.tensor([[7]], device=device)
            self._last_T_pos = torch.tensor([11], device=device)
            return attention_mask.unsqueeze(0).expand(3, -1, -1).clone()

    def original(*args, **kwargs):
        fallbacks.append((args, kwargs))
        return "mask", "positions"

    helper = Helper()
    model = types.SimpleNamespace(
        training=False,
        mask_helper=helper,
        vision_tower=types.SimpleNamespace(spatial_merge_size=2),
        _cached_image_grid_thw=torch.tensor([[1, 4, 4]]),
        _rpu_g05_original_build_causal_mask_and_position_ids=original,
    )
    input_ids = torch.tensor([[4, 5, 6]])
    token_types = torch.tensor([[1, 2, 1]])

    mask, first = module._rpu_g05_build_causal_mask_and_position_ids(
        model, input_ids, token_types
    )
    assert mask is None
    helper._mrope_position_deltas = torch.tensor([[-1]])
    helper._last_T_pos = torch.tensor([-1])
    _, second = module._rpu_g05_build_causal_mask_and_position_ids(
        model, input_ids, token_types.clone()
    )
    assert len(calls) == 1
    assert torch.equal(first, second)
    assert first.data_ptr() != second.data_ptr()
    assert helper._mrope_position_deltas.item() == 7
    assert helper._last_T_pos.item() == 11

    changed = token_types.clone()
    changed[0, 1] = 1
    module._rpu_g05_build_causal_mask_and_position_ids(model, input_ids, changed)
    assert len(calls) == 2
    model.training = True
    assert module._rpu_g05_build_causal_mask_and_position_ids(
        model, input_ids, changed
    ) == ("mask", "positions")
    assert len(fallbacks) == 1


def test_g05_prefill_accepts_release_mask_helper_without_last_t():
    module = _load_adapter()
    torch = __import__("torch")

    class ReleaseHelper:
        _mrope_position_deltas = None

        def _build_mrope_position_ids(self, attention_mask, device):
            self._mrope_position_deltas = torch.tensor([[3]], device=device)
            return attention_mask.unsqueeze(0).expand(3, -1, -1).clone()

    helper = ReleaseHelper()
    model = types.SimpleNamespace(
        training=False,
        mask_helper=helper,
        vision_tower=types.SimpleNamespace(spatial_merge_size=2),
        _cached_image_grid_thw=torch.tensor([[1, 4, 4]]),
        _rpu_g05_original_build_causal_mask_and_position_ids=lambda *args, **kwargs: None,
    )
    input_ids = torch.tensor([[4, 5, 6]])
    token_types = torch.tensor([[1, 2, 1]])

    _, first = module._rpu_g05_build_causal_mask_and_position_ids(
        model, input_ids, token_types
    )
    helper._mrope_position_deltas = torch.tensor([[-1]])
    _, second = module._rpu_g05_build_causal_mask_and_position_ids(
        model, input_ids, token_types.clone()
    )

    assert torch.equal(first, second)
    assert helper._mrope_position_deltas.item() == 3
    assert not hasattr(helper, "_last_T_pos")


def test_g05_policy_rejects_discrete_only_before_swizzle(monkeypatch):
    module = _load_adapter()
    monkeypatch.setattr(
        module,
        "patch_g05_vlm_for_rpu",
        lambda *args, **kwargs: pytest.fail("must reject before swizzling weights"),
    )
    policy = types.SimpleNamespace(
        training=False,
        continuous_action=False,
        discrete_action=True,
        model=object(),
    )

    with pytest.raises(NotImplementedError, match="discrete_action=false"):
        module.patch_g05_policy_for_rpu(policy, max_seq_len=1024)


def test_g05_nonrelease_mixed_action_policy_is_rejected_before_swizzle(monkeypatch):
    module = _load_adapter()
    monkeypatch.setattr(
        module,
        "patch_g05_vlm_for_rpu",
        lambda *args, **kwargs: pytest.fail("must reject before swizzling weights"),
    )
    policy_cls = type("G05PolicyQwen35", (), {})
    policy_cls.__module__ = "downstream_copy.models.g05.g05_policy_qwen35"
    policy = policy_cls()
    policy.training = False
    policy.continuous_action = True
    policy.discrete_action = True
    policy.return_continuous_action = True
    policy.predict_cot = False
    policy.model = types.SimpleNamespace(use_training_rtc=True)

    with pytest.raises(NotImplementedError, match="discrete"):
        module.patch_g05_policy_for_rpu(policy, max_seq_len=1024)


@pytest.mark.parametrize("exact_perf_override", [None, "0", "true"])
@pytest.mark.parametrize("weight_dtype,action_weight_dtype,linear_accumulation", [
    (weight, action, mode)
    for weight, action in (("fp16", "fp16"), ("w8a16", "w8a16"), ("w8a16", "nvfp4a16"))
    for mode in ("acc32", "acc16")
])
def test_g05_exact_release_mixed_flags_select_fm_only_and_forward_rtc(
    monkeypatch, exact_perf_override, linear_accumulation, weight_dtype, action_weight_dtype
):
    exact_perf_flags = ("RPU_LINEAR_ACC32",)
    for name in exact_perf_flags:
        if exact_perf_override is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, exact_perf_override)
    module = _load_adapter()
    torch = __import__("torch")
    monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    original_to = torch.Tensor.to

    def fake_to(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            args = ("cpu", *args[1:])
        if kwargs.get("device") == "rpu":
            kwargs = dict(kwargs, device="cpu")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_to)

    expected_precision = {}
    if weight_dtype != "fp16":
        expected_precision["weight_dtype"] = weight_dtype
    if linear_accumulation != "acc32":
        expected_precision["linear_accumulation"] = linear_accumulation
    policy_precision = dict(expected_precision)
    if action_weight_dtype != weight_dtype:
        policy_precision["action_weight_dtype"] = action_weight_dtype
    monkeypatch.setattr(module, "require_g05_action_nvfp4_native", lambda: None)
    patch_calls = []
    forward_calls = []
    fallback_calls = []
    ensure_calls = []
    monkeypatch.setattr(
        module,
        "_preflight_g05_vlm",
        lambda model: (patch_calls.append(("vlm_preflight",)), model.vlm)[1],
    )
    monkeypatch.setattr(
        module,
        "_check_g05_action_profile",
        lambda _config: patch_calls.append(("action_preflight",)),
    )
    monkeypatch.setattr(
        module,
        "_preflight_g05_vision",
        lambda model: patch_calls.append(("vision_preflight",)),
    )
    def patch_vlm(
        model, *, max_seq_len, _allow_certified_g05_k1=False,
        _execution_config=None, **precision,
    ):
        assert precision == expected_precision
        assert _allow_certified_g05_k1 is True
        assert _execution_config == {
            "prefill": {"chunk_size": "auto", "padding_budget": 64}
        }
        patch_calls.append(("vlm", model, max_seq_len))

    monkeypatch.setattr(module, "patch_g05_vlm_for_rpu", patch_vlm)
    def patch_action(expert, *, max_seq_len, _execution_config=None, **precision):
        expected = dict(expected_precision)
        if linear_accumulation == "acc16":
            expected["_g05_acc16_experiment"] = True
        if action_weight_dtype != weight_dtype:
            expected.update(action_weight_dtype=action_weight_dtype, _g05_exact_k1=True)
        assert precision == expected
        patch_calls.append(("action", expert, max_seq_len))
    monkeypatch.setattr(module, "patch_g05_action_expert_for_rpu", patch_action)
    def patch_vision(
        model, *, _allow_certified_g05_k1=False, _execution_config=None, **precision
    ):
        assert precision == expected_precision
        assert _allow_certified_g05_k1 is True
        model._rpu_g05_vision_ready = True
        patch_calls.append(("vision", model, _allow_certified_g05_k1))

    monkeypatch.setattr(module, "patch_g05_vision_for_rpu", patch_vision)

    def original_predict_action(self, batch):
        return batch

    def forward_inference(self, **kwargs):
        forward_calls.append((self.discrete_action, kwargs))
        return {"action": "fm-action"}

    def original_policy_forward(
        self,
        batch,
        inference_mode=False,
        return_dict=False,
        is_vlm_batch=False,
    ):
        fallback_calls.append(
            (batch, inference_mode, return_dict, is_vlm_batch)
        )
        return "fallback"

    policy_module_name = "vla_rtc.models.g05.g05_policy_qwen35"
    policy_module = types.ModuleType(policy_module_name)
    base_policy_module_name = "vla_rtc.models.g05.g05_policy"
    base_policy_module = types.ModuleType(base_policy_module_name)

    def ensure_sample_list(samples):
        ensure_calls.append(samples)
        return samples

    original_policy_forward.__module__ = base_policy_module_name
    base_policy_module._ensure_sample_list = ensure_sample_list
    monkeypatch.setitem(__import__("sys").modules, policy_module_name, policy_module)
    monkeypatch.setitem(
        __import__("sys").modules, base_policy_module_name, base_policy_module
    )
    policy_cls = type(
        "G05PolicyQwen35",
        (),
        {
            "forward": original_policy_forward,
            "predict_action": original_predict_action,
            "forward_inference": forward_inference,
        },
    )
    policy_cls.__module__ = policy_module_name
    model_cls = type("G05ModelQwen35", (), {})
    model_cls.__module__ = "vla_rtc.models.g05.g05_model_qwen35"
    model = model_cls()
    model.training = False
    model.vlm = types.SimpleNamespace(
        input_proj=torch.nn.Linear(2, 2, bias=False)
    )
    model.vision_tower = types.SimpleNamespace(
        config=types.SimpleNamespace(
            depth=24,
            num_heads=16,
            hidden_size=1024,
            intermediate_size=4096,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            in_channels=3,
            out_hidden_size=2048,
            hidden_act="gelu_pytorch_tanh",
            temporal_freq=0,
            spacetime_mode="factorized",
            token_drop_layer=None,
            temporal_pe_pretrain_frames=None,
            batch_all_cameras=False,
        )
    )
    model.proprio_embedder = types.SimpleNamespace(mlp=torch.nn.Linear(2, 2))
    model._forward_embed = lambda *_args, **_kwargs: None
    model.action_expert = types.SimpleNamespace(config=object(), training=False)
    model.fm_helper = types.SimpleNamespace(
        action_causal=False,
        horizon_steps=32,
        action_dim=27,
        num_inference_steps=10,
        time_convention="pi_convention",
        use_correlated_noise=False,
    )
    model.cfg = types.SimpleNamespace(
        ae_vlm_condition_mode="cross_attn_only",
        cond_steps=1,
        num_obs_steps=1,
        position_ids_type="pi0fast",
    )
    model.use_training_rtc = True
    model.max_inference_delay = 8
    model.inference_fm = lambda *_args, **_kwargs: None

    policy = policy_cls()
    policy.training = False
    policy.continuous_action = True
    policy.discrete_action = True
    policy.return_continuous_action = True
    policy.predict_cot = False
    policy.num_input_images = 3
    policy.model = model
    policy.processor = types.SimpleNamespace(batchify_action=False)

    assert module._is_vla_rtc_fm_release(policy, model)
    for name, bad_value in (
        ("patch_size", 14),
        ("temporal_patch_size", 1),
        ("spatial_merge_size", 4),
        ("in_channels", 1),
        ("out_hidden_size", 1024),
        ("hidden_act", "silu"),
    ):
        original = getattr(model.vision_tower.config, name)
        setattr(model.vision_tower.config, name, bad_value)
        assert not module._is_vla_rtc_fm_release(policy, model)
        setattr(model.vision_tower.config, name, original)
    model.vision_tower.config.temporal_freq = 4
    assert not module._is_vla_rtc_fm_release(policy, model)
    model.vision_tower.config.temporal_freq = 0
    policy.num_input_images = 18
    assert not module._is_vla_rtc_fm_release(policy, model)
    policy.num_input_images = 3
    assert module._is_vla_rtc_fm_release(policy, model)

    assert module.patch_g05_policy_for_rpu(
        policy, max_seq_len=1024, **policy_precision) is policy
    assert [os.environ.get(name) for name in exact_perf_flags] == [
        exact_perf_override
    ]
    assert policy.continuous_action is True
    assert policy.discrete_action is False
    assert model._rpu_g05_certified_k1_vision is (linear_accumulation == "acc32")
    assert model._rpu_g05_acc16_experiment is (linear_accumulation == "acc16")
    assert policy._rpu_g05_linear_accumulation == linear_accumulation
    assert model._rpu_g05_linear_accumulation == linear_accumulation
    assert policy.forward.__func__ is module._rpu_g05_policy_forward
    assert policy._rpu_g05_original_policy_forward.__func__ is original_policy_forward
    assert patch_calls == [
        ("action_preflight",),
        ("vlm_preflight",),
        ("vision_preflight",),
        ("vision", model, True),
        ("vlm", model, 1024),
        ("action", model.action_expert, 1024),
    ]

    batch = {
        "samples": ["sample"],
        "pixel_values": {"camera": torch.zeros(1)},
        "action": torch.zeros(1, 32, 27),
        "action_dim_is_pad": torch.zeros(1, 27, dtype=torch.bool),
        "prev_action": torch.ones(1, 32, 27),
        "inference_interval": torch.tensor([5]),
        "inference_delay": torch.tensor([7]),
    }
    original = {
        key: batch[key]
        for key in (
            "pixel_values",
            "action",
            "action_dim_is_pad",
            "prev_action",
            "inference_interval",
            "inference_delay",
        )
    }
    assert policy.forward(batch, inference_mode=True) is batch
    assert batch["action"] == "fm-action"
    assert ensure_calls == [["sample"]]
    assert len(forward_calls) == 1
    discrete_enabled, forwarded = forward_calls[0]
    assert discrete_enabled is False
    assert forwarded["samples"] == ["sample"]
    assert forwarded["pixel_values"] is original["pixel_values"]
    assert forwarded["actions"] is original["action"]
    for key in (
        "action_dim_is_pad",
        "prev_action",
        "inference_interval",
        "inference_delay",
    ):
        assert forwarded[key] is original[key]
    fallback_batch = {"samples": ["fallback"]}
    assert policy.forward(
        fallback_batch, inference_mode=True, return_dict=True
    ) == "fallback"
    assert fallback_calls == [(fallback_batch, True, True, False)]


def test_g05_release_policy_with_nonrelease_model_is_rejected_before_swizzle():
    module = _load_adapter()

    policy_cls = type("G05PolicyQwen35", (), {})
    policy_cls.__module__ = "vla_rtc.models.g05.g05_policy_qwen35"
    model_cls = type("G05ModelQwen35", (), {})
    model_cls.__module__ = "downstream_copy.models.g05.g05_model_qwen35"

    model = model_cls()
    model.use_training_rtc = True
    model.max_inference_delay = 8
    policy = policy_cls()
    policy.training = False
    policy.continuous_action = True
    policy.discrete_action = True
    policy.return_continuous_action = True
    policy.predict_cot = False
    policy.model = model

    with pytest.raises(NotImplementedError, match="discrete"):
        module.patch_g05_policy_for_rpu(policy, max_seq_len=1024)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"training": True}, "call policy.eval"),
        ({"continuous_action": False}, "continuous_action=true"),
        ({"predict_cot": True}, "predict_cot=false"),
    ],
)
def test_g05_policy_patch_rejects_unsupported_modes(monkeypatch, overrides, message):
    module = _load_adapter()
    monkeypatch.setattr(
        module,
        "patch_g05_vlm_for_rpu",
        lambda *args, **kwargs: pytest.fail("must reject before swizzling weights"),
    )
    values = dict(
        training=False,
        continuous_action=True,
        discrete_action=False,
        predict_cot=False,
        model=object(),
    )
    values.update(overrides)

    with pytest.raises(NotImplementedError, match=message):
        module.patch_g05_policy_for_rpu(types.SimpleNamespace(**values))


@pytest.mark.parametrize("perf_override", [None, "0"])
def test_g05_policy_patch_connects_continuous_action(monkeypatch, perf_override):
    perf_flags = ("RPU_FASTREPLAY_SKIP_SYNC",)
    linear_acc32 = "RPU_LINEAR_ACC32"
    for name in perf_flags:
        if perf_override is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, perf_override)
    if perf_override is None:
        monkeypatch.delenv(linear_acc32, raising=False)
    else:
        monkeypatch.setenv(linear_acc32, perf_override)
    module = _load_adapter()
    assert [os.environ.get(name) for name in perf_flags] == [perf_override] * len(perf_flags)
    assert os.environ.get(linear_acc32) == perf_override
    torch = __import__("torch")
    calls = []
    encode_calls = []
    expert = types.SimpleNamespace(config=object(), training=False)

    class EMAVectorQuantize:
        def __init__(self):
            self.inited = torch.tensor(True)

    class ResidualVectorQuantize:
        training = True
        n_codebooks = 4
        quantizers = [EMAVectorQuantize() for _ in range(4)]

    class ActionCodecV2Model:
        rvq = ResidualVectorQuantize()

    class ActionCodecV2Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.model = ActionCodecV2Model()
            self.key_dims = {
                "left_control": 9,
                "left_gripper": 1,
                "right_control": 9,
                "right_gripper": 1,
                "lower_body": 7,
            }
            self._partitioner = types.SimpleNamespace(merge_preprocessor=None)

        def get_decode_metadata(self):
            return types.SimpleNamespace(nn_chunked_key_count=3)

    class VQActionTokenizer:
        action_tokenizer = ActionCodecV2Wrapper()

    class Processor:
        batchify_action = True
        action_tokenizer = VQActionTokenizer()

        def encode_inference(
            self, samples, device, mode="fm", training=False, **kwargs
        ):
            encode_calls.append((samples, device, mode, training, kwargs))
            return "encoded"

        def _prepare_segments(self, samples, control_flag):
            assert control_flag == "context_only"
            return [[types.SimpleNamespace(sample_key="command", processor_key="text")]]

    processor = Processor()

    class Model:
        pass

    model = Model()
    model.training = False
    model.vlm = types.SimpleNamespace(layers=object(), norm=object(), config=object())
    model.action_expert = expert
    model.cfg = types.SimpleNamespace(
        ae_vlm_condition_mode="cross_attn_only",
        position_ids_type="pi0fast",
    )
    model.fm_helper = _canonical_fm_helper()
    original_inference_fm = lambda *_args, **_kwargs: None
    model.inference_fm = original_inference_fm
    predict_calls = []

    def original_predict_action(batch):
        predict_calls.append(("original", batch))
        return batch

    def forward_inference(**kwargs):
        predict_calls.append(("rpu", kwargs))
        return {"action": "generated"}

    policy = types.SimpleNamespace(
        training=False,
        continuous_action=True,
        discrete_action=False,
        predict_cot=False,
        model=model,
        processor=processor,
        predict_action=original_predict_action,
        forward_inference=forward_inference,
    )
    monkeypatch.setattr(module, "_check_g05_action_profile", lambda config: None)
    monkeypatch.setattr(module, "_check_g05_vlm_profile", lambda config: None)
    monkeypatch.setattr(
        module,
        "patch_g05_vlm_for_rpu",
        lambda value, *, max_seq_len, _execution_config=None: calls.append(
            ("vlm", value, max_seq_len)
        ),
    )
    monkeypatch.setattr(
        module,
        "patch_g05_action_expert_for_rpu",
        lambda value, *, max_seq_len, _execution_config=None: calls.append(
            ("action", value, max_seq_len)
        ),
    )

    assert module.patch_g05_policy_for_rpu(policy, max_seq_len=1536) is policy
    expected_perf = ["1" if perf_override is None else perf_override]
    assert [os.environ.get(name) for name in perf_flags] == expected_perf
    assert os.environ.get(linear_acc32) == perf_override
    assert calls == [
        ("vlm", model, 1536),
        ("action", expert, 1536),
    ]
    assert policy._rpu_g05_policy_ready
    monkeypatch.setattr(
        module, "_g05_policy_runtime_ready", lambda owner: owner is policy
    )
    assert module.patch_g05_policy_for_rpu(policy, max_seq_len=2048) is policy
    assert len(calls) == 2
    assert model._rpu_g05_original_inference_fm is original_inference_fm
    assert model.inference_fm.__func__ is module._rpu_g05_inference_fm
    assert processor.encode_inference.__func__ is module._rpu_g05_encode_inference
    assert policy._rpu_g05_original_predict_action is original_predict_action
    assert policy.predict_action.__func__ is module._rpu_g05_predict_action
    batch = {
        "samples": ["sample"],
        "pixel_values": "pixels",
        "action": "input-action",
        "action_dim_is_pad": "padding",
        "prev_action": "previous-action",
        "inference_interval": 5,
        "inference_delay": 7,
    }
    assert policy.predict_action(batch) is batch
    assert batch["action"] == "generated"
    assert predict_calls == [
        (
            "rpu",
            {
                "samples": ["sample"],
                "pixel_values": "pixels",
                "actions": "input-action",
                "action_dim_is_pad": "padding",
                "prev_action": "previous-action",
                "inference_interval": 5,
                "inference_delay": 7,
            },
        )
    ]
    policy.training = True
    with pytest.raises(NotImplementedError, match="inference only"):
        policy.predict_action(batch)
    policy.training = False

    sample = {
        "template": "<command_text_!><EOC><action_action>",
        "action": {
            "value": torch.zeros(1),
            "action_op_mask": torch.ones(1, dtype=torch.bool),
            "action_dim_is_pad": torch.zeros(1, dtype=torch.bool),
        },
    }
    torch.manual_seed(123)
    torch.rand(3)
    torch.randint(1, 5, (3,))
    expected_rng = torch.get_rng_state()
    torch.manual_seed(123)

    assert processor.encode_inference(
        [sample], device=torch.device("cpu"), mode="ar", training=False
    ) == "encoded"
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert "action" not in encode_calls[-1][0][0]
    assert "action" in sample

    processor._prepare_segments = types.MethodType(
        lambda self, samples, control_flag: [
            [types.SimpleNamespace(sample_key="action", processor_key="action")]
        ],
        processor,
    )
    before = torch.get_rng_state().clone()
    processor.encode_inference(
        [sample], device=torch.device("cpu"), mode="ar", training=False
    )
    assert torch.equal(torch.get_rng_state(), before)
    assert "action" in encode_calls[-1][0][0]


@pytest.mark.parametrize(
    ("profile_ok", "position_ids_type", "error", "message"),
    [
        (False, "pi0fast", ValueError, "bad VLM profile"),
        (True, "default", NotImplementedError, "position_ids_type=pi0fast"),
    ],
)
def test_g05_policy_preflights_vlm_before_vision_mutation(
    monkeypatch, profile_ok, position_ids_type, error, message
):
    module = _load_adapter()
    model = types.SimpleNamespace(
        training=False,
        vlm=types.SimpleNamespace(layers=object(), norm=object(), config=object()),
        action_expert=types.SimpleNamespace(config=object(), training=False),
        cfg=types.SimpleNamespace(
            ae_vlm_condition_mode="cross_attn_only",
            position_ids_type=position_ids_type,
        ),
        fm_helper=_canonical_fm_helper(),
    )
    policy = types.SimpleNamespace(
        training=False,
        continuous_action=True,
        discrete_action=False,
        predict_cot=False,
        model=model,
    )
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    monkeypatch.setattr(module, "_check_g05_action_profile", lambda config: None)

    def check_profile(config):
        if not profile_ok:
            raise ValueError("bad VLM profile")

    monkeypatch.setattr(module, "_check_g05_vlm_profile", check_profile)
    for name in (
        "patch_g05_vision_for_rpu",
        "patch_g05_vlm_for_rpu",
        "patch_g05_action_expert_for_rpu",
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"{_name} must not mutate before VLM preflight"
            ),
        )

    with pytest.raises(error, match=message):
        module.patch_g05_policy_for_rpu(policy)
    assert not hasattr(model, "_rpu_g05_vision_ready")
    assert not hasattr(model, "_rpu_g05_vlm_ready")


def test_g05_policy_preflights_proprio_before_vision_mutation(monkeypatch):
    module = _load_adapter()
    model = types.SimpleNamespace(
        training=False,
        vlm=types.SimpleNamespace(layers=object(), norm=object(), config=object()),
        action_expert=types.SimpleNamespace(config=object(), training=False),
        cfg=types.SimpleNamespace(
            ae_vlm_condition_mode="cross_attn_only",
            position_ids_type="pi0fast",
        ),
        fm_helper=_canonical_fm_helper(),
    )
    policy = types.SimpleNamespace(
        training=False,
        continuous_action=True,
        discrete_action=False,
        predict_cot=False,
        model=model,
    )
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    monkeypatch.setattr(module, "_check_g05_action_profile", lambda config: None)
    monkeypatch.setattr(module, "_check_g05_vlm_profile", lambda config: None)
    for name in (
        "patch_g05_vision_for_rpu",
        "patch_g05_vlm_for_rpu",
        "patch_g05_action_expert_for_rpu",
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"{_name} must not mutate before proprio preflight"
            ),
        )

    with pytest.raises(TypeError, match="model.proprio_embedder.mlp"):
        module.patch_g05_policy_for_rpu(policy)
    assert not hasattr(model, "_rpu_g05_vision_ready")
    assert not hasattr(model, "_rpu_g05_vlm_ready")


def test_g05_policy_rejects_non_cpu_host_before_swizzle(monkeypatch):
    module = _load_adapter()
    torch = __import__("torch")
    policy = types.SimpleNamespace(
        training=False,
        continuous_action=True,
        discrete_action=False,
        model=object(),
        parameters=lambda: iter(
            [torch.nn.Parameter(torch.empty(1, device="meta"))]
        ),
    )
    monkeypatch.setattr(
        module,
        "patch_g05_vlm_for_rpu",
        lambda *args, **kwargs: pytest.fail("must reject before swizzling weights"),
    )

    with pytest.raises(NotImplementedError, match="CPU-hosted"):
        module.patch_g05_policy_for_rpu(policy)


def test_g05_policy_rejects_action_mode_before_swizzle(monkeypatch):
    module = _load_adapter()
    expert = types.SimpleNamespace(config=object())
    model = types.SimpleNamespace(
        action_expert=expert,
        cfg=types.SimpleNamespace(ae_vlm_condition_mode="both"),
        fm_helper=_canonical_fm_helper(),
    )
    policy = types.SimpleNamespace(
        training=False,
        continuous_action=True,
        discrete_action=False,
        model=model,
    )
    monkeypatch.setattr(module, "_check_g05_action_profile", lambda config: None)
    monkeypatch.setattr(
        module,
        "patch_g05_vlm_for_rpu",
        lambda *args, **kwargs: pytest.fail("must reject before swizzling weights"),
    )

    with pytest.raises(NotImplementedError, match="cross_attn_only"):
        module.patch_g05_policy_for_rpu(policy)


def test_g05_policy_rejects_training_action_expert_before_swizzle(monkeypatch):
    module = _load_adapter()
    expert = types.SimpleNamespace(config=object(), training=True)
    model = types.SimpleNamespace(
        action_expert=expert,
        cfg=types.SimpleNamespace(ae_vlm_condition_mode="cross_attn_only"),
        fm_helper=_canonical_fm_helper(),
    )
    policy = types.SimpleNamespace(
        training=False,
        continuous_action=True,
        discrete_action=False,
        predict_cot=False,
        model=model,
    )
    monkeypatch.setattr(module, "_check_g05_action_profile", lambda config: None)
    monkeypatch.setattr(
        module,
        "patch_g05_vlm_for_rpu",
        lambda *args, **kwargs: pytest.fail("must reject before swizzling weights"),
    )

    with pytest.raises(NotImplementedError, match="call expert.eval"):
        module.patch_g05_policy_for_rpu(policy)


def test_g05_policy_rejects_short_action_capacity_before_swizzle(monkeypatch):
    module = _load_adapter()
    policy = types.SimpleNamespace(
        training=False,
        continuous_action=True,
        discrete_action=False,
        model=object(),
    )
    monkeypatch.setattr(
        module,
        "patch_g05_vlm_for_rpu",
        lambda *args, **kwargs: pytest.fail("must reject before swizzling weights"),
    )

    with pytest.raises(ValueError, match="32-token action horizon"):
        module.patch_g05_policy_for_rpu(policy, max_seq_len=31)


def test_g05_policy_rejects_action_horizon_before_swizzle(monkeypatch):
    module = _load_adapter()
    model = types.SimpleNamespace(
        action_expert=types.SimpleNamespace(config=object()),
        fm_helper=types.SimpleNamespace(horizon_steps=16),
    )
    policy = types.SimpleNamespace(
        training=False,
        continuous_action=True,
        discrete_action=False,
        model=model,
    )
    monkeypatch.setattr(
        module,
        "patch_g05_vlm_for_rpu",
        lambda *args, **kwargs: pytest.fail("must reject before swizzling weights"),
    )

    with pytest.raises(NotImplementedError, match="horizon_steps=32"):
        module.patch_g05_policy_for_rpu(policy)


def test_g05_action_prefix_bridge_is_complete_and_read_only():
    module = _load_adapter()
    physical = object()
    vlm_cache = types.SimpleNamespace(
        num_items=lambda: 64,
        _rpu_qwen3_5_cache=physical,
        _rpu_qwen3_5_generation=3,
    )
    model = types.SimpleNamespace(
        cfg=types.SimpleNamespace(ae_vlm_condition_mode="cross_attn_only"),
        fm_helper=types.SimpleNamespace(action_causal=False),
        vlm=types.SimpleNamespace(
            _rpu_g05_cache_generation=3,
            _rpu_g05_cache=physical,
        ),
    )

    facade = module._rpu_g05_build_prefix_action_kv(model, vlm_cache, 64)
    assert facade._rpu_g05_prefix_len == 64
    assert facade._rpu_g05_prefix_cache is vlm_cache
    with pytest.raises(NotImplementedError, match="complete read-only"):
        module._rpu_g05_build_prefix_action_kv(model, vlm_cache, 48)
    vlm_cache._rpu_qwen3_5_generation = 2
    with pytest.raises(RuntimeError, match="older turn"):
        module._rpu_g05_build_prefix_action_kv(model, vlm_cache, 64)


@pytest.mark.parametrize("entry", ("forward", "step", "loop", "rtc_trace"))
def test_g05_ready_action_miss_rejects_before_preparation_or_prime(monkeypatch, entry):
    import torch
    import rpu_backend.graph
    from test_graph_cache_freeze import _cache, _sig

    action = _load_action_adapter()
    cache = _cache()
    with cache.capture(_sig("production")):
        pass
    cache.freeze()
    before = dict(cache._impl.entries)
    state = types.SimpleNamespace(handle=33, hidden_size=1024, num_layers=24,
        action_dim_pad=32, action_prefix_lens=[245] * 24,
        action_graph_cache=cache, action_trace_graph_cache=cache)
    expert = types.SimpleNamespace(training=False, _rpu_qwen3_5=state)
    plan = types.SimpleNamespace(selected=types.SimpleNamespace(stage_tuple=
        types.SimpleNamespace(physical_descriptor=(1,))), graph_key_words=lambda: ())
    monkeypatch.setattr(action, "_plan_g05_action_execution", lambda *a, **kw: plan)
    monkeypatch.setattr(rpu_backend.graph, "GraphSignature", lambda **kw: _sig("unwarmed"))
    def preparation(*args, **kwargs):
        raise AssertionError("READY miss reached mutable preparation/one-shot")
    for name in ("_prepare_prefix", "_ensure_action_rope", "_adaptive_modulation",
                 "_adaptive_modulation_stack", "_run_action_cold_prime"):
        monkeypatch.setattr(action, name, preparation)
    def tensor(shape, pointer):
        return types.SimpleNamespace(shape=shape, device=types.SimpleNamespace(type="rpu"),
            dtype=torch.float16, data_ptr=lambda: pointer, is_contiguous=lambda: True)
    inputs, output = tensor((1, 32, 32), 1), tensor((1, 32, 32), 2)
    mask, rtc = tensor((32,), 3), tensor((1, 2, 32), 4)
    with pytest.raises(RuntimeError, match="READY miss"):
        if entry == "forward":
            action._rpu_g05_action_forward(expert, torch.zeros(1, 32, 1024),
                torch.zeros(1, 1, 32, 277), torch.zeros(32, 3),
                kv_cache=types.SimpleNamespace(_rpu_g05_prefix_len=245))
        elif entry == "step":
            action._rpu_g05_action_step(expert, inputs, mask, output, .1, None, [245] * 24, 245)
        elif entry == "loop":
            action._rpu_g05_action_loop(expert, inputs, mask, output, .1, [None] * 10, [245] * 24, 245)
        else:
            action._rpu_g05_action_rtc_trace(expert, inputs, mask, output, rtc, 2, .1,
                [None] * 10, [245] * 24, 245)
    assert cache.is_frozen() and cache._impl.entries == before


def test_action_shape_eviction_keeps_prepared_plan_and_ready_rejects():
    from test_graph_cache_freeze import _cache, _sig

    action = _load_action_adapter()
    cache = _cache()
    cache.begin_warmup()
    with cache.capture(_sig("old-shape")):
        pass
    calls = []
    new_plan = cache.prepare_plan(("new-shape",), lambda: calls.append("prepare") or object())
    action._evict_action_graph_entries(cache)
    assert cache.size() == 0
    with cache.capture(_sig("new-shape")):
        pass
    assert cache.prepare_plan(("new-shape",), lambda: pytest.fail("replanned stable shape")) is new_plan
    cache.freeze()
    with cache.capture(_sig("new-shape")) as graph:
        assert graph.state() == 3
    with pytest.raises(RuntimeError, match="READY"):
        action._evict_action_graph_entries(cache)
    assert cache.size() == 1 and calls == ["prepare"]
    # Resource/config invalidation retains the stronger existing clear contract.
    cache.begin_warmup()
    cache.clear()
    assert cache.prepare_plan(("new-shape",), lambda: object()) is not new_plan
