"""Controlled 27B admission and cold resource ordering, without device tensors."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters import qwen3_5 as adapter
from rpu_backend.adapters.qwen3_5 import text
from rpu_backend.adapters.qwen3_5.quant_scope import legacy_w8a16_quant_config
from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache, qwen3_5_kv_replication
from rpu_backend.api._execution import bind_execution_session
from rpu_backend.graph import GraphRuntimePolicy


def config():
    # Geometry transcribed from the MR56 Qwen3.8-27B-w8a16 config.json.
    return SimpleNamespace(
        hidden_size=5120, intermediate_size=17408, num_hidden_layers=64,
        num_attention_heads=24, num_key_value_heads=4, head_dim=256,
        linear_num_key_heads=16, linear_num_value_heads=48,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000,
                         "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                         "mrope_section": [11, 11, 10]},
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        quant_config=legacy_w8a16_quant_config(),
    )


@pytest.fixture
def legacy():
    value = config()
    value.layer_types *= 16
    return value


@pytest.mark.parametrize("chunk", ["auto", 64, 128])
def test_canonical_legacy_admission(legacy, monkeypatch, chunk):
    monkeypatch.setenv("QWEN3_5_QUANT_ALLOW_UNCERTIFIED", "1")
    adapter.Qwen3_5Adapter.preflight(legacy)
    adapter.Qwen3_5Adapter.preflight_execution(
        legacy, {"prefill": {"chunk_size": chunk}})
    assert text.CERTIFIED_CHUNK_ENVELOPE[(64, 5120)] == (8192, 128)


def test_rejected_legacy_requests_fail_before_loading(legacy, monkeypatch):
    monkeypatch.delenv("QWEN3_5_QUANT_ALLOW_UNCERTIFIED", raising=False)
    with pytest.raises(adapter.UnsupportedModelError, match="controlled"):
        adapter.Qwen3_5Adapter.preflight(legacy)
    with pytest.raises(adapter.UnsupportedModelError, match="ceiling 128"):
        adapter.Qwen3_5Adapter.preflight_execution(
            legacy, {"prefill": {"chunk_size": 192}})
    with pytest.raises(adapter.UnsupportedModelError, match="only text"):
        adapter.Qwen3_5Adapter.preflight_execution(
            legacy, {"vision": {"chunk_size": "auto"}})
    legacy.quant_config = None
    with pytest.raises(adapter.UnsupportedModelError, match="exact dense semantics"):
        adapter._check_profile(legacy)


def test_cache_uses_same_profile_geometry_without_ambient_selector(legacy, monkeypatch):
    class CaptureCache(Qwen3_5Cache):
        def __init__(self, **kwargs):
            self.geometry = kwargs

    for env in ("0", "1"):
        monkeypatch.setenv("RPU_QWEN3_5_KV_REP_MATCH_Q", env)
        assert qwen3_5_kv_replication(legacy) == 6
        cache = CaptureCache.from_config(legacy, max_seq_len=8192, device="cpu")
        assert cache.geometry["num_kv_heads"] == 24
        assert cache.geometry["attn_tp"] == 8
    # Ordinary siblings keep the previous effective eight-head cache.
    sibling = SimpleNamespace(hidden_size=4096, num_attention_heads=16, num_key_value_heads=4)
    assert qwen3_5_kv_replication(sibling) == 2
    legacy.num_attention_heads = 16
    with pytest.raises(ValueError, match="geometry"):
        CaptureCache.from_config(legacy, max_seq_len=128, device="cpu")


@pytest.mark.parametrize("fault", ["allocator", "unknown_allocator", "late_pool", "no_device", None])
def test_cold_resource_failure_precedes_weight_mutation(legacy, monkeypatch, fault):
    events = []

    class Model:
        def __init__(self):
            self.config = legacy
            self.model = SimpleNamespace()
            self.forward = object()

        def half(self):
            events.append("weight-mutation")
            raise RuntimeError("stop at mutation boundary")

    model = Model()
    owner = adapter.Qwen3_5Adapter.__new__(adapter.Qwen3_5Adapter)
    owner.model, owner._rpu_is_ready = model, False
    owner._execution_session = bind_execution_session(
        model, {}, entry_point="27B cold install test", graph_mode="BOUNDED_ONESHOT")
    owner._execution_session._bind_planner_owner(model.model)
    monkeypatch.setattr(adapter, "_claim_live_instance", lambda _: events.append("claim"))
    monkeypatch.setattr(adapter, "_release_live_instance", lambda _: events.append("release"))

    def claim_allocator(enabled):
        assert enabled is False
        events.append("allocator-claim")
        if fault == "unknown_allocator":
            raise RuntimeError("unavailable allocator claim")
        if fault == "allocator":
            raise RuntimeError("allocator policy already frozen to caching")

    # Faults must enter through the atomic cold claim used by production;
    # querying first would reintroduce the check-to-allocation race.
    monkeypatch.setattr(torch.rpu, "set_caching_allocator", claim_allocator)
    monkeypatch.setattr(torch.rpu, "get_caching_allocator",
                        lambda: pytest.fail("cold install must claim atomically"))

    def prepare(policy):
        assert policy.qwen35_legacy_27b_sdk_budget is True
        events.append("prepare-pool")
        if fault == "late_pool":
            raise RuntimeError("SDK pool frozen")
        return fault != "no_device"

    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", prepare)
    with pytest.raises((RuntimeError, adapter.RPUBackendError)):
        owner.to_rpu(max_seq_len=128)
    if fault is None:
        assert events == ["claim", "allocator-claim", "prepare-pool", "weight-mutation"]
        assert model._rpu_swizzle_started is True
    else:
        assert "weight-mutation" not in events
        assert not hasattr(model, "_rpu_swizzle_started")
        expected = ["claim", "allocator-claim"]
        if fault not in ("allocator", "unknown_allocator"):
            expected.append("prepare-pool")
        assert events == [*expected, "release"]


def test_text_graphs_share_the_prepared_cold_policy(legacy, monkeypatch):
    import rpu_backend

    calls = []
    policy = GraphRuntimePolicy(qwen35_legacy_27b_sdk_budget=True)

    class Cache:
        def __init__(self, *, max_entries, runtime_policy):
            calls.append(("decode", max_entries, runtime_policy))
        def clear(self):
            pass
        def cache_invariant_ok(self):
            return True

    class Graph:
        def __init__(self, *, runtime_policy):
            calls.append(("prefill", runtime_policy))
        def invalidate(self):
            pass

    def stop(*args, **kwargs):
        raise RuntimeError("stop before native install")

    monkeypatch.setattr(rpu_backend.graph, "GraphCache", Cache)
    monkeypatch.setattr(rpu_backend.graph, "Graph", Graph)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_linear_acc32",
                        lambda *_: pytest.fail("stopped install must not configure a native handle"),
                        raising=False)
    monkeypatch.setattr(text, "_install_qwen3_5_text_for_rpu_impl", stop)
    with pytest.raises(RuntimeError, match="before native"):
        text.install_qwen3_5_text_for_rpu(
            SimpleNamespace(), legacy, _graph_runtime_policy=policy)
    assert calls == [("decode", 1, policy), ("prefill", policy)]
    with pytest.raises(adapter.UnsupportedModelError, match="prepared cold"):
        text.install_qwen3_5_text_for_rpu(SimpleNamespace(), legacy)


def test_legacy_image_request_rejects_before_embedding_or_planning():
    from test_qwen3_5_python_contracts import _load_forward, _forward_self, _ForwardCache

    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported image reached model execution")

    model = _forward_self()
    model._qwen3_5_legacy_text_only = True
    model.model.embed_tokens = unexpected
    forward = _load_forward(run=unexpected, fuse=unexpected, preflight=unexpected)
    with pytest.raises(NotImplementedError, match="text-only"):
        forward(model, input_ids=torch.ones(1, 2, dtype=torch.long),
                past_key_values=_ForwardCache(), pixel_values=torch.ones(1),
                image_grid_thw=torch.tensor([[1, 2, 2]]),
                mm_token_type_ids=torch.ones(1, 2, dtype=torch.long))
