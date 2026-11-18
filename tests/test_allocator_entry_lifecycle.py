from __future__ import annotations

import pytest

from rpu_backend.adapters import qwen3_5


class _Model:
    pass


@pytest.mark.parametrize("precision", ["w8a16", "w4a16_pgrp"])
@pytest.mark.parametrize("allocator_fails", [False, True])
def test_qwen35_quantized_9b_text_claims_caching_before_weight_upload(
        monkeypatch, precision, allocator_fails):
    from types import SimpleNamespace
    from rpu_backend import graph as graph_module
    from rpu_backend.adapters.qwen3_5 import cores

    model = _Model()
    quant = {"method": precision}
    if precision == "w4a16_pgrp":
        quant["group_size"] = 32
    model.config = SimpleNamespace(quant_config=quant)
    model._rpu_execution = {}
    model.forward = original_forward = object()
    model.half = lambda: model
    adapter = object.__new__(qwen3_5.Qwen3_5Adapter)
    adapter.model, adapter._rpu_is_ready = model, False
    adapter._handle = adapter._graph_cache = None
    events = []
    monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    monkeypatch.setattr(cores, "is_dense_large_vl_candidate", lambda config: True)
    monkeypatch.setattr(cores, "validate_dense_large_vl_weights",
                        lambda owner: pytest.fail("quantized text must not use the FP16 arena validator"))
    monkeypatch.setattr(graph_module.GraphRuntimePolicy, "from_environment",
                        lambda **kwargs: pytest.fail("quantized text must not reserve FP16 arenas"))
    monkeypatch.setattr(qwen3_5, "validate_legacy_text_profile", lambda config: False)
    monkeypatch.setattr(qwen3_5, "_qwen3_5_text_runtime_state", lambda owner: None)
    monkeypatch.setattr(qwen3_5, "_resolve_text_install_options", lambda *a, **kw: (8192,))
    monkeypatch.setattr(qwen3_5, "_resolve_vision_install_options", lambda **kw: ())
    monkeypatch.setattr(qwen3_5, "_claim_live_instance", lambda owner: events.append("claim"))
    monkeypatch.setattr(qwen3_5, "_release_live_instance", lambda owner: events.append("release"))

    def select_allocator(enabled):
        events.append(("allocator", enabled))
        if allocator_fails:
            raise RuntimeError("allocator conflict")

    def half():
        events.append("weight-upload-boundary")
        raise RuntimeError("stop before device upload")

    model.half = half
    monkeypatch.setattr(qwen3_5.torch.rpu, "set_caching_allocator", select_allocator)
    with pytest.raises(RuntimeError, match="allocator conflict" if allocator_fails else "stop before device upload"):
        adapter.to_rpu()
    assert events == ["claim", ("allocator", True),
                      "release" if allocator_fails else "weight-upload-boundary"]
    assert getattr(model, "_rpu_swizzle_started", False) is (not allocator_fails)
    assert model.forward is original_forward
    assert adapter._handle is None and adapter._graph_cache is None


@pytest.mark.parametrize("quantized,failure", [
    (False, "claim"), (False, "allocator"), (False, "arenas"), (True, "arenas"),
])
def test_qwen35_9b_allocator_failure_is_cold_and_releases_owner(
        monkeypatch, quantized, failure):
    from types import SimpleNamespace
    from rpu_backend import graph as graph_module
    from rpu_backend.adapters.qwen3_5 import cores

    model = _Model()
    model.config = SimpleNamespace(quant_config={"method": "w8a16"} if quantized else None)
    model._rpu_execution = {}
    model.forward = original_forward = object()
    model.half = lambda: model
    adapter = object.__new__(qwen3_5.Qwen3_5Adapter)
    adapter.model = model
    adapter._rpu_is_ready = False
    adapter._handle = adapter._graph_cache = None
    events = []
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    monkeypatch.setattr(cores, "is_dense_large_vl_candidate", lambda config: True)
    monkeypatch.setattr(cores, "validate_dense_large_vl_weights", lambda owner: None)
    monkeypatch.setattr(qwen3_5, "validate_legacy_text_profile", lambda config: False)
    monkeypatch.setattr(qwen3_5, "_qwen3_5_text_runtime_state", lambda owner: None)
    monkeypatch.setattr(qwen3_5, "_resolve_text_install_options",
                        lambda *args, **kwargs: (8192,))
    monkeypatch.setattr(qwen3_5, "_resolve_vision_install_options", lambda **kwargs: ())

    def claim(owner):
        events.append("claim")
        if failure == "claim":
            raise RuntimeError("existing owner")

    def select_allocator(enabled):
        events.append(("allocator", enabled))
        if failure == "allocator":
            raise RuntimeError("allocator conflict")

    class GraphPolicy:
        @classmethod
        def from_environment(cls, **kwargs):
            assert kwargs == {"execution_core_count": 8, "graph_arena_count": 4,
                              "qwen35_dense_27b_sdk_budget": False}
            return cls()

        def prepare_arenas(self):
            events.append("arenas")
            return False

    monkeypatch.setattr(qwen3_5, "_claim_live_instance", claim)
    monkeypatch.setattr(qwen3_5, "_release_live_instance", lambda owner: events.append("release"))
    monkeypatch.setattr(qwen3_5.torch.rpu, "set_caching_allocator", select_allocator)
    monkeypatch.setattr(graph_module, "GraphRuntimePolicy", GraphPolicy)
    error_type = qwen3_5.RPUBackendError if failure == "arenas" else RuntimeError
    with pytest.raises(error_type, match={
        "claim": "existing owner", "allocator": "allocator conflict",
        "arenas": "accessible device hardware",
    }[failure]):
        adapter.to_rpu()

    expected = ["claim"]
    if failure != "claim":
        if not quantized:
            expected.append(("allocator", True))
        if failure == "arenas":
            expected.append("arenas")
        expected.append("release")
    assert events == expected
    assert not hasattr(model, "_rpu_swizzle_started")
    assert model.forward is original_forward
    assert adapter._handle is None and adapter._graph_cache is None
    assert not adapter._rpu_is_ready


@pytest.mark.parametrize("failure", ["claim", "allocator", "arenas"])
def test_qwen35_moe_claims_cached_storage_before_materialization(monkeypatch, failure):
    from rpu_backend import graph as graph_module
    from rpu_backend.adapters import qwen3_5_moe
    from rpu_backend.api import causal_lm

    model = _Model()
    model._rpu_execution = {}
    model.config = object()
    original_forward = object()
    model.forward = original_forward
    adapter = object.__new__(qwen3_5_moe.Qwen3_5MoeAdapter)
    adapter.model = model
    adapter._rpu_is_ready = False
    events = []

    # This fixture tests ownership order, not checkpoint geometry. Resolve the
    # real default topology without asking the profile validator to inspect object().
    from rpu_backend.adapters.qwen3_5_moe.cores import core_topology
    monkeypatch.setattr(qwen3_5_moe, "core_topology",
                        lambda execution, **kwargs: core_topology(execution))
    monkeypatch.setattr(qwen3_5_moe, "_qwen3_5_moe_text_runtime_state", lambda _: None)
    monkeypatch.setattr(qwen3_5_moe, "_resolve_text_install_options",
                        lambda *args, **kwargs: (8192,))
    monkeypatch.setattr(qwen3_5_moe, "_resolve_vision_install_options", lambda **kwargs: ())

    def claim(owner):
        events.append("claim")
        if failure == "claim":
            raise RuntimeError("existing owner")

    def select_allocator(enabled):
        events.append(("allocator", enabled))
        if failure == "allocator":
            raise RuntimeError("allocator conflict")

    class GraphPolicy:
        @classmethod
        def from_environment(cls, **kwargs):
            assert kwargs == {"execution_core_count": 8, "graph_arena_count": 4}
            return cls()

        def prepare_arenas(self):
            events.append("arenas")
            return False

    monkeypatch.setattr(qwen3_5_moe, "_claim_live_instance", claim)
    monkeypatch.setattr(causal_lm, "_release_live_instance",
                        lambda owner: events.append("release"))
    monkeypatch.setattr(qwen3_5_moe.torch.rpu, "set_caching_allocator", select_allocator)
    monkeypatch.setattr(graph_module, "GraphRuntimePolicy", GraphPolicy)

    error_type = qwen3_5_moe.RPUBackendError if failure == "arenas" else RuntimeError
    with pytest.raises(error_type, match={
        "claim": "existing owner", "allocator": "allocator conflict",
        "arenas": "accessible hardware",
    }[failure]):
        adapter.to_rpu()

    expected = ["claim"]
    if failure != "claim":
        expected.append(("allocator", True))
        if failure == "arenas":
            expected.append("arenas")
        expected.append("release")
    assert events == expected
    assert not hasattr(model, "_rpu_swizzle_started")
    assert model.forward is original_forward
    assert adapter._handle is None and adapter._graph_cache is None
    assert not adapter._rpu_is_ready


def test_qwen35_legacy_claims_direct_allocator_without_get_check(monkeypatch):
    model = _Model()
    model.config = object()
    model._rpu_execution = {}
    adapter = object.__new__(qwen3_5.Qwen3_5Adapter)
    adapter.model = model
    adapter._rpu_is_ready = False
    adapter._handle = None
    adapter._graph_cache = None
    events = []

    monkeypatch.setattr(qwen3_5, "_qwen3_5_text_runtime_state", lambda owner: None)
    monkeypatch.setattr(
        qwen3_5,
        "_resolve_text_install_options",
        lambda *args, **kwargs: (8192,),
    )
    monkeypatch.setattr(
        qwen3_5, "_resolve_vision_install_options", lambda **kwargs: ()
    )
    monkeypatch.setattr(
        qwen3_5, "validate_legacy_text_profile", lambda config: True
    )
    monkeypatch.setattr(
        qwen3_5, "_claim_live_instance", lambda owner: events.append("claim")
    )
    monkeypatch.setattr(
        qwen3_5, "_release_live_instance", lambda owner: events.append("release")
    )
    monkeypatch.setattr(
        qwen3_5.torch.rpu,
        "get_caching_allocator",
        lambda: pytest.fail("legacy path must not read allocator state"),
    )

    def fail_allocator(enabled):
        events.append(("allocator", enabled))
        raise RuntimeError("allocator conflict")

    monkeypatch.setattr(
        qwen3_5.torch.rpu, "set_caching_allocator", fail_allocator
    )
    with pytest.raises(RuntimeError, match="allocator conflict"):
        adapter.to_rpu()

    assert events == ["claim", ("allocator", False), "release"]
    assert not hasattr(model, "_rpu_swizzle_started")


def test_qwen35_legacy_arena_failure_releases_live_owner(monkeypatch):
    from rpu_backend import graph as graph_module

    model = _Model()
    model.config = object()
    model._rpu_execution = {}
    adapter = object.__new__(qwen3_5.Qwen3_5Adapter)
    adapter.model = model
    adapter._rpu_is_ready = False
    adapter._handle = None
    adapter._graph_cache = None
    events = []

    monkeypatch.setattr(qwen3_5, "_qwen3_5_text_runtime_state", lambda owner: None)
    monkeypatch.setattr(
        qwen3_5, "_resolve_text_install_options", lambda *args, **kwargs: (8192,),
    )
    monkeypatch.setattr(
        qwen3_5, "_resolve_vision_install_options", lambda **kwargs: (),
    )
    monkeypatch.setattr(
        qwen3_5, "validate_legacy_text_profile", lambda config: True,
    )
    monkeypatch.setattr(
        qwen3_5, "_claim_live_instance", lambda owner: events.append("claim"),
    )
    monkeypatch.setattr(
        qwen3_5, "_release_live_instance", lambda owner: events.append("release"),
    )
    monkeypatch.setattr(
        qwen3_5.torch.rpu,
        "set_caching_allocator",
        lambda enabled: events.append(("allocator", enabled)),
    )

    class RejectingGraphPolicy:
        @classmethod
        def from_environment(cls, **kwargs):
            assert kwargs == {"qwen35_legacy_27b_sdk_budget": True, "execution_core_count": 8}
            events.append("graph-policy")
            return cls()

        def prepare_arenas(self):
            events.append("prepare-arenas")
            return False

    monkeypatch.setattr(
        graph_module, "GraphRuntimePolicy", RejectingGraphPolicy,
    )

    with pytest.raises(qwen3_5.RPUBackendError, match="accessible device hardware"):
        adapter.to_rpu()

    assert events == [
        "graph-policy", "claim", ("allocator", False),
        "prepare-arenas", "release",
    ]
    assert not hasattr(model, "_rpu_swizzle_started")
