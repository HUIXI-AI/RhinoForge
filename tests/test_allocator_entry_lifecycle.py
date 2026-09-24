from __future__ import annotations

import pytest

from rpu_backend.adapters import qwen3_5


class _Model:
    pass


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

    with pytest.raises(qwen3_5.RPUBackendError, match="accessible RPU hardware"):
        adapter.to_rpu()

    assert events == [
        "graph-policy", "claim", ("allocator", False),
        "prepare-arenas", "release",
    ]
    assert not hasattr(model, "_rpu_swizzle_started")
