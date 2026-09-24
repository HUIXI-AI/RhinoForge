"""Public construction must not depend on excluded fixtures or mutate runtime early."""
import builtins

import pytest
import torch

from rpu_backend.adapters.rhinovla import pipeline
from rpu_backend.adapters.rhinovla.pipeline import RhinoVLAOnRPU
from rpu_backend.api import _execution, causal_lm


@pytest.mark.parametrize("missing", ["both", "model", "action_bundle"])
def test_missing_assets_fail_before_runtime_or_fixture_access(monkeypatch, missing):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("runtime")
        raise AssertionError("missing assets reached runtime setup")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("rpu_backend.tests"):
            calls.append("fixture")
            raise AssertionError("packaged construction imported excluded fixtures")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(_execution, "_require_execution_process_safe", forbidden)
    monkeypatch.setattr(causal_lm, "_claim_live_instance", forbidden)
    monkeypatch.setattr(torch.rpu, "set_caching_allocator", forbidden)
    monkeypatch.setattr(RhinoVLAOnRPU, "_initialize", forbidden)
    kwargs = {}
    if missing == "model":
        kwargs["action_bundle"] = object()
    elif missing == "action_bundle":
        kwargs["model"] = object()
    with pytest.raises(ValueError, match="official_runtime_factory"):
        RhinoVLAOnRPU(qin=object(), prefix_len=object(), **kwargs)
    assert calls == []


def test_explicit_model_and_bundle_reach_existing_initializer(monkeypatch):
    events = []
    model, bundle, qin, execution = object(), object(), object(), {}
    cold = object()
    monkeypatch.setattr(
        _execution, "_require_execution_process_safe",
        lambda: events.append("process-preflight"),
    )
    monkeypatch.setattr(
        causal_lm, "_claim_live_instance", lambda owner: events.append("claim"),
    )
    monkeypatch.setattr(
        pipeline,
        "_preflight_rhinovla_cold_install",
        lambda **kwargs: events.append("host-preflight") or cold,
    )
    monkeypatch.setattr(
        torch.rpu,
        "set_caching_allocator",
        lambda enabled: events.append(("allocator", enabled)),
    )

    def initialize(owner, **kwargs):
        events.append(kwargs)

    monkeypatch.setattr(RhinoVLAOnRPU, "_initialize", initialize)
    owner = RhinoVLAOnRPU(
        qin=qin, prefix_len=7, steps=10, model=model, action_bundle=bundle,
        rpu_execution=execution, flow_direction="official_descending",
    )
    assert events[:4] == [
        "process-preflight", "host-preflight", "claim", ("allocator", True)
    ]
    assert events[4] == dict(
        qin=qin, prefix_len=7, steps=10, instance_id=0, model=model,
        action_bundle=bundle, rpu_execution=execution,
        flow_direction="official_descending",
        _cold_preflight=cold,
    )
    del owner


def test_host_preflight_and_allocator_failures_do_not_poison_owner(monkeypatch):
    events = []
    monkeypatch.setattr(
        _execution,
        "_require_execution_process_safe",
        lambda: events.append("process-preflight"),
    )
    monkeypatch.setattr(
        causal_lm, "_claim_live_instance", lambda owner: events.append("claim")
    )
    monkeypatch.setattr(
        causal_lm, "_release_live_instance", lambda owner: events.append("release")
    )
    monkeypatch.setattr(
        torch.rpu,
        "set_caching_allocator",
        lambda enabled: events.append(("allocator", enabled)),
    )
    monkeypatch.setattr(
        pipeline,
        "_preflight_rhinovla_cold_install",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("bad config")),
    )
    with pytest.raises(ValueError, match="bad config"):
        RhinoVLAOnRPU(
            qin=object(), prefix_len=7, model=object(), action_bundle=object()
        )
    assert events == ["process-preflight"]

    events.clear()
    monkeypatch.setattr(
        pipeline, "_preflight_rhinovla_cold_install", lambda **kwargs: object()
    )

    def fail_allocator(enabled):
        events.append(("allocator", enabled))
        raise RuntimeError("allocator conflict")

    monkeypatch.setattr(torch.rpu, "set_caching_allocator", fail_allocator)
    with pytest.raises(RuntimeError, match="allocator conflict"):
        RhinoVLAOnRPU(
            qin=object(), prefix_len=7, model=object(), action_bundle=object()
        )
    assert events == [
        "process-preflight", "claim", ("allocator", True), "release"
    ]
