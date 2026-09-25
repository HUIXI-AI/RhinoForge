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


@pytest.mark.parametrize("expert,full", [(False, False), (True, False), (True, True)])
def test_explicit_pipeline_expert_scope_keeps_other_components_fp16(monkeypatch, expert, full):
    from types import SimpleNamespace as NS
    from rpu_backend.runtime import hw_attrs

    for name in tuple(__import__("os").environ):
        if name.startswith("RPU_RHINOVLA_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("RPU_RHINOVLA_PRECOMPUTE_ADARMS", "1")
    monkeypatch.setattr(hw_attrs, "validate_preinstall", lambda _: None)
    model = NS(qwen=NS(model=NS(model=NS(language_model=object(), visual=object()),
        config=NS(text_config=NS(num_hidden_layers=28, num_key_value_heads=8,
                                 head_dim=128), vision_config=object()))))
    cfg = NS(depth=18, mlp_dim=3072, action_horizon=30, use_state_token=True,
             width=1024, num_attention_heads=16, num_key_value_heads=8, head_dim=128)
    args = dict(qin={}, prefix_len=230, steps=10, model=model,
                action_bundle=(object(), NS(), object(), cfg),
                flow_direction="official_descending", expert_w8a16=expert,
                full_expert_w8a16=full, rpu_execution={"action": {"denoise_unroll": True}})
    cold = pipeline._preflight_rhinovla_cold_install(**args)
    assert cold["expert_w8a16"] is expert
    assert cold["full_expert_w8a16"] is full
    assert cold["full_w8a16"] is False
    if full:
        monkeypatch.setenv("RPU_RHINOVLA_PRECOMPUTE_ADARMS", "0")
        with pytest.raises(ValueError, match="PRECOMPUTE_ADARMS"):
            pipeline._preflight_rhinovla_cold_install(**args)
    monkeypatch.setenv("RPU_RHINOVLA_FULL_W8A16", "1")
    if not full:
        with pytest.raises(ValueError, match="conflicts"):
            pipeline._preflight_rhinovla_cold_install(**args)
