"""Cold, explicit all-stage G05 ACC16 experiment; no board required."""
import types

import pytest
import torch

from rpu_backend.adapters.g05 import action, quant
from rpu_backend.adapters.qwen3_5 import text, vision


@pytest.mark.parametrize("mode", ["acc32", "acc16"])
@pytest.mark.parametrize("component,stage", [
    ("vision_encoder", "vision"), ("language_model", "prefill"),
])
def test_accumulation_config_accepts_real_frozen_component_views(mode, component, stage):
    from rpu_backend.adapters.g05 import runtime

    root, components = runtime._resolve_g05_execution(
        None, max_seq_len=2048, entry_point="G05 precision test")
    config = components[component]
    assert not isinstance(config, dict)
    assert not isinstance(config[stage], dict)
    original = dict(config[stage])
    result = quant.linear_execution_config(config, mode, stage=stage)
    assert isinstance(result, dict) and isinstance(result[stage], dict)
    assert result[stage] == {**original, "linear_acc32": mode == "acc32"}
    if stage == "vision":
        assert vision._qwen35_vision_precision(result, certified_g05=True,
            g05_w8a16=True, g05_acc16_experiment=mode == "acc16") == (
                mode == "acc32", False)
    else:
        assert text._validate_qwen35_linear_accumulation(None, result) is (mode == "acc32")
    result[stage]["chunk_size"] = 128
    result["unrelated"] = {"value": 1}
    assert dict(config[stage]) == original
    assert "linear_acc32" not in config[stage] and "unrelated" not in config
    assert dict(root) == {}


def test_acc16_admission_precision_conflict_and_immutable_install(monkeypatch):
    from test_g05_adapter_contracts import _load_adapter

    runtime = _load_adapter()
    for value in (None, True, "fp16", "ACC16"):
        with pytest.raises(ValueError, match="linear_accumulation"):
            runtime.patch_g05_policy_for_rpu(object(), linear_accumulation=value)
    for dtype in ("fp16", "w8a16"):
        assert quant.validate_linear_accumulation(
            "acc16", weight_dtype=dtype, exact_k1=True) == "acc16"
        with pytest.raises(ValueError, match="exact K1"):
            quant.validate_linear_accumulation("acc16", weight_dtype=dtype, exact_k1=False)
    policy = types.SimpleNamespace(_rpu_g05_policy_ready=True,
        _rpu_g05_weight_dtype="w8a16", _rpu_g05_linear_accumulation="acc16")
    monkeypatch.setattr(runtime, "_g05_policy_runtime_ready", lambda _: True)
    assert runtime.patch_g05_policy_for_rpu(
        policy, weight_dtype="w8a16", linear_accumulation="acc16") is policy
    with pytest.raises(ValueError, match="linear_accumulation conflicts"):
        runtime.patch_g05_policy_for_rpu(policy, weight_dtype="w8a16")
    cfg = {"prefill": {"chunk_size": 256}}
    assert quant.linear_execution_config(cfg, "acc16") == {
        "prefill": {"chunk_size": 256, "linear_acc32": False}}
    assert cfg == {"prefill": {"chunk_size": 256}}
    for bad in (True, 0, "false"):
        with pytest.raises(ValueError, match="conflicts"):
            quant.linear_execution_config({"prefill": {"linear_acc32": bad}}, "acc16")
    with pytest.raises(ValueError, match="exact K1"):
        action.patch_g05_action_expert_for_rpu(object(), weight_dtype="w8a16",
                                               linear_accumulation="acc16")


@pytest.mark.parametrize("weight_dtype", ["fp16", "w8a16"])
def test_acc16_vlm_real_installer_propagates_before_ready(monkeypatch, request, weight_dtype):
    from test_g05_adapter_contracts import _load_adapter, _g05_vlm_installer_fixture

    runtime = _load_adapter()
    events = []
    model, vlm = _g05_vlm_installer_fixture(
        monkeypatch, runtime, events, request, linear_acc32=False)
    monkeypatch.setattr(runtime, "quantize_g05_text_weights", lambda _: None)
    kwargs = dict(_allow_certified_g05_k1=True, weight_dtype=weight_dtype,
                  linear_accumulation="acc16")
    assert runtime.patch_g05_vlm_for_rpu(model, **kwargs) is model
    assert ("linear_acc32", 23, False) in events
    assert ("linear_acc32", 23, True) not in events
    assert vlm._rpu_qwen3_5.linear_acc32 is False
    assert vlm._rpu_g05_linear_accumulation == "acc16"
    before = list(events)
    assert runtime.patch_g05_vlm_for_rpu(model, **kwargs) is model
    with pytest.raises(ValueError, match="linear_accumulation conflicts"):
        runtime.patch_g05_vlm_for_rpu(model, _allow_certified_g05_k1=True,
                                    weight_dtype=weight_dtype)
    assert events == before


@pytest.mark.parametrize("weight_dtype", ["fp16", "w8a16"])
def test_acc16_action_real_installer_uses_cold_shared_precision(monkeypatch, weight_dtype):
    from test_g05_w8a16_adapter import _toy_backbone

    expert = _toy_backbone(action=True).eval()
    monkeypatch.setattr(action, "_check_g05_action_profile", lambda _: None)
    monkeypatch.setattr(quant, "quantize_g05_text_weights", lambda *a, **kw: None)
    calls = []

    class ReachedColdInstall(Exception):
        pass

    def install(owner, config, **kwargs):
        calls.append(kwargs)
        assert owner is expert
        assert kwargs["execution_config"] == {"prefill": {"linear_acc32": False}}
        assert kwargs.get("_g05_w8a16", False) is (weight_dtype == "w8a16")
        raise ReachedColdInstall()

    monkeypatch.setattr(text, "install_qwen3_5_text_for_rpu", install)
    with pytest.raises(ReachedColdInstall):
        action.patch_g05_action_expert_for_rpu(expert, weight_dtype=weight_dtype,
            linear_accumulation="acc16", _g05_acc16_experiment=True)
    assert len(calls) == 1


@pytest.mark.parametrize("w8", [False, True])
def test_acc16_vision_private_scope_and_native_cold_setter(monkeypatch, w8):
    from rpu_backend.runtime import _native_retirement
    from test_qwen3_5_vision_helpers import _runtime_capable_vision

    cfg = {"vision": {"linear_acc32": False}}
    assert vision._qwen35_vision_precision(None, certified_g05=True) == (True, False)
    with pytest.raises(ValueError, match="requires ACC32"):
        vision._qwen35_vision_precision(cfg, certified_g05=True)
    for changes in ({"certified_g05": False},
                    {"execution_config": None},
                    {"execution_config": {"vision": {"linear_acc32": True}}}):
        kwargs = dict(execution_config=cfg, certified_g05=True,
                      g05_acc16_experiment=True, g05_w8a16=w8)
        kwargs.update(changes)
        with pytest.raises(ValueError, match="exact K1"):
            vision._qwen35_vision_precision(**kwargs)
    assert vision._qwen35_vision_precision(cfg, certified_g05=True,
        g05_acc16_experiment=True, g05_w8a16=w8) == (False, False)
    owner = _runtime_capable_vision()
    calls = []
    for name in ("qwen3_5_vision_set_chunk_size_cap", "qwen3_5_vision_set_cache_capacity",
                 "qwen3_5_vision_resolve_prefill_domain",
                 "qwen3_5_vision_set_prefill_chunk_size", "qwen3_5_vision_set_weight_scales",
                 "qwen3_5_vision_destroy"):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *_: None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_create", lambda: 41, raising=False)
    monkeypatch.setattr(_native_retirement, "_InstalledNativeResource",
        lambda *a, **kw: types.SimpleNamespace(finalizer=None, handle=None))

    class ReachedPrecision(Exception):
        pass

    def set_precision(handle, enabled):
        calls.append((handle, enabled))
        raise ReachedPrecision()

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_set_linear_acc32", set_precision, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_set_gelu_erf_ultra",
        lambda *_: pytest.fail("G05 must retain original GELU"), raising=False)
    with pytest.raises(ReachedPrecision):
        vision._install_qwen3_5_vision_for_rpu_impl(owner, execution_config=cfg,
            _allow_certified_g05_k1=True, _g05_w8a16=w8, _g05_acc16_experiment=True)
    assert calls == [(41, False)]
