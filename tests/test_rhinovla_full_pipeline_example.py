"""The explicit full-pipeline catalog scope must inspect installed child storage."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from test_rhinovla_factory_examples import installed_rhinovla_policy, helpers
from test_rhinovla_full_pipeline_precision import cold, precision

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "examples/configs/rhinovla/v3"


@pytest.fixture(params=["w8a16_pipeline.toml",
                        "w8a16_pipeline_high_fused_vector_qk.toml",
                        "rtc_w8a16_pipeline.toml",
                        "rtc_w8a16_pipeline_high_fused_vector_qk.toml"])
def config_path(request):
    return CONFIG_DIR / request.param


@pytest.fixture
def modules(monkeypatch, helpers):
    monkeypatch.syspath_prepend(str(ROOT))
    name = "rpu_backend.adapters.rhinovla.rtc"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "python/rpu_backend/adapters/rhinovla/rtc.py")
    rtc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rtc)
    monkeypatch.setitem(sys.modules, name, rtc)
    return importlib.import_module("examples.common.config"), importlib.import_module("examples.runners.rhinovla.runner")


def test_shared_runner_selects_caller_factory_and_preserves_controlled_v1(modules, monkeypatch):
    common, policies = modules
    runner = importlib.import_module("examples.common.measurement")
    selected = []
    monkeypatch.setattr(policies, "_factory", lambda runner: selected.append("factory"))
    monkeypatch.setattr(policies, "_official", lambda runner: selected.append("controlled"))
    runner._prepare(common.load_config(CONFIG_DIR / "w8a16_pipeline.toml"))
    runner._prepare(common.load_config(CONFIG_DIR.parent / "v1/fp16.toml"))
    assert selected == ["factory", "controlled"]


def test_full_pipeline_example_has_distinct_scope_and_explicit_acc16(modules, config_path):
    common, policies = modules
    config = common.load_config(config_path)
    policies.validate_rhinovla_config(config)
    assert policies._rhinovla_quantization(config) == {"expert_w8a16": True, "full_w8a16": True}
    assert policies._rhinovla_precision_profile(config) in policies._RHINOVLA_FULL_PIPELINE_PROFILES
    components = config["rpu_execution"]["components"]
    assert components["language_model"]["prefill"]["linear_acc32"] is False
    assert components["vision_encoder"]["vision"]["linear_acc32"] is False
    assert components["language_model"]["prefill"]["padding_rows"] == 10
    assert config["run"]["warmup"] == 2 and config["run"]["runs"] == 5
    assert config["cpu"]["threads"] == 12
    assert config["run"]["inference_mode"] is False
    config["example"]["opt_in"]["RPU_RHINOVLA_FULL_W8A16"] = "0"
    with pytest.raises(ValueError, match="both explicit W8"):
        policies.validate_rhinovla_config(config)


@pytest.mark.parametrize("failure", [None, "inventory", "child_identity"])
def test_real_examples_preparation_requires_full_inventory_before_predict(
        modules, monkeypatch, tmp_path, request, failure, config_path):
    from examples.common import preparation
    common, policies = modules
    import rpu_backend.api as api
    from rpu_backend.adapters.rhinovla import precision
    config = common.load_config(config_path)
    rtc = config["example"]["profile_id"] in policies._RHINOVLA_RTC_PROFILES
    factory_path = tmp_path / "factory.json"
    factory_path.write_text(json.dumps({"rtc": True, "qin": "qin.pt"} if rtc else {}))
    config["input"].update(factory_config=str(factory_path), runtime_factory=(
        policies._RHINOVLA_V3_FACTORY if rtc else "caller:create"))
    policy, expert, calls = installed_rhinovla_policy(request, full=True)
    runtime, controller = policy.runtime, policy._rhinovla_execution_controller
    runtime.er = expert
    runtime.text_model, runtime.vision_model = object(), object()
    controller.text_model, controller.vision_model = runtime.text_model, runtime.vision_model
    runtime._rhinovla_retirement_children.update(
        language_model=runtime.text_model, vision_encoder=runtime.vision_model)
    if failure == "child_identity":
        controller.vision_model = object()
    inspected = []
    def inspect(actual):
        assert actual is runtime
        inspected.append(actual)
        if failure == "inventory":
            raise ValueError("RhinoVLA full W8 requires actual vision storage")
        return {}
    monkeypatch.setattr(precision, "full_w8_inventory", inspect)
    payload = {"x0": torch.zeros(1)}
    if rtc:
        payload.update(delay=0, action_prefix=torch.empty(1, 0, 96))
    monkeypatch.setattr(policies, "_load_mapping", lambda *_: payload)
    monkeypatch.setattr(policies, "_checkpoint", lambda *_: Path("caller-checkpoint"))
    monkeypatch.setattr(api, "RhinoVLAPolicy", SimpleNamespace(from_factory=lambda *_a, **_k: policy))
    if failure:
        with pytest.raises(ValueError, match="RhinoVLA"):
            preparation.prepare(config)
        assert calls == ["close"]
        assert bool(inspected) is (failure == "inventory")
    else:
        infer, owner = preparation.prepare(config)
        assert inspected == [runtime] and calls == []
        assert torch.equal(infer(), torch.zeros(1))
        owner.close()
