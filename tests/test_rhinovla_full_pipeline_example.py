"""The explicit full-pipeline catalog scope must inspect installed child storage."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_catalog_policy_examples import installed_rhinovla_policy

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "examples/configs/rhinovla/v3"


@pytest.fixture(params=["full_pipeline_w8a16.toml",
                        "full_pipeline_w8a16_high_precision_vector_qk.toml"])
def config_path(request):
    return CONFIG_DIR / request.param


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    return importlib.import_module("_common"), importlib.import_module("_policy_runners")


def test_full_pipeline_example_has_distinct_scope_and_explicit_acc16(modules, config_path):
    common, policies = modules
    config = common.load_config(config_path)
    policies.validate_policy_config(config)
    assert policies._rhinovla_quantization(config) == {"expert_w8a16": True, "full_w8a16": True}
    assert config["example"]["profile_id"] in policies._RHINOVLA_FULL_PIPELINE_PROFILES
    components = config["rpu_execution"]["components"]
    assert components["language_model"]["prefill"]["linear_acc32"] is False
    assert components["vision_encoder"]["vision"]["linear_acc32"] is False
    assert components["language_model"]["prefill"]["padding_rows"] == 10
    assert config["run"]["warmup"] == 1 and config["run"]["runs"] == 3
    assert config["run"]["torch_num_threads"] == 12
    assert config["run"]["inference_mode"] is False
    config["example"]["opt_in"]["RPU_RHINOVLA_FULL_W8A16"] = "0"
    with pytest.raises(ValueError, match="both explicit W8"):
        policies.validate_policy_config(config)


@pytest.mark.parametrize("failure", [None, "inventory", "child_identity"])
def test_real_examples_preparation_requires_full_inventory_before_predict(
        modules, monkeypatch, tmp_path, request, failure, config_path):
    common, policies = modules
    import rpu_backend.api as api
    from rpu_backend.adapters.rhinovla import precision
    config = common.load_config(config_path)
    factory_path = tmp_path / "factory.json"
    factory_path.write_text("{}")
    config["input"].update(factory_config=str(factory_path), runtime_factory="caller:create")
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
    monkeypatch.setattr(policies, "_load_mapping", lambda *_: {"x0": torch.zeros(1)})
    monkeypatch.setattr(policies, "_checkpoint", lambda *_: Path("caller-checkpoint"))
    monkeypatch.setattr(api, "RhinoVLAPolicy", SimpleNamespace(from_factory=lambda *_a, **_k: policy))
    if failure:
        with pytest.raises(ValueError, match="RhinoVLA"):
            policies.prepare_policy(config)
        assert calls == ["close"]
        assert bool(inspected) is (failure == "inventory")
    else:
        infer, owner = policies.prepare_policy(config)
        assert inspected == [runtime] and calls == []
        assert torch.equal(infer(), torch.zeros(1))
        owner.close()
