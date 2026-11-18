"""Caller-factory examples bind precision to actual installed expert storage."""
import ast
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from test_rhinovla_full_pipeline_precision import cold, precision

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def helpers(monkeypatch, cold, precision):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    convert = ModuleType("rpu_backend.adapters.rhinovla.convert")
    path = ROOT / "python/rpu_backend/adapters/rhinovla/convert.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if
                isinstance(node, ast.FunctionDef) and node.name == "_gather_expert_scales" or
                isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_EXPERT_PROJECTIONS" for t in node.targets)]
    convert.torch = torch
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), vars(convert))
    monkeypatch.setitem(sys.modules, convert.__name__, convert)
    monkeypatch.setitem(sys.modules, "rpu_backend.adapters.rhinovla.precision", precision)
    sys.modules["rpu_backend.api"].RhinoVLAPolicy = SimpleNamespace()
    return None, importlib.import_module("examples.runners.rhinovla.runner")


def policy_config(target, dtype):
    return {"example": {"target": target, "dtype": dtype}, "input": {}, "rpu_execution": {}}


def rhinovla_quant_config(tmp_path, *, full=False):
    config = policy_config("rhinovla", "w8a16")
    scope = "full-expert" if full else "expert"
    config["example"]["profile_id"] = f"rhinovla.v3.w8a16.{scope}.trusted-factory"
    factory = tmp_path / "factory.json"
    factory.write_text("{}")
    config["input"] = {
        "runtime_factory": "caller.integration:create_runtime",
        "factory_config": str(factory), "request": "caller-request.pt",
        "trust_model_code": True, "num_steps": 10,
        "expert_w8a16": True, "full_w8a16": full,
    }
    return config


def installed_rhinovla_policy(request, *, quantized=True, full=False):
    """Use real storage checks with a host stub for native installation only."""
    from rpu_backend.adapters.rhinovla.runtime import _RhinoVLAExecutionController

    def projection():
        weight = torch.ones(2, 2, dtype=torch.int8 if quantized else torch.float16)
        result = SimpleNamespace(weight=weight)
        if quantized:
            result.weight_scale = torch.ones(2, dtype=torch.float16)
        return result
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(**{name: projection() for name in ("q_proj", "k_proj", "v_proj", "o_proj")}),
        mlp=SimpleNamespace(**{name: projection() for name in ("gate_proj", "up_proj", "down_proj")}),
    )
    expert = SimpleNamespace(layers=[layer], _rpu_full_w8a16=full)
    if full:
        weights = tuple([torch.ones(2, 2, dtype=torch.int8)] if i % 2 == 0
                        else [torch.zeros(2, dtype=torch.float16)] for i in range(6))
        expert._rpu_full_w8_cond_owners = {
            "weights": weights,
            "scales": tuple([torch.ones(2, dtype=torch.float16)] for _ in range(3)),
        }
    runtime = SimpleNamespace(_rhinovla_retirement_children={"action_expert": expert})
    controller = object.__new__(_RhinoVLAExecutionController)
    controller.runtime, controller.action_expert = runtime, expert
    runtime._rhinovla_execution_controller = controller
    calls = []
    def close():
        if getattr(runtime, "_rhinovla_closed", False):
            return
        calls.append("close")
        # No native resource was created by this host stub. Retire its fake
        # ownership graph so the real controller finalizer has nothing to own.
        runtime._rhinovla_closed = True
        runtime._rhinovla_retirement_children.clear()
        del runtime._rhinovla_execution_controller
        controller.runtime = None
    request.addfinalizer(close)
    policy = SimpleNamespace(runtime=runtime, _rhinovla_execution_controller=controller,
                             predict=lambda **request: request["x0"], close=close)
    return policy, expert, calls


@pytest.mark.parametrize("key", ["expert_w8a16", "full_w8a16"])
@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_rhinovla_quantization_requires_exact_booleans_before_factory(helpers, monkeypatch, tmp_path, key, value):
    _, policies = helpers
    config = rhinovla_quant_config(tmp_path)
    config["input"][key] = value
    monkeypatch.setattr(policies, "_load_mapping", lambda *_: pytest.fail("input loaded before rejection"))
    with pytest.raises(ValueError, match=f"{key} must be boolean"):
        importlib.import_module("examples.common.preparation").prepare(config)


@pytest.mark.parametrize("dtype,expert,full", [("w8a16", False, False), ("w8a16", False, True),
                                              ("fp16", True, False), ("fp16", True, True)])
def test_rhinovla_dtype_must_match_explicit_quantization(helpers, tmp_path, dtype, expert, full):
    _, policies = helpers
    config = rhinovla_quant_config(tmp_path)
    config["example"]["dtype"] = dtype
    config["input"].update(expert_w8a16=expert, full_w8a16=full)
    with pytest.raises(ValueError, match="requires explicit|fp16 conflicts"):
        policies.validate_rhinovla_config(config)


@pytest.mark.parametrize("missing", ["expert_w8a16", "full_w8a16"])
def test_rhinovla_w8_does_not_infer_missing_scope(helpers, tmp_path, missing):
    _, policies = helpers
    config = rhinovla_quant_config(tmp_path)
    del config["input"][missing]
    with pytest.raises(ValueError, match="requires explicit"):
        policies.validate_rhinovla_config(config)


@pytest.mark.parametrize("full", [False, True])
def test_rhinovla_w8_scope_is_bound_to_catalog_identity(helpers, tmp_path, full):
    _, policies = helpers
    config = rhinovla_quant_config(tmp_path, full=full)
    config["input"]["full_w8a16"] = not full
    with pytest.raises(ValueError, match="conflicts with the selected profile_id"):
        policies.validate_rhinovla_config(config)


@pytest.mark.parametrize("overrides", [{"expert_w8a16": False}, {"expert_w8a16": 1},
                                       {"full_w8a16": True}, {"full_w8a16": 0}])
def test_rhinovla_factory_json_quantization_conflicts_reject_before_factory(helpers, monkeypatch, tmp_path, overrides):
    _, policies = helpers
    import rpu_backend.api as api
    config = rhinovla_quant_config(tmp_path)
    Path(config["input"]["factory_config"]).write_text(json.dumps(overrides))
    monkeypatch.setattr(policies, "_load_mapping", lambda *_: {"x0": torch.zeros(1)})
    monkeypatch.setattr(api, "RhinoVLAPolicy", SimpleNamespace(from_factory=lambda *_a, **_k: pytest.fail("factory ran")))
    with pytest.raises(ValueError, match="factory_config.* conflicts"):
        importlib.import_module("examples.common.preparation").prepare(config)


def test_rhinovla_legacy_fp16_rejects_hidden_json_quantization(helpers, monkeypatch, tmp_path):
    _, policies = helpers
    config = rhinovla_quant_config(tmp_path)
    config["example"]["dtype"] = "fp16"
    del config["input"]["expert_w8a16"], config["input"]["full_w8a16"]
    Path(config["input"]["factory_config"]).write_text('{"expert_w8a16":true}')
    monkeypatch.setattr(policies, "_load_mapping", lambda *_: {"x0": torch.zeros(1)})
    with pytest.raises(ValueError, match="factory_config.expert_w8a16 conflicts"):
        importlib.import_module("examples.common.preparation").prepare(config)


@pytest.mark.parametrize("full", [False, True])
def test_rhinovla_quantization_reaches_factory_and_matches_real_storage(helpers, monkeypatch, tmp_path, request, full):
    _, policies = helpers
    import rpu_backend.api as api
    config = rhinovla_quant_config(tmp_path, full=full)
    policy, expert, calls = installed_rhinovla_policy(request, full=full)
    monkeypatch.setattr(policies, "_load_mapping", lambda *_: {"x0": torch.zeros(1)})
    monkeypatch.setattr(policies, "_checkpoint", lambda *_: Path("caller-checkpoint"))
    def create(name, values, **kwargs):
        assert values == {"checkpoint": "caller-checkpoint", "steps": 10, "trust_model_code": True,
                          "expert_w8a16": True, "full_w8a16": full}
        return policy
    monkeypatch.setattr(api, "RhinoVLAPolicy", SimpleNamespace(from_factory=create))
    infer, owner = importlib.import_module("examples.common.preparation").prepare(config)
    assert owner is policy and torch.equal(infer(), torch.zeros(1))
    assert calls == []
    owner.close()
    assert calls == ["close"]


@pytest.mark.parametrize("failure", ["ignored_expert", "mixed_weights", "ignored_full", "label_only_full",
                                      "wrong_adarms_dtype", "wrong_owner", "wrong_controller"])
def test_rhinovla_ignored_or_mismatched_quantization_closes_factory_owner(helpers, monkeypatch, tmp_path, request, failure):
    _, policies = helpers
    import rpu_backend.api as api
    full = failure in {"ignored_full", "label_only_full", "wrong_adarms_dtype"}
    config = rhinovla_quant_config(tmp_path, full=full)
    policy, expert, calls = installed_rhinovla_policy(request, quantized=failure != "ignored_expert", full=full)
    if failure == "mixed_weights":
        expert.layers[0].mlp.down_proj.weight = expert.layers[0].mlp.down_proj.weight.half()
    elif failure == "ignored_full":
        expert._rpu_full_w8a16 = False
    elif failure == "label_only_full":
        del expert._rpu_full_w8_cond_owners
    elif failure == "wrong_adarms_dtype":
        expert._rpu_full_w8_cond_owners["weights"][0][0] = torch.ones(2, 2, dtype=torch.float16)
    elif failure == "wrong_owner":
        policy.runtime._rhinovla_retirement_children["action_expert"] = object()
    elif failure == "wrong_controller":
        policy._rhinovla_execution_controller = object()
    monkeypatch.setattr(policies, "_load_mapping", lambda *_: {"x0": torch.zeros(1)})
    monkeypatch.setattr(policies, "_checkpoint", lambda *_: Path("caller-checkpoint"))
    monkeypatch.setattr(api, "RhinoVLAPolicy", SimpleNamespace(from_factory=lambda *_a, **_k: policy))
    with pytest.raises(ValueError, match="RhinoVLA"):
        importlib.import_module("examples.common.preparation").prepare(config)
    assert calls == ["close"]


@pytest.mark.parametrize("filename,full", [("fp16.toml", None), ("w8a16_action.toml", False),
                                          ("w8a16_action_adarms.toml", True)])
def test_rhinovla_v3_templates_keep_cold_optimizations_without_widening_precision(monkeypatch, filename, full):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "examples"))
    common = importlib.import_module("examples.common.config")
    config = common.load_config(root / "examples/configs/rhinovla/v3" / filename)
    gates = config["example"]["opt_in"]
    for name in ("VISION_BATCH_VIEWS", "VISION_PREP_CACHE", "DENOISE_STATIC_CONTEXT_CACHE",
                 "PRECOMPUTE_ADARMS", "GATED_NO_SUB", "FUSED_SILU_MUL", "EXPERT_FUSIONS",
                 "ADARMS_RESIDENT", "ALIGNED_KV"):
        assert gates["RPU_RHINOVLA_" + name] == "1"
    for name in ("FULL_W8A16", "EXPERT_W8A16", "PACKED_QKV", "SKIP_ADARMS_GEMV"):
        assert gates["RPU_RHINOVLA_" + name] == "0"
    if full is None:
        assert "expert_w8a16" not in config["input"] and "full_w8a16" not in config["input"]
    else:
        assert config["input"]["expert_w8a16"] is True
        assert config["input"]["full_w8a16"] is full
