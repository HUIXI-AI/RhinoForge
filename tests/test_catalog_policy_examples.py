"""Public catalog policy calls retain cold precision and runtime ownership."""
from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch


@pytest.fixture
def helpers(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("_policy_examples"), importlib.import_module("_policy_runners")


def pi_config(precision="fp16", cores=8, cameras=3):
    return {"pi05": {"precision": precision},
            "input": {"checkpoint": "/path/to/checkpoint", "batch": "/path/to/batch.pt",
                      "cameras": cameras, "text_tokens": 32, "num_steps": 10},
            "run": {"warmup": 1, "runs": 3}, "rpu_execution": {"model": {"num_cores": cores}}}


@pytest.mark.parametrize("cores,cameras,chunk", [(8, 2, 272), (8, 3, 400), (4, 3, None), (6, 3, None)])
def test_pi_cold_geometry_preserves_cores_and_pair_without_mutating_config(helpers, cores, cameras, chunk):
    pi, _ = helpers
    config = pi_config(cores=cores, cameras=cameras)
    before = copy.deepcopy(config)
    pi.validate_pi05_config(config)
    execution = pi._execution(config)
    assert execution["model"]["num_cores"] == cores
    assert execution.get("prefill", {}).get("chunk_size") == chunk
    assert config == before


@pytest.mark.parametrize("update,match", [
    ({"input": {"noise": "/tmp/noise.pt"}}, "unsupported keys"),
    ({"input": {"num_steps": 9}}, "num_steps=10"),
    ({"run": {"hwperf": True}}, "hardware trace"),
    ({"run": {"runs": 0}}, "runs"),
    ({"run": {"warmup": -1}}, "warmup"),
    ({"input": {"cameras": True}}, "cameras"),
    ({"pi05": {"precision": "w4a16"}}, "precision"),
    ({"rpu_execution": {"prefill": {"chunk_size": 384}}}, "paired"),
])
def test_pi_rejects_unsupported_or_silently_different_workloads(helpers, update, match):
    pi, _ = helpers
    config = pi_config()
    for section, fields in update.items():
        config[section].update(fields)
    with pytest.raises(ValueError, match=match):
        pi.validate_pi05_config(config)


def test_pi_component_override_and_generic_admission(helpers):
    pi, _ = helpers
    config = pi_config(cameras=2)
    config["rpu_execution"]["components"] = {"language_model": {"prefill": {"chunk_size": "auto"}},
                                             "action_expert": {"action": {"linear_acc32": True}}}
    execution = pi._execution(config)
    assert execution["components"]["language_model"]["prefill"]["chunk_size"] == 272
    assert execution["components"]["action_expert"]["action"]["linear_acc32"] is True
    config = pi_config("w8a16_action_nvfp4", cores=4)
    with pytest.raises(ValueError, match="TP4/TP6"):
        pi.validate_pi05_config(config)
    config = pi_config()
    config["run"]["warmup"] = 0
    pi.validate_pi05_config(config)


@pytest.mark.parametrize("precision,cores,expected", [
    ("fp16", 8, "fp16"), ("w8a16", 8, "w8a16"),
    ("w8a16_action_nvfp4", 8, "w8_action_nvfp4"),
    ("w8a16_prefill_w8a8_action_nvfp4", 8, "w8_prefill_a8_action_nvfp4"),
    ("fp16", 4, None), ("w8a16", 6, None),
])
def test_pi_public_policy_precision_and_lifetime(helpers, monkeypatch, tmp_path, precision, cores, expected):
    pi, _ = helpers
    import rpu_backend.api as api

    config = pi_config(precision, cores)
    config["input"]["checkpoint"] = str(tmp_path)
    quant = {"method": "w8a16"}
    if "nvfp4" in precision:
        quant = {"method": "nvfp4a16", "int4_components": ["expert"],
                 "mixed_int8_suffixes": ["k_proj.weight", "v_proj.weight"],
                 "nvfp4_abi": "striped_v2", "nvfp4_block_size": 16}
    if precision != "fp16":
        (tmp_path / "rpu_quant_config.json").write_text(json.dumps(quant))
    images = {f"observation.images.camera{i}": torch.ones(1, 3, 2, 2) for i in range(3)}
    batch = {**images, "observation.language.tokens": torch.ones(1, 32, dtype=torch.long),
             "observation.language.attention_mask": torch.ones(1, 32, dtype=torch.bool)}
    path = tmp_path / "batch.pt"
    torch.save(batch, path)
    config["input"]["batch"] = str(path)
    calls = []
    policy = SimpleNamespace(_lerobot_policy=SimpleNamespace(config=SimpleNamespace(
        image_features=images, chunk_size=50, max_action_dim=32, num_inference_steps=10)))
    policy.to = lambda device: calls.append(("to", device))
    policy.prepare_graphs = lambda batch, **kwargs: calls.append(("prepare", kwargs))
    policy.predict_action_chunk = lambda batch, **kwargs: calls.append(("infer", kwargs)) or torch.ones(1, 50, 32)
    policy.close = lambda: calls.append(("close",))
    env_before = dict(os.environ)
    def load(checkpoint, **kwargs):
        calls.append(("load", kwargs))
        assert kwargs["rpu_execution"]["model"]["num_cores"] == cores
        if cores != 8:
            assert os.environ["RPU_PI05_SIGLIP_BATCH"] == "0"
            assert os.environ["RPU_PI05_VISION_OWNER_LN"] == "0"
            assert os.environ["RPU_PI05_SIGLIP_W8A16"] == str(int(precision != "fp16"))
        return policy
    monkeypatch.setattr(api, "Pi05Policy", SimpleNamespace(from_pretrained=load))
    infer, owner = pi.prepare_pi05(config)
    selected = calls[0][1]["optimized_profile"]
    assert (selected["precision"] if selected else None) == expected
    assert calls[2] == ("prepare", {"num_steps": 10, "precompute_adarms": cores == 8})
    assert infer().shape == (1, 50, 32)
    owner.close()
    owner.close()
    assert calls.count(("close",)) == 1
    assert dict(os.environ) == env_before


def test_pi_cold_environment_restored_after_setup_error(helpers, monkeypatch, tmp_path):
    pi, _ = helpers
    import rpu_backend.api as api
    config = pi_config(cores=4)
    config["input"]["checkpoint"] = str(tmp_path)
    path = tmp_path / "batch.pt"
    torch.save({}, path)
    config["input"]["batch"] = str(path)
    before = dict(os.environ)
    def fail(*args, **kwargs):
        assert os.environ["RPU_PI05_SIGLIP_BATCH"] == "0"
        raise RuntimeError("construction failed")
    monkeypatch.setattr(api, "Pi05Policy", SimpleNamespace(from_pretrained=fail))
    with pytest.raises(RuntimeError, match="construction failed"):
        pi.prepare_pi05(config)
    assert dict(os.environ) == before


def test_pi_undeclared_diagnostic_input_replacement_is_rejected(helpers, monkeypatch):
    pi, _ = helpers
    monkeypatch.setenv("RPU_PI05_LOAD_NOISE", "/caller/noise.pt")
    with pytest.raises(ValueError, match="diagnostic input replacement"):
        pi.prepare_pi05(pi_config())


def policy_config(target, dtype="fp16"):
    return {"example": {"target": target, "dtype": dtype, "registry_alias": "model", "opt_in": {}},
            "input": {}, "run": {"warmup": 1, "runs": 3}, "rpu_execution": {}}


@pytest.mark.parametrize("target", ["gr00t", "internvla_n1", "wall_oss_rtc", "g05", "lingbot"])
def test_policy_catalog_does_not_expose_removed_or_different_admission_paths(helpers, target):
    _, policies = helpers
    with pytest.raises(ValueError, match="no public catalog"):
        policies.validate_policy_config(policy_config(target))


def test_wall_uses_actual_public_api_and_owns_facade_on_install_failure(helpers, monkeypatch):
    _, policies = helpers
    import rpu_backend.api as api
    config = policy_config("wall_oss", "w4a16-pgrp")
    config["input"].update(images=["camera.png"], camera_names=["face_view"], instruction="move",
                           proprioception=[0.0], fp16_registry_alias="fp16-model")
    monkeypatch.setattr(policies, "_images", lambda inputs: ["image"])
    monkeypatch.setattr(policies, "_checkpoint", lambda config: Path("checkpoint"))
    monkeypatch.setattr(policies, "_model_path", lambda *args, **kwargs: Path("fp16"))
    calls = []
    # Spec the real public facade: a fictional preflight_images method must
    # fail here instead of being invented by a permissive test double.
    policy = create_autospec(api.WallOssPolicy, instance=True, spec_set=True)
    def install(device):
        calls.append("install")
        raise RuntimeError("install failed")
    policy.to.side_effect = install
    policy.close.side_effect = lambda: calls.append("close")
    def load(checkpoint, **kwargs):
        assert kwargs["w4a16"] is True and kwargs["w8a16"] is False
        assert kwargs["fp16_ckpt_dir"] == "fp16"
        return policy
    loader = create_autospec(api.WallOssPolicy.from_checkpoint, side_effect=load)
    monkeypatch.setattr(api, "WallOssPolicy", SimpleNamespace(from_checkpoint=loader))
    with pytest.raises(RuntimeError, match="install failed"):
        policies.prepare_policy(config)
    assert calls == ["install", "close"]
    config["input"]["rtc_prefix_actions"] = [0.0]
    with pytest.raises(ValueError, match="unsupported wall_oss input fields"):
        policies.validate_policy_config(config)


@pytest.mark.parametrize("prefix", [None, [215, 225]])
def test_lingbot2_graph_scope_uses_policy_default_unless_explicit(helpers, monkeypatch, prefix):
    _, policies = helpers
    import rpu_backend.api as api
    config = policy_config("lingbot2", "fp16" if prefix is None else "w8a16")
    config["example"]["opt_in"] = {"RPU_LINGBOT2_ALLOW_UNVALIDATED": "1"}
    config["input"] = {"observation": "caller.pt"}
    if prefix is not None:
        config["input"]["prefix_length_range"] = prefix
    observation = {"value": torch.zeros(1)}
    monkeypatch.setattr(policies, "_load_mapping", lambda *_: observation)
    monkeypatch.setattr(policies, "_checkpoint", lambda *_: Path("checkpoint"))
    policy = create_autospec(api.Lingbot2Policy, instance=True, spec_set=True)
    policy.infer.return_value = SimpleNamespace(actions_full=torch.ones(1, 2))
    loader = create_autospec(api.Lingbot2Policy.from_checkpoint, return_value=policy)
    monkeypatch.setattr(api, "Lingbot2Policy", SimpleNamespace(from_checkpoint=loader))
    infer, owner = policies.prepare_policy(config)
    assert owner is policy
    assert loader.call_args.kwargs["dtype"] == config["example"]["dtype"]
    policy.prepare_graphs.assert_called_once_with(obs=observation, prefix_length_range=prefix)
    assert torch.equal(infer(), torch.ones(1, 2))


def test_rhinovla_uses_trusted_caller_factory_and_explicit_request(helpers, monkeypatch, tmp_path):
    _, policies = helpers
    import rpu_backend.api as api
    config = policy_config("rhinovla")
    factory = tmp_path / "factory.json"
    factory.write_text('{"model_root":"/caller/source"}')
    request = tmp_path / "request.pt"
    torch.save({"x0": torch.zeros(1, 2)}, request)
    config["input"] = {"runtime_factory": "caller.integration:create_runtime", "factory_config": str(factory),
                       "request": str(request), "trust_model_code": True, "num_steps": 10}
    monkeypatch.setattr(policies, "_checkpoint", lambda config: Path("caller-checkpoint"))
    calls = []
    policy = SimpleNamespace(predict=lambda **request: request["x0"], close=lambda: calls.append("close"))
    def create(name, values, **kwargs):
        assert name == "caller.integration:create_runtime"
        assert values == {"model_root": "/caller/source", "checkpoint": "caller-checkpoint",
                          "steps": 10, "trust_model_code": True}
        assert kwargs["rpu_execution"] == {}
        return policy
    monkeypatch.setattr(api, "RhinoVLAPolicy", SimpleNamespace(from_factory=create))
    infer, owner = policies.prepare_policy(config)
    assert torch.equal(infer(), torch.zeros(1, 2))
    assert owner is policy
    owner.close()
    assert calls == ["close"]
    factory.write_text('{"steps":9}')
    with pytest.raises(ValueError, match="factory_config.steps conflicts"):
        policies.prepare_policy(config)


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
        policies.prepare_policy(config)


@pytest.mark.parametrize("dtype,expert,full", [("w8a16", False, False), ("w8a16", False, True),
                                              ("fp16", True, False), ("fp16", True, True)])
def test_rhinovla_dtype_must_match_explicit_quantization(helpers, tmp_path, dtype, expert, full):
    _, policies = helpers
    config = rhinovla_quant_config(tmp_path)
    config["example"]["dtype"] = dtype
    config["input"].update(expert_w8a16=expert, full_w8a16=full)
    with pytest.raises(ValueError, match="requires explicit|fp16 conflicts"):
        policies.validate_policy_config(config)


@pytest.mark.parametrize("missing", ["expert_w8a16", "full_w8a16"])
def test_rhinovla_w8_does_not_infer_missing_scope(helpers, tmp_path, missing):
    _, policies = helpers
    config = rhinovla_quant_config(tmp_path)
    del config["input"][missing]
    with pytest.raises(ValueError, match="requires explicit"):
        policies.validate_policy_config(config)


@pytest.mark.parametrize("full", [False, True])
def test_rhinovla_w8_scope_is_bound_to_catalog_identity(helpers, tmp_path, full):
    _, policies = helpers
    config = rhinovla_quant_config(tmp_path, full=full)
    config["input"]["full_w8a16"] = not full
    with pytest.raises(ValueError, match="conflicts with the selected profile_id"):
        policies.validate_policy_config(config)


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
        policies.prepare_policy(config)


def test_rhinovla_legacy_fp16_rejects_hidden_json_quantization(helpers, monkeypatch, tmp_path):
    _, policies = helpers
    config = rhinovla_quant_config(tmp_path)
    config["example"]["dtype"] = "fp16"
    del config["input"]["expert_w8a16"], config["input"]["full_w8a16"]
    Path(config["input"]["factory_config"]).write_text('{"expert_w8a16":true}')
    monkeypatch.setattr(policies, "_load_mapping", lambda *_: {"x0": torch.zeros(1)})
    with pytest.raises(ValueError, match="factory_config.expert_w8a16 conflicts"):
        policies.prepare_policy(config)


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
    infer, owner = policies.prepare_policy(config)
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
        policies.prepare_policy(config)
    assert calls == ["close"]
