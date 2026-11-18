"""Official dense27B configs bind exact precision/core identity without hardware."""
import importlib
import json
from pathlib import Path

import pytest
from _support_status import expected_support_status

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "examples/configs/qwen3_5"
CASES = [(component, precision, cores) for component in ("text", "vl")
         for precision in ("fp16", "w4a16", "w8a16") for cores in (4, 6, 8)]


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("examples.common.config")


def _path(component, precision, cores):
    stem = {"fp16": "fp16", "w8a16": "w8a16_text",
            "w4a16": "w4a16_text_rtn_g32"}[precision]
    return FAMILY / f"{component}/27b/{stem}.toml"


@pytest.fixture
def metadata():
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5", "tie_word_embeddings": False,
        "image_token_id": 248056,
        "text_config": {
            "model_type": "qwen3_5_text", "hidden_size": 5120,
            "intermediate_size": 17408, "num_hidden_layers": 64,
            "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "vocab_size": 248320,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "hidden_act": "silu", "attention_bias": False, "attn_output_gate": True,
            "rms_norm_eps": 1e-6, "use_cache": True, "full_attention_interval": 4,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                "mrope_section": [11, 11, 10]},
        },
        "vision_config": {"depth": 27, "num_heads": 16, "hidden_size": 1152,
            "intermediate_size": 4304, "out_hidden_size": 5120,
            "in_channels": 3, "patch_size": 16, "temporal_patch_size": 2,
            "spatial_merge_size": 2, "num_position_embeddings": 2304,
            "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": []},
    }


@pytest.mark.parametrize("component,precision,cores", CASES)
def test_official_27b_profiles_match_exact_config_topology(common, component, precision, cores, tmp_path):
    path = _path(component, precision, cores)
    selected = tmp_path / "selected.toml"
    selected.write_text(path.read_text().replace("num_cores = 8", f"num_cores = {cores}"))
    config = common.load_config(selected)
    profile = common.planner_profiles()[config["example"]["profile_id"]]
    expected_alias = f"qwen3_5-27b-{precision}"
    assert profile["asset_alias"] == expected_alias
    assert profile["input_envelope"]["limits"]["allowed_num_cores"] == [4, 6, 8]
    assert "num_cores" not in profile["input_envelope"]["limits"]
    assert profile["hw_status"] == "IMPLEMENTED_UNVERIFIED_HW"
    expected_status = "IMPLEMENTED_UNVERIFIED_HW" if precision == "fp16" else "CONTROLLED_EVALUATION"
    assert profile["support"]["status"] == expected_support_status(profile, expected_status)
    assert config["rpu_execution"]["model"]["num_cores"] == cores
    if precision != "fp16":
        assert config["rpu_execution"]["prefill"]["linear_acc32"] is False
    assert config["run"]["runs"] == 5
    assert config["run"]["warmup"] == 2
    bad = tmp_path / "wrong_core.toml"
    bad.write_text(path.read_text().replace("num_cores = 8", "num_cores = 5"))
    with pytest.raises(common.ConfigError, match=r"model\.num_cores"):
        common.load_config(bad)
    for gate in config["example"]["opt_in"]:
        bad.write_text(path.read_text().replace(f'{gate} = "1"\n', ""))
        with pytest.raises(common.ConfigError, match="exact opt_in"):
            common.load_config(bad)


def test_matrix_is_a_pure_configuration_index(common):
    matrix = json.loads((FAMILY / "27b_matrix.json").read_text())
    cases = matrix["configs"]
    assert matrix["schema_version"] == 2
    assert len(cases) == len({row["id"] for row in cases}) == len({row["config"] for row in cases})
    assert cases and all("_tp4" not in row["config"] and "_tp6" not in row["config"] for row in cases)
    for row in cases:
        config = common.load_config(ROOT / row["config"])
        assert config["example"]["profile_id"] == row["profile_id"]
        assert row["component"] in ("text", "vl")
        assert row["precision"] in ("fp16", "w4a16", "w8a16")
        assert row["allowed_num_cores"] == [4, 6, 8]
        assert config["rpu_execution"]["model"]["num_cores"] in row["allowed_num_cores"]
        assert all(key not in row for key in ("reference_ids", "measured_dataset", "reference_prefill_tokens", "cpu_threads"))


@pytest.mark.parametrize("component,bits,version", [(component, bits, version)
    for component in ("text", "vl") for bits in (4, 8) for version in (1, 2)])
def test_checkpoint_preflight_accepts_equivalent_uniform_quant_semantics(common, metadata, component, bits, version, tmp_path):
    metadata["text_config"]["layer_types"] *= 16
    method = "w8a16" if bits == 8 else "w4a16_pgrp"
    spec = {"method": method, **({"group_size": 32} if bits == 4 else {})}
    metadata["quant_config"] = ({"format_version": 2, "default": spec}
                                if version == 2 else {"format_version": 1, **spec})
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    precision = "w8a16" if bits == 8 else "w4a16"
    common.validate_checkpoint_profile(common.load_config(_path(component, precision, 8)), str(tmp_path))


@pytest.mark.parametrize("mutation", ["other_size", "legacy", "wrong_precision", "missing_quant", "wrong_architecture"])
def test_checkpoint_preflight_rejects_relabelling(common, metadata, mutation, tmp_path):
    metadata["text_config"]["layer_types"] *= 16
    scope = common._backend_leaf("adapters/qwen3_5/quant_scope.py")
    metadata["quant_config"] = {"method": "w8a16"}
    if mutation == "other_size":
        metadata["text_config"]["hidden_size"] = 4096
    elif mutation == "legacy":
        metadata["quant_config"] = scope.legacy_w8a16_quant_config()
    elif mutation == "wrong_precision":
        metadata["quant_config"] = {"method": "w4a16_pgrp", "group_size": 32}
    elif mutation == "missing_quant":
        del metadata["quant_config"]
    else:
        metadata["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    for component in ("text", "vl"):
        with pytest.raises(common.ConfigError, match="rejects checkpoint"):
            common.validate_checkpoint_profile(
                common.load_config(_path(component, "w8a16", 8)), str(tmp_path))


def test_official_assets_keep_historical_profile_out_of_active_catalog(common):
    registry = common._backend_leaf("model_registry.py")
    assert "qwen3_8-27b-w8a16" not in registry.MODELS
    assert registry.ARCHIVED_MODELS["qwen3_8-27b-w8a16"] == "Qwen3.8-27B-w8a16"
    assert registry.archived_model_path("qwen3_8-27b-w8a16").name == "Qwen3.8-27B-w8a16"
    assert "qwen3_5-27b-w8a16" in registry.MODELS
    assert "qwen3_5-27b-w4a16" in registry.MODELS
    assert registry.MODELS["qwen3_5-27b-fp16"] == "Qwen3.5-27B-FP16"
    assert registry.MODELS["qwen3_5-27b"] == "Qwen3.5-27B"
    profiles = common.planner_profiles()
    for component in ("text", "vl"):
        config = common.load_config(_path(component, "fp16", 8))
        assert profiles[config["example"]["profile_id"]]["asset_alias"] == config["example"]["registry_alias"]
    assert not (FAMILY / "text/27b/w8a16_legacy_text.toml").exists()


RECIPES = {
    "w4_mlp_only": ("w4a16_mlp_rtn_g32", "w4-mlp-only-g32"),
    "w4_mlp_w8_attn_gdn": ("w4a16_mlp_rtn_g32_attn_gdn_w8a16", "w4-mlp-w8-attn-gdn-g32"),
    "activation_awq_g128_v1": ("w4a16_mlp_activation_awq_g128", "activation-awq-g128"),
}


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_official_27b_recipe_configs_bind_exact_scope(common, metadata, recipe, cores, tmp_path):
    scope = common._backend_leaf("adapters/qwen3_5/quant_scope.py")
    metadata["text_config"]["layer_types"] *= 16
    if recipe == "activation_awq_g128_v1":
        policy = scope._quanttrio_policy(scope.is_full_from_layer_types(metadata["text_config"]["layer_types"]))
        policy["profile"] = recipe
    else:
        policy = {"format_version": 2, "architecture": "qwen3_5", "profile": recipe,
                  "default": {"method": "fp16" if recipe == "w4_mlp_only" else "w8a16"},
                  "projection_overrides": {f"mlp.{role}": {"method": "w4a16_pgrp", "group_size": 32}
                                           for role in ("gate_proj", "up_proj", "down_proj")}}
    metadata["quant_config"] = policy
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    stem, public_recipe = RECIPES[recipe]
    for prefill in (None, 64, 4096):
        config = common.load_config(FAMILY / f"text/27b/{stem}.toml")
        if prefill is not None:
            config["input"].update(prefill_tokens=prefill, decode_steps=4096, stop_on_eos=False)
            common.validate_generation_options(config)
        profile = common.planner_profiles()[config["example"]["profile_id"]]
        assert profile["support"]["status"] == expected_support_status(profile, "IMPLEMENTED_UNVERIFIED_HW")
        assert config["example"]["registry_alias"] == f"qwen3_5-27b-{public_recipe}"
        config["rpu_execution"]["model"]["num_cores"] = cores
        common._validate_execution(config["rpu_execution"], profile=profile)
        common.validate_checkpoint_profile(config, str(tmp_path))
        metadata["text_config"]["hidden_size"] = 2048
        (tmp_path / "config.json").write_text(json.dumps(metadata))
        with pytest.raises(common.ConfigError):
            common.validate_checkpoint_profile(config, str(tmp_path))
        metadata["text_config"]["hidden_size"] = 5120
        (tmp_path / "config.json").write_text(json.dumps(metadata))


@pytest.mark.parametrize("component", ["text", "vl"])
@pytest.mark.parametrize("owner,key,value", [
    pytest.param("text_config", "hidden_size", 4096, id="other-size"),
    pytest.param("vision_config", "out_hidden_size", 4096, id="vision-geometry"),
    pytest.param(None, "architectures", ["Qwen3_5MoeForConditionalGeneration"],
                 id="wrong-architecture"),
    pytest.param(None, "quant_config", {"method": "w8a16"}, id="quantized"),
    pytest.param("text_config", "quant_config", {"method": "w8a16"},
                 id="nested-quantized"),
    pytest.param(None, "quantization_config", {"quant_method": "awq"},
                 id="foreign-quantization"),
])
def test_official_27b_fp16_checkpoint_preflight_is_bound(
        common, metadata, component, owner, key, value, tmp_path):
    metadata["text_config"]["layer_types"] *= 16
    config = common.load_config(_path(component, "fp16", 8))
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    common.validate_checkpoint_profile(config, str(tmp_path))
    (metadata if owner is None else metadata[owner])[key] = value
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    with pytest.raises(common.ConfigError, match="rejects checkpoint"):
        common.validate_checkpoint_profile(config, str(tmp_path))
