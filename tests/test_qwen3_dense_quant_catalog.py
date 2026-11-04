"""Exact dense recipe/config identities; no model loading or hardware required."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILENAMES = {
    "w4a16": "w4a16_text_rtn_g32",
    "w4a16_lm_head": "w4a16_text_head_rtn_g32",
}
SHAPES = {
    "0.6": (1024, 3072, 28, 16), "1.7": (2048, 6144, 28, 16),
    "4": (2560, 9728, 36, 32), "8": (4096, 12288, 36, 32),
    "14": (5120, 17408, 40, 40), "32": (5120, 25600, 64, 64),
}


def load_module(name, relative, monkeypatch):
    if relative.startswith("examples/"):
        return importlib.import_module(relative[:-3].replace("/", "."))
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def common(monkeypatch):
    return load_module("_common", "examples/common/config.py", monkeypatch)


@pytest.mark.parametrize("size", SHAPES)
@pytest.mark.parametrize("head", [False, True])
def test_dense_w4_profiles_bind_exact_recipe_geometry_and_cold_defaults(common, size, head):
    recipe = "w4a16_lm_head" if head else "w4a16"
    path = ROOT / f"examples/configs/qwen3/text/{size.replace('.', '_')}b/{CONFIG_FILENAMES[recipe]}.toml"
    config = common.load_config(path)
    profile = common.planner_profiles()[config["example"]["profile_id"]]
    limits = profile["input_envelope"]["limits"]
    expected_variant = "w4a16-g32-lm-head" if head else "w4a16-g32"
    assert profile["id"] == f"qwen3-{size}b.{expected_variant}.text"
    assert profile["asset_alias"] == config["example"]["registry_alias"] == f"qwen3-{size}b"
    assert limits["on_install_quantization"] == recipe
    assert tuple(limits[key] for key in (
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads"
    )) == SHAPES[size]
    assert (limits["num_key_value_heads"], limits["head_dim"], limits["vocab_size"]) == (8, 128, 151936)
    assert (limits["num_cores"], limits["batch_size"], limits["group_size"]) == (8, 1, 32)
    assert limits["quantization"]["lm_head"] == (
        "symmetric RTN signed W4/G32 on install" if head else "FP16")
    assert profile["support"]["status"] == "CONTROLLED_EVALUATION"
    assert profile["hw_status"] == "IMPLEMENTED_UNVERIFIED_HW"
    assert "formal_certification" not in limits
    assert config["input"]["prefill_tokens"] == 128
    assert config["input"]["decode_steps"] == 7
    assert config["rpu_execution"]["prefill"]["linear_acc32"] is False
    assert (config["run"]["warmup"], config["run"]["runs"]) == (2, 5)
    request = {"model": {"num_cores": 8}, "prefill": {"linear_acc32": True}}
    common._validate_execution(request, profile=profile)
    for cores in (4, 6):
        request["model"]["num_cores"] = cores
        with pytest.raises((ValueError, common.ConfigError)):
            common._validate_execution(request, profile=profile)


def test_dense_32b_w8_uses_existing_converted_asset_without_inheriting_support(common, monkeypatch, tmp_path):
    registry = load_module("dense_catalog_registry", "python/rpu_backend/model_registry.py", monkeypatch)
    monkeypatch.setenv("RPU_MODEL_CACHE", str(tmp_path))
    assert registry.model_path("qwen3-32b") == tmp_path / "Qwen3-32B"
    assert registry.model_path("qwen3-32b-w8a16-lmhead-int8") == tmp_path / "Qwen3-32B-w8a16"
    config = common.load_config(ROOT / "examples/configs/qwen3/text/32b/w8a16_text_head.toml")
    profile = common.planner_profiles()[config["example"]["profile_id"]]
    limits = profile["input_envelope"]["limits"]
    assert limits["checkpoint_format"] == "qwen3_32b_w8a16"
    assert "checkpoint_quant_profile" not in limits
    assert "on_install_quantization" not in limits
    assert "max_kv_len" not in limits
    assert limits["kv_capacity"] == "requested-at-install; allocation must succeed"
    assert limits["max_chunk_size"] == 64
    assert limits["load_device"] == "rpu"
    assert profile["support"]["status"] == "CONTROLLED_EVALUATION"
    assert profile["hw_status"] == "IMPLEMENTED_UNVERIFIED_HW"
    assert "formal_certification" not in limits


@pytest.mark.parametrize("recipe", ["w4a16", "w4a16_lm_head"])
def test_wrong_dense_size_is_rejected_before_tokenizer_or_model(common, monkeypatch, recipe):
    runner = load_module("dense_wrong_size_runner", "examples/runners/qwen3/runner.py", monkeypatch)
    config = common.load_config(ROOT / f"examples/configs/qwen3/text/4b/{CONFIG_FILENAMES[recipe]}.toml")
    limits = dict(common.planner_profiles()[config["example"]["profile_id"]]["input_envelope"]["limits"])
    limits["num_hidden_layers"] = 37
    tokenizer, model_loader = Mock(), Mock()
    for name, values in {
        "transformers": {
            "AutoConfig": SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(**limits))),
            "AutoTokenizer": SimpleNamespace(from_pretrained=tokenizer)},
        "rpu_backend": {"RPUCache": object(), "RPUModelForCausalLM": SimpleNamespace(from_pretrained=model_loader)},
        "rpu_backend.model_registry": {"model_path": lambda _: "/cache/wrong-size"},
    }.items():
        module = ModuleType(name)
        vars(module).update(values)
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(common.ConfigError, match="exact Qwen3 geometry"):
        runner.prepare(config, SimpleNamespace(owner=None))
    tokenizer.assert_not_called()
    model_loader.assert_not_called()


def test_dense_32b_w8_metadata_rejects_wrong_head_before_tokenizer_or_weights(common, monkeypatch, tmp_path):
    from transformers import AutoConfig, Qwen3Config
    # Import the actual predicate before replacing just the runner's model API.
    from rpu_backend.quant.qwen3_profiles import is_qwen3_32b_w8a16_config

    config = common.load_config(ROOT / "examples/configs/qwen3/text/32b/w8a16_text_head.toml")
    source = Qwen3Config(
        architectures=["Qwen3ForCausalLM"], hidden_size=5120, intermediate_size=25600,
        num_hidden_layers=64, num_attention_heads=64, num_key_value_heads=8,
        head_dim=128, vocab_size=151936, tie_word_embeddings=False,
        max_position_embeddings=40960, rms_norm_eps=1e-6)
    source.quant_config = {
        "method": "w8a16", "mode": "per_channel_symmetric", "qaxis": 0,
        "skip_modules": [], "quantized_lm_head": True,
        "quantized_embed_tokens": False, "lm_head_untied": True,
        "embed_tokens_untied": False,
    }
    assert is_qwen3_32b_w8a16_config(source)
    source.to_json_file(tmp_path / "config.json")
    common.validate_checkpoint_profile(config, str(tmp_path))
    source.quant_config["quantized_lm_head"] = False
    source.to_json_file(tmp_path / "config.json")
    runner = load_module("dense_wrong_w8_runner", "examples/runners/qwen3/runner.py", monkeypatch)
    tokenizer, model_loader = Mock(), Mock()
    for name, values in {
        "transformers": {"AutoConfig": AutoConfig,
                         "AutoTokenizer": SimpleNamespace(from_pretrained=tokenizer)},
        "rpu_backend": {"RPUCache": object(), "RPUModelForCausalLM": SimpleNamespace(from_pretrained=model_loader)},
        "rpu_backend.model_registry": {"model_path": lambda _: tmp_path},
    }.items():
        module = ModuleType(name)
        vars(module).update(values)
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(common.ConfigError, match="checkpoint format=qwen3_32b_w8a16"):
        runner.prepare(config, SimpleNamespace(owner=None))
    tokenizer.assert_not_called()
    model_loader.assert_not_called()
