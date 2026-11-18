"""Public QuantTrio examples admit only the selected original AWQ checkpoint."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from examples.common import config as common

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/qwen35_quanttrio_awq"


def config(size, prefill=64):
    value = common.load_config(
        ROOT / f"examples/configs/qwen3_5/text/{size}b/w4a16_mlp_awq_g128.toml")
    value["input"].update(prefill_tokens=prefill, decode_steps=4096, stop_on_eos=False)
    common.validate_generation_options(value)
    return value


@pytest.mark.parametrize("size", [2, 4, 9])
def test_public_awq_configs_bind_original_asset_without_mutation(size, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "awq_public_registry", ROOT / "python/rpu_backend/model_registry.py")
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)
    monkeypatch.setenv("RPU_MODEL_CACHE", str(tmp_path))
    short, long = config(size), config(size, 4096)
    alias = f"qwen3_5-{size}b-quanttrio-awq-g128"
    assert registry.model_path(alias) == tmp_path / f"Qwen3.5-{size}B-AWQ"
    checkpoint = tmp_path / f"Qwen3.5-{size}B-AWQ"
    checkpoint.mkdir()
    original = (FIXTURES / f"{size}b.json").read_bytes()
    (checkpoint / "config.json").write_bytes(original)
    base = common.load_config(
        ROOT / f"examples/configs/qwen3_5/text/{size}b/w4a16_mlp_awq_g128.toml")
    common.validate_checkpoint_profile(base, str(checkpoint))
    assert base["example"]["registry_alias"] == alias
    assert base["input"]["prompt"] and base["input"]["decode_steps"] == 4
    for value, length in ((short, 64), (long, 4096)):
        common.validate_checkpoint_profile(value, str(checkpoint))
        assert value["example"]["registry_alias"] == alias
        assert value["rpu_execution"]["prefill"]["linear_acc32"] is False
        assert value["rpu_execution"]["model"]["num_cores"] == 8
        profile = common.planner_profiles()[value["example"]["profile_id"]]
        for cores in (4, 6, 8):
            value["rpu_execution"]["model"]["num_cores"] = cores
            common._validate_execution(value["rpu_execution"], profile=profile)
        assert value["input"]["prefill_tokens"] == length
        assert value["input"]["decode_steps"] == 4096
        assert value["input"]["stop_on_eos"] is False
        assert value["run"]["warmup"] == 2 and value["run"]["runs"] == 5
    assert short["input"]["prompt"] == long["input"]["prompt"]
    assert (checkpoint / "config.json").read_bytes() == original
    assert list(checkpoint.iterdir()) == [checkpoint / "config.json"]


@pytest.mark.parametrize("change", ["float", "g32", "symmetric", "no_first_layer_exception", "other_size"])
def test_public_awq_profile_rejects_wrong_checkpoint_before_loading(tmp_path, change):
    value = json.loads((FIXTURES / "2b.json").read_text())
    if change == "float":
        del value["quantization_config"]
    elif change == "g32":
        value["quantization_config"]["group_size"] = 32
    elif change == "symmetric":
        value["quantization_config"]["zero_point"] = False
    elif change == "no_first_layer_exception":
        value["quantization_config"]["modules_to_not_convert"].remove("model.layers.0.")
    else:
        value = json.loads((FIXTURES / "4b.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(value))
    with pytest.raises(common.ConfigError, match="[Aa][Ww][Qq]"):
        common.validate_checkpoint_profile(config(2), str(tmp_path))
