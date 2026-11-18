"""Quantized text plus FP16 Vision keeps both controlled admission gates."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = [(size, bits) for size in ("0_8b", "2b", "4b") for bits in (4, 8)] + [("9b", 8)]


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    return importlib.import_module("examples.common.config")


@pytest.mark.parametrize("size,bits", CASES)
def test_quant_vision_example_binds_both_existing_contracts(common, size, bits, tmp_path):
    stem = "w8a16_text" if bits == 8 else "w4a16_text_rtn_g32"
    path = ROOT / f"examples/configs/qwen3_5/vl/{size}/{stem}.toml"
    config = common.load_config(path)
    profiles = common.planner_profiles()
    profile = profiles[config["example"]["profile_id"]]
    model_size = size.replace("_", ".")
    vision = profiles[f"qwen3_5-{model_size}.fp16.vision-controlled"]
    text = profiles[f"qwen3_5-{model_size}.w{bits}a16.text-controlled"]
    assert profile["asset_alias"] == text["asset_alias"]
    assert profile["precision"] == config["example"]["dtype"] == f"w{bits}a16"
    assert profile["authority"] == profile["owner"] == vision["authority"]
    assert profile["controlled_gates"] == text["controlled_gates"] + vision["controlled_gates"]
    assert profile["support"]["status"] == "SUPPORTED"
    assert profile["hw_status"] == "IMPLEMENTED_UNVERIFIED_HW"
    assert profile["input_envelope"]["kind"] == "BOUNDED"
    assert profile["input_envelope"]["limits"] == {"allowed_num_cores": [4, 6, 8]}
    assert config["rpu_execution"]["model"]["num_cores"] == 8
    # Core choice shares one recipe; its controlled gates remain mandatory.
    for cores in (4, 6, 8):
        bad = tmp_path / f"reduced_cores_{cores}.toml"
        bad.write_text(path.read_text().replace("num_cores = 8", f"num_cores = {cores}"))
        selected = common.load_config(bad)
        assert selected["example"]["profile_id"] == config["example"]["profile_id"]
        assert selected["rpu_execution"]["model"]["num_cores"] == cores
    # No opt-in is required, and a stale one is rejected by exact opt-in equality.
    assert not config["example"].get("opt_in")
    bad = tmp_path / "stale_opt_in.toml"
    bad.write_text(path.read_text().replace(
        "[model.opt_in]\n", '[model.opt_in]\nQWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED = "1"\n'))
    with pytest.raises(common.ConfigError, match="exact opt_in"):
        common.load_config(bad)
    bad = tmp_path / "mismatched_precision.toml"
    bad.write_text(path.read_text().replace(f'precision = "w{bits}a16"', 'precision = "fp16"'))
    with pytest.raises(common.ConfigError, match="requires dtype"):
        common.load_config(bad)
