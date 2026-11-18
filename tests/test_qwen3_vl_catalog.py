"""Public VL configs bind the composite core budget before loading assets."""
import copy
from pathlib import Path

import pytest

from examples.common import config as common


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "examples/configs/qwen3_vl").rglob("*.toml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: str(path.relative_to(ROOT)))
def test_vl_config_accepts_four_six_and_eight_core_budgets(path):
    config = common.load_config(path)
    profile = common.planner_profiles()[config["example"]["profile_id"]]
    for cores in (4, 6, 8):
        execution = copy.deepcopy(config["rpu_execution"])
        execution["model"]["num_cores"] = cores
        common._validate_execution(execution, profile=profile)
    for cores in (True, 5):
        with pytest.raises((ValueError, common.ConfigError)):
            common._validate_execution({"model": {"num_cores": cores}}, profile=profile)


@pytest.mark.parametrize("fault", ("unadmitted", "fixed", "precision", "runner", "component"))
def test_vl_core_budget_requires_the_registered_composite_profile(fault):
    config = common.load_config(CONFIGS[0])
    profile = copy.deepcopy(common.planner_profiles()[config["example"]["profile_id"]])
    limits = profile["input_envelope"]["limits"]
    if fault == "unadmitted":
        limits.pop("allowed_num_cores")
    elif fault == "fixed":
        limits["num_cores"] = 8
    elif fault == "precision":
        profile["precision"] = "hybrid-w8-w4-top2"
    elif fault == "runner":
        profile["example"]["runner_target"] = "qwen3"
    else:
        profile["component"] = "audio"
    with pytest.raises(common.ConfigError, match="registered dense precision profile"):
        common._validate_execution({"model": {"num_cores": 6}}, profile=profile)
