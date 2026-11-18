"""G0.5 camera templates at a cold 4/6/8-core budget: host contracts only.

The example TOMLs, the composite resolver, the cold-only session check, the
installers' refusals and the Qwen3.5 execution-topology rows are checked here
with small doubles. None of this is a board run or numerical evidence for the
reduced-core kernels.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import shutil
import subprocess

import pytest

from examples.common import config as common
from examples.runners.g05 import optimized as runner
from rpu_backend.adapters.g05 import action as g05_action
from rpu_backend.adapters.g05 import quant as g05_quant
from rpu_backend.adapters.g05 import runtime as g05
from rpu_backend.adapters.qwen3_5 import cores


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = sorted((ROOT / "examples/configs/g05").glob("*.toml"))
LEGACY = ROOT / "examples/configs/g05/legacy/fp16.toml"


def _cores(child):
    return child.get("model", {}).get("num_cores", 8)


def _with_budget(config, cores):
    config.setdefault("rpu_execution", {})["model"] = {"num_cores": cores}
    return config


# --------------------------------------------------------------------------- examples


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda path: path.name)
def test_every_camera_template_admits_each_budget(path):
    assert common.core_support(common.load_config(path)) == {4: None, 6: None, 8: None}


def test_legacy_profile_stays_eight_core_with_a_reason():
    support = common.core_support(common.load_config(LEGACY))
    assert support[8] is None and support[4] == support[6]
    assert support[4].startswith("旧版档案")
    with pytest.raises(common.ConfigError, match="num_cores = 6 is not supported here: 旧版档案"):
        common.load_config(LEGACY, {"rpu_execution": {"model": {"num_cores": 6}}})


@pytest.mark.parametrize("cores", (4, 6))
def test_reduced_template_refuses_eight_core_options(cores):
    path = ROOT / "examples/configs/g05/w8a16.toml"
    config = _with_budget(common.load_config(path), cores)
    config["g05"]["vision_camera_batch"] = 3
    with pytest.raises(ValueError, match="vision_camera_batch = 3 .*eight-core only"):
        runner.validate_example_config(config)
    config = _with_budget(common.load_config(path), cores)
    config["g05"]["math_precision"] = "high"
    with pytest.raises(ValueError, match="'high' is eight-core only"):
        runner.validate_example_config(config)
    config = common.load_config(path)
    config["rpu_execution"]["model"] = {"num_cores": 5}
    with pytest.raises(ValueError, match="num_cores = 4, 6 or 8"):
        runner.validate_example_config(config)


@pytest.mark.parametrize("cores", (4, 6, 8))
def test_reduced_budget_skips_the_eight_core_catalog_rows(cores):
    config = common.load_config(ROOT / "examples/configs/g05/w8a16_action_nvfp4.toml")
    if cores != 8:
        _with_budget(config, cores)
    resolved, selection = runner.resolve_row_config(config)
    prefill = resolved["rpu_execution"].get("components", {}).get(
        "language_model", {}).get("prefill", {})
    if cores == 8:
        # The 3-camera T32 row measured on eight cores selects B3 and 256 rows.
        assert selection["matched"] is True
        assert resolved["g05"]["vision_camera_batch"] == 3
        assert prefill["chunk_size"] == 256
        return
    assert selection["matched"] is False
    assert "eight-core measurements" in selection["reason"]
    assert resolved["g05"]["vision_camera_batch"] == 1
    assert prefill.get("chunk_size", "auto") == "auto"


# --------------------------------------------------------------------------- resolver


@pytest.mark.parametrize("budget", (4, 6, 8))
def test_resolver_derives_each_child_budget(budget):
    request = {} if budget == 8 else {"model": {"num_cores": budget}}
    root, components = g05._resolve_g05_execution(request, max_seq_len=2048, entry_point="test")
    assert g05.g05_core_budget(root) == budget
    vision, text, action = (components[name] for name in (
        g05._G05_VISION_COMPONENT, g05._G05_TEXT_COMPONENT, g05._G05_ACTION_COMPONENT))
    if budget == 8:
        assert all("model" not in child for child in components.values())
        return
    assert _cores(text) == budget
    assert _cores(vision) == _cores(action) == 4
    # Stage defaults survive the core scope.
    assert action["action"]["chunk_size"] == "auto"
    assert text["prefill"]["padding_budget"] == 64


@pytest.mark.parametrize(("request_", "match"), [
    ({"model": {"num_cores": 5}}, "4, 6 or 8"),
    ({"components": {"action_expert": {"model": {"num_cores": 4}}}}, "model"),
])
def test_resolver_refuses_odd_and_child_budgets(request_, match):
    with pytest.raises(ValueError, match=match):
        g05._resolve_g05_execution(request_, max_seq_len=2048, entry_point="test")


def test_reconfigure_keeps_the_cold_budget():
    root, _components = g05._resolve_g05_execution(
        {"model": {"num_cores": 6}}, max_seq_len=2048, entry_point="test")
    policy = NS(_rpu_execution=root,
                model=NS(vlm=NS(_rpu_g05_max_seq_len=2048), vision_tower=NS()))
    for change in ({}, {"model": {"num_cores": 4}}):
        with pytest.raises(ValueError, match="cold-only"):
            g05._validate_g05_execution_reconfigure(policy, change)


# --------------------------------------------------------------------------- installers


def test_policy_refuses_reduced_generic_profiles_before_mutation():
    policy = NS(training=False, continuous_action=True, discrete_action=False, model=NS())
    with pytest.raises(ValueError, match="num_cores=6 requires the exact vla_rtc K1 policy"):
        g05.patch_g05_policy_for_rpu(policy, rpu_execution={"model": {"num_cores": 6}})
    assert "_rpu_g05_policy_install_started" not in vars(policy)


def test_action_expert_runs_on_four_or_eight_lanes(monkeypatch):
    with pytest.raises(ValueError, match="four or eight cores"):
        g05_action.patch_g05_action_expert_for_rpu(
            NS(), _execution_config={"model": {"num_cores": 6}})
    monkeypatch.setattr(g05_quant, "require_g05_high_math_native", lambda **kwargs: None)
    owner = NS()
    with pytest.raises(ValueError, match="HIGH action math is eight-core only"):
        g05_action.patch_g05_action_expert_for_rpu(
            owner, _execution_config={"model": {"num_cores": 4}},
            math_precision="high", _g05_exact_k1=True)
    assert not vars(owner)
    with pytest.raises(ValueError, match="four or eight lanes"):
        g05_quant.prepare_g05_action_nvfp4(NS(), lanes=6)


def test_camera_grouping_is_eight_core_only():
    vision = NS(_rpu_g05_vision_forward_impl=lambda *args, **kwargs: None,
                _rpu_vision_num_cores=4)
    policy = NS(model=NS(vision_tower=vision), _rpu_g05_fm_only_customer_release=True,
                _execution_session=NS(require_cold=lambda: None))
    with pytest.raises(ValueError, match="eight-core only"):
        g05.configure_g05_vision_camera_batch(policy, 3)
    g05.configure_g05_vision_camera_batch(policy, 1)


def _g05_config(*, action):
    from test_g05_adapter_contracts import _config
    config = _config()
    if action:
        vars(config).update(hidden_size=1024, intermediate_size=4096, adaptive_mode="adaLN",
                            time_hidden_size=1024, input_type="linear", input_dim=27,
                            output_type="linear", output_dim=27,
                            layer_types=["full_attention"] * 24)
    return config


@pytest.mark.parametrize("w8", (False, True))
def test_core_profile_admits_the_exact_g05_owners(w8):
    text, action = _g05_config(action=False), _g05_config(action=True)
    if w8:
        text.quant_config = g05_quant.g05_w8_quant_config()
        action.quant_config = g05_quant.g05_w8_quant_config(action=True)
    for budget, mlp in ((4, 4), (6, 6)):
        topology = cores.validate_core_profile(text, {"model": {"num_cores": budget}})
        assert (topology.attn_tp, topology.mlp_tp) == (4, mlp)
        assert cores.text_mlp_intermediate_size(text, topology) == 6144
    topology = cores.validate_core_profile(action, {"model": {"num_cores": 4}})
    assert (topology.attn_tp, topology.mlp_tp) == (4, 4)
    with pytest.raises(ValueError, match="action expert runs on four cores"):
        cores.validate_core_profile(action, {"model": {"num_cores": 6}})
    text.quant_config = g05_quant.g05_w8_quant_config(action=True)
    with pytest.raises(ValueError, match="FP16 or its exact per-channel W8 scope"):
        cores.validate_core_profile(text, {"model": {"num_cores": 6}})


# --------------------------------------------------------------------------- native rows


def test_native_topology_admits_the_action_row_on_four_lanes_only(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    cpp = r'''
#include "fused/qwen3_5_execution_topology.h"
#include <cassert>
using namespace v3;
template <class F> void reject(F f) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    assert(false);
}
bool same(DecoderExecutionTopology t, int cores, int attention, int mlp) {
    return t.num_cores == cores && t.attention_tp == attention && t.mlp_tp == mlp;
}
int main() {
    assert(g05_action_profile_num_layers(8, 4, 256, 1024, 4096) == 24);
    assert(g05_action_profile_num_layers(8, 2, 256, 1024, 4096) == 0);
    // The headless G05 text owner is the existing Qwen3.5-2B row.
    for (int cores : {4, 6})
        assert(same(resolve_qwen35_reduced_execution_topology(cores, 8, 4, 256, 2048, 6144),
                    cores, 4, cores));
    assert(same(resolve_qwen35_reduced_execution_topology(4, 8, 4, 256, 1024, 4096), 4, 4, 4));
    reject([] { resolve_qwen35_reduced_execution_topology(6, 8, 4, 256, 1024, 4096); });
    reject([] { resolve_qwen35_reduced_execution_topology(4, 8, 2, 256, 1024, 4096); });
    reject([] { resolve_qwen35_reduced_execution_topology(4, 8, 4, 256, 1024, 2048); });
}
'''
    path, binary = tmp_path / "g05_cores.cpp", tmp_path / "g05_cores"
    path.write_text(cpp)
    compiled = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                               "-I", str(ROOT / "src"), str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr
