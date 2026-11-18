"""InternVLA-N1 System 2 RGB-D NavDP at a cold 4/6/8-core budget: host contracts only.

The example TOML, the composite resolver, the child owners' cold-only checks and
the native topology rows are checked here. The native methods are compiled from
production source with small doubles. None of this is a board run or numerical
evidence for the reduced-core kernels.
"""
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from examples.common import config as common
from rpu_backend.adapters.internvla_n1 import execution
from rpu_backend.adapters.navdp import runtime as navdp
from rpu_backend.adapters.wall_oss.vision import build_wall_oss_vision
from rpu_backend.runtime.topology import decoder_mlp_intermediate_size


ROOT = Path(__file__).resolve().parents[1]
TOML = ROOT / "examples/configs/internvla/navdp/w8a16_system2.toml"
PROFILE_ID = "internvla-n1-navdp.fp16.system2-rgbd-navdp"


def _cores(child):
    return child.get("model", {}).get("num_cores", 8)


def _block(source: str, signature: str) -> str:
    """Brace-matched body that follows ``signature``."""
    begin = source.index("{", source.index(signature))
    depth, position = 1, begin + 1
    while depth:
        depth += (source[position] == "{") - (source[position] == "}")
        position += 1
    return source[begin:position]


# --------------------------------------------------------------------------- examples


def test_example_toml_admits_every_budget():
    config = common.load_config(TOML)
    assert config["example"]["profile_id"] == PROFILE_ID
    assert common.core_support(config) == {4: None, 6: None, 8: None}


def test_core_choice_requires_the_registered_profile():
    profile = common.example_profiles()[PROFILE_ID]
    common._validate_execution({"model": {"num_cores": 6}}, profile=profile)
    other = {**profile, "id": "internvla-n1-navdp.fp16.other"}
    with pytest.raises(common.ConfigError, match="registered System 2"):
        common._validate_execution({"model": {"num_cores": 6}}, profile=other)
    with pytest.raises(ValueError, match="4, 6 or 8"):
        common._validate_execution({"model": {"num_cores": 5}}, profile=profile)


# --------------------------------------------------------------------------- resolver


@pytest.mark.parametrize("budget", (4, 6, 8))
def test_resolver_derives_each_child_budget(budget):
    request = {} if budget == 8 else {"model": {"num_cores": budget}}
    root, children = execution.resolve_internvla_execution(request, entry_point="test")
    assert execution.internvla_core_count(root) == budget
    four = {execution.VISION_COMPONENT, execution.FORMER_COMPONENT,
            execution.NAVDP_COMPONENT}
    for component, child in children.items():
        if budget == 8:
            assert "model" not in child, component
        elif component == execution.NEXTDIT_COMPONENT:
            # Left unchanged for its builder, which refuses a reduced budget.
            assert "model" not in child
            with pytest.raises(ValueError, match="eight-core owner"):
                execution.internvla_component_cores(budget, component)
        else:
            assert _cores(child) == (4 if component in four else budget), component


def test_children_cannot_choose_their_own_budget():
    with pytest.raises(ValueError, match="unknown stage"):
        execution.resolve_internvla_execution(
            {"components": {execution.BACKBONE_COMPONENT: {"model": {"num_cores": 4}}}},
            entry_point="test")
    with pytest.raises(ValueError, match="4, 6 or 8"):
        execution.resolve_internvla_execution({"model": {"num_cores": 5}}, entry_point="test")


@pytest.mark.parametrize(("component", "cores", "admitted"), [
    (execution.BACKBONE_COMPONENT, 6, True),
    (execution.RGB_COMPONENT, 6, True),
    (execution.VISION_COMPONENT, 4, True),
    (execution.VISION_COMPONENT, 6, False),
    (execution.FORMER_COMPONENT, 6, False),
    (execution.NAVDP_COMPONENT, 6, False),
    (execution.NEXTDIT_COMPONENT, 4, False),
])
def test_component_local_budget_matches_its_owner(component, cores, admitted):
    request = {"model": {"num_cores": cores}}
    if admitted:
        child = execution.component_execution(request, component, entry_point="test")
        assert _cores(child) == cores
    else:
        with pytest.raises(ValueError, match=f"no {cores}-core owner"):
            execution.component_execution(request, component, entry_point="test")


def test_controller_budget_is_cold_only():
    root, _ = execution.resolve_internvla_execution({}, entry_point="test")
    controller = execution.InternVLAExecutionController.__new__(
        execution.InternVLAExecutionController)
    controller.owner = SimpleNamespace(_rpu_execution=root)
    controller.children = {execution.BACKBONE_COMPONENT: object()}
    controller.validate({"prefill": {"chunk_size": "auto"}})
    with pytest.raises(ValueError, match="cold-only"):
        controller.validate({"model": {"num_cores": 6}})


def test_navdp_owner_has_four_and_eight_lanes_only():
    assert "model" not in navdp._resolve_navdp_execution({}, entry_point="test")
    four = navdp._resolve_navdp_execution({"model": {"num_cores": 4}}, entry_point="test")
    assert navdp.navdp_core_count(four) == 4
    with pytest.raises(ValueError, match="no 6-core owner"):
        navdp._resolve_navdp_execution({"model": {"num_cores": 6}}, entry_point="test")
    owner = navdp.NavdpRuntime.__new__(navdp.NavdpRuntime)
    owner._num_cores = 4
    assert navdp.navdp_core_count(owner._validate_execution({"model": {"num_cores": 4}})) == 4
    with pytest.raises(ValueError, match="cold-only"):
        owner._validate_execution({})


@pytest.mark.parametrize("cores", (6, 0, True, 4.0))
def test_wall_vision_builder_refuses_other_budgets_before_loading(cores):
    with pytest.raises(ValueError, match="must be 4 or 8"):
        build_wall_oss_vision(num_cores=cores)


def test_six_lane_backbone_mlp_uses_the_padded_width():
    assert decoder_mlp_intermediate_size(18944, 6) == 19008
    assert 19008 % (6 * 16) == 0
    assert decoder_mlp_intermediate_size(18944, 4) == decoder_mlp_intermediate_size(18944, 8) == 18944


# --------------------------------------------------------------------------- native


def test_navdp_native_owner_has_no_fixed_eight_core_launch():
    source = (ROOT / "src/fused/rpu_navdp_model.cpp").read_text()
    assert not re.search(r"\bNUM_CORES\b", source)


def test_native_topology_rows_admit_only_the_internvla_geometries(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required to compile the production topology methods")
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    vision = (ROOT / "src/fused/rpu_qwen25vl_vision_model.cpp").read_text()
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include "execution_topology.h"
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
using v3::DecoderExecutionTopology;
using v3::resolve_decoder_execution_topology;
struct FusedModelBase {
    struct Impl { int num_cores_ = 8; } impl;
    Impl* pimpl_ = &impl;
    virtual ~FusedModelBase() = default;
    int num_cores() const { return impl.num_cores_; }
    virtual DecoderExecutionTopology resolve_model_execution_topology(
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size) const __BASE_RESOLVE__
};
struct Qwen25Vision : FusedModelBase {
    explicit Qwen25Vision(int cores) { impl.num_cores_ = cores; }
    DecoderExecutionTopology resolve_model_execution_topology(
            int64_t nq, int64_t nkv, int64_t hd, int64_t h,
            int64_t intermediate) const override __RESOLVE__
};
bool same(const DecoderExecutionTopology& t, int cores, int attention, int mlp) {
    return t.num_cores == cores && t.attention_tp == attention && t.mlp_tp == mlp;
}
template <class F> void reject(F body) {
    bool rejected = false;
    try { body(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
int main() {
    // System 2 text: attention stays on four lanes, the MLP follows the budget.
    assert(v3::internvla_text_profile_num_layers(28, 4, 128, 3584, 18944) == 28);
    for (int cores : {4, 6, 8}) {
        assert(same(resolve_decoder_execution_topology(cores, 28, 4, 128, 3584, 18944),
                    cores, 4, cores));
    }
    assert(v3::decoder_mlp_intermediate_size(18944, 6) == 19008);
    assert(v3::decoder_mlp_intermediate_size(18944, 4) == 18944);
    for (int cores : {4, 6}) {
        reject([cores] { resolve_decoder_execution_topology(cores, 28, 4, 128, 3584, 19008); });
        reject([cores] { resolve_decoder_execution_topology(cores, 28, 4, 128, 3584, 18432); });
        reject([cores] { resolve_decoder_execution_topology(cores, 28, 2, 128, 3584, 18944); });
        reject([cores] { resolve_decoder_execution_topology(cores, 28, 4, 128, 3072, 18944); });
    }
    reject([] { resolve_decoder_execution_topology(5, 28, 4, 128, 3584, 18944); });
    // System 2 vision: four lanes for the exact padded ViT, default eight kept.
    assert(same(Qwen25Vision(8).resolve_model_execution_topology(16, 16, 80, 1280, 3456),
                8, 8, 8));
    assert(same(Qwen25Vision(4).resolve_model_execution_topology(16, 16, 80, 1280, 3456),
                4, 4, 4));
    reject([] { Qwen25Vision(4).resolve_model_execution_topology(16, 16, 80, 1280, 3420); });
    reject([] { Qwen25Vision(4).resolve_model_execution_topology(16, 16, 64, 1024, 3456); });
    reject([] { Qwen25Vision(4).resolve_model_execution_topology(12, 12, 80, 960, 3456); });
    reject([] { Qwen25Vision(6).resolve_model_execution_topology(16, 16, 80, 1280, 3456); });
}
'''
    for marker, value in (
        ("__BASE_RESOLVE__", _block(base, "DecoderExecutionTopology FusedModelBase::resolve_model_execution_topology(")),
        ("__RESOLVE__", _block(vision, "DecoderExecutionTopology resolve_model_execution_topology(")),
    ):
        cpp = cpp.replace(marker, value)
    path, binary = tmp_path / "internvla_cores.cpp", tmp_path / "internvla_cores"
    path.write_text(cpp)
    compiled = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                               "-I", str(ROOT / "src/core"), str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr
