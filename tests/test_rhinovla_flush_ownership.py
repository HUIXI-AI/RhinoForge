"""Device handoffs preserve tensor ownership without touching CPU cache data."""
import copy
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

from test_fmb_standalone_layout_rebuild import _definition
from test_rhinovla_expert_precision_scope import ROOT, _functions


@pytest.mark.parametrize("enabled", [False, True])
def test_expert_construction_preserves_process_flush_policy(monkeypatch, enabled):
    policy = NS(enabled=enabled)
    monkeypatch.setattr(torch, "rpu", NS(
        set_ddr_flush=lambda value: setattr(policy, "enabled", value)), raising=False)
    package = "rpu_backend.adapters.rhinovla"
    calls = []
    for name, attributes in {
        package + ".convert": {"convert_expert_for_rpu": lambda model, **kw: model},
        package + ".fused": {"patch_rhino_vla_for_rpu": lambda model, **kw: calls.append((model, kw))},
        "rpu_backend.api._execution": {"_require_execution_process_safe": lambda: None},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    build = _functions("python/rpu_backend/adapters/rhinovla/runtime.py",
                       ["build_rpu_expert"],
                       {"Any": object, "copy": copy, "__package__": package})["build_rpu_expert"]
    expert = NS(qwen_rotary_emb=torch.nn.Identity())
    converted = build(expert, prefix_len=16, suffix_len=31, max_seq_len=64)
    assert converted is not expert
    assert calls[0][0] is converted and calls[0][1]["prefix_len"] == 16
    assert policy.enabled is enabled


def test_vision_pops_are_device_views_with_stable_slot_checks(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp").read_text()
    methods = "\n".join(_definition(source, prefix) for prefix in (
        "std::vector<at::Tensor> pop_deepstack_snapshots(",
        "at::Tensor pop_pooler_merged(",
        "std::vector<at::Tensor> pop_deepstack_merged(",
        "std::vector<at::Tensor> pop_merged("))
    program = r'''
#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("missing slot"); } while (0)
namespace c10 { using Half = float; }
namespace at {
struct Tensor {
    std::shared_ptr<int> storage;
    template<class T> T* data_ptr() { throw std::runtime_error("CPU data access at device handoff"); }
};
}
void rpu_ddr_flush_force(void*) { throw std::runtime_error("flush at device handoff"); }
template<class Key> struct Slots {
    std::map<Key, at::Tensor> values;
    at::Tensor* find(Key key) { auto it=values.find(key); return it==values.end()?nullptr:&it->second; }
};
struct Model {
    std::vector<int64_t> deepstack_visual_indexes_{2, 5, 8};
    int64_t current_num_patches_=768;
    Slots<std::pair<int64_t,int64_t>> deepstack_ddr_slots_, deepstack_merged_ddr_slots_;
    Slots<int64_t> pooler_merged_ddr_slots_;
METHODS
};
int main() {
    Model m;
    m.pooler_merged_ddr_slots_.values[768]={std::make_shared<int>(10)};
    for(auto layer:m.deepstack_visual_indexes_) {
        m.deepstack_ddr_slots_.values[{layer,768}]={std::make_shared<int>(layer)};
        m.deepstack_merged_ddr_slots_.values[{layer,768}]={std::make_shared<int>(layer+20)};
    }
    auto raw=m.pop_deepstack_snapshots(), merged=m.pop_merged();
    assert(raw.size()==3 && merged.size()==4);
    assert(merged[0].storage==m.pop_pooler_merged().storage);
    for(int i=0;i<3;++i) {
        assert(*raw[i].storage==m.deepstack_visual_indexes_[i]);
        assert(*merged[i+1].storage==m.deepstack_visual_indexes_[i]+20);
    }
    // The return value retains the device storage and observes slot reuse.
    *m.pooler_merged_ddr_slots_.values[768].storage=99;
    assert(*merged[0].storage==99);
    m.pooler_merged_ddr_slots_.values.clear();
    assert(*merged[0].storage==99);
    m.current_num_patches_=256;
    int rejected=0;
    try { m.pop_deepstack_snapshots(); } catch(const std::runtime_error&) { ++rejected; }
    try { m.pop_pooler_merged(); } catch(const std::runtime_error&) { ++rejected; }
    try { m.pop_deepstack_merged(); } catch(const std::runtime_error&) { ++rejected; }
    assert(rejected==3);
}
'''.replace("METHODS", methods)
    path = tmp_path / "vision_views.cpp"
    path.write_text(program)
    binary = tmp_path / "vision_views"
    subprocess.run([compiler, "-std=c++17", str(path), "-o", str(binary)], check=True,
                   capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
