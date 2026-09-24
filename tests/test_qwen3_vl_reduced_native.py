"""Compile actual Vision admission paths without opening an RPU device."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_vision_reduced_weight_and_cache_admission(tmp_path):
    source = (ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp").read_text()
    start = source.index("        TORCH_CHECK(hidden_size % num_heads == 0,")
    weight_gate = source[start:source.index("        // Vision encoder is MHA", start)]
    start = source.index("        if (num_cores() != 8) {", source.index("    at::Tensor forward_impl("))
    cache_gate = source[start:source.index("        if (pipeline_dispatch) {", start)]
    start = source.index("    DecoderExecutionTopology resolve_model_execution_topology(")
    topology = source[start:source.index("    KvCostLayoutScope capture_kvinsert", start)]
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>
#include "src/core/execution_topology.h"
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
namespace at {
enum ScalarType { kHalf, kChar, kFloat };
constexpr int kPrivateUse1 = 1;
struct Device {
    int kind = kPrivateUse1;
    int type() const { return kind; }
    bool operator==(const Device& other) const { return kind == other.kind; }
};
struct Sizes : std::vector<int64_t> {
    using std::vector<int64_t>::vector;
    std::vector<int64_t> vec() const { return {begin(), end()}; }
};
struct Tensor {
    Sizes shape;
    ScalarType dtype = kHalf;
    Device where;
    bool dense = true, valid = true;
    Tensor(std::initializer_list<int64_t> dims) : shape(dims) {}
    bool defined() const { return valid; }
    bool is_contiguous() const { return dense; }
    Device device() const { return where; }
    ScalarType scalar_type() const { return dtype; }
    int64_t dim() const { return shape.size(); }
    int64_t size(int axis) const { return shape.at(axis); }
    Sizes sizes() const { return shape; }
};
using TensorList = std::vector<Tensor>;
}
using v3::DecoderExecutionTopology;
struct FusedModelBase {
    virtual ~FusedModelBase() = default;
    virtual DecoderExecutionTopology resolve_model_execution_topology(
        int64_t, int64_t, int64_t, int64_t, int64_t) const { return {8,8,8}; }
};
struct Vision : FusedModelBase {
    int cores = 4, commits = 0;
    int64_t N = 24, heads = 16, hd = 64, h = 1024, intermediate = 4096;
    int64_t num_patches_in = 784, image_batch_count = 1;
    bool pipeline_dispatch = false;
    double eps = 1e-6;
    std::vector<int64_t> deepstack_visual_indexes{5,11,17};
    at::TensorList q_w_list, k_w_list, v_w_list, o_w_list, fc1_w_list, fc2_w_list;
    at::TensorList ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list;
    at::TensorList q_b_list, k_b_list, v_b_list, o_b_list, fc1_b_list, fc2_b_list;
    at::TensorList q_scale_list, k_scale_list, v_scale_list, o_scale_list, fc1_scale_list, fc2_scale_list;
    std::vector<at::Tensor> k_caches, v_caches;
    at::Tensor input{1,784,1024};
    int num_cores() const { return cores; }
    int64_t num_layers() const { return N; }
    int64_t hidden_size() const { return h; }
    int64_t head_dim() const { return hd; }
    __TOPOLOGY__
    void configure(bool padded) {
        if (padded) { N=27; hd=80; h=1152; intermediate=4352; deepstack_visual_indexes={8,16,24}; }
        for (auto* list : {&q_w_list,&k_w_list,&v_w_list}) list->assign(N, at::Tensor{heads*hd,h});
        o_w_list.assign(N, at::Tensor{h,heads*hd});
        fc1_w_list.assign(N, at::Tensor{intermediate,h});
        fc2_w_list.assign(N, at::Tensor{h,intermediate});
        for (auto* list : {&ln1_w_list,&ln1_b_list,&ln2_w_list,&ln2_b_list,&o_b_list,&fc2_b_list})
            list->assign(N, at::Tensor{h});
        for (auto* list : {&q_b_list,&k_b_list,&v_b_list}) list->assign(N, at::Tensor{heads*hd});
        fc1_b_list.assign(N, at::Tensor{intermediate});
        input = at::Tensor{1,784,h};
        k_caches.assign(N,at::Tensor{1,128,4,hd/16,8,16,16});
        v_caches=k_caches;
    }
    void weights() {
        const int64_t hidden_size=h, intermediate_size=intermediate, num_heads=heads, head_dim=hd;
        __WEIGHT_GATE__
        const auto topology=resolve_model_execution_topology(heads,heads,hd,h,intermediate);
        assert(topology.num_cores == cores);
        ++commits;
    }
    void forward() {
        __CACHE_GATE__
        ++commits;
    }
};
int main() {
    for (bool padded : {false,true}) {
        Vision good; good.configure(padded); good.weights(); good.forward(); assert(good.commits==2);
        const std::vector<std::function<void(Vision&)>> bad_weights{
            [](auto& v){v.cores=6;}, [](auto& v){v.N-=1;},
            [](auto& v){v.deepstack_visual_indexes={0,1,2};},
            [](auto& v){v.q_w_list.back().shape[0]-=16;},
            [](auto& v){v.o_w_list.back().shape[1]-=16;},
            [](auto& v){v.fc1_w_list.back().shape[0]-=16;},
            [](auto& v){v.fc2_w_list.back().shape[1]-=16;},
            [](auto& v){v.q_b_list.back().shape[0]-=16;},
            [](auto& v){v.fc1_b_list.back().shape[0]-=16;},
            [](auto& v){v.fc2_b_list.back().dtype=at::kFloat;},
            [](auto& v){v.ln2_b_list.back().dense=false;},
            [](auto& v){v.k_w_list.back().where.kind=0;},
            [](auto& v){v.eps=1e-5;},
            [](auto& v){v.fc2_scale_list.push_back(at::Tensor{1});}
        };
        for (auto corrupt : bad_weights) {
            Vision bad;bad.configure(padded);corrupt(bad);bool rejected=false;
            try {bad.weights();} catch(const std::runtime_error&){rejected=true;}
            assert(rejected && bad.commits==0);
        }
        const std::vector<std::function<void(Vision&)>> bad_caches{
            [](auto& v){v.k_caches.back().shape[2]=2;},
            [](auto& v){v.v_caches.back().shape[4]=4;},
            [](auto& v){v.v_caches.back().shape[3]-=1;},
            [](auto& v){v.v_caches.pop_back();},
            [](auto& v){v.k_caches.back().dense=false;},
            [](auto& v){v.v_caches.back().where.kind=0;},
            [](auto& v){v.k_caches[0].shape[1]=INT64_MAX;},
            [](auto& v){v.k_caches.back().dtype=at::kFloat;},
            [](auto& v){v.image_batch_count=2;},
            [](auto& v){v.pipeline_dispatch=true;},
            [](auto& v){v.input.dtype=at::kFloat;}
        };
        for (auto corrupt : bad_caches) {
            Vision bad;bad.configure(padded);corrupt(bad);bool rejected=false;
            try {bad.forward();} catch(const std::runtime_error&){rejected=true;}
            assert(rejected && bad.commits==0);
        }
        Vision legacy;legacy.configure(padded);legacy.cores=8;legacy.weights();assert(legacy.commits==1);
    }
}
'''
    for marker, value in (("__TOPOLOGY__",topology),("__WEIGHT_GATE__",weight_gate),("__CACHE_GATE__",cache_gate)):
        cpp = cpp.replace(marker,value)
    path=tmp_path / "vision_admission.cpp"
    path.write_text(cpp)
    compiler=shutil.which("g++")
    assert compiler
    binary=path.with_suffix("")
    result=subprocess.run([compiler,"-std=c++17","-O0","-I",str(ROOT),str(path),"-o",str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    subprocess.run([str(binary)],check=True,capture_output=True,text=True)
