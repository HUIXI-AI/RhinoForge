"""Compile actual Vision admission paths without opening an RPU device."""
from pathlib import Path
import shutil
import subprocess

from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]


def test_vision_reduced_weight_and_cache_admission(tmp_path):
    source = (ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp").read_text()
    start = source.index("        TORCH_CHECK(hidden_size % num_heads == 0,")
    weight_gate = source[start:source.index("        // Vision encoder is MHA", start)]
    list_gate = _definition(source, "        auto check_list =") + ";"
    scale_gate = (_definition(source, "static void check_qwen3vl_vision_w4a16_projection(") + "\n"
                  + _definition(source, "static void check_qwen3vl_vision_nvfp4_projection(") + "\n"
                  + _definition(source, "static void check_qwen3vl_vision_w8a16_scale_list("))
    start = source.index("        if (num_cores() != 8) {", source.index("    at::Tensor forward_impl("))
    cache_gate = source[start:source.index("        if (pipeline_dispatch) {", start)]
    start = source.index("    DecoderExecutionTopology resolve_model_execution_topology(")
    topology = source[start:source.index("    KvCostLayoutScope capture_kvinsert", start)]
    cpp = r'''
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <set>
#include <vector>
#include "src/core/execution_topology.h"
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
namespace at {
enum ScalarType { kHalf, kChar, kByte, kFloat };
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
    std::vector<float> values;
    Tensor(std::initializer_list<int64_t> dims) : shape(dims) {}
    Tensor cpu() const { return *this; }
    template <class T> const T* data_ptr() const { return values.data(); }
    bool defined() const { return valid; }
    bool is_contiguous() const { return dense; }
    Device device() const { return where; }
    ScalarType scalar_type() const { return dtype; }
    int64_t dim() const { return shape.size(); }
    int64_t numel()const {int64_t n=1;for(auto d:shape)n*=d;return n;}
    int64_t size(int axis) const { return shape.at(axis); }
    Sizes sizes() const { return shape; }
};
using TensorList = std::vector<Tensor>;
}
namespace c10 {
// FP16 retirement: overflow to inf, underflow to zero.
struct Half {
    float v;
    explicit Half(float x) : v(std::fabs(x) > 65504.f ? std::copysign(INFINITY, x)
                               : std::fabs(x) < 2.98e-8f ? 0.f : x) {}
    explicit operator float() const { return v; }
};
}
__SCALE_GATE__
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
    bool pipeline_dispatch = false, packed_w4 = false;
    double eps = 1e-6;
    std::vector<int64_t> deepstack_visual_indexes{5,11,17};
    at::TensorList q_w_list, k_w_list, v_w_list, o_w_list, fc1_w_list, fc2_w_list;
    at::TensorList ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list;
    at::TensorList q_b_list, k_b_list, v_b_list, o_b_list, fc1_b_list, fc2_b_list;
    at::TensorList q_scale_list, k_scale_list, v_scale_list, o_scale_list, fc1_scale_list, fc2_scale_list;
    at::TensorList nvfp4_tensor_scales;
    static constexpr std::array<const char*, 6> kNvfp4TableNames{"q","k","v","o","fc1","fc2"};
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
    void quantize() {
        std::vector<at::TensorList*> weights{&q_w_list,&k_w_list,&v_w_list,&o_w_list,&fc1_w_list,&fc2_w_list};
        std::vector<at::TensorList*> scales{&q_scale_list,&k_scale_list,&v_scale_list,&o_scale_list,&fc1_scale_list,&fc2_scale_list};
        for(size_t j=0;j<weights.size();++j)for(auto& w:*weights[j]) {
            scales[j]->push_back(at::Tensor{w.size(0)});w.dtype=at::kChar;
        }
    }
    void quantize_w4() {
        // G32 pgrp: uint8 [N,K/2] and the controller-striped [32, numel/32] scale;
        // q/k/v/fc1 split columns, o/fc2 rows.
        packed_w4=true;
        std::vector<at::TensorList*> weights{&q_w_list,&k_w_list,&v_w_list,&o_w_list,&fc1_w_list,&fc2_w_list};
        std::vector<at::TensorList*> scales{&q_scale_list,&k_scale_list,&v_scale_list,&o_scale_list,&fc1_scale_list,&fc2_scale_list};
        for(size_t j=0;j<weights.size();++j)for(auto& w:*weights[j]) {
            const bool col=j!=3&&j!=5;
            const int64_t n=w.size(0),k=w.size(1),local_g=(col?k:k/cores)/32,local_n=col?n/cores:n;
            scales[j]->push_back(at::Tensor{32,cores*((local_g+3)/4)*((local_n+63)/64)*8});
            w.shape[1]=k/2;w.dtype=at::kByte;
        }
    }
    void quantize_nvfp4() {
        // NVFP4: uint8 [N,K/2] weight, striped uint8 [K/16,N] block scale and
        // one FP32 tensor-scale table per projection.
        packed_w4=true;
        std::vector<at::TensorList*> weights{&q_w_list,&k_w_list,&v_w_list,&o_w_list,&fc1_w_list,&fc2_w_list};
        std::vector<at::TensorList*> scales{&q_scale_list,&k_scale_list,&v_scale_list,&o_scale_list,&fc1_scale_list,&fc2_scale_list};
        for(size_t j=0;j<weights.size();++j)for(auto& w:*weights[j]) {
            const int64_t n=w.size(0),k=w.size(1);
            at::Tensor s{k/16,n};s.dtype=at::kByte;scales[j]->push_back(s);
            w.shape[1]=k/2;w.dtype=at::kByte;
        }
        at::Tensor table{(N+7)/8*8};table.dtype=at::kFloat;
        for(int64_t l=0;l<table.size(0);++l)table.values.push_back(0.5f+0.01f*l);
        nvfp4_tensor_scales.assign(6,table);
    }
    void weights() {
        const int64_t hidden_size=h, intermediate_size=intermediate, num_heads=heads, head_dim=hd;
        const bool nvfp4=!nvfp4_tensor_scales.empty();
        __LIST_GATE__
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
            [](auto& v){v.image_batch_count=0;},
            [](auto& v){v.image_batch_count=4;},
            [](auto& v){v.pipeline_dispatch=true;},
            [](auto& v){v.input.dtype=at::kFloat;}
        };
        for (auto corrupt : bad_caches) {
            Vision bad;bad.configure(padded);corrupt(bad);bool rejected=false;
            try {bad.forward();} catch(const std::runtime_error&){rejected=true;}
            assert(rejected && bad.commits==0);
        }
        Vision quant;quant.configure(padded);quant.quantize();quant.weights();quant.forward();assert(quant.commits==2);
        for(int fault=0;fault<5;++fault){Vision bad;bad.configure(padded);bad.quantize();
            if(fault==0)bad.q_scale_list.back().shape[0]-=16;
            if(fault==1)bad.fc2_scale_list.clear();
            if(fault==2)bad.v_w_list.back().dtype=at::kHalf;
            if(fault==3)bad.o_scale_list.back().dtype=at::kFloat;
            if(fault==4)bad.k_scale_list.back().where.kind=0;
            bool rejected=false;try{bad.weights();}catch(const std::runtime_error&){rejected=true;}
            assert(rejected && bad.commits==0);
        }
        // Packed W4: the 2B/4B tower and, at 4 cores, the padded 8B/32B tower;
        // a G32 payload or dtype fault is refused.
        for(int fault=-1;fault<4;++fault){Vision w4;w4.configure(padded);w4.quantize_w4();
            if(fault==0)w4.o_scale_list.back().shape[1]-=8;
            if(fault==1)w4.fc1_scale_list.back().shape[0]=16;
            if(fault==2)w4.k_w_list.back().dtype=at::kChar;
            if(fault==3)w4.fc2_w_list.back().shape[1]-=16;
            bool rejected=false;try{w4.weights();w4.forward();}catch(const std::runtime_error&){rejected=true;}
            assert(rejected==(fault>=0) && w4.commits==(rejected?0:2));
        }
        // At 8 cores the padded o/fc2 rows leave per-core K 160/544: refused.
        {Vision w4;w4.configure(padded);w4.cores=8;w4.quantize_w4();bool rejected=false;
            try{w4.weights();}catch(const std::runtime_error&){rejected=true;}
            assert(rejected==padded && w4.commits==(padded?0:1));}
        // NVFP4: the same exact tower; a block-scale, dtype or tensor-scale table fault is refused.
        for(int fault=-1;fault<7;++fault){Vision fp4;fp4.configure(padded);fp4.quantize_nvfp4();
            if(fault==0)fp4.q_scale_list.back().shape[0]-=1;
            if(fault==1)fp4.fc2_scale_list.back().dtype=at::kHalf;
            if(fault==2)fp4.o_w_list.back().dtype=at::kChar;
            if(fault==3)fp4.nvfp4_tensor_scales.pop_back();
            if(fault==4)fp4.nvfp4_tensor_scales[4].shape[0]=16;
            if(fault==5)fp4.nvfp4_tensor_scales[1].values[3]=0;
            if(fault==6)fp4.nvfp4_tensor_scales[5].values[23]=1e6f;
            bool rejected=false;try{fp4.weights();fp4.forward();}catch(const std::runtime_error&){rejected=true;}
            assert(rejected==(padded||fault>=0) && fp4.commits==(rejected?0:2));
        }
        Vision legacy;legacy.configure(padded);legacy.cores=8;legacy.weights();assert(legacy.commits==1);
    }
}
'''
    for marker, value in (("__SCALE_GATE__",scale_gate),("__LIST_GATE__",list_gate),("__TOPOLOGY__",topology),("__WEIGHT_GATE__",weight_gate),("__CACHE_GATE__",cache_gate)):
        cpp = cpp.replace(marker,value)
    path=tmp_path / "vision_admission.cpp"
    path.write_text(cpp)
    compiler=shutil.which("g++")
    assert compiler
    binary=path.with_suffix("")
    result=subprocess.run([compiler,"-std=c++17","-O0","-I",str(ROOT),str(path),"-o",str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    subprocess.run([str(binary)],check=True,capture_output=True,text=True)
