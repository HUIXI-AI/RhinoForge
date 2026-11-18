"""Execute the native audio admission and physical-weight checks on host doubles."""
from pathlib import Path

from test_qwen35_core_native import _function, _run_cpp

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/fused/rpu_siglip_model.cpp").read_text()


def test_audio_four_core_profile_and_input_boundaries(tmp_path):
    mode = _function(SOURCE, "set_audio_mode")
    start = SOURCE.index("    DecoderExecutionTopology resolve_model_execution_topology(")
    end = SOURCE.index("\n    void set_configured_chunk_size", start)
    topology = SOURCE[start:end].replace(" const override", " const")
    start = SOURCE.index("    void configure_chunk_for_shape(")
    end = SOURCE.index("        require_pi05_vision_owner_ln_profile", start)
    shape = SOURCE[start:end] + "}\n"
    code = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "core/execution_topology.h"
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error("rejected"); } while (false)
using v3::DecoderExecutionTopology;
namespace at { constexpr int kHalf=1; }
struct Weight { int type=at::kHalf; int scalar_type() const {return type;} };
struct Layer { Weight q_w,k_w,v_w,o_w,fc1_w,fc2_w; };
struct FusedModelBase {
    DecoderExecutionTopology resolve_model_execution_topology(int64_t,int64_t,int64_t,int64_t,int64_t) const {return {8,8,8};}
};
struct Model : FusedModelBase {
    int cores=4,layers=18,h=896,i=3584,q=16,d=64,projection_dim_=896,orig_head_dim_=0;
    double eps_=1e-5;
    bool has_patch_emb_=false,audio_mode_=false;
    int epochs=0;
    std::vector<Layer> layer_weights_{18};
    int num_layers() const {return layers;} int hidden_size() const {return h;}
    int intermediate_size() const {return i;} int num_q_heads() const {return q;}
    int head_dim() const {return d;} int num_cores() const {return cores;}
    void invalidate_model_state(){++epochs;}
    __MODE__
    __TOPOLOGY__
    __SHAPE__
};
template<class F> void rejects(F f) {bool failed=false;try{f();}catch(const std::runtime_error&){failed=true;}assert(failed);}
int main() {
    for (int cores : {4,8}) for (bool large : {false,true}) {
        Model m;m.cores=cores;
        if(large){m.layers=24;m.h=1024;m.i=4096;m.projection_dim_=1024;m.layer_weights_.resize(24);}
        const auto t=m.resolve_model_execution_topology(16,16,64,m.h,m.i);
        assert(t.num_cores==cores && t.attention_tp==cores && t.mlp_tp==cores);
        m.set_audio_mode();assert(m.audio_mode_ && m.orig_head_dim_==64 && m.epochs==1);
        for(int length : {7,13,64,103,104})m.configure_chunk_for_shape(length,1);
        if(cores==4){rejects([&]{m.configure_chunk_for_shape(6,1);});rejects([&]{m.configure_chunk_for_shape(105,1);});rejects([&]{m.configure_chunk_for_shape(14,2);});}
    }
    Model bare;rejects([&]{bare.configure_chunk_for_shape(64,1);});
    for(int fault=0;fault<7;++fault){Model m;
        if(fault==0)m.cores=6;if(fault==1)m.layers=19;if(fault==2)m.eps_=1e-6;
        if(fault==3)m.q=14;if(fault==4)m.has_patch_emb_=true;
        if(fault==5)m.projection_dim_=1024;if(fault==6)m.layer_weights_.back().v_w.type=2;
        rejects([&]{m.set_audio_mode();});assert(!m.audio_mode_ && m.epochs==0);
    }
    Model pi;pi.h=1152;pi.i=4352;pi.layers=27;pi.d=80;
    auto t=pi.resolve_model_execution_topology(16,16,80,1152,4352);assert(t.num_cores==4);
    pi.configure_chunk_for_shape(256,1);rejects([&]{pi.configure_chunk_for_shape(64,1);});
    rejects([&]{pi.resolve_model_execution_topology(16,16,64,1024,4095);});
}
'''
    _run_cpp(tmp_path, code.replace("__MODE__", mode).replace("__TOPOLOGY__", topology).replace("__SHAPE__", shape))


def test_audio_four_core_weight_boundary_keeps_vision_precision_contract(tmp_path):
    start = SOURCE.index("        int64_t precision_mask = 0;", SOURCE.index("    void set_weights("))
    end = SOURCE.index("        logical_intermediate_size_ =", start)
    body = SOURCE[start:end]
    code = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error("rejected"); } while (false)
namespace at {constexpr int kHalf=1,kChar=2;using IntArrayRef=std::vector<int64_t>;}
struct Tensor {
    std::vector<int64_t> dims;int type=at::kHalf;
    int scalar_type() const{return type;} int64_t size(int x)const{return dims.at(x);}
    int64_t numel()const{int64_t x=1;for(auto v:dims)x*=v;return x;}
    const auto& sizes()const{return dims;}
};
struct Model {
    int num_cores()const{return 4;}
    void validate(bool large,bool vision,int fault) {
        int64_t N=vision?27:large?24:18,num_heads=16,head_dim=vision?80:64;
        int64_t hidden_size=vision?1152:large?1024:896;
        int64_t intermediate_size=vision?4352:large?4096:3584;
        int64_t projection_dim=vision?2048:hidden_size;
        double eps=vision?1e-6:1e-5;
        int64_t a=num_heads*head_dim;
        std::vector<Tensor> q_w_list(N,{{a,hidden_size}}),k_w_list=q_w_list,v_w_list=q_w_list;
        std::vector<Tensor> o_w_list(N,{{hidden_size,a}}),fc1_w_list(N,{{intermediate_size,hidden_size}}),fc2_w_list(N,{{hidden_size,intermediate_size}});
        std::vector<Tensor> ln1_w_list(N,{{hidden_size}}),ln1_b_list=ln1_w_list,ln2_w_list=ln1_w_list,ln2_b_list=ln1_w_list,o_b_list=ln1_w_list,fc2_b_list=ln1_w_list;
        std::vector<Tensor> q_b_list(N,{{a}}),k_b_list=q_b_list,v_b_list=q_b_list,fc1_b_list(N,{{intermediate_size}});
        Tensor proj_w{{projection_dim,hidden_size}},proj_b{{projection_dim}},post_ln_w{{hidden_size}},post_ln_b{{hidden_size}};
        if(fault==1)q_w_list.back().dims[0]-=16;
        if(fault==2)for(auto& t:q_w_list)t.type=at::kChar;
        if(fault==3)fc1_b_list.back().dims[0]-=16;
        if(fault==4)post_ln_b.dims[0]-=16;
        if(fault==5)eps=1e-6;
        if(fault==6)v_w_list.back().type=at::kChar;
        __WEIGHTS__
        if(vision && fault==2)assert(precision_mask==1);
    }
};
int main(){Model m;
    for(bool large:{false,true}) {
        m.validate(large,false,0);
        for(int fault=1;fault<=6;++fault){bool bad=false;try{m.validate(large,false,fault);}catch(const std::runtime_error&){bad=true;}assert(bad);}
    }
    m.validate(false,true,0);m.validate(false,true,2);
    bool bad=false;try{m.validate(false,true,6);}catch(const std::runtime_error&){bad=true;}assert(bad);
}
'''
    _run_cpp(tmp_path, code.replace("__WEIGHTS__", body))
