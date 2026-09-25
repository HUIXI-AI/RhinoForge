"""Execute installed precision/owner admission from production C++ on the host."""
from pathlib import Path
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]


def test_expert_condition_and_action_io_have_independent_cold_contracts(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/fused/rpu_rhino_vla_model.cpp").read_text()
    io = source[source.index("void set_denoise_loop_weights("):]
    io = io[io.index("TORCH_CHECK"):io.index("full_action_w8a16_ = next_io_w8a16;")]
    table = source[source.index("void set_denoise_loop_adarms_tables("):]
    # Keep the complete production setter, including publication of owners.
    table = table[:table.index("void set_denoise_loop_time_proj_table(")]
    wrapper = _definition(source, "void rpu_rhino_vla_set_denoise_loop_adarms_tables_w8a16(")
    methods = "\n".join(_definition(source, prefix) for prefix in (
        "void validate_expert_transfer(", "std::vector<int64_t> full_action_w8a16_route_arguments("))
    helper = _definition(source, "void check_rhino_full_w8_projection(")
    names = "action_in_w action_in_b action_time_in_w action_time_in_b time_in_action_w time_in_time_w time_in_b time_out_w time_out_b state_w state_b state_mask_w state_mask_b action_mask_w action_mask_b final_norm_w final_norm_b action_out_w action_out_b cos sin".split()
    bindings = "\n".join(f"const auto& {name} = tensors[{i}];" for i, name in enumerate(names))
    program = r'''
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
namespace c10 { using Half = float; }
namespace at {
enum { kHalf=1, kChar=2, kPrivateUse1=3 };
struct Device { int value=kPrivateUse1; int type() const { return value; } bool operator==(Device d) const { return value==d.value; } };
struct Tensor {
    std::vector<int64_t> shape;
    int dtype=kHalf;
    Device dev;
    bool valid=true, contiguous=true;
    std::vector<float> data;
    Tensor(std::vector<int64_t> s={}, int t=kHalf):shape(s),dtype(t),data(8192,1.0f) {}
    bool defined() const { return valid; }
    Device device() const { return dev; }
    int scalar_type() const { return dtype; }
    bool is_contiguous() const { return contiguous; }
    auto sizes() const { return shape; }
    int64_t dim() const { return shape.size(); }
    int64_t size(int i) const { return shape.at(i); }
    int64_t numel() const { return std::accumulate(shape.begin(),shape.end(),int64_t(1),std::multiplies<int64_t>()); }
    int64_t nbytes() const { return numel()*2; }
    template<class T=void> T* data_ptr() const { return reinterpret_cast<T*>(const_cast<float*>(data.data())); }
};
using TensorList=std::vector<Tensor>;
}
bool graph_active=false, transaction_active=false;
int flushes=0;
struct RpuKernelGraph { static bool has_active() { return graph_active; } };
struct RpuExecutionCoordinator {
    static void require_graph_quiescent(const char*) { TORCH_CHECK(!graph_active); }
    static void check_current_thread_execution_allowed(const char*) { TORCH_CHECK(!transaction_active); }
};
void rpu_ddr_flush_force_sized(void*,int64_t) { ++flushes; }
void rpu_ddr_flush_force(c10::Half*) { ++flushes; }
namespace rhino_lkn { uintptr_t RpuGetDevAddr(void* pointer) { return reinterpret_cast<uintptr_t>(pointer); } }
HELPER
struct Model {
    struct Config { bool expert_fusions=true, adarms_resident=true, packed_qkv=false, aligned_kv=true,
        precompute_adarms=true, skip_adarms_gemv=false, precompute_time_proj=false, fold_action_time_in=false; } cold_config_;
    bool expert_w8a16_=true, expert_cond_w8a16_=true, full_action_w8a16_=false, high_precision_=false;
    bool denoise_adarms_tables_ready_=true, denoise_loop_weights_ready_=true, loop_mode_=true, direct_action_input_=true;
    int64_t last_chunk=0;
    int64_t num_steps_=10, action_dim_=96, state_dim_=96, action_horizon_=30, suffix_len_=31;
    at::TensorList full_io_scales_,full_cold_weights_,full_cold_scales_,full_cold_biases_;
    bool gates_tanh_precomputed_=false;
    at::Tensor adarms_pair_table_,adarms_final_table_;
    uintptr_t adarms_pair_table_src_base_=0,adarms_final_table_src_base_=0;
    int generation=0;
    void invalidate_model_state() { ++generation; }
    int64_t num_layers() const { return 18; }
    int64_t hidden_size() const { return 1024; }
    int64_t intermediate_size() const { return 3072; }
    int64_t num_q_heads() const { return 16; }
    int64_t num_kv_heads() const { return 8; }
    int64_t head_dim() const { return 128; }
    int64_t attn_tp() const { return 8; }
    int64_t get_last_resolved_chunk_size() const { return last_chunk; }
    void check_io(const at::TensorList& tensors, const at::TensorList& io_scales) {
        int64_t action_dim=96,action_dim_pad=96,state_dim=96,state_dim_pad=96,action_horizon=30,suffix_len=31;
        bool direct_action_input=true;
        BINDINGS
        IO
    }
    TABLE
    METHODS
};
Model* selected_model=nullptr;
struct RhinoVLARegistry {
    static Model* get(int64_t handle,const char*) { assert(handle==7 && selected_model); return selected_model; }
};
WRAPPER
void install_tables(const at::Tensor& pair,const at::Tensor& final,
                    const at::TensorList& weights,const at::TensorList& scales,const at::TensorList& biases,
                    bool gates=false,bool high=false) {
    rpu_rhino_vla_set_denoise_loop_adarms_tables_w8a16(7,pair,final,weights,scales,biases,gates,high);
}
template<class F> void reject(F f) { bool caught=false; try { f(); } catch(const std::runtime_error&) { caught=true; } assert(caught); }
at::TensorList io_tensors(bool w8) {
    at::TensorList t(21,at::Tensor({1024}));
    for(int i:{0,9,11,13}) t[i]=at::Tensor({1024,96},w8?at::kChar:at::kHalf);
    t[15]=at::Tensor({3072,1024},w8?at::kChar:at::kHalf); t[16]=at::Tensor({3072});
    t[17]=at::Tensor({96,1024},w8?at::kChar:at::kHalf); t[18]=at::Tensor({96});
    t[19]=t[20]=at::Tensor({31,64}); return t;
}
at::TensorList io_scales() { return {at::Tensor({1024}),at::Tensor({1024}),at::Tensor({1024}),at::Tensor({1024}),at::Tensor({3072}),at::Tensor({96})}; }
int main() {
    for(bool full:{false,true}) {
        Model m; selected_model=&m; m.full_action_w8a16_=full;
        const int count=full?19:18;
        for(int i=0;i<count;++i) {
            int n=i==18?3072:6144;
            m.full_cold_weights_.emplace_back(std::vector<int64_t>{n,1024},at::kChar);
            m.full_cold_scales_.emplace_back(std::vector<int64_t>{n});
            m.full_cold_biases_.emplace_back(std::vector<int64_t>{n});
        }
        m.full_io_scales_=full?io_scales():at::TensorList{};
        auto tensors=io_tensors(full);
        m.check_io(tensors,m.full_io_scales_);
        for(int boundary=0;boundary<3;++boundary) {
            graph_active=boundary==0; transaction_active=boundary==1; m.last_chunk=boundary==2?31:0;
            const int before_flushes=flushes;
            reject([&]{m.check_io(tensors,m.full_io_scales_);});
            assert(flushes==before_flushes && m.full_action_w8a16_==full);
            assert(m.full_cold_weights_.size()==count && m.full_cold_scales_.size()==count && m.full_cold_biases_.size()==count);
            assert(m.full_io_scales_.size()==(full?6:0) && m.denoise_adarms_tables_ready_);
            graph_active=transaction_active=false; m.last_chunk=0;
        }
        at::Tensor pair({10,18,6144}), final({10,3072});
        auto check=[&] { install_tables(pair,final,m.full_cold_weights_,m.full_cold_scales_,m.full_cold_biases_); };
        check(); m.validate_expert_transfer(31);
        assert(m.generation==1 && m.full_action_w8a16_==full && m.expert_cond_w8a16_);
        assert(m.adarms_pair_table_.sizes()==pair.sizes() && m.adarms_final_table_.sizes()==final.sizes());
        const auto installed_generation=m.generation;
        const auto installed_pair_addr=m.adarms_pair_table_src_base_;
        const auto installed_final_addr=m.adarms_final_table_src_base_;
        for(int boundary=0;boundary<3;++boundary) {
            graph_active=boundary==0; transaction_active=boundary==1; m.last_chunk=boundary==2?31:0;
            const int before_flushes=flushes; reject(check);
            assert(flushes==before_flushes && m.denoise_adarms_tables_ready_ && m.full_cold_weights_.size()==count);
            assert(m.generation==installed_generation && m.adarms_pair_table_src_base_==installed_pair_addr &&
                   m.adarms_final_table_src_base_==installed_final_addr && m.full_action_w8a16_==full);
            graph_active=transaction_active=false; m.last_chunk=0;
        }
        assert(m.full_action_w8a16_route_arguments()==std::vector<int64_t>({18,10,full?37:36,full?5:0,8,16}));
        auto saved=m.full_cold_weights_; m.full_cold_weights_.pop_back(); reject(check); reject([&]{m.validate_expert_transfer(31);});
        m.full_cold_weights_=saved; m.full_cold_weights_.push_back(saved[0]); reject(check);
        m.full_cold_weights_=saved; m.full_cold_weights_[0].dtype=at::kHalf; reject(check); m.full_cold_weights_=saved;
        m.full_cold_scales_[0].data[0]=0; reject(check); m.full_cold_scales_[0].data[0]=1;
        m.full_cold_biases_[0].dtype=at::kChar; reject(check); m.full_cold_biases_[0].dtype=at::kHalf;
        m.denoise_loop_weights_ready_=false; reject(check); m.denoise_loop_weights_ready_=true;
        graph_active=true; reject(check); graph_active=false;
        final.shape[0]=9; reject(check); final.shape[0]=10;
        m.full_action_w8a16_=!full; reject(check); reject([&]{m.validate_expert_transfer(31);}); m.full_action_w8a16_=full;
        assert(m.generation==installed_generation && m.adarms_pair_table_src_base_==installed_pair_addr &&
               m.adarms_final_table_src_base_==installed_final_addr && m.full_action_w8a16_==full);
        auto wrong_scales=m.full_cold_scales_; wrong_scales.pop_back();
        reject([&]{install_tables(pair,final,m.full_cold_weights_,wrong_scales,m.full_cold_biases_);});
        auto wrong_biases=m.full_cold_biases_; wrong_biases.pop_back();
        reject([&]{install_tables(pair,final,m.full_cold_weights_,m.full_cold_scales_,wrong_biases);});
        auto wrong_weights=m.full_cold_weights_; wrong_weights.resize(full?18:19,wrong_weights[0]);
        wrong_scales=m.full_cold_scales_; wrong_scales.resize(full?18:19,wrong_scales[0]);
        wrong_biases=m.full_cold_biases_; wrong_biases.resize(full?18:19,wrong_biases[0]);
        reject([&]{install_tables(pair,final,wrong_weights,wrong_scales,wrong_biases);});
        assert(m.generation==installed_generation && m.full_action_w8a16_==full && m.expert_cond_w8a16_);
        m.expert_cond_w8a16_=false; reject(check); m.expert_cond_w8a16_=true;
        assert(m.generation==installed_generation && m.full_action_w8a16_==full);
        auto bad=tensors; bad[0].dtype=full?at::kHalf:at::kChar; reject([&]{m.check_io(bad,m.full_io_scales_);});
        bad=tensors; bad[1].dtype=at::kChar; reject([&]{m.check_io(bad,m.full_io_scales_);});
        if(full) {
            auto scales=io_scales(); scales.pop_back(); reject([&]{m.check_io(tensors,scales);});
            m.high_precision_=true; m.check_io(tensors,m.full_io_scales_);
            install_tables(pair,final,m.full_cold_weights_,m.full_cold_scales_,m.full_cold_biases_,false,true);
            m.expert_cond_w8a16_=false; reject([&]{m.check_io(tensors,m.full_io_scales_);});
        } else {
            m.high_precision_=true; reject([&]{m.check_io(tensors,{});}); reject([&]{m.validate_expert_transfer(31);});
            reject([&]{install_tables(pair,final,m.full_cold_weights_,m.full_cold_scales_,m.full_cold_biases_,false,true);});
        }
    }
}
'''
    for key, value in {"HELPER": helper, "WRAPPER": wrapper, "BINDINGS": bindings, "IO": io, "TABLE": table, "METHODS": methods}.items():
        program = program.replace("\n" + key + "\n", "\n" + value + "\n") if key in {"HELPER", "WRAPPER"} else program.replace("        " + key + "\n", value + "\n").replace("    " + key + "\n", value + "\n")
    path = tmp_path / "precision.cpp"
    path.write_text(program)
    binary = tmp_path / "precision"
    subprocess.run([compiler, "-std=c++17", "-O0", str(path), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
