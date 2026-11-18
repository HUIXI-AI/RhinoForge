"""Compile real native admission and launch code without opening the device."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _function(source, name):
    start = source.index("void " + name + "(")
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                    lambda m: " " * len(m[0]), source)
    brace = masked.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return source[start:end]


def _run_cpp(tmp_path, source):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    cpp = tmp_path / "probe.cpp"
    exe = tmp_path / "probe"
    cpp.write_text(source)
    subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src"),
                    "-I", str(ROOT / "src/core"), "-I", str(ROOT / "src/ops"),
                    str(cpp), "-o", str(exe)], check=True, capture_output=True, text=True)
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)


def test_qwen35_native_profile_admission_does_not_widen_other_decoders(tmp_path):
    _run_cpp(tmp_path, r'''
#include <cassert>
#include <array>
#include "fused/qwen3_5_execution_topology.h"
int main() {
    const std::array<std::array<int64_t, 6>, 4> profiles{{
        {{8,4,256,1024,3584,24}}, {{8,4,256,2048,6144,24}}, {{16,4,256,2560,9216,32}},
        {{16,4,256,4096,12288,32}}}};
    for (const auto& p : profiles) for (int cores : {4,6}) {
        const auto t = v3::resolve_qwen35_reduced_execution_topology(
            cores,p[0],p[1],p[2],p[3],p[4]);
        assert(t.num_cores == cores && t.attention_tp == 4 && t.mlp_tp == cores);
        const auto physical=v3::decoder_mlp_intermediate_size(p[4],t.mlp_tp);
        assert(t.physical_kv_cores == 8 && physical % (32 * cores) == 0);
        assert(physical == (p[4]==3584 && cores==6 ? 3648 : p[4]));
        assert(v3::qwen35_reduced_profile_num_layers(p[0],p[1],p[2],p[3],p[4]) == p[5]);
        for (int axis=0;axis<5;++axis) {
            auto bad=p; ++bad[axis]; bool rejected=false;
            try {v3::resolve_qwen35_reduced_execution_topology(
                cores,bad[0],bad[1],bad[2],bad[3],bad[4]);}
            catch(const std::invalid_argument&) {rejected=true;}
            assert(rejected);
        }
        bool generic_rejected=false;
        try {v3::resolve_decoder_execution_topology(cores,p[0],p[1],p[2],p[3],p[4]);}
        catch(const std::invalid_argument&) {generic_rejected=true;}
        assert(generic_rejected);
    }
    for(int cores : {4,6}) {
        assert(v3::qwen35_reduced_profile_num_layers(8,4,256,1024,3648)==0);
        bool rejected=false;
        try {v3::resolve_qwen35_reduced_execution_topology(cores,8,4,256,1024,3648);}
        catch(const std::invalid_argument&) {rejected=true;}
        assert(rejected);
        const auto legacy=v3::resolve_qwen35_reduced_execution_topology(cores,24,24,256,5120,17408);
        assert(legacy.num_cores==cores && legacy.attention_tp==4 && legacy.mlp_tp==cores);
        assert(v3::qwen35_reduced_profile_num_layers(24,24,256,5120,17408)==64);
        assert(v3::qwen35_reduced_profile_num_layers(24,4,256,5120,17408)==0);
        assert(v3::qwen35_reduced_profile_num_layers(24,24,256,5120,17472)==0);
        assert(v3::decoder_mlp_intermediate_size(17408,cores)==(cores==6?17472:17408));
        bool generic_legacy_rejected=false;
        try {v3::resolve_decoder_execution_topology(cores,24,24,256,5120,17408);}
        catch(const std::invalid_argument&) {generic_legacy_rejected=true;}
        assert(generic_legacy_rejected);
    }
    assert(v3::qwen35_gdn_scalar_slots(48,4)==16);
    assert(v3::qwen35_gdn_scalar_slots(48,8)==8);
    assert(v3::qwen35_gdn_scalar_slots(32,4)==8);
    for(int cores : {4,6,8}) {
        assert(v3::decoder_mlp_intermediate_size(3584,cores)==(cores==6?3648:3584));
        assert(v3::decoder_mlp_intermediate_size(9728,cores)==(cores==6?9792:9728));
        assert(v3::decoder_mlp_intermediate_size(3600,cores)==3600);
    }
    for(int cores : {0,1,2,3,5,7,8,9}) {
        bool rejected=false;
        try {v3::resolve_qwen35_reduced_execution_topology(cores,8,4,256,2048,6144);}
        catch(const std::invalid_argument&) {rejected=true;}
        assert(rejected);
    }
}
''')


def test_binary_launchers_honor_core_prefix_without_changing_register_math(tmp_path):
    source = (ROOT / "src/ops/rpu_eltwise_binary.cpp").read_text()
    names = ("rpu_launch_eltwise_binary_scalar_spm_kernel",
             "rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel",
             "rpu_launch_eltwise_binary_1xC_NxC_spm_kernel",
             "rpu_launch_eltwise_binary_Bx1xC_BxNxC_spm_kernel")
    functions = "\n".join(_function(source, name) for name in names)
    kernel_ids = sorted(set(re.findall(r"KernelId::(\w+)", functions)))
    shim = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
constexpr int NUM_CORES_BINARY=8, VLM_ENTRIES=384, WARP_VECTOR_SIZE=256;
constexpr int THD_VECTOR_SIZE=16, WARP_SIZE=16, FP16_RAB_DATA_TYPE_CFG=7;
template<class A,class B> auto CeilDiv(A a,B b) {return (a+b-1)/b;}
namespace c10 {struct Half {uint16_t x; Half(float f=0):x(static_cast<uint16_t>(f*1024)) {}};}
enum class ValuOpType {MUL, ADD, SUB};
enum class KernelId {__IDS__};
struct Op {int op=2,custom_op=0;};
Op get_valu_op_info(ValuOpType) {return {};}
struct Kernel_t {
    std::string name;
    std::map<int,uint16_t> regs;
    void reset_regs(){regs.clear();}
    void set_regs(int i,uint16_t value){regs[i]=value;}
} kernel;
struct Snapshot {std::string name;std::map<int,uint16_t> regs;std::vector<uint16_t> grid;};
struct Queue {
    int n=0, calls=0;bool broadcast=false;
    Snapshot snapshot;
    void set_broadcast_mode(bool value){broadcast=value;}
    void enqueu_kernel(Kernel_t& k,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
        assert(broadcast && static_cast<int>(cores.size())==n);
        for(int i=0;i<n;++i) assert(cores[i]==i);
        ++calls;snapshot={k.name,k.regs,grid};
    }
} queue;
Kernel_t* get_kernel(KernelId id){kernel.name=std::to_string(static_cast<int>(id));return &kernel;}
Queue* get_queue(int n){queue.n=n;return &queue;}
#define GET_KERNEL(id) get_kernel(id)
#define GET_QUEUE(n) get_queue(n)
struct RpuKernelGraph {
    static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
    Kernel_t* get_kernel_reset(const std::string& name){kernel.name=name;kernel.reset_regs();return &kernel;}
};
void rpu_set_legacy_scm_u16_checked(Kernel_t* k,uint32_t i,uint16_t v,const char*){k->set_regs(i,v);}
'''.replace("__IDS__", ",".join(kernel_ids))
    main = r'''
int main() {
    std::vector<std::function<void(int)>> cases;
    for(auto op : {ValuOpType::ADD,ValuOpType::SUB,ValuOpType::MUL}) {
        cases.push_back([=](int nc){rpu_launch_eltwise_binary_scalar_spm_kernel(4096,0.5,8192,32768,op,nc);});
        for(bool reverse : {false,true}) for(int64_t c : {4,16,128,256,2048,16384}) {
            cases.push_back([=](int nc){rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                4096,8192,8192,64,c,1.0,op,reverse,nc);});
            cases.push_back([=](int nc){rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
                4096,8192,8192,64,c,1.0,op,reverse,nc);});
            cases.push_back([=](int nc){rpu_launch_eltwise_binary_Bx1xC_BxNxC_spm_kernel(
                4096,8192,8192,4,64,c,1.0,op,reverse,nc);});
        }
    }
    for(const auto& call : cases) {
        call(8);const auto baseline=queue.snapshot;
        for(int nc : {1,4,6}) {
            const int before=queue.calls;call(nc);assert(queue.calls==before+1);
            assert(queue.snapshot.name==baseline.name && queue.snapshot.regs==baseline.regs &&
                   queue.snapshot.grid==baseline.grid);
        }
        for(int nc : {0,9}) {
            bool rejected=false;const int before=queue.calls;
            try{call(nc);}catch(const std::runtime_error&){rejected=true;}
            assert(rejected && queue.calls==before);
        }
    }
}
'''
    _run_cpp(tmp_path, shim + functions + main)


def test_actual_qwen35_weight_guards_distinguish_logical_and_prepared_mlp(tmp_path):
    """Run actual pre-mutation validation and FMB metadata, without device tensors."""
    source = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    weights = _function(source, "Qwen3_5Model::set_weights")
    signature, body = weights.split("{", 1)
    validation = body[:body.index("    action_nvfp4_ =")]
    parameter_call, = re.findall(r"    set_model_params\([^;]+;", weights)
    lists = re.findall(r"at::TensorList\s+(\w+)", signature)
    # Tensor-scale tables are optional profile input, not one entry per layer.
    layer_lists = [name for name in lists if name != "nvfp4_tensor_scales"
                   and not name.endswith("_negative_zero_list")]
    nvfp4_header = (ROOT / "src/fused/g05_nvfp4.h").read_text()
    nvfp4_metadata = nvfp4_header[
        nvfp4_header.index("struct G05Nvfp4Tables"):
        nvfp4_header.index("    void install(")
    ]
    text_nvfp4_header = (ROOT / "src/fused/qwen35_text_nvfp4.h").read_text()
    text_nvfp4_metadata = text_nvfp4_header[
        text_nvfp4_header.index("struct Qwen35TextNvfp4Tables"):
        text_nvfp4_header.index("    void install(")
    ]
    kernel_header = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    start = kernel_header.index("enum class KernelId")
    kernel_ids = kernel_header[start:kernel_header.index("};", start) + 2]

    def method_body(text, signature):
        start = text.index(signature)
        begin = text.index("{", start)
        return text[begin:text.index("\n}", begin) + 2]

    cpp = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#include "fused/qwen3_5_execution_topology.h"
#include "ops/rpu_linear_tiling.h"
#include "ops/rpu_pi_wall_tiling.h"
using namespace v3;
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while(false)
namespace at {
constexpr int kPrivateUse1=1, kHalf=2, kChar=3, kByte=4;
using IntArrayRef=std::vector<int64_t>;
struct Tensor {
    IntArrayRef shape; int dtype=kHalf, device_type=kPrivateUse1; bool contiguous=true;
    Tensor()=default;
    Tensor(std::initializer_list<int64_t> dims):shape(dims) {}
    bool defined() const {return !shape.empty();}
    int64_t numel() const {int64_t n=defined()?1:0;for(auto x:shape)n*=x;return n;}
    int64_t dim() const {return shape.size();}
    int64_t size(int axis) const {return shape.at(axis);}
    const IntArrayRef& sizes() const {return shape;}
    int scalar_type() const {if(!defined()) throw std::runtime_error("undefined dtype");return dtype;}
    bool is_contiguous() const {return contiguous;}
    struct Device {int value;int type() const{return value;}
        bool operator==(const Device& other)const{return value==other.value;}};
    Device device() const {return {device_type};}
    Tensor cpu() const {assert(false);return *this;}
    template<class T> const T* data_ptr() const {assert(false);return nullptr;}
    // Text NVFP4 FP8-NaN block-scale scan; unreachable for the dense inputs.
    Tensor bitwise_and(int) const {assert(false);return *this;}
    Tensor eq(int) const {assert(false);return *this;}
    Tensor any() const {assert(false);return *this;}
    template<class T> T item() const {assert(false);return T{};}
};
using TensorList=std::vector<Tensor>;
}
at::Tensor rpu_retain_linear_quant_scale(const at::Tensor&,const at::Tensor& zero,int = 8) {return zero;}
// These device preflights must stay unreachable for the ordinary dense inputs.
__NVFP4_METADATA__
    void install(at::TensorList) {assert(false);}
};
// The dense-text NVFP4 owner stays uninstalled: no input supplies its thirteen tables.
__TEXT_NVFP4_METADATA__
    void install(at::TensorList,int64_t) {assert(false);}
};
__KERNEL_IDS__
struct KernelCache {
    static KernelCache& instance() {static KernelCache cache;return cache;}
    bool has_loaded(KernelId) const {assert(false);return false;}
};
void rpu_require_high_precision_math_kernels() {assert(false);}
struct FusedModelBase {
    struct Impl {
        int num_cores_=8, attn_tp_=8, mlp_tp_=8;
        int64_t num_q_heads_=0,num_kv_heads_=0,head_dim_=0,hidden_size_=0,intermediate_size_=0;
    } impl;
    Impl* pimpl_=&impl;
    int num_cores() const {return impl.num_cores_;}
    virtual int64_t resolve_model_mlp_intermediate_size(int64_t logical_size) const {
        return decoder_mlp_intermediate_size(logical_size,impl.mlp_tp_);
    }
    virtual DecoderExecutionTopology resolve_model_execution_topology(
        int64_t num_q_heads,int64_t num_kv_heads,int64_t head_dim,
        int64_t hidden_size,int64_t intermediate_size) const __BASE_RESOLVE__
    void set_model_params(int64_t num_q_heads,int64_t num_kv_heads,int64_t head_dim,
        int64_t hidden_size,int64_t intermediate_size) __BASE_SET_PARAMS__
};
struct Qwen3_5Model : FusedModelBase {
    __LIST_FIELDS__
    at::Tensor cos{64,32},sin{64,32},final_norm_w{1024};
    at::IntArrayRef layer_is_full,mrope_section{11,11,10};
    int64_t num_q_heads=8,num_kv_heads=4,head_dim=256,hidden_size=1024,intermediate_size=3584;
    int64_t gdn_num_v_heads=16,gdn_key_head_dim=128,gdn_value_head_dim=128;
    int64_t gdn_conv_dim=6144,gdn_conv_kernel=4;
    bool use_silu=true,action_mode_=false,z2_bound_=false,linear_acc32_=false,validated=false;
    bool high_precision_=false,mlp_packed_w4_=false,mlp_awq_g128_=false;
    Qwen35TextNvfp4Tables text_nvfp4_;
    int64_t resolve_model_mlp_intermediate_size(int64_t logical_size) const override __MODEL_MLP_WIDTH__
    double eps=1e-6;
    int gdn_tp() const {return num_cores()==8?8:4;}
    DecoderExecutionTopology resolve_model_execution_topology(
        int64_t nq,int64_t nkv,int64_t hd,int64_t h,int64_t intermediate) const override __MODEL_RESOLVE__
    explicit Qwen3_5Model(int cores) {
        impl.num_cores_=cores;num_kv_heads=cores==8?8:4;
        __LIST_INIT__
        const int gc=gdn_tp();
        const int64_t physical=cores==6?3648:3584;
        for(int i=0;i<24;++i) {
            layer_is_full.push_back(i%4==3);
            input_norm_list[i]=post_norm_list[i]=at::Tensor{1024};
            gate_list[i]=up_list[i]=at::Tensor{physical,1024};down_list[i]={1024,physical};
            if(layer_is_full[i]) {
                q_w_list[i]=attn_gate_list[i]=at::Tensor{2048,1024};
                k_w_list[i]=v_w_list[i]=at::Tensor{num_kv_heads*256,1024};
                o_w_list[i]={1024,2048};q_norm_list[i]=k_norm_list[i]=at::Tensor{256};
            } else {
                gdn_q_list[i]=gdn_k_list[i]=gdn_v_list[i]=gdn_in_z_list[i]=at::Tensor{2048,1024};
                gdn_out_list[i]={1024,2048};gdn_b_bg_list[i]=gdn_a_bg_list[i]=at::Tensor{16*gc,1024};
                gdn_A_log_list[i]=gdn_dt_bias_list[i]=at::Tensor{8*gc};gdn_norm_list[i]={128};
                gdn_cq_list[i]=gdn_ck_list[i]=gdn_cv_list[i]=at::Tensor{gc,4,2048/gc};
            }
        }
    }
    void use_w8_2b() {
        hidden_size=2048;intermediate_size=6144;final_norm_w={2048};
        const auto quant=[](at::Tensor& w,at::Tensor& scale) {
            w.dtype=at::kChar;scale={w.size(0)};
        };
        for(int i=0;i<24;++i) {
            input_norm_list[i]=post_norm_list[i]=at::Tensor{2048};
            gate_list[i]=up_list[i]=at::Tensor{6144,2048};down_list[i]={2048,6144};
            quant(gate_list[i],gate_scale_list[i]);quant(up_list[i],up_scale_list[i]);
            quant(down_list[i],down_scale_list[i]);
            if(layer_is_full[i]) {
                q_w_list[i]=attn_gate_list[i]=at::Tensor{2048,2048};
                k_w_list[i]=v_w_list[i]=at::Tensor{num_kv_heads*256,2048};
                o_w_list[i]={2048,2048};
                quant(q_w_list[i],q_scale_list[i]);quant(k_w_list[i],k_scale_list[i]);
                quant(v_w_list[i],v_scale_list[i]);quant(o_w_list[i],o_scale_list[i]);
                quant(attn_gate_list[i],attn_gate_scale_list[i]);
            } else {
                gdn_q_list[i]=gdn_k_list[i]=gdn_v_list[i]=gdn_in_z_list[i]=at::Tensor{2048,2048};
                gdn_out_list[i]={2048,2048};
                gdn_b_bg_list[i]=gdn_a_bg_list[i]=at::Tensor{16*gdn_tp(),2048};
                quant(gdn_q_list[i],gdn_q_scale_list[i]);quant(gdn_k_list[i],gdn_k_scale_list[i]);
                quant(gdn_v_list[i],gdn_v_scale_list[i]);quant(gdn_in_z_list[i],gdn_in_z_scale_list[i]);
                quant(gdn_out_list[i],gdn_out_scale_list[i]);
            }
        }
    }

    void use_awq_2b() {
        use_w8_2b();
        for(auto* list : {&q_w_list,&k_w_list,&v_w_list,&o_w_list,&attn_gate_list,
                &gdn_q_list,&gdn_k_list,&gdn_v_list,&gdn_in_z_list,&gdn_out_list,
                &gdn_b_bg_list,&gdn_a_bg_list,&gate_list,&up_list,&down_list})
            for(auto& weight:*list) weight.dtype=at::kHalf;
        for(auto* list : {&q_scale_list,&k_scale_list,&v_scale_list,&o_scale_list,
                &attn_gate_scale_list,&gdn_q_scale_list,&gdn_k_scale_list,&gdn_v_scale_list,
                &gdn_in_z_scale_list,&gdn_out_scale_list,&gdn_b_bg_scale_list,
                &gdn_a_bg_scale_list,&gate_scale_list,&up_scale_list,&down_scale_list})
            for(auto& scale:*list) scale={};
        gate_negative_zero_list.resize(24);up_negative_zero_list.resize(24);down_negative_zero_list.resize(24);
        const auto quant=[this](at::Tensor& w,at::Tensor& s,at::Tensor& z,bool row) {
            const int cores=num_cores();
            const auto local_n=w.size(0)/(row?1:cores),local_g=w.size(1)/128/(row?cores:1);
            const auto elements=((local_n+63)/64)*((local_g+3)/4)*cores*4*64;
            w.shape[0]=((local_n+31)/32)*32*(row?1:cores);
            w.dtype=at::kByte;w.shape[1]/=2;s={128,elements/128};z=s;
        };
        for(int i=1;i<24;++i) {
            quant(gate_list[i],gate_scale_list[i],gate_negative_zero_list[i],false);
            quant(up_list[i],up_scale_list[i],up_negative_zero_list[i],false);
            quant(down_list[i],down_scale_list[i],down_negative_zero_list[i],true);
        }
    }
    void use_legacy() {
        num_q_heads=num_kv_heads=24;hidden_size=5120;intermediate_size=17408;
        gdn_num_v_heads=48;gdn_conv_dim=10240;final_norm_w={5120};
        layer_is_full.clear();
        __LEGACY_LIST_INIT__
        const int gc=gdn_tp();
        const int64_t physical=num_cores()==6?17472:17408;
        const auto quant=[](at::Tensor& w,at::Tensor& s) {w.dtype=at::kChar;s={w.size(0)};};
        for(int i=0;i<64;++i) {
            layer_is_full.push_back(i%4==3);
            input_norm_list[i]=post_norm_list[i]=at::Tensor{5120};
            gate_list[i]=up_list[i]=at::Tensor{physical,5120};down_list[i]={5120,physical};
            quant(gate_list[i],gate_scale_list[i]);quant(up_list[i],up_scale_list[i]);quant(down_list[i],down_scale_list[i]);
            if(layer_is_full[i]) {
                q_w_list[i]=k_w_list[i]=v_w_list[i]=attn_gate_list[i]=at::Tensor{6144,5120};
                o_w_list[i]={5120,6144};q_norm_list[i]=k_norm_list[i]=at::Tensor{256};
                quant(q_w_list[i],q_scale_list[i]);quant(k_w_list[i],k_scale_list[i]);
                quant(v_w_list[i],v_scale_list[i]);quant(o_w_list[i],o_scale_list[i]);quant(attn_gate_list[i],attn_gate_scale_list[i]);
            } else {
                gdn_q_list[i]=gdn_k_list[i]=at::Tensor{2048,5120};
                gdn_v_list[i]=gdn_in_z_list[i]=at::Tensor{6144,5120};
                gdn_out_list[i]={5120,6144};gdn_b_bg_list[i]=gdn_a_bg_list[i]=at::Tensor{16*gc,5120};
                gdn_A_log_list[i]=gdn_dt_bias_list[i]=at::Tensor{qwen35_gdn_scalar_slots(48,gc)*gc};
                gdn_norm_list[i]={128};
                gdn_cq_list[i]=gdn_ck_list[i]=at::Tensor{gc,4,2048/gc};gdn_cv_list[i]={gc,4,6144/gc};
            }
        }
    }
    void use_modern27(bool w4=false) {
        use_legacy();
        const auto quant=[w4,this](at::Tensor& w,at::Tensor& s,bool row=false,bool mlp=false) {
            if(!w4) {w.dtype=at::kChar;s={w.size(0)};return;}
            const int cores=mlp?num_cores():gdn_tp();
            if(mlp && num_cores()==6) w.shape[row?1:0]=17472;
            const auto local_n=w.size(0)/(row?1:cores),local_g=w.size(1)/32/(row?cores:1);
            const auto elements=((local_n+63)/64)*((local_g+3)/4)*cores*4*64;
            w.shape[0]=((local_n+31)/32)*32*(row?1:cores);
            w.dtype=at::kByte;w.shape[1]/=2;s={32,elements/32};
        };
        for(int i=0;i<64;++i) {
            quant(gate_list[i],gate_scale_list[i],false,true);quant(up_list[i],up_scale_list[i],false,true);
            quant(down_list[i],down_scale_list[i],true,true);
            if(layer_is_full[i]) {
                quant(q_w_list[i],q_scale_list[i]);quant(k_w_list[i],k_scale_list[i]);
                quant(v_w_list[i],v_scale_list[i]);quant(attn_gate_list[i],attn_gate_scale_list[i]);
                quant(o_w_list[i],o_scale_list[i],true);
            } else {
                quant(gdn_q_list[i],gdn_q_scale_list[i]);quant(gdn_k_list[i],gdn_k_scale_list[i]);
                quant(gdn_v_list[i],gdn_v_scale_list[i]);quant(gdn_in_z_list[i],gdn_in_z_scale_list[i]);
                quant(gdn_b_bg_list[i],gdn_b_bg_scale_list[i]);quant(gdn_a_bg_list[i],gdn_a_bg_scale_list[i]);
                quant(gdn_out_list[i],gdn_out_scale_list[i],true);
            }
        }
    }
    void use_recipe27(const std::string& recipe) {
        use_legacy();
        for(auto* list : {&q_w_list,&k_w_list,&v_w_list,&o_w_list,&attn_gate_list,
                &gdn_q_list,&gdn_k_list,&gdn_v_list,&gdn_in_z_list,&gdn_out_list,
                &gdn_b_bg_list,&gdn_a_bg_list,&gate_list,&up_list,&down_list})
            for(auto& weight:*list) weight.dtype=at::kHalf;
        for(auto* list : {&q_scale_list,&k_scale_list,&v_scale_list,&o_scale_list,
                &attn_gate_scale_list,&gdn_q_scale_list,&gdn_k_scale_list,&gdn_v_scale_list,
                &gdn_in_z_scale_list,&gdn_out_scale_list,&gdn_b_bg_scale_list,
                &gdn_a_bg_scale_list,&gate_scale_list,&up_scale_list,&down_scale_list})
            for(auto& scale:*list) scale={};
        const bool awq=recipe=="awq";
        const bool packed=recipe=="mlp_only"||recipe=="mlp_w8"||recipe=="main_w4"||awq;
        const int mlp_dtype=packed?at::kByte:recipe=="main_w8"?at::kChar:at::kHalf;
        const int attn_dtype=recipe=="main_w4"?at::kByte:
            recipe=="mlp_w8"||recipe=="main_w8"?at::kChar:at::kHalf;
        const int gdn_dtype=recipe=="mlp_w8"||recipe=="main_w8"||recipe=="main_w4"?at::kChar:at::kHalf;
        const int scalar_dtype=recipe=="mlp_w8"?at::kChar:at::kHalf;
        if(awq) {gate_negative_zero_list.resize(64);up_negative_zero_list.resize(64);down_negative_zero_list.resize(64);}
        const auto quant=[this](at::Tensor& w,at::Tensor& scale,int dtype,bool row,bool mlp,int group) {
            if(dtype==at::kHalf)return;
            if(dtype==at::kChar) {w.dtype=dtype;scale={w.size(0)};return;}
            const int cores=mlp?num_cores():gdn_tp();
            const auto local_n=w.size(0)/(row?1:cores),local_g=w.size(1)/group/(row?cores:1);
            const auto elements=((local_n+63)/64)*((local_g+3)/4)*cores*4*64;
            w.shape[0]=((local_n+31)/32)*32*(row?1:cores);
            w.dtype=dtype;w.shape[1]/=2;scale={group,elements/group};
        };
        for(int i=0;i<64;++i) {
            // The first AWQ layer is FP16 but shares every later layer's
            // physical MLP width. This is the actual Python caller layout.
            if(packed && num_cores()==6) {
                const int64_t physical = awq ? 17664 : 17472;
                gate_list[i].shape[0]=up_list[i].shape[0]=physical;
                down_list[i].shape[1]=physical;
            }
            const int dtype=awq&&i==0?at::kHalf:mlp_dtype;
            quant(gate_list[i],gate_scale_list[i],dtype,false,true,awq?128:32);
            quant(up_list[i],up_scale_list[i],dtype,false,true,awq?128:32);
            quant(down_list[i],down_scale_list[i],dtype,true,true,awq?128:32);
            if(awq && i>0) {
                gate_negative_zero_list[i]=gate_scale_list[i];
                up_negative_zero_list[i]=up_scale_list[i];
                down_negative_zero_list[i]=down_scale_list[i];
            }
            if(layer_is_full[i]) {
                quant(q_w_list[i],q_scale_list[i],attn_dtype,false,false,32);
                quant(k_w_list[i],k_scale_list[i],attn_dtype,false,false,32);
                quant(v_w_list[i],v_scale_list[i],attn_dtype,false,false,32);
                quant(attn_gate_list[i],attn_gate_scale_list[i],attn_dtype,false,false,32);
                quant(o_w_list[i],o_scale_list[i],attn_dtype,true,false,32);
            } else {
                quant(gdn_q_list[i],gdn_q_scale_list[i],gdn_dtype,false,false,32);
                quant(gdn_k_list[i],gdn_k_scale_list[i],gdn_dtype,false,false,32);
                quant(gdn_v_list[i],gdn_v_scale_list[i],gdn_dtype,false,false,32);
                quant(gdn_in_z_list[i],gdn_in_z_scale_list[i],gdn_dtype,false,false,32);
                quant(gdn_out_list[i],gdn_out_scale_list[i],gdn_dtype,true,false,32);
                quant(gdn_b_bg_list[i],gdn_b_bg_scale_list[i],scalar_dtype,false,false,32);
                quant(gdn_a_bg_list[i],gdn_a_bg_scale_list[i],scalar_dtype,false,false,32);
            }
        }
    }
    void validate() {
        __VALIDATION__
        // Actual late logical call, isolated from RPU allocations and uploads.
        mlp_packed_w4_=packed_mlp;
        mlp_awq_g128_=!gate_negative_zero_list.empty();
        __PARAMETER_CALL__
        validated=true;
    }
};
template<class F> void reject(int cores,F mutate) {
    Qwen3_5Model m(cores);mutate(m);bool rejected=false;
    try {m.validate();} catch(const std::runtime_error&) {rejected=true;}
    catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected && !m.validated && m.impl.hidden_size_==0 && m.impl.intermediate_size_==0);
}
template<class F> void reject_w8(int cores,F mutate) {
    Qwen3_5Model m(cores);m.use_w8_2b();mutate(m);bool rejected=false;
    try {m.validate();} catch(const std::exception&) {rejected=true;}
    assert(rejected && !m.validated && m.impl.hidden_size_==0);
}
template<class F> void reject_legacy(int cores,F mutate) {
    Qwen3_5Model m(cores);m.use_legacy();mutate(m);bool rejected=false;
    try {m.validate();} catch(const std::runtime_error&) {rejected=true;}
    catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected && !m.validated && m.impl.hidden_size_==0 && m.impl.intermediate_size_==0);
}
template<class F> void reject_modern27(int cores,bool w4,F mutate) {
    Qwen3_5Model m(cores);m.use_modern27(w4);mutate(m);bool rejected=false;
    try {m.validate();} catch(const std::exception&) {rejected=true;}
    assert(rejected && !m.validated && m.impl.hidden_size_==0);
}
template<class F> void reject_recipe27(int cores,const std::string& recipe,F mutate) {
    Qwen3_5Model m(cores);m.use_recipe27(recipe);mutate(m);bool rejected=false;
    try {m.validate();} catch(const std::exception&) {rejected=true;}
    assert(rejected && !m.validated && m.impl.hidden_size_==0);
}
int main() {
    // Ordinary eight-core full-attention callers carry no GDN owners.
    { Qwen3_5Model full(8);
      full.layer_is_full.assign(24,1);
      for(auto* list : {&full.q_w_list,&full.k_w_list,&full.v_w_list,&full.o_w_list,
              &full.attn_gate_list,&full.q_norm_list,&full.k_norm_list})
          for(int i=0;i<24;++i) (*list)[i]=(*list)[3];
      for(auto* list : {&full.gdn_q_list,&full.gdn_b_bg_list})
          for(auto& tensor : *list) tensor={};
      full.validate();assert(full.validated);
    }
    for(int cores : {4,6,8}) { Qwen3_5Model awq(cores);awq.use_awq_2b();awq.validate();assert(awq.validated); }
    for(int fault=0;fault<11;++fault) {
        Qwen3_5Model awq(8);awq.use_awq_2b();
        if(fault==0)awq.gate_negative_zero_list.pop_back();
        if(fault==1)awq.up_negative_zero_list[1]={};
        if(fault==2)awq.down_negative_zero_list[1].device_type=0;
        if(fault==3)awq.gate_negative_zero_list[0]=awq.gate_negative_zero_list[1];
        if(fault==4)awq.q_w_list[3].dtype=at::kChar;
        if(fault==5)awq.intermediate_size=8192;
        if(fault==6)awq.num_q_heads=16;
        if(fault==7)awq.layer_is_full[0]=1;
        if(fault==8)awq.eps=1e-5;
        if(fault==9)awq.gdn_conv_dim=8192;
        if(fault==10)awq.impl.num_cores_=4;
        bool rejected=false;try{awq.validate();}catch(const std::exception&){rejected=true;}
        assert(rejected&&!awq.validated&&awq.impl.hidden_size_==0);
    }
    for(int cores:{4,6,8}) {
        Qwen3_5Model m(cores);m.validate();
        assert(m.validated && m.intermediate_size==3584);
        assert(m.impl.intermediate_size_==(cores==6?3648:3584));
        assert(m.impl.mlp_tp_==cores && m.impl.attn_tp_==(cores==8?8:4));
        Qwen3_5Model modern(cores);modern.use_modern27();modern.validate();
        assert(modern.validated && modern.impl.hidden_size_==5120);
        assert(modern.impl.intermediate_size_==(cores==6?17472:17408));
        reject_modern27(cores,false,[](auto& m){m.gdn_a_bg_list[62].dtype=at::kHalf;m.gdn_a_bg_scale_list[62]={};});
        reject_modern27(cores,false,[](auto& m){m.gate_list[63].dtype=at::kHalf;m.gate_scale_list[63]={};});
        reject_modern27(cores,false,[](auto& m){m.gdn_q_scale_list[62]={2047};});
        {
            Qwen3_5Model w4(cores);w4.use_modern27(true);w4.validate();assert(w4.validated);
            assert(w4.gdn_b_bg_list[0].shape==at::IntArrayRef({32*w4.gdn_tp(),2560}));
            // Logical N16/core is still the output geometry, but the old N16
            // byte owner and arbitrary overpadding are no longer accepted.
            reject_modern27(cores,true,[](auto& m){m.gdn_b_bg_list[0].shape[0]/=2;});
            reject_modern27(cores,true,[](auto& m){m.gdn_a_bg_list[62].shape[0]/=2;});
            reject_modern27(cores,true,[](auto& m){m.gdn_b_bg_list[0].shape[0]*=2;});
            reject_modern27(cores,true,[](auto& m){m.gdn_q_list[0].shape[0]+=32*m.gdn_tp();});
            reject_modern27(cores,true,[](auto& m){m.gdn_out_list[0].shape[0]+=32;});
            reject_modern27(cores,true,[](auto& m){m.gate_list[0].shape[0]+=32*m.num_cores();});
            if(cores==6)
                reject_modern27(cores,true,[](auto& m){m.gdn_b_bg_list[0].shape[0]=32*m.num_cores();});
            reject_modern27(cores,true,[](auto& m){m.gdn_a_bg_scale_list[62]={64,320};});
            reject_modern27(cores,true,[](auto& m){m.gdn_a_bg_scale_list[62]={32,1};});
            reject_modern27(cores,true,[](auto& m){m.down_list[63].shape[1]++;});
            reject_modern27(cores,true,[](auto& m){m.gdn_q_list[62].dtype=at::kChar;});
        }
        for(const std::string recipe : {"fp16","mlp_only","mlp_w8","main_w4","main_w8","awq"}) {
            Qwen3_5Model candidate(cores);candidate.use_recipe27(recipe);candidate.validate();
            assert(candidate.validated && candidate.impl.num_kv_heads_==24);
            assert(candidate.impl.attn_tp_==(cores==8?8:4) && candidate.impl.mlp_tp_==cores);
            assert(candidate.impl.intermediate_size_==(cores!=6 ? 17408 : (recipe=="awq" ? 17664 : 17472)));
            reject_recipe27(cores,recipe,[](auto& m){m.gate_list[63].contiguous=false;});
            reject_recipe27(cores,recipe,[](auto& m){m.gate_list[63].dtype=99;});
            reject_recipe27(cores,recipe,[](auto& m){m.gdn_out_list[62].device_type=0;});
            reject_recipe27(cores,recipe,[](auto& m){m.gdn_q_list[62].dtype=99;});
            reject_recipe27(cores,recipe,[](auto& m){m.num_kv_heads=4;});
            reject_recipe27(cores,recipe,[](auto& m){m.num_kv_heads=8;});
            reject_recipe27(cores,recipe,[](auto& m){m.num_q_heads=32;});
            reject_recipe27(cores,recipe,[](auto& m){m.layer_is_full[63]=0;});
            reject_recipe27(cores,recipe,[](auto& m){m.gdn_num_v_heads=32;});
            reject_recipe27(cores,recipe,[](auto& m){m.down_list[63].shape[1]++;});
            reject_recipe27(cores,recipe,[](auto& m){m.gdn_b_bg_list[62].dtype=99;});
            reject_recipe27(cores,recipe,[](auto& m){m.q_w_list[63].dtype=99;});
            if(recipe!="fp16")
                reject_recipe27(cores,recipe,[](auto& m){m.gate_scale_list[63]={32,1};});
            else
                reject_recipe27(cores,recipe,[](auto& m){m.gate_scale_list[63]={1};});
            if(recipe=="awq") {
                reject_recipe27(cores,recipe,[](auto& m){m.down_negative_zero_list[63]={};});
                reject_recipe27(cores,recipe,[](auto& m){m.up_negative_zero_list[0]=m.up_negative_zero_list[1];});
                reject_recipe27(cores,recipe,[](auto& m){m.down_negative_zero_list[63].contiguous=false;});
                reject_recipe27(cores,recipe,[](auto& m){m.up_negative_zero_list[63].device_type=0;});
                reject_recipe27(cores,recipe,[](auto& m){m.gate_scale_list[63].shape[0]=32;});
                reject_recipe27(cores,recipe,[](auto& m){m.gate_list[0].dtype=at::kByte;});
                if(cores==6)
                    reject_recipe27(cores,recipe,[](auto& m){m.gate_list[0]={17472,5120};});
            } else if(recipe=="mlp_only" || recipe=="mlp_w8" || recipe=="main_w4") {
                reject_recipe27(cores,recipe,[](auto& m){
                    m.gate_scale_list[63].shape[0]=128;
                    m.gate_scale_list[63].shape[1]/=4;
                });
            }
        }
    }
    for(int cores:{4,6}) {
        Qwen3_5Model w8(cores);w8.use_w8_2b();w8.validate();
        assert(w8.validated && w8.impl.hidden_size_==2048 && w8.impl.intermediate_size_==6144);
        assert(w8.impl.attn_tp_==4 && w8.impl.mlp_tp_==cores && w8.gdn_tp()==4);
        reject_w8(cores,[](auto& m){m.gdn_b_bg_list[1].dtype=at::kChar;m.gdn_b_bg_scale_list[1]={64};});
        reject_w8(cores,[](auto& m){m.gdn_a_bg_list[1].dtype=at::kChar;m.gdn_a_bg_scale_list[1]={64};});
        reject_w8(cores,[](auto& m){m.gdn_v_list[0].dtype=at::kHalf;m.gdn_v_scale_list[0]={};});
        reject_w8(cores,[](auto& m){m.gdn_q_scale_list[0]={2047};});
        reject_w8(cores,[](auto& m){m.gdn_out_scale_list[0].dtype=at::kChar;});
        reject_w8(cores,[](auto& m){m.gate_list[23].dtype=at::kByte;});
        reject_w8(cores,[](auto& m){m.down_scale_list[23]={};});
        Qwen3_5Model w8_acc32(cores);w8_acc32.use_w8_2b();
        w8_acc32.linear_acc32_=true;w8_acc32.validate();
        assert(w8_acc32.validated && w8_acc32.impl.hidden_size_==2048);
        Qwen3_5Model legacy(cores);legacy.use_legacy();legacy.validate();
        assert(legacy.validated && legacy.intermediate_size==17408);
        assert(legacy.impl.intermediate_size_==(cores==6?17472:17408));
        assert(legacy.impl.num_kv_heads_==24 && legacy.impl.attn_tp_==4 && legacy.impl.mlp_tp_==cores);
        reject_legacy(cores,[](auto& m){m.gdn_A_log_list[62]={32};});
        reject_legacy(cores,[](auto& m){m.gdn_dt_bias_list[62]={32};});
        reject_legacy(cores,[](auto& m){m.gate_list[63].dtype=at::kHalf;m.gate_scale_list[63]={};});
        reject_legacy(cores,[](auto& m){m.gate_list[63].dtype=at::kByte;});
        reject_legacy(cores,[](auto& m){m.gdn_q_list[62].dtype=at::kChar;m.gdn_q_scale_list[62]={2048};});
        reject_legacy(cores,[](auto& m){m.gate_scale_list[63]={17407};});
        reject_legacy(cores,[](auto& m){m.num_kv_heads=4;});
        reject_legacy(cores,[](auto& m){m.gdn_num_v_heads=32;});
        reject_legacy(cores,[](auto& m){m.layer_is_full[63]=0;});
        if(cores==6) {
            reject_legacy(cores,[](auto& m){m.gate_list[63]={17408,5120};m.gate_list[63].dtype=at::kChar;});
            reject_legacy(cores,[](auto& m){m.gate_scale_list[63]={17408};});
        }
        for(int projection=0;projection<3;++projection) for(int layer:{0,23}) {
            reject(cores,[=](auto& m){
                const int64_t wrong=cores==6?3584:3648;
                if(projection==0)m.gate_list[layer]={wrong,1024};
                if(projection==1)m.up_list[layer]={wrong,1024};
                if(projection==2)m.down_list[layer]={1024,wrong};
            });
        }
        reject(cores,[](auto& m){m.intermediate_size=3648;});
        reject(cores,[](auto& m){m.hidden_size=1025;});
        reject(cores,[](auto& m){m.gdn_num_v_heads=32;m.gdn_conv_dim=8192;});
        reject(cores,[](auto& m){m.gdn_v_list[0]={4096,1024};});
        reject(cores,[](auto& m){m.gdn_norm_list[0]={256};});
        reject(cores,[](auto& m){m.layer_is_full[23]=0;});
        reject(cores,[](auto& m){m.layer_is_full.pop_back();});
        reject(cores,[](auto& m){m.gate_list[23].dtype=9;});
        reject(cores,[](auto& m){m.gate_list[23].device_type=0;});
        reject(cores,[](auto& m){m.gate_list[23].contiguous=false;});
        reject(cores,[](auto& m){m.gate_scale_list[23]={3648};});
        reject(cores,[](auto& m){m.eps=1e-5;});
        reject(cores,[](auto& m){m.mrope_section={10,11,11};});
        reject(cores,[](auto& m){m.action_mode_=true;});
    }
}
'''
    for marker, value in (
        ("__LIST_FIELDS__", "\n".join(f"at::TensorList {name};" for name in lists)),
        ("__LIST_INIT__", "\n".join(f"{name}.resize(24);" for name in layer_lists)),
        ("__LEGACY_LIST_INIT__", "\n".join(f"{name}.assign(64,at::Tensor{{}});" for name in layer_lists)),
        ("__NVFP4_METADATA__", nvfp4_metadata), ("__KERNEL_IDS__", kernel_ids),
        ("__TEXT_NVFP4_METADATA__", text_nvfp4_metadata),
        ("__VALIDATION__", validation), ("__PARAMETER_CALL__", parameter_call),
        ("__BASE_SET_PARAMS__", method_body(base, "void FusedModelBase::set_model_params(")),
        ("__BASE_RESOLVE__", method_body(base, "DecoderExecutionTopology FusedModelBase::resolve_model_execution_topology(")),
        ("__MODEL_MLP_WIDTH__", method_body(source, "int64_t Qwen3_5Model::resolve_model_mlp_intermediate_size(")),
        ("__MODEL_RESOLVE__", method_body(source, "DecoderExecutionTopology Qwen3_5Model::resolve_model_execution_topology(")),
    ):
        cpp = cpp.replace(marker, value)
    _run_cpp(tmp_path, cpp)


def test_actual_qwen35_packed_shape_guard_matches_python_storage(tmp_path):
    import torch
    from rpu_backend.quant.int4_pgrp_pack import swizzle_pack_int4_pgrp, w4_storage_shape

    source = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    start = source.index("            const auto shape = [i]")
    shape_guard = source[start:source.index("\n            };", start) + len("\n            };")]
    cases = []
    for cores in (4, 6, 8):
        for partition in (0, 1):
            for local_n in (16, 32, 48):
                n, k = (local_n * cores, 96) if partition else (local_n, 96 * cores)
                codes = torch.zeros((n, k), dtype=torch.int8)
                weight = swizzle_pack_int4_pgrp(codes, partition, cores).reshape(
                    w4_storage_shape(n, k, partition, cores))
                cases.append("{" + ",".join(map(str, (n, k, partition, cores, *weight.shape))) + "}")
    _run_cpp(tmp_path, r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while(false)
namespace at {
constexpr int kHalf=2,kChar=3,kByte=4;
using IntArrayRef=std::vector<int64_t>;
struct Tensor {
    IntArrayRef shape; int dtype=kByte;
    bool defined() const {return !shape.empty();}
    const IntArrayRef& sizes() const {return shape;}
    int scalar_type() const {return dtype;}
};
}
int main() {
    const int64_t i=0;
    __SHAPE_GUARD__
    struct Case {int64_t n,k,partition,cores,pn,pk;};
    for(const Case c : {__CASES__}) {
        at::Tensor packed{{c.pn,c.pk}};
        shape(packed,{c.n,c.k},"projection",c.partition,c.cores);
        auto reject=[&](at::Tensor tensor,int partition,int64_t cores) {
            bool rejected=false;
            try {shape(tensor,{c.n,c.k},"projection",partition,cores);}
            catch(const std::runtime_error&) {rejected=true;}
            assert(rejected);
        };
        // Reject malformed dimensions even when their product/byte count matches.
        reject({{c.pn/2,c.pk*2}},c.partition,c.cores);
        reject({{c.pn+32,c.pk}},c.partition,c.cores);
        if(c.n!=c.pn) reject({{c.n,c.pk}},c.partition,c.cores);
        reject(packed,-1,c.cores);
        reject(packed,c.partition,0);
        for(int dtype:{at::kHalf,at::kChar})
            shape({{c.n,c.k},dtype},{c.n,c.k},"unpacked",c.partition,c.cores);
    }
}
'''.replace("__SHAPE_GUARD__", shape_guard).replace("__CASES__", ",".join("Case" + c for c in cases)))


def test_actual_gdn_scalar_dma_copies_every_local_head(tmp_path):
    source = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    start = source.index("    const int64_t scalar_slots = qwen35_gdn_scalar_slots(H, nc);")
    end = source.index("    if (decode) {\n        consume_manifest_route", start)
    scalar_dma = source[start:end]
    _run_cpp(tmp_path, r'''
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
#include "fused/qwen3_5_execution_topology.h"
using namespace v3;
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while(false)
namespace c10 {using Half=uint16_t;}
struct Tensor {
    std::vector<uint16_t> data;
    bool defined() const {return !data.empty();}
    template<class T> const T* data_ptr() const {return data.data();}
};
struct Layer {Tensor gdn_A_log,gdn_dt_bias;};
std::vector<std::vector<uint16_t>> received[2];
int D(const char* name) {return std::string(name)=="gdn_Al"?0:1;}
void rpu_launch_ddr_scatter_spm_dma(const uint16_t* src,int64_t n,int64_t stride,int dst,int nc) {
    assert(stride==n*2 && stride%16==0);
    received[dst].assign(nc,{});
    for(int core=0;core<nc;++core)
        received[dst][core].assign(src+core*stride/2,src+core*stride/2+n);
}
int main() {
    constexpr int DWIDTH=2;
    for(int64_t H:{16,32,48}) for(int nc:{4,8}) for(bool decode:{true,false}) {
        const int64_t local=H/nc, slots=((local+7)/8)*8;
        Layer lw;lw.gdn_A_log.data.resize(nc*slots);lw.gdn_dt_bias.data.resize(nc*slots);
        std::vector<Tensor> gdn_neg_exp_A_(1);gdn_neg_exp_A_[0].data.resize(nc*slots);
        const int layer_idx=0;
        for(int core=0;core<nc;++core) for(int head=0;head<local;++head) {
            const int slot=core*slots+head, value=core*local+head+1;
            lw.gdn_A_log.data[slot]=value;lw.gdn_dt_bias.data[slot]=value+100;
            gdn_neg_exp_A_[0].data[slot]=value+200;
        }
        __SCALAR_DMA__
        for(int core=0;core<nc;++core) for(int head=0;head<slots;++head) {
            assert(received[0][core][head]==(head<local?core*local+head+1+(decode?0:200):0));
            assert(received[1][core][head]==(head<local?core*local+head+101:0));
        }
    }
}
'''.replace("__SCALAR_DMA__", scalar_dma))


def test_actual_dense27_linear_route_preserves_existing_padded_routing(tmp_path):
    source = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    start = source.index("FmbLinearRouteSelector Qwen3_5Model::linear_route_selector(")
    route = source[start:source.index("\n}", start) + 2]
    _run_cpp(tmp_path, r'''
#include <cassert>
#include <cstdint>
#include <initializer_list>
namespace at {
enum ScalarType {kHalf,kChar,kByte};
struct Tensor {ScalarType dtype; ScalarType scalar_type() const {return dtype;}};
}
enum class FmbLinearRouteSelector {AUTO_TILE,GEMV};
struct Qwen3_5Model {
    bool linear_acc32_=false;
    int cores=8; int64_t layers=64,hidden=5120,intermediate=17408,heads=24,dimension=256;
    int num_cores() const {return cores;}
    int64_t num_layers() const {return layers;}
    int64_t hidden_size() const {return hidden;}
    int64_t intermediate_size() const {return intermediate;}
    int64_t num_q_heads() const {return heads;}
    int64_t head_dim() const {return dimension;}
    FmbLinearRouteSelector linear_route_selector(const at::Tensor&,int64_t) const;
};
__ROUTE__
int main() {
    for(int cores:{4,6,8}) for(int64_t physical:{17408,17472}) {
        if(cores!=6 && physical!=17408)continue;
        Qwen3_5Model model;model.cores=cores;model.intermediate=physical;
        for(auto dtype:{at::kHalf,at::kChar,at::kByte}) for(int rows:{1,64}) {
            const bool padded_decode = physical!=17408 && rows==1 && dtype!=at::kByte;
            const auto expected = padded_decode ? FmbLinearRouteSelector::GEMV
                                                 : FmbLinearRouteSelector::AUTO_TILE;
            assert(model.linear_route_selector({dtype},rows)==expected);
        }
    }
    Qwen3_5Model small;small.layers=24;small.hidden=2048;small.intermediate=6144;small.heads=8;
    for(auto dtype:{at::kHalf,at::kChar}) {
        assert(small.linear_route_selector({dtype},1)==FmbLinearRouteSelector::GEMV);
        small.linear_acc32_=true;
        assert(small.linear_route_selector({dtype},1)==FmbLinearRouteSelector::AUTO_TILE);
        small.linear_acc32_=false;
    }
}
'''.replace("__ROUTE__", route))
