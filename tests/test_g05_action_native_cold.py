"""Actual G05 native query/schedule bodies; no backend load or device execution."""
from pathlib import Path
import re
import subprocess

from test_siglip_cold_profile import _body
from test_g05_vision_owner_model import _block
from pi_wall_host_catalog import with_actual_linear_contracts
from test_g05_q_gate_native import _with_actual_g05_site_helpers


def _precision_site_methods(source):
    names = ("text_gateup_pair_site", "action_nvfp4_o_resident_site",
             "action_nvfp4_down_resident_site", "action_nvfp4_swiglu_site",
             "action_gateup_pair_site")
    return "\n".join(f"int64_t {name}() const " + _body(
        source, f"int64_t Qwen3_5Model::{name}(") for name in names)


def test_actual_action_resident_gateup_cold_owner_routes_lifetime_and_order(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    header = (Path(__file__).resolve().parents[1] / "src/fused/rpu_qwen3_5_model.h").read_text()
    identity = _block(header, "std::vector<int64_t> kvinsert_cost_weight_identity() const")
    weights = _block(header, "struct LayerWeights")
    methods = {
        "PREPARE": ("void prepare_action_gateup_resident()", "void Qwen3_5Model::prepare_action_gateup_resident("),
        "MOD_PROFILE": ("bool use_action_mod_body() const", "bool Qwen3_5Model::use_action_mod_body("),
        "PROFILE": ("bool use_action_gateup_resident(int64_t rows) const", "bool Qwen3_5Model::use_action_gateup_resident("),
        "ARGS": ("std::vector<int64_t> action_gateup_resident_arguments(int64_t layer,int64_t projection) const",
                 "std::vector<int64_t> Qwen3_5Model::action_gateup_resident_arguments("),
        "PRELOAD": ("void preload_action_gateup_resident()", "void Qwen3_5Model::preload_action_gateup_resident("),
        "LAUNCH": ("void launch_action_gateup_resident(int64_t layer,int64_t projection,uint32_t output)",
                   "void Qwen3_5Model::launch_action_gateup_resident("),
        "RESIDENT_COUNT": ("int64_t action_gateup_resident_layers() const", "int64_t Qwen3_5Model::action_gateup_resident_layers("),
        "LAYOUT": ("int64_t subclass_layout_hash() const", "int64_t Qwen3_5Model::subclass_layout_hash("),
    }
    mlp = _body(source, "void Qwen3_5Model::emit_mlp_and_output(")
    mlp = mlp[mlp.index("    const int64_t mlp_elems ="):mlp.index("    if (action_mlp_tail_handoff(layer_idx,chunk,true))")]
    reset = source[source.index("    action_mode_ = false;", source.index("void Qwen3_5Model::set_weights(")):]
    reset = reset[:reset.index("    action_rtc_active_ = false;")]
    cpp = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
#define DWIDTH 2
#define NUM_CORES 8
namespace c10 { struct Half { explicit Half(double) {} }; enum class DeviceType{PrivateUse1,CPU}; }
namespace at {
constexpr int kHalf=2,kChar=3;
struct Tensor {
    int id=0,type=kHalf;int64_t rows=4096,cols=1024;int rank=2;
    bool is_contiguous()const{return true;}int dim()const{return rank;}int64_t size(int i)const{return i?cols:rows;}
    struct Device{c10::DeviceType type()const{return c10::DeviceType::PrivateUse1;}};Device device()const{return {};}
    bool defined() const {return id!=0;}
    int scalar_type() const {return type;}
};
}
namespace detail { int64_t layout_mix(int64_t a,int64_t b) {return uint64_t(a)*31+uint64_t(b);} }
enum class KernelId { G05_GATEUP_RESIDENT_FP16_ACC32_M304N80K128,G05_GATEUP_RESIDENT_W8A16_ACC32_M240N96K128,G05_GATEUP_RESIDENT_W8A16_ACC16_M352N96K128 };
struct KernelCache {
    bool available=true,acc16_available=true; int queries=0;
    static KernelCache& instance() {static KernelCache k;return k;}
    bool has_loaded(KernelId id) {++queries;return available&&(acc16_available||id!=KernelId::G05_GATEUP_RESIDENT_W8A16_ACC16_M352N96K128);}
};
int packs=0,pack_generation=0;std::vector<int>scales_seen;
at::Tensor rpu_pack_g05_gateup_resident_fp16(const at::Tensor& gate,const at::Tensor& up) {
    assert(gate.id>=11 && gate.id<=13 && up.id==gate.id+90);
    ++packs;return {900+gate.id-11};
}
at::Tensor rpu_pack_g05_gateup_resident_w8a16(const at::Tensor&g,const at::Tensor&u){assert(g.id>=11&&g.id<=16&&u.id==g.id+90&&g.type==at::kChar&&u.type==at::kChar);++packs;return {1900+100*pack_generation+g.id-11,at::kChar};}
enum class FmbRouteFamily { LINEAR, MUTABLE_DMA, ACTIVATION };
enum class ValuOpType { SILU, MUL };
enum class StorageClass { Temp };
enum class BufferScope { LayerWide };
__ENUM__
constexpr int64_t qwen35_text_dma_route(Qwen35TextMutableDmaRoute r) {return int64_t(r);}
__CONSTANTS__
struct ChunkInfo {int64_t len=32,idx=0;};
struct Route {int64_t site;FmbRouteFamily family;int64_t selector,flags;
              std::vector<int64_t> arguments;int64_t invocation;};
struct BufferDecl {const char* name;int64_t size,first,last;StorageClass storage;
                   int64_t per_layer;const char* alias;BufferScope scope;};
struct Call {int kind;uint32_t input,weight,output;};
std::vector<Call> calls;
int dmas=0;
constexpr uint32_t resident_base=0x40100000;
void rpu_launch_ddr_scatter_spm_dma(const at::Tensor& owner,int64_t offset,int64_t elements,
                                    int64_t stride,uint32_t dst,int cores) {
    assert(owner.id==900+dmas%3 && offset==0 && elements==1048576 && stride==2097152 &&
           dst==resident_base+(owner.id-900)*2097152 && cores==8);++dmas;
}
void rpu_launch_g05_gateup_resident_fp16_kernel(uint32_t input,uint32_t weight,uint32_t output) {
    const auto matrix=(weight-resident_base)/1048576;
    assert(matrix<6 && (weight-resident_base)%1048576==0);
    calls.push_back({matrix%2==0?1:3,input,weight,output});
}
void rpu_launch_g05_gateup_resident_w8a16_preload(const at::Tensor& owner,uint32_t dst){assert(owner.type==at::kChar&&owner.id==1900+100*pack_generation+dmas%6&&dst==resident_base+(owner.id%100)*1048576);++dmas;}
std::vector<bool>linear_modes;
void rpu_launch_g05_gateup_resident_w8a16_kernel(uint32_t input,uint32_t weight,uint32_t output,const at::Tensor&scale,bool linear_acc32=true){linear_modes.push_back(linear_acc32);const auto matrix=(weight-resident_base)/524288;assert(matrix<12&&(weight-resident_base)%524288==0&&scale.defined()&&scale.rank==1&&scale.id==403+matrix);scales_seen.push_back(scale.id);calls.push_back({matrix%2==0?1:3,input,weight,output});}
void rpu_launch_eltwise_unary_spm_kernel(uint32_t input,uint32_t output,int64_t elems,
                                       ValuOpType,bool,int cores) {
    assert(elems==32*512 && cores==8);calls.push_back({2,input,0,output});
}
void rpu_launch_eltwise_binary_spm_kernel(uint32_t input,uint32_t weight,uint32_t output,
        int64_t elems,ValuOpType,c10::Half,int cores) {
    assert(elems==32*512 && cores==8);calls.push_back({4,input,weight,output});
}
struct Weights __WEIGHTS__;
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>& out,const at::Tensor&t){out.insert(out.end(),{t.id,t.type,t.rank,t.rows,t.cols});}
void append_kvinsert_cost_scalar_identity(std::vector<int64_t>&out,double value){out.push_back(static_cast<int64_t>(value*1e12));}
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>&v,int owner){v.push_back(owner);}
struct Model {
 static constexpr int64_t QWEN35_TEXT_SILU_ROUND_MUL_SITE=7739768473980724746LL;
 bool text_silu_round_mul_available_=false;
 void prepare_text_silu_round_mul(){}
 template<class...T>bool use_text_silu_round_mul(T...)const{return false;}
 std::vector<int64_t>text_silu_round_mul_arguments()const{return {};}
 template<class...T>void launch_text_silu_round_mul(T...){assert(false);}

 static constexpr int64_t QWEN35_TEXT_OWNER_RMS_SITE=8722488789555919585LL;
 bool text_owner_rms_available_=false;
 void prepare_text_owner_rms(){}
 std::vector<int64_t> text_owner_rms_arguments()const{return {};}

 static constexpr int64_t QWEN35_ACTION_GATE_KV_OWNER_W8_SITE=5392460504564407201LL;
 static constexpr int64_t QWEN35_ACTION_QK_NORM_ROPE_GROUP4_SITE=696118694548312598LL;
 static constexpr int64_t QWEN35_ACTION_OWNER_CACHE_JOINT_SITE=7450172011194584301LL;
 static constexpr int64_t QWEN35_ACTION_ATTN_SIGMOID_MUL_SITE=8546526474721653363LL;
 bool action_owner_cache_joint_available_=false;
 void prepare_action_owner_cache_joint(){}
 bool use_action_owner_cache_joint(int64_t,int64_t)const{return false;}
 std::vector<int64_t> action_owner_cache_joint_abi()const{return {};}
 template<class...T>std::vector<int64_t> action_owner_cache_joint_arguments(T...)const{return {};}
 static constexpr int64_t QWEN35_ACTION_TAIL_SITE=7035949248365271364LL;
 static constexpr int64_t QWEN35_ACTION_MLP_TAIL_SITE=4368884220398897667LL;
 bool action_mlp_tail_available_=false;
 void prepare_action_mlp_tail(){}
 template<class...T>bool use_action_mlp_tail(T...)const{return false;}
 template<class...T>bool action_mlp_tail_handoff(T...){return false;}
 std::vector<int64_t>action_mlp_tail_arguments()const{return {};}
 template<class...T>void launch_action_mlp_tail(T...){assert(false);}
 static constexpr int64_t QWEN35_TEXT_MLP_TAIL_SITE=8475656447499929675LL;
 bool text_mlp_tail_available_=false;
 void prepare_text_mlp_tail(){}
 template<class...T>bool use_text_mlp_tail(T...)const{return false;}
 template<class...T>bool text_mlp_tail_handoff(T...){return false;}
 std::vector<int64_t>text_mlp_tail_arguments(int64_t=245)const{return {};}
 template<class...T>void launch_text_mlp_tail(T...){assert(false);}
 bool action_tail_available_=false;
 void prepare_action_tail(){}
 bool use_action_tail(int64_t)const{return false;}
 std::vector<int64_t>action_tail_arguments()const{return {};}
 template<class...T>void launch_action_tail(T...){assert(false);}
 bool action_attn_sigmoid_mul_available_=false;
 void prepare_action_attn_sigmoid_mul(){}
 bool use_action_attn_sigmoid_mul(int64_t)const{return false;}
 std::vector<int64_t> action_attn_sigmoid_mul_arguments()const{return {};}
 template<class...T>void launch_action_attn_sigmoid_mul(T...){assert(false);}
 static constexpr int64_t QWEN35_TEXT_GDN_OWNER_RMS_SITE=7871266237092441430LL;
 bool text_gdn_owner_rms_available_=false;
 std::vector<int64_t> text_gdn_owner_rms_arguments()const{return {};}
 bool action_nvfp4_qk_norm_rope_available_=false;
 bool use_action_nvfp4_qk_norm_rope(int64_t,int64_t)const{return false;}
 std::vector<int64_t> action_nvfp4_qk_norm_rope_arguments()const{return {};}
 void prepare_action_nvfp4_qk_norm_rope(){}
 template<class...T>void launch_action_nvfp4_qk_norm_rope(T...){assert(false);}
 bool action_gate_kv_owner_available_=false;
 bool use_action_gate_kv_owner(int64_t,int64_t)const{return false;}
 std::vector<int64_t> action_gate_kv_owner_arguments(int64_t)const{return {};}
 void prepare_action_gate_kv_owner(){}
 template<class...T>void launch_action_gate_kv_owner(T...){assert(false);}
 bool action_nvfp4_swiglu_available_=false;
 bool use_action_nvfp4_swiglu(int64_t) const{return false;}
 std::vector<int64_t> action_nvfp4_swiglu_arguments() const{return {};}
 void prepare_action_nvfp4_swiglu(){}
 template<class...T>void launch_action_nvfp4_swiglu(T...){assert(false);}

 // This legacy fixture exercises FP16/W8; mixed NVFP4 has its own actual-body test.
 struct Nvfp4Stub {bool enabled()const{return false;}std::vector<int>values;};
 Nvfp4Stub action_nvfp4_;
 // The dense-text NVFP4 owner is uninstalled in this probe (enabled() false, no tables).
 Nvfp4Stub text_nvfp4_;static constexpr int64_t kTextNvfp4TablesSite=3184997657402357068LL;
 template<class...T>std::vector<int64_t>action_gateup_pair_arguments(T...)const{return {};}
 bool use_action_nvfp4(int64_t)const{return false;}
 std::vector<int64_t>action_nvfp4_arguments(int)const{assert(false);return {};}
 static constexpr int64_t QWEN35_ACTION_NVFP4_LINEAR_SITE=1171762392203719440LL;

 bool use_g05_gdn_local_slice(int64_t)const{return false;}
 std::vector<int64_t>g05_gdn_local_slice_arguments(int64_t)const{return {};}

 bool action_rtc_norm_shift_available_=false,action_sub_gated_residual_available_=false,action_rtc_sub_gated_residual_available_=false;
 bool action_ring_rtc_residual_available_=false;
 bool use_action_ring_rtc_residual(int64_t)const{return false;}
 std::vector<int64_t>action_ring_rtc_residual_arguments()const{return {};}
 template<class...T>void launch_action_ring_rtc_residual(T...){ }
 bool action_down_partial_available_=false;at::Tensor action_down_partial_;
 bool use_action_down_partial(int64_t,int64_t)const{return false;}
 std::vector<int64_t>action_down_partial_arguments(bool)const{return {};}
 void prepare_action_down_partial(){}
 template<class...T>void launch_action_down_partial(T...){assert(false);}
 bool action_nvfp4_o_resident_available_=false;std::pair<int,int>action_nvfp4_o_resident_;
 bool use_action_nvfp4_o_resident(int64_t)const{return false;}
 std::vector<int64_t>action_nvfp4_o_resident_arguments(int64_t,int64_t)const{return {};}
 void prepare_action_nvfp4_o_resident(){}void preload_action_nvfp4_o_resident(){}
 template<class...T>void launch_action_nvfp4_o_resident(T...){assert(false);}
 bool action_nvfp4_down_resident_available_=false;std::pair<int,int>action_nvfp4_down_resident_;
 bool use_action_nvfp4_down_resident(int64_t,int64_t)const{return false;}
 std::vector<int64_t>action_nvfp4_down_resident_arguments(int64_t,int64_t)const{return {};}
 void prepare_action_nvfp4_down_resident(){}void preload_action_nvfp4_down_resident(){}
 template<class...T>void launch_action_nvfp4_down_resident(T...){assert(false);}
 bool action_o_resident_available_=false,action_o_resident_n128_=false;
 bool use_action_o_resident(int64_t)const{return false;}
 std::vector<int64_t>action_o_resident_arguments(int64_t,bool)const{return {};}
 bool action_mod_body_fp16_=false,action_mod_body_w8_=false;
 bool action_owner_cache_available_=false;std::vector<int64_t>action_k_owner_norm_rope_arguments()const{return {};}
 bool action_qkv_gate_available_=false;
 bool action_gateup_resident_w8_=false,action_kv_owner_w8_=false,action_q_gate_w8_=false,action_gateup_pair_nvfp4_=false,action_gateup_pair_w8_=false,text_gateup_pair_w8_=false;
    bool action_mode_=true,linear_acc32_=true,has_gdn_=false,use_silu_=true;
    bool action_step_active_=false,action_loop_active_=false,complete=true;
    bool action_gateup_resident_available_=false;
    bool action_kv_owner_available_=false,action_q_gate_available_=false,action_gateup_pair_available_=false,action_silu_round_mul_available_=false,text_gateup_pair_available_=false;
    bool fast_replay_enabled_=false,has_qk_norm_=false,has_mrope_=false;
    double eps_=1e-6;int64_t chunk_size_cap_=0,sdpa_kernel_=0;
    at::Tensor final_norm_w_,action_input_w_,action_input_b_,action_output_w_,action_output_b_;
    bool action_norm_shift_available_=true,action_gated_residual_available_=true;
    int64_t action_num_steps_=1,action_len_=32,layers=24,h=1024,intermediate=4096,cores=8;
    int64_t heads=8,kv_heads=8,hd=256,tp=8,body_iter=0;
    std::vector<at::Tensor> action_gateup_resident_,action_o_resident_;
    std::vector<Weights> layer_weights_{24};
    std::vector<uint8_t> layer_is_full_=std::vector<uint8_t>(24,1);
    std::vector<Route> routes;
    int receipts=0;
    Model() {
        for(int i=0;i<24;++i) {
            auto& w=layer_weights_[i];
            w.q_w=w.k_w=w.v_w=w.o_w=w.attn_gate_w={1};
            w.gate_w={10+i};w.up_w={100+i};w.down_w={200+i};
        }
    }
    int64_t num_layers() const {return layers;}
    int64_t hidden_size() const {return h;}
    int64_t intermediate_size() const {return intermediate;}
    int64_t num_cores() const {return cores;}
    int64_t mlp_tp() const {return tp;}
    int64_t attn_tp() const {return tp;}
    int64_t num_q_heads() const {return heads;}
    int64_t num_kv_heads() const {return kv_heads;}
    int64_t head_dim() const {return hd;}
    std::vector<int64_t> cold_topology_arguments() const {return {cores};}
    bool use_action_silu_round_mul(int64_t) const {return false;}
    bool use_action_gateup_pair(int64_t) const {return false;}
    void launch_action_gateup_pair(int,const ChunkInfo&) {assert(false);}
    void launch_action_silu_round_mul(int,const ChunkInfo&) {assert(false);}
    bool use_text_gateup_pair(int64_t) const {return false;}
    void launch_text_gateup_pair(int,const ChunkInfo&) {assert(false);}
    bool use_action_q_gate(int64_t) const {return false;}
 bool use_action_qkv_gate(int64_t)const{return false;}std::vector<int64_t>action_qkv_gate_arguments()const{return {};}
    bool use_action_kv_owner(int64_t) const {return false;}
    Model& ctx() {return *this;}
    bool has_complete_physical_manifest() const {return complete;}
    uint32_t addr(int,const char* name) const {
        std::string n(name);
        if(n=="action_gateup_resident")return resident_base;
        if(n=="residual1")return 0x40000000;
        if(n=="gate")return 0x40010000;
        if(n=="up")return 0x40018000;
        return 0x40020000;
    }
    void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,
            std::vector<int64_t> args,int64_t invocation) {
        if(family==FmbRouteFamily::ACTIVATION)return;
        auto r=std::find_if(routes.begin(),routes.end(),[&](const auto& r) {
            return r.family==family && r.site==site && r.invocation==invocation;});
        TORCH_CHECK(r!=routes.end() && r->selector==selector && r->flags==flags && r->arguments==args);
        ++receipts;
    }
    void launch_linear(int64_t,uint32_t input,const at::Tensor& weight,uint32_t output,
            int64_t,int64_t,int64_t,int,int,int,const at::Tensor&,int=-1,int=-1,const at::Tensor& = {}) {
        calls.push_back({weight.id<100?6:weight.id<200?7:5,input,0,output});
    }
    __PREPARE__ __MOD_PROFILE__ __PROFILE__ __ARGS__ __PRELOAD__ __LAUNCH__ __RESIDENT_COUNT__ __LAYOUT__
    void plan(int64_t physical_len=32) {
        struct {std::vector<Route> routes;} manifest;ChunkInfo chunk{physical_len,0};
        if(use_action_gateup_resident(chunk.len)) __LINEAR_MANIFEST__
        if(use_action_gateup_resident(physical_len)) __DMA_MANIFEST__
        routes=manifest.routes;
    }
    std::vector<BufferDecl> buffers() {
        std::vector<BufferDecl> decls;
        if(use_action_gateup_resident(action_len_)) __ALLOCATION__
        return decls;
    }
    void mlp(int layer_idx) {
        const auto& lw=layer_weights_[layer_idx];ChunkInfo chunk;
        const int64_t seq_len=chunk.len,h=hidden_size();
        __MLP__
    }
    std::vector<int64_t> kvinsert_cost_weight_identity()const __IDENTITY__
    void reset_weights() { __RESET__ }
};
template<class F>void reject(F f) {try{f();assert(false);}catch(const std::runtime_error&) {}}
int main() {
    Model m; m.prepare_action_gateup_resident();
    assert(packs==3 && m.action_gateup_resident_available_ && !m.use_action_gateup_resident(32));
    // Cold setup does not depend on transient forward state. The first cold
    // ten-body query sees exactly the layout and routes needed by forward.
    m.action_step_active_=m.action_loop_active_=true;m.action_num_steps_=10;
    m.plan();assert(m.routes.size()==9 && m.use_action_gateup_resident(32));
    // FP16 ACC16 reuses only the proven neutral modulation preload, never ACC32 resident GEMMs.
    {auto x=m;x.linear_acc32_=false;const int before=packs;x.prepare_action_gateup_resident();
     assert(x.action_mod_body_fp16_&&!x.action_mod_body_w8_&&x.use_action_mod_body()&&!x.action_gateup_resident_available_&&x.action_gateup_resident_.empty()&&packs==before);
     const auto identity=x.kvinsert_cost_weight_identity();const auto hash=x.subclass_layout_hash();
     {auto absent=x;absent.action_mod_body_fp16_=false;assert(!absent.use_action_mod_body()&&absent.kvinsert_cost_weight_identity()!=identity&&absent.subclass_layout_hash()!=hash);}
     for(int layer=0;layer<24;++layer)for(int family=0;family<8;++family){auto bad=x;auto&w=bad.layer_weights_[layer];at::Tensor*all[]={&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.attn_gate_w,&w.gate_w,&w.up_w,&w.down_w};all[family]->type=at::kChar;bad.prepare_action_gateup_resident();assert(!bad.action_mod_body_fp16_&&!bad.action_mod_body_w8_&&!bad.use_action_mod_body()&&packs==before);}
     x.reset_weights();assert(!x.action_mod_body_fp16_&&!x.action_mod_body_w8_);
    }
    // The production descriptor rejects negative route arguments. A preload
    // sentinel must therefore use an unsigned semantic value, not -1.
    for(const auto& r:m.routes) {
        for(auto value:r.arguments)assert(value>=0);
        assert(r.arguments.size()==16 && r.arguments[0]==3 && r.arguments[2]==3 &&
               r.arguments[10]==6291456);
    }
    for(int projection=0;projection<6;++projection) {
        const auto& r=m.routes[projection];
        assert(r.invocation==projection && r.arguments[1]==1+projection/2 &&
               r.arguments[14]==projection%2 && r.arguments[15]==projection*1048576);
    }
    for(int slot=0;slot<3;++slot) {
        const auto& r=m.routes[6+slot];
        assert(r.invocation==slot && r.arguments[1]==slot+1 &&
               r.arguments[14]==2 && r.arguments[15]==slot*2097152);
    }
    reject([&]{m.action_gateup_resident_arguments(1,-1);});
    reject([&]{m.action_gateup_resident_arguments(0,0);});
    reject([&]{m.action_gateup_resident_arguments(4,0);});
    auto buffers=m.buffers();assert(buffers.size()==1 && buffers[0].size==6291456 &&
        buffers[0].first==0 && buffers[0].last==31 && buffers[0].scope==BufferScope::LayerWide);
    const auto resident_hash=m.subclass_layout_hash();
    for(int flag=0;flag<3;++flag){auto x=m;const auto old=x.kvinsert_cost_weight_identity();
        if(flag==0)x.action_rtc_norm_shift_available_=true;else if(flag==1)x.action_sub_gated_residual_available_=true;else x.action_rtc_sub_gated_residual_available_=true;
        assert(x.subclass_layout_hash()!=resident_hash&&x.kvinsert_cost_weight_identity()!=old);
        x.reset_weights();assert(!x.action_rtc_norm_shift_available_&&!x.action_sub_gated_residual_available_&&!x.action_rtc_sub_gated_residual_available_);
    }
    auto& cache=KernelCache::instance();const int queries=cache.queries;cache.available=false;
    for(int body=0;body<10;++body) {
        m.body_iter=body;m.preload_action_gateup_resident();
        for(int layer=1;layer<=3;++layer) {
            calls.clear();m.mlp(layer);
            assert(dmas==3 && calls.size()==5);
            for(int i=0;i<5;++i)assert(calls[i].kind==i+1);
            assert(calls[0].input==0x40000000 &&
                   calls[0].weight==resident_base+(layer-1)*2097152 &&
                   calls[0].output==0x40010000 &&
                   calls[2].weight==resident_base+(layer-1)*2097152+1048576 &&
                   calls[2].output==0x40018000);
        }
    }
    assert(m.receipts==63 && packs==3 && cache.queries==queries);
    // Other layers still execute both ordinary Linears, and a later full
    // graph invocation reloads body zero once rather than trusting stale SPM.
    for(int layer:{0,4,23}) {
        calls.clear();m.mlp(layer);assert(calls[0].kind==6 && calls[2].kind==7);
    }
    m.body_iter=0;m.preload_action_gateup_resident();assert(dmas==6);
    calls.clear();m.complete=false;
    reject([&]{m.preload_action_gateup_resident();});reject([&]{m.launch_action_gateup_resident(1,0,0x40010000);});
    assert(dmas==6 && calls.empty());m.complete=true;
    m.plan();m.routes.back().arguments[10]=2097152;
    reject([&]{m.preload_action_gateup_resident();});assert(dmas==6);
    m.plan();m.routes[0].arguments.back()=1048576;
    reject([&]{m.launch_action_gateup_resident(1,0,0x40010000);});assert(calls.empty());
    m.plan();m.routes[5].selector=1;
    reject([&]{m.launch_action_gateup_resident(3,1,0x40018000);});
    m.body_iter=10;reject([&]{m.preload_action_gateup_resident();});
    assert(!m.use_action_gateup_resident(31) && !m.use_action_gateup_resident(64));
    const auto third_owner=m.action_gateup_resident_[2];m.action_gateup_resident_[2]={};
    assert(!m.use_action_gateup_resident(32));m.action_gateup_resident_[2]=third_owner;
    m.action_loop_active_=false;m.action_num_steps_=1;m.plan();
    assert(m.routes.empty() && m.buffers().empty() && m.subclass_layout_hash()!=resident_hash);
    calls.clear();m.mlp(1);assert(calls[0].kind==6 && calls[2].kind==7);
    // Missing payload and mismatched cold profiles do not pack or allocate.
    Model absent;absent.prepare_action_gateup_resident();assert(!absent.action_gateup_resident_available_ && packs==3);
    cache.available=true;
    for(int vary=0;vary<10;++vary) {
        Model other;
        if(vary==0)other.layers=18;if(vary==1)other.h=512;if(vary==2)other.intermediate=8192;
        if(vary==3)other.cores=4;if(vary==4)other.linear_acc32_=false;if(vary==5)other.action_len_=64;
        if(vary==6)other.layer_is_full_[0]=0;if(vary==7)other.layer_weights_[1].gate_w.type=1;
        if(vary==8)other.heads=16;if(vary==9)other.action_mode_=false;
        other.prepare_action_gateup_resident();assert(!other.action_gateup_resident_available_ && packs==3);
    }
    m.reset_weights();assert(!m.action_gateup_resident_available_);
    for(const auto& owner:m.action_gateup_resident_)assert(!owner.defined());
 // W8 retains six independent owners in the original six-MiB root.
 Model w8;cache.available=true;packs=0;dmas=0;
 for(int layer=0;layer<24;++layer){auto&w=w8.layer_weights_[layer];for(auto*t:{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.attn_gate_w,&w.gate_w,&w.up_w,&w.down_w})t->type=at::kChar;w.gate_ws={401+2*layer};w.up_ws={402+2*layer};w.down_ws={501+layer};w.gate_ws.rank=w.up_ws.rank=w.down_ws.rank=1;}
 w8.prepare_action_gateup_resident();assert(w8.action_gateup_resident_w8_&&w8.action_gateup_resident_available_&&packs==6&&w8.action_gateup_resident_layers()==6&&w8.action_gateup_resident_.size()==6&&!w8.use_action_gateup_resident(32));
 for(int i=0;i<6;++i)assert(w8.action_gateup_resident_[i].id==1900+i);
 w8.action_step_active_=w8.action_loop_active_=true;w8.action_num_steps_=10;w8.plan();assert(w8.routes.size()==18&&w8.buffers()[0].size==6291456&&w8.subclass_layout_hash()!=resident_hash);
 auto cost=w8.kvinsert_cost_weight_identity();const auto w8_hash=w8.subclass_layout_hash();
 {auto x=w8;++x.action_gateup_resident_.back().id;assert(x.kvinsert_cost_weight_identity()!=cost);}
 {auto x=w8;++x.layer_weights_[6].up_ws.id;assert(x.kvinsert_cost_weight_identity()!=cost);}
 {auto x=w8;x.action_gateup_resident_.back()={};assert(!x.use_action_gateup_resident(32));}
 {auto x=w8;x.action_gateup_resident_.resize(3);assert(!x.use_action_gateup_resident(32)&&x.kvinsert_cost_weight_identity()!=cost);}
 for(int i=0;i<12;++i){const auto&r=w8.routes[i];assert(r.site==QWEN35_ACTION_GATEUP_RESIDENT_W8_SITE&&r.selector==6&&r.invocation==i&&r.arguments.size()==18&&r.arguments[0]==6&&r.arguments[1]==1+i/2&&r.arguments[2]==6&&r.arguments[10]==6291456&&r.arguments[14]==i%2&&r.arguments[15]==i*524288);}
 for(int i=0;i<6;++i){const auto&r=w8.routes[i+12];assert(r.site==QWEN35_ACTION_GATEUP_PRELOAD_W8_SITE&&r.invocation==i&&r.arguments[1]==i+1&&r.arguments[14]==2&&r.arguments[15]==i*1048576);}
 // Every preload receipt, including member six, must validate before any DMA.
 for(int arg:{0,2,10,15}){auto x=w8;++x.routes.back().arguments[arg];reject([&]{x.preload_action_gateup_resident();});assert(dmas==0);}
 {auto x=w8;x.routes.pop_back();reject([&]{x.preload_action_gateup_resident();});assert(dmas==0);}
 const int cold_queries=cache.queries;cache.available=false;
 for(int body=0;body<10;++body){w8.body_iter=body;w8.preload_action_gateup_resident();for(int layer=1;layer<=6;++layer){calls.clear();w8.mlp(layer);assert(calls.size()==5&&calls[0].weight==resident_base+(layer-1)*1048576&&calls[2].weight==resident_base+(layer-1)*1048576+524288);for(int i=0;i<5;++i)assert(calls[i].kind==i+1);}}
 assert(dmas==6&&packs==6&&cache.queries==cold_queries&&scales_seen.size()==120);
 for(int i=0;i<120;++i)assert(scales_seen[i]==403+i%12);
 for(int layer:{0,7,23}){calls.clear();w8.mlp(layer);assert(calls[0].kind==6&&calls[2].kind==7);}
 // A new full invocation reloads all six owners, while bodies1..9 do not.
 w8.body_iter=0;w8.preload_action_gateup_resident();assert(dmas==12);
 calls.clear();w8.routes[11].arguments[2]=3;reject([&]{w8.launch_action_gateup_resident(6,1,0x40018000);});assert(calls.empty());w8.plan();
 w8.routes.back().arguments[0]=3;w8.routes.back().arguments[2]=3;w8.routes.back().arguments[10]=3145728;reject([&]{w8.preload_action_gateup_resident();});assert(dmas==12);w8.plan();
 w8.complete=false;reject([&]{w8.preload_action_gateup_resident();});reject([&]{w8.launch_action_gateup_resident(6,1,0x40018000);});assert(dmas==12&&calls.empty());w8.complete=true;
 reject([&]{w8.action_gateup_resident_arguments(7,0);});reject([&]{w8.launch_action_gateup_resident(7,0,0x40010000);});
 // Missing old payload clears cold ownership and preserves generic W8 calls.
 auto absent_w8=w8;absent_w8.prepare_action_gateup_resident();assert(!absent_w8.action_gateup_resident_available_&&!absent_w8.action_gateup_resident_w8_&&absent_w8.action_gateup_resident_.empty()&&packs==6);absent_w8.plan();assert(absent_w8.routes.empty()&&absent_w8.buffers().empty()&&absent_w8.subclass_layout_hash()!=w8_hash);calls.clear();absent_w8.mlp(6);assert(calls[0].kind==6&&calls[2].kind==7);
 // Reinstallation creates fresh owners; cost identity must bind owner six.
 cache.available=true;++pack_generation;w8.prepare_action_gateup_resident();assert(packs==12&&w8.action_gateup_resident_.back().id==2005&&w8.kvinsert_cost_weight_identity()!=cost);cost=w8.kvinsert_cost_weight_identity();
 {auto x=w8;x.layer_weights_.back().up_ws.rows=2048;x.prepare_action_gateup_resident();assert(!x.action_gateup_resident_available_&&!x.action_gateup_resident_w8_&&x.action_gateup_resident_.empty()&&packs==12);}
 // ACC16 uses the same six owners/arena and ordering, with its own tile and descriptor.
 {auto x=w8;x.linear_acc32_=false;x.prepare_action_gateup_resident();x.plan();assert(x.action_mod_body_w8_&&x.use_action_gateup_resident(32)&&x.buffers()[0].size==6291456&&x.subclass_layout_hash()!=w8_hash&&x.kvinsert_cost_weight_identity()!=cost);
  for(const auto&r:x.routes)assert(r.arguments.size()==19&&r.arguments[0]==6&&r.arguments[11]==64&&r.arguments.back()==16);
  auto valid=x.routes;x.routes=w8.routes;const int before_dma=dmas;reject([&]{x.preload_action_gateup_resident();});assert(dmas==before_dma);reject([&]{x.launch_action_gateup_resident(1,0,0x40010000);});x.routes=valid;
  linear_modes.clear();for(int body=0;body<10;++body){x.body_iter=body;x.preload_action_gateup_resident();for(int layer=1;layer<=6;++layer){calls.clear();x.mlp(layer);for(int i=0;i<5;++i)assert(calls[i].kind==i+1);}}
  assert(dmas==before_dma+6&&linear_modes.size()==120&&std::none_of(linear_modes.begin(),linear_modes.end(),[](bool mode){return mode;}));
  cache.acc16_available=false;x.prepare_action_gateup_resident();x.plan();assert(x.action_mod_body_w8_&&x.use_action_mod_body()&&!x.action_gateup_resident_available_&&x.action_gateup_resident_.empty()&&x.routes.empty());calls.clear();x.mlp(6);assert(calls[0].kind==6&&calls[2].kind==7);cache.acc16_available=true;
  // Every projection in every layer participates in the cold W8 proof.
  for(int layer=0;layer<24;++layer)for(int projection=0;projection<8;++projection){auto bad=x;auto&w=bad.layer_weights_[layer];at::Tensor*weights[]={&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.attn_gate_w,&w.gate_w,&w.up_w,&w.down_w};weights[projection]->type=at::kHalf;bad.prepare_action_gateup_resident();assert(!bad.action_mod_body_w8_&&!bad.use_action_mod_body());}
 }
 w8.reset_weights();assert(!w8.action_mod_body_w8_);assert(!w8.action_gateup_resident_available_&&!w8.action_gateup_resident_w8_&&w8.action_gateup_resident_.empty()&&w8.kvinsert_cost_weight_identity()!=cost);

}
'''
    for key, (declaration, signature) in methods.items():
        cpp = cpp.replace(f"__{key}__", declaration + " " + _body(source, signature))
    cpp = cpp.replace("__ENUM__", "enum class Qwen35TextMutableDmaRoute : int64_t " +
                      _body(source, "enum class Qwen35TextMutableDmaRoute") + ";")
    cpp = cpp.replace("__CONSTANTS__", "\n".join(re.findall(
        r"constexpr int64_t QWEN35_(?:TEXT_GDN_LOCAL_SLICE_SITE|ACTION_(?:DOWN_\w+|O_\w+|GATEUP_\w+|RESIDENT_\w+|KV_OWNER_\w+|Q_GATE_\w+|QKV_GATE_\w+|OWNER_CACHE_INSERT_SITE|SILU_ROUND_MUL_SITE|FUSED_NORM_SHIFT_SITE|RTC_RESIDUAL_GATE_SITE|RING_RTC_RESIDUAL_SITE|RESIDUAL_GATE_SITE)|TEXT_MLP_SILU_SITE|TEXT_GATEUP_PAIR_\w+)\s*=.*?;",
        source, re.S)))
    allocation_source = source[source.index(
        "std::vector<BufferDecl> Qwen3_5Model::declare_legacy_buffers("):]
    for key, text, predicate, indent in (
        ("LINEAR_MANIFEST", source, "if (use_action_gateup_resident(chunk.len))", 8),
        ("DMA_MANIFEST", source, "if (use_action_gateup_resident(physical_len))", 8),
        ("ALLOCATION", allocation_source, "if (use_action_gateup_resident(action_len_))", 4),
    ):
        begin = text.index("{", text.index(predicate))
        end = text.index("\n" + " " * indent + "}", begin) + indent + 2
        cpp = cpp.replace(f"__{key}__", text[begin:end])
    cpp = cpp.replace("__MLP__", mlp).replace("__RESET__", reset)
    cpp = cpp.replace("__IDENTITY__", identity).replace("__WEIGHTS__", weights)
    cpp = cpp.replace("struct Model {", "struct Model {\n" + _precision_site_methods(source), 1)
    path, binary = tmp_path / "g05-resident.cpp", tmp_path / "g05-resident"
    cpp = with_actual_linear_contracts(_with_actual_g05_site_helpers(cpp))
    path.write_text(cpp)
    compiled = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-I", str(Path(__file__).resolve().parents[1] / "src/ops"), "-I", str(Path(__file__).resolve().parents[1] / "src/core"), str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_actual_action_modulation_body_dma_offsets_layout_and_stale_routes(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    methods = {
        "PROFILE": ("bool use_action_mod_body() const", "bool Qwen3_5Model::use_action_mod_body("),
        "ARGS": ("std::vector<int64_t> action_mod_body_arguments(int64_t body) const",
                 "std::vector<int64_t> Qwen3_5Model::action_mod_body_arguments("),
        "LOAD": ("void load_adaptive_mod_row(int64_t row)", "void Qwen3_5Model::load_adaptive_mod_row("),
        "RESIDENT_COUNT": ("int64_t action_gateup_resident_layers() const", "int64_t Qwen3_5Model::action_gateup_resident_layers("),
        "LAYOUT": ("int64_t subclass_layout_hash() const", "int64_t Qwen3_5Model::subclass_layout_hash("),
    }
    allocation = source[source.index("    if (action_mode_) {\n        const int64_t modulation_rows") :]
    allocation = allocation[:allocation.index("    if (action_input_w_.defined())")]
    manifest = source[source.index("        if (use_action_mod_body()) {", source.index(
        "FmbPhysicalExecutionManifest Qwen3_5Model::physical_manifest_for_candidate(")) :]
    manifest = manifest[:manifest.index("        if ((action_descriptor_route_flags_")]
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
#define DWIDTH 2
#define NUM_CORES 8
namespace c10 { struct Half { uint16_t bits; }; }
namespace detail { int64_t layout_mix(int64_t a,int64_t b) { return a*31+b; } }
enum class FmbRouteFamily { MUTABLE_DMA };
enum class StorageClass { Temp };
__ENUM__
constexpr int64_t qwen35_text_dma_route(Qwen35TextMutableDmaRoute r) {return int64_t(r);}
__SITE__
__RESIDENT_CONSTANTS__
struct Route { int64_t site; FmbRouteFamily family; int64_t selector,flags;
               std::vector<int64_t> arguments; int64_t invocation=0; };
struct BufferDecl { const char* name; int64_t size,first,last; StorageClass storage;
                    int64_t flags; const char* alias; };
struct Tensor {bool defined() const {return true;}};
struct Dma {uint64_t base; int64_t offset,elements; uint32_t destination; int cores;};
std::vector<Dma> dmas;
void rpu_launch_ddr_broadcast_spm_dma_mutable(uint64_t* base,int64_t off,int64_t n,uint32_t dst,int cores) {
    dmas.push_back({*base,off,n,dst,cores});
}
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>&v,int owner){v.push_back(owner);}
struct Model {
 static constexpr int64_t QWEN35_TEXT_SILU_ROUND_MUL_SITE=7739768473980724746LL;
 bool text_silu_round_mul_available_=false;
 void prepare_text_silu_round_mul(){}
 template<class...T>bool use_text_silu_round_mul(T...)const{return false;}
 std::vector<int64_t>text_silu_round_mul_arguments()const{return {};}
 template<class...T>void launch_text_silu_round_mul(T...){assert(false);}

 static constexpr int64_t QWEN35_TEXT_OWNER_RMS_SITE=8722488789555919585LL;
 bool text_owner_rms_available_=false;
 struct TextDisabledWeight{int input_norm_w=0,post_norm_w=0,gate_w=0,up_w=0,down_w=0,gate_ws=0,up_ws=0,down_ws=0,q_w=0,k_w=0,v_w=0,o_w=0,attn_gate_w=0,q_ws=0,k_ws=0,v_ws=0,o_ws=0,attn_gate_ws=0,gdn_q_w=0,gdn_k_w=0,gdn_v_w=0,gdn_in_z_w=0,gdn_out_w=0,gdn_q_ws=0,gdn_k_ws=0,gdn_v_ws=0,gdn_in_z_ws=0,gdn_out_ws=0,gdn_b_bg_w=0,gdn_a_bg_w=0,gdn_b_bg_ws=0,gdn_a_bg_ws=0;};std::vector<TextDisabledWeight>layer_weights_;std::vector<uint8_t>layer_is_full_;
 void prepare_text_owner_rms(){}
 std::vector<int64_t> text_owner_rms_arguments()const{return {};}

 static constexpr int64_t QWEN35_ACTION_GATE_KV_OWNER_W8_SITE=5392460504564407201LL;
 static constexpr int64_t QWEN35_ACTION_QK_NORM_ROPE_GROUP4_SITE=696118694548312598LL;
 static constexpr int64_t QWEN35_ACTION_OWNER_CACHE_JOINT_SITE=7450172011194584301LL;
 static constexpr int64_t QWEN35_ACTION_ATTN_SIGMOID_MUL_SITE=8546526474721653363LL;
 bool action_owner_cache_joint_available_=false;
 void prepare_action_owner_cache_joint(){}
 bool use_action_owner_cache_joint(int64_t,int64_t)const{return false;}
 std::vector<int64_t> action_owner_cache_joint_abi()const{return {};}
 template<class...T>std::vector<int64_t> action_owner_cache_joint_arguments(T...)const{return {};}
 static constexpr int64_t QWEN35_ACTION_TAIL_SITE=7035949248365271364LL;
 static constexpr int64_t QWEN35_ACTION_MLP_TAIL_SITE=4368884220398897667LL;
 bool action_mlp_tail_available_=false;
 void prepare_action_mlp_tail(){}
 template<class...T>bool use_action_mlp_tail(T...)const{return false;}
 template<class...T>bool action_mlp_tail_handoff(T...){return false;}
 std::vector<int64_t>action_mlp_tail_arguments()const{return {};}
 template<class...T>void launch_action_mlp_tail(T...){assert(false);}
 static constexpr int64_t QWEN35_TEXT_MLP_TAIL_SITE=8475656447499929675LL;
 bool text_mlp_tail_available_=false;
 void prepare_text_mlp_tail(){}
 template<class...T>bool use_text_mlp_tail(T...)const{return false;}
 template<class...T>bool text_mlp_tail_handoff(T...){return false;}
 std::vector<int64_t>text_mlp_tail_arguments(int64_t=245)const{return {};}
 template<class...T>void launch_text_mlp_tail(T...){assert(false);}
 bool action_tail_available_=false;
 void prepare_action_tail(){}
 bool use_action_tail(int64_t)const{return false;}
 std::vector<int64_t>action_tail_arguments()const{return {};}
 template<class...T>void launch_action_tail(T...){assert(false);}
 bool action_attn_sigmoid_mul_available_=false;
 void prepare_action_attn_sigmoid_mul(){}
 bool use_action_attn_sigmoid_mul(int64_t)const{return false;}
 std::vector<int64_t> action_attn_sigmoid_mul_arguments()const{return {};}
 template<class...T>void launch_action_attn_sigmoid_mul(T...){assert(false);}
 static constexpr int64_t QWEN35_TEXT_GDN_OWNER_RMS_SITE=7871266237092441430LL;
 bool text_gdn_owner_rms_available_=false;
 std::vector<int64_t> text_gdn_owner_rms_arguments()const{return {};}
 bool action_nvfp4_qk_norm_rope_available_=false;
 bool use_action_nvfp4_qk_norm_rope(int64_t,int64_t)const{return false;}
 std::vector<int64_t> action_nvfp4_qk_norm_rope_arguments()const{return {};}
 void prepare_action_nvfp4_qk_norm_rope(){}
 template<class...T>void launch_action_nvfp4_qk_norm_rope(T...){assert(false);}
 bool action_gate_kv_owner_available_=false;
 bool use_action_gate_kv_owner(int64_t,int64_t)const{return false;}
 std::vector<int64_t> action_gate_kv_owner_arguments(int64_t)const{return {};}
 void prepare_action_gate_kv_owner(){}
 template<class...T>void launch_action_gate_kv_owner(T...){assert(false);}
 bool action_nvfp4_swiglu_available_=false;
 bool use_action_nvfp4_swiglu(int64_t) const{return false;}
 std::vector<int64_t> action_nvfp4_swiglu_arguments() const{return {};}
 void prepare_action_nvfp4_swiglu(){}
 template<class...T>void launch_action_nvfp4_swiglu(T...){assert(false);}

 // This legacy fixture exercises FP16/W8; mixed NVFP4 has its own actual-body test.
 struct Nvfp4Stub {bool enabled()const{return false;}std::vector<int>values;};
 Nvfp4Stub action_nvfp4_;
 // The dense-text NVFP4 owner is uninstalled in this probe (enabled() false, no tables).
 Nvfp4Stub text_nvfp4_;static constexpr int64_t kTextNvfp4TablesSite=3184997657402357068LL;
 template<class...T>std::vector<int64_t>action_gateup_pair_arguments(T...)const{return {};}
 bool use_action_nvfp4(int64_t)const{return false;}
 std::vector<int64_t>action_nvfp4_arguments(int)const{assert(false);return {};}
 static constexpr int64_t QWEN35_ACTION_NVFP4_LINEAR_SITE=1171762392203719440LL;

 bool use_g05_gdn_local_slice(int64_t)const{return false;}
 std::vector<int64_t>g05_gdn_local_slice_arguments(int64_t)const{return {};}

 bool action_rtc_norm_shift_available_=false,action_sub_gated_residual_available_=false,action_rtc_sub_gated_residual_available_=false;
 bool action_ring_rtc_residual_available_=false;
 bool use_action_ring_rtc_residual(int64_t)const{return false;}
 std::vector<int64_t>action_ring_rtc_residual_arguments()const{return {};}
 template<class...T>void launch_action_ring_rtc_residual(T...){ }
 bool action_down_partial_available_=false;
 bool use_action_down_partial(int64_t,int64_t)const{return false;}
 std::vector<int64_t>action_down_partial_arguments(bool)const{return {};}
 void prepare_action_down_partial(){}
 template<class...T>void launch_action_down_partial(T...){assert(false);}
 bool action_nvfp4_o_resident_available_=false;std::pair<int,int>action_nvfp4_o_resident_;
 bool use_action_nvfp4_o_resident(int64_t)const{return false;}
 std::vector<int64_t>action_nvfp4_o_resident_arguments(int64_t,int64_t)const{return {};}
 void prepare_action_nvfp4_o_resident(){}void preload_action_nvfp4_o_resident(){}
 template<class...T>void launch_action_nvfp4_o_resident(T...){assert(false);}
 bool action_nvfp4_down_resident_available_=false;std::pair<int,int>action_nvfp4_down_resident_;
 bool use_action_nvfp4_down_resident(int64_t,int64_t)const{return false;}
 std::vector<int64_t>action_nvfp4_down_resident_arguments(int64_t,int64_t)const{return {};}
 void prepare_action_nvfp4_down_resident(){}void preload_action_nvfp4_down_resident(){}
 template<class...T>void launch_action_nvfp4_down_resident(T...){assert(false);}
 bool action_o_resident_available_=false,action_o_resident_n128_=false;
 bool use_action_o_resident(int64_t)const{return false;}
 std::vector<int64_t>action_o_resident_arguments(int64_t,bool)const{return {};}
 bool action_mod_body_fp16_=false,action_mod_body_w8_=false;
 bool action_owner_cache_available_=false;std::vector<int64_t>action_k_owner_norm_rope_arguments()const{return {};}
 bool action_qkv_gate_available_=false;
 bool action_gateup_resident_w8_=false,action_kv_owner_w8_=false,action_q_gate_w8_=false,action_gateup_pair_nvfp4_=false,action_gateup_pair_w8_=false,text_gateup_pair_w8_=false;
    bool action_mode_=true,action_step_active_=true,action_loop_active_=true,linear_acc32_=true;
    bool has_gdn_=false,action_rtc_active_=true,complete=true;
    int64_t action_num_steps_=10,action_len_=32,layers=24,h=1024,cores=8,body_iter=0;
    uint64_t adaptive_mod_live_base_=0x100000;
    uint32_t adaptive_mod_row_offset_=0;
    Tensor adaptive_mod_ref_;
    std::vector<Route> routes;
    int receipts=0;
    Model& ctx() {return *this;}
    bool has_complete_physical_manifest() const {return complete;}
    int64_t hidden_size() const {return h;}
    int64_t num_layers() const {return layers;}
    int64_t num_cores() const {return cores;}
    bool use_action_gateup_resident(int64_t) const {return false;}
    bool use_action_silu_round_mul(int64_t) const {return false;}
    bool use_action_gateup_pair(int64_t) const {return false;}
    bool use_text_gateup_pair(int64_t) const {return false;}
    bool use_action_q_gate(int64_t) const {return false;}
 bool use_action_qkv_gate(int64_t)const{return false;}std::vector<int64_t>action_qkv_gate_arguments()const{return {};}
    bool use_action_kv_owner(int64_t) const {return false;}
    std::vector<int64_t> cold_topology_arguments() const {return {cores};}
    uint32_t addr(int,const char*) const {return 0x80000;}
    void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,
                               std::vector<int64_t> args,int64_t invocation) {
        bool found=false;
        for(const auto& r: routes) if(r.family==family && r.site==site && r.invocation==invocation) {
            TORCH_CHECK(r.selector==selector && r.flags==flags && r.arguments==args); found=true;
        }
        TORCH_CHECK(found); ++receipts;
    }
    void consume_manifest_route(FmbRouteFamily f,int64_t site,int64_t selector) {
        consume_physical_route(f,site,selector,0,{},0);
    }
    __PROFILE__
    __ARGS__
    __LOAD__
    __RESIDENT_COUNT__ __LAYOUT__
    std::vector<BufferDecl> buffers() {
        auto A=[](int64_t n){return (n+255)&~int64_t(255);};
        std::vector<BufferDecl> decls;
        __ALLOCATION__
        return decls;
    }
    void plan() {
        struct {std::vector<Route> routes;} manifest;
        auto append_route=[&](FmbRouteFamily f,int64_t site,int64_t selector) {
            manifest.routes.push_back({site,f,selector,0,{},0});};
        __MANIFEST__
        routes=manifest.routes;
    }
};
template<class F> void reject(F f) {try{f();assert(false);}catch(const std::runtime_error&) {}}
int main() {
    Model m; m.plan(); const auto body_hash=m.subclass_layout_hash();
    assert(body_hash!=0 && m.routes.size()==10);
    assert(m.buffers()[0].size==602112 && m.buffers()[0].first==1 && m.buffers()[0].last==31);
    // Ten bodies, each with all clean/noisy rows. An RTC toggle changes DMA
    // geometry, but keeps the same capacity; a new DDR tensor changes the base.
    for (bool rtc: {true,false,true}) {
        m.action_rtc_active_=rtc; m.plan(); dmas.clear(); m.receipts=0;
        m.adaptive_mod_live_base_+=0x100000;
        const int variants=rtc?2:1;
        for(int b=0;b<10;++b) {
            m.body_iter=b;
            for(int row=0;row<49;++row) {
                m.load_adaptive_mod_row(row);
                assert(m.adaptive_mod_row_offset_==row*variants*3072*2);
                assert(m.adaptive_mod_row_offset_+variants*3072*2<=m.buffers()[0].size);
            }
            assert(dmas.size()==size_t(b+1));
            const auto& d=dmas.back();
            assert(d.base==m.adaptive_mod_live_base_ && d.offset==b*49*variants*3072*2 &&
                   d.elements==49*variants*3072 && d.destination==0x80000 && d.cores==8);
        }
        assert(m.receipts==10 && m.subclass_layout_hash()==body_hash);
    }
    dmas.clear(); m.body_iter=0; m.complete=false;
    reject([&]{m.load_adaptive_mod_row(0);}); assert(dmas.empty());
    m.complete=true; m.plan(); m.routes[0].arguments[1]=1;
    reject([&]{m.load_adaptive_mod_row(0);}); assert(dmas.empty());
    m.plan(); m.body_iter=10; reject([&]{m.load_adaptive_mod_row(0);});
    m.body_iter=0; reject([&]{m.load_adaptive_mod_row(49);}); assert(dmas.empty());
    // The proven W8 ACC16 body preserves exactly one complete-table DMA per body.
    {Model x;x.linear_acc32_=false;assert(!x.use_action_mod_body());x.action_mod_body_w8_=true;x.plan();assert(x.use_action_mod_body()&&x.routes.size()==10&&x.buffers()[0].size==602112&&x.subclass_layout_hash()!=body_hash);dmas.clear();
     for(int body=0;body<10;++body){x.body_iter=body;for(int row=0;row<49;++row)x.load_adaptive_mod_row(row);assert(dmas.size()==size_t(body+1)&&dmas.back().offset==body*49*2*3072*2&&dmas.back().elements==49*2*3072);}}
    dmas.clear();
    // Single-step cold prime keeps the old small allocation and row DMA.
    m.action_loop_active_=false; m.action_num_steps_=1; m.plan();
    assert(m.subclass_layout_hash()!=body_hash && m.buffers()[0].size==12288);
    m.load_adaptive_mod_row(48);
    assert(m.adaptive_mod_row_offset_==0 && dmas.back().offset==48*2*3072*2 &&
           dmas.back().elements==2*3072);
    for(auto vary: {0,1,2,3,4,5,6}) {
        Model other;
        if(vary==0)other.h=512; if(vary==1)other.layers=18; if(vary==2)other.cores=4;
        if(vary==3)other.linear_acc32_=false; if(vary==4)other.action_len_=64;
        if(vary==5)other.has_gdn_=true; if(vary==6)other.action_step_active_=false;
        assert(!other.use_action_mod_body());
    }
}
'''
    for key, (declaration, signature) in methods.items():
        cpp = cpp.replace(f"__{key}__", declaration + " " + _body(source, signature))
    cpp = cpp.replace("__ENUM__", "enum class Qwen35TextMutableDmaRoute : int64_t " +
                      _body(source, "enum class Qwen35TextMutableDmaRoute") + ";")
    cpp = cpp.replace("__SITE__", re.search(
        r"constexpr int64_t QWEN35_TEXT_ADAPTIVE_MOD_DMA_SITE\s*=.*?;", source, re.S)[0])
    cpp = cpp.replace("__RESIDENT_CONSTANTS__", "\n".join(re.findall(
        r"constexpr int64_t QWEN35_(?:TEXT_GDN_LOCAL_SLICE_SITE|ACTION_(?:DOWN_\w+|O_\w+|GATEUP_RESIDENT_\w+|GATEUP_PAIR_\w+|GATEUP_SWIGLU_\w+|RESIDENT_\w+|KV_OWNER_\w+|Q_GATE_\w+|QKV_GATE_\w+|OWNER_CACHE_INSERT_SITE|SILU_ROUND_MUL_SITE|FUSED_NORM_SHIFT_SITE|RTC_RESIDUAL_GATE_SITE|RING_RTC_RESIDUAL_SITE|RESIDUAL_GATE_SITE)|TEXT_GATEUP_PAIR_\w+)\s*=.*?;",
        source, re.S)))
    cpp = cpp.replace("__ALLOCATION__", allocation).replace("__MANIFEST__", manifest)
    cpp = cpp.replace("struct Model {", "struct Model {\n" + _precision_site_methods(source), 1)
    path, binary = tmp_path / "g05-mod-body.cpp", tmp_path / "g05-mod-body"
    cpp = with_actual_linear_contracts(_with_actual_g05_site_helpers(cpp))
    path.write_text(cpp)
    compiled = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-I", str(Path(__file__).resolve().parents[1] / "src/ops"), "-I", str(Path(__file__).resolve().parents[1] / "src/core"), str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_actual_action_cold_schedule_matches_first_forward_and_restores_state(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    base = (root / "src/core/fused_model_base.h").read_text()
    lifecycle = "enum class FmbGraphLifecycle : int " + _body(base, "enum class FmbGraphLifecycle") + ";"
    constants = source[source.index("constexpr int64_t QWEN35_ACTION_ROUTE_IO"):
                       source.index("enum class Qwen35TextAllReduceRoute")]
    constants += re.search(r"constexpr int64_t QWEN35_TEXT_ACTION_SCHEDULE_SITE\s*=.*?;", source, re.S)[0]
    query = _body(source, "std::vector<int64_t> Qwen3_5Model::resolve_action_stage_domain(")
    manifest = source[source.index("    const int64_t action_body_iterations =", source.index(
        "FmbPhysicalExecutionManifest Qwen3_5Model::physical_manifest_for_candidate(")):]
    manifest = manifest[:manifest.index("    const int64_t full_layer_count")]
    consume = source[source.index("        const int64_t action_body_iterations =", source.index(
        "void Qwen3_5Model::build_layer_subgraph(")):]
    consume = consume[:consume.index("        const int64_t full_layer_count")]
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
namespace c10 {
template<class F> struct Exit { F f; ~Exit() { f(); } };
template<class F> Exit<F> make_scope_exit(F f) { return {f}; }
}
__LIFECYCLE__
enum class FmbRouteFamily { GRAPH_SCHEDULE };
__CONSTANTS__
struct Route { int64_t site; FmbRouteFamily family; int64_t selector, flags; std::vector<int64_t> args; };
struct Weight { bool defined() const { return true; } };
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>&v,int owner){v.push_back(owner);}
struct Model {
 static constexpr int64_t QWEN35_TEXT_SILU_ROUND_MUL_SITE=7739768473980724746LL;
 bool text_silu_round_mul_available_=false;
 void prepare_text_silu_round_mul(){}
 template<class...T>bool use_text_silu_round_mul(T...)const{return false;}
 std::vector<int64_t>text_silu_round_mul_arguments()const{return {};}
 template<class...T>void launch_text_silu_round_mul(T...){assert(false);}

 static constexpr int64_t QWEN35_TEXT_OWNER_RMS_SITE=8722488789555919585LL;
 bool text_owner_rms_available_=false;
 void prepare_text_owner_rms(){}
 std::vector<int64_t> text_owner_rms_arguments()const{return {};}

 static constexpr int64_t QWEN35_ACTION_GATE_KV_OWNER_W8_SITE=5392460504564407201LL;
 static constexpr int64_t QWEN35_ACTION_QK_NORM_ROPE_GROUP4_SITE=696118694548312598LL;
 static constexpr int64_t QWEN35_ACTION_OWNER_CACHE_JOINT_SITE=7450172011194584301LL;
 static constexpr int64_t QWEN35_ACTION_ATTN_SIGMOID_MUL_SITE=8546526474721653363LL;
 bool action_owner_cache_joint_available_=false;
 void prepare_action_owner_cache_joint(){}
 bool use_action_owner_cache_joint(int64_t,int64_t)const{return false;}
 std::vector<int64_t> action_owner_cache_joint_abi()const{return {};}
 template<class...T>std::vector<int64_t> action_owner_cache_joint_arguments(T...)const{return {};}
 static constexpr int64_t QWEN35_ACTION_TAIL_SITE=7035949248365271364LL;
 static constexpr int64_t QWEN35_ACTION_MLP_TAIL_SITE=4368884220398897667LL;
 bool action_mlp_tail_available_=false;
 void prepare_action_mlp_tail(){}
 template<class...T>bool use_action_mlp_tail(T...)const{return false;}
 template<class...T>bool action_mlp_tail_handoff(T...){return false;}
 std::vector<int64_t>action_mlp_tail_arguments()const{return {};}
 template<class...T>void launch_action_mlp_tail(T...){assert(false);}
 static constexpr int64_t QWEN35_TEXT_MLP_TAIL_SITE=8475656447499929675LL;
 bool text_mlp_tail_available_=false;
 void prepare_text_mlp_tail(){}
 template<class...T>bool use_text_mlp_tail(T...)const{return false;}
 template<class...T>bool text_mlp_tail_handoff(T...){return false;}
 std::vector<int64_t>text_mlp_tail_arguments(int64_t=245)const{return {};}
 template<class...T>void launch_text_mlp_tail(T...){assert(false);}
 bool action_tail_available_=false;
 void prepare_action_tail(){}
 bool use_action_tail(int64_t)const{return false;}
 std::vector<int64_t>action_tail_arguments()const{return {};}
 template<class...T>void launch_action_tail(T...){assert(false);}
 bool action_attn_sigmoid_mul_available_=false;
 void prepare_action_attn_sigmoid_mul(){}
 bool use_action_attn_sigmoid_mul(int64_t)const{return false;}
 std::vector<int64_t> action_attn_sigmoid_mul_arguments()const{return {};}
 template<class...T>void launch_action_attn_sigmoid_mul(T...){assert(false);}
 static constexpr int64_t QWEN35_TEXT_GDN_OWNER_RMS_SITE=7871266237092441430LL;
 bool text_gdn_owner_rms_available_=false;
 std::vector<int64_t> text_gdn_owner_rms_arguments()const{return {};}
 bool action_nvfp4_qk_norm_rope_available_=false;
 bool use_action_nvfp4_qk_norm_rope(int64_t,int64_t)const{return false;}
 std::vector<int64_t> action_nvfp4_qk_norm_rope_arguments()const{return {};}
 void prepare_action_nvfp4_qk_norm_rope(){}
 template<class...T>void launch_action_nvfp4_qk_norm_rope(T...){assert(false);}
 bool action_gate_kv_owner_available_=false;
 bool use_action_gate_kv_owner(int64_t,int64_t)const{return false;}
 std::vector<int64_t> action_gate_kv_owner_arguments(int64_t)const{return {};}
 void prepare_action_gate_kv_owner(){}
 template<class...T>void launch_action_gate_kv_owner(T...){assert(false);}
 bool action_nvfp4_swiglu_available_=false;
 bool use_action_nvfp4_swiglu(int64_t) const{return false;}
 std::vector<int64_t> action_nvfp4_swiglu_arguments() const{return {};}
 void prepare_action_nvfp4_swiglu(){}
 template<class...T>void launch_action_nvfp4_swiglu(T...){assert(false);}

 // This legacy fixture exercises FP16/W8; mixed NVFP4 has its own actual-body test.
 struct Nvfp4Stub {bool enabled()const{return false;}std::vector<int>values;};
 Nvfp4Stub action_nvfp4_;
 template<class...T>std::vector<int64_t>action_gateup_pair_arguments(T...)const{return {};}
 bool use_action_nvfp4(int64_t)const{return false;}
 std::vector<int64_t>action_nvfp4_arguments(int)const{assert(false);return {};}
 static constexpr int64_t QWEN35_ACTION_NVFP4_LINEAR_SITE=1171762392203719440LL;

 bool use_g05_gdn_local_slice(int64_t)const{return false;}
 std::vector<int64_t>g05_gdn_local_slice_arguments(int64_t)const{return {};}

 bool action_rtc_norm_shift_available_=false,action_sub_gated_residual_available_=false,action_rtc_sub_gated_residual_available_=false;
 bool action_ring_rtc_residual_available_=false;
 bool use_action_ring_rtc_residual(int64_t)const{return false;}
 std::vector<int64_t>action_ring_rtc_residual_arguments()const{return {};}
 template<class...T>void launch_action_ring_rtc_residual(T...){ }
 bool action_down_partial_available_=false;
 bool use_action_down_partial(int64_t,int64_t)const{return false;}
 std::vector<int64_t>action_down_partial_arguments(bool)const{return {};}
 void prepare_action_down_partial(){}
 template<class...T>void launch_action_down_partial(T...){assert(false);}
 bool action_nvfp4_o_resident_available_=false;std::pair<int,int>action_nvfp4_o_resident_;
 bool use_action_nvfp4_o_resident(int64_t)const{return false;}
 std::vector<int64_t>action_nvfp4_o_resident_arguments(int64_t,int64_t)const{return {};}
 void prepare_action_nvfp4_o_resident(){}void preload_action_nvfp4_o_resident(){}
 template<class...T>void launch_action_nvfp4_o_resident(T...){assert(false);}
 bool action_nvfp4_down_resident_available_=false;std::pair<int,int>action_nvfp4_down_resident_;
 bool use_action_nvfp4_down_resident(int64_t,int64_t)const{return false;}
 std::vector<int64_t>action_nvfp4_down_resident_arguments(int64_t,int64_t)const{return {};}
 void prepare_action_nvfp4_down_resident(){}void preload_action_nvfp4_down_resident(){}
 template<class...T>void launch_action_nvfp4_down_resident(T...){assert(false);}
 bool action_o_resident_available_=false,action_o_resident_n128_=false;
 bool use_action_o_resident(int64_t)const{return false;}
 std::vector<int64_t>action_o_resident_arguments(int64_t,bool)const{return {};}
 bool action_mod_body_fp16_=false,action_mod_body_w8_=false;
 bool action_qkv_gate_available_=false;
 bool action_gateup_resident_w8_=false,action_kv_owner_w8_=false,action_q_gate_w8_=false,action_gateup_pair_nvfp4_=false,action_gateup_pair_w8_=false,text_gateup_pair_w8_=false;
    bool action_mode_=true, action_descriptor_include_io_=false;
    bool action_step_active_=false, action_loop_active_=false, action_rtc_active_=false,
         action_trace_active_=false;
    int64_t action_len_=32, action_dim_pad_=32, action_descriptor_route_flags_=0,
            action_num_steps_=1;
    FmbGraphLifecycle action_descriptor_graph_lifecycle_=FmbGraphLifecycle::COMPOSITE_CHILD;
    Weight action_input_w_;
    bool fail=false;
    std::vector<int64_t> actual;
    Model& ctx() { return *this; }
    void consume_physical_route(FmbRouteFamily, int64_t, int64_t, int64_t,
            std::vector<int64_t> args, int64_t) { actual=std::move(args); }
    std::vector<int64_t> forward_schedule() {
        struct { int64_t idx=0; } chunk;
        __CONSUME__
        return actual;
    }
    std::vector<int64_t> resolve_prefill_stage_domain_for_shape(
            int64_t, int64_t, std::nullopt_t, bool, int64_t, int64_t) {
        if (fail) throw std::runtime_error("SPM rejection");
        struct { std::vector<Route> routes; } manifest;
        __MANIFEST__
        return manifest.routes[0].args;
    }
    std::vector<int64_t> encode_fmb_prefill_stage_domain(std::vector<int64_t> v) { return v; }
    std::vector<int64_t> query(int64_t requested_action_len, int64_t max_prefix_len,
            int64_t requested_chunk_size, bool include_action_io, int64_t action_route_flags,
            int64_t graph_lifecycle, int64_t num_steps) __QUERY__
    auto state() const { return std::make_tuple(action_step_active_, action_loop_active_,
        action_num_steps_, action_rtc_active_, action_trace_active_, action_descriptor_include_io_,
        action_descriptor_route_flags_, action_descriptor_graph_lifecycle_); }
};
int main() {
    Model m;
    const int64_t child=static_cast<int64_t>(FmbGraphLifecycle::COMPOSITE_CHILD);
    // First cold call is the actual 10-step IO route, not the previous state.
    auto old=m.state();
    auto first=m.query(32,245,0,true,1,child,10);
    assert(m.state()==old);
    m.action_step_active_=true; m.action_loop_active_=true; m.action_num_steps_=10;
    assert(first==m.forward_schedule());
    // Queries are independent of prior RTC/trace/loop use, including throws.
    m.action_rtc_active_=m.action_trace_active_=true;
    old=m.state();
    auto hidden=m.query(32,245,0,false,0,2,1);
    auto step=m.query(32,245,0,true,1,2,1);
    auto rtc=m.query(32,245,0,true,3,child,10);
    assert(m.state()==old && first!=step && step!=hidden && rtc!=first);
    assert((hidden==std::vector<int64_t>{1,1,0,0,0,1,32,32}));
    assert((step==std::vector<int64_t>{1,1,0,0,1,1,32,32}));
    assert((rtc==std::vector<int64_t>{1,1,1,1,1,10,32,32}));
    for (auto bad : std::vector<std::vector<int64_t>>{{0,0,child,10}, {1,3,child,1}, {1,1,2,10}, {1,1,child,2}}) {
        try { m.query(32,245,0,bad[0],bad[1],bad[2],bad[3]); assert(false); }
        catch (const std::runtime_error&) {}
        assert(m.state()==old);
    }
    m.fail=true;
    try { m.query(32,245,0,false,0,2,1); assert(false); } catch (const std::runtime_error&) {}
    assert(m.state()==old);
}
'''
    cpp = (cpp.replace("__QUERY__", query).replace("__MANIFEST__", manifest).replace("__CONSUME__", consume)
           .replace("__LIFECYCLE__", lifecycle).replace("__CONSTANTS__", constants))
    path, binary = tmp_path / "g05-cold.cpp", tmp_path / "g05-cold"
    cpp = with_actual_linear_contracts(_with_actual_g05_site_helpers(cpp))
    path.write_text(cpp)
    compiled = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-I", str(Path(__file__).resolve().parents[1] / "src/ops"), "-I", str(Path(__file__).resolve().parents[1] / "src/core"), str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_actual_action_norm_shift_cold_route_and_rtc_segments(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    cache_header = (root / "src/core/rpu_kernel_cache.h").read_text()
    cache_source = (root / "src/core/rpu_kernel_cache.inc").read_text()
    kernel_ids = "enum class KernelId : uint16_t " + _body(cache_header, "enum class KernelId") + ";"
    availability = "class KernelPayloadAvailability final " + _body(
        cache_header, "class KernelPayloadAvailability final") + ";"
    publication = re.search(r"payload_availability_\.publish\(.*?;", cache_source, re.S)[0]
    enable = _body(source, "void Qwen3_5Model::enable_action_mode()")
    topology = (root / "src/fused/qwen3_5_execution_topology.h").read_text()
    action_row = ("inline int g05_action_profile_num_layers(int64_t q_heads, int64_t effective_kv_heads,"
                  " int64_t head_dim, int64_t hidden, int64_t intermediate)" +
                  _body(topology, "inline int g05_action_profile_num_layers("))
    route = _body(source, "int64_t Qwen3_5Model::action_norm_route(")
    apply = _body(source, "void Qwen3_5Model::apply_adaptive_norm(")
    norm_args = _body(source, "std::vector<int64_t> Qwen3_5Model::action_norm_arguments(")
    residual_route = _body(source, "int64_t Qwen3_5Model::action_residual_route(")
    residual_apply = _body(source, "void Qwen3_5Model::apply_adaptive_residual_gate(")
    raw_residual = _body(source, "void Qwen3_5Model::apply_raw_residual_gate(")
    manifest_source = source[source.index(
        "FmbPhysicalExecutionManifest Qwen3_5Model::physical_manifest_for_candidate("):]
    begin = manifest_source.index("        if (action_mode_) {\n            const int64_t norm_route")
    manifest = manifest_source[begin:manifest_source.index("        if (use_action_nvfp4(chunk.len))", begin)]
    constants = "\n".join(re.findall(
        r"constexpr int64_t QWEN35_ACTION_(?:(?:FUSED|RMS)_NORM_SHIFT|(?:RTC_)?RESIDUAL_GATE|RING_RTC_RESIDUAL)_SITE\s*=.*?;",
        source, re.S))
    cpp = r'''
#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
#define DWIDTH 2
#define NUM_CORES 8
namespace c10 { struct Half { explicit Half(double) {} }; }
__KERNEL_IDS__
__AVAILABILITY__
struct KernelCache {
    KernelPayloadAvailability payload_availability_;
    void publish(bool available, bool gated=true, bool rtc=false, bool sub=false, bool rtc_sub=false, bool ring=false) {
        void* fast_cache_[static_cast<size_t>(KernelId::_COUNT)]{};
        fast_cache_[static_cast<size_t>(KernelId::PI05_ADARMS_NORM_SHIFT_H1024)] = available ? this : nullptr;
        fast_cache_[static_cast<size_t>(KernelId::PI05_GATED_RESIDUAL_H1024)] = gated ? this : nullptr;
        KernelPayloadAvailability::ExtendedPayloads pi05_extended_payloads;
        if(rtc)pi05_extended_payloads.set(static_cast<size_t>(KernelId::G05_RTC_ADARMS_NORM_SHIFT_M32H1024)-KernelPayloadAvailability::kExtendedBegin);
        if(sub)pi05_extended_payloads.set(static_cast<size_t>(KernelId::G05_SUB_GATED_RESIDUAL_H1024)-KernelPayloadAvailability::kExtendedBegin);
        if(rtc_sub)pi05_extended_payloads.set(static_cast<size_t>(KernelId::G05_RTC_SUB_GATED_RESIDUAL_M32H1024)-KernelPayloadAvailability::kExtendedBegin);
        if(ring)pi05_extended_payloads.set(static_cast<size_t>(KernelId::G05_RING_RTC_SUB_GATED_RESIDUAL_M32H1024)-KernelPayloadAvailability::kExtendedBegin);
        __PUBLICATION__
    }
    static KernelCache& instance() { static KernelCache k; return k; }
    bool has_loaded(KernelId id) const { return payload_availability_.has_loaded(id); }
};
enum class FmbRouteFamily { NORMALIZATION, GRAPH_SCHEDULE };
enum class ValuOpType { ADD, SUB, MUL };
__CONSTANTS__
struct ChunkInfo { int64_t len=32, offset=0, idx=0; };
struct Route {
    int64_t site; FmbRouteFamily family; int64_t selector, flags;
    std::vector<int64_t> args; int64_t invocation;
};
struct Call { int kind; uint32_t input, output, scale, shift; int64_t rows, hidden, cores; uint32_t noisy_scale=0,noisy_shift=0;int64_t clean_rows=-1; };
std::vector<Call> calls;
void rpu_launch_pi05_adarms_norm_shift_spm_kernel(uint32_t input, uint32_t output,
        uint32_t scale, uint32_t shift, int64_t rows, double, int cores) {
    calls.push_back({2,input,output,scale,shift,rows,1024,cores});
}
void rpu_launch_rmsnorm_spm_kernel(uint32_t input, uint32_t output,
        uint32_t scale, int64_t rows, int64_t hidden, double) {
    calls.push_back({1,input,output,scale,0,rows,hidden,8});
}
void rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(uint32_t shift, uint32_t input,
        uint32_t output, int64_t rows, int64_t hidden, c10::Half, ValuOpType op, bool,
        int cores) {
    calls.push_back({op==ValuOpType::MUL ? 4 : 0,input,output,0,shift,rows,hidden,cores});
}
void rpu_launch_eltwise_binary_spm_kernel(uint32_t input, uint32_t residual,
        uint32_t output, int64_t elems, ValuOpType op, c10::Half, int cores) {
    calls.push_back({op==ValuOpType::SUB ? 3 : 5,input,output,0,residual,elems/1024,1024,cores});
}
void rpu_launch_pi05_gated_residual_spm_kernel(uint32_t input, uint32_t residual,
        uint32_t gate, uint32_t output, int64_t rows, int cores) {
    calls.push_back({6,input,output,gate,residual,rows,1024,cores});
}
void rpu_launch_g05_rtc_adarms_norm_shift_spm_kernel(uint32_t input,uint32_t output,
        uint32_t scale,uint32_t shift,uint32_t noisy_scale,uint32_t noisy_shift,
        int64_t clean_rows,double,int cores){
    calls.push_back({7,input,output,scale,shift,32,1024,cores,noisy_scale,noisy_shift,clean_rows});
}
void rpu_launch_g05_sub_gated_residual_spm_kernel(uint32_t input,uint32_t residual,
        uint32_t gate,uint32_t output,int64_t rows,int cores){
    calls.push_back({8,input,output,gate,residual,rows,1024,cores});
}
void rpu_launch_g05_rtc_sub_gated_residual_spm_kernel(uint32_t input,uint32_t residual,
        uint32_t clean,uint32_t noisy,uint32_t output,int64_t clean_rows,int cores){
    calls.push_back({9,input,output,clean,residual,32,1024,cores,noisy,0,clean_rows});
}
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>&v,int owner){v.push_back(owner);}
__ACTION_ROW__
struct Model {
 static constexpr int64_t QWEN35_TEXT_SILU_ROUND_MUL_SITE=7739768473980724746LL;
 bool text_silu_round_mul_available_=false;
 void prepare_text_silu_round_mul(){}
 template<class...T>bool use_text_silu_round_mul(T...)const{return false;}
 std::vector<int64_t>text_silu_round_mul_arguments()const{return {};}
 template<class...T>void launch_text_silu_round_mul(T...){assert(false);}

 static constexpr int64_t QWEN35_TEXT_OWNER_RMS_SITE=8722488789555919585LL;
 bool text_owner_rms_available_=false;
 void prepare_text_owner_rms(){}
 std::vector<int64_t> text_owner_rms_arguments()const{return {};}

 static constexpr int64_t QWEN35_ACTION_GATE_KV_OWNER_W8_SITE=5392460504564407201LL;
 static constexpr int64_t QWEN35_ACTION_QK_NORM_ROPE_GROUP4_SITE=696118694548312598LL;
 static constexpr int64_t QWEN35_ACTION_OWNER_CACHE_JOINT_SITE=7450172011194584301LL;
 static constexpr int64_t QWEN35_ACTION_ATTN_SIGMOID_MUL_SITE=8546526474721653363LL;
 bool action_owner_cache_joint_available_=false;
 void prepare_action_owner_cache_joint(){}
 bool use_action_owner_cache_joint(int64_t,int64_t)const{return false;}
 std::vector<int64_t> action_owner_cache_joint_abi()const{return {};}
 template<class...T>std::vector<int64_t> action_owner_cache_joint_arguments(T...)const{return {};}
 static constexpr int64_t QWEN35_ACTION_TAIL_SITE=7035949248365271364LL;
 static constexpr int64_t QWEN35_ACTION_MLP_TAIL_SITE=4368884220398897667LL;
 bool action_mlp_tail_available_=false;
 void prepare_action_mlp_tail(){}
 template<class...T>bool use_action_mlp_tail(T...)const{return false;}
 template<class...T>bool action_mlp_tail_handoff(T...){return false;}
 std::vector<int64_t>action_mlp_tail_arguments()const{return {};}
 template<class...T>void launch_action_mlp_tail(T...){assert(false);}
 static constexpr int64_t QWEN35_TEXT_MLP_TAIL_SITE=8475656447499929675LL;
 bool text_mlp_tail_available_=false;
 void prepare_text_mlp_tail(){}
 template<class...T>bool use_text_mlp_tail(T...)const{return false;}
 template<class...T>bool text_mlp_tail_handoff(T...){return false;}
 std::vector<int64_t>text_mlp_tail_arguments(int64_t=245)const{return {};}
 template<class...T>void launch_text_mlp_tail(T...){assert(false);}
 bool action_tail_available_=false;
 void prepare_action_tail(){}
 bool use_action_tail(int64_t)const{return false;}
 std::vector<int64_t>action_tail_arguments()const{return {};}
 template<class...T>void launch_action_tail(T...){assert(false);}
 bool action_attn_sigmoid_mul_available_=false;
 void prepare_action_attn_sigmoid_mul(){}
 bool use_action_attn_sigmoid_mul(int64_t)const{return false;}
 std::vector<int64_t> action_attn_sigmoid_mul_arguments()const{return {};}
 template<class...T>void launch_action_attn_sigmoid_mul(T...){assert(false);}
 static constexpr int64_t QWEN35_TEXT_GDN_OWNER_RMS_SITE=7871266237092441430LL;
 bool text_gdn_owner_rms_available_=false;
 std::vector<int64_t> text_gdn_owner_rms_arguments()const{return {};}
 bool action_nvfp4_qk_norm_rope_available_=false;
 bool use_action_nvfp4_qk_norm_rope(int64_t,int64_t)const{return false;}
 std::vector<int64_t> action_nvfp4_qk_norm_rope_arguments()const{return {};}
 void prepare_action_nvfp4_qk_norm_rope(){}
 template<class...T>void launch_action_nvfp4_qk_norm_rope(T...){assert(false);}
 bool action_gate_kv_owner_available_=false;
 bool use_action_gate_kv_owner(int64_t,int64_t)const{return false;}
 std::vector<int64_t> action_gate_kv_owner_arguments(int64_t)const{return {};}
 void prepare_action_gate_kv_owner(){}
 template<class...T>void launch_action_gate_kv_owner(T...){assert(false);}
 bool action_nvfp4_swiglu_available_=false;
 bool use_action_nvfp4_swiglu(int64_t) const{return false;}
 std::vector<int64_t> action_nvfp4_swiglu_arguments() const{return {};}
 void prepare_action_nvfp4_swiglu(){}
 template<class...T>void launch_action_nvfp4_swiglu(T...){assert(false);}

 // This legacy fixture exercises FP16/W8; mixed NVFP4 has its own actual-body test.
 struct Nvfp4Stub {bool enabled()const{return false;}std::vector<int>values;};
 Nvfp4Stub action_nvfp4_;
 template<class...T>std::vector<int64_t>action_gateup_pair_arguments(T...)const{return {};}
 bool use_action_nvfp4(int64_t)const{return false;}
 std::vector<int64_t>action_nvfp4_arguments(int)const{assert(false);return {};}
 static constexpr int64_t QWEN35_ACTION_NVFP4_LINEAR_SITE=1171762392203719440LL;

 bool use_g05_gdn_local_slice(int64_t)const{return false;}
 std::vector<int64_t>g05_gdn_local_slice_arguments(int64_t)const{return {};}

 bool action_rtc_norm_shift_available_=false,action_sub_gated_residual_available_=false,action_rtc_sub_gated_residual_available_=false;
 bool action_ring_rtc_residual_available_=false;
 bool use_action_ring_rtc_residual(int64_t)const{return false;}
 std::vector<int64_t>action_ring_rtc_residual_arguments()const{return {};}
 template<class...T>void launch_action_ring_rtc_residual(T...){ }
 bool action_down_partial_available_=false;
 bool use_action_down_partial(int64_t,int64_t)const{return false;}
 std::vector<int64_t>action_down_partial_arguments(bool)const{return {};}
 void prepare_action_down_partial(){}
 template<class...T>void launch_action_down_partial(T...){assert(false);}
 bool action_nvfp4_o_resident_available_=false;std::pair<int,int>action_nvfp4_o_resident_;
 bool use_action_nvfp4_o_resident(int64_t)const{return false;}
 std::vector<int64_t>action_nvfp4_o_resident_arguments(int64_t,int64_t)const{return {};}
 void prepare_action_nvfp4_o_resident(){}void preload_action_nvfp4_o_resident(){}
 template<class...T>void launch_action_nvfp4_o_resident(T...){assert(false);}
 bool action_nvfp4_down_resident_available_=false;std::pair<int,int>action_nvfp4_down_resident_;
 bool use_action_nvfp4_down_resident(int64_t,int64_t)const{return false;}
 std::vector<int64_t>action_nvfp4_down_resident_arguments(int64_t,int64_t)const{return {};}
 void prepare_action_nvfp4_down_resident(){}void preload_action_nvfp4_down_resident(){}
 template<class...T>void launch_action_nvfp4_down_resident(T...){assert(false);}
 bool action_o_resident_available_=false,action_o_resident_n128_=false;
 bool use_action_o_resident(int64_t)const{return false;}
 std::vector<int64_t>action_o_resident_arguments(int64_t,bool)const{return {};}
 bool action_mod_body_fp16_=false,action_mod_body_w8_=false;
 bool action_qkv_gate_available_=false;
 bool action_gateup_resident_w8_=false,action_kv_owner_w8_=false,action_q_gate_w8_=false,action_gateup_pair_nvfp4_=false,action_gateup_pair_w8_=false,text_gateup_pair_w8_=false;
    bool action_mode_=false, action_norm_shift_available_=false, has_gdn_=false;
    bool action_gated_residual_available_=false;
    bool action_rtc_active_=false, complete=true;
    uint32_t adaptive_mod_row_offset_=0;
    int64_t action_rtc_prefix_len_=0, h=1024, cores=8, resolved=0, layers=24;
    double eps_=1e-6;
    std::vector<int64_t> action_prefix_lens_;
    int invalidations=0, receipts=0;
    Route expected{}, expected_gate{};
    int64_t hidden_size() const { return h; }
    int64_t num_cores() const { return cores; }
    int64_t num_layers() const { return layers; }
    int64_t attn_tp() const { return cores; }
    int64_t mlp_tp() const { return cores; }
    int64_t num_q_heads() const { return 8; }
    int64_t num_kv_heads() const { return cores == 8 ? 8 : 4; }
    int64_t head_dim() const { return 256; }
    int64_t intermediate_size() const { return 4096; }
    int64_t get_last_resolved_chunk_size() const { return resolved; }
    void set_chunk_size_override(int64_t) {}
    void invalidate_model_state() { ++invalidations; }
    Model& ctx() { return *this; }
    bool has_complete_physical_manifest() const { return complete; }
    uint32_t addr(int, const char*) const { return 0x80000; }
    void consume_physical_route(FmbRouteFamily family, int64_t site,
            int64_t selector, int64_t flags, std::vector<int64_t> args, int64_t invocation) {
        const auto& want=family==FmbRouteFamily::NORMALIZATION ? expected : expected_gate;
        TORCH_CHECK(family==want.family && site==want.site && selector==want.selector &&
                    flags==want.flags && args==want.args && invocation==want.invocation);
        ++receipts;
    }
    void enable_action_mode() __ENABLE__
    int64_t action_norm_route(int64_t rows) const __ROUTE__
    std::vector<int64_t> action_norm_arguments(int64_t rows) const __NORM_ARGS__
    int64_t action_residual_route(int64_t rows) const __RESIDUAL_ROUTE__
    void apply_adaptive_norm(uint32_t input, uint32_t output, const ChunkInfo& chunk) __APPLY__
    void apply_adaptive_residual_gate(uint32_t output, uint32_t residual, const ChunkInfo& chunk) __RESIDUAL_APPLY__
    void apply_raw_residual_gate(uint32_t output, uint32_t residual, uint32_t gate,
                                 int64_t seq_len, int64_t route) __RAW_RESIDUAL__
    void plan(const ChunkInfo& chunk) {
        const int plan=0;const auto physical_len=chunk.len,logical_len=chunk.len;const int64_t position=0;struct{int64_t batch_size;}layout{1};
        struct { std::vector<Route> routes; } manifest;
        __MANIFEST__
        assert(manifest.routes.size()==2);
        expected=manifest.routes.front();
        expected_gate=manifest.routes.back();
    }
    void run(const ChunkInfo& chunk, uint32_t input=0x10000, uint32_t output=0x30000) {
        plan(chunk); calls.clear(); receipts=0; apply_adaptive_norm(input,output,chunk);
    }
    void run_gate(const ChunkInfo& chunk) {
        plan(chunk); calls.clear(); receipts=0; apply_adaptive_residual_gate(0x30000,0x10000,chunk);
    }
};
template<class F> void reject(F f) {
    try { f(); assert(false); } catch (const std::runtime_error&) {}
}
int main() {
    ChunkInfo chunk;
    auto& payload=KernelCache::instance();
    reject([&]{payload.has_loaded(KernelId::PI05_ADARMS_NORM_SHIFT_H1024);});
    payload.publish(true);
    Model m; m.enable_action_mode();
    assert(m.invalidations==1 && m.action_prefix_lens_.size()==24);
    m.run(chunk);
    assert(m.expected.site==QWEN35_ACTION_FUSED_NORM_SHIFT_SITE && m.receipts==1);
    assert(calls.size()==1 && calls[0].kind==2 && calls[0].rows==32 && calls[0].cores==8);
    assert(calls[0].input==0x10000 && calls[0].output==0x30000 &&
           calls[0].scale==0x80000 && calls[0].shift==0x80800);
    // The payload is sampled at installation, never reconsidered during planning or REPLAY.
    payload.publish(false); assert(m.action_norm_route(32)==2);
    Model absent; absent.enable_action_mode();
    // Reduced action mode admits only the exact four-lane G05 expert.
    {Model four; four.cores=4; four.enable_action_mode(); assert(four.action_mode_);
     Model six; six.cores=6; reject([&]{six.enable_action_mode();}); assert(!six.action_mode_);
     Model wide; wide.cores=4; wide.h=2048; reject([&]{wide.enable_action_mode();}); assert(!wide.action_mode_);}
    payload.publish(true); absent.run(chunk);
    assert(absent.expected.site==QWEN35_ACTION_RMS_NORM_SHIFT_SITE && absent.receipts==1);
    assert(calls.size()==2 && calls[0].kind==1 && calls[1].kind==0);
    // RTC clean/noisy pieces retain their row and modulation offsets, including in-place final norm.
    m.action_rtc_active_=true; m.action_rtc_prefix_len_=5;
    m.run(chunk,0x10000,0x10000);
    assert(m.receipts==2 && calls.size()==2 && calls[0].rows==5 && calls[1].rows==27);
    assert(calls[1].input==0x12800 && calls[1].output==0x12800 &&
           calls[1].scale==0x81800 && calls[1].shift==0x82000);
    for (auto prefix : {0,32}) {
        m.action_rtc_prefix_len_=prefix; m.run(chunk);
        assert(calls.size()==1 && calls[0].rows==32 && m.receipts==1);
        assert(calls[0].scale==(prefix ? 0x80000U : 0x81800U));
    }
    chunk.offset=8; m.action_rtc_prefix_len_=13; m.run(chunk);
    assert(calls[0].rows==5 && calls[1].rows==27);
    // Unsupported ABI or numerical parameters preserve the two-operation path.
    m.action_rtc_active_=false; chunk.offset=0;
    m.h=512; m.run(chunk); assert(calls.size()==2 && calls[0].hidden==512);
    m.h=1024; chunk.len=65; m.run(chunk); assert(calls.size()==2 && calls[0].rows==65);
    chunk.len=32; m.cores=4; assert(m.action_norm_route(32)==1); m.cores=8;
    for (auto eps : {0.,2.,double(NAN)}) { m.eps_=eps; assert(m.action_norm_route(32)==1); }
    m.eps_=1e-6; m.action_mode_=false; assert(m.action_norm_route(32)==1); m.action_mode_=true;
    // A missing/stale descriptor must reject before dispatching any kernel.
    calls.clear(); m.complete=false;
    reject([&]{m.apply_adaptive_norm(0x10000,0x30000,chunk);}); assert(calls.empty());
    m.complete=true; m.plan(chunk); m.expected.selector=1;
    reject([&]{m.apply_adaptive_norm(0x10000,0x30000,chunk);}); assert(calls.empty());
    // Gated residual preserves SUB and consumes its descriptor before any arithmetic.
    m.run_gate(chunk);
    assert(m.expected_gate.site==QWEN35_ACTION_RESIDUAL_GATE_SITE && m.receipts==1);
    assert(calls.size()==2 && calls[0].kind==3 && calls[1].kind==6);
    assert(calls[1].input==0x30000 && calls[1].output==0x30000 &&
           calls[1].scale==0x81000 && calls[1].shift==0x10000 && calls[1].rows==32);
    m.action_rtc_active_=true; m.action_rtc_prefix_len_=5; m.run_gate(chunk);
    assert(m.receipts==2 && calls.size()==4 && calls[1].rows==5 && calls[3].rows==27);
    assert(calls[3].input==0x32800 && calls[3].output==0x32800 &&
           calls[3].scale==0x82800 && calls[3].shift==0x12800);
    for (auto prefix : {0,32}) {
        m.action_rtc_prefix_len_=prefix; m.run_gate(chunk);
        assert(calls.size()==2 && calls[1].rows==32);
        assert(calls[1].scale==(prefix ? 0x81000U : 0x82800U));
    }
    chunk.offset=8; m.action_rtc_prefix_len_=13; m.run_gate(chunk);
    assert(calls[1].rows==5 && calls[3].rows==27);
    // A later body-table row must reach both consumers, including RTC's
    // second triple. This catches accidentally reading row zero for all layers.
    for (uint32_t offset : {12288U, 48U*12288U}) {
        m.adaptive_mod_row_offset_=offset;
        m.run(chunk);
        assert(calls[0].scale==0x80000U+offset && calls[1].scale==0x81800U+offset);
        m.run_gate(chunk);
        assert(calls[1].scale==0x81000U+offset && calls[3].scale==0x82800U+offset);
    }
    m.adaptive_mod_row_offset_=0;
    chunk.offset=0; m.action_rtc_active_=false;
    payload.publish(true,false); assert(m.action_residual_route(32)==2);
    Model no_gate; no_gate.enable_action_mode(); payload.publish(true,true);
    no_gate.run_gate(chunk);
    assert(no_gate.expected_gate.selector==1 && calls.size()==3 &&
           calls[0].kind==3 && calls[1].kind==4 && calls[2].kind==5);
    m.h=512; assert(m.action_residual_route(32)==1); m.h=1024;
    m.cores=4; assert(m.action_residual_route(32)==1); m.cores=8;
    assert(m.action_residual_route(0)==1 && m.action_residual_route(65)==1);
    m.action_mode_=false; assert(m.action_residual_route(32)==1); m.action_mode_=true;
    calls.clear(); m.complete=false;
    reject([&]{m.apply_adaptive_residual_gate(0x30000,0x10000,chunk);}); assert(calls.empty());
    m.complete=true; m.plan(chunk); m.expected_gate.selector=1;
    reject([&]{m.apply_adaptive_residual_gate(0x30000,0x10000,chunk);}); assert(calls.empty());

    // Each optional capability is independently frozen during cold install.
    // New kernels do not depend on the old Pi payloads being present.
    for(bool rtc_payload:{false,true})for(bool sub_payload:{false,true}){
        payload.publish(false,false,rtc_payload,sub_payload);
        Model fresh;fresh.enable_action_mode();fresh.action_rtc_active_=true;
        fresh.action_rtc_prefix_len_=5;payload.publish(false,false,false,false);
        fresh.run(chunk);assert(fresh.expected.selector==(rtc_payload?3:1));
        assert(calls.size()==(rtc_payload?1U:4U));
        fresh.run_gate(chunk);assert(fresh.expected_gate.selector==(sub_payload?3:1));
        assert(calls.size()==(sub_payload?2U:6U));
    }
    payload.publish(true,true,true,true);Model fused;fused.enable_action_mode();
    fused.action_rtc_active_=true;
    const std::vector<int64_t> norm_args={32,1024,8,1,2,6144,0,32};
    // Real dry manifest never binds a previous runtime prefix. The full stage
    // query binds RTC flags but intentionally has no prefix-length argument.
    for(int stale:{-100,0,5,32,100}){
        fused.action_rtc_prefix_len_=stale;fused.plan(chunk);
        assert(fused.expected.selector==3&&fused.expected.site==QWEN35_ACTION_FUSED_NORM_SHIFT_SITE&&fused.expected.args==norm_args);
    }
    for(int prefix:{0,5,32})for(uint32_t offset:{0U,12288U,48U*12288U}){
        fused.action_rtc_prefix_len_=prefix;fused.adaptive_mod_row_offset_=offset;
        fused.run(chunk,0x10000,0x10000);
        assert(fused.receipts==1&&calls.size()==1);
        const auto&c=calls[0];assert(c.kind==7&&c.input==0x10000&&c.output==c.input&&c.rows==32&&c.cores==8&&c.clean_rows==prefix);
        assert(c.scale==0x80000+offset&&c.shift==0x80800+offset&&c.noisy_scale==0x81800+offset&&c.noisy_shift==0x82000+offset);
        fused.run_gate(chunk);assert(fused.expected_gate.selector==3);
        assert(calls.size()==(prefix==5?2U:1U)&&fused.receipts==int(calls.size()));
        assert(calls[0].kind==8&&calls[0].rows==(prefix==5?5:32));
        assert(calls[0].scale==(prefix==0?0x82800U:0x81000U)+offset);
        if(prefix==5){assert(calls[1].kind==8&&calls[1].rows==27&&calls[1].input==0x32800&&calls[1].shift==0x12800&&calls[1].scale==0x82800+offset);}
    }
    fused.adaptive_mod_row_offset_=0;fused.action_rtc_prefix_len_=13;chunk.offset=8;
    fused.run(chunk);assert(calls.size()==1&&calls[0].clean_rows==5);
    chunk.offset=0;fused.action_rtc_prefix_len_=5;
    for(size_t field=0;field<norm_args.size();++field){fused.plan(chunk);++fused.expected.args[field];calls.clear();reject([&]{fused.apply_adaptive_norm(0x10000,0x30000,chunk);});assert(calls.empty());}
    for(int fault=0;fault<5;++fault){
        fused.plan(chunk);auto&r=fused.expected;
        if(fault==0)r.selector=2;if(fault==1)r.flags=1;if(fault==2)++r.invocation;
        if(fault==3)r.family=FmbRouteFamily::GRAPH_SCHEDULE;if(fault==4)++r.site;
        calls.clear();reject([&]{fused.apply_adaptive_norm(0x10000,0x30000,chunk);});assert(calls.empty());
    }
    fused.plan(chunk);fused.expected_gate.selector=2;calls.clear();
    reject([&]{fused.apply_adaptive_residual_gate(0x30000,0x10000,chunk);});assert(calls.empty());
    fused.complete=false;reject([&]{fused.apply_adaptive_norm(0x10000,0x30000,chunk);});assert(calls.empty());fused.complete=true;
    fused.action_rtc_active_=false;fused.run(chunk);assert(fused.expected.selector==2&&calls.size()==1&&calls[0].kind==2);
    fused.run_gate(chunk);assert(calls.size()==1&&calls[0].kind==8&&calls[0].rows==32);
    fused.action_rtc_active_=true;chunk.len=31;fused.run(chunk);assert(fused.expected.selector==2&&calls.size()==2);
    chunk.len=32;fused.h=512;assert(fused.action_norm_route(32)==1&&fused.action_residual_route(32)==1);fused.h=1024;
    fused.cores=4;assert(fused.action_norm_route(32)==1&&fused.action_residual_route(32)==1);fused.cores=8;
    for(auto eps:{0.,2.,double(NAN)}){fused.eps_=eps;assert(fused.action_norm_route(32)==1);}fused.eps_=1e-6;

    payload.publish(false,false,false,false,true);Model rtc_gate;rtc_gate.enable_action_mode();
    payload.publish(false,false,false,false,false); // Hot route uses the frozen optional capability.
    rtc_gate.action_rtc_active_=true;chunk.len=32;chunk.offset=0;
    const std::vector<int64_t> rtc_args={32,1024,8,1,2,6144,0,32,4};
    for(int clean=0;clean<=32;++clean){
        rtc_gate.action_rtc_prefix_len_=clean;rtc_gate.run_gate(chunk);
        assert(rtc_gate.expected_gate.site==QWEN35_ACTION_RTC_RESIDUAL_GATE_SITE&&rtc_gate.expected_gate.selector==4);
        assert(rtc_gate.expected_gate.args==rtc_args&&rtc_gate.receipts==1&&calls.size()==1);
        assert(calls[0].kind==9&&calls[0].clean_rows==clean&&calls[0].input==0x30000&&calls[0].output==0x30000);
        assert(calls[0].shift==0x10000&&calls[0].scale==0x81000&&calls[0].noisy_scale==0x82800);
    }
    for(int stale:{-1,0,5,16,32,100}){rtc_gate.action_rtc_prefix_len_=stale;rtc_gate.plan(chunk);assert(rtc_gate.expected_gate.args==rtc_args);}
    rtc_gate.action_rtc_prefix_len_=13;chunk.offset=8;rtc_gate.run_gate(chunk);assert(calls[0].clean_rows==5);chunk.offset=0;
    for(size_t field=0;field<rtc_args.size();++field){rtc_gate.plan(chunk);++rtc_gate.expected_gate.args[field];calls.clear();reject([&]{rtc_gate.apply_adaptive_residual_gate(0x30000,0x10000,chunk);});assert(calls.empty());}
    for(int fault=0;fault<4;++fault){rtc_gate.plan(chunk);auto&g=rtc_gate.expected_gate;
        if(fault==0){g.selector=3;}if(fault==1){g.flags=1;}if(fault==2){++g.invocation;}if(fault==3){++g.site;}
        calls.clear();reject([&]{rtc_gate.apply_adaptive_residual_gate(0x30000,0x10000,chunk);});assert(calls.empty());}
    rtc_gate.complete=false;calls.clear();reject([&]{rtc_gate.apply_adaptive_residual_gate(0x30000,0x10000,chunk);});assert(calls.empty());rtc_gate.complete=true;
    rtc_gate.action_rtc_active_=false;assert(rtc_gate.action_residual_route(32)==1);
    rtc_gate.action_rtc_active_=true;assert(rtc_gate.action_residual_route(31)==1);

    payload.publish(false,false,false,false,false,true);Model ring_mode;ring_mode.enable_action_mode();
    assert(ring_mode.action_ring_rtc_residual_available_);
    payload.publish(false,false,false,false,false,false);
    assert(ring_mode.action_ring_rtc_residual_available_); // capability is frozen cold
    Model old_ring;old_ring.enable_action_mode();assert(!old_ring.action_ring_rtc_residual_available_);
    m.resolved=32; reject([&]{m.enable_action_mode();});
    payload.payload_availability_.withdraw();
    reject([&]{payload.has_loaded(KernelId::PI05_ADARMS_NORM_SHIFT_H1024);});
    reject([&]{payload.has_loaded(KernelId::PI05_GATED_RESIDUAL_H1024);});
}
'''
    for key, value in (("CONSTANTS", constants), ("ENABLE", enable), ("ROUTE", route),
                       ("APPLY", apply), ("NORM_ARGS", norm_args), ("MANIFEST", manifest), ("KERNEL_IDS", kernel_ids),
                       ("AVAILABILITY", availability), ("PUBLICATION", publication),
                       ("RESIDUAL_ROUTE", residual_route), ("RESIDUAL_APPLY", residual_apply),
                       ("RAW_RESIDUAL", raw_residual), ("ACTION_ROW", action_row)):
        cpp = cpp.replace(f"__{key}__", value)
    path, binary = tmp_path / "g05-norm-shift.cpp", tmp_path / "g05-norm-shift"
    cpp = with_actual_linear_contracts(_with_actual_g05_site_helpers(cpp))
    path.write_text(cpp)
    compiled = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-I", str(Path(__file__).resolve().parents[1] / "src/ops"), "-I", str(Path(__file__).resolve().parents[1] / "src/core"), str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
