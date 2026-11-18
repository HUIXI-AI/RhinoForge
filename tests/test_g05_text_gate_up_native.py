"""Actual G05 Text Gate/Up cold, descriptor and MLP code on host doubles."""
from pathlib import Path
import importlib
import json
import re

import pytest

from test_g05_vision_owner_model import _block
from test_g05_q_gate_native import _compile_run

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("w8", [False, True], ids=["fp16", "w8a16"])
def test_text_gate_up_actual_cold_install_order_manifest_and_mlp_fallback(tmp_path, w8):
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    header = (ROOT / 'src/fused/rpu_qwen3_5_model.h').read_text()
    methods = [
        ('void prepare_text_gateup_pair()', 'void Qwen3_5Model::prepare_text_gateup_pair('),
        ('bool use_text_gateup_pair(int64_t rows) const', 'bool Qwen3_5Model::use_text_gateup_pair('),
        ('std::vector<int64_t> text_gateup_pair_arguments() const', 'std::vector<int64_t> Qwen3_5Model::text_gateup_pair_arguments('),
        ('void launch_text_gateup_pair(int layer,const ChunkInfo& chunk)', 'void Qwen3_5Model::launch_text_gateup_pair('),
        ('void set_linear_acc32(bool enabled)', 'void Qwen3_5Model::set_linear_acc32('),
        ('void set_fast_replay(bool enabled)', 'void Qwen3_5Model::set_fast_replay('),
        ('void set_retained_prefill_graph(bool enabled)', 'void Qwen3_5Model::set_retained_prefill_graph('),
        ('int64_t action_gateup_resident_layers() const', 'int64_t Qwen3_5Model::action_gateup_resident_layers('),
        ('int64_t subclass_layout_hash() const', 'int64_t Qwen3_5Model::subclass_layout_hash('),
    ]
    manifest = _block(source, 'if (use_text_gateup_pair(chunk.len))')
    mlp = _block(source, 'void Qwen3_5Model::emit_mlp_and_output(')
    mlp = mlp[mlp.index('    const int64_t mlp_elems ='):mlp.index('    if (action_mlp_tail_handoff(layer_idx,chunk,true))')]
    install = _block(source, 'void Qwen3_5Model::set_weights(')
    install = install[install.rindex('    prepare_text_gateup_pair();'):]
    constants = '\n'.join(re.findall(r'constexpr int64_t QWEN35_(?:TEXT_GDN_LOCAL_SLICE_SITE|ACTION_[A-Z0-9_]+|TEXT_GATEUP_PAIR(?:_W8)?_SITE|TEXT_MLP_SILU_SITE)\s*=.*?;', source, re.S))
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
constexpr bool W8=__W8__;
#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
#define DWIDTH 2
#define NUM_CORES 8
namespace c10{struct Half{explicit Half(double){}};enum class DeviceType{PrivateUse1,CPU};}
namespace at{constexpr int kHalf=1,kChar=2;
struct Tensor{
 bool present=true,contiguous=true;int dtype=kHalf,rank=2;int64_t rows=6144,cols=2048,owner=0;std::string tag;
 c10::DeviceType location=c10::DeviceType::PrivateUse1;inline static int proofs=0;
 bool defined()const{++proofs;return present;}int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}
 struct Device{c10::DeviceType value;c10::DeviceType type()const{return value;}};Device device()const{return {location};}
 int64_t offset=0;int64_t storage_offset()const{return offset;}int dim()const{return rank;}int64_t size(int i)const{return i?cols:rows;}int64_t numel()const{return present?(rank==1?rows:rows*cols):0;}
};}
void append_kvinsert_cost_scalar_identity(std::vector<int64_t>&out,double x){int64_t b;std::memcpy(&b,&x,8);out.push_back(b);}
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>&out,const at::Tensor&t){out.push_back(t.owner);}
namespace detail{int64_t layout_mix(int64_t a,int64_t b){return uint64_t(a)*131+uint64_t(b);}}
enum class KernelId{G05_TEXT_GATE_UP_FP16_ACC16_M256N768K2048,G05_TEXT_GATE_UP_FP16_ACC32_M256N768K2048,G05_TEXT_GATE_UP_W8A16_ACC32_M256N768K2048,G05_TEXT_GATE_UP_W8A16_ACC16_M256N768K2048};
struct KernelCache{bool available=true,w8_available=true,acc16_available=true,fp16_acc16_available=true;int queries=0;KernelId last;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId id){++queries;last=id;return available&&(fp16_acc16_available||id!=KernelId::G05_TEXT_GATE_UP_FP16_ACC16_M256N768K2048)&&(w8_available||id!=KernelId::G05_TEXT_GATE_UP_W8A16_ACC32_M256N768K2048)&&(acc16_available||id!=KernelId::G05_TEXT_GATE_UP_W8A16_ACC16_M256N768K2048);}};
struct LayerWeights __WEIGHTS__;
enum class FmbRouteFamily{LINEAR,ACTIVATION};enum class ValuOpType{SILU,MUL};
__CONSTANTS__
struct ChunkInfo{int64_t len=256,idx=2,offset=512;};
struct Route{int64_t site;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>args;int64_t invocation;};
std::vector<std::string>launches;std::vector<std::pair<int64_t,int64_t>>weights_seen;
void rpu_launch_g05_text_gate_up_linear_fp16_acc32_kernel(uint32_t in,const at::Tensor&gate,const at::Tensor&up,uint32_t go,uint32_t uo){assert(in==0x10000&&go==0x110000&&uo==0x170000);assert(gate.owner+1==up.owner);weights_seen.push_back({gate.owner,up.owner});launches.push_back("pair");}
void rpu_launch_g05_text_gate_up_linear_fp16_acc16_kernel(uint32_t in,const at::Tensor&gate,const at::Tensor&up,uint32_t go,uint32_t uo){rpu_launch_g05_text_gate_up_linear_fp16_acc32_kernel(in,gate,up,go,uo);launches.back()="pair_fp16_acc16";}
std::vector<bool>linear_modes;
void rpu_launch_g05_text_gate_up_linear_w8a16_acc32_kernel(uint32_t in,const at::Tensor&gate,const at::Tensor&up,uint32_t go,uint32_t uo,const at::Tensor&gs,const at::Tensor&us,bool linear_acc32=true){linear_modes.push_back(linear_acc32);assert(gate.dtype==at::kChar&&up.dtype==at::kChar);assert(gs.owner==gate.owner+100&&us.owner==up.owner+100);rpu_launch_g05_text_gate_up_linear_fp16_acc32_kernel(in,gate,up,go,uo);launches.back()="pair_w8";}
void rpu_launch_eltwise_unary_spm_kernel(uint32_t in,uint32_t out,int64_t n,ValuOpType op,bool,int cores){assert(in==0x110000&&out==in&&n>0&&op==ValuOpType::SILU&&cores==8);launches.push_back("silu");}
void rpu_launch_eltwise_binary_spm_kernel(uint32_t in,uint32_t up,uint32_t out,int64_t n,ValuOpType op,c10::Half,int cores){assert(in==0x110000&&up==0x170000&&out==0x10000&&n>0&&op==ValuOpType::MUL&&cores==8);launches.push_back("mul");}
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>&v,int owner){v.push_back(owner);}
struct Model{
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
 std::vector<int64_t>text_mlp_tail_arguments()const{return {};}
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

 // NVFP4 is disabled in this FP16/W8 caller fixture.
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
 bool action_mode_=false,use_silu_=true,linear_acc32_=false,has_gdn_=true,fast_replay_enabled_=false,fast_replay_active_=false;
 bool retained_prefill_graph_=false,z2_bound_=false,debug=false,text_gateup_pair_available_=false,text_gateup_pair_w8_=false,complete=true;
 bool action_qkv_gate_available_=false;
 bool action_gateup_resident_w8_=false,action_kv_owner_w8_=false,action_q_gate_w8_=false,action_gateup_pair_nvfp4_=false,action_gateup_pair_w8_=false;
 bool has_qk_norm_=true,has_mrope_=true,action_norm_shift_available_=false,action_gated_residual_available_=false,action_gateup_resident_available_=false,action_kv_owner_available_=false,action_q_gate_available_=false,action_gateup_pair_available_=false,action_silu_round_mul_available_=false;
 int64_t layers=24,h=2048,intermediate=6144,cores=8,tp=8,heads=8,kv=8,hd=256,action_len_=32,chunk_size_cap_=0,sdpa_kernel_=0,last_resolved=0;
 double eps_=1e-6;int invalidations=0;
 std::vector<LayerWeights>layer_weights_{24};std::vector<uint8_t>layer_is_full_=std::vector<uint8_t>(24);std::vector<Route>routes;
 std::vector<at::Tensor>action_gateup_resident_,action_o_resident_;at::Tensor final_norm_w_,action_input_w_,action_input_b_,action_output_w_,action_output_b_;
 Model(){for(int i=0;i<24;++i){layer_is_full_[i]=(i+1)%4==0;auto&w=layer_weights_[i];w.gate_w.owner=2*i+1;w.up_w.owner=2*i+2;w.gate_w.tag="gate";w.up_w.tag="up";w.down_w.tag="down";w.gate_ws.present=w.up_ws.present=W8;w.gate_ws.rank=w.up_ws.rank=1;w.gate_ws.owner=w.gate_w.owner+100;w.up_ws.owner=w.up_w.owner+100;if(W8)w.gate_w.dtype=w.up_w.dtype=at::kChar;}}
 int64_t num_layers()const{return layers;}int64_t hidden_size()const{return h;}int64_t intermediate_size()const{return intermediate;}
 int64_t num_cores()const{return cores;}int64_t attn_tp()const{return tp;}int64_t mlp_tp()const{return tp;}int64_t gdn_tp()const{return tp;}int64_t num_q_heads()const{return heads;}int64_t num_kv_heads()const{return kv;}int64_t head_dim()const{return hd;}
 int64_t get_last_resolved_chunk_size()const{return last_resolved;}void invalidate_model_state(){++invalidations;}bool get_debug_export()const{return debug;}
 Model&ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}
 std::vector<int64_t>cold_topology_arguments()const{return {cores};}
 bool use_action_mod_body()const{return false;}bool use_action_gateup_resident(int64_t)const{return false;}bool use_action_gateup_pair(int64_t)const{return false;}bool use_action_silu_round_mul(int64_t)const{return false;}bool use_action_q_gate(int64_t)const{return false;}
 bool use_action_qkv_gate(int64_t)const{return false;}std::vector<int64_t>action_qkv_gate_arguments()const{return {};}bool use_action_kv_owner(int64_t)const{return false;}
 void launch_action_gateup_resident(int,int,uint32_t){assert(false);}void launch_action_gateup_pair(int,const ChunkInfo&){assert(false);}void launch_action_silu_round_mul(int,const ChunkInfo&){assert(false);}
 uint32_t addr(int,const char*n)const{std::string s(n);return s=="residual1"?0x10000:s=="gate"?0x110000:s=="up"?0x170000:s=="input_norm"?0x210000:0x310000;}
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t>args,int64_t inv){if(family==FmbRouteFamily::ACTIVATION){assert(site==QWEN35_TEXT_MLP_SILU_SITE);return;}auto i=std::find_if(routes.begin(),routes.end(),[&](auto&r){return r.site==site&&r.invocation==inv;});TORCH_CHECK(i!=routes.end()&&i->family==family&&i->selector==selector&&i->flags==flags&&i->args==args);}
 void launch_linear(int64_t,uint32_t in,const at::Tensor&w,uint32_t out,int64_t rows,int64_t,int64_t,int,int64_t,int,const at::Tensor&,int=-1,int=-1,const at::Tensor& negative_zero={}){assert(negative_zero.owner==0);assert(rows>0&&in==0x10000);if(w.tag=="gate")assert(out==0x110000);if(w.tag=="up")assert(out==0x170000);if(w.tag=="down")assert(out==0x310000);launches.push_back(w.tag);}
 __METHODS__
 std::vector<int64_t>kvinsert_cost_weight_identity()const __IDENTITY__
 void installed(){__INSTALL__
 void plan(ChunkInfo chunk={}){struct{std::vector<Route>routes;}manifest;if(use_text_gateup_pair(chunk.len))__MANIFEST__ routes=manifest.routes;}
 void mlp(int layer_idx,ChunkInfo chunk={}){const auto&lw=layer_weights_[layer_idx];int64_t seq_len=chunk.len,h=hidden_size();__MLP__ (void)residual_buf;}
};
template<class F>void reject(F f){auto before=launches;try{f();assert(false);}catch(const std::runtime_error&){}assert(launches==before);}
int main(){
 Model m;auto&cache=KernelCache::instance();
 // Execute production setter bodies in G05's real installation order.
 m.installed();assert(!m.text_gateup_pair_available_&&m.invalidations==1);
 m.set_linear_acc32(true);assert(!m.text_gateup_pair_available_&&m.invalidations==2);
 m.set_fast_replay(true);assert(m.text_gateup_pair_available_&&m.invalidations==3&&!m.use_text_gateup_pair(256));
 m.set_retained_prefill_graph(true);assert(m.use_text_gateup_pair(256)&&m.invalidations==4);
 m.plan();assert(m.routes.size()==1);assert(m.routes[0].site==(W8?267678052960427182LL:751944759910871833LL)&&m.routes[0].selector==1&&m.routes[0].flags==0&&m.routes[0].invocation==2);
 assert(m.routes[0].args==(W8?std::vector<int64_t>({3,256,768,2048,8,32,288,80,128,10,20,4096,1536,4,24,8,6144,2,32,20,34,4,36,1}):std::vector<int64_t>({3,256,768,2048,8,32,368,64,128,12,24,4096,1536,2,24,8,6144,2,32,20,34,4,36,1})));
 assert(m.text_gateup_pair_w8_==W8&&cache.last==(W8?KernelId::G05_TEXT_GATE_UP_W8A16_ACC32_M256N768K2048:KernelId::G05_TEXT_GATE_UP_FP16_ACC32_M256N768K2048));
 const auto hash=m.subclass_layout_hash();const auto identity=m.kvinsert_cost_weight_identity();
 const int proofs=at::Tensor::proofs,queries=cache.queries;
 for(int layer=0;layer<24;++layer){launches.clear();m.mlp(layer);assert(launches==std::vector<std::string>({W8?"pair_w8":"pair","silu","mul","down"}));assert(weights_seen.back()==std::make_pair(int64_t(2*layer+1),int64_t(2*layer+2)));}
 assert(at::Tensor::proofs==proofs&&cache.queries==queries);
 for(int arg=0;arg<int(m.routes[0].args.size());++arg){auto x=m;++x.routes[0].args[arg];reject([&]{x.mlp(0);});}
 for(int field=0;field<6;++field){auto x=m;auto&r=x.routes[0];if(field==0)++r.selector;if(field==1)++r.flags;if(field==2)++r.invocation;if(field==3)++r.site;if(field==4)r.args.pop_back();if(field==5)r.family=FmbRouteFamily::ACTIVATION;reject([&]{x.mlp(0);});}
 {auto x=m;x.routes.clear();reject([&]{x.mlp(0);});}m.complete=false;reject([&]{m.mlp(0);});m.complete=true;
 reject([&]{m.launch_text_gateup_pair(-1,{});});reject([&]{m.launch_text_gateup_pair(24,{});});reject([&]{m.launch_text_gateup_pair(0,{128,2,0});});reject([&]{m.launch_text_gateup_pair(0,{256,3,0});});
 cache.available=false;m.set_fast_replay(true);assert(!m.text_gateup_pair_available_);m.plan();assert(m.routes.empty()&&m.subclass_layout_hash()!=hash&&m.kvinsert_cost_weight_identity()!=identity);launches.clear();m.mlp(0);assert(launches==std::vector<std::string>({"gate","silu","up","mul","down"}));
 cache.available=true;m.set_fast_replay(true);assert(m.text_gateup_pair_available_&&m.subclass_layout_hash()==hash&&m.kvinsert_cost_weight_identity()==identity);
 if(W8){cache.w8_available=false;m.installed();assert(!m.text_gateup_pair_available_&&!m.text_gateup_pair_w8_);m.plan();launches.clear();m.mlp(0);assert(launches==std::vector<std::string>({"gate","silu","up","mul","down"}));cache.w8_available=true;m.installed();m.plan();assert(m.text_gateup_pair_available_&&m.text_gateup_pair_w8_);}
 // Any one of all 48 installed owners can withdraw the capability on reinstall.
 for(int layer=0;layer<24;++layer)for(int which=0;which<2;++which){auto x=m;(which?x.layer_weights_[layer].up_w:x.layer_weights_[layer].gate_w).dtype=W8?at::kHalf:at::kChar;x.installed();assert(!x.text_gateup_pair_available_&&x.invalidations==m.invalidations+1);}
 for(int which=0;which<2;++which)for(int fault=0;fault<6;++fault){auto x=m;auto&w=which?x.layer_weights_.back().up_w:x.layer_weights_.back().gate_w;if(fault==0)w.present=false;if(fault==1)w.contiguous=false;if(fault==2)w.location=c10::DeviceType::CPU;if(fault==3)w.rank=1;if(fault==4)w.rows=4096;if(fault==5)w.cols=1024;x.installed();assert(!x.text_gateup_pair_available_);}
 for(int layer=0;layer<24;++layer)for(int which=0;which<2;++which){
  if(W8){for(int fault=0;fault<6;++fault){auto x=m;auto&s=which?x.layer_weights_[layer].up_ws:x.layer_weights_[layer].gate_ws;if(fault==0)s.present=false;if(fault==1)s.dtype=at::kChar;if(fault==2)s.location=c10::DeviceType::CPU;if(fault==3)s.contiguous=false;if(fault==4)s.rank=2;if(fault==5)s.rows=6143;x.installed();assert(!x.text_gateup_pair_available_&&!x.text_gateup_pair_w8_);}}
  else{auto x=m;(which?x.layer_weights_[layer].up_ws:x.layer_weights_[layer].gate_ws).present=true;x.installed();assert(!x.text_gateup_pair_available_);}
 }
 if(W8){auto x=m;++x.layer_weights_.back().up_ws.owner;assert(x.kvinsert_cost_weight_identity()!=identity);}
 for(int layer=0;layer<24;++layer){auto x=m;x.layer_is_full_[layer]=!x.layer_is_full_[layer];x.installed();assert(!x.text_gateup_pair_available_);}
 for(int fault=0;fault<10;++fault){auto x=m;if(fault==0)x.action_mode_=true;if(fault==1)x.has_gdn_=false;if(fault==2)x.use_silu_=false;if(fault==3)x.h=1024;if(fault==4)x.intermediate=4096;if(fault==5)x.cores=4;if(fault==6)x.tp=4;if(fault==7)x.heads=16;if(fault==8)x.layer_weights_.pop_back();if(fault==9)x.layer_is_full_.pop_back();x.prepare_text_gateup_pair();assert(!x.text_gateup_pair_available_);}
 {auto x=m;x.layer_weights_.back().up_w.owner=999;assert(x.kvinsert_cost_weight_identity()!=identity);}
 auto fallback=[&](auto change,ChunkInfo chunk=ChunkInfo{}){auto x=m;change(x);x.plan(chunk);assert(x.routes.empty()&&!x.use_text_gateup_pair(chunk.len));launches.clear();x.mlp(0,chunk);assert(launches==std::vector<std::string>({"gate","silu","up","mul","down"}));};
 for(int rows:{1,32,64,128,512})fallback([](auto&){},{rows,2,0});
 fallback([](auto&x){x.set_fast_replay(false);});fallback([](auto&x){x.set_retained_prefill_graph(false);});fallback([](auto&x){x.z2_bound_=true;});fallback([](auto&x){x.debug=true;});
 m.set_linear_acc32(false);assert(m.text_gateup_pair_available_);m.set_linear_acc32(true);assert(m.text_gateup_pair_available_);m.set_fast_replay(false);assert(!m.text_gateup_pair_available_);m.set_fast_replay(true);assert(m.text_gateup_pair_available_);
 if(W8){auto x=m;x.set_linear_acc32(false);x.plan();assert(x.text_gateup_pair_available_&&cache.last==KernelId::G05_TEXT_GATE_UP_W8A16_ACC16_M256N768K2048&&x.routes[0].args[5]==16&&x.routes[0].args[6]==400&&x.subclass_layout_hash()!=hash&&x.kvinsert_cost_weight_identity()!=identity);
  auto valid=x.routes;x.routes=m.routes;reject([&]{x.mlp(0);});x.routes=valid;linear_modes.clear();launches.clear();x.mlp(23);assert(launches==std::vector<std::string>({"pair_w8","silu","mul","down"})&&linear_modes==std::vector<bool>({false}));
  cache.acc16_available=false;x.installed();x.plan();assert(!x.text_gateup_pair_available_&&x.routes.empty());linear_modes.clear();launches.clear();x.mlp(23);assert(launches==std::vector<std::string>({"gate","silu","up","mul","down"})&&linear_modes.empty());cache.acc16_available=true;
 }
 if(!W8){auto x=m;x.set_linear_acc32(false);x.plan();assert(x.text_gateup_pair_available_&&cache.last==KernelId::G05_TEXT_GATE_UP_FP16_ACC16_M256N768K2048&&x.routes[0].args==std::vector<int64_t>({3,256,768,2048,8,16,432,80,128,10,20,4096,1536,2,24,8,6144,2,32,20,34,4,36,1})&&x.subclass_layout_hash()!=hash&&x.kvinsert_cost_weight_identity()!=identity);
  auto alias=x;alias.routes[0].site=751944759910871833LL;reject([&]{alias.mlp(0);});
  auto valid=x.routes;x.routes=m.routes;reject([&]{x.mlp(0);});x.routes=valid;
  for(int layer=0;layer<24;++layer){launches.clear();x.mlp(layer);assert(launches==std::vector<std::string>({"pair_fp16_acc16","silu","mul","down"}));}
  for(int layer=0;layer<24;++layer)for(int side=0;side<2;++side){auto y=x;(side?y.layer_weights_[layer].up_w:y.layer_weights_[layer].gate_w).offset=256;y.installed();assert(!y.text_gateup_pair_available_);}
  cache.fp16_acc16_available=false;x.installed();x.plan();assert(!x.text_gateup_pair_available_&&x.routes.empty());launches.clear();x.mlp(23);assert(launches==std::vector<std::string>({"gate","silu","up","mul","down"}));m.installed();assert(m.text_gateup_pair_available_);cache.fp16_acc16_available=true;
 }
 // Action precision is fixed before installing its weights, even before first planning.
 {auto x=m;x.action_mode_=true;const int count=x.invalidations;x.set_linear_acc32(true);assert(x.invalidations==count+1);reject([&]{x.set_linear_acc32(false);});assert(x.linear_acc32_&&x.invalidations==count+1);}
 m.last_resolved=256;const int invalidations=m.invalidations;reject([&]{m.set_linear_acc32(false);});reject([&]{m.set_fast_replay(false);});reject([&]{m.set_retained_prefill_graph(false);});assert(m.invalidations==invalidations&&m.linear_acc32_&&m.fast_replay_enabled_&&m.retained_prefill_graph_);
}
'''
    for marker, value in {
        '__W8__': '1' if w8 else '0',
        '__METHODS__': '\n'.join(decl + _block(source, signature) for decl, signature in methods),
        '__WEIGHTS__': _block(header, 'struct LayerWeights'),
        '__IDENTITY__': _block(header, 'std::vector<int64_t> kvinsert_cost_weight_identity() const'),
        '__INSTALL__': install, '__CONSTANTS__': constants,
        '__MANIFEST__': manifest, '__MLP__': mlp,
    }.items():
        cpp = cpp.replace(marker, value)
    _compile_run(tmp_path, 'text_pair', cpp)
