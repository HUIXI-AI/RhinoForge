"""Actual Gate/Up cold admission, MLP scheduling and launch ABI on CPU doubles."""
from pathlib import Path
import re

from test_siglip_cold_profile import _body
from test_g05_q_gate_native import _compile_run

ROOT = Path(__file__).resolve().parents[1]


def test_gate_up_pair_cold_owners_resident_priority_and_descriptor_failure(tmp_path):
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    methods = [
        ('void prepare_action_gateup_pair()', 'void Qwen3_5Model::prepare_action_gateup_pair('),
        ('bool use_action_gateup_pair(int64_t rows) const', 'bool Qwen3_5Model::use_action_gateup_pair('),
        ('bool use_action_mod_body() const', 'bool Qwen3_5Model::use_action_mod_body('),
        ('std::vector<int64_t> action_gateup_pair_arguments() const', 'std::vector<int64_t> Qwen3_5Model::action_gateup_pair_arguments('),
        ('void launch_action_gateup_pair(int layer,const ChunkInfo& chunk)', 'void Qwen3_5Model::launch_action_gateup_pair('),
        ('int64_t action_gateup_resident_layers() const', 'int64_t Qwen3_5Model::action_gateup_resident_layers('),
        ('int64_t subclass_layout_hash() const', 'int64_t Qwen3_5Model::subclass_layout_hash('),
    ]
    begin = source.index('{', source.index('if (use_action_gateup_pair(chunk.len))'))
    end = source.index('\n        }', begin) + len('\n        }')
    manifest = source[begin:end]
    mlp = _body(source, 'void Qwen3_5Model::emit_mlp_and_output(')
    mlp = mlp[mlp.index('    const int64_t mlp_elems ='):mlp.index('    if (action_mlp_tail_handoff(layer_idx,chunk,true))')]
    constants = '\n'.join(re.findall(r'constexpr int64_t QWEN35_(?:TEXT_GDN_LOCAL_SLICE_SITE|ACTION_(?:DOWN_\w+|O_\w+|GATEUP_\w+|RESIDENT_\w+|Q_GATE_\w+|QKV_GATE_\w+|OWNER_CACHE_\w+|KV_OWNER_\w+|SILU_ROUND_MUL_SITE|FUSED_NORM_SHIFT_SITE|RTC_RESIDUAL_GATE_SITE|RING_RTC_RESIDUAL_SITE|RESIDUAL_GATE_SITE)|TEXT_MLP_SILU_SITE|TEXT_GATEUP_PAIR_\w+)\s*=.*?;', source, re.S))
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
#define DWIDTH 2
int tensor_proofs=0;
namespace c10{struct Half{explicit Half(double){}};enum class DeviceType{PrivateUse1,CPU};}
namespace at{constexpr int kHalf=1,kChar=3,kByte=2;
struct Tensor{
 bool present=true,contiguous=true;int dtype=kHalf,rank=2;int64_t storage_offset_=0;int64_t rows=4096,cols=1024;uint64_t owner=0;std::string tag;
 c10::DeviceType location=c10::DeviceType::PrivateUse1;
 bool defined()const{++tensor_proofs;return present;}int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}
 struct Device{c10::DeviceType value;c10::DeviceType type()const{return value;}bool operator!=(const Device&b)const{return value!=b.value;}};Device device()const{return {location};}
 int64_t numel()const{return present?(rank==1?rows:rows*cols):0;}int64_t storage_offset()const{return storage_offset_;}int dim()const{return rank;}int64_t size(int i)const{return i?cols:rows;}
};}
enum class FmbLinearRouteSelector{G05_NVFP4_PAIRED_ACC32=13,G05_NVFP4_RESIDENT_ACC32=14,G05_NVFP4_SWIGLU_ACC32=15,G05_NVFP4_PAIRED_ACC16=9,G05_NVFP4_SWIGLU_ACC16=11};
struct G05Nvfp4Tables{static constexpr const char*names[]={"qts","ots","gts","uts","dts"};};
template<class...T>void rpu_launch_g05_gate_up_paired_nvfp4_acc16_kernel(T...){assert(false);}
template<class...T>void rpu_launch_g05_gate_up_paired_nvfp4_acc32_kernel(T...){assert(false);}
namespace detail{int64_t layout_mix(int64_t a,int64_t b){return uint64_t(a)*131+uint64_t(b);}}
enum class KernelId{G05_GATE_UP_PAIRED_NVFP4_ACC32_M32N512K1024_N128,G05_GATE_UP_FP16_ACC16_M32N512K1024,G05_GATE_UP_PAIRED_NVFP4_ACC16_M32N512K1024_N128,G05_GATE_UP_FP16_ACC32_M32N512K1024,G05_GATE_UP_W8A16_ACC32_M32N512K1024,G05_GATE_UP_W8A16_ACC16_M32N512K1024};
struct KernelCache{bool available=true,fp16_acc16=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId id)const{return available&&(fp16_acc16||id!=KernelId::G05_GATE_UP_FP16_ACC16_M32N512K1024);}};
struct Weights{at::Tensor input_norm_w,post_norm_w,gate_w,up_w,down_w,gate_ws,up_ws,down_ws,gate_negative_zero,up_negative_zero,down_negative_zero,q_w,k_w,v_w,o_w,attn_gate_w,q_ws,k_ws,v_ws,o_ws,attn_gate_ws,gdn_q_w,gdn_k_w,gdn_v_w,gdn_in_z_w,gdn_out_w,gdn_q_ws,gdn_k_ws,gdn_v_ws,gdn_in_z_ws,gdn_out_ws,gdn_b_bg_w,gdn_a_bg_w,gdn_b_bg_ws,gdn_a_bg_ws;};
enum class FmbRouteFamily{LINEAR,ACTIVATION};enum class ValuOpType{SILU,MUL};
__CONSTANTS__
struct ChunkInfo{int64_t len=32,idx=2;};
struct Route{int64_t site;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>args;int64_t invocation;};
std::vector<std::string>launches;std::vector<std::pair<uint64_t,uint64_t>>weights_seen;
void rpu_launch_g05_gate_up_linear_fp16_acc32_kernel(uint32_t in,const at::Tensor&gate,const at::Tensor&up,uint32_t go,uint32_t uo){assert(in==0x10000&&go==0x20000&&uo==0x28000&&gate.owner!=up.owner);launches.push_back("pair");weights_seen.push_back({gate.owner,up.owner});}
void rpu_launch_g05_gate_up_linear_fp16_acc16_kernel(uint32_t in,const at::Tensor&gate,const at::Tensor&up,uint32_t go,uint32_t uo){rpu_launch_g05_gate_up_linear_fp16_acc32_kernel(in,gate,up,go,uo);launches.back()="pair-fp16-acc16";}
void rpu_launch_eltwise_unary_spm_kernel(uint32_t in,uint32_t out,int64_t n,ValuOpType type,bool,int cores){assert(in==0x20000&&out==in&&n==16384&&type==ValuOpType::SILU&&cores==8);launches.push_back("silu");}
void rpu_launch_eltwise_binary_spm_kernel(uint32_t in,uint32_t rhs,uint32_t out,int64_t n,ValuOpType type,c10::Half,int cores){assert(in==0x20000&&rhs==0x28000&&out==in&&n==16384&&type==ValuOpType::MUL&&cores==8);launches.push_back("mul");}
std::vector<bool>linear_modes;
void rpu_launch_g05_gate_up_linear_w8a16_acc32_kernel(uint32_t,const at::Tensor&a,const at::Tensor&b,const at::Tensor&as,const at::Tensor&bs,uint32_t,uint32_t,bool linear_acc32=true){linear_modes.push_back(linear_acc32);assert(a.dtype==at::kChar&&b.dtype==at::kChar&&as.rank==1&&bs.rank==1&&as.owner!=bs.owner);launches.push_back("pair-w8");weights_seen.push_back({a.owner,b.owner});}
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>&v,int owner){v.push_back(owner);}
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>&v,const at::Tensor&t){v.push_back(t.owner);}
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

 // This legacy fixture exercises FP16/W8; mixed NVFP4 has its own actual-body test.
 struct Nvfp4Stub {bool enabled()const{return false;}std::vector<int>values;};
 Nvfp4Stub action_nvfp4_;
 // The dense-text NVFP4 owner is uninstalled in this probe (enabled() false, no tables).
 Nvfp4Stub text_nvfp4_;static constexpr int64_t kTextNvfp4TablesSite=3184997657402357068LL;
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
 bool action_gateup_resident_w8_=false,action_kv_owner_w8_=false,action_q_gate_w8_=false,action_gateup_pair_nvfp4_=false,action_gateup_pair_w8_=false,text_gateup_pair_w8_=false;
 bool action_mode_=true,use_silu_=true,linear_acc32_=true,has_gdn_=false,action_gateup_pair_available_=false,resident=true;
 bool action_step_active_=false,action_loop_active_=false,complete=true;
 int64_t action_len_=32,action_num_steps_=1,layers=24,h=1024,intermediate=4096,cores=8,heads=8,kv=8,hd=256,tp=8;
 std::vector<Weights>layer_weights_{24};std::vector<uint8_t>layer_is_full_=std::vector<uint8_t>(24,1);std::vector<Route>routes;
 Model(){for(int i=0;i<24;++i){auto&w=layer_weights_[i];w.gate_w.owner=2*i+1;w.up_w.owner=2*i+2;w.gate_w.tag="gate";w.up_w.tag="up";w.down_w.tag="down";}}
 int64_t num_layers()const{return layers;}int64_t hidden_size()const{return h;}int64_t intermediate_size()const{return intermediate;}
 int64_t num_cores()const{return cores;}int64_t attn_tp()const{return tp;}int64_t mlp_tp()const{return tp;}int64_t num_q_heads()const{return heads;}int64_t num_kv_heads()const{return kv;}int64_t head_dim()const{return hd;}
 Model&ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}
 bool use_text_gateup_pair(int64_t)const{return false;}
 void launch_text_gateup_pair(int,const ChunkInfo&){assert(false);}
 std::vector<int64_t>cold_topology_arguments()const{return {cores};}bool use_action_q_gate(int64_t)const{return false;}
 bool use_action_qkv_gate(int64_t)const{return false;}std::vector<int64_t>action_qkv_gate_arguments()const{return {};}bool use_action_kv_owner(int64_t)const{return false;}
 bool use_action_silu_round_mul(int64_t)const{return false;}
 void launch_action_silu_round_mul(int,const ChunkInfo&){assert(false);}
 bool use_action_gateup_resident(int64_t rows)const{return resident&&use_action_mod_body()&&rows==32;}
 uint32_t addr(int,const char*n)const{std::string name(n);return name=="residual1"?0x10000:name=="gate"?0x20000:name=="up"?0x28000:0x30000;}
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t>args,int64_t inv){if(family==FmbRouteFamily::ACTIVATION){assert(site==QWEN35_TEXT_MLP_SILU_SITE);return;}auto i=std::find_if(routes.begin(),routes.end(),[&](auto&r){return r.site==site&&r.invocation==inv;});TORCH_CHECK(i!=routes.end()&&i->family==family&&i->selector==selector&&i->flags==flags&&i->args==args);}
 void launch_linear(int64_t,uint32_t,const at::Tensor&w,uint32_t,int64_t,int64_t,int64_t,int,int64_t,int,const at::Tensor&scale,int=-1,int=-1,const at::Tensor& negative_zero={}){assert(negative_zero.owner==0);if(w.dtype==at::kChar)assert(scale.rank==1&&scale.owner!=0);launches.push_back(w.tag);}
 void launch_action_gateup_resident(int layer,int projection,uint32_t out){assert(layer>=1&&layer<=action_gateup_resident_layers()&&use_action_gateup_resident(32));assert(out==(projection?0x28000u:0x20000u));launches.push_back(projection?"resident-up":"resident-gate");}
 __METHODS__
 void plan(ChunkInfo chunk={}){struct{std::vector<Route>routes;}manifest;if(use_action_gateup_pair(chunk.len))__MANIFEST__ routes=manifest.routes;}
 void mlp(int layer_idx,ChunkInfo chunk={}){const auto&lw=layer_weights_[layer_idx];int64_t seq_len=chunk.len,h=hidden_size();__MLP__ (void)residual_buf;}
};
template<class F>void reject(F f){auto before=launches;try{f();assert(false);}catch(const std::runtime_error&){}assert(launches==before);}
int main(){
 Model m;m.prepare_action_gateup_pair();const int cold_proofs=tensor_proofs;assert(m.action_gateup_pair_available_&&cold_proofs>=48);
 m.plan();assert(m.routes.empty()&&!m.use_action_gateup_pair(32));m.mlp(1);assert(launches==std::vector<std::string>({"gate","silu","up","mul","down"}));
 m.action_step_active_=m.action_loop_active_=true;m.action_num_steps_=10;m.plan();const auto paired_hash=m.subclass_layout_hash();
 assert(m.routes.size()==1&&m.routes[0].args==std::vector<int64_t>({4,32,512,1024,8,32,160,96,256,6,12,2048,1024,2,32,3,20,34,2,4,36,1}));
 for(int layer=0;layer<24;++layer){launches.clear();m.mlp(layer);if(layer>=1&&layer<=3){assert(launches==std::vector<std::string>({"resident-gate","silu","resident-up","mul","down"}));reject([&]{m.launch_action_gateup_pair(layer,{});});}else{assert(launches==std::vector<std::string>({"pair","silu","mul","down"}));assert(weights_seen.back()==std::make_pair(uint64_t(2*layer+1),uint64_t(2*layer+2)));}}
 assert(tensor_proofs==cold_proofs);m.complete=false;reject([&]{m.mlp(0);});m.complete=true;m.routes[0].args[12]=512;reject([&]{m.mlp(0);});m.plan();
 // Stale resident count rejects before a pair or downstream SiLU/MUL runs.
 m.resident=false;reject([&]{m.mlp(0);});m.plan();assert(m.routes[0].args[15]==0);launches.clear();m.mlp(1);assert(launches==std::vector<std::string>({"pair","silu","mul","down"}));m.resident=true;reject([&]{m.mlp(0);});m.plan();
 reject([&]{m.launch_action_gateup_pair(24,{});});reject([&]{m.launch_action_gateup_pair(-1,{});});reject([&]{m.launch_action_gateup_pair(0,{31,2});});reject([&]{m.launch_action_gateup_pair(0,{32,3});});
 KernelCache::instance().available=false;m.prepare_action_gateup_pair();assert(!m.action_gateup_pair_available_);assert(m.subclass_layout_hash()!=paired_hash);m.plan();assert(m.routes.empty());launches.clear();m.mlp(0);assert(launches==std::vector<std::string>({"gate","silu","up","mul","down"}));launches.clear();m.mlp(2);assert(launches==std::vector<std::string>({"resident-gate","silu","resident-up","mul","down"}));
 KernelCache::instance().available=true;m.prepare_action_gateup_pair();assert(m.action_gateup_pair_available_&&m.subclass_layout_hash()==paired_hash);
 for(int which:{0,1})for(int fault=0;fault<7;++fault){auto x=m;auto&w=which?x.layer_weights_.back().up_w:x.layer_weights_.back().gate_w;
  if(fault==0){w.present=false;}if(fault==1){w.dtype=2;}if(fault==2){w.location=c10::DeviceType::CPU;}if(fault==3){w.contiguous=false;}if(fault==4){w.rank=3;}if(fault==5){w.rows=2048;}if(fault==6){w.cols=2048;}
  x.prepare_action_gateup_pair();assert(!x.action_gateup_pair_available_);x.plan();assert(x.routes.empty());
 }
 for(int fault=0;fault<9;++fault){auto x=m;if(fault==0)x.linear_acc32_=false;if(fault==1)x.has_gdn_=true;if(fault==2)x.layer_is_full_.back()=0;if(fault==3)x.heads=16;if(fault==4)x.kv=2;if(fault==5)x.intermediate=2048;if(fault==6)x.tp=4;if(fault==7)x.layer_weights_.pop_back();if(fault==8)x.use_silu_=false;x.prepare_action_gateup_pair();assert(!x.action_gateup_pair_available_);}
 for(int fault=0;fault<4;++fault){auto x=m;if(fault==0)x.action_num_steps_=1;if(fault==1)x.action_loop_active_=false;if(fault==2)x.action_step_active_=false;if(fault==3)x.action_len_=16;assert(!x.use_action_gateup_pair(32));x.plan();assert(x.routes.empty());}
 // A genuine optional FP16 ACC16 parent has a distinct mode/tile receipt.
 {auto x=m;x.linear_acc32_=false;x.action_mod_body_fp16_=true;x.resident=false;
  for(auto&w:x.layer_weights_)w.gate_ws.present=w.up_ws.present=false;
  x.prepare_action_gateup_pair();x.plan();assert(x.use_action_gateup_pair(32)&&x.routes[0].args==std::vector<int64_t>({4,32,512,1024,8,16,208,96,256,6,12,2048,1024,2,32,0,20,34,2,4,36,1})&&x.subclass_layout_hash()!=paired_hash);
  auto alias=x;alias.routes[0].site=5257909631264440265LL;reject([&]{alias.mlp(0);});auto valid=x.routes;x.routes=m.routes;reject([&]{x.mlp(0);});x.routes=valid;
  for(int layer=0;layer<24;++layer){launches.clear();x.mlp(layer);assert(launches==std::vector<std::string>({"pair-fp16-acc16","silu","mul","down"}));}
  for(int layer=0;layer<24;++layer)for(int side=0;side<2;++side){auto y=x;(side?y.layer_weights_[layer].up_w:y.layer_weights_[layer].gate_w).storage_offset_=256;y.prepare_action_gateup_pair();assert(!y.action_gateup_pair_available_);auto z=x;(side?z.layer_weights_[layer].up_ws:z.layer_weights_[layer].gate_ws).present=true;z.prepare_action_gateup_pair();assert(!z.action_gateup_pair_available_);}
  KernelCache::instance().fp16_acc16=false;x.prepare_action_gateup_pair();x.plan();assert(x.routes.empty()&&!x.action_gateup_pair_available_);launches.clear();x.mlp(23);assert(launches==std::vector<std::string>({"gate","silu","up","mul","down"}));m.prepare_action_gateup_pair();assert(m.action_gateup_pair_available_);KernelCache::instance().fp16_acc16=true;
 }
 for(auto&w:m.layer_weights_){w.gate_w.dtype=w.up_w.dtype=w.down_w.dtype=at::kChar;w.gate_ws.rank=w.up_ws.rank=w.down_ws.rank=1;w.gate_ws.owner=w.gate_w.owner+100;w.up_ws.owner=w.up_w.owner+200;w.down_ws.owner=301;}
 m.action_gateup_resident_w8_=true;m.prepare_action_gateup_pair();assert(m.action_gateup_pair_available_&&m.action_gateup_pair_w8_&&m.action_gateup_resident_layers()==6);m.plan();
 assert(m.routes[0].site==QWEN35_ACTION_GATEUP_PAIR_W8_SITE&&m.routes[0].args==std::vector<int64_t>({4,32,512,1024,8,32,48,80,512,7,14,2048,1024,2,32,6,20,34,1,4,36,1}));
 const int w8proofs=tensor_proofs;launches.clear();m.mlp(0);assert(launches==std::vector<std::string>({"pair-w8","silu","mul","down"}));
 for(int layer:{4,5,6}){launches.clear();m.mlp(layer);assert(launches==std::vector<std::string>({"resident-gate","silu","resident-up","mul","down"}));reject([&]{m.launch_action_gateup_pair(layer,{});});}
 launches.clear();m.mlp(7);assert(launches==std::vector<std::string>({"pair-w8","silu","mul","down"}));assert(tensor_proofs==w8proofs);
 {auto x=m;x.linear_acc32_=false;x.action_mod_body_w8_=true;x.prepare_action_gateup_pair();x.plan();assert(x.use_action_gateup_pair(32)&&x.routes[0].args[5]==16&&x.routes[0].args[6]==64&&x.subclass_layout_hash()!=m.subclass_layout_hash());auto valid=x.routes;x.routes=m.routes;reject([&]{x.mlp(7);});x.routes=valid;linear_modes.clear();launches.clear();x.mlp(7);assert(linear_modes==std::vector<bool>({false})&&launches==std::vector<std::string>({"pair-w8","silu","mul","down"}));for(int layer:{4,5,6}){launches.clear();x.mlp(layer);assert(launches==std::vector<std::string>({"resident-gate","silu","resident-up","mul","down"}));}}
 m.routes[0].args[15]=3;launches.clear();reject([&]{m.mlp(7);});assert(launches.empty());m.plan();m.layer_weights_.back().up_ws.rank=2;m.prepare_action_gateup_pair();assert(!m.action_gateup_pair_available_);m.layer_weights_.back().up_ws.rank=1;
 KernelCache::instance().available=false;m.prepare_action_gateup_pair();m.plan();launches.clear();m.mlp(0);assert(launches==std::vector<std::string>({"gate","silu","up","mul","down"}));

}
'''
    for marker, value in {'__METHODS__': '\n'.join(decl + _body(source, signature) for decl, signature in methods),
                          '__CONSTANTS__': constants, '__MANIFEST__': manifest, '__MLP__': mlp}.items():
        cpp = cpp.replace(marker, value)
    _compile_run(tmp_path, 'cold', cpp)


def test_gate_up_pair_actual_launcher_ddr_owners_grid_and_spm_boundaries(tmp_path):
    source = (ROOT / 'src/ops/rpu_linear.cpp').read_text()
    cpp = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>
#include "rpu_linear_tiling.h"
#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10{using Half=uint16_t;enum class DeviceType{PrivateUse1,CPU};}
namespace at{constexpr int kHalf=1,kChar=3,kByte=2;
struct Tensor{uint64_t address;bool present=true,contiguous=true;int dtype=kHalf,rank=2;int64_t rows=4096,cols=1024;c10::DeviceType location=c10::DeviceType::PrivateUse1;
 bool defined()const{return present;}int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}int dim()const{return rank;}int64_t size(int i)const{return i?cols:rows;}
 struct Device{c10::DeviceType value;c10::DeviceType type()const{return value;}};Device device()const{return {location};}template<class T>T*data_ptr()const{return reinterpret_cast<T*>(address);}
};}
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8303616;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int core,int off)const{assert(core==0);return 0x20000000+off;}}allocator;
#define SPM_ALLOC allocator
enum class KernelId{G05_GATE_UP_PAIRED_NVFP4_ACC16_M32N512K1024_N128,G05_GATE_UP_FP16_ACC32_M32N512K1024};constexpr auto ID=KernelId::G05_GATE_UP_FP16_ACC32_M32N512K1024;
struct KernelCache{bool available=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId)const{return available;}};
enum class GraphDdrRegisterRole{ModelWeight};enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor owner;};
struct Kernel{std::map<int,uint16_t>regs;void reset_regs(){regs.clear();}void set_regs(int i,uint16_t v){regs[i]=v;}void set_regs(int,uint32_t)=delete;};
struct Submission{std::map<int,uint16_t>regs;std::vector<GraphDdrRegisterOperandSpec>operands;};struct Writer{void write(Kernel&);};
struct RpuKernelGraph{bool staged=false,pending=false,written=false;int stages=0,lookups=0;Kernel kernel;std::vector<GraphDdrRegisterOperandSpec>operands;std::vector<Submission>submissions;
 static RpuKernelGraph&active(){static RpuKernelGraph graph;return graph;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec>specs){assert(id==ID&&!staged&&!pending&&specs.size()==2);staged=true;written=false;operands=std::move(specs);++stages;return {};}
 Kernel*get(KernelId id){assert(id==ID&&staged&&!pending&&KernelCache::instance().available);pending=true;++lookups;return &kernel;}
};
void Writer::write(Kernel&kernel){auto&g=RpuKernelGraph::active();assert(g.staged&&g.pending&&!g.written);for(auto&operand:g.operands){uint64_t v=operand.owner.address>>8;kernel.set_regs(operand.abi.lo,static_cast<uint16_t>(v));kernel.set_regs(operand.abi.hi,static_cast<uint16_t>(v>>16));}g.written=true;}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
std::vector<uint64_t>flushed;void rpu_ddr_flush(c10::Half*p){flushed.push_back(reinterpret_cast<uint64_t>(p));}
struct Queue{bool broadcast=false;void set_broadcast_mode(bool v){broadcast=v;}void enqueu_kernel(Kernel&k,std::vector<uint16_t>grid,std::vector<uint8_t>cores){auto&g=RpuKernelGraph::active();assert(broadcast&&g.staged&&g.pending&&g.written);assert(grid==std::vector<uint16_t>({12,1,1})&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));g.submissions.push_back({k.regs,g.operands});g.operands.clear();g.staged=g.pending=g.written=false;}}queue;
Queue*get_queue(int cores){assert(cores==8);return &queue;}
#define GET_QUEUE(cores) get_queue(cores)
void launch(uint32_t input,const at::Tensor&gate_weight,const at::Tensor&up_weight,uint32_t gate_output,uint32_t up_output)__BODY__
uint32_t pair(const Submission&s,int reg){return uint32_t(s.regs.at(reg))|(uint32_t(s.regs.at(reg+1))<<16);}
template<class F>void reject(F f){auto&g=RpuKernelGraph::active();auto before=g.submissions.size(),flushes=flushed.size();int stages=g.stages,lookups=g.lookups;try{f();assert(false);}catch(const std::runtime_error&){}assert(g.submissions.size()==before&&flushed.size()==flushes&&g.stages==stages&&g.lookups==lookups&&!g.staged&&!g.pending);}
int main(){
 auto&g=RpuKernelGraph::active();const uint32_t base=allocator.addr(0,0);
 for(uint32_t pass=0;pass<2;++pass){const uint32_t in=base+0x20000+pass*0x40000,q=in+65536,gate=q+32768;at::Tensor qw{0x1234567800ULL+pass*0x400000},gw{0x2345678900ULL+pass*0x800000};launch(in,qw,gw,q,gate);const auto&s=g.submissions.back();
  assert(pair(s,0)==in&&pair(s,4)==q&&pair(s,36)==gate);assert(pair(s,2)==qw.address>>8&&pair(s,32)==gw.address>>8);
  assert(pair(s,6)==0&&s.regs.at(8)==1&&s.regs.at(9)==32&&pair(s,10)==512&&s.regs.at(12)==1024&&s.regs.at(13)==64&&s.regs.at(14)==0&&s.regs.at(15)==8);
  assert(pair(s,16)==2048&&pair(s,18)==1024&&pair(s,20)==0&&pair(s,22)==0&&pair(s,24)==0&&s.regs.at(26)==2&&s.regs.at(27)==2);
  assert(s.operands.size()==2&&s.operands[0].owner.address==qw.address&&s.operands[1].owner.address==gw.address);
  for(int i=0;i<2;++i){auto&abi=s.operands[i].abi;assert(abi.role==GraphDdrRegisterRole::ModelWeight&&abi.access==GraphDdrRegisterAccess::Read&&abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi&&abi.lo==(i?32:2)&&abi.hi==(i?33:3));}
  assert(flushed[pass*2]==qw.address&&flushed[pass*2+1]==gw.address);
 }
 const at::Tensor qw{0x1234567800ULL},gw{0x2345678900ULL};const uint32_t in=base+0x20000,q=in+65536,gate=q+32768;
 launch(base,qw,gw,base+65536,base+98304);
 const uint32_t end=base+SpmAllocator::SPM_USABLE;launch(end-131072,qw,gw,end-65536,end-32768);
 assert(pair(g.submissions.back(),36)==end-32768);
 for(auto regions:{std::array<uint32_t,3>{in,in,gate},{in,q,q},{in,q,in+256},{in,q-256,gate},{in,q,gate-256},{in+1,q,gate},{base-256,q,gate},{in,q,base+SpmAllocator::SPM_USABLE-8192}})reject([&]{launch(regions[0],qw,gw,regions[1],regions[2]);});
 allocator.initialized=false;reject([&]{launch(in,qw,gw,q,gate);});allocator.initialized=true;KernelCache::instance().available=false;reject([&]{launch(in,qw,gw,q,gate);});KernelCache::instance().available=true;
 for(int which:{0,1})for(int fault=0;fault<7;++fault){auto first=qw,second=gw;auto&w=which?second:first;if(fault==0){w.present=false;}if(fault==1){w.dtype=2;}if(fault==2){w.contiguous=false;}if(fault==3){w.location=c10::DeviceType::CPU;}if(fault==4){w.rank=1;}if(fault==5){w.rows=1024;}if(fault==6){w.cols=512;}reject([&]{launch(in,first,second,q,gate);});}
}
'''
    cpp = cpp.replace('__BODY__', _body(source, 'void rpu_launch_g05_gate_up_linear_fp16_acc32_kernel('))
    _compile_run(tmp_path, 'launcher', cpp)
