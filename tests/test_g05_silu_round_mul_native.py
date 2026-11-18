"""Actual rounded SiLU/Mul admission, MLP scheduling and launch ABI on CPU doubles."""
from pathlib import Path
import re

from test_siglip_cold_profile import _body
from test_g05_q_gate_native import _compile_run

ROOT = Path(__file__).resolve().parents[1]


def test_silu_round_mul_cold_manifest_resident_priority_and_fallback(tmp_path):
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    methods = [
        ('void prepare_action_silu_round_mul()', 'void Qwen3_5Model::prepare_action_silu_round_mul('),
        ('bool use_action_silu_round_mul(int64_t rows) const', 'bool Qwen3_5Model::use_action_silu_round_mul('),
        ('std::vector<int64_t> action_silu_round_mul_arguments() const', 'std::vector<int64_t> Qwen3_5Model::action_silu_round_mul_arguments('),
        ('void launch_action_silu_round_mul(int layer,const ChunkInfo& chunk)', 'void Qwen3_5Model::launch_action_silu_round_mul('),
        ('void prepare_action_silu_high_mul()', 'void Qwen3_5Model::prepare_action_silu_high_mul('),
        ('bool use_action_silu_high_mul(int64_t rows) const', 'bool Qwen3_5Model::use_action_silu_high_mul('),
        ('std::vector<int64_t> action_silu_high_mul_arguments() const', 'std::vector<int64_t> Qwen3_5Model::action_silu_high_mul_arguments('),
        ('void launch_action_silu_high_mul(int layer,const ChunkInfo& chunk)', 'void Qwen3_5Model::launch_action_silu_high_mul('),
        ('void prepare_action_gateup_pair()', 'void Qwen3_5Model::prepare_action_gateup_pair('),
        ('bool use_action_gateup_pair(int64_t rows) const', 'bool Qwen3_5Model::use_action_gateup_pair('),
        ('bool use_action_mod_body() const', 'bool Qwen3_5Model::use_action_mod_body('),
        ('std::vector<int64_t> action_gateup_pair_arguments() const', 'std::vector<int64_t> Qwen3_5Model::action_gateup_pair_arguments('),
        ('void launch_action_gateup_pair(int layer,const ChunkInfo& chunk)', 'void Qwen3_5Model::launch_action_gateup_pair('),
        ('int64_t action_gateup_resident_layers() const', 'int64_t Qwen3_5Model::action_gateup_resident_layers('),
        ('int64_t subclass_layout_hash() const', 'int64_t Qwen3_5Model::subclass_layout_hash('),
        ('int64_t high_precision_site() const', 'int64_t Qwen3_5Model::high_precision_site('),
        ('std::vector<int64_t> high_precision_arguments() const', 'std::vector<int64_t> Qwen3_5Model::high_precision_arguments('),
    ]
    begin = source.index('{', source.index('if (use_action_gateup_pair(chunk.len))'))
    end = source.index('\n        }', begin) + len('\n        }')
    manifest = source[begin:end]
    begin = source.index('        if (use_action_silu_round_mul(chunk.len))')
    end = source.index('\n        }', begin) + len('\n        }')
    fusion_manifest = source[begin:end]
    begin = source.index('        if (use_action_silu_high_mul(chunk.len))')
    end = source.index('\n        }', begin) + len('\n        }')
    high_manifest = source[begin:end]
    begin = source.index('        if (use_silu_ && (!use_action_silu_round_mul(chunk.len)')
    end = source.index('\n        }', begin) + len('\n        }')
    base_manifest = source[begin:end]
    mlp = _body(source, 'void Qwen3_5Model::emit_mlp_and_output(')
    mlp = mlp[mlp.index('    const int64_t mlp_elems ='):mlp.index('    if (action_mlp_tail_handoff(layer_idx,chunk,true))')]
    constants = '\n'.join(re.findall(r'constexpr int64_t QWEN35_(?:TEXT_GDN_LOCAL_SLICE_SITE|ACTION_(?:DOWN_\w+|O_\w+|GATEUP_\w+|RESIDENT_\w+|Q_GATE_\w+|QKV_GATE_\w+|OWNER_CACHE_\w+|KV_OWNER_\w+|SILU_ROUND_MUL_SITE|SILU_HIGH_MUL_SITE|FUSED_NORM_SHIFT_SITE|RTC_RESIDUAL_GATE_SITE|RING_RTC_RESIDUAL_SITE|RESIDUAL_GATE_SITE)|TEXT_MLP_SILU_SITE|TEXT_GATEUP_PAIR_\w+)\s*=.*?;', source, re.S))
    constants += '\n' + '\n'.join(re.findall(
        r'constexpr int64_t QWEN35_(?:TEXT|ACTION)_HIGH_PRECISION_SITE\s*=.*?;', source, re.S))
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
enum class KernelId{RHINOVLA_SILU_HIGH_MUL,G05_GATE_UP_PAIRED_NVFP4_ACC32_M32N512K1024_N128,G05_GATE_UP_FP16_ACC16_M32N512K1024,G05_GATE_UP_PAIRED_NVFP4_ACC16_M32N512K1024_N128,G05_GATE_UP_FP16_ACC32_M32N512K1024,G05_SILU_ROUND_MUL_M32N512,G05_GATE_UP_W8A16_ACC32_M32N512K1024,G05_GATE_UP_W8A16_ACC16_M32N512K1024};
struct KernelCache{bool pair=true,fusion=true;bool high_fusion=true;int queries=0;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId id){++queries;return (id==KernelId::G05_GATE_UP_FP16_ACC16_M32N512K1024||id==KernelId::G05_GATE_UP_FP16_ACC32_M32N512K1024||id==KernelId::G05_GATE_UP_W8A16_ACC32_M32N512K1024||id==KernelId::G05_GATE_UP_W8A16_ACC16_M32N512K1024)?pair:id==KernelId::RHINOVLA_SILU_HIGH_MUL?high_fusion:fusion;}};
struct Weights{at::Tensor input_norm_w,post_norm_w,gate_w,up_w,down_w,gate_ws,up_ws,down_ws,gate_negative_zero,up_negative_zero,down_negative_zero,q_w,k_w,v_w,o_w,attn_gate_w,q_ws,k_ws,v_ws,o_ws,attn_gate_ws,gdn_q_w,gdn_k_w,gdn_v_w,gdn_in_z_w,gdn_out_w,gdn_q_ws,gdn_k_ws,gdn_v_ws,gdn_in_z_ws,gdn_out_ws,gdn_b_bg_w,gdn_a_bg_w,gdn_b_bg_ws,gdn_a_bg_ws;};
enum class FmbRouteFamily{LINEAR,ACTIVATION};enum class ValuOpType{SILU,MUL};
enum class RpuUnaryPrecision : uint8_t {BASE,HIGH};
__CONSTANTS__
struct ChunkInfo{int64_t len=32,idx=2;};
struct Route{int64_t site;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>args;int64_t invocation;};
std::vector<std::string>launches;std::vector<std::pair<uint64_t,uint64_t>>weights_seen;
void rpu_launch_g05_gate_up_linear_fp16_acc32_kernel(uint32_t in,const at::Tensor&gate,const at::Tensor&up,uint32_t go,uint32_t uo){assert(in==0x10000&&go==0x20000&&uo==0x28000&&gate.owner!=up.owner);launches.push_back("pair");weights_seen.push_back({gate.owner,up.owner});}
void rpu_launch_g05_gate_up_linear_fp16_acc16_kernel(uint32_t in,const at::Tensor&gate,const at::Tensor&up,uint32_t go,uint32_t uo){rpu_launch_g05_gate_up_linear_fp16_acc32_kernel(in,gate,up,go,uo);launches.back()="pair-fp16-acc16";}
void rpu_launch_eltwise_unary_spm_kernel(uint32_t in,uint32_t out,int64_t n,ValuOpType type,bool,int cores,RpuUnaryPrecision precision=RpuUnaryPrecision::BASE){assert(in==0x20000&&out==in&&n==16384&&type==ValuOpType::SILU&&cores==8);launches.push_back(precision==RpuUnaryPrecision::HIGH?"high-silu":"silu");}
void rpu_launch_eltwise_binary_spm_kernel(uint32_t in,uint32_t rhs,uint32_t out,int64_t n,ValuOpType type,c10::Half,int cores){assert(in==0x20000&&rhs==0x28000&&out==in&&n==16384&&type==ValuOpType::MUL&&cores==8);launches.push_back("mul");}
void rpu_launch_g05_silu_round_mul_spm_kernel(uint32_t gate,uint32_t up){assert(gate==0x20000&&up==0x28000);launches.push_back("round-mul");}
void rpu_launch_rhinovla_silu_high_mul_spm_kernel(uint32_t gate,uint32_t up,uint32_t output,int64_t elements,int cores){assert(gate==0x20000&&up==0x28000&&output==gate&&elements==16384&&cores==8);launches.push_back("high-mul");}
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

 // NVFP4 is disabled in this FP16/W8 caller fixture.
 struct Nvfp4Stub {bool enabled()const{return false;}std::vector<int>values;};
 Nvfp4Stub action_nvfp4_;
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
 bool action_silu_round_mul_available_=false;
 bool high_precision_=false;
 bool action_silu_high_mul_available_=false;
 bool has_qk_norm_=true;
 int64_t gdn_nvh_=0,gdn_dk_=0,gdn_dv_=0;
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
 bool use_action_gateup_resident(int64_t rows)const{return resident&&use_action_mod_body()&&rows==32;}
 uint32_t addr(int,const char*n)const{std::string name(n);return name=="residual1"?0x10000:name=="gate"?0x20000:name=="up"?0x28000:0x30000;}
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t>args,int64_t inv){auto i=std::find_if(routes.begin(),routes.end(),[&](auto&r){return r.site==site&&r.invocation==inv;});TORCH_CHECK(i!=routes.end()&&i->family==family&&i->selector==selector&&i->flags==flags&&i->args==args);}
 void launch_linear(int64_t,uint32_t,const at::Tensor&w,uint32_t,int64_t,int64_t,int64_t,int,int64_t,int,const at::Tensor&scale,int=-1,int=-1,const at::Tensor& negative_zero={}){assert(negative_zero.owner==0);if(w.dtype==at::kChar)assert(scale.rank==1&&scale.owner!=0);launches.push_back(w.tag);}
 void launch_action_gateup_resident(int layer,int projection,uint32_t out){assert(layer>=1&&layer<=action_gateup_resident_layers()&&use_action_gateup_resident(32));assert(out==(projection?0x28000u:0x20000u));launches.push_back(projection?"resident-up":"resident-gate");}
 __METHODS__
 void plan(ChunkInfo chunk={}){const int plan=0,physical_len=chunk.len,logical_len=32,position=0;struct{int batch_size=1;}layout;struct{std::vector<Route>routes;}manifest;if(use_action_gateup_pair(chunk.len))__MANIFEST__ __FUSION_MANIFEST__ __HIGH_MANIFEST__ __BASE_MANIFEST__ routes=manifest.routes;}
 void mlp(int layer_idx,ChunkInfo chunk={}){const auto&lw=layer_weights_[layer_idx];int64_t seq_len=chunk.len,h=hidden_size();__MLP__ (void)residual_buf;}
};
template<class F>void reject(F f){auto before=launches;try{f();assert(false);}catch(const std::runtime_error&){}assert(launches==before);}
int main(){
 Model m;auto&cache=KernelCache::instance();m.prepare_action_gateup_pair();m.prepare_action_silu_round_mul();const int cold_proofs=tensor_proofs;assert(m.action_silu_round_mul_available_&&m.action_gateup_pair_available_&&cold_proofs>=48&&cache.queries==2);
 {auto x=m;x.high_precision_=true;x.prepare_action_silu_round_mul();assert(x.action_gateup_pair_available_&&!x.action_silu_round_mul_available_&&!x.use_action_silu_round_mul(32));}
 auto has=[&](int64_t site){return std::any_of(m.routes.begin(),m.routes.end(),[&](auto&r){return r.site==site;});};
 m.plan();assert(!m.use_action_silu_round_mul(32)&&!has(QWEN35_ACTION_SILU_ROUND_MUL_SITE)&&has(QWEN35_TEXT_MLP_SILU_SITE));m.mlp(1);assert(launches==std::vector<std::string>({"gate","silu","up","mul","down"}));
 m.action_step_active_=m.action_loop_active_=true;m.action_num_steps_=10;m.plan();const auto fused_hash=m.subclass_layout_hash();
 assert(m.action_silu_round_mul_arguments()==std::vector<int64_t>({1,32,512,8,16384,2048,8,134,3}));assert(has(QWEN35_ACTION_SILU_ROUND_MUL_SITE)&&has(QWEN35_TEXT_MLP_SILU_SITE));
 for(int layer=0;layer<24;++layer){launches.clear();m.mlp(layer);if(layer>=1&&layer<=3){assert(launches==std::vector<std::string>({"resident-gate","silu","resident-up","mul","down"}));reject([&]{m.launch_action_silu_round_mul(layer,{});});}else assert(launches==std::vector<std::string>({"pair","round-mul","down"}));}
 assert(cache.queries==2&&tensor_proofs==cold_proofs);
 m.complete=false;reject([&]{m.launch_action_silu_round_mul(0,{});});m.complete=true;
 for(auto&r:m.routes)if(r.site==QWEN35_ACTION_SILU_ROUND_MUL_SITE)r.args[7]=135;
 reject([&]{m.launch_action_silu_round_mul(0,{});});launches.clear();try{m.mlp(0);assert(false);}catch(const std::runtime_error&){}assert(launches==std::vector<std::string>({"pair"}));m.plan();
 for(int layer:{-1,24})reject([&]{m.launch_action_silu_round_mul(layer,{});});
 reject([&]{m.launch_action_silu_round_mul(0,{31,2});});reject([&]{m.launch_action_silu_round_mul(0,{32,3});});
 m.resident=false;reject([&]{m.launch_action_silu_round_mul(0,{});});m.plan();assert(m.action_silu_round_mul_arguments().back()==0&&!has(QWEN35_TEXT_MLP_SILU_SITE));launches.clear();m.mlp(1);assert(launches==std::vector<std::string>({"pair","round-mul","down"}));
 m.resident=true;reject([&]{m.launch_action_silu_round_mul(0,{});});m.plan();
 // Missing rounded payload retains pair + original rounded SiLU and MUL.
 cache.fusion=false;m.prepare_action_silu_round_mul();assert(!m.action_silu_round_mul_available_&&m.action_gateup_pair_available_&&m.subclass_layout_hash()!=fused_hash);m.plan();assert(!has(QWEN35_ACTION_SILU_ROUND_MUL_SITE)&&has(QWEN35_TEXT_MLP_SILU_SITE));launches.clear();m.mlp(0);assert(launches==std::vector<std::string>({"pair","silu","mul","down"}));
 cache.fusion=true;m.prepare_action_silu_round_mul();assert(m.action_silu_round_mul_available_&&m.subclass_layout_hash()==fused_hash);
 // Fusion cannot bypass its Gate/Up metadata proof or missing pair payload.
 cache.pair=false;m.prepare_action_gateup_pair();m.prepare_action_silu_round_mul();assert(!m.action_gateup_pair_available_&&!m.action_silu_round_mul_available_);m.plan();launches.clear();m.mlp(0);assert(launches==std::vector<std::string>({"gate","silu","up","mul","down"}));
 cache.pair=true;m.prepare_action_gateup_pair();m.prepare_action_silu_round_mul();m.layer_weights_.back().up_w.dtype=2;m.prepare_action_gateup_pair();m.prepare_action_silu_round_mul();assert(!m.action_silu_round_mul_available_);m.layer_weights_.back().up_w.dtype=at::kHalf;m.prepare_action_gateup_pair();m.prepare_action_silu_round_mul();
 for(int fault=0;fault<4;++fault){auto x=m;if(fault==0)x.action_num_steps_=1;if(fault==1)x.action_loop_active_=false;if(fault==2)x.action_step_active_=false;if(fault==3)x.action_len_=16;assert(!x.use_action_silu_round_mul(32));x.plan();assert(std::none_of(x.routes.begin(),x.routes.end(),[](auto&r){return r.site==QWEN35_ACTION_SILU_ROUND_MUL_SITE;}));}
 m.plan({31,2});assert(!has(QWEN35_ACTION_SILU_ROUND_MUL_SITE));
 m.use_silu_=false;assert(!m.use_action_silu_round_mul(32));m.prepare_action_gateup_pair();m.prepare_action_silu_round_mul();m.plan();assert(!has(QWEN35_TEXT_MLP_SILU_SITE)&&!has(QWEN35_ACTION_SILU_ROUND_MUL_SITE));m.resident=false;launches.clear();m.mlp(0);assert(launches==std::vector<std::string>({"gate","up","mul","down"}));
 // The neutral Action activation follows the genuine FP16 ACC16 pair.
 {Model x;x.linear_acc32_=false;x.action_mod_body_fp16_=true;x.resident=false;x.action_step_active_=x.action_loop_active_=true;x.action_num_steps_=10;
  for(auto&w:x.layer_weights_)w.gate_ws.present=w.up_ws.present=false;
  x.prepare_action_gateup_pair();x.prepare_action_silu_round_mul();x.plan();assert(x.use_action_silu_round_mul(32)&&x.action_gateup_pair_arguments()[5]==16);
  launches.clear();x.mlp(23);assert(launches==std::vector<std::string>({"pair-fp16-acc16","round-mul","down"}));
 }
 // W8 uses the actual helper to exclude all six resident layers from pair/fusion.
 Model w8;cache.pair=cache.fusion=true;w8.action_gateup_resident_w8_=true;
 for(auto&w:w8.layer_weights_){w.gate_w.dtype=w.up_w.dtype=w.down_w.dtype=at::kChar;w.gate_ws.rank=w.up_ws.rank=w.down_ws.rank=1;w.gate_ws.owner=w.gate_w.owner+100;w.up_ws.owner=w.up_w.owner+200;w.down_ws.owner=301;}
 w8.prepare_action_gateup_pair();w8.prepare_action_silu_round_mul();w8.action_step_active_=w8.action_loop_active_=true;w8.action_num_steps_=10;w8.plan();
 assert(w8.action_gateup_resident_layers()==6&&w8.action_gateup_pair_arguments()[15]==6&&w8.action_silu_round_mul_arguments().back()==6);
 for(int layer:{4,5,6}){launches.clear();w8.mlp(layer);assert(launches==std::vector<std::string>({"resident-gate","silu","resident-up","mul","down"}));reject([&]{w8.launch_action_gateup_pair(layer,{});});reject([&]{w8.launch_action_silu_round_mul(layer,{});});}
 launches.clear();w8.mlp(7);assert(launches==std::vector<std::string>({"pair-w8","round-mul","down"}));
 {auto x=w8;x.linear_acc32_=false;x.action_mod_body_w8_=true;x.prepare_action_gateup_pair();x.prepare_action_silu_round_mul();x.plan();assert(x.use_action_silu_round_mul(32)&&x.action_gateup_pair_arguments()[5]==16&&x.action_gateup_pair_arguments()[6]==64&&x.action_silu_round_mul_arguments()==w8.action_silu_round_mul_arguments());linear_modes.clear();launches.clear();x.mlp(7);assert(linear_modes==std::vector<bool>({false})&&launches==std::vector<std::string>({"pair-w8","round-mul","down"}));for(int layer:{4,5,6}){launches.clear();x.mlp(layer);assert(launches==std::vector<std::string>({"resident-gate","silu","resident-up","mul","down"}));reject([&]{x.launch_action_silu_round_mul(layer,{});});}}
 for(auto&r:w8.routes){if(r.site==QWEN35_ACTION_GATEUP_PAIR_W8_SITE)r.args[15]=3;}launches.clear();reject([&]{w8.mlp(7);});assert(launches.empty());w8.plan();
 for(auto&r:w8.routes)if(r.site==QWEN35_ACTION_SILU_ROUND_MUL_SITE)r.args.back()=3;
 reject([&]{w8.launch_action_silu_round_mul(7,{});});try{w8.mlp(7);assert(false);}catch(const std::runtime_error&){}assert(launches==std::vector<std::string>({"pair-w8"}));w8.plan();
 cache.fusion=false;w8.prepare_action_silu_round_mul();w8.plan();launches.clear();w8.mlp(7);assert(launches==std::vector<std::string>({"pair-w8","silu","mul","down"}));
 launches.clear();w8.mlp(6);assert(launches==std::vector<std::string>({"resident-gate","silu","resident-up","mul","down"}));
 cache.pair=false;w8.prepare_action_gateup_pair();w8.prepare_action_silu_round_mul();w8.plan();launches.clear();w8.mlp(7);assert(launches==std::vector<std::string>({"gate","silu","up","mul","down"}));

 // HIGH uses its own optional cold capability and consumed activation route.
 // All four FP16/W8 accumulation combinations keep their original pair mode;
 // resident layers retain Gate -> HIGH SiLU -> Up -> ordinary FP16 MUL.
 cache.pair=cache.fusion=cache.high_fusion=true;
 for(bool quantized:{false,true})for(bool acc32:{false,true}){
  Model x;x.high_precision_=true;x.linear_acc32_=acc32;
  x.action_step_active_=x.action_loop_active_=true;x.action_num_steps_=10;
  x.action_mod_body_w8_=quantized;x.action_mod_body_fp16_=!quantized;
  x.resident=quantized||acc32;x.action_gateup_resident_w8_=quantized;
  for(auto&w:x.layer_weights_){
   if(quantized){w.gate_w.dtype=w.up_w.dtype=w.down_w.dtype=at::kChar;w.gate_ws.rank=w.up_ws.rank=w.down_ws.rank=1;w.gate_ws.owner=w.gate_w.owner+100;w.up_ws.owner=w.up_w.owner+200;w.down_ws.owner=301;}
   else if(!acc32)w.gate_ws.present=w.up_ws.present=false;
  }
  x.prepare_action_gateup_pair();x.prepare_action_silu_round_mul();x.prepare_action_silu_high_mul();x.plan();
  assert(x.action_gateup_pair_available_&&!x.action_silu_round_mul_available_&&x.use_action_silu_high_mul(32));
  auto route=std::find_if(x.routes.begin(),x.routes.end(),[](const auto&r){return r.site==QWEN35_ACTION_SILU_HIGH_MUL_SITE;});
  assert(route!=x.routes.end()&&route->family==FmbRouteFamily::ACTIVATION&&route->args[16]==(acc32?32:16));
  assert(std::none_of(x.routes.begin(),x.routes.end(),[](const auto&r){return r.site==QWEN35_ACTION_SILU_ROUND_MUL_SITE;}));
  const auto before=x.subclass_layout_hash();const int hot_queries=cache.queries;
  cache.high_fusion=false; // A REPLAY never resamples payload availability.
  for(int layer=0;layer<24;++layer){
   launches.clear();x.mlp(layer);
   if(x.resident&&layer>=1&&layer<=x.action_gateup_resident_layers()){
    assert(launches==std::vector<std::string>({"resident-gate","high-silu","resident-up","mul","down"}));
    reject([&]{x.launch_action_silu_high_mul(layer,{});});
   }else assert(launches==std::vector<std::string>({quantized?"pair-w8":acc32?"pair":"pair-fp16-acc16","high-mul","down"}));
  }
  assert(cache.queries==hot_queries&&x.subclass_layout_hash()==before);
  const auto good=x.routes;const size_t argument_count=route->args.size();
  for(size_t index=0;index<argument_count;++index){
   auto r=std::find_if(x.routes.begin(),x.routes.end(),[](const auto&r){return r.site==QWEN35_ACTION_SILU_HIGH_MUL_SITE;});
   ++r->args[index];reject([&]{x.launch_action_silu_high_mul(0,{});});x.routes=good;
  }
  x.complete=false;reject([&]{x.launch_action_silu_high_mul(0,{});});x.complete=true;
  for(int layer:{-1,24})reject([&]{x.launch_action_silu_high_mul(layer,{});});
  reject([&]{x.launch_action_silu_high_mul(0,{31,2});});
  reject([&]{x.launch_action_silu_high_mul(0,{32,3});});
  x.prepare_action_silu_high_mul();x.plan();assert(!x.action_silu_high_mul_available_&&x.subclass_layout_hash()!=before);
  launches.clear();x.mlp(0);assert(launches==std::vector<std::string>({quantized?"pair-w8":acc32?"pair":"pair-fp16-acc16","high-silu","mul","down"}));
  cache.high_fusion=true;x.prepare_action_silu_high_mul();x.plan();assert(x.subclass_layout_hash()==before);
  for(int fault=0;fault<5;++fault){auto bad=x;if(fault==0)bad.action_mode_=false;if(fault==1)bad.use_silu_=false;if(fault==2)bad.has_gdn_=true;if(fault==3)bad.action_gateup_pair_available_=false;if(fault==4)bad.high_precision_=false;bad.prepare_action_silu_high_mul();assert(!bad.action_silu_high_mul_available_);}
 }
 // A dormant HIGH capability never changes the BASE layout identity.
 Model base;const auto base_hash=base.subclass_layout_hash();base.action_silu_high_mul_available_=true;base.action_newton_norm_shift_available_=true;assert(base.subclass_layout_hash()==base_hash);

}

'''
    for marker, value in {'__METHODS__': '\n'.join(decl + _body(source, signature) for decl, signature in methods),
                          '__CONSTANTS__': constants, '__MANIFEST__': manifest, '__MLP__': mlp,
                          '__FUSION_MANIFEST__': fusion_manifest, '__HIGH_MANIFEST__': high_manifest, '__BASE_MANIFEST__': base_manifest}.items():
        cpp = cpp.replace(marker, value)
    _compile_run(tmp_path, 'cold', cpp)


def test_silu_round_mul_actual_launcher_spm_and_base_register_abi(tmp_path):
    source = (ROOT / 'src/ops/rpu_eltwise_unary.cpp').read_text()
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
#include "rpu_eltwise.h"
#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10{struct Half{uint16_t x;explicit Half(double value):x(0x3c00){assert(value==1.0);}};}
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8303616;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int core,int off)const{assert(core==0);return 0x20000000+off;}}allocator;
#define SPM_ALLOC allocator
enum class KernelId{G05_GATE_UP_PAIRED_NVFP4_ACC16_M32N512K1024_N128,G05_SILU_ROUND_MUL_M32N512};constexpr auto ID=KernelId::G05_SILU_ROUND_MUL_M32N512;
struct KernelCache{bool available=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId id)const{assert(id==ID);return available;}};
struct Kernel{std::map<int,uint16_t>regs;int resets=0;void reset_regs(){regs.clear();++resets;}void set_regs(int i,uint16_t v){regs[i]=v;}void set_regs(int,uint32_t)=delete;}kernel;
int lookups=0;bool lookup_failure=false,pending=false;
Kernel*get(KernelId id){assert(id==ID&&!pending);++lookups;if(lookup_failure)return nullptr;pending=true;return &kernel;}
#define GET_KERNEL(id) get(id)
std::vector<std::map<int,uint16_t>>submissions;
struct Queue{bool broadcast=false;void set_broadcast_mode(bool v){broadcast=v;}void enqueu_kernel(Kernel&k,std::vector<uint16_t>grid,std::vector<uint8_t>cores){assert(pending&&broadcast&&grid==std::vector<uint16_t>({8,1,1})&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));submissions.push_back(k.regs);pending=false;}}queue;
Queue*get_queue(int cores){assert(cores==8);return &queue;}
#define GET_QUEUE(cores) get_queue(cores)
void launch(uint32_t gate,uint32_t up)__BODY__
uint32_t pair(const std::map<int,uint16_t>&regs,int i){return uint32_t(regs.at(i))|(uint32_t(regs.at(i+1))<<16);}
template<class F>void reject(F f,bool may_lookup=false){auto count=submissions.size();int before=lookups,resets=kernel.resets;try{f();assert(false);}catch(const std::runtime_error&){}assert(submissions.size()==count&&kernel.resets==resets&&!pending);assert(lookups==before+(may_lookup?1:0));}
int main(){
 const uint32_t base=allocator.addr(0,0),end=base+SpmAllocator::SPM_USABLE;
 for(auto addresses:{std::pair<uint32_t,uint32_t>{base,base+32768},{base+65536,base+98304},{end-32768,end-65536}}){launch(addresses.first,addresses.second);const auto&r=submissions.back();assert(r.at(0)==2048&&r.at(1)==2048);assert(pair(r,4)==addresses.first&&pair(r,6)==addresses.first&&pair(r,12)==addresses.second&&pair(r,8)==4096);for(int i:{56,57,58})assert(r.at(i)==0x3c00);assert(r.at(59)==0&&r.at(60)==0&&r.at(61)==1);assert(r.at(62)==((31<<9)|FP16_RAB_DATA_TYPE_CFG)&&r.at(63)==134);}
 for(auto addresses:{std::pair<uint32_t,uint32_t>{base,base},{base,base+32512},{base+32512,base},{base+1,base+32768},{base,base+32769},{base-256,base+32768},{base,end-32512},{end-32512,base},{0xffff8000u,base}})reject([&]{launch(addresses.first,addresses.second);});
 allocator.initialized=false;reject([&]{launch(base,base+32768);});allocator.initialized=true;
 KernelCache::instance().available=false;reject([&]{launch(base,base+32768);});KernelCache::instance().available=true;
 lookup_failure=true;reject([&]{launch(base,base+32768);},true);lookup_failure=false;
 launch(base+32768,base);assert(pair(submissions.back(),4)==base+32768&&pair(submissions.back(),12)==base);

}
'''
    cpp = cpp.replace('__BODY__', _body(source, 'void rpu_launch_g05_silu_round_mul_spm_kernel('))
    _compile_run(tmp_path, 'launcher', cpp)
