"""Actual joint W8 attention cold/descriptor/caller bodies on CPU owner doubles."""
from pathlib import Path
import re

from test_g05_q_gate_native import _compile_run
from test_g05_vision_owner_model import _block

ROOT = Path(__file__).resolve().parents[1]

_OWNER_CACHE_STUBS = '''
 bool use_action_owner_cache(int64_t,int64_t)const{return false;}
 void prepare_action_owner_cache(){action_owner_cache_available_=false;}
 std::vector<int64_t>action_qkv_gate_owner_only_arguments()const{return {};}
 std::vector<int64_t>action_k_owner_norm_rope_arguments()const{return {};}
 void launch_action_qkv_gate_owner_only(int,const ChunkInfo&){assert(false);}
'''


def _qkv_gate_model_cpp():
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    header = (ROOT / 'src/fused/rpu_qwen3_5_model.h').read_text()
    methods = []
    for stem in ('q_gate', 'kv_owner', 'qkv_gate'):
        methods.extend([
            (f'void prepare_action_{stem}()', f'void Qwen3_5Model::prepare_action_{stem}('),
            (f'bool use_action_{stem}(int64_t rows) const', f'bool Qwen3_5Model::use_action_{stem}('),
            (f'std::vector<int64_t> action_{stem}_arguments() const', f'std::vector<int64_t> Qwen3_5Model::action_{stem}_arguments('),
            (f'void launch_action_{stem}(int layer,const ChunkInfo& chunk)', f'void Qwen3_5Model::launch_action_{stem}('),
        ])
    methods += [
        ('bool use_action_mod_body() const', 'bool Qwen3_5Model::use_action_mod_body('),
        ('int64_t action_gateup_resident_layers() const', 'int64_t Qwen3_5Model::action_gateup_resident_layers('),
        ('int64_t subclass_layout_hash() const', 'int64_t Qwen3_5Model::subclass_layout_hash('),
    ]
    attention = _block(source, 'void Qwen3_5Model::build_full_attention(')
    caller = attention[attention.index('    const bool owner_cache ='):attention.index('    // Phase 3: QK RMSNorm')]
    manifest = _block(source, 'FmbPhysicalExecutionManifest Qwen3_5Model::physical_manifest_for_candidate(')
    manifest = manifest[manifest.index('        if (use_action_owner_cache(chunk.len,position))'):manifest.index('        if (use_action_gateup_resident(chunk.len))')]
    install = _block(source, 'void Qwen3_5Model::set_action_io_weights(')
    install = install[install.index('    prepare_action_gateup_pair();'):-1]
    reset = _block(source, 'void Qwen3_5Model::set_weights(')
    reset = reset[reset.index('    action_mode_ = false;'):reset.index('    action_rtc_prefix_len_ = 0;')]
    constants = '\n'.join(re.findall(r'constexpr int64_t QWEN35_(?:TEXT_GDN_LOCAL_SLICE_SITE|ACTION_[A-Z0-9_]+|TEXT_GATEUP_PAIR(?:_W8)?_SITE)\s*=.*?;', source, re.S))
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
#define DWIDTH 2
int metadata=0,replica_proofs=0;
namespace c10{enum class DeviceType{PrivateUse1,CPU};}
namespace at{constexpr int kHalf=1,kChar=3,kByte=2;
struct Tensor{
 bool present=true,contiguous=true,replicated=true;int dtype=kHalf,rank=2;int64_t rows=2048,cols=1024,owner=0;std::string tag;
 c10::DeviceType location=c10::DeviceType::PrivateUse1;
 bool defined()const{++metadata;return present;}int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}
 struct Device{c10::DeviceType value;c10::DeviceType type()const{return value;}};Device device()const{return {location};}
 int dim()const{return rank;}int64_t size(int i)const{return i?cols:rows;}
};}
struct Weights __WEIGHTS__;
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>&v,const at::Tensor&t){v.insert(v.end(),{t.owner,t.dtype,t.rank,t.rows,t.cols});}
void append_kvinsert_cost_scalar_identity(std::vector<int64_t>&v,double x){v.push_back(static_cast<int64_t>(x*1e12));}
namespace detail{int64_t layout_mix(int64_t a,int64_t b){return uint64_t(a)*131+uint64_t(b);}}
enum class KernelId{G05_GATE_UP_PAIRED_NVFP4_ACC16_M32N512K1024_N128,G05_Q_GATE_FP16_ACC32_M32N256K1024,G05_Q_GATE_W8A16_ACC32_M32N256K1024,G05_Q_GATE_W8A16_ACC16_M32N256K1024,G05_KV_OWNER_FP16_ACC32_M32N256K1024,G05_KV_OWNER_W8A16_ACC32_M32N256K1024,G05_KV_OWNER_W8A16_ACC16_M32N256K1024,G05_KV_OWNER_FANOUT_M32D256,G05_QKV_GATE_OWNER_W8A16_ACC32_M32N256K1024,G05_QKV_GATE_OWNER_W8A16_ACC16_M32N256K1024};
struct KernelCache{bool q=true,kv=true,fanout=true,joint=true;bool acc16_available=true;int queries=0;static KernelCache&instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId id){++queries;if(!acc16_available&&(id==KernelId::G05_Q_GATE_W8A16_ACC16_M32N256K1024||id==KernelId::G05_KV_OWNER_W8A16_ACC16_M32N256K1024||id==KernelId::G05_QKV_GATE_OWNER_W8A16_ACC16_M32N256K1024))return false;if(id==KernelId::G05_Q_GATE_FP16_ACC32_M32N256K1024||id==KernelId::G05_Q_GATE_W8A16_ACC32_M32N256K1024||id==KernelId::G05_Q_GATE_W8A16_ACC16_M32N256K1024)return q;if(id==KernelId::G05_KV_OWNER_FANOUT_M32D256)return fanout;if(id==KernelId::G05_QKV_GATE_OWNER_W8A16_ACC32_M32N256K1024||id==KernelId::G05_QKV_GATE_OWNER_W8A16_ACC16_M32N256K1024)return joint;return kv;}};
// This is the immutable byte-replica proof boundary, already tested by its
// own native helper/probe. Here we verify the model reuses its cached result.
bool rpu_g05_kv_weight_is_replicated_fp16(const at::Tensor&w){++replica_proofs;return w.present&&w.dtype==at::kHalf&&w.replicated;}
bool rpu_g05_kv_weight_is_replicated_w8a16(const at::Tensor&w,const at::Tensor&s){++replica_proofs;return w.present&&w.dtype==at::kChar&&w.rank==2&&w.rows==2048&&w.cols==1024&&w.replicated&&s.present&&s.dtype==at::kHalf&&s.rank==1&&s.rows==2048&&s.replicated;}
enum class FmbRouteFamily{LINEAR,ROPE,KV_INSERT,OTHER};
__CONSTANTS__
struct ChunkInfo{int64_t len=32,idx=2,offset=0;};
struct Route{int64_t site;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>args;int64_t invocation;};
std::vector<std::string>launches;std::vector<int64_t>joint_owners;
void rpu_launch_g05_q_gate_linear_fp16_acc32_kernel(uint32_t,const at::Tensor&,const at::Tensor&,uint32_t,uint32_t){launches.push_back("qgate-fp16");}
std::vector<bool>linear_modes;
void rpu_launch_g05_q_gate_linear_w8a16_acc32_kernel(uint32_t,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,uint32_t,uint32_t,bool linear_acc32=true){linear_modes.push_back(linear_acc32);launches.push_back("qgate-w8");}
void rpu_launch_g05_kv_owner_linear_fp16_acc32_kernel(uint32_t,const at::Tensor&,const at::Tensor&,uint32_t,uint32_t){launches.push_back("kv-fp16");}
void rpu_launch_g05_kv_owner_linear_w8a16_acc32_kernel(uint32_t,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,uint32_t,uint32_t,bool linear_acc32=true){linear_modes.push_back(linear_acc32);launches.push_back("kv-w8");}
void rpu_launch_g05_qkv_gate_owner_linear_w8a16_acc32_kernel(uint32_t input,const at::Tensor&q,const at::Tensor&g,const at::Tensor&k,const at::Tensor&v,const at::Tensor&qs,const at::Tensor&gs,const at::Tensor&ks,const at::Tensor&vs,uint32_t qo,uint32_t go,uint32_t ko,uint32_t vo,bool linear_acc32=true){linear_modes.push_back(linear_acc32);
 assert(input==0x10000&&qo==0x20000&&go==0x24000&&ko==0x28000&&vo==0x2c000);
 assert(q.dtype==at::kChar&&g.dtype==at::kChar&&k.dtype==at::kChar&&v.dtype==at::kChar);
 assert(g.owner==q.owner+1&&k.owner==q.owner+2&&v.owner==q.owner+3);
 assert(qs.owner==q.owner+10000&&gs.owner==g.owner+10000&&ks.owner==k.owner+10000&&vs.owner==v.owner+10000);
 joint_owners.push_back(q.owner);launches.push_back("joint");
}
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
 bool action_mode_=true,linear_acc32_=true,has_gdn_=false,use_silu_=true;
 bool action_step_active_=false,action_loop_active_=false,complete=true,action_rtc_active_=false;
 bool action_owner_cache_available_=false;int64_t position=245;
 bool action_q_gate_available_=false,action_kv_owner_available_=false,action_qkv_gate_available_=false;
 bool action_q_gate_w8_=false,action_kv_owner_w8_=false,action_gateup_resident_w8_=false,action_gateup_pair_nvfp4_=false,action_gateup_pair_w8_=false,text_gateup_pair_w8_=false;
 bool action_gateup_resident_available_=false,action_norm_shift_available_=false,action_gated_residual_available_=false,action_gateup_pair_available_=false,action_silu_round_mul_available_=false,text_gateup_pair_available_=false;
 bool fast_replay_enabled_=false,has_qk_norm_=true,has_mrope_=true;
 int64_t action_len_=32,action_num_steps_=1,layers=24,h=1024,intermediate=4096,cores=8,heads=8,kv=8,hd=256,tp=8,chunk_size_cap_=0,sdpa_kernel_=0;double eps_=1e-6;
 std::vector<Weights>layer_weights_{24};std::vector<uint8_t>layer_is_full_=std::vector<uint8_t>(24,1);std::vector<Route>routes;std::vector<at::Tensor>action_gateup_resident_,action_o_resident_;
 at::Tensor final_norm_w_,action_input_w_,action_input_b_,action_output_w_,action_output_b_;int invalidations=0;
 explicit Model(bool w8=true){for(int layer=0;layer<24;++layer){auto&w=layer_weights_[layer];at::Tensor*ws[]={&w.q_w,&w.attn_gate_w,&w.k_w,&w.v_w};at::Tensor*ss[]={&w.q_ws,&w.attn_gate_ws,&w.k_ws,&w.v_ws};const char*tags[]={"q","gate","k","v"};for(int p=0;p<4;++p){ws[p]->owner=4*layer+p+1;ws[p]->dtype=w8?at::kChar:at::kHalf;ws[p]->tag=tags[p];ss[p]->owner=ws[p]->owner+10000;ss[p]->rank=1;}}}
 int64_t num_layers()const{return layers;}int64_t hidden_size()const{return h;}int64_t intermediate_size()const{return intermediate;}int64_t num_cores()const{return cores;}int64_t attn_tp()const{return tp;}int64_t mlp_tp()const{return tp;}int64_t num_q_heads()const{return heads;}int64_t num_kv_heads()const{return kv;}int64_t head_dim()const{return hd;}
 Model&ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}
 uint32_t addr(int,const char*n)const{std::string x(n);return x=="input_norm"?0x10000:x=="q"?0x20000:x=="attn_gate"?0x24000:x=="k"?0x28000:0x2c000;}
 void consume_physical_route(FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t>args,int64_t inv){auto it=std::find_if(routes.begin(),routes.end(),[&](const auto&r){return r.site==site&&r.invocation==inv;});TORCH_CHECK(it!=routes.end()&&it->family==f&&it->selector==selector&&it->flags==flags&&it->args==args);}
 void launch_linear(int64_t,uint32_t,const at::Tensor&w,uint32_t,int64_t,int64_t,int64_t,int,int64_t,int,const at::Tensor&scale,int=-1,int=-1,const at::Tensor& negative_zero={}){assert(negative_zero.owner==0);if(w.dtype==at::kChar){assert(scale.owner==w.owner+10000);}launches.push_back(w.tag);}
 std::vector<int64_t>cold_topology_arguments()const{return {cores};}
 bool use_text_gateup_pair(int64_t)const{return false;}bool use_action_gateup_resident(int64_t)const{return false;}bool use_action_gateup_pair(int64_t)const{return false;}bool use_action_silu_round_mul(int64_t)const{return false;}
 // The full 24x8 cold dtype proof is executed in test_g05_action_native_cold.
 void prepare_action_gateup_resident(){action_mod_body_w8_=layer_weights_[0].q_w.dtype==at::kChar;}void prepare_action_gateup_pair(){}void prepare_action_silu_round_mul(){}void prepare_action_o_resident(){}void invalidate_model_state(){++invalidations;}
 __OWNER_CACHE_STUBS__
 __METHODS__
 std::vector<int64_t>kvinsert_cost_weight_identity()const __IDENTITY__
 void install(){__INSTALL__}
 void reset(){__RESET__}
 void plan(ChunkInfo chunk={}){struct{std::vector<Route>routes;}manifest;__MANIFEST__ routes=manifest.routes;}
 void phase2(int layer_idx,ChunkInfo chunk={}){const auto&lw=layer_weights_[layer_idx];int64_t seq_len=chunk.len,nq=num_q_heads(),nkv=num_kv_heads(),hd=head_dim(),h=hidden_size(),tp=attn_tp();__CALLER__}
 void loop(){action_step_active_=action_loop_active_=true;action_num_steps_=10;plan();}
};
template<class F>void reject(F f){auto before=launches;try{f();assert(false);}catch(const std::runtime_error&){}assert(launches==before);}
int main(){
 auto&cache=KernelCache::instance();Model m;m.install();assert(m.invalidations==1&&m.action_qkv_gate_available_&&replica_proofs==48);
 // The new admission reuses both old proofs; it cannot scan or repack weights.
 int proof_snapshot=replica_proofs,metadata_snapshot=metadata;m.prepare_action_qkv_gate();assert(replica_proofs==proof_snapshot&&metadata==metadata_snapshot);
 assert(!m.use_action_qkv_gate(32));m.plan();assert(m.routes.empty());launches.clear();m.phase2(23);assert(launches==std::vector<std::string>({"q","k","v","gate"}));
 m.loop();assert(m.routes.size()==1&&m.routes[0].site==2447844358394284528LL&&m.routes[0].selector==1&&m.routes[0].flags==0&&m.routes[0].invocation==2);
 assert(m.routes[0].args==std::vector<int64_t>({2,32,256,1024,8,32,96,48,512,6,18,2048,512,2,32,40,48,20,34,42,50,4,2,0,4,2,6,16384,1,4,36,44,52,1}));
 auto identity=m.kvinsert_cost_weight_identity();auto hash=m.subclass_layout_hash();auto queries=cache.queries;metadata_snapshot=metadata;
 for(int step=0;step<10;++step){for(int layer=0;layer<24;++layer){launches.clear();m.phase2(layer);assert(launches==std::vector<std::string>({"joint"})&&joint_owners.back()==4*layer+1);}}
 assert(cache.queries==queries&&replica_proofs==proof_snapshot&&metadata==metadata_snapshot);
 // Every complete-descriptor field is checked before the first projection.
 for(size_t i=0;i<m.routes[0].args.size();++i){auto x=m;++x.routes[0].args[i];reject([&]{x.phase2(0);});}
 for(int fault=0;fault<8;++fault){auto x=m;if(fault==0)x.complete=false;if(fault==1)x.routes.clear();if(fault==2)++x.routes[0].selector;if(fault==3)++x.routes[0].flags;if(fault==4)++x.routes[0].invocation;if(fault==5)++x.routes[0].site;if(fault==6)x.routes[0].family=FmbRouteFamily::OTHER;if(fault==7)x.routes[0].args.pop_back();reject([&]{x.phase2(0);});}
 reject([&]{m.launch_action_qkv_gate(-1,{});});reject([&]{m.launch_action_qkv_gate(24,{});});reject([&]{m.launch_action_qkv_gate(0,{31,2});});reject([&]{m.launch_action_qkv_gate(0,{32,3});});
 {auto x=m;x.action_qkv_gate_available_=false;assert(x.kvinsert_cost_weight_identity()!=identity&&x.subclass_layout_hash()!=hash);}
 {auto x=m;++x.layer_weights_.back().v_ws.owner;assert(x.kvinsert_cost_weight_identity()!=identity);}
 // Old optional REF preserves the existing Q/gate + KV-owner pair of routes.
 cache.joint=false;auto old=m;old.install();old.plan();assert(!old.action_qkv_gate_available_&&old.routes.size()==2);assert(old.routes[0].site==QWEN35_ACTION_Q_GATE_W8_SITE&&old.routes[1].site==QWEN35_ACTION_KV_OWNER_W8_SITE);launches.clear();old.phase2(23);assert(launches==std::vector<std::string>({"qgate-w8","kv-w8"}));cache.joint=true;
 Model fp16(false);fp16.install();fp16.loop();assert(!fp16.action_qkv_gate_available_&&fp16.routes.size()==2);launches.clear();fp16.phase2(23);assert(launches==std::vector<std::string>({"qgate-fp16","kv-fp16"}));
 // ACC16 keeps the combined route and passes its actual mode to the native wrapper.
 {Model x;x.linear_acc32_=false;x.install();x.loop();assert(x.action_mod_body_w8_&&x.action_qkv_gate_available_&&x.routes.size()==1&&x.routes[0].args[5]==16&&x.routes[0].args[6]==48&&x.subclass_layout_hash()!=hash&&x.kvinsert_cost_weight_identity()!=identity);
  const auto valid=x.routes;x.routes=m.routes;reject([&]{x.phase2(0);});x.routes=valid;linear_modes.clear();launches.clear();x.phase2(23);assert(launches==std::vector<std::string>({"joint"})&&linear_modes==std::vector<bool>({false}));
  cache.joint=false;x.install();x.plan();assert(x.routes.size()==2);linear_modes.clear();launches.clear();x.phase2(23);assert(launches==std::vector<std::string>({"qgate-w8","kv-w8"})&&linear_modes==std::vector<bool>({false,false}));cache.joint=true;
  cache.acc16_available=false;x.install();x.plan();assert(x.action_mod_body_w8_&&!x.action_qkv_gate_available_&&x.routes.empty());linear_modes.clear();launches.clear();x.phase2(23);assert(launches==std::vector<std::string>({"q","k","v","gate"})&&linear_modes.empty());cache.acc16_available=true;
 }
 // Last-layer scale/replica failures must invalidate the reused cold proofs.
 for(int which=0;which<4;++which){auto x=m;auto&w=x.layer_weights_.back();if(which==0)w.attn_gate_ws.present=false;if(which==1)w.q_ws.rows=1024;if(which==2)w.v_ws.replicated=false;if(which==3)w.k_w.replicated=false;x.install();assert(!x.action_qkv_gate_available_);x.plan();assert(x.routes.size()==1);if(which<2)assert(x.action_kv_owner_available_&&!x.action_q_gate_available_);else assert(!x.action_kv_owner_available_&&x.action_q_gate_available_);}
 {auto x=m;x.layer_weights_.back().v_w.replicated=false;x.install();x.plan();launches.clear();x.phase2(23);assert(launches==std::vector<std::string>({"qgate-w8","k","v"}));}
 // Either parent payload or fanout disappearance prevents joint admission.
 for(int fault=0;fault<3;++fault){cache.q=cache.kv=cache.fanout=true;if(fault==0)cache.q=false;if(fault==1)cache.kv=false;if(fault==2)cache.fanout=false;auto x=m;x.install();assert(!x.action_qkv_gate_available_);}cache.q=cache.kv=cache.fanout=true;
 {auto x=m;cache.fanout=false;int before=replica_proofs;x.prepare_action_qkv_gate();assert(!x.action_qkv_gate_available_&&replica_proofs==before);cache.fanout=true;}
 for(int fault=0;fault<5;++fault){auto x=m;if(fault==0)x.action_num_steps_=1;if(fault==1)x.action_step_active_=false;if(fault==2)x.action_loop_active_=false;if(fault==3)x.action_len_=16;if(fault==4)x.action_mode_=false;assert(!x.use_action_qkv_gate(32));x.plan();assert(x.routes.empty());launches.clear();x.phase2(0);assert(launches==std::vector<std::string>({"q","k","v","gate"}));}
 m.reset();assert(!m.action_qkv_gate_available_&&!m.action_q_gate_available_&&!m.action_kv_owner_available_&&!m.use_action_qkv_gate(32)&&m.kvinsert_cost_weight_identity()!=identity);m.plan();assert(m.routes.empty());
 m.action_mode_=true;m.install();m.loop();assert(m.action_qkv_gate_available_&&m.routes.size()==1&&m.subclass_layout_hash()==hash);
}
'''
    replacements = {
        '__OWNER_CACHE_STUBS__': _OWNER_CACHE_STUBS,
        '__METHODS__': '\n'.join(decl + _block(source, marker) for decl, marker in methods),
        '__WEIGHTS__': _block(header, 'struct LayerWeights'),
        '__IDENTITY__': _block(header, 'std::vector<int64_t> kvinsert_cost_weight_identity() const'),
        '__CONSTANTS__': constants, '__CALLER__': caller, '__MANIFEST__': manifest,
        '__INSTALL__': install, '__RESET__': reset,
    }
    for marker,value in replacements.items():
        cpp = cpp.replace(marker,value)
    return cpp


def test_qkv_gate_joint_cold_manifest_phase2_fallback_reset_and_identity(tmp_path):
    _compile_run(tmp_path,'joint_model',_qkv_gate_model_cpp())
