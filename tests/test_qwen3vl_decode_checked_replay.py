"""Run the actual owner-local replay implementation with real ATen metadata.

Only the graph/device/allocator boundary is emulated. No replacement cache or
state machine: begin, hook, finish, ownership, grammar and all patches are the
production C++ source. Full SDK/Torch translation-unit syntax is a separate gate.
"""
from pathlib import Path
import subprocess
import torch

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r'''
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <array>
#include <cassert>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <typeinfo>
#include <vector>
#define RECORD_FUNCTION(...)
using at::Tensor;
uint64_t RpuGetDevAddr(const void* p){return ((reinterpret_cast<uintptr_t>(p)>>4)<<8)&((1ULL<<40)-256);}
int flush_count=0;
void rpu_ddr_flush(void*){++flush_count;}
bool debug_export=false;
bool get_debug_export(){return debug_export;}
struct RegisterPatch{size_t kernel_idx;uint32_t reg_idx;uint64_t value;uint8_t width;uint32_t stride=0;};
enum class GraphNodeKind{Kernel,Dma,Barrier,Branch,Host};
struct Kernel {size_t kernel_idx=0;bool register_census=false;std::vector<uint8_t> core_ids{0,1,2,3,4,5,6,7};std::optional<size_t> kernel_id;std::string kernel_name;};
const char* KERNEL_ID_NAMES[]={"unused"};
struct Dma{int semantic_endpoint_id=0,semantic_spm_peer_id=0;};
struct Node{GraphNodeKind kind=GraphNodeKind::Dma;Kernel kernel;Dma dma;
 const Kernel& as_kernel()const{return kernel;}const Dma& as_dma()const{return dma;}};
struct GraphOpStreamStamp;
struct RpuKernelGraph{
 enum class State{PASSTHROUGH,RECORDING,REPLAYING};
 inline static RpuKernelGraph* current=nullptr;
 State mode=State::RECORDING;size_t cursor=0;uint64_t lifetime=1,generation=2,signature=3,segment=4,topology=0;
 bool census=false,patch_failure=false;int skips=0,ordinary=0;
 struct Policy{bool fmb_fast_replay=false,fmb_deep_fast_replay=false;}policy;
 std::vector<Node> nodes;std::vector<RegisterPatch> applied;
 static bool has_active(){return current;}static auto& active(){assert(current);return *current;}
 State state()const{return mode;}bool kernel_register_census_active()const{return census;}
 const Policy& runtime_policy()const{return policy;}
 struct Stats{uint64_t graph_lifetime_id;};Stats debug_stats()const{return {lifetime};}
 GraphOpStreamStamp op_stream_stamp()const;
 bool is_op_stream_stamp_current(const GraphOpStreamStamp&)const;
 uint64_t build_topology_hash()const{return topology;}
 struct Plan{const std::vector<Node>* steps;};Plan replay_plan()const{return {&nodes};}
 void skip_op_stream_with_patches(const std::vector<RegisterPatch>& p,std::initializer_list<int>){
  TORCH_CHECK(!patch_failure,"patch failed");assert(mode==State::REPLAYING&&!census);
  for(const auto& x:p)assert(x.width==2);applied=p;cursor=nodes.size();++skips;
 }
};
struct GraphOpStreamStamp{const RpuKernelGraph* graph=nullptr;RpuKernelGraph::State state=RpuKernelGraph::State::PASSTHROUGH;
 uint64_t build_generation=0,signature_identity=0,signature_segment_key=0;size_t position=0,extent=0;};
GraphOpStreamStamp RpuKernelGraph::op_stream_stamp()const{return {this,mode,generation,signature,segment,mode==State::RECORDING?nodes.size():cursor,nodes.size()};}
bool RpuKernelGraph::is_op_stream_stamp_current(const GraphOpStreamStamp&s)const{
 return s.graph==this&&s.build_generation==generation&&s.signature_identity==signature&&s.signature_segment_key==segment&&s.position<=nodes.size()&&s.extent<=nodes.size();}
struct Allocator{uint32_t base=0x40000000;size_t mark=99999;
 uint32_t addr(int core,int offset)const{return base+core*0x1000000+offset;}
 size_t temporary_mark()const{return mark;}}SPM_ALLOC;
enum class SdpaKernelType{FLASH_ATTN_SPM,OTHER};
namespace v3 {
enum class CausalDecoderPlanningMode{ORDINARY,OTHER};
enum class ChunkMode{SEQUENTIAL,OTHER};
enum class AttentionExecutionPolicy{AUTO,DDR_KV,SPM_KV_BY_MHA};
class CausalDecoderModel{
public:
 virtual ~CausalDecoderModel()=default;
 struct LayerWeights{Tensor q_w,k_w,v_w,q_ws,k_ws,v_ws,qkv_w,qkv_ws,o_w,o_ws,gate_w,gate_ws,up_w,up_ws,down_w,down_ws,input_norm_w,post_norm_w,q_norm_w,k_norm_w;};
 std::vector<LayerWeights> layer_weights_=std::vector<LayerWeights>(36);
 Tensor cos_,sin_,final_norm_w_,qwen3vl_4b_qkv_scale_bank_,lm_head_w_,lm_head_w_scale_,position_ids_keepalive_;
 bool qwen3vl_4b_decode_o_ring_norm_fused_=false;
 bool qwen3vl_fp16_decode_profile_=false,linear_acc32_=false;
 bool qwen3vl_ordinary_resident_prefill_profile_=false,qwen3vl_4b_w8_profile_=true;
 bool adarms_=false,qwen3vl_pooler_z1_dispatch_=false,qwen3vl_multiview_text_composite_dispatch_=false;
 bool has_qk_norm_=true,has_mrope_=true;
 CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
 int layers=36,h=2560,intermediate=9728,qheads=32;
 bool qwen3vl_decode_gemv(int rows)const{return (qwen3vl_fp16_decode_profile_||qwen3vl_4b_w8_profile_)&&!linear_acc32_&&rows==1;}
 bool admit=true,partial_mrope_active_=false,qwen3vl_4b_decode_gateup_fused_=true,qwen3vl_4b_packed_qkv_weights_=true;
 bool fuse_lm_head_=true,reuse_lm_head_output_=false,use_silu_=true,nvfp4_=false,has_qkv_bias_=false;
 bool qwen3_spm_kv_by_mha_enabled_=true,equal_two_prefill_=false,adarms_mutable_=false,adarms_unroll_=false;
 bool adarms_schedule_select_=false,adarms_fused_bcast_enabled_=false;
 bool fast_replay_skip_layer_loop_=false,preload_replay_skip_=false;
 int64_t vocab_size_=151936,lm_head_exact_candidates_=0,configured_chunk_size_cap_=0;
 SdpaKernelType sdpa_kernel_=SdpaKernelType::FLASH_ATTN_SPM;
 std::vector<int32_t> mrope_section_{24,20,20};std::vector<int64_t> deepstack_lang_layers_{0,1,2};
 struct Envelope{int64_t max_kv_len=2048,chunk=256;}envelope;
 struct Context{int64_t seq_len=1,batch_size=1,position=0;bool complete=true;
  AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV;
  struct Stage{ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;struct Compute{std::vector<int> chunks{0};}compute;}stage_plan;
  bool has_complete_physical_manifest()const{return complete;}}context;
 uint64_t model_generation=7,layout=123;uint32_t norm_delta=0;
 int num_layers()const{return layers;}int hidden_size()const{return h;}int intermediate_size()const{return intermediate;}
 int num_q_heads()const{return qheads;}int num_kv_heads()const{return 8;}int head_dim()const{return 128;}
 int num_cores()const{return 8;}int attn_tp()const{return 8;}int mlp_tp()const{return 8;}int lm_head_tp()const{return 8;}
 bool norm_slab_active()const{return false;}
 bool qwen3vl_4b_decode_qk_norm_mrope(int,int,bool causal,bool mask,int,bool partial)const{return admit&&causal&&!mask&&!partial;}
 const Envelope& chunk_envelope()const{return envelope;}
 uint64_t installed_model_state_generation()const{return model_generation;}
 uint64_t checked_layer_body_replay_layout_hash()const{return layout;}
 Context& ctx(){return context;}
 uint32_t layer_addr(int layer,int core,const char* name)const{
  int role=!std::strcmp(name,"norm_w")?0:!std::strcmp(name,"post_norm_w")?1:!std::strcmp(name,"q_norm_w")?2:3;
  return SPM_ALLOC.addr(core,10000+layer*16384+role*4096+norm_delta);}
 uint32_t addr(int core,const char*)const{return SPM_ALLOC.addr(core,7777+norm_delta);}
 struct Qwen3VlDecodeReplayState;std::shared_ptr<Qwen3VlDecodeReplayState> qwen3vl_decode_replay_state_;
 bool begin_qwen3vl_decode_replay(const Tensor&,at::TensorList,at::TensorList,const std::optional<Tensor>&,int64_t,bool,at::IntArrayRef);
 bool try_checked_layer_body_replay();void finish_qwen3vl_decode_replay();void cancel_qwen3vl_decode_replay()noexcept;
};
}
'''

CASES = r'''
using Model=v3::CausalDecoderModel;
alignas(256) std::array<char,256> dummy{};
// Metadata-only payload owners: the tested code never reads a weight value.
// No multi-GB fixture is allocated and no numeric kernel result is fabricated.
Tensor metadata(at::ScalarType dtype,at::IntArrayRef shape){return at::from_blob(dummy.data(),shape,at::TensorOptions().dtype(dtype));}
bool combined_route=false,combined_o=false,fp16_route=false;
int64_t cold_envelope=2048;
struct Fixture{
 Model m;Tensor hidden=metadata(at::kHalf,{1,1,2560});
 std::vector<Tensor> k,v;std::vector<int64_t> descriptor{71,82,93};
 Fixture(){
  m.envelope.max_kv_len=cold_envelope;
  m.qwen3vl_4b_decode_o_ring_norm_fused_=combined_o;
  for(auto&w:m.layer_weights_){
   w.q_w=metadata(at::kChar,{4096,2560});w.qkv_w=metadata(at::kChar,{6144,2560});w.qkv_ws=metadata(at::kHalf,{6144});
   w.o_w=metadata(at::kChar,{2560,4096});w.o_ws=metadata(at::kHalf,{2560+(combined_o?256:0)}).narrow(0,0,2560);
   w.gate_w=metadata(at::kChar,{9728,2560});w.up_w=metadata(at::kChar,{9728,2560});
   w.gate_ws=metadata(at::kHalf,{9728+256}).narrow(0,0,9728);w.up_ws=metadata(at::kHalf,{9728+256}).narrow(0,0,9728);
   w.down_w=metadata(at::kChar,{2560,9728});w.down_ws=metadata(at::kHalf,{2560});
   w.input_norm_w=metadata(at::kHalf,{2560});w.post_norm_w=metadata(at::kHalf,{2560});
   w.q_norm_w=metadata(at::kHalf,{128});w.k_norm_w=metadata(at::kHalf,{128});
   k.push_back(metadata(at::kHalf,{1,128,1,8,8,16,16}));v.push_back(metadata(at::kHalf,{1,128,1,8,8,16,16}));
  }
  m.cos_=metadata(at::kHalf,{8192,64});m.sin_=metadata(at::kHalf,{8192,64});m.final_norm_w_=metadata(at::kHalf,{2560});
  m.qwen3vl_4b_qkv_scale_bank_=metadata(at::kHalf,{36,6144});
  m.lm_head_w_=metadata(at::kChar,{151936,2560});m.lm_head_w_scale_=metadata(at::kHalf,{151936});
  m.position_ids_keepalive_=at::zeros({8192+8,3},at::TensorOptions().dtype(at::kInt)).narrow(0,0,8192);
  if(fp16_route){
   m.qwen3vl_fp16_decode_profile_=true;m.qwen3vl_4b_w8_profile_=false;m.qwen3vl_ordinary_resident_prefill_profile_=true;
   m.qwen3vl_4b_decode_gateup_fused_=m.qwen3vl_4b_packed_qkv_weights_=false;
   m.qwen3vl_4b_qkv_scale_bank_=Tensor{};m.lm_head_w_scale_=Tensor{};
   m.lm_head_w_=metadata(at::kHalf,{151936,2560});
   for(auto&w:m.layer_weights_){
    w.qkv_w=w.qkv_ws=w.o_ws=w.gate_ws=w.up_ws=w.down_ws=Tensor{};
    w.q_w=metadata(at::kHalf,{4096,2560});w.k_w=metadata(at::kHalf,{1024,2560});w.v_w=metadata(at::kHalf,{1024,2560});
    w.o_w=metadata(at::kHalf,{2560,4096});w.gate_w=metadata(at::kHalf,{9728,2560});
    w.up_w=metadata(at::kHalf,{9728,2560});w.down_w=metadata(at::kHalf,{2560,9728});
   }
  }
 }
 bool begin(RpuKernelGraph&g,int p){RpuKernelGraph::current=&g;g.cursor=0;m.context.position=p;
  return m.begin_qwen3vl_decode_replay(hidden,k,v,std::nullopt,p,true,descriptor);}
 void cache_rows(int rows){for(size_t l=0;l<36;++l){
  k[l]=metadata(at::kHalf,{1,rows/16,1,8,8,16,16});
  v[l]=metadata(at::kHalf,{1,rows/16,1,8,8,16,16});}}
};
void populate(RpuKernelGraph&g){
 auto add=[&](const char*n){Node node;node.kind=GraphNodeKind::Kernel;node.kernel.kernel_idx=19+g.nodes.size()*7;node.kernel.kernel_name=n;g.nodes.push_back(node);};
 for(int l=0;l<36;++l){
  add("llama_rms_norm");
  if(fp16_route){add("llama_gemv");add("llama_gemv");add("llama_gemv");}
  else add("llama_gemv_wint8");
  if(combined_route)add("qwen3vl_qk_norm_mrope_kv_insert_d128");
  else {add("qwen3vl_qk_norm_mrope_d128");add("llama_insert_kcache_multiwarp");add("llama_insert_vcache_multiwarp");}
  add("llm_fp16_32b_prefill_flash_attn_univ_dp");
  if(combined_o)add("qwen3vl_o_ring_norm_m1_h2560_w8");
  else {add(fp16_route?"llama_gemv":"llama_gemv_wint8");add("llm_all_reduce_residual_nopace");add("llama_rms_norm");}
  if(fp16_route)for(const char*n:{"llama_gemv","unary","llama_gemv","binary_sameshape","llama_gemv","llm_all_reduce_residual_nopace"})add(n);
  else for(const char*n:{"qwen3vl_gateup_swiglu_gemv","llama_gemv_wint8","llm_all_reduce_residual_nopace"})add(n);
  if(l<3)add("binary_sameshape");
 }
 add("llama_rms_norm");add(fp16_route?"parallel_linear_m480n64k128":"parallel_linear_w8a16_m288n128k128");
}
constexpr size_t prefix=5;
void preload(RpuKernelGraph&g){if(g.mode==RpuKernelGraph::State::RECORDING)g.nodes.resize(prefix);else g.cursor=prefix;}
void call(Fixture&f,RpuKernelGraph&g,int p){
 bool active=f.begin(g,p);assert(active);
 try{
  if(!f.m.try_checked_layer_body_replay()){
   preload(g);++g.ordinary;if(g.mode==RpuKernelGraph::State::RECORDING)populate(g);else g.cursor=g.nodes.size();
  }
  f.m.finish_qwen3vl_decode_replay();f.m.cancel_qwen3vl_decode_replay();
 }catch(...){f.m.cancel_qwen3vl_decode_replay();throw;}
 if(g.mode==RpuKernelGraph::State::RECORDING){g.mode=RpuKernelGraph::State::REPLAYING;g.topology=9001;}
}
template<class F>void rejects(F f){bool bad=false;try{f();}catch(const c10::Error&){bad=true;}assert(bad);}
void patches(const RpuKernelGraph&g,int p){
 const int stride=combined_route?10:14,positions=combined_route?2:6;
 assert(g.applied.size()==size_t(36*stride));const int s=p+1;
 for(int l=0;l<36;++l){
  const size_t node=prefix+l*((combined_route?10:12)-(combined_o?2:0)+(fp16_route?5:0))+std::min(l,3);
  for(int j=0;j<positions;++j){const auto& x=g.applied[l*stride+j];
   assert(x.reg_idx==uint32_t(j%2)&&x.value==uint64_t(j%2?0:p)&&x.width==2);
   assert(x.kernel_idx==g.nodes[node+(fp16_route?4:2)+j/2].kernel.kernel_idx);
  }
  const std::array<int,8> regs{0,1,4,5,16,17,29,31};
  const std::array<int,8> values{s,0,(s+15)/16,(p+15)/16,p,0,(s+15)/16,(s+15)/16};
  for(int j=0;j<8;++j){const auto& x=g.applied[l*stride+positions+j];
   assert(x.reg_idx==uint32_t(regs[j])&&x.value==uint64_t(values[j])&&x.width==2);
   assert(x.kernel_idx==g.nodes[node+(fp16_route?4:2)+positions/2].kernel.kernel_idx);
  }
 }
}
Tensor quant_scale(at::ScalarType dtype,int n,int k,bool row){
 if(dtype==at::kHalf)return Tensor{};
 if(dtype==at::kChar)return metadata(at::kHalf,{n});
 const int ln=row?n:n/8,lk=row?k/8:k;
 const int count=((lk/32+3)/4)*((ln+63)/64)*8*4*64;
 return metadata(at::kHalf,{32,count/32});
}
void ordinary_fixture(Fixture& f,int layers,at::ScalarType dtype,bool acc32,bool large=false){
 auto&m=f.m;m.layers=layers;m.h=layers==28?2048:2560;m.intermediate=layers==28?6144:9728;m.qheads=layers==28?16:32;
 m.envelope.chunk=layers==28?320:256;
 m.linear_acc32_=acc32;m.qwen3vl_ordinary_resident_prefill_profile_=true;m.qwen3vl_4b_w8_profile_=false;
 m.qwen3vl_fp16_decode_profile_=dtype==at::kHalf;m.qwen3vl_4b_packed_qkv_weights_=m.qwen3vl_4b_decode_gateup_fused_=m.qwen3vl_4b_decode_o_ring_norm_fused_=false;
 if(large){assert(layers==36);m.h=4096;m.intermediate=12288;m.envelope={320,128};
  m.qwen3vl_ordinary_resident_prefill_profile_=m.qwen3vl_fp16_decode_profile_=false;}
 m.layer_weights_.resize(layers);f.k.resize(layers);f.v.resize(layers);
 if(layers==28||large)m.position_ids_keepalive_=at::zeros({8192,3},at::TensorOptions().dtype(at::kInt));
 f.hidden=metadata(at::kHalf,{1,1,m.h});
 auto projection=[&](Tensor&w,Tensor&scale,int n,int k,bool row){w=metadata(dtype,{n,dtype==at::kByte?k/2:k});scale=quant_scale(dtype,n,k,row);};
 for(auto&w:m.layer_weights_){
  projection(w.q_w,w.q_ws,m.qheads*128,m.h,false);projection(w.k_w,w.k_ws,1024,m.h,false);projection(w.v_w,w.v_ws,1024,m.h,false);
  projection(w.o_w,w.o_ws,m.h,m.qheads*128,true);projection(w.gate_w,w.gate_ws,m.intermediate,m.h,false);
  projection(w.up_w,w.up_ws,m.intermediate,m.h,false);projection(w.down_w,w.down_ws,m.h,m.intermediate,true);
  w.qkv_w=w.qkv_ws=Tensor{};w.input_norm_w=metadata(at::kHalf,{m.h});w.post_norm_w=metadata(at::kHalf,{m.h});
 }
 m.final_norm_w_=metadata(at::kHalf,{m.h});m.qwen3vl_4b_qkv_scale_bank_=Tensor{};
 projection(m.lm_head_w_,m.lm_head_w_scale_,151936,m.h,false);
 for(int l=0;l<layers;++l){f.k[l]=metadata(at::kHalf,{1,272,1,8,8,16,16});f.v[l]=metadata(at::kHalf,{1,272,1,8,8,16,16});}
}
void populate_ordinary(RpuKernelGraph&g,const Model&m,bool fused){
 const auto p=v3::CausalDecoderModel::Qwen3VlDecodeReplayState::profile(m);
 const auto names=rpu_pl_tiling::autotile_kernel_names();std::string lin,head;
 for(const auto&n:names)if(v3::linear_name(n,p,true)){head=n;break;}
 lin=p.gemv?"llama_gemv":head;assert(!head.empty());
 auto add=[&](const std::string&n){Node node;node.kind=GraphNodeKind::Kernel;node.kernel.kernel_idx=19+g.nodes.size()*7;node.kernel.kernel_name=n;g.nodes.push_back(node);};
 for(int l=0;l<m.layers;++l){
  add("llama_rms_norm");add(lin);add(lin);add(lin);
  if(fused)add("qwen3vl_qk_norm_mrope_kv_insert_d128");
  else{add("llama_rms_norm");add("llama_rms_norm");add("llama_mrope_interleave");add("llama_mrope_interleave");add("llama_insert_kcache_multiwarp");add("llama_insert_vcache_multiwarp");}
  add("llm_fp16_32b_prefill_flash_attn_univ_dp");add(lin);add("llm_all_reduce_residual_nopace");add("llama_rms_norm");
  add(lin);add("unary");add(lin);add("binary_sameshape");add(lin);add("llm_all_reduce_residual_nopace");if(l<3)add("binary_sameshape");
 }add("llama_rms_norm");add(head);
}
void cross_profile_cases(){
 for(int layers:{28,36})for(auto dtype:{at::kHalf,at::kChar,at::kByte})for(bool acc32:{false,true}){
  Fixture f;ordinary_fixture(f,layers,dtype,acc32);RpuKernelGraph g;
  if(acc32&&dtype!=at::kHalf){assert(!f.begin(g,511));continue;}
  auto run=[&](int p){assert(f.begin(g,p));
   if(!f.m.try_checked_layer_body_replay()){preload(g);++g.ordinary;if(g.mode==RpuKernelGraph::State::RECORDING)populate_ordinary(g,f.m,layers==36);else g.cursor=g.nodes.size();}
   f.m.finish_qwen3vl_decode_replay();f.m.cancel_qwen3vl_decode_replay();
   g.mode=RpuKernelGraph::State::REPLAYING;g.topology=9001;
  };
  run(511);assert(f.m.qwen3vl_decode_replay_state_->recipe);
  for(int p:{512,4095,4096,4351}){
   run(p);assert(g.applied.size()==size_t(layers*(layers==28?16:10)));
   for(const auto&patch:g.applied){const auto it=std::find_if(g.nodes.begin(),g.nodes.end(),[&](const Node&n){return n.kind==GraphNodeKind::Kernel&&n.kernel.kernel_idx==patch.kernel_idx;});assert(it!=g.nodes.end());
    const auto&name=it->kernel.kernel_name;
    if(name=="llm_fp16_32b_prefill_flash_attn_univ_dp"){
     uint64_t value=0;switch(patch.reg_idx){case 0:value=p+1;break;case 1:case 17:break;case 4:case 29:case 31:value=(p+16)/16;break;case 5:value=(p+15)/16;break;case 16:value=p;break;default:assert(false);}assert(patch.value==value);
    }else{assert(name=="llama_mrope_interleave"||name=="llama_insert_kcache_multiwarp"||name=="llama_insert_vcache_multiwarp"||name=="qwen3vl_qk_norm_mrope_kv_insert_d128");assert(patch.reg_idx<2&&patch.value==uint64_t(patch.reg_idx?0:p));}
   }
  }assert(g.ordinary==1&&g.skips==4);
  // All separate projections/scales and both KV lists remain owner-bound.
  f.m.layer_weights_[3].k_w.unsafeGetTensorImpl()->bump_version();run(513);assert(g.ordinary==2);run(514);assert(g.ordinary==2);
  if(dtype!=at::kHalf){f.m.layer_weights_[9].v_ws.unsafeGetTensorImpl()->bump_version();run(515);assert(g.ordinary==3);run(516);assert(g.ordinary==3);}
  int ordinary=g.ordinary;f.v[7]=metadata(at::kHalf,{1,272,1,8,8,16,16});run(517);assert(g.ordinary==ordinary+1);run(518);assert(g.ordinary==ordinary+1);
  // A malformed packed group scale cannot install a replacement recipe.
  if(dtype==at::kByte){f.m.layer_weights_[0].down_ws=metadata(at::kHalf,{32,1});run(519);run(520);assert(g.ordinary==ordinary+3);}
 }
}
void exact_8b_cases(){
 for(int64_t envelope:{320,4176}){
 Fixture f;ordinary_fixture(f,36,at::kHalf,true,true);RpuKernelGraph g;
 f.m.envelope.max_kv_len=envelope;
 assert(!f.m.qwen3vl_decode_gemv(1));
 auto run=[&](int p){assert(f.begin(g,p));
  if(!f.m.try_checked_layer_body_replay()){preload(g);++g.ordinary;if(g.mode==RpuKernelGraph::State::RECORDING)populate_ordinary(g,f.m,false);else g.cursor=g.nodes.size();}
  f.m.finish_qwen3vl_decode_replay();f.m.cancel_qwen3vl_decode_replay();
  g.mode=RpuKernelGraph::State::REPLAYING;g.topology=9001;
 };
 run(213);assert(f.m.qwen3vl_decode_replay_state_->recipe);
 for(int p:{214,228,229,313,314,4095,4096}){run(p);assert(g.applied.size()==36*16);
  for(int l=0;l<36;++l)for(int j=0;j<8;++j){const auto& patch=g.applied[l*16+j];assert(patch.value==uint64_t(j%2?0:p));}}
 assert(g.ordinary==1&&g.skips==7);
 f.m.layer_weights_[35].down_w.unsafeGetTensorImpl()->bump_version();run(315);assert(g.ordinary==2);run(316);assert(g.ordinary==2);
 f.k[1]=metadata(at::kHalf,{1,272,1,8,8,16,16});run(317);assert(g.ordinary==3);run(318);assert(g.ordinary==3);
 // Publishing new weights/scales invalidates the native model generation.
 f.m.layer_weights_[17].up_ws=metadata(at::kHalf,{12288});++f.m.model_generation;
 run(319);run(320);assert(g.ordinary==5);
 for(int fault=0;fault<7;++fault){Fixture bad;ordinary_fixture(bad,36,at::kHalf,true,true);RpuKernelGraph other;
  if(fault==0)bad.m.linear_acc32_=false;
  if(fault==1)bad.m.envelope.max_kv_len=4192;
  if(fault==2)bad.m.envelope.chunk=256;
  if(fault==3)bad.m.intermediate=12304;
  if(fault==4)bad.m.layer_weights_[0].q_w=metadata(at::kChar,{4096,4096});
  if(fault==5)bad.m.layers=64;
  if(fault==6)bad.m.layer_weights_[0].q_w=Tensor{};
  assert(!bad.begin(other,213));}
 }
}
int main(){
 cross_profile_cases();
 exact_8b_cases();
 for(int64_t cap:{2048,4176}){cold_envelope=cap;
 for(bool o_route:{false,true})for(bool route:{false,true}){combined_route=route;combined_o=o_route;
 {Fixture f;RpuKernelGraph g;call(f,g,15);assert(g.ordinary==1&&g.skips==0);
  for(int p:{16,17,95,96,111,112,511,512,513,1535,2047,15}){
   auto* thw=f.m.position_ids_keepalive_.data_ptr<int32_t>()+p*3;thw[0]=7;thw[1]=13;thw[2]=2;
   const int old=flush_count;call(f,g,p);assert(flush_count-old==36*22+3);patches(g,p);
   assert(thw[0]==7&&thw[1]==13&&thw[2]==2);
  }assert(g.ordinary==1&&g.skips==12);
  // Mutable KV contents/version change is permitted; immutable version isn't.
  f.k[0].unsafeGetTensorImpl()->bump_version();call(f,g,17);assert(g.ordinary==1);
  f.m.layer_weights_[0].gate_w.unsafeGetTensorImpl()->bump_version();call(f,g,18);assert(g.ordinary==2);
  call(f,g,19);assert(g.ordinary==2);
  // Same signature, new storage/owner: ordinary emission before the new binding.
  const auto* old=f.k[0].unsafeGetTensorImpl();f.k[0]=metadata(at::kHalf,{1,128,1,8,8,16,16});
  call(f,g,20);assert(g.ordinary==3);call(f,g,21);assert(g.ordinary==3);
  for(const auto&o:f.m.qwen3vl_decode_replay_state_->recipe->owners)assert(o.impl!=old);
  // Actual scalar dtype snapshot catches same-TensorImpl .data alias rebinding.
  auto& gate=f.m.layer_weights_[0].gate_ws;auto* impl=gate.unsafeGetTensorImpl();auto ver=gate._version();
  gate.set_data(gate.view(at::kBFloat16));assert(gate.unsafeGetTensorImpl()==impl&&gate._version()==ver);
  call(f,g,22);assert(g.ordinary==4);call(f,g,23);assert(g.ordinary==5);
  gate.set_data(gate.view(at::kHalf));call(f,g,24);assert(g.ordinary==5); // restored original immutable owner is valid
  // Legal entry miss A->B->A, then only the current binding is reused.
  RpuKernelGraph b;b.lifetime=90;b.signature=91;call(f,b,25);assert(b.ordinary==1);
  call(f,g,26);assert(g.ordinary==6);call(f,g,27);assert(g.ordinary==6);
  ++g.generation;call(f,g,28);assert(g.ordinary==7);call(f,g,29);assert(g.ordinary==7);
  ++g.lifetime;call(f,g,30);assert(g.ordinary==8);
  ++f.m.model_generation;call(f,g,31);assert(g.ordinary==9);
  f.descriptor[1]++;call(f,g,32);assert(g.ordinary==10);
 }
 // Preload owners use the same actual immutable-owner proof as body operands.
 {Fixture f;RpuKernelGraph g;call(f,g,15);call(f,g,16);
  f.m.layer_weights_[0].input_norm_w.unsafeGetTensorImpl()->bump_version();
  call(f,g,17);assert(g.ordinary==2);call(f,g,18);assert(g.ordinary==2);
  f.m.final_norm_w_=metadata(at::kHalf,{2560});
  call(f,g,19);assert(g.ordinary==3);call(f,g,20);assert(g.ordinary==3);
 }
 // O scale carries initialized overfetch backing and remains a versioned owner.
 if(combined_o){Fixture f;RpuKernelGraph g;call(f,g,15);call(f,g,16);
  f.m.layer_weights_[0].o_ws.unsafeGetTensorImpl()->bump_version();
  call(f,g,17);assert(g.ordinary==2);call(f,g,18);assert(g.ordinary==2);
  f.m.layer_weights_[0].o_ws=metadata(at::kHalf,{2816}).narrow(0,0,2560);
  call(f,g,19);assert(g.ordinary==3);call(f,g,20);assert(g.ordinary==3);
 }
 // Matched proof corruption fails before any patch/visibility, never fallback.
 for(int variant=0;variant<9;++variant){Fixture f;RpuKernelGraph g;call(f,g,15);call(f,g,16);int old=g.skips,fl=flush_count;
  assert(f.begin(g,17));
  if(variant==0)g.topology++;
  if(variant==1)g.nodes.push_back(Node{});
  if(variant==2)g.cursor++;
  if(variant==3)f.m.layout++;
  if(variant==4)f.m.norm_delta+=256;
  if(variant==5)SPM_ALLOC.base+=256;
  if(variant==6)SPM_ALLOC.mark+=256;
  if(variant==7)f.m.model_generation++;
  if(variant==8)f.m.qwen3vl_decode_replay_state_->recipe->cache_capacity+=16;
  rejects([&]{f.m.try_checked_layer_body_replay();});assert(g.skips==old&&flush_count==fl);
  f.m.cancel_qwen3vl_decode_replay();assert(!f.m.qwen3vl_decode_replay_state_->recipe);
  SPM_ALLOC.base=0x40000000;SPM_ALLOC.mark=99999;
 }
 {Fixture f;RpuKernelGraph g;call(f,g,15);g.patch_failure=true;
  assert(f.begin(g,16));rejects([&]{f.m.try_checked_layer_body_replay();});
  assert(g.ordinary==1&&g.skips==0);f.m.cancel_qwen3vl_decode_replay();assert(!f.m.qwen3vl_decode_replay_state_->recipe);}
 // Static backing proofs, versionless weights and inadmissible recorded nodes
 // leave original execution available without ever installing a recipe.
 for(int variant=0;variant<(combined_o?7:6);++variant){Fixture f;RpuKernelGraph g;
  if(variant==0)f.m.layer_weights_[0].gate_ws=metadata(at::kHalf,{9728});
  if(variant==1)f.m.position_ids_keepalive_=at::zeros({8192,3},at::TensorOptions().dtype(at::kInt));
  if(variant==2){c10::InferenceMode inference;f.m.layer_weights_[0].gate_w=metadata(at::kChar,{9728,2560});}
  if(variant==6)f.m.layer_weights_[0].o_ws=metadata(at::kHalf,{2560});
  assert(f.begin(g,15));assert(!f.m.try_checked_layer_body_replay());preload(g);populate(g);
  if(variant==3)g.nodes.back().kind=GraphNodeKind::Host;
  if(variant==4)g.nodes[prefix].kernel.register_census=true;
  if(variant==5)g.nodes[0].dma.semantic_endpoint_id=1;
  f.m.finish_qwen3vl_decode_replay();assert(!f.m.qwen3vl_decode_replay_state_->recipe);f.m.cancel_qwen3vl_decode_replay();
 }
 {Fixture f;RpuKernelGraph g;
  for(int p:{-1,2048})rejects([&]{f.begin(g,p);});
  for(int axis=0;axis<3;++axis){auto* row=f.m.position_ids_keepalive_.data_ptr<int32_t>()+15*3;
   row[axis]=8192;rejects([&]{f.begin(g,15);});row[axis]=-1;rejects([&]{f.begin(g,15);});row[axis]=0;}
  f.m.has_mrope_=false;assert(!f.begin(g,15));f.m.has_mrope_=true;
  for(int64_t unknown:{0,336,2047,2049,4096,4175,4177,8192}){
   f.m.envelope.max_kv_len=unknown;assert(!f.begin(g,15));
  }f.m.envelope.max_kv_len=cap;
  f.m.envelope.chunk=320;assert(!f.begin(g,15));f.m.envelope.chunk=256;
  // Every layer and both K/V must share the same physical capacity.
  for(int blocks:{129,264})for(bool key:{false,true}){
   auto& cache=key?f.k:f.v;auto saved=cache[17];
   cache[17]=metadata(at::kHalf,{1,blocks,1,8,8,16,16});
   assert(!f.begin(g,15));cache[17]=saved;
  }
  g.census=true;assert(!f.begin(g,15));g.census=false;
  g.policy.fmb_fast_replay=true;assert(!f.begin(g,15));g.policy.fmb_fast_replay=false;
  f.m.partial_mrope_active_=true;assert(!f.begin(g,15));f.m.partial_mrope_active_=false;
 }
 // Real public matrix allocations include small caches; none is exactly 2048.
 // The existing short allocation and the gate's 4352-row allocation also replay.
 for(int rows:{16,496,576,640,1088,1152,1392,2048,2112,2176,2608,3504,4160,4224,4352}){
  Fixture f;f.cache_rows(rows);RpuKernelGraph g;call(f,g,0);
  assert(f.m.qwen3vl_decode_replay_state_->recipe->cache_capacity==rows);
  for(int p:{rows-2,rows-1,0}){call(f,g,p);patches(g,p);}
  assert(g.ordinary==1&&g.skips==3);const int skips=g.skips;
  rejects([&]{f.begin(g,rows);});assert(g.skips==skips);
 }
 {Fixture f;f.cache_rows(4352);RpuKernelGraph g;call(f,g,2047);
  for(int p:{2048,4095,4096,4351,2047}){call(f,g,p);patches(g,p);}
  assert(g.ordinary==1&&g.skips==5);
  // Smaller tables cannot borrow the packed cache's extra rows.
  f.m.cos_=metadata(at::kHalf,{4096,64});f.m.sin_=metadata(at::kHalf,{4096,64});
  call(f,g,4095);assert(g.ordinary==2);const int skips=g.skips;
  rejects([&]{f.begin(g,4096);});assert(g.skips==skips);
  f.m.position_ids_keepalive_.data_ptr<int32_t>()[4095*3+2]=4096;
  rejects([&]{f.begin(g,4095);});assert(g.skips==skips);
 }
 // Replacing every cache with another valid capacity must re-emit before skip.
 {Fixture f;f.cache_rows(576);RpuKernelGraph g;call(f,g,511);call(f,g,512);
  f.cache_rows(4352);call(f,g,4095);assert(g.ordinary==2&&g.skips==1);
  call(f,g,4096);assert(g.ordinary==2&&g.skips==2);patches(g,4096);
  f.cache_rows(576);call(f,g,513);assert(g.ordinary==3&&g.skips==2);
  call(f,g,514);assert(g.ordinary==3&&g.skips==3);patches(g,514);
 }
 for(int variant=0;variant<7;++variant){Fixture f;RpuKernelGraph g;
  if(variant==0)f.cache_rows(0);
  if(variant==1)f.cache_rows(4368);
  if(variant==2)f.k[4]=metadata(at::kHalf,{1,128,1,8,8,16});
  if(variant==3)f.v[7]=metadata(at::kFloat,{1,128,1,8,8,16,16});
  if(variant==4)f.k[12]=f.k[12].transpose(5,6);
  if(variant==5)f.v[33]=metadata(at::kHalf,{2,128,1,8,8,16,16});
  if(variant==6)f.k[0]=Tensor{};
  assert(!f.begin(g,15)&&g.skips==0);
 }
 {Fixture f;RpuKernelGraph g;call(f,g,15);assert(f.begin(g,16));
  f.m.context.attention_policy=v3::AttentionExecutionPolicy::SPM_KV_BY_MHA;
  assert(!f.m.try_checked_layer_body_replay()&&g.skips==0);f.m.cancel_qwen3vl_decode_replay();}
 // Complete grammar is authoritative: remove/reorder/duplicate/tail extras fail.
 {RpuKernelGraph g;populate(g);std::vector<v3::DecodeKernelSlot> k;
  for(size_t i=0;i<g.nodes.size();++i)k.push_back({g.nodes[i].kernel.kernel_idx,i,g.nodes[i].kernel.kernel_name});
  std::vector<RegisterPatch> p;assert(v3::bind_decode_patches(k,p,v3::DecodeProfile{})&&p.size()==size_t(combined_route?360:504));
  // An individually valid alternate layer must not create a mixed recipe.
  for(int changed=0;changed<2;++changed){
   const int layer_end=(combined_route?11:13)-(combined_o?2:0);
   if(changed==0)combined_route=!route;else combined_o=!o_route;
   RpuKernelGraph alternate;populate(alternate);
   const int alt_end=(combined_route?11:13)-(combined_o?2:0);
   combined_route=route;combined_o=o_route;
   auto mixed=k;mixed.erase(mixed.begin(),mixed.begin()+layer_end);
   std::vector<v3::DecodeKernelSlot> first;
   for(int i=0;i<alt_end;++i)first.push_back({0,0,alternate.nodes[i].kernel.kernel_name});
   mixed.insert(mixed.begin(),first.begin(),first.end());
   for(size_t i=0;i<mixed.size();++i){mixed[i].kernel=i;mixed[i].node=i;}
   assert(!v3::bind_decode_patches(mixed,p,v3::DecodeProfile{}));
  }
  auto bad=k;bad.erase(bad.begin()+7);assert(!v3::bind_decode_patches(bad,p,v3::DecodeProfile{}));
  bad=k;std::swap(bad[3].name,bad[4].name);assert(!v3::bind_decode_patches(bad,p,v3::DecodeProfile{}));
  bad=k;bad[2].kernel=bad[1].kernel;assert(!v3::bind_decode_patches(bad,p,v3::DecodeProfile{}));
  bad=k;bad.push_back({99999,99999,"llama_rms_norm"});assert(!v3::bind_decode_patches(bad,p,v3::DecodeProfile{}));
 }
 }
 }
 // Old REF: separate Gate/Up GEMVs plus the existing fused silu_mul.
 {combined_route=true;combined_o=false;Fixture f;f.m.qwen3vl_4b_decode_gateup_fused_=false;
  RpuKernelGraph g;assert(f.begin(g,511));assert(!f.m.try_checked_layer_body_replay());preload(g);populate(g);
  for(size_t n=0;n<g.nodes.size();++n)if(g.nodes[n].kernel.kernel_name=="qwen3vl_gateup_swiglu_gemv"){
   Node a=g.nodes[n],b=a,c=a;a.kernel.kernel_name=b.kernel.kernel_name="llama_gemv_wint8";c.kernel.kernel_name="silu_mul";
   g.nodes[n]=a;g.nodes.insert(g.nodes.begin()+n+1,{b,c});n+=2;
  }
  for(size_t n=0;n<g.nodes.size();++n)g.nodes[n].kernel.kernel_idx=19+n*7;
  f.m.finish_qwen3vl_decode_replay();f.m.cancel_qwen3vl_decode_replay();
  assert(f.m.qwen3vl_decode_replay_state_->recipe);g.mode=RpuKernelGraph::State::REPLAYING;g.topology=9001;
  call(f,g,512);assert(g.skips==1);
 }
 // FP16 keeps separate Q/K/V and ordinary SiLU/Mul. Its actual stream and
 // all seven projection owners must be proved; W8 recipes cannot be reused.
 fp16_route=true;combined_o=false;cold_envelope=4176;
 for(bool fused:{false,true}){combined_route=fused;
  {Fixture f;f.cache_rows(4352);RpuKernelGraph g;call(f,g,511);
   assert(g.ordinary==1&&g.skips==0);
   for(int p:{512,513,4095,4096,4351}){call(f,g,p);patches(g,p);}
   assert(g.ordinary==1&&g.skips==5);
   for(auto member:{&Model::LayerWeights::q_w,&Model::LayerWeights::k_w,&Model::LayerWeights::v_w,
                    &Model::LayerWeights::o_w,&Model::LayerWeights::gate_w,&Model::LayerWeights::up_w,&Model::LayerWeights::down_w}){
    auto& w=f.m.layer_weights_[19].*member;
    const int ordinary=g.ordinary;
    w.unsafeGetTensorImpl()->bump_version();call(f,g,514);assert(g.ordinary==ordinary+1);
    call(f,g,515);assert(g.ordinary==ordinary+1);
   }
   f.cache_rows(576);call(f,g,513);int ordinary=g.ordinary;call(f,g,514);assert(g.ordinary==ordinary);
   rejects([&]{f.begin(g,576);});
  }
  for(int bad=0;bad<6;++bad){Fixture f;RpuKernelGraph g;
   if(bad==0)f.m.layer_weights_[35].q_w=metadata(at::kChar,{4096,2560});
   if(bad==1)f.m.layer_weights_[0].k_w=metadata(at::kHalf,{1024,2048});
   if(bad==2)f.m.layer_weights_[9].v_w=Tensor{};
   if(bad==3){c10::InferenceMode inference;f.m.layer_weights_[7].o_w=metadata(at::kHalf,{2560,4096});}
   assert(f.begin(g,511));assert(!f.m.try_checked_layer_body_replay());preload(g);populate(g);
   if(bad==4)g.nodes[prefix+1].kernel.kernel_name="llama_gemv_wint8";
   if(bad==5)g.nodes.back().kernel.kernel_name="parallel_linear_acc32_m352n64k128";
   f.m.finish_qwen3vl_decode_replay();assert(!f.m.qwen3vl_decode_replay_state_->recipe);f.m.cancel_qwen3vl_decode_replay();
  }
  {Fixture f;RpuKernelGraph g;f.m.linear_acc32_=true;assert(f.begin(g,511));f.m.cancel_qwen3vl_decode_replay();}
  {Fixture f;RpuKernelGraph g;f.m.lm_head_w_scale_=metadata(at::kHalf,{151936});assert(!f.begin(g,511));}
  {Fixture f;RpuKernelGraph g;call(f,g,511);call(f,g,512);assert(f.begin(g,513));
   f.m.norm_delta+=256;rejects([&]{f.m.try_checked_layer_body_replay();});f.m.cancel_qwen3vl_decode_replay();
   assert(!f.m.qwen3vl_decode_replay_state_->recipe);}
 }
}
'''


def test_actual_decode_owner_recipe_and_invalidation(tmp_path):
    source = (ROOT / "src/fused/rpu_qwen3vl_decode_replay.cpp").read_text()
    source = source.replace('#include "rpu_qwen3_model.h"', '')
    source = source.replace('#include "../ops/rpu_linear_tiling.h"', '#include "' + str(ROOT / 'src/ops/rpu_linear_tiling.h') + '"')
    assert source.count('t.device().type() != at::kPrivateUse1') == 2
    source = source.replace('t.device().type() != at::kPrivateUse1', 't.device().type() != at::kCPU')
    cpp = tmp_path / "checked_replay.cpp"
    cpp.write_text(HARNESS + source + CASES)
    torch_root = Path(torch.__file__).resolve().parent
    exe = tmp_path / "checked_replay"
    subprocess.run([
        "g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(torch_root / "include"), str(cpp),
        "-L", str(torch_root / "lib"), "-ltorch_cpu", "-lc10",
        f"-Wl,-rpath,{torch_root / 'lib'}", "-o", str(exe),
    ], check=True)
    subprocess.run([str(exe)], check=True)
