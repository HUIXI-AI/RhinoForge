"""Run the exact C224 Prefill owner-norm producer and consumers without a device."""
from pathlib import Path
import re

import pytest

from pi_wall_host_catalog import CATALOG_INCLUDE, absolute_candidate_includes
from test_wall_base_nvfp4_qkv_bias import META
from test_wall_base_paired_kv import block, compile_run


ROOT = Path(__file__).resolve().parents[1]


def _chain(source, marker):
    """Extract the original if plus its immediately following else arms."""
    start = source.index(marker)
    first = block(source[start:], marker)
    rest = source[start + len(first):].lstrip()
    while rest.startswith('else'):
        arm = block(rest, 'else')
        first += ' ' + arm
        rest = rest[len(arm):].lstrip()
    return first


@pytest.mark.parametrize("fp16", [False, True], ids=["w8", "fp16"])
def test_actual_prefill_owner_norm_cold_complete_and_fallback(tmp_path, fp16):
    decoder = (ROOT / 'src/fused/rpu_qwen3_model.h').read_text()
    planner = (ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    base = (ROOT / 'src/core/fused_model_base.h').read_text()
    decls = (ROOT / 'src/core/rpu_kernel_decls.h').read_text()
    cache = (ROOT / 'src/core/rpu_kernel_cache.h').read_text()
    runtime = (ROOT / 'src/ops/rpu_pi_wall_runtime.h').read_text()
    start = decoder.index('        wall_base_prefill_shared_input_ =')
    cold = decoder[start:decoder.index('        qwen3vl_4b_decode_gateup_fused_ =', start)]
    start = decoder.index('        wall_base_fp16_long_prefill_ =')
    cold += decoder[start:decoder.index('        // Cold opt-in by actual storage', start)]
    helpers = '\n'.join(block(decoder, marker) for marker in (
        '    bool wall_small_prefill_fusion(',
        '    bool wall_base_prefill_shared_input_enabled(',
        '    bool wall_base_long_prefill_qkv_fusion(',
        '    bool wall_base_fp16_long_prefill_fusion(',
        '    bool wall_base_fp16_shared_gateup_enabled(',
        '    std::vector<int64_t> wall_base_fp16_shared_gateup_arguments(',
        '    void emit_wall_base_fp16_shared_gateup(',
        '    bool wall_base_prefill_owner_norm_enabled(',
        '    std::vector<int64_t> wall_base_prefill_owner_norm_arguments(',
        '    void consume_wall_base_prefill_owner_norm(',
        '    bool ordinary_rmsnorm_route_enabled() const',
        '    RpuRmsNormSpmRoute consume_ordinary_rmsnorm_route(',
        '    RpuRmsNormSpmRoute consume_wall_rmsnorm_route(',
        '    int64_t wall_base_prefill_shared_gateup_site(',
        '    std::vector<int64_t> wall_base_prefill_shared_input_arguments(',
        '    void emit_wall_base_prefill_shared_input(',
        '    FmbPhysicalManifestForwardCapability\n    physical_manifest_forward_capability(',
    )).replace(' const override', ' const')
    norm_producer = block(decoder, '        if (ordinary_rmsnorm_route_enabled() && layout.is_causal &&')
    owner_producer = _chain(decoder, 'if (wall_prefill_owner_norm) {')
    pre_marker = 'else {\n                append(FmbRouteFamily::ALL_REDUCE,\n                       CAUSAL_DECODER_PREPARE_SITE,'
    # The sorting fallback is an else-if; this is the ordinary-O else arm.
    pre_start = decoder.index(pre_marker, decoder.index('if (wall_prefill_owner_norm) {'))
    pre_producer = block(decoder[pre_start:], pre_marker).split('{', 1)[1].rsplit('}', 1)[0]
    o_body = block(decoder, '        } else {\n            // O-proj row-partition').split('{', 1)[1].rsplit('}', 1)[0]
    shared_producer = block(decoder, '            if (wall_base_fp16_shared_gateup_enabled(')
    shared_producer += ' ' + block(decoder, 'else if (wall_base_prefill_shared_input_enabled(')
    consumer = '\n'.join(block(planner, marker).replace('FmbPhysicalManifestConsumer::', '') for marker in (
        'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(',
        'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(',
    ))
    constants = '\n'.join(re.search(r'inline constexpr int64_t ' + name + r'\s*=.*?;', decoder, re.S).group()
                          for name in (
        'WALL_BASE_PREFILL_O_OWNER_NORM_SITE', 'CAUSAL_DECODER_PREPARE_SITE',
        'CAUSAL_DECODER_O_SITE', 'CAUSAL_DECODER_REDUCE_SITE', 'CAUSAL_DECODER_MIXED_REDUCE_SITE',
        'CAUSAL_DECODER_ORDINARY_RMSNORM_SITE', 'CAUSAL_DECODER_POST_MIXED_RMSNORM_SITE',
        'RHINOVLA_TEXT_HIGH_RMSNORM_SITE', 'RHINOVLA_TEXT_HIGH_ACTIVATION_SITE',
        'WALL_BASE_PREFILL_SHARED_GATEUP_SITE', 'WALL_BASE_PREFILL_SHARED_SILU_SITE',
        'WALL_BASE_PREFILL_SHARED_NOUP_SITE', 'WALL_BASE_PREFILL_SHARED_DOWN_SITE',
        'WALL_BASE_PREFILL_SHARED_REDUCE_SITE', 'WALL_BASE_FP16_PREFILL_SHARED_GATEUP_SITE',
    ))
    source = CATALOG_INCLUDE + META + r'''
#include <atomic>
#include <bitset>
#include <cmath>
#include <memory>
#include <tuple>
#undef TORCH_INTERNAL_ASSERT
#define TORCH_INTERNAL_ASSERT(x) TORCH_CHECK(x)
KERNEL_IDS
AVAILABILITY
PI_WALL_ID
struct KernelCache {
 KernelPayloadAvailability availability;
 static auto& instance(){static KernelCache k;return k;}
 bool has_loaded(KernelId id)const{return availability.has_loaded(id);}
};
PI_WALL_RUNTIME
constexpr bool FP16=IS_FP16;
void publish(bool candidate=true,bool fp16_qkv=true,bool w8_payloads=true,bool fp16_gateup=true){
 KernelPayloadAvailability::ExtendedPayloads b;
 for(const auto& p:rpu_pi_wall::profiles){
  const bool shared=p.operation==rpu_pi_wall::Operation::SharedRaw||
   p.operation==rpu_pi_wall::Operation::SharedSiluUp||p.operation==rpu_pi_wall::Operation::SharedSilu;
  if(p.weight==rpu_pl_tiling::WeightType::Int8&&(shared||p.operation==rpu_pi_wall::Operation::Qkv))
   b.set(size_t(pi_wall_kernel_id(p.ordinal))-KernelPayloadAvailability::kExtendedBegin,w8_payloads);
  if(p.weight==rpu_pl_tiling::WeightType::Fp16&&p.operation==rpu_pi_wall::Operation::Qkv)
   b.set(size_t(pi_wall_kernel_id(p.ordinal))-KernelPayloadAvailability::kExtendedBegin,fp16_qkv);
  if(p.weight==rpu_pl_tiling::WeightType::Fp16&&p.operation==rpu_pi_wall::Operation::SharedSilu)
   b.set(size_t(pi_wall_kernel_id(p.ordinal))-KernelPayloadAvailability::kExtendedBegin,fp16_gateup);
 }
 b.set(size_t(KernelId::WALL_BASE_PREFILL_O_OWNER_RING_NORM_M224)-KernelPayloadAvailability::kExtendedBegin,candidate);
 KernelCache::instance().availability.publish(false,false,false,false,false,false,b,
  false,false,false,false,false,false,false,w8_payloads);
}
enum class ChunkMode {SEQUENTIAL,KV_FIRST};
enum class CausalDecoderPlanningMode {ORDINARY,WALL_Z1};
ROUTE_FAMILY
GRAPH_LIFECYCLE
ACCUMULATION
MANIFEST_STATE
enum class FmbLinearRouteSelector {AUTO_TILE=1,GEMV=2,ROW_WEIGHT_REUSE=5};
LINEAR_ROUTES
REDUCE_ROUTES
RMS_TYPES
RMS_RESOLVE
RMS_HIGH_RESOLVE
RMS_CONTRACT
struct ChunkInfo {int64_t idx,offset,len,kv_seq_len;};
struct ResolvedFmbStageChunks {struct{int64_t chunk_size,num_chunks;}plan;std::vector<ChunkInfo> chunks;};
struct FmbThreeStageChunkPlan {ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;ResolvedFmbStageChunks input,qkv,compute;};
MAKE_STAGE
struct FmbRouteManifestEntry {int64_t site_id;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t> arguments;int64_t invocation;};
struct FmbPhysicalExecutionManifest {
 std::vector<FmbRouteManifestEntry> routes;int64_t logical_length=402,physical_length=448,execution_padding_rows=46;
 FmbGraphLifecycle graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
 FmbLinearAccumulationPolicy linear_accumulation=FmbLinearAccumulationPolicy::ACC16;
 FmbPhysicalManifestState state=FmbPhysicalManifestState::COMPLETE;
};
struct FmbPhysicalManifestForwardCapability {bool supported;FmbGraphLifecycle lifecycle;};
struct Context {
 FmbPhysicalExecutionManifest value;mutable std::vector<uint8_t> consumed_;
 bool ready=true,is_causal=true;int64_t position=0,batch_size=1;std::optional<int> attention_mask;
 FmbThreeStageChunkPlan stage_plan;
 bool complete()const{return ready;}bool has_complete_physical_manifest()const{return ready;}
 const auto& manifest()const{return value;}
 CONSUMER
 auto& consume_physical_route(FmbRouteFamily f,int64_t s,int64_t r,int64_t flags,at::IntArrayRef a,int64_t i=0)const{
  return consume_route(f,s,r,flags,a,i);}
};
CONSTANTS
constexpr int64_t FMB_SHARED_MLP_SILU_MUL_SITE=91,kCausalDecoderRmsNormRowsSite=92;
enum class ValuOpType {ADD};
struct Counts {int prepare=0,o=0,fused=0,old_ring=0,norm=0,gate=0,down=0,down_ring=0;}counts;
std::vector<uint32_t> gammas;
int64_t causal_ring_route(int64_t rows,int64_t h){assert(rows>0&&h==2048);return 2;}
void rpu_prepare_ring_all_reduce_input(uint32_t p,int64_t rows,int64_t h,int tp,int cores){
 assert(p==0x400000&&rows==224&&h==2048&&tp==8&&cores==8);++counts.prepare;}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t in,const at::Tensor& weight,uint32_t out,
 int64_t m,int64_t n,int64_t k,int part,int cores,int bias,const at::Tensor& scale,
 uint32_t ts=0,uint16_t layer=0,bool acc32=false,bool gemv=false,bool planar=false,bool reuse=false){
 assert(weight.scalar_type()==(FP16?at::kHalf:at::kChar));
 if(FP16)assert(!scale.defined());
 assert(m==224&&n==2048&&part==0&&cores==8&&bias==0&&!ts&&!acc32&&!gemv&&!planar&&!reuse);
 if(k==2048){assert(in==0x300000&&out==0x400000&&weight.address==uintptr_t(100+layer));++counts.o;}
 else {assert(k==11008&&in==0x500000&&out==0x700000);++counts.down;}
}
void rpu_launch_wall_base_prefill_o_owner_ring_norm_m224_kernel(uint32_t partial,uint32_t residual,
 uint32_t raw,uint32_t gamma,double eps){assert(partial==0x400000&&residual==0x100000&&raw==0x200000&&eps==1e-6);
 assert(gamma>=0x800000&&gamma<0x800000+36*4096);gammas.push_back(gamma);++counts.fused;}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t p,uint32_t r,uint32_t out,int64_t m,int64_t h,int tp,int cores){
 assert(m==224&&h==2048&&tp==8&&cores==8);
 if(p==0x400000){assert(r==0x100000&&out==0x200000);++counts.old_ring;}
 else {assert(p==0x700000&&r==0x200000&&out==0x100000);++counts.down_ring;}
}
void rpu_launch_rmsnorm_spm_kernel(uint32_t r,uint32_t out,uint32_t gamma,int64_t m,int64_t h,double eps,RpuRmsNormSpmRoute route,int cores){
 assert(r==0x200000&&out==0x100000&&m==224&&h==2048&&eps==1e-6&&cores==8&&route==RpuRmsNormSpmRoute::V32_GRID8);
 assert(gamma>=0x800000);++counts.norm;}
void rpu_launch_wall_base_prefill_gate_up_shared_silu_no_up_spm_kernel(uint32_t in,const at::Tensor&,const at::Tensor&,
 uint32_t gate,uint32_t up,int64_t rows,const at::Tensor&,const at::Tensor&){
 assert(in==0x100000&&gate==0x500000&&up==0x600000&&rows==224);++counts.gate;}
void rpu_launch_wall_base_prefill_gate_up_shared_fp16_spm_kernel(uint32_t in,const at::Tensor& gate_w,const at::Tensor& up_w,uint32_t gate,int64_t rows){
 assert(FP16&&gate_w.scalar_type()==at::kHalf&&up_w.scalar_type()==at::kHalf);
 assert(in==0x100000&&gate==0x500000&&rows==224);++counts.gate;}
auto rpu_launch_wall_base_prefill_gate_up_shared_silu_spm_kernel=rpu_launch_wall_base_prefill_gate_up_shared_silu_no_up_spm_kernel;
auto rpu_launch_wall_base_prefill_gate_up_shared_input_spm_kernel=rpu_launch_wall_base_prefill_gate_up_shared_silu_no_up_spm_kernel;
template<class... A>void rpu_launch_silu_mul_spm_kernel(A...){assert(false);}
template<class... A>void rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(A...){assert(false);}
template<class... A>void rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(A...){assert(false);}
template<class... A>void rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(A...){assert(false);}
struct Model {
 bool lingbot_folded_rmsnorm_=false;
 template<class...T>void emit_lingbot_nogamma_norm(T...){throw std::runtime_error("unbound LingBot normalization");}
 struct LayerWeights {at::Tensor q_w,k_w,v_w,o_w,gate_w,up_w,down_w,o_ws,gate_ws,up_ws,down_ws,post_norm_w;};
 std::vector<LayerWeights> layer_weights_=std::vector<LayerWeights>(36);WallSortingConfig wall_sorting_;
 bool has_qkv_bias_=true,has_qk_norm_=false,has_mrope_=true,use_silu_=true,scale=true;
 bool linear_acc32_=false,nvfp4_=false,decoder_mixed_precision_=false,adarms_=false,debug=false;
 bool wall_prefill_mixed_requested_=false,wall_prefill_staged_mlp_=false,wall_prefill_owner_direct_kv_=false,decode_gate_up_fusion_=false;
 bool pre_layers_residual1_ready_=false,wall_z1_dispatch_=false,qwen3vl_pooler_z1_dispatch_=false,qwen3vl_multiview_text_composite_dispatch_=false;
 bool wall_base_prefill_shared_input_=false,wall_base_prefill_shared_silu_=false,wall_base_prefill_no_up_store_=false;
 bool wall_base_fp16_long_prefill_=false,wall_base_fp16_shared_gateup_=false;
 bool wall_base_prefill_long_qkv_=false,wall_base_prefill_owner_norm_=false,rhinovla_high_precision_=false,rhinovla_high_precision_fusions_=false,qwen3vl_2b_w8_profile_=false;
 uint32_t wall_rmsnorm_capability_=RPU_RMSNORM_CAP_BASE;
 RpuRmsNormSpmContract ordinary_rmsnorm_spm_contract_{RPU_RMSNORM_CAP_ALL};
 std::vector<int32_t> mrope_section_{16,24,24};std::vector<int64_t> deepstack_lang_layers_;
 CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
 int64_t hidden=2048,intermediate=11008,nq=16,nkv=8,hd=128,tp=8,seq_len_=448;double eps_=1e-6;
 bool bad_alias=false;Context context;
 int64_t hidden_size()const{return hidden;}int64_t intermediate_size()const{return intermediate;}
 int64_t num_layers()const{return layer_weights_.size();}int64_t num_q_heads()const{return nq;}
 int64_t num_kv_heads()const{return nkv;}int64_t head_dim()const{return hd;}
 int attn_tp()const{return tp;}int mlp_tp()const{return tp;}int num_cores()const{return tp;}
 bool get_debug_export()const{return debug;}auto& ctx(){return context;}const auto& ctx()const{return context;}
 uint32_t addr(int,std::string s)const{return s=="residual1"?0x100000:s=="residual2"?0x200000:
  s=="input_norm"?(bad_alias?0x280000:0x200000):s=="output"?0x300000:s=="oproj"?0x400000:
  s=="gate"?0x500000:s=="up"?0x600000:0x700000;}
 uint32_t norm_post_addr(int layer,int)const{return 0x800000+layer*4096;}
 uint32_t layer_addr(int layer,int core,const char*)const{return norm_post_addr(layer,core);}
 uint32_t nvfp4_scale_addr(const char*)const{return 0;}
 bool qwen3vl_decode_gemv(int64_t)const{return false;}
 void emit_adarms_mut_refresh_post(int)const{assert(!adarms_);}
 RpuRmsNormSpmRoute consume_delivery_rmsnorm_route(int64_t,int64_t,int64_t,int)const{assert(false);return RpuRmsNormSpmRoute::BASE;}
 int consume_wall_mixed_rmsnorm_route(int64_t,int64_t,int64_t,int64_t)const{assert(false);return 0;}
 void consume_causal_route(FmbRouteFamily f,int64_t site,int64_t selector,int64_t invocation=0){context.consume_physical_route(f,site,selector,0,{},invocation);}
 bool consume_ordinary_o_route(int64_t site,int64_t,int64_t){consume_causal_route(FmbRouteFamily::LINEAR,site,int64_t(CausalDecoderLinearRoute::AUTO_TILE));return false;}
 bool ordinary_row_weight_reuse_plan_enabled(const FmbThreeStageChunkPlan&,bool,int64_t)const{return false;}
 std::vector<int64_t> ordinary_o_route_arguments(int64_t,bool)const{return {};}
 auto decoder_projection_route(int64_t)const{return CausalDecoderLinearRoute::AUTO_TILE;}
 HELPERS
 void bind(){
  int64_t N=num_layers(),hidden_size=hidden,intermediate_size=intermediate,num_q_heads=nq,num_kv_heads=nkv,head_dim=hd;
  bool has_scale=scale,has_nvfp4=nvfp4_,use_silu=use_silu_;double eps=eps_;
  std::vector<int64_t> mrope_section(mrope_section_.begin(),mrope_section_.end()),deepstack_lang_layers=deepstack_lang_layers_;
  std::vector<at::Tensor> q_w_list,k_w_list,v_w_list,o_w_list,gate_list,up_list,down_list,post_norm_list;
  for(auto& l:layer_weights_){q_w_list.push_back(l.q_w);k_w_list.push_back(l.k_w);v_w_list.push_back(l.v_w);o_w_list.push_back(l.o_w);
   gate_list.push_back(l.gate_w);up_list.push_back(l.up_w);down_list.push_back(l.down_w);post_norm_list.push_back(l.post_norm_w);}
  COLD
 }
 Model(){wall_sorting_.silu_mul=true;wall_sorting_.fp16_transfer_mask=FP16?2:0;scale=!FP16;for(int i=0;i<36;++i){auto& l=layer_weights_[i];
  for(auto* t:{&l.q_w,&l.k_w,&l.v_w,&l.o_w,&l.gate_w,&l.up_w,&l.down_w}){t->dtype=FP16?at::kHalf:at::kChar;}
  for(auto* t:{&l.o_ws,&l.gate_ws,&l.up_ws,&l.down_ws})t->present=!FP16;
  l.o_w.address=100+i;}
  set_rows(448);bind();}
 void set_rows(int64_t p){seq_len_=p;auto& plan=context.stage_plan;
  plan.input=make_stage({{0,0,p,p}});plan.qkv=plan.compute=make_stage({{0,0,p/2,p/2},{1,p/2,p/2,p}});}
 bool enabled()const{return wall_base_prefill_owner_norm_enabled(context.stage_plan,seq_len_,context.position,context.is_causal,context.attention_mask.has_value(),context.batch_size);}
 void build(int64_t logical=402){auto& plan=context.stage_plan;auto physical_len=seq_len_,position=context.position;
  struct{bool is_causal,use_attn_mask;int64_t batch_size;}layout{context.is_causal,context.attention_mask.has_value(),context.batch_size};
  const bool wall_prefill_owner_norm=enabled(),wall_action_linear_ring=false,qwen3vl4b_fused_o=false,wall_base_nvfp4_o_ring_norm=false,qwen3vl4b_fused_qk=false;
  context.value={};auto& manifest=context.value;manifest.physical_length=physical_len;manifest.logical_length=logical;manifest.execution_padding_rows=physical_len-logical;
  auto append=[&](FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags=0,std::vector<int64_t> args={},int64_t invocation=0){manifest.routes.push_back({site,f,selector,flags,args,invocation});};
  auto append_ring=[&](int64_t site,int64_t rows,int64_t i){append(FmbRouteFamily::ALL_REDUCE,site,causal_ring_route(rows,2048),0,{},i);};
  auto append_linear=[&](int64_t site,CausalDecoderLinearRoute route){append(FmbRouteFamily::LINEAR,site,int64_t(route));};
  NORM_PRODUCER
  PRE_PRODUCER
  for(const auto& chunk:plan.compute.chunks){OWNER_PRODUCER
   SHARED_PRODUCER
  }
  std::sort(manifest.routes.begin(),manifest.routes.end(),[](auto& a,auto& b){return std::make_tuple(a.family,a.site_id,a.invocation)<std::make_tuple(b.family,b.site_id,b.invocation);});
  context.consumed_.assign(manifest.routes.size(),0);
 }
 void emit_o(int layer,int chunk_index){const auto& chunk=context.stage_plan.compute.chunks.at(chunk_index);const auto& lw=layer_weights_.at(layer);
  int64_t rows=chunk.len,h=hidden,nq=this->nq,hd=this->hd,bs=context.batch_size;int tp=this->tp;int layer_idx=layer;
  uint32_t input_residual=addr(0,"residual1");bool residual_is_bf16=false;
  O_BODY
 }
 void emit_mlp(int layer,int i){
  if(FP16)emit_wall_base_fp16_shared_gateup(layer_weights_.at(layer),context.stage_plan.compute.chunks.at(i));
  else emit_wall_base_prefill_shared_input(layer_weights_.at(layer),context.stage_plan.compute.chunks.at(i));}
};
template<class F>void rejects(F&& f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}assert(failed);}
auto route_index(const Model& m,int64_t site,int64_t invocation=0){
 for(size_t i=0;i<m.context.value.routes.size();++i)if(m.context.value.routes[i].site_id==site&&m.context.value.routes[i].invocation==invocation)return i;
 return m.context.value.routes.size();}
int main(){
 publish();Model m;assert(m.enabled());m.build();
 assert(m.wall_base_fp16_long_prefill_==FP16&&m.wall_base_prefill_long_qkv_==!FP16);
 assert(m.context.value.logical_length==402&&m.context.value.physical_length==448&&m.context.value.execution_padding_rows==46);
 assert(m.physical_manifest_forward_capability(m.context.value).supported);
 for(int i=0;i<2;++i){
  assert(m.wall_base_prefill_owner_norm_arguments(i)==std::vector<int64_t>({1,224,2048,8,28,7,4,63488,1,6,1,1,448,i*224}));
  assert(route_index(m,CAUSAL_DECODER_REDUCE_SITE,i)==m.context.value.routes.size());
  assert(route_index(m,WALL_BASE_PREFILL_O_OWNER_NORM_SITE,i)<m.context.value.routes.size());
  auto ni=route_index(m,CAUSAL_DECODER_ORDINARY_RMSNORM_SITE,3*i);assert(ni<m.context.value.routes.size());
  assert(m.context.value.routes[ni].arguments==std::vector<int64_t>({224,2048,7}));
  // Input and final norm both still consume role0; postnorm now owns the new receipt.
  assert(m.consume_wall_rmsnorm_route(224,2048,i)==RpuRmsNormSpmRoute::V32_GRID8);
  assert(m.consume_wall_rmsnorm_route(224,2048,i)==RpuRmsNormSpmRoute::V32_GRID8);
 }
 counts={};gammas.clear();
 for(int layer=0;layer<36;++layer)for(int i=0;i<2;++i){m.emit_o(layer,i);m.emit_mlp(layer,i);}
 assert(counts.prepare==72&&counts.o==72&&counts.fused==72&&counts.old_ring==0&&counts.norm==0);
 assert(counts.gate==72&&counts.down==72&&counts.down_ring==72);
 for(int layer=0;layer<36;++layer)for(int i=0;i<2;++i)assert(gammas[layer*2+i]==uint32_t(0x800000+layer*4096));
 for(auto used:m.context.consumed_)assert(used==1);
 // Every new route argument and structural identity is checked before its fused launch.
 for(int i=0;i<2;++i){size_t index=route_index(m,WALL_BASE_PREFILL_O_OWNER_NORM_SITE,i);
  for(size_t a=0;a<14;++a){auto bad=m;++bad.context.value.routes[index].arguments[a];
   assert(!bad.physical_manifest_forward_capability(bad.context.value).supported);
   int before=counts.fused;rejects([&]{bad.emit_o(0,i);});assert(counts.fused==before);}
  for(int mutation=0;mutation<5;++mutation){auto bad=m;auto& r=bad.context.value.routes[index];
   switch(mutation){case 0:r.flags=1;break;case 1:r.selector++;break;case 2:r.invocation=2;break;case 3:r.invocation=-1;break;case 4:r.family=FmbRouteFamily::LINEAR;break;}
   assert(!bad.physical_manifest_forward_capability(bad.context.value).supported);int before=counts.fused;
   rejects([&]{bad.emit_o(0,i);});assert(counts.fused==before);}
 }
 for(int mutation=0;mutation<6;++mutation){auto bad=m;auto& manifest=bad.context.value;
  switch(mutation){case 0:manifest.routes.push_back(manifest.routes[route_index(bad,WALL_BASE_PREFILL_O_OWNER_NORM_SITE)]);break;
   case 1:manifest.routes.erase(manifest.routes.begin()+route_index(bad,WALL_BASE_PREFILL_O_OWNER_NORM_SITE));break;
   case 2:manifest.physical_length=512;break;case 3:manifest.linear_accumulation=FmbLinearAccumulationPolicy::ACC32;break;
   case 4:bad.wall_base_prefill_owner_norm_=false;break;case 5:bad.debug=true;break;}
  assert(!bad.physical_manifest_forward_capability(manifest).supported);
 }
 {auto bad=m;auto& r=bad.context.value.routes;r.erase(std::remove_if(r.begin(),r.end(),[](auto& x){return x.site_id==WALL_BASE_PREFILL_O_OWNER_NORM_SITE;}),r.end());
  // No-site is allowed at earlycap for other plans, but this actual C224 consume fails.
  assert(bad.physical_manifest_forward_capability(bad.context.value).supported);
  int before=counts.fused;rejects([&]{bad.emit_o(0,0);});assert(counts.fused==before);}
 {auto bad=m;bad.context.ready=false;rejects([&]{bad.consume_wall_base_prefill_owner_norm(bad.context.stage_plan.compute.chunks[0],0x100000);});}
 {auto bad=m;bad.bad_alias=true;rejects([&]{bad.consume_wall_base_prefill_owner_norm(bad.context.stage_plan.compute.chunks[0],0x100000);});}
 rejects([&]{m.consume_wall_base_prefill_owner_norm(m.context.stage_plan.compute.chunks[0],0x180000);});
 for(int mutation=0;mutation<4;++mutation){auto c=m.context.stage_plan.compute.chunks[0];
  switch(mutation){case 0:c.idx=-1;break;case 1:c.idx=2;break;case 2:c.len=223;break;case 3:c.offset=224;break;}
  rejects([&]{m.consume_wall_base_prefill_owner_norm(c,0x100000);});}
 // Plan rejection uses its explicit argument and never the prior forward's plan.
 for(int stage=0;stage<3;++stage)for(int mutation=0;mutation<8;++mutation){auto bad=m;
  auto* s=stage==0?&bad.context.stage_plan.input:stage==1?&bad.context.stage_plan.qkv:&bad.context.stage_plan.compute;
  switch(mutation){case 0:s->plan.chunk_size+=16;break;case 1:s->plan.num_chunks++;break;case 2:s->chunks.pop_back();break;
   case 3:s->chunks[0].idx++;break;case 4:s->chunks[0].offset+=16;break;case 5:s->chunks[0].len-=16;break;
   case 6:s->chunks[0].kv_seq_len++;break;case 7:s->chunks.push_back(s->chunks[0]);break;}
  assert(!bad.enabled());rejects([&]{bad.consume_wall_base_prefill_owner_norm(m.context.stage_plan.compute.chunks[0],0x100000);});}
 for(int64_t p:{32,224,336,384,512,640}){auto bad=m;bad.set_rows(p);assert(!bad.enabled());}
 for(auto cap:std::array<uint32_t,2>{RPU_RMSNORM_CAP_BASE,RPU_RMSNORM_CAP_BASE|RPU_RMSNORM_CAP_V16}){auto bad=m;bad.ordinary_rmsnorm_spm_contract_.capability=cap;assert(!bad.enabled());}
 {auto good=m;good.ordinary_rmsnorm_spm_contract_.capability=RPU_RMSNORM_CAP_BASE|RPU_RMSNORM_CAP_V32;assert(good.enabled());}
 for(bool Model::* flag:{&Model::debug,&Model::rhinovla_high_precision_,&Model::linear_acc32_,&Model::nvfp4_,&Model::decoder_mixed_precision_,
    &Model::adarms_,&Model::wall_prefill_mixed_requested_,&Model::wall_prefill_staged_mlp_,&Model::wall_prefill_owner_direct_kv_,
    &Model::decode_gate_up_fusion_,&Model::pre_layers_residual1_ready_,&Model::wall_z1_dispatch_,&Model::qwen3vl_pooler_z1_dispatch_,&Model::qwen3vl_multiview_text_composite_dispatch_}){
  auto bad=m;bad.*flag=true;assert(!bad.enabled());}
 for(int mutation=0;mutation<8;++mutation){auto bad=m;
  switch(mutation){case 0:bad.eps_=1e-5;break;case 1:bad.context.position=1;break;case 2:bad.context.is_causal=false;break;
   case 3:bad.context.attention_mask=1;break;case 4:bad.context.batch_size=2;break;case 5:bad.context.stage_plan.chunk_mode=ChunkMode::KV_FIRST;break;
   case 6:bad.wall_sorting_.silu_mul=false;bad.bind();break;case 7:bad.planning_mode_=CausalDecoderPlanningMode::WALL_Z1;break;}
  assert(!bad.enabled());}
 for(int mutation=0;mutation<7;++mutation){auto bad=m;
  switch(mutation){case 0:bad.layer_weights_[35].post_norm_w.dtype=at::kFloat;break;case 1:bad.layer_weights_[0].q_w.dtype=FP16?at::kChar:at::kHalf;break;
   case 2:bad.layer_weights_[35].down_w.dtype=at::kByte;break;case 3:bad.eps_=1e-5;break;case 4:bad.scale=FP16;break;
   case 5:bad.mrope_section_[1]=16;break;case 6:bad.layer_weights_.pop_back();break;}
  bad.bind();assert(!bad.wall_base_prefill_owner_norm_&&!bad.enabled());}
 // Every projection belongs to the cold uniform-precision proof, including
 // later layers. FP16 activations alone cannot admit a mixed weight owner.
 for(auto member:{&Model::LayerWeights::q_w,&Model::LayerWeights::k_w,&Model::LayerWeights::v_w,
                  &Model::LayerWeights::o_w,&Model::LayerWeights::gate_w,&Model::LayerWeights::up_w,
                  &Model::LayerWeights::down_w}){
  auto bad=m;(bad.layer_weights_[18].*member).dtype=FP16?at::kChar:at::kHalf;
  bad.bind();assert(!bad.enabled());}
 if(FP16){auto bad=m;bad.wall_sorting_.fp16_transfer_mask=0;bad.bind();assert(!bad.enabled());}
 // The two cold proofs are independent. Removing the other precision's
 // projection payloads cannot disable this activation-only ring/norm route.
 publish(true,false,true);{Model probe;assert(probe.enabled()==!FP16);}
 publish(true,true,false);{Model probe;assert(probe.enabled()==FP16);}
 publish(true,true,true,false);{Model probe;assert(probe.enabled());
  assert(!probe.wall_base_fp16_shared_gateup_);probe.build();
  counts={};for(int i=0;i<2;++i)probe.emit_o(0,i);
  assert(counts.fused==2&&counts.old_ring==0&&counts.norm==0);}
 publish();
 // Existing REF stays identical: only O ring+postnorm return; all other calls remain.
 publish(false);Model old;old.build();assert(!old.wall_base_prefill_owner_norm_&&!old.enabled());
 assert(!old.physical_manifest_forward_capability(m.context.value).supported);
 assert(old.physical_manifest_forward_capability(old.context.value).supported);
 counts={};for(int i=0;i<2;++i){old.emit_o(0,i);old.emit_mlp(0,i);}
 assert(counts.prepare==2&&counts.o==2&&counts.fused==0&&counts.old_ring==2&&counts.norm==2&&counts.down==2&&counts.down_ring==2);
 assert(m.enabled());publish();assert(!old.enabled());old.bind();assert(old.enabled());
 // Earlycap cannot use stale ctx; complete no-site non-C224 plans remain allowed.
 {auto old_plan=m;old_plan.set_rows(512);old_plan.build(490);
  assert(old_plan.physical_manifest_forward_capability(old_plan.context.value).supported);
  assert(old_plan.physical_manifest_forward_capability(m.context.value).supported);}
}
'''
    rms_types = decls[decls.index('enum RpuRmsNormCapability'):decls.index('enum class RpuMixedRmsNormSpmRoute')]
    substitutions = {
        'KERNEL_IDS': absolute_candidate_includes(block(cache, 'enum class KernelId : uint16_t') + ';'),
        'AVAILABILITY': block(cache, 'class KernelPayloadAvailability final') + ';',
        'PI_WALL_ID': block(cache, 'inline KernelId pi_wall_kernel_id('),
        'PI_WALL_RUNTIME': block(runtime, 'inline bool rpu_pi_wall_operation_loaded('),
        'ROUTE_FAMILY': block(base, 'enum class FmbRouteFamily : int') + ';',
        'GRAPH_LIFECYCLE': block(base, 'enum class FmbGraphLifecycle : int') + ';',
        'ACCUMULATION': block(base, 'enum class FmbLinearAccumulationPolicy : int') + ';',
        'MANIFEST_STATE': block(base, 'enum class FmbPhysicalManifestState : int') + ';',
        'LINEAR_ROUTES': block(decoder, 'enum class CausalDecoderLinearRoute') + ';',
        'REDUCE_ROUTES': block(decoder, 'enum class CausalDecoderAllReduceRoute') + ';',
        'RMS_TYPES': rms_types,
        'RMS_RESOLVE': block(decls, 'inline bool rpu_rmsnorm_capability_valid(') + '\n' + block(decls, 'inline RpuRmsNormSpmRoute rpu_resolve_rmsnorm_spm_route('),
        'RMS_HIGH_RESOLVE': block(decls, 'inline RpuRmsNormSpmRoute rpu_resolve_high_precision_rmsnorm_spm_route('),
        'RMS_CONTRACT': block(decls, 'struct RpuRmsNormSpmContract') + ';',
        'MAKE_STAGE': block(planner, 'ResolvedFmbStageChunks make_stage('),
        'CONSUMER': consumer, 'CONSTANTS': constants, 'HELPERS': helpers,
        'NORM_PRODUCER': norm_producer, 'PRE_PRODUCER': pre_producer,
        'OWNER_PRODUCER': owner_producer, 'SHARED_PRODUCER': shared_producer,
        'O_BODY': o_body,
    }
    for key, value in substitutions.items():
        source = re.sub(r'(?m)^[ \t]*' + re.escape(key) + r'[ \t]*$', lambda _: value, source)
    source = source.replace('IS_FP16', 'true' if fp16 else 'false')
    source = source.replace('COLD', cold).replace('{OWNER_PRODUCER', '{' + owner_producer)
    compile_run(tmp_path, 'wall_prefill_owner_norm_routes', source)
