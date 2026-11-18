"""Execute the actual long FP16 QKV and shared-input GateUp COMPLETE routes."""
from pathlib import Path
import re

from pi_wall_host_catalog import CATALOG_INCLUDE, absolute_candidate_includes
from test_wall_base_nvfp4_qkv_bias import META
from test_wall_base_paired_kv import block, compile_run

ROOT = Path(__file__).resolve().parents[1]


def test_actual_fp16_long_cold_plan_complete_and_activation(tmp_path):
    decoder = (ROOT / 'src/fused/rpu_qwen3_model.h').read_text()
    planner = (ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    base = (ROOT / 'src/core/fused_model_base.cpp').read_text()
    base_header = (ROOT / 'src/core/fused_model_base.h').read_text()
    cache = (ROOT / 'src/core/rpu_kernel_cache.h').read_text()
    runtime = (ROOT / 'src/ops/rpu_pi_wall_runtime.h').read_text()
    start = decoder.index('        wall_base_fp16_long_prefill_ =')
    end = decoder.index(');', decoder.index('        wall_base_fp16_shared_gateup_ =', start)) + 2
    cold = decoder[start:end]
    helpers = '\n'.join(block(decoder, marker) for marker in (
        '    bool wall_base_fp16_long_prefill_fusion(',
        '    std::vector<int64_t> wall_small_prefill_qkv_arguments(',
        '    bool wall_base_fp16_shared_gateup_enabled(',
        '    std::vector<int64_t> wall_base_fp16_shared_gateup_arguments(',
        '    void emit_wall_base_fp16_shared_gateup(',
    ))
    producer = block(decoder, 'else if (wall_long_prefill_qkv || wall_fp16_long_prefill_fused) {')[5:]
    consumer = block(decoder, 'else if (wall_p640_staged0_w8a16 || wall_small_prefill_fused ||')[5:]
    admission = re.findall(r'const bool wall_fp16_long_prefill_fused =[^;]+;', decoder)
    assert len(admission) == 2
    start = decoder.index('        const bool shared_prefill_silu =')
    activation_producer = decoder[start:decoder.index('        std::sort(manifest.routes', start)]
    mlp_start = decoder.index('        bool needs_shared_mlp = false;')
    mlp_producer = decoder[mlp_start:start]
    mlp_dispatch = block(decoder[decoder.index('void build_layer_subgraph('):], 'else if (wall_base_fp16_shared_gateup_enabled(')[5:]
    mlp_constants = '\n'.join(re.search(r'inline constexpr int64_t ' + name + r'\s*=.*?;', decoder, re.S).group()
                              for name in sorted(set(re.findall(r'\b(?:WALL_BASE|QWEN3VL|LINGBOT2)_\w+_SITE\b', mlp_producer))))
    mlp_call = decoder[decoder.index('            if (rhinovla_high_precision_) consume_rhinovla_high_activation(rows, chunk.idx);'):]
    fuse = re.search(r'/\*fuse_silu_mul=\*/(.*?),\s*/\*skip_down=', mlp_call, re.S).group(1)
    bind = re.search(r'/\*bind_silu_mul_route=\*/(.*?),\s*/\*residual_spm_addr=', mlp_call, re.S).group(1)
    activation_consumer = block(base, 'else if (fuse_silu_mul) {')[5:]
    consume = '\n'.join(block(planner, marker).replace('FmbPhysicalManifestConsumer::', '')
                        for marker in (
        'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(',
        'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(',
    ))
    # The default IntArrayRef={} is in the signature, before the function body.
    shared_start = base.index('void consume_shared_runtime_route(')
    shared_brace = base.index(') {', shared_start) + 2
    shared_consumer = base[shared_start:shared_brace] + block(base[shared_brace:], '{')
    q_site = re.search(r'inline constexpr int64_t CAUSAL_DECODER_Q_SITE\s*=.*?;', decoder, re.S).group()
    route_enum = block(decoder, 'enum class CausalDecoderLinearRoute').replace(
        'static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE)', '1').replace(
        'static_cast<int64_t>(FmbLinearRouteSelector::GEMV)', '2').replace(
        'static_cast<int64_t>(FmbLinearRouteSelector::ROW_WEIGHT_REUSE)', '5') + ';'
    source = CATALOG_INCLUDE + META + r'''
#include <atomic>
#include <bitset>
#include <memory>
#include <tuple>
#include "fused/rpu_qwen3_w8a8_contract.h"
#include "fused/rpu_qwen3_c192_contract.h"
namespace qwen3_w8a8 = v3::qwen3_w8a8;
namespace qwen3_c192 = v3::qwen3_c192;
namespace qwen3_c512 = v3::qwen3_c512;
#undef TORCH_INTERNAL_ASSERT
#define TORCH_INTERNAL_ASSERT(x) TORCH_CHECK(x)
KERNEL_IDS
AVAILABILITY
PI_WALL_ID
struct KernelCache {KernelPayloadAvailability availability;
 static auto& instance(){static KernelCache k;return k;}
 bool has_loaded(KernelId id)const{return availability.has_loaded(id);}};
PI_WALL_RUNTIME
void publish(bool candidate=true,bool gateup=false){KernelPayloadAvailability::ExtendedPayloads bits;
 for(const auto& p:rpu_pi_wall::profiles){
  if(p.weight!=rpu_pl_tiling::WeightType::Fp16)continue;
  if(p.operation==rpu_pi_wall::Operation::Qkv)
   bits.set(size_t(pi_wall_kernel_id(p.ordinal))-KernelPayloadAvailability::kExtendedBegin,candidate);
  if(p.operation==rpu_pi_wall::Operation::SharedSilu)
   bits.set(size_t(pi_wall_kernel_id(p.ordinal))-KernelPayloadAvailability::kExtendedBegin,gateup);
 }
 KernelCache::instance().availability.publish(false,false,false,false,false,false,bits);}
enum class ChunkMode {SEQUENTIAL,KV_FIRST};
enum class CausalDecoderPlanningMode {ORDINARY,WALL_Z1};
enum class FmbRouteFamily {ATTENTION,LINEAR,ACTIVATION,ALL_REDUCE,COLLECTIVE};
ROUTE_ENUM
Q_SITE
MLP_CONSTANTS
SHARED_FLAGS
constexpr int64_t FMB_SHARED_MLP_SILU_MUL_SITE=91;
struct ChunkInfo {int64_t idx,offset,len,kv_seq_len;};
struct Stage {struct{int64_t chunk_size,num_chunks;}plan;std::vector<ChunkInfo>chunks;};
struct FmbThreeStageChunkPlan {ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;Stage input,qkv,compute;};
struct FmbRouteManifestEntry {int64_t site_id;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>arguments;int64_t invocation;};
struct Manifest {std::vector<FmbRouteManifestEntry> routes;uint32_t shared_flags=0;};
void append_fmb_shared_runtime_routes(Manifest&m,const FmbThreeStageChunkPlan&,int64_t h,uint32_t flags,int64_t bs,int cores,int tp){
 (void)h;(void)bs;(void)cores;(void)tp;m.shared_flags=flags;}
int64_t causal_ring_route(int64_t m,int64_t h){assert(m>0&&h==2048);return 2;}
struct Context {Manifest value;mutable std::vector<uint8_t> consumed_;
 bool ready=true,is_causal=true;int64_t position=0,batch_size=1,physical_route_invocation=0;
 std::optional<int> attention_mask;FmbThreeStageChunkPlan stage_plan;
 bool complete()const{return ready;}bool has_complete_physical_manifest()const{return ready;}
 const auto& manifest()const{return value;}
 CONSUME
 const auto& consume_physical_route(FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,at::IntArrayRef args,int64_t i=0)const{
  return consume_route(f,site,selector,flags,args,i);}
};
using InferenceContext=Context;
SHARED_CONSUMER
enum class ActivationKind {NONE,SILU};
struct Launch {int layer;int64_t rows;};
std::vector<Launch> launches,gates,downs;std::vector<int64_t> activations,rings;
void rpu_launch_wall_base_prefill_gate_up_shared_fp16_spm_kernel(uint32_t in,const at::Tensor& g,const at::Tensor& u,uint32_t out,int64_t rows){
 int l=int(g.address)-400;assert(l>=0&&l<36&&u.address==500+l&&g.dtype==at::kHalf&&u.dtype==at::kHalf);
 assert(in==0x800000&&out==0x400000);gates.push_back({l,rows});}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t in,const at::Tensor& w,uint32_t out,int64_t rows,int64_t n,int64_t k,int part,int cores,int bias,const at::Tensor& scale){
 int l=int(w.address)-600;assert(l>=0&&l<36&&w.dtype==at::kHalf&&!scale.defined());
 assert(in==0x400000&&out==0x600000&&n==2048&&k==11008&&part==0&&cores==8&&!bias);downs.push_back({l,rows});}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t down,uint32_t residual,uint32_t out,int64_t rows,int64_t h,int cores,int outcores){
 assert(down==0x600000&&residual==0xa00000&&out==0x800000&&h==2048&&cores==8&&outcores==8);rings.push_back(rows);}
void rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(uint32_t input,const at::Tensor& qw,const at::Tensor& kw,const at::Tensor& vw,
 uint32_t q,uint32_t k,uint32_t v,uint32_t qb,uint32_t kb,uint32_t vb,const at::Tensor& qs,const at::Tensor& ks,const at::Tensor& vs,
 int64_t rows,bool owner,const WallSortingConfig& sorting){
 int layer=int(qw.address)-100;
 assert(layer>=0&&layer<36&&kw.address==200+layer&&vw.address==300+layer);
 assert(qw.dtype==at::kHalf&&kw.dtype==at::kHalf&&vw.dtype==at::kHalf&&!qs.defined()&&!ks.defined()&&!vs.defined());
 assert(input==0x100000&&q==0x300000&&k==0x340000&&v==0x360000&&!owner);
 assert(qb==0x700000+layer*1024&&kb==0x710000+layer*1024&&vb==0x720000+layer*1024);
 assert(sorting.silu_mul&&(sorting.fp16_transfer_mask&2));launches.push_back({layer,rows});
}
void rpu_launch_silu_mul_spm_kernel(uint32_t gate,uint32_t up,uint32_t output,int64_t elements,int cores){
 assert(gate==0x400000&&up==0x500000&&output==gate&&cores==8);activations.push_back(elements);}
struct Model {
 struct LayerWeights {at::Tensor q_w,k_w,v_w,o_w,gate_w,up_w,down_w,q_ws,k_ws,v_ws,down_ws;};
 std::vector<LayerWeights> layer_weights_=std::vector<LayerWeights>(36);
 WallSortingConfig wall_sorting_;
 bool wall_base_fp16_shared_gateup_=false,qwen3vl_4b_decode_gateup_fused_=false;
 bool wall_base_fp16_long_prefill_=false,scale=false,has_qkv_bias_=true,has_qk_norm_=false,has_mrope_=true,use_silu_=true;
 bool linear_acc32_=false,nvfp4_=false,decoder_mixed_precision_=false,wall_prefill_mixed_requested_=false;
 bool wall_prefill_staged_mlp_=false,wall_prefill_owner_direct_kv_=false,decode_gate_up_fusion_=false,adarms_=false;
 bool pre_layers_residual1_ready_=false,wall_z1_dispatch_=false,qwen3vl_pooler_z1_dispatch_=false,qwen3vl_multiview_text_composite_dispatch_=false;
 bool wall_base_prefill_shared_silu_=false,wall_action_fuse_silu_mul_=false;
 std::vector<int32_t> mrope_section_={16,24,24};std::vector<int64_t> deepstack_lang_layers_;
 CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
 int64_t hidden=2048,intermediate=11008,nq=16,nkv=8,hd=128,tp=8,mtp=8,cores=8,seq_len_=448;
 int64_t hidden_size()const{return hidden;}int64_t intermediate_size()const{return intermediate;}
 int64_t num_layers()const{return layer_weights_.size();}int64_t num_q_heads()const{return nq;}
 int64_t num_kv_heads()const{return nkv;}int64_t head_dim()const{return hd;}
 int attn_tp()const{return tp;}int mlp_tp()const{return mtp;}int num_cores()const{return cores;}
 Context context;Context& ctx(){return context;}
 struct Impl {Context&ctx_;int64_t intermediate_size_;}impl{context,11008};Impl*pimpl_=&impl;
 uint32_t addr(int,std::string s){return s=="input_norm"?0x100000:s=="q"?0x300000:s=="k"?0x340000:s=="v"?0x360000:s=="gate"?0x400000:s=="residual1"?0x800000:s=="residual2"?0xa00000:s=="down"?0x600000:0x500000;}
 uint32_t qkv_fused_bias_addr(int l,std::string s){return (s=="q_bias"?0x700000:s=="k_bias"?0x710000:0x720000)+l*1024;}
 HELPERS
 template<class...T>bool lingbot_prefill_gateup_for_plan(T...)const{return false;}
 std::vector<int64_t>lingbot_prefill_gateup_arguments()const{TORCH_CHECK(false,"unbound Lingbot GateUp");return {};}
 bool qwen3_c256_=false,rhinovla_high_precision_=false;
 bool ordinary_silu_mul_route_enabled()const{return false;}bool use_planned_silu_mul()const{return false;}
 bool wall_base_prefill_shared_input_enabled(const FmbThreeStageChunkPlan&,int64_t,int64_t,bool,bool,int64_t)const{return false;}
 int64_t wall_base_prefill_shared_gateup_site()const{return 0;}
 std::vector<int64_t>wall_base_prefill_shared_input_arguments(int64_t)const{return {};}
 template<class...T>bool qwen3vl_4b_prefill_swiglu(T...)const{return false;}
 template<class...T>bool qwen3vl_decode_mlp(T...)const{return false;}
 std::vector<int64_t> qwen3vl_prefill_gateup_contract_arguments(int64_t)const{TORCH_CHECK(false,"unbound Qwen prefill GateUp");return {};}
 std::vector<int64_t> qwen3vl_decode_gateup_contract_arguments()const{TORCH_CHECK(false,"unbound Qwen decode GateUp");return {};}
 bool qwen3vl_decode_gemv(int64_t)const{return false;}
 void bind(){const int64_t N=num_layers(),hidden_size=hidden,intermediate_size=intermediate,num_q_heads=nq,num_kv_heads=nkv,head_dim=hd;
  const bool has_scale=scale,has_nvfp4=nvfp4_,use_silu=use_silu_;
  const std::vector<int64_t>mrope_section(mrope_section_.begin(),mrope_section_.end()),deepstack_lang_layers=deepstack_lang_layers_;
  std::vector<at::Tensor>q_w_list,k_w_list,v_w_list,o_w_list,gate_list,up_list,down_list;
  for(auto& lw:layer_weights_){q_w_list.push_back(lw.q_w);k_w_list.push_back(lw.k_w);v_w_list.push_back(lw.v_w);o_w_list.push_back(lw.o_w);gate_list.push_back(lw.gate_w);up_list.push_back(lw.up_w);down_list.push_back(lw.down_w);}
  COLD
 }
 Model(){wall_sorting_.silu_mul=true;wall_sorting_.fp16_transfer_mask=2;
  for(int l=0;l<36;++l){auto& lw=layer_weights_[l];lw.q_w.address=100+l;lw.k_w.address=200+l;lw.v_w.address=300+l;
   lw.gate_w.address=400+l;lw.up_w.address=500+l;lw.down_w.address=600+l;
   lw.q_ws.present=lw.k_ws.present=lw.v_ws.present=lw.down_ws.present=false;}set_rows(448);bind();}
 void set_rows(int64_t p){seq_len_=p;auto& plan=context.stage_plan;plan.input.plan={p,1};plan.input.chunks={{0,0,p,p}};
  plan.qkv.plan=plan.compute.plan={p/2,2};plan.qkv.chunks=plan.compute.chunks={{0,0,p/2,p/2},{1,p/2,p/2,p}};}
 bool enabled()const{return wall_base_fp16_long_prefill_fusion(context.stage_plan,seq_len_,context.position,context.is_causal,context.attention_mask.has_value(),context.batch_size);}
 void build(){const auto& plan=context.stage_plan;const int64_t physical_len=seq_len_,position=context.position;
  struct{bool is_causal,use_attn_mask;int64_t batch_size;int attention_policy;}layout{context.is_causal,context.attention_mask.has_value(),context.batch_size,0};
  const bool wall_long_prefill_qkv=false,wall_small_prefill_fused=false;
  PRODUCER_ADMISSION
  context.value.routes.clear();
  auto&manifest=context.value;const bool multiview_mode=false,pooler_z1_mode=false,p640_staged=false,wall_action_linear_ring=false;
  const bool qwen_a8=false,qwen_inplace=false,qwen_c512=false;
  auto append_a8_quant=[&](const ChunkInfo&,int64_t){};
  auto append_a8_linear=[&](const ChunkInfo&,qwen3_w8a8::Projection){};
  auto append=[&](FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t> args,int64_t i){context.value.routes.push_back({site,family,selector,flags,args,i});};
  auto append_ring=[&](int64_t site,int64_t rows,int64_t i){append(FmbRouteFamily::ALL_REDUCE,site,causal_ring_route(rows,2048),0,{},i);};
  PRODUCER
  MLP_PRODUCER
  ACTIVATION_PRODUCER
  std::sort(context.value.routes.begin(),context.value.routes.end(),[](auto&a,auto&b){return std::make_tuple(a.family,a.site_id,a.invocation)<std::make_tuple(b.family,b.site_id,b.invocation);});
  context.consumed_.assign(context.value.routes.size(),0);
 }
 void emit_qkv(int invocation=0,int layer_idx=0){const int64_t bs=context.batch_size;
  const auto&chunk=context.stage_plan.compute.chunks.at(invocation);const int64_t rows=chunk.len;const auto&lw=layer_weights_.at(layer_idx);
  const bool wall_long_prefill_qkv=false,wall_small_prefill_fused=false,wall_p640_staged0_w8a16=false;
  CONSUMER_ADMISSION
  CONSUMER
 }
 void emit_activation(int invocation=0){const int64_t bs=context.batch_size;const auto&chunk=context.stage_plan.compute.chunks.at(invocation);
  const int64_t rows=chunk.len,seq_len=rows,elems=rows*intermediate/mtp;const int kNumCores=mtp;
  context.physical_route_invocation=chunk.idx;const auto act=ActivationKind::SILU;
  const bool wall_small_prefill_fused=false,wall_p640_staged0_w8a16=false,wall_action_w8a16_silu_mul=false,rtc_prefill_mixed=false;
  CONSUMER_ADMISSION
  const bool fuse_silu_mul=FUSE;
  const bool bind_silu_mul_route=BIND;
  ACTIVATION_CONSUMER
 }
 void emit_mlp(int invocation=0,int layer=0){const auto&chunk=context.stage_plan.compute.chunks.at(invocation);const auto&lw=layer_weights_.at(layer);const auto bs=context.batch_size;
  MLP_DISPATCH else {emit_activation(invocation);}}
 int64_t layout_identity()const{int64_t h=0;LAYOUT_IDENTITY return h;}
 std::vector<int64_t> cost_identity()const{std::vector<int64_t>identity;COST_IDENTITY return identity;}
 std::vector<int64_t> topology_identity()const{std::vector<int64_t>arguments;TOPOLOGY_IDENTITY return arguments;}
};
template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}assert(failed);}
void fallback(Model&m){assert(!m.enabled());assert(!m.wall_base_fp16_shared_gateup_enabled(m.context.stage_plan,m.seq_len_,m.context.position,m.context.is_causal,m.context.attention_mask.has_value(),m.context.batch_size));m.build();assert(m.context.value.routes.empty());
 const auto q=launches.size(),a=activations.size();m.emit_qkv();m.emit_activation();assert(q==launches.size()&&(m.cores!=8||a==activations.size()));}
int main(){
 publish();
 for(int64_t p:{448,512,640}){Model m;m.set_rows(p);assert(m.enabled());m.build();const auto valid=m.context.value.routes;const int64_t rows=p/2;
  assert(valid.size()==4);
  const auto selected=rpu_pi_wall::select(rpu_pi_wall::Operation::Qkv,rpu_pl_tiling::WeightType::Fp16,false,rows,256,2048);
  for(int c=0;c<2;++c){const auto&q=valid[c];const auto&a=valid[c+2];
   assert(q.family==FmbRouteFamily::LINEAR&&q.site_id==CAUSAL_DECODER_Q_SITE&&q.selector==8&&q.flags==0&&q.invocation==c);
   assert(q.arguments==std::vector<int64_t>({rows,2048,2048,1024,8,selected.tile.m_tile,selected.tile.n_tile,
    selected.tile.k_tile,16,selected.grid_x,selected.grid_y,2,32,40,20,34,42,4,36,44,6,38,46}));
   assert(a.family==FmbRouteFamily::ACTIVATION&&a.site_id==FMB_SHARED_MLP_SILU_MUL_SITE&&a.selector==1&&a.flags==0&&a.invocation==c);
   assert(a.arguments==std::vector<int64_t>({rows,1376,8}));}
  launches.clear();activations.clear();
  for(int c=0;c<2;++c)for(int l=0;l<36;++l){m.emit_qkv(c,l);m.emit_activation(c);}
  assert(launches.size()==72&&activations==std::vector<int64_t>(72,rows*1376));
  for(size_t i=0;i<72;++i)assert(launches[i].layer==int(i%36)&&launches[i].rows==rows);
  assert(m.context.consumed_==std::vector<uint8_t>({1,1,1,1}));
  // Each producer argument and identity is independently consumed, including invocation 1.
  for(int r=0;r<4;++r){const int invocation=r%2;auto emit=[&]{if(r<2)m.emit_qkv(invocation,35);else m.emit_activation(invocation);};
   for(size_t a=0;a<valid[r].arguments.size();++a){m.context.value.routes=valid;++m.context.value.routes[r].arguments[a];const auto q=launches.size(),v=activations.size();rejects(emit);assert(q==launches.size()&&v==activations.size());}
   for(int f=0;f<7;++f){m.context.value.routes=valid;auto&route=m.context.value.routes[r];switch(f){
    case 0:++route.selector;break;case 1:++route.flags;break;case 2:route.invocation=2;break;case 3:++route.site_id;break;
    case 4:route.family=FmbRouteFamily::ATTENTION;break;case 5:route.arguments.push_back(0);break;case 6:m.context.value.routes.erase(m.context.value.routes.begin()+r);break;}
    const auto q=launches.size(),v=activations.size();rejects(emit);assert(q==launches.size()&&v==activations.size());}
  }
  m.context.value.routes=valid;m.context.ready=false;rejects([&]{m.emit_qkv();});m.context.ready=true;
  // Cold route choice is bound independently into layout, cost and topology identity.
  const auto layout=m.layout_identity();const auto cost=m.cost_identity(),topology=m.topology_identity();
  assert(layout!=0&&cost==std::vector<int64_t>({CAUSAL_DECODER_Q_SITE,16,FMB_SHARED_MLP_SILU_MUL_SITE,1})&&cost==topology);
  m.wall_base_fp16_long_prefill_=false;assert(m.layout_identity()==0&&m.cost_identity().empty()&&m.topology_identity().empty());
  for(int stage=0;stage<3;++stage)for(int bad=0;bad<13;++bad){Model x;x.set_rows(p);auto&plan=x.context.stage_plan;Stage&s=stage==0?plan.input:stage==1?plan.qkv:plan.compute;
   switch(bad){case 0:++s.plan.chunk_size;break;case 1:++s.plan.num_chunks;break;case 2:s.chunks.pop_back();break;case 3:s.chunks.push_back(s.chunks.back());break;
    case 4:++s.chunks.front().idx;break;case 5:++s.chunks.front().offset;break;case 6:--s.chunks.front().len;break;case 7:--s.chunks.front().kv_seq_len;break;
    case 8:++s.chunks.back().idx;break;case 9:--s.chunks.back().offset;break;case 10:++s.chunks.back().len;break;case 11:++s.chunks.back().kv_seq_len;break;case 12:plan.chunk_mode=ChunkMode::KV_FIRST;break;}
   assert(!x.enabled());x.build();assert(x.context.value.routes.empty());}
 }
 publish(true,true);
 // Every projection and layer is checked at installation, including the final layer.
 for(int l=0;l<36;++l)for(auto member:{&Model::LayerWeights::q_w,&Model::LayerWeights::k_w,&Model::LayerWeights::v_w,&Model::LayerWeights::o_w,&Model::LayerWeights::gate_w,&Model::LayerWeights::up_w,&Model::LayerWeights::down_w})
  for(auto dtype:{at::kChar,at::kByte}){Model x;(x.layer_weights_[l].*member).dtype=dtype;x.bind();fallback(x);}
 for(bool Model::*flag:{&Model::scale,&Model::has_qk_norm_,&Model::linear_acc32_,&Model::nvfp4_,&Model::decoder_mixed_precision_,
   &Model::wall_prefill_mixed_requested_,&Model::wall_prefill_staged_mlp_,&Model::wall_prefill_owner_direct_kv_,&Model::decode_gate_up_fusion_,&Model::adarms_,
   &Model::pre_layers_residual1_ready_,&Model::wall_z1_dispatch_,&Model::qwen3vl_pooler_z1_dispatch_,&Model::qwen3vl_multiview_text_composite_dispatch_}){Model x;x.*flag=true;x.bind();fallback(x);}
 for(int64_t Model::*dim:{&Model::hidden,&Model::intermediate,&Model::nq,&Model::nkv,&Model::hd,&Model::tp,&Model::mtp,&Model::cores}){Model x;++(x.*dim);x.bind();fallback(x);}
 for(int bad=0;bad<15;++bad){Model x;switch(bad){case 0:x.wall_sorting_.silu_mul=false;break;case 1:x.wall_sorting_.fp16_transfer_mask=4;break;
  case 2:x.has_qkv_bias_=false;break;case 3:x.use_silu_=false;break;case 4:x.mrope_section_[1]=16;break;case 5:x.deepstack_lang_layers_={0};break;
  case 6:x.layer_weights_.pop_back();break;case 7:x.layer_weights_.push_back(x.layer_weights_.back());break;case 8:x.has_mrope_=false;break;
  case 9:x.planning_mode_=CausalDecoderPlanningMode::WALL_Z1;break;case 10:x.context.position=1;break;case 11:x.context.is_causal=false;break;
  case 12:x.context.attention_mask=1;break;case 13:x.context.batch_size=2;break;case 14:x.context.stage_plan.chunk_mode=ChunkMode::KV_FIRST;break;}x.bind();fallback(x);}
 for(int64_t p:{32,224,256,288,320,336,384,416,447,449,480,511,513,576,639,641,672}){Model x;x.set_rows(p);fallback(x);}

 // New capability is independent: a missing symbol preserves all R52 routes.
 for(bool q_loaded:{false,true})for(bool g_loaded:{false,true}){
  publish(q_loaded,g_loaded);Model x;assert(x.wall_base_fp16_long_prefill_==q_loaded);
  assert(x.wall_base_fp16_shared_gateup_==(q_loaded&&g_loaded));}
 for(int64_t p:{448,512,640})for(bool loaded:{false,true,false,true}){
  publish(true,loaded);Model m;m.set_rows(p);m.build();const auto valid=m.context.value.routes;const auto rows=p/2;
  auto matching=[&](int64_t site){return std::count_if(valid.begin(),valid.end(),[&](auto&r){return r.site_id==site;});};
  assert(matching(CAUSAL_DECODER_Q_SITE)==2);
  assert(matching(WALL_BASE_FP16_PREFILL_SHARED_GATEUP_SITE)==(loaded?2:0));
  assert(matching(WALL_BASE_PREFILL_SHARED_DOWN_SITE)==(loaded?2:0)&&matching(WALL_BASE_PREFILL_SHARED_REDUCE_SITE)==(loaded?2:0));
  assert(matching(FMB_SHARED_MLP_SILU_MUL_SITE)==(loaded?0:2));
  assert(m.context.value.shared_flags==(FMB_SHARED_LAYER_INPUT_DMA|(loaded?0u:uint32_t(FMB_SHARED_MLP_AUTO_TILE|FMB_SHARED_MLP_RING_REDUCE))));
  launches.clear();activations.clear();gates.clear();downs.clear();rings.clear();
  for(int c=0;c<2;++c)for(int l=0;l<36;++l){m.emit_qkv(c,l);m.emit_mlp(c,l);}
  assert(launches.size()==72&&activations.size()==(loaded?0:72));
  assert(gates.size()==(loaded?72:0)&&downs.size()==gates.size()&&rings.size()==gates.size());
  for(size_t i=0;i<gates.size();++i){assert(gates[i].layer==int(i%36)&&gates[i].rows==rows);assert(downs[i].layer==gates[i].layer&&downs[i].rows==rows&&rings[i]==rows);}
  assert(std::all_of(m.context.consumed_.begin(),m.context.consumed_.end(),[](auto x){return x==1;}));
  if(!loaded)continue;
  auto oldlayout=m.layout_identity();auto oldcost=m.cost_identity(),oldtop=m.topology_identity();
  m.wall_base_fp16_shared_gateup_=false;
  assert(m.layout_identity()!=oldlayout&&m.cost_identity()!=oldcost&&m.topology_identity()!=oldtop&&m.enabled());
  assert(m.cost_identity()==std::vector<int64_t>({CAUSAL_DECODER_Q_SITE,16,FMB_SHARED_MLP_SILU_MUL_SITE,1}));
  m.wall_base_fp16_shared_gateup_=true;
  for(size_t r=0;r<valid.size();++r){const auto&v=valid[r];if(v.site_id==CAUSAL_DECODER_Q_SITE)continue;
   if(v.site_id==WALL_BASE_FP16_PREFILL_SHARED_GATEUP_SITE){
    const auto selected=rpu_pi_wall::select(rpu_pi_wall::Operation::SharedSilu,rpu_pl_tiling::WeightType::Fp16,false,rows,1376,2048);
    assert(v.arguments==std::vector<int64_t>({2,rows,11008,2048,1376,8,selected.tile.m_tile,selected.tile.n_tile,
     selected.tile.k_tile,selected.grid_x,selected.grid_y,1,16,1,2,32,26,27}));
   }
   else if(v.site_id==WALL_BASE_PREFILL_SHARED_DOWN_SITE)assert(v.arguments==std::vector<int64_t>({rows,2048,11008,0,8}));
   else assert(v.site_id==WALL_BASE_PREFILL_SHARED_REDUCE_SITE&&v.arguments.empty());
   auto fail=[&]{auto g=gates.size(),d=downs.size(),a=activations.size(),reduce=rings.size();rejects([&]{m.emit_mlp(v.invocation,35);});
    assert(activations.size()==a&&rings.size()==reduce);
    assert(gates.size()==g+(v.site_id==WALL_BASE_FP16_PREFILL_SHARED_GATEUP_SITE?0:1));
    assert(downs.size()==d+(v.site_id==WALL_BASE_PREFILL_SHARED_REDUCE_SITE?1:0));};
   for(size_t a=0;a<v.arguments.size();++a){m.context.value.routes=valid;++m.context.value.routes[r].arguments[a];fail();}
   for(int f=0;f<7;++f){m.context.value.routes=valid;auto&route=m.context.value.routes[r];switch(f){
    case 0:++route.selector;break;case 1:++route.flags;break;case 2:route.invocation=2;break;case 3:++route.site_id;break;
    case 4:route.family=FmbRouteFamily::ATTENTION;break;case 5:route.arguments.push_back(0);break;case 6:m.context.value.routes.erase(m.context.value.routes.begin()+r);break;}fail();}
  }
  m.context.value.routes=valid;m.context.ready=false;rejects([&]{m.emit_mlp();});m.context.ready=true;
  for(int bad=0;bad<4;++bad){auto chunk=m.context.stage_plan.compute.chunks[1];switch(bad){case 0:chunk.idx=2;break;case 1:++chunk.offset;break;case 2:--chunk.len;break;case 3:--chunk.kv_seq_len;break;}
   auto g=gates.size();rejects([&]{m.emit_wall_base_fp16_shared_gateup(m.layer_weights_[35],chunk);});assert(g==gates.size());}
  for(int stage=0;stage<3;++stage)for(int bad=0;bad<13;++bad){Model x;x.set_rows(p);auto&plan=x.context.stage_plan;Stage&s=stage==0?plan.input:stage==1?plan.qkv:plan.compute;
   switch(bad){case 0:++s.plan.chunk_size;break;case 1:++s.plan.num_chunks;break;case 2:s.chunks.pop_back();break;case 3:s.chunks.push_back(s.chunks.back());break;
    case 4:++s.chunks.front().idx;break;case 5:++s.chunks.front().offset;break;case 6:--s.chunks.front().len;break;case 7:--s.chunks.front().kv_seq_len;break;
    case 8:++s.chunks.back().idx;break;case 9:--s.chunks.back().offset;break;case 10:++s.chunks.back().len;break;case 11:++s.chunks.back().kv_seq_len;break;case 12:plan.chunk_mode=ChunkMode::KV_FIRST;break;}
   auto g=gates.size();rejects([&]{x.emit_wall_base_fp16_shared_gateup(x.layer_weights_[0],m.context.stage_plan.compute.chunks[0]);});assert(g==gates.size());}
 }
 // Actual atomic publication supports absent/present/absent without a legacy availability bit.
 for(bool loaded:{false,true,false,true}){publish(loaded);Model x;assert(x.wall_base_fp16_long_prefill_==loaded);if(!loaded)fallback(x);}
}
'''
    layout = block(decoder, '    int64_t subclass_layout_hash() const override')
    # Keep the exact new identity clauses; unrelated owner fields are intentionally absent.
    cost_start = decoder.index('    std::vector<int64_t> causal_kvinsert_cost_weight_identity(')
    cost_body = block(decoder[cost_start:], '    std::vector<int64_t> causal_kvinsert_cost_weight_identity(')
    topology = block(decoder, '    std::vector<int64_t> wall_sorting_route_arguments() const')
    source = source.replace('struct Model {', '''namespace detail {int64_t layout_mix(int64_t h,int64_t v){return (uint64_t(h)^uint64_t(v))*1099511628211ULL;}}
struct Model {''')
    for name, value in (
        ('KERNEL_IDS', absolute_candidate_includes(block(cache, 'enum class KernelId') + ';')),
        ('AVAILABILITY', block(cache, 'class KernelPayloadAvailability final') + ';'),
        ('PI_WALL_ID', block(cache, 'inline KernelId pi_wall_kernel_id(')),
        ('PI_WALL_RUNTIME', block(runtime, 'inline bool rpu_pi_wall_operation_loaded(')),
        ('HELPERS', helpers), ('COLD', cold), ('CONSUME', consume),
        ('SHARED_CONSUMER', shared_consumer),
        ('ROUTE_ENUM', route_enum), ('Q_SITE', q_site),
        ('MLP_CONSTANTS', mlp_constants), ('SHARED_FLAGS', block(base_header, 'enum FmbSharedRuntimeRouteMask') + ';'),
        ('MLP_PRODUCER', mlp_producer), ('MLP_DISPATCH', mlp_dispatch),
        ('PRODUCER_ADMISSION', admission[0]), ('CONSUMER_ADMISSION', admission[1]),
        ('ACTIVATION_PRODUCER', activation_producer), ('PRODUCER', producer),
        ('ACTIVATION_CONSUMER', activation_consumer), ('CONSUMER', consumer),
        ('FUSE', fuse), ('BIND', bind),
        ('LAYOUT_IDENTITY', block(layout, 'if (wall_base_fp16_long_prefill_)') + block(layout, 'if (wall_base_fp16_shared_gateup_)')),
        ('COST_IDENTITY', block(cost_body, 'if (wall_base_fp16_long_prefill_)') + re.search(r'if \(wall_base_fp16_shared_gateup_\)[\s\S]*?;', cost_body).group()),
        ('TOPOLOGY_IDENTITY', re.search(r'if \(wall_base_fp16_long_prefill_\)[\s\S]*?;', topology).group() + re.search(r'if \(wall_base_fp16_shared_gateup_\)[\s\S]*?;', topology).group()),
    ):
        source = re.sub(r"\b" + name + r"\b", lambda _: value, source)
    compile_run(tmp_path, 'wall_fp16_long_prefill_routes', source)


def test_actual_fp16_shared_leaf_ranges_typed_packet_and_fallback(tmp_path):
    linear = (ROOT / 'src/ops/rpu_linear.cpp').read_text()
    leaf = block(linear, 'void rpu_launch_wall_base_prefill_gate_up_shared_fp16_spm_kernel(')
    optional = block((ROOT / 'src/core/rpu_kernel_cache.inc').read_text(),
                     'static bool is_optional_kernel(')
    source = CATALOG_INCLUDE + META + r'''
#include "ops/rpu_linear_tiling.h"
#include <unordered_set>
OPTIONAL
enum class KernelId:std::size_t {};
KernelId pi_wall_kernel_id(std::size_t ordinal){return static_cast<KernelId>(ordinal);}
KernelId chosen;
enum class GraphDdrRegisterRole {ModelWeight};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel {std::array<uint16_t,68> regs{};int fail=-1,resets=0;
 void reset_regs(){regs.fill(0);++resets;}
 uint32_t set_regs(int r,uint16_t v){if(r==fail)return 7;regs.at(r)=v;return 0;}}kernel;
int staged=0,queued=0;bool deny_stage=false,deny_write=false,kernel_present=true;
struct Writer {std::vector<GraphDdrRegisterOperandSpec> operands;void write(Kernel& k){
 TORCH_CHECK(!deny_write);for(const auto& o:operands){auto a=o.tensor.address>>8;
 TORCH_CHECK(k.set_regs(o.abi.lo,uint16_t(a))==0);TORCH_CHECK(k.set_regs(o.abi.hi,uint16_t(a>>16))==0);}}};
struct RpuKernelGraph {static auto& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  TORCH_CHECK(!deny_stage);chosen=id;assert(v.size()==2);
  const int pairs[]={2,32};for(int i=0;i<2;++i){const auto&o=v[i];assert(o.abi.role==GraphDdrRegisterRole::ModelWeight&&o.abi.access==GraphDdrRegisterAccess::Read&&o.abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi&&o.abi.lo==pairs[i]&&o.abi.hi==pairs[i]+1&&o.tensor.dtype==at::kHalf);}
  ++staged;return {v};}};
struct Queue {bool broadcast=false;std::vector<uint16_t> grid;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> g,std::vector<uint8_t> cores){
 assert(broadcast&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));grid=g;++queued;}}queue;
#define GET_KERNEL(id) (kernel_present?&::kernel:nullptr)
#define GET_QUEUE(n) (&::queue)
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}uint32_t addr(int core,int off)const{return base+core*0x1000000u+off;}}spm;
#define SPM_ALLOC spm
struct KernelCache {bool loaded=true;static auto& instance(){static KernelCache c;return c;}bool has_loaded(KernelId)const{return loaded;}};
uint64_t RpuGetDevAddr(void*p){return reinterpret_cast<uintptr_t>(p);}void rpu_ddr_flush(void*){}
LEAF
uint32_t word(int r){return kernel.regs[r]|uint32_t(kernel.regs[r+1])<<16;}
template<class F>void rejects(F f,bool preflight=true){auto s=staged,q=queued;bool failed=false;try{f();}catch(const std::exception&){failed=true;}assert(failed&&queued==q&&(!preflight||staged==s));}
int main(){
 assert(!is_optional_kernel("llama_insert_kv_cache_multiwarp_v16"));
 at::Tensor g,u;g.shape=u.shape={11008,2048};g.dtype=u.dtype=at::kHalf;g.address=0x10000000;u.address=0x14000000;
 uint32_t input=0x20000100,output=0x20200000;int64_t rows=224;
 auto run=[&]{rpu_launch_wall_base_prefill_gate_up_shared_fp16_spm_kernel(input,g,u,output,rows);};
 for(int64_t m:{224,256,320}){rows=m;
  const auto selected=rpu_pi_wall::select(rpu_pi_wall::Operation::SharedSilu,rpu_pl_tiling::WeightType::Fp16,false,rows,1376,2048);
  assert(!is_optional_kernel(rpu_pi_wall::candidate_names[selected.ordinal]));
  for(int repeat=0;repeat<2;++repeat){kernel.regs.fill(0xffff);auto old=kernel.resets;run();assert(kernel.resets==old+1);
  assert(chosen==pi_wall_kernel_id(selected.ordinal)&&queue.grid==std::vector<uint16_t>({selected.grid_x,selected.grid_y,1}));
  assert(word(0)==input&&word(2)==g.address>>8&&word(4)==output&&word(6)==0&&kernel.regs[8]==1&&kernel.regs[9]==rows);
  assert(word(10)==1376&&kernel.regs[12]==2048&&kernel.regs[13]==128&&kernel.regs[14]==0&&kernel.regs[15]==8);
  assert(word(16)==4096&&word(18)==2752&&word(20)==0&&word(22)==0&&word(24)==0&&word(32)==u.address>>8);
  assert(kernel.regs[26]==2&&kernel.regs[27]==2&&word(28)==0&&word(30)==0);}}
 for(int m:{0,32,208,223,225,240,255,257,272,304,319,321,336,384,448,512,640}){rows=m;rejects(run);}rows=320;
 const auto input_old=input,output_old=output;
 for(auto* address:{&input,&output}){auto old=*address;*address=0;rejects(run);*address=old+32;rejects(run);*address=spm.base-256;rejects(run);*address=spm.base+SpmAllocator::SPM_USABLE-256;rejects(run);*address=old;}
 output=input+256;rejects(run);output=input;rejects(run);output=output_old;
 // Adjacency is legal; partial overlap and envelope truncation are not.
 output=input+rows*2048*2;run();output=output_old;
 output=spm.base+SpmAllocator::SPM_USABLE-rows*1376*2;run();output+=256;rejects(run);output=output_old;
 input=spm.base+SpmAllocator::SPM_USABLE-rows*2048*2;run();input+=256;rejects(run);input=input_old;
 // Core 0 still fits, but the core-7 broadcast envelope overflows uint32.
 spm.base=0xf8f80000;input=spm.base;output=spm.base+0x200000;rejects(run);spm.base=0x20000000;input=input_old;output=output_old;
 for(auto*t:{&g,&u}){auto old=*t;
  t->present=false;rejects(run);*t=old;t->shape[0]--;rejects(run);*t=old;t->shape[1]--;rejects(run);*t=old;
  for(auto dtype:{at::kChar,at::kByte,at::kFloat}){t->dtype=dtype;rejects(run);}*t=old;
  t->storage_off=1;rejects(run);*t=old;t->contiguous=false;rejects(run);*t=old;t->where.kind=c10::DeviceType::CPU;rejects(run);*t=old;
  t->address=0;rejects(run);*t=old;t->address++;rejects(run);*t=old;t->address=uint64_t{1}<<40;rejects(run);*t=old;
  t->address=(uint64_t{1}<<40)-256;rejects(run);*t=old;
  t->address=(uint64_t{1}<<40)-t->nbytes();run();t->address+=256;rejects(run);*t=old;
 }
 auto old=u;u.where.index=1;rejects(run);u=old;u.address=g.address+256;rejects(run);u=old;
 u.address=g.address+g.nbytes();run();u=old;
 KernelCache::instance().loaded=false;rejects(run);KernelCache::instance().loaded=true;
 spm.ready=false;rejects(run);spm.ready=true;
 deny_stage=true;rejects(run);deny_stage=false;
 deny_write=true;rejects(run,false);deny_write=false;
 kernel_present=false;rejects(run,false);kernel_present=true;
 for(int r:{2,3,32,33}){kernel.fail=r;rejects(run,false);kernel.fail=-1;}
}
'''
    compile_run(tmp_path, 'wall_fp16_shared_leaf', source.replace('LEAF', leaf).replace('OPTIONAL', optional))
