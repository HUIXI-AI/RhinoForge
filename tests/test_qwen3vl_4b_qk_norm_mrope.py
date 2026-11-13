"""Execute the exact QK fusion predicate, manifest and emitter with native consumer."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess

import pytest
from test_qwen3vl_prefill_gateup_launcher import definition, unbound_lingbot_prefill_members

ROOT = Path(__file__).resolve().parents[1]


def test_owner_manifest_and_dispatch(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    source = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    base = (ROOT / "src/core/fused_model_base.h").read_text()
    planner = (ROOT / "src/core/fmb_three_stage_chunk_plan.cpp").read_text()
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    enums = "\n".join(definition(text, f"enum class {name}") + ";" for text, name in (
        (base, "ChunkMode"), (base, "FmbPhysicalManifestState"), (base, "FmbGraphLifecycle"),
        (base, "FmbLinearAccumulationPolicy"), (base, "FmbRouteFamily"),
        (base, "AttentionExecutionPolicy"),
        (source, "CausalDecoderPlanningMode"), (source, "CausalDecoderRopeRoute"),
        (decls, "RpuRmsNormSpmRoute")))
    structs = "\n".join(definition(base, f"struct {name}") + ";" for name in
                         ("ChunkInfo", "FmbRouteManifestEntry", "FmbPhysicalExecutionManifest"))
    constants = "\n".join(re.findall(
        r"inline constexpr int64_t (?:CAUSAL_DECODER_(?:KV_INSERT|WALL_BASE_PAIRED_KV)|QWEN3VL_4B_DECODE_QK_NORM_MROPE(?:_KV_INSERT)?|CAUSAL_DECODER_ORDINARY_RMSNORM|CAUSAL_DECODER_MROPE_POINTER|RHINOVLA_TEXT_HIGH_(?:RMSNORM|ACTIVATION))_SITE\s*=\s*[^;]+;", source))
    constants += "\n" + "\n".join(re.findall(r"inline constexpr int64_t LINGBOT2_\w+_SITE\s*=\s*[^;]+;",source))
    norm_constants = decls[decls.index("enum RpuRmsNormCapability"):decls.index("enum class RpuRmsNormSpmRoute")]
    norm_contract = "\n".join(definition(decls, m) for m in (
        "inline bool rpu_rmsnorm_capability_valid(", "inline RpuRmsNormSpmRoute rpu_resolve_rmsnorm_spm_route(",
        "inline RpuRmsNormSpmRoute rpu_resolve_high_precision_rmsnorm_spm_route("))
    norm_contract += "\n" + definition(decls, "struct RpuRmsNormSpmContract") + ";"
    start = source.index("FmbPhysicalExecutionManifest physical_manifest_for_candidate(")
    append = source.index("        auto append =", start)
    append_text = source[append:source.index("        auto append_attention =", append)]
    selected = source.index("        const bool qwen3vl4b_fused_qk_kv =", start)
    selected_text = source[selected:source.index("        if (wall_rmsnorm_capability_", selected)]
    rope_text = definition(source[source.index("        } else if (!wall_base_nvfp4_qkv_rope && !wall_action_fused_qkv_partial_mrope", start):],
                           "if (!wall_base_nvfp4_qkv_rope && !wall_action_fused_qkv_partial_mrope")
    dispatch_start = source.index("        const bool qwen3vl4b_fused_qk_kv =", source.index("// RoPE 经 emit_qk_rope"))
    dispatch_text = source[dispatch_start:source.index(" else if (has_qk_head_norm)", dispatch_start)]
    kv_start = source.index("        } else if (!qwen3vl4b_fused_qk_kv) {", dispatch_start)
    kv_body = definition(source[kv_start:], "if (!qwen3vl4b_fused_qk_kv)")
    kv_manifest = definition(source[source.index("                if (!qwen3vl_2b_w8_profile_ && !qwen3vl4b_fused_qk_kv)"):],
                             "if (!qwen3vl_2b_w8_profile_ && !qwen3vl4b_fused_qk_kv)")
    consumer = "\n".join(definition(planner, marker).replace("FmbPhysicalManifestConsumer::", "") for marker in (
        "bool FmbPhysicalManifestConsumer::complete() const",
        "const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(",
        "const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route("))
    consumer = consumer.replace("consume_route(", "consume_physical_route(")
    predicates = "\n".join(definition(source, m) for m in (
        "bool qwen3vl_decode_gemv(", "bool qwen3vl_decode_mlp(",
        "bool qwen3vl_4b_decode_qk_norm_mrope(", "bool ordinary_rmsnorm_route_enabled() const"))
    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#define TORCH_CHECK(condition,...) do {if(!(condition))throw std::runtime_error(#condition);}while(false)
#define TORCH_INTERNAL_ASSERT(condition) assert(condition)
namespace at {using IntArrayRef=std::vector<int64_t>;struct Tensor{int id=0;};}
__ENUMS__
__STRUCTS__
__CONSTANTS__
__NORM_CONSTANTS__
__NORM_CONTRACT__
struct Stage {std::vector<ChunkInfo> chunks;};
struct Plan {ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;Stage compute;};
struct Layout {bool is_causal=true,use_attn_mask=false;int64_t batch_size=1;
  AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::AUTO;};
struct Context {
  FmbPhysicalExecutionManifest manifest_;mutable std::vector<uint8_t> consumed_;
 const FmbPhysicalExecutionManifest& manifest()const{return manifest_;}
  Plan stage_plan;int64_t position=0,batch_size=1,seq_len=1;bool is_causal=true;
  std::optional<int> attention_mask;
  AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV;
  std::vector<at::Tensor> kc{36},vc{36};std::vector<at::Tensor>* k_caches=&kc;std::vector<at::Tensor>* v_caches=&vc;
  __CONSUMER__
  void bind(FmbPhysicalExecutionManifest manifest) {
    std::sort(manifest.routes.begin(),manifest.routes.end(),[](const auto& a,const auto& b){
      return std::make_tuple(int64_t(a.family),a.site_id,a.invocation)<std::make_tuple(int64_t(b.family),b.site_id,b.invocation);});
    manifest_=std::move(manifest);consumed_.assign(manifest_.routes.size(),0);
  }
};
int calls=0;uint32_t live_base=0;int64_t live_position=0;int live_layer=0;
void rpu_launch_qwen3vl_4b_decode_qk_norm_mrope_spm_kernel(uint32_t q,uint32_t k,uint32_t qg,uint32_t kg,
  const at::Tensor& cos,const at::Tensor& sin,const at::Tensor& pos,int64_t p,double eps){
  assert(q==live_base+0x1000&&k==live_base+0x2000);
  assert(qg==live_base+0x3000+live_layer*512&&kg==qg+256);
  assert(cos.id==1&&sin.id==2&&pos.id==3&&p==live_position&&eps==1e-6);++calls;
}
int kv_calls=0;
void rpu_launch_qwen3vl_4b_decode_qk_norm_mrope_kv_insert_spm_kernel(uint32_t q,uint32_t k,uint32_t v,uint32_t qg,uint32_t kg,
  const at::Tensor& cos,const at::Tensor& sin,const at::Tensor& pos,const at::Tensor& kc,const at::Tensor& vc,int64_t p,double eps){
  rpu_launch_qwen3vl_4b_decode_qk_norm_mrope_spm_kernel(q,k,qg,kg,cos,sin,pos,p,eps);
  assert(v==live_base+0x2800&&kc.id==100+live_layer&&vc.id==200+live_layer);++kv_calls;
}
struct KvInsertSegmentPlan {};
struct Offset {uint32_t value;};
void rpu_launch_insert_kvcache_spm_unified_with_plan(const at::Tensor&,const at::Tensor&,uint32_t,uint32_t,
  int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,const KvInsertSegmentPlan&){++kv_calls;}
void rpu_launch_wall_base_paired_kv_spm_kernel(const at::Tensor&,const at::Tensor&,uint32_t,uint32_t,int64_t,const KvInsertSegmentPlan&){
  throw std::runtime_error("unexpected Wall paired KV launch");}
struct Model {
 __LINGBOT_UNBOUND__
 bool lingbot_prefill_qk_norm_rope_for_plan(const Plan&,int64_t,int64_t,bool,bool,int64_t,AttentionExecutionPolicy,bool)const{return false;}
 bool emit_lingbot_prefill_qk_norm_rope(int,const ChunkInfo&,int64_t)const{return false;}
 template<class... Args>bool emit_lingbot_prefill_kv_pair(Args...)const{return false;}
 bool qwen3vl_fp16_decode_profile_=false;
 bool rhinovla_high_precision_=false,rhinovla_high_precision_fusions_=false;
     bool qwen3vl_4b_w8_profile_=true,qwen3vl_4b_mlp_w4_profile_=false;
     bool qwen3vl_4b_decode_qk_fused_=true,qwen3vl_4b_decode_qk_kv_fused_=false;
  bool linear_acc32_=false,decoder_mixed_precision_=false,adarms_=false;
  bool qwen3vl_pooler_z1_dispatch_=false,qwen3vl_multiview_text_composite_dispatch_=false;
  bool has_qk_norm_=true,has_mrope_=true,partial_mrope_active_=false,qwen3vl_2b_w8_profile_=false;
  CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
  uint32_t wall_rmsnorm_capability_=RPU_RMSNORM_CAP_BASE;
  RpuRmsNormSpmContract ordinary_rmsnorm_spm_contract_{RPU_RMSNORM_CAP_BASE|RPU_RMSNORM_CAP_V16|RPU_RMSNORM_CAP_V32};
  int64_t seq_len_=1,cache_batch_slot_=0,kv_insert_spm_rows_=0;double eps_=1e-6;int qheads=32,kheads=8,hdim=128,tp=8;
  Context context;at::Tensor cos_{1},sin_{2},position_ids_keepalive_{3};
  Context& ctx(){return context;}int num_cores()const{return tp;}int attn_tp()const{return tp;}
  int intermediate_size()const{return 9728;}int mlp_tp()const{return tp;}
  int64_t hidden_size()const{return 2560;}int num_q_heads()const{return qheads;}
  int num_kv_heads()const{return kheads;}int head_dim()const{return hdim;}
  uint32_t addr(int core,const char* name) const {assert(core==0);return live_base+(std::string(name)=="q"?0x1000:std::string(name)=="k"?0x2000:0x2800);}
  uint32_t norm_q_addr(int layer,int core)const{assert(core==0);return live_base+0x3000+layer*512;}
  uint32_t norm_k_addr(int layer,int core)const{return norm_q_addr(layer,core)+256;}
  std::vector<int64_t> bind_cold_topology_arguments(std::vector<int64_t> a)const{return a;}
  Offset addr_offset(const char* name)const{return {addr(0,name)-live_base};}
  int plans_consumed=0,kv_manifest_rows=0;
  KvInsertSegmentPlan consume_causal_kv_plan(int64_t,int64_t,int64_t,int64_t){++plans_consumed;return {};}
  bool wall_base_paired_kv_enabled(const Plan&,const KvInsertSegmentPlan&,int64_t,int64_t,int64_t,bool,bool,int64_t,bool)const{return false;}
  std::vector<int64_t> wall_base_paired_kv_arguments(int64_t)const{throw std::runtime_error("unexpected Wall paired KV payload");}
  __PREDICATES__
  FmbPhysicalExecutionManifest selected_manifest(const Plan& plan,const Layout& layout,int64_t physical_len,bool partial_mrope){
    FmbPhysicalExecutionManifest manifest;manifest.state=FmbPhysicalManifestState::COMPLETE;
    const bool wall_action_fused_qkv_partial_mrope=false,tensor_rope=false,wall_base_nvfp4_qkv_rope=false;
    const int64_t logical_len=physical_len,position=ctx().position;
    const auto rope_site=CAUSAL_DECODER_MROPE_POINTER_SITE;
    const auto rope_route=CausalDecoderRopeRoute::MROPE;
    __APPEND__
    __SELECTED__
    __ROPE__
    auto append_kv=[&](int64_t,int64_t,int64_t,int64_t,int64_t,bool){++kv_manifest_rows;return KvInsertSegmentPlan{};};
    const auto& chunk=plan.compute.chunks.front();
    __KV_MANIFEST__
    return manifest;
  }
  __EMITTER__
  void dispatch(const ChunkInfo& chunk,int layer_idx,int64_t cos_sin_start){
    const auto bs=ctx().batch_size;
    __DISPATCH__ else {--calls;}
    const auto& k_cache=(*ctx().k_caches)[layer_idx];const auto& v_cache=(*ctx().v_caches)[layer_idx];
    const int64_t seq_len=chunk.len,k_spm_stride=256,nkv=8,hd=128,tp=8,slot0=0,k_cache_batch_stride=0,v_cache_batch_stride=0;
    __KV_BODY__
  }
};
Plan plan_for(int64_t rows){Plan p;p.compute.chunks.push_back({0,0,rows,rows});return p;}
void rejected(const std::function<void()>& fn){calls=0;bool caught=false;try{fn();}catch(const std::runtime_error&){caught=true;}assert(caught&&calls==0);}
int main(){
 Model m;Layout layout;auto plan=plan_for(1);m.context.stage_plan=plan;
 for(int i=0;i<4;++i){
  live_position=std::vector<int64_t>{15,16,1535,15}[i];live_layer=i;live_base=0x40000000+i*0x10000;
  auto manifest=m.selected_manifest(plan,layout,1,false);assert(manifest.routes.size()==2);
  m.context.bind(manifest);m.context.position=live_position;
  // Input RMS remains independently described and BASE; head norms+RoPE replaced by one site.
  auto ordinary=m.context.find_route(FmbRouteFamily::NORMALIZATION,CAUSAL_DECODER_ORDINARY_RMSNORM_SITE,0);
  assert(ordinary.selector==int64_t(RpuRmsNormSpmRoute::BASE));
  m.context.consume_physical_route(ordinary.family,ordinary.site_id,ordinary.selector,ordinary.flags,ordinary.arguments,0);
  calls=0;m.dispatch(plan.compute.chunks[0],live_layer,live_position);assert(calls==1);
  assert(std::all_of(m.context.consumed_.begin(),m.context.consumed_.end(),[](auto n){return n==1;}));
 }
 const auto canonical=m.selected_manifest(plan,layout,1,false);
 for(int fault=0;fault<8;++fault){
  auto bad=canonical;auto it=std::find_if(bad.routes.begin(),bad.routes.end(),[](auto& r){return r.site_id==QWEN3VL_4B_DECODE_QK_NORM_MROPE_SITE;});
  if(fault==0)bad.routes.erase(it);if(fault==1)it->selector=2;if(fault==2)it->flags=1;
  if(fault==3)it->invocation=1;if(fault==4)it->arguments[1]=8;if(fault==5)it->family=FmbRouteFamily::ROPE;
  if(fault==6)it->arguments.push_back(0);if(fault==7)bad.state=FmbPhysicalManifestState::UNSPECIFIED;
  m.context.bind(bad);rejected([&]{m.dispatch(plan.compute.chunks[0],live_layer,live_position);});
 }
 auto excluded=[&](Model bad,Layout l,int64_t physical,int64_t rows,bool partial,ChunkMode mode){
  auto p=plan_for(rows);p.chunk_mode=mode;bad.context.stage_plan=p;bad.seq_len_=physical;
  bad.context.is_causal=l.is_causal;bad.context.batch_size=l.batch_size;bad.partial_mrope_active_=partial;
  if(l.use_attn_mask)bad.context.attention_mask=1;
  auto manifest=bad.selected_manifest(p,l,physical,partial);
  assert(std::none_of(manifest.routes.begin(),manifest.routes.end(),[](auto& r){return r.site_id==QWEN3VL_4B_DECODE_QK_NORM_MROPE_SITE||r.site_id==QWEN3VL_4B_DECODE_QK_NORM_MROPE_KV_INSERT_SITE;}));
  assert(std::any_of(manifest.routes.begin(),manifest.routes.end(),[](auto& r){return r.family==FmbRouteFamily::ROPE;}));
  bad.context.bind(manifest);calls=0;bad.dispatch(p.compute.chunks[0],0,0);assert(calls==-1);
  rejected([&]{bad.emit_qwen3vl_4b_decode_qk_norm_mrope(0,p.compute.chunks[0],0);});
  rejected([&]{bad.emit_qwen3vl_4b_decode_qk_norm_mrope_kv_insert(0,p.compute.chunks[0],0);});
 };
 for(auto field:{&Model::qwen3vl_4b_decode_qk_fused_,&Model::has_qk_norm_,&Model::has_mrope_}){
  Model b;b.*field=false;excluded(b,layout,1,1,false,ChunkMode::SEQUENTIAL);}
 for(auto field:{&Model::decoder_mixed_precision_,&Model::adarms_,&Model::qwen3vl_pooler_z1_dispatch_,&Model::qwen3vl_multiview_text_composite_dispatch_}){
  Model b;b.*field=true;excluded(b,layout,1,1,false,ChunkMode::SEQUENTIAL);}
 for(auto shape:std::vector<std::pair<int64_t,int64_t>>{{16,16},{256,256},{1,2},{2,1}})excluded(Model{},layout,shape.first,shape.second,false,ChunkMode::SEQUENTIAL);
 for(auto field:{&Model::qheads,&Model::kheads,&Model::hdim,&Model::tp}){Model b;b.*field=(field==&Model::tp?4:b.*field*2);excluded(b,layout,1,1,false,ChunkMode::SEQUENTIAL);}
 {Model b;b.eps_=1e-5;excluded(b,layout,1,1,false,ChunkMode::SEQUENTIAL);}
 {Model b;b.planning_mode_=CausalDecoderPlanningMode::WALL_Z1;excluded(b,layout,1,1,false,ChunkMode::SEQUENTIAL);}
 {auto l=layout;l.use_attn_mask=true;excluded(Model{},l,1,1,false,ChunkMode::SEQUENTIAL);}
 {auto l=layout;l.is_causal=false;excluded(Model{},l,1,1,false,ChunkMode::SEQUENTIAL);}
 {auto l=layout;l.batch_size=2;excluded(Model{},l,1,1,false,ChunkMode::SEQUENTIAL);}
 excluded(Model{},layout,1,1,true,ChunkMode::SEQUENTIAL);excluded(Model{},layout,1,1,false,ChunkMode::KV_FIRST);
 // FP16 ACC32 and W4 share the exact FP16 activation postprocessing.
 // Neither requires the W8 Linear profile, nor changes its accumulation mode.
 for(bool acc32:{false,true}){
  Model other;other.qwen3vl_4b_w8_profile_=false;other.linear_acc32_=acc32;
  other.context.stage_plan=plan;
  auto manifest=other.selected_manifest(plan,layout,1,false);
  other.context.bind(manifest);calls=0;live_layer=0;live_position=0;live_base=0x40000000;
  other.dispatch(plan.compute.chunks[0],0,0);assert(calls==1&&other.linear_acc32_==acc32);
 }
 // Old REF preserves both head BASE contracts and the original RoPE entry.
 Model old;old.qwen3vl_4b_decode_qk_fused_=false;
 const auto old_manifest=old.selected_manifest(plan,layout,1,false);assert(old_manifest.routes.size()==4);
 for(int role=0;role<3;++role){old.context.bind(old_manifest);assert(old.context.find_route(FmbRouteFamily::NORMALIZATION,CAUSAL_DECODER_ORDINARY_RMSNORM_SITE,role).selector==1);}
 // Installing the new capability cannot consume an old descriptor.
 m.context.bind(old_manifest);rejected([&]{m.dispatch(plan.compute.chunks[0],0,0);});
 // New payload owns both cache writes and removes the standalone KV route.
 for(bool old_payload:{false,true}){
  Model fused;fused.qwen3vl_4b_decode_qk_fused_=old_payload;fused.qwen3vl_4b_decode_qk_kv_fused_=true;
  fused.context.stage_plan=plan;
  for(int layer=0;layer<4;++layer){
   live_layer=layer;live_position=std::vector<int64_t>{15,16,1535,15}[layer];live_base=0x50000000+layer*0x10000;
   fused.context.kc[layer].id=100+layer;fused.context.vc[layer].id=200+layer;
   auto manifest=fused.selected_manifest(plan,layout,1,false);assert(fused.kv_manifest_rows==0);
   assert(std::none_of(manifest.routes.begin(),manifest.routes.end(),[](auto& r){return r.site_id==QWEN3VL_4B_DECODE_QK_NORM_MROPE_SITE;}));
   fused.context.bind(manifest);calls=kv_calls=0;
   fused.dispatch(plan.compute.chunks[0],layer,live_position);assert(calls==1&&kv_calls==1&&fused.plans_consumed==0);
  }
  const auto canonical=fused.selected_manifest(plan,layout,1,false);
  for(int fault=0;fault<8;++fault){
   auto bad=canonical;auto it=std::find_if(bad.routes.begin(),bad.routes.end(),[](auto& r){return r.site_id==QWEN3VL_4B_DECODE_QK_NORM_MROPE_KV_INSERT_SITE;});
   if(fault==0)bad.routes.erase(it);if(fault==1)it->selector=2;if(fault==2)it->flags=1;
   if(fault==3)it->invocation=1;if(fault==4)it->arguments[1]=8;if(fault==5)it->family=FmbRouteFamily::ROPE;
   if(fault==6)it->arguments.push_back(0);if(fault==7)bad.state=FmbPhysicalManifestState::UNSPECIFIED;
   fused.context.bind(bad);kv_calls=0;rejected([&]{fused.dispatch(plan.compute.chunks[0],live_layer,live_position);});assert(kv_calls==0);
  }
  for(auto shape:std::vector<std::pair<int64_t,int64_t>>{{16,16},{256,256},{1,2},{2,1}})excluded(fused,layout,shape.first,shape.second,false,ChunkMode::SEQUENTIAL);
  {auto l=layout;l.use_attn_mask=true;excluded(fused,l,1,1,false,ChunkMode::SEQUENTIAL);}
  {auto l=layout;l.is_causal=false;excluded(fused,l,1,1,false,ChunkMode::SEQUENTIAL);}
  {auto l=layout;l.batch_size=2;excluded(fused,l,1,1,false,ChunkMode::SEQUENTIAL);}
  excluded(fused,layout,1,1,true,ChunkMode::SEQUENTIAL);excluded(fused,layout,1,1,false,ChunkMode::KV_FIRST);
  fused.context.bind(canonical);fused.context.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
  rejected([&]{fused.emit_qwen3vl_4b_decode_qk_norm_mrope_kv_insert(live_layer,plan.compute.chunks[0],live_position);});
  fused.context.attention_policy=AttentionExecutionPolicy::DDR_KV;fused.cache_batch_slot_=1;
  rejected([&]{fused.emit_qwen3vl_4b_decode_qk_norm_mrope_kv_insert(live_layer,plan.compute.chunks[0],live_position);});
 }
 // Missing new payload keeps the original QK and standalone KV path.
 {Model fallback;fallback.context.stage_plan=plan;auto mf=fallback.selected_manifest(plan,layout,1,false);
  assert(fallback.kv_manifest_rows==1);fallback.context.bind(mf);calls=kv_calls=0;
  fallback.dispatch(plan.compute.chunks[0],live_layer,live_position);assert(calls==1&&kv_calls==1&&fallback.plans_consumed==1);}
 // BASE-only references also work: the fused site never requires vector payload availability.
 m.ordinary_rmsnorm_spm_contract_.capability=RPU_RMSNORM_CAP_BASE;
 auto base_only=m.selected_manifest(plan,layout,1,false);assert(base_only.routes.size()==1);
 m.context.bind(base_only);calls=0;m.dispatch(plan.compute.chunks[0],live_layer,live_position);assert(calls==1);
}
'''
    for key, value in {"ENUMS": enums, "STRUCTS": structs, "CONSTANTS": constants,
                       "NORM_CONSTANTS": norm_constants, "NORM_CONTRACT": norm_contract,
                       "CONSUMER": consumer, "PREDICATES": predicates, "APPEND": append_text,
                       "SELECTED": selected_text, "ROPE": rope_text,
                       "EMITTER": definition(source, "void emit_qwen3vl_4b_decode_qk_norm_mrope(") + "\n" + definition(source, "void emit_qwen3vl_4b_decode_qk_norm_mrope_kv_insert("),
                       "KV_MANIFEST": kv_manifest, "KV_BODY": kv_body,
                       "DISPATCH": dispatch_text}.items():
        program = program.replace(f"__{key}__", value)
    program = program.replace("__LINGBOT_UNBOUND__", unbound_lingbot_prefill_members(source))
    cpp, exe = tmp_path / "qk_owner.cpp", tmp_path / "qk_owner"
    cpp.write_text(program)
    result = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-Wno-misleading-indentation",
                            str(cpp), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)




def test_cold_capability_layout_and_logical_position_backing(tmp_path):
    """Execute the cold profile, payload binding, narrow allocation and layout hash."""
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    source = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    base = (ROOT / "src/core/fused_model_base.h").read_text()
    profile_start = source.index("        qwen3vl_4b_w8_profile_ = N == 36")
    # Keep actual Qwen admission and payload bindings while excluding the
    # interleaved Wall profile, which is tested under its own model geometry.
    profile = source[profile_start:source.index("        wall_base_prefill_shared_input_ =", profile_start)]
    profile += "\n" + "\n".join(re.search(
        rf"        {field} = [^;]+;", source[profile_start:]).group(0)
        for field in ("qwen3vl_4b_decode_gateup_fused_", "qwen3vl_4b_decode_qk_fused_",
                      "qwen3vl_4b_decode_qk_kv_fused_", "qwen3vl_4b_decode_o_ring_norm_fused_",
                      "qwen3vl_4b_decode_down_ring_norm_fused_"))
    wall_fields = "\n".join(re.findall(r"    bool wall_base_\w+_ = false;", source))
    capacity_start = source.index("            const int64_t mrope_capacity")
    capacity_binding = source[capacity_start:source.index("            ka_cpu_pos_published_rows_.assign", capacity_start)]
    padding_start = source.index("            if (!position_ids_keepalive_.defined()", capacity_start)
    padding = source[padding_start:source.index("            // partial_mrope keepalives", padding_start)]
    constants = definition(base, "enum class FmbLinearRouteSelector") + ";\n"
    constants += definition(source, "enum class CausalDecoderLinearRoute") + ";\n"
    constants += "\n".join(re.findall(r"inline constexpr int64_t \w+\s*=\s*[^;]+;", source))
    constants += "\n" + re.search(r"inline constexpr int64_t FMB_SHARED_MLP_SILU_MUL_SITE\s*=\s*[^;]+;", base).group(0)
    layout = definition(source, "int64_t subclass_layout_hash() const override").replace(" override", "")
    # Keep the checked-in set_weights invalidation requirement explicit; actual
    # invalidation and Graph lifecycle are separately covered by FMB tests.
    setter_start = source.index("void set_weights(")
    setter = definition(source[source.index("\n    {", setter_start):], "\n    {")
    assert setter.rsplit("invalidate_model_state();", 1)[1].strip().startswith("// D-503: last non-empty statement")
    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <limits>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (false)
namespace at {
constexpr int kInt=1,kChar=2,kHalf=3,kPrivateUse1=4,kByte=5;
struct TensorOptions {TensorOptions& dtype(int){return *this;}TensorOptions& device(int){return *this;}};
struct Tensor {
 bool present=false;int dtype=kChar;int64_t rows=0,cols=3;
 std::shared_ptr<std::vector<int32_t>> owner;
 bool defined()const{return present;}int scalar_type()const{return dtype;}
 int64_t size(int d)const{return d==0?rows:cols;}
 struct Device {int type()const{return kPrivateUse1;}};Device device()const{return {};}
 Tensor narrow(int axis,int64_t start,int64_t count)const {assert(axis==0&&start==0&&count<=rows);Tensor t=*this;t.rows=count;return t;}
 void zero_(){std::fill(owner->begin(),owner->begin()+rows*cols,0);}
};
Tensor zeros(std::vector<int64_t> shape,TensorOptions){Tensor t;t.present=true;t.rows=shape[0];t.cols=shape[1];t.dtype=kInt;t.owner=std::make_shared<std::vector<int32_t>>(t.rows*t.cols,0);return t;}
}
enum class KernelId {QWEN3VL_GATEUP_SWIGLU_GEMV,QWEN3VL_QK_NORM_MROPE_D128,QWEN3VL_QK_NORM_MROPE_KV_INSERT_D128,QWEN3VL_O_RING_NORM_M1_H2560_W8};
struct KernelCache {bool qk=false,qkkv=false,o=false;int queries=0;static KernelCache& instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId id){++queries;return (id==KernelId::QWEN3VL_QK_NORM_MROPE_D128&&qk)||(id==KernelId::QWEN3VL_QK_NORM_MROPE_KV_INSERT_D128&&qkkv)||(id==KernelId::QWEN3VL_O_RING_NORM_M1_H2560_W8&&o);}};
namespace detail {__MIX__}
__CONSTANTS__
constexpr int NUM_CORES=8;
struct Model {
 int64_t N=36,hidden_size=2560,intermediate_size=9728,num_q_heads=32,num_kv_heads=8,head_dim=128;
 __WALL_FIELDS__
 __LINGBOT_UNBOUND__
 bool rhinovla_high_precision_=false,rhinovla_high_precision_fusions_=false;
 bool tts_codec_mode_=false;std::vector<at::Tensor> tts_predictor_heads_,lightnav_heads_;
 bool has_scale=true,has_nvfp4=false,use_silu=true,has_qk_norm_=true,has_qkv_bias_=false;
 std::vector<int64_t> mrope_section{24,20,20},deepstack_lang_layers{0,1,2};
 std::vector<at::Tensor> q_w_list{36},k_w_list{36},v_w_list{36},o_w_list{36},gate_list{36},up_list{36},down_list{36};
 bool qwen3vl_4b_w8_profile_=false,qwen3vl_4b_mlp_w4_profile_=false;
 bool qwen3vl_4b_decode_gateup_fused_=false,qwen3vl_4b_decode_qk_fused_=false,qwen3vl_4b_decode_qk_kv_fused_=false,qwen3vl_4b_decode_o_ring_norm_fused_=false,qwen3vl_4b_decode_down_ring_norm_fused_=false;
 bool explicit_mask_residency_requested_=false;
 bool wall_prefill_staged_mlp_=false,decode_gate_up_fusion_=false,prefill_packed_gate_up_strided_=false,qwen3vl_4b_packed_qkv_weights_=false;
 std::vector<int64_t> deepstack_lang_layers_{0,1,2};int configured_vocab_size_=0,cores=8;
 int num_cores()const{return cores;}int attn_tp()const{return cores;}int mlp_tp()const{return cores;}int lm_head_tp()const{return cores;}
 struct Table{int64_t rows=8192;int64_t size(int)const{return rows;}}cos;
 struct Envelope{int64_t max_kv_len=0;bool declared()const{return max_kv_len>0;}}envelope;
 const Envelope& chunk_envelope()const{return envelope;}
 at::Tensor position_ids_keepalive_;
 void cold(){__PROFILE__ __CAPACITY__ __PADDING__}
 __LAYOUT__
};
int main(){
 for(int64_t capacity:{1,8193,16384,32768}){Model m;m.cos.rows=capacity*2;m.envelope.max_kv_len=capacity;m.cold();
  assert(m.position_ids_keepalive_.rows==capacity&&m.position_ids_keepalive_.owner->size()==static_cast<size_t>((capacity+8)*3));
  assert(static_cast<size_t>((capacity-1)*12+32)<=m.position_ids_keepalive_.owner->size()*4);
  auto owner=m.position_ids_keepalive_.owner;m.cold();assert(m.position_ids_keepalive_.owner==owner);
 }
 for(bool overflow:{false,true}){Model m;m.cos.rows=overflow?std::numeric_limits<int64_t>::max():8192;
  m.envelope.max_kv_len=overflow?m.cos.rows:8193;bool rejected=false;
  try{m.cold();}catch(const std::runtime_error&){rejected=true;}
  assert(rejected&&!m.position_ids_keepalive_.defined());
 }
 auto& cap=KernelCache::instance();Model old;old.cold();assert(old.qwen3vl_4b_w8_profile_&&!old.qwen3vl_4b_decode_qk_fused_);
 const auto oldhash=old.subclass_layout_hash();assert(old.position_ids_keepalive_.rows==8192&&old.position_ids_keepalive_.owner->size()==8200*3);
 auto oldowner=old.position_ids_keepalive_.owner;old.position_ids_keepalive_.owner->at(1)=99;old.cold();assert(oldowner==old.position_ids_keepalive_.owner&&oldowner->at(1)==0);
 cap.qk=true;Model fused;fused.cold();assert(fused.qwen3vl_4b_decode_qk_fused_&&!fused.qwen3vl_4b_decode_gateup_fused_);
 assert(fused.subclass_layout_hash()!=oldhash);const auto fusedhash=fused.subclass_layout_hash();
 assert(fused.position_ids_keepalive_.rows==8192&&fused.position_ids_keepalive_.cols==3);
 assert(fused.position_ids_keepalive_.owner->size()==8200*3);assert(std::all_of(fused.position_ids_keepalive_.owner->begin(),fused.position_ids_keepalive_.owner->end(),[](auto x){return x==0;}));
 // Final logical row's 32B physical read fits; its extra backing does not add logical positions.
 assert(8191*12+32<=fused.position_ids_keepalive_.owner->size()*4);assert(fused.position_ids_keepalive_.size(0)==8192);
 const int cold_queries=cap.queries;cap.qk=false;
 assert(fused.qwen3vl_4b_decode_qk_fused_&&fused.subclass_layout_hash()==fusedhash&&cap.queries==cold_queries);
 fused.cold();assert(!fused.qwen3vl_4b_decode_qk_fused_&&fused.subclass_layout_hash()==oldhash);
 cap.qkkv=true;Model new_only;new_only.cold();assert(!new_only.qwen3vl_4b_decode_qk_fused_&&new_only.qwen3vl_4b_decode_qk_kv_fused_);
 assert(new_only.position_ids_keepalive_.rows==8192&&new_only.position_ids_keepalive_.owner->size()==8200*3);
 assert(new_only.subclass_layout_hash()!=oldhash&&new_only.subclass_layout_hash()!=fusedhash);
 auto newhash=new_only.subclass_layout_hash();const auto newqueries=cap.queries;cap.qkkv=false;
 assert(new_only.qwen3vl_4b_decode_qk_kv_fused_&&new_only.subclass_layout_hash()==newhash&&cap.queries==newqueries);
 new_only.cold();assert(!new_only.qwen3vl_4b_decode_qk_kv_fused_&&new_only.subclass_layout_hash()==oldhash);
 cap.o=true;Model fused_o;fused_o.cold();assert(fused_o.qwen3vl_4b_decode_o_ring_norm_fused_&&fused_o.qwen3vl_4b_decode_down_ring_norm_fused_);
 const auto ohash=fused_o.subclass_layout_hash();assert(ohash!=oldhash&&ohash!=fusedhash&&ohash!=newhash);
 const auto oqueries=cap.queries;cap.o=false;assert(fused_o.subclass_layout_hash()==ohash&&cap.queries==oqueries);
 fused_o.cold();assert(!fused_o.qwen3vl_4b_decode_o_ring_norm_fused_&&fused_o.subclass_layout_hash()==oldhash);
 cap.qk=true;cap.qkkv=true;cap.o=true;
 for(int dtype:{at::kHalf,at::kByte}){
  Model b;b.has_scale=dtype!=at::kHalf;
  for(auto field:{&Model::q_w_list,&Model::k_w_list,&Model::v_w_list,&Model::o_w_list,&Model::gate_list,&Model::up_list,&Model::down_list})
   for(auto& weight:b.*field)weight.dtype=dtype;
  b.cold();assert(!b.qwen3vl_4b_w8_profile_&&b.qwen3vl_4b_decode_qk_fused_&&b.qwen3vl_4b_decode_qk_kv_fused_);
  assert(!b.qwen3vl_4b_decode_gateup_fused_&&!b.qwen3vl_4b_decode_o_ring_norm_fused_);
  assert(b.position_ids_keepalive_.rows==8192&&b.position_ids_keepalive_.owner->size()==8200*3);
 }
 for(int fault=0;fault<12;++fault){Model b;
   if(fault==0)b.N=35;if(fault==1)b.cores=4;if(fault==2)b.has_scale=false;if(fault==3)b.has_nvfp4=true;
   if(fault==4)b.hidden_size=2048;if(fault==5)b.intermediate_size=6144;if(fault==6)b.num_q_heads=16;
   if(fault==7)b.mrope_section={16,24,24};if(fault==8)b.deepstack_lang_layers={1,2,3};
   if(fault==9)b.use_silu=false;if(fault==10)b.has_qk_norm_=false;if(fault==11)b.has_qkv_bias_=true;
   b.cold();assert(!b.qwen3vl_4b_w8_profile_&&!b.qwen3vl_4b_decode_qk_fused_&&!b.qwen3vl_4b_decode_qk_kv_fused_&&!b.qwen3vl_4b_decode_o_ring_norm_fused_&&b.position_ids_keepalive_.owner->size()==8200*3);
 }
 for(auto field:{&Model::q_w_list,&Model::k_w_list,&Model::v_w_list,&Model::o_w_list,&Model::gate_list,&Model::up_list,&Model::down_list}){
  Model b;(b.*field)[17].dtype=at::kHalf;b.cold();assert(!b.qwen3vl_4b_decode_qk_fused_);
 }
}
'''
    for key, value in {"MIX": definition(base, "inline int64_t layout_mix("), "CONSTANTS": constants,
                       "PROFILE": profile, "CAPACITY": capacity_binding, "PADDING": padding, "LAYOUT": layout, "WALL_FIELDS": wall_fields}.items():
        program = program.replace(f"__{key}__", value)
    program = program.replace("__LINGBOT_UNBOUND__", unbound_lingbot_prefill_members(source))
    cpp, exe = tmp_path / "qk_cold.cpp", tmp_path / "qk_cold"
    cpp.write_text(program)
    result = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-Wno-misleading-indentation",
                             str(cpp), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)
