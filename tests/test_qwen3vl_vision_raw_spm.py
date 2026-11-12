"""Execute Vision's native capability/manifest/BufferDecl code on a CPU arena.

No device kernels are mocked as numeric evidence. Small tensor/profile doubles
supply cold metadata; actual native declarations, routes, SDPA constraints and
SPM first-fit allocator determine the tested behavior.
"""
from pathlib import Path
import json
import re
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]


def _program():
    model = (ROOT / 'src/fused/rpu_qwen3vl_vision_model.cpp').read_text()
    header = (ROOT / 'src/core/fused_model_base.h').read_text()
    helpers = (ROOT / 'src/core/rpu_helpers.h').read_text()
    n1200 = (ROOT / 'src/ops/rpu_qwen3vl_vision_sdpa.h').read_text()
    allocator = (ROOT / 'src/core/rpu_spm_allocator.cpp').read_text()
    stage_source = (ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    base_source = (ROOT / 'src/core/fused_model_base.cpp').read_text()
    out = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#define TORCH_CHECK(c,...) do {if(!(c))throw std::runtime_error("native contract rejected");}while(0)
#define TORCH_INTERNAL_ASSERT(c,...) TORCH_CHECK(c)
#define DWIDTH 2
template<class T> T CeilDiv(T a,T b){return (a+b-1)/b;}
int64_t Align(int64_t a,int64_t b){return CeilDiv(a,b)*b;}
class FusedModelBase {
public:
 struct ChunkEnvelope { int64_t max_kv_len=4096, chunk=0; } chunk_envelope_;
 const ChunkEnvelope& chunk_envelope() const { return chunk_envelope_; }
};
namespace at {constexpr int kChar=1,kHalf=2,kPrivateUse1=19;}
namespace c10 {using Half=uint16_t;}
struct Tensor {
 int dtype=at::kChar,id=0;int64_t elements=2000000;
 bool present=true,contiguous=true;int device_kind=at::kPrivateUse1;
 std::vector<int64_t> shape{4096,4096};
 int scalar_type()const{return dtype;}bool defined()const{return present;}
 bool is_contiguous()const{return contiguous;}int64_t dim()const{return shape.size();}
 int64_t size(int i)const{return shape.at(i);}
 int64_t numel()const{int64_t n=1;for(auto d:shape)n*=d;return n;}
 struct Device{int kind;int type()const{return kind;}};
 Device device()const{return {device_kind};}
 template<class T>T* data_ptr()const{return reinterpret_cast<T*>(uintptr_t(id+1)*0x10000000ULL);}
};
namespace at {using Tensor=::Tensor;}
'''
    for prefix in (
        'enum class ChunkMode', 'enum class InterLayerIO', 'enum class AttentionExecutionPolicy',
        'enum class FmbRopeTableResidency', 'enum class StorageClass',
        'enum class BufferScope', 'struct BufferDecl', 'enum class FmbForwardOperandResidency', 'struct LayoutContext',
        'struct ChunkInfo', 'struct ChunkPlan', 'struct ResolvedFmbStageChunks',
        'enum class FmbSpanBoundaryPolicy', 'struct FmbStageBoundaryPolicies',
        'struct FmbExecutionSpan', 'struct FmbThreeStageChunkPlan',
        'enum class FmbPhysicalManifestState', 'enum class FmbGraphLifecycle',
        'enum class FmbLinearAccumulationPolicy', 'enum class FmbRouteFamily',
        'enum class FmbLinearRouteSelector', 'struct FmbRouteManifestEntry',
        'enum class FmbSharedAllReduceRouteSelector',
        'struct FmbPhysicalExecutionManifest', 'struct FmbPrefillStageCandidate',
    ):
        out += _definition(header, prefix) + ';\n'
    # Full pure tiling prefix includes the production raw capability and VLM
    # predicate, not a test's copy of their shape rules.
    end = _definition(helpers, 'inline bool sdpa_by_mha_spm_is_valid(')
    pure_helpers = helpers[helpers.index('inline int64_t sdpa_tile_m_v16_initial('):helpers.index(end)+len(end)]
    for device_helper in ('inline void rpu_set_scm_u32_checked(',
                          'inline void rpu_set_legacy_scm_u16_checked('):
        pure_helpers = pure_helpers.replace(_definition(helpers, device_helper), '')
    out += pure_helpers
    out += n1200.replace('#pragma once', '').replace('#include "rpu_helpers.h"', '')
    out += '\n'.join(re.findall(r'(?:inline )?constexpr (?:int64_t|uint32_t) (?:QWEN3VL_|RHINOVLA_)[\s\S]*?;', model[:model.index('// First Qwen3-VL pooler-only')]))
    for prefix in ('enum class Qwen3VLVisionRopeRoute ', 'enum class Qwen3VLVisionRopeRouteRequest', 'enum class Qwen3VLVisionMutableDmaRoute'):
        out += _definition(model, prefix) + ';\n'
    # DDR KV route payload resolution is unrelated to the new raw capability;
    # keep only its shape and selector in the double. RAW must never call it.
    out = out.replace('KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 |\n    KV_INSERT_CAP_HYBRID2 | KV_INSERT_CAP_HYBRID3', '15')
    out += r'''
constexpr int64_t FMB_ROUTE_FLAG_ROPE_TABLE_DDR=1LL<<56,FMB_ROUTE_FLAG_ROPE_TABLE_SPM=1LL<<57;
constexpr int FMB_SHARED_LAYER_INPUT_DMA=1;
struct KvInsertSegmentPlan {int64_t n;int route()const{return 1;}int64_t physical_rows()const{return n;}};
using KvInsertRouteArguments=std::array<int64_t,2>;
int kv_resolutions=0;
KvInsertSegmentPlan resolve_kvinsert_plan_auto(int64_t,FmbGraphLifecycle,int64_t,int64_t n,int64_t,int,int64_t,int64_t,uint32_t){++kv_resolutions;return {n};}
KvInsertRouteArguments rpu_kvinsert_route_arguments(const KvInsertSegmentPlan& p,int,int64_t,int64_t){return {0,p.n};}
void append_fmb_shared_runtime_routes(FmbPhysicalExecutionManifest&,const FmbThreeStageChunkPlan&,int64_t,int,int,int,int){}
class SpmAllocator {
public:
 struct AllocRequest {int64_t bytes;int first_step,last_step,scope;};
 struct AliasedPlan {std::vector<uint32_t> offsets;size_t peak_bytes=0;};
 static size_t align_up(size_t n){return (n+255)&~size_t{255};}
 static AliasedPlan plan_temporary_aliased(const std::vector<AllocRequest>&);
};
'''
    out += _definition(allocator, 'SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(')
    # Use the actual dry/live chunk builder and composition validator. In
    # particular, kv_seq_len is the chunk end, even for bidirectional Vision;
    # attention's full key length comes from its separate layout/descriptor.
    out += _definition(base_source, 'static std::vector<ChunkInfo> make_fmb_chunks(')
    out += '\nnamespace detail {\n'
    for prefix in ('int64_t validate_resolved_chunk_coverage(',
                   'void validate_fmb_execution_spans(',
                   'bool fmb_chunks_respect_boundary_policy('):
        out += _definition(stage_source, prefix) + '\n'
    out += '}\n' + _definition(stage_source, 'struct ValidatedStage ') + ';\n'
    for prefix in ('ResolvedFmbStageChunks make_stage(',
                   'ValidatedStage validate_stage(', 'bool same_schedule(',
                   'void validate_fmb_three_stage_chunk_plan(',
                   'FmbThreeStageChunkPlan compose_fmb_three_stage_chunk_plan(\n'
                   '    std::vector<ChunkInfo> input_chunks,\n'
                   '    std::vector<ChunkInfo> qkv_chunks,\n'
                   '    std::vector<ChunkInfo> compute_chunks,\n'
                   '    std::vector<FmbExecutionSpan> spans,\n'
                   '    ChunkMode chunk_mode,'):
        out += _definition(stage_source, prefix) + '\n'
    for prefix in ('uint64_t mix_u64(', 'uint64_t physical_manifest_fingerprint_impl(',
                   'uint64_t fmb_physical_manifest_fingerprint(',
                   'FmbRopeTableResidency fmb_rope_table_residency(',
                   'AttentionExecutionPolicy fmb_attention_execution_policy(',
                   'void validate_physical_manifest_impl(',
                   'std::tuple<int64_t, int64_t> physical_capability_rank(',
                   'bool fmb_prefill_stage_candidate_less('):
        out += _definition(stage_source, prefix) + '\n'
    out += _definition(stage_source, 'int64_t fmb_ring_all_reduce_route_selector(') + '\n'
    out += r'''
struct ModelDynamicConfig {
 ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;
 InterLayerIO inter_layer_io=InterLayerIO::AUTO;
 AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV;
};
void check_fixed_attention(const ModelDynamicConfig& dyn_cfg,
                          std::optional<AttentionExecutionPolicy> manifest_attention_layout_policy){
'''
    out += _definition(base_source, 'if (manifest_attention_layout_policy.has_value() &&')
    out += '}\n'
    out += r'''
class Qwen3VLVisionModel:public FusedModelBase {
public:
 int64_t current_num_patches_=256,image_batch_count_=1,merger_out_hidden_=2560,orig_head_dim_=64;
 int64_t cores=8,layers=24,hidden=1024,intermediate=4096,heads=16,hd=64;
 bool linear_acc32_=false,pipeline=false,delivery=false,fused_merger_enabled_=true;
 bool rhinovla_high_precision_=false;
 bool tap_merger_enabled_=false;
 int64_t lingbot_residual_resident_mode_=0,lingbot_spm_kv_mode_=0;
 int64_t lingbot_qkv_joint_mode_=0,lingbot_fc1_gelu_mode_=0;
 bool has_rope_=true,rope_spm_enabled_=true,z1_dispatch_=false,multiview_composite_dispatch_=false;
 bool multiview_dry_prepared_=false,multiview_composite_prepared_=false,prefix_scatter_active_=false;
 bool n1200_tm160_enabled_=false,delivery_compatibility_=false;
 bool fp16_chunked_merger_enabled_=false,large_image_auto_chunk_=false;
 bool fp16_compact_encoder_enabled_=false;
 bool w8_compact_encoder_enabled_=false;
 int64_t configured_chunk_size_=0,last_resolved=0,invalidations=0;
 bool native_composite_prepare_active_=false,z1_adopted_=false,z1_bound_=false;
 int64_t z1_prepared_num_patches_=0;
 int64_t max_hw_=48,merge_hidden_=4096,z1_retained_deepstack_count_=0;
 Qwen3VLVisionRopeRouteRequest rope_route_request_=Qwen3VLVisionRopeRouteRequest::AUTO;
 struct Weights {Tensor q_w,k_w,v_w,o_w,fc1_w,fc2_w;};
 std::vector<Weights> layer_weights_=std::vector<Weights>(24);
 std::vector<int64_t> deepstack_visual_indexes_{5,11,17};
 Tensor merger_m0_w_,merger_m2_w_;
 std::vector<Tensor> ds_merger_m0_w_{3},ds_merger_m2_w_{3};
 Tensor merger_m0_scale_,merger_m2_scale_;
 std::vector<Tensor> ds_merger_m0_scale_{3},ds_merger_m2_scale_{3};
 Tensor position_idx_keepalive_{at::kHalf,0,8192,false};
 int64_t num_cores()const{return cores;}int64_t num_layers()const{return layers;}
 int64_t hidden_size()const{return hidden;}int64_t intermediate_size()const{return intermediate;}
 int64_t num_q_heads()const{return heads;}int64_t head_dim()const{return hd;}
 int attn_tp()const{return cores;}int mlp_tp()const{return cores;}
 bool pipeline_dispatch_active()const{return pipeline;}
 bool qwen3vl_2b_delivery_profile_active()const{return delivery;}
 int64_t get_last_resolved_chunk_size()const{return last_resolved;}
 void invalidate_model_state(){++invalidations;}
 bool rope_tables_use_persistent_storage()const{return pipeline;}
 void emit_preload_weights(){}
 // Ordinary raw owner leaves resident weight priming unbound.
 bool resident_preloads_enabled()const{return false;}
 void preload_resident_merger(int){throw std::runtime_error("unexpected resident preload");}
'''
    for prefix in (
        'bool ordinary_fp16_chunked_merger_profile(',
        'bool fp16_chunked_merger_active(', 'bool fp16_chunked_merger_plan(',
        'void set_fp16_compact_encoder(', 'bool fp16_compact_encoder_active(',
        'bool fp16_compact_encoder_chunk(', 'bool fp16_compact_encoder_plan(',
        'void set_w8_compact_encoder(', 'bool ordinary_w8_compact_encoder_profile(',
        'bool w8_compact_encoder_active(', 'bool compact_encoder_active(',
        'bool compact_encoder_chunk(', 'bool compact_encoder_plan(',
        'int64_t compact_encoder_schedule_site(',
        'std::vector<int64_t> compact_encoder_schedule_arguments(',
        'int64_t merger_compute_chunk_size(', 'std::vector<ChunkInfo> merger_compute_chunks(',
        'std::vector<int64_t> fp16_compact_residual_arguments(',
        'bool merger_active(', 'bool adaptive_w8_merger(', 'bool deepstack_merger_active(',
        'void set_fp16_chunked_merger(', 'bool large_image_auto_chunk_profile(',
        'int64_t max_keepalive_seq(',
        'int64_t subclass_chunk_size_cap(', 'bool subclass_chunk_size_valid(',
        'bool lingbot_residual_resident_chunk(',
        'bool lingbot_residual_resident_for_plan(',
        'std::vector<int64_t> lingbot_residual_resident_arguments(',
        'bool lingbot_spm_kv_chunk(', 'bool lingbot_spm_kv_for_plan(',
        'bool lingbot_spm_kv_layout(', 'std::vector<int64_t> lingbot_spm_kv_arguments(',
        'bool lingbot_qkv_joint_for_plan(', 'std::vector<int64_t> lingbot_qkv_joint_arguments(',
        'bool lingbot_fc1_gelu_for_plan(', 'std::vector<int64_t> lingbot_fc1_gelu_arguments(',
        'bool tap_merger_for_plan(', 'static const char* resident_merger_name(',
        'bool ordinary_w8_resident_schedule_for_plan(',
        'bool ordinary_w8_split_raw_geometry(',
        'bool ordinary_w8_resident_profile(', 'bool ordinary_w8_resident_chunk(',
        'bool ordinary_w8_split_raw_profile(',
        'std::vector<int64_t> ordinary_resident_schedule_arguments(',
        'bool ordinary_n1200_tm160_profile(', 'bool n1200_tm160_eligible(',
        'std::vector<int64_t> n1200_tm160_arguments(',
        'bool ordinary_w8_raw_chunk(', 'ModelDynamicConfig dynamic_config(',
        'bool subclass_spm_kv_by_mha_eligible(',
        'std::vector<int64_t> vision_attention_arguments(',
        'int64_t vision_raw_query_chunk_size(',
        'std::vector<int64_t> vision_raw_attention_arguments(',
        'std::vector<int64_t> vision_split_raw_attention_arguments(',
        'int64_t vision_ring_route(',
        'FmbPhysicalExecutionManifest make_physical_manifest_for_candidate(',
        'std::vector<FmbPhysicalExecutionManifest>\n    physical_manifest_domain_for_candidate(',
        'std::vector<BufferDecl> declare_buffers(',
    ):
        out += _definition(model, prefix).replace(' override', '') + '\n'
    out += r'''
};
FmbThreeStageChunkPlan plan_for(int64_t n,int64_t chunk=0,int64_t batch=1){
 if(chunk==0)chunk=n;std::vector<FmbExecutionSpan>spans;
 for(int64_t b=0;b<batch;++b)spans.push_back({b*(n/batch),n/batch,b});
 return compose_fmb_three_stage_chunk_plan({{0,0,n,n}},
  make_fmb_chunks(n,0,chunk),make_fmb_chunks(n,0,chunk),std::move(spans),
  chunk>=n?ChunkMode::SEQUENTIAL:ChunkMode::KV_FIRST,{});
}
LayoutContext layout_for(int64_t n,int64_t chunk=0){
 LayoutContext c;c.chunk_size=chunk?chunk:n;c.max_kv_seq_len=n;c.num_layers=24;c.is_causal=false;
 c.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;c.rope_table_residency=FmbRopeTableResidency::SPM;return c;
}
void w8_compact_weights(Qwen3VLVisionModel& m){
 m.linear_acc32_=false;
 for(auto&w:m.layer_weights_)w.q_w.dtype=w.k_w.dtype=w.v_w.dtype=w.o_w.dtype=w.fc1_w.dtype=w.fc2_w.dtype=at::kChar;
 for(int i=-1;i<3;++i){
  auto& m0=i<0?m.merger_m0_w_:m.ds_merger_m0_w_.at(i);
  auto& m2=i<0?m.merger_m2_w_:m.ds_merger_m2_w_.at(i);
  auto& s0=i<0?m.merger_m0_scale_:m.ds_merger_m0_scale_.at(i);
  auto& s2=i<0?m.merger_m2_scale_:m.ds_merger_m2_scale_.at(i);
  m0.dtype=m2.dtype=at::kChar;m0.shape={4096,4096};m2.shape={m.merger_out_hidden_,4096};
  s0.dtype=s2.dtype=at::kHalf;s0.shape={4096};s2.shape={m.merger_out_hidden_};
 }
}
void check_ledger(Qwen3VLVisionModel& m,int64_t n){
 const auto d=m.declare_buffers(layout_for(n));std::vector<SpmAllocator::AllocRequest> req;
 std::vector<std::string> names;std::map<std::string,int64_t> bytes;int64_t persistent=0;
 for(const auto& b:d){
  bytes[b.name]=b.size;
  if(b.alias_of)continue;
  if(b.storage==StorageClass::Temp){req.push_back({b.size,b.phase_start,b.phase_end,int(b.scope)});names.push_back(b.name);}
  else persistent+=b.size*(b.storage==StorageClass::PersistentPerLayer?b.per_layer:1);
 }
 const auto p=SpmAllocator::plan_temporary_aliased(req);std::map<std::string,uint32_t> offsets;
 for(size_t i=0;i<names.size();++i)offsets[names[i]]=p.offsets[i];
 assert(bytes.at("sdpa_tmp")==n*2*64*2 && !bytes.count("q_comp"));
 for(const auto& a:{"q","k","v","sdpa_out","sdpa_tmp"})for(const auto& b:{"q","k","v","sdpa_out","sdpa_tmp"}){
  if(std::string(a)==b)continue;
  assert(offsets[a]+bytes[a]<=offsets[b]||offsets[b]+bytes[b]<=offsets[a]);
 }
 assert(persistent==378880);assert(p.peak_bytes==(n==256?1838080u:n==512?3673088u:n==768?5508096u:7343104u));
 assert(p.peak_bytes+persistent+392192<=8303616); // co-resident Text norms
 std::cerr<<"N"<<n<<" temp="<<p.peak_bytes<<" persistent="<<persistent<<" Vt="<<bytes["sdpa_tmp"]<<"\n";
}
void reject_faults(){
 for(int fault=0;fault<25;++fault){
  Qwen3VLVisionModel m;auto p=plan_for(256);auto c=layout_for(256);int64_t pos=0;
  switch(fault){
   case 0:m.merger_out_hidden_=4096;break;case 1:m.layer_weights_[23].fc2_w.dtype=at::kHalf;break;
   case 2:m.linear_acc32_=true;break;case 3:m.cores=4;break;case 4:m.hd=80;break;
   case 5:m.layers=27;break;case 6:m.pipeline=true;break;case 7:m.delivery=true;break;
   case 8:m.image_batch_count_=2;break;case 9:c.is_causal=true;break;case 10:c.use_attn_mask=true;break;
   case 11:c.batch_size=2;break;case 12:pos=1;break;case 13:p.chunk_mode=ChunkMode::KV_FIRST;break;
   case 14:p.qkv.chunks.push_back(p.qkv.chunks.front());break;case 15:p.input.chunks.front().offset=16;break;
   case 16:p.compute.chunks.front().len=128;break;case 17:p.spans.front().len=128;break;
   case 18:c.chunk_size=128;break;case 19:c.kv_insert_chunk_size=128;break;case 20:c.max_kv_seq_len=512;break;
   case 21:p.compute.chunks.front().kv_seq_len=512;break;case 22:m.merger_m0_w_.dtype=at::kHalf;break;
   case 23:m.heads=8;break;case 24:m.hidden=1152;break;
  }
  assert(!m.subclass_spm_kv_by_mha_eligible(p,c,pos));
 }
 for(int64_t n:{128,196,384,640,784,1024}){Qwen3VLVisionModel m;m.current_num_patches_=n;assert(!m.subclass_spm_kv_by_mha_eligible(plan_for(n),layout_for(n),0));}
 Qwen3VLVisionModel m;m.current_num_patches_=512;assert(!m.subclass_spm_kv_by_mha_eligible(plan_for(512,256),layout_for(512,256),0));
 for(int fault=0;fault<8;++fault){
  Qwen3VLVisionModel pair;pair.current_num_patches_=512;pair.image_batch_count_=2;
  auto p=plan_for(512,0,2);auto c=layout_for(512);
  assert(pair.subclass_spm_kv_by_mha_eligible(p,c,0));
  if(fault==0)p.spans={{0,512,0}};
  if(fault==1)p.spans[1].offset=240;
  if(fault==2)p.spans[1].len=240;
  if(fault==3)p.spans[1].group_id=0;
  if(fault==4)p.qkv.chunks.push_back(p.qkv.chunks.front());
  if(fault==5){pair.current_num_patches_=1536;p=plan_for(1536,0,2);c=layout_for(1536);}
  if(fault==6)pair.image_batch_count_=3;
  if(fault==7)pair.linear_acc32_=true;
  assert(!pair.subclass_spm_kv_by_mha_eligible(p,c,0));
 }
}
int main(){
 reject_faults();std::cout<<"[";bool first=true;
 for(auto shape:std::vector<std::pair<int64_t,int64_t>>{{256,1},{512,1},{768,1},{1024,1},{512,2},{768,3},{1024,2}}){
  auto [n,batch]=shape;Qwen3VLVisionModel m;m.current_num_patches_=n;m.image_batch_count_=batch;auto p=plan_for(n,0,batch);auto c=layout_for(n);
  const bool raw_eligible=!(n==1024&&batch==1);
  auto domain=m.physical_manifest_domain_for_candidate(p,c,n,n,0);
  const auto cfg=m.dynamic_config(p.compute.plan);
  assert(cfg.attention_policy==(!raw_eligible?AttentionExecutionPolicy::DDR_KV:AttentionExecutionPolicy::AUTO));
  // The common native resolver sorts AFTER the owner returns its domain.
  std::sort(domain.begin(),domain.end(),[&](const auto& a,const auto& b){
   return fmb_prefill_stage_candidate_less({n,n,n,p,a},{n,n,n,p,b});
  });
  assert(domain.size()==(raw_eligible?4:2));
  if(raw_eligible)check_ledger(m,n);
  for(const auto& f:domain){
   validate_physical_manifest_impl(f,n,0);
   int raw=0,kv=0;for(const auto& r:f.routes){raw+=r.family==FmbRouteFamily::ATTENTION&&r.selector==2;kv+=r.family==FmbRouteFamily::KV_INSERT;
    if(r.family==FmbRouteFamily::ATTENTION){
     check_fixed_attention(cfg,static_cast<AttentionExecutionPolicy>(r.selector));
     if(r.selector==2){
      auto stale=cfg;stale.attention_policy=AttentionExecutionPolicy::DDR_KV;bool rejected=false;
      try{check_fixed_attention(stale,AttentionExecutionPolicy::SPM_KV_BY_MHA);}catch(const std::runtime_error&){rejected=true;}
      assert(rejected);
      assert(r.site_id==QWEN3VL_RAW_SPM_ATTN_SITE);
      const int64_t image_rows=n/batch,sq=image_rows==768?384:image_rows;
      std::vector<int64_t> expected{batch,sq,image_rows,16,16,64,8,0,64};
      if(image_rows==768)expected.push_back(r.invocation*384);
      assert(r.arguments==expected);assert(r.invocation<(image_rows==768?2:1));
      assert(sdpa_by_mha_spm_is_valid(batch,sq,image_rows,16,16,64,8,0));
     }
    }}
   assert(kv==(raw?0:1));assert(!raw||raw==(n/batch==768?2:1));assert(f.kv_insert_physical_rows==n);
   if(!first)std::cout<<",";first=false;std::cout<<"{\"n\":"<<n<<",\"batch\":"<<batch<<",\"routes\":[";bool rf=true;
   for(const auto& r:f.routes){if(!rf)std::cout<<",";rf=false;std::cout<<"["<<int(r.family)<<","<<r.site_id<<","<<r.selector<<","<<r.flags<<",[";for(size_t i=0;i<r.arguments.size();++i){if(i)std::cout<<",";std::cout<<r.arguments[i];}std::cout<<"],"<<r.invocation<<"]";}
   std::cout<<"]}";
  }
  if(raw_eligible){kv_resolutions=0;m.make_physical_manifest_for_candidate(p,c,n,n,0,true);assert(kv_resolutions==0);}
  m.rope_route_request_=Qwen3VLVisionRopeRouteRequest::DDR;assert(m.physical_manifest_domain_for_candidate(p,c,n,n,0).size()==(raw_eligible?2:1));
  m.rope_route_request_=Qwen3VLVisionRopeRouteRequest::SPM;assert(m.physical_manifest_domain_for_candidate(p,c,n,n,0).size()==(raw_eligible?2:1));
 }
 std::cout<<"]\n";
}
'''
    return out


@pytest.fixture(scope='module')
def native_domain(tmp_path_factory):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    folder = tmp_path_factory.mktemp('qwen3vl_raw_spm')
    cpp, exe = folder / 'native.cpp', folder / 'native'
    cpp.write_text(_program())
    built = subprocess.run([compiler, '-std=c++17', '-O0', str(cpp), '-o', str(exe)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(exe)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    return json.loads(ran.stdout)


def test_native_raw_spm_capability_manifest_and_layout(native_domain):
    assert len(native_domain) == 26  # Four B1 and three bounded packed RAW/DDR x RoPE domains.


def test_compact_encoder_uses_real_joint_layout_and_raw_manifest(tmp_path):
    """Budget the installed layout with the same first-fit/FMB admission code."""
    base = (ROOT / 'src/core/fused_model_base.cpp').read_text()
    allocator = (ROOT / 'src/core/rpu_spm_allocator.h').read_text()
    text = _program().split('int main(){', 1)[0]
    constants = allocator[allocator.index('    static constexpr size_t SPM_TOTAL'):
                          allocator.index('    static constexpr int    NUM_CORES')]
    text = text.replace('class SpmAllocator {\npublic:',
                        'class SpmAllocator {\npublic:\n' + constants)
    text += r'''
struct Watermarks {
 int64_t free=0,temporary=0;
 int64_t free_space()const{return free;}
 int64_t temporary_used()const{return temporary;}
} watermarks;
#define SPM_ALLOC watermarks
namespace detail {int64_t estimate_temporary_total(const std::vector<BufferDecl>&);}
'''
    for marker in ('struct FixedSpmOverhead',
                   'static FixedSpmOverhead estimate_fixed_overhead(',
                   'int64_t detail::estimate_temporary_total(',
                   'static int64_t available_temporary_spm_after_reset(',
                   'static bool final_chunk_layout_fits(',
                   'static void validate_final_chunk_layout_fits('):
        text += _definition(base, marker) + (';\n' if marker.startswith('struct') else '\n')
    # Ordinary Qwen3-VL Text keeps these five norms resident for FP16/W8/W4.
    # Execute their native size/declaration statements, excluding DMA callbacks
    # that do not affect the ledger. No test copy of the norm-size arithmetic.
    decoder = _definition((ROOT / 'src/fused/rpu_qwen3_model.h').read_text(),
                          'std::vector<BufferDecl> causal_baseline_buffer_declarations(')
    text += '\nstd::vector<BufferDecl> text_norm_declarations(int64_t h,int64_t nl){\n'
    text += 'int64_t hd=128;bool adarms_fused=false;std::vector<BufferDecl> decls;\n'
    text += _definition(decoder, 'auto A = ') + ';\n'
    for marker in ('int64_t nw  = ', 'int64_t hnw = '):
        start = decoder.index(marker)
        text += decoder[start:decoder.index(';', start)+1] + '\n'
    for name in ('input_norm', 'post_norm', 'q_norm', 'k_norm', 'final_norm'):
        start = decoder.index(f'BufferDecl {name};')
        text += decoder[start:decoder.index(f'{name}.preload_callback', start)]
        text += f'decls.push_back({name});\n'
    text += 'return decls;}\n'
    text += r'''
template<class F>void rejects(F f){bool raised=false;try{f();}catch(const std::runtime_error&){raised=true;}assert(raised);}
int main(){
 // Every final/DeepStack projection and scale must retain the installed W8
 // metadata. Scale values are checked at registration, not faked by this probe.
 for(int ordinal=-1;ordinal<3;++ordinal)for(bool second:{false,true})for(int fault=0;fault<12;++fault){
  Qwen3VLVisionModel m;w8_compact_weights(m);
  auto&w=ordinal<0?(second?m.merger_m2_w_:m.merger_m0_w_):(second?m.ds_merger_m2_w_[ordinal]:m.ds_merger_m0_w_[ordinal]);
  auto&s=ordinal<0?(second?m.merger_m2_scale_:m.merger_m0_scale_):(second?m.ds_merger_m2_scale_[ordinal]:m.ds_merger_m0_scale_[ordinal]);
  if(fault==0)w.present=false;if(fault==1)w.dtype=at::kHalf;if(fault==2)w.shape[0]-=1;
  if(fault==3)w.shape[1]-=1;if(fault==4)w.device_kind=0;if(fault==5)w.contiguous=false;
  if(fault==6)s.present=false;if(fault==7)s.dtype=at::kChar;if(fault==8)s.shape[0]-=1;
  if(fault==9)s.shape.push_back(1);if(fault==10)s.device_kind=0;if(fault==11)s.contiguous=false;
  rejects([&]{m.set_w8_compact_encoder(true);});assert(!m.w8_compact_encoder_enabled_&&m.invalidations==0);
 }
 for(int fault=0;fault<16;++fault){
  Qwen3VLVisionModel m;w8_compact_weights(m);
  if(fault==0)m.linear_acc32_=true;if(fault==1)m.cores=4;if(fault==2)m.image_batch_count_=2;
  if(fault==3)m.layer_weights_.back().fc2_w.dtype=at::kHalf;if(fault==4)m.deepstack_visual_indexes_[2]=18;
  if(fault==5)m.delivery_compatibility_=true;if(fault==6)m.prefix_scatter_active_=true;
  if(fault==7)m.multiview_composite_prepared_=true;if(fault==8)m.fp16_chunked_merger_enabled_=true;
  if(fault==9)m.ds_merger_m0_w_.pop_back();if(fault==10)m.ds_merger_m2_w_.pop_back();
  if(fault==11)m.ds_merger_m0_scale_.pop_back();if(fault==12)m.ds_merger_m2_scale_.pop_back();
  if(fault==13)m.last_resolved=608;if(fault==14)m.hidden=1152;if(fault==15)m.merge_hidden_=2048;
  rejects([&]{m.set_w8_compact_encoder(true);});assert(!m.w8_compact_encoder_enabled_&&m.invalidations==0);
 }
 for(bool w8:{false,true})for(int64_t output:{2048,2560}){
  Qwen3VLVisionModel m;m.current_num_patches_=1200;m.merger_out_hidden_=output;m.linear_acc32_=true;
  m.merger_m0_w_.dtype=m.merger_m2_w_.dtype=at::kHalf;
  for(auto* weights:{&m.ds_merger_m0_w_,&m.ds_merger_m2_w_})for(auto&w:*weights)w.dtype=at::kHalf;
  for(auto&w:m.layer_weights_)w.q_w.dtype=w.k_w.dtype=w.v_w.dtype=w.o_w.dtype=w.fc1_w.dtype=w.fc2_w.dtype=at::kHalf;
  if(w8){
   w8_compact_weights(m);assert(!m.compact_encoder_active());
   assert(!m.subclass_spm_kv_by_mha_eligible(plan_for(1200),layout_for(1200),0));
   rejects([&]{m.set_fp16_compact_encoder(true);});
   m.set_w8_compact_encoder(true);assert(!m.fp16_compact_encoder_active());
  }
  else{m.set_fp16_chunked_merger(true);m.set_fp16_compact_encoder(true);}
  const auto plan=plan_for(1200);const auto layout=layout_for(1200);
  assert(m.subclass_chunk_size_cap(1200,0)==1200);
  assert(m.subclass_chunk_size_valid(1200,1200,0));
  assert(m.subclass_spm_kv_by_mha_eligible(plan,layout,0));
  const auto cfg=m.dynamic_config(plan.compute.plan);
  assert(cfg.chunk_mode==ChunkMode::SEQUENTIAL&&cfg.inter_layer_io==InterLayerIO::SPM_RESIDENT);
  assert(cfg.attention_policy==AttentionExecutionPolicy::SPM_KV_BY_MHA);
  auto manifest=m.make_physical_manifest_for_candidate(plan,layout,1200,1200,0,true);
  validate_physical_manifest_impl(manifest,1200,0);
  assert(manifest.linear_accumulation==(w8?FmbLinearAccumulationPolicy::ACC16:FmbLinearAccumulationPolicy::ACC32));
  int attention=0;
  for(auto&r:manifest.routes){
   assert(r.family!=FmbRouteFamily::KV_INSERT);
   if(r.family==FmbRouteFamily::ATTENTION){
    assert(r.site_id==QWEN3VL_RAW_SPM_ATTN_SITE&&r.invocation<2);
    const int64_t offset=r.invocation*608,rows=r.invocation==0?608:592;
    assert(r.arguments==std::vector<int64_t>({1,rows,1200,16,16,64,8,0,64,offset}));
    assert(sdpa_by_mha_spm_is_valid(1,rows,1200,16,16,64,8,0));++attention;
   }
  }
  assert(attention==2);
  const auto schedule=std::find_if(manifest.routes.begin(),manifest.routes.end(),[&](auto&r){return r.site_id==(w8?QWEN3VL_W8_COMPACT_SCHEDULE_SITE:QWEN3VL_FP16_COMPACT_SCHEDULE_SITE);});
  assert(schedule!=manifest.routes.end());
  std::vector<int64_t> schedule_args{1200,1,24,1024,4096,16,64,8,output,608,592};
  if(w8)schedule_args.insert(schedule_args.end(),{8,16});assert(schedule->arguments==schedule_args);
  for(auto&r:manifest.routes)if(w8)assert(r.site_id!=QWEN3VL_FP16_COMPACT_SCHEDULE_SITE&&r.site_id!=QWEN3VL_W8_RESIDENT_SCHEDULE_SITE);
  // Scalar enumeration probes declarations before it has a physical
  // descriptor; that default-DDR probe must budget the future raw Vt, too.
  auto unbound=layout;unbound.attention_policy=AttentionExecutionPolicy::DDR_KV;
  unbound.rope_table_residency=FmbRopeTableResidency::UNSPECIFIED;
  unbound.stage_plan_fingerprint=0;
  auto provisional=m.declare_buffers(unbound);
  auto vt=std::find_if(provisional.begin(),provisional.end(),[](const auto&d){return std::string(d.name)=="sdpa_tmp";});
  assert(vt!=provisional.end()&&vt->size==307200);
  auto bound=unbound;bound.stage_plan_fingerprint=17;
  rejects([&]{m.declare_buffers(bound);});
  auto decls=m.declare_buffers(layout);std::map<std::string,BufferDecl> by_name;
  std::vector<SpmAllocator::AllocRequest> requests;std::vector<std::string> names;
  for(auto&d:decls){by_name.emplace(d.name,d);if(d.storage==StorageClass::Temp&&!d.alias_of){requests.push_back({d.size,d.phase_start,d.phase_end,int(d.scope)});names.push_back(d.name);}}
  const auto allocation=SpmAllocator::plan_temporary_aliased(requests);std::map<std::string,uint32_t> offsets;
  for(size_t i=0;i<names.size();++i)offsets[names[i]]=allocation.offsets[i];
  auto& a=by_name.at("residual1");auto& b=by_name.at("input_norm");auto& r=by_name.at("compact_residual");
  assert(a.size==2457600&&b.size==2457600&&r.size==307200);
  assert(a.scope==BufferScope::LayerWide&&b.scope==BufferScope::LayerWide&&r.scope==BufferScope::LayerWide);
  auto& o=by_name.at("oproj");assert(o.alias_of&&std::string(o.alias_of)=="input_norm"&&o.scope==b.scope);
  auto& fc=by_name.at("fc1");assert(!fc.alias_of&&fc.phase_start==5&&fc.phase_end==6);
  assert(!by_name.count("q_comp"));
  for(auto name:{"q","k","v","sdpa_out","sdpa_tmp"})assert(by_name.at(name).size==307200);
  for(auto x:{"residual1","input_norm","compact_residual"})for(auto y:{"residual1","input_norm","compact_residual","q","k","v","sdpa_out","sdpa_tmp","fc1"}){
   if(std::string(x)==y)continue;
   assert(offsets.at(x)+by_name.at(x).size<=offsets.at(y)||offsets.at(y)+by_name.at(y).size<=offsets.at(x));
  }
  size_t post_temp=0,post_persistent=0;
  for(auto&d:decls)if(std::string(d.name).find("merger_")==0){
   assert(!d.alias_of);
   if(d.storage==StorageClass::Temp){assert(d.scope==BufferScope::OutsideLayerLoop);++post_temp;}
   else{assert(w8&&d.storage==StorageClass::Persistent);++post_persistent;}
  }
  assert(post_temp==(w8?5u:9u)&&post_persistent==(w8?4u:0u));
  const auto fixed=estimate_fixed_overhead(decls);
  assert(fixed.persistent==(w8?(output==2048?377856:378880):356352)&&fixed.temp_per_layer==0);
  assert(allocation.peak_bytes==6761472&&detail::estimate_temporary_total(decls)==6761472);
  const auto text_decls=text_norm_declarations(output,output==2048?28:36);
  const auto text_fixed=estimate_fixed_overhead(text_decls);
  assert(text_fixed.persistent==(output==2048?247808:392192)&&text_fixed.temp_per_layer==0);
  // Co-resident Text norms remain charged, both before and after this tower's
  // persistent installation; resetting a previous temporary arena frees it.
  for(bool installed:{false,true}){
   watermarks.temporary=8192;
   watermarks.free=SpmAllocator::SPM_USABLE-text_fixed.persistent-(installed?fixed.persistent:0)-watermarks.temporary;
   assert(final_chunk_layout_fits(provisional,installed));
   assert(final_chunk_layout_fits(decls,installed));
   validate_final_chunk_layout_fits(decls,installed,"compact test");
   const int64_t physical_limit=detail::estimate_temporary_total(decls)+65536+(installed?0:fixed.persistent);
   watermarks.free=physical_limit-watermarks.temporary;
   assert(final_chunk_layout_fits(decls,installed));
   watermarks.free=physical_limit-watermarks.temporary-1;
   assert(!final_chunk_layout_fits(decls,installed));
   rejects([&]{validate_final_chunk_layout_fits(decls,installed,"compact overflow");});
  }
  for(int fault=0;fault<8;++fault){
   auto p=plan;auto l=layout;int64_t position=0;
   if(fault==0)p.compute.chunks[0].len=1184;if(fault==1)p.qkv.chunks[0].offset=16;
   if(fault==2)p.chunk_mode=ChunkMode::KV_FIRST;if(fault==3)l.is_causal=true;
   if(fault==4)l.use_attn_mask=true;if(fault==5)l.batch_size=2;
   if(fault==6)l.kv_insert_chunk_size=608;if(fault==7)position=1;
   assert(!m.subclass_spm_kv_by_mha_eligible(p,l,position));
   rejects([&]{m.make_physical_manifest_for_candidate(p,l,1200,1200,position,true);});
  }
  if(w8){
   auto persistent_identity=[](const auto& declarations){
    std::vector<std::tuple<std::string,int64_t,int,int,int>> result;
    for(auto&d:declarations)if(d.storage==StorageClass::Persistent||d.storage==StorageClass::PersistentPerLayer)
     result.emplace_back(d.name,d.size,int(d.storage),d.per_layer,int(d.scope));
    return result;
   };
   auto identity=persistent_identity(decls);
   for(int n:{256,784,1200,1536})for(int exact:{0,144,256,608}){
    if(n==1200&&exact==0)continue;
    m.current_num_patches_=n;m.configured_chunk_size_=exact;
    assert(!m.w8_compact_encoder_active());
    auto old_layout=layout_for(n,exact?exact:144);old_layout.attention_policy=AttentionExecutionPolicy::DDR_KV;
    auto fallback=m.declare_buffers(old_layout);
    assert(persistent_identity(fallback)==identity);
    Qwen3VLVisionModel old;old.merger_out_hidden_=output;old.current_num_patches_=n;old.configured_chunk_size_=exact;w8_compact_weights(old);
    assert(m.subclass_chunk_size_cap(n,0)==old.subclass_chunk_size_cap(n,0));
    if(exact)assert(m.subclass_chunk_size_cap(n,0)==exact);
    auto reference=old.declare_buffers(old_layout);assert(fallback.size()==reference.size());
    for(size_t i=0;i<fallback.size();++i){auto&a=fallback[i];auto&b=reference[i];
     assert(std::string(a.name)==b.name&&a.size==b.size&&a.storage==b.storage&&a.scope==b.scope&&a.phase_start==b.phase_start&&a.phase_end==b.phase_end);
     assert(std::string(a.alias_of?a.alias_of:"")==std::string(b.alias_of?b.alias_of:""));
    }
   }
   m.current_num_patches_=1200;m.configured_chunk_size_=0;
   assert(persistent_identity(m.declare_buffers(layout))==identity);
   m.last_resolved=1200;const auto count=m.invalidations;
   rejects([&]{m.set_w8_compact_encoder(false);});assert(m.w8_compact_encoder_enabled_&&m.invalidations==count);
  }
 }
}
'''
    cpp, exe = tmp_path / 'compact.cpp', tmp_path / 'compact'
    cpp.write_text(text)
    built = subprocess.run(['c++', '-std=c++17', '-O0', str(cpp), '-o', str(exe)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(exe)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr


@pytest.mark.parametrize('rows,batch', ((256, 1), (512, 1), (768, 1), (1024, 1), (512, 2), (768, 3), (1024, 2)))
def test_public_auto_selects_admitted_native_raw_route(native_domain, monkeypatch, rows, batch):
    import torch
    from rpu_backend.adapters.qwen3_vl import vision
    from test_execution_planner import _native_candidate_v2

    # Use actual C++ manifests above; only the existing test wire encoder fills
    # the outer native ABI and fingerprints. The public Vision planner consumes
    # and ranks this whole domain without a route selector or monkeypatch of its
    # scorer. Device-only domain discovery is the sole replaced op.
    records = []
    for item in native_domain:
        if item['n'] != rows or item['batch'] != batch:
            continue
        record = _native_candidate_v2(rows, (rows, rows, rows), descriptor_version=3,
            graph_lifecycle=4, routes=tuple(tuple(route) for route in item['routes']))
        if batch > 1:
            # The native owner validated all image spans above; encode that
            # same shape in the existing test transport, without changing policy.
            span_start = 15 + 4 * sum(record[11:14])
            assert record[span_start:span_start+3] == [0, rows, -1]
            record[14] = batch
            image_rows = rows // batch
            record[span_start:span_start+3] = [word for image in range(batch)
                for word in (image * image_rows, image_rows, image)]
            record[1] = len(record)
        records.append(record)
    domain = [1, len(records), *(word for record in records for word in (len(record), *record))]
    monkeypatch.setattr(torch.ops.rpu, 'qwen3vl_vision_resolve_stage_domain', lambda *args: domain, raising=False)
    result = vision._plan_qwen3_vl_vision_execution(17, rows, batch, None)
    metadata = dict(result.selected.stage_tuple.physical_metadata)
    # Metadata counts distinct sites; N768 has two invocations of one site.
    assert metadata['raw_attention_site_count'] == int(not (rows == 1024 and batch == 1))
    assert metadata['capability_attention_ddr_site_count'] == int(rows == 1024 and batch == 1)
    assert metadata['rope_table_residency'] == 2


def test_unaligned_packed_raw_domain_and_joint_spm_budget(tmp_path):
    """Real declarations/first-fit/final admission for packed B2/B3 tails."""
    base = (ROOT / 'src/core/fused_model_base.cpp').read_text()
    allocator = (ROOT / 'src/core/rpu_spm_allocator.h').read_text()
    text = _program().split('int main(){', 1)[0]
    constants = allocator[allocator.index('    static constexpr size_t SPM_TOTAL'):
                          allocator.index('    static constexpr int    NUM_CORES')]
    text = text.replace('class SpmAllocator {\npublic:',
                        'class SpmAllocator {\npublic:\n' + constants)
    text += r'''
struct Watermarks {
 int64_t free=0,temporary=0;
 int64_t free_space()const{return free;}
 int64_t temporary_used()const{return temporary;}
} watermarks;
#define SPM_ALLOC watermarks
namespace detail {int64_t estimate_temporary_total(const std::vector<BufferDecl>&);}
'''
    for marker in ('struct FixedSpmOverhead',
                   'static FixedSpmOverhead estimate_fixed_overhead(',
                   'int64_t detail::estimate_temporary_total(',
                   'static int64_t available_temporary_spm_after_reset(',
                   'static bool final_chunk_layout_fits('):
        text += _definition(base, marker) + (';\n' if marker.startswith('struct') else '\n')
    text += r'''
template<class F>void rejects(F fn){bool caught=false;try{fn();}catch(const std::runtime_error&){caught=true;}assert(caught);}
int main(){
 for(int precision:{0,1,2})for(int64_t batch:{2,3})for(int64_t image_rows:{4,12,132,196,252}){
  Qwen3VLVisionModel m;m.image_batch_count_=batch;m.current_num_patches_=batch*image_rows;m.merger_out_hidden_=2048;
  if(precision==2)for(auto&w:m.layer_weights_)
   w.q_w.dtype=w.k_w.dtype=w.v_w.dtype=w.o_w.dtype=w.fc1_w.dtype=w.fc2_w.dtype=at::kHalf;
  if(precision!=0){
   m.merger_m0_w_.dtype=m.merger_m2_w_.dtype=at::kHalf;
   for(auto* weights:{&m.ds_merger_m0_w_,&m.ds_merger_m2_w_})for(auto&w:*weights)w.dtype=at::kHalf;
  }
  const int64_t n=m.current_num_patches_,capacity=CeilDiv(n,int64_t{16})*16,pad=CeilDiv(image_rows,int64_t{16})*16;
  const auto plan=plan_for(n,capacity,batch);auto layout=layout_for(n);
  assert(plan.compute.plan.chunk_size==n&&plan.compute.chunks[0].len==n);
  assert(m.subclass_spm_kv_by_mha_eligible(plan,layout,0));
  auto nominal=layout_for(n,capacity);assert(m.subclass_spm_kv_by_mha_eligible(plan,nominal,0));
  nominal.kv_insert_chunk_size=capacity+16;assert(!m.subclass_spm_kv_by_mha_eligible(plan,nominal,0));
  for(auto rope:{Qwen3VLVisionRopeRouteRequest::AUTO,Qwen3VLVisionRopeRouteRequest::SPM,Qwen3VLVisionRopeRouteRequest::DDR}){
   m.rope_route_request_=rope;kv_resolutions=0;
   auto domain=m.physical_manifest_domain_for_candidate(plan,layout,n,n,0);
   assert(domain.size()==(rope==Qwen3VLVisionRopeRouteRequest::AUTO?2u:1u)&&kv_resolutions==0);
   for(auto&f:domain){
    validate_physical_manifest_impl(f,n,0);assert(f.physical_length==n&&f.logical_length==n&&f.execution_padding_rows==0);
    int attention=0,merged_copies=0;
    for(auto&r:f.routes){
     if(r.site_id==QWEN3VL_W8_RESIDENT_SCHEDULE_SITE){
      std::vector<int64_t> expected{n,batch,24,1024,4096,16,64,8,2048};
      if(precision==2)expected.insert(expected.end(),{16,16});
      assert(r.arguments==expected);
     }
     assert(r.family!=FmbRouteFamily::KV_INSERT);
     if(r.family==FmbRouteFamily::ATTENTION){
      assert(r.selector==2&&r.site_id==QWEN3VL_SPLIT_RAW_SPM_ATTN_SITE);
      assert(r.invocation==attention++);
      assert(r.arguments==m.vision_split_raw_attention_arguments(n,r.invocation));
     }
     if(r.site_id==QWEN3VL_POOLER_MERGED_COPY_SITE||r.site_id==QWEN3VL_DEEPSTACK_MERGED_COPY_SITE){
      assert(r.arguments==std::vector<int64_t>({n/4,2048,0}));++merged_copies;
     }
    }
    assert(attention==batch&&merged_copies==4);
   }
  }
  auto ddr=layout;ddr.attention_policy=AttentionExecutionPolicy::DDR_KV;
  rejects([&]{m.make_physical_manifest_for_candidate(plan,ddr,n,n,0,true);});
  for(int fault=0;fault<8;++fault){
   auto p=plan;auto l=layout;
   if(fault==0)p.spans[1].offset+=4;if(fault==1)p.spans[1].len+=4;
   if(fault==2)p.spans[1].group_id=0;if(fault==3)p.compute.chunks[0].len=capacity;
   if(fault==4)l.batch_size=2;if(fault==5)l.use_attn_mask=true;
   if(fault==6)l.kv_insert_chunk_size=capacity+16;if(fault==7)p.chunk_mode=ChunkMode::KV_FIRST;
   // Aligned total lengths can equal their capacity; mutate by one full tile.
   if(fault==3&&capacity==n)p.compute.chunks[0].len+=16;
   assert(!m.subclass_spm_kv_by_mha_eligible(p,l,0));
  }
  for(int64_t c:{n,capacity}){
   auto l=layout_for(n,c);const auto decls=m.declare_buffers(l);
   std::map<std::string,BufferDecl> by;std::map<std::string,uint32_t> offsets;
   std::vector<SpmAllocator::AllocRequest> requests;std::vector<std::string> names;
   for(auto&d:decls){by.emplace(d.name,d);if(d.storage==StorageClass::Temp&&!d.alias_of){requests.push_back({d.size,d.phase_start,d.phase_end,int(d.scope)});names.emplace_back(d.name);}}
   const auto allocation=SpmAllocator::plan_temporary_aliased(requests);
   for(size_t i=0;i<names.size();++i)offsets[names[i]]=allocation.offsets[i];
   assert(by.at("k").size==(n+pad-image_rows)*256&&by.at("v").size==by.at("k").size);
   assert(by.at("sdpa_tmp").size==pad*256&&!by.count("q_comp"));
   for(auto a:{"q","k","v","sdpa_out","sdpa_tmp"})for(auto b:{"q","k","v","sdpa_out","sdpa_tmp"}){
    if(std::string(a)==b)continue;
    assert(offsets[a]+by.at(a).size<=offsets[b]||offsets[b]+by.at(b).size<=offsets[a]);
   }
   for(int64_t image=0;image<batch;++image){
    assert((image*image_rows+pad)*256<=by.at("k").size);
    assert((image+1)*image_rows*256<=by.at("sdpa_out").size);
    assert((image*image_rows*256)%256==0);
   }
   assert(n/4*2048*2<=by.at("merger_out").size);
   const auto fixed=estimate_fixed_overhead(decls);assert(fixed.persistent==377856);
   for(bool installed:{false,true}){
    watermarks.temporary=8192;
    const int64_t other_persistent=1586688-fixed.persistent;
    watermarks.free=SpmAllocator::SPM_USABLE-other_persistent-(installed?fixed.persistent:0)-watermarks.temporary;
    if(image_rows==196)assert(final_chunk_layout_fits(decls,installed));
    const int64_t limit=detail::estimate_temporary_total(decls)+65536+(installed?0:fixed.persistent);
    watermarks.free=limit-watermarks.temporary;assert(final_chunk_layout_fits(decls,installed));
    --watermarks.free;assert(!final_chunk_layout_fits(decls,installed));
   }
   if(image_rows==196){
    assert(allocation.peak_bytes==(batch==2?(c==n?2819072u:2872320u):(c==n?4224000u:4250624u)));
    std::cerr<<"B"<<batch<<" S196 C"<<c<<" temp="<<allocation.peak_bytes<<" persistent="<<fixed.persistent<<" K/V="<<by.at("k").size<<" Vt="<<by.at("sdpa_tmp").size<<"\n";
   }
  }
 }
 // Aligned B3/S256 uses the existing B3 raw kernel and the same complete
 // four-step composite budget, for both new FP16-merger combinations.
 for(bool fp16_encoder:{false,true}){
  Qwen3VLVisionModel m;m.image_batch_count_=3;m.current_num_patches_=768;m.merger_out_hidden_=2048;
  m.merger_m0_w_.dtype=m.merger_m2_w_.dtype=at::kHalf;
  if(fp16_encoder)for(auto&w:m.layer_weights_)
   w.q_w.dtype=w.k_w.dtype=w.v_w.dtype=w.o_w.dtype=w.fc1_w.dtype=w.fc2_w.dtype=at::kHalf;
  const auto plan=plan_for(768,768,3);auto layout=layout_for(768);
  assert(m.subclass_spm_kv_by_mha_eligible(plan,layout,0));
  assert(m.physical_manifest_domain_for_candidate(plan,layout,768,768,0).size()==4);
  const auto decls=m.declare_buffers(layout);const auto fixed=estimate_fixed_overhead(decls);
  assert(fixed.persistent==377856&&detail::estimate_temporary_total(decls)==5508096);
  for(bool installed:{false,true}){
   watermarks.temporary=8192;
   watermarks.free=SpmAllocator::SPM_USABLE-(1586688-fixed.persistent)-(installed?fixed.persistent:0)-watermarks.temporary;
   assert(final_chunk_layout_fits(decls,installed));
   watermarks.free=detail::estimate_temporary_total(decls)+65536+(installed?0:fixed.persistent)-watermarks.temporary-1;
   assert(!final_chunk_layout_fits(decls,installed));
  }
  layout.attention_policy=AttentionExecutionPolicy::DDR_KV;
  auto fallback=m.make_physical_manifest_for_candidate(plan,layout,768,768,0,true);
  const auto attention=std::find_if(fallback.routes.begin(),fallback.routes.end(),[](const auto&r){return r.family==FmbRouteFamily::ATTENTION;});
  assert(attention!=fallback.routes.end()&&fmb_attention_execution_policy(*attention)==AttentionExecutionPolicy::DDR_KV);
 }
}
'''
    cpp, exe = tmp_path / 'split_budget.cpp', tmp_path / 'split_budget'
    cpp.write_text(text)
    built = subprocess.run(['c++', '-std=c++17', '-O0', str(cpp), '-o', str(exe)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(exe)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr


def test_native_raw_emitter_consumes_actual_query_strips(tmp_path):
    """Execute the real branch and COMPLETE consumer, recording launcher actuals."""
    model = (ROOT / 'src/fused/rpu_qwen3vl_vision_model.cpp').read_text()
    header = (ROOT / 'src/core/fused_model_base.h').read_text()
    stage = (ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    text = _program().split('int main(){')[0]
    text += '\n#include <memory>\n'
    text += 'namespace at {using IntArrayRef=const std::vector<int64_t>&;}\n'
    text += _definition(header, 'enum class FmbChunkTopology ') + ';\n'
    text += _definition(header, 'class FmbPreparedStageCandidate {') + ';\n'
    text += _definition(header, 'class FmbPhysicalManifestConsumer ') + ';\n'
    for prefix in ('void FmbPhysicalManifestConsumer::reset(',
                   'const FmbPhysicalExecutionManifest&\nFmbPhysicalManifestConsumer::manifest()',
                   'bool FmbPhysicalManifestConsumer::complete()',
                   'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(',
                   'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(',
                   'void FmbPhysicalManifestConsumer::require_all_consumed()'):
        text += _definition(stage, prefix) + '\n'
    text += r'''
std::vector<std::vector<int64_t>> launches;
void rpu_launch_v_transpose_spm(uint32_t v,uint32_t vt,int64_t b,int64_t sk,int64_t heads,int64_t d,int cores){
 launches.push_back({0,v,vt,b,sk,heads,d,cores});
}
void rpu_launch_sdpa_by_mha_spm(uint32_t q,uint32_t k,uint32_t vt,uint32_t out,uint32_t mask,int type,
 double scale,int64_t b,int64_t sq,int64_t sk,int64_t qh,int64_t kh,int64_t d,int cores){
 assert(mask==0 && type==0 && scale==0.125);
 assert(sdpa_by_mha_spm_is_valid(b,sq,sk,qh,kh,d,cores,type));
 launches.push_back({1,q,k,vt,out,b,sq,sk,qh,kh,d,cores});
}
struct Context {
 AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
 FmbPhysicalManifestConsumer consumer;
 bool has_complete_physical_manifest()const{return consumer.complete();}
 void consume_physical_route(FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,
    at::IntArrayRef args,int64_t invocation=0){consumer.consume_route(f,site,selector,flags,args,invocation);}
};
struct Runtime:Qwen3VLVisionModel {
 Context context;Context& ctx(){return context;}
 uint32_t addr(int,const char* name){return std::map<std::string,uint32_t>{
  {"q",0x40001000},{"k",0x40100000},{"v",0x40200000},{"sdpa_tmp",0x40300000},{"sdpa_out",0x40400000}}.at(name);}
 void zero_owner_spm(uint32_t address,int64_t elements){launches.push_back({2,address,elements});}
'''
    text += _definition(model, 'void emit_split_raw_attention(')
    text += _definition(model, 'void consume_ordinary_w8_resident_schedule(')
    text += r'''
 void emit(int64_t seq_len){const int64_t local_heads=2,nq=16,hd=64;const double attn_scale=0.125;
'''
    body = model[model.index('void build_layer_subgraph('):]
    text += _definition(body, 'if (ctx().attention_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA &&') + ' else '
    text += _definition(body, 'if (ctx().attention_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA)')
    text += r'''
 }
};
int main(){
 assert(!sdpa_by_mha_spm_is_valid(1,768,768,16,16,64,8,0));
 for(auto shape:std::vector<std::tuple<int64_t,int64_t,bool>>{{256,1,false},{512,1,false},{768,1,false},{512,2,false},{768,3,false},{1024,2,false},{1200,1,false},{1200,1,true}}){
  auto [n,batch,w8]=shape;Runtime m;m.current_num_patches_=n;m.image_batch_count_=batch;
  if(w8){w8_compact_weights(m);m.set_w8_compact_encoder(true);}
  else if(n==1200){
   m.linear_acc32_=true;m.merger_m0_w_.dtype=m.merger_m2_w_.dtype=at::kHalf;
   for(auto* weights:{&m.ds_merger_m0_w_,&m.ds_merger_m2_w_})for(auto&w:*weights)w.dtype=at::kHalf;
   for(auto&w:m.layer_weights_)w.q_w.dtype=w.k_w.dtype=w.v_w.dtype=w.o_w.dtype=w.fc1_w.dtype=w.fc2_w.dtype=at::kHalf;
   m.set_fp16_chunked_merger(true);m.set_fp16_compact_encoder(true);
  }
  auto f=m.make_physical_manifest_for_candidate(plan_for(n,0,batch),layout_for(n),n,n,0,true);
  f.routes.erase(std::remove_if(f.routes.begin(),f.routes.end(),[](const auto& r){return r.family!=FmbRouteFamily::ATTENTION;}),f.routes.end());
  validate_physical_manifest_impl(f,n,0);
  m.context.consumer.reset(f);launches.clear();m.emit(n);m.context.consumer.require_all_consumed();
  assert(launches.size()==(n/batch==768||n==1200?3:2));
  assert(launches[0]==std::vector<int64_t>({0,0x40200000,0x40300000,batch,n/batch,16,64,8}));
  const int64_t sq=n==1200?608:n/batch==768?384:n/batch;
  for(size_t i=1;i<launches.size();++i){const int64_t offset=(i-1)*sq,delta=offset*256;
   const int64_t actual_sq=std::min(sq,n/batch-offset);
   assert(launches[i]==std::vector<int64_t>({1,0x40001000+delta,0x40100000,0x40300000,0x40400000+delta,batch,actual_sq,n/batch,16,16,64,8}));}
  if(n/batch==768||n==1200){
   for(int fault=0;fault<3;++fault){auto bad=f;
    if(fault==0)bad.routes.pop_back();
    else if(fault==1)bad.routes[1].arguments.back()=0;
    else bad.routes[1].invocation=2;
    m.context.consumer.reset(bad);launches.clear();bool rejected=false;
    try{m.emit(n);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
    assert(launches.size()==2); // only the first legal attention call was emitted
   }
  }
  if(batch>1){
   for(int fault=0;fault<4;++fault){auto bad=f;
    if(fault==0)bad.routes[0].arguments[0]=1;
    if(fault==1)bad.routes[0].arguments[1]=2*n/batch;
    if(fault==2)bad.routes[0].arguments[2]=2*n/batch;
    if(fault==3)bad.routes[0].invocation=1;
    m.context.consumer.reset(bad);launches.clear();bool rejected=false;
    try{m.emit(n);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
    assert(launches.empty()); // reject merged-image attention before transpose
   }
  }
 }
 for(int precision:{0,1,2})for(int64_t batch:{2,3})for(int64_t image_rows:{196,256}){
  const int64_t n=image_rows*batch;Runtime m;m.image_batch_count_=batch;m.current_num_patches_=n;
  if(precision==2)for(auto&w:m.layer_weights_)
   w.q_w.dtype=w.k_w.dtype=w.v_w.dtype=w.o_w.dtype=w.fc1_w.dtype=w.fc2_w.dtype=at::kHalf;
  if(precision!=0){
   m.merger_m0_w_.dtype=m.merger_m2_w_.dtype=at::kHalf;
   for(auto* weights:{&m.ds_merger_m0_w_,&m.ds_merger_m2_w_})for(auto&w:*weights)w.dtype=at::kHalf;
  }
  auto f=m.make_physical_manifest_for_candidate(plan_for(n,CeilDiv(n,int64_t{16})*16,batch),layout_for(n),n,n,0,true);
  auto schedule=f;
  schedule.routes.erase(std::remove_if(schedule.routes.begin(),schedule.routes.end(),[](const auto&r){return r.site_id!=QWEN3VL_W8_RESIDENT_SCHEDULE_SITE;}),schedule.routes.end());
  assert(schedule.routes.size()==1);
  m.context.consumer.reset(schedule);m.consume_ordinary_w8_resident_schedule();m.context.consumer.require_all_consumed();
  if(precision==2){
   auto stale=schedule;stale.routes[0].arguments.resize(9);m.context.consumer.reset(stale);
   bool rejected=false;try{m.consume_ordinary_w8_resident_schedule();}catch(const std::runtime_error&){rejected=true;}
   assert(rejected); // the old W8 schedule cannot authorize FP16 emission
  }
  f.routes.erase(std::remove_if(f.routes.begin(),f.routes.end(),[](const auto&r){return r.family!=FmbRouteFamily::ATTENTION;}),f.routes.end());
  if(image_rows==256){
   validate_physical_manifest_impl(f,n,0);assert(f.routes.size()==1);
   m.context.consumer.reset(f);launches.clear();m.emit(n);m.context.consumer.require_all_consumed();
   assert(launches==std::vector<std::vector<int64_t>>({
    {0,0x40200000,0x40300000,batch,256,16,64,8},
    {1,0x40001000,0x40100000,0x40300000,0x40400000,batch,256,256,16,16,64,8}}));
   continue;
  }
  validate_physical_manifest_impl(f,n,0);assert(f.routes.size()==size_t(batch));
  for(int replay=0;replay<3;++replay){
   m.context.consumer.reset(f);launches.clear();m.emit(n);m.context.consumer.require_all_consumed();
   assert(launches.size()==size_t(2+2*batch));
   assert(launches[0]==std::vector<int64_t>({2,0x40100000+n*256,12*128}));
   assert(launches[1]==std::vector<int64_t>({2,0x40200000+n*256,12*128}));
   for(int64_t image=0;image<batch;++image){
    const int64_t delta=image*196*256;
    assert(launches[2+2*image]==std::vector<int64_t>({0,0x40200000+delta,0x40300000,1,196,16,64,8}));
    assert(launches[3+2*image]==std::vector<int64_t>({1,0x40001000+delta,0x40100000+delta,0x40300000,0x40400000+delta,1,196,196,16,16,64,8}));
    assert(f.routes[image].arguments==std::vector<int64_t>({1,196,196,16,16,64,8,0,64,batch,image,image*196,n,208}));
   }
  }
  for(size_t field=0;field<f.routes[0].arguments.size();++field){
   auto bad=f;++bad.routes[0].arguments[field];m.context.consumer.reset(bad);launches.clear();bool rejected=false;
   try{m.emit(n);}catch(const std::runtime_error&){rejected=true;}
   assert(rejected&&launches.empty()); // COMPLETE bound before tail clearing or transpose
  }
  for(int fault=0;fault<3;++fault){
   auto bad=f;if(fault==0)bad.routes[0].selector=1;
   if(fault==1)bad.routes[0].invocation=9;if(fault==2)bad.routes[0].flags=1;
   m.context.consumer.reset(bad);launches.clear();bool rejected=false;
   try{m.emit(n);}catch(const std::runtime_error&){rejected=true;}
   assert(rejected&&launches.empty());
  }
  auto bad=f;bad.routes.pop_back();m.context.consumer.reset(bad);launches.clear();bool rejected=false;
  try{m.emit(n);}catch(const std::runtime_error&){rejected=true;}
  assert(rejected&&launches.size()==size_t(2+2*(batch-1)));
 }
}
'''
    cpp, exe = tmp_path / 'native_emitter.cpp', tmp_path / 'native_emitter'
    cpp.write_text(text)
    built = subprocess.run(['c++', '-std=c++17', '-O0', str(cpp), '-o', str(exe)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(exe)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr



def test_native_resident_schedule_prefers_exact_lingbot_contract(tmp_path):
    """Run the real schedule selection, declarations and COMPLETE consumers."""
    model = (ROOT / 'src/fused/rpu_qwen3vl_vision_model.cpp').read_text()
    header = (ROOT / 'src/core/fused_model_base.h').read_text()
    stage = (ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    # Reuse the existing cold metadata doubles and production pure helpers.
    # Keep only the earlier profile helpers; unrelated launcher bodies are not needed.
    text = _program().split('bool ordinary_w8_split_raw_geometry(', 1)[0]
    text = text.replace('std::runtime_error("native contract rejected")',
                        'std::runtime_error("native contract rejected: " #c)', 1)
    consumer = '\n#include <memory>\nnamespace at {using IntArrayRef=const std::vector<int64_t>&;}\n'
    for prefix in ('enum class FmbChunkTopology ', 'class FmbPreparedStageCandidate {',
                   'class FmbPhysicalManifestConsumer '):
        consumer += _definition(header, prefix) + ';\n'
    for prefix in ('void FmbPhysicalManifestConsumer::reset(',
                   'const FmbPhysicalExecutionManifest&\nFmbPhysicalManifestConsumer::manifest()',
                   'bool FmbPhysicalManifestConsumer::complete()',
                   'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(',
                   'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(',
                   'void FmbPhysicalManifestConsumer::require_all_consumed()'):
        consumer += _definition(stage, prefix) + '\n'
    consumer += r"""
struct Context {
 FmbThreeStageChunkPlan stage_plan; FmbPhysicalManifestConsumer consumer;
 int64_t position=0,batch_size=1; bool is_causal=false,input_in_spm=false,output_to_spm=false;
 std::optional<Tensor> attention_mask;
 AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV;
 bool has_complete_physical_manifest()const{return consumer.complete();}
 const FmbPhysicalExecutionManifest& physical_manifest()const{return consumer.manifest();}
 void consume_physical_route(FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,
    at::IntArrayRef args,int64_t invocation=0){consumer.consume_route(f,site,selector,flags,args,invocation);}
};
"""
    text = text.replace('class Qwen3VLVisionModel:public FusedModelBase {',
                        consumer + '\nclass Qwen3VLVisionModel:public FusedModelBase {', 1)
    text += r"""
 Context context;Context& ctx(){return context;}
"""
    for prefix in ('bool ordinary_w8_split_raw_geometry(',
                   'bool ordinary_w8_resident_profile(', 'bool ordinary_w8_resident_chunk(',
                   'bool ordinary_n1200_tm160_profile(',
                   'std::vector<int64_t> ordinary_resident_schedule_arguments(',
                   'void consume_ordinary_w8_resident_schedule(',
                   'bool consume_lingbot_residual_resident_schedule('):
        text += _definition(model, prefix) + '\n'
    text += r"""
 FmbPhysicalExecutionManifest schedule(const FmbThreeStageChunkPlan& plan,int64_t physical_len){
  FmbPhysicalExecutionManifest manifest;manifest.state=FmbPhysicalManifestState::COMPLETE;
  const int64_t logical_len=physical_len,position=0;
  LayoutContext layout;layout.is_causal=false;layout.attention_policy=AttentionExecutionPolicy::DDR_KV;
  const bool lingbot_spm_kv=false;
  auto append=[&](FmbRouteFamily family,int64_t site,int64_t selector,std::vector<int64_t> args){
   manifest.routes.push_back({site,family,selector,0,std::move(args),0});
  };
"""
    text += _definition(model, 'if (ordinary_w8_resident_schedule_for_plan(plan, physical_len))')
    text += _definition(model, 'if (lingbot_residual_resident_chunk(plan.compute.plan.chunk_size))')
    text += r"""
  return manifest;
 }
 void emit(int layer_idx,const ChunkInfo& chunk){
  const bool lingbot_residual=consume_lingbot_residual_resident_schedule(layer_idx,chunk);
  const bool delivery=qwen3vl_2b_delivery_profile_active();
  const bool compact=compact_encoder_active();
  const bool sequential=ctx().stage_plan.chunk_mode==ChunkMode::SEQUENTIAL;
"""
    body = model[model.index('void build_layer_subgraph('):]
    check_start = body.index('        TORCH_CHECK(!sequential || delivery || compact || lingbot_residual ||')
    check_end = body.index(';', check_start) + 1
    text += body[check_start:check_end]
    text += _definition(body, 'if (sequential && !delivery && layer_idx == 0 && chunk.idx == 0)')
    text += r"""
 }
};
FmbThreeStageChunkPlan plan_for(int64_t n,int64_t batch){
 std::vector<FmbExecutionSpan> spans;
 for(int64_t b=0;b<batch;++b)spans.push_back({b*(n/batch),n/batch,b});
 return compose_fmb_three_stage_chunk_plan({{0,0,n,n}},make_fmb_chunks(n,0,n),
  make_fmb_chunks(n,0,n),std::move(spans),ChunkMode::SEQUENTIAL,{});
}
int main(){
 for(bool fp16:{false,true}){
  Qwen3VLVisionModel m;m.current_num_patches_=768;m.image_batch_count_=3;
  if(fp16){
   for(auto&w:m.layer_weights_)w.q_w.dtype=w.k_w.dtype=w.v_w.dtype=w.o_w.dtype=w.fc1_w.dtype=w.fc2_w.dtype=at::kHalf;
   m.merger_m0_w_.dtype=m.merger_m2_w_.dtype=at::kHalf;
  }
  m.tap_merger_enabled_=true;m.lingbot_residual_resident_mode_=fp16?2:1;
  const auto p=plan_for(768,3);m.context.stage_plan=p;
  assert(m.ordinary_w8_resident_chunk(768,768)); // both cold capabilities overlap
  assert(m.lingbot_residual_resident_for_plan(p));
  assert(!m.ordinary_w8_resident_schedule_for_plan(p,768));
  auto selected=m.schedule(p,768);
  assert(selected.routes.size()==1&&selected.routes.front().site_id==QWEN3VL_LINGBOT_RESIDUAL_RESIDENT_SITE);
  m.context.consumer.reset(selected);
  for(int layer=0;layer<24;++layer){
   m.context.input_in_spm=layer!=0;m.context.output_to_spm=layer!=23;
   m.emit(layer,p.compute.chunks.front());
  }
  m.context.consumer.require_all_consumed();
  auto malformed=p;malformed.spans[1].offset+=16;
  bool rejected=false;try{m.schedule(malformed,768);}catch(const std::runtime_error&){rejected=true;}
  assert(rejected); // malformed exact traversal cannot fall through to generic emission
  m.tap_merger_enabled_=false;m.lingbot_residual_resident_mode_=0;
  auto ordinary=m.schedule(p,768);
  assert(ordinary.routes.size()==1&&ordinary.routes.front().site_id==QWEN3VL_W8_RESIDENT_SCHEDULE_SITE);
  m.context.input_in_spm=false;m.context.output_to_spm=true;m.context.consumer.reset(ordinary);
  m.emit(0,p.compute.chunks.front());m.context.consumer.require_all_consumed();
  if(fp16)assert(ordinary.routes.front().arguments.size()==11);
  else assert(ordinary.routes.front().arguments.size()==9);
  m.tap_merger_enabled_=true;m.lingbot_residual_resident_mode_=fp16?2:1;
  auto stale=selected;stale.routes.push_back(ordinary.routes.front());
  std::sort(stale.routes.begin(),stale.routes.end(),[](const auto&a,const auto&b){return a.site_id<b.site_id;});
  m.context.consumer.reset(stale);
  m.emit(0,p.compute.chunks.front());rejected=false;
  try{m.context.consumer.require_all_consumed();}catch(const std::runtime_error&){rejected=true;}
  assert(rejected); // the old double declaration remains fail closed
  selected.routes.front().arguments.front()+=1;m.context.consumer.reset(selected);rejected=false;
  try{m.emit(0,p.compute.chunks.front());}catch(const std::runtime_error&){rejected=true;}
  assert(rejected); // actual consumer still checks the exact schedule arguments
  m.current_num_patches_=512;m.image_batch_count_=2;
  const auto two=plan_for(512,2);m.context.stage_plan=two;
  assert(!m.lingbot_residual_resident_for_plan(two));
  auto fallback=m.schedule(two,512);
  assert(fallback.routes.size()==1&&fallback.routes.front().site_id==QWEN3VL_W8_RESIDENT_SCHEDULE_SITE);
  m.context.consumer.reset(fallback);m.emit(0,two.compute.chunks.front());m.context.consumer.require_all_consumed();
 }
}
"""
    cpp, exe = tmp_path / 'resident_schedule.cpp', tmp_path / 'resident_schedule'
    cpp.write_text(text)
    built = subprocess.run(['c++', '-std=c++17', '-O0', str(cpp), '-o', str(exe)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(exe)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr
