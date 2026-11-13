"""Exercise actual cold LingBot QKV selection and COMPLETE emission on CPU."""
from pathlib import Path
import re

import torch

from test_siglip_attention_dry_live_contract import _function, _run_native

ROOT = Path(__file__).resolve().parents[1]


def test_actual_long_prefill_deepstack_alias_preserves_layout_and_consumption(tmp_path):
    source = (ROOT / 'src/fused/rpu_qwen3_model.h').read_text()
    core = (ROOT / 'src/core/fused_model_base.cpp').read_text()
    helpers = (ROOT / 'src/core/rpu_helpers.h').read_text()
    allocator_source = (ROOT / 'src/core/rpu_spm_allocator.cpp').read_text()
    declarations = _function(source, 'std::vector<BufferDecl> causal_baseline_buffer_declarations(')
    declarations = declarations[declarations.index('{') + 1:
                                declarations.index('// Norm-weight preloads')]
    # Execute the complete production temporary declaration body, including
    # its real sizes, scopes, aliases and first-fit allocation. Persistent
    # preload callbacks are outside this local owner-layout check.
    selector = _function(source, 'bool lingbot_prefill_deepstack_reuses_down(')
    sdpa_types = '\n'.join(_function(helpers, marker) + ';' for marker in (
        'struct SdpaTiling', 'enum class SdpaKernelType', 'struct SdpaConfig'))
    sdpa_methods = '\n'.join(_function(helpers, marker) for marker in (
        'inline int64_t sdpa_tile_m_v16_initial(',
        'inline uint32_t GetMhaAttnUnivVlmNeed(',
        'inline std::optional<int64_t> choose_tile_k_conservative(',
        'inline std::optional<int64_t> choose_tile_k(',
        'inline bool sdpa_try_compute_tiling(', 'inline SdpaTiling sdpa_compute_tiling(',
        'inline int64_t sdpa_compute_tmp_v16_size('))
    allocator = _function(allocator_source, 'SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(')
    aliases = _function(core, 'static void resolve_aliases_impl(')
    mlp = _function(core, 'void FusedModelBase::emit_mlp_pipeline(')
    down_start = mlp.index('        emit_linear(\n            addr(0, "gate"), down_w, addr(0, "down"),')
    up_write = mlp.index('addr(0, "residual1"), up_w, addr(0, "up")')
    up_last_read = mlp.rindex('addr(0, "gate"), addr(0, "up"), addr(0, "gate")')
    assert up_write < up_last_read < down_start
    assert 'addr(0, "up")' not in mlp[down_start:]
    down_end = mlp.index('        if (residual_is_bf16)', down_start)
    down_emit = mlp[down_start:down_end]
    ring = _function(mlp[down_end:], '        } else {')
    ring = ring[ring.index('{') + 1:-1]
    body = _function(source, 'void build_layer_subgraph(')
    ds_marker = body.index('// ★ Phase 8.5 (R-Phase 2): DeepStack visual_embed inject')
    assert body.rindex('emit_mlp_pipeline(', 0, ds_marker) < ds_marker
    ds = _function(body[ds_marker:], '} else if (!deepstack_lang_layers_.empty())')
    ds = ds[ds.index('{') + 1:-1]
    constant = re.search(r'inline constexpr int64_t QWEN3_FUSED_LM_HEAD_MAX_VOCAB\s*=\s*[^;]+;', source).group()
    ds_constant = re.search(r'inline constexpr int64_t CAUSAL_DECODER_DEEPSTACK_SITE\s*=\s*[^;]+;', source).group()
    program = r'''
#include "core/fused_model_base.h"
#include "core/rpu_spm_allocator.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <unordered_map>
using namespace v3;
#define CeilDiv(x,y) (((x)+(y)-1)/(y))
#define Align(x,y) (((x)+(y)-1)/(y)*(y))
#define SDPA_MAX_VLM_SIZE 400
#define RPU_SDPA_TILE_K_CONSERVATIVE_MODE 1
constexpr int NUM_CORES=8,DWIDTH=2;
__SDPA_TYPES__
__SDPA_METHODS__
__CONSTANTS__
enum class CausalDecoderPlanningMode {ORDINARY,OTHER};
enum class CausalDecoderMutableDmaRoute {DDR_BROADCAST_TO_SPM=1};
enum class ValuOpType {ADD};
__ALLOCATOR__
__ALIASES__
struct PipelineContext {int physical_route_invocation=0;};
struct PipelineState {int hidden_size_=2560,intermediate_size_=9728;PipelineContext ctx_;};
struct Memory {
 std::vector<c10::Half> bytes;
 std::vector<c10::Half> visual;
 std::unordered_map<std::string,uint32_t> offsets;
 std::vector<std::string> events;
 bool down_consumed=false;
};
Memory* active=nullptr;
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t down,uint32_t residual,uint32_t out,
 int64_t rows,int64_t hidden,int input_cores,int output_cores,bool){
 assert(input_cores==8&&output_cores==8);
 assert(!active->down_consumed);
 for(int64_t i=0;i<rows*hidden;++i)
  active->bytes[out/2+i]=c10::Half(float(active->bytes[down/2+i])+float(active->bytes[residual/2+i]));
 active->down_consumed=true;active->events.push_back("ring");
}
void rpu_launch_ddr_broadcast_spm_dma_mutable(uint64_t*,int64_t source_offset,int64_t count,
 uint32_t out,int cores){
 assert(cores==8&&source_offset==0&&active->down_consumed);
 assert(size_t(count)==active->visual.size());
 std::copy(active->visual.begin(),active->visual.end(),active->bytes.begin()+out/2);
 active->events.push_back("deepstack_dma");
}
void rpu_launch_eltwise_binary_spm_kernel(uint32_t in,uint32_t other,uint32_t out,
 int64_t count,ValuOpType,c10::Half,int cores){
 assert(cores==8&&in==out&&active->down_consumed);
 for(int64_t i=0;i<count;++i)
  active->bytes[out/2+i]=c10::Half(float(active->bytes[in/2+i])+float(active->bytes[other/2+i]));
 active->events.push_back("deepstack_add");
}
namespace v3 {int64_t fmb_ring_all_reduce_route_selector(int64_t,int64_t,bool,int){return 1;}}
void consume_shared_runtime_route(PipelineContext&,FmbRouteFamily,int64_t,int64_t,int64_t,at::IntArrayRef,int,int){ }
struct Model {
 bool lingbot_prefill_mode_bound_=true,lingbot_folded_rmsnorm_=true,lingbot_prefill_kv_only_=true;
 bool wall_prefill_staged_mlp_=false,decode_gate_up_fusion_=false,prefill_packed_gate_up_strided_=false;
 bool qwen3vl_multiview_text_dry_prepared_=false,qwen3vl_multiview_text_composite_prepared_=false;
 int64_t qwen3vl_pooler_z1_retained_deepstack_count_=0;
 bool qwen3vl_2b_w8_profile_=false;
 std::vector<int64_t>deepstack_lang_layers_{0,1,2};
 CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
 PipelineState state;PipelineState*pimpl_=&state;
 int64_t hidden_size()const{return 2560;}int64_t intermediate_size()const{return 9728;}
 int64_t num_q_heads()const{return 32;}int64_t num_kv_heads()const{return 8;}
 int64_t head_dim()const{return 128;}int64_t num_layers()const{return 36;}
 int mlp_tp()const{return 8;}int attn_tp()const{return 8;}int lm_head_tp()const{return 8;}
 int num_cores()const{return 8;}
 bool qwen3vl_4b_packed_decode_qkv(int64_t,bool,bool,int64_t)const{return false;}
 SdpaConfig make_sdpa_config(int mask)const{return {SdpaKernelType::FLASH_ATTN_SPM,128,32,8,8,mask,0};}
 __SELECTOR__
 std::vector<BufferDecl> declarations(const LayoutContext&ctx){__DECLARATIONS__ return decls;}
 uint32_t addr(int,const char*name)const{return active->offsets.at(name);}
 uint64_t deepstack_dense_src_base_[3]{};
 int64_t chunk_row_offset_(const ChunkInfo&chunk)const{return chunk.offset;}
 struct Context{int physical_route_invocation=0;};Context context;
 Context&ctx(){return context;}
 template<class...T>void consume_causal_route(T...){ }
 void emit_linear(uint32_t,const at::Tensor&,uint32_t out,int64_t n,int64_t k,int,
 const at::Tensor&,uint32_t){
  assert(!active->down_consumed&&n==2560&&k==9728);
  for(size_t i=0;i<active->visual.size();++i)active->bytes[out/2+i]=c10::Half(float(i%5));
  active->events.push_back("down");
 }
 void finish_mlp_and_deepstack(int layer_idx,const ChunkInfo&chunk){
  const int64_t seq_len=chunk.len,rows=chunk.len,h=hidden_size();
  const int kNumCores=8;const bool pi05_xor3=false;
  const uint32_t residual=addr(0,"residual2"),down_nvfp4_ts_addr=0;
  const at::Tensor down_w,down_ws;
  __DOWN_EMIT__
  __RING__
  __DEEPSTACK__
 }
};
struct LayoutProof {std::vector<BufferDecl>decls;std::unordered_map<std::string,uint32_t>offsets;size_t peak;};
LayoutProof plan(Model&m,const LayoutContext&layout){
 auto decls=m.declarations(layout);std::vector<SpmAllocator::AllocRequest>requests;
 std::vector<std::string>names;
 for(const auto&d:decls)if(d.storage==StorageClass::Temp&&!d.alias_of){
  requests.push_back({d.size,d.phase_start,d.phase_end,int(d.scope)});names.push_back(d.name);}
 auto allocation=SpmAllocator::plan_temporary_aliased(requests);
 std::unordered_map<std::string,uint32_t>offsets;
 for(size_t i=0;i<names.size();++i)offsets[names[i]]=allocation.offsets[i];
 resolve_aliases_impl(decls,offsets);
 return {std::move(decls),std::move(offsets),allocation.peak_bytes};
}
const BufferDecl&find(const LayoutProof&p,const char*name){
 return *std::find_if(p.decls.begin(),p.decls.end(),[&](const auto&d){return std::string(d.name)==name;});}
LayoutContext full(int64_t rows){LayoutContext l;l.chunk_size=rows;l.max_kv_seq_len=rows;return l;}
int main(){Model enabled,ordinary;ordinary.lingbot_prefill_kv_only_=false;
 for(int64_t rows:{176,224,320,400}){
  auto layout=full(rows);auto selected=plan(enabled,layout),baseline=plan(ordinary,layout);
  assert(find(selected,"down").phase_start==(rows==400?8:7)&&find(selected,"down").phase_end==8);
  if(rows<320){assert(selected.peak==baseline.peak&&selected.offsets==baseline.offsets);
   assert(find(selected,"deepstack_scratch").alias_of==nullptr);continue;}
  if(rows==320)assert(selected.peak==6471680&&baseline.peak==8110080);
  else{
   assert(selected.peak==7430144&&baseline.peak==10137600);
   assert(selected.offsets.at("down")==selected.offsets.at("up"));
   assert(find(selected,"up").phase_end==7&&find(selected,"down").phase_start==8);
   assert(find(selected,"sdpa_tmp").size==262144);
  }
  const auto&alias=find(selected,"deepstack_scratch");
  assert(alias.size==0&&std::string(alias.alias_of)=="down"&&alias.scope==BufferScope::LayerWide);
  assert(alias.phase_start==8&&alias.phase_end==8&&find(selected,"down").size==rows*2560*2);
  assert(selected.offsets.at("deepstack_scratch")==selected.offsets.at("down"));
  // The real declaration-derived peak admits a complete resident top that
  // rejects the old independent scratch. No planner/available limit is widened.
  constexpr int64_t resident_top=854272,reserve=65536;
  const int64_t available=int64_t(SpmAllocator::SPM_USABLE)-resident_top-reserve;
  assert(selected.peak<=size_t(available)&&baseline.peak>size_t(available));
  std::cout<<"actual_layout "<<rows<<" selected_peak="<<selected.peak
   <<" ordinary_peak="<<baseline.peak<<" available="<<available<<"\n";
  for(int layer=0;layer<3;++layer){std::vector<c10::Half>expected;
   for(auto*proof:{&baseline,&selected}){
    Memory memory;memory.offsets=proof->offsets;memory.bytes.resize(proof->peak/2,c10::Half(-31.f));
    memory.visual.resize(rows*2560,c10::Half(float(2+layer)));
    std::fill_n(memory.bytes.begin()+memory.offsets.at("residual2")/2,rows*2560,c10::Half(1.f));
    active=&memory;enabled.finish_mlp_and_deepstack(layer,{0,0,rows,rows});
    assert(memory.events==std::vector<std::string>({"down","ring","deepstack_dma","deepstack_add"}));
    std::vector<c10::Half>output(memory.bytes.begin()+memory.offsets.at("residual1")/2,
     memory.bytes.begin()+memory.offsets.at("residual1")/2+rows*2560);
    for(int64_t i=0;i<rows*2560;++i)assert(float(output[i])==float(i%5+3+layer));
    if(expected.empty())expected=output;else assert(std::memcmp(expected.data(),output.data(),output.size()*2)==0);
   }
  }
 }
 for(int64_t rows:{320,400})for(int fault=0;fault<10;++fault){Model bad;auto layout=full(rows);
  if(fault==0)bad.lingbot_prefill_mode_bound_=false;
  if(fault==1)bad.lingbot_folded_rmsnorm_=false;
  if(fault==2)bad.lingbot_prefill_kv_only_=false;
  if(fault==3)bad.planning_mode_=CausalDecoderPlanningMode::OTHER;
  if(fault==4)bad.decode_gate_up_fusion_=true;
  if(fault==5)bad.wall_prefill_staged_mlp_=true;
  if(fault==6)layout.is_causal=false;
  if(fault==7)layout.use_attn_mask=true;
  if(fault==8)layout.batch_size=2;
  if(fault==9)layout.kv_insert_chunk_size=rows/2;
  assert(!bad.lingbot_prefill_deepstack_reuses_down(layout));
 }
 auto split=full(320);split.chunk_size=160;assert(!enabled.lingbot_prefill_deepstack_reuses_down(split));
 auto append=full(320);append.max_kv_seq_len=336;assert(!enabled.lingbot_prefill_deepstack_reuses_down(append));
 auto long_split=full(400);long_split.chunk_size=200;assert(!enabled.lingbot_prefill_deepstack_reuses_down(long_split));
 auto long_append=full(400);long_append.max_kv_seq_len=416;assert(!enabled.lingbot_prefill_deepstack_reuses_down(long_append));
 assert(!enabled.lingbot_prefill_deepstack_reuses_down(full(384)));
 std::cout<<"long_prefix_alias_pass\n";
}
'''
    for key, value in {
        '__SDPA_TYPES__': sdpa_types, '__SDPA_METHODS__': sdpa_methods,
        '__CONSTANTS__': constant + '\n' + ds_constant,
        '__ALLOCATOR__': allocator, '__ALIASES__': aliases, '__SELECTOR__': selector,
        '__DECLARATIONS__': declarations, '__DOWN_EMIT__': down_emit,
        '__RING__': ring, '__DEEPSTACK__': ds,
    }.items():
        program = program.replace(key, value)
    output = _run_native(tmp_path, program, torch)
    print(output, end='')
    assert 'long_prefix_alias_pass' in output


def test_actual_lingbot_prefill_qkv_owner(tmp_path):
    source = (ROOT / 'src/fused/rpu_qwen3_model.h').read_text()
    names = ('LINGBOT2_PREFILL_QKV_MULTIBASE_SITE', 'CAUSAL_DECODER_Q_SITE',
             'CAUSAL_DECODER_K_SITE', 'CAUSAL_DECODER_V_SITE')
    constants = '\n'.join(re.search(rf'inline constexpr int64_t {n}\s*=\s*[^;]+;', source).group() for n in names)
    methods = '\n'.join(_function(source, m) for m in (
        'void set_lingbot_prefill_mode(', 'int64_t select_lingbot_prefill_qkv_mode(',
        'bool lingbot_prefill_qkv_for_plan(', 'std::vector<int64_t> lingbot_prefill_qkv_arguments()',
        'bool validate_lingbot_prefill_qkv_schedule(',
        'bool select_lingbot_prefill_gateup_shared_a('))
    # Device metadata is the only host substitution; tensor owners, shapes,
    # precision and real FMB route matching/consumption remain executable.
    methods = methods.replace('at::kPrivateUse1', 'at::kCPU')
    manifest = _function(source, 'FmbPhysicalExecutionManifest physical_manifest_for_candidate(')
    append = _function(manifest, 'if (lingbot_prefill_qkv_for_plan(plan, physical_len, position,')
    fallback_append = _function(manifest, 'if (!every_wall_action_qkv && !every_wall_p640_staged0_qkv)')
    fallback_append = fallback_append[fallback_append.index('{') + 1:-1]
    emit = _function(source, 'if (validate_lingbot_prefill_qkv_schedule(layer_idx, chunk))')
    tail = _function(source, 'if (lingbot_prefill_kv_only_ && is_last_layer)')
    qkv_start = source.index('        if (validate_lingbot_prefill_qkv_schedule(')
    phase3 = source.index('// Phase 3: QK RMSNorm + RoPE', qkv_start)
    # Main also has Wall NVFP4 Q-only branches; select the final ordinary Q/K/V
    # fallback of this same emission chain, not the first Q-site occurrence.
    start = source.rindex('            consume_causal_route(\n                FmbRouteFamily::LINEAR, CAUSAL_DECODER_Q_SITE,', qkv_start, phase3)
    stop = source.index('\n        }', start)
    ordinary = source[start:stop]
    layout = _function(_function(source, 'int64_t subclass_layout_hash() const'), 'if (lingbot_prefill_qkv_mode_)')
    cost = _function(_function(source, 'std::vector<int64_t> causal_kvinsert_cost_weight_identity() const'), 'if (lingbot_prefill_qkv_mode_)')
    guard_start = source.index('TORCH_CHECK(!lingbot_prefill_mode_bound_,', source.index('    void set_weights('))
    guard = source[guard_start:source.index(';', guard_start) + 1]
    program = r'''
#include "core/fused_model_base.h"
#include <ATen/Parallel.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <typeinfo>
using namespace v3;
__CONSTANTS__
__ENUMS__
enum class KernelId : uint16_t {LINGBOT2_PREFILL_QKV_MULTIBASE_W8_M224,LINGBOT2_PREFILL_QKV_MULTIBASE_FP16_M224,LINGBOT2_PREFILL_GATEUP_SHARED_A_W8_M224};
struct KernelCache{bool w8=true,fp16=true;static KernelCache&instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId id)const{return id==KernelId::LINGBOT2_PREFILL_QKV_MULTIBASE_W8_M224?w8:fp16;}};
struct RpuExecutionCoordinator {static bool quiet;static void require_graph_quiescent(const char*){TORCH_CHECK(quiet,"busy");}
 static void check_current_thread_execution_allowed(const char*){}};bool RpuExecutionCoordinator::quiet=true;
struct RpuKernelGraph{static bool has_active(){return false;}};
struct RpuRmsNormSpmContract{int capability=0;};
RpuRmsNormSpmContract rpu_snapshot_rmsnorm_nogamma_spm_contract(){return {7};}
struct Context{
 FmbThreeStageChunkPlan stage_plan;FmbPhysicalManifestConsumer consumer;
 int64_t seq_len=224,position=0,batch_size=1;bool is_causal=true;std::optional<int>attention_mask;
 AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV;
 bool has_complete_physical_manifest()const{return consumer.complete();}
 const FmbPhysicalExecutionManifest&physical_manifest()const{return consumer.manifest();}
 void consume_physical_route(FmbRouteFamily f,int64_t s,int64_t r,int64_t flags,at::IntArrayRef a={},int64_t i=0){consumer.consume_route(f,s,r,flags,a,i);}
};
struct Layer {at::Tensor q_w,k_w,v_w,o_w,gate_w,up_w,down_w,q_ws,k_ws,v_ws,gate_ws,up_ws,down_ws;};
int fused=0,ordinary=0,tails=0;const Layer*expected=nullptr;bool live_payload=true;
void rpu_launch_lingbot2_prefill_qkv_m224_spm_kernel(uint32_t x,const at::Tensor&q,const at::Tensor&k,const at::Tensor&v,
 uint32_t qo,uint32_t ko,uint32_t vo,const at::Tensor&qs,const at::Tensor&ks,const at::Tensor&vs){
 TORCH_CHECK(live_payload,"missing payload");assert(x==0x1000&&qo==0x2000&&ko==0x3000&&vo==0x4000);
 assert(q.is_same(expected->q_w)&&k.is_same(expected->k_w)&&v.is_same(expected->v_w));
 assert(qs.is_same(expected->q_ws)&&ks.is_same(expected->k_ws)&&vs.is_same(expected->v_ws));++fused;
}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t x,const at::Tensor&w,uint32_t out,int64_t m,int64_t n,int64_t k,
 int partition,int cores,uint32_t bias,const at::Tensor&s,int,uint16_t,bool,bool,bool){
 assert(x==0x1000&&m>0&&k==2560&&partition==1&&cores==8&&bias==0);
 if(out==0x2000){assert(n==4096&&w.is_same(expected->q_w)&&s.is_same(expected->q_ws));}
 else if(out==0x3000){assert(n==1024&&w.is_same(expected->k_w)&&s.is_same(expected->k_ws));}
 else{assert(out==0x4000&&n==1024&&w.is_same(expected->v_w)&&s.is_same(expected->v_ws));}++ordinary;
}
struct CausalDecoderModel {
 virtual ~CausalDecoderModel()=default;
 std::vector<Layer>layer_weights_;int64_t lingbot_prefill_qkv_mode_=0,resolved=0,invalidations=0,seq_len_=224;
 bool qwen3vl_4b_w8_profile_=false,lingbot_prefill_gateup_shared_a_=false; // New MLP owner is exercised separately.
 bool lingbot_prefill_mode_bound_=false,lingbot_folded_rmsnorm_=false,lingbot_prefill_kv_only_=false;
 bool linear_acc32_=false,has_mrope_=true,has_qk_norm_=true,has_qkv_bias_=false,use_silu_=true,adarms_=false;
 bool nvfp4_=false,decoder_mixed_precision_=false,rhinovla_high_precision_=false,qwen3vl_2b_w8_profile_=false,fuse_lm_head_=false;
 bool qwen3vl_pooler_z1_dispatch_=false,qwen3vl_multiview_text_composite_dispatch_=false,wall_z1_dispatch_=false;
 std::vector<int32_t>mrope_section_{24,20,20};std::vector<int64_t>deepstack_lang_layers_{0,1,2};
 RpuRmsNormSpmContract lingbot_nogamma_contract_;CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
 int64_t cores=8;int64_t num_cores()const{return cores;}int64_t num_layers()const{return 36;}
 int64_t hidden_size()const{return 2560;}int64_t intermediate_size()const{return 9728;}
 int64_t num_q_heads()const{return 32;}int64_t num_kv_heads()const{return 8;}int64_t head_dim()const{return 128;}
 bool wall_sorting_route_enabled()const{return false;}int64_t get_last_resolved_chunk_size()const{return resolved;}
 void invalidate_model_state(){++invalidations;}
 Context context;Context&ctx(){return context;}
 __METHODS__
 int64_t layout_identity()const {int64_t h=0;__LAYOUT__ return h;}
 std::vector<int64_t>cost_identity()const {std::vector<int64_t>identity;__COST__ return identity;}
 void rebind_weights(){__GUARD__ layer_weights_.clear();++invalidations;}
 CausalDecoderLinearRoute decoder_projection_route(int64_t)const{return CausalDecoderLinearRoute::AUTO_TILE;}
 FmbPhysicalExecutionManifest manifest(const FmbThreeStageChunkPlan&plan,const LayoutContext&layout,int64_t physical_len,int64_t position){
  FmbPhysicalExecutionManifest manifest;manifest.state=FmbPhysicalManifestState::COMPLETE;
  manifest.logical_length=217;manifest.physical_length=physical_len;
  auto append=[&](FmbRouteFamily f,int64_t s,int64_t r,int64_t flags,std::vector<int64_t>a){manifest.routes.push_back({s,f,r,flags,std::move(a),0});};
  auto append_linear=[&](int64_t s,CausalDecoderLinearRoute r){append(FmbRouteFamily::LINEAR,s,int64_t(r),0,{});};
  __APPEND__ else {__FALLBACK_APPEND__} return manifest;
 }
 uint32_t addr(int,const char*n)const{auto s=std::string(n);return s=="input_norm"?0x1000:s=="q"?0x2000:s=="k"?0x3000:0x4000;}
 uint32_t qkv_fused_bias_addr(int,const char*)const{return 0;}uint32_t nvfp4_scale_addr(const char*)const{return 0;}
 bool qwen3vl_decode_gemv(int64_t)const{return false;}void emit_nvfp4_qkv_bias(int,const char*,const char*,int64_t,int64_t){}
 void consume_causal_route(FmbRouteFamily f,int64_t s,int64_t r){ctx().consume_physical_route(f,s,r,0);}
 void emit_lingbot_prefill_kv_tail(int layer,const ChunkInfo&){assert(layer==35);++tails;}
 void emit(int layer_idx=0){
  const auto&chunk=ctx().stage_plan.compute.chunks.front();const auto&lw=layer_weights_.at(layer_idx);
  const bool is_last_layer=layer_idx==35;int64_t rows=ctx().batch_size*chunk.len,nq=32,nkv=8,hd=128,h=2560,tp=8;
  __TAIL__
  __EMIT__ else {__ORDINARY__}
 }
};
FmbThreeStageChunkPlan plan(int64_t n=224){return compose_fmb_three_stage_chunk_plan({{0,0,n,n}},{{0,0,n,n}},{{0,0,n,n}}, {{0,n,0}},ChunkMode::SEQUENTIAL,{});}
LayoutContext layout(){LayoutContext l;l.is_causal=true;l.batch_size=1;l.attention_policy=AttentionExecutionPolicy::DDR_KV;return l;}
void install(CausalDecoderModel&m,bool w8){auto d=w8?at::kChar:at::kHalf;Layer l;
 l.q_w=at::empty({4096,2560},d);l.k_w=at::empty({1024,2560},d);l.v_w=at::empty({1024,2560},d);
 l.o_w=l.gate_w=l.up_w=l.down_w=at::empty({1},d);
 if(w8){l.q_ws=at::ones({4096},at::kHalf);l.k_ws=at::ones({1024},at::kHalf);l.v_ws=at::ones({1024},at::kHalf);}
 m.layer_weights_.assign(36,l);m.context.stage_plan=plan();}
template<class F>void rejects(F f){auto n=fused+ordinary+tails;bool fail=false;try{f();}catch(const c10::Error&){fail=true;}assert(fail&&n==fused+ordinary+tails);}
void execute(CausalDecoderModel&m,bool want_fused){
 auto l=layout();l.is_causal=m.context.is_causal;l.batch_size=m.context.batch_size;l.use_attn_mask=m.context.attention_mask.has_value();l.attention_policy=m.context.attention_policy;
 auto desc=m.manifest(m.context.stage_plan,l,m.context.seq_len,m.context.position);m.context.consumer.reset(desc);
 assert(desc.routes.size()==(want_fused?1:3));expected=&m.layer_weights_[0];int f=fused,o=ordinary;
 m.seq_len_=m.context.seq_len;m.emit();m.context.consumer.require_all_consumed();assert(fused-f==int(want_fused)&&ordinary-o==(want_fused?0:3));
}
int main(){at::set_num_threads(1);
 int64_t fp16_hash=0;std::vector<int64_t>fp16_cost;
 for(bool w8:{false,true}){
  CausalDecoderModel m;install(m,w8);m.set_lingbot_prefill_mode(true,true);
  assert(m.invalidations==1&&m.lingbot_prefill_qkv_mode_==(w8?1:2));
  auto args=m.lingbot_prefill_qkv_arguments();assert(args==std::vector<int64_t>({1,w8?1:2,224,2560,4096,1024,1024,8,0,w8?320:384,w8?112:96,w8?512:480,w8?48:64,128,w8?11:10}));
  execute(m,true);auto desc=m.context.consumer.manifest();
  for(int layer=1;layer<35;++layer){expected=&m.layer_weights_[layer];m.emit(layer);}m.emit(35);assert(tails==(w8?2:1));
  for(size_t i=0;i<args.size();++i){auto bad=desc;++bad.routes[0].arguments[i];m.context.consumer.reset(bad);rejects([&]{m.emit();});}
  for(int field=0;field<4;++field){auto bad=desc;auto&r=bad.routes[0];if(field==0)r.selector++;if(field==1)r.flags++;if(field==2)r.invocation++;if(field==3)r.family=FmbRouteFamily::ATTENTION;
   m.context.consumer.reset(bad);rejects([&]{m.emit();});}
  // A pre-fusion descriptor cannot silently restore the old three launches.
  auto old=desc;old.routes.clear();for(auto site:{CAUSAL_DECODER_Q_SITE,CAUSAL_DECODER_K_SITE,CAUSAL_DECODER_V_SITE})old.routes.push_back({site,FmbRouteFamily::LINEAR,1,0,{},0});
  m.context.consumer.reset(old);rejects([&]{m.emit();});m.context.consumer.reset(desc);live_payload=false;rejects([&]{m.emit();});live_payload=true;
  auto ptr=m.layer_weights_[0].q_w.unsafeGetTensorImpl();auto hash=m.layout_identity();auto cost=m.cost_identity();
  rejects([&]{m.rebind_weights();});rejects([&]{m.set_lingbot_prefill_mode(true,true);});assert(m.invalidations==1&&ptr==m.layer_weights_[0].q_w.unsafeGetTensorImpl()&&hash==m.layout_identity()&&cost==m.cost_identity());
  if(!w8){fp16_hash=hash;fp16_cost=cost;}else{assert(hash!=fp16_hash&&cost!=fp16_cost);}
  // Mutating cold geometry in the fixture cannot consume a stale fused route.
  m.linear_acc32_=true;m.context.consumer.reset(desc);rejects([&]{m.emit();});execute(m,false);
 }
 for(bool w8:{false,true})for(int kind=0;kind<6;++kind){CausalDecoderModel m;install(m,w8);
  auto&cache=KernelCache::instance();if(kind==0){cache.w8=cache.fp16=false;}if(kind==1)m.linear_acc32_=true;
  if(kind==2)m.layer_weights_.back().q_w=at::empty({4095,2560},w8?at::kChar:at::kHalf);
  if(kind==3)m.layer_weights_.back().v_ws=at::empty({1023},at::kHalf);
  m.set_lingbot_prefill_mode(kind!=4 && kind!=5,kind!=5);
  assert(m.lingbot_prefill_qkv_mode_==0);execute(m,false);cache.w8=cache.fp16=true;
 }
 // Longer legal single chunks use the original separate Q/K/V path; the
 // installed M224 payload must never be reused for a different row count.
 for(bool w8:{false,true})for(int64_t n:{176,320,400}){CausalDecoderModel m;
  install(m,w8);m.set_lingbot_prefill_mode(true,true);
  m.context.seq_len=n;m.context.stage_plan=plan(n);execute(m,false);}
 CausalDecoderModel base;install(base,true);base.set_lingbot_prefill_mode(true,true);
 for(int kind=0;kind<10;++kind){auto m=base;
  if(kind==0){m.context.seq_len=192;m.context.stage_plan=plan(192);}if(kind==1)m.context.batch_size=2;
  if(kind==2)m.context.position=1;if(kind==3)m.context.is_causal=false;if(kind==4)m.context.attention_mask=1;
  if(kind==5)m.context.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
  if(kind==6)m.context.stage_plan.chunk_mode=ChunkMode::KV_FIRST;
  if(kind==7)m.context.stage_plan.qkv.chunks={{0,0,112,112},{1,112,112,224}};
  if(kind==8)m.context.stage_plan.input.plan.chunk_size=112;
  if(kind==9)m.qwen3vl_pooler_z1_dispatch_=true;execute(m,false);
 }
 // Each independent planner stage must carry the same complete single chunk.
 for(int stage=0;stage<3;++stage)for(int bad=0;bad<6;++bad){auto m=base;
  auto*p=stage==0?&m.context.stage_plan.input:stage==1?&m.context.stage_plan.qkv:&m.context.stage_plan.compute;
  if(bad==0)p->chunks={{0,0,112,112},{1,112,112,224}};
  if(bad==1)p->plan.chunk_size=112;if(bad==2)p->chunks[0].idx=1;
  if(bad==3)p->chunks[0].offset=16;if(bad==4)p->chunks[0].len=208;if(bad==5)p->chunks[0].kv_seq_len=240;
  execute(m,false);
 }
 {auto m=base;m.context.seq_len=1;m.context.stage_plan=plan(1);execute(m,false);}
 {CausalDecoderModel m;install(m,true);execute(m,false);}
 {auto m=base;m.context.consumer.reset(FmbPhysicalExecutionManifest{});rejects([&]{m.emit();});}
 {CausalDecoderModel m;install(m,true);m.layer_weights_.back().v_w=at::empty({1024,2560},at::kHalf);
  rejects([&]{m.set_lingbot_prefill_mode(true,true);});assert(!m.lingbot_prefill_mode_bound_&&m.invalidations==0);}
 std::cout<<"actual cold selection, identities, COMPLETE consumers and 35 full-layer / terminal traversal PASS\n";
}
'''
    substitutions = {'CONSTANTS': constants, 'ENUMS': '\n'.join(_function(source, 'enum class ' + n) + ';' for n in ('CausalDecoderPlanningMode', 'CausalDecoderLinearRoute')),
        'METHODS': methods, 'APPEND': append, 'FALLBACK_APPEND': fallback_append, 'TAIL': tail,
        'EMIT': emit, 'ORDINARY': ordinary, 'LAYOUT': layout.replace('detail::layout_mix', 'v3::detail::layout_mix'), 'COST': cost, 'GUARD': guard}
    for key, value in substitutions.items():
        program = program.replace('__' + key + '__', value)
    assert 'PASS' in _run_native(tmp_path, program, torch,
        extra_sources=[ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp'])


def test_actual_kv_only_long_prefix_uses_native_single_chunk_domain(tmp_path):
    source = (ROOT / 'src/fused/rpu_qwen3_model.h').read_text()
    helpers = (ROOT / 'src/core/rpu_helpers.h').read_text()
    norm_decls = (ROOT / 'src/core/rpu_kernel_decls.h').read_text()
    norm = '\n'.join(_function(norm_decls, marker) + ';' for marker in (
        'enum RpuRmsNormCapability', 'enum class RpuRmsNormSpmRoute'))
    norm += '\n' + '\n'.join(re.search(
        rf'inline constexpr [^;]*\b{name}\s*=[^;]+;', norm_decls).group()
        for name in ('RPU_RMSNORM_CAP_ALL', 'RPU_RMSNORM_SCM_U16_MAX',
                     'RPU_RMSNORM_MAX_COLS'))
    norm += '\n' + '\n'.join(_function(norm_decls, marker) for marker in (
        'inline bool rpu_rmsnorm_capability_valid(',
        'inline RpuRmsNormSpmRoute rpu_resolve_rmsnorm_spm_route('))
    methods = '\n'.join(_function(source, marker) for marker in (
        'bool lingbot_prefill_kv_only_shape(', 'bool lingbot_prefill_kv_only_for_plan(',
        'std::vector<FmbPhysicalExecutionManifest> physical_manifest_domain_for_candidate('))
    methods = methods.replace(' const override {', ' const {')
    # Execute the real launcher tiling and legality predicates rather than
    # substituting "valid" for the P288 sync shape which caused this regression.
    sdpa = '\n'.join(_function(helpers, marker) for marker in (
        'inline int64_t sdpa_tile_m_v16_initial(',
        'inline uint32_t GetMhaAttnUnivVlmNeed(',
        'inline std::optional<int64_t> choose_tile_k_conservative(',
        'inline std::optional<int64_t> choose_tile_k(',
        'inline bool sdpa_try_compute_tiling(', 'inline SdpaTiling sdpa_compute_tiling(',
        'inline int64_t sdpa_compute_tmp_v16_size(',
        'static inline bool sdpa_call_valid(', 'inline bool sdpa_is_valid_chunk_size('))
    types = '\n'.join(_function(helpers, marker) + ';' for marker in (
        'struct SdpaTiling', 'enum class SdpaKernelType', 'struct SdpaConfig'))
    constants = re.search(r'inline constexpr int64_t LINGBOT2_PREFILL_KV_TAIL_SITE\s*=\s*[^;]+;', source).group()
    manifest = _function(source, 'FmbPhysicalExecutionManifest physical_manifest_for_candidate(')
    tail_declaration = _function(manifest[manifest.index('if (lingbot_prefill_mode_bound_)'):],
                                 'if (lingbot_prefill_kv_only_)')
    forward = _function(source, 'bool kv_only_entry = false)')
    start = forward.index('        TORCH_CHECK(kv_only_entry == lingbot_prefill_kv_only_')
    stop = forward.index('        TORCH_CHECK(!qwen3vl_multiview_text_dry_prepared_', start)
    forward_guard = forward[start:stop]
    tail = _function(source, 'void emit_lingbot_prefill_kv_tail(')
    # Keep the actual terminal shape guard and COMPLETE route consumption;
    # GEMM/RoPE/KV arithmetic is covered by the retained operator contracts.
    tail = tail[tail.index('        TORCH_CHECK('):tail.index('        consume_causal_route(')]
    program = r'''
#include "core/fused_model_base.h"
#include <ATen/Parallel.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
using namespace v3;
#define CeilDiv(x,y) (((x)+(y)-1)/(y))
#define SDPA_MAX_VLM_SIZE 400
#define RPU_SDPA_TILE_K_CONSERVATIVE_MODE 1
__SDPA_TYPES__
__SDPA__
__NORM__
__CONSTANTS__
enum class CausalDecoderPlanningMode {ORDINARY, OTHER};
struct Context {
 FmbThreeStageChunkPlan stage_plan;FmbPhysicalManifestConsumer consumer;
 bool is_causal=true;int64_t position=0,seq_len=224;
 void consume_physical_route(FmbRouteFamily f,int64_t s,int64_t r,int64_t flags,
       at::IntArrayRef a={},int64_t i=0){consumer.consume_route(f,s,r,flags,a,i);}
};
struct Model {
 bool lingbot_prefill_kv_only_=true;
 CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
 SdpaConfig config{SdpaKernelType::FLASH_ATTN_SPM,128,32,8,8,1,0};
 Context context;std::vector<int>layer_weights_=std::vector<int>(36);
 int declarations=0,forward_admissions=0;
 Context&ctx(){return context;}
 SdpaConfig make_sdpa_config(int mask)const{auto c=config;c.attn_mask_type=mask;return c;}
 __METHODS__
 FmbPhysicalExecutionManifest physical_manifest_for_candidate(const FmbThreeStageChunkPlan&plan,
       const LayoutContext&layout,int64_t physical_len,int64_t logical_len,int64_t position)const{
  FmbPhysicalExecutionManifest manifest;manifest.state=FmbPhysicalManifestState::COMPLETE;
  manifest.logical_length=logical_len;manifest.physical_length=physical_len;
  manifest.kv_logical_length=logical_len;manifest.kv_insert_physical_rows=physical_len;
  auto append=[&](FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,
                  std::vector<int64_t>arguments){manifest.routes.push_back({site,f,selector,flags,std::move(arguments),0});};
  __TAIL_DECLARATION__
  return manifest;
 }
 void forward_boundary(const at::Tensor&hidden_states,bool kv_only_entry=true,
       int64_t position=0,bool is_causal=true,std::optional<int>attention_mask={},
       int64_t batch_slot=0,bool allow_batch_decode=false,int64_t cos_sin_offset=-1){
  __FORWARD_GUARD__
  ++forward_admissions;
 }
 void terminal(int layer_idx,const ChunkInfo&chunk){__TAIL_GUARD_CONSUME__}
};
FmbThreeStageChunkPlan one(int64_t n){return compose_fmb_three_stage_chunk_plan(
 {{0,0,n,n}},{{0,0,n,n}},{{0,0,n,n}},{{0,n,0}},ChunkMode::SEQUENTIAL,{});}
LayoutContext layout(){LayoutContext l;l.is_causal=true;l.batch_size=1;
 l.attention_policy=AttentionExecutionPolicy::DDR_KV;return l;}
template<class F>void rejects(F f){bool failed=false;try{f();}catch(const c10::Error&){failed=true;}assert(failed);}
int main(){at::set_num_threads(1);Model m;auto l=layout();
 // P288's true SDPA sync product is 12: preserving all tokens needs a legal
 // execution bucket, not removal of that operator safety check.
 assert(!sdpa_is_valid_chunk_size(m.config,288,288,0));
 assert(sdpa_is_valid_chunk_size(m.config,144,288,0));
 assert(sdpa_is_valid_chunk_size(m.config,320,320,0));
 // Longer single chunks still require the real GQA/grid synchronization
 // boundary: 384 rows has product 12, while 400 rows has product 16.
 for(int64_t n:{352,368,384})assert(!sdpa_is_valid_chunk_size(m.config,n,n,0));
 assert(sdpa_is_valid_chunk_size(m.config,400,400,0));
 assert(sdpa_compute_tmp_v16_size(m.config,400)*32==262144);
 // Use the installed capability resolver: C400 is V16 for hidden/K norm,
 // while its four Q heads give 1600 rows and keep the V32 math grouping.
 assert(rpu_resolve_rmsnorm_spm_route(RPU_RMSNORM_CAP_ALL,400,2560)==RpuRmsNormSpmRoute::V16_GRID8);
 assert(rpu_resolve_rmsnorm_spm_route(RPU_RMSNORM_CAP_ALL,400,128)==RpuRmsNormSpmRoute::V16_GRID8);
 assert(rpu_resolve_rmsnorm_spm_route(RPU_RMSNORM_CAP_ALL,1600,128)==RpuRmsNormSpmRoute::V32_GRID8);
 assert(rpu_resolve_rmsnorm_spm_route(RPU_RMSNORM_CAP_BASE,400,128)==RpuRmsNormSpmRoute::BASE);
 for(int64_t n=176;n<=448;n+=16){
  const auto tile=sdpa_compute_tiling(m.config,n);
  std::cout<<"sdpa_domain "<<n<<" valid="<<sdpa_is_valid_chunk_size(m.config,n,n,0)
   <<" tm="<<tile.tile_m<<" tn="<<tile.tile_n<<" tk="<<tile.tile_k
   <<" tmp="<<sdpa_compute_tmp_v16_size(m.config,n)*32<<"\n";
 }
 for(auto lengths:{std::pair<int64_t,int64_t>{224,217},{320,278},{400,342},{400,374}}){
  const auto n=lengths.first;auto p=one(n);m.context.seq_len=n;m.context.stage_plan=p;
  auto domain=m.physical_manifest_domain_for_candidate(p,l,n,lengths.second,0);
  assert(domain.size()==1&&domain[0].routes.size()==1);
  assert(domain[0].routes[0].arguments==std::vector<int64_t>({35,n,2560,8,128,8}));
  m.context.consumer.reset(domain[0]);m.forward_boundary(at::empty({1,n,2560},at::kHalf));
  m.terminal(35,p.compute.chunks[0]);m.context.consumer.require_all_consumed();
  auto stale=domain[0];stale.routes[0].arguments[1]=n-16;m.context.consumer.reset(stale);
  rejects([&]{m.terminal(35,p.compute.chunks[0]);});
 }
 assert(m.forward_admissions==4);
 assert(m.physical_manifest_domain_for_candidate(one(288),l,288,278,0).empty());
 rejects([&]{m.forward_boundary(at::empty({1,288,2560},at::kHalf));});
 // The C144x2 shape is legal for ordinary causal SDPA, but this terminal ABI
 // is still one chunk at offset zero. No invented multi-chunk tail is admitted.
 auto multi=compose_fmb_three_stage_chunk_plan(
 {{0,0,144,144},{1,144,144,288}},{{0,0,144,144},{1,144,144,288}},
 {{0,0,144,144},{1,144,144,288}},{{0,288,0}},ChunkMode::SEQUENTIAL,{});
 assert(m.physical_manifest_domain_for_candidate(multi,l,288,278,0).empty());
 for(int64_t rows:{320,400})for(int stage=0;stage<3;++stage)for(int fault=0;fault<6;++fault){auto p=one(rows);
  auto&s=stage==0?p.input:stage==1?p.qkv:p.compute;
  if(fault==0)s.plan.num_chunks=2;if(fault==1)s.plan.chunk_size=rows-16;
  if(fault==2)s.chunks[0].idx=1;if(fault==3)s.chunks[0].offset=16;
  if(fault==4)s.chunks[0].len=rows-16;if(fault==5)s.chunks[0].kv_seq_len=rows+16;
  assert(m.physical_manifest_domain_for_candidate(p,l,rows,rows-16,0).empty());}
 for(int fault=0;fault<5;++fault){auto bad=l;auto p=one(320);int64_t pos=0;
  if(fault==0)bad.batch_size=2;if(fault==1)bad.is_causal=false;
  if(fault==2)bad.use_attn_mask=true;if(fault==3)pos=16;
  if(fault==4)p.chunk_mode=ChunkMode::KV_FIRST;
  assert(m.physical_manifest_domain_for_candidate(p,bad,320,278,pos).empty());}
 m.config.num_q_heads=0;
 assert(m.physical_manifest_domain_for_candidate(one(320),l,320,278,0).empty());
 rejects([&]{m.forward_boundary(at::empty({1,320,2560},at::kHalf));});
 assert(m.forward_admissions==4);
 std::cout<<"native single-chunk SDPA domain and COMPLETE KV-only terminal PASS\n";
}
'''
    for name, value in {'SDPA_TYPES': types, 'SDPA': sdpa, 'NORM': norm, 'CONSTANTS': constants,
        'METHODS': methods, 'TAIL_DECLARATION': tail_declaration,
        'FORWARD_GUARD': forward_guard, 'TAIL_GUARD_CONSUME': tail}.items():
        program = program.replace('__' + name + '__', value)
    output = _run_native(tmp_path, program, torch,
        extra_sources=[ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp'])
    print(output, end='')
    assert 'PASS' in output
