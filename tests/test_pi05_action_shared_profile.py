"""Execute Action P544/P576/P608/P640/P672/P800/P832/P864/P896/P928, four/ten-step and FP16 admission."""
from pathlib import Path
import re
from test_siglip_attention_dry_live_contract import _function, _run_native

ROOT=Path(__file__).resolve().parents[1]


def test_action_real_prefix_fp16_owner_and_payload_admission(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
    import torch
    source=(ROOT/'src/fused/rpu_pi05_denoise_step_model.cpp').read_text()
    header = (ROOT / "src/fused/rpu_pi05_denoise_step_model.h").read_text()
    policy = _function(header, "    FmbLinearAccumulationPolicy linear_accumulation_policy(")
    free=('bool pi05_action_prefix_admitted(', 'bool pi05_is_fp16_action(',
          'bool pi05_fp16_projection_owners_admitted(', 'bool pi05_action_rows_admitted(',
          'int64_t pi05_kv_insert_rows(', 'bool pi05_q_rope_tables_admitted(',
          'int64_t pi05_geglu_m50_selector(', 'bool pi05_graph_dma_shape_admitted(',
          'bool pi05_kv1_pair_owner_w8_double_buffered(',
          'bool pi05_kv1_direct_cache_overlapped(',
          'bool pi05_kv1_direct_cache_rotated(',
          'bool pi05_kv1_direct_cache_q_rope_pack(',
          'std::vector<int64_t> pi05_kv1_pair_owner_policy_arguments(',
          'std::vector<int64_t> pi05_kv1_pair_owner_linear_arguments(',
          'bool pi05_action_steps_admitted(')
    helpers=[]
    for name in free:
        value=_function(source,name)
        if name in free[1:3]:value='template <typename Layers>\n'+value
        helpers.append(value)
    methods=[]
    for name,kind in (('use_spm_adarms_table','bool'),('use_pi05_graph_dma_profile','bool'),
                      ('use_pi05_ring_xor3','bool'),('use_pi05_xor3_gated_attn','bool'),
                      ('use_pi05_kv1_direct_cache','bool'),
                      ('use_pi05_geglu_m50','bool'),('pi05_geglu_m50_arguments','std::vector<int64_t>'),
                      ('validate_pi05_geglu_m50_manifest','void')):
        method=_function(source,kind+' Pi05DenoiseStepModel::'+name+'(')
        methods.append(method.replace('Pi05DenoiseStepModel::',''))
    constants='\n'.join(re.findall(r'constexpr int64_t PI05_DENOISE_(?:GEGLU_M50_\w+|GEGLU_FP16_\w+)\s*=.*?;',source,re.S))
    body='\n'.join(helpers+methods).replace('at::kPrivateUse1','at::kCPU')
    # Substitute only the CPU device and payload boundary, retaining all actual
    # dtype/owner/shape/route logic. The projection caller gates use production
    # owner checks and are independently invalidated below.
    code=r'''
#include "core/fused_model_base.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
#include <algorithm>
using namespace v3;
enum class KernelId : uint16_t {
 PI05_DENOISE_KV1_PAIR_OWNER_FP16_M50N32X2K1024,
 PI05_DENOISE_KV1_PAIR_OWNER_W8A16_M50N32X2K1024,
 PI05_DENOISE_KV1_DIRECT_CACHE_M50D256P64,
 PI05_DENOISE_GATE_UP_GEGLU_FP16_M608N32K128,
 PI05_DENOISE_GATE_UP_GEGLU_W8A16_M512N48K128,
 PI05_DENOISE_KV1_DIRECT_CACHE_M10D256P16,
 PI05_RING_XOR3_M10N1024,
 PI05_XOR3_GATED_RESIDUAL_M10N1024,
 PI05_KV_OWNER_W8_DB,
 PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_OVERLAP,
 PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT,
 PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64
};
struct KernelCache {
 bool kv=true,geglu=true,h10=true,kvdb=false,kvov=false,rot=false,rpack=false;
 static KernelCache& instance(){static KernelCache k;return k;}
 bool has_loaded(KernelId id)const {
   if(id==KernelId::PI05_KV_OWNER_W8_DB)return kvdb;
   if(id==KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_OVERLAP)return kvov;
   if(id==KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT)return rot;
   if(id==KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64)return rpack;
   if(id==KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M10D256P16||id==KernelId::PI05_RING_XOR3_M10N1024||
      id==KernelId::PI05_XOR3_GATED_RESIDUAL_M10N1024)return h10;
   return id==pi_wall_kernel_id(rpu_pi_kv_linear::pair(true).ordinal)?kv:geglu;
 }
};
// The rotated split-KV readers and writer load together and serve chunks of
// more than 16 rows (rpu_sdpa.cpp).
bool pi05_split_kv_rotated(int64_t rows){return rows>16&&KernelCache::instance().has_loaded(KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT);}
// The query RoPE pack serves chunks whose 4 * ceil(T / 16) blocks cover the
// eight cores, up to 127 rows (rpu_sdpa.cpp); its payload is loaded here.
bool pi05_split_kv_rope_pack(int64_t rows){return rows>0&&rows<=127&&4*((rows+15)/16)>=8;}
''' + constants + '\n'+'\n'.join(helpers).replace('at::kPrivateUse1','at::kCPU')+r'''
struct LayerWeights {
 at::Tensor q_w,k_w,v_w,o_w,gate_proj_w,up_proj_w,down_proj_w;
 at::Tensor q_ws,k_ws,v_ws,o_ws,gate_ws,up_ws,down_ws,k_kv1_w,v_kv1_w,k_kv1_ws,v_kv1_ws;
};
struct Model {
 bool linear_acc32_=false;
 struct { bool enabled()const{return false;} } nvfp4_;
__POLICY__
 int64_t planned_prefix_len_=800,rope_position_=-1,planned_num_steps_=10,chunk_size_=50;
 int64_t planned_kv_physical_rows_=64,planned_k_rope_cache_capacity_=2048,max_action_dim_=32;
 bool planned_route_profile_valid_=true,planned_loop_mode_=true;
 bool pi05_kv1_direct_cache_opt_in_=true,pi05_kv1_pair_owner_opt_in_=true;
 bool pi05_kv1_pair_owner_full_owners_=true,pi05_kv1_direct_pair_owners_=false;
 bool pi05_geglu_m50_opt_in_=true;
 bool pi05_ring_xor3_opt_in_=true,pi05_xor3_gated_attn_opt_in_=true,pi05_k_rope_insert_opt_in_=true;
 bool pi05_gateup_resident_opt_in_=false,pi05_adarms_body_stream_opt_in_=false;
 bool pi05_elide_unused_dma_opt_in_=false,pi05_spm_action_staging_opt_in_=false;
 bool pi05_q_rope_epilogue_opt_in_=false,pi05_w4_fastpath_opt_in_=false;
 int64_t pi05_gateup_resident_layers_=0;
 at::Tensor adarms_table_=at::empty({10,37,3072},at::kHalf);
 at::Tensor cos_=at::empty({2048,128},at::kHalf),sin_=at::empty({2048,128},at::kHalf);
 std::vector<LayerWeights> layer_weights_;
 int hidden_size()const{return 1024;} int intermediate_size()const{return 4096;}
 int num_layers()const{return 18;} int num_cores()const{return 8;} int attn_tp()const{return 8;}
 int num_q_heads()const{return 8;} int num_kv_heads()const{return 8;} int head_dim()const{return 256;}
 bool pi05_projection_fastpath_admitted()const{return pi05_fp16_projection_owners_admitted(layer_weights_);}
 bool use_pi05_kv1_pair_owner(int64_t rows)const{return pi05_kv1_pair_owner_opt_in_&&use_pi05_kv1_direct_cache(rows);}
 Model(){
   LayerWeights w;
   w.q_w=w.k_w=w.v_w=at::empty({2048,1024},at::kHalf);
   w.o_w=at::empty({1024,2048},at::kHalf);
   w.gate_proj_w=w.up_proj_w=at::empty({4096,1024},at::kHalf);
   w.down_proj_w=at::empty({1024,4096},at::kHalf);
   layer_weights_.assign(18,w);
 }
'''+'\n'.join(methods).replace('at::kPrivateUse1','at::kCPU')+r'''
};
template<class F>void rejects(F f){bool caught=false;try{f();}catch(const c10::Error&){caught=true;}assert(caught);}
int main(){
 at::set_num_threads(1);Model owner;
 auto make=[&](){
  FmbPhysicalExecutionManifest m;m.state=FmbPhysicalManifestState::COMPLETE;
  m.logical_length=m.physical_length=50;m.execution_padding_rows=0;
  m.kv_logical_length=owner.planned_prefix_len_+50;m.kv_insert_physical_rows=64;
  m.graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
  m.linear_accumulation=FmbLinearAccumulationPolicy::ACC16;
  m.routes={{PI05_DENOISE_GEGLU_M50_POLICY_SITE,FmbRouteFamily::GRAPH_SCHEDULE,1,0,owner.pi05_geglu_m50_arguments(),0},
            {PI05_DENOISE_GEGLU_M50_LINEAR_SITE,FmbRouteFamily::LINEAR,PI05_DENOISE_GEGLU_FP16_M50_M608N32K128,0,owner.pi05_geglu_m50_arguments(),0}};
  return m;
 };
 for(int64_t steps:{4,10})for(int64_t prefix:{544,576,608,640,672,800,832,864,896,928}){
  owner.planned_prefix_len_=prefix;owner.planned_num_steps_=steps;
  owner.adarms_table_=at::empty({steps,37,3072},at::kHalf);
  assert(owner.use_spm_adarms_table()&&owner.use_pi05_kv1_direct_cache(50)&&owner.use_pi05_geglu_m50(50));
  assert(owner.use_pi05_ring_xor3(50)&&owner.use_pi05_xor3_gated_attn(50));
  auto m=make();owner.validate_pi05_geglu_m50_manifest(m);
  auto kv=pi05_kv1_pair_owner_policy_arguments(true,prefix-13,prefix,true,steps,50);
  assert(kv[0]==7&&kv[2]==steps&&kv[3]==prefix&&kv[6]==prefix+50&&kv[22]==0&&kv[33]==prefix-13);
  auto linear=pi05_kv1_pair_owner_linear_arguments(prefix,true,steps,50);
  assert(linear[0]==6&&linear[1]==steps&&linear[2]==prefix&&linear[13]==0);
  assert(owner.pi05_geglu_m50_arguments()[2]==steps);
  if(steps==10){
   assert(kv==pi05_kv1_pair_owner_policy_arguments(true,prefix-13,prefix,true,10,50));
   assert(linear==pi05_kv1_pair_owner_linear_arguments(prefix,true,10,50));
  }
  auto wrong_schedule=m;
  wrong_schedule.routes.back().arguments[2]=(steps==4?10:4);
  rejects([&]{owner.validate_pi05_geglu_m50_manifest(wrong_schedule);});
  auto table=owner.adarms_table_;
  // Optimized FP16 consumes a prepared AdaRMS table just like W8. Selecting
  // its XOR3 profile while skipping precomputation must fail closed before
  // graph preparation; the ordinary no-table route has different cold flags.
  owner.adarms_table_=at::Tensor();
  assert(!owner.use_spm_adarms_table()&&!owner.use_pi05_graph_dma_profile(50));
  assert(!owner.use_pi05_ring_xor3(50)&&!owner.use_pi05_xor3_gated_attn(50));
  rejects([&]{owner.validate_pi05_geglu_m50_manifest(m);});
  owner.adarms_table_=at::empty({steps==4?10:4,37,3072},at::kHalf);
  assert(!owner.use_spm_adarms_table());
  rejects([&]{owner.validate_pi05_geglu_m50_manifest(m);});
  owner.adarms_table_=table;
  FmbThreeStageChunkPlan plan;plan.chunk_mode=ChunkMode::SEQUENTIAL;
  plan.spans={{0,50}};plan.input.chunks=plan.qkv.chunks=plan.compute.chunks={{0,0,50,prefix+50}};
  assert(pi05_graph_dma_shape_admitted(plan,50,50,prefix,1,true,false,AttentionExecutionPolicy::DDR_KV));
  plan.compute.chunks[0].kv_seq_len=prefix+64;
  assert(!pi05_graph_dma_shape_admitted(plan,50,50,prefix,1,true,false,AttentionExecutionPolicy::DDR_KV));
  for(int mutation=0;mutation<14;++mutation){auto bad=m;
   switch(mutation){
    case 0:bad.kv_logical_length=prefix+64;break;
    case 1:bad.kv_logical_length=(prefix==544?850:594);break;
    case 2:bad.physical_length=64;break;
    case 3:bad.execution_padding_rows=14;break;
    case 4:bad.kv_insert_physical_rows=50;break;
    case 5:bad.linear_accumulation=FmbLinearAccumulationPolicy::MIXED_BY_SITE;break;
    case 6:bad.routes.erase(bad.routes.begin());break;
    case 7:bad.routes.pop_back();break;
    case 8:bad.routes.push_back(bad.routes.front());break;
    case 9:bad.routes.push_back(bad.routes.back());break;
    case 10:bad.routes.back().selector=PI05_DENOISE_GEGLU_M50_M512N48K128;break;
    case 11:bad.routes.back().arguments[0]=1;break;
    case 12:bad.routes.back().arguments[3]=(prefix==544?800:544);break;
    case 13:bad.routes.back().invocation=1;break;
   }rejects([&]{owner.validate_pi05_geglu_m50_manifest(bad);});
  }
  auto& cache=KernelCache::instance();cache.kv=false;assert(!owner.use_pi05_kv1_direct_cache(50));rejects([&]{owner.validate_pi05_geglu_m50_manifest(m);});cache.kv=true;
  cache.geglu=false;rejects([&]{owner.validate_pi05_geglu_m50_manifest(m);});cache.geglu=true;
  owner.pi05_geglu_m50_opt_in_=false;rejects([&]{owner.validate_pi05_geglu_m50_manifest(m);});owner.pi05_geglu_m50_opt_in_=true;
 }
 // H10: M10 rows over a P16 physical KV insert, FP16 owners and the three
 // M10 payloads only; everything else keeps the H50 profile.
 for(int64_t prefix:{544,800}){
  owner.planned_prefix_len_=prefix;owner.planned_num_steps_=10;
  owner.adarms_table_=at::empty({10,37,3072},at::kHalf);
  owner.chunk_size_=10;owner.planned_kv_physical_rows_=16;
  assert(owner.use_spm_adarms_table()&&owner.use_pi05_graph_dma_profile(10));
  assert(owner.use_pi05_ring_xor3(10)&&owner.use_pi05_xor3_gated_attn(10));
  assert(owner.use_pi05_kv1_direct_cache(10)&&owner.use_pi05_geglu_m50(10));
  assert(!owner.use_pi05_graph_dma_profile(50)&&!owner.use_pi05_kv1_direct_cache(50));
  auto m=make();m.logical_length=m.physical_length=10;
  m.kv_logical_length=prefix+10;m.kv_insert_physical_rows=16;
  owner.validate_pi05_geglu_m50_manifest(m);
  auto args=owner.pi05_geglu_m50_arguments();
  assert(args[4]==10&&args[5]==16&&args[6]==prefix+10&&args[34]==10*2048&&args[35]==10*1024);
  for(int mutation=0;mutation<3;++mutation){auto bad=m;
   if(mutation==0)bad.kv_insert_physical_rows=64;
   if(mutation==1)bad.physical_length=50;
   if(mutation==2)bad.kv_logical_length=prefix+50;
   rejects([&]{owner.validate_pi05_geglu_m50_manifest(bad);});
  }
  auto kv=pi05_kv1_pair_owner_policy_arguments(true,prefix-13,prefix,true,10,10);
  assert(kv[4]==10&&kv[5]==16&&kv[6]==prefix+10&&kv[29]==1&&kv[33]==prefix-13);
  assert(kv[19]==static_cast<int64_t>(KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M10D256P16));
  assert(pi05_kv1_pair_owner_policy_arguments(true,prefix-13,prefix,true,10,50)[29]==4);
  assert(pi05_kv1_pair_owner_linear_arguments(prefix,true,10,10)[3]==10);
  // A W8 owner plans and dispatches the double-buffered sibling whenever the
  // cold REF loads it, at both horizons; FP16 never takes it.
  for(int64_t rows:{10,50}){
   auto& kc=KernelCache::instance();
   const auto pair=[&](bool fp16){return pi05_kv1_pair_owner_policy_arguments(true,prefix-13,prefix,fp16,10,rows);};
   // Otherwise both owners keep the catalog's selected pair payload.
   assert(pair(false)[18]==static_cast<int64_t>(pi_wall_kernel_id(rpu_pi_kv_linear::pair(false).ordinal)));
   kc.kvdb=true;
   assert(pair(false)[18]==static_cast<int64_t>(KernelId::PI05_KV_OWNER_W8_DB));
   assert(pair(false)[0]==6&&pair(false)[4]==rows);
   assert(pair(true)[18]==static_cast<int64_t>(pi_wall_kernel_id(rpu_pi_kv_linear::pair(true).ordinal)));
   assert(!pi05_kv1_pair_owner_w8_double_buffered(65));
   kc.kvdb=false;
   // Either owner's consumer is the load-overlapped direct cache whenever the
   // cold REF loads it; otherwise the published M10/M50 payload.
   const auto published=static_cast<int64_t>(rows==10?KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M10D256P16
                                                      :KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M50D256P64);
   assert(pair(false)[19]==published&&pair(true)[19]==published);
   kc.kvov=true;
   const auto overlap=static_cast<int64_t>(KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_OVERLAP);
   assert(pair(false)[19]==overlap&&pair(true)[19]==overlap);
   assert(pair(false)[29]==(rows==10?1:4));
   assert(!pi05_kv1_direct_cache_overlapped(65));
   // The slot-rotated writer, loaded with the rotated readers, takes precedence
   // above 16 rows; M10 keeps the overlapped writer.
   kc.rot=true;
   const auto rot=static_cast<int64_t>(rows>16?KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT
                                              :KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_OVERLAP);
   assert(pair(false)[19]==rot&&pair(true)[19]==rot);
   assert(!pi05_kv1_direct_cache_rotated(65));
   // The rotated writer fused with the query RoPE pack takes precedence where
   // both admit the chunk and the cold REF loads it. At most 16 rows it needs
   // neither: it runs with mask 0 in place of the overlapped writer.
   kc.rpack=true;
   const auto rpack=static_cast<int64_t>(KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64);
   assert(pair(false)[19]==rpack&&pair(true)[19]==rpack);
   assert(pi05_kv1_direct_cache_q_rope_pack(17)&&pi05_kv1_direct_cache_q_rope_pack(16));
   assert(pi05_kv1_direct_cache_q_rope_pack(1)&&!pi05_kv1_direct_cache_q_rope_pack(0));
   assert(!pi05_kv1_direct_cache_q_rope_pack(65));
   kc.rot=false;
   assert(!pi05_kv1_direct_cache_q_rope_pack(50)&&pi05_kv1_direct_cache_q_rope_pack(10));
   kc.rpack=false;
   kc.kvov=false;
  }
  FmbThreeStageChunkPlan plan;plan.chunk_mode=ChunkMode::SEQUENTIAL;
  plan.spans={{0,10}};plan.input.chunks=plan.qkv.chunks=plan.compute.chunks={{0,0,10,prefix+10}};
  assert(pi05_graph_dma_shape_admitted(plan,10,10,prefix,1,true,false,AttentionExecutionPolicy::DDR_KV));
  assert(!pi05_graph_dma_shape_admitted(plan,10,16,prefix,1,true,false,AttentionExecutionPolicy::DDR_KV));
  // A missing M10 payload, a P64 insert or a W8 owner fails closed.
  auto& cache=KernelCache::instance();cache.h10=false;
  assert(!owner.use_pi05_ring_xor3(10)&&!owner.use_pi05_xor3_gated_attn(10));
  assert(!owner.use_pi05_kv1_direct_cache(10));cache.h10=true;
  owner.planned_kv_physical_rows_=64;assert(!owner.use_pi05_kv1_direct_cache(10));
  owner.planned_kv_physical_rows_=16;
  auto saved=owner.layer_weights_[7];
  owner.layer_weights_[7].gate_proj_w=at::empty({4096,1024},at::kChar);
  assert(!owner.use_pi05_ring_xor3(10)&&!owner.use_pi05_kv1_direct_cache(10)&&!owner.use_pi05_geglu_m50(10));
  owner.layer_weights_[7]=saved;
  owner.chunk_size_=16;assert(!owner.use_spm_adarms_table());
  owner.chunk_size_=50;owner.planned_kv_physical_rows_=64;
 }
 for(int64_t prefix:{0,543,560,575,577,607,609,799,801,831,833,863,865,895,897}){owner.planned_prefix_len_=prefix;assert(!owner.use_spm_adarms_table());assert(!owner.use_pi05_kv1_direct_cache(50));}
 owner.planned_prefix_len_=544;
 for(int64_t steps:{0,1,3,5,9,11}){
  owner.planned_num_steps_=steps;
  owner.adarms_table_=at::empty({steps,37,3072},at::kHalf);
  assert(!pi05_action_steps_admitted(steps));
  assert(!owner.use_spm_adarms_table()&&!owner.use_pi05_kv1_direct_cache(50));
 }
 owner.planned_num_steps_=10;owner.adarms_table_=at::empty({10,37,3072},at::kHalf);
 auto saved=owner.layer_weights_[7];
 owner.layer_weights_[7].down_ws=at::empty({1024},at::kHalf);assert(!pi05_fp16_projection_owners_admitted(owner.layer_weights_));
 owner.layer_weights_[7]=saved;owner.layer_weights_[7].gate_proj_w=at::empty({4096,1024},at::kChar);assert(!pi05_fp16_projection_owners_admitted(owner.layer_weights_));
 owner.layer_weights_[7]=saved;owner.layer_weights_[7].k_w=at::empty({256,1024},at::kHalf);assert(!pi05_fp16_projection_owners_admitted(owner.layer_weights_));
 owner.layer_weights_[7]=saved;owner.layer_weights_[7].q_w=at::empty({2049,1024},at::kHalf).slice(0,1);assert(!pi05_fp16_projection_owners_admitted(owner.layer_weights_));
 owner.layer_weights_[7]=saved;assert(pi05_fp16_projection_owners_admitted(owner.layer_weights_));
 owner.planned_kv_physical_rows_=50;assert(!owner.use_pi05_kv1_direct_cache(50));owner.planned_kv_physical_rows_=64;
 owner.pi05_kv1_pair_owner_opt_in_=false;owner.pi05_kv1_direct_pair_owners_=true;assert(!owner.use_pi05_kv1_direct_cache(50));
}
'''
    code = code.replace("__POLICY__", policy)
    _run_native(tmp_path,code,torch)


def test_action_pair_cold_validator_uses_real_fp16_and_w8_inverse_swizzles(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
    import torch
    source=(ROOT/'src/ops/rpu_linear.cpp').read_text()
    swizzle=_function(source,'at::Tensor tp_col_swizzle_mc_weight(')
    validator=_function(source,'void rpu_validate_pi05_kv1_pair_owner_w8a16(').replace(
        'c10::DeviceType::PrivateUse1','c10::DeviceType::CPU')
    code=r'''
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
#include <cstring>
#include <map>
struct RpuKernelGraph {
 static bool active;
 static bool has_active(){return active;}
};
bool RpuKernelGraph::active=false;
uint64_t RpuGetDevAddr(const void* p){
 static std::map<const void*,uint64_t> pointers;
 auto it=pointers.find(p);if(it!=pointers.end())return it->second;
 uint64_t address=0x82000000ULL+pointers.size()*0x1000000ULL;
 pointers.emplace(p,address);return address;
}
'''+swizzle+'\n'+validator+r'''
template<class F>void rejects(F f){bool caught=false;try{f();}catch(const c10::Error&){caught=true;}assert(caught);}
int main(){
 at::set_num_threads(1);
 for(bool half:{false,true}){
  auto dtype=half?at::kHalf:at::kChar;
  auto logical=at::randint(-10,11,{256,1024},at::kInt).to(dtype);
  auto k=tp_col_swizzle_mc_weight(logical.repeat({8,1}),8,half?2:1);
  auto v=tp_col_swizzle_mc_weight((logical+1).repeat({8,1}),8,half?2:1);
  at::Tensor ks,vs;
  if(!half){ks=at::ones({2048},at::kHalf);vs=at::ones({2048},at::kHalf);}
  rpu_validate_pi05_kv1_pair_owner_w8a16(k,ks,v,vs);
  auto broken=logical.repeat({8,1});broken.index({700,341}).fill_(33);
  broken=tp_col_swizzle_mc_weight(broken,8,half?2:1);
  rejects([&]{rpu_validate_pi05_kv1_pair_owner_w8a16(broken,ks,v,vs);});
  rejects([&]{rpu_validate_pi05_kv1_pair_owner_w8a16(k,ks,k,ks);});
  RpuKernelGraph::active=true;
  rejects([&]{rpu_validate_pi05_kv1_pair_owner_w8a16(k,ks,v,vs);});
  RpuKernelGraph::active=false;
  if(half){
   rejects([&]{rpu_validate_pi05_kv1_pair_owner_w8a16(k,at::ones({2048},at::kHalf),v,vs);});
   rejects([&]{rpu_validate_pi05_kv1_pair_owner_w8a16(k,ks,v.to(at::kChar),vs);});
  }else{
   auto badscale=ks.clone();badscale.index({800}).fill_(2);
   rejects([&]{rpu_validate_pi05_kv1_pair_owner_w8a16(k,badscale,v,vs);});
   rejects([&]{rpu_validate_pi05_kv1_pair_owner_w8a16(k,at::Tensor(),v,vs);});
  }
 }
}
'''
    _run_native(tmp_path,code,torch)


def test_cond_dma_consumption_uses_current_alias_only_when_selected(tmp_path, monkeypatch):
    """Actual conditional DMA block and mutable replay, independent of publication.

    The whole-forward fixture checks publication. Here DDR already contains the
    current rows, and the real consumer must select the right rows and mode.
    """
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    source = (ROOT / "src/fused/rpu_pi05_denoise_step_model.cpp").read_text()
    body = _function(source, "void Pi05DenoiseStepModel::emit_pre_layers_body(")
    consumer = _function(body, "    if (!loop_mode_ || !adarms_table_.defined())")
    code = r'''
#include <ATen/ATen.h>
#include <cassert>
#include <functional>
#include <map>
#include <vector>
std::map<void*,std::vector<c10::Half>> device;
std::vector<std::vector<c10::Half>> reads;
std::vector<std::function<void()>> nodes;
bool recording=false,replaying=false;
void rpu_launch_ddr_scatter_spm_dma_mutable(const uint64_t* base,int64_t offset,
    int64_t elements,int64_t stride,int64_t,int cores){
 auto run=[=]{auto& src=device.at(reinterpret_cast<void*>(*base));std::vector<c10::Half> out;
   for(int c=0;c<cores;++c)for(int64_t i=0;i<elements;++i)out.push_back(src.at((offset+c*stride)/2+i));
   reads.push_back(out);};
 if(recording)nodes.push_back(run);else if(!replaying)run();
}
enum class FmbRouteFamily {MUTABLE_DMA};
enum class Pi05DenoiseMutableDmaRoute {DDR_SCATTER_TO_SPM};
constexpr int PI05_DENOISE_PRE_COND_DMA_SITE=1,DWIDTH=2;
struct Context {
 int64_t body_iter=0;
 bool has_complete_physical_manifest() const{return false;}
 template<class... T>void consume_physical_route(T...){}
};
struct Model {
 bool loop_mode_=true;
 at::Tensor adarms_table_;
 uint64_t cond_all_src_base_=0,cond_src_base_=0;
 Context context;
 Context& ctx(){return context;}
 int condition_tp() const{return 8;}
 int addr(int,const char*) const{return 0;}
 void consume(){const int64_t h=16;
__CONSUMER__
 }
};
int main(){
 for(bool loop:{false,true})for(bool table:{false,true}){
  Model m;m.loop_mode_=loop;if(table)m.adarms_table_=at::ones({1},at::kHalf);
  auto backing=at::zeros({5,16},at::kHalf),cond=backing.narrow(0,1,3);
  device.clear();device[cond.data_ptr()]=std::vector<c10::Half>(cond.numel());
  m.cond_all_src_base_=m.cond_src_base_=reinterpret_cast<uint64_t>(cond.data_ptr());
  const bool consumes=!loop||!table;
  auto write=[&](int value){auto& bytes=device.at(cond.data_ptr());
   for(size_t i=0;i<bytes.size();++i)bytes[i]=c10::Half(float(value+int(i/16)));};
  auto check=[&](int value){
   assert(reads.size()==(consumes?2:0));
   for(size_t i=0;i<reads.size();++i)for(c10::Half x:reads[i])
    assert(float(x)==value+(loop?int(i):0));
  };
  reads.clear();write(7);
  m.context.body_iter=0;m.consume();m.context.body_iter=1;m.consume();check(7);
  nodes.clear();recording=true;replaying=false;
  m.context.body_iter=0;m.consume();m.context.body_iter=1;m.consume();
  recording=false;replaying=true;
  assert(nodes.size()==(consumes?2:0));
  for(int value:{7,-3,7}){
   reads.clear();write(value);
   m.context.body_iter=0;m.consume();m.context.body_iter=1;m.consume();
   for(auto& run:nodes)run();check(value);
  }
  replaying=false;
 }
}
'''
    _run_native(tmp_path, code.replace("__CONSUMER__", consumer), torch)
