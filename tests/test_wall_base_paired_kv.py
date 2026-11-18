"""Execute the Wall H32 paired-KV admission and COMPLETE route with host metadata."""
from pathlib import Path
import shutil
import subprocess

import pytest
from _native_runtime_identity import runtime_identity_include
from pi_wall_host_catalog import CATALOG_INCLUDE, absolute_candidate_includes

ROOT = Path(__file__).resolve().parents[1]
DECODER = (ROOT / 'src/fused/rpu_qwen3_model.h').read_text()
LAUNCHER = (ROOT / 'src/ops/rpu_llama_kvcache.cpp').read_text()


def block(source, marker):
    start = source.index(marker)
    brace = source.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def compile_run(tmp_path, name, source):
    import torch
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    torch_root = Path(torch.__file__).resolve().parent
    cpp = tmp_path / (name + '.cpp')
    if 'RPU_PL_CANDIDATE' in source:
        # Extracted KernelId/name tables depend on the current generated catalog.
        source = CATALOG_INCLUDE + absolute_candidate_includes(source)
    cpp.write_text(source)
    binary = tmp_path / name
    command = [compiler, '-std=c++17', '-O0', '-Wall', '-Wextra',
               f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}',
               '-I' + str(ROOT / 'src'), '-I' + str(ROOT / 'src/ops'), '-I' + str(ROOT / 'src/core'), '-I' + str(runtime_identity_include(tmp_path)), '-I' + str(torch_root / 'include'),
               str(cpp), str(ROOT / 'src/ops/rpu_kvinsert_segment_plan.cpp'),
               '-L' + str(torch_root / 'lib'), '-Wl,-rpath,' + str(torch_root / 'lib'),
               '-lc10', '-o', str(binary)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr


COMMON = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/core/DeviceType.h>
#include "ops/rpu_kvinsert_segment_plan.h"
#include "core/rpu_wall_sorting_config.h"
#undef TORCH_CHECK
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
constexpr int NUM_CORES=8;
namespace at {
enum ScalarType { kHalf,kChar,kFloat,kByte };
struct Device {
 c10::DeviceType kind=c10::DeviceType::PrivateUse1; int index=0;
 c10::DeviceType type() const { return kind; }
 bool operator==(Device b) const { return kind==b.kind && index==b.index; }
};
struct Tensor {
 uintptr_t address=0x10000000; std::vector<int64_t> shape{1,27,1,8,8,16,16};
 ScalarType dtype=kHalf; bool present=true,contiguous=true; Device where;
 bool defined()const{return present;} Device device()const{return where;}
 ScalarType scalar_type()const{return dtype;} bool is_contiguous()const{return contiguous;}
 int64_t dim()const{return shape.size();} int64_t size(size_t i)const{return shape.at(i);}
 size_t nbytes()const{size_t n=2;for(auto d:shape)n*=d;return n;}
 void* data_ptr()const{return reinterpret_cast<void*>(address);}
 template<class T> T* data_ptr()const{return reinterpret_cast<T*>(address);}
};
using IntArrayRef=c10::ArrayRef<int64_t>;
}
auto exact_plan(int64_t p=384,int64_t rows=32,KvInsertRoute route=KvInsertRoute::ALIGNED_V16) {
 return rpu_resolve_kvinsert_segment_plan(p,rows,rows,8,8,128,
  KV_INSERT_CAP_V2|KV_INSERT_CAP_V16,route);
}
'''


def test_actual_launcher_typed_registers_ranges_and_qwen_boundary(tmp_path):
    launch = block(LAUNCHER, 'void rpu_launch_wall_base_paired_kv_spm_kernel(')
    qwen_guard = LAUNCHER.split('void launch_owned_paired_kv_insert(', 1)[1]
    qwen_guard = qwen_guard[qwen_guard.index('    TORCH_CHECK('):qwen_guard.index('    const KernelId kernel_id =')]
    source = COMMON + r'''
enum class KernelId { QWEN3VL_INSERT_KV_CACHE_V16 };
enum class GraphDdrRegisterRole { KeyCache,ValueCache };
enum class GraphDdrRegisterAccess { Write };
enum class GraphDdrRegisterEncoding { DevAddrShift8LoHi };
struct GraphDdrRegisterAbi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;
 GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel {std::array<uint16_t,68> regs{};int writes=0;bool reject=false;
 void reset_regs(){regs.fill(0);writes=0;}
 uint32_t set_regs(uint32_t i,uint16_t x){if(reject)return 3;regs.at(i)=x;++writes;return 0;}
 uint32_t set_regs(uint32_t,int32_t)=delete;
} kernel;
int staged=0,queued=0;
struct Writer {std::vector<GraphDdrRegisterOperandSpec> operands;
 void write(Kernel& k){for(const auto& o:operands){auto a=o.tensor.address>>8;
  assert(k.set_regs(o.abi.lo,uint16_t(a))==0);assert(k.set_regs(o.abi.hi,uint16_t(a>>16))==0);}}
};
struct RpuKernelGraph {
 static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId,std::vector<GraphDdrRegisterOperandSpec> v){
  ++staged;assert(v.size()==2);assert(v[0].abi.role==GraphDdrRegisterRole::KeyCache);
  assert(v[1].abi.role==GraphDdrRegisterRole::ValueCache);
  assert(v[0].abi.lo==8 && v[0].abi.hi==9 && v[1].abi.lo==12 && v[1].abi.hi==13);
  assert(v[0].abi.access==GraphDdrRegisterAccess::Write && v[1].abi.access==GraphDdrRegisterAccess::Write);
  return {v};}
};
struct Queue {bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(broadcast && grid==std::vector<uint16_t>({2,1,1}));
  assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++queued;}
} queue;
#define GET_KERNEL(id) (&::kernel)
#define GET_QUEUE(n) (&::queue)
void rpu_ddr_flush(void*){}
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;
 uint32_t base=0x20000000;bool is_initialized()const{return ready;}
 uint32_t addr(int c,uint32_t off)const{return base+uint32_t(c)*0x1000000u+off;}
} spm;
#define SPM_ALLOC spm
LAUNCH
void qwen_admit(at::Tensor k_cache,at::Tensor v_cache,int64_t position,int64_t seq_len,
 int64_t num_kv_heads=8,int64_t head_dim=128,int num_cores=8,bool decode=false) { const bool lingbot_m224=false; QWEN_GUARD }
int main(){
 at::Tensor k,v;v.address=0x18000000;
 auto run=[&](at::Tensor key,at::Tensor value,uint32_t ko,uint32_t vo,const KvInsertSegmentPlan& plan,
              bool expected,int rows=32){
  const int before=staged,old_queued=queued;bool accepted=true;
  try{rpu_launch_wall_base_paired_kv_spm_kernel(key,value,ko,vo,rows,plan);}
  catch(const std::exception&){accepted=false;}
  assert(accepted==expected);assert(staged==before+int(expected));assert(queued==old_queued+int(expected));
 };
 for(int p:{336,384}){
  const auto plan=exact_plan(p);k.shape[1]=v.shape[1]=(p+32)/16;
  run(k,v,0,8192,plan,true);
  assert(kernel.regs[0]==p && kernel.regs[1]==0 && kernel.regs[2]==1 && kernel.regs[3]==128);
  assert(kernel.regs[6]==0 && kernel.regs[7]==0x2000);
  assert(kernel.regs[10]==8192 && kernel.regs[11]==0x2000 && kernel.regs[21]==8);
  assert(kernel.regs[8]==0 && kernel.regs[9]==0x10 && kernel.regs[12]==0 && kernel.regs[13]==0x18);
  auto bad=k;--bad.shape[1];run(bad,v,0,8192,plan,false);
 }
 const auto plan=exact_plan();
 run(k,v,0,8192,exact_plan(384,32,KvInsertRoute::V2),false);
 run(k,v,0,8192,exact_plan(320),false);run(k,v,0,8192,exact_plan(384,16),false);
 run(k,v,0,8192,plan,false,31);run(k,v,1,8192,plan,false);
 run(k,v,0,8160,plan,false);run(k,v,8192,0,plan,true);
 run(k,v,SpmAllocator::SPM_USABLE-8192,0,plan,true);
 run(k,v,SpmAllocator::SPM_USABLE-8192+32,0,plan,false);
 run(k,v,UINT32_MAX-31,0,plan,false);
 spm.ready=false;run(k,v,0,8192,plan,false);spm.ready=true;
 spm.base=0xf8fff000;run(k,v,0,8192,plan,false);spm.base=0x20000000;
 for(int i:{0,2,3,4,5,6}){auto bad=k;++bad.shape[i];run(bad,v,0,8192,plan,false);}
 {auto bad=k;bad.shape.pop_back();run(bad,v,0,8192,plan,false);}
 {auto bad=k;bad.contiguous=false;run(bad,v,0,8192,plan,false);}
 {auto bad=k;bad.present=false;run(bad,v,0,8192,plan,false);}
 {auto bad=k;bad.dtype=at::kFloat;run(bad,v,0,8192,plan,false);}
 {auto bad=k;bad.where.kind=c10::DeviceType::CPU;run(bad,v,0,8192,plan,false);}
 {auto bad=v;bad.where.index=1;run(k,bad,0,8192,plan,false);}
 {auto bad=v;bad.address=k.address+256;run(k,bad,0,8192,plan,false);}
 {auto bad=v;bad.address=k.address+k.nbytes();run(k,bad,0,8192,plan,true);}
 {auto bad=k;bad.address=v.address+256;run(bad,v,0,8192,plan,false);}
 kernel.reject=true;bool rejected=false;const auto old=queued;
 try{rpu_launch_wall_base_paired_kv_spm_kernel(k,v,0,8192,32,plan);}catch(const std::exception&){rejected=true;}
 assert(rejected && old==queued);kernel.reject=false;
 k.shape[1]=v.shape[1]=38;qwen_admit(k,v,384,192);
 rejected=false;try{qwen_admit(k,v,384,32);}catch(const std::exception&){rejected=true;}assert(rejected);
}
'''
    compile_run(tmp_path, 'paired_launch', source.replace('LAUNCH', launch).replace('QWEN_GUARD', qwen_guard))


def test_complete_pair_schedule_and_fallback(tmp_path):
    helper = '\n'.join(block(DECODER, marker) for marker in (
        '    bool wall_base_paired_kv_profile() const',
        '    bool wall_base_paired_kv_enabled(',
        '    static std::vector<int64_t> wall_base_paired_kv_arguments(',
        '    bool wall_action_fp16_transfer_enabled() const',
        '    static bool wall_action_qkv_weights_compatible(',
        '    static bool wall_action_ring_weights_compatible(',
        '    KvInsertSegmentPlan consume_causal_kv_plan(',
    ))
    producer = block(DECODER, '                    if (wall_base_paired_kv_enabled(')
    emit_start = DECODER.index('        if (wall_base_paired_kv_enabled(', DECODER.index('void build_layer_subgraph'))
    emit = block(DECODER[emit_start:], '        if (wall_base_paired_kv_enabled(')
    tail = DECODER[emit_start + len(emit):]
    emit += ' ' + block(tail, 'else if (!emit_lingbot_prefill_kv_pair(')
    consume = block((ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp').read_text(),
                    'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(')
    consume = consume.replace('FmbPhysicalManifestConsumer::consume_route', 'consume_physical_route')
    cold_start = DECODER.index('        wall_base_paired_kv_ = wall_base_paired_kv_profile()')
    cold = DECODER[cold_start:DECODER.index(';', cold_start) + 1]
    source = COMMON + r'''
enum class ChunkMode {SEQUENTIAL,KV_FIRST};
struct ChunkInfo {int64_t idx=0,offset=0,len=32,kv_seq_len=416;};
struct FmbThreeStageChunkPlan {ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;
 struct{std::vector<ChunkInfo> chunks{{}};}compute;};
enum class FmbRouteFamily {KV_INSERT,GRAPH_SCHEDULE};
struct FmbRouteManifestEntry {int64_t site_id;FmbRouteFamily family;int64_t selector,flags;
 std::vector<int64_t> arguments;int64_t invocation;};
constexpr int64_t CAUSAL_DECODER_WALL_BASE_PAIRED_KV_SITE=1134963394920026477LL;
constexpr int64_t CAUSAL_DECODER_KV_INSERT_SITE=6070783309895644632LL;
struct Manifest {std::vector<FmbRouteManifestEntry> routes;};
struct Context {Manifest manifest_;mutable std::vector<int> consumed_;bool complete=true,is_causal=false;
 int64_t seq_len=32,position=384;std::optional<int> attention_mask{1};FmbThreeStageChunkPlan stage_plan;
 bool has_complete_physical_manifest()const{return complete;}const Manifest& manifest()const{return manifest_;}
 const FmbRouteManifestEntry& find_route(FmbRouteFamily f,int64_t s,int64_t i)const {
  for(const auto& r:manifest_.routes)if(r.family==f&&r.site_id==s&&r.invocation==i)return r;
  throw std::runtime_error("missing site");}
 const auto& find_physical_route(FmbRouteFamily f,int64_t s,int64_t i)const{return find_route(f,s,i);}
 CONSUME
};
struct KernelCache {bool marker=false,program=true;int queries=0;
 static auto& instance(){static KernelCache c;return c;}
 void* get_kernel(const char*){++queries;return marker?this:nullptr;}
 void* get_program(const char*){++queries;return program?this:nullptr;}
};
int paired=0,separate=0;
void rpu_launch_wall_base_paired_kv_spm_kernel(at::Tensor&,at::Tensor&,uint32_t,uint32_t,int64_t,const KvInsertSegmentPlan&){++paired;}
template<class... T> void rpu_launch_insert_kvcache_spm_unified_with_plan(T&&...){++separate;}
struct Model {
 struct LayerWeights {at::Tensor q_w,k_w,v_w,o_w,gate_w,up_w,down_w,input_norm_w,post_norm_w;
  at::Tensor q_ws,k_ws,v_ws,o_ws,gate_ws,up_ws,down_ws;};
 std::vector<LayerWeights> layer_weights_{36};WallSortingConfig wall_sorting_;at::Tensor final_norm_w_;
 bool wall_action_fuse_silu_mul_=true,pre_layers_residual1_ready_=true,has_qkv_bias_=true,has_qk_norm_=false;
 bool has_mrope_=true,use_silu_=true,linear_acc32_=false,nvfp4_=false,decoder_mixed_precision_=false;
 bool wall_base_paired_kv_=false,adarms_=false,wall_action_kv_owner_fanout_=false,partial_mrope_active_=true;
 int64_t cache_batch_slot_=0,kv_insert_spm_rows_=32,seq_len_=32;std::vector<int64_t> deepstack_lang_layers_;
 int nh=16,nkv=8,hd=128,hidden=1024,intermediate=2048,cores=8;
 Context context;Context& ctx(){return context;}const Context& ctx()const{return context;}
 int num_layers()const{return layer_weights_.size();}int hidden_size()const{return hidden;}
 int intermediate_size()const{return intermediate;}int num_q_heads()const{return nh;}
 int num_kv_heads()const{return nkv;}int head_dim()const{return hd;}int num_cores()const{return cores;}
 int attn_tp()const{return cores;}int mlp_tp()const{return cores;}
 struct Offset{uint32_t value;};Offset addr_offset(const char*)const{return {0};}
 KvInsertSegmentPlan restore_kvinsert_plan(int64_t,c10::ArrayRef<int64_t> args,int t,int n,int d)const{
  return rpu_kvinsert_segment_plan_from_route_arguments(args,t,n,d);}
 HELPERS
 void install(bool fp16,bool marker,bool nvfp4=false){
  nvfp4_=nvfp4;
  wall_sorting_.silu_mul=true;wall_sorting_.fp16_transfer_mask=fp16?4:0;
  for(auto& l:layer_weights_){for(auto* w:{&l.q_w,&l.k_w,&l.v_w,&l.o_w,&l.gate_w,&l.up_w,&l.down_w})w->dtype=nvfp4?at::kByte:fp16?at::kHalf:at::kChar;
   for(auto* w:{&l.q_ws,&l.k_ws,&l.v_ws,&l.o_ws,&l.gate_ws,&l.up_ws,&l.down_ws}){w->present=!fp16;w->dtype=nvfp4?at::kByte:at::kHalf;}}
  KernelCache::instance().marker=marker;COLD
 }
 void build(const KvInsertSegmentPlan& kv_plan){
  auto& plan=context.stage_plan;const auto& chunk=plan.compute.chunks.front();
  const int64_t physical_len=seq_len_,logical_len=context.seq_len,position=context.position;
  struct{bool is_causal=false,use_attn_mask=true;int batch_size=1;} layout;
  const bool partial_mrope=partial_mrope_active_;
  auto append=[&](FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t> args,int64_t invocation){
   context.manifest_.routes.push_back({site,f,selector,flags,args,invocation});};
  context.manifest_.routes.clear();auto wire=rpu_kvinsert_route_arguments(kv_plan,8,8,128);
  append(FmbRouteFamily::KV_INSERT,CAUSAL_DECODER_KV_INSERT_SITE,int64_t(kv_plan.route()),0,{wire.begin(),wire.end()},0);
  PRODUCER
  context.consumed_.assign(context.manifest_.routes.size(),0);
 }
 void emit(){
  const auto& chunk=context.stage_plan.compute.chunks.front();
  auto kv_plan=consume_causal_kv_plan(CAUSAL_DECODER_KV_INSERT_SITE,chunk.idx,context.position,context.seq_len);
  at::Tensor k_cache,v_cache;int64_t bs=1,k_spm_stride=8192,nkv=8,hd=128,tp=8,slot0=0,k_cache_batch_stride=0,v_cache_batch_stride=0;
  EMIT
 }
 template<class...T>bool emit_lingbot_prefill_kv_pair(T...){return false;}
};
int main(){
 for(int precision:{0,1,2})for(int p:{336,384})for(bool marker:{false,true}){
  Model m;m.context.position=p;m.context.stage_plan.compute.chunks[0].kv_seq_len=p+32;m.install(precision==1,marker,precision==2);
  assert(m.wall_base_paired_kv_==marker);auto v16=exact_plan(p);auto v2=exact_plan(p,32,KvInsertRoute::V2);
  for(const auto& plan:{v16,v2}){
   const bool pair=marker&&plan.route()==KvInsertRoute::ALIGNED_V16;
   m.build(plan);assert(m.context.manifest_.routes.size()==size_t(pair?2:1));paired=separate=0;m.emit();
   assert(paired==int(pair)&&separate==int(!pair));assert(std::all_of(m.context.consumed_.begin(),m.context.consumed_.end(),[](int n){return n==1;}));
  }
  if(marker){
   m.build(v16);m.context.manifest_.routes.back().arguments[2]++;paired=separate=0;bool failed=false;
   try{m.emit();}catch(const std::exception&){failed=true;}assert(failed&&paired==0&&separate==0);
   m.build(v16);m.context.manifest_.routes.pop_back();failed=false;try{m.emit();}catch(const std::exception&){failed=true;}assert(failed);
   m.build(v16);m.context.complete=false;failed=false;try{m.emit();}catch(const std::exception&){failed=true;}assert(failed);
  }
 }
 Model m;m.install(false,true);auto plan=exact_plan();auto& stage=m.context.stage_plan;
 auto enabled=[&](int p=384,int physical=32,int logical=32,bool causal=false,bool mask=true,int batch=1,bool rope=true){
  return m.wall_base_paired_kv_enabled(stage,plan,physical,logical,p,causal,mask,batch,rope);};
 assert(enabled());assert(!enabled(320));assert(!enabled(384,20));assert(!enabled(384,32,20));
 assert(!enabled(384,32,32,true));assert(!enabled(384,32,32,false,false));assert(!enabled(384,32,32,false,true,2));assert(!enabled(384,32,32,false,true,1,false));
 m.adarms_=true;assert(!enabled());m.adarms_=false;m.cache_batch_slot_=1;assert(!enabled());m.cache_batch_slot_=0;
 m.kv_insert_spm_rows_=0;assert(enabled());m.kv_insert_spm_rows_=32;
 stage.compute.chunks[0].offset=16;assert(!enabled());stage.compute.chunks[0].offset=0;
 stage.compute.chunks[0].kv_seq_len=400;assert(!enabled());stage.compute.chunks[0].kv_seq_len=416;
 stage.compute.chunks.push_back({});assert(!enabled());stage.compute.chunks.pop_back();
 stage.chunk_mode=ChunkMode::KV_FIRST;assert(!enabled());stage.chunk_mode=ChunkMode::SEQUENTIAL;
 m.wall_action_kv_owner_fanout_=true;assert(!enabled());m.wall_action_kv_owner_fanout_=false;
 assert(m.wall_base_paired_kv_profile());m.hidden=2048;assert(!m.wall_base_paired_kv_profile());m.hidden=1024;
 m.wall_sorting_.silu_mul=false;assert(!m.wall_base_paired_kv_profile());m.wall_sorting_.silu_mul=true;
 m.layer_weights_[17].up_w.dtype=at::kHalf;assert(!m.wall_base_paired_kv_profile());
 // NVFP4 must be uniform across every projection and layer. A W8 fallback or
 // byte-packed W4 with FP16 scales must retain the original separate writers.
 Model nv;nv.install(false,true,true);assert(nv.wall_base_paired_kv_profile());
 for(auto member:{&Model::LayerWeights::q_w,&Model::LayerWeights::k_w,
                 &Model::LayerWeights::v_w,&Model::LayerWeights::o_w,
                 &Model::LayerWeights::gate_w,&Model::LayerWeights::up_w,
                 &Model::LayerWeights::down_w}){
  auto& tensor=nv.layer_weights_[17].*member;
  for(auto dtype:{at::kHalf,at::kChar}){tensor.dtype=dtype;assert(!nv.wall_base_paired_kv_profile());}
  tensor.dtype=at::kByte;tensor.present=false;assert(!nv.wall_base_paired_kv_profile());tensor.present=true;
 }
 for(auto member:{&Model::LayerWeights::q_ws,&Model::LayerWeights::k_ws,
                 &Model::LayerWeights::v_ws,&Model::LayerWeights::o_ws,
                 &Model::LayerWeights::gate_ws,&Model::LayerWeights::up_ws,
                 &Model::LayerWeights::down_ws}){
  auto& tensor=nv.layer_weights_[35].*member;tensor.dtype=at::kHalf;
  assert(!nv.wall_base_paired_kv_profile());tensor.dtype=at::kByte;
  tensor.present=false;assert(!nv.wall_base_paired_kv_profile());tensor.present=true;
 }
 for(auto* norm:{&nv.layer_weights_[0].input_norm_w,&nv.layer_weights_[35].post_norm_w,&nv.final_norm_w_}){
  norm->dtype=at::kFloat;assert(!nv.wall_base_paired_kv_profile());norm->dtype=at::kHalf;
 }
 nv.linear_acc32_=true;assert(!nv.wall_base_paired_kv_profile());nv.linear_acc32_=false;
 nv.decoder_mixed_precision_=true;assert(!nv.wall_base_paired_kv_profile());nv.decoder_mixed_precision_=false;
 KernelCache::instance().program=false;nv.install(false,true,true);assert(!nv.wall_base_paired_kv_);
 nv.build(exact_plan());paired=separate=0;nv.emit();assert(paired==0&&separate==1);
 KernelCache::instance().program=true;nv.install(false,true,true);assert(nv.wall_base_paired_kv_);
 nv.nvfp4_=false;assert(!nv.wall_base_paired_kv_profile());
}
'''
    for key, value in [('CONSUME', consume), ('HELPERS', helper), ('COLD', cold),
                       ('PRODUCER', producer), ('EMIT', emit)]:
        source = source.replace(key, value)
    compile_run(tmp_path, 'paired_manifest', source)
