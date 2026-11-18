"""Native memory/ABI and COMPLETE owner regressions for the exact owner49 chain."""
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run
from test_siglip_attention_dry_live_contract import _function, _run_native


def test_owner49_actual_launcher_full_roots_alias_and_registers(tmp_path):
    src = (ROOT/'src/fused/hyvit2_owner_layernorm.h').read_text()
    functions = '\n'.join(block(src, name) for name in (
        'inline void launch_hyvit2_vision_owner_spm(',
        'inline void launch_hyvit2_vision_owner_layernorm_spm(',
        'inline void launch_hyvit2_vision_compact_residual_spm('))
    code = COMMON + r'''
#include <cmath>
#include <cstring>
enum class KernelId {HYVIT2_VISION_OWNER_LAYERNORM_M392N1152,HYVIT2_VISION_COMPACT_RESIDUAL_M392N1152};
enum class GraphKernelNoDdrProof {RingAllReduceSpmResidual};
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;
 uint32_t base=0x20000000;bool ready=true;
 bool is_initialized()const{return ready;}
 uint32_t addr(int c,uint32_t o)const{return base+uint32_t(c)*0x1000000+o;}
} spm;
#define SPM_ALLOC spm
struct Kernel {std::array<uint16_t,4176> regs;void reset_regs(){regs.fill(0xffff);}
 void set_regs(uint32_t i,uint16_t v){assert(i<=66 || (i>=128&&i<=255) || (i>=4096&&i<4176));regs.at(i)=v;}
} kernel;
int fetched=0,staged=0,queued=0;bool available=true;KernelId selected;
Kernel* get_kernel(KernelId id){++fetched;selected=id;return available?&kernel:nullptr;}
#define GET_KERNEL(id) get_kernel(id)
void rpu_set_legacy_scm_u16_checked(Kernel* k,uint32_t i,uint16_t v,const char*){k->set_regs(i,v);}
struct RpuKernelGraph {static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 bool kernel_register_census_active()const{return true;}
 void stage_kernel_no_ddr(KernelId id,GraphKernelNoDdrProof p){assert(id==selected && p==GraphKernelNoDdrProof::RingAllReduceSpmResidual);++staged;}};
struct Queue {bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> g,std::vector<uint8_t> c){assert(broadcast&&g==std::vector<uint16_t>({8,1,1})&&c==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++queued;}} queue;
#define GET_QUEUE(n) (&::queue)
''' + functions + r'''
int main(){
 const uint32_t b=spm.base,F=903168,S=112896;
 const std::array<uint32_t,6> good{b+0x100000,b+0x200000,b+0x300000,b+0x400000,b,b+0x1000};
 auto run=[&](std::array<uint32_t,6> a,bool compact,bool pass,int rows=392,double eps=1e-6){
  int q=queued,f=fetched,s=staged;bool ok=true;
  try{launch_hyvit2_vision_owner_spm(a[0],a[1],a[2],a[3],a[4],a[5],eps,rows,compact);}catch(const std::exception&){ok=false;}
  assert(ok==pass&&queued==q+int(pass)&&fetched==f+int(pass)&&staged==s+int(pass));
 };
 auto pair=[&](int i,int step=1){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+step])<<16);};
 for(bool compact:{false,true}){
  run(good,compact,true);
  assert(selected==(compact?KernelId::HYVIT2_VISION_COMPACT_RESIDUAL_M392N1152:KernelId::HYVIT2_VISION_OWNER_LAYERNORM_M392N1152));
  assert(kernel.regs[0]==8&&kernel.regs[1]==1&&pair(2)==63488&&pair(4)==good[1]-b&&kernel.regs[6]==0);
  assert(kernel.regs[42]==0&&kernel.regs[64]==8&&kernel.regs[65]==1&&kernel.regs[66]==1);
  for(int i=128;i<=255;++i)assert(kernel.regs[i]==0);
  if(!compact){assert(pair(8)==good[2]&&pair(10)==good[3]&&pair(12)==good[4]&&pair(14)==good[5]);
   assert(kernel.regs[16]==49&&kernel.regs[17]==1152&&kernel.regs[20]==c10::Half(1e-6f*1152).x);
   assert(pair(22)==0&&pair(24)==2304&&kernel.regs[26]==1&&kernel.regs[27]==16&&kernel.regs[28]==c10::Half(std::sqrt(1152.f)).x&&kernel.regs[29]==1);
  }
  for(int c=0;c<8;++c){assert(pair(4096+c,8)==spm.addr(c,good[0]-b)-b);
   assert(pair(4112+c,8)==spm.addr(c,good[2]-b)-b);
   assert(pair(4128+c,8)==c*S&&pair(4144+c,8)==S/2);
   assert(pair(4160+c,8)==(compact?0:spm.addr(c,good[3]-b)-b));}
  for(size_t i=0;i<(compact?3:6);++i){
   for(auto addr:{good[i]+1,b-256}){auto a=good;a[i]=addr;run(a,compact,false);}
   for(size_t j=0;j<(compact?3:6);++j)if(i!=j){auto a=good;a[i]=good[j];run(a,compact,!compact&&((i==0&&j==3)||(i==3&&j==0)));}
   auto a=good;const uint32_t bytes=i<4?F:2304;
   a[i]=b+((SpmAllocator::SPM_USABLE-bytes)&~255u);run(a,compact,true);a[i]+=256;run(a,compact,false);
  }
  for(int rows:{0,196,391,393,512,588})run(good,compact,false,rows);
 }
 for(double eps:{0.,1e-5,-1e-6,double(NAN),double(INFINITY)})run(good,false,false,392,eps);
 auto a=good;a[3]=a[0]+256;run(a,false,false);
 spm.ready=false;run(good,false,false);spm.ready=true;
 auto before=fetched;auto q=queued;available=false;bool rejected=false;
 try{launch_hyvit2_vision_owner_spm(good[0],good[1],good[2],good[3],good[4],good[5],1e-6,392,false);}catch(const std::exception&){rejected=true;}
 assert(rejected&&fetched==before+1&&queued==q&&staged==q);
}
'''
    compile_run(tmp_path,'hyvit2_owner49_launcher',code)


def test_owner49_actual_complete_admission_and_consumption(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
    import torch
    src=(ROOT/'src/fused/rpu_hyvit2_vision_model.cpp').read_text()
    methods='\n'.join(_function(src,'    '+name) for name in (
        'bool owner_layernorm_enabled(', 'std::vector<int64_t> owner_layernorm_arguments(',
        'void validate_owner_layernorm_manifest(', 'void consume_owner_layernorm_route('))
    constants='\n'.join(line for line in src.splitlines() if line.startswith('constexpr int64_t HYVIT2_') and any(x in line for x in ('OWNER_LAYERNORM_SITE','COMPACT_RESIDUAL_SITE','ATTN_ALL_REDUCE_SITE','MLP_ALL_REDUCE_SITE','PACKED_RAW_SPM_ATTN_SITE')))
    dry=_function(src,'    FmbPhysicalExecutionManifest physical_manifest_for_candidate(')
    first=dry.index('            if (owner_layernorm_enabled(chunk.len))')
    arm1=_function(dry[first:],'            if (owner_layernorm_enabled(chunk.len))')
    # Include its actual legacy else, not a rewritten mock route.
    end1=first+len(arm1)
    arm1+=_function(dry[end1:],' else {')
    second=dry.index('            if (owner_layernorm_enabled(chunk.len))',end1)
    arm2=_function(dry[second:],'            if (owner_layernorm_enabled(chunk.len))')
    arm2+=_function(dry[second+len(arm2):],' else {')
    entries=[]
    for marker, parameters in (
        ('    at::Tensor forward(', 'const at::Tensor& input'),
        ('    at::Tensor forward_multi(', 'at::TensorList images')):
        entry=_function(src.replace('planned_stage_descriptor = {}','planned_stage_descriptor'),marker)
        start=entry.index('const int64_t patch_rows = validate_patch_inputs(')
        end=entry.index('/*position=*/0);',start)+len('/*position=*/0);')
        entries.append('void '+('enter_batched' if 'input' in parameters else 'enter_list')+
                       '('+parameters+', at::IntArrayRef planned_stage_descriptor) {\n'+entry[start:end]+'\n}')
    code=r'''
#include "core/fused_model_base.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
using namespace v3;
constexpr int NUM_CORES=8;
'''+constants+r'''
struct HostContext {FmbPhysicalManifestConsumer consumer;AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
 int position=0,batch_size=1;bool is_causal=false,input_in_spm=true,output_to_spm=true;
 c10::optional<at::Tensor> attention_mask;FmbThreeStageChunkPlan stage_plan;
 bool has_complete_physical_manifest()const{return consumer.complete();}
 const FmbRouteManifestEntry& consume_physical_route(FmbRouteFamily f,int64_t s,int64_t r,int64_t flags,at::IntArrayRef a,int64_t i)const{return consumer.consume_route(f,s,r,flags,a,i);}};
struct Owner {
 bool cold_owner_layernorm_=true,cold_kv_spm_only_=true,cold_per_image_attention_=false;
 int num_cores()const{return NUM_CORES;}
 struct {bool kvpad16=true;} cold_config_;
 int64_t image_batch_count_=2,orig_head_dim_=72;double eps_=1e-6;
 int layers=27,width=1152;HostContext state;
 struct W {at::Tensor q_w,k_w,v_w,o_w,fc1_w,fc2_w,q_ws,k_ws,v_ws,o_ws,fc1_ws,fc2_ws;};
 std::vector<W> layer_weights_;std::vector<int> layer_bias_norm_=std::vector<int>(27);
 int num_layers()const{return layers;}int hidden_size()const{return width;}int intermediate_size()const{return 4352;}
 int num_q_heads()const{return 16;}int num_kv_heads()const{return 16;}int head_dim()const{return 80;}
 HostContext& ctx(){return state;}
'''+methods+r'''
 // Only preparation/storage are stubs: actual forward prefixes below execute
 // the production order before calling the real manifest and owner validators.
 struct Prepared {FmbPrefillStageCandidate value;const FmbPrefillStageCandidate&candidate()const{return value;}};
 std::shared_ptr<Prepared> prepared=std::make_shared<Prepared>();int admitted=0;
 auto prepare_stage_candidate(at::IntArrayRef){
  auto& routes=prepared->value.physical_manifest.routes;
  std::sort(routes.begin(),routes.end(),[](const auto& l,const auto& r){
   return std::make_tuple(int64_t(l.family),l.site_id,l.invocation)<std::make_tuple(int64_t(r.family),r.site_id,r.invocation);});
  return prepared;
 }
 int64_t validate_patch_inputs(at::TensorList images,bool batched)const{
  TORCH_CHECK(!images.empty()&&(!batched||images.size()==1));int64_t n=0;
  for(const auto& im:images){TORCH_CHECK(im.dim()==4&&im.size(0)>0&&im.size(0)<=8&&
   (batched||im.size(0)==1)&&im.size(1)==3&&im.size(2)==224&&im.size(3)==224);n+=im.size(0);}
  TORCH_CHECK(n<=8);return n*196;
 }
 void begin_external_physical_manifest_prologue(const FmbPhysicalExecutionManifest& m,int64_t rows,int64_t position){
  validate_fmb_physical_manifest(m,rows,position);validate_owner_layernorm_manifest(m);++admitted;
 }
'''+ '\n'.join(entries)+r'''
 FmbPhysicalExecutionManifest dry(int rows=392)const{
  FmbPhysicalExecutionManifest manifest;manifest.state=FmbPhysicalManifestState::COMPLETE;
  manifest.logical_length=manifest.physical_length=manifest.kv_logical_length=rows;
  manifest.kv_insert_physical_rows=cold_config_.kvpad16?((rows+15)/16*16):rows;
  manifest.graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;manifest.linear_accumulation=FmbLinearAccumulationPolicy::ACC16;
  manifest.routes.push_back({HYVIT2_PACKED_RAW_SPM_ATTN_SITE,FmbRouteFamily::ATTENTION,int64_t(AttentionExecutionPolicy::SPM_KV_BY_MHA),0,{image_batch_count_,rows,rows,16,80,72,8},0});
  auto append=[&](FmbRouteFamily f,int64_t s,int64_t r,std::vector<int64_t> a,int64_t flags,int64_t i){manifest.routes.push_back({s,f,r,flags,std::move(a),i});};
  bool raw_spm=true;int64_t physical_len=rows,logical_len=rows,position=0,h=1152,tp=8,invocation=0;
  ChunkInfo chunk{0,0,rows,rows};
'''+arm1+'\n'+arm2+r'''
  return manifest;
 }
};
template<class F>void rejects(F f){bool bad=false;try{f();}catch(const c10::Error&){bad=true;}assert(bad);}
int main(){at::set_num_threads(1);Owner o;
 for(bool fp16:{false,true}){
  auto w=at::ones({1},at::TensorOptions().dtype(fp16?at::kHalf:at::kChar));
  auto s=fp16?at::Tensor():at::ones({1},at::TensorOptions().dtype(at::kHalf));
  o.layer_weights_.assign(27,{w,w,w,w,w,w,s,s,s,s,s,s});
  auto valid=o.dry();assert(valid.routes[1].arguments.size()==20);o.validate_owner_layernorm_manifest(valid);
  auto two=at::empty({2,3,224,224});auto three=at::empty({3,3,224,224});auto image=at::empty({1,3,224,224});
  o.prepared->value.physical_manifest=valid;
  // Cold planning generated M392 under a temporary image_count=2 then
  // restored the initial1. Both production entry prefixes must bind actual
  // cameras before the prologue examines this descriptor.
  for(bool batched:{false,true}){
   o.image_batch_count_=1;const auto before=o.admitted;
   if(batched)o.enter_batched(two,{1});else o.enter_list({image,image},{1});
   assert(o.image_batch_count_==2&&o.admitted==before+1);
   o.image_batch_count_=3;auto legacy=o.dry(588);o.image_batch_count_=2;
   o.prepared->value.physical_manifest=legacy;
   if(batched)o.enter_batched(three,{1});else o.enter_list({image,image,image},{1});
   assert(o.image_batch_count_==3&&o.admitted==before+2);
   o.prepared->value.physical_manifest=valid;
   if(batched)o.enter_batched(two,{1});else o.enter_list({image,image},{1});
   assert(o.image_batch_count_==2&&o.admitted==before+3);
   // Actual three-camera request cannot borrow the two-camera descriptor.
   rejects([&]{if(batched)o.enter_batched(three,{1});else o.enter_list({image,image,image},{1});});
   assert(o.admitted==before+3);
   // Malformed input fails before either shape binding or descriptor admission.
   o.image_batch_count_=2;
   rejects([&]{if(batched)o.enter_batched(at::empty({0,3,224,224}),{1});else o.enter_list({}, {1});});
   assert(o.image_batch_count_==2&&o.admitted==before+3);
   auto bad=valid;++bad.routes[1].arguments[5];o.prepared->value.physical_manifest=bad;
   rejects([&]{if(batched)o.enter_batched(two,{1});else o.enter_list({image,image},{1});});
   assert(o.admitted==before+3);o.prepared->value.physical_manifest=valid;
  }
  o.image_batch_count_=2;
  o.state.stage_plan.compute.chunks={{0,0,392,392}};o.state.stage_plan.qkv.chunks={{0,0,392,392}};
  o.state.consumer.reset(valid);o.consume_owner_layernorm_route(false,{0,0,392,392},0);o.consume_owner_layernorm_route(true,{0,0,392,392},0);
  for(int role:{1,2})for(size_t i=0;i<20;++i){auto bad=valid;++bad.routes[role].arguments[i];rejects([&]{o.validate_owner_layernorm_manifest(bad);});}
  for(int mutation=0;mutation<20;++mutation){auto bad=valid;
   switch(mutation){case 0:bad.state=FmbPhysicalManifestState::UNSPECIFIED;break;
    case 1:bad.logical_length=391;break;case 2:bad.physical_length=400;break;case 3:bad.execution_padding_rows=8;break;
    case 4:bad.kv_logical_length=400;break;case 5:bad.kv_insert_physical_rows=392;break;
    case 6:bad.graph_lifecycle=FmbGraphLifecycle::UNSPECIFIED;break;case 7:bad.linear_accumulation=FmbLinearAccumulationPolicy::MIXED_BY_SITE;break;
    case 8:bad.routes.pop_back();break;case 9:bad.routes.erase(bad.routes.begin()+1);break;
    case 10:bad.routes.push_back(bad.routes[1]);break;case 11:bad.routes[2].selector=2;break;
    case 12:bad.routes[1].family=FmbRouteFamily::NORMALIZATION;break;case 13:bad.routes[1].flags=1;break;
    case 14:bad.routes[2].invocation=1;break;case 15:bad.routes[0].selector=int64_t(AttentionExecutionPolicy::DDR_KV);break;
    case 16:bad.routes[0].arguments[0]=3;break;case 17:bad.routes[1].site_id=HYVIT2_ATTN_ALL_REDUCE_SITE;break;
    case 18:bad.routes.push_back({HYVIT2_MLP_ALL_REDUCE_SITE,FmbRouteFamily::ALL_REDUCE,1,0,{},0});break;
    case 19:bad.routes[2].arguments=bad.routes[1].arguments;break;}
   rejects([&]{o.validate_owner_layernorm_manifest(bad);});}
  for(bool* flag:{&o.cold_owner_layernorm_,&o.cold_kv_spm_only_}){*flag=false;assert(!o.owner_layernorm_enabled(392));rejects([&]{o.validate_owner_layernorm_manifest(valid);});auto legacy=o.dry();o.validate_owner_layernorm_manifest(legacy);assert(legacy.routes[1].site_id==HYVIT2_ATTN_ALL_REDUCE_SITE);*flag=true;}
  o.image_batch_count_=3;assert(!o.owner_layernorm_enabled(588));auto legacy=o.dry(588);o.validate_owner_layernorm_manifest(legacy);assert(legacy.routes[1].site_id==HYVIT2_ATTN_ALL_REDUCE_SITE);o.image_batch_count_=2;
  for(int rows:{0,196,391,393,512,588})assert(!o.owner_layernorm_enabled(rows));
  o.eps_=1e-5;assert(!o.owner_layernorm_enabled(392));rejects([&]{o.validate_owner_layernorm_manifest(valid);});o.eps_=1e-6;
  o.layer_weights_[26].fc2_w=at::ones({1},at::TensorOptions().dtype(fp16?at::kChar:at::kHalf));assert(!o.owner_layernorm_enabled(392));o.layer_weights_[26].fc2_w=w;
  for(int mutation=0;mutation<8;++mutation){o.state.consumer.reset(valid);
   ChunkInfo chunk{0,0,392,392};
   switch(mutation){case 0:o.state.position=16;break;case 1:o.state.batch_size=2;break;
    case 2:o.state.is_causal=true;break;case 3:o.state.input_in_spm=false;break;case 4:o.state.output_to_spm=false;break;
    case 5:chunk.idx=1;break;case 6:chunk.offset=196;break;case 7:chunk.kv_seq_len=400;break;}
   rejects([&]{o.consume_owner_layernorm_route(false,chunk,1);});
   o.state.position=0;o.state.batch_size=1;o.state.is_causal=false;o.state.input_in_spm=o.state.output_to_spm=true;
  }
 }
}
'''
    _run_native(tmp_path,code,torch,extra_sources=(ROOT/"src/core/fmb_three_stage_chunk_plan.cpp",))


def test_owner49_cold_pair_is_atomic_and_bound_once(tmp_path):
    src=(ROOT/'src/fused/rpu_hyvit2_vision_model.cpp').read_text()
    method=block(src,'    void set_runtime_config(')
    code=COMMON+r'''
#include <string>
struct KernelCache {
 bool owner=false,compact=false;
 static KernelCache& instance(){static KernelCache c;return c;}
 void* get_kernel(const char* name)const{
  if(std::string(name)=="hyvit2_vision_owner_reduce_layernorm_m392n1152")return owner?(void*)this:nullptr;
  if(std::string(name)=="hyvit2_vision_xor3_compact_residual_raw_full_m392n1152")return compact?(void*)this:nullptr;
  return nullptr;
 }
 void* get_program(const char*)const{return (void*)this;}
};
struct Owner {
 bool cold_kv_spm_only_=true,cold_per_image_attention_=false,cold_owner_layernorm_=false;
 bool cold_config_bound_=false,production_config_bound_=false;int invalidations=0;
 int num_cores()const{return 8;}
 struct {bool fast_replay,fast_replay_preload,mask_once,kvpad16,q_inplace;}cold_config_;
 void invalidate_model_state(){++invalidations;}
'''+method+r'''
};
int main(){
 for(bool kv:{false,true})for(bool first:{false,true})for(bool second:{false,true}){
  auto& cache=KernelCache::instance();cache.owner=first;cache.compact=second;
  Owner o;o.cold_kv_spm_only_=kv;o.set_runtime_config(true,true,true,true,true);
  assert(o.cold_owner_layernorm_==(kv&&first&&second)&&o.cold_config_bound_&&o.invalidations==1);
  cache.owner=!first;cache.compact=!second;
  assert(o.cold_owner_layernorm_==(kv&&first&&second));
  bool bad=false;try{o.set_runtime_config(false,false,false,false,false);}catch(const std::exception&){bad=true;}
  assert(bad&&o.invalidations==1&&o.cold_config_.fast_replay);
 }
}
'''
    compile_run(tmp_path,'hyvit2_owner49_cold',code)
