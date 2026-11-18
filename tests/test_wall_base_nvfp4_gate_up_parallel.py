"""Execute real Wall H32 shared/parallel Gate/Up and COMPLETE layer boundaries."""
from pathlib import Path
from test_wall_base_paired_kv import block, compile_run
from test_wall_base_nvfp4_qkv_bias import META

ROOT=Path(__file__).resolve().parents[1]
DECODER=(ROOT/'src/fused/rpu_qwen3_model.h').read_text()


def test_actual_parallel_gate_up_typed_abi_and_ranges(tmp_path):
    linear=(ROOT/'src/ops/rpu_linear.cpp').read_text()
    leaf='\n'.join(block(linear,m) for m in (
        'static void launch_wall_base_nvfp4_gate_up_parallel(',
        'void rpu_launch_wall_base_nvfp4_gate_up_parallel_spm_kernel(',
        'void rpu_launch_wall_base_nvfp4_gate_up_parallel_silu_spm_kernel(',
        'void rpu_launch_wall_base_nvfp4_gate_up_shared_silu_spm_kernel('))
    source=META+r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId{WALL_BASE_NVFP4_GATE_UP_PARALLEL_H32,WALL_BASE_NVFP4_GATE_UP_PARALLEL_SILU_H32,WALL_BASE_NVFP4_GATE_UP_SHARED_SILU_H32};
int mode=0;
KernelId expected_id(){return mode==2?KernelId::WALL_BASE_NVFP4_GATE_UP_SHARED_SILU_H32:
 mode==1?KernelId::WALL_BASE_NVFP4_GATE_UP_PARALLEL_SILU_H32:KernelId::WALL_BASE_NVFP4_GATE_UP_PARALLEL_H32;}
enum class GraphDdrRegisterRole{PackedFp4ModelWeight,Fp8BlockScale};
enum class GraphDdrRegisterAccess{Read};
enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel{std::array<uint16_t,132> regs{};std::vector<int> counter_writes;int fail_reg=-1;void reset_regs(){regs.fill(0xffff);counter_writes.clear();}
 uint32_t set_regs(int r,uint16_t v){if(r==fail_reg)return 5;regs.at(r)=v;return 0;}
 uint32_t set_regs(int r,uint32_t v){if(r==fail_reg)return 5;assert(r%2==0);regs.at(r)=uint16_t(v);regs.at(r+1)=uint16_t(v>>16);if(r>=128)counter_writes.push_back(r);return 0;}
 uint32_t set_regs(int,int)=delete;}kernel;
int staged=0,queued=0;
struct Writer{std::vector<GraphDdrRegisterOperandSpec> operands;void write(Kernel& k){for(auto& o:operands){auto a=o.tensor.address>>8;k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph{static auto& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  assert(id==expected_id()&&v.size()==4);++staged;
  const int lo[]={2,20,32,34};for(int i=0;i<4;++i)assert(v[i].abi.lo==lo[i]&&v[i].abi.hi==lo[i]+1&&v[i].abi.access==GraphDdrRegisterAccess::Read&&v[i].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi&&v[i].abi.role==(i%2==0?GraphDdrRegisterRole::PackedFp4ModelWeight:GraphDdrRegisterRole::Fp8BlockScale));return {v};}};
struct Queue{bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel& k,std::vector<uint16_t> grid,std::vector<uint8_t> cores){assert(broadcast&&grid==std::vector<uint16_t>({4,1,uint16_t(mode==2?1:2)})&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));
  if(mode){assert(k.counter_writes==std::vector<int>({128,130}));for(int r:{128,129,130,131})assert(k.regs[r]==0);}
  else assert(k.counter_writes.empty());
  ++queued;for(int r:{128,129,130,131})k.regs[r]=uint16_t(r+7);}}queue;
#define GET_KERNEL(id) (&::kernel)
#define GET_QUEUE(n) (&::queue)
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}uint32_t addr(int core,int off){return base+core*0x1000000u+off;}}spm;
#define SPM_ALLOC spm
struct KernelCache{bool loaded=true;static auto& instance(){static KernelCache c;return c;}bool has_loaded(KernelId id){assert(id==expected_id());return loaded;}};
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}void rpu_ddr_flush(void*){}
LEAF
uint32_t word(int r){return kernel.regs[r]|(uint32_t(kernel.regs[r+1])<<16);}
template<class F>void rejects(F f){int s=staged,q=queued;bool fail=false;try{f();}catch(const std::exception&){fail=true;}assert(fail&&s==staged&&q==queued);}
int main(){
 for(int candidate:{0,1,2}){mode=candidate;
 at::Tensor g,u,gs,us;g.shape=u.shape={2048,512};gs.shape=us.shape={64,2048};
 g.dtype=u.dtype=gs.dtype=us.dtype=at::kByte;g.address=0x10000000;u.address=0x10200000;gs.address=0x10400000;us.address=0x10500000;
 std::array<uint32_t,5> sp{0x20001000,0x20020000,0x20030000,0x20040000,0x20040100};int64_t count=40;uint16_t layer=34;
 auto run=[&]{auto launch=mode==2?rpu_launch_wall_base_nvfp4_gate_up_shared_silu_spm_kernel:
  mode==1?rpu_launch_wall_base_nvfp4_gate_up_parallel_silu_spm_kernel:rpu_launch_wall_base_nvfp4_gate_up_parallel_spm_kernel;launch(sp[0],g,u,sp[1],sp[2],gs,us,sp[3],sp[4],count,layer);};
 for(int i=0;i<2;++i){run();assert(word(0)==sp[0]&&word(4)==sp[1]&&word(38)==sp[2]&&word(24)==sp[3]&&word(36)==sp[4]);
  assert(word(2)==g.address>>8&&word(20)==gs.address>>8&&word(32)==u.address>>8&&word(34)==us.address>>8);
  assert(kernel.regs[8]==32&&word(10)==256&&kernel.regs[12]==1024&&kernel.regs[13]==16&&kernel.regs[14]==0&&kernel.regs[15]==8&&word(16)==2048&&word(18)==512&&word(22)==4096&&kernel.regs[26]==1&&kernel.regs[27]==34);}
 layer=35;rejects(run);layer=34;count=39;rejects(run);count=40;
 for(int i=0;i<5;++i){auto old=sp[i];sp[i]=0;rejects(run);sp[i]=old+1;rejects(run);sp[i]=0x207fff00;rejects(run);sp[i]=old;}
 for(int i=1;i<5;++i){auto old=sp[i];sp[i]=sp[0]+32;rejects(run);sp[i]=old;}
 for(auto* t:{&g,&u,&gs,&us}){auto old=*t;t->shape[0]--;rejects(run);*t=old;t->dtype=at::kHalf;rejects(run);*t=old;t->storage_off=1;rejects(run);*t=old;t->contiguous=false;rejects(run);*t=old;t->address++;rejects(run);*t=old;t->address=(uint64_t{1}<<40)-256;rejects(run);*t=old;t->where.kind=c10::DeviceType::CPU;rejects(run);*t=old;}
 auto old=u;u.address=g.address;rejects(run);u=old;u.where.index=1;rejects(run);u=old;
 KernelCache::instance().loaded=false;rejects(run);KernelCache::instance().loaded=true;
 spm.ready=false;rejects(run);spm.ready=true;
 if(mode)for(int r:{128,130}){kernel.fail_reg=r;int before=queued;bool failed=false;try{run();}catch(const std::exception&){failed=true;}assert(failed&&queued==before);kernel.fail_reg=-1;}
 }
}
'''
    compile_run(tmp_path,'parallel_gate_up_leaf',source.replace('LEAF',leaf))


def test_actual_parallel_gate_up_complete_profile_and_layers(tmp_path):
    markers=['    bool wall_base_nvfp4_o_ring_norm_enabled(', '    bool wall_base_nvfp4_down_ring_norm_enabled(', '    bool wall_base_nvfp4_gate_up_profile() const', '    bool wall_base_nvfp4_gate_up_enabled(', '    static std::vector<int64_t> wall_base_nvfp4_gate_up_arguments(', '    static std::vector<int64_t> wall_base_nvfp4_gate_up_silu_arguments(', '    static std::vector<int64_t> wall_base_nvfp4_gate_up_shared_silu_arguments(', '    void emit_wall_base_nvfp4_gate_up(']
    helpers='\n'.join(block(DECODER,m) for m in markers)
    start=DECODER.index('        wall_base_nvfp4_gate_up_ =');end=DECODER.index('        wall_base_nvfp4_gate_up_shared_silu_ =',start);cold=DECODER[start:DECODER.index(';',end)+1]
    producer=block(DECODER,'        if (wall_base_nvfp4_gate_up_enabled(plan, physical_len, logical_len,')
    emission=block(DECODER,'else if (wall_base_nvfp4_gate_up && !is_last_layer) {')[5:]
    planner=(ROOT/'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    consume=block(planner,'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(').replace('FmbPhysicalManifestConsumer::consume_route','consume_physical_route').replace('int64_t invocation) const','int64_t invocation = 0) const')
    find=block(planner,'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(').replace('FmbPhysicalManifestConsumer::find_route','find_route')
    source=META+r'''
enum class ChunkMode{SEQUENTIAL,KV_FIRST};
struct ChunkInfo{int64_t idx=0,offset=0,len=32,kv_seq_len=416;};
struct FmbThreeStageChunkPlan{ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;struct{std::vector<ChunkInfo> chunks{{}};}compute;};
enum class FmbRouteFamily{ATTENTION,LINEAR,COLLECTIVE};
constexpr int64_t CAUSAL_DECODER_WALL_BASE_NVFP4_GATE_UP_SITE=6854732981276560913LL,CAUSAL_DECODER_WALL_BASE_NVFP4_GATE_UP_SILU_SITE=3122792047702542432LL,CAUSAL_DECODER_WALL_BASE_NVFP4_GATE_UP_SHARED_SILU_SITE=7052916936419045721LL;
struct FmbRouteManifestEntry{int64_t site_id;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t> arguments;int64_t invocation;};
struct Manifest{std::vector<FmbRouteManifestEntry> routes;};
struct Context{Manifest manifest_;mutable std::vector<int> consumed_;bool complete_flag=true;int64_t position=384;
 bool complete()const{return complete_flag;}const Manifest& manifest()const{return manifest_;}
 FIND
 CONSUME
};
struct KernelCache{bool kernel=true,program=true,fused_kernel=true,fused_program=true,shared_kernel=true,shared_program=true;static auto& instance(){static KernelCache c;return c;}
 void* get_kernel(std::string s){if(s=="wall_base_nvfp4_gate_up_parallel_h32")return kernel?this:nullptr;
  if(s=="wall_base_nvfp4_gate_up_parallel_silu_h32")return fused_kernel?this:nullptr;
  assert(s=="wall_base_nvfp4_gate_up_shared_silu_h32");return shared_kernel?this:nullptr;}
 void* get_program(std::string s){if(s=="wall_base_nvfp4_gate_up_parallel_h32")return program?this:nullptr;
  if(s=="wall_base_nvfp4_gate_up_parallel_silu_h32")return fused_program?this:nullptr;
  assert(s=="wall_base_nvfp4_gate_up_shared_silu_h32");return shared_program?this:nullptr;}};
int calls=0,silu=0,fused_calls=0,shared_calls=0;std::vector<uint16_t> called_layers;
void check_launch(uint32_t x,uint32_t g,uint32_t u,uint32_t gt,uint32_t ut,int64_t count,uint16_t layer){assert(x==1&&g==2&&u==3&&gt==4&&ut==5&&count==40&&layer<35);++calls;called_layers.push_back(layer);}
void rpu_launch_wall_base_nvfp4_gate_up_parallel_spm_kernel(uint32_t x,const at::Tensor&,const at::Tensor&,uint32_t g,uint32_t u,const at::Tensor&,const at::Tensor&,uint32_t gt,uint32_t ut,int64_t count,uint16_t layer){check_launch(x,g,u,gt,ut,count,layer);}
void rpu_launch_wall_base_nvfp4_gate_up_parallel_silu_spm_kernel(uint32_t x,const at::Tensor&,const at::Tensor&,uint32_t g,uint32_t u,const at::Tensor&,const at::Tensor&,uint32_t gt,uint32_t ut,int64_t count,uint16_t layer){check_launch(x,g,u,gt,ut,count,layer);++fused_calls;}
void rpu_launch_wall_base_nvfp4_gate_up_shared_silu_spm_kernel(uint32_t x,const at::Tensor&,const at::Tensor&,uint32_t g,uint32_t u,const at::Tensor&,const at::Tensor&,uint32_t gt,uint32_t ut,int64_t count,uint16_t layer){check_launch(x,g,u,gt,ut,count,layer);++shared_calls;}
void rpu_launch_silu_mul_spm_kernel(uint32_t g,uint32_t u,uint32_t out,int64_t elems,int cores){assert(g==2&&u==3&&out==2&&elems==8192&&cores==8);++silu;}
struct Model{
 struct LayerWeights{at::Tensor gate_w,up_w,gate_ws,up_ws;}lw;
 Context context;Context& ctx(){return context;}FmbThreeStageChunkPlan plan;
 bool wall_base_nvfp4_down_ring_norm_=true,wall_base_nvfp4_o_ring_norm_=true,wall_base_nvfp4_gate_up_=false,wall_base_nvfp4_gate_up_silu_=false,wall_base_nvfp4_gate_up_shared_silu_=false,adarms_=false,decoder_mixed_precision_=false,wall_action_weight_residency_=false,wall_action_kv_owner_fanout_=false;
 WallSortingConfig wall_sorting_;int cache_batch_slot_=0;at::Tensor gate_nvfp4_tensor_scales_,up_nvfp4_tensor_scales_;
 int64_t physical_len=32,logical_len=32;bool causal=false,mask=true,partial=true;int64_t batch=1;
 uint32_t addr(int,std::string s){return s=="residual1"?1:s=="gate"?2:3;}
 uint32_t nvfp4_scale_addr(std::string s){return s=="nvfp4_gate_ts"?4:5;}
 HELPERS
 void bind(){COLD}
 Model(){for(auto* t:{&gate_nvfp4_tensor_scales_,&up_nvfp4_tensor_scales_}){t->dtype=at::kFloat;t->shape={40};}bind();}
 bool enabled(){return wall_base_nvfp4_gate_up_enabled(plan,physical_len,logical_len,context.position,causal,mask,batch,partial);}
 void build(){int64_t position=context.position;bool partial_mrope=partial;
  struct{bool is_causal,use_attn_mask;int64_t batch_size;}layout{causal,mask,batch};context.manifest_.routes.clear();
  auto append=[&](FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t> args,int64_t invocation){context.manifest_.routes.push_back({site,f,selector,flags,args,invocation});};
  PRODUCER
  context.consumed_.assign(context.manifest_.routes.size(),0);
 }
 int old_mlp=0;
 void emit(int layer_idx){auto& chunk=plan.compute.chunks.front();bool wall_base_nvfp4_gate_up=enabled(),is_last_layer=layer_idx==35;
  EMISSION else {++old_mlp;}
 }
};
template<class F>void rejects(F f){int before=calls,act=silu;bool fail=false;try{f();}catch(const std::exception&){fail=true;}assert(fail&&before==calls&&act==silu);}
int main(){
 for(int mode:{0,1,2})for(int64_t position:{336,384,448,512,640}){
 auto& cache=KernelCache::instance();cache={};cache.fused_kernel=cache.fused_program=mode!=0;
 cache.shared_kernel=cache.shared_program=mode!=1;
 calls=silu=fused_calls=shared_calls=0;called_layers.clear();Model m;m.context.position=position;m.plan.compute.chunks[0].kv_seq_len=position+32;
 assert(m.wall_base_nvfp4_gate_up_silu_==(mode!=0)&&m.wall_base_nvfp4_gate_up_shared_silu_==(mode==2));
 m.build();assert(m.enabled()&&m.context.manifest_.routes.size()==1);auto route=m.context.manifest_.routes[0];
 const auto expected=mode==2?std::vector<int64_t>({1,32,position,1,2048,1024,256,8,32,64,128,40,4,1,1,0,35,0,4,2048,0,3}):
  mode==1?std::vector<int64_t>({1,32,position,1,2048,1024,256,8,320,64,128,40,4,1,2,0,35,0,8,1024,0,7}):
  std::vector<int64_t>({1,32,position,1,2048,1024,256,8,320,64,128,40,4,1,2,0,35,1});
 const auto site=mode==2?CAUSAL_DECODER_WALL_BASE_NVFP4_GATE_UP_SHARED_SILU_SITE:
  mode==1?CAUSAL_DECODER_WALL_BASE_NVFP4_GATE_UP_SILU_SITE:CAUSAL_DECODER_WALL_BASE_NVFP4_GATE_UP_SITE;
 assert(route.arguments==expected&&route.site_id==site&&route.selector==(mode?15:11));
 for(int layer=0;layer<36;++layer)m.emit(layer);
 assert(calls==35&&silu==(mode?0:35)&&fused_calls==(mode==1?35:0)&&shared_calls==(mode==2?35:0)&&m.old_mlp==1);
 assert(called_layers.size()==35);for(int layer=0;layer<35;++layer)assert(called_layers[layer]==layer);
 for(int used:m.context.consumed_)assert(used==1);
 for(size_t i=0;i<route.arguments.size();++i){m.context.manifest_.routes[0]=route;++m.context.manifest_.routes[0].arguments[i];rejects([&]{m.emit(0);});}
 for(int field:{0,1,2,3}){m.context.manifest_.routes[0]=route;
  if(field==0)++m.context.manifest_.routes[0].site_id;
  if(field==1)++m.context.manifest_.routes[0].selector;
  if(field==2)++m.context.manifest_.routes[0].flags;
  if(field==3)++m.context.manifest_.routes[0].invocation;
  rejects([&]{m.emit(0);});}
 m.context.manifest_.routes[0]=route;m.context.complete_flag=false;rejects([&]{m.emit(0);});m.context.complete_flag=true;
 m.context.manifest_.routes.clear();rejects([&]{m.emit(0);});m.build();
 m.context.manifest_.routes[0]=route;rejects([&]{m.emit_wall_base_nvfp4_gate_up(m.lw,35,m.plan.compute.chunks[0]);});
 m.context.position=352;rejects([&]{m.emit_wall_base_nvfp4_gate_up(m.lw,0,m.plan.compute.chunks[0]);});
 }
 // Each missing half of the cold marker selects the next installed route.
 // A shared marker cannot bypass a missing fused or raw dependency.
 for(auto member:{&KernelCache::shared_kernel,&KernelCache::shared_program}){
  auto& c=KernelCache::instance();c={};c.*member=false;Model x;
  assert(x.enabled()&&x.wall_base_nvfp4_gate_up_silu_&&!x.wall_base_nvfp4_gate_up_shared_silu_);
  x.build();assert(x.context.manifest_.routes[0].site_id==CAUSAL_DECODER_WALL_BASE_NVFP4_GATE_UP_SILU_SITE);
  int sc=shared_calls,fc=fused_calls,act=silu;x.emit(0);assert(shared_calls==sc&&fused_calls==fc+1&&silu==act);
 }
 for(auto member:{&KernelCache::fused_kernel,&KernelCache::fused_program}){
  auto& c=KernelCache::instance();c={};c.*member=false;Model x;
  assert(x.enabled()&&!x.wall_base_nvfp4_gate_up_silu_&&!x.wall_base_nvfp4_gate_up_shared_silu_);
  x.build();assert(x.context.manifest_.routes[0].site_id==CAUSAL_DECODER_WALL_BASE_NVFP4_GATE_UP_SITE);
  int sc=shared_calls,fc=fused_calls,act=silu;x.emit(0);assert(shared_calls==sc&&fused_calls==fc&&silu==act+1);
 }
 for(auto member:{&KernelCache::kernel,&KernelCache::program}){
  auto& c=KernelCache::instance();c={};c.*member=false;Model x;
  assert(!x.enabled()&&!x.wall_base_nvfp4_gate_up_silu_&&!x.wall_base_nvfp4_gate_up_shared_silu_);
  x.build();assert(x.context.manifest_.routes.empty());int before=calls,act=silu;
  x.emit(0);assert(x.old_mlp==1&&calls==before&&silu==act);
 }
 KernelCache::instance()={};
 for(int i=0;i<13;++i){Model x;switch(i){case 0:x.context.position=352;x.plan.compute.chunks[0].kv_seq_len=384;break;case 1:x.wall_base_nvfp4_down_ring_norm_=false;break;case 2:x.physical_len=31;break;case 3:x.logical_len=31;break;case 4:x.causal=true;break;case 5:x.mask=false;break;case 6:x.partial=false;break;case 7:x.batch=2;break;case 8:x.plan.compute.chunks[0].offset=1;break;case 9:x.plan.chunk_mode=ChunkMode::KV_FIRST;break;case 10:x.wall_base_nvfp4_o_ring_norm_=false;break;case 11:x.cache_batch_slot_=1;break;case 12:x.context.position=336;break;}assert(!x.enabled());x.build();assert(x.context.manifest_.routes.empty());}
 for(int i=0;i<2;++i){Model x;auto& t=i?x.up_nvfp4_tensor_scales_:x.gate_nvfp4_tensor_scales_;t.shape={39};x.bind();assert(!x.enabled());}
}
'''
    for key,value in [('HELPERS',helpers),('COLD',cold),('PRODUCER',producer),('EMISSION',emission),('FIND',find),('CONSUME',consume)]:source=source.replace(key,value)
    compile_run(tmp_path,'parallel_gate_up_route',source)
