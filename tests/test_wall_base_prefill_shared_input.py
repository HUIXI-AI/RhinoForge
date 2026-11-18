"""Exercise the real shared-input W8 leaf and exact Prefill COMPLETE consumer."""
from pathlib import Path
from pi_wall_host_catalog import CATALOG_INCLUDE
from test_wall_base_paired_kv import block, compile_run
from test_wall_base_nvfp4_qkv_bias import META

ROOT=Path(__file__).resolve().parents[1]
D=(ROOT/'src/fused/rpu_qwen3_model.h').read_text()


def test_typed_leaf_ranges_and_packet(tmp_path):
    linear=(ROOT/'src/ops/rpu_linear.cpp').read_text()
    leaf='\n'.join(block(linear,m) for m in (
        'static void launch_wall_base_prefill_gate_up_shared(',
        'void rpu_launch_wall_base_prefill_gate_up_shared_input_spm_kernel(',
        'void rpu_launch_wall_base_prefill_gate_up_shared_silu_spm_kernel(',
        'void rpu_launch_wall_base_prefill_gate_up_shared_silu_no_up_spm_kernel('))
    source=CATALOG_INCLUDE+META+r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId:std::size_t{};
KernelId pi_wall_kernel_id(std::size_t ordinal){return static_cast<KernelId>(ordinal);}
KernelId chosen;
enum class GraphDdrRegisterRole{Int8ModelWeight,PerChannelScale};
enum class GraphDdrRegisterAccess{Read};
enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel{std::array<uint16_t,68> regs{};int fail=-1;void reset_regs(){regs.fill(0xffff);}
 uint32_t set_regs(int r,uint16_t v){if(r==fail)return 7;regs.at(r)=v;return 0;}}kernel;
int staged=0,queued=0;bool fused=false,no_up=false;
struct Writer{std::vector<GraphDdrRegisterOperandSpec> operands;void write(Kernel& k){for(auto& o:operands){auto a=o.tensor.address>>8;k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph{static auto& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  chosen=id;
  assert(v.size()==4);++staged;const int lo[]={2,20,32,34};
  for(int i=0;i<4;++i)assert(v[i].abi.lo==lo[i]&&v[i].abi.hi==lo[i]+1&&v[i].abi.access==GraphDdrRegisterAccess::Read&&v[i].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi&&v[i].abi.role==(i%2?GraphDdrRegisterRole::PerChannelScale:GraphDdrRegisterRole::Int8ModelWeight));return {v};}};
struct Queue{bool broadcast=false;std::vector<uint16_t> grid;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> g,std::vector<uint8_t> cores){assert(broadcast&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));grid=g;++queued;}}queue;
#define GET_KERNEL(id) (&::kernel)
#define GET_QUEUE(n) (&::queue)
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}uint32_t addr(int core,int off){return base+core*0x1000000u+off;}}spm;
#define SPM_ALLOC spm
struct KernelCache{bool loaded=true;static auto& instance(){static KernelCache c;return c;}bool has_loaded(KernelId){return loaded;}};
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}void rpu_ddr_flush(void*){}
LEAF
uint32_t word(int r){return kernel.regs[r]|(uint32_t(kernel.regs[r+1])<<16);}
template<class F>void rejects(F f){int s=staged,q=queued;bool fail=false;try{f();}catch(const std::exception&){fail=true;}assert(fail&&s==staged&&q==queued);}
int main(){
 for(int mode:{0,1,2}){fused=mode>0;no_up=mode==2;
 at::Tensor g,u,gs,us;g.shape=u.shape={11008,2048};gs.shape=us.shape={11008};
 g.dtype=u.dtype=at::kChar;gs.dtype=us.dtype=at::kHalf;
 g.address=0x10000000;u.address=0x14000000;gs.address=0x18000000;us.address=0x18100000;
 std::array<uint32_t,3> sp{0x20000100,0x20200000,0x20400000};int64_t rows=336;
 auto run=[&]{auto launch=no_up?rpu_launch_wall_base_prefill_gate_up_shared_silu_no_up_spm_kernel:fused?rpu_launch_wall_base_prefill_gate_up_shared_silu_spm_kernel:rpu_launch_wall_base_prefill_gate_up_shared_input_spm_kernel;launch(sp[0],g,u,sp[1],sp[2],rows,gs,us);};
 for(auto m:{224,256,320,336,384}){rows=m;if(!no_up&&m<336){rejects(run);continue;}
  const auto operation=no_up?rpu_pi_wall::Operation::SharedSilu:fused?rpu_pi_wall::Operation::SharedSiluUp:rpu_pi_wall::Operation::SharedRaw;
  const auto selected=rpu_pi_wall::select(operation,rpu_pl_tiling::WeightType::Int8,false,rows,1376,2048);
  for(int repeat=0;repeat<2;++repeat){run();
   assert(chosen==pi_wall_kernel_id(selected.ordinal)&&queue.grid==std::vector<uint16_t>({selected.grid_x,selected.grid_y,1}));
   assert(word(0)==sp[0]&&word(4)==sp[1]&&word(36)==sp[2]);
   assert(word(2)==g.address>>8&&word(32)==u.address>>8&&word(20)==gs.address>>8&&word(34)==us.address>>8);
   assert(kernel.regs[8]==1&&kernel.regs[9]==rows&&word(10)==1376&&kernel.regs[12]==2048&&kernel.regs[13]==128);
   assert(word(6)==0&&kernel.regs[14]==0&&kernel.regs[15]==8&&word(16)==4096&&word(18)==2752);
   assert(word(22)==0&&word(24)==0&&kernel.regs[26]==2&&kernel.regs[27]==1&&word(28)==0);
  }
 }
 for(int m:{0,32,208,240,272,304,352,400,448,512,640}){rows=m;rejects(run);}rows=384;
 rejects([&]{launch_wall_base_prefill_gate_up_shared(sp[0],g,u,sp[1],sp[2],rows,gs,us,false,true);});
 for(int i=0;i<3;++i){auto old=sp[i];sp[i]=0;rejects(run);sp[i]=old+32;rejects(run);sp[i]=0x207fff00;rejects(run);sp[i]=old;}
 for(int i=1;i<3;++i){auto old=sp[i];sp[i]=sp[0]+256;rejects(run);sp[i]=old;}
 for(auto* t:{&g,&u,&gs,&us}){auto old=*t;t->shape[0]--;rejects(run);*t=old;t->dtype=at::kByte;rejects(run);*t=old;t->storage_off=1;rejects(run);*t=old;t->contiguous=false;rejects(run);*t=old;t->address++;rejects(run);*t=old;t->address=(uint64_t{1}<<40)-256;rejects(run);*t=old;t->where.kind=c10::DeviceType::CPU;rejects(run);*t=old;}
 auto old=u;u.address=g.address;rejects(run);u=old;u.where.index=1;rejects(run);u=old;
 KernelCache::instance().loaded=false;rejects(run);KernelCache::instance().loaded=true;
 spm.ready=false;rejects(run);spm.ready=true;
 for(int r:{36,37}){kernel.fail=r;int before=queued;bool failed=false;try{run();}catch(const std::exception&){failed=true;}assert(failed&&queued==before);kernel.fail=-1;}
 }
}
'''
    compile_run(tmp_path,'prefill_shared_leaf',source.replace('LEAF',leaf))


def test_complete_profile_fallback_and_consumption(tmp_path):
    helpers='\n'.join(block(D,m) for m in (
        '    bool wall_small_prefill_fusion(',
        '    bool wall_base_prefill_shared_input_enabled(',
        '    int64_t wall_base_prefill_shared_gateup_site(',
        '    std::vector<int64_t> wall_base_prefill_shared_input_arguments(',
        '    void emit_wall_base_prefill_shared_input('))
    start=D.index('        wall_base_prefill_shared_input_ =');end=D.index('        qwen3vl_4b_decode_gateup_fused_ =',start);cold=D[start:end]
    producer=block(D,'else if (wall_base_prefill_shared_input_enabled(')[5:]
    a=D.index('        const bool shared_prefill_silu =');b=D.index('        std::sort(manifest.routes',a)
    activation_producer=D[a:b]
    planner=(ROOT/'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    find=block(planner,'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(').replace('FmbPhysicalManifestConsumer::find_route','find_route')
    consume=block(planner,'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(').replace('FmbPhysicalManifestConsumer::consume_route','consume_physical_route')
    constants='\n'.join(x for x in D.splitlines() if x.startswith('inline constexpr int64_t WALL_BASE_PREFILL_SHARED_'))
    runtime=block((ROOT/'src/ops/rpu_pi_wall_runtime.h').read_text(),
                  'inline bool rpu_pi_wall_operation_loaded(')
    source=CATALOG_INCLUDE+META+r'''
#include <tuple>
enum class ChunkMode{SEQUENTIAL,KV_FIRST};
enum class CausalDecoderPlanningMode{ORDINARY,WALL_Z1};
enum class CausalDecoderLinearRoute:int64_t{AUTO_TILE=1};
enum class KernelId:std::size_t{};
KernelId pi_wall_kernel_id(std::size_t ordinal){return static_cast<KernelId>(ordinal);}
enum class FmbRouteFamily{ATTENTION,LINEAR,ACTIVATION,ALL_REDUCE,COLLECTIVE};
CONSTANTS
constexpr int64_t FMB_SHARED_MLP_SILU_MUL_SITE=91;
struct ChunkInfo{int64_t idx=0,offset=0,len=336,kv_seq_len=336;};
struct FmbThreeStageChunkPlan{ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;struct{struct{int64_t chunk_size=336;}plan;std::vector<ChunkInfo> chunks{{}};}compute;};
struct FmbRouteManifestEntry{int64_t site_id;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t> arguments;int64_t invocation;};
struct Manifest{std::vector<FmbRouteManifestEntry> routes;};
struct Context{Manifest manifest_;mutable std::vector<int> consumed_;bool complete_flag=true,is_causal=true;int64_t position=0,batch_size=1;std::optional<int> attention_mask;FmbThreeStageChunkPlan stage_plan;
 bool complete()const{return complete_flag;}bool has_complete_physical_manifest()const{return complete_flag;}const Manifest& manifest()const{return manifest_;}
 FIND
 CONSUME
};
struct KernelCache{
 bool loaded=true,fused_loaded=false,no_up_loaded=false;
 std::optional<std::size_t> missing;
 static auto& instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId id)const{
  if(missing&&*missing==static_cast<std::size_t>(id))return false;
  for(const auto& p:rpu_pi_wall::profiles)if(p.ordinal==static_cast<std::size_t>(id)){
   switch(p.operation){case rpu_pi_wall::Operation::SharedRaw:return loaded;
    case rpu_pi_wall::Operation::SharedSiluUp:return fused_loaded;
    case rpu_pi_wall::Operation::SharedSilu:return no_up_loaded;default:return true;}
  }return false;
 }
};
RUNTIME
int calls=0,activation=0,down=0,ring=0,fused_calls=0,no_up_calls=0;
bool admitted_rows(int64_t m){return m==224||m==256||m==320||m==336||m==384;}
int64_t causal_ring_route(int64_t,int64_t){return 2;}
void rpu_launch_wall_base_prefill_gate_up_shared_input_spm_kernel(uint32_t x,const at::Tensor&,const at::Tensor&,uint32_t g,uint32_t u,int64_t m,const at::Tensor&,const at::Tensor&){assert(x==1&&g==2&&u==3&&(m==336||m==384));++calls;}
void rpu_launch_wall_base_prefill_gate_up_shared_silu_spm_kernel(uint32_t x,const at::Tensor& g,const at::Tensor& u,uint32_t go,uint32_t up,int64_t m,const at::Tensor& gs,const at::Tensor& us){rpu_launch_wall_base_prefill_gate_up_shared_input_spm_kernel(x,g,u,go,up,m,gs,us);++fused_calls;}
void rpu_launch_wall_base_prefill_gate_up_shared_silu_no_up_spm_kernel(uint32_t x,const at::Tensor& g,const at::Tensor& u,uint32_t go,uint32_t up,int64_t m,const at::Tensor& gs,const at::Tensor& us){assert(x==1&&go==2&&up==3&&admitted_rows(m));++calls;++fused_calls;++no_up_calls;}
void rpu_launch_silu_mul_spm_kernel(uint32_t g,uint32_t u,uint32_t out,int64_t n,int c){assert(g==2&&u==3&&out==2&&(n==336*1376||n==384*1376)&&c==8);++activation;}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t g,const at::Tensor&,uint32_t d,int64_t m,int64_t n,int64_t k,int part,int c,int b,const at::Tensor&){assert(g==2&&d==4&&admitted_rows(m)&&n==2048&&k==11008&&part==0&&c==8&&b==0);++down;}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t d,uint32_t r,uint32_t o,int64_t m,int64_t n,int c,int oc){assert(d==4&&r==5&&o==1&&admitted_rows(m)&&n==2048&&c==8&&oc==8);++ring;}
struct Model{
 struct LayerWeights{at::Tensor q_w,k_w,v_w,o_w,gate_w,up_w,down_w,gate_ws,up_ws,down_ws;};
 std::vector<LayerWeights> layer_weights_=std::vector<LayerWeights>(36);
 WallSortingConfig wall_sorting_;
 bool has_qkv_bias_=true,has_qk_norm_=false,has_mrope_=true,use_silu_=true;
 bool linear_acc32_=false,nvfp4_=false,decoder_mixed_precision_=false;
 bool wall_prefill_mixed_requested_=false,wall_prefill_staged_mlp_=false,wall_prefill_owner_direct_kv_=false,decode_gate_up_fusion_=false,adarms_=false;
 bool pre_layers_residual1_ready_=false,wall_z1_dispatch_=false,qwen3vl_pooler_z1_dispatch_=false,qwen3vl_multiview_text_composite_dispatch_=false;
 bool wall_base_prefill_shared_input_=false,wall_base_prefill_shared_silu_=false,wall_base_prefill_no_up_store_=false,wall_base_prefill_long_qkv_=false,scale=true;
 std::vector<int32_t> mrope_section_={16,24,24};std::vector<int64_t> deepstack_lang_layers_;
 CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
 int64_t hidden=2048,intermediate=11008,nq=16,nkv=8,hd=128,tp=8,seq_len_=336;
 int64_t hidden_size()const{return hidden;}int64_t intermediate_size()const{return intermediate;}
 int64_t num_layers()const{return layer_weights_.size();}int64_t num_q_heads()const{return nq;}int64_t num_kv_heads()const{return nkv;}int64_t head_dim()const{return hd;}
 int attn_tp()const{return tp;}int mlp_tp()const{return tp;}int num_cores()const{return tp;}
 Context context;Context& ctx(){return context;}
 uint32_t addr(int,std::string s){return s=="residual1"?1:s=="gate"?2:s=="up"?3:s=="down"?4:5;}
 HELPERS
 void bind(){int64_t N=num_layers(),hidden_size=hidden,intermediate_size=intermediate,num_q_heads=nq,num_kv_heads=nkv,head_dim=hd;
  bool has_scale=scale,has_nvfp4=nvfp4_,use_silu=use_silu_;
  std::vector<int64_t> mrope_section(mrope_section_.begin(),mrope_section_.end()),deepstack_lang_layers=deepstack_lang_layers_;
  std::vector<at::Tensor> q_w_list,k_w_list,v_w_list,o_w_list,gate_list,up_list,down_list;
  for(auto& lw:layer_weights_){q_w_list.push_back(lw.q_w);k_w_list.push_back(lw.k_w);v_w_list.push_back(lw.v_w);o_w_list.push_back(lw.o_w);gate_list.push_back(lw.gate_w);up_list.push_back(lw.up_w);down_list.push_back(lw.down_w);}
  COLD
 }
 Model(){wall_sorting_.silu_mul=true;for(auto& lw:layer_weights_)for(auto* t:{&lw.q_w,&lw.k_w,&lw.v_w,&lw.o_w,&lw.gate_w,&lw.up_w,&lw.down_w})t->dtype=at::kChar;bind();}
 bool enabled(){return wall_base_prefill_shared_input_enabled(context.stage_plan,seq_len_,context.position,context.is_causal,context.attention_mask.has_value(),context.batch_size);}
 bool wall_base_fp16_shared_gateup_enabled(const FmbThreeStageChunkPlan&,int64_t,int64_t,bool,bool,int64_t)const{return false;}
 bool ordinary_silu_mul_route_enabled()const{return false;}
 void build(){auto& plan=context.stage_plan;int64_t physical_len=seq_len_,position=context.position;
  struct{bool is_causal,use_attn_mask;int64_t batch_size;}layout{context.is_causal,context.attention_mask.has_value(),context.batch_size};
  context.manifest_.routes.clear();
  auto append=[&](FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t> args,int64_t invocation){context.manifest_.routes.push_back({site,f,selector,flags,args,invocation});};
  auto append_ring=[&](int64_t site,int64_t rows,int64_t idx){append(FmbRouteFamily::ALL_REDUCE,site,causal_ring_route(rows,2048),0,{},idx);};
  for(auto& chunk:plan.compute.chunks){PRODUCER}
  bool wall_small_prefill_fused=wall_small_prefill_fusion(plan,physical_len,position,layout.is_causal,layout.use_attn_mask,layout.batch_size);
  const bool wall_fp16_long_prefill_fused=false,qwen_c512=false;
  ACTIVATION_PRODUCER
  std::sort(context.manifest_.routes.begin(),context.manifest_.routes.end(),[](auto& a,auto& b){return std::make_tuple(a.family,a.site_id,a.invocation)<std::make_tuple(b.family,b.site_id,b.invocation);});
  context.consumed_.assign(context.manifest_.routes.size(),0);
 }
 void set_rows(int64_t physical){seq_len_=physical;auto& compute=context.stage_plan.compute;
  if(physical==336||physical==384){compute.plan.chunk_size=physical;compute.chunks={{0,0,physical,physical}};}
  else{const int64_t rows=physical/2;compute.plan.chunk_size=rows;compute.chunks={{0,0,rows,rows},{1,rows,rows,physical}};}}
 void emit(int chunk=0,int layer=0){emit_wall_base_prefill_shared_input(layer_weights_.at(layer),context.stage_plan.compute.chunks.at(chunk));}
};
template<class F>void rejects(F f,int gates=0,int downs=0){int before=calls,old_down=down,old_ring=ring;bool fail=false;try{f();}catch(const std::exception&){fail=true;}assert(fail&&calls==before+gates&&down==old_down+downs&&ring==old_ring);}
int main(){
 // Current catalog preserves three output modes; only no-Up admits long plans.
 for(int mode:{0,1,2})for(int64_t physical:{336,384,448,512,640}){
  bool fused=mode>0,no_up=mode==2,long_prefix=physical>384;
  KernelCache::instance().fused_loaded=fused;KernelCache::instance().no_up_loaded=no_up;
  calls=activation=down=ring=fused_calls=no_up_calls=0;Model x;x.set_rows(physical);x.build();
  if(long_prefix&&!no_up){assert(!x.enabled()&&x.context.manifest_.routes.empty());rejects([&]{x.emit();});continue;}
  const int chunks=long_prefix?2:1;const int64_t rows=physical/chunks;
  assert(x.enabled()&&x.context.manifest_.routes.size()==size_t(chunks*(fused?3:4)));
  if(long_prefix)assert(!x.wall_small_prefill_fusion(x.context.stage_plan,physical,0,true,false,1));
  // Consume each invocation across the actual 36 distinct layer objects.
  for(int c=0;c<chunks;++c)for(int layer=0;layer<36;++layer)x.emit(c,layer);
  assert(calls==36*chunks&&calls==activation+fused_calls&&calls==down&&calls==ring);
  assert(fused_calls==(fused?36*chunks:0)&&no_up_calls==(no_up?36*chunks:0));
  for(int v:x.context.consumed_)assert(v==1);
  auto& routes=x.context.manifest_.routes;
  for(int c=0;c<chunks;++c){
   auto it=std::find_if(routes.begin(),routes.end(),[&](auto& r){return r.site_id==x.wall_base_prefill_shared_gateup_site()&&r.invocation==c;});assert(it!=routes.end());auto valid=*it;
   const auto operation=no_up?rpu_pi_wall::Operation::SharedSilu:fused?rpu_pi_wall::Operation::SharedSiluUp:rpu_pi_wall::Operation::SharedRaw;
   const auto selected=rpu_pi_wall::select(operation,rpu_pl_tiling::WeightType::Int8,false,rows,1376,2048);
   std::vector<int64_t> expected{4,rows,11008,2048,1376,8,selected.tile.m_tile,selected.tile.n_tile,
    selected.tile.k_tile,selected.grid_x,selected.grid_y,1,fused?1:0,no_up?1:2,2,32,20,34,4,36,26,27};
   assert(valid.arguments==expected&&valid.invocation==c&&valid.site_id==(no_up?WALL_BASE_PREFILL_SHARED_NOUP_SITE:fused?WALL_BASE_PREFILL_SHARED_SILU_SITE:WALL_BASE_PREFILL_SHARED_GATEUP_SITE));
   for(size_t i=0;i<valid.arguments.size();++i){*it=valid;++it->arguments[i];rejects([&]{x.emit(c);});}
   for(int field:{0,1,2}){*it=valid;if(field==0)it->selector++;if(field==1)it->flags++;if(field==2)it->invocation=chunks;rejects([&]{x.emit(c);});}*it=valid;
   x.context.complete_flag=false;rejects([&]{x.emit(c);});x.context.complete_flag=true;
   // Later COMPLETE sites must reject before their own launch, even after Gate ran.
   for(int64_t site:{WALL_BASE_PREFILL_SHARED_DOWN_SITE,WALL_BASE_PREFILL_SHARED_REDUCE_SITE}){
    auto tail=std::find_if(routes.begin(),routes.end(),[&](auto& r){return r.site_id==site&&r.invocation==c;});assert(tail!=routes.end());auto original=*tail;
    const int emitted_down=site==WALL_BASE_PREFILL_SHARED_REDUCE_SITE?1:0;
    for(size_t i=0;i<original.arguments.size();++i){*tail=original;++tail->arguments[i];rejects([&]{x.emit(c);},1,emitted_down);}
    *tail=original;tail->arguments.push_back(0);rejects([&]{x.emit(c);},1,emitted_down);
    for(int field:{0,1,2}){*tail=original;if(field==0)tail->selector++;if(field==1)tail->flags++;if(field==2)tail->invocation=chunks;rejects([&]{x.emit(c);},1,emitted_down);}*tail=original;
   }
  }
 }
 // Cold operation availability is hierarchical; no-Up alone cannot widen admission.
 for(bool base:{false,true})for(bool fused:{false,true})for(bool no_up:{false,true}){
  auto& cache=KernelCache::instance();cache.loaded=base;cache.fused_loaded=fused;cache.no_up_loaded=no_up;
  Model x;x.set_rows(448);x.build();const bool eligible=base&&fused&&no_up;
  assert(x.enabled()==eligible&&x.wall_base_prefill_no_up_store_==eligible);
  assert(x.context.manifest_.routes.size()==size_t(eligible?6:0));if(!eligible)rejects([&]{x.emit();});
 }
 auto& cache=KernelCache::instance();cache.loaded=true;cache.fused_loaded=false;cache.no_up_loaded=true;
 Model partial;assert(partial.enabled()&&!partial.wall_base_prefill_shared_silu_&&!partial.wall_base_prefill_no_up_store_);partial.build();assert(partial.context.manifest_.routes.size()==4);
 cache.fused_loaded=true;cache.no_up_loaded=true;
 auto rejected=[](Model& x){x.bind();assert(!x.enabled());x.build();
  // Short-prefix fallback may retain its independent SiLU route; no fused MLP route is legal.
  assert(std::all_of(x.context.manifest_.routes.begin(),x.context.manifest_.routes.end(),[](auto& r){return r.family==FmbRouteFamily::ACTIVATION&&r.site_id==FMB_SHARED_MLP_SILU_MUL_SITE;}));
  if(x.seq_len_>384)assert(x.context.manifest_.routes.empty());rejects([&]{x.emit();});};
 // Retain the pre-existing short-prefix negative boundary coverage.
 for(int i=0;i<12;++i){Model x;switch(i){case 0:x.layer_weights_[35].up_w.dtype=at::kHalf;break;case 1:x.wall_sorting_.silu_mul=false;break;case 2:x.linear_acc32_=true;break;case 3:x.context.position=1;break;case 4:x.context.attention_mask=1;break;case 5:x.context.stage_plan.chunk_mode=ChunkMode::KV_FIRST;break;case 6:x.context.stage_plan.compute.chunks[0].offset=16;break;case 7:x.context.batch_size=2;break;case 8:x.context.is_causal=false;break;case 9:x.scale=false;break;case 10:x.hidden=1024;break;case 11:x.seq_len_=352;break;}
  rejected(x);}
 // Exact two-chunk schedule, start offset, each invocation index and each KV end.
 for(int64_t physical:{448,512,640})for(int bad=0;bad<16;++bad){
  Model x;x.set_rows(physical);auto& compute=x.context.stage_plan.compute;
  switch(bad){
   case 0:x.seq_len_=416;break;case 1:x.seq_len_=576;break;case 2:x.seq_len_=672;break;
   case 3:compute.chunks.pop_back();break;case 4:compute.chunks.push_back(compute.chunks.back());break;
   case 5:compute.plan.chunk_size+=16;break;
   case 6:compute.chunks[0].idx=1;break;case 7:compute.chunks[1].idx=0;break;
   case 8:compute.chunks[0].offset=16;break;case 9:compute.chunks[1].offset+=16;break;
   case 10:compute.chunks[0].len-=16;break;case 11:compute.chunks[1].len+=16;break;
   case 12:compute.chunks[0].kv_seq_len+=16;break;case 13:compute.chunks[1].kv_seq_len-=16;break;
   case 14:x.context.stage_plan.chunk_mode=ChunkMode::KV_FIRST;break;
   case 15:std::swap(compute.chunks[0],compute.chunks[1]);break;
  }rejected(x);
 }
 for(bool Model::* flag:{&Model::linear_acc32_,&Model::nvfp4_,&Model::decoder_mixed_precision_,
     &Model::wall_prefill_mixed_requested_,&Model::wall_prefill_staged_mlp_,&Model::wall_prefill_owner_direct_kv_,
     &Model::decode_gate_up_fusion_,&Model::adarms_,&Model::pre_layers_residual1_ready_,&Model::wall_z1_dispatch_,
     &Model::qwen3vl_pooler_z1_dispatch_,&Model::qwen3vl_multiview_text_composite_dispatch_,&Model::has_qk_norm_}){
  Model x;x.set_rows(448);x.*flag=true;rejected(x);
 }
 for(int64_t Model::* dimension:{&Model::hidden,&Model::intermediate,&Model::nq,&Model::nkv,&Model::hd,&Model::tp}){
  Model x;x.set_rows(448);++(x.*dimension);rejected(x);
 }
 for(int bad=0;bad<14;++bad){Model x;x.set_rows(448);switch(bad){
  case 0:x.context.position=1;break;case 1:x.context.attention_mask=1;break;
  case 2:x.context.batch_size=2;break;case 3:x.context.is_causal=false;break;
  case 4:x.wall_sorting_.silu_mul=false;break;case 5:x.has_mrope_=false;break;
  case 6:x.scale=false;break;case 7:x.use_silu_=false;break;case 8:x.has_qkv_bias_=false;break;
  case 9:x.mrope_section_[1]=16;break;case 10:x.deepstack_lang_layers_.push_back(0);break;
  case 11:x.layer_weights_.pop_back();break;case 12:x.planning_mode_=CausalDecoderPlanningMode::WALL_Z1;break;
  case 13:x.layer_weights_[35].up_w.dtype=at::kHalf;break;
 }rejected(x);}
}

'''
    for key,value in [('HELPERS',helpers),('COLD',cold),('ACTIVATION_PRODUCER',activation_producer),('PRODUCER',producer),('CONSTANTS',constants),('RUNTIME',runtime),('FIND',find),('CONSUME',consume)]:
        source=source.replace(key,value)
    compile_run(tmp_path,'prefill_shared_route',source)
