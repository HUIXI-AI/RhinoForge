"""Execute actual Wall H32 NVFP4 O admission, COMPLETE routing and leaf ABI."""
from pathlib import Path
import re
import pytest

from test_wall_base_paired_kv import block, compile_run
from test_wall_base_nvfp4_qkv_bias import META

ROOT = Path(__file__).resolve().parents[1]
DECODER = (ROOT / 'src/fused/rpu_qwen3_model.h').read_text()
LINEAR = (ROOT / 'src/ops/rpu_linear.cpp').read_text()


@pytest.mark.parametrize("projection", ["o", "down"])
def test_actual_linear_launcher_ranges_alias_scm_and_counter_packet(tmp_path, projection):
    launch = block(LINEAR, 'static void launch_wall_base_nvfp4_linear_ring_norm(') + '\n' + block(
        LINEAR, f'void rpu_launch_wall_base_nvfp4_{projection}_ring_norm_spm_kernel(')
    optional = block((ROOT / 'src/core/rpu_kernel_cache.inc').read_text(), 'static bool is_optional_kernel(')
    scm_helper = block((ROOT / 'src/core/rpu_helpers.h').read_text(), 'inline void rpu_set_legacy_scm_u16_checked(')
    source = META + r'''
#include "ops/rpu_linear_tiling.h"
#include <cmath>
#include <limits>
#include <unordered_set>
OPTIONAL
enum class KernelId {WALL_BASE_NVFP4_O_RING_NORM_H32};
enum class GraphDdrRegisterRole {PackedFp4ModelWeight,Fp8BlockScale};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;
 GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel {std::array<uint16_t,4352> regs{};std::vector<int> counter_writes;int reject=-1;
 void reset_regs(){regs.fill(0xffff);counter_writes.clear();}
 uint32_t set_regs(uint32_t r,uint16_t v){if(int(r)==reject)return 5;regs.at(r)=v;return 0;}
 uint32_t set_regs(uint32_t r,uint32_t v){if(int(r)==reject)return 5;assert(r%2==0&&r<4096);
  regs.at(r)=uint16_t(v);regs.at(r+1)=uint16_t(v>>16);if(r>=128)counter_writes.push_back(r);return 0;}
 uint32_t set_regs(uint32_t,int32_t)=delete;
} kernel;
namespace rhino_lkn {using Kernel_t=Kernel;constexpr int kKernelRegOk=0;}
constexpr int SCM_REG_OFFSET=4096;
SCM_HELPER
struct KernelCache {bool loaded=true;static auto& instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId)const{return loaded;}};
int staged=0,queued=0;
struct Writer {std::vector<GraphDdrRegisterOperandSpec> operands;
 void write(Kernel& k){for(const auto& o:operands){auto p=o.tensor.address>>8;
  assert(k.set_regs(o.abi.lo,uint16_t(p))==0);assert(k.set_regs(o.abi.hi,uint16_t(p>>16))==0);}}
};
struct RpuKernelGraph {static auto& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  assert(id==KernelId::WALL_BASE_NVFP4_O_RING_NORM_H32);++staged;assert(v.size()==2);
  assert(v[0].abi.role==GraphDdrRegisterRole::PackedFp4ModelWeight&&v[0].abi.lo==2&&v[0].abi.hi==3);
  assert(v[1].abi.role==GraphDdrRegisterRole::Fp8BlockScale&&v[1].abi.lo==20&&v[1].abi.hi==21);
  for(auto& o:v)assert(o.abi.access==GraphDdrRegisterAccess::Read&&o.abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
  return {v};}
};
struct Queue {bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel& k,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(broadcast&&grid==std::vector<uint16_t>({8,1,1})&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));
  assert(k.counter_writes.size()==64);for(int i=0;i<64;++i){int r=128+2*i;
   assert(k.counter_writes[i]==r&&k.regs[r]==0&&k.regs[r+1]==0);k.regs[r]=19;k.regs[r+1]=3;}
  ++queued;}
} queue;
#define GET_KERNEL(id) (&::kernel)
#define GET_QUEUE(n) (&::queue)
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}uint32_t addr(int core,uint32_t off)const{return base+core*0x1000000u+off;}
} spm;
#define SPM_ALLOC spm
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}
void rpu_ddr_flush(void*){}
LAUNCH
uint32_t u32(int r){return kernel.regs[r]|(uint32_t(kernel.regs[r+1])<<16);}
// The real SDK stores even SCM halfwords high and odd low. sldc's legacy
// index selects the opposite halfword, so raw set_regs would swap core c^1.
uint16_t sldc(int i){return kernel.regs[(4096+i)^1];}
uint32_t scm_pair(int field,int core){return sldc(field+core)|(uint32_t(sldc(field+8+core))<<16);}
int main(){
 assert(is_optional_kernel("wall_base_nvfp4_o_ring_norm_h32"));
 assert(!is_optional_kernel("llama_insert_kv_cache_multiwarp_v16"));
 constexpr uint32_t x=0x20000000,p=x+0x4000,r=x+0x14000,raw=x+0x24000,g=x+0x34000,t=g+2048;
 at::Tensor w,s;w.dtype=s.dtype=at::kByte;w.shape={1024,1024};s.shape={128,1024};s.address=0x18000000;
 using Slabs=std::array<uint32_t,7>;const Slabs exact{x,p,r,raw,r,g,t};
 auto run=[&](Slabs a,bool expected,at::Tensor weight,at::Tensor scale,int elems=40,int layer=35,double eps=1e-6){
  int before=staged,old=queued;bool accepted=true;
  try{rpu_launch_wall_base_nvfp4_o_ring_norm_spm_kernel(a[0],weight,a[1],a[2],a[3],a[4],a[5],scale,a[6],elems,layer,eps);}
  catch(const std::exception&){accepted=false;}
  assert(accepted==expected&&staged==before+int(expected)&&queued==old+int(expected));
 };
 for(int repeat=0;repeat<2;++repeat){
  run(exact,true,w,s);assert(u32(0)==x&&u32(2)==w.address/256&&u32(4)==p&&u32(6)==0);
  assert(kernel.regs[8]==32&&u32(10)==1024&&kernel.regs[12]==256&&kernel.regs[13]==4);
  assert(kernel.regs[14]==0&&kernel.regs[15]==8&&u32(16)==512&&u32(18)==2048);
  assert(u32(20)==s.address/256&&u32(22)==16384&&u32(24)==t&&kernel.regs[26]==0&&kernel.regs[27]==35);
  assert(kernel.regs[32]==8&&kernel.regs[33]==1&&u32(34)==63488&&u32(36)==r-x&&kernel.regs[38]==0);
  assert(kernel.regs[40]==1024&&kernel.regs[41]==c10::Half(1.0f/1024).x&&kernel.regs[42]==2048);
  assert(u32(44)==raw&&u32(46)==r&&u32(48)==g&&kernel.regs[50]==c10::Half(1e-6).x);
  for(int c=0;c<8;++c){assert(scm_pair(0,c)==uint32_t(c*0x1000000u+p-x));
   assert(scm_pair(16,c)==uint32_t(c*0x1000000u+raw-x));assert(scm_pair(32,c)==uint32_t(c*8192));
   assert(scm_pair(48,c)==4096);}
 }
 auto a=exact;a[4]=x+0x38000;run(a,true,w,s);
 for(int i=0;i<7;++i){a=exact;a[i]++;run(a,false,w,s);a=exact;a[i]=0;run(a,false,w,s);}
 for(int i=0;i<7;++i)for(int j=0;j<i;++j){if(i==4&&j==2)continue;
  a=exact;a[i]=a[j];run(a,false,w,s);}
 a=exact;a[4]=r+32;run(a,false,w,s);a=exact;a[6]=x+SpmAllocator::SPM_USABLE-160;run(a,true,w,s);
 a[6]+=32;run(a,false,w,s);spm.ready=false;run(exact,false,w,s);spm.ready=true;
 spm.base=0xf8fff000;for(int i=0;i<7;++i)a[i]=exact[i]-x+spm.base;run(a,false,w,s);spm.base=x;
 for(int n:{36,48})run(exact,false,w,s,n);run(exact,false,w,s,40,36);
 for(double eps:{0.0,-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})run(exact,false,w,s,40,35,eps);
 for(int which:{0,1}){
  auto bad=which?s:w;auto reject=[&](at::Tensor v){run(exact,false,which?w:v,which?v:s);};
  bad.dtype=at::kHalf;reject(bad);bad=which?s:w;bad.present=false;reject(bad);bad=which?s:w;bad.contiguous=false;reject(bad);
  bad=which?s:w;bad.storage_off=1;reject(bad);bad=which?s:w;bad.shape[0]++;reject(bad);
  bad=which?s:w;bad.where.kind=c10::DeviceType::CPU;reject(bad);bad=which?s:w;bad.address++;reject(bad);
  bad=which?s:w;bad.address=uint64_t{1}<<40;reject(bad);
 }
 auto bad=s;bad.where.index=1;run(exact,false,w,bad);bad=s;bad.address=w.address+256;run(exact,false,w,bad);
 KernelCache::instance().loaded=false;run(exact,false,w,s);KernelCache::instance().loaded=true;
 for(int reg:{32,64,128,132,254,4096}){kernel.reject=reg;int before=queued;bool rejected=false;
  try{rpu_launch_wall_base_nvfp4_o_ring_norm_spm_kernel(x,w,p,r,raw,r,g,s,t,40,35,1e-6);}catch(const std::exception&){rejected=true;}
  assert(rejected&&queued==before);kernel.reject=-1;}
}
'''
    if projection == 'down':
        source = source.replace('WALL_BASE_NVFP4_O_RING_NORM_H32', 'WALL_BASE_NVFP4_DOWN_RING_NORM_H32')
        source = source.replace('wall_base_nvfp4_o_ring_norm', 'wall_base_nvfp4_down_ring_norm')
        source = re.sub(r'\b35\b', '34', source)
        source = source.replace('run(exact,false,w,s,40,36)', 'run(exact,false,w,s,40,35)')
    for key, value in [('LAUNCH', launch), ('OPTIONAL', optional), ('SCM_HELPER', scm_helper)]:
        source = source.replace(key, value)
    compile_run(tmp_path, 'wall_' + projection + '_launch', source)


def test_actual_o_cold_profile_complete_manifest_and_fallback(tmp_path):
    helpers = '\n'.join(block(DECODER, marker) for marker in (
        '    bool wall_base_paired_kv_profile() const',
        '    bool wall_action_fp16_transfer_enabled() const',
        '    static bool wall_action_qkv_weights_compatible(',
        '    static bool wall_action_ring_weights_compatible(',
        '    bool wall_base_nvfp4_o_ring_norm_profile() const',
        '    bool wall_base_nvfp4_o_ring_norm_enabled(',
        '    static std::vector<int64_t> wall_base_nvfp4_o_ring_norm_arguments(',
    ))
    cold_start = DECODER.index('        wall_base_nvfp4_o_ring_norm_ =')
    cold = DECODER[cold_start:DECODER.index(';', cold_start) + 1]
    producer_ring = block(DECODER, 'if (!wall_action_linear_ring && !qwen3vl4b_fused_o &&')
    start = DECODER.index('            if (qwen3vl4b_fused_o) {')
    stop = DECODER.index('            if (!deepstack_lang_layers_.empty() &&', start)
    producer_o = DECODER[start:stop]
    shared_mlp = block(DECODER, 'if (needs_shared_mlp && !p640_staged && !wall_action_linear_ring && !decode_gate_up_fusion_)')
    consumer = block(DECODER, 'if (wall_base_nvfp4_o_ring_norm) {')
    # First occurrence is producer; select the actual O consumer after layer entry.
    layer = DECODER[DECODER.index('    void build_layer_subgraph('):]
    consumer = block(layer, 'if (wall_base_nvfp4_o_ring_norm) {')
    planner = (ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    consume = block(planner, 'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(')
    consume = consume.replace('FmbPhysicalManifestConsumer::consume_route', 'consume_physical_route')
    consume = consume.replace('int64_t invocation) const', 'int64_t invocation = 0) const')
    find = block(planner, 'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(')
    find = find.replace('FmbPhysicalManifestConsumer::find_route', 'find_route')
    source = META + r'''
enum class ChunkMode{SEQUENTIAL,KV_FIRST};
struct ChunkInfo{int64_t idx=0,offset=0,len=32,kv_seq_len=416;};
struct FmbThreeStageChunkPlan{ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;struct{std::vector<ChunkInfo> chunks{{}};}compute;};
enum class FmbRouteFamily{ATTENTION,LINEAR,ALL_REDUCE,COLLECTIVE};
enum class CausalDecoderLinearRoute{AUTO_TILE=1,WALL_RING_POSTNORM=3,ROW_WEIGHT_REUSE=5,WALL_BASE_NVFP4_O_RING_NORM=9};
enum class CausalDecoderAllReduceRoute{PREPARE_RING_INPUT=1,FP16_PARTIAL_BF16_RESIDUAL=2};
enum class RpuRmsNormSpmRoute{BASE=0};constexpr uint32_t RPU_RMSNORM_CAP_BASE=1;
constexpr int64_t CAUSAL_DECODER_WALL_BASE_NVFP4_O_RING_NORM_SITE=1,CAUSAL_DECODER_O_SITE=2,CAUSAL_DECODER_PREPARE_SITE=3,
 CAUSAL_DECODER_REDUCE_SITE=4,CAUSAL_DECODER_WALL_RING_O_SITE=5,CAUSAL_DECODER_WALL_RING_DOWN_SITE=6,
 CAUSAL_DECODER_MIXED_REDUCE_SITE=7,QWEN3VL_4B_DECODE_O_RING_NORM_SITE=8;
constexpr uint32_t FMB_SHARED_MLP_GEMV=1,FMB_SHARED_MLP_AUTO_TILE=2,FMB_SHARED_MLP_RING_REDUCE=4,
 FMB_SHARED_MLP_PREPARE_INPUT=8,FMB_SHARED_MLP_ACC32_OUT_BF16=16,FMB_SHARED_MLP_BF16_FP16_REDUCE=32,FMB_SHARED_MLP_FP16_BF16_REDUCE=64;
namespace rpu_pl_tiling {template<class... T>bool supports_w8a16_row_weight_reuse(T...){return false;}}
struct FmbRouteManifestEntry{int64_t site_id;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t> arguments;int64_t invocation;};
struct Manifest{std::vector<FmbRouteManifestEntry> routes;};
struct Context{Manifest manifest_;mutable std::vector<int> consumed_;bool complete_flag=true,is_causal=false;
 int64_t seq_len=32,position=384,batch_size=1;std::optional<int> attention_mask{1};FmbThreeStageChunkPlan stage_plan;
 bool complete()const{return complete_flag;}bool has_complete_physical_manifest()const{return complete_flag;}const Manifest& manifest()const{return manifest_;}
 FIND
 CONSUME
};
struct KernelCache{bool kernel=true,program=true;static auto& instance(){static KernelCache c;return c;}
 void* get_kernel(std::string s){assert(s=="wall_base_nvfp4_o_ring_norm_h32");return kernel?this:nullptr;}
 void* get_program(std::string s){assert(s=="wall_base_nvfp4_o_ring_norm_h32");return program?this:nullptr;}
};
int fused_calls=0;
void rpu_launch_wall_base_nvfp4_o_ring_norm_spm_kernel(uint32_t input,const at::Tensor&,uint32_t partial,uint32_t residual,
 uint32_t raw,uint32_t norm,uint32_t gamma,const at::Tensor&,uint32_t ts,int64_t count,uint16_t layer,double eps){
 assert(input==1&&partial==2&&residual==4&&raw==3&&norm==4&&gamma==5&&ts==6&&count==40&&layer==35&&eps==1e-6);++fused_calls;}
struct Model{
 struct LayerWeights{at::Tensor q_w,k_w,v_w,o_w,gate_w,up_w,down_w,input_norm_w,post_norm_w;
  at::Tensor q_ws,k_ws,v_ws,o_ws,gate_ws,up_ws,down_ws;};
 std::vector<LayerWeights> layer_weights_{36};WallSortingConfig wall_sorting_;at::Tensor final_norm_w_,o_nvfp4_tensor_scales_;
 bool nvfp4_=true,linear_acc32_=false,wall_action_fuse_silu_mul_=true,pre_layers_residual1_ready_=true,has_qkv_bias_=true,has_qk_norm_=false;
 bool has_mrope_=true,use_silu_=true,decoder_mixed_precision_=false,adarms_=false,wall_action_weight_residency_=false,wall_action_kv_owner_fanout_=false;
 bool wall_base_nvfp4_o_ring_norm_=false,partial_mrope_active_=true,rhinovla_high_precision_=false,decode_gate_up_fusion_=false;
 int64_t seq_len_=32,cache_batch_slot_=0;uint32_t wall_rmsnorm_capability_=RPU_RMSNORM_CAP_BASE;double eps_=1e-6;
 std::vector<int64_t> deepstack_lang_layers_;int hidden=1024;Context context;
 Context& ctx(){return context;}const Context& ctx()const{return context;}
 int num_layers()const{return layer_weights_.size();}int hidden_size()const{return hidden;}
 int intermediate_size()const{return 2048;}int num_q_heads()const{return 16;}int num_kv_heads()const{return 8;}
 int head_dim()const{return 128;}int num_cores()const{return 8;}int attn_tp()const{return 8;}int mlp_tp()const{return 8;}
 bool ordinary_row_weight_reuse_plan_enabled(const FmbThreeStageChunkPlan&,bool,int64_t)const{return false;}
 std::vector<int64_t> ordinary_o_route_arguments(int64_t,bool)const{return {};}
 CausalDecoderLinearRoute decoder_projection_route(int64_t)const{return CausalDecoderLinearRoute::AUTO_TILE;}
 bool qwen3vl_decode_gemv(int64_t)const{return false;}
 static std::vector<int64_t> qwen3vl_decode_o_contract_arguments(){return {};}
 uint32_t addr(int,std::string s)const{return s=="output"?1:s=="oproj"?2:s=="residual2"?3:4;}
 uint32_t norm_post_addr(int,int)const{return 5;}uint32_t nvfp4_scale_addr(const char*)const{return 6;}
 HELPERS
 void bind(){COLD}
 void install(int precision=2){nvfp4_=precision==2;wall_sorting_.silu_mul=true;wall_sorting_.fp16_transfer_mask=precision==0?4:0;
  o_nvfp4_tensor_scales_.dtype=at::kFloat;o_nvfp4_tensor_scales_.shape={40};
  for(auto& l:layer_weights_){
   for(auto* w:{&l.q_w,&l.k_w,&l.v_w,&l.o_w,&l.gate_w,&l.up_w,&l.down_w})w->dtype=precision==2?at::kByte:precision==1?at::kChar:at::kHalf;
   for(auto* s:{&l.q_ws,&l.k_ws,&l.v_ws,&l.o_ws,&l.gate_ws,&l.up_ws,&l.down_ws}){s->dtype=precision==2?at::kByte:at::kHalf;s->present=precision!=0;}
  }bind();}
 bool enabled()const{return wall_base_nvfp4_o_ring_norm_enabled(context.stage_plan,seq_len_,context.seq_len,context.position,
  context.is_causal,context.attention_mask.has_value(),context.batch_size,partial_mrope_active_);}
 uint32_t shared_routes=0;
 void build(){
  auto& plan=context.stage_plan;int64_t physical_len=seq_len_,position=context.position;
  struct Layout{bool is_causal,use_attn_mask;int64_t batch_size;}layout{context.is_causal,context.attention_mask.has_value(),context.batch_size};
  const bool wall_base_nvfp4_o_ring_norm=enabled(),wall_action_linear_ring=false,qwen3vl4b_fused_o=false,p640_staged=false,needs_shared_mlp=true;
  context.manifest_.routes.clear();
  auto append=[&](FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags=0,std::vector<int64_t> args={},int64_t invocation=0){context.manifest_.routes.push_back({site,f,selector,flags,args,invocation});};
  auto append_linear=[&](int64_t site,CausalDecoderLinearRoute route){append(FmbRouteFamily::LINEAR,site,int64_t(route));};
  auto append_ring=[&](int64_t site,int64_t,int64_t invocation){append(FmbRouteFamily::ALL_REDUCE,site,1,0,{},invocation);};
  for(const auto& chunk:plan.compute.chunks){PRODUCER_RING}
  PRODUCER_O
  shared_routes=0;SHARED_MLP
  context.consumed_.assign(context.manifest_.routes.size(),0);
 }
 void emit(){const bool wall_base_nvfp4_o_ring_norm=enabled();auto& chunk=context.stage_plan.compute.chunks.front();
  const int layer_idx=35;auto& lw=layer_weights_[layer_idx];const uint32_t input_residual=4;CONSUMER
 }
};
int main(){
 for(int p:{336,384,448,512,640})for(int precision:{0,1,2}){
  Model m;m.context.position=p;m.context.stage_plan.compute.chunks[0].kv_seq_len=p+32;m.install(precision);
  assert(m.wall_base_nvfp4_o_ring_norm_==(precision==2));m.build();fused_calls=0;m.emit();
  assert(fused_calls==(precision==2?1:0));assert(m.shared_routes==(FMB_SHARED_MLP_AUTO_TILE|FMB_SHARED_MLP_RING_REDUCE));
  auto& routes=m.context.manifest_.routes;
  if(precision==2){assert(routes.size()==1&&routes[0].site_id==CAUSAL_DECODER_WALL_BASE_NVFP4_O_RING_NORM_SITE&&m.context.consumed_[0]==1);}
  else{assert(routes.size()==3);for(auto site:{CAUSAL_DECODER_PREPARE_SITE,CAUSAL_DECODER_O_SITE,CAUSAL_DECODER_REDUCE_SITE})
   assert(std::count_if(routes.begin(),routes.end(),[&](auto r){return r.site_id==site;})==1);}
 }
 for(auto member:{&KernelCache::kernel,&KernelCache::program}){auto& cache=KernelCache::instance();cache.*member=false;
  Model m;m.install();assert(!m.wall_base_nvfp4_o_ring_norm_);m.build();fused_calls=0;m.emit();assert(fused_calls==0&&m.context.manifest_.routes.size()==3);cache.*member=true;}
 Model m;m.install();assert(m.enabled());
 auto cold_reject=[&](){m.bind();assert(!m.wall_base_nvfp4_o_ring_norm_);m.build();fused_calls=0;m.emit();assert(fused_calls==0);};
 m.linear_acc32_=true;cold_reject();m.linear_acc32_=false;
 m.wall_action_fuse_silu_mul_=false;cold_reject();m.wall_action_fuse_silu_mul_=true;
 m.wall_sorting_.silu_mul=false;cold_reject();m.wall_sorting_.silu_mul=true;
 m.pre_layers_residual1_ready_=false;cold_reject();m.pre_layers_residual1_ready_=true;
 m.hidden=2048;cold_reject();m.hidden=1024;m.wall_rmsnorm_capability_=5;cold_reject();m.wall_rmsnorm_capability_=1;
 m.rhinovla_high_precision_=true;cold_reject();m.rhinovla_high_precision_=false;
 m.layer_weights_[12].o_w.dtype=at::kChar;cold_reject();m.layer_weights_[12].o_w.dtype=at::kByte;
 m.layer_weights_[12].o_ws.dtype=at::kHalf;cold_reject();m.layer_weights_[12].o_ws.dtype=at::kByte;
 for(int n:{36,48}){m.o_nvfp4_tensor_scales_.shape={n};cold_reject();}m.o_nvfp4_tensor_scales_.shape={40};
 m.o_nvfp4_tensor_scales_.dtype=at::kHalf;cold_reject();m.o_nvfp4_tensor_scales_.dtype=at::kFloat;m.bind();
 auto fallback=[&](){assert(!m.enabled());m.build();fused_calls=0;m.emit();assert(fused_calls==0&&m.shared_routes==6);};
 m.context.position=320;fallback();m.context.position=384;m.seq_len_=20;fallback();m.seq_len_=32;
 m.context.seq_len=20;fallback();m.context.seq_len=32;m.context.is_causal=true;fallback();m.context.is_causal=false;
 m.context.attention_mask.reset();fallback();m.context.attention_mask=1;m.context.batch_size=2;fallback();m.context.batch_size=1;
 m.partial_mrope_active_=false;fallback();m.partial_mrope_active_=true;m.adarms_=true;fallback();m.adarms_=false;
 m.wall_action_weight_residency_=true;fallback();m.wall_action_weight_residency_=false;
 m.wall_action_kv_owner_fanout_=true;fallback();m.wall_action_kv_owner_fanout_=false;
 m.wall_sorting_.action_activation_reuse=true;fallback();m.wall_sorting_.action_activation_reuse=false;
 m.cache_batch_slot_=1;fallback();m.cache_batch_slot_=0;
 m.context.stage_plan.chunk_mode=ChunkMode::KV_FIRST;fallback();m.context.stage_plan.chunk_mode=ChunkMode::SEQUENTIAL;
 m.context.stage_plan.compute.chunks[0].offset=16;fallback();m.context.stage_plan.compute.chunks[0].offset=0;
 m.context.stage_plan.compute.chunks[0].len=16;fallback();m.context.stage_plan.compute.chunks[0].len=32;
 m.context.stage_plan.compute.chunks[0].kv_seq_len=400;fallback();m.context.stage_plan.compute.chunks[0].kv_seq_len=416;
 m.context.stage_plan.compute.chunks.push_back({});fallback();m.context.stage_plan.compute.chunks.pop_back();
 for(int variant:{0,1,2,3,4}){m.build();
  if(variant==0)m.context.manifest_.routes[0].arguments[2]=336;
  if(variant==1)m.context.manifest_.routes[0].selector=1;
  if(variant==2)m.context.complete_flag=false;
  if(variant==3)m.context.manifest_.routes.clear();
  if(variant==4)m.context.manifest_.routes[0].arguments.back()=0;
  fused_calls=0;bool rejected=false;try{m.emit();}catch(const std::exception&){rejected=true;}
  assert(rejected&&fused_calls==0);m.context.complete_flag=true;
 }
 m.build();m.wall_base_nvfp4_o_ring_norm_=false;assert(!m.enabled());
 // Conversely, installing a fallback manifest cannot authorize fused O.
 m.build();m.wall_base_nvfp4_o_ring_norm_=true;fused_calls=0;bool rejected=false;
 try{m.emit();}catch(const std::exception&){rejected=true;}assert(rejected&&fused_calls==0);
}
'''
    for key, value in [('HELPERS', helpers), ('COLD', cold), ('PRODUCER_RING', producer_ring),
                       ('PRODUCER_O', producer_o), ('SHARED_MLP', shared_mlp),
                       ('CONSUMER', consumer), ('CONSUME', consume), ('FIND', find)]:
        source = re.sub(r'\b' + key + r'\b', lambda _: value, source)
    compile_run(tmp_path, 'wall_o_route', source)
