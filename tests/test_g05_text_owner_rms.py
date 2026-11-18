"""Actual Text owner RMS wrapper, cold capability and layer-local caller contracts."""
from pathlib import Path
from test_g05_q_gate_native import _compile_run
from test_g05_vision_owner_model import _block

ROOT=Path(__file__).resolve().parents[1]


def test_text_owner_rms_wrapper_registers_and_bounds(tmp_path):
    source=(ROOT/'src/ops/rpu_pi05_owner_norm.cpp').read_text()
    marker='void rpu_launch_g05_text_owner_rms_m256h2048_kernel('
    begin=source.index(marker)
    function=source[begin:source.index('{',begin)]+_block(source,marker)
    cpp=r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10{struct Half{uint16_t x;explicit Half(double v):x(v==1.0/2048?0x1000:0x11){assert(v==1.0/2048||v==1e-6);}};}
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8303616;bool ready=true;bool is_initialized()const{return ready;}uint32_t addr(int core,uint32_t off)const{return 0x40000000u+core*0x1000000u+off;}}allocator;
#define SPM_ALLOC allocator
enum class KernelId{G05_TEXT_RING_OWNER_RMS_M256H2048};constexpr auto ID=KernelId::G05_TEXT_RING_OWNER_RMS_M256H2048;
struct KernelCache{bool available=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId)const{return available;}};
constexpr int kKernelRegOk=0;
struct Kernel{std::map<uint32_t,uint16_t>regs,scm;void reset_regs(){regs.clear();scm.clear();}int set_regs(uint32_t r,uint16_t v){regs[r]=v;return 0;}int set_regs(uint32_t,uint32_t)=delete;};
enum class GraphKernelNoDdrProof{RingAllReduceSpmResidual};
struct RpuKernelGraph{
 enum class State{RECORDING,REPLAYING,PASSTHROUGH};State current=State::RECORDING;bool census=true;int stages=0,gets=0;bool staged=false,pending=false;Kernel kernel;
 static RpuKernelGraph&active(){static RpuKernelGraph g;return g;}State state()const{return current;}bool kernel_register_census_active()const{return census;}
 void stage_kernel_no_ddr(KernelId id,GraphKernelNoDdrProof){assert(id==ID&&!staged&&!pending);staged=true;++stages;}
 Kernel*get(KernelId id){assert(id==ID&&!pending&&(census?staged:!staged));pending=true;++gets;return &kernel;}
};
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
void rpu_set_legacy_scm_u16_checked(Kernel*k,uint32_t r,uint16_t v,const char*){assert(r>=4096&&r<4176);k->scm[r]=v;}
struct Queue{int calls=0;bool broadcast=false;void set_broadcast_mode(bool v){broadcast=v;}void enqueu_kernel(Kernel&,std::vector<uint16_t>grid,std::vector<uint8_t>cores){auto&g=RpuKernelGraph::active();assert(broadcast&&g.pending&&grid==std::vector<uint16_t>({8,1,1})&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));g.pending=g.staged=false;++calls;}}queue;
Queue*get_queue(int c){assert(c==8);return &queue;}
#define GET_QUEUE(c) get_queue(c)
__FUNCTION__
uint32_t pair(const std::map<uint32_t,uint16_t>&m,uint32_t r,int delta=1){return uint32_t(m.at(r))|(uint32_t(m.at(r+delta))<<16);}
template<class F>void reject(F f){auto&g=RpuKernelGraph::active();int stages=g.stages,gets=g.gets,calls=queue.calls;try{f();assert(false);}catch(const std::runtime_error&){}assert(stages==g.stages&&gets==g.gets&&calls==queue.calls&&!g.staged&&!g.pending);}
int main(){
 std::array<uint32_t,4>a{0x4000ff00,0x4014ff00,0x4028ff00,0x403cff00};
 auto run=[&]{rpu_launch_g05_text_owner_rms_m256h2048_kernel(a[0],a[1],a[2],a[3]);};
 auto&g=RpuKernelGraph::active();
 for(bool census:{true,false})for(auto state:{RpuKernelGraph::State::RECORDING,RpuKernelGraph::State::REPLAYING}){
  g.census=census;g.current=state;g.kernel.regs[128]=99;run();const auto&r=g.kernel.regs;const auto&s=g.kernel.scm;
  assert(r.at(0)==8&&r.at(1)==2&&pair(r,2)==63488&&pair(r,4)==a[1]-0x40000000&&r.at(6)==0);
  assert(r.at(16)==2048&&r.at(17)==0x1000&&pair(r,18)==4096&&pair(r,20)==a[2]&&pair(r,22)==a[1]&&pair(r,24)==a[3]&&r.at(26)==0x11);
  assert(r.at(64)==8&&r.at(65)==1&&r.at(66)==1);for(int i=128;i<256;++i)assert(r.at(i)==0);
  for(uint32_t c=0;c<8;++c){assert(pair(s,4096+c,8)==a[0]-0x40000000+c*0x1000000);assert(pair(s,4112+c,8)==a[2]-0x40000000+c*0x1000000);assert(pair(s,4128+c,8)==c*131072);assert(pair(s,4144+c,8)==65536);assert(pair(s,4160+c,8)==a[1]-0x40000000+c*0x1000000);}
 }
 for(int i=0;i<4;++i){auto save=a[i];for(uint32_t bad:{0u,save+1,0x407eb600u}){a[i]=bad;reject(run);}a[i]=save;}
 for(int i=0;i<4;++i)for(int j=0;j<i;++j){auto save=a[i];a[i]=a[j];reject(run);a[i]=save;}
 allocator.ready=false;reject(run);allocator.ready=true;g.current=RpuKernelGraph::State::PASSTHROUGH;reject(run);g.current=RpuKernelGraph::State::RECORDING;
 KernelCache::instance().available=false;reject(run);
}
'''
    _compile_run(tmp_path,'text_owner_wrapper',cpp.replace('__FUNCTION__',function))


def test_text_owner_rms_cold_complete_and_layer_local_skip(tmp_path):
    source=(ROOT/'src/fused/rpu_qwen3_5_model.cpp').read_text()
    header=(ROOT/'src/fused/rpu_qwen3_5_model.h').read_text()
    methods=[
        ('bool has_text_fp16_projection_owners()const','bool Qwen3_5Model::has_text_fp16_projection_owners('),
        ('void prepare_text_gdn_owner_rms()','void Qwen3_5Model::prepare_text_gdn_owner_rms('),
        ('bool use_text_gdn_owner_rms(const FmbThreeStageChunkPlan&plan,int64_t physical_len,int64_t logical_len,int64_t position,int64_t batch)const','bool Qwen3_5Model::use_text_gdn_owner_rms('),
        ('std::vector<int64_t>text_gdn_owner_rms_arguments(int64_t logical_len=245)const','std::vector<int64_t> Qwen3_5Model::text_gdn_owner_rms_arguments('),
        ('void launch_text_gdn_owner_rms(int layer,const ChunkInfo&chunk)','void Qwen3_5Model::launch_text_gdn_owner_rms('),
        ('void prepare_text_owner_rms()','void Qwen3_5Model::prepare_text_owner_rms('),
        ('bool use_text_owner_rms(const FmbThreeStageChunkPlan&plan,int64_t physical_len,int64_t logical_len,int64_t position,int64_t batch)const','bool Qwen3_5Model::use_text_owner_rms('),
        ('std::vector<int64_t>text_owner_rms_arguments(int64_t logical_len=245)const','std::vector<int64_t> Qwen3_5Model::text_owner_rms_arguments('),
        ('void launch_text_owner_rms(int layer,const ChunkInfo&chunk)','void Qwen3_5Model::launch_text_owner_rms('),
        ('void set_linear_acc32(bool enabled)','void Qwen3_5Model::set_linear_acc32('),
        ('void set_fast_replay(bool enabled)','void Qwen3_5Model::set_fast_replay('),
    ]
    definitions='\n'.join(decl+_block(source,marker) for decl,marker in methods)
    body=_block(source,'void Qwen3_5Model::build_layer_subgraph(')
    body=body[body.index('    bool text_owner_norm_emitted = false;'):body.rindex('}')]
    attention=_block(source,'void Qwen3_5Model::build_full_attention(')
    attention=attention[attention.index('    // Phase 5:'):attention.rindex('}')]
    tail=_block(source,'void Qwen3_5Model::emit_mlp_and_output(')
    tail=tail[tail.index('    // post-attention RMSNorm:'):tail.index('    // MLP (SwiGLU)')]
    mlp=_block(source,'void Qwen3_5Model::emit_mlp_and_output(')
    ring=mlp[mlp.index('    if (action_mlp_tail_handoff(layer_idx,chunk,true))'):mlp.index('    // Split the last decoder-layer output')]
    marker='if (use_text_owner_rms(plan,physical_len,logical_len,position,layout.batch_size))'
    start=source.index(marker);end=source.index('\n        }\n        if (has_gdn)',start)
    manifest=source[start:end]
    gdn=_block(source,'void Qwen3_5Model::build_gdn(')
    gdn=gdn[gdn.index('    // out_proj (row-parallel)'):gdn.rindex('}')]
    marker='if (use_text_gdn_owner_rms(plan,physical_len,logical_len,position,layout.batch_size))'
    start=source.index(marker);end=source.index('\n        }\n        append_route(',start)
    gdn_manifest=source[start:end]
    gdn_layout=_block(source,'if (!action_mode_ && text_gdn_owner_rms_available_)')
    gdn_identity=_block(header,'if (!action_mode_ && text_gdn_owner_rms_available_)')
    layout=_block(source,'if (!action_mode_ && text_owner_rms_available_)')
    identity=_block(header,'if (!action_mode_ && text_owner_rms_available_)')
    cpp=r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10{enum class DeviceType{PrivateUse1,CPU};}
namespace at{constexpr int kHalf=1,kChar=2;struct Tensor{
 bool present=true,contiguous=true;int dtype=kChar,rank=2,index=0;int64_t count=2048,offset=0,owner=0,rows=2048,cols=2048;c10::DeviceType location=c10::DeviceType::PrivateUse1;
 bool defined()const{return present;}int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}int dim()const{return rank;}int64_t storage_offset()const{return offset;}int64_t numel()const{return count;}int64_t size(int d)const{return d==0?rows:cols;}
 struct Device{c10::DeviceType value;int index;c10::DeviceType type()const{return value;}bool operator!=(Device b)const{return value!=b.value||index!=b.index;}bool operator==(Device b)const{return !(*this!=b);}};Device device()const{return {location,index};}
};}
struct Weights{at::Tensor attn_gate_w,attn_gate_ws,down_w,down_ws,gate_w,gate_ws,gdn_A_log,gdn_a_bg_w,gdn_a_bg_ws,gdn_b_bg_w,gdn_b_bg_ws,gdn_conv_k_w,gdn_conv_q_w,gdn_conv_v_w,gdn_dt_bias,gdn_in_z_w,gdn_in_z_ws,gdn_k_w,gdn_k_ws,gdn_norm_w,gdn_out_w,gdn_out_ws,gdn_q_w,gdn_q_ws,gdn_v_w,gdn_v_ws,input_norm_w,k_norm_w,k_w,k_ws,o_w,o_ws,post_norm_w,q_norm_w,q_w,q_ws,up_w,up_ws,v_w,v_ws;};
enum class KernelId{G05_TEXT_RING_OWNER_RMS_M256H2048};struct KernelCache{bool available=true;int calls=0;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId){++calls;return available;}};
enum class FmbRouteFamily{ALL_REDUCE,MUTABLE_DMA};enum class Qwen35TextAllReduceRoute{PREPARE_RING_INPUT};enum class Qwen35TextMutableDmaRoute{SPM_SCATTER_TO_DDR};int64_t qwen35_text_dma_route(Qwen35TextMutableDmaRoute x){return static_cast<int64_t>(x);}enum class RpuRmsNormSpmRoute{BASE};
constexpr int64_t QWEN35_TEXT_OWNER_RMS_SITE=8722488789555919585LL,QWEN35_TEXT_FULL_ALL_REDUCE_SITE=1,QWEN35_TEXT_MLP_ALL_REDUCE_SITE=2,QWEN35_TEXT_GDN_OWNER_RMS_SITE=7871266237092441430LL,QWEN35_TEXT_GDN_ALL_REDUCE_SITE=3,QWEN35_TEXT_GDN_PREPARE_ALL_REDUCE_SITE=4,QWEN35_TEXT_GDN_STATE_DECODE_STORE_DMA_SITE=5,QWEN35_TEXT_GDN_CONV_DECODE_STORE_DMA_SITE=6;
struct ChunkInfo{int64_t len=256,idx=0,offset=0;};struct Stage{std::vector<ChunkInfo>chunks{{}};};struct FmbThreeStageChunkPlan{Stage qkv,compute;};
struct Route{int64_t site;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>args;int64_t invocation;};
struct Manifest{int64_t logical_length=245;std::vector<Route>routes;};
std::vector<std::string>launches;bool fail_producer=false;
void rpu_launch_g05_text_owner_rms_m256h2048_kernel(uint32_t x,uint32_t r,uint32_t raw,uint32_t g){assert((x==0x10000||x==0x510000)&&r==0x110000&&raw==0x210000&&g>=0x400000);if(fail_producer)throw std::runtime_error("producer failed");launches.push_back(x==0x10000?"owner":"owner_gdn");}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t x,uint32_t r,uint32_t y,int64_t m,int64_t n,int tp,int cores){assert(m==256&&n==2048&&tp==8&&cores==8);if(x==0x10000||x==0x510000){assert(r==0x110000&&y==0x210000);launches.push_back(x==0x10000?"ring_o":"ring_gdn");}else{assert(x==0x310000&&r==0x210000&&y==0x110000);launches.push_back("ring_mlp");}}
void rpu_launch_rmsnorm_spm_kernel(uint32_t x,uint32_t y,uint32_t g,int64_t m,int64_t n,double eps,RpuRmsNormSpmRoute,int cores){assert(x==0x210000&&y==0x110000&&g>=0x400000&&m==256&&n==2048&&eps==1e-6&&cores==8);launches.push_back("norm");}
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>&v,const at::Tensor&t){v.push_back(t.owner);}
namespace detail{int64_t layout_mix(int64_t a,int64_t b){return uint64_t(a)*131+uint64_t(b);}}
template<class...T>void rpu_prepare_ring_all_reduce_input(T...){assert(false);}
template<class...T>void rpu_launch_spm_scatter_ddr_dma_mutable(T...){assert(false);}
struct Model{
 template<class...T>bool action_mlp_tail_handoff(T...){return false;}
 template<class...T>void launch_action_mlp_tail(T...){assert(false);}
 static constexpr int64_t QWEN35_TEXT_MLP_TAIL_SITE=8475656447499929675LL;
 bool text_mlp_tail_available_=false;
 void prepare_text_mlp_tail(){}
 template<class...T>bool use_text_mlp_tail(T...)const{return false;}
 template<class...T>bool text_mlp_tail_handoff(T...){return false;}
 std::vector<int64_t>text_mlp_tail_arguments(int64_t=245)const{return {};}
 template<class...T>void launch_text_mlp_tail(T...){assert(false);}

 bool text_silu_round_mul_available_=false;
 void prepare_text_silu_round_mul(){}
 template<class...T>bool use_text_silu_round_mul(T...)const{return false;}
 std::vector<int64_t>text_silu_round_mul_arguments(int64_t=245)const{return {};}
 template<class...T>void launch_text_silu_round_mul(T...){assert(false);}
 bool use_action_tail(int64_t)const{return false;}
 template<class...T>void launch_action_tail(T...){assert(false);}
 bool text_gdn_owner_rms_available_=false,text_owner_rms_available_=false,action_mode_=false,linear_acc32_=false,fast_replay_enabled_=true,use_silu_=true,has_gdn_=true,z2_bound_=false,debug=false,retained_prefill_graph_=true,complete=true;
 int64_t layers=24,hidden=2048,intermediate=6144,cores=8,tp=8,heads=8,kvheads=8,hd=256,last=0;
 int64_t gdn_nvh_=16,gdn_dk_=128,gdn_dv_=128,gdn_conv_dim_=6144,gdn_kc_=4;
 std::array<uint64_t,24>gdn_state_live_addr_{},conv_state_live_addr_{};
 int64_t seq_len=256,position=0,batch_size=1;double eps_=1e-6;int invalidations=0;
 struct{bool enabled()const{return false;}}action_nvfp4_;
 // The dense-text NVFP4 owner is uninstalled in this probe (enabled() false).
 struct{bool enabled()const{return false;}}text_nvfp4_;
 std::vector<Weights>layer_weights_{24};std::vector<uint8_t>layer_is_full_=std::vector<uint8_t>(24);FmbThreeStageChunkPlan stage_plan;Manifest manifest;
 Model(){for(int i=0;i<24;++i){layer_is_full_[i]=(i+1)%4==0;auto&n=layer_weights_[i].post_norm_w;n.dtype=at::kHalf;n.rank=1;n.owner=i+1;
  auto&w=layer_weights_[i];w.gate_w.rows=w.up_w.rows=6144;w.down_w.cols=6144;
  std::array<at::Tensor*,4>ss{&w.gdn_out_ws,&w.gate_ws,&w.up_ws,&w.down_ws};
  for(int f=0;f<4;++f){ss[f]->dtype=at::kHalf;ss[f]->rank=1;ss[f]->count=f==1||f==2?6144:2048;}
 }}
 int64_t num_layers()const{return layers;}int64_t hidden_size()const{return hidden;}int64_t intermediate_size()const{return intermediate;}int64_t num_cores()const{return cores;}int64_t attn_tp()const{return tp;}int64_t gdn_tp()const{return tp;}int64_t mlp_tp()const{return tp;}int64_t num_q_heads()const{return heads;}int64_t num_kv_heads()const{return kvheads;}int64_t head_dim()const{return hd;}bool get_debug_export()const{return debug;}
 int64_t get_last_resolved_chunk_size()const{return last;}void prepare_text_gateup_pair(){}void invalidate_model_state(){++invalidations;}
 Model&ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}const Manifest&physical_manifest()const{return manifest;}
 uint32_t addr(int,const char*name)const{std::string s(name);return s=="gdn_c_m"?0x510000:s=="oproj"?0x10000:s=="residual1"?0x110000:s=="down"?0x310000:0x210000;}
 uint32_t layer_addr(int layer,int,const char*)const{return 0x400000+layer*4096;}uint32_t layer_input_residual_addr(int,const ChunkInfo&)const{return addr(0,"residual1");}
 int64_t text_ring_route(int64_t,int64_t)const{return 1;}bool use_action_ring_rtc_residual(int64_t)const{return false;}
 template<class...T>void launch_action_ring_rtc_residual(T...){assert(false);}template<class...T>void apply_adaptive_residual_gate(T...){assert(false);}template<class...T>void apply_adaptive_norm(T...){assert(false);}void load_adaptive_mod_row(int){assert(false);}
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t>args,int64_t invocation){auto i=std::find_if(manifest.routes.begin(),manifest.routes.end(),[&](auto&r){return r.site==site&&r.invocation==invocation;});TORCH_CHECK(complete&&i!=manifest.routes.end()&&i->family==family&&i->selector==selector&&i->flags==flags&&i->args==args);}
 void consume_manifest_route(FmbRouteFamily f,int64_t site,int64_t s,int64_t invocation){consume_physical_route(f,site,s,0,{},invocation);}
 __METHODS__
 void installed(){prepare_text_owner_rms();invalidate_model_state();}
 void plan(){auto&plan=stage_plan;int64_t physical_len=seq_len,logical_len=manifest.logical_length;struct{int64_t batch_size;}layout{batch_size};ChunkInfo chunk=stage_plan.compute.chunks.front();manifest.routes.clear();auto append_route=[&](FmbRouteFamily f,int64_t site,int64_t selector,int64_t invocation){manifest.routes.push_back({site,f,selector,0,{},invocation});};__MANIFEST__ __GDN_MANIFEST__ manifest.routes.push_back({QWEN35_TEXT_MLP_ALL_REDUCE_SITE,FmbRouteFamily::ALL_REDUCE,1,0,{},0});}
 int64_t layout_hash(){int64_t hash=0;if(!action_mode_&&text_owner_rms_available_)__LAYOUT__ if(!action_mode_&&text_gdn_owner_rms_available_)__GDN_LAYOUT__ return hash;}
 std::vector<int64_t>identity(){std::vector<int64_t>identity;if(!action_mode_&&text_owner_rms_available_)__IDENTITY__ if(!action_mode_&&text_gdn_owner_rms_available_)__GDN_IDENTITY__ for(auto&w:layer_weights_)append_kvinsert_cost_tensor_identity(identity,w.post_norm_w);return identity;}
 void build_full_attention(int layer_idx,const ChunkInfo&chunk,bool*text_owner_norm_emitted,bool*action_norm_emitted=nullptr){int64_t h=hidden_size(),tp=attn_tp();__ATTENTION__}
 void launch_linear(int64_t,uint32_t input,const at::Tensor&,uint32_t output,int64_t m,int64_t n,int64_t k,int part,int cores,int bias,const at::Tensor&){assert(input==addr(0,"input_norm")&&output==addr(0,"gdn_c_m")&&m==256&&n==2048&&k==2048&&part==0&&cores==8&&!bias);launches.push_back("proj_gdn");}
 void build_gdn(int layer_idx,const ChunkInfo&chunk,bool decode,bool*text_owner_norm_emitted=nullptr){
  const auto&lw=layer_weights_[layer_idx];auto D=[&](const char*n){return addr(0,n);};
  int64_t L=chunk.len,hidden=hidden_size(),value_dim=2048,Hc=2,Dk=128,Dv=128,Kc=4,lconv=768;int nc=gdn_tp();
  const uint32_t gated_buf=D("gdn_c_out_tok"),m_buf=D("gdn_c_m");__GDN__
 }
 void emit_mlp_and_output(int layer_idx,const ChunkInfo&chunk,bool text_owner_norm_emitted,bool action_norm_emitted=false){int64_t h=hidden_size();__TAIL__ launches.push_back("mlp");const uint32_t residual_buf=addr(0,"input_norm");__RING__}
 void layer(int layer_idx,const ChunkInfo&chunk={}){__BODY__}
};
template<class F>void reject(F f){try{f();assert(false);}catch(const std::runtime_error&){};}
int main(){
 Model m;m.installed();assert(m.text_owner_rms_available_&&m.text_gdn_owner_rms_available_);m.plan();const auto hash=m.layout_hash();const auto identity=m.identity();const int queries=KernelCache::instance().calls;
 {auto x=m;x.high_precision_=true;x.installed();assert(!x.text_owner_rms_available_&&!x.text_gdn_owner_rms_available_&&!x.text_mlp_tail_available_);}
 assert(m.manifest.routes.size()==3&&m.manifest.routes[0].site==QWEN35_TEXT_OWNER_RMS_SITE);for(auto x:m.text_owner_rms_arguments())assert(x>=0);
 for(int layer=0;layer<24;++layer){launches.clear();m.layer(layer);assert(launches==(layer%4==3?std::vector<std::string>{"owner","mlp","ring_mlp"}:std::vector<std::string>{"proj_gdn","owner_gdn","mlp","ring_mlp"}));}
 assert(KernelCache::instance().calls==queries);
 for(int layer:{3,7,11,15,19,23})for(int family=0;family<8;++family){auto x=m;auto&w=x.layer_weights_[layer];std::array<at::Tensor*,8>ws{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.attn_gate_w,&w.gate_w,&w.up_w,&w.down_w};ws[family]->dtype=at::kHalf;x.installed();assert(!x.text_owner_rms_available_);}
 for(int layer:{3,7,11,15,19,23})for(int fault=0;fault<8;++fault){auto x=m;auto&n=x.layer_weights_[layer].post_norm_w;if(fault==0){n.present=false;}if(fault==1){n.dtype=at::kChar;}if(fault==2){n.contiguous=false;}if(fault==3){n.rank=2;}if(fault==4){n.count=2047;}if(fault==5){n.offset=128;}if(fault==6){n.location=c10::DeviceType::CPU;}if(fault==7){n.index=1;}x.installed();assert(!x.text_owner_rms_available_);x.plan();launches.clear();x.layer(layer);assert(launches==std::vector<std::string>({"ring_o","norm","mlp","ring_mlp"}));}
 for(int bad=0;bad<24;++bad){auto x=m;x.layer_is_full_[bad]=!x.layer_is_full_[bad];x.installed();assert(!x.text_owner_rms_available_);}
 for(int fault=0;fault<12;++fault){auto x=m;if(fault==0){x.action_mode_=true;}if(fault==1)continue;if(fault==2){x.fast_replay_enabled_=false;}if(fault==3){x.use_silu_=false;}if(fault==4){x.has_gdn_=false;}if(fault==5){x.eps_=1e-5;}if(fault==6){x.hidden=1024;}if(fault==7){x.cores=4;}if(fault==8){x.tp=4;}if(fault==9){x.layer_weights_.pop_back();}if(fault==10){x.layer_is_full_.pop_back();}if(fault==11){x.intermediate=4096;}x.installed();assert(!x.text_owner_rms_available_);}
 for(int fault=0;fault<12;++fault){auto x=m;if(fault==0){x.z2_bound_=true;}if(fault==1){x.debug=true;}if(fault==2){x.retained_prefill_graph_=false;}if(fault==3){x.manifest.logical_length=244;}if(fault==4){x.seq_len=128;}if(fault==5){x.position=245;}if(fault==6){x.batch_size=2;}if(fault==7)x.stage_plan.qkv.chunks.push_back({});if(fault==8)x.stage_plan.compute.chunks.push_back({});if(fault==9){x.stage_plan.qkv.chunks[0].offset=256;}if(fault==10){x.stage_plan.compute.chunks[0].len=128;}if(fault==11){x.stage_plan.compute.chunks[0].idx=1;}assert(!x.use_text_owner_rms(x.stage_plan,x.seq_len,x.manifest.logical_length,x.position,x.batch_size));}
 for(size_t arg=0;arg<m.manifest.routes[0].args.size();++arg){auto x=m;++x.manifest.routes[0].args[arg];launches.clear();reject([&]{x.layer(3);});assert(launches.empty());}
 for(int fault=0;fault<5;++fault){auto x=m;if(fault==0){++x.manifest.routes[0].flags;}if(fault==1){++x.manifest.routes[0].selector;}if(fault==2){++x.manifest.routes[0].invocation;}if(fault==3){++x.manifest.routes[0].site;}if(fault==4){x.manifest.routes.erase(x.manifest.routes.begin());}launches.clear();reject([&]{x.layer(3);});assert(launches.empty());}
 fail_producer=true;launches.clear();reject([&]{m.layer(3);});assert(launches.empty());fail_producer=false;launches.clear();m.layer(4);assert(launches==std::vector<std::string>({"proj_gdn","owner_gdn","mlp","ring_mlp"}));
 {auto x=m;x.text_gdn_owner_rms_available_=false;reject([&]{x.emit_mlp_and_output(4,{},true);});}m.complete=false;launches.clear();reject([&]{m.launch_text_owner_rms(3,{});});assert(launches.empty());m.complete=true;
 KernelCache::instance().available=false;m.installed();assert(!m.text_owner_rms_available_&&!m.text_gdn_owner_rms_available_&&m.layout_hash()!=hash&&m.identity()!=identity);m.plan();launches.clear();m.layer(3);assert(launches==std::vector<std::string>({"ring_o","norm","mlp","ring_mlp"}));launches.clear();m.layer(0);assert(launches==std::vector<std::string>({"proj_gdn","ring_gdn","norm","mlp","ring_mlp"}));KernelCache::instance().available=true;m.installed();assert(m.layout_hash()==hash&&m.identity()==identity);m.plan();
 {auto x=m;++x.layer_weights_[23].post_norm_w.owner;assert(x.layout_hash()!=hash&&x.identity()!=identity);}

 // GDN-only faults cannot remove the existing six-FA route. Conversely, an
 // FA weight failure does not remove the independent eighteen-GDN route.
 for(int layer=0;layer<24;++layer)if(layer%4!=3){
  for(int family=0;family<4;++family)for(int fault=0;fault<16;++fault){
   auto x=m;auto&w=x.layer_weights_[layer];
   std::array<at::Tensor*,4>weights{&w.gdn_out_w,&w.gate_w,&w.up_w,&w.down_w};
   std::array<at::Tensor*,4>scales{&w.gdn_out_ws,&w.gate_ws,&w.up_ws,&w.down_ws};
   auto&v=*weights[family];auto&s=*scales[family];
   if(fault==0){v.present=false;}if(fault==1){v.dtype=at::kHalf;}if(fault==2){v.contiguous=false;}
   if(fault==3){v.offset=256;}if(fault==4){v.rank=1;}if(fault==5){--v.rows;}if(fault==6){--v.cols;}if(fault==7){v.location=c10::DeviceType::CPU;}
   if(fault==8){s.present=false;}if(fault==9){s.dtype=at::kChar;}if(fault==10){s.contiguous=false;}if(fault==11){s.offset=128;}
   if(fault==12){s.rank=2;}if(fault==13){--s.count;}if(fault==14){s.index=1;}if(fault==15){s.location=c10::DeviceType::CPU;}
   x.installed();assert(!x.text_gdn_owner_rms_available_&&x.text_owner_rms_available_);assert(x.layout_hash()!=hash&&x.identity()!=identity);
   if(fault==3||fault==11){x.plan();launches.clear();x.layer(layer);assert(launches==std::vector<std::string>({"proj_gdn","ring_gdn","norm","mlp","ring_mlp"}));}
  }
  for(int fault=0;fault<8;++fault){auto x=m;auto&n=x.layer_weights_[layer].post_norm_w;
   if(fault==0){n.present=false;}if(fault==1){n.dtype=at::kChar;}if(fault==2){n.contiguous=false;}if(fault==3){n.rank=2;}
   if(fault==4){--n.count;}if(fault==5){n.offset=128;}if(fault==6){n.location=c10::DeviceType::CPU;}if(fault==7){n.index=1;}
   x.installed();assert(!x.text_gdn_owner_rms_available_&&x.text_owner_rms_available_);
  }
 }
 {auto x=m;x.layer_weights_[3].q_w.dtype=at::kHalf;x.installed();assert(!x.text_owner_rms_available_&&x.text_gdn_owner_rms_available_);x.plan();launches.clear();x.layer(0);assert(launches==std::vector<std::string>({"proj_gdn","owner_gdn","mlp","ring_mlp"}));}
 for(int fault=0;fault<5;++fault){auto x=m;if(fault==0){++x.gdn_nvh_;}if(fault==1){++x.gdn_dk_;}if(fault==2){++x.gdn_dv_;}if(fault==3){++x.gdn_conv_dim_;}if(fault==4){++x.gdn_kc_;}x.installed();assert(!x.text_gdn_owner_rms_available_&&x.text_owner_rms_available_);}
 m.plan();assert(m.manifest.routes[1].site==QWEN35_TEXT_GDN_OWNER_RMS_SITE);for(auto v:m.text_gdn_owner_rms_arguments())assert(v>=0);
 for(size_t arg=0;arg<m.manifest.routes[1].args.size();++arg){auto x=m;++x.manifest.routes[1].args[arg];launches.clear();reject([&]{x.layer(0);});assert(launches==std::vector<std::string>({"proj_gdn"}));}
 for(int fault=0;fault<5;++fault){auto x=m;if(fault==0){++x.manifest.routes[1].flags;}if(fault==1){++x.manifest.routes[1].selector;}if(fault==2){++x.manifest.routes[1].invocation;}if(fault==3){++x.manifest.routes[1].site;}if(fault==4){x.manifest.routes.erase(x.manifest.routes.begin()+1);}launches.clear();reject([&]{x.layer(0);});assert(launches==std::vector<std::string>({"proj_gdn"}));}
 fail_producer=true;launches.clear();reject([&]{m.layer(0);});assert(launches==std::vector<std::string>({"proj_gdn"}));fail_producer=false;
 {auto x=m;x.text_gdn_owner_rms_available_=false;x.plan();launches.clear();x.layer(1);assert(launches==std::vector<std::string>({"proj_gdn","ring_gdn","norm","mlp","ring_mlp"}));}
 {auto x=m;++x.layer_weights_[0].post_norm_w.owner;assert(x.layout_hash()!=hash&&x.identity()!=identity);}
 for(int fault=0;fault<12;++fault){auto x=m;if(fault==0){x.z2_bound_=true;}if(fault==1){x.debug=true;}if(fault==2){x.retained_prefill_graph_=false;}if(fault==3){x.manifest.logical_length=244;}if(fault==4){x.seq_len=128;}if(fault==5){x.position=245;}if(fault==6){x.batch_size=2;}if(fault==7)x.stage_plan.qkv.chunks.push_back({});if(fault==8)x.stage_plan.compute.chunks.push_back({});if(fault==9){x.stage_plan.qkv.chunks[0].offset=256;}if(fault==10){x.stage_plan.compute.chunks[0].len=128;}if(fault==11){x.stage_plan.compute.chunks[0].idx=1;}assert(!x.use_text_gdn_owner_rms(x.stage_plan,x.seq_len,x.manifest.logical_length,x.position,x.batch_size));}
 {auto x=m;x.complete=false;launches.clear();reject([&]{x.launch_text_gdn_owner_rms(0,{});});assert(launches.empty());}
 {auto x=m;launches.clear();reject([&]{x.launch_text_gdn_owner_rms(3,{});});assert(launches.empty());}

 {auto x=m;x.set_linear_acc32(true);assert(x.text_owner_rms_available_&&x.text_gdn_owner_rms_available_&&x.text_owner_rms_arguments()[11]==32&&x.text_gdn_owner_rms_arguments()[11]==32&&x.layout_hash()!=hash&&x.identity()!=identity);launches.clear();reject([&]{x.layer(3);});assert(launches.empty());launches.clear();reject([&]{x.layer(0);});assert(launches==std::vector<std::string>({"proj_gdn"}));x.plan();for(int layer=0;layer<24;++layer){launches.clear();x.layer(layer);assert(launches==(layer%4==3?std::vector<std::string>{"owner","mlp","ring_mlp"}:std::vector<std::string>{"proj_gdn","owner_gdn","mlp","ring_mlp"}));}}
 m.set_linear_acc32(true);assert(m.text_owner_rms_available_&&m.text_gdn_owner_rms_available_);m.set_linear_acc32(false);assert(m.text_owner_rms_available_);m.set_fast_replay(false);assert(!m.text_owner_rms_available_&&!m.text_gdn_owner_rms_available_);m.set_fast_replay(true);assert(m.text_owner_rms_available_);m.last=256;reject([&]{m.set_fast_replay(false);});assert(m.text_owner_rms_available_);
}
'''
    cpp=cpp.replace('#include <algorithm>','#include <algorithm>\n#include <array>')
    for key,value in dict(METHODS=definitions,MANIFEST=manifest,LAYOUT=layout,IDENTITY=identity,ATTENTION=attention,TAIL=tail,RING=ring,BODY=body,GDN=gdn,GDN_MANIFEST=gdn_manifest,GDN_LAYOUT=gdn_layout,GDN_IDENTITY=gdn_identity).items():
        cpp=cpp.replace('__'+key+'__',value)
    _compile_run(tmp_path,'text_owner_model',cpp)
