"""Host execution of time-table admission, typed DMA bounds and cold emitters."""
from pathlib import Path
import re

from test_gr00t_adaln_table import _run
from test_siglip_attention_dry_live_contract import _function

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/fused/rpu_gr00t_dit_model.cpp"
HELPER = ROOT / "src/ops/rpu_gr00t_adaln.cpp"


def test_time_tables_cold_boundary_replay_and_descriptor_drift(tmp_path):
    source = SOURCE.read_text()
    methods = "\n".join(_function(source, marker) for marker in (
        "    void check_time_prepare_state(", "    void clear_time_tables(",
        "    void begin_time_inference(", "    bool time_table_profile(",
        "    std::vector<int64_t> time_table_arguments(", "    bool time_tables_active(",
        "    bool load_time_table(",
    ))
    constants = "\n".join(re.findall(r"constexpr int64_t GR00T_DIT_TIME_TABLE_\w+ = \d+LL;", source))
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(ok, ...) do{if(!(ok))throw std::runtime_error(#ok);}while(false)
#define DWIDTH 2
struct Version{bool valid=true;int value=0;bool enabled()const{return valid;}int current_version()const{return value;}};
struct Impl{Version v;const Version&version_counter()const{return v;}};
namespace at{struct Tensor{bool live=false;int tag=0;std::shared_ptr<Impl>p=std::make_shared<Impl>();bool defined()const{return live;}Impl*unsafeGetTensorImpl()const{return p.get();}};}
struct RpuKernelGraph{static bool active;static bool has_active(){return active;}};bool RpuKernelGraph::active=false;
enum class FmbRouteFamily{GRAPH_SCHEDULE,MUTABLE_DMA};
enum class Gr00tDiTMutableDmaRoute:int64_t{DDR_BROADCAST_TO_SPM=2};
__CONSTANTS__
struct Context{
 bool complete=true;int64_t batch_size=1,body_iter=0,seq_len=41,selector=2;int consumed=0;
 std::vector<int64_t>args;
 bool has_complete_physical_manifest()const{return complete;}
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selection,int64_t flags,const std::vector<int64_t>&a,int64_t invocation){
  assert(flags==0);TORCH_CHECK(a==args);
  if(family==FmbRouteFamily::GRAPH_SCHEDULE){assert(site==GR00T_DIT_TIME_TABLE_SCHEDULE_SITE);TORCH_CHECK(selection==selector&&invocation==0);}
  else{assert(site==GR00T_DIT_TIME_TABLE_DMA_SITE);TORCH_CHECK(selection==2&&invocation/2==body_iter);}
  ++consumed;
 }
};
struct Dma{int tag;int64_t off;};std::vector<Dma>dmas;int64_t table_steps=0;
void rpu_launch_ddr_broadcast_spm_dma_mutable(uint64_t&base,const at::Tensor&t,int64_t off,int64_t elements,uint32_t address,int cores){
 const bool tau=t.tag==1;const int64_t bytes=tau?122880:6144;
 assert(t.defined()&&base==(tau?0x80000000:0x90000000)&&off>=0&&off+elements*2<=table_steps*bytes&&elements*2==bytes&&cores==1);
 assert(address==(tau?0x10040000:0x10020000));dmas.push_back({t.tag,off});
}
struct Model{
 bool inference_started_=false,denoise_ready_=true,denoise_mode_=true,loop_mode_=true,z2_bound_=false;
 int64_t z2_prepared_seq_len_=0,num_steps_=4,action_horizon_=40,cross_dim_=2048,time_table_generation_=1,resolved=0;
 int64_t layers=32,h=1536,ff=6144,q=32,kv=32,hd=48,tp=8;
 at::Tensor time_tau_table_,time_po1_table_,adaln_table_;
 at::Tensor ae2t_w_{true},ae2t_ws_,po1_w_{true},po1_b_{true},po1_ws_;
 Model(int64_t steps=4,bool quantized=false){num_steps_=steps;ae2t_ws_.live=po1_ws_.live=quantized;}
 uint64_t time_tau_src_base_=0x80000000,time_po1_src_base_=0x90000000;
 std::vector<std::pair<at::Tensor,int64_t>>time_sources_;
 Context context;
 int64_t num_layers()const{return layers;}int64_t hidden_size()const{return h;}int64_t intermediate_size()const{return ff;}
 int64_t num_q_heads()const{return q;}int64_t num_kv_heads()const{return kv;}int64_t head_dim()const{return hd;}int64_t attn_tp()const{return tp;}
 int64_t get_last_resolved_chunk_size()const{return resolved;}
 Context&ctx(){return context;}uint32_t addr(int,const char*n)const{return std::string(n)=="enc2"?0x10040000:0x10020000;}
 __METHODS__
 void record(){context.args=time_table_arguments(context.seq_len,context.batch_size);context.selector=time_table_profile(context.seq_len,context.batch_size)?2:1;}
 void install(){time_tau_table_={true,1};time_po1_table_={true,2};record();}
};
int main(){
 for(int steps:{4,10})for(bool quantized:{false,true}){
 table_steps=steps;dmas.clear();Model model(steps,quantized);model.check_time_prepare_state();model.adaln_table_.live=true;model.check_time_prepare_state();
 {Model m4(steps,quantized);m4.tp=4;m4.check_time_prepare_state();}
 for(int bad=0;bad<12;++bad){Model x(steps,quantized);switch(bad){case 0:x.inference_started_=true;break;case 1:x.resolved=48;break;case 2:x.z2_bound_=true;break;case 3:x.z2_prepared_seq_len_=82;break;case 4:x.num_steps_=3;break;case 5:x.time_tau_table_.live=true;break;case 6:x.tp=2;break;case 7:x.ae2t_w_.p->v.valid=false;break;case 8:x.po1_b_.p->v.valid=false;break;case 9:x.action_horizon_=39;break;case 10:x.num_steps_=5;break;case 11:x.po1_ws_.live=true;x.po1_ws_.p->v.valid=false;break;}
  try{x.check_time_prepare_state();assert(false);}catch(const std::runtime_error&){}
 }
 RpuKernelGraph::active=true;try{model.check_time_prepare_state();assert(false);}catch(const std::runtime_error&){}RpuKernelGraph::active=false;
 model.install();
 for(int replay=0;replay<3;++replay)for(int body=0;body<steps;++body){model.context.body_iter=body;assert(model.time_tables_active());assert(model.load_time_table(true));assert(model.load_time_table(false));}
 assert(dmas.size()==6*steps);for(int i=0;i<6*steps;++i){assert(dmas[i].tag==1+i%2);assert(dmas[i].off==(i/2%steps)*(i%2?6144:122880));}
 assert(dmas[dmas.size()-2].off+122880==steps*122880&&dmas.back().off+6144==steps*6144);
 for(int bad=0;bad<8;++bad){Model x=model;switch(bad){case 0:++x.time_table_generation_;break;case 1:x.num_steps_=3;break;case 2:x.loop_mode_=false;break;case 3:x.context.batch_size=2;break;case 4:x.h=1024;break;case 5:x.context.selector=1;break;case 6:x.num_steps_=5;break;case 7:x.num_steps_=steps==4?10:4;break;}
  try{x.time_tables_active();assert(false);}catch(const std::runtime_error&){}
 }
 // The old manifest cannot authorize either table DMA after a 4<->10 switch.
 for(bool tau:{false,true}){Model stale=model;stale.num_steps_=steps==4?10:4;stale.context.body_iter=0;const auto before=dmas.size();
  try{stale.load_time_table(tau);assert(false);}catch(const std::runtime_error&){}assert(dmas.size()==before);}
 for(int bad=0;bad<7;++bad){Model x=model;switch(bad){case 0:x.loop_mode_=false;break;case 1:x.denoise_mode_=false;break;case 2:x.z2_bound_=true;break;case 3:x.z2_prepared_seq_len_=82;break;case 4:x.context.seq_len=48;break;case 5:x.context.complete=false;break;case 6:x.action_horizon_=39;break;}x.record();assert(!x.time_tables_active()&&!x.load_time_table(true)&&!x.load_time_table(false));}
 for(int body:{-1,steps})for(bool tau:{false,true}){model.context.body_iter=body;try{model.load_time_table(tau);assert(false);}catch(const std::runtime_error&){} }
 at::Tensor owner{true};model.time_sources_.push_back({owner,0});model.begin_time_inference();owner.p->v.value=1;
 try{model.begin_time_inference();assert(false);}catch(const std::runtime_error&){}
 owner.p->v.value=0;owner.p->v.valid=false;try{model.begin_time_inference();assert(false);}catch(const std::runtime_error&){}
 model.inference_started_=true;model.clear_time_tables();assert(!model.time_tau_table_.defined()&&!model.time_po1_table_.defined()&&model.time_sources_.empty());
 assert(model.time_tau_src_base_==0&&model.time_po1_src_base_==0&&model.inference_started_);
 try{model.check_time_prepare_state();assert(false);}catch(const std::runtime_error&){}
 }
}
'''.replace("__CONSTANTS__", constants).replace("__METHODS__", methods)
    _run(tmp_path, cpp)


def test_time_tables_ownership_and_original_fallback():
    source = SOURCE.read_text().replace("= {}", "= at::IntArrayRef()")
    prepare = _function(source, "    void prepare_time_tables(")
    assert "enter_physical(" in prepare and "exit_physical_noexcept(execution)" in prepare
    assert "RpuExecutionCleanupGuard" not in prepare
    assert prepare.index("check_time_prepare_state()") < prepare.index("rpu_gr00t_precompute_time_glue(") < prepare.index("time_tau_table_ =")
    assert "auto cond = cond_table_t_.clone()" in prepare and "auto tau = tau_table_.clone()" in prepare
    assert "tau_table_ = std::move(tau)" in prepare
    for entry in ("forward", "step_forward", "unroll_forward", "unroll_forward_z2"):
        body = _function(source, f"    at::Tensor {entry}(")
        assert body.index("begin_adaln_inference()") < body.index("begin_time_inference()") < body.index("TORCH_CHECK(")
    for setter in ("set_weights", "set_denoise_weights"):
        body = _function(source, f"    void {setter}(")
        assert "clear_time_tables()" in body and "inference_started_ = false" not in body
    declare = _function(source, "    std::vector<BufferDecl> baseline_buffer_declarations(")
    assert '&Gr00tDiTModel::tau_table_, /*route_invocation=*/0' in declare
    assert 'time_tau_table_' not in declare and 'time_po1_table_' not in declare
    pre = _function(source, "    void emit_pre_layers_body(")
    post = _function(source, "    void emit_post_layers_body(")
    assert "if (!time_tables)" in pre and "if (time_tables) load_time_table(/*tau=*/true)" in pre
    assert 'else sc_lin(tau_step, ae2t_w_, addr(0, "enc2"), AH, h, h, /*bias=*/0, ae2t_ws_)' in pre
    assert "if (!load_time_table(/*tau=*/false))" in post
    assert 'sc_lin(addr(0, "cond_full"), po1_w_' in post
    helper = _function(HELPER.read_text(), "std::tuple<at::Tensor, at::Tensor> rpu_gr00t_precompute_time_glue(")
    assert "ScopedTemporary" in helper and "reset_temporary" not in helper
    assert "input, tau_weight, result, horizon, h, h, 1, 1, 0, tau_scale" in helper
    assert "input, po1_weight, result, 1, output, h, 1, 1, bias, po1_scale" in helper
    scalar = "rpu_launch_eltwise_binary_scalar_spm_kernel(scale, c10::Half(1.0), scale, h, ValuOpType::ADD)"
    assert scalar in helper


def test_time_cold_emitter_layout_and_failure_cleanup(tmp_path):
    helper = _function(HELPER.read_text(), "std::tuple<at::Tensor, at::Tensor> rpu_gr00t_precompute_time_glue(")
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
#define TORCH_CHECK(ok, ...) do{if(!(ok))throw std::runtime_error(#ok);}while(false)
namespace c10{using Half=float;}
namespace at{
enum Type{kPrivateUse1,kHalf,kChar};struct Device{Type type()const{return kPrivateUse1;}bool operator==(const Device&)const{return true;}};
struct Tensor{
 bool live=false;Type dtype=kHalf;std::vector<int64_t>shape;int tag=0;
 std::shared_ptr<std::vector<float>>data=std::make_shared<std::vector<float>>();
 bool defined()const{return live;}Device device()const{return {};}
 Type scalar_type()const{return dtype;}bool is_contiguous()const{return true;}
 int64_t dim()const{return shape.size();}int64_t size(int i)const{return shape.at(i);}
 int64_t numel()const{int64_t n=1;for(auto s:shape)n*=s;return n;}
 int options()const{return 0;}template<class T>T*data_ptr()const{return data->data();}
};
Tensor empty(std::vector<int64_t>shape,int){Tensor t;t.live=true;t.shape=shape;t.data->resize(t.numel());return t;}
}
struct RpuKernelGraph{static bool active;static bool has_active(){return active;}};bool RpuKernelGraph::active=false;
struct RpuExecutionCoordinator{
 static int token;static void require_graph_quiescent(const char*){TORCH_CHECK(!RpuKernelGraph::active);}
 static int current_physical_token_if_owned(const char*){return token;}
};int RpuExecutionCoordinator::token=1;
struct Allocator{
 static constexpr int64_t SPM_PLANNING_BUDGET=8303708;int64_t used=102400;
 bool is_initialized()const{return true;}void init(){}int64_t total_used()const{return used;}int64_t free_space()const{return SPM_PLANNING_BUDGET-used;}
 uint32_t alloc_temporary(int64_t n){assert(n%256==0);auto o=used;used+=n;return o;}
 uint32_t addr(int,uint32_t off){return 0x10000000+off;}
};Allocator SPM_ALLOC;
struct SpmAllocator:Allocator{struct ScopedTemporary{int64_t mark=SPM_ALLOC.used;~ScopedTemporary(){SPM_ALLOC.used=mark;}};};
struct ScopedEagerAtomicBatch{ScopedEagerAtomicBatch(bool,int){}void submit(){}};
void rpu_ddr_flush_force(float*){}
void rpu_ddr_flush(float*){}
enum class ValuOpType{ADD};
std::map<uint32_t,std::vector<float>>spm;
int linear_calls=0,copy_calls=0,scalar_calls=0,fail_at=-1;int64_t tau_copy_end=0,po1_copy_end=0;
void rpu_launch_ddr_broadcast_spm_dma(const at::Tensor&t,int64_t off,int64_t elems,uint32_t dst,int cores){
 assert(off>=0&&off+elems<=t.numel()&&cores==1);spm[dst].assign(t.data->begin()+off,t.data->begin()+off+elems);
}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t input,const at::Tensor&w,uint32_t out,int64_t m,int64_t n,int64_t k,int partition,int cores,uint32_t bias,const at::Tensor&scale){
 const bool tau=w.tag==1;assert(m==(tau?40:1)&&n==(tau?1536:3072)&&k==1536&&partition==1&&cores==1);
 assert(spm[input].size()==m*k&&input!=out&&out!=bias);assert(tau?bias==0:spm[bias].size()==3072);
 assert(w.dtype==at::kChar?scale.live:!scale.live);
 if(++linear_calls==fail_at)throw std::runtime_error("injected kernel failure");
 spm[out].resize(m*n);for(int i=0;i<m*n;++i)spm[out][i]=spm[input][i%(m*k)]+w.tag+(tau?0:spm[bias][i]);
}
void rpu_launch_eltwise_binary_scalar_spm_kernel(uint32_t input,float value,uint32_t out,int64_t n,ValuOpType){
 assert(input==out&&value==1&&n==1536);auto base=out-1536*2;assert(spm[base].size()==3072);
 for(int i=1536;i<3072;++i)spm[base][i]+=value;++scalar_calls;
}
void rpu_launch_spm_copy_ddr_dma(uint32_t input,const at::Tensor&out,int64_t off,int64_t n){
 assert(off>=0&&off+n<=out.numel()&&n==(out.dim()==3?61440:3072));
 assert(off==(copy_calls/2)*n);std::copy(spm[input].begin(),spm[input].end(),out.data->begin()+off);
 (out.dim()==3?tau_copy_end:po1_copy_end)=off+n;++copy_calls;
}
__HELPER__
int main(){
 auto tw=at::empty({1536,1536},0),pw=at::empty({3072,1536},0),pb=at::empty({3072},0);tw.tag=1;pw.tag=2;
 for(int i=0;i<3072;++i)(*pb.data)[i]=i%13;at::Tensor ts,ps;
 for(int steps:{4,10})for(bool quantized:{false,true}){
  auto tau=at::empty({steps,40,1536},0),cond=at::empty({steps,1536},0);
  for(int s=0;s<steps;++s){for(int i=0;i<61440;++i)(*tau.data)[s*61440+i]=100*s+i%40;for(int i=0;i<1536;++i)(*cond.data)[s*1536+i]=10*s+i%7;}
  auto run=[&](){return rpu_gr00t_precompute_time_glue(tau,cond,tw,ts,pw,pb,ps);};
  tw.dtype=pw.dtype=quantized?at::kChar:at::kHalf;ts=quantized?at::empty({1536},0):at::Tensor{};ps=quantized?at::empty({3072},0):at::Tensor{};
  linear_calls=copy_calls=scalar_calls=0;auto tables=run();assert(SPM_ALLOC.used==102400&&linear_calls==steps*2&&copy_calls==steps*2&&scalar_calls==steps);
  const auto&t=std::get<0>(tables);const auto&p=std::get<1>(tables);assert(t.shape==tau.shape&&p.shape==std::vector<int64_t>({steps,3072}));
  assert(tau_copy_end==t.numel()&&po1_copy_end==p.numel());
  for(int i=0;i<t.numel();++i)assert((*t.data)[i]==(*tau.data)[i]+1);
  for(int s=0;s<steps;++s)for(int i=0;i<3072;++i)assert((*p.data)[s*3072+i]==(*cond.data)[s*1536+i%1536]+2+i%13+(i>=1536?1:0));
 for(int failure:{5,steps*2}){linear_calls=copy_calls=0;fail_at=failure;try{run();assert(false);}catch(const std::runtime_error&){}
  assert(SPM_ALLOC.used==102400&&linear_calls==failure&&copy_calls==failure-1);fail_at=-1;}
 for(int bad=0;bad<8;++bad){const auto start=SPM_ALLOC.used;
  if(bad==0)tau.shape={steps,39,1536};if(bad==1)RpuExecutionCoordinator::token=0;if(bad==2)RpuKernelGraph::active=true;
  if(bad==3)SPM_ALLOC.used=Allocator::SPM_PLANNING_BUDGET-251903;if(bad==4)ps.shape={1536};
  if(bad==5){tau.shape={3,40,1536};cond.shape={3,1536};}if(bad==6){tau.shape={5,40,1536};cond.shape={5,1536};}
  if(bad==7)cond.shape={steps==4?10:4,1536};
  if(bad==4&&!quantized)ps.live=true;
  const auto before=SPM_ALLOC.used;try{run();assert(false);}catch(const std::runtime_error&){}assert(SPM_ALLOC.used==before);
  SPM_ALLOC.used=start;RpuExecutionCoordinator::token=1;RpuKernelGraph::active=false;tau.shape={steps,40,1536};cond.shape={steps,1536};ps.shape={3072};ps.live=quantized;
 }
 }
}
'''.replace("__HELPER__", helper)
    _run(tmp_path, cpp)
