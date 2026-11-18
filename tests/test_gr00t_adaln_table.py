"""Host execution of GR00T's native table admission, routes and cold emitter."""
from pathlib import Path
import re
import shutil
import subprocess

from test_siglip_attention_dry_live_contract import _function

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/fused/rpu_gr00t_dit_model.cpp"


def _run(tmp_path, source):
    cpp, exe = tmp_path / "adaln.cpp", tmp_path / "adaln"
    cpp.write_text(source)
    result = subprocess.run([shutil.which("c++"), "-std=c++17", "-O0", str(cpp), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_adaln_cold_boundary_body_replay_and_descriptor_drift(tmp_path):
    source = SOURCE.read_text()
    methods = "\n".join(_function(source, marker) for marker in (
        "    void check_adaln_prepare_state(", "    void clear_adaln_table(",
        "    void begin_adaln_inference(", "    bool adaln_table_profile(",
        "    int64_t adaln_table_body_bytes(", "    std::vector<int64_t> adaln_table_arguments(",
        "    uint32_t adaln_layer_scale(",
    ))
    dma = _function(source, "    bool load_adaln_body_table(")
    constants = "\n".join(re.findall(r"constexpr int64_t GR00T_DIT_ADALN_TABLE_\w+ = \d+LL;", source))
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
namespace at{struct Tensor{bool live=false;std::shared_ptr<Impl>p=std::make_shared<Impl>();bool defined()const{return live;}Impl*unsafeGetTensorImpl()const{return p.get();}};}
struct RpuKernelGraph{static bool active;static bool has_active(){return active;}};bool RpuKernelGraph::active=false;
enum class FmbRouteFamily{GRAPH_SCHEDULE,MUTABLE_DMA};
enum class Gr00tDiTMutableDmaRoute:int64_t{DDR_BROADCAST_TO_SPM=2};
__CONSTANTS__
struct ChunkInfo{int64_t len=41,idx=0;};
struct Context{
 bool complete=true;int64_t batch_size=1,body_iter=0,seq_len=41,selector=2;int consumed=0;
 std::vector<int64_t>args;
 bool has_complete_physical_manifest()const{return complete;}
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selection,int64_t flags,const std::vector<int64_t>&a,int64_t invocation){
  assert(flags==0);TORCH_CHECK(a==args);
  if(family==FmbRouteFamily::GRAPH_SCHEDULE){assert(site==GR00T_DIT_ADALN_TABLE_SCHEDULE_SITE);TORCH_CHECK(selection==selector&&invocation==0);}
  else{assert(site==GR00T_DIT_ADALN_TABLE_DMA_SITE);TORCH_CHECK(selection==2&&invocation==body_iter);}
  ++consumed;
 }
};
std::vector<int64_t>dma_offsets;int64_t table_steps=0;
void rpu_launch_ddr_broadcast_spm_dma_mutable(uint64_t&,const at::Tensor&t,int64_t off,int64_t elements,uint32_t addr,int cores){
 assert(t.defined()&&off>=0&&off+elements*2<=table_steps*196608&&elements==98304&&addr==0x10040000&&(cores==4||cores==8));dma_offsets.push_back(off);
}
struct Model{
 bool inference_started_=false,denoise_ready_=true,denoise_mode_=true,loop_mode_=true,z2_bound_=false;
 int64_t z2_prepared_seq_len_=0,num_steps_=4,action_horizon_=40,cross_dim_=2048,adaln_table_generation_=1,resolved=0;
 int64_t layers=32,h=1536,ff=6144,q=32,kv=32,hd=48,tp=8;
 at::Tensor adaln_table_;uint64_t adaln_table_src_base_=0;
 std::vector<std::pair<at::Tensor,int64_t>>adaln_sources_;
 struct Layer{at::Tensor adaln_w{true},adaln_b{true},adaln_ws;};std::vector<Layer>layer_weights_=std::vector<Layer>(32);
 Model(int64_t steps=4,bool quantized=false){num_steps_=steps;for(auto&layer:layer_weights_)layer.adaln_ws.live=quantized;}
 Context context;int gemvs=0;
 int64_t num_layers()const{return layers;}int64_t hidden_size()const{return h;}int64_t intermediate_size()const{return ff;}
 int64_t num_q_heads()const{return q;}int64_t num_kv_heads()const{return kv;}int64_t head_dim()const{return hd;}int64_t attn_tp()const{return tp;}
 int64_t get_last_resolved_chunk_size()const{return resolved;}
 Context&ctx(){return context;}uint32_t addr(int,const char*n)const{return std::string(n)=="adaln_body_table"?0x10040000:0x10010000;}
 template<class...T>void dit_adaln_gemv(T...){++gemvs;}
 __METHODS__
 int num_cores()const{return static_cast<int>(tp);}
 __DMA__
 void preload(){load_adaln_body_table();}
 void record(const ChunkInfo&c){context.args=adaln_table_arguments(c.len,context.batch_size);context.selector=adaln_table_profile(c.len,context.batch_size)?2:1;}
};
int main(){
 for(int steps:{4,10})for(bool quantized:{false,true}){
 table_steps=steps;dma_offsets.clear();Model model(steps,quantized);ChunkInfo chunk;model.check_adaln_prepare_state();
 {Model m4(steps,quantized);m4.tp=4;m4.check_adaln_prepare_state();}
 for(int bad=0;bad<10;++bad){Model x(steps,quantized);switch(bad){case 0:x.inference_started_=true;break;case 1:x.resolved=48;break;case 2:x.z2_bound_=true;break;case 3:x.z2_prepared_seq_len_=82;break;case 4:x.num_steps_=3;break;case 5:x.adaln_table_.live=true;break;case 6:x.tp=2;break;case 7:x.layer_weights_[0].adaln_w.p->v.valid=false;break;case 8:x.num_steps_=5;break;case 9:x.layer_weights_[0].adaln_ws.live=true;x.layer_weights_[0].adaln_ws.p->v.valid=false;break;}
  try{x.check_adaln_prepare_state();assert(false);}catch(const std::runtime_error&){}
 }
 RpuKernelGraph::active=true;try{model.check_adaln_prepare_state();assert(false);}catch(const std::runtime_error&){}RpuKernelGraph::active=false;
 model.begin_adaln_inference();model.clear_adaln_table();try{model.check_adaln_prepare_state();assert(false);}catch(const std::runtime_error&){}
 model.adaln_table_.live=true;model.record(chunk);
 for(int replay=0;replay<3;++replay)for(int body=0;body<steps;++body){model.context.body_iter=body;model.preload();
  for(int layer=0;layer<32;++layer){auto a=model.adaln_layer_scale(layer,chunk);assert(a==0x10040000+layer*6144&&a+6144<=0x10040000+196608);}}
 assert(model.gemvs==0&&dma_offsets.size()==3*steps);for(int i=0;i<3*steps;++i)assert(dma_offsets[i]==(i%steps)*196608);
 assert(dma_offsets.back()+196608==steps*196608);
 for(int bad=0;bad<8;++bad){Model x=model;switch(bad){case 0:++x.adaln_table_generation_;break;case 1:x.num_steps_=3;break;case 2:x.loop_mode_=false;break;case 3:x.context.batch_size=2;break;case 4:x.h=1024;break;case 5:x.context.selector=1;break;case 6:x.num_steps_=5;break;case 7:x.num_steps_=steps==4?10:4;break;}
  try{x.adaln_layer_scale(0,chunk);assert(false);}catch(const std::runtime_error&){}
 }
 // A valid new step count must still reject the old descriptor before DMA.
 {Model stale=model;stale.num_steps_=steps==4?10:4;stale.context.body_iter=0;const auto before=dma_offsets.size();
  try{stale.preload();assert(false);}catch(const std::runtime_error&){}assert(dma_offsets.size()==before);}
 for(int bad=0;bad<4;++bad){Model x=model;switch(bad){case 0:x.loop_mode_=false;break;case 1:x.denoise_mode_=false;break;case 2:x.z2_bound_=true;break;case 3:x.z2_prepared_seq_len_=82;break;}x.record(chunk);assert(x.adaln_layer_scale(0,chunk)==0x10010000&&x.gemvs==1);}
 for(int layer:{-1,32})try{model.adaln_layer_scale(layer,chunk);assert(false);}catch(const std::runtime_error&){}
 for(int body:{-1,steps}){model.context.body_iter=body;try{model.preload();assert(false);}catch(const std::runtime_error&){} }
 model.context.complete=false;assert(model.adaln_layer_scale(0,chunk)==0x10010000&&model.gemvs==1);
 at::Tensor owner{true};model.adaln_sources_.push_back({owner,0});model.begin_adaln_inference();owner.p->v.value=1;
 try{model.begin_adaln_inference();assert(false);}catch(const std::runtime_error&){}
 owner.p->v.value=0;owner.p->v.valid=false;try{model.begin_adaln_inference();assert(false);}catch(const std::runtime_error&){}
 model.clear_adaln_table();assert(!model.adaln_table_.defined()&&model.adaln_sources_.empty()&&model.inference_started_);
 }
}
'''.replace("__CONSTANTS__", constants).replace("__METHODS__", methods).replace("__DMA__", dma)
    _run(tmp_path, cpp)


def test_adaln_table_preparation_has_owned_execution_and_shared_arithmetic():
    source = SOURCE.read_text().replace("= {}", "= at::IntArrayRef()")
    prepare = _function(source, "    void prepare_adaln_table(")
    assert "enter_physical(" in prepare and "exit_physical_noexcept(execution)" in prepare
    assert "RpuExecutionCleanupGuard" not in prepare
    assert prepare.index("check_adaln_prepare_state()") < prepare.index("rpu_gr00t_precompute_adaln(")
    assert prepare.index("rpu_gr00t_precompute_adaln(") < prepare.index("adaln_table_ =")
    for entry in ("forward", "step_forward", "unroll_forward", "unroll_forward_z2"):
        body = _function(source, f"    at::Tensor {entry}(")
        assert body.index("begin_adaln_inference()") < body.index("TORCH_CHECK(")
    for setter in ("set_weights", "set_denoise_weights"):
        body = _function(source, f"    void {setter}(")
        assert "clear_adaln_table()" in body and "inference_started_ = false" not in body
    helper = (ROOT / "src/ops/rpu_gr00t_adaln.cpp").read_text()
    assert "ScopedTemporary" in helper and "reset_temporary" not in helper
    assert "current_physical_token_if_owned" in helper and "!RpuKernelGraph::has_active()" in helper
    operators = ("rpu_launch_ddr_broadcast_spm_dma(", "rpu_launch_linear_spm_to_spm_acc16_kernel(",
                 "rpu_launch_all_reduce_sum_residual_kernel(", "rpu_launch_eltwise_binary_scalar_spm_kernel(")
    hot = _function(source, "    void dit_adaln_gemv(")
    assert [helper.index(op) for op in operators] == sorted(helper.index(op) for op in operators)
    assert [hot.index(op) for op in operators] == sorted(hot.index(op) for op in operators)
    assert "1, output, h, 0, cores, 0, scales[layer]" in helper
    assert "(step * layers + layer) * output" in helper


def test_adaln_cold_emitter_bounds_and_scratch_cleanup(tmp_path):
    helper = _function((ROOT / "src/ops/rpu_gr00t_adaln.cpp").read_text(),
                       "at::Tensor rpu_gr00t_precompute_adaln(")
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
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
using TensorList=std::vector<Tensor>;
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
void rpu_ddr_flush_force(float*){}
enum class ValuOpType{ADD};
std::map<uint32_t,std::vector<float>>spm;
int step=-1,linear_calls=0,copy_calls=0,fail_at=-1;int64_t last_copy_end=0;
void rpu_launch_ddr_scatter_spm_dma(const at::Tensor&t,int64_t off,int64_t elems,int64_t stride,uint32_t dst,int cores){
 assert((cores==4||cores==8)&&off>=0&&off%1536==0&&off+elems+(cores-1)*stride/2<=t.numel()&&elems==1536/cores&&stride==elems*2);step=off/1536;spm[dst].assign(elems,step);
}
void rpu_launch_ddr_broadcast_spm_dma(const at::Tensor&t,int64_t off,int64_t elems,uint32_t dst,int cores){
 assert(off==0&&elems==3072&&(cores==4||cores==8));spm[dst].assign(elems,t.tag);
}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t input,const at::Tensor&w,uint32_t out,int64_t m,int64_t n,int64_t k,int partition,int cores,int bias,const at::Tensor&scale){
 assert(m==1&&n==3072&&k==1536&&partition==0&&(cores==4||cores==8)&&bias==0&&static_cast<int64_t>(spm[input].size())==1536/cores);
 if(++linear_calls==fail_at)throw std::runtime_error("injected kernel failure");
 assert(w.dtype==at::kChar?scale.live:!scale.live);spm[out].assign(n,100*step+w.tag);
}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t partial,uint32_t bias,uint32_t out,int64_t m,int64_t n,int incores,int outcores){
 assert(partial!=bias&&partial!=out&&bias!=out&&m==1&&n==3072&&(incores==4||incores==8)&&outcores==incores);
 spm[out].resize(n);for(int i=0;i<n;++i)spm[out][i]=spm[partial][i]+spm[bias][i];
}
void rpu_launch_eltwise_binary_scalar_spm_kernel(uint32_t in,float value,uint32_t out,int64_t n,ValuOpType,int cores=8){assert(in==out&&value==1&&n==1536&&(cores==4||cores==8));for(int i=0;i<n;++i)spm[out][i]+=value;}
void rpu_launch_spm_copy_ddr_dma(uint32_t in,const at::Tensor&out,int64_t off,int64_t n){
 assert(n==3072&&off==copy_calls*3072&&off+n<=out.numel());std::copy(spm[in].begin(),spm[in].end(),out.data->begin()+off);last_copy_end=off+n;++copy_calls;
}
__HELPER__
int main(){
 at::TensorList weights(32),biases(32),scales(32);
 for(int l=0;l<32;++l){weights[l].live=true;weights[l].shape={3072,1536};weights[l].tag=l;
  biases[l].live=true;biases[l].shape={3072};biases[l].tag=1000+l;}
 for(int cores:{4,8})for(int steps:{4,10})for(bool quantized:{false,true}){
  at::Tensor cond=at::empty({steps,1536},0);
  for(int l=0;l<32;++l){weights[l].dtype=quantized?at::kChar:at::kHalf;scales[l].live=quantized;scales[l].shape={3072};}
  linear_calls=copy_calls=0;auto table=rpu_gr00t_precompute_adaln(cond,weights,biases,scales,cores);
  assert(SPM_ALLOC.used==102400&&linear_calls==steps*32&&copy_calls==steps*32&&table.numel()==steps*98304);
  assert(table.shape==std::vector<int64_t>({steps,32,3072})&&last_copy_end==table.numel());
  for(int s=0;s<steps;++s)for(int l=0;l<32;++l)for(int i=0;i<3072;++i)
   assert((*table.data)[(s*32+l)*3072+i]==100*s+1000+2*l+(i<1536?1:0));
 for(int failure:{17,steps*32}){linear_calls=copy_calls=0;fail_at=failure;try{rpu_gr00t_precompute_adaln(cond,weights,biases,scales,cores);assert(false);}catch(const std::runtime_error&){}
  assert(SPM_ALLOC.used==102400&&linear_calls==failure&&copy_calls==failure-1);fail_at=-1;}
 for(int bad=0;bad<6;++bad){auto c=cond;const auto start=SPM_ALLOC.used;int bad_cores=cores;
  if(bad==0)c.shape={3,1536};if(bad==1)c.shape={5,1536};if(bad==2)RpuExecutionCoordinator::token=0;if(bad==3)RpuKernelGraph::active=true;if(bad==4)SPM_ALLOC.used=Allocator::SPM_PLANNING_BUDGET-(cores==4?19199:18943);if(bad==5)bad_cores=6;
  const auto before=SPM_ALLOC.used;try{rpu_gr00t_precompute_adaln(c,weights,biases,scales,bad_cores);assert(false);}catch(const std::runtime_error&){}
  assert(SPM_ALLOC.used==before);SPM_ALLOC.used=start;RpuExecutionCoordinator::token=1;RpuKernelGraph::active=false;
 }
 }
}
'''.replace("__HELPER__", helper)
    _run(tmp_path, cpp)
