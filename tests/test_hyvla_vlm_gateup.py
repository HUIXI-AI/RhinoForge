"""Execute the HyVLA M240 fused FP16 launcher and cold manifest contracts."""
import re

import pytest

from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run

DECODER = (ROOT / "src/fused/rpu_hyvla_vlm_model.cpp").read_text()


@pytest.mark.parametrize("profile", ["vlm", "expert"])
def test_actual_launcher_owned_ddr_registers_spm_bounds_and_grid(tmp_path, profile):
    cpp = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    launch = block(cpp, "template <int64_t Rows, int64_t Hidden, int64_t Intermediate>")
    launch += "\n" + block(cpp, f"void rpu_launch_hyvla_{profile}_gateup_swiglu_fp16_spm_kernel(")
    source = COMMON + r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId : uint16_t {WALL_PREFILL_GATE_UP_SWIGLU_FP16_M320N112K128};
enum class GraphDdrRegisterRole {ModelWeight};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;
 GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel {std::array<uint16_t,68> regs{};
 void reset_regs(){regs.fill(0);}
 uint32_t set_regs(uint32_t i,uint16_t v){regs.at(i)=v;return 0;}
 uint32_t set_regs(uint32_t,int32_t)=delete;
} kernel;
int staged=0,queued=0,flushed=0;
struct Writer {std::vector<GraphDdrRegisterOperandSpec> operands;
 void write(Kernel& k){for(const auto& o:operands){auto a=o.tensor.address>>8;
  assert(k.set_regs(o.abi.lo,uint16_t(a))==0);assert(k.set_regs(o.abi.hi,uint16_t(a>>16))==0);}}
};
struct RpuKernelGraph {
 static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId,std::vector<GraphDdrRegisterOperandSpec> v){
  ++staged;assert(v.size()==2);
  for(auto& o:v){assert(o.abi.role==GraphDdrRegisterRole::ModelWeight);
   assert(o.abi.access==GraphDdrRegisterAccess::Read);
   assert(o.abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
   assert(o.tensor.nbytes()==6144*2048*2);}
  assert(v[0].abi.lo==2 && v[0].abi.hi==3 && v[1].abi.lo==32 && v[1].abi.hi==33);
  return {v};}
};
struct Queue {bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(broadcast && grid==std::vector<uint16_t>({10,1,1}));
  assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++queued;}
} queue;
#define GET_KERNEL(id) (&::kernel)
#define GET_QUEUE(n) (&::queue)
void rpu_ddr_flush(void*){++flushed;}
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;
 uint32_t base=0x20000000;bool is_initialized()const{return ready;}
 uint32_t addr(int c,uint32_t off)const{return base+uint32_t(c)*0x1000000u+off;}
} spm;
#define SPM_ALLOC spm
LAUNCH
int main(){
 at::Tensor gate,up;gate.shape=up.shape={6144,2048};up.address=0x18000000;
 auto run=[&](at::Tensor g,at::Tensor u,uint32_t in,uint32_t out,bool expected,int rows=240){
  const int old=queued,before=staged,flush_before=flushed;bool accepted=true;
  try{rpu_launch_hyvla_vlm_gateup_swiglu_fp16_spm_kernel(in,g,u,out,rows);}
  catch(const std::exception&){accepted=false;}
  assert(accepted==expected && queued==old+int(expected));
  assert(staged==before+int(expected) && flushed==flush_before+2*int(expected));
 };
 run(gate,up,0x20000000,0x20200000,true);
 auto pair=[&](int i){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+1])<<16);};
 assert(pair(0)==0x20000000 && pair(4)==0x20200000);
 assert(pair(2)==gate.address>>8 && pair(32)==up.address>>8);
 assert(kernel.regs[8]==1 && kernel.regs[9]==240 && pair(10)==768 && kernel.regs[12]==2048 && kernel.regs[13]==128);
 assert(kernel.regs[14]==0 && kernel.regs[15]==8 && pair(16)==4096 && pair(18)==1536);
 assert(pair(20)==0 && pair(22)==0 && pair(24)==0 && kernel.regs[26]==2 && kernel.regs[27]==2 && pair(28)==0);
 for(int rows:{0,51,239,241,320,640})run(gate,up,0x20000000,0x20200000,false,rows);
 for(auto in:{0x1fffff00u,0x20000001u,0x207fff00u})run(gate,up,in,0x20200000,false);
 for(auto out:{0x1fffff00u,0x20200001u,0x207fff00u})run(gate,up,0x20000000,out,false);
 run(gate,up,0x20000000,0x20010000,false);
 run(gate,up,0x20200000,0x20200000-256,false);
 run(gate,up,0x20000000,0x20000000+240*2048*2,true);
 run(gate,up,0x20000000,spm.base+SpmAllocator::SPM_USABLE-240*768*2,true);
 run(gate,up,0x20000000,spm.base+SpmAllocator::SPM_USABLE-240*768*2+256,false);
 spm.ready=false;run(gate,up,0x20000000,0x20200000,false);spm.ready=true;
 spm.base=0xf8ff0000;run(gate,up,0xf8ff0000,0xf91f0000,false);spm.base=0x20000000;
 for(bool first:{false,true})for(int failure=0;failure<8;++failure){auto bad=gate;
  switch(failure){case 0:bad.present=false;break;case 1:bad.contiguous=false;break;
   case 2:bad.dtype=at::kChar;break;case 3:bad.where.kind=c10::DeviceType::CPU;break;
   case 4:bad.shape={6144,2047};break;case 5:bad.shape={6143,2048};break;
   case 6:bad.shape={1,6144,2048};break;case 7:bad.address+=128;break;}
  run(first?bad:gate,first?up:bad,0x20000000,0x20200000,false);
 }
 up.where.index=1;run(gate,up,0x20000000,0x20200000,false);
}
'''
    if profile == "expert":
        source = source.replace("hyvla_vlm_", "hyvla_expert_")
        source = source.replace("({10,1,1})", "({3,1,1})")
        source = source.replace("{0,51,239,241,320,640}", "{0,50,52,64,320,640}")
        source = source.replace("kernel.regs[13]==128", "kernel.regs[13]==64")
        replacements = {"6144":"2048", "2048":"1024", "768":"256", "240":"51",
                        "4096":"2048", "1536":"512", "6143":"2047", "2047":"1023"}
        source = re.sub(r"\b(?:" + "|".join(replacements) + r")\b",
                        lambda m: replacements[m.group()], source)
    compile_run(tmp_path, profile + "_gateup_launch", source.replace("LAUNCH", launch))


def test_actual_owner_eligibility_and_complete_manifest(tmp_path):
    markers = (
        "bool HyVlaVlmModel::gateup_swiglu_enabled(",
        "bool HyVlaVlmModel::row_merge_enabled(",
        "bool HyVlaVlmModel::vision_gu176_enabled(",
        "std::vector<int64_t> HyVlaVlmModel::vision_gu176_arguments(",
        "bool HyVlaVlmModel::paired_rmsnorm_enabled(",
        "std::vector<int64_t> HyVlaVlmModel::paired_rmsnorm_arguments(",
        "std::vector<int64_t> HyVlaVlmModel::row_merge_arguments(",
        "std::vector<int64_t> HyVlaVlmModel::gateup_swiglu_arguments(",
        "std::vector<int64_t> HyVlaVlmModel::prefill_kv_only_arguments(",
        "FmbPhysicalManifestForwardCapability\nHyVlaVlmModel::physical_manifest_forward_capability(",
    )
    methods = "\n".join(block(DECODER, x).replace("HyVlaVlmModel::", "") for x in markers)
    constants = "\n".join(re.findall(r"constexpr int64_t HYVLA_VLM_\w+_SITE = \d+LL;", DECODER))
    constants += "\n" + re.search(r"constexpr int64_t kHyVlaVlmRmsNormFullRowsSite = \d+LL;", DECODER).group()
    header = (ROOT / "src/fused/rpu_hyvla_vlm_model.h").read_text()
    identity = block(header, "std::vector<int64_t> kvinsert_cost_weight_identity() const override").replace(" override", "")
    rmsnorm = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    rms_contract = rmsnorm[rmsnorm.index("enum RpuRmsNormCapability"):
                          rmsnorm.index("// Matches LlamaGenerator's Newton scheduling")]
    source = COMMON + constants + rms_contract + r'''
constexpr unsigned MOT_NOMERGE_QKV=1,MOT_NOMERGE_MLP=2;
enum class FmbRouteFamily {GRAPH_SCHEDULE,LINEAR,ACTIVATION};
enum class FmbLinearRouteSelector {AUTO_TILE=1};
enum class HyVlaVlmActivationRoute {SILU_UNARY=1,MUL=2,SILU_MUL_FUSED=3};
enum class FmbGraphLifecycle {COMPOSITE_CHILD};
struct Route {int64_t site_id;FmbRouteFamily family;int64_t selector,flags;
 std::vector<int64_t> arguments;int64_t invocation;};
struct FmbPhysicalExecutionManifest {int64_t logical_length=240,physical_length=240,kv_logical_length=240;
 std::vector<Route> routes;};
struct FmbPhysicalManifestForwardCapability {bool supported=false;FmbGraphLifecycle lifecycle;};
struct Model {
 struct LayerW {at::Tensor q_w,k_w,v_w,o_w,gate_w,up_w,down_w;
  at::Tensor q_ws,k_ws,v_ws,o_ws,gate_ws,up_ws,down_ws;
  at::Tensor q_norm_w,k_norm_w,input_norm_w,post_norm_w,q_bias,k_bias,v_bias;};
 bool cold_gateup_swiglu_available_=true,cold_prefill_kv_only_=true,cold_silu_mul_=false;
 bool cold_w8_optimizations_available_=false,cold_w8_vlm_gateup_available_=false;
 bool cold_vision_gu176_available_=false,standard_three_camera_mask_=false;
 bool cold_row_merge_available_=false,cold_paired_rmsnorm_available_=false;unsigned cold_mot_norm_nomerge_=3;
 bool cold_paired_rmsnorm_v32_available_=false;
 bool cold_fast_replay_=true,cold_fast_replay_preload_=true,cold_mask_once_=true,cold_partial_rope_=true;
 struct Contract {bool vector=true;uint32_t capability=RPU_RMSNORM_CAP_ALL;
  RpuRmsNormSpmRoute resolve(int64_t rows,int64_t cols)const{
  return rpu_resolve_rmsnorm_spm_route(vector?capability:RPU_RMSNORM_CAP_BASE,rows,cols);}} cold_rmsnorm_spm_contract_;
 double eps_hyvla_=1e-5;
 int64_t moe_chunk_size_=240;int l=32,h=2048,i=6144,q=16,kv=8,d=128,tp=8;
 int num_layers()const{return l;}int hidden_size()const{return h;}int intermediate_size()const{return i;}
 int num_q_heads()const{return q;}int num_kv_heads()const{return kv;}int head_dim()const{return d;}
 int attn_tp()const{return tp;}
 std::vector<LayerW> text_layers_{32},moe_layers_{32};
 at::Tensor final_norm_text_;
 std::vector<int64_t> causal_kvinsert_cost_weight_identity()const{return {1};}
 static void append_kvinsert_cost_scalar_identity(std::vector<int64_t>& v,double x){v.push_back(c10::Half(x).x);}
 static void append_kvinsert_cost_tensor_identity(std::vector<int64_t>& v,const at::Tensor& t){v.push_back(t.address);}
 Model(){for(auto* tower:{&text_layers_,&moe_layers_})for(auto& w:*tower)
  for(auto* s:{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws})s->present=false;}
 METHODS
 IDENTITY
};
int main(){
 Model m;assert(m.gateup_swiglu_enabled(240,0));
 assert(m.gateup_swiglu_arguments()==std::vector<int64_t>({3,32,31,2,240,0,2048,6144,768,8,320,80,128,10,0,2,32,20,34,26,27,28,29}));
 for(bool* flag:{&m.cold_gateup_swiglu_available_,&m.cold_prefill_kv_only_}){
  *flag=false;assert(!m.gateup_swiglu_enabled(240,0));*flag=true;}
 m.cold_silu_mul_=true;assert(!m.gateup_swiglu_enabled(240,0));m.cold_silu_mul_=false;
 for(int* dim:{&m.l,&m.h,&m.i,&m.q,&m.kv,&m.d,&m.tp}){++*dim;assert(!m.gateup_swiglu_enabled(240,0));--*dim;}
 m.moe_chunk_size_=224;assert(!m.gateup_swiglu_enabled(240,0));m.moe_chunk_size_=240;
 assert(!m.gateup_swiglu_enabled(224,0) && !m.gateup_swiglu_enabled(240,16));
 for(auto* tower:{&m.text_layers_,&m.moe_layers_}){
  for(auto* w:{&tower->back().gate_w,&tower->front().q_w}){
   w->dtype=at::kChar;assert(!m.gateup_swiglu_enabled(240,0));w->dtype=at::kHalf;
   w->present=false;assert(!m.gateup_swiglu_enabled(240,0));w->present=true;}
  tower->back().up_ws.present=true;assert(!m.gateup_swiglu_enabled(240,0));tower->back().up_ws.present=false;
 }
 FmbPhysicalExecutionManifest manifest;
 manifest.routes={{HYVLA_VLM_PREFILL_KV_ONLY_SITE,FmbRouteFamily::GRAPH_SCHEDULE,1,0,m.prefill_kv_only_arguments(240),0},
  {HYVLA_VLM_GATEUP_SWIGLU_SITE,FmbRouteFamily::GRAPH_SCHEDULE,1,0,m.gateup_swiglu_arguments(),0}};
 auto admitted=[&](const FmbPhysicalExecutionManifest& v){try{return m.physical_manifest_forward_capability(v).supported;}
  catch(const std::exception&){return false;}};
 assert(admitted(manifest));
 auto v=manifest;v.routes.pop_back();assert(!admitted(v));
 v=manifest;v.routes.push_back(v.routes.back());assert(!admitted(v));
 for(size_t a=0;a<manifest.routes.back().arguments.size();++a){v=manifest;++v.routes.back().arguments[a];assert(!admitted(v));}
 v=manifest;v.routes.back().family=FmbRouteFamily::LINEAR;assert(!admitted(v));
 v=manifest;++v.routes.back().flags;assert(!admitted(v));
 v=manifest;++v.routes.back().selector;assert(!admitted(v));
 v=manifest;++v.routes.back().invocation;assert(!admitted(v));
 for(int64_t* dim:{&manifest.logical_length,&manifest.physical_length,&manifest.kv_logical_length}){
  ++*dim;assert(!admitted(manifest));--*dim;}
 for(auto site:{HYVLA_VLM_MLP_GATE_LINEAR_SITE,HYVLA_VLM_MLP_UP_LINEAR_SITE,HYVLA_VLM_MLP_SILU_MUL_SITE,
               HYVLA_VLM_MLP_SILU_SITE,HYVLA_VLM_MLP_MUL_SITE}){
  v=manifest;v.routes.push_back({site,FmbRouteFamily::LINEAR,1,0,{},0});assert(!admitted(v));}
 m.cold_gateup_swiglu_available_=false;assert(!admitted(manifest));
 v=manifest;v.routes.pop_back();assert(admitted(v));
 m.cold_prefill_kv_only_=false;v.routes.clear();assert(admitted(v));

 // W8 only opts in through its own cold marker and uniform tower precision.
 m.cold_prefill_kv_only_=true;
 for(auto* tower:{&m.text_layers_,&m.moe_layers_})for(auto& w:*tower){
  for(auto* weight:{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w})weight->dtype=at::kChar;
  for(auto* scale:{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws}){scale->present=true;scale->shape={6144};}
 }
 assert(!m.gateup_swiglu_enabled(240,0));m.cold_w8_vlm_gateup_available_=true;
 assert(!m.gateup_swiglu_enabled(240,0));m.cold_w8_optimizations_available_=true;
 assert(m.gateup_swiglu_enabled(240,0));
 assert(m.gateup_swiglu_arguments()==std::vector<int64_t>({4,32,31,2,240,0,2048,6144,768,8,288,80,128,10,0,2,32,20,34,26,27,28,29}));
 auto w8=manifest;w8.routes.back().arguments=m.gateup_swiglu_arguments();
 assert(admitted(w8) && !admitted(manifest));
 for(auto* tower:{&m.text_layers_,&m.moe_layers_}){
  auto& w=tower->back();w.up_ws.present=false;assert(!admitted(w8));w.up_ws.present=true;
  w.gate_w.dtype=at::kHalf;assert(!admitted(w8));w.gate_w.dtype=at::kChar;
 }
 m.cold_w8_vlm_gateup_available_=false;assert(!admitted(w8));
 for(auto* tower:{&m.text_layers_,&m.moe_layers_})for(auto& w:*tower){
  for(auto* weight:{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w})weight->dtype=at::kHalf;
  for(auto* scale:{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws})scale->present=false;
 }
 m.cold_w8_optimizations_available_=false;
 m.cold_prefill_kv_only_=true;m.cold_row_merge_available_=true;
 assert(m.row_merge_enabled(240,0));
 auto merge=manifest;merge.routes.pop_back();
 assert(!admitted(merge));
 merge.routes.push_back({HYVLA_VLM_ROW_MERGE_SITE,FmbRouteFamily::GRAPH_SCHEDULE,1,0,m.row_merge_arguments(),0});
 assert(admitted(merge));
 auto dup=merge;dup.routes.push_back(merge.routes.back());assert(!admitted(dup));
 for(size_t i=0;i<m.row_merge_arguments().size();++i){v=merge;++v.routes.back().arguments[i];assert(!admitted(v));}
 for(int mode=0;mode<4;++mode){v=merge;
  if(mode==0)v.routes.back().family=FmbRouteFamily::LINEAR;
  if(mode==1)v.routes.back().selector=2;
  if(mode==2)v.routes.back().flags=1;
  if(mode==3)v.routes.back().invocation=1;
  assert(!admitted(v));}
 m.cold_row_merge_available_=false;assert(!admitted(merge));m.cold_row_merge_available_=true;
 for(int mode:{0,1,2}){m.cold_mot_norm_nomerge_=mode;assert(!m.row_merge_enabled(240,0));}
 m.cold_mot_norm_nomerge_=3;
 assert(!m.row_merge_enabled(224,0)&&!m.row_merge_enabled(240,1));
 m.text_layers_[0].q_w.dtype=at::kChar;assert(!m.row_merge_enabled(240,0));
 m.text_layers_[0].q_w.dtype=at::kHalf;
 m.moe_layers_.back().up_ws.present=true;assert(!m.row_merge_enabled(240,0));
 m.moe_layers_.back().up_ws.present=false;
 assert(!m.paired_rmsnorm_enabled(240,0));m.cold_paired_rmsnorm_available_=true;
 assert(m.paired_rmsnorm_enabled(240,0));assert(!admitted(merge));
 auto paired=merge;paired.routes.push_back({HYVLA_VLM_PAIRED_RMSNORM_SITE,FmbRouteFamily::GRAPH_SCHEDULE,1,0,m.paired_rmsnorm_arguments(),0});
 assert(admitted(paired));
 assert(m.paired_rmsnorm_arguments()==std::vector<int64_t>({1,32,31,31,240,0,2048,8,2,16,8,30,3,5,168}));
 v=paired;v.routes.push_back(v.routes.back());assert(!admitted(v));
 for(size_t j=0;j<paired.routes.back().arguments.size();++j){v=paired;++v.routes.back().arguments[j];assert(!admitted(v));}
 for(int mode=0;mode<4;++mode){v=paired;
  if(mode==0)v.routes.back().family=FmbRouteFamily::LINEAR;
  if(mode==1)v.routes.back().selector=2;
  if(mode==2)v.routes.back().flags=1;
  if(mode==3)v.routes.back().invocation=1;
  assert(!admitted(v));}
 v=paired;v.routes.push_back({kHyVlaVlmRmsNormFullRowsSite,FmbRouteFamily::LINEAR,5,0,{},0});assert(!admitted(v));
 for(int64_t* dim:{&paired.logical_length,&paired.physical_length,&paired.kv_logical_length}){
  ++*dim;assert(!admitted(paired));--*dim;}
 m.eps_hyvla_=1e-6;assert(!m.paired_rmsnorm_enabled(240,0));assert(!admitted(paired));m.eps_hyvla_=1e-5;
 m.cold_rmsnorm_spm_contract_.vector=false;assert(!m.paired_rmsnorm_enabled(240,0));assert(!admitted(paired));m.cold_rmsnorm_spm_contract_.vector=true;
 m.cold_row_merge_available_=false;assert(!m.paired_rmsnorm_enabled(240,0));m.cold_row_merge_available_=true;
 m.cold_paired_rmsnorm_available_=false;assert(!admitted(paired));assert(admitted(merge));
 // Eight exact prefix owners: row merge remains available with auto RMSNorm.
 // Multiples of 32 select V32; V16 paired is never substituted.
 m.cold_paired_rmsnorm_available_=true;
 for(int rows:{128,144,160,176,192,208,224,240}){
  const auto old_identity=m.kvinsert_cost_weight_identity();
  const int64_t old_rows=m.moe_chunk_size_;m.moe_chunk_size_=rows;
  const auto new_identity=m.kvinsert_cost_weight_identity();
  size_t changed=0;for(size_t j=0;j<old_identity.size();++j)changed+=old_identity[j]!=new_identity[j];
  assert(changed==size_t(old_rows!=rows));
  assert(m.row_merge_enabled(rows,0));
  assert(!m.row_merge_enabled(rows+16,0));
  m.cold_rmsnorm_spm_contract_.capability=RPU_RMSNORM_CAP_ALL;
  const bool auto_paired=rows%32!=0;
  assert(m.paired_rmsnorm_enabled(rows,0)==auto_paired);
  auto sized=paired;sized.logical_length=sized.physical_length=sized.kv_logical_length=rows;
  sized.routes[0].arguments=m.prefill_kv_only_arguments(rows);
  sized.routes[1].arguments=m.row_merge_arguments();sized.routes[2].arguments=m.paired_rmsnorm_arguments();
  assert(sized.routes[1].arguments[4]==rows && sized.routes[2].arguments[4]==rows);
  assert(sized.routes[2].arguments[11]==rows/8);
  if(!auto_paired){assert(!admitted(sized));sized.routes.pop_back();assert(admitted(sized));
   sized.routes.push_back(paired.routes.back());sized.routes.back().arguments=m.paired_rmsnorm_arguments();}
  m.cold_rmsnorm_spm_contract_.capability=RPU_RMSNORM_CAP_BASE|RPU_RMSNORM_CAP_V16;
  sized.routes.back().arguments=m.paired_rmsnorm_arguments();
  assert(m.paired_rmsnorm_enabled(rows,0) && admitted(sized));
  for(size_t route=1;route<3;++route){
   v=sized;v.routes.erase(v.routes.begin()+route);assert(!admitted(v));
   v=sized;v.routes.push_back(sized.routes[route]);assert(!admitted(v));
   for(size_t j=0;j<sized.routes[route].arguments.size();++j){v=sized;++v.routes[route].arguments[j];assert(!admitted(v));}
   v=sized;v.routes[route].arguments[4]=rows==240?192:240;assert(!admitted(v));
  }
  for(int64_t* dim:{&sized.logical_length,&sized.physical_length,&sized.kv_logical_length}){
   *dim+=16;assert(!admitted(sized));*dim-=16;}
  assert(!m.gateup_swiglu_enabled(rows,0)); // The rejected VLM Gate/Up candidate stays absent.
  // A separate marker pairs the original auto V32 arithmetic. V16 recipes
  // cannot masquerade as V32, and the old REF retains the two standalone calls.
  auto vector32=m;vector32.cold_rmsnorm_spm_contract_.capability=RPU_RMSNORM_CAP_ALL;
  const auto before_v32=vector32.kvinsert_cost_weight_identity();
  vector32.cold_paired_rmsnorm_v32_available_=true;
  const auto after_v32=vector32.kvinsert_cost_weight_identity();
  changed=0;for(size_t j=0;j<before_v32.size();++j)changed+=before_v32[j]!=after_v32[j];assert(changed==1);
  const auto admits_v32=[&](const FmbPhysicalExecutionManifest& x){
   try{return vector32.physical_manifest_forward_capability(x).supported;}catch(const std::exception&){return false;}
  };
  auto v32=sized;v32.routes.back().arguments=vector32.paired_rmsnorm_arguments();
  assert(vector32.paired_rmsnorm_enabled(rows,0) && admits_v32(v32));
  assert(v32.routes.back().arguments[0]==(auto_paired?1:2));
  assert(v32.routes.back().arguments[9]==(auto_paired?16:32));
  assert(v32.routes.back().arguments[13]==(auto_paired?5:6));
  for(size_t j=0;j<v32.routes.back().arguments.size();++j){auto bad=v32;++bad.routes.back().arguments[j];assert(!admits_v32(bad));}
  for(int field=0;field<4;++field){auto bad=v32;auto& r=bad.routes.back();
   if(field==0)r.family=FmbRouteFamily::LINEAR;if(field==1)++r.selector;
   if(field==2)++r.flags;if(field==3)++r.invocation;assert(!admits_v32(bad));}
  {auto bad=v32;bad.routes.push_back(bad.routes.back());assert(!admits_v32(bad));}
  {auto bad=v32;bad.routes.push_back({kHyVlaVlmRmsNormFullRowsSite,FmbRouteFamily::LINEAR,6,0,{},0});assert(!admits_v32(bad));}
  if(!auto_paired){
   assert(!admits_v32(sized));
   vector32.cold_paired_rmsnorm_available_=false;assert(admits_v32(v32));
   vector32.cold_paired_rmsnorm_v32_available_=false;
   assert(!vector32.paired_rmsnorm_enabled(rows,0) && !admits_v32(v32));
   auto fallback=v32;fallback.routes.pop_back();assert(admits_v32(fallback));
  }
  Model quant=m;
  for(auto* tower:{&quant.text_layers_,&quant.moe_layers_})for(auto& w:*tower){
   for(auto* weight:{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w})weight->dtype=at::kChar;
   for(auto* scale:{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws}){scale->present=true;scale->shape={6144};}}
  assert(!quant.row_merge_enabled(rows,0));quant.cold_w8_optimizations_available_=true;
  assert(quant.row_merge_enabled(rows,0) && quant.paired_rmsnorm_enabled(rows,0));
  assert(quant.physical_manifest_forward_capability(sized).supported);
  {auto q32=quant;q32.cold_rmsnorm_spm_contract_.capability=RPU_RMSNORM_CAP_ALL;
   q32.cold_paired_rmsnorm_v32_available_=true;assert(q32.physical_manifest_forward_capability(v32).supported);}
  // A two-camera mask cannot establish the three-camera proof: every short
  // prefix retains its full-row route even when the GU176 payload is present.
  quant.cold_vision_gu176_available_=true;
  assert(!quant.vision_gu176_enabled(rows,0));
  assert(quant.physical_manifest_forward_capability(sized).supported);
  quant.cold_vision_gu176_available_=false;
  const auto w8_base=quant;
  if(rows>=192){
  auto quant=w8_base;
  quant.cold_vision_gu176_available_=true;
  quant.standard_three_camera_mask_=true;
  const bool selected=rows==224;
  assert(quant.vision_gu176_enabled(rows,0)==selected);
  assert(quant.vision_gu176_arguments()==std::vector<int64_t>({1,32,31,31,1,rows,176,0,2048,6144,768,8,
      1,3,147,1,2,176*768*2,(rows-176)*768,0,rows,1,0,1}));
  auto gu=sized;
  gu.routes.push_back({HYVLA_VLM_VISION_GU176_SITE,FmbRouteFamily::GRAPH_SCHEDULE,1,0,quant.vision_gu176_arguments(),0});
  const size_t vision_index=gu.routes.size()-1;
  for(auto site:{HYVLA_VLM_MLP_GATE_LINEAR_SITE,HYVLA_VLM_MLP_UP_LINEAR_SITE})
   gu.routes.push_back({site,FmbRouteFamily::LINEAR,1,0,{rows,6144,2048,1,8},0});
  gu.routes.push_back({HYVLA_VLM_MLP_SILU_SITE,FmbRouteFamily::ACTIVATION,1,0,{0,rows*768,8},0});
  gu.routes.push_back({HYVLA_VLM_MLP_MUL_SITE,FmbRouteFamily::ACTIVATION,2,0,{0,rows*768,8},0});
  gu.routes.push_back({HYVLA_VLM_MLP_DOWN_LINEAR_SITE,FmbRouteFamily::LINEAR,1,0,{rows,2048,6144,0,8},0});
  auto gu_ok=[&](const auto& desc){try{return quant.physical_manifest_forward_capability(desc).supported;}
   catch(const std::exception&){return false;}};
  if(!selected){assert(!gu_ok(gu) && gu_ok(sized));}
  else{
  assert(gu_ok(gu) && !gu_ok(sized));
  // The retired two-camera GU128 receipt must never admit the GU176 route.
  {auto bad=gu;bad.routes[vision_index].arguments[0]=2;
   bad.routes[vision_index].arguments[6]=128;
   bad.routes[vision_index].arguments[13]=2;
   bad.routes[vision_index].arguments[14]=98;
   assert(!gu_ok(bad));}
  for(size_t ri=vision_index;ri<gu.routes.size();++ri){
   auto bad=gu;bad.routes.erase(bad.routes.begin()+ri);assert(!gu_ok(bad));
   bad=gu;bad.routes.push_back(bad.routes[ri]);assert(!gu_ok(bad));
   for(size_t ai=0;ai<gu.routes[ri].arguments.size();++ai){bad=gu;++bad.routes[ri].arguments[ai];assert(!gu_ok(bad));}
   bad=gu;bad.routes[ri].family=FmbRouteFamily::GRAPH_SCHEDULE;
   if(ri==vision_index)bad.routes[ri].family=FmbRouteFamily::LINEAR;assert(!gu_ok(bad));
   for(int field=0;field<3;++field){bad=gu;
    if(field==0)++bad.routes[ri].selector;if(field==1)++bad.routes[ri].flags;if(field==2)++bad.routes[ri].invocation;
    assert(!gu_ok(bad));}
  }
  for(bool* flag:{&quant.cold_vision_gu176_available_,&quant.standard_three_camera_mask_}){
   auto identity_before=quant.kvinsert_cost_weight_identity();*flag=false;
   assert(!quant.vision_gu176_enabled(rows,0) && !gu_ok(gu));
   auto identity_after=quant.kvinsert_cost_weight_identity();size_t changed=0;
   for(size_t j=0;j<identity_before.size();++j)changed+=identity_before[j]!=identity_after[j];assert(changed==1);
   assert(gu_ok(sized));*flag=true;
  }
  quant.cold_silu_mul_=true;assert(!quant.vision_gu176_enabled(rows,0));quant.cold_silu_mul_=false;
  assert(!quant.vision_gu176_enabled(rows,16) && !quant.vision_gu176_enabled(rows+16,0));
  quant.cold_w8_vlm_gateup_available_=true;
  if(rows==240)assert(!quant.vision_gu176_enabled(rows,0));quant.cold_w8_vlm_gateup_available_=false;
  for(auto* tower:{&quant.text_layers_,&quant.moe_layers_}){
   tower->back().up_w.dtype=at::kByte;assert(!quant.vision_gu176_enabled(rows,0));tower->back().up_w.dtype=at::kChar;
   tower->back().up_ws.present=false;assert(!quant.vision_gu176_enabled(rows,0));tower->back().up_ws.present=true;
  }
  m.cold_vision_gu176_available_=true;m.standard_three_camera_mask_=true;
  assert(!m.vision_gu176_enabled(rows,0)); // ordinary FP16 retains its route
  m.cold_vision_gu176_available_=false;m.standard_three_camera_mask_=false;
  }
  }
  quant.text_layers_.back().up_w.dtype=at::kByte;assert(!quant.row_merge_enabled(rows,0));
 }
 for(int rows:{112,127,129,200,241,256}){m.moe_chunk_size_=rows;
  assert(!m.row_merge_enabled(rows,0) && !m.paired_rmsnorm_enabled(rows,0));}
}
'''
    compile_run(tmp_path, "vlm_gateup_owner", source.replace("METHODS", methods).replace("IDENTITY", identity))


def test_actual_cold_binding_loads_private_marker_before_querying_availability(tmp_path):
    method = block(DECODER, "void HyVlaVlmModel::set_runtime_config(").replace("HyVlaVlmModel::", "")
    source = COMMON + r'''
constexpr unsigned MOT_NOMERGE_QKV=1,MOT_NOMERGE_MLP=2,RPU_RMSNORM_CAP_ALL=7;
struct RpuRmsNormSpmContract {uint32_t capability=0;};
bool rpu_rmsnorm_capability_valid(uint32_t c){return c&1;}
RpuRmsNormSpmContract rpu_snapshot_rmsnorm_spm_contract(uint32_t c){return {c};}
struct KernelCache {
 bool marker_exists=true,payload_exists=true,marker_loaded=false;int lazy_loads=0;bool w8_marker_exists=false,v32_marker_exists=false;
 static KernelCache& instance(){static KernelCache k;return k;}
 void* get_kernel(const std::string& name){
  if(name=="hyvla_mot_paired_rmsnorm_v32")return v32_marker_exists?this:nullptr;
  if(name=="hyvla_vlm_vision_gu176_w8")return w8_marker_exists?this:nullptr;
  if(name=="hyvla_w8a16_optimizations_s240_s51")return nullptr;
  if(name=="hyvla_vlm_gateup_swiglu_w8_v2")return w8_marker_exists&&payload_exists?this:nullptr;
  if(name=="hyvla_mot_row_merge_s240" || name=="hyvla_mot_paired_rmsnorm_s240")return payload_exists?this:nullptr;
  assert(name=="hyvla_vlm_gateup_swiglu_fp16_v2");
  ++lazy_loads;marker_loaded=marker_exists&&payload_exists;return marker_loaded?this:nullptr;}
 void* get_program(const std::string& name){
  if(name=="hyvla_vlm_gateup_swiglu_fp16_v2")return marker_loaded?this:nullptr;
  assert(name=="wall_prefill_gate_up_swiglu_fp16_m320n112k128" || name=="fused_gate_up_swiglu_w8a16_m320n112k128");return payload_exists?this:nullptr;}
};
struct Model {
 bool cold_config_bound_=false,cold_gateup_swiglu_available_=false,cold_fast_replay_=false;
 bool cold_w8_optimizations_available_=false,cold_w8_vlm_gateup_available_=false;
 bool cold_vision_gu176_available_=false,standard_three_camera_mask_=false;
 bool cold_row_merge_available_=false,cold_paired_rmsnorm_available_=false;
 bool cold_paired_rmsnorm_v32_available_=false;
 bool cold_fast_replay_preload_=false,cold_mask_once_=false,cold_partial_rope_=false,cold_silu_mul_=false;
 unsigned cold_mot_norm_nomerge_=0;std::vector<int> text_layers_;
 RpuRmsNormSpmContract cold_rmsnorm_spm_contract_;int invalidations=0;
 void invalidate_model_state(){++invalidations;}
 METHOD
};
int main(){
 for(bool marker:{false,true})for(bool payload:{false,true})for(bool w8_marker:{false,true})for(bool v32_marker:{false,true}){
  auto& cache=KernelCache::instance();cache={marker,payload,false,0,w8_marker,v32_marker};
  assert(cache.get_program("hyvla_vlm_gateup_swiglu_fp16_v2")==nullptr);
  Model m;m.set_runtime_config(true,true,true,true,3,false,1);
  assert(m.cold_gateup_swiglu_available_==(marker&&payload));
  assert(m.cold_vision_gu176_available_==w8_marker);
  assert(m.cold_w8_vlm_gateup_available_==(w8_marker&&payload));
  assert(m.cold_row_merge_available_==payload && m.cold_paired_rmsnorm_available_==payload);
  assert(m.cold_paired_rmsnorm_v32_available_==v32_marker);
  assert(cache.lazy_loads==1 && m.cold_config_bound_ && m.invalidations==1);
  cache.marker_exists=!marker;cache.payload_exists=!payload;cache.w8_marker_exists=!w8_marker;
  cache.v32_marker_exists=!v32_marker;
  bool rejected=false;try{m.set_runtime_config(true,true,true,true,3,false,1);}
  catch(const std::exception&){rejected=true;}
  assert(rejected && cache.lazy_loads==1 && m.cold_gateup_swiglu_available_==(marker&&payload));
  assert(m.cold_paired_rmsnorm_available_==payload);
  assert(m.cold_paired_rmsnorm_v32_available_==v32_marker);
  assert(m.cold_vision_gu176_available_==w8_marker);
  assert(m.cold_w8_vlm_gateup_available_==(w8_marker&&payload));
 }
}
'''
    compile_run(tmp_path, "vlm_gateup_cold", source.replace("METHOD", method))
