"""Execute exact decode MLP admission, descriptor and emitter with real consumer."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess

import pytest

from test_qwen3vl_prefill_gateup_launcher import definition

ROOT = Path(__file__).resolve().parents[1]
SITES = ("GATE","UP","ACTIVATION","DOWN","REDUCE")


@pytest.mark.parametrize("size,fused", [("4b", False), ("4b", True), ("2b", False)])
def test_decode_mlp_manifest_and_dispatch(tmp_path, size, fused):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required for actual native route consumer test")
    source = (ROOT/"src/fused/rpu_qwen3_model.h").read_text()
    base = (ROOT/"src/core/fused_model_base.h").read_text()
    planner = (ROOT/"src/core/fmb_three_stage_chunk_plan.cpp").read_text()
    enums = "\n".join(definition(text,f"enum class {name}")+";" for text,name in (
        (base,"AttentionExecutionPolicy"),(base,"ChunkMode"),(base,"FmbPhysicalManifestState"),(base,"FmbGraphLifecycle"),
        (base,"FmbLinearAccumulationPolicy"),(base,"FmbRouteFamily"),(base,"FmbLinearRouteSelector"),
        (base,"FmbSharedAllReduceRouteSelector"),(source,"CausalDecoderLinearRoute"),
        (source,"CausalDecoderPlanningMode")))
    structs = "\n".join(definition(base,f"struct {name}")+";" for name in
                         ("ChunkInfo","FmbRouteManifestEntry","FmbPhysicalExecutionManifest"))
    constants = "\n".join(re.findall(r"inline constexpr int64_t (?:QWEN3VL_4B_(?:PREFILL|DECODE)_\w+|WALL_BASE_(?:FP16_)?PREFILL_\w+)_SITE\s*=\s*[^;]+;",source))
    constants += "\n" + re.search(r"inline constexpr int64_t LINGBOT2_PREFILL_GATEUP_SHARED_A_SITE\s*=\s*[^;]+;",source).group(0)
    predicates = "\n".join(definition(source,marker) for marker in
                           ("bool qwen3vl_decode_gemv(","bool qwen3vl_decode_mlp(",
                            "CausalDecoderLinearRoute decoder_projection_route("))
    start = source.index("FmbPhysicalExecutionManifest physical_manifest_for_candidate(")
    append = source.index("        auto append =",start)
    append_text = source[append:source.index("        auto append_attention =",append)]
    ring = source.index("        auto append_ring =",start)
    ring_text = source[ring:source.index("        auto append_kv =",ring)]
    selected = source.index("        bool needs_shared_mlp = false;",start)
    selected_text = source[selected:source.index("        uint32_t shared_routes = 0;",selected)]
    # Execute the actual decode branch rather than a separately spelled policy.
    dispatch_start = source.index("        } else if (ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL &&\n            qwen3vl_decode_mlp(")
    dispatch = definition(source[dispatch_start:],"if (ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL &&")
    consumer = "\n".join(definition(planner,marker).replace("FmbPhysicalManifestConsumer::", "")
        for marker in ("bool FmbPhysicalManifestConsumer::complete() const",
          "const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(",
          "const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route("))
    consumer = consumer.replace("consume_route(","consume_physical_route(")
    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#define TORCH_CHECK(condition,...) do {if(!(condition))throw std::runtime_error(#condition);}while(false)
#define TORCH_INTERNAL_ASSERT(condition) assert(condition)
namespace c10 {using Half=uint16_t;}
namespace at {using IntArrayRef=std::vector<int64_t>;struct Tensor{int id=0;};}
__ENUMS__
__STRUCTS__
__CONSTANTS__
__RING__
struct Stage {std::vector<ChunkInfo> chunks;};
struct Plan {ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;Stage compute;};
struct Layout {bool is_causal=true,use_attn_mask=false;int64_t batch_size=1;
  AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV;};
struct Context {
  FmbPhysicalExecutionManifest manifest_;mutable std::vector<uint8_t> consumed_;
 const FmbPhysicalExecutionManifest& manifest()const{return manifest_;}
  Plan stage_plan;int64_t position=0,batch_size=1;bool is_causal=true;
  std::optional<int> attention_mask;
  __CONSUMER__
  void bind(FmbPhysicalExecutionManifest manifest) {
    std::sort(manifest.routes.begin(),manifest.routes.end(),[](const auto& a,const auto& b) {
      return std::make_tuple(int64_t(a.family),a.site_id,a.invocation)<std::make_tuple(int64_t(b.family),b.site_id,b.invocation);
    });
    manifest_=std::move(manifest);consumed_.assign(manifest_.routes.size(),0);
  }
};
struct LayerWeights {at::Tensor gate_w{1},up_w{2},down_w{3},gate_ws{4},up_ws{5},down_ws{6};};
std::vector<int> emitted;uint32_t live_base=0;
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t input,const at::Tensor& weight,uint32_t output,
    int64_t rows,int64_t n,int64_t k,int partition,int cores,uint32_t bias,const at::Tensor& scale,
    uint32_t nvfp4=0,uint16_t layer=0,bool acc32=false,bool gemv=false) {
  assert(rows==1&&cores==8&&bias==0&&nvfp4==0&&layer==0&&!acc32);
  if(weight.id==1||weight.id==2){
    assert(input==live_base+0x1000&&output==live_base+(weight.id==1?0x3000:0x4000));
    assert(n==__I__&&k==__H__&&partition==1&&gemv&&scale.id==weight.id+3);
    emitted.push_back(weight.id);
  } else {
    assert(weight.id==3&&input==live_base+0x3000&&output==live_base+0x5000);
    assert(n==__H__&&k==__I__&&partition==0&&gemv&&scale.id==6);emitted.push_back(4);
  }
}
void rpu_launch_silu_mul_spm_kernel(uint32_t gate,uint32_t up,uint32_t output,int64_t count,int cores){
  assert(gate==live_base+0x3000&&up==live_base+0x4000&&output==gate&&count==__LOCAL_I__&&cores==8);emitted.push_back(3);
}
    void rpu_launch_qwen3vl_4b_decode_gate_up_swiglu_w8a16_spm_kernel(uint32_t input,
    const at::Tensor& gate,const at::Tensor& up,uint32_t output,
    const at::Tensor& gs,const at::Tensor& us){
  assert(input==live_base+0x1000&&output==live_base+0x3000);
      assert(gate.id==1&&up.id==2&&gs.id==4&&us.id==5);emitted.push_back(6);
    }
    void rpu_launch_qwen3vl_4b_decode_o_ring_norm_spm_kernel(
        uint32_t,const at::Tensor&,const at::Tensor&,uint32_t,uint32_t,uint32_t,
        uint32_t,double,int64_t=4096){throw std::runtime_error("unexpected fused Down");}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t partial,uint32_t residual,uint32_t output,
    int64_t rows,int64_t cols,int producer_cores,int cores) {
  assert(partial==live_base+0x5000&&residual==live_base+0x2000&&output==live_base+0x1000);
  assert(rows==1&&cols==__H__&&producer_cores==8&&cores==8);emitted.push_back(5);
}
struct Model {
 bool qwen3vl_fp16_decode_profile_=false;
 bool qwen3vl_4b_w8_profile_=__IS_4B__,qwen3vl_4b_mlp_w4_profile_=false;
 bool linear_acc32_=false,decoder_mixed_precision_=false,adarms_=false;
 bool qwen3vl_4b_decode_gateup_fused_=__FUSED__,qwen3vl_4b_decode_down_ring_norm_fused_=false;
  bool qwen3vl_pooler_z1_dispatch_=false,qwen3vl_multiview_text_composite_dispatch_=false;
  CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
      int64_t seq_len_=1;double eps_=1e-6;bool partial_mrope_active_=false;Context context;bool last_shared=false;
  Context& ctx(){return context;}int num_cores() const{return 8;}int64_t hidden_size() const{return __H__;}
      int64_t intermediate_size() const{return __I__;}
      int num_layers() const{return 36;}
      uint32_t norm_input_addr(int,int) const{return live_base+0x6000;}
      bool qwen3vl_4b_decode_down_ring_norm(int,int64_t,int64_t,bool,bool,int64_t,bool)const{return false;}
  std::vector<int64_t> bind_cold_topology_arguments(std::vector<int64_t>) const {throw std::runtime_error("unexpected non-TP8");}
  uint32_t addr(int core,const char* name) const {
    assert(core==0);std::string s=name;
    if(s=="residual1")return live_base+0x1000;if(s=="residual2")return live_base+0x2000;
    if(s=="gate")return live_base+0x3000;if(s=="up")return live_base+0x4000;if(s=="down")return live_base+0x5000;
    throw std::runtime_error("unknown SPM slot");
  }
  bool lingbot_prefill_gateup_for_plan(const Plan&,int64_t,int64_t,bool,bool,int64_t,AttentionExecutionPolicy)const{return false;}
  std::vector<int64_t> lingbot_prefill_gateup_arguments()const{throw std::runtime_error("unbound LingBot MLP payload");}
  __PREDICATES__
  // Prefill's real sibling is covered by test_qwen3vl_4b_prefill_swiglu.py.
  bool qwen3vl_4b_prefill_swiglu(int64_t,int64_t,int64_t,bool,bool,int64_t)const{return false;}
  // Ordinary Qwen profiles have no Wall cold capabilities. Keep the added
  // sibling branches present, and fail if their payload path is reached.
  bool wall_base_fp16_shared_gateup_enabled(const Plan&,int64_t,int64_t,bool,bool,int64_t)const{return false;}
  bool wall_base_prefill_shared_input_enabled(const Plan&,int64_t,int64_t,bool,bool,int64_t)const{return false;}
  int64_t wall_base_prefill_shared_gateup_site()const{throw std::runtime_error("unexpected Wall route");}
  std::vector<int64_t> wall_base_fp16_shared_gateup_arguments(int64_t)const{throw std::runtime_error("unexpected Wall payload");}
  std::vector<int64_t> wall_base_prefill_shared_input_arguments(int64_t)const{throw std::runtime_error("unexpected Wall payload");}
  __MEMBER_RING__
  FmbPhysicalExecutionManifest mlp_manifest(const Plan& plan,const Layout& layout,int64_t physical_len,int64_t position) {
    FmbPhysicalExecutionManifest manifest;manifest.state=FmbPhysicalManifestState::COMPLETE;
    __APPEND__
    __APPEND_RING__
    __SELECTED__
    last_shared=needs_shared_mlp;return manifest;
  }
  __EMITTER__
      void dispatch(const LayerWeights& lw,const ChunkInfo& chunk){
        const int layer_idx=0;
    const int64_t bs=ctx().batch_size;
    __DISPATCH__ else {emitted.push_back(9);}
  }
};
Plan plan_for(int64_t rows){Plan p;p.compute.chunks.push_back({0,0,rows,rows});return p;}
void rejected(const std::function<void()>& operation,int count){
  emitted.clear();bool failed=false;try{operation();}catch(const std::runtime_error&){failed=true;}
  assert(failed&&emitted.size()==size_t(count));
}
int main(){
  constexpr bool fused=__FUSED__;
  LayerWeights weights;Layout layout;Model m;auto plan=plan_for(1);m.context.stage_plan=plan;
  if (!__IS_4B__) {
    // Ordinary 2B keeps shared AUTO_TILE and must not acquire the 4B-only
    // GEMV/fused-activation body, including at long mutable KV positions.
    for (int64_t pos:{0,1,513,4096}) {
      assert(!m.qwen3vl_decode_gemv(1));
      assert(m.decoder_projection_route(1)==CausalDecoderLinearRoute::AUTO_TILE);
      auto manifest=m.mlp_manifest(plan,layout,1,pos);
      assert(m.last_shared&&manifest.routes.empty());
      m.context.position=pos;m.context.bind(manifest);live_base=0x40000000+pos*256;
      emitted.clear();m.dispatch(weights,plan.compute.chunks[0]);
      assert(emitted==std::vector<int>({9}));
          rejected([&]{m.emit_qwen3vl_decode_mlp(weights,0,plan.compute.chunks[0]);},0);
    }
    return 0;
  }
  std::vector<FmbRouteManifestEntry> canonical;
  for(int64_t pos:{0,1,513,4096}){
    auto manifest=m.mlp_manifest(plan,layout,1,pos);
    assert(!m.last_shared&&manifest.routes.size()==(fused?3:5));
    if(canonical.empty())canonical=manifest.routes;
    else for(size_t i=0;i<canonical.size();++i){assert(manifest.routes[i].arguments==canonical[i].arguments);}
    m.context.position=pos;m.context.bind(manifest);live_base=0x40000000+pos*256;
    emitted.clear();m.dispatch(weights,plan.compute.chunks[0]);
    assert(emitted==(fused?std::vector<int>({6,4,5}):std::vector<int>({1,2,3,4,5})));
    assert(std::all_of(m.context.consumed_.begin(),m.context.consumed_.end(),[](auto n){return n==1;}));
    assert(m.context.find_route(FmbRouteFamily::ALL_REDUCE,QWEN3VL_4B_DECODE_MLP_REDUCE_SITE,0).selector==1);
    assert(m.context.find_route(FmbRouteFamily::LINEAR,QWEN3VL_4B_DECODE_MLP_DOWN_SITE,0).selector==
      static_cast<int64_t>(CausalDecoderLinearRoute::GEMV));
  }
  auto excluded=[&](Model model,Layout l,int64_t physical,int64_t rows,ChunkMode mode){
    auto p=plan_for(rows);p.chunk_mode=mode;auto manifest=model.mlp_manifest(p,l,physical,19);
    assert(model.last_shared&&manifest.routes.empty());model.context.bind(manifest);model.context.stage_plan=p;
    model.seq_len_=physical;model.context.is_causal=l.is_causal;model.context.batch_size=l.batch_size;
    if(l.use_attn_mask)model.context.attention_mask=1;
    emitted.clear();model.dispatch(weights,p.compute.chunks[0]);assert(emitted==std::vector<int>({9}));
        rejected([&]{model.emit_qwen3vl_decode_mlp(weights,0,p.compute.chunks[0]);},0);
  };
  for(auto field:{&Model::linear_acc32_,&Model::decoder_mixed_precision_,&Model::adarms_,
      &Model::qwen3vl_pooler_z1_dispatch_,&Model::qwen3vl_multiview_text_composite_dispatch_}){
    Model bad;bad.*field=true;excluded(bad,layout,1,1,ChunkMode::SEQUENTIAL);}
  {Model bad;bad.qwen3vl_4b_w8_profile_=false;excluded(bad,layout,1,1,ChunkMode::SEQUENTIAL);}
  if(!fused){
   Model mixed;mixed.qwen3vl_4b_w8_profile_=false;mixed.qwen3vl_4b_mlp_w4_profile_=true;
   assert(mixed.qwen3vl_decode_mlp(1,1,true,false,1));
  }
  for(auto mode:{CausalDecoderPlanningMode::WALL_Z1,CausalDecoderPlanningMode::QWEN3VL_POOLER_Z1,
      CausalDecoderPlanningMode::QWEN3VL_MULTIVIEW_TEXT}){Model bad;bad.planning_mode_=mode;excluded(bad,layout,1,1,ChunkMode::SEQUENTIAL);}
  excluded(Model{},layout,1,1,ChunkMode::KV_FIRST);
  for(auto shape:std::vector<std::pair<int64_t,int64_t>>{{0,1},{2,1},{96,96},{256,1},{1,2}})
    excluded(Model{},layout,shape.first,shape.second,ChunkMode::SEQUENTIAL);
  {auto l=layout;l.is_causal=false;excluded(Model{},l,1,1,ChunkMode::SEQUENTIAL);}
  {auto l=layout;l.use_attn_mask=true;excluded(Model{},l,1,1,ChunkMode::SEQUENTIAL);}
  {auto l=layout;l.batch_size=2;excluded(Model{},l,1,1,ChunkMode::SEQUENTIAL);}
  const auto sites=fused
    ? std::vector<int64_t>{QWEN3VL_4B_DECODE_MLP_GATEUP_FUSED_SITE,QWEN3VL_4B_DECODE_MLP_DOWN_SITE,QWEN3VL_4B_DECODE_MLP_REDUCE_SITE}
    : std::vector<int64_t>{QWEN3VL_4B_DECODE_MLP_GATE_SITE,QWEN3VL_4B_DECODE_MLP_UP_SITE,
        QWEN3VL_4B_DECODE_MLP_ACTIVATION_SITE,QWEN3VL_4B_DECODE_MLP_DOWN_SITE,QWEN3VL_4B_DECODE_MLP_REDUCE_SITE};
  auto manifest=m.mlp_manifest(plan,layout,1,0);
  // build14's AUTO Down lost the long-input teacher cosine gate. A stale
  // descriptor selecting that route must fail before the Down launch.
  {auto bad=manifest;auto down=std::find_if(bad.routes.begin(),bad.routes.end(),[](auto& r){
      return r.site_id==QWEN3VL_4B_DECODE_MLP_DOWN_SITE;});
   down->selector=static_cast<int64_t>(CausalDecoderLinearRoute::AUTO_TILE);
   m.context.bind(bad);rejected([&]{m.dispatch(weights,plan.compute.chunks[0]);},fused?1:3);}
  for(size_t index=0;index<sites.size();++index)for(int fault=0;fault<6;++fault){
    auto bad=manifest;auto it=std::find_if(bad.routes.begin(),bad.routes.end(),[&](auto& r){return r.site_id==sites[index];});
    assert(it!=bad.routes.end());
    if(fault==0)bad.routes.erase(it);if(fault==1)it->selector+=10;
    if(fault==2)it->flags=1;if(fault==3)it->invocation=1;
    if(fault==4)it->arguments.push_back(99);if(fault==5)it->family=FmbRouteFamily::KV_INSERT;
    m.context.bind(bad);rejected([&]{m.dispatch(weights,plan.compute.chunks[0]);},index);
  }
  auto bad=manifest;bad.state=FmbPhysicalManifestState::UNSPECIFIED;m.context.bind(bad);
  rejected([&]{m.dispatch(weights,plan.compute.chunks[0]);},0);
  // A descriptor built for the other cold payload capability must fail before
  // either implementation launches; old REF and fused REF are different routes.
  m.qwen3vl_4b_decode_gateup_fused_=!fused;m.context.bind(manifest);
  rejected([&]{m.dispatch(weights,plan.compute.chunks[0]);},0);
}
'''
    replacements = {"ENUMS":enums,"STRUCTS":structs,"CONSTANTS":constants,"PREDICATES":predicates,
        "RING":definition(planner,"int64_t fmb_ring_all_reduce_route_selector("),
        "MEMBER_RING":definition(source,"int64_t causal_ring_route(int64_t rows, int64_t cols) const"),
        "CONSUMER":consumer,"APPEND":append_text,"APPEND_RING":ring_text,"SELECTED":selected_text,
        "EMITTER":definition(source,"void emit_qwen3vl_decode_mlp("),"DISPATCH":dispatch,
        "FUSED":"true" if fused else "false", "H":str(2048 if size == "2b" else 2560),
        "I":str(6144 if size == "2b" else 9728), "LOCAL_I":str(768 if size == "2b" else 1216),
        "IS_4B":"false" if size == "2b" else "true"}
    for key,value in replacements.items():program=program.replace(f"__{key}__",value)
    cpp,exe = tmp_path/"decode_mlp.cpp",tmp_path/"decode_mlp"
    cpp.write_text(program)
    subprocess.run([compiler,"-std=c++17","-Wall","-Wextra","-Werror","-Wno-misleading-indentation",
                    str(cpp),"-o",str(exe)],check=True,capture_output=True,text=True)
    subprocess.run([str(exe)],check=True,capture_output=True,text=True)
