"""Run production prefill predicate/manifest/emitter and actual route consumers."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_qwen3vl_prefill_gateup_launcher import definition


ROOT = Path(__file__).resolve().parents[1]


def test_prefill_manifest_and_dispatch(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("native physical-route test requires g++")
    source = (ROOT/"src/fused/rpu_qwen3_model.h").read_text()
    base = (ROOT/"src/core/fused_model_base.h").read_text()
    planner = (ROOT/"src/core/fmb_three_stage_chunk_plan.cpp").read_text()
    constants = "\n".join(re.search(rf"inline constexpr int64_t {name}\s*=\s*[^;]+;",source).group()
                          for name in ("QWEN3VL_4B_PREFILL_GATEUP_SITE","QWEN3VL_4B_PREFILL_DOWN_SITE",
                                       "QWEN3VL_4B_PREFILL_MLP_REDUCE_SITE",
                                       "QWEN3VL_4B_DECODE_MLP_GATEUP_FUSED_SITE",
                                       "QWEN3VL_4B_DECODE_MLP_GATE_SITE","QWEN3VL_4B_DECODE_MLP_UP_SITE",
                                       "QWEN3VL_4B_DECODE_MLP_ACTIVATION_SITE","QWEN3VL_4B_DECODE_MLP_DOWN_SITE",
                                       "QWEN3VL_4B_DECODE_MLP_REDUCE_SITE",
                                       "QWEN3VL_4B_DECODE_MLP_DOWN_RING_NORM_SITE",
                                       "LINGBOT2_PREFILL_GATEUP_SHARED_A_SITE",
                                       "WALL_BASE_FP16_PREFILL_SHARED_GATEUP_SITE",
                                       "WALL_BASE_PREFILL_SHARED_DOWN_SITE",
                                       "WALL_BASE_PREFILL_SHARED_REDUCE_SITE"))
    enums = "\n".join(definition(text,f"enum class {name}")+";" for text,name in (
        (base,"AttentionExecutionPolicy"),(base,"ChunkMode"),(base,"FmbPhysicalManifestState"),(base,"FmbGraphLifecycle"),
        (base,"FmbLinearAccumulationPolicy"),(base,"FmbRouteFamily"),(base,"FmbLinearRouteSelector"),
        (base,"FmbSharedAllReduceRouteSelector"),(source,"CausalDecoderLinearRoute"),
        (source,"CausalDecoderPlanningMode")))
    structs = "\n".join(definition(base,f"struct {name}")+";" for name in
                         ("ChunkInfo","FmbRouteManifestEntry","FmbPhysicalExecutionManifest"))
    predicate = definition(source,"bool qwen3vl_4b_prefill_swiglu(")
    predicate += "\n" + definition(source,"std::vector<int64_t> lingbot_prefill_gateup_arguments(")
    emitter = definition(source,"void emit_qwen3vl_4b_prefill_mlp(")
    ring = definition(planner,"int64_t fmb_ring_all_reduce_route_selector(")
    member_ring = definition(source,"int64_t causal_ring_route(int64_t rows, int64_t cols) const")
    manifest_start = source.index("FmbPhysicalExecutionManifest physical_manifest_for_candidate(")
    append_start = source.index("        auto append =",manifest_start)
    append = source[append_start:source.index("        auto append_attention =",append_start)]
    ring_start = source.index("        auto append_ring =",manifest_start)
    append_ring = source[ring_start:source.index("        auto append_kv =",ring_start)]
    selected_start = source.index("        bool needs_shared_mlp = false;",manifest_start)
    selected = source[selected_start:source.index("        uint32_t shared_routes = 0;",selected_start)]
    dispatch = definition(source, "if (ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL &&\n            qwen3vl_4b_prefill_swiglu(")
    consumer = "\n".join(definition(planner,signature).replace("FmbPhysicalManifestConsumer::", "")
                          for signature in ("bool FmbPhysicalManifestConsumer::complete() const",
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
  Plan stage_plan;int64_t position=0,batch_size=1,seq_len=560;bool is_causal=true;
  AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV;
  std::optional<int> attention_mask;
  const FmbPhysicalExecutionManifest& manifest() const {return manifest_;}
  __CONSUMER__
  void bind(FmbPhysicalExecutionManifest manifest) {
    std::sort(manifest.routes.begin(),manifest.routes.end(),[](const auto& a,const auto& b) {
      return std::make_tuple(int64_t(a.family),a.site_id,a.invocation)<std::make_tuple(int64_t(b.family),b.site_id,b.invocation);
    });
    manifest_=std::move(manifest);consumed_.assign(manifest_.routes.size(),0);
  }
};
struct LayerWeights {at::Tensor gate_w{1},up_w{2},down_w{3},gate_ws{4},up_ws{5},down_ws{6};};
std::vector<int> emitted;int64_t expected_rows=0;
void rpu_launch_qwen3vl_4b_prefill_gate_up_swiglu_w8a16_spm_kernel(uint32_t input,const at::Tensor& gate,
    const at::Tensor& up,uint32_t output,int64_t rows,const at::Tensor& gs,const at::Tensor& us) {
  assert(input==0x40100000 && output==0x40300000 && rows==expected_rows);
  assert(gate.id==1 && up.id==2 && gs.id==4 && us.id==5);emitted.push_back(1);
}
void rpu_launch_lingbot2_prefill_gate_up_shared_a_w8_m224_spm_kernel(
    uint32_t,const at::Tensor&,const at::Tensor&,uint32_t,const at::Tensor&,const at::Tensor&) {
  throw std::runtime_error("unbound LingBot shared-A launcher");
}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t input,const at::Tensor& weight,uint32_t output,
    int64_t rows,int64_t n,int64_t k,int partition,int cores,uint32_t bias,const at::Tensor& scale) {
  assert(input==0x40300000 && output==0x40400000 && rows==expected_rows);
  assert(n==2560 && k==9728 && partition==0 && cores==8 && bias==0 && weight.id==3 && scale.id==6);
  emitted.push_back(2);
}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t partial,uint32_t residual,uint32_t output,
    int64_t rows,int64_t cols,int producer_cores,int cores) {
  assert(partial==0x40400000 && residual==0x40200000 && output==0x40100000);
  assert(rows==expected_rows && cols==2560 && producer_cores==8 && cores==8);emitted.push_back(3);
}
struct Model {
  bool qwen3vl_4b_w8_profile_=true,linear_acc32_=false,decoder_mixed_precision_=false,adarms_=false;
      bool qwen3vl_4b_decode_gateup_fused_=false,qwen3vl_4b_decode_down_ring_norm_fused_=false;
  bool qwen3vl_pooler_z1_dispatch_=false,qwen3vl_multiview_text_composite_dispatch_=false;
  CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
  int64_t seq_len_=560;Context context;bool last_shared=false;
  Context& ctx(){return context;}
  int num_cores() const{return 8;}
  int64_t hidden_size() const{return 2560;}
  int64_t intermediate_size() const{return 9728;}
  std::vector<int64_t> bind_cold_topology_arguments(std::vector<int64_t>) const {
    throw std::runtime_error("unexpected non-TP8 path");
  }
  uint32_t addr(int core,const char* name) const {
    assert(core==0);std::string s=name;
    if(s=="residual1")return 0x40100000;
    if(s=="residual2")return 0x40200000;
    if(s=="gate")return 0x40300000;
    if(s=="down")return 0x40400000;
    throw std::runtime_error("unknown buffer");
  }
  // Ordinary Qwen prefill has no cold LingBot shared-A capability.
  bool lingbot_prefill_gateup_for_plan(const Plan&,int64_t,int64_t,bool,bool,int64_t,AttentionExecutionPolicy)const{return false;}
  template<typename... Args> bool wall_base_fp16_shared_gateup_enabled(Args...) const {return false;}
  template<typename... Args> bool wall_base_prefill_shared_input_enabled(Args...) const {return false;}
  std::vector<int64_t> wall_base_fp16_shared_gateup_arguments(int64_t) const {
    throw std::runtime_error("unexpected Wall route");
  }
  std::vector<int64_t> wall_base_prefill_shared_input_arguments(int64_t) const {
    throw std::runtime_error("unexpected Wall route");
  }
  int64_t wall_base_prefill_shared_gateup_site() const {
    throw std::runtime_error("unexpected Wall route");
  }
  __PREDICATE__
  // This harness isolates prefill; the decode sibling is executed with its
  // real predicate/manifest/emitter in test_qwen3vl_4b_decode_mlp.py.
  bool qwen3vl_decode_mlp(int64_t,int64_t,bool,bool,int64_t) const {return false;}
  __MEMBER_RING__
  FmbPhysicalExecutionManifest mlp_manifest(const Plan& plan,const Layout& layout,int64_t physical_len,int64_t position) {
    FmbPhysicalExecutionManifest manifest;manifest.state=FmbPhysicalManifestState::COMPLETE;
    __APPEND__
    __APPEND_RING__
    __SELECTED__
    last_shared=needs_shared_mlp;return manifest;
  }
  __EMITTER__
  void dispatch(const LayerWeights& lw,const ChunkInfo& chunk) {
    ctx().seq_len=seq_len_;
    const int64_t bs=ctx().batch_size;
    __DISPATCH__ else {emitted.push_back(4);}
  }
};
Plan plan_for(std::vector<int64_t> lengths) {
  Plan p;int i=0;int64_t offset=0;
  for(auto n:lengths){p.compute.chunks.push_back({i++,offset,n,offset+n});offset+=n;}
  return p;
}
void rejected(const std::function<void()>& operation,int emitted_count) {
  emitted.clear();bool failed=false;
  try{operation();}catch(const std::runtime_error&){failed=true;}
  assert(failed && emitted.size()==size_t(emitted_count));
}
int main() {
  LayerWeights weights;Layout layout;
  // Real tail-bearing P560/C224, repeated C256 and single-chunk shapes.
  for(const auto& lengths:std::vector<std::vector<int64_t>>{{224,224,112},{256,256},{16},{96},{112},{128},{224},{256}}) {
    Model m;Plan p=plan_for(lengths);m.context.stage_plan=p;
    int64_t total=0;for(auto n:lengths)total+=n;m.seq_len_=total;
    auto manifest=m.mlp_manifest(p,layout,total,0);
    assert(!m.last_shared && manifest.routes.size()==3*lengths.size());m.context.bind(manifest);
    for(const auto& c:p.compute.chunks) {
      emitted.clear();expected_rows=c.len;m.dispatch(weights,c);
      assert(emitted==std::vector<int>({1,2,3}));
      const auto& fusion=m.context.find_route(FmbRouteFamily::LINEAR,QWEN3VL_4B_PREFILL_GATEUP_SITE,c.idx);
      assert(fusion.arguments==std::vector<int64_t>({c.len,9728,2560,8,320,112}));
      const auto& ring=m.context.find_route(FmbRouteFamily::ALL_REDUCE,QWEN3VL_4B_PREFILL_MLP_REDUCE_SITE,c.idx);
      assert(ring.selector==(c.len==16?1:2));
    }
    assert(std::all_of(m.context.consumed_.begin(),m.context.consumed_.end(),[](uint8_t x){return x==1;}));
  }
  // A non-aligned tail falls back while earlier legal chunks retain their sites.
  {Model m;auto p=plan_for({224,17});auto manifest=m.mlp_manifest(p,layout,241,0);
   assert(m.last_shared && manifest.routes.size()==3);m.context.stage_plan=p;m.context.bind(manifest);m.seq_len_=241;
   expected_rows=17;emitted.clear();m.dispatch(weights,p.compute.chunks[1]);assert(emitted==std::vector<int>({4}));}
  auto false_profile=[&](Model m,Layout l,Plan p,int64_t physical,int64_t pos) {
    auto manifest=m.mlp_manifest(p,l,physical,pos);assert(m.last_shared && manifest.routes.empty());
    m.seq_len_=physical;m.context.position=pos;m.context.stage_plan=p;m.context.is_causal=l.is_causal;
    m.context.batch_size=l.batch_size;if(l.use_attn_mask)m.context.attention_mask=1;
    m.context.bind(manifest);emitted.clear();m.dispatch(weights,p.compute.chunks.front());assert(emitted==std::vector<int>({4}));
  };
  for(auto member:{&Model::linear_acc32_,&Model::decoder_mixed_precision_,&Model::adarms_,
                   &Model::qwen3vl_pooler_z1_dispatch_,&Model::qwen3vl_multiview_text_composite_dispatch_}) {
    Model m;m.*member=true;false_profile(m,layout,plan_for({96}),96,0);
  }
  {Model m;m.qwen3vl_4b_w8_profile_=false;false_profile(m,layout,plan_for({96}),96,0);}
  for(auto mode:{CausalDecoderPlanningMode::WALL_Z1,CausalDecoderPlanningMode::QWEN3VL_POOLER_Z1,
                 CausalDecoderPlanningMode::QWEN3VL_MULTIVIEW_TEXT}) {
    Model m;m.planning_mode_=mode;false_profile(m,layout,plan_for({96}),96,0);
  }
  for(int64_t n:{1,15,17,272})false_profile(Model{},layout,plan_for({n}),n,0);
  false_profile(Model{},layout,plan_for({96}),96,1);
  {auto l=layout;l.is_causal=false;false_profile(Model{},l,plan_for({96}),96,0);}
  {auto l=layout;l.use_attn_mask=true;false_profile(Model{},l,plan_for({96}),96,0);}
  {auto l=layout;l.batch_size=2;false_profile(Model{},l,plan_for({96}),96,0);}
  {auto p=plan_for({96});p.chunk_mode=ChunkMode::KV_FIRST;false_profile(Model{},layout,p,96,0);}
  // Real consumer rejects drift at the matching physical site, before that op.
  Model m;auto p=plan_for({224,224,112});m.context.stage_plan=p;
  auto manifest=m.mlp_manifest(p,layout,560,0);auto chunk=p.compute.chunks.back();expected_rows=chunk.len;
  for(int fault=0;fault<7;++fault) {
    auto bad=manifest;
    auto gate=std::find_if(bad.routes.begin(),bad.routes.end(),[&](const auto& r){return r.site_id==QWEN3VL_4B_PREFILL_GATEUP_SITE && r.invocation==2;});
    auto down=std::find_if(bad.routes.begin(),bad.routes.end(),[&](const auto& r){return r.site_id==QWEN3VL_4B_PREFILL_DOWN_SITE && r.invocation==2;});
    if(fault==0)gate->arguments[0]=128;
    if(fault==1)gate->selector=2;
    if(fault==2)gate->flags=1;
    if(fault==3)gate->family=FmbRouteFamily::ALL_REDUCE;
    if(fault==4)gate->invocation=3;
    if(fault==5)bad.routes.erase(down);
    if(fault==6)down->arguments[2]=6144;
    m.context.bind(bad);rejected([&]{m.dispatch(weights,chunk);},fault>=5?1:0);
  }
  auto bad=manifest;
  bad.routes.erase(std::remove_if(bad.routes.begin(),bad.routes.end(),[](const auto& r){
    return r.site_id==QWEN3VL_4B_PREFILL_MLP_REDUCE_SITE && r.invocation==2;}),bad.routes.end());
  m.context.bind(bad);rejected([&]{m.dispatch(weights,chunk);},2);
  bad=manifest;bad.state=FmbPhysicalManifestState::UNSPECIFIED;m.context.bind(bad);
  rejected([&]{m.dispatch(weights,chunk);},0);
}
'''
    for token,text in {"ENUMS":enums,"STRUCTS":structs,"CONSTANTS":constants,"RING":ring,
                       "CONSUMER":consumer,"PREDICATE":predicate,"MEMBER_RING":member_ring,
                       "APPEND":append,"APPEND_RING":append_ring,"SELECTED":selected,
                       "EMITTER":emitter,"DISPATCH":dispatch}.items():
        program=program.replace(f"__{token}__",text)
    cpp,exe=tmp_path/"prefill_manifest.cpp",tmp_path/"prefill_manifest"
    cpp.write_text(program)
    compiled = subprocess.run([compiler,"-std=c++17","-Wall","-Wextra","-Werror",str(cpp),"-o",str(exe)],
                              capture_output=True,text=True)
    assert compiled.returncode == 0, compiled.stderr
    subprocess.run([str(exe)],check=True,capture_output=True,text=True)
