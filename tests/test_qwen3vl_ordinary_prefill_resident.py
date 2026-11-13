"""Qwen3-VL cold admission, decode routing and causal schedule checks; no HW claim."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from test_qwen3vl_4b_w8_prefill_resident import _method


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src/fused/rpu_qwen3_model.h"


def test_native_cold_admission_schedule_and_causal_equivalence(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    source = HEADER.read_text()
    start = source.index("        const auto resident_projection_dtype =")
    loop = "for (int64_t i = 0; qwen3vl_ordinary_resident_prefill_profile_ && i < N; ++i)"
    loop_start = source.index(loop, start)
    cold = source[start:loop_start] + _method(source[loop_start:], loop)
    fp16_start = source.index("        qwen3vl_fp16_decode_profile_ =")
    cold += source[fp16_start:source.index(";", fp16_start) + 1]
    methods = "\n".join(_method(source, marker) for marker in (
        "bool qwen3vl_decode_gemv(", "bool qwen3vl_decode_mlp(",
        "bool qwen3vl_ordinary_resident_prefill(",
        "std::vector<int64_t> qwen3vl_ordinary_prefill_schedule_arguments(",
        "void consume_qwen3vl_ordinary_prefill_schedule("))
    manifest = _method(source, "if (qwen3vl_ordinary_resident_prefill(\n                physical_len")
    config = _method(source, "if (qwen3vl_ordinary_resident_prefill(\n                ctx().seq_len")
    fmb = (ROOT / "src/core/fused_model_base.cpp").read_text()
    traversal = fmb[fmb.index("struct FmbExecutionTraversalSpec {"):
                    fmb.index("int64_t validate_fmb_resolved_chunks(")]
    base = (ROOT / "src/core/fused_model_base.h").read_text()
    traversal = ("namespace detail {\n" + _method(
        base, "inline bool fmb_kv_first_pair_carry_eligible(") + "\n}\n" + traversal)
    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <typeinfo>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
#define TORCH_INTERNAL_ASSERT(condition, ...) TORCH_CHECK(condition)
namespace at {
enum ScalarType { kByte=0, kChar=1, kFloat=6, kHalf=5 };
struct Tensor { ScalarType dtype; ScalarType scalar_type() const { return dtype; } };
}
enum class ChunkMode { SEQUENTIAL, KV_FIRST };
enum class InterLayerIO { AUTO, SPM_RESIDENT, DDR_PINGPONG };
enum class CausalDecoderPlanningMode { ORDINARY, OTHER };
enum class FmbRouteFamily { GRAPH_SCHEDULE };
enum class SpmFmbOccurrenceKind { LayerBody, KvInsertBody };
enum class BufferScope { LayerWide, Compute, KvInsert };
using SpmFmbScopeMask = uint32_t;
constexpr SpmFmbScopeMask spm_fmb_scope_bit(BufferScope s) { return 1u << static_cast<int>(s); }
constexpr int64_t CAUSAL_DECODER_VL_PREFILL_SCHEDULE_SITE=1348148692389361716LL;
struct ChunkInfo { int64_t idx, offset, len, kv_seq_len; };
struct ChunkPlan { int64_t chunk_size=128, num_chunks=0; };
struct Compute { ChunkPlan plan; std::vector<ChunkInfo> chunks; };
struct StagePlan { ChunkMode chunk_mode=ChunkMode::SEQUENTIAL; Compute compute; };
struct ModelDynamicConfig {
 InterLayerIO inter_layer_io=InterLayerIO::AUTO; bool chunk_outer_within_group=false;
};
struct Context {
 StagePlan stage_plan;
 int64_t seq_len=272, position=0, batch_size=1;
 bool is_causal=true, input_in_spm=false, output_to_spm=true;
 std::optional<int> attention_mask;
 int consumed=0;
 std::vector<int64_t> descriptor;
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,
                             int64_t flags,const std::vector<int64_t>& args) {
  assert(family==FmbRouteFamily::GRAPH_SCHEDULE && site==CAUSAL_DECODER_VL_PREFILL_SCHEDULE_SITE);
  assert(selector==1 && flags==0); TORCH_CHECK(args==descriptor); ++consumed;
 }
};
struct LayerWeights { at::Tensor q_w; };
struct CausalDecoderModel {
 int64_t N=28,H=2048,I=6144,Q=16,KV=8,D=128,cores=8;
 bool has_scale=true,has_nvfp4=false,use_silu=true,has_qk_norm_=true,has_qkv_bias_=false;
 bool qwen3vl_fp16_decode_profile_=false,qwen3vl_4b_w8_profile_=false;
 bool qwen3vl_ordinary_resident_prefill_profile_=false,linear_acc32_=false,adarms_=false;
 bool decoder_mixed_precision_=false,decode_gate_up_fusion_=false,wall_prefill_staged_mlp_=false;
 bool pre_layers_residual1_ready_=false,wall_z1_dispatch_=false,qwen3vl_pooler_z1_dispatch_=false;
 bool qwen3vl_multiview_text_composite_dispatch_=false;
 CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
 std::vector<int64_t> mrope_section{24,20,20},deepstack_lang_layers{0,1,2};
 std::vector<at::Tensor> q_w_list,k_w_list,v_w_list,o_w_list,gate_list,up_list,down_list;
 std::vector<LayerWeights> layer_weights_;
 Context context;
 virtual ~CausalDecoderModel()=default;
 int64_t num_cores()const{return cores;} int64_t num_layers()const{return N;}
 int64_t hidden_size()const{return H;} int64_t intermediate_size()const{return I;}
 int64_t num_q_heads()const{return Q;} int64_t num_kv_heads()const{return KV;}
 int64_t head_dim()const{return D;} Context& ctx(){return context;}
 void setup(int size,at::ScalarType dtype) {
  if(size==4){N=36;H=2560;I=9728;Q=32;}
  has_scale=dtype!=at::kHalf;
  for(auto* list:{&q_w_list,&k_w_list,&v_w_list,&o_w_list,&gate_list,&up_list,&down_list})
   list->assign(N,{dtype});
  layer_weights_.assign(N,{{dtype}});
 }
 void cold(){
  const int64_t hidden_size=H,intermediate_size=I,num_q_heads=Q,num_kv_heads=KV,head_dim=D;
  __COLD__
 }
 __METHODS__
 void make_plan(int64_t length) {
  context.seq_len=length;context.stage_plan.compute.chunks.clear();context.consumed=0;
  for(int64_t off=0;off<length;off+=128)
   context.stage_plan.compute.chunks.push_back({off/128,off,std::min<int64_t>(128,length-off),std::min(off+128,length)});
  context.stage_plan.compute.plan.num_chunks=context.stage_plan.compute.chunks.size();
 }
 void manifest(){
  const auto& plan=context.stage_plan;
  struct Layout { bool is_causal,use_attn_mask;int64_t batch_size; };
  const Layout layout{context.is_causal,context.attention_mask.has_value(),context.batch_size};
  const int64_t physical_len=context.seq_len,position=context.position;
  context.descriptor.clear();
  auto append=[&](FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,const std::vector<int64_t>& args){
   assert(family==FmbRouteFamily::GRAPH_SCHEDULE && site==CAUSAL_DECODER_VL_PREFILL_SCHEDULE_SITE);
   assert(selector==1 && flags==0);context.descriptor=args;
  };
  __MANIFEST__
 }
 ModelDynamicConfig configure(){
  ModelDynamicConfig cfg;const auto& plan=context.stage_plan.compute.plan;
  __CONFIG__
  return cfg;
 }
};
struct DerivedDecoder : CausalDecoderModel {};
__TRAVERSAL__
// Integer arithmetic isolates the real traversal's dependency/offset contract
// from device floating point. Every layer requires its complete earlier KV prefix;
// DeepStack is row-specific and final norm stands in at the last layer only.
std::vector<std::vector<int64_t>> run(CausalDecoderModel& model,int64_t length,bool resident){
 model.make_plan(length);model.manifest();
 std::vector<ChunkInfo> empty;
 FmbExecutionTraversalSpec spec;
 spec.num_layers=model.N;spec.cross_batch=model.N;
 spec.chunks=&model.ctx().stage_plan.compute.chunks;spec.kv_insert_chunks=&empty;
 spec.inter_layer_io=resident?InterLayerIO::SPM_RESIDENT:InterLayerIO::DDR_PINGPONG;
 spec.chunk_outer_within_group=resident;
 std::vector<std::vector<int64_t>> hidden(model.N+1,std::vector<int64_t>(length,-1)),kv(model.N);
 for(int64_t r=0;r<length;++r)hidden[0][r]=r%31;
 std::vector<int64_t> spm;
 for_each_fmb_execution_step(spec,0,[&](const FmbExecutionStepView& step){
  const auto& chunk=*step.chunk;const auto layer=step.layer;
  assert(step.kind==SpmFmbOccurrenceKind::LayerBody && step.group_id==0);
  assert(kv[layer].size()==static_cast<size_t>(chunk.offset));
  assert(chunk.kv_seq_len==chunk.offset+chunk.len);
  if(resident){
   model.ctx().input_in_spm=step.input_in_spm;model.ctx().output_to_spm=step.output_to_spm;
   model.consume_qwen3vl_ordinary_prefill_schedule(layer,chunk);
  }
  if(!step.input_in_spm)spm.assign(hidden[layer].begin()+chunk.offset,hidden[layer].begin()+chunk.offset+chunk.len);
  for(int64_t r=0;r<chunk.len;++r){
   const auto row=chunk.offset+r;assert(spm[r]>=0);kv[layer].push_back(spm[r]);
   const int64_t prefix=row==0?0:kv[layer][row-1];
   const auto ds=layer<3 && row%7<3?(layer+1)*(row+1):0;
   spm[r]=(spm[r]*3+prefix+ds+layer+(layer+1==model.N?17:0))%1000003;
   hidden[layer+1][row]=spm[r];
  }
 });
 for(const auto& layer:kv)assert(layer.size()==static_cast<size_t>(length));
 assert(model.ctx().consumed==(resident && length>128?1:0));
 return hidden;
}
int main(){
 for(int size:{2,4})for(auto dtype:{at::kHalf,at::kChar,at::kByte})for(bool acc32:{false,true}){
  CausalDecoderModel model;model.setup(size,dtype);model.linear_acc32_=acc32;model.cold();model.make_plan(272);
  const bool admitted=!(size==4 && dtype==at::kChar);
  assert(model.qwen3vl_ordinary_resident_prefill_profile_==admitted);
  const bool fp16_decode=dtype==at::kHalf;
  assert(model.qwen3vl_fp16_decode_profile_==fp16_decode);
  assert(model.qwen3vl_decode_gemv(1)==(fp16_decode && !acc32));
  assert(!model.qwen3vl_decode_gemv(16));
  assert(!model.qwen3vl_decode_mlp(1,1,true,false,1));
  model.manifest();assert(model.configure().chunk_outer_within_group==admitted);
  assert(model.configure().inter_layer_io==(admitted?InterLayerIO::SPM_RESIDENT:InterLayerIO::AUTO));
  if(!admitted){assert(model.ctx().descriptor.empty());continue;}
  assert((model.ctx().descriptor==std::vector<int64_t>{272,128,3,model.N,model.H,model.I,model.Q,8,128,8,dtype,acc32}));
  for(int64_t length:{128,256,272,560})assert(run(model,length,true)==run(model,length,false));
  model.make_plan(272);model.manifest();const auto args=model.ctx().descriptor;
  for(size_t i=0;i<args.size();++i){
   model.ctx().descriptor=args;++model.ctx().descriptor[i];model.ctx().input_in_spm=false;model.ctx().output_to_spm=true;
   bool rejected=false;try{model.consume_qwen3vl_ordinary_prefill_schedule(0,{0,0,128,128});}catch(const std::runtime_error&){rejected=true;}
   assert(rejected);
  }
 }
 for(int fault=0;fault<15;++fault){
  CausalDecoderModel b;b.setup(2,at::kChar);
  if(fault==0)b.N=27;if(fault==1)b.H=2560;if(fault==2)b.I=9728;if(fault==3)b.Q=32;
  if(fault==4)b.KV=4;if(fault==5)b.D=64;if(fault==6)b.cores=4;if(fault==7)b.has_scale=false;
  if(fault==8)b.has_nvfp4=true;if(fault==9)b.use_silu=false;if(fault==10)b.has_qk_norm_=false;
  if(fault==11)b.has_qkv_bias_=true;if(fault==12)b.mrope_section={16,24,24};
  if(fault==13)b.deepstack_lang_layers={5,11,17};if(fault==14)b.deepstack_lang_layers={};
  b.cold();assert(!b.qwen3vl_ordinary_resident_prefill_profile_);
 }
 for(auto field:{&CausalDecoderModel::q_w_list,&CausalDecoderModel::k_w_list,&CausalDecoderModel::v_w_list,
                 &CausalDecoderModel::o_w_list,&CausalDecoderModel::gate_list,&CausalDecoderModel::up_list,&CausalDecoderModel::down_list}){
  for(auto dtype:{at::kHalf,at::kChar,at::kByte}){
   for(int size:{2,4}) {
    CausalDecoderModel b;b.setup(size,dtype);(b.*field)[b.N-1].dtype=at::kFloat;b.cold();
    assert(!b.qwen3vl_ordinary_resident_prefill_profile_);
    assert(!b.qwen3vl_fp16_decode_profile_ && !b.qwen3vl_decode_gemv(1));
   }
  }
 }
 DerivedDecoder derived;derived.setup(2,at::kChar);derived.cold();assert(!derived.qwen3vl_ordinary_resident_prefill_profile_);
 for(int fault=0;fault<15;++fault){
  CausalDecoderModel b;b.setup(4,at::kHalf);
  if(fault==0)b.N=35;if(fault==1)b.H=2544;if(fault==2)b.I=9744;if(fault==3)b.Q=16;
  if(fault==4)b.KV=4;if(fault==5)b.D=64;if(fault==6)b.cores=4;if(fault==7)b.has_scale=true;
  if(fault==8)b.has_nvfp4=true;if(fault==9)b.use_silu=false;if(fault==10)b.has_qk_norm_=false;
  if(fault==11)b.has_qkv_bias_=true;if(fault==12)b.mrope_section={16,24,24};
  if(fault==13)b.deepstack_lang_layers={5,11,17};if(fault==14)b.deepstack_lang_layers={};
  b.cold();assert(!b.qwen3vl_fp16_decode_profile_ && !b.qwen3vl_decode_gemv(1));
 }
 DerivedDecoder fp16_derived;fp16_derived.setup(4,at::kHalf);fp16_derived.cold();
 assert(!fp16_derived.qwen3vl_fp16_decode_profile_);
 CausalDecoderModel model;model.setup(2,at::kChar);model.cold();
 for(int fault=0;fault<9;++fault){
  model.make_plan(272);model.ctx().seq_len=fault==0?1:272;model.ctx().position=fault==1?1:0;
  model.ctx().is_causal=fault!=2;model.ctx().attention_mask=fault==3?std::optional<int>(1):std::nullopt;
  model.ctx().batch_size=fault==4?2:1;model.planning_mode_=fault==5?CausalDecoderPlanningMode::OTHER:CausalDecoderPlanningMode::ORDINARY;
  if(fault==6)model.make_plan(128);if(fault==7)model.make_plan(1);
  if(fault==8)model.qwen3vl_ordinary_resident_prefill_profile_=false;
  model.manifest();assert(model.ctx().descriptor.empty());assert(!model.configure().chunk_outer_within_group);
 }
 for(auto flag:{&CausalDecoderModel::adarms_,&CausalDecoderModel::decoder_mixed_precision_,
                &CausalDecoderModel::decode_gate_up_fusion_,&CausalDecoderModel::wall_prefill_staged_mlp_,
                &CausalDecoderModel::pre_layers_residual1_ready_,&CausalDecoderModel::wall_z1_dispatch_,
                &CausalDecoderModel::qwen3vl_pooler_z1_dispatch_,&CausalDecoderModel::qwen3vl_multiview_text_composite_dispatch_}){
  CausalDecoderModel b;b.setup(2,at::kByte);b.cold();b.make_plan(272);b.*flag=true;
  b.manifest();assert(b.ctx().descriptor.empty());assert(!b.configure().chunk_outer_within_group);
 }
 for(int fault=0;fault<4;++fault){
  CausalDecoderModel b;b.setup(4,at::kHalf);b.cold();b.make_plan(272);b.manifest();
  if(fault==0)b.ctx().stage_plan.chunk_mode=ChunkMode::KV_FIRST;
  if(fault==1)b.ctx().input_in_spm=true;if(fault==2)b.ctx().output_to_spm=false;
  const int layer=fault==3?35:0;
  bool rejected=false;try{b.consume_qwen3vl_ordinary_prefill_schedule(layer,{0,0,128,128});}catch(const std::runtime_error&){rejected=true;}
  assert(rejected);
 }
}
'''
    for token, value in {"COLD": cold, "METHODS": methods, "MANIFEST": manifest,
                         "CONFIG": config, "TRAVERSAL": traversal}.items():
        program = program.replace(f"__{token}__", value)
    cpp, exe = tmp_path / "ordinary_resident.cpp", tmp_path / "ordinary_resident"
    cpp.write_text(program)
    compiled = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                               "-Wno-misleading-indentation", str(cpp), "-o", str(exe)],
                              capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([str(exe)], capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr


def test_complete_schedule_wiring_preserves_existing_arithmetic():
    source = HEADER.read_text()
    body = _method(source, "void build_layer_subgraph(")
    assert "consume_qwen3vl_ordinary_prefill_schedule(layer_idx, chunk);" in body
    # The new cold flag must not select any projection, activation or SPM layout.
    for marker in ("std::vector<BufferDecl> causal_baseline_buffer_declarations(",
                   "bool ordinary_silu_mul_route_enabled(",
                   "bool qwen3vl_4b_prefill_swiglu(",
                   "bool qwen3vl_decode_gemv("):
        assert "qwen3vl_ordinary_resident_prefill_profile_" not in _method(source, marker)
