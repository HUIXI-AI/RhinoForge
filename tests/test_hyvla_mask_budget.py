"""HyVLA mask residency: real declarations, complete budget and frozen route.

The native probe retains all Persistent/Temp declarations, including action
hooks and resident weights. Only preload callback bodies and tensor/device
access are inert; allocator, capacity predicate and mask emission are real.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]


def _program():
    fmb = (ROOT / "src/core/fused_model_base.h").read_text()
    fmb_cpp = (ROOT / "src/core/fused_model_base.cpp").read_text()
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    allocator = (ROOT / "src/core/rpu_spm_allocator.cpp").read_text()
    causal = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    native = (ROOT / "src/fused/rpu_hyvla_expert_model.cpp").read_text()
    code = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "core/rpu_spm_allocator.h"
#define TORCH_CHECK(x, ...) do {if(!(x))throw std::runtime_error(#x);}while(0)
#define TORCH_INTERNAL_ASSERT(x) assert(x)
#define DWIDTH 2
#define NUM_CORES 8
#define SDPA_MAX_VLM_SIZE 400
#define QWEN3_FUSED_LM_HEAD_MAX_VOCAB 200000
template<class A,class B> int64_t CeilDiv(A a,B b){return (a+b-1)/b;}
template<class A,class B> int64_t Align(A a,B b){return CeilDiv(a,b)*b;}
namespace at {struct Tensor {bool present=true; bool defined()const{return present;} int64_t numel()const{return 32;}};}
struct RpuKernelGraph {static inline bool active=false;static inline int checks=0;
 static bool has_active(){++checks;return active;}};
struct Watermarks {int64_t free=0,temporary=0;
 int64_t free_space()const{return free;}int64_t temporary_used()const{return temporary;}}watermarks;
#undef SPM_ALLOC
#define SPM_ALLOC watermarks
'''
    for marker in ("struct SdpaTiling", "enum class SdpaKernelType", "struct SdpaConfig"):
        code += _definition(helpers, marker) + ";\n"
    for marker in (
        "inline int64_t sdpa_tile_m_v16_initial(", "inline uint32_t GetMhaAttnUnivVlmNeed(",
        "inline std::optional<int64_t> choose_tile_k_aggressive(",
        "inline std::optional<int64_t> choose_tile_k_conservative(",
        "inline std::optional<int64_t> choose_tile_k(", "inline bool sdpa_try_compute_tiling(",
        "inline SdpaTiling sdpa_compute_tiling(", "inline int64_t sdpa_compute_tmp_v16_size(",
    ):
        code += _definition(helpers, marker) + "\n"
    code += _definition(allocator, "SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(")
    code += "\nnamespace v3 {class FusedModelBase;\n"
    for marker in (
        "enum class StorageClass", "enum class BufferScope", "struct BufferDecl",
        "enum class AttentionExecutionPolicy", "enum class FmbRopeTableResidency",
        "enum class FmbForwardOperandResidency", "struct LayoutContext",
        "enum class FmbRouteFamily", "enum class FmbPhysicalManifestState",
        "enum class FmbGraphLifecycle", "enum class FmbLinearAccumulationPolicy",
        "struct FmbRouteManifestEntry", "struct FmbPhysicalExecutionManifest", "struct ChunkInfo",
    ):
        code += _definition(fmb, marker) + ";\n"
    code += "namespace detail {int64_t estimate_temporary_total(const std::vector<BufferDecl>&);}\n"
    code += _definition(fmb_cpp, "int64_t detail::estimate_temporary_total(")
    for marker in ("struct FixedSpmOverhead", "static FixedSpmOverhead estimate_fixed_overhead(",
                   "static int64_t available_temporary_spm_after_reset(", "static bool final_chunk_layout_fits("):
        code += _definition(fmb_cpp, marker) + (";\n" if marker.startswith("struct") else "\n")
    code += r'''
class FusedModelBase {public:
 struct Impl{bool persistent_allocated_=false;}impl; Impl* pimpl_=&impl;
 int64_t minimum_eager_spm_bytes()const{return 0;}
 bool declared_spm_layout_fits(const std::vector<BufferDecl>&)const;
};
'''
    code += _definition(fmb_cpp, "bool FusedModelBase::declared_spm_layout_fits(") + "\n}\n"
    residency = (ROOT / "src/core/rpu_spm_residency.h").read_text()
    code += residency[residency.index("namespace v3::detail {"):]
    code += "\nnamespace v3 {\n"
    code += r'''
class CausalDecoderModel:public FusedModelBase {public:
 int64_t h=1024,nq=16,nkv=8,hd=128,inter=2048,layers=32;
 bool wall_prefill_staged_mlp_=false,decode_gate_up_fusion_=false,prefill_packed_gate_up_strided_=false;
 bool qwen3vl_multiview_text_dry_prepared_=false,qwen3vl_multiview_text_composite_prepared_=false;
 bool lingbot_folded_rmsnorm_=false,lingbot_prefill_kv_only_=false;
 bool lingbot_prefill_deepstack_reuses_down(const LayoutContext&)const{return false;}
 int qwen3vl_pooler_z1_retained_deepstack_count_=0;
 bool adarms_mutable_=false,adarms_fused_bcast_enabled_=false,adarms_=false;
 bool has_qk_norm_=true,has_qkv_bias_=false,nvfp4_=false;
 std::vector<int>deepstack_lang_layers_;
 struct LayerWeights{at::Tensor q_bias_w,k_bias_w,v_bias_w;};
 at::Tensor q_nvfp4_tensor_scales_,k_nvfp4_tensor_scales_,v_nvfp4_tensor_scales_,o_nvfp4_tensor_scales_;
 at::Tensor gate_nvfp4_tensor_scales_,up_nvfp4_tensor_scales_,down_nvfp4_tensor_scales_;
 bool norm_slab_active()const{return false;}int64_t norm_slab_size_bytes()const{return 0;}
 bool qwen3vl_4b_packed_decode_qkv(int64_t,bool,bool,int64_t)const{return false;}
 int64_t hidden_size()const{return h;}int64_t num_q_heads()const{return nq;}
 int64_t num_kv_heads()const{return nkv;}int64_t head_dim()const{return hd;}
 int64_t intermediate_size()const{return inter;}int64_t num_layers()const{return layers;}
 int attn_tp()const{return 8;}int mlp_tp()const{return 8;}int lm_head_tp()const{return 8;}
 SdpaConfig make_sdpa_config(int m)const{return {SdpaKernelType::FLASH_ATTN_SPM,hd,nq,nkv,8,m,0};}
'''
    declaration = _definition(causal, "std::vector<BufferDecl> causal_baseline_buffer_declarations(")
    # Keep every storage declaration and size expression. Callbacks only emit
    # device transfers; removing their bodies does not remove their backing.
    while match := re.search(r"\w+\.preload_callback\s*=\s*\[", declaration):
        callback = _definition(declaration, match.group())
        start = declaration.index(callback)
        end = declaration.index(";", start + len(callback)) + 1
        declaration = declaration[:start] + declaration[end:]
    code += declaration + "\n};\n"
    for marker in ("static int64_t hyvla_grow_kv_slots(", "static bool hyvla_grow_slot(",
                   "static void hyvla_widen_sdpa_mask_slot("):
        code += _definition(native, marker) + "\n"
    code += _definition(native, "enum class HyVlaExpertGraphScheduleRoute") + ";\n"
    code += r'''
constexpr int64_t HYVLA_EXPERT_MASK_SCHEDULE_SITE=1728596774874433151LL;
constexpr int64_t HYVLA_S51_M16_SDPA_TMP_BYTES=262144,HYVLA_S51_K128_SDPA_TMP_BYTES=32768;
class HyVlaExpertModel:public CausalDecoderModel {public:
 bool cold_mask_once_=true,cold_kvpad16_=true,cold_rmsnorm_pad16_=true,unroll_mode_=true,exact_profile=true;
 int64_t action_dim_pad_=32;int action_mlp_cores_=8;
 at::Tensor gateup_resident_bundle_;
 int resident_fp8_scale_layers_=0,o_resident_layers_=0;
 int gateup_resident_layers()const{return 24;}
 int64_t resident_storage_bytes()const{return 6291456+resident_fp8_scale_layers_*32768;}
 bool gateup_resident_profile()const{return exact_profile;}
 bool sdpa_m16_enabled(int64_t,int64_t)const{return false;}
 bool sdpa_k128_enabled(int64_t,int64_t)const{return false;}
 bool optimized_prefix_supported(int64_t)const;
 std::vector<BufferDecl> baseline_buffer_declarations(const LayoutContext&)const;
 std::vector<BufferDecl> planned_buffer_declarations(const LayoutContext&,bool*=nullptr)const;
 std::vector<BufferDecl> declare_buffers(const LayoutContext&);
 FmbForwardOperandResidency mask_residency_for_candidate(const LayoutContext&)const;
 FmbForwardOperandResidency physical_forward_operand_residency(const FmbPhysicalExecutionManifest&)const;
 int64_t mask_schedule_for_layout(const LayoutContext&)const;
};
'''.replace("bool*=", "bool* =")
    for marker in (
        "bool HyVlaExpertModel::optimized_prefix_supported(",
        "std::vector<BufferDecl> HyVlaExpertModel::baseline_buffer_declarations(",
        "std::vector<BufferDecl> HyVlaExpertModel::planned_buffer_declarations(",
        "FmbForwardOperandResidency HyVlaExpertModel::mask_residency_for_candidate(",
        "std::vector<BufferDecl> HyVlaExpertModel::declare_buffers(",
        "int64_t HyVlaExpertModel::mask_schedule_for_layout(",
        "FmbForwardOperandResidency HyVlaExpertModel::physical_forward_operand_residency(",
    ):
        code += _definition(native, marker) + "\n"
    code += r'''
int uploads=0;
void sdpa_dma_mask_to_spm(int,uint32_t,int64_t,int64_t,int){++uploads;}
struct MaskTensor{int64_t cols=211;int64_t size(int)const{return cols;}};
struct Context{
 FmbPhysicalExecutionManifest manifest;
 std::optional<MaskTensor> attention_mask=MaskTensor{};
 int64_t body_iter=0,batch_size=1;bool is_causal=false;
 AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV;
 const FmbPhysicalExecutionManifest&physical_manifest()const{return manifest;}
 const FmbRouteManifestEntry&find_physical_route(FmbRouteFamily,int64_t,int)const{return manifest.routes.at(0);}
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,
                             std::initializer_list<int64_t>args,int invocation)const{
  const auto&r=manifest.routes.at(0);
  TORCH_CHECK(r.family==family&&r.site_id==site&&r.selector==selector&&r.flags==flags&&
              r.invocation==invocation&&r.arguments==std::vector<int64_t>(args));
 }
};
void emit_mask(HyVlaExpertModel&model,Context&context,int layer_idx){
 ChunkInfo chunk{0,0,51,context.attention_mask->cols};int64_t seq_len=51,allocation_rows=51;
 int tp=model.attn_tp(),prepared_attn_mask_=0;uint32_t mask_off=0;
'''
    body = _definition(native, "void HyVlaExpertModel::build_layer_subgraph(")
    start = body.index("    const auto& mask_route =")
    end = body.index("    if (ctx().has_complete_physical_manifest()) {", start)
    emit = body[start:end].replace("ctx()", "context")
    for name in ("num_layers", "mask_schedule_for_layout", "physical_forward_operand_residency"):
        emit = emit.replace(name + "(", "model." + name + "(")
    code += emit.replace("cold_mask_once_", "model.cold_mask_once_") + "}\n"
    code += r'''
template<class F>void rejects(F fn){bool bad=false;try{fn();}catch(const std::runtime_error&){bad=true;}assert(bad);}
void set_watermarks(HyVlaExpertModel&m,int64_t held,int64_t fixed,bool warm){
 m.impl.persistent_allocated_=warm;watermarks.temporary=4096;
 watermarks.free=SpmAllocator::SPM_USABLE-held-(warm?fixed:0)-4096;
}
const BufferDecl&slot(const std::vector<BufferDecl>&d,const char*n){
 for(const auto&b:d)if(std::string(b.name)==n)return b;throw std::runtime_error("missing");
}
FmbPhysicalExecutionManifest manifest(int64_t rows,int64_t prefix,int selector,bool once=true){
 FmbPhysicalExecutionManifest m;m.state=FmbPhysicalManifestState::COMPLETE;
 m.logical_length=m.physical_length=rows;m.kv_logical_length=prefix+rows;
 m.routes.push_back({HYVLA_EXPERT_MASK_SCHEDULE_SITE,FmbRouteFamily::GRAPH_SCHEDULE,selector,0,
                     {once?1:0,rows,prefix+rows,8,32},0});return m;
}
void verify_nonoverlap(const std::vector<BufferDecl>&d){
 std::vector<SpmAllocator::AllocRequest>r;size_t mi=0;
 for(const auto&b:d)if(b.storage==StorageClass::Temp&&!b.alias_of){
  if(std::string(b.name)=="sdpa_mask"){mi=r.size();assert(b.phase_start==0&&b.phase_end==9);}
  r.push_back({b.size,b.phase_start,b.phase_end,int(b.scope)});
 }
 const auto p=SpmAllocator::plan_temporary_aliased(r);
 for(size_t i=0;i<r.size();++i)if(i!=mi)
  assert(p.offsets[i]+r[i].bytes<=p.offsets[mi]||p.offsets[mi]+r[mi].bytes<=p.offsets[i]);
}
}
int main(){
 using namespace v3;using Mode=FmbForwardOperandResidency;
 HyVlaExpertModel m;LayoutContext lc;lc.chunk_size=51;lc.planning_chunk_capacity=64;
 lc.num_layers=32;lc.batch_size=1;lc.use_attn_mask=true;lc.is_causal=false;
 // FP16 GU6, W8 GU12, NVFP4 GU24 all own exactly 6 MiB/core.
 // NVFP4 additionally retains all seven complete aligned FP32 scale tables.
 for(int precision:{16,8,4})for(int prefix:{128,144,160,176,192,208,224,240}){
  m.nvfp4_=precision==4;lc.max_kv_seq_len=prefix+51;
  const auto base=m.baseline_buffer_declarations(lc);
  const auto fixed=estimate_fixed_overhead(base);
  assert(fixed.persistent==149504+(m.nvfp4_?1792:0)&&fixed.temp_per_layer==0);
  const auto trial=detail::make_forward_spm_residency_candidate(base,{"sdpa_mask"});
  auto capacity=lc;capacity.chunk_size=64;
  const auto cap=detail::make_forward_spm_residency_candidate(m.baseline_buffer_declarations(capacity),{"sdpa_mask"});
  assert(trial.valid&&cap.valid);verify_nonoverlap(trial.declarations);
  const auto required=std::max(trial.retained_peak_bytes,cap.retained_peak_bytes);
  for(bool warm:{false,true}){
   set_watermarks(m,0,fixed.persistent,warm);
   assert(m.mask_residency_for_candidate(lc)==Mode::FORWARD);
   const int64_t threshold=SpmAllocator::SPM_USABLE-fixed.persistent-65536-required;
   set_watermarks(m,threshold,fixed.persistent,warm);
   assert(m.mask_schedule_for_layout(lc)==7);
   set_watermarks(m,threshold+256,fixed.persistent,warm);
   assert(final_chunk_layout_fits(base,warm));
   assert(final_chunk_layout_fits(trial.declarations,warm));
   assert(!final_chunk_layout_fits(cap.declarations,warm));
   assert(m.mask_residency_for_candidate(lc)==Mode::PER_LAYER&&m.mask_schedule_for_layout(lc)==4);
   // Binding cannot reselect when occupancy changes or capture is active.
   lc.forward_operand_residency=Mode::FORWARD;RpuKernelGraph::active=true;RpuKernelGraph::checks=0;
   assert(m.mask_residency_for_candidate(lc)==Mode::FORWARD&&m.mask_schedule_for_layout(lc)==7);
   verify_nonoverlap(m.declare_buffers(lc));assert(RpuKernelGraph::checks==0);
   auto bound_capacity=capacity;bound_capacity.forward_operand_residency=Mode::FORWARD;
   assert(!final_chunk_layout_fits(m.declare_buffers(bound_capacity),warm));
   lc.forward_operand_residency=Mode::PER_LAYER;
   assert(slot(m.declare_buffers(lc),"sdpa_mask").phase_start==1&&m.mask_schedule_for_layout(lc)==4);
   RpuKernelGraph::active=false;lc.forward_operand_residency=Mode::UNSPECIFIED;
  }
 }
 lc.max_kv_seq_len=211;m.nvfp4_=false;
 auto fixed=estimate_fixed_overhead(m.baseline_buffer_declarations(lc));set_watermarks(m,0,fixed.persistent,false);
 assert(m.mask_schedule_for_layout(lc)==7);
 RpuKernelGraph::active=true;rejects([&]{m.mask_residency_for_candidate(lc);});RpuKernelGraph::active=false;
 // An owner outside the exact optimization profile has no installed resident
 // bundle; otherwise the existing installation invariant correctly rejects it.
 m.exact_profile=false;m.gateup_resident_bundle_.present=false;
 auto legacy=m.baseline_buffer_declarations(lc);
 const auto old_policy=detail::plan_forward_spm_residency(legacy,{"sdpa_mask"});
 assert(m.mask_schedule_for_layout(lc)==(old_policy.retained?7:4));
 m.exact_profile=true;m.gateup_resident_bundle_.present=true;
 m.cold_mask_once_=false;assert(m.mask_schedule_for_layout(lc)==3);
 lc.forward_operand_residency=Mode::FORWARD;rejects([&]{m.declare_buffers(lc);});
 lc.forward_operand_residency=static_cast<Mode>(3);rejects([&]{m.declare_buffers(lc);});
 rejects([&]{m.mask_residency_for_candidate(lc);});
 lc.forward_operand_residency=Mode::UNSPECIFIED;m.cold_mask_once_=true;
 lc.use_attn_mask=false;assert(m.mask_residency_for_candidate(lc)==Mode::PER_LAYER);
 lc.forward_operand_residency=Mode::FORWARD;rejects([&]{m.declare_buffers(lc);});
 lc.use_attn_mask=true;
 for(int rows:{51,64})for(int prefix:{128,160,208,240})for(int selector:{4,7}){
  auto d=manifest(rows,prefix,selector);
  assert(m.physical_forward_operand_residency(d)==(selector==7?Mode::FORWARD:Mode::PER_LAYER));
  for(int field=0;field<5;++field){auto bad=d;++bad.routes[0].arguments[field];rejects([&]{m.physical_forward_operand_residency(bad);});}
  for(int field=0;field<7;++field){auto bad=d;
   if(field==0)bad.routes.clear();if(field==1)bad.routes.push_back(bad.routes[0]);
   if(field==2)bad.routes[0].family=FmbRouteFamily::LINEAR;if(field==3)bad.routes[0].flags=1;
   if(field==4)bad.routes[0].invocation=1;if(field==5)bad.routes[0].selector=5;
   if(field==6)bad.routes[0].arguments.pop_back();rejects([&]{m.physical_forward_operand_residency(bad);});
  }
 }
 // Three fresh graph invocations, each uploads its mask once;
 // per-body and disabled schedules retain ten / 320 uploads respectively.
 Context c;
 for(int selector:{3,4,7})for(int replay=0;replay<3;++replay){
  m.cold_mask_once_=selector!=3;c.manifest=manifest(51,160,selector,m.cold_mask_once_);
  uploads=0;RpuKernelGraph::active=true;RpuKernelGraph::checks=0;
  for(int body=0;body<10;++body){c.body_iter=body;for(int layer=0;layer<32;++layer)emit_mask(m,c,layer);}
  assert(uploads==(selector==3?320:selector==4?10:1)&&RpuKernelGraph::checks==0);
  RpuKernelGraph::active=false;
 }
 c.body_iter=0;c.manifest=manifest(51,160,7);m.cold_mask_once_=true;
 c.manifest.routes[0].arguments[2]=259;uploads=0;rejects([&]{emit_mask(m,c,0);});assert(uploads==0);
 m.cold_mask_once_=false;auto d=manifest(51,160,3,false);
 assert(m.physical_forward_operand_residency(d)==Mode::PER_LAYER);
 d.routes[0].selector=7;rejects([&]{m.physical_forward_operand_residency(d);});
 d.state=FmbPhysicalManifestState::UNSPECIFIED;d.routes.clear();
 assert(m.physical_forward_operand_residency(d)==Mode::UNSPECIFIED);
 std::cout<<"HyVLA complete budget, frozen mask route and per-forward upload verified\n";
}
'''
    return code


def test_hyvla_complete_mask_budget_and_frozen_schedule(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    cpp, binary = tmp_path / "hyvla_mask.cpp", tmp_path / "hyvla_mask"
    cpp.write_text(_program())
    result = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src"),
                             str(cpp), "-o", str(binary)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    print(result.stdout)


def test_gu8_reuses_finished_sdpa_scratch_with_stable_three_owner_budget(tmp_path):
    code = _program().split("int main(){", 1)[0]
    code += r"""
int main(){
 using namespace v3;using Mode=FmbForwardOperandResidency;
 HyVlaExpertModel m;m.nvfp4_=true;m.resident_fp8_scale_layers_=8;
 LayoutContext l;l.chunk_size=51;l.planning_chunk_capacity=64;l.max_kv_seq_len=259;
 l.num_layers=32;l.batch_size=1;l.use_attn_mask=true;l.is_causal=false;
 for(int rows:{51,64}){
  l.chunk_size=rows;auto d=m.declare_buffers(l);auto f=estimate_fixed_overhead(d);
  assert(f.persistent==151296&&f.temp_per_layer==0);
  assert(slot(d,"gateup_resident_weights").size==6553600);
  assert(std::string(slot(d,"sdpa_tmp").alias_of)=="oproj");
  assert(slot(d,"sdpa_tmp").size==0&&slot(d,"oproj").size==(rows==51?104448:131072));
  assert(detail::estimate_temporary_total(d)==(rows==51?7055104:7086080));
  for(bool warm:{false,true}){
   // First action can have expert alone. Warm replay has all three owners;
   // temporary buffers are recovered and Persistent is charged exactly once.
   set_watermarks(m,warm?1059840:0,f.persistent,warm);
   assert(final_chunk_layout_fits(d,warm));assert(m.mask_schedule_for_layout(l)==4);
   if(warm)assert(available_temporary_spm_after_reset(d,warm)==7110912);
   const auto threshold=SpmAllocator::SPM_USABLE-f.persistent-65536-detail::estimate_temporary_total(d);
   set_watermarks(m,threshold,f.persistent,warm);assert(final_chunk_layout_fits(d,warm));
   set_watermarks(m,threshold+256,f.persistent,warm);assert(!final_chunk_layout_fits(d,warm));
  }
  l.forward_operand_residency=Mode::FORWARD;
  rejects([&]{m.declare_buffers(l);});rejects([&]{m.mask_schedule_for_layout(l);});
  l.forward_operand_residency=Mode::UNSPECIFIED;
 }
 l.chunk_size=51;Context c;c.attention_mask->cols=259;c.manifest=manifest(51,208,4);uploads=0;
 for(int body=0;body<10;++body){int before=uploads;c.body_iter=body;for(int layer=0;layer<32;++layer)emit_mask(m,c,layer);assert(uploads-before==1);}
 assert(uploads==10);
 m.nvfp4_=false;rejects([&]{m.declare_buffers(l);});m.nvfp4_=true;
 m.o_resident_layers_=1;rejects([&]{m.declare_buffers(l);});
}
"""
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    cpp, binary = tmp_path / "hyvla_gu8.cpp", tmp_path / "hyvla_gu8"
    cpp.write_text(code)
    result = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src"),
                             str(cpp), "-o", str(binary)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
