"""HY/HALO action mask residency: production declarations and real allocator.

No model math, hardware execution, or performance claim is inferred here.
Persistent callbacks are omitted from the standalone allocator fixture only:
they cannot change the temporary peak and their product declarations stay intact.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]


def _program():
    base = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    hy = (ROOT / "src/fused/rpu_hyvla_expert_model.cpp").read_text()
    halo = (ROOT / "src/fused/rpu_halo_action_expert_model.cpp").read_text()
    fmb = (ROOT / "src/core/fused_model_base.h").read_text()
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    allocator = (ROOT / "src/core/rpu_spm_allocator.cpp").read_text()
    allocator_h = (ROOT / "src/core/rpu_spm_allocator.h").read_text()
    p = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do {if(!(c))throw std::runtime_error("guard");}while(0)
#define TORCH_INTERNAL_ASSERT(c, ...) TORCH_CHECK(c, __VA_ARGS__)
#define DWIDTH 2
#define NUM_CORES 8
#define SDPA_MAX_VLM_SIZE 400
#define RPU_SDPA_TILE_K_CONSERVATIVE_MODE 1
#define QWEN3_FUSED_LM_HEAD_MAX_VOCAB 200000
template<class T> T CeilDiv(T a,T b) {return (a+b-1)/b;}
template<class T,class U> T Align(T a,U b) {return CeilDiv(a,static_cast<T>(b))*b;}
'''
    for marker in ("struct SdpaTiling", "enum class SdpaKernelType", "struct SdpaConfig"):
        p += _definition(helpers, marker) + ";\n"
    for marker in (
        "inline int64_t sdpa_tile_m_v16_initial(",
        "inline uint32_t GetMhaAttnUnivVlmNeed(",
        "inline std::optional<int64_t> choose_tile_k_aggressive(",
        "inline std::optional<int64_t> choose_tile_k_conservative(",
        "inline std::optional<int64_t> choose_tile_k(",
        "inline bool sdpa_try_compute_tiling(",
        "inline SdpaTiling sdpa_compute_tiling(",
        "inline int64_t sdpa_compute_tmp_v16_size(",
    ):
        p += _definition(helpers, marker) + "\n"
    p += "class SpmAllocator {public: static constexpr size_t ALIGN=256, SPM_USABLE=8*1024*1024-1024;\n"
    p += "static size_t align_up(size_t n){return (n+255)&~size_t{255};}\n"
    for marker in ("struct AllocRequest", "struct AliasedPlan"):
        p += _definition(allocator_h, marker) + ";\n"
    p += "static AliasedPlan plan_temporary_aliased(const std::vector<AllocRequest>&);};\n"
    p += _definition(allocator, "SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(")
    p += "namespace v3 { class FusedModelBase {};\n"
    for marker in ("enum class StorageClass", "enum class BufferScope", "struct BufferDecl", "enum class FmbForwardOperandResidency", "struct LayoutContext"):
        if marker == "struct LayoutContext":
            p += "enum class AttentionExecutionPolicy {DDR_KV,SPM_KV_BY_MHA};\n"
            p += "enum class FmbRopeTableResidency {UNSPECIFIED};\n"
        p += _definition(fmb, marker) + ";\n"
    p += "namespace detail {int64_t estimate_temporary_total(const std::vector<BufferDecl>&);}\n"
    p += _definition((ROOT / "src/core/fused_model_base.cpp").read_text(),
                     "int64_t detail::estimate_temporary_total(")
    p += "}\n"
    residency = (ROOT / "src/core/rpu_spm_residency.h").read_text()
    p += residency[residency.index("namespace v3::detail {"):]
    p += "\nnamespace v3 {\n"
    tiling = (ROOT / "src/ops/rpu_hyvla_sdpa_tiling.h").read_text()
    p += "\n".join(re.findall(r"constexpr int64_t HYVLA_S51_\w+_SDPA_TMP_BYTES = \d+;", tiling))
    p += r'''
constexpr int64_t HYVLA_EXPERT_MASK_SCHEDULE_SITE=1728596774874433151LL;
enum class FmbRouteFamily {GRAPH_SCHEDULE};
enum class FmbPhysicalManifestState {COMPLETE};
struct FmbRouteManifestEntry {
 FmbRouteFamily family=FmbRouteFamily::GRAPH_SCHEDULE;
 int64_t site_id=HYVLA_EXPERT_MASK_SCHEDULE_SITE,flags=0,invocation=0,selector=7;
 std::vector<int64_t> arguments={1,51,291,8,32};
};
struct FmbPhysicalExecutionManifest {
 FmbPhysicalManifestState state=FmbPhysicalManifestState::COMPLETE;
 int64_t physical_length=51,kv_logical_length=291;
 std::vector<FmbRouteManifestEntry> routes;
};
'''
    p += r'''
class CausalDecoderModel {public:
 int64_t h=1024,nq=16,nkv=8,hd=128,inter=2048,layers=32;
 bool wall_prefill_staged_mlp_=false,decode_gate_up_fusion_=false;
 bool prefill_packed_gate_up_strided_=false;
 bool qwen3vl_multiview_text_dry_prepared_=false,qwen3vl_multiview_text_composite_prepared_=false;
 int qwen3vl_pooler_z1_retained_deepstack_count_=0;
 std::vector<int> deepstack_lang_layers_;
 int64_t hidden_size()const{return h;} int64_t num_q_heads()const{return nq;}
 int64_t num_kv_heads()const{return nkv;} int64_t head_dim()const{return hd;}
 int64_t intermediate_size()const{return inter;} int64_t num_layers()const{return layers;}
 int attn_tp()const{return std::min(int(nkv),8);} int mlp_tp()const{return 8;}
 int lm_head_tp()const{return 8;}
 // This fixture has no packed Qwen weights or LingBot deepstack input.
 bool qwen3vl_4b_packed_decode_qkv(int64_t,bool,bool,int64_t)const{return false;}
 bool lingbot_prefill_deepstack_reuses_down(const LayoutContext&)const{return false;}
 SdpaConfig make_sdpa_config(int m)const{return {SdpaKernelType::FLASH_ATTN_SPM,hd,nq,nkv,attn_tp(),m,0};}
'''
    decl = _definition(base, "std::vector<BufferDecl> causal_baseline_buffer_declarations(")
    cut = decl.index("        // Norm-weight preloads")
    assert "StorageClass::Temp" not in decl[cut:]
    p += decl[:cut] + "return decls;\n}\nstd::vector<BufferDecl> declare_buffers(const LayoutContext& l){return causal_baseline_buffer_declarations(l); }\n};\n"
    # HY's baseline consists solely of its exact extra Temp declarations.
    for marker in ("static int64_t hyvla_grow_kv_slots(", "static bool hyvla_grow_slot(",
                   "static void hyvla_widen_sdpa_mask_slot("):
        p += _definition(hy, marker) + "\n"
    p += _definition(hy, "enum class HyVlaExpertGraphScheduleRoute") + ";\n"
    p += r'''
class HyVlaExpertModel:public CausalDecoderModel {public:
 int resident_fp8_scale_layers_=0,o_resident_layers_=0;bool nvfp4_=false;
 int gateup_resident_layers()const{return 0;}
 bool cold_mask_once_=true,cold_kvpad16_=true,cold_rmsnorm_pad16_=true,unroll_mode_=true;
 int64_t action_dim_pad_=32;int action_mlp_cores_=8;
 struct UnboundBundle {bool defined()const{return false;}} gateup_resident_bundle_;
 bool sdpa_m16_enabled(int64_t,int64_t)const{return false;}
 bool sdpa_k128_enabled(int64_t,int64_t)const{return false;}
 bool gateup_resident_profile()const{return false;}
 bool optimized_prefix_supported(int64_t)const{return true;}
 int64_t resident_storage_bytes()const{throw std::runtime_error("unbound resident weights");}
 bool declared_spm_layout_fits(const std::vector<BufferDecl>&)const{
  throw std::runtime_error("resident profile is not bound in this fixture");
 }
 std::vector<BufferDecl> baseline_buffer_declarations(const LayoutContext&)const;
 std::vector<BufferDecl> planned_buffer_declarations(const LayoutContext&,bool* = nullptr)const;
 int64_t mask_schedule_for_layout(const LayoutContext&)const;
 FmbForwardOperandResidency mask_residency_for_candidate(const LayoutContext&)const;
 FmbForwardOperandResidency physical_forward_operand_residency(const FmbPhysicalExecutionManifest&)const;
};
class HaloActionExpertModel:public CausalDecoderModel {public:
 int64_t moe_chunk_size_=18;
 std::vector<BufferDecl> baseline_buffer_declarations(const LayoutContext&)const;
 std::vector<BufferDecl> declare_buffers(const LayoutContext&l){return baseline_buffer_declarations(l);}
 // Exercise the no-growth helper independently of HALO's capacity admission.
 std::vector<BufferDecl> planned_buffer_declarations(const LayoutContext&l,bool* retained=nullptr){
  auto d=declare_buffers(l);auto plan=detail::plan_forward_spm_residency(d,{"sdpa_mask"});
  if(retained)*retained=plan.retained;return d;
 }
 int64_t mask_schedule_for_layout(const LayoutContext&l){bool keep=false;planned_buffer_declarations(l,&keep);return keep?2:1;}
};
'''
    for model, source in (("HyVlaExpertModel", hy), ("HaloActionExpertModel", halo)):
        method = "baseline_buffer_declarations"
        decl = _definition(source, f"std::vector<BufferDecl> {model}::{method}(")
        if model.startswith("Halo"):
            cut = decl.index("    // ── 行掩码 Persistent")
            assert "StorageClass::Temp" not in decl[cut:]
            decl = decl[:cut] + "return d;\n}"
        p += decl + "\n"
        if model.startswith("Hy"):
            p += _definition(source, f"std::vector<BufferDecl> {model}::planned_buffer_declarations(") + "\n"
            p += _definition(source, f"FmbForwardOperandResidency {model}::mask_residency_for_candidate(") + "\n"
            p += _definition(source, f"FmbForwardOperandResidency {model}::physical_forward_operand_residency(") + "\n"
            p += _definition(source, f"int64_t {model}::mask_schedule_for_layout(") + "\n"
    body = _definition(hy, "void HyVlaExpertModel::build_layer_subgraph(")
    start = body.index("    const auto& mask_route =")
    end = body.index("    if (ctx().has_complete_physical_manifest()) {", start)
    emit = body[start:end].replace("ctx()", "context").replace("num_layers()", "model.num_layers()")
    emit = emit.replace("mask_schedule_for_layout(layout)", "model.mask_schedule_for_layout(layout)")
    emit = emit.replace("cold_mask_once_", "model.cold_mask_once_")
    emit = emit.replace("physical_forward_operand_residency(", "model.physical_forward_operand_residency(")
    p += r'''
struct ChunkInfo {int idx=0;int64_t offset=0,len=51,kv_seq_len=291;};
struct MaskTensor {int64_t size(int)const{return 291;}};
struct Context {
 struct Route {int64_t selector=7,flags=0;std::vector<int64_t>args={1,51,291,8,32};}route;
 std::optional<MaskTensor>attention_mask=MaskTensor{};
 bool is_causal=false;int64_t body_iter=0,batch_size=1;
 AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV;
 const Route& find_physical_route(FmbRouteFamily,int64_t,int)const{return route;}
 FmbPhysicalExecutionManifest physical_manifest()const{
  FmbPhysicalExecutionManifest manifest;
  FmbRouteManifestEntry mask;mask.selector=route.selector;mask.flags=route.flags;mask.arguments=route.args;
  manifest.routes.push_back(mask);return manifest;
 }
 void consume_physical_route(FmbRouteFamily,int64_t,int64_t selector,int64_t flags,
                            std::initializer_list<int64_t> args,int)const{
  TORCH_CHECK(selector==route.selector&&flags==route.flags&&std::vector<int64_t>(args)==route.args);
 }
};
int uploads=0;
void sdpa_dma_mask_to_spm(int,uint32_t,int64_t,int64_t,int){++uploads;}
void emit_mask(HyVlaExpertModel& model,Context&context,int layer_idx){
 ChunkInfo chunk;int64_t seq_len=51,allocation_rows=51;int tp=model.attn_tp();
 uint32_t mask_off=0;int prepared_attn_mask_=0;
'''
    p += emit + "}\n"
    p += r'''
void verify_route(){
 HyVlaExpertModel model;Context context;
 for(bool once:{false,true}){
  model.cold_mask_once_=once;context.route.selector=once?7:3;context.route.args[0]=once?1:0;uploads=0;
  for(int body=0;body<10;++body){context.body_iter=body;for(int layer=0;layer<32;++layer)emit_mask(model,context,layer);}
  assert(uploads==(once?1:320));
 }
 model.cold_mask_once_=true;context.body_iter=0;
 for(int field=0;field<7;++field){
  context.route={};if(field==0)context.route.selector=-1;else if(field==1)context.route.flags=1;else ++context.route.args[field-2];
  uploads=0;bool rejected=false;try{emit_mask(model,context,0);}catch(const std::runtime_error&){rejected=true;}
  assert(rejected&&uploads==0);
 }
}
'''
    p += r'''
void verify_retention(const std::vector<BufferDecl>& d) {
 std::vector<SpmAllocator::AllocRequest> requests;size_t mask=0;
 for(auto& b:d) if(b.storage==StorageClass::Temp&&!b.alias_of) {
  if(std::string(b.name)=="sdpa_mask") {mask=requests.size();assert(b.phase_start==0&&b.phase_end==9);}
  requests.push_back({b.size,b.phase_start,b.phase_end,int(b.scope)});
 }
 auto plan=SpmAllocator::plan_temporary_aliased(requests);
 size_t lo=plan.offsets[mask],hi=lo+requests[mask].bytes;
 std::vector<unsigned char> spm(plan.peak_bytes,0);
 std::fill(spm.begin()+lo,spm.begin()+hi,0xa5);
 for(int body=0;body<10;++body)for(size_t i=0;i<requests.size();++i)if(i!=mask){
  size_t a=plan.offsets[i],b=a+requests[i].bytes;
  assert(b<=lo||a>=hi);
  std::fill(spm.begin()+a,spm.begin()+b,body+1);
 }
 assert(std::all_of(spm.begin()+lo,spm.begin()+hi,[](auto x){return x==0xa5;}));
}
bool same(const std::vector<BufferDecl>&a,const std::vector<BufferDecl>&b){
 if(a.size()!=b.size())return false;
 for(size_t i=0;i<a.size();++i) if(a[i].size!=b[i].size||a[i].phase_start!=b[i].phase_start||
  a[i].phase_end!=b[i].phase_end||a[i].scope!=b[i].scope||a[i].storage!=b[i].storage)return false;
 return true;
}
template<class Model> void run(Model& m,const char* name,int64_t rows,int64_t kv) {
 LayoutContext l;l.chunk_size=rows;l.max_kv_seq_len=kv;l.use_attn_mask=true;l.is_causal=false;
 auto before=m.baseline_buffer_declarations(l);bool retained=false;
 auto after=m.planned_buffer_declarations(l,&retained);
 auto old_peak=detail::estimate_temporary_total(before),new_peak=detail::estimate_temporary_total(after);
 assert(new_peak<=old_peak);if(retained)verify_retention(after);else assert(same(before,after));
 auto recheck=before;auto decision=detail::plan_forward_spm_residency(recheck,{"sdpa_mask"});
 assert(decision.retained==retained);assert(same(recheck,after));
 // Future outside-hook scratch is a hard fallback, not implicit alias safety.
 auto outside=before;outside.push_back({"future_hook",256,0,0,StorageClass::Temp,0,nullptr,BufferScope::OutsideLayerLoop});
 auto saved=outside;assert(!detail::plan_forward_spm_residency(outside,{"sdpa_mask"}).retained);assert(same(saved,outside));
 auto missing=before;assert(!detail::plan_forward_spm_residency(missing,{"missing"}).retained);assert(same(before,missing));
 auto duplicate=before;for(auto&d:before)if(std::string(d.name)=="sdpa_mask"){duplicate.push_back(d);break;}
 saved=duplicate;assert(!detail::plan_forward_spm_residency(duplicate,{"sdpa_mask"}).retained);assert(same(saved,duplicate));
 std::cout<<name<<","<<m.attn_tp()<<","<<rows<<","<<kv<<","<<retained<<","<<old_peak<<","<<new_peak<<","<<decision.retained_peak_bytes<<","<<m.mask_schedule_for_layout(l)<<"\n";
}
}
int main(){using namespace v3;verify_route();
 HyVlaExpertModel hy;
 for(int nkv:{4,8})for(int rows:{51,64})for(bool unroll:{false,true}){
  hy.nkv=nkv;hy.unroll_mode_=unroll;run(hy,unroll?"hy_unroll":"hy_step",rows,291);
 }
 LayoutContext l;l.chunk_size=51;l.max_kv_seq_len=291;l.use_attn_mask=true;l.is_causal=false;
 hy.cold_mask_once_=false;bool retained=true;auto original=hy.baseline_buffer_declarations(l);
 auto disabled=hy.planned_buffer_declarations(l,&retained);assert(!retained&&same(original,disabled));
 assert(hy.mask_schedule_for_layout(l)==3);
 HaloActionExpertModel halo;halo.h=1536;halo.nq=12;halo.nkv=2;halo.inter=8960;halo.layers=28;
 for(int rows:{18,32})for(int kv:{4724,8192})run(halo,"halo",rows,kv);
 // A true capacity-growth layout must preserve old offsets/phases completely.
 std::vector<BufferDecl> tight={{"mask",256,3,5,StorageClass::Temp},{"mlp",512,7,8,StorageClass::Temp}};
 auto saved=tight;auto negative=detail::plan_forward_spm_residency(tight,{"mask"});
 assert(!negative.retained&&negative.retained_peak_bytes>negative.original_peak_bytes&&same(saved,tight));
}
'''
    return p


def test_actual_allocator_covers_action_hooks_or_keeps_original_layout(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    cpp, binary = tmp_path / "mask.cpp", tmp_path / "mask"
    cpp.write_text(_program())
    build = subprocess.run([compiler, "-std=c++17", "-O0", str(cpp), "-o", str(binary)], text=True, capture_output=True)
    assert build.returncode == 0, build.stderr
    result = subprocess.run([str(binary)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    print(result.stdout)
    assert len(result.stdout.splitlines()) == 12


def test_each_owner_uses_complete_pure_declarations_for_descriptor_and_emit():
    for name, site in (("HyVlaExpertModel", "HYVLA_EXPERT_MASK_SCHEDULE_SITE"),):
        source = (ROOT / "src/fused" / ("rpu_hyvla_expert_model.cpp" if name.startswith("Hy") else "rpu_halo_action_expert_model.cpp")).read_text()
        baseline = _definition(source, f"std::vector<BufferDecl> {name}::baseline_buffer_declarations(")
        planned = _definition(source, f"std::vector<BufferDecl> {name}::planned_buffer_declarations(")
        manifest = _definition(source, f"{name}::physical_manifest_for_candidate(")
        body = _definition(source, f"void {name}::build_layer_subgraph(")
        assert "causal_baseline_buffer_declarations(" in baseline
        assert "CausalDecoderModel::declare_buffers(" not in baseline
        assert "baseline_buffer_declarations(layout)" in planned
        assert 'plan_forward_spm_residency(declarations, {"sdpa_mask"})' in planned
        assert "mask_schedule_for_layout(layout)" in manifest and site in manifest
        assert "mask_schedule_for_layout(layout)" in body and site in body
        assert body.index("mask_schedule_for_layout(layout)") < body.index("sdpa_dma_mask_to_spm(")
        assert body.count("sdpa_dma_mask_to_spm(") == 1
    hy = (ROOT / "src/fused/rpu_hyvla_expert_model.cpp").read_text()
    baseline = _definition(hy, "std::vector<BufferDecl> HyVlaExpertModel::baseline_buffer_declarations(")
    assert "kv_pad_rows_ =" not in baseline
    assert "const int64_t rmsnorm_pad_rows_" in baseline
    assert "ctx().body_iter == 0 && layer_idx == 0" in hy
