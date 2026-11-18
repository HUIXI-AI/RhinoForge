"""Host-only checks of the real Z2 Text cold producer/selected-descriptor sink."""

from pathlib import Path
import shutil
import subprocess

from test_native_kvinsert_cost_domain import _body


ROOT = Path(__file__).resolve().parents[1]


def test_text_dry_prepare_prime_and_forward_share_actual_descriptor_authority():
    source = (ROOT / "src/fused/rpu_qwen3_model.cpp").read_text()
    dry = _body(source, "CausalDecoderModel::prepare_qwen3vl_multiview_text_dry_layout")
    prepare = _body(source, "CausalDecoderModel::prepare_qwen3vl_multiview_text_composite_layout")
    prime = _body(source, "CausalDecoderModel::prime_qwen3vl_multiview_text_inputs")
    helper = _body(source, "CausalDecoderModel::bind_multiview_text_stage_descriptor")
    assert "shape.stage_plan, shape.allocation_layout, /*partial_mrope=*/true" in dry
    assert dry.index("bind_multiview_text_stage_descriptor(") < dry.index("prepare_spm_pipeline_component_for_cpu_contract(")
    assert "std::move(descriptor)" in dry
    assert "shape.stage_plan, shape.allocation_layout," in prepare
    assert "qwen3vl_multiview_text_stage_descriptor_ = std::move(descriptor)" in prepare
    assert "encode_causal_physical_stage_descriptor" not in prime
    assert prime.index("bind_multiview_text_stage_descriptor(") < prime.index("position_dst.copy_(position_ids)")
    assert "qwen3vl_multiview_text_stage_descriptor_);" in prime
    assert 'kvinsert_cost_domain("causal_decoder", selected_descriptor)' in helper
    assert "kvinsert_cost_owner_prefix(selected_descriptor) == owner_prefix" in helper
    assert "layout = bind_physical_layout_context(layout, manifest)" in helper
    header = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    scope = header.split("KvCostLayoutScope capture_kvinsert_cost_layout_scope() override", 1)[1].split("\n    }", 1)[0]
    assert "qwen3vl_multiview_text_dry_prepared_" in scope
    assert "qwen3vl_multiview_text_composite_prepared_" in scope
    outer = (ROOT / "src/fused/rpu_qwen3vl_multiview_spm_z2.cpp").read_text()
    assert "selected Text descriptor disagrees with actual prime" in outer
    assert "lingbot2_child_stage_descriptor(planner_descriptor, true, operation)" in outer


def test_actual_text_binding_preserves_native_exact_and_rejects_stale_or_foreign(tmp_path):
    """Execute the actual binding method, with only codec/SDK boundaries doubled."""
    body = _body((ROOT / "src/fused/rpu_qwen3_model.cpp").read_text(),
                 "CausalDecoderModel::bind_multiview_text_stage_descriptor")
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <memory>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(cond, ...) do { if (!(cond)) throw std::runtime_error("rejected"); } while(0)
namespace at { using IntArrayRef=std::vector<int64_t>; }
constexpr int QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN=225;
constexpr int QWEN3VL_MULTIVIEW_TEXT_RESOLVED_CHUNK_SIZE=240;
constexpr int CAUSAL_DECODER_PARTIAL_MROPE_TENSOR_SITE=11;
constexpr int CAUSAL_DECODER_MROPE_TENSOR_SITE=12;
enum class FmbPhysicalManifestState {COMPLETE, INCOMPLETE};
enum class FmbGraphLifecycle {COMPOSITE_CHILD, RETAINED_CACHE};
enum class CausalDecoderPlanningMode {QWEN3VL_MULTIVIEW_TEXT};
enum class FmbRouteFamily {ROPE, KV_INSERT};
struct Route {FmbRouteFamily family;int64_t site_id;};
struct Plan {struct Stage {struct {int64_t chunk_size=225;} plan;} input;uint64_t fingerprint=17;};
using FmbThreeStageChunkPlan=Plan;
struct Manifest {
 FmbPhysicalManifestState state=FmbPhysicalManifestState::COMPLETE;
 FmbGraphLifecycle graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
 int64_t logical_length=225,physical_length=225;uint64_t fingerprint=31;
 std::vector<Route> routes{{FmbRouteFamily::ROPE,11}};
};
struct Candidate {int64_t input_chunk_size=225,qkv_chunk_size=240,compute_chunk_size=240;
 Plan stage_plan;Manifest physical_manifest;};
std::map<int64_t,Candidate> candidates;
struct Prepared {Candidate value;const Candidate& candidate() const {return value;}};
std::shared_ptr<Prepared> prepare_stage_candidate(const std::vector<int64_t>& raw) {return std::make_shared<Prepared>(Prepared{candidates.at(raw.at(0))});}
uint64_t fmb_three_stage_chunk_plan_fingerprint(const Plan& p) {return p.fingerprint;}
uint64_t fmb_physical_manifest_fingerprint(const Manifest& m) {return m.fingerprint;}
int fmb_rope_table_residency(const Manifest&) {return 3;}
struct FmbPhysicalManifestConsumer {FmbPhysicalManifestConsumer(const Manifest&) {} std::optional<int> attention_layout_policy(){return 4;}};
enum class FmbForwardOperandResidency {UNSPECIFIED,PER_LAYER,FORWARD};
struct LayoutContext {uint64_t stage_plan_fingerprint=0,physical_manifest_fingerprint=0;int rope_table_residency=0,attention_policy=0;FmbForwardOperandResidency forward_operand_residency=FmbForwardOperandResidency::UNSPECIFIED;};
struct CausalDecoderModel {
 FmbForwardOperandResidency physical_forward_operand_residency(const Manifest&)const{return FmbForwardOperandResidency::UNSPECIFIED;}
 LayoutContext bind_physical_layout_context(LayoutContext layout,const Manifest& manifest)const __BIND_LAYOUT__
 int encodes=0;bool current=true;std::map<int64_t,std::vector<int64_t>> prefixes;
 std::vector<int64_t> encode_causal_physical_stage_descriptor(const Plan&,const LayoutContext&,
     int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t rope,FmbGraphLifecycle,CausalDecoderPlanningMode) {
   ++encodes;candidates[101]=Candidate{};candidates[101].physical_manifest.routes[0].site_id=rope?11:12;return {101};
 }
 void bind_kvinsert_cost_owner_prefix(at::IntArrayRef d,std::vector<int64_t> p) {prefixes[d.at(0)]=p;}
 int kvinsert_cost_domain(const char*,at::IntArrayRef d) {TORCH_CHECK(current && prefixes.count(d.at(0)), "stale");return 0;}
 std::vector<int64_t> kvinsert_cost_owner_prefix(at::IntArrayRef d) {return prefixes.at(d.at(0));}
 std::vector<int64_t> bind_multiview_text_stage_descriptor(const Plan&,LayoutContext&,bool,at::IntArrayRef={});
};
std::vector<int64_t> CausalDecoderModel::bind_multiview_text_stage_descriptor(
 const Plan& stage_plan,LayoutContext& layout,bool partial_mrope,at::IntArrayRef selected_descriptor) __BODY__
template<class F> void rejects(F f) {bool failed=false;try{f();}catch(const std::exception&){failed=true;}assert(failed);}
int main() {
 CausalDecoderModel model;Plan plan;LayoutContext dry;
 auto original=model.bind_multiview_text_stage_descriptor(plan,dry,true);
 assert(original==std::vector<int64_t>{101} && model.encodes==1);
 assert(dry.stage_plan_fingerprint==17 && dry.physical_manifest_fingerprint==31);
 // Stand-in for genuine native mint: only this owner's admitted map authorizes it.
 candidates[102]=candidates[101];candidates[102].physical_manifest.fingerprint=99;
 model.prefixes[102]=model.prefixes[101];LayoutContext selected;
 assert(model.bind_multiview_text_stage_descriptor(plan,selected,true,{102})==std::vector<int64_t>{102});
 assert(model.encodes==1 && selected.physical_manifest_fingerprint==99);
 assert(model.bind_multiview_text_stage_descriptor(plan,selected,true,{102})==std::vector<int64_t>{102});
 assert(model.encodes==1); // prime must not overwrite the prepared EXACT route with AUTO.
 rejects([&]{model.bind_multiview_text_stage_descriptor(plan,selected,true,{103});});
 model.current=false;rejects([&]{model.bind_multiview_text_stage_descriptor(plan,selected,true,{102});});model.current=true;
 rejects([&]{model.bind_multiview_text_stage_descriptor(plan,selected,false,{102});});
 model.prefixes[102]={0};rejects([&]{model.bind_multiview_text_stage_descriptor(plan,selected,true,{102});});
 model.prefixes[102]=model.prefixes[101];candidates[102].physical_manifest.graph_lifecycle=FmbGraphLifecycle::RETAINED_CACHE;
 rejects([&]{model.bind_multiview_text_stage_descriptor(plan,selected,true,{102});});
 candidates[102]=candidates[101];candidates[102].qkv_chunk_size=224;
 rejects([&]{model.bind_multiview_text_stage_descriptor(plan,selected,true,{102});});
 candidates[102]=candidates[101];candidates[102].stage_plan.fingerprint=18;
 rejects([&]{model.bind_multiview_text_stage_descriptor(plan,selected,true,{102});});
 candidates[102]=candidates[101];candidates[102].physical_manifest.routes[0].site_id=12;
 rejects([&]{model.bind_multiview_text_stage_descriptor(plan,selected,true,{102});});
 LayoutContext canary;assert(model.bind_multiview_text_stage_descriptor(plan,canary,false)==original);
 assert(model.encodes==2 && candidates[101].physical_manifest.routes[0].site_id==12);
}
'''.replace("__BODY__", body).replace("__BIND_LAYOUT__", _body(
        (ROOT / "src/core/fused_model_base.cpp").read_text(),
        "FusedModelBase::bind_physical_layout_context"))
    source = tmp_path / "text_binding.cpp"
    source.write_text(cpp)
    binary = tmp_path / "text_binding"
    result = subprocess.run([shutil.which("g++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                             str(source), "-o", str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(binary)], check=True)


def test_actual_grouped_fast_predicate_keeps_only_baseline_kernel_stream(tmp_path):
    source = (ROOT / "src/fused/rpu_qwen3vl_multiview_spm_z2.cpp").read_text()
    body = _body(source, "uses_lingbot2_multiview_grouped_outer_fast")
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <optional>
#include <memory>
#include <vector>
enum class KvInsertRoute {V2=1,ALIGNED_V16=2};
namespace v3 {
enum class FmbRouteFamily {KV_INSERT,OTHER};
struct Route {FmbRouteFamily family;int64_t site_id,invocation,selector;};
struct Candidate {struct {std::vector<Route> routes;} physical_manifest;};
Candidate vision, text;
struct Prepared {Candidate value;const Candidate& candidate() const {return value;}};
struct Owner {std::shared_ptr<Prepared> prepare_stage_candidate(const std::vector<int64_t>& d) {return std::make_shared<Prepared>(Prepared{d.at(0)==0?vision:text});}};
namespace qwen3vl_pooler_z1_internal {
Owner& multiview_vision_owner(int64_t) {static Owner owner;return owner;}
Owner& multiview_text_owner(int64_t) {static Owner owner;return owner;}
}
namespace lingbot_v2_moe_internal {enum class ExactSuffixProfileKind {GroupedW8A16,GroupedW4A16,DenseFp16};}
}
using Kind=v3::lingbot_v2_moe_internal::ExactSuffixProfileKind;
struct Profile {Kind kind=Kind::GroupedW8A16;};
struct Qwen3VlMultiviewZ2State {int64_t vision_handle=2,text_handle=3,expert_handle=1;std::optional<Profile> expert_profile=Profile{};std::vector<int64_t> planner_descriptor;};
std::vector<int64_t> lingbot2_child_stage_descriptor(const std::vector<int64_t>&,bool text,const char*) {return {text?1:0};}
bool uses_lingbot2_multiview_grouped_outer_fast(const Qwen3VlMultiviewZ2State& s) __BODY__
int main() {
 Qwen3VlMultiviewZ2State s;
 v3::vision.physical_manifest.routes={{v3::FmbRouteFamily::KV_INSERT,2982723969499025021LL,0,2}};
 v3::text.physical_manifest.routes={{v3::FmbRouteFamily::KV_INSERT,6070783309895644632LL,0,1}};
 assert(uses_lingbot2_multiview_grouped_outer_fast(s));
 s.expert_profile->kind=Kind::GroupedW4A16;assert(uses_lingbot2_multiview_grouped_outer_fast(s));
 v3::vision.physical_manifest.routes[0].selector=1;assert(!uses_lingbot2_multiview_grouped_outer_fast(s));
 v3::vision.physical_manifest.routes[0].selector=2;
 v3::text.physical_manifest.routes[0].selector=4;assert(!uses_lingbot2_multiview_grouped_outer_fast(s));
 v3::text.physical_manifest.routes[0].selector=1;
 ++v3::text.physical_manifest.routes[0].site_id;assert(!uses_lingbot2_multiview_grouped_outer_fast(s));
 --v3::text.physical_manifest.routes[0].site_id;
 v3::vision.physical_manifest.routes.push_back(v3::vision.physical_manifest.routes[0]);
 assert(!uses_lingbot2_multiview_grouped_outer_fast(s));v3::vision.physical_manifest.routes.pop_back();
 v3::vision.physical_manifest.routes[0].invocation=1;assert(!uses_lingbot2_multiview_grouped_outer_fast(s));
 v3::vision.physical_manifest.routes[0].invocation=0;s.expert_profile->kind=Kind::DenseFp16;
 assert(!uses_lingbot2_multiview_grouped_outer_fast(s));
 s.expert_profile->kind=Kind::GroupedW8A16;v3::text.physical_manifest.routes.clear();
 assert(!uses_lingbot2_multiview_grouped_outer_fast(s));
 s.expert_profile.reset();assert(!uses_lingbot2_multiview_grouped_outer_fast(s));
}
'''.replace("__BODY__", body)
    path = tmp_path / "grouped_mode.cpp"
    path.write_text(cpp)
    binary = tmp_path / "grouped_mode"
    result = subprocess.run([shutil.which("g++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                             str(path), "-o", str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(binary)], check=True)
    replay = _body(source, "run_replay")
    assert "if (outer_fast)" in replay
    assert "drive_replay_occurrences(" in replay
    assert "coordinator.finish_replay()" in replay
    assert "s.retained_graph->end()" in replay
