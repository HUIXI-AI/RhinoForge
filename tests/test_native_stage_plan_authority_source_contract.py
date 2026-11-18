import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_native_domain_exports_complete_true_three_stage_descriptors() -> None:
    base = _read("src/core/fused_model_base.cpp")
    codec = _read("src/core/fmb_three_stage_chunk_plan.cpp")

    assert "stage_plan.input.plan.chunk_size" in base
    assert "input_chunks == nullptr ? compute_chunk_size" not in base
    assert "encode_fmb_prefill_stage_domain" in codec
    assert "candidate.stage_plan.input.chunks" in codec
    assert "candidate.stage_plan.qkv.chunks" in codec
    assert "candidate.stage_plan.compute.chunks" in codec
    assert "candidate.stage_plan.spans" in codec
    assert "fmb_three_stage_chunk_plan_fingerprint" in codec
    assert "physical_manifest_domain_for_candidate(" in base
    assert "std::move(physical_manifest)" in base


def test_qwen35_fixed_decode_reuses_one_complete_descriptor_across_positions(
) -> None:
    base = _read("src/core/fused_model_base.cpp")
    codec = _read("src/core/fmb_three_stage_chunk_plan.cpp")
    qwen = _read("src/fused/rpu_qwen3_5_model.cpp")
    adapter = _read("python/rpu_backend/adapters/qwen3_5/text.py")
    g05 = _read("python/rpu_backend/adapters/g05/runtime.py")

    assert "rebase_fmb_retained_decode_candidate(" in base
    assert "COMPLETE single-token RETAINED_CACHE descriptor" in codec
    assert "candidate.physical_manifest.logical_length == 1" in codec
    assert "validated.position_base == 0" in codec
    assert "chunk.kv_seq_len = position + chunk.offset + chunk.len" in codec
    assert "resolve_decode_stage_descriptor()" in qwen
    assert "decode_stage_descriptor=None" in adapter
    decode = adapter[adapter.index("def _plan_qwen3_5_decode("):
                     adapter.index("def run_qwen3_5_text(")]
    assert "qwen3_5_resolve_decode_stage_descriptor(state.handle)" in decode
    assert "return [1, 1, len(descriptor), *descriptor]" in decode
    assert "plan_native_component_execution(" in decode
    assert 'component_id="language_model", stage="decode"' in decode
    assert "graph_mode=GRAPH_RETAINED_CACHE" in decode
    assert (
        "state.decode_stage_descriptor = result.selected.stage_tuple.physical_descriptor"
        in decode
    )
    assert "decode_plan_result = _plan_qwen3_5_decode(text_model)" in adapter
    assert (
        "planned_stage_descriptor = decode_plan_result.selected.stage_tuple.physical_descriptor"
        in adapter
    )
    # The graph signature uses the full physical digest, while the native
    # consumer must still receive the complete descriptor selected above.
    tree = ast.parse(adapter)
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "*plan.graph_key_words()" in ast.unparse(functions["_qwen3_5_decode_signature"])
    assert "graph_cache.capture(_qwen3_5_decode_signature(st, decode_plan_result))" in adapter
    forwards = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "torch.ops.rpu.qwen3_5_forward"
    ]
    assert len(forwards) == 1
    assert ast.unparse(forwards[0].args[-1]) == "planned_stage_descriptor"
    # The prepared later-step closure reuses the same plan's signature and
    # complete descriptor; neither depends on the position.
    step = ast.unparse(functions["_qwen3_5_decode_step"])
    assert "signature = _qwen3_5_decode_signature(st, plan)" in step
    assert "descriptor = plan.selected.stage_tuple.physical_descriptor" in step
    assert "forward = torch.ops.rpu.qwen3_5_forward" in step
    step_calls = [node for node in ast.walk(functions["_qwen3_5_decode_step"])
                  if isinstance(node, ast.Call) and ast.unparse(node.func) == "forward"]
    assert len(step_calls) == 1 and ast.unparse(step_calls[0].args[-1]) == "descriptor"
    assert "prefill_plan_result.graph_mode" in adapter
    assert "if is_prefill else GRAPH_RETAINED_CACHE" in adapter
    assert "qwen3_5_set_retained_prefill_graph(handle, True)" in g05


def test_qwen35_padded_prefill_descriptor_keeps_logical_length() -> None:
    base = _read("src/core/fused_model_base.cpp")
    qwen = _read("src/fused/rpu_qwen3_5_model.cpp")
    adapter = _read("python/rpu_backend/adapters/qwen3_5/text.py")

    assert "logical_len = logical_len == 0 ? seq_len : logical_len" in base
    assert "physical_len - logical_len" in qwen
    assert "position + logical_len" in qwen
    assert "st.handle, n, real_len" in adapter
    assert '"RETAINED_CACHE"' in adapter


def test_qwen35_ordinary_native_forward_cannot_bypass_descriptor() -> None:
    qwen = _read("src/fused/rpu_qwen3_5_model.cpp")
    ordinary = qwen[qwen.index("at::Tensor Qwen3_5Model::forward("):
                    qwen.index("at::Tensor Qwen3_5Model::forward_z2(")]

    assert "planned_chunk_size == 0" in ordinary
    assert "!planned_stage_descriptor.empty()" in ordinary
    assert "requires one complete planner stage descriptor" in ordinary


def test_qwen35_vision_forwards_the_full_native_stage_descriptor() -> None:
    registrations = _read("src/core/rpu_dispatch_registrations.inc")
    native = _read("src/fused/rpu_qwen3_5_vision_model.cpp")
    adapter = _read("python/rpu_backend/adapters/qwen3_5/vision.py")

    schema = registrations[
        registrations.index('m.def("qwen3_5_vision_forward('):
        registrations.index("// Qwen2.5-VL Vision Encoder", registrations.index(
            'm.def("qwen3_5_vision_forward('
        ))
    ]
    assert "int[] planned_stage_descriptor=[]" in schema
    assert "encode_fmb_prefill_stage_candidate(candidate)" in native
    assert "prepare_stage_candidate(planned_stage_descriptor)" in native
    assert "/*planned_chunk_size=*/0, planned_stage_descriptor" in native
    assert "Qwen3.5 Vision forward requires " in native
    assert "one complete planner stage descriptor" in native
    tree = ast.parse(adapter)
    op_args = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "op_args"
                for target in node.targets)
    ]
    assert len(op_args) == 1 and isinstance(op_args[0], ast.Tuple)
    assert ast.unparse(op_args[0].elts[-1]) == (
        "vision_plan.selected.stage_tuple.physical_descriptor"
    )
    forwards = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "torch.ops.rpu.qwen3_5_vision_forward"
    ]
    assert forwards
    assert all(len(call.args) == 1 and isinstance(call.args[0], ast.Starred)
               and ast.unparse(call.args[0].value) == "op_args"
               for call in forwards)


def test_forward_consumes_the_decoded_immutable_candidate() -> None:
    base = _read("src/core/fused_model_base.cpp")

    assert "prepare_stage_candidate(planned_stage_descriptor)" in base
    assert "planned_stage_candidate->physical_manifest.state" in base
    assert "FmbPhysicalManifestState::UNSPECIFIED" in base
    assert "Validate only the exact immutable winner" in base
    assert "pimpl_->ctx_.stage_plan = planned_stage_candidate->stage_plan" in base
    assert "candidate.stage_plan.chunk_mode == dyn_cfg.chunk_mode" in base
    assert "candidate.stage_plan.qkv.chunks, kv_insert_chunks" in base


def test_complete_attention_routes_are_direct_authority_and_replay_receipts_are_guarded(
) -> None:
    base = _read("src/core/fused_model_base.cpp")
    codec = _read("src/core/fmb_three_stage_chunk_plan.cpp")
    header = _read("src/core/fused_model_base.h")

    assert "AttentionExecutionPolicy fmb_attention_execution_policy(" in codec
    assert "Site-specific attention-kernel variants may use flags/arguments" in header
    assert "route.selector == resolved_selector && route.flags == resolved_flags" in codec
    assert "arguments_match" in codec
    assert "AttentionExecutionPolicy::DDR_KV" in codec
    assert "AttentionExecutionPolicy::SPM_KV_BY_MHA" in codec
    assert "consume_attention_route(" in codec

    direct = base.index(
        "resolved_attention_policy = *manifest_attention_layout_policy"
    )
    legacy = base.index(
        "detail::resolve_attention_execution_policy(", direct
    )
    assert "detail::resolve_attention_execution_policy(" not in base[direct:legacy]

    branch = base.index("physical_graph.record_physical_manifest_branch(")
    fast_skip = base.index("if (fast_skip) {", branch)
    inherit = base.index(
        "inherit_physical_routes_from_matching_graph_replay()", fast_skip
    )
    assert branch < fast_skip < inherit
    assert "matching_physical_manifest_replay_branch" in base[branch:inherit]


def test_production_owner_domains_use_canonical_encoder_and_descriptor_forward() -> None:
    registrations = _read("src/core/rpu_dispatch_registrations.inc")
    owner_domains = {
        "src/fused/rpu_qwen3_model.h": (
            "causal_decoder_resolve_prefill_stage_domain",
        ),
        "src/fused/rpu_qwen3_5_model.cpp": (
            "qwen3_5_resolve_prefill_stage_domain",
            "qwen3_5_resolve_action_stage_domain",
        ),
        "src/fused/rpu_gemma_model.cpp": (
            "gemma_resolve_prefill_stage_domain",
        ),
    }

    for family in ("causal_decoder", "qwen3_5", "gemma"):
        assert f"{family}_resolve_prefill_stage_domain" in registrations
        schema = next(
            line
            for line in registrations.splitlines()
            if f'm.def("{family}_forward(' in line
        )
        assert "int[] planned_stage_descriptor=[]" in schema
    assert "stage_descriptor_domain" not in registrations
    for path, domains in owner_domains.items():
        source = _read(path)
        assert "result.reserve(domain.size() * 3)" not in source
        assert source.count("encode_fmb_prefill_stage_domain(") == len(domains)
        for domain in domains:
            assert domain in registrations


def test_canonical_exact_rejects_overflow_and_only_legacy_override_clamps() -> None:
    base = _read("src/core/fused_model_base.cpp")

    assert "requested_chunk_size <= rounded_execution_len" in _read(
        "src/core/fmb_three_stage_chunk_plan.cpp"
    )
    assert "std::min(s.chunk_size_override_, hi)" in base
    assert "std::min(planned_chunk_size" not in base
    assert "compute_chunk_size != requested_chunk_size" in base
    assert "qkv_chunk_size != requested_chunk_size" in base
    assert "RPU_PLANNER_REJECT:EXACT_MISMATCH:" in base
    assert "RPU_PLANNER_REJECT:NO_FEASIBLE:" in base
    assert "RPU_PLANNER_REJECT:CAPABILITY:" in base


def test_shape_overflow_is_rejected_before_planner_mask_allocation() -> None:
    gemma = _read("src/fused/rpu_gemma_model.cpp")
    gemma_start = gemma.index("int64_t resolve_prefill_chunk_size(")
    gemma_end = gemma.index("std::vector<int64_t> resolve_prefill_stage_domain(")
    gemma_scalar = gemma[gemma_start:gemma_end]
    gemma_domain = gemma[gemma_end:gemma.index("at::Tensor forward(", gemma_end)]
    causal = _read("src/fused/rpu_qwen3_model.h")
    causal_start = causal.index("std::vector<int64_t> resolve_prefill_stage_domain(")
    causal_end = causal.index("// ── Model state", causal_start)
    causal_domain = causal[causal_start:causal_end]

    for planner in (gemma_scalar, gemma_domain, causal_domain):
        assert planner.index("validate_fmb_planning_shape(") < planner.index(
            "at::empty("
        )


def test_extracted_complete_manifest_immediate_and_graph_authority(tmp_path):
    """Real native binding/consumer and extracted FMB control flow; no SDK."""
    import subprocess
    import torch

    base, header = _read("src/core/fused_model_base.cpp"), _read("src/core/fused_model_base.h")
    start = header.index("    bool has_complete_physical_manifest() const {")
    context = header[start:header.index("\n};", start)].replace("private:\n    friend class FusedModelBase;", "")
    start = base.index("    const FmbPhysicalExecutionManifest unspecified_physical_manifest{};")
    entry = base[start:base.index("    // Deep fast-replay", start)]
    # The receipt is emitted after dynamic_config(), which may read an RPU mask
    # through a host view. Include that real second part of the entry flow;
    # stopping at Deep fast-replay would only exercise the deferred-branch flag.
    start = base.index("    auto dyn_cfg = dynamic_config(plan);", start)
    manifest_branch = base[start:base.index("    validate_fmb_chunk_mode(", start)]
    start = base.index("void FusedModelBase::begin_external_physical_manifest_prologue(")
    external = base[start:base.index("\nvoid FusedModelBase::cancel_external_physical_manifest_prologue", start)].replace("FusedModelBase::", "Probe::")
    start = base.index("    const bool inside_graph = graph_dma::active();", base.index("void FusedModelBase::run_preload_callbacks_("))
    preload = base[start:base.index("    // graph-naive:", start)]
    harness = r'''
#include "core/fused_model_base.h"
#include <cassert>
using namespace v3;
class RpuKernelGraph {
public:
    enum class State {PASSTHROUGH, RECORDING, REPLAYING};
    struct Stamp {
        const RpuKernelGraph* graph;
        State state;
        uint64_t build_generation, signature_identity, signature_segment_key;
        size_t position;
    };
    static inline bool active_scope=false, reject_branch=false;
    static inline State current=State::PASSTHROUGH;
    static inline uint64_t built=0; static inline int branches=0;
    static bool has_active(){return active_scope;}
    static RpuKernelGraph& active(){TORCH_CHECK(active_scope); static RpuKernelGraph g; return g;}
    State state(){return current;}
    Stamp op_stream_stamp(){return {this,current,7,8,9,static_cast<size_t>(branches)};}
    void record_physical_manifest_branch(uint64_t fingerprint){
        TORCH_CHECK(fingerprint!=0 && !reject_branch); ++branches;
        if(current==State::RECORDING) built=fingerprint;
        else TORCH_CHECK(current==State::REPLAYING && built==fingerprint);
    }
};
using GraphOpStreamStamp=RpuKernelGraph::Stamp;
struct ProbeBuildTrace {
    const RpuKernelGraph* graph_identity=nullptr;
    uint64_t graph_build_generation=7, graph_signature_identity=8,
        graph_signature_segment_key=9, manifest_prefix_fingerprint=0;
    size_t manifest_prefix_begin=0, manifest_prefix_end=0;
};
namespace graph_dma {bool active(){return RpuKernelGraph::active_scope;}}
struct {int persistent_generation(){return 7;}} SPM_ALLOC;
struct ProbeContext {
'''
    probe = r'''
};
struct Probe {
    struct {ProbeContext ctx_; int cached_preload_gen_=7; bool preload_callbacks_dirty_=false;
        std::vector<FmbRouteManifestEntry> resident_preload_routes_;
        ProbeBuildTrace pipeline_build_trace_;} state;
    decltype(state)* pimpl_=&state; int preloads=0, branches_on_entry=0;
    bool resident_preloads_enabled()const{return false;}
    bool build_trace_run=false;
    std::shared_ptr<const FmbPreparedStageCandidate> prepared_candidate;
    int dynamic_config(const ChunkPlan&){
        // A new receipt must not make the host-view configuration phase flush
        // a metadata-only graph prefix. An external prologue is already bound.
        assert(RpuKernelGraph::branches==branches_on_entry); return 0;
    }
    FmbPhysicalManifestForwardCapability physical_manifest_forward_capability(const FmbPhysicalExecutionManifest&){return {true,FmbGraphLifecycle::COMPOSITE_CHILD};}
    void begin_external_physical_manifest_prologue(const FmbPhysicalExecutionManifest&,int64_t,int64_t);
    bool enter(const FmbPrefillStageCandidate* planned_stage_candidate,int64_t seq_len=16,int64_t position=0){
        branches_on_entry=RpuKernelGraph::branches;
'''
    main = r'''
int main(){
    Probe p; auto& c=p.state.ctx_;
    FmbPrefillStageCandidate candidate;
    auto& m=candidate.physical_manifest;
    m.state=FmbPhysicalManifestState::COMPLETE;
    m.logical_length=m.physical_length=m.kv_logical_length=m.kv_insert_physical_rows=16;
    m.graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
    m.linear_accumulation=FmbLinearAccumulationPolicy::ACC16;
    m.routes={{11,FmbRouteFamily::LINEAR,1,0,{},0},{22,FmbRouteFamily::MUTABLE_DMA,1,0,{},0}};
    auto fails=[](auto fn){bool failed=false;try{fn();}catch(const c10::Error&){failed=true;} assert(failed);};
    auto consume=[&](int64_t site){c.consume_physical_route(site==11?FmbRouteFamily::LINEAR:FmbRouteFamily::MUTABLE_DMA,site,1,0);};
    // Real PASSTHROUGH has no branch/proof, yet requires every exact route.
    assert(!p.enter(&candidate)); assert(RpuKernelGraph::branches==0);
    fails([&]{c.require_all_physical_routes_consumed();});
    consume(11); fails([&]{c.require_all_physical_routes_consumed();});
    fails([&]{c.consume_physical_route(FmbRouteFamily::MUTABLE_DMA,22,2,0);});
    consume(22); c.require_all_physical_routes_consumed();
    assert(!p.enter(&candidate)); fails([&]{c.require_all_physical_routes_consumed();});
    // A clean COMPLETE immediate call cannot skip listed Persistent preloads.
    p.preload(); assert(p.preloads==1);
    assert(!p.enter(nullptr)); p.preload(); assert(p.preloads==1);
    p.state.preload_callbacks_dirty_=true; p.preload(); assert(p.preloads==2);
    p.state.preload_callbacks_dirty_=false;
    fails([&]{c.find_physical_route(FmbRouteFamily::LINEAR,11);});
    fails([&]{p.enter(&candidate,15);});
    assert(!p.enter(&candidate)); consume(11); consume(22); c.require_all_physical_routes_consumed();
    // An external prelude consumes its real route before the body binds it.
    p.begin_external_physical_manifest_prologue(m,16,0); consume(22);
    assert(!p.enter(&candidate)); consume(11); c.require_all_physical_routes_consumed();
    assert(RpuKernelGraph::branches==0);
    p.begin_external_physical_manifest_prologue(m,16,0);
    auto changed=candidate; changed.physical_manifest.routes[0].selector=2;
    fails([&]{p.enter(&changed);}); c.cancel_external_physical_manifest_prologue();
    assert(!p.enter(&candidate)); fails([&]{c.require_all_physical_routes_consumed();});
    // BUILD/REPLAY still bind the exact fingerprint. No Graph proof survives
    // an immediate invocation, a changed descriptor or a failed prologue.
    RpuKernelGraph::active_scope=true; RpuKernelGraph::current=RpuKernelGraph::State::RECORDING;
    p.build_trace_run=true;
    auto& trace=p.state.pipeline_build_trace_;
    trace.graph_identity=&RpuKernelGraph::active();
    assert(!p.enter(&candidate)); assert(RpuKernelGraph::branches==1);
    assert(trace.manifest_prefix_fingerprint==fmb_physical_manifest_fingerprint(m));
    assert(trace.manifest_prefix_begin==0 && trace.manifest_prefix_end==1);
    consume(11); consume(22); c.require_all_physical_routes_consumed();
    p.build_trace_run=false;
    RpuKernelGraph::current=RpuKernelGraph::State::REPLAYING;
    assert(p.enter(&candidate)); c.inherit_physical_routes_from_matching_graph_replay(); c.require_all_physical_routes_consumed();
    fails([&]{p.enter(&changed);});
    RpuKernelGraph::reject_branch=true;
    fails([&]{p.begin_external_physical_manifest_prologue(m,16,0);});
    assert(!c.has_complete_physical_manifest() && !c.external_physical_manifest_prologue_pending_);
    RpuKernelGraph::reject_branch=false;
    RpuKernelGraph::current=RpuKernelGraph::State::RECORDING;
    p.begin_external_physical_manifest_prologue(m,16,0); consume(22);
    int branches=RpuKernelGraph::branches;
    assert(!p.enter(&candidate) && RpuKernelGraph::branches==branches); consume(11); c.require_all_physical_routes_consumed();
    RpuKernelGraph::current=RpuKernelGraph::State::REPLAYING;
    p.begin_external_physical_manifest_prologue(m,16,0); consume(22);
    assert(p.enter(&candidate)); consume(11); c.require_all_physical_routes_consumed();
    RpuKernelGraph::active_scope=false; RpuKernelGraph::current=RpuKernelGraph::State::PASSTHROUGH;
    assert(!p.enter(&candidate)); fails([&]{c.require_all_physical_routes_consumed();});
    consume(11); consume(22); c.require_all_physical_routes_consumed();
}
'''
    cpp, binary = tmp_path / "complete_immediate.cpp", tmp_path / "complete_immediate"
    cpp.write_text(harness + context + probe + entry + "ChunkPlan plan;\n" + manifest_branch + "return matching_physical_manifest_replay_branch;}\nvoid preload(){" + preload + "++preloads;}\n};\n" + external + main)
    torch_root = Path(torch.__file__).resolve().parent
    compiled = subprocess.run(["g++", "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I" + str(ROOT / "src"), "-I" + str(torch_root / "include"), "-I" + str(torch_root / "include/torch/csrc/api/include"),
        str(cpp), str(ROOT / "src/core/fmb_three_stage_chunk_plan.cpp"), "-L" + str(torch_root / "lib"),
        "-Wl,-rpath," + str(torch_root / "lib"), "-ltorch_cpu", "-lc10", "-o", str(binary)], capture_output=True, text=True, timeout=120)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr


def test_wall_cached_descriptor_restores_its_own_branch_after_other_queries(tmp_path):
    """Run the actual branch admission method with decoded host descriptors."""
    import subprocess
    source = _read('src/fused/rpu_wall_oss_action_step_model.cpp')
    start = source.index('void WallOssActionStepModel::validate_planned_action_route_profile(')
    end = source.index('\nvoid WallOssActionStepModel::consume_action_route(', start)
    method = source[start:end].replace('WallOssActionStepModel::', 'Probe::')
    declarations = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <memory>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("reject"); } while (false)
namespace at { using IntArrayRef = const std::vector<int64_t>&; }
enum class FmbRouteFamily { LINEAR };
enum class FmbPhysicalManifestState { COMPLETE, EMPTY };
enum class FmbGraphLifecycle { COMPOSITE_CHILD, RETAINED_CACHE };
struct FmbRouteManifestEntry { FmbRouteFamily family; int64_t site_id, selector, flags, invocation; std::vector<int64_t> arguments; };
struct Manifest { FmbPhysicalManifestState state=FmbPhysicalManifestState::COMPLETE; FmbGraphLifecycle graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD; std::vector<FmbRouteManifestEntry> routes; };
struct Candidate { Manifest physical_manifest; };
constexpr int64_t WALL_ACTION_PRE_W3_LINEAR_SITE=217794874262964361LL;
int64_t wall_action_linear_route(){ return 1; }
Candidate selected;
struct Prepared { Candidate value; const Candidate& candidate() const {return value;} };
std::shared_ptr<Prepared> prepare_stage_candidate(at::IntArrayRef) {return std::make_shared<Prepared>(Prepared{selected});}
struct Probe {
 bool planned_action_route_profile_valid_=false, planned_loop_mode_=false;
 int64_t planned_num_steps_=0;
 bool planned_b_step_is_bias_=false, planned_have_te_=false;
 int64_t planned_rtc_delay_=0;
 bool planned_velocity_blend_=false;
 void validate_planned_action_route_profile(at::IntArrayRef,bool,int64_t,bool,bool,int64_t,bool);
};
'''
    checks = r'''
int main() {
 Probe owner;
 const std::vector<int64_t> descriptor{1};
 auto use = [&](bool loop, int64_t steps, bool bias, bool te, int64_t rtc, bool velocity) {
   owner.validate_planned_action_route_profile(descriptor,loop,steps,bias,te,rtc,velocity);
 };
 Candidate a, b;
 a.physical_manifest.routes={{FmbRouteFamily::LINEAR,WALL_ACTION_PRE_W3_LINEAR_SITE,1,0,0,{0,1,0,0,1,1}}};
 b.physical_manifest.routes={{FmbRouteFamily::LINEAR,WALL_ACTION_PRE_W3_LINEAR_SITE,1,0,0,{3,0,1,1,0,10}}};
 selected=a; use(false,1,true,true,0,false);
 selected=b; use(true,10,false,false,3,true);
 assert(owner.planned_loop_mode_ && owner.planned_num_steps_==10);
 selected=a; use(false,1,true,true,0,false);
 assert(!owner.planned_loop_mode_ && owner.planned_num_steps_==1 && owner.planned_have_te_);
 auto reject = [&](auto change) {
   selected=b; change(); bool failed=false;
   try { use(true,10,false,false,3,true); } catch (const std::runtime_error&) { failed=true; }
   assert(failed && !owner.planned_loop_mode_ && owner.planned_num_steps_==1);
 };
 reject([&]{selected.physical_manifest.routes[0].arguments[5]=9;});
 reject([&]{selected.physical_manifest.routes[0].arguments[4]=1;});
 reject([&]{selected.physical_manifest.routes[0].arguments[0]=2;});
 reject([&]{selected.physical_manifest.routes[0].arguments.resize(4);});
 reject([&]{selected.physical_manifest.state=FmbPhysicalManifestState::EMPTY;});
 reject([&]{selected.physical_manifest.routes.clear();});
}
'''
    cpp, binary = tmp_path / 'wall_profile.cpp', tmp_path / 'wall_profile'
    cpp.write_text(declarations + method + checks)
    compiled = subprocess.run(['g++', '-std=c++17', str(cpp), '-o', str(binary)], capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert executed.returncode == 0, executed.stderr
