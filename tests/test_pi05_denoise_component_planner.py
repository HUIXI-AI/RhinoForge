"""Board-free contracts for Pi0.5 denoise native planner authority."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _body(source, signature):
    """Extract a top-level native body without importing the old CI harness."""
    begin = source.index("{", source.index(signature))
    return source[begin:source.index("\n}", begin) + 2]


def test_pi05_denoise_complete_descriptor_controls_all_configurable_sites():
    native = (
        ROOT / "src/fused/rpu_pi05_denoise_step_model.cpp"
    ).read_text(encoding="utf-8")
    runtime = (
        ROOT / "python/rpu_backend/adapters/pi05/runtime.py"
    ).read_text(encoding="utf-8")
    registrations = (
        ROOT / "src/core/rpu_dispatch_registrations.inc"
    ).read_text(encoding="utf-8")

    assert "resolve_action_stage_domain(" in native
    assert "physical_manifest_for_candidate(" in native
    assert "FmbGraphLifecycle::COMPOSITE_CHILD" in native
    for site_id in (
        "799604933306379572",   # Q linear
        "1383321966524453364",  # K linear
        "5742068087814012512",  # V linear
        "1426665372740531219",  # Q RoPE
        "1167105004526897534",  # K RoPE
        "1046939961912038728",  # typed KV insert
        "5847544569257884056",  # DDR-required attention
        "4334879675031300667",  # ring prepare
        "907382291428438484",   # O linear
        "1690541798343266616",  # attention all-reduce
        "1463285755170168969",  # input mutable DMA
        "787521761862726649",   # cond mutable DMA
        "7775333525413465564",  # input projection
        "8195424785009525971",  # output projection
        "7012523434720195011",  # loop output DMA
        "5240453042848998434",  # step output DMA
        "4681095848313937529",  # AdaRMS GEMV linear
        "7943696681017534225",  # AdaRMS GEMV all-reduce
    ):
        assert site_id in native
    assert "consume_physical_attention_route(" in native
    assert "restore_kvinsert_plan(" in native
    assert "PI05_DENOISE_KV_INSERT_SITE, manifest.graph_lifecycle," in native
    assert "rpu_kvinsert_segment_plan_from_route_arguments(" not in native
    assert "rpu_launch_insert_kcache_spm_unified_with_plan(" in native
    assert "rpu_launch_insert_vcache_spm_unified_with_plan(" in native
    assert "rpu_launch_insert_kvcache_spm_unified(" not in native
    assert "RPU_PI05_KVINSERT_PAD16" not in native

    assert "_plan_pi05_denoise_action_execution(" in runtime
    assert "pi05_denoise_step_resolve_action_stage_domain" in runtime
    assert "action_plan.selected.stage_tuple.physical_descriptor" in runtime
    assert "*_action_plan.graph_key_words()" not in runtime
    assert "*action_plan.graph_key_words()" in runtime
    assert "pi05_denoise_step_resolve_action_stage_domain" in registrations
    assert "int[] planned_stage_descriptor=[]" in registrations


def test_pi05_pad16_knob_is_only_a_cold_legacy_planner_input():
    runtime = (
        ROOT / "python/rpu_backend/adapters/pi05/runtime.py"
    ).read_text(encoding="utf-8")
    assert "_cold_kvinsert_pad16_request(owner)" in runtime
    assert 'name = "_rpu_legacy_kvinsert_pad16_request"' in runtime
    assert '("legacy_pad16_request", int(' in runtime


def test_legacy_pi05_driver_is_fixed_and_forwards_the_child_descriptor():
    source = (
        ROOT / "src/fused/rpu_pi05_model.cpp"
    ).read_text(encoding="utf-8")

    assert "class Pi05Model {" in source
    assert "class Pi05Model : public FusedModelBase" not in source
    assert "at::IntArrayRef planned_stage_descriptor" in source
    assert "rpu_adarms_forward(" in source
    assert "planned_stage_descriptor);" in source
    assert "GraphCache" not in source
    assert "execution_reconfigure" not in source


def test_actual_native_cached_descriptor_survives_step_loop_step_and_rejects_drift(tmp_path):
    """Execute native query/admission/route-consumer bodies; no model or board."""
    import re
    import subprocess

    source = (ROOT / "src/fused/rpu_pi05_denoise_step_model.cpp").read_text()
    base = (ROOT / "src/core/fused_model_base.h").read_text()
    header = (ROOT / "src/fused/rpu_pi05_denoise_step_model.h").read_text()
    policy = header[header.index("    FmbLinearAccumulationPolicy linear_accumulation_policy() const {"):]
    policy = policy.split("    void validate_pi05_runtime_policies", 1)[0]
    query = _body(source, "std::vector<int64_t> Pi05DenoiseStepModel::resolve_action_stage_domain(")
    restore = _body(source, "void Pi05DenoiseStepModel::restore_planned_route_profile(")
    # Unrelated W4 selector constants are not used by this restoration fixture.
    constants = "\n".join(c for c in re.findall(r"constexpr int64_t PI05_DENOISE_\w+\s*=.*?;", source, re.S)
                          if "Pi05DenoiseW4GeGluLinearRouteSelector" not in c)
    enums = "\n".join("enum class " + name + " : int64_t " + _body(source, "enum class " + name) + ";"
                      for name in ("Pi05DenoiseMutableDmaRoute", "Pi05DenoiseMaskScheduleRoute"))
    manifest = _body(source, "Pi05DenoiseStepModel::physical_manifest_for_candidate(")
    # This fixture targets descriptor restoration, with optional kernels OFF.
    # Extract the actual admission/header and the four relevant DMA/body routes,
    # rather than importing unrelated fused-kernel owner implementations.
    prefix = manifest[:manifest.index("    if (pi05_w4_fastpath_opt_in_)")]
    sites = ("PI05_DENOISE_PRE_X_DMA_SITE", "PI05_DENOISE_PRE_COND_DMA_SITE",
             "PI05_DENOISE_PRE_LINEAR_SITE", "PI05_DENOISE_POST_LINEAR_SITE",
             "planned_loop_mode_ ? PI05_DENOISE_POST_LOOP_DMA_SITE")
    routes = []
    for site in sites:
        end = manifest.index(";", manifest.index(site)) + 1
        start = manifest.rfind("append", 0, manifest.index(site))
        routes.append(manifest[start:end])
    mask_schedule = manifest[manifest.index("        // This route is part of the physical descriptor:"):]
    mask_schedule = mask_schedule[:mask_schedule.index(
        "        append(\n            FmbRouteFamily::ALL_REDUCE,")]
    manifest = (prefix + "\n".join(routes)
                + "\nfor (const auto& chunk : plan.compute.chunks) {\n"
                + "const int64_t invocation = chunk.idx;\n" + mask_schedule
                + "}\nreturn manifest;\n}")
    cold_flags = sorted(set(re.findall(r"\b(pi05_\w+_opt_in_|pi05_w4_fastpath_profile_)\b", query)))
    cold_predicates = sorted(set(re.findall(r"\b(use_pi05_\w+)\(", query)))
    cold_state = "\n".join(f"bool {name}=false;" for name in cold_flags)
    cold_state += "\n" + "\n".join(
        f"template<class... T> bool {name}(T...) const {{ return false; }}"
        for name in cold_predicates)
    consumer = _body(source, "void Pi05DenoiseStepModel::emit_pre_layers_body(")
    start = consumer.index("    if (ctx().has_complete_physical_manifest())", consumer.index("// [A3]"))
    consumer = consumer[start:consumer.index("    rpu_launch_linear_spm_to_spm_acc16_kernel(", start)]
    post_consumer = _body(source, "void Pi05DenoiseStepModel::emit_post_layers_body(")
    start = post_consumer.index("    if (ctx().has_complete_physical_manifest())", post_consumer.index("// [Z2]"))
    post_consumer = post_consumer[start:post_consumer.index("    rpu_launch_linear_spm_to_spm_acc16_kernel(", start)]
    consumer += post_consumer
    capture_start = base.index("static KvCostLayoutScope capture_kvinsert_cost_layout_fields(")
    capture = base[base.index("{", capture_start):base.index("\n    }", capture_start) + 6]
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
namespace c10 {
template<class F> struct Scope { F f; ~Scope() { f(); } };
template<class F> Scope<F> make_scope_exit(F f) { return {f}; }
}
namespace at {
using IntArrayRef = std::vector<int64_t>;
struct Tensor { bool defined() const { return false; } int64_t size(int) const { return 0; } };
constexpr int kHalf=0, kCPU=0;
struct TensorOptions { TensorOptions dtype(int) { return *this; } TensorOptions device(int) { return *this; } };
Tensor empty(std::initializer_list<int64_t>, TensorOptions) { return {}; }
}
namespace detail { void validate_fmb_planning_shape(int64_t n, int64_t p, const char*) {
    TORCH_CHECK(n>0 && p>=0); } }
enum class FmbPhysicalManifestState { UNSPECIFIED, COMPLETE };
enum class FmbGraphLifecycle { UNSPECIFIED, COMPOSITE_CHILD };
enum class FmbLinearAccumulationPolicy { UNSPECIFIED, ACC16, ACC32, MIXED_BY_SITE };
enum class FmbRouteFamily { LINEAR, MUTABLE_DMA, ALL_REDUCE, GRAPH_SCHEDULE };
enum class FmbLinearRouteSelector { AUTO_TILE=1 };
__CONSTANTS__
__ENUMS__
int64_t pi05_denoise_ring_route(int64_t, int64_t, int64_t) { return 1; }
struct FmbRouteManifestEntry { int64_t site_id; FmbRouteFamily family; int64_t selector, flags;
    std::vector<int64_t> arguments; int64_t invocation; };
struct FmbPhysicalExecutionManifest {
    FmbPhysicalManifestState state=FmbPhysicalManifestState::UNSPECIFIED;
    int64_t logical_length=0, physical_length=0, execution_padding_rows=0,
            kv_logical_length=0, kv_insert_physical_rows=0;
    FmbGraphLifecycle graph_lifecycle=FmbGraphLifecycle::UNSPECIFIED;
    FmbLinearAccumulationPolicy linear_accumulation=FmbLinearAccumulationPolicy::UNSPECIFIED;
    std::vector<FmbRouteManifestEntry> routes;
};
struct FmbThreeStageChunkPlan {
    struct Chunk { int64_t idx=0, len=50; };
    struct { std::vector<Chunk> chunks{{0,50}}; } compute;
};
struct LayoutContext { bool use_attn_mask=true, is_causal=false; };
struct Candidate { FmbPhysicalExecutionManifest physical_manifest; };
std::vector<Candidate> descriptors;
Candidate decode_fmb_prefill_stage_candidate(at::IntArrayRef descriptor) { return descriptors.at(descriptor.at(0)); }
struct Prepared {Candidate value;const Candidate& candidate() const {return value;}};
std::shared_ptr<Prepared> prepare_stage_candidate(at::IntArrayRef descriptor) {return std::make_shared<Prepared>(Prepared{decode_fmb_prefill_stage_candidate(descriptor)});}
struct RpuKernelGraph { static bool active; static bool has_active() { return active; } };
bool RpuKernelGraph::active=true;
struct Model {
    const bool linear_acc32_;
    explicit Model(bool linear_acc32=false) : linear_acc32_(linear_acc32) {}
    __ACCUMULATION_POLICY__
    struct { bool value=false; bool enabled() const { return value; } } nvfp4_;
    bool planned_route_profile_valid_=false, planned_loop_mode_=false, loop_mode_=false, fail=false;
    int64_t planned_num_steps_=1, planned_kv_physical_rows_=0, chunk_size_=50, num_steps_=1;
    int queries=0, policy_checks=0;
    bool policy_fail=false;
    int64_t planned_prefix_len_=0,planned_k_rope_cache_capacity_=0;
    int64_t pi05_gateup_resident_layers_=0,rope_position_=-1;
    at::Tensor adarms_table_;
    __COLD_STATE__
    FmbPhysicalExecutionManifest current;
    using KvCostLayoutScope=std::function<void(const std::function<void()>&)>;
    template<typename... Fields> static KvCostLayoutScope capture_kvinsert_cost_layout_fields(Fields&... fields) __CAPTURE__
    KvCostLayoutScope capture_kvinsert_cost_layout_scope() {
        return capture_kvinsert_cost_layout_fields(planned_route_profile_valid_, planned_loop_mode_,
                                                 planned_num_steps_, planned_kv_physical_rows_,
                                                 planned_prefix_len_, planned_k_rope_cache_capacity_);
    }
    int64_t hidden_size() const { return 1024; }
    int64_t num_cores() const { return 8; }
    int64_t attn_tp() const { return 8; }
    int64_t num_layers() const { return 18; }
    int64_t condition_tp() const { return 8; }
    std::vector<int64_t> core_profile_arguments() const { return {}; }
    FmbPhysicalExecutionManifest physical_manifest_for_candidate(const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout, int64_t physical_len, int64_t logical_len, int64_t position) const __MANIFEST__
    std::vector<int64_t> resolve_prefill_stage_domain_for_shape(int64_t n,int64_t p,
            std::optional<at::Tensor>,bool,int64_t,int64_t logical_len) {
        ++queries;
        if (fail) throw std::runtime_error("oracle reject");
        descriptors.push_back({physical_manifest_for_candidate({}, {}, n,logical_len,p)});
        return {static_cast<int64_t>(descriptors.size()-1)};
    }
    std::vector<int64_t> encode_fmb_prefill_stage_domain(std::vector<int64_t> v) { return v; }
    std::vector<int64_t> query(int64_t execution_len,int64_t logical_len,int64_t position,
        int64_t kv_len,int64_t cache_capacity,int64_t requested_chunk_size,bool prefer_pad16,
        bool loop_mode,int64_t num_steps) __QUERY__
    void restore(at::IntArrayRef descriptor,bool loop_mode,int64_t num_steps,
                 int64_t prefix_len,int64_t cache_capacity) __RESTORE__
    Model& ctx() { return *this; }
    bool has_complete_physical_manifest() const { return true; }
    void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,
                                int64_t flags,std::vector<int64_t> args={}) {
        for (const auto& r: current.routes) if(r.family==family && r.site_id==site) {
            TORCH_CHECK(r.selector==selector && r.flags==flags && r.arguments==args); return;
        }
        throw std::runtime_error("missing actual route");
    }
    void consume() { __CONSUMER__ }
    void validate_pi05_runtime_policies(at::IntArrayRef d) {
        ++policy_checks;
        if (!d.empty()) {
            const auto& manifest=decode_fmb_prefill_stage_candidate(d).physical_manifest;
            TORCH_CHECK(planned_route_profile_valid_ &&
                planned_kv_physical_rows_==manifest.kv_insert_physical_rows &&
                planned_prefix_len_==manifest.kv_logical_length-chunk_size_);
        }
        TORCH_CHECK(!policy_fail);
    }
    auto state() const { return std::make_tuple(planned_route_profile_valid_,planned_loop_mode_,
                                              planned_num_steps_,planned_kv_physical_rows_,
                                              planned_prefix_len_,planned_k_rope_cache_capacity_); }
    void forward(at::IntArrayRef d,bool loop,int64_t steps,int64_t position=800,int64_t capacity=864) {
        restore(d,loop,steps,position,capacity); loop_mode_=loop;num_steps_=steps;
        current=decode_fmb_prefill_stage_candidate(d).physical_manifest;consume();
    }
};
int main() {
    Model m;const auto empty=m.state();
    auto a=m.query(50,50,800,850,864,0,true,false,1);
    assert(m.state()==empty);
    auto b=m.query(50,50,800,850,864,0,false,true,10);
    assert(m.state()==empty && m.queries==2);
    m.forward(a,false,1);assert(m.planned_kv_physical_rows_==64);
    m.forward(b,true,10);assert(m.planned_kv_physical_rows_==50);
    m.forward(a,false,1);assert(m.planned_kv_physical_rows_==64 && m.queries==2);
    const auto before=m.state();
    const auto valid_manifest=m.current;
    for(auto& route:m.current.routes) if(route.site_id==PI05_DENOISE_POST_LINEAR_SITE)
        route.selector=99;
    try { m.consume();assert(false); } catch(const std::runtime_error&) {}
    assert(m.state()==before);m.current=valid_manifest;
    m.fail=true;try { m.query(50,50,800,850,864,0,false,true,4);assert(false); }
    catch(const std::runtime_error&) {} assert(m.state()==before);m.fail=false;
    auto reject=[&](at::IntArrayRef d,bool loop,int64_t steps,int64_t pos,int64_t cap) {
        try { m.restore(d,loop,steps,pos,cap);assert(false); } catch(const std::runtime_error&) {}
        assert(m.state()==before);
    };
    assert(m.policy_checks==3);
    m.policy_fail=true;
    reject(b,true,10,800,864);
    assert(m.policy_checks==4);
    m.policy_fail=false;
    reject(a,true,10,800,864);reject(b,true,9,800,864);reject(b,true,30,800,864);
    reject(a,false,1,800,850);reject(a,false,1,801,865);reject({},false,1,800,864);
    for(int mutation=0;mutation<6;++mutation) {
        auto bad=descriptors[a[0]];
        if(mutation==0)bad.physical_manifest.kv_insert_physical_rows=80;
        if(mutation==1)bad.physical_manifest.routes.erase(bad.physical_manifest.routes.begin()+2);
        if(mutation==2)bad.physical_manifest.routes[2].arguments={0}; // stale ABI
        if(mutation==3)bad.physical_manifest.routes[2].selector=99;
        if(mutation==4) { bad.physical_manifest.kv_logical_length=851; }
        if(mutation==5)bad.physical_manifest.linear_accumulation=FmbLinearAccumulationPolicy::ACC32;
        descriptors.push_back(bad);reject({int64_t(descriptors.size()-1)},false,1,800,864);
    }
    // Cold and restored descriptors bind the all-body mask schedule, including
    // geometry, topology and exact body count; no partially admitted state leaks.
    for(int mutation=0;mutation<12;++mutation) {
        auto bad=descriptors[a[0]];
        auto& routes=bad.physical_manifest.routes;
        size_t index=0;
        while(index<routes.size() && routes[index].site_id!=PI05_DENOISE_MASK_SCHEDULE_SITE)++index;
        assert(index<routes.size());
        if(mutation==0)routes.erase(routes.begin()+index);
        if(mutation==1)routes[index].selector=99;
        if(mutation==2)routes[index].flags=1;
        if(mutation==3)routes[index].invocation=1;
        if(mutation==4)routes.push_back(routes[index]);
        if(mutation>=5)++routes[index].arguments[mutation-5];
        descriptors.push_back(bad);reject({int64_t(descriptors.size()-1)},false,1,800,864);
    }
    Model nv; nv.nvfp4_.value=true;
    auto nv_desc=nv.query(50,50,800,850,864,0,false,true,10);
    nv.forward(nv_desc,true,10);
    assert(nv.current.linear_accumulation==FmbLinearAccumulationPolicy::ACC16);
    // The immutable per-handle accumulation selection binds the descriptor;
    // optional NVFP4 storage does not change that selection by itself.
    nv.restore(b,true,10,800,864);
    m.restore(nv_desc,true,10,800,864);
    Model acc32(true);
    auto acc32_desc=acc32.query(50,50,800,850,864,0,false,true,10);
    acc32.forward(acc32_desc,true,10);
    assert(acc32.current.linear_accumulation==FmbLinearAccumulationPolicy::ACC32);
    const auto acc16_before=m.state();
    try { m.restore(acc32_desc,true,10,800,864); assert(false); }
    catch(const std::runtime_error&) {} assert(m.state()==acc16_before);
    const auto acc32_before=acc32.state();
    try { acc32.restore(b,true,10,800,864); assert(false); }
    catch(const std::runtime_error&) {} assert(acc32.state()==acc32_before);
    RpuKernelGraph::active=false;
    m.restore({},false,1,800,850);assert(m.planned_kv_physical_rows_==50);
}
'''
    for name, value in {"QUERY": query, "RESTORE": restore, "CAPTURE": capture,
                        "MANIFEST": manifest, "CONSUMER": consumer,
                        "CONSTANTS": constants, "ENUMS": enums, "COLD_STATE": cold_state,
                        "ACCUMULATION_POLICY": policy}.items():
        cpp = cpp.replace("__" + name + "__", value)
    path, binary = tmp_path / "pi05-descriptor.cpp", tmp_path / "pi05-descriptor"
    path.write_text(cpp)
    compiled = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    # The two original callers restore this request before any mode invalidation.
    for method in ("step_forward", "denoise_loop_forward"):
        body = _body(source, "void Pi05DenoiseStepModel::" + method + "(")
        assert body.index("restore_planned_route_profile(") < body.index("invalidate_model_state(")
