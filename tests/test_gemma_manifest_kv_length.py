"""Execute Gemma's production manifest builder with small host-only doubles."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gemma_host_executable(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("gemma_manifest")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    source = (ROOT / "src/fused/rpu_gemma_model.cpp").read_text()
    method = source[source.index("    FmbPhysicalExecutionManifest physical_manifest_for_candidate("):]
    method = method.split("\n    FmbPhysicalManifestForwardCapability", 1)[0]
    method = method.replace(") const override {", ") const {")
    helpers = source[source.index("    RpuRmsNormSpmRoute consume_rmsnorm_route("):]
    helpers = helpers.split("    std::vector<int64_t> kvinsert_cost_weight_identity()", 1)[0]
    declarations = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    resolver = declarations[declarations.index("enum RpuRmsNormCapability"):
                            declarations.index("RpuRmsNormSpmContract rpu_snapshot_rmsnorm_spm_contract")]
    consumer_source = (ROOT / "src/core/fmb_three_stage_chunk_plan.cpp").read_text()
    consume = consumer_source[consumer_source.index("const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route("):]
    consume = consume.split("\nAttentionExecutionPolicy", 1)[0]
    consume = consume.replace("FmbPhysicalManifestConsumer::consume_route", "consume_physical_route")
    # Execute all three production launch expressions, including the resolver
    # and the real physical-consumer mismatch check, against a native sink.
    # The ordinary KV_FIRST body has three RMS sites. Optimized Pi pair
    # emitters have their own calls outside this body.
    ordinary = source[source.index("    void emit_kv_first_body("):]
    ordinary = ordinary.split("\nprivate:", 1)[0]
    launches = re.findall(r"rpu_launch_rmsnorm_spm_kernel\([\s\S]*?\);", ordinary)
    assert len(launches) == 3
    dispatch = "\n".join(
        "if (site == " + re.search(r"consume_rmsnorm_route\((GEMMA_\w+)", call).group(1)
        + ") { " + call + " return; }" for call in launches
    )
    # Unrelated route identities remain distinct while executing the real
    # builder; the test does not need the native ABI or a device allocation.
    names = sorted(set(re.findall(r"\b(?:GEMMA_|FMB_SHARED_|PI05_)[A-Z0-9_]+\b", method)))
    constants = "\n".join(f"constexpr int64_t {name} = {index + 1};" for index, name in enumerate(names))
    # This host double selects the ordinary Gemma path. Pi-only policies stay
    # OFF while the production builder and RMS route consumer execute below.
    cold_flags = sorted(set(re.findall(r"\b(pi05_\w+_)\b", method)))
    cold_helpers = sorted(set(re.findall(r"\b((?:use_)?pi05_\w+)\(", method)))
    cold_state = "\n".join(f"bool {name}=false;" for name in cold_flags)
    for name in cold_helpers:
        if name.endswith("arguments"):
            result, value = "std::vector<int64_t>", "{}"
        elif name.endswith("routes"):
            result, value = "std::vector<Entry>", "{}"
        elif name.endswith("accumulation"):
            result, value = "FmbLinearAccumulationPolicy", "FmbLinearAccumulationPolicy::ACC16"
        else:
            result, value = "int64_t", "400" if name == "pi05_pair_rows" else "0"
        cold_state += f"\ntemplate<class... T> {result} {name}(T...) const {{ return {value}; }}"
    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>
#define TORCH_INTERNAL_ASSERT(condition) assert(condition)
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("invalid plan"); } while (0)
enum class ChunkMode { KV_FIRST };
enum class AttentionExecutionPolicy { DDR_KV, SPM_KV_BY_MHA };
enum class FmbPhysicalManifestState { COMPLETE };
enum class FmbGraphLifecycle { COMPOSITE_CHILD };
enum class FmbLinearAccumulationPolicy { ACC16 };
enum class FmbLinearRouteSelector { AUTO_TILE };
enum class GemmaRopeRoute { ROPE_SPM, FULL_ROPE_TILED };
enum class GemmaAllReduceRoute { PREPARE_RING_INPUT };
enum class FmbRouteFamily { ATTENTION, LINEAR, ROPE, KV_INSERT, ALL_REDUCE, NORMALIZATION, GRAPH_SCHEDULE, COLLECTIVE };
struct ChunkInfo { int64_t idx, offset, len, kv_seq_len; };
struct Stage { std::vector<ChunkInfo> chunks; };
struct FmbThreeStageChunkPlan { ChunkMode chunk_mode = ChunkMode::KV_FIRST; Stage qkv, compute; };
struct LayoutContext { AttentionExecutionPolicy attention_policy = AttentionExecutionPolicy::DDR_KV; };
struct Entry { int64_t site; FmbRouteFamily family; int64_t selector, flags; std::vector<int64_t> arguments; int64_t invocation; };
struct FmbPhysicalExecutionManifest {
    FmbPhysicalManifestState state;
    int64_t logical_length, physical_length, execution_padding_rows;
    int64_t kv_logical_length, kv_insert_physical_rows;
    FmbGraphLifecycle graph_lifecycle;
    FmbLinearAccumulationPolicy linear_accumulation;
    std::vector<Entry> routes;
};
struct KvInsertSegmentPlan { int64_t route() const { return 2; } };
constexpr int64_t KV_INSERT_CAP_V2 = 2, KV_INSERT_CAP_V16 = 4;
enum class KvInsertRoute { ALIGNED_V16 };
template<class... T> KvInsertSegmentPlan rpu_resolve_kvinsert_segment_plan(T...) { return {}; }
using KvInsertRouteArguments = std::vector<int64_t>;
KvInsertSegmentPlan rpu_resolve_kvinsert_segment_plan_auto(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) { return {}; }
KvInsertRouteArguments rpu_kvinsert_route_arguments(const KvInsertSegmentPlan&, int64_t, int64_t, int64_t) { return {}; }
int64_t fmb_ring_all_reduce_route_selector(int64_t, int64_t, bool, int=8) { return 1; }
void append_fmb_shared_runtime_routes(FmbPhysicalExecutionManifest&, const FmbThreeStageChunkPlan&, int64_t, int64_t, int64_t, bool, int, int) {}
''' + constants + resolver + r'''
using FmbRouteManifestEntry = Entry;
namespace at { using IntArrayRef = std::vector<int64_t>; }
struct Context {
    FmbPhysicalExecutionManifest manifest_;
    const auto& manifest() const { return manifest_; }
    mutable std::vector<int> consumed_;
    const Entry& find_route(FmbRouteFamily family, int64_t site, int64_t invocation) const {
        for (const auto& entry : manifest_.routes) {
            if (entry.family == family && entry.site == site && entry.invocation == invocation) return entry;
        }
        throw std::runtime_error("missing physical route");
    }
''' + consume + r'''
};
struct Launch { int64_t rows, width; double eps; RpuRmsNormSpmRoute route; };
std::vector<Launch> launched;
void rpu_launch_rmsnorm_spm_kernel(uint32_t, uint32_t, uint32_t, int64_t rows, int64_t width, double eps, RpuRmsNormSpmRoute route, int cores) {
    assert(cores == 8);
    launched.push_back({rows, width, eps, route});
}
class GemmaGeometry {
public:
    bool subclass_spm_kv_by_mha_eligible(const FmbThreeStageChunkPlan&, const LayoutContext&, int64_t) const { return false; }
    int64_t q = 8, kv = 1, hd = 256, tp = 1, h = 2048, layers = 18;
    double eps_ = 1e-6;
    RpuRmsNormSpmContract rmsnorm_spm_contract_{
        RPU_RMSNORM_CAP_GENERIC_DIRECT_SPM};
    Context context;
    int64_t num_q_heads() const { return q; }
    int64_t num_kv_heads() const { return kv; }
    int64_t head_dim() const { return hd; }
    int64_t attn_tp() const { return tp; }
    int64_t hidden_size() const { return h; }
    int64_t num_layers() const { return layers; }
    int64_t num_cores() const { return 8; }
    int64_t mlp_tp() const { return 8; }
    std::vector<int64_t> core_profile_arguments() const { return {}; }
    const Context& ctx() const { return context; }
    uint32_t addr(int, const char*) const { return 0; }
    uint32_t layer_addr(int, int, const char*) const { return 0; }
''' + cold_state + helpers + r'''
    void emit_for_test(int64_t site, const ChunkInfo& chunk) {
        const int64_t seq_len = chunk.len;
        const int layer_idx = 0;
        const char* kv_norm = "input_norm";
        const char* mlp_input = "residual2";
''' + dispatch + r'''
        throw std::runtime_error("unknown RMS site");
    }
    void bind(FmbPhysicalExecutionManifest manifest) {
        context.manifest_ = std::move(manifest);
        context.consumed_.assign(context.manifest_.routes.size(), 0);
    }
''' + method + r'''
};
void test_kv_length() {
    GemmaGeometry model;
    LayoutContext layout;
    for (int64_t position : {0, 32}) {
        FmbThreeStageChunkPlan plan;
        // KV insertion and compute have different partitions; every compute
        // query still attends to the complete 400-row inserted prefix.
        plan.qkv.chunks = {{0,0,384,position+384}, {1,384,16,position+400}};
        plan.compute.chunks = {{0,0,256,position+256}, {1,256,144,position+400}};
        const auto manifest = model.physical_manifest_for_candidate(plan, layout, 400, 391, position);
        int attention_count = 0;
        for (const auto& route : manifest.routes) {
            if (route.family != FmbRouteFamily::ATTENTION) continue;
            if (route.arguments.size() != 8 || route.arguments[0] != 1 ||
                route.arguments[1] != plan.compute.chunks[route.invocation].len ||
                route.arguments[2] != position + 400) throw std::runtime_error("incorrect KV range");
            ++attention_count;
        }
        if (attention_count != 2 || manifest.kv_logical_length != position + 391 ||
            manifest.execution_padding_rows != 9) throw std::runtime_error("incorrect manifest lengths");
    }
}
void test_rms_routes() {
    LayoutContext layout;
    // Capabilities describe payload families; exact fixed-grid schedules are
    // append-only physical selectors.
    static_assert(RPU_RMSNORM_CAP_ALL == 7);
    static_assert(static_cast<int64_t>(RpuRmsNormSpmRoute::V16) == 2);
    static_assert(static_cast<int64_t>(RpuRmsNormSpmRoute::V32) == 3);
    static_assert(static_cast<int64_t>(RpuRmsNormSpmRoute::V16_GRID8) == 5);
    static_assert(static_cast<int64_t>(RpuRmsNormSpmRoute::V32_GRID8) == 6);
    assert(rpu_rmsnorm_capability_valid(RPU_RMSNORM_CAP_ALL));
    assert(!rpu_rmsnorm_capability_valid(0));
    assert(!rpu_rmsnorm_capability_valid(8));
    assert(rpu_resolve_rmsnorm_spm_route(
        RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V16, 32, 2048) ==
        RpuRmsNormSpmRoute::V16_GRID8);
    assert(rpu_resolve_rmsnorm_spm_route(
        RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V32, 32, 2048) ==
        RpuRmsNormSpmRoute::V32_GRID8);
    assert(rpu_resolve_rmsnorm_spm_route(
        RPU_RMSNORM_CAP_ALL, 16, 8192) ==
        RpuRmsNormSpmRoute::V16_GRID8);
    assert(rpu_resolve_rmsnorm_spm_route(
        RPU_RMSNORM_CAP_ALL, 32, 8192) ==
        RpuRmsNormSpmRoute::V32_GRID8);
    assert(rpu_resolve_rmsnorm_spm_route(
        RPU_RMSNORM_CAP_ALL, 32, 8193) ==
        RpuRmsNormSpmRoute::BASE);
    bool rejected_unrepresentable_base = false;
    try {
        (void)rpu_resolve_rmsnorm_spm_route(
            RPU_RMSNORM_CAP_ALL, 8 * 65536, 2048);
    } catch (const std::runtime_error&) {
        rejected_unrepresentable_base = true;
    }
    assert(rejected_unrepresentable_base);
    // Attention topology does not choose an RMSNorm schedule. Only the
    // operator's rows and width affect the exact route.
    for (int variant = 0; variant <= 7; ++variant) {
        GemmaGeometry model;
        model.kv = 8; model.tp = 8;
        if (variant == 1) model.h = 1024;
        if (variant == 2) model.layers = 17;
        if (variant == 3) model.q = 16;
        if (variant == 4) model.kv = 1;
        if (variant == 5) model.hd = 128;
        if (variant == 6) model.tp = 4;
        if (variant == 7) model.h = 8193;
        for (int64_t rows : {1, 15, 16, 32, 144, 320, 400, 401}) {
            FmbThreeStageChunkPlan plan;
            // QKV and compute partitions differ: route rows must come from
            // their own stage, never the total prefix or the other stage.
            plan.qkv.chunks = {{0, 0, rows, rows}, {1, rows, 16, rows + 16}};
            plan.compute.chunks = {{0, 0, 16, 16}, {1, 16, rows, rows + 16}};
            auto manifest = model.physical_manifest_for_candidate(plan, layout, rows + 16, rows + 16, 0);
            model.bind(manifest);
            int normalization_count = 0;
            for (const auto& entry : manifest.routes) {
                if (entry.family != FmbRouteFamily::NORMALIZATION) continue;
                const auto& stage = entry.site == GEMMA_INPUT_RMSNORM_SITE ? plan.qkv : plan.compute;
                const auto& chunk = stage.chunks.at(entry.invocation);
                const auto expected =
                    model.h <= 8192 && chunk.len % 32 == 0
                        ? RpuRmsNormSpmRoute::V32_GRID8
                        : model.h <= 8192 && chunk.len % 16 == 0
                            ? RpuRmsNormSpmRoute::V16_GRID8
                            : RpuRmsNormSpmRoute::BASE;
                assert(entry.selector == static_cast<int64_t>(expected));
                // Capability is a frozen payload/ABI fact. Width eligibility is
                // resolved separately, so C8193 records BASE with the same cap.
                assert(entry.arguments == std::vector<int64_t>({
                    chunk.len, model.h, RPU_RMSNORM_CAP_GENERIC_DIRECT_SPM}));
                launched.clear();
                model.emit_for_test(entry.site, chunk);
                assert(launched.size() == 1);
                assert(launched[0].route == expected && launched[0].rows == chunk.len &&
                       launched[0].width == model.h && launched[0].eps == model.eps_);
                ++normalization_count;
                // A self-consistent legacy selector is still the wrong exact
                // route for a newly resolved plan and must fail before launch.
                for (auto& stored : model.context.manifest_.routes) {
                    if (stored.family == entry.family && stored.site == entry.site &&
                        stored.invocation == entry.invocation)
                        stored.selector = static_cast<int64_t>(RpuRmsNormSpmRoute::V16);
                }
                launched.clear();
                bool rejected = false;
                try { model.emit_for_test(entry.site, chunk); }
                catch (const std::runtime_error&) { rejected = true; }
                assert(rejected && launched.empty());
                model.bind(manifest);
            }
            assert(normalization_count == 6);
        }
    }
}
int main(int argc, char** argv) {
    assert(argc == 2);
    if (std::string(argv[1]) == "kv") test_kv_length();
    else if (std::string(argv[1]) == "rms") test_rms_routes();
    else return 1;
}
'''
    path = tmp_path / "gemma_manifest.cpp"
    path.write_text(program)
    executable = tmp_path / "gemma_manifest"
    subprocess.run([compiler, "-std=c++17", str(path), "-o", str(executable)], check=True)
    return executable


def test_kv_first_manifest_uses_full_prefix_for_every_compute_chunk(gemma_host_executable):
    subprocess.run([str(gemma_host_executable), "kv"], check=True)


def test_gemma_rms_routes_use_operator_shape_and_reject_stale_plan(gemma_host_executable):
    subprocess.run([str(gemma_host_executable), "rms"], check=True)


def test_rms_vector_routes_preserve_legacy_abi_and_use_canonical_schedule():
    source = (ROOT / "src/ops/rpu_rmsnorm.cpp").read_text()
    legacy_v16 = source.split("case RpuRmsNormSpmRoute::V16:", 1)[1]
    legacy_v16 = legacy_v16.split("case RpuRmsNormSpmRoute::V16_GRID8:", 1)[0]
    canonical_v16 = source.split("case RpuRmsNormSpmRoute::V16_GRID8:", 1)[1]
    canonical_v16 = canonical_v16.split("case RpuRmsNormSpmRoute::V32:", 1)[0]
    legacy_v32 = source.split("case RpuRmsNormSpmRoute::V32:", 1)[1]
    legacy_v32 = legacy_v32.split("case RpuRmsNormSpmRoute::V32_GRID8:", 1)[0]
    canonical_v32 = source.split("case RpuRmsNormSpmRoute::V32_GRID8:", 1)[1]
    canonical_v32 = canonical_v32.split("default:", 1)[0]
    assert re.search(r"grid_x\s*=\s*static_cast<uint16_t>\(M / 16\)", legacy_v16)
    assert re.search(r"reg3\s*=\s*16\s*;", legacy_v16)
    assert "C <= 8192" in legacy_v16
    assert re.search(r"grid_x\s*=\s*static_cast<uint16_t>\(M / 32\)", legacy_v32)
    assert re.search(r"reg3\s*=\s*32\s*;", legacy_v32)
    assert "C <= 8192" in legacy_v32
    for route, kernel in (
        (canonical_v16, "RMS_NORM_SPM_V16"),
        (canonical_v32, "RMS_NORM_SPM_V32"),
    ):
        assert f"GET_KERNEL(KernelId::{kernel})" in route
        assert re.search(r"grid_x\s*=\s*8\s*;", route)
        assert re.search(r"reg3\s*=\s*static_cast<uint16_t>\(M / 8\)\s*;", route)
