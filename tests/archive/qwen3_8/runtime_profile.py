"""Run actual native profile selectors with CPU doubles; no model evidence."""
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _function(source, signature):
    start = source.index(signature)
    return source[start:source.index("\n}", start) + 2]


def test_27b_native_budget_and_decode_routes(tmp_path):
    graph = (ROOT / "src/graph/graph_runtime_execute.cpp").read_text()
    runtime = (ROOT / "src/graph/graph_runtime.cpp").read_text()
    infra = (ROOT / "src/graph/graph_infra.h").read_text()
    model = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    startup = (ROOT / "src/core/rpu_backend.cpp").read_text()
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
struct SegmentResourceBudget { size_t entries, kd_bytes, instr_bytes; };
__RESOURCE_USE__
__COMPLETION__
__EXCEEDS__
__DEFER_MODE__
__SNAPSHOT_MODE__
__POLICY__
__LINGBOT__
__BUDGET__
bool accessible=false;
int claims=0, allocations=0;
uint32_t expected_slots=3;
bool rpu_device_nodes_accessible() { return accessible; }
struct RpuLknBatchConfigSnapshot { int64_t max_entries=65536, kd_buf_mb=8, instr_buf_mb=64; };
RpuLknBatchConfigSnapshot rpu_lkn_batch_config_at_load() { return {}; }
__CURRENT_CONFIG__
__VALIDATE__
struct RpuExecutionCleanupGuard {
    explicit RpuExecutionCleanupGuard(const char*) { ++claims; }
    ~RpuExecutionCleanupGuard() { --claims; }
};
namespace rhino_lkn {
struct GraphArenaRequirement { uint64_t command_bytes, instruction_bytes; uint32_t count, reserved; };
enum GraphArenaPoolResult { GRAPH_ARENA_POOL_SUCCESS=0, INVALID_ARGUMENT=1, FROZEN=2, ALLOCATION_FAILED=3 };
GraphArenaPoolResult outcome=GRAPH_ARENA_POOL_SUCCESS;
GraphArenaPoolResult ConfigureGraphArenaPool(const GraphArenaRequirement* requirement, size_t count) {
    assert(claims==1 && count==1 && requirement->count==expected_slots && requirement->reserved==0);
    assert(requirement->command_bytes==(8u<<20) && requirement->instruction_bytes==(64u<<20));
    ++allocations;
    return outcome;
}
struct KernelLaunchFootprint { uint64_t command_reservation_bytes=64, code_reservation_bytes=4096; };
constexpr int kKernelQueryOk=0;
struct Kernel_t {
    int query_launch_footprint(KernelLaunchFootprint*) { return kKernelQueryOk; }
};
}
enum class GraphNodeKind { Kernel, Dma, Barrier };
struct GraphNode {
    GraphNodeKind kind=GraphNodeKind::Dma;
    struct Kernel { size_t kernel_idx=0; } kernel;
    struct Dma { int channel=0; } dma;
    const Kernel& as_kernel() const { return kernel; }
    const Dma& as_dma() const { return dma; }
};
__NODE_RESOURCES__
__PREPARE__
template<class F> void reject(F body) {
    bool rejected=false;
    try { body(); } catch (const std::runtime_error&) { rejected=true; }
    assert(rejected && claims==0);
}
namespace at {
enum ScalarType { kHalf, kChar, kByte, kFloat };
struct Tensor { ScalarType dtype; ScalarType scalar_type() const { return dtype; } };
}
enum class FmbLinearRouteSelector { AUTO_TILE, GEMV };
struct Qwen3_5Model {
    bool linear_acc32_=false;
    int64_t layers=64, hidden=5120, intermediate=17408, heads=24, dimension=256;
    int64_t num_layers() const { return layers; }
    int64_t hidden_size() const { return hidden; }
    int64_t intermediate_size() const { return intermediate; }
    int64_t num_q_heads() const { return heads; }
    int64_t head_dim() const { return dimension; }
    FmbLinearRouteSelector linear_route_selector(const at::Tensor&, int64_t) const;
};
__ROUTE__
int main() {
    for (const char* name : {"LKN_MAX_BATCH_ENTRIES", "LKN_KD_BUF_MB", "LKN_INSTR_BUF_MB"}) unsetenv(name);
    GraphRuntimePolicy policy;
    assert(policy.graph_arena_count==0);
    validate_graph_runtime_policy(policy);
    reject([&] { prepare_graph_runtime_arenas(policy); });
    policy.qwen35_legacy_27b_sdk_budget=true;
    assert(!prepare_graph_runtime_arenas(policy) && allocations==0);
    accessible=true;
    assert(prepare_graph_runtime_arenas(policy) && allocations==1 && claims==0);
    assert(prepare_graph_runtime_arenas(policy) && allocations==2 && claims==0);
    for (auto outcome : {rhino_lkn::INVALID_ARGUMENT, rhino_lkn::FROZEN, rhino_lkn::ALLOCATION_FAILED}) {
        rhino_lkn::outcome=outcome;
        reject([&] { prepare_graph_runtime_arenas(policy); });
    }
    const int before=allocations;
    policy.lkn_kd_buf_mb=16;
    reject([&] { prepare_graph_runtime_arenas(policy); });
    assert(allocations==before);
    policy.lkn_kd_buf_mb=8;
    // SDK parses current values at Queue construction. A cold frozen policy
    // cannot approve weight upload after those effective values have changed.
    for (const char* name : {"LKN_MAX_BATCH_ENTRIES", "LKN_KD_BUF_MB", "LKN_INSTR_BUF_MB"}) {
        setenv(name, "1", 1);
        reject([&] { prepare_graph_runtime_arenas(policy); });
        assert(allocations==before);
        unsetenv(name);
    }
    rhino_lkn::outcome=rhino_lkn::GRAPH_ARENA_POOL_SUCCESS;
    // Prefixes, fallback tokens and caps deliberately match the pinned SDK.
    setenv("LKN_MAX_BATCH_ENTRIES", "65536suffix", 1);
    setenv("LKN_KD_BUF_MB", "garbage", 1);
    setenv("LKN_INSTR_BUF_MB", "0", 1);
    assert(prepare_graph_runtime_arenas(policy));
    setenv("LKN_MAX_BATCH_ENTRIES", "-1", 1);
    setenv("LKN_KD_BUF_MB", "999999", 1);
    setenv("LKN_INSTR_BUF_MB", "-1", 1);
    const auto capped=rpu_lkn_batch_config_at_preflight();
    assert(capped.max_entries==(1<<22) && capped.kd_buf_mb==256 && capped.instr_buf_mb==1024);
    reject([&] { prepare_graph_runtime_arenas(policy); });
    if (sizeof(size_t)==8) {
        setenv("LKN_KD_BUF_MB", "17592186044416", 1);
        assert(rpu_lkn_batch_config_at_preflight().kd_buf_mb==0);
        reject([&] { prepare_graph_runtime_arenas(policy); });
    }
    for (const char* name : {"LKN_MAX_BATCH_ENTRIES", "LKN_KD_BUF_MB", "LKN_INSTR_BUF_MB"}) unsetenv(name);
    policy.qwen35_legacy_27b_sdk_budget=false;
    // The explicit pool is independent of legacy segment sizing. Its real
    // validator and prepare function must fail before an SDK call on bad
    // counts, ambiguity or a changed load/preflight snapshot.
    expected_slots=4;
    policy.graph_arena_count=4;
    for (int cores : {4,6,8}) {
        policy.execution_core_count=cores;
        const auto ordinary_budget=segment_resource_budget("rpu_qwen3_5",policy);
        assert(ordinary_budget.entries==(cores==8 ? 8192u : 4096u));
        assert(ordinary_budget.kd_bytes==(4u<<20) && ordinary_budget.instr_bytes==(32u<<20));
        const int prior=allocations;
        accessible=false;
        assert(!prepare_graph_runtime_arenas(policy) && allocations==prior && claims==0);
        accessible=true;
        assert(prepare_graph_runtime_arenas(policy) && allocations==prior+1 && claims==0);
    }
    // The official owner retains its full SDK budget and four cold arenas
    // independently of main's per-instance batch snapshot selection.
    policy.qwen35_dense_27b_sdk_budget=true;
    for (auto mode : {GraphBatchSnapshotMode::Auto, GraphBatchSnapshotMode::Off}) {
        policy.batch_snapshot_mode=mode;
        for (int cores : {4,6,8}) {
            policy.execution_core_count=cores;
            const auto official_budget=segment_resource_budget("rpu_qwen3_5",policy);
            assert(official_budget.entries==65536);
            assert(official_budget.kd_bytes==(8u<<20) && official_budget.instr_bytes==(64u<<20));
            const int prior=allocations;
            assert(prepare_graph_runtime_arenas(policy) && allocations==prior+1 && claims==0);
            policy.graph_arena_count=3;
            reject([&] { prepare_graph_runtime_arenas(policy); });
            assert(allocations==prior+1);
            policy.graph_arena_count=4;
        }
    }
    policy.qwen35_dense_27b_sdk_budget=false;
    policy.batch_snapshot_mode=GraphBatchSnapshotMode::Auto;
    // Both inclusive positive boundaries are passed unchanged to the SDK;
    // this test double reserves no memory and never accesses a device.
    for (size_t count : {size_t{1},size_t{4096}}) {
        expected_slots=static_cast<uint32_t>(count);
        policy.graph_arena_count=count;
        assert(prepare_graph_runtime_arenas(policy) && claims==0);
    }
    const int before_bad_count=allocations;
    for (size_t count : {size_t{4097},std::numeric_limits<size_t>::max()}) {
        policy.graph_arena_count=count;
        reject([&] { validate_graph_runtime_policy(policy); });
        reject([&] { prepare_graph_runtime_arenas(policy); });
        assert(allocations==before_bad_count);
    }
    policy.graph_arena_count=4;
    expected_slots=4;
    policy.qwen35_legacy_27b_sdk_budget=true;
    reject([&] { validate_graph_runtime_policy(policy); });
    reject([&] { prepare_graph_runtime_arenas(policy); });
    assert(allocations==before_bad_count);
    policy.qwen35_legacy_27b_sdk_budget=false;
    policy.lkn_instr_buf_mb=32;
    reject([&] { prepare_graph_runtime_arenas(policy); });
    assert(allocations==before_bad_count);
    policy.lkn_instr_buf_mb=64;
    for (const char* name : {"LKN_MAX_BATCH_ENTRIES", "LKN_KD_BUF_MB", "LKN_INSTR_BUF_MB"}) {
        setenv(name,"1",1);
        reject([&] { prepare_graph_runtime_arenas(policy); });
        assert(allocations==before_bad_count);
        unsetenv(name);
    }
    for (auto outcome : {rhino_lkn::INVALID_ARGUMENT,rhino_lkn::FROZEN,rhino_lkn::ALLOCATION_FAILED}) {
        rhino_lkn::outcome=outcome;
        reject([&] { prepare_graph_runtime_arenas(policy); });
    }
    rhino_lkn::outcome=rhino_lkn::GRAPH_ARENA_POOL_SUCCESS;
    assert(prepare_graph_runtime_arenas(policy));
    policy.graph_arena_count=0;
    const int before_default=allocations;
    reject([&] { prepare_graph_runtime_arenas(policy); });
    assert(allocations==before_default);
    policy.execution_core_count=8;
    auto budget=segment_resource_budget("rpu_qwen3_5", policy);
    assert(budget.entries==8192 && budget.kd_bytes==(4u<<20) && budget.instr_bytes==(32u<<20));
    // Reduced-core text must split dense batches without changing the
    // process SDK limits. Exercise the real budget and cut predicate together.
    for (int cores : {4,6,8}) {
        policy.execution_core_count=cores;
        const size_t ceiling=cores==8 ? 8192 : 4096;
        for (size_t sdk_entries : {size_t{6000},size_t{65536},size_t{131072}}) {
            policy.lkn_max_batch_entries=sdk_entries;
            const auto text_budget=segment_resource_budget("qwen3_causal_decoder",policy);
            assert(text_budget.entries==std::min(ceiling,sdk_entries/2));
            assert(text_budget.kd_bytes==(4u<<20) && text_budget.instr_bytes==(32u<<20));
            SegmentResourceUse full;
            full.entries=text_budget.entries-1;
            SegmentResourceUse one;
            one.entries=1;
            one.kd_bytes=0;
            assert(!segment_would_exceed(full,one,text_budget));
            ++full.entries;
            assert(segment_would_exceed(full,one,text_budget));
        }
    }
    policy.execution_core_count=8;
    policy.lkn_max_batch_entries=65536;
    policy.qwen35_legacy_27b_sdk_budget=true;
    budget=segment_resource_budget("rpu_qwen3_5", policy);
    assert(budget.entries==65536 && budget.kd_bytes==(8u<<20) && budget.instr_bytes==(64u<<20));
    for (const char* label : {"qwen3_causal_decoder", "rpu_g05_qwen3_5_prefill", "qwen3_5_vision", ""}) {
        assert(segment_resource_budget(label, policy).entries==8192);
    }
    policy.lkn_max_batch_entries=16;
    policy.lkn_kd_buf_mb=1;
    policy.lkn_instr_buf_mb=2;
    budget=segment_resource_budget("rpu_qwen3_5", policy);
    assert(budget.entries==16 && budget.kd_bytes==(1u<<20) && budget.instr_bytes==(2u<<20));
    assert(segment_resource_budget("foreign", policy).entries==8);
    // A DMA on stream 2 requires its own completion packet beyond the 16-byte
    // batch tail. This boundary previously passed the planner but failed SDK.
    const std::vector<std::unique_ptr<rhino_lkn::Kernel_t>> kernels;
    GraphNode dma;
    dma.dma.channel=1;
    const auto stream2=segment_node_resources(dma, kernels);
    SegmentResourceUse empty;
    assert(!segment_would_exceed(empty,stream2,{99,184,4096}));
    assert(segment_would_exceed(empty,stream2,{99,183,4096}));
    // A node that cannot fit even after resetting the segment remains a hard
    // resource failure; no raised budget or count exemption can make it fit.
    assert(segment_would_exceed(empty,stream2,{99,176,4096}));
    SegmentResourceUse used=empty;
    used.kd_bytes+=stream2.kd_bytes+segment_new_completion_bytes(used,stream2);
    used.active_streams|=stream2.active_streams;
    used.entries+=stream2.entries;
    assert(!segment_would_exceed(used,stream2,{99,344,4096}));
    dma.dma.channel=7;
    const auto stream14=segment_node_resources(dma,kernels);
    assert(segment_would_exceed(used,stream14,{99,351,4096}));
    assert(!segment_would_exceed(used,stream14,{99,352,4096}));
    dma.dma.channel=0;
    assert(!segment_would_exceed(empty,segment_node_resources(dma,kernels),{99,176,4096}));
    dma.dma.channel=8;
    reject([&] { segment_node_resources(dma,kernels); });
    Qwen3_5Model large;
    for (auto dtype : {at::kHalf, at::kChar, at::kByte, at::kFloat}) {
        for (int64_t m : {1, 64, 128}) {
            assert(large.linear_route_selector({dtype}, m)==FmbLinearRouteSelector::AUTO_TILE);
        }
    }
    for (const auto row : {std::initializer_list<int64_t>{24,1024,3584,8},
                          {24,2048,6144,8}, {32,2560,9216,16}, {32,4096,12288,16}}) {
        Qwen3_5Model small;
        auto it=row.begin(); small.layers=*it++; small.hidden=*it++; small.intermediate=*it++; small.heads=*it;
        for (auto dtype : {at::kHalf, at::kChar}) {
            assert(small.linear_route_selector({dtype},1)==FmbLinearRouteSelector::GEMV);
            assert(small.linear_route_selector({dtype},64)==FmbLinearRouteSelector::AUTO_TILE);
        }
        assert(small.linear_route_selector({at::kByte},1)==FmbLinearRouteSelector::AUTO_TILE);
        small.linear_acc32_=true;
        assert(small.linear_route_selector({at::kHalf},1)==FmbLinearRouteSelector::AUTO_TILE);
        assert(small.linear_route_selector({at::kChar},1)==FmbLinearRouteSelector::AUTO_TILE);
    }
}
'''
    cpp = cpp.replace("__LINGBOT__", _function(
        graph, "static bool lingbot2_exact_graph_owns_sdk_budget("))
    cpp = cpp.replace("__BUDGET__", _function(
        graph, "static SegmentResourceBudget segment_resource_budget("))
    cpp = cpp.replace("__ROUTE__", _function(
        model, "FmbLinearRouteSelector Qwen3_5Model::linear_route_selector("))
    cpp = cpp.replace("__PREPARE__", _function(
        runtime, "bool prepare_graph_runtime_arenas("))
    cpp = cpp.replace("__VALIDATE__", _function(
        runtime, "void validate_graph_runtime_policy("))
    for marker, signature in (
        ("__DEFER_MODE__", "enum class GraphHostOpDeferMode :"),
        ("__SNAPSHOT_MODE__", "enum class GraphBatchSnapshotMode :"),
        ("__POLICY__", "struct GraphRuntimePolicy {"),
    ):
        start = infra.index(signature)
        cpp = cpp.replace(marker, infra[start:infra.index("\n};", start) + 3])
    cpp = cpp.replace("__CURRENT_CONFIG__", _function(
        startup, "RpuLknBatchConfigSnapshot rpu_lkn_batch_config_at_preflight("))
    cpp = cpp.replace("__RESOURCE_USE__", graph[graph.index("struct SegmentResourceUse {"):
        graph.index("\n};", graph.index("struct SegmentResourceUse {")) + 3])
    for marker, signature in (
        ("__COMPLETION__", "static size_t segment_new_completion_bytes("),
        ("__EXCEEDS__", "static bool segment_would_exceed("),
        ("__NODE_RESOURCES__", "static SegmentResourceUse segment_node_resources("),
    ):
        cpp = cpp.replace(marker, _function(graph, signature))
    source = tmp_path / "profile.cpp"
    source.write_text(cpp)
    executable = tmp_path / "profile"
    compiled = subprocess.run(
        ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(executable)],
        capture_output=True, text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    subprocess.run([str(executable)], check=True)
