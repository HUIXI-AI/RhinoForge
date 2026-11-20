"""Execute native Graph segmentation and queue admission with a host recorder.

Production function bodies are compiled unchanged; only SDK allocation and
unrelated Graph payloads are stubbed. No device, Torch import, or planning files
are needed. Numerical BUILD/REPLAY remains a separate hardware gate.
"""

from pathlib import Path
import ast
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _function(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def native_budget_probe(tmp_path_factory):
    compiler = shutil.which("g++")
    assert compiler is not None, "native Graph contract checks require g++"
    execute = (ROOT / "src/graph/graph_runtime_execute.cpp").read_text()
    runtime = (ROOT / "src/graph/graph_runtime.cpp").read_text()
    infra = (ROOT / "src/graph/graph_infra.h").read_text()
    child = (ROOT / "src/graph/graph_runtime_child.cpp").read_text()
    internal = (ROOT / "src/graph/graph_runtime_internal.h").read_text()
    capture = (ROOT / "src/graph/graph_runtime_capture.cpp").read_text()
    # Use the public Python signature's real stable op-id conversion, without
    # importing Torch or loading a backend, for the native scope checks.
    signature_source = (ROOT / "python/rpu_backend/graph/_runtime.py").read_text()
    signature_fn = next(node for node in ast.parse(signature_source).body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "_graph_fnv1a_64")
    signature_namespace = {}
    exec(compile(ast.Module(body=[signature_fn], type_ignores=[]),
                 "graph/_runtime.py", "exec"), signature_namespace)
    causal_decoder_op_id = signature_namespace["_graph_fnv1a_64"]("rpu_causal_decoder")
    policy = infra[infra.index("struct GraphRuntimePolicy {"):]
    policy = policy[:policy.index("\n};") + 3]
    functions = "\n\n".join([
        _function(runtime, "void validate_graph_runtime_policy("),
        _function(internal, "inline void validate_graph_dma_channel("),
        _function(execute, "uint8_t validated_kernel_core_extent("),
        _function(execute, "static size_t segment_new_completion_bytes("),
        _function(execute, "static bool segment_would_exceed("),
        _function(execute, "static bool causal_14b_c4_initial_prefill_needs_small_segments("),
        _function(execute, "void RpuKernelGraph::build_segments_from_nodes("),
        _function(execute, "::rhino_lkn::Queue_t* RpuKernelGraph::ensure_private_queue("),
        _function(execute, "void RpuKernelGraph::execute_ddr_copy("),
        "template <typename ND>\n" + _function(capture, "void RpuKernelGraph::capture_data("),
    ])
    def first_guard(body):
        start = body.index("TORCH_CHECK(")
        return body[start:body.index(";", start) + 1]
    sync_guard = first_guard(_function(execute, "void RpuKernelGraph::sync_point("))
    downgrade_guard = first_guard(_function(runtime, "void RpuKernelGraph::mark_non_replayable("))
    prepare = _function(execute, "uint32_t RpuKernelGraph::prepare_segment_queue(")
    prepare_preflight = prepare[prepare.index("{") + 1:prepare.index("// P2 改造:")]
    # Exercise the actual admission statement at each child attachment entry.
    child_guards = []
    for name in ("register_child", "replace_window_with_child"):
        body = _function(child, f"void RpuKernelGraph::{name}(")
        start = body.index("    TORCH_CHECK(\n        child->runtime_policy_.execution_core_count")
        child_guards.append(body[start:body.index(";", start) + 1])
    extract = _function(child, "std::shared_ptr<RpuKernelGraph> RpuKernelGraph::extract_child_window(")
    constructor = next(line for line in extract.splitlines()
                       if "auto child = std::make_shared<RpuKernelGraph>" in line)

    replay_guards = []
    for match in re.finditer(r"TORCH_CHECK\(core_ids == kn\.core_ids,", execute):
        replay_guards.append(execute[match.start():execute.index(";", match.start()) + 1])
    assert len(replay_guards) == 2
    cache_header = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    graph_header = (ROOT / "src/graph/graph_runtime.h").read_text()
    sdk_cache_get = _function(cache_header, "inline ::rhino_lkn::Queue_t* get(")
    proxy_cache_get = _function(graph_header, "RpuQueue* get(")
    typed_add = _function(cache_header, "inline void rpu_add_dma_checked(")

    harness = r'''
#include <algorithm>
#include <array>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <variant>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
#define RECORD_FUNCTION(...) do {} while (0)
namespace at {
constexpr uint8_t kHalf = 5;
enum class RecordScope { FUNCTION };
struct RecordFunction {
    explicit RecordFunction(RecordScope) {}
    bool isActive() const { return false; }
    void before(const char*) {}
    void end() {}
};
}
enum class GraphHostOpDeferMode : uint8_t { Auto, ForceOff, ForceOn };
enum class GraphBatchSnapshotMode : uint8_t { Auto, Off };
__POLICY__
namespace rhino_lkn {
struct Buffer_t { std::shared_ptr<std::vector<uint8_t>> data; };
struct Submission { size_t cores, src_offset, dst_offset, bytes; int channel; };
struct Queue_t {
    inline static int creations = 0, arenas = 0, leases = 0, waits = 0;
    inline static std::string failure;
    inline static std::vector<Submission> submissions;
    size_t count, src_offset = 0, dst_offset = 0, bytes = 0;
    int channel = -1;
    bool pending = false, built = false, arena = false, in_flight = false;
    std::shared_ptr<std::vector<uint8_t>> src, dst;
    explicit Queue_t(size_t n) : count(n) { ++creations; }
    ~Queue_t() { if (in_flight) ++waits; release(); }
    void release() {
        if (arena) { --arenas; arena = false; }
        if (src) { --leases; src.reset(); }
        if (dst) { --leases; dst.reset(); }
        pending = built = in_flight = false;
    }
    uint32_t add_dma_kernel(const Buffer_t& source, size_t so, Buffer_t& target,
                            size_t to, size_t n, int ch) {
        assert(!pending && !built);
        if (failure == "add") return 11;
        assert(ch >= 0 && ch < 8);
        assert(so + n <= source.data->size() && to + n <= target.data->size());
        src = source.data; dst = target.data; leases += 2;
        src_offset = so; dst_offset = to; bytes = n; channel = ch; pending = true;
        return 0;
    }
    uint32_t build_batch() {
        assert(pending && !built);
        arena = true; ++arenas;
        if (failure == "build") return 12;
        pending = false; built = true;
        return 0;
    }
    uint32_t enqueu_batch(bool wait_finish) {
        assert(built && wait_finish);
        if (failure == "enqueue") return 13;
        if (failure == "throw_enqueue") {
            in_flight = true;
            throw std::runtime_error("original submission failure");
        }
        std::memcpy(dst->data() + dst_offset, src->data() + src_offset, bytes);
        submissions.push_back({count, src_offset, dst_offset, bytes, channel});
        return 0;
    }
    uint32_t discard_completed_batch() {
        assert(built && !in_flight);
        if (failure == "discard") return 14;
        release(); return 0;
    }
};
}
struct RpuDmaEndpoint {
    rhino_lkn::Buffer_t* owner = nullptr; size_t offset = 0;
    explicit operator bool() const { return owner != nullptr; }
};
struct RpuCpuDmaEndpointRequest {
    const void* cpu_ptr = nullptr; size_t bytes = 0; const char* where = nullptr;
};
struct RpuDmaSubmissionLease {
    RpuDmaSubmissionLease() = default;
    RpuDmaSubmissionLease(const RpuDmaSubmissionLease&) = delete;
    RpuDmaSubmissionLease& operator=(const RpuDmaSubmissionLease&) = delete;
    RpuDmaSubmissionLease(RpuDmaSubmissionLease&&) = default;
    RpuDmaSubmissionLease& operator=(RpuDmaSubmissionLease&&) = default;
};
std::vector<rhino_lkn::Buffer_t*> live_buffers;
RpuDmaEndpoint rpu_resolve_cpu_dma_endpoint(const void* pointer, size_t n, const char*) {
    const auto p = reinterpret_cast<uintptr_t>(pointer);
    for (auto* buffer : live_buffers) {
        const auto base = reinterpret_cast<uintptr_t>(buffer->data->data());
        if (p >= base && p - base <= buffer->data->size() &&
                n <= buffer->data->size() - (p - base)) return {buffer, p - base};
    }
    throw std::runtime_error("unowned endpoint");
}
RpuDmaSubmissionLease rpu_resolve_cpu_dma_endpoints(
        const RpuCpuDmaEndpointRequest* requests, size_t count,
        RpuDmaEndpoint* endpoints) {
    std::vector<RpuDmaEndpoint> pending;
    pending.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        pending.push_back(rpu_resolve_cpu_dma_endpoint(
            requests[i].cpu_ptr, requests[i].bytes, requests[i].where));
    }
    std::copy(pending.begin(), pending.end(), endpoints);
    return {};
}
__TYPED_ADD__
namespace rhino_lkn {
constexpr uint32_t kBatchSnapshotAuto=1;
struct BatchSnapshotOptionsV1 { uint32_t mode=0, contracts=0; };
}
struct SnapshotBridge {
    uint32_t configure(rhino_lkn::Queue_t*, const rhino_lkn::BatchSnapshotOptionsV1*) const { return 0; }
};
const SnapshotBridge& batch_snapshot_bridge() { static SnapshotBridge bridge; return bridge; }
enum class CopyKind { HOST_MEMCPY, DDR_TO_DDR };
struct MemcpyNodeData {
    void* dst = nullptr; const void* src = nullptr; size_t bytes = 0;
    int dma_channel = -1; CopyKind kind = CopyKind::DDR_TO_DDR;
    uint64_t src_dev_addr = 0, dst_dev_addr = 0;
};
struct MemsetNodeData {};
__SIGNATURE__
__BRANCH__
std::string format_hex(uint64_t) { return ""; }
struct QueueCache {
    std::array<rhino_lkn::Queue_t*, 8> queues_{};
    ~QueueCache() { for (auto* queue : queues_) delete queue; }
    __SDK_CACHE_GET__
};
struct RpuQueue { int count; explicit RpuQueue(int n): count(n) {} };
struct RpuQueueCache {
    std::array<std::unique_ptr<RpuQueue>, 8> queues_;
    RpuQueueCache() { for (int i=0; i<8; ++i) queues_[i]=std::make_unique<RpuQueue>(i+1); }
    __PROXY_CACHE_GET__
};
struct QueueLaunchState {
    bool broadcast_mode = true;
    bool flush_icache = true;
    bool operator!=(const QueueLaunchState& rhs) const {
        return broadcast_mode != rhs.broadcast_mode || flush_icache != rhs.flush_icache;
    }
};
enum class GraphNodeKind { Kernel, Dma, Barrier, ChildGraph, Memcpy, Branch };
template <typename ND> struct NodeKindOf;
template <> struct NodeKindOf<MemcpyNodeData> {
    static constexpr auto value = GraphNodeKind::Memcpy;
};
struct RpuKernelGraph;
struct GraphNode {
    GraphNodeKind kind = GraphNodeKind::Kernel;
    std::vector<uint8_t> core_ids;
    QueueLaunchState queue_state;
    uint8_t channel = 0, self_stream = 0, target_stream = 0;
    bool isolate = false;
    std::shared_ptr<RpuKernelGraph> child;
    BranchNodeData branch;
    std::variant<MemcpyNodeData> data;
    GraphNode() = default;
    GraphNode(GraphNodeKind k, MemcpyNodeData value) : kind(k), data(value) {}
    GraphNode& as_kernel() { return *this; }
    const GraphNode& as_kernel() const { return *this; }
    const GraphNode& as_dma() const { return *this; }
    const GraphNode& as_barrier_node() const { return *this; }
    const GraphNode& as_child_graph() const { return *this; }
    const BranchNodeData& as_branch() const { return branch; }
};
struct Census {
    size_t segment_id = 0, start_idx = 0, end_idx = 0, core_count = 0;
    size_t kernel_count = 0, dma_count = 0, barrier_count = 0;
};
struct Segment {
    size_t segment_id = 0, start_idx = 0, end_idx = 0;
    std::vector<std::pair<size_t, size_t>> prologue_spans;
    std::vector<uint8_t> core_ids;
    QueueLaunchState queue_state;
    Census census;
    template <typename Fn>
    void for_each_node_index(Fn&& fn) const {
        for (const auto& [b, e] : prologue_spans) {
            for (size_t i = b; i < e; ++i) fn(i);
        }
        for (size_t i = start_idx; i < end_idx; ++i) fn(i);
    }
};
struct SegmentResourceUse { size_t entries = 0, kd_bytes = 0, instr_bytes = 0; uint16_t active_streams = 0; };
using SegmentResourceBudget = SegmentResourceUse;
size_t test_entry_budget = 1000;
bool test_native_budget = false;
__NATIVE_BUDGET__
SegmentResourceBudget segment_resource_budget(const std::string& label, const GraphRuntimePolicy& policy, bool require_single_segment) {
    if (test_native_budget) return native_segment_resource_budget(label, policy, require_single_segment);
    return {test_entry_budget, 1000, 1000};
}
SegmentResourceUse segment_node_resources(const GraphNode& node, const std::vector<int>&) {
    assert(node.kind != GraphNodeKind::Branch); return {1, 1, 1};
}
bool p7_should_force_isolate_kernel(const GraphNode& node) { return node.isolate; }
bool siglip_should_isolate_patch_embed_kernel(const std::string&, const GraphNode&, bool) { return false; }
std::string kernel_label_for_segment_split(const GraphNode&) { return "test"; }
bool log_at(int) { return false; }
inline void validate_graph_dma_channel(int, size_t, const char*);
struct RpuKernelGraph {
    enum class State { PASSTHROUGH, RECORDING, BUILT, REPLAYING };
    State state_ = State::RECORDING;
    size_t cursor_ = 0;
    bool kernel_register_census_active_ = false, has_kernel_register_census_nodes_ = false;
    bool canonical_dma_build_trace_active_ = false, canonical_dma_callback_active_ = false;
    std::optional<int> pending_kernel_register_census_;
    GraphRuntimePolicy runtime_policy_;
    // Snapshot admission/authority is executed in the prepared-Queue fixture;
    // this recorder isolates main's core/DDR-copy admission contract.
    bool batch_snapshot_available_=false;
    uint32_t batch_snapshot_contracts_=0;
    std::string batch_snapshot_fallback_reason_;
    void bind_batch_snapshot_admission() {}
    void invalidate_batch_snapshots() {}
    bool require_single_segment_ = false;
    explicit RpuKernelGraph(GraphRuntimePolicy policy = {}, bool single = false) : runtime_policy_(policy), require_single_segment_(single) {}
    std::vector<GraphNode> nodes_;
    std::vector<Segment> segments_;
    std::vector<size_t> safe_split_boundaries_;
    std::vector<std::pair<size_t, size_t>> preload_prologue_spans_;
    std::vector<int> kernels_;
    std::optional<GraphSignature> pending_signature_;
    bool has_built_signature_ = false;
    GraphSignature built_signature_;
    uint64_t graph_lifetime_id_ = 1;
    struct PreparedQueueSlot {
        std::unique_ptr<rhino_lkn::Queue_t> queue;
        uint8_t core_num = 0;
        bool batch_snapshot_configured = false;
        int built_segment_idx = -1;
        bool hw_perf_enabled_at_build = false;
        std::shared_ptr<const int> fixed_dma_table;
        std::vector<int> kernel_register_tokens,
            pending_kernel_register_tokens;
        bool kernel_register_tokens_valid = false;
        // Replay packet contents are tested by the prepared-queue harness;
        // this core-budget double only models the queue replacement hook.
        void invalidate_mutable_dma_packets() noexcept {}
    };
    std::array<PreparedQueueSlot, 9> private_queue_slots_{};
    std::unique_ptr<rhino_lkn::Queue_t> data_dma_queue_;
    struct Stats {
        size_t dma_count = 0, barrier_count = 0, child_graph_count = 0;
        std::vector<uint64_t> child_graph_lifetime_ids;
        std::vector<Census> segment_census;
    } last_stats_;
    void execute_ddr_copy(const MemcpyNodeData&);
    template <typename ND> void capture_data(ND);
    bool replayable_ = true;
    void sync_admission() { __SYNC_GUARD__ }
    void downgrade_admission() { __DOWNGRADE_GUARD__ replayable_ = false; }
    bool capture_replay_child_skip_data(const MemcpyNodeData&) { return false; }
    void resolve_recording_input_deps(int, const std::vector<uint64_t>&) {}
    void track_recording_writer_dev_addr(uint64_t, int) {}
    void prepare_dma_preflight(Segment& seg) { __PREPARE_DMA_PREFLIGHT__ }
    void build_segments_from_nodes();
    rhino_lkn::Queue_t* ensure_private_queue(size_t, size_t);
    rhino_lkn::Queue_t* ensure_private_queue(size_t count) {
        return ensure_private_queue(0, count);
    }
    void register_budget(const std::shared_ptr<RpuKernelGraph>& child) { __REGISTER_GUARD__ }
    void replay_child_cores(const GraphNode& kn, const std::vector<uint8_t>& core_ids) { __CHILD_REPLAY_GUARD__ }
    void replay_parent_cores(const GraphNode& kn, const std::vector<uint8_t>& core_ids) { __PARENT_REPLAY_GUARD__ }
    void replace_budget(const std::shared_ptr<RpuKernelGraph>& child) { __REPLACE_GUARD__ }
    std::shared_ptr<RpuKernelGraph> clone_policy() const {
        __CHILD_CONSTRUCTOR__
        return child;
    }
};
__FUNCTIONS__
GraphNode kernel(std::initializer_list<uint8_t> cores, bool isolate = false) {
    GraphNode node; node.core_ids = cores; node.isolate = isolate; return node;
}
GraphNode dma(uint8_t channel = 0) {
    GraphNode node; node.kind = GraphNodeKind::Dma; node.channel = channel; return node;
}
GraphNode barrier() { GraphNode node; node.kind = GraphNodeKind::Barrier; return node; }
GraphNode branch(bool metadata = true) {
    GraphNode node; node.kind=GraphNodeKind::Branch; node.branch.branch_key=71;
    node.branch.label=metadata ? BranchNodeData::physical_manifest_label() : "ordinary";
    return node;
}
GraphSignature causal_prefill_signature() {
    GraphSignature sig;
    sig.op_id = __CAUSAL_OP_ID__;
    sig.op_id_str = "rpu_causal_decoder";
    sig.shapes = {1,192,192,5120};
    sig.dyn_dims = {40,0,2,4,4,4,4,8,0,0,32,0,
                   759653830,991875516,3022072701,1526521971,
                   2372142790,3680631350,1668011494,1603273980};
    sig.dtypes = {at::kHalf};
    return sig;
}
template<class F> void rejects(F&& f) {
    bool failed = false;
    try { f(); } catch (const std::runtime_error&) { failed = true; }
    assert(failed);
}
void check_segment(RpuKernelGraph& graph, size_t count) {
    graph.build_segments_from_nodes();
    assert(graph.segments_.size() == 1);
    const auto& segment = graph.segments_.front();
    assert(segment.core_ids.size() == count);
    assert(segment.census.core_count == count);
    for (size_t i = 0; i < count; ++i) assert(segment.core_ids[i] == i);
    assert(graph.ensure_private_queue(count)->count == count);
}
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode = argv[1];
    GraphRuntimePolicy policy; policy.execution_core_count = 6;
    if (mode == "profile_scope") {
        test_native_budget = true;
        policy.execution_core_count = 4;
        const auto sig = causal_prefill_signature();
        const auto check = [&](std::optional<GraphSignature> key,
                               GraphRuntimePolicy owner, size_t pieces) {
            RpuKernelGraph graph(owner);
            graph.pending_signature_ = key;
            graph.nodes_.assign(1025, kernel({0,1,2,3}));
            graph.build_segments_from_nodes();
            assert(graph.segments_.size() == pieces);
            if (pieces == 2) assert(graph.segments_[0].end_idx == 1024);
            assert(graph.runtime_policy_.lkn_max_batch_entries == owner.lkn_max_batch_entries);
            assert(graph.runtime_policy_.lkn_kd_buf_mb == owner.lkn_kd_buf_mb);
            assert(graph.runtime_policy_.lkn_instr_buf_mb == owner.lkn_instr_buf_mb);
        };
        check(sig, policy, 2);
        check(std::nullopt, policy, 1);
        // Debug labels do not change native policy selection.
        for (const std::string label : {"", "not a causal decoder"}) {
            auto changed=sig; changed.op_id_str=label; check(changed,policy,2);
        }
        auto changed=sig; ++changed.op_id; check(changed,policy,1);
        for (size_t i=0; i<sig.shapes.size(); ++i) {
            changed=sig; ++changed.shapes[i]; check(changed,policy,1);
        }
        for (size_t i=0; i<12; ++i) {
            changed=sig; ++changed.dyn_dims[i]; check(changed,policy,1);
        }
        // Plan hashes are not weight-precision tags. Same-shape W8/W4 plan
        // identities get the small budget too; hardware numerics are separate.
        for (size_t i=12; i<20; ++i) {
            changed=sig; ++changed.dyn_dims[i]; check(changed,policy,2);
        }
        changed=sig; changed.dyn_dims.pop_back(); check(changed,policy,1);
        changed=sig; changed.dyn_dims.push_back(0); check(changed,policy,1);
        changed=sig; changed.dtypes={6}; check(changed,policy,1);
        changed=sig; changed.flags=1; check(changed,policy,1);
        changed=sig; changed.branch_key=1; check(changed,policy,1);
        changed=sig; changed.segment_key=1; check(changed,policy,1);
        changed=sig; changed.shapes={1,1,1,5120};
        changed.dyn_dims[10]=0; changed.dyn_dims[11]=-1; check(changed,policy,1);
        for (int cores : {6,8}) {
            auto other=policy; other.execution_core_count=cores; check(sig,other,1);
        }
        RpuKernelGraph built(policy);
        built.has_built_signature_=true; built.built_signature_=sig;
        built.nodes_.assign(1025,kernel({0,1,2,3}));
        built.build_segments_from_nodes(); assert(built.segments_.size()==2);
        built.pending_signature_=GraphSignature{};
        built.build_segments_from_nodes(); assert(built.segments_.size()==1);
        const auto baseline=native_segment_resource_budget("rpu_causal_decoder",policy,false);
        assert(baseline.entries==65536 && baseline.kd_bytes==(size_t{8}<<20));
        assert(baseline.instr_bytes==(size_t{64}<<20));
        assert(policy.lkn_max_batch_entries==65536 && policy.lkn_kd_buf_mb==8
               && policy.lkn_instr_buf_mb==64);
        // The unified ceiling cannot enlarge a smaller caller/SDK resource limit.
        auto lower=policy; lower.lkn_max_batch_entries=1024;
        lower.lkn_kd_buf_mb=2; lower.lkn_instr_buf_mb=8;
        RpuKernelGraph small(lower); small.pending_signature_=sig;
        small.nodes_.assign(1025,kernel({0,1,2,3}));
        small.build_segments_from_nodes(); assert(small.segments_.size()==2);
        assert(small.segments_[0].end_idx==1024 && small.segments_[1].end_idx==1025);
        const auto lower_budget=native_segment_resource_budget("rpu_causal_decoder",lower,false);
        assert(lower_budget.kd_bytes==(size_t{2}<<20)
               && lower_budget.instr_bytes==(size_t{8}<<20));
        // Safe split boundaries + preload prologue replay on segment_id > 0.
        test_native_budget=false; test_entry_budget=10;
        RpuKernelGraph split_safe(policy);
        split_safe.nodes_.assign(16, kernel({0,1,2,3}));
        split_safe.preload_prologue_spans_={{0, 2}};
        split_safe.safe_split_boundaries_={2, 7, 12};
        split_safe.build_segments_from_nodes();
        assert(split_safe.segments_.size()==3);
        // Segment 0 covers [0, 7): cuts cleanly at safe boundary 7 before hitting 10.
        assert(split_safe.segments_[0].prologue_spans.empty());
        assert(split_safe.segments_[0].start_idx==0 && split_safe.segments_[0].end_idx==7);
        // Segment 1 prepends prologue [0, 2) (2 entries), so body [7, 12) + 2 = 7 <= 10,
        // and adding [12, 16) would reach 11 > 10 -> cuts cleanly at safe boundary 12!
        assert((split_safe.segments_[1].prologue_spans==std::vector<std::pair<size_t,size_t>>{{0, 2}}));
        assert(split_safe.segments_[1].start_idx==7 && split_safe.segments_[1].end_idx==12);
        assert(split_safe.segments_[1].census.kernel_count==5);
        assert((split_safe.segments_[2].prologue_spans==std::vector<std::pair<size_t,size_t>>{{0, 2}}));
        assert(split_safe.segments_[2].start_idx==12 && split_safe.segments_[2].end_idx==16);
        assert(split_safe.segments_[2].census.kernel_count==4);
    } else if (mode == "profile_fence") {
        test_native_budget=true; policy.execution_core_count=4;
        RpuKernelGraph graph(policy); graph.pending_signature_=causal_prefill_signature();
        graph.nodes_.assign(1023,kernel({0,1,2,3}));
        auto begin=barrier(); begin.self_stream=2; begin.target_stream=0;
        auto joined=barrier(); joined.self_stream=0; joined.target_stream=2;
        graph.nodes_.push_back(begin); graph.nodes_.push_back(dma(1));
        graph.nodes_.push_back(branch()); graph.nodes_.push_back(kernel({0,1,2,3}));
        graph.nodes_.push_back(joined); graph.nodes_.push_back(kernel({0,1,2,3}));
        graph.build_segments_from_nodes();
        assert(graph.segments_.size()==2);
        // The indivisible burst crosses 1024; cut only after its join. The
        // metadata marker consumes no entry and cannot prematurely split it.
        assert(graph.segments_[0].end_idx==1028 && graph.segments_[1].start_idx==1028);
        assert(graph.segments_[0].census.kernel_count==1024);
        assert(graph.segments_[0].census.dma_count==1
               && graph.segments_[0].census.barrier_count==2);
        assert(graph.segments_[0].core_ids==std::vector<uint8_t>({0,1,2,3}));
        assert(graph.ensure_private_queue(4)->count==4);
    } else if (mode == "segments") {
        for (int count : {8, 6, 4, 6, 1, 8}) {
            policy.execution_core_count = count;
            RpuKernelGraph graph(policy);
            graph.nodes_ = {dma()}; check_segment(graph, count);
            graph.nodes_ = {barrier()}; check_segment(graph, count);
            graph.nodes_ = {dma(), kernel({0})}; check_segment(graph, count);
            // Highest admitted DMA engine is in the owner's resource prefix.
            graph.nodes_ = {dma(count - 1)}; check_segment(graph, count);
            if (count < 8) {
                graph.nodes_ = {dma(count)};
                rejects([&] { graph.build_segments_from_nodes(); });
                assert(graph.segments_.empty());
            }
        }
        policy.execution_core_count = 6;
        RpuKernelGraph graph(policy);
        graph.nodes_ = {dma(), kernel({0, 1, 2, 3}), kernel({0, 1, 2, 3, 4, 5})};
        check_segment(graph, 6);
        const int before_replay = rhino_lkn::Queue_t::creations;
        auto* retained = graph.ensure_private_queue(6);
        for (int replay = 0; replay < 3; ++replay) assert(graph.ensure_private_queue(6) == retained);
        assert(rhino_lkn::Queue_t::creations == before_replay);
        graph.nodes_ = {kernel({0, 1, 2, 3})}; check_segment(graph, 4);
        // Compute-first mixed segments must also reserve the DMA engine.
        graph.nodes_ = {kernel({0, 1, 2, 3}), dma(5)}; check_segment(graph, 6);
        graph.prepare_dma_preflight(graph.segments_[0]);
        graph.segments_[0].core_ids.resize(4);
        rejects([&] { graph.prepare_dma_preflight(graph.segments_[0]); });
        graph.nodes_ = {kernel({0, 1, 2, 3}), dma(3)}; check_segment(graph, 4);
        graph.nodes_ = {kernel({4, 5})}; check_segment(graph, 6);
        graph.nodes_ = {kernel({4, 5}, true)}; check_segment(graph, 6);
        graph.nodes_ = {kernel({0, 1, 2, 3}, true), dma(), kernel({0, 1, 2, 3, 4, 5})};
        graph.build_segments_from_nodes();
        assert(graph.segments_.size() == 2);
        assert(graph.segments_[0].core_ids.size() == 4 && graph.segments_[1].core_ids.size() == 6);
    } else if (mode == "open_fence") {
        policy.execution_core_count = 8;
        RpuKernelGraph graph(policy);
        auto fence = [](uint8_t self, uint8_t target) {
            auto node = barrier(); node.self_stream = self; node.target_stream = target; return node;
        };
        auto other = kernel({0}); other.queue_state.flush_icache = false;
        // A deferred burst stays open across an independent kernel and is
        // joined before the queue-state cut.
        graph.nodes_ = {kernel({0}), fence(2, 0), dma(1), kernel({0}), fence(0, 2), other};
        graph.build_segments_from_nodes();
        assert(graph.segments_.size() == 2 && graph.segments_[0].end_idx == 5);
        // No forced cut may separate the burst from its join.
        graph.nodes_ = {kernel({0}), fence(2, 0), dma(1), other, fence(0, 2)};
        rejects([&] { graph.build_segments_from_nodes(); });
        graph.nodes_ = {fence(2, 0), dma(1), kernel({0}, true), fence(0, 2)};
        rejects([&] { graph.build_segments_from_nodes(); });
        graph.nodes_ = {fence(4, 0), kernel({0}), GraphNode(GraphNodeKind::Memcpy, MemcpyNodeData{}), fence(0, 4)};
        rejects([&] { graph.build_segments_from_nodes(); });
        // Channel 0 is stream 0 itself and never opens a fence.
        graph.nodes_ = {dma(0), other, GraphNode(GraphNodeKind::Memcpy, MemcpyNodeData{})};
        graph.build_segments_from_nodes(); assert(graph.segments_.size() == 2);
    } else if (mode == "single_segment") {
        policy.execution_core_count = 8;
        RpuKernelGraph graph(policy);
        graph.require_single_segment_ = true;
        graph.nodes_ = {branch(), kernel({0}), branch(), kernel({0}), branch()};
        check_segment(graph, 1);
        graph.nodes_ = {kernel({0}), branch(false), kernel({0})};
        rejects([&] { graph.build_segments_from_nodes(); });
        assert(graph.segments_.empty());
        test_entry_budget = 1;
        graph.nodes_ = {kernel({0}), kernel({0})};
        rejects([&] { graph.build_segments_from_nodes(); });
        test_entry_budget = 1000;
        auto other = kernel({0}); other.queue_state.flush_icache = false;
        graph.nodes_ = {kernel({0}), other};
        rejects([&] { graph.build_segments_from_nodes(); });
        graph.nodes_ = {branch()};
        rejects([&] { graph.build_segments_from_nodes(); });
        // The graph-aware CPU dispatcher executes immediately after capture or
        // sync. Even an empty prefix must reject before that CPU side effect.
        graph.nodes_.clear();
        const int creations = rhino_lkn::Queue_t::creations;
        int cpu_effects = 0;
        for (auto state : {RpuKernelGraph::State::RECORDING,
                           RpuKernelGraph::State::REPLAYING}) {
            graph.state_ = state;
            rejects([&] { graph.sync_admission(); ++cpu_effects; });
            rejects([&] { graph.capture_data(MemcpyNodeData{}); ++cpu_effects; });
            rejects([&] { graph.downgrade_admission(); ++cpu_effects; });
            assert(graph.replayable_ && graph.nodes_.empty() && cpu_effects == 0);
        }
        assert(rhino_lkn::Queue_t::creations == creations);
        // Existing owners keep admission and downgrade behavior.
        graph.require_single_segment_ = false;
        graph.state_ = RpuKernelGraph::State::RECORDING;
        graph.sync_admission(); graph.capture_data(MemcpyNodeData{});
        graph.downgrade_admission();
        assert(!graph.replayable_ && graph.nodes_.size() == 1);

    } else if (mode == "metadata") {
        RpuKernelGraph graph(policy);
        graph.nodes_={kernel({0}),branch(),kernel({0})};
        check_segment(graph,1);
        assert(graph.nodes_.size()==3 && graph.segments_[0].end_idx==3);
        assert(graph.segments_[0].census.kernel_count==2);
        graph.nodes_={kernel({0}),branch(false),kernel({0})};
        graph.build_segments_from_nodes(); assert(graph.segments_.size()==2);
        for (int bad=0;bad<3;++bad) {
            auto invalid=branch();
            if(bad==0) invalid.branch.branch_key=0;
            if(bad==1) invalid.branch.label="not metadata";
            if(bad==2) invalid.branch.branch_signature.flags=1;
            graph.nodes_={kernel({0}),invalid,kernel({0})};
            graph.build_segments_from_nodes(); assert(graph.segments_.size()==2);
        }
        graph.nodes_={branch(),branch()}; graph.build_segments_from_nodes();
        assert(graph.segments_.empty() && graph.nodes_.size()==2);
        graph.nodes_={branch(),kernel({0}),branch(),kernel({0}),branch()};
        check_segment(graph,1); assert(graph.segments_[0].start_idx==1);
        assert(graph.segments_[0].end_idx==5 && graph.nodes_.size()==5);
        // Metadata consumes no entry budget and cannot suppress a real cut.
        test_entry_budget=2;
        graph.nodes_={kernel({0}),branch(),kernel({0}),branch(),kernel({0})};
        graph.build_segments_from_nodes(); assert(graph.segments_.size()==2);
        assert(graph.segments_[0].end_idx==4 && graph.segments_[1].start_idx==4);
        // Nor can metadata break an outstanding cross-stream fence.
        test_entry_budget=1;
        auto wait=barrier(); wait.self_stream=0;wait.target_stream=2;
        graph.nodes_={dma(1),branch(),kernel({0}),wait,kernel({0})};
        graph.build_segments_from_nodes(); assert(graph.segments_.size()==2);
        assert(graph.segments_[0].end_idx==4 && graph.segments_[1].start_idx==4);
        test_entry_budget=1000;
        auto other=kernel({0}); other.queue_state.flush_icache=false;
        graph.nodes_={kernel({0}),branch(),other};
        graph.build_segments_from_nodes(); assert(graph.segments_.size()==2);
    } else if (mode == "admission") {
        for (int count : {0, -1, 9, 256}) {
            auto invalid = policy; invalid.execution_core_count = count;
            rejects([&] { validate_graph_runtime_policy(invalid); });
        }
        for (int count : {1, 4, 6, 8}) {
            auto valid = policy; valid.execution_core_count = count;
            validate_graph_runtime_policy(valid);
        }
        RpuKernelGraph graph(policy);
        const int before = rhino_lkn::Queue_t::creations;
        for (const auto& bad : {kernel({}), kernel({0, 6}), kernel({6}), kernel({7}),
                               kernel({0, 1, 2, 3, 4, 5, 6}), kernel({1, 0}), kernel({0, 0}), kernel({0, 2})}) {
            rejects([&] { validated_kernel_core_extent(policy, bad.core_ids); });
            graph.nodes_ = {dma(), kernel({0}), bad};
            rejects([&] { graph.build_segments_from_nodes(); });
            assert(graph.segments_.empty());
        }
        for (size_t count : {0, 7, 8, 256}) rejects([&] { graph.ensure_private_queue(count); });
        for (int count : {4, 6, 8}) {
            graph.runtime_policy_.execution_core_count = count;
            validate_graph_dma_channel(count - 1, count, "highest valid");
            for (int channel : {-1, count, 8, 255}) {
                rejects([&] { validate_graph_dma_channel(channel, count, "invalid"); });
                if (channel >= 0) {
                    graph.nodes_ = {kernel({0}), dma(channel)};
                    rejects([&] { graph.build_segments_from_nodes(); });
                    assert(graph.segments_.empty());
                    GraphNode copy; copy.kind = GraphNodeKind::Memcpy;
                    std::get<MemcpyNodeData>(copy.data).dma_channel = channel;
                    graph.nodes_ = {kernel({0}), copy};
                    rejects([&] { graph.build_segments_from_nodes(); });
                    assert(graph.segments_.empty());
                }
            }
        }
        assert(rhino_lkn::Queue_t::creations == before);
    } else if (mode == "children") {
        policy.force_oneshot_on_replay = true;
        policy.lkn_kd_buf_mb = 3;
        RpuKernelGraph parent(policy);
        parent.require_single_segment_ = true;
        auto inherited = parent.clone_policy();
        assert(inherited->require_single_segment_);
        assert(inherited->runtime_policy_.execution_core_count == 6);
        assert(inherited->runtime_policy_.force_oneshot_on_replay);
        assert(inherited->runtime_policy_.lkn_kd_buf_mb == 3);
        parent.require_single_segment_ = false;
        for (int count : {4, 6, 8}) {
            auto child_policy = policy; child_policy.execution_core_count = count;
            auto child = std::make_shared<RpuKernelGraph>(child_policy);
            GraphNode node; node.kind = GraphNodeKind::ChildGraph; node.child = child;
            parent.nodes_ = {node};
            if (count <= 6) {
                parent.register_budget(child); parent.replace_budget(child);
                parent.build_segments_from_nodes();
                assert(parent.last_stats_.child_graph_count == 1);
            } else {
                rejects([&] { parent.register_budget(child); });
                rejects([&] { parent.replace_budget(child); });
                rejects([&] { parent.build_segments_from_nodes(); });
            }
        }
        RpuKernelGraph default_parent;
        default_parent.register_budget(inherited);
        default_parent.replace_budget(inherited);
    } else if (mode == "replay") {
        RpuKernelGraph graph(policy);
        auto recorded = kernel({0, 1, 2, 3});
        for (int replay = 0; replay < 3; ++replay) {
            graph.replay_child_cores(recorded, {0, 1, 2, 3});
            graph.replay_parent_cores(recorded, {0, 1, 2, 3});
        }
        // Both spans fit budget6; changing a prepared kernel's span is still invalid.
        assert(validated_kernel_core_extent(policy, {0, 1, 2, 3, 4, 5}) == 6);
        rejects([&] { graph.replay_child_cores(recorded, {0, 1, 2, 3, 4, 5}); });
        rejects([&] { graph.replay_parent_cores(recorded, {0, 1, 2, 3, 4, 5}); });
        MemcpyNodeData copy; copy.bytes = 64; copy.dma_channel = 3;
        graph.capture_data(copy);
        assert(graph.nodes_.size() == 1);
        graph.state_ = RpuKernelGraph::State::REPLAYING;
        for (int replay=0; replay<3; ++replay) {
            graph.cursor_ = 0; graph.capture_data(copy); assert(graph.cursor_ == 1);
        }
        for (int channel : {0, 5, 6, 7, 8}) {
            auto changed = copy; changed.dma_channel = channel;
            graph.cursor_ = 0;
            rejects([&] { graph.capture_data(changed); });
            assert(graph.cursor_ == 0 && graph.nodes_.size() == 1);
            assert(std::get<MemcpyNodeData>(graph.nodes_[0].data).dma_channel == 3);
        }
        for (int channel : {6, 7, 8}) {
            RpuKernelGraph fresh(policy); copy.dma_channel = channel;
            rejects([&] { fresh.capture_data(copy); });
            assert(fresh.nodes_.empty());
        }
    } else if (mode == "queue_bounds") {
        QueueCache sdk; RpuQueueCache proxy;
        const int before = rhino_lkn::Queue_t::creations;
        for (int64_t count : {0, -1, 9, 256, 260}) {
            rejects([&] { sdk.get(count); });
            rejects([&] { proxy.get(count); });
        }
        assert(before == rhino_lkn::Queue_t::creations);
        for (int count : {1, 4, 6, 8}) {
            bool fresh = false;
            auto* queue = sdk.get(count, &fresh);
            assert(fresh && queue->count == count);
            assert(sdk.get(count, &fresh) == queue && !fresh);
            assert(proxy.get(count)->count == count);
        }
    } else if (mode == "ddr_copy") {
        using Q = rhino_lkn::Queue_t;
        rhino_lkn::Buffer_t source{std::make_shared<std::vector<uint8_t>>(160)};
        rhino_lkn::Buffer_t target{std::make_shared<std::vector<uint8_t>>(192)};
        live_buffers = {&source, &target};
        for (size_t i=0; i<source.data->size(); ++i) (*source.data)[i]=i;
        MemcpyNodeData copy{target.data->data()+64, source.data->data()+32, 64, 7};
        for (int count : {1, 4, 6, 8}) {
            policy.execution_core_count = count;
            copy.dma_channel = count - 1;
            RpuKernelGraph graph(policy);
            auto* retained = graph.ensure_private_queue(count);
            retained->built = retained->arena = true; ++Q::arenas;
            graph.execute_ddr_copy(copy);
            auto* direct = graph.data_dma_queue_.get();
            const int before = Q::creations;
            for (int replay=0; replay<3; ++replay) graph.execute_ddr_copy(copy);
            assert(direct && direct != retained && graph.data_dma_queue_.get() == direct);
            assert(Q::creations == before && retained->built && retained->arena);
            assert(Q::arenas == 1 && Q::leases == 0);
            assert(source.data.use_count() == 1 && target.data.use_count() == 1);
            const auto& submit = Q::submissions.back();
            assert(submit.cores == count && submit.channel == count - 1);
            assert(submit.src_offset == 32 && submit.dst_offset == 64 && submit.bytes == 64);
            assert(std::equal(source.data->begin()+32, source.data->begin()+96, target.data->begin()+64));
            copy.dma_channel = -1; graph.execute_ddr_copy(copy);
            assert(Q::submissions.back().channel == 0); copy.dma_channel = count - 1;
            const auto submission_count = Q::submissions.size();
            auto outside = copy; outside.dma_channel = count;
            rejects([&] { graph.execute_ddr_copy(outside); });
            assert(Q::submissions.size() == submission_count && graph.data_dma_queue_.get() == direct);
        }
        assert(Q::arenas == 0 && Q::leases == 0);
        policy.execution_core_count = 6;
        copy.dma_channel = 5;
        for (const std::string failure : {"add", "build", "enqueue", "discard", "throw_enqueue"}) {
            RpuKernelGraph graph(policy);
            Q::failure = failure;
            try { graph.execute_ddr_copy(copy); assert(false); }
            catch (const std::runtime_error& error) {
                if (failure == "throw_enqueue") assert(std::string(error.what()) == "original submission failure");
            }
            assert(!graph.data_dma_queue_ && Q::arenas == 0 && Q::leases == 0);
            assert(source.data.use_count() == 1 && target.data.use_count() == 1);
            Q::failure.clear();
            graph.execute_ddr_copy(copy);
            assert(Q::arenas == 0 && Q::leases == 0);
        }
        assert(Q::waits == 1);
        RpuKernelGraph graph(policy);
        const int before = Q::creations;
        for (size_t bytes : {size_t{0}, size_t{16}, (size_t{8}<<20)+32}) {
            auto bad = copy; bad.bytes = bytes; rejects([&] { graph.execute_ddr_copy(bad); });
        }
        for (int channel : {-2, 6, 7, 8}) {
            auto bad = copy; bad.dma_channel = channel; rejects([&] { graph.execute_ddr_copy(bad); });
        }
        auto bad = copy; bad.src = nullptr;
        rejects([&] { graph.execute_ddr_copy(bad); });
        bad = copy; bad.dst = target.data->data()+160;
        rejects([&] { graph.execute_ddr_copy(bad); });
        for (int count : {0, -1, 9, 256, 260}) {
            graph.runtime_policy_.execution_core_count = count;
            rejects([&] { graph.execute_ddr_copy(copy); });
        }
        assert(Q::creations == before && Q::arenas == 0 && Q::leases == 0);
    } else { assert(false); }
}
'''
    for marker, text in (
        ("__POLICY__", policy), ("__FUNCTIONS__", functions),
        ("__CAUSAL_OP_ID__", str(causal_decoder_op_id) + "LL"),
        ("__NATIVE_BUDGET__", "\n".join([
            _function(execute, "static bool lingbot2_exact_graph_owns_sdk_budget("),
            _function(execute, "static SegmentResourceBudget segment_resource_budget(")
                .replace("segment_resource_budget(", "native_segment_resource_budget(", 1),
        ])),
        ("__SYNC_GUARD__", sync_guard), ("__DOWNGRADE_GUARD__", downgrade_guard),
        ("__SIGNATURE__", _function(infra, "struct GraphSignature {") + ";"),
        ("__BRANCH__", _function(graph_header, "struct BranchNodeData {") + ";"),
        ("__REGISTER_GUARD__", child_guards[0]),
        ("__REPLACE_GUARD__", child_guards[1]),
        ("__CHILD_CONSTRUCTOR__", constructor),
        ("__CHILD_REPLAY_GUARD__", replay_guards[0]),
        ("__PARENT_REPLAY_GUARD__", replay_guards[1]),
        ("__SDK_CACHE_GET__", sdk_cache_get), ("__PROXY_CACHE_GET__", proxy_cache_get),
        ("__TYPED_ADD__", typed_add),
        ("__PREPARE_DMA_PREFLIGHT__", prepare_preflight),
    ):
        harness = harness.replace(marker, text)
    directory = tmp_path_factory.mktemp("graph-core-budget")
    source = directory / "probe.cpp"
    source.write_text(harness)
    binary = directory / "probe"
    compiled = subprocess.run([compiler, "-std=c++17", "-O0", str(source), "-o", str(binary)],
                              capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    return binary


@pytest.mark.parametrize("mode", ["segments", "open_fence", "single_segment", "metadata", "admission", "children", "replay", "queue_bounds", "ddr_copy", "profile_scope", "profile_fence"])
def test_native_graph_execution_budget(native_budget_probe, mode):
    subprocess.run([str(native_budget_probe), mode], check=True, capture_output=True, text=True)


def test_enqueue_rejects_outside_budget_before_dispatch_and_fixture_binds_owner():
    execute = (ROOT / "src/graph/graph_runtime_execute.cpp").read_text()
    start = execute.index("const QueueLaunchState& queue_state) {")
    body = execute[start:execute.index("case State::RECORDING:", start)]
    assert body.index("validated_kernel_core_extent(runtime_policy_, core_ids, kernel.get_name())") < body.index("switch (state_)")
    assert "passthrough_kernel_queue(core_extent)" in body
    binding = (ROOT / "src/graph/graph_pybind.cpp").read_text()
    assert 'py::arg("execution_core_count") = 8' in binding
