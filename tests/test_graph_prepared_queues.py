"""Compile production queue/replay methods against an instrumented host SDK.

No backend import, RPU device, or opaque operator assets are required. The mock
checks address publication and resource ownership; actual SDK/cache coherency
and model numerics still require the separately serialized board regression.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXECUTE = ROOT / "src/graph/graph_runtime_execute.cpp"
HEADER = ROOT / "src/graph/graph_runtime.h"


def _between(text, start, end):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


def test_kernel_param_dirty_witness_covers_every_replay_mutation_path():
    execute = EXECUTE.read_text()
    child = (ROOT / "src/graph/graph_runtime_child.cpp").read_text()
    census = (ROOT / "src/graph/graph_register_census.cpp").read_text()
    runtime = (ROOT / "src/graph/graph_runtime.cpp").read_text()

    getters = _between(
        execute,
        "::rhino_lkn::Kernel_t* RpuKernelGraph::get_kernel(KernelId id)",
        "// =============================================================================\n// get_kernel_reset",
    )
    # Both KernelId/name overloads cover their own Graph and ChildGraph owner.
    assert getters.count("kernel_params_dirty_this_replay_ = true;") == 4

    patch_skip = _between(
        execute,
        "void RpuKernelGraph::skip_op_stream_with_patches(",
        "// =============================================================================\n// build_segments_from_nodes",
    )
    assert patch_skip.index("kernel_params_dirty_this_replay_ = true;") < (
        patch_skip.index("apply_register_patch_to_kernel(")
    )
    assert child.count("kernel_params_dirty_this_replay_ = true;") == 2
    typed_writer = _between(
        census,
        "void RpuKernelGraph::write_pending_kernel_ddr_registers(",
        "void RpuKernelGraph::consume_pending_kernel_register_census(",
    )
    assert typed_writer.index("kernel_params_dirty_this_replay_ = true;") < (
        typed_writer.index("write_register_pairs(")
    )

    sync_only = _between(
        execute,
        "uint32_t RpuKernelGraph::launch_segment_sync_only(",
        "static void mirror_per_segment_replay_count",
    )
    predicate = _between(
        sync_only,
        "const bool can_skip_param_sync =",
        "if (!can_skip_param_sync)",
    )
    assert "!kernel_params_dirty_this_replay_" in predicate
    assert "kernel_register_tokens_match" in predicate
    assert "seg.mutable_dmas.empty()" not in predicate
    assert "op_stream_fully_skipped_" not in predicate
    token_capture = _between(
        execute,
        "bool RpuKernelGraph::capture_segment_kernel_register_tokens(",
        "uint32_t RpuKernelGraph::launch_segment_sync_only(",
    )
    assert "query_register_state_token" in token_capture
    assert sync_only.index("__sync_synchronize();") < sync_only.index(
        "wq.enqueu_batch"
    )

    replay = _between(
        execute,
        "void RpuKernelGraph::execute_graph_for_replaying()",
        "// =============================================================================\n// execute_graph_oneshot",
    )
    assert replay.rstrip().endswith("kernel_params_dirty_this_replay_ = false;\n}")
    begin_hit = _between(runtime, "case State::BUILT:", "default:")
    assert "kernel_params_dirty_this_replay_ = false;" in begin_hit


def test_prepare_segment_queue_preflights_transactionally(tmp_path):
    """Compile the production prepare method against a fail-injecting SDK."""
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = EXECUTE.read_text()
    header = HEADER.read_text()
    queue_bounds = _between(
        source, "uint8_t validated_kernel_core_extent(",
        "void validate_semantic_dma_owner_for_execution(",
    )
    dma_bounds = _between(
        (ROOT / "src/graph/graph_runtime_internal.h").read_text(),
        "inline void validate_graph_dma_channel(",
        "inline void apply_register_patch_to_kernel(",
    )
    infra = (ROOT / "src/graph/graph_infra.h").read_text()
    metadata_types = (_between(infra, "struct GraphSignature {", "\n};") + "\n};\n" +
                      _between(header, "struct BranchNodeData {", "\n};") + "\n};\n")
    checked_address = _between(
        source,
        "uint64_t checked_dma_live_address(",
        "\n}\n\n}  // namespace",
    ) + "\n}"
    prepare = _between(
        source,
        "uint32_t RpuKernelGraph::prepare_segment_queue(",
        "\n}\n\n::rhino_lkn::Queue_t* RpuKernelGraph::launch_segment_for_replay",
    ) + "\n}"
    coalesce = _between(
        source, "void coalesce_fixed_dma_allocation_witnesses(",
        "\nuint8_t validated_kernel_core_extent(",
    )
    segment = _between(
        header,
        "    struct PreparedMutableDmaSlot {",
        "\n};\n\nstruct ReplayPlan",
    )
    harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("contract rejection"); } while (0)
#define RECORD_FUNCTION(...) do {} while (0)
struct Owner {};
Owner owner;
struct RpuDmaEndpoint {
    Owner* owner = nullptr;
    uint64_t offset = 0;
    uint64_t allocation_id = 0;
};
struct RpuDeviceDmaEndpointRequest {
    uint64_t device_addr = 0;
    size_t bytes = 0;
    const char* where = nullptr;
};
struct RpuDmaAllocationWitness {
    uint64_t allocation_id = 0;
    uint64_t expected_device_addr = 0;
    size_t bytes = 0;
    const char* where = nullptr;
};
int active_leases = 0;
struct RpuDmaSubmissionLease {
    bool active = false;
    RpuDmaSubmissionLease() = default;
    explicit RpuDmaSubmissionLease(bool value) : active(value) {
        if (active) ++active_leases;
    }
    ~RpuDmaSubmissionLease() { reset(); }
    RpuDmaSubmissionLease(const RpuDmaSubmissionLease&) = delete;
    RpuDmaSubmissionLease& operator=(const RpuDmaSubmissionLease&) = delete;
    RpuDmaSubmissionLease(RpuDmaSubmissionLease&& other) noexcept
        : active(other.active) { other.active = false; }
    RpuDmaSubmissionLease& operator=(RpuDmaSubmissionLease&& other) noexcept {
        if (this != &other) { reset(); active = other.active; other.active = false; }
        return *this;
    }
    explicit operator bool() const { return active; }
    void reset() { if (active) { --active_leases; active = false; } }
};
int resolve_calls = 0;
int fail_request = -1;
std::vector<RpuDeviceDmaEndpointRequest> last_requests;
RpuDmaSubmissionLease rpu_resolve_device_dma_endpoints(
        const RpuDeviceDmaEndpointRequest* requests, size_t count,
        RpuDmaEndpoint* endpoints) {
    ++resolve_calls;
    last_requests.clear();
    if (count != 0) last_requests.assign(requests, requests + count);
    std::vector<RpuDmaEndpoint> pending;
    pending.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (static_cast<int>(i) == fail_request) {
            throw std::runtime_error("injected endpoint failure");
        }
        pending.push_back({&owner, requests[i].device_addr,
                           requests[i].device_addr + 100});
    }
    std::copy(pending.begin(), pending.end(), endpoints);
    return RpuDmaSubmissionLease(count != 0);
}
template <typename T>
void validate_semantic_dma_owner_for_execution(
        const T&, size_t, const char*) {}
namespace rhino_lkn {
struct Kernel_t {};
class Queue_t {
public:
    int mutations = 0;
    int builds = 0;
    std::string entries;
    void set_enable_hw_perf(bool) { ++mutations; }
    void set_broadcast_mode(int) { ++mutations; }
    void set_flush_icache(bool) { ++mutations; }
    void add_barrier(uint8_t, uint8_t) { ++mutations; entries += 'B'; }
    uint32_t build_rc = 0;
    bool throw_build = false;
    uint32_t build_batch() {
        ++mutations; ++builds;
        if (throw_build) throw std::runtime_error("injected build failure");
        return build_rc;
    }
};
}
template <typename Grid, typename Cores>
void rpu_add_kernel_mutable_checked(
        rhino_lkn::Queue_t& q, rhino_lkn::Kernel_t&,
        const Grid&, const Cores&, const char*) { ++q.mutations; q.entries += 'K'; }
void rpu_add_dma_mutable_checked(
        rhino_lkn::Queue_t& q, const RpuDmaEndpoint&,
        const RpuDmaEndpoint&, size_t, uint8_t, const char*) {
    ++q.mutations; q.entries += 'D';
}
void rpu_add_dma_checked(
        rhino_lkn::Queue_t& q, const RpuDmaEndpoint&,
        const RpuDmaEndpoint&, size_t, uint8_t, const char*) {
    ++q.mutations; q.entries += 'D';
}
''' + metadata_types + r'''
enum class GraphNodeKind { Kernel, Dma, Barrier, Branch, Invalid };
struct KernelNodeData {
    size_t kernel_idx = 0;
    std::vector<uint32_t> grid_dims{1, 1, 1};
    std::vector<uint8_t> core_ids{0};
};
struct BarrierNodeData { uint8_t self_stream = 0, target_stream = 0; };
struct DmaNodeData {
    enum class Variant { Fixed, MutableSrc, MutableDst };
    Variant variant = Variant::Fixed;
    uint64_t src_addr = 0, dst_addr = 0;
    uint64_t semantic_endpoint_id = 0;
    const uint64_t* live_base = nullptr;
    int64_t live_offset = 0;
    size_t bytes = 16;
    uint8_t channel = 0;
    RpuDmaEndpoint src_endpoint, dst_endpoint;
};
struct GraphNode {
    GraphNodeKind kind = GraphNodeKind::Dma;
    KernelNodeData kernel;
    DmaNodeData dma;
    BarrierNodeData barrier;
    BranchNodeData branch;
    KernelNodeData& as_kernel() { return kernel; }
    DmaNodeData& as_dma() { return dma; }
    BarrierNodeData& as_barrier_node() { return barrier; }
    const BranchNodeData& as_branch() const { return branch; }
};
struct Segment {
    size_t start_idx = 0, end_idx = 0;
    std::vector<uint8_t> core_ids{0};
    struct { int broadcast_mode = 0; bool flush_icache = false; } queue_state;
    struct { size_t dma_count = 0; } census;
''' + segment + r'''
};
struct RpuKernelGraph {
    std::vector<GraphNode> nodes_;
    std::vector<std::unique_ptr<rhino_lkn::Kernel_t>> kernels_;
    struct GraphRuntimePolicy { int execution_core_count = 8; } runtime_policy_;
    uint32_t prepare_segment_queue(
        Segment&, rhino_lkn::Queue_t&, bool, RpuDmaSubmissionLease&,
        bool retain_replay_state = true);
};
using GraphRuntimePolicy = RpuKernelGraph::GraphRuntimePolicy;
''' + dma_bounds + queue_bounds + checked_address + coalesce + prepare + r'''
void assert_sentinel(const Segment& seg) {
    assert(seg.mutable_dmas.size() == 1);
    assert(seg.mutable_dmas[0].fixed_addr == 333);
    assert(seg.mutable_dmas[0].fixed_allocation_id == 444);
    assert(seg.fixed_dma_table->bindings.size() == 1);
    assert(seg.fixed_dma_table->bindings[0].src_addr == 555);
    assert(seg.fixed_dma_table->bindings[0].src_allocation_id == 666);
    assert(seg.fixed_dma_table->witnesses.size() == 1);
    assert(seg.fixed_dma_table->witnesses[0].allocation_id == 666);
    assert(seg.fixed_dma_table->witnesses[0].expected_device_addr == 555);
}
void seed_sentinel(Segment& seg, const uint64_t* live) {
    seg.mutable_dmas.push_back({
        9, 8, Segment::PreparedMutableDmaSlot::Kind::MutableSrc,
        live, 7, 333, 444, 32, 3});
    auto table = std::make_shared<Segment::PreparedFixedDmaTable>();
    table->bindings.push_back({9, 555, 556, 666, 667, 32, 3});
    table->witnesses.push_back({666, 555, 32, "sentinel"});
    seg.fixed_dma_table = std::move(table);
}
int main() {
    {
        // A compute-first segment must reserve the later DMA's engine before
        // endpoint resolution or any SDK batch mutation.
        RpuKernelGraph graph;
        graph.kernels_.push_back(std::make_unique<rhino_lkn::Kernel_t>());
        graph.nodes_.resize(2);
        graph.nodes_[0].kind = GraphNodeKind::Kernel;
        graph.nodes_[0].kernel.core_ids = {0, 1, 2, 3};
        graph.nodes_[1].dma.channel = 5;
        graph.nodes_[1].dma.src_addr = 1000;
        graph.nodes_[1].dma.dst_addr = 2000;
        Segment seg; seg.end_idx = 2; seg.census.dma_count = 1;
        seg.core_ids = {0, 1, 2, 3};
        rhino_lkn::Queue_t queue;
        RpuDmaSubmissionLease lease;
        bool rejected = false;
        try { graph.prepare_segment_queue(seg, queue, false, lease); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && resolve_calls == 0 && queue.mutations == 0);
        seg.core_ids = {0, 1, 2, 3, 4, 5};
        assert(graph.prepare_segment_queue(seg, queue, false, lease) == 0);
        assert(resolve_calls == 1 && queue.builds == 1);
        resolve_calls = 0;
    }
    {
        // Real prepare scans preserve K/DMA/Barrier order and mutable DMA IDs;
        // authenticated metadata never becomes an SDK packet or sidecar.
        RpuKernelGraph graph;
        graph.nodes_.resize(8);
        graph.kernels_.push_back(std::make_unique<rhino_lkn::Kernel_t>());
        graph.nodes_[0].kind=GraphNodeKind::Kernel;
        graph.nodes_[6].kind=GraphNodeKind::Kernel;
        for(size_t i:{1,4,7}) {
            graph.nodes_[i].kind=GraphNodeKind::Branch;
            graph.nodes_[i].branch.branch_key=70+i;
            graph.nodes_[i].branch.label=BranchNodeData::physical_manifest_label();
        }
        graph.nodes_[3].kind=GraphNodeKind::Barrier;
        uint64_t live=1000;
        for(size_t i:{2,5}) {
            auto& dma=graph.nodes_[i].dma;
            dma.variant=DmaNodeData::Variant::MutableSrc;
            dma.live_base=&live;dma.dst_addr=2000;dma.live_offset=i;
        }
        Segment seg;seg.end_idx=8;seg.census.dma_count=2;
        rhino_lkn::Queue_t queue;RpuDmaSubmissionLease lease;
        assert(graph.prepare_segment_queue(seg,queue,false,lease)==0);
        assert(queue.entries=="KDBDK" && queue.builds==1);
        assert(seg.mutable_dmas.size()==2 && seg.fixed_dma_table->bindings.empty());
        assert(seg.mutable_dmas[0].node_idx==2 && seg.mutable_dmas[0].dma_id==0);
        assert(seg.mutable_dmas[1].node_idx==5 && seg.mutable_dmas[1].dma_id==1);
        for(int bad=0;bad<3;++bad) {
            auto saved=graph.nodes_[4].branch;
            if(bad==0) graph.nodes_[4].branch.branch_key=0;
            if(bad==1) graph.nodes_[4].branch.label="ordinary";
            if(bad==2) graph.nodes_[4].branch.branch_signature.flags=1;
            rhino_lkn::Queue_t rejected_queue;RpuDmaSubmissionLease rejected_lease;
            bool rejected=false;
            try { graph.prepare_segment_queue(seg,rejected_queue,false,rejected_lease); }
            catch(const std::runtime_error&) { rejected=true; }
            assert(rejected && rejected_queue.mutations==0 && !rejected_lease);
            graph.nodes_[4].branch=saved;
        }
    }
    resolve_calls = 0;
    {
        // The last request fails after the complete four-endpoint envelope was
        // presented. No Queue, node endpoint, or Segment sidecar is published.
        uint64_t sentinel_live = 12;
        RpuKernelGraph graph;
        graph.nodes_.resize(2);
        graph.nodes_[0].dma.src_addr = 1000;
        graph.nodes_[0].dma.dst_addr = 1016;
        graph.nodes_[1].dma.src_addr = 1032;
        graph.nodes_[1].dma.dst_addr = 1048;
        graph.nodes_[0].dma.src_endpoint = {&owner, 77, 88};
        graph.nodes_[1].dma.dst_endpoint = {&owner, 99, 111};
        Segment seg;
        seg.end_idx = 2;
        seg.census.dma_count = 2;
        seed_sentinel(seg, &sentinel_live);
        const auto old_table = seg.fixed_dma_table;
        rhino_lkn::Queue_t queue;
        fail_request = 3;
        bool rejected = false;
        RpuDmaSubmissionLease failed_lease;
        try { graph.prepare_segment_queue(seg, queue, false, failed_lease); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
        assert(resolve_calls == 1 && last_requests.size() == 4);
        assert(last_requests[3].device_addr == 1048);
        assert(queue.mutations == 0 && queue.builds == 0);
        assert(graph.nodes_[0].dma.src_endpoint.offset == 77);
        assert(graph.nodes_[1].dma.dst_endpoint.offset == 99);
        assert_sentinel(seg);
        assert(seg.fixed_dma_table == old_table);
        assert(!failed_lease && active_leases == 0);

        fail_request = -1;
        RpuDmaSubmissionLease lease;
        assert(graph.prepare_segment_queue(seg, queue, true, lease) == 0);
        assert(resolve_calls == 2 && last_requests.size() == 4);
        assert(queue.builds == 1);
        assert(seg.fixed_dma_table && seg.fixed_dma_table != old_table);
        assert(old_table->bindings[0].src_addr == 555);
        assert(seg.mutable_dmas.empty() && seg.fixed_dma_table->bindings.size() == 2);
        assert(seg.fixed_dma_table->witnesses.size() == 4);
        for (size_t i = 0; i < 4; ++i) {
            assert(seg.fixed_dma_table->witnesses[i].expected_device_addr == 1000 + i * 16);
            assert(seg.fixed_dma_table->witnesses[i].allocation_id == 1100 + i * 16);
            assert(seg.fixed_dma_table->witnesses[i].bytes == graph.nodes_[i / 2].dma.bytes);
        }
        assert(graph.nodes_[0].dma.src_endpoint.offset == 1000);
        assert(graph.nodes_[1].dma.dst_endpoint.offset == 1048);
        assert(lease && active_leases == 1);
    }
    {
        // Repeated broadcast ranges keep both SDK entries and all node
        // sidecars, but publish only one live range per logical allocation.
        RpuKernelGraph graph;
        graph.nodes_.resize(2);
        for (size_t i = 0; i < 2; ++i) {
            graph.nodes_[i].dma.src_addr = 1000;
            graph.nodes_[i].dma.dst_addr = 2000;
            graph.nodes_[i].dma.bytes = 16 * (i + 1);
        }
        Segment seg;
        seg.end_idx = 2;
        seg.census.dma_count = 2;
        rhino_lkn::Queue_t queue;
        RpuDmaSubmissionLease lease;
        assert(graph.prepare_segment_queue(seg, queue, false, lease) == 0);
        assert(queue.entries == "DD" && seg.fixed_dma_table->bindings.size() == 2);
        assert(seg.fixed_dma_table->bindings[0].bytes == 16 && seg.fixed_dma_table->bindings[1].bytes == 32);
        assert(seg.fixed_dma_table->witnesses.size() == 2);
        assert(seg.fixed_dma_table->witnesses[0].allocation_id == 1100);
        assert(seg.fixed_dma_table->witnesses[1].allocation_id == 2100);
        for (const auto& witness : seg.fixed_dma_table->witnesses)
            assert(witness.bytes == 32);
    }
    for (bool retain_replay_state : {false, true}) {
      for (bool throws : {false, true}) {
        // Failed SDK preparation must not publish the new witness table or
        // endpoints, even after all ranges and queue entries were prepared.
        RpuKernelGraph graph;
        graph.nodes_.resize(1);
        auto& dma = graph.nodes_[0].dma;
        dma.src_addr = 1000; dma.dst_addr = 1016;
        dma.src_endpoint = {&owner, 77, 88};
        Segment seg; seg.end_idx = 1; seg.census.dma_count = 1;
        uint64_t live = 12; seed_sentinel(seg, &live);
        const auto old_table = seg.fixed_dma_table;
        rhino_lkn::Queue_t queue;
        queue.build_rc = 2; queue.throw_build = throws;
        RpuDmaSubmissionLease lease;
        bool threw = false;
        try { assert(graph.prepare_segment_queue(
            seg, queue, false, lease, retain_replay_state) == 2); }
        catch (const std::runtime_error&) { threw = true; }
        assert(threw == throws && queue.builds == 1);
        assert_sentinel(seg);
        assert(seg.fixed_dma_table == old_table);
        assert(dma.src_endpoint.offset == 77);
        assert(!lease && active_leases == 0);
      }
    }
    {
        // A successful one-shot still resolves the whole DMA envelope and
        // holds its submission lease, but publishes no replay-only sidecars.
        uint64_t live = 900;
        RpuKernelGraph graph;
        graph.nodes_.resize(2);
        graph.nodes_[0].dma.src_addr = 1000;
        graph.nodes_[0].dma.dst_addr = 2000;
        auto& mutable_dma = graph.nodes_[1].dma;
        mutable_dma.variant = DmaNodeData::Variant::MutableSrc;
        mutable_dma.live_base = &live;
        mutable_dma.live_offset = 16;
        mutable_dma.dst_addr = 3000;
        Segment seg; seg.end_idx = 2; seg.census.dma_count = 2;
        seed_sentinel(seg, &live);
        const auto old_table = seg.fixed_dma_table;
        rhino_lkn::Queue_t queue;
        RpuDmaSubmissionLease lease;
        assert(graph.prepare_segment_queue(seg, queue, false, lease, false) == 0);
        assert(queue.entries == "DD" && queue.builds == 1);
        assert(last_requests.size() == 4);
        assert(graph.nodes_[0].dma.src_endpoint.offset == 1000);
        assert(mutable_dma.src_endpoint.offset == 916);
        assert(seg.mutable_dmas.empty() && !seg.fixed_dma_table);
        assert(old_table->bindings[0].src_addr == 555);
        assert(lease && active_leases == 1);
        lease.reset();
        assert(active_leases == 0);
    }
    {
        // Kernel domains must fit the cold execution budget even when a
        // malformed Segment advertises a larger queue prefix.
        RpuKernelGraph graph;
        graph.runtime_policy_.execution_core_count = 4;
        graph.kernels_.push_back(std::make_unique<rhino_lkn::Kernel_t>());
        graph.nodes_.resize(1);
        graph.nodes_[0].kind = GraphNodeKind::Kernel;
        Segment seg; seg.end_idx = 1; seg.core_ids = {0, 1, 2, 3, 4, 5};
        for (const auto& ids : std::vector<std::vector<uint8_t>>{
                 {0, 1, 2, 3, 4}, {4}, {0, 2}, {2, 1}, {}}) {
            graph.nodes_[0].kernel.core_ids = ids;
            rhino_lkn::Queue_t queue;
            RpuDmaSubmissionLease lease;
            const int before_resolve = resolve_calls;
            bool rejected = false;
            try { graph.prepare_segment_queue(seg, queue, false, lease); }
            catch (const std::runtime_error&) { rejected = true; }
            assert(rejected && queue.mutations == 0 && !lease);
            assert(resolve_calls == before_resolve);
        }
        graph.nodes_[0].kernel.core_ids = {2, 3};
        seg.core_ids = {0, 1, 2, 3};
        rhino_lkn::Queue_t queue;
        RpuDmaSubmissionLease lease;
        assert(graph.prepare_segment_queue(seg, queue, false, lease) == 0);
        assert(queue.entries == "K" && queue.builds == 1);
    }
    {
        // An empty DMA envelope is a valid batch operation.
        resolve_calls = 0;
        fail_request = -1;
        RpuKernelGraph graph;
        Segment seg;
        rhino_lkn::Queue_t queue;
        RpuDmaSubmissionLease lease;
        assert(graph.prepare_segment_queue(seg, queue, false, lease) == 0);
        assert(resolve_calls == 1 && last_requests.empty());
        assert(queue.builds == 1);
        assert(seg.mutable_dmas.empty() && seg.fixed_dma_table->bindings.empty());
        assert(seg.fixed_dma_table->witnesses.empty());
        assert(!lease && active_leases == 0);
    }
    for (bool positive_overflow : {true, false}) {
        uint64_t live = positive_overflow
            ? std::numeric_limits<uint64_t>::max() - 2 : 0;
        RpuKernelGraph graph;
        graph.nodes_.resize(1);
        auto& dma = graph.nodes_[0].dma;
        dma.variant = DmaNodeData::Variant::MutableSrc;
        dma.live_base = &live;
        dma.live_offset = positive_overflow
            ? 3 : std::numeric_limits<int64_t>::min();
        dma.dst_addr = 2048;
        dma.src_endpoint = {&owner, 71, 72};
        Segment seg;
        seg.end_idx = 1;
        seg.census.dma_count = 1;
        seed_sentinel(seg, &live);
        rhino_lkn::Queue_t queue;
        resolve_calls = 0;
        bool rejected = false;
        RpuDmaSubmissionLease lease;
        try { graph.prepare_segment_queue(seg, queue, false, lease); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && resolve_calls == 0);
        assert(queue.mutations == 0 && queue.builds == 0);
        assert(dma.src_endpoint.offset == 71);
        assert_sentinel(seg);
        assert(!lease && active_leases == 0);
    }
}
'''
    cpp = tmp_path / "prepare_transaction.cpp"
    exe = tmp_path / "prepare_transaction"
    cpp.write_text(harness)
    subprocess.run(
        [compiler, "-std=c++17", "-O0", "-g", "-Wall", "-Wextra",
         str(cpp), "-o", str(exe)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("budget", [0, 8], ids=["safe-default", "bounded-retention"])
def test_production_prepared_queue_methods(tmp_path, budget):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = EXECUTE.read_text()
    header = HEADER.read_text()
    resources = _between(source, "namespace {\n\n// Queue_t allocates its batch arena", "uint32_t RpuKernelGraph::prepare_segment_queue")
    # Exercise the retained implementation in this isolated host executable;
    # never change the production default or a loaded backend's policy.
    resources = re.sub(r"kExtraPreparedQueueBudget = \d+;", f"kExtraPreparedQueueBudget = {budget};", resources)
    replay = _between(source, "::rhino_lkn::Queue_t* RpuKernelGraph::launch_segment_for_replay", "static void mirror_per_segment_replay_count")
    segment = _between(header, "    struct PreparedMutableDmaSlot {", "\n};\n\nstruct ReplayPlan")
    slots = _between(header, "    static constexpr size_t kMaxPreparedQueueSlots", "    // P6 (debug-level): per-graph cumulative")
    harness = r'''
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/types.h>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("contract rejection"); } while (0)
#define RECORD_FUNCTION(...) do {} while (0)
struct Owner {};
Owner owner;
struct RpuDmaEndpoint {
    Owner* owner = nullptr;
    uint64_t offset = 0;
    uint64_t allocation_id = 0;
};
struct RpuDeviceDmaEndpointRequest {
    uint64_t device_addr = 0;
    size_t bytes = 0;
    const char* where = nullptr;
};
struct RpuDmaAllocationWitness {
    uint64_t allocation_id = 0;
    uint64_t expected_device_addr = 0;
    size_t bytes = 0;
    const char* where = nullptr;
};
int active_submission_leases = 0;
int submission_without_lease = 0;
struct RpuDmaSubmissionLease {
    bool active = false;
    RpuDmaSubmissionLease() = default;
    explicit RpuDmaSubmissionLease(bool value) : active(value) {
        if (active) ++active_submission_leases;
    }
    ~RpuDmaSubmissionLease() { reset(); }
    RpuDmaSubmissionLease(const RpuDmaSubmissionLease&) = delete;
    RpuDmaSubmissionLease& operator=(const RpuDmaSubmissionLease&) = delete;
    RpuDmaSubmissionLease(RpuDmaSubmissionLease&& other) noexcept
        : active(other.active) { other.active = false; }
    RpuDmaSubmissionLease& operator=(RpuDmaSubmissionLease&& other) noexcept {
        if (this != &other) { reset(); active = other.active; other.active = false; }
        return *this;
    }
    explicit operator bool() const { return active; }
    void reset() {
        if (active) { --active_submission_leases; active = false; }
    }
};
std::map<uint64_t, uint64_t> allocation_ids;
std::map<uint64_t, Owner*> endpoint_owners;
int endpoint_resolve_calls = 0;
RpuDmaEndpoint rpu_resolve_device_dma_endpoint(uint64_t offset, size_t, const char*) {
    ++endpoint_resolve_calls;
    const auto it = allocation_ids.find(offset);
    const auto owner_it = endpoint_owners.find(offset);
    return {owner_it == endpoint_owners.end() ? &owner : owner_it->second,
            offset, it == allocation_ids.end() ? 0 : it->second};
}
RpuDmaSubmissionLease rpu_resolve_device_dma_endpoints(
        const RpuDeviceDmaEndpointRequest* requests, size_t count,
        RpuDmaEndpoint* endpoints) {
    std::vector<RpuDmaEndpoint> resolved;
    resolved.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        resolved.push_back(rpu_resolve_device_dma_endpoint(
            requests[i].device_addr, requests[i].bytes, requests[i].where));
    }
    std::copy(resolved.begin(), resolved.end(), endpoints);
    return RpuDmaSubmissionLease(count != 0);
}
RpuDmaSubmissionLease rpu_validate_live_dma_allocation_witnesses(
        const RpuDmaAllocationWitness* witnesses, size_t count) {
    bool has_ddr_witness = false;
    for (size_t i = 0; i < count; ++i) {
        if (witnesses[i].allocation_id == 0) continue;
        has_ddr_witness = true;
        const auto it = allocation_ids.find(witnesses[i].expected_device_addr);
        TORCH_CHECK(it != allocation_ids.end() &&
                        it->second == witnesses[i].allocation_id,
                    "stale allocation witness");
    }
    return RpuDmaSubmissionLease(has_ddr_witness);
}
template <typename T>
void validate_semantic_dma_owner_for_execution(
        const T&, size_t, const char*) {}
struct Transfer { uint64_t src, dst; bool operator==(const Transfer& x) const { return src == x.src && dst == x.dst; } };
namespace rhino_lkn {
static constexpr uint32_t kKernelQueryOk = 0;
class Kernel_t {
public:
    explicit Kernel_t(int* value) : value(value) {}
    uint32_t query_register_state_token(uint64_t* out) const noexcept {
        if (!out || !query_ok) return 2;
        *out = token;
        return kKernelQueryOk;
    }
    void write(int next) { *value = next; ++token; }
    int* value;
    uint64_t token = 1;
    bool query_ok = true;
};
class Queue_t {
public:
    static inline int live = 0;
    static inline int injected_failure = 0;
    explicit Queue_t(size_t cores) : cores(cores) { ++live; }
    ~Queue_t() { --live; }
    size_t cores;
    bool perf = false;
    bool built = false;
    int fail_sync = 0;
    int fail_submit = 0;
    int fail_update_at = -1;
    int updates = 0;
    int syncs = 0;
    int submits = 0;
    int* kernel = nullptr;
    bool has_dma = false;
    int published_kernel = 0;
    std::vector<Transfer> pending, published, last;
    int last_kernel = 0;
    uint32_t update_dma_kernel(uint32_t id, Owner&, uint64_t src, Owner&, uint64_t dst, size_t) {
        if (updates++ == fail_update_at) { injected_failure = 1; return 4; }
        // Launch patches mutable DMA packets in the built kd_buf directly.
        published.at(id) = {src, dst}; return 0;
    }
    uint32_t sync_mutable_params() {
        ++syncs;
        if (fail_sync) { injected_failure = 2; return fail_sync; }
        // This API copies mutable kernel cfg packets only. DMA publication is
        // intentionally independent from this scan in the real Launch ABI.
        if (kernel) published_kernel = *kernel;
        return 0;
    }
    uint32_t enqueu_batch(bool) {
        ++submits;
        if (has_dma && active_submission_leases == 0) {
            ++submission_without_lease;
        }
        if (fail_submit) { injected_failure = 3; return fail_submit; }
        last = published; last_kernel = published_kernel; return 0;
    }
};
}
enum class GraphNodeKind { Kernel, Dma };
struct KernelNodeData { size_t kernel_idx = 0; };
struct DmaNodeData {
    enum class Variant { Fixed, MutableSrc, MutableDst };
    Variant variant = Variant::Fixed;
    uint64_t src_addr = 0, dst_addr = 0;
    uint64_t semantic_endpoint_id = 0;
    const uint64_t* live_base = nullptr;
    int64_t live_offset = 0;
    size_t bytes = 16;
    uint8_t channel = 0;
};
struct GraphNode {
    GraphNodeKind kind = GraphNodeKind::Dma;
    KernelNodeData kernel;
    DmaNodeData dma;
    const KernelNodeData& as_kernel() const { return kernel; }
    const DmaNodeData& as_dma() const { return dma; }
};
struct Segment {
    size_t start_idx = 0, end_idx = 0;
    std::vector<uint8_t> core_ids{0,1,2,3,4,5,6,7};
    int kernel = 0;
''' + segment + r'''
};
struct RpuKernelGraph {
    std::vector<GraphNode> nodes_;
    std::vector<Segment> segments_;
    std::vector<std::unique_ptr<rhino_lkn::Kernel_t>> kernels_;
    struct {
        bool fast_replay_skip_sync = false;
        int execution_core_count = 8;
    } runtime_policy_;
    bool op_stream_fully_skipped_ = false;
    bool kernel_params_dirty_this_replay_ = false;
    struct { int prepared_segment_hit_total = 0, prepared_segment_miss_total = 0, hw_batch_submit_total = 0; } last_stats_;
''' + slots + r'''
    int prepares = 0;
    int fail_build_count = 0;
    bool throw_build = false;
    ~RpuKernelGraph() { release_prepared_queues(); }
    void release_prepared_queues() noexcept;
    size_t prepared_queue_slot_index(size_t, bool);
    rhino_lkn::Queue_t* ensure_private_queue(size_t, size_t);
    uint32_t prepare_segment_queue(
            Segment& seg, rhino_lkn::Queue_t& q, bool perf,
            RpuDmaSubmissionLease& submission_lease,
            bool retain_replay_state = true) {
        assert(retain_replay_state);  // This fixture exercises retained replay.
        ++prepares;
        if (throw_build) throw std::runtime_error("SDK failed");
        if (fail_build_count) { --fail_build_count; return 2; }
        seg.mutable_dmas.clear();
        auto table = std::make_shared<Segment::PreparedFixedDmaTable>();
        q.pending.clear(); q.published.clear(); q.perf = perf; q.built = true;
        q.has_dma = seg.start_idx != seg.end_idx;
        q.kernel = &seg.kernel; q.published_kernel = seg.kernel;
        for (size_t i = seg.start_idx; i < seg.end_idx; ++i) {
            if (nodes_.at(i).kind == GraphNodeKind::Kernel) continue;
            const auto& dma = nodes_.at(i).dma;
            if (dma.variant == DmaNodeData::Variant::Fixed) {
                const auto src = rpu_resolve_device_dma_endpoint(
                    dma.src_addr, dma.bytes, "host prepare fixed src");
                const auto dst = rpu_resolve_device_dma_endpoint(
                    dma.dst_addr, dma.bytes, "host prepare fixed dst");
                table->bindings.push_back({i,dma.src_addr,dma.dst_addr,
                                          src.allocation_id,
                                          dst.allocation_id,
                                          dma.bytes,dma.channel});
                if (src.allocation_id) table->witnesses.push_back(
                    {src.allocation_id, dma.src_addr, dma.bytes, "host fixed src"});
                if (dst.allocation_id) table->witnesses.push_back(
                    {dst.allocation_id, dma.dst_addr, dma.bytes, "host fixed dst"});
            } else {
                const bool src = dma.variant == DmaNodeData::Variant::MutableSrc;
                seg.mutable_dmas.push_back({i,static_cast<uint32_t>(q.pending.size()),
                    src ? Segment::PreparedMutableDmaSlot::Kind::MutableSrc : Segment::PreparedMutableDmaSlot::Kind::MutableDst,
                    dma.live_base,dma.live_offset,
                    src ? dma.dst_addr : dma.src_addr,
                    rpu_resolve_device_dma_endpoint(
                        src ? dma.dst_addr : dma.src_addr, dma.bytes,
                        "host prepare mutable fixed side").allocation_id,
                    dma.bytes,0});
                q.pending.push_back({src ? *dma.live_base + dma.live_offset : dma.src_addr,
                                     src ? dma.dst_addr : *dma.live_base + dma.live_offset});
            }
        }
        seg.fixed_dma_table = std::move(table);
        q.published = q.pending;
        submission_lease = RpuDmaSubmissionLease(q.has_dma);
        return 0;
    }
    rhino_lkn::Queue_t* launch_segment_for_replay(Segment&, bool);
    bool capture_segment_kernel_register_tokens(
        const Segment&, std::vector<std::pair<size_t, uint64_t>>&) const;
    uint32_t launch_segment_sync_only(Segment&, PreparedQueueSlot&);
};
''' + resources + replay + r'''
void init(RpuKernelGraph& g, uint64_t& input) {
    g.nodes_.resize(4); g.segments_.resize(2);
    for (size_t i = 0; i < 2; ++i) {
        auto& kn = g.nodes_[i * 2];
        kn.kind = GraphNodeKind::Kernel;
        kn.kernel.kernel_idx = i;
        auto& d = g.nodes_[i * 2 + 1].dma;
        d.variant = DmaNodeData::Variant::MutableSrc;
        d.live_base = &input; d.live_offset = static_cast<int64_t>(i * 16); d.dst_addr = 1000 + i * 16;
        auto& s = g.segments_[i]; s.start_idx = i * 2; s.end_idx = i * 2 + 2; s.kernel = static_cast<int>(10+i);
        g.kernels_.push_back(std::make_unique<rhino_lkn::Kernel_t>(&s.kernel));
    }
}
int main() {
    {
        // Mutable DMA publication is independent from kernel-argument sync. An
        // unchanged kernel token may skip the full scan after the DMA packets
        // are updated and published.
        uint64_t mutable_src = 100, mutable_dst = 200;
        RpuKernelGraph contract;
        contract.nodes_.resize(4); contract.segments_.resize(1);
        contract.segments_[0].end_idx = 4;
        contract.segments_[0].kernel = 7;
        contract.nodes_[0].kind = GraphNodeKind::Kernel;
        contract.nodes_[0].kernel.kernel_idx = 0;
        contract.kernels_.push_back(std::make_unique<rhino_lkn::Kernel_t>(
            &contract.segments_[0].kernel));
        contract.nodes_[1].kind = GraphNodeKind::Kernel;
        contract.nodes_[1].kernel.kernel_idx = 1;
        contract.kernels_.push_back(std::make_unique<rhino_lkn::Kernel_t>(
            &contract.segments_[0].kernel));
        auto& src_dma = contract.nodes_[2].dma;
        src_dma.variant = DmaNodeData::Variant::MutableSrc;
        src_dma.live_base = &mutable_src;
        src_dma.dst_addr = 1000;
        auto& dst_dma = contract.nodes_[3].dma;
        dst_dma.variant = DmaNodeData::Variant::MutableDst;
        dst_dma.live_base = &mutable_dst;
        dst_dma.src_addr = 2000;
        contract.runtime_policy_.fast_replay_skip_sync = true;

        auto* q = contract.launch_segment_for_replay(
            contract.segments_[0], false);
        mutable_src = 300; mutable_dst = 400;
        contract.op_stream_fully_skipped_ = true;
        contract.kernel_params_dirty_this_replay_ = false;
        int before_sync = q->syncs;
        q = contract.launch_segment_for_replay(contract.segments_[0], false);
        assert(q->syncs == before_sync);
        assert(q->last == std::vector<Transfer>(
            {{mutable_src,1000},{2000,mutable_dst}}));
        assert(q->last_kernel == 7);

        // An identical packet may skip the SDK update, never the fresh owner
        // resolution or submission lease. Both DMA variants participate.
        const int stable_updates = q->updates;
        const int stable_resolutions = endpoint_resolve_calls;
        const int stable_submits = q->submits;
        q = contract.launch_segment_for_replay(contract.segments_[0], false);
        assert(q->updates == stable_updates && q->submits == stable_submits + 1);
        assert(endpoint_resolve_calls == stable_resolutions + 4);
        assert(active_submission_leases == 0 && submission_without_lease == 0);

        // SPM endpoints have no allocation ID: equal offsets in a new root
        // owner still require a new typed packet proof.
        Owner alternate_owner;
        endpoint_owners[mutable_src] = &alternate_owner;
        q = contract.launch_segment_for_replay(contract.segments_[0], false);
        assert(q->updates == stable_updates + 1);
        endpoint_owners.erase(mutable_src);
        q = contract.launch_segment_for_replay(contract.segments_[0], false);
        assert(q->updates == stable_updates + 2);

        // A live mutable endpoint may intentionally change allocations at the
        // same offset; its identity must invalidate the typed packet snapshot.
        allocation_ids[mutable_dst] = 501;
        q = contract.launch_segment_for_replay(contract.segments_[0], false);
        assert(q->updates == stable_updates + 3);
        allocation_ids.erase(mutable_dst);
        q = contract.launch_segment_for_replay(contract.segments_[0], false);
        assert(q->updates == stable_updates + 4);

        // The public Kernel_t token catches a successful register mutation even
        // if a caller failed to mark the coarse Graph witness.
        mutable_src = 500; mutable_dst = 600;
        contract.kernels_[0]->write(9);
        contract.op_stream_fully_skipped_ = true;
        contract.kernel_params_dirty_this_replay_ = false;
        before_sync = q->syncs;
        q = contract.launch_segment_for_replay(contract.segments_[0], false);
        assert(q->syncs == before_sync + 1);
        assert(q->last == std::vector<Transfer>(
            {{mutable_src,1000},{2000,mutable_dst}}));
        assert(q->last_kernel == 9);

        // Tokens are compared for every kernel in the segment. A mutation in
        // only the second kernel must invalidate the whole prepared batch.
        contract.kernels_[1]->write(11);
        contract.kernel_params_dirty_this_replay_ = false;
        before_sync = q->syncs;
        q = contract.launch_segment_for_replay(contract.segments_[0], false);
        assert(q->syncs == before_sync + 1);
        assert(q->last_kernel == 11);

        // A failed public query is never treated as an unchanged kernel. The
        // first successful query after recovery also performs a full sync,
        // establishes a new valid snapshot, and only then permits a skip.
        contract.kernels_[0]->query_ok = false;
        contract.kernel_params_dirty_this_replay_ = false;
        before_sync = q->syncs;
        q = contract.launch_segment_for_replay(contract.segments_[0], false);
        assert(q->syncs == before_sync + 1);
        contract.kernels_[0]->query_ok = true;
        before_sync = q->syncs;
        q = contract.launch_segment_for_replay(contract.segments_[0], false);
        assert(q->syncs == before_sync + 1);
        before_sync = q->syncs;
        q = contract.launch_segment_for_replay(contract.segments_[0], false);
        assert(q->syncs == before_sync);

        // The coarse witness remains a conservative backstop even when the
        // fine-grained token is unchanged.
        contract.kernel_params_dirty_this_replay_ = true;
        before_sync = q->syncs;
        q = contract.launch_segment_for_replay(contract.segments_[0], false);
        assert(q->syncs == before_sync + 1);
    }
    assert(g_extra_prepared_queues == 0 && rhino_lkn::Queue_t::live == 0);
    {
        RpuKernelGraph reduced;
        reduced.runtime_policy_.execution_core_count = 4;
        auto* q = reduced.ensure_private_queue(0, 4);
        assert(q->cores == 4);
        for (size_t invalid_cores : {size_t{0}, size_t{6}, size_t{8}}) {
            bool rejected = false;
            try { reduced.ensure_private_queue(0, invalid_cores); }
            catch (const std::runtime_error&) { rejected = true; }
            assert(rejected && reduced.private_queue_slots_[0].queue.get() == q);
        }
    }
    assert(g_extra_prepared_queues == 0 && rhino_lkn::Queue_t::live == 0);
    {
        // A clean segment with fixed DMA retains the optimization: there is no
        // mutable packet publication and no kernel mutation to synchronize.
        RpuKernelGraph fixed;
        fixed.nodes_.resize(1); fixed.segments_.resize(1);
        fixed.segments_[0].end_idx = 1;
        fixed.nodes_[0].dma.variant = DmaNodeData::Variant::Fixed;
        fixed.nodes_[0].dma.src_addr = 3000;
        fixed.nodes_[0].dma.dst_addr = 4000;
        allocation_ids[3000] = 101;
        allocation_ids[4000] = 102;
        fixed.runtime_policy_.fast_replay_skip_sync = true;
        auto* q = fixed.launch_segment_for_replay(fixed.segments_[0], false);
        auto& seg = fixed.segments_[0];
        auto& slot = fixed.private_queue_slots_[0];
        const auto initial_table = seg.fixed_dma_table;
        assert(initial_table && slot.fixed_dma_table == initial_table);
        const int initial_prepares = fixed.prepares;
        int before_sync = q->syncs;
        fixed.kernel_params_dirty_this_replay_ = false;
        q = fixed.launch_segment_for_replay(fixed.segments_[0], false);
        assert(q->syncs == before_sync);
        assert(fixed.prepares == initial_prepares);
        assert(seg.fixed_dma_table == initial_table &&
               slot.fixed_dma_table == initial_table);
        fixed.kernel_params_dirty_this_replay_ = true;
        q = fixed.launch_segment_for_replay(fixed.segments_[0], false);
        assert(q->syncs == before_sync + 1);

        // Equal values from another cold snapshot are not the table from which
        // this Queue_t was built. Pointer reuse cannot alias while either owner
        // still holds its shared reference.
        slot.fixed_dma_table =
            std::make_shared<const Segment::PreparedFixedDmaTable>(*initial_table);
        q = fixed.launch_segment_for_replay(seg, false);
        assert(fixed.prepares == initial_prepares + 1);
        assert(seg.fixed_dma_table != initial_table &&
               slot.fixed_dma_table == seg.fixed_dma_table);
        assert(initial_table->bindings[0].src_addr == 3000);

        // Shared table identity does not excuse actual node topology drift.
        for (int field = 0; field < 3; ++field) {
            const auto saved = fixed.nodes_[0].dma;
            if (field == 0) fixed.nodes_[0].dma.bytes += 16;
            if (field == 1) fixed.nodes_[0].dma.channel = 1;
            if (field == 2)
                fixed.nodes_[0].dma.variant = DmaNodeData::Variant::MutableSrc;
            const int before = fixed.prepares;
            bool rejected = false;
            try { fixed.launch_segment_for_replay(seg, false); }
            catch (const std::runtime_error&) { rejected = true; }
            assert(rejected && fixed.prepares == before);
            fixed.nodes_[0].dma = saved;
        }

        // A build failure preserves Segment's previous snapshot but clears the
        // slot. A submit failure may leave the newly prepared Segment snapshot,
        // yet it must never publish that snapshot as a usable queue batch.
        const auto before_failed_build = seg.fixed_dma_table;
        fixed.fail_build_count = 1;
        bool rejected = false;
        try { fixed.launch_segment_for_replay(seg, true); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && !slot.queue && !slot.fixed_dma_table);
        assert(seg.fixed_dma_table == before_failed_build);
        q = fixed.launch_segment_for_replay(seg, false);
        const auto before_failed_submit = seg.fixed_dma_table;
        q->fail_submit = 6;
        rejected = false;
        try { fixed.launch_segment_for_replay(seg, true); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && !slot.queue && !slot.fixed_dma_table);
        assert(seg.fixed_dma_table != before_failed_submit);
        fixed.launch_segment_for_replay(seg, false);
        assert(slot.fixed_dma_table == seg.fixed_dma_table);

        fixed.ensure_private_queue(0, 4);
        assert(!slot.fixed_dma_table && slot.built_segment_idx == -1);
        fixed.launch_segment_for_replay(seg, false);
        assert(slot.fixed_dma_table == seg.fixed_dma_table);
        fixed.release_prepared_queues();
        assert(!slot.fixed_dma_table && slot.built_segment_idx == -1);
        assert(seg.fixed_dma_table);  // Segment remains the lifetime baseline.
    }
    assert(g_extra_prepared_queues == 0 && rhino_lkn::Queue_t::live == 0);
    {
        uint64_t a = 100, b = 200;
        RpuKernelGraph first, second;
        init(first,a); init(second,b);
        first.runtime_policy_.fast_replay_skip_sync = true;
        second.runtime_policy_.fast_replay_skip_sync = true;
        // Cross-entry alternating replay, multiple segments and mutable values.
        for (int round = 0; round < 32; ++round) {
            a += 128; b += 64;
            for (auto* g : {&first,&second}) {
                g->op_stream_fully_skipped_ = (round % 2 == 0);
                g->kernel_params_dirty_this_replay_ =
                    !g->op_stream_fully_skipped_;
                for (size_t i = 0; i < 2; ++i) {
                    if (!g->op_stream_fully_skipped_) {
                        g->kernels_[i]->write(g->segments_[i].kernel + 3);
                    }
                    auto* q = g->launch_segment_for_replay(g->segments_[i],false);
                    const auto& d = g->nodes_[i * 2 + 1].dma;
                    assert(q->last == std::vector<Transfer>({{*d.live_base + d.live_offset,d.dst_addr}}));
                    assert(q->last_kernel == g->segments_[i].kernel);
                }
            }
        }
        if (kExtraPreparedQueueBudget > 0) {
            assert(first.last_stats_.prepared_segment_hit_total > 40);
            assert(second.last_stats_.prepared_segment_hit_total > 40);
        }
        // A legacy caller may replace both the live-base pointer and fixed side.
        first.nodes_[3].dma.live_base = &b;
        first.nodes_[3].dma.dst_addr = 4048;
        auto* q = first.launch_segment_for_replay(first.segments_[1],false);
        assert(q->last.at(0) == (Transfer{b+16,4048}));
        // Instrumentation belongs to the prepared batch, not to a global flag.
        int before = first.prepares;
        q = first.launch_segment_for_replay(first.segments_[1],true);
        assert(q->perf && first.prepares == before+1);
        q = first.launch_segment_for_replay(first.segments_[1],false);
        assert(!q->perf && first.prepares == before+2);
        // A failed sync cannot leave a reusable half-updated fingerprint.
        const size_t failing_slot = kExtraPreparedQueueBudget > 0 ? 1 : 0;
        q->fail_sync = 3;
        first.kernel_params_dirty_this_replay_ = true;
        bool rejected = false;
        try { first.launch_segment_for_replay(first.segments_[1],false); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
        assert(!first.private_queue_slots_[failing_slot].queue);
        assert(!first.private_queue_slots_[failing_slot].fixed_dma_table);
        before = first.prepares;
        first.launch_segment_for_replay(first.segments_[1],false);
        assert(first.prepares == before+1);
        first.release_prepared_queues(); first.release_prepared_queues();
    }
    assert(g_extra_prepared_queues == 0 && rhino_lkn::Queue_t::live == 0);
    {
        // Every in-place failure discards the exact prepared slot. A legacy
        // fixed-side move remains uncommitted until all updates, the optional
        // kernel sync, and submission have succeeded.
        for (int failure_stage = 0; failure_stage < 3; ++failure_stage) {
            uint64_t live = 11000;
            RpuKernelGraph transactional;
            transactional.nodes_.resize(3);
            transactional.segments_.resize(2);
            auto& seg = transactional.segments_[1];
            seg.start_idx = 0;
            seg.end_idx = 3;
            for (size_t i = 0; i < 3; ++i) {
                auto& dma = transactional.nodes_[i].dma;
                dma.variant = DmaNodeData::Variant::MutableSrc;
                dma.live_base = &live;
                dma.live_offset = static_cast<int64_t>(i * 16);
                dma.dst_addr = 12000 + i * 16;
                allocation_ids[dma.dst_addr] = 70 + i;
            }
            auto* q = transactional.launch_segment_for_replay(seg, false);
            // The standalone harness starts without a recorded cold batch.
            // Establish its table, then release the cold fallback queue before
            // testing isolated retained-slot failure cleanup below.
            transactional.release_prepared_queues();
            q = transactional.launch_segment_for_replay(seg, false);
            const size_t exact_slot =
                kExtraPreparedQueueBudget > 0 ? 1 : 0;
            assert(transactional.private_queue_slots_[exact_slot].queue.get() == q);
            const uint64_t old_fixed = seg.mutable_dmas[0].fixed_addr;
            const uint64_t old_id = seg.mutable_dmas[0].fixed_allocation_id;
            transactional.nodes_[0].dma.dst_addr = 12128;
            allocation_ids[12128] = 99;
            rhino_lkn::Queue_t::injected_failure = 0;
            if (failure_stage == 0) {
                live += 64;  // Force all three packets to change before failure.
                q->fail_update_at = q->updates + 1;  // second DMA update
            } else if (failure_stage == 1) {
                q->fail_sync = 5;
            } else {
                q->fail_submit = 6;
            }
            bool rejected = false;
            try { transactional.launch_segment_for_replay(seg, false); }
            catch (const std::runtime_error&) { rejected = true; }
            assert(rejected);
            assert(rhino_lkn::Queue_t::injected_failure == failure_stage + 1);
            assert(!transactional.private_queue_slots_[exact_slot].queue);
            assert(!transactional.private_queue_slots_[exact_slot].fixed_dma_table);
            assert(transactional.private_queue_slots_[exact_slot].mutable_dma_packets.empty());
            assert(transactional.private_queue_slots_[exact_slot].pending_mutable_dma_packets.empty());
            for (size_t i = 0; i < transactional.private_queue_slots_.size(); ++i) {
                if (i != exact_slot) {
                    assert(!transactional.private_queue_slots_[i].queue);
                }
            }
            assert(seg.mutable_dmas[0].fixed_addr == old_fixed);
            assert(seg.mutable_dmas[0].fixed_allocation_id == old_id);
            q = transactional.launch_segment_for_replay(seg, false);
            assert(q->last.at(0) == (Transfer{live,12128}));
            assert(seg.mutable_dmas[0].fixed_addr == 12128);
            assert(seg.mutable_dmas[0].fixed_allocation_id == 99);
        }
    }
    assert(g_extra_prepared_queues == 0 && rhino_lkn::Queue_t::live == 0);
    {
        // Both semantic mutable variants retain one fixed endpoint.  Reissuing
        // that endpoint's logical allocation at the same address must fail.
        for (bool mutable_src : {true, false}) {
          for (bool force_rebuild : {false, true}) {
            uint64_t live = mutable_src ? 7000 : 7100;
            const uint64_t fixed = (mutable_src ? 8000 : 8100) +
                                   (force_rebuild ? 64 : 0);
            RpuKernelGraph semantic;
            semantic.nodes_.resize(1); semantic.segments_.resize(1);
            semantic.segments_[0].end_idx = 1;
            auto& dma = semantic.nodes_[0].dma;
            dma.variant = mutable_src ? DmaNodeData::Variant::MutableSrc
                                      : DmaNodeData::Variant::MutableDst;
            dma.semantic_endpoint_id = mutable_src ? 1 : 2;
            dma.live_base = &live;
            if (mutable_src) dma.dst_addr = fixed;
            else dma.src_addr = fixed;
            allocation_ids[fixed] = mutable_src ? 21 : 31;
            semantic.launch_segment_for_replay(semantic.segments_[0], false);
            const int before = semantic.prepares;
            allocation_ids[fixed] = mutable_src ? 22 : 32;
            bool aba_rejected = false;
            try {
                semantic.launch_segment_for_replay(
                    semantic.segments_[0], force_rebuild);
            } catch (const std::runtime_error&) { aba_rejected = true; }
            assert(aba_rejected && semantic.prepares == before);
          }
        }
    }
    {
        uint64_t live = 7200;
        RpuKernelGraph legacy;
        legacy.nodes_.resize(1); legacy.segments_.resize(1);
        legacy.segments_[0].end_idx = 1;
        auto& dma = legacy.nodes_[0].dma;
        dma.variant = DmaNodeData::Variant::MutableSrc;
        dma.live_base = &live;
        dma.dst_addr = 8200;
        allocation_ids[8200] = 41;
        auto* q = legacy.launch_segment_for_replay(
            legacy.segments_[0], false);
        const int before = legacy.prepares;

        // Legacy mutable DMA still permits an intentional fixed-address move.
        dma.dst_addr = 8232;
        allocation_ids[8232] = 42;
        q = legacy.launch_segment_for_replay(legacy.segments_[0], false);
        assert(legacy.prepares == before);
        assert(q->last.at(0) == (Transfer{live,8232}));

        // The moved address becomes the identity baseline for later replays.
        allocation_ids[8232] = 43;
        bool aba_rejected = false;
        try {
            legacy.launch_segment_for_replay(legacy.segments_[0], false);
        } catch (const std::runtime_error&) { aba_rejected = true; }
        assert(aba_rejected && legacy.prepares == before);
    }
    {
        // A legal address drift in an earlier fixed binding must not stop the
        // scan before a later same-address allocation ABA is checked.
        RpuKernelGraph mixed_fixed;
        mixed_fixed.nodes_.resize(2); mixed_fixed.segments_.resize(1);
        mixed_fixed.segments_[0].end_idx = 2;
        mixed_fixed.nodes_[0].dma.src_addr = 9000;
        mixed_fixed.nodes_[0].dma.dst_addr = 9016;
        mixed_fixed.nodes_[1].dma.src_addr = 9032;
        mixed_fixed.nodes_[1].dma.dst_addr = 9048;
        allocation_ids[9000] = 51;
        allocation_ids[9016] = 52;
        allocation_ids[9032] = 53;
        allocation_ids[9048] = 54;
        mixed_fixed.launch_segment_for_replay(
            mixed_fixed.segments_[0], false);
        const int before = mixed_fixed.prepares;
        mixed_fixed.nodes_[0].dma.src_addr = 9064;
        allocation_ids[9064] = 55;
        allocation_ids[9032] = 56;
        bool aba_rejected = false;
        try {
            mixed_fixed.launch_segment_for_replay(
                mixed_fixed.segments_[0], false);
        } catch (const std::runtime_error&) { aba_rejected = true; }
        assert(aba_rejected && mixed_fixed.prepares == before);
    }
    {
        RpuKernelGraph fixed;
        fixed.nodes_.resize(1); fixed.segments_.resize(1);
        fixed.segments_[0].end_idx = 1;
        fixed.nodes_[0].dma.src_addr = 64; fixed.nodes_[0].dma.dst_addr = 128;
        allocation_ids[64] = 11;
        fixed.launch_segment_for_replay(fixed.segments_[0],false);
        int before = fixed.prepares;
        // Reissuing a logical allocation at the exact same device address must
        // not hit or rebuild a Queue_t that was prepared for the old Storage.
        allocation_ids[64] = 12;
        bool aba_rejected = false;
        try { fixed.launch_segment_for_replay(fixed.segments_[0],false); }
        catch (const std::runtime_error&) { aba_rejected = true; }
        assert(aba_rejected && fixed.prepares == before);

        // Ordinary fixed-address drift retains the existing rebuild contract.
        fixed.nodes_[0].dma.src_addr = 256;
        allocation_ids[256] = 13;
        fixed.launch_segment_for_replay(fixed.segments_[0],false);
        assert(fixed.prepares == before+1);
        fixed.runtime_policy_.fast_replay_skip_sync = true;
        fixed.op_stream_fully_skipped_ = true;
        fixed.kernel_params_dirty_this_replay_ = false;
        auto* q = fixed.launch_segment_for_replay(fixed.segments_[0],false);
        assert(q->syncs == 0);  // Mutable-DMA-free fast replay remains fast.
    }
    if (kExtraPreparedQueueBudget > 0) {
        uint64_t input = 512;
        RpuKernelGraph limited;
        init(limited,input);
        limited.launch_segment_for_replay(limited.segments_[1],false);
        const int before = limited.prepares;
        limited.fail_build_count = 1;
        // The optional arena failure falls back, releases its token, and does
        // not let the entry claim an unbuilt batch as a prepared hit.
        limited.launch_segment_for_replay(limited.segments_[1],false);
        assert(limited.prepares == before + 2);
        assert(limited.private_queue_slots_[1].promotion_denied);
        assert(g_extra_prepared_queues == 0);
        std::vector<std::unique_ptr<RpuKernelGraph>> many;
        for (int i = 0; i < 12; ++i) {
            auto g = std::make_unique<RpuKernelGraph>(); init(*g,input);
            g->launch_segment_for_replay(g->segments_[1],false);
            g->launch_segment_for_replay(g->segments_[1],false);
            many.push_back(std::move(g));
        }
        assert(g_extra_prepared_queues == kExtraPreparedQueueBudget);
        many.clear();
        assert(g_extra_prepared_queues == 0);
    }
    assert(rhino_lkn::Queue_t::live == 0);
    assert(active_submission_leases == 0);
    assert(submission_without_lease == 0);
    std::cout << "prepared queues: address freshness, entry isolation, bounded ownership and failure recovery passed\n";
}
'''
    cpp = tmp_path / "prepared_queues.cpp"
    exe = tmp_path / "prepared_queues"
    cpp.write_text(harness)
    subprocess.run([compiler, "-std=c++17", "-O0", "-g", "-Wall", "-Wextra", str(cpp), "-o", str(exe)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert "failure recovery passed" in result.stdout


def test_queue_ownership_and_oneshot_lifecycle():
    source = EXECUTE.read_text()
    runtime = (ROOT / "src/graph/graph_runtime.cpp").read_text()
    assert "kExtraPreparedQueueBudget = 0;" in source
    # A recorded occurrence owns its Kernel_t registers; only Program_t shares
    # read-only program data, so later entries cannot mutate its register array.
    recording = _between(source, "case State::RECORDING: {", "case State::REPLAYING: {")
    assert "std::make_unique<::rhino_lkn::Kernel_t>" in recording
    assert "kernels_.push_back(std::move(k))" in recording
    destructor = _between(runtime, "RpuKernelGraph::~RpuKernelGraph()", "\n}")
    assert "release_prepared_queues();" in destructor
    invalidate = _between(runtime, "void RpuKernelGraph::invalidate_impl", "    nodes_.clear();")
    assert invalidate.index("release_prepared_queues();") < invalidate.index("kernels_.clear();")
    oneshot = source[source.index("void RpuKernelGraph::execute_graph_oneshot()") :]
    assert "QueueCache::instance()" not in oneshot
    assert "std::unique_ptr<::rhino_lkn::Queue_t> one_shot_queue" in oneshot
    assert "/*retain_replay_state=*/false" in oneshot
    assert oneshot.index("wq->discard_completed_batch()") < oneshot.index("++seg_idx_oneshot")


def test_public_hwperf_metadata_writer(tmp_path):
    """Host writer needs no private Launch header and cannot clobber a file."""
    import json
    import stat

    source = EXECUTE.read_text()
    assert '"r1/' not in source and "rhino_lkn::detail::" not in source
    helpers = _between(source, "std::string build_hw_perf_dump_path(", "// Publish host companion metadata")
    cpp = tmp_path / "companion.cpp"
    binary = tmp_path / "companion"
    cpp.write_text(r'''
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <sstream>
#include <string>
#include <unistd.h>
''' + helpers + r'''
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string root(argv[1]);
    const std::string label = "../../escaped/quote\"\n\\\x01";
    auto path = build_hw_perf_dump_path(root,label,false,2,3);
    assert(std::filesystem::path(path).parent_path() == root);
    FILE* file = open_graph_companion_file(path);
    assert(file);
    std::fputc('"',file); write_graph_json_string(file,label); std::fputc('"',file);
    assert(std::fclose(file) == 0);
    assert(open_graph_companion_file(path) == nullptr);
    const std::string link = root + "/symlink";
    assert(::symlink(path.c_str(),link.c_str()) == 0);
    assert(open_graph_companion_file(link) == nullptr);
    std::puts(path.c_str());
}
''')
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    subprocess.run([compiler, "-std=c++17", str(cpp), "-o", str(binary)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary), str(tmp_path)], check=True, capture_output=True, text=True)
    output = Path(result.stdout.strip())
    assert output.parent == tmp_path
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text()) == '../../escaped/quote"\n\\\x01'
