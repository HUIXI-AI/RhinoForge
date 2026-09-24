"""Run production segment/queue methods with a small host resource recorder."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _function(source, signature):
    start = source.index(signature)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def queue_probe(tmp_path_factory):
    compiler = shutil.which("g++")
    assert compiler, "native queue checks require g++"
    execute = (ROOT / "src/graph/graph_runtime_execute.cpp").read_text()
    internal = (ROOT / "src/graph/graph_runtime_internal.h").read_text()
    functions = _function(internal, "void validate_graph_dma_channel(") + "\n"
    functions += "\n".join(_function(execute, signature) for signature in (
        "uint8_t validated_kernel_core_extent(",
        "static size_t segment_new_completion_bytes(",
        "static bool segment_would_exceed(",
        "void RpuKernelGraph::build_segments_from_nodes(",
    ))
    sdk_get = _function((ROOT / "src/core/rpu_kernel_cache.h").read_text(),
                        "inline ::rhino_lkn::Queue_t* get(")
    proxy_get = _function((ROOT / "src/graph/graph_runtime.h").read_text(),
                          "RpuQueue* get(")
    harness = r'''
#include <algorithm>
#include <array>
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
#include <variant>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
namespace rhino_lkn {
struct Queue_t {
    inline static int creations = 0;
    int count;
    explicit Queue_t(int n) : count(n) { ++creations; }
};
}
struct QueueCache {
    std::array<rhino_lkn::Queue_t*, 8> queues_{};
    ~QueueCache() { for (auto* q : queues_) delete q; }
    __SDK_GET__
};
struct RpuQueue { int count; explicit RpuQueue(int n) : count(n) {} };
struct RpuQueueCache {
    std::array<std::unique_ptr<RpuQueue>, 8> queues_;
    RpuQueueCache() { for (int i=0; i<8; ++i) queues_[i]=std::make_unique<RpuQueue>(i+1); }
    __PROXY_GET__
};
struct GraphRuntimePolicy {
    bool siglip_isolate_patch_embed = false;
    int execution_core_count = 8;
};
struct GraphSignature { std::string op_id_str; };
struct QueueLaunchState {
    bool broadcast_mode = true, flush_icache = false;
    bool operator!=(const QueueLaunchState& rhs) const {
        return broadcast_mode != rhs.broadcast_mode || flush_icache != rhs.flush_icache;
    }
};
enum class GraphNodeKind { Kernel, Dma, Barrier, ChildGraph, Memcpy, Branch };
enum class CopyKind { HOST_MEMCPY, DDR_TO_DDR };
struct MemcpyNodeData { CopyKind kind=CopyKind::HOST_MEMCPY; int dma_channel=-1; };
struct RpuKernelGraph;
struct GraphNode {
    GraphNodeKind kind = GraphNodeKind::Kernel;
    std::vector<uint8_t> core_ids;
    QueueLaunchState queue_state;
    uint8_t channel = 0, self_stream = 0, target_stream = 0;
    bool isolate = false;
    std::shared_ptr<RpuKernelGraph> child;
    std::variant<MemcpyNodeData> data;
    GraphNode& as_kernel() { return *this; }
    const GraphNode& as_kernel() const { return *this; }
    const GraphNode& as_dma() const { return *this; }
    const GraphNode& as_barrier_node() const { return *this; }
    const GraphNode& as_child_graph() const { return *this; }
    const GraphNode& as_branch() const { return *this; }
    bool is_physical_manifest_metadata() const { return false; }
};
struct Census {
    size_t segment_id=0, start_idx=0, end_idx=0, core_count=0;
    size_t kernel_count=0, dma_count=0, barrier_count=0;
};
struct Segment {
    size_t segment_id=0, start_idx=0, end_idx=0;
    std::vector<uint8_t> core_ids;
    QueueLaunchState queue_state;
    Census census;
};
struct SegmentResourceUse { size_t entries=0, kd_bytes=0, instr_bytes=0; uint16_t active_streams=0; };
using SegmentResourceBudget = SegmentResourceUse;
SegmentResourceBudget segment_resource_budget(const std::string&, const GraphRuntimePolicy&) {
    return {1000, 1000, 1000};
}
SegmentResourceUse segment_node_resources(const GraphNode&, const std::vector<int>&) { return {1,1,1}; }
bool p7_should_force_isolate_kernel(const GraphNode& node) { return node.isolate; }
bool siglip_should_isolate_patch_embed_kernel(const std::string&, const GraphNode&, bool) { return false; }
std::string kernel_label_for_segment_split(const GraphNode&) { return "test"; }
bool log_at(int) { return false; }
struct RpuKernelGraph {
    GraphRuntimePolicy runtime_policy_;
    std::vector<GraphNode> nodes_;
    std::vector<Segment> segments_;
    std::vector<int> kernels_;
    std::optional<GraphSignature> pending_signature_;
    bool has_built_signature_ = false;
    GraphSignature built_signature_;
    uint64_t graph_lifetime_id_ = 1;
    struct {
        size_t dma_count=0, barrier_count=0, child_graph_count=0;
        std::vector<uint64_t> child_graph_lifetime_ids;
        std::vector<Census> segment_census;
    } last_stats_;
    void build_segments_from_nodes();
};
__FUNCTIONS__
GraphNode kernel(std::initializer_list<uint8_t> cores, bool isolate=false) {
    GraphNode node; node.core_ids=cores; node.isolate=isolate; return node;
}
GraphNode dma(uint8_t channel) { GraphNode node; node.kind=GraphNodeKind::Dma; node.channel=channel; return node; }
template<class F> void rejects(F&& f) {
    bool failed=false; try { f(); } catch (const std::runtime_error&) { failed=true; } assert(failed);
}
int main(int argc, char** argv) {
    assert(argc==2);
    if (std::string(argv[1]) == "segments") {
        RpuKernelGraph graph;
        auto check = [&](std::vector<GraphNode> nodes, size_t count) {
            graph.nodes_=std::move(nodes); graph.build_segments_from_nodes();
            assert(graph.segments_.size()==1);
            const auto& segment=graph.segments_[0];
            assert(segment.core_ids.size()==count && segment.census.core_count==count);
            for (size_t i=0; i<count; ++i) assert(segment.core_ids[i]==i);
        };
        check({kernel({0,1,2,3}), dma(5)}, 6);
        check({kernel({0,1,2,3}), dma(3)}, 4);
        check({kernel({4,5})}, 6);
        check({kernel({4,5}, true)}, 6);
        check({dma(0), kernel({0})}, 8); // Existing DMA-first behavior remains.
        check({kernel({0,1,2,3,4,5,6,7}), dma(7)}, 8);
        for (auto bad : {kernel({}), kernel({8}), kernel({1,0}), kernel({0,0}), kernel({0,2}), dma(8)}) {
            graph.nodes_={kernel({0}, true), bad};
            rejects([&] { graph.build_segments_from_nodes(); });
            assert(graph.segments_.empty()); // No partial segment publication.
        }
        graph.runtime_policy_.execution_core_count = 4;
        check({dma(0), kernel({0})}, 4);
        check({kernel({0,1}), dma(3)}, 4);
        GraphNode copy; copy.kind=GraphNodeKind::Memcpy;
        copy.data=MemcpyNodeData{CopyKind::DDR_TO_DDR,4};
        GraphNode child; child.kind=GraphNodeKind::ChildGraph;
        child.child=std::make_shared<RpuKernelGraph>();
        child.child->runtime_policy_.execution_core_count=6;
        for (auto bad : {kernel({4}), dma(4), copy, child}) {
            graph.nodes_={kernel({0}, true), bad};
            rejects([&] { graph.build_segments_from_nodes(); });
            assert(graph.segments_.empty());
        }
    } else {
        QueueCache sdk; RpuQueueCache proxy;
        const int before=rhino_lkn::Queue_t::creations;
        for (int64_t count : {0,-1,9,256,260}) {
            rejects([&] { sdk.get(count); }); rejects([&] { proxy.get(count); });
        }
        assert(rhino_lkn::Queue_t::creations==before);
        for (int count : {1,8}) {
            bool fresh=false;
            auto* q=sdk.get(count, &fresh); assert(fresh && q->count==count);
            assert(sdk.get(count, &fresh)==q && !fresh && proxy.get(count)->count==count);
        }
    }
}
'''
    for marker, code in (("__FUNCTIONS__", functions), ("__SDK_GET__", sdk_get),
                         ("__PROXY_GET__", proxy_get)):
        harness = harness.replace(marker, code)
    source = tmp_path_factory.mktemp("queue-bounds") / "probe.cpp"
    source.write_text(harness)
    binary = source.with_suffix("")
    result = subprocess.run([compiler, "-std=c++17", "-O0", str(source), "-o", str(binary)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return binary


@pytest.mark.parametrize("mode", ["segments", "queue_bounds"])
def test_native_queue_resources(queue_probe, mode):
    subprocess.run([str(queue_probe), mode], check=True, capture_output=True, text=True)
