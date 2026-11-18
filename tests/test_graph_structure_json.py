"""Compile the production formatter on host-only Graph fixtures."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_graph_json_preserves_topology_without_addresses(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    stub = r'''
#include <cstdint>
#include <optional>
#include <memory>
#include <string>
#include <vector>
#include <variant>
enum class KernelId { TEST, _COUNT };
const char* KERNEL_ID_NAMES[] = {"test_kernel"};
enum class GraphNodeKind { Kernel, Memcpy, Memset, ChildGraph, Branch, HostCallback, Tier3Oneshot, Dma, Barrier };
enum class CopyKind { HOST_MEMCPY, DDR_TO_DDR, DDR_TO_SPM, SPM_TO_DDR, SPM_TO_SPM };
enum class HostCallbackTier { Tier1, Tier2, Tier3, Reject };
struct FakeOp {
    struct Name { std::string name = "aten::add"; };
    struct Schema { Name operator_name() const { return {}; } };
    Schema schema() const { return {}; }
};
struct QueueState { bool broadcast_mode = true, flush_icache = false; };
struct NodeData {
    enum class Variant { Fixed, MutableSrc, MutableDst };
    size_t kernel_idx = 0;
    std::optional<KernelId> kernel_id;
    std::string kernel_name = "quote\"line\n\\tab\t";
    std::vector<uint8_t> core_ids{0,1};
    std::vector<uint16_t> grid_dims{2,1,1};
    QueueState queue_state;
    CopyKind kind = CopyKind::HOST_MEMCPY;
    uint64_t bytes = 64, value = 7, branch_key = 123;
    HostCallbackTier tier = HostCallbackTier::Tier1;
    std::optional<FakeOp> op_handle = FakeOp{};
    std::vector<int> output_pool{1,2};
    std::vector<uint64_t> output_pool_bytes{16,32};
    std::vector<int64_t> input_dep_node_ids{1}, output_dep_node_ids{2}, transient_resource_ids{3};
    Variant variant = Variant::MutableSrc;
    int channel = 1, self_stream = 2, target_stream = 3;
    uint64_t semantic_endpoint_id = 99, semantic_spm_peer_id = 1;
    bool semantic_canonical_member_emission = true;
};
using KernelNodeData = NodeData;
using MemcpyNodeData = NodeData;
using MemsetNodeData = NodeData;
using HostCallbackNodeData = NodeData;
using Tier3OneshotNodeData = NodeData;
using DmaNodeData = NodeData;
struct GraphNode {
    GraphNodeKind kind;
    std::variant<NodeData> data{NodeData{}};
    const NodeData& as_kernel() const { return std::get<NodeData>(data); }
    const NodeData& as_memcpy() const { return as_kernel(); }
    const NodeData& as_branch() const { return as_kernel(); }
    const NodeData& as_host_callback() const { return as_kernel(); }
    const NodeData& as_tier3_oneshot() const { return as_kernel(); }
    const NodeData& as_dma() const { return as_kernel(); }
    const NodeData& as_barrier_node() const { return as_kernel(); }
};
struct Segment {
    int64_t segment_id;
    size_t start_idx, end_idx;
    std::vector<uint8_t> core_ids{0,1};
    QueueState queue_state;
    uint64_t replay_count = 4, signature_layer_hash = 1234;
};
enum class ResourceKind { StableOutput, ZeroCopyStorageHolder, TransientTier3Buffer };
enum class LifetimeState { Active, Released };
const char* resource_kind_name(ResourceKind value) {
    switch (value) {
    case ResourceKind::StableOutput: return "StableOutput";
    case ResourceKind::ZeroCopyStorageHolder: return "ZeroCopyStorageHolder";
    case ResourceKind::TransientTier3Buffer: return "TransientTier3Buffer";
    }
    return "Unknown";
}
const char* lifetime_state_name(LifetimeState value) {
    return value == LifetimeState::Active ? "Active" : "Released";
}
constexpr int64_t kInvalidPoolSlotId = -1;
struct GraphOwnedResource {
    ResourceKind kind;
    int64_t owner_node_id, nbytes;
    LifetimeState state;
    int64_t pool_slot_id = kInvalidPoolSlotId;
    uint64_t allocator_block_id = 0;
    const void* ptr = reinterpret_cast<void*>(0x12345678);
    uint64_t dev_addr = 0x87654321;
};
namespace rhino_lkn {
struct KernelLaunchFootprint {
    uint64_t command_reservation_bytes = 0, code_reservation_bytes = 0;
};
struct Kernel {
    uint32_t query_launch_footprint(KernelLaunchFootprint* result) const {
        result->command_reservation_bytes = 256;
        result->code_reservation_bytes = 512;
        return 0;
    }
};
}
struct RpuKernelGraph {
    enum class State { PASSTHROUGH, RECORDING, BUILT, REPLAYING };
    State state_ = State::BUILT;
    bool replayable_ = true;
    bool has_built_signature_ = true;
    std::string non_replayable_reason_ = "copy owner 0x12345678 changed\n retry";
    std::vector<std::unique_ptr<rhino_lkn::Kernel>> kernels_;
    std::vector<GraphNode> nodes_;
    std::vector<Segment> segments_;
    std::vector<uint64_t> boundary_flush_ptrs_;
    std::vector<GraphOwnedResource> host_callback_persistent_owners_;
    std::vector<GraphOwnedResource> host_callback_zero_copy_holders_;
    std::vector<GraphOwnedResource> tier3_oneshot_transient_owners_;
    std::string dump_resources() const;
    std::string dump_tree() const;
    std::string dump_replay_plan() const;
    std::string dump_nodes_jsonl() const;
    std::string dump_kernel_args_jsonl() const;
    std::string dump_segments_jsonl() const;
};
'''
    production = (ROOT / "src/graph/graph_runtime_dump.cpp").read_text()
    production = production.replace('#include "graph/graph_runtime.h"', stub)
    program = production + r'''
#include <iostream>
int main() {
    RpuKernelGraph graph;
    graph.kernels_.push_back(std::make_unique<rhino_lkn::Kernel>());
    for (int kind = 0; kind != 9; ++kind)
        graph.nodes_.push_back({static_cast<GraphNodeKind>(kind)});
    graph.segments_.push_back({0, 0, 2, {0,1}, {true, false}, 4, 1234});
    graph.segments_.push_back({1, 7, 9, {0,1}, {true, false}, 4, 1234});
    graph.host_callback_persistent_owners_.push_back({ResourceKind::StableOutput, 5, 64, LifetimeState::Active, 2, 9});
    graph.host_callback_zero_copy_holders_.push_back({ResourceKind::ZeroCopyStorageHolder, 6, 128, LifetimeState::Active});
    graph.tier3_oneshot_transient_owners_.push_back({ResourceKind::TransientTier3Buffer, 7, 32, LifetimeState::Released});
    std::cout << graph.dump_nodes_jsonl() << "SEGMENTS\n" << graph.dump_segments_jsonl() << "RESOURCES\n" << graph.dump_resources() << "CAPACITY\n" << graph.dump_kernel_args_jsonl() << "PLAN\n" << graph.dump_replay_plan();
}
'''
    source = tmp_path / "formatter.cpp"
    source.write_text(program)
    binary = tmp_path / "formatter"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    node_text, tail = result.stdout.split("SEGMENTS\n")
    segment_text, resource_text = tail.split("RESOURCES\n")
    resource_text, tail = resource_text.split("CAPACITY\n")
    capacity_text, plan_text = tail.split("PLAN\n")
    capacities = [json.loads(line) for line in capacity_text.splitlines()]
    assert capacities == [{"node_id": 0, "kernel_idx": 0,
        "kernel": 'quote"line\n\\tab\t', "raw_state": "unavailable_in_release",
        "query_status": 0, "command_reservation_bytes": 256,
        "code_reservation_bytes": 512}]
    assert "has_built_sig=true" in plan_text
    assert 'non_replayable_reason="copy owner <address> changed\\n retry"' in plan_text
    assert "12345678" not in plan_text
    assert "state=BUILT total=3 StableOutput=1 ZeroCopyStorageHolder=1 TransientTier3Buffer=1" in resource_text
    assert "StableOutput owner=5 nbytes=64 state=Active pool_slot=2 block_id=9" in resource_text
    assert "ZeroCopyStorageHolder owner=6 nbytes=128 state=Active\n" in resource_text
    assert "TransientTier3Buffer owner=7 nbytes=32 state=Released\n" in resource_text
    assert all(value not in resource_text for value in ("ptr=", "dev=", "0x", "12345678", "87654321"))
    nodes = [json.loads(line) for line in node_text.splitlines()]
    segments = [json.loads(line) for line in segment_text.splitlines()]
    assert [n["node_id"] for n in nodes] == list(range(9))
    assert [n["segment_id"] for n in nodes] == [0,0,-1,-1,-1,-1,-1,1,1]
    assert nodes[0]["kernel"] == 'quote"line\n\\tab\t'
    assert nodes[0]["cores"] == [0,1]
    assert nodes[1]["copy_kind"] == "host_memcpy"
    assert nodes[5]["op"] == "aten::add"
    assert nodes[5]["output_bytes"] == [16,32]
    assert nodes[6]["input_dep_node_ids"] == [1]
    assert nodes[7]["variant"] == "msrc"
    assert nodes[7]["semantic_endpoint_id"] == 99
    assert nodes[7]["semantic_spm_peer_bound"] is True
    assert segments[0]["node_indices"] == [0,1]
    assert segments[1]["node_indices"] == [7,8]
    assert segments[1]["replay_count"] == 4
    for row in nodes + segments:
        assert not any(part in key for key in row for part in ("addr", "register", "args", "instr", "live_base", "stable_devs"))
        assert "label" not in row


def test_graph_diagnostic_categories_cover_mixed_operators(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    import re
    source = (ROOT / "src/graph/graph_runtime_execute.cpp").read_text()
    start = source.index("const char* kernel_id_short_name(")
    end = source.index("\n}", start) + 2
    function = source[start:end]
    expected = {
        "RMS_NORM_BF16IN_FP16OUT_V32": "RMSNORM_BF16IN",
        "LLM_ALL_REDUCE_BF16_PARTIAL_FP16_RESIDUAL": "ALLREDUCE_MIXED_PACED",
        "LLM_ALL_REDUCE_FP16_PARTIAL_BF16_RESIDUAL": "ALLREDUCE_MIXED_PACED",
        "LLM_ALL_REDUCE_BF16_PARTIAL_FP16_RESIDUAL_NOPACE": "ALLREDUCE_MIXED_NOPACE",
        "LLM_ALL_REDUCE_FP16_PARTIAL_BF16_RESIDUAL_NOPACE": "ALLREDUCE_MIXED_NOPACE",
        "LLM_ALL_REDUCE_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_SPM": "ALLREDUCE_MIXED_LOCAL_SPM",
    }
    names = sorted((set(re.findall(r"KernelId::(\w+)", function)) | expected.keys()) - {"PL_AUTOTILE_BEGIN"})
    code = (
        '#include "rpu_linear_tiling.h"\n#include <iostream>\n'
        "enum class KernelId {" + ",".join(names) + ",PL_AUTOTILE_BEGIN=1000};\n" + function
    )
    code += (
        "\nint main() {\n"
        + "\n".join(
            f'std::cout << kernel_id_short_name(KernelId::{name}) << "\\n";'
            for name in expected
        )
        + "\nfor (size_t i = 0; i < rpu_pl_tiling::kParallelCandidateCount; ++i) {"
        + "\n  const std::string want = rpu_pl_tiling::detail::parallel_candidates[i].acc32 ? \"LINEAR_ACC32\" : \"LINEAR_ACC16\";"
        + "\n  if (want != kernel_id_short_name(static_cast<KernelId>(1000 + i))) return 1;"
        + "\n}\n}\n"
    )
    path = tmp_path / "categories.cpp"
    path.write_text(code)
    binary = tmp_path / "categories"
    subprocess.run([compiler, "-std=c++17", "-I", str(ROOT / "src/ops"), str(path), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == list(expected.values())
