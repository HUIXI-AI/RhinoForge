// Read-only public Graph structure; no addresses, register words or payloads.
#include "graph/graph_runtime.h"
#include <sstream>

namespace {

const char* graph_state_name(RpuKernelGraph::State state) {
    switch (state) {
    case RpuKernelGraph::State::PASSTHROUGH: return "PASSTHROUGH";
    case RpuKernelGraph::State::RECORDING: return "RECORDING";
    case RpuKernelGraph::State::BUILT: return "BUILT";
    case RpuKernelGraph::State::REPLAYING: return "REPLAYING";
    }
    return "UNKNOWN";
}

const char* graph_node_kind_name(GraphNodeKind kind) {
    switch (kind) {
    case GraphNodeKind::Kernel: return "Kernel";
    case GraphNodeKind::Memcpy: return "Memcpy";
    case GraphNodeKind::Memset: return "Memset";
    case GraphNodeKind::ChildGraph: return "ChildGraph";
    case GraphNodeKind::Branch: return "Branch";
    case GraphNodeKind::HostCallback: return "HostCallback";
    case GraphNodeKind::Tier3Oneshot: return "Tier3Oneshot";
    case GraphNodeKind::Dma: return "Dma";
    case GraphNodeKind::Barrier: return "Barrier";
    }
    return "Unknown";
}

const char* copy_kind_name(CopyKind kind) {
    switch (kind) {
    case CopyKind::HOST_MEMCPY: return "host";
    case CopyKind::DDR_TO_DDR: return "ddr_to_ddr";
    case CopyKind::DDR_TO_SPM: return "ddr_to_spm";
    case CopyKind::SPM_TO_DDR: return "spm_to_ddr";
    case CopyKind::SPM_TO_SPM: return "spm_to_spm";
    }
    return "unknown";
}

const char* dma_variant_name(DmaNodeData::Variant variant) {
    switch (variant) {
    case DmaNodeData::Variant::Fixed: return "fixed";
    case DmaNodeData::Variant::MutableSrc: return "mutable_src";
    case DmaNodeData::Variant::MutableDst: return "mutable_dst";
    }
    return "unknown";
}

template <typename T>
void append_values(std::ostringstream& out, const std::vector<T>& values) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ",";
        out << static_cast<uint64_t>(values[i]);
    }
    out << "]";
}

std::string graph_kernel_name(const KernelNodeData& kernel) {
    if (kernel.kernel_id.has_value()) {
        const size_t id = static_cast<size_t>(*kernel.kernel_id);
        if (id < static_cast<size_t>(KernelId::_COUNT)) {
            return KERNEL_ID_NAMES[id];
        }
        return "kernel_id#" + std::to_string(id);
    }
    return kernel.kernel_name.empty() ? "<unnamed>" : kernel.kernel_name;
}

std::string safe_node_summary(
        size_t node_id, const GraphNode& node, int64_t segment_id) {
    std::ostringstream out;
    out << "node[" << node_id << "] kind=" << graph_node_kind_name(node.kind)
        << " segment=" << segment_id;
    switch (node.kind) {
    case GraphNodeKind::Kernel: {
        const auto& kernel = node.as_kernel();
        out << " kernel=" << graph_kernel_name(kernel) << " cores=";
        append_values(out, kernel.core_ids);
        out << " grid=";
        append_values(out, kernel.grid_dims);
        break;
    }
    case GraphNodeKind::Memcpy: {
        const auto& copy = node.as_memcpy();
        out << " copy=" << copy_kind_name(copy.kind)
            << " bytes=" << copy.bytes;
        break;
    }
    case GraphNodeKind::Memset: {
        const auto& fill = std::get<MemsetNodeData>(node.data);
        out << " bytes=" << fill.bytes
            << " value=" << static_cast<uint32_t>(fill.value);
        break;
    }
    case GraphNodeKind::Branch:
        out << " branch_key=0x" << std::hex << node.as_branch().branch_key;
        break;
    case GraphNodeKind::HostCallback: {
        const auto& callback = node.as_host_callback();
        out << " tier=" << static_cast<uint32_t>(callback.tier)
            << " outputs=" << callback.output_pool.size();
        break;
    }
    case GraphNodeKind::Tier3Oneshot: {
        const auto& oneshot = node.as_tier3_oneshot();
        out << " input_deps=" << oneshot.input_dep_node_ids.size()
            << " output_deps=" << oneshot.output_dep_node_ids.size();
        break;
    }
    case GraphNodeKind::Dma: {
        const auto& dma = node.as_dma();
        out << " variant=" << dma_variant_name(dma.variant)
            << " bytes=" << dma.bytes
            << " channel=" << static_cast<uint32_t>(dma.channel);
        break;
    }
    case GraphNodeKind::Barrier: {
        const auto& barrier = node.as_barrier_node();
        out << " self_stream=" << static_cast<uint32_t>(barrier.self_stream)
            << " target_stream="
            << static_cast<uint32_t>(barrier.target_stream);
        break;
    }
    case GraphNodeKind::ChildGraph:
        break;
    }
    return out.str();
}

std::vector<int64_t> graph_segment_ids(
        size_t node_count, const std::vector<Segment>& segments) {
    std::vector<int64_t> result(node_count, -1);
    for (const auto& segment : segments) {
        for (size_t i = segment.start_idx; i < segment.end_idx; ++i) {
            result[i] = static_cast<int64_t>(segment.segment_id);
        }
    }
    return result;
}

void append_segment_summary(std::ostringstream& out, const Segment& segment) {
    out << "segment[" << segment.segment_id << "] nodes=["
        << segment.start_idx << "," << segment.end_idx << ") cores=";
    append_values(out, segment.core_ids);
    out << " broadcast=" << (segment.queue_state.broadcast_mode ? 1 : 0)
        << " flush_icache=" << (segment.queue_state.flush_icache ? 1 : 0)
        << " replay_count=" << segment.replay_count;
}

}  // namespace

std::string RpuKernelGraph::dump_replay_plan() const {
    std::ostringstream out;
    out << "GraphReplayPlan state=" << graph_state_name(state_)
        << " replayable=" << (replayable_ ? "true" : "false")
        << " graph_size=" << nodes_.size()
        << " segments=" << segments_.size()
        << " boundary_flush_count=" << boundary_flush_ptrs_.size() << "\n";
    for (const auto& segment : segments_) {
        out << "  ";
        append_segment_summary(out, segment);
        out << "\n";
    }
    const auto segment_ids = graph_segment_ids(nodes_.size(), segments_);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        out << "  " << safe_node_summary(i, nodes_[i], segment_ids[i]) << "\n";
    }
    return out.str();
}

std::string RpuKernelGraph::dump_tree() const {
    std::ostringstream out;
    out << "GraphTree state=" << graph_state_name(state_)
        << " replayable=" << (replayable_ ? "true" : "false")
        << " graph_size=" << nodes_.size()
        << " segments=" << segments_.size()
        << " boundary_flush_count=" << boundary_flush_ptrs_.size() << "\n";
    const auto segment_ids = graph_segment_ids(nodes_.size(), segments_);
    for (const auto& segment : segments_) {
        out << "  ";
        append_segment_summary(out, segment);
        out << "\n";
        for (size_t i = segment.start_idx; i < segment.end_idx; ++i) {
            out << "    " << safe_node_summary(i, nodes_[i], segment_ids[i])
                << "\n";
        }
    }
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (segment_ids[i] < 0) {
            out << "  inter_segment "
                << safe_node_summary(i, nodes_[i], -1) << "\n";
        }
    }
    return out.str();
}
