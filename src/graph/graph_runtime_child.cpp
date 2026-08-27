// graph_runtime_child.cpp — ChildGraph registration, patching, and replay helpers.
//
// This file implements parent/child graph composition: register_child,
// RegisterPatch/DataPatch application, prepared child replay, branch marker
// capture, and window replacement helpers.
#include "graph/graph_runtime.h"
#include "graph/graph_runtime_internal.h"

#include <memory>
#include <utility>

static void mirror_per_segment_replay_count(
        GraphStats& stats,
        const std::vector<Segment>& segments) {
    stats.per_segment_replay_count.clear();
    stats.per_segment_replay_count.reserve(segments.size());
    for (const auto& seg : segments) {
        stats.per_segment_replay_count.push_back(seg.replay_count);
    }
}

// ChildGraph registration, input patching, and prepared replay
// =============================================================================

void RpuKernelGraph::register_child(std::shared_ptr<RpuKernelGraph> child,
                                    std::vector<RegisterPatch> input_patches,
                                    GraphSignature child_signature,
                                    std::vector<DataPatch> data_patches) {
    RpuExecutionCoordinator::require_no_physical_claim(
        "RpuKernelGraph::register_child");
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "register_child is forbidden by the typed DDR-register census");
    TORCH_CHECK(state_ == State::RECORDING,
                "register_child() must be called during RECORDING; "
                "current state=", static_cast<int>(state_));
    TORCH_CHECK(child && child->state() == State::BUILT && child->replayable(),
                "register_child: child graph must be BUILT and replayable; "
                "got state=", child ? static_cast<int>(child->state()) : -1,
                " replayable=", child ? child->replayable() : false);
    TORCH_CHECK(
        !child->has_retained_physical_arena_guard() &&
            !child->has_provisional_physical_build(),
        "register_child: a provisional or retained physical Graph cannot be "
        "embedded as a prepared child");
    TORCH_CHECK(
        !child->has_semantic_spm_producer_yield_marker_nodes(),
        "register_child: semantic SPM producer-yield Graphs require their "
        "own FMB replay arm and cannot be embedded as direct children");
    TORCH_CHECK(child.get() != this,
                "register_child: cannot register self as child");
    // 默认严格契约:child 的 nodes_ 只允许 Kernel。传 DataPatch 时
    // opt-in 放宽 Memcpy 节点,但仍拒绝 Memset / nested ChildGraph / HostCallback
    // / Tier3 等未建补丁语义的节点。replay_prepared_child 走 prepared_wq +
    // RegisterPatch 的轻路径,不重跑 op stream:
    //   - Memcpy/Memset 的 dst/src 不会被 capture_data 在 REPLAYING 期重新覆写
    //   - 没被 RegisterPatch 覆盖的 kernel reg 仍然指向 RECORDING 期录下的
    //     dev_addr,一旦那只 tensor 被 GC,父图 replay 立刻 use-after-free
    //   - 嵌套 ChildGraph 涉及 patch 链路,v1 不展开
    // OutputPatch 落地后再放宽输出 ownership 相关限制。
    bool has_memcpy = false;
    for (size_t i = 0; i < child->nodes_.size(); ++i) {
        const auto& n = child->nodes_[i];
        has_memcpy = has_memcpy || n.kind == GraphNodeKind::Memcpy;
        TORCH_CHECK(n.kind == GraphNodeKind::Kernel ||
                        n.kind == GraphNodeKind::Memcpy,
                    "register_child: child node #", i, " is kind=",
                    static_cast<int>(n.kind),
                    " (supports Kernel, plus Memcpy when DataPatch "
                    "entries are supplied; Memset/nested ChildGraph/"
                    "HostCallback/Tier3 deferred). "
                    "child_signature=", child_signature.to_string());
    }
    TORCH_CHECK(!has_memcpy || !data_patches.empty(),
                "register_child: child contains Memcpy nodes but no DataPatch "
                "entries were provided");
    // 校验 patch 的 kernel_idx 不越界,避免 replay 时 set_regs 崩
    for (const auto& p : input_patches) {
        TORCH_CHECK(p.kernel_idx < child->kernels_.size(),
                    "register_child: RegisterPatch.kernel_idx=", p.kernel_idx,
                    " out of range for child (size=", child->kernels_.size(),
                    ")");
    }
    for (const auto& p : data_patches) {
        TORCH_CHECK(p.node_idx < child->nodes_.size(),
                    "register_child: DataPatch.node_idx=",
                    p.node_idx, " out of range for child (size=",
                    child->nodes_.size(), ")");
        TORCH_CHECK(child->nodes_[p.node_idx].kind == GraphNodeKind::Memcpy,
                    "register_child: DataPatch.node_idx=",
                    p.node_idx, " is not a Memcpy node");
        TORCH_CHECK(p.field == 0 || p.field == 1,
                    "register_child: DataPatch.field must be "
                    "0(Memcpy.dst) or 1(Memcpy.src); got ",
                    static_cast<int>(p.field));
    }

    ChildGraphNodeData data;
    child->ever_embedded_as_child_ = true;
    data.child = std::move(child);
    data.child_signature = std::move(child_signature);
    data.input_patches = std::move(input_patches);
    data.data_patches = std::move(data_patches);
    capture_data<ChildGraphNodeData>(std::move(data));
}

void RpuKernelGraph::patch_child_inputs(size_t node_idx,
                                        std::vector<RegisterPatch> patches) {
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "patch_child_inputs is forbidden by the typed DDR-register "
                "census");
    TORCH_CHECK(state_ == State::REPLAYING,
                "patch_child_inputs() must be called during REPLAYING; "
                "current state=", static_cast<int>(state_));
    TORCH_CHECK(node_idx < nodes_.size(),
                "patch_child_inputs: node_idx=", node_idx,
                " out of range (nodes_.size=", nodes_.size(), ")");
    TORCH_CHECK(nodes_[node_idx].kind == GraphNodeKind::ChildGraph,
                "patch_child_inputs: node ", node_idx,
                " is not a ChildGraph node");
    TORCH_CHECK(cursor_ == node_idx || cursor_ == node_idx + 1,
                "patch_child_inputs: replay cursor mismatch for ChildGraph "
                "node ", node_idx, " (cursor=", cursor_, ")");
    const bool consume_marker = (cursor_ == node_idx);
    auto& cgd = nodes_[node_idx].as_child_graph();
    // 防御:patch 的 kernel_idx 必须仍在 child kernels_ 范围内。
    for (const auto& p : patches) {
        TORCH_CHECK(p.kernel_idx < cgd.child->kernels_.size(),
                    "patch_child_inputs: RegisterPatch.kernel_idx=",
                    p.kernel_idx, " out of range for child (size=",
                    cgd.child->kernels_.size(), ")");
    }
    cgd.input_patches = std::move(patches);
    if (consume_marker) {
        cursor_++;
    }
}

void RpuKernelGraph::patch_child_data(size_t node_idx,
                                      std::vector<DataPatch> patches) {
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "patch_child_data is forbidden by the typed DDR-register "
                "census");
    TORCH_CHECK(state_ == State::REPLAYING,
                "patch_child_data() must be called during REPLAYING; "
                "current state=", static_cast<int>(state_));
    TORCH_CHECK(node_idx < nodes_.size(),
                "patch_child_data: node_idx=", node_idx,
                " out of range (nodes_.size=", nodes_.size(), ")");
    TORCH_CHECK(nodes_[node_idx].kind == GraphNodeKind::ChildGraph,
                "patch_child_data: node ", node_idx,
                " is not a ChildGraph node");
    TORCH_CHECK(cursor_ == node_idx || cursor_ == node_idx + 1,
                "patch_child_data: replay cursor mismatch for ChildGraph "
                "node ", node_idx, " (cursor=", cursor_, ")");
    const bool consume_marker = (cursor_ == node_idx);
    auto& cgd = nodes_[node_idx].as_child_graph();
    for (const auto& p : patches) {
        TORCH_CHECK(p.node_idx < cgd.child->nodes_.size(),
                    "patch_child_data: DataPatch.node_idx=", p.node_idx,
                    " out of range for child (size=",
                    cgd.child->nodes_.size(), ")");
        TORCH_CHECK(cgd.child->nodes_[p.node_idx].kind == GraphNodeKind::Memcpy,
                    "patch_child_data: DataPatch.node_idx=", p.node_idx,
                    " is not a Memcpy node");
        TORCH_CHECK(p.field == 0 || p.field == 1,
                    "patch_child_data: DataPatch.field must be 0(Memcpy.dst) "
                    "or 1(Memcpy.src); got ", static_cast<int>(p.field));
    }
    cgd.data_patches = std::move(patches);
    if (consume_marker) {
        cursor_++;
    }
}

ChildGraphNodeData& RpuKernelGraph::enter_replay_child_skip(
        const char* caller) {
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                caller, ": ChildGraph skip is forbidden by the typed "
                "DDR-register census");
    TORCH_CHECK(state_ == State::REPLAYING,
                caller, ": child skip is only valid during REPLAYING; state=",
                static_cast<int>(state_));
    if (!replay_child_skip_.active) {
        TORCH_CHECK(cursor_ < nodes_.size(),
                    caller, ": replay cursor overflow entering child skip: ",
                    cursor_, " >= ", nodes_.size());
        TORCH_CHECK(nodes_[cursor_].kind == GraphNodeKind::ChildGraph,
                    caller, ": expected ChildGraph node at parent cursor ",
                    cursor_, ", got kind=",
                    static_cast<int>(nodes_[cursor_].kind));
        replay_child_skip_.active = true;
        replay_child_skip_.parent_node_idx = cursor_;
        replay_child_skip_.child_cursor = 0;
    }

    auto& cgd = nodes_[replay_child_skip_.parent_node_idx].as_child_graph();
    TORCH_CHECK(cgd.child &&
                    cgd.child->state() == State::BUILT &&
                    cgd.child->replayable(),
                caller, ": ChildGraph child is not BUILT/replayable");
    TORCH_CHECK(replay_child_skip_.child_cursor <= cgd.child->nodes_.size(),
                caller, ": child cursor overflow ",
                replay_child_skip_.child_cursor, " > ",
                cgd.child->nodes_.size());
    return cgd;
}

void RpuKernelGraph::finish_replay_child_skip_if_complete() {
    if (!replay_child_skip_.active) {
        return;
    }
    auto& cgd = nodes_[replay_child_skip_.parent_node_idx].as_child_graph();
    TORCH_CHECK(cgd.child != nullptr,
                "finish_replay_child_skip_if_complete: missing child graph");
    if (replay_child_skip_.child_cursor != cgd.child->nodes_.size()) {
        return;
    }

    std::vector<DataPatch> data_patches;
    for (size_t i = 0; i < cgd.child->nodes_.size(); ++i) {
        const auto& n = cgd.child->nodes_[i];
        if (n.kind != GraphNodeKind::Memcpy) {
            continue;
        }
        const auto& m = n.as_memcpy();
        if (m.dst != nullptr) {
            data_patches.push_back(DataPatch{
                i,
                static_cast<uint8_t>(0),
                reinterpret_cast<uint64_t>(m.dst),
                m.dst_dev_addr,
            });
        }
        if (m.src != nullptr) {
            data_patches.push_back(DataPatch{
                i,
                static_cast<uint8_t>(1),
                reinterpret_cast<uint64_t>(m.src),
                m.src_dev_addr,
            });
        }
    }
    if (!data_patches.empty()) {
        cgd.data_patches = std::move(data_patches);
    }

    cursor_ = replay_child_skip_.parent_node_idx + 1;
    replay_child_skip_ = ReplayChildSkipState{};
}

void RpuKernelGraph::record_branch(uint64_t branch_key,
                                   GraphSignature branch_signature,
                                   std::string label) {
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "record_branch is forbidden by the typed DDR-register census");
    TORCH_CHECK(state_ == State::RECORDING || state_ == State::REPLAYING,
                "record_branch() must be called during RECORDING/REPLAYING; "
                "current state=", static_cast<int>(state_));
    BranchNodeData data;
    data.branch_key = branch_key;
    data.branch_signature = std::move(branch_signature);
    data.label = std::move(label);
    capture_data<BranchNodeData>(std::move(data));
}

void RpuKernelGraph::replay_prepared_child(
        const std::vector<RegisterPatch>& patches) {
    RpuExecutionCoordinator::require_no_physical_claim(
        "RpuKernelGraph::replay_prepared_child");
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "replay_prepared_child is forbidden by the typed DDR-register "
                "census");
    check_foreign_graph_execution_allowed(
        "replay_prepared_child: direct child replay");
    TORCH_CHECK(state_ == State::BUILT && replayable_,
                "replay_prepared_child: child must be BUILT and replayable; "
                "state=", static_cast<int>(state_),
                " replayable=", replayable_);
    TORCH_CHECK(
        !has_retained_physical_arena_guard() &&
            !has_provisional_physical_build(),
        "replay_prepared_child: provisional/retained physical Graphs require "
        "ordinary exact signed replay");
    // v1 严格契约(同 register_child):只允许 Kernel 节点。data-node
    // dst/src 不在直接 replay 路径上被覆写,kernel reg 中没被 patch 的
    // dev_addr 也不会刷新;含 Memcpy/Memset 的 child 必须等 DataPatch v2。
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& n = nodes_[i];
        TORCH_CHECK(n.kind == GraphNodeKind::Kernel,
                    "replay_prepared_child: node #", i, " is kind=",
                    static_cast<int>(n.kind),
                    " (only Kernel allowed in v1)");
    }
    // 1) 应用 patches 到自身 kernels_
    for (const auto& p : patches) {
        TORCH_CHECK(p.kernel_idx < kernels_.size(),
                    "replay_prepared_child: RegisterPatch.kernel_idx=",
                    p.kernel_idx, " out of range (kernels_.size=",
                    kernels_.size(), ")");
        apply_register_patch_to_kernel(*this, p, kernels_[p.kernel_idx].get());
    }
    // 2) 临时切到 REPLAYING 跑 prepared segments;不动 active_stack_ /
    //    SPM_POOL.defer_depth_ —— 子图的 SPM 在 RECORDING 期就锁定,父 scope
    //    的 defer 状态由父 begin/end 配对维护。
    cursor_ = 0;
    boundary_flush_ptrs_.clear();
    local_spm_pool_.reset_cursor();
    global_spm_pool_.reset_cursor();
    state_ = State::REPLAYING;
    execute_graph_for_replaying();
    // 3) 切回 BUILT,以便父 graph 后续 forward 可再次 replay 同一 child。
    state_ = State::BUILT;
}

void RpuKernelGraph::replay_prepared_child_with_data_patches(
        const std::vector<RegisterPatch>& register_patches,
        const std::vector<DataPatch>& data_patches) {
    RpuExecutionCoordinator::require_no_physical_claim(
        "RpuKernelGraph::replay_prepared_child_with_data_patches");
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "replay_prepared_child_with_data_patches is forbidden by the "
                "typed DDR-register census");
    check_foreign_graph_execution_allowed(
        "replay_prepared_child_with_data_patches: direct child replay");
    TORCH_CHECK(state_ == State::BUILT && replayable_,
                "replay_prepared_child_with_data_patches: child must be BUILT "
                "and replayable; state=", static_cast<int>(state_),
                " replayable=", replayable_);
    TORCH_CHECK(
        !has_retained_physical_arena_guard() &&
            !has_provisional_physical_build(),
        "replay_prepared_child_with_data_patches: provisional/retained "
        "physical Graphs require ordinary exact signed replay");
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& n = nodes_[i];
        TORCH_CHECK(n.kind == GraphNodeKind::Kernel ||
                        n.kind == GraphNodeKind::Memcpy,
                    "replay_prepared_child_with_data_patches: node #", i,
                    " is kind=", static_cast<int>(n.kind),
                    " (optimized child replay supports only Kernel/Memcpy)");
    }
    for (const auto& p : register_patches) {
        TORCH_CHECK(p.kernel_idx < kernels_.size(),
                    "replay_prepared_child_with_data_patches: "
                    "RegisterPatch.kernel_idx=", p.kernel_idx,
                    " out of range (kernels_.size=", kernels_.size(), ")");
        apply_register_patch_to_kernel(*this, p, kernels_[p.kernel_idx].get());
    }
    for (const auto& p : data_patches) {
        TORCH_CHECK(p.node_idx < nodes_.size(),
                    "replay_prepared_child_with_data_patches: "
                    "DataPatch.node_idx=", p.node_idx,
                    " out of range (nodes_.size=", nodes_.size(), ")");
        TORCH_CHECK(nodes_[p.node_idx].kind == GraphNodeKind::Memcpy,
                    "replay_prepared_child_with_data_patches: "
                    "DataPatch.node_idx=", p.node_idx,
                    " is not a Memcpy node");
        TORCH_CHECK(p.field == 0 || p.field == 1,
                    "replay_prepared_child_with_data_patches: DataPatch.field "
                    "must be 0(Memcpy.dst) or 1(Memcpy.src); got ",
                    static_cast<int>(p.field));
        TORCH_CHECK(p.ptr != 0,
                    "replay_prepared_child_with_data_patches: DataPatch.ptr "
                    "must be non-zero");
        auto& m = nodes_[p.node_idx].as_memcpy();
        void* ptr = reinterpret_cast<void*>(
            static_cast<uintptr_t>(p.ptr));
        const uint64_t dev = p.dev_addr != 0
            ? p.dev_addr
            : rhino_lkn::RpuGetDevAddr(ptr);
        if (p.field == 0) {
            m.dst = ptr;
            m.dst_dev_addr = dev;
        } else {
            m.src = ptr;
            m.src_dev_addr = dev;
        }
        if (m.kind == CopyKind::DDR_TO_DDR) {
            TORCH_CHECK(m.dst_dev_addr != 0 && m.src_dev_addr != 0,
                        "replay_prepared_child_with_data_patches: DDR_TO_DDR "
                        "Memcpy node requires non-zero dst/src dev_addr after "
                        "patch; node_idx=", p.node_idx);
        }
    }

    cursor_ = 0;
    boundary_flush_ptrs_.clear();
    local_spm_pool_.reset_cursor();
    global_spm_pool_.reset_cursor();
    state_ = State::REPLAYING;
    execute_graph_for_replaying();
    state_ = State::BUILT;
}

void RpuKernelGraph::replace_window_with_child(
        size_t start_idx, size_t end_idx,
        std::shared_ptr<RpuKernelGraph> child,
        std::vector<RegisterPatch> input_patches,
        std::vector<DataPatch> data_patches,
        GraphSignature child_signature) {
    RpuExecutionCoordinator::require_no_physical_claim(
        "RpuKernelGraph::replace_window_with_child");
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "replace_window_with_child is forbidden by the typed "
                "DDR-register census");
    TORCH_CHECK(state_ == State::BUILT && replayable_,
                "replace_window_with_child: parent must be BUILT and "
                "replayable; state=", static_cast<int>(state_),
                " replayable=", replayable_);
    // A BUILD observed by FMB/SPM tracing has already published node windows
    // and/or a full topology identity. Rewriting nodes_ in place would leave
    // those stamps apparently current under the same generation. Fail before
    // validating or moving any child payload; the owner must invalidate and
    // rebuild so fresh stamps receive a fresh BUILD generation.
    TORCH_CHECK(!topology_hash_requested_ && build_topology_hash_ == 0,
                "replace_window_with_child: observed BUILD topology is sealed; "
                "invalidate and rebuild before replacing a node window");
    TORCH_CHECK(start_idx < end_idx && end_idx <= nodes_.size(),
                "replace_window_with_child: invalid window [", start_idx,
                ", ", end_idx, ") for nodes_.size=", nodes_.size());
    TORCH_CHECK(child && child->state() == State::BUILT && child->replayable(),
                "replace_window_with_child: child graph must be BUILT and "
                "replayable; got state=",
                child ? static_cast<int>(child->state()) : -1,
                " replayable=", child ? child->replayable() : false);
    TORCH_CHECK(
        !child->has_retained_physical_arena_guard() &&
            !child->has_provisional_physical_build(),
        "replace_window_with_child: a provisional or retained physical "
        "Graph cannot be embedded as a prepared child");
    TORCH_CHECK(
        !child->has_semantic_spm_producer_yield_marker_nodes(),
        "replace_window_with_child: semantic SPM producer-yield Graphs "
        "cannot be detached from their owning FMB replay arm");
    TORCH_CHECK(child.get() != this,
                "replace_window_with_child: cannot register self as child");

    bool has_memcpy = false;
    for (size_t i = 0; i < child->nodes_.size(); ++i) {
        const auto& n = child->nodes_[i];
        has_memcpy = has_memcpy || n.kind == GraphNodeKind::Memcpy;
        TORCH_CHECK(n.kind == GraphNodeKind::Kernel ||
                        n.kind == GraphNodeKind::Memcpy,
                    "replace_window_with_child: child node #", i,
                    " is kind=", static_cast<int>(n.kind),
                    " (optimized child replay supports only Kernel/Memcpy; "
                    "Memset/nested ChildGraph/HostCallback/Tier3 deferred). "
                    "child_signature=", child_signature.to_string());
    }
    TORCH_CHECK(!has_memcpy || !data_patches.empty(),
                "replace_window_with_child: child contains Memcpy nodes but "
                "no DataPatch entries were provided");
    for (const auto& p : input_patches) {
        TORCH_CHECK(p.kernel_idx < child->kernels_.size(),
                    "replace_window_with_child: RegisterPatch.kernel_idx=",
                    p.kernel_idx, " out of range for child (size=",
                    child->kernels_.size(), ")");
    }
    for (const auto& p : data_patches) {
        TORCH_CHECK(p.node_idx < child->nodes_.size(),
                    "replace_window_with_child: DataPatch.node_idx=",
                    p.node_idx, " out of range for child (size=",
                    child->nodes_.size(), ")");
        TORCH_CHECK(child->nodes_[p.node_idx].kind == GraphNodeKind::Memcpy,
                    "replace_window_with_child: DataPatch.node_idx=",
                    p.node_idx, " is not a Memcpy node");
        TORCH_CHECK(p.field == 0 || p.field == 1,
                    "replace_window_with_child: DataPatch.field must be "
                    "0(Memcpy.dst) or 1(Memcpy.src); got ",
                    static_cast<int>(p.field));
    }

    ChildGraphNodeData data;
    child->ever_embedded_as_child_ = true;
    data.child = std::move(child);
    data.child_signature = std::move(child_signature);
    data.input_patches = std::move(input_patches);
    data.data_patches = std::move(data_patches);
    GraphNode node{GraphNodeKind::ChildGraph, std::move(data)};

    auto first = nodes_.begin() + static_cast<std::ptrdiff_t>(start_idx);
    auto last = nodes_.begin() + static_cast<std::ptrdiff_t>(end_idx);
    first = nodes_.erase(first, last);
    nodes_.insert(first, std::move(node));

    release_prepared_queues();
    build_segments_from_nodes();
    size_t kernels = 0;
    size_t data_nodes = 0;
    for (const auto& n : nodes_) {
        if (n.kind == GraphNodeKind::Kernel) {
            ++kernels;
        } else {
            ++data_nodes;
        }
    }
    last_stats_.kernel_count = kernels;
    last_stats_.data_node_count = data_nodes;
    last_stats_.segment_count = segments_.size();
    mirror_per_segment_replay_count(last_stats_, segments_);
    cursor_ = 0;
    replay_child_skip_ = ReplayChildSkipState{};
}
