// graph_runtime_execute.cpp — kernel dispatch, segment launch, and node execution.
//
// This file contains the hot replay/record execution path: GET_KERNEL-backed
// enqueue, fast replay skip helpers, segment construction/preparation, data node
// execution, and execute_graph_* loops.
#include "graph/graph_runtime.h"
#include "graph/graph_runtime_internal.h"
#include "rpu_copy_safety.h"
#include "rpu_profile.h"        // P2 (debug-level): log_at(N) gate
#include <fcntl.h>

#include <ATen/record_function.h>
#include <c10/util/ScopeExit.h>
#include <algorithm>
#include <map>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <thread>
#include <unistd.h>             // ::getpid (HW perf dump path)
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// Unlike per-Graph diagnostics, this witness survives failed prepare graphs
// being retired before the caller takes its after-snapshot.
std::atomic<uint64_t> process_hwperf_evidence_failure_total{0};

void coalesce_fixed_dma_allocation_witnesses(
        std::vector<RpuDmaAllocationWitness>& witnesses) {
    // A live logical allocation is one immutable contiguous range. All of its
    // fixed endpoint ranges fit iff their envelope fits. Build that value once;
    // replay still checks every node/address and revalidates the live identity
    // and lease. Keep different allocation IDs separate, even in one DDR arena.
    std::unordered_map<uint64_t, size_t> fixed_allocation_indices;
    size_t fixed_allocation_count = 0;
    for (size_t i = 0; i < witnesses.size(); ++i) {
        const auto witness = witnesses[i];
        TORCH_CHECK(witness.allocation_id != 0 &&
                        witness.where != nullptr &&
                        witness.expected_device_addr != 0 && witness.bytes > 0 &&
                        witness.bytes - 1 <=
                            std::numeric_limits<uint64_t>::max() -
                                witness.expected_device_addr,
                    "prepare_segment_queue: invalid fixed allocation range");
        const auto inserted = fixed_allocation_indices.emplace(
            witness.allocation_id, fixed_allocation_count);
        if (inserted.second) {
            witnesses[fixed_allocation_count++] = witness;
            continue;
        }
        auto& combined = witnesses[inserted.first->second];
        const uint64_t first = std::min(
            combined.expected_device_addr, witness.expected_device_addr);
        // Inclusive ends preserve valid ranges whose last byte is UINT64_MAX.
        const uint64_t last = std::max(
            combined.expected_device_addr + (combined.bytes - 1),
            witness.expected_device_addr + (witness.bytes - 1));
        TORCH_CHECK(last - first < std::numeric_limits<size_t>::max(),
                    "prepare_segment_queue: fixed allocation envelope overflow");
        combined.expected_device_addr = first;
        combined.bytes = static_cast<size_t>(last - first) + 1;
        combined.where = "launch_segment_for_replay/Fixed allocation range";
    }
    witnesses.resize(fixed_allocation_count);
}

uint8_t validated_kernel_core_extent(
        const GraphRuntimePolicy& policy,
        const std::vector<uint8_t>& core_ids) {
    TORCH_CHECK(!core_ids.empty(), "Graph kernel core IDs must not be empty");
    TORCH_CHECK(core_ids.size() <= static_cast<size_t>(policy.execution_core_count),
                "Graph kernel core count exceeds execution_core_count");
    for (size_t i = 0; i < core_ids.size(); ++i) {
        TORCH_CHECK(core_ids[i] < policy.execution_core_count,
                    "Graph kernel core ID ", static_cast<int>(core_ids[i]),
                    " exceeds execution_core_count=", policy.execution_core_count);
        // LKN encodes a kernel domain as count + first core, not a bit mask.
        TORCH_CHECK(static_cast<size_t>(core_ids[i]) == core_ids.front() + i,
                    "Graph kernel core IDs must be a contiguous ascending range");
    }
    // Queue_t owns a prefix even when this kernel starts on a later core.
    return static_cast<uint8_t>(core_ids.back() + 1);
}

void validate_semantic_dma_owner_for_execution(
        const DmaNodeData& dma, size_t node_idx, const char* caller) {
    if (dma.semantic_endpoint_id == 0) return;
    const std::shared_ptr<const uint64_t> owner =
        dma.semantic_owner_generation.lock();
    TORCH_CHECK(
        owner != nullptr && dma.semantic_expected_owner_generation != 0 &&
            *owner == dma.semantic_expected_owner_generation,
        caller, ": semantic DMA owner is stale before execution at node ",
        node_idx);
    TORCH_CHECK(
        dma.semantic_spm_peer_id == 0 ||
            dma.semantic_canonical_member_emission,
        caller, ": typed SPM peer lacks canonical provenance at node ",
        node_idx);
}

bool is_legacy_spm_transport_kernel_name(const std::string& name) {
    // These transports were retired from KernelId and the current operator
    // refs. Dynamic callers must still be rejected before they can record an
    // unowned node inside a canonical FMB DMA trace.
    static constexpr const char* kLegacyNames[] = {
        "ddr2spm_multi_core",
        "ddr2spm_multi_core_v2",
        "ddr2spm_multi_core_scatter",
        "spm2ddr_multi_core",
        "spm2ddr_multi_core_v2",
        "spm_gather_ddr",
        "ddr_scatter_spm",
    };
    for (const char* legacy_name : kLegacyNames) {
        if (name == legacy_name) return true;
    }
    return false;
}

// Direct enqueu_kernel invalidates any prepared batch on the same Queue_t,
// so keep PASSTHROUGH
// kernels isolated from QueueCache, which immediate DMA wrappers use for
// one-shot batches. A single queue is enough because direct launches are
// synchronous; recreating it on a rare core-count change also caps SDK
// BufferPool use at one extra slot.
struct PassthroughKernelQueue {
    std::unique_ptr<::rhino_lkn::Queue_t> queue;
    uint8_t core_num = 0;
};

PassthroughKernelQueue& passthrough_kernel_queue_state() {
    static PassthroughKernelQueue state;
    return state;
}

::rhino_lkn::Queue_t* passthrough_kernel_queue(size_t core_num) {
    TORCH_CHECK(core_num >= 1 && core_num <= 8,
                "PASSTHROUGH kernel core count must be in [1, 8], got ",
                core_num);
    auto& state = passthrough_kernel_queue_state();
    if (!state.queue || state.core_num != core_num) {
        state.queue.reset();
        state.queue = std::make_unique<::rhino_lkn::Queue_t>(core_num);
        state.core_num = static_cast<uint8_t>(core_num);
    }
    return state.queue.get();
}

}  // namespace

// rpu_shutdown() calls this before tearing down DDRManager.
void graph_clear_passthrough_kernel_queues() {
    auto& state = passthrough_kernel_queue_state();
    state.queue.reset();
    state.core_num = 0;
}

// =============================================================================
// get_kernel
// =============================================================================

const KernelNodeData*
RpuKernelGraph::next_replay_kernel_for_semantic_spm_producer_yield() const {
    if (state_ != State::REPLAYING) return nullptr;

    if (replay_child_skip_.active ||
        (cursor_ < nodes_.size() &&
         nodes_[cursor_].kind == GraphNodeKind::ChildGraph)) {
        const size_t parent_node_idx = replay_child_skip_.active
            ? replay_child_skip_.parent_node_idx : cursor_;
        if (parent_node_idx >= nodes_.size() ||
            nodes_[parent_node_idx].kind != GraphNodeKind::ChildGraph) {
            return nullptr;
        }
        const auto& child = nodes_[parent_node_idx].as_child_graph().child;
        if (!child) return nullptr;
        size_t scan = replay_child_skip_.active
            ? replay_child_skip_.child_cursor : 0;
        while (scan < child->nodes_.size() &&
               child->nodes_[scan].kind != GraphNodeKind::Kernel) {
            ++scan;
        }
        return scan < child->nodes_.size()
            ? &child->nodes_[scan].as_kernel() : nullptr;
    }

    size_t scan = cursor_;
    while (scan < nodes_.size() &&
           nodes_[scan].kind != GraphNodeKind::Kernel) {
        ++scan;
    }
    return scan < nodes_.size() ? &nodes_[scan].as_kernel() : nullptr;
}

void RpuKernelGraph::
validate_semantic_spm_producer_yield_before_kernel_lookup() const {
    TORCH_CHECK(!semantic_spm_producer_yield_poisoned_,
                "kernel lookup is forbidden in a poisoned semantic SPM "
                "producer-yield scope");
    const bool direct_marker_graph =
        has_semantic_spm_producer_yield_nodes_;
    const bool pending = pending_semantic_spm_producer_yield_.has_value();
    bool child_may_have_marker = false;
    if (state_ == State::REPLAYING && !direct_marker_graph && !pending &&
        (replay_child_skip_.active ||
         (cursor_ < nodes_.size() &&
          nodes_[cursor_].kind == GraphNodeKind::ChildGraph))) {
        const KernelNodeData* child_kernel =
            next_replay_kernel_for_semantic_spm_producer_yield();
        child_may_have_marker = child_kernel != nullptr &&
            child_kernel->semantic_spm_producer_yield.has_value();
    }
    if (!direct_marker_graph && !pending && !child_may_have_marker) return;

    TORCH_CHECK(state_ == State::RECORDING || state_ == State::REPLAYING,
                "semantic SPM producer-yield kernel lookup requires "
                "RECORDING or REPLAYING");
    const KernelNodeData* retained =
        next_replay_kernel_for_semantic_spm_producer_yield();
    if (!pending) {
        TORCH_CHECK(
            retained == nullptr ||
                !retained->semantic_spm_producer_yield.has_value(),
            "marked semantic SPM producer-yield kernel requires an exact "
            "staged marker during ordinary REPLAY");
        return;
    }

    const auto& staged = *pending_semantic_spm_producer_yield_;
    const std::shared_ptr<const uint64_t> owner =
        staged.owner_generation.lock();
    TORCH_CHECK(!staged.kernel_lookup_seen &&
                    staged.lookup_kernel == nullptr && owner != nullptr &&
                    staged.expected_owner_generation != 0 &&
                    *owner == staged.expected_owner_generation,
                "semantic SPM producer-yield marker is duplicate or stale "
                "before kernel lookup");
    if (state_ == State::REPLAYING) {
        verify_semantic_spm_producer_yield_replay_arm("kernel lookup");
        TORCH_CHECK(retained != nullptr &&
                        retained->semantic_spm_producer_yield.has_value() &&
                        retained->semantic_spm_producer_yield
                                ->semantic_yield_id ==
                            staged.semantic_yield_id &&
                        retained->semantic_spm_producer_yield->writer_kind ==
                            staged.writer_kind,
                    "semantic SPM producer-yield marker does not match the "
                    "next retained kernel");
    }
}

void RpuKernelGraph::bind_semantic_spm_producer_yield_kernel_lookup(
        ::rhino_lkn::Kernel_t& kernel) {
    if (!pending_semantic_spm_producer_yield_.has_value()) return;
    validate_semantic_spm_producer_yield_before_kernel_lookup();
    auto& staged = *pending_semantic_spm_producer_yield_;
    staged.lookup_kernel = &kernel;
    staged.kernel_lookup_seen = true;
}

void RpuKernelGraph::attach_semantic_spm_producer_yield_to_recording_node(
        KernelNodeData& node,
        const ::rhino_lkn::Kernel_t& kernel) const {
    if (!pending_semantic_spm_producer_yield_.has_value()) return;
    const auto& staged = *pending_semantic_spm_producer_yield_;
    const std::shared_ptr<const uint64_t> owner =
        staged.owner_generation.lock();
    TORCH_CHECK(state_ == State::RECORDING && staged.kernel_lookup_seen &&
                    staged.lookup_kernel == &kernel && owner != nullptr &&
                    staged.expected_owner_generation != 0 &&
                    *owner == staged.expected_owner_generation &&
                    staged.semantic_yield_id != 0,
                "semantic SPM producer-yield marker did not reach its exact "
                "recording kernel");
    TORCH_CHECK(!node.semantic_spm_producer_yield.has_value(),
                "recording kernel already has a semantic SPM producer-yield "
                "marker");
    KernelNodeData::SemanticSpmProducerYieldMarker marker;
    marker.semantic_yield_id = staged.semantic_yield_id;
    marker.writer_kind = staged.writer_kind;
    node.semantic_spm_producer_yield.emplace(std::move(marker));
}

void RpuKernelGraph::validate_semantic_spm_producer_yield_replay_node(
        const KernelNodeData& node,
        const ::rhino_lkn::Kernel_t& kernel) const {
    const bool marked = node.semantic_spm_producer_yield.has_value();
    const bool pending = pending_semantic_spm_producer_yield_.has_value();
    TORCH_CHECK(marked == pending,
                marked
                    ? "marked semantic SPM producer-yield kernel was not staged"
                    : "semantic SPM producer-yield marker targeted an unmarked kernel");
    if (!marked) return;
    const auto& staged = *pending_semantic_spm_producer_yield_;
    const auto& retained = *node.semantic_spm_producer_yield;
    const std::shared_ptr<const uint64_t> owner =
        staged.owner_generation.lock();
    TORCH_CHECK(state_ == State::REPLAYING &&
                    staged.kernel_lookup_seen &&
                    staged.lookup_kernel == &kernel && owner != nullptr &&
                    staged.expected_owner_generation != 0 &&
                    *owner == staged.expected_owner_generation &&
                    staged.semantic_yield_id == retained.semantic_yield_id &&
                    staged.writer_kind == retained.writer_kind,
                "semantic SPM producer-yield replay marker or kernel drifted "
                "before enqueue");
    verify_semantic_spm_producer_yield_replay_arm("kernel enqueue");
}

void RpuKernelGraph::
commit_semantic_spm_producer_yield_after_kernel_enqueue() noexcept {
    if (!pending_semantic_spm_producer_yield_.has_value()) return;
    if (state_ == State::RECORDING) {
        has_semantic_spm_producer_yield_nodes_ = true;
    }
    pending_semantic_spm_producer_yield_.reset();
}

::rhino_lkn::Kernel_t* RpuKernelGraph::get_kernel(KernelId id) {
    check_foreign_graph_execution_allowed(
        "RpuKernelGraph: foreign Graph kernel lookup");
    TORCH_CHECK(!canonical_dma_poisoned_,
                "RpuKernelGraph: kernel lookup is forbidden in a poisoned "
                "canonical FMB DMA scope");
    // Legacy DDR/SPM transports have no KernelId entry; only the dynamic-name
    // overload below needs to reject them in a canonical DMA callback.
    validate_semantic_spm_producer_yield_before_kernel_lookup();
    // All coordinator/canonical checks above are read-only. Typed admission
    // still precedes KernelCache/program lookup, clone, reset, or cursor work.
    admit_kernel_register_lookup(id);
    switch (state_) {
    case State::PASSTHROUGH:
        return KernelCache::instance().get(id);

    case State::RECORDING: {
        ::rhino_lkn::Program_t* prog = KernelCache::instance().get_program(id);
        TORCH_CHECK(prog != nullptr,
                    "RpuKernelGraph: program not found for kernel id ",
                    static_cast<int>(id));

        pending_kernel_idx_ = kernels_.size();
        pending_kernel_id_ = id;

        auto k = std::make_unique<::rhino_lkn::Kernel_t>(
            *prog, KERNEL_ID_NAMES[static_cast<size_t>(id)]);
        ::rhino_lkn::Kernel_t* raw = k.get();
        kernels_.push_back(std::move(k));
        return raw;
    }

    case State::REPLAYING: {
        if (replay_child_skip_.active ||
            (cursor_ < nodes_.size() &&
             nodes_[cursor_].kind == GraphNodeKind::ChildGraph)) {
            auto& cgd = enter_replay_child_skip("get_kernel");
            auto& child = *cgd.child;
            size_t scan = replay_child_skip_.child_cursor;
            while (scan < child.nodes_.size() &&
                   child.nodes_[scan].kind != GraphNodeKind::Kernel) {
                ++scan;
            }
            TORCH_CHECK(scan < child.nodes_.size(),
                        "RpuKernelGraph: no upcoming child Kernel node for "
                        "get_kernel at parent cursor ",
                        replay_child_skip_.parent_node_idx,
                        " child_cursor=", replay_child_skip_.child_cursor);
            auto& kn = child.nodes_[scan].as_kernel();
            TORCH_CHECK(kn.kernel_id.has_value() && id == *kn.kernel_id,
                        "RpuKernelGraph: child kernel_id mismatch in replay "
                        "at child node ", scan, " (parent cursor=",
                        replay_child_skip_.parent_node_idx, ")");
            // The returned pointer belongs to the child Graph.  Mark that
            // queue's parameters before callers can reset/write its registers.
            child.kernel_params_dirty_this_replay_ = true;
            return child.kernels_[kn.kernel_idx].get();
        }
        // get_kernel 在 Op 代码中调用时机 —— 通常发生在本 op 的一串
        // ddr_to_spm/rpu_memcpy 之前,之后才会调用 enqueu_kernel 推进 cursor_。
        // RECORDING 分支只往 kernels_ 里推,不往 nodes_ 里推节点,所以 nodes_
        // 的下一个 Kernel 节点可能跟当前 cursor_ 中间夹着若干 data node
        // (MemcpyNodeData / MemsetNodeData)。这里前向扫描到下一个 Kernel
        // 节点并取其 kernels_ 索引,不动 cursor_ —— 后续 capture_data 和
        // enqueu_kernel 自行按序推进 cursor_。
        size_t scan = cursor_;
        while (scan < nodes_.size() &&
               nodes_[scan].kind != GraphNodeKind::Kernel) {
            ++scan;
        }
        TORCH_CHECK(scan < nodes_.size(),
                    "RpuKernelGraph: no upcoming Kernel node for get_kernel "
                    "at cursor ", cursor_);
        auto& kn = nodes_[scan].as_kernel();
        TORCH_CHECK(kn.kernel_id.has_value() && id == *kn.kernel_id,
                    "RpuKernelGraph: kernel_id mismatch in replay at node ",
                    scan, " (cursor=", cursor_, ")");
        kernel_params_dirty_this_replay_ = true;
        return kernels_[kn.kernel_idx].get();
    }

    case State::BUILT:
        TORCH_CHECK(false,
                    "RpuKernelGraph: get_kernel() called in BUILT state "
                    "(forgot to call begin()?)");
    }

    __builtin_unreachable();
}

// =============================================================================
// get_kernel (string) — Dynamic kernel API 的底层分派
// =============================================================================
// 和 get_kernel(KernelId) 完全对称:PASSTHROUGH 走 shared cache,RECORDING
// 通过 Program_t 克隆 Kernel_t 独享 reg,REPLAYING 按 cursor 前向扫 Kernel node
// 校验 kernel_name 一致。

::rhino_lkn::Kernel_t* RpuKernelGraph::get_kernel(const std::string& name) {
    check_foreign_graph_execution_allowed(
        "RpuKernelGraph: foreign Graph dynamic kernel lookup");
    TORCH_CHECK(!canonical_dma_poisoned_,
                "RpuKernelGraph: dynamic kernel lookup is forbidden in a "
                "poisoned canonical FMB DMA scope");
    TORCH_CHECK(!(canonical_dma_build_trace_active_ ||
                  canonical_dma_callback_active_) ||
                    !is_legacy_spm_transport_kernel_name(name),
                "RpuKernelGraph: dynamic legacy DDR/SPM transport kernel is "
                "forbidden inside a canonical FMB DMA callback");
    validate_semantic_spm_producer_yield_before_kernel_lookup();
    // Exact-name policy admission precedes every Graph/raw lookup mutation.
    admit_kernel_register_lookup(name);
    switch (state_) {
    case State::PASSTHROUGH:
        return KernelCache::instance().get_kernel(name);

    case State::RECORDING: {
        ::rhino_lkn::Program_t* prog = KernelCache::instance().get_program(name);
        TORCH_CHECK(prog != nullptr,
                    "RpuKernelGraph: program not found for kernel name '",
                    name, "'");

        pending_kernel_idx_ = kernels_.size();
        pending_kernel_name_ = name;
        pending_kernel_id_.reset();  // dynamic 路径,enum 侧必须空

        auto k = std::make_unique<::rhino_lkn::Kernel_t>(*prog, name.c_str());
        ::rhino_lkn::Kernel_t* raw = k.get();
        kernels_.push_back(std::move(k));
        return raw;
    }

    case State::REPLAYING: {
        if (replay_child_skip_.active ||
            (cursor_ < nodes_.size() &&
             nodes_[cursor_].kind == GraphNodeKind::ChildGraph)) {
            auto& cgd = enter_replay_child_skip("get_kernel(name)");
            auto& child = *cgd.child;
            size_t scan = replay_child_skip_.child_cursor;
            while (scan < child.nodes_.size() &&
                   child.nodes_[scan].kind != GraphNodeKind::Kernel) {
                ++scan;
            }
            TORCH_CHECK(scan < child.nodes_.size(),
                        "RpuKernelGraph: no upcoming child Kernel node for "
                        "get_kernel(name='", name, "') at parent cursor ",
                        replay_child_skip_.parent_node_idx,
                        " child_cursor=", replay_child_skip_.child_cursor);
            auto& kn = child.nodes_[scan].as_kernel();
            TORCH_CHECK(!kn.kernel_name.empty() && name == kn.kernel_name,
                        "RpuKernelGraph: child kernel_name drift in replay at "
                        "child node ", scan, " (parent cursor=",
                        replay_child_skip_.parent_node_idx,
                        ") expected='", kn.kernel_name, "' got='", name, "'");
            child.kernel_params_dirty_this_replay_ = true;
            return child.kernels_[kn.kernel_idx].get();
        }
        size_t scan = cursor_;
        while (scan < nodes_.size() &&
               nodes_[scan].kind != GraphNodeKind::Kernel) {
            ++scan;
        }
        TORCH_CHECK(scan < nodes_.size(),
                    "RpuKernelGraph: no upcoming Kernel node for get_kernel(name='",
                    name, "') at cursor ", cursor_);
        auto& kn = nodes_[scan].as_kernel();
        TORCH_CHECK(!kn.kernel_name.empty() && name == kn.kernel_name,
                    "RpuKernelGraph: kernel_name drift in replay at node ",
                    scan, " (cursor=", cursor_,
                    ") expected='", kn.kernel_name, "' got='", name, "'");
        kernel_params_dirty_this_replay_ = true;
        return kernels_[kn.kernel_idx].get();
    }

    case State::BUILT:
        TORCH_CHECK(false,
                    "RpuKernelGraph: get_kernel(name) called in BUILT state "
                    "(forgot to call begin()?)");
    }

    __builtin_unreachable();
}

// =============================================================================
// get_kernel_reset — 取 kernel 并 reset_regs（Op 端公开入口）
// =============================================================================

::rhino_lkn::Kernel_t* RpuKernelGraph::get_kernel_reset(KernelId id) {
    ::rhino_lkn::Kernel_t* k = get_kernel(id);
    TORCH_CHECK(
        k != nullptr,
        "RpuKernelGraph: kernel not found for KernelId ",
        static_cast<int>(id), " ('",
        KERNEL_ID_NAMES[static_cast<size_t>(id)], "')");
    bind_semantic_spm_producer_yield_kernel_lookup(*k);
    bind_pending_kernel_register_lookup(*k);
    k->reset_regs();
    return k;
}

::rhino_lkn::Kernel_t* RpuKernelGraph::get_kernel_reset(const std::string& name) {
    ::rhino_lkn::Kernel_t* k = get_kernel(name);
    TORCH_CHECK(k != nullptr,
                "RpuKernelGraph: kernel not found for name '", name, "'");
    bind_semantic_spm_producer_yield_kernel_lookup(*k);
    bind_pending_kernel_register_lookup(*k);
    k->reset_regs();
    return k;
}

// =============================================================================
// enqueue —— 默认 queue state 版本（供内部和兼容路径使用）
// =============================================================================

void RpuKernelGraph::enqueue(::rhino_lkn::Kernel_t& kernel,
                             const std::vector<uint16_t>& grid_dims,
                             const std::vector<uint8_t>& core_ids) {
    QueueLaunchState default_state;  // broadcast=true, flush_icache=false
    enqueue(kernel, grid_dims, core_ids, default_state);
}

// =============================================================================
// enqueue —— 带 queue_state 的版本（RpuQueue proxy 使用）
// =============================================================================

void RpuKernelGraph::enqueue(::rhino_lkn::Kernel_t& kernel,
                             const std::vector<uint16_t>& grid_dims,
                             const std::vector<uint8_t>& core_ids,
                             const QueueLaunchState& queue_state) {
    check_foreign_graph_execution_allowed(
        "RpuKernelGraph: foreign Graph kernel enqueue");
    TORCH_CHECK(!semantic_spm_producer_yield_poisoned_,
                "kernel enqueue is forbidden in a poisoned semantic SPM "
                "producer-yield scope");
    const uint8_t core_extent =
        validated_kernel_core_extent(runtime_policy_, core_ids);
    switch (state_) {
    case State::PASSTHROUGH: {
        // Keep the direct path on a queue
        // isolated from immediate DMA batches, because enqueu_kernel invalidates
        // prepared state on the same Queue_t.
        ::rhino_lkn::Queue_t* wq =
            passthrough_kernel_queue(core_extent);
        // 每次都完整写回 queue state，避免上次调用的残留
        wq->set_broadcast_mode(queue_state.broadcast_mode);
        wq->set_flush_icache(queue_state.flush_icache);
        TORCH_CHECK(wq->enqueu_kernel(kernel, grid_dims, core_ids) == 0,
                    "RpuKernelGraph::enqueue PASSTHROUGH: enqueu_kernel failed");
        break;
    }

    case State::RECORDING: {
        TORCH_CHECK(
            !(canonical_dma_build_trace_active_ ||
              canonical_dma_callback_active_) ||
                (!pending_kernel_name_.has_value() ||
                 !is_legacy_spm_transport_kernel_name(
                     *pending_kernel_name_)),
            "RpuKernelGraph: pre-staged legacy DDR/SPM transport kernel is "
            "forbidden inside a canonical FMB DMA scope");
        // 若没有 pending_kernel_id_，说明调用者没走 get_kernel(KernelId)，
        // 即 raw/dynamic kernel (rpu_bmm 走 KernelCache::get_kernel(string)
        // 就会落在这里)。按 v2_plan §12.4 标准降级：
        //   1. execute_graph_oneshot() 落地已录前缀（保证 §1.2 时序）
        //   2. 对前缀做 pre + post flush_boundary_ptrs()，保证本轮 kernel
        //      之前的 CPU 产物和前缀 kernel 之后的写回都与 eager 语义一致
        //   3. clear boundary_flush_ptrs_ —— 切 PASSTHROUGH 后后续 flush
        //      走 eager 分支，不再使用这个集合
        //   4. 清 kernels_ / nodes_ / pending_kernel_id_ + state=PASSTHROUGH
        //   5. eager launch raw kernel（复用 PASSTHROUGH 路径）
        //
        // 为什么 state 必须切到 PASSTHROUGH，不能保持 RECORDING 让后缀继续录：
        // rpu_bmm 这类 raw-kernel op **kernel 之后紧跟 spm_to_ddr** (见
        // src/rpu_bmm.cpp:255)，其中 src 是栈上 LocalSPM_t 的 cpu_ptr。若
        // 保持 RECORDING，spm_to_ddr 会被 capture_data 进 nodes_ 延迟到
        // end() 才执行，期间 LocalSPM_t 已析构、SPM slot 被复用 → 从已复用
        // 的地址读取垃圾数据。PASSTHROUGH 保证 spm_to_ddr 紧跟 kernel eager
        // 执行，时序正确。
        // Dynamic kernel API:pending_kernel_name_ 也算"正规路径"标记。只有
        // 两者都空时才是 §12.4 raw-kernel 降级(调用者既没走 GET_KERNEL(id) 也
        // 没走 active().get_kernel_reset(name))。
        if (!pending_kernel_id_.has_value() && !pending_kernel_name_.has_value()) {
            TORCH_CHECK(
                composite_fmb_invocation_.phase ==
                    CompositeFmbInvocationState::Phase::None,
                "RpuKernelGraph: raw-kernel downgrade is forbidden while "
                "a composite FMB invocation is open");
            TORCH_CHECK(!kernel_register_census_active_ &&
                            !has_kernel_register_census_nodes_ &&
                            !pending_kernel_register_census_.has_value(),
                        "RpuKernelGraph: raw-kernel downgrade is forbidden by "
                        "the typed DDR-register census");
            TORCH_CHECK(semantic_dma_owner_generation_.expired() &&
                            semantic_dma_expected_owner_generation_ == 0 &&
                            !has_semantic_dma_nodes_ &&
                            !semantic_dma_replay_owner_armed_ &&
                            !has_semantic_spm_peer_nodes_ &&
                            !semantic_spm_peer_replay_armed_ &&
                            !pending_semantic_spm_producer_yield_.has_value() &&
                            !has_semantic_spm_producer_yield_nodes_ &&
                            !semantic_spm_producer_yield_replay_armed_ &&
                            !semantic_spm_producer_yield_poisoned_ &&
                            !canonical_dma_poisoned_ &&
                            !canonical_dma_build_trace_active_ &&
                            !canonical_dma_build_trace_complete_ &&
                            !canonical_dma_callback_active_ &&
                            !canonical_dma_burst_permit_.active,
                        "RpuKernelGraph: raw-kernel fallback is forbidden "
                        "after semantic/canonical DMA authority has been "
                        "established");
            static bool warned_once = false;
            if (!warned_once) {
                warned_once = true;
                fprintf(stderr,
                    "[RpuKernelGraph] WARN: kernel launched without KernelId "
                    "or kernel_name path. Graph scope oneshots the "
                    "already-recorded prefix and falls back to PASSTHROUGH "
                    "for the remainder.\n");
            }

            flush_boundary_ptrs();
            if (!nodes_.empty()) {
                execute_graph_oneshot();
            }
            flush_boundary_ptrs();
            boundary_flush_ptrs_.clear();

            kernels_.clear();
            nodes_.clear();
            pending_kernel_id_.reset();
            pending_kernel_name_.reset();
            replayable_ = false;
            state_ = State::PASSTHROUGH;

            enqueue(kernel, grid_dims, core_ids, queue_state);
            break;
        }

        GraphNode node{
            GraphNodeKind::Kernel,
            KernelNodeData{
                pending_kernel_id_,                              // optional<KernelId>
                pending_kernel_idx_,
                pending_kernel_name_.value_or(std::string{}),    // "" 表示 enum 路径
                grid_dims,
                core_ids,
                queue_state
            }
        };
        attach_semantic_spm_producer_yield_to_recording_node(
            node.as_kernel(), kernel);
        if (kernel_register_census_active_ ||
            has_kernel_register_census_nodes_) {
            consume_pending_kernel_register_census(
                kernel, &node.as_kernel(), /*replay_node_idx=*/0);
        }
        nodes_.push_back(std::move(node));
        commit_semantic_spm_producer_yield_after_kernel_enqueue();
        pending_kernel_id_.reset();
        pending_kernel_name_.reset();
        break;
    }

    case State::REPLAYING: {
        if (replay_child_skip_.active ||
            (cursor_ < nodes_.size() &&
             nodes_[cursor_].kind == GraphNodeKind::ChildGraph)) {
            auto& cgd = enter_replay_child_skip("enqueue");
            auto& child = *cgd.child;
            TORCH_CHECK(replay_child_skip_.child_cursor < child.nodes_.size(),
                        "RpuKernelGraph: child replay cursor overflow: ",
                        replay_child_skip_.child_cursor, " >= ",
                        child.nodes_.size());
            TORCH_CHECK(
                child.nodes_[replay_child_skip_.child_cursor].kind ==
                    GraphNodeKind::Kernel,
                "RpuKernelGraph: expected child Kernel node at child cursor ",
                replay_child_skip_.child_cursor, " (parent cursor=",
                replay_child_skip_.parent_node_idx, ")");
            auto& kn = child.nodes_[replay_child_skip_.child_cursor].as_kernel();
            TORCH_CHECK(grid_dims == kn.grid_dims,
                        "RpuKernelGraph: child grid_dims mismatch in replay "
                        "at child cursor ", replay_child_skip_.child_cursor);
            TORCH_CHECK(core_ids == kn.core_ids,
                        "RpuKernelGraph: child core_ids mismatch in replay "
                        "at child cursor ", replay_child_skip_.child_cursor);
            TORCH_CHECK(queue_state == kn.queue_state,
                        "RpuKernelGraph: child queue_state mismatch in replay "
                        "at child cursor ", replay_child_skip_.child_cursor);
            validate_semantic_spm_producer_yield_replay_node(kn, kernel);
            ++replay_child_skip_.child_cursor;
            commit_semantic_spm_producer_yield_after_kernel_enqueue();
            finish_replay_child_skip_if_complete();
            break;
        }
        TORCH_CHECK(cursor_ < nodes_.size(),
                    "RpuKernelGraph: replay cursor overflow: ", cursor_,
                    " >= ", nodes_.size());
        TORCH_CHECK(nodes_[cursor_].kind == GraphNodeKind::Kernel,
                    "RpuKernelGraph: expected Kernel node at cursor ", cursor_);
        auto& kn = nodes_[cursor_].as_kernel();
        TORCH_CHECK(grid_dims == kn.grid_dims,
                    "RpuKernelGraph: grid_dims mismatch in replay at cursor ",
                    cursor_);
        TORCH_CHECK(core_ids == kn.core_ids,
                    "RpuKernelGraph: core_ids mismatch in replay at cursor ",
                    cursor_);
        TORCH_CHECK(queue_state == kn.queue_state,
                    "RpuKernelGraph: queue_state mismatch in replay at cursor ",
                    cursor_);
        validate_semantic_spm_producer_yield_replay_node(kn, kernel);
        if (kernel_register_census_active_ ||
            has_kernel_register_census_nodes_) {
            consume_pending_kernel_register_census(
                kernel, /*recording_node=*/nullptr, cursor_);
        }
        cursor_++;
        commit_semantic_spm_producer_yield_after_kernel_enqueue();
        break;
    }

    case State::BUILT:
        TORCH_CHECK(false, "RpuKernelGraph: enqueue() called in BUILT state");
    }
}

// =============================================================================
// sync_point
// =============================================================================

void RpuKernelGraph::sync_point() {
    check_foreign_graph_execution_allowed(
        "RpuKernelGraph: foreign Graph sync_point");
    TORCH_CHECK(
        composite_fmb_invocation_.phase ==
            CompositeFmbInvocationState::Phase::None,
        "RpuKernelGraph: sync_point is forbidden while a composite FMB "
        "invocation is open");
    TORCH_CHECK(!pending_semantic_spm_producer_yield_.has_value() &&
                    !semantic_spm_producer_yield_poisoned_ &&
                    !(state_ == State::RECORDING &&
                      has_semantic_spm_producer_yield_nodes_),
                "RpuKernelGraph: sync_point is forbidden by semantic SPM "
                "producer-yield authority");
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "RpuKernelGraph: sync_point is forbidden by the typed "
                "DDR-register census");
    TORCH_CHECK(!canonical_dma_poisoned_ &&
                    !canonical_dma_build_trace_active_ &&
                    !canonical_dma_callback_active_ &&
                    !canonical_dma_burst_permit_.active,
                "RpuKernelGraph: sync_point is forbidden inside a poisoned "
                "or active canonical FMB DMA scope");
    if (state_ != State::RECORDING) {
        return;
    }

    // 如果 nodes_ 为空（例如 embedding CPU fallback 发生在第一个 RPU kernel 之前），
    // flush 是 no-op，也不需要标记 non-replayable。
    if (nodes_.empty()) {
        return;
    }

    // Eager boundary flush 让 CPU fallback 能读到最新 DDR 数据，
    // 但保持 RECORDING 状态继续录制后续 kernel。
    // 标记为 non-replayable，因为 sync_point 打断了连续 batch 的前提。
    //
    // §12.7 flush：这里同样做 pre + post 双 flush。原因与 end() 一致：
    // sync_point 之前本轮可能已有 CPU 产物需要先对 RPU 可见，sync_point 之后
    // CPU fallback 又要立刻读取 prefix kernel 的写回结果。tensor_refs 交给最终
    // end() 底部扫尾。
    flush_boundary_ptrs();
    execute_graph_oneshot();
    flush_boundary_ptrs();
    boundary_flush_ptrs_.clear();

    kernels_.clear();
    nodes_.clear();
    // 清理 pending kernel id/name：防御性修复，避免 sync_point 在
    // get_kernel 和 enqueue 之间被调用时留下悬空的 pending 状态。
    pending_kernel_id_.reset();
    pending_kernel_name_.reset();
    replayable_ = false;
}

// =============================================================================
// skip_op_stream_for_fast_replay — 2.3 P3 prototype
// =============================================================================
// 调用方契约见 graph_runtime.h 注释。把 cursor_ 推到 nodes_.size(),让 end()
// 的"replay incomplete"检查通过,直接走 execute_graph_for_replaying。

void RpuKernelGraph::skip_op_stream_for_fast_replay() {
    RECORD_FUNCTION("rpu_graph::skip_op_stream_for_fast_replay", {});
    TORCH_CHECK(state_ == State::REPLAYING,
                "skip_op_stream_for_fast_replay: must be in REPLAYING; "
                "state=", static_cast<int>(state_));
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "skip_op_stream_for_fast_replay is forbidden by the typed "
                "DDR-register census");
    TORCH_CHECK(!canonical_dma_poisoned_ &&
                    !canonical_dma_callback_active_ &&
                    !canonical_dma_burst_permit_.active,
                "skip_op_stream_for_fast_replay: canonical FMB DMA scope "
                "must be closed and unpoisoned");
    TORCH_CHECK(!has_semantic_spm_peer_nodes_ ||
                    semantic_spm_peer_replay_armed_,
                "skip_op_stream_for_fast_replay: typed SPM peer REPLAY "
                "requires an armed schema-v13 token");
    TORCH_CHECK(!has_semantic_dma_nodes_ ||
                    semantic_dma_replay_owner_armed_,
                "skip_op_stream_for_fast_replay: semantic DMA REPLAY "
                "requires an armed owning FMB token");
    TORCH_CHECK(!pending_semantic_spm_producer_yield_.has_value(),
                "skip_op_stream_for_fast_replay: semantic SPM producer-yield "
                "marker remains pending");
    TORCH_CHECK(
        !has_semantic_spm_producer_yield_nodes_,
        "skip_op_stream_for_fast_replay: a semantic SPM producer-yield "
        "Graph must replay its PostFn terminal writers ordinarily");
    cursor_ = nodes_.size();
    // This is a flow marker only.  A caller may already have replayed a kernel
    // prefix (HyViT patch embed is one example), in which case the independent
    // kernel_params_dirty_this_replay_ witness remains set.
    op_stream_fully_skipped_ = true;
}

// =============================================================================
// mark_post_fn_cursor / skip_layer_body_for_fast_replay — D-501 post_fn coexist
// =============================================================================
// Records the # nodes emitted before post_fn so fast-replay can skip ONLY the
// layer-body re-walk and leave post_fn's nodes ahead for the normal replay path
// to re-emit. See graph_runtime.h decl comments for the caller contract.

void RpuKernelGraph::mark_post_fn_cursor() {
    TORCH_CHECK(!pending_semantic_spm_producer_yield_.has_value(),
                "mark_post_fn_cursor is forbidden while a semantic SPM "
                "producer-yield marker is pending");
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "mark_post_fn_cursor is forbidden by the typed "
                "DDR-register census");
    if (state_ == State::RECORDING) {           // only meaningful while appending nodes
        post_fn_cursor_ = nodes_.size();         // # nodes emitted before post_fn
        has_post_fn_cursor_ = true;
    }
}

void RpuKernelGraph::skip_layer_body_for_fast_replay() {
    RECORD_FUNCTION("rpu_graph::skip_layer_body_for_fast_replay", {});
    TORCH_CHECK(state_ == State::REPLAYING,
                "skip_layer_body_for_fast_replay: must be in REPLAYING; state=",
                static_cast<int>(state_));
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "skip_layer_body_for_fast_replay is forbidden by the typed "
                "DDR-register census");
    TORCH_CHECK(!canonical_dma_poisoned_ &&
                    !canonical_dma_callback_active_ &&
                    !canonical_dma_burst_permit_.active,
                "skip_layer_body_for_fast_replay: canonical FMB DMA scope "
                "must be closed and unpoisoned");
    TORCH_CHECK(!has_semantic_spm_peer_nodes_ ||
                    semantic_spm_peer_replay_armed_,
                "skip_layer_body_for_fast_replay: typed SPM peer REPLAY "
                "requires an armed schema-v13 token");
    TORCH_CHECK(!has_semantic_dma_nodes_ ||
                    semantic_dma_replay_owner_armed_,
                "skip_layer_body_for_fast_replay: semantic DMA REPLAY "
                "requires an armed owning FMB token");
    TORCH_CHECK(!pending_semantic_spm_producer_yield_.has_value(),
                "skip_layer_body_for_fast_replay: semantic SPM "
                "producer-yield marker remains pending");
    verify_semantic_spm_producer_yield_replay_arm(
        "skip_layer_body_for_fast_replay");
    TORCH_CHECK(has_post_fn_cursor_,
                "skip_layer_body_for_fast_replay: post_fn cursor not recorded");
    TORCH_CHECK(post_fn_cursor_ <= nodes_.size(),
                "skip_layer_body_for_fast_replay: post_fn_cursor_ ", post_fn_cursor_,
                " > nodes_.size() ", nodes_.size());
    TORCH_CHECK(
        semantic_spm_producer_yield_id_digest_for_window(
            /*begin=*/0, post_fn_cursor_) == 0,
        "skip_layer_body_for_fast_replay: layer-body prefix contains a "
        "semantic SPM producer yield and must replay ordinarily");
    cursor_ = post_fn_cursor_;                    // leave the post_fn nodes ahead for re-emit
}

// 2.3.1 — fast replay fullscope. Apply RegisterPatch + DataPatch lists
// onto the captured kernels / nodes_ before advancing cursor, so the
// caller's g.end() drives execute_graph_for_replaying with this step's
// dev_addrs / scalar register values. Reuses the same patch helpers as
// `replay_prepared_child_with_data_patches` so semantics stay
// byte-identical to the ChildGraph path.
//
// Caller contract:
//   - state_ == REPLAYING (caller already entered via begin(sig)).
//   - Patches reference indices into the BUILT graph's kernels_ /
//     nodes_; out-of-range / wrong-kind patches abort the call with a
//     clear TORCH_CHECK message.
//   - DataPatch.field semantics mirror DataPatch in graph_signature.h:
//     0 = Memcpy.dst, 1 = Memcpy.src.
//
// Pybind exposes this as `skip_op_stream_with_patches`. Default
// behavior (no patches) is identical to skip_op_stream_for_fast_replay.

void RpuKernelGraph::skip_op_stream_with_patches(
        const std::vector<RegisterPatch>& register_patches,
        const std::vector<DataPatch>& data_patches) {
    RECORD_FUNCTION("rpu_graph::skip_op_stream_with_patches", {});
    TORCH_CHECK(state_ == State::REPLAYING,
                "skip_op_stream_with_patches: must be in REPLAYING; "
                "state=", static_cast<int>(state_));
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "skip_op_stream_with_patches is forbidden by the typed "
                "DDR-register census");
    TORCH_CHECK(!canonical_dma_poisoned_ &&
                    !canonical_dma_callback_active_ &&
                    !canonical_dma_burst_permit_.active,
                "skip_op_stream_with_patches: canonical FMB DMA scope must "
                "be closed and unpoisoned");
    TORCH_CHECK(!has_semantic_spm_peer_nodes_ ||
                    semantic_spm_peer_replay_armed_,
                "skip_op_stream_with_patches: typed SPM peer REPLAY "
                "requires an armed schema-v13 token");
    TORCH_CHECK(!has_semantic_dma_nodes_ ||
                    semantic_dma_replay_owner_armed_,
                "skip_op_stream_with_patches: semantic DMA REPLAY requires "
                "an armed owning FMB token");
    TORCH_CHECK(!pending_semantic_spm_producer_yield_.has_value(),
                "skip_op_stream_with_patches: semantic SPM producer-yield "
                "marker remains pending");
    TORCH_CHECK(
        !has_semantic_spm_producer_yield_nodes_,
        "skip_op_stream_with_patches: semantic SPM producer-yield terminal "
        "kernels cannot be patched or fully skipped");

    for (const auto& p : register_patches) {
        TORCH_CHECK(p.kernel_idx < kernels_.size(),
                    "skip_op_stream_with_patches: RegisterPatch.kernel_idx=",
                    p.kernel_idx, " out of range (kernels_.size=",
                    kernels_.size(), ")");
        kernel_params_dirty_this_replay_ = true;
        apply_register_patch_to_kernel(*this, p, kernels_[p.kernel_idx].get());
    }
    for (const auto& p : data_patches) {
        TORCH_CHECK(p.node_idx < nodes_.size(),
                    "skip_op_stream_with_patches: DataPatch.node_idx=",
                    p.node_idx, " out of range (nodes_.size=",
                    nodes_.size(), ")");
        TORCH_CHECK(nodes_[p.node_idx].kind == GraphNodeKind::Memcpy,
                    "skip_op_stream_with_patches: DataPatch.node_idx=",
                    p.node_idx, " is not a Memcpy node");
        TORCH_CHECK(p.field == 0 || p.field == 1,
                    "skip_op_stream_with_patches: DataPatch.field must be "
                    "0 (Memcpy.dst) or 1 (Memcpy.src); got ",
                    static_cast<int>(p.field));
        TORCH_CHECK(p.ptr != 0,
                    "skip_op_stream_with_patches: DataPatch.ptr must be "
                    "non-zero");
        auto& m = nodes_[p.node_idx].as_memcpy();
        void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(p.ptr));
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
    }

    cursor_ = nodes_.size();
}

// =============================================================================
// build_segments_from_nodes — 线性扫 nodes_:连续 Kernel/Dma/Barrier 节点在
// queue_state 兼容且 entries/kd/instr 安全预算允许时合并为 segment；其它
// data 节点作为 segment 边界。跨 stream DMA fence 未闭合时不在中间切段。
//
// 不按 core_ids 切段:SDK rhino_launch_queue.h doc 显式声明 batch 内可 mix
// 不同 core_ids 的 kernel (add_kernel(k1,_,{0,1}); add_kernel(k2,_,{0}) 同
// batch)。1-core kernel 拿到 R1_KCORE_USED_PARAM_ADDR=(1<<16|0) + reg21=1,
// 应当只 dispatch 到 core 0;8-core kernel 拿到 (8<<16|0) + reg21=8,8 核都
// 执行。seg.core_ids 覆盖所有 kernel core span 和 DMA engine 的 [0..extent-1] 前缀；
// DMA-first/only 段使用 owner 的 execution_core_count。它决定 Queue_t 资源域。
// =============================================================================

// 诊断 helper:读 RPU_P7_SPLIT_KERNELS env (comma-separated kernel-name
// substrings),把匹配 kernel 强制隔成 solo segment (queue 前缀覆盖 kn.core_ids,
// 独立 Queue_t)。用于 bisect "哪个 kernel 在 merge 段里破坏正确性"。空 env =
// 全部 merge (默认行为)。Substring 匹配 KERNEL_ID_NAMES[id] 全名 (如
// "binary_sameshape" / "im2col" / "pad_const" / "transpose") 或 dynamic
// kernel_name。空 token 跳过。在 build_segments_from_nodes 内调用,静态初始化
// 只读一次。
static bool p7_should_force_isolate_kernel(const KernelNodeData& kn) {
    static const std::vector<std::string> tokens = []() {
        std::vector<std::string> result;
        const char* e = std::getenv("RPU_P7_SPLIT_KERNELS");
        if (!e || !*e) return result;
        std::string env_val(e);
        size_t pos = 0;
        while (pos < env_val.size()) {
            size_t comma = env_val.find(',', pos);
            std::string tok = env_val.substr(
                pos, comma == std::string::npos ? std::string::npos
                                                : comma - pos);
            while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t'))
                tok.erase(tok.begin());
            while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t'))
                tok.pop_back();
            if (!tok.empty()) result.push_back(std::move(tok));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        return result;
    }();
    if (tokens.empty()) return false;

    std::string label;
    if (!kn.kernel_name.empty()) {
        label = kn.kernel_name;
    } else if (kn.kernel_id) {
        int idx = static_cast<int>(*kn.kernel_id);
        if (idx >= 0 && idx < static_cast<int>(KernelId::_COUNT)) {
            label = KERNEL_ID_NAMES[idx];
        } else {
            return false;
        }
    } else {
        return false;
    }
    for (const auto& t : tokens) {
        if (label.find(t) != std::string::npos) return true;
    }
    return false;
}

static std::string kernel_label_for_segment_split(const KernelNodeData& kn) {
    if (!kn.kernel_name.empty()) {
        return kn.kernel_name;
    }
    if (kn.kernel_id) {
        int idx = static_cast<int>(*kn.kernel_id);
        if (idx >= 0 && idx < static_cast<int>(KernelId::_COUNT)) {
            return KERNEL_ID_NAMES[idx];
        }
    }
    return {};
}

static bool siglip_should_isolate_patch_embed_kernel(
        const std::string& graph_label,
        const KernelNodeData& kn,
        bool enabled) {
    // 2026-06-04: default OFF. The 2026-05-23 P8 always-merge path already
    // handles mixed 1-core patch-embed + 8-core encoder kernels in one segment
    // — pi05/siglip (single + 3-image) can be fused into
    // ONE graph with no graph.end() crash. main re-added this per-kernel
    // isolation defensively when integrating its 3-image SigLIP C++; it splits
    // pi05_siglip_compute into ~14 solo segments. Opt back in with
    // RPU_SIGLIP_ISOLATE_PATCH_EMBED=1 only if a graph.end() crash reappears.
    if (!enabled) {
        return false;
    }
    if (graph_label != "pi05_siglip_compute") {
        return false;
    }
    const std::string label = kernel_label_for_segment_split(kn);
    if (label.empty()) {
        return false;
    }

    // SigLIP patch embedding mixes 1-core image prep kernels with the 8-core
    // encoder in one Python GraphCache scope. The SDK accepts mixed core_ids in
    // principle, but this graph was observed (in main's pre-P8 integration) to
    // crash at graph.end() when these kernels are merged into the same segment.
    return label.find("pad_const_NxC_NxCP") != std::string::npos ||
           label.find("transpose_ncb_c16") != std::string::npos ||
           label.find("im2col_coren") != std::string::npos ||
           label.find("_1core_") != std::string::npos;
}

// Keep replay batches comfortably below both the SDK's declared limits and a
// smaller execution envelope. Entry count alone is insufficient because
// instruction and kd footprints vary by kernel, so mirror Queue_t::build_batch's
// exact accounting here.
//
// The SDK defaults are 65536 entries / 8 MiB kd / 64 MiB instructions.  These
// budgets intentionally retain at least 2x headroom for firmware behavior not
// covered by build_batch's host-side checks.  If a process lowers an LKN limit,
// use half of that configured value as well.  A single indivisible node or
// fenced multi-stream DMA burst may exceed the soft budget; the SDK's hard
// build_batch guard remains the final authority for those cases.
struct SegmentResourceBudget {
    size_t entries;
    size_t kd_bytes;
    size_t instr_bytes;
};

struct SegmentResourceUse {
    size_t entries = 0;
    size_t kd_bytes = 16;  // Queue_t::build_batch final + end-marker packets.
    size_t instr_bytes = 0;
    uint16_t active_streams = 0;
};

static size_t segment_new_completion_bytes(const SegmentResourceUse& current,
                                           const SegmentResourceUse& next) {
    // Launch appends one completion SYNC for each active nonzero stream, even
    // if explicit fences already joined it. A repeated stream costs only once.
    uint16_t newly_active = next.active_streams & ~current.active_streams & 0xfffe;
    size_t bytes = 0;
    while (newly_active != 0) {
        bytes += sizeof(uint64_t);
        newly_active &= newly_active - 1;
    }
    return bytes;
}

static bool lingbot2_exact_graph_owns_sdk_budget(
        const std::string& graph_label,
        const SegmentResourceBudget& sdk_budget,
        bool multiview_spm_z2) {
    if (!multiview_spm_z2 ||
        sdk_budget.entries != 262144 ||
        sdk_budget.kd_bytes != (size_t{64} << 20) ||
        sdk_budget.instr_bytes != (size_t{512} << 20)) {
        return false;
    }
    return graph_label == "lingbot2_multiview_spm_z2_bootstrap" ||
           graph_label == "lingbot2_multiview_spm_z2_retained" ||
           graph_label == "lingbot2_denoise_unroll";
}

static SegmentResourceBudget segment_resource_budget(
        const std::string& graph_label,
        const GraphRuntimePolicy& runtime_policy) {
    constexpr size_t kMiB = 1u << 20;
    const size_t sdk_entries = runtime_policy.lkn_max_batch_entries;
    const size_t sdk_kd = runtime_policy.lkn_kd_buf_mb * kMiB;
    const size_t sdk_instr = runtime_policy.lkn_instr_buf_mb * kMiB;
    const SegmentResourceBudget sdk_budget{
        sdk_entries, sdk_kd, sdk_instr};
    // The exact LingBot profile that shipped as one SDK batch retains only its
    // sealed envelope. The legacy 27B text owner's retained decode also uses
    // its SDK envelope: splitting rebuilds all but the final batch on REPLAY.
    // The owner binds this cold policy before construction; graph labels alone
    // never enlarge a budget, and all three SDK resource ceilings still apply.
    if (lingbot2_exact_graph_owns_sdk_budget(
            graph_label, sdk_budget,
            runtime_policy.lingbot2_multiview_spm_z2) ||
        (runtime_policy.qwen35_legacy_27b_sdk_budget &&
         graph_label == "rpu_qwen3_5")) {
        return sdk_budget;
    }
    // Reduced-core execution uses a conservative per-segment entry budget.
    const size_t entry_ceiling =
        runtime_policy.execution_core_count < 8 ? 4096 : 8192;
    return {
        std::max<size_t>(1, std::min(entry_ceiling, sdk_entries / 2)),
        std::max<size_t>(16, std::min<size_t>(4 * kMiB, sdk_kd / 2)),
        std::max<size_t>(4096, std::min<size_t>(32 * kMiB, sdk_instr / 2)),
    };
}

static SegmentResourceUse segment_node_resources(
        const GraphNode& node,
        const std::vector<std::unique_ptr<::rhino_lkn::Kernel_t>>& kernels) {
    SegmentResourceUse use;
    use.entries = 1;
    use.kd_bytes = 0;
    switch (node.kind) {
    case GraphNodeKind::Kernel: {
        const auto& kn = node.as_kernel();
        TORCH_CHECK(kn.kernel_idx < kernels.size() && kernels[kn.kernel_idx],
                    "build_segments_from_nodes: invalid kernel index ",
                    kn.kernel_idx, " for ", kernels.size(), " kernels");
        ::rhino_lkn::KernelLaunchFootprint footprint{};
        TORCH_CHECK(
            kernels[kn.kernel_idx]->query_launch_footprint(&footprint) ==
                ::rhino_lkn::kKernelQueryOk,
            "build_segments_from_nodes: cannot query kernel launch footprint");
        TORCH_CHECK(
            footprint.command_reservation_bytes <=
                    std::numeric_limits<size_t>::max() &&
                footprint.code_reservation_bytes <=
                    std::numeric_limits<size_t>::max(),
            "build_segments_from_nodes: kernel launch footprint overflow");
        use.kd_bytes =
            static_cast<size_t>(footprint.command_reservation_bytes);
        use.instr_bytes =
            static_cast<size_t>(footprint.code_reservation_bytes);
        break;
    }
    case GraphNodeKind::Dma: {
        use.kd_bytes = 20 * sizeof(uint64_t);
        const auto channel = node.as_dma().channel;
        TORCH_CHECK(channel >= 0 && channel < 8,
                    "segment_node_resources: DMA channel must be in [0, 7]");
        use.active_streams = uint16_t{1} << (channel * 2);
        break;
    }
    case GraphNodeKind::Barrier:
        use.kd_bytes = sizeof(uint64_t);
        break;
    default:
        TORCH_CHECK(false,
                    "segment_node_resources called for non-batch node kind ",
                    static_cast<int>(node.kind));
    }
    return use;
}

static bool segment_would_exceed(const SegmentResourceUse& current,
                                 const SegmentResourceUse& next,
                                 const SegmentResourceBudget& budget) {
    const auto sum_exceeds = [](size_t lhs, size_t rhs, size_t limit) {
        return rhs > limit || lhs > limit - rhs;
    };
    const bool kd_exceeded =
        sum_exceeds(current.kd_bytes, next.kd_bytes, budget.kd_bytes);
    return sum_exceeds(current.entries, next.entries, budget.entries) ||
           kd_exceeded ||
           (!kd_exceeded && segment_new_completion_bytes(current, next) >
               budget.kd_bytes - current.kd_bytes - next.kd_bytes) ||
           sum_exceeds(current.instr_bytes, next.instr_bytes,
                       budget.instr_bytes);
}

void RpuKernelGraph::build_segments_from_nodes() {
    segments_.clear();
    last_stats_.segment_census.clear();
    last_stats_.dma_count = 0;
    last_stats_.barrier_count = 0;
    last_stats_.child_graph_count = 0;
    last_stats_.child_graph_lifetime_ids.clear();
    if (nodes_.empty()) return;

    // Validate the complete graph before forming any executable segment. Child
    // graphs retain private queues, so their resource policy must also fit.
    for (const auto& node : nodes_) {
        if (node.kind == GraphNodeKind::Kernel) {
            validated_kernel_core_extent(runtime_policy_, node.as_kernel().core_ids);
        } else if (node.kind == GraphNodeKind::Dma) {
            validate_graph_dma_channel(node.as_dma().channel,
                                       runtime_policy_.execution_core_count,
                                       "Graph BUILD/Dma");
        } else if (node.kind == GraphNodeKind::Memcpy) {
            const auto& copy = std::get<MemcpyNodeData>(node.data);
            if (copy.kind == CopyKind::DDR_TO_DDR) {
                validate_graph_dma_channel(
                    copy.dma_channel == -1 ? 0 : copy.dma_channel,
                    runtime_policy_.execution_core_count,
                    "Graph BUILD/DDR_TO_DDR");
            }
        } else if (node.kind == GraphNodeKind::ChildGraph) {
            const auto& child = node.as_child_graph().child;
            TORCH_CHECK(child, "Graph BUILD contains a null child graph");
            TORCH_CHECK(
                child->runtime_policy_.execution_core_count <=
                    runtime_policy_.execution_core_count,
                "Graph child execution_core_count exceeds parent budget");
        }
    }

    const GraphSignature* graph_signature = pending_signature_.has_value()
        ? &*pending_signature_
        : (has_built_signature_ ? &built_signature_ : nullptr);
    const std::string graph_label =
        (graph_signature != nullptr && !graph_signature->op_id_str.empty())
            ? graph_signature->op_id_str
            : std::string{};

    // 切段条件 (2026-05-23 P8 — always-merge):
    //   1. queue_state 变化 (BARRIER_NEXT / instr-pause 等差异)
    //   2. data nodes delimit segments, except Dma/Barrier and authenticated
    //      zero-entry COMPLETE manifest metadata (ordinary Branch still cuts).
    //   3. RPU_P7_SPLIT_KERNELS env 命中 — 把匹配 kernel 隔成 solo segment
    //      (诊断 only,bisect 单 kernel 正确性问题)
    //   4. SDK footprint 达到共享安全预算 — 在跨 stream DMA fence 完整闭合的
    //      边界贪心切段，避免大图虽 build_batch rc=0 却在硬件静默错算。
    constexpr size_t kNoOpenSeg = std::numeric_limits<size_t>::max();
    const SegmentResourceBudget resource_budget =
        segment_resource_budget(graph_label, runtime_policy_);
    size_t seg_start = kNoOpenSeg;
    uint8_t seg_max_num_cores = 0;
    QueueLaunchState seg_queue_state;
    SegmentResourceUse seg_resources;
    size_t resource_split_count = 0;
    size_t peak_entries = 0;
    size_t peak_kd_bytes = 0;
    size_t peak_instr_bytes = 0;
    // A non-zero stream is safe to split only after stream 0 has waited for it.
    // DMA wrappers encode a burst as pre barriers (stream N waits for 0), DMAs,
    // then post barriers (stream 0 waits for N).  Keeping this set non-empty
    // across that entire burst prevents an automatic cut inside its fence.
    std::vector<uint8_t> pending_streams;

    auto resource_reset = [&]() {
        seg_resources = SegmentResourceUse{};
        pending_streams.clear();
    };

    auto mark_pending_stream = [&](uint8_t stream) {
        if (stream == 0) return;
        if (std::find(pending_streams.begin(), pending_streams.end(), stream) ==
            pending_streams.end()) {
            pending_streams.push_back(stream);
        }
    };

    auto mark_joined_stream = [&](uint8_t stream) {
        pending_streams.erase(
            std::remove(pending_streams.begin(), pending_streams.end(), stream),
            pending_streams.end());
    };

    auto account_node = [&](const GraphNode& node,
                            const SegmentResourceUse& use) {
        seg_resources.entries += use.entries;
        seg_resources.kd_bytes +=
            use.kd_bytes + segment_new_completion_bytes(seg_resources, use);
        seg_resources.active_streams |= use.active_streams;
        seg_resources.instr_bytes += use.instr_bytes;
        if (node.kind == GraphNodeKind::Dma) {
            mark_pending_stream(static_cast<uint8_t>(node.as_dma().channel * 2));
        } else if (node.kind == GraphNodeKind::Barrier) {
            const auto& barrier = node.as_barrier_node();
            if (barrier.self_stream == 0) {
                mark_joined_stream(barrier.target_stream);
            } else if (barrier.target_stream == 0) {
                mark_pending_stream(barrier.self_stream);
            }
        }
    };

    auto close_segment = [&](size_t end_exclusive) {
        if (seg_start == kNoOpenSeg) return;
        Segment seg;
        seg.segment_id = segments_.size();
        seg.start_idx = seg_start;
        seg.end_idx = end_exclusive;
        const uint8_t n = seg_max_num_cores == 0
            ? static_cast<uint8_t>(runtime_policy_.execution_core_count)
            : seg_max_num_cores;
        seg.core_ids.resize(n);
        for (uint8_t k = 0; k < n; ++k) seg.core_ids[k] = k;
        seg.queue_state = seg_queue_state;
        seg.census.segment_id = seg.segment_id;
        seg.census.start_idx = seg.start_idx;
        seg.census.end_idx = seg.end_idx;
        seg.census.core_count = seg.core_ids.size();
        for (size_t j = seg.start_idx; j < seg.end_idx; ++j) {
            switch (nodes_[j].kind) {
            case GraphNodeKind::Kernel: ++seg.census.kernel_count; break;
            case GraphNodeKind::Dma: ++seg.census.dma_count; break;
            case GraphNodeKind::Barrier: ++seg.census.barrier_count; break;
            case GraphNodeKind::Branch:
                TORCH_CHECK(nodes_[j].as_branch().is_physical_manifest_metadata(),
                            "non-metadata Branch in segment census");
                break;
            default: TORCH_CHECK(false, "non-batch node in segment census");
            }
        }
        last_stats_.dma_count += seg.census.dma_count;
        last_stats_.barrier_count += seg.census.barrier_count;
        last_stats_.segment_census.push_back(seg.census);
        segments_.push_back(std::move(seg));
        peak_entries = std::max(peak_entries, seg_resources.entries);
        peak_kd_bytes = std::max(peak_kd_bytes, seg_resources.kd_bytes);
        peak_instr_bytes = std::max(peak_instr_bytes, seg_resources.instr_bytes);
        seg_start = kNoOpenSeg;
        seg_max_num_cores = 0;
        resource_reset();
    };

    for (size_t i = 0; i < nodes_.size(); ++i) {
        switch (nodes_[i].kind) {
        case GraphNodeKind::Kernel: {
            auto& kn = nodes_[i].as_kernel();
            const uint8_t kn_num =
                validated_kernel_core_extent(runtime_policy_, kn.core_ids);
            if (p7_should_force_isolate_kernel(kn) ||
                siglip_should_isolate_patch_embed_kernel(
                    graph_label, kn,
                    runtime_policy_.siglip_isolate_patch_embed)) {
                close_segment(i);
                seg_start = i;
                seg_max_num_cores = kn_num;
                seg_queue_state = kn.queue_state;
                account_node(nodes_[i], segment_node_resources(nodes_[i], kernels_));
                close_segment(i + 1);
                break;
            }

            const SegmentResourceUse node_resources =
                segment_node_resources(nodes_[i], kernels_);
            if (seg_start != kNoOpenSeg && pending_streams.empty() &&
                segment_would_exceed(seg_resources, node_resources,
                                     resource_budget)) {
                close_segment(i);
                ++resource_split_count;
            }

            if (seg_start == kNoOpenSeg) {
                seg_start = i;
                seg_max_num_cores = kn_num;
                seg_queue_state = kn.queue_state;
            } else if (seg_queue_state != kn.queue_state) {
                close_segment(i);
                seg_start = i;
                seg_max_num_cores = kn_num;
                seg_queue_state = kn.queue_state;
            } else if (kn_num > seg_max_num_cores) {
                seg_max_num_cores = kn_num;
            }
            account_node(nodes_[i], node_resources);
            // else: extend current segment
            break;
        }
        case GraphNodeKind::Dma:
        case GraphNodeKind::Barrier: {
            const SegmentResourceUse node_resources =
                segment_node_resources(nodes_[i], kernels_);
            if (seg_start != kNoOpenSeg && pending_streams.empty() &&
                segment_would_exceed(seg_resources, node_resources,
                                     resource_budget)) {
                close_segment(i);
                ++resource_split_count;
            }
            // DMA engines share the driver's allocated core prefix. Endpoints
            // retain the physical eight-controller memory layout, but a DMA
            // channel must also fit this segment's actual queue resource span.
            if (seg_start == kNoOpenSeg) {
                seg_start = i;
                seg_max_num_cores =
                    static_cast<uint8_t>(runtime_policy_.execution_core_count);
                seg_queue_state = QueueLaunchState{};
            }
            if (nodes_[i].kind == GraphNodeKind::Dma) {
                seg_max_num_cores = std::max<uint8_t>(
                    seg_max_num_cores,
                    static_cast<uint8_t>(nodes_[i].as_dma().channel + 1));
            }
            account_node(nodes_[i], node_resources);
            // else: extend current segment
            break;
        }
        case GraphNodeKind::Branch:
            if (nodes_[i].as_branch().is_physical_manifest_metadata()) {
                // Do not create an empty segment, change queue state, consume
                // SDK budget, or close an outstanding cross-stream fence.
                break;
            }
            [[fallthrough]];
        default:
            // 其它 data node (Memcpy / Memset / ChildGraph / Branch /
            // HostCallback / Tier3Oneshot) 按段边界处理,跟原 v3.0 语义一致。
            if (nodes_[i].kind == GraphNodeKind::ChildGraph) {
                const auto& child = nodes_[i].as_child_graph().child;
                TORCH_CHECK(child, "Graph BUILD contains a null child graph");
                ++last_stats_.child_graph_count;
                last_stats_.child_graph_lifetime_ids.push_back(child->graph_lifetime_id_);
            }
            close_segment(i);
            break;
        }
    }
    close_segment(nodes_.size());

    if (resource_split_count > 0 && log_at(3)) {
        std::cerr << "[GRAPH-INSTR] AUTO-SPLIT "
                  << (graph_label.empty() ? std::string("<unlabeled>") : graph_label)
                  << ": cuts=" << resource_split_count
                  << " segments=" << segments_.size()
                  << " peak=[entries=" << peak_entries
                  << " kd=" << peak_kd_bytes
                  << " instr=" << peak_instr_bytes << "]"
                  << " budget=[entries=" << resource_budget.entries
                  << " kd=" << resource_budget.kd_bytes
                  << " instr=" << resource_budget.instr_bytes << "]\n";
    }

    // HALO_NODE_HIST emits a node histogram by kind and kernel label before
    // batch submission, so diagnostics remain available if preparation fails.
    if (std::getenv("HALO_NODE_HIST")) {
        std::map<std::string, int> hist;
        int n_kernel = 0, n_dma = 0, n_barrier = 0, n_other = 0;
        for (auto& nd : nodes_) {
            if (nd.kind == GraphNodeKind::Kernel) {
                ++n_kernel;
                std::string lbl = kernel_label_for_segment_split(nd.as_kernel());
                hist[lbl.empty() ? std::string("<unnamed-kernel>") : lbl]++;
            } else if (nd.kind == GraphNodeKind::Dma) {
                ++n_dma;
            } else if (nd.kind == GraphNodeKind::Barrier) {
                ++n_barrier;
            } else {
                ++n_other;
            }
        }
        fprintf(stderr,
                "[NODE_HIST] total=%zu kernel=%d dma=%d barrier=%d other=%d segments=%zu\n",
                nodes_.size(), n_kernel, n_dma, n_barrier, n_other, segments_.size());
        std::vector<std::pair<std::string, int>> v(hist.begin(), hist.end());
        std::sort(v.begin(), v.end(),
                  [](const std::pair<std::string,int>& a,
                     const std::pair<std::string,int>& b) { return a.second > b.second; });
        for (auto& p : v) {
            fprintf(stderr, "[NODE_HIST]   %-44s %d\n", p.first.c_str(), p.second);
        }
    }
}

// =============================================================================
// segment queue preparation / launch
// =============================================================================

namespace {

// Queue_t allocates its batch arena lazily at build_batch(). Keep the
// incremental retained-batch cost bounded across every GraphCache in the
// process; slot 0 is the pre-existing per-entry fallback and is not counted.
// Extra per-segment prepared queues are a performance-only optimization. Keep
// every Graph entry on its single private queue: the exact LingBot2 envelope
// proved that retaining the optional second queue across many entries can make
// a later replay observe stale data, while slot-0 rebuild is correct.
constexpr size_t kExtraPreparedQueueBudget = 0;
std::atomic<size_t> g_extra_prepared_queues{0};

bool try_acquire_extra_prepared_queue() {
    size_t used = g_extra_prepared_queues.load(std::memory_order_relaxed);
    while (used < kExtraPreparedQueueBudget) {
        if (g_extra_prepared_queues.compare_exchange_weak(
                used, used + 1, std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

void release_extra_prepared_queue() noexcept {
    size_t used = g_extra_prepared_queues.load(std::memory_order_relaxed);
    while (used != 0 && !g_extra_prepared_queues.compare_exchange_weak(
               used, used - 1, std::memory_order_acq_rel,
               std::memory_order_relaxed)) {}
}

uint64_t checked_dma_live_address(
        uint64_t base, int64_t signed_offset,
        size_t node_idx, const char* where) {
    if (signed_offset >= 0) {
        const uint64_t offset = static_cast<uint64_t>(signed_offset);
        TORCH_CHECK(
            base <= std::numeric_limits<uint64_t>::max() - offset,
            where, ": mutable DMA address overflows at node ", node_idx,
            " (base=", base, " offset=", signed_offset, ")");
        return base + offset;
    }
    // Avoid negating INT64_MIN in the signed domain.
    const uint64_t magnitude =
        static_cast<uint64_t>(-(signed_offset + 1)) + 1;
    TORCH_CHECK(
        base >= magnitude,
        where, ": mutable DMA address underflows at node ", node_idx,
        " (base=", base, " offset=", signed_offset, ")");
    return base - magnitude;
}

}  // namespace

void RpuKernelGraph::release_prepared_queues() noexcept {
    for (auto& slot : private_queue_slots_) {
        slot.queue.reset();
        slot.core_num = 0;
        slot.built_segment_idx = -1;
        slot.hw_perf_enabled_at_build = false;
        slot.fixed_dma_table.reset();
        slot.invalidate_mutable_dma_packets();
        slot.kernel_register_tokens.clear();
        slot.pending_kernel_register_tokens.clear();
        slot.kernel_register_tokens_valid = false;
        slot.promotion_denied = false;
        if (slot.consumes_extra_budget) {
            slot.consumes_extra_budget = false;
            release_extra_prepared_queue();
        }
    }
}

size_t RpuKernelGraph::prepared_queue_slot_index(
        size_t segment_idx,
        bool retain) {
    TORCH_CHECK(segment_idx < segments_.size(),
                "prepared queue segment index out of range: ", segment_idx,
                " >= ", segments_.size());
    if (!retain || segment_idx == 0 ||
        segment_idx >= private_queue_slots_.size()) {
        return 0;
    }
    auto& slot = private_queue_slots_[segment_idx];
    if (slot.promotion_denied) return 0;
    if (!slot.consumes_extra_budget) {
        if (!try_acquire_extra_prepared_queue()) {
            slot.promotion_denied = true;
            return 0;
        }
        slot.consumes_extra_budget = true;
    }
    return segment_idx;
}

::rhino_lkn::Queue_t* RpuKernelGraph::ensure_private_queue(
        size_t queue_slot_idx,
        size_t num_cores) {
    TORCH_CHECK(num_cores >= 1 &&
                    num_cores <= static_cast<size_t>(runtime_policy_.execution_core_count),
                "Graph private queue core count exceeds execution_core_count");
    TORCH_CHECK(queue_slot_idx < private_queue_slots_.size(),
                "prepared queue slot index out of range: ", queue_slot_idx,
                " >= ", private_queue_slots_.size());
    auto& slot = private_queue_slots_[queue_slot_idx];
    const uint8_t want = static_cast<uint8_t>(num_cores);
    if (!slot.queue || slot.core_num != want) {
        // 最大 core 域变化时丢掉旧 queue 重建(RAII)。段内混核不走
        // 本分支：queue 按最大域构造，每个 kernel 的 core_ids 在
        // prepare_segment_queue 中独立传给 SDK。重建只丢失 kd_buf reuse
        // 收益，不影响正确性。
        slot.queue = std::make_unique<::rhino_lkn::Queue_t>(num_cores);
        slot.core_num = want;
        slot.built_segment_idx = -1;
        slot.hw_perf_enabled_at_build = false;
        slot.fixed_dma_table.reset();
        slot.invalidate_mutable_dma_packets();
        slot.kernel_register_tokens.clear();
        slot.pending_kernel_register_tokens.clear();
        slot.kernel_register_tokens_valid = false;
    }
    return slot.queue.get();
}

uint32_t RpuKernelGraph::prepare_segment_queue(
        Segment& seg,
        ::rhino_lkn::Queue_t& wq,
        bool hw_perf_enabled,
        RpuDmaSubmissionLease& submission_lease,
        bool retain_replay_state) {
    RECORD_FUNCTION("rpu_graph::prepare_segment_queue", {});
    // Validate the whole segment before appending SDK work. This also covers
    // fallback rebuilds and a compute-first segment with later DMA engines.
    for (size_t i = seg.start_idx; i < seg.end_idx; ++i) {
        if (nodes_[i].kind == GraphNodeKind::Dma) {
            validate_graph_dma_channel(nodes_[i].as_dma().channel,
                                       seg.core_ids.size(),
                                       "prepare_segment_queue");
        } else if (nodes_[i].kind == GraphNodeKind::Kernel) {
            TORCH_CHECK(validated_kernel_core_extent(runtime_policy_, nodes_[i].as_kernel().core_ids) <=
                            seg.core_ids.size(),
                        "Graph kernel exceeds the segment queue's core prefix");
        }
    }
    // P2 改造:
    // - Kernel 走 add_kernel_mutable (regs 后续可通过 sync_mutable_params 重刷)。
    // - Mutable DMA 走 add_dma_kernel_mutable (返回的 slot 索引按 SDK 内部
    //   sequential 计数,与 next_mutable_dma_id 一致),同时 push 一条
    //   PreparedMutableDmaSlot 进 seg.mutable_dmas 供 sync-only REPLAY 用。
    // - Fixed DMA / Barrier 走非 mutable 入口,**不**增 dma_id (与 SDK 一致)。
    // BUILD 结尾的 build_batch() 把 kd_buf 落定;后续若调 sync_mutable_params +
    // enqueu_batch 即等价于 BATCH_CTX 时代的 prepare-once-launch-many 快路径。
    // Build sidecars transactionally.  The retained vectors are ownership
    // witnesses for the last successful prepared batch; clearing them before a
    // failed rebuild would let a later same-address allocation erase an ABA
    // mismatch simply by forcing the prepare path.
    struct PreparedDmaResolution {
        size_t node_idx = 0;
        uint64_t src_addr = 0;
        uint64_t dst_addr = 0;
        RpuDmaEndpoint src_endpoint;
        RpuDmaEndpoint dst_endpoint;
    };
    std::vector<PreparedDmaResolution> dma_resolutions;
    std::vector<RpuDeviceDmaEndpointRequest> endpoint_requests;
    dma_resolutions.reserve(seg.census.dma_count);
    endpoint_requests.reserve(seg.census.dma_count * 2);

    // Resolve the complete DMA envelope before touching Queue_t.  Besides
    // reducing thousands of registry lock acquisitions to one per segment,
    // this makes a stale final endpoint fail before an earlier kernel or DMA
    // has been appended to the SDK batch.
    for (size_t i = seg.start_idx; i < seg.end_idx; ++i) {
        auto& node = nodes_[i];
        switch (node.kind) {
        case GraphNodeKind::Kernel:
        case GraphNodeKind::Barrier:
            break;
        case GraphNodeKind::Branch:
            TORCH_CHECK(node.as_branch().is_physical_manifest_metadata(),
                        "prepare_segment_queue: non-metadata Branch in segment");
            break;
        case GraphNodeKind::Dma: {
            auto& dma = node.as_dma();
            validate_semantic_dma_owner_for_execution(
                dma, i, "prepare_segment_queue");
            TORCH_CHECK(
                dma.variant == DmaNodeData::Variant::Fixed ||
                    dma.variant == DmaNodeData::Variant::MutableSrc ||
                    dma.variant == DmaNodeData::Variant::MutableDst,
                "prepare_segment_queue: invalid DMA variant at node ", i);
            TORCH_CHECK(dma.channel < 8,
                        "prepare_segment_queue: DMA channel out of range at "
                        "node ", i, ": ", static_cast<int>(dma.channel));
            uint64_t src = dma.src_addr;
            uint64_t dst = dma.dst_addr;
            if (dma.variant == DmaNodeData::Variant::MutableSrc) {
                TORCH_CHECK(dma.live_base != nullptr,
                            "prepare_segment_queue: MutableSrc DMA missing "
                            "live_base at node ", i);
                src = checked_dma_live_address(
                    *dma.live_base, dma.live_offset, i,
                    "prepare_segment_queue/MutableSrc");
            } else if (dma.variant == DmaNodeData::Variant::MutableDst) {
                TORCH_CHECK(dma.live_base != nullptr,
                            "prepare_segment_queue: MutableDst DMA missing "
                            "live_base at node ", i);
                dst = checked_dma_live_address(
                    *dma.live_base, dma.live_offset, i,
                    "prepare_segment_queue/MutableDst");
            }
            dma_resolutions.push_back(PreparedDmaResolution{
                /*node_idx=*/i, src, dst, {}, {}});
            endpoint_requests.push_back(RpuDeviceDmaEndpointRequest{
                src, dma.bytes, "prepare_segment_queue/src"});
            endpoint_requests.push_back(RpuDeviceDmaEndpointRequest{
                dst, dma.bytes, "prepare_segment_queue/dst"});
            break;
        }
        default:
            TORCH_CHECK(false,
                        "prepare_segment_queue: unexpected node kind ",
                        static_cast<int>(node.kind), " in segment at node ",
                        i, " (only Kernel / Dma / Barrier allowed)");
        }
    }

    std::vector<RpuDmaEndpoint> endpoints(endpoint_requests.size());
    auto resolved_endpoint_lease = rpu_resolve_device_dma_endpoints(
        endpoint_requests.data(), endpoint_requests.size(), endpoints.data());
    for (size_t i = 0; i < dma_resolutions.size(); ++i) {
        dma_resolutions[i].src_endpoint = endpoints[i * 2];
        dma_resolutions[i].dst_endpoint = endpoints[i * 2 + 1];
    }

    std::vector<Segment::PreparedMutableDmaSlot> prepared_mutable_dmas;
    std::shared_ptr<Segment::PreparedFixedDmaTable> prepared_fixed_table;
    // These sidecars are consumed only by a later REPLAY. One-shot submission
    // needs the same complete endpoint preflight and lease above, but discards
    // its segments after the synchronous launch.
    if (retain_replay_state) {
        prepared_fixed_table =
            std::make_shared<Segment::PreparedFixedDmaTable>();
        prepared_mutable_dmas.reserve(seg.census.dma_count);
        prepared_fixed_table->bindings.reserve(seg.census.dma_count);
    }
    uint32_t next_mutable_dma_id = 0;
    size_t next_dma_resolution = 0;
    // Instrumentation is baked into the batch. Snapshot once per execution
    // and set both true and false explicitly for a reused queue.
    wq.set_enable_hw_perf(hw_perf_enabled);
    wq.set_broadcast_mode(seg.queue_state.broadcast_mode);
    wq.set_flush_icache(seg.queue_state.flush_icache);
    for (size_t i = seg.start_idx; i < seg.end_idx; ++i) {
        auto& node = nodes_[i];
        switch (node.kind) {
        case GraphNodeKind::Kernel: {
            const auto& kn = node.as_kernel();
            // P7-redo:per-kernel core_ids 落到 SDK,实现段内 mix 1-core/8-core
            // kernel。SDK r1_queue.cc:1119-1120 把 (kn.core_ids.size()<<16 |
            // kn.core_ids[0]) 写进每 kernel 的 R1_KCORE_USED_PARAM_ADDR;
            // 1-core ROPE 拿到 (1<<16|0) + reg21=1,只 dispatch 到 core 0,
            // 不浪费 8 核冗余 fetch(对比 BATCH_CTX 时代的强制 8 核 dispatch)。
            rpu_add_kernel_mutable_checked(wq, *kernels_[kn.kernel_idx],
                                           kn.grid_dims, kn.core_ids,
                                           "prepare_segment_queue/Kernel");
            break;
        }
        case GraphNodeKind::Dma: {
            auto& dma = node.as_dma();
            TORCH_CHECK(next_dma_resolution < dma_resolutions.size() &&
                            dma_resolutions[next_dma_resolution].node_idx == i,
                        "prepare_segment_queue: DMA resolution order drift at "
                        "node ", i);
            const auto& resolved = dma_resolutions[next_dma_resolution++];
            if (dma.variant == DmaNodeData::Variant::MutableSrc) {
                rpu_add_dma_mutable_checked(
                    wq, resolved.src_endpoint, resolved.dst_endpoint,
                    dma.bytes, dma.channel,
                    "prepare_segment_queue/MutableSrc");
                if (retain_replay_state) prepared_mutable_dmas.push_back(
                    Segment::PreparedMutableDmaSlot{
                    /*node_idx*/   i,
                    /*dma_id*/     next_mutable_dma_id++,
                    Segment::PreparedMutableDmaSlot::Kind::MutableSrc,
                    /*live_base*/  dma.live_base,
                    /*live_offset*/dma.live_offset,
                    /*fixed_addr*/  resolved.dst_addr,
                    /*fixed_allocation_id*/
                        resolved.dst_endpoint.allocation_id,
                    /*bytes*/      dma.bytes,
                    /*channel*/    dma.channel,
                });
            } else if (dma.variant == DmaNodeData::Variant::MutableDst) {
                rpu_add_dma_mutable_checked(
                    wq, resolved.src_endpoint, resolved.dst_endpoint,
                    dma.bytes, dma.channel,
                    "prepare_segment_queue/MutableDst");
                if (retain_replay_state) prepared_mutable_dmas.push_back(
                    Segment::PreparedMutableDmaSlot{
                    /*node_idx*/   i,
                    /*dma_id*/     next_mutable_dma_id++,
                    Segment::PreparedMutableDmaSlot::Kind::MutableDst,
                    /*live_base*/  dma.live_base,
                    /*live_offset*/dma.live_offset,
                    /*fixed_addr*/  resolved.src_addr,
                    /*fixed_allocation_id*/
                        resolved.src_endpoint.allocation_id,
                    /*bytes*/      dma.bytes,
                    /*channel*/    dma.channel,
                });
            } else {
                // Fixed:non-mutable 入口,不增 dma_id (SDK 内部计数也只对
                // add_dma_kernel_mutable 递增)。
                rpu_add_dma_checked(wq, resolved.src_endpoint,
                                    resolved.dst_endpoint, dma.bytes,
                                    dma.channel,
                                    "prepare_segment_queue/Fixed");
                if (retain_replay_state) {
                    prepared_fixed_table->bindings.push_back(
                        Segment::PreparedFixedDmaSlot{
                        i, resolved.src_addr, resolved.dst_addr,
                        resolved.src_endpoint.allocation_id,
                        resolved.dst_endpoint.allocation_id,
                        dma.bytes, dma.channel});
                    if (resolved.src_endpoint.allocation_id != 0) {
                        prepared_fixed_table->witnesses.push_back({
                            resolved.src_endpoint.allocation_id, resolved.src_addr,
                            dma.bytes,
                            "launch_segment_for_replay/Fixed/src identity"});
                    }
                    if (resolved.dst_endpoint.allocation_id != 0) {
                        prepared_fixed_table->witnesses.push_back({
                            resolved.dst_endpoint.allocation_id, resolved.dst_addr,
                            dma.bytes,
                            "launch_segment_for_replay/Fixed/dst identity"});
                    }
                }
            }
            break;
        }
        case GraphNodeKind::Barrier: {
            const auto& b = node.as_barrier_node();
            wq.add_barrier(b.self_stream, b.target_stream);
            break;
        }
        case GraphNodeKind::Branch:
            TORCH_CHECK(node.as_branch().is_physical_manifest_metadata(),
                        "prepare_segment_queue: metadata identity changed before emission");
            break;
        default:
            TORCH_CHECK(false,
                        "prepare_segment_queue: validated segment topology "
                        "changed before emission at node ", i);
        }
    }
    TORCH_CHECK(next_dma_resolution == dma_resolutions.size(),
                "prepare_segment_queue: DMA resolution count drift");
    if (retain_replay_state) {
        coalesce_fixed_dma_allocation_witnesses(prepared_fixed_table->witnesses);
    }
    // Every caller checks this status; only optional retained slots can retry
    // through the fallback queue after releasing their arena budget.  Publish
    // the new ownership witnesses only for a complete prepared batch.
    const uint32_t rc = wq.build_batch();
    if (rc == 0) {
        for (const auto& resolved : dma_resolutions) {
            auto& dma = nodes_[resolved.node_idx].as_dma();
            dma.src_endpoint = resolved.src_endpoint;
            dma.dst_endpoint = resolved.dst_endpoint;
        }
        seg.mutable_dmas = std::move(prepared_mutable_dmas);
        seg.fixed_dma_table = std::move(prepared_fixed_table);
        // The caller submits the prepared Queue_t after this helper returns.
        // Move the registry lease out only for a complete batch so every raw
        // Buffer_t endpoint remains live through enqueu_batch(wait_finish=true).
        submission_lease = std::move(resolved_endpoint_lease);
    }
    return rc;
}

::rhino_lkn::Queue_t* RpuKernelGraph::launch_segment_for_replay(
        Segment& seg,
        bool hw_perf_enabled) {
    RECORD_FUNCTION("rpu_graph::segment_launch", {});

    // segment fingerprint = (segments_ 中的指针位置,core_num)。后续判定 sync-only
    // 资格,以及为 fallback 路径写新值时用。指针算术安全:segments_ 在 BUILT 期
    // 不重新分配,seg 始终是 segments_[idx]。
    const ssize_t seg_idx = static_cast<ssize_t>(&seg - segments_.data());

    struct FixedDmaPreflight {
        bool addresses_match = false;
        RpuDmaSubmissionLease submission_lease;
    };
    const auto fixed_dmas_match = [&] {
        if (!seg.fixed_dma_table) return FixedDmaPreflight{};
        const auto& bindings = seg.fixed_dma_table->bindings;
        bool addresses_match = true;
        for (const auto& binding : bindings) {
            TORCH_CHECK(binding.node_idx < nodes_.size() &&
                            nodes_[binding.node_idx].kind == GraphNodeKind::Dma,
                        "launch_segment_for_replay: fixed DMA sidecar is stale "
                        "at node ", binding.node_idx);
            const auto& dma = nodes_[binding.node_idx].as_dma();
            TORCH_CHECK(
                dma.variant == DmaNodeData::Variant::Fixed &&
                    dma.bytes == binding.bytes &&
                    dma.channel == binding.channel,
                "launch_segment_for_replay: fixed DMA sidecar topology drift "
                "at node ", binding.node_idx);
            addresses_match = addresses_match &&
                dma.src_addr == binding.src_addr &&
                dma.dst_addr == binding.dst_addr;
        }
        if (addresses_match) {
            // The entire binding table still matches. Reuse its immutable
            // witness values, but reacquire live allocation authority and the
            // submission lease each time; address equality alone is not safe.
            auto lease = rpu_validate_live_dma_allocation_witnesses(
                seg.fixed_dma_table->witnesses.data(),
                seg.fixed_dma_table->witnesses.size());
            return FixedDmaPreflight{true, std::move(lease)};
        }
        // Address movement requires only the unchanged endpoints to retain
        // their old identities. Do not validate stale addresses of deliberately
        // moved endpoints, and do not stop before a later unchanged ABA.
        std::vector<RpuDmaAllocationWitness> witnesses;
        witnesses.reserve(bindings.size() * 2);
        for (const auto& binding : bindings) {
            const auto& dma = nodes_[binding.node_idx].as_dma();
            if (dma.src_addr == binding.src_addr) {
                if (binding.src_allocation_id != 0) {
                    witnesses.push_back(RpuDmaAllocationWitness{
                        binding.src_allocation_id, binding.src_addr,
                        binding.bytes,
                        "launch_segment_for_replay/Fixed/src identity"});
                }
            }
            if (dma.dst_addr == binding.dst_addr) {
                if (binding.dst_allocation_id != 0) {
                    witnesses.push_back(RpuDmaAllocationWitness{
                        binding.dst_allocation_id, binding.dst_addr,
                        binding.bytes,
                        "launch_segment_for_replay/Fixed/dst identity"});
                }
            }
        }
        // Keep scanning after an address mismatch: a later unchanged numeric
        // address can still carry a released/reused logical allocation.
        auto submission_lease = rpu_validate_live_dma_allocation_witnesses(
            witnesses.data(), witnesses.size());
        return FixedDmaPreflight{
            addresses_match, std::move(submission_lease)};
    };
    struct MutableRebuildPreflight {
        RpuDmaSubmissionLease witness_lease;
        RpuDmaSubmissionLease moved_endpoint_lease;
    };
    const auto validate_mutable_fixed_sidecars_for_rebuild = [&] {
        // A rebuild is not an ownership reset: accepting a same-address
        // allocation with a new identity would erase the only witness that the
        // old Storage was released. Legacy descriptors may still move to a
        // different fixed address; a successful prepare records that baseline.
        std::vector<RpuDmaAllocationWitness> witnesses;
        std::vector<RpuDeviceDmaEndpointRequest> moved_requests;
        witnesses.reserve(seg.mutable_dmas.size());
        moved_requests.reserve(seg.mutable_dmas.size());
        for (const auto& binding : seg.mutable_dmas) {
            TORCH_CHECK(
                binding.node_idx < nodes_.size() &&
                    nodes_[binding.node_idx].kind == GraphNodeKind::Dma,
                "launch_segment_for_replay: mutable DMA sidecar is stale at node ",
                binding.node_idx);
            const auto& dma = nodes_[binding.node_idx].as_dma();
            const auto expected_variant =
                binding.kind == Segment::PreparedMutableDmaSlot::Kind::MutableSrc
                ? DmaNodeData::Variant::MutableSrc
                : DmaNodeData::Variant::MutableDst;
            TORCH_CHECK(
                dma.variant == expected_variant && dma.bytes == binding.bytes &&
                    dma.channel == binding.channel,
                "launch_segment_for_replay: mutable DMA sidecar topology drift "
                "at node ", binding.node_idx);
            const uint64_t fixed_addr =
                binding.kind == Segment::PreparedMutableDmaSlot::Kind::MutableSrc
                ? dma.dst_addr : dma.src_addr;
            TORCH_CHECK(
                dma.semantic_endpoint_id == 0 || fixed_addr == binding.fixed_addr,
                "launch_segment_for_replay: semantic mutable DMA fixed endpoint "
                "address drift at node ", binding.node_idx);
            const char* where =
                binding.kind == Segment::PreparedMutableDmaSlot::Kind::MutableSrc
                ? "launch_segment_for_replay/MutableSrc/dst identity preflight"
                : "launch_segment_for_replay/MutableDst/src identity preflight";
            if (fixed_addr == binding.fixed_addr) {
                witnesses.push_back(RpuDmaAllocationWitness{
                    binding.fixed_allocation_id, fixed_addr,
                    binding.bytes, where});
            } else {
                // Legacy descriptors may deliberately move their fixed side.
                // Resolve every moved range before prepare mutates Queue_t.
                moved_requests.push_back(RpuDeviceDmaEndpointRequest{
                    fixed_addr, binding.bytes, where});
            }
        }
        auto witness_lease = rpu_validate_live_dma_allocation_witnesses(
            witnesses.data(), witnesses.size());
        std::vector<RpuDmaEndpoint> moved_endpoints(moved_requests.size());
        auto moved_endpoint_lease = rpu_resolve_device_dma_endpoints(
            moved_requests.data(), moved_requests.size(),
            moved_endpoints.data());
        return MutableRebuildPreflight{
            std::move(witness_lease), std::move(moved_endpoint_lease)};
    };
    auto fixed_dma_preflight = fixed_dmas_match();
    const bool segment_fixed_dmas_match =
        fixed_dma_preflight.addresses_match;
    const size_t segment_idx = static_cast<size_t>(seg_idx);
    const bool already_retained = segment_idx != 0 &&
        segment_idx < private_queue_slots_.size() &&
        private_queue_slots_[segment_idx].consumes_extra_budget;
    size_t slot_idx = prepared_queue_slot_index(
        segment_idx, segment_fixed_dmas_match || already_retained);
    PreparedQueueSlot* slot = &private_queue_slots_[slot_idx];
    const auto discard_slot_batch = [](PreparedQueueSlot& target) noexcept {
        target.queue.reset();
        target.core_num = 0;
        target.built_segment_idx = -1;
        target.hw_perf_enabled_at_build = false;
        target.fixed_dma_table.reset();
        target.invalidate_mutable_dma_packets();
        target.kernel_register_tokens.clear();
        target.pending_kernel_register_tokens.clear();
        target.kernel_register_tokens_valid = false;
    };

    // Sync-only fast path:本 segment 的 kd_buf 仍在对应 queue slot 里
    // (前一次 build_batch 之后没有其他 segment 覆写过) → 跳过 prepare_segment_queue
    // 的 add_kernel_mutable + add_dma_kernel_mutable + build_batch 整段重建,
    // 校验本轮 DMA 后仅更新变化的 packet，再同步 kernel 参数并提交。
    // 多段 graph 的首次 replay 会在有界预算内逐段 prepare；后续
    // replay 稳定命中。预算耗尽的段回落 slot 0 并保持旧行为。
    // A shared immutable table identifies the prepared packet bindings without
    // comparing a second cold copy. Actual nodes and live allocations were
    // checked above; table identity never substitutes for those checks.
    if (segment_fixed_dmas_match &&
        slot->fixed_dma_table &&
        slot->fixed_dma_table == seg.fixed_dma_table &&
        slot->queue &&
        slot->built_segment_idx == seg_idx &&
        slot->hw_perf_enabled_at_build == hw_perf_enabled &&
        slot->core_num == static_cast<uint8_t>(seg.core_ids.size())) {
        RECORD_FUNCTION("rpu_graph::segment_launch_sync_only", {});
        const auto invalidate_failed_queue = [&] {
            // update_dma_kernel mutates kd_buf in place.  If a later update,
            // sync, or submit fails, discard the Queue_t so a partial packet
            // set can never be reused by the fallback prepare path.
            discard_slot_batch(*slot);
        };
        uint32_t rc = 0;
        try {
            rc = launch_segment_sync_only(seg, *slot);
        } catch (...) {
            invalidate_failed_queue();
            throw;
        }
        if (rc != 0) invalidate_failed_queue();
        TORCH_CHECK(rc == 0,
                    "launch_segment_sync_only: SDK rc=", rc,
                    " on segment idx=", seg_idx,
                    " (mutable_dmas.size=", seg.mutable_dmas.size(), ")");
        ++last_stats_.prepared_segment_hit_total;
        ++last_stats_.hw_batch_submit_total;
        return slot->queue.get();
    }

    // Keep the old fixed-side identities and any deliberately moved endpoints
    // leased through the replacement prepare and synchronous submission.  An
    // ABA between preflight and prepare must not silently establish a new
    // baseline for a released Storage.
    [[maybe_unused]] auto mutable_rebuild_preflight =
        validate_mutable_fixed_sidecars_for_rebuild();

    // Fallback: full rebuild on mapped queue slot。覆写前先把 idx
    // 失效,防止异常路径 (build_batch 抛错) 留下半 build 状态被下次误判命中。
    ::rhino_lkn::Queue_t* wq = nullptr;
    uint32_t build_rc = 0;
    bool pending_kernel_tokens_valid = false;
    RpuDmaSubmissionLease prepared_submission_lease;
    try {
        wq = ensure_private_queue(slot_idx, seg.core_ids.size());
        slot->built_segment_idx = -1;
        slot->fixed_dma_table.reset();
        slot->invalidate_mutable_dma_packets();
        build_rc = prepare_segment_queue(
            seg, *wq, hw_perf_enabled, prepared_submission_lease);
        if (build_rc == 0) {
            pending_kernel_tokens_valid =
                capture_segment_kernel_register_tokens(
                    seg, slot->pending_kernel_register_tokens);
        }
    } catch (...) {
        discard_slot_batch(*slot);
        if (slot_idx != 0) {
            slot->promotion_denied = true;
            if (slot->consumes_extra_budget) {
                slot->consumes_extra_budget = false;
                release_extra_prepared_queue();
            }
        }
        throw;
    }
    if (build_rc != 0 && slot_idx != 0) {
        // A retained arena is an optimization only. If the SDK cannot allocate
        // or build it, release the process token and keep this segment on the
        // pre-existing slot-0 path for the rest of the graph lifetime.
        discard_slot_batch(*slot);
        slot->promotion_denied = true;
        if (slot->consumes_extra_budget) {
            slot->consumes_extra_budget = false;
            release_extra_prepared_queue();
        }
        slot_idx = 0;
        slot = &private_queue_slots_[0];
        pending_kernel_tokens_valid = false;
        try {
            wq = ensure_private_queue(slot_idx, seg.core_ids.size());
            slot->built_segment_idx = -1;
            slot->fixed_dma_table.reset();
            slot->invalidate_mutable_dma_packets();
            build_rc = prepare_segment_queue(
                seg, *wq, hw_perf_enabled, prepared_submission_lease);
            if (build_rc == 0) {
                pending_kernel_tokens_valid =
                    capture_segment_kernel_register_tokens(
                        seg, slot->pending_kernel_register_tokens);
            }
        } catch (...) {
            discard_slot_batch(*slot);
            throw;
        }
    }
    if (build_rc != 0) {
        discard_slot_batch(*slot);
    }
    TORCH_CHECK(build_rc == 0,
                "prepare_segment_queue: build_batch SDK rc=", build_rc,
                " (!=0 = no kernels / batch entries > kMaxBatchKernels [default "
                "65536] / kd_buf|instr_buf overflow); segment idx=", seg_idx,
                " nodes=", (seg.end_idx - seg.start_idx));
    uint32_t rc_fallback;
    {
        // Keep host command preparation separate from submission/device wait
        // in CPU profiles; both are included in segment_launch above.
        RECORD_FUNCTION("rpu_graph::segment_submit_wait", {});
        rc_fallback = wq->enqueu_batch(/*wait_finish=*/true);
    }
    if (rc_fallback != 0) {
        discard_slot_batch(*slot);
    }
    TORCH_CHECK(rc_fallback == 0,
                "launch_segment_for_replay(fallback): enqueu_batch SDK rc=",
                rc_fallback, " on segment idx=", seg_idx);
    ++last_stats_.hw_batch_submit_total;
    slot->fixed_dma_table = seg.fixed_dma_table;
    slot->built_segment_idx = seg_idx;
    slot->hw_perf_enabled_at_build = hw_perf_enabled;
    slot->kernel_register_tokens.swap(
        slot->pending_kernel_register_tokens);
    slot->kernel_register_tokens_valid = pending_kernel_tokens_valid;
    ++last_stats_.prepared_segment_miss_total;
    return wq;
}

bool RpuKernelGraph::capture_segment_kernel_register_tokens(
        const Segment& seg,
        std::vector<std::pair<size_t, uint64_t>>& tokens) const {
    tokens.clear();
    for (size_t i = seg.start_idx; i < seg.end_idx; ++i) {
        if (nodes_[i].kind != GraphNodeKind::Kernel) continue;
        const auto& node = nodes_[i].as_kernel();
        if (node.kernel_idx >= kernels_.size() || !kernels_[node.kernel_idx]) {
            tokens.clear();
            return false;
        }
        uint64_t token = 0;
        if (kernels_[node.kernel_idx]->query_register_state_token(&token) !=
            ::rhino_lkn::kKernelQueryOk) {
            tokens.clear();
            return false;
        }
        tokens.emplace_back(node.kernel_idx, token);
    }
    return true;
}

uint32_t RpuKernelGraph::launch_segment_sync_only(
        Segment& seg,
        PreparedQueueSlot& prepared_slot) {
    TORCH_CHECK(prepared_slot.queue != nullptr,
                "launch_segment_sync_only: prepared queue is missing");
    auto& wq = *prepared_slot.queue;
    // A dirty invocation or disabled skip policy already requires a full sync.
    // Avoid collecting tokens that cannot authorize a skip; successful submit
    // below invalidates the old snapshot so the next clean replay syncs once
    // before establishing a new baseline.
    const bool may_skip_param_sync =
        runtime_policy_.fast_replay_skip_sync &&
        !kernel_params_dirty_this_replay_;
    if (!may_skip_param_sync) {
        prepared_slot.pending_kernel_register_tokens.clear();
    }
    // This path is admitted only for the same prepared segment and immutable
    // BUILT topology. Its committed snapshot already enumerates every kernel;
    // do not rediscover that list by walking all DMA/barrier/metadata nodes on
    // every replay. Still query every live register token, including writes
    // made outside the Graph's coarse dirty witness.
    bool tokens_unchanged = prepared_slot.kernel_register_tokens_valid;
    const bool current_tokens_valid = may_skip_param_sync && [&] {
        if (!prepared_slot.kernel_register_tokens_valid) {
            return capture_segment_kernel_register_tokens(
                seg, prepared_slot.pending_kernel_register_tokens);
        }
        auto& pending = prepared_slot.pending_kernel_register_tokens;
        pending.clear();
        pending.reserve(prepared_slot.kernel_register_tokens.size());
        for (const auto& committed : prepared_slot.kernel_register_tokens) {
            const size_t kernel_idx = committed.first;
            uint64_t token = 0;
            if (kernel_idx >= kernels_.size() || !kernels_[kernel_idx] ||
                kernels_[kernel_idx]->query_register_state_token(&token) !=
                    ::rhino_lkn::kKernelQueryOk) {
                pending.clear();
                return false;
            }
            pending.emplace_back(kernel_idx, token);
            tokens_unchanged = tokens_unchanged && token == committed.second;
        }
        return true;
    }();
    const bool kernel_register_tokens_match =
        current_tokens_valid && tokens_unchanged;
    // 前提:caller 已确认 wq 的 kd_buf
    // 是本段刚 build 的状态;mutable_dmas 与 SDK 内部 dma_id 计数一致。
    //
    // Resolve every mutable DMA before patching changed bindings with the typed
    // SDK API (size=0 preserves the built transfer size). Only an exact binding
    // from a successful submit on this same queue/build can skip that patch.
    // Kernel parameters are synchronized independently before submission.
    //
    // Fixed DMA cannot be patched after build. The caller exact-compares every
    // fixed src/dst binding before entering this path; any drift takes the full
    // prepare path and refreshes the sidecar. For mutable DMA, validate and
    // resolve the complete segment before patching any kd_buf packet, then
    // publish legacy fixed-side baselines only after a successful submission.
    struct MutableDmaResolution {
        uint64_t fixed_addr = 0;
        RpuDmaEndpoint src_endpoint;
        RpuDmaEndpoint dst_endpoint;
    };
    std::vector<MutableDmaResolution> resolutions;
    std::vector<RpuDeviceDmaEndpointRequest> endpoint_requests;
    resolutions.reserve(seg.mutable_dmas.size());
    endpoint_requests.reserve(seg.mutable_dmas.size() * 2);
    for (const auto& slot : seg.mutable_dmas) {
        TORCH_CHECK(
            slot.node_idx < nodes_.size() &&
                nodes_[slot.node_idx].kind == GraphNodeKind::Dma,
            "launch_segment_sync_only: mutable DMA node sidecar is stale at "
            "node ", slot.node_idx);
        const auto& dma = nodes_[slot.node_idx].as_dma();
        validate_semantic_dma_owner_for_execution(
            dma, slot.node_idx, "launch_segment_sync_only");
        const auto expected_variant =
            slot.kind == Segment::PreparedMutableDmaSlot::Kind::MutableSrc
            ? DmaNodeData::Variant::MutableSrc
            : DmaNodeData::Variant::MutableDst;
        TORCH_CHECK(
            dma.variant == expected_variant && dma.bytes == slot.bytes &&
                dma.channel == slot.channel,
            "launch_segment_sync_only: mutable DMA topology drift at node ",
            slot.node_idx);
        TORCH_CHECK(
            dma.semantic_endpoint_id == 0 ||
                (dma.live_base == slot.live_base &&
                 dma.live_offset == slot.live_offset),
            "launch_segment_sync_only: mutable DMA slot drift at node ",
            slot.node_idx);
        TORCH_CHECK(dma.live_base != nullptr,
                    "launch_segment_sync_only: mutable DMA missing live_base "
                    "at node ", slot.node_idx);
        const uint64_t live = checked_dma_live_address(
            *dma.live_base, dma.live_offset, slot.node_idx,
            "launch_segment_sync_only");
        const uint64_t fixed_addr =
            slot.kind == Segment::PreparedMutableDmaSlot::Kind::MutableSrc
            ? dma.dst_addr : dma.src_addr;
        TORCH_CHECK(
            dma.semantic_endpoint_id == 0 || fixed_addr == slot.fixed_addr,
            "launch_segment_sync_only: semantic mutable DMA fixed endpoint "
            "address drift at node ", slot.node_idx);
        if (slot.kind == Segment::PreparedMutableDmaSlot::Kind::MutableSrc) {
            endpoint_requests.push_back(RpuDeviceDmaEndpointRequest{
                live, slot.bytes, "launch_segment_sync_only/MutableSrc/src"});
            endpoint_requests.push_back(RpuDeviceDmaEndpointRequest{
                fixed_addr, slot.bytes,
                "launch_segment_sync_only/MutableSrc/dst"});
        } else {
            endpoint_requests.push_back(RpuDeviceDmaEndpointRequest{
                fixed_addr, slot.bytes,
                "launch_segment_sync_only/MutableDst/src"});
            endpoint_requests.push_back(RpuDeviceDmaEndpointRequest{
                live, slot.bytes, "launch_segment_sync_only/MutableDst/dst"});
        }
        resolutions.push_back(MutableDmaResolution{fixed_addr, {}, {}});
    }

    std::vector<RpuDmaEndpoint> endpoints(endpoint_requests.size());
    [[maybe_unused]] auto mutable_submission_lease =
        rpu_resolve_device_dma_endpoints(
        endpoint_requests.data(), endpoint_requests.size(), endpoints.data());
    auto& pending_packets = prepared_slot.pending_mutable_dma_packets;
    pending_packets.clear();
    pending_packets.reserve(seg.mutable_dmas.size());
    for (size_t i = 0; i < resolutions.size(); ++i) {
        auto& resolved = resolutions[i];
        resolved.src_endpoint = endpoints[i * 2];
        resolved.dst_endpoint = endpoints[i * 2 + 1];
        const auto& slot = seg.mutable_dmas[i];
        const auto& fixed_endpoint =
            slot.kind == Segment::PreparedMutableDmaSlot::Kind::MutableSrc
            ? resolved.dst_endpoint : resolved.src_endpoint;
        TORCH_CHECK(
            resolved.fixed_addr != slot.fixed_addr ||
                fixed_endpoint.allocation_id == slot.fixed_allocation_id,
            "launch_segment_sync_only: fixed DMA logical allocation identity "
            "drift at node ", slot.node_idx,
            " (same device address was released and reused)");
        pending_packets.push_back(PreparedQueueSlot::MutableDmaPacketBinding{
            slot.node_idx, slot.dma_id, slot.kind, slot.bytes, slot.channel,
            resolved.src_endpoint, resolved.dst_endpoint});
    }

    // The backend's fresh endpoint batch and lease above remain authoritative.
    // Managed DDR identities are monotonic; SPM roots are immutable for the
    // process execution lifetime. Thus equal live owners/identities/offsets and
    // topology reuse the previous successful typed SDK proof. Never dereference
    // a historical owner, cache a lease, or compare device addresses alone.
    const bool packet_snapshot_matches_size =
        prepared_slot.mutable_dma_packets.size() == pending_packets.size();
    for (size_t i = 0; i < seg.mutable_dmas.size(); ++i) {
        if (packet_snapshot_matches_size &&
            prepared_slot.mutable_dma_packets[i] == pending_packets[i]) {
            continue;
        }
        const auto& slot = seg.mutable_dmas[i];
        const auto& resolved = resolutions[i];
        const uint32_t rc = wq.update_dma_kernel(
            slot.dma_id,
            *resolved.src_endpoint.owner, resolved.src_endpoint.offset,
            *resolved.dst_endpoint.owner, resolved.dst_endpoint.offset,
            /*size=*/0);
        if (rc != 0) {
            // 直接返回给 caller 让 TORCH_CHECK 给出 dma_id / segment idx 上下文。
            return rc;
        }
    }
    // Kernel_t's public register-state token is the authoritative fine-grained
    // witness: it advances after every successful register write.  Keep the
    // coarse Graph witness as a conservative backstop, while using the token
    // snapshot to catch writes made through paths that did not mark it. Mutable
    // DMA publication is independent: changed packet slots have been patched
    // and unchanged slots retain the same successful typed binding. The full
    // barrier below publishes all mapped command writes before submission.
    // Therefore an unchanged mutable-DMA segment no longer needs a full
    // scan/copy of every mutable kernel argument block.
    const bool can_skip_param_sync =
        runtime_policy_.fast_replay_skip_sync &&
        !kernel_params_dirty_this_replay_ &&
        kernel_register_tokens_match;
    if (!can_skip_param_sync) {
        RECORD_FUNCTION("rpu_graph::mutable_param_sync", {});
        const uint32_t rc_sync = wq.sync_mutable_params();
        if (rc_sync != 0) return rc_sync;
    }
    // Launch batch arenas are uncached CPU/device mappings.  Publish both the
    // mutable-DMA packet writes and any kernel-argument copies before the
    // submission doorbell.  This is intentionally shape/model independent.
    __sync_synchronize();
    RECORD_FUNCTION("rpu_graph::segment_submit_wait", {});
    const uint32_t rc_submit = wq.enqueu_batch(/*wait_finish=*/true);
    if (rc_submit != 0) return rc_submit;

    prepared_slot.kernel_register_tokens.swap(
        prepared_slot.pending_kernel_register_tokens);
    prepared_slot.kernel_register_tokens_valid = current_tokens_valid;
    // No partial update/sync/submit failure may establish a reusable snapshot.
    // The caller destroys the queue and both snapshots on every such failure.
    prepared_slot.mutable_dma_packets.swap(pending_packets);

    for (size_t i = 0; i < seg.mutable_dmas.size(); ++i) {
        auto& slot = seg.mutable_dmas[i];
        const auto& resolved = resolutions[i];
        const auto& fixed_endpoint =
            slot.kind == Segment::PreparedMutableDmaSlot::Kind::MutableSrc
            ? resolved.dst_endpoint : resolved.src_endpoint;
        // A legacy mutable descriptor may replace its fixed address. Commit
        // that new identity only after every patch, optional kernel sync, and
        // the device submission have all succeeded.
        slot.fixed_addr = resolved.fixed_addr;
        slot.fixed_allocation_id = fixed_endpoint.allocation_id;
    }
    return 0;
}

static void mirror_per_segment_replay_count(
        GraphStats& stats,
        const std::vector<Segment>& segments) {
    stats.per_segment_replay_count.clear();
    stats.per_segment_replay_count.reserve(segments.size());
    for (const auto& seg : segments) {
        stats.per_segment_replay_count.push_back(seg.replay_count);
    }
}

// =============================================================================
// execute_data_node — 执行单个 data node（Memcpy / Memset）
// =============================================================================
// Memcpy: ::memcpy(dst, src, bytes)
// Memset: ::memset(dst, value, bytes)
// 末尾统一 __sync_synchronize()，确保 host 写入对后续 RPU kernel 可见。
// 三条 execute_graph_* 路径（recording / replaying / oneshot）共享此 helper。

namespace {
// 1.1 P0:把 data-node kind 映射成稳定的 trace 名,供 Chrome tracing 直接搜。
const char* data_node_trace_name(GraphNodeKind k) {
    switch (k) {
    case GraphNodeKind::Memcpy:       return "rpu_graph::data_memcpy";
    case GraphNodeKind::Memset:       return "rpu_graph::data_memset";
    case GraphNodeKind::ChildGraph:   return "rpu_graph::data_child_graph";
    case GraphNodeKind::Branch:       return "rpu_graph::data_branch";
    case GraphNodeKind::HostCallback: return "rpu_graph::data_host_callback";
    case GraphNodeKind::Tier3Oneshot: return "rpu_graph::data_tier3_oneshot";
    case GraphNodeKind::Kernel:       return "rpu_graph::data_kernel_unexpected";
    // Dma / Barrier 是 in-segment 节点,prepare_segment_queue 走 batch 路径处理,
    // 不会走 execute_data_node — 但 trace 工具可能在任意上下文调,给一个名字。
    case GraphNodeKind::Dma:          return "rpu_graph::data_dma_unexpected";
    case GraphNodeKind::Barrier:      return "rpu_graph::data_barrier_unexpected";
    }
    return "rpu_graph::data_unknown";
}
}

void RpuKernelGraph::execute_ddr_copy(const MemcpyNodeData& copy) {
    // DDR address striping remains physical. The selected DMA engine must be
    // available in the owner's allocated execution resource prefix.
    TORCH_CHECK(runtime_policy_.execution_core_count >= 1 &&
                    runtime_policy_.execution_core_count <= 8,
                "execute_data_node: DDR copy core budget must be in [1, 8]");
    TORCH_CHECK(copy.bytes > 0 && copy.bytes <= (size_t{8} << 20) &&
                    copy.bytes % 32 == 0,
                "execute_data_node: DDR copy requires 32-byte-aligned size in [1, 8 MiB]");
    TORCH_CHECK(copy.dma_channel >= -1 && copy.dma_channel < 8,
                "execute_data_node: DDR copy channel must be -1 or in [0, 7]");
    const int channel = copy.dma_channel >= 0 ? copy.dma_channel : 0;
    validate_graph_dma_channel(channel, runtime_policy_.execution_core_count,
                               "execute_data_node/DDR_TO_DDR");
    const RpuCpuDmaEndpointRequest requests[] = {
        {copy.dst, copy.bytes, "execute_data_node/DDR_TO_DDR/dst"},
        {copy.src, copy.bytes, "execute_data_node/DDR_TO_DDR/src"}};
    RpuDmaEndpoint endpoints[2];
    [[maybe_unused]] auto submission_lease =
        rpu_resolve_cpu_dma_endpoints(requests, 2, endpoints);
    const auto& dst = endpoints[0];
    const auto& src = endpoints[1];
    try {
        if (!data_dma_queue_) {
            data_dma_queue_ = std::make_unique<::rhino_lkn::Queue_t>(
                static_cast<uint8_t>(runtime_policy_.execution_core_count));
        }
        auto& queue = *data_dma_queue_;
        rpu_add_dma_checked(queue, src, dst, copy.bytes, channel,
                            "execute_data_node/DDR_TO_DDR");
        const uint32_t build_rc = queue.build_batch();
        TORCH_CHECK(build_rc == 0,
                    "execute_data_node: DDR copy build_batch SDK rc=", build_rc);
        const uint32_t enqueue_rc = queue.enqueu_batch(/*wait_finish=*/true);
        TORCH_CHECK(enqueue_rc == 0,
                    "execute_data_node: DDR copy enqueu_batch SDK rc=", enqueue_rc);
        const uint32_t discard_rc = queue.discard_completed_batch();
        TORCH_CHECK(discard_rc == 0,
                    "execute_data_node: DDR copy discard_completed_batch SDK rc=", discard_rc);
    } catch (...) {
        // SDK destruction waits for any submitted work and releases pending
        // typed endpoint leases as well as partially built command arenas.
        // In particular, build failure cannot leave a stale DMA for a retry.
        data_dma_queue_.reset();
        throw;
    }
}

void RpuKernelGraph::execute_data_node(const GraphNode& node) {
    RECORD_FUNCTION(data_node_trace_name(node.kind), {});
    switch (node.kind) {
    case GraphNodeKind::Memcpy: {
        const auto& m = std::get<MemcpyNodeData>(node.data);
        // D1 诊断:env RPU_GRAPH_DDR_SPM_LOG=1 打印 execute 期 src 内容
        // checksum,跟 ddr_to_spm_impl 创建时 checksum 比对 — 若不一致说明
        // 执行期 src ptr 上的内容已被覆写 / 改变(stale memory residue 或
        // op-stream ordering 问题)。
        static const bool ddr_spm_log = []() {
            const char* e = std::getenv("RPU_GRAPH_DDR_SPM_LOG");
            return e && *e && *e != '0';
        }();
        if (ddr_spm_log && m.kind == CopyKind::HOST_MEMCPY) {
            const auto* b = reinterpret_cast<const uint8_t*>(m.src);
            uint64_t h = 0xcbf29ce484222325ull;
            const size_t n = m.bytes < 64 ? m.bytes : 64;
            for (size_t i = 0; i < n; ++i) {
                h ^= b[i];
                h *= 0x100000001b3ull;
            }
            std::fprintf(
                stderr,
                "[D2S-EXEC] node=%zu src=0x%lx dst=0x%lx bytes=%zu chk=0x%lx\n",
                static_cast<size_t>(&node - nodes_.data()),
                reinterpret_cast<uint64_t>(m.src),
                reinterpret_cast<uint64_t>(m.dst),
                m.bytes, h);
        }
        switch (m.kind) {
        case CopyKind::HOST_MEMCPY:
            ::memcpy(m.dst, m.src, m.bytes);
            break;
        case CopyKind::DDR_TO_DDR: {
            execute_ddr_copy(m);
            g_memcpy_counter.dma_ddr_to_ddr.fetch_add(
                1, std::memory_order_relaxed);
            break;
        }
        case CopyKind::DDR_TO_SPM:
        case CopyKind::SPM_TO_DDR:
        case CopyKind::SPM_TO_SPM:
            TORCH_CHECK(false,
                        "execute_data_node: CopyKind ",
                        static_cast<int>(m.kind),
                        " not implemented; SPM<->DDR DMA requires sliced API");
        }
        break;
    }
    case GraphNodeKind::Memset: {
        const auto& m = std::get<MemsetNodeData>(node.data);
        ::memset(m.dst, m.value, m.bytes);
        break;
    }
    case GraphNodeKind::ChildGraph: {
        const auto& c = std::get<ChildGraphNodeData>(node.data);
        TORCH_CHECK(c.child &&
                    c.child->state() == State::BUILT &&
                    c.child->replayable(),
                    "RpuKernelGraph::execute_data_node: ChildGraph not replayable");
        RpuKernelGraph* previous_parent = c.child->execution_parent_;
        c.child->execution_parent_ = this;
        auto restore_parent = c10::make_scope_exit([&] {
            c.child->execution_parent_ = previous_parent;
        });
        // Child replay participates in the parent graph lifecycle. The parent
        // REPLAYING end() already does boundary flush before and after the
        // executor; repeating the full boundary set around every child window
        // makes auto-promoted children materially slower than the flat graph.
        if (c.data_patches.empty()) {
            c.child->replay_prepared_child(c.input_patches);
        } else {
            c.child->replay_prepared_child_with_data_patches(
                c.input_patches, c.data_patches);
        }
        const auto& child_records = c.child->last_stats_.last_segment_executions;
        last_stats_.last_segment_executions.insert(
            last_stats_.last_segment_executions.end(),
            child_records.begin(), child_records.end());
        break;
    }
    case GraphNodeKind::Branch:
        // G6 BranchNode is a marker/segment boundary only. Cache separation is
        // handled by GraphSignature.branch_key; device-side branch dispatch is
        // intentionally deferred to G7/G8.
        break;
    case GraphNodeKind::HostCallback: {
        auto& hc = const_cast<GraphNode&>(node).as_host_callback();
        TORCH_CHECK(hc.op_handle.has_value(),
                    "RpuKernelGraph::execute_data_node: HostCallback op_handle empty");
        TORCH_CHECK(hc.tier != HostCallbackTier::Reject,
                    "RpuKernelGraph::execute_data_node: Reject tier should not "
                    "have entered HostCallback path; op=",
                    hc.op_handle->operator_name().name);
        TORCH_CHECK(hc.tier != HostCallbackTier::Tier3 ||
                    state_ != State::REPLAYING,
                    "RpuKernelGraph::execute_data_node: Tier3 HostCallback "
                    "should not reach REPLAYING executor (scope must have been "
                    "marked non-replayable). op=",
                    hc.op_handle->operator_name().name);
        // (1) pre-flush: 让 host 看到 RPU 之前写入的 DDR
        flush_boundary_ptrs();
        // (2) 重建 stack: live_tensor_args 提供本轮 tensor,args_template
        //     提供非 tensor scalar 槽
        torch::jit::Stack stack;
        stack.reserve(hc.args_template.size());
        size_t next_t = 0;
        for (size_t i = 0; i < hc.args_template.size(); ++i) {
            // tensor_arg_indices 升序,所以 next_t 顺序对应
            if (next_t < hc.tensor_arg_indices.size() &&
                hc.tensor_arg_indices[next_t] == i) {
                TORCH_CHECK(next_t < hc.live_tensor_args.size(),
                            "HostCallback executor: live_tensor_args underflow "
                            "at ", next_t);
                stack.emplace_back(hc.live_tensor_args[next_t]);
                ++next_t;
            } else {
                stack.emplace_back(hc.args_template[i]);
            }
        }
        // (3) zero-copy CPU redispatch。
        //     Tier1 (.out variant) → alias=true:避免对 zero-copy CPU return
        //     view 做 .to(rpu),否则 storage allocator=nullptr 会触发 heap
        //     corruption (topk.values / sort.values SIGABRT 根因)。原 .out
        //     RPU tensor 被替回 stack,后续 (4) 仍把数据 copy 到 stable buffer。
        //     Tier2 (functional,无 out args) → alias=false:CPU op 内部自己
        //     alloc fresh CPU tensor,.to(rpu) 安全;后续 (4) 同样 copy。
        //     Tier3 → alias=false:不缓存,本轮按需用。
        const bool alias_for_replay =
            (hc.tier == HostCallbackTier::Tier1);
        run_cpu_fallback_zerocopy(*hc.op_handle, &stack,
                                   /*alias_output_args=*/alias_for_replay);
        // (4) Tier1 + Tier2: copy fresh return → graph-owned stable output_pool[r],
        //     替换 stack slot,登记 post-CPU boundary flush。下游 RPU op 永远
        //     看到 hc.output_pool_dev_addrs[r] 这个稳定 dev_addr。
        if (hc.tier == HostCallbackTier::Tier1 ||
            hc.tier == HostCallbackTier::Tier2) {
            const size_t num_returns =
                hc.op_handle->schema().returns().size();
            TORCH_CHECK(stack.size() >= num_returns,
                        "HostCallback executor: stack underflow after CPU op");
            TORCH_CHECK(hc.output_pool.size() == num_returns,
                        "HostCallback executor: output_pool.size=",
                        hc.output_pool.size(),
                        " != num_returns=", num_returns,
                        " op=", hc.op_handle->operator_name().name);
            const size_t ret_start = stack.size() - num_returns;
            for (size_t r = 0; r < num_returns; ++r) {
                c10::IValue& ret = stack[ret_start + r];
                if (!ret.isTensor()) continue;
                at::Tensor fresh_out = ret.toTensor();
                at::Tensor& stable = hc.output_pool[r];
                TORCH_CHECK(stable.defined(),
                            "HostCallback executor: output_pool[", r,
                            "] not defined (RECORDING 期 adopt_returns 应已分配); op=",
                            hc.op_handle->operator_name().name);
                // copy CPU/RPU fresh tensor → stable RPU buffer
                stable.copy_(fresh_out);
                // 替换 stack slot,下游(若有)直接读 stable
                ret = stable;
                // 登记 post-CPU boundary flush:host 写入对后续 RPU 可见
                record_boundary_flush(stable.data_ptr());
                // sanity check:stable 地址应当与 RECORDING 期录的一致
                if (r < hc.output_pool_dev_addrs.size()) {
                    uint64_t dev = rhino_lkn::RpuGetDevAddr(stable.data_ptr());
                    TORCH_CHECK(dev == hc.output_pool_dev_addrs[r],
                                "HostCallback executor: output_pool[", r,
                                "] dev_addr drift; recorded=",
                                hc.output_pool_dev_addrs[r], " got=", dev,
                                " op=", hc.op_handle->operator_name().name,
                                " (graph-owned buffer 不应漂移,可能是 cache "
                                "entry 状态损坏)");
                }
            }
        }
        // (5) post-flush: host 写入对后续 RPU kernel 可见
        flush_boundary_ptrs();
        // 注:不清 live_tensor_args —— REPLAYING 期 op stream 重 dispatch
        // 时 capture_host_callback_replay_args 会覆写;RECORDING-collapse
        // 路径(execute_graph_for_recording / oneshot)是一次性消费,清不清
        // 都没影响。
        break;
    }
    case GraphNodeKind::Kernel:
        TORCH_CHECK(false,
                    "RpuKernelGraph::execute_data_node called on Kernel node");
    case GraphNodeKind::Tier3Oneshot: {
        // G3 m2-5 — Tier3OneshotNode 节点级 oneshot 真跑路径。三条 execute
        // path 共用本逻辑:
        //   RECORDING-collapse (execute_graph_for_recording):RECORDING 期
        //     capture_cpu_fallback 已经真跑过一次给当前 forward 下游用,这里
        //     end() 时重建 fresh return + copy 进 transient buffer 给 RECORDING
        //     期 set_regs 写好的下游 segment kernel 拿到正确的 RNG 真值。
        //   oneshot (execute_graph_oneshot):m2-2 过渡期路径(scope 仍
        //     mark_non_replayable),m2-5 起默认不再走该路径,但保留兜底。
        //   REPLAYING (execute_graph_for_replaying):本步 op stream 期
        //     capture_tier3_oneshot_replay_args 已覆写 live_tensor_args + 合成
        //     transient buffer return 到 stack;executor 在每个 segment 边界
        //     真跑 CPU op + copy fresh→transient,下个 segment kernel
        //     launch_prepared_batch 时读到本步 RNG 值。
        auto& t = const_cast<GraphNode&>(node).as_tier3_oneshot();
        TORCH_CHECK(t.op_handle.has_value(),
                    "RpuKernelGraph::execute_data_node: Tier3Oneshot op_handle empty");
        // (1) pre-flush:让 host 看到 RPU 之前写入的 DDR
        flush_boundary_ptrs();
        // (2) 重建 stack:live_tensor_args 提供本轮 tensor,args_template 提供
        //     非 tensor scalar 槽(同 HostCallback 重建逻辑)
        torch::jit::Stack stack;
        stack.reserve(t.args_template.size());
        size_t next_t = 0;
        for (size_t i = 0; i < t.args_template.size(); ++i) {
            if (next_t < t.tensor_arg_indices.size() &&
                t.tensor_arg_indices[next_t] == i) {
                TORCH_CHECK(next_t < t.live_tensor_args.size(),
                            "Tier3Oneshot executor: live_tensor_args underflow "
                            "at ", next_t);
                stack.emplace_back(t.live_tensor_args[next_t]);
                ++next_t;
            } else {
                stack.emplace_back(t.args_template[i]);
            }
        }
        // (3) zero-copy CPU redispatch。Tier3 不做 alias output args
        //     (RNG / dyn-shape 的 functional return 没有 .out= variant 需要
        //     替回 stack 的语义);CPU op 自己 alloc fresh CPU tensor → .to(rpu)。
        run_cpu_fallback_zerocopy(*t.op_handle, &stack,
                                   /*alias_output_args=*/false);
        // (4) m2-4 — 把 fresh return 数据 copy 到 transient buffer(地址稳定,
        //     内容刷新)。下游 RPU kernel 在 RECORDING 期 set_regs 写的是 transient
        //     buffer dev_addr,oneshot 跑下游时读到的就是本步刚 copy 进去的
        //     RNG 真值。
        //     不再覆写 t.last_output_dev_addrs — 那是 RECORDING adopt_returns
        //     设的 transient buffer dev_addr(跨 replay 稳定),被 m2-3 dep 推断
        //     的 writer 表引用。oneshot 期 fresh return 是临时的,不应入元数据。
        const size_t num_returns = t.op_handle->schema().returns().size();
        if (stack.size() >= num_returns) {
            const size_t ret_start = stack.size() - num_returns;
            for (size_t r = 0; r < num_returns &&
                     r < t.transient_resource_ids.size(); ++r) {
                const c10::IValue& ret = stack[ret_start + r];
                const int res_id = t.transient_resource_ids[r];
                if (!ret.isTensor() || res_id < 0) continue;
                const at::Tensor& fresh_rt = ret.toTensor();
                if (!fresh_rt.defined()) continue;
                const at::Tensor& transient_rt =
                    tier3_transient_tensor_at(static_cast<size_t>(res_id));
                TORCH_CHECK(transient_rt.defined(),
                            "execute_data_node: transient buffer not "
                            "defined for res_id=", res_id);
                transient_rt.copy_(fresh_rt);
                record_boundary_flush(transient_rt.data_ptr());
            }
        }
        // (5) post-flush
        flush_boundary_ptrs();
        break;
    }
    case GraphNodeKind::Dma:
    case GraphNodeKind::Barrier:
        // Dma / Barrier 是 in-segment 节点,正常路径由 prepare_segment_queue
        // 在 batch 内顺序处理;走到 execute_data_node 说明 build_segments_from_nodes
        // 把它们当作段边界了 — 这是 invariant 违例,直接抛错。
        TORCH_CHECK(false,
                    "execute_data_node: Dma / Barrier node should not reach "
                    "execute_data_node (in-segment kind ",
                    static_cast<int>(node.kind), ")");
    }
    __sync_synchronize();
}

// =============================================================================
// GRAPH-INSTR helpers (P2, debug-level session)
// =============================================================================
//
// kernel_id_short_name: return a readable label for the most common KernelId
// values; fall back to "KernelId#N" for the rest. Kept narrow on purpose — the
// histogram is observability, not a contract; we'd rather have a maintenance-
// free fallback than a 96-case switch that goes stale.
//
// kernel_label / kernel_histogram_string / kernel_summary_string: prefer
// KernelNodeData.kernel_name (dynamic kernels carry descriptive strings) and
// fall back to kernel_id_short_name(*kernel_id).

namespace {

const char* kernel_id_short_name(KernelId id) {
    switch (id) {
    // RMSNorm family
    case KernelId::RMS_NORM_BF16:           return "RMSNORM";
    case KernelId::RMS_NORM_BF16_SPM:       return "RMSNORM_SPM";
    case KernelId::RMS_NORM_NEWTON_SPM:     return "RMSNORM_NEWTON_SPM";
    // RoPE family
    case KernelId::ROPE:                    return "ROPE";
    case KernelId::ROPE_DDR:                return "ROPE_DDR";
    case KernelId::MROPE:                   return "MROPE";
    case KernelId::ROPE_2D_DDR:             return "ROPE_2D_DDR";
    case KernelId::ROPE_2D_SPM:             return "ROPE_2D_SPM";
    // Linear family
    case KernelId::PL_AT_FP16_ACC32_M208N128:
    case KernelId::PL_AT_FP16_ACC32_M240N112:
    case KernelId::PL_AT_FP16_ACC32_M272N96:
    case KernelId::PL_AT_FP16_ACC32_M304N80:
    case KernelId::PL_AT_FP16_ACC32_M352N64:
    case KernelId::PL_AT_FP16_ACC32_M416N48:
    case KernelId::PL_AT_FP16_ACC32_M496N32:
                                            return "LINEAR_ACC32";
    // SDPA family
    case KernelId::SDPA_FLASH_ATTN_SPM:     return "SDPA_SPM";
    case KernelId::SDPA_FLASH_ATTN_SPM_VCTXLEN:
                                            return "SDPA_VCTXLEN";
    // Softmax / LayerNorm / reduce
    case KernelId::SOFTMAX_C16_GAUTO:       return "SOFTMAX";
    case KernelId::LAYER_NORM:              return "LAYERNORM";
    case KernelId::LAYER_NORM_SIMPLE:       return "LAYERNORM_SIMPLE";
    case KernelId::REDUCE_MEAN_LAST_DIM_DDR:
                                            return "REDUCE_MEAN";
    // Eltwise binary (bucketed; ~20 variants share the same conceptual op)
    case KernelId::BINARY_SAMESHAPE:
    case KernelId::BINARY_SAMESHAPE_DDR:
    case KernelId::BINARY_SCALAR:
    case KernelId::BINARY_SCALAR_DDR:
    case KernelId::BINARY_SCALAR_POWER_DDR:
    case KernelId::BINARY_NX1_NXC256_BATCH:
    case KernelId::BINARY_NX1_NXC256_BATCH_BOPA:
    case KernelId::BINARY_NX1_NXC_V16_BATCH:
    case KernelId::BINARY_NX1_NXC_V16_BATCH_BOPA:
    case KernelId::BINARY_1XC_NXC_V256_BATCH:
    case KernelId::BINARY_1XC_NXC_V256_BATCH_BOPA:
    case KernelId::BINARY_1XC_NXC_V16_BATCH:
    case KernelId::BINARY_1XC_NXC_V16_BATCH_BOPA:
    case KernelId::BINARY_NX1_NXC256_BATCH_DDR:
    case KernelId::BINARY_NX1_NXC256_BATCH_BOPA_DDR:
    case KernelId::BINARY_NX1_NXC_V16_BATCH_DDR:
    case KernelId::BINARY_NX1_NXC_V16_BATCH_BOPA_DDR:
    case KernelId::BINARY_1XC_NXC_V256_BATCH_DDR:
    case KernelId::BINARY_1XC_NXC_V256_BATCH_BOPA_DDR:
    case KernelId::BINARY_1XC_NXC_V16_BATCH_DDR:
    case KernelId::BINARY_1XC_NXC_V16_BATCH_BOPA_DDR: return "BINARY";
    // Unary
    case KernelId::UNARY_DDR:               return "UNARY";
    case KernelId::UNARY_GELU_TANH_SPM:
    case KernelId::UNARY_GELU_ERF_SPM:      return "GELU";
    case KernelId::UNARY_SOFTPLUS_SPM:      return "SOFTPLUS";
    case KernelId::MEMSET_SPM_MULTI_CORE_V2:
                                            return "MEMSET";
    // KV-cache insert (Llama / Qwen / Gemma share the family in the histogram)
    case KernelId::LLAMA_INSERT_KCACHE:     return "KV_INSERT_K";
    case KernelId::LLAMA_INSERT_VCACHE:     return "KV_INSERT_V";
    case KernelId::LLAMA_INSERT_KCACHE_V16: return "KV_INSERT_K_V16";
    case KernelId::LLAMA_INSERT_VCACHE_V16: return "KV_INSERT_V_V16";
    // LLM reductions / collectives
    case KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_PACED:
                                            return "ALLREDUCE_RING_PACED";
    case KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_NOPACE:
                                            return "ALLREDUCE_RING_NOPACE";
    case KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_LOCAL_SPM:
                                            return "ALLREDUCE_RING_LOCAL_SPM";
    case KernelId::ALL_GATHER_MULTI_CORE:              return "ALL_GATHER";
    default:                                return nullptr;  // numeric fallback
    }
}

std::string kernel_label(const KernelNodeData& k) {
    if (!k.kernel_name.empty()) return k.kernel_name;
    if (k.kernel_id) {
        if (const char* n = kernel_id_short_name(*k.kernel_id)) return n;
        return "KernelId#" + std::to_string(static_cast<int>(*k.kernel_id));
    }
    return "<unknown>";
}

// kernel_histogram_string — count kernels by label across [start, end) (or
// across all nodes if start==end==0). DMA/Barrier nodes contribute to "DMA"
// and "Barrier" buckets so the INFO summary captures the full graph mix.
// Returns "LABEL:count, ..." sorted by count desc; empty buckets dropped.
std::string kernel_histogram_string(const std::vector<GraphNode>& nodes,
                                    size_t start = 0, size_t end = 0) {
    if (start == 0 && end == 0) end = nodes.size();
    std::unordered_map<std::string, size_t> bucket;
    for (size_t i = start; i < end; ++i) {
        const auto& n = nodes[i];
        switch (n.kind) {
        case GraphNodeKind::Kernel: ++bucket[kernel_label(n.as_kernel())]; break;
        case GraphNodeKind::Dma:     ++bucket["DMA"];     break;
        case GraphNodeKind::Barrier: ++bucket["Barrier"]; break;
        default: break;  // data nodes excluded — histogram is hardware-side
        }
    }
    std::vector<std::pair<std::string, size_t>> v(bucket.begin(), bucket.end());
    std::sort(v.begin(), v.end(),
              [](const auto& a, const auto& b) {
                  return a.second != b.second ? a.second > b.second : a.first < b.first;
              });
    std::ostringstream oss;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ", ";
        oss << v[i].first << ":" << v[i].second;
    }
    return oss.str();
}

// kernel_summary_string — one-line per-segment summary for DEBUG.
// "seg[N] kernels=[A..B) count=C cores=D dom=NAME×K dma=M => private_queue ..."
std::string kernel_summary_string(const std::vector<GraphNode>& nodes,
                                  const Segment& seg, size_t seg_idx) {
    std::unordered_map<std::string, size_t> bucket;
    size_t dma = 0;
    for (size_t i = seg.start_idx; i < seg.end_idx; ++i) {
        const auto& n = nodes[i];
        if (n.kind == GraphNodeKind::Kernel) ++bucket[kernel_label(n.as_kernel())];
        else if (n.kind == GraphNodeKind::Dma) ++dma;
    }
    std::string dom = "-";
    size_t dom_count = 0;
    for (const auto& [k, c] : bucket) {
        if (c > dom_count || (c == dom_count && k < dom)) { dom = k; dom_count = c; }
    }
    const size_t kernels_in_seg =
        (seg.end_idx > seg.start_idx) ? (seg.end_idx - seg.start_idx) : 0;
    std::ostringstream oss;
    oss << "seg[" << seg_idx << "] kernels=[" << seg.start_idx << ".." << seg.end_idx
        << ") count=" << kernels_in_seg
        << " cores=" << seg.core_ids.size()
        << " dom=" << dom << "x" << dom_count
        << " dma=" << dma;
    return oss.str();
}

// The public Launch API owns sanitized Chrome serialization. Graph companion
// metadata uses only host lifecycle/census values and an address-free filename.
std::string build_hw_perf_dump_path(const std::string& output_dir,
                                   const std::string& graph_name,
                                   bool is_replay, size_t seg_idx, uint64_t seq) {
    std::ostringstream filename;
    filename << "rpu_hwperf_pid" << static_cast<int>(::getpid())
             << "_seq" << seq << "_graph" << std::hex
             << std::hash<std::string>{}(graph_name) << std::dec
             << (is_replay ? "_replay" : "_build")
             << "_seg" << seg_idx << ".json";
    return (std::filesystem::path(output_dir) / filename.str()).string();
}

FILE* open_graph_companion_file(const std::string& path) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL |
                                      O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) return nullptr;
    FILE* file = ::fdopen(fd, "w");
    if (file == nullptr) {
        ::close(fd);
        ::unlink(path.c_str());
    }
    return file;
}

void write_graph_json_string(FILE* file, const std::string& value) {
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            std::fputc('\\', file);
            std::fputc(c, file);
        } else if (c < 0x20) {
            std::fprintf(file, "\\u%04x", static_cast<unsigned int>(c));
        } else {
            std::fputc(c, file);
        }
    }
}

// Publish host companion metadata only after successful write and close.
// An existing file or symlink is never replaced. Failed .part files cannot look like evidence.
int64_t write_hw_perf_graph_companion(const GraphSegmentExecution& record) {
    const std::string temporary = record.companion_path + ".part";
    FILE* file = open_graph_companion_file(temporary);
    if (file == nullptr) return 3;
    const auto string_field = [&](const char* key, const std::string& value) {
        std::fprintf(file, ",\"%s\":\"", key);
        write_graph_json_string(file, value);
        std::fputc('"', file);
    };
    const auto number_field = [&](const char* key, uint64_t value) {
        std::fprintf(file, ",\"%s\":%llu", key,
                     static_cast<unsigned long long>(value));
    };
    std::fputs("{\"schema\":\"rpu-graph-hwperf-v2\"", file);
    string_field("hwperf_file", std::filesystem::path(record.hwperf_path).filename().string());
    string_field("graph_name", record.graph_name);
    string_field("execution_kind", record.execution_kind);
    number_field("hwperf_sequence", record.hwperf_sequence);
    number_field("graph_lifetime_id", record.graph_lifetime_id);
    number_field("build_generation", record.build_generation);
    number_field("execution_ordinal", record.execution_ordinal);
    number_field("parent_graph_lifetime_id", record.parent_graph_lifetime_id);
    number_field("parent_execution_ordinal", record.parent_execution_ordinal);
    std::fputs(",\"execution_ancestors\":[", file);
    for (size_t i = 0; i < record.execution_ancestors.size(); ++i) {
        const auto& ancestor = record.execution_ancestors[i];
        std::fprintf(file, "%s[%llu,%llu]", i == 0 ? "" : ",",
                     static_cast<unsigned long long>(ancestor.first),
                     static_cast<unsigned long long>(ancestor.second));
    }
    std::fputc(']', file);
    std::fprintf(file, ",\"native_composite\":%s",
                 record.native_composite ? "true" : "false");
    number_field("graph_kernel_count", record.graph_kernel_count);
    number_field("graph_dma_count", record.graph_dma_count);
    number_field("graph_barrier_count", record.graph_barrier_count);
    number_field("graph_child_count", record.graph_child_count);
    number_field("graph_segment_count", record.graph_segment_count);
    number_field("segment_id", record.segment.segment_id);
    number_field("start_idx", record.segment.start_idx);
    number_field("end_idx", record.segment.end_idx);
    number_field("manifest_metadata_count", record.segment.manifest_metadata_count());
    number_field("core_count", record.segment.core_count);
    number_field("kernel_count", record.segment.kernel_count);
    number_field("dma_count", record.segment.dma_count);
    number_field("barrier_count", record.segment.barrier_count);
    std::fputs("}\n", file);
    const bool write_failed = std::ferror(file) != 0;
    const bool close_failed = std::fclose(file) != 0;
    const bool published = !write_failed && !close_failed &&
        ::link(temporary.c_str(), record.companion_path.c_str()) == 0;
    ::unlink(temporary.c_str());  // Only the task-created temporary diagnostic.
    return published ? 0 : 3;
}

}  // anonymous namespace

uint64_t rpu_get_hwperf_evidence_failure_total() {
    return process_hwperf_evidence_failure_total.load(std::memory_order_relaxed);
}

void RpuKernelGraph::begin_execution_stats() {
    TORCH_CHECK(execution_ordinal_ != std::numeric_limits<uint64_t>::max(),
                "Graph execution ordinal exhausted");
    ++execution_ordinal_;
    last_stats_.last_segment_executions.clear();
    last_stats_.hwperf_evidence_failure_total = hwperf_evidence_failure_total_;
}

void RpuKernelGraph::record_segment_execution(
        const Segment& segment, ::rhino_lkn::Queue_t& queue,
        const char* execution_kind, const std::string& graph_name,
        bool hwperf_enabled, bool hwperf_prepared,
        const std::string& output_dir) {
    if (!hwperf_enabled) return;  // No per-segment tracing cost in ordinary inference.
    GraphSegmentExecution record;
    record.graph_lifetime_id = graph_lifetime_id_;
    record.build_generation = build_generation_;
    record.execution_ordinal = execution_ordinal_;
    // Bounded diagnostic state only; no second invocation log. Keep the actual
    // owners also for failure propagation without an unbounded/cyclic walk.
    std::vector<RpuKernelGraph*> execution_owners{this};
    bool ancestry_valid = true;
    for (auto* parent = execution_parent_; parent != nullptr;
         parent = parent->execution_parent_) {
        if (record.execution_ancestors.size() == 64 ||
            std::find(execution_owners.begin(), execution_owners.end(), parent) !=
                execution_owners.end()) {
            ancestry_valid = false;
            break;
        }
        execution_owners.push_back(parent);
        record.execution_ancestors.emplace_back(
            parent->graph_lifetime_id_, parent->execution_ordinal_);
    }
    if (execution_parent_ != nullptr) {
        record.parent_graph_lifetime_id = execution_parent_->graph_lifetime_id_;
        record.parent_execution_ordinal = execution_parent_->execution_ordinal_;
    }
    record.graph_name = graph_name;
    record.execution_kind = execution_kind;
    record.native_composite =
        composite_fmb_invocation_.phase != CompositeFmbInvocationState::Phase::None;
    record.graph_kernel_count = last_stats_.kernel_count;
    record.graph_dma_count = last_stats_.dma_count;
    record.graph_barrier_count = last_stats_.barrier_count;
    record.graph_child_count = last_stats_.child_graph_count;
    record.graph_segment_count = last_stats_.segment_count;
    record.segment = segment.census;
    record.hwperf_rc = !ancestry_valid ? -5 : !hwperf_prepared ? -3 : -2;
    if (ancestry_valid && hwperf_prepared && rpu_hw_perf_try_consume_dump()) {
        record.hwperf_sequence = rpu_hw_perf_next_dump_seq();
        record.hwperf_path = build_hw_perf_dump_path(
            output_dir, graph_name, std::strcmp(execution_kind, "replay") == 0,
            segment.segment_id, record.hwperf_sequence);
        record.hwperf_rc = record.hwperf_path.empty() ? -4 :
            queue.dump_hw_perf_chrome(record.hwperf_path.c_str());
        if (record.hwperf_rc == 0) {
            record.companion_path = record.hwperf_path + ".graph";
            record.companion_rc = write_hw_perf_graph_companion(record);
        }
        if ((record.hwperf_rc != 0 || record.companion_rc != 0) && log_at(2)) {
            std::cerr << "[HW_PERF] graph evidence failed: dump_rc="
                      << record.hwperf_rc << " companion_rc=" << record.companion_rc
                      << " path=" << record.hwperf_path << "\n";
        }
    }
    if (record.hwperf_rc != 0 || record.companion_rc != 0) {
        uint64_t previous = process_hwperf_evidence_failure_total.load(
            std::memory_order_relaxed);
        do {
            TORCH_CHECK(previous != std::numeric_limits<uint64_t>::max(),
                        "process HWPerf evidence failure counter exhausted");
        } while (!process_hwperf_evidence_failure_total.compare_exchange_weak(
            previous, previous + 1, std::memory_order_relaxed));
        // Child failures must remain visible even when that same child is
        // called successfully again later in the enclosing Python forward.
        for (RpuKernelGraph* owner : execution_owners) {
            TORCH_CHECK(owner->hwperf_evidence_failure_total_ !=
                            std::numeric_limits<uint64_t>::max(),
                        "HWPerf evidence failure counter exhausted");
            ++owner->hwperf_evidence_failure_total_;
            owner->last_stats_.hwperf_evidence_failure_total =
                owner->hwperf_evidence_failure_total_;
        }
    }
    last_stats_.last_segment_executions.push_back(std::move(record));
}

// =============================================================================
// execute_graph_for_recording — RECORDING 收尾
// =============================================================================
// build_segments_from_nodes + 段间交错执行 data nodes + 每段从 QueueCache 取
// Queue_t + prepare + launch。Queue_t 跨段复用 (按 core_num 全局缓存,进程
// 内最多 8 个),不再 per-segment 持有。
//
// 执行时序：
//   1. exec data[0 .. seg0.start_idx)
//   2. QueueCache.get + prepare + launch seg0 (enqueu_batch wait_finish=true)
//   3. exec data[seg0.end_idx .. seg1.start_idx)
//   4. QueueCache.get + prepare + launch seg1 (同一 Queue_t,build_batch 替换)
//   ...
//   N+1. exec data[segN.end_idx .. nodes_.size())
//
// 每个 enqueu_batch(wait_finish=true) 是 SYNC，所以下一组 data node 执行时
// 上一 segment kernel 已完成，保持 RECORDING 期 push 顺序等价性。

void RpuKernelGraph::execute_graph_for_recording() {
    begin_execution_stats();
    build_segments_from_nodes();
    // Phase 4 stats:Kernel / data node 计数 + segment 数 + A5+A6 HostCallback
    // tier 分桶 + Tier2 stable output 字节数 + non_replayable_reason mirror.
    // P6 (debug-level): hoisted ABOVE the INFO log line so the summary can
    // print last_stats_.kernel_count without recomputing.
    {
        size_t kn = 0, dn = 0;
        size_t hc1 = 0, hc2 = 0, hc3 = 0, hc_bytes = 0;
        size_t t3o = 0;
        for (auto& n : nodes_) {
            if (n.kind == GraphNodeKind::Kernel) {
                ++kn;
            } else {
                ++dn;
                if (n.kind == GraphNodeKind::HostCallback) {
                    auto& hc = n.as_host_callback();
                    if (hc.tier == HostCallbackTier::Tier1) ++hc1;
                    else if (hc.tier == HostCallbackTier::Tier2) {
                        ++hc2;
                        for (auto& t : hc.output_pool) {
                            if (t.defined()) hc_bytes += t.numel() * t.element_size();
                        }
                    } else if (hc.tier == HostCallbackTier::Tier3) ++hc3;
                } else if (n.kind == GraphNodeKind::Tier3Oneshot) {
                    ++t3o;
                }
            }
        }
        last_stats_.kernel_count = kn;
        last_stats_.data_node_count = dn;
        last_stats_.segment_count = segments_.size();
        last_stats_.host_callback_tier1_count = hc1;
        last_stats_.host_callback_tier2_count = hc2;
        last_stats_.host_callback_tier3_count = hc3;
        last_stats_.host_callback_output_bytes = hc_bytes;
        last_stats_.tier3_oneshot_count = t3o;
        last_stats_.non_replayable_reason = non_replayable_reason_;
        mirror_per_segment_replay_count(last_stats_, segments_);
    }
    // [GRAPH-INSTR] BUILD summary. P6 (debug-level): consolidated one-line
    // INFO output is emitted AFTER the segment loop, so it can carry the
    // `fresh_queue=K/N` count too — keeping both pieces of context on the
    // same line avoids the "looks like two separate events" UX confusion.
    // DEBUG-only kernel histogram + per-segment lines fire separately below.
    //
    // Read op_id from `pending_signature_` — `built_signature_` only gets
    // assigned AFTER execute_graph_for_recording returns (graph_runtime.cpp
    // line 217). `pending_signature_` was set during begin(sig).
    const std::string gid_storage =
        (pending_signature_.has_value() && !pending_signature_->op_id_str.empty())
            ? pending_signature_->op_id_str
            : std::string("<unlabeled>");
    const std::string& gid = gid_storage;
    // Per-BUILD global sequence so the 3 stderr lines (mix / per-seg / summary)
    // are visually identifiable as one event even when interleaved with
    // other output. Captured once here, threaded through the 3 emit sites.
    static std::atomic<int64_t> g_build_seq{0};
    const int64_t build_seq = ++g_build_seq;
    if (log_at(4)) {
        std::cerr << "[GRAPH-INSTR] BUILD " << gid << " build#" << build_seq
                  << " mix=[" << kernel_histogram_string(nodes_) << "]\n";
    }
    // Cold BUILD uses fallback slot 0. Optional additional prepared queues
    // are admitted lazily on REPLAY within the bounded process budget.
    const HwPerfConfig hw_perf_cfg = rpu_hw_perf_snapshot();

    size_t next_node = 0;
    size_t seg_idx_for_instr = 0;
    size_t fresh_alloc_count = 0;
    for (auto& seg : segments_) {
        for (size_t i = next_node; i < seg.start_idx; ++i) {
            execute_data_node(nodes_[i]);
        }
        const size_t slot_idx = prepared_queue_slot_index(
            seg_idx_for_instr, /*retain=*/false);
        auto& slot = private_queue_slots_[slot_idx];
        const bool was_present = slot.queue &&
            slot.core_num == static_cast<uint8_t>(seg.core_ids.size());
        ::rhino_lkn::Queue_t* wq = ensure_private_queue(slot_idx, seg.core_ids.size());
        const bool fresh = !was_present;
        if (fresh) ++fresh_alloc_count;
        // [GRAPH-INSTR] per-segment summary (FRESH = first allocation of this
        // entry's private queue for this core_num; cached = reuse). P2: gated
        // at DEBUG(4) because per-segment lines dominate stderr noise at the
        // default INFO level. P6: also prefix the op_id label.
        if (log_at(4)) {
            std::cerr << "[GRAPH-INSTR] BUILD " << gid << " build#" << build_seq << " "
                      << kernel_summary_string(nodes_, seg, seg_idx_for_instr)
                      << " => private_queue " << (fresh ? "FRESH" : "cached") << "\n";
        }

        // P2 — invalidate fingerprint 前先 reset,避免 build_batch 抛错留下半状态。
        // HW perf trace `set_enable_hw_perf` is asserted inside
        // prepare_segment_queue (shared with REPLAY-fallback-rebuild path).
        slot.built_segment_idx = -1;
        slot.fixed_dma_table.reset();
        slot.invalidate_mutable_dma_packets();
        RpuDmaSubmissionLease submission_lease;
        const uint32_t build_rc = prepare_segment_queue(
            seg, *wq, hw_perf_cfg.enabled, submission_lease);
        TORCH_CHECK(build_rc == 0,
                    "prepare_segment_queue: build_batch SDK rc=", build_rc,
                    " during recording on segment idx=", seg_idx_for_instr);
        const bool pending_kernel_tokens_valid =
            capture_segment_kernel_register_tokens(
                seg, slot.pending_kernel_register_tokens);
        // 首次 launch 紧跟 build_batch,kd_buf 里的 regs 是刚 build 时的当前值,
        // 不需要 sync_mutable_params。enqueu_batch(wait_finish=true) SYNC 返回
        // 后,本 entry 的 private_queue 处于 idle 态,可被下一 segment 复用
        // (build_batch 会清掉之前的 kd_buf state)。
        const uint32_t rc_build = wq->enqueu_batch(/*wait_finish=*/true);
        TORCH_CHECK(rc_build == 0,
                    "execute_graph_for_recording: enqueu_batch SDK rc=", rc_build,
                    " on segment idx=", seg_idx_for_instr);
        ++last_stats_.hw_batch_submit_total;
        // Publish the fingerprint only after successful synchronous submit.
        slot.fixed_dma_table = seg.fixed_dma_table;
        slot.built_segment_idx = static_cast<ssize_t>(seg_idx_for_instr);
        slot.hw_perf_enabled_at_build = hw_perf_cfg.enabled;
        slot.kernel_register_tokens.swap(
            slot.pending_kernel_register_tokens);
        slot.kernel_register_tokens_valid = pending_kernel_tokens_valid;

        // HW perf trace — dump BUILD-time Chrome JSON per segment, gated by
        // the process-global budget (max_dumps decremented atomically).
        record_segment_execution(
            seg, *wq, "build", gid, hw_perf_cfg.enabled,
            /*hwperf_prepared=*/true, hw_perf_cfg.output_dir);

        ++seg_idx_for_instr;
        next_node = seg.end_idx;
    }
    if (log_at(3)) {
        std::cerr << "[GRAPH-INSTR] BUILD " << gid << " build#" << build_seq
                  << ": segments=" << segments_.size()
                  << " nodes=" << nodes_.size()
                  << " kernels=" << last_stats_.kernel_count
                  << " fresh_queue=" << fresh_alloc_count << "/" << segments_.size()
                  << "\n";
    }
    for (size_t i = next_node; i < nodes_.size(); ++i) {
        execute_data_node(nodes_[i]);
    }
}

// =============================================================================
// execute_graph_for_replaying — REPLAYING 收尾
// =============================================================================
// Each segment reuses its prepared slot or rebuilds after a fingerprint miss;
// interleaved data nodes consume the bindings updated during capture replay.

void RpuKernelGraph::execute_graph_for_replaying() {
    begin_execution_stats();
    verify_semantic_spm_producer_yield_replay_arm(
        "execute_graph_for_replaying");
    RECORD_FUNCTION("rpu_graph::execute_replay", {});
    // P6 (debug-level): per-graph REPLAY visibility.
    //   INFO  (3): one-line "FIRST REPLAY" announcement on the transition
    //              BUILD→REPLAY-1 per signature (so users see "graph X started
    //              replaying"). Subsequent replays muted at INFO.
    //   DEBUG (4): one line per replay, including cumulative call count.
    ++total_replay_count_;
    {
        const std::string& gid =
            built_signature_.op_id_str.empty() ? std::string("<unlabeled>")
                                                : built_signature_.op_id_str;
        if (log_at(4)) {
            std::cerr << "[GRAPH-INSTR] REPLAY " << gid
                      << " call#" << total_replay_count_
                      << " segments=" << segments_.size()
                      << " kernels=" << last_stats_.kernel_count << "\n";
        } else if (total_replay_count_ == 1 && log_at(3)) {
            std::cerr << "[GRAPH-INSTR] REPLAY " << gid
                      << " FIRST hit: segments=" << segments_.size()
                      << " kernels=" << last_stats_.kernel_count << "\n";
        }
    }
    // During REPLAY, nodes and segments retain their built tier and byte counts.
    // Reuse those stable statistics from last_stats_; refresh only dynamic fields
    // such as segment count, non-replayable reason and per-segment replay count.
    {
        RECORD_FUNCTION("rpu_graph::replay_stats_refresh", {});
        last_stats_.segment_count = segments_.size();
        last_stats_.non_replayable_reason = non_replayable_reason_;
        // last_stats_.host_callback_tier{1,2,3}_count / host_callback_output_bytes
        // / tier3_oneshot_count 已在 RECORDING 期 (execute_graph_for_recording)
        // 写好,REPLAYING 不变。
    }
    // A profiling toggle invalidates the prepared batch fingerprint; replay
    // rebuilds with the current snapshot before publishing its census witness.
    const HwPerfConfig hw_perf_cfg_replay = rpu_hw_perf_snapshot();
    const std::string hw_perf_gid =
        built_signature_.op_id_str.empty() ? std::string("rpu_graph")
                                            : built_signature_.op_id_str;
    {
        RECORD_FUNCTION("rpu_graph::replay_segment_loop", {});
        size_t next_node = 0;
        for (auto& seg : segments_) {
            for (size_t i = next_node; i < seg.start_idx; ++i) {
                execute_data_node(nodes_[i]);
            }
            ::rhino_lkn::Queue_t* executed_queue =
                launch_segment_for_replay(seg, hw_perf_cfg_replay.enabled);
            ++seg.replay_count;

            record_segment_execution(
                seg, *executed_queue, "replay", hw_perf_gid,
                hw_perf_cfg_replay.enabled, /*hwperf_prepared=*/true,
                hw_perf_cfg_replay.output_dir);

            next_node = seg.end_idx;
        }
        for (size_t i = next_node; i < nodes_.size(); ++i) {
            execute_data_node(nodes_[i]);
        }
    }
    mirror_per_segment_replay_count(last_stats_, segments_);
    op_stream_fully_skipped_ = false;   // consume: next replay re-decides
    kernel_params_dirty_this_replay_ = false;
}

// =============================================================================
// execute_graph_oneshot — non-replayable 降级路径：按 segment 分组一次性提交
// kernel + data nodes use a function-local queue, independent of immediate
// DMA and retained Graph queues. Each completed segment returns its arena lease.
// =============================================================================

void RpuKernelGraph::execute_graph_oneshot() {
    begin_execution_stats();
    if (nodes_.empty()) return;

    build_segments_from_nodes();
    // 同 execute_graph_for_recording:扫一遍累计 tier 分桶 + Tier2 字节数。
    {
        size_t kn = 0, dn = 0;
        size_t hc1 = 0, hc2 = 0, hc3 = 0, hc_bytes = 0;
        size_t t3o = 0;
        for (auto& n : nodes_) {
            if (n.kind == GraphNodeKind::Kernel) {
                ++kn;
            } else {
                ++dn;
                if (n.kind == GraphNodeKind::HostCallback) {
                    auto& hc = n.as_host_callback();
                    if (hc.tier == HostCallbackTier::Tier1) ++hc1;
                    else if (hc.tier == HostCallbackTier::Tier2) {
                        ++hc2;
                        for (auto& t : hc.output_pool) {
                            if (t.defined()) hc_bytes += t.numel() * t.element_size();
                        }
                    } else if (hc.tier == HostCallbackTier::Tier3) ++hc3;
                } else if (n.kind == GraphNodeKind::Tier3Oneshot) {
                    ++t3o;
                }
            }
        }
        last_stats_.kernel_count = kn;
        last_stats_.data_node_count = dn;
        last_stats_.segment_count = segments_.size();
        last_stats_.host_callback_tier1_count = hc1;
        last_stats_.host_callback_tier2_count = hc2;
        last_stats_.host_callback_tier3_count = hc3;
        last_stats_.host_callback_output_bytes = hc_bytes;
        last_stats_.tier3_oneshot_count = t3o;
        last_stats_.non_replayable_reason = non_replayable_reason_;
        mirror_per_segment_replay_count(last_stats_, segments_);
    }
    // HW perf trace — oneshot path. No cache, no REPLAY; dump per segment
    // once, gated by global budget. set_enable_hw_perf is asserted inside
    // prepare_segment_queue (shared with BUILD/REPLAY-fallback).
    const HwPerfConfig hw_perf_cfg_oneshot = rpu_hw_perf_snapshot();
    const std::string hw_perf_gid_oneshot =
        (pending_signature_.has_value() &&
         !pending_signature_->op_id_str.empty())
            ? pending_signature_->op_id_str
            : (built_signature_.op_id_str.empty()
                   ? std::string("rpu_graph_oneshot")
                   : built_signature_.op_id_str);

    size_t next_node = 0;
    size_t seg_idx_oneshot = 0;
    // Match Launch's core count at construction and return every completed
    // arena lease before running the next segment or interleaved data node.
    std::unique_ptr<::rhino_lkn::Queue_t> one_shot_queue;
    size_t one_shot_queue_core_count = 0;
    for (auto& seg : segments_) {
        for (size_t i = next_node; i < seg.start_idx; ++i) {
            execute_data_node(nodes_[i]);
        }
        const size_t seg_core_count = seg.core_ids.size();
        TORCH_CHECK(seg_core_count >= 1 && seg_core_count <= 8,
                    "execute_graph_oneshot: segment core count must be in [1, 8]");
        if (!one_shot_queue || seg_core_count != one_shot_queue_core_count) {
            one_shot_queue = std::make_unique<::rhino_lkn::Queue_t>(
                static_cast<uint8_t>(seg_core_count));
            one_shot_queue_core_count = seg_core_count;
        }
        ::rhino_lkn::Queue_t* wq = one_shot_queue.get();
        // 复用 prepare_segment_queue,自动处理 Kernel / Dma / Barrier 三种段内节点。
        // oneshot 不需要 sync_mutable_params 复用 kd_buf,prepare 内部用
        // add_kernel_mutable 没有副作用(kd_buf 一次性执行后丢弃)。
        // P2: prepare_segment_queue 现在改为 Segment& (要填 seg.mutable_dmas);
        // oneshot 路径填的 slot 在下面 segments_.clear() 时就丢了,无副作用。
        RpuDmaSubmissionLease submission_lease;
        const uint32_t build_rc = prepare_segment_queue(
            seg, *wq, hw_perf_cfg_oneshot.enabled, submission_lease,
            /*retain_replay_state=*/false);
        TORCH_CHECK(build_rc == 0,
                    "prepare_segment_queue: build_batch SDK rc=", build_rc,
                    " during one-shot on segment idx=", seg_idx_oneshot);
        const uint32_t rc_oneshot = wq->enqueu_batch(/*wait_finish=*/true);
        TORCH_CHECK(rc_oneshot == 0,
                    "execute_graph_oneshot: enqueu_batch SDK rc=", rc_oneshot,
                    " on segment idx=", seg_idx_oneshot);
        ++last_stats_.hw_batch_submit_total;

        record_segment_execution(
            seg, *wq, "oneshot", hw_perf_gid_oneshot,
            hw_perf_cfg_oneshot.enabled, /*hwperf_prepared=*/true,
            hw_perf_cfg_oneshot.output_dir);
        const uint32_t rc_discard = wq->discard_completed_batch();
        TORCH_CHECK(rc_discard == 0,
                    "execute_graph_oneshot: discard_completed_batch SDK rc=",
                    rc_discard, " on segment idx=", seg_idx_oneshot);

        ++seg_idx_oneshot;
        next_node = seg.end_idx;
    }
    for (size_t i = next_node; i < nodes_.size(); ++i) {
        execute_data_node(nodes_[i]);
    }
    mirror_per_segment_replay_count(last_stats_, segments_);
    segments_.clear();
}
