// graph_runtime_execute.cpp — kernel dispatch, segment launch, and node execution.
//
// This file contains the hot replay/record execution path: GET_KERNEL-backed
// enqueue, fast replay skip helpers, segment construction/preparation, data node
// execution, and execute_graph_* loops.
#include "graph/graph_runtime.h"
#include "graph/graph_runtime_internal.h"
#include "rpu_copy_safety.h"
#include "rpu_profile.h"        // log_at(N) debug-level gate

#include <ATen/record_function.h>
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
    // Reject unsupported dynamic transports before they can record an unowned
    // node inside a canonical FMB DMA trace.
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

// Direct enqueu_kernel invalidates any prepared batch on the same Queue_t, so keep PASSTHROUGH
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
            *prog, KERNEL_ID_NAMES[static_cast<uint8_t>(id)]);
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
        KERNEL_ID_NAMES[static_cast<uint8_t>(id)], "')");
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
    switch (state_) {
    case State::PASSTHROUGH: {
        // Keep the direct path on a queue
        // isolated from immediate DMA batches, because enqueu_kernel invalidates
        // prepared state on the same Queue_t.
        ::rhino_lkn::Queue_t* wq =
            passthrough_kernel_queue(core_ids.size());
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
        // 即 raw/dynamic kernel。此时按以下规则降级：
        //   1. execute_graph_oneshot() 落地已录前缀并保持录制顺序
        //   2. 对前缀做 pre + post flush_boundary_ptrs()，保证本轮 kernel
        //      之前的 CPU 产物和前缀 kernel 之后的写回都与 eager 语义一致
        //   3. clear boundary_flush_ptrs_ —— 切 PASSTHROUGH 后后续 flush
        //      走 eager 分支，不再使用这个集合
        //   4. 清 kernels_ / nodes_ / pending_kernel_id_ + state=PASSTHROUGH
        //   5. eager launch raw kernel（复用 PASSTHROUGH 路径）
        //
        // 为什么 state 必须切到 PASSTHROUGH，不能保持 RECORDING 让后缀继续录：
        // rpu_bmm 这类 raw-kernel op **kernel 之后紧跟 spm_to_ddr**，其中 src
        // 是栈上 LocalSPM_t 的 cpu_ptr。若
        // 保持 RECORDING，spm_to_ddr 会被 capture_data 进 nodes_ 延迟到
        // end() 才执行，期间 LocalSPM_t 已析构、SPM slot 被复用 → 从已复用
        // 的地址读取垃圾数据。PASSTHROUGH 保证 spm_to_ddr 紧跟 kernel eager
        // 执行，时序正确。
        // Dynamic kernel API:pending_kernel_name_ 也算"正规路径"标记。只有
        // 两者都空时才是 raw-kernel 降级（调用者既没走 GET_KERNEL(id) 也
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
// skip_op_stream_for_fast_replay
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
    // No op-stream nodes re-emitted this replay → no kernel set_regs → kd_buf
    // params are unchanged → sync_mutable_params() is redundant (see member doc).
    op_stream_fully_skipped_ = true;
}

// =============================================================================
// mark_post_fn_cursor / skip_layer_body_for_fast_replay — post_fn coexistence
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
// 执行。seg.core_ids 永远 = max(kn.num_cores)→[0..max-1],仅做 Queue_t 构造
// 尺寸 hint。
// =============================================================================

// 诊断 helper:读 RPU_P7_SPLIT_KERNELS env (comma-separated kernel-name
// substrings),把匹配 kernel 强制隔成 solo segment (seg.core_ids = kn.core_ids,
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
        const KernelNodeData& kn) {
    // Default OFF: mixed 1-core patch-embed and 8-core encoder kernels share a
    // segment. RPU_SIGLIP_ISOLATE_PATCH_EMBED=1 enables dedicated segments.
    static const bool kEnabled = [] {
        const char* e = std::getenv("RPU_SIGLIP_ISOLATE_PATCH_EMBED");
        return e && *e && std::string(e) != "0" &&
               std::string(e) != "false" && std::string(e) != "False";
    }();
    if (!kEnabled) {
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
    // principle, but graph.end() can fail when these kernels are merged into
    // the same segment.
    return label.find("pad_const_NxC_NxCP") != std::string::npos ||
           label.find("transpose_ncb_c16") != std::string::npos ||
           label.find("im2col_coren") != std::string::npos ||
           label.find("_1core_") != std::string::npos;
}

// Segment replay batches using Rhino Launch's exact resource accounting and
// conservative soft limits. A single indivisible node or fenced multi-stream
// DMA burst may exceed a soft limit; the Launch hard limit remains authoritative.
struct SegmentResourceBudget {
    size_t entries;
    size_t kd_bytes;
    size_t instr_bytes;
};

struct SegmentResourceUse {
    size_t entries = 0;
    size_t kd_bytes = 16;  // Queue_t::build_batch final + end-marker packets.
    size_t instr_bytes = 0;
};

static size_t segment_env_limit(const char* name,
                                size_t default_value,
                                size_t max_value,
                                bool megabytes) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') return default_value;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (end == text || parsed == 0) return default_value;
    if (!megabytes) {
        return std::min(static_cast<size_t>(parsed), max_value);
    }
    const size_t max_mb = max_value >> 20;
    return std::min(static_cast<size_t>(parsed), max_mb) << 20;
}

static bool lingbot2_exact_graph_owns_sdk_budget(
        const std::string& graph_label,
        const SegmentResourceBudget& sdk_budget) {
    const char* enabled = std::getenv("RPU_LINGBOT2_MULTIVIEW_SPM_Z2");
    if (enabled == nullptr || enabled[0] != '1' || enabled[1] != '\0' ||
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
        const std::string& graph_label) {
    constexpr size_t kMiB = 1u << 20;
    const size_t sdk_entries = segment_env_limit(
        "LKN_MAX_BATCH_ENTRIES", 65536, 1u << 22, false);
    const size_t sdk_kd = segment_env_limit(
        "LKN_KD_BUF_MB", 8 * kMiB, 256 * kMiB, true);
    const size_t sdk_instr = segment_env_limit(
        "LKN_INSTR_BUF_MB", 64 * kMiB, 1024 * kMiB, true);
    const SegmentResourceBudget sdk_budget{
        sdk_entries, sdk_kd, sdk_instr};
    // The exact LingBot profile that shipped as one SDK batch retains only its
    // sealed envelope. Every other graph keeps the generic Qwen safety splitter.
    if (lingbot2_exact_graph_owns_sdk_budget(graph_label, sdk_budget)) {
        return sdk_budget;
    }
    return {
        std::max<size_t>(1, std::min<size_t>(8192, sdk_entries / 2)),
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
        uint64_t* args = nullptr;
        char* instr = nullptr;
        size_t args_size = 0;
        size_t instr_size = 0;
        kernels[kn.kernel_idx]->get_kernel_args_instr(
            &args, &args_size, &instr, &instr_size);
        (void)args;
        (void)instr;
        use.kd_bytes = 48 + args_size;
        TORCH_CHECK(instr_size <= std::numeric_limits<size_t>::max() - 0xFFF,
                    "build_segments_from_nodes: kernel instruction size overflow");
        use.instr_bytes = (instr_size + 0xFFF) & ~size_t{0xFFF};
        break;
    }
    case GraphNodeKind::Dma:
        use.kd_bytes = 20 * sizeof(uint64_t);
        break;
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
    return sum_exceeds(current.entries, next.entries, budget.entries) ||
           sum_exceeds(current.kd_bytes, next.kd_bytes, budget.kd_bytes) ||
           sum_exceeds(current.instr_bytes, next.instr_bytes,
                       budget.instr_bytes);
}

void RpuKernelGraph::build_segments_from_nodes() {
    segments_.clear();
    if (nodes_.empty()) return;

    const std::string graph_label =
        (pending_signature_.has_value() && !pending_signature_->op_id_str.empty())
            ? pending_signature_->op_id_str
            : (built_signature_.op_id_str.empty()
                   ? std::string{}
                   : built_signature_.op_id_str);

    // 切段条件：
    //   1. queue_state 变化 (BARRIER_NEXT / instr-pause 等差异)
    //   2. data 节点 (Memcpy / Memset / ChildGraph / Branch /
    //      HostCallback / Tier3Oneshot 切段;Dma / Barrier 不切)
    //   3. RPU_P7_SPLIT_KERNELS env 命中 — 把匹配 kernel 隔成 solo segment
    //      (诊断 only,bisect 单 kernel 正确性问题)
    //   4. SDK footprint 达到共享安全预算 — 在跨 stream DMA fence 完整闭合的
    //      边界贪心切段，避免大图虽 build_batch rc=0 却在硬件静默错算。
    constexpr size_t kNoOpenSeg = std::numeric_limits<size_t>::max();
    const SegmentResourceBudget resource_budget =
        segment_resource_budget(graph_label);
    size_t seg_start = kNoOpenSeg;
    uint8_t seg_max_num_cores = 0;
    std::vector<uint8_t> seg_isolated_core_ids;  // 非空 = SPLIT_KERNELS 命中的 solo 段
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
        seg_resources.kd_bytes += use.kd_bytes;
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
        if (!seg_isolated_core_ids.empty()) {
            seg.core_ids = std::move(seg_isolated_core_ids);
        } else {
            const uint8_t n = seg_max_num_cores == 0 ? uint8_t{8}
                                                     : seg_max_num_cores;
            seg.core_ids.resize(n);
            for (uint8_t k = 0; k < n; ++k) seg.core_ids[k] = k;
        }
        seg.queue_state = seg_queue_state;
        segments_.push_back(std::move(seg));
        peak_entries = std::max(peak_entries, seg_resources.entries);
        peak_kd_bytes = std::max(peak_kd_bytes, seg_resources.kd_bytes);
        peak_instr_bytes = std::max(peak_instr_bytes, seg_resources.instr_bytes);
        seg_start = kNoOpenSeg;
        seg_max_num_cores = 0;
        seg_isolated_core_ids.clear();
        resource_reset();
    };

    for (size_t i = 0; i < nodes_.size(); ++i) {
        switch (nodes_[i].kind) {
        case GraphNodeKind::Kernel: {
            auto& kn = nodes_[i].as_kernel();
            const uint8_t kn_num = static_cast<uint8_t>(kn.core_ids.size());
            if (p7_should_force_isolate_kernel(kn) ||
                siglip_should_isolate_patch_embed_kernel(graph_label, kn)) {
                close_segment(i);
                seg_start = i;
                seg_max_num_cores = kn_num;
                seg_isolated_core_ids = kn.core_ids;
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
            // 顶替原 BATCH_CTX.add_dma / add_barrier 顺序进 batch 的语义。
            // 不约束 segment.queue_state;若 segment 还没开始,用 8 核作为
            // Queue_t 构造尺寸 hint (matches v5 BATCH_CTX 典型 core_ids.size()==8)。
            if (seg_start == kNoOpenSeg) {
                seg_start = i;
                seg_max_num_cores = 8;
                seg_queue_state = QueueLaunchState{};
            }
            account_node(nodes_[i], node_resources);
            // else: extend current segment
            break;
        }
        default:
            // 其它 data node (Memcpy / Memset / ChildGraph / Branch /
            // HostCallback / Tier3Oneshot) 按段边界处理。
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

    // HALO_NODE_HIST: env 门控的节点直方图诊断。在 enqueu_batch 之前打印，
    // 即使段超 32768 也能看到构成；按 kind + kernel 标签计数。
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

void RpuKernelGraph::release_prepared_queues() {
    // Queue_t 是 per-cache-entry (private_queue_),依靠 RpuKernelGraph
    // 析构(RpuGraphCache::evict / clear)自动释放;Python ABI hook 仍是 no-op
    // (callers 不需要主动释放;若需要诊断可显式 reset)。
}

::rhino_lkn::Queue_t* RpuKernelGraph::ensure_private_queue(size_t num_cores) {
    const uint8_t want = static_cast<uint8_t>(num_cores);
    if (!private_queue_ || private_queue_core_num_ != want) {
        // 不同 segment 最大 core 域时丢掉旧 queue 重建(RAII)。段内混核不走
        // 本分支：queue 按最大域构造，每个 kernel 的 core_ids 在
        // prepare_segment_queue 中独立传给 SDK。跨段重建只丢失 kd_buf reuse
        // 收益，不影响正确性。
        private_queue_ = std::make_unique<::rhino_lkn::Queue_t>(num_cores);
        private_queue_core_num_ = want;
        // 新 Queue 没有已 build 的 kd_buf，fingerprint 失效。
        private_queue_built_segment_idx_ = -1;
    }
    return private_queue_.get();
}

void RpuKernelGraph::prepare_segment_queue(
        Segment& seg,
        ::rhino_lkn::Queue_t& wq) {
    // Prepared-queue contract:
    // - Kernel 走 add_kernel_mutable (regs 后续可通过 sync_mutable_params 重刷)。
    // - Mutable DMA 走 add_dma_kernel_mutable (返回的 slot 索引按 SDK 内部
    //   sequential 计数,与 next_mutable_dma_id 一致),同时 push 一条
    //   PreparedMutableDmaSlot 进 seg.mutable_dmas 供 sync-only REPLAY 用。
    // - Fixed DMA / Barrier 走非 mutable 入口,**不**增 dma_id (与 SDK 一致)。
    // BUILD 结尾的 build_batch() 把 kd_buf 落定;后续若调 sync_mutable_params +
    // enqueu_batch 即等价于 BATCH_CTX 时代的 prepare-once-launch-many 快路径。
    seg.mutable_dmas.clear();
    uint32_t next_mutable_dma_id = 0;
    // HW perf trace MUST be set before build_batch because the flag is baked
    // into kd_buf packets.
    // Asserted here so both BUILD (execute_graph_for_recording) and REPLAY-
    // fallback-rebuild (launch_segment_for_replay) paths get coverage from a
    // single site. **Symmetric set required**: Queue_t::set_enable_hw_perf
    // is sticky;once turned on (e.g. previous run with --hw-perf),subsequent
    // runs without --hw-perf would still embed perf packets in kd_buf unless
    // we explicitly set it to false here. Each segment build re-asserts the
    // current global state, ensuring no stale sticky-flag carry-over.
    wq.set_enable_hw_perf(rpu_hw_perf_snapshot().enabled);
    wq.set_broadcast_mode(seg.queue_state.broadcast_mode);
    wq.set_flush_icache(seg.queue_state.flush_icache);
    for (size_t i = seg.start_idx; i < seg.end_idx; ++i) {
        const auto& node = nodes_[i];
        switch (node.kind) {
        case GraphNodeKind::Kernel: {
            const auto& kn = node.as_kernel();
            // Per-kernel core_ids support mixed 1-core/8-core kernels in one
            // segment. The queue packs (kn.core_ids.size()<<16 | kn.core_ids[0])
            // into each kernel's R1_KCORE_USED_PARAM_ADDR;
            // 1-core ROPE 拿到 (1<<16|0) + reg21=1,只 dispatch 到 core 0,
            // 不浪费 8 核冗余 fetch(对比 BATCH_CTX 时代的强制 8 核 dispatch)。
            rpu_add_kernel_mutable_checked(wq, *kernels_[kn.kernel_idx],
                                           kn.grid_dims, kn.core_ids,
                                           "prepare_segment_queue/Kernel");
            break;
        }
        case GraphNodeKind::Dma: {
            const auto& dma = node.as_dma();
            validate_semantic_dma_owner_for_execution(
                dma, i, "prepare_segment_queue");
            uint64_t src = dma.src_addr;
            uint64_t dst = dma.dst_addr;
            if (dma.variant == DmaNodeData::Variant::MutableSrc) {
                TORCH_CHECK(dma.live_base != nullptr,
                            "prepare_segment_queue: MutableSrc DMA missing "
                            "live_base at node ", i);
                src = (*dma.live_base) +
                      static_cast<uint64_t>(dma.live_offset);
                rpu_add_dma_mutable_checked(wq, src, dst, dma.bytes,
                                            dma.channel,
                                            "prepare_segment_queue/MutableSrc");
                seg.mutable_dmas.push_back(Segment::PreparedMutableDmaSlot{
                    /*node_idx*/   i,
                    /*dma_id*/     next_mutable_dma_id++,
                    Segment::PreparedMutableDmaSlot::Kind::MutableSrc,
                    /*live_base*/  dma.live_base,
                    /*live_offset*/dma.live_offset,
                    /*fixed_addr*/ dst,
                    /*bytes*/      dma.bytes,
                    /*channel*/    dma.channel,
                });
            } else if (dma.variant == DmaNodeData::Variant::MutableDst) {
                TORCH_CHECK(dma.live_base != nullptr,
                            "prepare_segment_queue: MutableDst DMA missing "
                            "live_base at node ", i);
                dst = (*dma.live_base) +
                      static_cast<uint64_t>(dma.live_offset);
                rpu_add_dma_mutable_checked(wq, src, dst, dma.bytes,
                                            dma.channel,
                                            "prepare_segment_queue/MutableDst");
                seg.mutable_dmas.push_back(Segment::PreparedMutableDmaSlot{
                    /*node_idx*/   i,
                    /*dma_id*/     next_mutable_dma_id++,
                    Segment::PreparedMutableDmaSlot::Kind::MutableDst,
                    /*live_base*/  dma.live_base,
                    /*live_offset*/dma.live_offset,
                    /*fixed_addr*/ src,
                    /*bytes*/      dma.bytes,
                    /*channel*/    dma.channel,
                });
            } else {
                // Fixed:non-mutable 入口,不增 dma_id (SDK 内部计数也只对
                // add_dma_kernel_mutable 递增)。
                rpu_add_dma_checked(wq, src, dst, dma.bytes, dma.channel,
                                    "prepare_segment_queue/Fixed");
            }
            break;
        }
        case GraphNodeKind::Barrier: {
            const auto& b = node.as_barrier_node();
            wq.add_barrier(b.self_stream, b.target_stream);
            break;
        }
        default:
            TORCH_CHECK(false,
                        "prepare_segment_queue: unexpected node kind ",
                        static_cast<int>(node.kind), " in segment at node ",
                        i, " (only Kernel / Dma / Barrier allowed)");
        }
    }
    // SDK Queue_t::build_batch() returns a non-zero rc on failure:
    // return 1 for no kernels / return 2 for batch entries > kMaxBatchKernels
    // [= lkn_max_batch_entries(), default 65536, env LKN_MAX_BATCH_ENTRIES] /
    // kd_buf|instr_buf overflow — the 8MB kd-buffer is the real practical cap) —
    // it LOG_ERRORs and RETURNS, it does NOT throw. Discarding the rc would let an
    // over-limit segment proceed with a half-built batch → silent corruption /
    // downstream failure. Convert it to a loud, universal guard so an input or
    // model change that exceeds an SDK limit surfaces here.
    const uint32_t build_rc = wq.build_batch();
    TORCH_CHECK(build_rc == 0,
                "prepare_segment_queue: build_batch SDK rc=", build_rc,
                " (!=0 = no kernels / batch entries > kMaxBatchKernels [default "
                "65536] / kd_buf|instr_buf overflow); segment nodes=",
                (seg.end_idx - seg.start_idx));
}

void RpuKernelGraph::launch_segment_for_replay(Segment& seg) {
    RECORD_FUNCTION("rpu_graph::segment_launch", {});

    // segment fingerprint = (segments_ 中的指针位置,core_num)。后续判定 sync-only
    // 资格,以及为 fallback 路径写新值时用。指针算术安全:segments_ 在 BUILT 期
    // 不重新分配,seg 始终是 segments_[idx]。
    const ssize_t seg_idx = static_cast<ssize_t>(&seg - segments_.data());

    // Sync-only path: when this segment still owns the private queue's prepared
    // batch, update mutable state and enqueue without rebuilding it.
    if (private_queue_ &&
        private_queue_built_segment_idx_ == seg_idx &&
        private_queue_core_num_ == static_cast<uint8_t>(seg.core_ids.size())) {
        RECORD_FUNCTION("rpu_graph::segment_launch_sync_only", {});
        const uint32_t rc = launch_segment_sync_only(seg);
        TORCH_CHECK(rc == 0,
                    "launch_segment_sync_only: SDK rc=", rc,
                    " on segment idx=", seg_idx,
                    " (mutable_dmas.size=", seg.mutable_dmas.size(), ")");
        ++last_stats_.prepared_segment_hit_total;
        ++last_stats_.hw_batch_submit_total;
        return;
    }

    // Fallback: full rebuild on private_queue_。覆写前先把 idx
    // 失效,防止异常路径 (build_batch 抛错) 留下半 build 状态被下次误判命中。
    ::rhino_lkn::Queue_t* wq = ensure_private_queue(seg.core_ids.size());
    private_queue_built_segment_idx_ = -1;
    prepare_segment_queue(seg, *wq);
    const uint32_t rc_fallback = wq->enqueu_batch(/*wait_finish=*/true);
    TORCH_CHECK(rc_fallback == 0,
                "launch_segment_for_replay(fallback): enqueu_batch SDK rc=",
                rc_fallback, " on segment idx=", seg_idx);
    ++last_stats_.hw_batch_submit_total;
    private_queue_built_segment_idx_ = seg_idx;
    ++last_stats_.prepared_segment_miss_total;
}

uint32_t RpuKernelGraph::launch_segment_sync_only(Segment& seg) {
    // 前提:caller (launch_segment_for_replay) 已确认 private_queue_ 的 kd_buf
    // 是本段刚 build 的状态;mutable_dmas 与 SDK 内部 dma_id 计数一致。
    //
    // 顺序:先 update_dma_kernel 把每个 mutable DMA 的 live src/dst 重写到本轮
    // dev_addr,再 sync_mutable_params 把本轮 op stream 调过 set_regs 的
    // kernel 参数刷进 kd_buf,最后 enqueu_batch 一次发射。
    //
    // size=0 是 SDK address-only 路径,避免重走完整的
    // add_dma_kernel_mutable + build_batch 链。
    //
    // Fixed DMA addresses freeze at BUILD. Per-forward src/dst addresses must
    // use the mutable variant with stable live-base storage, or replay uses a
    // stale DDR address.
    for (const auto& slot : seg.mutable_dmas) {
        TORCH_CHECK(
            slot.node_idx < nodes_.size() &&
                nodes_[slot.node_idx].kind == GraphNodeKind::Dma,
            "launch_segment_sync_only: mutable DMA node sidecar is stale at "
            "node ", slot.node_idx);
        const auto& dma = nodes_[slot.node_idx].as_dma();
        TORCH_CHECK(
            dma.semantic_endpoint_id == 0 ||
                (dma.live_base == slot.live_base &&
                 dma.live_offset == slot.live_offset),
            "launch_segment_sync_only: mutable DMA slot drift at node ",
            slot.node_idx);
        const uint64_t live = (*slot.live_base) +
                              static_cast<uint64_t>(slot.live_offset);
        uint64_t src, dst;
        if (slot.kind == Segment::PreparedMutableDmaSlot::Kind::MutableSrc) {
            src = live;
            dst = slot.fixed_addr;
        } else {
            src = slot.fixed_addr;
            dst = live;
        }
        const uint32_t rc = private_queue_->update_dma_kernel(
            slot.dma_id, src, dst, /*size=*/0);
        if (rc != 0) {
            // 直接返回给 caller 让 TORCH_CHECK 给出 dma_id / segment idx 上下文。
            return rc;
        }
    }
    // Skip the redundant param re-sync when the op-stream was FULLY skipped this
    // replay (fast-replay, no post_fn) — no set_regs ran, so the kd_buf kernel
    // params are unchanged from BUILD/last-sync. The mutable DMAs above are written
    // directly via update_dma_kernel (independent of sync_mutable_params). Gated
    // on RPU_FASTREPLAY_SKIP_SYNC.
    if (!(fast_replay_skip_sync_ && op_stream_fully_skipped_)) {
        const uint32_t rc_sync = private_queue_->sync_mutable_params();
        if (rc_sync != 0) return rc_sync;
    }
    return private_queue_->enqueu_batch(/*wait_finish=*/true);
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
// 把 data-node kind 映射成稳定的 trace 名，供 Chrome tracing 直接搜索。
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
            const int ch = m.dma_channel >= 0 ? m.dma_channel : 0;
            const int32_t copy_rc = rhino_lkn::CopyMemoryChannel(
                m.dst_dev_addr, m.src_dev_addr, m.bytes, ch);
            TORCH_CHECK(copy_rc == 0,
                        "execute_data_node: CopyMemoryChannel SDK rc=",
                        copy_rc);
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
        // Child replay participates in the parent lifecycle. The parent already
        // provides boundary flushes around the executor.
        if (c.data_patches.empty()) {
            c.child->replay_prepared_child(c.input_patches);
        } else {
            c.child->replay_prepared_child_with_data_patches(
                c.input_patches, c.data_patches);
        }
        break;
    }
    case GraphNodeKind::Branch:
        // BranchNode is a marker/segment boundary only. Cache separation is
        // handled by GraphSignature.branch_key; device-side branch dispatch is
        // handled separately from device-side branch dispatch.
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
        // HostCallback executor checksum 日志由 RPU_GRAPH_HCB_CHECKSUM=1
        // 启用，默认关闭。它记录 _to_copy 节点的 live input、fresh output 和
        // stable post-copy 三段 checksum，用于定位 stable refresh 或 fresh
        // tensor freshness 问题。
        static const bool hcb_checksum = []() {
            const char* e = std::getenv("RPU_GRAPH_HCB_CHECKSUM");
            return e && *e && *e != '0';
        }();
        const std::string hcb_op_name = hc.op_handle->operator_name().name;
        const bool hcb_log = hcb_checksum && hcb_op_name == "aten::_to_copy";
        auto hcb_checksum_str = [](const at::Tensor& t) -> std::string {
            if (!t.defined()) return "<undef>";
            std::ostringstream o;
            o << "shape=" << t.sizes() << " dtype=" << t.dtype();
            o << " dev=0x" << std::hex
              << reinterpret_cast<uint64_t>(t.data_ptr()) << std::dec;
            try {
                at::Tensor cpu = t.is_cpu() ? t : t.cpu();
                auto stype = cpu.scalar_type();
                if (stype == c10::kHalf || stype == c10::kFloat ||
                    stype == c10::kDouble) {
                    o << " sum=" << cpu.to(at::kDouble).sum().item<double>();
                } else if (stype == c10::kLong) {
                    o << " sum=" << cpu.sum().item<int64_t>();
                } else if (stype == c10::kInt) {
                    o << " sum=" << cpu.sum().item<int32_t>();
                } else if (stype == c10::kBool) {
                    o << " any=" << cpu.any().item<bool>();
                }
                if (cpu.numel() > 0 && cpu.numel() <= 4) {
                    o << " val=" << cpu;
                }
            } catch (const std::exception& e) {
                o << " sum=<err:" << e.what() << ">";
            }
            return o.str();
        };
        const char* hcb_state = state_ == State::RECORDING ? "REC"
                              : state_ == State::REPLAYING ? "REP"
                              : "OS"; /* oneshot */
        if (hcb_log) {
            std::cerr << "[HCB-CK] " << hcb_op_name << " state=" << hcb_state
                      << " node=" << (&node - nodes_.data())
                      << " stage=in:";
            for (size_t k = 0; k < hc.live_tensor_args.size(); ++k) {
                std::cerr << "\n  live[" << k << "]: "
                          << hcb_checksum_str(hc.live_tensor_args[k]);
            }
            std::cerr << "\n";
        }
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
        //     corruption。原 .out
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
                if (hcb_log) {
                    std::cerr << "[HCB-CK] " << hcb_op_name << " state="
                              << hcb_state << " stage=fresh_out["
                              << r << "]: " << hcb_checksum_str(fresh_out)
                              << "\n";
                }
                // copy CPU/RPU fresh tensor → stable RPU buffer
                stable.copy_(fresh_out);
                // 替换 stack slot,下游(若有)直接读 stable
                ret = stable;
                // 登记 post-CPU boundary flush:host 写入对后续 RPU 可见
                record_boundary_flush(stable.data_ptr());
                if (hcb_log) {
                    std::cerr << "[HCB-CK] " << hcb_op_name << " state="
                              << hcb_state << " stage=stable_post_copy["
                              << r << "]: " << hcb_checksum_str(stable)
                              << "\n";
                }
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
        if (hcb_log) {
            std::cerr << "[HCB-CK] " << hcb_op_name << " state=" << hcb_state
                      << " stage=post_flush_done\n";
        }
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
        // Tier3OneshotNode 节点级 oneshot 真跑路径。三条 execute
        // path 共用本逻辑:
        //   RECORDING-collapse (execute_graph_for_recording):RECORDING 期
        //     capture_cpu_fallback 已经真跑过一次给当前 forward 下游用,这里
        //     end() 时重建 fresh return + copy 进 transient buffer 给 RECORDING
        //     期 set_regs 写好的下游 segment kernel 拿到正确的 RNG 真值。
        //   oneshot (execute_graph_oneshot):保留 non-replayable scope 的兜底路径。
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
        // (4) 把 fresh return 数据 copy 到 transient buffer（地址稳定，
        //     内容刷新)。下游 RPU kernel 在 RECORDING 期 set_regs 写的是 transient
        //     buffer dev_addr,oneshot 跑下游时读到的就是本步刚 copy 进去的
        //     RNG 真值。
        //     不再覆写 t.last_output_dev_addrs — 那是 RECORDING adopt_returns
        //     设的 transient buffer dev_addr（跨 replay 稳定），被依赖推断
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
// GRAPH-INSTR helpers
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
    case KernelId::RMS_NORM_BF16IN_FP16OUT_V32:
                                            return "RMSNORM_BF16IN";
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
    case KernelId::PL_ACC32_OUT_BF16_M416N48:
    case KernelId::PL_ACC32_OUT_BF16_M496N32:
                                            return "LINEAR_ACC32_BF16";
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
    case KernelId::LLM_ALL_REDUCE_SUM_RESIDUAL:
                                            return "ALLREDUCE_RESIDUAL";
    case KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_PACED:
                                            return "ALLREDUCE_RING_PACED";
    case KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_NOPACE:
                                            return "ALLREDUCE_RING_NOPACE";
    case KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_LOCAL_SPM:
                                            return "ALLREDUCE_RING_LOCAL_SPM";
    case KernelId::LLM_ALL_REDUCE_BF16_PARTIAL_FP16_RESIDUAL:
    case KernelId::LLM_ALL_REDUCE_FP16_PARTIAL_BF16_RESIDUAL:
                                            return "ALLREDUCE_MIXED";
    case KernelId::LLM_ALL_REDUCE_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_SPM:
                                            return "ALLREDUCE_MIXED_LOCAL_SPM";
    case KernelId::LLM_REDUCE_SUM_SCATTER_RESIDUAL_V2: return "REDUCE_SCATTER_V2";
    case KernelId::LLM_REDUCE_SUM_SCATTER_RESIDUAL:    return "REDUCE_SCATTER";
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

// HW perf trace dump-path builder:
//   <dir>/rpu_hwperf_pid<P>_tid<TID>_seq<S>_<GRAPH>_<build|replay>_seg<N>.json
// `_seg<N>` suffix is v5-specific: each graph emits one Chrome-trace JSON per
// segment (= one Queue_t.enqueu_batch call), matching SDK's "one perf buffer
// per build_batch" granularity. Returns empty string on snprintf truncation.
std::string build_hw_perf_dump_path(const std::string& output_dir,
                                     const std::string& graph_name,
                                     bool is_replay,
                                     size_t seg_idx) {
    const uint64_t seq = rpu_hw_perf_next_dump_seq();
    const size_t tid_hash =
        std::hash<std::thread::id>{}(std::this_thread::get_id());
    char buf[1024];
    const int n = std::snprintf(buf, sizeof(buf),
        "%s/rpu_hwperf_pid%d_tid%zx_seq%llu_%s_%s_seg%zu.json",
        output_dir.c_str(),
        static_cast<int>(::getpid()),
        tid_hash,
        static_cast<unsigned long long>(seq),
        graph_name.c_str(),
        is_replay ? "replay" : "build",
        seg_idx);
    if (n < 0 || n >= static_cast<int>(sizeof(buf))) {
        if (log_at(2)) {
            std::cerr << "[HW_PERF] dump path truncated (snprintf rc=" << n
                      << ", output_dir len=" << output_dir.size()
                      << ", graph=" << graph_name << ") — skipping dump\n";
        }
        return std::string();
    }
    return buf;
}

}  // anonymous namespace

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
    build_segments_from_nodes();
    // Capture stats：Kernel / data node 计数 + segment 数 + HostCallback
    // tier 分桶 + Tier2 stable output 字节数 + non_replayable_reason mirror.
    // Hoisted above the INFO log line so the summary can
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
    // [GRAPH-INSTR] BUILD summary: consolidated one-line
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
    // Each cache entry owns one Queue_t. Multi-segment graphs build batches
    // serially; the prepared-segment index identifies the current queue owner.
    // Buffer lifetime is bounded by the GraphCache entry limit and RAII.
    // HW perf trace BUILD-time snapshot. Single read of the global config
    // before the loop so a mid-build toggle flip can't half-enable some
    // segments.
    const HwPerfConfig hw_perf_cfg = rpu_hw_perf_snapshot();
    hw_perf_enabled_at_build_ = hw_perf_cfg.enabled;

    size_t next_node = 0;
    size_t seg_idx_for_instr = 0;
    size_t fresh_alloc_count = 0;
    for (auto& seg : segments_) {
        for (size_t i = next_node; i < seg.start_idx; ++i) {
            execute_data_node(nodes_[i]);
        }
        const bool was_present =
            static_cast<bool>(private_queue_) &&
            private_queue_core_num_ == static_cast<uint8_t>(seg.core_ids.size());
        ::rhino_lkn::Queue_t* wq = ensure_private_queue(seg.core_ids.size());
        const bool fresh = !was_present;
        if (fresh) ++fresh_alloc_count;
        // [GRAPH-INSTR] per-segment summary (FRESH = first allocation of this
        // entry's private queue for this core_num; cached = reuse). Gated
        // at DEBUG(4) because per-segment lines dominate stderr noise at the
        // default INFO level and prefixed with the op_id label.
        if (log_at(4)) {
            std::cerr << "[GRAPH-INSTR] BUILD " << gid << " build#" << build_seq << " "
                      << kernel_summary_string(nodes_, seg, seg_idx_for_instr)
                      << " => private_queue " << (fresh ? "FRESH" : "cached") << "\n";
        }

        // invalidate fingerprint 前先 reset，避免 build_batch 抛错留下半状态。
        // HW perf trace `set_enable_hw_perf` is asserted inside
        // prepare_segment_queue (shared with REPLAY-fallback-rebuild path).
        private_queue_built_segment_idx_ = -1;
        prepare_segment_queue(seg, *wq);
        // 首次 launch 紧跟 build_batch,kd_buf 里的 regs 是刚 build 时的当前值,
        // 不需要 sync_mutable_params。enqueu_batch(wait_finish=true) SYNC 返回
        // 后,本 entry 的 private_queue 处于 idle 态,可被下一 segment 复用
        // (build_batch 会清掉之前的 kd_buf state)。
        const uint32_t rc_build = wq->enqueu_batch(/*wait_finish=*/true);
        TORCH_CHECK(rc_build == 0,
                    "execute_graph_for_recording: enqueu_batch SDK rc=", rc_build,
                    " on segment idx=", seg_idx_for_instr);
        ++last_stats_.hw_batch_submit_total;
        // 标记本段为 private_queue_ 当前 build 的内容，供下次 REPLAY
        // 比对 (单段 graph 一定命中;多段 graph 只有 last seg 命中)。
        private_queue_built_segment_idx_ = static_cast<ssize_t>(seg_idx_for_instr);

        // HW perf trace — dump BUILD-time Chrome JSON per segment, gated by
        // the process-global budget (max_dumps decremented atomically).
        if (hw_perf_cfg.enabled && rpu_hw_perf_try_consume_dump()) {
            const std::string path = build_hw_perf_dump_path(
                hw_perf_cfg.output_dir, gid, /*is_replay=*/false,
                seg_idx_for_instr);
            if (!path.empty()) {
                const uint32_t drc = wq->dump_hw_perf_chrome(path.c_str());
                if (drc != 0 && log_at(2)) {
                    std::cerr << "[HW_PERF] dump_hw_perf_chrome rc=" << drc
                              << " errno=" << errno
                              << " (" << std::strerror(errno)
                              << ") path=" << path << "\n";
                }
            }
        }

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
// 每段走 launch_segment_for_replay -> QueueCache.get + prepare_segment_queue
// (build_batch) + enqueu_batch；段间 data nodes 重复执行 (capture_data 已在
// REPLAYING 期覆写过各 data node 的 dst/src)。

void RpuKernelGraph::execute_graph_for_replaying() {
    verify_semantic_spm_producer_yield_replay_arm(
        "execute_graph_for_replaying");
    RECORD_FUNCTION("rpu_graph::execute_replay", {});
    // Per-graph REPLAY visibility.
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
    // REPLAYING 期 nodes_ + segments_ 不变，Tier 分桶 + Tier2 字节
    // 数 + Tier3Oneshot 数都是 BUILT 期的稳定值。execute_graph_for_recording /
    // execute_graph_oneshot 已经把它们写进 last_stats_,这里只需 mirror 真正
    // 动态的字段 (segment_count、non_replayable_reason、per-segment replay
    // count)。重复扫描没有必要,同时保留循环会让
    // "REPLAYING 期能否触碰 nodes_ tier 字段"语义不清。
    {
        RECORD_FUNCTION("rpu_graph::replay_stats_refresh", {});
        last_stats_.segment_count = segments_.size();
        last_stats_.non_replayable_reason = non_replayable_reason_;
        // last_stats_.host_callback_tier{1,2,3}_count / host_callback_output_bytes
        // / tier3_oneshot_count 已在 RECORDING 期 (execute_graph_for_recording)
        // 写好,REPLAYING 不变。
    }
    // HW perf trace REPLAY-time snapshot. Gate dump on `enabled_at_build_`
    // (BUILD set the flag on Queue_t so perf buffer exists) AND current
    // `enabled` (user can disable mid-session to stop further files), AND
    // the budget. Mirrors BATCH_CTX-era `GraphCache::end` REPLAY block.
    const HwPerfConfig hw_perf_cfg_replay = rpu_hw_perf_snapshot();
    const bool hw_perf_dump_eligible =
        hw_perf_cfg_replay.enabled && hw_perf_enabled_at_build_;
    const std::string hw_perf_gid =
        built_signature_.op_id_str.empty() ? std::string("rpu_graph")
                                            : built_signature_.op_id_str;
    {
        RECORD_FUNCTION("rpu_graph::replay_segment_loop", {});
        size_t next_node = 0;
        size_t seg_idx_for_dump = 0;
        for (auto& seg : segments_) {
            for (size_t i = next_node; i < seg.start_idx; ++i) {
                execute_data_node(nodes_[i]);
            }
            launch_segment_for_replay(seg);
            ++seg.replay_count;

            // HW perf trace — dump REPLAY-time Chrome JSON per segment.
            // private_queue_ is the Queue_t that just ran enqueu_batch
            // (either sync-only fast path or fallback rebuild). Sticky
            // `set_enable_hw_perf(true)` from BUILD keeps the perf buffer
            // populated on every enqueu_batch.
            if (hw_perf_dump_eligible && private_queue_ &&
                rpu_hw_perf_try_consume_dump()) {
                const std::string path = build_hw_perf_dump_path(
                    hw_perf_cfg_replay.output_dir, hw_perf_gid,
                    /*is_replay=*/true, seg_idx_for_dump);
                if (!path.empty()) {
                    const uint32_t drc =
                        private_queue_->dump_hw_perf_chrome(path.c_str());
                    if (drc != 0 && log_at(2)) {
                        std::cerr << "[HW_PERF] dump_hw_perf_chrome rc="
                                  << drc << " errno=" << errno
                                  << " (" << std::strerror(errno)
                                  << ") path=" << path << "\n";
                    }
                }
            }

            ++seg_idx_for_dump;
            next_node = seg.end_idx;
        }
        for (size_t i = next_node; i < nodes_.size(); ++i) {
            execute_data_node(nodes_[i]);
        }
    }
    mirror_per_segment_replay_count(last_stats_, segments_);
    op_stream_fully_skipped_ = false;   // consume: next replay re-decides
}

// =============================================================================
// execute_graph_oneshot — non-replayable 降级路径：按 segment 分组一次性提交
// kernel + 段间交错执行 data nodes，每段从 shared QueueCache 取 Queue_t
//（与 immediate DMA 共用 one-shot pool；cacheable recording 使用 entry
// private queue），执行完丢弃 segments_。
// =============================================================================

void RpuKernelGraph::execute_graph_oneshot() {
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
    for (auto& seg : segments_) {
        for (size_t i = next_node; i < seg.start_idx; ++i) {
            execute_data_node(nodes_[i]);
        }
        ::rhino_lkn::Queue_t* wq = QueueCache::instance().get(
            seg.core_ids.size());
        // 复用 prepare_segment_queue,自动处理 Kernel / Dma / Barrier 三种段内节点。
        // oneshot 不需要 sync_mutable_params 复用 kd_buf,prepare 内部用
        // add_kernel_mutable 没有副作用(kd_buf 一次性执行后丢弃)。
        // prepare_segment_queue 接收 Segment& 以填充 seg.mutable_dmas；
        // oneshot 路径填的 slot 在下面 segments_.clear() 时就丢了,无副作用。
        prepare_segment_queue(seg, *wq);
        const uint32_t rc_oneshot = wq->enqueu_batch(/*wait_finish=*/true);
        TORCH_CHECK(rc_oneshot == 0,
                    "execute_graph_oneshot: enqueu_batch SDK rc=", rc_oneshot,
                    " on segment idx=", seg_idx_oneshot);
        ++last_stats_.hw_batch_submit_total;

        if (hw_perf_cfg_oneshot.enabled && rpu_hw_perf_try_consume_dump()) {
            const std::string path = build_hw_perf_dump_path(
                hw_perf_cfg_oneshot.output_dir, hw_perf_gid_oneshot,
                /*is_replay=*/false, seg_idx_oneshot);
            if (!path.empty()) {
                const uint32_t drc = wq->dump_hw_perf_chrome(path.c_str());
                if (drc != 0 && log_at(2)) {
                    std::cerr << "[HW_PERF] dump_hw_perf_chrome rc=" << drc
                              << " errno=" << errno
                              << " (" << std::strerror(errno)
                              << ") path=" << path << "\n";
                }
            }
        }
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
