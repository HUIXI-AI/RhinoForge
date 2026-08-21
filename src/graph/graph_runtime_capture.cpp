// graph_runtime_capture.cpp — capture/replay argument reconstruction.
//
// This file records data-node payloads during RECORDING and rebuilds the live
// stack/argument state during REPLAYING for memcpy/memset, HostCallback,
// Tier3 oneshot, and child graph skip nodes.
#include "graph/graph_runtime.h"
#include "graph/graph_runtime_internal.h"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

static std::string format_hex(uint64_t v) {
    std::ostringstream o;
    o << "0x" << std::hex << v;
    return o.str();
}

template <typename ND>
bool RpuKernelGraph::capture_replay_child_skip_data(ND data) {
    if (state_ != State::REPLAYING) {
        return false;
    }
    if (!replay_child_skip_.active &&
        (cursor_ >= nodes_.size() ||
         nodes_[cursor_].kind != GraphNodeKind::ChildGraph)) {
        return false;
    }

    auto& cgd = enter_replay_child_skip("capture_replay_child_skip_data");
    auto& child = *cgd.child;
    TORCH_CHECK(replay_child_skip_.child_cursor < child.nodes_.size(),
                "capture_replay_child_skip_data: child cursor overflow ",
                replay_child_skip_.child_cursor, " >= ",
                child.nodes_.size());
    auto& node = child.nodes_[replay_child_skip_.child_cursor];
    TORCH_CHECK(node.kind == NodeKindOf<ND>::value,
                "capture_replay_child_skip_data: child node kind mismatch at "
                "cursor ", replay_child_skip_.child_cursor,
                " expected=", static_cast<int>(NodeKindOf<ND>::value),
                " got=", static_cast<int>(node.kind));
    auto& rec = std::get<ND>(node.data);
    if constexpr (std::is_same_v<ND, MemcpyNodeData>) {
        TORCH_CHECK(rec.bytes == data.bytes,
                    "capture_replay_child_skip_data: Memcpy bytes mismatch at "
                    "child cursor ", replay_child_skip_.child_cursor,
                    " expected=", rec.bytes, " got=", data.bytes);
        TORCH_CHECK(rec.kind == data.kind,
                    "capture_replay_child_skip_data: CopyKind drift at child "
                    "cursor ", replay_child_skip_.child_cursor,
                    " expected=", static_cast<int>(rec.kind),
                    " got=", static_cast<int>(data.kind));
    } else if constexpr (std::is_same_v<ND, MemsetNodeData>) {
        TORCH_CHECK(rec.bytes == data.bytes,
                    "capture_replay_child_skip_data: Memset bytes mismatch at "
                    "child cursor ", replay_child_skip_.child_cursor,
                    " expected=", rec.bytes, " got=", data.bytes);
    } else if constexpr (std::is_same_v<ND, BranchNodeData>) {
        TORCH_CHECK(rec.branch_key == data.branch_key,
                    "capture_replay_child_skip_data: branch-key mismatch at "
                    "child cursor ", replay_child_skip_.child_cursor,
                    " expected=", format_hex(rec.branch_key),
                    " got=", format_hex(data.branch_key));
    }
    rec = std::move(data);
    ++replay_child_skip_.child_cursor;
    finish_replay_child_skip_if_complete();
    return true;
}

template <typename ND>
void RpuKernelGraph::capture_data(ND data) {
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "typed DDR-register census admits only Graph-visible "
                "Kernel/Dma/Barrier nodes; rejected node kind ",
                static_cast<int>(NodeKindOf<ND>::value));
    if constexpr (std::is_same_v<ND, MemcpyNodeData> ||
                  std::is_same_v<ND, MemsetNodeData>) {
        TORCH_CHECK(!canonical_dma_build_trace_active_ &&
                        !canonical_dma_callback_active_,
                    "RpuKernelGraph: Memcpy/Memset data node is forbidden "
                    "inside a canonical FMB DMA BUILD trace/callback");
    }
    switch (state_) {
    case State::RECORDING: {
        const int node_id = static_cast<int>(nodes_.size());
        // Memcpy 节点的 dep 推断:src 是 input(扫上游 writer),dst 是
        // output(注册给后续节点 lookup)。在 push 之前读 data 字段(push 后
        // data 已 move)。
        uint64_t mem_src_dev = 0;
        uint64_t mem_dst_dev = 0;
        if constexpr (std::is_same_v<ND, MemcpyNodeData>) {
            mem_src_dev = data.src_dev_addr;
            mem_dst_dev = data.dst_dev_addr;
        }
        nodes_.push_back(GraphNode{NodeKindOf<ND>::value, std::move(data)});
        if constexpr (std::is_same_v<ND, MemcpyNodeData>) {
            // push 后再 resolve / register,避免自循环命中 + 顺序一致 (consumer
            // 总是 post-push)。
            std::vector<uint64_t> input_devs;
            if (mem_src_dev != 0) input_devs.push_back(mem_src_dev);
            resolve_recording_input_deps(node_id, input_devs);
            track_recording_writer_dev_addr(mem_dst_dev, node_id);
        }
        break;
    }

    case State::REPLAYING: {
        if (capture_replay_child_skip_data(data)) {
            break;
        }
        TORCH_CHECK(cursor_ < nodes_.size(),
                    "RpuKernelGraph: capture_data cursor overflow: ",
                    cursor_, " >= ", nodes_.size());
        TORCH_CHECK(nodes_[cursor_].kind == NodeKindOf<ND>::value,
                    "RpuKernelGraph: data-node kind mismatch at cursor ",
                    cursor_);
        auto& rec = std::get<ND>(nodes_[cursor_].data);
        if constexpr (std::is_same_v<ND, MemcpyNodeData>) {
            TORCH_CHECK(rec.bytes == data.bytes,
                        "RpuKernelGraph: data-node bytes mismatch at cursor ",
                        cursor_, " expected=", rec.bytes, " got=", data.bytes);
            // CopyKind 也是 replay signature 的一部分。
            // 若 RECORDING 把某 node 分类为 DDR_TO_DDR 而 REPLAYING 下同一
            // 调用点因 dst/src 分配漂移查不到 dev_addr,kind 就会降回
            // HOST_MEMCPY。这意味着 graph 结构已变,应当 recapture 而不是
            // 静默混用 DMA / host memcpy。
            TORCH_CHECK(rec.kind == data.kind,
                        "RpuKernelGraph: CopyKind drift at cursor ",
                        cursor_,
                        " expected=", static_cast<int>(rec.kind),
                        " got=", static_cast<int>(data.kind));
        } else if constexpr (std::is_same_v<ND, MemsetNodeData>) {
            TORCH_CHECK(rec.bytes == data.bytes,
                        "RpuKernelGraph: data-node bytes mismatch at cursor ",
                        cursor_, " expected=", rec.bytes, " got=", data.bytes);
        } else if constexpr (std::is_same_v<ND, BranchNodeData>) {
            TORCH_CHECK(rec.branch_key == data.branch_key,
                        "RpuKernelGraph: branch-key mismatch at cursor ",
                        cursor_, " expected=", format_hex(rec.branch_key),
                        " got=", format_hex(data.branch_key));
        }
        rec = data;  // 覆写 dst/src/dev_addr(memset: value),允许每轮 replay 漂移
        cursor_++;
        break;
    }

    case State::PASSTHROUGH:
    case State::BUILT:
        TORCH_CHECK(false,
                    "RpuKernelGraph: capture_data() called in invalid state ",
                    static_cast<int>(state_));
    }
}

// 显式实例化。ChildGraphNodeData 承载 ChildGraph 节点，BranchNodeData 承载
// branch marker，HostCallbackNodeData 承载 HostCallback 节点。
template void RpuKernelGraph::capture_data<MemcpyNodeData>(MemcpyNodeData);
template void RpuKernelGraph::capture_data<MemsetNodeData>(MemsetNodeData);
template void RpuKernelGraph::capture_data<ChildGraphNodeData>(ChildGraphNodeData);
template void RpuKernelGraph::capture_data<BranchNodeData>(BranchNodeData);
template void RpuKernelGraph::capture_data<HostCallbackNodeData>(HostCallbackNodeData);
template void RpuKernelGraph::capture_data<Tier3OneshotNodeData>(Tier3OneshotNodeData);

// =============================================================================
// record_dma_* / record_barrier — BATCH_CTX 顶替捕获入口
// =============================================================================
//
// 跟 capture_data 平行的另一类入口:
//   - capture_data 是 host-side data node(Memcpy / Memset 等),
//     execute_data_node 在段边界执行。
//   - record_dma_* / record_barrier 是 Queue_t 批次内的 DMA/Barrier 节点,
//     必须跟 Kernel 节点一起留在同一 Segment 内,由 prepare_segment_queue 顺序
//     调 wq.add_dma_kernel / add_barrier 进 batch。
//
// 实现哲学跟 capture_data 一致:
//   RECORDING : push 进 nodes_(不计入 dep 跟踪,DMA 不属于 Memcpy 路径)
//   REPLAYING : 校验当前 cursor 节点 kind + 关键字段稳定后,覆写本轮值并推 cursor
//   PASSTHROUGH / BUILT : 编程错误,直接抛 (BATCH_CTX 原本就只在 graph 内合法)

// 校验+push 逻辑直接放在成员函数内,Dma/Barrier 不参与 recording_dev_addr_writer_
// 跟踪 — DMA src/dst 是绝对 dev_addr,不是 caching allocator 派的 tensor,graph
// executor 拿不到 Tensor 对象,也无从 register 上游 writer。
void RpuKernelGraph::record_dma_fixed(uint64_t src_addr, uint64_t dst_addr,
                                       size_t bytes, uint8_t channel) {
    TORCH_CHECK(!canonical_dma_build_trace_active_ &&
                    !canonical_dma_callback_active_,
                "RpuKernelGraph: legacy fixed DMA is forbidden inside a "
                "canonical FMB member trace/callback");
    DmaNodeData data;
    data.variant  = DmaNodeData::Variant::Fixed;
    data.src_addr = src_addr;
    data.dst_addr = dst_addr;
    data.bytes    = bytes;
    data.channel  = channel;
    record_dma_node(std::move(data));
}

void RpuKernelGraph::record_dma_mutable_src(const uint64_t* live_src_base,
                                             int64_t src_offset,
                                             uint64_t dst_addr,
                                             size_t bytes,
                                             uint8_t channel) {
    TORCH_CHECK(!canonical_dma_build_trace_active_ &&
                    !canonical_dma_callback_active_,
                "RpuKernelGraph: legacy MutableSrc DMA is forbidden inside "
                "a canonical FMB member trace/callback");
    TORCH_CHECK(live_src_base != nullptr,
                "RpuKernelGraph: record_dma_mutable_src null live_src_base");
    DmaNodeData data;
    data.variant     = DmaNodeData::Variant::MutableSrc;
    data.src_addr    = 0;
    data.dst_addr    = dst_addr;
    data.bytes       = bytes;
    data.channel     = channel;
    data.live_base   = live_src_base;
    data.live_offset = src_offset;
    record_dma_node(std::move(data));
}

void RpuKernelGraph::record_dma_mutable_dst(uint64_t src_addr,
                                             const uint64_t* live_dst_base,
                                             int64_t dst_offset,
                                             size_t bytes,
                                             uint8_t channel) {
    TORCH_CHECK(!canonical_dma_build_trace_active_ &&
                    !canonical_dma_callback_active_,
                "RpuKernelGraph: legacy MutableDst DMA is forbidden inside "
                "a canonical FMB member trace/callback");
    TORCH_CHECK(live_dst_base != nullptr,
                "RpuKernelGraph: record_dma_mutable_dst null live_dst_base");
    DmaNodeData data;
    data.variant     = DmaNodeData::Variant::MutableDst;
    data.src_addr    = src_addr;
    data.dst_addr    = 0;
    data.bytes       = bytes;
    data.channel     = channel;
    data.live_base   = live_dst_base;
    data.live_offset = dst_offset;
    record_dma_node(std::move(data));
}

namespace {

const GraphSignature* semantic_dma_signature(
        const RpuKernelGraph::State state,
        const std::optional<GraphSignature>& pending,
        bool has_built,
        const GraphSignature& built) {
    if (state == RpuKernelGraph::State::RECORDING && pending.has_value()) {
        return &*pending;
    }
    if (state == RpuKernelGraph::State::REPLAYING && has_built) {
        return &built;
    }
    return nullptr;
}

}  // namespace

void RpuKernelGraph::bind_semantic_dma_owner_for_fmb(
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation) {
    TORCH_CHECK(state_ == State::RECORDING && owner_generation != nullptr &&
                    expected_owner_generation != 0 &&
                    *owner_generation == expected_owner_generation,
                "RpuKernelGraph: semantic DMA owner binding requires one "
                "live RECORDING owner generation");
    const std::shared_ptr<const uint64_t> existing =
        semantic_dma_owner_generation_.lock();
    TORCH_CHECK(existing == nullptr ||
                    (existing.get() == owner_generation.get() &&
                     semantic_dma_expected_owner_generation_ ==
                         expected_owner_generation),
                "RpuKernelGraph: semantic DMA owner binding drift within one "
                "BUILD");
    semantic_dma_owner_generation_ = owner_generation;
    semantic_dma_expected_owner_generation_ = expected_owner_generation;
}

void RpuKernelGraph::arm_semantic_dma_owner_replay_for_fmb(
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation) {
    const std::shared_ptr<const uint64_t> build_owner =
        semantic_dma_owner_generation_.lock();
    TORCH_CHECK(
        state_ == State::REPLAYING && scope_open_ &&
            !active_stack_.empty() && active_stack_.back() == this &&
            has_semantic_dma_nodes_ && owner_generation != nullptr &&
            build_owner != nullptr &&
            build_owner.get() == owner_generation.get() &&
            semantic_dma_expected_owner_generation_ ==
                expected_owner_generation &&
            *owner_generation == expected_owner_generation,
        "RpuKernelGraph: active semantic DMA REPLAY owner differs from the "
        "sealed BUILD owner");
    semantic_dma_replay_owner_armed_ = true;
}

void RpuKernelGraph::arm_semantic_spm_peer_replay_for_fmb(
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation) {
    TORCH_CHECK(has_semantic_spm_peer_nodes_,
                "RpuKernelGraph: typed SPM peer REPLAY arm requires a "
                "schema-v13 BUILD");
    arm_semantic_dma_owner_replay_for_fmb(
        owner_generation, expected_owner_generation);
    semantic_spm_peer_replay_armed_ = true;
}

void RpuKernelGraph::begin_canonical_dma_build_trace_for_fmb(
        uint64_t profile_hash) {
    const GraphSignature* signature = semantic_dma_signature(
        state_, pending_signature_, has_built_signature_, built_signature_);
    const std::shared_ptr<const uint64_t> owner =
        semantic_dma_owner_generation_.lock();
    TORCH_CHECK(
        state_ == State::RECORDING && scope_open_ &&
            !active_stack_.empty() && active_stack_.back() == this &&
            owner != nullptr &&
            semantic_dma_expected_owner_generation_ != 0 &&
            *owner == semantic_dma_expected_owner_generation_ &&
            signature != nullptr && signature->segment_key == profile_hash &&
            !canonical_dma_build_trace_active_ &&
            !canonical_dma_callback_active_ &&
            !pending_kernel_id_.has_value() &&
            !pending_kernel_name_.has_value() &&
            !canonical_dma_poisoned_,
        "RpuKernelGraph: canonical FMB DMA BUILD trace owner/profile/scope is "
        "stale or has a pending kernel");
    canonical_dma_build_trace_active_ = true;
    canonical_dma_build_trace_complete_ = false;
    canonical_dma_build_trace_profile_hash_ = profile_hash;
    canonical_dma_build_trace_node_begin_ = nodes_.size();
    canonical_dma_build_trace_node_end_ = nodes_.size();
}

void RpuKernelGraph::complete_canonical_dma_build_trace_for_fmb() {
    TORCH_CHECK(
        state_ == State::RECORDING &&
            canonical_dma_build_trace_active_ &&
            !canonical_dma_build_trace_complete_ &&
            !canonical_dma_callback_active_ &&
            !canonical_dma_burst_permit_.active &&
            !canonical_dma_poisoned_,
        "RpuKernelGraph: canonical FMB DMA BUILD trace cannot complete from "
        "its current state");
    canonical_dma_build_trace_complete_ = true;
    canonical_dma_build_trace_node_end_ = nodes_.size();
}

void RpuKernelGraph::begin_canonical_dma_callback_for_fmb(
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation,
        uint64_t profile_hash,
        size_t layer_group_ordinal,
        size_t member_count) {
    const GraphSignature* signature = semantic_dma_signature(
        state_, pending_signature_, has_built_signature_, built_signature_);
    const std::shared_ptr<const uint64_t> bound_owner =
        semantic_dma_owner_generation_.lock();
    TORCH_CHECK(
        scope_open_ && !active_stack_.empty() &&
            active_stack_.back() == this &&
            (state_ == State::RECORDING || state_ == State::REPLAYING) &&
            !canonical_dma_poisoned_ &&
            !canonical_dma_callback_active_ &&
            !canonical_dma_burst_permit_.active &&
            owner_generation != nullptr && bound_owner != nullptr &&
            bound_owner.get() == owner_generation.get() &&
            semantic_dma_expected_owner_generation_ ==
                expected_owner_generation &&
            *owner_generation == expected_owner_generation &&
            signature != nullptr && signature->segment_key == profile_hash &&
            (state_ != State::RECORDING ||
             (canonical_dma_build_trace_active_ &&
              !canonical_dma_build_trace_complete_ &&
              canonical_dma_build_trace_profile_hash_ == profile_hash)) &&
            (state_ != State::REPLAYING ||
             (semantic_dma_replay_owner_armed_ &&
              (!has_semantic_spm_peer_nodes_ ||
               semantic_spm_peer_replay_armed_))),
        "RpuKernelGraph: canonical FMB DMA callback owner/profile/Graph "
        "scope is stale or unarmed");
    canonical_dma_callback_active_ = true;
    canonical_dma_callback_profile_hash_ = profile_hash;
    canonical_dma_callback_owner_generation_ = owner_generation;
    canonical_dma_callback_expected_owner_generation_ =
        expected_owner_generation;
    canonical_dma_callback_layer_group_ordinal_ = layer_group_ordinal;
    canonical_dma_callback_member_count_ = member_count;
    canonical_dma_callback_next_member_ = 0;
    canonical_dma_callback_next_role_ =
        static_cast<uint8_t>(1);  // SpmFmbDmaEndpointRole::Ingress
}

void RpuKernelGraph::end_canonical_dma_callback_for_fmb() {
    TORCH_CHECK(canonical_dma_callback_active_ &&
                    !canonical_dma_burst_permit_.active &&
                    canonical_dma_callback_next_member_ ==
                        canonical_dma_callback_member_count_ &&
                    canonical_dma_callback_next_role_ == 1,
                "RpuKernelGraph: canonical FMB DMA callback ended without "
                "the exact ordered member/role cover");
    canonical_dma_callback_active_ = false;
    canonical_dma_callback_profile_hash_ = 0;
    canonical_dma_callback_owner_generation_.reset();
    canonical_dma_callback_expected_owner_generation_ = 0;
    canonical_dma_callback_layer_group_ordinal_ = 0;
    canonical_dma_callback_member_count_ = 0;
    canonical_dma_callback_next_member_ = 0;
    canonical_dma_callback_next_role_ = 1;
}

void RpuKernelGraph::cancel_canonical_dma_callback_for_fmb() noexcept {
    if (canonical_dma_callback_active_ ||
        canonical_dma_burst_permit_.active) {
        canonical_dma_poisoned_ = true;
    }
    canonical_dma_burst_permit_ = {};
    canonical_dma_callback_active_ = false;
    canonical_dma_callback_profile_hash_ = 0;
    canonical_dma_callback_owner_generation_.reset();
    canonical_dma_callback_expected_owner_generation_ = 0;
    canonical_dma_callback_layer_group_ordinal_ = 0;
    canonical_dma_callback_member_count_ = 0;
    canonical_dma_callback_next_member_ = 0;
    canonical_dma_callback_next_role_ = 1;
}

void RpuKernelGraph::begin_canonical_dma_burst_for_fmb(
        const GraphDmaSemanticEndpoint& endpoint,
        uint64_t spm_peer_id) {
    const std::shared_ptr<const uint64_t> endpoint_owner =
        endpoint.owner_generation_.lock();
    const std::shared_ptr<const uint64_t> callback_owner =
        canonical_dma_callback_owner_generation_.lock();
    TORCH_CHECK(
        canonical_dma_callback_active_ &&
            !canonical_dma_burst_permit_.active &&
            endpoint.requires_canonical_burst_ &&
            endpoint.semantic_id_ != 0 && endpoint_owner != nullptr &&
            callback_owner != nullptr &&
            endpoint_owner.get() == callback_owner.get() &&
            endpoint.expected_owner_generation_ ==
                canonical_dma_callback_expected_owner_generation_ &&
            endpoint.profile_hash_ ==
                canonical_dma_callback_profile_hash_ &&
            endpoint.layer_group_ordinal_ ==
                canonical_dma_callback_layer_group_ordinal_ &&
            canonical_dma_callback_next_member_ <
                canonical_dma_callback_member_count_ &&
            endpoint.member_ordinal_ ==
                canonical_dma_callback_next_member_ &&
            endpoint.role_ == canonical_dma_callback_next_role_ &&
            endpoint.byte_count_ > 0 &&
            endpoint.expected_occurrence_count_ >= 1 &&
            endpoint.expected_occurrence_count_ <= 8,
        "RpuKernelGraph: canonical FMB DMA burst endpoint is stale, foreign, "
        "or outside its current callback");
    canonical_dma_burst_permit_.active = true;
    canonical_dma_burst_permit_.endpoint_id = endpoint.semantic_id_;
    canonical_dma_burst_permit_.spm_peer_id = spm_peer_id;
    canonical_dma_burst_permit_.variant =
        static_cast<DmaNodeData::Variant>(endpoint.expected_dma_variant_);
    canonical_dma_burst_permit_.bytes = endpoint.byte_count_;
    canonical_dma_burst_permit_.occurrence_count =
        endpoint.expected_occurrence_count_;
    canonical_dma_burst_permit_.next_channel = 0;
}

void RpuKernelGraph::end_canonical_dma_burst_for_fmb() {
    TORCH_CHECK(
        canonical_dma_callback_active_ &&
            canonical_dma_burst_permit_.active &&
            canonical_dma_burst_permit_.next_channel ==
                canonical_dma_burst_permit_.occurrence_count,
        "RpuKernelGraph: canonical FMB DMA burst did not consume its exact "
        "declared channel cover");
    if (canonical_dma_callback_next_role_ == 1) {
        canonical_dma_callback_next_role_ = 2;
    } else {
        TORCH_INTERNAL_ASSERT(canonical_dma_callback_next_role_ == 2);
        canonical_dma_callback_next_role_ = 1;
        ++canonical_dma_callback_next_member_;
    }
    canonical_dma_burst_permit_ = {};
}

void RpuKernelGraph::cancel_canonical_dma_burst_for_fmb() noexcept {
    if (canonical_dma_burst_permit_.active) {
        canonical_dma_poisoned_ = true;
    }
    canonical_dma_burst_permit_ = {};
}

bool RpuKernelGraph::validate_canonical_dma_before_record(
        const GraphDmaSemanticEndpoint& endpoint,
        DmaNodeData::Variant variant,
        size_t bytes,
        uint8_t channel) const {
    TORCH_CHECK(
        !canonical_dma_build_trace_active_ ||
            (endpoint.requires_canonical_burst_ &&
             canonical_dma_callback_active_),
        "RpuKernelGraph: non-canonical semantic DMA is forbidden inside a "
        "canonical FMB DMA BUILD trace");
    if (!endpoint.requires_canonical_burst_ &&
        !canonical_dma_callback_active_) {
        return false;
    }
    TORCH_CHECK(
        endpoint.requires_canonical_burst_ &&
            canonical_dma_callback_active_ &&
            canonical_dma_burst_permit_.active &&
            canonical_dma_burst_permit_.endpoint_id ==
                endpoint.semantic_id_ &&
            canonical_dma_burst_permit_.variant == variant &&
            canonical_dma_burst_permit_.bytes == bytes &&
            canonical_dma_burst_permit_.next_channel == channel &&
            canonical_dma_burst_permit_.next_channel <
                canonical_dma_burst_permit_.occurrence_count,
        "RpuKernelGraph: canonical FMB DMA requires the current framework "
        "burst permit and exact next channel");
    return true;
}

void RpuKernelGraph::commit_canonical_dma_after_record(bool canonical) {
    if (!canonical) return;
    TORCH_INTERNAL_ASSERT(canonical_dma_burst_permit_.active &&
                          canonical_dma_burst_permit_.next_channel <
                              canonical_dma_burst_permit_.occurrence_count);
    ++canonical_dma_burst_permit_.next_channel;
}

void RpuKernelGraph::record_dma_fixed(
        const GraphDmaSemanticEndpoint& endpoint,
        uint64_t src_addr, uint64_t dst_addr,
        size_t bytes, uint8_t channel) {
    const bool canonical = validate_canonical_dma_before_record(
        endpoint, DmaNodeData::Variant::Fixed, bytes, channel);
    const std::shared_ptr<const uint64_t> owner =
        endpoint.owner_generation_.lock();
    const GraphSignature* signature = semantic_dma_signature(
        state_, pending_signature_, has_built_signature_, built_signature_);
    const std::shared_ptr<const uint64_t> bound_owner =
        semantic_dma_owner_generation_.lock();
    TORCH_CHECK(endpoint.semantic_id_ != 0 && owner != nullptr &&
                    *owner == endpoint.expected_owner_generation_,
                "RpuKernelGraph: semantic fixed DMA endpoint is stale");
    TORCH_CHECK(
        state_ != State::RECORDING ||
            (bound_owner != nullptr &&
             bound_owner.get() == owner.get() &&
             semantic_dma_expected_owner_generation_ ==
                 endpoint.expected_owner_generation_),
        "RpuKernelGraph: semantic fixed DMA owner differs from the armed FMB "
        "trace");
    TORCH_CHECK(
        state_ != State::REPLAYING ||
            (semantic_dma_replay_owner_armed_ && bound_owner != nullptr &&
             bound_owner.get() == owner.get()),
        "RpuKernelGraph: semantic fixed DMA requires an armed owning FMB "
        "REPLAY scope");
    TORCH_CHECK(signature != nullptr &&
                    signature->segment_key == endpoint.profile_hash_,
                "RpuKernelGraph: semantic fixed DMA profile/signature drift");
    TORCH_CHECK(endpoint.expected_dma_variant_ ==
                    static_cast<uint8_t>(DmaNodeData::Variant::Fixed) &&
                    endpoint.byte_count_ == bytes,
                "RpuKernelGraph: semantic fixed DMA descriptor drift");
    TORCH_CHECK(bytes > 0 && bytes % 16 == 0 &&
                    endpoint.expected_occurrence_count_ >= 1 &&
                    endpoint.expected_occurrence_count_ <= 8 &&
                    channel < endpoint.expected_occurrence_count_,
                "RpuKernelGraph: semantic fixed DMA launch shape/channel "
                "is outside its declared endpoint");
    DmaNodeData data;
    data.variant = DmaNodeData::Variant::Fixed;
    data.src_addr = src_addr;
    data.dst_addr = dst_addr;
    data.bytes = bytes;
    data.channel = channel;
    data.semantic_endpoint_id = endpoint.semantic_id_;
    data.semantic_expected_occurrence_count =
        endpoint.expected_occurrence_count_;
    data.semantic_owner_generation = endpoint.owner_generation_;
    data.semantic_expected_owner_generation =
        endpoint.expected_owner_generation_;
    data.semantic_canonical_member_emission = canonical;
    data.semantic_spm_peer_id = canonical
        ? canonical_dma_burst_permit_.spm_peer_id : 0;
    record_dma_node(std::move(data));
    commit_canonical_dma_after_record(canonical);
}

void RpuKernelGraph::record_dma_mutable_src(
        const GraphDmaSemanticEndpoint& endpoint,
        const uint64_t* live_src_base, int64_t src_offset,
        uint64_t dst_addr, size_t bytes, uint8_t channel) {
    TORCH_CHECK(live_src_base != nullptr,
                "RpuKernelGraph: semantic MutableSrc null live_base");
    const bool canonical = validate_canonical_dma_before_record(
        endpoint, DmaNodeData::Variant::MutableSrc, bytes, channel);
    const std::shared_ptr<const uint64_t> owner =
        endpoint.owner_generation_.lock();
    const GraphSignature* signature = semantic_dma_signature(
        state_, pending_signature_, has_built_signature_, built_signature_);
    const std::shared_ptr<const uint64_t> bound_owner =
        semantic_dma_owner_generation_.lock();
    TORCH_CHECK(endpoint.semantic_id_ != 0 && owner != nullptr &&
                    *owner == endpoint.expected_owner_generation_,
                "RpuKernelGraph: semantic MutableSrc endpoint is stale");
    TORCH_CHECK(
        state_ != State::RECORDING ||
            (bound_owner != nullptr &&
             bound_owner.get() == owner.get() &&
             semantic_dma_expected_owner_generation_ ==
                 endpoint.expected_owner_generation_),
        "RpuKernelGraph: semantic MutableSrc owner differs from the armed "
        "FMB trace");
    TORCH_CHECK(
        state_ != State::REPLAYING ||
            (semantic_dma_replay_owner_armed_ && bound_owner != nullptr &&
             bound_owner.get() == owner.get()),
        "RpuKernelGraph: semantic MutableSrc requires an armed owning FMB "
        "REPLAY scope");
    TORCH_CHECK(signature != nullptr &&
                    signature->segment_key == endpoint.profile_hash_,
                "RpuKernelGraph: semantic MutableSrc profile/signature drift");
    TORCH_CHECK(endpoint.expected_dma_variant_ ==
                    static_cast<uint8_t>(
                        DmaNodeData::Variant::MutableSrc) &&
                    endpoint.byte_count_ == bytes && src_offset >= 0 &&
                    static_cast<uint64_t>(src_offset) ==
                        endpoint.byte_begin_,
                "RpuKernelGraph: semantic MutableSrc descriptor drift");
    TORCH_CHECK(bytes > 0 && bytes % 16 == 0 &&
                    endpoint.expected_occurrence_count_ >= 1 &&
                    endpoint.expected_occurrence_count_ <= 8 &&
                    channel < endpoint.expected_occurrence_count_,
                "RpuKernelGraph: semantic MutableSrc launch shape/channel "
                "is outside its declared endpoint");
    DmaNodeData data;
    data.variant = DmaNodeData::Variant::MutableSrc;
    data.dst_addr = dst_addr;
    data.bytes = bytes;
    data.channel = channel;
    data.live_base = live_src_base;
    data.live_offset = src_offset;
    data.semantic_endpoint_id = endpoint.semantic_id_;
    data.semantic_expected_occurrence_count =
        endpoint.expected_occurrence_count_;
    data.semantic_owner_generation = endpoint.owner_generation_;
    data.semantic_expected_owner_generation =
        endpoint.expected_owner_generation_;
    data.semantic_canonical_member_emission = canonical;
    data.semantic_spm_peer_id = canonical
        ? canonical_dma_burst_permit_.spm_peer_id : 0;
    record_dma_node(std::move(data));
    commit_canonical_dma_after_record(canonical);
}

void RpuKernelGraph::record_dma_mutable_dst(
        const GraphDmaSemanticEndpoint& endpoint,
        uint64_t src_addr, const uint64_t* live_dst_base,
        int64_t dst_offset, size_t bytes, uint8_t channel) {
    TORCH_CHECK(live_dst_base != nullptr,
                "RpuKernelGraph: semantic MutableDst null live_base");
    const bool canonical = validate_canonical_dma_before_record(
        endpoint, DmaNodeData::Variant::MutableDst, bytes, channel);
    const std::shared_ptr<const uint64_t> owner =
        endpoint.owner_generation_.lock();
    const GraphSignature* signature = semantic_dma_signature(
        state_, pending_signature_, has_built_signature_, built_signature_);
    const std::shared_ptr<const uint64_t> bound_owner =
        semantic_dma_owner_generation_.lock();
    TORCH_CHECK(endpoint.semantic_id_ != 0 && owner != nullptr &&
                    *owner == endpoint.expected_owner_generation_,
                "RpuKernelGraph: semantic MutableDst endpoint is stale");
    TORCH_CHECK(
        state_ != State::RECORDING ||
            (bound_owner != nullptr &&
             bound_owner.get() == owner.get() &&
             semantic_dma_expected_owner_generation_ ==
                 endpoint.expected_owner_generation_),
        "RpuKernelGraph: semantic MutableDst owner differs from the armed "
        "FMB trace");
    TORCH_CHECK(
        state_ != State::REPLAYING ||
            (semantic_dma_replay_owner_armed_ && bound_owner != nullptr &&
             bound_owner.get() == owner.get()),
        "RpuKernelGraph: semantic MutableDst requires an armed owning FMB "
        "REPLAY scope");
    TORCH_CHECK(signature != nullptr &&
                    signature->segment_key == endpoint.profile_hash_,
                "RpuKernelGraph: semantic MutableDst profile/signature drift");
    TORCH_CHECK(endpoint.expected_dma_variant_ ==
                    static_cast<uint8_t>(
                        DmaNodeData::Variant::MutableDst) &&
                    endpoint.byte_count_ == bytes && dst_offset >= 0 &&
                    static_cast<uint64_t>(dst_offset) ==
                        endpoint.byte_begin_,
                "RpuKernelGraph: semantic MutableDst descriptor drift");
    TORCH_CHECK(bytes > 0 && bytes % 16 == 0 &&
                    endpoint.expected_occurrence_count_ >= 1 &&
                    endpoint.expected_occurrence_count_ <= 8 &&
                    channel < endpoint.expected_occurrence_count_,
                "RpuKernelGraph: semantic MutableDst launch shape/channel "
                "is outside its declared endpoint");
    DmaNodeData data;
    data.variant = DmaNodeData::Variant::MutableDst;
    data.src_addr = src_addr;
    data.bytes = bytes;
    data.channel = channel;
    data.live_base = live_dst_base;
    data.live_offset = dst_offset;
    data.semantic_endpoint_id = endpoint.semantic_id_;
    data.semantic_expected_occurrence_count =
        endpoint.expected_occurrence_count_;
    data.semantic_owner_generation = endpoint.owner_generation_;
    data.semantic_expected_owner_generation =
        endpoint.expected_owner_generation_;
    data.semantic_canonical_member_emission = canonical;
    data.semantic_spm_peer_id = canonical
        ? canonical_dma_burst_permit_.spm_peer_id : 0;
    record_dma_node(std::move(data));
    commit_canonical_dma_after_record(canonical);
}

void RpuKernelGraph::record_dma_node(DmaNodeData data) {
    check_kernel_register_node_emission("Graph-visible DMA emission");
    consume_kernel_register_dma_burst(data);
    switch (state_) {
    case State::RECORDING:
        if (data.semantic_endpoint_id != 0) {
            has_semantic_dma_nodes_ = true;
        }
        if (data.semantic_spm_peer_id != 0) {
            TORCH_CHECK(data.semantic_endpoint_id != 0 &&
                            data.semantic_canonical_member_emission,
                        "RpuKernelGraph: typed SPM peer requires one "
                        "canonical semantic DMA endpoint");
            has_semantic_spm_peer_nodes_ = true;
        }
        nodes_.push_back(GraphNode{GraphNodeKind::Dma, std::move(data)});
        observe_kernel_register_node(GraphNodeKind::Dma, nullptr);
        break;
    case State::REPLAYING: {
        TORCH_CHECK(cursor_ < nodes_.size(),
                    "RpuKernelGraph: record_dma_fixed cursor overflow ", cursor_);
        TORCH_CHECK(nodes_[cursor_].kind == GraphNodeKind::Dma,
                    "RpuKernelGraph: kind mismatch at cursor ", cursor_,
                    " expected Dma got ", static_cast<int>(nodes_[cursor_].kind));
        auto& rec = nodes_[cursor_].as_dma();
        TORCH_CHECK(rec.variant == data.variant,
                    "RpuKernelGraph: DMA variant drift at cursor ", cursor_);
        TORCH_CHECK(rec.bytes == data.bytes,
                    "RpuKernelGraph: DMA bytes drift at cursor ", cursor_,
                    " expected=", rec.bytes, " got=", data.bytes);
        TORCH_CHECK(rec.channel == data.channel,
                    "RpuKernelGraph: DMA channel drift at cursor ", cursor_,
                    " expected=", (int)rec.channel,
                    " got=", (int)data.channel);
        TORCH_CHECK(rec.semantic_endpoint_id == data.semantic_endpoint_id,
                    "RpuKernelGraph: DMA semantic endpoint ID drift at cursor ",
                    cursor_);
        TORCH_CHECK(
            rec.semantic_spm_peer_id == data.semantic_spm_peer_id,
            "RpuKernelGraph: DMA semantic SPM peer ID drift at cursor ",
            cursor_);
        if (kernel_register_census_active_) {
            TORCH_CHECK(
                rec.live_offset == data.live_offset &&
                    rec.semantic_expected_occurrence_count ==
                        data.semantic_expected_occurrence_count &&
                    rec.semantic_canonical_member_emission ==
                        data.semantic_canonical_member_emission,
                "RpuKernelGraph: schema-5 DMA ordered topology drift at cursor ",
                cursor_);
            if (rec.outer_fast_owner_bound || data.outer_fast_owner_bound) {
                TORCH_CHECK(
                    rec.outer_fast_owner_bound &&
                        data.outer_fast_owner_bound &&
                        rec.outer_fast_ddr_side ==
                            data.outer_fast_ddr_side &&
                        rec.outer_fast_owner_ordinal ==
                            data.outer_fast_owner_ordinal &&
                        rec.outer_fast_owner_topology_id ==
                            data.outer_fast_owner_topology_id &&
                        rec.outer_fast_owner_byte_offset ==
                            data.outer_fast_owner_byte_offset &&
                        rec.src_addr == data.src_addr &&
                        rec.dst_addr == data.dst_addr &&
                        rec.live_base == data.live_base,
                    "RpuKernelGraph: schema-5 owner-bound DMA address/owner "
                    "drift at cursor ", cursor_);
            }
        }
        if (rec.semantic_endpoint_id != 0) {
            const std::shared_ptr<const uint64_t> recorded_owner =
                rec.semantic_owner_generation.lock();
            const std::shared_ptr<const uint64_t> current_owner =
                data.semantic_owner_generation.lock();
            TORCH_CHECK(
                recorded_owner != nullptr && current_owner != nullptr &&
                    recorded_owner.get() == current_owner.get() &&
                    rec.semantic_expected_owner_generation ==
                        data.semantic_expected_owner_generation &&
                    rec.semantic_expected_occurrence_count ==
                        data.semantic_expected_occurrence_count &&
                    rec.semantic_canonical_member_emission ==
                        data.semantic_canonical_member_emission &&
                    rec.src_addr == data.src_addr &&
                    rec.dst_addr == data.dst_addr &&
                    rec.live_base == data.live_base &&
                    rec.live_offset == data.live_offset,
                "RpuKernelGraph: semantic DMA owner, fixed endpoint, or slot "
                "drift at cursor ", cursor_);
        } else {
            // Legacy ID-0 semantics are intentionally unchanged: only shape
            // and channel are replay-validated, then live fields are replaced.
            rec.src_addr = data.src_addr;
            rec.dst_addr = data.dst_addr;
            rec.live_base = data.live_base;
            rec.live_offset = data.live_offset;
        }
        observe_kernel_register_node(GraphNodeKind::Dma, nullptr);
        cursor_++;
        break;
    }
    case State::PASSTHROUGH:
    case State::BUILT:
        TORCH_CHECK(false,
                    "RpuKernelGraph: record_dma_fixed called in invalid state ",
                    static_cast<int>(state_));
    }
}

void RpuKernelGraph::record_barrier(uint8_t self_stream, uint8_t target_stream) {
    check_kernel_register_node_emission("Graph-visible Barrier emission");
    TORCH_CHECK(!canonical_dma_callback_active_ ||
                    canonical_dma_burst_permit_.active,
                "RpuKernelGraph: canonical FMB callback barrier requires "
                "the current framework DMA burst permit");
    BarrierNodeData data;
    data.self_stream   = self_stream;
    data.target_stream = target_stream;
    switch (state_) {
    case State::RECORDING:
        nodes_.push_back(GraphNode{GraphNodeKind::Barrier, std::move(data)});
        observe_kernel_register_node(GraphNodeKind::Barrier, nullptr);
        break;
    case State::REPLAYING: {
        TORCH_CHECK(cursor_ < nodes_.size(),
                    "RpuKernelGraph: record_barrier cursor overflow ", cursor_);
        TORCH_CHECK(nodes_[cursor_].kind == GraphNodeKind::Barrier,
                    "RpuKernelGraph: kind mismatch at cursor ", cursor_,
                    " expected Barrier got ",
                    static_cast<int>(nodes_[cursor_].kind));
        auto& rec = nodes_[cursor_].as_barrier_node();
        TORCH_CHECK(rec.self_stream == self_stream,
                    "RpuKernelGraph: barrier self_stream drift at cursor ", cursor_,
                    " expected=", (int)rec.self_stream, " got=", (int)self_stream);
        TORCH_CHECK(rec.target_stream == target_stream,
                    "RpuKernelGraph: barrier target_stream drift at cursor ",
                    cursor_,
                    " expected=", (int)rec.target_stream,
                    " got=", (int)target_stream);
        observe_kernel_register_node(GraphNodeKind::Barrier, nullptr);
        cursor_++;
        break;
    }
    case State::PASSTHROUGH:
    case State::BUILT:
        TORCH_CHECK(false,
                    "RpuKernelGraph: record_barrier called in invalid state ",
                    static_cast<int>(state_));
    }
}

// =============================================================================
// HostCallbackCapture — A5+A6 RECORDING-period RAII helper
// =============================================================================
//
// v1 行为:
//   ctor:
//     - 录 args_template 骨架(tensor 槽 = at::Tensor()占位,scalar/list = 实值)
//     - 收集 tensor_arg_indices, output_arg_indices(由 collect_output_arg_indices)
//     - 给 input RPU tensor 登记 boundary_flush(host 跑前 RPU 写入需要可见)
//     - 所有 Tier:output_pool / output_pool_dev_addrs 留空,由 adopt_returns
//       (Tier 1+2)或 adopt_dynamic_return(Tier 3)在 CPU op 跑完后填
//     - 最后 capture_data<HostCallbackNodeData>(...) push 节点;node_idx_ 记下
//   adopt_returns (Tier 1+2,旧名 adopt_functional_return):
//     - CPU op 跑完后,stack returns 是 fresh CPU/RPU tensor
//     - 给每个 return 在 RPU device 上 at::empty() + copy_(),作为 graph-owned
//       稳定 buffer;dev_addr 跨 replay 不变
//     - 替换 stack 返回 slot 为稳定 buffer(下游 RPU op capture 时看到稳定 dev_addr)
//     - 写 output_pool / output_pool_dev_addrs / output_pool_bytes
//     - 给稳定 buffer 登记 boundary_flush(post-CPU host 写入需要 RPU 可见)
//     注:Tier 1 的 .out= 输入 arg 不再被 trust(Dynamo 总是 functional 形式,
//        PT runtime 每次 alloc fresh .out=,trust 不成立);统一走 stable buffer。
//   adopt_dynamic_return (Tier 3):
//     - 仅把 stack returns 的 tensor 加进 output_pool 防本轮 forward
//       内被 GC(scope 已 mark_non_replayable,跨轮不复用)
//
// dtor 当前为空；executor 的收尾由生命周期钩子负责。
//
// 数据流时序的 v1 简化:CPU op 在 ctor → run_cpu_fallback_zerocopy 期间 eager
// 跑;此时 prefix RPU kernel 尚未 launch(在 end() 才 launch),所以本轮 ctor
// 跑 CPU op 看到的是 prefix kernel 写出之前的 DDR 旧值。executor 在 end
// 阶段会 re-execute CPU op(那时 prefix segment 已 SYNC launch 完成),覆盖正确
// 输出。下游 RPU kernel 节点 capture 时记下的 dev_addr 仍然正确,因为 dev_addr
// 跨 ctor / executor 一致(同一个 .out= input arg 或 stable buffer)。因此在
// capture scope 内直接用 .item() 读取该值仍可能看到旧值。

namespace {

// 取一个 tensor IValue 的 RPU dev_addr;如果不是 RPU device 或 ptr 为空,返回 0。
// 用于 Tier1 的 .out= input arg 预记录 + executor 校验。
uint64_t rpu_dev_addr_of(const at::Tensor& t) {
    if (!t.defined() || t.device().type() != c10::DeviceType::PrivateUse1) {
        return 0;
    }
    return rhino_lkn::RpuGetDevAddr(t.data_ptr());
}

}  // namespace

HostCallbackCapture::HostCallbackCapture(RpuKernelGraph& g,
                                         const c10::OperatorHandle& op,
                                         torch::jit::Stack& stack,
                                         HostCallbackTier tier)
    : g_(&g), tier_(tier), node_idx_(0), num_args_(0),
      stack_size_before_returns_(0) {
    TORCH_CHECK(g.state() == RpuKernelGraph::State::RECORDING,
                "HostCallbackCapture: requires RECORDING state, got ",
                static_cast<int>(g.state()));
    TORCH_CHECK(!g.kernel_register_census_active_ &&
                    !g.has_kernel_register_census_nodes_ &&
                    !g.pending_kernel_register_census_.has_value(),
                "HostCallbackCapture is forbidden by the typed DDR-register "
                "census");
    TORCH_CHECK(tier_ != HostCallbackTier::Reject,
                "HostCallbackCapture: tier=Reject should be filtered before "
                "constructing this guard; op=", op.schema().name());

    const auto& schema = op.schema();
    num_args_ = schema.arguments().size();
    TORCH_CHECK(stack.size() >= num_args_,
                "HostCallbackCapture: stack underflow (size=", stack.size(),
                " num_args=", num_args_, ") for op=", schema.name());
    const size_t stack_start = stack.size() - num_args_;
    // run_cpu_fallback_zerocopy 跑完后 stack 末尾会变成 returns,而前 num_args_
    // 槽被 ATen 移除(boxed redispatch 的标准 stack 语义)。所以"返回起始下标"
    // 等于现在的 stack_start。caller 后续在 adopt_* 用 stack.size() - num_returns
    // 拿 returns。
    stack_size_before_returns_ = stack_start;

    HostCallbackNodeData data;
    data.op_handle = op;       // optional<OperatorHandle> 直接赋 OperatorHandle 隐式 wrap
    data.tier = tier_;
    data.args_template.reserve(num_args_);

    auto output_idx_set = collect_output_arg_indices(schema);

    for (size_t i = 0; i < num_args_; ++i) {
        const c10::IValue& iv = stack[stack_start + i];
        if (iv.isTensor()) {
            data.tensor_arg_indices.push_back(i);
            data.args_template.push_back(at::Tensor{});  // 占位
            const at::Tensor& t = iv.toTensor();
            // 同时填 live_tensor_args:RECORDING 期就用本轮 tensor;REPLAYING op
            // stream 重跑时 capture_host_callback_replay_args 会用本轮新 tensor
            // 覆盖。executor 直接用 live_tensor_args + args_template 重建 stack。
            data.live_tensor_args.push_back(t);
            // RPU input tensor 跑 CPU op 前需要 DDR 对 host 可见 → 登记 pre-flush
            if (t.defined() &&
                t.device().type() == c10::DeviceType::PrivateUse1) {
                g.record_boundary_flush(t.data_ptr());
            }
        } else {
            // 非 Tensor IValue(int / bool / Scalar / List[int] 等)直接存值。
            // replay 期 capture_host_callback_replay_args 校验同 slot 一致。
            data.args_template.push_back(iv);
        }
    }

    data.output_arg_indices = std::move(output_idx_set);

    // Tier1 .out variant: the functional default wrapper
    // (`topk(self, k)` -> alloc values_fresh / indices_fresh -> 调
    //  `topk.values(...,values_fresh, indices_fresh)`)在 dispatcher fallback
    // 之后 return values_fresh / indices_fresh,**不读 stack 替换后的 stable**。
    // 也就是 Python 端 `v, i = topk(...)` 拿到的 i 是 fresh out arg,不是 stable。
    // 下游 `i.to(fp16)` 在 op stream 期需要 defer(REPLAYING 期 fresh out arg
    // 内容要等 executor 才刷),否则读 stale。
    // 解决:在每次 host_callback 调用入口把 .out input args 的 storage 也
    // register 到 admission live 表(只对当前 op stream step 有效;begin
    // (PASSTHROUGH→RECORDING / BUILT→REPLAYING)清,host_callback dispatch
    // 时入新地址,避免跨 step caching allocator 复用 fresh 地址给无关 tensor
    // 时被误命中)。
    if (tier_ == HostCallbackTier::Tier1) {
        for (size_t out_idx : data.output_arg_indices) {
            const c10::IValue& iv = stack[stack_start + out_idx];
            if (iv.isTensor()) {
                g.register_host_callback_live_tier1_out(iv.toTensor());
            }
        }
    }

    // output_pool / output_pool_dev_addrs / output_pool_bytes 在 adopt_returns
    // (Tier 1+2)或 adopt_dynamic_return(Tier 3)阶段填写。Dynamo fx graph 是
    // functional 形式,PT runtime 每次 alloc 全新 .out= 输入,因此统一使用
    // graph-owned stable buffer:CPU op 跑完后由 adopt_returns 把
    // 数据 copy 到 stable buffer,stack slot 替换为 stable buffer。

    // push 进 nodes_;capture_data 走 RECORDING 分支。
    node_idx_ = g.nodes_.size();
    // push 之前先收集 input dev_addrs(用 data.live_tensor_args,
    // 跟 data 里 push 的 tensor 一一对应)。
    std::vector<uint64_t> input_devs;
    input_devs.reserve(data.live_tensor_args.size());
    for (const at::Tensor& t : data.live_tensor_args) {
        if (t.defined() &&
            t.device().type() == c10::DeviceType::PrivateUse1) {
            input_devs.push_back(rhino_lkn::RpuGetDevAddr(t.data_ptr()));
        }
    }
    g.capture_data<HostCallbackNodeData>(std::move(data));
    TORCH_CHECK(g.nodes_.size() == node_idx_ + 1 &&
                g.nodes_[node_idx_].kind == GraphNodeKind::HostCallback,
                "HostCallbackCapture: capture_data did not push HostCallback "
                "node as expected");
    // push 之后调 resolve_recording_input_deps:
    //   - HostCallbackNodeData 自己不存 input_dep 字段(语义不需要 — replay 期
    //     stack 重建跟 dep 无关),只触发反向回填上游 Tier3Oneshot 的 output_dep。
    //   - 返回的 deps 列表丢弃。
    g.resolve_recording_input_deps(static_cast<int>(node_idx_), input_devs);
}

HostCallbackCapture::~HostCallbackCapture() = default;

HostCallbackNodeData& HostCallbackCapture::node_data() {
    TORCH_CHECK(g_ != nullptr && node_idx_ < g_->nodes_.size(),
                "HostCallbackCapture::node_data: graph or node_idx invalid");
    auto& node = g_->nodes_[node_idx_];
    TORCH_CHECK(node.kind == GraphNodeKind::HostCallback,
                "HostCallbackCapture::node_data: node ", node_idx_,
                " is not a HostCallback node");
    return node.as_host_callback();
}

void HostCallbackCapture::adopt_returns(torch::jit::Stack& stack) {
    // Tier 1+2 共用:CPU op 跑完后,把 stack returns 的 fresh tensor 数据
    // copy 到 graph-owned stable RPU buffer,替换 stack slot,登记 post-CPU
    // boundary flush。这样下游 RPU op 永远看到稳定 dev_addr。Tier 3 走
    // adopt_dynamic_return(non-replayable,只 keep-alive 防 GC)。
    TORCH_CHECK(tier_ == HostCallbackTier::Tier1 ||
                tier_ == HostCallbackTier::Tier2,
                "adopt_returns: only valid for Tier1/Tier2; got tier=",
                static_cast<int>(tier_));
    auto& node = node_data();

    TORCH_CHECK(stack.size() >= stack_size_before_returns_,
                "adopt_returns: stack shrunk below pre-call mark");
    const size_t num_rets = stack.size() - stack_size_before_returns_;
    node.output_pool.reserve(num_rets);
    node.output_pool_dev_addrs.reserve(num_rets);
    node.output_pool_bytes.reserve(num_rets);

    for (size_t r = 0; r < num_rets; ++r) {
        const size_t idx = stack_size_before_returns_ + r;
        c10::IValue& iv = stack[idx];
        TORCH_CHECK(iv.isTensor(),
                    "HostCallback adopt_returns: return ", r,
                    " must be Tensor; op=", node.op_handle->schema().name(),
                    " got ", iv.tagKind());
        at::Tensor fresh_out = iv.toTensor();
        // 在 RPU device 上分配同 shape/dtype 的稳定 buffer,copy 数据过去。
        // 不能让 buffer 落 CPU 后再 .to(rpu) —— 那样每次新 dev_addr。
        at::Tensor stable = at::empty(
            fresh_out.sizes(),
            fresh_out.options().device(c10::DeviceType::PrivateUse1));
        stable.copy_(fresh_out);
        const uint64_t dev = rpu_dev_addr_of(stable);
        node.output_pool.push_back(stable);
        node.output_pool_dev_addrs.push_back(dev);
        node.output_pool_bytes.push_back(stable.numel() * stable.element_size());
        // 替换 stack slot:下游 op 读到稳定 RPU buffer
        iv = stable;
        // 登记 post-CPU boundary flush(host 写入要让 RPU 可见)
        if (stable.defined()) {
            g_->record_boundary_flush(stable.data_ptr());
        }
        // 把这块 stable buffer 登记到 admission persistent 表,
        // 让下游的 `_to_copy(self=stable)`(典型 Tier2 functional return
        // `f(...).to(fp16)`)命中后走 deferred 拿当步真值;Qwen3 rotary 内
        // `inv_freq.half() / position_ids.half()` 跟此 chain 无关,不命中,
        // 保持 inline,避免误推进 host_callback Tier2 链。
        // 跨 replay 不变,所以入 persistent 表(begin/abort/invalidate 才清)。
        g_->register_host_callback_persistent_stable(stable);
        // 注册 stable output dev_addr 到 RECORDING dev_addr writer 表。
        // 下游消费者(下游 HostCallback 节点 / 下游 Memcpy / Tier3Oneshot ctor 期)
        // 扫 input dev_addr 时 lookup 到本节点 → 给本节点 push consumer 到
        // output_dep(本节点不是 Tier3Oneshot,output_dep 只在 Tier3 节点有意义,
        // 所以这条对 HostCallback 自己无副作用,纯粹是为了被下游 Tier3Oneshot
        // ctor 路径反向追到)。
        g_->track_recording_writer_dev_addr(dev,
                                            static_cast<int>(node_idx_));
        if (const char* e = std::getenv("RPU_GRAPH_TO_COPY_LOG");
            e && *e && *e != '0') {
            const at::Tensor& iv_t = iv.toTensor();
            std::cerr << "[ADOPT-RET] r=" << r
                      << " stable.data_ptr=0x" << std::hex
                      << reinterpret_cast<uintptr_t>(stable.data_ptr())
                      << " stable.dev=0x" << dev
                      << " iv.data_ptr=0x"
                      << reinterpret_cast<uintptr_t>(iv_t.data_ptr())
                      << std::dec << "\n";
        }
    }
}

void RpuKernelGraph::capture_host_callback_replay_args(
    const c10::OperatorHandle& op, torch::jit::Stack& stack) {
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "HostCallback REPLAY is forbidden by the typed DDR-register "
                "census");
    TORCH_CHECK(state_ == State::REPLAYING,
                "capture_host_callback_replay_args: requires REPLAYING; got ",
                static_cast<int>(state_));
    TORCH_CHECK(cursor_ < nodes_.size(),
                "capture_host_callback_replay_args: cursor ", cursor_,
                " >= nodes.size ", nodes_.size());
    TORCH_CHECK(nodes_[cursor_].kind == GraphNodeKind::HostCallback,
                "capture_host_callback_replay_args: cursor ", cursor_,
                " is not HostCallback (kind=",
                static_cast<int>(nodes_[cursor_].kind), ")");
    auto& hc = nodes_[cursor_].as_host_callback();
    TORCH_CHECK(hc.op_handle.has_value(),
                "capture_host_callback_replay_args: recorded op_handle empty");
    TORCH_CHECK(hc.op_handle->operator_name() == op.operator_name(),
                "capture_host_callback_replay_args: op drift at cursor ",
                cursor_, "; recorded='",
                hc.op_handle->operator_name().name, "' got='",
                op.operator_name().name, "'");
    TORCH_CHECK(hc.tier != HostCallbackTier::Tier3,
                "capture_host_callback_replay_args: Tier3 should not reach "
                "REPLAYING dispatch (scope must have been mark_non_replayable, "
                "end goes oneshot not REPLAYING). op=",
                op.operator_name().name);

    const auto& schema = op.schema();
    const size_t num_args = schema.arguments().size();
    TORCH_CHECK(stack.size() >= num_args,
                "capture_host_callback_replay_args: stack underflow (size=",
                stack.size(), " num_args=", num_args, ")");
    const size_t stack_start = stack.size() - num_args;

    // 用本轮 stack tensor 覆盖 live_tensor_args;校验 scalar 不漂移
    hc.live_tensor_args.clear();
    hc.live_tensor_args.reserve(hc.tensor_arg_indices.size());
    size_t next_t = 0;
    for (size_t i = 0; i < num_args; ++i) {
        const c10::IValue& live = stack[stack_start + i];
        if (next_t < hc.tensor_arg_indices.size() &&
            hc.tensor_arg_indices[next_t] == i) {
            TORCH_CHECK(live.isTensor(),
                        "capture_host_callback_replay_args: arg ", i,
                        " expected Tensor (recorded as tensor slot) got ",
                        live.tagKind());
            hc.live_tensor_args.push_back(live.toTensor());
            ++next_t;
            // 当前轮 input tensor 也要 boundary flush(host 跑 CPU op 前 RPU
            // 写入要可见;executor 调 flush_boundary_ptrs 时会一并下发)
            const at::Tensor& t = live.toTensor();
            if (t.defined() &&
                t.device().type() == c10::DeviceType::PrivateUse1) {
                record_boundary_flush(t.data_ptr());
            }
        } else {
            TORCH_CHECK(i < hc.args_template.size() && live == hc.args_template[i],
                        "capture_host_callback_replay_args: scalar/list arg ", i,
                        " drifted at op=", op.operator_name().name);
        }
    }

    // During REPLAYING, register the current step's .out input
    // args 的 storage 到 admission **live** 表。begin(BUILT→REPLAYING)已经
    // 把 live 表清空(每步 fresh 地址);这里把本步 PT functional default
    // wrapper alloc 的 fresh `.out` args 重新登记,让下游 `i.to(fp16)` 能命中
    // admission 走 deferred(REPLAYING op stream 期 fresh 内容要等 executor
    // 才刷,inline 会读 stale)。详细动机见 RECORDING ctor 内同名 register
    // 调用块的注释。
    if (hc.tier == HostCallbackTier::Tier1) {
        for (size_t out_idx : hc.output_arg_indices) {
            const c10::IValue& iv = stack[stack_start + out_idx];
            if (iv.isTensor()) {
                register_host_callback_live_tier1_out(iv.toTensor());
            }
        }
    }

    // 合成返回值给 op stream:Tier 1+2 都用 graph-owned stable output_pool,
    // 不再依赖入参 .out= 槽地址。executor 跑 CPU op 时也会把 fresh return
    // copy_ 回 output_pool[r]。
    const size_t num_returns = schema.returns().size();
    std::vector<c10::IValue> returns;
    returns.reserve(num_returns);
    if (hc.tier == HostCallbackTier::Tier1 ||
        hc.tier == HostCallbackTier::Tier2) {
        TORCH_CHECK(hc.output_pool.size() == num_returns,
                    "HostCallback Tier", static_cast<int>(hc.tier),
                    ": output_pool.size=", hc.output_pool.size(),
                    " != num_returns=", num_returns,
                    " op=", op.operator_name().name);
        for (size_t r = 0; r < num_returns; ++r) {
            returns.push_back(hc.output_pool[r]);
        }
    }
    // Tier3 上面已经抛过

    // 替换 stack:消费 args,push returns
    stack.erase(stack.end() - num_args, stack.end());
    for (auto& ret : returns) {
        stack.push_back(std::move(ret));
    }

    ++cursor_;
}

// =============================================================================
// Tier3OneshotNode REPLAYING op stream 入口
// =============================================================================
//
// 镜像 capture_host_callback_replay_args 模式但路由到 Tier3OneshotNode。op stream
// 期 custom_cpu_fallback 走到 Tier3 op 时调:不真跑(execute_data_node 在 end()
// 内部按 segment 边界真跑 + copy fresh→transient),只覆写 live_tensor_args、
// 合成 transient buffer 返回、推进 cursor。

void RpuKernelGraph::capture_tier3_oneshot_replay_args(
    const c10::OperatorHandle& op, torch::jit::Stack& stack) {
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "Tier3 REPLAY is forbidden by the typed DDR-register census");
    TORCH_CHECK(state_ == State::REPLAYING,
                "capture_tier3_oneshot_replay_args: requires REPLAYING; got ",
                static_cast<int>(state_));
    TORCH_CHECK(cursor_ < nodes_.size(),
                "capture_tier3_oneshot_replay_args: cursor ", cursor_,
                " >= nodes.size ", nodes_.size());
    TORCH_CHECK(nodes_[cursor_].kind == GraphNodeKind::Tier3Oneshot,
                "capture_tier3_oneshot_replay_args: cursor ", cursor_,
                " is not Tier3Oneshot (kind=",
                static_cast<int>(nodes_[cursor_].kind), ")");
    auto& t = nodes_[cursor_].as_tier3_oneshot();
    TORCH_CHECK(t.op_handle.has_value(),
                "capture_tier3_oneshot_replay_args: recorded op_handle empty");
    TORCH_CHECK(t.op_handle->operator_name() == op.operator_name(),
                "capture_tier3_oneshot_replay_args: op drift at cursor ",
                cursor_, "; recorded='",
                t.op_handle->operator_name().name, "' got='",
                op.operator_name().name, "'");

    const auto& schema = op.schema();
    const size_t num_args = schema.arguments().size();
    TORCH_CHECK(stack.size() >= num_args,
                "capture_tier3_oneshot_replay_args: stack underflow (size=",
                stack.size(), " num_args=", num_args, ")");
    const size_t stack_start = stack.size() - num_args;

    // 用本轮 stack tensor 覆盖 live_tensor_args;校验 scalar 不漂
    t.live_tensor_args.clear();
    t.live_tensor_args.reserve(t.tensor_arg_indices.size());
    size_t next_t = 0;
    for (size_t i = 0; i < num_args; ++i) {
        const c10::IValue& live = stack[stack_start + i];
        if (next_t < t.tensor_arg_indices.size() &&
            t.tensor_arg_indices[next_t] == i) {
            TORCH_CHECK(live.isTensor(),
                        "capture_tier3_oneshot_replay_args: arg ", i,
                        " expected Tensor (recorded as tensor slot) got ",
                        live.tagKind());
            t.live_tensor_args.push_back(live.toTensor());
            ++next_t;
            const at::Tensor& t_arg = live.toTensor();
            // 当前轮 input tensor 也要 boundary flush(host 跑 CPU op 前 RPU
            // 写入要可见)
            if (t_arg.defined() &&
                t_arg.device().type() == c10::DeviceType::PrivateUse1) {
                record_boundary_flush(t_arg.data_ptr());
            }
        } else {
            TORCH_CHECK(i < t.args_template.size() && live == t.args_template[i],
                        "capture_tier3_oneshot_replay_args: scalar/list arg ", i,
                        " drifted at op=", op.operator_name().name);
        }
    }

    // 合成返回值给 op stream:transient buffer 引用(跨 replay dev_addr 稳定,
    // 内容由本轮 execute_data_node 真跑后 copy 进去 — op stream 期 Python 拿
    // 到 tensor 时内容是上轮 stale 的,但下游 RPU op 的实际跑时机是 end()
    // 之后,那时候内容已被刷成本步 RNG 真值)。
    const size_t num_returns = schema.returns().size();
    TORCH_CHECK(t.transient_resource_ids.size() == num_returns,
                "capture_tier3_oneshot_replay_args: transient_resource_ids.size=",
                t.transient_resource_ids.size(),
                " != num_returns=", num_returns,
                " op=", op.operator_name().name);
    std::vector<c10::IValue> returns;
    returns.reserve(num_returns);
    for (size_t r = 0; r < num_returns; ++r) {
        const int res_id = t.transient_resource_ids[r];
        if (res_id < 0) {
            // 非 Tensor return — 理论上 classify_host_op 保证 Tier3 全 Tensor
            // return,这条是防御
            returns.push_back(c10::IValue());
        } else {
            returns.push_back(tier3_transient_tensor_at(
                static_cast<size_t>(res_id)));
        }
    }

    // 替换 stack:消费 args,push returns
    stack.erase(stack.end() - num_args, stack.end());
    for (auto& ret : returns) {
        stack.push_back(std::move(ret));
    }

    ++cursor_;
}

void HostCallbackCapture::adopt_dynamic_return(torch::jit::Stack& stack) {
    TORCH_CHECK(tier_ == HostCallbackTier::Tier3,
                "adopt_dynamic_return: only valid for Tier3; got tier=",
                static_cast<int>(tier_));
    auto& node = node_data();
    TORCH_CHECK(stack.size() >= stack_size_before_returns_,
                "adopt_dynamic_return: stack shrunk below pre-call mark");
    const size_t num_rets = stack.size() - stack_size_before_returns_;
    // Tier3 不承诺跨 replay 稳定,只 keep_alive 防当前 forward 内 GC。复用
    // output_pool 字段(语义:graph-owned tensor refs);output_pool_dev_addrs
    // 留空(跨 replay 没意义)。
    for (size_t r = 0; r < num_rets; ++r) {
        const c10::IValue& iv = stack[stack_size_before_returns_ + r];
        if (iv.isTensor()) {
            node.output_pool.push_back(iv.toTensor());
        }
    }
}

// =============================================================================
// Tier3OneshotCapture — RECORDING 期 RAII guard
// =============================================================================
//
// 跟 HostCallbackCapture 同构,但录的是 Tier3OneshotNodeData(GraphNodeKind::
// Tier3Oneshot)。Tier3 op 走这条路径以保留节点级 oneshot 收益。

Tier3OneshotCapture::Tier3OneshotCapture(RpuKernelGraph& g,
                                          const c10::OperatorHandle& op,
                                          torch::jit::Stack& stack)
    : g_(&g), node_idx_(0), num_args_(0), stack_size_before_returns_(0) {
    TORCH_CHECK(g.state() == RpuKernelGraph::State::RECORDING,
                "Tier3OneshotCapture: requires RECORDING state, got ",
                static_cast<int>(g.state()));
    TORCH_CHECK(!g.kernel_register_census_active_ &&
                    !g.has_kernel_register_census_nodes_ &&
                    !g.pending_kernel_register_census_.has_value(),
                "Tier3OneshotCapture is forbidden by the typed DDR-register "
                "census");

    const auto& schema = op.schema();
    num_args_ = schema.arguments().size();
    TORCH_CHECK(stack.size() >= num_args_,
                "Tier3OneshotCapture: stack underflow (size=", stack.size(),
                " num_args=", num_args_, ") for op=", schema.name());
    const size_t stack_start = stack.size() - num_args_;
    stack_size_before_returns_ = stack_start;

    Tier3OneshotNodeData data;
    data.op_handle = op;
    data.args_template.reserve(num_args_);

    for (size_t i = 0; i < num_args_; ++i) {
        const c10::IValue& iv = stack[stack_start + i];
        if (iv.isTensor()) {
            data.tensor_arg_indices.push_back(i);
            data.args_template.push_back(at::Tensor{});
            const at::Tensor& t = iv.toTensor();
            data.live_tensor_args.push_back(t);
            // RPU input tensor 跑 CPU op 前需 DDR 对 host 可见 → 登记 pre-flush
            if (t.defined() &&
                t.device().type() == c10::DeviceType::PrivateUse1) {
                g.record_boundary_flush(t.data_ptr());
            }
        } else {
            // 非 Tensor IValue 直接存值;REPLAYING capture_replay_args
            // 校验同槽不漂移。
            data.args_template.push_back(iv);
        }
    }

    // push 节点之前先收集 input dev_addrs(用 data.live_tensor_args
    // 跟 data 里 push 的 tensor 一一对应)。push 节点之后调
    // resolve_recording_input_deps 填 input_dep_node_ids,同时给上游
    // Tier3Oneshot 节点反向回填 output_dep_node_ids。
    std::vector<uint64_t> input_devs;
    input_devs.reserve(data.live_tensor_args.size());
    for (const at::Tensor& t : data.live_tensor_args) {
        if (t.defined() &&
            t.device().type() == c10::DeviceType::PrivateUse1) {
            input_devs.push_back(rhino_lkn::RpuGetDevAddr(t.data_ptr()));
        }
    }
    // output_dep_node_ids / transient_resource_ids 由下面的依赖与资源跟踪填充。

    node_idx_ = g.nodes_.size();
    g.capture_data<Tier3OneshotNodeData>(std::move(data));
    TORCH_CHECK(g.nodes_.size() == node_idx_ + 1 &&
                g.nodes_[node_idx_].kind == GraphNodeKind::Tier3Oneshot,
                "Tier3OneshotCapture: capture_data did not push Tier3Oneshot "
                "node as expected");
    // push 后再 resolve,避免拿自己 node_id(虽然 dev addr 还没 register
    // 进去,但 push 之后是干净路径)。
    auto deps = g.resolve_recording_input_deps(static_cast<int>(node_idx_),
                                                input_devs);
    g.nodes_[node_idx_].as_tier3_oneshot().input_dep_node_ids = std::move(deps);
}

Tier3OneshotCapture::~Tier3OneshotCapture() = default;

Tier3OneshotNodeData& Tier3OneshotCapture::node_data() {
    TORCH_CHECK(g_ != nullptr && node_idx_ < g_->nodes_.size(),
                "Tier3OneshotCapture::node_data: graph or node_idx invalid");
    auto& node = g_->nodes_[node_idx_];
    TORCH_CHECK(node.kind == GraphNodeKind::Tier3Oneshot,
                "Tier3OneshotCapture::node_data: node ", node_idx_,
                " is not a Tier3Oneshot node");
    return node.as_tier3_oneshot();
}

void Tier3OneshotCapture::adopt_returns(torch::jit::Stack& stack) {
    auto& node = node_data();
    TORCH_CHECK(stack.size() >= stack_size_before_returns_,
                "Tier3OneshotCapture::adopt_returns: stack shrunk below "
                "pre-call mark");
    const size_t num_rets = stack.size() - stack_size_before_returns_;
    node.last_output_dev_addrs.clear();
    node.last_output_dev_addrs.reserve(num_rets);
    node.transient_resource_ids.clear();
    node.transient_resource_ids.reserve(num_rets);
    for (size_t r = 0; r < num_rets; ++r) {
        c10::IValue& iv = stack[stack_size_before_returns_ + r];
        if (iv.isTensor()) {
            at::Tensor fresh = iv.toTensor();
            // alloc graph-managed transient buffer:同 shape/dtype,
            // RPU device,跨 replay dev_addr 稳定。fresh return 内容 copy 进去,
            // stack slot 替换为 transient buffer 让下游消费者读稳定 dev_addr。
            // 跟 HostCallback Tier1/2 stable buffer 同模式,只差语义上不承诺
            // 内容稳定(每步 RNG 由 REPLAYING 真跑后重 copy 进去)。
            //
            // 不能让 buffer 落 CPU 后再 .to(rpu) — 那样每步新 dev_addr,
            // REPLAYING 期下游 RPU kernel 拿到上轮录的 dev_addr 就崩。
            at::Tensor transient = at::empty(
                fresh.sizes(),
                fresh.options().device(c10::DeviceType::PrivateUse1));
            if (fresh.defined() && transient.defined()) {
                transient.copy_(fresh);
            }
            const uint64_t dev = rpu_dev_addr_of(transient);
            // 入 owner tree:owner_node_id = Tier3 节点自己 idx,dev 用 caller
            // 视角(同 last_output_dev_addrs)避免 sibling RpuGetDevAddr 跨调用
            // 偶现 1-page 漂移导致 owner tree dev 跟节点元数据脱钩。
            const size_t res_id = g_->register_tier3_transient_buffer(
                static_cast<int>(node_idx_), transient, dev);
            node.transient_resource_ids.push_back(static_cast<int>(res_id));
            node.last_output_dev_addrs.push_back(dev);
            // 替换 stack slot:下游 op 直接读 transient buffer。
            iv = transient;
            // output_keepalives 保留 fresh return 引用。
            node.output_keepalives.push_back(fresh);
            // post-write boundary flush:transient buffer host 写入对后续 RPU
            // 可见(executor 跑下游 RPU kernel 前会 flush)。
            if (transient.defined()) {
                g_->record_boundary_flush(transient.data_ptr());
            }
        } else {
            node.last_output_dev_addrs.push_back(0);
            node.transient_resource_ids.push_back(-1);
        }
    }
    // 注册 transient buffer dev_addr 到 RECORDING writer 表。
    // 跨 replay 稳定 dev_addr,下游消费者 lookup 到本节点 → 给本节点 push
    // output_dep。Kernel 消费者通过 stack slot 读 transient buffer dev_addr,
    // graph-层 dep 推断只覆盖 HostCallback/Memcpy 类消费者。
    for (uint64_t dev : node.last_output_dev_addrs) {
        g_->track_recording_writer_dev_addr(dev, static_cast<int>(node_idx_));
    }
}
