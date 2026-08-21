// =============================================================================
// SpmAllocator 实现
// =============================================================================

#include "rpu_spm_allocator.h"
#include "rhino_launch_buffer.h"

#include <torch/torch.h>  // TORCH_CHECK

using namespace rhino_lkn;

// =============================================================================
// 初始化
// =============================================================================

void SpmAllocator::init() {
    if (initialized_) return;

    blocks_.resize(NUM_CORES, nullptr);
    for (int c = 0; c < NUM_CORES; ++c) {
        blocks_[c] = new GlobalSPM_t(SPM_TOTAL, read_write, kStride32B, 2, c);
        base_addr_[c] = (uint32_t)blocks_[c]->get_rpu_addr();
        base_cpu_ptr_[c] = blocks_[c]->get_cpu_ptr();
    }

    t_end_ = 0;
    sp_floor_ = SPM_USABLE;
    p_start_ = sp_floor_;
    initialized_ = true;

}

SpmAllocator::~SpmAllocator() {
    // 不在析构中释放 — rpu_shutdown 已清理硬件资源
    blocks_.clear();
}

// =============================================================================
// 分配
// =============================================================================

uint32_t SpmAllocator::alloc_temporary(size_t size) {
    TORCH_CHECK(!pipeline_arena_locked_,
        "SpmAllocator: temporary allocation is forbidden while physical "
        "pipeline arena epoch ", pipeline_arena_epoch_, " is active");
    TORCH_CHECK(initialized_, "SpmAllocator not initialized");
    size_t aligned = align_up(size);
    TORCH_CHECK(t_end_ + aligned <= p_start_,
        "SpmAllocator OOM: temporary alloc ", aligned, " bytes, "
        "t_end=", t_end_, " p_start=", p_start_,
        " free=", p_start_ - t_end_);

    uint32_t offset = (uint32_t)t_end_;
    t_end_ += aligned;
    return offset;
}

SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(
    const std::vector<AllocRequest>& requests)
{
    size_t n = requests.size();
    if (n == 0) return {};

    // Per-buffer state: aligned size + computed relative offset
    struct Slot {
        size_t aligned_bytes;
        int first_step, last_step;
        int scope;
        size_t rel_offset;
        bool placed;
    };
    std::vector<Slot> slots(n);
    for (size_t i = 0; i < n; i++) {
        slots[i] = {align_up(requests[i].bytes),
                     requests[i].first_step, requests[i].last_step,
                     requests[i].scope, 0, false};
    }

    // Lifetime conflict check with KV_FIRST subgraph awareness.
    //
    // LayerWide (scope 0) buffers are alive across ALL subgraphs, so they
    // always conflict with KvInsert/Compute buffers — their phase numbers
    // only cover one subgraph's step space and cannot express cross-subgraph
    // lifetime. Two LayerWide buffers still use phase-overlap among themselves.
    //
    // KvInsert (1) vs Compute (2) never conflict — disjoint time windows.
    //
    // OutsideLayerLoop (3) is used by pre_layers_fn/post_layers_fn temporaries
    // whose data crosses the layer-loop boundary through DDR. It therefore never
    // conflicts with LayerWide/KvInsert/Compute. Two OutsideLayerLoop buffers
    // still use phase overlap, allowing disjoint pre-layer and post-layer phases
    // to alias. (Persistent* slots are a different StorageClass and never appear
    // in this request list.)
    //
    // Same-scope buffers use standard phase overlap.
    auto has_conflict = [](const Slot& a, const Slot& b) {
        // OutsideLayerLoop vs layer-loop scope → never conflict
        if ((a.scope == 3) != (b.scope == 3))
            return false;
        // KvInsert vs Compute → never conflict (disjoint time windows)
        if ((a.scope == 1 && b.scope == 2) ||
            (a.scope == 2 && b.scope == 1))
            return false;
        // LayerWide vs non-LayerWide → always conflict
        // (LayerWide spans all subgraphs, phase ranges are unreliable)
        if ((a.scope == 0) != (b.scope == 0))
            return true;
        // Same scope → standard phase overlap check
        return a.first_step <= b.last_step && a.last_step >= b.first_step;
    };

    // Sort indices by first_step (stable, preserves order for ties)
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return slots[a].first_step < slots[b].first_step;
    });

    // Greedy first-fit: for each buffer, find lowest offset with no lifetime conflict
    size_t peak = 0;
    for (size_t idx : order) {
        auto& s = slots[idx];

        // Collect occupied intervals from buffers with overlapping lifetime
        std::vector<std::pair<size_t, size_t>> occupied;
        for (size_t j = 0; j < n; j++) {
            if (j == idx || !slots[j].placed) continue;
            if (has_conflict(slots[j], s)) {
                occupied.push_back({slots[j].rel_offset,
                                    slots[j].rel_offset + slots[j].aligned_bytes});
            }
        }
        std::sort(occupied.begin(), occupied.end());

        // First-fit scan
        size_t candidate = 0;
        for (auto& [start, end] : occupied) {
            if (candidate + s.aligned_bytes <= start) break;
            if (end > candidate) candidate = end;
        }

        s.rel_offset = candidate;
        s.placed = true;
        peak = std::max(peak, candidate + s.aligned_bytes);
    }

    AliasedPlan plan;
    plan.offsets.resize(n);
    for (size_t i = 0; i < n; i++) {
        plan.offsets[i] = static_cast<uint32_t>(slots[i].rel_offset);
    }
    plan.peak_bytes = peak;
    return plan;
}

std::vector<uint32_t> SpmAllocator::alloc_temporary_aliased(
    const std::vector<AllocRequest>& requests)
{
    TORCH_CHECK(initialized_, "SpmAllocator not initialized");
    AliasedPlan plan = plan_temporary_aliased(requests);
    if (plan.offsets.empty()) return {};

    // Allocate one contiguous temporary block for the whole group.
    uint32_t base = alloc_temporary(plan.peak_bytes);
    for (uint32_t& offset : plan.offsets) {
        offset += base;
    }
    return plan.offsets;
}

uint32_t SpmAllocator::alloc_persistent(size_t size) {
    TORCH_CHECK(!pipeline_arena_locked_,
        "SpmAllocator: persistent allocation is forbidden while physical "
        "pipeline arena epoch ", pipeline_arena_epoch_, " is active");
    TORCH_CHECK(initialized_, "SpmAllocator not initialized");
    size_t aligned = align_up(size);
    TORCH_CHECK(p_start_ >= aligned && p_start_ - aligned >= t_end_,
        "SpmAllocator OOM: persistent alloc ", aligned, " bytes, "
        "t_end=", t_end_, " p_start=", p_start_,
        " free=", p_start_ - t_end_);

    p_start_ -= aligned;
    return (uint32_t)p_start_;
}

uint32_t SpmAllocator::alloc_super_persistent(size_t size) {
    TORCH_CHECK(!pipeline_arena_locked_,
        "SpmAllocator: super-persistent allocation is forbidden while physical "
        "pipeline arena epoch ", pipeline_arena_epoch_, " is active");
    TORCH_CHECK(initialized_, "SpmAllocator not initialized");
    size_t aligned = align_up(size);
    // super-persistent 必须在普通 persistent 之上分配。如果 p_start_ < sp_floor_
    // 说明已经有 persistent 占用了 [p_start_, sp_floor_) 区间, 此时分配 super-
    // persistent 会与 persistent 重叠 — 调用方必须先 reset_all 清空 persistent。
    TORCH_CHECK(p_start_ == sp_floor_,
        "alloc_super_persistent: must be called when no regular persistent "
        "is allocated (call reset_all first). p_start=", p_start_,
        " sp_floor=", sp_floor_);
    TORCH_CHECK(sp_floor_ >= aligned && sp_floor_ - aligned >= t_end_,
        "SpmAllocator OOM: super_persistent alloc ", aligned, " bytes, "
        "t_end=", t_end_, " sp_floor=", sp_floor_,
        " free=", sp_floor_ - t_end_);

    sp_floor_ -= aligned;
    p_start_ = sp_floor_;  // 同步下移 persistent 起点
    return (uint32_t)sp_floor_;
}

// =============================================================================
// 释放
// =============================================================================

void SpmAllocator::reset_temporary() {
    TORCH_CHECK(!pipeline_arena_locked_,
        "SpmAllocator: reset_temporary is forbidden while physical pipeline "
        "arena epoch ", pipeline_arena_epoch_, " is active");
    t_end_ = 0;
    generation_++;
    // NOTE: persistent_generation_ NOT incremented — persistent SPM data survives.
    // Subsystems that allocated persistent buffers can detect via persistent_generation()
    // that their data is still valid (avoids unnecessary rebuild + re-DMA).
}

void SpmAllocator::reset_all() {
    TORCH_CHECK(!pipeline_arena_locked_,
        "SpmAllocator: reset_all is forbidden while physical pipeline arena "
        "epoch ", pipeline_arena_epoch_, " is active");
    t_end_ = 0;
    p_start_ = sp_floor_;  // 只重置到 super-persistent 底部, 保留 super-persistent 数据
    generation_++;
    persistent_generation_++;  // persistent buffers were wiped — subsystems must re-allocate
}

// =============================================================================
// Instance lifecycle — release the super_persistent floor when no
// FusedModelBase instance is alive so sequential model loads do not accumulate
// stale allocations.
// =============================================================================

void SpmAllocator::register_instance() {
    TORCH_CHECK(active_instance_count_ >= 0,
                "SpmAllocator::register_instance: counter went negative (",
                active_instance_count_, "); FusedModelBase ctor/dtor pairing broken");
    ++active_instance_count_;
}

void SpmAllocator::unregister_instance() {
    TORCH_CHECK(active_instance_count_ > 0,
                "SpmAllocator::unregister_instance: count already 0; "
                "dtor called without matching ctor (or double-dtor)");
    --active_instance_count_;
    if (active_instance_count_ == 0 && sp_floor_ < SPM_USABLE) {
        if (pipeline_arena_locked_) {
            deferred_floor_release_ = true;
            return;
        }
        // All model instances destroyed → no live kernel graph references
        // a super_persistent SPM address. Safe to release the floor.
        // Implies regular persistent is also cleared (reset_all semantics).
        t_end_  = 0;
        sp_floor_ = SPM_USABLE;
        p_start_  = sp_floor_;
        ++generation_;
        ++persistent_generation_;
        deferred_floor_release_ = false;
    }
}

void SpmAllocator::release_deferred_floor_if_idle() noexcept {
    if (!deferred_floor_release_ || pipeline_arena_locked_ ||
        active_instance_count_ != 0) {
        return;
    }
    t_end_ = 0;
    sp_floor_ = SPM_USABLE;
    p_start_ = sp_floor_;
    ++generation_;
    ++persistent_generation_;
    deferred_floor_release_ = false;
}

// =============================================================================
// Debug
// =============================================================================

void SpmAllocator::dump(const char* label) const {
    printf("[SpmAllocator] %s: temporary=%.1fKB (%.1f%%)  persistent=%.1fKB (%.1f%%)  "
           "super_persistent=%.1fKB (%.1f%%)  free=%.1fKB  total_used=%.1fKB/%.1fKB\n",
           label,
           temporary_used() / 1024.0,
           temporary_used() * 100.0 / SPM_USABLE,
           persistent_used() / 1024.0,
           persistent_used() * 100.0 / SPM_USABLE,
           super_persistent_used() / 1024.0,
           super_persistent_used() * 100.0 / SPM_USABLE,
           free_space() / 1024.0,
           total_used() / 1024.0,
           SPM_USABLE / 1024.0);
}
