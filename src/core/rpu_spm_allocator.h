#pragma once
// =============================================================================
// SpmAllocator — 对向增长 bump allocator
//
// 管理 8 core × 8MB SPM（预留 1KB，SPM_USABLE ~8MB），零 new/delete 开销。
// persistent 从顶部向下增长，temporary 从底部向上增长。
// 所有 core 共用同一对 (t_end, p_start) 指针 → 天然保证偏移一致。
//
// 用法:
//   auto& spm = SpmAllocator::instance();
//   uint32_t q_off = spm.alloc_temporary(q_size);     // 临时 buffer
//   uint32_t w_off = spm.alloc_persistent(norm_size);  // 持久 buffer
//   spm.reset_temporary();   // 释放所有临时，保留持久
//   spm.reset_all();         // 释放全部
//
//   // 获取硬件地址（用于 kernel 寄存器）
//   uint32_t hw_addr = spm.addr(core_id, offset);
//   // 获取 CPU 指针（用于 DMA）
//   void* cpu_ptr = spm.cpu_ptr(core_id, offset);
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace rhino_lkn {
class GlobalSPM_t;
}

namespace v3 {
class SpmPipelineLease;
}

class SpmAllocator {
public:
    static constexpr size_t SPM_TOTAL    = 8 * 1024 * 1024;   // 8MB per core
    static constexpr size_t SPM_RESERVED = 1 * 1024;             // 1KB reserved
    static constexpr size_t SPM_USABLE   = SPM_TOTAL - SPM_RESERVED;  // ~8MB
    // Shared model/operator planning ceiling: floor(0.99 * SPM_USABLE).
    // 8,303,708 B ~= 7.919 MiB/core, leaving a raw 83,876 B guard. The
    // largest 256-byte-aligned plan is 8,303,616 B (83,968 B physical guard).
    static constexpr size_t SPM_PLANNING_BUDGET =
        SPM_USABLE * 99 / 100;
    static constexpr size_t ALIGN        = 256;                 // 256B alignment
    static constexpr int    NUM_CORES    = 8;

    static SpmAllocator& instance() {
        static SpmAllocator alloc;
        return alloc;
    }

    // =========================================================================
    // 初始化：预分配每个 core 的 8MB GlobalSPM_t
    // 必须在 RPU 硬件初始化后、首次使用前调用
    // =========================================================================
    void init();
    bool is_initialized() const { return initialized_; }

    // =========================================================================
    // 分配
    // =========================================================================

    // 临时 buffer：从底部向上增长，reset_temporary 释放
    uint32_t alloc_temporary(size_t size);

    // 批量分配临时 buffer 并自动做生命周期别名复用。
    // 生命周期不重叠的 buffer 共享同一 SPM 区域，峰值 = max(concurrent) 而非 sum(all)。
    // requests: {bytes, first_step, last_step, scope} 数组（step 含义由调用方定义）
    // scope: 0=LayerWide, 1=KvInsert, 2=Compute, 3=OutsideLayerLoop.
    //   - LayerWide always conflicts with KvInsert/Compute (spans all subgraphs,
    //     phase ranges are unreliable across scope boundaries).
    //   - KvInsert vs Compute: never conflict (disjoint time windows).
    //   - OutsideLayerLoop never conflicts with layer-loop scopes; phase ranges
    //     still govern overlap with other OutsideLayerLoop requests.
    //   - Same scope: standard phase overlap check.
    // 返回: 每个 buffer 的 SPM offset（与 alloc_temporary 返回值同语义）
    struct AllocRequest {
        int64_t bytes;
        int first_step;
        int last_step;
        int scope = 0;
    };

    struct AliasedPlan {
        std::vector<uint32_t> offsets;  // relative to the group's base
        size_t peak_bytes = 0;
    };

    // Pure first-fit planner: no RPU/SPM initialization or allocation required.
    static AliasedPlan plan_temporary_aliased(
        const std::vector<AllocRequest>& requests);

    std::vector<uint32_t> alloc_temporary_aliased(const std::vector<AllocRequest>& requests);

    // 持久 buffer：从顶部向下增长，仅 reset_all 释放
    uint32_t alloc_persistent(size_t size);

    // 超持久 buffer：在 SPM 顶部预留区域，永远不被 reset_all 清空。
    //
    // 用途: 跨子系统共享 SPM 时, 让子系统 A 的权重数据在子系统 B 调用 reset_all
    // 后仍然有效。例如 Pi0.5 流水线 (SigLIP → Gemma VLM → AdaRMS), 三者轮流执
    // 行各自的 Path 1 (调用 reset_all), 默认行为下 SigLIP 在 cycle 2 必须重新
    // DMA 27 层权重 (~272 个 DMA kernels) 因为 Gemma/AdaRMS 的 reset_all 已经
    // 把它的 persistent SPM 区域覆写。把权重放到 super-persistent 后, reset_all
    // 只重置到 sp_floor_ 而不是 SPM_USABLE, SigLIP 权重得以保留, SIGLIP_WEIGHTS
    // 与 SIGLIP_COMPUTE 都能跨 cycle REPLAY。
    //
    // 约束: 只能在 reset_all 后、第一次 alloc_persistent 之前分配; 一旦分配就
    // 永久占用 SPM 顶部那段地址 (sp_floor_ 单调下降, 不会被任何 reset 重置)。
    uint32_t alloc_super_persistent(size_t size);

    // =========================================================================
    // 释放
    // =========================================================================

    // 释放所有临时 buffer（保留持久 + 超持久）
    void reset_temporary();

    // 释放临时 + 普通持久（保留超持久）
    // 注: super-persistent 区域永不释放, 只在进程退出时随 SPM 一起回收。
    void reset_all();

    // 作用域式临时分配：mark() 记下当前 temporary 水位, release_to(mark) 只回退
    // 到该水位。给 eager 算子借用 SPM 用 —— reset_temporary() 是全量清零, 会把
    // 融合子系统正在使用的 temporary 一并抹掉; mark/release 则只归还自己那段。
    //   const auto mk = SPM_ALLOC.temporary_mark();
    //   uint32_t off = SPM_ALLOC.alloc_temporary(bytes);
    //   ... use ...
    //   SPM_ALLOC.temporary_release_to(mk);   // 或用下面的 ScopedTemporary
    size_t temporary_mark() const { return t_end_; }
    void temporary_release_to(size_t mark) {
        if (mark <= t_end_) t_end_ = mark;
    }

    // RAII 版本：析构时自动回退到构造时的水位。
    class ScopedTemporary {
    public:
        ScopedTemporary() : mark_(SpmAllocator::instance().temporary_mark()) {}
        ~ScopedTemporary() { SpmAllocator::instance().temporary_release_to(mark_); }
        ScopedTemporary(const ScopedTemporary&) = delete;
        ScopedTemporary& operator=(const ScopedTemporary&) = delete;
        size_t mark() const { return mark_; }
    private:
        size_t mark_;
    };

    // =========================================================================
    // 地址转换
    // =========================================================================

    // offset → 硬件 SPM 地址（用于 kernel 寄存器，broadcast 模式下所有 core 同一 offset）
    uint32_t addr(int core_id, uint32_t offset) const {
        return base_addr_[core_id] + offset;
    }

    // offset → CPU 指针（用于 DMA: ddr_to_spm / spm_to_ddr）
    void* cpu_ptr(int core_id, uint32_t offset) const {
        return static_cast<char*>(base_cpu_ptr_[core_id]) + offset;
    }

    // =========================================================================
    // 查询
    // =========================================================================

    size_t temporary_used() const { return t_end_; }
    size_t persistent_used() const { return sp_floor_ - p_start_; }
    size_t super_persistent_used() const { return SPM_USABLE - sp_floor_; }
    size_t total_used() const { return temporary_used() + persistent_used() + super_persistent_used(); }
    size_t free_space() const { return p_start_ - t_end_; }

    // Generation: incremented on every reset (temporary OR all), used by clients
    // to detect invalidation. Use persistent_generation() to distinguish whether
    // persistent buffers were also wiped.
    uint64_t generation() const { return generation_; }

    // Persistent generation: incremented ONLY on reset_all(). Subsystems with
    // persistent allocations use this to detect whether their persistent SPM
    // data was actually wiped (vs only temporary being reset). This avoids the
    // pessimistic "any reset == full rebuild" fallback in FusedLayerBase, which
    // would force SigLIP to BUILD weights and main graphs every forward instead
    // of REPLAY.
    uint64_t persistent_generation() const { return persistent_generation_; }

    // =========================================================================
    // Instance lifecycle.
    //
    // Counts live FusedModelBase instances. super_persistent is process-wide
    // monotonic during a "session" (count > 0), but when all model instances
    // are destroyed (count → 0), the floor is released back so the next fresh
    // load does not accumulate stale allocations. Safe because every kernel
    // graph that bakes a super_persistent SPM address is owned by some
    // FusedModelBase instance — when count drops to 0, no graph references
    // those addresses anymore.
    //
    // Required ordering: register_instance() called from FusedModelBase ctor,
    // unregister_instance() from dtor. Both protected by the GIL (single-threaded
    // dispatch invariant — see model_handle_registry.h "Thread-safety contract").
    // =========================================================================
    void register_instance();
    void unregister_instance();
    int  active_instance_count() const { return active_instance_count_; }

    void dump(const char* label = "") const;

private:
    SpmAllocator() = default;
    ~SpmAllocator();
    SpmAllocator(const SpmAllocator&) = delete;
    SpmAllocator& operator=(const SpmAllocator&) = delete;

    static size_t align_up(size_t size) {
        return (size + ALIGN - 1) & ~(ALIGN - 1);
    }

    bool initialized_ = false;

    // 每 core 的预分配 SPM block
    std::vector<rhino_lkn::GlobalSPM_t*> blocks_;  // [NUM_CORES]
    uint32_t base_addr_[NUM_CORES] = {};             // 硬件 base 地址
    void*    base_cpu_ptr_[NUM_CORES] = {};           // CPU base 指针

    // 对向增长指针（offset，所有 core 共用）
    // SPM 布局 (从低到高):
    //   [0, t_end_)              temporary (向上)
    //   [t_end_, p_start_)       free
    //   [p_start_, sp_floor_)    persistent (向下, 从 sp_floor_)
    //   [sp_floor_, SPM_USABLE)  super-persistent (向下, 从 SPM_USABLE)
    // reset_temporary: t_end_=0
    // reset_all: t_end_=0, p_start_=sp_floor_  (super-persistent 保留)
    // alloc_super_persistent: sp_floor_-=size, p_start_=sp_floor_ (单调下降, 永不重置)
    size_t t_end_    = 0;           // temporary 顶部（从 0 向上）
    size_t p_start_  = SPM_USABLE;  // persistent 底部（从 sp_floor_ 向下）
    size_t sp_floor_ = SPM_USABLE;  // super-persistent 底部（从 SPM_USABLE 向下, 单调）
    uint64_t generation_ = 0;      // reset 计数（temporary OR all），用于客户端失效检测
    uint64_t persistent_generation_ = 0;  // reset_all 计数，用于精确判断 persistent 是否被清空
    int active_instance_count_ = 0;  // Live FusedModelBase count.
    bool deferred_floor_release_ = false;

    // FusedModelBase destruction is noexcept.  If the last instance disappears
    // during a physical lease, defer the super-persistent floor mutation until
    // that lease releases outside Graph instead of invalidating baked addresses.
    void release_deferred_floor_if_idle() noexcept;

    // A physical SpmPipelineLease reserves [0, t_end_) outside Graph capture
    // and freezes every allocator mutation until the synchronous pipeline has
    // completed.  Only SpmPipelineLease may toggle this bit; all public bump /
    // reset entry points fail loudly while it is set.
    bool pipeline_arena_locked_ = false;
    uint64_t pipeline_arena_epoch_ = 0;

    friend class v3::SpmPipelineLease;
};

#define SPM_ALLOC (SpmAllocator::instance())
