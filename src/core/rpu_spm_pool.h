// rpu_spm_pool.h — SPM Buffer Pool for reducing allocation/deallocation overhead
#pragma once

#include <cstdint>
#include <cstdio>
#include <map>
#include <unordered_map>
#include <vector>
#include <c10/util/Exception.h>  // TORCH_CHECK (depth-aware defer scope)
#include "rpu_profile.h"  // Align macro
#include "rhino_launch_buffer.h"

// =============================================================================
// SPM Buffer Pool - 减少 SPM 分配/释放开销
// =============================================================================
// 对于固定形状的推理，SPM buffer 大小是确定的，可以重用
// 使用 size-class 池化策略，按 aligned size 分类管理
// =============================================================================

constexpr int SPM_BANK_SIZE = 32;

// -----------------------------------------------------------------------------
// RPU_CHECK_SPM — the eager ops build LocalSPM_t on the stack and then memcpy
// through get_cpu_ptr(). Buffer_t's ctor leaves cpu_ptr_ = nullptr and the
// derived ctor fills it ONLY ON SUCCESS, so exhaustion otherwise leaves a null
// pointer for the following memcpy. This check converts that condition into an
// error naming the buffer and byte count.
//
// It does NOT make the eager ops coexist with a live SPM_ALLOC. That needs
// routing them through SPM_ALLOC; this only makes the incompatibility explicit.
#define RPU_CHECK_SPM(buf, bytes)                                              \
    TORCH_CHECK((buf).get_cpu_ptr() != nullptr,                                \
                "SPM allocation failed for '" #buf "' (", (bytes),             \
                " bytes). The SPM pool is exhausted, most likely because "     \
                "SPM_ALLOC already owns it — which is the case once any fused "\
                "model or SPM op has run in this process. Eager ops that use "  \
                "LocalSPM_t cannot share SPM with SPM_ALLOC; see rpu_spm_pool.h.")

class SpmBufferPool {
public:
    static SpmBufferPool& instance() {
        static SpmBufferPool inst;
        return inst;
    }

    // 获取指定大小的 GlobalSPM_t buffer (8核版本)
    ::rhino_lkn::GlobalSPM_t* acquire_global(size_t size, int core_id) {
        size_t aligned_size = Align(size, SPM_BANK_SIZE);
        uint64_t key = make_key(aligned_size, core_id);

        auto& pool = global_pools_[key];
        if (!pool.empty()) {
            auto* buf = pool.back();
            pool.pop_back();
            return buf;
        }
        return new ::rhino_lkn::GlobalSPM_t(aligned_size, ::rhino_lkn::read_write,
                                            ::rhino_lkn::kStride32B, 2, core_id);
    }

    // 释放 GlobalSPM_t buffer
    void release_global(::rhino_lkn::GlobalSPM_t* buf, size_t size, int core_id) {
        if (!buf) return;
        if (enable_pooling_) {
            size_t aligned_size = Align(size, SPM_BANK_SIZE);
            uint64_t key = make_key(aligned_size, core_id);
            global_pools_[key].push_back(buf);
        } else {
            delete buf;  // 真正释放 SPM 空间
        }
    }

    // 启用/禁用池化（默认禁用，即直接释放 SPM）
    void set_pooling_enabled(bool enabled) { enable_pooling_ = enabled; }
    bool is_pooling_enabled() const { return enable_pooling_; }

    // 获取指定大小的 LocalSPM_t buffer
    ::rhino_lkn::LocalSPM_t* acquire_local(size_t size, int core_id) {
        size_t aligned_size = Align(size, SPM_BANK_SIZE);
        uint64_t key = make_key(aligned_size, core_id);

        auto& pool = local_pools_[key];
        if (!pool.empty()) {
            auto* buf = pool.back();
            pool.pop_back();
            return buf;
        }
        return new ::rhino_lkn::LocalSPM_t(aligned_size, ::rhino_lkn::read_write,
                                           ::rhino_lkn::kStride32B, 2, core_id);
    }

    // 归还 LocalSPM_t buffer 到池中
    // Graph scope(可嵌套)内自动 defer,最外层 exit_defer_scope() 统一归还。
    void release_local(::rhino_lkn::LocalSPM_t* buf, size_t size, int core_id) {
        if (!buf) return;
        if (defer_depth_ > 0) {
            deferred_releases_.push_back({buf, size, core_id});
            return;
        }
        release_local_impl(buf, size, core_id);
    }

    // Graph scope 控制(depth-aware,支持 nested begin/end):
    //   enter_defer_scope : depth++,首次翻 true
    //   exit_defer_scope  : depth--;depth==0 时自动 flush_deferred
    // 旧 API set_defer_mode(bool) 保留为兼容入口,转发到 enter/exit。
    void enter_defer_scope() { ++defer_depth_; }
    void exit_defer_scope() {
        TORCH_CHECK(defer_depth_ > 0,
                    "SpmBufferPool: exit_defer_scope without matching enter");
        --defer_depth_;
        if (defer_depth_ == 0) {
            flush_deferred();
        }
    }
    void set_defer_mode(bool v) {
        if (v) enter_defer_scope();
        else   exit_defer_scope();   // depth==0 时 flush_deferred 自动触发
    }
    bool is_defer_mode() const { return defer_depth_ > 0; }
    size_t defer_depth() const { return defer_depth_; }

    void flush_deferred() {
        for (auto& d : deferred_releases_) {
            release_local_impl(d.spm, d.size, d.core);
        }
        deferred_releases_.clear();
    }

    // 打印各 core 的 SPM 使用情况
    void dump_usage(const char* label) {
        printf("[SPM_POOL] %s — pooled (free) buffers:\n", label);
        std::map<int, std::pair<int, size_t>> core_stats;
        for (auto& [key, pool] : global_pools_) {
            int core_id = static_cast<int>(key & 0xFF);
            size_t size = static_cast<size_t>(key >> 8);
            core_stats[core_id].first += (int)pool.size();
            core_stats[core_id].second += size * pool.size();
        }
        for (auto& [core, stats] : core_stats) {
            printf("  core %d: %d free bufs, %.1f KB pooled\n",
                   core, stats.first, stats.second / 1024.0);
        }
        if (core_stats.empty()) {
            printf("  (no pooled buffers)\n");
        }
    }

    // 清空所有池化的 buffer (用于测试/shutdown)
    void clear() {
        // 强制把任何未配对的 enter_defer_scope 归零,保证 shutdown 后续调用
        // 不会误判 defer。如果 depth>0 这里就被踩到,几乎一定是 begin/end
        // 不配对的 bug,但 shutdown 路径不抛错,只把状态清干净。
        defer_depth_ = 0;
        flush_deferred();
        for (auto& [key, pool] : global_pools_) {
            for (auto* buf : pool) delete buf;
            pool.clear();
        }
        for (auto& [key, pool] : local_pools_) {
            for (auto* buf : pool) delete buf;
            pool.clear();
        }
    }

private:
    SpmBufferPool() = default;
    ~SpmBufferPool() { clear(); }
    SpmBufferPool(const SpmBufferPool&) = delete;
    SpmBufferPool& operator=(const SpmBufferPool&) = delete;

    void release_local_impl(::rhino_lkn::LocalSPM_t* buf, size_t size, int core_id) {
        size_t aligned_size = Align(size, SPM_BANK_SIZE);
        uint64_t key = make_key(aligned_size, core_id);
        local_pools_[key].push_back(buf);
    }

    static uint64_t make_key(size_t size, int core_id) {
        return (static_cast<uint64_t>(size) << 8) | static_cast<uint64_t>(core_id);
    }

    struct DeferredRelease {
        ::rhino_lkn::LocalSPM_t* spm;
        size_t size;
        int core;
    };

    bool enable_pooling_ = false;
    // depth-aware defer:每次 RpuKernelGraph::begin() enter_defer_scope() ++,
    // end()/abort() exit_defer_scope() --。depth==0 时 release_local 立即归还;
    // depth>0 期间 release_local 推到 deferred_releases_,等最外层 exit 时统一
    // flush。nested begin/end 不会踩翻外层的 defer 状态。
    size_t defer_depth_ = 0;
    std::vector<DeferredRelease> deferred_releases_;
    std::unordered_map<uint64_t, std::vector<::rhino_lkn::GlobalSPM_t*>> global_pools_;
    std::unordered_map<uint64_t, std::vector<::rhino_lkn::LocalSPM_t*>> local_pools_;
};

// 便捷宏
#define SPM_POOL (SpmBufferPool::instance())
