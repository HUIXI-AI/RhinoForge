// rpu_copy_safety.h — copy-capture safety checks and counter.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

// Safety criteria:
//   bytes == 0              → safe（no-op）
//   bytes > 128 MB          → unsafe（异常大块，单 frame 不应出现）
//   dst/src 区间重叠        → unsafe（memcpy UB，capture_data 无法保持语义）
//   dst/src 距调用栈帧 < 256 KB → unsafe（疑似 stack-local，replay 期生命周期不稳）
namespace rpu_copy_safety_detail {
inline constexpr size_t kMaxSafeBytes = 128ULL * 1024 * 1024;
inline constexpr size_t kStackProximityBytes = 256 * 1024;

inline bool regions_overlap(const void* a, const void* b, size_t bytes) {
    if (a == nullptr || b == nullptr || bytes == 0) return false;
    const auto pa = reinterpret_cast<uintptr_t>(a);
    const auto pb = reinterpret_cast<uintptr_t>(b);
    const uintptr_t a_end = pa + bytes;
    const uintptr_t b_end = pb + bytes;
    return (pa < b_end) && (pb < a_end);
}

inline bool near_stack(const void* p, uintptr_t anchor) {
    if (p == nullptr) return false;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(p);
    const uintptr_t diff = addr > anchor ? addr - anchor : anchor - addr;
    return diff < kStackProximityBytes;
}
}  // namespace rpu_copy_safety_detail

inline bool is_unsafe_copy(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return false;
    if (bytes > rpu_copy_safety_detail::kMaxSafeBytes) return true;
    if (rpu_copy_safety_detail::regions_overlap(dst, src, bytes)) return true;

    volatile char anchor = 0;
    const uintptr_t a = reinterpret_cast<uintptr_t>(&anchor);
    return rpu_copy_safety_detail::near_stack(dst, a) ||
           rpu_copy_safety_detail::near_stack(src, a);
}

inline bool is_unsafe_memset(void* dst, size_t bytes) {
    if (bytes == 0) return false;
    if (bytes > rpu_copy_safety_detail::kMaxSafeBytes) return true;

    volatile char anchor = 0;
    const uintptr_t a = reinterpret_cast<uintptr_t>(&anchor);
    return rpu_copy_safety_detail::near_stack(dst, a);
}

// CopyCounter — profiling counters for capture and execution paths.
// 前四类是 capture-time 分类;后面是 execute-time DMA 计数,用来区分
// "node 被分类为 DMA-able" 和 "DMA 路径真的在 replay/oneshot 时跑过"。
struct CopyCounter {
    std::atomic<uint64_t> captured_as_node{0};  // 安全路径：capture_data 成功
    std::atomic<uint64_t> eager_fallback{0};    // RECORDING unsafe: sync_point + eager
    std::atomic<uint64_t> blocked{0};           // REPLAYING unsafe: TORCH_CHECK(false)
    std::atomic<uint64_t> passthrough{0};       // PASSTHROUGH eager
    std::atomic<uint64_t> dma_ddr_to_ddr{0};    // execute-time CopyMemoryChannel count
    std::atomic<uint64_t> dma_ddr_to_spm{0};    // reserved until sliced SPM↔DDR DMA exists
    std::atomic<uint64_t> dma_spm_to_ddr{0};    // reserved until sliced SPM↔DDR DMA exists

    void reset() {
        captured_as_node.store(0);
        eager_fallback.store(0);
        blocked.store(0);
        passthrough.store(0);
        dma_ddr_to_ddr.store(0);
        dma_ddr_to_spm.store(0);
        dma_spm_to_ddr.store(0);
    }
};

inline CopyCounter g_memcpy_counter;
inline CopyCounter g_memset_counter;
