// rpu_profile.h — Profile data structures, macros, global debug switches
#pragma once

#include <atomic>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <chrono>
#include <string>

// Forward declaration for DDR flush
namespace rhino_lkn {
    void RpuDdrFlush(void* ptr);
    void RpuDdrFlush(void* ptr, size_t size);   // sized overload: flush only [ptr, ptr+size)
}

// =============================================================================
// Compile-time Profile Switch (set to 0 to completely remove profile code)
// =============================================================================
#define RPU_PROFILE_ENABLED 0
#define RPU_PROFILE_INTERVAL 20       // 每50次打印一次 (linear, binary, etc.)
#define RPU_PROFILE_INTERVAL_SDPA 10   // SDPA 打印频率 (每次都打印)

// =============================================================================
// Global Debug and Profile Switches
// =============================================================================
// Unified 6-level debug verbosity.
// 0 SILENT / 1 ERROR / 2 WARN / 3 INFO (default) / 4 DEBUG / 5 TRACE
// Init from RPU_LOG_LEVEL env at .so load (parse_log_level_env_or_default).
// log_at(N) is the relaxed-load fast gate.
extern std::atomic<int> g_rpu_debug_level;
inline bool log_at(int level) {
    return g_rpu_debug_level.load(std::memory_order_relaxed) >= level;
}
extern bool g_rpu_profile_enabled;  // 控制性能分析输出 (kernel timing 等)
extern bool g_rpu_ddr_flush_enabled; // 控制是否执行 intra-RPU DDR flush (默认关闭,
                                     // 见 rpu_backend.cpp 定义处; 这是性能开关,
                                     // 不是 CPU<->RPU 边界正确性开关)
extern bool g_rpu_ddr_flush_force_enabled; // 控制 CPU<->RPU 边界 force flush (默认开启，保证正确性)

// =============================================================================
// Profile Data Structures
// =============================================================================
#if RPU_PROFILE_ENABLED

struct WrapperProfileData {
    double preprocess_ms = 0;   // 前处理 (device check, dtype check, broadcast check)
    double alloc_ms = 0;        // 分配输出 tensor
    double contiguous_ms = 0;   // contiguous 调用
    double kernel_ms = 0;       // kernel 执行总时间
    double postprocess_ms = 0;  // 后处理
    double total_ms = 0;        // 总时间
};

struct KernelProfileData {
    double preprocess_ms = 0;   // 前处理 (get kernel, reset regs)
    double flush_ms = 0;        // DDR flush
    double set_regs_ms = 0;     // set_regs
    double enqueue_ms = 0;      // enqueu_kernel
    double postflush_ms = 0;    // 后处理 flush
    double total_ms = 0;        // 总时间
};

// Thread-local profile accumulators
struct ProfileAccumulator {
    int call_count = 0;
    WrapperProfileData wrapper_sum;
    KernelProfileData kernel_sum;

    void add_wrapper(const WrapperProfileData& d) {
        wrapper_sum.preprocess_ms += d.preprocess_ms;
        wrapper_sum.alloc_ms += d.alloc_ms;
        wrapper_sum.contiguous_ms += d.contiguous_ms;
        wrapper_sum.kernel_ms += d.kernel_ms;
        wrapper_sum.postprocess_ms += d.postprocess_ms;
        wrapper_sum.total_ms += d.total_ms;
    }

    void add_kernel(const KernelProfileData& d) {
        kernel_sum.preprocess_ms += d.preprocess_ms;
        kernel_sum.flush_ms += d.flush_ms;
        kernel_sum.set_regs_ms += d.set_regs_ms;
        kernel_sum.enqueue_ms += d.enqueue_ms;
        kernel_sum.postflush_ms += d.postflush_ms;
        kernel_sum.total_ms += d.total_ms;
    }

    void reset() {
        call_count = 0;
        wrapper_sum = WrapperProfileData();
        kernel_sum = KernelProfileData();
    }
};

// Thread-local storage for passing kernel profile data to wrapper
extern thread_local KernelProfileData g_last_kernel_profile;

// Profile accumulators for different ops
extern thread_local ProfileAccumulator g_profile_binary_sameshape;
extern thread_local ProfileAccumulator g_profile_binary_scalar;
extern thread_local ProfileAccumulator g_profile_binary_Nx1_NxC;
extern thread_local ProfileAccumulator g_profile_binary_1xC_NxC;
extern thread_local ProfileAccumulator g_profile_linear;
extern thread_local ProfileAccumulator g_profile_rmsnorm;
extern thread_local ProfileAccumulator g_profile_sdpa;

// Helper macros for timing
#define PROFILE_START() auto _prof_start = std::chrono::high_resolution_clock::now()
#define PROFILE_CHECKPOINT(var) do { \
    auto _now = std::chrono::high_resolution_clock::now(); \
    var = std::chrono::duration<double, std::milli>(_now - _prof_start).count(); \
    _prof_start = _now; \
} while(0)
#define PROFILE_ELAPSED_MS(start) \
    std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count()

// Print profile helper
inline void print_profile(const char* op_name, int count,
                          const WrapperProfileData& w, const KernelProfileData& k) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[PROFILE " << op_name << " #" << count << "] "
              << "wrapper(pre=" << w.preprocess_ms
              << " alloc=" << w.alloc_ms
              << " contig=" << w.contiguous_ms
              << " kern=" << w.kernel_ms
              << " post=" << w.postprocess_ms
              << " tot=" << w.total_ms << "ms) "
              << "kernel(pre=" << k.preprocess_ms
              << " flush=" << k.flush_ms
              << " regs=" << k.set_regs_ms
              << " enq=" << k.enqueue_ms
              << " post=" << k.postflush_ms
              << " tot=" << k.total_ms << "ms)"
              << std::endl;
}

// Print average profile
inline void print_avg_profile(const char* op_name, const ProfileAccumulator& acc) {
    if (acc.call_count == 0) return;
    double n = acc.call_count;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[PROFILE " << op_name << " avg/" << acc.call_count << "] "
              << "wrapper(pre=" << acc.wrapper_sum.preprocess_ms / n
              << " alloc=" << acc.wrapper_sum.alloc_ms / n
              << " contig=" << acc.wrapper_sum.contiguous_ms / n
              << " kern=" << acc.wrapper_sum.kernel_ms / n
              << " post=" << acc.wrapper_sum.postprocess_ms / n
              << " tot=" << acc.wrapper_sum.total_ms / n << "ms) "
              << "kernel(pre=" << acc.kernel_sum.preprocess_ms / n
              << " flush=" << acc.kernel_sum.flush_ms / n
              << " regs=" << acc.kernel_sum.set_regs_ms / n
              << " enq=" << acc.kernel_sum.enqueue_ms / n
              << " post=" << acc.kernel_sum.postflush_ms / n
              << " tot=" << acc.kernel_sum.total_ms / n << "ms)"
              << std::endl;
}

#define PROFILE_RECORD_AND_PRINT(op_name, acc, w_data, k_data) do { \
    acc.call_count++; \
    acc.add_wrapper(w_data); \
    acc.add_kernel(k_data); \
    if (acc.call_count % RPU_PROFILE_INTERVAL == 0) { \
        print_avg_profile(op_name, acc); \
    } \
} while(0)

// SDPA-specific profile macro with separate interval
#define PROFILE_RECORD_AND_PRINT_SDPA(op_name, acc, w_data, k_data) do { \
    acc.call_count++; \
    acc.add_wrapper(w_data); \
    acc.add_kernel(k_data); \
    if (acc.call_count % RPU_PROFILE_INTERVAL_SDPA == 0) { \
        print_avg_profile(op_name, acc); \
    } \
} while(0)

#else  // RPU_PROFILE_ENABLED == 0

#define PROFILE_START() ((void)0)
#define PROFILE_CHECKPOINT(var) ((void)0)
#define PROFILE_ELAPSED_MS(start) (0.0)
#define PROFILE_RECORD_AND_PRINT(op_name, acc, w_data, k_data) ((void)0)
#define PROFILE_RECORD_AND_PRINT_SDPA(op_name, acc, w_data, k_data) ((void)0)

struct WrapperProfileData {};
struct KernelProfileData {};
struct ProfileAccumulator { int call_count = 0; };

#endif  // RPU_PROFILE_ENABLED

// Helper functions to set/get debug and profile flags
void rpu_set_debug_level(int level);   // canonical
int  rpu_get_debug_level();
void rpu_set_debug(bool enabled);      // BC shim: true→4 DEBUG, false→1 ERROR
bool rpu_get_debug();                  // BC shim: level >= 4
void rpu_set_profile(bool enabled);
bool rpu_get_profile();
void rpu_set_ddr_flush(bool enabled);
bool rpu_get_ddr_flush();
void rpu_set_ddr_flush_force(bool enabled);
bool rpu_get_ddr_flush_force();

// Reset all profile accumulators (call before profiling runs)
void rpu_reset_profile_accumulators();

// =============================================================================
// HW perf trace — RPU cycle-precision kernel/DMA timestamp dump (Chrome JSON).
//
// Snapshot pattern: setter takes mutex and stamps all fields + bumps dump
// budget; hot-path callers read via rpu_hw_perf_snapshot() (one mutex op per
// graph forward, not per kernel).
//
// Toggle semantics: the adapter's graph signature/admission policy must keep
// trace-enabled and trace-disabled captures distinct. The setter does not
// eagerly clear adapter-owned GraphCache instances.
//
// File naming:
// rpu_hwperf_pid<P>_tid<TID>_seq<S>_<OP_ID>_<build|replay>_seg<N>.json
// in `output_dir`. Multiple threads / processes never collide.
// =============================================================================
struct HwPerfConfig {
    bool enabled = false;
    std::string output_dir = ".";
    int64_t max_dumps = 32;
};

HwPerfConfig rpu_hw_perf_snapshot();
void rpu_set_hw_perf_trace(bool enabled,
                           const std::string& output_dir,
                           int64_t max_dumps);
bool rpu_get_hw_perf_trace();

// Process-global atomic dump budget. Try to claim one dump slot — returns
// true iff a budget slot was available (and consumed). Decremented monotonically
// per setter call.
bool rpu_hw_perf_try_consume_dump();

// Process-global atomic dump sequence — unique per JSON file. Monotonic across
// threads, never resets within a process.
uint64_t rpu_hw_perf_next_dump_seq();

// Conditional DDR flush macro - only executes if g_rpu_ddr_flush_enabled is true.
// Use this for intra-RPU kernel-to-kernel sync points that are redundant when the
// data never leaves DDR/SPM. set_ddr_flush(False) turns these into no-ops for perf.
#define rpu_ddr_flush(ptr) do { \
    if (g_rpu_ddr_flush_enabled) { \
        rhino_lkn::RpuDdrFlush((ptr)); \
    } \
} while(0)

// Boundary DDR flush - CPU<->RPU 边界一致性 flush。默认开启 (g_rpu_ddr_flush_force_enabled=true)
// 以保证正确性; 可通过 set_ddr_flush_force(False) 关闭以量化 flush 对性能的影响 (decode 场景
// 可能显著)，但关闭后 CPU cache 脏行不会回写共享 DDR，下一 RPU kernel 可能读到旧值。
// 与 rpu_ddr_flush 的区别：此宏独立开关，不受 set_ddr_flush 影响,绕过 graph capture。
#define rpu_ddr_flush_force(ptr) do { \
    if (g_rpu_ddr_flush_force_enabled) { \
        rhino_lkn::RpuDdrFlush( \
            const_cast<void*>(static_cast<const void*>(ptr))); \
    } \
} while(0)

// Sized boundary flush — flush only [ptr, ptr+nbytes) instead of the whole
// allocation. Same gating as rpu_ddr_flush_force; use when only a known sub-range
// of a large stable buffer was written (e.g. M-RoPE keepalive slices: flush the
// seq_len rows actually refreshed, not the full 8192-row table).
#define rpu_ddr_flush_force_sized(ptr, nbytes) do { \
    if (g_rpu_ddr_flush_force_enabled) { \
        rhino_lkn::RpuDdrFlush(const_cast<void*>(static_cast<const void*>(ptr)), (nbytes)); \
    } \
} while(0)

// =============================================================================
// Utility Macros
// =============================================================================
#define Align(x, y) (((x) + (y)-1) / (y) * (y))
#define CeilDiv(x, y) (((x) + (y)-1) / (y))
