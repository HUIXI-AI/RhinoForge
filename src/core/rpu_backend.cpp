// Runtime controls, dispatcher registration, and Python bindings.
#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>

#include <torch/extension.h>

#include <ATen/native/TensorShape.h>
#include <ATen/Parallel.h>
#include <ATen/ops/view.h>
#include <c10/util/Half.h>
#include <pybind11/pybind11.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "rpu_ops.h"
#include "rpu_tensor_ops.h"
#include "graph/execution_coordinator.h"

#include "rpu_runtime_state.h"  // Shared runtime state and accessors.

// Forward declaration of the graph-aware CPU fallback dispatch entry (实现在
// src/graph/custom_cpu_fallback_dispatch.cpp)。不直接 include graph_runtime.h
// 到 rpu_backend.cpp,避免 c10::StorageImpl::decref_pyobject 等 inline body
// 在本 TU 实例化 — 跟 build 时 link 的 torch lib (miniconda3 python3.13 路径)
// 不匹配,会触发 undefined symbol。c10::OperatorHandle 跟 torch::jit::Stack 在
// 上面的 <torch/extension.h> / <ATen/...> 已经被 include,这里直接用 ::完整名。
void dispatch_graph_aware_cpu_fallback(const c10::OperatorHandle& op,
                                       torch::jit::Stack* stack);
// spm_alloc_reset_temporary 的 graph-aware 入口。RECORDING 期
// mark_non_replayable + 仍执行 host reset;REPLAYING 期
// TORCH_CHECK fail;PASSTHROUGH/BUILT 走原 eager 行为。实现在
// src/graph/graph_spm_reset_guard.cpp,同样为避免 graph_runtime.h 污染本 TU。
void graph_aware_spm_alloc_reset_temporary();
void process_guarded_spm_alloc_init();
void process_guarded_spm_alloc_reset_all();
// PASSTHROUGH direct-kernel queues live in graph_runtime_execute.cpp and must
// release their DDR command buffers before RpuDdrShutdown().
void graph_clear_passthrough_kernel_queues();
#include "rpu_spm_allocator.h"
#include "rpu_caching_allocator.h"
#include "graph/graph_pybind.h"  // Graph / GraphCache / GraphSignature bindings
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <atomic>
#include <filesystem>
#include <mutex>

using namespace at;

namespace {

struct LknBatchConfig {
    int64_t max_entries;
    int64_t kd_buf_mb;
    int64_t instr_buf_mb;
};

int64_t parse_lkn_env(const char* name, int64_t default_value,
                      int64_t max_value) {
    const char* env = std::getenv(name);
    if (!env || !*env) return default_value;
    char* end = nullptr;
    const unsigned long value = std::strtoul(env, &end, 10);
    if (end == env || value == 0) return default_value;
    return std::min<int64_t>(value, max_value);
}

LknBatchConfig read_lkn_batch_config() {
    return {
        parse_lkn_env("LKN_MAX_BATCH_ENTRIES", 65'536, 1 << 22),
        parse_lkn_env("LKN_KD_BUF_MB", 8, 256),
        parse_lkn_env("LKN_INSTR_BUF_MB", 64, 1'024),
    };
}

// The SDK BufferPool is also constructed while this extension is loaded. Keep
// the same-time snapshot so callers cannot mistake post-import env mutations
// for a resized pool.
const LknBatchConfig kLknBatchConfigAtLoad = read_lkn_batch_config();

std::vector<int64_t> rpu_get_lkn_batch_config() {
    const LknBatchConfig current = read_lkn_batch_config();
    return {
        kLknBatchConfigAtLoad.max_entries,
        kLknBatchConfigAtLoad.kd_buf_mb,
        kLknBatchConfigAtLoad.instr_buf_mb,
        current.max_entries,
        current.kd_buf_mb,
        current.instr_buf_mb,
    };
}

}  // namespace

// =============================================================================
// Global Debug and Profile Switches - 默认关闭
// =============================================================================

// Strict env parse for RPU_LOG_LEVEL: must be a complete integer in [0,5];
// invalid value falls back to INFO(3) with a one-time stderr warn. Plain
// `atoi` would silently treat "xyz" as 0 (SILENT) and discard a configuration
// error — we want loud-fail behavior at .so load time.
static int parse_log_level_env_or_default() {
    const char* env = std::getenv("RPU_LOG_LEVEL");
    if (!env || !*env) return 3;  // default INFO
    char* end = nullptr;
    long v = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || v < 0 || v > 5) {
        std::cerr << "[WARN] RPU_LOG_LEVEL=\"" << env
                  << "\" invalid (expected 0..5); falling back to INFO(3)\n";
        return 3;
    }
    return static_cast<int>(v);
}
std::atomic<int> g_rpu_debug_level{parse_log_level_env_or_default()};
bool g_rpu_profile_enabled = false; // 控制性能分析输出
bool g_rpu_caching_allocator_enabled = false; // 控制是否使用 caching allocator
bool g_rpu_ddr_flush_enabled = false; // intra-RPU flush perf knob (默认关闭)
bool g_rpu_ddr_flush_force_enabled = true; // CPU<->RPU boundary flush (默认开启)

// Thread-local profile data
#if RPU_PROFILE_ENABLED
thread_local KernelProfileData g_last_kernel_profile;
thread_local ProfileAccumulator g_profile_binary_sameshape;
thread_local ProfileAccumulator g_profile_binary_scalar;
thread_local ProfileAccumulator g_profile_binary_Nx1_NxC;
thread_local ProfileAccumulator g_profile_binary_1xC_NxC;
thread_local ProfileAccumulator g_profile_linear;
thread_local ProfileAccumulator g_profile_rmsnorm;
thread_local ProfileAccumulator g_profile_sdpa;
#endif

void rpu_set_debug_level(int level) {
    TORCH_CHECK(level >= 0 && level <= 5,
                "rpu_set_debug_level: level must be 0..5, got ", level);
    g_rpu_debug_level.store(level, std::memory_order_relaxed);
}
int rpu_get_debug_level() {
    return g_rpu_debug_level.load(std::memory_order_relaxed);
}
// BC mapping: set_debug(False) selects ERROR(1), below fallback logs at WARN(2);
// set_debug(True) selects DEBUG(4).
void rpu_set_debug(bool enabled) { rpu_set_debug_level(enabled ? 4 : 1); }
bool rpu_get_debug() { return rpu_get_debug_level() >= 4; }
void rpu_set_profile(bool enabled) { g_rpu_profile_enabled = enabled; }
bool rpu_get_profile() { return g_rpu_profile_enabled; }
void rpu_set_caching_allocator(bool enabled) { g_rpu_caching_allocator_enabled = enabled; }
bool rpu_get_caching_allocator() { return g_rpu_caching_allocator_enabled; }
pybind11::dict rpu_stat_to_dict(const rpu::Stat& stat) {
  pybind11::dict d;
  d["current"] = stat.current;
  d["peak"] = stat.peak;
  d["allocated"] = stat.allocated;
  d["freed"] = stat.freed;
  return d;
}
pybind11::dict rpu_get_memory_stats_py() {
  const rpu::DeviceStats stats = rpu::get_memory_stats();
  pybind11::dict d;
  d["allocation"] = rpu_stat_to_dict(stats.allocation);
  d["reserved_bytes"] = rpu_stat_to_dict(stats.reserved_bytes);
  d["allocated_bytes"] = rpu_stat_to_dict(stats.allocated_bytes);
  d["active_bytes"] = rpu_stat_to_dict(stats.active_bytes);
  d["segment"] = rpu_stat_to_dict(stats.segment);
  d["inactive_split_bytes"] = rpu_stat_to_dict(stats.inactive_split_bytes);
  d["num_device_alloc"] = stats.num_device_alloc;
  d["num_device_free"] = stats.num_device_free;
  d["num_ooms"] = stats.num_ooms;
  d["num_alloc_retries"] = stats.num_alloc_retries;
  d["largest_available_block"] = static_cast<int64_t>(
      rpu::get_caching_allocator().getLargestAvailableBlock());
  d["total_allocated_memory"] = static_cast<int64_t>(
      rpu::get_caching_allocator().getTotalAllocatedMemory());
  d["total_cached_memory"] = static_cast<int64_t>(
      rpu::get_caching_allocator().getTotalCachedMemory());
  return d;
}
void rpu_set_ddr_flush(bool enabled) { g_rpu_ddr_flush_enabled = enabled; }
bool rpu_get_ddr_flush() { return g_rpu_ddr_flush_enabled; }
void rpu_set_ddr_flush_force(bool enabled) { g_rpu_ddr_flush_force_enabled = enabled; }
bool rpu_get_ddr_flush_force() { return g_rpu_ddr_flush_force_enabled; }

namespace {

#if defined(__aarch64__)
float32x4_t load_half4(const c10::Half* values) {
  static_assert(sizeof(c10::Half) == sizeof(uint16_t));
  uint16_t bits[4];
  std::memcpy(bits, values, sizeof(bits));
  return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(bits)));
}
#endif

int64_t argmax_row_fp16(const c10::Half* row, int64_t cols) {
#if defined(__aarch64__)
  const c10::Half* values = row;
  if (cols >= 8) {
    int64_t offset = 0;
    float32x4_t max0 = vdupq_n_f32(-INFINITY);
    float32x4_t max1 = max0;
    for (; offset + 8 <= cols; offset += 8) {
      const float32x4_t value0 = load_half4(values + offset);
      const float32x4_t value1 = load_half4(values + offset + 4);
      const uint32x4_t ordered0 = vceqq_f32(value0, value0);
      const uint32x4_t ordered1 = vceqq_f32(value1, value1);
      if (vminvq_u32(vandq_u32(ordered0, ordered1)) == 0) {
        for (int64_t lane = offset; lane < offset + 8; ++lane) {
          if (std::isnan(static_cast<float>(values[lane]))) return lane;
        }
      }
      max0 = vmaxq_f32(max0, value0);
      max1 = vmaxq_f32(max1, value1);
    }
    float best = vmaxvq_f32(vmaxq_f32(max0, max1));
    for (int64_t lane = offset; lane < cols; ++lane) {
      const float value = static_cast<float>(values[lane]);
      if (std::isnan(value)) return lane;
      if (value > best) best = value;
    }

    const float32x4_t best4 = vdupq_n_f32(best);
    offset = 0;
    for (; offset + 8 <= cols; offset += 8) {
      const uint32x4_t equal0 = vceqq_f32(
          load_half4(values + offset), best4);
      const uint32x4_t equal1 = vceqq_f32(
          load_half4(values + offset + 4), best4);
      if (vmaxvq_u32(vorrq_u32(equal0, equal1)) != 0) {
        for (int64_t lane = offset; lane < offset + 8; ++lane) {
          if (static_cast<float>(values[lane]) == best) return lane;
        }
      }
    }
    for (; offset < cols; ++offset) {
      if (static_cast<float>(values[offset]) == best) return offset;
    }
  }
#endif

  int64_t best_index = 0;
  float best_value = static_cast<float>(row[0]);
  if (std::isnan(best_value)) return 0;
  for (int64_t col = 1; col < cols; ++col) {
    const float value = static_cast<float>(row[col]);
    if (std::isnan(value)) return col;
    if (value > best_value) {
      best_value = value;
      best_index = col;
    }
  }
  return best_index;
}

at::Tensor host_top1_impl(
    const at::Tensor& logits, bool copy_result_to_rpu) {
  const bool on_rpu =
      logits.device().type() == c10::DeviceType::PrivateUse1;
  TORCH_CHECK(on_rpu || logits.device().is_cpu(),
              "host top1: logits must be on CPU or RPU, got ",
              logits.device());
  TORCH_CHECK(logits.scalar_type() == at::kHalf,
              "host top1: logits must be fp16, got ",
              logits.scalar_type());
  TORCH_CHECK(logits.dim() >= 1,
              "host top1: logits must have at least one dim");
  TORCH_CHECK(logits.is_contiguous(),
              "host top1: logits must be contiguous");

  const int64_t vocab = logits.size(-1);
  TORCH_CHECK(vocab > 0, "host top1: empty last dimension");
  TORCH_CHECK(logits.numel() % vocab == 0,
              "host top1: numel must be divisible by vocab");
  const int64_t rows = logits.numel() / vocab;

  at::Tensor cpu_logits = on_rpu
      ? rpu_to_cpu_zerocopy(logits, /*flush=*/true)
      : logits;
  at::Tensor out_cpu = at::empty({rows}, at::TensorOptions()
      .dtype(at::kLong)
      .device(at::kCPU));

  const c10::Half* in = cpu_logits.data_ptr<c10::Half>();
  int64_t* out = out_cpu.data_ptr<int64_t>();
  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      out[row] = argmax_row_fp16(in + row * vocab, vocab);
    }
  });

  std::vector<int64_t> out_sizes = logits.sizes().vec();
  out_sizes.pop_back();
  at::Tensor result = out_cpu.reshape(out_sizes);
  return copy_result_to_rpu ? result.to(logits.device()) : result;
}

}  // namespace

at::Tensor rpu_lm_head_logits_top1(const at::Tensor& logits) {
  TORCH_CHECK(logits.device().type() == c10::DeviceType::PrivateUse1,
              "lm_head_logits_top1: logits must be on RPU device");
  return host_top1_impl(logits, /*copy_result_to_rpu=*/true);
}

at::Tensor rpu_argmax_lastdim_host(const at::Tensor& logits) {
  TORCH_CHECK(logits.dim() == 2,
              "argmax_lastdim_host expects 2-D [rows, cols] logits, got ",
              logits.dim(), "-D");
  return host_top1_impl(logits, /*copy_result_to_rpu=*/false);
}

// =============================================================================
// HW perf trace global config — see rpu_profile.h for design notes.
// =============================================================================
static std::mutex g_hw_perf_mu;
static HwPerfConfig g_hw_perf_cfg{};
static std::atomic<int64_t>  g_hw_perf_dumps_left{0};
static std::atomic<uint64_t> g_hw_perf_dump_seq{0};

HwPerfConfig rpu_hw_perf_snapshot() {
    std::lock_guard<std::mutex> lock(g_hw_perf_mu);
    return g_hw_perf_cfg;
}

void rpu_set_hw_perf_trace(bool enabled,
                           const std::string& output_dir,
                           int64_t max_dumps) {
    {
        std::lock_guard<std::mutex> lock(g_hw_perf_mu);
        g_hw_perf_cfg.enabled = enabled;
        g_hw_perf_cfg.output_dir = output_dir;
        g_hw_perf_cfg.max_dumps = max_dumps;
    }
    // Reset dump budget so each setter call refreshes the per-process limit.
    g_hw_perf_dumps_left.store(max_dumps, std::memory_order_relaxed);

    // Best-effort recursive mkdir — if output_dir is missing, dump_hw_perf_chrome
    // would return rc=3 on every call. Use std::filesystem::create_directories
    // (recursive) so callers can pass nested paths like "/tmp/run_42/traces".
    // Failures (e.g. permission denied) are logged but non-fatal: the dump
    // site still reports its own rc != 0 with errno.
    if (enabled && !output_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(output_dir, ec);
        if (ec && log_at(2)) {
            std::cerr << "[HW_PERF] create_directories('" << output_dir
                      << "') failed: " << ec.message()
                      << " — dumps will fail with rc=3 until dir exists\n";
        }
    }
}

bool rpu_get_hw_perf_trace() {
    std::lock_guard<std::mutex> lock(g_hw_perf_mu);
    return g_hw_perf_cfg.enabled;
}

bool rpu_hw_perf_try_consume_dump() {
    int64_t old = g_hw_perf_dumps_left.load(std::memory_order_relaxed);
    while (old > 0) {
        if (g_hw_perf_dumps_left.compare_exchange_weak(
                old, old - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

uint64_t rpu_hw_perf_next_dump_seq() {
    return g_hw_perf_dump_seq.fetch_add(1, std::memory_order_relaxed);
}

// Reset all profile accumulators
void rpu_reset_profile_accumulators() {
#if RPU_PROFILE_ENABLED
    g_profile_binary_sameshape.reset();
    g_profile_binary_scalar.reset();
    g_profile_binary_Nx1_NxC.reset();
    g_profile_binary_1xC_NxC.reset();
    g_profile_linear.reset();
    g_profile_rmsnorm.reset();
    g_profile_sdpa.reset();
    g_last_kernel_profile = KernelProfileData();
#endif
}

enum class RpuBackendLifecycleState {
    kCold,
    kReady,
    kShutdown,
};

static std::mutex g_rpu_backend_lifecycle_mu;
static RpuBackendLifecycleState g_rpu_backend_lifecycle =
    RpuBackendLifecycleState::kCold;

static bool rpu_runtime_is_available() {
    std::lock_guard<std::mutex> lock(g_rpu_backend_lifecycle_mu);
    return g_rpu_backend_lifecycle == RpuBackendLifecycleState::kReady &&
           rpu_device_nodes_accessible();
}

// The pybind composition root owns process-runtime startup. Keep allocator and
// guard registration ahead of KernelCache construction: registrations are
// intentionally process-lifetime, while the kernel/queue/DDR resources below
// have an explicit reverse-order shutdown path.
static void initialize_rpu_backend_runtime() {
    std::lock_guard<std::mutex> lock(g_rpu_backend_lifecycle_mu);
    if (g_rpu_backend_lifecycle == RpuBackendLifecycleState::kReady) {
        return;
    }
    TORCH_CHECK(
        g_rpu_backend_lifecycle == RpuBackendLifecycleState::kCold,
        "RPU backend runtime cannot be reinitialized after shutdown");

    initialize_rpu_tensor_runtime();
    KernelCache::instance().initialize();
    g_rpu_backend_lifecycle = RpuBackendLifecycleState::kReady;
}

// Shutdown: 释放所有缓存的资源
// 必须在程序退出前、device runtime 还活着的时候调用
void rpu_shutdown() {
    std::lock_guard<std::mutex> lock(g_rpu_backend_lifecycle_mu);
    if (g_rpu_backend_lifecycle != RpuBackendLifecycleState::kReady) {
        return;
    }
    RpuExecutionCleanupGuard cleanup("rpu_shutdown", /*shutting_down=*/true);
    // Mark first so a repeated/manual+atexit call is idempotent even if one of
    // the best-effort cleanup operations below reports an exception.
    g_rpu_backend_lifecycle = RpuBackendLifecycleState::kShutdown;

    // Stop cached allocations immediately. This must not depend on the current
    // feature toggle: callers may disable the allocator after creating cached
    // tensors, and those tensors must never observe DDR after teardown starts.
    rpu::RPUCachingAllocator::get().mark_shutdown();

    // 0. SpmAllocator 不需要显式清理（析构时跳过，硬件由下面的 clear 处理）

    // 1. 清空 SPM buffer pool
    SPM_POOL.clear();

    // 2. 清空 caching allocator
    rpu::empty_cache();

    // 3. Adapter-owned graph caches are released by Python; Queue_t 由 QueueCache 全局
    //    缓存(进程内最多 8 个,按 core_num 分桶)。RpuKernelGraph 实例由 Python
    //    管理,持有的 Queue_t 在 Python GC 时随 instance 析构释放,跟 shutdown
    //    sequence 解耦。

    // 4. 清空 Kernel cache (释放 Program_t/Kernel_t)
    KernelCache::instance().clear();

    // 5. 清空 immediate DMA 与 PASSTHROUGH kernel queue caches
    QueueCache::instance().clear();
    graph_clear_passthrough_kernel_queues();

    // 6. DDR teardown must follow every subsystem that owns DDR resources.
    rhino_lkn::RpuDdrShutdown();

    if (log_at(3)) {
        std::cout << "[RPU] Shutdown complete - all cached resources released" << std::endl;
    }
}

// KernelCache remains in this translation unit because its process-wide
// singleton and Rhino program objects have SDK-sensitive initialization and
// teardown ordering. The implementation lives in a separate source fragment
// so the composition file stays navigable without changing that ABI/lifetime.
#include "rpu_kernel_cache.inc"
#include "rpu_tensor_ops.inc"

// 尝试 view；如果失败则 fallback 到 contiguous + view
static Tensor reshape_fallback(const Tensor& self, IntArrayRef sizes) {
    // 1. Fast path: try view (does NOT allocate)
    try {
        return self.view(sizes);
    } catch (const c10::Error& e) {
        // view 失败 -> fallback path
    }

    // 2. Slow fallback: make contiguous + view
    Tensor c = self.contiguous();
    return c.view(sizes);
}

// CPU fallback profiling (for auto fallback)
static thread_local std::unordered_map<std::string, int> cpu_fallback_counts;
static thread_local int total_cpu_fallback_count = 0;

// Manual fallback profiling
static thread_local std::unordered_map<std::string, int> manual_fallback_counts;
static thread_local std::unordered_map<std::string, double> manual_fallback_times;
static thread_local int total_manual_fallback_count = 0;

#define MANUAL_FALLBACK_LOG(op_name) \
  do { \
    manual_fallback_counts[op_name]++; \
    total_manual_fallback_count++; \
    if (log_at(2) && (manual_fallback_counts[op_name] <= 3 || manual_fallback_counts[op_name] % 100 == 0)) { \
      std::cout << "[MANUAL_FALLBACK] " << op_name << " (count=" << manual_fallback_counts[op_name] << ")" << std::endl; \
    } \
  } while(0)

// =============================================================================
// custom_cpu_fallback — graph-aware HostCallback dispatcher hook
// =============================================================================
// thin wrapper:log + 转发到 dispatch_graph_aware_cpu_fallback (实现在
// src/graph/custom_cpu_fallback_dispatch.cpp,fork graph state 4 个分支)。
// 这样 rpu_backend.cpp 不需要 include graph_runtime.h,避免 c10::StorageImpl
// 等 inline 在本 TU 实例化触发 undefined symbol（构建与运行时 torch 库不一致时
// 会出现）；只有 graph/*.cpp 里 include 才不会有这个问题,因为 link 时 graph/*.o
// 已经 settled)。
void custom_cpu_fallback(const c10::OperatorHandle &op,
                         torch::jit::Stack *stack) {
  // Log which op is falling back to CPU
  std::string op_name = op.schema().name();
  cpu_fallback_counts[op_name]++;
  total_cpu_fallback_count++;

  // Print first 3 occurrences of each op, then every 100th (only when debug enabled)
  if (log_at(2) && (cpu_fallback_counts[op_name] <= 3 || cpu_fallback_counts[op_name] % 100 == 0)) {
    std::cout << "[CPU_FALLBACK #" << total_cpu_fallback_count << "] "
              << op_name << " (count=" << cpu_fallback_counts[op_name] << ")" << std::endl;
  }

  // 走 graph-aware dispatch (PASSTHROUGH/BUILT eager / RECORDING tier 分派 /
  // REPLAYING capture_*_replay_args)。所有 graph state fork 逻辑跟 HostCallback /
  // Tier3 RAII 都在 graph/custom_cpu_fallback_dispatch.cpp 里。
  dispatch_graph_aware_cpu_fallback(op, stack);
}

// ------------------------ Registration ------------------------

Tensor isfinite_fallback_to_cpu(const Tensor &self) {
  // 零拷贝优化
  auto cpu_self = rpu_to_cpu_zerocopy(self);
  auto result = at::empty(self.sizes(), self.options().dtype(at::kBool));
  auto cpu_result = rpu_to_cpu_zerocopy(result);
  // isfinite 没有 _out 版本，需要手动复制
  auto tmp = at::isfinite(cpu_self);
  cpu_result.copy_(tmp);
  rpu_ddr_flush_force(result.data_ptr());
  return result;
}

// ---------------- 注册 allocator 与 operators ----------------

// ============================================================================
// 融合执行API: QKV Attention (SPM版本)
// ============================================================================
// 将QKV推理流程全程保留在SPM上，避免DDR与SPM之间的数据搬运开销
// 数据流: hidden_states(DDR) -> Linear(QKV) -> SPM -> RMSNorm -> ROPE -> SDPA -> output(DDR)
// 详细实现见 rpu_fused_attention.cpp
// ============================================================================
// 声明：实际实现在 rpu_fused_attention.cpp
// at::Tensor rpu_fused_qkv_attention(...) - 见 rpu_ops.h

// ================== Custom RPU Ops Namespace ==================
// Define custom ops under rpu:: namespace for direct dispatch (bypass pybind11 overhead)

// Multi-core Conv2d accepting pre-converted NHWC tensors (no layout conversion overhead)
static at::Tensor rpu_conv2d_mc_nhwc(
    const at::Tensor& input_nhwc,   // [batch, inh, inw, cin_padded] NHWC on RPU
    const at::Tensor& weight_nhwc,  // [cout, kh, kw, cin_padded] NHWC on RPU
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t core_num)
{
    int64_t N = input_nhwc.size(0);
    int64_t H = input_nhwc.size(1);
    int64_t W = input_nhwc.size(2);
    int64_t Cin = input_nhwc.size(3);
    int64_t Cout = weight_nhwc.size(0);
    int64_t Kh = weight_nhwc.size(1);
    int64_t Kw = weight_nhwc.size(2);
    int64_t sh = stride[0], sw = stride.size() > 1 ? stride[1] : stride[0];
    int64_t ph = padding[0], pw = padding.size() > 1 ? padding[1] : padding[0];
    int64_t dh = dilation[0], dw = dilation.size() > 1 ? dilation[1] : dilation[0];

    int kh_eff = Kh + (Kh - 1) * (dh - 1);
    int kw_eff = Kw + (Kw - 1) * (dw - 1);
    int outh = (H + 2 * ph - kh_eff) / sh + 1;
    int outw = (W + 2 * pw - kw_eff) / sw + 1;

    // Compute multi-core output channel layout
    int cin_v16 = CeilDiv((int)Cin, 16);
    int cpc = Cout / core_num;
    int cpc_v16 = CeilDiv(cpc, 16);
    int num_flt_v16 = cpc_v16 * core_num;
    int num_flt_v16_elems = num_flt_v16 * 16;

    bool has_bias = bias_opt.has_value() && bias_opt->defined();

    // Bias padding to num_flt_v16 width
    at::Tensor bias_padded;
    c10::Half* bias_ptr = nullptr;
    if (has_bias) {
        if (num_flt_v16_elems != Cout) {
            bias_padded = at::zeros({num_flt_v16_elems}, bias_opt->options());
            bias_padded.slice(0, 0, Cout).copy_(*bias_opt);
        } else {
            bias_padded = bias_opt->contiguous();
        }
        bias_ptr = bias_padded.data_ptr<c10::Half>();
    }

    // Output NHWC tensor
    at::Tensor output_nhwc = at::empty(
        {N, outh, outw, num_flt_v16_elems}, input_nhwc.options());

    // Input/weight already NHWC contiguous on RPU — just flush
    rpu_ddr_flush(const_cast<void*>(static_cast<const void*>(input_nhwc.data_ptr())));
    rpu_ddr_flush(const_cast<void*>(static_cast<const void*>(weight_nhwc.data_ptr())));
    if (has_bias) rpu_ddr_flush(bias_ptr);

    rpu_launch_conv2d_multicore_ddr_kernel(
        input_nhwc.data_ptr<c10::Half>(),
        weight_nhwc.data_ptr<c10::Half>(),
        bias_ptr,
        output_nhwc.data_ptr<c10::Half>(),
        N, Cin, H, W, Cout, Kh, Kw,
        ph, ph, pw, pw, sh, sw, dh, dw,
        has_bias, core_num);

    rpu_ddr_flush(output_nhwc.data_ptr());

    // Return NHWC directly — caller handles NCHW conversion
    // Slice valid channels if padded
    if (num_flt_v16_elems != Cout) {
        return output_nhwc.slice(3, 0, Cout).contiguous();
    }
    return output_nhwc;
}

// Dispatcher registration remains in this translation unit; the include is a
// source-organization boundary, not a separately compiled component.
#include "rpu_dispatch_registrations.inc"

// CPython bindings are likewise part of this composition translation unit.
#include "rpu_pybind.inc"
