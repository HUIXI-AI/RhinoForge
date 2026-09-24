// Runtime controls, dispatcher registration, and Python bindings.
#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>

#include <torch/extension.h>

#include <ATen/native/TensorShape.h>
#include <ATen/Parallel.h>
#include <ATen/ops/view.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/squeeze_native.h>
#include <c10/util/Half.h>
#include <pybind11/pybind11.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "rpu_ops.h"
#include "rpu_host_argmax.h"
#include "rpu_tensor_ops.h"
#include "graph/execution_coordinator.h"

#include "rpu_runtime_state.h"  // v5-07: g_chunk_size_override + g_debug_tensors + 9 accessors (D-06)

// P7.1a: forward-decl 的 graph-aware CPU fallback dispatch entry (实现在
// src/graph/custom_cpu_fallback_dispatch.cpp)。不直接 include graph_runtime.h
// 到 rpu_backend.cpp,避免 c10::StorageImpl::decref_pyobject 等 inline body
// 在本 TU 实例化 — 跟 build 时 link 的 torch lib (miniconda3 python3.13 路径)
// 不匹配,会触发 undefined symbol。c10::OperatorHandle 跟 torch::jit::Stack 在
// 上面的 <torch/extension.h> / <ATen/...> 已经被 include,这里直接用 ::完整名。
void dispatch_graph_aware_cpu_fallback(const c10::OperatorHandle& op,
                                       torch::jit::Stack* stack);
// 支线 S (docs/graph_rules.md §12.4):spm_alloc_reset_temporary graph-aware
// 入口。RECORDING 期 mark_non_replayable + 仍执行 host reset;REPLAYING 期
// TORCH_CHECK fail;PASSTHROUGH/BUILT 走原 eager 行为。实现在
// src/graph/graph_spm_reset_guard.cpp,同样为避免 graph_runtime.h 污染本 TU。
void graph_aware_spm_alloc_reset_temporary();
bool graph_aware_rpu_to_cpu_zerocopy_sync();
void process_guarded_spm_alloc_init();
void process_guarded_spm_alloc_reset_all();
// PASSTHROUGH direct-kernel queues live in graph_runtime_execute.cpp and must
// release their DDR command buffers before RpuDdrShutdown().
void graph_clear_passthrough_kernel_queues();
#include "rpu_spm_allocator.h"
#include "rpu_caching_allocator.h"
#include "rpu_allocator_policy.h"
#include "fused_model_base.h"    // v3::detail::prefer_balanced_chunk (chunk_helpers)
#include "graph/graph_pybind.h"  // P5: expose Graph / GraphCache / GraphSignature to Python
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <atomic>
#include <filesystem>
#include <unistd.h>
#include "rhino_launch_def.h"
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
    return {
        kLknBatchConfigAtLoad.max_entries,
        kLknBatchConfigAtLoad.kd_buf_mb,
        kLknBatchConfigAtLoad.instr_buf_mb,
        kLknBatchConfigAtLoad.max_entries,
        kLknBatchConfigAtLoad.kd_buf_mb,
        kLknBatchConfigAtLoad.instr_buf_mb,
    };
}

}  // namespace

RpuLknBatchConfigSnapshot rpu_lkn_batch_config_at_load() {
    return {
        kLknBatchConfigAtLoad.max_entries,
        kLknBatchConfigAtLoad.kd_buf_mb,
        kLknBatchConfigAtLoad.instr_buf_mb,
    };
}

RpuLknBatchConfigSnapshot rpu_lkn_batch_config_at_preflight() {
    // Mirror the pinned Launch SDK's lkn_max_batch_entries/lkn_env_buf_bytes,
    // including prefix parsing and unsigned byte-shift semantics. This is a
    // rejection check, never a way to resize an already bound runtime policy.
    const auto sdk_value = [](const char* name, size_t fallback,
                              size_t maximum, bool buffer_mb) -> int64_t {
        const char* raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0') return fallback;
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(raw, &end, 10);
        if (end == raw || parsed == 0) return fallback;
        if (buffer_mb) {
            const size_t bytes = static_cast<size_t>(parsed) << 20;
            return static_cast<int64_t>(std::min(bytes, maximum << 20) >> 20);
        }
        return static_cast<int64_t>(std::min(static_cast<size_t>(parsed), maximum));
    };
    return {
        sdk_value("LKN_MAX_BATCH_ENTRIES", 65'536, 1 << 22, false),
        sdk_value("LKN_KD_BUF_MB", 8, 256, true),
        sdk_value("LKN_INSTR_BUF_MB", 64, 1'024, true),
    };
}

static bool rpu_host_tensor_byte_ranges_overlap(
    const at::Tensor& first, const at::Tensor& second) {
  const size_t first_bytes = static_cast<size_t>(first.nbytes());
  const size_t second_bytes = static_cast<size_t>(second.nbytes());
  if (first_bytes == 0 || second_bytes == 0) return false;
  const uintptr_t first_start = reinterpret_cast<uintptr_t>(first.data_ptr());
  const uintptr_t second_start = reinterpret_cast<uintptr_t>(second.data_ptr());
  // CPU zero-copy DDR views have distinct Storage objects. Compare actual
  // contiguous byte ranges, without an overflowing end-address addition.
  return first_start <= second_start
      ? second_start - first_start < first_bytes
      : first_start - second_start < second_bytes;
}

void rpu_qwen3vl_scatter_row_parts_unflushed_(
    const at::Tensor& destination,
    const at::Tensor& row_indices,
    at::TensorList sources) {
  TORCH_CHECK(destination.device().type() == c10::DeviceType::PrivateUse1,
              "qwen3vl_scatter_row_parts_unflushed_: destination must be on RPU");
  TORCH_CHECK(row_indices.device().is_cpu() &&
                  row_indices.scalar_type() == at::kLong &&
                  row_indices.dim() == 1 && row_indices.is_contiguous(),
              "qwen3vl_scatter_row_parts_unflushed_: row_indices must be "
              "contiguous CPU int64 1D");
  TORCH_CHECK(destination.scalar_type() == at::kHalf &&
                  destination.dim() == 2 && destination.is_contiguous(),
              "qwen3vl_scatter_row_parts_unflushed_: destination must be "
              "contiguous RPU fp16 2D");
  TORCH_CHECK(!sources.empty(),
              "qwen3vl_scatter_row_parts_unflushed_: sources must be non-empty");
  TORCH_CHECK(!destination.requires_grad(),
              "qwen3vl_scatter_row_parts_unflushed_: inference destination only");
  TORCH_CHECK(!rpu_host_tensor_byte_ranges_overlap(destination, row_indices),
              "qwen3vl_scatter_row_parts_unflushed_: destination overlaps row_indices");

  int64_t source_rows = 0;
  for (const at::Tensor& source : sources) {
    const auto device_type = source.device().type();
    TORCH_CHECK(device_type == c10::DeviceType::CPU ||
                    device_type == c10::DeviceType::PrivateUse1,
                "qwen3vl_scatter_row_parts_unflushed_: source must be on CPU or RPU");
    TORCH_CHECK(source.scalar_type() == at::kHalf && source.dim() == 2 &&
                    source.is_contiguous() &&
                    source.size(1) == destination.size(1),
                "qwen3vl_scatter_row_parts_unflushed_: every source must be "
                "contiguous fp16 [rows, destination.size(1)]");
    source_rows += source.size(0);
  }
  TORCH_CHECK(source_rows == row_indices.numel(),
              "qwen3vl_scatter_row_parts_unflushed_: source rows ", source_rows,
              " != indices ", row_indices.numel());

  const int64_t* indices = row_indices.data_ptr<int64_t>();
  for (int64_t row = 0; row < source_rows; ++row) {
    TORCH_CHECK(indices[row] >= 0 && indices[row] < destination.size(0),
                "qwen3vl_scatter_row_parts_unflushed_: row index out of range: ",
                indices[row]);
    TORCH_CHECK(row == 0 || indices[row] > indices[row - 1],
                "qwen3vl_scatter_row_parts_unflushed_: indices must be "
                "strictly increasing");
  }

  const int64_t width = destination.size(1);
  c10::Half* dst = destination.data_ptr<c10::Half>();
  int64_t checked_rows = 0;
  for (const at::Tensor& source : sources) {
    if (rpu_host_tensor_byte_ranges_overlap(destination, source)) {
      const c10::Half* src = source.data_ptr<c10::Half>();
      for (int64_t row = 0; row < source.size(0); ++row) {
        TORCH_CHECK(src + row * width ==
                        dst + indices[checked_rows + row] * width,
                    "qwen3vl_scatter_row_parts_unflushed_: overlapping source "
                    "must already occupy its exact destination rows");
      }
    }
    checked_rows += source.size(0);
  }
  // These host-visible writes bypass ATen's in-place kernels. Keep views and
  // Python caches informed when no-grad (rather than inference) tensors are used.
  if (!destination.is_inference()) destination.unsafeGetTensorImpl()->bump_version();
  int64_t index_offset = 0;
  for (const at::Tensor& source : sources) {
    const c10::Half* src = source.data_ptr<c10::Half>();
    int64_t begin = 0;
    while (begin < source.size(0)) {
      const int64_t dst_row = indices[index_offset + begin];
      int64_t end = begin + 1;
      while (end < source.size(0) &&
             indices[index_offset + end] ==
                 indices[index_offset + end - 1] + 1) {
        ++end;
      }
      const size_t run_bytes = static_cast<size_t>(end - begin) *
          static_cast<size_t>(width) * sizeof(c10::Half);
      c10::Half* dst_run = dst + dst_row * width;
      const c10::Half* src_run = src + begin * width;
      if (dst_run != src_run) {
        std::memcpy(dst_run, src_run, run_bytes);
      }
      begin = end;
    }
    index_offset += source.size(0);
  }
  // Intentionally no DDR flush: the decoder prologue force-flushes the
  // complete stable input immediately before its mutable DMA consumes it.
}

void rpu_qwen3vl_gather_embedding_from_rpu_unflushed_(
    const at::Tensor& destination,
    const at::Tensor& vocabulary,
    const at::Tensor& token_ids,
    int64_t skip_token_id) {
  TORCH_CHECK(destination.device().type() == c10::DeviceType::PrivateUse1 &&
                  vocabulary.device().type() == c10::DeviceType::PrivateUse1,
              "qwen3vl_gather_embedding_from_rpu_unflushed_: destination and "
              "vocabulary must be on RPU");
  TORCH_CHECK(token_ids.device().is_cpu() ||
                  token_ids.device().type() == c10::DeviceType::PrivateUse1,
              "qwen3vl_gather_embedding_from_rpu_unflushed_: token_ids must "
              "be on CPU or host-visible RPU DDR");
  TORCH_CHECK(destination.scalar_type() == at::kHalf &&
                  vocabulary.scalar_type() == at::kHalf,
              "qwen3vl_gather_embedding_from_rpu_unflushed_: destination and "
              "vocabulary must be fp16");
  TORCH_CHECK(token_ids.scalar_type() == at::kLong,
              "qwen3vl_gather_embedding_from_rpu_unflushed_: token_ids must be int64");
  TORCH_CHECK(destination.dim() == 2 && vocabulary.dim() == 2 &&
                  token_ids.dim() == 1,
              "qwen3vl_gather_embedding_from_rpu_unflushed_: expected destination/"
              "vocabulary 2D and token_ids 1D");
  TORCH_CHECK(destination.is_contiguous() && vocabulary.is_contiguous() &&
                  token_ids.is_contiguous(),
              "qwen3vl_gather_embedding_from_rpu_unflushed_: all tensors must be contiguous");
  TORCH_CHECK(destination.size(0) == token_ids.numel() &&
                  destination.size(1) == vocabulary.size(1),
              "qwen3vl_gather_embedding_from_rpu_unflushed_: shape mismatch");
  TORCH_CHECK(!destination.requires_grad(),
              "qwen3vl_gather_embedding_from_rpu_unflushed_: inference destination only");
  TORCH_CHECK(!rpu_host_tensor_byte_ranges_overlap(destination, vocabulary),
              "qwen3vl_gather_embedding_from_rpu_unflushed_: destination overlaps vocabulary");
  TORCH_CHECK(!rpu_host_tensor_byte_ranges_overlap(destination, token_ids),
              "qwen3vl_gather_embedding_from_rpu_unflushed_: destination overlaps token_ids");

  const int64_t rows = destination.size(0);
  const int64_t width = destination.size(1);
  const int64_t vocab_rows = vocabulary.size(0);
  const int64_t* ids = token_ids.data_ptr<int64_t>();
  // Validate the whole request before the first write or version change.
  for (int64_t row = 0; row < rows; ++row) {
    TORCH_CHECK(ids[row] == skip_token_id ||
                    (ids[row] >= 0 && ids[row] < vocab_rows),
                "qwen3vl_gather_embedding_from_rpu_unflushed_: token id out of range: ",
                ids[row]);
  }
  c10::Half* dst = destination.data_ptr<c10::Half>();
  const c10::Half* vocab = vocabulary.data_ptr<c10::Half>();
  const size_t row_bytes = static_cast<size_t>(width) * sizeof(c10::Half);
  if (!destination.is_inference()) destination.unsafeGetTensorImpl()->bump_version();
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t token_id = ids[row];
    if (token_id == skip_token_id) continue;
    std::memcpy(dst + row * width, vocab + token_id * width, row_bytes);
  }
}

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
RpuTensorAllocatorPolicy g_rpu_tensor_allocator_policy;
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
// BC shims: set_debug(False)→ERROR(1) preserves the old fallback-suppression
// behavior (CPU_FALLBACK / MANUAL_FALLBACK live at WARN(2)); set_debug(True)→
// DEBUG(4) matches the most verbose pre-P1a state.
void rpu_set_debug(bool enabled) { rpu_set_debug_level(enabled ? 4 : 1); }
bool rpu_get_debug() { return rpu_get_debug_level() >= 4; }
void rpu_set_profile(bool enabled) { g_rpu_profile_enabled = enabled; }
bool rpu_get_profile() { return g_rpu_profile_enabled; }
void rpu_set_caching_allocator(bool enabled) {
  const auto result = g_rpu_tensor_allocator_policy.claim(enabled);
  TORCH_CHECK(
      result != RpuTensorAllocatorPolicy::ClaimResult::kConflict,
      "RPU tensor allocator policy is already frozen to ",
      g_rpu_tensor_allocator_policy.snapshot().caching ? "caching" : "direct",
      "; cannot switch to ", enabled ? "caching" : "direct",
      " in the same process. Start a fresh Python process to use a different policy.");
}
bool rpu_get_caching_allocator() {
  return g_rpu_tensor_allocator_policy.snapshot().caching;
}
bool rpu_caching_allocator_for_allocation() {
  return g_rpu_tensor_allocator_policy.freeze_for_allocation();
}
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
  const auto allocator_policy = g_rpu_tensor_allocator_policy.snapshot();
  pybind11::dict d;
  d["caching_allocator_enabled"] = allocator_policy.caching;
  d["caching_allocator_policy_frozen"] = allocator_policy.frozen;
  d["allocation"] = rpu_stat_to_dict(stats.allocation);
  d["reserved_bytes"] = rpu_stat_to_dict(stats.reserved_bytes);
  d["allocated_bytes"] = rpu_stat_to_dict(stats.allocated_bytes);
  d["active_bytes"] = rpu_stat_to_dict(stats.active_bytes);
  d["segment"] = rpu_stat_to_dict(stats.segment);
  // Only shared caching-allocator HostDDR mappings, excluding Launch, SPM
  // and direct allocations. Keep the public release counters address-free.
  d["caching_allocator_mapping"] = rpu_stat_to_dict(stats.segment);
  d["cached_idle_mapping"] =
      rpu::get_caching_allocator().getCachedSegmentCount();
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

uint16_t fp16_order_key(uint16_t bits) {
  const uint16_t abs_bits = bits & 0x7fffu;
  if (abs_bits > 0x7c00u) return 0;
  if (abs_bits == 0) return 0x8000u;
  return (bits & 0x8000u)
      ? static_cast<uint16_t>(~bits)
      : static_cast<uint16_t>(bits ^ 0x8000u);
}

#if defined(__aarch64__)
__attribute__((target("arch=armv8.2-a+fp16")))
float16x8_t load_numeric_half8(const c10::Half* values, int64_t offset) {
  const uint16x8_t bits = vld1q_u16(
      reinterpret_cast<const uint16_t*>(values + offset));
  const uint16x8_t abs_bits = vandq_u16(bits, vdupq_n_u16(0x7fffu));
  const uint16x8_t sanitized_bits = vbslq_u16(
      vcgtq_u16(abs_bits, vdupq_n_u16(0x7c00u)),
      vdupq_n_u16(0xfc00u), bits);
  return vreinterpretq_f16_u16(sanitized_bits);
}

__attribute__((target("arch=armv8.2-a+fp16")))
uint16_t horizontal_max_half_key(float16x8_t values) {
  const float16_t max_value = vmaxvq_f16(values);
  uint16_t max_bits;
  static_assert(sizeof(max_bits) == sizeof(max_value));
  std::memcpy(&max_bits, &max_value, sizeof(max_bits));
  return fp16_order_key(max_bits);
}
#endif

#if defined(__aarch64__)
__attribute__((target("arch=armv8.2-a+fp16")))
#endif
std::pair<int64_t, int64_t> top2_row_fp16(
    const c10::Half* row, int64_t cols) {
  TORCH_INTERNAL_ASSERT(cols >= 2);
  uint16_t best_key = 0;
  uint16_t second_key = 0;
  int64_t best_index = 0;
  int64_t second_index = 0;
  const auto insert_candidate = [&](uint16_t key, int64_t index) {
    if (key > best_key) {
      second_key = best_key;
      second_index = best_index;
      best_key = key;
      best_index = index;
    } else if (key > second_key) {
      second_key = key;
      second_index = index;
    }
  };

  int64_t offset = 0;
#if defined(__aarch64__)
  for (; offset + 32 <= cols; offset += 32) {
    const float16x8_t values0 = load_numeric_half8(row, offset);
    const float16x8_t values1 = load_numeric_half8(row, offset + 8);
    const float16x8_t values2 = load_numeric_half8(row, offset + 16);
    const float16x8_t values3 = load_numeric_half8(row, offset + 24);
    const float16x8_t group_max = vmaxq_f16(
        vmaxq_f16(values0, values1), vmaxq_f16(values2, values3));
    if (horizontal_max_half_key(group_max) > second_key) {
      alignas(16) uint16_t lane_bits[32];
      std::memcpy(lane_bits, row + offset, sizeof(lane_bits));
      for (int lane = 0; lane < 32; ++lane) {
        insert_candidate(fp16_order_key(lane_bits[lane]), offset + lane);
      }
    }
  }
  for (; offset + 8 <= cols; offset += 8) {
    const float16x8_t values = load_numeric_half8(row, offset);
    if (horizontal_max_half_key(values) > second_key) {
      alignas(16) uint16_t lane_bits[8];
      std::memcpy(lane_bits, row + offset, sizeof(lane_bits));
      for (int lane = 0; lane < 8; ++lane) {
        insert_candidate(fp16_order_key(lane_bits[lane]), offset + lane);
      }
    }
  }
#endif
  for (; offset < cols; ++offset) {
    insert_candidate(fp16_order_key(row[offset].x), offset);
  }
  return {best_index, second_index};
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

  at::Tensor cpu_logits = on_rpu
      ? rpu_to_cpu_zerocopy(logits, /*flush=*/true)
      : logits;
  at::Tensor result = rpu::host_argmax::reduce_cpu_rows(cpu_logits);
  return copy_result_to_rpu ? result.to(logits.device()) : result;
}

}  // namespace

at::Tensor rpu_lm_head_logits_top1(const at::Tensor& logits) {
  TORCH_CHECK(logits.device().type() == c10::DeviceType::PrivateUse1,
              "lm_head_logits_top1: logits must be on RPU device");
  return host_top1_impl(logits, /*copy_result_to_rpu=*/true);
}

at::Tensor rpu_argmax_lastdim_host(const at::Tensor& logits) {
  if (logits.device().is_cpu()) {
    return rpu::host_argmax::argmax_lastdim_cpu(logits);
  }
  TORCH_CHECK(logits.dim() == 2,
              "argmax_lastdim_host expects 2-D [rows, cols] logits, got ",
              logits.dim(), "-D");
  return host_top1_impl(logits, /*copy_result_to_rpu=*/false);
}

void rpu_zero_contiguous_tensor_blocks_sized(
    at::TensorList tensors, int64_t block_index) {
  TORCH_CHECK(!tensors.empty(),
              "zero_contiguous_tensor_blocks_sized: tensors must be non-empty");
  std::vector<void*> flush_ptrs;
  std::vector<size_t> flush_sizes;
  if (g_rpu_ddr_flush_force_enabled) {
    flush_ptrs.reserve(tensors.size());
    flush_sizes.reserve(tensors.size());
  }
  for (const at::Tensor& tensor : tensors) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                "zero_contiguous_tensor_blocks_sized: every tensor must be on RPU");
    TORCH_CHECK(tensor.is_contiguous() && tensor.dim() >= 2,
                "zero_contiguous_tensor_blocks_sized: tensors must be contiguous "
                "with a block dimension");
    TORCH_CHECK(tensor.size(0) == 1,
                "zero_contiguous_tensor_blocks_sized: batch size must be one");
    TORCH_CHECK(block_index >= 0 && block_index < tensor.size(1),
                "zero_contiguous_tensor_blocks_sized: block index out of range");
    TORCH_CHECK(!tensor.requires_grad(),
                "zero_contiguous_tensor_blocks_sized: inference tensors only");
  }
  for (const at::Tensor& tensor : tensors) {
    const size_t block_bytes = static_cast<size_t>(tensor.nbytes()) /
        static_cast<size_t>(tensor.size(1));
    if (!tensor.is_inference()) tensor.unsafeGetTensorImpl()->bump_version();
    if (block_bytes == 0) {
      continue;
    }
    auto* ptr = static_cast<uint8_t*>(tensor.data_ptr()) +
        static_cast<size_t>(block_index) * block_bytes;
    std::memset(ptr, 0, block_bytes);
    if (g_rpu_ddr_flush_force_enabled) {
      flush_ptrs.push_back(ptr);
      flush_sizes.push_back(block_bytes);
    }
  }
  if (!flush_ptrs.empty()) {
    rhino_lkn::RpuDdrFlushBatch(
        flush_ptrs.data(), flush_sizes.data(), flush_ptrs.size());
  }
}

at::Tensor rpu_lm_head_logits_top2_rescore_cpu(
    const at::Tensor& logits,
    const at::Tensor& hidden,
    const at::Tensor& raw_weight,
    int64_t candidate_count) {
  TORCH_CHECK(logits.device().type() == c10::DeviceType::PrivateUse1 &&
                  logits.scalar_type() == at::kHalf &&
                  logits.dim() >= 1 && logits.is_contiguous(),
              "lm_head_logits_top2_rescore_cpu: logits must be contiguous RPU fp16");
  TORCH_CHECK(hidden.device().type() == c10::DeviceType::PrivateUse1 &&
                  hidden.scalar_type() == at::kHalf && hidden.is_contiguous(),
              "lm_head_logits_top2_rescore_cpu: hidden must be contiguous RPU fp16");
  TORCH_CHECK(raw_weight.device().is_cpu() &&
                  raw_weight.scalar_type() == at::kHalf &&
                  raw_weight.dim() == 2 && raw_weight.is_contiguous(),
              "lm_head_logits_top2_rescore_cpu: raw_weight must be contiguous "
              "CPU fp16 [vocab, hidden]");
  TORCH_CHECK(candidate_count == 2,
              "lm_head_logits_top2_rescore_cpu: only candidate_count=2 is "
              "validated, got ", candidate_count);

  const int64_t vocab = logits.size(-1);
  TORCH_CHECK(vocab >= 2 && logits.numel() % vocab == 0,
              "lm_head_logits_top2_rescore_cpu: invalid logits shape ",
              logits.sizes());
  const int64_t rows = logits.numel() / vocab;
  const int64_t hidden_dim = raw_weight.size(1);
  TORCH_CHECK(raw_weight.size(0) == vocab,
              "lm_head_logits_top2_rescore_cpu: raw_weight vocab mismatch: ",
              raw_weight.size(0), " != ", vocab);
  TORCH_CHECK(hidden.numel() == rows * hidden_dim,
              "lm_head_logits_top2_rescore_cpu: hidden size mismatch: ",
              hidden.numel(), " != ", rows * hidden_dim);

  graph_aware_rpu_to_cpu_zerocopy_sync();
  if (g_rpu_ddr_flush_force_enabled) {
    void* flush_ptrs[2] = {logits.data_ptr(), hidden.data_ptr()};
    size_t flush_sizes[2] = {
        static_cast<size_t>(logits.nbytes()),
        static_cast<size_t>(hidden.nbytes())};
    rhino_lkn::RpuDdrFlushBatch(flush_ptrs, flush_sizes, 2);
  }
  at::Tensor result = at::empty(
      {rows}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  const c10::Half* logits_data = logits.data_ptr<c10::Half>();
  const c10::Half* hidden_data = hidden.data_ptr<c10::Half>();
  const c10::Half* weight_data = raw_weight.data_ptr<c10::Half>();
  int64_t* output = result.data_ptr<int64_t>();
  for (int64_t row = 0; row < rows; ++row) {
      const auto candidates = top2_row_fp16(
          logits_data + row * vocab, vocab);
      const c10::Half* hidden_row = hidden_data + row * hidden_dim;
      const c10::Half* best_weight =
          weight_data + candidates.first * hidden_dim;
      const c10::Half* second_weight =
          weight_data + candidates.second * hidden_dim;
      float best_score = 0.0f;
      float second_score = 0.0f;
      for (int64_t index = 0; index < hidden_dim; ++index) {
        const float hidden_value = static_cast<float>(hidden_row[index]);
        best_score += hidden_value * static_cast<float>(best_weight[index]);
        second_score += hidden_value * static_cast<float>(second_weight[index]);
      }
      output[row] = second_score > best_score
          ? candidates.second : candidates.first;
  }

  std::vector<int64_t> result_sizes = logits.sizes().vec();
  result_sizes.pop_back();
  return result.reshape(result_sizes);
}

// =============================================================================
// HW perf trace global config — see rpu_profile.h for design notes.
// =============================================================================
namespace {
std::mutex g_hw_perf_mu;
HwPerfConfig g_hw_perf_cfg{};
std::atomic<bool> g_hw_perf_enabled_fast{false};
std::atomic<int64_t> g_hw_perf_dumps_left{0};
std::atomic<uint64_t> g_hw_perf_dump_seq{0};

bool secure_hw_perf_output_dir(const std::string& output_dir) {
  namespace fs = std::filesystem;
  const fs::path requested(output_dir);
  if (requested.empty()) return false;

  // Walk one component at a time so a missing path can be created without
  // following a symlink. Existing parent directories may be shared (e.g.
  // /tmp); only the final directory is required to be private.
  std::error_code ec;
  fs::path current = requested.is_absolute() ? requested.root_path()
                                             : fs::current_path(ec);
  if (ec) return false;
  std::vector<fs::path> created;
  for (const auto& component : requested.relative_path()) {
    if (component == ".") continue;
    // Do not allow the diagnostic path to escape the caller's chosen base.
    if (component == ".." || component.empty()) return false;
    const fs::path next = current / component;
    ec.clear();
    const fs::file_status status = fs::symlink_status(next, ec);
    if (!ec) {
      if (fs::is_symlink(status) || !fs::is_directory(status)) return false;
      current = next;
      continue;
    }
    if (ec != std::errc::no_such_file_or_directory) return false;
    ec.clear();
    if (fs::create_directory(next, ec)) {
      created.push_back(next);
    } else if (ec) {
      return false;
    }
    // A concurrent creator may have won the race; verify it is still a real
    // directory before continuing and never accept a symlink substitution.
    ec.clear();
    const fs::file_status made = fs::symlink_status(next, ec);
    if (ec || fs::is_symlink(made) || !fs::is_directory(made)) return false;
    current = next;
  }

  // Newly-created components get private permissions. Never rewrite an
  // existing directory's mode: callers may intentionally share its parent,
  // and a diagnostic setter must not mutate unrelated filesystem state.
  for (const fs::path& path : created) {
    ec.clear();
    fs::permissions(path, fs::perms::owner_all, fs::perm_options::replace, ec);
    if (ec) return false;
  }

  ec.clear();
  const fs::file_status status = fs::symlink_status(requested, ec);
  if (ec || fs::is_symlink(status) || !fs::is_directory(status)) return false;
  const auto perms = status.permissions();
  if ((perms & (fs::perms::group_all | fs::perms::others_all)) !=
      fs::perms::none)
    return false;
  return ::access(output_dir.c_str(), W_OK | X_OK) == 0;
}
}  // namespace

bool rpu_hw_perf_runtime_compatible() noexcept {
  if (&rhino_lkn::rhino_launch_runtime_build_flavor == nullptr ||
      &rhino_lkn::rhino_launch_runtime_capabilities == nullptr) {
    return false;
  }
  constexpr uint32_t kReleaseFlavor = 0;
  constexpr uint64_t kSanitizedChrome = UINT64_C(1) << 0;
  constexpr uint64_t kDevelopmentApi = UINT64_C(1) << 1;
  return rhino_lkn::rhino_launch_runtime_build_flavor() == kReleaseFlavor &&
      (rhino_lkn::rhino_launch_runtime_capabilities() & kSanitizedChrome) != 0 &&
      (rhino_lkn::rhino_launch_runtime_capabilities() & kDevelopmentApi) == 0;
}

HwPerfConfig rpu_hw_perf_snapshot() {
  std::lock_guard<std::mutex> lock(g_hw_perf_mu);
  return g_hw_perf_cfg;
}

void rpu_set_hw_perf_trace(bool enabled, const std::string& output_dir,
                           int64_t max_dumps) {
  TORCH_CHECK(!output_dir.empty(),
              "set_hw_perf_trace: output_dir must not be empty");
  TORCH_CHECK(max_dumps >= 0,
              "set_hw_perf_trace: max_dumps must be non-negative, got ",
              max_dumps);
  if (enabled) {
    TORCH_CHECK(
        rpu_hw_perf_runtime_compatible(),
        "set_hw_perf_trace: loaded Rhino Launch is not the sanitized Release "
        "runtime (identity/capability query unavailable or mismatched)");
    TORCH_CHECK(
        secure_hw_perf_output_dir(output_dir),
        "set_hw_perf_trace: output_dir must be a private, writable directory "
        "without symlink components: ", output_dir);
  }

  // Publish the config and its refreshed budget as one setter transaction.
  // An execution that snapshots the new config must not observe the old cap.
  {
    std::lock_guard<std::mutex> lock(g_hw_perf_mu);
    g_hw_perf_cfg = {enabled, output_dir, max_dumps};
    g_hw_perf_dumps_left.store(max_dumps, std::memory_order_relaxed);
    g_hw_perf_enabled_fast.store(enabled, std::memory_order_release);
  }
}

bool rpu_hw_perf_enabled_fast() noexcept {
  return g_hw_perf_enabled_fast.load(std::memory_order_acquire);
}

bool rpu_get_hw_perf_trace() {
  return rpu_hw_perf_enabled_fast();
}

bool rpu_hw_perf_try_consume_dump() {
  int64_t remaining =
      g_hw_perf_dumps_left.load(std::memory_order_relaxed);
  while (remaining > 0) {
    if (g_hw_perf_dumps_left.compare_exchange_weak(
            remaining, remaining - 1, std::memory_order_relaxed)) {
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

    TORCH_CHECK(rpu_hw_perf_runtime_compatible(),
                "RhinoForge requires the sanitized Release Rhino Launch runtime");
    initialize_rpu_tensor_runtime();
    KernelCache::instance().initialize();
    g_rpu_backend_lifecycle = RpuBackendLifecycleState::kReady;
}

// Shutdown: 释放所有缓存的资源
// 必须在程序退出前、Rhino runtime 还活着的时候调用
void rpu_shutdown() {
    std::lock_guard<std::mutex> lock(g_rpu_backend_lifecycle_mu);
    if (g_rpu_backend_lifecycle != RpuBackendLifecycleState::kReady) {
        return;
    }
    RpuExecutionCleanupGuard cleanup("rpu_shutdown", /*shutting_down=*/true);
    // Mark first so a repeated/manual+atexit call is idempotent even if one of
    // the best-effort cleanup operations below reports an exception.
    g_rpu_backend_lifecycle = RpuBackendLifecycleState::kShutdown;

    // Close the process-wide tensor allocation gate before any Graph or DDR
    // resource teardown. This waits for an in-flight direct/caching allocation
    // to finish, rejects every later non-empty allocation, and serializes with
    // a direct tensor's SDK free.
    rpu_tensor_ddr_lifecycle().begin_shutdown();

    // Retire retained queues and graph allocations before tearing down their
    // Programs and backing allocators. The cleanup guard permits same-thread
    // nested graph invalidation while excluding all new execution.
    invalidate_registered_rpu_kernel_graphs();

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

    // 6. 显式 teardown DDRManager (HxilBufModuleClose) — 必须在所有持有 DDR
    //    资源的子系统 (SPM_POOL/caching_allocator/QueueCache 等) 清理完之后,
    //    在 libHxil 静态析构 ~HxilBufFactory 自检之前, 否则进程退出时会有
    //    4 条 [WARN: ~HxilBufFactory:23]: pAttListjPriv ... noise warning。
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
// custom_cpu_fallback — P7.1a graph-aware HostCallback dispatcher hook
// =============================================================================
// thin wrapper:log + 转发到 dispatch_graph_aware_cpu_fallback (实现在
// src/graph/custom_cpu_fallback_dispatch.cpp,fork graph state 4 个分支)。
// Keep graph_runtime.h and its inline Torch dependencies in the graph translation
// unit. The backend composition unit only forwards the dispatcher call; build-time
// and runtime Torch libraries must use a compatible ABI.
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

  // P7.1a: 走 graph-aware dispatch (PASSTHROUGH/BUILT eager / RECORDING tier 分派 /
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

// (v5-12 D-09: AdaRMS GEMV bisection harness forward decl + schema removed from
//  production; the harness now lives in librpu_conformance.so via
//  TORCH_LIBRARY_FRAGMENT — see Ledger §6 T2.)

// Dispatcher registration remains in this translation unit; the include is a
// source-organization boundary, not a separately compiled component.
#include "rpu_dispatch_registrations.inc"

// CPython bindings are likewise part of this composition translation unit.
#include "rpu_pybind.inc"
