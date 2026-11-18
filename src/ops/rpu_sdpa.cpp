// rpu_sdpa.cpp — native SPM SDPA launchers.
//
// rpu_launch_sdpa_spm_unified_kernel_v2 is the fused-model FP16 path;
// sdpa_dma_mask_to_spm and related helpers prepare its operands.
// The eager unified-layout entry point computes on CPU.
//
// device unified layout is [batch, seq_len, num_heads, head_dim]. K/V caches use
// the swizzled format produced by rpu_launch_llama_insert_kcache/_vcache.
// Other SDPA variants live in sibling rpu_sdpa_*.cpp files.

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <array>
#include <utility>
#include "rhino_launch_buffer.h"

#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include "rpu_qwen3vl_vision_sdpa.h"
#include "rpu_hyvla_sdpa_tiling.h"
#include "rpu_spm_allocator.h"
#include <c10/util/Half.h>
#include <cmath>
#include <map>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

// Constants
#define MAX_CORES 8
#define V16_NUM 16
#define SCM_PARAMS_PER_CORE 8  // 4 addresses * 2 uint16 each
#define SCM_REG_OFFSET 4096    // SCM params are set via set_regs(idx + 4096, value)

// Attention mask types
enum AttnMaskType {
  MASK_NONE = 0,
  MASK_LTM = 1,   // Lower Triangular Mask (causal)
  MASK_4D = 2,    // [batch, num_heads, seq_q, seq_k]
  MASK_3D = 3,    // [num_heads, seq_q, seq_k]
  MASK_2D = 4,    // [seq_q, seq_k]
  MASK_1D = 5,    // [seq_k]
};

at::Tensor rpu_temporal_sdpa_composite_test(
    const at::Tensor& q_shard,
    const at::Tensor& k_shard,
    const at::Tensor& v_shard)
{
  constexpr int64_t cores = 8;
  constexpr int64_t frames = 6;
  constexpr int64_t head_dim = 64;
  auto check = [=](const at::Tensor& x, const char* name) {
    TORCH_CHECK(x.device().type() == at::kPrivateUse1 &&
                x.scalar_type() == at::kHalf && x.is_contiguous(),
                name, " must be contiguous fp16 device");
    TORCH_CHECK(x.dim() == 4 && x.size(0) == cores &&
                x.size(1) == frames && x.size(3) == head_dim,
                name, " must be [8,6,P*2,64], got ", x.sizes());
  };
  check(q_shard, "q_shard");
  check(k_shard, "k_shard");
  check(v_shard, "v_shard");
  TORCH_CHECK(q_shard.sizes() == k_shard.sizes() &&
              q_shard.sizes() == v_shard.sizes(),
              "temporal composite Q/K/V shapes must match");
  TORCH_CHECK(q_shard.size(2) > 0 && q_shard.size(2) % 2 == 0,
              "temporal composite local heads must be positive and even");

  const int64_t patch_group = q_shard.size(2) / 2;
  const int64_t heads = patch_group * 16;
  const int64_t elems_per_core = q_shard.numel() / cores;
  const int64_t shard_bytes = elems_per_core * sizeof(c10::Half);
  const int64_t core_stride_bytes = q_shard.stride(0) * sizeof(c10::Half);

  if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
  SdpaConfig cfg{SdpaKernelType::FLASH_ATTN_SPM,
                 head_dim, heads, heads, static_cast<int>(cores), MASK_LTM};
  const int64_t tmp_bytes = Align(
      sdpa_compute_tmp_v16_size(cfg, frames) * 32, 256);
  using AR = SpmAllocator::AllocRequest;
  auto offsets = SPM_ALLOC.alloc_temporary_aliased({
      AR{shard_bytes, 1, 2},
      AR{shard_bytes, 1, 2},
      AR{shard_bytes, 1, 2},
      AR{shard_bytes, 2, 3},
      AR{tmp_bytes,   2, 2},
  });
  const uint32_t q_off = offsets[0];
  const uint32_t k_off = offsets[1];
  const uint32_t v_off = offsets[2];
  const uint32_t out_off = offsets[3];
  const uint32_t tmp_off = offsets[4];

  rpu_launch_ddr_scatter_spm_dma_immediate(
      const_cast<c10::Half*>(q_shard.data_ptr<c10::Half>()),
      elems_per_core, core_stride_bytes, SPM_ALLOC.addr(0, q_off), cores);
  rpu_launch_ddr_scatter_spm_dma_immediate(
      const_cast<c10::Half*>(k_shard.data_ptr<c10::Half>()),
      elems_per_core, core_stride_bytes, SPM_ALLOC.addr(0, k_off), cores);
  rpu_launch_ddr_scatter_spm_dma_immediate(
      const_cast<c10::Half*>(v_shard.data_ptr<c10::Half>()),
      elems_per_core, core_stride_bytes, SPM_ALLOC.addr(0, v_off), cores);

  auto key_cache = at::empty(
      {1, 1, patch_group * 2, 4, 8, 16, 16}, q_shard.options());
  auto value_cache = at::empty_like(key_cache);
  const KvInsertSegmentPlan kv_plan =
      rpu_resolve_kvinsert_segment_plan_auto(
          /*position=*/0, frames, frames, cores, heads, head_dim,
          KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16);
  rpu_launch_insert_kvcache_spm_unified_with_plan(
      key_cache, value_cache, k_off, v_off, heads, head_dim, cores,
      /*k_cache_batch_offset_elems=*/0,
      /*v_cache_batch_offset_elems=*/0,
      /*spm_rows=*/frames, kv_plan);
  rpu_launch_sdpa_spm_unified_kernel_v2(
      key_cache, value_cache,
      MASK_LTM, 1.0 / 8.0,
      q_off, out_off, tmp_off, 0,
      frames, heads, heads, head_dim, frames,
      cores, cores);

  auto output = at::empty_like(q_shard);
  rpu_launch_spm_scatter_ddr_dma_immediate(
      SPM_ALLOC.addr(0, out_off), output.data_ptr<c10::Half>(),
      elems_per_core, core_stride_bytes, cores);
  SPM_ALLOC.reset_temporary();
  return output;
}

// float32_to_uint32 now in rpu_helpers.h (included via rpu_ops.h)

static at::Tensor sdpa_cache_batch_operand(
    const at::Tensor& cache,
    int64_t cache_batch_offset_elems,
    const char* what) {
  if (cache_batch_offset_elems == 0) return cache;

  TORCH_CHECK(cache_batch_offset_elems > 0 && cache.dim() == 7 &&
                  cache.is_contiguous() && cache.stride(0) > 0 &&
                  cache_batch_offset_elems % cache.stride(0) == 0,
              what, ": cache_batch_offset_elems=", cache_batch_offset_elems,
              " must select one contiguous batch slot (stride0=",
              cache.stride(0), ")");
  const int64_t batch_slot = cache_batch_offset_elems / cache.stride(0);
  TORCH_CHECK(batch_slot < cache.size(0),
              what, ": batch slot ", batch_slot,
              " out of range for batch=", cache.size(0));
  const int64_t off_bytes =
      cache_batch_offset_elems * static_cast<int64_t>(sizeof(c10::Half));
  TORCH_CHECK(off_bytes % 256 == 0,
              what, ": batch slot byte offset ", off_bytes,
              " must be 256-aligned for the v128 cache address");
  return cache.select(0, batch_slot);
}

// ============================================================================
// SPM Unified Layout SDPA Kernel
// ============================================================================
// SDPA SPM Unified Kernel V2
// tile_m_v16: sQry<=128 ? CeilDiv(sQry,16) : 8
// tile_k_v16: from sdpa_compute_tiling (= tile_m_v16 initially, may VLM-shrink)
// grid_dim_x: CeilDiv(sQry, tile_m)  (seq tiles)
// grid_dim_y: num_heads_per_core      (heads)
// ============================================================================

// Published tail payload is parameterized by logical key rows. Keep this
// extension bounded to the common single-prefix LingBot Action geometries;
// other lengths retain the ordinary SDPA selector.
bool rpu_lingbot2_sdpa_tail_t51_key_rows_supported(int64_t key_rows) {
  return key_rows == 275 || key_rows == 291 || key_rows == 307 ||
      key_rows == 371 || key_rows == 451;
}

static void launch_sdpa_spm_unified_impl(
    const at::Tensor &key,
    const at::Tensor &value,
    int attn_mask_type,
    std::optional<double> scale,
    uint32_t q_off, uint32_t output_off, uint32_t sdpa_tmp_off, uint32_t sdpa_mask_off,
    int64_t seq_q, int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t kv_seq_len,
    int num_cores,
    int virtual_num_cores,
    int64_t cache_batch_offset_elems,
    bool use_16b,
    bool qwen3vl_tm160,
    const SdpaTiling* admitted_tiling,
    std::optional<KernelId> admitted_kernel = std::nullopt,
    bool lingbot2_tail = false)
{
  // Validate before divisions, cache operand staging, or narrowed queue/grid ABI.
  TORCH_CHECK(num_cores >= 1 && num_cores <= MAX_CORES &&
                  virtual_num_cores >= -1 && virtual_num_cores <= MAX_CORES,
              "SDPA SPM requires 1..8 real cores and -1/0 or 1..8 virtual cores");
  TORCH_CHECK(seq_q > 0 && seq_q <= UINT16_MAX &&
                  kv_seq_len > 0 && kv_seq_len <= int64_t{UINT16_MAX} * 16 &&
                  num_heads > 0 && num_heads <= UINT16_MAX &&
                  num_kv_heads > 0 && num_kv_heads <= UINT16_MAX &&
                  head_dim > 0 && head_dim <= UINT16_MAX && head_dim % 16 == 0,
              "SDPA SPM geometry exceeds its positive register domain");
  TORCH_CHECK(num_heads % num_cores == 0 &&
                  ((num_kv_heads % num_cores == 0) ||
                   (num_cores % num_kv_heads == 0)) &&
                  num_heads % num_kv_heads == 0,
              "SDPA SPM requires integral attention head partitions and GQA");
  TORCH_CHECK(!admitted_kernel ||
                  (!lingbot2_tail && admitted_tiling &&
                   ((*admitted_kernel == KernelId::HYVLA_EXPERT_SDPA_M16_S51 &&
                     admitted_tiling->tile_m_v16 == 1 && admitted_tiling->tile_m == 16 &&
                     admitted_tiling->tile_k_v16 == 16 && admitted_tiling->tile_k == 256) ||
                    (*admitted_kernel == KernelId::HYVLA_EXPERT_SDPA_K128_S51 &&
                     admitted_tiling->tile_m_v16 == 4 && admitted_tiling->tile_m == 64 &&
                     admitted_tiling->tile_k_v16 == 8 && admitted_tiling->tile_k == 128)) &&
                   admitted_tiling->tile_n_v16 == 8 && admitted_tiling->tile_n == 128 &&
                   !use_16b && !qwen3vl_tm160 && seq_q == 51 &&
                   rpu_hyvla_s51_key_rows_supported(kv_seq_len) &&
                   head_dim == 128 && num_heads == 16 && num_kv_heads == 8 &&
                   num_cores == 8 && virtual_num_cores == 8 && attn_mask_type == 4 &&
                   cache_batch_offset_elems == 0),
              "HyVLA attention alias requires its exact admitted physical schedule");
  const KernelId sdpa_kernel_id = admitted_kernel.value_or(lingbot2_tail
      ? KernelId::LINGBOT2_SDPA_T51_K275_TAIL : use_16b
      ? KernelId::SDPA_FLASH_ATTN_SPM_16B
      : KernelId::SDPA_FLASH_ATTN_SPM);

  if (lingbot2_tail) {
    TORCH_CHECK(seq_q == 51 && rpu_lingbot2_sdpa_tail_t51_key_rows_supported(kv_seq_len) &&
                    num_heads == 32 &&
                    num_kv_heads == 8 && head_dim == 128 && num_cores == 8 &&
                    virtual_num_cores == 8 && attn_mask_type == MASK_2D &&
                    !scale.has_value() && cache_batch_offset_elems == 0 &&
                    !use_16b && !qwen3vl_tm160 && admitted_tiling == nullptr,
                "Lingbot2 SDPA tail requires admitted T51/K{275,291,307,371,451}/Q32/KV8/D128/TP8 MASK2D");
    TORCH_CHECK(KernelCache::instance().has_loaded(sdpa_kernel_id),
                "Lingbot2 SDPA tail optional payload is absent");
    TORCH_CHECK(GET_QUEUE(8) != nullptr, "Lingbot2 SDPA tail eight-core queue is absent");
  }

  int64_t core_num = num_cores;  // real cores for dispatch and SPM addressing

  // Virtual tp for parameter computation (cache layout, head distribution, registers).
  // Use virtual TP=8 with an eight-way cache layout, even when fewer cores run.
  int64_t vtp = (virtual_num_cores > 0) ? virtual_num_cores : core_num;

  TORCH_CHECK(!qwen3vl_tm160 ||
                  (!use_16b && seq_q == 576 && kv_seq_len == 576 &&
                   head_dim == 64 && num_heads == 16 && num_kv_heads == 16 &&
                   core_num == 8 && vtp == 8 && attn_mask_type == 0 &&
                   cache_batch_offset_elems == 0),
              "Qwen3-VL tm160 requires the exact noncausal 2B/P576 Vision profile");

  // Compute "virtual" total head counts as runtime does:
  //   numHead = q_heads_per_core * virtual_tp
  //   nKVHead = kv_heads_per_core * virtual_tp
  // This inflates the totals but per-core values come out correct.
  int64_t q_heads_per_core = num_heads / core_num;
  int64_t kv_heads_per_core = CeilDiv(num_kv_heads, core_num);
  int64_t virtual_num_heads = q_heads_per_core * vtp;
  int64_t virtual_num_kv_heads = kv_heads_per_core * vtp;
  TORCH_CHECK(virtual_num_heads <= UINT16_MAX && virtual_num_kv_heads <= UINT16_MAX,
              "SDPA SPM virtual head counts exceed uint16 registers");

  size_t dwidth = sizeof(c10::Half);
  int64_t dwidth_v16 = dwidth * V16_NUM;

  // Per-core head counts (same whether computed from real or virtual)
  int64_t num_heads_per_core = q_heads_per_core;  // = num_heads / core_num

  // ===== K/V already in DDR in swizzle format =====
  TORCH_CHECK(
      cache_batch_offset_elems == 0 ||
          !RpuKernelGraph::active().kernel_register_census_active(),
      "typed DDR-register census does not yet encode a KV-cache batch "
      "offset; refusing an unproven SDPA cache subrange");
  const at::Tensor key_operand = sdpa_cache_batch_operand(
      key, cache_batch_offset_elems, "sdpa_spm_unified_kernel_v2(key)");
  const at::Tensor value_operand = sdpa_cache_batch_operand(
      value, cache_batch_offset_elems, "sdpa_spm_unified_kernel_v2(value)");
  auto register_writer =
      RpuKernelGraph::active().stage_kernel_ddr_registers(
          sdpa_kernel_id,
          std::vector<GraphDdrRegisterOperandSpec>{
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::KeyCache,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                      50, 51},
                  key_operand},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::ValueCache,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                      52, 53},
                  value_operand}});
  if (lingbot2_tail) {
    RpuKernelGraph::active().keep_alive(key_operand);
    RpuKernelGraph::active().keep_alive(value_operand);
  }
  c10::Half* k_ptr = key_operand.data_ptr<c10::Half>();
  c10::Half* v_ptr = value_operand.data_ptr<c10::Half>();
  rpu_ddr_flush(k_ptr);
  rpu_ddr_flush(v_ptr);

  // ===== Calculate SDPA parameters using virtual tp =====
  int64_t nkv_head_per_core = CeilDiv(virtual_num_kv_heads, vtp);
  int64_t gqa_group_size = num_heads_per_core / nkv_head_per_core;

  // Cache layout parameters — must match RPUCache allocation (uses effective_kv_slots)
  int64_t nkv_head_chunk = std::min(vtp, virtual_num_kv_heads);
  int64_t nkv_head_vx = CeilDiv(virtual_num_kv_heads, vtp);
  int64_t core_num_per_kv_head = CeilDiv(vtp, virtual_num_kv_heads);

  int64_t head_dim_chunk = 16;
  int64_t head_dim_vx = CeilDiv(head_dim, head_dim_chunk);
  int64_t seq_k = kv_seq_len;
  int64_t seq_k_chunk = 16;
  int64_t seq_k_vx = CeilDiv(seq_k, seq_k_chunk);
  int64_t seq_v_chunk = seq_k_chunk;
  int64_t seq_v_vx = seq_k_vx;

  int64_t seq_q_v16 = CeilDiv(seq_q, 16);
  int64_t seq_k_v16 = CeilDiv(seq_k, 16);
  int64_t seq_q_acc = seq_k - seq_q;
  int64_t seq_q_acc_v16 = CeilDiv(seq_q_acc, 16);

  int64_t head_dim_q_v16 = CeilDiv(head_dim, 16);
  int64_t head_dim_k_v16 = head_dim_q_v16;
  int64_t head_dim_v_v16 = head_dim_q_v16;

  // Kernel tiling.
  // Use virtual head counts and virtual tp to match the rest of the kernel
  // parameter computation in this function (cache strides assume virtual_attn_tp).
  SdpaConfig cfg{SdpaKernelType::FLASH_ATTN_SPM,
                 head_dim, virtual_num_heads, virtual_num_kv_heads,
                 static_cast<int>(vtp), attn_mask_type};
  SdpaTiling t = sdpa_compute_tiling(cfg, seq_q);
  if (qwen3vl_tm160) {
    const auto tile_k = choose_tile_k_conservative(10, t.tile_n_v16, 0, 4);
    TORCH_CHECK(tile_k.has_value() &&
                    *tile_k * 4 <= t.tile_k_v16 * CeilDiv(seq_q, t.tile_m),
                "Qwen3-VL tm160 exceeds the declared SDPA workspace");
    t.tile_m_v16 = 10;
    t.tile_k_v16 = *tile_k;
    t.tile_m = 160;
    t.tile_k = *tile_k * 16;
  }
  if (admitted_tiling) {
    TORCH_CHECK(!qwen3vl_tm160,
                "ordinary SDPA tiling cannot share the delivery selector");
    t = *admitted_tiling;
  }
  int64_t tile_m_v16 = t.tile_m_v16;
  int64_t tile_n_v16 = t.tile_n_v16;
  int64_t tile_k_v16 = t.tile_k_v16;

  int64_t tile_m = tile_m_v16 * 16;
  int64_t tile_n = tile_n_v16 * 16;
  int64_t tile_k = tile_k_v16 * 16;
  TORCH_CHECK(tile_m > 0 && tile_m <= UINT16_MAX &&
                  tile_n > 0 && tile_n <= UINT16_MAX &&
                  tile_k > 0 && tile_k <= UINT16_MAX,
              "SDPA SPM tile dimensions exceed uint16 registers");

  // Mask type (passed directly, mask data already in SPM if needed)
  AttnMaskType mask_type = static_cast<AttnMaskType>(attn_mask_type);
  bool is_causal = (mask_type == MASK_LTM);

  // Scale factor
  float scale_factor = scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim)));
  uint32_t scale_u32 = float32_to_uint32(scale_factor);

  // Grid dimensions: x=seq_tiles, y=heads, z=batch.
  uint16_t grid_dim_x = CeilDiv(seq_q, tile_m);
  uint16_t grid_dim_y = num_heads_per_core;
  uint16_t grid_dim_z = 1;  // batch=1 always

  sdpa_validate_kernel_call(seq_q, seq_k,
                            tile_m_v16, tile_n_v16, tile_k_v16, tile_m,
                            grid_dim_x, gqa_group_size,
                            head_dim_q_v16, head_dim_k_v16, head_dim_v_v16,
                            attn_mask_type);

  // KV cache addressing
  int64_t row_size = 512;
  int64_t page_size = row_size * nkv_head_chunk;
  int64_t chapter_size = page_size * head_dim_vx;
  int64_t section_size = chapter_size * nkv_head_vx;

  int64_t row_size_v128 = row_size >> 8;
  int64_t page_size_v128 = page_size >> 8;
  int64_t chapter_size_v128 = chapter_size >> 8;
  int64_t section_size_v128 = section_size >> 8;
  TORCH_CHECK(section_size_v128 <= UINT16_MAX,
              "SDPA SPM cache section stride exceeds uint16 registers");

  // Use pre-allocated tmp addresses from spm_cache (managed by SimpleSpmBuffers)
  std::vector<uint32_t> tmp_addr_v16_vec(core_num);
  for (int i = 0; i < core_num; ++i) {
    tmp_addr_v16_vec[i] = SPM_ALLOC.addr(i, sdpa_tmp_off) >> 5;
  }

  // Mask data is already in SPM (loaded by caller via weight preload or DMA).
  // No CPU memcpy here — fully batch-compatible.

  // Define register setup lambda
  // Uses virtual values (vtp, virtual_num_heads, virtual_num_kv_heads) for cache
  // addressing registers, but real core_num for SPM address setup and dispatch.
  auto setup_regs = [=](Kernel_t* kernel) {
    kernel->set_regs(0, (uint16_t)(seq_k & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(seq_k >> 16));
    kernel->set_regs(2, (uint16_t)seq_q);
    kernel->set_regs(3, (uint16_t)seq_q_v16);
    kernel->set_regs(4, (uint16_t)seq_k_v16);
    kernel->set_regs(5, (uint16_t)seq_q_acc_v16);
    kernel->set_regs(6, (uint16_t)virtual_num_heads);    // virtual total Q heads
    kernel->set_regs(7, (uint16_t)virtual_num_kv_heads);  // virtual total KV heads
    kernel->set_regs(8, (uint16_t)num_heads_per_core);
    kernel->set_regs(9, (uint16_t)nkv_head_per_core);
    kernel->set_regs(10, (uint16_t)head_dim);
    kernel->set_regs(11, (uint16_t)head_dim);
    kernel->set_regs(12, (uint16_t)head_dim);
    kernel->set_regs(13, (uint16_t)head_dim_q_v16);
    kernel->set_regs(14, (uint16_t)head_dim_k_v16);
    kernel->set_regs(15, (uint16_t)head_dim_v_v16);
    kernel->set_regs(16, (uint16_t)(seq_q_acc & 0xFFFF));
    kernel->set_regs(17, (uint16_t)(seq_q_acc >> 16));
    kernel->set_regs(18, (uint16_t)vtp);  // virtual tp for cache addressing
    kernel->set_regs(19, (uint16_t)gqa_group_size);

    kernel->set_regs(20, (uint16_t)section_size_v128);
    kernel->set_regs(21, (uint16_t)chapter_size_v128);
    kernel->set_regs(22, (uint16_t)page_size_v128);
    kernel->set_regs(23, (uint16_t)row_size_v128);
    kernel->set_regs(24, (uint16_t)nkv_head_chunk);
    kernel->set_regs(25, (uint16_t)nkv_head_vx);
    kernel->set_regs(26, (uint16_t)head_dim_chunk);
    kernel->set_regs(27, (uint16_t)head_dim_vx);
    kernel->set_regs(28, (uint16_t)seq_k_chunk);
    kernel->set_regs(29, (uint16_t)seq_k_vx);
    kernel->set_regs(30, (uint16_t)seq_v_chunk);
    kernel->set_regs(31, (uint16_t)seq_v_vx);
    kernel->set_regs(32, (uint16_t)core_num_per_kv_head);

    kernel->set_regs(40, (uint16_t)(scale_u32 & 0xFFFF));
    kernel->set_regs(41, (uint16_t)(scale_u32 >> 16));
    kernel->set_regs(42, (uint16_t)1);  // has_scale

    kernel->set_regs(56, (uint16_t)mask_type);

    kernel->set_regs(58, (uint16_t)tile_m);
    kernel->set_regs(59, (uint16_t)tile_n);
    kernel->set_regs(60, (uint16_t)tile_k);
    kernel->set_regs(61, (uint16_t)tile_m_v16);
    kernel->set_regs(62, (uint16_t)tile_n_v16);
    kernel->set_regs(63, (uint16_t)tile_k_v16);

    kernel->set_regs(64, (uint16_t)grid_dim_x);
    kernel->set_regs(65, (uint16_t)grid_dim_y);
    kernel->set_regs(66, (uint16_t)grid_dim_z);

    // For SPM kernel: Q input and output addresses are in SPM (v16 units: addr >> 5)
    for (int i = 0; i < core_num; ++i) {
      uint32_t q_spm_addr_v16 = SPM_ALLOC.addr(i, q_off) >> 5;
      uint32_t out_spm_addr_v16 = SPM_ALLOC.addr(i, output_off) >> 5;
      uint32_t tmp_addr_v16 = tmp_addr_v16_vec[i];
      uint32_t mask_addr_v16 = (mask_type == MASK_2D) ? (SPM_ALLOC.addr(i, sdpa_mask_off) >> 5) : 0;

      int scm_base = SCM_REG_OFFSET + i * SCM_PARAMS_PER_CORE;
      rpu_set_scm_u32_checked(
          kernel, scm_base + 0, q_spm_addr_v16, "rpu_sdpa query");
      rpu_set_scm_u32_checked(
          kernel, scm_base + 2, tmp_addr_v16, "rpu_sdpa temporary");
      rpu_set_scm_u32_checked(
          kernel, scm_base + 4, mask_addr_v16, "rpu_sdpa mask");
      rpu_set_scm_u32_checked(
          kernel, scm_base + 6, out_spm_addr_v16, "rpu_sdpa output");
    }
  };

  std::vector<uint8_t> core_list;
  for (int i = 0; i < core_num; ++i) {
    core_list.push_back(static_cast<uint8_t>(i));
  }

  // Graph-aware unified path: GET_KERNEL / GET_QUEUE 透明处理 PASSTHROUGH
  // (立即发射) 和 RECORDING / REPLAYING (走 RpuKernelGraph batch 捕获)。
  Kernel_t* kernel = GET_KERNEL(sdpa_kernel_id);
  TORCH_CHECK(kernel != nullptr, "Failed to get SDPA SPM kernel");
  setup_regs(kernel);
  register_writer.write(*kernel);

  auto* wq = GET_QUEUE(core_num);
  wq->set_broadcast_mode(true);
  wq->set_flush_icache(false);

  wq->enqueu_kernel(*kernel, {grid_dim_x, grid_dim_y, grid_dim_z}, core_list);
}

void rpu_launch_sdpa_spm_unified_kernel_v2(
    const at::Tensor& key, const at::Tensor& value,
    int attn_mask_type, std::optional<double> scale,
    uint32_t q_off, uint32_t output_off, uint32_t sdpa_tmp_off, uint32_t sdpa_mask_off,
    int64_t seq_q, int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t kv_seq_len, int num_cores, int virtual_num_cores,
    int64_t cache_batch_offset_elems, bool use_16b, bool qwen3vl_tm160) {
  launch_sdpa_spm_unified_impl(
      key, value, attn_mask_type, scale,
      q_off, output_off, sdpa_tmp_off, sdpa_mask_off,
      seq_q, num_heads, num_kv_heads, head_dim, kv_seq_len,
      num_cores, virtual_num_cores, cache_batch_offset_elems,
      use_16b, qwen3vl_tm160, nullptr);
}

// Bounded LingBot Action tail SDPA. Arguments are allocator offsets, converted
// once by the shared SCM setup. The unchanged packed cache includes ceil16(K)
// physical rows, and the explicit row-major mask uses that same stride.
void rpu_launch_lingbot2_sdpa_tail_t51_spm_kernel(
    const at::Tensor& key, const at::Tensor& value,
    uint32_t q_off, uint32_t output_off, uint32_t tmp_off, uint32_t mask_off,
    int64_t kv_seq_len) {
  TORCH_CHECK(rpu_lingbot2_sdpa_tail_t51_key_rows_supported(kv_seq_len),
              "Lingbot2 SDPA tail key rows are outside the bounded Action profile");
  const int64_t physical_rows = CeilDiv(kv_seq_len, int64_t{16}) * 16;
  const auto cache_ok = [physical_rows](const at::Tensor& tensor) {
    if (!(tensor.defined() && tensor.scalar_type() == at::kHalf &&
          tensor.device().type() == c10::DeviceType::PrivateUse1 &&
          tensor.layout() == c10::Layout::Strided && tensor.is_contiguous() &&
          tensor.dim() == 7 && tensor.size(0) == 1 && tensor.size(1) >= physical_rows / 16 &&
          tensor.size(2) == 1 && tensor.size(3) == 8 && tensor.size(4) == 8 &&
          tensor.size(5) == 16 && tensor.size(6) == 16 &&
          tensor.storage_offset() >= 0)) return false;
    const uint64_t storage_bytes = tensor.storage().nbytes();
    const uint64_t offset = static_cast<uint64_t>(tensor.storage_offset());
    if (offset > storage_bytes / sizeof(c10::Half) ||
        tensor.nbytes() > storage_bytes - offset * sizeof(c10::Half)) return false;
    const uint64_t address = RpuGetDevAddr(tensor.data_ptr());
    constexpr uint64_t limit = uint64_t{1} << 40;
    return address != 0 && (address & 255u) == 0 && address < limit &&
        tensor.nbytes() <= limit - address;
  };
  TORCH_CHECK(cache_ok(key) && cache_ok(value) && key.device() == value.device() &&
                  key.sizes() == value.sizes(),
              "Lingbot2 SDPA tail requires complete same-device aligned FP16 KV caches "
              "[1,capacity/16,1,8,8,16,16] with capacity >= ceil16(K)");
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Lingbot2 SDPA tail requires initialized SPM");
  const std::array<std::pair<uint32_t, uint64_t>, 4> ranges{{
      {q_off, 51 * 512 * 2}, {output_off, 51 * 512 * 2},
      {tmp_off, 65536}, {mask_off, uint64_t{51} * static_cast<uint64_t>(physical_rows) * 2}}};
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto [offset, bytes] = ranges[i];
    TORCH_CHECK((offset & 31u) == 0 && bytes <= SpmAllocator::SPM_USABLE &&
                    offset <= SpmAllocator::SPM_USABLE - bytes,
                "Lingbot2 SDPA tail requires complete aligned SPM offset intervals");
    for (int core = 0; core < 8; ++core) {
      const uint64_t base = SPM_ALLOC.addr(core, 0);
      TORCH_CHECK((base & 31u) == 0 && base + offset + bytes <= (uint64_t{1} << 32),
                  "Lingbot2 SDPA tail SPM interval exceeds its U32 address domain");
    }
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(uint64_t{offset} + bytes <= ranges[j].first ||
                      uint64_t{ranges[j].first} + ranges[j].second <= offset,
                  "Lingbot2 SDPA tail query, output, temporary and mask must be disjoint");
  }
  const SdpaConfig cfg{SdpaKernelType::FLASH_ATTN_SPM, 128, 32, 8, 8, MASK_2D};
  const auto tile = sdpa_compute_tiling(cfg, 51);
  TORCH_CHECK(tile.tile_m_v16 == 4 && tile.tile_n_v16 == 8 && tile.tile_k_v16 == 16,
              "Lingbot2 SDPA tail requires its original physical tile 4/8/16");
  launch_sdpa_spm_unified_impl(
      key, value, MASK_2D, std::nullopt, q_off, output_off, tmp_off, mask_off,
      51, 32, 8, 128, kv_seq_len, 8, 8, 0, false, false, nullptr, std::nullopt, true);
}


// Preserve the already-published six-argument ABI and exact K275 behavior.
void rpu_launch_lingbot2_sdpa_t51_k275_tail_spm_kernel(
    const at::Tensor& key, const at::Tensor& value,
    uint32_t q_off, uint32_t output_off, uint32_t tmp_off, uint32_t mask_off) {
  rpu_launch_lingbot2_sdpa_tail_t51_spm_kernel(
      key, value, q_off, output_off, tmp_off, mask_off, 275);
}

void rpu_launch_sdpa_qwen3vl_n1200_tm160(
    const at::Tensor& key, const at::Tensor& value,
    uint32_t q_off, uint32_t output_off, uint32_t sdpa_tmp_off,
    int64_t seq_q, int64_t declared_tmp_bytes) {
  const auto t = rpu_qwen3vl_n1200_tm160_tiling(seq_q, 1200, declared_tmp_bytes);
  TORCH_CHECK(t.has_value(), "ordinary Qwen3-VL N1200 Tm160 geometry/workspace mismatch");
  launch_sdpa_spm_unified_impl(
      key, value, /*MASK_NONE=*/0, /*1/sqrt(64)=*/0.125,
      q_off, output_off, sdpa_tmp_off, /*mask_off=*/0,
      seq_q, /*Q=*/16, /*KV=*/16, /*D=*/64, /*Sk=*/1200,
      /*cores=*/8, /*vtp=*/8, /*cache_offset=*/0,
      /*use_16b=*/false, /*delivery_tm160=*/false, &*t);
}

// Both explicit HyVLA schedules share complete cache and SPM owner checks.
// The caller selects a bounded tile first; no generic SDPA selector changes.
static void launch_sdpa_hyvla_s51_checked(
    const at::Tensor& key, const at::Tensor& value,
    uint32_t q_off, uint32_t output_off, uint32_t tmp_off, uint32_t mask_off,
    int64_t query_rows, int64_t key_rows, int64_t declared_tmp_bytes,
    const std::optional<SdpaTiling>& tile, KernelId kernel_id) {
  TORCH_CHECK(tile.has_value() && SPM_ALLOC.is_initialized(),
              "HyVLA attention requires Q51/P{192,208,224,240}/D128/GQA2/MASK2D and its declared scratch");
  for (const auto* cache : {&key, &value}) {
    TORCH_CHECK(cache->defined() && cache->is_contiguous() &&
                    cache->device().type() == c10::DeviceType::PrivateUse1 &&
                    cache->scalar_type() == at::kHalf && cache->dim() == 7 &&
                    cache->size(0) == 1 && cache->size(1) >= CeilDiv(key_rows, int64_t{16}) &&
                    cache->size(2) == 1 && cache->size(3) == 8 && cache->size(4) == 8 &&
                    cache->size(5) == 16 && cache->size(6) == 16,
                "HyVLA attention requires FP16 device [1,ceil16(K)/16,1,8,8,16,16] cache capacity");
    const uint64_t address = rhino_lkn::RpuGetDevAddr(cache->data_ptr());
    constexpr uint64_t limit = UINT64_C(1) << 40;
    TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit &&
                    cache->nbytes() <= limit - address,
                "HyVLA attention requires complete encodable cache DDR ranges");
  }
  TORCH_CHECK(key.device() == value.device() && key.size(1) == value.size(1),
              "HyVLA attention cache device/capacity mismatch");
  const std::array<std::pair<uint32_t, uint64_t>, 4> regions{{
      {q_off, 51 * 256 * 2}, {output_off, 51 * 256 * 2},
      {tmp_off, uint64_t(declared_tmp_bytes)}, {mask_off, uint64_t(51 * CeilDiv(key_rows, int64_t{16}) * 16 * 2)}}};
  for (size_t i = 0; i < regions.size(); ++i) {
    const uint64_t offset = regions[i].first, bytes = regions[i].second;
    TORCH_CHECK(offset % 256 == 0 && offset + bytes <= SpmAllocator::SPM_USABLE,
                "HyVLA attention requires complete aligned SPM offset owners");
    for (int core = 0; core < 8; ++core)
      TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + offset + bytes <= uint64_t(UINT32_MAX) + 1,
                  "HyVLA attention SPM address overflows on a core");
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(offset + bytes <= regions[j].first ||
                      uint64_t(regions[j].first) + regions[j].second <= offset,
                  "HyVLA attention requires disjoint query/output/scratch/mask owners");
  }
  launch_sdpa_spm_unified_impl(
      key, value, /*MASK_2D=*/4, std::nullopt,
      q_off, output_off, tmp_off, mask_off,
      query_rows, /*Q=*/16, /*KV=*/8, /*D=*/128, key_rows,
      /*cores=*/8, /*vtp=*/8, /*cache_offset=*/0,
      /*use_16b=*/false, /*delivery_tm160=*/false, &*tile, kernel_id);
}

// Identical N/K tiling and online K order; additional query tiles require
// independently declared 256KiB scratch.
void rpu_launch_sdpa_hyvla_s51_m16(
    const at::Tensor& key, const at::Tensor& value,
    uint32_t q_off, uint32_t output_off, uint32_t tmp_off, uint32_t mask_off,
    int64_t query_rows, int64_t key_rows, int64_t declared_tmp_bytes) {
  launch_sdpa_hyvla_s51_checked(key, value, q_off, output_off, tmp_off, mask_off,
      query_rows, key_rows, declared_tmp_bytes,
      rpu_hyvla_s51_m16_tiling(query_rows, key_rows, declared_tmp_bytes),
      KernelId::HYVLA_EXPERT_SDPA_M16_S51);
}

// Keep M64/N128, halve K and scratch. Online-softmax rounding differs from
// K256; numerical diagnostics belong to this separately admitted profile.
void rpu_launch_sdpa_hyvla_s51_k128(
    const at::Tensor& key, const at::Tensor& value,
    uint32_t q_off, uint32_t output_off, uint32_t tmp_off, uint32_t mask_off,
    int64_t query_rows, int64_t key_rows, int64_t declared_tmp_bytes) {
  launch_sdpa_hyvla_s51_checked(key, value, q_off, output_off, tmp_off, mask_off,
      query_rows, key_rows, declared_tmp_bytes,
      rpu_hyvla_s51_k128_tiling(query_rows, key_rows, declared_tmp_bytes),
      KernelId::HYVLA_EXPERT_SDPA_K128_S51);
}

// =============================================================================
// SDPA Mask: prepare (CPU side) + DMA (device side, batch-compatible)
// =============================================================================
//
// Route B mask slot: the prepared mask is `.to(at::kPrivateUse1)`-allocated
// fresh each call (data_ptr unstable across forwards) — graph-mode DMA bakes
// the src addr at BUILD and can't rebind on REPLAY. Each FusedModelBase owns
// one typed cache containing a stable device tensor per
// (seq_q, seq_k_aligned, dtype, device). The model-owned strong reference keeps
// the address stable across sequential host threads and releases it with the
// model, without process-global owner keys or ABA-prone raw identities.
//
// Caller-level mask caches (e.g. adarms's prepared_mask_input_ref_) still
// short-circuit redundant sdpa_prepare_mask calls when the same input tensor
// is passed across forwards; Route B handles the OTHER path (cache miss
// during BUILD then REPLAY of the same shape — now safe because the slot
// pointer is stable).
struct SdpaStableMaskCache::Impl {
    struct Key {
        int64_t seq_q;
        int64_t seq_k_aligned;
        at::ScalarType dtype;
        at::DeviceType device;
        // Several same-shape chunk masks can be alive at once. Without this
        // identity, the last copy_ overwrites every earlier chunk's DMA source.
        int64_t ordinal;

        bool operator<(const Key& other) const {
            if (seq_q != other.seq_q) return seq_q < other.seq_q;
            if (seq_k_aligned != other.seq_k_aligned) {
                return seq_k_aligned < other.seq_k_aligned;
            }
            if (dtype != other.dtype) return dtype < other.dtype;
            if (device != other.device) return device < other.device;
            return ordinal < other.ordinal;
        }
    };

    std::mutex mutex;
    std::map<Key, at::Tensor> slots;
};

SdpaStableMaskCache::SdpaStableMaskCache()
    : pimpl_(std::make_unique<Impl>()) {}

SdpaStableMaskCache::~SdpaStableMaskCache() = default;

at::Tensor SdpaStableMaskCache::stable_slot(
    int64_t seq_q,
    int64_t seq_k_aligned,
    at::TensorOptions options,
    int64_t ordinal) {
    Impl::Key key{seq_q, seq_k_aligned,
                  options.dtype().toScalarType(),
                  options.device().type(), ordinal};
    std::lock_guard<std::mutex> lock(pimpl_->mutex);
    auto [it, inserted] = pimpl_->slots.try_emplace(key);
    if (inserted || !it->second.defined()) {
        it->second = at::empty({seq_q, seq_k_aligned}, options);
    }
    return it->second;
}

// CPU side of an explicit mask: reduce dims to [seq_q, seq_k], FP16, and pad
// the key columns to a multiple of 16. Skips pad_nd when seq_k is aligned.
static at::Tensor sdpa_mask_padded_cpu(
    const at::Tensor& attn_mask, int64_t seq_q, int64_t seq_k)
{
    at::Tensor mask_2d = attn_mask;
    if (mask_2d.dim() == 4) {
        mask_2d = mask_2d.select(0, 0).select(0, 0);
    } else if (mask_2d.dim() == 3) {
        mask_2d = mask_2d.select(0, 0);
    }
    TORCH_CHECK(mask_2d.dim() == 2, "SDPA mask must reduce to 2D, got ", mask_2d.dim(), "D");
    TORCH_CHECK(mask_2d.size(0) == seq_q && mask_2d.size(1) == seq_k,
                "SDPA mask shape [", mask_2d.size(0), ",", mask_2d.size(1),
                "] != expected [", seq_q, ",", seq_k, "]");
    if (mask_2d.scalar_type() != at::kHalf) mask_2d = mask_2d.to(at::kHalf);
    if (mask_2d.device().type() != c10::DeviceType::CPU) mask_2d = mask_2d.to(at::kCPU);
    mask_2d = mask_2d.contiguous();

    int64_t seq_k_aligned = Align(seq_k, 16);
    // Pad the beyond-kv_seq columns with -inf (masked), NOT 0 (attended): those columns map to
    // key slots [seq_k, seq_k_aligned) that hold stale cache K/V. With 0 the kernel attends them,
    // inflating the softmax denominator (½-magnitude bug on the wall-oss denoise). -inf masks them.
    return (seq_k_aligned != seq_k)
        ? at::constant_pad_nd(mask_2d, {0, seq_k_aligned - seq_k},
                              -std::numeric_limits<float>::infinity())
        : mask_2d;
}

// CPU side: reduce dims, convert type, pad if needed, move to RPU DDR.
PreparedMask sdpa_prepare_mask(
    const c10::optional<at::Tensor> &attn_mask,
    bool is_causal,
    int64_t seq_q, int64_t seq_k,
    SdpaStableMaskCache& stable_cache,
    int64_t ordinal)
{
    if (is_causal) return {{}, 1};   // MASK_LTM
    if (!attn_mask.has_value() || !attn_mask->defined()) return {{}, 0};  // MASK_NONE

    const at::Tensor mask_padded = sdpa_mask_padded_cpu(*attn_mask, seq_q, seq_k);

    // Route B: copy into the shape-keyed stable slot so graph-mode DMA can
    // bake its address at BUILD and have it stay valid across REPLAYs. Publish
    // directly from CPU: a temporary device upload followed by host-backed device
    // copy_ adds an unnecessary allocation and two cache-maintenance passes.
    at::Tensor slot = stable_cache.stable_slot(
        seq_q, mask_padded.size(1), mask_padded.options().device(at::kPrivateUse1), ordinal);
    slot.copy_(mask_padded);
    return {slot, 4};  // MASK_2D, stable slot
}

// device side: DMA prepared mask DDR → SPM (batch-compatible).
void sdpa_dma_mask_to_spm(
    const PreparedMask &mask,
    uint32_t sdpa_mask_off,
    int64_t seq_q, int64_t seq_k,
    int num_cores)
{
    if (mask.mask_type <= 1) return;  // MASK_NONE or MASK_LTM — no DMA needed

    int64_t seq_k_v16 = CeilDiv(seq_k, (int64_t)16);
    int64_t mask_elements = seq_q * seq_k_v16 * 16;

    // mask.ddr_tensor is the stable Route B slot (see sdpa_prepare_mask);
    // DMA src is safe to bake. Graph-mode broadcast DMA replaces the legacy
    // rpu_launch_ddr2spm_multicore here — eliminates the last graph-mode
    // legacy launcher in src/ops/.
    if (graph_dma::active()) {
        rpu_launch_ddr_broadcast_spm_dma(
            mask.ddr_tensor, /*src_offset_elements=*/0, mask_elements,
            SPM_ALLOC.addr(0, sdpa_mask_off), num_cores);
    } else {
        // Immediate-mode fallback for non-graph callers (currently none — the
        // helper is only reached from build_layer_subgraph). Kept for safety.
        rpu_launch_ddr_broadcast_spm_dma_immediate(
            mask.ddr_tensor.data_ptr<c10::Half>(), mask_elements,
            SPM_ALLOC.addr(0, sdpa_mask_off), num_cores);
    }
}

// Legacy combined API: prepare + DMA in one call.
int sdpa_load_mask_to_spm(
    const c10::optional<at::Tensor> &attn_mask,
    bool is_causal,
    uint32_t sdpa_mask_off,
    int64_t seq_q, int64_t seq_k,
    int num_cores,
    at::Tensor* out_mask_ddr_ref,
    SdpaStableMaskCache& stable_cache)
{
    if (out_mask_ddr_ref) *out_mask_ddr_ref = at::Tensor();

    PreparedMask prepared = sdpa_prepare_mask(
        attn_mask, is_causal, seq_q, seq_k, stable_cache);

    if (out_mask_ddr_ref && prepared.ddr_tensor.defined()) {
        *out_mask_ddr_ref = prepared.ddr_tensor;
    }

    sdpa_dma_mask_to_spm(prepared, sdpa_mask_off, seq_q, seq_k, num_cores);
    return prepared.mask_type;
}

// ============================================================================
// Pi0.5 Action split-KV attention (rhino-ops rasm/pi05_split_kv_attn)
// ============================================================================
// The Action expert has one KV head over eight Q heads, so the ordinary kernel
// walks every key block once per core with only T query rows. Here groups of
// four consecutive cores pack their four heads into 4T rows and split the key
// blocks between them; each core then merges its own head's four partials.
// The tile rule is the one rhino-ops cmodel/check.py tested and the board
// measured: the fewest KV steps over the share first, then the fewest row
// blocks, then the least row padding.

namespace {

constexpr int64_t kSplitKvGroup = 4;
constexpr int64_t kSplitKvHeadDim = 256;
constexpr int64_t kSplitKvEntries = kSplitKvHeadDim / V16_NUM;  // 32-byte entries per row
constexpr int64_t kSplitKvPackRows = 16;   // rows per warp of the pack kernel
constexpr int64_t kSplitKvMergeRows = 16;  // rows per warp of the merge kernel
// The RoPE pack: 16 rows per warp, and the packed head stride T * 512 bytes is
// a 16-bit register.
constexpr int64_t kSplitKvRopeWarpRows = 16;
constexpr int64_t kSplitKvRopeMaxRows = UINT16_MAX / (kSplitKvHeadDim * 2);
constexpr float kSplitKvMaskFloor = -30000.0f;

// The SDPA VLM estimate of a type-4 mask flash-attn tile (tn = 16 entries).
int64_t split_kv_vlm_need(int64_t tm, int64_t tk) {
  const int64_t tn = kSplitKvEntries;
  return std::max({tn * tm, kSplitKvEntries * tm, tm * tk}) + kSplitKvEntries * tk +
         tm * std::max(tk, tn) + tm * 4 + 8;
}

bool split_kv_fits(int64_t tm, int64_t tk) {
  return tm * kSplitKvEntries * tk <= 1024 && tk <= 16 && split_kv_vlm_need(tm, tk) <= 400;
}

// The plan's tile_m and grid suit the DDR kernel, whose warp 0 alone reads the
// cache and hands each step's tiles to the other warps through TMP and four
// core barriers. The SPM-share siblings have neither: every warp reads its own
// K and V tiles from the share. So they split the plan's padded rows over
// more warps, the smallest tile_m whose grid of at most eight blocks covers
// exactly rows_pad. tile_k (each row's arithmetic) and rows_pad (the mask,
// packed-row and partial layouts) stay the plan's. On R1 hardware, bare metal,
// pack + attention + merge at R = position / 16, the same bytes: H10 tile_m
// 3 -> 1 (grid 1 -> 3) 5,438 -> 4,990 cycles, H50 5 -> 3 (grid 3 -> 5)
// 9,500 -> 8,966.
std::pair<int64_t, int64_t> split_kv_share_tiles(const Pi05SplitKvPlan& p) {
  for (int64_t m = 1; m < p.tile_m_v16; ++m) {
    const int64_t gx = CeilDiv(p.rows, 16 * m);
    if (gx <= 8 && gx * 16 * m == p.rows_pad && split_kv_fits(m, p.tile_k_v16)) return {m, gx};
  }
  return {p.tile_m_v16, p.grid_x};
}

// First key block of member k's share of its group.
int64_t split_kv_start(const Pi05SplitKvPlan& p, int64_t k) {
  const int64_t base = p.blocks / p.group, extra = p.blocks % p.group;
  return k * base + std::min(k, extra);
}

void split_kv_set_scm(Kernel_t* kernel, const std::array<uint32_t, 4>& offsets,
                      const std::array<bool, 4>& used, const char* what) {
  for (int c = 0; c < MAX_CORES; ++c) {
    const int base = SCM_REG_OFFSET + c * SCM_PARAMS_PER_CORE;
    for (int k = 0; k < 4; ++k) {
      rpu_set_scm_u32_checked(
          kernel, base + 2 * k, used[k] ? SPM_ALLOC.addr(c, offsets[k]) >> 5 : 0, what);
    }
  }
}

void split_kv_enqueue(Kernel_t* kernel, uint16_t grid_x, uint16_t grid_y = 1) {
  static const std::vector<uint8_t> cores{0, 1, 2, 3, 4, 5, 6, 7};
  auto* wq = GET_QUEUE(MAX_CORES);
  wq->set_broadcast_mode(true);
  wq->set_flush_icache(false);
  wq->enqueu_kernel(*kernel, {grid_x, grid_y, 1}, cores);
}

// pi05_split_kv_pack_q over already rotated query rows.
void split_kv_launch_pack(uint32_t q_off, uint32_t qp_off, const Pi05SplitKvPlan& plan) {
  const int64_t T = plan.seq_q;
  Kernel_t* pack = GET_KERNEL(KernelId::PI05_SPLIT_KV_PACK_Q);
  TORCH_CHECK(pack != nullptr, "Pi0.5 split-KV pack payload is absent");
  pack->reset_regs();
  pack->set_regs(0, (uint16_t)T);
  pack->set_regs(1, (uint16_t)kSplitKvPackRows);
  pack->set_regs(2, (uint16_t)plan.group);
  pack->set_regs(3, (uint16_t)kSplitKvEntries);
  pack->set_regs(4, (uint16_t)(T * kSplitKvEntries));
  const uint16_t pack_grid = (uint16_t)(plan.group * CeilDiv(T, kSplitKvPackRows));
  pack->set_regs(64, pack_grid);
  pack->set_regs(65, (uint16_t)1);
  pack->set_regs(66, (uint16_t)1);
  split_kv_set_scm(pack, {q_off, qp_off, 0, 0}, {true, true, false, false},
                   "Pi0.5 split-KV pack");
  split_kv_enqueue(pack, pack_grid);
}

// pi05_split_kv_pack_q_rope_g4d256 (rhino-ops rasm/pi05_split_kv_rope_pack):
// the published partial_mrope register image over the group's four heads
// (params 0..3 cos/sin, 8/9 position, 10 = D, 11 = group, 12 = T, 13 = rotary,
// 14 = tables in DDR), the local packed block in params 6/7 and every core's
// query rows in SCM slot 0; a warp covers 16 rows. With rows_off >= 0 the
// tables are this core's SPM rows instead: params 0..3 their byte addresses,
// position 0, param 14 = 0, and no DDR operand.
constexpr int64_t kSplitKvRopeRowBytes = kSplitKvHeadDim / 2 * 2;  // one FP16 cos or sin row
constexpr int64_t kSplitKvRopeGridX = 2;  // ceil(D / 2 / 64): the D halves of a row

// A launch of its own takes the rows pi05_split_kv_rope_pack admits; another
// payload carrying the pack's code (`fused`) any row count the pack handles.
void split_kv_check_rope_pack(const Pi05SplitKvPlan& plan, int64_t rows_off, bool fused) {
  const int64_t T = plan.seq_q;
  TORCH_CHECK((fused ? T > 0 && T <= kSplitKvRopeMaxRows : pi05_split_kv_rope_pack(T)) &&
                  plan.group == kSplitKvGroup,
              "Pi0.5 split-KV RoPE pack payload is absent or the rows are not admitted");
  TORCH_CHECK(rows_off < 0 || (rows_off % 32 == 0 &&
                               rows_off + 2 * T * kSplitKvRopeRowBytes <=
                                   (int64_t)SpmAllocator::SPM_USABLE),
              "Pi0.5 split-KV RoPE pack: SPM rows at ", rows_off,
              " must be 32-byte aligned and hold ", 2 * T, " rows");
}

// The register image without the grid: params 0..3 when the tables are SPM
// rows (the DDR writer fills them otherwise), 6..14 and SCM slot 0.
void split_kv_write_rope_pack_regs(Kernel_t* pack, int64_t position, int64_t rows_off,
                                   uint32_t q_off, uint32_t qp_off, const Pi05SplitKvPlan& plan) {
  const int64_t T = plan.seq_q;
  const bool spm_rows = rows_off >= 0;
  if (spm_rows) {
    const uint32_t cos_rows = SPM_ALLOC.addr(0, (uint32_t)rows_off);
    const uint32_t sin_rows = cos_rows + (uint32_t)(T * kSplitKvRopeRowBytes);
    pack->set_regs(0, (uint16_t)(cos_rows & 0xFFFF));
    pack->set_regs(1, (uint16_t)(cos_rows >> 16));
    pack->set_regs(2, (uint16_t)(sin_rows & 0xFFFF));
    pack->set_regs(3, (uint16_t)(sin_rows >> 16));
  }
  const uint32_t qp = SPM_ALLOC.addr(0, qp_off);
  const uint64_t pos = spm_rows ? 0 : static_cast<uint64_t>(position);
  pack->set_regs(6, (uint16_t)(qp & 0xFFFF));
  pack->set_regs(7, (uint16_t)(qp >> 16));
  pack->set_regs(8, (uint16_t)(pos & 0xFFFF));
  pack->set_regs(9, (uint16_t)(pos >> 16));
  pack->set_regs(10, (uint16_t)kSplitKvHeadDim);
  pack->set_regs(11, (uint16_t)plan.group);
  pack->set_regs(12, (uint16_t)T);
  pack->set_regs(13, (uint16_t)kSplitKvHeadDim);
  pack->set_regs(14, (uint16_t)(spm_rows ? 0 : 1));
  split_kv_set_scm(pack, {q_off, 0, 0, 0}, {true, false, false, false},
                   "Pi0.5 split-KV RoPE pack");
}

void split_kv_launch_rope_pack(const at::Tensor& cos, const at::Tensor& sin, int64_t position,
                               int64_t rows_off, uint32_t q_off, uint32_t qp_off,
                               const Pi05SplitKvPlan& plan) {
  const int64_t T = plan.seq_q;
  split_kv_check_rope_pack(plan, rows_off, /*fused=*/false);
  const bool spm_rows = rows_off >= 0;
  TORCH_CHECK(!spm_rows || (!cos.defined() && !sin.defined()),
              "Pi0.5 split-KV RoPE pack: SPM rows come without DDR tables");
  const auto is_table = [](const at::Tensor& t) {
    return t.defined() && t.device().type() == c10::DeviceType::PrivateUse1 &&
           t.scalar_type() == at::kHalf && t.layout() == c10::Layout::Strided &&
           t.is_contiguous() && t.dim() == 2 && t.size(1) == kSplitKvHeadDim / 2;
  };
  TORCH_CHECK(spm_rows ||
                  (is_table(cos) && is_table(sin) && cos.sizes() == sin.sizes() && position >= 0 &&
                   position <= cos.size(0) && T <= cos.size(0) - position &&
                   position <= UINT32_MAX),
              "Pi0.5 split-KV RoPE pack: cos/sin must be matching contiguous FP16 RPU tables "
              "[rows, 128] covering positions [", position, ", ", position + T, ")");
  std::optional<GraphKernelRegisterWriter> register_writer;
  if (!spm_rows) {
    register_writer.emplace(RpuKernelGraph::active().stage_kernel_ddr_registers(
        KernelId::PI05_SPLIT_KV_PACK_Q_ROPE,
        std::vector<GraphDdrRegisterOperandSpec>{
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{GraphDdrRegisterRole::SemanticInput,
                                    GraphDdrRegisterAccess::Read,
                                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 0, 1},
                cos},
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{GraphDdrRegisterRole::SemanticInput,
                                    GraphDdrRegisterAccess::Read,
                                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
                sin}}));
    rpu_ddr_flush(cos.data_ptr<c10::Half>());
    rpu_ddr_flush(sin.data_ptr<c10::Half>());
  }
  Kernel_t* pack = GET_KERNEL(KernelId::PI05_SPLIT_KV_PACK_Q_ROPE);
  TORCH_CHECK(pack != nullptr, "Pi0.5 split-KV RoPE pack payload is absent");
  pack->reset_regs();
  split_kv_write_rope_pack_regs(pack, position, rows_off, q_off, qp_off, plan);
  const uint16_t grid_x = (uint16_t)kSplitKvRopeGridX;
  const uint16_t grid_y = (uint16_t)CeilDiv(T, kSplitKvRopeWarpRows);
  pack->set_regs(64, grid_x);
  pack->set_regs(65, grid_y);
  pack->set_regs(66, (uint16_t)1);
  if (register_writer) register_writer->write(*pack);
  split_kv_enqueue(pack, grid_x, grid_y);
}

// The merge's params 0, 1, 2, 3, 6, 8 and 9 for `plan`, in that order.
std::array<uint16_t, 7> split_kv_merge_params(const Pi05SplitKvPlan& plan) {
  const auto T = static_cast<uint16_t>(plan.seq_q);
  return {T, static_cast<uint16_t>(plan.merge_rows), static_cast<uint16_t>(plan.group),
          static_cast<uint16_t>(kSplitKvEntries), T, static_cast<uint16_t>(kSplitKvEntries),
          static_cast<uint16_t>(plan.rows_pad)};
}
constexpr std::array<int, 7> kSplitKvMergeParams{0, 1, 2, 3, 6, 8, 9};

// The rotated-load sibling gives the same bytes; its cores read different
// cores at every load step instead of all reading the same. It is taken for
// chunks of more than 16 rows: at H10 (one merge block per core) the board
// measured no gain (merge +0.005 ms per denoise), at H50 -0.11 ms. Its
// overlapped sibling, also same bytes, computes the weights while the rows
// load; the board measured -111 cycles per merge at H50 and +116 at H10.
KernelId split_kv_merge_id(int64_t seq_q) {
  const auto& cache = KernelCache::instance();
  return seq_q <= 16 ? KernelId::PI05_SPLIT_KV_MERGE
      : cache.has_loaded(KernelId::PI05_SPLIT_KV_MERGE_ROT_OV) ? KernelId::PI05_SPLIT_KV_MERGE_ROT_OV
      : cache.has_loaded(KernelId::PI05_SPLIT_KV_MERGE_ROT)    ? KernelId::PI05_SPLIT_KV_MERGE_ROT
                                                              : KernelId::PI05_SPLIT_KV_MERGE;
}

uint16_t split_kv_merge_grid(const Pi05SplitKvPlan& plan) {
  return static_cast<uint16_t>(CeilDiv(plan.seq_q, plan.merge_rows));
}

// SCM slots 0, 1 and 3: the partial rows, m (and l) behind them, the output.
void split_kv_set_merge_scm(Kernel_t* kernel, const Pi05SplitKvMerge& merge) {
  const uint32_t m_off =
      merge.part_off + static_cast<uint32_t>(merge.plan.rows * kSplitKvHeadDim * 2);
  split_kv_set_scm(kernel, {merge.part_off, m_off, 0, merge.output_off},
                   {true, true, false, true}, "Pi0.5 split-KV merge");
}

}  // namespace

void pi05_split_kv_check_rope_pack_spm(const Pi05SplitKvPlan& plan, int64_t rope_rows_off) {
  TORCH_CHECK(rope_rows_off >= 0, "Pi0.5 split-KV RoPE pack in a fused launch needs SPM rows");
  split_kv_check_rope_pack(plan, rope_rows_off, /*fused=*/true);
}

uint16_t pi05_split_kv_write_rope_pack_spm_regs(Kernel_t* kernel, int64_t rope_rows_off,
                                                uint32_t q_off, uint32_t qp_off,
                                                const Pi05SplitKvPlan& plan) {
  pi05_split_kv_check_rope_pack_spm(plan, rope_rows_off);
  split_kv_write_rope_pack_regs(kernel, 0, rope_rows_off, q_off, qp_off, plan);
  return (uint16_t)(kSplitKvRopeGridX * CeilDiv(plan.seq_q, kSplitKvRopeWarpRows));
}

void rpu_launch_pi05_split_kv_merge(const Pi05SplitKvMerge& m) {
  const KernelId id = split_kv_merge_id(m.plan.seq_q);
  TORCH_CHECK(m.rot_ov == (id == KernelId::PI05_SPLIT_KV_MERGE_ROT_OV),
              "Pi0.5 split-KV merge drifted from the one its attention chose");
  Kernel_t* merge = GET_KERNEL(id);
  TORCH_CHECK(merge != nullptr, "Pi0.5 split-KV merge payload is absent");
  merge->reset_regs();
  const auto values = split_kv_merge_params(m.plan);
  for (size_t i = 0; i < values.size(); ++i) merge->set_regs(kSplitKvMergeParams[i], values[i]);
  const uint16_t merge_grid = split_kv_merge_grid(m.plan);
  merge->set_regs(64, merge_grid);
  merge->set_regs(65, (uint16_t)1);
  merge->set_regs(66, (uint16_t)1);
  split_kv_set_merge_scm(merge, m);
  split_kv_enqueue(merge, merge_grid);
}

uint16_t pi05_split_kv_write_merge_o_regs(Kernel_t* kernel, const Pi05SplitKvMerge& m) {
  const auto values = split_kv_merge_params(m.plan);
  for (size_t i = 0; i < values.size(); ++i) kernel->set_regs(28 + (int)i, values[i]);
  const uint16_t merge_grid = split_kv_merge_grid(m.plan);
  kernel->set_regs(35, merge_grid);
  split_kv_set_merge_scm(kernel, m);
  return merge_grid;
}

int64_t Pi05SplitKvPlan::qp_bytes() const {
  return rows_pad * kSplitKvHeadDim * 2;  // the attention reads whole row tiles
}
int64_t Pi05SplitKvPlan::part_bytes() const {
  // m sits right behind the rows and l rows_pad entries behind m; the merge
  // reads 16 entries past the last row's l.
  return rows * kSplitKvHeadDim * 2 + (2 * rows_pad + 16) * 32;
}
int64_t Pi05SplitKvPlan::tmp_bytes() const {
  return kSplitKvEntries * tile_k_v16 * 16 * grid_x * 32;
}
int64_t Pi05SplitKvPlan::mask_bytes() const {
  return rows_pad * 16 * width * 2;
}
int64_t Pi05SplitKvPlan::share_bytes() const {
  return 2 * width * 16 * kSplitKvHeadDim * 2;  // [K | V][width blocks][16 keys x 256]
}

std::optional<Pi05SplitKvPlan> pi05_split_kv_attention_plan(
    int64_t seq_q, int64_t kv_seq_len, int64_t num_q_heads,
    int64_t kv_source_heads, int64_t head_dim, int num_cores) {
  if (num_q_heads != MAX_CORES || kv_source_heads != 1 || head_dim != kSplitKvHeadDim ||
      num_cores != MAX_CORES || seq_q <= 0 || kv_seq_len <= 0) {
    return std::nullopt;
  }
  auto& cache = KernelCache::instance();
  if (!cache.has_loaded(KernelId::PI05_SPLIT_KV_PACK_Q) ||
      !cache.has_loaded(KernelId::PI05_FLASH_ATTN_SPLIT_KV_DP) ||
      !cache.has_loaded(KernelId::PI05_SPLIT_KV_MERGE)) {
    return std::nullopt;
  }
  Pi05SplitKvPlan p;
  p.seq_q = seq_q;
  p.seq_k = kv_seq_len;
  p.group = kSplitKvGroup;
  p.rows = p.group * seq_q;
  p.blocks = CeilDiv(kv_seq_len, (int64_t)16);
  // The kernel's causal offset seq_k - rows must stay nonnegative, every member
  // needs a key block, and rows * 16 (param34) must fit a register.
  if (p.rows > kv_seq_len || p.blocks < p.group ||
      p.rows * kSplitKvEntries > UINT16_MAX || p.blocks > UINT16_MAX) {
    return std::nullopt;
  }
  p.width = CeilDiv(p.blocks, p.group);
  std::tuple<int64_t, int64_t, int64_t> best{};
  const int64_t max_m = std::min<int64_t>(11, CeilDiv(p.rows, (int64_t)16));
  for (int64_t m = 1; m <= max_m; ++m) {
    int64_t k = std::min<int64_t>(p.width, 16);
    while (k > 0 && !split_kv_fits(m, k)) --k;
    if (k == 0) continue;
    const int64_t gx = CeilDiv(p.rows, 16 * m);
    const auto key = std::make_tuple(CeilDiv(p.width, k), gx, gx * 16 * m);
    if (p.tile_m_v16 == 0 || key < best) {
      best = key;
      p.tile_m_v16 = m;
      p.tile_k_v16 = k;
      p.grid_x = gx;
      p.rows_pad = gx * 16 * m;
    }
  }
  if (p.tile_m_v16 == 0) return std::nullopt;
  // As for the pack: the rows spread over the eight warps, the bytes unchanged.
  p.merge_rows = kSplitKvMergeRows;
  return p;
}

void pi05_split_kv_prepare_mask(const at::Tensor& attn_mask,
                                const Pi05SplitKvPlan& plan, at::Tensor& slot) {
  const int64_t cols = 16 * plan.width;
  TORCH_CHECK(slot.defined() && slot.device().type() == at::kPrivateUse1 &&
                  slot.scalar_type() == at::kHalf && slot.is_contiguous() &&
                  slot.dim() == 3 && slot.size(0) == MAX_CORES &&
                  slot.size(1) == plan.rows_pad && slot.size(2) == cols,
              "Pi0.5 split-KV mask slot must be [8, rows_pad, 16 * width] FP16 on the device");
  // -inf (masked keys and the alignment padding) becomes -30000, finite.
  const at::Tensor padded =
      sdpa_mask_padded_cpu(attn_mask, plan.seq_q, plan.seq_k).clamp_min(kSplitKvMaskFloor);
  at::Tensor host = at::zeros({MAX_CORES, plan.rows_pad, cols}, padded.options());
  for (int64_t k = 0; k < plan.group; ++k) {
    // Member k's key columns; past the last block nothing is visible.
    const int64_t first = 16 * split_kv_start(plan, k);
    const int64_t have = std::min(cols, padded.size(1) - first);
    at::Tensor share = at::full({plan.seq_q, cols}, kSplitKvMaskFloor, padded.options());
    share.narrow(1, 0, have).copy_(padded.narrow(1, first, have));
    for (int64_t c = k; c < MAX_CORES; c += plan.group) {
      for (int64_t h = 0; h < plan.group; ++h) {
        host[c].narrow(0, h * plan.seq_q, plan.seq_q).copy_(share);
      }
    }
  }
  slot.copy_(host);
}

bool pi05_split_kv_rope_pack(int64_t seq_q) {
  // The fused pack absorbs the query RoPE into its own grid of
  // group * ceil(T / kSplitKvPackRows) blocks. At T <= kSplitKvPackRows that is
  // four blocks, so the RoPE it takes over would run on half of the eight
  // cores, while the partial_mrope kernel it replaces spreads over all of
  // them. Measured on R1 hardware with only the expansion REF swapped: at
  // T = 10 the fused kernel costs 1.033 ms per denoise replay against 0.890 ms
  // for the separate pack plus RoPE, +0.099 ms end to end, and at T = 50 it
  // costs 1.744 ms against 1.992 ms, -0.328 ms end to end. So admit it only
  // once its grid covers every core.
  return seq_q > 0 && seq_q <= kSplitKvRopeMaxRows &&
         kSplitKvGroup * CeilDiv(seq_q, kSplitKvPackRows) >= MAX_CORES &&
         KernelCache::instance().has_loaded(KernelId::PI05_SPLIT_KV_PACK_Q_ROPE);
}

bool pi05_split_kv_spm_share() {
  return KernelCache::instance().has_loaded(KernelId::PI05_FLASH_ATTN_SPLIT_KV_SPM_DP);
}

bool pi05_split_kv_rotated(int64_t seq_q) {
  // All or none: a writer that skips slots needs every attention launch of the
  // forward, with or without the share, to read the rotated slots. A chunk of
  // at most 16 rows writes one cache block, whose read slots sit on two DDR
  // controllers that take as many stores as with every slot written; there
  // the rotation only adds the readers' slot arithmetic (board A/B with the
  // rotation at every chunk size: H10 +0.075 ms, H50 -0.63 ms).
  auto& cache = KernelCache::instance();
  return seq_q > 16 && cache.has_loaded(KernelId::PI05_FLASH_ATTN_SPLIT_KV_DP_ROT) &&
         cache.has_loaded(KernelId::PI05_FLASH_ATTN_SPLIT_KV_SPM_DP_ROT) &&
         cache.has_loaded(KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT);
}

uint16_t pi05_split_kv_rotated_skip_mask(const Pi05SplitKvPlan& plan,
                                         int64_t first_block, int64_t end_block) {
  TORCH_CHECK(plan.group > 0 && 0 <= first_block && first_block < end_block &&
                  end_block <= plan.blocks,
              "Pi0.5 split-KV rotated writer: blocks [", first_block, ", ", end_block,
              ") outside the plan's ", plan.blocks);
  // Core c reads member c % group's blocks [S, E), block b from slot
  // (c + b - S) mod 8; the writer stores slot h of block b unless bit
  // (h - b) mod 8 is set. A core whose range holds a written block therefore
  // needs bit (c - S) mod 8 clear, for every block it reads; every other slot
  // may be skipped.
  uint16_t mask = 0xff;
  for (int c = 0; c < MAX_CORES; ++c) {
    const int64_t k = c % plan.group;
    const int64_t start = split_kv_start(plan, k);
    if (first_block < split_kv_start(plan, k + 1) && start < end_block) {
      mask &= static_cast<uint16_t>(~(1u << ((c - start) & 7)));
    }
  }
  return mask;
}

bool rpu_launch_pi05_split_kv_attention(
    const at::Tensor& key, const at::Tensor& value,
    uint32_t q_off, uint32_t output_off, uint32_t qp_off, uint32_t part_off,
    uint32_t tmp_off, uint32_t mask_off, const Pi05SplitKvPlan& plan,
    const at::Tensor& rope_cos, const at::Tensor& rope_sin, int64_t rope_position,
    int64_t share_off, int64_t refetch_block, int64_t rope_rows_off, bool q_packed,
    Pi05SplitKvMerge* defer_merge) {
  const auto check = pi05_split_kv_attention_plan(
      plan.seq_q, plan.seq_k, MAX_CORES, 1, kSplitKvHeadDim, MAX_CORES);
  TORCH_CHECK(check && check->rows == plan.rows && check->width == plan.width &&
                  check->tile_m_v16 == plan.tile_m_v16 &&
                  check->tile_k_v16 == plan.tile_k_v16 &&
                  check->grid_x == plan.grid_x && check->rows_pad == plan.rows_pad,
              "Pi0.5 split-KV attention plan drifted from its layout");
  const int64_t T = plan.seq_q, rows = plan.rows, seq_k = plan.seq_k;
  const int64_t m_units = rows * kSplitKvEntries;  // part slot -> m, 32-byte units
  // Before any launch, so a rejected share leaves nothing enqueued.
  const bool share = share_off >= 0;
  TORCH_CHECK(!share || (pi05_split_kv_spm_share() && share_off % 32 == 0 && qp_off % 32 == 0 &&
                         share_off + plan.share_bytes() <= (int64_t)SpmAllocator::SPM_USABLE &&
                         refetch_block >= 0 && refetch_block <= plan.blocks),
              "Pi0.5 split-KV SPM share: payload absent, share unaligned or R = ",
              refetch_block, " outside [0, ", plan.blocks, "]");

  TORCH_CHECK(!q_packed || (!rope_cos.defined() && !rope_sin.defined() && rope_rows_off < 0),
              "Pi0.5 split-KV attention: packed query rows come without RoPE arguments");

  // 1. Pack: every member gets its group's query rows, head-major; rotated
  // here when the caller left them unrotated. The direct cache's fused launch
  // has already packed them when q_packed.
  if (q_packed) {
    // Nothing to launch.
  } else if (rope_cos.defined() || rope_rows_off >= 0) {
    split_kv_launch_rope_pack(rope_cos, rope_sin, rope_position, rope_rows_off, q_off, qp_off,
                              plan);
  } else {
    split_kv_launch_pack(q_off, qp_off, plan);
  }

  // 2. Attention over this core's key blocks: the SDPA register image for one
  // head per core over the packed rows (K/V caches as the ordinary launcher
  // stages them), plus the group size and the m/l offsets. The SPM-share
  // sibling adds the share (32-byte units from the packed rows) and R.
  // The rotated siblings read block b from slot (c + b - S) mod 8, S the first
  // block of the core's range: the same bytes here, and the only reader of a
  // cache whose writer skipped slots. The late-V share siblings, when loaded,
  // fill K only and copy each warp's first V tile on the first step under its
  // softmax (same registers and bytes; on the board the fill's core barrier
  // must be passed again before any warp reads V from the share).
  const bool rotated = pi05_split_kv_rotated(T);
  const auto late_v = [](KernelId base, KernelId sibling) {
    return KernelCache::instance().has_loaded(sibling) ? sibling : base;
  };
  const KernelId attn_id =
      share ? (rotated ? late_v(KernelId::PI05_FLASH_ATTN_SPLIT_KV_SPM_DP_ROT,
                                KernelId::PI05_FLASH_ATTN_SPLIT_KV_SPM_DP_ROT_LATE_V)
                       : late_v(KernelId::PI05_FLASH_ATTN_SPLIT_KV_SPM_DP,
                                KernelId::PI05_FLASH_ATTN_SPLIT_KV_SPM_DP_LATE_V))
            : (rotated ? KernelId::PI05_FLASH_ATTN_SPLIT_KV_DP_ROT
                       : KernelId::PI05_FLASH_ATTN_SPLIT_KV_DP);
  auto register_writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      attn_id,
      std::vector<GraphDdrRegisterOperandSpec>{
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{GraphDdrRegisterRole::KeyCache,
                                  GraphDdrRegisterAccess::Read,
                                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 50, 51},
              key},
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{GraphDdrRegisterRole::ValueCache,
                                  GraphDdrRegisterAccess::Read,
                                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 52, 53},
              value}});
  rpu_ddr_flush(key.data_ptr<c10::Half>());
  rpu_ddr_flush(value.data_ptr<c10::Half>());
  const auto [tile_m, grid_x] =
      share ? split_kv_share_tiles(plan) : std::make_pair(plan.tile_m_v16, plan.grid_x);
  const int64_t acc = seq_k - rows;
  const int64_t seq_k_v16 = CeilDiv(seq_k, (int64_t)16);
  const int64_t row_size = 512, page_size = row_size * MAX_CORES;
  const int64_t chapter_size = page_size * kSplitKvEntries;
  const uint32_t scale_u32 =
      float32_to_uint32(1.0f / std::sqrt(static_cast<float>(kSplitKvHeadDim)));
  Kernel_t* attn = GET_KERNEL(attn_id);
  TORCH_CHECK(attn != nullptr, "Pi0.5 split-KV attention payload is absent");
  attn->reset_regs();
  const std::pair<int, int64_t> regs[] = {
      {0, seq_k & 0xFFFF}, {1, seq_k >> 16}, {2, rows}, {3, CeilDiv(rows, (int64_t)16)},
      {4, seq_k_v16}, {5, CeilDiv(acc, (int64_t)16)},
      {6, MAX_CORES}, {7, MAX_CORES}, {8, 1}, {9, 1},
      {10, kSplitKvHeadDim}, {11, kSplitKvHeadDim}, {12, kSplitKvHeadDim},
      {13, kSplitKvEntries}, {14, kSplitKvEntries}, {15, kSplitKvEntries},
      {16, acc & 0xFFFF}, {17, acc >> 16}, {18, MAX_CORES}, {19, 1},
      {20, chapter_size >> 8}, {21, chapter_size >> 8}, {22, page_size >> 8},
      {23, row_size >> 8}, {24, MAX_CORES}, {25, 1}, {26, 16}, {27, kSplitKvEntries},
      {28, 16}, {29, seq_k_v16}, {30, 16}, {31, seq_k_v16}, {32, 1},
      {33, plan.group}, {34, m_units}, {35, plan.rows_pad},
      {40, scale_u32 & 0xFFFF}, {41, scale_u32 >> 16}, {42, 1},
      {56, MASK_2D},
      {58, tile_m * 16}, {59, kSplitKvEntries * 16}, {60, plan.tile_k_v16 * 16},
      {61, tile_m}, {62, kSplitKvEntries}, {63, plan.tile_k_v16},
      {64, grid_x}, {65, 1}, {66, 1},
  };
  for (const auto& [idx, v] : regs) {
    TORCH_CHECK(v >= 0 && v <= UINT16_MAX, "Pi0.5 split-KV param", idx, " out of range: ", v);
    attn->set_regs(idx, (uint16_t)v);
  }
  if (share) {
    // Two's complement: the share may sit in front of the packed rows.
    const uint32_t delta = static_cast<uint32_t>((share_off >> 5) - (int64_t)(qp_off >> 5));
    attn->set_regs(36, (uint16_t)(delta & 0xFFFF));
    attn->set_regs(37, (uint16_t)(delta >> 16));
    attn->set_regs(38, (uint16_t)refetch_block);
  }
  split_kv_set_scm(attn, {qp_off, tmp_off, mask_off, part_off}, {true, true, true, true},
                   "Pi0.5 split-KV attention");
  register_writer.write(*attn);
  split_kv_enqueue(attn, (uint16_t)grid_x);

  // 3. Merge the group's partials of this core's head into its output rows.
  const KernelId merge_id = split_kv_merge_id(T);
  const Pi05SplitKvMerge m{plan, part_off, output_off,
                           merge_id == KernelId::PI05_SPLIT_KV_MERGE_ROT_OV};
  // The O linear reads the merged rows next; its merge + O sibling runs the
  // merge on extra blocks of the same launch (rasm/pi05_split_kv_merge_o_sspm).
  const KernelId merge_o =
      m.rot_ov ? KernelId::PI05_SPLIT_KV_MERGE_ROT_OV_O_NVFP4_ACC16_M64N64K128_DB_SSPM
               : KernelId::PI05_SPLIT_KV_MERGE_O_NVFP4_ACC16_M64N64K128_DB_SSPM;
  if (defer_merge != nullptr && merge_id != KernelId::PI05_SPLIT_KV_MERGE_ROT &&
      KernelCache::instance().has_loaded(merge_o)) {
    *defer_merge = m;
    return true;
  }
  rpu_launch_pi05_split_kv_merge(m);
  return false;
}
