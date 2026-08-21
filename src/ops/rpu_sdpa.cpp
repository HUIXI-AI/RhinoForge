// Native FP16 SDPA for RPU unified layout
// [batch, seq_len, num_heads, head_dim].
#include <limits>
#include "rhino_launch_buffer.h"

#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"
#include "rhino_launch_file.h"
#include "rpu_ops.h"
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
// SDPA SPM unified launch geometry:
// tile_m_v16: sQry<=128 ? CeilDiv(sQry,16) : 8
// tile_k_v16: from sdpa_compute_tiling (= tile_m_v16 initially, may VLM-shrink)
// grid_dim_x: CeilDiv(sQry, tile_m)  (seq tiles)
// grid_dim_y: num_heads_per_core      (heads)
// ============================================================================

void rpu_launch_sdpa_spm_unified_kernel_v2(
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
    bool use_16b)
{
  const KernelId sdpa_kernel_id = use_16b
      ? KernelId::SDPA_FLASH_ATTN_SPM_16B
      : KernelId::SDPA_FLASH_ATTN_SPM;

  int64_t core_num = num_cores;  // real cores for dispatch and SPM addressing

  // Virtual tp for parameter computation (cache layout, head distribution, registers).
  // Keep virtual_attn_tp=8 so cache strides match the tp=8 cache allocation,
  // even when fewer cores actually run the kernel.
  int64_t vtp = (virtual_num_cores > 0) ? virtual_num_cores : core_num;

  // Compute "virtual" total head counts as runtime does:
  //   numHead = q_heads_per_core * virtual_tp
  //   nKVHead = kv_heads_per_core * virtual_tp
  // This inflates the totals but per-core values come out correct.
  int64_t q_heads_per_core = num_heads / core_num;
  int64_t kv_heads_per_core = CeilDiv(num_kv_heads, core_num);
  int64_t virtual_num_heads = q_heads_per_core * vtp;
  int64_t virtual_num_kv_heads = kv_heads_per_core * vtp;

  // Validate constraints (on real values)
  TORCH_CHECK(num_heads % core_num == 0, "num_heads must be divisible by core_num (", core_num, ")");
  TORCH_CHECK((num_kv_heads % core_num == 0) || (core_num % num_kv_heads == 0),
              "num_kv_heads must divide core_num or be divisible by core_num");

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
  int64_t tile_m_v16 = t.tile_m_v16;
  int64_t tile_n_v16 = t.tile_n_v16;
  int64_t tile_k_v16 = t.tile_k_v16;

  int64_t tile_m = tile_m_v16 * 16;
  int64_t tile_n = tile_n_v16 * 16;
  int64_t tile_k = tile_k_v16 * 16;

  // Mask type (passed directly, mask data already in SPM if needed)
  AttnMaskType mask_type = static_cast<AttnMaskType>(attn_mask_type);
  bool is_causal = (mask_type == MASK_LTM);

  // Scale factor
  float scale_factor = scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim)));
  uint32_t scale_u32 = float32_to_uint32(scale_factor);

  // Grid dimensions: x=sequence tiles, y=heads, z=batch.
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
      kernel->set_regs(scm_base + 0, (uint16_t)(q_spm_addr_v16 & 0xFFFF));
      kernel->set_regs(scm_base + 1, (uint16_t)(q_spm_addr_v16 >> 16));
      kernel->set_regs(scm_base + 2, (uint16_t)(tmp_addr_v16 & 0xFFFF));
      kernel->set_regs(scm_base + 3, (uint16_t)(tmp_addr_v16 >> 16));
      kernel->set_regs(scm_base + 4, (uint16_t)(mask_addr_v16 & 0xFFFF));
      kernel->set_regs(scm_base + 5, (uint16_t)(mask_addr_v16 >> 16));
      kernel->set_regs(scm_base + 6, (uint16_t)(out_spm_addr_v16 & 0xFFFF));
      kernel->set_regs(scm_base + 7, (uint16_t)(out_spm_addr_v16 >> 16));
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

// =============================================================================
// SDPA Mask: prepare (CPU side) + DMA (RPU side, batch-compatible)
// =============================================================================
//
// Route B mask slot: the prepared mask is `.to(at::kPrivateUse1)`-allocated
// fresh each call (data_ptr unstable across forwards) — graph-mode DMA bakes
// the src addr at BUILD and can't rebind on REPLAY. Each FusedModelBase owns
// one typed cache containing a stable RPU tensor per
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

// CPU side: reduce dims, convert type, pad if needed, move to RPU DDR.
// Skips pad_nd entirely when seq_k is already v16-aligned.
PreparedMask sdpa_prepare_mask(
    const c10::optional<at::Tensor> &attn_mask,
    bool is_causal,
    int64_t seq_q, int64_t seq_k,
    SdpaStableMaskCache& stable_cache,
    int64_t ordinal)
{
    if (is_causal) return {{}, 1};   // MASK_LTM
    if (!attn_mask.has_value() || !attn_mask->defined()) return {{}, 0};  // MASK_NONE

    at::Tensor mask_2d = attn_mask.value();
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
    at::Tensor mask_padded = (seq_k_aligned != seq_k)
        ? at::constant_pad_nd(mask_2d, {0, seq_k_aligned - seq_k},
                              -std::numeric_limits<float>::infinity())
        : mask_2d;

    // Route B: copy into the shape-keyed stable slot so graph-mode DMA can
    // bake its address at BUILD and have it stay valid across REPLAYs.
    at::Tensor mask_rpu_fresh = mask_padded.to(at::kPrivateUse1);
    at::Tensor slot = stable_cache.stable_slot(
        seq_q, seq_k_aligned, mask_rpu_fresh.options(), ordinal);
    slot.copy_(mask_rpu_fresh);
    return {slot, 4};  // MASK_2D, stable slot
}

// RPU side: DMA prepared mask DDR → SPM (batch-compatible).
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
    // DMA src is safe to bake. Graph mode uses broadcast DMA rather than the
    // direct rpu_launch_ddr2spm_multicore path.
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
