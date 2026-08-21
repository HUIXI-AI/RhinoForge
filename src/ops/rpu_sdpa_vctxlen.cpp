// rpu_sdpa_vctxlen.cpp
// SPM-based SDPA kernel wrapper using the vctxlen (variable context length) kernel.
//
// Kernel binary: llm_fp16_32b_prefill_flash_attn_univ_vctxlen
// Register layout and tiling follow the FLASH_ATTN_SPM launch contract.
// All param registers are set (including sKeyV16, sQryAcc, sKeyVx, sValVx).

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include <c10/util/Half.h>
#include <cmath>
#include <cstdint>

using namespace at;
using namespace ::rhino_lkn;

// float32_to_uint32 and SCM_REG_OFFSET defined in rpu_helpers.h
#define SCM_PARAMS_PER_CORE 8  // 4 addresses × 2 uint16 each

namespace {

constexpr int kWallVisionCores = 8;
constexpr int kWallVisionBatch = 3;
constexpr int kWallVisionSeq = 576;
constexpr int kWallVisionHeadsPerCore = 2;
constexpr int kWallVisionHeadDim = 80;
constexpr size_t kWallVisionTensorBytes =
    kWallVisionBatch * kWallVisionSeq * kWallVisionHeadsPerCore *
    kWallVisionHeadDim * sizeof(c10::Half);
constexpr size_t kWallVisionMaskBytes =
    kWallVisionSeq * kWallVisionSeq * sizeof(c10::Half);

uint32_t exact_wall_spm_offset(
    uint32_t addr, size_t bytes, const char* operand) {
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "Wall Vision SISC launcher requires initialized SPM");
    const uint32_t base = SPM_ALLOC.addr(0, 0);
    TORCH_CHECK(addr >= base && (addr & 31u) == 0,
                operand, " must be a 32-byte-aligned core-0 SPM address, got ",
                addr);
    const uint64_t offset = static_cast<uint64_t>(addr) - base;
    TORCH_CHECK(offset + bytes <= SpmAllocator::SPM_USABLE,
                operand, " exceeds one core's SPM range");
    return static_cast<uint32_t>(offset);
}

void set_scm_spm_ptr(
    Kernel_t* kernel, int reg, int core, uint32_t offset) {
    const uint32_t addr_v16 = SPM_ALLOC.addr(core, offset) >> 5;
    kernel->set_regs(reg, static_cast<uint16_t>(addr_v16));
    kernel->set_regs(reg + 1, static_cast<uint16_t>(addr_v16 >> 16));
}

std::vector<uint8_t> exact_wall_cores() {
    std::vector<uint8_t> cores(kWallVisionCores);
    for (int i = 0; i < kWallVisionCores; ++i) {
        cores[i] = static_cast<uint8_t>(i);
    }
    return cores;
}

}  // namespace

void rpu_launch_sdpa_spm_vctxlen_kernel(
    const at::Tensor &key,
    const at::Tensor &value,
    int attn_mask_type,
    std::optional<double> scale,
    uint32_t q_off, uint32_t output_off, uint32_t sdpa_tmp_off, uint32_t sdpa_mask_off,
    int64_t seq_q, int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t kv_seq_len,
    int num_cores,
    int virtual_num_cores)
{
    int64_t core_num = num_cores;
    int64_t vtp = (virtual_num_cores > 0) ? virtual_num_cores : core_num;

    // Virtual head counts (same convention as existing SPM launcher)
    int64_t q_heads_per_core = num_heads / core_num;
    int64_t kv_heads_per_core = CeilDiv(num_kv_heads, core_num);
    int64_t virtual_num_heads = q_heads_per_core * vtp;
    int64_t virtual_num_kv_heads = kv_heads_per_core * vtp;

    TORCH_CHECK(num_heads % core_num == 0,
                "num_heads must be divisible by core_num (", core_num, ")");
    TORCH_CHECK((num_kv_heads % core_num == 0) || (core_num % num_kv_heads == 0),
                "num_kv_heads must divide core_num or be divisible by core_num");

    int64_t num_heads_per_core = q_heads_per_core;
    int64_t nkv_head_per_core = CeilDiv(virtual_num_kv_heads, vtp);
    int64_t gqa_group_size = num_heads_per_core / nkv_head_per_core;

    int64_t nkv_head_chunk = std::min(vtp, virtual_num_kv_heads);
    int64_t nkv_head_vx = CeilDiv(virtual_num_kv_heads, vtp);
    int64_t core_num_per_kv_head = CeilDiv(vtp, virtual_num_kv_heads);

    int64_t head_dim_chunk = 16;
    int64_t head_dim_vx = CeilDiv(head_dim, head_dim_chunk);

    int64_t seq_q_v16 = CeilDiv(seq_q, (int64_t)16);
    int64_t head_dim_v16 = CeilDiv(head_dim, (int64_t)16);

    // sKey-dependent values required by the vctxlen launch contract.
    int64_t seq_k = kv_seq_len;
    int64_t seq_k_v16 = CeilDiv(seq_k, (int64_t)16);
    int64_t seq_q_acc = seq_k - seq_q;
    int64_t seq_q_acc_v16 = (seq_q_acc > 0) ? CeilDiv(seq_q_acc, (int64_t)16) : 0;
    int64_t seq_k_chunk = 16;
    int64_t seq_k_vx = CeilDiv(seq_k, seq_k_chunk);
    int64_t seq_v_chunk = seq_k_chunk;
    int64_t seq_v_vx = seq_k_vx;

    // Launch tiling (same as FLASH_ATTN_SPM).
    SdpaConfig tiling_cfg{SdpaKernelType::FLASH_ATTN_SPM,
                          head_dim, virtual_num_heads, virtual_num_kv_heads,
                          static_cast<int>(vtp), attn_mask_type};
    SdpaTiling tiling = sdpa_compute_tiling(tiling_cfg, seq_q);

    // Scale
    float scale_factor = scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim)));
    uint32_t scale_u32 = float32_to_uint32(scale_factor);

    // Grid dimensions
    uint16_t grid_dim_x = CeilDiv(seq_q, tiling.tile_m);
    uint16_t grid_dim_y = num_heads_per_core;
    uint16_t grid_dim_z = 1;

    sdpa_validate_kernel_call(seq_q, seq_k,
                              tiling.tile_m_v16, tiling.tile_n_v16, tiling.tile_k_v16,
                              tiling.tile_m,
                              grid_dim_x, gqa_group_size,
                              head_dim_v16, head_dim_v16, head_dim_v16,
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

    // DDR K/V addresses (glarge format: >>8)
    c10::Half* k_ptr = key.data_ptr<c10::Half>();
    c10::Half* v_ptr = value.data_ptr<c10::Half>();
    rpu_ddr_flush(k_ptr);
    rpu_ddr_flush(v_ptr);

    uint64_t ddr_k_cache_v128 = RpuGetDevAddr(k_ptr) >> 8;
    uint64_t ddr_v_cache_v128 = RpuGetDevAddr(v_ptr) >> 8;

    // Pre-compute tmp SPM addresses (stack array, core_num always ≤ 8)
    uint32_t tmp_addr_v16_arr[8];
    for (int i = 0; i < core_num; ++i) {
        tmp_addr_v16_arr[i] = SPM_ALLOC.addr(i, sdpa_tmp_off) >> 5;
    }

    // Register setup lambda
    auto setup_regs = [=](Kernel_t* kernel) {
        // reg[0,1]: context_length as uint32 value
        kernel->set_regs(0, (uint16_t)(kv_seq_len & 0xFFFF));
        kernel->set_regs(1, (uint16_t)(kv_seq_len >> 16));

        // reg[2,3]: sQry
        kernel->set_regs(2, (uint16_t)seq_q);
        kernel->set_regs(3, (uint16_t)seq_q_v16);

        // reg[4,5]: sKeyV16, sQryAccV16 (required for vctxlen too)
        kernel->set_regs(4, (uint16_t)seq_k_v16);
        kernel->set_regs(5, (uint16_t)seq_q_acc_v16);

        // reg[6-9]: head counts
        kernel->set_regs(6, (uint16_t)virtual_num_heads);
        kernel->set_regs(7, (uint16_t)virtual_num_kv_heads);
        kernel->set_regs(8, (uint16_t)num_heads_per_core);
        kernel->set_regs(9, (uint16_t)nkv_head_per_core);

        // reg[10-15]: head dimensions
        kernel->set_regs(10, (uint16_t)head_dim);
        kernel->set_regs(11, (uint16_t)head_dim);
        kernel->set_regs(12, (uint16_t)head_dim);
        kernel->set_regs(13, (uint16_t)head_dim_v16);
        kernel->set_regs(14, (uint16_t)head_dim_v16);
        kernel->set_regs(15, (uint16_t)head_dim_v16);

        // reg[16,17]: sQryAcc (required for vctxlen too)
        kernel->set_regs(16, (uint16_t)(seq_q_acc & 0xFFFF));
        kernel->set_regs(17, (uint16_t)(seq_q_acc >> 16));

        // reg[18,19]: coreNum, gqaGroupSize
        kernel->set_regs(18, (uint16_t)vtp);
        kernel->set_regs(19, (uint16_t)gqa_group_size);

        // reg[20-27]: KV cache params
        kernel->set_regs(20, (uint16_t)section_size_v128);
        kernel->set_regs(21, (uint16_t)chapter_size_v128);
        kernel->set_regs(22, (uint16_t)page_size_v128);
        kernel->set_regs(23, (uint16_t)row_size_v128);
        kernel->set_regs(24, (uint16_t)nkv_head_chunk);
        kernel->set_regs(25, (uint16_t)nkv_head_vx);
        kernel->set_regs(26, (uint16_t)head_dim_chunk);
        kernel->set_regs(27, (uint16_t)head_dim_vx);

        // reg[28-31]: sKey/sVal chunk and vx
        kernel->set_regs(28, (uint16_t)seq_k_chunk);
        kernel->set_regs(29, (uint16_t)seq_k_vx);
        kernel->set_regs(30, (uint16_t)seq_v_chunk);
        kernel->set_regs(31, (uint16_t)seq_v_vx);

        // reg[32]: coreNumPerKVHead
        kernel->set_regs(32, (uint16_t)core_num_per_kv_head);

        // reg[40-42]: scale
        kernel->set_regs(40, (uint16_t)(scale_u32 & 0xFFFF));
        kernel->set_regs(41, (uint16_t)(scale_u32 >> 16));
        kernel->set_regs(42, (uint16_t)1);  // has_scale

        // reg[50-53]: K/V cache DDR addresses (glarge)
        kernel->set_regs(50, (uint16_t)(ddr_k_cache_v128 & 0xFFFF));
        kernel->set_regs(51, (uint16_t)(ddr_k_cache_v128 >> 16));
        kernel->set_regs(52, (uint16_t)(ddr_v_cache_v128 & 0xFFFF));
        kernel->set_regs(53, (uint16_t)(ddr_v_cache_v128 >> 16));

        // reg[56]: attn_mask_type (preserved from caller)
        kernel->set_regs(56, (uint16_t)attn_mask_type);

        // reg[58-63]: tiling
        kernel->set_regs(58, (uint16_t)tiling.tile_m);
        kernel->set_regs(59, (uint16_t)tiling.tile_n);
        kernel->set_regs(60, (uint16_t)tiling.tile_k);
        kernel->set_regs(61, (uint16_t)tiling.tile_m_v16);
        kernel->set_regs(62, (uint16_t)tiling.tile_n_v16);
        kernel->set_regs(63, (uint16_t)tiling.tile_k_v16);

        // reg[64-66]: grid dimensions
        kernel->set_regs(64, (uint16_t)grid_dim_x);
        kernel->set_regs(65, (uint16_t)grid_dim_y);
        kernel->set_regs(66, (uint16_t)grid_dim_z);

        // atom_ids (reg 128+ in tile IR) are set via SCM in rpu_backend,
        // not via set_regs param space. Skipped — kernel uses default zeros.

        // SCM: per-core SPM addresses (Q, tmp, mask, output) in v16 units (>>5)
        for (int i = 0; i < core_num; ++i) {
            uint32_t q_spm_addr_v16 = SPM_ALLOC.addr(i, q_off) >> 5;
            uint32_t out_spm_addr_v16 = SPM_ALLOC.addr(i, output_off) >> 5;
            uint32_t tmp_v16 = tmp_addr_v16_arr[i];
            uint32_t mask_v16 = (sdpa_mask_off > 0) ? (SPM_ALLOC.addr(i, sdpa_mask_off) >> 5) : 0;

            int scm_base = SCM_REG_OFFSET + i * SCM_PARAMS_PER_CORE;
            kernel->set_regs(scm_base + 0, (uint16_t)(q_spm_addr_v16 & 0xFFFF));
            kernel->set_regs(scm_base + 1, (uint16_t)(q_spm_addr_v16 >> 16));
            kernel->set_regs(scm_base + 2, (uint16_t)(tmp_v16 & 0xFFFF));
            kernel->set_regs(scm_base + 3, (uint16_t)(tmp_v16 >> 16));
            kernel->set_regs(scm_base + 4, (uint16_t)(mask_v16 & 0xFFFF));
            kernel->set_regs(scm_base + 5, (uint16_t)(mask_v16 >> 16));
            kernel->set_regs(scm_base + 6, (uint16_t)(out_spm_addr_v16 & 0xFFFF));
            kernel->set_regs(scm_base + 7, (uint16_t)(out_spm_addr_v16 >> 16));
        }
    };

      Kernel_t* kernel = GET_KERNEL(KernelId::SDPA_FLASH_ATTN_SPM_VCTXLEN);
      TORCH_CHECK(kernel != nullptr, "Failed to get SDPA vctxlen SPM kernel");
      kernel->reset_regs();
      setup_regs(kernel);

      std::vector<uint8_t> core_list;
      for (int i = 0; i < core_num; ++i) {
          core_list.push_back(static_cast<uint8_t>(i));
      }

      auto* wq = GET_QUEUE(core_num);
      wq->set_broadcast_mode(true);
      wq->set_flush_icache(false);
      wq->enqueu_kernel(*kernel, {grid_dim_x, grid_dim_y, grid_dim_z}, core_list);
}

void rpu_launch_v_transpose_spm_batch3_576(
    uint32_t input_spm,
    uint32_t output_spm) {
    const uint32_t input_off =
        exact_wall_spm_offset(input_spm, kWallVisionTensorBytes, "V input");
    const uint32_t output_off =
        exact_wall_spm_offset(output_spm, kWallVisionTensorBytes, "V transpose output");
    TORCH_CHECK(input_off != output_off,
                "Wall Vision V transpose requires distinct input/output buffers");

    Kernel_t* kernel = GET_KERNEL(KernelId::V_TRANSPOSE_SPM_BATCH3_576);
    TORCH_CHECK(kernel != nullptr, "Failed to get Wall Vision V-transpose kernel");
    kernel->reset_regs();

    kernel->set_regs(0, kWallVisionSeq);
    kernel->set_regs(1, 0);
    kernel->set_regs(2, 0);   // sValAcc low
    kernel->set_regs(3, 0);   // sValAcc high
    kernel->set_regs(6, kWallVisionSeq);
    kernel->set_regs(7, kWallVisionSeq / 16);
    kernel->set_regs(9, kWallVisionHeadsPerCore);
    kernel->set_regs(12, kWallVisionHeadDim);
    kernel->set_regs(14, kWallVisionHeadDim / 16);
    kernel->set_regs(15, kWallVisionHeadDim / 16);
    kernel->set_regs(59, kWallVisionHeadDim);
    kernel->set_regs(60, 128);
    kernel->set_regs(62, kWallVisionHeadDim / 16);
    kernel->set_regs(63, 8);
    kernel->set_regs(64, 1);
    kernel->set_regs(65, kWallVisionHeadsPerCore);
    kernel->set_regs(66, kWallVisionBatch);

    constexpr int kScmStride = 4;
    for (int core = 0; core < kWallVisionCores; ++core) {
        const int base = SCM_REG_OFFSET + core * kScmStride;
        set_scm_spm_ptr(kernel, base, core, input_off);
        set_scm_spm_ptr(kernel, base + 2, core, output_off);
    }

    auto* queue = GET_QUEUE(kWallVisionCores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel, {1, kWallVisionHeadsPerCore, kWallVisionBatch},
        exact_wall_cores());
}

void rpu_launch_sdpa_by_mha_spm_batch3_576(
    uint32_t q_spm,
    uint32_t k_spm,
    uint32_t v_transposed_spm,
    uint32_t output_spm,
    uint32_t mask_spm,
    int attn_mask_type,
    double scale) {
    TORCH_CHECK(attn_mask_type == 0 || attn_mask_type == 4,
                "Wall Vision by-MHA supports only mask type 0 or 4, got ",
                attn_mask_type);
    TORCH_CHECK(std::isfinite(scale),
                "Wall Vision by-MHA scale must be finite, got ", scale);

    const uint32_t q_off =
        exact_wall_spm_offset(q_spm, kWallVisionTensorBytes, "Q");
    const uint32_t k_off =
        exact_wall_spm_offset(k_spm, kWallVisionTensorBytes, "K");
    const uint32_t v_off = exact_wall_spm_offset(
        v_transposed_spm, kWallVisionTensorBytes, "V transpose");
    const uint32_t output_off = exact_wall_spm_offset(
        output_spm, kWallVisionTensorBytes, "attention output");
    TORCH_CHECK(q_off != output_off,
                "Wall Vision by-MHA requires distinct Q/output buffers");
    uint32_t mask_off = 0;
    if (attn_mask_type == 4) {
        TORCH_CHECK(mask_spm != 0,
                    "Wall Vision MASK_2D requires a non-zero SPM address");
        mask_off = exact_wall_spm_offset(
            mask_spm, kWallVisionMaskBytes, "attention mask");
    }

    Kernel_t* kernel = GET_KERNEL(KernelId::SDPA_BY_MHA_SPM_BATCH3_576);
    TORCH_CHECK(kernel != nullptr, "Failed to get Wall Vision by-MHA kernel");
    kernel->reset_regs();

    kernel->set_regs(0, kWallVisionSeq);
    kernel->set_regs(1, 0);
    kernel->set_regs(2, kWallVisionSeq);
    kernel->set_regs(3, kWallVisionSeq / 16);
    kernel->set_regs(4, kWallVisionSeq / 16);
    kernel->set_regs(5, 0);
    kernel->set_regs(6, kWallVisionHeadsPerCore * kWallVisionCores);
    kernel->set_regs(7, kWallVisionHeadsPerCore * kWallVisionCores);
    kernel->set_regs(8, kWallVisionHeadsPerCore);
    kernel->set_regs(9, kWallVisionHeadsPerCore);
    kernel->set_regs(10, kWallVisionHeadDim);
    kernel->set_regs(11, kWallVisionHeadDim);
    kernel->set_regs(12, kWallVisionHeadDim);
    kernel->set_regs(13, kWallVisionHeadDim / 16);
    kernel->set_regs(14, kWallVisionHeadDim / 16);
    kernel->set_regs(15, kWallVisionHeadDim / 16);
    kernel->set_regs(16, 0);
    kernel->set_regs(17, 0);
    kernel->set_regs(18, kWallVisionCores);
    kernel->set_regs(19, 1);
    const uint32_t scale_bits = float32_to_uint32(static_cast<float>(scale));
    kernel->set_regs(40, static_cast<uint16_t>(scale_bits));
    kernel->set_regs(41, static_cast<uint16_t>(scale_bits >> 16));
    kernel->set_regs(42, 1);
    kernel->set_regs(56, static_cast<uint16_t>(attn_mask_type));
    kernel->set_regs(58, 80);
    kernel->set_regs(59, 80);
    kernel->set_regs(60, 256);
    kernel->set_regs(61, 5);
    kernel->set_regs(62, 5);
    kernel->set_regs(63, 16);
    kernel->set_regs(64, kWallVisionHeadsPerCore);
    kernel->set_regs(65, 8);
    kernel->set_regs(66, kWallVisionBatch);

    constexpr int kScmStride = 12;
    for (int core = 0; core < kWallVisionCores; ++core) {
        const int base = SCM_REG_OFFSET + core * kScmStride;
        set_scm_spm_ptr(kernel, base, core, q_off);
        if (attn_mask_type == 4) {
            set_scm_spm_ptr(kernel, base + 4, core, mask_off);
        }
        set_scm_spm_ptr(kernel, base + 6, core, output_off);
        set_scm_spm_ptr(kernel, base + 8, core, k_off);
        set_scm_spm_ptr(kernel, base + 10, core, v_off);
    }

    auto* queue = GET_QUEUE(kWallVisionCores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel, {kWallVisionHeadsPerCore, 8, kWallVisionBatch},
        exact_wall_cores());
}
