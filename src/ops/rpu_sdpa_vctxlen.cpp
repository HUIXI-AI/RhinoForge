// rpu_sdpa_vctxlen.cpp
// SPM-based SDPA kernel wrapper using the vctxlen (variable context length) kernel.
//
// Kernel binary: llm_fp16_32b_prefill_flash_attn_univ_vctxlen
// Register layout and tiling follow the FLASH_ATTN_SPM launch contract.
// All param registers are set (including sKeyV16, sQryAcc, sKeyVx, sValVx).

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include <c10/util/Half.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

// float32_to_uint32 and SCM_REG_OFFSET defined in rpu_helpers.h
#define SCM_PARAMS_PER_CORE 8  // 4 addresses × 2 uint16 each

namespace {

struct SpmRegion {
    uint32_t offset;
    size_t bytes;
    const char* name;
};

size_t checked_spm_bytes(
    const char* operand, std::initializer_list<int64_t> dimensions) {
    size_t bytes = sizeof(c10::Half);
    for (const int64_t dim : dimensions) {
        TORCH_CHECK(dim > 0, operand, " dimensions must be positive");
        TORCH_CHECK(
            static_cast<uint64_t>(dim) <= SpmAllocator::SPM_USABLE / bytes,
            operand, " requires more than one core's SPM capacity");
        bytes *= static_cast<size_t>(dim);
    }
    return bytes;
}

SpmRegion checked_spm_region(
    uint32_t addr, size_t bytes, const char* operand) {
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "raw-SPM attention launcher requires initialized SPM");
    const uint32_t base = SPM_ALLOC.addr(0, 0);
    TORCH_CHECK(addr >= base && (addr & 31u) == 0,
                operand, " must be a 32-byte-aligned core-0 SPM address");
    const uint64_t offset = static_cast<uint64_t>(addr) - base;
    TORCH_CHECK(offset + bytes <= SpmAllocator::SPM_USABLE,
                operand, " exceeds one core's SPM range");
    return {static_cast<uint32_t>(offset), bytes, operand};
}

template <size_t N>
void check_spm_regions_disjoint(const std::array<SpmRegion, N>& regions) {
    for (size_t i = 0; i < regions.size(); ++i) {
        if (regions[i].bytes == 0) {
            continue;
        }
        const uint64_t i_end = regions[i].offset + regions[i].bytes;
        for (size_t j = i + 1; j < regions.size(); ++j) {
            if (regions[j].bytes == 0) {
                continue;
            }
            const uint64_t j_end = regions[j].offset + regions[j].bytes;
            TORCH_CHECK(
                i_end <= regions[j].offset || j_end <= regions[i].offset,
                "raw-SPM attention regions overlap: ", regions[i].name,
                " and ", regions[j].name);
        }
    }
}

void set_scm_spm_ptr(
    Kernel_t* kernel, int reg, int core, uint32_t offset) {
    const uint32_t addr_v16 = SPM_ALLOC.addr(core, offset) >> 5;
    rpu_set_scm_u32_checked(
        kernel, reg, addr_v16, "rpu_sdpa_vctxlen SPM pointer");
}

std::vector<uint8_t> spm_cores(int num_cores) {
    std::vector<uint8_t> cores(num_cores);
    for (int i = 0; i < num_cores; ++i) {
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
            rpu_set_scm_u32_checked(
                kernel, scm_base + 0, q_spm_addr_v16,
                "rpu_sdpa_vctxlen query");
            rpu_set_scm_u32_checked(
                kernel, scm_base + 2, tmp_v16,
                "rpu_sdpa_vctxlen temporary");
            rpu_set_scm_u32_checked(
                kernel, scm_base + 4, mask_v16,
                "rpu_sdpa_vctxlen mask");
            rpu_set_scm_u32_checked(
                kernel, scm_base + 6, out_spm_addr_v16,
                "rpu_sdpa_vctxlen output");
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

void rpu_launch_v_transpose_spm(
    uint32_t input_spm,
    uint32_t output_spm,
    int64_t batch,
    int64_t seq_k,
    int64_t num_kv_heads,
    int64_t head_dim,
    int num_cores) {
    constexpr int64_t kV16 = 16;
    constexpr int64_t kTileK = 128;
    const int64_t u16_max = std::numeric_limits<uint16_t>::max();

    TORCH_CHECK(num_cores > 0 && num_cores <= SpmAllocator::NUM_CORES,
                "V transpose num_cores must be in [1,8], got ", num_cores);
    TORCH_CHECK(batch > 0 && batch <= u16_max,
                "V transpose batch must fit uint16, got ", batch);
    TORCH_CHECK(seq_k > 0 && seq_k <= u16_max,
                "V transpose seq_k must fit uint16, got ", seq_k);
    TORCH_CHECK(num_kv_heads > 0 && num_kv_heads <= u16_max,
                "V transpose num_kv_heads must fit uint16, got ",
                num_kv_heads);
    TORCH_CHECK(head_dim > 0 && head_dim <= u16_max && head_dim % kV16 == 0,
                "V transpose head_dim must be a positive uint16 multiple of 16, got ",
                head_dim);
    TORCH_CHECK(num_kv_heads % num_cores == 0,
                "V transpose num_kv_heads must be divisible by num_cores");
    TORCH_CHECK(batch == 1 || seq_k % kV16 == 0,
                "V transpose batch>1 requires seq_k to be 16-aligned");

    const int64_t num_kv_heads_per_core = num_kv_heads / num_cores;
    TORCH_CHECK(num_kv_heads_per_core <= 32,
                "V transpose supports at most 32 KV heads per core");
    const int64_t seq_k_v16 = CeilDiv(seq_k, kV16);
    const int64_t seq_k_aligned = seq_k_v16 * kV16;
    const int64_t head_dim_v16 = head_dim / kV16;

    const size_t input_bytes = checked_spm_bytes(
        "V input", {batch, seq_k, num_kv_heads_per_core, head_dim});
    const size_t output_bytes = checked_spm_bytes(
        "V transpose output",
        {batch, num_kv_heads_per_core, head_dim, seq_k_aligned});
    const SpmRegion input =
        checked_spm_region(input_spm, input_bytes, "V input");
    const SpmRegion output = checked_spm_region(
        output_spm, output_bytes, "V transpose output");
    const uint32_t input_off = input.offset;
    const uint32_t output_off = output.offset;
    TORCH_CHECK(input_off != output_off,
                "raw-SPM V transpose requires distinct input/output buffers");
    check_spm_regions_disjoint(std::array<SpmRegion, 2>{input, output});

    Kernel_t* kernel = GET_KERNEL(KernelId::V_TRANSPOSE_SPM_VCTXLEN);
    TORCH_CHECK(kernel != nullptr, "Failed to get raw-SPM V-transpose kernel");
    kernel->reset_regs();

    kernel->set_regs(0, static_cast<uint16_t>(seq_k));
    kernel->set_regs(1, static_cast<uint16_t>(seq_k >> 16));
    kernel->set_regs(2, static_cast<uint16_t>(0));  // sValAcc low
    kernel->set_regs(3, static_cast<uint16_t>(0));  // sValAcc high
    kernel->set_regs(6, static_cast<uint16_t>(seq_k));
    kernel->set_regs(7, static_cast<uint16_t>(seq_k_v16));
    kernel->set_regs(9, static_cast<uint16_t>(num_kv_heads_per_core));
    kernel->set_regs(12, static_cast<uint16_t>(head_dim));
    kernel->set_regs(14, static_cast<uint16_t>(head_dim_v16));
    kernel->set_regs(15, static_cast<uint16_t>(head_dim_v16));
    kernel->set_regs(59, static_cast<uint16_t>(head_dim));
    kernel->set_regs(60, static_cast<uint16_t>(kTileK));
    kernel->set_regs(62, static_cast<uint16_t>(head_dim_v16));
    kernel->set_regs(63, static_cast<uint16_t>(kTileK / kV16));
    kernel->set_regs(64, static_cast<uint16_t>(1));
    kernel->set_regs(65, static_cast<uint16_t>(num_kv_heads_per_core));
    kernel->set_regs(66, static_cast<uint16_t>(batch));

    constexpr int kScmStride = 4;
    for (int core = 0; core < num_cores; ++core) {
        const int base = SCM_REG_OFFSET + core * kScmStride;
        set_scm_spm_ptr(kernel, base, core, input_off);
        set_scm_spm_ptr(kernel, base + 2, core, output_off);
    }

    auto* queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel, {1, static_cast<uint16_t>(num_kv_heads_per_core),
                  static_cast<uint16_t>(batch)},
        spm_cores(num_cores));
}

void rpu_launch_sdpa_by_mha_spm(
    uint32_t q_spm,
    uint32_t k_spm,
    uint32_t v_transposed_spm,
    uint32_t output_spm,
    uint32_t mask_spm,
    int attn_mask_type,
    double scale,
    int64_t batch,
    int64_t seq_q,
    int64_t seq_k,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int num_cores) {
    constexpr int64_t kV16 = 16;
    constexpr int64_t kTileKV16 = 16;
    constexpr int64_t kMaxTileElementsV16 = 1024;
    const int64_t u16_max = std::numeric_limits<uint16_t>::max();
    const int64_t u32_max = std::numeric_limits<uint32_t>::max();

    TORCH_CHECK(attn_mask_type == 0 || attn_mask_type == 1 ||
                    attn_mask_type == 4,
                "raw-SPM by-MHA supports only mask type 0, 1, or 4, got ",
                attn_mask_type);
    const float scale_f32 = static_cast<float>(scale);
    TORCH_CHECK(std::isfinite(scale) && std::isfinite(scale_f32),
                "raw-SPM by-MHA scale must be finite in fp32, got ", scale);
    TORCH_CHECK(num_cores > 0 && num_cores <= SpmAllocator::NUM_CORES,
                "raw-SPM by-MHA num_cores must be in [1,8], got ", num_cores);
    TORCH_CHECK(batch > 0 && batch <= u16_max,
                "raw-SPM by-MHA batch must fit uint16, got ", batch);
    TORCH_CHECK(seq_q > 0 && seq_q <= u16_max,
                "raw-SPM by-MHA seq_q must fit uint16, got ", seq_q);
    TORCH_CHECK(seq_k >= seq_q && seq_k <= u32_max,
                "raw-SPM by-MHA requires uint32 seq_k >= seq_q, got seq_q=",
                seq_q, " seq_k=", seq_k);
    TORCH_CHECK(num_q_heads > 0 && num_q_heads <= u16_max &&
                    num_kv_heads > 0 && num_kv_heads <= u16_max,
                "raw-SPM by-MHA head counts must be positive uint16 values");
    TORCH_CHECK(head_dim > 0 && head_dim <= u16_max && head_dim % kV16 == 0,
                "raw-SPM by-MHA head_dim must be a positive uint16 multiple of 16, got ",
                head_dim);
    TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                "raw-SPM by-MHA requires num_q_heads divisible by num_kv_heads");
    TORCH_CHECK(num_q_heads % num_cores == 0,
                "raw-SPM by-MHA requires num_q_heads divisible by num_cores");
    TORCH_CHECK(num_kv_heads % num_cores == 0,
                "raw-SPM by-MHA requires num_kv_heads divisible by num_cores");
    TORCH_CHECK(batch == 1 || seq_k % kV16 == 0,
                "raw-SPM by-MHA batch>1 requires seq_k to be 16-aligned");

    const int64_t seq_q_v16 = CeilDiv(seq_q, kV16);
    const int64_t seq_k_v16 = CeilDiv(seq_k, kV16);
    const int64_t seq_k_aligned = seq_k_v16 * kV16;
    const int64_t seq_q_acc = seq_k - seq_q;
    const int64_t seq_q_acc_v16 = CeilDiv(seq_q_acc, kV16);
    TORCH_CHECK(seq_k_v16 <= u16_max && seq_q_acc_v16 <= u16_max,
                "raw-SPM by-MHA seq_k/accumulated context exceeds uint16-v16 registers");
    if (attn_mask_type == 1) {
        TORCH_CHECK(
            seq_q_acc == 0 ||
                (seq_q_acc % kV16 == 0 && seq_q_acc % seq_q == 0),
            "raw-SPM by-MHA LTM requires accumulated context to be 16-aligned "
            "and a multiple of seq_q");
    }

    const int64_t num_q_heads_per_core = num_q_heads / num_cores;
    const int64_t num_kv_heads_per_core = num_kv_heads / num_cores;
    const int64_t gqa_group_size =
        num_q_heads_per_core / num_kv_heads_per_core;
    const int64_t head_dim_v16 = head_dim / kV16;
    const int64_t tile_m_v16 =
        seq_q <= 128 ? CeilDiv(seq_q, kV16) : 5;
    const int64_t tile_n_v16 = head_dim_v16;
    TORCH_CHECK(tile_m_v16 * tile_n_v16 * kTileKV16 <=
                    kMaxTileElementsV16,
                "raw-SPM by-MHA tile exceeds 1024 v16 elements");
    if (attn_mask_type == 1) {
        TORCH_CHECK(kTileKV16 % tile_m_v16 == 0,
                    "raw-SPM by-MHA LTM requires tile_k_v16 divisible by tile_m_v16");
    }
    const int64_t tile_m = tile_m_v16 * kV16;
    const int64_t tile_n = tile_n_v16 * kV16;
    const int64_t tile_k = kTileKV16 * kV16;
    const int64_t grid_dim_y = CeilDiv(seq_q, tile_m);

    // The by-MHA binary swaps the conventional grid axes (heads on X,
    // query tiles on Y), but it has the same sync-group and accumulated-LTM
    // hardware limits as the DDR-cache SDPA launchers.  Pass query tiles as
    // the validator's grid dimension so an otherwise legal VLM/tile shape
    // cannot escape with query_tiles * GQA > 8 (or the broken product-6
    // accumulated-LTM case).
    sdpa_validate_kernel_call(
        seq_q, seq_k,
        tile_m_v16, tile_n_v16, kTileKV16, tile_m,
        grid_dim_y, gqa_group_size,
        head_dim_v16, head_dim_v16, head_dim_v16,
        attn_mask_type);

    const size_t q_bytes = checked_spm_bytes(
        "Q", {batch, seq_q, num_q_heads_per_core, head_dim});
    const size_t k_bytes = checked_spm_bytes(
        "K", {batch, seq_k_aligned, num_kv_heads_per_core, head_dim});
    const size_t v_bytes = checked_spm_bytes(
        "V transpose",
        {batch, num_kv_heads_per_core, head_dim, seq_k_aligned});
    const size_t output_bytes = checked_spm_bytes(
        "attention output",
        {batch, seq_q, num_q_heads_per_core, head_dim});
    const SpmRegion q = checked_spm_region(q_spm, q_bytes, "Q");
    const SpmRegion k = checked_spm_region(k_spm, k_bytes, "K");
    const SpmRegion v = checked_spm_region(
        v_transposed_spm, v_bytes, "V transpose");
    const SpmRegion output = checked_spm_region(
        output_spm, output_bytes, "attention output");
    SpmRegion mask{0, 0, "attention mask"};
    uint32_t mask_off = 0;
    if (attn_mask_type == 4) {
        TORCH_CHECK(mask_spm != 0,
                    "raw-SPM by-MHA MASK_2D requires a non-zero SPM address");
        const size_t mask_bytes = checked_spm_bytes(
            "attention mask", {seq_q, seq_k_aligned});
        mask = checked_spm_region(mask_spm, mask_bytes, "attention mask");
        mask_off = mask.offset;
    }
    const uint32_t q_off = q.offset;
    const uint32_t k_off = k.offset;
    const uint32_t v_off = v.offset;
    const uint32_t output_off = output.offset;
    TORCH_CHECK(q_off != output_off,
                "raw-SPM by-MHA requires distinct Q/output buffers");
    check_spm_regions_disjoint(
        std::array<SpmRegion, 5>{q, k, v, output, mask});

    Kernel_t* kernel = GET_KERNEL(KernelId::SDPA_BY_MHA_SPM_VCTXLEN);
    TORCH_CHECK(kernel != nullptr, "Failed to get raw-SPM by-MHA kernel");
    kernel->reset_regs();

    kernel->set_regs(0, static_cast<uint16_t>(seq_k));
    kernel->set_regs(1, static_cast<uint16_t>(seq_k >> 16));
    kernel->set_regs(2, static_cast<uint16_t>(seq_q));
    kernel->set_regs(3, static_cast<uint16_t>(seq_q_v16));
    kernel->set_regs(4, static_cast<uint16_t>(seq_k_v16));
    kernel->set_regs(5, static_cast<uint16_t>(seq_q_acc_v16));
    kernel->set_regs(6, static_cast<uint16_t>(num_q_heads));
    kernel->set_regs(7, static_cast<uint16_t>(num_kv_heads));
    kernel->set_regs(8, static_cast<uint16_t>(num_q_heads_per_core));
    kernel->set_regs(9, static_cast<uint16_t>(num_kv_heads_per_core));
    kernel->set_regs(10, static_cast<uint16_t>(head_dim));
    kernel->set_regs(11, static_cast<uint16_t>(head_dim));
    kernel->set_regs(12, static_cast<uint16_t>(head_dim));
    kernel->set_regs(13, static_cast<uint16_t>(head_dim_v16));
    kernel->set_regs(14, static_cast<uint16_t>(head_dim_v16));
    kernel->set_regs(15, static_cast<uint16_t>(head_dim_v16));
    kernel->set_regs(16, static_cast<uint16_t>(seq_q_acc));
    kernel->set_regs(17, static_cast<uint16_t>(seq_q_acc >> 16));
    kernel->set_regs(18, static_cast<uint16_t>(num_cores));
    kernel->set_regs(19, static_cast<uint16_t>(gqa_group_size));
    const uint32_t scale_bits = float32_to_uint32(scale_f32);
    kernel->set_regs(40, static_cast<uint16_t>(scale_bits));
    kernel->set_regs(41, static_cast<uint16_t>(scale_bits >> 16));
    kernel->set_regs(42, static_cast<uint16_t>(1));
    kernel->set_regs(56, static_cast<uint16_t>(attn_mask_type));
    kernel->set_regs(58, static_cast<uint16_t>(tile_m));
    kernel->set_regs(59, static_cast<uint16_t>(tile_n));
    kernel->set_regs(60, static_cast<uint16_t>(tile_k));
    kernel->set_regs(61, static_cast<uint16_t>(tile_m_v16));
    kernel->set_regs(62, static_cast<uint16_t>(tile_n_v16));
    kernel->set_regs(63, static_cast<uint16_t>(kTileKV16));
    kernel->set_regs(64, static_cast<uint16_t>(num_q_heads_per_core));
    kernel->set_regs(65, static_cast<uint16_t>(grid_dim_y));
    kernel->set_regs(66, static_cast<uint16_t>(batch));

    constexpr int kScmStride = 12;
    for (int core = 0; core < num_cores; ++core) {
        const int base = SCM_REG_OFFSET + core * kScmStride;
        set_scm_spm_ptr(kernel, base, core, q_off);
        if (attn_mask_type == 4) {
            set_scm_spm_ptr(kernel, base + 4, core, mask_off);
        }
        set_scm_spm_ptr(kernel, base + 6, core, output_off);
        set_scm_spm_ptr(kernel, base + 8, core, k_off);
        set_scm_spm_ptr(kernel, base + 10, core, v_off);
    }

    auto* queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel, {static_cast<uint16_t>(num_q_heads_per_core),
                  static_cast<uint16_t>(grid_dim_y),
                  static_cast<uint16_t>(batch)},
        spm_cores(num_cores));
}
