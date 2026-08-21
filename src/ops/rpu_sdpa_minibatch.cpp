// rpu_sdpa_minibatch.cpp
// SPM-based minibatch SDPA with per-image K/V slicing. Image i attends only
// cache rows [i*per_image_ctx_len, (i+1)*per_image_ctx_len). The operator
// supports MHA only, and its public cache layout matches the unified K/V insert
// wrappers.

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include <c10/util/Half.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

using namespace at;
using namespace ::rhino_lkn;

// float32_to_uint32 and SCM_REG_OFFSET defined in rpu_helpers.h
#define SCM_PARAMS_PER_CORE 8  // 4 addresses × 2 uint16 each

void rpu_launch_sdpa_spm_minibatch_kernel(
    const at::Tensor &key,
    const at::Tensor &value,
    int attn_mask_type,
    std::optional<double> scale,
    uint32_t q_off, uint32_t output_off, uint32_t sdpa_tmp_off, uint32_t sdpa_mask_off,
    int64_t seq_q, int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t per_image_ctx_len, int64_t image_batch_count,
    int num_cores,
    int virtual_num_cores,
    int tile_m_override,
    bool use_16b)
{
    int64_t core_num = num_cores;
    int64_t vtp = (virtual_num_cores > 0) ? virtual_num_cores : core_num;

    // Minibatch hard constraint: MHA only.
    TORCH_CHECK(num_heads == num_kv_heads,
                "minibatch SDPA requires num_kv_heads == num_heads (MHA); got nq=",
                num_heads, " nkv=", num_kv_heads);
    TORCH_CHECK(image_batch_count >= 1, "image_batch_count must be >= 1");
    TORCH_CHECK(per_image_ctx_len > 0 && per_image_ctx_len * image_batch_count == seq_q,
                "seq_q (", seq_q, ") must equal per_image_ctx_len (", per_image_ctx_len,
                ") * image_batch_count (", image_batch_count, ")");

    // Virtual head counts (same convention as the SPM/vctxlen launchers)
    int64_t q_heads_per_core = num_heads / core_num;
    int64_t kv_heads_per_core = CeilDiv(num_kv_heads, core_num);
    int64_t virtual_num_heads = q_heads_per_core * vtp;
    int64_t virtual_num_kv_heads = kv_heads_per_core * vtp;

    TORCH_CHECK(num_heads % core_num == 0,
                "num_heads must be divisible by core_num (", core_num, ")");

    int64_t num_heads_per_core = q_heads_per_core;
    int64_t nkv_head_per_core = CeilDiv(virtual_num_kv_heads, vtp);
    int64_t gqa_group_size = num_heads_per_core / nkv_head_per_core;  // = 1 (MHA)

    int64_t nkv_head_chunk = std::min(vtp, virtual_num_kv_heads);
    int64_t nkv_head_vx = CeilDiv(virtual_num_kv_heads, vtp);
    int64_t core_num_per_kv_head = CeilDiv(vtp, virtual_num_kv_heads);

    int64_t head_dim_chunk = 16;
    int64_t head_dim_vx = CeilDiv(head_dim, head_dim_chunk);
    int64_t head_dim_v16 = CeilDiv(head_dim, (int64_t)16);

    int64_t seq_q_v16 = CeilDiv(seq_q, (int64_t)16);

    // Tiling uses the total packed query length.
    SdpaConfig tiling_cfg{SdpaKernelType::FLASH_ATTN_SPM,
                          head_dim, virtual_num_heads, virtual_num_kv_heads,
                          static_cast<int>(vtp), attn_mask_type};
    SdpaTiling tiling = sdpa_compute_tiling(tiling_cfg, seq_q);
    if (tile_m_override > 0) {
        TORCH_CHECK(tile_m_override % 16 == 0,
                    "minibatch SDPA tile_m_override must be a multiple of 16");
        tiling.tile_m = tile_m_override;
        tiling.tile_m_v16 = tile_m_override / 16;
        // The short-sequence isolation path uses the same K and M tile.
        tiling.tile_k = tile_m_override;
        tiling.tile_k_v16 = tile_m_override / 16;
        TORCH_CHECK(
            tiling.tile_m_v16 * tiling.tile_n_v16 * tiling.tile_k_v16 <= 1024,
            "minibatch SDPA tile override exceeds the VLM tile budget");
    }

    // Scale
    float scale_factor = scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim)));
    uint32_t scale_u32 = float32_to_uint32(scale_factor);

    // Each image must own a whole number of launch blocks.
    uint16_t grid_dim_x = CeilDiv(seq_q, tiling.tile_m);
    uint16_t grid_dim_y = num_heads_per_core;
    uint16_t grid_dim_z = 1;
    TORCH_CHECK(grid_dim_x % image_batch_count == 0,
                "grid_dim_x (", grid_dim_x, ") must be divisible by image_batch_count (",
                image_batch_count, "); pick tile_m so per_image_ctx_len % tile_m == 0");
    TORCH_CHECK(per_image_ctx_len % tiling.tile_m == 0,
                "per_image_ctx_len (", per_image_ctx_len, ") must be a multiple of tile_m (",
                tiling.tile_m, ")");
    // The operator requires at least two launch blocks per image.
    TORCH_CHECK(per_image_ctx_len / tiling.tile_m >= 2,
                "minibatch SDPA needs >=2 grid blocks per image; got per_image_ctx_len (",
                per_image_ctx_len, ") / tile_m (", tiling.tile_m,
                ") = ", per_image_ctx_len / tiling.tile_m,
                " (1-block-per-image corrupts memory — pick smaller tile_m)");
    // Launch constraint: when the grid product exceeds 8, it must be a
    // multiple of 8.
    if ((int64_t)grid_dim_x * gqa_group_size > 8) {
        TORCH_CHECK(((int64_t)grid_dim_x * gqa_group_size) % 8 == 0,
                    "minibatch SDPA: grid_dim_x*gqa_group_size (", grid_dim_x * gqa_group_size,
                    ") must be a multiple of 8 when > 8");
    }

    // KV cache addressing (IDENTICAL to vctxlen / FLASH_ATTN_SPM)
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

    auto setup_regs = [=](Kernel_t* kernel) {
        // reg[0,1]: per_image_ctx_len (per-image KV length; the minibatch
        // kernel derives per-image sKeyVx = ceil(per_image_ctx_len/16) from it)
        kernel->set_regs(0, (uint16_t)(per_image_ctx_len & 0xFFFF));
        kernel->set_regs(1, (uint16_t)(per_image_ctx_len >> 16));

        // reg[2,3]: total packed sQry
        kernel->set_regs(2, (uint16_t)seq_q);
        kernel->set_regs(3, (uint16_t)seq_q_v16);

        // NOTE: reg[4,5] (seq_k_v16, seq_q_acc_v16) and reg[16,17] (seq_q_acc)
        // are intentionally NOT set — the minibatch wrapper omits them; the
        // kernel computes per-image addressing from reg[0,1] + reg[34].
        // reset_regs() leaves them 0.

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

        // reg[28-31]: sKey/sVal chunk + per-image vx. EMPIRICALLY REQUIRED:
        // removing reg[29]/reg[31] makes the K/V load produce NaN (N=1 test),
        // so the kernel DOES read them. Per-image vx = ceil(per_image_ctx/16).
        int64_t seq_k_chunk = 16;
        int64_t seq_k_vx = CeilDiv(per_image_ctx_len, seq_k_chunk);
        kernel->set_regs(28, (uint16_t)seq_k_chunk);
        kernel->set_regs(29, (uint16_t)seq_k_vx);
        kernel->set_regs(30, (uint16_t)seq_k_chunk);
        kernel->set_regs(31, (uint16_t)seq_k_vx);

        // reg[32]: coreNumPerKVHead
        kernel->set_regs(32, (uint16_t)core_num_per_kv_head);

        // reg[34]: image_batch_count (THE minibatch dispatch literal)
        kernel->set_regs(34, (uint16_t)image_batch_count);

        // reg[40-42]: scale
        kernel->set_regs(40, (uint16_t)(scale_u32 & 0xFFFF));
        kernel->set_regs(41, (uint16_t)(scale_u32 >> 16));
        kernel->set_regs(42, (uint16_t)1);  // has_scale

        // reg[50-53]: K/V cache DDR addresses (glarge)
        kernel->set_regs(50, (uint16_t)(ddr_k_cache_v128 & 0xFFFF));
        kernel->set_regs(51, (uint16_t)(ddr_k_cache_v128 >> 16));
        kernel->set_regs(52, (uint16_t)(ddr_v_cache_v128 & 0xFFFF));
        kernel->set_regs(53, (uint16_t)(ddr_v_cache_v128 >> 16));

        // reg[56]: attn_mask_type (minibatch path uses 0 = no mask)
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

    Kernel_t* kernel = GET_KERNEL(
        use_16b
            ? KernelId::SDPA_FLASH_ATTN_SPM_VCTXLEN_MINIBATCH_16B
            : KernelId::SDPA_FLASH_ATTN_SPM_VCTXLEN_MINIBATCH);
    TORCH_CHECK(kernel != nullptr, "Failed to get SDPA vctxlen minibatch SPM kernel");
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
