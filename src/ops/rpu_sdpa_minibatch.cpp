// rpu_sdpa_minibatch.cpp
// SPM-based SDPA using the vctxlen *minibatch* kernel — per-image K/V slicing.
//
// Kernel binary: llm_fp16_32b_prefill_flash_attn_univ_vctxlen_minibatch
// reg[0,1] is per_image_ctx_len; reg[34] is image_batch_count.
// Image i attends only cache rows [i*per_image_ctx_len, (i+1)*per_image_ctx_len).
// No mask. Requires num_kv_heads == num_heads (MHA); GQA is unsupported.
//
// The cache swizzle stride params (section/chapter/page/row, nKVHeadChunk,
// headDimVx) are computed IDENTICALLY to the plain vctxlen / FLASH_ATTN_SPM
// launchers, so a K/V cache populated by rpu_launch_insert_{k,v}cache_spm_unified
// at position i*per_image_ctx_len lines up with this kernel's per-image read.

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include <c10/util/Half.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <utility>

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

    // Minibatch hard constraint: MHA only (no warp-shared KV cache).
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

    // Compute tiling from the total packed seq_q.
    SdpaConfig tiling_cfg{SdpaKernelType::FLASH_ATTN_SPM,
                          head_dim, virtual_num_heads, virtual_num_kv_heads,
                          static_cast<int>(vtp), attn_mask_type};
    SdpaTiling tiling = sdpa_compute_tiling(tiling_cfg, seq_q);
    if (tile_m_override > 0) {
        TORCH_CHECK(tile_m_override % 16 == 0,
                    "minibatch SDPA tile_m_override must be a multiple of 16");
        tiling.tile_m = tile_m_override;
        tiling.tile_m_v16 = tile_m_override / 16;
        // The short-sequence isolation path uses the same K tile as M.  This
        // keeps the temporary workspace equal to the default packed-sequence
        // allocation while providing at least two blocks per image.
        tiling.tile_k = tile_m_override;
        tiling.tile_k_v16 = tile_m_override / 16;
        TORCH_CHECK(
            tiling.tile_m_v16 * tiling.tile_n_v16 * tiling.tile_k_v16 <= 1024,
            "minibatch SDPA tile override exceeds the VLM tile budget");
    }

    // Scale
    float scale_factor = scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim)));
    uint32_t scale_u32 = float32_to_uint32(scale_factor);

    // Grid: gridDim.x must be evenly divisible by image_batch_count, and each
    // image's queries (per_image_ctx_len) must be a whole number of tile_m
    // blocks — else the block→image mapping (blkid.x / (gdim.x/N)) misroutes.
    uint16_t grid_dim_x = CeilDiv(seq_q, tiling.tile_m);
    uint16_t grid_dim_y = num_heads_per_core;
    uint16_t grid_dim_z = 1;
    TORCH_CHECK(grid_dim_x % image_batch_count == 0,
                "grid_dim_x (", grid_dim_x, ") must be divisible by image_batch_count (",
                image_batch_count, "); pick tile_m so per_image_ctx_len % tile_m == 0");
    TORCH_CHECK(per_image_ctx_len % tiling.tile_m == 0,
                "per_image_ctx_len (", per_image_ctx_len, ") must be a multiple of tile_m (",
                tiling.tile_m, ")");
    // Require at least two grid blocks per image: the one-block geometry
    // violates this payload's memory-isolation contract.
    TORCH_CHECK(per_image_ctx_len / tiling.tile_m >= 2,
                "minibatch SDPA needs >=2 grid blocks per image; got per_image_ctx_len (",
                per_image_ctx_len, ") / tile_m (", tiling.tile_m,
                ") = ", per_image_ctx_len / tiling.tile_m,
                " (1-block-per-image corrupts memory — pick smaller tile_m)");
    // grid_dim_x*gqa_group_size must be a multiple of 8 when greater than 8.
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
            rpu_set_scm_u32_checked(
                kernel, scm_base + 0, q_spm_addr_v16,
                "rpu_sdpa_minibatch query");
            rpu_set_scm_u32_checked(
                kernel, scm_base + 2, tmp_v16,
                "rpu_sdpa_minibatch temporary");
            rpu_set_scm_u32_checked(
                kernel, scm_base + 4, mask_v16,
                "rpu_sdpa_minibatch mask");
            rpu_set_scm_u32_checked(
                kernel, scm_base + 6, out_spm_addr_v16,
                "rpu_sdpa_minibatch output");
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
    wq->set_flush_icache(true);
    wq->enqueu_kernel(*kernel, {grid_dim_x, grid_dim_y, grid_dim_z}, core_list);
}

void rpu_launch_lingbot2_vision_sdpa_spm_kernel(
    uint32_t q_addr, uint32_t k_addr, uint32_t vt_addr,
    uint32_t output_addr, uint32_t tmp_addr) {
    // Cold owner admission fixes B3/S256/QKV16/D64/TP8. Preserve the original
    // 32b minibatch M128/N64/K256 register ABI; only K/V input staging changes.
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "Lingbot2 Vision SPM SDPA requires initialized SPM");
    constexpr uint64_t kOperandBytes = 3 * 256 * 2 * 64 * 2;
    constexpr uint64_t kTmpBytes = 393216;
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint64_t>, 5> ranges{{
        {q_addr, kOperandBytes}, {k_addr, kOperandBytes},
        {vt_addr, kOperandBytes}, {output_addr, kOperandBytes},
        {tmp_addr, kTmpBytes}}};
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto [address, bytes] = ranges[i];
        TORCH_CHECK(address >= base && (address & 31u) == 0 &&
                        bytes <= SpmAllocator::SPM_USABLE &&
                        uint64_t{address} - base <= SpmAllocator::SPM_USABLE - bytes &&
                        uint64_t{address} + bytes <= (uint64_t{1} << 32),
                    "Lingbot2 Vision SPM SDPA requires complete aligned absolute SPM intervals");
        for (size_t j = 0; j < i; ++j) {
            TORCH_CHECK(uint64_t{address} + bytes <= ranges[j].first ||
                            uint64_t{ranges[j].first} + ranges[j].second <= address,
                        "Lingbot2 Vision SPM SDPA Q/K/Vt/output/tmp must be disjoint");
        }
    }
    // Validate every physical core before looking up/resetting the kernel.
    std::array<std::array<uint32_t, 5>, 8> addresses_v16{};
    for (int core = 0; core < 8; ++core) {
        const uint64_t core_base = SPM_ALLOC.addr(core, 0);
        for (size_t i = 0; i < ranges.size(); ++i) {
            const uint64_t address = core_base + ranges[i].first - base;
            TORCH_CHECK((address & 31u) == 0 &&
                            address + ranges[i].second <= (uint64_t{1} << 32),
                        "Lingbot2 Vision SPM SDPA per-core address exceeds aligned u32 storage");
            addresses_v16[core][i] = static_cast<uint32_t>(address >> 5);
        }
    }
    constexpr KernelId id = KernelId::LINGBOT2_VISION_SDPA_SPM_B3S256H16D64_32B;
    TORCH_CHECK(KernelCache::instance().has_loaded(id),
                "Lingbot2 Vision SPM SDPA optional payload is absent");
    auto* queue = GET_QUEUE(8);
    TORCH_CHECK(queue != nullptr, "Lingbot2 Vision SPM SDPA eight-core queue is absent");
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel != nullptr, "Lingbot2 Vision SPM SDPA kernel is absent");
    kernel->reset_regs();
    // Same setup as the ordinary wrapper with default tile_m_override=0.
    // DDR address registers 50..53 remain zero and are unused by this payload.
    const std::pair<uint16_t, uint16_t> parameters[] = {
        {0, 256}, {2, 768}, {3, 48}, {6, 16}, {7, 16}, {8, 2}, {9, 2},
        {10, 64}, {11, 64}, {12, 64}, {13, 4}, {14, 4}, {15, 4},
        {18, 8}, {19, 1}, {20, 128}, {21, 64}, {22, 16}, {23, 2},
        {24, 8}, {25, 2}, {26, 16}, {27, 4},
        {28, 16}, {29, 16}, {30, 16}, {31, 16}, {32, 1}, {34, 3},
        {41, 0x3e00}, {42, 1},
        {58, 128}, {59, 64}, {60, 256}, {61, 8}, {62, 4}, {63, 16},
        {64, 6}, {65, 2}, {66, 1}};
    for (const auto& [reg, value] : parameters) kernel->set_regs(reg, value);
    for (int core = 0; core < 8; ++core) {
        const int old_base = SCM_REG_OFFSET + core * 8;
        const int kv_base = SCM_REG_OFFSET + 64 + core * 4;
        const auto& a = addresses_v16[core];
        rpu_set_scm_u32_checked(kernel, old_base, a[0], "Lingbot2 Vision Q");
        rpu_set_scm_u32_checked(kernel, old_base + 2, a[4], "Lingbot2 Vision tmp");
        rpu_set_scm_u32_checked(kernel, old_base + 4, 0, "Lingbot2 Vision no mask");
        rpu_set_scm_u32_checked(kernel, old_base + 6, a[3], "Lingbot2 Vision output");
        rpu_set_scm_u32_checked(kernel, kv_base, a[1], "Lingbot2 Vision K");
        rpu_set_scm_u32_checked(kernel, kv_base + 2, a[2], "Lingbot2 Vision V transpose");
    }
    queue->set_broadcast_mode(true);
    queue->set_flush_icache(true);
    queue->enqueu_kernel(*kernel, {6, 2, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}
