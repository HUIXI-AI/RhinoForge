// RPU Conv2d wrappers for NHWC FP16 tensors in DDR and SPM.
// Kernel launch ABI:
//   [0:1]  spm_in_v16      [2:3]  spm_flt_v16     [4:5]  spm_out_v16
//   [6]    inhw             [7]    outhw            [8]    flthw
//   [9]    inh              [10]   inw              [11]   batch
//   [12]   group            [13]   num_chl_per_grp  [14]   num_chl_per_grp_v16
//   [15]   num_chl_per_grp_v16_pad  [16] num_chl_v16_tail
//   [17]   flth             [18]   fltw             [19]   num_chl_per_grp_v16_tail
//   [20]   num_flt_per_grp  [21]   num_flt_per_grp_v16  [22] num_flt_v16_tail
//   [23]   outh             [24]   outw
//   [25]   strideh          [26]   stridew
//   [27]   padh             [28]   padH             [29]   padw    [30] padW
//   [31]   holeh            [32]   holew
//   [33]   bias_flag        [34:35] spm_bias_v16
//   [36]   act              [37]   elt              [38:39] spm_elt_v16
//   [40]   eltAct
//   [64]   grid_dim_x       [65]   grid_dim_y       [66]   grid_dim_z

#include "rpu_ops.h"
#include "graph/execution_coordinator.h"
#include "rhino_launch_buffer.h"

#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"
#include <c10/util/Half.h>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

using namespace at;
using namespace ::rhino_lkn;

// =============================================================================
// Helpers
// =============================================================================

static constexpr int DEFAULT_TILE_M = 128;
static constexpr int DEFAULT_TILE_N = 128;
static constexpr int SCM_REG_OFFSET_CONV = 4096;
static constexpr int MC_SCM_PARAMS_PER_CORE = 22;

// Map (fsize_idx, tile_k) -> KernelId
// fsize_idx: 0=f1, 1=f3, 2=fn
// tile_k:    48, 64, 96, 128
static KernelId select_conv_kernel_id(int kh, int kw, int tile_k) {
    // fsize index
    int fs;
    if (kh == 1 && kw == 1) fs = 0;       // f1
    else if (kh == 3 && kw == 3) fs = 1;   // f3
    else fs = 2;                            // fn

    // tile_k index
    int tk;
    switch (tile_k) {
        case 48:  tk = 0; break;
        case 64:  tk = 1; break;
        case 96:  tk = 2; break;
        case 128: tk = 3; break;
        default:
            TORCH_CHECK(false, "rpu_conv2d: unsupported tile_k=", tile_k);
    }

    // KernelId layout: CONV_F1_W128X128_K48 .. K128, F3_K48..K128, FN_K48..K128
    // Each fsize block has 4 entries (k48, k64, k96, k128)
    return static_cast<KernelId>(
        static_cast<int>(KernelId::CONV_F1_W128X128_K48) + fs * 4 + tk);
}

// Select tile_k based on cin_per_group
// Returns the smallest tile_k that covers cin_per_group, within {48, 64, 96, 128}
static int select_tile_k(int cin_per_group) {
    int cin_v16 = CeilDiv(cin_per_group, 16);
    static const int candidates[] = {48, 64, 96, 128};
    for (int tk : candidates) {
        if (tk >= cin_v16 * 16) return tk;
    }
    return 128;  // fallback to largest registered tile_k
}

// Select tile_n for multi-core conv: largest of {128, 64, 32} that <= cpc
static int select_tile_n_mc(int cpc) {
    if (cpc >= 128) return 128;
    if (cpc >= 64) return 64;
    if (cpc >= 32) return 32;
    return 0;
}

// Select tile_k for multi-core conv based on cin
static int select_tile_k_mc(int cin) {
    if (cin >= 64) return 64;
    return 32;  // cin=16..48 → k32
}

// Map (fsize, tile_n, tile_k) -> multi-core KernelId
// Layout: k32 group [tn32_f1..fn, tn64_f1..fn, tn128_f1..fn],
//         k64 group [tn32_f1..fn, tn64_f1..fn, tn128_f1..fn]
static KernelId select_conv_mc_kernel_id(int kh, int kw, int tile_n, int tile_k) {
    int fs = (kh == 1 && kw == 1) ? 0 : (kh == 3 && kw == 3) ? 1 : 2;
    int tn = (tile_n == 32) ? 0 : (tile_n == 64) ? 1 : 2;
    int tk = (tile_k == 32) ? 0 : 1;  // 0=k32, 1=k64
    // Each tile_k group has 9 entries (3 tile_n × 3 fsize)
    return static_cast<KernelId>(
        static_cast<int>(KernelId::CONV_MC_F1_W128X32_K32) + tk * 9 + tn * 3 + fs);
}

// Compute conv output spatial dims
static inline void compute_conv_output_dims(
    int inh, int inw, int kh, int kw,
    int padh, int padH, int padw, int padW,
    int strideh, int stridew, int dilationh, int dilationw,
    int &outh, int &outw)
{
    int kh_eff = kh + (kh - 1) * (dilationh - 1);
    int kw_eff = kw + (kw - 1) * (dilationw - 1);
    outh = (inh + padh + padH - kh_eff) / strideh + 1;
    outw = (inw + padw + padW - kw_eff) / stridew + 1;
}

// Check whether the conv fits in single-core SPM (8MB)
bool rpu_conv2d_fits_in_spm(int batch, int inh, int inw, int cin,
                             int cout, int kh, int kw, int groups,
                             int outh, int outw)
{
    int cin_per_grp = cin / groups;
    int cout_per_grp = cout / groups;
    int tail_v16 = 1;

    int cin_per_grp_v16  = CeilDiv(cin_per_grp, 16);
    int cin_v16_tail     = cin_per_grp_v16 * groups + tail_v16;
    int cout_per_grp_v16 = CeilDiv(cout_per_grp, 16);
    int cout_v16_tail    = cout_per_grp_v16 * groups + tail_v16;
    int cin_per_grp_v16_tail = cin_per_grp_v16 + tail_v16;
    int flthwc_tail = kh * kw * cin_per_grp_v16_tail;

    size_t spm_in  = (size_t)batch * inh * inw * cin_v16_tail;
    size_t spm_flt = (size_t)cout * flthwc_tail;
    size_t spm_out = (size_t)batch * outh * outw * cout_v16_tail;
    size_t spm_bias = cout_v16_tail;
    size_t total_bytes = (spm_in + spm_flt + spm_out + spm_bias) * 32;

    constexpr size_t SPM_SIZE = 8 * 1024 * 1024;
    return total_bytes <= SPM_SIZE;
}

// =============================================================================
// Common register setup (shared by DDR and SPM versions)
// =============================================================================
static void setup_conv2d_regs(
    Kernel_t* kernel,
    uint32_t spm_in_v16, uint32_t spm_flt_v16,
    uint32_t spm_out_v16, uint32_t spm_bias_v16,
    int batch, int cin, int inh, int inw,
    int cout, int kh, int kw,
    int padh, int padH, int padw, int padW,
    int strideh, int stridew, int dilationh, int dilationw,
    int groups, bool has_bias,
    int outh, int outw, int tile_k,
    uint16_t grid_dim_x, uint16_t grid_dim_y, uint16_t grid_dim_z)
{
    int cin_per_grp = cin / groups;
    int cout_per_grp = cout / groups;
    int tail_v16 = 1;

    int cin_per_grp_v16 = CeilDiv(cin_per_grp, 16);
    int tile_k_v16 = CeilDiv(tile_k, 16);
    int cin_per_grp_v16_pad = CeilDiv(cin_per_grp_v16, tile_k_v16) * tile_k_v16;
    int cin_v16_tail = cin_per_grp_v16 * groups + tail_v16;
    int cin_per_grp_v16_tail = cin_per_grp_v16 + tail_v16;

    int cout_per_grp_v16 = CeilDiv(cout_per_grp, 16);
    int cout_v16_tail = cout_per_grp_v16 * groups + tail_v16;

    int inhw = inh * inw;
    int outhw = outh * outw;
    int flthw = kh * kw;

    kernel->reset_regs();

    // SPM addresses (low/high 16-bit)
    kernel->set_regs(0, (uint16_t)(spm_in_v16  & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(spm_in_v16  >> 16));
    kernel->set_regs(2, (uint16_t)(spm_flt_v16 & 0xFFFF));
    kernel->set_regs(3, (uint16_t)(spm_flt_v16 >> 16));
    kernel->set_regs(4, (uint16_t)(spm_out_v16 & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(spm_out_v16 >> 16));

    // Dimensions
    kernel->set_regs(6,  (uint16_t)inhw);
    kernel->set_regs(7,  (uint16_t)outhw);
    kernel->set_regs(8,  (uint16_t)flthw);
    kernel->set_regs(9,  (uint16_t)inh);
    kernel->set_regs(10, (uint16_t)inw);
    kernel->set_regs(11, (uint16_t)batch);
    kernel->set_regs(12, (uint16_t)groups);

    // Channel params
    kernel->set_regs(13, (uint16_t)cin_per_grp);
    kernel->set_regs(14, (uint16_t)cin_per_grp_v16);
    kernel->set_regs(15, (uint16_t)cin_per_grp_v16_pad);
    kernel->set_regs(16, (uint16_t)cin_v16_tail);
    kernel->set_regs(17, (uint16_t)kh);
    kernel->set_regs(18, (uint16_t)kw);
    kernel->set_regs(19, (uint16_t)cin_per_grp_v16_tail);

    // Output channel params
    kernel->set_regs(20, (uint16_t)cout_per_grp);
    kernel->set_regs(21, (uint16_t)cout_per_grp_v16);
    kernel->set_regs(22, (uint16_t)cout_v16_tail);

    // Output spatial
    kernel->set_regs(23, (uint16_t)outh);
    kernel->set_regs(24, (uint16_t)outw);

    // Stride
    kernel->set_regs(25, (uint16_t)strideh);
    kernel->set_regs(26, (uint16_t)stridew);

    // Padding (asymmetric top/bottom, left/right)
    kernel->set_regs(27, (uint16_t)padh);
    kernel->set_regs(28, (uint16_t)padH);
    kernel->set_regs(29, (uint16_t)padw);
    kernel->set_regs(30, (uint16_t)padW);

    // Dilation (hole)
    kernel->set_regs(31, (uint16_t)dilationh);
    kernel->set_regs(32, (uint16_t)dilationw);

    // Bias
    kernel->set_regs(33, (uint16_t)(has_bias ? 1 : 0));
    kernel->set_regs(34, (uint16_t)(spm_bias_v16 & 0xFFFF));
    kernel->set_regs(35, (uint16_t)(spm_bias_v16 >> 16));

    // Activation: 0=none
    kernel->set_regs(36, (uint16_t)0);
    // Element-wise add: 0=none
    kernel->set_regs(37, (uint16_t)0);
    kernel->set_regs(38, (uint16_t)0);
    kernel->set_regs(39, (uint16_t)0);
    kernel->set_regs(40, (uint16_t)0);

    // Grid
    kernel->set_regs(64, grid_dim_x);
    kernel->set_regs(65, grid_dim_y);
    kernel->set_regs(66, grid_dim_z);
}

// =============================================================================
// DDR version — accepts DDR pointers, DMA to SPM, execute, DMA back
// =============================================================================

void rpu_launch_conv2d_ddr_kernel(
    const c10::Half* input_nhwc,
    const c10::Half* weight_nhwc,
    const c10::Half* bias_ptr,
    c10::Half* output_nhwc,
    int batch, int cin, int inh, int inw,
    int cout, int kh, int kw,
    int padh, int padH, int padw, int padW,
    int strideh, int stridew,
    int dilationh, int dilationw,
    int groups, bool has_bias)
{
    int cin_per_grp  = cin / groups;
    int cout_per_grp = cout / groups;
    int tail_v16 = 1;

    int cin_per_grp_v16      = CeilDiv(cin_per_grp, 16);
    int cin_v16_tail         = cin_per_grp_v16 * groups + tail_v16;
    int cin_per_grp_v16_tail = cin_per_grp_v16 + tail_v16;
    int cout_per_grp_v16     = CeilDiv(cout_per_grp, 16);
    int cout_v16_tail        = cout_per_grp_v16 * groups + tail_v16;
    int flthwc_tail          = kh * kw * cin_per_grp_v16_tail;

    int outh, outw;
    compute_conv_output_dims(inh, inw, kh, kw, padh, padH, padw, padW,
                             strideh, stridew, dilationh, dilationw,
                             outh, outw);

    // SPM buffer sizes (v16 units)
    uint32_t spm_in_v16_size   = batch * inh * inw * cin_v16_tail;
    uint32_t spm_flt_v16_size  = cout * flthwc_tail;
    uint32_t spm_out_v16_size  = batch * outh * outw * cout_v16_tail;
    uint32_t spm_bias_v16_size = has_bias ? cout_v16_tail : 0;

    // SPM layout: [input | filter | output | bias]
    uint32_t spm_in_v16  = 0;
    uint32_t spm_flt_v16 = spm_in_v16  + spm_in_v16_size;
    uint32_t spm_out_v16 = spm_flt_v16 + spm_flt_v16_size;
    uint32_t spm_bias_v16_addr = spm_out_v16 + spm_out_v16_size;

    size_t dwidth_v16 = 32;  // sizeof(fp16) * 16

    // Allocate SPM buffers on core 0
    LocalSPM_t spm_input(Align(spm_in_v16_size * dwidth_v16, SPM_BANK_SIZE),
                          read_write, kStride32B, 2, 0);
    RPU_CHECK_SPM(spm_input, Align(spm_in_v16_size * dwidth_v16, SPM_BANK_SIZE));
    LocalSPM_t spm_filter(Align(spm_flt_v16_size * dwidth_v16, SPM_BANK_SIZE),
                           read_write, kStride32B, 2, 0);
    RPU_CHECK_SPM(spm_filter, Align(spm_flt_v16_size * dwidth_v16, SPM_BANK_SIZE));
    LocalSPM_t spm_output(Align(spm_out_v16_size * dwidth_v16, SPM_BANK_SIZE),
                           read_write, kStride32B, 2, 0);
    RPU_CHECK_SPM(spm_output, Align(spm_out_v16_size * dwidth_v16, SPM_BANK_SIZE));

    // DMA: DDR -> SPM
    ddr_to_spm(spm_input.get_cpu_ptr(), input_nhwc,
               spm_in_v16_size * dwidth_v16);
    ddr_to_spm(spm_filter.get_cpu_ptr(), weight_nhwc,
               spm_flt_v16_size * dwidth_v16);

    LocalSPM_t* spm_bias_buf = nullptr;
    if (has_bias) {
        spm_bias_buf = graph_acquire_local_spm(
            Align(spm_bias_v16_size * dwidth_v16, SPM_BANK_SIZE),
            read_write, kStride32B, 2, 0);
        ddr_to_spm(spm_bias_buf->get_cpu_ptr(), bias_ptr,
                   spm_bias_v16_size * dwidth_v16);
    }

    // Kernel selection via KernelId
    int tile_k = select_tile_k(cin_per_grp);
    KernelId kid = select_conv_kernel_id(kh, kw, tile_k);
    Kernel_t* kernel = GET_KERNEL(kid);
    TORCH_CHECK(kernel, "rpu_conv2d_ddr: kernel not found for KernelId=",
                static_cast<int>(kid));

    int outhw = outh * outw;
    uint16_t grid_dim_x = CeilDiv(cout_per_grp, DEFAULT_TILE_N);
    uint16_t grid_dim_y = CeilDiv(outhw, DEFAULT_TILE_M);
    uint16_t grid_dim_z = batch * groups;

    setup_conv2d_regs(kernel,
        spm_in_v16, spm_flt_v16, spm_out_v16,
        has_bias ? spm_bias_v16_addr : 0,
        batch, cin, inh, inw, cout, kh, kw,
        padh, padH, padw, padW,
        strideh, stridew, dilationh, dilationw,
        groups, has_bias, outh, outw, tile_k,
        grid_dim_x, grid_dim_y, grid_dim_z);

    // Launch on core 0
    auto* wq = GET_QUEUE(1);
    wq->enqueu_kernel(*kernel, {grid_dim_x, grid_dim_y, grid_dim_z}, {0});

    // DMA: SPM -> DDR
    spm_to_ddr(output_nhwc, spm_output.get_cpu_ptr(),
               spm_out_v16_size * dwidth_v16);

    graph_release_local_spm(spm_bias_buf);
}

// =============================================================================
// Multi-core conv2d (ConvCorenColumn)
//
// Column partition: input split by batch, weight split by cout.
// Key: kernel reads filter via SCM[core_col_idx], enabling cross-core filter
// access. Each core produces the FULL output (all channels) for its batch slice.
//
// Shared regs: [0] kh [1] kw [2] flthw [3] core_num [4] cin [5] cin_v16
//   [6] cin_v16_pad [10] cpc [11] cpc_v16 [12] num_flt_v16
//   [13] sh [14] sw [15] dh [16] dw [17] bias [18] act [19] elt [20] eltAct
//   [64] grid_x [65] grid_y [66] grid_z
//
// SCM (set_regs 4096 + j*22 + offset, per core j):
//   [0:1] spm_in [2:3] spm_flt [4:5] spm_out [6:7] spm_bias [8:9] spm_elt
//   [10] inh [11] inw [12] outh [13] outw [14] inhw [15] outhw
//   [16] padh [17] padH [18] padw [19] padW [20] batch_per_core [21] reserved
// =============================================================================

bool rpu_conv2d_mc_fits_in_spm(int batch, int inh, int inw, int cin,
                                int cout, int kh, int kw,
                                int outh, int outw, int core_num)
{
    int cin_v16 = CeilDiv(cin, 16);
    constexpr int MC_TK_V16 = 4;  // tile_k=64 → 4 v16
    int bpc = CeilDiv(batch, core_num);
    int cpc = cout / core_num;
    int cpc_v16 = CeilDiv(cpc, 16);
    int num_flt_v16 = cpc_v16 * core_num;
    int flthw = kh * kw;
    int outhw = outh * outw;

    size_t spm_in  = (size_t)bpc * inh * inw * cin_v16;
    size_t spm_flt = (size_t)flthw * cin_v16 * cpc;
    size_t spm_out = (size_t)bpc * outhw * num_flt_v16;
    size_t spm_bias = num_flt_v16;
    size_t total_bytes = (spm_in + spm_flt + spm_out + spm_bias) * 32;

    constexpr size_t SPM_SIZE = 8 * 1024 * 1024;
    return total_bytes <= SPM_SIZE;
}

// Static SPM cache for multi-core conv — uses SpmAllocator (offset-based)
#include "rpu_spm_allocator.h"

static struct McConvSpmCache {
    // Offsets (same for all cores)
    uint32_t in_off = 0, flt_off = 0, out_off = 0, bias_off = 0;
    size_t in_bytes = 0, flt_bytes = 0, out_bytes = 0, bias_bytes = 0;
    int core_num = 0;
    bool weight_loaded = false;
    bool bias_loaded = false;
    bool allocated = false;
    uint64_t alloc_gen = 0;

    void ensure(int nc, size_t ib, size_t fb, size_t ob, size_t bb) {
        if (allocated && alloc_gen == SPM_ALLOC.generation() &&
            core_num == nc && in_bytes == ib && flt_bytes == fb &&
            out_bytes == ob && bias_bytes == bb) return;

        if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

        // Conv2d 使用 temporary（phase 结束后释放）
        // 如果之前分配过，需要 reset 再重新分配
        if (allocated) SPM_ALLOC.reset_temporary();

        core_num = nc;
        in_bytes = ib; flt_bytes = fb; out_bytes = ob; bias_bytes = bb;
        in_off   = SPM_ALLOC.alloc_temporary(ib);
        flt_off  = SPM_ALLOC.alloc_temporary(fb);
        out_off  = SPM_ALLOC.alloc_temporary(ob);
        bias_off = (bb > 0) ? SPM_ALLOC.alloc_temporary(bb) : 0;
        weight_loaded = false;
        bias_loaded = false;
        allocated = true;
        alloc_gen = SPM_ALLOC.generation();
    }

    void evict_all() {
        if (allocated) SPM_ALLOC.reset_temporary();
        in_off = flt_off = out_off = bias_off = 0;
        core_num = 0;
        weight_loaded = false;
        bias_loaded = false;
        allocated = false;
    }
} g_mc_conv_spm_cache;

void rpu_launch_conv2d_multicore_ddr_kernel(
    const c10::Half* input_nhwc,   // DDR [batch, inh, inw, cin_padded]
    const c10::Half* weight_nhwc,  // DDR [cout, kh, kw, cin_padded]
    const c10::Half* bias_ptr,     // DDR [num_flt_v16 * 16] or nullptr
    c10::Half* output_nhwc,        // DDR [batch, outh, outw, num_flt_v16 * 16]
    int batch, int cin, int inh, int inw,
    int cout, int kh, int kw,
    int padh, int padH, int padw, int padW,
    int strideh, int stridew,
    int dilationh, int dilationw,
    bool has_bias, int core_num)
{
    int outh, outw;
    compute_conv_output_dims(inh, inw, kh, kw, padh, padH, padw, padW,
                             strideh, stridew, dilationh, dilationw,
                             outh, outw);

    int cin_v16 = CeilDiv(cin, 16);
    int mc_tile_k = select_tile_k_mc(cin);
    int tile_k_v16 = CeilDiv(mc_tile_k, 16);
    int cin_v16_pad = CeilDiv(cin_v16, tile_k_v16) * tile_k_v16;

    int bpc = CeilDiv(batch, core_num);
    int cpc = cout / core_num;
    int cpc_v16 = CeilDiv(cpc, 16);
    int num_flt_v16 = cpc_v16 * core_num;

    int flthw = kh * kw;
    int outhw = outh * outw;
    int inhw = inh * inw;

    constexpr size_t DW16 = 32;  // bytes per v16

    // Per-core SPM sizes (v16 units)
    uint32_t spm_in_v16_size  = bpc * inh * inw * cin_v16;
    uint32_t spm_flt_v16_size = flthw * cin_v16 * cpc;
    uint32_t spm_out_v16_size = bpc * outhw * num_flt_v16;
    uint32_t spm_bias_v16_size = has_bias ? num_flt_v16 : 0;

    // SPM allocation — use static cache to avoid re-allocation per call
    size_t in_bytes  = Align(spm_in_v16_size * DW16, SPM_BANK_SIZE);
    size_t flt_bytes = Align(spm_flt_v16_size * DW16, SPM_BANK_SIZE);
    size_t out_bytes = Align(spm_out_v16_size * DW16, SPM_BANK_SIZE);
    size_t bias_bytes = has_bias ? Align(spm_bias_v16_size * DW16, SPM_BANK_SIZE) : 0;

    g_mc_conv_spm_cache.ensure(core_num, in_bytes, flt_bytes, out_bytes, bias_bytes);
    uint32_t in_off   = g_mc_conv_spm_cache.in_off;
    uint32_t flt_off  = g_mc_conv_spm_cache.flt_off;
    uint32_t out_off  = g_mc_conv_spm_cache.out_off;
    uint32_t bias_off = g_mc_conv_spm_cache.bias_off;

    // DDR strides
    size_t in_batch_bytes  = (size_t)inh * inw * cin_v16 * DW16;
    size_t flt_core_bytes  = (size_t)cpc * kh * kw * cin_v16 * DW16;
    size_t out_core_bytes  = (size_t)bpc * outhw * num_flt_v16 * DW16;

    for (int c = 0; c < core_num; c++) {
        // DMA input batch slice (every call — input changes)
        ddr_to_spm(SPM_ALLOC.cpu_ptr(c, in_off),
                   reinterpret_cast<const char*>(input_nhwc) + c * bpc * in_batch_bytes,
                   spm_in_v16_size * DW16);

        // DMA weight (first call only — weight is constant)
        if (!g_mc_conv_spm_cache.weight_loaded) {
            ddr_to_spm(SPM_ALLOC.cpu_ptr(c, flt_off),
                       reinterpret_cast<const char*>(weight_nhwc) + c * flt_core_bytes,
                       spm_flt_v16_size * DW16);
        }

        // DMA bias (first call only — bias is constant)
        if (has_bias && !g_mc_conv_spm_cache.bias_loaded) {
            ddr_to_spm(SPM_ALLOC.cpu_ptr(c, bias_off), bias_ptr,
                       spm_bias_v16_size * DW16);
        }
    }
    g_mc_conv_spm_cache.weight_loaded = true;
    if (has_bias) g_mc_conv_spm_cache.bias_loaded = true;

    // Kernel selection: tile_n based on cpc, tile_k based on cin
    int tile_n = select_tile_n_mc(cpc);
    TORCH_CHECK(tile_n > 0, "rpu_conv2d_mc: cpc=", cpc, " too small for any tile_n");
    KernelId kid = select_conv_mc_kernel_id(kh, kw, tile_n, mc_tile_k);
    Kernel_t* kernel = GET_KERNEL(kid);
    TORCH_CHECK(kernel, "rpu_conv2d_mc: kernel not found for KernelId=",
                static_cast<int>(kid));

    uint16_t grid_dim_x = CeilDiv(cpc, tile_n);
    uint16_t grid_dim_y = CeilDiv(outhw, DEFAULT_TILE_M);
    uint16_t grid_dim_z = bpc * core_num;

    // Shared registers
    kernel->reset_regs();
    kernel->set_regs(0, (uint16_t)kh);
    kernel->set_regs(1, (uint16_t)kw);
    kernel->set_regs(2, (uint16_t)flthw);
    kernel->set_regs(3, (uint16_t)core_num);
    kernel->set_regs(4, (uint16_t)cin);
    kernel->set_regs(5, (uint16_t)cin_v16);
    kernel->set_regs(6, (uint16_t)cin_v16_pad);
    kernel->set_regs(10, (uint16_t)cpc);
    kernel->set_regs(11, (uint16_t)cpc_v16);
    kernel->set_regs(12, (uint16_t)num_flt_v16);
    kernel->set_regs(13, (uint16_t)strideh);
    kernel->set_regs(14, (uint16_t)stridew);
    kernel->set_regs(15, (uint16_t)dilationh);
    kernel->set_regs(16, (uint16_t)dilationw);
    kernel->set_regs(17, (uint16_t)(has_bias ? 1 : 0));
    kernel->set_regs(18, (uint16_t)0);
    kernel->set_regs(19, (uint16_t)0);
    kernel->set_regs(20, (uint16_t)0);
    kernel->set_regs(64, grid_dim_x);
    kernel->set_regs(65, grid_dim_y);
    kernel->set_regs(66, grid_dim_z);

    // Per-core SCM: set_regs(4096 + j*22 + offset)
    // Kernel reads filter from SCM[core_col_idx*22] for cross-core access
    for (int j = 0; j < core_num; j++) {
        int base = SCM_REG_OFFSET_CONV + j * MC_SCM_PARAMS_PER_CORE;
        uint32_t in_v16   = SPM_ALLOC.addr(j, in_off) / DW16;
        uint32_t flt_v16  = SPM_ALLOC.addr(j, flt_off) / DW16;
        uint32_t out_v16  = SPM_ALLOC.addr(j, out_off) / DW16;
        uint32_t bias_v16 = has_bias ? (SPM_ALLOC.addr(j, bias_off) / DW16) : 0;

        kernel->set_regs(base + 0,  (uint16_t)(in_v16 & 0xFFFF));
        kernel->set_regs(base + 1,  (uint16_t)(in_v16 >> 16));
        kernel->set_regs(base + 2,  (uint16_t)(flt_v16 & 0xFFFF));
        kernel->set_regs(base + 3,  (uint16_t)(flt_v16 >> 16));
        kernel->set_regs(base + 4,  (uint16_t)(out_v16 & 0xFFFF));
        kernel->set_regs(base + 5,  (uint16_t)(out_v16 >> 16));
        kernel->set_regs(base + 6,  (uint16_t)(bias_v16 & 0xFFFF));
        kernel->set_regs(base + 7,  (uint16_t)(bias_v16 >> 16));
        kernel->set_regs(base + 8,  (uint16_t)0);
        kernel->set_regs(base + 9,  (uint16_t)0);
        kernel->set_regs(base + 10, (uint16_t)inh);
        kernel->set_regs(base + 11, (uint16_t)inw);
        kernel->set_regs(base + 12, (uint16_t)outh);
        kernel->set_regs(base + 13, (uint16_t)outw);
        kernel->set_regs(base + 14, (uint16_t)inhw);
        kernel->set_regs(base + 15, (uint16_t)outhw);
        kernel->set_regs(base + 16, (uint16_t)padh);
        kernel->set_regs(base + 17, (uint16_t)padH);
        kernel->set_regs(base + 18, (uint16_t)padw);
        kernel->set_regs(base + 19, (uint16_t)padW);
        kernel->set_regs(base + 20, (uint16_t)bpc);
        kernel->set_regs(base + 21, (uint16_t)0);
    }

    // Launch on all cores with broadcast mode
    auto* wq = GET_QUEUE(UNIFIED_NUM_CORES);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> core_ids(core_num);
    for (int i = 0; i < core_num; i++) core_ids[i] = (uint8_t)i;
    wq->enqueu_kernel(*kernel, {grid_dim_x, grid_dim_y, grid_dim_z}, core_ids);

    // DMA output: each core has full-width output for its batch slice
    for (int c = 0; c < core_num; c++) {
        spm_to_ddr(reinterpret_cast<char*>(output_nhwc) + c * out_core_bytes,
                   SPM_ALLOC.cpu_ptr(c, out_off),
                   spm_out_v16_size * DW16);
    }

    // SPM buffers kept in g_mc_conv_spm_cache — no release per call
}

void rpu_release_conv2d_spm_cache() {
    RpuExecutionCleanupGuard cleanup("release_conv2d_spm_cache");
    g_mc_conv_spm_cache.evict_all();
}

// =============================================================================
// PyTorch convolution_overrideable wrapper
// Accepts NCHW tensors, handles layout conversion, falls back to CPU when needed
// Single-core if fits, multi-core if needed, CPU fallback as last resort.
// =============================================================================

at::Tensor rpu_conv2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups)
{
    // --- Gate: fall back to CPU for unsupported cases ---
    if (transposed || input.scalar_type() != at::kHalf) {
        goto cpu_fallback;
    }

    {
        int64_t N    = input.size(0);
        int64_t Cin  = input.size(1);
        int64_t H    = input.size(2);
        int64_t W    = input.size(3);
        int64_t Cout = weight.size(0);
        int64_t Kh   = weight.size(2);
        int64_t Kw   = weight.size(3);

        int64_t sh = stride[0], sw = stride.size() > 1 ? stride[1] : stride[0];
        int64_t ph = padding[0], pw = padding.size() > 1 ? padding[1] : padding[0];
        int64_t dh = dilation[0], dw = dilation.size() > 1 ? dilation[1] : dilation[0];

        // cin must be divisible by 16 for RPU kernel
        if (Cin % 16 != 0) {
            if (log_at(2))
                std::cout << "[RPU] CONV2D CPU_FALLBACK: cin=" << Cin
                          << " not divisible by 16" << std::endl;
            goto cpu_fallback;
        }

        int outh, outw;
        compute_conv_output_dims(H, W, Kh, Kw, ph, ph, pw, pw,
                                 sh, sw, dh, dw, outh, outw);

        bool has_bias = bias_opt.has_value() && bias_opt->defined();

        // -------- Try single-core path (group conv supported) --------
        if (groups == 1 || (Cin / groups) % 16 == 0) {
            int tile_k = select_tile_k(Cin / groups);
            if (tile_k <= 128 &&
                rpu_conv2d_fits_in_spm(N, H, W, Cin, Cout, Kh, Kw, groups, outh, outw)) {

                KernelId kid = select_conv_kernel_id(Kh, Kw, tile_k);
                if (GET_KERNEL(kid)) {
                    if (log_at(4))
                        std::cout << "[RPU] CONV2D single-core" << std::endl;

                    // NCHW -> NHWC
                    at::Tensor input_nhwc  = input.permute({0, 2, 3, 1}).contiguous();
                    at::Tensor weight_nhwc = weight.permute({0, 2, 3, 1}).contiguous();

                    int tail_v16 = 1;
                    int cin_per_grp = Cin / groups;
                    int cin_per_grp_v16 = CeilDiv(cin_per_grp, 16);
                    int cin_v16_tail = cin_per_grp_v16 * groups + tail_v16;
                    int cin_v16_tail_elems = cin_v16_tail * 16;
                    int cin_per_grp_v16_tail = cin_per_grp_v16 + tail_v16;
                    int weight_cin_padded = cin_per_grp_v16_tail * 16;
                    int cout_per_grp_v16 = CeilDiv((int)(Cout / groups), 16);
                    int cout_v16_tail = cout_per_grp_v16 * groups + tail_v16;
                    int cout_v16_tail_elems = cout_v16_tail * 16;

                    at::Tensor input_padded;
                    if (cin_v16_tail_elems != Cin) {
                        input_padded = at::zeros({N, H, W, cin_v16_tail_elems}, input.options());
                        input_padded.slice(3, 0, Cin).copy_(input_nhwc);
                    } else {
                        input_padded = input_nhwc;
                    }

                    at::Tensor weight_padded;
                    if (weight_cin_padded != cin_per_grp) {
                        weight_padded = at::zeros({Cout, Kh, Kw, weight_cin_padded}, weight.options());
                        weight_padded.slice(3, 0, cin_per_grp).copy_(weight_nhwc);
                    } else {
                        weight_padded = weight_nhwc;
                    }

                    at::Tensor bias_padded;
                    c10::Half* bias_ptr = nullptr;
                    if (has_bias) {
                        if (cout_v16_tail_elems != Cout) {
                            bias_padded = at::zeros({cout_v16_tail_elems}, bias_opt->options());
                            bias_padded.slice(0, 0, Cout).copy_(*bias_opt);
                        } else {
                            bias_padded = bias_opt->contiguous();
                        }
                        bias_ptr = bias_padded.data_ptr<c10::Half>();
                    }

                    at::Tensor output_padded = at::empty(
                        {N, outh, outw, cout_v16_tail_elems}, input.options());

                    rpu_ddr_flush(input_padded.data_ptr());
                    rpu_ddr_flush(weight_padded.data_ptr());
                    if (has_bias) rpu_ddr_flush(bias_ptr);

                    rpu_launch_conv2d_ddr_kernel(
                        input_padded.data_ptr<c10::Half>(),
                        weight_padded.data_ptr<c10::Half>(),
                        bias_ptr,
                        output_padded.data_ptr<c10::Half>(),
                        N, Cin, H, W, Cout, Kh, Kw,
                        ph, ph, pw, pw, sh, sw, dh, dw,
                        groups, has_bias);

                    rpu_ddr_flush(output_padded.data_ptr());
                    at::Tensor output_valid = output_padded.slice(3, 0, Cout);
                    return output_valid.permute({0, 3, 1, 2}).contiguous();
                }
            }
        }

        // -------- Try multi-core path (group==1, cin%16==0) --------
        if (groups == 1 && Cin % 16 == 0 && Cout % 16 == 0) {
            for (int nc = UNIFIED_NUM_CORES; nc >= 2; nc--) {
                if (N % nc != 0 || Cout % nc != 0) continue;

                int cpc_mc = Cout / nc;
                int tn_mc = select_tile_n_mc(cpc_mc);
                if (tn_mc == 0) continue;

                if (!rpu_conv2d_mc_fits_in_spm(N, H, W, Cin, Cout, Kh, Kw,
                                                outh, outw, nc))
                    continue;

                int tk_mc = select_tile_k_mc(Cin);
                KernelId kid_mc = select_conv_mc_kernel_id(Kh, Kw, tn_mc, tk_mc);
                if (!GET_KERNEL(kid_mc)) continue;

                if (log_at(4))
                    std::cout << "[RPU] CONV2D multi-core (" << nc << " cores)"
                              << std::endl;

                at::Tensor input_nhwc  = input.permute({0, 2, 3, 1}).contiguous();
                at::Tensor weight_nhwc = weight.permute({0, 2, 3, 1}).contiguous();

                int cin_v16 = CeilDiv((int)Cin, 16);
                int cin_padded = cin_v16 * 16;
                int cpc_v16 = CeilDiv((int)(Cout / nc), 16);
                int num_flt_v16 = cpc_v16 * nc;
                int num_flt_v16_elems = num_flt_v16 * 16;

                at::Tensor input_padded;
                if (cin_padded != Cin) {
                    input_padded = at::zeros({N, H, W, cin_padded}, input.options());
                    input_padded.slice(3, 0, Cin).copy_(input_nhwc);
                } else {
                    input_padded = input_nhwc;
                }

                at::Tensor weight_padded;
                if (cin_padded != Cin) {
                    weight_padded = at::zeros({Cout, Kh, Kw, cin_padded}, weight.options());
                    weight_padded.slice(3, 0, Cin).copy_(weight_nhwc);
                } else {
                    weight_padded = weight_nhwc;
                }

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

                at::Tensor output_padded = at::empty(
                    {N, outh, outw, num_flt_v16_elems}, input.options());

                rpu_ddr_flush(input_padded.data_ptr());
                rpu_ddr_flush(weight_padded.data_ptr());
                if (has_bias) rpu_ddr_flush(bias_ptr);

                rpu_launch_conv2d_multicore_ddr_kernel(
                    input_padded.data_ptr<c10::Half>(),
                    weight_padded.data_ptr<c10::Half>(),
                    bias_ptr,
                    output_padded.data_ptr<c10::Half>(),
                    N, Cin, H, W, Cout, Kh, Kw,
                    ph, ph, pw, pw, sh, sw, dh, dw,
                    has_bias, nc);

                rpu_ddr_flush(output_padded.data_ptr());
                at::Tensor output_valid = output_padded.slice(3, 0, Cout);
                return output_valid.permute({0, 3, 1, 2}).contiguous();
            }
        }

        if (log_at(2))
            std::cout << "[RPU] CONV2D CPU_FALLBACK: no RPU path available" << std::endl;
    }

cpu_fallback:
    {
        if (log_at(2))
            std::cout << "[RPU] CONV2D CPU_FALLBACK" << std::endl;
        auto input_dtype = input.scalar_type();
        auto cpu_input  = rpu_to_cpu_zerocopy(input).to(at::kFloat);
        auto cpu_weight = rpu_to_cpu_zerocopy(weight).to(at::kFloat);
        c10::optional<at::Tensor> cpu_bias;
        if (bias_opt.has_value() && bias_opt->defined())
            cpu_bias = rpu_to_cpu_zerocopy(*bias_opt).to(at::kFloat);
        auto cpu_out = at::convolution(cpu_input, cpu_weight, cpu_bias,
                                       stride, padding, dilation,
                                       transposed, output_padding, groups);
        return cpu_out.to(input_dtype).to(input.device());
    }
}
