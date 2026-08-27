// rpu_kernel_cache.h — KernelId enum, kernel name table, KernelCache, QueueCache
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <c10/util/Exception.h>   // TORCH_CHECK — rpu_add_dma_checked
#include "rhino_launch_program.h"
#include "rhino_launch_kernel.h"
#include "rhino_launch_queue.h"
#include "rpu_dma_endpoint.h"

// =============================================================================
// Kernel ID 枚举 - 用于 O(1) 数组索引访问 (比 unordered_map 快 10x+)
// =============================================================================
enum class KernelId : uint8_t {
    // Eltwise binary kernels - SPM versions
    BINARY_SAMESHAPE = 0,
    BINARY_NX1_NXC256_BATCH,
    BINARY_NX1_NXC256_BATCH_BOPA,
    BINARY_NX1_NXC_V16_BATCH,
    BINARY_NX1_NXC_V16_BATCH_BOPA,
    BINARY_1XC_NXC_V256_BATCH,
    BINARY_1XC_NXC_V256_BATCH_BOPA,
    BINARY_1XC_NXC_V16_BATCH,
    BINARY_1XC_NXC_V16_BATCH_BOPA,
    // Eltwise binary kernels - DDR versions
    BINARY_SAMESHAPE_DDR,
    BINARY_SCALAR_DDR,
    BINARY_SCALAR,              // SPM version of binary_scalar
    BINARY_SCALAR_POWER_DDR,
    BINARY_NX1_NXC256_BATCH_DDR,
    BINARY_NX1_NXC256_BATCH_BOPA_DDR,
    BINARY_NX1_NXC_V16_BATCH_DDR,
    BINARY_NX1_NXC_V16_BATCH_BOPA_DDR,
    BINARY_1XC_NXC_V256_BATCH_DDR,
    BINARY_1XC_NXC_V256_BATCH_BOPA_DDR,
    BINARY_1XC_NXC_V16_BATCH_DDR,
    BINARY_1XC_NXC_V16_BATCH_BOPA_DDR,
    // Eltwise unary kernels
    UNARY_DDR,
    // Other kernels
    SOFTMAX_C16_GAUTO,
    LAYER_NORM,
    LAYER_NORM_SIMPLE,
    LAYER_NORM_BF16,
    SDPA_FLASH_ATTN_SPM,  // SPM version: Q input and output in SPM
    SDPA_FLASH_ATTN_SPM_16B,
    REDUCE_MEAN_LAST_DIM_DDR,
    // RMSNorm kernels admitted by the current operator contract.
    RMS_NORM_BF16,
    RMS_NORM_BF16_SPM,  // SPM版本: input/output在SPM
    // Newton-refined rsqrt variant (llama_rms_norm_newton): identical reg ABI to
    // llama_rms_norm, adds a Newton-Raphson step to the base approximate-rsqrt path
    // for near-fp32 rsqrt precision. Selected at runtime by RPU_RMSNORM_NEWTON.
    RMS_NORM_NEWTON_SPM,
    // RoPE kernel
    ROPE,
    ROPE_DDR,
    // M-RoPE kernel for Qwen3-VL text execution:
    //   position_ids live in DDR (typical prefill/decode path); cos/sin tables
    //   share the 1D RoPE [max_seq, head_dim/2] format. Per-axis (T/H/W)
    //   selection is performed at runtime via strobe masks set by the launcher.
    MROPE,
    // Partial M-RoPE kernel (wall-oss prefill/denoise perf): host-precomputed
    //   per-token interleaved cos/sin [seq, rotary_dim/2] streamed in DDR — no
    //   in-kernel position_ids gather / strobe masks. Bit-exact to MROPE @ factor
    //   1.0. See src/ops/rpu_mrope.cpp::rpu_launch_partial_mrope_spm_kernel.
    PARTIAL_MROPE,
    // 2D RoPE kernel for the Qwen3-VL vision encoder:
    //   ROPE_2D_DDR: FreqCos / FreqSin tables live in DDR (vision-encoder default).
    //   ROPE_2D_SPM: tables live in SPM (perf variant for small max_hw, currently
    //   reserved — vision encoder uses the DDR variant only).
    //   position_idx is [num_tokens, 2] int16 DDR (row, col pair per patch).
    ROPE_2D_DDR,
    ROPE_2D_SPM,
    // Fast bilinear positional-embedding interpolation (Qwen3-VL / Qwen3.5
    // vision STEP 0). Inputs live in DDR; output lives in SPM.
    FAST_POS_EMBED_INTERP,
    // LLaMA KV-Cache kernels
    LLAMA_INSERT_VCACHE,
    LLAMA_INSERT_KCACHE,
    LLAMA_INSERT_VCACHE_V16,   // 16 tokens/warp, aligned prefill (pos%16==0, seq%16==0)
    LLAMA_INSERT_KCACHE_V16,
    // Standalone all-gather kernels. All-reduce uses the fused ring family below.
    ALL_GATHER_MULTI_CORE,
    // Small-chunk schedule of the SAME all-gather operation. Both kernels are
    // full 2D strided gathers; they differ only in the parallelization axis.
    // multi_core maps threads to byte offsets within a chunk and serializes over
    // rows; little_chunk maps threads to rows and serializes over chunk bytes.
    // rhino-ops picks by chunk width: <= 1024B -> little_chunk (below that,
    // multi_core leaves most of its 128 threads idle). See rpu_all_gather.cpp.
    ALL_GATHER_MULTI_CORE_LITTLE_CHUNK,
    // SPM unary kernels (for fused decoder layer MLP)
    UNARY_SPM,
    // DDR to SPM memcpy (multi-core)
    // SPM to DDR memcpy (multi-core)
    // DDR/SPM memcpy v2 variants support large transfers safely.
    // Argmax/Argmin kernels
    ARGMAX_REDUCEC_TILEN,
    ARGMAX_REDUCEC_TILEN_C128,
    ARGMAX_REDUCEC_TILEN_SLI,
    ARGMAX_REDUCEC_TILEN_C128_SLI,
    ARGMAX_REDUCEN_TILEC,
    ARGMAX_REDUCEN_TILEC_SLI,
    ARGMIN_REDUCEC_TILEN,
    ARGMIN_REDUCEC_TILEN_C128,
    ARGMIN_REDUCEC_TILEN_SLI,
    ARGMIN_REDUCEC_TILEN_C128_SLI,
    ARGMIN_REDUCEN_TILEC,
    ARGMIN_REDUCEN_TILEC_SLI,
    // SPM memset (zero-fill) multi-core
    MEMSET_SPM_MULTI_CORE_V2,
    // SDPA vctxlen (dynamic context length)
    SDPA_FLASH_ATTN_SPM_VCTXLEN,
    // SDPA vctxlen minibatch — per-image K/V slicing (image_batch_count in
    // reg[34]); each image's queries attend only its own per_image_ctx_len K/V.
    // Requires nKVHead == numHead (MHA). Used by multi-image vision packing.
    SDPA_FLASH_ATTN_SPM_VCTXLEN_MINIBATCH,
    SDPA_FLASH_ATTN_SPM_VCTXLEN_MINIBATCH_16B,
    // Fixed-shape 16-bank variants used by the InternVLA standalone path.
    // Conv2d SPM kernels (1core, tile_m=128, tile_n=128, univ)
    CONV_F1_W128X128_K48,
    CONV_F1_W128X128_K64,
    CONV_F1_W128X128_K96,
    CONV_F1_W128X128_K128,
    CONV_F3_W128X128_K48,
    CONV_F3_W128X128_K64,
    CONV_F3_W128X128_K96,
    CONV_F3_W128X128_K128,
    CONV_FN_W128X128_K48,
    CONV_FN_W128X128_K64,
    CONV_FN_W128X128_K96,
    CONV_FN_W128X128_K128,
    // Conv2d multi-core column kernels (coren, univ)
    // tile_k=32 group
    CONV_MC_F1_W128X32_K32, CONV_MC_F3_W128X32_K32, CONV_MC_FN_W128X32_K32,
    CONV_MC_F1_W128X64_K32, CONV_MC_F3_W128X64_K32, CONV_MC_FN_W128X64_K32,
    CONV_MC_F1_W128X128_K32, CONV_MC_F3_W128X128_K32, CONV_MC_FN_W128X128_K32,
    // tile_k=64 group
    CONV_MC_F1_W128X32_K64, CONV_MC_F3_W128X32_K64, CONV_MC_FN_W128X32_K64,
    CONV_MC_F1_W128X64_K64, CONV_MC_F3_W128X64_K64, CONV_MC_FN_W128X64_K64,
    CONV_MC_F1_W128X128_K64, CONV_MC_F3_W128X128_K64, CONV_MC_FN_W128X128_K64,
    // Im2col kernels (multi-core, SPM in → SPM out)
    IM2COL_M64XN32,
    IM2COL_M128XN32,
    IM2COL_M64XN64,
    IM2COL_M128XN64,
    // Pad kernel (constant fill, SPM in → SPM out)
    PAD_CONST_NXC,
    // Transpose kernel (3D, SPM in → SPM out)
    TRANSPOSE_NCB_C16,
    TRANSPOSE_NBC_C16,
    PARALLEL_LINEAR_WNVFP4A16_ACC16,     // canonical NVFP4 correctness fallback
    PARALLEL_LINEAR_WNVFP4A16_ACC32,     // fixed m128/n128 NVFP4 ACC32 family
    // Qwen3.5 GDN kernels. The auto-tile block below must remain contiguous
    // because rpu_linear.cpp indexes it by arithmetic offset. This enum order
    // must mirror KERNEL_ID_NAMES.
    // Qwen3.5 partial RoPE 1D — DECODE. Same partial rotate-half, but position-lookup
    // cos/sin (grid: NUM_ELT_PER_WRP=512, grid_y=num_tokens). Bound to kernel
    // "partial_rope_1d" in the matching operator asset.
    PARTIAL_ROPE_1D,
    // Qwen3.5 GDN decode kernels in the matching operator asset.
    L2NORM,
    FLA_CONV1D,
    // Qwen3.5 GDN gated-norm: fused silu(x)*y (default ref).
    LLAMA_SILU_MUL,
    // Sum over rows (axis 0) of [rows, cols<256] → [cols] (default ref).
    REDUCE_SUM_NON_LAST_DIM_OUT_SEG,
    // Qwen3.5 GDN prefill chunk: cumulative sum along last dim (default ref).
    CUMSUM,
    // Qwen3.5 GDN prefill chunk: (I - M)^-1 for strict lower-tri 64x64 M (default ref).
    UNIT_TRIL_INV,
    // --- unconditional generated auto-tile acc16 variants --------------------
    // 7 w8a16 + 7 fp16 per-shape tiled parallel_linear[_w8a16]_m{m}n{n}k128. The
    // order (w8a16 then fp16, n_tile 128->32) is LOAD-BEARING: rpu_linear.cpp's
    // autotile_linear_kernel_id() indexes by arithmetic offset from
    // PL_AT_W8A16_M288N128. Names in KERNEL_ID_NAMES below MUST match
    // rpu_pl_tiling::autotile_kernel_name() exactly. m_tile = mtile_w8a16/_w16(n).
    PL_AT_W8A16_M288N128,   // n_tile=128
    PL_AT_W8A16_M320N112,   // n_tile=112
    PL_AT_W8A16_M352N96,    // n_tile=96
    PL_AT_W8A16_M400N80,    // n_tile=80
    PL_AT_W8A16_M448N64,    // n_tile=64
    PL_AT_W8A16_M512N48,    // n_tile=48
    PL_AT_W8A16_M592N32,    // n_tile=32
    PL_AT_FP16_M320N128,    // n_tile=128
    PL_AT_FP16_M352N112,    // n_tile=112
    PL_AT_FP16_M384N96,     // n_tile=96
    PL_AT_FP16_M432N80,     // n_tile=80
    PL_AT_FP16_M480N64,     // n_tile=64
    PL_AT_FP16_M528N48,     // n_tile=48
    PL_AT_FP16_M608N32,     // n_tile=32
    // --- unconditional generated auto-tile group-wise int4 pgrp variants -----
    // 7 per-shape tiled parallel_linear_wint4a16_pgrp_
    // m{m}n{n}k128 (AWQ-style group-wise). Same load-bearing order (n_tile 128->32);
    // m_tile = rpu_pl_tiling::mtile_int4(n). rpu_linear.cpp's autotile_pgrp_kernel_id()
    // indexes by arithmetic offset from PL_AT_INT4_PGRP_M304N128. Names in
    // KERNEL_ID_NAMES below MUST match rpu_pl_tiling::autotile_kernel_name_int4() exactly.
    PL_AT_INT4_PGRP_M304N128,   // n_tile=128
    PL_AT_INT4_PGRP_M336N112,   // n_tile=112
    PL_AT_INT4_PGRP_M368N96,    // n_tile=96
    PL_AT_INT4_PGRP_M416N80,    // n_tile=80
    PL_AT_INT4_PGRP_M464N64,    // n_tile=64
    PL_AT_INT4_PGRP_M512N48,    // n_tile=48
    PL_AT_INT4_PGRP_M592N32,    // n_tile=32
    // --- operator-asset v1.0.0 auto-tile ACC32 variants ------------------
    // Keep each 7-entry family contiguous in n_tile order 128 -> 32. The
    // launchers select a member by arithmetic offset from the first ID.
    PL_AT_W8A16_ACC32_M192N128,
    PL_AT_W8A16_ACC32_M208N112,
    PL_AT_W8A16_ACC32_M240N96,
    PL_AT_W8A16_ACC32_M288N80,
    PL_AT_W8A16_ACC32_M336N64,
    PL_AT_W8A16_ACC32_M400N48,
    PL_AT_W8A16_ACC32_M496N32,
    PL_AT_FP16_ACC32_M208N128,
    PL_AT_FP16_ACC32_M240N112,
    PL_AT_FP16_ACC32_M272N96,
    PL_AT_FP16_ACC32_M304N80,
    PL_AT_FP16_ACC32_M352N64,
    PL_AT_FP16_ACC32_M416N48,
    PL_AT_FP16_ACC32_M496N32,
    PL_AT_INT4_PGRP_ACC32_M208N128,
    PL_AT_INT4_PGRP_ACC32_M224N112,
    PL_AT_INT4_PGRP_ACC32_M256N96,
    PL_AT_INT4_PGRP_ACC32_M304N80,
    PL_AT_INT4_PGRP_ACC32_M352N64,
    PL_AT_INT4_PGRP_ACC32_M400N48,
    PL_AT_INT4_PGRP_ACC32_M496N32,
    // Explicit high-accuracy unary formulas.
    UNARY_GELU_TANH_SPM,
    UNARY_GELU_ERF_SPM,
    UNARY_SOFTPLUS_SPM,
    // Fused ring all-reduce + residual. The launcher selects no-pace only for
    // the generator-certified sub-11-KiB/core envelope.
    LLM_ALL_REDUCE_RESIDUAL_RING_PACED,
    LLM_ALL_REDUCE_RESIDUAL_RING_NOPACE,
    // Raw-SPM V transpose + by-MHA attention; launchers validate the runtime
    // shape.
    V_TRANSPOSE_SPM_VCTXLEN,
    SDPA_BY_MHA_SPM_VCTXLEN,
    // Optional vector-row RMSNorm variants. Callers enforce exact row
    // divisibility and fall back to the base kernel when unavailable.
    RMS_NORM_SPM_V16,
    RMS_NORM_SPM_V32,
    _COUNT
};

// Kernel 名称映射表 (与 KernelId 枚举顺序对应)
static constexpr const char* KERNEL_ID_NAMES[] = {
    // SPM binary kernels (no _ddr suffix)
    "binary_sameshape",
    "binary_Nx1_NxC256_batch",
    "binary_Nx1_NxC256_batch_bopa",
    "binary_Nx1_NxC_v16_batch",
    "binary_Nx1_NxC_v16_batch_bopa",
    "binary_1xC_NxC_v256_batch",
    "binary_1xC_NxC_v256_batch_bopa",
    "binary_1xC_NxC_v16_batch",
    "binary_1xC_NxC_v16_batch_bopa",
    // DDR binary kernels (_ddr suffix)
    "binary_sameshape_ddr",
    "binary_scalar_ddr",
    "binary_scalar",
    "binary_scalar_power_ddr",
    "binary_Nx1_NxC256_batch_ddr",
    "binary_Nx1_NxC256_batch_bopa_ddr",
    "binary_Nx1_NxC_v16_batch_ddr",
    "binary_Nx1_NxC_v16_batch_bopa_ddr",
    "binary_1xC_NxC_v256_batch_ddr",
    "binary_1xC_NxC_v256_batch_bopa_ddr",
    "binary_1xC_NxC_v16_batch_ddr",
    "binary_1xC_NxC_v16_batch_bopa_ddr",
    // Unary kernels
    "unary_ddr",
    "softmax_c16_gauto",
    "layer_norm",
    "layer_norm_simple",
    "layer_norm_bf16",
    "llm_fp16_32b_prefill_flash_attn_univ_dp",  // SPM version
    "llm_fp16_16b_prefill_flash_attn_univ_dp",
    "reduce_mean_last_dim_ddr",
    "llama_rms_norm_bf16_ddr",
    "llama_rms_norm",  // SPM variant
    "llama_rms_norm_newton",  // Newton-refined rsqrt (RPU_RMSNORM_NEWTON)
    // RoPE kernel
    "llama_rope",
    "llama_rope_ddr",
    // M-RoPE kernel for Qwen3-VL text execution
    "llama_mrope_interleave",
    // Partial M-RoPE kernel (wall-oss prefill/denoise perf)
    "partial_mrope",
    // 2D RoPE kernel for the Qwen3-VL vision encoder
    "rope_2d_ddr",
    "rope_2d_spm",
    // Fast bilinear pos-embed interpolation (Qwen3-VL / Qwen3.5 vision STEP 0)
    "fast_pos_embed_interpolate",
    // LLaMA KV-Cache kernels
    "llama_insert_vcache_multiwarp",
    "llama_insert_kcache_multiwarp",
    "llama_insert_vcache_multiwarp_v16",
    "llama_insert_kcache_multiwarp_v16",
    // Standalone all-gather kernels
    "all_gather_multi_core",
    "all_gather_multi_core_little_chunk",
    // SPM unary kernels
    "unary",
    // DDR to SPM memcpy (multi-core)
    // SPM to DDR memcpy (multi-core)
    // DDR to SPM memcpy v2 (nested loop fix)
    // SPM to DDR memcpy v2 (nested loop fix)
    // Argmax/Argmin kernels
    "argmax_reduceC_tileN",
    "argmax_reduceC_tileN_C128",
    "argmax_reduceC_tileN_sli",
    "argmax_reduceC_tileN_C128_sli",
    "argmax_reduceN_tileC",
    "argmax_reduceN_tileC_sli",
    "argmin_reduceC_tileN",
    "argmin_reduceC_tileN_C128",
    "argmin_reduceC_tileN_sli",
    "argmin_reduceC_tileN_C128_sli",
    "argmin_reduceN_tileC",
    "argmin_reduceN_tileC_sli",
    // SPM memset (zero-fill) multi-core
    "memset_spm_multi_core_v2",
    // SDPA vctxlen (dynamic context length)
    "llm_fp16_32b_prefill_flash_attn_univ_vctxlen",
    // SDPA vctxlen minibatch (per-image K/V slicing)
    "llm_fp16_32b_prefill_flash_attn_univ_vctxlen_minibatch",
    "llm_fp16_16b_prefill_flash_attn_univ_vctxlen_minibatch",
    // Conv2d SPM kernels
    "conv_fp16_f1_spm_16b_w128x128_k48_1core_buf1_lpaddr_univ",
    "conv_fp16_f1_spm_16b_w128x128_k64_1core_buf1_lpaddr_univ",
    "conv_fp16_f1_spm_16b_w128x128_k96_1core_buf1_lpaddr_univ",
    "conv_fp16_f1_spm_16b_w128x128_k128_1core_buf1_lpaddr_univ",
    "conv_fp16_f3_spm_16b_w128x128_k48_1core_buf1_lpaddr_univ",
    "conv_fp16_f3_spm_16b_w128x128_k64_1core_buf1_lpaddr_univ",
    "conv_fp16_f3_spm_16b_w128x128_k96_1core_buf1_lpaddr_univ",
    "conv_fp16_f3_spm_16b_w128x128_k128_1core_buf1_lpaddr_univ",
    "conv_fp16_fn_spm_16b_w128x128_k48_1core_buf1_lpaddr_univ",
    "conv_fp16_fn_spm_16b_w128x128_k64_1core_buf1_lpaddr_univ",
    "conv_fp16_fn_spm_16b_w128x128_k96_1core_buf1_lpaddr_univ",
    "conv_fp16_fn_spm_16b_w128x128_k128_1core_buf1_lpaddr_univ",
    // Conv2d multi-core column kernels (tile_k=32, tile_n=32/64/128)
    "conv_fp16_f1_spm_column_w128x32_k32_coren_buf1_sIsWsO_univ",
    "conv_fp16_f3_spm_column_w128x32_k32_coren_buf1_sIsWsO_univ",
    "conv_fp16_fn_spm_column_w128x32_k32_coren_buf1_sIsWsO_univ",
    "conv_fp16_f1_spm_column_w128x64_k32_coren_buf1_sIsWsO_univ",
    "conv_fp16_f3_spm_column_w128x64_k32_coren_buf1_sIsWsO_univ",
    "conv_fp16_fn_spm_column_w128x64_k32_coren_buf1_sIsWsO_univ",
    "conv_fp16_f1_spm_column_w128x128_k32_coren_buf1_sIsWsO_univ",
    "conv_fp16_f3_spm_column_w128x128_k32_coren_buf1_sIsWsO_univ",
    "conv_fp16_fn_spm_column_w128x128_k32_coren_buf1_sIsWsO_univ",
    // Conv2d multi-core column kernels (tile_k=64, tile_n=32/64/128)
    "conv_fp16_f1_spm_column_w128x32_k64_coren_buf1_sIsWsO_univ",
    "conv_fp16_f3_spm_column_w128x32_k64_coren_buf1_sIsWsO_univ",
    "conv_fp16_fn_spm_column_w128x32_k64_coren_buf1_sIsWsO_univ",
    "conv_fp16_f1_spm_column_w128x64_k64_coren_buf1_sIsWsO_univ",
    "conv_fp16_f3_spm_column_w128x64_k64_coren_buf1_sIsWsO_univ",
    "conv_fp16_fn_spm_column_w128x64_k64_coren_buf1_sIsWsO_univ",
    "conv_fp16_f1_spm_column_w128x128_k64_coren_buf1_sIsWsO_univ",
    "conv_fp16_f3_spm_column_w128x128_k64_coren_buf1_sIsWsO_univ",
    "conv_fp16_fn_spm_column_w128x128_k64_coren_buf1_sIsWsO_univ",
    // Im2col kernels
    "im2col_coren_m64xn32_buf1_sIsO",
    "im2col_coren_m128xn32_buf1_sIsO",
    "im2col_coren_m64xn64_buf1_sIsO",
    "im2col_coren_m128xn64_buf1_sIsO",
    // Pad kernel
    "pad_const_NxC_NxCP",
    // Transpose kernel
    "transpose_ncb_c16",
    "transpose_nbc_c16",
    "parallel_linear_wNVFP4a16_acc16",
    "parallel_linear_wNVFP4a16",
    // Qwen3.5 GDN names; order must match KernelId above.
    "partial_rope_1d",
    "l2norm",
    "llm_fla_conv1d_w16a16_acc16",
    "llama_silu_mul",
    "reduce_sum_non_last_dim_out_seg",
    "cumsum_reduceC_tileN",
    "unit_tril_inv",
    // auto-tile acc16 variants — MUST match the KernelId order above and
    // rpu_pl_tiling::autotile_kernel_name(is_fp16, mtile_w8a16/_w16(n), n).
    "parallel_linear_w8a16_m288n128k128",
    "parallel_linear_w8a16_m320n112k128",
    "parallel_linear_w8a16_m352n96k128",
    "parallel_linear_w8a16_m400n80k128",
    "parallel_linear_w8a16_m448n64k128",
    "parallel_linear_w8a16_m512n48k128",
    "parallel_linear_w8a16_m592n32k128",
    "parallel_linear_m320n128k128",
    "parallel_linear_m352n112k128",
    "parallel_linear_m384n96k128",
    "parallel_linear_m432n80k128",
    "parallel_linear_m480n64k128",
    "parallel_linear_m528n48k128",
    "parallel_linear_m608n32k128",
    // auto-tile group-wise int4 pgrp variants — MUST match the KernelId order above
    // and rpu_pl_tiling::autotile_kernel_name_int4(mtile_int4(n), n).
    "parallel_linear_wint4a16_pgrp_m304n128k128",
    "parallel_linear_wint4a16_pgrp_m336n112k128",
    "parallel_linear_wint4a16_pgrp_m368n96k128",
    "parallel_linear_wint4a16_pgrp_m416n80k128",
    "parallel_linear_wint4a16_pgrp_m464n64k128",
    "parallel_linear_wint4a16_pgrp_m512n48k128",
    "parallel_linear_wint4a16_pgrp_m592n32k128",
    // auto-tile W8A16 ACC32 variants
    "parallel_linear_w8a16_acc32_m192n128k128",
    "parallel_linear_w8a16_acc32_m208n112k128",
    "parallel_linear_w8a16_acc32_m240n96k128",
    "parallel_linear_w8a16_acc32_m288n80k128",
    "parallel_linear_w8a16_acc32_m336n64k128",
    "parallel_linear_w8a16_acc32_m400n48k128",
    "parallel_linear_w8a16_acc32_m496n32k128",
    // auto-tile FP16 ACC32 variants
    "parallel_linear_acc32_m208n128k128",
    "parallel_linear_acc32_m240n112k128",
    "parallel_linear_acc32_m272n96k128",
    "parallel_linear_acc32_m304n80k128",
    "parallel_linear_acc32_m352n64k128",
    "parallel_linear_acc32_m416n48k128",
    "parallel_linear_acc32_m496n32k128",
    // auto-tile W4A16 pgrp ACC32 variants
    "parallel_linear_wint4a16_pgrp_acc32_m208n128k128",
    "parallel_linear_wint4a16_pgrp_acc32_m224n112k128",
    "parallel_linear_wint4a16_pgrp_acc32_m256n96k128",
    "parallel_linear_wint4a16_pgrp_acc32_m304n80k128",
    "parallel_linear_wint4a16_pgrp_acc32_m352n64k128",
    "parallel_linear_wint4a16_pgrp_acc32_m400n48k128",
    "parallel_linear_wint4a16_pgrp_acc32_m496n32k128",
    "unary_gelu_tanh",
    "unary_gelu_erf",
    "unary_softplus",
    "llm_all_reduce_residual",
    "llm_all_reduce_residual_nopace",
    "llm_fp16_32b_prefill_trans_prjv_dp_vctxlen",
    "llm_fp16_16b_prefill_flash_attn_by_mha_univ_vctxlen",
    // Optional RMSNorm variants.
    "llama_rms_norm_v16",
    "llama_rms_norm_v32",
};
static_assert(sizeof(KERNEL_ID_NAMES) / sizeof(KERNEL_ID_NAMES[0]) == static_cast<size_t>(KernelId::_COUNT),
              "KERNEL_ID_NAMES size must match KernelId::_COUNT");

struct CachedKernel {
    std::unique_ptr<::rhino_lkn::Program_t> program;
    std::unique_ptr<::rhino_lkn::Kernel_t> kernel;
};

class KernelCache {
public:
    static KernelCache& instance() {
        static KernelCache inst;
        return inst;
    }

    // 初始化：加载所有需要的 Program 和 Kernel
    void initialize();

    // O(1) 快速获取 - 通过枚举 ID 直接数组索引 (推荐热点路径使用)
    inline ::rhino_lkn::Kernel_t* get(KernelId id) const {
        return fast_cache_[static_cast<uint8_t>(id)];
    }

    // 获取已缓存的 Kernel (通过名称查找，较慢)
    ::rhino_lkn::Kernel_t* get_kernel(const std::string& kernel_name);

    // 快速获取 kernel - 用于固定名称的热点路径 (兼容旧代码)
    // 使用 static 局部缓存避免重复哈希查找
    inline ::rhino_lkn::Kernel_t* get_kernel_fast(const char* kernel_name,
                                                   ::rhino_lkn::Kernel_t*& cached_ptr) {
        if (__builtin_expect(cached_ptr != nullptr, 1)) {
            return cached_ptr;
        }
        cached_ptr = get_kernel(kernel_name);
        return cached_ptr;
    }

    // 获取已缓存的 Program (兼容旧代码)
    ::rhino_lkn::Program_t* get_program(const std::string& kernel_name);

    // 通过 KernelId 获取 Program (用于 batch 模式创建独立 kernel 实例)
    inline ::rhino_lkn::Program_t* get_program(KernelId id) {
        return get_program(KERNEL_ID_NAMES[static_cast<uint8_t>(id)]);
    }

    // 检查是否已初始化
    bool is_initialized() const { return initialized_; }

    // 清空所有缓存的 kernel/program (用于 shutdown)
    void clear() {
        kernels_.clear();  // unique_ptr 会自动释放 Program_t/Kernel_t
        for (size_t i = 0; i < static_cast<size_t>(KernelId::_COUNT); ++i) {
            fast_cache_[i] = nullptr;
        }
        initialized_ = false;
    }

private:
    KernelCache() : initialized_(false) {
        // 初始化快速缓存数组为 nullptr
        for (size_t i = 0; i < static_cast<size_t>(KernelId::_COUNT); ++i) {
            fast_cache_[i] = nullptr;
        }
    }
    ~KernelCache() { clear(); }
    KernelCache(const KernelCache&) = delete;
    KernelCache& operator=(const KernelCache&) = delete;

    // 填充快速缓存数组
    void populate_fast_cache();
    CachedKernel load_kernel_from_oplib(
        const std::string& path, const std::string& kernel_name);

    bool initialized_;
    std::unordered_map<std::string, CachedKernel> kernels_;
    // O(1) 访问的快速缓存数组 - 在 initialize() 时填充
    ::rhino_lkn::Kernel_t* fast_cache_[static_cast<size_t>(KernelId::_COUNT)];
};

// 宏: 通过 KernelId 获取 kernel (最快，推荐使用)
#define GET_KERNEL(id) (KernelCache::instance().get(id))

// 宏: 用于固定 kernel 名称的快速获取 (兼容旧代码)
#define GET_KERNEL_CACHED(name) \
    ([&]() -> ::rhino_lkn::Kernel_t* { \
        static ::rhino_lkn::Kernel_t* _cached_##__LINE__ = nullptr; \
        return KernelCache::instance().get_kernel_fast(name, _cached_##__LINE__); \
    }())

// =============================================================================
// Queue Cache Manager - 缓存 Queue_t 以减少每次调用的构造/析构开销
// RPU 最多 8 个 core: 1 core = core0, 4 cores = 0,1,2,3
// =============================================================================

class QueueCache {
public:
    static QueueCache& instance() {
        static QueueCache inst;
        return inst;
    }

    // O(1) 获取指定核数的 Queue
    //
    // fresh (可选 out-param):传 nullptr 时忽略;非 nullptr 时,*fresh = true
    // 表示本次调用刚新建了 Queue_t (打了一次 BufferPool::AcquireBuffer),
    // false 表示命中 cache 复用。graph runtime 用它做 "实际 Queue_t allocated"
    // 计数，用于限制 BufferPool 压力。
    inline ::rhino_lkn::Queue_t* get(uint8_t core_num, bool* fresh = nullptr) {
        uint8_t idx = core_num - 1;
        if (__builtin_expect(queues_[idx] != nullptr, 1)) {
            if (fresh) *fresh = false;
            return queues_[idx];
        }
        queues_[idx] = new ::rhino_lkn::Queue_t(core_num);
        if (fresh) *fresh = true;
        return queues_[idx];
    }

    // 清空所有缓存的 Queue (用于 shutdown)
    void clear() {
        for (int i = 0; i < 8; ++i) {
            delete queues_[i];
            queues_[i] = nullptr;
        }
    }

private:
    QueueCache() {
        for (int i = 0; i < 8; ++i) {
            queues_[i] = nullptr;
        }
    }
    ~QueueCache() { clear(); }
    QueueCache(const QueueCache&) = delete;
    QueueCache& operator=(const QueueCache&) = delete;

    ::rhino_lkn::Queue_t* queues_[8];  // 索引 0-7 对应 1-8 核
};

// 宏: 获取指定核数的 Queue (O(1) 数组索引)
#define GET_QUEUE(core_num) (QueueCache::instance().get(core_num))

// =============================================================================
// Checked batch-DMA submission — the one chokepoint every add_dma_kernel*
// caller routes through (src/ops/rpu_memcpy.cpp immediate/PASSTHROUGH wrappers
// + src/graph/graph_runtime_execute.cpp prepare_segment_queue).
// =============================================================================
//
// Reject invalid endpoints and rejected launch submissions before execution.
// This keeps host-side graph construction fail-closed instead of running a
// partial transfer plan.
inline void rpu_add_dma_checked(::rhino_lkn::Queue_t& q,
                                const RpuDmaEndpoint& src,
                                const RpuDmaEndpoint& dst,
                                size_t bytes, int channel,
                                const char* where) {
    TORCH_CHECK(src && dst,
                "rpu_add_dma_checked(", where,
                "): missing managed DMA endpoint");
    const uint32_t rc = q.add_dma_kernel(
        *src.owner, src.offset, *dst.owner, dst.offset, bytes, channel);
    TORCH_CHECK(rc == 0,
                "rpu_add_dma_checked(", where, "): add_dma_kernel returned ", rc,
                " (src_offset=", src.offset, " dst_offset=", dst.offset,
                " bytes=", bytes, " ch=", channel,
                "). The DMA was NOT appended; build_batch() "
                "would still succeed and the transfer would silently vanish.");
}

inline void rpu_add_dma_checked(::rhino_lkn::Queue_t& q,
                                uint64_t src_addr, uint64_t dst_addr,
                                size_t bytes, int channel,
                                const char* where) {
    const auto src = rpu_resolve_device_dma_endpoint(src_addr, bytes, where);
    const auto dst = rpu_resolve_device_dma_endpoint(dst_addr, bytes, where);
    rpu_add_dma_checked(q, src, dst, bytes, channel, where);
}

inline void rpu_add_dma_mutable_checked(::rhino_lkn::Queue_t& q,
                                        const RpuDmaEndpoint& src,
                                        const RpuDmaEndpoint& dst,
                                        size_t bytes, int channel,
                                        const char* where) {
    TORCH_CHECK(src && dst,
                "rpu_add_dma_mutable_checked(", where,
                "): missing managed DMA endpoint");
    const uint32_t rc = q.add_dma_kernel_mutable(
        *src.owner, src.offset, *dst.owner, dst.offset, bytes, channel);
    TORCH_CHECK(rc == 0,
                "rpu_add_dma_mutable_checked(", where,
                "): add_dma_kernel_mutable returned ", rc,
                " (src_offset=", src.offset, " dst_offset=", dst.offset,
                " bytes=", bytes, " ch=", channel,
                "). The DMA was NOT appended, and the mutable "
                "dma_id sequence the caller records would be off by one.");
}

inline void rpu_add_dma_mutable_checked(::rhino_lkn::Queue_t& q,
                                        uint64_t src_addr, uint64_t dst_addr,
                                        size_t bytes, int channel,
                                        const char* where) {
    const auto src = rpu_resolve_device_dma_endpoint(src_addr, bytes, where);
    const auto dst = rpu_resolve_device_dma_endpoint(dst_addr, bytes, where);
    rpu_add_dma_mutable_checked(q, src, dst, bytes, channel, where);
}

// Preserve the launch API return code: a rejected kernel must not leave a
// seemingly valid batch that is missing one compute step.
inline void rpu_add_kernel_mutable_checked(
        ::rhino_lkn::Queue_t& q, ::rhino_lkn::Kernel_t& kernel,
        const std::vector<uint16_t>& grid_dims,
        const std::vector<uint8_t>& core_ids,
        const char* where) {
    const uint32_t rc = q.add_kernel_mutable(kernel, grid_dims, core_ids);
    TORCH_CHECK(rc == 0,
                "rpu_add_kernel_mutable_checked(", where,
                "): add_kernel_mutable returned ", rc,
                ". The kernel was NOT appended; build_batch() would still "
                "succeed and the compute step would silently vanish.");
}
