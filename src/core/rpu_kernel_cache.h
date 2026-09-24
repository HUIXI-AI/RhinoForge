// rpu_kernel_cache.h — KernelId enum, kernel name table, KernelCache, QueueCache
#pragma once

#include <atomic>
#include <bitset>
#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <c10/util/Exception.h>   // TORCH_CHECK — rpu_add_dma_checked
#include "rhino_launch_program.h"
#include "rhino_launch_kernel.h"
#include "rhino_launch_queue.h"
#include "rpu_dma_endpoint.h"
#include "rpu_oplib_snapshot.h"

// =============================================================================
// Kernel ID 枚举 - 用于 O(1) 数组索引访问 (比 unordered_map 快 10x+)
// =============================================================================
// Host-side lookup ID; no serialized/device ABI stores its underlying bytes.
enum class KernelId : uint16_t {
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
    // RMSNorm kernels
    // Note: llama_rms_norm_fp16 was dropped from rhinoOpLib v0.1.1_38977cf7 —
    // the enum entry was unused (no GET_KERNEL caller) and is removed here.
    RMS_NORM_BF16,
    RMS_NORM_BF16_SPM,  // SPM版本: input/output在SPM
    // Newton-refined rsqrt variant, used only by explicit precision routes
    // and isolated conformance probes. Ordinary routes retain their arithmetic.
    RMS_NORM_NEWTON_SPM,
    // RoPE kernel
    ROPE,
    ROPE_DDR,
    // M-RoPE kernel (Qwen3-VL R-Phase 1):
    //   position_ids live in DDR (typical prefill/decode path); cos/sin tables
    //   share the 1D RoPE [max_seq, head_dim/2] format. Per-axis (T/H/W)
    //   selection is performed at runtime via strobe masks set by the launcher.
    MROPE,
    // Partial M-RoPE kernel (wall-oss prefill/denoise perf): host-precomputed
    //   per-token interleaved cos/sin [seq, rotary_dim/2] streamed in DDR — no
    //   in-kernel position_ids gather / strobe masks. Bit-exact to MROPE @ factor
    //   1.0. See src/ops/rpu_mrope.cpp::rpu_launch_partial_mrope_spm_kernel.
    PARTIAL_MROPE,
    // 2D RoPE kernel (Qwen3-VL R-Phase 3 vision encoder):
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
    ALL_GATHER_MULTI_CORE_LITTLE_CHUNK,
    // SPM unary kernels (for fused decoder layer MLP)
    UNARY_SPM,
    // DDR to SPM memcpy (multi-core)
    // SPM to DDR memcpy (multi-core)
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
    // Tiled NVFP4 ACC32, ordered by n_tile 128..32 for dispatch indexing.
    PL_AT_NVFP4_ACC32_M176N128,
    PL_AT_NVFP4_ACC32_M208N112,
    PL_AT_NVFP4_ACC32_M240N96,
    PL_AT_NVFP4_ACC32_M272N80,
    PL_AT_NVFP4_ACC32_M320N64,
    PL_AT_NVFP4_ACC32_M384N48,
    PL_AT_NVFP4_ACC32_M480N32,
    // Qwen3.5 GDN kernels. The auto-tile block below must remain contiguous
    // because rpu_linear.cpp indexes it by arithmetic offset. This enum order
    // must mirror KERNEL_ID_NAMES.
    // Qwen3.5 partial RoPE 1D — DECODE. Same partial rotate-half, but position-lookup
    // cos/sin (grid: NUM_ELT_PER_WRP=512, grid_y=num_tokens).
    PARTIAL_ROPE_1D,
    // Qwen3.5 GDN decode 核心，由匹配的运行时资产提供。
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
    // --- auto-tile ACC32 variants --------------------------------------
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
    UNARY_GELU_ERF_ULTRA_SPM,
    UNARY_SOFTPLUS_SPM,
    // Fused ring all-reduce + residual. The launcher selects no-pace only for
    // the sub-11-KiB/core envelope.
    LLM_ALL_REDUCE_RESIDUAL_RING_PACED,
    LLM_ALL_REDUCE_RESIDUAL_RING_NOPACE,
    // Raw-SPM V transpose + by-MHA attention. Append-only to keep existing
    // KernelId values stable; launchers validate the runtime shape.
    V_TRANSPOSE_SPM_VCTXLEN,
    SDPA_BY_MHA_SPM_VCTXLEN,
    // Compact-residual ring used by the public Vision owner.
    LLM_ALL_REDUCE_RESIDUAL_RING_LOCAL_SPM,
    // `llama_rms_norm` multi-row payload family. Canonical selectors use the
    // validated fixed grid.x=8 contract with reg[3]=M/8; append-only legacy
    // selectors retain the former grid.x=M/V, reg[3]=V ABI. Both payloads are
    // optional, so handles snapshot their actual availability at install time.
    RMS_NORM_SPM_V16,
    RMS_NORM_SPM_V32,
    // 同族的 Newton 多行版只被 conformance 探针取用。
    // Base Newton reuses the conformance-only RMS_NORM_NEWTON_SPM above.
    RMS_NORM_SPM_NEWTON_V16,
    RMS_NORM_SPM_NEWTON_V32,
    // Optional per-core-SPM cos/sin table variant of llama_mrope_interleave.
    // Its address encoding is distinct from the DDR-table ABI; callers must use
    // that variant's register contract. Missing optional payloads are skipped.
    MROPE_SPM_TBL,
    // Shared W8 Gate/Up/SwiGLU payload used by the public VL prefill owner.
    FUSED_GATE_UP_SWIGLU_W8A16_M320N112,
    FLA_CONV1D_NOINPLACE,
    // Decode-only fused GDN kernels shipped by the main op-lib ref.
    QWEN3_5_RMS_NORM_GATED,
    QWEN3_5_RANK1_FMA,
    QWEN3_5_MUL_REDUCE_ROWS,
    // Shared decode-only M=1 GEMV kernels. Appended to preserve every existing
    // KernelId; Linear keeps this route opt-in.
    LLAMA_GEMV,
    LLAMA_GEMV_WINT8,
    // Controlled Qwen3-VL routes in the standard expansion library.
    QWEN3VL_PARTIAL_MROPE_QK_FUSED,
    QWEN3VL_INSERT_KV_CACHE_V16,
    QWEN3VL_INSERT_KV_CACHE_DECODE,
    QWEN3VL_RMS_NORM_MULTIWARP_V64,
    QWEN3VL_PREFILL_QKV_PLANAR_DIRECT,
    QWEN3VL_SILU_MUL_EXACT_STRIDED,
    // Bounded shared row W8A16 schedule: one N80 weight tile across M blocks.
    // Optional expansion payload; append-only to preserve existing identities.
    W8A16_ROW_WEIGHT_REUSE_M352N80K256,
    PI05_GATE_UP_GEGLU_W8A16_M400N80,
    PI05_ADARMS_NORM_SHIFT_H1024,
    PI05_GATED_RESIDUAL_H1024,
    PI05_RING_XOR3_M768N1152,
    PI05_RING_XOR3_M400N2048,
    PI05_RING_XOR3_M50N1024,
    PI05_K_ROPE_INSERT_M50D256P64,
    PI05_DENOISE_GATEUP_RESIDENT_W8A16_M512N48K128,
    PI05_PREFILL_O_LINEAR_XOR3_RMSNORM_M400N2048K256,
    PI05_DOWN_WEIGHT_OUTER_W8A16_C400X2_M160N80K128,
    PI05_OWNER_NORM_FULL_RESIDUAL_M400N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M400N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M400N2048,
    PI05_PREFILL_QKV_WEIGHT_OUTER_W8A16_C400X2_M128N64K128,
    PI05_PREFILL_O_WEIGHT_OUTER_W8A16_C400X2_M128N80K128,
    PI05_VISION_FC_WEIGHT_OUTER_W8A16_M768N80K128,
    PI05_DENOISE_Q_PAIR16_ROPE_M50N256K1024,
    // Exact Pi0.5 denoise continuation: XOR3 sum, FP16 gate, residual add.
    PI05_XOR3_GATED_RESIDUAL_M50N1024,
    // Owner-local alias of the generated M592/N32/K128 W8A16 program.  The
    // distinct ID makes KV1 typed-census counts independent of other linears.
    PI05_KV1_STRIPE_W8A16_M592N32,
    // Optional expansion payload: consumes pair-mapped compact K/V stripes
    // and writes the exact P800/M50/PAD64 physical-KV8 cache suffix.
    PI05_DENOISE_KV1_DIRECT_CACHE_M50D256P64,
    // Optional expansion payload: reads the installed full replicated K/V
    // W8A16 owners and emits both pair-mapped compact projections at once.
    PI05_DENOISE_KV1_PAIR_OWNER_W8A16_M50N32X2K1024,
    // Optional exact-prefill expansion payload: reads the installed K2048
    // physical-KV8 owners and emits compact logical-KV1 K/V together.
    PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M400N32X2K2048,
    // Optional exact-prefill expansion payload: applies K RoPE to the compact
    // pair and writes one aligned C400 segment directly to physical-KV8 cache.
    PI05_PREFILL_KV1_DIRECT_CACHE_M400D256P400,
    // Optional exact-prefill expansion payload: reuses each Gate/Up weight
    // tile across both C400 halves and retires the production FP16 GeGLU.
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C400X2_M160N80K128,
    // Exact M50/H1024/I4096: original N48 ACC16 Gate/Up plus FP16 GeGLU.
    PI05_DENOISE_GATE_UP_GEGLU_W8A16_M512N48K128,
    PI05_DENOISE_GATE_UP_GEGLU_WINT4A16_PGRP_M512N48K128,
    PI05_DENOISE_GATE_UP_GEGLU_WINT4A16_PGRP_M464N64K128,
    PARALLEL_LINEAR_NVFP4_V2_M320N64,
    PARALLEL_LINEAR_NVFP4_V2_M384N48,
    PI05_DENOISE_GATE_UP_GEGLU_NVFP4_M320N64K128,
    PI05_OWNER_NORM_COMPACT_A8_M400N2048,
    PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_M48N32K2048,
    PI05_PREFILL_GATE_UP_GEGLU_ONLINE_W8A8_P544_C272,
    // Optional exact Vision owner96 LayerNorm and compact-residual companion.
    PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152,
    PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152,
    // Append-only Pi0.5 dtype/camera coverage variants. Internal IDs exceed
    // 255; cache and graph name lookups must retain the complete index.
    PI05_DENOISE_KV1_PAIR_OWNER_FP16_M50N32X2K1024,
    PI05_DENOISE_GATE_UP_GEGLU_FP16_M608N32K128,
    PI05_VISION_OWNER_REDUCE_LAYERNORM_M512N1152,
    PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M512N1152,
    PI05_RING_XOR3_M512N1152,
    PI05_RING_XOR3_M272N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M272N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M272N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M272N2048,
    PI05_OWNER_NORM_COMPACT_A8_M272N2048,
    PI05_PREFILL_KV1_DIRECT_CACHE_M272D256P272,
    PI05_PREFILL_O_WEIGHT_OUTER_FP16_C400X2_M128N80K128,
    PI05_PREFILL_O_WEIGHT_OUTER_FP16_C272X2_M128N80K128,
    PI05_PREFILL_O_WEIGHT_OUTER_W8A16_C272X2_M128N80K128,
    PI05_DOWN_WEIGHT_OUTER_FP16_C400X2_M160N80K128,
    PI05_DOWN_WEIGHT_OUTER_FP16_C272X2_M160N80K128,
    PI05_DOWN_WEIGHT_OUTER_W8A16_C272X2_M160N80K128,
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C400X2_M160N80K128,
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C272X2_M160N80K128,
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C272X2_M160N80K128,
    PI05_PREFILL_KV1_PAIR_OWNER_FP16_M400N32X2K2048,
    PI05_PREFILL_KV1_PAIR_OWNER_FP16_M272N32X2K2048,
    PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M272N32X2K2048,
    PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C272_M48N32K2048,
    PI05_RING_XOR3_M416N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M416N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M416N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M416N2048,
    PI05_OWNER_NORM_COMPACT_A8_M416N2048,
    PI05_PREFILL_KV1_DIRECT_CACHE_M416D256P416,
    PI05_PREFILL_O_WEIGHT_OUTER_FP16_C416X2_M128N80K128,
    PI05_PREFILL_O_WEIGHT_OUTER_W8A16_C416X2_M128N80K128,
    PI05_DOWN_WEIGHT_OUTER_FP16_C416X2_M128N80K128,
    PI05_DOWN_WEIGHT_OUTER_W8A16_C416X2_M128N80K128,
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C416X2_M128N80K128,
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C416X2_M128N80K128,
    PI05_PREFILL_KV1_PAIR_OWNER_FP16_M416N32X2K2048,
    PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M416N32X2K2048,
    PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C416_M48N32K2048,
    PI05_RING_XOR3_M288N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M288N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M288N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M288N2048,
    PI05_OWNER_NORM_COMPACT_A8_M288N2048,
    PI05_PREFILL_KV1_DIRECT_CACHE_M288D256P288,
    PI05_PREFILL_O_WEIGHT_OUTER_FP16_C288X2_M128N80K128,
    PI05_PREFILL_O_WEIGHT_OUTER_W8A16_C288X2_M128N80K128,
    PI05_DOWN_WEIGHT_OUTER_FP16_C288X2_M160N80K128,
    PI05_DOWN_WEIGHT_OUTER_W8A16_C288X2_M160N80K128,
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C288X2_M160N80K128,
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C288X2_M160N80K128,
    PI05_PREFILL_KV1_PAIR_OWNER_FP16_M288N32X2K2048,
    PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M288N32X2K2048,
    PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C288_M48N32K2048,
    PI05_RING_XOR3_M448N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M448N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M448N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M448N2048,
    PI05_OWNER_NORM_COMPACT_A8_M448N2048,
    PI05_PREFILL_KV1_DIRECT_CACHE_M448D256P448,
    PI05_PREFILL_O_WEIGHT_OUTER_FP16_C448X2_M96N80K128,
    PI05_PREFILL_O_WEIGHT_OUTER_W8A16_C448X2_M96N80K128,
    PI05_DOWN_WEIGHT_OUTER_FP16_C448X2_M96N80K128,
    PI05_DOWN_WEIGHT_OUTER_W8A16_C448X2_M96N80K128,
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C448X2_M96N80K128,
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C448X2_M96N80K128,
    PI05_PREFILL_KV1_PAIR_OWNER_FP16_M448N32X2K2048,
    PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M448N32X2K2048,
    PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C448_M48N32K2048,
    PI05_RING_XOR3_M320N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M320N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M320N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M320N2048,
    PI05_OWNER_NORM_COMPACT_A8_M320N2048,
    PI05_PREFILL_KV1_DIRECT_CACHE_M320D256P320,
    PI05_PREFILL_O_WEIGHT_OUTER_FP16_C320X2_M128N80K128,
    PI05_PREFILL_O_WEIGHT_OUTER_W8A16_C320X2_M128N80K128,
    PI05_DOWN_WEIGHT_OUTER_FP16_C320X2_M160N80K128,
    PI05_DOWN_WEIGHT_OUTER_W8A16_C320X2_M160N80K128,
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C320X2_M160N80K128,
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C320X2_M160N80K128,
    PI05_PREFILL_KV1_PAIR_OWNER_FP16_M320N32X2K2048,
    PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M320N32X2K2048,
    PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C320_M48N32K2048,
    PI05_RING_XOR3_M432N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M432N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M432N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M432N2048,
    PI05_OWNER_NORM_COMPACT_A8_M432N2048,
    PI05_PREFILL_KV1_DIRECT_CACHE_M432D256P432,
    PI05_PREFILL_O_WEIGHT_OUTER_FP16_C432X2_M112N80K128,
    PI05_PREFILL_O_WEIGHT_OUTER_W8A16_C432X2_M112N80K128,
    PI05_DOWN_WEIGHT_OUTER_FP16_C432X2_M112N80K128,
    PI05_DOWN_WEIGHT_OUTER_W8A16_C432X2_M112N80K128,
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C432X2_M112N80K128,
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C432X2_M112N80K128,
    PI05_PREFILL_KV1_PAIR_OWNER_FP16_M432N32X2K2048,
    PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M432N32X2K2048,
    PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C432_M48N32K2048,
    PI05_RING_XOR3_M304N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M304N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M304N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M304N2048,
    PI05_OWNER_NORM_COMPACT_A8_M304N2048,
    PI05_PREFILL_KV1_DIRECT_CACHE_M304D256P304,
    PI05_PREFILL_O_WEIGHT_OUTER_FP16_C304X2_M128N80K128,
    PI05_PREFILL_O_WEIGHT_OUTER_W8A16_C304X2_M128N80K128,
    PI05_DOWN_WEIGHT_OUTER_FP16_C304X2_M160N80K128,
    PI05_DOWN_WEIGHT_OUTER_W8A16_C304X2_M160N80K128,
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C304X2_M160N80K128,
    PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C304X2_M160N80K128,
    PI05_PREFILL_KV1_PAIR_OWNER_FP16_M304N32X2K2048,
    PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M304N32X2K2048,
    PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C304_M48N32K2048,
    // Optional main-ref exact Qwen3-VL-4B W8 decode Gate/Up/SwiGLU.
    QWEN3VL_GATEUP_SWIGLU_GEMV,
    // Optional main-ref exact Qwen3-VL-4B W8 decode BASE RMS + M-RoPE.
    QWEN3VL_QK_NORM_MROPE_D128,
    // Same QK arithmetic followed by the original single-row V2 K/V writers.
    QWEN3VL_QK_NORM_MROPE_KV_INSERT_D128,
    // Exact W8 M1 O GEMV, canonical residual ring, and BASE post RMSNorm.
    QWEN3VL_O_RING_NORM_M1_H2560_W8,
    // Additional generic precision and sparse-MoE payloads.
    GROUPED_PARALLEL_LINEAR_ACC16,
    GROUPED_PARALLEL_LINEAR_8BIT_ACC16,
    MOE_TOPK_SCALAR_FP16,
    MOE_TOPK_BY_KVSORT_FP16,
    MOE_BINCOUNT_SCALAR,
    MOE_TOPK_PROB_NORM_FP16,
    CAST_UINT16_INT32,
    GATHER_SCALAR_FP16,
    GATHER_FP16,
    MOE_GATHER_MUL_V2_FP16,
    SCATTERND_ADD_INT16_FP16,
    GROUPED_PARALLEL_LINEAR_ACC32,
    GROUPED_PARALLEL_LINEAR_8BIT_ACC32,
    PL_AT_NVFP4_ACC16_M176N128,
    PL_AT_NVFP4_ACC16_M208N112,
    PL_AT_NVFP4_ACC16_M240N96,
    PL_AT_NVFP4_ACC16_M272N80,
    PL_AT_NVFP4_ACC16_M320N64,
    PL_AT_NVFP4_ACC16_M384N48,
    PL_AT_NVFP4_ACC16_M480N32,
    PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M320N64K128,
    UNARY_SILU_HIGH_PRECISION,
    UNARY_TANH_HIGH_PRECISION,
    UNARY_GELU_ERF_HIGH_PRECISION,
    RHINOVLA_NEWTON_NORM_SHIFT_H1024,
    RHINOVLA_SILU_HIGH_MUL,
    UNARY_GELU_TANH_HIGH_PRECISION,
    _COUNT
};
static_assert(static_cast<size_t>(KernelId::_COUNT) <= UINT16_MAX,
              "KernelId and its count must fit uint16_t");

// Atomic publication for optional payloads queried while installing a model
// handle. KernelCache's raw pointer table remains protected by the existing
// runtime execution lifecycle; capability discovery reads only this value, so
// it cannot race clear() while Program/Kernel objects are being destroyed.
class KernelPayloadAvailability final {
public:
    static constexpr size_t kExtendedBegin = static_cast<size_t>(
        KernelId::PI05_DENOISE_KV1_PAIR_OWNER_FP16_M50N32X2K1024);
    using ExtendedPayloads = std::bitset<static_cast<size_t>(KernelId::_COUNT) - kExtendedBegin>;

    void publish(bool rmsnorm_v16, bool rmsnorm_v32,
                 bool w8a16_row_weight_reuse = false,
                 bool vision_owner_ln = false,
                 bool vision_compact_residual = false,
                 bool pi05_a8_c272 = false,
                 const ExtendedPayloads& pi05_extended_payloads = {},
                 bool qwen3vl_gateup_swiglu = false,
                 bool qwen3vl_qk_norm_mrope = false,
                 bool qwen3vl_qk_norm_mrope_kv_insert = false,
                 bool qwen3vl_o_ring_norm = false,
                 bool rmsnorm_newton = false,
                 bool rmsnorm_newton_v16 = false,
                 bool rmsnorm_newton_v32 = false) {
        uint32_t flags = 0;
        if (w8a16_row_weight_reuse) flags |= kW8A16RowWeightReuse;
        if (rmsnorm_v16) flags |= kRmsNormV16;
        if (rmsnorm_v32) flags |= kRmsNormV32;
        if (vision_owner_ln) flags |= kVisionOwnerLn;
        if (vision_compact_residual) flags |= kVisionCompactResidual;
        if (pi05_a8_c272) flags |= kPi05A8C272;
        if (qwen3vl_gateup_swiglu) flags |= kQwen3VlGateUp;
        if (qwen3vl_qk_norm_mrope) flags |= kQwen3VlQkNormMrope;
        if (qwen3vl_qk_norm_mrope_kv_insert) flags |= kQwen3VlQkNormMropeKvInsert;
        if (qwen3vl_o_ring_norm) flags |= kQwen3VlORingNorm;
        if (rmsnorm_newton) flags |= kRmsNormNewton;
        if (rmsnorm_newton_v16) flags |= kRmsNormNewtonV16;
        if (rmsnorm_newton_v32) flags |= kRmsNormNewtonV32;
        // The complete capability set is one immutable publication, even when
        // it spans multiple machine words. Allocate before changing state so
        // allocation failure cannot expose a partial table.
        const auto next = std::make_shared<const Snapshot>(
            Snapshot{flags, pi05_extended_payloads});
        std::atomic_store_explicit(&state_, next, std::memory_order_release);
    }

    void withdraw() noexcept {
        std::atomic_store_explicit(&state_, std::shared_ptr<const Snapshot>{},
                                   std::memory_order_release);
    }

    bool has_loaded(KernelId id) const {
        const auto state = std::atomic_load_explicit(&state_, std::memory_order_acquire);
        TORCH_CHECK(state,
            "KernelCache loaded-kernel query requires initialized runtime");
        switch (id) {
            case KernelId::RMS_NORM_NEWTON_SPM:
                return (state->flags & kRmsNormNewton) != 0;
            case KernelId::RMS_NORM_SPM_NEWTON_V16:
                return (state->flags & kRmsNormNewtonV16) != 0;
            case KernelId::RMS_NORM_SPM_NEWTON_V32:
                return (state->flags & kRmsNormNewtonV32) != 0;
            case KernelId::RMS_NORM_SPM_V16:
                return (state->flags & kRmsNormV16) != 0;
            case KernelId::RMS_NORM_SPM_V32:
                return (state->flags & kRmsNormV32) != 0;
            case KernelId::QWEN3VL_GATEUP_SWIGLU_GEMV:
                return (state->flags & kQwen3VlGateUp) != 0;
            case KernelId::QWEN3VL_QK_NORM_MROPE_D128:
                return (state->flags & kQwen3VlQkNormMrope) != 0;
            case KernelId::QWEN3VL_QK_NORM_MROPE_KV_INSERT_D128:
                return (state->flags & kQwen3VlQkNormMropeKvInsert) != 0;
            case KernelId::QWEN3VL_O_RING_NORM_M1_H2560_W8:
                return (state->flags & kQwen3VlORingNorm) != 0;
            case KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152:
                return (state->flags & kVisionOwnerLn) != 0;
            case KernelId::PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152:
                return (state->flags & kVisionCompactResidual) != 0;
            case KernelId::PI05_PREFILL_GATE_UP_GEGLU_ONLINE_W8A8_P544_C272:
                return (state->flags & kPi05A8C272) != 0;
            case KernelId::W8A16_ROW_WEIGHT_REUSE_M352N80K256:
                return (state->flags & kW8A16RowWeightReuse) != 0;
            default:
                if (id >= KernelId::PI05_DENOISE_KV1_PAIR_OWNER_FP16_M50N32X2K1024 &&
                    id < KernelId::_COUNT) {
                    return state->extended.test(static_cast<size_t>(id) - kExtendedBegin);
                }
                TORCH_CHECK(false,
                    "KernelCache atomic payload availability is not registered "
                    "for KernelId ", static_cast<int>(id));
                return false;
        }
    }

private:
    static constexpr uint32_t kRmsNormV16 = 1u << 1;
    static constexpr uint32_t kRmsNormV32 = 1u << 2;
    static constexpr uint32_t kW8A16RowWeightReuse = 1u << 6;
    static constexpr uint32_t kVisionOwnerLn = 1u << 3;
    static constexpr uint32_t kVisionCompactResidual = 1u << 4;
    static constexpr uint32_t kPi05A8C272 = 1u << 5;
    static constexpr uint32_t kQwen3VlGateUp = 1u << 7;
    static constexpr uint32_t kQwen3VlQkNormMrope = 1u << 8;
    static constexpr uint32_t kQwen3VlQkNormMropeKvInsert = 1u << 9;
    static constexpr uint32_t kQwen3VlORingNorm = 1u << 10;
    static constexpr uint32_t kRmsNormNewton = 1u << 11;
    static constexpr uint32_t kRmsNormNewtonV16 = 1u << 12;
    static constexpr uint32_t kRmsNormNewtonV32 = 1u << 13;
    struct Snapshot {
        uint32_t flags;
        ExtendedPayloads extended;
    };
    // C++17 shared_ptr atomic free functions keep readers on a single snapshot.
    // This contains values only; raw kernel use still requires the runtime's
    // existing execution lifecycle protection. nullptr means withdrawn.
    std::shared_ptr<const Snapshot> state_;
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
    "llama_rms_norm",  // SPM版本 (renamed from llama_rms_norm_bf16 in v0.1.1_38977cf7)
    "llama_rms_norm_newton",  // Explicit Newton-refined rsqrt opt-in
    // RoPE kernel
    "llama_rope",
    "llama_rope_ddr",
    // M-RoPE kernel (Qwen3-VL R-Phase 1)
    "llama_mrope_interleave",
    // Partial M-RoPE kernel (wall-oss prefill/denoise perf)
    "partial_mrope",
    // 2D RoPE kernel (Qwen3-VL R-Phase 3 vision encoder)
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
    "parallel_linear_wnvfp4a16_acc32_m176n128k128",
    "parallel_linear_wnvfp4a16_acc32_m208n112k128",
    "parallel_linear_wnvfp4a16_acc32_m240n96k128",
    "parallel_linear_wnvfp4a16_acc32_m272n80k128",
    "parallel_linear_wnvfp4a16_acc32_m320n64k128",
    "parallel_linear_wnvfp4a16_acc32_m384n48k128",
    "parallel_linear_wnvfp4a16_acc32_m480n32k128",
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
    "unary_gelu_erf_ultra_precision",
    "unary_softplus",
    "llm_all_reduce_residual",
    "llm_all_reduce_residual_nopace",
    "llm_fp16_32b_prefill_trans_prjv_dp_vctxlen",
    "llm_fp16_16b_prefill_flash_attn_by_mha_univ_vctxlen",
    "llm_all_reduce_residual_local_spm",
    // Hy-VLA optional RMSNorm/M-RoPE variants and FP16 ACC32 probes.
    "llama_rms_norm_v16",
    "llama_rms_norm_v32",
    "llama_rms_norm_newton_v16",
    "llama_rms_norm_newton_v32",
    "llama_mrope_interleave_local_spm_tbl",
    "fused_gate_up_swiglu_w8a16_m320n112k128",
    "llm_fla_conv1d_noinplace_w16a16_acc16",
    "qwen3_5_rms_norm_gated",
    "qwen3_5_rank1_fma",
    "qwen3_5_mul_reduce_rows",
    "llama_gemv",
    "llama_gemv_wint8",
    "partial_mrope_qk_fused",
    "llama_insert_kv_cache_multiwarp_v16",
    "llama_insert_kv_cache_multiwarp_decode",
    "llama_rms_norm_multiwarp_n_v64",
    "parallel_linear_w8a16_m448n64k128_qkv_planar",
    "llama_silu_mul_exact_strided",
    "w8a16_row_weight_reuse_m352n80k256",
    "pi05_gate_up_geglu_w8a16_m400n80k128",
    "pi05_adarms_norm_shift_h1024",
    "pi05_gated_residual_h1024",
    "pi05_all_reduce_residual_xor3_m768n1152",
    "pi05_all_reduce_residual_xor3_m400n2048",
    "pi05_all_reduce_residual_xor3_m50n1024",
    "pi05_k_rope_insert_m50d256p64",
    "pi05_denoise_gateup_resident_w8a16_m512n48k128",
    "pi05_prefill_o_linear_xor3_rmsnorm_m400n2048k256",
    "pi05_down_weight_outer_w8a16_c400x2_m160n80k128",
    "pi05_owner_norm_full_residual_m400n2048",
    "pi05_owner_norm_compact_residual_m400n2048",
    "pi05_xor3_compact_residual_raw_full_m400n2048",
    "pi05_prefill_qkv_weight_outer_w8a16_c400x2_m128n64k128",
    "pi05_prefill_o_weight_outer_w8a16_c400x2_m128n80k128",
    "pi05_vision_fc_weight_outer_w8a16_m768n80k128",
    "pi05_denoise_q_pair16_rope_m50n256k1024",
    "pi05_xor3_gated_residual_m50n1024",
    "parallel_linear_w8a16_m592n32k128",
    "pi05_denoise_kv1_direct_cache_m50d256p64",
    "pi05_denoise_kv1_pair_owner_w8a16_m50n32x2k1024",
    "pi05_prefill_kv1_pair_owner_w8a16_m400n32x2k2048",
    "pi05_prefill_kv1_direct_cache_m400d256p400",
    "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c400x2_m160n80k128",
    "pi05_denoise_gate_up_geglu_w8a16_m512n48k128",
    "pi05_denoise_gate_up_geglu_wint4a16_pgrp_m512n48k128",
    "pi05_denoise_gate_up_geglu_wint4a16_pgrp_m464n64k128",
    "parallel_linear_wnvfp4a16_acc32_m320n64k128",
    "parallel_linear_wnvfp4a16_acc32_m384n48k128",
    "pi05_denoise_gate_up_geglu_nvfp4_acc32_m320n64k128",
    "pi05_owner_norm_compact_a8_m400n2048",
    "pi05_prefill_gate_up_geglu_w8a8_split_m48n32k2048",
    "pi05_prefill_gate_up_geglu_online_w8a8_p544_c272_m48n32k2048",
    "pi05_vision_owner_reduce_layernorm_m768n1152",
    "pi05_vision_xor3_compact_residual_raw_full_m768n1152_epoch",
    "pi05_denoise_kv1_pair_owner_fp16_m50n32x2k1024",
    "pi05_denoise_gate_up_geglu_fp16_m608n32k128",
    "pi05_vision_owner_reduce_layernorm_m512n1152",
    "pi05_vision_xor3_compact_residual_raw_full_m512n1152_epoch",
    "pi05_all_reduce_residual_xor3_m512n1152",
    "pi05_all_reduce_residual_xor3_m272n2048",
    "pi05_owner_norm_full_residual_m272n2048",
    "pi05_owner_norm_compact_residual_m272n2048",
    "pi05_xor3_compact_residual_raw_full_m272n2048",
    "pi05_owner_norm_compact_a8_m272n2048",
    "pi05_prefill_kv1_direct_cache_m272d256p272",
    "pi05_prefill_o_weight_outer_fp16_c400x2_m128n80k128",
    "pi05_prefill_o_weight_outer_fp16_c272x2_m128n80k128",
    "pi05_prefill_o_weight_outer_w8a16_c272x2_m128n80k128",
    "pi05_down_weight_outer_fp16_c400x2_m160n80k128",
    "pi05_down_weight_outer_fp16_c272x2_m160n80k128",
    "pi05_down_weight_outer_w8a16_c272x2_m160n80k128",
    "pi05_prefill_gate_up_geglu_weight_outer_fp16_c400x2_m160n80k128",
    "pi05_prefill_gate_up_geglu_weight_outer_fp16_c272x2_m160n80k128",
    "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c272x2_m160n80k128",
    "pi05_prefill_kv1_pair_owner_fp16_m400n32x2k2048",
    "pi05_prefill_kv1_pair_owner_fp16_m272n32x2k2048",
    "pi05_prefill_kv1_pair_owner_w8a16_m272n32x2k2048",
    "pi05_prefill_gate_up_geglu_w8a8_split_c272_m48n32k2048",
    "pi05_all_reduce_residual_xor3_m416n2048",
    "pi05_owner_norm_full_residual_m416n2048",
    "pi05_owner_norm_compact_residual_m416n2048",
    "pi05_xor3_compact_residual_raw_full_m416n2048",
    "pi05_owner_norm_compact_a8_m416n2048",
    "pi05_prefill_kv1_direct_cache_m416d256p416",
    "pi05_prefill_o_weight_outer_fp16_c416x2_m128n80k128",
    "pi05_prefill_o_weight_outer_w8a16_c416x2_m128n80k128",
    "pi05_down_weight_outer_fp16_c416x2_m128n80k128",
    "pi05_down_weight_outer_w8a16_c416x2_m128n80k128",
    "pi05_prefill_gate_up_geglu_weight_outer_fp16_c416x2_m128n80k128",
    "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c416x2_m128n80k128",
    "pi05_prefill_kv1_pair_owner_fp16_m416n32x2k2048",
    "pi05_prefill_kv1_pair_owner_w8a16_m416n32x2k2048",
    "pi05_prefill_gate_up_geglu_w8a8_split_c416_m48n32k2048",
    "pi05_all_reduce_residual_xor3_m288n2048",
    "pi05_owner_norm_full_residual_m288n2048",
    "pi05_owner_norm_compact_residual_m288n2048",
    "pi05_xor3_compact_residual_raw_full_m288n2048",
    "pi05_owner_norm_compact_a8_m288n2048",
    "pi05_prefill_kv1_direct_cache_m288d256p288",
    "pi05_prefill_o_weight_outer_fp16_c288x2_m128n80k128",
    "pi05_prefill_o_weight_outer_w8a16_c288x2_m128n80k128",
    "pi05_down_weight_outer_fp16_c288x2_m160n80k128",
    "pi05_down_weight_outer_w8a16_c288x2_m160n80k128",
    "pi05_prefill_gate_up_geglu_weight_outer_fp16_c288x2_m160n80k128",
    "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c288x2_m160n80k128",
    "pi05_prefill_kv1_pair_owner_fp16_m288n32x2k2048",
    "pi05_prefill_kv1_pair_owner_w8a16_m288n32x2k2048",
    "pi05_prefill_gate_up_geglu_w8a8_split_c288_m48n32k2048",
    "pi05_all_reduce_residual_xor3_m448n2048",
    "pi05_owner_norm_full_residual_m448n2048",
    "pi05_owner_norm_compact_residual_m448n2048",
    "pi05_xor3_compact_residual_raw_full_m448n2048",
    "pi05_owner_norm_compact_a8_m448n2048",
    "pi05_prefill_kv1_direct_cache_m448d256p448",
    "pi05_prefill_o_weight_outer_fp16_c448x2_m96n80k128",
    "pi05_prefill_o_weight_outer_w8a16_c448x2_m96n80k128",
    "pi05_down_weight_outer_fp16_c448x2_m96n80k128",
    "pi05_down_weight_outer_w8a16_c448x2_m96n80k128",
    "pi05_prefill_gate_up_geglu_weight_outer_fp16_c448x2_m96n80k128",
    "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c448x2_m96n80k128",
    "pi05_prefill_kv1_pair_owner_fp16_m448n32x2k2048",
    "pi05_prefill_kv1_pair_owner_w8a16_m448n32x2k2048",
    "pi05_prefill_gate_up_geglu_w8a8_split_c448_m48n32k2048",
    "pi05_all_reduce_residual_xor3_m320n2048",
    "pi05_owner_norm_full_residual_m320n2048",
    "pi05_owner_norm_compact_residual_m320n2048",
    "pi05_xor3_compact_residual_raw_full_m320n2048",
    "pi05_owner_norm_compact_a8_m320n2048",
    "pi05_prefill_kv1_direct_cache_m320d256p320",
    "pi05_prefill_o_weight_outer_fp16_c320x2_m128n80k128",
    "pi05_prefill_o_weight_outer_w8a16_c320x2_m128n80k128",
    "pi05_down_weight_outer_fp16_c320x2_m160n80k128",
    "pi05_down_weight_outer_w8a16_c320x2_m160n80k128",
    "pi05_prefill_gate_up_geglu_weight_outer_fp16_c320x2_m160n80k128",
    "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c320x2_m160n80k128",
    "pi05_prefill_kv1_pair_owner_fp16_m320n32x2k2048",
    "pi05_prefill_kv1_pair_owner_w8a16_m320n32x2k2048",
    "pi05_prefill_gate_up_geglu_w8a8_split_c320_m48n32k2048",
    "pi05_all_reduce_residual_xor3_m432n2048",
    "pi05_owner_norm_full_residual_m432n2048",
    "pi05_owner_norm_compact_residual_m432n2048",
    "pi05_xor3_compact_residual_raw_full_m432n2048",
    "pi05_owner_norm_compact_a8_m432n2048",
    "pi05_prefill_kv1_direct_cache_m432d256p432",
    "pi05_prefill_o_weight_outer_fp16_c432x2_m112n80k128",
    "pi05_prefill_o_weight_outer_w8a16_c432x2_m112n80k128",
    "pi05_down_weight_outer_fp16_c432x2_m112n80k128",
    "pi05_down_weight_outer_w8a16_c432x2_m112n80k128",
    "pi05_prefill_gate_up_geglu_weight_outer_fp16_c432x2_m112n80k128",
    "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c432x2_m112n80k128",
    "pi05_prefill_kv1_pair_owner_fp16_m432n32x2k2048",
    "pi05_prefill_kv1_pair_owner_w8a16_m432n32x2k2048",
    "pi05_prefill_gate_up_geglu_w8a8_split_c432_m48n32k2048",
    "pi05_all_reduce_residual_xor3_m304n2048",
    "pi05_owner_norm_full_residual_m304n2048",
    "pi05_owner_norm_compact_residual_m304n2048",
    "pi05_xor3_compact_residual_raw_full_m304n2048",
    "pi05_owner_norm_compact_a8_m304n2048",
    "pi05_prefill_kv1_direct_cache_m304d256p304",
    "pi05_prefill_o_weight_outer_fp16_c304x2_m128n80k128",
    "pi05_prefill_o_weight_outer_w8a16_c304x2_m128n80k128",
    "pi05_down_weight_outer_fp16_c304x2_m160n80k128",
    "pi05_down_weight_outer_w8a16_c304x2_m160n80k128",
    "pi05_prefill_gate_up_geglu_weight_outer_fp16_c304x2_m160n80k128",
    "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c304x2_m160n80k128",
    "pi05_prefill_kv1_pair_owner_fp16_m304n32x2k2048",
    "pi05_prefill_kv1_pair_owner_w8a16_m304n32x2k2048",
    "pi05_prefill_gate_up_geglu_w8a8_split_c304_m48n32k2048",
    "qwen3vl_gateup_swiglu_gemv",
    "qwen3vl_qk_norm_mrope_d128",
    "qwen3vl_qk_norm_mrope_kv_insert_d128",
    "qwen3vl_o_ring_norm_m1_h2560_w8",
    "grouped_parallel_linear_acc16",
    "grouped_parallel_linear_w8a16_acc16",
    "topk_scalar_fp16",
    "topk_by_kvsort_fp16",
    "bincount_scalar",
    "llm_topk_prob_norm",
    "unary_cast_uint16_int32",
    "gather_scalar",
    "gather",
    "llm_gather_mul_v2",
    "scatternd_int16_reduction",
    "grouped_parallel_linear_acc32",
    "grouped_parallel_linear_w8a16_acc32",
    "parallel_linear_wnvfp4a16_acc16_m176n128k128",
    "parallel_linear_wnvfp4a16_acc16_m208n112k128",
    "parallel_linear_wnvfp4a16_acc16_m240n96k128",
    "parallel_linear_wnvfp4a16_acc16_m272n80k128",
    "parallel_linear_wnvfp4a16_acc16_m320n64k128",
    "parallel_linear_wnvfp4a16_acc16_m384n48k128",
    "parallel_linear_wnvfp4a16_acc16_m480n32k128",
    "pi05_denoise_gate_up_geglu_nvfp4_acc16_m320n64k128",
    "unary_silu_high_precision",
    "unary_tanh_high_precision",
    "unary_gelu_erf_high_precision",
    "rhinovla_newton_norm_shift_h1024",
    "rhinovla_silu_high_mul",
    "unary_gelu_tanh_high_precision",
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

    // Cold profile admission against the frozen public operator manifest.
    // This does not select a fallback or load a kernel while recording a Graph.
    void require_names(const std::vector<std::string>& names);

    // O(1) 快速获取 - 通过枚举 ID 直接数组索引 (推荐热点路径使用)
    inline ::rhino_lkn::Kernel_t* get(KernelId id) const {
        return fast_cache_[static_cast<size_t>(id)];
    }

    // Read-only payload availability for cold capability snapshots.  Callers
    // must already be inside the runtime-ready lifecycle; unlike get_kernel(),
    // this never attempts a dynamic load and never mutates Graph state.
    inline bool has_loaded(KernelId id) const {
        return payload_availability_.has_loaded(id);
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
        return get_program(KERNEL_ID_NAMES[static_cast<size_t>(id)]);
    }

    // 检查是否已初始化
    bool is_initialized() const { return initialized_; }

    // Original canonical paths and immutable bytes of successful Program loads.
    std::vector<std::string> loaded_oplib_paths() const;
    std::vector<std::pair<std::string, std::string>> loaded_oplib_bytes() const;

    // 清空所有缓存的 kernel/program (用于 shutdown)
    void clear() {
        // Revoke capability discovery before destroying any pointer it was
        // derived from. Concurrent readers observe either the prior immutable
        // publication or an uninitialized-runtime rejection, never raw cache
        // storage participating in teardown.
        payload_availability_.withdraw();
        kernels_.clear();  // unique_ptr 会自动释放 Program_t/Kernel_t
        {
            std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
            loaded_oplib_paths_.clear();
            oplib_symbols_.clear();
            oplib_snapshots_.clear();  // Programs were destroyed before their files.
        }
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
    std::shared_ptr<OplibSnapshot> oplib_snapshot(const std::string& path);
    const std::unordered_set<std::string>& kernel_manifest_names(const std::string& path);

    bool initialized_;
    KernelPayloadAvailability payload_availability_;
    std::unordered_map<std::string, CachedKernel> kernels_;
    mutable std::mutex loaded_oplib_paths_mutex_;
    std::set<std::string> loaded_oplib_paths_;
    std::unordered_map<std::string, std::shared_ptr<OplibSnapshot>> oplib_snapshots_;
    std::unordered_map<std::string, std::unordered_set<std::string>> oplib_symbols_;
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
    // 计数 (P8.0 BufferPool exhaustion 修复)。
    inline ::rhino_lkn::Queue_t* get(int64_t core_num, bool* fresh = nullptr) {
        TORCH_CHECK(core_num >= 1 && core_num <= 8,
                    "QueueCache core count must be in [1,8]");
        const auto idx = static_cast<uint8_t>(core_num - 1);
        if (__builtin_expect(queues_[idx] != nullptr, 1)) {
            if (fresh) *fresh = false;
            return queues_[idx];
        }
        queues_[idx] = new ::rhino_lkn::Queue_t(static_cast<uint8_t>(core_num));
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
    const RpuDeviceDmaEndpointRequest requests[] = {
        {src_addr, bytes, where}, {dst_addr, bytes, where}};
    RpuDmaEndpoint endpoints[2];
    [[maybe_unused]] auto submission_lease =
        rpu_resolve_device_dma_endpoints(requests, 2, endpoints);
    rpu_add_dma_checked(
        q, endpoints[0], endpoints[1], bytes, channel, where);
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
    const RpuDeviceDmaEndpointRequest requests[] = {
        {src_addr, bytes, where}, {dst_addr, bytes, where}};
    RpuDmaEndpoint endpoints[2];
    [[maybe_unused]] auto submission_lease =
        rpu_resolve_device_dma_endpoints(requests, 2, endpoints);
    rpu_add_dma_mutable_checked(
        q, endpoints[0], endpoints[1], bytes, channel, where);
}

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
