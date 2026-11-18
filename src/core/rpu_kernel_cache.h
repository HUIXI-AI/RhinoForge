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
#include "rpu_linear_tiling.h"
#include "rpu_pi_wall_tiling.h"

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
    // DDR to SPM memcpy v2 (multi-core, nested loop fix for loop.gt.s16 overflow)
    // SPM to DDR memcpy v2 (multi-core, nested loop fix for loop.gt.s16 overflow)
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
    // cos/sin (grid: NUM_ELT_PER_WRP=512, grid_y=num_tokens). Bound to kernel
    // "partial_rope_1d" (rhino-ops asm/partial_rope/).
    PARTIAL_ROPE_1D,
    // Qwen3.5 GDN decode 核心(be627ce5 起在默认 ref 中)。
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
    // Explicit high-accuracy unary formulas.
    UNARY_GELU_TANH_SPM,
    UNARY_GELU_ERF_SPM,
    UNARY_GELU_ERF_ULTRA_SPM,
    UNARY_SOFTPLUS_SPM,
    // Fused ring all-reduce + residual. The launcher selects no-pace only for
    // the generator-certified sub-11-KiB/core envelope.
    LLM_ALL_REDUCE_RESIDUAL_RING_PACED,
    LLM_ALL_REDUCE_RESIDUAL_RING_NOPACE,
    // Wall-OSS RTC mixed-precision path. These kernels live in the expansion
    // ref and are optional unless that path is enabled. Keep them at the end
    // so existing KernelId values stay stable.
    LLM_ALL_REDUCE_BF16_PARTIAL_FP16_RESIDUAL,
    LLM_ALL_REDUCE_BF16_PARTIAL_FP16_RESIDUAL_NOPACE,
    LLM_ALL_REDUCE_FP16_PARTIAL_BF16_RESIDUAL,
    LLM_ALL_REDUCE_FP16_PARTIAL_BF16_RESIDUAL_NOPACE,
    RMS_NORM_BF16IN_FP16OUT_V32,
    // Raw-SPM V transpose + by-MHA attention. Append-only to keep existing
    // KernelId values stable; launchers validate the runtime shape.
    V_TRANSPOSE_SPM_VCTXLEN,
    SDPA_BY_MHA_SPM_VCTXLEN,
    // Exact Wall RTC Vision compact-residual ring variants. Append-only: the
    // independent symbols make an old expansion ref fail at lookup instead of
    // interpreting the new residual mode with an old kernel.
    LLM_ALL_REDUCE_RESIDUAL_RING_LOCAL_SPM,
    LLM_ALL_REDUCE_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_SPM,
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
    // R50 —— `llama_mrope_interleave_local_spm_tbl`：MROPE 的 cos/sin 表放在
    // **per-core SPM** 而不是 DDR 的孪生（R48b 的完整差集扫出来、ref 里有、全仓库
    // 未接线）。动机：R49 实测 `llama_rope` 的成本 = grid.x × ~196 ns，与搬多少
    // 元素**无关**（640 枪里元素量差一倍的两组落在同一个 10.0 µs 桶），而同构的
    // `rms_norm`（同 grid、同 8 核 broadcast、操作数全在 SPM）只要 76 ns/行 ——
    // 唯一的结构差就是 rope 的 cos/sin 在 DDR。
    // ⚠️ 契约是**推**的（照 `rope_2d_ddr`→`rope_2d_spm` 的先例：cos/sin 寄存器
    // 从 DDR 的 `>>8` 改成裸 per-core SPM 字节偏移），必须由隔离探针逐位证实。
    // optional ⇒ ref 里没有时自动跳过。
    MROPE_SPM_TBL,
    // Exact Wall P640/C320 W8A16 QKV. The optional expansion symbol keeps
    // independently swizzled Q/K/V DDR bases and writes unified-head SPM.
    // Exact Wall Vision 3x576 W8A16 QKV, packed as M1728/H1280.
    // New IDs are append-only: keep the existing Wall/Vision values stable.
    // Exact Wall Action H20 W8A16 QKV, packed as M20/K1024.
    // Exact Wall Action H20 W8A16 gate+up dual-base projection.
    // Exact Wall Prefill M320/M640 gate+up+SwiGLU with bounded tile scratch.
    // Exact Wall P640/C320 attention ring + compact residual + post RMSNorm.
    WALL_RING_POSTNORM_FP16_PARTIAL_BF16_RESIDUAL_M320N2048,
    // Exact Wall P640/C320 compact-residual in/out ring + post RMSNorm.
    WALL_RING_POSTNORM_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT_M320N2048,
    // Exact Wall Vision M576 gate+up+SwiGLU with per-projection bias.
    // Exact Wall Vision M576 down ring + compact raw carry + next RMSNorm.
    WALL_DOWN_POSTNORM_M576_FP16_PARTIAL_FP16_RESIDUAL,
    WALL_DOWN_POSTNORM_M576_BF16_PARTIAL_FP16_RESIDUAL,
    WALL_DOWN_POSTNORM_M576_FP16_PARTIAL_BF16_RESIDUAL,
    // Exact Wall Vision B3/S576 native-window raw-SPM attention.
    WALL_WINDOW_SDPA_BY_MHA_B3S576H16D80,
    // Exact Wall Vision M576 compact-residual in/out ring + post RMSNorm.
    WALL_RING_POSTNORM_M576_FP16_PARTIAL_FP16_RESIDUAL_LOCAL_INOUT,
    WALL_RING_POSTNORM_M576_BF16_PARTIAL_FP16_RESIDUAL_LOCAL_INOUT,
    WALL_RING_POSTNORM_M576_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT,
    // Exact Wall P640 packed QKV with one N64 weight load for two M320 halves.
    // Exact Wall P640 Down with one N64 weight load for two C320 halves.
    // Exact Wall P640 GateUp with one N96 weight load across five M128 tiles.
    // Exact Wall P640 O-proj with one N64 weight load for two C320 halves.
    // Exact Wall Vision GateUp with one N96 weight load across three images.
    // Exact Wall Vision QKV with one N96 weight load across three images.
    // Exact Wall Action M20 QKV with in-place partial M-RoPE on Q/K.
    WALL_ACTION_QKV_PARTIAL_MROPE_W8A16_M20N64,
    // Exact Wall Action M20 row-linear + ring + residual + RMSNorm.
    // Exact Wall P640 pair: two contiguous C320 carriers in one launch.
    WALL_RING_POSTNORM_PAIR_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT_M320X2N2048,
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
    // Qwen3.5 sparse-MoE kernels. Append-only to preserve all existing IDs.
    MOE_TOPK_SCALAR_FP16,
    MOE_TOPK_BY_KVSORT_FP16,
    MOE_BINCOUNT_SCALAR,
    MOE_TOPK_PROB_NORM_FP16,
    CAST_UINT16_INT32,
    GATHER_SCALAR_FP16,
    GATHER_FP16,
    MOE_GATHER_MUL_V2_FP16,
    SCATTERND_ADD_INT16_FP16,
    // Bounded shared row W8A16 schedule: one N80 weight tile across M blocks.
    // Optional expansion payload; append-only to preserve existing identities.
    // Qwen3-VL 4B W8 decode fused routes.
    QWEN3VL_GATEUP_SWIGLU_GEMV,
    QWEN3VL_QK_NORM_MROPE_D128,
    QWEN3VL_QK_NORM_MROPE_KV_INSERT_D128,
    QWEN3VL_O_RING_NORM_M1_H2560_W8,
    PI05_ADARMS_NORM_SHIFT_H1024,
    PI05_GATED_RESIDUAL_H1024,
    PI05_RING_XOR3_M768N1152,
    PI05_RING_XOR3_M400N2048,
    PI05_RING_XOR3_M50N1024,
    PI05_K_ROPE_INSERT_M50D256P64,
    PI05_OWNER_NORM_FULL_RESIDUAL_M400N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M400N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M400N2048,
    // Exact Pi0.5 denoise continuation: XOR3 sum, FP16 gate, residual add.
    PI05_XOR3_GATED_RESIDUAL_M50N1024,
    // Owner-local alias of the generated M592/N32/K128 W8A16 program.  The
    // distinct ID makes KV1 typed-census counts independent of other linears.
    // Optional expansion payload: consumes pair-mapped compact K/V stripes
    // and writes the exact P800/M50/PAD64 physical-KV8 cache suffix.
    PI05_DENOISE_KV1_DIRECT_CACHE_M50D256P64,
    // Optional expansion payload: reads the installed full replicated K/V
    // W8A16 owners and emits both pair-mapped compact projections at once.
    // Optional exact-prefill expansion payload: reads the installed K2048
    // physical-KV8 owners and emits compact logical-KV1 K/V together.
    // Optional exact-prefill expansion payload: applies K RoPE to the compact
    // pair and writes one aligned C400 segment directly to physical-KV8 cache.
    PI05_PREFILL_KV1_DIRECT_CACHE_M400D256P400,
    // Optional exact-prefill expansion payload: reuses each Gate/Up weight
    // tile across both C400 halves and retires the production FP16 GeGLU.
    // Exact M50/H1024/I4096: original N48 ACC16 Gate/Up plus FP16 GeGLU.
    PARALLEL_LINEAR_NVFP4_V2_M320N64,
    PARALLEL_LINEAR_NVFP4_V2_M384N48,
    PI05_DENOISE_GATE_UP_GEGLU_NVFP4_M320N64K128,
    PI05_OWNER_NORM_COMPACT_A8_M400N2048,
    // Optional exact Vision owner96 LayerNorm and compact-residual companion.
    PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152,
    PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152,
    // Append-only Pi0.5 dtype/camera coverage variants. Internal IDs exceed
    // 255; cache and graph name lookups must retain the complete index.
    PI05_VISION_OWNER_REDUCE_LAYERNORM_M512N1152,
    PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M512N1152,
    PI05_RING_XOR3_M512N1152,
    PI05_RING_XOR3_M272N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M272N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M272N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M272N2048,
    PI05_OWNER_NORM_COMPACT_A8_M272N2048,
    PI05_PREFILL_KV1_DIRECT_CACHE_M272D256P272,
    PI05_RING_XOR3_M416N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M416N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M416N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M416N2048,
    PI05_OWNER_NORM_COMPACT_A8_M416N2048,
    PI05_PREFILL_KV1_DIRECT_CACHE_M416D256P416,
    PI05_RING_XOR3_M288N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M288N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M288N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M288N2048,
    PI05_OWNER_NORM_COMPACT_A8_M288N2048,
    PI05_PREFILL_KV1_DIRECT_CACHE_M288D256P288,
    PI05_RING_XOR3_M448N2048,
    PI05_RING_XOR3_M464N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M448N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M464N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M448N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M464N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M448N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M464N2048,
    PI05_OWNER_NORM_COMPACT_A8_M448N2048,
    PI05_OWNER_NORM_COMPACT_A8_M464N2048,
    PI05_PREFILL_KV1_DIRECT_CACHE_M448D256P448,
    PI05_PREFILL_KV1_DIRECT_CACHE_M464D256P464,
    PI05_RING_XOR3_M320N2048,
    PI05_RING_XOR3_M336N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M320N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M336N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M320N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M336N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M320N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M336N2048,
    PI05_OWNER_NORM_COMPACT_A8_M320N2048,
    PI05_OWNER_NORM_COMPACT_A8_M336N2048,
    PI05_PREFILL_KV1_DIRECT_CACHE_M320D256P320,
    PI05_PREFILL_KV1_DIRECT_CACHE_M336D256P336,
    PI05_RING_XOR3_M432N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M432N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M432N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M432N2048,
    PI05_OWNER_NORM_COMPACT_A8_M432N2048,
    PI05_PREFILL_KV1_DIRECT_CACHE_M432D256P432,
    PI05_RING_XOR3_M304N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M304N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M304N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M304N2048,
    PI05_OWNER_NORM_COMPACT_A8_M304N2048,
    PI05_PREFILL_KV1_DIRECT_CACHE_M304D256P304,
    // Exact Wall Action kernels with core-local compact SPM weights.
    WALL_ACTION_QKV_PARTIAL_MROPE_W8A16_M20N64_SPM_WEIGHT,
    // Exact Wall Action DDR QKV: one K/V owner per four-core KV group,
    // followed by a remote-SPM read/local-SPM write fanout.
    WALL_ACTION_QKV_OWNER_PARTIAL_MROPE_W8A16_M20N64,
    WALL_ACTION_KV_OWNER_FANOUT_M20,
    // Exact Wall Prefill P640 QKV: K/V are computed only by the four owners
    // consumed by WALL_PREFILL_OWNER_DIRECT_KV_M640.
    // Exact Wall Prefill P640 owner K/V -> RoPE/transpose -> DDR cache.
    WALL_PREFILL_OWNER_DIRECT_KV_M640,
    // Experimental exact Vision row Linear: full local-K weight reuse over M1728.
    // Adopted parcel-sorting schedules; all main-branch IDs above stay fixed.
    // Prefill row activation reuse.
    // Prefill MLP and paired postnorm retile.
    WALL_RING_POSTNORM_PAIR_TILE40_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT_M320X2N2048,
    WALL_ACTION_QKV_FP16_PARTIAL_MROPE_M20N64K128,
    WALL_ACTION_QKV_FP16_OWNER_PARTIAL_MROPE_M20N64K128,
    WALL_ACTION_QKV_FP16_PARTIAL_MROPE_SPM_WEIGHT_M20N64K128,
    // NVFP4 ACC16 tiles retain the ACC32 family geometry and physical ABI.
    PL_AT_NVFP4_ACC16_M176N128,
    PL_AT_NVFP4_ACC16_M208N112,
    PL_AT_NVFP4_ACC16_M240N96,
    PL_AT_NVFP4_ACC16_M272N80,
    PL_AT_NVFP4_ACC16_M320N64,
    PL_AT_NVFP4_ACC16_M384N48,
    PL_AT_NVFP4_ACC16_M480N32,
    // Distinct ACC16 GeGLU payload; the historical ACC32 ID remains unchanged.
    PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M320N64K128,
    // Optional explicit high-precision unary payloads. Append-only IDs.
    UNARY_SILU_HIGH_PRECISION,
    UNARY_TANH_HIGH_PRECISION,
    UNARY_GELU_ERF_HIGH_PRECISION,
    RHINOVLA_NEWTON_NORM_SHIFT_H1024,
    RHINOVLA_SILU_HIGH_MUL,
    UNARY_GELU_TANH_HIGH_PRECISION,
    GR00T_FF1_RESIDENT_W8A16_M512N48K128,
    GR00T_QKV_MULTIBASE_W8A16_M592N32K128,
    GR00T_ALL_REDUCE_RESIDUAL_XOR3_M41N1536,
    // Grouped unsigned INT4, plain FP16 group-scale/negative-zero ABI.
    HYVLA_EXPERT_O_RING_POSTNORM_S51,
    HYVLA_EXPERT_QK_ROPE_S51,
    HYVLA_EXPERT_GATEUP_W8_S51_N48,
    HYVLA_W8_RESIDENT_SWIGLU_S51,
    HYVLA_EXPERT_O_RING_POSTNORM_W8_S51,
    HYVLA_EXPERT_O_RESIDENT_W8_S51,
    HYVLA_NVFP4_GATEUP_SWIGLU_S51,
    HYVLA_NVFP4_GATEUP_RESIDENT_S51,
    HYVLA_EXPERT_O_RING_POSTNORM_NVFP4_S51,
    HYVLA_EXPERT_O_RING_POSTNORM_NVFP4_S51_N128,
    HYVLA_NVFP4_GATEUP_SWIGLU_INPUT_REUSE_S51,
    HYVLA_EXPERT_DOWN_NVFP4_S51_N128,
    HYVLA_EXPERT_DOWN_NEXTNORM_NVFP4_S51,
    HYVLA_EXPERT_DOWN_NEXTNORM_NVFP4_S51_N128,
    HYVLA_EXPERT_QK_ROPE_NORM_S51,
    HYVLA_EXPERT_KV_NVFP4_S51,
    HYVLA_EXPERT_QKV_NVFP4_S51,
    HYVLA_EXPERT_QK_ROPE_NORM_KV_S51,
    HYVLA_NVFP4_RESIDENT_SWIGLU_S51,
    HYVLA_EXPERT_Q_NVFP4_S51_N32,
    HYVLA_EXPERT_Q_NVFP4_S51_OVERLAP,
    HYVLA_NVFP4_RESIDENT_SCALES_S51,
    HYVLA_NVFP4_RESIDENT_SCALES_SWIGLU_S51,
    HYVLA_EXPERT_QKV_NVFP4_S51_OVERLAP,
    HYVLA_EXPERT_SDPA_M16_S51,
    HYVLA_EXPERT_O_RING_POSTNORM_NVFP4_S51_N128_OVERLAP,
    HYVLA_EXPERT_DOWN_NEXTNORM_NVFP4_S51_OVERLAP,
    HYVLA_NVFP4_GATEUP_SWIGLU_S51_OVERLAP,
    HYVLA_NVFP4_RESIDENT_SWIGLU_S51_OVERLAP,
    HYVLA_EXPERT_SDPA_K128_S51,
    HYVLA_VLM_VISION_GU176_W8,
    HYVLA_EXPERT_KV_FP16_S51,
    HYVLA_EXPERT_KV_W8_S51,
    HYVIT2_VISION_OWNER_LAYERNORM_M392N1152,
    HYVIT2_VISION_COMPACT_RESIDUAL_M392N1152,
    WALL_BASE_NVFP4_QKV_BIAS_BARRIER_N64,
    WALL_BASE_NVFP4_QKV_BIAS_BARRIER_N32,
    WALL_BASE_NVFP4_O_RING_NORM_H32,
    WALL_BASE_NVFP4_DOWN_RING_NORM_H32,
    WALL_BASE_PARTIAL_MROPE_QK_H32,
    WALL_BASE_NVFP4_GATE_UP_PARALLEL_H32,
    WALL_BASE_NVFP4_KV_PARALLEL_BIAS_H32,
    WALL_BASE_NVFP4_QKV_PARALLEL_BIAS_H32,
    WALL_BASE_NVFP4_QKV_BIAS_ROPE_H32,
    WALL_BASE_NVFP4_GATE_UP_PARALLEL_SILU_H32,
    WALL_BASE_NVFP4_GATE_UP_SHARED_SILU_H32,
    WALL_BASE_VISION_OWNER_RING_NORM_M672,
    WALL_BASE_VISION_DOWN_OWNER_RING_NORM_M672,
    WALL_BASE_VISION_QK_ROPE_2D_SPM,
    WALL_BASE_VISION_RAGGED_SDPA,
    WALL_BASE_PREFILL_O_OWNER_RING_NORM_M224,
    G05_GATEUP_RESIDENT_FP16_ACC32_M304N80K128,
    G05_KV_OWNER_FP16_ACC32_M32N256K1024,
    G05_KV_OWNER_FANOUT_M32D256,
    G05_VISION_OWNER_REDUCE_LAYERNORM_M256N1024,
    G05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M256N1024,
    G05_Q_GATE_FP16_ACC32_M32N256K1024,
    G05_GATE_UP_FP16_ACC32_M32N512K1024,
    G05_SILU_ROUND_MUL_M32N512,
    G05_VISION_QKV_FP16_ACC32_M256N128K1024,
    G05_TEXT_GATE_UP_FP16_ACC32_M256N768K2048,
    G05_GATEUP_RESIDENT_W8A16_ACC32_M240N96K128,
    G05_KV_OWNER_W8A16_ACC32_M32N256K1024,
    G05_Q_GATE_W8A16_ACC32_M32N256K1024,
    G05_GATE_UP_W8A16_ACC32_M32N512K1024,
    G05_VISION_QKV_W8A16_ACC32_M256N128K1024,
    G05_TEXT_GATE_UP_W8A16_ACC32_M256N768K2048,
    G05_QKV_GATE_OWNER_W8A16_ACC32_M32N256K1024,
    G05_K_OWNER_RMS_NORM_M32D256,
    G05_K_OWNER_PARTIAL_MROPE_M32D256R64,
    G05_OWNER_INSERT_KCACHE_MULTIWARP,
    G05_OWNER_INSERT_KCACHE_MULTIWARP_V16,
    G05_OWNER_INSERT_VCACHE_MULTIWARP,
    G05_OWNER_INSERT_VCACHE_MULTIWARP_V16,
    // Genuine generated ACC16 W8 specializations; historical IDs stay fixed.
    G05_GATEUP_RESIDENT_W8A16_ACC16_M352N96K128,
    G05_KV_OWNER_W8A16_ACC16_M32N256K1024,
    G05_Q_GATE_W8A16_ACC16_M32N256K1024,
    G05_GATE_UP_W8A16_ACC16_M32N512K1024,
    G05_VISION_QKV_W8A16_ACC16_M256N128K1024,
    G05_TEXT_GATE_UP_W8A16_ACC16_M256N768K2048,
    G05_QKV_GATE_OWNER_W8A16_ACC16_M32N256K1024,
    G05_SUB_GATED_RESIDUAL_H1024,
    G05_RTC_ADARMS_NORM_SHIFT_M32H1024,
    G05_O_RESIDENT_W8A16_ACC16_M32N1024K256,
    G05_RTC_SUB_GATED_RESIDUAL_M32H1024,
    G05_RING_RTC_SUB_GATED_RESIDUAL_M32H1024,
    G05_DOWN_PARTIAL672_W8A16_ACC16_M32N1024K512,
    G05_VISION_FC2_WEIGHT_OUTER_W8A16_ACC16_M768N1024K512,
    G05_GATE_UP_PAIRED_NVFP4_ACC16_M32N512K1024_N128,
    G05_O_RESIDENT_NVFP4_ACC16_M32N1024K256_N80,
    G05_DOWN_RESIDENT_NVFP4_ACC16_M32N1024K512_N64,
    G05_GATE_UP_SWIGLU_NVFP4_ACC16_M32N512K1024_N128,
    G05_GATE_KV_OWNER_W8A16_ACC16_M32N256K1024_N128,
    G05_Q_NORM_PARTIAL_ROPE_M32D256R64_GROUP4,
    G05_K_OWNER_NORM_PARTIAL_ROPE_M32D256R64_GROUP4,
    G05_TEXT_RING_OWNER_RMS_M256H2048,
    G05_OWNER_INSERT_KV_CACHE_M32D256_P0,
    G05_OWNER_INSERT_KV_CACHE_M32D256_P245,
    G05_ATTN_SIGMOID_ROUND_MUL_M32D256,
    G05_RING_RTC_THEN_NORM_SHIFT_M32H1024,
    G05_GATE_UP_FP16_ACC16_M32N512K1024,
    G05_TEXT_GATE_UP_FP16_ACC16_M256N768K2048,
    G05_GATE_UP_PAIRED_NVFP4_ACC32_M32N512K1024_N128,
    G05_O_RESIDENT_NVFP4_ACC32_M32N1024K256_N80,
    G05_DOWN_RESIDENT_NVFP4_ACC32_M32N1024K512_N64,
    G05_GATE_UP_SWIGLU_NVFP4_ACC32_M32N512K1024_N128,
    HYVLA_EXPERT_RMSNORM_V16_BITCOPY,
    // Optional LingBot2 FP16 grouped GateUp; older refs retain the base route.
    LINGBOT2_GROUPED_GATEUP_EXPERT_Y4_ACC16,
    // Optional fixed-profile compaction of the selector's exact membership.
    LINGBOT2_ROUTE_COMPACT_E32_T51_TP64,
    // Existing fused top-k/normalization payload; model admission is separate.
    MOE_EXACT_TOPK_DENSE_ROUTE_FP16,
    // Optional MAIN payloads for the exact LingBot2 dense-bank Gate/Up route.
    LINGBOT2_ROUTED_COL_FP16_M608N32,
    LINGBOT2_ROUTED_COL_W8_M592N32,
    // Guarded fused top-k; never substitute the unguarded MAIN payload.
    LINGBOT2_TOPK_DENSE_ROUTE_SAFE,
    // Optional no-gamma normalization for cold-bound folded weights.
    RMSNORM_NOGAMMA,
    RMSNORM_NOGAMMA_V16,
    RMSNORM_NOGAMMA_V32,
    // Optional exact-profile ordered FP16 sparse combine; old refs use scatter.
    LINGBOT2_COMBINE_T51_R204_H768_SCAN,
    // Optional fixed E32/K4 lane-pair selection with the unchanged SAFE numeric tail.
    LINGBOT2_TOPK_E32_K4_LANE2_SAFE,
    LINGBOT2_SWIGLU_ROUTE_R204_D64,
    LINGBOT2_COMBINE_T51_R204_H768_INVERSE,
    LINGBOT2_ROUTE_COMPACT_INVERSE_E32_T51_TP64,
    // Optional INT8-weight GateUp with the same TP8/expert-y4 ownership.
    LINGBOT2_GROUPED_GATEUP_W8A16_EXPERT_Y4_ACC16,
    // Ordered routed combine followed by one FP16 shared-expert ADD.
    LINGBOT2_COMBINE_SHARED_T51_R204_H768_INVERSE,
    // Optional raw-U16 routed input loads; preserve the existing y4 math.
    LINGBOT2_GROUPED_GATEUP_ROUTED_EXPERT_Y4_ACC16,
    LINGBOT2_GROUPED_GATEUP_ROUTED_W8A16_EXPERT_Y4_ACC16,
    // Optional same-warp routed GateUp, FP16 SwiGLU and routed-weight multiply.
    LINGBOT2_GROUPED_GATEUP_ROUTED_FP16_SWIGLU_ROUTE,
    LINGBOT2_GROUPED_GATEUP_ROUTED_W8_SWIGLU_ROUTE,
    // Optional fixed-profile W8 Down with one combined scale shuffle.
    LINGBOT2_GROUPED_DOWN_W8A16_ACC16_SCALE_SHUFFLE,
    // Optional exact51 paired K/V writer; preserves the physical tail rows.
    LINGBOT2_INSERT_KV_T51,
    // Optional exact shared SiLU with compact sixteen-half vector scheduling.
    LINGBOT2_SHARED_SILU_T51_D96_VECTOR,
    // Optional exact Action QKV dispatch, preserving each auto ACC16 tile.
    LINGBOT2_QKV_MULTIBASE_FP16_T51,
    LINGBOT2_QKV_MULTIBASE_W8_T51,
    // Optional exact Action SDPA with an effective 32-key final tile.
    LINGBOT2_SDPA_T51_K275_TAIL,
    // Optional exact M768/H1024/I4096 Vision FC1 + original BASE tanh-GELU.
    LINGBOT2_VISION_FC1_GELU_FP16,
    LINGBOT2_VISION_FC1_GELU_W8,
    // Optional exact language-prefill QKV dispatch with unchanged auto tiles.
    LINGBOT2_PREFILL_QKV_MULTIBASE_FP16_M224,
    LINGBOT2_PREFILL_QKV_MULTIBASE_W8_M224,
    // Optional original 32b minibatch Vision SDPA with raw SPM K/V inputs.
    LINGBOT2_VISION_SDPA_SPM_B3S256H16D64_32B,
    // Optional original N64 Down followed by ordered Top4 and shared FP16 ADDs.
    LINGBOT2_GROUPED_DOWN_FP16_ORDERED_COMBINE,
    LINGBOT2_GROUPED_DOWN_W8_ORDERED_COMBINE,
    // Optional T51/Tp64 lane2 Top4, bit transposes and inverse route compact.
    LINGBOT2_TOPK_COMPACT_INVERSE_E32_T51_TP64,
    // Optional exact H768 AdaRMS bit-diagonal shuffle; LingBot only.
    LINGBOT2_ADARMS_NORM_SHIFT_H768_DIAGONAL,
    // Optional exact M768/N1024/K1024 Vision QKV with independent bias owners.
    LINGBOT2_VISION_QKV_JOINT_FP16_M768,
    LINGBOT2_VISION_QKV_JOINT_W8_M768,
    // Optional exact Action paired RoPE followed by same-tile paired KV writes.
    LINGBOT2_QK_ROPE_KV_T51,
    // Optional exact W8 M224 Prefill shared-A Gate/Up and original SwiGLU.
    LINGBOT2_PREFILL_GATEUP_SHARED_A_W8_M224,
    // Optional exact T51/H768/E32 Router ACC32 + BASE sigmoid + Tp64 ones.
    LINGBOT2_ROUTER_SIGMOID_PAD_T51,
    // Optional exact T51/H768/localI96 W8 shared Gate/Up + resident SiLU.
    LINGBOT2_SHARED_GATEUP_RESIDENT_SILU_W8_T51,
    // Optional exact T51/H768/localI96 FP16 shared Gate/Up + resident SiLU.
    LINGBOT2_SHARED_GATEUP_RESIDENT_SILU_FP16_T51,
    // Optional exact T51/H768 post-attention ring + original AdaRMS norm/shift.
    LINGBOT2_ATTN_ALL_REDUCE_ADARMS_T51_H768,
    // Optional M224 learned-gamma V32 Q/K norms plus original partial RoPE.
    LINGBOT2_PREFILL_QK_NORM_ROPE_M224,
    // Generated main Linear candidates. Marker aliases do not occupy slots.
    HYVLA_EXPERT_GATEUP_SWIGLU_FP16_V2,
    HYVLA_VLM_GATEUP_SWIGLU_FP16_V2,
    FUSED_GEGLU_W16A16_ACC16_M80N96K512,
    FUSED_GEGLU_W16A16_ACC32_M64N96K512,
    FUSED_GEGLU_W4A16_PGRP_ACC16_M464N64K128,
    FUSED_GEGLU_W4A16_PGRP_ACC16_M736N64K64,
    FUSED_GEGLU_W4A16_PGRP_ACC32_M352N64K128,
    FUSED_GEGLU_W4A16_PGRP_ACC32_M496N64K64,
    FUSED_GEGLU_W8A16_ACC16_M64N80K512,
    FUSED_GEGLU_W8A16_ACC32_M96N80K384,
    HYVLA_GATEUP_RESIDENT_PAIR_FP16_V2,
    HYVLA_GATEUP_RESIDENT_PAIR_W8_V2,
    HYVLA_RESIDENT_SWIGLU_FP16_V2,
    HYVLA_VLM_GATEUP_SWIGLU_W8_V2,
    PL_AUTOTILE_BEGIN,
#define RPU_PL_CANDIDATE(id, name, ordinal) id = PL_AUTOTILE_BEGIN + ordinal,
#define RPU_GROUPED_PL_CANDIDATE(id, name, ordinal)
#include "rpu_linear_candidates.inc"
#undef RPU_PL_CANDIDATE
#undef RPU_GROUPED_PL_CANDIDATE
    GROUPED_PL_AUTOTILE_BEGIN = PL_AUTOTILE_BEGIN + rpu_pl_tiling::kParallelCandidateCount,
#define RPU_PL_CANDIDATE(id, name, ordinal)
#define RPU_GROUPED_PL_CANDIDATE(id, name, ordinal) id = GROUPED_PL_AUTOTILE_BEGIN + ordinal,
#include "rpu_linear_candidates.inc"
#undef RPU_PL_CANDIDATE
#undef RPU_GROUPED_PL_CANDIDATE
    PI_WALL_BEGIN = GROUPED_PL_AUTOTILE_BEGIN + rpu_pl_tiling::kGroupedCandidateCount,
#define RPU_PI_WALL_CANDIDATE(id, name, ordinal) id = PI_WALL_BEGIN + ordinal,
#include "rpu_pi_wall_candidates.inc"
#undef RPU_PI_WALL_CANDIDATE
    GROUPED_PL_FP16_ACC16_DEVICE_M = PI_WALL_BEGIN + rpu_pi_wall::kCandidateCount,
    GROUPED_PL_W8_ACC32_DEVICE_M = GROUPED_PL_FP16_ACC16_DEVICE_M + 1,
    GROUPED_PL_FP16_ACC32_N80_DEVICE_M = GROUPED_PL_W8_ACC32_DEVICE_M + 1,
    GROUPED_PL_FP16_ACC32_N32_DEVICE_M = GROUPED_PL_FP16_ACC32_N80_DEVICE_M + 1,
    GROUPED_PL_W8_ACC16_N48_DEVICE_M = GROUPED_PL_FP16_ACC32_N32_DEVICE_M + 1,
    GROUPED_PL_W8_ACC32_N48_DEVICE_M = GROUPED_PL_W8_ACC16_N48_DEVICE_M + 1,
    MOE_KVSORT_FP16 = GROUPED_PL_W8_ACC32_N48_DEVICE_M + 1,
#define GIL(indexed, dtype, id, symbol, m, n, k) id,
#include "rpu_grouped_indexed_linear_tiles.inc"
#undef GIL
    PI05_KV_OWNER_W8_DB,
    PI05_NVFP4_LINEAR_DB,
    PI05_NVFP4_LINEAR_DB_TSCALE,
    PI05_NVFP4_GEGLU_DB,
    PI05_NVFP4_GEGLU_DB_TSCALE,
#define GIL(indexed, dtype, id, symbol, m, n, k) id,
#include "rpu_grouped_indexed_linear_current_tiles.inc"
#undef GIL
    PI05_NVFP4_LINEAR_DB_SSPM,
    PI05_NVFP4_GEGLU_DB_SSPM,
    PI05_Q_KV1_PAIR_DB_SSPM,
    // Optional Pi0.5 H10 (M10, physical KV insert P16) denoise siblings of the
    // M50 FP16 XOR3 ring, XOR3 gated residual and KV1 direct-cache payloads.
    PI05_RING_XOR3_M10N1024,
    PI05_XOR3_GATED_RESIDUAL_M10N1024,
    PI05_DENOISE_KV1_DIRECT_CACHE_M10D256P16,
    // Optional Pi0.5 Action split-KV attention: pack a group's query rows,
    // attend over one core's share of the KV blocks, merge the partials.
    PI05_SPLIT_KV_PACK_Q,
    PI05_FLASH_ATTN_SPLIT_KV_DP,
    PI05_SPLIT_KV_MERGE,
    // Optional low-barrier sibling of the Pi0.5 M10 XOR3 + gated residual: the
    // same register ABI, SCM tables, atomics reset and output bytes.
    PI05_RING_GATED_RESIDUAL_LB_M10N1024,
    // Optional exact-M50 sibling of PI05_ADARMS_NORM_SHIFT_H1024: eight
    // row-strided blocks with resident scale/shift and the same output bytes.
    PI05_ADARMS_NORM_SHIFT_H1024_M50_WARP8,
    // Optional sibling of both PI05_DENOISE_KV1_DIRECT_CACHE_M10D256P16 and
    // _M50D256P64 (one instruction stream) with K/V loads overlapped: same ABI,
    // grids and bytes.
    PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_OVERLAP,
    // Optional sibling of PI05_SPLIT_KV_PACK_Q that also rotates the query rows
    // with the published partial_mrope arithmetic: the same packed bytes as
    // PARTIAL_MROPE followed by the pack, from unrotated rows.
    PI05_SPLIT_KV_PACK_Q_ROPE,
    // Optional sibling of PI05_FLASH_ATTN_SPLIT_KV_DP that keeps this core's
    // K/V blocks in a local SPM share: a fill prologue copies blocks
    // [max(S, R), E) from DDR, every warp then reads its tiles from the share.
    // The same register ABI plus param36/37 (share, 32-byte units from the
    // packed rows) and param38 = R; the same bytes.
    PI05_FLASH_ATTN_SPLIT_KV_SPM_DP,
    // Optional siblings of PI05_FLASH_ATTN_SPLIT_KV_DP and _SPM_DP that read
    // key block b from head slot (c + b - S) mod 8, S the first block of the
    // core's key range, instead of slot c: the same register ABI, and the same
    // bytes while every slot of a block holds the same KV head.
    PI05_FLASH_ATTN_SPLIT_KV_DP_ROT,
    PI05_FLASH_ATTN_SPLIT_KV_SPM_DP_ROT,
    // Optional sibling of PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_OVERLAP whose
    // param25 skips the tile stores of cache block b into head slot h when bit
    // (h - b) mod 8 is set, i.e. the slot that rotated reader core (h - b) mod 8
    // would read; param25 = 0 gives the same bytes.
    PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT,
    // Optional fusion of PI05_XOR3_GATED_RESIDUAL_M50N1024 and
    // PI05_ADARMS_NORM_SHIFT_H1024_M50_WARP8 into one launch: whole-row shards
    // give every row one owner, which normalizes it before the single
    // all-gather publishes it. Same inputs, same two FP16 roundings, same
    // normalized bytes; the raw residual it also publishes is compact.
    PI05_XOR3_GATED_ADARMS_M50N1024,
    // The same whole-row-shard continuation without the norm: the published
    // gated residual's own work and output, over shards of whole rows. A chain
    // of fused launches needs it, because the fused kernel publishes the raw
    // residual compactly and only a whole-row reader can consume that, and the
    // last link of a chain must publish a complete raw residual.
    PI05_XOR3_GATED_ROWS_M50N1024,
    // Optional sibling of PI05_SPLIT_KV_MERGE: core k of a group loads part
    // (k + j) mod group at load step j into that part's slots, so the group's
    // cores read different cores' SPM at every step. The same register ABI,
    // grid, VALU order and output bytes.
    PI05_SPLIT_KV_MERGE_ROT,
    // PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT (blocks < param28, its
    // registers 32 higher) and PI05_SPLIT_KV_PACK_Q_ROPE (the 2 * ceil(T / 16)
    // blocks after them, its registers 0..14 and SCM slot 0) in one launch,
    // grid (param28 + 2 * ceil(T / 16), 1, 1): the same cache and packed bytes
    // as the two launches. With grid (param28, 1, 1) it is the writer alone.
    PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64,
    // Optional sibling of PI05_ADARMS_NORM_SHIFT_H1024: block b normalizes rows
    // 2b and 2b + 1 (param14 = rows; at an odd count the last block's second
    // row is zero-trip), grid ceil(rows / 2), the same output bytes.
    PI05_ADARMS_NORM_SHIFT_H1024_ROW_PAIR,
    // Optional sibling of PI05_SPLIT_KV_MERGE_ROT: every load step's m and l
    // first, then every step's rows, and the weights and their spread are
    // computed while the rows load. The same register ABI, grid, parts, slots,
    // VALU order and output bytes.
    PI05_SPLIT_KV_MERGE_ROT_OV,
    // Optional sibling of PI05_XOR3_GATED_ADARMS_M50N1024 whose reduce reads
    // two peers per paced step (four pacing barriers instead of eight) into
    // the same slots in the same order. The same register ABI with p0 = 8,
    // grid, SCM table and output bytes.
    PI05_XOR3_GATED_ADARMS_TWOPEER_M50N1024,
    // PI05_ADARMS_NORM_SHIFT_H1024_ROW_PAIR (params 40..54 = its 0..14, blocks
    // below ceil(rows / 2)) and PI05_NVFP4_GEGLU_DB_SSPM
    // (params 0..37, grid 8) in one launch for M <= 16: the norm runs while the
    // Gate's first weight tile loads, a core barrier, then the GeGLU. The
    // GeGLU's register ABI and core counters; the two launches' bytes.
    PI05_DENOISE_ADARMS_GATE_UP_GEGLU_NVFP4_ACC16_M16N64K128_DB_SSPM,
    // Optional siblings of PI05_NVFP4_GEGLU_DB_SSPM
    // and PI05_DENOISE_ADARMS_GATE_UP_GEGLU_NVFP4_ACC16_M16N64K128_DB_SSPM:
    // blocks 0..7 run the Gate slices (and the norm), blocks 8..15 the Up
    // slices, which join before the GELU tail. The same registers with
    // param64 = 16, grid (16, 1, 1), the same core counters and bytes.
    PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M64N64K128_DB_SSPM_SPLIT,
    PI05_DENOISE_ADARMS_GATE_UP_GEGLU_NVFP4_ACC16_M16N64K128_DB_SSPM_SPLIT,
    // PI05_SPLIT_KV_MERGE (or PI05_SPLIT_KV_MERGE_ROT_OV) on blocks below
    // param35 = its grid, its params 0, 1, 2, 3, 6, 8, 9 moved to 28..34 and
    // its SCM slots, then PI05_NVFP4_LINEAR_DB_SSPM (params 0..27) on
    // the blocks after them, which wait at core counter 0 for the merge rows
    // before their first X load; grid (param35 + N64 blocks, 1, 1), the same
    // bytes as the two launches.
    PI05_SPLIT_KV_MERGE_O_NVFP4_ACC16_M64N64K128_DB_SSPM,
    PI05_SPLIT_KV_MERGE_ROT_OV_O_NVFP4_ACC16_M64N64K128_DB_SSPM,
    // PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64 with the writer's
    // cos/sin rows read from the pack's SPM RoPE rows instead of the DDR
    // tables: params 40/41 and 42/43 are their absolute SPM addresses and the
    // logical row 44/45 is 0. Only a launch that packs (SPM rows present)
    // takes it; the same cache and packed bytes.
    PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_SPM_ROPE_M64,
    // Optional siblings of PI05_FLASH_ATTN_SPLIT_KV_SPM_DP and _SPM_DP_ROT whose
    // fill copies K (and the V of a warp's later tiles) only: each warp loads
    // the V of its first fill tile from DDR on the first step while its softmax
    // runs, stores it into the share in front of the V share load and passes
    // the fill's core barrier again. The same register ABI, grid, share layout
    // and output bytes.
    PI05_FLASH_ATTN_SPLIT_KV_SPM_DP_LATE_V,
    PI05_FLASH_ATTN_SPLIT_KV_SPM_DP_ROT_LATE_V,
    // Optional sibling of PI05_XOR3_GATED_ADARMS_TWOPEER_M50N1024 whose reduce
    // runs on warp 0 alone: one pacing barrier, then every peer unpaced in
    // ring order, in two VLD stages so that the first stage's accumulate runs
    // under the second stage's loads. The same slots and order, register ABI
    // with p0 = 8, grid, SCM table and output bytes; a core's first segment
    // must fit 25 VLM entries of one warp and the rest of its shard 6.
    PI05_XOR3_GATED_ADARMS_ONEWARP_M50N1024,
    // Optional sibling of PI05_Q_KV1_PAIR_DB_SSPM whose Q weights are read
    // from each core's own SPM: params 2/3 are the absolute SPM address of the
    // layer's per-core run (its 512-byte plane of every 4 KiB DDR unit, in unit
    // order). Every other register, the grid and the output bytes are the sspm
    // kernel's.
    PI05_Q_KV1_PAIR_DB_WSPM,
    // Wall Qwen3.5 (Qwen3.5-2B VLA) exact optional expansion payloads.
    // Append-only after every existing ID; availability is published through
    // KernelPayloadAvailability's extended bitset like the other appended IDs.
    // Three-camera T192 (P960/C480) per-row prefill kernels. The C480 owner
    // norms differ from C464 only in the owner-row loop bound; the XOR3 ring,
    // the compact raw-full XOR3 and the KV1 direct cache are the C464
    // instructions under their own C480 names. There is no C480 A8 norm.
    PI05_RING_XOR3_M480N2048,
    PI05_OWNER_NORM_FULL_RESIDUAL_M480N2048,
    PI05_OWNER_NORM_COMPACT_RESIDUAL_M480N2048,
    PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M480N2048,
    PI05_PREFILL_KV1_DIRECT_CACHE_M480D256P480,
    // Dedicated activation-INT8 family; A16 tile ordinals remain unchanged.
#define RPU_W8A8_CANDIDATE(id, name, ordinal, m, n, k, local_k, flags) id,
#include "rpu_linear_w8a8_candidates.inc"
#undef RPU_W8A8_CANDIDATE
    DYNAMIC_QUANTIZE_W8A8,
    LAYER_NORM_QUANTIZE_W8A8_H1152,
    GELU_TANH_QUANTIZE_W8A8_H544,
    PI05_VISION_OWNER_LAYERNORM_QUANTIZE_W8A8_M512N1152,
    PI05_VISION_OWNER_LAYERNORM_QUANTIZE_W8A8_M768N1152,
    RMS_NORM_QUANTIZE_W8A8_H5120_M128,
    RMS_NORM_QUANTIZE_W8A8_H5120_M192,
    RMS_NORM_QUANTIZE_W8A8_H5120_M256,
    MUL_QUANTIZE_W8A8_H3200_M128,
    MUL_QUANTIZE_W8A8_H3200_M192,
    QWEN3_UP_MUL_W8A16_M320N112K128,
    QWEN3_UP_MUL_W8A8_M272N64K384,
    QWEN3_UP_MUL_W8A8_M512N64K128,
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
    static constexpr size_t kExtendedBegin = 0;
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
                 bool rmsnorm_newton_v32 = false,
                 bool wall_prefill_qkv = false,
                 bool pi05_norm_shift = false,
                 bool pi05_gated_residual = false,
                 bool nvfp4_v2_m320n64 = false,
                 bool nvfp4_v2_m384n48 = false,
                 bool qwen3vl_silu_mul_exact_strided = false) {
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
        if (wall_prefill_qkv) flags |= kWallPrefillQkv;
        if (pi05_norm_shift) flags |= kPi05NormShift;
        if (pi05_gated_residual) flags |= kPi05GatedResidual;
        if (nvfp4_v2_m320n64) flags |= kNvfp4V2M320N64;
        if (nvfp4_v2_m384n48) flags |= kNvfp4V2M384N48;
        if (qwen3vl_silu_mul_exact_strided) flags |= kQwen3VlSiluMulExactStrided;
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
            case KernelId::PARALLEL_LINEAR_NVFP4_V2_M320N64:
                return (state->flags & kNvfp4V2M320N64) != 0;
            case KernelId::PARALLEL_LINEAR_NVFP4_V2_M384N48:
                return (state->flags & kNvfp4V2M384N48) != 0;
            case KernelId::PI05_ADARMS_NORM_SHIFT_H1024:
                return (state->flags & kPi05NormShift) != 0;
            case KernelId::PI05_GATED_RESIDUAL_H1024:
                return (state->flags & kPi05GatedResidual) != 0;
            case KernelId::QWEN3VL_SILU_MUL_EXACT_STRIDED:
                return (state->flags & kQwen3VlSiluMulExactStrided) != 0;
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
            default:
                if (id < KernelId::_COUNT) {
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
    static constexpr uint32_t kWallPrefillQkv = 1u << 14;
    static constexpr uint32_t kPi05NormShift = 1u << 15;
    static constexpr uint32_t kPi05GatedResidual = 1u << 16;
    static constexpr uint32_t kNvfp4V2M320N64 = 1u << 17;
    static constexpr uint32_t kNvfp4V2M384N48 = 1u << 18;
    static constexpr uint32_t kQwen3VlSiluMulExactStrided = 1u << 19;
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
    "unary_gelu_tanh",
    "unary_gelu_erf",
    "unary_gelu_erf_ultra_precision",
    "unary_softplus",
    "llm_all_reduce_residual",
    "llm_all_reduce_residual_nopace",
    // Wall-OSS RTC mixed-precision Vision kernels.
    "llm_all_reduce_residual_bf16_partial_fp16_residual",
    "llm_all_reduce_residual_bf16_partial_fp16_residual_nopace",
    "llm_all_reduce_residual_fp16_partial_bf16_residual",
    "llm_all_reduce_residual_fp16_partial_bf16_residual_nopace",
    "llama_rms_norm_bf16in_fp16out_v32",
    "llm_fp16_32b_prefill_trans_prjv_dp_vctxlen",
    "llm_fp16_16b_prefill_flash_attn_by_mha_univ_vctxlen",
    "llm_all_reduce_residual_local_spm",
    "llm_all_reduce_residual_fp16_partial_bf16_residual_local_spm",
    // Hy-VLA optional RMSNorm/M-RoPE variants and FP16 ACC32 probes.
    "llama_rms_norm_v16",
    "llama_rms_norm_v32",
    "llama_rms_norm_newton_v16",
    "llama_rms_norm_newton_v32",
    "llama_mrope_interleave_local_spm_tbl",
    "wall_ring_postnorm_fp16_partial_bf16_residual_m320n2048",
    "wall_ring_postnorm_fp16_partial_bf16_residual_local_inout_m320n2048",
    "wall_down_postnorm_m576_fp16_partial_fp16_residual",
    "wall_down_postnorm_m576_bf16_partial_fp16_residual",
    "wall_down_postnorm_m576_fp16_partial_bf16_residual",
    "llm_fp16_16b_prefill_flash_attn_by_mha_wall_window_b3s576h16d80",
    "wall_ring_postnorm_m576_fp16_partial_fp16_residual_local_inout_xor3",
    "wall_ring_postnorm_m576_bf16_partial_fp16_residual_local_inout_xor3",
    "wall_ring_postnorm_m576_fp16_partial_bf16_residual_local_inout_xor3",
    "wall_action_qkv_w8_ddr_partial_mrope_v2",
    "wall_ring_postnorm_pair_fp16_partial_bf16_residual_local_inout_m320x2n2048",
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
    "qwen3vl_qkv_planar_w8_v2",
    "llama_silu_mul_exact_strided",
    "topk_scalar_fp16",
    "topk_by_kvsort_fp16",
    "bincount_scalar",
    "llm_topk_prob_norm",
    "unary_cast_uint16_int32",
    "gather_scalar",
    "gather",
    "llm_gather_mul_v2",
    "scatternd_int16_reduction",
    "qwen3vl_gateup_swiglu_w8_m1_v2",
    "qwen3vl_qk_norm_mrope_d128",
    "qwen3vl_qk_norm_mrope_kv_insert_d128",
    "qwen3vl_o_ring_norm_w8_m1_h2560_v2",
    "pi05_adarms_norm_shift_h1024",
    "pi05_gated_residual_h1024",
    "pi05_all_reduce_residual_xor3_m768n1152",
    "pi05_all_reduce_residual_xor3_m400n2048",
    "pi05_all_reduce_residual_xor3_m50n1024",
    "pi05_k_rope_insert_m50d256p64",
    "pi05_owner_norm_full_residual_m400n2048",
    "pi05_owner_norm_compact_residual_m400n2048",
    "pi05_xor3_compact_residual_raw_full_m400n2048",
    "pi05_xor3_gated_residual_m50n1024",
    "pi05_denoise_kv1_direct_cache_m50d256p64",
    "pi05_prefill_kv1_direct_cache_m400d256p400",
    "parallel_linear_wnvfp4a16_acc32_m320n64k128",
    "parallel_linear_wnvfp4a16_acc32_m384n48k128",
    "pi05_denoise_gate_up_geglu_nvfp4_acc32_m320n64k128",
    "pi05_owner_norm_compact_a8_m400n2048",
    "pi05_vision_owner_reduce_layernorm_m768n1152",
    "pi05_vision_xor3_compact_residual_raw_full_m768n1152_epoch",
    "pi05_vision_owner_reduce_layernorm_m512n1152",
    "pi05_vision_xor3_compact_residual_raw_full_m512n1152_epoch",
    "pi05_all_reduce_residual_xor3_m512n1152",
    "pi05_all_reduce_residual_xor3_m272n2048",
    "pi05_owner_norm_full_residual_m272n2048",
    "pi05_owner_norm_compact_residual_m272n2048",
    "pi05_xor3_compact_residual_raw_full_m272n2048",
    "pi05_owner_norm_compact_a8_m272n2048",
    "pi05_prefill_kv1_direct_cache_m272d256p272",
    "pi05_all_reduce_residual_xor3_m416n2048",
    "pi05_owner_norm_full_residual_m416n2048",
    "pi05_owner_norm_compact_residual_m416n2048",
    "pi05_xor3_compact_residual_raw_full_m416n2048",
    "pi05_owner_norm_compact_a8_m416n2048",
    "pi05_prefill_kv1_direct_cache_m416d256p416",
    "pi05_all_reduce_residual_xor3_m288n2048",
    "pi05_owner_norm_full_residual_m288n2048",
    "pi05_owner_norm_compact_residual_m288n2048",
    "pi05_xor3_compact_residual_raw_full_m288n2048",
    "pi05_owner_norm_compact_a8_m288n2048",
    "pi05_prefill_kv1_direct_cache_m288d256p288",
    "pi05_all_reduce_residual_xor3_m448n2048",
    "pi05_all_reduce_residual_xor3_m464n2048",
    "pi05_owner_norm_full_residual_m448n2048",
    "pi05_owner_norm_full_residual_m464n2048",
    "pi05_owner_norm_compact_residual_m448n2048",
    "pi05_owner_norm_compact_residual_m464n2048",
    "pi05_xor3_compact_residual_raw_full_m448n2048",
    "pi05_xor3_compact_residual_raw_full_m464n2048",
    "pi05_owner_norm_compact_a8_m448n2048",
    "pi05_owner_norm_compact_a8_m464n2048",
    "pi05_prefill_kv1_direct_cache_m448d256p448",
    "pi05_prefill_kv1_direct_cache_m464d256p464",
    "pi05_all_reduce_residual_xor3_m320n2048",
    "pi05_all_reduce_residual_xor3_m336n2048",
    "pi05_owner_norm_full_residual_m320n2048",
    "pi05_owner_norm_full_residual_m336n2048",
    "pi05_owner_norm_compact_residual_m320n2048",
    "pi05_owner_norm_compact_residual_m336n2048",
    "pi05_xor3_compact_residual_raw_full_m320n2048",
    "pi05_xor3_compact_residual_raw_full_m336n2048",
    "pi05_owner_norm_compact_a8_m320n2048",
    "pi05_owner_norm_compact_a8_m336n2048",
    "pi05_prefill_kv1_direct_cache_m320d256p320",
    "pi05_prefill_kv1_direct_cache_m336d256p336",
    "pi05_all_reduce_residual_xor3_m432n2048",
    "pi05_owner_norm_full_residual_m432n2048",
    "pi05_owner_norm_compact_residual_m432n2048",
    "pi05_xor3_compact_residual_raw_full_m432n2048",
    "pi05_owner_norm_compact_a8_m432n2048",
    "pi05_prefill_kv1_direct_cache_m432d256p432",
    "pi05_all_reduce_residual_xor3_m304n2048",
    "pi05_owner_norm_full_residual_m304n2048",
    "pi05_owner_norm_compact_residual_m304n2048",
    "pi05_xor3_compact_residual_raw_full_m304n2048",
    "pi05_owner_norm_compact_a8_m304n2048",
    "pi05_prefill_kv1_direct_cache_m304d256p304",
    "wall_action_qkv_w8_compact_partial_mrope_v2",
    "wall_action_qkv_w8_owner_partial_mrope_v2",
    "kv_owner_fanout_m20",
    "owner_direct_kv_m640",
    "wall_ring_postnorm_pair_tile40_fp16_partial_bf16_residual_local_inout_m320x2n2048",
    "wall_action_qkv_fp16_ddr_partial_mrope_v2",
    "wall_action_qkv_fp16_owner_partial_mrope_v2",
    "wall_action_qkv_fp16_compact_partial_mrope_v2",
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
    "gr00t_ff1_resident_w8a16_v2",
    "gr00t_qkv_multibase_w8a16_v2",
    "gr00t_all_reduce_residual_xor3_m41n1536",
    "hyvla_expert_o_ring_postnorm_fp16_s51_v2",
    "hyvla_expert_qk_rope_s51",
    "hyvla_expert_gateup_swiglu_w8_v2",
    "hyvla_resident_swiglu_w8_v2",
    "hyvla_expert_o_ring_postnorm_w8_s51_v2",
    "hyvla_expert_o_resident_w8_v2",
    "hyvla_nvfp4_gateup_swiglu_s51",
    "hyvla_nvfp4_gateup_resident_s51",
    "hyvla_expert_o_ring_postnorm_nvfp4_s51",
    "hyvla_expert_o_ring_postnorm_nvfp4_s51_n128",
    "hyvla_nvfp4_gateup_swiglu_input_reuse_s51",
    "hyvla_expert_down_nvfp4_s51_n128",
    "hyvla_expert_down_nextnorm_nvfp4_s51",
    "hyvla_expert_down_nextnorm_nvfp4_s51_n128",
    "hyvla_expert_qk_rope_norm_s51",
    "hyvla_expert_kv_nvfp4_s51",
    "hyvla_expert_qkv_nvfp4_s51",
    "hyvla_expert_qk_rope_norm_kv_s51",
    "hyvla_nvfp4_resident_swiglu_s51",
    "hyvla_expert_q_nvfp4_s51_n32",
    "hyvla_expert_q_nvfp4_s51_overlap",
    "hyvla_nvfp4_resident_scales_s51",
    "hyvla_nvfp4_resident_scales_swiglu_s51",
    "hyvla_expert_qkv_nvfp4_s51_overlap",
    "hyvla_expert_sdpa_m16_s51",
    "hyvla_expert_o_ring_postnorm_nvfp4_s51_n128_overlap",
    "hyvla_expert_down_nextnorm_nvfp4_s51_overlap",
    "hyvla_nvfp4_gateup_swiglu_s51_overlap",
    "hyvla_nvfp4_resident_swiglu_s51_overlap",
    "hyvla_expert_sdpa_k128_s51",
    "hyvla_vlm_vision_gu176_w8",
    "hyvla_expert_kv_fp16_s51_v2",
    "hyvla_expert_kv_w8_s51_v2",
    "hyvit2_vision_owner_reduce_layernorm_m392n1152",
    "hyvit2_vision_xor3_compact_residual_raw_full_m392n1152",
    "wall_base_nvfp4_qkv_bias_barrier_n64",
    "wall_base_nvfp4_qkv_bias_barrier_n32",
    "wall_base_nvfp4_o_ring_norm_h32",
    "wall_base_nvfp4_down_ring_norm_h32",
    "wall_base_partial_mrope_qk_h32",
    "wall_base_nvfp4_gate_up_parallel_h32",
    "wall_base_nvfp4_kv_parallel_bias_h32",
    "wall_base_nvfp4_qkv_parallel_bias_h32",
    "wall_base_nvfp4_qkv_bias_rope_h32",
    "wall_base_nvfp4_gate_up_parallel_silu_h32",
    "wall_base_nvfp4_gate_up_shared_silu_h32",
    "wall_base_vision_owner_ring_norm_m672_h1280",
    "wall_base_vision_down_owner_ring_norm_m672_h1280",
    "wall_base_vision_qk_rope_2d_spm_m672_m324",
    "wall_base_vision_ragged_sdpa_b1_s324_s336_h16d80",
    "wall_base_prefill_o_owner_ring_norm_m224_h2048",
    "g05_linear_fp16_acc32_spm_weight_m304n80k128_v2",
    "g05_kv_owner_fp16_acc32_m32n256k1024_v2",
    "g05_kv_owner_fanout_m32d256",
    "g05_vision_owner_reduce_layernorm_m256n1024",
    "g05_vision_xor3_compact_residual_raw_full_m256n1024",
    "g05_q_gate_fp16_acc32_m32n256k1024_v2",
    "g05_gate_up_fp16_acc32_m32n512k1024_v2",
    "g05_silu_round_mul_m32n512",
    "g05_vision_qkv_fp16_acc32_m256n128k1024_v2",
    "g05_text_gate_up_fp16_acc32_m256n768k2048_v2",
    "g05_gateup_resident_w8a16_acc32_m240n96k128_v2",
    "g05_kv_owner_w8a16_acc32_m32n256k1024_v2",
    "g05_q_gate_w8a16_acc32_m32n256k1024_v2",
    "g05_gate_up_w8a16_acc32_m32n512k1024_v2",
    "g05_vision_qkv_w8a16_acc32_m256n128k1024_v2",
    "g05_text_gate_up_w8a16_acc32_m256n768k2048_v2",
    "g05_qkv_gate_owner_w8a16_acc32_m32n256k1024_v2",
    "g05_k_owner_rms_norm_m32d256",
    "g05_k_owner_partial_mrope_m32d256r64",
    "g05_owner_insert_kcache_multiwarp",
    "g05_owner_insert_kcache_multiwarp_v16",
    "g05_owner_insert_vcache_multiwarp",
    "g05_owner_insert_vcache_multiwarp_v16",
    "g05_gateup_resident_w8a16_acc16_m352n96k128_v2",
    "g05_kv_owner_w8a16_acc16_m32n256k1024_v2",
    "g05_q_gate_w8a16_acc16_m32n256k1024_v2",
    "g05_gate_up_w8a16_acc16_m32n512k1024_v2",
    "g05_vision_qkv_w8a16_acc16_m256n128k1024_v2",
    "g05_text_gate_up_w8a16_acc16_m256n768k2048_v2",
    "g05_qkv_gate_owner_w8a16_acc16_m32n256k1024_v2",
    "g05_sub_gated_residual_h1024",
    "g05_rtc_adarms_norm_shift_m32h1024",
    "g05_o_resident_w8a16_acc16_m32n1024k256_v2",
    "g05_rtc_sub_gated_residual_m32h1024",
    "g05_ring_rtc_sub_gated_residual_m32h1024",
    "g05_down_partial672_w8a16_acc16_m32n1024k512_v2",
    "g05_vision_fc2_weight_outer_w8a16_acc16_m768n1024k512_v2",
    "g05_gate_up_paired_nvfp4_acc16_m32n512k1024_n128",
    "g05_o_resident_nvfp4_acc16_m32n1024k256_n80",
    "g05_down_resident_nvfp4_acc16_m32n1024k512_n64",
    "g05_gate_up_swiglu_nvfp4_acc16_m32n512k1024_n128",
    "g05_gate_kv_owner_w8a16_acc16_m32n256k1024_n128_v2",
    "g05_q_norm_partial_rope_m32d256r64_group4",
    "g05_k_owner_norm_partial_rope_m32d256r64_group4",
    "g05_text_ring_owner_rms_m256h2048",
    "g05_owner_insert_kv_cache_m32d256_p0",
    "g05_owner_insert_kv_cache_m32d256_p245",
    "g05_attn_sigmoid_round_mul_m32d256",
    "g05_ring_rtc_then_norm_shift_m32h1024",
    "g05_gate_up_fp16_acc16_m32n512k1024_v2",
    "g05_text_gate_up_fp16_acc16_m256n768k2048_v2",
    "g05_gate_up_paired_nvfp4_acc32_m32n512k1024_n128",
    "g05_o_resident_nvfp4_acc32_m32n1024k256_n80",
    "g05_down_resident_nvfp4_acc32_m32n1024k512_n64",
    "g05_gate_up_swiglu_nvfp4_acc32_m32n512k1024_n128",
    "hyvla_expert_rmsnorm_v16_bitcopy",
    "grouped_parallel_linear_acc16_expert_y4",
    "lingbot2_route_compact_e32_t51_tp64",
    "topk_norm_scalar_fp16",
    "lingbot2_routed_col_fp16_m64n64k512_v2",
    "lingbot2_routed_col_w8_m64n64k512_v2",
    "lingbot2_topk_dense_route_safe",
    "llama_rms_norm_nogamma",
    "llama_rms_norm_nogamma_v16",
    "llama_rms_norm_nogamma_v32",
    "lingbot2_combine_t51_r204_h768_scan",
    "lingbot2_topk_e32_k4_lane2_safe",
    "lingbot2_swiglu_route_r204_d64",
    "lingbot2_combine_t51_r204_h768_inverse",
    "lingbot2_route_compact_inverse_e32_t51_tp64",
    "grouped_parallel_linear_w8a16_acc16_expert_y4",
    "lingbot2_combine_shared_t51_r204_h768_inverse",
    "lingbot2_grouped_gateup_routed_acc16_expert_y4",
    "lingbot2_grouped_gateup_routed_w8a16_acc16_expert_y4",
    "lingbot2_grouped_gateup_routed_fp16_swiglu_route",
    "lingbot2_grouped_gateup_routed_w8_swiglu_route",
    "lingbot2_grouped_down_w8a16_acc16_scale_shuffle",
    "lingbot2_insert_kv_t51",
    "lingbot2_shared_silu_t51_d96_vector",
    "lingbot2_qkv_multibase_fp16_t51_v2",
    "lingbot2_qkv_multibase_w8_t51_v2",
    "lingbot2_sdpa_t51_k275_tail",
    "lingbot2_vision_fc1_gelu_fp16_v2",
    "lingbot2_vision_fc1_gelu_w8_v2",
    "lingbot2_prefill_qkv_multibase_fp16_m224_v2",
    "lingbot2_prefill_qkv_multibase_w8_m224_v2",
    "lingbot2_vision_sdpa_spm_b3s256h16d64_32b",
    "lingbot2_grouped_down_fp16_ordered_combine",
    "lingbot2_grouped_down_w8_ordered_combine",
    "lingbot2_topk_lane2_compact_inverse_e32_t51_tp64",
    "lingbot2_adarms_norm_shift_h768_diagonal",
    "lingbot2_vision_qkv_joint_fp16_m768_v2",
    "lingbot2_vision_qkv_joint_w8_m768_v2",
    "lingbot2_qk_rope_kv_t51",
    "lingbot2_prefill_gateup_shared_a_w8_m224_v2",
    "lingbot2_router_sigmoid_pad_t51_h768_e32",
    "lingbot2_shared_gateup_resident_silu_w8_t51_h768_i96_v2",
    "lingbot2_shared_gateup_resident_silu_fp16_t51_h768_i96_v2",
    "lingbot2_ring_adarms_norm_shift_t51_h768",
    "lingbot2_prefill_qk_norm_rope_m224",
    "hyvla_expert_gateup_swiglu_fp16_v2",
    "hyvla_vlm_gateup_swiglu_fp16_v2",
    "fused_geglu_w16a16_acc16_m80n96k512",
    "fused_geglu_w16a16_acc32_m64n96k512",
    "fused_geglu_w4a16_pgrp_acc16_m464n64k128",
    "fused_geglu_w4a16_pgrp_acc16_m736n64k64",
    "fused_geglu_w4a16_pgrp_acc32_m352n64k128",
    "fused_geglu_w4a16_pgrp_acc32_m496n64k64",
    "fused_geglu_w8a16_acc16_m64n80k512",
    "fused_geglu_w8a16_acc32_m96n80k384",
    "hyvla_gateup_resident_pair_fp16_v2",
    "hyvla_gateup_resident_pair_w8_v2",
    "hyvla_resident_swiglu_fp16_v2",
    "hyvla_vlm_gateup_swiglu_w8_v2",
#define RPU_PL_CANDIDATE(id, name, ordinal) name,
#define RPU_GROUPED_PL_CANDIDATE(id, name, ordinal) name,
#include "rpu_linear_candidates.inc"
#undef RPU_PL_CANDIDATE
#undef RPU_GROUPED_PL_CANDIDATE
#define RPU_PI_WALL_CANDIDATE(id, name, ordinal) name,
#include "rpu_pi_wall_candidates.inc"
#undef RPU_PI_WALL_CANDIDATE
    "grouped_parallel_linear_m96n80k512_device_m",
    "grouped_parallel_linear_w8a16_acc32_m48n112k384_device_m",
    "grouped_parallel_linear_acc32_m80n80k512_device_m",
    "grouped_parallel_linear_acc32_m80n32k768_device_m",
    "grouped_parallel_linear_w8a16_m48n48k768_device_m",
    "grouped_parallel_linear_w8a16_acc32_m96n48k512_device_m",
    "kvsort_dir_v1",
#define GIL(indexed, dtype, id, symbol, m, n, k) symbol,
#include "rpu_grouped_indexed_linear_tiles.inc"
#undef GIL
    "pi_kv_owner_w8a16_acc16_m64n32k128_db",
    "parallel_linear_wnvfp4a16_acc16_m64n64k128_db",
    "parallel_linear_wnvfp4a16_acc16_m64n64k128_db_tscale",
    "pi05_denoise_gate_up_geglu_nvfp4_acc16_m64n64k128_db",
    "pi05_denoise_gate_up_geglu_nvfp4_acc16_m64n64k128_db_tscale",
#define GIL(indexed, dtype, id, symbol, m, n, k) symbol,
#include "rpu_grouped_indexed_linear_current_tiles.inc"
#undef GIL
    "parallel_linear_wnvfp4a16_acc16_m64n64k128_db_sspm",
    "pi05_denoise_gate_up_geglu_nvfp4_acc16_m64n64k128_db_sspm",
    "pi05_denoise_q_kv1_pair_owner_m64_db_sspm",
    "pi05_all_reduce_residual_xor3_m10n1024",
    "pi05_xor3_gated_residual_m10n1024",
    "pi05_denoise_kv1_direct_cache_m10d256p16",
    "pi05_split_kv_pack_q",
    "pi05_flash_attn_split_kv_dp",
    "pi05_split_kv_merge",
    "pi05_ring_gated_residual_lb_m10n1024",
    "pi05_adarms_norm_shift_h1024_m50_warp8",
    "pi05_denoise_kv1_direct_cache_m64d256_overlap",
    "pi05_split_kv_pack_q_rope_g4d256",
    "pi05_flash_attn_split_kv_spm_dp",
    "pi05_flash_attn_split_kv_dp_rot",
    "pi05_flash_attn_split_kv_spm_dp_rot",
    "pi05_denoise_kv1_direct_cache_m64d256_rot",
    "pi05_xor3_gated_adarms_m50n1024",
    "pi05_xor3_gated_rows_m50n1024",
    "pi05_split_kv_merge_rot",
    "pi05_denoise_kv1_direct_cache_rot_q_rope_pack_m64",
    "pi05_adarms_norm_shift_h1024_row_pair",
    "pi05_split_kv_merge_rot_ov",
    "pi05_xor3_gated_adarms_twopeer_m50n1024",
    "pi05_denoise_adarms_gate_up_geglu_nvfp4_acc16_m16n64k128_db_sspm",
    "pi05_denoise_gate_up_geglu_nvfp4_acc16_m64n64k128_db_sspm_split",
    "pi05_denoise_adarms_gate_up_geglu_nvfp4_acc16_m16n64k128_db_sspm_split",
    "pi05_split_kv_merge_o_nvfp4_acc16_m64n64k128_db_sspm",
    "pi05_split_kv_merge_rot_ov_o_nvfp4_acc16_m64n64k128_db_sspm",
    "pi05_denoise_kv1_direct_cache_rot_q_rope_pack_spm_rope_m64",
    "pi05_flash_attn_split_kv_spm_dp_late_v",
    "pi05_flash_attn_split_kv_spm_dp_rot_late_v",
    "pi05_xor3_gated_adarms_onewarp_m50n1024",
    "pi05_denoise_q_kv1_pair_owner_m64_db_wspm",
    "pi05_all_reduce_residual_xor3_m480n2048",
    "pi05_owner_norm_full_residual_m480n2048",
    "pi05_owner_norm_compact_residual_m480n2048",
    "pi05_xor3_compact_residual_raw_full_m480n2048",
    "pi05_prefill_kv1_direct_cache_m480d256p480",
#define RPU_W8A8_CANDIDATE(id, name, ordinal, m, n, k, local_k, flags) name,
#include "rpu_linear_w8a8_candidates.inc"
#undef RPU_W8A8_CANDIDATE
    "dynamic_quantize_w8a8",
    "layer_norm_quantize_w8a8_h1152",
    "gelu_tanh_quantize_w8a8_h544",
    "pi05_vision_owner_layernorm_quantize_w8a8_m512n1152",
    "pi05_vision_owner_layernorm_quantize_w8a8_m768n1152",
    "rms_norm_quantize_w8a8_h5120_m128",
    "rms_norm_quantize_w8a8_h5120_m192",
    "rms_norm_quantize_w8a8_h5120_m256",
    "mul_quantize_w8a8_h3200_m128",
    "mul_quantize_w8a8_h3200_m192",
    "qwen3_up_mul_w8a16_m320n112k128",
    "qwen3_up_mul_w8a8_m272n64k384",
    "qwen3_up_mul_w8a8_m512n64k128",
};
static_assert(sizeof(KERNEL_ID_NAMES) / sizeof(KERNEL_ID_NAMES[0]) == static_cast<size_t>(KernelId::_COUNT),
              "KERNEL_ID_NAMES size must match KernelId::_COUNT");

// Tile identity is generated from the same catalog as the enum/name table.
inline KernelId parallel_linear_kernel_id(rpu_pl_tiling::WeightType type,
                                          bool acc32,
                                          const rpu_pl_tiling::TilePick& tile) {
    return static_cast<KernelId>(static_cast<size_t>(KernelId::PL_AUTOTILE_BEGIN) +
        rpu_pl_tiling::parallel_kernel_ordinal(type, acc32, tile));
}

inline KernelId grouped_parallel_linear_kernel_id(rpu_pl_tiling::WeightType type,
                                                  bool acc32,
                                                  const rpu_pl_tiling::TilePick& tile) {
    return static_cast<KernelId>(static_cast<size_t>(KernelId::GROUPED_PL_AUTOTILE_BEGIN) +
        rpu_pl_tiling::grouped_kernel_ordinal(type, acc32, tile));
}

inline KernelId pi_wall_kernel_id(std::size_t ordinal) {
    TORCH_CHECK(ordinal < rpu_pi_wall::kCandidateCount,
                "Pi/Wall Linear candidate ordinal is out of range");
    return static_cast<KernelId>(static_cast<std::size_t>(KernelId::PI_WALL_BEGIN) + ordinal);
}

struct CachedKernel {
    std::unique_ptr<::rhino_lkn::Program_t> program;
    std::unique_ptr<::rhino_lkn::Kernel_t> kernel;
    uint32_t graph_snapshot_contracts = 0;
};

class KernelCache {
public:
    static KernelCache& instance() {
        static KernelCache inst;
        return inst;
    }

    // 初始化：加载所有需要的 Program 和 Kernel
    void initialize();
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

    // Why a declared kernel may legitimately be ABSENT from its designated ref.
    // Absence is not itself a defect, but these reasons are not interchangeable
    // and the loader used to treat them the same way: skip. The mixed soft-load
    // set skipped without printing anything at all, so a fused route gated on an
    // absent payload degraded to its unfused fallback with correct results and
    // real numbers, and nothing in the process said the fast path never fired.
    enum class KernelIntent {
        kRequired,  // absent from its ref ⇒ fail-fast (unchanged)
        kOptional,  // the degraded route is a supported configuration
        kPending,   // declared here, not published upstream yet
        kRejected,  // evaluated upstream and turned down; PRESENCE is the defect
    };
    struct KernelResolution {
        std::string name;
        std::string role;  // "main" | "expansion" | "mixed"
        KernelIntent intent;
        bool resolved;
    };
    // Per-declaration resolution against the active refs, in declaration order.
    // Lets a caller assert which gated routes actually have their payload
    // instead of inferring capability from whole-file op-lib identity.
    std::vector<KernelResolution> kernel_resolution() const;

    // Cold Graph admission from successfully loaded, audited immutable payloads.
    uint32_t graph_snapshot_program_contracts(const std::string& kernel_name) const;

    // 清空所有缓存的 kernel/program (用于 shutdown)
    void clear() {
        // Revoke capability discovery before destroying any pointer it was
        // derived from. Concurrent readers observe either the prior immutable
        // publication or an uninitialized-runtime rejection, never raw cache
        // storage participating in teardown.
        std::atomic_store_explicit(
            &graph_snapshot_contracts_,
            std::shared_ptr<const std::unordered_map<std::string, uint32_t>>{},
            std::memory_order_release);
        payload_availability_.withdraw();
        kernels_.clear();  // unique_ptr 会自动释放 Program_t/Kernel_t
        {
            std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
            loaded_oplib_paths_.clear();
            oplib_symbols_.clear();
            oplib_sha256_.clear();
            kernel_resolution_.clear();
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
    std::shared_ptr<const std::unordered_map<std::string, uint32_t>>
        graph_snapshot_contracts_snapshot() const;
    CachedKernel load_kernel_from_oplib(
        const std::string& path, const std::string& kernel_name);
    std::shared_ptr<OplibSnapshot> oplib_snapshot(const std::string& path);
    const std::unordered_set<std::string>& kernel_manifest_names(const std::string& path);

    bool initialized_;
    KernelPayloadAvailability payload_availability_;
    std::shared_ptr<const std::unordered_map<std::string, uint32_t>>
        graph_snapshot_contracts_;
    std::unordered_map<std::string, CachedKernel> kernels_;
    mutable std::mutex loaded_oplib_paths_mutex_;
    std::set<std::string> loaded_oplib_paths_;
    std::unordered_map<std::string, std::shared_ptr<OplibSnapshot>> oplib_snapshots_;
    std::unordered_map<std::string, std::string> oplib_sha256_;
    std::unordered_map<std::string, std::unordered_set<std::string>> oplib_symbols_;
    std::vector<KernelResolution> kernel_resolution_;
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

    // Failed preparation may leave pending work without a completed batch.
    // Destroy only the queue owned by this caller before another graph reuses it.
    void evict(int64_t core_num, ::rhino_lkn::Queue_t* expected) noexcept {
        if (core_num < 1 || core_num > 8) return;
        auto& queue = queues_[static_cast<size_t>(core_num - 1)];
        if (queue != expected) return;
        delete queue;
        queue = nullptr;
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
// Turns two silent-corruption modes into loud failures:
//
//  1. **Dropped return code.** add_dma_kernel / add_dma_kernel_mutable return a
//     status (0 = success, see rhino_launch_queue.h). On a non-zero status the
//     SDK does not append the transfer, but build_batch() still succeeds — so a
//     missing DMA is indistinguishable from a healthy graph and surfaces only as
//     a wrong-but-deterministic numeric result somewhere downstream.
//
//  2. **Zero device address.** rhino_lkn::RpuGetDevAddr returns 0 on a lookup
//     miss (r1_buffer.cc:576-578) and none of its ~169 call sites guard it, so a
//     miss bakes src/dst = 0 into the descriptor and the DMA reads from / writes
//     to physical address 0. Every legitimate endpoint here is either an SPM
//     address (>= SPM_START = 0x40000000, r1_utils.h:16) or a HostDDR_t device
//     address, so 0 is never valid and is checked rather than propagated.
//     Same family as the SDK queue contract — an unvalidated device address.
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

// Same dropped-return-code treatment for the kernel sibling (failure mode 1
// above). On the pinned SDK Queue_t::add_kernel_mutable unconditionally returns 0
// (r1_queue.cc:695), so this is future-proofing, not a live bug fix: if a later
// SDK grows a rejection path, a dropped kernel would otherwise leave build_batch()
// succeeding on a graph that is silently missing one compute step.
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
