// Kernel/program discovery and process-wide cache implementation fragment.
#include "rpu_ops.h"
#include "rpu_kernel_manifest.h"
#include "rpu_linear_tiling.h"

#include <dlfcn.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// KernelCache 实现 - 预加载所有 kernel 以减少运行时开销
// =============================================================================

// The runtime consumes one combined operator asset. Deploy it next to this
// shared object or point RPU_KERNEL_LIB_PATH at it explicitly.
static std::string get_kernel_lib_path() {
    const char* env = std::getenv("RPU_KERNEL_LIB_PATH");
    if (env && *env) return std::string(env);

    constexpr const char* REF_NAME = "rhinoOpLib_rhinoforge_v1.1.0.ref";

    auto try_dir = [](const std::string& dir) -> std::string {
        std::string p = dir + "/" + REF_NAME;
        return access(p.c_str(), F_OK) == 0 ? p : std::string{};
    };

    Dl_info info;
    if (dladdr((void*)get_kernel_lib_path, &info) && info.dli_fname) {
        std::string so_dir(info.dli_fname);
        auto pos = so_dir.rfind('/');
        if (pos != std::string::npos) {
            so_dir = so_dir.substr(0, pos);
        }
        // Co-located next to the .so (production install layout).
        if (auto p = try_dir(so_dir); !p.empty()) return p;
        // Dev tree: build/rpu_backend/*.so → ../../src/<versioned asset>.
        if (auto p = try_dir(so_dir + "/../../src"); !p.empty()) return p;
    }

    // Last-ditch fallback: project-relative path from cwd. Caller (KernelCache)
    // will fail loudly with a clear "cannot open ..." if neither dladdr nor
    // cwd resolves to a real file — better than a hardcoded absolute path
    // tied to one developer's machine.
    return std::string("src/") + REF_NAME;
}

static const std::string& KERNEL_LIB_PATH() {
    static const std::string path = get_kernel_lib_path();
    return path;
}

#define RHINO_OP_LIB_PATH KERNEL_LIB_PATH().c_str()
#define GEMM_LIB_PATH     KERNEL_LIB_PATH().c_str()
#define SOFTMAX_LIB_PATH  KERNEL_LIB_PATH().c_str()

// 需要预加载的 kernel 列表
static const std::vector<std::pair<std::string, std::string>> KERNEL_LIST = {
    // Optional fused kernels supplied by the matching runtime asset.
    {"partial_mrope_qk_fused", RHINO_OP_LIB_PATH},
    {"llama_insert_kv_cache_multiwarp_v16", RHINO_OP_LIB_PATH},
    {"llama_insert_kv_cache_multiwarp_decode", RHINO_OP_LIB_PATH},
    {"llama_rms_norm_multiwarp_n_v64", RHINO_OP_LIB_PATH},
    {"parallel_linear_w8a16_m448n64k128_qkv_planar", RHINO_OP_LIB_PATH},
    {"llama_silu_mul_exact_strided", RHINO_OP_LIB_PATH},
    // Qwen3.5 decode fused kernels shipped by the current main op-lib ref.
    {"qwen3_5_rms_norm_gated", RHINO_OP_LIB_PATH},
    {"qwen3_5_rank1_fma", RHINO_OP_LIB_PATH},
    {"qwen3_5_mul_reduce_rows", RHINO_OP_LIB_PATH},
    {"llama_gemv", RHINO_OP_LIB_PATH},
    {"llama_gemv_wint8", RHINO_OP_LIB_PATH},
    {"qwen3vl_gateup_swiglu_gemv", RHINO_OP_LIB_PATH},
    {"qwen3vl_qk_norm_mrope_d128", RHINO_OP_LIB_PATH},
    {"qwen3vl_qk_norm_mrope_kv_insert_d128", RHINO_OP_LIB_PATH},
    {"qwen3vl_o_ring_norm_m1_h2560_w8", RHINO_OP_LIB_PATH},
    // rhinoOpLib kernels
    {"parallel_linear_wnvfp4a16_acc32_m176n128k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc32_m208n112k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc32_m240n96k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc32_m272n80k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc32_m320n64k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc32_m384n48k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc32_m480n32k128", RHINO_OP_LIB_PATH},
    {"binary_sameshape", RHINO_OP_LIB_PATH},
    {"binary_sameshape_ddr", RHINO_OP_LIB_PATH},
    {"binary_scalar", RHINO_OP_LIB_PATH},
    {"binary_scalar_ddr", RHINO_OP_LIB_PATH},
    {"binary_scalar_power", RHINO_OP_LIB_PATH},
    {"binary_scalar_power_ddr", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC256_batch", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC256_batch_ddr", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC256_batch_bopa", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC256_batch_bopa_ddr", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC_v16_batch", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC_v16_batch_ddr", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC_v16_batch_bopa", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC_v16_batch_bopa_ddr", RHINO_OP_LIB_PATH},
    // V2 tile-based Nx1_NxC (ConvertEltwiseBinary_Nx1_NxC_V2): runtime routes c%16!=0 || c>1024
    // || n>8192 here (e.g. GDN recurrent state *= exp(g_last), c=Dk*Dv=16384).
    {"binary_Nx1_NxC_tileN", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC_tileC", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC_tileN_bopa", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC_tileC_bopa", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_v256_batch", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_v256_batch_ddr", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_v256_batch_bopa", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_v256_batch_bopa_ddr", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_v16_batch", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_v16_batch_ddr", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_v16_batch_bopa", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_v16_batch_bopa_ddr", RHINO_OP_LIB_PATH},
    // V2 tile-based 1xC_NxC (ConvertEltwiseBinary_1xC_NxC_V2): the runtime routes
    // c%16!=0 || c>1024 || n>8192 here (e.g. GDN prep dt_bias/neg_exp_A, c=vg_c=4).
    {"binary_1xC_NxC_tileN", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_tileC", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_tileN_bopa", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_tileC_bopa", RHINO_OP_LIB_PATH},
    {"unary", RHINO_OP_LIB_PATH},
    {"unary_gelu_tanh", RHINO_OP_LIB_PATH},
    {"unary_gelu_erf", RHINO_OP_LIB_PATH},
    {"unary_gelu_erf_ultra_precision", RHINO_OP_LIB_PATH},
    {"unary_softplus", RHINO_OP_LIB_PATH},
    {"unary_ddr", RHINO_OP_LIB_PATH},
    {"layer_norm", RHINO_OP_LIB_PATH},
    {"layer_norm_simple", RHINO_OP_LIB_PATH},
    {"layer_norm_bf16", RHINO_OP_LIB_PATH},
    // gemm kernels (常用尺寸) - bmm 动态生成名称，这里预加载常见的组合
    // 格式: gemm_fp16_spm_16b_wMxN_kK_1core_buf1_nt_lpaddr_{peak|univ}
    {"gemm_fp16_spm_16b_w64x64_k64_1core_buf1_nt_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w64x64_k64_1core_buf1_nt_lpaddr_univ", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w64x64_k128_1core_buf1_nt_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w64x64_k128_1core_buf1_nt_lpaddr_univ", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w128x128_k64_1core_buf1_nt_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w128x128_k64_1core_buf1_nt_lpaddr_univ", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w128x128_k128_1core_buf1_nt_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w128x128_k128_1core_buf1_nt_lpaddr_univ", GEMM_LIB_PATH},
    // GDN chunk A@B (nt_) with N=Dv/Dk=128, M=C=64: value_new/k_cumdecay (K=C=64),
    // out=kcd@S / q@S (K=dk=128), recur_out=attn@v (K=C=64).
    {"gemm_fp16_spm_16b_w64x128_k64_1core_buf1_nt_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w64x128_k64_1core_buf1_nt_lpaddr_univ", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w64x128_k128_1core_buf1_nt_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w64x128_k128_1core_buf1_nt_lpaddr_univ", GEMM_LIB_PATH},
    // GDN chunk transposing bmm: rowByRow (nn_) for attn = q@kᵀ (M=N=C=64, K=dk=128),
    // colByCol (tt_) for state += kᵀ@v (M=N=dk/dv=128, K=C=64). Kernels already in oplib.
    {"gemm_fp16_spm_16b_w64x64_k128_1core_buf1_nn_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w64x64_k128_1core_buf1_nn_lpaddr_univ", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w128x128_k64_1core_buf1_tt_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w128x128_k64_1core_buf1_tt_lpaddr_univ", GEMM_LIB_PATH},
    // softmax kernel
    {"softmax_c16_gauto", SOFTMAX_LIB_PATH},
    // flash attention kernel
    {"llm_fp16_32b_prefill_flash_attn_univ_dp", RHINO_OP_LIB_PATH},
    {"llm_fp16_16b_prefill_flash_attn_univ_dp", RHINO_OP_LIB_PATH},
    {"llm_fp16_32b_prefill_flash_attn_univ_vctxlen", RHINO_OP_LIB_PATH},  // vctxlen (dynamic context length)
    {"llm_fp16_32b_prefill_flash_attn_univ_vctxlen_minibatch", RHINO_OP_LIB_PATH},  // per-image K/V slicing (multi-image vision packing)
    {"llm_fp16_16b_prefill_flash_attn_univ_vctxlen_minibatch", RHINO_OP_LIB_PATH},
    {"llm_fp16_32b_prefill_trans_prjv_dp_vctxlen", RHINO_OP_LIB_PATH},
    {"llm_fp16_16b_prefill_flash_attn_by_mha_univ_vctxlen", RHINO_OP_LIB_PATH},
    // RoPE kernels
    {"llama_rope", RHINO_OP_LIB_PATH},
    {"all_gather_multi_core", RHINO_OP_LIB_PATH},
    {"all_gather_multi_core_little_chunk", RHINO_OP_LIB_PATH},
    {"llm_all_reduce_residual", RHINO_OP_LIB_PATH},
    {"llm_all_reduce_residual_nopace", RHINO_OP_LIB_PATH},
    // tile kernels (for expand/repeat optimization)
    {"tile_Nx1_NxC", RHINO_OP_LIB_PATH},
    {"tile_Nx1_NxC_v16", RHINO_OP_LIB_PATH},
    {"tile_general_largeC", RHINO_OP_LIB_PATH},
    {"tile_general_smallC", RHINO_OP_LIB_PATH},
    // reduce mean kernels
    {"reduce_mean_last_dim", RHINO_OP_LIB_PATH},
    {"reduce_mean_last_dim_ddr", RHINO_OP_LIB_PATH},
    // RMSNorm kernels (llama_rms_norm_fp16 dropped, llama_rms_norm_bf16 renamed
    // to llama_rms_norm in rhinoOpLib v0.1.1_38977cf7)
    {"llama_rms_norm_bf16_ddr", RHINO_OP_LIB_PATH},
    {"llama_rms_norm", RHINO_OP_LIB_PATH},  // SPM版本
    // R48b —— 多行/warp 变体（**optional**：ref 里没有就跳过，不影响启动）
    {"llama_rms_norm_v16", RHINO_OP_LIB_PATH},
    {"llama_rms_norm_v32", RHINO_OP_LIB_PATH},
    // Base Newton is registered below; all three Newton kernels are consumed
    // only by the isolated conformance probe.
    {"llama_rms_norm_newton_v16", RHINO_OP_LIB_PATH},
    {"llama_rms_norm_newton_v32", RHINO_OP_LIB_PATH},
    // R50 —— mrope 的 SPM 表孪生（optional；只被隔离探针 / 打开开关的路径取用）
    {"llama_mrope_interleave_local_spm_tbl", RHINO_OP_LIB_PATH},
    {"llama_rms_norm_newton", RHINO_OP_LIB_PATH},  // conformance-only
    {"llama_rope_ddr", RHINO_OP_LIB_PATH},
    // M-RoPE kernel (Qwen3-VL R-Phase 1)
    {"llama_mrope_interleave", RHINO_OP_LIB_PATH},
    // Partial RoPE kernels (Qwen3.5 partial rotary):
    // partial_mrope = prefill, partial_rope_1d = decode.
    {"partial_mrope", RHINO_OP_LIB_PATH},
    {"partial_rope_1d", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN decode 核心:l2norm + causal conv1d-update。
    {"l2norm", RHINO_OP_LIB_PATH},
    {"llm_fla_conv1d_w16a16_acc16", RHINO_OP_LIB_PATH},
    {"llm_fla_conv1d_noinplace_w16a16_acc16", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN prefill chunk: cumulative sum (in default ref).
    {"cumsum_reduceC_tileN", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN prefill chunk: strict-lower-tri (I - M)^-1 (in default ref).
    {"unit_tril_inv", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN gated-norm fused silu(x)*y (in default ref).
    {"llama_silu_mul", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN: sum over rows (axis 0), replaces tree-reduce (in default ref).
    {"reduce_sum_non_last_dim_out_seg", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN prefill chunk: slice/insert_slice (per-chunk PCS/PCI + conv halo) +
    // Bx1xC broadcast eltwise (decay = g_cumsum[row] - g_cumsum[col]). These are
    // lazy-loadable in passthrough but the recorded fused graph needs them preloaded.
    {"slice_last_dim_axis", RHINO_OP_LIB_PATH},
    {"slice_single_axis", RHINO_OP_LIB_PATH},
    {"binary_Bx1xC_BxNxC_tileN", RHINO_OP_LIB_PATH},
    {"binary_Bx1xC_BxNxC_tileC", RHINO_OP_LIB_PATH},
    {"binary_Bx1xC_BxNxC_tileN_bopa", RHINO_OP_LIB_PATH},
    {"binary_Bx1xC_BxNxC_tileC_bopa", RHINO_OP_LIB_PATH},
    // 2D RoPE kernel (Qwen3-VL R-Phase 3 vision encoder)
    {"rope_2d_ddr", RHINO_OP_LIB_PATH},
    {"rope_2d_spm", RHINO_OP_LIB_PATH},
    // Fast bilinear pos-embed interpolation (Qwen3-VL / Qwen3.5 vision STEP 0)
    {"fast_pos_embed_interpolate", RHINO_OP_LIB_PATH},
    // LLaMA KV-Cache kernels
    {"llama_insert_vcache_multiwarp", RHINO_OP_LIB_PATH},
    {"llama_insert_kcache_multiwarp", RHINO_OP_LIB_PATH},
    {"llama_insert_vcache_multiwarp_v16", RHINO_OP_LIB_PATH},
    {"llama_insert_kcache_multiwarp_v16", RHINO_OP_LIB_PATH},
    // DDR to SPM memcpy (multi-core)
    // SPM to DDR memcpy (multi-core)
    // Argmax/Argmin kernels
    {"argmax_reduceC_tileN", RHINO_OP_LIB_PATH},
    {"argmax_reduceC_tileN_C128", RHINO_OP_LIB_PATH},
    {"argmax_reduceC_tileN_sli", RHINO_OP_LIB_PATH},
    {"argmax_reduceC_tileN_C128_sli", RHINO_OP_LIB_PATH},
    {"argmax_reduceN_tileC", RHINO_OP_LIB_PATH},
    {"argmax_reduceN_tileC_sli", RHINO_OP_LIB_PATH},
    {"argmin_reduceC_tileN", RHINO_OP_LIB_PATH},
    {"argmin_reduceC_tileN_C128", RHINO_OP_LIB_PATH},
    {"argmin_reduceC_tileN_sli", RHINO_OP_LIB_PATH},
    {"argmin_reduceC_tileN_C128_sli", RHINO_OP_LIB_PATH},
    {"argmin_reduceN_tileC", RHINO_OP_LIB_PATH},
    {"argmin_reduceN_tileC_sli", RHINO_OP_LIB_PATH},
    // memset lives in the expansion ref when absent from the main ref.
    {"memset_spm_multi_core_v2", RHINO_OP_LIB_PATH},
    // DDR to SPM memcpy, scatter read DDR, broadcast write spm
    // GDN prefill chunk zeroing: `fill` (1D) replaces the absent memset_v2 (rhino's
    // CreateFill drives this same kernel). Used by rpu_launch_fill_spm_kernel.
    {"fill", RHINO_OP_LIB_PATH},
    // Conv2d SPM kernels (1core, tile_m=128, tile_n=128, univ)
    {"conv_fp16_f1_spm_16b_w128x128_k48_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f1_spm_16b_w128x128_k64_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f1_spm_16b_w128x128_k96_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f1_spm_16b_w128x128_k128_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_16b_w128x128_k48_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_16b_w128x128_k64_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_16b_w128x128_k96_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_16b_w128x128_k128_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_16b_w128x128_k48_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_16b_w128x128_k64_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_16b_w128x128_k96_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_16b_w128x128_k128_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    // Conv2d multi-core column kernels (tile_k=32, tile_n=32/64/128)
    {"conv_fp16_f1_spm_column_w128x32_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_column_w128x32_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_column_w128x32_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f1_spm_column_w128x64_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_column_w128x64_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_column_w128x64_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f1_spm_column_w128x128_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_column_w128x128_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_column_w128x128_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    // Conv2d multi-core column kernels (tile_k=64, tile_n=32/64/128)
    {"conv_fp16_f1_spm_column_w128x32_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_column_w128x32_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_column_w128x32_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f1_spm_column_w128x64_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_column_w128x64_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_column_w128x64_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f1_spm_column_w128x128_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_column_w128x128_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_column_w128x128_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    // Im2col kernels (multi-core, SPM→SPM, for patch embedding fusion)
    {"im2col_coren_m64xn32_buf1_sIsO", RHINO_OP_LIB_PATH},
    {"im2col_coren_m128xn32_buf1_sIsO", RHINO_OP_LIB_PATH},
    {"im2col_coren_m64xn64_buf1_sIsO", RHINO_OP_LIB_PATH},
    {"im2col_coren_m128xn64_buf1_sIsO", RHINO_OP_LIB_PATH},
    // Pad kernel (constant fill, SPM→SPM)
    {"pad_const_NxC_NxCP", RHINO_OP_LIB_PATH},
    // Transpose kernel (3D, SPM→SPM)
    {"transpose_ncb_c16", RHINO_OP_LIB_PATH},
    {"transpose_nbc_c16", RHINO_OP_LIB_PATH},
    // transpose_nbc_c16 above also serves the GDN/NextDiT perm{1,0,2} path.
    // perm{0,2,1} = order 021 for the chunk bmm rowByRow/colByCol pre-transpose.
    {"transpose_bcn_c16", RHINO_OP_LIB_PATH},
    // 4D perm{0,2,1,3} used by G0.5 compact STEP0 Graph capture.
    {"transpose_qnbc_c16", RHINO_OP_LIB_PATH},
    // SPM-gather-DDR (empirical test for paligemma2 attn_tp scatter fix).
    // DDR scatter → per-core SPM (address-tabular). Preloaded for future callers.
    {"grouped_parallel_linear_acc16", RHINO_OP_LIB_PATH},
    {"grouped_parallel_linear_w8a16_acc16", RHINO_OP_LIB_PATH},
    {"topk_scalar_fp16", RHINO_OP_LIB_PATH},
    {"topk_by_kvsort_fp16", RHINO_OP_LIB_PATH},
    {"bincount_scalar", RHINO_OP_LIB_PATH},
    {"llm_topk_prob_norm", RHINO_OP_LIB_PATH},
    {"unary_cast_uint16_int32", RHINO_OP_LIB_PATH},
    {"gather_scalar", RHINO_OP_LIB_PATH},
    {"gather", RHINO_OP_LIB_PATH},
    {"llm_gather_mul_v2", RHINO_OP_LIB_PATH},
    {"scatternd_int16_reduction", RHINO_OP_LIB_PATH},
    {"grouped_parallel_linear_acc32", RHINO_OP_LIB_PATH},
    {"grouped_parallel_linear_w8a16_acc32", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc16_m176n128k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc16_m208n112k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc16_m240n96k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc16_m272n80k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc16_m320n64k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc16_m384n48k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc16_m480n32k128", RHINO_OP_LIB_PATH},
    {"pi05_denoise_gate_up_geglu_nvfp4_acc16_m320n64k128", RHINO_OP_LIB_PATH},
    {"unary_silu_high_precision", RHINO_OP_LIB_PATH},
    {"unary_tanh_high_precision", RHINO_OP_LIB_PATH},
    {"unary_gelu_erf_high_precision", RHINO_OP_LIB_PATH},
    {"rhinovla_newton_norm_shift_h1024", RHINO_OP_LIB_PATH},
    {"rhinovla_silu_high_mul", RHINO_OP_LIB_PATH},
    {"unary_gelu_tanh_high_precision", RHINO_OP_LIB_PATH},
};

static void print_tensor_info(const at::Tensor &t, const std::string &name) {
    std::cout << "[" << name << "] "
              << "device=" << t.device()
              << ", dtype=" << t.scalar_type()
              << ", layout=" << t.layout()
              << ", sizes=[";
    for (int i = 0; i < t.dim(); i++) {
        std::cout << t.size(i);
        if (i != t.dim() - 1) std::cout << ", ";
    }
    std::cout << "], strides=[";
    for (int i = 0; i < t.dim(); i++) {
        std::cout << t.stride(i);
        if (i != t.dim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}
// Kernels in KERNEL_LIST that the backend tolerates being ABSENT from both refs
// (skipped, not fatal). Everything else is REQUIRED — absent from both refs ⇒ fail-fast, so a
// missing/misconfigured expansion ref can't silently init an empty cache (→ later nullptr).
static bool is_optional_kernel(const std::string& name) {
    static const std::unordered_set<std::string> optional = {
        "pi05_denoise_gate_up_geglu_nvfp4_acc16_m320n64k128",
        "unary_silu_high_precision",
        "unary_tanh_high_precision",
        "unary_gelu_erf_high_precision",
        "rhinovla_newton_norm_shift_h1024",
        "rhinovla_silu_high_mul",
        "unary_gelu_tanh_high_precision",

        // RMSNorm vector kernels are optional at process startup. An explicit
        // production route still fails at its launcher if V16/V32 is absent.
        "llama_rms_norm_newton",
        "llama_rms_norm_v16", "llama_rms_norm_v32",
        "llama_rms_norm_newton_v16", "llama_rms_norm_newton_v32",
        "qwen3vl_gateup_swiglu_gemv",
        "qwen3vl_qk_norm_mrope_d128",
        "qwen3vl_qk_norm_mrope_kv_insert_d128",
        "qwen3vl_o_ring_norm_m1_h2560_w8",
        // R50 —— mrope SPM 表孪生：同上，默认路径不碰。
        "llama_mrope_interleave_local_spm_tbl",
    };
    return optional.count(name) != 0;
}

// The sidecar is public metadata; operator payloads remain opaque. Cache the
// allowlist before constructing any Program, because Launch aborts on a miss.
const std::unordered_set<std::string>& KernelCache::kernel_manifest_names(
    const std::string& lib_path) {
    std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
    auto found = oplib_symbols_.find(lib_path);
    if (found != oplib_symbols_.end()) return found->second;
    return oplib_symbols_.emplace(
        lib_path, rpu_kernel_manifest::load(lib_path)).first->second;
}

static std::string canonical_payload_path(const std::string& path) {
    std::error_code error;
    const auto canonical = std::filesystem::canonical(path, error);
    TORCH_CHECK(
        !error, "cannot canonicalize runtime payload '", path,
        "': ", error.message());
    return canonical.string();
}

static const std::string& FROZEN_KERNEL_LIB_PATH() {
    static const std::string path = canonical_payload_path(KERNEL_LIB_PATH());
    return path;
}

std::shared_ptr<OplibSnapshot> KernelCache::oplib_snapshot(const std::string& path) {
    std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
    auto found = oplib_snapshots_.find(path);
    if (found != oplib_snapshots_.end()) return found->second;
    auto snapshot = std::make_shared<OplibSnapshot>(path);
    oplib_snapshots_.emplace(path, snapshot);
    return snapshot;
}

CachedKernel KernelCache::load_kernel_from_oplib(
    const std::string& path, const std::string& kernel_name) {
    // Callers pass the frozen canonical role path. Never resolve it afresh:
    // replacement/deletion must not change which snapshot a later Program uses.
    TORCH_CHECK(kernel_manifest_names(path).count(kernel_name) != 0,
                "kernel not listed in the operator manifest: ", kernel_name);
    const auto snapshot = oplib_snapshot(path);
    CachedKernel cached;
    cached.program = std::make_unique<::rhino_lkn::Program_t>();
    const int32_t rc = cached.program->create_with_binary_file(
        snapshot->path(), kernel_name.c_str());
    TORCH_CHECK(
        rc == 0, "failed to load kernel '", kernel_name, "' from '",
        path, "': rc=", rc);
    cached.kernel = std::make_unique<::rhino_lkn::Kernel_t>(
        *cached.program, kernel_name.c_str());
    {
        std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
        loaded_oplib_paths_.insert(path);
    }
    return cached;
}

std::vector<std::string> KernelCache::loaded_oplib_paths() const {
    std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
    return {loaded_oplib_paths_.begin(), loaded_oplib_paths_.end()};
}

std::vector<std::pair<std::string, std::string>> KernelCache::loaded_oplib_bytes() const {
    std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
    TORCH_CHECK(initialized_, "runtime oplib identity requires completed kernel initialization");
    std::vector<std::pair<std::string, std::string>> result;
    for (const auto& path : loaded_oplib_paths_) {
        result.emplace_back(path, oplib_snapshots_.at(path)->bytes());
    }
    return result;
}

void KernelCache::initialize() {
    if (initialized_) return;

    if (log_at(3)) {
        std::cout << "[KernelCache] Initializing kernel cache..." << std::endl;
    }
    auto start = std::chrono::high_resolution_clock::now();

    const std::string& main_path = FROZEN_KERNEL_LIB_PATH();
    const auto& available = kernel_manifest_names(main_path);

    std::vector<std::string> missing_required;
    for (const auto& [kernel_name, lib_path] : KERNEL_LIST) {
        if (available.count(kernel_name) == 0) {
            // Probe before create_with_binary_file: the SDK aborts on a missing
            // symbol, so required kernels fail through an actionable exception.
            if (is_optional_kernel(kernel_name)) {
                if (log_at(3)) {
                    std::cout << "[KernelCache] optional kernel '" << kernel_name
                              << "' absent from combined asset; skipping" << std::endl;
                }
            } else {
                missing_required.push_back(
                    kernel_name);
            }
            continue;
        }
        try {
            CachedKernel cached = load_kernel_from_oplib(
                main_path, kernel_name);
            kernels_.erase(kernel_name);
            kernels_.emplace(kernel_name, std::move(cached));
        } catch (const std::exception& e) {
            if (log_at(2)) {
                std::cerr << "[KernelCache] Warning: Failed to load kernel '"
                          << kernel_name << "': " << e.what() << std::endl;
            }
            // Probe said the ref ships this symbol but the load still threw ⇒ corrupt
            // or ABI-incompatible ref. A REQUIRED kernel must abort via the same
            // fail-fast as the absent-from-both-refs path, not leave a nullptr behind.
            if (!is_optional_kernel(kernel_name)) {
                missing_required.push_back(
                    kernel_name);
            }
        }
    }

    // Fail-fast instead of letting one ref silently satisfy the other role.
    TORCH_CHECK(missing_required.empty(),
                "[KernelCache] ", missing_required.size(),
                " required kernel(s) absent from combined operator asset '",
                main_path, "'. First missing: '",
                (missing_required.empty() ? std::string() : missing_required.front()),
                "'. Check the .ref install and RPU_KERNEL_LIB_PATH.");

    // Load all six 7-tile families from the active main
    // ref: FP16/W8/W4 x ACC16/ACC32. Missing one is an artifact mismatch, not a
    // recoverable fallback, because the corresponding base kernels are retired.
    {
        const std::string& at_lib = main_path;
        auto at_names = rpu_pl_tiling::autotile_kernel_names();
        const auto& at_symbols = kernel_manifest_names(at_lib);
        std::vector<std::string> missing_at;
        for (const auto& kname : at_names) {
            if (!at_symbols.count(kname)) missing_at.push_back(kname);
        }
        // Program_t may terminate the process instead of throwing when a symbol
        // is absent. Inspect the ref first so an old/wrong main artifact fails
        // through a normal, actionable TORCH_CHECK.
        TORCH_CHECK(
            missing_at.empty(),
            "[KernelCache] autotile ref mismatch: ", missing_at.size(), "/",
            at_names.size(), " required tiled linear kernels are absent from '",
            at_lib, "'. First missing: '",
            (missing_at.empty() ? std::string() : missing_at.front()),
            "'. Install the op ref matching the selected runtime profile.");
        int at_loaded = 0;
        for (const auto& kname : at_names) {
            try {
                CachedKernel cached = load_kernel_from_oplib(at_lib, kname);
                // Some tiled names are also part of the fixed preload list.
                // Destroy a replaced Kernel before its owning Program.
                kernels_.erase(kname);
                kernels_.emplace(kname, std::move(cached));
                ++at_loaded;
            } catch (const std::exception& e) {
                if (log_at(2)) {
                    std::cerr << "[KernelCache] autotile: failed to load '" << kname
                              << "' from " << at_lib << ": " << e.what() << std::endl;
                }
            }
        }
        TORCH_CHECK(
            at_loaded == static_cast<int>(at_names.size()),
            "[KernelCache] autotile ref mismatch: loaded ", at_loaded, "/",
            at_names.size(), " required tiled linear kernels from '", at_lib,
            "'. Install the op ref matching the selected runtime profile.");
        if (log_at(3)) {
            std::cout << "[KernelCache] autotile: loaded " << at_loaded
                      << "/" << at_names.size() << " tiled linear variants from "
                      << at_lib << std::endl;
        }
    }

    // Preload optional fused and compact-residual operators from the public asset.
    // Missing symbols are tolerated so unrelated installs remain unchanged;
    // an exact-profile caller fails at lookup with a targeted error.
    {
        const std::string& mixed_lib = main_path;
        int mixed_loaded = 0;
        static constexpr const char* kMixedNames[] = {
            "llm_all_reduce_residual_local_spm",
            "fused_gate_up_swiglu_w8a16_m320n112k128",
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
            "w8a16_row_weight_reuse_m352n80k256",
        };
        if (access(mixed_lib.c_str(), F_OK) == 0) {
            const auto& mixed_symbols = kernel_manifest_names(mixed_lib);
            for (const char* name : kMixedNames) {
                const std::string kname(name);
                if (mixed_symbols.count(kname) == 0) continue;
                try {
                    CachedKernel cached = load_kernel_from_oplib(
                        mixed_lib, kname);
                    kernels_.erase(kname);
                    kernels_.emplace(kname, std::move(cached));
                    ++mixed_loaded;
                } catch (const std::exception& e) {
                    if (log_at(2)) {
                        std::cerr << "[KernelCache] mixed: failed to load '"
                                  << kname << "' from " << mixed_lib << ": "
                                  << e.what() << std::endl;
                    }
                }
            }
        }
        if (log_at(3) && mixed_loaded != 0) {
            std::cout << "[KernelCache] mixed: loaded " << mixed_loaded
                      << "/" << sizeof(kMixedNames) / sizeof(kMixedNames[0])
                      << " kernels from "
                      << mixed_lib << std::endl;
        }
    }

    {
        const std::set<std::string> expected_paths = {
            main_path};
        std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
        TORCH_CHECK(
            loaded_oplib_paths_ == expected_paths,
            "[KernelCache] loaded op-lib receipt does not match the frozen "
            "combined payload");
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    if (log_at(3)) {
        std::cout << "[KernelCache] Loaded " << kernels_.size() << " kernels in "
                  << elapsed << "ms" << std::endl;
    }

    // 填充快速缓存数组 (O(1) 访问)
    populate_fast_cache();

    // Publish only after the complete pointer table is ready. Capability
    // readers acquire this immutable summary instead of racing the raw cache.
    // Commit initialized_ after publication succeeds: constructing the immutable
    // snapshot may throw, and a failed initialization must remain retryable.
    KernelPayloadAvailability::ExtendedPayloads pi05_extended_payloads;
    const size_t extended_begin = static_cast<size_t>(
        KernelId::PI05_DENOISE_KV1_PAIR_OWNER_FP16_M50N32X2K1024);
    for (size_t i = extended_begin; i < static_cast<size_t>(KernelId::_COUNT); ++i) {
        if (fast_cache_[i]) pi05_extended_payloads.set(i - extended_begin);
    }
    payload_availability_.publish(
        fast_cache_[static_cast<size_t>(KernelId::RMS_NORM_SPM_V16)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::RMS_NORM_SPM_V32)] != nullptr,
        fast_cache_[static_cast<size_t>(
            KernelId::W8A16_ROW_WEIGHT_REUSE_M352N80K256)] != nullptr,
        fast_cache_[static_cast<size_t>(
            KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152)] != nullptr,
        fast_cache_[static_cast<size_t>(
            KernelId::PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152)] != nullptr,
        fast_cache_[static_cast<size_t>(
            KernelId::PI05_PREFILL_GATE_UP_GEGLU_ONLINE_W8A8_P544_C272)] != nullptr,
        pi05_extended_payloads,
        fast_cache_[static_cast<size_t>(KernelId::QWEN3VL_GATEUP_SWIGLU_GEMV)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::QWEN3VL_QK_NORM_MROPE_D128)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::QWEN3VL_QK_NORM_MROPE_KV_INSERT_D128)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::QWEN3VL_O_RING_NORM_M1_H2560_W8)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::RMS_NORM_NEWTON_SPM)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::RMS_NORM_SPM_NEWTON_V16)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::RMS_NORM_SPM_NEWTON_V32)] != nullptr);
    initialized_ = true;
}

void KernelCache::require_names(const std::vector<std::string>& names) {
    initialize();
    const auto& available = kernel_manifest_names(FROZEN_KERNEL_LIB_PATH());
    for (const auto& name : names) {
        TORCH_CHECK(!name.empty() && available.count(name) != 0,
                    "Selected model profile requires operator '", name,
                    "'; install the matching RhinoForge operator asset before loading weights");
        const auto loaded = kernels_.find(name);
        TORCH_CHECK(loaded != kernels_.end() && loaded->second.kernel != nullptr,
                    "Selected model profile operator '", name,
                    "' was listed but could not be loaded; install a valid matching "
                    "RhinoForge operator asset before loading weights");
    }
}

void KernelCache::populate_fast_cache() {
    // 根据 KernelId 枚举和 KERNEL_ID_NAMES 填充 fast_cache_ 数组
    for (size_t i = 0; i < static_cast<size_t>(KernelId::_COUNT); ++i) {
        const char* name = KERNEL_ID_NAMES[i];
        auto it = kernels_.find(name);
        if (it != kernels_.end()) {
            fast_cache_[i] = it->second.kernel.get();
        } else {
            fast_cache_[i] = nullptr;
            const bool optional_id = is_optional_kernel(name) ||
                (i >= static_cast<size_t>(KernelId::W8A16_ROW_WEIGHT_REUSE_M352N80K256)) ||
                i == static_cast<size_t>(KernelId::FUSED_GATE_UP_SWIGLU_W8A16_M320N112) ||
                i == static_cast<size_t>(KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_LOCAL_SPM);

            // 这些是热点 kernel，如果找不到则警告
            if (log_at(2) && !optional_id) {
                std::cerr << "[KernelCache] Warning: Fast cache kernel '"
                          << name << "' not found" << std::endl;
            }
        }
    }
    if (log_at(3)) {
        std::cout << "[KernelCache] Populated fast cache with "
                  << static_cast<size_t>(KernelId::_COUNT) << " kernel pointers" << std::endl;
    }
}

::rhino_lkn::Kernel_t* KernelCache::get_kernel(const std::string& kernel_name) {
    auto it = kernels_.find(kernel_name);
    if (it != kernels_.end()) {
        return it->second.kernel.get();
    }

    // 如果没找到，尝试动态加载 (fallback)
    if (log_at(2)) {
        std::cerr << "[KernelCache] Warning: Kernel '" << kernel_name
                  << "' not in cache, loading dynamically..." << std::endl;
    }

    const std::string& main_path = FROZEN_KERNEL_LIB_PATH();
    const std::string* selected_path =
        kernel_manifest_names(main_path).count(kernel_name) ? &main_path : nullptr;
    if (selected_path != nullptr) {
        try {
            CachedKernel cached = load_kernel_from_oplib(
                *selected_path, kernel_name);
            auto* ptr = cached.kernel.get();
            kernels_.erase(kernel_name);
            kernels_.emplace(kernel_name, std::move(cached));
            return ptr;
        } catch (...) {
        }
    }

    return nullptr;
}

::rhino_lkn::Program_t* KernelCache::get_program(const std::string& kernel_name) {
    auto it = kernels_.find(kernel_name);
    if (it != kernels_.end()) {
        return it->second.program.get();
    }
    return nullptr;
}
