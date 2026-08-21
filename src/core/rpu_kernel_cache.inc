// Kernel/program discovery and process-wide cache implementation fragment.
#include "rpu_ops.h"
#include "rpu_linear_tiling.h"

#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// KernelCache 实现 - 预加载所有 kernel 以减少运行时开销
// =============================================================================

// The runtime consumes one combined op library. Deploy it next to this shared
// object or point RPU_KERNEL_LIB_PATH at it explicitly.
static std::string get_kernel_lib_path() {
    const char* env = std::getenv("RPU_KERNEL_LIB_PATH");
    if (env && *env) return std::string(env);

    constexpr const char* REF_NAME = "rhinoOpLib_current.ref";
    Dl_info info{};
    TORCH_CHECK(
        dladdr(reinterpret_cast<void*>(get_kernel_lib_path), &info) != 0 &&
            info.dli_fname != nullptr && *info.dli_fname != '\0',
        "cannot resolve the RhinoForge extension path; set RPU_KERNEL_LIB_PATH");
    return (std::filesystem::path(info.dli_fname).parent_path() / REF_NAME).string();
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
    // rhinoOpLib kernels
    {"parallel_linear_wINT4a16_per_channel_quant", RHINO_OP_LIB_PATH},   // Pi0.5 W4
    {"parallel_linear_wNVFP4a16_acc16", RHINO_OP_LIB_PATH},
    {"parallel_linear_wNVFP4a16", RHINO_OP_LIB_PATH},
    {"llama_gemv_wint8", RHINO_OP_LIB_PATH},
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
    // Tile-based Nx1_NxC: runtime routes c%16!=0 || c>1024 || n>8192
    // here (e.g. GDN recurrent state *= exp(g_last), c=Dk*Dv=16384).
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
    // Tile-based 1xC_NxC: the runtime routes c%16!=0 || c>1024 || n>8192
    // here (e.g. GDN prep dt_bias/neg_exp_A, c=vg_c=4).
    {"binary_1xC_NxC_tileN", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_tileC", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_tileN_bopa", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_tileC_bopa", RHINO_OP_LIB_PATH},
    {"unary", RHINO_OP_LIB_PATH},
    {"unary_gelu_tanh", RHINO_OP_LIB_PATH},
    {"unary_gelu_erf", RHINO_OP_LIB_PATH},
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
    {"llama_reduce_sum", RHINO_OP_LIB_PATH},
    {"llm_reduce_sum_scatter_residual_v2", RHINO_OP_LIB_PATH},
    {"llm_reduce_sum_scatter_residual", RHINO_OP_LIB_PATH},
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
    // RMSNorm kernels.
    {"llama_rms_norm_bf16_ddr", RHINO_OP_LIB_PATH},
    {"llama_rms_norm", RHINO_OP_LIB_PATH},  // SPM版本
    // Optional multi-row/warp variants; a missing symbol does not block startup.
    {"llama_rms_norm_v16", RHINO_OP_LIB_PATH},
    {"llama_rms_norm_v32", RHINO_OP_LIB_PATH},
    // `llama_rms_norm_newton` is registered separately for the production
    // RPU_RMSNORM_NEWTON path; only its two warp variants are added here.
    {"llama_rms_norm_newton_v16", RHINO_OP_LIB_PATH},
    {"llama_rms_norm_newton_v32", RHINO_OP_LIB_PATH},
    // Optional M-RoPE variant with SPM-resident tables.
    {"llama_mrope_interleave_local_spm_tbl", RHINO_OP_LIB_PATH},
    {"llama_rms_norm_newton", RHINO_OP_LIB_PATH},  // Newton-refined rsqrt (RPU_RMSNORM_NEWTON)
    {"llama_rope_ddr", RHINO_OP_LIB_PATH},
    // M-RoPE kernel (Qwen3-VL)
    {"llama_mrope_interleave", RHINO_OP_LIB_PATH},
    // Partial RoPE kernels (Qwen3.5 partial rotary):
    // partial_mrope = prefill, partial_rope_1d = decode.
    {"partial_mrope", RHINO_OP_LIB_PATH},
    {"partial_rope_1d", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN decode: l2norm + causal conv1d update.
    {"l2norm", RHINO_OP_LIB_PATH},
    {"llm_fla_conv1d_w16a16_acc16", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN prefill chunk: cumulative sum (combined operator asset).
    {"cumsum_reduceC_tileN", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN prefill chunk: strict-lower-tri (I - M)^-1 (combined operator asset).
    {"unit_tril_inv", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN gated-norm fused silu(x)*y (combined operator asset).
    {"llama_silu_mul", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN: sum over rows (axis 0), replaces tree-reduce (combined operator asset).
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
    // 2D RoPE kernels used by the vision encoder.
    {"rope_2d_ddr", RHINO_OP_LIB_PATH},
    {"rope_2d_spm", RHINO_OP_LIB_PATH},
    // LLaMA KV-Cache kernels
    {"llama_insert_vcache_multiwarp", RHINO_OP_LIB_PATH},
    {"llama_insert_kcache_multiwarp", RHINO_OP_LIB_PATH},
    {"llama_insert_vcache_multiwarp_v16", RHINO_OP_LIB_PATH},
    {"llama_insert_kcache_multiwarp_v16", RHINO_OP_LIB_PATH},
    // GEMV kernel (for Decode phase M=1 optimization)
    {"llama_gemv", RHINO_OP_LIB_PATH},
    // DDR to SPM memcpy (multi-core)
    // SPM to DDR memcpy (multi-core)
    // DDR to SPM memcpy v2 (keeps each block within the signed 16-bit limit)
    // SPM to DDR memcpy v2 (keeps each block within the signed 16-bit limit)
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
    {"memset_spm_multi_core_v2", RHINO_OP_LIB_PATH},
    // DDR to SPM memcpy, scatter read DDR, broadcast write spm
    // GDN prefill chunk zeroing uses the combined asset's 1D `fill` kernel.
    // Used by rpu_launch_fill_spm_kernel.
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
// Kernels in KERNEL_LIST that may be absent from a compatible combined operator asset.
// Everything else is required.
static bool is_optional_kernel(const std::string& name) {
    static const std::unordered_set<std::string> optional = {
        // Generic ring dispatch is opt-in and may fall back on older op-libs.
        // Exact `profile` users fail at the call site if either symbol is absent.
        "llm_all_reduce_residual", "llm_all_reduce_residual_nopace",
        // Multi-row/warp RMSNorm variants are used only when
        // `RPU_RMSNORM_VWARP` is enabled; missing symbols are allowed otherwise.
        "llama_rms_norm_v16", "llama_rms_norm_v32",
        "llama_rms_norm_newton_v16", "llama_rms_norm_newton_v32",
        // Optional M-RoPE variant with SPM-resident tables.
        "llama_mrope_interleave_local_spm_tbl",
    };
    // `llama_rms_norm_newton` is not optional: RPU_RMSNORM_NEWTON selects it as
    // a production operator, so the matched asset must provide it.
    return optional.count(name) != 0;
}

static std::string canonical_payload_path(const std::string& path) {
    std::error_code error;
    const auto canonical = std::filesystem::canonical(path, error);
    TORCH_CHECK(
        !error, "cannot canonicalize runtime payload '", path,
        "': ", error.message());
    return canonical.string();
}

CachedKernel KernelCache::load_kernel_from_oplib(
    const std::string& path, const std::string& kernel_name) {
    const std::string canonical = canonical_payload_path(path);
    CachedKernel cached;
    cached.program = std::make_unique<::rhino_lkn::Program_t>();
    const int32_t rc = cached.program->create_with_binary_file(
        canonical, kernel_name.c_str());
    TORCH_CHECK(
        rc == 0, "failed to load kernel '", kernel_name, "' from '",
        canonical, "': rc=", rc);
    cached.kernel = std::make_unique<::rhino_lkn::Kernel_t>(
        *cached.program, kernel_name.c_str());
    {
        std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
        loaded_oplib_paths_.insert(canonical);
    }
    return cached;
}

std::vector<std::string> KernelCache::loaded_oplib_paths() const {
    std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
    return {loaded_oplib_paths_.begin(), loaded_oplib_paths_.end()};
}

void KernelCache::initialize() {
    if (initialized_) return;

    if (log_at(3)) {
        std::cout << "[KernelCache] Initializing kernel cache..." << std::endl;
    }
    auto start = std::chrono::high_resolution_clock::now();
    const std::string combined_ref = canonical_payload_path(KERNEL_LIB_PATH());

    std::vector<std::string> missing_required;
    for (const auto& entry : KERNEL_LIST) {
        const std::string& kernel_name = entry.first;
        try {
            CachedKernel cached = load_kernel_from_oplib(combined_ref, kernel_name);
            kernels_.erase(kernel_name);
            kernels_.emplace(kernel_name, std::move(cached));
        } catch (const std::exception& e) {
            if (log_at(2)) {
                std::cerr << "[KernelCache] Warning: Failed to load kernel '"
                          << kernel_name << "': " << e.what() << std::endl;
            }
            if (!is_optional_kernel(kernel_name)) {
                missing_required.push_back(kernel_name);
            }
        }
    }

    TORCH_CHECK(missing_required.empty(),
                "[KernelCache] combined operator asset '", combined_ref,
                "' is incompatible: ", missing_required.size(),
                " required kernel(s) failed to load. First: '",
                (missing_required.empty() ? std::string() : missing_required.front()),
                "'. Install the combined operator asset matching this RhinoForge build or set "
                "RPU_KERNEL_LIB_PATH to it.");

    // Auto-tile (default ON): load all six 7-tile families from the combined
    // operator asset: FP16/W8/W4 x ACC16/ACC32. Missing one is an artifact mismatch, not a
    // recoverable fallback, because the corresponding base kernels are retired.
    if (rpu_pl_tiling::linear_autotile_enabled()) {
        auto at_names = rpu_pl_tiling::autotile_kernel_names();
        int at_loaded = 0;
        for (const auto& kname : at_names) {
            try {
                CachedKernel cached = load_kernel_from_oplib(combined_ref, kname);
                // Some tiled names are also part of the fixed preload list.
                // Destroy a replaced Kernel before its owning Program.
                kernels_.erase(kname);
                kernels_.emplace(kname, std::move(cached));
                ++at_loaded;
            } catch (const std::exception& e) {
                if (log_at(2)) {
                    std::cerr << "[KernelCache] autotile: failed to load '" << kname
                              << "' from " << combined_ref << ": " << e.what()
                              << std::endl;
                }
            }
        }
        TORCH_CHECK(
            at_loaded == static_cast<int>(at_names.size()),
            "[KernelCache] autotile operator asset mismatch: loaded ", at_loaded, "/",
            at_names.size(), " required tiled linear kernels from '", combined_ref,
            "'. Install the combined operator asset matching the selected runtime profile.");
        if (log_at(3)) {
            std::cout << "[KernelCache] autotile ON: loaded " << at_loaded
                      << "/" << at_names.size() << " tiled linear variants from "
                      << combined_ref << std::endl;
        }
    }

    // Preload the optional Wall mixed/local-SPM/QKV family from the combined operator asset.
    {
        int mixed_loaded = 0;
        static constexpr const char* kMixedNames[] = {
            "parallel_linear_acc32_out_bf16_m208n128k128",
            "parallel_linear_acc32_out_bf16_m240n112k128",
            "parallel_linear_acc32_out_bf16_m272n96k128",
            "parallel_linear_acc32_out_bf16_m304n80k128",
            "parallel_linear_acc32_out_bf16_m352n64k128",
            "parallel_linear_acc32_out_bf16_m416n48k128",
            "parallel_linear_acc32_out_bf16_m496n32k128",
            "parallel_linear_w8a16_acc32_out_bf16_m400n48k128",
            "llm_all_reduce_residual_bf16_partial_fp16_residual",
            "llm_all_reduce_residual_bf16_partial_fp16_residual_nopace",
            "llm_all_reduce_residual_fp16_partial_bf16_residual",
            "llm_all_reduce_residual_fp16_partial_bf16_residual_nopace",
            "llm_all_reduce_residual_local_spm",
            "llm_all_reduce_residual_fp16_partial_bf16_residual_local_spm",
            "llama_reduce_sum_bf16_partial_fp16_residual",
            "llama_reduce_sum_fp16_partial_bf16_residual",
            "llama_rms_norm_bf16in_fp16out",
            "llama_rms_norm_bf16in_fp16out_v16",
            "llama_rms_norm_bf16in_fp16out_v32",
            "qkv_parallel_w8a16_multibase_m448n64k128",
        };
        for (const char* name : kMixedNames) {
            const std::string kname(name);
            try {
                CachedKernel cached = load_kernel_from_oplib(combined_ref, kname);
                kernels_.erase(kname);
                kernels_.emplace(kname, std::move(cached));
                ++mixed_loaded;
            } catch (const std::exception& e) {
                if (log_at(2)) {
                    std::cerr << "[KernelCache] mixed: failed to load '"
                              << kname << "' from " << combined_ref << ": "
                              << e.what() << std::endl;
                }
            }
        }
        if (log_at(3) && mixed_loaded != 0) {
            std::cout << "[KernelCache] mixed: loaded " << mixed_loaded
                      << "/" << sizeof(kMixedNames) / sizeof(kMixedNames[0])
                      << " kernels from "
                      << combined_ref << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    if (log_at(3)) {
        std::cout << "[KernelCache] Loaded " << kernels_.size() << " kernels in "
                  << elapsed << "ms" << std::endl;
    }

    // 填充快速缓存数组 (O(1) 访问)
    populate_fast_cache();

    initialized_ = true;
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
            // Auto-tile IDs occupy one contiguous range and are absent only for
            // an explicit legacy-asset opt-out.
            bool at_optin = (i >= static_cast<size_t>(KernelId::PL_AT_W8A16_M288N128) &&
                             i <= static_cast<size_t>(KernelId::PL_AT_INT4_PGRP_ACC32_M496N32) &&
                             !rpu_pl_tiling::linear_autotile_enabled());
            bool optional_id =
                (i >= static_cast<size_t>(
                         KernelId::PL_ACC32_OUT_BF16_M416N48)
                     && i <= static_cast<size_t>(
                         KernelId::RMS_NORM_BF16IN_FP16OUT_V32))
                 || (i >= static_cast<size_t>(
                         KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_LOCAL_SPM)
                     && i <= static_cast<size_t>(
                         KernelId::LLM_ALL_REDUCE_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_SPM))
                 || (i >= static_cast<size_t>(KernelId::RMS_NORM_SPM_V16)
                     && i <= static_cast<size_t>(KernelId::MROPE_SPM_TBL))
                 || i == static_cast<size_t>(
                         KernelId::WALL_QKV_W8A16_MULTIBASE_M448N64);
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

    try {
        CachedKernel cached =
            load_kernel_from_oplib(KERNEL_LIB_PATH(), kernel_name);
        auto* ptr = cached.kernel.get();
        kernels_.erase(kernel_name);
        kernels_.emplace(kernel_name, std::move(cached));
        return ptr;
    } catch (const std::exception& e) {
        if (log_at(2)) {
            std::cerr << "[KernelCache] failed to load dynamic kernel '"
                      << kernel_name << "' from the combined op library: "
                      << e.what() << std::endl;
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
