// Generic FP16 SPM MoE selection for backends that keep expert execution dense.
//
// Input scores stay token-major [T,E] for top-k and expert-major [E,T] for the
// downstream dense combine.  Selection is strict and deterministic:
//
//   choice[t,e] = unbiased_score[t,e] + correction_bias[e]
//   selected     = exactly top_k(choice), ties by lower expert index
//   weight[e,t]  = unbiased_score[t,e] if selected else 0
//   weight      /= sum_e(weight)
//
// correction_bias therefore changes membership only; it never leaks into the
// routed weights.  No FP32 value is materialised in SPM.  The three dynamic
// kernels used here already ship in rhinoOpLib_current.ref:
// topk_by_select_fp16, unary_cast_uint16_int32, and
// scatter_elements_v2_spm_smallC.

#include "rpu_kernel_decls.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

#include <c10/util/Half.h>

#include <cstdint>

using namespace ::rhino_lkn;

namespace {

Kernel_t* graph_kernel_by_name(const char* name) {
    // get_kernel() performs the dynamic .ref lookup.  The graph's named-kernel
    // path only clones an already-cached program, so prime the cache first.
    TORCH_CHECK(KernelCache::instance().get_kernel(name) != nullptr,
                "backend_moe_select: kernel '", name,
                "' is absent from the active rhino oplib");
    Kernel_t* kernel = RpuKernelGraph::active().get_kernel_reset(name);
    TORCH_CHECK(kernel != nullptr,
                "backend_moe_select: failed to prepare kernel '", name, "'");
    return kernel;
}

void launch_topk_by_select_fp16(
    uint32_t in_key_addr, uint32_t in_value_u16_addr,
    uint32_t out_key_addr, uint32_t out_value_u16_addr,
    int64_t n, int64_t c, int64_t k) {
    TORCH_CHECK(n > 0 && n <= UINT16_MAX,
                "backend_moe_select: topk n must be in [1,65535], got ", n);
    TORCH_CHECK(c > 0 && c < UINT16_MAX && k > 0 && k <= c,
                "backend_moe_select: require 0 < k <= c < 65535, got k=",
                k, " c=", c);

    Kernel_t* kernel = graph_kernel_by_name("topk_by_select_fp16");
    kernel->set_regs(0,  static_cast<uint16_t>(in_key_addr & 0xFFFF));
    kernel->set_regs(1,  static_cast<uint16_t>(in_key_addr >> 16));
    kernel->set_regs(2,  static_cast<uint16_t>(in_value_u16_addr & 0xFFFF));
    kernel->set_regs(3,  static_cast<uint16_t>(in_value_u16_addr >> 16));
    kernel->set_regs(4,  static_cast<uint16_t>(out_key_addr & 0xFFFF));
    kernel->set_regs(5,  static_cast<uint16_t>(out_key_addr >> 16));
    kernel->set_regs(6,  static_cast<uint16_t>(out_value_u16_addr & 0xFFFF));
    kernel->set_regs(7,  static_cast<uint16_t>(out_value_u16_addr >> 16));
    kernel->set_regs(8,  static_cast<uint16_t>(k));
    // Host register contract: cv16_num deliberately includes the tail vector
    // even when c is v16-aligned.
    kernel->set_regs(10, static_cast<uint16_t>(c / 16 + 1));
    kernel->set_regs(11, static_cast<uint16_t>(c % 16));
    kernel->set_regs(12, static_cast<uint16_t>(c));
    kernel->set_regs(13, static_cast<uint16_t>(n));
    const uint16_t grid_x = static_cast<uint16_t>((n + 15) / 16);
    kernel->set_regs(64, grid_x);
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(1));

    auto* queue = GET_QUEUE(1);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {grid_x, 1, 1}, {0});
}

void launch_cast_uint16_int32(
    uint32_t in_addr, uint32_t out_addr, int64_t num_elements) {
    TORCH_CHECK(num_elements > 0,
                "backend_moe_select: cast element count must be positive");
    constexpr int64_t kElementsPerBlock = 63 * 256;
    const int64_t blocks =
        (num_elements + kElementsPerBlock - 1) / kElementsPerBlock;
    TORCH_CHECK(blocks <= UINT16_MAX,
                "backend_moe_select: cast grid exceeds uint16");
    const int64_t last =
        num_elements - (blocks - 1) * kElementsPerBlock;

    Kernel_t* kernel = graph_kernel_by_name("unary_cast_uint16_int32");
    kernel->set_regs(0, static_cast<uint16_t>(kElementsPerBlock));
    kernel->set_regs(1, static_cast<uint16_t>(last));
    kernel->set_regs(4, static_cast<uint16_t>(in_addr & 0xFFFF));
    kernel->set_regs(5, static_cast<uint16_t>(in_addr >> 16));
    kernel->set_regs(6, static_cast<uint16_t>(out_addr & 0xFFFF));
    kernel->set_regs(7, static_cast<uint16_t>(out_addr >> 16));
    kernel->set_regs(64, static_cast<uint16_t>(blocks));
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(1));

    auto* queue = GET_QUEUE(1);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel, {static_cast<uint16_t>(blocks), 1, 1}, {0});
}

void launch_scatter_axis0_mask(
    uint32_t data_addr, uint32_t indices_i32_addr,
    uint32_t updates_addr, int64_t n, int64_t m, int64_t c) {
    TORCH_CHECK(n >= m && n > 0 && m > 0,
                "backend_moe_select: scatter requires n >= m > 0, got n=",
                n, " m=", m);
    TORCH_CHECK(c > 0 && c <= 128,
                "backend_moe_select: small-C scatter supports c in [1,128], got ",
                c);
    TORCH_CHECK(n <= UINT32_MAX && m <= UINT32_MAX && c <= UINT16_MAX,
                "backend_moe_select: scatter shape exceeds register range");

    Kernel_t* kernel =
        graph_kernel_by_name("scatter_elements_v2_spm_smallC");
    kernel->set_regs(0, static_cast<uint16_t>(data_addr & 0xFFFF));
    kernel->set_regs(1, static_cast<uint16_t>(data_addr >> 16));
    kernel->set_regs(2, static_cast<uint16_t>(indices_i32_addr & 0xFFFF));
    kernel->set_regs(3, static_cast<uint16_t>(indices_i32_addr >> 16));
    kernel->set_regs(4, static_cast<uint16_t>(updates_addr & 0xFFFF));
    kernel->set_regs(5, static_cast<uint16_t>(updates_addr >> 16));
    kernel->set_regs(6, static_cast<uint16_t>(n & 0xFFFF));
    kernel->set_regs(7, static_cast<uint16_t>((n >> 16) & 0xFFFF));
    kernel->set_regs(8, static_cast<uint16_t>(m & 0xFFFF));
    kernel->set_regs(9, static_cast<uint16_t>((m >> 16) & 0xFFFF));
    kernel->set_regs(10, static_cast<uint16_t>(c));
    kernel->set_regs(11, static_cast<uint16_t>(0));  // reduction=none
    kernel->set_regs(12, static_cast<uint16_t>(1));  // one core
    const uint16_t grid_x = static_cast<uint16_t>((m + 127) / 128);
    const uint16_t grid_y = static_cast<uint16_t>((c + 255) / 256);
    kernel->set_regs(64, grid_x);
    kernel->set_regs(65, grid_y);
    kernel->set_regs(66, static_cast<uint16_t>(1));

    auto* queue = GET_QUEUE(1);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {grid_x, grid_y, 1}, {0});
}

}  // namespace

void rpu_launch_backend_moe_select_fp16_spm(
    uint32_t scores_token_major_addr,
    uint32_t scores_expert_major_addr,
    uint32_t correction_bias_addr,
    uint32_t scatter_ids_u16_addr,
    uint32_t choice_token_major_addr,
    uint32_t topk_keys_addr,
    uint32_t topk_ids_u16_addr,
    uint32_t topk_ids_i32_addr,
    uint32_t dense_mask_addr,
    uint32_t reduce_scratch_addr,
    uint32_t dense_weights_addr,
    int64_t token_rows,
    int64_t num_experts,
    int64_t top_k) {
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "backend_moe_select: SPM allocator is not initialized");
    TORCH_CHECK(token_rows > 0 && token_rows % 16 == 0 && token_rows <= 128,
                "backend_moe_select: token_rows must be v16-aligned and <=128, got ",
                token_rows);
    TORCH_CHECK(num_experts >= 16 && num_experts % 16 == 0 &&
                    (num_experts & (num_experts - 1)) == 0,
                "backend_moe_select: num_experts must be a v16-aligned power of two, got ",
                num_experts);
    TORCH_CHECK(top_k > 0 && top_k <= num_experts,
                "backend_moe_select: invalid top_k=", top_k);
    TORCH_CHECK(num_experts <= (UINT16_MAX - 1) / token_rows,
                "backend_moe_select: flattened scatter ids exceed uint16");

    const int64_t et = num_experts * token_rows;
    const int64_t tk = top_k * token_rows;

    // Bias is selection-only.  Keep the unbiased scores in both source layouts
    // untouched for the later gather-by-mask and normalisation.
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
        correction_bias_addr, scores_token_major_addr,
        choice_token_major_addr,
        /*n=*/token_rows, /*c=*/num_experts, c10::Half(1.0f),
        ValuOpType::ADD, /*is_bopa=*/false);

    launch_topk_by_select_fp16(
        choice_token_major_addr, scatter_ids_u16_addr,
        topk_keys_addr, topk_ids_u16_addr,
        token_rows, num_experts, top_k);

    // The carried uint16 value is already the flattened [E,T] destination
    // index (expert*T + token).  Widen [T,k] as one contiguous [T*k,1]
    // vector.  scatter_elements reduce=none only supports one index per M row;
    // C=1 is therefore the exact supported contract, unlike a varying [k,T]
    // index matrix.
    launch_cast_uint16_int32(
        topk_ids_u16_addr, topk_ids_i32_addr, tk);

    rpu_launch_fill_spm_kernel(
        dense_mask_addr, et, c10::Half(0.0f), /*num_cores=*/1);
    // Top-k keys are dead after selection; reuse that equally sized slot as
    // [T*k,1] FP16 ones. Preserve the uint16 ids for conformance readback.
    rpu_launch_fill_spm_kernel(
        topk_keys_addr, tk, c10::Half(1.0f), /*num_cores=*/1);
    launch_scatter_axis0_mask(
        dense_mask_addr, topk_ids_i32_addr, topk_keys_addr,
        /*n=*/et, /*m=*/tk, /*c=*/1);

    // Gather from UNBIASED scores, then normalise over exactly top_k experts.
    rpu_launch_eltwise_binary_spm_kernel(
        scores_expert_major_addr, dense_mask_addr, dense_weights_addr,
        et, ValuOpType::MUL, c10::Half(1.0f), /*num_cores=*/1);
    int64_t rows = num_experts / 2;
    rpu_launch_eltwise_binary_spm_kernel(
        dense_weights_addr,
        dense_weights_addr + rows * token_rows * sizeof(c10::Half),
        reduce_scratch_addr, rows * token_rows,
        ValuOpType::ADD, c10::Half(1.0f), /*num_cores=*/1);
    for (rows /= 2; rows >= 1; rows /= 2) {
        rpu_launch_eltwise_binary_spm_kernel(
            reduce_scratch_addr,
            reduce_scratch_addr + rows * token_rows * sizeof(c10::Half),
            reduce_scratch_addr, rows * token_rows,
            ValuOpType::ADD, c10::Half(1.0f), /*num_cores=*/1);
    }
    // The reference adds 1e-20 before normalising.  That value is not
    // representable in FP16, so use the smallest positive FP16 subnormal.  It
    // rounds away for ordinary sums, while keeping an all-underflow selected
    // set finite (zero weights instead of 0/0 NaNs).
    rpu_launch_eltwise_binary_scalar_spm_kernel(
        reduce_scratch_addr,
        c10::Half(0x0001, c10::Half::from_bits()),
        reduce_scratch_addr, token_rows, ValuOpType::ADD);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
        reduce_scratch_addr, dense_weights_addr, dense_weights_addr,
        /*n=*/num_experts, /*c=*/token_rows, c10::Half(1.0f),
        ValuOpType::DIV, /*is_bopa=*/true);
}
