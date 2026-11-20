// Cold, synchronous GR00T AdaLN and time-projection table construction.
// Scratch is eager-only; only the returned DDR tensors survive each call.
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_spm_allocator.h"
#include "execution_coordinator.h"

using namespace ::rhino_lkn;

at::Tensor rpu_gr00t_precompute_adaln(
    const at::Tensor& cond, at::TensorList weights,
    at::TensorList biases, at::TensorList scales, int cores) {
    constexpr int64_t layers = 32, h = 1536, output = 2 * h;
    RpuExecutionCoordinator::require_graph_quiescent("gr00t_precompute_adaln");
    TORCH_CHECK(RpuExecutionCoordinator::current_physical_token_if_owned("gr00t_precompute_adaln") != 0,
                "GR00T AdaLN cold preparation requires its owned execution claim");
    TORCH_CHECK(!RpuKernelGraph::has_active(), "GR00T AdaLN preparation is cold-only");
    TORCH_CHECK(cores == 4 || cores == 8,
                "GR00T AdaLN preparation requires four or eight execution cores");
    TORCH_CHECK(cond.defined() && cond.device().type() == at::kPrivateUse1 &&
                    cond.scalar_type() == at::kHalf && cond.is_contiguous() &&
                    cond.dim() == 2 && (cond.size(0) == 4 || cond.size(0) == 10) && cond.size(1) == h &&
                    weights.size() == layers && biases.size() == layers && scales.size() == layers,
                "GR00T AdaLN preparation requires FP16 cond[4 or 10,1536] and 32 installed projections");
    const int64_t steps = cond.size(0);
    for (int64_t layer = 0; layer < layers; ++layer) {
        const auto& w = weights[layer]; const auto& b = biases[layer];
        const auto& s = scales[layer];
        TORCH_CHECK(w.defined() && w.device() == cond.device() && w.is_contiguous() &&
                        w.dim() == 2 && w.size(0) == output && w.size(1) == h &&
                        (w.scalar_type() == at::kHalf || w.scalar_type() == at::kChar) &&
                        b.defined() && b.device() == cond.device() && b.is_contiguous() &&
                        b.scalar_type() == at::kHalf && b.dim() == 1 && b.numel() == output,
                    "GR00T AdaLN preparation weight/bias geometry or dtype mismatch");
        TORCH_CHECK(w.scalar_type() == at::kChar
                        ? s.defined() && s.device() == cond.device() && s.is_contiguous() &&
                          s.scalar_type() == at::kHalf && s.dim() == 1 && s.numel() == output
                        : !s.defined(),
                    "GR00T AdaLN preparation scale does not match installed precision");
    }
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    const int64_t input_bytes = ((h / cores * 2 + 255) / 256) * 256;
    const int64_t scratch_bytes = input_bytes + 3 * output * 2;
    TORCH_CHECK(SPM_ALLOC.total_used() + scratch_bytes <= SpmAllocator::SPM_PLANNING_BUDGET &&
                    SPM_ALLOC.free_space() >= scratch_bytes,
                "GR00T AdaLN cold scratch exceeds the SPM planning budget");
    SpmAllocator::ScopedTemporary scratch;
    const uint32_t input = SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(input_bytes));
    const uint32_t bias = SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(output * 2));
    const uint32_t partial = SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(output * 2));
    const uint32_t result = SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(output * 2));
    auto table = at::empty({steps, layers, output}, cond.options());
    rpu_ddr_flush_force(cond.data_ptr<c10::Half>());
    ScopedEagerAtomicBatch batch(/*enabled=*/true, cores);
    for (int64_t step = 0; step < steps; ++step) {
        rpu_launch_ddr_scatter_spm_dma(cond, step * h, h / cores,
                                     h / cores * 2, input, cores);
        for (int64_t layer = 0; layer < layers; ++layer) {
            rpu_launch_ddr_broadcast_spm_dma(biases[layer], 0, output, bias, cores);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                input, weights[layer], partial, 1, output, h, 0, cores, 0, scales[layer]);
            rpu_launch_all_reduce_sum_residual_kernel(partial, bias, result, 1, output, cores, cores);
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                result, c10::Half(1.0), result, h, ValuOpType::ADD, cores);
            rpu_launch_spm_copy_ddr_dma(result, table, (step * layers + layer) * output, output);
        }
    }
    batch.submit();
    rpu_ddr_flush(table.data_ptr<c10::Half>());
    return table;
}

std::tuple<at::Tensor, at::Tensor> rpu_gr00t_precompute_time_glue(
    const at::Tensor& tau, const at::Tensor& cond,
    const at::Tensor& tau_weight, const at::Tensor& tau_scale,
    const at::Tensor& po1_weight, const at::Tensor& po1_bias, const at::Tensor& po1_scale) {
    constexpr int64_t horizon = 40, h = 1536, output = 2 * h;
    RpuExecutionCoordinator::require_graph_quiescent("gr00t_precompute_time_glue");
    TORCH_CHECK(RpuExecutionCoordinator::current_physical_token_if_owned("gr00t_precompute_time_glue") != 0,
                "GR00T time preparation requires its owned execution claim");
    TORCH_CHECK(!RpuKernelGraph::has_active(), "GR00T time preparation is cold-only");
    TORCH_CHECK(tau.defined() && tau.device().type() == at::kPrivateUse1 &&
                    tau.scalar_type() == at::kHalf && tau.is_contiguous() &&
                    tau.dim() == 3 && (tau.size(0) == 4 || tau.size(0) == 10) &&
                    tau.size(1) == horizon && tau.size(2) == h &&
                    cond.defined() && cond.device() == tau.device() &&
                    cond.scalar_type() == at::kHalf && cond.is_contiguous() &&
                    cond.dim() == 2 && cond.size(0) == tau.size(0) && cond.size(1) == h,
                "GR00T time preparation requires matching FP16 tau[4 or 10,40,1536] and cond[4 or 10,1536]");
    const int64_t steps = tau.size(0);
    const auto check_projection = [&](const at::Tensor& w, const at::Tensor& scale, int64_t n) {
        TORCH_CHECK(w.defined() && w.device() == tau.device() && w.is_contiguous() &&
                        w.dim() == 2 && w.size(0) == n && w.size(1) == h &&
                        (w.scalar_type() == at::kHalf || w.scalar_type() == at::kChar),
                    "GR00T time preparation weight geometry or dtype mismatch");
        TORCH_CHECK(w.scalar_type() == at::kChar
                        ? scale.defined() && scale.device() == tau.device() && scale.is_contiguous() &&
                          scale.scalar_type() == at::kHalf && scale.dim() == 1 && scale.numel() == n
                        : !scale.defined(),
                    "GR00T time preparation scale does not match installed precision");
    };
    check_projection(tau_weight, tau_scale, h);
    check_projection(po1_weight, po1_scale, output);
    TORCH_CHECK(po1_bias.defined() && po1_bias.device() == tau.device() &&
                    po1_bias.scalar_type() == at::kHalf && po1_bias.is_contiguous() &&
                    po1_bias.dim() == 1 && po1_bias.numel() == output,
                "GR00T time preparation requires FP16 po1 bias[3072]");
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    constexpr int64_t tensor_bytes = horizon * h * 2;
    constexpr int64_t scratch_bytes = 2 * tensor_bytes + output * 2;
    TORCH_CHECK(SPM_ALLOC.total_used() + scratch_bytes <= SpmAllocator::SPM_PLANNING_BUDGET &&
                    SPM_ALLOC.free_space() >= scratch_bytes,
                "GR00T time cold scratch exceeds the SPM planning budget");
    SpmAllocator::ScopedTemporary scratch;
    const uint32_t input = SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(tensor_bytes));
    const uint32_t result = SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(tensor_bytes));
    const uint32_t bias = SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(output * 2));
    auto tau_table = at::empty({steps, horizon, h}, tau.options());
    auto po1_table = at::empty({steps, output}, cond.options());
    rpu_ddr_flush_force(tau.data_ptr<c10::Half>());
    rpu_ddr_flush_force(cond.data_ptr<c10::Half>());
    ScopedEagerAtomicBatch batch(/*enabled=*/true, /*num_cores=*/8);
    rpu_launch_ddr_broadcast_spm_dma(po1_bias, 0, output, bias, 1);
    for (int64_t step = 0; step < steps; ++step) {
        rpu_launch_ddr_broadcast_spm_dma(tau, step * horizon * h, horizon * h, input, 1);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            input, tau_weight, result, horizon, h, h, 1, 1, 0, tau_scale);
        rpu_launch_spm_copy_ddr_dma(result, tau_table, step * horizon * h, horizon * h);
        rpu_launch_ddr_broadcast_spm_dma(cond, step * h, h, input, 1);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            input, po1_weight, result, 1, output, h, 1, 1, bias, po1_scale);
        const uint32_t scale = result + h * 2;
        // Preserve norm_out's [shift, 1+scale] order and its default core count.
        rpu_launch_eltwise_binary_scalar_spm_kernel(scale, c10::Half(1.0), scale, h, ValuOpType::ADD);
        rpu_launch_spm_copy_ddr_dma(result, po1_table, step * output, output);
    }
    batch.submit();
    rpu_ddr_flush(tau_table.data_ptr<c10::Half>());
    rpu_ddr_flush(po1_table.data_ptr<c10::Half>());
    return {tau_table, po1_table};
}
