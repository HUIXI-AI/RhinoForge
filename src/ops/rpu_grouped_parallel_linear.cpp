// Grouped tensor-parallel Linear for MoE expert projections.
//

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;

namespace {

constexpr int64_t kModeInt8 =
    static_cast<int64_t>(GroupedParallelLinearWeightMode::INT8);
constexpr int64_t kModeFp16 =
    static_cast<int64_t>(GroupedParallelLinearWeightMode::FP16);
constexpr int64_t kModeFp8E4M3 =
    static_cast<int64_t>(GroupedParallelLinearWeightMode::FP8_E4M3);

void set_reg_u32(Kernel_t* kernel, int lo_reg, uint32_t value) {
    kernel->set_regs(lo_reg, static_cast<uint16_t>(value & 0xFFFF));
    kernel->set_regs(lo_reg + 1, static_cast<uint16_t>(value >> 16));
}

int64_t resolve_weight_mode(const at::Tensor& weight, int64_t requested) {
    if (requested < 0) {
        if (weight.scalar_type() == at::kHalf) return kModeFp16;
        TORCH_CHECK(
            false,
            "grouped_parallel_linear: raw uint8 storage requires explicit "
            "weight_dtype_mode=5 (fp8_e4m3)");
    }

    TORCH_CHECK(requested == kModeFp16 || requested == kModeInt8 || requested == kModeFp8E4M3,
                "grouped_parallel_linear: this Qwen3.5 integration supports "
                "only mode 1(int8), 2(fp16), or 5(fp8_e4m3); got ",
                requested);
    if (requested == kModeFp16) {
        TORCH_CHECK(weight.scalar_type() == at::kHalf,
                    "grouped_parallel_linear: mode 2 requires fp16 weight storage");
    } else if (requested == kModeInt8) {
        TORCH_CHECK(weight.scalar_type() == at::kChar,
                    "grouped_parallel_linear: mode 1 requires signed int8 weight storage");
    } else {
        TORCH_CHECK(weight.scalar_type() == at::kByte,
                    "grouped_parallel_linear: mode 5 requires raw uint8 "
                    "E4M3 storage");
    }
    return requested;
}

uint32_t checked_u32(uint64_t value, const char* what) {
    TORCH_CHECK(value <= std::numeric_limits<uint32_t>::max(),
                "grouped_parallel_linear: ", what, " exceeds u32: ", value);
    return static_cast<uint32_t>(value);
}

}  // namespace

void rpu_launch_grouped_parallel_linear_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& packed_weight,
    uint32_t m_sizes_spm_addr,
    const c10::optional<at::Tensor>& scale_opt,
    uint32_t output_spm_addr,
    uint32_t bias_spm_addr,
    int64_t local_n,
    int64_t local_k,
    int64_t group,
    bool is_col_parallel,
    int num_cores,
    int64_t weight_dtype_mode,
    bool linear_acc32) {
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "grouped_parallel_linear: num_cores must be in [1,8], got ",
                num_cores);
    TORCH_CHECK(group > 0 && group <= std::numeric_limits<uint16_t>::max(),
                "grouped_parallel_linear: group must fit u16, got ", group);
    TORCH_CHECK(local_n > 0 && local_n <= std::numeric_limits<uint16_t>::max() &&
                    local_n % 16 == 0,
                "grouped_parallel_linear: local_n must be positive, fit u16, "
                "and be divisible by 16; got ", local_n);
    TORCH_CHECK(local_k > 0 && local_k <= std::numeric_limits<uint16_t>::max() &&
                    local_k % 16 == 0,
                "grouped_parallel_linear: local_k must be positive, fit u16, "
                "and be divisible by 16; got ", local_k);
    TORCH_CHECK(local_k * static_cast<int64_t>(sizeof(c10::Half)) <=
                    std::numeric_limits<uint16_t>::max(),
                "grouped_parallel_linear: fp16 input row stride exceeds u16");
    TORCH_CHECK(local_n * static_cast<int64_t>(sizeof(c10::Half)) <=
                    std::numeric_limits<uint16_t>::max(),
                "grouped_parallel_linear: fp16 output row stride exceeds u16");

    TORCH_CHECK(packed_weight.defined() &&
                    packed_weight.device().type() == at::kPrivateUse1 &&
                    packed_weight.is_contiguous(),
                "grouped_parallel_linear: packed_weight must be a contiguous RPU tensor");
    const int64_t mode = resolve_weight_mode(packed_weight, weight_dtype_mode);
    const uint64_t weight_dwidth = packed_weight.element_size();
    const uint64_t weight_group_bytes =
        static_cast<uint64_t>(local_n) * static_cast<uint64_t>(local_k) *
        static_cast<uint64_t>(num_cores) * weight_dwidth;
    const uint64_t expected_weight_bytes =
        static_cast<uint64_t>(group) * weight_group_bytes;
    TORCH_CHECK(static_cast<uint64_t>(packed_weight.nbytes()) == expected_weight_bytes,
                "grouped_parallel_linear: packed_weight has ", packed_weight.nbytes(),
                " bytes, expected ", expected_weight_bytes,
                " for group/local_n/local_k/tp=", group, "/", local_n, "/",
                local_k, "/", num_cores);

    uint32_t scale_group_bytes = 0;
    if (mode != kModeFp16) {
        TORCH_CHECK(scale_opt.has_value() && scale_opt->defined(),
                    "grouped_parallel_linear: 8-bit weights require fp16 scale");
        const at::Tensor& scale = *scale_opt;
        TORCH_CHECK(scale.device().type() == at::kPrivateUse1 &&
                        scale.scalar_type() == at::kHalf && scale.is_contiguous(),
                    "grouped_parallel_linear: scale must be contiguous fp16 on RPU");
        const uint64_t global_n = static_cast<uint64_t>(local_n) *
                                  (is_col_parallel ? num_cores : 1);
        const uint64_t expected_scale_bytes =
            static_cast<uint64_t>(group) * global_n * sizeof(c10::Half);
        TORCH_CHECK(static_cast<uint64_t>(scale.nbytes()) == expected_scale_bytes,
                    "grouped_parallel_linear: scale has ", scale.nbytes(),
                    " bytes, expected ", expected_scale_bytes,
                    " ([group, global_n] fp16)");
        scale_group_bytes = checked_u32(global_n * sizeof(c10::Half),
                                        "scale group byte step");
    } else {
        TORCH_CHECK(!scale_opt.has_value() || !scale_opt->defined(),
                    "grouped_parallel_linear: fp16 mode does not consume scale");
    }

    const KernelId kernel_id =
        mode == kModeFp16
            ? (linear_acc32 ? KernelId::GROUPED_PARALLEL_LINEAR_ACC32
                            : KernelId::GROUPED_PARALLEL_LINEAR_ACC16)
            : (linear_acc32 ? KernelId::GROUPED_PARALLEL_LINEAR_8BIT_ACC32
                            : KernelId::GROUPED_PARALLEL_LINEAR_8BIT_ACC16);
    std::vector<GraphDdrRegisterOperandSpec> ddr_operands{
        GraphDdrRegisterOperandSpec{
            GraphDdrRegisterAbi{
                mode == kModeFp16
                    ? GraphDdrRegisterRole::ModelWeight
                    : (mode == kModeInt8
                        ? GraphDdrRegisterRole::Int8ModelWeight
                        : GraphDdrRegisterRole::Fp8ModelWeight),
                GraphDdrRegisterAccess::Read,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                2, 3},
            packed_weight}};
    if (mode != kModeFp16) {
        ddr_operands.push_back(GraphDdrRegisterOperandSpec{
            GraphDdrRegisterAbi{
                GraphDdrRegisterRole::PerChannelScale,
                GraphDdrRegisterAccess::Read,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                20, 21},
            *scale_opt});
    }
    auto register_writer =
        RpuKernelGraph::active().stage_kernel_ddr_registers(
            kernel_id, std::move(ddr_operands));
    rpu_ddr_flush(packed_weight.data_ptr());
    if (mode != kModeFp16) {
        rpu_ddr_flush(scale_opt->data_ptr());
    }

    Kernel_t* kernel = GET_KERNEL(kernel_id);
    TORCH_CHECK(kernel != nullptr,
                "grouped_parallel_linear: required kernel missing from rhino op lib");
    kernel->reset_regs();
    set_reg_u32(kernel, 0, input_spm_addr);
    set_reg_u32(kernel, 4, m_sizes_spm_addr);
    set_reg_u32(kernel, 6, output_spm_addr);
    set_reg_u32(kernel, 8, bias_spm_addr);
    kernel->set_regs(10, static_cast<uint16_t>(group));
    kernel->set_regs(11, static_cast<uint16_t>(local_n));
    kernel->set_regs(12, static_cast<uint16_t>(local_k));
    kernel->set_regs(13, static_cast<uint16_t>(local_k / 16));
    kernel->set_regs(14, static_cast<uint16_t>(bias_spm_addr != 0));
    kernel->set_regs(15, static_cast<uint16_t>(num_cores));
    kernel->set_regs(16, static_cast<uint16_t>(local_k * sizeof(c10::Half)));
    kernel->set_regs(17, static_cast<uint16_t>(local_n * sizeof(c10::Half)));
    set_reg_u32(kernel, 18,
                checked_u32(weight_group_bytes, "weight group byte step"));
    if (mode == kModeFp16) {
        set_reg_u32(kernel, 20, 0);
    }
    set_reg_u32(kernel, 22, scale_group_bytes);
    kernel->set_regs(24, static_cast<uint16_t>(is_col_parallel));
    kernel->set_regs(25, static_cast<uint16_t>(mode));

    const uint64_t grid_x64 = (static_cast<uint64_t>(local_n) + 63) / 64;
    TORCH_CHECK(grid_x64 <= std::numeric_limits<uint16_t>::max(),
                "grouped_parallel_linear: grid.x exceeds u16");
    const uint16_t grid_x = static_cast<uint16_t>(grid_x64);
    kernel->set_regs(64, grid_x);
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(1));
    register_writer.write(*kernel);

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back(static_cast<uint8_t>(i));
    wq->enqueu_kernel(*kernel, {grid_x, 1, 1}, cores);
}

// Conformance-only wrapper. It returns each core's local output (column shard
// or row-parallel partial), isolating this kernel from all-gather/all-reduce.
at::Tensor rpu_grouped_parallel_linear_test(
    const at::Tensor& input_per_core,
    const at::Tensor& packed_weight,
    const at::Tensor& m_sizes,
    const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& bias_per_core,
    int64_t local_n,
    int64_t local_k,
    bool is_col_parallel,
    int64_t num_cores,
    int64_t weight_dtype_mode,
    bool linear_acc32) {
    RECORD_FUNCTION("rpu::grouped_parallel_linear_test", {});
    TORCH_CHECK(RpuKernelGraph::active().state() == RpuKernelGraph::State::PASSTHROUGH,
                "grouped_parallel_linear_test is an immediate conformance wrapper; "
                "it cannot be captured or replayed");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "grouped_parallel_linear_test: num_cores must be in [1,8]");
    TORCH_CHECK(input_per_core.device().type() == at::kPrivateUse1 &&
                    input_per_core.scalar_type() == at::kHalf &&
                    input_per_core.is_contiguous() && input_per_core.dim() == 3,
                "grouped_parallel_linear_test: input_per_core must be contiguous "
                "fp16 RPU [num_cores,M,local_k]");
    TORCH_CHECK(input_per_core.size(0) == num_cores &&
                    input_per_core.size(2) == local_k,
                "grouped_parallel_linear_test: input shape/dims mismatch");
    TORCH_CHECK(m_sizes.device().type() == at::kPrivateUse1 &&
                    m_sizes.scalar_type() == at::kShort &&
                    m_sizes.is_contiguous() && m_sizes.dim() == 1,
                "grouped_parallel_linear_test: m_sizes must be contiguous int16 "
                "RPU [group]");
    const int64_t group = m_sizes.numel();
    const int64_t m = input_per_core.size(1);
    TORCH_CHECK(m > 0, "grouped_parallel_linear_test: M must be positive");

    if (bias_per_core.has_value() && bias_per_core->defined()) {
        TORCH_CHECK(bias_per_core->device().type() == at::kPrivateUse1 &&
                        bias_per_core->scalar_type() == at::kHalf &&
                        bias_per_core->is_contiguous() &&
                        bias_per_core->dim() == 2 &&
                        bias_per_core->size(0) == num_cores &&
                        bias_per_core->size(1) == local_n,
                    "grouped_parallel_linear_test: bias_per_core must be contiguous "
                    "fp16 RPU [num_cores,local_n]");
    }

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    const int64_t input_bytes = m * local_k * sizeof(c10::Half);
    // The immediate DDR->SPM DMA accepts 16-byte multiples only.  Kernels
    // still consume exactly `group` entries; the padded tail is transport-only.
    const int64_t m_sizes_dma_elements = ((group + 7) / 8) * 8;
    const int64_t m_sizes_bytes = m_sizes_dma_elements * sizeof(int16_t);
    const int64_t output_bytes = m * local_n * sizeof(c10::Half);
    const int64_t bias_bytes = local_n * sizeof(c10::Half);

    at::Tensor m_sizes_dma = m_sizes;
    if (m_sizes_dma_elements != group) {
        m_sizes_dma = at::zeros({m_sizes_dma_elements}, m_sizes.options());
        m_sizes_dma.narrow(0, 0, group).copy_(m_sizes);
    }

    using AR = SpmAllocator::AllocRequest;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{input_bytes, 1, 2},
        AR{m_sizes_bytes, 1, 2},
        AR{bias_bytes, 1, 2},
        AR{output_bytes, 2, 3},
    });
    const uint32_t input_addr = SPM_ALLOC.addr(0, offsets[0]);
    const uint32_t m_sizes_addr = SPM_ALLOC.addr(0, offsets[1]);
    const uint32_t bias_addr = SPM_ALLOC.addr(0, offsets[2]);
    const uint32_t output_addr = SPM_ALLOC.addr(0, offsets[3]);

    try {
        rpu_launch_ddr_scatter_spm_dma_immediate(
            const_cast<c10::Half*>(input_per_core.data_ptr<c10::Half>()),
            m * local_k, input_bytes, input_addr, static_cast<int>(num_cores));
        rpu_launch_ddr_broadcast_spm_dma_immediate(
            reinterpret_cast<c10::Half*>(m_sizes_dma.data_ptr<int16_t>()),
            m_sizes_dma_elements, m_sizes_addr, static_cast<int>(num_cores));

        uint32_t launch_bias_addr = 0;
        if (bias_per_core.has_value() && bias_per_core->defined()) {
            rpu_launch_ddr_scatter_spm_dma_immediate(
                const_cast<c10::Half*>(bias_per_core->data_ptr<c10::Half>()),
                local_n, bias_bytes, bias_addr, static_cast<int>(num_cores));
            launch_bias_addr = bias_addr;
        }

        rpu_launch_grouped_parallel_linear_spm_kernel(
            input_addr, packed_weight, m_sizes_addr, scale, output_addr,
            launch_bias_addr, local_n, local_k, group, is_col_parallel,
            static_cast<int>(num_cores), weight_dtype_mode, linear_acc32);

        auto output = at::empty({num_cores, m, local_n}, input_per_core.options());
        rpu_launch_spm_scatter_ddr_dma_immediate(
            output_addr, output.data_ptr<c10::Half>(), m * local_n,
            output_bytes, static_cast<int>(num_cores));
        SPM_ALLOC.reset_temporary();
        return output;
    } catch (...) {
        SPM_ALLOC.reset_temporary();
        throw;
    }
}
