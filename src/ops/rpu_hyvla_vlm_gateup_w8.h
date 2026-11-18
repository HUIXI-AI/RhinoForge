#pragma once
#include "rpu_ops.h"
#include "rpu_kernel_decls.h"
#include "rpu_spm_allocator.h"
#include "rpu_linear_tiling.h"
#include "rpu_specialized_linear_contracts.h"

// Exact M240/H2048/I6144/TP8 VLM adapter. Both branches retain their own
// installed INT8 weights and FP16 per-channel scales. The existing Wall payload
// computes Gate/Up and the fused SwiGLU operation. Separate SiLU writes an
// FP16 intermediate, so cross-route outputs may differ at rounding boundaries.
inline void launch_hyvla_vlm_gateup_w8(
    uint32_t input, uint32_t output, int64_t rows,
    const at::Tensor& gate_weight, const at::Tensor& up_weight,
    const at::Tensor& gate_scale, const at::Tensor& up_scale) {
    TORCH_CHECK(rows == 240 && SPM_ALLOC.is_initialized(),
                "HyVLA VLM W8 Gate/Up requires M240 and initialized SPM");
    for (auto* weight : {&gate_weight, &up_weight})
        TORCH_CHECK(weight->defined() && weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->scalar_type() == at::kChar && weight->is_contiguous() &&
                    weight->dim() == 2 && weight->size(0) == 6144 && weight->size(1) == 2048,
                    "HyVLA VLM W8 Gate/Up requires installed INT8 [6144,2048]");
    for (auto* scale : {&gate_scale, &up_scale})
        TORCH_CHECK(scale->defined() && scale->device() == gate_weight.device() &&
                    scale->scalar_type() == at::kHalf && scale->is_contiguous() &&
                    scale->dim() == 1 && scale->size(0) == 6144,
                    "HyVLA VLM W8 Gate/Up requires owned FP16 [6144] scales");
    TORCH_CHECK(up_weight.device() == gate_weight.device(), "HyVLA VLM W8 device mismatch");
    const std::array<std::pair<uint32_t, uint64_t>, 2> regions{{
        {input, 240*2048*2}, {output, 240*768*2}}};
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    for (const auto& region : regions) {
        const uint64_t start = region.first, bytes = region.second;
        TORCH_CHECK(start >= base && start % 256 == 0 &&
                    start - base <= SpmAllocator::SPM_USABLE - bytes,
                    "HyVLA VLM W8 requires complete aligned SPM regions");
        for (int core = 0; core < 8; ++core)
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + start - base + bytes - 1 <= UINT32_MAX,
                        "HyVLA VLM W8 absolute SPM range exceeds uint32");
    }
    TORCH_CHECK(uint64_t(input) + regions[0].second <= output ||
                    uint64_t(output) + regions[1].second <= input,
                "HyVLA VLM W8 requires disjoint SPM operands");
    const KernelId id = KernelId::HYVLA_VLM_GATEUP_SWIGLU_W8_V2;
    std::vector<GraphDdrRegisterOperandSpec> operands;
    auto add = [&](const at::Tensor& tensor, GraphDdrRegisterRole role, int lo) {
        TORCH_CHECK(rhino_lkn::RpuGetDevAddr(tensor.data_ptr()) % 256 == 0,
                    "HyVLA VLM W8 DDR alignment");
        operands.push_back({GraphDdrRegisterAbi{role, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, uint16_t(lo), uint16_t(lo+1)}, tensor});
    };
    add(gate_weight, GraphDdrRegisterRole::Int8ModelWeight, 2);
    add(gate_scale, GraphDdrRegisterRole::PerChannelScale, 20);
    add(up_weight, GraphDdrRegisterRole::Int8ModelWeight, 32);
    add(up_scale, GraphDdrRegisterRole::PerChannelScale, 34);
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id, operands);
    for (auto* tensor : {&gate_weight, &up_weight, &gate_scale, &up_scale}) {
        rpu_ddr_flush(tensor->data_ptr());
        RpuKernelGraph::active().keep_alive(*tensor);
    }
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel != nullptr, "HyVLA VLM W8 Gate/Up payload missing");
    kernel->reset_regs();
    rpu_pl_tiling::set_parallel_linear_regs(*kernel,
        rpu_pl_tiling::ParallelLinearRegisterArgs{
            input, 0, output, 0, 240, 768, 2048, 0, 8, 4096, 1536, 0, 1, 1});
    writer.write(*kernel);
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, rpu_specialized_linear::grid(id), {0,1,2,3,4,5,6,7});
}
