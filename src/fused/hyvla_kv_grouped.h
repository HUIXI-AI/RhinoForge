#pragma once
#include "rpu_helpers.h"
#include "rpu_kernel_decls.h"
#include "rpu_linear_tiling.h"
#include "rpu_specialized_linear_contracts.h"

// Two independent K/V DDR owners and SPM outputs; Q keeps its ordinary route.
inline void launch_hyvla_kv_grouped_spm(
    uint32_t input, const std::vector<at::Tensor>& weights,
    const std::vector<at::Tensor>& scales, const std::vector<uint32_t>& outputs,
    int64_t rows = 51, int cores = 8) {
    TORCH_CHECK(rows == 51 && cores == 8 && SPM_ALLOC.is_initialized() &&
                    weights.size() == 2 && scales.size() == 2 && outputs.size() == 2,
                "HyVLA grouped KV requires M51/K1024/N1024/TP8 and two operands");
    const bool w8 = weights[0].scalar_type() == at::kChar;
    for (size_t i = 0; i < 2; ++i) {
        const auto& w = weights[i];
        const auto& s = scales[i];
        TORCH_CHECK(w.defined() && w.device().type() == c10::DeviceType::PrivateUse1 &&
                        w.scalar_type() == (w8 ? at::kChar : at::kHalf) &&
                        w.is_contiguous() && w.dim() == 2 &&
                        w.size(0) == 1024 && w.size(1) == 1024 &&
                        w.device() == weights[0].device(),
                    "HyVLA grouped KV requires uniform contiguous FP16/INT8 [1024,1024]");
        TORCH_CHECK(w8 ? (s.defined() && s.device() == w.device() &&
                         s.scalar_type() == at::kHalf && s.is_contiguous() &&
                         s.dim() == 1 && s.size(0) == 1024) : !s.defined(),
                    "HyVLA grouped KV requires independent W8 FP16 [1024] scales, none for FP16");
    }
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint64_t>, 3> regions{{
        {input, 51 * 1024 * 2}, {outputs[0], 51 * 128 * 2}, {outputs[1], 51 * 128 * 2}}};
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto [address, bytes] = regions[i];
        TORCH_CHECK(address >= base && address % 256 == 0 &&
                        uint64_t(address) - base + bytes <= SpmAllocator::SPM_USABLE,
                    "HyVLA grouped KV requires complete aligned absolute SPM owners");
        for (int core = 0; core < 8; ++core)
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + address - base + bytes <=
                            uint64_t(UINT32_MAX) + 1,
                        "HyVLA grouped KV SPM range overflows on a core");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(uint64_t(address) + bytes <= regions[j].first ||
                            uint64_t(regions[j].first) + regions[j].second <= address,
                        "HyVLA grouped KV requires disjoint SPM owners");
    }
    const KernelId id = w8 ? KernelId::HYVLA_EXPERT_KV_W8_S51
                           : KernelId::HYVLA_EXPERT_KV_FP16_S51;
    const std::array<uint16_t, 2> weight_slots{2, 32}, scale_slots{20, 34};
    std::vector<GraphDdrRegisterOperandSpec> operands;
    for (size_t i = 0; i < 2; ++i) {
        for (int scale = 0; scale < (w8 ? 2 : 1); ++scale) {
            const auto& tensor = scale ? scales[i] : weights[i];
            const uint64_t address = rhino_lkn::RpuGetDevAddr(tensor.data_ptr());
            constexpr uint64_t limit = UINT64_C(1) << 40;
            TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit &&
                            uint64_t(tensor.nbytes()) <= limit - address,
                        "HyVLA grouped KV requires complete aligned shift8 DDR owners");
            const uint16_t lo = scale ? scale_slots[i] : weight_slots[i];
            operands.push_back({GraphDdrRegisterAbi{
                scale ? GraphDdrRegisterRole::PerChannelScale :
                        (w8 ? GraphDdrRegisterRole::Int8ModelWeight : GraphDdrRegisterRole::ModelWeight),
                GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                lo, uint16_t(lo + 1)}, tensor});
        }
    }
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel, "HyVLA grouped KV payload unavailable");
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id, operands);
    for (const auto& operand : operands) {
        rpu_ddr_flush(operand.tensor.data_ptr());
        RpuKernelGraph::active().keep_alive(operand.tensor);
    }
    kernel->reset_regs();
    rpu_pl_tiling::set_parallel_linear_regs(*kernel, {
        input, 0, outputs[0], 0, 51, 128, 1024, 0, 8, 2048, 256, 0, 1, uint16_t(w8 ? 1 : 2)});
    rpu_pl_tiling::set_u32_reg_pair(*kernel, 36, outputs[1]);
    writer.write(*kernel);
    const uint16_t grid = rpu_specialized_linear::contract(id).gx;
    kernel->set_regs(64, grid);
    kernel->set_regs(65, uint16_t{1});
    kernel->set_regs(66, uint16_t{1});
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {grid, uint16_t{1}, uint16_t{1}}, {0, 1, 2, 3, 4, 5, 6, 7});
}
