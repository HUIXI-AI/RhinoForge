#pragma once
#include "rpu_helpers.h"
#include "ops/rpu_linear_tiling.h"
#include "rpu_specialized_linear_contracts.h"

// Exact W8 O projection with installed core-major 16x32 weight tiles in SPM.
// The model retains the cold bundle/preload owner; scales stay in DDR.
inline void launch_hyvla_w8_o_resident_spm(
    uint32_t input, const at::Tensor& scale, uint32_t output,
    uint32_t weight_spm, int64_t rows) {
    TORCH_CHECK(rows == 51 && SPM_ALLOC.is_initialized(),
                "HyVLA W8 resident O requires M51 and initialized SPM");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint64_t>, 3> regions{{
        {input, 51 * 256 * 2}, {output, 51 * 1024 * 2}, {weight_spm, 262144}}};
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto [address, bytes] = regions[i];
        TORCH_CHECK(address % 256 == 0 && address >= base &&
                        uint64_t(address) - base + bytes <= SpmAllocator::SPM_USABLE,
                    "HyVLA W8 resident O requires complete aligned absolute SPM ranges");
        for (int core = 0; core < 8; ++core)
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + address - base + bytes <=
                            uint64_t(UINT32_MAX) + 1,
                        "HyVLA W8 resident O SPM address overflows on a core");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(uint64_t(address) + bytes <= regions[j].first ||
                            uint64_t(regions[j].first) + regions[j].second <= address,
                        "HyVLA W8 resident O requires disjoint owners");
    }
    TORCH_CHECK(scale.defined() && scale.is_contiguous() &&
                    scale.device().type() == c10::DeviceType::PrivateUse1 &&
                    scale.scalar_type() == at::kHalf && scale.dim() == 1 && scale.size(0) == 1024,
                "HyVLA W8 resident O requires owned device FP16 channel scales [1024]");
    const uint64_t saddr = rhino_lkn::RpuGetDevAddr(scale.data_ptr());
    constexpr uint64_t limit = UINT64_C(1) << 40;
    TORCH_CHECK(saddr % 256 == 0 && saddr < limit && scale.nbytes() <= limit - saddr,
                "HyVLA W8 resident O requires a complete encodable DDR scale range");
    constexpr auto id = KernelId::HYVLA_EXPERT_O_RESIDENT_W8_S51;
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id, {
        {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
            GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale}});
    rpu_ddr_flush(scale.data_ptr());
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel, "HyVLA W8 resident O expansion kernel unavailable");
    kernel->reset_regs();
    rpu_pl_tiling::set_parallel_linear_regs(*kernel,
        rpu_pl_tiling::ParallelLinearRegisterArgs{
            input, 0, output, 0, 51, 1024, 256, 0, 8, 512, 2048, 0, 0, 1});
    rpu_pl_tiling::set_u32_reg_pair(*kernel, 2, weight_spm);
    writer.write(*kernel);
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, rpu_specialized_linear::grid(id),
                        {0, 1, 2, 3, 4, 5, 6, 7});
}
