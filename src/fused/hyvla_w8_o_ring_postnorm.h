#pragma once
#include "rpu_helpers.h"
#include "ops/rpu_linear_tiling.h"

// Exact M51 ring partition: 6528 FP16 elements/core, paced shared ABI.
// The old residual becomes the normalized owner after the collective finishes.
inline void launch_hyvla_w8_o_ring_postnorm_spm(
    uint32_t input, const at::Tensor& weight, const at::Tensor& scale,
    uint32_t partial, uint32_t residual_norm, uint32_t raw, uint32_t gamma,
    int64_t rows, int64_t cols, int64_t norm_rows, double eps, int cores) {
    TORCH_CHECK(rows == 51 && cols == 1024 && norm_rows == 64 &&
                    eps == 1e-5 && cores == 8 && SPM_ALLOC.is_initialized(),
                "HyVLA ring postnorm requires M51/H1024/R64/eps1e-5/TP8");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint64_t>, 5> regions{{
        {partial, 51 * 1024 * 2}, {residual_norm, 64 * 1024 * 2},
        {raw, 64 * 1024 * 2}, {gamma, 1024 * 2}, {input, 51 * 256 * 2}}};
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto [address, bytes] = regions[i];
        TORCH_CHECK(address % 256 == 0 && address >= base &&
                        uint64_t(address) - base + bytes <= SpmAllocator::SPM_USABLE,
                    "HyVLA ring postnorm requires complete aligned absolute SPM ranges");
        for (int core = 0; core < 8; ++core)
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + address - base + bytes <=
                            uint64_t(UINT32_MAX) + 1,
                        "HyVLA ring postnorm SPM address overflows on a core");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(uint64_t(address) + bytes <= regions[j].first ||
                            uint64_t(regions[j].first) + regions[j].second <= address,
                        "HyVLA ring postnorm requires five disjoint owners");
    }
    TORCH_CHECK(weight.defined() && weight.is_contiguous() &&
                    weight.device().type() == c10::DeviceType::PrivateUse1 &&
                    weight.scalar_type() == at::kChar && weight.dim() == 2 &&
                    weight.size(0) == 1024 && weight.size(1) == 2048,
                "HyVLA O fusion requires exact contiguous INT8 device weight [1024,2048]");
    TORCH_CHECK(scale.defined() && scale.is_contiguous() &&
                    scale.device() == weight.device() && scale.scalar_type() == at::kHalf &&
                    scale.dim() == 1 && scale.size(0) == 1024,
                "HyVLA W8 O fusion requires owned FP16 channel scales [1024]");
    const uint64_t saddr = rhino_lkn::RpuGetDevAddr(scale.data_ptr());
    const uint64_t waddr = rhino_lkn::RpuGetDevAddr(weight.data_ptr());
    constexpr uint64_t limit = UINT64_C(1) << 40;
    TORCH_CHECK(waddr % 256 == 0 && waddr < limit && weight.nbytes() <= limit - waddr,
                "HyVLA O fusion requires a complete encodable DDR weight range");
    TORCH_CHECK(saddr % 256 == 0 && saddr < limit && scale.nbytes() <= limit - saddr,
                "HyVLA W8 O fusion requires a complete encodable DDR scale range");
    constexpr auto id = KernelId::HYVLA_EXPERT_O_RING_POSTNORM_W8_S51;
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id, {
        {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
            GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, weight},
        {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
            GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale}});
    rpu_ddr_flush(weight.data_ptr());
    rpu_ddr_flush(scale.data_ptr());
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel, "HyVLA O ring postnorm expansion kernel unavailable");
    kernel->reset_regs();
    auto u16 = [&](uint32_t r, uint32_t v) { kernel->set_regs(r, uint16_t(v)); };
    auto u32 = [&](uint32_t r, uint32_t v) { u16(r, v); u16(r + 1, v >> 16); };
    rpu_pl_tiling::set_parallel_linear_regs(*kernel,
        rpu_pl_tiling::ParallelLinearRegisterArgs{
            input, 0, partial, 0, 51, 1024, 256, 0, 8, 512, 2048, 0, 0, 1});
    writer.write(*kernel);
    u16(32, 8); u16(33, 1); u32(34, 31 * 256 * 8);
    u32(36, residual_norm - base); u16(38, 0);
    u16(40, 1024); u16(41, c10::Half(1.f / 1024).x);
    u16(42, 2048); u16(43, 8);
    u32(44, raw); u32(46, residual_norm); u32(48, gamma);
    u16(50, c10::Half(eps).x);
    u16(64, 8); u16(65, 1); u16(66, 1);
    kernel->set_regs(128, uint32_t{0});
    for (int core = 0; core < 8; ++core) {
        const uint32_t in = SPM_ALLOC.addr(core, uint32_t(partial - base)) - base;
        const uint32_t out = SPM_ALLOC.addr(core, uint32_t(raw - base)) - base;
        const std::array<uint32_t, 4> values{{in, out, uint32_t(core * 6528 * 2), 6528}};
        for (int i = 0; i < 4; ++i) {
            rpu_set_legacy_scm_u16_checked(kernel, 4096 + i * 16 + core,
                                          uint16_t(values[i]), "HyVLA ring postnorm");
            rpu_set_legacy_scm_u16_checked(kernel, 4096 + i * 16 + 8 + core,
                                          uint16_t(values[i] >> 16), "HyVLA ring postnorm");
        }
    }
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {uint16_t(8), uint16_t(1), uint16_t(1)},
                        {0, 1, 2, 3, 4, 5, 6, 7});
}
