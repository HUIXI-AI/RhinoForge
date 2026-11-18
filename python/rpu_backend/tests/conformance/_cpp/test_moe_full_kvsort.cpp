// Immediate conformance for the production full-sort launcher. All SPM
// operands have real initialized guards and the actual reported capacities.
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

#include <ATen/record_function.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> rpu_test_moe_full_kvsort(
    const at::Tensor& input_keys, const at::Tensor& input_values, bool descending) {
    RECORD_FUNCTION("rpu::test_moe_full_kvsort", {});
    TORCH_CHECK(RpuKernelGraph::active().state() == RpuKernelGraph::State::PASSTHROUGH,
                "full KVSort conformance requires PASSTHROUGH");
    TORCH_CHECK(input_keys.device().type() == at::kPrivateUse1 &&
                    input_keys.scalar_type() == at::kHalf && input_keys.dim() == 2 &&
                    input_keys.is_contiguous(),
                "full KVSort test: keys must be contiguous RPU FP16 [cores,C]");
    TORCH_CHECK(input_values.device() == input_keys.device() &&
                    input_values.scalar_type() == at::kShort &&
                    input_values.is_contiguous() && input_values.sizes() == input_keys.sizes(),
                "full KVSort test: values must be matching contiguous RPU raw int16 [cores,C]");
    const int64_t cores = input_keys.size(0), c = input_keys.size(1);
    TORCH_CHECK(cores >= 1 && cores <= 8, "full KVSort test: cores must be in [1,8]");
    const auto [in_elements, out_elements] = rpu_moe_kvsort_fp16_spm_sizes(c);
    const auto keys_cpu = input_keys.cpu();
    const auto values_cpu = input_values.cpu();
    TORCH_CHECK(at::isfinite(keys_cpu).all().item<bool>(),
                "full KVSort test: keys must be finite FP16 values");
    constexpr int64_t guard_elements = 16;  // 32 bytes before and after every operand.
    constexpr int16_t canary = 0x5aa5;
    const std::array<int64_t, 4> extents{{in_elements, in_elements, out_elements, out_elements}};
    std::array<at::Tensor, 4> staging;
    std::array<at::Tensor, 4> transport;
    for (size_t operand = 0; operand < extents.size(); ++operand) {
        staging[operand] = at::full({cores, extents[operand] + 2 * guard_elements}, canary,
            at::TensorOptions().dtype(at::kShort).device(at::kCPU));
        if (operand < 2) {
            const auto& source = operand == 0 ? keys_cpu : values_cpu;
            for (int64_t core = 0; core < cores; ++core)
                std::memcpy(staging[operand].data_ptr<int16_t>() +
                                core * (extents[operand] + 2 * guard_elements) + guard_elements,
                            static_cast<const uint16_t*>(source.data_ptr()) + core * c,
                            c * sizeof(uint16_t));
        }
        transport[operand] = staging[operand].to(input_keys.device());
    }
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    using AR = SpmAllocator::AllocRequest;
    const auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{(in_elements + 2 * guard_elements) * 2, 1, 3},
        AR{(in_elements + 2 * guard_elements) * 2, 1, 3},
        AR{(out_elements + 2 * guard_elements) * 2, 1, 3},
        AR{(out_elements + 2 * guard_elements) * 2, 1, 3},
    });
    try {
        for (size_t i = 0; i < extents.size(); ++i)
            rpu_launch_ddr_scatter_spm_dma_immediate(
                reinterpret_cast<c10::Half*>(transport[i].data_ptr<int16_t>()),
                extents[i] + 2 * guard_elements, (extents[i] + 2 * guard_elements) * 2,
                SPM_ALLOC.addr(0, offsets[i]), static_cast<int>(cores));
        rpu_launch_moe_kvsort_fp16_spm_kernel(
            SPM_ALLOC.addr(0, offsets[0]) + guard_elements * 2,
            SPM_ALLOC.addr(0, offsets[1]) + guard_elements * 2,
            SPM_ALLOC.addr(0, offsets[2]) + guard_elements * 2,
            SPM_ALLOC.addr(0, offsets[3]) + guard_elements * 2,
            c, descending, static_cast<int>(cores));
        std::array<at::Tensor, 4> dumps;
        for (size_t i = 0; i < extents.size(); ++i) {
            rpu_launch_spm_scatter_ddr_dma_immediate(
                SPM_ALLOC.addr(0, offsets[i]),
                reinterpret_cast<c10::Half*>(transport[i].data_ptr<int16_t>()),
                extents[i] + 2 * guard_elements, (extents[i] + 2 * guard_elements) * 2,
                static_cast<int>(cores));
            dumps[i] = transport[i].cpu();
            const auto* words = dumps[i].data_ptr<int16_t>();
            for (int64_t core = 0; core < cores; ++core) {
                const auto* row = words + core * (extents[i] + 2 * guard_elements);
                for (int64_t lane = 0; lane < guard_elements; ++lane)
                    TORCH_CHECK(row[lane] == canary && row[guard_elements + extents[i] + lane] == canary,
                                "full KVSort test: SPM guard_elements changed for operand ", i,
                                " core ", core, " lane ", lane);
                if (i < 2) {
                    const auto& source = i == 0 ? keys_cpu : values_cpu;
                    TORCH_CHECK(std::memcmp(row + guard_elements,
                        static_cast<const uint16_t*>(source.data_ptr()) + core * c,
                        c * sizeof(uint16_t)) == 0, "full KVSort test: logical input changed");
                    const uint16_t pad = i == 0 ? (descending ? 0xfc00 : 0x7c00) : 0;
                    for (int64_t lane = c; lane < in_elements; ++lane)
                        TORCH_CHECK(static_cast<uint16_t>(row[guard_elements + lane]) == pad,
                                    "full KVSort test: input tail not initialized");
                }
            }
        }
        auto output_keys = at::empty({cores, c}, keys_cpu.options());
        auto output_values = at::empty({cores, c}, values_cpu.options());
        for (int64_t core = 0; core < cores; ++core) {
            std::memcpy(static_cast<uint16_t*>(output_keys.data_ptr()) + core * c,
                dumps[2].data_ptr<int16_t>() + core * (out_elements + 2 * guard_elements) + guard_elements, c * 2);
            std::memcpy(output_values.data_ptr<int16_t>() + core * c,
                dumps[3].data_ptr<int16_t>() + core * (out_elements + 2 * guard_elements) + guard_elements, c * 2);
        }
        SPM_ALLOC.reset_temporary();
        return {output_keys, output_values};
    } catch (...) {
        SPM_ALLOC.reset_temporary();
        throw;
    }
}
