// Bounded probe of the production zero-padding wrapper, including short rows.
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include <torch/library.h>
#include <cstring>

namespace {
at::Tensor pad_channel_probe(const at::Tensor& input, const at::Tensor& seed,
                            int64_t rows, int64_t cols, int64_t front,
                            int64_t tail, int64_t cores) {
    RpuExecutionCoordinator::require_graph_quiescent("pad channel probe");
    RpuExecutionCoordinator::check_current_thread_execution_allowed("pad channel probe");
    TORCH_CHECK(RpuKernelGraph::active().state() == RpuKernelGraph::State::PASSTHROUGH,
                "pad channel probe requires PASSTHROUGH");
    const auto valid = [](const at::Tensor& t) {
        return t.device().type() == at::kPrivateUse1 && t.scalar_type() == at::kHalf &&
            t.is_contiguous() && t.dim() == 1 && t.numel() % 8 == 0;
    };
    TORCH_CHECK(valid(input) && valid(seed) && input.device() == seed.device(),
                "pad channel probe requires flat aligned RPU FP16 buffers");
    TORCH_CHECK(rows > 0 && rows <= 128 && cols > 0 && cols <= 65536 &&
                    front >= 0 && front <= 65536 && tail >= 0 && tail <= 65536 &&
                    cores >= 1 && cores <= 8,
                "pad channel probe dimensions are out of bounds");
    const int64_t output_elements = rows * (cols + front + tail);
    TORCH_CHECK(input.numel() >= rows * cols && input.numel() <= (1 << 20) &&
                    seed.numel() >= output_elements + 8 && seed.numel() <= (1 << 20),
                "pad channel probe buffers must include the output tail canary");
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    SpmAllocator::ScopedTemporary temporary;
    const uint32_t x = SPM_ALLOC.alloc_temporary(input.numel() * sizeof(c10::Half));
    const uint32_t y = SPM_ALLOC.alloc_temporary(seed.numel() * sizeof(c10::Half));
    rpu_launch_ddr_broadcast_spm_dma_immediate(input.data_ptr<c10::Half>(), input.numel(),
                                             SPM_ALLOC.addr(0, x), cores);
    rpu_launch_ddr_broadcast_spm_dma_immediate(seed.data_ptr<c10::Half>(), seed.numel(),
                                             SPM_ALLOC.addr(0, y), cores);
    rpu_launch_pad_channel_spm(x, y, rows, cols, front, tail, cores);
    auto result = at::empty({cores, seed.numel()},
        at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
    for (int core = 0; core < cores; ++core) {
        std::memcpy(result.data_ptr<c10::Half>() + core * seed.numel(),
                    SPM_ALLOC.cpu_ptr(core, y), seed.numel() * sizeof(c10::Half));
    }
    return result;
}

TORCH_LIBRARY_FRAGMENT(rpu, m) {
    m.def("test_pad_channel(Tensor input, Tensor seed, int rows, int cols, "
          "int front, int tail, int cores=8) -> Tensor", TORCH_FN(pad_channel_probe));
}
}  // namespace
