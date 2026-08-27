// rpu_reduce.cpp - Reduce operations for RPU backend (DDR version)
#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include <c10/util/Half.h>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace ::rhino_lkn;

// Host launch constants.
#define WARP_SIZE 16
#define WARP_NUM 8
#define VLM_ENTRY_NUM 384

// Helper: convert float to uint32 bit representation
static inline uint32_t float_to_bits(float f) {
    union { float f; uint32_t u; } u;
    u.f = f;
    return u.u;
}

void rpu_launch_reduce_mean_last_dim_kernel(const at::Tensor &input,
                                             at::Tensor &output,
                                             int64_t axis) {
    size_t dwidth = sizeof(c10::Half);
    auto shape = input.sizes();
    int ndim = shape.size();

    // Compute dimensions
    int64_t axis_size = shape[axis];
    int64_t outer_num = 1;
    for (int i = 0; i < axis; i++) {
        outer_num *= shape[i];
    }
    int64_t inner_num = 1;
    for (int i = axis + 1; i < ndim; i++) {
        inner_num *= shape[i];
    }

    // For last_dim reduction
    int64_t inner_num_per_wrp = inner_num;

    int64_t max_axis_v16_per_thd = 16;
    int64_t axis_v16_per_thd = std::min(max_axis_v16_per_thd, CeilDiv(axis_size, (int64_t)16));
    int64_t axis_size_per_thd = axis_v16_per_thd * 16;

    int64_t max_outer_size_per_thd = CeilDiv(CeilDiv(outer_num, (int64_t)WARP_NUM), (int64_t)WARP_SIZE);
    int64_t outer_num_per_thd = std::min(max_outer_size_per_thd,
                                          (int64_t)(VLM_ENTRY_NUM / (axis_size_per_thd + 2)));
    int64_t outer_num_per_wrp = outer_num_per_thd * WARP_SIZE;

    int64_t axis_chunk_num = CeilDiv(axis_size, axis_size_per_thd);

    // Byte steps
    int64_t x_axis_byte_step = inner_num * dwidth;
    int64_t y_axis_byte_step = inner_num * dwidth;
    int64_t x_outer_byte_step = axis_size * x_axis_byte_step;
    int64_t y_outer_byte_step = y_axis_byte_step;  // keepdim=false: no axis expansion

    // 1/axis_size for mean calculation (fp32)
    float one_cth = 1.0f / (float)axis_size;
    uint32_t one_cth_bits = float_to_bits(one_cth);

    // Get DDR kernel (O(1) 数组索引)
    Kernel_t* kernel = GET_KERNEL(KernelId::REDUCE_MEAN_LAST_DIM_DDR);
    TORCH_CHECK(kernel != nullptr, "Failed to get reduce_mean_last_dim_ddr kernel");
    kernel->reset_regs();

    auto* wq = GET_QUEUE(1);

    // Get tensor data pointers
    c10::Half *in_ptr = input.data_ptr<c10::Half>();
    c10::Half *out_ptr = output.data_ptr<c10::Half>();

    // Flush DDR
    rpu_ddr_flush(in_ptr);
    rpu_ddr_flush(out_ptr);

    // Get DDR device addresses (v128 = 256B units)
    uint64_t x_addr_v128 = RpuGetDevAddr(in_ptr) >> 8;
    uint64_t y_addr_v128 = RpuGetDevAddr(out_ptr) >> 8;

    // Set registers according to the host launch ABI.
    kernel->set_regs(0, (uint16_t)(x_addr_v128 & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(x_addr_v128 >> 16));
    kernel->set_regs(2, (uint16_t)(y_addr_v128 & 0xFFFF));
    kernel->set_regs(3, (uint16_t)(y_addr_v128 >> 16));

    kernel->set_regs(4, (uint16_t)(outer_num & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(outer_num >> 16));
    kernel->set_regs(6, (uint16_t)(axis_size & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(axis_size >> 16));
    kernel->set_regs(8, (uint16_t)(inner_num & 0xFFFF));
    kernel->set_regs(9, (uint16_t)(inner_num >> 16));

    kernel->set_regs(10, (uint16_t)(one_cth_bits & 0xFFFF));
    kernel->set_regs(11, (uint16_t)(one_cth_bits >> 16));

    kernel->set_regs(12, (uint16_t)(x_outer_byte_step & 0xFFFF));
    kernel->set_regs(13, (uint16_t)(x_outer_byte_step >> 16));
    kernel->set_regs(14, (uint16_t)(y_outer_byte_step & 0xFFFF));
    kernel->set_regs(15, (uint16_t)(y_outer_byte_step >> 16));
    kernel->set_regs(16, (uint16_t)(x_axis_byte_step & 0xFFFF));
    kernel->set_regs(17, (uint16_t)(x_axis_byte_step >> 16));
    kernel->set_regs(18, (uint16_t)(y_axis_byte_step & 0xFFFF));
    kernel->set_regs(19, (uint16_t)(y_axis_byte_step >> 16));

    kernel->set_regs(20, (uint16_t)outer_num_per_wrp);
    kernel->set_regs(21, (uint16_t)inner_num_per_wrp);
    kernel->set_regs(22, (uint16_t)axis_size_per_thd);
    kernel->set_regs(23, (uint16_t)axis_chunk_num);

    // Grid dimensions
    uint16_t grid_x = (uint16_t)CeilDiv(inner_num, inner_num_per_wrp);
    uint16_t grid_y = (uint16_t)CeilDiv(outer_num, outer_num_per_wrp);
    uint16_t grid_z = 1;

    wq->enqueu_kernel(*kernel, {grid_x, grid_y, grid_z}, {0});

    // Flush output
    rpu_ddr_flush(out_ptr);
}

// PyTorch mean.dim wrapper
at::Tensor rpu_mean_dim(const at::Tensor &self, at::OptionalIntArrayRef dim,
                        bool keepdim, c10::optional<at::ScalarType> dtype) {
    // Check device
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
                "rpu_mean_dim: input must be on RPU device");

    // Only support fp16
    if (self.scalar_type() != at::kHalf) {
        auto cpu_self = rpu_to_cpu_zerocopy(self);
        auto cpu_out = at::mean(cpu_self, dim, keepdim, dtype);
        return cpu_out.to(self.device());
    }

    // Get dimensions to reduce
    std::vector<int64_t> reduce_dims;
    if (dim.has_value()) {
        for (auto d : dim.value()) {
            int64_t nd = d < 0 ? d + self.dim() : d;
            reduce_dims.push_back(nd);
        }
    } else {
        // Reduce all dimensions
        for (int64_t i = 0; i < self.dim(); i++) {
            reduce_dims.push_back(i);
        }
    }

    // Currently only support single last dimension reduction
    if (reduce_dims.size() == 1 && reduce_dims[0] == self.dim() - 1) {
        // Compute output shape
        std::vector<int64_t> out_shape;
        for (int64_t i = 0; i < self.dim(); i++) {
            if (i == reduce_dims[0]) {
                if (keepdim) {
                    out_shape.push_back(1);
                }
            } else {
                out_shape.push_back(self.size(i));
            }
        }

        // Allocate output
        auto output = rpu_empty_strided(out_shape, compute_contiguous_strides(out_shape),
                                        self.scalar_type(), self.layout(), self.device(), false);

        // Launch kernel
        rpu_launch_reduce_mean_last_dim_kernel(self.contiguous(), output, reduce_dims[0]);

        return output;
    }

    // Fallback for unsupported cases
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_out = at::mean(cpu_self, dim, keepdim, dtype);
    return cpu_out.to(self.device());
}

// =============================================================================
// SPM sum over rows (axis 0): [rows, cols<256] fp16 → [cols], in ONE launch.
// Kernel "reduce_sum_non_last_dim_out_seg". Host ABI is Sum over axis 0 with
// WARP_SIZE=16 and VLM_ENTRIES_FP16=400.
// Addresses are absolute 32-bit SPM (regs 0/1 = in, 2/3 = out), like l2norm.
// =============================================================================
void rpu_launch_reduce_sum_rows_spm_kernel(uint32_t in_spm_addr,
                                           uint32_t out_spm_addr,
                                           int64_t rows, int64_t cols,
                                           int num_cores) {
    TORCH_CHECK(cols > 0 && cols < 256,
                "reduce_sum_rows: cols must be in (0,256) (out_seg variant)");
    TORCH_CHECK(rows > 0, "reduce_sum_rows: rows must be positive");

    const uint32_t axis_size = (uint32_t)rows;   // reduced dim (axis 0)
    const uint32_t outer_num = 1;
    const uint32_t inner_num = (uint32_t)cols;   // kept dim (last, < 256)
    const uint32_t dw = 2;                        // fp16

    const uint32_t in_axis_step   = inner_num * dw;
    const uint32_t in_outer_step  = axis_size * inner_num * dw;
    const uint32_t out_axis_step  = inner_num * dw;
    const uint32_t out_outer_step = inner_num * dw;   // axis==0, outer size 1
    const uint32_t one_cth_bits   = float_to_bits(1.0f / (float)axis_size);

    // Partition for the inner_num < 256 branch.
    constexpr uint32_t WS = 16, WNPL = 8, VLM = 400;
    auto cdiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
    const uint32_t inner_v16 = std::min<uint32_t>(16, cdiv(inner_num, 16));
    const uint32_t inner_num_per_wrp = inner_v16 * 16;
    const uint32_t axis_size_per_thd =
        std::min<uint32_t>(VLM / inner_v16 - 2, axis_size);
    const uint32_t max_outer_thd = cdiv(cdiv(outer_num, WNPL), WS);
    const uint32_t outer_thd = std::min<uint32_t>(
        max_outer_thd, VLM / (axis_size_per_thd + 2) / inner_v16);
    const uint32_t outer_num_per_wrp = outer_thd * WS;
    const uint32_t axis_chunk_num = cdiv(axis_size, axis_size_per_thd);
    const uint32_t grid_x = cdiv(inner_num, inner_num_per_wrp);
    const uint32_t grid_y = cdiv(outer_num, outer_num_per_wrp);

    Kernel_t* kernel = GET_KERNEL(KernelId::REDUCE_SUM_NON_LAST_DIM_OUT_SEG);
    TORCH_CHECK(kernel != nullptr,
                "Failed to get reduce_sum_non_last_dim_out_seg kernel");
    kernel->reset_regs();
    auto u32 = [&](int i, uint32_t v) {
        kernel->set_regs(i, (uint16_t)(v & 0xFFFF));
        kernel->set_regs(i + 1, (uint16_t)(v >> 16));
    };
    u32(0, in_spm_addr);
    u32(2, out_spm_addr);
    u32(4, outer_num);
    u32(6, axis_size);
    u32(8, inner_num);
    u32(10, one_cth_bits);
    u32(12, in_outer_step);
    u32(14, out_outer_step);
    u32(16, in_axis_step);
    u32(18, out_axis_step);
    kernel->set_regs(20, (uint16_t)outer_num_per_wrp);
    kernel->set_regs(21, (uint16_t)inner_num_per_wrp);
    kernel->set_regs(22, (uint16_t)axis_size_per_thd);
    kernel->set_regs(23, (uint16_t)axis_chunk_num);

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {(uint16_t)grid_x, (uint16_t)grid_y, (uint16_t)1},
                      cores);
}
