// rpu_argmax.cpp - Argmax/Argmin kernel implementation for RPU
#include "rhino_launch_buffer.h"

#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

using namespace ::rhino_lkn;

// =============================================================================
// Helper: Select kernel ID based on parameters
// =============================================================================
static KernelId get_argmax_kernel_id(int64_t reduce_dim, int64_t C, bool select_last_index) {
    if (reduce_dim == 1) {  // reduceC
        bool is_C128 = (C <= 128);
        if (select_last_index) {
            return is_C128 ? KernelId::ARGMAX_REDUCEC_TILEN_C128_SLI
                           : KernelId::ARGMAX_REDUCEC_TILEN_SLI;
        } else {
            return is_C128 ? KernelId::ARGMAX_REDUCEC_TILEN_C128
                           : KernelId::ARGMAX_REDUCEC_TILEN;
        }
    } else {  // reduceN
        return select_last_index ? KernelId::ARGMAX_REDUCEN_TILEC_SLI
                                 : KernelId::ARGMAX_REDUCEN_TILEC;
    }
}

static KernelId get_argmin_kernel_id(int64_t reduce_dim, int64_t C, bool select_last_index) {
    if (reduce_dim == 1) {  // reduceC
        bool is_C128 = (C <= 128);
        if (select_last_index) {
            return is_C128 ? KernelId::ARGMIN_REDUCEC_TILEN_C128_SLI
                           : KernelId::ARGMIN_REDUCEC_TILEN_SLI;
        } else {
            return is_C128 ? KernelId::ARGMIN_REDUCEC_TILEN_C128
                           : KernelId::ARGMIN_REDUCEC_TILEN;
        }
    } else {  // reduceN
        return select_last_index ? KernelId::ARGMIN_REDUCEN_TILEC_SLI
                                 : KernelId::ARGMIN_REDUCEN_TILEC;
    }
}

// =============================================================================
// Kernel launch: Argmax reduce over C dimension (last dim)
// Input: [B, N, C], Output: [B, N]
// =============================================================================
static void rpu_launch_argmax_reduceC_kernel(const at::Tensor &input, at::Tensor &output,
                                             bool select_last_index) {
    TORCH_CHECK(input.dim() == 3, "argmax reduceC expects 3D input [B, N, C]");

    int64_t B = input.size(0);
    int64_t N = input.size(1);
    int64_t C = input.size(2);
    int64_t BN = B * N;

    bool is_C128 = (C <= 128);
    uint64_t x_dwidth = sizeof(c10::Half);
    uint64_t r_dwidth = sizeof(int32_t);

    // Get kernel
    KernelId kernel_id = get_argmax_kernel_id(1, C, select_last_index);
    Kernel_t* kernel = GET_KERNEL(kernel_id);
    TORCH_CHECK(kernel != nullptr, "Failed to get argmax kernel");

    // Calculate buffer sizes
    uint64_t x_bytes = B * N * C * x_dwidth;
    uint64_t r_bytes = B * N * r_dwidth;

    // Allocate SPM buffers.
    // The public wrapper contract requires workspace only for the *_C128
    // variants. Avoid allocating it for other variants, where a large C would
    // otherwise exceed the per-core SPM budget.
    std::optional<LocalSPM_t> spm_workspace;
    if (is_C128) {
        const uint64_t workspace_bytes = Align(Align(C, 16) * 256 * 2, 32);
        spm_workspace.emplace(workspace_bytes, read_write, kStride32B, 2, 0);
        RPU_CHECK_SPM(*spm_workspace, workspace_bytes);
    }
    LocalSPM_t spm_input(Align(x_bytes, 32), read_write, kStride32B, 2, 0);
    RPU_CHECK_SPM(spm_input, Align(x_bytes, 32));
    LocalSPM_t spm_output(Align(r_bytes, 32), read_write, kStride32B, 2, 0);
    RPU_CHECK_SPM(spm_output, Align(r_bytes, 32));

    // Copy input to SPM
    const c10::Half* in_ptr = input.data_ptr<c10::Half>();
    ddr_to_spm(spm_input.get_cpu_ptr(), in_ptr, x_bytes);

    // Calculate grid dimensions
    int64_t normal_blk_n = is_C128 ? 256 : 16;
    int64_t blk_cnt = CeilDiv(BN, normal_blk_n);
    int64_t last_blk_n = BN - normal_blk_n * (blk_cnt - 1);

    // Configure kernel registers
    kernel->reset_regs();

    uint32_t spm_x_base = spm_input.get_rpu_addr();
    uint32_t spm_r_base = spm_output.get_rpu_addr();

    kernel->set_regs(0, (uint16_t)normal_blk_n);
    kernel->set_regs(1, (uint16_t)last_blk_n);
    kernel->set_regs(2, (uint16_t)C);

    kernel->set_regs(4, (uint16_t)(spm_x_base & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(spm_x_base >> 16));

    kernel->set_regs(6, (uint16_t)(spm_r_base & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(spm_r_base >> 16));

    uint64_t c_stride = C * x_dwidth;
    kernel->set_regs(8, (uint16_t)(c_stride & 0xFFFF));
    kernel->set_regs(9, (uint16_t)(c_stride >> 16));

    uint64_t x_stride = normal_blk_n * C * x_dwidth;
    kernel->set_regs(10, (uint16_t)(x_stride & 0xFFFF));
    kernel->set_regs(11, (uint16_t)(x_stride >> 16));

    uint64_t r_stride = normal_blk_n * r_dwidth;
    kernel->set_regs(12, (uint16_t)(r_stride & 0xFFFF));
    kernel->set_regs(13, (uint16_t)(r_stride >> 16));

    uint16_t x_dwidth_lsu = (x_dwidth == 2) ? 1 : 0;
    kernel->set_regs(14, x_dwidth_lsu);

    uint64_t x_end_offset = (C - 16) * x_dwidth;
    kernel->set_regs(16, (uint16_t)(x_end_offset & 0xFFFF));
    kernel->set_regs(17, (uint16_t)(x_end_offset >> 16));

    uint64_t x_batch_stride = BN * C * x_dwidth;
    kernel->set_regs(18, (uint16_t)(x_batch_stride & 0xFFFF));
    kernel->set_regs(19, (uint16_t)(x_batch_stride >> 16));

    uint64_t r_batch_stride = C * r_dwidth;
    kernel->set_regs(20, (uint16_t)(r_batch_stride & 0xFFFF));
    kernel->set_regs(21, (uint16_t)(r_batch_stride >> 16));

    if (is_C128) {
        uint32_t spm_workspace_base = spm_workspace->get_rpu_addr();
        kernel->set_regs(22, (uint16_t)(spm_workspace_base & 0xFFFF));
        kernel->set_regs(23, (uint16_t)(spm_workspace_base >> 16));
    }

    // Grid dimensions
    kernel->set_regs(64, (uint16_t)blk_cnt);
    kernel->set_regs(65, (uint16_t)1);
    kernel->set_regs(66, (uint16_t)1);

    // Execute kernel
    auto* wq = GET_QUEUE(1);
    wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, 1, 1}, {0});

    // Copy output from SPM
    int32_t* out_ptr = output.data_ptr<int32_t>();
    spm_to_ddr(out_ptr, spm_output.get_cpu_ptr(), r_bytes);
}

// =============================================================================
// Kernel launch: Argmax reduce over N dimension
// Input: [B, N, C], Output: [B, C]
// =============================================================================
static void rpu_launch_argmax_reduceN_kernel(const at::Tensor &input, at::Tensor &output,
                                             bool select_last_index) {
    TORCH_CHECK(input.dim() == 3, "argmax reduceN expects 3D input [B, N, C]");

    int64_t B = input.size(0);
    int64_t N = input.size(1);
    int64_t C = input.size(2);

    uint64_t x_dwidth = sizeof(c10::Half);
    uint64_t r_dwidth = sizeof(int32_t);

    // Get kernel
    KernelId kernel_id = get_argmax_kernel_id(0, C, select_last_index);
    Kernel_t* kernel = GET_KERNEL(kernel_id);
    TORCH_CHECK(kernel != nullptr, "Failed to get argmax kernel");

    // Calculate buffer sizes
    uint64_t x_bytes = B * N * C * x_dwidth;
    uint64_t r_bytes = B * C * r_dwidth;

    // Allocate SPM buffers
    uint64_t x_base = 0;
    uint64_t r_base = Align(x_bytes, 32);

    LocalSPM_t spm_input(Align(x_bytes, 32), read_write, kStride32B, 2, 0);
    RPU_CHECK_SPM(spm_input, Align(x_bytes, 32));
    LocalSPM_t spm_output(Align(r_bytes, 32), read_write, kStride32B, 2, 0);
    RPU_CHECK_SPM(spm_output, Align(r_bytes, 32));

    // Copy input to SPM
    const c10::Half* in_ptr = input.data_ptr<c10::Half>();
    ddr_to_spm(spm_input.get_cpu_ptr(), in_ptr, x_bytes);

    // Calculate grid dimensions
    int64_t blk_cnt = CeilDiv(C, 256);
    int64_t normal_blk_c = 256;
    int64_t last_blk_c = C - normal_blk_c * (blk_cnt - 1);

    // Configure kernel registers
    kernel->reset_regs();

    uint32_t spm_x_base = spm_input.get_rpu_addr();
    uint32_t spm_r_base = spm_output.get_rpu_addr();

    kernel->set_regs(0, (uint16_t)normal_blk_c);
    kernel->set_regs(1, (uint16_t)last_blk_c);
    kernel->set_regs(2, (uint16_t)N);

    kernel->set_regs(4, (uint16_t)(spm_x_base & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(spm_x_base >> 16));

    kernel->set_regs(6, (uint16_t)(spm_r_base & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(spm_r_base >> 16));

    uint64_t c_stride = C * x_dwidth;
    kernel->set_regs(8, (uint16_t)(c_stride & 0xFFFF));
    kernel->set_regs(9, (uint16_t)(c_stride >> 16));

    uint64_t x_stride = 256 * x_dwidth;
    kernel->set_regs(10, (uint16_t)(x_stride & 0xFFFF));
    kernel->set_regs(11, (uint16_t)(x_stride >> 16));

    uint64_t r_stride = 256 * r_dwidth;
    kernel->set_regs(12, (uint16_t)(r_stride & 0xFFFF));
    kernel->set_regs(13, (uint16_t)(r_stride >> 16));

    uint16_t x_dwidth_lsu = (x_dwidth == 2) ? 1 : 0;
    kernel->set_regs(14, x_dwidth_lsu);

    uint64_t x_end_offset = (N - 1) * C * x_dwidth;
    kernel->set_regs(16, (uint16_t)(x_end_offset & 0xFFFF));
    kernel->set_regs(17, (uint16_t)(x_end_offset >> 16));

    uint64_t x_batch_stride = N * C * x_dwidth;
    kernel->set_regs(18, (uint16_t)(x_batch_stride & 0xFFFF));
    kernel->set_regs(19, (uint16_t)(x_batch_stride >> 16));

    uint64_t r_batch_stride = C * r_dwidth;
    kernel->set_regs(20, (uint16_t)(r_batch_stride & 0xFFFF));
    kernel->set_regs(21, (uint16_t)(r_batch_stride >> 16));

    // Grid dimensions
    kernel->set_regs(64, (uint16_t)blk_cnt);
    kernel->set_regs(65, (uint16_t)B);
    kernel->set_regs(66, (uint16_t)1);

    // Execute kernel
    auto* wq = GET_QUEUE(1);
    wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)B, 1}, {0});

    // Copy output from SPM
    int32_t* out_ptr = output.data_ptr<int32_t>();
    spm_to_ddr(out_ptr, spm_output.get_cpu_ptr(), r_bytes);
}

// =============================================================================
// Kernel launch: Argmin reduce over C dimension (last dim)
// =============================================================================
static void rpu_launch_argmin_reduceC_kernel(const at::Tensor &input, at::Tensor &output,
                                             bool select_last_index) {
    TORCH_CHECK(input.dim() == 3, "argmin reduceC expects 3D input [B, N, C]");

    int64_t B = input.size(0);
    int64_t N = input.size(1);
    int64_t C = input.size(2);
    int64_t BN = B * N;

    bool is_C128 = (C <= 128);
    uint64_t x_dwidth = sizeof(c10::Half);
    uint64_t r_dwidth = sizeof(int32_t);

    // Get kernel
    KernelId kernel_id = get_argmin_kernel_id(1, C, select_last_index);
    Kernel_t* kernel = GET_KERNEL(kernel_id);
    TORCH_CHECK(kernel != nullptr, "Failed to get argmin kernel");

    // Calculate buffer sizes
    uint64_t x_bytes = B * N * C * x_dwidth;
    uint64_t r_bytes = B * N * r_dwidth;

    // Allocate SPM buffers — workspace only for the C128 kernel, see the note
    // in rpu_launch_argmax_reduceC_kernel above. This is the launcher the
    // [1,32000] argmin actually reached, and the 16 MB it asked for is what
    // killed it in a cold process.
    std::optional<LocalSPM_t> spm_workspace;
    if (is_C128) {
        const uint64_t workspace_bytes = Align(Align(C, 16) * 256 * 2, 32);
        spm_workspace.emplace(workspace_bytes, read_write, kStride32B, 2, 0);
        RPU_CHECK_SPM(*spm_workspace, workspace_bytes);
    }
    LocalSPM_t spm_input(Align(x_bytes, 32), read_write, kStride32B, 2, 0);
    RPU_CHECK_SPM(spm_input, Align(x_bytes, 32));
    LocalSPM_t spm_output(Align(r_bytes, 32), read_write, kStride32B, 2, 0);
    RPU_CHECK_SPM(spm_output, Align(r_bytes, 32));

    // Copy input to SPM
    const c10::Half* in_ptr = input.data_ptr<c10::Half>();
    ddr_to_spm(spm_input.get_cpu_ptr(), in_ptr, x_bytes);

    // Calculate grid dimensions
    int64_t normal_blk_n = is_C128 ? 256 : 16;
    int64_t blk_cnt = CeilDiv(BN, normal_blk_n);
    int64_t last_blk_n = BN - normal_blk_n * (blk_cnt - 1);

    // Configure kernel registers (same as argmax)
    kernel->reset_regs();

    uint32_t spm_x_base = spm_input.get_rpu_addr();
    uint32_t spm_r_base = spm_output.get_rpu_addr();

    kernel->set_regs(0, (uint16_t)normal_blk_n);
    kernel->set_regs(1, (uint16_t)last_blk_n);
    kernel->set_regs(2, (uint16_t)C);

    kernel->set_regs(4, (uint16_t)(spm_x_base & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(spm_x_base >> 16));

    kernel->set_regs(6, (uint16_t)(spm_r_base & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(spm_r_base >> 16));

    uint64_t c_stride = C * x_dwidth;
    kernel->set_regs(8, (uint16_t)(c_stride & 0xFFFF));
    kernel->set_regs(9, (uint16_t)(c_stride >> 16));

    uint64_t x_stride = normal_blk_n * C * x_dwidth;
    kernel->set_regs(10, (uint16_t)(x_stride & 0xFFFF));
    kernel->set_regs(11, (uint16_t)(x_stride >> 16));

    uint64_t r_stride = normal_blk_n * r_dwidth;
    kernel->set_regs(12, (uint16_t)(r_stride & 0xFFFF));
    kernel->set_regs(13, (uint16_t)(r_stride >> 16));

    uint16_t x_dwidth_lsu = (x_dwidth == 2) ? 1 : 0;
    kernel->set_regs(14, x_dwidth_lsu);

    uint64_t x_end_offset = (C - 16) * x_dwidth;
    kernel->set_regs(16, (uint16_t)(x_end_offset & 0xFFFF));
    kernel->set_regs(17, (uint16_t)(x_end_offset >> 16));

    uint64_t x_batch_stride = BN * C * x_dwidth;
    kernel->set_regs(18, (uint16_t)(x_batch_stride & 0xFFFF));
    kernel->set_regs(19, (uint16_t)(x_batch_stride >> 16));

    uint64_t r_batch_stride = C * r_dwidth;
    kernel->set_regs(20, (uint16_t)(r_batch_stride & 0xFFFF));
    kernel->set_regs(21, (uint16_t)(r_batch_stride >> 16));

    if (is_C128) {
        uint32_t spm_workspace_base = spm_workspace->get_rpu_addr();
        kernel->set_regs(22, (uint16_t)(spm_workspace_base & 0xFFFF));
        kernel->set_regs(23, (uint16_t)(spm_workspace_base >> 16));
    }

    kernel->set_regs(64, (uint16_t)blk_cnt);
    kernel->set_regs(65, (uint16_t)1);
    kernel->set_regs(66, (uint16_t)1);

    auto* wq = GET_QUEUE(1);
    wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, 1, 1}, {0});

    int32_t* out_ptr = output.data_ptr<int32_t>();
    spm_to_ddr(out_ptr, spm_output.get_cpu_ptr(), r_bytes);
}

// =============================================================================
// Kernel launch: Argmin reduce over N dimension
// =============================================================================
static void rpu_launch_argmin_reduceN_kernel(const at::Tensor &input, at::Tensor &output,
                                             bool select_last_index) {
    TORCH_CHECK(input.dim() == 3, "argmin reduceN expects 3D input [B, N, C]");

    int64_t B = input.size(0);
    int64_t N = input.size(1);
    int64_t C = input.size(2);

    uint64_t x_dwidth = sizeof(c10::Half);
    uint64_t r_dwidth = sizeof(int32_t);

    KernelId kernel_id = get_argmin_kernel_id(0, C, select_last_index);
    Kernel_t* kernel = GET_KERNEL(kernel_id);
    TORCH_CHECK(kernel != nullptr, "Failed to get argmin kernel");

    uint64_t x_bytes = B * N * C * x_dwidth;
    uint64_t r_bytes = B * C * r_dwidth;

    LocalSPM_t spm_input(Align(x_bytes, 32), read_write, kStride32B, 2, 0);
    RPU_CHECK_SPM(spm_input, Align(x_bytes, 32));
    LocalSPM_t spm_output(Align(r_bytes, 32), read_write, kStride32B, 2, 0);
    RPU_CHECK_SPM(spm_output, Align(r_bytes, 32));

    const c10::Half* in_ptr = input.data_ptr<c10::Half>();
    ddr_to_spm(spm_input.get_cpu_ptr(), in_ptr, x_bytes);

    int64_t blk_cnt = CeilDiv(C, 256);
    int64_t normal_blk_c = 256;
    int64_t last_blk_c = C - normal_blk_c * (blk_cnt - 1);

    kernel->reset_regs();

    uint32_t spm_x_base = spm_input.get_rpu_addr();
    uint32_t spm_r_base = spm_output.get_rpu_addr();

    kernel->set_regs(0, (uint16_t)normal_blk_c);
    kernel->set_regs(1, (uint16_t)last_blk_c);
    kernel->set_regs(2, (uint16_t)N);

    kernel->set_regs(4, (uint16_t)(spm_x_base & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(spm_x_base >> 16));

    kernel->set_regs(6, (uint16_t)(spm_r_base & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(spm_r_base >> 16));

    uint64_t c_stride = C * x_dwidth;
    kernel->set_regs(8, (uint16_t)(c_stride & 0xFFFF));
    kernel->set_regs(9, (uint16_t)(c_stride >> 16));

    uint64_t x_stride = 256 * x_dwidth;
    kernel->set_regs(10, (uint16_t)(x_stride & 0xFFFF));
    kernel->set_regs(11, (uint16_t)(x_stride >> 16));

    uint64_t r_stride = 256 * r_dwidth;
    kernel->set_regs(12, (uint16_t)(r_stride & 0xFFFF));
    kernel->set_regs(13, (uint16_t)(r_stride >> 16));

    uint16_t x_dwidth_lsu = (x_dwidth == 2) ? 1 : 0;
    kernel->set_regs(14, x_dwidth_lsu);

    uint64_t x_end_offset = (N - 1) * C * x_dwidth;
    kernel->set_regs(16, (uint16_t)(x_end_offset & 0xFFFF));
    kernel->set_regs(17, (uint16_t)(x_end_offset >> 16));

    uint64_t x_batch_stride = N * C * x_dwidth;
    kernel->set_regs(18, (uint16_t)(x_batch_stride & 0xFFFF));
    kernel->set_regs(19, (uint16_t)(x_batch_stride >> 16));

    uint64_t r_batch_stride = C * r_dwidth;
    kernel->set_regs(20, (uint16_t)(r_batch_stride & 0xFFFF));
    kernel->set_regs(21, (uint16_t)(r_batch_stride >> 16));

    kernel->set_regs(64, (uint16_t)blk_cnt);
    kernel->set_regs(65, (uint16_t)B);
    kernel->set_regs(66, (uint16_t)1);

    auto* wq = GET_QUEUE(1);
    wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)B, 1}, {0});

    int32_t* out_ptr = output.data_ptr<int32_t>();
    spm_to_ddr(out_ptr, spm_output.get_cpu_ptr(), r_bytes);
}

// =============================================================================
// Public API: rpu_launch_argmax_kernel
// =============================================================================
void rpu_launch_argmax_kernel(const at::Tensor &input, at::Tensor &output,
                              int64_t reduce_dim, bool select_last_index) {
    if (reduce_dim == 1) {
        rpu_launch_argmax_reduceC_kernel(input, output, select_last_index);
    } else {
        rpu_launch_argmax_reduceN_kernel(input, output, select_last_index);
    }
}

// =============================================================================
// Public API: rpu_launch_argmin_kernel
// =============================================================================
void rpu_launch_argmin_kernel(const at::Tensor &input, at::Tensor &output,
                              int64_t reduce_dim, bool select_last_index) {
    if (reduce_dim == 1) {
        rpu_launch_argmin_reduceC_kernel(input, output, select_last_index);
    } else {
        rpu_launch_argmin_reduceN_kernel(input, output, select_last_index);
    }
}

// =============================================================================
// PyTorch wrapper: rpu_argmax
// =============================================================================
at::Tensor rpu_argmax(const at::Tensor &self, c10::optional<int64_t> dim,
                      bool keepdim) {
    // Check device
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
                "rpu_argmax: input must be on RPU device");

    // Convert to FP16 if needed
    at::Tensor input = self;
    if (self.scalar_type() != at::ScalarType::Half) {
        if (log_at(4)) {
            std::cout << "[RPU_ARGMAX] Converting input from " << self.scalar_type()
                      << " to Half" << std::endl;
        }
        input = self.to(at::ScalarType::Half);
    }

    // Ensure contiguous
    input = input.contiguous();

    // Handle no dim case - reduce over all elements (flatten first)
    if (!dim.has_value()) {
        // Flatten and find argmax
        at::Tensor flat = input.flatten();
        at::Tensor output = at::empty({}, at::TensorOptions()
            .dtype(at::ScalarType::Int)
            .device(self.device()));

        // Reshape to [1, 1, N] for kernel
        at::Tensor input_3d = flat.reshape({1, 1, flat.numel()});
        at::Tensor output_3d = at::empty({1, 1}, at::TensorOptions()
            .dtype(at::ScalarType::Int)
            .device(self.device()));

        rpu_launch_argmax_kernel(input_3d, output_3d, 1, false);
        return output_3d.reshape({});
    }

    int64_t reduce_dim = dim.value();
    int64_t ndim = input.dim();

    // Normalize negative dim
    if (reduce_dim < 0) {
        reduce_dim += ndim;
    }
    TORCH_CHECK(reduce_dim >= 0 && reduce_dim < ndim,
                "argmax: dim ", dim.value(), " out of range for tensor with ", ndim, " dimensions");

    // Reshape to 3D [B, N, C] format for kernel
    // The kernel expects:
    // - reduce_dim=0: reduce over N dimension, input [B, N, C], output [B, C]
    // - reduce_dim=1: reduce over C dimension, input [B, N, C], output [B, N]

    std::vector<int64_t> output_sizes;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != reduce_dim) {
            output_sizes.push_back(input.size(i));
        }
    }

    // For simplicity, we reshape input to 3D
    // Case 1: 1D input [L] -> [1, 1, L] (reduce_dim must be 0 -> reduces to scalar)
    // Case 2: 2D input [N, C] -> [1, N, C]
    // Case 3: 3D input [B, N, C] -> [B, N, C]
    // Case 4: Higher dims -> flatten leading dims

    at::Tensor input_3d;
    int64_t kernel_reduce_dim;

    if (ndim == 1) {
        // [L] -> [1, 1, L], reduce last dim
        input_3d = input.reshape({1, 1, input.size(0)});
        kernel_reduce_dim = 1;  // reduce over C
    } else if (ndim == 2) {
        // [N, C] -> [1, N, C]
        input_3d = input.reshape({1, input.size(0), input.size(1)});
        kernel_reduce_dim = reduce_dim;  // 0 or 1
    } else if (ndim == 3) {
        // Already 3D [B, N, C]
        // Kernel expects:
        //   kernel_reduce_dim=0 -> reduce over N (middle dim), output [B, C]
        //   kernel_reduce_dim=1 -> reduce over C (last dim), output [B, N]
        if (reduce_dim == 2) {
            // User wants to reduce over C (last dim)
            input_3d = input;
            kernel_reduce_dim = 1;  // reduceC
        } else if (reduce_dim == 1) {
            // User wants to reduce over N (middle dim)
            input_3d = input;
            kernel_reduce_dim = 0;  // reduceN
        } else {
            // reduce_dim == 0: reduce over B (batch dim). Kernel can't reduce
            // over batch; fall back to CPU. Mirrors the >3D path below.
            if (log_at(2)) {
                std::cout << "[RPU_ARGMAX] 3D reduce_dim=0 (batch), falling back to CPU" << std::endl;
            }
            auto cpu_input = rpu_to_cpu_zerocopy(self);
            auto cpu_result = at::argmax(cpu_input, dim, keepdim);
            return cpu_to_rpu_zerocopy(cpu_result.to(at::ScalarType::Int));
        }
    } else {
        // Higher dims: flatten leading dims into batch
        int64_t batch_size = 1;
        for (int64_t i = 0; i < ndim - 2; i++) {
            batch_size *= input.size(i);
        }
        input_3d = input.reshape({batch_size, input.size(-2), input.size(-1)});

        // Map reduce_dim to kernel's 0 (reduceN) or 1 (reduceC)
        if (reduce_dim == ndim - 1) {
            // Reduce over last dim (C)
            kernel_reduce_dim = 1;  // reduceC
        } else if (reduce_dim == ndim - 2) {
            // Reduce over second-to-last dim (N)
            kernel_reduce_dim = 0;  // reduceN
        } else {
            // Reducing over a batch dimension - complex case
            // For now, fallback to CPU for non-standard cases
            if (log_at(2)) {
                std::cout << "[RPU_ARGMAX] Complex reduction, falling back to CPU" << std::endl;
            }
            auto cpu_input = rpu_to_cpu_zerocopy(self);
            auto cpu_result = at::argmax(cpu_input, dim, keepdim);
            return cpu_to_rpu_zerocopy(cpu_result.to(at::ScalarType::Int));
        }
    }

    // Create output tensor
    std::vector<int64_t> output_3d_sizes;
    if (kernel_reduce_dim == 0) {
        output_3d_sizes = {input_3d.size(0), input_3d.size(2)};  // [B, C]
    } else {
        output_3d_sizes = {input_3d.size(0), input_3d.size(1)};  // [B, N]
    }

    at::Tensor output_3d = at::empty(output_3d_sizes, at::TensorOptions()
        .dtype(at::ScalarType::Int)
        .device(self.device()));

    // Launch kernel
    rpu_launch_argmax_kernel(input_3d, output_3d, kernel_reduce_dim, false);

    // Reshape output to match expected shape
    if (keepdim) {
        std::vector<int64_t> keepdim_sizes;
        for (int64_t i = 0; i < ndim; i++) {
            if (i == reduce_dim) {
                keepdim_sizes.push_back(1);
            } else {
                keepdim_sizes.push_back(input.size(i));
            }
        }
        return output_3d.reshape(keepdim_sizes);
    } else {
        return output_3d.reshape(output_sizes);
    }
}

// =============================================================================
// PyTorch wrapper: rpu_argmin
// =============================================================================
at::Tensor rpu_argmin(const at::Tensor &self, c10::optional<int64_t> dim,
                      bool keepdim) {
    // Same logic as argmax, just use argmin kernel
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
                "rpu_argmin: input must be on RPU device");

    at::Tensor input = self;
    if (self.scalar_type() != at::ScalarType::Half) {
        if (log_at(4)) {
            std::cout << "[RPU_ARGMIN] Converting input from " << self.scalar_type()
                      << " to Half" << std::endl;
        }
        input = self.to(at::ScalarType::Half);
    }

    input = input.contiguous();

    if (!dim.has_value()) {
        at::Tensor flat = input.flatten();
        at::Tensor input_3d = flat.reshape({1, 1, flat.numel()});
        at::Tensor output_3d = at::empty({1, 1}, at::TensorOptions()
            .dtype(at::ScalarType::Int)
            .device(self.device()));

        rpu_launch_argmin_kernel(input_3d, output_3d, 1, false);
        return output_3d.reshape({});
    }

    int64_t reduce_dim = dim.value();
    int64_t ndim = input.dim();

    if (reduce_dim < 0) {
        reduce_dim += ndim;
    }
    TORCH_CHECK(reduce_dim >= 0 && reduce_dim < ndim,
                "argmin: dim ", dim.value(), " out of range for tensor with ", ndim, " dimensions");

    std::vector<int64_t> output_sizes;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != reduce_dim) {
            output_sizes.push_back(input.size(i));
        }
    }

    at::Tensor input_3d;
    int64_t kernel_reduce_dim;

    if (ndim == 1) {
        input_3d = input.reshape({1, 1, input.size(0)});
        kernel_reduce_dim = 1;
    } else if (ndim == 2) {
        input_3d = input.reshape({1, input.size(0), input.size(1)});
        kernel_reduce_dim = reduce_dim;
    } else if (ndim == 3) {
        // Already 3D [B, N, C]
        // Kernel expects:
        //   kernel_reduce_dim=0 -> reduce over N (middle dim), output [B, C]
        //   kernel_reduce_dim=1 -> reduce over C (last dim), output [B, N]
        if (reduce_dim == 2) {
            // User wants to reduce over C (last dim)
            input_3d = input;
            kernel_reduce_dim = 1;  // reduceC
        } else if (reduce_dim == 1) {
            // User wants to reduce over N (middle dim)
            input_3d = input;
            kernel_reduce_dim = 0;  // reduceN
        } else {
            // reduce_dim == 0: reduce over B (batch dim). Kernel can't reduce
            // over batch; fall back to CPU. Mirrors the >3D path below.
            if (log_at(2)) {
                std::cout << "[RPU_ARGMIN] 3D reduce_dim=0 (batch), falling back to CPU" << std::endl;
            }
            auto cpu_input = rpu_to_cpu_zerocopy(self);
            auto cpu_result = at::argmin(cpu_input, dim, keepdim);
            return cpu_to_rpu_zerocopy(cpu_result.to(at::ScalarType::Int));
        }
    } else {
        // Higher dims: flatten leading dims into batch
        int64_t batch_size = 1;
        for (int64_t i = 0; i < ndim - 2; i++) {
            batch_size *= input.size(i);
        }
        input_3d = input.reshape({batch_size, input.size(-2), input.size(-1)});

        // Map reduce_dim to kernel's 0 (reduceN) or 1 (reduceC)
        if (reduce_dim == ndim - 1) {
            // Reduce over last dim (C)
            kernel_reduce_dim = 1;  // reduceC
        } else if (reduce_dim == ndim - 2) {
            // Reduce over second-to-last dim (N)
            kernel_reduce_dim = 0;  // reduceN
        } else {
            // Reducing over a batch dimension - complex case
            // For now, fallback to CPU for non-standard cases
            if (log_at(2)) {
                std::cout << "[RPU_ARGMIN] Complex reduction, falling back to CPU" << std::endl;
            }
            auto cpu_input = rpu_to_cpu_zerocopy(self);
            auto cpu_result = at::argmin(cpu_input, dim, keepdim);
            return cpu_to_rpu_zerocopy(cpu_result.to(at::ScalarType::Int));
        }
    }

    std::vector<int64_t> output_3d_sizes;
    if (kernel_reduce_dim == 0) {
        output_3d_sizes = {input_3d.size(0), input_3d.size(2)};
    } else {
        output_3d_sizes = {input_3d.size(0), input_3d.size(1)};
    }

    at::Tensor output_3d = at::empty(output_3d_sizes, at::TensorOptions()
        .dtype(at::ScalarType::Int)
        .device(self.device()));

    rpu_launch_argmin_kernel(input_3d, output_3d, kernel_reduce_dim, false);

    if (keepdim) {
        std::vector<int64_t> keepdim_sizes;
        for (int64_t i = 0; i < ndim; i++) {
            if (i == reduce_dim) {
                keepdim_sizes.push_back(1);
            } else {
                keepdim_sizes.push_back(input.size(i));
            }
        }
        return output_3d.reshape(keepdim_sizes);
    } else {
        return output_3d.reshape(output_sizes);
    }
}
