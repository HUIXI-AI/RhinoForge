// Qwen3.5 decode-only fused GDN launchers. The ABI matches the kernels in the
// current main op-lib ref. All addresses are absolute SPM addresses.
#include <ATen/ATen.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "rpu_ops.h"

using namespace ::rhino_lkn;

namespace {
std::vector<uint8_t> core_list(int num_cores) {
    std::vector<uint8_t> cores;
    cores.reserve(num_cores);
    for (int i = 0; i < num_cores; ++i) cores.push_back(static_cast<uint8_t>(i));
    return cores;
}

void set_u32(Kernel_t* kernel, int reg, uint32_t value) {
    kernel->set_regs(reg, static_cast<uint16_t>(value & 0xffff));
    kernel->set_regs(reg + 1, static_cast<uint16_t>(value >> 16));
}
}  // namespace

void rpu_launch_qwen3_5_rms_norm_gated(
    uint32_t x_addr, uint32_t z_addr, uint32_t normw_addr, uint32_t y_addr,
    int64_t M, int64_t Dv, double eps, int num_cores) {
    TORCH_CHECK(num_cores >= 1, "rms_norm_gated: num_cores must be >= 1");
    auto* kernel = GET_KERNEL(KernelId::QWEN3_5_RMS_NORM_GATED);
    TORCH_CHECK(kernel, "qwen3_5_rms_norm_gated kernel not loaded");
    const c10::Half inv_c = static_cast<c10::Half>(1.0f / static_cast<float>(Dv));
    const c10::Half eps_h = static_cast<c10::Half>(static_cast<float>(eps));
    kernel->reset_regs();
    kernel->set_regs(0, static_cast<uint16_t>(Dv));
    kernel->set_regs(1, inv_c.x);
    kernel->set_regs(2, static_cast<uint16_t>(Dv * 2));
    // Current main launch has distinct 16-bit and 32-bit overloads. Keep this
    // scalar in exactly one register; an untyped literal selects the 32-bit
    // overload and an odd-index pair starting at reg3 is rejected.
    kernel->set_regs(3, static_cast<uint16_t>(1));
    set_u32(kernel, 4, x_addr);
    set_u32(kernel, 6, y_addr);
    set_u32(kernel, 8, normw_addr);
    kernel->set_regs(10, eps_h.x);
    set_u32(kernel, 11, z_addr);
    auto* queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(num_cores > 1);
    queue->enqueu_kernel(*kernel, {static_cast<uint16_t>(M), 1, 1},
                          core_list(num_cores));
}

void rpu_launch_qwen3_5_rank1_fma(
    uint32_t state_addr, uint32_t delta_addr, uint32_t kbc_addr,
    int64_t M, int64_t Dk, int64_t Dv, int num_cores) {
    TORCH_CHECK(Dk == 128, "rank1_fma: kernel requires Dk=128");
    TORCH_CHECK(num_cores >= 1, "rank1_fma: num_cores must be >= 1");
    auto* kernel = GET_KERNEL(KernelId::QWEN3_5_RANK1_FMA);
    TORCH_CHECK(kernel, "qwen3_5_rank1_fma kernel not loaded");
    kernel->reset_regs();
    kernel->set_regs(0, static_cast<uint16_t>(Dv));
    kernel->set_regs(2, static_cast<uint16_t>(Dv * 2));
    set_u32(kernel, 4, state_addr);
    set_u32(kernel, 6, delta_addr);
    set_u32(kernel, 8, kbc_addr);
    auto* queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(num_cores > 1);
    queue->enqueu_kernel(
        *kernel, {static_cast<uint16_t>(M * Dk), 1, 1}, core_list(num_cores));
}

void rpu_launch_qwen3_5_mul_reduce_rows(
    uint32_t state_addr, uint32_t out_addr, uint32_t vec_splat_addr,
    uint32_t scratch_addr, int64_t M, int64_t Dk, int64_t Dv, int num_cores) {
    TORCH_CHECK(Dk == 128 && Dv == 128,
                "mul_reduce_rows: kernel requires Dk=Dv=128");
    TORCH_CHECK(num_cores >= 1, "mul_reduce_rows: num_cores must be >= 1");
    auto* kernel = GET_KERNEL(KernelId::QWEN3_5_MUL_REDUCE_ROWS);
    TORCH_CHECK(kernel, "qwen3_5_mul_reduce_rows kernel not loaded");

    const uint32_t outer_num = static_cast<uint32_t>(M);
    const uint32_t axis_size = static_cast<uint32_t>(Dk);
    const uint32_t inner_num = static_cast<uint32_t>(Dv);
    const uint32_t x_axis_step = inner_num * 2;
    const uint32_t x_outer_step = axis_size * inner_num * 2;
    const uint32_t y_outer_step = inner_num * 2;
    const uint32_t vec_outer_step = axis_size * 16 * 2;
    constexpr uint32_t warp_size = 16;
    constexpr uint32_t workers_per_lane = 8;
    constexpr uint32_t vlm = 400;
    const auto cdiv = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
    const uint32_t inner_v16 = std::min<uint32_t>(16, cdiv(inner_num, 16));
    const uint32_t inner_per_warp = inner_v16 * 16;
    const uint32_t axis_per_thread = std::min<uint32_t>(
        (vlm - 2 * inner_v16 - axis_size) / inner_v16, axis_size);
    const uint32_t max_outer_threads = cdiv(cdiv(outer_num, workers_per_lane), warp_size);
    const uint32_t outer_threads = std::min<uint32_t>(
        max_outer_threads, vlm / (axis_per_thread + 2) / inner_v16);
    const uint32_t outer_per_warp = outer_threads * warp_size;
    const uint32_t axis_chunks = cdiv(axis_size, axis_per_thread);
    const uint32_t grid_x = cdiv(inner_num, inner_per_warp);
    const uint32_t grid_y = cdiv(outer_num, outer_per_warp);

    kernel->reset_regs();
    set_u32(kernel, 0, state_addr);
    set_u32(kernel, 2, out_addr);
    set_u32(kernel, 4, outer_num);
    set_u32(kernel, 6, axis_size);
    set_u32(kernel, 8, inner_num);
    set_u32(kernel, 10, 0);
    set_u32(kernel, 12, x_outer_step);
    set_u32(kernel, 14, y_outer_step);
    set_u32(kernel, 16, x_axis_step);
    set_u32(kernel, 18, 2);
    kernel->set_regs(20, static_cast<uint16_t>(outer_per_warp));
    kernel->set_regs(21, static_cast<uint16_t>(inner_per_warp));
    kernel->set_regs(22, static_cast<uint16_t>(axis_per_thread));
    kernel->set_regs(23, static_cast<uint16_t>(axis_chunks));
    set_u32(kernel, 24, vec_splat_addr);
    kernel->set_regs(26, static_cast<uint16_t>(vec_outer_step & 0xffff));
    kernel->set_regs(27, static_cast<uint16_t>(vec_outer_step >> 16));
    // reg29 begins the following 32-bit SPM address, so reg28 is a single
    // 16-bit scalar rather than a 32-bit pair.
    kernel->set_regs(28, static_cast<uint16_t>(32));
    set_u32(kernel, 29, scratch_addr);
    auto* queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel, {static_cast<uint16_t>(grid_x), static_cast<uint16_t>(grid_y), 1},
        core_list(num_cores));
}
