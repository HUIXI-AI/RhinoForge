// Contiguous FP16 indexed Gather for Qwen3.5 MoE routing.
//

#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;

namespace {

void set_reg_u32(Kernel_t* kernel, int lo_reg, uint32_t value) {
    kernel->set_regs(lo_reg, static_cast<uint16_t>(value & 0xFFFF));
    kernel->set_regs(lo_reg + 1, static_cast<uint16_t>(value >> 16));
}

size_t align16(size_t value) {
    return (value + 15) & ~size_t(15);
}

size_t align256(size_t value) {
    return (value + 255) & ~size_t(255);
}

uint32_t alloc_spm_bytes(size_t bytes) {
    return SPM_ALLOC.alloc_temporary(align256(bytes));
}

uint32_t spm_local_offset(uint32_t unified_address, const char* name) {
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "gather_fp16: SPM allocator must be initialized");
    const uint32_t base = SPM_ALLOC.addr(0, 0);
    TORCH_CHECK(unified_address >= base &&
                    static_cast<uint64_t>(unified_address - base) <
                        SpmAllocator::SPM_TOTAL,
                "gather_fp16: ", name,
                " must be a core-0 unified SPM address, got ", unified_address,
                " base=", base);
    return unified_address - base;
}

struct GatherParams {
    uint32_t indices_num;
    uint32_t axis_size;
    uint32_t inner_num;
    uint32_t input_outer_byte_step;
    uint32_t output_outer_byte_step;
    uint32_t input_axis_byte_step;
    uint32_t output_axis_byte_step;
    uint16_t grid_x;
    uint16_t grid_y;
    uint16_t grid_z;
};

GatherParams make_gather_params(int64_t input_rows,
                                int64_t input_cols,
                                int64_t axis,
                                int64_t indices_num,
                                int64_t num_cores) {
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "gather_fp16: num_cores must be in [1,8], got ", num_cores);
    TORCH_CHECK(input_rows > 0 &&
                    input_rows <= std::numeric_limits<uint32_t>::max() &&
                    input_cols > 0 &&
                    input_cols <= std::numeric_limits<uint32_t>::max(),
                "gather_fp16: input rows/cols must fit u32, got ", input_rows,
                "x", input_cols);
    TORCH_CHECK(axis == 0 || axis == 1,
                "gather_fp16: axis must be 0 or 1, got ", axis);
    TORCH_CHECK(indices_num > 0 &&
                    indices_num <= std::numeric_limits<uint32_t>::max(),
                "gather_fp16: indices_num must fit u32, got ", indices_num);

    const uint64_t element_bytes = sizeof(uint16_t);
    const uint64_t input_row_bytes =
        static_cast<uint64_t>(input_cols) * element_bytes;
    const uint64_t input_bytes =
        static_cast<uint64_t>(input_rows) * input_row_bytes;
    const uint64_t output_row_bytes = axis == 0
        ? static_cast<uint64_t>(input_cols) * element_bytes
        : static_cast<uint64_t>(indices_num) * element_bytes;
    const uint64_t output_outer_bytes = axis == 0
        ? static_cast<uint64_t>(indices_num) * output_row_bytes
        : output_row_bytes;
    TORCH_CHECK(input_bytes <= std::numeric_limits<uint32_t>::max() &&
                    output_outer_bytes <=
                        std::numeric_limits<uint32_t>::max(),
                "gather_fp16: contiguous byte strides must fit u32");

    const uint64_t outer_num = axis == 0 ? 1 : input_rows;
    const uint64_t axis_size = axis == 0 ? input_rows : input_cols;
    const uint64_t inner_num = axis == 0 ? input_cols : 1;
    const uint64_t input_axis_byte_step = axis == 0 ? input_row_bytes : 2;
    const uint64_t output_axis_byte_step = input_axis_byte_step;
    const uint64_t input_outer_byte_step =
        axis == 0 ? input_bytes : input_row_bytes;
    const uint64_t output_outer_byte_step = output_outer_bytes;
    const uint64_t inner_num_per_warp = indices_num == 1 ? 16640 : 256;
    const uint64_t grid_x =
        (inner_num + inner_num_per_warp - 1) / inner_num_per_warp;
    const uint64_t grid_y = (static_cast<uint64_t>(indices_num) + 127) / 128;
    TORCH_CHECK(grid_x <= std::numeric_limits<uint16_t>::max() &&
                    grid_y <= std::numeric_limits<uint16_t>::max() &&
                    outer_num <= std::numeric_limits<uint16_t>::max(),
                "gather_fp16: grid dimensions must fit u16, got ", grid_x,
                "x", grid_y, "x", outer_num);

    return {
        static_cast<uint32_t>(indices_num),
        static_cast<uint32_t>(axis_size),
        static_cast<uint32_t>(inner_num),
        static_cast<uint32_t>(input_outer_byte_step),
        static_cast<uint32_t>(output_outer_byte_step),
        static_cast<uint32_t>(input_axis_byte_step),
        static_cast<uint32_t>(output_axis_byte_step),
        static_cast<uint16_t>(grid_x),
        static_cast<uint16_t>(grid_y),
        static_cast<uint16_t>(outer_num),
    };
}

}  // namespace

void rpu_launch_gather_fp16_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t indices_spm_addr,
    uint32_t output_spm_addr,
    int64_t input_rows,
    int64_t input_cols,
    int64_t axis,
    int64_t indices_num,
    int num_cores) {
    const GatherParams params = make_gather_params(
        input_rows, input_cols, axis, indices_num, num_cores);
    const KernelId kernel_id = indices_num == 1
        ? KernelId::GATHER_SCALAR_FP16
        : KernelId::GATHER_FP16;
    Kernel_t* kernel = GET_KERNEL(kernel_id);
    TORCH_CHECK(kernel != nullptr,
                "gather_fp16: ", indices_num == 1 ? "gather_scalar" : "gather",
                " is absent from the active oplib");
    kernel->reset_regs();
    set_reg_u32(kernel, 0, spm_local_offset(input_spm_addr, "input"));
    set_reg_u32(kernel, 2, spm_local_offset(indices_spm_addr, "indices"));
    set_reg_u32(kernel, 4, spm_local_offset(output_spm_addr, "output"));
    set_reg_u32(kernel, 6, params.indices_num);
    set_reg_u32(kernel, 8, params.axis_size);
    set_reg_u32(kernel, 10, params.inner_num);
    set_reg_u32(kernel, 12, params.input_outer_byte_step);
    set_reg_u32(kernel, 14, params.output_outer_byte_step);
    set_reg_u32(kernel, 16, params.input_axis_byte_step);
    set_reg_u32(kernel, 18, params.output_axis_byte_step);
    kernel->set_regs(20, static_cast<uint16_t>(0));
    kernel->set_regs(21, static_cast<uint16_t>(0));
    kernel->set_regs(22, static_cast<uint16_t>(0));
    kernel->set_regs(64, params.grid_x);
    kernel->set_regs(65, params.grid_y);
    kernel->set_regs(66, params.grid_z);

    auto* queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    cores.reserve(num_cores);
    for (int core = 0; core < num_cores; ++core) {
        cores.push_back(static_cast<uint8_t>(core));
    }
    queue->enqueu_kernel(
        *kernel, {params.grid_x, params.grid_y, params.grid_z}, cores);
}

// Conformance-only CPU staging for contiguous 2D FP16 Gather.
at::Tensor rpu_gather_fp16_spm_test(
    const at::Tensor& input,
    const at::Tensor& indices,
    int64_t axis,
    int64_t num_cores) {
    RECORD_FUNCTION("rpu::gather_fp16_spm_test", {});
    TORCH_CHECK(input.device().is_cpu() && indices.device().is_cpu(),
                "gather_fp16_spm_test: inputs must be CPU staging tensors");
    TORCH_CHECK(input.scalar_type() == at::kHalf && input.dim() == 2 &&
                    input.is_contiguous(),
                "gather_fp16_spm_test: input must be contiguous FP16 [D0,D1]");
    TORCH_CHECK(indices.scalar_type() == at::kInt && indices.dim() == 1 &&
                    indices.is_contiguous(),
                "gather_fp16_spm_test: indices must be contiguous I32 [I]");

    const int64_t input_rows = input.size(0);
    const int64_t input_cols = input.size(1);
    const int64_t indices_num = indices.numel();
    make_gather_params(input_rows, input_cols, axis, indices_num, num_cores);
    const int32_t* index_data = indices.data_ptr<int32_t>();
    const int64_t axis_size = axis == 0 ? input_rows : input_cols;
    for (int64_t i = 0; i < indices_num; ++i) {
        TORCH_CHECK(index_data[i] >= 0 && index_data[i] < axis_size,
                    "gather_fp16_spm_test: index out of range at ", i,
                    ": ", index_data[i], " for axis size ", axis_size);
    }

    const int64_t output_rows = axis == 0 ? indices_num : input_rows;
    const int64_t output_cols = axis == 0 ? input_cols : indices_num;
    const size_t input_bytes =
        static_cast<size_t>(input_rows * input_cols) * sizeof(uint16_t);
    const size_t indices_bytes =
        static_cast<size_t>(indices_num) * sizeof(int32_t);
    const size_t output_bytes =
        static_cast<size_t>(output_rows * output_cols) * sizeof(uint16_t);
    const size_t input_transport_bytes = align16(input_bytes);
    const size_t indices_transport_bytes = align16(indices_bytes);
    const size_t output_transport_bytes = align16(output_bytes);
    std::vector<uint8_t> input_transport(input_transport_bytes, 0);
    std::vector<uint8_t> indices_transport(indices_transport_bytes, 0);
    std::memcpy(input_transport.data(), input.data_ptr(), input_bytes);
    std::memcpy(indices_transport.data(), indices.data_ptr(), indices_bytes);

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    SPM_ALLOC.reset_temporary();
    const uint32_t input_off = alloc_spm_bytes(input_bytes);
    const uint32_t indices_off = alloc_spm_bytes(indices_bytes);
    const uint32_t output_off = alloc_spm_bytes(output_bytes);

    try {
        for (int core = 0; core < num_cores; ++core) {
            std::memcpy(SPM_ALLOC.cpu_ptr(core, input_off),
                        input_transport.data(), input_transport_bytes);
            std::memcpy(SPM_ALLOC.cpu_ptr(core, indices_off),
                        indices_transport.data(), indices_transport_bytes);
        }
        rpu_launch_gather_fp16_spm_kernel(
            SPM_ALLOC.addr(0, input_off), SPM_ALLOC.addr(0, indices_off),
            SPM_ALLOC.addr(0, output_off), input_rows, input_cols, axis,
            indices_num, static_cast<int>(num_cores));

        at::Tensor output = at::empty(
            {num_cores, output_rows, output_cols},
            at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
        std::vector<uint8_t> output_transport(output_transport_bytes);
        for (int64_t core = 0; core < num_cores; ++core) {
            std::memcpy(output_transport.data(),
                        SPM_ALLOC.cpu_ptr(static_cast<int>(core), output_off),
                        output_transport_bytes);
            std::memcpy(static_cast<char*>(output.data_ptr()) +
                            core * output_bytes,
                        output_transport.data(), output_bytes);
        }
        SPM_ALLOC.reset_temporary();
        return output;
    } catch (...) {
        SPM_ALLOC.reset_temporary();
        throw;
    }
}
