// Qwen3.5 MoE route recombination with in-place ScatterNd(add).
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

constexpr uint32_t kNormalBlockN = 2048;
constexpr uint16_t kFp16AddValuMode = 0x0092;

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
                "scatternd_add: SPM allocator must be initialized");
    const uint32_t base = SPM_ALLOC.addr(0, 0);
    TORCH_CHECK(unified_address >= base &&
                    static_cast<uint64_t>(unified_address - base) <
                        SpmAllocator::SPM_TOTAL,
                "scatternd_add: ", name,
                " must be a core-0 unified SPM address, got ", unified_address,
                " base=", base);
    return unified_address - base;
}

struct ScatterAddParams {
    uint16_t last_block_n;
    uint16_t row_size;
    uint32_t indices_stride;
    uint32_t updates_stride;
    uint32_t row_bytes;
    uint32_t num_indices;
    uint16_t grid_x;
};

ScatterAddParams make_scatter_add_params(int64_t data_rows,
                                         int64_t num_indices,
                                         int64_t row_size,
                                         int64_t num_cores) {
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "scatternd_add: num_cores must be in [1,8], got ", num_cores);
    TORCH_CHECK(data_rows > 0 && data_rows <= 65536,
                "scatternd_add: T must be in [1,65536] for U16 indices, got ",
                data_rows);
    TORCH_CHECK(num_indices > 0 &&
                    num_indices <=
                        static_cast<int64_t>(
                            std::numeric_limits<uint16_t>::max()) *
                            kNormalBlockN,
                "scatternd_add: ceil(N/2048) must fit u16, got N=",
                num_indices);
    TORCH_CHECK(row_size > 0 &&
                    row_size <= std::numeric_limits<uint16_t>::max(),
                "scatternd_add: H must fit u16, got ", row_size);

    const uint64_t grid_x =
        (static_cast<uint64_t>(num_indices) + kNormalBlockN - 1) /
        kNormalBlockN;
    const uint64_t last_block_n =
        num_indices - (grid_x - 1) * kNormalBlockN;
    const uint64_t indices_stride = kNormalBlockN * sizeof(uint16_t);
    const uint64_t updates_stride =
        kNormalBlockN * static_cast<uint64_t>(row_size) * sizeof(uint16_t);
    const uint64_t row_bytes =
        static_cast<uint64_t>(row_size) * sizeof(uint16_t);
    TORCH_CHECK(updates_stride <= std::numeric_limits<uint32_t>::max(),
                "scatternd_add: update block byte stride must fit u32");

    return {
        static_cast<uint16_t>(last_block_n),
        static_cast<uint16_t>(row_size),
        static_cast<uint32_t>(indices_stride),
        static_cast<uint32_t>(updates_stride),
        static_cast<uint32_t>(row_bytes),
        static_cast<uint32_t>(num_indices),
        static_cast<uint16_t>(grid_x),
    };
}

}  // namespace

void rpu_launch_scatternd_add_u16_fp16_spm_kernel(
    uint32_t data_spm_addr,
    uint32_t indices_spm_addr,
    uint32_t updates_spm_addr,
    int64_t data_rows,
    int64_t num_indices,
    int64_t row_size,
    int num_cores) {
    const ScatterAddParams params = make_scatter_add_params(
        data_rows, num_indices, row_size, num_cores);
    Kernel_t* kernel = GET_KERNEL(KernelId::SCATTERND_ADD_INT16_FP16);
    TORCH_CHECK(kernel != nullptr,
                "scatternd_add: scatternd_int16_reduction is absent from the "
                "active oplib");
    kernel->reset_regs();
    kernel->set_regs(0, static_cast<uint16_t>(kNormalBlockN));
    kernel->set_regs(1, params.last_block_n);
    kernel->set_regs(2, static_cast<uint16_t>(1));  // index tuple width
    kernel->set_regs(3, params.row_size);
    kernel->set_regs(4, kFp16AddValuMode);
    set_reg_u32(kernel, 6, spm_local_offset(data_spm_addr, "data"));
    set_reg_u32(kernel, 8, spm_local_offset(indices_spm_addr, "indices"));
    set_reg_u32(kernel, 10, spm_local_offset(updates_spm_addr, "updates"));
    set_reg_u32(kernel, 12, params.indices_stride);
    set_reg_u32(kernel, 14, params.updates_stride);
    set_reg_u32(kernel, 16, 0);
    set_reg_u32(kernel, 18, 0);
    set_reg_u32(kernel, 20, 0);
    set_reg_u32(kernel, 24, 0);
    set_reg_u32(kernel, 26, 0);
    set_reg_u32(kernel, 28, params.row_bytes);
    set_reg_u32(kernel, 30, params.num_indices);
    kernel->set_regs(64, params.grid_x);
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(1));

    auto* queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    cores.reserve(num_cores);
    for (int core = 0; core < num_cores; ++core) {
        cores.push_back(static_cast<uint8_t>(core));
    }
    queue->enqueu_kernel(*kernel, {params.grid_x, 1, 1}, cores);
}

// Conformance-only CPU staging for the Qwen [T,H] route-reduction projection.
at::Tensor rpu_scatternd_add_u16_fp16_test(
    const at::Tensor& data,
    const at::Tensor& indices,
    const at::Tensor& updates,
    int64_t num_cores) {
    RECORD_FUNCTION("rpu::scatternd_add_u16_fp16_test", {});
    TORCH_CHECK(data.device().is_cpu() && indices.device().is_cpu() &&
                    updates.device().is_cpu(),
                "scatternd_add_u16_fp16_test: inputs must be CPU staging tensors");
    TORCH_CHECK(data.scalar_type() == at::kHalf &&
                    updates.scalar_type() == at::kHalf,
                "scatternd_add_u16_fp16_test: data and updates must be FP16");
    TORCH_CHECK(indices.scalar_type() == at::kShort,
                "scatternd_add_u16_fp16_test: indices must be raw int16/U16");
    TORCH_CHECK(data.is_contiguous() && indices.is_contiguous() &&
                    updates.is_contiguous() && data.dim() == 2 &&
                    indices.dim() == 1 && updates.dim() == 2,
                "scatternd_add_u16_fp16_test: require contiguous data [T,H], "
                "indices [N], and updates [N,H]");
    TORCH_CHECK(indices.numel() == updates.size(0) &&
                    data.size(1) == updates.size(1),
                "scatternd_add_u16_fp16_test: N/H dimensions must match");

    const int64_t data_rows = data.size(0);
    const int64_t num_indices = indices.numel();
    const int64_t row_size = data.size(1);
    make_scatter_add_params(data_rows, num_indices, row_size, num_cores);
    const uint16_t* index_data =
        reinterpret_cast<const uint16_t*>(indices.data_ptr<int16_t>());
    for (int64_t i = 0; i < num_indices; ++i) {
        TORCH_CHECK(index_data[i] < data_rows,
                    "scatternd_add_u16_fp16_test: index out of range at ", i,
                    ": ", index_data[i], " for T=", data_rows);
    }

    const size_t data_bytes =
        static_cast<size_t>(data_rows * row_size) * sizeof(uint16_t);
    const size_t indices_bytes =
        static_cast<size_t>(num_indices) * sizeof(uint16_t);
    const size_t updates_bytes =
        static_cast<size_t>(num_indices * row_size) * sizeof(uint16_t);
    const size_t data_transport_bytes = align16(data_bytes);
    const size_t indices_transport_bytes = align16(indices_bytes);
    const size_t updates_transport_bytes = align16(updates_bytes);
    std::vector<uint8_t> data_transport(data_transport_bytes, 0);
    std::vector<uint8_t> indices_transport(indices_transport_bytes, 0);
    std::vector<uint8_t> updates_transport(updates_transport_bytes, 0);
    std::memcpy(data_transport.data(), data.data_ptr(), data_bytes);
    std::memcpy(indices_transport.data(), indices.data_ptr(), indices_bytes);
    std::memcpy(updates_transport.data(), updates.data_ptr(), updates_bytes);

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    SPM_ALLOC.reset_temporary();
    const uint32_t data_off = alloc_spm_bytes(data_bytes);
    const uint32_t indices_off = alloc_spm_bytes(indices_bytes);
    const uint32_t updates_off = alloc_spm_bytes(updates_bytes);

    try {
        for (int core = 0; core < num_cores; ++core) {
            std::memcpy(SPM_ALLOC.cpu_ptr(core, data_off),
                        data_transport.data(), data_transport_bytes);
            std::memcpy(SPM_ALLOC.cpu_ptr(core, indices_off),
                        indices_transport.data(), indices_transport_bytes);
            std::memcpy(SPM_ALLOC.cpu_ptr(core, updates_off),
                        updates_transport.data(), updates_transport_bytes);
        }

        rpu_launch_scatternd_add_u16_fp16_spm_kernel(
            SPM_ALLOC.addr(0, data_off), SPM_ALLOC.addr(0, indices_off),
            SPM_ALLOC.addr(0, updates_off), data_rows, num_indices, row_size,
            static_cast<int>(num_cores));

        at::Tensor output = at::empty(
            {num_cores, data_rows, row_size},
            at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
        std::vector<uint8_t> output_transport(data_transport_bytes);
        for (int64_t core = 0; core < num_cores; ++core) {
            std::memcpy(output_transport.data(),
                        SPM_ALLOC.cpu_ptr(static_cast<int>(core), data_off),
                        data_transport_bytes);
            std::memcpy(static_cast<char*>(output.data_ptr()) +
                            core * data_bytes,
                        output_transport.data(), data_bytes);
        }
        SPM_ALLOC.reset_temporary();
        return output;
    } catch (...) {
        SPM_ALLOC.reset_temporary();
        throw;
    }
}
