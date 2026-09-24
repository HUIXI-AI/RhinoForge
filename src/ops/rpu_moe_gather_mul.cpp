// Qwen3.5 MoE routed-score multiplication for pre-gathered token rows.
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
                "moe_gather_mul_v2: SPM allocator must be initialized");
    const uint32_t base = SPM_ALLOC.addr(0, 0);
    TORCH_CHECK(unified_address >= base &&
                    static_cast<uint64_t>(unified_address - base) <
                        SpmAllocator::SPM_TOTAL,
                "moe_gather_mul_v2: ", name,
                " must be a core-0 unified SPM address, got ", unified_address,
                " base=", base);
    return unified_address - base;
}

struct GatherMulParams {
    uint16_t num_tokens;
    uint16_t num_experts;
    uint32_t num_indices;
    uint16_t hidden_dims;
    uint32_t token_t_byte_step;
    uint32_t scale_t_byte_step;
    uint32_t output_t_byte_step;
    uint16_t grid_x;
    uint16_t grid_y;
};

GatherMulParams make_gather_mul_params(int64_t num_tokens,
                                       int64_t num_experts,
                                       int64_t num_indices,
                                       int64_t hidden_dims,
                                       int64_t num_cores) {
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "moe_gather_mul_v2: num_cores must be in [1,8], got ",
                num_cores);
    TORCH_CHECK(num_tokens > 0 &&
                    num_tokens <= std::numeric_limits<uint16_t>::max(),
                "moe_gather_mul_v2: T must fit u16, got ", num_tokens);
    TORCH_CHECK(num_experts > 0 &&
                    num_experts <= std::numeric_limits<uint16_t>::max(),
                "moe_gather_mul_v2: E must fit u16, got ", num_experts);
    TORCH_CHECK(num_indices > 0 &&
                    num_indices <= std::numeric_limits<uint32_t>::max(),
                "moe_gather_mul_v2: N must fit u32, got ", num_indices);
    // The current llm_gather_mul_v2 oplib is numerically certified only for a
    // single hidden-dimension warp. Upstream gen_attr and Qwen use D <= 4096;
    // D=4097 produces sparse large errors even with a non-aliased output.
    TORCH_CHECK(hidden_dims > 0 && hidden_dims <= 4096,
                "moe_gather_mul_v2: require D in [1,4096] for the current "
                "oplib, got ", hidden_dims);

    const uint64_t grid_x =
        (static_cast<uint64_t>(hidden_dims) + 4095) / 4096;
    const uint64_t grid_y =
        (static_cast<uint64_t>(num_indices) + 63) / 64;
    TORCH_CHECK(grid_x <= std::numeric_limits<uint16_t>::max() &&
                    grid_y <= std::numeric_limits<uint16_t>::max(),
                "moe_gather_mul_v2: grid dimensions must fit u16, got ",
                grid_x, "x", grid_y);

    return {
        static_cast<uint16_t>(num_tokens),
        static_cast<uint16_t>(num_experts),
        static_cast<uint32_t>(num_indices),
        static_cast<uint16_t>(hidden_dims),
        static_cast<uint32_t>(hidden_dims * sizeof(uint16_t)),
        static_cast<uint32_t>(num_experts * sizeof(uint16_t)),
        static_cast<uint32_t>(hidden_dims * sizeof(uint16_t)),
        static_cast<uint16_t>(grid_x),
        static_cast<uint16_t>(grid_y),
    };
}

}  // namespace

void rpu_launch_moe_gather_mul_v2_fp16_spm_kernel(
    uint32_t token_spm_addr,
    uint32_t scale_spm_addr,
    uint32_t token_indices_spm_addr,
    uint32_t expert_indices_spm_addr,
    uint32_t output_spm_addr,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t num_indices,
    int64_t hidden_dims,
    int num_cores) {
    const GatherMulParams params = make_gather_mul_params(
        num_tokens, num_experts, num_indices, hidden_dims, num_cores);
    Kernel_t* kernel = GET_KERNEL(KernelId::MOE_GATHER_MUL_V2_FP16);
    TORCH_CHECK(kernel != nullptr,
                "moe_gather_mul_v2: llm_gather_mul_v2 is absent from the "
                "active oplib");
    kernel->reset_regs();
    set_reg_u32(kernel, 0, spm_local_offset(token_spm_addr, "token"));
    set_reg_u32(kernel, 2, spm_local_offset(scale_spm_addr, "scale"));
    set_reg_u32(
        kernel, 4, spm_local_offset(token_indices_spm_addr, "token_indices"));
    set_reg_u32(
        kernel, 6, spm_local_offset(expert_indices_spm_addr, "expert_indices"));
    set_reg_u32(kernel, 8, spm_local_offset(output_spm_addr, "output"));
    kernel->set_regs(10, params.num_tokens);
    kernel->set_regs(11, params.num_experts);
    set_reg_u32(kernel, 12, params.num_indices);
    kernel->set_regs(14, params.hidden_dims);
    set_reg_u32(kernel, 16, params.token_t_byte_step);
    set_reg_u32(kernel, 18, params.scale_t_byte_step);
    set_reg_u32(kernel, 20, params.output_t_byte_step);
    kernel->set_regs(64, params.grid_x);
    kernel->set_regs(65, params.grid_y);
    kernel->set_regs(66, static_cast<uint16_t>(1));

    auto* queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    cores.reserve(num_cores);
    for (int core = 0; core < num_cores; ++core) {
        cores.push_back(static_cast<uint8_t>(core));
    }
    queue->enqueu_kernel(
        *kernel, {params.grid_x, params.grid_y, 1}, cores);
}

// Conformance-only CPU staging. Raw int16 tensors carry unsigned U16 indices.
at::Tensor rpu_moe_gather_mul_v2_test(
    const at::Tensor& token,
    const at::Tensor& scale,
    const at::Tensor& token_indices,
    const at::Tensor& expert_indices,
    bool in_place,
    int64_t num_cores) {
    RECORD_FUNCTION("rpu::moe_gather_mul_v2_test", {});
    TORCH_CHECK(token.device().is_cpu() && scale.device().is_cpu() &&
                    token_indices.device().is_cpu() &&
                    expert_indices.device().is_cpu(),
                "moe_gather_mul_v2_test: inputs must be CPU staging tensors");
    TORCH_CHECK(token.scalar_type() == at::kHalf &&
                    scale.scalar_type() == at::kHalf,
                "moe_gather_mul_v2_test: token and scale must be FP16");
    TORCH_CHECK(token_indices.scalar_type() == at::kShort &&
                    expert_indices.scalar_type() == at::kShort,
                "moe_gather_mul_v2_test: indices must be raw int16/U16");
    TORCH_CHECK(token.is_contiguous() && scale.is_contiguous() &&
                    token_indices.is_contiguous() &&
                    expert_indices.is_contiguous() && token.dim() == 2 &&
                    scale.dim() == 2 && token_indices.dim() == 1 &&
                    expert_indices.dim() == 1,
                "moe_gather_mul_v2_test: require contiguous token [N,D], "
                "scale [T,E], and indices [N]");
    TORCH_CHECK(token.size(0) == token_indices.numel() &&
                    token_indices.numel() == expert_indices.numel(),
                "moe_gather_mul_v2_test: token and index lengths must match");

    const int64_t num_tokens = scale.size(0);
    const int64_t num_experts = scale.size(1);
    const int64_t num_indices = token.size(0);
    const int64_t hidden_dims = token.size(1);
    make_gather_mul_params(
        num_tokens, num_experts, num_indices, hidden_dims, num_cores);
    TORCH_CHECK(num_indices % num_tokens == 0,
                "moe_gather_mul_v2_test: N must be an integer multiple of T");

    const uint16_t* token_index_data =
        reinterpret_cast<const uint16_t*>(token_indices.data_ptr<int16_t>());
    const uint16_t* expert_index_data =
        reinterpret_cast<const uint16_t*>(expert_indices.data_ptr<int16_t>());
    for (int64_t i = 0; i < num_indices; ++i) {
        TORCH_CHECK(token_index_data[i] < num_tokens,
                    "moe_gather_mul_v2_test: token index out of range at ", i,
                    ": ", token_index_data[i]);
        TORCH_CHECK(expert_index_data[i] < num_experts,
                    "moe_gather_mul_v2_test: expert index out of range at ", i,
                    ": ", expert_index_data[i]);
    }

    const size_t token_bytes =
        static_cast<size_t>(num_indices * hidden_dims) * sizeof(uint16_t);
    const size_t scale_bytes =
        static_cast<size_t>(num_tokens * num_experts) * sizeof(uint16_t);
    const size_t indices_bytes =
        static_cast<size_t>(num_indices) * sizeof(uint16_t);
    const size_t output_bytes = token_bytes;
    const size_t token_transport_bytes = align16(token_bytes);
    const size_t scale_transport_bytes = align16(scale_bytes);
    const size_t indices_transport_bytes = align16(indices_bytes);
    const size_t output_transport_bytes = align16(output_bytes);
    std::vector<uint8_t> token_transport(token_transport_bytes, 0);
    std::vector<uint8_t> scale_transport(scale_transport_bytes, 0);
    std::vector<uint8_t> token_indices_transport(indices_transport_bytes, 0);
    std::vector<uint8_t> expert_indices_transport(indices_transport_bytes, 0);
    std::memcpy(token_transport.data(), token.data_ptr(), token_bytes);
    std::memcpy(scale_transport.data(), scale.data_ptr(), scale_bytes);
    std::memcpy(
        token_indices_transport.data(), token_indices.data_ptr(), indices_bytes);
    std::memcpy(expert_indices_transport.data(), expert_indices.data_ptr(),
                indices_bytes);

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    SPM_ALLOC.reset_temporary();
    const uint32_t token_off = alloc_spm_bytes(token_bytes);
    const uint32_t scale_off = alloc_spm_bytes(scale_bytes);
    const uint32_t token_indices_off = alloc_spm_bytes(indices_bytes);
    const uint32_t expert_indices_off = alloc_spm_bytes(indices_bytes);
    const uint32_t output_off = in_place
        ? token_off
        : alloc_spm_bytes(output_bytes);

    try {
        for (int core = 0; core < num_cores; ++core) {
            std::memcpy(SPM_ALLOC.cpu_ptr(core, token_off),
                        token_transport.data(), token_transport_bytes);
            std::memcpy(SPM_ALLOC.cpu_ptr(core, scale_off),
                        scale_transport.data(), scale_transport_bytes);
            std::memcpy(SPM_ALLOC.cpu_ptr(core, token_indices_off),
                        token_indices_transport.data(), indices_transport_bytes);
            std::memcpy(SPM_ALLOC.cpu_ptr(core, expert_indices_off),
                        expert_indices_transport.data(), indices_transport_bytes);
        }

        rpu_launch_moe_gather_mul_v2_fp16_spm_kernel(
            SPM_ALLOC.addr(0, token_off), SPM_ALLOC.addr(0, scale_off),
            SPM_ALLOC.addr(0, token_indices_off),
            SPM_ALLOC.addr(0, expert_indices_off),
            SPM_ALLOC.addr(0, output_off), num_tokens, num_experts, num_indices,
            hidden_dims, static_cast<int>(num_cores));

        at::Tensor output = at::empty(
            {num_cores, num_indices, hidden_dims},
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
