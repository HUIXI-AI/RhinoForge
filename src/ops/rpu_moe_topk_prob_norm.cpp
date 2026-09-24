// Qwen3.5 MoE routing-score normalization.
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

size_t align256(size_t value) {
    return (value + 255) & ~size_t(255);
}

size_t align16(size_t value) {
    return (value + 15) & ~size_t(15);
}

uint32_t alloc_spm_bytes(size_t bytes) {
    return SPM_ALLOC.alloc_temporary(align256(bytes));
}

void validate_shape(int64_t num_tokens,
                    int64_t num_experts,
                    int64_t k_threshold,
                    int64_t num_cores) {
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "moe_topk_prob_norm: num_cores must be in [1,8], got ",
                num_cores);
    TORCH_CHECK(
        num_tokens > 0 &&
            num_tokens <=
                static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        "moe_topk_prob_norm: T must fit u32, got ", num_tokens);
    TORCH_CHECK(num_experts > 0 &&
                    num_experts <= std::numeric_limits<uint16_t>::max(),
                "moe_topk_prob_norm: E must fit u16, got ", num_experts);
    TORCH_CHECK(k_threshold > 0 && k_threshold <= num_experts &&
                    k_threshold <= std::numeric_limits<uint16_t>::max(),
                "moe_topk_prob_norm: require 0 < K <= E and K fit u16, got K=",
                k_threshold, " E=", num_experts);
    const int64_t grid_y = (num_tokens + 255) / 256;
    TORCH_CHECK(grid_y <= std::numeric_limits<uint16_t>::max(),
                "moe_topk_prob_norm: ceil(T/256) must fit u16, got ", grid_y);
}

}  // namespace

// Graph-facing softmax_c16_gauto launcher used by MoE routing. Each logical
// [N,C] row occupies ceil(C/16)*16 FP16 elements in local SPM.
void rpu_launch_softmax_c16_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t output_spm_addr,
    int64_t n,
    int64_t c,
    int num_cores) {
    constexpr uint64_t kMaxVlmPerThread = 253;
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "softmax_c16: num_cores must be in [1,8], got ", num_cores);
    TORCH_CHECK(n > 0 && n <= std::numeric_limits<uint16_t>::max(),
                "softmax_c16: N must be in [1,65535], got ", n);
    TORCH_CHECK(c > 0 && c <= 2048,
                "softmax_c16: C must be in [1,2048], got ", c);
    TORCH_CHECK(input_spm_addr % 32 == 0 && output_spm_addr % 32 == 0,
                "softmax_c16: SPM addresses must be 32-byte aligned");
    const uint64_t c_v16 = CeilDiv(c, 16);
    const uint64_t row_bytes = c_v16 * 32;
    TORCH_CHECK(static_cast<uint64_t>(n) * row_bytes * 2 <=
                    SpmAllocator::SPM_PLANNING_BUDGET,
                "softmax_c16: padded input and output exceed the per-core "
                "SPM planning budget");

    const uint64_t c_v256 = CeilDiv(c, 256);
    const uint64_t num_vlm_per_c = c_v16 + 2;
    const uint64_t min_n_per_warp = NUM_THD_PER_WARP;
    const uint64_t min_n_per_launch = NUM_THD_PER_LAUNCH;
    const uint64_t max_n_per_thd = kMaxVlmPerThread / num_vlm_per_c;
    const uint64_t max_n_per_warp =
        max_n_per_thd * NUM_THD_PER_WARP;
    const uint64_t max_n_per_launch =
        max_n_per_warp * NUM_WARP_PER_LAUNCH;
    TORCH_CHECK(max_n_per_thd > 0,
                "softmax_c16: C exceeds the kernel VLM schedule");

    uint64_t n_per_thd = 0;
    uint64_t grid_dim_x = 1;
    if (static_cast<uint64_t>(n) <= min_n_per_launch) {
        grid_dim_x = CeilDiv(n, min_n_per_warp);
        n_per_thd = 1;
    } else if (static_cast<uint64_t>(n) <= max_n_per_launch) {
        grid_dim_x = NUM_WARP_PER_LAUNCH;
        n_per_thd = CeilDiv(n, min_n_per_launch);
    } else {
        grid_dim_x = CeilDiv(n, max_n_per_warp);
        n_per_thd = max_n_per_thd;
    }
    TORCH_CHECK(n_per_thd * num_vlm_per_c <= kMaxVlmPerThread &&
                    grid_dim_x <= std::numeric_limits<uint16_t>::max(),
                "softmax_c16: launch schedule exceeds the kernel ABI");

    Kernel_t* kernel = GET_KERNEL(KernelId::SOFTMAX_C16_GAUTO);
    TORCH_CHECK(kernel != nullptr,
                "softmax_c16: softmax_c16_gauto is absent from the active oplib");
    kernel->reset_regs();
    set_reg_u32(kernel, 0, input_spm_addr / 32);
    set_reg_u32(kernel, 2, output_spm_addr / 32);
    kernel->set_regs(4, static_cast<uint16_t>(n));
    kernel->set_regs(5, static_cast<uint16_t>(c));
    kernel->set_regs(6, static_cast<uint16_t>(c_v16));
    kernel->set_regs(7, static_cast<uint16_t>(c_v256));
    kernel->set_regs(8, static_cast<uint16_t>(c_v16));
    kernel->set_regs(9, static_cast<uint16_t>(c_v16 * 16));
    kernel->set_regs(10, static_cast<uint16_t>(n_per_thd));
    kernel->set_regs(64, static_cast<uint16_t>(grid_dim_x));
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(1));

    auto* queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    cores.reserve(num_cores);
    for (int core = 0; core < num_cores; ++core) {
        cores.push_back(static_cast<uint8_t>(core));
    }
    queue->enqueu_kernel(
        *kernel, {static_cast<uint16_t>(grid_dim_x), 1, 1}, cores);
}

void rpu_launch_moe_topk_prob_norm_fp16_spm_kernel(
    uint32_t token_prob_spm_addr,
    uint32_t topk_prob_spm_addr,
    uint32_t output_spm_addr,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t k_threshold,
    int num_cores) {
    validate_shape(num_tokens, num_experts, k_threshold, num_cores);

    const uint32_t token_prob_t_byte_step =
        static_cast<uint32_t>(num_experts * sizeof(uint16_t));
    const uint32_t topk_prob_t_byte_step =
        static_cast<uint32_t>(k_threshold * sizeof(uint16_t));
    const uint32_t output_t_byte_step = token_prob_t_byte_step;
    const int64_t grid_x = (num_experts + 255) / 256;
    const int64_t grid_y = (num_tokens + 255) / 256;

    Kernel_t* kernel = GET_KERNEL(KernelId::MOE_TOPK_PROB_NORM_FP16);
    TORCH_CHECK(kernel != nullptr,
                "moe_topk_prob_norm: llm_topk_prob_norm is absent from the "
                "active oplib");
    kernel->reset_regs();
    set_reg_u32(kernel, 0, token_prob_spm_addr);
    set_reg_u32(kernel, 2, topk_prob_spm_addr);
    set_reg_u32(kernel, 4, output_spm_addr);
    set_reg_u32(kernel, 6, static_cast<uint32_t>(num_tokens));
    kernel->set_regs(8, static_cast<uint16_t>(num_experts));
    kernel->set_regs(9, static_cast<uint16_t>(k_threshold));
    set_reg_u32(kernel, 10, token_prob_t_byte_step);
    set_reg_u32(kernel, 12, topk_prob_t_byte_step);
    set_reg_u32(kernel, 14, output_t_byte_step);
    kernel->set_regs(64, static_cast<uint16_t>(grid_x));
    kernel->set_regs(65, static_cast<uint16_t>(grid_y));
    kernel->set_regs(66, static_cast<uint16_t>(1));

    auto* queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    cores.reserve(num_cores);
    for (int core = 0; core < num_cores; ++core) {
        cores.push_back(static_cast<uint8_t>(core));
    }
    queue->enqueu_kernel(
        *kernel,
        {static_cast<uint16_t>(grid_x), static_cast<uint16_t>(grid_y), 1},
        cores);
}

// Conformance-only CPU staging. Runtime semantics normalize every [T,E] entry
// by the sum of that row's selected [T,K] probabilities.
at::Tensor rpu_moe_topk_prob_norm_test(
    const at::Tensor& token_prob,
    const at::Tensor& topk_prob,
    int64_t num_cores) {
    RECORD_FUNCTION("rpu::moe_topk_prob_norm_test", {});
    TORCH_CHECK(token_prob.device().is_cpu() && topk_prob.device().is_cpu(),
                "moe_topk_prob_norm_test: inputs must be CPU staging tensors");
    TORCH_CHECK(token_prob.scalar_type() == at::kHalf &&
                    topk_prob.scalar_type() == at::kHalf,
                "moe_topk_prob_norm_test: inputs must be fp16");
    TORCH_CHECK(token_prob.is_contiguous() && topk_prob.is_contiguous() &&
                    token_prob.dim() == 2 && topk_prob.dim() == 2 &&
                    token_prob.size(0) == topk_prob.size(0),
                "moe_topk_prob_norm_test: inputs must be contiguous [T,E] "
                "and [T,K]");

    const int64_t num_tokens = token_prob.size(0);
    const int64_t num_experts = token_prob.size(1);
    const int64_t k_threshold = topk_prob.size(1);
    validate_shape(num_tokens, num_experts, k_threshold, num_cores);

    const size_t token_prob_bytes =
        static_cast<size_t>(num_tokens * num_experts) * sizeof(uint16_t);
    const size_t topk_prob_bytes =
        static_cast<size_t>(num_tokens * k_threshold) * sizeof(uint16_t);
    const size_t output_bytes = token_prob_bytes;
    const size_t token_transport_bytes = align16(token_prob_bytes);
    const size_t topk_transport_bytes = align16(topk_prob_bytes);
    const size_t output_transport_bytes = align16(output_bytes);
    std::vector<uint8_t> token_transport(token_transport_bytes, 0);
    std::vector<uint8_t> topk_transport(topk_transport_bytes, 0);
    std::memcpy(token_transport.data(), token_prob.data_ptr(), token_prob_bytes);
    std::memcpy(topk_transport.data(), topk_prob.data_ptr(), topk_prob_bytes);

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    SPM_ALLOC.reset_temporary();
    const uint32_t token_prob_off = alloc_spm_bytes(token_prob_bytes);
    const uint32_t topk_prob_off = alloc_spm_bytes(topk_prob_bytes);
    const uint32_t output_off = alloc_spm_bytes(output_bytes);

    try {
        for (int core = 0; core < num_cores; ++core) {
            std::memcpy(SPM_ALLOC.cpu_ptr(core, token_prob_off),
                        token_transport.data(), token_transport_bytes);
            std::memcpy(SPM_ALLOC.cpu_ptr(core, topk_prob_off),
                        topk_transport.data(), topk_transport_bytes);
        }

        rpu_launch_moe_topk_prob_norm_fp16_spm_kernel(
            SPM_ALLOC.addr(0, token_prob_off),
            SPM_ALLOC.addr(0, topk_prob_off), SPM_ALLOC.addr(0, output_off),
            num_tokens, num_experts, k_threshold,
            static_cast<int>(num_cores));

        at::Tensor output = at::empty(
            {num_cores, num_tokens, num_experts},
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
