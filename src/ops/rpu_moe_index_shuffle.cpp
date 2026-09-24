// Qwen3.5 MoE sparse-routing primitives.
//

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/Half.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
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

bool is_raw16_dtype(at::ScalarType dtype) {
    return dtype == at::kHalf || dtype == at::kShort;
}

constexpr int64_t KVSORT_SLICE_SIZE = 4096;

uint32_t kvsort_addr_v16(uint32_t address, const char* name) {
    TORCH_CHECK(address % 32 == 0,
                "moe_topk_by_kvsort: ", name,
                " address must be 32-byte aligned, got ", address);
    return address / 32;
}

void launch_moe_topk_by_kvsort_fp16_pass(
    uint32_t input_key_spm_addr,
    uint32_t input_value_spm_addr,
    uint32_t output_key_spm_addr,
    uint32_t output_value_spm_addr,
    int64_t n,
    int64_t c,
    int64_t k,
    bool descending,
    int num_cores) {
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "moe_topk_by_kvsort: num_cores must be in [1,8], got ",
                num_cores);
    TORCH_CHECK(n > 0 && n <= std::numeric_limits<uint16_t>::max(),
                "moe_topk_by_kvsort: n must fit u16, got ", n);
    TORCH_CHECK(c > 0 &&
                    c <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                "moe_topk_by_kvsort: c must fit u32, got ", c);
    TORCH_CHECK(k > 10 && k <= c && k <= KVSORT_SLICE_SIZE,
                "moe_topk_by_kvsort: require 10 < k <= min(c,4096), got k=",
                k, " c=", c);

    const int64_t c_pad = static_cast<int64_t>(align16(c));
    const int64_t c_v16 = (c + 15) / 16;
    const int64_t c_pad_v16 = c_pad / 16;
    const int64_t c_slice_v16 = KVSORT_SLICE_SIZE / 16;
    const int64_t c_slice_v16_per_thread = c_slice_v16 / 16;
    const int64_t c_slice_count =
        (c_pad + KVSORT_SLICE_SIZE - 1) / KVSORT_SLICE_SIZE;
    const int64_t k_v16 = (k + 15) / 16;
    TORCH_CHECK(c_pad_v16 <= std::numeric_limits<uint16_t>::max(),
                "moe_topk_by_kvsort: c_pad_v16 must fit u16, got ",
                c_pad_v16);
    TORCH_CHECK(c_slice_count <= std::numeric_limits<uint16_t>::max(),
                "moe_topk_by_kvsort: slice count must fit u16, got ",
                c_slice_count);

    Kernel_t* kernel = GET_KERNEL(KernelId::MOE_TOPK_BY_KVSORT_FP16);
    TORCH_CHECK(
        kernel != nullptr,
        "moe_topk_by_kvsort: topk_by_kvsort_fp16 is absent from the active oplib");
    kernel->reset_regs();
    set_reg_u32(kernel, 0, kvsort_addr_v16(input_key_spm_addr, "input key"));
    set_reg_u32(kernel, 2,
                kvsort_addr_v16(input_value_spm_addr, "input value"));
    set_reg_u32(kernel, 4,
                kvsort_addr_v16(output_key_spm_addr, "output key"));
    set_reg_u32(kernel, 6,
                kvsort_addr_v16(output_value_spm_addr, "output value"));
    set_reg_u32(kernel, 8, static_cast<uint32_t>(c));
    set_reg_u32(kernel, 10, static_cast<uint32_t>(c_pad));
    kernel->set_regs(12, static_cast<uint16_t>(c_v16));
    kernel->set_regs(13, static_cast<uint16_t>(c_pad_v16));
    kernel->set_regs(14, static_cast<uint16_t>(c_slice_v16));
    kernel->set_regs(15, static_cast<uint16_t>(c_slice_v16_per_thread));
    kernel->set_regs(16, static_cast<uint16_t>(4));
    kernel->set_regs(17, static_cast<uint16_t>(c_slice_v16_per_thread));
    kernel->set_regs(18, static_cast<uint16_t>(4));
    kernel->set_regs(19, static_cast<uint16_t>(8));
    kernel->set_regs(20, static_cast<uint16_t>(1));
    kernel->set_regs(21, static_cast<uint16_t>(32));
    kernel->set_regs(
        25, static_cast<uint16_t>(descending ? ALU_MIN_FP16_WMODE
                                             : ALU_MAX_FP16_WMODE));
    kernel->set_regs(
        26, static_cast<uint16_t>(descending ? ALU_MAX_FP16_WMODE
                                             : ALU_MIN_FP16_WMODE));
    kernel->set_regs(
        27, static_cast<uint16_t>(descending ? ALU_LT_IF16_OU8_WMODE
                                             : ALU_GT_IF16_OU8_WMODE));
    kernel->set_regs(
        28, static_cast<uint16_t>(descending ? ALU_GT_IF16_OU8_WMODE
                                             : ALU_LT_IF16_OU8_WMODE));
    kernel->set_regs(
        29, static_cast<uint16_t>(descending ? NEG_INF_FP16 : POS_INF_FP16));
    kernel->set_regs(30, static_cast<uint16_t>(k));
    kernel->set_regs(31, static_cast<uint16_t>(k_v16));
    kernel->set_regs(64, static_cast<uint16_t>(c_slice_count));
    kernel->set_regs(65, static_cast<uint16_t>(n));
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
        {static_cast<uint16_t>(c_slice_count), static_cast<uint16_t>(n), 1},
        cores);
}

}  // namespace

void rpu_launch_moe_topk_scalar_fp16_spm_kernel(
    uint32_t input_key_spm_addr,
    uint32_t input_value_spm_addr,
    uint32_t output_key_spm_addr,
    uint32_t output_value_spm_addr,
    int64_t n,
    int64_t c,
    int64_t k,
    bool descending,
    bool cvt_key,
    int num_cores) {
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "moe_topk_scalar: num_cores must be in [1,8], got ",
                num_cores);
    TORCH_CHECK(n > 0 && n <= std::numeric_limits<uint16_t>::max(),
                "moe_topk_scalar: n must fit u16, got ", n);
    TORCH_CHECK(c > 0 && c < std::numeric_limits<uint16_t>::max(),
                "moe_topk_scalar: c must be in [1,65534], got ", c);
    TORCH_CHECK(k > 0 && k <= c && k <= 10,
                "moe_topk_scalar: require 0 < k <= min(c,10), got k=", k,
                " c=", c);

    const int64_t c_v16 = (c + 15) / 16;
    const int64_t c_v16_per_thread = (c_v16 + 15) / 16;

    Kernel_t* kernel = GET_KERNEL(KernelId::MOE_TOPK_SCALAR_FP16);
    TORCH_CHECK(kernel != nullptr,
                "moe_topk_scalar: topk_scalar_fp16 is absent from the active oplib");
    kernel->reset_regs();
    set_reg_u32(kernel, 0, input_key_spm_addr);
    set_reg_u32(kernel, 2, input_value_spm_addr);
    set_reg_u32(kernel, 4, output_key_spm_addr);
    set_reg_u32(kernel, 6, output_value_spm_addr);
    kernel->set_regs(8, static_cast<uint16_t>(c));
    kernel->set_regs(10, static_cast<uint16_t>(c_v16));
    kernel->set_regs(11, static_cast<uint16_t>(c_v16_per_thread));
    kernel->set_regs(12, static_cast<uint16_t>(c_v16_per_thread));
    kernel->set_regs(13, static_cast<uint16_t>(k));
    kernel->set_regs(14, static_cast<uint16_t>(descending));
    kernel->set_regs(15, static_cast<uint16_t>(
        descending ? NEG_INF_FP16 : POS_INF_FP16));
    kernel->set_regs(16, static_cast<uint16_t>(sizeof(c10::Half)));
    kernel->set_regs(17, static_cast<uint16_t>(cvt_key));
    kernel->set_regs(64, static_cast<uint16_t>(n));
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(1));

    auto* queue = GET_QUEUE(num_cores);
    // Queue broadcast mode also supports graph recording.
    queue->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    cores.reserve(num_cores);
    for (int core = 0; core < num_cores; ++core) {
        cores.push_back(static_cast<uint8_t>(core));
    }
    queue->enqueu_kernel(
        *kernel, {static_cast<uint16_t>(n), 1, 1}, cores);
}

void rpu_launch_moe_topk_by_kvsort_fp16_spm_kernel(
    uint32_t input_key_spm_addr,
    uint32_t input_value_spm_addr,
    uint32_t output_key_spm_addr,
    uint32_t output_value_spm_addr,
    uint32_t workspace_key_spm_addr,
    uint32_t workspace_value_spm_addr,
    int64_t n,
    int64_t c,
    int64_t k,
    bool descending,
    int num_cores) {
    TORCH_CHECK(n > 0 && n <= std::numeric_limits<uint16_t>::max(),
                "moe_topk_by_kvsort: n must fit u16, got ", n);
    TORCH_CHECK(c > 0 &&
                    c <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                "moe_topk_by_kvsort: c must fit u32, got ", c);
    TORCH_CHECK(k > 10 && k <= c && k <= KVSORT_SLICE_SIZE,
                "moe_topk_by_kvsort: require 10 < k <= min(c,4096), got k=",
                k, " c=", c);
    const int64_t k_pad = static_cast<int64_t>(align16(k));
    int64_t cur_c = c;
    int64_t next_c =
        k_pad * ((cur_c + KVSORT_SLICE_SIZE - 1) / KVSORT_SLICE_SIZE);
    bool first_pass = true;

    if (cur_c > KVSORT_SLICE_SIZE) {
        TORCH_CHECK(workspace_key_spm_addr != 0 &&
                        workspace_value_spm_addr != 0,
                    "moe_topk_by_kvsort: C>4096 requires key/value workspace");
    }
    while (cur_c > KVSORT_SLICE_SIZE) {
        TORCH_CHECK(next_c < cur_c,
                    "moe_topk_by_kvsort: merge must shrink C, got current C=",
                    cur_c, " next C=", next_c, " K=", k);
        launch_moe_topk_by_kvsort_fp16_pass(
            first_pass ? input_key_spm_addr : workspace_key_spm_addr,
            first_pass ? input_value_spm_addr : workspace_value_spm_addr,
            workspace_key_spm_addr, workspace_value_spm_addr,
            n, cur_c, k, descending, num_cores);
        first_pass = false;
        cur_c = next_c;
        next_c =
            k_pad * ((cur_c + KVSORT_SLICE_SIZE - 1) / KVSORT_SLICE_SIZE);
    }
    launch_moe_topk_by_kvsort_fp16_pass(
        first_pass ? input_key_spm_addr : workspace_key_spm_addr,
        first_pass ? input_value_spm_addr : workspace_value_spm_addr,
        output_key_spm_addr, output_value_spm_addr,
        n, cur_c, k, descending, num_cores);
}

void rpu_launch_moe_bincount_scalar_spm_kernel(
    uint32_t input_key_spm_addr,
    uint32_t output_key_spm_addr,
    uint32_t output_bin_spm_addr,
    int64_t c,
    int64_t maxk,
    bool cvt_key,
    bool output_cvt_key,
    int num_cores) {
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "moe_bincount_scalar: num_cores must be in [1,8], got ",
                num_cores);
    TORCH_CHECK(c > 0 && c <= std::numeric_limits<uint16_t>::max(),
                "moe_bincount_scalar: c must be in [1,65535], got ", c);
    TORCH_CHECK(maxk > 0 && maxk < 1024,
                "moe_bincount_scalar: maxk must be in [1,1023], got ",
                maxk);

    const int64_t c_v16 = (c + 15) / 16;
    const int64_t c_v16_per_thread = (c_v16 + 15) / 16;
    const int64_t maxk_v16 = (maxk + 15) / 16;
    const int64_t grid_x = (maxk + 255) / 256;

    Kernel_t* kernel = GET_KERNEL(KernelId::MOE_BINCOUNT_SCALAR);
    TORCH_CHECK(kernel != nullptr,
                "moe_bincount_scalar: bincount_scalar is absent from the "
                "active oplib");
    kernel->reset_regs();
    set_reg_u32(kernel, 0, input_key_spm_addr);
    set_reg_u32(kernel, 2, output_key_spm_addr);
    set_reg_u32(kernel, 4, output_bin_spm_addr);
    kernel->set_regs(6, static_cast<uint16_t>(c));
    kernel->set_regs(7, static_cast<uint16_t>(c_v16));
    kernel->set_regs(8, static_cast<uint16_t>(c_v16_per_thread));
    kernel->set_regs(9, static_cast<uint16_t>(c_v16_per_thread));
    kernel->set_regs(10, static_cast<uint16_t>(sizeof(uint16_t)));
    kernel->set_regs(11, static_cast<uint16_t>(maxk));
    kernel->set_regs(12, static_cast<uint16_t>(maxk_v16));
    kernel->set_regs(13, static_cast<uint16_t>(cvt_key));
    kernel->set_regs(14, static_cast<uint16_t>(output_cvt_key));
    kernel->set_regs(64, static_cast<uint16_t>(grid_x));
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
        *kernel, {static_cast<uint16_t>(grid_x), 1, 1}, cores);
}

// Conformance-only wrapper. Input is broadcast to each selected core and each
// core's local result is returned. Values intentionally accept FP16 or int16:
// the kernel never interprets those carried two-byte payload bits.
std::tuple<at::Tensor, at::Tensor> rpu_moe_topk_scalar_test(
    const at::Tensor& input_keys,
    const at::Tensor& input_values,
    int64_t k,
    bool descending,
    bool cvt_key,
    int64_t num_cores) {
    RECORD_FUNCTION("rpu::moe_topk_scalar_test", {});
    TORCH_CHECK(input_keys.device().is_cpu() && input_values.device().is_cpu(),
                "moe_topk_scalar_test: inputs must be CPU staging tensors");
    TORCH_CHECK(input_keys.scalar_type() == at::kHalf &&
                    is_raw16_dtype(input_values.scalar_type()),
                "moe_topk_scalar_test: keys must be fp16 and values fp16/int16");
    TORCH_CHECK(input_keys.is_contiguous() && input_values.is_contiguous() &&
                    input_keys.dim() == 2 &&
                    input_values.sizes() == input_keys.sizes(),
                "moe_topk_scalar_test: inputs must be contiguous, equal [N,C]");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "moe_topk_scalar_test: num_cores must be in [1,8]");

    const int64_t n = input_keys.size(0);
    const int64_t c = input_keys.size(1);
    const size_t input_bytes = static_cast<size_t>(n * c) * sizeof(uint16_t);
    const size_t output_bytes = static_cast<size_t>(n * k) * sizeof(uint16_t);
    const size_t input_transport_bytes = align16(input_bytes);
    const size_t output_transport_bytes = align16(output_bytes);
    std::vector<uint8_t> key_input_transport(input_transport_bytes, 0);
    std::vector<uint8_t> value_input_transport(input_transport_bytes, 0);
    std::memcpy(key_input_transport.data(), input_keys.data_ptr(), input_bytes);
    std::memcpy(value_input_transport.data(), input_values.data_ptr(), input_bytes);

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    SPM_ALLOC.reset_temporary();
    const uint32_t key_in_off = alloc_spm_bytes(input_bytes);
    const uint32_t value_in_off = alloc_spm_bytes(input_bytes);
    const uint32_t key_out_off = alloc_spm_bytes(output_bytes);
    const uint32_t value_out_off = alloc_spm_bytes(output_bytes);

    try {
        for (int core = 0; core < num_cores; ++core) {
            std::memcpy(SPM_ALLOC.cpu_ptr(core, key_in_off),
                        key_input_transport.data(), input_transport_bytes);
            std::memcpy(SPM_ALLOC.cpu_ptr(core, value_in_off),
                        value_input_transport.data(), input_transport_bytes);
        }

        rpu_launch_moe_topk_scalar_fp16_spm_kernel(
            SPM_ALLOC.addr(0, key_in_off), SPM_ALLOC.addr(0, value_in_off),
            SPM_ALLOC.addr(0, key_out_off), SPM_ALLOC.addr(0, value_out_off),
            n, c, k, descending, cvt_key, static_cast<int>(num_cores));

        const at::ScalarType key_dtype = cvt_key ? at::kShort : at::kHalf;
        at::Tensor output_keys = at::empty(
            {num_cores, n, k},
            at::TensorOptions().dtype(key_dtype).device(at::kCPU));
        at::Tensor output_values = at::empty(
            {num_cores, n, k},
            at::TensorOptions().dtype(input_values.scalar_type()).device(at::kCPU));
        std::vector<uint8_t> key_output_transport(output_transport_bytes);
        std::vector<uint8_t> value_output_transport(output_transport_bytes);
        for (int core = 0; core < num_cores; ++core) {
            std::memcpy(key_output_transport.data(),
                        SPM_ALLOC.cpu_ptr(core, key_out_off),
                        output_transport_bytes);
            std::memcpy(value_output_transport.data(),
                        SPM_ALLOC.cpu_ptr(core, value_out_off),
                        output_transport_bytes);
            std::memcpy(static_cast<char*>(output_keys.data_ptr()) +
                            core * output_bytes,
                        key_output_transport.data(), output_bytes);
            std::memcpy(static_cast<char*>(output_values.data_ptr()) +
                            core * output_bytes,
                        value_output_transport.data(), output_bytes);
        }
        SPM_ALLOC.reset_temporary();
        return {output_keys, output_values};
    } catch (...) {
        SPM_ALLOC.reset_temporary();
        throw;
    }
}

// Conformance-only kvsort wrapper. merge_all=false exposes [N,ceil(C/4096),K];
// merge_all=true returns global [N,K].
std::tuple<at::Tensor, at::Tensor> rpu_moe_topk_by_kvsort_test(
    const at::Tensor& input_keys,
    const at::Tensor& input_values,
    int64_t k,
    bool descending,
    bool merge_all,
    int64_t num_cores) {
    RECORD_FUNCTION("rpu::moe_topk_by_kvsort_test", {});
    TORCH_CHECK(input_keys.device().is_cpu() && input_values.device().is_cpu(),
                "moe_topk_by_kvsort_test: inputs must be CPU staging tensors");
    TORCH_CHECK(input_keys.scalar_type() == at::kHalf &&
                    is_raw16_dtype(input_values.scalar_type()),
                "moe_topk_by_kvsort_test: keys must be fp16 and values fp16/int16");
    TORCH_CHECK(input_keys.is_contiguous() && input_values.is_contiguous() &&
                    input_keys.dim() == 2 &&
                    input_values.sizes() == input_keys.sizes(),
                "moe_topk_by_kvsort_test: inputs must be contiguous, equal [N,C]");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "moe_topk_by_kvsort_test: num_cores must be in [1,8]");

    const int64_t n = input_keys.size(0);
    const int64_t c = input_keys.size(1);
    TORCH_CHECK(n > 0 && n <= std::numeric_limits<uint16_t>::max(),
                "moe_topk_by_kvsort_test: N must fit u16, got ", n);
    TORCH_CHECK(c > 0 &&
                    c <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                "moe_topk_by_kvsort_test: C must fit u32, got ", c);
    TORCH_CHECK(k > 10 && k <= c && k <= KVSORT_SLICE_SIZE,
                "moe_topk_by_kvsort_test: require "
                "10 < k <= min(C,4096), got N=", n, " C=", c, " K=", k);
    const int64_t c_pad = static_cast<int64_t>(align16(c));
    TORCH_CHECK(c_pad / 16 <= std::numeric_limits<uint16_t>::max(),
                "moe_topk_by_kvsort_test: align16(C)/16 must fit u16, got ",
                c_pad / 16);
    const int64_t slices =
        (c_pad + KVSORT_SLICE_SIZE - 1) / KVSORT_SLICE_SIZE;
    const int64_t k_pad = static_cast<int64_t>(align16(k));
    const int64_t output_slices = merge_all ? 1 : slices;
    const size_t input_physical_elements = static_cast<size_t>(n * c_pad);
    const size_t output_physical_elements =
        static_cast<size_t>(n * output_slices * k_pad);
    const size_t input_bytes = input_physical_elements * sizeof(uint16_t);
    const size_t output_bytes = output_physical_elements * sizeof(uint16_t);
    const size_t workspace_half_bytes =
        static_cast<size_t>(n * slices * k_pad) * sizeof(uint16_t);
    const uint16_t pad_key = descending ? NEG_INF_FP16 : POS_INF_FP16;

    std::vector<uint16_t> key_input_transport(input_physical_elements, pad_key);
    std::vector<uint16_t> value_input_transport(input_physical_elements, 0);
    const auto* key_src = static_cast<const uint16_t*>(input_keys.data_ptr());
    const auto* value_src = static_cast<const uint16_t*>(input_values.data_ptr());
    for (int64_t row = 0; row < n; ++row) {
        std::memcpy(key_input_transport.data() + row * c_pad,
                    key_src + row * c,
                    static_cast<size_t>(c) * sizeof(uint16_t));
        std::memcpy(value_input_transport.data() + row * c_pad,
                    value_src + row * c,
                    static_cast<size_t>(c) * sizeof(uint16_t));
    }

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    SPM_ALLOC.reset_temporary();
    const uint32_t key_in_off = alloc_spm_bytes(input_bytes);
    const uint32_t value_in_off = alloc_spm_bytes(input_bytes);
    const uint32_t key_out_off = alloc_spm_bytes(output_bytes);
    const uint32_t value_out_off = alloc_spm_bytes(output_bytes);
    uint32_t workspace_key_off = 0;
    uint32_t workspace_value_off = 0;
    if (merge_all && c > KVSORT_SLICE_SIZE) {
        workspace_key_off = alloc_spm_bytes(workspace_half_bytes);
        workspace_value_off = alloc_spm_bytes(workspace_half_bytes);
    }

    try {
        for (int core = 0; core < num_cores; ++core) {
            std::memcpy(SPM_ALLOC.cpu_ptr(core, key_in_off),
                        key_input_transport.data(), input_bytes);
            std::memcpy(SPM_ALLOC.cpu_ptr(core, value_in_off),
                        value_input_transport.data(), input_bytes);
        }

        if (merge_all) {
            rpu_launch_moe_topk_by_kvsort_fp16_spm_kernel(
                SPM_ALLOC.addr(0, key_in_off), SPM_ALLOC.addr(0, value_in_off),
                SPM_ALLOC.addr(0, key_out_off), SPM_ALLOC.addr(0, value_out_off),
                workspace_key_off == 0 ? 0 : SPM_ALLOC.addr(0, workspace_key_off),
                workspace_value_off == 0 ? 0
                                         : SPM_ALLOC.addr(0, workspace_value_off),
                n, c, k, descending, static_cast<int>(num_cores));
        } else {
            launch_moe_topk_by_kvsort_fp16_pass(
                SPM_ALLOC.addr(0, key_in_off), SPM_ALLOC.addr(0, value_in_off),
                SPM_ALLOC.addr(0, key_out_off), SPM_ALLOC.addr(0, value_out_off),
                n, c, k, descending, static_cast<int>(num_cores));
        }

        std::vector<int64_t> output_shape = merge_all
            ? std::vector<int64_t>{num_cores, n, k}
            : std::vector<int64_t>{num_cores, n, slices, k};
        at::Tensor output_keys = at::empty(
            output_shape,
            at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
        at::Tensor output_values = at::empty(
            output_shape,
            at::TensorOptions().dtype(input_values.scalar_type()).device(at::kCPU));
        std::vector<uint16_t> key_output_transport(output_physical_elements);
        std::vector<uint16_t> value_output_transport(output_physical_elements);
        auto* key_dst = static_cast<uint16_t*>(output_keys.data_ptr());
        auto* value_dst = static_cast<uint16_t*>(output_values.data_ptr());
        const size_t logical_block_bytes = static_cast<size_t>(k) * sizeof(uint16_t);
        for (int64_t core = 0; core < num_cores; ++core) {
            std::memcpy(key_output_transport.data(),
                        SPM_ALLOC.cpu_ptr(static_cast<int>(core), key_out_off),
                        output_bytes);
            std::memcpy(value_output_transport.data(),
                        SPM_ALLOC.cpu_ptr(static_cast<int>(core), value_out_off),
                        output_bytes);
            for (int64_t block = 0; block < n * output_slices; ++block) {
                const size_t physical_offset = static_cast<size_t>(block * k_pad);
                const size_t logical_offset = static_cast<size_t>(
                    (core * n * output_slices + block) * k);
                std::memcpy(key_dst + logical_offset,
                            key_output_transport.data() + physical_offset,
                            logical_block_bytes);
                std::memcpy(value_dst + logical_offset,
                            value_output_transport.data() + physical_offset,
                            logical_block_bytes);
            }
        }
        SPM_ALLOC.reset_temporary();
        return {output_keys, output_values};
    } catch (...) {
        SPM_ALLOC.reset_temporary();
        throw;
    }
}

// Conformance-only CPU staging for bincount_scalar.
// Input is a sorted one-dimensional expert-ID array.
std::tuple<at::Tensor, at::Tensor> rpu_moe_bincount_scalar_test(
    const at::Tensor& input_keys,
    int64_t maxk,
    bool cvt_key,
    bool output_cvt_key,
    int64_t num_cores) {
    RECORD_FUNCTION("rpu::moe_bincount_scalar_test", {});
    TORCH_CHECK(input_keys.device().is_cpu(),
                "moe_bincount_scalar_test: input must be a CPU staging tensor");
    TORCH_CHECK(input_keys.is_contiguous() && input_keys.dim() == 1,
                "moe_bincount_scalar_test: input must be contiguous [C]");
    TORCH_CHECK(is_raw16_dtype(input_keys.scalar_type()),
                "moe_bincount_scalar_test: input must be fp16 or int16");
    TORCH_CHECK((cvt_key && input_keys.scalar_type() == at::kHalf) ||
                    (!cvt_key && input_keys.scalar_type() == at::kShort),
                "moe_bincount_scalar_test: cvt_key=true requires fp16 input; "
                "cvt_key=false requires raw int16/U16 input");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "moe_bincount_scalar_test: num_cores must be in [1,8]");

    const int64_t c = input_keys.numel();
    TORCH_CHECK(c > 0 && c <= std::numeric_limits<uint16_t>::max(),
                "moe_bincount_scalar_test: C must be in [1,65535], got ", c);
    TORCH_CHECK(maxk > 0 && maxk < 1024,
                "moe_bincount_scalar_test: maxk must be in [1,1023], got ",
                maxk);
    const size_t input_bytes = static_cast<size_t>(c) * sizeof(uint16_t);
    const size_t key_output_bytes = input_bytes;
    const size_t bin_output_bytes =
        static_cast<size_t>(maxk) * sizeof(uint16_t);
    const size_t input_transport_bytes = align16(input_bytes);
    const size_t key_output_transport_bytes = align16(key_output_bytes);
    const size_t bin_output_transport_bytes = align16(bin_output_bytes);
    std::vector<uint8_t> input_transport(input_transport_bytes, 0);
    std::memcpy(input_transport.data(), input_keys.data_ptr(), input_bytes);

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    SPM_ALLOC.reset_temporary();
    const uint32_t input_key_off = alloc_spm_bytes(input_bytes);
    const uint32_t output_key_off = alloc_spm_bytes(key_output_bytes);
    const uint32_t output_bin_off = alloc_spm_bytes(bin_output_bytes);

    try {
        for (int core = 0; core < num_cores; ++core) {
            std::memcpy(SPM_ALLOC.cpu_ptr(core, input_key_off),
                        input_transport.data(), input_transport_bytes);
        }

        rpu_launch_moe_bincount_scalar_spm_kernel(
            SPM_ALLOC.addr(0, input_key_off),
            SPM_ALLOC.addr(0, output_key_off),
            SPM_ALLOC.addr(0, output_bin_off), c, maxk, cvt_key,
            output_cvt_key, static_cast<int>(num_cores));

        const at::ScalarType output_key_dtype =
            output_cvt_key ? at::kShort : input_keys.scalar_type();
        at::Tensor output_keys = at::empty(
            {num_cores, c},
            at::TensorOptions().dtype(output_key_dtype).device(at::kCPU));
        at::Tensor output_bins = at::empty(
            {num_cores, maxk},
            at::TensorOptions().dtype(at::kShort).device(at::kCPU));
        std::vector<uint8_t> key_output_transport(key_output_transport_bytes);
        std::vector<uint8_t> bin_output_transport(bin_output_transport_bytes);
        for (int64_t core = 0; core < num_cores; ++core) {
            std::memcpy(key_output_transport.data(),
                        SPM_ALLOC.cpu_ptr(static_cast<int>(core), output_key_off),
                        key_output_transport_bytes);
            std::memcpy(bin_output_transport.data(),
                        SPM_ALLOC.cpu_ptr(static_cast<int>(core), output_bin_off),
                        bin_output_transport_bytes);
            std::memcpy(static_cast<char*>(output_keys.data_ptr()) +
                            core * key_output_bytes,
                        key_output_transport.data(), key_output_bytes);
            std::memcpy(static_cast<char*>(output_bins.data_ptr()) +
                            core * bin_output_bytes,
                        bin_output_transport.data(), bin_output_bytes);
        }
        SPM_ALLOC.reset_temporary();
        return {output_keys, output_bins};
    } catch (...) {
        SPM_ALLOC.reset_temporary();
        throw;
    }
}
