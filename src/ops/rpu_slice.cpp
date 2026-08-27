// rpu_slice.cpp - Slice kernel implementation for RPU
#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"  // for SPM_ALLOC (slice_spm_test only)
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace ::rhino_lkn;

static inline int64_t slice_shp_prod(const std::vector<int64_t>& s,
                                     int64_t lo, int64_t hi) {
    int64_t p = 1;
    for (int64_t i = lo; i < hi; ++i) p *= s[i];
    return p;
}

static size_t slice_checked_multiply(size_t lhs,
                                     size_t rhs,
                                     const char* quantity) {
    TORCH_CHECK(lhs == 0 ||
                    rhs <= std::numeric_limits<size_t>::max() / lhs,
                "slice_spm: ", quantity, " overflows size_t");
    return lhs * rhs;
}

static size_t slice_checked_add(size_t lhs,
                                size_t rhs,
                                const char* quantity) {
    TORCH_CHECK(rhs <= std::numeric_limits<size_t>::max() - lhs,
                "slice_spm: ", quantity, " overflows size_t");
    return lhs + rhs;
}

static void validate_slice_spm_interval(uint64_t address,
                                        size_t byte_count,
                                        const char* interval_name) {
    TORCH_CHECK(byte_count > 0 && SPM_ALLOC.is_initialized(),
                "slice_spm: cannot validate ", interval_name,
                " without a nonempty interval and initialized SPM");
    const uint64_t capacity = SpmAllocator::SPM_USABLE;
    for (int core = 0; core < SpmAllocator::NUM_CORES; ++core) {
        const uint64_t spm_base = SPM_ALLOC.addr(core, 0);
        if (address >= spm_base && address - spm_base <= capacity &&
            byte_count <= capacity - (address - spm_base)) {
            return;
        }
    }
    TORCH_CHECK(false, "slice_spm: complete ", interval_name,
                " interval is outside every current allocator core's SPM");
}

// Multi-core contiguous SPM slice with unit step and at most two sliced axes.
void rpu_launch_slice_spm_kernel(uint32_t input_spm_addr, uint32_t output_spm_addr,
                                 const std::vector<int64_t>& input_shape,
                                 const std::vector<int64_t>& output_shape,
                                 const std::vector<int64_t>& begins,
                                 int num_cores) {
    const int64_t ndim = (int64_t)input_shape.size();
    TORCH_CHECK(ndim >= 1 && (int64_t)output_shape.size() == ndim &&
                    (int64_t)begins.size() == ndim,
                "slice_spm: input/output/begins rank must match");
    TORCH_CHECK(num_cores >= 1 && num_cores <= SpmAllocator::NUM_CORES,
                "slice_spm: num_cores must be in [1, ",
                SpmAllocator::NUM_CORES, "]");
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "slice_spm: SPM allocator is not initialized");
    constexpr int64_t dwidth = sizeof(c10::Half);

    for (int64_t i = 0; i < ndim; ++i) {
        TORCH_CHECK(input_shape[i] > 0 && output_shape[i] > 0,
                    "slice_spm: every input/output dimension must be > 0");
        TORCH_CHECK(output_shape[i] <= input_shape[i] && begins[i] >= 0 &&
                        begins[i] <= input_shape[i] - output_shape[i],
                    "slice_spm: begins/output shape exceed input at axis ", i);
        TORCH_CHECK(
            static_cast<uint64_t>(input_shape[i]) <=
                    static_cast<uint64_t>(
                        std::numeric_limits<size_t>::max()) &&
                static_cast<uint64_t>(output_shape[i]) <=
                    static_cast<uint64_t>(
                        std::numeric_limits<size_t>::max()),
            "slice_spm: a dimension does not fit size_t");
    }

    std::vector<size_t> in_step_size(ndim);
    size_t input_numel = 1;
    size_t output_numel = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        in_step_size[i] = input_numel;
        input_numel = slice_checked_multiply(
            input_numel, static_cast<size_t>(input_shape[i]),
            "input element count");
        output_numel = slice_checked_multiply(
            output_numel, static_cast<size_t>(output_shape[i]),
            "output element count");
    }
    const size_t input_byte_count = slice_checked_multiply(
        input_numel, static_cast<size_t>(dwidth), "input byte count");
    const size_t output_byte_count = slice_checked_multiply(
        output_numel, static_cast<size_t>(dwidth), "output byte count");
    // Require the declared source buffer as well as the actual strided read
    // envelope to remain in SPM.  The stronger source check also bounds every
    // byte stride encoded below before the dynamic Kernel lookup can mutate
    // Graph state.
    validate_slice_spm_interval(
        input_spm_addr, input_byte_count, "declared input");
    validate_slice_spm_interval(
        output_spm_addr, output_byte_count, "write");

    size_t start_element = 0;
    size_t last_element_delta = 0;
    for (int64_t i = 0; i < ndim; ++i) {
        start_element = slice_checked_add(
            start_element,
            slice_checked_multiply(
                static_cast<size_t>(begins[i]), in_step_size[i],
                "input start element"),
            "input start element");
        last_element_delta = slice_checked_add(
            last_element_delta,
            slice_checked_multiply(
                static_cast<size_t>(output_shape[i] - 1),
                in_step_size[i], "last read element"),
            "last read element");
    }
    const size_t last_read_element = slice_checked_add(
        start_element, last_element_delta, "last read element");
    TORCH_CHECK(last_read_element < input_numel,
                "slice_spm: computed read interval exceeds input shape");
    const size_t start_offset = slice_checked_multiply(
        start_element, static_cast<size_t>(dwidth), "input start byte");
    const size_t read_byte_count = slice_checked_multiply(
        last_read_element - start_element + 1,
        static_cast<size_t>(dwidth), "read byte count");
    const size_t in_addr_size = slice_checked_add(
        static_cast<size_t>(input_spm_addr), start_offset,
        "input SPM address");
    TORCH_CHECK(in_addr_size <= std::numeric_limits<uint32_t>::max(),
                "slice_spm: input SPM address exceeds uint32_t");
    validate_slice_spm_interval(
        static_cast<uint64_t>(in_addr_size), read_byte_count, "read");

    std::vector<int64_t> in_step(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        TORCH_CHECK(in_step_size[i] <=
                        static_cast<size_t>(
                            std::numeric_limits<int64_t>::max()),
                    "slice_spm: input stride exceeds int64_t");
        in_step[i] = static_cast<int64_t>(in_step_size[i]);
    }

    std::vector<int64_t> axes;
    for (int64_t i = 0; i < ndim; ++i)
        if (input_shape[i] != output_shape[i]) axes.push_back(i);
    TORCH_CHECK(!axes.empty(), "slice_spm: input == output (nothing to slice)");
    TORCH_CHECK(axes.size() <= 2, "slice_spm: at most 2 sliced axes supported");

    const int64_t last_axis = axes.back();
    const int64_t last_slice_num = output_shape[last_axis];
    const int64_t inner_num = in_step[last_axis];
    const int64_t input_inner_byte_step = inner_num * dwidth;   // step=1
    const int64_t output_inner_byte_step = inner_num * dwidth;

    constexpr uint16_t in256 = 0;   // SPM; never admit the DDR input mode.
    constexpr uint16_t out256 = 0;  // SPM; never admit the DDR output mode.
    static_assert(in256 == 0 && out256 == 0);
    const uint32_t in_addr = static_cast<uint32_t>(in_addr_size);
    const uint32_t out_addr = output_spm_addr;

    const std::string kernel_name =
        (last_axis == ndim - 1) ? "slice_last_dim_axis" : "slice_single_axis";
    auto& graph = RpuKernelGraph::active();
    graph.stage_kernel_no_ddr(kernel_name, GraphKernelNoDdrProof::SliceSpm);
    Kernel_t* kernel = graph.get_kernel_reset(kernel_name);
    TORCH_CHECK(kernel != nullptr, "slice_spm: failed to get ", kernel_name, " kernel");

    int64_t gx, gy, gz;
    if (kernel_name == "slice_last_dim_axis") {
        int64_t middle_num, outer_num;
        int64_t in_axis1, in_axis0, out_axis1, out_axis0;  // byte steps
        if (axes.size() == 2) {
            const int64_t a0 = axes[0], a1 = axes[1];
            middle_num = 1;
            for (int64_t i = a0; i < a1; ++i)
                middle_num *= std::min(input_shape[i], output_shape[i]);
            outer_num = slice_shp_prod(output_shape, 0, a0);
            in_axis1  = slice_shp_prod(input_shape,  a1, ndim) * dwidth;
            in_axis0  = slice_shp_prod(input_shape,  a0, ndim) * dwidth;
            out_axis1 = slice_shp_prod(output_shape, a1, ndim) * dwidth;
            out_axis0 = slice_shp_prod(output_shape, a0, ndim) * dwidth;
        } else {
            middle_num = slice_shp_prod(output_shape, 0, last_axis);
            outer_num  = 1;
            in_axis1  = slice_shp_prod(input_shape,  last_axis, ndim) * dwidth;
            in_axis0  = in_axis1;
            out_axis1 = slice_shp_prod(output_shape, last_axis, ndim) * dwidth;
            out_axis0 = out_axis1;
        }
        const int64_t MAXW = 1024, W = 16;
        const int64_t slice_per_wrp = std::min(MAXW, last_slice_num);
        const int64_t mid_per_thd =
            std::min(CeilDiv(MAXW, W) / CeilDiv(slice_per_wrp, W), CeilDiv(middle_num, W));
        TORCH_CHECK(mid_per_thd >= 1, "slice_spm: mid_per_thd < 1");
        const int64_t mid_per_wrp = mid_per_thd * W;
        const int64_t outer_per_wrp = CeilDiv(MAXW, W) / CeilDiv(slice_per_wrp, W) / mid_per_thd;
        TORCH_CHECK(outer_per_wrp >= 1, "slice_spm: outer_per_wrp < 1");
        gx = CeilDiv(last_slice_num, slice_per_wrp);
        gy = CeilDiv(middle_num, mid_per_wrp);
        gz = CeilDiv(outer_num, outer_per_wrp);

        kernel->set_regs(0,  (uint16_t)(in_addr & 0xFFFF));
        kernel->set_regs(1,  (uint16_t)(in_addr >> 16));
        kernel->set_regs(2,  (uint16_t)(out_addr & 0xFFFF));
        kernel->set_regs(3,  (uint16_t)(out_addr >> 16));
        kernel->set_regs(4,  in256);
        kernel->set_regs(5,  out256);
        kernel->set_regs(6,  (uint16_t)input_shape[last_axis]);
        kernel->set_regs(7,  (uint16_t)last_slice_num);
        kernel->set_regs(8,  (uint16_t)(input_inner_byte_step & 0xFFFF));
        kernel->set_regs(9,  (uint16_t)(input_inner_byte_step >> 16));
        kernel->set_regs(10, (uint16_t)(output_inner_byte_step & 0xFFFF));
        kernel->set_regs(11, (uint16_t)(output_inner_byte_step >> 16));
        kernel->set_regs(12, (uint16_t)(in_axis1 & 0xFFFF));
        kernel->set_regs(13, (uint16_t)(in_axis1 >> 16));
        kernel->set_regs(14, (uint16_t)(out_axis1 & 0xFFFF));
        kernel->set_regs(15, (uint16_t)(out_axis1 >> 16));
        kernel->set_regs(16, (uint16_t)(inner_num & 0xFFFF));
        kernel->set_regs(17, (uint16_t)(inner_num >> 16));
        kernel->set_regs(18, (uint16_t)(middle_num & 0xFFFF));
        kernel->set_regs(19, (uint16_t)(middle_num >> 16));
        kernel->set_regs(20, (uint16_t)(in_axis0 & 0xFFFF));
        kernel->set_regs(21, (uint16_t)(in_axis0 >> 16));
        kernel->set_regs(22, (uint16_t)(out_axis0 & 0xFFFF));
        kernel->set_regs(23, (uint16_t)(out_axis0 >> 16));
        kernel->set_regs(24, (uint16_t)(outer_num & 0xFFFF));
        kernel->set_regs(25, (uint16_t)(outer_num >> 16));
        kernel->set_regs(26, (uint16_t)slice_per_wrp);
        kernel->set_regs(27, (uint16_t)mid_per_wrp);
        kernel->set_regs(28, (uint16_t)outer_per_wrp);
        // reg30/31 (start_offset) left 0 — folded into in_addr.
        kernel->set_regs(64, (uint16_t)gx);
        kernel->set_regs(65, (uint16_t)gy);
        kernel->set_regs(66, (uint16_t)gz);
    } else {
        // slice_single_axis: slice along a non-last axis (GDN chunk: [H,N,C,*] axis 1).
        const int64_t outer_num = slice_shp_prod(input_shape, 0, last_axis);
        const int64_t in_outer  = slice_shp_prod(input_shape,  last_axis, ndim) * dwidth;
        const int64_t out_outer = slice_shp_prod(output_shape, last_axis, ndim) * dwidth;
        const int64_t INNERW = 4096, SLICEW = 16;
        gx = CeilDiv(inner_num, INNERW);
        gy = CeilDiv(last_slice_num, SLICEW);
        gz = outer_num;

        kernel->set_regs(0,  (uint16_t)(in_addr & 0xFFFF));
        kernel->set_regs(1,  (uint16_t)(in_addr >> 16));
        kernel->set_regs(2,  (uint16_t)(out_addr & 0xFFFF));
        kernel->set_regs(3,  (uint16_t)(out_addr >> 16));
        kernel->set_regs(4,  in256);
        kernel->set_regs(5,  out256);
        kernel->set_regs(6,  (uint16_t)input_shape[last_axis]);
        kernel->set_regs(7,  (uint16_t)last_slice_num);
        kernel->set_regs(8,  (uint16_t)(input_inner_byte_step & 0xFFFF));
        kernel->set_regs(9,  (uint16_t)(input_inner_byte_step >> 16));
        kernel->set_regs(10, (uint16_t)(output_inner_byte_step & 0xFFFF));
        kernel->set_regs(11, (uint16_t)(output_inner_byte_step >> 16));
        kernel->set_regs(12, (uint16_t)(in_outer & 0xFFFF));
        kernel->set_regs(13, (uint16_t)(in_outer >> 16));
        kernel->set_regs(14, (uint16_t)(out_outer & 0xFFFF));
        kernel->set_regs(15, (uint16_t)(out_outer >> 16));
        kernel->set_regs(16, (uint16_t)(inner_num & 0xFFFF));
        kernel->set_regs(17, (uint16_t)(inner_num >> 16));
        // reg18 (start_offset) left 0 — folded into in_addr.
        kernel->set_regs(64, (uint16_t)gx);
        kernel->set_regs(65, (uint16_t)gy);
        kernel->set_regs(66, (uint16_t)gz);
    }

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {(uint16_t)gx, (uint16_t)gy, (uint16_t)gz}, cores);
}
