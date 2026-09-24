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

// ======================== Slice kernel launch ========================
void rpu_launch_slice_kernel(const at::Tensor &input,
                             at::Tensor &output,
                             const SliceParams &params) {

    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
                "input not on RPU");

    // Ensure output has same dtype as input
    if (output.scalar_type() != input.scalar_type()) {
        output = output.to(input.scalar_type());
    }

    using namespace ::rhino_lkn;

    const size_t dwidth = sizeof(c10::Half);  // 2 bytes for FP16
    const int64_t ndim = input.dim();

    // Ensure input is half type
    TORCH_CHECK(input.scalar_type() == at::kHalf,
                "rpu_launch_slice_kernel: only FP16 is supported");

    // Get base tensor
    at::Tensor base;
    if (input._base().defined()) {
        base = input._base();
    } else {
        base = input;
    }

    // Get shape info
    std::vector<int64_t> input_shape(ndim);
    std::vector<int64_t> output_shape(ndim);
    for (int i = 0; i < ndim; ++i) {
        input_shape[i] = base.size(i); // 输入是base的input
        output_shape[i] = output.size(i); // 输出本身不变
    }

    size_t input_nbytes = base.numel() * dwidth;
    size_t output_nbytes = output.numel() * dwidth;

    if (log_at(4)) {
        std::cout << "[rpu_launch_slice_kernel] input_nbytes = " << input_nbytes
                  << ", output_nbytes = " << output_nbytes << std::endl;
        // print_tensor_info(base, "rpu_launch_slice_kernel - base");
        // print_tensor_info(output, "rpu_launch_slice_kernel - output");
        std::cout << "[rpu_launch_slice_kernel] axes: ";
        for (auto a : params.axes) std::cout << a << " ";
        std::cout << ", begins: ";
        for (auto b : params.begins) std::cout << b << " ";
        std::cout << ", ends: ";
        for (auto e : params.ends) std::cout << e << " ";
        std::cout << ", steps: ";
        for (auto s : params.steps) std::cout << s << " ";
        std::cout << std::endl;
    }

    // Calculate strides in elements
    std::vector<int64_t> input_step(ndim);
    std::vector<int64_t> output_step(ndim);
    int64_t stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        input_step[i] = stride;
        stride *= input_shape[i];
    }
    stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        output_step[i] = stride;
        stride *= output_shape[i];
    }

    // Calculate start offset in bytes
    int64_t start_offset = 0;
    for (size_t i = 0; i < params.axes.size(); ++i) {
        start_offset += params.begins[i] * input_step[params.axes[i]] * dwidth;
    }
    if (log_at(4)) {
        std::cout << "[rpu_launch_slice_kernel] start_offset = " << start_offset << " bytes" << std::endl;
    }

    // The production slice kernels address input/output directly in DDR using
    // 256-byte units. The commented SPM prototype below is retained only as
    // historical implementation context.
    constexpr uint16_t input_enable_256b = 1;
    constexpr uint16_t output_enable_256b = 1;

    c10::Half *in_ptr = base.data_ptr<c10::Half>();
    c10::Half *out_ptr = output.data_ptr<c10::Half>();

    // 直接使用 tensor 的 DDR 地址，无需中间 buffer
    // rpu_ddr_flush(in_ptr);


    // Get last axis info
    int64_t last_axis = params.axes.back();
    int64_t last_step = params.steps.back();
    int64_t last_slice_num = output_shape[last_axis];  // number of elements in output along sliced axis

    // Select kernel based on slice pattern
    std::string kernel_name;
    if (last_axis == ndim - 1 && last_step == 1) {
        kernel_name = "slice_last_dim_axis";
    } else {
        kernel_name = "slice_single_axis";
    }

    Kernel_t* kernel = KernelCache::instance().get_kernel(kernel_name);
    TORCH_CHECK(kernel != nullptr, "Failed to get ", kernel_name, " kernel from cache");
    kernel->reset_regs();

    if (log_at(4)) {
        std::cout << "[rpu_launch_slice_kernel] Using kernel: " << kernel_name << std::endl;
    }

    // Calculate common parameters
    int64_t inner_num = input_step[last_axis];
    int64_t input_inner_byte_step = inner_num * last_step * dwidth;
    int64_t output_inner_byte_step = inner_num * dwidth;

    // Work queue
    // Context_t ctx;
    auto* wq = GET_QUEUE(1);

    // Set kernel parameters based on kernel type
    if (kernel_name == "slice_last_dim_axis") {
        // slice_last_dim_axis: optimized for last dimension slice with step=1
        int64_t middle_num, outer_num;
        int64_t input_axis1_byte_step, output_axis1_byte_step;
        int64_t input_axis0_byte_step, output_axis0_byte_step;

        if (params.axes.size() == 2) {
            int64_t axis0 = params.axes[0];
            int64_t axis1 = params.axes[1];  // 等价于 axes[-1]

            // middle_num = prod(min(input_shape, output_shape)[axis0:axis1])
            middle_num = 1;
            for (int i = axis0; i < axis1; ++i) {
                middle_num *= std::min(input_shape[i], output_shape[i]);
            }

            // outer_num = prod(output_shape[:axis0])
            outer_num = 1;
            for (int i = 0; i < axis0; ++i) {
                outer_num *= output_shape[i];
            }

            // byte steps
            input_axis1_byte_step = 1;
            for (int i = axis1; i < ndim; ++i) {
                input_axis1_byte_step *= input_shape[i];
            }
            input_axis1_byte_step *= dwidth;

            input_axis0_byte_step = 1;
            for (int i = axis0; i < ndim; ++i) {
                input_axis0_byte_step *= input_shape[i];
            }
            input_axis0_byte_step *= dwidth;

            output_axis1_byte_step = 1;
            for (int i = axis1; i < ndim; ++i) {
                output_axis1_byte_step *= output_shape[i];
            }
            output_axis1_byte_step *= dwidth;

            output_axis0_byte_step = 1;
            for (int i = axis0; i < ndim; ++i) {
                output_axis0_byte_step *= output_shape[i];
            }
            output_axis0_byte_step *= dwidth;

        } else {
            // Single axis (last axis)
            int64_t axis = params.axes[0]; // axes[-1]

            // middle_num = prod(output_shape[:axis])
            middle_num = 1;
            for (int i = 0; i < axis; ++i) {
                middle_num *= output_shape[i];
            }
            outer_num = 1;
            // byte steps
            input_axis1_byte_step = 1;
            for (int i = axis; i < ndim; ++i) {
                input_axis1_byte_step *= input_shape[i];
            }
            input_axis1_byte_step *= dwidth;

            input_axis0_byte_step = input_axis1_byte_step;

            output_axis1_byte_step = 1;
            for (int i = axis; i < ndim; ++i) {
                output_axis1_byte_step *= output_shape[i];
            }
            output_axis1_byte_step *= dwidth;

            output_axis0_byte_step = output_axis1_byte_step;
        }

        // Calculate workload distribution
        const int64_t max_slice_num_per_wrp = 1024;
        const int64_t warp_size = 16;
        int64_t slice_num_per_wrp = std::min(max_slice_num_per_wrp, last_slice_num);

        int64_t middle_num_per_thd = std::min(
            CeilDiv(max_slice_num_per_wrp, warp_size) / CeilDiv(slice_num_per_wrp, warp_size),
            CeilDiv(middle_num, warp_size)
        );
        // if (middle_num_per_thd < 1) middle_num_per_thd = 1;
        // concat 算子
        assert(middle_num_per_thd >= 1);
        int64_t middle_num_per_wrp = middle_num_per_thd * warp_size;

        int64_t outer_num_per_wrp = CeilDiv(max_slice_num_per_wrp, warp_size) /
                                   CeilDiv(slice_num_per_wrp, warp_size) / middle_num_per_thd;
        // if (outer_num_per_wrp < 1) outer_num_per_wrp = 1;
        assert(outer_num_per_wrp >= 1);

        // Grid dimensions
        int64_t grid_dim_x = CeilDiv(last_slice_num, slice_num_per_wrp);
        int64_t grid_dim_y = CeilDiv(middle_num, middle_num_per_wrp);
        int64_t grid_dim_z = CeilDiv(outer_num, outer_num_per_wrp);

        // Calculate addresses (直接使用 tensor DDR 地址)
        uint64_t input_addr = RpuGetDevAddr(in_ptr);
        uint64_t output_addr = RpuGetDevAddr(out_ptr);

        if (input_enable_256b) input_addr >>= 8;
        if (output_enable_256b) output_addr >>= 8;

        // Set launch parameters.
        kernel->set_regs(0, (uint16_t)(input_addr & 0xFFFF));
        kernel->set_regs(1, (uint16_t)(input_addr >> 16));
        kernel->set_regs(2, (uint16_t)(output_addr & 0xFFFF));
        kernel->set_regs(3, (uint16_t)(output_addr >> 16));
        kernel->set_regs(4, input_enable_256b);
        kernel->set_regs(5, output_enable_256b);
        kernel->set_regs(6, (uint16_t)input_shape[last_axis]);
        kernel->set_regs(7, (uint16_t)last_slice_num);
        kernel->set_regs(8, (uint16_t)(input_inner_byte_step & 0xFFFF));
        kernel->set_regs(9, (uint16_t)(input_inner_byte_step >> 16));
        kernel->set_regs(10, (uint16_t)(output_inner_byte_step & 0xFFFF));
        kernel->set_regs(11, (uint16_t)(output_inner_byte_step >> 16));
        kernel->set_regs(12, (uint16_t)(input_axis1_byte_step & 0xFFFF));
        kernel->set_regs(13, (uint16_t)(input_axis1_byte_step >> 16));
        kernel->set_regs(14, (uint16_t)(output_axis1_byte_step & 0xFFFF));
        kernel->set_regs(15, (uint16_t)(output_axis1_byte_step >> 16));
        kernel->set_regs(16, (uint16_t)(inner_num & 0xFFFF));
        kernel->set_regs(17, (uint16_t)(inner_num >> 16));
        kernel->set_regs(18, (uint16_t)(middle_num & 0xFFFF));
        kernel->set_regs(19, (uint16_t)(middle_num >> 16));
        kernel->set_regs(20, (uint16_t)(input_axis0_byte_step & 0xFFFF));
        kernel->set_regs(21, (uint16_t)(input_axis0_byte_step >> 16));
        kernel->set_regs(22, (uint16_t)(output_axis0_byte_step & 0xFFFF));
        kernel->set_regs(23, (uint16_t)(output_axis0_byte_step >> 16));
        kernel->set_regs(24, (uint16_t)(outer_num & 0xFFFF));
        kernel->set_regs(25, (uint16_t)(outer_num >> 16));
        kernel->set_regs(26, (uint16_t)slice_num_per_wrp);
        kernel->set_regs(27, (uint16_t)middle_num_per_wrp);
        kernel->set_regs(28, (uint16_t)outer_num_per_wrp);
        // 有对齐要求，需要从偶数开始
        kernel->set_regs(30, static_cast<uint16_t>(start_offset & 0xFFFF));
        kernel->set_regs(31, (uint16_t)(start_offset >> 16));
        kernel->set_regs(64, (uint16_t)grid_dim_x);
        kernel->set_regs(65, (uint16_t)grid_dim_y);
        kernel->set_regs(66, (uint16_t)grid_dim_z);

        if (log_at(4)) {
            std::cout << "=== Slice Kernel Parameters ===" << std::endl;

            std::cout << "input_addr=" << std::hex << input_addr << std::dec
                      << ", output_addr=" << std::hex << output_addr << std::dec << std::endl;
            std::cout << "input_inner_byte_step=" << input_inner_byte_step
                      << ", output_inner_byte_step=" << output_inner_byte_step << std::endl;
            std::cout << "input_axis1_byte_step=" << input_axis1_byte_step
                      << ", output_axis1_byte_step=" << output_axis1_byte_step << std::endl;
            std::cout << "input_axis0_byte_step=" << input_axis0_byte_step
                      << ", output_axis0_byte_step=" << output_axis0_byte_step << std::endl;
            std::cout << "inner_num=" << inner_num
                      << ", middle_num=" << middle_num
                      << ", outer_num=" << outer_num << std::endl;
            std::cout << "slice_num_per_wrp=" << slice_num_per_wrp
                      << ", middle_num_per_wrp=" << middle_num_per_wrp
                      << ", outer_num_per_wrp=" << outer_num_per_wrp << std::endl;
            std::cout << "input shape[last_axis]=" << input_shape[last_axis]
                      << ", last_slice_num=" << last_slice_num << std::endl;
            std::cout << "enabled_256b: input=" << input_enable_256b
                      << ", output=" << output_enable_256b << std::endl;
            std::cout << "middle_num_per_thd=" << middle_num_per_thd << std::endl;
            std::cout << "grid=(" << grid_dim_x << "," << grid_dim_y << "," << grid_dim_z << ")" << std::endl;
        }
        // ctx.recode_queue(&wq);
        // ctx.record_buffer(&ddr_input);
        // ctx.record_buffer(&ddr_output);
        // ctx.dump_init_buf();

        // Enqueue kernel
        wq->enqueu_kernel(*kernel, {(uint16_t)grid_dim_x, (uint16_t)grid_dim_y, (uint16_t)grid_dim_z}, {0});

    } else {
        // slice_single_axis: general single axis slice
        int64_t outer_num = 1;
        for (int i = 0; i < last_axis; ++i) {
            outer_num *= input_shape[i];
        }
        int64_t input_outer_byte_step = 1;
        for (int i = last_axis; i < ndim; ++i) {
            input_outer_byte_step *= input_shape[i];
        }
        input_outer_byte_step *= dwidth;

        int64_t output_outer_byte_step = 1;
        for (int i = last_axis; i < ndim; ++i) {
            output_outer_byte_step *= output_shape[i];
        }
        output_outer_byte_step *= dwidth;

        // Fixed workload distribution for slice_single_axis
        const int64_t inner_num_per_wrp = 4096;
        const int64_t slice_num_per_wrp = 16;

        // Grid dimensions
        int64_t grid_dim_x = CeilDiv(inner_num, inner_num_per_wrp);
        int64_t grid_dim_y = CeilDiv(last_slice_num, slice_num_per_wrp);
        int64_t grid_dim_z = outer_num;

        // Calculate addresses (直接使用 tensor DDR 地址)
        uint64_t input_addr = RpuGetDevAddr(in_ptr);
        uint64_t output_addr = RpuGetDevAddr(out_ptr);
        if (input_enable_256b) input_addr >>= 8;
        if (output_enable_256b) output_addr >>= 8;

        // ctx.recode_queue(&wq);
        // ctx.record_buffer(&ddr_input);
        // ctx.record_buffer(&ddr_output);
        // ctx.dump_init_buf();

        // Set registers
        kernel->set_regs(0, (uint16_t)(input_addr & 0xFFFF));
        kernel->set_regs(1, (uint16_t)(input_addr >> 16));
        kernel->set_regs(2, (uint16_t)(output_addr & 0xFFFF));
        kernel->set_regs(3, (uint16_t)(output_addr >> 16));
        kernel->set_regs(4, input_enable_256b);
        kernel->set_regs(5, output_enable_256b);
        kernel->set_regs(6, (uint16_t)input_shape[last_axis]);
        kernel->set_regs(7, (uint16_t)last_slice_num);
        kernel->set_regs(8, (uint16_t)(input_inner_byte_step & 0xFFFF));
        kernel->set_regs(9, (uint16_t)(input_inner_byte_step >> 16));
        kernel->set_regs(10, (uint16_t)(output_inner_byte_step & 0xFFFF));
        kernel->set_regs(11, (uint16_t)(output_inner_byte_step >> 16));
        kernel->set_regs(12, (uint16_t)(input_outer_byte_step & 0xFFFF));
        kernel->set_regs(13, (uint16_t)(input_outer_byte_step >> 16));
        kernel->set_regs(14, (uint16_t)(output_outer_byte_step & 0xFFFF));
        kernel->set_regs(15, (uint16_t)(output_outer_byte_step >> 16));
        kernel->set_regs(16, (uint16_t)(inner_num & 0xFFFF));
        kernel->set_regs(17, (uint16_t)(inner_num >> 16));
        kernel->set_regs(18, (uint16_t)(start_offset));
        kernel->set_regs(64, (uint16_t)grid_dim_x);
        kernel->set_regs(65, (uint16_t)grid_dim_y);
        kernel->set_regs(66, (uint16_t)grid_dim_z);

        if (log_at(4)) {
            std::cout << "\n[slice_single_axis] Register Parameters:" << std::endl;
            std::cout << "  param[0]  = " << std::hex << (input_addr & 0xFFFF) << std::dec << "  # input_addr low" << std::endl;
            std::cout << "  param[1]  = " << std::hex << (input_addr >> 16) << std::dec << "  # input_addr high" << std::endl;
            std::cout << "  param[2]  = " << std::hex << (output_addr & 0xFFFF) << std::dec << "  # output_addr low" << std::endl;
            std::cout << "  param[3]  = " << std::hex << (output_addr >> 16) << std::dec << "  # output_addr high" << std::endl;
            std::cout << "  param[4]  = " << input_enable_256b << "  # input_enable_256b" << std::endl;
            std::cout << "  param[5]  = " << output_enable_256b << "  # output_enable_256b" << std::endl;
            std::cout << "  param[6]  = " << input_shape[last_axis] << "  # shape[axes[-1]]" << std::endl;
            std::cout << "  param[7]  = " << last_slice_num << "  # last_slice_num" << std::endl;
            std::cout << "  param[8]  = " << (input_inner_byte_step & 0xFFFF) << "  # input_inner_byte_step low" << std::endl;
            std::cout << "  param[9]  = " << (input_inner_byte_step >> 16) << "  # input_inner_byte_step high" << std::endl;
            std::cout << "  param[10] = " << (output_inner_byte_step & 0xFFFF) << "  # output_inner_byte_step low" << std::endl;
            std::cout << "  param[11] = " << (output_inner_byte_step >> 16) << "  # output_inner_byte_step high" << std::endl;
            std::cout << "  param[12] = " << (input_outer_byte_step & 0xFFFF) << "  # input_outer_byte_step low" << std::endl;
            std::cout << "  param[13] = " << (input_outer_byte_step >> 16) << "  # input_outer_byte_step high" << std::endl;
            std::cout << "  param[14] = " << (output_outer_byte_step & 0xFFFF) << "  # output_outer_byte_step low" << std::endl;
            std::cout << "  param[15] = " << (output_outer_byte_step >> 16) << "  # output_outer_byte_step high" << std::endl;
            std::cout << "  param[16] = " << (inner_num & 0xFFFF) << "  # inner_num low" << std::endl;
            std::cout << "  param[17] = " << (inner_num >> 16) << "  # inner_num high" << std::endl;
            std::cout << "  param[18] = " << start_offset << "  # start_offset" << std::endl;
            std::cout << "  param[64] = " << grid_dim_x << "  # gridDim.x" << std::endl;
            std::cout << "  param[65] = " << grid_dim_y << "  # gridDim.y" << std::endl;
            std::cout << "  param[66] = " << grid_dim_z << "  # gridDim.z" << std::endl;
        }

        // Enqueue kernel
        wq->enqueu_kernel(*kernel, {(uint16_t)grid_dim_x, (uint16_t)grid_dim_y, (uint16_t)grid_dim_z}, {0});
    }

    // Flush output, 确保 CPU 能读到 RPU 写入的最新数据
    // rpu_ddr_flush(out_ptr);

    // ==================== SPM 版本 (注释保留) ====================
    // spm_to_ddr(out_ptr, spm_output.get_cpu_ptr(), aligned_output_nbytes);

    // ctx.export_case();

}

// ============ Slice SPM Kernel Launch (multi-core, fused-graph) ============
// SPM-resident slice: out[output_shape] = in[input_shape] sliced at `begins`
// (step=1, sliced axes = where input_shape != output_shape). SAME slice_*
// kernels as the eager rpu_launch_slice_kernel above (and my insert_slice — slice
// is insert_slice with src/dest flipped). Differs in: operands ALREADY in SPM
// (caller passes byte addresses, in256/out256=0, no DDR 256b path), graph-aware
// kernel fetch, multi-core broadcast (each core slices its OWN per-core SPM). The
// read-side start_offset is folded into in_addr (reg18 / reg30-31 left 0), exactly
// like insert_slice folds the write-side offset into out_addr. Assumes contiguous
// (step=1) slices, <=2 differing axes — covers the GDN chunk (slice chunk i along
// axis 1 [H,N,C,*]->[H,1,C,*], and the C-1 column slice).
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

// Isolated test: stage `input` into SPM, slice [begins : begins+sizes] (step 1),
// DMA the packed result back. Golden = input narrowed per dim. Single-core.
at::Tensor rpu_slice_spm_test(const at::Tensor& input, c10::IntArrayRef begins,
                              c10::IntArrayRef sizes) {
    TORCH_CHECK(input.scalar_type() == at::kHalf, "slice_spm_test: fp16 only");
    TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                "slice_spm_test: input must be on RPU");
    const int64_t ndim = input.dim();
    TORCH_CHECK((int64_t)begins.size() == ndim && (int64_t)sizes.size() == ndim,
                "slice_spm_test: begins/sizes rank must match input");
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    auto in_c = input.contiguous();

    std::vector<int64_t> in_shape(ndim), out_shape(ndim), beg(ndim);
    int64_t out_numel = 1;
    for (int64_t i = 0; i < ndim; ++i) {
        in_shape[i] = in_c.size(i);
        out_shape[i] = sizes[i];
        beg[i] = begins[i];
        out_numel *= sizes[i];
    }
    const int64_t in_bytes = in_c.numel() * 2, out_bytes = out_numel * 2;
    using AR = SpmAllocator::AllocRequest;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{in_bytes, 1, 2},   // source (read by the kernel)
        AR{out_bytes, 1, 3},  // packed dest (written; alive across kernel + DMA-out)
    });
    const uint32_t in_addr = SPM_ALLOC.addr(0, offsets[0]);
    const uint32_t out_addr = SPM_ALLOC.addr(0, offsets[1]);

    rpu_launch_ddr_broadcast_spm_dma_immediate(
        const_cast<c10::Half*>(in_c.data_ptr<c10::Half>()), in_c.numel(), in_addr, 1);
    rpu_launch_slice_spm_kernel(in_addr, out_addr, in_shape, out_shape, beg, 1);

    auto output = at::empty(out_shape, in_c.options());
    rpu_launch_spm_copy_ddr_dma_immediate(out_addr, output.data_ptr<c10::Half>(), out_numel);
    SPM_ALLOC.reset_temporary();
    return output;
}
