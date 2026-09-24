// rpu_permute.cpp - Permute kernel implementation for RPU
#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"  // for SPM_ALLOC (isolated SPM tests only)
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ::rhino_lkn;

// ======================== Permute (Transpose) Kernel Launch ========================
// Implements permute/transpose optimization using transpose kernels
// Supports:
//   3D: input (B, N, C) -> output with permuted dimensions
//   4D: input (Q, B, N, C) -> output with permuted dimensions
void rpu_launch_permute_kernel(const at::Tensor &input,
                               at::Tensor &output,
                               const std::vector<int64_t> &dims) {

    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
                "input not on RPU");

    // Ensure output has same dtype as input
    if (output.scalar_type() != input.scalar_type()) {
        output = output.to(input.scalar_type());
    }

    using namespace ::rhino_lkn;

    size_t dwidth = sizeof(c10::Half);
    int64_t rank = input.dim();

    // 确保输入是 half 类型
    TORCH_CHECK(input.scalar_type() == at::kHalf,
                "rpu_launch_permute_kernel: only FP16 is supported");

    // Get base tensor (the original data before expand)
    at::Tensor base;
    if (input._base().defined()) {
        base = input._base();
    } else {
        base = input;
    }

    size_t input_nbytes = base.numel() * dwidth; // 转置操作不改变元素个数
    size_t output_nbytes = output.numel() * dwidth;

    if (log_at(4)) {
        std::cout << "rpu_launch_permute_kernel: input size = " << input_nbytes
                  << " bytes, output size = " << output_nbytes << " bytes" << std::endl;
        // print_tensor_info(base, "rpu_launch_permute_kernel - base");
        // print_tensor_info(output, "rpu_launch_permute_kernel - output");
        std::cout << "rpu_launch_permute_kernel: dims = [";
        for (size_t i = 0; i < dims.size(); i++) {
            std::cout << dims[i];
            if (i < dims.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // 获取输入输出指针
    c10::Half *in_ptr = base.data_ptr<c10::Half>();
    c10::Half *out_ptr = output.data_ptr<c10::Half>();
    // size_t aligned_input_nbytes = CeilDiv(input_nbytes, 256) * 256;  // DDR 版本对齐到256字节

    // 直接使用 tensor 的 DDR 地址，无需中间 buffer
    // rpu_ddr_flush(in_ptr);

    // glarge format = address / 256
    uint64_t ddr_input_addr = RpuGetDevAddr(in_ptr) >> 8;
    uint64_t ddr_output_addr = RpuGetDevAddr(out_ptr) >> 8;

    if (log_at(4)) {
        std::cout << std::hex << std::showbase;
        std::cout << "[permute] DDR in_ptr           = "
                  << reinterpret_cast<uintptr_t>(in_ptr) << "\n";
        std::cout << "[permute] DDR out_ptr          = "
                  << reinterpret_cast<uintptr_t>(out_ptr) << "\n";
        std::cout << "[permute] ddr_input addr       = "
                  << (ddr_input_addr << 8) << "\n";
        std::cout << "[permute] ddr_output addr      = "
                  << (ddr_output_addr << 8) << "\n";
        std::cout << std::dec;
    }

    // ==================== SPM 版本 (注释保留) ====================
    // RpuDdrFlush(in_ptr);
    // size_t aligned_input_nbytes = CeilDiv(input_nbytes, 32) * 32;
    //
    // // 申请 SPM 空间, 需要对齐到32Bytes
    // LocalSPM_t spm_core0_input(aligned_input_nbytes, read_write, kStride32B, 2, 0);
    // LocalSPM_t spm_core0_output(aligned_input_nbytes, read_write, kStride32B, 2, 0);
    //
    // // 拷贝数据到 SPM
    // std::memcpy(spm_core0_input.get_cpu_ptr(), in_ptr, aligned_input_nbytes);
    //
    // std::cout << std::hex << std::showbase;
    // std::cout << "[permute] DDR in_ptr           = "
    //           << reinterpret_cast<uintptr_t>(in_ptr) << "\n";
    // std::cout << "[permute] DDR out_ptr          = "
    //           << reinterpret_cast<uintptr_t>(out_ptr) << "\n";
    //
    // std::cout << "[permute] SPM input cpu_ptr    = "
    //           << reinterpret_cast<uintptr_t>(spm_core0_input.get_cpu_ptr()) << "\n";
    // std::cout << "[permute] SPM output cpu_ptr   = "
    //           << reinterpret_cast<uintptr_t>(spm_core0_output.get_cpu_ptr()) << "\n";
    //
    // std::cout << std::dec;

    // Queue_t wq(1);
    auto* wq = GET_QUEUE(1);  // 复用缓存的 queue

    // Context for export case
    // Context_t ctx;
    // ctx.recode_queue(&wq);
    // ctx.record_buffer(&ddr_input);
    // ctx.record_buffer(&ddr_output);
    // ctx.dump_init_buf();

    if (rank == 3) {
        // 3D transpose: input shape (B, N, C)
        int64_t b = base.size(0);
        int64_t n = base.size(1);
        int64_t c = base.size(2);

        // 将 dims 转换为 order 字符串
        std::string order = std::to_string(dims[0]) + std::to_string(dims[1]) + std::to_string(dims[2]);

        // ==================== DDR 版本 ====================
        // 选择 DDR kernel 名称
        std::string kernel_name;
        if (order == "102") {
            kernel_name = "transpose_nbc_c16_ddr";
        } else if (order == "201") {
            kernel_name = "transpose_cbn_c16_ddr";
        } else if (order == "021") {
            kernel_name = "transpose_bcn_c16_ddr";
        } else if (order == "210") {
            kernel_name = "transpose_cnb_c16_ddr";
        } else if (order == "120") {
            kernel_name = "transpose_ncb_c16_ddr";
        } else {
            TORCH_CHECK(false, "rpu_launch_permute_kernel: unsupported 3D permute order ", order);
        }

        // ==================== SPM 版本 (注释保留) ====================
        // std::string kernel_name;
        // if (order == "102") {
        //     kernel_name = "transpose_nbc_c16";
        // } else if (order == "201") {
        //     kernel_name = "transpose_cbn_c16";
        // } else if (order == "021") {
        //     kernel_name = "transpose_bcn_c16";
        // } else if (order == "210") {
        //     kernel_name = "transpose_cnb_c16";
        // } else if (order == "120") {
        //     kernel_name = "transpose_ncb_c16";
        // } else {
        //     TORCH_CHECK(false, "rpu_launch_permute_kernel: unsupported 3D permute order ", order);
        // }

        Kernel_t* kernel = KernelCache::instance().get_kernel(kernel_name);
        TORCH_CHECK(kernel != nullptr, "Failed to get ", kernel_name, " kernel from cache");
        kernel->reset_regs();

        // 计算参数
        int64_t c_v16 = CeilDiv(c, V16_NUM);
        int64_t n_v16 = CeilDiv(n, V16_NUM);
        int64_t b_v16 = CeilDiv(b, V16_NUM);

        if (log_at(4)) {
            std::cout << "\n========== rpu_launch_permute_kernel 3D DEBUG ==========\n";
            std::cout << "[permute] kernel_name: " << kernel_name << "\n";
            std::cout << "[permute] order: " << order << "\n";
            std::cout << "[permute] input shape: (" << b << ", " << n << ", " << c << ")\n";
            std::cout << "[permute] input numel: " << base.numel() << "\n";
            std::cout << "[permute] input_nbytes: " << input_nbytes << "\n";
            std::cout << "[permute] output_nbytes: " << output_nbytes << "\n";
            std::cout << "[permute] b_v16=" << b_v16 << ", n_v16=" << n_v16 << ", c_v16=" << c_v16 << "\n";
            std::cout << std::hex << std::showbase;
            std::cout << "[permute] ddr_input_addr (raw): " << ddr_input_addr << "\n";
            std::cout << "[permute] ddr_output_addr (raw): " << ddr_output_addr << "\n";
            std::cout << "[permute] ddr_input_addr << 8: " << (ddr_input_addr << 8) << "\n";
            std::cout << "[permute] ddr_output_addr << 8: " << (ddr_output_addr << 8) << "\n";
            std::cout << std::dec << std::noshowbase;
        }

        // input/output 不带 tail (PyTorch tensor 是紧凑存储的)
        int64_t c_tail = c;  // in_tail = 0
        int64_t n_tail = n;  // out_tail = 0
        int64_t b_tail = b;

        // ==================== SPM 版本 (注释保留) ====================
        // uint32_t spm_in_byte_base = 0;
        // uint32_t spm_out_byte_base = aligned_input_nbytes;

        // VLM 参数计算
        constexpr int64_t MAX_FP16_VLM_V16_PER_THD_LOCAL = 25;
        int64_t c_v16_per_wrp = std::min((int64_t)MAX_FP16_VLM_V16_PER_THD_LOCAL, c_v16);

        int64_t grid_dim_x, grid_dim_y, grid_dim_z;
        int64_t c_per_thd, c_per_wrp, c_v256_per_wrp;

        // 根据 order 的最后一位选择配置方式
        if (order == "102") {
            // tailc config: output tail is C
            int64_t n_v16_per_wrp = std::min(n_v16, MAX_FP16_VLM_V16_PER_THD_LOCAL / c_v16_per_wrp);
            c_v256_per_wrp = c_v16_per_wrp / V16_NUM;

            grid_dim_x = CeilDiv(n_v16, n_v16_per_wrp);
            grid_dim_y = CeilDiv(c_v16, c_v16_per_wrp);
            grid_dim_z = b;

            c_per_thd = c_v16_per_wrp * V16_NUM;
            c_per_wrp = c_v16_per_wrp * V16_NUM;
            int64_t n_per_thd = n_v16_per_wrp * V16_NUM;
            int64_t n_per_wrp = n_v16_per_wrp * V16_NUM;

            // ==================== DDR 版本 ====================
            // 设置寄存器 (参考 get_tailc_config，使用 DDR 地址)
            kernel->set_regs(0, (uint16_t)(ddr_input_addr & 0xFFFF));
            kernel->set_regs(1, (uint16_t)(ddr_input_addr >> 16));
            kernel->set_regs(2, (uint16_t)(ddr_output_addr & 0xFFFF));
            kernel->set_regs(3, (uint16_t)(ddr_output_addr >> 16));
            kernel->set_regs(4, (uint16_t)1);  // data_type: 1 for fp16

            kernel->set_regs(6, (uint16_t)c);
            kernel->set_regs(7, (uint16_t)c_tail);  // c_in_tail
            kernel->set_regs(8, (uint16_t)c_tail);  // c_out_tail

            kernel->set_regs(9, (uint16_t)n);
            kernel->set_regs(10, (uint16_t)n_v16);

            kernel->set_regs(12, (uint16_t)c_per_thd);
            kernel->set_regs(13, (uint16_t)c_per_wrp);
            kernel->set_regs(14, (uint16_t)c_v16_per_wrp);
            kernel->set_regs(15, (uint16_t)c_v256_per_wrp);

            kernel->set_regs(16, (uint16_t)n_per_thd);
            kernel->set_regs(17, (uint16_t)n_per_wrp);
            kernel->set_regs(18, (uint16_t)n_v16_per_wrp);

            kernel->set_regs(19, (uint16_t)b);

            if (log_at(4)) {
                std::cout << "[permute 102] tailc config registers:\n";
                std::cout << "  reg[0-1] ddr_input_addr: lo=" << (ddr_input_addr & 0xFFFF) << ", hi=" << (ddr_input_addr >> 16) << "\n";
                std::cout << "  reg[2-3] ddr_output_addr: lo=" << (ddr_output_addr & 0xFFFF) << ", hi=" << (ddr_output_addr >> 16) << "\n";
                std::cout << "  reg[4] data_type: 1 (fp16)\n";
                std::cout << "  reg[6] c: " << c << "\n";
                std::cout << "  reg[7] c_in_tail: " << c_tail << "\n";
                std::cout << "  reg[8] c_out_tail: " << c_tail << "\n";
                std::cout << "  reg[9] n: " << n << "\n";
                std::cout << "  reg[10] n_v16: " << n_v16 << "\n";
                std::cout << "  reg[12] c_per_thd: " << c_per_thd << "\n";
                std::cout << "  reg[13] c_per_wrp: " << c_per_wrp << "\n";
                std::cout << "  reg[14] c_v16_per_wrp: " << c_v16_per_wrp << "\n";
                std::cout << "  reg[15] c_v256_per_wrp: " << c_v256_per_wrp << "\n";
                std::cout << "  reg[16] n_per_thd: " << n_per_thd << "\n";
                std::cout << "  reg[17] n_per_wrp: " << n_per_wrp << "\n";
                std::cout << "  reg[18] n_v16_per_wrp: " << n_v16_per_wrp << "\n";
                std::cout << "  reg[19] b: " << b << "\n";
            }

            // ==================== SPM 版本 (注释保留) ====================
            // kernel->set_regs(0, (uint16_t)(spm_in_byte_base & 0xFFFF));
            // kernel->set_regs(1, (uint16_t)(spm_in_byte_base >> 16));
            // kernel->set_regs(2, (uint16_t)(spm_out_byte_base & 0xFFFF));
            // kernel->set_regs(3, (uint16_t)(spm_out_byte_base >> 16));

        } else if (order == "201" || order == "021") {
            // tailn config: output tail is N
            int64_t n_v16_per_wrp = std::min(n_v16, MAX_FP16_VLM_V16_PER_THD_LOCAL / c_v16_per_wrp);
            int64_t n_v16_tail = n_v16 + 1;
            c_v256_per_wrp = c_v16_per_wrp / V16_NUM;

            grid_dim_x = CeilDiv(n_v16, n_v16_per_wrp);
            grid_dim_y = CeilDiv(c_v16, c_v16_per_wrp);
            grid_dim_z = b;

            c_per_thd = c_v16_per_wrp * V16_NUM;
            c_per_wrp = c_v16_per_wrp * V16_NUM;
            int64_t n_per_thd = n_v16_per_wrp * V16_NUM;
            int64_t n_per_wrp = n_v16_per_wrp * V16_NUM;

            // ==================== DDR 版本 ====================
            // 设置寄存器 (参考 get_tailn_config，使用 DDR 地址)
            kernel->set_regs(0, (uint16_t)(ddr_input_addr & 0xFFFF));
            kernel->set_regs(1, (uint16_t)(ddr_input_addr >> 16));
            kernel->set_regs(2, (uint16_t)(ddr_output_addr & 0xFFFF));
            kernel->set_regs(3, (uint16_t)(ddr_output_addr >> 16));
            kernel->set_regs(4, (uint16_t)1);  // data_type: 1 for fp16

            kernel->set_regs(6, (uint16_t)c);
            kernel->set_regs(7, (uint16_t)c_tail);

            kernel->set_regs(8, (uint16_t)n);
            kernel->set_regs(9, (uint16_t)n_v16);
            kernel->set_regs(10, (uint16_t)n_tail);
            kernel->set_regs(11, (uint16_t)n_v16_tail);

            kernel->set_regs(12, (uint16_t)c_per_thd);
            kernel->set_regs(13, (uint16_t)c_per_wrp);
            kernel->set_regs(14, (uint16_t)c_v16_per_wrp);
            kernel->set_regs(15, (uint16_t)c_v256_per_wrp);

            kernel->set_regs(16, (uint16_t)n_per_thd);
            kernel->set_regs(17, (uint16_t)n_per_wrp);
            kernel->set_regs(18, (uint16_t)n_v16_per_wrp);

            kernel->set_regs(19, (uint16_t)b);

            if (log_at(4)) {
                std::cout << "[permute " << order << "] tailn config registers:\n";
                std::cout << "  reg[0-1] ddr_input_addr: lo=" << (ddr_input_addr & 0xFFFF) << ", hi=" << (ddr_input_addr >> 16) << "\n";
                std::cout << "  reg[2-3] ddr_output_addr: lo=" << (ddr_output_addr & 0xFFFF) << ", hi=" << (ddr_output_addr >> 16) << "\n";
                std::cout << "  reg[4] data_type: 1 (fp16)\n";
                std::cout << "  reg[6] c: " << c << "\n";
                std::cout << "  reg[7] c_tail: " << c_tail << "\n";
                std::cout << "  reg[8] n: " << n << "\n";
                std::cout << "  reg[9] n_v16: " << n_v16 << "\n";
                std::cout << "  reg[10] n_tail: " << n_tail << "\n";
                std::cout << "  reg[11] n_v16_tail: " << n_v16_tail << "\n";
                std::cout << "  reg[12] c_per_thd: " << c_per_thd << "\n";
                std::cout << "  reg[13] c_per_wrp: " << c_per_wrp << "\n";
                std::cout << "  reg[14] c_v16_per_wrp: " << c_v16_per_wrp << "\n";
                std::cout << "  reg[15] c_v256_per_wrp: " << c_v256_per_wrp << "\n";
                std::cout << "  reg[16] n_per_thd: " << n_per_thd << "\n";
                std::cout << "  reg[17] n_per_wrp: " << n_per_wrp << "\n";
                std::cout << "  reg[18] n_v16_per_wrp: " << n_v16_per_wrp << "\n";
                std::cout << "  reg[19] b: " << b << "\n";
            }

            // ==================== SPM 版本 (注释保留) ====================
            // kernel->set_regs(0, (uint16_t)(spm_in_byte_base & 0xFFFF));
            // kernel->set_regs(1, (uint16_t)(spm_in_byte_base >> 16));
            // kernel->set_regs(2, (uint16_t)(spm_out_byte_base & 0xFFFF));
            // kernel->set_regs(3, (uint16_t)(spm_out_byte_base >> 16));

        } else {
            // tailb config: output tail is B (order "210" or "120")
            int64_t b_v16_per_wrp = std::min(b_v16, MAX_FP16_VLM_V16_PER_THD_LOCAL / c_v16_per_wrp);
            int64_t b_v16_tail = b_v16 + 1;
            c_v256_per_wrp = c_v16_per_wrp / V16_NUM;

            grid_dim_x = CeilDiv(b_v16, b_v16_per_wrp);
            grid_dim_y = CeilDiv(c_v16, c_v16_per_wrp);
            grid_dim_z = n;

            c_per_thd = c_v16_per_wrp * V16_NUM;
            c_per_wrp = c_v16_per_wrp * V16_NUM;
            int64_t b_per_thd = b_v16_per_wrp * V16_NUM;
            int64_t b_per_wrp = b_v16_per_wrp * V16_NUM;

            // ==================== DDR 版本 ====================
            // 设置寄存器 (参考 get_tailb_config，使用 DDR 地址)
            kernel->set_regs(0, (uint16_t)(ddr_input_addr & 0xFFFF));
            kernel->set_regs(1, (uint16_t)(ddr_input_addr >> 16));
            kernel->set_regs(2, (uint16_t)(ddr_output_addr & 0xFFFF));
            kernel->set_regs(3, (uint16_t)(ddr_output_addr >> 16));
            kernel->set_regs(4, (uint16_t)1);  // data_type: 1 for fp16

            kernel->set_regs(6, (uint16_t)c);
            kernel->set_regs(7, (uint16_t)c_tail);

            kernel->set_regs(8, (uint16_t)b);
            kernel->set_regs(9, (uint16_t)b_v16);
            kernel->set_regs(10, (uint16_t)b_tail);
            kernel->set_regs(11, (uint16_t)b_v16_tail);

            kernel->set_regs(12, (uint16_t)c_per_thd);
            kernel->set_regs(13, (uint16_t)c_per_wrp);
            kernel->set_regs(14, (uint16_t)c_v16_per_wrp);
            kernel->set_regs(15, (uint16_t)c_v256_per_wrp);

            kernel->set_regs(16, (uint16_t)b_per_thd);
            kernel->set_regs(17, (uint16_t)b_per_wrp);
            kernel->set_regs(18, (uint16_t)b_v16_per_wrp);

            kernel->set_regs(19, (uint16_t)n);

            if (log_at(4)) {
                std::cout << "[permute " << order << "] tailb config registers:\n";
                std::cout << "  reg[0-1] ddr_input_addr: lo=" << (ddr_input_addr & 0xFFFF) << ", hi=" << (ddr_input_addr >> 16) << "\n";
                std::cout << "  reg[2-3] ddr_output_addr: lo=" << (ddr_output_addr & 0xFFFF) << ", hi=" << (ddr_output_addr >> 16) << "\n";
                std::cout << "  reg[4] data_type: 1 (fp16)\n";
                std::cout << "  reg[6] c: " << c << "\n";
                std::cout << "  reg[7] c_tail: " << c_tail << "\n";
                std::cout << "  reg[8] b: " << b << "\n";
                std::cout << "  reg[9] b_v16: " << b_v16 << "\n";
                std::cout << "  reg[10] b_tail: " << b_tail << "\n";
                std::cout << "  reg[11] b_v16_tail: " << b_v16_tail << "\n";
                std::cout << "  reg[12] c_per_thd: " << c_per_thd << "\n";
                std::cout << "  reg[13] c_per_wrp: " << c_per_wrp << "\n";
                std::cout << "  reg[14] c_v16_per_wrp: " << c_v16_per_wrp << "\n";
                std::cout << "  reg[15] c_v256_per_wrp: " << c_v256_per_wrp << "\n";
                std::cout << "  reg[16] b_per_thd: " << b_per_thd << "\n";
                std::cout << "  reg[17] b_per_wrp: " << b_per_wrp << "\n";
                std::cout << "  reg[18] b_v16_per_wrp: " << b_v16_per_wrp << "\n";
                std::cout << "  reg[19] n: " << n << "\n";
            }

            // ==================== SPM 版本 (注释保留) ====================
            // kernel->set_regs(0, (uint16_t)(spm_in_byte_base & 0xFFFF));
            // kernel->set_regs(1, (uint16_t)(spm_in_byte_base >> 16));
            // kernel->set_regs(2, (uint16_t)(spm_out_byte_base & 0xFFFF));
            // kernel->set_regs(3, (uint16_t)(spm_out_byte_base >> 16));
        }

        // 设置 grid dims
        kernel->set_regs(64, (uint16_t)grid_dim_x);
        kernel->set_regs(65, (uint16_t)grid_dim_y);
        kernel->set_regs(66, (uint16_t)grid_dim_z);

        if (log_at(4)) {
            std::cout << "[permute] grid_dims: (" << grid_dim_x << ", " << grid_dim_y << ", " << grid_dim_z << ")\n";
            std::cout << "  reg[64] grid_dim_x: " << grid_dim_x << "\n";
            std::cout << "  reg[65] grid_dim_y: " << grid_dim_y << "\n";
            std::cout << "  reg[66] grid_dim_z: " << grid_dim_z << "\n";
            std::cout << "========== END permute 3D DEBUG ==========\n" << std::endl;
        }

        // wq->set_warp_num(4);  // (dead eager path; method removed from RpuQueue API)
        wq->enqueu_kernel(*kernel, {(uint16_t)grid_dim_x, (uint16_t)grid_dim_y, (uint16_t)grid_dim_z}, {0});

        // Flush output, 确保 CPU 能读到 RPU 写入的最新数据
        // rpu_ddr_flush(out_ptr);

        // ==================== SPM 版本 (注释保留) ====================
        // spm_to_ddr(out_ptr, spm_core0_output.get_cpu_ptr(), aligned_input_nbytes);

        return;

    } else if (rank == 4) {
        // 4D transpose: input shape (Q, B, N, C)
        int64_t q = base.size(0);
        int64_t b = base.size(1);
        int64_t n = base.size(2);
        int64_t c = base.size(3);

        // 将 dims 转换为 order 字符串
        std::string order = std::to_string(dims[0]) + std::to_string(dims[1]) +
                           std::to_string(dims[2]) + std::to_string(dims[3]);

        // ==================== DDR 版本 ====================
        // 选择 DDR kernel 名称
        std::string kernel_name;
        if (order == "0213") {
            kernel_name = "transpose_qnbc_c16_ddr";
        } else if (order == "2103") {
            kernel_name = "transpose_nbqc_c16_ddr";
        } else if (order == "0321") {
            kernel_name = "transpose_qcnb_c16_ddr";
        } else {
            TORCH_CHECK(false, "rpu_launch_permute_kernel: unsupported 4D permute order ", order);
        }

        // ==================== SPM 版本 (注释保留) ====================
        // std::string kernel_name;
        // if (order == "0213") {
        //     kernel_name = "transpose_qnbc_c16";
        // } else if (order == "2103") {
        //     kernel_name = "transpose_nbqc_c16";
        // } else if (order == "0321") {
        //     kernel_name = "transpose_qcnb_c16";
        // } else {
        //     TORCH_CHECK(false, "rpu_launch_permute_kernel: unsupported 4D permute order ", order);
        // }

        Kernel_t* kernel = KernelCache::instance().get_kernel(kernel_name);
        TORCH_CHECK(kernel != nullptr, "Failed to get ", kernel_name, " kernel from cache");
        kernel->reset_regs();

        if (log_at(4)) {
            std::cout << "\n========== rpu_launch_permute_kernel 4D DEBUG ==========\n";
            std::cout << "[permute 4D] kernel_name: " << kernel_name << "\n";
            std::cout << "[permute 4D] order: " << order << "\n";
            std::cout << "[permute 4D] input shape: (q=" << q << ", b=" << b << ", n=" << n << ", c=" << c << ")\n";
            std::cout << "[permute 4D] input numel: " << base.numel() << "\n";
            std::cout << "[permute 4D] input_nbytes: " << input_nbytes << "\n";
            std::cout << "[permute 4D] output_nbytes: " << output_nbytes << "\n";
            std::cout << std::hex << std::showbase;
            std::cout << "[permute 4D] ddr_input_addr (raw): " << ddr_input_addr << "\n";
            std::cout << "[permute 4D] ddr_output_addr (raw): " << ddr_output_addr << "\n";
            std::cout << std::dec << std::noshowbase;
        }

        // 计算参数
        constexpr int64_t MAX_FP16_VLM_V16_PER_THD_LOCAL = 25;
        int64_t c_v16 = CeilDiv(c, V16_NUM);
        int64_t n_v16 = CeilDiv(n, V16_NUM);
        int64_t b_v16 = CeilDiv(b, V16_NUM);

        // input/output 不带 tail
        int64_t c_tail = c;
        int64_t b_tail = b;

        // ==================== SPM 版本 (注释保留) ====================
        // uint32_t spm_in_byte_base = 0;
        // uint32_t spm_out_byte_base = aligned_input_nbytes;

        int64_t c_v16_per_wrp = std::min((int64_t)MAX_FP16_VLM_V16_PER_THD_LOCAL, c_v16);
        int64_t c_v256_per_wrp = c_v16_per_wrp / V16_NUM;
        int64_t c_per_thd = c_v16_per_wrp * V16_NUM;
        int64_t c_per_wrp = c_v16_per_wrp * V16_NUM;

        int64_t grid_dim_x, grid_dim_y, grid_dim_z;

        if (order == "0213" || order == "2103") {
            // tailc config
            int64_t n_v16_per_wrp = std::min(n_v16, MAX_FP16_VLM_V16_PER_THD_LOCAL / c_v16_per_wrp);
            int64_t n_per_thd = n_v16_per_wrp * V16_NUM;
            int64_t n_per_wrp = n_v16_per_wrp * V16_NUM;

            grid_dim_x = CeilDiv(n_v16, n_v16_per_wrp);
            grid_dim_y = CeilDiv(c_v16, c_v16_per_wrp);
            grid_dim_z = b * q;

            // ==================== DDR 版本 ====================
            // 设置寄存器，使用 DDR 地址
            kernel->set_regs(0, (uint16_t)(ddr_input_addr & 0xFFFF));
            kernel->set_regs(1, (uint16_t)(ddr_input_addr >> 16));
            kernel->set_regs(2, (uint16_t)(ddr_output_addr & 0xFFFF));
            kernel->set_regs(3, (uint16_t)(ddr_output_addr >> 16));
            kernel->set_regs(4, (uint16_t)1);  // data_type: 1 for fp16

            kernel->set_regs(6, (uint16_t)c);
            kernel->set_regs(7, (uint16_t)c_tail);  // c_in_tail
            kernel->set_regs(8, (uint16_t)c_tail);  // c_out_tail

            kernel->set_regs(9, (uint16_t)n);
            kernel->set_regs(10, (uint16_t)n_v16);

            kernel->set_regs(12, (uint16_t)c_per_thd);
            kernel->set_regs(13, (uint16_t)c_per_wrp);
            kernel->set_regs(14, (uint16_t)c_v16_per_wrp);
            kernel->set_regs(15, (uint16_t)c_v256_per_wrp);

            kernel->set_regs(16, (uint16_t)n_per_thd);
            kernel->set_regs(17, (uint16_t)n_per_wrp);
            kernel->set_regs(18, (uint16_t)n_v16_per_wrp);

            kernel->set_regs(19, (uint16_t)b);
            kernel->set_regs(20, (uint16_t)q);

            if (log_at(4)) {
                std::cout << "[permute 4D " << order << "] tailc config registers:\n";
                std::cout << "  reg[0-1] ddr_input_addr: lo=" << (ddr_input_addr & 0xFFFF) << ", hi=" << (ddr_input_addr >> 16) << "\n";
                std::cout << "  reg[2-3] ddr_output_addr: lo=" << (ddr_output_addr & 0xFFFF) << ", hi=" << (ddr_output_addr >> 16) << "\n";
                std::cout << "  reg[4] data_type: 1 (fp16)\n";
                std::cout << "  reg[6] c: " << c << "\n";
                std::cout << "  reg[7] c_in_tail: " << c_tail << "\n";
                std::cout << "  reg[8] c_out_tail: " << c_tail << "\n";
                std::cout << "  reg[9] n: " << n << "\n";
                std::cout << "  reg[10] n_v16: " << n_v16 << "\n";
                std::cout << "  reg[12] c_per_thd: " << c_per_thd << "\n";
                std::cout << "  reg[13] c_per_wrp: " << c_per_wrp << "\n";
                std::cout << "  reg[14] c_v16_per_wrp: " << c_v16_per_wrp << "\n";
                std::cout << "  reg[15] c_v256_per_wrp: " << c_v256_per_wrp << "\n";
                std::cout << "  reg[16] n_per_thd: " << n_per_thd << "\n";
                std::cout << "  reg[17] n_per_wrp: " << n_per_wrp << "\n";
                std::cout << "  reg[18] n_v16_per_wrp: " << n_v16_per_wrp << "\n";
                std::cout << "  reg[19] b: " << b << "\n";
                std::cout << "  reg[20] q: " << q << "\n";
            }

            // ==================== SPM 版本 (注释保留) ====================
            // kernel->set_regs(0, (uint16_t)(spm_in_byte_base & 0xFFFF));
            // kernel->set_regs(1, (uint16_t)(spm_in_byte_base >> 16));
            // kernel->set_regs(2, (uint16_t)(spm_out_byte_base & 0xFFFF));
            // kernel->set_regs(3, (uint16_t)(spm_out_byte_base >> 16));

        } else {
            // tailb config (order "0321")
            int64_t b_v16_per_wrp = std::min(b_v16, MAX_FP16_VLM_V16_PER_THD_LOCAL / c_v16_per_wrp);
            int64_t b_v16_tail = b_v16 + 1;
            int64_t b_per_thd = b_v16_per_wrp * V16_NUM;
            int64_t b_per_wrp = b_v16_per_wrp * V16_NUM;

            grid_dim_x = CeilDiv(b_v16, b_v16_per_wrp);
            grid_dim_y = CeilDiv(c_v16, c_v16_per_wrp);
            grid_dim_z = n * q;

            // ==================== DDR 版本 ====================
            // 设置寄存器，使用 DDR 地址
            kernel->set_regs(0, (uint16_t)(ddr_input_addr & 0xFFFF));
            kernel->set_regs(1, (uint16_t)(ddr_input_addr >> 16));
            kernel->set_regs(2, (uint16_t)(ddr_output_addr & 0xFFFF));
            kernel->set_regs(3, (uint16_t)(ddr_output_addr >> 16));
            kernel->set_regs(4, (uint16_t)1);  // data_type: 1 for fp16

            kernel->set_regs(6, (uint16_t)c);
            kernel->set_regs(7, (uint16_t)c_tail);

            kernel->set_regs(8, (uint16_t)b);
            kernel->set_regs(9, (uint16_t)b_v16);
            kernel->set_regs(10, (uint16_t)b_tail);
            kernel->set_regs(11, (uint16_t)b_v16_tail);

            kernel->set_regs(12, (uint16_t)c_per_thd);
            kernel->set_regs(13, (uint16_t)c_per_wrp);
            kernel->set_regs(14, (uint16_t)c_v16_per_wrp);
            kernel->set_regs(15, (uint16_t)c_v256_per_wrp);

            kernel->set_regs(16, (uint16_t)b_per_thd);
            kernel->set_regs(17, (uint16_t)b_per_wrp);
            kernel->set_regs(18, (uint16_t)b_v16_per_wrp);

            kernel->set_regs(19, (uint16_t)n);
            kernel->set_regs(20, (uint16_t)q);

            if (log_at(4)) {
                std::cout << "[permute 4D 0321] tailb config registers:\n";
                std::cout << "  reg[0-1] ddr_input_addr: lo=" << (ddr_input_addr & 0xFFFF) << ", hi=" << (ddr_input_addr >> 16) << "\n";
                std::cout << "  reg[2-3] ddr_output_addr: lo=" << (ddr_output_addr & 0xFFFF) << ", hi=" << (ddr_output_addr >> 16) << "\n";
                std::cout << "  reg[4] data_type: 1 (fp16)\n";
                std::cout << "  reg[6] c: " << c << "\n";
                std::cout << "  reg[7] c_tail: " << c_tail << "\n";
                std::cout << "  reg[8] b: " << b << "\n";
                std::cout << "  reg[9] b_v16: " << b_v16 << "\n";
                std::cout << "  reg[10] b_tail: " << b_tail << "\n";
                std::cout << "  reg[11] b_v16_tail: " << b_v16_tail << "\n";
                std::cout << "  reg[12] c_per_thd: " << c_per_thd << "\n";
                std::cout << "  reg[13] c_per_wrp: " << c_per_wrp << "\n";
                std::cout << "  reg[14] c_v16_per_wrp: " << c_v16_per_wrp << "\n";
                std::cout << "  reg[15] c_v256_per_wrp: " << c_v256_per_wrp << "\n";
                std::cout << "  reg[16] b_per_thd: " << b_per_thd << "\n";
                std::cout << "  reg[17] b_per_wrp: " << b_per_wrp << "\n";
                std::cout << "  reg[18] b_v16_per_wrp: " << b_v16_per_wrp << "\n";
                std::cout << "  reg[19] n: " << n << "\n";
                std::cout << "  reg[20] q: " << q << "\n";
            }

            // ==================== SPM 版本 (注释保留) ====================
            // kernel->set_regs(0, (uint16_t)(spm_in_byte_base & 0xFFFF));
            // kernel->set_regs(1, (uint16_t)(spm_in_byte_base >> 16));
            // kernel->set_regs(2, (uint16_t)(spm_out_byte_base & 0xFFFF));
            // kernel->set_regs(3, (uint16_t)(spm_out_byte_base >> 16));
        }

        // 设置 grid dims
        kernel->set_regs(64, (uint16_t)grid_dim_x);
        kernel->set_regs(65, (uint16_t)grid_dim_y);
        kernel->set_regs(66, (uint16_t)grid_dim_z);

        if (log_at(4)) {
            std::cout << "[permute 4D] grid_dims: (" << grid_dim_x << ", " << grid_dim_y << ", " << grid_dim_z << ")\n";
            std::cout << "  reg[64] grid_dim_x: " << grid_dim_x << "\n";
            std::cout << "  reg[65] grid_dim_y: " << grid_dim_y << "\n";
            std::cout << "  reg[66] grid_dim_z: " << grid_dim_z << "\n";
            std::cout << "[permute 4D] computed params:\n";
            std::cout << "  c_v16=" << c_v16 << ", n_v16=" << n_v16 << ", b_v16=" << b_v16 << "\n";
            std::cout << "  c_v16_per_wrp=" << c_v16_per_wrp << ", c_v256_per_wrp=" << c_v256_per_wrp << "\n";
            std::cout << "  c_per_thd=" << c_per_thd << ", c_per_wrp=" << c_per_wrp << "\n";
            std::cout << "========== END permute 4D DEBUG ==========\n" << std::endl;
        }

        // wq->set_warp_num(4);  // (dead eager path; method removed from RpuQueue API)
        wq->enqueu_kernel(*kernel, {(uint16_t)grid_dim_x, (uint16_t)grid_dim_y, (uint16_t)grid_dim_z}, {0});

        // Flush output, 确保 CPU 能读到 RPU 写入的最新数据
        // rpu_ddr_flush(out_ptr);

        // ==================== SPM 版本 (注释保留) ====================
        // spm_to_ddr(out_ptr, spm_core0_output.get_cpu_ptr(), aligned_input_nbytes);

        return;

    } else {
        TORCH_CHECK(false, "rpu_launch_permute_kernel: only 3D and 4D tensors are supported, got ", rank, "D");
    }
}

// ====== Permute3d SPM Kernel Launch (multi-core, fused-graph) ======
// SPM-resident 3D permute: out = in.permute(perm), perm a permutation of {0,1,2};
// out shape = [in.size(perm[0]), in.size(perm[1]), in.size(perm[2])]. Uses the SAME
// transpose_{nbc,cbn,bcn,cnb,ncb}_c16 kernels as the eager rpu_launch_permute_kernel
// above but WITHOUT the _ddr suffix (SPM-resident), with SPM byte addresses (no >>8),
// graph-aware kernel fetch, and multi-core broadcast (each core permutes its OWN
// per-core SPM). Use MAX_FP16_VLM_V16_PER_THD for SPM launch geometry.
// Assumes contiguous in/out (in_tail=out_tail=0). GDN chunk uses perm{1,0,2}
// ([L,Hc,d]->[Hc,L,d]); c=d=128 (16-aligned), n=Hc handled natively.
void rpu_launch_permute3d_spm_kernel(uint32_t input_spm_addr, uint32_t output_spm_addr,
                                     int64_t b, int64_t n, int64_t c,
                                     const std::vector<int64_t>& perm, int num_cores) {
    TORCH_CHECK(perm.size() == 3, "permute3d_spm: perm must have 3 elements");
    const std::string order =
        std::to_string(perm[0]) + std::to_string(perm[1]) + std::to_string(perm[2]);

    std::string kernel_name;
    if (order == "102")      kernel_name = "transpose_nbc_c16";
    else if (order == "201") kernel_name = "transpose_cbn_c16";
    else if (order == "021") kernel_name = "transpose_bcn_c16";
    else if (order == "210") kernel_name = "transpose_cnb_c16";
    else if (order == "120") kernel_name = "transpose_ncb_c16";
    else TORCH_CHECK(false, "permute3d_spm: unsupported order ", order);

    Kernel_t* kernel = RpuKernelGraph::active().get_kernel_reset(kernel_name);
    TORCH_CHECK(kernel != nullptr, "permute3d_spm: failed to get ", kernel_name, " kernel");

    const int64_t c_v16 = CeilDiv(c, V16_NUM);
    const int64_t n_v16 = CeilDiv(n, V16_NUM);
    const int64_t b_v16 = CeilDiv(b, V16_NUM);
    const int64_t c_v16_per_wrp = std::min((int64_t)MAX_FP16_VLM_V16_PER_THD, c_v16);
    const int64_t c_v256_per_wrp = c_v16_per_wrp / V16_NUM;
    const int64_t c_per_thd = c_v16_per_wrp * V16_NUM;
    const int64_t c_per_wrp = c_v16_per_wrp * V16_NUM;
    const int64_t c_tail = c, n_tail = n, b_tail = b;  // contiguous -> no tail padding

    // common addr + c regs
    kernel->set_regs(0, (uint16_t)(input_spm_addr & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(input_spm_addr >> 16));
    kernel->set_regs(2, (uint16_t)(output_spm_addr & 0xFFFF));
    kernel->set_regs(3, (uint16_t)(output_spm_addr >> 16));
    kernel->set_regs(4, (uint16_t)1);  // data_type=1 (fp16)
    kernel->set_regs(6, (uint16_t)c);
    kernel->set_regs(7, (uint16_t)c_tail);  // c_in_tail

    int64_t gx, gy, gz;
    const char last = order[2];
    if (last == '2') {
        // get_tailc_config (e.g. 102): output v16 tail on c
        const int64_t n_v16_per_wrp =
            std::min(n_v16, (int64_t)MAX_FP16_VLM_V16_PER_THD / c_v16_per_wrp);
        const int64_t n_per_thd = n_v16_per_wrp * V16_NUM;
        const int64_t n_per_wrp = n_v16_per_wrp * V16_NUM;
        gx = CeilDiv(n_v16, n_v16_per_wrp);
        gy = CeilDiv(c_v16, c_v16_per_wrp);
        gz = b;
        kernel->set_regs(8,  (uint16_t)c_tail);  // c_out_tail
        kernel->set_regs(9,  (uint16_t)n);
        kernel->set_regs(10, (uint16_t)n_v16);
        kernel->set_regs(12, (uint16_t)c_per_thd);
        kernel->set_regs(13, (uint16_t)c_per_wrp);
        kernel->set_regs(14, (uint16_t)c_v16_per_wrp);
        kernel->set_regs(15, (uint16_t)c_v256_per_wrp);
        kernel->set_regs(16, (uint16_t)n_per_thd);
        kernel->set_regs(17, (uint16_t)n_per_wrp);
        kernel->set_regs(18, (uint16_t)n_v16_per_wrp);
        kernel->set_regs(19, (uint16_t)b);
    } else if (last == '1') {
        // get_tailn_config (e.g. 201 / 021): output v16 tail on n
        const int64_t n_v16_per_wrp =
            std::min(n_v16, (int64_t)MAX_FP16_VLM_V16_PER_THD / c_v16_per_wrp);
        const int64_t n_v16_tail = n_v16 + 1;
        const int64_t n_per_thd = n_v16_per_wrp * V16_NUM;
        const int64_t n_per_wrp = n_v16_per_wrp * V16_NUM;
        gx = CeilDiv(n_v16, n_v16_per_wrp);
        gy = CeilDiv(c_v16, c_v16_per_wrp);
        gz = b;
        kernel->set_regs(8,  (uint16_t)n);
        kernel->set_regs(9,  (uint16_t)n_v16);
        kernel->set_regs(10, (uint16_t)n_tail);
        kernel->set_regs(11, (uint16_t)n_v16_tail);
        kernel->set_regs(12, (uint16_t)c_per_thd);
        kernel->set_regs(13, (uint16_t)c_per_wrp);
        kernel->set_regs(14, (uint16_t)c_v16_per_wrp);
        kernel->set_regs(15, (uint16_t)c_v256_per_wrp);
        kernel->set_regs(16, (uint16_t)n_per_thd);
        kernel->set_regs(17, (uint16_t)n_per_wrp);
        kernel->set_regs(18, (uint16_t)n_v16_per_wrp);
        kernel->set_regs(19, (uint16_t)b);
    } else {
        // get_tailb_config (e.g. 120 / 210): output v16 tail on b
        const int64_t b_v16_per_wrp =
            std::min(b_v16, (int64_t)MAX_FP16_VLM_V16_PER_THD / c_v16_per_wrp);
        const int64_t b_v16_tail = b_v16 + 1;
        const int64_t b_per_thd = b_v16_per_wrp * V16_NUM;
        const int64_t b_per_wrp = b_v16_per_wrp * V16_NUM;
        gx = CeilDiv(b_v16, b_v16_per_wrp);
        gy = CeilDiv(c_v16, c_v16_per_wrp);
        gz = n;
        kernel->set_regs(8,  (uint16_t)b);
        kernel->set_regs(9,  (uint16_t)b_v16);
        kernel->set_regs(10, (uint16_t)b_tail);
        kernel->set_regs(11, (uint16_t)b_v16_tail);
        kernel->set_regs(12, (uint16_t)c_per_thd);
        kernel->set_regs(13, (uint16_t)c_per_wrp);
        kernel->set_regs(14, (uint16_t)c_v16_per_wrp);
        kernel->set_regs(15, (uint16_t)c_v256_per_wrp);
        kernel->set_regs(16, (uint16_t)b_per_thd);
        kernel->set_regs(17, (uint16_t)b_per_wrp);
        kernel->set_regs(18, (uint16_t)b_v16_per_wrp);
        kernel->set_regs(19, (uint16_t)n);
    }
    kernel->set_regs(64, (uint16_t)gx);
    kernel->set_regs(65, (uint16_t)gy);
    kernel->set_regs(66, (uint16_t)gz);

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {(uint16_t)gx, (uint16_t)gy, (uint16_t)gz}, cores);
}

// SPM-resident 4D 0213 permute: [q,b,n,c] -> [q,n,b,c]. This is the SPM
// sibling of the eager DDR path's transpose_qnbc_c16_ddr branch above. Keep the
// production surface deliberately narrow: G0.5 patch layout only needs 0213.
void rpu_launch_permute4d_0213_spm_kernel(
    uint32_t input_spm_addr, uint32_t output_spm_addr,
    int64_t q, int64_t b, int64_t n, int64_t c, int num_cores) {
    TORCH_CHECK(q > 0 && b > 0 && n > 0 && c > 0,
                "permute4d_0213_spm: dimensions must be positive");
    TORCH_CHECK(c % V16_NUM == 0,
                "permute4d_0213_spm: c must be 16-aligned, got ", c);
    constexpr int64_t kU16Max = 65535;
    TORCH_CHECK(q <= kU16Max && b <= kU16Max && n <= kU16Max &&
                    c <= kU16Max && b * q <= kU16Max,
                "permute4d_0213_spm: dimensions exceed u16 register/grid range: q=",
                q, " b=", b, " n=", n, " c=", c);
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "permute4d_0213_spm: num_cores must be in [1,8], got ", num_cores);

    Kernel_t* kernel =
        RpuKernelGraph::active().get_kernel_reset("transpose_qnbc_c16");
    TORCH_CHECK(kernel != nullptr,
                "permute4d_0213_spm: failed to get transpose_qnbc_c16 kernel");

    const int64_t c_v16 = CeilDiv(c, V16_NUM);
    const int64_t n_v16 = CeilDiv(n, V16_NUM);
    const int64_t c_v16_per_wrp =
        std::min((int64_t)MAX_FP16_VLM_V16_PER_THD, c_v16);
    const int64_t n_v16_per_wrp =
        std::min(n_v16,
                 (int64_t)MAX_FP16_VLM_V16_PER_THD / c_v16_per_wrp);
    const int64_t c_v256_per_wrp = c_v16_per_wrp / V16_NUM;
    const int64_t c_per_thd = c_v16_per_wrp * V16_NUM;
    const int64_t c_per_wrp = c_per_thd;
    const int64_t n_per_thd = n_v16_per_wrp * V16_NUM;
    const int64_t n_per_wrp = n_per_thd;
    const int64_t gx = CeilDiv(n_v16, n_v16_per_wrp);
    const int64_t gy = CeilDiv(c_v16, c_v16_per_wrp);
    const int64_t gz = b * q;

    kernel->set_regs(0, (uint16_t)(input_spm_addr & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(input_spm_addr >> 16));
    kernel->set_regs(2, (uint16_t)(output_spm_addr & 0xFFFF));
    kernel->set_regs(3, (uint16_t)(output_spm_addr >> 16));
    kernel->set_regs(4, (uint16_t)1);  // data_type=1 (fp16)
    kernel->set_regs(6, (uint16_t)c);
    kernel->set_regs(7, (uint16_t)c);  // c_in_tail: contiguous, unpadded
    kernel->set_regs(8, (uint16_t)c);  // c_out_tail
    kernel->set_regs(9, (uint16_t)n);
    kernel->set_regs(10, (uint16_t)n_v16);
    kernel->set_regs(12, (uint16_t)c_per_thd);
    kernel->set_regs(13, (uint16_t)c_per_wrp);
    kernel->set_regs(14, (uint16_t)c_v16_per_wrp);
    kernel->set_regs(15, (uint16_t)c_v256_per_wrp);
    kernel->set_regs(16, (uint16_t)n_per_thd);
    kernel->set_regs(17, (uint16_t)n_per_wrp);
    kernel->set_regs(18, (uint16_t)n_v16_per_wrp);
    kernel->set_regs(19, (uint16_t)b);
    kernel->set_regs(20, (uint16_t)q);
    kernel->set_regs(64, (uint16_t)gx);
    kernel->set_regs(65, (uint16_t)gy);
    kernel->set_regs(66, (uint16_t)gz);

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel,
                      {(uint16_t)gx, (uint16_t)gy, (uint16_t)gz}, cores);
}

// Isolated test: stage `input` [b,n,c] into SPM, permute by `perm`, DMA the result
// back. Golden = input.permute(perm). Single-core. Pure copy -> bit-exact.
at::Tensor rpu_permute3d_spm_test(const at::Tensor& input, c10::IntArrayRef perm) {
    TORCH_CHECK(input.dim() == 3, "permute3d_spm_test: 3D input only");
    TORCH_CHECK(input.scalar_type() == at::kHalf, "permute3d_spm_test: fp16 only");
    TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                "permute3d_spm_test: input must be on RPU");
    TORCH_CHECK(perm.size() == 3, "permute3d_spm_test: perm must have 3 elements");
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    auto in_c = input.contiguous();
    const int64_t b = in_c.size(0), n = in_c.size(1), c = in_c.size(2);
    const std::vector<int64_t> p = {perm[0], perm[1], perm[2]};
    const int64_t dims[3] = {b, n, c};
    const int64_t ob = dims[p[0]], on = dims[p[1]], oc = dims[p[2]];

    const int64_t bytes = in_c.numel() * 2;  // permute preserves numel
    using AR = SpmAllocator::AllocRequest;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{bytes, 1, 2},   // input
        AR{bytes, 1, 3},   // output
    });
    const uint32_t in_addr = SPM_ALLOC.addr(0, offsets[0]);
    const uint32_t out_addr = SPM_ALLOC.addr(0, offsets[1]);

    rpu_launch_ddr_broadcast_spm_dma_immediate(
        const_cast<c10::Half*>(in_c.data_ptr<c10::Half>()), in_c.numel(), in_addr, 1);
    rpu_launch_permute3d_spm_kernel(in_addr, out_addr, b, n, c, p, 1);

    auto output = at::empty({ob, on, oc}, in_c.options());
    rpu_launch_spm_copy_ddr_dma_immediate(out_addr, output.data_ptr<c10::Half>(),
                                          in_c.numel());
    SPM_ALLOC.reset_temporary();
    return output;
}

// Isolated test for the production 0213-only 4D SPM launcher. Pure movement,
// so the hardware result must be bit-exact with input.permute(0,2,1,3).
at::Tensor rpu_permute4d_0213_spm_test(const at::Tensor& input) {
    TORCH_CHECK(input.dim() == 4, "permute4d_0213_spm_test: 4D input only");
    TORCH_CHECK(input.scalar_type() == at::kHalf,
                "permute4d_0213_spm_test: fp16 only");
    TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                "permute4d_0213_spm_test: input must be on RPU");
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    auto in_c = input.contiguous();
    const int64_t q = in_c.size(0), b = in_c.size(1);
    const int64_t n = in_c.size(2), c = in_c.size(3);
    const int64_t bytes = in_c.numel() * sizeof(c10::Half);
    using AR = SpmAllocator::AllocRequest;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{bytes, 1, 2},
        AR{bytes, 1, 3},
    });
    const uint32_t in_addr = SPM_ALLOC.addr(0, offsets[0]);
    const uint32_t out_addr = SPM_ALLOC.addr(0, offsets[1]);

    rpu_launch_ddr_broadcast_spm_dma_immediate(
        const_cast<c10::Half*>(in_c.data_ptr<c10::Half>()),
        in_c.numel(), in_addr, 1);
    rpu_launch_permute4d_0213_spm_kernel(
        in_addr, out_addr, q, b, n, c, 1);

    auto output = at::empty({q, n, b, c}, in_c.options());
    rpu_launch_spm_copy_ddr_dma_immediate(
        out_addr, output.data_ptr<c10::Half>(), in_c.numel());
    SPM_ALLOC.reset_temporary();
    return output;
}
