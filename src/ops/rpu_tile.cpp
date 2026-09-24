// rpu_tile.cpp - Tile/repeat kernel implementation for RPU
#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ::rhino_lkn;

// ======================== Tile Kernel Launch ========================
// Implements expand/repeat optimization using tile kernels
// Supports:
//   2D: input (N, 1) -> output (N, C) or (1, N) -> output (C, N)
//   3D: input (B, N, C) -> output (B*r0, N*r1, C*r2)
void rpu_launch_tile_kernel(const at::Tensor &input,
                            at::Tensor &output,
                            const std::vector<int64_t> &repeats) {

    TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1,
                "input not on RPU");

    // Ensure output has same dtype as input
    if (output.scalar_type() != input.scalar_type())
    {
      output = output.to(input.scalar_type());
    }

    using namespace ::rhino_lkn;

    constexpr size_t WARP_SIZE = 16;
    constexpr size_t VLM_ENTRIES = 384;
    constexpr size_t THD_VECTOR_SIZE = 16;

    size_t dwidth = sizeof(c10::Half);
    int64_t rank = input.dim();

    // Get base tensor (the original data before expand)
    at::Tensor base;
    if (input._base().defined()) {
        base = input._base();
    } else {
        base = input;
    }

    // 计算base和output的bytesize
    size_t base_nbytes = base.numel() * dwidth;
    size_t output_nbytes = output.numel() * dwidth;

    // 打印基本信息
    if (log_at(4)) {
        std::cout << "[rpu_launch_tile_kernel] base_nbytes = " << base_nbytes
                  << ", output_nbytes = " << output_nbytes << std::endl;
        // print_tensor_info(base, "rpu_launch_tile_kernel - base");
        // print_tensor_info(output, "rpu_launch_tile_kernel - output");
        std::cout << "[rpu_launch_tile_kernel] repeats: ";
        for (auto r : repeats) std::cout << r << " ";
        std::cout << ", rank = " << rank << std::endl;
    }

    // 确保输入是 half 类型
    TORCH_CHECK(base.scalar_type() == at::kHalf,
                "rpu_launch_tile_kernel: only FP16 is supported");

    // 直接使用 tensor 的 DDR 地址，无需中间 buffer
    c10::Half *in_ptr = base.data_ptr<c10::Half>();
    c10::Half *out_ptr = output.data_ptr<c10::Half>();

    rpu_ddr_flush(in_ptr);

    // glarge format = address / 256
    uint64_t ddr_input_addr = RpuGetDevAddr(in_ptr) >> 8;
    uint64_t ddr_output_addr = RpuGetDevAddr(out_ptr) >> 8;

    if (log_at(4)) {
        std::cout << std::hex << std::showbase;
        std::cout << "[tile] DDR in_ptr           = "
                  << reinterpret_cast<uintptr_t>(in_ptr) << "\n";
        std::cout << "[tile] DDR out_ptr          = "
                  << reinterpret_cast<uintptr_t>(out_ptr) << "\n";
        std::cout << "[tile] ddr_input addr       = "
                  << (ddr_input_addr << 8) << "\n";
        std::cout << "[tile] ddr_output addr      = "
                  << (ddr_output_addr << 8) << "\n";
        std::cout << std::dec << std::noshowbase;
    }


    // ==================== SPM 版本 (注释保留) ====================
    // // std::cout << "rpu_launch_tile_kernel: in ptr " << in_ptr  << std::endl;
    // RpuDdrFlush(in_ptr);
    //
    // auto t_now = std::chrono::high_resolution_clock::now();
    // double this_preprocess_time = std::chrono::duration<double, std::milli>(t_now - t_prev).count();
    // t_prev = t_now;
    // std::cout << "rpu_launch_tile_kernel: flush input ddr time " << this_preprocess_time << " ms" << std::endl;
    //
    // // 申请spm空间
    // // LocalSPM_t(size_t size, MemFlag flag, MemStride stride, uint64_t alignment,
    // //          int core_id);
    // LocalSPM_t spm_core0_input(base_nbytes, read_write, kStride32B, 2, 0);
    // LocalSPM_t spm_core0_output(output_nbytes, read_write, kStride32B, 2, 0);
    //
    // t_now = std::chrono::high_resolution_clock::now();
    // double this_alloc_time = std::chrono::duration<double, std::milli>(t_now - t_prev).count();
    // t_prev = t_now;
    // std::cout << "rpu_launch_tile_kernel: spm alloc time " << this_alloc_time << " ms" << std::endl;
    //
    // // std::cout << "move input form ddr to spm " << std::endl;
    //
    // // CopyToDevice(spm_core0_input, in_ptr, base_nbytes); // 会卡死
    // // 打印inptr地址以及输入的spm地址
    // std::cout << "rpu_launch_tile_kernel: in_ptr addr " << static_cast<void*>(in_ptr)
    //           << ", spm_core0_input addr " << spm_core0_input.get_cpu_ptr() << std::endl;
    // // base_nbytes向上对齐到32字节
    // size_t aligned_base_nbytes = CeilDiv(base_nbytes, 32) * 32;
    // std::memcpy(spm_core0_input.get_cpu_ptr(), in_ptr, aligned_base_nbytes);
    // // std::memcpy(spm_core0_input.get_cpu_ptr(), in_ptr, base_nbytes);
    //
    // t_now = std::chrono::high_resolution_clock::now();
    // double this_copyin_time = std::chrono::duration<double, std::milli>(t_now - t_prev).count();
    // t_prev = t_now;
    // std::cout << "rpu_launch_tile_kernel: move input to spm time " << this_copyin_time << " ms" << std::endl;
    // // std::cout << "rpu_launch_tile_kernel: move input to spm finish " << std::endl;

    auto* wq = GET_QUEUE(1);  // 复用缓存的 queue

    // Context for export case
    // Context_t ctx;
    // ctx.recode_queue(&wq);
    // ctx.record_buffer(&ddr_input);
    // ctx.record_buffer(&ddr_output);
    // ctx.dump_init_buf();

    if (rank == 2) {
      // 2D case: tile_Nx1_NxC
      int64_t N = base.size(0);
      int64_t C = output.size(1);  // C = repeat[-1]
      int64_t Broadcast_N = N;
      int64_t Broadcast_C = C;

      // ==================== DDR 版本 ====================
      // 选择 kernel: v16 版本当 C % 16 == 0，使用 DDR 版本
      bool use_v16 = (C % 16 == 0);
      std::string kernel_name = use_v16 ? "tile_Nx1_NxC_v16_ddr" : "tile_Nx1_NxC_ddr";

      Kernel_t* kernel = KernelCache::instance().get_kernel(kernel_name);
      TORCH_CHECK(kernel != nullptr, "Failed to get ", kernel_name, " kernel from cache");
      kernel->reset_regs();
      // std::cout << "rpu_launch_tile_kernel: Using kernel " << kernel_name << std::endl;

      // 计算 block 参数
      size_t chunk_N_per_loop = WARP_SIZE;
      size_t max_loopcnt = VLM_ENTRIES;
      size_t blk_cnt = CeilDiv(Broadcast_N, chunk_N_per_loop * max_loopcnt);
      size_t normal_blk_loopcnt = CeilDiv(Broadcast_N, blk_cnt * chunk_N_per_loop);
      // assert(normal_blk_loopcnt <= 0xFFFF);

      // 计算 last block 参数
      int64_t remain_data = Broadcast_N - normal_blk_loopcnt * chunk_N_per_loop * (blk_cnt - 1);
      int64_t last_blk_loopcnt = remain_data / chunk_N_per_loop;
      int64_t rmd_threads = remain_data % chunk_N_per_loop;
      if (remain_data < 0) {
        // 输出错误
        TORCH_CHECK(false, "rpu_launch_tile_kernel: remain_data < 0, something wrong");
      }

      // 设置 per-thread loop counts (SCM args)
      uint16_t thread_params[16] = {0};
      for (int i = 0; i < 16; i++) {
          thread_params[i] = (uint16_t)last_blk_loopcnt;
          if (i < rmd_threads) {
              thread_params[i] += 1;
          }
      }

      // 设置寄存器
      kernel->set_regs(0, (uint16_t)normal_blk_loopcnt); // 空了一个寄存器

      // DDR 版本: param2/3 = 输入 DDR 基地址, param4/5 = 输出 DDR 基地址
      kernel->set_regs(2, (uint16_t)(ddr_input_addr & 0xFFFF));
      kernel->set_regs(3, (uint16_t)(ddr_input_addr >> 16));
      kernel->set_regs(4, (uint16_t)(ddr_output_addr & 0xFFFF));
      kernel->set_regs(5, (uint16_t)(ddr_output_addr >> 16));

      uint32_t block_data_stride = normal_blk_loopcnt * chunk_N_per_loop * dwidth;
      kernel->set_regs(6, (uint16_t)(block_data_stride & 0xFFFF));
      kernel->set_regs(7, (uint16_t)(block_data_stride >> 16));

      // C_loopcnt
      if (use_v16) {
          kernel->set_regs(8, (uint16_t)(Broadcast_C / 16));
      } else {
          kernel->set_regs(8, (uint16_t)Broadcast_C);
      }

      // SPM 版本 (注释保留)
      // kernel->set_regs(61, (uint16_t)0x1); // lsu mode, local memory and 32bytes vector
      kernel->set_regs(64, (uint16_t)blk_cnt); // gridDim.x
      kernel->set_regs(65, (uint16_t)0x1); // gridDim.y
      kernel->set_regs(66, (uint16_t)0x1); // gridDim.z

      // 设置 SCM args (per-thread loop counts)
      for (int i = 0; i < WARP_SIZE; i++) {
          rpu_set_legacy_scm_u16_checked(
              kernel, (uint32_t)i + 4096, thread_params[i], "tile");
      }

      // 打印 2D case 参数配置
      if (log_at(4)) {
          std::cout << "[rpu_launch_tile_kernel] 2D case kernel config:" << std::endl;
          std::cout << "  kernel_name = " << kernel_name << std::endl;
          std::cout << "  N = " << N << ", C = " << C << std::endl;
          std::cout << "  Broadcast_N = " << Broadcast_N << ", Broadcast_C = " << Broadcast_C << std::endl;
          std::cout << "  blk_cnt = " << blk_cnt << ", normal_blk_loopcnt = " << normal_blk_loopcnt << std::endl;
          std::cout << "  last_blk_loopcnt = " << last_blk_loopcnt << ", rmd_threads = " << rmd_threads << std::endl;
          std::cout << "  block_data_stride = " << block_data_stride << std::endl;
          std::cout << "  ddr_input_addr = 0x" << std::hex << (ddr_input_addr << 8)
                    << ", ddr_output_addr = 0x" << (ddr_output_addr << 8) << std::dec << std::endl;
          std::cout << "  Registers:" << std::endl;
          std::cout << "    reg[0] (normal_blk_loopcnt) = " << normal_blk_loopcnt << std::endl;
          std::cout << "    reg[2-3] (ddr_input_addr>>8) = 0x" << std::hex << ddr_input_addr << std::dec << std::endl;
          std::cout << "    reg[4-5] (ddr_output_addr>>8) = 0x" << std::hex << ddr_output_addr << std::dec << std::endl;
          std::cout << "    reg[6-7] (block_data_stride) = " << block_data_stride << std::endl;
          std::cout << "    reg[8] (C_loopcnt) = " << (use_v16 ? Broadcast_C / 16 : Broadcast_C) << std::endl;
          std::cout << "    reg[64-66] (gridDim) = (" << blk_cnt << ", 1, 1)" << std::endl;
          std::cout << "  thread_params (SCM): ";
          for (int i = 0; i < 16; i++) std::cout << thread_params[i] << " ";
          std::cout << std::endl;
      }

      wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1}, {0});

      // Flush output, 确保 CPU 能读到 RPU 写入的最新数据
      rpu_ddr_flush(out_ptr);

      // ==================== SPM 版本 (注释保留) ====================
      // c10::Half *out = output.data_ptr<c10::Half>();
      // spm_to_ddr(out, spm_core0_output.get_cpu_ptr(), output_nbytes);

      // ctx.export_case();
      return;
    }

    else if (rank == 3) {
        // 3D case: tile_general_largeC / tile_general_smallC
        int64_t dim0 = base.size(0);  // B
        int64_t dim1 = base.size(1);  // N
        int64_t dim2 = base.size(2);  // C
        int64_t repeat0 = repeats[0];
        int64_t repeat1 = repeats[1];
        int64_t repeat2 = repeats[2];

        // ==================== DDR 版本 ====================
        bool is_largeC = (dim2 >= 256);
        std::string kernel_name = is_largeC ? "tile_general_largeC_ddr" : "tile_general_smallC_ddr";

        Kernel_t* kernel = KernelCache::instance().get_kernel(kernel_name);
        TORCH_CHECK(kernel != nullptr, "Failed to get ", kernel_name, " kernel from cache");
        kernel->reset_regs();

        // SPM 版本 (注释保留)
        // uint32_t a_base = 0;
        // uint32_t r_base = a_base + CeilDiv(base_nbytes, 32) * 32;

        int64_t blkcnt_x, blkcnt_y;
        int64_t Input_Stride_dim0, Input_Stride_dim1;
        int64_t Output_Stride_dim0, Output_Stride_dim1;
        int64_t Output_lpstep_dim0, Output_lpstep_dim1, Output_lpstep_dim2;
        int64_t Input_loop_delta_dim0 = 0, Output_loop_delta_dim0 = 0;
        int64_t dim0_loop_tile = 16;

        if (is_largeC) {
            blkcnt_x = dim0;
            blkcnt_y = dim1;

            Input_Stride_dim1 = dim2 * dwidth;
            Input_Stride_dim0 = dim1 * Input_Stride_dim1;

            Output_Stride_dim1 = dim2 * repeat2 * dwidth;
            Output_Stride_dim0 = dim1 * repeat1 * Output_Stride_dim1;

            Output_lpstep_dim2 = dim2 * dwidth;
            Output_lpstep_dim1 = Output_Stride_dim1 * dim1;
            Output_lpstep_dim0 = Output_Stride_dim0 * dim0;
        } else {
            blkcnt_y = CeilDiv(dim1, WARP_SIZE);
            blkcnt_x = CeilDiv(dim0, dim0_loop_tile);

            Input_Stride_dim1 = WARP_SIZE * dim2 * dwidth;
            Input_Stride_dim0 = dim0_loop_tile * dim1 * dim2 * dwidth;

            Output_Stride_dim1 = WARP_SIZE * repeat2 * dim2 * dwidth;
            Output_Stride_dim0 = dim0_loop_tile * repeat1 * dim1 * repeat2 * dim2 * dwidth;

            Output_lpstep_dim2 = dim2 * dwidth;
            Output_lpstep_dim1 = dim1 * repeat2 * Output_lpstep_dim2;
            Output_lpstep_dim0 = dim0 * repeat1 * Output_lpstep_dim1;

            Input_loop_delta_dim0 = dim1 * dim2 * dwidth;
            Output_loop_delta_dim0 = dim1 * repeat1 * dim2 * repeat2 * dwidth;
            kernel->set_regs(26, (uint16_t)(Input_loop_delta_dim0 & 0xFFFF));
            kernel->set_regs(27, (uint16_t)(Input_loop_delta_dim0 >> 16));
            kernel->set_regs(28, (uint16_t)(Output_loop_delta_dim0 & 0xFFFF));
            kernel->set_regs(29, (uint16_t)(Output_loop_delta_dim0 >> 16));
            kernel->set_regs(30, (uint16_t)dim0_loop_tile);
        }

        // DDR 版本: param0/1 = 输入 DDR 基地址, param2/3 = 输出 DDR 基地址
        kernel->set_regs(0, (uint16_t)(ddr_input_addr & 0xFFFF));
        kernel->set_regs(1, (uint16_t)(ddr_input_addr >> 16));
        kernel->set_regs(2, (uint16_t)(ddr_output_addr & 0xFFFF));
        kernel->set_regs(3, (uint16_t)(ddr_output_addr >> 16));

        // SPM 版本 (注释保留)
        // kernel->set_regs(0, (uint16_t)(a_base & 0xFFFF));
        // kernel->set_regs(1, (uint16_t)(a_base >> 16));
        // kernel->set_regs(2, (uint16_t)(r_base & 0xFFFF));
        // kernel->set_regs(3, (uint16_t)(r_base >> 16));

        kernel->set_regs(4, (uint16_t)dim0);
        kernel->set_regs(5, (uint16_t)dim1);
        kernel->set_regs(6, (uint16_t)dim2);

        kernel->set_regs(7, (uint16_t)repeat0);
        kernel->set_regs(8, (uint16_t)repeat1);
        kernel->set_regs(9, (uint16_t)repeat2);

        kernel->set_regs(10, (uint16_t)(Input_Stride_dim0 & 0xFFFF));
        kernel->set_regs(11, (uint16_t)(Input_Stride_dim0 >> 16));
        kernel->set_regs(12, (uint16_t)(Input_Stride_dim1 & 0xFFFF));
        kernel->set_regs(13, (uint16_t)(Input_Stride_dim1 >> 16));

        kernel->set_regs(14, (uint16_t)(Output_Stride_dim0 & 0xFFFF));
        kernel->set_regs(15, (uint16_t)(Output_Stride_dim0 >> 16));
        kernel->set_regs(16, (uint16_t)(Output_Stride_dim1 & 0xFFFF));
        kernel->set_regs(17, (uint16_t)(Output_Stride_dim1 >> 16));

        kernel->set_regs(20, (uint16_t)(Output_lpstep_dim0 & 0xFFFF));
        kernel->set_regs(21, (uint16_t)(Output_lpstep_dim0 >> 16));
        kernel->set_regs(22, (uint16_t)(Output_lpstep_dim1 & 0xFFFF));
        kernel->set_regs(23, (uint16_t)(Output_lpstep_dim1 >> 16));
        kernel->set_regs(24, (uint16_t)(Output_lpstep_dim2 & 0xFFFF));
        kernel->set_regs(25, (uint16_t)(Output_lpstep_dim2 >> 16));

        kernel->set_regs(64, (uint16_t)blkcnt_x); // gridDim.x
        kernel->set_regs(65, (uint16_t)blkcnt_y); // gridDim.y
        kernel->set_regs(66, (uint16_t)0x1); // gridDim.z

        // 打印 3D case 参数配置
        if (log_at(4)) {
            std::cout << "[rpu_launch_tile_kernel] 3D case kernel config:" << std::endl;
            std::cout << "  kernel_name = " << kernel_name << " (is_largeC = " << is_largeC << ")" << std::endl;
            std::cout << "  dim0 = " << dim0 << ", dim1 = " << dim1 << ", dim2 = " << dim2 << std::endl;
            std::cout << "  repeat0 = " << repeat0 << ", repeat1 = " << repeat1 << ", repeat2 = " << repeat2 << std::endl;
            std::cout << "  blkcnt_x = " << blkcnt_x << ", blkcnt_y = " << blkcnt_y << std::endl;
            std::cout << "  ddr_input_addr = 0x" << std::hex << (ddr_input_addr << 8)
                      << ", ddr_output_addr = 0x" << (ddr_output_addr << 8) << std::dec << std::endl;
            std::cout << "  Registers:" << std::endl;
            std::cout << "    reg[0-1] (ddr_input_addr>>8) = 0x" << std::hex << ddr_input_addr << std::dec << std::endl;
            std::cout << "    reg[2-3] (ddr_output_addr>>8) = 0x" << std::hex << ddr_output_addr << std::dec << std::endl;
            std::cout << "    reg[4] (dim0) = " << dim0 << std::endl;
            std::cout << "    reg[5] (dim1) = " << dim1 << std::endl;
            std::cout << "    reg[6] (dim2) = " << dim2 << std::endl;
            std::cout << "    reg[7] (repeat0) = " << repeat0 << std::endl;
            std::cout << "    reg[8] (repeat1) = " << repeat1 << std::endl;
            std::cout << "    reg[9] (repeat2) = " << repeat2 << std::endl;
            std::cout << "    reg[10-11] (Input_Stride_dim0) = " << Input_Stride_dim0 << std::endl;
            std::cout << "    reg[12-13] (Input_Stride_dim1) = " << Input_Stride_dim1 << std::endl;
            std::cout << "    reg[14-15] (Output_Stride_dim0) = " << Output_Stride_dim0 << std::endl;
            std::cout << "    reg[16-17] (Output_Stride_dim1) = " << Output_Stride_dim1 << std::endl;
            std::cout << "    reg[20-21] (Output_lpstep_dim0) = " << Output_lpstep_dim0 << std::endl;
            std::cout << "    reg[22-23] (Output_lpstep_dim1) = " << Output_lpstep_dim1 << std::endl;
            std::cout << "    reg[24-25] (Output_lpstep_dim2) = " << Output_lpstep_dim2 << std::endl;
            std::cout << "    reg[64-66] (gridDim) = (" << blkcnt_x << ", " << blkcnt_y << ", 1)" << std::endl;
            if (!is_largeC) {
                std::cout << "  smallC extra params:" << std::endl;
                std::cout << "    reg[26-27] (Input_loop_delta_dim0) = " << Input_loop_delta_dim0 << std::endl;
                std::cout << "    reg[28-29] (Output_loop_delta_dim0) = " << Output_loop_delta_dim0 << std::endl;
                std::cout << "    reg[30] (dim0_loop_tile) = " << dim0_loop_tile << std::endl;
            }
        }

        wq->enqueu_kernel(*kernel, {(uint16_t)blkcnt_x, (uint16_t)blkcnt_y, (uint16_t)1}, {0});

        // Flush output, 确保 CPU 能读到 RPU 写入的最新数据
        rpu_ddr_flush(out_ptr);

        // ==================== SPM 版本 (注释保留) ====================
        // c10::Half *out = output.data_ptr<c10::Half>();
        // spm_to_ddr(out, spm_core0_output.get_cpu_ptr(), output_nbytes);

        // ctx.export_case();
        return;
    } else {
        TORCH_CHECK(false, "rpu_launch_tile_kernel: only 2D and 3D tensors are supported");
    }
}

// ===================== Tile SPM Kernel Launch (multi-core) =====================
// SPM-resident 3D tile: input [dim0,dim1,dim2] -> output [dim0*r0, dim1*r1, dim2*r2].
// Reuses tile_general_smallC / largeC with raw SPM byte addresses:
// param0/1=input base, param2/3=output base.
//
// GDN use: GQA-expand q/k via the [N,1,Dk] repeat-middle trick — view q as
// [num_k_heads, 1, Dk] and tile {1, rep, 1} -> [num_k_heads, rep, Dk], which
// flattens to [h0,h0,h1,h1,...] = repeat_interleave on the head dim.
// With num_cores=8, each core tiles its own heads at the same SPM offsets.
void rpu_launch_tile_spm_kernel(uint32_t input_spm_addr, uint32_t output_spm_addr,
                                int64_t dim0, int64_t dim1, int64_t dim2,
                                int64_t repeat0, int64_t repeat1, int64_t repeat2,
                                int num_cores) {
    using namespace ::rhino_lkn;
    constexpr int64_t WARP_SIZE = 16;
    const int64_t dwidth = (int64_t)sizeof(c10::Half);

    const bool is_largeC = (dim2 >= 256);
    const std::string kernel_name =
        is_largeC ? "tile_general_largeC" : "tile_general_smallC";
    // Graph-aware kernel fetch — NOT KernelCache::get_kernel(name) (the RAW string
    // path: sets neither pending_kernel_id_ nor pending_kernel_name_, so during graph
    // RECORDING it hits the §12.4 raw-kernel fallback → oneshots the prefix, switches
    // to PASSTHROUGH, marks the graph non-replayable + emits the "enqueu_kernel after
    // build_batch" WARN). get_kernel_reset(name) sets pending_kernel_name_ so the
    // launch records into the graph; in PASSTHROUGH (isolated test) it falls back to
    // the cache. Mirrors GET_KERNEL(id) used by fla_conv1d/eltwise/etc.
    Kernel_t* kernel = RpuKernelGraph::active().get_kernel_reset(kernel_name);
    TORCH_CHECK(kernel != nullptr,
                "rpu_launch_tile_spm_kernel: failed to get ", kernel_name, " kernel");

    int64_t blkcnt_x, blkcnt_y;
    int64_t Input_Stride_dim0, Input_Stride_dim1;
    int64_t Output_Stride_dim0, Output_Stride_dim1;
    int64_t Output_lpstep_dim0, Output_lpstep_dim1, Output_lpstep_dim2;
    const int64_t dim0_loop_tile = 16;

    if (is_largeC) {
        blkcnt_x = dim0;
        blkcnt_y = dim1;
        Input_Stride_dim1 = dim2 * dwidth;
        Input_Stride_dim0 = dim1 * Input_Stride_dim1;
        Output_Stride_dim1 = dim2 * repeat2 * dwidth;
        Output_Stride_dim0 = dim1 * repeat1 * Output_Stride_dim1;
        Output_lpstep_dim2 = dim2 * dwidth;
        Output_lpstep_dim1 = Output_Stride_dim1 * dim1;
        Output_lpstep_dim0 = Output_Stride_dim0 * dim0;
    } else {
        blkcnt_y = (dim1 + WARP_SIZE - 1) / WARP_SIZE;
        blkcnt_x = (dim0 + dim0_loop_tile - 1) / dim0_loop_tile;
        Input_Stride_dim1 = WARP_SIZE * dim2 * dwidth;
        Input_Stride_dim0 = dim0_loop_tile * dim1 * dim2 * dwidth;
        Output_Stride_dim1 = WARP_SIZE * repeat2 * dim2 * dwidth;
        Output_Stride_dim0 =
            dim0_loop_tile * repeat1 * dim1 * repeat2 * dim2 * dwidth;
        Output_lpstep_dim2 = dim2 * dwidth;
        Output_lpstep_dim1 = dim1 * repeat2 * Output_lpstep_dim2;
        Output_lpstep_dim0 = dim0 * repeat1 * Output_lpstep_dim1;
        const int64_t Input_loop_delta_dim0 = dim1 * dim2 * dwidth;
        const int64_t Output_loop_delta_dim0 =
            dim1 * repeat1 * dim2 * repeat2 * dwidth;
        kernel->set_regs(26, (uint16_t)(Input_loop_delta_dim0 & 0xFFFF));
        kernel->set_regs(27, (uint16_t)(Input_loop_delta_dim0 >> 16));
        kernel->set_regs(28, (uint16_t)(Output_loop_delta_dim0 & 0xFFFF));
        kernel->set_regs(29, (uint16_t)(Output_loop_delta_dim0 >> 16));
        kernel->set_regs(30, (uint16_t)dim0_loop_tile);
    }

    // SPM byte addresses: param0/1 = input base, param2/3 = output base.
    kernel->set_regs(0, (uint16_t)(input_spm_addr & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(input_spm_addr >> 16));
    kernel->set_regs(2, (uint16_t)(output_spm_addr & 0xFFFF));
    kernel->set_regs(3, (uint16_t)(output_spm_addr >> 16));

    kernel->set_regs(4, (uint16_t)dim0);
    kernel->set_regs(5, (uint16_t)dim1);
    kernel->set_regs(6, (uint16_t)dim2);
    kernel->set_regs(7, (uint16_t)repeat0);
    kernel->set_regs(8, (uint16_t)repeat1);
    kernel->set_regs(9, (uint16_t)repeat2);

    kernel->set_regs(10, (uint16_t)(Input_Stride_dim0 & 0xFFFF));
    kernel->set_regs(11, (uint16_t)(Input_Stride_dim0 >> 16));
    kernel->set_regs(12, (uint16_t)(Input_Stride_dim1 & 0xFFFF));
    kernel->set_regs(13, (uint16_t)(Input_Stride_dim1 >> 16));

    kernel->set_regs(14, (uint16_t)(Output_Stride_dim0 & 0xFFFF));
    kernel->set_regs(15, (uint16_t)(Output_Stride_dim0 >> 16));
    kernel->set_regs(16, (uint16_t)(Output_Stride_dim1 & 0xFFFF));
    kernel->set_regs(17, (uint16_t)(Output_Stride_dim1 >> 16));

    kernel->set_regs(20, (uint16_t)(Output_lpstep_dim0 & 0xFFFF));
    kernel->set_regs(21, (uint16_t)(Output_lpstep_dim0 >> 16));
    kernel->set_regs(22, (uint16_t)(Output_lpstep_dim1 & 0xFFFF));
    kernel->set_regs(23, (uint16_t)(Output_lpstep_dim1 >> 16));
    kernel->set_regs(24, (uint16_t)(Output_lpstep_dim2 & 0xFFFF));
    kernel->set_regs(25, (uint16_t)(Output_lpstep_dim2 >> 16));

    kernel->set_regs(64, (uint16_t)blkcnt_x);  // gridDim.x
    kernel->set_regs(65, (uint16_t)blkcnt_y);  // gridDim.y
    kernel->set_regs(66, (uint16_t)1);         // gridDim.z

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel,
                      {(uint16_t)blkcnt_x, (uint16_t)blkcnt_y, (uint16_t)1}, cores);
}

// Isolated test: load a 3D fp16 input into SPM, tile by `repeats`, return output.
// Single-core (num_cores=1) so the golden is a plain np.tile / torch repeat.
at::Tensor rpu_tile_spm_test(const at::Tensor& input, c10::IntArrayRef repeats) {
    TORCH_CHECK(input.scalar_type() == at::kHalf, "tile_spm_test: input must be fp16");
    TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                "tile_spm_test: input must be on RPU");
    TORCH_CHECK(input.dim() == 3 && (int64_t)repeats.size() == 3,
                "tile_spm_test: only 3D input + 3-elem repeats supported");
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

    auto in_c = input.contiguous();
    const int64_t d0 = in_c.size(0), d1 = in_c.size(1), d2 = in_c.size(2);
    const int64_t r0 = repeats[0], r1 = repeats[1], r2 = repeats[2];

    const int64_t in_bytes = in_c.numel() * 2;
    const int64_t out_numel = in_c.numel() * r0 * r1 * r2;
    const int64_t out_bytes = out_numel * 2;
    using AR = SpmAllocator::AllocRequest;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{in_bytes, 1, 2},   // input (read by the kernel)
        AR{out_bytes, 1, 3},  // output (written; alive across the kernel)
    });
    const uint32_t in_addr = SPM_ALLOC.addr(0, offsets[0]);
    const uint32_t out_addr = SPM_ALLOC.addr(0, offsets[1]);

    rpu_launch_ddr_broadcast_spm_dma_immediate(
        const_cast<c10::Half*>(in_c.data_ptr<c10::Half>()), in_c.numel(), in_addr, 1);
    rpu_launch_tile_spm_kernel(in_addr, out_addr, d0, d1, d2, r0, r1, r2, 1);

    auto output = at::empty({d0 * r0, d1 * r1, d2 * r2}, in_c.options());
    rpu_launch_spm_copy_ddr_dma_immediate(out_addr, output.data_ptr<c10::Half>(),
                                          out_numel);
    SPM_ALLOC.reset_temporary();
    return output;
}
