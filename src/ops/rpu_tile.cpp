// rpu_tile.cpp - Tile/repeat kernel implementation for RPU
#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
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

    size_t dwidth = sizeof(c10::Half);
    int64_t rank = input.dim();

    // Get base tensor (the original data before expand)
    at::Tensor base;
    if (input._base().defined()) {
        base = input._base();
    } else {
        base = input;
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

    auto* wq = GET_QUEUE(1);  // 复用缓存的 queue

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

      kernel->set_regs(64, (uint16_t)blk_cnt); // gridDim.x
      kernel->set_regs(65, (uint16_t)0x1); // gridDim.y
      kernel->set_regs(66, (uint16_t)0x1); // gridDim.z

      // 设置 SCM args (per-thread loop counts)
      for (int i = 0; i < WARP_SIZE; i++) {
          rpu_set_legacy_scm_u16_checked(
              kernel, (uint32_t)i + 4096, thread_params[i], "tile");
      }

      wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1}, {0});

      // Flush output, 确保 CPU 能读到 RPU 写入的最新数据
      rpu_ddr_flush(out_ptr);

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

        wq->enqueu_kernel(*kernel, {(uint16_t)blkcnt_x, (uint16_t)blkcnt_y, (uint16_t)1}, {0});

        // Flush output, 确保 CPU 能读到 RPU 写入的最新数据
        rpu_ddr_flush(out_ptr);

        return;
    } else {
        TORCH_CHECK(false, "rpu_launch_tile_kernel: only 2D and 3D tensors are supported");
    }
}

// ===================== Tile SPM Kernel Launch (multi-core) =====================
// SPM-resident 3D tile: input [dim0,dim1,dim2] -> output [dim0*r0, dim1*r1, dim2*r2].
// Reuses the same tile_general_smallC / largeC kernels as the eager DDR path.
// The host ABI takes raw SPM byte addresses for the input and output bases.
//
// GDN use: GQA-expand q/k via the [N,1,Dk] repeat-middle trick — view q as
// [num_k_heads, 1, Dk] and tile {1, rep, 1} -> [num_k_heads, rep, Dk], which
// flattens to [h0,h0,h1,h1,...] = repeat_interleave on the head dim. With eight
// SPMD cores, each core tiles its own heads at the
// same SPM offsets (broadcast mode).
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
    // Use the graph-aware lookup so capture records this launch; eager execution
    // falls back to the cache.
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

    // SPM byte addresses (a_g1B): param0/1 = input base, param2/3 = output base.
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
