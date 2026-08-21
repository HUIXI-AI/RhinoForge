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

// Multi-core 3D SPM tile over contiguous FP16 tensors.
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
