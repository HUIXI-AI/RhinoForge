// rpu_cast.cpp — on-device dtype cast (unary_cast_uint8_fp16).
//
// Hardware kernel: `unary_cast_uint8_fp16`, fetched by name via KernelCache.
// Stage uint8 DDR input in SPM, cast to fp16 SPM, then copy the result to DDR.
// The cast is exact for 0..255; param2 stays 0.
// Register contract:
//   param0 (reg0):   normal_blk_n = 63 * WARP_VECTOR_SIZE(256) = 16128 (per-block elems)
//   param1 (reg1):   last_blk_n
//   param4 (reg4/5): input  SPM byte addr (g1b, absolute)
//   param6 (reg6/7): output SPM byte addr (g1b, absolute)
//   grid (reg64-66): {ceil(n/16128), 1, 1}
// The cos-gate is the oracle that this packing is correct.

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <limits>
#include <vector>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;

static constexpr int kCastNormalBlk = 63 * 256;  // 16128

// Cast n contiguous uint8 elements at in_spm_addr → n fp16 at out_spm_addr (core-0 SPM).
void rpu_launch_cast_uint8_fp16_kernel(uint32_t in_spm_addr,
                                       uint32_t out_spm_addr,
                                       int64_t n)
{
    TORCH_CHECK(n > 0, "rpu_launch_cast_uint8_fp16_kernel: n must be positive, got ", n);

    const int normal_blk = kCastNormalBlk;
    const int blk_cnt = static_cast<int>((n + normal_blk - 1) / normal_blk);
    const int last_blk = static_cast<int>(n - static_cast<int64_t>(normal_blk) * (blk_cnt - 1));

    // Pre-load the program into the shared cache. get_kernel() has the dynamic .ref
    // fallback (loads + caches program+kernel by name); get_program() — which the graph's
    // dynamic-name RECORDING path calls — does NOT, so without this the graph fetch fails
    // "program not found". Idempotent (a cheap map hit after the first load).
    TORCH_CHECK(KernelCache::instance().get_kernel("unary_cast_uint8_fp16") != nullptr,
                "Failed to load unary_cast_uint8_fp16 kernel from .ref");
    // Graph-aware by-name fetch (dynamic-kernel API): PASSTHROUGH returns the shared
    // cache kernel; RECORDING clones + registers pending_kernel_name so the following
    // proxy enqueue records a graph node (a raw KernelCache::get_kernel would leave the
    // graph without a kernel_name → "launched without KernelId or kernel_name" WARN →
    // passthrough fallback, defeating the capture). Resets regs internally.
    Kernel_t *kernel = RpuKernelGraph::active().get_kernel_reset("unary_cast_uint8_fp16");
    TORCH_CHECK(kernel != nullptr, "Failed to get unary_cast_uint8_fp16 kernel from .ref");

    kernel->set_regs(0, (uint16_t)normal_blk);                     // param0: elems/block
    kernel->set_regs(1, (uint16_t)last_blk);                       // param1: last-block elems
    kernel->set_regs(4, (uint16_t)(in_spm_addr & 0xFFFF));         // param4 lo: in SPM byte addr
    kernel->set_regs(5, (uint16_t)((in_spm_addr >> 16) & 0xFFFF)); // param5 hi
    kernel->set_regs(6, (uint16_t)(out_spm_addr & 0xFFFF));        // param6 lo: out SPM byte addr
    kernel->set_regs(7, (uint16_t)((out_spm_addr >> 16) & 0xFFFF));// param7 hi
    kernel->set_regs(64, (uint16_t)blk_cnt);                       // grid.x
    kernel->set_regs(65, (uint16_t)1);                             // grid.y
    kernel->set_regs(66, (uint16_t)1);                             // grid.z

    auto *wq = GET_QUEUE(1);
    wq->set_broadcast_mode(true);
    wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1}, {(uint8_t)0});
}

// torch op: cast_uint8_fp16(input uint8) -> fp16, all RPU. Flattens to n = numel,
// stages through SPM (DMA in → cast → DMA out). See file header.
//
// ponytail: single-shot SPM staging (in n bytes + out 2n bytes must fit one core's
// SPM). Fine for vision patch_embed (≤ ~2.7 MB); chunk if a huge tensor ever needs it.
at::Tensor rpu_cast_uint8_fp16(const at::Tensor &input)
{
    RECORD_FUNCTION("rpu::cast_uint8_fp16", {});
    TORCH_CHECK(input.scalar_type() == at::kByte,
                "cast_uint8_fp16: input must be uint8, got ", input.scalar_type());
    TORCH_CHECK(input.is_contiguous(), "cast_uint8_fp16: input must be contiguous");
    TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                "cast_uint8_fp16: input must be on RPU device");

    const int64_t n = input.numel();
    TORCH_CHECK(n > 0, "cast_uint8_fp16: empty input");
    // Input DMA moves n uint8 bytes; add_dma requires a 16-byte-aligned byte count.
    TORCH_CHECK((n % 16) == 0, "cast_uint8_fp16: numel must be a multiple of 16, got ", n);

    auto out = at::empty(input.sizes(), input.options().dtype(at::kHalf));

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    // Input (uint8, n bytes) and output (fp16, 2n bytes) are live simultaneously
    // (the cast reads in / writes out), so give them distinct SPM regions.
    const uint32_t in_off = SPM_ALLOC.alloc_temporary(n);
    const uint32_t out_off = SPM_ALLOC.alloc_temporary(n * 2);
    const uint32_t in_spm = SPM_ALLOC.addr(0, in_off);
    const uint32_t out_spm = SPM_ALLOC.addr(0, out_off);

    // 1. uint8 DDR → SPM (immediate: build_batch + enqueu_batch(true) WAITS, so
    //    SPM holds the input before the cast runs). Step 2 direct-launches on
    //    the dedicated PASSTHROUGH kernel queue, isolated from this immediate
    //    DMA QueueCache batch. The helper also consumes the completed one-shot
    //    state before returning.
    rpu_launch_ddr_spm_bytes_immediate(input.data_ptr(), n, in_spm);
    // 2. cast SPM(uint8) → SPM(fp16).
    rpu_launch_cast_uint8_fp16_kernel(in_spm, out_spm, n);
    // 3. fp16 SPM → DDR.
    rpu_launch_spm_copy_ddr_dma_immediate(out_spm, out.data_ptr<c10::Half>(), n);

    SPM_ALLOC.reset_temporary();
    return out;
}

// torch op: cast_uint8_fp16_into(input uint8, out fp16) -> (), all RPU. Graph-CAPTURABLE
// out-param variant: writes the cast into a caller-provided output. Inside a graph
// capture (graph_dma::active()) the 3 staging steps record as graph nodes and REPLAY as
// one batch. Fixed graph DMAs bake the src/dst at BUILD, so the caller must pass persistent
// stable-address buffers and refresh only their CONTENTS per frame (input via a flushed
// upload; the graph runtime orders DDR→SPM→cast→SPM→DDR via segments — no hand barriers).
// Outside a capture it falls back to the immediate path (same as rpu_cast_uint8_fp16).
void rpu_cast_uint8_fp16_into(const at::Tensor &input, at::Tensor &output)
{
    RECORD_FUNCTION("rpu::cast_uint8_fp16_into", {});
    TORCH_CHECK(input.scalar_type() == at::kByte,
                "cast_uint8_fp16_into: input must be uint8, got ", input.scalar_type());
    TORCH_CHECK(output.scalar_type() == at::kHalf,
                "cast_uint8_fp16_into: output must be fp16, got ", output.scalar_type());
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
                "cast_uint8_fp16_into: input and output must be contiguous");
    TORCH_CHECK(input.device().type() == at::kPrivateUse1
                && output.device().type() == at::kPrivateUse1,
                "cast_uint8_fp16_into: input and output must be on RPU device");
    const int64_t n = input.numel();
    TORCH_CHECK(n > 0 && output.numel() == n,
                "cast_uint8_fp16_into: numel mismatch (in ", n, " out ", output.numel(), ")");
    TORCH_CHECK((n % 16) == 0, "cast_uint8_fp16_into: numel must be a multiple of 16, got ", n);

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    const uint32_t in_off = SPM_ALLOC.alloc_temporary(n);
    const uint32_t out_off = SPM_ALLOC.alloc_temporary(n * 2);
    const uint32_t in_spm = SPM_ALLOC.addr(0, in_off);
    const uint32_t out_spm = SPM_ALLOC.addr(0, out_off);

    if (graph_dma::active()) {
        // Graph path: fixed DMAs record as nodes (replayed in one batch). The uint8
        // input is reinterpreted as n/2 fp16 elements so the fp16 broadcast helper moves
        // exactly n bytes DDR→core-0 SPM (dtype-agnostic byte copy). Coherency: the
        // caller flushes the input buffer via its per-frame upload; the output is
        // consumed on-device by the following GEMM.
        rpu_launch_ddr_broadcast_spm_dma(
            reinterpret_cast<c10::Half *>(input.data_ptr()), n / 2, in_spm, /*num_cores=*/1);
        rpu_launch_cast_uint8_fp16_kernel(in_spm, out_spm, n);
        rpu_launch_spm_copy_ddr_dma(out_spm, output.data_ptr<c10::Half>(), n);
    } else {
        rpu_launch_ddr_spm_bytes_immediate(input.data_ptr(), n, in_spm);
        rpu_launch_cast_uint8_fp16_kernel(in_spm, out_spm, n);
        rpu_launch_spm_copy_ddr_dma_immediate(out_spm, output.data_ptr<c10::Half>(), n);
    }

    SPM_ALLOC.reset_temporary();
}

// Qwen3.5 MoE route-index widening. Generic Gather consumes I32 indices,
// while the routing kernels produce sorted raw U16 token IDs.
void rpu_launch_cast_uint16_int32_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t output_spm_addr,
    int64_t num_elements,
    int num_cores)
{
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "cast_uint16_int32: num_cores must be in [1,8], got ",
                num_cores);
    TORCH_CHECK(num_elements > 0,
                "cast_uint16_int32: num_elements must be positive, got ",
                num_elements);
    const int64_t block_count =
        (num_elements + kCastNormalBlk - 1) / kCastNormalBlk;
    TORCH_CHECK(block_count <= std::numeric_limits<uint16_t>::max(),
                "cast_uint16_int32: grid.x must fit u16, got N=",
                num_elements);
    const int64_t last_block =
        num_elements - (block_count - 1) * kCastNormalBlk;
    TORCH_CHECK(input_spm_addr % 2 == 0 && output_spm_addr % 32 == 0,
                "cast_uint16_int32: input must be u16 aligned and output "
                "must be 32-byte aligned");

    Kernel_t* kernel = GET_KERNEL(KernelId::CAST_UINT16_INT32);
    TORCH_CHECK(kernel != nullptr,
                "cast_uint16_int32: unary_cast_uint16_int32 is absent from "
                "the active oplib");
    kernel->reset_regs();
    kernel->set_regs(0, static_cast<uint16_t>(kCastNormalBlk));
    kernel->set_regs(1, static_cast<uint16_t>(last_block));
    kernel->set_regs(4, static_cast<uint16_t>(input_spm_addr & 0xFFFF));
    kernel->set_regs(5, static_cast<uint16_t>(input_spm_addr >> 16));
    kernel->set_regs(6, static_cast<uint16_t>(output_spm_addr & 0xFFFF));
    kernel->set_regs(7, static_cast<uint16_t>(output_spm_addr >> 16));
    kernel->set_regs(64, static_cast<uint16_t>(block_count));
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
        *kernel, {static_cast<uint16_t>(block_count), 1, 1}, cores);
}
