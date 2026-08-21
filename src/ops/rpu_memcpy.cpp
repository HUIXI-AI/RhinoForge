// Multi-core DDR/SPM transfer wrappers.

#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"  // for SPM_ALLOC (LOCAL-mode offset extraction)
#include <c10/util/Half.h>
#include <cstdint>
#include <limits>
#include <vector>
#include <unistd.h> // for usleep


using namespace ::rhino_lkn;

// =============================================================================
// DDR to SPM Multi-Core Kernel Launcher
// =============================================================================
//
// Public launch ABI:
//   param0/1: element count
//   param2/3: iteration stride
//   param4/5: DDR source address in 256-byte units
//   param6/7: SPM destination byte address
//   param10/11: block stride
//   param12/13: core stride
// A zero core stride broadcasts the same DDR region to each core's local SPM.
// =============================================================================

constexpr int NUM_CORES_MEMCPY = 8;
constexpr int MAX_BLOCKS_MEMCPY = 8;

// Submit a synchronous one-shot batch and explicitly consume its prepared
// state. Queue_t intentionally retains built batches for GraphCache replay;
// immediate DMA wrappers do not replay them. Leaving that state live makes the
// next eager kernel invalidate it through the SDK WARN path.
static void submit_immediate_batch(Queue_t* wq, const char* caller) {
    const uint32_t build_rc = wq->build_batch();
    TORCH_CHECK(build_rc == 0, caller, ": build_batch SDK rc=", build_rc);
    const uint32_t enqueue_rc = wq->enqueu_batch(/*wait_finish=*/true);
    TORCH_CHECK(enqueue_rc == 0, caller, ": enqueu_batch SDK rc=", enqueue_rc);
    const uint32_t discard_rc = wq->discard_completed_batch();
    TORCH_CHECK(discard_rc == 0,
                caller, ": discard_completed_batch SDK rc=", discard_rc);
}

// =============================================================================
// Store-loop split for the v2 transfer contract.
// =============================================================================
// The v2 ABI accepts an outer count plus a bounded remainder. Split large
// transfers into 32256-element chunks so both fields stay in range.
// 32256 = 126 * 256, aligned to the 256-element loop step.
// =============================================================================
constexpr int64_t STORE_INNER_STEP = 32256;  // 126 * 256, max safe for s16

static void compute_store_split(int64_t elements_per_block,
                                uint16_t& store_outer, uint16_t& store_remainder) {
    if (elements_per_block <= STORE_INNER_STEP) {
        store_outer = 0;
        store_remainder = (uint16_t)elements_per_block;
    } else {
        int64_t outer = elements_per_block / STORE_INNER_STEP;
        int64_t rem = elements_per_block - outer * STORE_INNER_STEP;
        if (rem == 0) {
            outer--;
            rem = STORE_INNER_STEP;
        }
        store_outer = (uint16_t)outer;
        store_remainder = (uint16_t)rem;
    }
}





// =============================================================================
// SPM Memset (多核, 8 个 core 的 SPM 同一地址清零)
// =============================================================================
// 将全部 8 个 core 的 SPM 在 spm_addr 起始的 num_elements 个 fp16 元素清零。
// 参数约定与 ddr2spm_multi_core_v2 一致，使用 memset_spm_multi_core_v2 kernel。
// =============================================================================

void rpu_launch_memset_spm_multicore(
    uint32_t spm_addr,
    int64_t num_elements)
{
    TORCH_CHECK((spm_addr % SPM_BANK_SIZE) == 0,
                "memset_spm: spm_addr must be 32-byte aligned");
    TORCH_CHECK(num_elements > 0,
                "memset_spm: num_elements must be > 0");

    size_t dwidth = sizeof(c10::Half);

    constexpr size_t V16_PER_LOOP = 256 / 16;

    int num_blocks = (num_elements % MAX_BLOCKS_MEMCPY == 0) ? MAX_BLOCKS_MEMCPY : 1;
    int64_t elements_per_block = num_elements / num_blocks;

    uint32_t loop_end = elements_per_block;
    uint32_t step_v16 = V16_PER_LOOP;
    uint32_t block_stride = elements_per_block * dwidth;

    uint16_t store_outer, store_remainder;
    compute_store_split(elements_per_block, store_outer, store_remainder);

    auto setup_regs = [=](Kernel_t* kernel) {
        kernel->set_regs(0, (uint16_t)(loop_end & 0xFFFF));
        kernel->set_regs(1, (uint16_t)(loop_end >> 16));
        kernel->set_regs(2, (uint16_t)(step_v16 & 0xFFFF));
        kernel->set_regs(3, (uint16_t)(step_v16 >> 16));
        kernel->set_regs(4, (uint16_t)0);
        kernel->set_regs(5, (uint16_t)0);
        kernel->set_regs(6, (uint16_t)(spm_addr & 0xFFFF));
        kernel->set_regs(7, (uint16_t)(spm_addr >> 16));
        kernel->set_regs(8, store_outer);
        kernel->set_regs(9, store_remainder);
        kernel->set_regs(10, (uint16_t)(block_stride & 0xFFFF));
        kernel->set_regs(11, (uint16_t)(block_stride >> 16));
        kernel->set_regs(12, (uint16_t)0);
        kernel->set_regs(13, (uint16_t)0);
        kernel->set_regs(14, (uint16_t)STORE_INNER_STEP);
        kernel->set_regs(15, (uint16_t)(STORE_INNER_STEP * dwidth));
    };

      Kernel_t* kernel = GET_KERNEL(KernelId::MEMSET_SPM_MULTI_CORE_V2);
      TORCH_CHECK(kernel != nullptr, "Failed to get memset_spm_multi_core_v2 kernel");
      kernel->reset_regs();
      setup_regs(kernel);

      auto* wq = GET_QUEUE(NUM_CORES_MEMCPY);
      wq->set_broadcast_mode(true);
      wq->enqueu_kernel(*kernel, {(uint16_t)num_blocks, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

// =============================================================================
// Fill SPM (multi-core broadcast): write `value` into num_elements halfs at the
// per-core SPM addr. Uses the graph-aware operator-library `fill` (1D) kernel.
// Host ABI: reg0/1 SPM byte offset, reg2/3 element count, reg4 fp16 value,
// reg5 mem-flag (0=SPM), grid.x=ceil(n/2048). Graph-aware fetch records it into
// the fused graph, so get_program requires it to be preloaded.
void rpu_launch_fill_spm_kernel(uint32_t spm_addr, int64_t num_elements,
                                c10::Half value, int num_cores,
                                int core_begin) {
    TORCH_CHECK(num_elements > 0,
                "fill_spm: num_elements must be > 0");
    TORCH_CHECK(core_begin >= 0 && num_cores >= 1 &&
                    core_begin + num_cores <= SpmAllocator::NUM_CORES,
                "fill_spm: selected core range must be inside [0, ",
                SpmAllocator::NUM_CORES, "), got [", core_begin, ", ",
                core_begin + num_cores, ")");
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "fill_spm: SPM allocator is not initialized");
    const uint64_t element_count = static_cast<uint64_t>(num_elements);
    TORCH_CHECK(element_count <=
                    SpmAllocator::SPM_USABLE / sizeof(c10::Half),
                "fill_spm: write byte count exceeds one core's usable SPM");
    const size_t byte_count =
        static_cast<size_t>(element_count * sizeof(c10::Half));
    const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
    const uint64_t candidate = spm_addr;
    TORCH_CHECK(candidate >= spm_base &&
                    candidate - spm_base <=
                        SpmAllocator::SPM_USABLE - byte_count,
                "fill_spm: complete write interval is outside the current "
                "allocator's core-0 SPM");

    constexpr uint16_t mem_flag = 0;  // SPM; never admit the DDR fill mode.
    static_assert(mem_flag == 0);
    auto& graph = RpuKernelGraph::active();
    graph.stage_kernel_no_ddr("fill", GraphKernelNoDdrProof::FillSpm);
    Kernel_t* kernel = graph.get_kernel_reset("fill");
    TORCH_CHECK(kernel != nullptr, "fill_spm: failed to get fill kernel");
    const int64_t VAL_NUM_PER_WRP = 2048;
    const uint16_t gx = (uint16_t)(
        (num_elements + VAL_NUM_PER_WRP - 1) / VAL_NUM_PER_WRP);
    kernel->set_regs(0, (uint16_t)(spm_addr & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(spm_addr >> 16));
    kernel->set_regs(2, (uint16_t)(num_elements & 0xFFFF));
    kernel->set_regs(3, (uint16_t)(num_elements >> 16));
    kernel->set_regs(4, value.x);  // fp16 fill value (0 for zeroing)
    kernel->set_regs(5, mem_flag);
    kernel->set_regs(64, gx);
    kernel->set_regs(65, (uint16_t)1);
    kernel->set_regs(66, (uint16_t)1);
    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) {
        cores.push_back(static_cast<uint8_t>(core_begin + i));
    }
    wq->enqueu_kernel(*kernel, {gx, 1, 1}, cores);
}

// =============================================================================
// DDR -> SPM Scatter (多核, 每个 core 从不同 DDR 区域读取)
// =============================================================================
// Each core reads DDR data from ddr_base + core_id * core_stride_bytes,
// writes to the SAME local SPM address (spm_addr) on each core.
//
// Implementation: per-core kernel launch with a compensated SPM base. The
// public wrapper ABI applies core_stride to both DDR and SPM addresses:
//   DDR_addr = ddr_base + core_id * stride  (wanted)
//   SPM_addr = spm_base + core_id * stride  (unwanted)
// We set spm_base = spm_addr - core_id * stride per launch, so:
//   effective SPM = (spm_addr - core_id*stride) + core_id*stride = spm_addr
//
// core_stride is expressed in bytes. DDR base must be 256-byte aligned.
// =============================================================================




// =============================================================================
// DMA-based memcpy primitives (graph-only)
// =============================================================================
// File-local helper: recover the local SPM offset from the unified-mode
// address returned by FusedModelBase::addr(0, name), then compose the
// per-core absolute address from SPM_ALLOC.addr(core, offset), matching the
// SPM→DDR direction.
static inline uint64_t spm_unified_to_per_core_abs(int core,
                                                    uint32_t spm_addr_unified) {
    uint32_t local_offset = spm_addr_unified - SPM_ALLOC.addr(0, 0);
    return static_cast<uint64_t>(SPM_ALLOC.addr(core, local_offset));
}

static void validate_fp16_dma_owner_slice(const at::Tensor& tensor,
                                          size_t byte_begin,
                                          size_t byte_count,
                                          const char* caller) {
    TORCH_CHECK(tensor.defined() &&
                    tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == at::kHalf &&
                    tensor.layout() == c10::Layout::Strided &&
                    tensor.is_contiguous(),
                caller, ": owner must be a contiguous FP16 RPU Tensor");
    const size_t tensor_bytes = tensor.nbytes();
    TORCH_CHECK(byte_count > 0 && byte_begin <= tensor_bytes &&
                    byte_count <= tensor_bytes - byte_begin,
                caller, ": owner byte slice [", byte_begin, ",",
                byte_begin + byte_count, ") exceeds Tensor bytes ",
                tensor_bytes);
}

static size_t checked_size_t_multiply(uint64_t lhs,
                                      uint64_t rhs,
                                      const char* caller,
                                      const char* quantity) {
    const uint64_t limit = std::numeric_limits<size_t>::max();
    TORCH_CHECK(lhs == 0 || rhs <= limit / lhs,
                caller, ": ", quantity, " overflows size_t");
    return static_cast<size_t>(lhs * rhs);
}

static size_t checked_size_t_add(size_t lhs,
                                 size_t rhs,
                                 const char* caller,
                                 const char* quantity) {
    TORCH_CHECK(rhs <= std::numeric_limits<size_t>::max() - lhs,
                caller, ": ", quantity, " overflows size_t");
    return lhs + rhs;
}

void rpu_launch_ddr_broadcast_spm_dma(c10::Half* ddr_ptr,
                                       int64_t num_elements,
                                       uint32_t spm_addr_unified,
                                       int num_cores) {
    TORCH_CHECK(ddr_ptr != nullptr, "ddr_broadcast_spm_dma: null ddr_ptr");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "ddr_broadcast_spm_dma: num_cores 1..8, got ", num_cores);
    TORCH_CHECK(num_elements > 0,
                "ddr_broadcast_spm_dma: num_elements > 0, got ", num_elements);

    const size_t bytes = static_cast<size_t>(num_elements) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0,
                "ddr_broadcast_spm_dma: bytes (", bytes, ") must be 16-byte aligned");
    TORCH_CHECK(bytes <= (8ULL << 20),
                "ddr_broadcast_spm_dma: single-DMA cap is 8 MB, got ", bytes,
                " bytes (auto-split is future work)");

    // 顶替 v5 BATCH_MODE_ACTIVE 入口契约:graph scope 内走 graph_dma capture(进 batch);
    // PASSTHROUGH 下自动 fallback 到 immediate 一次性 batch + enqueu_batch。
    // FusedModelBase::run_all_layers 不再外包 graph scope,subclass build_layer_subgraph
    // 在两种状态下都得能跑。
    if (!graph_dma::active()) {
        rpu_launch_ddr_broadcast_spm_dma_immediate(ddr_ptr, num_elements,
                                                    spm_addr_unified, num_cores);
        return;
    }

    rpu_ddr_flush(ddr_ptr);
    const uint64_t ddr_abs = RpuGetDevAddr(ddr_ptr);

    // Pre: ch=1..nc-1 streams wait for stream 0 (upstream compute or ch-0 DMA).
    for (int ch = 1; ch < num_cores; ++ch)
        graph_dma::emit_barrier(static_cast<uint8_t>(ch * 2), 0);

    // num_cores DMAs: same DDR src, per-core SPM dst.
    for (int c = 0; c < num_cores; ++c)
        graph_dma::emit_fixed(ddr_abs,
                          spm_unified_to_per_core_abs(c, spm_addr_unified),
                          bytes, c);

    // Post: stream 0 (downstream compute) waits for ch=1..nc-1.
    for (int ch = 1; ch < num_cores; ++ch)
        graph_dma::emit_barrier(0, static_cast<uint8_t>(ch * 2));
}

void rpu_launch_ddr_broadcast_spm_dma(
        const at::Tensor& ddr_tensor,
        int64_t src_offset_elements,
        int64_t num_elements,
        uint32_t spm_addr_unified,
        int num_cores) {
    TORCH_CHECK(src_offset_elements >= 0 && num_elements > 0,
                "ddr_broadcast_spm_dma owned: invalid element range");
    const size_t byte_begin =
        static_cast<size_t>(src_offset_elements) * sizeof(c10::Half);
    const size_t byte_count =
        static_cast<size_t>(num_elements) * sizeof(c10::Half);
    validate_fp16_dma_owner_slice(
        ddr_tensor, byte_begin, byte_count,
        "ddr_broadcast_spm_dma owned");
    if (graph_dma::active()) {
        graph_dma::stage_census_fixed_burst(
            ddr_tensor, GraphOuterFastDmaSide::Source,
            byte_begin, byte_count, static_cast<size_t>(num_cores));
    }
    rpu_launch_ddr_broadcast_spm_dma(
        ddr_tensor.data_ptr<c10::Half>() + src_offset_elements,
        num_elements, spm_addr_unified, num_cores);
}

void rpu_launch_ddr_broadcast_spm_dma_mutable(const uint64_t* live_src_base,
                                                int64_t src_offset_bytes,
                                                int64_t num_elements,
                                                uint32_t spm_addr_unified,
                                                int num_cores) {
    TORCH_CHECK(live_src_base != nullptr,
                "ddr_broadcast_spm_dma_mutable: null live_src_base");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "ddr_broadcast_spm_dma_mutable: num_cores 1..8, got ", num_cores);
    TORCH_CHECK(num_elements > 0,
                "ddr_broadcast_spm_dma_mutable: num_elements > 0, got ", num_elements);

    const size_t bytes = static_cast<size_t>(num_elements) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0,
                "ddr_broadcast_spm_dma_mutable: bytes (", bytes, ") must be 16-byte aligned");
    TORCH_CHECK(bytes <= (8ULL << 20),
                "ddr_broadcast_spm_dma_mutable: single-DMA cap is 8 MB, got ", bytes,
                " bytes (auto-split is future work)");

    // PASSTHROUGH fallback — deref live_src_base 一次,直接发 immediate batch。
    // graph_dma::emit_mutable_* 的"跨 replay re-deref"语义在 PASSTHROUGH 无意义,
    // 退化成跟 _immediate 等价的一次性 batch。
    if (!graph_dma::active()) {
        RpuKernelGraph::check_direct_immediate_dma_allowed(
            "ddr_broadcast_spm_dma_mutable");
        const uint64_t ddr_abs = (*live_src_base) +
                                 static_cast<uint64_t>(src_offset_bytes);
        Queue_t* wq = QueueCache::instance().get(static_cast<uint8_t>(num_cores));
        for (int ch = 1; ch < num_cores; ++ch)
            wq->add_barrier(static_cast<uint8_t>(ch * 2), 0);
        for (int c = 0; c < num_cores; ++c)
            rpu_add_dma_checked(*wq, ddr_abs,
                                spm_unified_to_per_core_abs(c, spm_addr_unified),
                                bytes, c, "ddr_broadcast_spm_dma_mutable");
        for (int ch = 1; ch < num_cores; ++ch)
            wq->add_barrier(0, static_cast<uint8_t>(ch * 2));
        submit_immediate_batch(wq, "ddr_broadcast_spm_dma_mutable");
        return;
    }

    // No rpu_ddr_flush here — caller flushes the live buffer per forward.
    // Same barrier structure as the immutable broadcast wrapper.
    for (int ch = 1; ch < num_cores; ++ch)
        graph_dma::emit_barrier(static_cast<uint8_t>(ch * 2), 0);

    for (int c = 0; c < num_cores; ++c)
        graph_dma::emit_mutable_src(live_src_base, src_offset_bytes,
                                   spm_unified_to_per_core_abs(c, spm_addr_unified),
                                   bytes, c);

    for (int ch = 1; ch < num_cores; ++ch)
        graph_dma::emit_barrier(0, static_cast<uint8_t>(ch * 2));
}

void rpu_launch_ddr_broadcast_spm_dma_mutable(
        uint64_t& live_src_base,
        const at::Tensor& owner_tensor,
        int64_t src_offset_bytes,
        int64_t num_elements,
        uint32_t spm_addr_unified,
        int num_cores) {
    TORCH_CHECK(src_offset_bytes >= 0 && num_elements > 0,
                "ddr_broadcast_spm_dma_mutable owned: invalid byte range");
    const size_t byte_begin = static_cast<size_t>(src_offset_bytes);
    const size_t byte_count =
        static_cast<size_t>(num_elements) * sizeof(c10::Half);
    validate_fp16_dma_owner_slice(
        owner_tensor, byte_begin, byte_count,
        "ddr_broadcast_spm_dma_mutable owned");
    if (graph_dma::active()) {
        graph_dma::stage_census_mutable_burst(
            live_src_base, owner_tensor, GraphOuterFastDmaSide::Source,
            byte_begin, byte_count, static_cast<size_t>(num_cores));
    }
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &live_src_base, src_offset_bytes, num_elements,
        spm_addr_unified, num_cores);
}

namespace {

void validate_canonical_dma_burst_shape(
        const char* caller, size_t bytes, int num_cores) {
    TORCH_CHECK(graph_dma::active(), caller,
                ": schema-v12 canonical DMA requires an active Graph");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                caller, ": num_cores must be 1..8, got ", num_cores);
    TORCH_CHECK(bytes > 0 && bytes % 16 == 0 && bytes <= (8ULL << 20),
                caller, ": bytes must be 16-byte aligned and <= 8 MiB, got ",
                bytes);
}

void emit_canonical_pre_barriers(int num_cores) {
    for (int channel = 1; channel < num_cores; ++channel) {
        graph_dma::emit_barrier(static_cast<uint8_t>(channel * 2), 0);
    }
}

void emit_canonical_post_barriers(int num_cores) {
    for (int channel = 1; channel < num_cores; ++channel) {
        graph_dma::emit_barrier(0, static_cast<uint8_t>(channel * 2));
    }
}

}  // namespace

void rpu_launch_canonical_ddr_broadcast_spm_dma_fixed(
        const GraphDmaSemanticEndpoint& endpoint,
        uint64_t ddr_src_addr,
        size_t bytes,
        const std::array<uint64_t, 8>& spm_dst_addrs,
        int num_cores) {
    validate_canonical_dma_burst_shape(
        "canonical_ddr_broadcast_spm_dma_fixed", bytes, num_cores);
    TORCH_CHECK(ddr_src_addr != 0,
                "canonical_ddr_broadcast_spm_dma_fixed: null DDR address");
    emit_canonical_pre_barriers(num_cores);
    for (int core = 0; core < num_cores; ++core) {
        TORCH_CHECK(spm_dst_addrs[core] != 0,
                    "canonical_ddr_broadcast_spm_dma_fixed: null SPM peer");
        graph_dma::emit_fixed(
            endpoint, ddr_src_addr, spm_dst_addrs[core], bytes,
            static_cast<uint8_t>(core));
    }
    emit_canonical_post_barriers(num_cores);
}

void rpu_launch_canonical_ddr_broadcast_spm_dma_mutable_src(
        const GraphDmaSemanticEndpoint& endpoint,
        const uint64_t* live_src_base,
        int64_t src_offset_bytes,
        size_t bytes,
        const std::array<uint64_t, 8>& spm_dst_addrs,
        int num_cores) {
    validate_canonical_dma_burst_shape(
        "canonical_ddr_broadcast_spm_dma_mutable_src", bytes, num_cores);
    TORCH_CHECK(live_src_base != nullptr && src_offset_bytes >= 0,
                "canonical_ddr_broadcast_spm_dma_mutable_src: invalid live "
                "source");
    emit_canonical_pre_barriers(num_cores);
    for (int core = 0; core < num_cores; ++core) {
        TORCH_CHECK(spm_dst_addrs[core] != 0,
                    "canonical_ddr_broadcast_spm_dma_mutable_src: null SPM "
                    "peer");
        graph_dma::emit_mutable_src(
            endpoint, live_src_base, src_offset_bytes, spm_dst_addrs[core],
            bytes, static_cast<uint8_t>(core));
    }
    emit_canonical_post_barriers(num_cores);
}

void rpu_launch_canonical_spm_copy_ddr_dma_fixed(
        const GraphDmaSemanticEndpoint& endpoint,
        uint64_t spm_src_addr,
        uint64_t ddr_dst_addr,
        size_t bytes) {
    validate_canonical_dma_burst_shape(
        "canonical_spm_copy_ddr_dma_fixed", bytes, /*num_cores=*/1);
    TORCH_CHECK(spm_src_addr != 0 && ddr_dst_addr != 0,
                "canonical_spm_copy_ddr_dma_fixed: null physical peer");
    graph_dma::emit_fixed(
        endpoint, spm_src_addr, ddr_dst_addr, bytes, /*channel=*/0);
}

// File-local helper: pre/post barrier fence for an N-channel DMA burst.
// All channels c>0 wait for stream 0 on entry; stream 0 waits for all c>0 on
// exit. Same convention used by rpu_launch_ddr_broadcast_spm_dma. No-op when
// num_cores == 1 (single channel, no cross-stream sync needed).
static inline void emit_scatter_pre_barriers(int num_cores) {
    for (int ch = 1; ch < num_cores; ++ch)
        graph_dma::emit_barrier(static_cast<uint8_t>(ch * 2), 0);
}

static inline void emit_scatter_post_barriers(int num_cores) {
    for (int ch = 1; ch < num_cores; ++ch)
        graph_dma::emit_barrier(0, static_cast<uint8_t>(ch * 2));
}

void rpu_launch_ddr_scatter_spm_dma(c10::Half* ddr_base,
                                     int64_t   elements_per_core,
                                     int64_t   core_stride_bytes,
                                     uint32_t  spm_addr_unified,
                                     int       num_cores) {
    TORCH_CHECK(ddr_base != nullptr, "ddr_scatter_spm_dma: null ddr_base");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "ddr_scatter_spm_dma: num_cores 1..8, got ", num_cores);
    TORCH_CHECK(elements_per_core > 0,
                "ddr_scatter_spm_dma: elements_per_core > 0, got ",
                elements_per_core);

    const size_t bytes =
        static_cast<size_t>(elements_per_core) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0,
                "ddr_scatter_spm_dma: per-core bytes (", bytes,
                ") must be 16-byte aligned");
    TORCH_CHECK((core_stride_bytes % 16) == 0,
                "ddr_scatter_spm_dma: core_stride_bytes (", core_stride_bytes,
                ") must be 16-byte aligned");
    TORCH_CHECK(bytes <= (8ULL << 20),
                "ddr_scatter_spm_dma: single-DMA cap is 8 MB, got ", bytes);

    // PASSTHROUGH fallback — graph scope 不 active 时退到 immediate 路径。
    if (!graph_dma::active()) {
        rpu_launch_ddr_scatter_spm_dma_immediate(
            ddr_base, elements_per_core, core_stride_bytes,
            spm_addr_unified, num_cores);
        return;
    }

    rpu_ddr_flush(ddr_base);
    const uint64_t ddr_abs = RpuGetDevAddr(ddr_base);

    emit_scatter_pre_barriers(num_cores);
    for (int c = 0; c < num_cores; ++c) {
        graph_dma::emit_fixed(ddr_abs + static_cast<uint64_t>(c) *
                                       static_cast<uint64_t>(core_stride_bytes),
                          spm_unified_to_per_core_abs(c, spm_addr_unified),
                          bytes, c);
    }
    emit_scatter_post_barriers(num_cores);
}

void rpu_launch_ddr_scatter_spm_dma(
        const at::Tensor& ddr_tensor,
        int64_t src_offset_elements,
        int64_t elements_per_core,
        int64_t core_stride_bytes,
        uint32_t spm_addr_unified,
        int num_cores) {
    TORCH_CHECK(src_offset_elements >= 0 && elements_per_core > 0 &&
                    core_stride_bytes >= 0 && num_cores > 0,
                "ddr_scatter_spm_dma owned: invalid owner range");
    const size_t byte_begin =
        static_cast<size_t>(src_offset_elements) * sizeof(c10::Half);
    const size_t per_core_bytes =
        static_cast<size_t>(elements_per_core) * sizeof(c10::Half);
    const size_t byte_count =
        static_cast<size_t>(num_cores - 1) *
            static_cast<size_t>(core_stride_bytes) +
        per_core_bytes;
    validate_fp16_dma_owner_slice(
        ddr_tensor, byte_begin, byte_count,
        "ddr_scatter_spm_dma owned");
    if (graph_dma::active()) {
        graph_dma::stage_census_fixed_burst(
            ddr_tensor, GraphOuterFastDmaSide::Source,
            byte_begin, byte_count, static_cast<size_t>(num_cores));
    }
    rpu_launch_ddr_scatter_spm_dma(
        ddr_tensor.data_ptr<c10::Half>() + src_offset_elements,
        elements_per_core, core_stride_bytes, spm_addr_unified, num_cores);
}

void rpu_launch_spm_scatter_spm_dma(uint32_t  spm_src_unified,
                                     int64_t   elements_per_core,
                                     int64_t   core_stride_bytes,
                                     uint32_t  spm_dst_unified,
                                     int       num_cores) {
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "spm_scatter_spm_dma: num_cores 1..8, got ", num_cores);
    TORCH_CHECK(elements_per_core > 0,
                "spm_scatter_spm_dma: elements_per_core > 0, got ",
                elements_per_core);
    const size_t bytes =
        static_cast<size_t>(elements_per_core) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0 && (core_stride_bytes % 16) == 0,
                "spm_scatter_spm_dma: byte counts must be 16-byte aligned");

    const uint64_t src0 =
        spm_unified_to_per_core_abs(/*core=*/0, spm_src_unified);
    if (!graph_dma::active()) {
        RpuKernelGraph::check_direct_immediate_dma_allowed(
            "spm_scatter_spm_dma");
        Queue_t* wq = QueueCache::instance().get(static_cast<uint8_t>(num_cores));
        for (int ch = 1; ch < num_cores; ++ch)
            wq->add_barrier(static_cast<uint8_t>(ch * 2), 0);
        for (int c = 0; c < num_cores; ++c) {
            const uint64_t src =
                src0 + static_cast<uint64_t>(c) * core_stride_bytes;
            const uint64_t dst =
                spm_unified_to_per_core_abs(c, spm_dst_unified);
            rpu_add_dma_checked(
                *wq, src, dst, bytes, c, "spm_scatter_spm_dma");
        }
        for (int ch = 1; ch < num_cores; ++ch)
            wq->add_barrier(0, static_cast<uint8_t>(ch * 2));
        submit_immediate_batch(wq, "spm_scatter_spm_dma");
        return;
    }

    emit_scatter_pre_barriers(num_cores);
    for (int c = 0; c < num_cores; ++c) {
        const uint64_t src =
            src0 + static_cast<uint64_t>(c) * core_stride_bytes;
        const uint64_t dst =
            spm_unified_to_per_core_abs(c, spm_dst_unified);
        graph_dma::emit_fixed(src, dst, bytes, c);
    }
    emit_scatter_post_barriers(num_cores);
}

void rpu_launch_spm_local_shard_copy_dma(uint32_t spm_src_unified,
                                          int64_t elements_per_core,
                                          int64_t core_stride_bytes,
                                          uint32_t spm_dst_unified,
                                          int num_cores) {
    constexpr const char* caller = "spm_local_shard_copy_dma";
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                caller, ": num_cores must be in [1,8], got ", num_cores);
    TORCH_CHECK(elements_per_core > 0 && core_stride_bytes >= 0,
                caller, ": element count must be positive and stride non-negative");
    const size_t bytes =
        static_cast<size_t>(elements_per_core) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0 && (core_stride_bytes % 16) == 0,
                caller, ": byte count and stride must be 16-byte aligned");
    TORCH_CHECK(core_stride_bytes == 0
                    || static_cast<uint64_t>(core_stride_bytes) >= bytes,
                caller, ": non-zero stride must cover one shard");

    const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
    TORCH_CHECK(spm_src_unified >= spm_base && spm_dst_unified >= spm_base,
                caller, ": source and destination must be core-0 unified SPM addresses");
    const uint64_t src_off = spm_src_unified - spm_base;
    const uint64_t dst_off = spm_dst_unified - spm_base;
    const uint64_t last_src_end = src_off
        + static_cast<uint64_t>(num_cores - 1) * core_stride_bytes + bytes;
    TORCH_CHECK(last_src_end <= SpmAllocator::SPM_USABLE
                    && dst_off + bytes <= SpmAllocator::SPM_USABLE,
                caller, ": source or destination exceeds per-core SPM");
    for (int c = 0; c < num_cores; ++c) {
        const uint64_t src_begin =
            src_off + static_cast<uint64_t>(c) * core_stride_bytes;
        TORCH_CHECK(src_begin + bytes <= dst_off
                        || dst_off + bytes <= src_begin,
                    caller, ": overlapping same-core DMA is unsupported");
    }

    auto emit = [&](auto&& add_dma) {
        for (int c = 0; c < num_cores; ++c) {
            const uint64_t src = SPM_ALLOC.addr(
                c, static_cast<uint32_t>(
                    src_off + static_cast<uint64_t>(c) * core_stride_bytes));
            const uint64_t dst =
                SPM_ALLOC.addr(c, static_cast<uint32_t>(dst_off));
            add_dma(c, src, dst);
        }
    };

    if (!graph_dma::active()) {
        RpuKernelGraph::check_direct_immediate_dma_allowed(caller);
        Queue_t* wq = QueueCache::instance().get(static_cast<uint8_t>(num_cores));
        for (int ch = 1; ch < num_cores; ++ch)
            wq->add_barrier(static_cast<uint8_t>(ch * 2), 0);
        emit([&](int channel, uint64_t src, uint64_t dst) {
            rpu_add_dma_checked(*wq, src, dst, bytes, channel, caller);
        });
        for (int ch = 1; ch < num_cores; ++ch)
            wq->add_barrier(0, static_cast<uint8_t>(ch * 2));
        submit_immediate_batch(wq, caller);
        return;
    }

    emit_scatter_pre_barriers(num_cores);
    emit([&](int channel, uint64_t src, uint64_t dst) {
        graph_dma::emit_fixed(src, dst, bytes, channel);
    });
    emit_scatter_post_barriers(num_cores);
}

void rpu_launch_spm_local_copy_dma(uint32_t spm_src_unified,
                                    int64_t elements_per_core,
                                    uint32_t spm_dst_unified,
                                    int num_cores) {
    rpu_launch_spm_local_shard_copy_dma(
        spm_src_unified, elements_per_core, /*core_stride_bytes=*/0,
        spm_dst_unified, num_cores);
}

void rpu_launch_spm_gather_spm_dma(uint32_t spm_src_unified,
                                    int64_t elements_per_core,
                                    uint32_t spm_dst_unified,
                                    int num_cores) {
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "spm_gather_spm_dma: num_cores 1..8, got ", num_cores);
    TORCH_CHECK(elements_per_core > 0,
                "spm_gather_spm_dma: elements_per_core > 0, got ",
                elements_per_core);
    const size_t bytes =
        static_cast<size_t>(elements_per_core) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0,
                "spm_gather_spm_dma: byte count must be 16-byte aligned");

    const uint64_t dst0 =
        spm_unified_to_per_core_abs(/*core=*/0, spm_dst_unified);
    if (!graph_dma::active()) {
        RpuKernelGraph::check_direct_immediate_dma_allowed(
            "spm_gather_spm_dma");
        Queue_t* wq = QueueCache::instance().get(static_cast<uint8_t>(num_cores));
        for (int ch = 1; ch < num_cores; ++ch)
            wq->add_barrier(static_cast<uint8_t>(ch * 2), 0);
        for (int c = 0; c < num_cores; ++c) {
            const uint64_t src =
                spm_unified_to_per_core_abs(c, spm_src_unified);
            const uint64_t dst = dst0 + static_cast<uint64_t>(c) * bytes;
            rpu_add_dma_checked(
                *wq, src, dst, bytes, c, "spm_gather_spm_dma");
        }
        for (int ch = 1; ch < num_cores; ++ch)
            wq->add_barrier(0, static_cast<uint8_t>(ch * 2));
        submit_immediate_batch(wq, "spm_gather_spm_dma");
        return;
    }

    emit_scatter_pre_barriers(num_cores);
    for (int c = 0; c < num_cores; ++c) {
        const uint64_t src =
            spm_unified_to_per_core_abs(c, spm_src_unified);
        const uint64_t dst = dst0 + static_cast<uint64_t>(c) * bytes;
        graph_dma::emit_fixed(src, dst, bytes, c);
    }
    emit_scatter_post_barriers(num_cores);
}

void rpu_launch_ddr_scatter_spm_dma_mutable(const uint64_t* live_src_base,
                                             int64_t         src_offset_bytes,
                                             int64_t         elements_per_core,
                                             int64_t         core_stride_bytes,
                                             uint32_t        spm_addr_unified,
                                             int             num_cores) {
    TORCH_CHECK(live_src_base != nullptr,
                "ddr_scatter_spm_dma_mutable: null live_src_base");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "ddr_scatter_spm_dma_mutable: num_cores 1..8, got ", num_cores);
    TORCH_CHECK(elements_per_core > 0,
                "ddr_scatter_spm_dma_mutable: elements_per_core > 0, got ",
                elements_per_core);

    const size_t bytes =
        static_cast<size_t>(elements_per_core) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0,
                "ddr_scatter_spm_dma_mutable: per-core bytes (", bytes,
                ") must be 16-byte aligned");
    TORCH_CHECK((core_stride_bytes % 16) == 0,
                "ddr_scatter_spm_dma_mutable: core_stride_bytes (",
                core_stride_bytes, ") must be 16-byte aligned");
    TORCH_CHECK(bytes <= (8ULL << 20),
                "ddr_scatter_spm_dma_mutable: single-DMA cap is 8 MB, got ",
                bytes);

    // PASSTHROUGH fallback — deref live_src_base 一次,直接发 immediate batch。
    if (!graph_dma::active()) {
        RpuKernelGraph::check_direct_immediate_dma_allowed(
            "ddr_scatter_spm_dma_mutable");
        const uint64_t base = *live_src_base;
        Queue_t* wq = QueueCache::instance().get(static_cast<uint8_t>(num_cores));
        for (int ch = 1; ch < num_cores; ++ch)
            wq->add_barrier(static_cast<uint8_t>(ch * 2), 0);
        for (int c = 0; c < num_cores; ++c) {
            const uint64_t src = base + static_cast<uint64_t>(
                src_offset_bytes + static_cast<int64_t>(c) * core_stride_bytes);
            rpu_add_dma_checked(*wq, src,
                                spm_unified_to_per_core_abs(c, spm_addr_unified),
                                bytes, c, "ddr_scatter_spm_dma_mutable");
        }
        for (int ch = 1; ch < num_cores; ++ch)
            wq->add_barrier(0, static_cast<uint8_t>(ch * 2));
        submit_immediate_batch(wq, "ddr_scatter_spm_dma_mutable");
        return;
    }

    // No rpu_ddr_flush here — caller flushes the live buffer per forward.
    emit_scatter_pre_barriers(num_cores);
    for (int c = 0; c < num_cores; ++c) {
        graph_dma::emit_mutable_src(
            live_src_base,
            src_offset_bytes +
                static_cast<int64_t>(c) * core_stride_bytes,
            spm_unified_to_per_core_abs(c, spm_addr_unified),
            bytes, c);
    }
    emit_scatter_post_barriers(num_cores);
}

void rpu_launch_ddr_scatter_spm_dma_mutable(
        uint64_t& live_src_base,
        const at::Tensor& owner_tensor,
        int64_t src_offset_bytes,
        int64_t elements_per_core,
        int64_t core_stride_bytes,
        uint32_t spm_addr_unified,
        int num_cores) {
    TORCH_CHECK(src_offset_bytes >= 0 && elements_per_core > 0 &&
                    core_stride_bytes >= 0 && num_cores > 0,
                "ddr_scatter_spm_dma_mutable owned: invalid owner range");
    const size_t byte_begin = static_cast<size_t>(src_offset_bytes);
    const size_t per_core_bytes =
        static_cast<size_t>(elements_per_core) * sizeof(c10::Half);
    const size_t byte_count =
        static_cast<size_t>(num_cores - 1) *
            static_cast<size_t>(core_stride_bytes) +
        per_core_bytes;
    validate_fp16_dma_owner_slice(
        owner_tensor, byte_begin, byte_count,
        "ddr_scatter_spm_dma_mutable owned");
    if (graph_dma::active()) {
        graph_dma::stage_census_mutable_burst(
            live_src_base, owner_tensor, GraphOuterFastDmaSide::Source,
            byte_begin, byte_count, static_cast<size_t>(num_cores));
    }
    rpu_launch_ddr_scatter_spm_dma_mutable(
        &live_src_base, src_offset_bytes, elements_per_core,
        core_stride_bytes, spm_addr_unified, num_cores);
}

void rpu_launch_spm_scatter_ddr_dma(uint32_t  spm_addr_unified,
                                     c10::Half* ddr_base,
                                     int64_t   elements_per_core,
                                     int64_t   core_stride_bytes,
                                     int       num_cores) {
    TORCH_CHECK(ddr_base != nullptr, "spm_scatter_ddr_dma: null ddr_base");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "spm_scatter_ddr_dma: num_cores 1..8, got ", num_cores);
    TORCH_CHECK(elements_per_core > 0,
                "spm_scatter_ddr_dma: elements_per_core > 0, got ",
                elements_per_core);

    const size_t bytes =
        static_cast<size_t>(elements_per_core) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0,
                "spm_scatter_ddr_dma: per-core bytes (", bytes,
                ") must be 16-byte aligned");
    TORCH_CHECK((core_stride_bytes % 16) == 0,
                "spm_scatter_ddr_dma: core_stride_bytes (", core_stride_bytes,
                ") must be 16-byte aligned");
    TORCH_CHECK(bytes <= (8ULL << 20),
                "spm_scatter_ddr_dma: single-DMA cap is 8 MB, got ", bytes);

    // PASSTHROUGH fallback — graph scope 不 active 时退到 immediate 路径。
    if (!graph_dma::active()) {
        rpu_launch_spm_scatter_ddr_dma_immediate(
            spm_addr_unified, ddr_base, elements_per_core,
            core_stride_bytes, num_cores);
        return;
    }

    rpu_ddr_flush(ddr_base);
    const uint64_t ddr_abs = RpuGetDevAddr(ddr_base);

    emit_scatter_pre_barriers(num_cores);
    for (int c = 0; c < num_cores; ++c) {
        graph_dma::emit_fixed(spm_unified_to_per_core_abs(c, spm_addr_unified),
                          ddr_abs + static_cast<uint64_t>(c) *
                                       static_cast<uint64_t>(core_stride_bytes),
                          bytes, c);
    }
    emit_scatter_post_barriers(num_cores);
}

void rpu_launch_spm_scatter_ddr_dma(
        uint32_t spm_addr_unified,
        const at::Tensor& ddr_tensor,
        int64_t dst_offset_elements,
        int64_t elements_per_core,
        int64_t core_stride_bytes,
        int num_cores) {
    TORCH_CHECK(dst_offset_elements >= 0 && elements_per_core > 0 &&
                    core_stride_bytes >= 0 && num_cores >= 1 &&
                    num_cores <= 8,
                "spm_scatter_ddr_dma owned: invalid owner range");
    constexpr const char* caller = "spm_scatter_ddr_dma owned";
    const size_t byte_begin = checked_size_t_multiply(
        static_cast<uint64_t>(dst_offset_elements), sizeof(c10::Half),
        caller, "byte_begin");
    const size_t per_core_bytes = checked_size_t_multiply(
        static_cast<uint64_t>(elements_per_core), sizeof(c10::Half),
        caller, "per_core_bytes");
    TORCH_CHECK((per_core_bytes % 16) == 0 &&
                    (core_stride_bytes % 16) == 0 &&
                    per_core_bytes <= (8ULL << 20),
                "spm_scatter_ddr_dma owned: DMA bytes/stride are invalid");
    const size_t stride_span = checked_size_t_multiply(
        static_cast<uint64_t>(num_cores - 1),
        static_cast<uint64_t>(core_stride_bytes), caller,
        "byte_count stride span");
    const size_t byte_count = checked_size_t_add(
        stride_span, per_core_bytes, caller, "byte_count");
    validate_fp16_dma_owner_slice(
        ddr_tensor, byte_begin, byte_count, caller);
    if (graph_dma::active()) {
        graph_dma::stage_census_fixed_burst(
            ddr_tensor, GraphOuterFastDmaSide::Destination,
            byte_begin, byte_count, static_cast<size_t>(num_cores));
    }
    rpu_launch_spm_scatter_ddr_dma(
        spm_addr_unified,
        ddr_tensor.data_ptr<c10::Half>() + dst_offset_elements,
        elements_per_core, core_stride_bytes, num_cores);
}

void rpu_launch_spm_scatter_ddr_dma_mutable(uint32_t        spm_addr_unified,
                                             const uint64_t* live_dst_base,
                                             int64_t         dst_offset_bytes,
                                             int64_t         elements_per_core,
                                             int64_t         core_stride_bytes,
                                             int             num_cores) {
    TORCH_CHECK(live_dst_base != nullptr,
                "spm_scatter_ddr_dma_mutable: null live_dst_base");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "spm_scatter_ddr_dma_mutable: num_cores 1..8, got ", num_cores);
    TORCH_CHECK(elements_per_core > 0,
                "spm_scatter_ddr_dma_mutable: elements_per_core > 0, got ",
                elements_per_core);

    const size_t bytes =
        static_cast<size_t>(elements_per_core) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0,
                "spm_scatter_ddr_dma_mutable: per-core bytes (", bytes,
                ") must be 16-byte aligned");
    TORCH_CHECK((core_stride_bytes % 16) == 0,
                "spm_scatter_ddr_dma_mutable: core_stride_bytes (",
                core_stride_bytes, ") must be 16-byte aligned");
    TORCH_CHECK(bytes <= (8ULL << 20),
                "spm_scatter_ddr_dma_mutable: single-DMA cap is 8 MB, got ",
                bytes);

    // PASSTHROUGH fallback — deref live_dst_base 一次,直接发 immediate batch。
    if (!graph_dma::active()) {
        RpuKernelGraph::check_direct_immediate_dma_allowed(
            "spm_scatter_ddr_dma_mutable");
        const uint64_t base = *live_dst_base;
        Queue_t* wq = QueueCache::instance().get(static_cast<uint8_t>(num_cores));
        for (int ch = 1; ch < num_cores; ++ch)
            wq->add_barrier(static_cast<uint8_t>(ch * 2), 0);
        for (int c = 0; c < num_cores; ++c) {
            const uint64_t dst = base + static_cast<uint64_t>(
                dst_offset_bytes + static_cast<int64_t>(c) * core_stride_bytes);
            rpu_add_dma_checked(*wq,
                                spm_unified_to_per_core_abs(c, spm_addr_unified),
                                dst, bytes, c, "spm_scatter_ddr_dma_mutable");
        }
        for (int ch = 1; ch < num_cores; ++ch)
            wq->add_barrier(0, static_cast<uint8_t>(ch * 2));
        submit_immediate_batch(wq, "spm_scatter_ddr_dma_mutable");
        return;
    }

    // No rpu_ddr_flush — caller is responsible for live dst buffer flushes.
    emit_scatter_pre_barriers(num_cores);
    for (int c = 0; c < num_cores; ++c) {
        graph_dma::emit_mutable_dst(
            spm_unified_to_per_core_abs(c, spm_addr_unified),
            live_dst_base,
            dst_offset_bytes +
                static_cast<int64_t>(c) * core_stride_bytes,
            bytes, c);
    }
    emit_scatter_post_barriers(num_cores);
}

void rpu_launch_spm_copy_ddr_dma(uint32_t spm_addr_unified,
                                  c10::Half* ddr_ptr,
                                  int64_t num_elements) {
    TORCH_CHECK(ddr_ptr != nullptr, "spm_copy_ddr_dma: null ddr_ptr");
    TORCH_CHECK(num_elements > 0,
                "spm_copy_ddr_dma: num_elements > 0, got ", num_elements);

    const size_t bytes = static_cast<size_t>(num_elements) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0,
                "spm_copy_ddr_dma: bytes (", bytes, ") must be 16-byte aligned");
    TORCH_CHECK(bytes <= (8ULL << 20),
                "spm_copy_ddr_dma: single-DMA cap is 8 MB, got ", bytes, " bytes");

    // PASSTHROUGH fallback — graph scope 不 active 时退到 immediate 路径。
    if (!graph_dma::active()) {
        rpu_launch_spm_copy_ddr_dma_immediate(
            spm_addr_unified, ddr_ptr, num_elements);
        return;
    }

    rpu_ddr_flush(ddr_ptr);
    const uint64_t ddr_abs = RpuGetDevAddr(ddr_ptr);

    // 1 DMA on channel 0 (stream 0): same stream as up/down-stream compute,
    // no barriers needed.
    graph_dma::emit_fixed(spm_unified_to_per_core_abs(/*core=*/0, spm_addr_unified),
                      ddr_abs, bytes, /*ch=*/0);
}

void rpu_launch_spm_copy_ddr_dma(
        uint32_t spm_addr_unified,
        const at::Tensor& ddr_tensor,
        int64_t dst_offset_elements,
        int64_t num_elements) {
    TORCH_CHECK(dst_offset_elements >= 0 && num_elements > 0,
                "spm_copy_ddr_dma owned: invalid element range");
    const size_t byte_begin =
        static_cast<size_t>(dst_offset_elements) * sizeof(c10::Half);
    const size_t byte_count =
        static_cast<size_t>(num_elements) * sizeof(c10::Half);
    validate_fp16_dma_owner_slice(
        ddr_tensor, byte_begin, byte_count, "spm_copy_ddr_dma owned");
    if (graph_dma::active()) {
        graph_dma::stage_census_fixed_burst(
            ddr_tensor, GraphOuterFastDmaSide::Destination,
            byte_begin, byte_count, /*expected_occurrence_count=*/1);
    }
    rpu_launch_spm_copy_ddr_dma(
        spm_addr_unified,
        ddr_tensor.data_ptr<c10::Half>() + dst_offset_elements,
        num_elements);
}

void rpu_launch_spm_copy_ddr_dma_mutable(uint32_t        spm_addr_unified,
                                          const uint64_t* live_dst_base,
                                          int64_t         dst_offset_bytes,
                                          int64_t         num_elements) {
    TORCH_CHECK(live_dst_base != nullptr,
                "spm_copy_ddr_dma_mutable: null live_dst_base");
    TORCH_CHECK(num_elements > 0,
                "spm_copy_ddr_dma_mutable: num_elements > 0, got ",
                num_elements);

    const size_t bytes = static_cast<size_t>(num_elements) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0,
                "spm_copy_ddr_dma_mutable: bytes (", bytes,
                ") must be 16-byte aligned");
    TORCH_CHECK(bytes <= (8ULL << 20),
                "spm_copy_ddr_dma_mutable: single-DMA cap is 8 MB, got ",
                bytes);

    // PASSTHROUGH fallback — deref live_dst_base 一次,直接发 immediate batch。
    if (!graph_dma::active()) {
        RpuKernelGraph::check_direct_immediate_dma_allowed(
            "spm_copy_ddr_dma_mutable");
        const uint64_t dst = (*live_dst_base) +
                             static_cast<uint64_t>(dst_offset_bytes);
        Queue_t* wq = QueueCache::instance().get(/*core_num=*/1);
        rpu_add_dma_checked(*wq,
                            spm_unified_to_per_core_abs(/*core=*/0, spm_addr_unified),
                            dst, bytes, /*ch=*/0, "spm_copy_ddr_dma_mutable");
        submit_immediate_batch(wq, "spm_copy_ddr_dma_mutable");
        return;
    }

    // No rpu_ddr_flush — caller is responsible for live dst buffer flushes.
    // Single channel 0 DMA, no barriers needed.
    graph_dma::emit_mutable_dst(
        spm_unified_to_per_core_abs(/*core=*/0, spm_addr_unified),
        live_dst_base, dst_offset_bytes, bytes, /*ch=*/0);
}

void rpu_launch_spm_copy_ddr_dma_mutable(
        uint32_t spm_addr_unified,
        uint64_t& live_dst_base,
        const at::Tensor& owner_tensor,
        int64_t dst_offset_bytes,
        int64_t num_elements) {
    TORCH_CHECK(dst_offset_bytes >= 0 && num_elements > 0,
                "spm_copy_ddr_dma_mutable owned: invalid byte range");
    const size_t byte_begin = static_cast<size_t>(dst_offset_bytes);
    const size_t byte_count =
        static_cast<size_t>(num_elements) * sizeof(c10::Half);
    validate_fp16_dma_owner_slice(
        owner_tensor, byte_begin, byte_count,
        "spm_copy_ddr_dma_mutable owned");
    if (graph_dma::active()) {
        graph_dma::stage_census_mutable_burst(
            live_dst_base, owner_tensor, GraphOuterFastDmaSide::Destination,
            byte_begin, byte_count, /*expected_occurrence_count=*/1);
    }
    rpu_launch_spm_copy_ddr_dma_mutable(
        spm_addr_unified, &live_dst_base, dst_offset_bytes, num_elements);
}

// =============================================================================
// DMA-based memcpy primitives (immediate / graph-outside variants)
// =============================================================================
// One-shot DMA batches for calls outside a normal active Graph scope, plus the
// plain PASSTHROUGH compatibility case described below.
//
// MUST pass RpuKernelGraph::check_direct_immediate_dma_allowed before any data
// or SDK side effect.  A normal RECORDING / REPLAYING or nested scope would
// execute synchronously out of band instead of recording an ordered DMA node.
// The sole compatibility exception is one plain scope already downgraded to
// PASSTHROUGH by the raw-kernel fallback.
//
// Queue source: QueueCache::instance().get(num_cores) — the process-wide
// per-core_num one-shot pool also used by execute_graph_oneshot (one Queue_t
// per core_num, max 8). PASSTHROUGH kernels use a separate direct-launch queue;
// cacheable GraphCache BUILD / REPLAY uses each entry's private queue.
// submit_immediate_batch consumes the completed one-shot batch state before
// returning, so a later one-shot user never observes a stale prepared batch.

void rpu_launch_ddr_broadcast_spm_dma_immediate(c10::Half* ddr_ptr,
                                                 int64_t num_elements,
                                                 uint32_t spm_addr_unified,
                                                 int num_cores) {
    RpuKernelGraph::check_direct_immediate_dma_allowed(
        "ddr_broadcast_spm_dma_immediate");
    TORCH_CHECK(ddr_ptr != nullptr,
                "ddr_broadcast_spm_dma_immediate: null ddr_ptr");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "ddr_broadcast_spm_dma_immediate: num_cores 1..8, got ",
                num_cores);
    TORCH_CHECK(num_elements > 0,
                "ddr_broadcast_spm_dma_immediate: num_elements > 0, got ",
                num_elements);

    const size_t bytes = static_cast<size_t>(num_elements) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0,
                "ddr_broadcast_spm_dma_immediate: bytes (", bytes,
                ") must be 16-byte aligned");
    TORCH_CHECK(bytes <= (8ULL << 20),
                "ddr_broadcast_spm_dma_immediate: single-DMA cap is 8 MB, "
                "got ", bytes, " bytes (auto-split is future work)");

    rpu_ddr_flush(ddr_ptr);
    const uint64_t ddr_abs = RpuGetDevAddr(ddr_ptr);

    Queue_t* wq = QueueCache::instance().get(static_cast<uint8_t>(num_cores));

    // Pre-barriers: ch=1..nc-1 streams wait for stream 0.
    for (int ch = 1; ch < num_cores; ++ch)
        wq->add_barrier(static_cast<uint8_t>(ch * 2), 0);

    // num_cores DMAs: same DDR src, per-core SPM dst.
    for (int c = 0; c < num_cores; ++c)
        rpu_add_dma_checked(*wq, ddr_abs,
                            spm_unified_to_per_core_abs(c, spm_addr_unified),
                            bytes, c, "ddr_broadcast_spm_dma_immediate");

    // Post-barriers: stream 0 waits for ch=1..nc-1.
    for (int ch = 1; ch < num_cores; ++ch)
        wq->add_barrier(0, static_cast<uint8_t>(ch * 2));

    submit_immediate_batch(wq, "ddr_broadcast_spm_dma_immediate");
}

void rpu_launch_spm_copy_ddr_dma_immediate(uint32_t spm_addr_unified,
                                            c10::Half* ddr_ptr,
                                            int64_t num_elements) {
    RpuKernelGraph::check_direct_immediate_dma_allowed(
        "spm_copy_ddr_dma_immediate");
    TORCH_CHECK(ddr_ptr != nullptr,
                "spm_copy_ddr_dma_immediate: null ddr_ptr");
    TORCH_CHECK(num_elements > 0,
                "spm_copy_ddr_dma_immediate: num_elements > 0, got ",
                num_elements);

    const size_t bytes = static_cast<size_t>(num_elements) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0,
                "spm_copy_ddr_dma_immediate: bytes (", bytes,
                ") must be 16-byte aligned");
    TORCH_CHECK(bytes <= (8ULL << 20),
                "spm_copy_ddr_dma_immediate: single-DMA cap is 8 MB, got ",
                bytes, " bytes");

    rpu_ddr_flush(ddr_ptr);
    const uint64_t ddr_abs = RpuGetDevAddr(ddr_ptr);

    // Single-core read: 1 DMA on channel 0, no barriers needed.
    Queue_t* wq = QueueCache::instance().get(/*core_num=*/1);
    rpu_add_dma_checked(*wq,
                        spm_unified_to_per_core_abs(/*core=*/0, spm_addr_unified),
                        ddr_abs, bytes, /*ch=*/0, "spm_copy_ddr_dma_immediate");

    submit_immediate_batch(wq, "spm_copy_ddr_dma_immediate");

    // Post-flush so subsequent host reads see DMA-written data.
    rpu_ddr_flush(ddr_ptr);
}

// Byte-granular DDR → core-0 SPM copy (immediate). The Half-typed broadcast helper
// scales num_elements by sizeof(Half); this one takes a raw byte count, needed for
// non-fp16 staging like the uint8 input of unary_cast_uint8_fp16. Single core, no
// barriers (like spm_copy_ddr). DMA still requires a 16-byte-aligned byte count.
void rpu_launch_ddr_spm_bytes_immediate(void* ddr_ptr,
                                        int64_t num_bytes,
                                        uint32_t spm_addr_unified) {
    RpuKernelGraph::check_direct_immediate_dma_allowed(
        "ddr_spm_bytes_immediate");
    TORCH_CHECK(ddr_ptr != nullptr, "ddr_spm_bytes_immediate: null ddr_ptr");
    TORCH_CHECK(num_bytes > 0,
                "ddr_spm_bytes_immediate: num_bytes > 0, got ", num_bytes);
    TORCH_CHECK((num_bytes % 16) == 0,
                "ddr_spm_bytes_immediate: bytes (", num_bytes,
                ") must be 16-byte aligned");
    TORCH_CHECK(num_bytes <= (8LL << 20),
                "ddr_spm_bytes_immediate: single-DMA cap is 8 MB, got ", num_bytes);

    rpu_ddr_flush(ddr_ptr);
    const uint64_t ddr_abs = RpuGetDevAddr(ddr_ptr);

    Queue_t* wq = QueueCache::instance().get(/*core_num=*/1);
    rpu_add_dma_checked(*wq, ddr_abs,
                        spm_unified_to_per_core_abs(/*core=*/0, spm_addr_unified),
                        static_cast<size_t>(num_bytes), /*ch=*/0,
                        "ddr_spm_bytes_immediate");
    submit_immediate_batch(wq, "ddr_spm_bytes_immediate");
}

void rpu_launch_ddr_scatter_spm_dma_immediate(c10::Half* ddr_base,
                                               int64_t   elements_per_core,
                                               int64_t   core_stride_bytes,
                                               uint32_t  spm_addr_unified,
                                               int       num_cores) {
    RpuKernelGraph::check_direct_immediate_dma_allowed(
        "ddr_scatter_spm_dma_immediate");
    TORCH_CHECK(ddr_base != nullptr,
                "ddr_scatter_spm_dma_immediate: null ddr_base");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "ddr_scatter_spm_dma_immediate: num_cores 1..8, got ",
                num_cores);
    TORCH_CHECK(elements_per_core > 0,
                "ddr_scatter_spm_dma_immediate: elements_per_core > 0, got ",
                elements_per_core);

    const size_t bytes =
        static_cast<size_t>(elements_per_core) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0,
                "ddr_scatter_spm_dma_immediate: per-core bytes (", bytes,
                ") must be 16-byte aligned");
    TORCH_CHECK((core_stride_bytes % 16) == 0,
                "ddr_scatter_spm_dma_immediate: core_stride_bytes (",
                core_stride_bytes, ") must be 16-byte aligned");
    TORCH_CHECK(bytes <= (8ULL << 20),
                "ddr_scatter_spm_dma_immediate: single-DMA cap is 8 MB, got ",
                bytes);

    rpu_ddr_flush(ddr_base);
    const uint64_t ddr_abs = RpuGetDevAddr(ddr_base);

    Queue_t* wq = QueueCache::instance().get(static_cast<uint8_t>(num_cores));

    // Pre-barriers: ch=1..nc-1 streams wait for stream 0.
    for (int ch = 1; ch < num_cores; ++ch)
        wq->add_barrier(static_cast<uint8_t>(ch * 2), 0);

    // Per-core DMAs: each core c reads ddr_base + c*stride, writes to its own SPM.
    for (int c = 0; c < num_cores; ++c)
        rpu_add_dma_checked(*wq, ddr_abs + static_cast<uint64_t>(c) *
                                       static_cast<uint64_t>(core_stride_bytes),
                            spm_unified_to_per_core_abs(c, spm_addr_unified),
                            bytes, c, "ddr_scatter_spm_dma_immediate");

    // Post-barriers: stream 0 waits for ch=1..nc-1.
    for (int ch = 1; ch < num_cores; ++ch)
        wq->add_barrier(0, static_cast<uint8_t>(ch * 2));

    submit_immediate_batch(wq, "ddr_scatter_spm_dma_immediate");
}

void rpu_launch_spm_scatter_ddr_dma_immediate(uint32_t  spm_addr_unified,
                                               c10::Half* ddr_base,
                                               int64_t   elements_per_core,
                                               int64_t   core_stride_bytes,
                                               int       num_cores) {
    RpuKernelGraph::check_direct_immediate_dma_allowed(
        "spm_scatter_ddr_dma_immediate");
    TORCH_CHECK(ddr_base != nullptr,
                "spm_scatter_ddr_dma_immediate: null ddr_base");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "spm_scatter_ddr_dma_immediate: num_cores 1..8, got ",
                num_cores);
    TORCH_CHECK(elements_per_core > 0,
                "spm_scatter_ddr_dma_immediate: elements_per_core > 0, got ",
                elements_per_core);

    const size_t bytes =
        static_cast<size_t>(elements_per_core) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0,
                "spm_scatter_ddr_dma_immediate: per-core bytes (", bytes,
                ") must be 16-byte aligned");
    TORCH_CHECK((core_stride_bytes % 16) == 0,
                "spm_scatter_ddr_dma_immediate: core_stride_bytes (",
                core_stride_bytes, ") must be 16-byte aligned");
    TORCH_CHECK(bytes <= (8ULL << 20),
                "spm_scatter_ddr_dma_immediate: single-DMA cap is 8 MB, got ",
                bytes);

    rpu_ddr_flush(ddr_base);
    const uint64_t ddr_abs = RpuGetDevAddr(ddr_base);

    Queue_t* wq = QueueCache::instance().get(static_cast<uint8_t>(num_cores));

    for (int ch = 1; ch < num_cores; ++ch)
        wq->add_barrier(static_cast<uint8_t>(ch * 2), 0);

    // Per-core DMAs: each core c reads its own SPM, writes to ddr_base + c*stride.
    for (int c = 0; c < num_cores; ++c)
        rpu_add_dma_checked(*wq,
                            spm_unified_to_per_core_abs(c, spm_addr_unified),
                            ddr_abs + static_cast<uint64_t>(c) *
                                       static_cast<uint64_t>(core_stride_bytes),
                            bytes, c, "spm_scatter_ddr_dma_immediate");

    for (int ch = 1; ch < num_cores; ++ch)
        wq->add_barrier(0, static_cast<uint8_t>(ch * 2));

    submit_immediate_batch(wq, "spm_scatter_ddr_dma_immediate");

    // Post-flush so subsequent host reads see scatter-written data.
    rpu_ddr_flush(ddr_base);
}
