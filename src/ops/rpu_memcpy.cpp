// rpu_memcpy.cpp - DDR to SPM memory copy for device backend (multi-core)
//
// 功能: 将 DDR 上的 fp16 tensor 拷贝到 8 个 core 的 SPM 上
// 每个 core 获得相同的数据副本
//

#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"  // for SPM_ALLOC (LOCAL-mode offset extraction)
#include <c10/util/Half.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
#include <unistd.h> // for usleep


using namespace ::rhino_lkn;

// =============================================================================
// DDR to SPM Multi-Core Kernel Launcher
// =============================================================================
//
// 参数约定:
//   param0/param1   : 元素数量
//   param2/param3   : 步进值 (v16 数量)
//   param4/param5   : DDR 源地址 (32-bit, 已右移8位, 256B粒度)
//   param6/param7   : SPM 目标地址 (32-bit, 字节地址)
//   param10/param11 : block_stride (32-bit) - 每个 block 的偏移量
//   param12/param13 : core_stride (32-bit) - 每个 core 的偏移量
//
// 多核行为:
//   设置 core_stride = 0，所有 core 从同一个 DDR 地址读取
//   写入各自的 local SPM (相同的 local 地址)
//
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
// Store-count split for the v2 transfer contract
// =============================================================================
// Split large transfers into full 32256-element chunks and a bounded remainder
// so both fields stay in range.
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
// SPM Memset (selected execution cores, default 8)
// =============================================================================
// Clear num_elements fp16 values at the same SPM address on the selected cores.
// 参数约定与 ddr2spm_multi_core_v2 一致，使用 memset_spm_multi_core_v2 kernel。
// =============================================================================

void rpu_launch_memset_spm_multicore(
    uint32_t spm_addr,
    int64_t num_elements, int num_cores)
{
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "memset_spm: execution cores must be in [1,8]");
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

      auto* wq = GET_QUEUE(num_cores);
      wq->set_broadcast_mode(true);
      std::vector<uint8_t> cores;
      for (int i = 0; i < num_cores; ++i)
          cores.push_back(static_cast<uint8_t>(i));
      wq->enqueu_kernel(*kernel, {(uint16_t)num_blocks, 1, 1}, cores);
}

// =============================================================================
// Fill SPM (multi-core broadcast): write `value` into num_elements halfs at the
// per-core SPM addr. Uses the graph-aware oplib `fill` (1D) kernel — the one
// rhino's CreateFill drives. Reg layout from
// rhino-ops FillGenerator.get_config (1d branch): reg0/1 SPM byte offset, reg2/3
// element count, reg4 fp16 value, reg5 mem-flag (0=SPM), grid.x=ceil(n/2048). Graph-
// aware fetch so it records into the fused graph (get_program needs it preloaded).
KernelRegisterImage fill_spm_registers(uint32_t spm_addr, int64_t num_elements,
                                       c10::Half value) {
    TORCH_CHECK(num_elements > 0,
                "fill_spm: num_elements must be > 0");
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
    const int64_t VAL_NUM_PER_WRP = 2048;
    const uint16_t gx = (uint16_t)(
        (num_elements + VAL_NUM_PER_WRP - 1) / VAL_NUM_PER_WRP);
    KernelRegisterImage image;
    image.set(0, (uint16_t)(spm_addr & 0xFFFF));
    image.set(1, (uint16_t)(spm_addr >> 16));
    image.set(2, (uint16_t)(num_elements & 0xFFFF));
    image.set(3, (uint16_t)(num_elements >> 16));
    image.set(4, value.x);  // fp16 fill value (0 for zeroing)
    image.set(5, mem_flag);
    image.set(64, gx);
    image.set(65, (uint16_t)1);
    image.set(66, (uint16_t)1);
    image.grid = {gx, 1, 1};
    return image;
}

void rpu_launch_fill_spm_kernel(uint32_t spm_addr, int64_t num_elements,
                                c10::Half value, int num_cores,
                                int core_begin) {
    TORCH_CHECK(core_begin >= 0 && num_cores >= 1 &&
                    core_begin + num_cores <= SpmAllocator::NUM_CORES,
                "fill_spm: selected core range must be inside [0, ",
                SpmAllocator::NUM_CORES, "), got [", core_begin, ", ",
                core_begin + num_cores, ")");
    const KernelRegisterImage image = fill_spm_registers(spm_addr, num_elements, value);
    auto& graph = RpuKernelGraph::active();
    graph.stage_kernel_no_ddr("fill", GraphKernelNoDdrProof::FillSpm);
    Kernel_t* kernel = graph.get_kernel_reset("fill");
    TORCH_CHECK(kernel != nullptr, "fill_spm: failed to get fill kernel");
    for (size_t i = 0; i < image.count; ++i)
        kernel->set_regs(image.regs[i].first, image.regs[i].second);
    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) {
        cores.push_back(static_cast<uint8_t>(core_begin + i));
    }
    wq->enqueu_kernel(*kernel, {image.grid[0], image.grid[1], image.grid[2]}, cores);
}

// =============================================================================
// DDR -> SPM Scatter (多核, 每个 core 从不同 DDR 区域读取)
// =============================================================================
// Each core reads DDR data from ddr_base + core_id * core_stride_bytes,
// writes to the SAME local SPM address (spm_addr) on each core.
//
// core_stride is in bytes; DDR base must be 256-byte aligned.
// =============================================================================




// =============================================================================
// DMA-based memcpy primitives (graph-only)
// =============================================================================
// File-local helper: recover the local SPM offset from the unified-mode
// address returned by FusedModelBase::addr(0, name), then compose the
// per-core absolute address from SPM_ALLOC.addr(core, offset). Matches the
// existing pattern for the SPM→DDR direction. See spec §5.
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
                caller, ": owner must be a contiguous FP16 device Tensor");
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

namespace {

// Stream 0 belongs to compute. The remaining DMA engines must stay inside
// the active Graph's immutable allocated core prefix, not the copy's fanout.
int deferred_lane_count() {
    const int cores = RpuKernelGraph::active().runtime_policy().execution_core_count;
    TORCH_CHECK(cores >= 1 && cores <= 8, "deferred DMA: invalid Graph core budget");
    return cores - 1;
}

uint8_t next_deferred_lane(DeferredDmaChannels& deferred) {
    const int lanes = deferred_lane_count();
    TORCH_CHECK(lanes > 0 && deferred.next < lanes,
                "deferred DMA: no available lane or token from a different core budget");
    const uint8_t ch = static_cast<uint8_t>(1 + deferred.next);
    deferred.next = static_cast<uint8_t>((deferred.next + 1) % lanes);
    return ch;
}

// End of lane `lane` when `total` bytes are cut into `lanes` equal
// 64-byte-aligned runs; the last lane takes the remainder.
size_t deferred_lane_end(size_t total, int lane, int lanes) {
    return lane + 1 == lanes
        ? total : total * static_cast<size_t>(lane + 1) / lanes / 64 * 64;
}

// The next channel of `deferred`'s rotation, fenced behind stream 0's work so
// far.
uint8_t open_deferred_lane(DeferredDmaChannels& deferred) {
    const uint8_t ch = next_deferred_lane(deferred);
    graph_dma::emit_barrier(static_cast<uint8_t>(ch * 2), 0);
    deferred.used |= static_cast<uint8_t>(1u << ch);
    return ch;
}

struct DeferredBroadcastPiece {
    int core;       // destination core
    size_t offset;  // byte offset into the copy (source and destination)
    size_t bytes;
    int lane;       // 0..6, one DMA channel per lane
};

// Every core's copy, laid end to end, cut into one equal 64-byte-aligned run
// per lane. Eight per-core copies on seven channels would leave one channel a
// second copy to finish alone.
std::vector<DeferredBroadcastPiece> deferred_broadcast_pieces(
        size_t bytes, int num_cores) {
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8 && bytes % 64 == 0 &&
                    bytes <= (8ULL << 20),
                "ddr_broadcast_spm_dma_mutable deferred: ", num_cores, " cores of ",
                bytes, " bytes; needs 1..8 cores and a 64-byte multiple of at most 8 MB");
    const int lanes = std::min(num_cores, deferred_lane_count());
    TORCH_CHECK(lanes > 0, "deferred broadcast requires a non-compute DMA lane");
    const size_t total = bytes * static_cast<size_t>(num_cores);
    std::vector<DeferredBroadcastPiece> pieces;
    size_t begin = 0;
    for (int lane = 0; lane < lanes; ++lane) {
        const size_t end = deferred_lane_end(total, lane, lanes);
        for (size_t at = begin; at < end;) {
            const int core = static_cast<int>(at / bytes);
            const size_t offset = at % bytes;
            const size_t n = std::min(end - at, bytes - offset);
            pieces.push_back({core, offset, n, lane});
            at += n;
        }
        begin = end;
    }
    return pieces;
}

}  // namespace

void rpu_launch_ddr_broadcast_spm_dma_mutable(const uint64_t* live_src_base,
                                                int64_t src_offset_bytes,
                                                int64_t num_elements,
                                                uint32_t spm_addr_unified,
                                                int num_cores,
                                                DeferredDmaChannels* deferred,
                                                int64_t sync_elements) {
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
    TORCH_CHECK(sync_elements >= 0 && sync_elements < num_elements &&
                    (deferred != nullptr || sync_elements == 0) &&
                    (sync_elements * sizeof(c10::Half)) % 16 == 0,
                "ddr_broadcast_spm_dma_mutable: synchronous prefix ", sync_elements,
                " must be a 16-byte multiple below ", num_elements,
                " and needs a deferred token");

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
    // With one allocated core there is no independent DMA stream. Complete
    // the entire copy on stream 0 and leave the token empty.
    const bool can_defer = deferred != nullptr && deferred_lane_count() > 0;
    const size_t sync_bytes = can_defer
        ? static_cast<size_t>(sync_elements) * sizeof(c10::Half) : bytes;
    if (sync_bytes > 0) {
        for (int ch = 1; ch < num_cores; ++ch)
            graph_dma::emit_barrier(static_cast<uint8_t>(ch * 2), 0);

        for (int c = 0; c < num_cores; ++c)
            graph_dma::emit_mutable_src(live_src_base, src_offset_bytes,
                                       spm_unified_to_per_core_abs(c, spm_addr_unified),
                                       sync_bytes, c);

        for (int ch = 1; ch < num_cores; ++ch)
            graph_dma::emit_barrier(0, static_cast<uint8_t>(ch * 2));
    }
    if (!can_defer) return;

    // The remainder leaves stream 0 free: allocated nonzero channels only, each fenced
    // behind the work above, joined by rpu_join_deferred_dma.
    uint8_t channels[8] = {}, fenced = 0;
    for (int c = 0; c < num_cores; ++c) {
        channels[c] = next_deferred_lane(*deferred);
        if (!(fenced & (1u << channels[c]))) {
            graph_dma::emit_barrier(static_cast<uint8_t>(channels[c] * 2), 0);
            fenced |= static_cast<uint8_t>(1u << channels[c]);
        }
    }
    for (int c = 0; c < num_cores; ++c)
        graph_dma::emit_mutable_src(
            live_src_base, src_offset_bytes + static_cast<int64_t>(sync_bytes),
            spm_unified_to_per_core_abs(c, spm_addr_unified) + sync_bytes,
            bytes - sync_bytes, channels[c]);
    deferred->used |= fenced;
}

void rpu_launch_ddr_broadcast_spm_dma_mutable(
        uint64_t& live_src_base,
        const at::Tensor& owner_tensor,
        int64_t src_offset_bytes,
        int64_t num_elements,
        uint32_t spm_addr_unified,
        int num_cores,
        DeferredDmaChannels* deferred) {
    TORCH_CHECK(src_offset_bytes >= 0 && num_elements > 0,
                "ddr_broadcast_spm_dma_mutable owned: invalid byte range");
    const size_t byte_begin = static_cast<size_t>(src_offset_bytes);
    const size_t byte_count =
        static_cast<size_t>(num_elements) * sizeof(c10::Half);
    validate_fp16_dma_owner_slice(
        owner_tensor, byte_begin, byte_count,
        "ddr_broadcast_spm_dma_mutable owned");
    if (graph_dma::active()) {
        if (deferred != nullptr && deferred_lane_count() > 0) {
            // One census occurrence per piece; stream 0 is left free.
            const auto pieces = deferred_broadcast_pieces(byte_count, num_cores);
            graph_dma::stage_census_mutable_burst(
                live_src_base, owner_tensor, GraphOuterFastDmaSide::Source,
                byte_begin, byte_count, pieces.size());
            int lane = -1;
            uint8_t ch = 0;
            for (const auto& piece : pieces) {
                if (piece.lane != lane) {
                    lane = piece.lane;
                    ch = open_deferred_lane(*deferred);
                }
                graph_dma::emit_mutable_src(
                    &live_src_base, src_offset_bytes + static_cast<int64_t>(piece.offset),
                    spm_unified_to_per_core_abs(piece.core, spm_addr_unified) + piece.offset,
                    piece.bytes, ch);
            }
            return;
        }
        graph_dma::stage_census_mutable_burst(
            live_src_base, owner_tensor, GraphOuterFastDmaSide::Source,
            byte_begin, byte_count, static_cast<size_t>(num_cores));
    }
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &live_src_base, src_offset_bytes, num_elements,
        spm_addr_unified, num_cores);
}

void rpu_join_deferred_dma(DeferredDmaChannels& deferred) {
    for (int ch = 1; ch < 8; ++ch) {
        if (deferred.used & (1u << ch))
            graph_dma::emit_barrier(0, static_cast<uint8_t>(ch * 2));
    }
    deferred = DeferredDmaChannels{};
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

void rpu_launch_spm_group_broadcast_dma(uint32_t spm_src_unified,
                                        int64_t elements_per_core,
                                        uint32_t spm_dst_unified,
                                        int source_cores,
                                        int num_cores,
                                        int source_core_base,
                                        DeferredDmaChannels* deferred) {
    constexpr const char* caller = "spm_group_broadcast_dma";
    TORCH_CHECK(SPM_ALLOC.is_initialized(), caller, ": SPM is not initialized");
    TORCH_CHECK(source_cores >= 1 && source_cores <= num_cores
                    && num_cores <= 8 && num_cores % source_cores == 0,
                caller, ": source cores must divide the consumer prefix [1,8]");
    TORCH_CHECK(source_core_base >= 0
                    && source_core_base + source_cores <= num_cores,
                caller, ": source cores must lie inside the consumer prefix");
    TORCH_CHECK(elements_per_core > 0
                    && static_cast<uint64_t>(elements_per_core)
                        <= SpmAllocator::SPM_USABLE / sizeof(c10::Half),
                caller, ": element count exceeds one SPM window");
    const uint64_t bytes =
        static_cast<uint64_t>(elements_per_core) * sizeof(c10::Half);
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    TORCH_CHECK(spm_src_unified >= base && spm_dst_unified >= base,
                caller, ": expected core-0 unified SPM addresses");
    const uint64_t src_off = spm_src_unified - base;
    const uint64_t dst_off = spm_dst_unified - base;
    TORCH_CHECK((bytes % 16) == 0 && (src_off % 16) == 0 && (dst_off % 16) == 0,
                caller, ": source, destination and byte count must be 16-byte aligned");
    TORCH_CHECK(src_off + bytes <= SpmAllocator::SPM_USABLE
                    && dst_off + bytes <= SpmAllocator::SPM_USABLE,
                caller, ": source or destination exceeds per-core SPM");
    // Different channels may run in any order. A destination on a producer
    // core must never overwrite source data still needed by another channel.
    TORCH_CHECK(src_off + bytes <= dst_off || dst_off + bytes <= src_off,
                caller, ": source and destination local ranges must be disjoint");
    TORCH_CHECK(graph_dma::active(), caller, ": requires an active Graph");

    const int repeat = num_cores / source_cores;
    auto source = [&](int c) {
        return spm_unified_to_per_core_abs(source_core_base + c / repeat,
                                           spm_src_unified);
    };
    if (deferred != nullptr && deferred_lane_count() > 0) {
        // Channel 0 is stream 0 itself; use only the allocated nonzero lanes. Any
        // engine reaches every core's SPM; the channel only picks the stream.
        uint8_t channels[8] = {};
        uint8_t fenced = 0;
        for (int c = 0; c < num_cores; ++c) {
            channels[c] = next_deferred_lane(*deferred);
            if (!(fenced & (1u << channels[c]))) {
                // Fence this burst, not an earlier one: stream 0 may have
                // run more work since the token last used the channel.
                graph_dma::emit_barrier(static_cast<uint8_t>(channels[c] * 2), 0);
                fenced |= static_cast<uint8_t>(1u << channels[c]);
            }
        }
        for (int c = 0; c < num_cores; ++c) {
            graph_dma::emit_fixed(source(c),
                                  spm_unified_to_per_core_abs(c, spm_dst_unified),
                                  bytes, channels[c]);
        }
        deferred->used |= fenced;
        return;
    }
    emit_scatter_pre_barriers(num_cores);
    for (int c = 0; c < num_cores; ++c) {
        graph_dma::emit_fixed(
            source(c), spm_unified_to_per_core_abs(c, spm_dst_unified), bytes, c);
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
        int64_t num_elements,
        DeferredDmaChannels* deferred,
        bool source_replicated) {
    TORCH_CHECK(dst_offset_elements >= 0 && num_elements > 0,
                "spm_copy_ddr_dma owned: invalid element range");
    TORCH_CHECK(!source_replicated || deferred != nullptr,
                "spm_copy_ddr_dma owned: a replicated source spreads deferred runs only");
    const size_t byte_begin =
        static_cast<size_t>(dst_offset_elements) * sizeof(c10::Half);
    const size_t byte_count =
        static_cast<size_t>(num_elements) * sizeof(c10::Half);
    validate_fp16_dma_owner_slice(
        ddr_tensor, byte_begin, byte_count, "spm_copy_ddr_dma owned");
    if (graph_dma::active()) {
        if (deferred != nullptr && deferred_lane_count() > 0) {
            // One run per allocated nonzero channel; stream 0 is left free.
            const int lanes = deferred_lane_count();
            TORCH_CHECK(byte_count % 64 == 0 && byte_count >= 64 * lanes &&
                            byte_count <= (8ULL << 20),
                        "spm_copy_ddr_dma deferred: ", byte_count, " bytes must be a "
                        "64-byte multiple of at least one run per channel and at most 8 MB");
            graph_dma::stage_census_fixed_burst(
                ddr_tensor, GraphOuterFastDmaSide::Destination,
                byte_begin, byte_count, /*expected_occurrence_count=*/lanes);
            c10::Half* ddr_ptr = ddr_tensor.data_ptr<c10::Half>() + dst_offset_elements;
            rpu_ddr_flush(ddr_ptr);
            const uint64_t ddr_abs = RpuGetDevAddr(ddr_ptr);
            size_t begin = 0;
            for (int lane = 0; lane < lanes; ++lane) {
                const size_t end = deferred_lane_end(byte_count, lane, lanes);
                // Every run reading core 0 contends with the overlapped kernel's
                // own core-0 SPM traffic (measured ~80 us per Pi prefill layer).
                const uint64_t spm_abs = spm_unified_to_per_core_abs(
                    source_replicated ? lane + 1 : 0, spm_addr_unified);
                graph_dma::emit_fixed(spm_abs + begin, ddr_abs + begin, end - begin,
                                      open_deferred_lane(*deferred));
                begin = end;
            }
            return;
        }
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
// One-shot batches that execute synchronously. Replace the immediate-mode
// branches of rpu_launch_ddr2spm_multicore / rpu_launch_spm2ddr_multicore at
// call sites that run outside a normal active Graph scope (SigLIP patch
// embedding, standalone preprocess), plus the single plain PASSTHROUGH
// compatibility suffix described below. The legacy kernel-launch path has
// known correctness issues at certain sizes; this DMA path is the structural
// fix.
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

    if (g_rpu_ddr_flush_enabled) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(ddr_ptr);
        if (bytes <= std::numeric_limits<uintptr_t>::max() - address) {
            rhino_lkn::RpuDdrFlush(ddr_ptr, bytes);
        } else {
            rpu_ddr_flush(ddr_ptr);
        }
    }
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

// Internal companion of assemble_inputs_embeds_parts(prefix_is_embeddings).
// The assembler admits tensor geometry/current SPM first. This synchronous
// helper retains the caller's Tensor owners and temporary until completion;
// QueueCache owns only immediate/one-shot queues, never GraphCache entries.
void rpu_launch_assembled_prefix_parts_dma_immediate(
    const at::Tensor& prefix, at::TensorList parts, const at::Tensor& runs,
    const at::Tensor& output, uint32_t spm, int64_t chunk_rows) {
    RpuKernelGraph::check_direct_immediate_dma_allowed(
        "assembled_prefix_parts_dma_immediate");
    const size_t bytes = prefix.nbytes();
    TORCH_CHECK(bytes > 0 && bytes % 16 == 0 &&
                output.nbytes() == bytes,
                "assembled_prefix_parts_dma_immediate: invalid prefix/output DMA bytes");
    const int64_t* run_data = runs.data_ptr<int64_t>();
    const int64_t row_bytes = prefix.size(1) * static_cast<int64_t>(sizeof(c10::Half));
    TORCH_CHECK(row_bytes > 0 && row_bytes % 16 == 0 && chunk_rows > 0 &&
                chunk_rows <= static_cast<int64_t>(SpmAllocator::SPM_PLANNING_BUDGET) / row_bytes &&
                chunk_rows <= prefix.size(0),
                "assembled_prefix_parts_dma_immediate: invalid scratch row geometry");
    const size_t scratch_bytes = static_cast<size_t>(chunk_rows) * row_bytes;
    for (size_t index = 0; index < parts.size(); ++index) {
        const size_t part_bytes = parts[index].nbytes();
        TORCH_CHECK(part_bytes > 0 && part_bytes % 16 == 0,
                    "assembled_prefix_parts_dma_immediate: invalid part DMA bytes");
        TORCH_CHECK(run_data[2 * index] >= 0 &&
                    static_cast<uint64_t>(run_data[2 * index]) <= bytes / row_bytes &&
                    part_bytes <= bytes - static_cast<size_t>(run_data[2 * index]) * row_bytes,
                    "assembled_prefix_parts_dma_immediate: part outside assembled rows");
    }

    // Resolve every full logical owner range before publication or queue work.
    // The lease and caller's Tensor/ScopedTemporary owners cover the complete
    // synchronous submission, including every reuse of the scratch region.
    std::vector<RpuDeviceDmaEndpointRequest> requests;
    requests.reserve(parts.size() + 3);
    requests.push_back({RpuGetDevAddr(prefix.data_ptr()), bytes,
                        "assembled_prefix_parts_dma_immediate/prefix"});
    for (const auto& part : parts) {
        requests.push_back({RpuGetDevAddr(part.data_ptr()), part.nbytes(),
                            "assembled_prefix_parts_dma_immediate/part"});
    }
    const size_t output_index = requests.size();
    requests.push_back({RpuGetDevAddr(output.data_ptr()), bytes,
                        "assembled_prefix_parts_dma_immediate/output"});
    const size_t scratch_index = requests.size();
    requests.push_back({spm_unified_to_per_core_abs(/*core=*/0, spm), scratch_bytes,
                        "assembled_prefix_parts_dma_immediate/scratch"});
    for (const auto& request : requests) {
        TORCH_CHECK(request.device_addr != 0 && (request.device_addr & 15u) == 0 &&
                    request.bytes > 0 && request.bytes % 16 == 0 &&
                    request.bytes <= std::numeric_limits<uint64_t>::max() - request.device_addr,
                    "assembled_prefix_parts_dma_immediate: invalid DMA address/range");
    }
    std::vector<RpuDmaEndpoint> endpoints(requests.size());
    [[maybe_unused]] auto submission_lease = rpu_resolve_device_dma_endpoints(
        requests.data(), requests.size(), endpoints.data());
    for (size_t index = 0; index < endpoints.size(); ++index) {
        TORCH_CHECK(endpoints[index] && requests[index].bytes <=
                    std::numeric_limits<size_t>::max() - endpoints[index].offset,
                    "assembled_prefix_parts_dma_immediate: invalid managed DMA offset");
    }

    Queue_t* wq = QueueCache::instance().get(/*core_num=*/1);
    try {
        auto publish_source = [&](const at::Tensor& source) {
            const size_t source_bytes = source.nbytes();
            auto* pointer = source.data_ptr<c10::Half>();
            if (g_rpu_ddr_flush_enabled) {
                const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
                if (source_bytes <= std::numeric_limits<uintptr_t>::max() - address) {
                    rhino_lkn::RpuDdrFlush(pointer, source_bytes);
                } else {
                    rpu_ddr_flush(pointer);
                }
            }
        };
        publish_source(prefix);
        for (const auto& part : parts) publish_source(part);
        auto* output_pointer = output.data_ptr<c10::Half>();
        rpu_ddr_flush(output_pointer);
        auto shifted = [&](size_t index, size_t offset) {
            auto endpoint = endpoints[index];
            endpoint.offset += offset;
            return endpoint;
        };
        for (int64_t first = 0; first < prefix.size(0); first += chunk_rows) {
            const int64_t length = std::min(chunk_rows, prefix.size(0) - first);
            const size_t chunk_bytes = static_cast<size_t>(length) * row_bytes;
            const size_t first_byte = static_cast<size_t>(first) * row_bytes;
            rpu_add_dma_checked(*wq, shifted(0, first_byte), endpoints[scratch_index],
                                chunk_bytes, /*ch=*/0,
                                "assembled_prefix_parts_dma_immediate/prefix_chunk");
            for (size_t index = 0; index < parts.size(); ++index) {
                const int64_t start = run_data[2 * index];
                const int64_t begin = std::max(first, start);
                const int64_t end = std::min(first + length,
                                            start + run_data[2 * index + 1]);
                if (begin >= end) continue;
                rpu_add_dma_checked(*wq,
                    shifted(index + 1, static_cast<size_t>(begin - start) * row_bytes),
                    shifted(scratch_index, static_cast<size_t>(begin - first) * row_bytes),
                    static_cast<size_t>(end - begin) * row_bytes, /*ch=*/0,
                    "assembled_prefix_parts_dma_immediate/part_chunk");
            }
            rpu_add_dma_checked(*wq, endpoints[scratch_index], shifted(output_index, first_byte),
                                chunk_bytes, /*ch=*/0,
                                "assembled_prefix_parts_dma_immediate/output_chunk");
        }
        submit_immediate_batch(wq, "assembled_prefix_parts_dma_immediate");
        rpu_ddr_flush(output_pointer);
    } catch (...) {
        // Every normal submission waits and discards its completed state.
        // On SDK/add failure, destroy cached immediate queues before the
        // caller releases scratch/owners; destruction drains pending work and
        // drops partially built arenas. GraphCache's private queues stay owned
        // by their entries and are not in QueueCache.
        QueueCache::instance().clear();
        throw;
    }
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

    if (g_rpu_ddr_flush_enabled) {
        const size_t core_steps = static_cast<size_t>(num_cores - 1);
        const uint64_t stride = static_cast<uint64_t>(core_stride_bytes);
        const bool bounded = core_stride_bytes >= 0 &&
            (core_steps == 0 ||
             stride <= (std::numeric_limits<size_t>::max() - bytes) / core_steps);
        const size_t span = bounded ? bytes + core_steps * stride : 0;
        const uintptr_t address = reinterpret_cast<uintptr_t>(ddr_base);
        if (bounded && span <= std::numeric_limits<uintptr_t>::max() - address) {
            rhino_lkn::RpuDdrFlush(ddr_base, span);
        } else {
            // Raw callers may use a negative stride. Preserve their existing
            // publication when no forward byte range can be represented.
            rpu_ddr_flush(ddr_base);
        }
    }
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

void rpu_launch_spm_scatter_rows_ddr_dma_immediate(uint32_t  spm_addr_unified,
                                                    c10::Half* ddr_base,
                                                    int64_t   rows,
                                                    int64_t   elements_per_core,
                                                    int64_t   spm_row_stride_bytes,
                                                    int64_t   ddr_row_stride_bytes,
                                                    int64_t   core_stride_bytes,
                                                    int       num_cores) {
    constexpr const char* caller = "spm_scatter_rows_ddr_dma_immediate";
    RpuKernelGraph::check_direct_immediate_dma_allowed(caller);
    TORCH_CHECK(!graph_dma::active(), caller, ": eager only");
    TORCH_CHECK(ddr_base != nullptr, caller, ": null ddr_base");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                caller, ": num_cores 1..8, got ", num_cores);
    TORCH_CHECK(rows > 0 && elements_per_core > 0, caller,
                ": rows and elements_per_core must be positive");
    const size_t bytes =
        static_cast<size_t>(elements_per_core) * sizeof(c10::Half);
    TORCH_CHECK(bytes % 16 == 0 && spm_row_stride_bytes % 16 == 0 &&
                    ddr_row_stride_bytes % 16 == 0 && core_stride_bytes % 16 == 0,
                caller, ": bytes and strides must be 16-byte aligned");
    TORCH_CHECK(bytes <= (8ULL << 20), caller, ": single-DMA cap is 8 MB, got ", bytes);
    const uint32_t spm_base = SPM_ALLOC.addr(0, 0);
    TORCH_CHECK(spm_addr_unified >= spm_base &&
                    spm_row_stride_bytes >= static_cast<int64_t>(bytes) &&
                    static_cast<uint64_t>(spm_addr_unified - spm_base) +
                            static_cast<uint64_t>(rows - 1) *
                                static_cast<uint64_t>(spm_row_stride_bytes) +
                            bytes <= SpmAllocator::SPM_USABLE,
                caller, ": SPM rows overlap or leave the core SPM");

    rpu_ddr_flush(ddr_base);
    const uint64_t ddr_abs = RpuGetDevAddr(ddr_base);
    Queue_t* wq = QueueCache::instance().get(static_cast<uint8_t>(num_cores));

    // Same barrier frame as one scatter; row r of core c lands at
    // ddr_base + r*ddr_row_stride + c*core_stride, so every row is one batch.
    for (int ch = 1; ch < num_cores; ++ch)
        wq->add_barrier(static_cast<uint8_t>(ch * 2), 0);
    for (int c = 0; c < num_cores; ++c)
        for (int64_t r = 0; r < rows; ++r)
            rpu_add_dma_checked(
                *wq,
                spm_unified_to_per_core_abs(
                    c, spm_addr_unified +
                           static_cast<uint32_t>(r * spm_row_stride_bytes)),
                ddr_abs + static_cast<uint64_t>(r) *
                              static_cast<uint64_t>(ddr_row_stride_bytes) +
                    static_cast<uint64_t>(c) *
                        static_cast<uint64_t>(core_stride_bytes),
                bytes, c, caller);
    for (int ch = 1; ch < num_cores; ++ch)
        wq->add_barrier(0, static_cast<uint8_t>(ch * 2));

    submit_immediate_batch(wq, caller);
    rpu_ddr_flush(ddr_base);
}
