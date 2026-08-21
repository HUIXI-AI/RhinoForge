// On-device embedding gather. It loads `num_ids` fp16 rows from the DDR vocab
// table into contiguous SPM scratch, then uses DMA for the DDR result. The
// operator is loaded by name from the combined operator asset.
//
// Public launch ABI; a 32-bit value occupies two consecutive parameters:
//     param0 (reg0/1): ids_count   param2 (reg2): blk=32   param3 (reg3): last_blk
//     param4 (reg4/5): ids>>8      param6 (reg6/7): vocab>>8
//     param8 (reg8/9): output addr param10 (reg10): hidden  param21 (reg21): tp
//     grid (reg64-66): {ceil(ids/32),1,1}

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;

// Gather num_ids fp16 rows from vocab (DDR) into out_spm_addr (core-0 SPM).
void rpu_launch_gather_embedding_kernel(
    c10::Half *vocab_ptr,   // [V, H] fp16 DDR (256B aligned)
    int32_t *ids_ptr,       // [num_ids] int32 DDR (256B aligned)
    uint32_t out_spm_addr,  // [num_ids, H] fp16 SPM (core-0 base)
    int64_t num_ids,
    int64_t hidden_size,
    int num_cores)
{
    TORCH_CHECK(vocab_ptr != nullptr && ids_ptr != nullptr,
                "rpu_launch_gather_embedding_kernel: null pointer");
    TORCH_CHECK(num_ids > 0 && hidden_size > 0,
                "rpu_launch_gather_embedding_kernel: num_ids/hidden must be positive (",
                num_ids, "/", hidden_size, ")");

    const uint64_t ids_addr_raw = RpuGetDevAddr(ids_ptr);
    const uint64_t vocab_addr_raw = RpuGetDevAddr(vocab_ptr);
    // The wrapper ABI encodes DDR addresses in 256-byte units.
    TORCH_CHECK((ids_addr_raw & 0xFFu) == 0,
                "gather_embedding: ids must be 256B aligned (got ", ids_addr_raw, ")");
    TORCH_CHECK((vocab_addr_raw & 0xFFu) == 0,
                "gather_embedding: vocab must be 256B aligned (got ", vocab_addr_raw, ")");

    const uint64_t ids_addr = ids_addr_raw >> 8;
    const uint64_t vocab_addr = vocab_addr_raw >> 8;
    const uint32_t out_addr = out_spm_addr;

    const int normal_blk = 32;
    const int blk_cnt = static_cast<int>((num_ids + normal_blk - 1) / normal_blk);
    const int last_blk = static_cast<int>(num_ids - static_cast<int64_t>(normal_blk) * (blk_cnt - 1));
    const int tp = num_cores;

    Kernel_t *kernel = KernelCache::instance().get_kernel("llama_gather_embedding");
    TORCH_CHECK(kernel != nullptr, "Failed to get llama_gather_embedding kernel from .ref");

    kernel->set_regs(0, (uint16_t)(num_ids & 0xFFFF));          // param0 lo: ids_count
    kernel->set_regs(1, (uint16_t)((num_ids >> 16) & 0xFFFF));  // param0 hi
    kernel->set_regs(2, (uint16_t)normal_blk);                  // param2: ids/block
    kernel->set_regs(3, (uint16_t)last_blk);                    // param3: last block ids
    kernel->set_regs(4, (uint16_t)(ids_addr & 0xFFFF));         // param4 lo: ids addr>>8
    kernel->set_regs(5, (uint16_t)(ids_addr >> 16));            // param4 hi
    kernel->set_regs(6, (uint16_t)(vocab_addr & 0xFFFF));       // param6 lo: vocab addr>>8
    kernel->set_regs(7, (uint16_t)(vocab_addr >> 16));          // param6 hi
    kernel->set_regs(8, (uint16_t)(out_addr & 0xFFFF));         // param8 lo: out SPM addr
    kernel->set_regs(9, (uint16_t)((out_addr >> 16) & 0xFFFF)); // param9 hi
    kernel->set_regs(10, (uint16_t)hidden_size);                // param10: hidden_size
    kernel->set_regs(21, (uint16_t)tp);                         // param21: tp
    kernel->set_regs(64, (uint16_t)blk_cnt);                    // grid.x
    kernel->set_regs(65, (uint16_t)1);                          // grid.y
    kernel->set_regs(66, (uint16_t)1);                          // grid.z

    auto *wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> core_list;
    for (int i = 0; i < num_cores; ++i) core_list.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1}, core_list);
}

// torch op: gather_embedding(vocab[V,H] fp16, ids[N] int32) -> [N,H] fp16, all RPU.
// Stages through SPM (gather writes SPM, then DMA SPM→DDR) — see the file header
// for why a direct >4 GB DDR output can't work with this kernel's g1b store.
at::Tensor rpu_gather_embedding(const at::Tensor &vocab, const at::Tensor &ids)
{
    RECORD_FUNCTION("rpu::gather_embedding", {});
    TORCH_CHECK(vocab.dim() == 2 && vocab.scalar_type() == at::kHalf,
                "gather_embedding: vocab must be [V,H] fp16, got dim=", vocab.dim(),
                " dtype=", vocab.scalar_type());
    TORCH_CHECK(ids.dim() == 1 && ids.scalar_type() == at::kInt,
                "gather_embedding: ids must be [N] int32, got dim=", ids.dim(),
                " dtype=", ids.scalar_type());
    TORCH_CHECK(vocab.is_contiguous() && ids.is_contiguous(),
                "gather_embedding: vocab and ids must be contiguous");
    TORCH_CHECK(vocab.device().type() == at::kPrivateUse1
                && ids.device().type() == at::kPrivateUse1,
                "gather_embedding: vocab and ids must be on RPU device");

    const int64_t H = vocab.size(1);
    const int64_t N = ids.size(0);
    TORCH_CHECK(H % 16 == 0, "gather_embedding: hidden_size must be a multiple of 16, got ", H);

    auto out = at::empty({N, H}, vocab.options());
    const int64_t n_elem = N * H;
    const int64_t out_bytes = n_elem * 2;

    // No flushes here. Flushes are CPU<->RPU cache-coherency ops, needed only at host
    // boundaries — this op is pure device→device (kernel reads vocab/ids from DDR,
    // writes SPM, DMAs SPM→DDR). Coherency contract: (1) vocab + ids were flushed by
    // their .to('rpu') upload (rpu_copy_ flushes CPU→RPU) and the CPU never rewrites
    // them; (2) the output is consumed on-device by the next kernel (no CPU read) — or,
    // if a caller reads it back, the .cpu() copy (rpu_copy_ RPU→CPU) flushes it. A
    // per-call flush of the immutable vocabulary table is redundant.
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    using AR = SpmAllocator::AllocRequest;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{out_bytes, 1, 2},   // gather output (phase 1: kernel-write → phase 2: DMA-out)
    });
    const uint32_t out_spm_addr = SPM_ALLOC.addr(0, offsets[0]);

    rpu_launch_gather_embedding_kernel(
        const_cast<c10::Half *>(vocab.data_ptr<c10::Half>()),
        const_cast<int32_t *>(ids.data_ptr<int32_t>()),
        out_spm_addr, N, H, /*num_cores=*/1);

    // SPM → DDR (core-0 SPM holds the full [N,H] gathered rows).
    rpu_launch_spm_copy_ddr_dma_immediate(out_spm_addr, out.data_ptr<c10::Half>(), n_elem);

    SPM_ALLOC.reset_temporary();
    return out;
}

// torch op: assemble_inputs_embeds(vocab[V,H] fp16, ids[N] int32, merged[M,H] fp16,
// img_start) -> [N,H] fp16, all RPU. Fully on-device inputs_embeds assembly for a
// SINGLE contiguous image-token block: gather token embeddings into an SPM scratch,
// scatter the M vision rows into [img_start, img_start+M) via a DDR→SPM DMA, then one
// SPM→DDR copy. NO host memcpy and NO coherency flushes — every step is device→device
// (kernel + DMA). ids must be pre-padded by the caller to the final N (= padded_len),
// so pad rows gather vocab[0] (masked out downstream). The coherency contract is the
// same as gather_embedding (vocab/ids/merged flushed by their .to('rpu') uploads;
// output consumed on-device).
at::Tensor rpu_assemble_inputs_embeds(const at::Tensor &vocab, const at::Tensor &ids,
                                      const at::Tensor &merged, int64_t img_start)
{
    RECORD_FUNCTION("rpu::assemble_inputs_embeds", {});
    TORCH_CHECK(vocab.dim() == 2 && vocab.scalar_type() == at::kHalf,
                "assemble_inputs_embeds: vocab must be [V,H] fp16");
    TORCH_CHECK(ids.dim() == 1 && ids.scalar_type() == at::kInt,
                "assemble_inputs_embeds: ids must be [N] int32");
    TORCH_CHECK(merged.dim() == 2 && merged.scalar_type() == at::kHalf,
                "assemble_inputs_embeds: merged must be [M,H] fp16");
    TORCH_CHECK(vocab.is_contiguous() && ids.is_contiguous() && merged.is_contiguous(),
                "assemble_inputs_embeds: inputs must be contiguous");
    TORCH_CHECK(vocab.device().type() == at::kPrivateUse1
                && ids.device().type() == at::kPrivateUse1
                && merged.device().type() == at::kPrivateUse1,
                "assemble_inputs_embeds: inputs must be on RPU device");
    const int64_t H = vocab.size(1);
    const int64_t N = ids.size(0);
    const int64_t M = merged.size(0);
    TORCH_CHECK(merged.size(1) == H, "assemble_inputs_embeds: merged hidden ", merged.size(1),
                " != vocab hidden ", H);
    TORCH_CHECK(H % 16 == 0, "assemble_inputs_embeds: hidden must be mult of 16, got ", H);
    TORCH_CHECK(img_start >= 0 && img_start + M <= N,
                "assemble_inputs_embeds: image block [", img_start, ",", img_start + M,
                ") out of range [0,", N, ")");

    auto out = at::empty({N, H}, vocab.options());
    const int64_t n_elem = N * H;

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    using AR = SpmAllocator::AllocRequest;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{n_elem * 2, 1, 3},   // assembly scratch (gather-write → merge-write → DMA-out)
    });
    const uint32_t spm = SPM_ALLOC.addr(0, offsets[0]);

    // 1. gather all N token embeddings → SPM (image rows get vocab[image_pad], overwritten next).
    rpu_launch_gather_embedding_kernel(
        const_cast<c10::Half *>(vocab.data_ptr<c10::Half>()),
        const_cast<int32_t *>(ids.data_ptr<int32_t>()),
        spm, N, H, /*num_cores=*/1);
    // 2. scatter the M vision rows into the contiguous image block (DDR→SPM, core 0).
    rpu_launch_ddr_broadcast_spm_dma_immediate(
        const_cast<c10::Half *>(merged.data_ptr<c10::Half>()),
        M * H, spm + (uint32_t)(img_start * H * 2), /*num_cores=*/1);
    // 3. SPM → DDR output.
    rpu_launch_spm_copy_ddr_dma_immediate(spm, out.data_ptr<c10::Half>(), n_elem);

    SPM_ALLOC.reset_temporary();
    return out;
}

// torch op: assemble_inputs_embeds_rev(vocab[V,H], ids[N], merged_window[M,H],
//   rev_idx[M] int32, runs[n_runs,2] int64 CPU) -> [N,H] fp16, all-device.
// Folds the vision window-REVERSE into the embed assembly: gather token embeddings
// into SPM, then for each image run gather merged_window[rev_idx[off:off+len]]
// DIRECTLY into that run's SPM slot (off accumulates across runs). This replaces the
// separate vision reverse-gather (gather_embedding(merged_window, rev_idx) → a merged
// DDR round-trip) + the per-run scatter copies with ONE op — the two gathers seen in
// the trace collapse to (token-gather + per-run reverse-gather), no intermediate DDR
// buffer. `rev_idx[k]` = the window-order row of the k-th spatial-order vision token;
// `runs[r] = (token_start, len)` of image-pad run r (len rows of merged go there).
// merged_window is the RAW vision output (window order, NOT reverse-applied).
at::Tensor rpu_assemble_inputs_embeds_rev(const at::Tensor &vocab, const at::Tensor &ids,
                                          const at::Tensor &merged_window,
                                          const at::Tensor &rev_idx, const at::Tensor &runs)
{
    RECORD_FUNCTION("rpu::assemble_inputs_embeds_rev", {});
    TORCH_CHECK(vocab.dim() == 2 && vocab.scalar_type() == at::kHalf,
                "assemble_inputs_embeds_rev: vocab must be [V,H] fp16");
    TORCH_CHECK(ids.dim() == 1 && ids.scalar_type() == at::kInt,
                "assemble_inputs_embeds_rev: ids must be [N] int32");
    TORCH_CHECK(merged_window.dim() == 2 && merged_window.scalar_type() == at::kHalf,
                "assemble_inputs_embeds_rev: merged_window must be [M,H] fp16");
    TORCH_CHECK(rev_idx.dim() == 1 && rev_idx.scalar_type() == at::kInt,
                "assemble_inputs_embeds_rev: rev_idx must be [M] int32");
    TORCH_CHECK(vocab.is_contiguous() && ids.is_contiguous() && merged_window.is_contiguous()
                && rev_idx.is_contiguous(), "assemble_inputs_embeds_rev: device inputs must be contiguous");
    TORCH_CHECK(vocab.device().type() == at::kPrivateUse1 && ids.device().type() == at::kPrivateUse1
                && merged_window.device().type() == at::kPrivateUse1
                && rev_idx.device().type() == at::kPrivateUse1,
                "assemble_inputs_embeds_rev: vocab/ids/merged_window/rev_idx must be on RPU");
    TORCH_CHECK(runs.dim() == 2 && runs.size(1) == 2 && runs.scalar_type() == at::kLong
                && runs.device().type() == at::kCPU && runs.is_contiguous(),
                "assemble_inputs_embeds_rev: runs must be a contiguous CPU int64 [n_runs,2]");
    const int64_t H = vocab.size(1);
    const int64_t N = ids.size(0);
    const int64_t M = merged_window.size(0);
    TORCH_CHECK(merged_window.size(1) == H && rev_idx.size(0) == M,
                "assemble_inputs_embeds_rev: shape mismatch (H/M)");
    TORCH_CHECK(H % 16 == 0, "assemble_inputs_embeds_rev: hidden must be mult of 16, got ", H);

    auto out = at::empty({N, H}, vocab.options());
    const int64_t n_elem = N * H;

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    using AR = SpmAllocator::AllocRequest;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{n_elem * 2, 1, 3},   // assembly scratch (token-gather → per-run reverse-gather → DMA-out)
    });
    const uint32_t spm = SPM_ALLOC.addr(0, offsets[0]);

    // 1. gather all N token embeddings → SPM (image rows get vocab[image_pad], overwritten next).
    rpu_launch_gather_embedding_kernel(
        const_cast<c10::Half *>(vocab.data_ptr<c10::Half>()),
        const_cast<int32_t *>(ids.data_ptr<int32_t>()),
        spm, N, H, /*num_cores=*/1);
    // 2. per-run: gather merged_window[rev_idx[off:off+len]] → spm[run_start, ...].
    int32_t *rev_base = const_cast<int32_t *>(rev_idx.data_ptr<int32_t>());
    const int64_t *runs_p = runs.data_ptr<int64_t>();
    const int64_t n_runs = runs.size(0);
    int64_t off = 0;
    for (int64_t r = 0; r < n_runs; ++r) {
        const int64_t rs = runs_p[2 * r];
        const int64_t rl = runs_p[2 * r + 1];
        TORCH_CHECK(rs >= 0 && rs + rl <= N, "assemble_inputs_embeds_rev: run [", rs, ",", rs + rl,
                    ") out of [0,", N, ")");
        // rev_idx slice ptr must be 256B aligned for the gather (ids read via glarge>>8).
        // int32 → 256B/4 = 64-element granularity. Wall-OSS runs are 64-token merge blocks.
        TORCH_CHECK((off % 64) == 0,
                    "assemble_inputs_embeds_rev: rev_idx run offset ", off, " not 64-aligned (256B)");
        rpu_launch_gather_embedding_kernel(
            const_cast<c10::Half *>(merged_window.data_ptr<c10::Half>()),
            rev_base + off,
            spm + (uint32_t)(rs * H * 2),
            rl, H, /*num_cores=*/1);
        off += rl;
    }
    TORCH_CHECK(off == M, "assemble_inputs_embeds_rev: sum(run_lens) ", off, " != merged rows ", M);
    // 3. SPM → DDR output.
    rpu_launch_spm_copy_ddr_dma_immediate(spm, out.data_ptr<c10::Half>(), n_elem);

    SPM_ALLOC.reset_temporary();
    return out;
}
