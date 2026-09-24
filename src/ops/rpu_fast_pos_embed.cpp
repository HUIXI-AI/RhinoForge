// rpu_fast_pos_embed.cpp — bilinear positional-embedding interpolation launcher
// for the Qwen3-VL / Qwen3.5 vision encoder (STEP 0).
//
// Hardware kernel: `fast_pos_embed_interpolate` (DDR inputs, SPM output).
//
//
// Contract (see the reg-layout block in rpu_kernel_decls.h for the full story):
//   - grid_mapping holds RAW FRACTIONAL (h, w) coords in fp16, already permuted
//     into spatial-merge block order. The kernel does the bilinear itself and
//     does no reordering.
//   - pos_offset shifts the grid_mapping READ only; the output is written
//     densely from out_spm_addr.
//   - Output is always SPM; in the vision graph every core computes the same
//     full-width result redundantly (it is consumed as a full-width residual).

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;

namespace {
constexpr int kNumTokenPerWarp = 8;
constexpr int kNumEltPerWarp   = 2048;

inline uint16_t ceil_div_u16(int64_t a, int64_t b) {
    return static_cast<uint16_t>((a + b - 1) / b);
}
}  // namespace

namespace {
void launch_fast_pos_embed_common(
    uint64_t grid_reg_val,
    uint64_t wgt_reg_val,
    uint32_t out_spm_addr,
    int64_t pos_offset,
    int64_t hidden_size,
    int64_t num_grid_h_side,
    int64_t num_grid_w_side,
    int64_t num_tokens,
    int num_cores)
{
    TORCH_CHECK(pos_offset >= 0,
                "fast_pos_embed_interp: pos_offset must be non-negative, got ",
                pos_offset);
    TORCH_CHECK(hidden_size > 0 && num_tokens > 0,
                "fast_pos_embed_interp: hidden_size/num_tokens must be positive "
                "(got ", hidden_size, "/", num_tokens, ")");
    TORCH_CHECK(num_grid_h_side > 0 && num_grid_w_side > 0,
                "fast_pos_embed_interp: grid sides must be positive (got ",
                num_grid_h_side, "x", num_grid_w_side, ")");

    // regs 8-11 must fit signed 16-bit.
    TORCH_CHECK(hidden_size <= 32767 && num_tokens <= 32767 &&
                num_grid_h_side <= 32767 && num_grid_w_side <= 32767,
                "fast_pos_embed_interp: hidden_size/num_tokens/grid sides must fit "
                "in int16 (got ", hidden_size, "/", num_tokens, "/",
                num_grid_h_side, "x", num_grid_w_side, ")");

    const uint64_t pos_off_u = static_cast<uint64_t>(pos_offset);

    const uint16_t grid_dim_x = ceil_div_u16(hidden_size, kNumEltPerWarp);
    const uint16_t grid_dim_y = ceil_div_u16(num_tokens, kNumTokenPerWarp);
    const uint16_t grid_dim_z = 1;

    Kernel_t *kernel = GET_KERNEL(KernelId::FAST_POS_EMBED_INTERP);
    TORCH_CHECK(kernel != nullptr,
                "Failed to get fast_pos_embed_interpolate kernel");

    // reg[0]/[1]: grid_mapping base
    kernel->set_regs(0, (uint16_t)(grid_reg_val & 0xFFFF));
    kernel->set_regs(1, (uint16_t)((grid_reg_val >> 16) & 0xFFFF));

    // reg[2]/[3]: pos_embed_weight base
    kernel->set_regs(2, (uint16_t)(wgt_reg_val & 0xFFFF));
    kernel->set_regs(3, (uint16_t)((wgt_reg_val >> 16) & 0xFFFF));

    // reg[4]/[5]: patch_pos_embeds output SPM base (raw bytes)
    kernel->set_regs(4, (uint16_t)(out_spm_addr & 0xFFFF));
    kernel->set_regs(5, (uint16_t)((out_spm_addr >> 16) & 0xFFFF));

    // reg[6]/[7]: pos_offset (uint32) — indexes grid_mapping only.
    kernel->set_regs(6, (uint16_t)(pos_off_u & 0xFFFF));
    kernel->set_regs(7, (uint16_t)((pos_off_u >> 16) & 0xFFFF));

    // reg[8..11]: hidden_size / num_grid_H_side / num_grid_W_side / num_tokens.
    // NOTE the asymmetry: reg[9] (H side) is read ONLY for the ceil clamp, while
    // reg[10] (W side) is also the row stride of the embedding table. Swapping
    // them is silent whenever the grid is square (Qwen3.5 is 48x48).
    kernel->set_regs(8,  (uint16_t)hidden_size);
    kernel->set_regs(9,  (uint16_t)num_grid_h_side);
    kernel->set_regs(10, (uint16_t)num_grid_w_side);
    kernel->set_regs(11, (uint16_t)num_tokens);

    // reg[64..66]: grid dimensions, also supplied to enqueu_kernel below.
    kernel->set_regs(64, grid_dim_x);
    kernel->set_regs(65, grid_dim_y);
    kernel->set_regs(66, grid_dim_z);


    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> core_list;
    for (int i = 0; i < num_cores; ++i) core_list.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {grid_dim_x, grid_dim_y, grid_dim_z}, core_list);
}
}  // namespace

// DDR-input variant used by the vision graph.
void rpu_launch_fast_pos_embed_interp_ddr_kernel(
    void *grid_mapping_ptr,
    void *pos_embed_weight_ptr,
    uint32_t out_spm_addr,
    int64_t pos_offset,
    int64_t hidden_size,
    int64_t num_grid_h_side,
    int64_t num_grid_w_side,
    int64_t num_tokens,
    int num_cores)
{
    TORCH_CHECK(grid_mapping_ptr != nullptr && pos_embed_weight_ptr != nullptr,
                "rpu_launch_fast_pos_embed_interp_ddr_kernel: grid_mapping_ptr / "
                "pos_embed_weight_ptr must be non-null");

    // Registers hold addr/256, so both inputs must be 256B aligned. PyTorch's
    // default allocator only guarantees 64B, so the caller has to pad — same
    // story as rpu_rope_2d.cpp:71-74.
    const uint64_t grid_addr_raw = RpuGetDevAddr(grid_mapping_ptr);
    const uint64_t wgt_addr_raw  = RpuGetDevAddr(pos_embed_weight_ptr);
    TORCH_CHECK((grid_addr_raw & 0xFFu) == 0,
                "rpu_launch_fast_pos_embed_interp_ddr_kernel: grid_mapping must be "
                "256B aligned (got ", grid_addr_raw, ")");
    TORCH_CHECK((wgt_addr_raw & 0xFFu) == 0,
                "rpu_launch_fast_pos_embed_interp_ddr_kernel: pos_embed_weight must "
                "be 256B aligned (got ", wgt_addr_raw, ")");

    rpu_ddr_flush(pos_embed_weight_ptr);
    rpu_ddr_flush(grid_mapping_ptr);

    launch_fast_pos_embed_common(
        grid_addr_raw >> 8, wgt_addr_raw >> 8, out_spm_addr, pos_offset,
        hidden_size, num_grid_h_side, num_grid_w_side, num_tokens,
        num_cores);
}

// =============================================================================
// rpu_fast_pos_embed_interp_test — torch op wrapper for isolated testing.
//
// The inputs remain in DDR, matching the production vision graph. The wrapper
// allocates only the SPM output and copies it back for comparison.
// =============================================================================

at::Tensor rpu_fast_pos_embed_interp_test(
    const at::Tensor& grid_mapping,
    const at::Tensor& pos_embed_weight,
    int64_t num_grid_h_side,
    int64_t num_grid_w_side,
    int64_t num_tokens,
    int64_t pos_offset)
{
    RECORD_FUNCTION("rpu::fast_pos_embed_interp_test", {});

    TORCH_CHECK(grid_mapping.dim() == 2 && grid_mapping.size(1) == 2
                && grid_mapping.scalar_type() == at::kHalf,
                "fast_pos_embed_interp_test: grid_mapping must be [N, 2] fp16, got dim=",
                grid_mapping.dim(), " dtype=", grid_mapping.scalar_type());
    TORCH_CHECK(pos_embed_weight.dim() == 2 && pos_embed_weight.scalar_type() == at::kHalf,
                "fast_pos_embed_interp_test: pos_embed_weight must be 2D fp16");
    TORCH_CHECK(grid_mapping.is_contiguous() && pos_embed_weight.is_contiguous(),
                "fast_pos_embed_interp_test: inputs must be contiguous");
    TORCH_CHECK(grid_mapping.device().type() == at::kPrivateUse1
                && pos_embed_weight.device().type() == at::kPrivateUse1,
                "fast_pos_embed_interp_test: inputs must be on RPU device");
    TORCH_CHECK(pos_embed_weight.size(0) == num_grid_h_side * num_grid_w_side,
                "fast_pos_embed_interp_test: pos_embed_weight.size(0)=",
                pos_embed_weight.size(0), " must equal H_side*W_side=",
                num_grid_h_side * num_grid_w_side);
    TORCH_CHECK(pos_offset >= 0 && pos_offset + num_tokens <= grid_mapping.size(0),
                "fast_pos_embed_interp_test: pos_offset(", pos_offset, ") + num_tokens(",
                num_tokens, ") must fit in grid_mapping rows (", grid_mapping.size(0), ")");

    const int64_t hidden_size = pos_embed_weight.size(1);

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

    const int64_t out_elems = num_tokens * hidden_size;
    const int64_t out_bytes = out_elems * 2;
    const int num_cores = 1;  // single-core for the test harness

    using AR = SpmAllocator::AllocRequest;

    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{out_bytes, 1, 2},  // output (phase 1-2: kernel-write → DMA-out)
    });
    const uint32_t out_addr = SPM_ALLOC.addr(0, offsets[0]);

    rpu_launch_fast_pos_embed_interp_ddr_kernel(
        const_cast<c10::Half*>(grid_mapping.data_ptr<c10::Half>()),
        const_cast<c10::Half*>(pos_embed_weight.data_ptr<c10::Half>()),
        out_addr, pos_offset, hidden_size,
        num_grid_h_side, num_grid_w_side, num_tokens, num_cores);

    auto output = at::empty({num_tokens, hidden_size}, grid_mapping.options());
    rpu_launch_spm_copy_ddr_dma_immediate(
        out_addr, output.data_ptr<c10::Half>(), out_elems);

    SPM_ALLOC.reset_temporary();
    return output;
}
