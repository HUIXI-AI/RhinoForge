// 2D RoPE launcher for vision features in SPM.
// Position indices and cosine/sine tables are caller-owned DDR tensors.
// position_idx contract (per-forward, built by Python adapter from grid_thw):
//   [num_tokens, 2] int16 DDR, 256B-aligned. coords[i] = (row_idx, col_idx)
//   of patch i. row_idx, col_idx ∈ [0, max_hw).
//
// FreqCos / FreqSin contract (static once per model — shared across blocks):
//   [max_hw, head_dim/4] FP16 DDR. Same table is indexed by BOTH row and col
//   (see HF Qwen3VLVisionRotaryEmbedding + rot_pos_emb).
//
// pos_offset is always 0 here — the position_idx table is sized to num_tokens.

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;

namespace {
constexpr int kNumTokenPerWarp = 128;
constexpr int kNumEltPerWarp   = 32;

inline uint16_t ceil_div_u16(int64_t a, int64_t b) {
    return static_cast<uint16_t>((a + b - 1) / b);
}
}  // namespace

void rpu_launch_rope_2d_ddr_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    c10::Half *cos_ptr,
    c10::Half *sin_ptr,
    int16_t *position_idx_ddr_ptr,
    int64_t pos_offset,
    int64_t num_tokens,
    int64_t head_num,
    int64_t head_dim,
    int64_t head_dim_pad,
    int num_cores)
{
    TORCH_CHECK(cos_ptr != nullptr && sin_ptr != nullptr,
                "rpu_launch_rope_2d_ddr_kernel: cos_ptr/sin_ptr must be non-null");
    TORCH_CHECK(position_idx_ddr_ptr != nullptr,
                "rpu_launch_rope_2d_ddr_kernel: position_idx_ddr_ptr must be non-null");
    TORCH_CHECK(pos_offset >= 0,
                "rpu_launch_rope_2d_ddr_kernel: pos_offset must be non-negative, got ",
                pos_offset);
    TORCH_CHECK(num_tokens > 0 && head_num > 0 && head_dim > 0,
                "rpu_launch_rope_2d_ddr_kernel: num_tokens/head_num/head_dim must be positive "
                "(got ", num_tokens, "/", head_num, "/", head_dim, ")");
    TORCH_CHECK(head_dim % 4 == 0,
                "rpu_launch_rope_2d_ddr_kernel: head_dim must be divisible by 4 (kernel splits "
                "into row/col halves of head_dim/4 lanes each), got ", head_dim);
    TORCH_CHECK(head_dim_pad >= head_dim,
                "rpu_launch_rope_2d_ddr_kernel: head_dim_pad must be >= head_dim "
                "(", head_dim_pad, " vs ", head_dim, ")");

    // 256B-alignment requirement (>> 8 baked into register layout). Caller-side
    // allocations must satisfy this — PyTorch's default allocator returns 64B-
    // aligned addresses, so caller is responsible for padding cos/sin tables and
    // the position_idx tensor to 256B boundaries.
    const uint64_t pos_addr_raw = RpuGetDevAddr(position_idx_ddr_ptr);
    const uint64_t cos_addr_raw = RpuGetDevAddr(cos_ptr);
    const uint64_t sin_addr_raw = RpuGetDevAddr(sin_ptr);
    TORCH_CHECK((pos_addr_raw & 0xFFu) == 0,
                "rpu_launch_rope_2d_ddr_kernel: position_idx_ddr_ptr must be 256B aligned (got ",
                pos_addr_raw, ")");
    TORCH_CHECK((cos_addr_raw & 0xFFu) == 0,
                "rpu_launch_rope_2d_ddr_kernel: cos_ptr must be 256B aligned (got ",
                cos_addr_raw, ")");
    TORCH_CHECK((sin_addr_raw & 0xFFu) == 0,
                "rpu_launch_rope_2d_ddr_kernel: sin_ptr must be 256B aligned (got ",
                sin_addr_raw, ")");

    rpu_ddr_flush(cos_ptr);
    rpu_ddr_flush(sin_ptr);
    // position_idx flush is the caller's job (per-forward Python copy_in).

    const uint64_t pos_addr = pos_addr_raw >> 8;
    const uint64_t cos_addr = cos_addr_raw >> 8;
    const uint64_t sin_addr = sin_addr_raw >> 8;

    const uint32_t x_addr = x_spm_addr;
    const uint32_t y_addr = y_spm_addr;
    const uint64_t pos_off_u = static_cast<uint64_t>(pos_offset);

    const uint16_t grid_dim_x = ceil_div_u16(head_dim / 4, kNumEltPerWarp);
    const uint16_t grid_dim_y = ceil_div_u16(num_tokens, kNumTokenPerWarp);
    const uint16_t grid_dim_z = 1;

    Kernel_t *kernel = GET_KERNEL(KernelId::ROPE_2D_DDR);
    TORCH_CHECK(kernel != nullptr, "Failed to get rope_2d_ddr kernel");

    // reg[0]/[1]: position_idx DDR addr (>> 8)
    kernel->set_regs(0, (uint16_t)(pos_addr & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(pos_addr >> 16));

    // reg[2]/[3]: FreqCos DDR addr (>> 8)
    kernel->set_regs(2, (uint16_t)(cos_addr & 0xFFFF));
    kernel->set_regs(3, (uint16_t)(cos_addr >> 16));

    // reg[4]/[5]: FreqSin DDR addr (>> 8)
    kernel->set_regs(4, (uint16_t)(sin_addr & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(sin_addr >> 16));

    // reg[6]/[7]: input_hidden SPM addr
    kernel->set_regs(6, (uint16_t)(x_addr & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(x_addr >> 16));

    // reg[8]/[9]: output_hidden SPM addr
    kernel->set_regs(8, (uint16_t)(y_addr & 0xFFFF));
    kernel->set_regs(9, (uint16_t)(y_addr >> 16));

    // reg[10]/[11]: pos_offset (uint32)
    kernel->set_regs(10, (uint16_t)(pos_off_u & 0xFFFF));
    kernel->set_regs(11, (uint16_t)((pos_off_u >> 16) & 0xFFFF));

    // reg[12..15]: head_dim / head_num / num_tokens / head_dim_pad
    kernel->set_regs(12, (uint16_t)head_dim);
    kernel->set_regs(13, (uint16_t)head_num);
    kernel->set_regs(14, (uint16_t)num_tokens);
    kernel->set_regs(15, (uint16_t)head_dim_pad);

    // reg[64..66]: grid dimensions.
    kernel->set_regs(64, grid_dim_x);
    kernel->set_regs(65, grid_dim_y);
    kernel->set_regs(66, grid_dim_z);

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> core_list;
    for (int i = 0; i < num_cores; ++i) core_list.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel,
                      {grid_dim_x, grid_dim_y, grid_dim_z},
                      core_list);
}

// =============================================================================
// rpu_launch_rope_2d_spm_kernel — SPM cos/sin variant.
//
// Identical to the DDR launcher EXCEPT: (1) cos/sin come in as SPM addresses and
// are packed RAW into reg[2..5] (no >>8) — the same encoding as the x/y SPM regs;
// (2) no rpu_ddr_flush / 256B-alignment on cos/sin (they are SPM, not DDR);
// (3) KernelId::ROPE_2D_SPM. position_idx remains in DDR (reg[0]/[1] >>8).
// =============================================================================
static void launch_rope_2d_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    uint32_t cos_spm_addr,
    uint32_t sin_spm_addr,
    int16_t *position_idx_ddr_ptr,
    int64_t pos_offset,
    int64_t num_tokens,
    int64_t head_num,
    int64_t head_dim,
    int64_t head_dim_pad,
    int num_cores,
    GraphKernelRegisterWriter* register_writer)
{
    TORCH_CHECK(position_idx_ddr_ptr != nullptr,
                "rpu_launch_rope_2d_spm_kernel: position_idx_ddr_ptr must be non-null");
    TORCH_CHECK(pos_offset >= 0,
                "rpu_launch_rope_2d_spm_kernel: pos_offset must be non-negative, got ",
                pos_offset);
    TORCH_CHECK(num_tokens > 0 && head_num > 0 && head_dim > 0,
                "rpu_launch_rope_2d_spm_kernel: num_tokens/head_num/head_dim must be positive "
                "(got ", num_tokens, "/", head_num, "/", head_dim, ")");
    TORCH_CHECK(head_dim % 4 == 0,
                "rpu_launch_rope_2d_spm_kernel: head_dim must be divisible by 4 (kernel splits "
                "into row/col halves of head_dim/4 lanes each), got ", head_dim);
    TORCH_CHECK(head_dim_pad >= head_dim,
                "rpu_launch_rope_2d_spm_kernel: head_dim_pad must be >= head_dim "
                "(", head_dim_pad, " vs ", head_dim, ")");

    // Only position_idx is in DDR here — reg[0]/[1] hold its addr>>8. It must be
    // 256B-aligned (>> 8 baked into the register layout). cos/sin are in SPM so
    // they get no flush / alignment check; their regs take a raw byte offset.
    const uint64_t pos_addr_raw = RpuGetDevAddr(position_idx_ddr_ptr);
    TORCH_CHECK((pos_addr_raw & 0xFFu) == 0,
                "rpu_launch_rope_2d_spm_kernel: position_idx_ddr_ptr must be 256B aligned (got ",
                pos_addr_raw, ")");
    // position_idx flush is the caller's job (per-forward Python copy_in).

    const uint64_t pos_addr = pos_addr_raw >> 8;

    const uint32_t x_addr = x_spm_addr;
    const uint32_t y_addr = y_spm_addr;
    const uint64_t pos_off_u = static_cast<uint64_t>(pos_offset);

    const uint16_t grid_dim_x = ceil_div_u16(head_dim / 4, kNumEltPerWarp);
    const uint16_t grid_dim_y = ceil_div_u16(num_tokens, kNumTokenPerWarp);
    const uint16_t grid_dim_z = 1;

    Kernel_t *kernel = GET_KERNEL(KernelId::ROPE_2D_SPM);
    TORCH_CHECK(kernel != nullptr, "Failed to get rope_2d_spm kernel");

    // reg[2]/[3]: FreqCos SPM addr (RAW per-core byte offset — no >>8)
    kernel->set_regs(2, (uint16_t)(cos_spm_addr & 0xFFFF));
    kernel->set_regs(3, (uint16_t)(cos_spm_addr >> 16));

    // reg[4]/[5]: FreqSin SPM addr (RAW per-core byte offset — no >>8)
    kernel->set_regs(4, (uint16_t)(sin_spm_addr & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(sin_spm_addr >> 16));

    // reg[6]/[7]: input_hidden SPM addr
    kernel->set_regs(6, (uint16_t)(x_addr & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(x_addr >> 16));

    // reg[8]/[9]: output_hidden SPM addr
    kernel->set_regs(8, (uint16_t)(y_addr & 0xFFFF));
    kernel->set_regs(9, (uint16_t)(y_addr >> 16));

    // reg[10]/[11]: pos_offset (uint32)
    kernel->set_regs(10, (uint16_t)(pos_off_u & 0xFFFF));
    kernel->set_regs(11, (uint16_t)((pos_off_u >> 16) & 0xFFFF));

    // reg[12..15]: head_dim / head_num / num_tokens / head_dim_pad
    kernel->set_regs(12, (uint16_t)head_dim);
    kernel->set_regs(13, (uint16_t)head_num);
    kernel->set_regs(14, (uint16_t)num_tokens);
    kernel->set_regs(15, (uint16_t)head_dim_pad);

    // reg[64..66]: grid dimensions.
    kernel->set_regs(64, grid_dim_x);
    kernel->set_regs(65, grid_dim_y);
    kernel->set_regs(66, grid_dim_z);

    // reg[0]/[1]: position_idx DDR addr (>> 8).  Under typed census the
    // move-only writer is the sole authority for this DDR register pair and
    // runs after every non-DDR register has been configured.
    if (register_writer != nullptr) {
        register_writer->write(*kernel);
    } else {
        kernel->set_regs(0, (uint16_t)(pos_addr & 0xFFFF));
        kernel->set_regs(1, (uint16_t)(pos_addr >> 16));
    }

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> core_list;
    for (int i = 0; i < num_cores; ++i) core_list.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel,
                      {grid_dim_x, grid_dim_y, grid_dim_z},
                      core_list);
}

void rpu_launch_rope_2d_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    uint32_t cos_spm_addr,
    uint32_t sin_spm_addr,
    int16_t *position_idx_ddr_ptr,
    int64_t pos_offset,
    int64_t num_tokens,
    int64_t head_num,
    int64_t head_dim,
    int64_t head_dim_pad,
    int num_cores)
{
    TORCH_CHECK(
        !RpuKernelGraph::active().kernel_register_census_active(),
        "typed DDR-register census forbids raw rope_2d_spm position input");
    launch_rope_2d_spm_kernel(
        x_spm_addr, y_spm_addr, cos_spm_addr, sin_spm_addr,
        position_idx_ddr_ptr, pos_offset, num_tokens, head_num, head_dim,
        head_dim_pad, num_cores, /*register_writer=*/nullptr);
}

void rpu_launch_rope_2d_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    uint32_t cos_spm_addr,
    uint32_t sin_spm_addr,
    const at::Tensor& position_idx,
    int64_t pos_offset,
    int64_t num_tokens,
    int64_t head_num,
    int64_t head_dim,
    int64_t head_dim_pad,
    int num_cores)
{
    TORCH_CHECK(num_tokens > 0 && head_num > 0 && head_dim > 0 &&
                    head_dim % 4 == 0 && head_dim_pad >= head_dim &&
                    num_cores >= 1 && num_cores <= 8,
                "rpu_launch_rope_2d_spm_kernel owned: invalid launch geometry");
    TORCH_CHECK(position_idx.defined() &&
                    position_idx.device().type() ==
                        c10::DeviceType::PrivateUse1 &&
                    position_idx.scalar_type() == at::kShort &&
                    position_idx.layout() == c10::Layout::Strided &&
                    position_idx.is_contiguous() && position_idx.dim() == 2 &&
                    position_idx.size(1) == 2,
                "rpu_launch_rope_2d_spm_kernel owned: position_idx must be "
                "a contiguous int16 RPU Tensor [rows,2]");
    TORCH_CHECK(pos_offset >= 0 && num_tokens > 0 &&
                    pos_offset <= position_idx.size(0) &&
                    num_tokens <= position_idx.size(0) - pos_offset,
                "rpu_launch_rope_2d_spm_kernel owned: position row range [",
                pos_offset, ",", pos_offset + num_tokens,
                ") exceeds ", position_idx.size(0), " rows");

    auto register_writer =
        RpuKernelGraph::active().stage_kernel_ddr_registers(
            KernelId::ROPE_2D_SPM,
            std::vector<GraphDdrRegisterOperandSpec>{
                GraphDdrRegisterOperandSpec{
                    GraphDdrRegisterAbi{
                        GraphDdrRegisterRole::SemanticInput,
                        GraphDdrRegisterAccess::Read,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                        0, 1},
                    position_idx}});
    launch_rope_2d_spm_kernel(
        x_spm_addr, y_spm_addr, cos_spm_addr, sin_spm_addr,
        position_idx.data_ptr<int16_t>(), pos_offset, num_tokens, head_num,
        head_dim, head_dim_pad, num_cores, &register_writer);
}
