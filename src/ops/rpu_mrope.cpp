// M-RoPE launcher for FP16 [tokens, heads, head_dim] tensors in SPM.
// Position IDs and cosine/sine tables are caller-owned DDR tensors.
// Kernel launch ABI:
//   reg[0]/reg[1]   pos_offset (uint32 split into two uint16; identical encoding
//                   to start_pos of 1D RoPE — absolute index into position_ids)
//   reg[2]          head_dim
//   reg[3]          local_heads (== heads_num)
//   reg[4]/reg[5]   x SPM address (32-bit)
//   reg[6]/reg[7]   cos DDR address (>> 8)
//   reg[8]/reg[9]   sin DDR address (>> 8)
//   reg[10]/reg[11] y SPM address (32-bit)
//   reg[12]/reg[13] position_ids DDR address (>> 8) — mrope-only
//   reg[21]         tp (num_cores)
//   reg[4096..]     T/H/W strobe masks: 3 × WARP_SIZE uint16 values
//
// pos_offset semantics: absolute index into position_ids_ddr. The caller owns
// a `[MAX_KEEPALIVE_SEQ, 3]` int32 DDR buffer that is overwritten in place at
// row `ctx().position` per forward; the launcher receives
// `pos_offset = ctx().position + chunk.offset`.

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <array>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;

namespace {
constexpr int kWarpSize = 16;
constexpr int kStrobeMaskCount = 3;  // T / H / W
}  // namespace

static void launch_mrope_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    c10::Half *cos_ptr,
    c10::Half *sin_ptr,
    int32_t *position_ids_ddr_ptr,
    int64_t pos_offset,
    const std::array<std::array<uint16_t, kWarpSize>, kStrobeMaskCount> &strobe_masks,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int num_cores,
    GraphKernelRegisterWriter* register_writer)
{
    TORCH_CHECK(cos_ptr != nullptr && sin_ptr != nullptr,
                "rpu_launch_mrope_spm_kernel: cos_ptr/sin_ptr must be non-null");
    TORCH_CHECK(position_ids_ddr_ptr != nullptr,
                "rpu_launch_mrope_spm_kernel: position_ids_ddr_ptr must be non-null");
    TORCH_CHECK(pos_offset >= 0,
                "rpu_launch_mrope_spm_kernel: pos_offset must be non-negative, got ",
                pos_offset);
    TORCH_CHECK(seq_len > 0 && local_heads > 0 && head_dim > 0,
                "rpu_launch_mrope_spm_kernel: seq_len/local_heads/head_dim must be positive "
                "(got ", seq_len, "/", local_heads, "/", head_dim, ")");

    // Flush cos/sin DDR — same contract as 1D rope launcher. position_ids buffer
    // flush is the caller's responsibility (it's a per-forward Python copy_in).
    rpu_ddr_flush(cos_ptr);
    rpu_ddr_flush(sin_ptr);

    const uint64_t cos_addr = RpuGetDevAddr(cos_ptr) >> 8;
    const uint64_t sin_addr = RpuGetDevAddr(sin_ptr) >> 8;
    const uint64_t pos_addr = RpuGetDevAddr(position_ids_ddr_ptr) >> 8;

    const uint32_t x_addr = x_spm_addr;
    const uint32_t y_addr = y_spm_addr;
    const uint64_t pos_off_u = static_cast<uint64_t>(pos_offset);

    Kernel_t* kernel = GET_KERNEL(KernelId::MROPE);
    TORCH_CHECK(kernel != nullptr, "Failed to get llama_mrope_interleave kernel");

    // reg[0]/reg[1]: pos_offset (uint32 split)
    kernel->set_regs(0, (uint16_t)(pos_off_u & 0xFFFF));
    kernel->set_regs(1, (uint16_t)((pos_off_u >> 16) & 0xFFFF));

    kernel->set_regs(2, (uint16_t)head_dim);
    kernel->set_regs(3, (uint16_t)local_heads);

    // reg[4]/reg[5]: x SPM addr
    kernel->set_regs(4, (uint16_t)(x_addr & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(x_addr >> 16));

    // reg[10]/reg[11]: y SPM addr
    kernel->set_regs(10, (uint16_t)(y_addr & 0xFFFF));
    kernel->set_regs(11, (uint16_t)(y_addr >> 16));

    kernel->set_regs(21, (uint16_t)num_cores);

    // SCM strobe masks: T/H/W × WARP_SIZE uint16 entries.
    for (int axis = 0; axis < kStrobeMaskCount; ++axis) {
        for (int lane = 0; lane < kWarpSize; ++lane) {
            const int reg_idx = SCM_REG_OFFSET + axis * kWarpSize + lane;
            kernel->set_regs(reg_idx, strobe_masks[axis][lane]);
        }
    }

    // reg[6..9] and reg[12]/reg[13] are the DDR operands. Under typed
    // census the move-only writer is their sole register-write authority and
    // runs after every non-DDR register has been configured.
    if (register_writer != nullptr) {
        register_writer->write(*kernel);
    } else {
        kernel->set_regs(6, (uint16_t)(cos_addr & 0xFFFF));
        kernel->set_regs(7, (uint16_t)(cos_addr >> 16));
        kernel->set_regs(8, (uint16_t)(sin_addr & 0xFFFF));
        kernel->set_regs(9, (uint16_t)(sin_addr >> 16));
        kernel->set_regs(12, (uint16_t)(pos_addr & 0xFFFF));
        kernel->set_regs(13, (uint16_t)(pos_addr >> 16));
    }

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> core_list;
    for (int i = 0; i < num_cores; ++i) core_list.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {(uint16_t)seq_len, (uint16_t)1, (uint16_t)1},
                      core_list);
}

void rpu_launch_mrope_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    c10::Half *cos_ptr,
    c10::Half *sin_ptr,
    int32_t *position_ids_ddr_ptr,
    int64_t pos_offset,
    const std::array<std::array<uint16_t, kWarpSize>, kStrobeMaskCount> &strobe_masks,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int num_cores)
{
    TORCH_CHECK(
        !RpuKernelGraph::active().kernel_register_census_active(),
        "typed DDR-register census forbids raw mrope_spm DDR operands");
    launch_mrope_spm_kernel(
        x_spm_addr, y_spm_addr, cos_ptr, sin_ptr,
        position_ids_ddr_ptr, pos_offset, strobe_masks, seq_len, local_heads,
        head_dim, num_cores, /*register_writer=*/nullptr);
}

void rpu_launch_mrope_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& position_ids,
    int64_t pos_offset,
    const std::array<std::array<uint16_t, kWarpSize>, kStrobeMaskCount> &strobe_masks,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int num_cores)
{
    TORCH_CHECK(seq_len > 0 && local_heads > 0 && head_dim > 0 &&
                    (head_dim % 2) == 0 &&
                    num_cores >= 1 && num_cores <= 8,
                "rpu_launch_mrope_spm_kernel owned: invalid launch geometry");
    const auto is_fp16_table = [](const at::Tensor& tensor) {
        return tensor.defined() &&
               tensor.device().type() == c10::DeviceType::PrivateUse1 &&
               tensor.scalar_type() == at::kHalf &&
               tensor.layout() == c10::Layout::Strided &&
               tensor.is_contiguous() && tensor.dim() == 2;
    };
    TORCH_CHECK(is_fp16_table(cos) && is_fp16_table(sin) &&
                    cos.sizes() == sin.sizes() &&
                    cos.size(1) == head_dim / 2,
                "rpu_launch_mrope_spm_kernel owned: cos/sin must be matching "
                "contiguous FP16 RPU tables [rows,head_dim/2]");
    TORCH_CHECK(position_ids.defined() &&
                    position_ids.device().type() ==
                        c10::DeviceType::PrivateUse1 &&
                    position_ids.scalar_type() == at::kInt &&
                    position_ids.layout() == c10::Layout::Strided &&
                    position_ids.is_contiguous() &&
                    position_ids.dim() == 2 && position_ids.size(1) == 3,
                "rpu_launch_mrope_spm_kernel owned: position_ids must be a "
                "contiguous int32 RPU Tensor [rows,3]");
    TORCH_CHECK(pos_offset >= 0 && seq_len > 0 &&
                    pos_offset <= cos.size(0) &&
                    seq_len <= cos.size(0) - pos_offset &&
                    pos_offset <= position_ids.size(0) &&
                    seq_len <= position_ids.size(0) - pos_offset,
                "rpu_launch_mrope_spm_kernel owned: row range [", pos_offset,
                ",", pos_offset + seq_len,
                ") exceeds cos/sin or position owner rows");

    auto register_writer =
        RpuKernelGraph::active().stage_kernel_ddr_registers(
            KernelId::MROPE,
            std::vector<GraphDdrRegisterOperandSpec>{
                GraphDdrRegisterOperandSpec{
                    GraphDdrRegisterAbi{
                        GraphDdrRegisterRole::ModelWeight,
                        GraphDdrRegisterAccess::Read,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                        6, 7},
                    cos},
                GraphDdrRegisterOperandSpec{
                    GraphDdrRegisterAbi{
                        GraphDdrRegisterRole::ModelWeight,
                        GraphDdrRegisterAccess::Read,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                        8, 9},
                    sin},
                GraphDdrRegisterOperandSpec{
                    GraphDdrRegisterAbi{
                        GraphDdrRegisterRole::SemanticInput,
                        GraphDdrRegisterAccess::Read,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                        12, 13},
                    position_ids}});
    launch_mrope_spm_kernel(
        x_spm_addr, y_spm_addr, cos.data_ptr<c10::Half>(),
        sin.data_ptr<c10::Half>(), position_ids.data_ptr<int32_t>(), pos_offset,
        strobe_masks, seq_len, local_heads, head_dim, num_cores,
        &register_writer);
}

// Compute T/H/W strobe masks for the launch ABI.
// Returns 3 × WARP_SIZE uint16 strobe masks (T / H / W axis). Each bit selects
// whether the rotary half-dim at that lane belongs to that axis. mrope_section
// must have size 3 and sum to head_dim/2.

std::array<std::array<uint16_t, kWarpSize>, kStrobeMaskCount>
rpu_compute_mrope_strobe_masks(
    const std::vector<int32_t> &mrope_section,
    int32_t head_dim)
{
    TORCH_CHECK(mrope_section.size() == 3,
                "rpu_compute_mrope_strobe_masks: mrope_section must have size 3, got ",
                mrope_section.size());
    TORCH_CHECK(head_dim > 0 && (head_dim % 2) == 0,
                "rpu_compute_mrope_strobe_masks: head_dim must be positive and even, got ",
                head_dim);
    const int32_t head_dim_half = head_dim / 2;
    constexpr int kThdVectorSize = 16;
    TORCH_CHECK(head_dim_half <= kWarpSize * kThdVectorSize,
                "rpu_compute_mrope_strobe_masks: head_dim/2=", head_dim_half,
                " exceeds WARP_SIZE*THD_VECTOR_SIZE=", kWarpSize * kThdVectorSize);

    std::array<std::array<uint16_t, kWarpSize>, kStrobeMaskCount> masks{};

    auto set_bit = [](std::array<uint16_t, kWarpSize> &mask_arr,
                      int32_t bit_pos, bool value) {
        const int thd_idx = bit_pos / kThdVectorSize;
        const int lane_idx = bit_pos % kThdVectorSize;
        const uint16_t bit_val = static_cast<uint16_t>(1U << lane_idx);
        if (value) {
            mask_arr[thd_idx] = static_cast<uint16_t>(mask_arr[thd_idx] | bit_val);
        } else {
            mask_arr[thd_idx] = static_cast<uint16_t>(
                mask_arr[thd_idx] & static_cast<uint16_t>(~bit_val));
        }
    };

    // T mask: initialize to all-ones across rotary half-dim
    for (int32_t i = 0; i < head_dim_half; ++i) {
        set_bit(masks[0], i, true);
    }
    // H mask overrides T at pos = 1 + 3*i
    for (int32_t i = 0; i < mrope_section[1]; ++i) {
        const int32_t pos = 1 + i * 3;
        set_bit(masks[1], pos, true);
        set_bit(masks[0], pos, false);
    }
    // W mask overrides T at pos = 2 + 3*i
    for (int32_t i = 0; i < mrope_section[2]; ++i) {
        const int32_t pos = 2 + i * 3;
        set_bit(masks[2], pos, true);
        set_bit(masks[0], pos, false);
    }

    return masks;
}

// Partial M-RoPE launchers live in rpu_partial_mrope.cpp.
