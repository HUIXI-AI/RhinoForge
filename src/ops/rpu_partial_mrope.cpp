// Partial M-RoPE for FP16 [tokens, heads, head_dim] tensors in SPM.
// Cosine and sine tables are contiguous DDR tensors.
// Kernel launch ABI:
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;

static void launch_partial_mrope_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    c10::Half *cos_ptr,
    c10::Half *sin_ptr,
    int64_t pos_offset,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int num_cores,
    GraphKernelRegisterWriter* register_writer)
{
    TORCH_CHECK(cos_ptr != nullptr && sin_ptr != nullptr,
                "rpu_launch_partial_mrope_spm_kernel: cos_ptr/sin_ptr must be non-null");
    TORCH_CHECK(pos_offset >= 0,
                "rpu_launch_partial_mrope_spm_kernel: pos_offset must be >= 0, got ",
                pos_offset);
    TORCH_CHECK(seq_len > 0 && local_heads > 0 && head_dim > 0 && rotary_dim > 0,
                "rpu_launch_partial_mrope_spm_kernel: seq_len/local_heads/head_dim/"
                "rotary_dim must be positive (got ", seq_len, "/", local_heads, "/",
                head_dim, "/", rotary_dim, ")");
    TORCH_CHECK(rotary_dim <= head_dim && (rotary_dim % 2) == 0,
                "rpu_launch_partial_mrope_spm_kernel: rotary_dim=", rotary_dim,
                " must be even and <= head_dim=", head_dim);

    // cos/sin retain the static-DDR-table contract (caller-owned tables).
    rpu_ddr_flush(cos_ptr);
    rpu_ddr_flush(sin_ptr);

    const uint64_t cos_raw_addr = RpuGetDevAddr(cos_ptr);
    const uint64_t sin_raw_addr = RpuGetDevAddr(sin_ptr);
    TORCH_CHECK((cos_raw_addr & 0xFF) == 0 && (sin_raw_addr & 0xFF) == 0,
                "rpu_launch_partial_mrope_spm_kernel: cos/sin DDR addresses must "
                "be 256-byte aligned before encoding with >> 8 (got cos=",
                cos_raw_addr, ", sin=", sin_raw_addr, ")");
    const uint64_t cos_addr = cos_raw_addr >> 8;
    const uint64_t sin_addr = sin_raw_addr >> 8;
    const uint64_t pos_off_u = static_cast<uint64_t>(pos_offset);

    Kernel_t* kernel = GET_KERNEL(KernelId::PARTIAL_MROPE);
    TORCH_CHECK(kernel != nullptr, "Failed to get partial_mrope kernel");

    // reg[0..3] are the DDR operands. Under typed census the move-only writer
    // is their sole register-write authority.
    if (register_writer != nullptr) {
        register_writer->write(*kernel);
    } else {
        kernel->set_regs(0, (uint16_t)(cos_addr & 0xFFFF));
        kernel->set_regs(1, (uint16_t)(cos_addr >> 16));
        kernel->set_regs(2, (uint16_t)(sin_addr & 0xFFFF));
        kernel->set_regs(3, (uint16_t)(sin_addr >> 16));
    }
    // reg[4]/reg[5]: input SPM addr
    kernel->set_regs(4, (uint16_t)(x_spm_addr & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(x_spm_addr >> 16));
    // reg[6]/reg[7]: output SPM addr
    kernel->set_regs(6, (uint16_t)(y_spm_addr & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(y_spm_addr >> 16));
    // reg[8]/reg[9]: position_offset (uint32 split)
    kernel->set_regs(8, (uint16_t)(pos_off_u & 0xFFFF));
    kernel->set_regs(9, (uint16_t)((pos_off_u >> 16) & 0xFFFF));
    // reg[10..14]: scalars
    kernel->set_regs(10, (uint16_t)head_dim);
    kernel->set_regs(11, (uint16_t)local_heads);
    kernel->set_regs(12, (uint16_t)seq_len);     // num_tokens
    kernel->set_regs(13, (uint16_t)rotary_dim);
    kernel->set_regs(14, (uint16_t)1);           // is_freq_in_ddr

    const uint16_t grid_x = (uint16_t)((rotary_dim / 2 + 63) / 64);
    const uint16_t grid_y = (uint16_t)((seq_len + 63) / 64);

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> core_list;
    for (int i = 0; i < num_cores; ++i) core_list.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {grid_x, grid_y, (uint16_t)1}, core_list);
}

void rpu_launch_partial_mrope_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    c10::Half *cos_ptr,
    c10::Half *sin_ptr,
    int64_t pos_offset,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int num_cores)
{
    TORCH_CHECK(
        !RpuKernelGraph::active().kernel_register_census_active(),
        "typed DDR-register census forbids raw partial_mrope DDR operands");
    launch_partial_mrope_spm_kernel(
        x_spm_addr, y_spm_addr, cos_ptr, sin_ptr, pos_offset, seq_len,
        local_heads, head_dim, rotary_dim, num_cores,
        /*register_writer=*/nullptr);
}

void rpu_launch_partial_mrope_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t pos_offset,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int num_cores)
{
    const auto is_fp16_table = [](const at::Tensor& tensor) {
        return tensor.defined() &&
               tensor.device().type() == c10::DeviceType::PrivateUse1 &&
               tensor.scalar_type() == at::kHalf &&
               tensor.layout() == c10::Layout::Strided &&
               tensor.is_contiguous() && tensor.dim() == 2;
    };
    TORCH_CHECK(
        is_fp16_table(cos) && is_fp16_table(sin) &&
            cos.sizes() == sin.sizes() && rotary_dim > 0 &&
            (rotary_dim % 2) == 0 && cos.size(1) == rotary_dim / 2,
        "rpu_launch_partial_mrope_spm_kernel owned: cos/sin must be matching "
        "contiguous FP16 RPU tables [rows,rotary_dim/2]");
    TORCH_CHECK(
        pos_offset >= 0 && seq_len > 0 && pos_offset <= cos.size(0) &&
            seq_len <= cos.size(0) - pos_offset,
        "rpu_launch_partial_mrope_spm_kernel owned: row range [", pos_offset,
        ",", pos_offset + seq_len, ") exceeds cos/sin owner rows");

    auto register_writer =
        RpuKernelGraph::active().stage_kernel_ddr_registers(
            KernelId::PARTIAL_MROPE,
            std::vector<GraphDdrRegisterOperandSpec>{
                GraphDdrRegisterOperandSpec{
                    GraphDdrRegisterAbi{
                        GraphDdrRegisterRole::SemanticInput,
                        GraphDdrRegisterAccess::Read,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                        0, 1},
                    cos},
                GraphDdrRegisterOperandSpec{
                    GraphDdrRegisterAbi{
                        GraphDdrRegisterRole::SemanticInput,
                        GraphDdrRegisterAccess::Read,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                        2, 3},
                    sin}});
    launch_partial_mrope_spm_kernel(
        x_spm_addr, y_spm_addr, cos.data_ptr<c10::Half>(),
        sin.data_ptr<c10::Half>(), pos_offset, seq_len, local_heads,
        head_dim, rotary_dim, num_cores, &register_writer);
}

// Decode variant with the same launch ABI as partial M-RoPE. It uses one token
// per y-block and grid_x = ceil(rotary_dim/2 / 512).
void rpu_launch_partial_rope_1d_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    c10::Half *cos_ptr,
    c10::Half *sin_ptr,
    int64_t pos_offset,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int num_cores)
{
    TORCH_CHECK(cos_ptr != nullptr && sin_ptr != nullptr,
                "rpu_launch_partial_rope_1d_spm_kernel: cos_ptr/sin_ptr must be non-null");
    TORCH_CHECK(pos_offset >= 0,
                "rpu_launch_partial_rope_1d_spm_kernel: pos_offset must be >= 0, got ",
                pos_offset);
    TORCH_CHECK(seq_len > 0 && local_heads > 0 && head_dim > 0 && rotary_dim > 0,
                "rpu_launch_partial_rope_1d_spm_kernel: seq_len/local_heads/head_dim/"
                "rotary_dim must be positive (got ", seq_len, "/", local_heads, "/",
                head_dim, "/", rotary_dim, ")");
    TORCH_CHECK(rotary_dim <= head_dim && (rotary_dim % 2) == 0,
                "rpu_launch_partial_rope_1d_spm_kernel: rotary_dim=", rotary_dim,
                " must be even and <= head_dim=", head_dim);

    rpu_ddr_flush(cos_ptr);
    rpu_ddr_flush(sin_ptr);

    const uint64_t cos_raw_addr = RpuGetDevAddr(cos_ptr);
    const uint64_t sin_raw_addr = RpuGetDevAddr(sin_ptr);
    TORCH_CHECK((cos_raw_addr & 0xFF) == 0 && (sin_raw_addr & 0xFF) == 0,
                "rpu_launch_partial_rope_1d_spm_kernel: cos/sin DDR addresses "
                "must be 256-byte aligned before encoding with >> 8 (got cos=",
                cos_raw_addr, ", sin=", sin_raw_addr, ")");
    const uint64_t cos_addr = cos_raw_addr >> 8;
    const uint64_t sin_addr = sin_raw_addr >> 8;
    const uint64_t pos_off_u = static_cast<uint64_t>(pos_offset);

    Kernel_t* kernel = GET_KERNEL(KernelId::PARTIAL_ROPE_1D);
    TORCH_CHECK(kernel != nullptr, "Failed to get partial_rope_1d kernel");

    kernel->set_regs(0, (uint16_t)(cos_addr & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(cos_addr >> 16));
    kernel->set_regs(2, (uint16_t)(sin_addr & 0xFFFF));
    kernel->set_regs(3, (uint16_t)(sin_addr >> 16));
    kernel->set_regs(4, (uint16_t)(x_spm_addr & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(x_spm_addr >> 16));
    kernel->set_regs(6, (uint16_t)(y_spm_addr & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(y_spm_addr >> 16));
    kernel->set_regs(8, (uint16_t)(pos_off_u & 0xFFFF));
    kernel->set_regs(9, (uint16_t)((pos_off_u >> 16) & 0xFFFF));
    kernel->set_regs(10, (uint16_t)head_dim);
    kernel->set_regs(11, (uint16_t)local_heads);
    kernel->set_regs(12, (uint16_t)seq_len);     // num_tokens
    kernel->set_regs(13, (uint16_t)rotary_dim);
    kernel->set_regs(14, (uint16_t)1);           // is_freq_in_ddr

    const uint16_t grid_x = (uint16_t)((rotary_dim / 2 + 511) / 512);  // NUM_ELT_PER_WRP=512
    const uint16_t grid_y = (uint16_t)seq_len;                          // 1 token per y-block

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> core_list;
    for (int i = 0; i < num_cores; ++i) core_list.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {grid_x, grid_y, (uint16_t)1}, core_list);
}
