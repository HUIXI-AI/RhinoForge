// rpu_partial_mrope.cpp — Qwen3.5 partial M-RoPE launcher (prefill + decode).
//
// Hardware kernel: `partial_mrope` (KernelId::PARTIAL_MROPE).
// It writes only the first `rotary_dim` values of each
// head (rotate-half WITHIN rotary_dim); callers alias input/output when the
// remaining head values must pass through.
//
// The host cos/sin tables contain the 3D M-RoPE (T/H/W) interleaving.
// This kernel takes no strobe masks or position_ids. For text-only input the
// tables reduce to standard partial RoPE. cos/sin are [seq, rotary_dim/2] DDR tables indexed by
// (position_offset + token).
//
// Register layout:
//   reg[0]/reg[1]   FreqCos DDR base (>> 8)
//   reg[2]/reg[3]   FreqSin DDR base (>> 8)
//   reg[4]/reg[5]   input  SPM address (32-bit)
//   reg[6]/reg[7]   output SPM address (32-bit)
//   reg[8]/reg[9]   position_offset (uint32 split; absolute token index)
//   reg[10]         head_dim
//   reg[11]         heads_num (per-core local heads)
//   reg[12]         num_tokens (seq_len of this chunk)
//   reg[13]         rotary_dim (e.g. 64 = head_dim * 0.25)
//   reg[14]         is_freq_in_ddr (1 = DDR tables)
// Grid: x = ceil(rotary_dim/2 / 64), y = ceil(num_tokens / 64), z = 1.

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

    // reg[0..3] are the DDR operands. Under typed census the move-only writer
    // is their sole register-write authority and snapshots the final state.
    if (register_writer != nullptr) {
        register_writer->write(*kernel);
    } else {
        kernel->set_regs(0, (uint16_t)(cos_addr & 0xFFFF));
        kernel->set_regs(1, (uint16_t)(cos_addr >> 16));
        kernel->set_regs(2, (uint16_t)(sin_addr & 0xFFFF));
        kernel->set_regs(3, (uint16_t)(sin_addr >> 16));
    }

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

void rpu_launch_partial_mrope_qk_spm_kernel(
    uint32_t q_x_spm_addr,
    uint32_t q_y_spm_addr,
    uint32_t k_x_spm_addr,
    uint32_t k_y_spm_addr,
    const at::Tensor& cos_il,
    const at::Tensor& sin_il,
    int64_t pos_offset,
    int64_t num_tokens,
    int64_t q_local_heads,
    int64_t k_local_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int num_cores) {
    const auto is_table = [](const at::Tensor& tensor) {
        return tensor.defined() &&
            tensor.device().type() == c10::DeviceType::PrivateUse1 &&
            tensor.scalar_type() == at::kHalf && tensor.dim() == 2 &&
            tensor.is_contiguous();
    };
    TORCH_CHECK(is_table(cos_il) && is_table(sin_il) &&
                    cos_il.sizes() == sin_il.sizes() &&
                    cos_il.device() == sin_il.device() && cos_il.size(1) == 64,
                "fused QK partial M-RoPE requires matching owned FP16 RPU "
                "cos/sin tables [rows,64]");
    TORCH_CHECK(pos_offset >= 0 && pos_offset <= cos_il.size(0) &&
                    num_tokens > 0 && num_tokens <= cos_il.size(0) - pos_offset &&
                    static_cast<uint64_t>(pos_offset) <= UINT32_MAX,
                "fused QK partial M-RoPE position exceeds table owner capacity");
    TORCH_CHECK(
        pos_offset >= 0 && (num_tokens == 192 || num_tokens == 1) &&
            q_local_heads == 2 && k_local_heads == 1 && head_dim == 128 &&
            rotary_dim == 128 && num_cores == 8,
        "fused QK partial M-RoPE supports only Qwen3-VL 2B "
        "seq=192/1,qh=2,kh=1,dim=128,cores=8");

    auto register_writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
        KernelId::QWEN3VL_PARTIAL_MROPE_QK_FUSED,
        std::vector<GraphDdrRegisterOperandSpec>{
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{
                    GraphDdrRegisterRole::SemanticInput,
                    GraphDdrRegisterAccess::Read,
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 0, 1},
                cos_il},
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{
                    GraphDdrRegisterRole::SemanticInput,
                    GraphDdrRegisterAccess::Read,
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
                sin_il}});
    rpu_ddr_flush(cos_il.data_ptr<c10::Half>());
    rpu_ddr_flush(sin_il.data_ptr<c10::Half>());
    const uint64_t pos_off_u = static_cast<uint64_t>(pos_offset);

    Kernel_t* kernel = GET_KERNEL(KernelId::QWEN3VL_PARTIAL_MROPE_QK_FUSED);
    TORCH_CHECK(kernel != nullptr,
                "Failed to get partial_mrope_qk_fused kernel");
    kernel->reset_regs();
    kernel->set_regs(4, static_cast<uint16_t>(q_x_spm_addr & 0xFFFF));
    kernel->set_regs(5, static_cast<uint16_t>(q_x_spm_addr >> 16));
    kernel->set_regs(6, static_cast<uint16_t>(q_y_spm_addr & 0xFFFF));
    kernel->set_regs(7, static_cast<uint16_t>(q_y_spm_addr >> 16));
    kernel->set_regs(8, static_cast<uint16_t>(pos_off_u & 0xFFFF));
    kernel->set_regs(9, static_cast<uint16_t>(pos_off_u >> 16));
    kernel->set_regs(10, static_cast<uint16_t>(head_dim));
    kernel->set_regs(11, static_cast<uint16_t>(q_local_heads));
    kernel->set_regs(12, static_cast<uint16_t>(num_tokens));
    kernel->set_regs(13, static_cast<uint16_t>(rotary_dim));
    kernel->set_regs(14, static_cast<uint16_t>(1));

    const uint32_t x_delta = k_x_spm_addr - q_x_spm_addr;
    const uint32_t y_delta = k_y_spm_addr - q_y_spm_addr;
    kernel->set_regs(15, static_cast<uint16_t>(x_delta & 0xFFFF));
    kernel->set_regs(16, static_cast<uint16_t>(x_delta >> 16));
    kernel->set_regs(17, static_cast<uint16_t>(y_delta & 0xFFFF));
    kernel->set_regs(18, static_cast<uint16_t>(y_delta >> 16));
    kernel->set_regs(
        19, static_cast<uint16_t>(k_local_heads - q_local_heads));

    register_writer.write(*kernel);
    auto* queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel,
        {1, static_cast<uint16_t>((num_tokens + 63) / 64), 2},
        {0, 1, 2, 3, 4, 5, 6, 7});
}

// Fixed production entry: typed DDR operands have the same owners/encodings
// as the original tensor-owned RoPE and K-only V16 insert. This is the sole
// boundary converting the public KV offset to the original absolute RoPE ABI.
void rpu_launch_pi05_k_rope_insert_spm_kernel(
    uint32_t k_off, const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& k_cache, int64_t logical_position,
    int64_t spm_rows, const KvInsertSegmentPlan& plan)
{
    rpu_validate_kvinsert_segment_plan(plan, 8, 8, 256);
    TORCH_CHECK(plan.route() == KvInsertRoute::PAD16_V16 &&
                    plan.logical_rows() == 50 && plan.physical_rows() == 64 &&
                    plan.segment_count() == 1 &&
                    plan.segment(0).kernel == KvInsertKernel::V16 &&
                    (plan.segment(0).position == 544 || plan.segment(0).position == 800 ||
                     plan.segment(0).position == 832 || plan.segment(0).position == 576 ||
                     plan.segment(0).position == 864 || plan.segment(0).position == 896 || plan.segment(0).position == 608 || plan.segment(0).position == 640) &&
                    plan.segment(0).token_offset == 0 &&
                    plan.segment(0).rows == 64 && spm_rows == 64,
                "Pi0.5 K-RoPE insert requires the complete M50/P544/P576/P608/P640/P800/P832/P864/P896/PAD16 V16 plan");
    constexpr uint32_t kBytes = 64 * 256 * sizeof(c10::Half);
    TORCH_CHECK((k_off & 255) == 0 &&
                    k_off <= SpmAllocator::SPM_USABLE - kBytes,
                "Pi0.5 K-RoPE insert requires an aligned full64-row SPM region");
    const auto is_fp16 = [](const at::Tensor& tensor) {
        return tensor.defined() && tensor.device().type() == at::kPrivateUse1 &&
               tensor.scalar_type() == at::kHalf &&
               tensor.layout() == c10::Layout::Strided && tensor.is_contiguous();
    };
    TORCH_CHECK(is_fp16(cos) && is_fp16(sin) && cos.dim() == 2 &&
                    cos.sizes() == sin.sizes() && cos.size(1) == 128 &&
                    logical_position >= 0 && logical_position <= UINT32_MAX &&
                    logical_position <= cos.size(0) &&
                    50 <= cos.size(0) - logical_position,
                "Pi0.5 K-RoPE insert logical position exceeds owned FP16 tables");
    TORCH_CHECK(is_fp16(k_cache) && k_cache.dim() == 7 &&
                    k_cache.size(0) == 1 &&
                    k_cache.size(1) >= (plan.segment(0).position + 64) / 16 &&
                    k_cache.size(2) == 1 && k_cache.size(3) == 16 &&
                    k_cache.size(4) == 8 && k_cache.size(5) == 16 &&
                    k_cache.size(6) == 16,
                "Pi0.5 K-RoPE insert requires DDR cache [1,S,1,16,8,16,16] covering the suffix");
    for (const at::Tensor* owner : {&cos, &sin, &k_cache}) {
        const uint64_t address = RpuGetDevAddr(owner->data_ptr());
        TORCH_CHECK(address != 0 && (address & 255) == 0 &&
                        (address >> 8) <= UINT32_MAX,
                    "Pi0.5 K-RoPE insert DDR address exceeds shift8 lo/hi ABI");
        rpu_ddr_flush(owner->data_ptr<c10::Half>());
    }
    auto register_writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
        KernelId::PI05_K_ROPE_INSERT_M50D256P64,
        std::vector<GraphDdrRegisterOperandSpec>{
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{GraphDdrRegisterRole::SemanticInput,
                    GraphDdrRegisterAccess::Read,
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 0, 1}, cos},
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{GraphDdrRegisterRole::SemanticInput,
                    GraphDdrRegisterAccess::Read,
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, sin},
            GraphDdrRegisterOperandSpec{
                GraphDdrRegisterAbi{GraphDdrRegisterRole::KeyCache,
                    GraphDdrRegisterAccess::Write,
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi, 16, 17}, k_cache}});
    const uint32_t k_absolute = SPM_ALLOC.addr(0, k_off);
    Kernel_t* kernel = GET_KERNEL(KernelId::PI05_K_ROPE_INSERT_M50D256P64);
    TORCH_CHECK(kernel, "Missing optional Pi0.5 K-RoPE insert expansion kernel");
    kernel->reset_regs();
    const auto set = [&](uint32_t index, uint16_t value) {
        TORCH_CHECK(kernel->set_regs(index, value) == kKernelRegOk,
                    "Pi0.5 K-RoPE insert register setter rejected index ", index);
    };
    set(4, uint16_t(k_absolute)); set(5, uint16_t(k_absolute >> 16));
    set(6, uint16_t(k_absolute)); set(7, uint16_t(k_absolute >> 16));
    set(8, uint16_t(logical_position));
    set(9, uint16_t(uint64_t(logical_position) >> 16));
    set(10, 256); set(11, 1); set(12, 50); set(13, 256); set(14, 1);
    set(18, uint16_t(plan.segment(0).position)); set(19, 0); set(20, 64); set(21, 8);
    set(64, 2); set(65, 1); set(66, 1);
    // Candidate rendezvous uses slots1/2; original V16 mutex uses slot0.
    // Reset the full launch atomic bank for every occurrence/REPLAY node.
    for (uint32_t index = 128; index <= 255; ++index) set(index, 0);
    register_writer.write(*kernel);  // sole authority for p0/1, p2/3, p16/17
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {2,1,1}, {0,1,2,3,4,5,6,7});
    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());
}

// Decode variant with the same register layout as partial_mrope.
// Grid: x=ceil(rotary_dim/2/512), y=num_tokens.
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

// =============================================================================
// rpu_partial_mrope_test — isolated unit-test harness (mirrors
// rpu_rope_2d_test_ddr). This production op retains main's four-argument,
// full-rotary contract; Qwen3.5's variable rotary_dim harness is the separate
// conformance-only partial_rope_1d_test op below.
// =============================================================================
at::Tensor rpu_partial_mrope_test(
    const at::Tensor& input,
    const at::Tensor& cos_il,
    const at::Tensor& sin_il,
    int64_t pos_offset)
{
    RECORD_FUNCTION("rpu::partial_mrope_test", {});

    TORCH_CHECK((input.dim() == 3 || input.dim() == 4) &&
                    input.scalar_type() == at::kHalf,
                "partial_mrope_test: input must be fp16 [tokens,heads,head_dim] "
                "or [cores,tokens,local_heads,head_dim], got dim=", input.dim(),
                " dtype=", input.scalar_type());
    TORCH_CHECK(cos_il.dim() == 2 && cos_il.scalar_type() == at::kHalf
                && sin_il.dim() == 2 && sin_il.scalar_type() == at::kHalf,
                "partial_mrope_test: cos_il/sin_il must be 2D fp16");
    TORCH_CHECK(input.is_contiguous() && cos_il.is_contiguous() && sin_il.is_contiguous(),
                "partial_mrope_test: all inputs must be contiguous");
    TORCH_CHECK(input.device().type() == at::kPrivateUse1
                && cos_il.device().type() == at::kPrivateUse1
                && sin_il.device().type() == at::kPrivateUse1,
                "partial_mrope_test: all inputs must be on RPU device");

    const int num_cores = input.dim() == 4 ? static_cast<int>(input.size(0)) : 1;
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "partial_mrope_test: cores must be in [1,8], got ", num_cores);
    const int64_t num_tokens = input.size(input.dim() - 3);
    const int64_t num_heads = input.size(input.dim() - 2);
    const int64_t head_dim = input.size(input.dim() - 1);
    const int64_t rotary_dim = head_dim;
    TORCH_CHECK(cos_il.size(1) == rotary_dim / 2,
                "partial_mrope_test: cos_il.size(1)=", cos_il.size(1),
                " must equal rotary_dim/2=", rotary_dim / 2);
    TORCH_CHECK(pos_offset + num_tokens <= cos_il.size(0),
                "partial_mrope_test: pos_offset+num_tokens (", pos_offset + num_tokens,
                ") exceeds cos_il rows (", cos_il.size(0), ")");

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

    const int64_t elements_per_core = num_tokens * num_heads * head_dim;
    const int64_t bytes = elements_per_core * 2;

    using AR = SpmAllocator::AllocRequest;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{bytes, 1, 2},  // input  (DMA-in → kernel-read)
        AR{bytes, 2, 3},  // output (kernel-write → DMA-out)
    });
    const uint32_t in_addr  = SPM_ALLOC.addr(0, offsets[0]);
    const uint32_t out_addr = SPM_ALLOC.addr(0, offsets[1]);

    if (num_cores == 1) {
        rpu_launch_ddr_broadcast_spm_dma_immediate(
            const_cast<c10::Half*>(input.data_ptr<c10::Half>()),
            elements_per_core, in_addr, num_cores);
    } else {
        rpu_launch_ddr_scatter_spm_dma_immediate(
            const_cast<c10::Half*>(input.data_ptr<c10::Half>()),
            elements_per_core, bytes, in_addr, num_cores);
    }

    rpu_launch_partial_mrope_spm_kernel(
        in_addr, out_addr,
        const_cast<c10::Half*>(cos_il.data_ptr<c10::Half>()),
        const_cast<c10::Half*>(sin_il.data_ptr<c10::Half>()),
        pos_offset, num_tokens, num_heads, head_dim, rotary_dim, num_cores);

    auto output = at::empty_like(input);
    if (num_cores == 1) {
        rpu_launch_spm_copy_ddr_dma_immediate(
            out_addr, output.data_ptr<c10::Half>(), elements_per_core);
    } else {
        rpu_launch_spm_scatter_ddr_dma_immediate(
            out_addr, output.data_ptr<c10::Half>(),
            elements_per_core, bytes, num_cores);
    }

    SPM_ALLOC.reset_temporary();
    return output;
}

// =============================================================================
// rpu_partial_rope_1d_test — isolated single-core test harness for the DECODE
// kernel (partial_rope_1d). Identical to rpu_partial_mrope_test except it calls
// rpu_launch_partial_rope_1d_spm_kernel. cos/sin are [max_seq, rotary_dim/2]
// position-lookup DDR tables.
// =============================================================================
at::Tensor rpu_partial_rope_1d_test(
    const at::Tensor& input,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t rotary_dim,
    int64_t pos_offset)
{
    TORCH_CHECK(input.dim() == 3 && input.scalar_type() == at::kHalf,
                "partial_rope_1d_test: input must be 3D fp16 [tokens, heads, head_dim]");
    TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2,
                "partial_rope_1d_test: cos/sin must be 2D [max_seq, rotary_dim/2]");
    TORCH_CHECK(input.is_contiguous() && cos.is_contiguous() && sin.is_contiguous(),
                "partial_rope_1d_test: inputs must be contiguous");
    TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                "partial_rope_1d_test: inputs must be on RPU device");
    TORCH_CHECK(cos.size(1) == rotary_dim / 2,
                "partial_rope_1d_test: cos.size(1)=", cos.size(1),
                " must equal rotary_dim/2=", rotary_dim / 2);

    const int64_t num_tokens = input.size(0);
    const int64_t num_heads  = input.size(1);
    const int64_t head_dim   = input.size(2);

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

    const int64_t bytes = num_tokens * num_heads * head_dim * 2;
    const int num_cores = 1;

    using AR = SpmAllocator::AllocRequest;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{bytes, 1, 2},  // input  (DMA-in → kernel-read)
        AR{bytes, 2, 3},  // output (kernel-write → DMA-out)
    });
    const uint32_t in_addr  = SPM_ALLOC.addr(0, offsets[0]);

    rpu_launch_ddr_broadcast_spm_dma_immediate(
        const_cast<c10::Half*>(input.data_ptr<c10::Half>()),
        num_tokens * num_heads * head_dim, in_addr, num_cores);

    // In-place (x == y): kernel writes only the rotary span; pass-through survives.
    rpu_launch_partial_rope_1d_spm_kernel(
        in_addr, in_addr,
        const_cast<c10::Half*>(cos.data_ptr<c10::Half>()),
        const_cast<c10::Half*>(sin.data_ptr<c10::Half>()),
        pos_offset, num_tokens, num_heads, head_dim, rotary_dim, num_cores);

    auto output = at::empty_like(input);
    rpu_launch_spm_copy_ddr_dma_immediate(
        in_addr, output.data_ptr<c10::Half>(),
        num_tokens * num_heads * head_dim);

    SPM_ALLOC.reset_temporary();
    return output;
}
