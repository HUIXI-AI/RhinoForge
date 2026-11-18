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
//   reg[14]         is_freq_in_ddr (1 = DDR tables, 0 = SPM byte addresses in
//                   reg[0..3], rows read from position_offset on)
// Grid: x = ceil(rotary_dim/2 / 64), y = ceil(num_tokens / 64), z = 1.

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;

static void check_partial_mrope_shape(int64_t seq_len, int64_t local_heads,
                                      int64_t head_dim, int64_t rotary_dim)
{
    TORCH_CHECK(seq_len > 0 && local_heads > 0 && head_dim > 0 && rotary_dim > 0,
                "rpu_launch_partial_mrope_spm_kernel: seq_len/local_heads/head_dim/"
                "rotary_dim must be positive (got ", seq_len, "/", local_heads, "/",
                head_dim, "/", rotary_dim, ")");
    TORCH_CHECK(rotary_dim <= head_dim && (rotary_dim % 2) == 0,
                "rpu_launch_partial_mrope_spm_kernel: rotary_dim=", rotary_dim,
                " must be even and <= head_dim=", head_dim);
}

// reg[4..14]; the caller sets reg[0..3] before or after.
static void set_partial_mrope_regs(Kernel_t* kernel,
                                   uint32_t x_spm_addr,
                                   uint32_t y_spm_addr,
                                   uint64_t pos_off_u,
                                   int64_t seq_len,
                                   int64_t local_heads,
                                   int64_t head_dim,
                                   int64_t rotary_dim,
                                   bool freq_in_ddr)
{
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
    kernel->set_regs(14, (uint16_t)(freq_in_ddr ? 1 : 0));  // is_freq_in_ddr
}

static void enqueue_partial_mrope(Kernel_t* kernel, int64_t seq_len,
                                  int64_t rotary_dim, int num_cores)
{
    const uint16_t grid_x = (uint16_t)((rotary_dim / 2 + 63) / 64);
    const uint16_t grid_y = (uint16_t)((seq_len + 63) / 64);

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> core_list;
    for (int i = 0; i < num_cores; ++i) core_list.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {grid_x, grid_y, (uint16_t)1}, core_list);
}

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
    check_partial_mrope_shape(seq_len, local_heads, head_dim, rotary_dim);

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

    set_partial_mrope_regs(kernel, x_spm_addr, y_spm_addr, pos_off_u, seq_len,
                           local_heads, head_dim, rotary_dim, /*freq_in_ddr=*/true);

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
    enqueue_partial_mrope(kernel, seq_len, rotary_dim, num_cores);
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
        "contiguous FP16 device tables [rows,rotary_dim/2]");
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

void rpu_launch_partial_mrope_spm_table_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    uint32_t cos_spm_addr,
    uint32_t sin_spm_addr,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int num_cores)
{
    check_partial_mrope_shape(seq_len, local_heads, head_dim, rotary_dim);
    Kernel_t* kernel = GET_KERNEL(KernelId::PARTIAL_MROPE);
    TORCH_CHECK(kernel != nullptr, "Failed to get partial_mrope kernel");
    // reg[0..3]: SPM byte addresses of the staged rows (no DDR operand).
    kernel->set_regs(0, (uint16_t)(cos_spm_addr & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(cos_spm_addr >> 16));
    kernel->set_regs(2, (uint16_t)(sin_spm_addr & 0xFFFF));
    kernel->set_regs(3, (uint16_t)(sin_spm_addr >> 16));
    set_partial_mrope_regs(kernel, x_spm_addr, y_spm_addr, /*pos_off_u=*/0, seq_len,
                           local_heads, head_dim, rotary_dim, /*freq_in_ddr=*/false);
    enqueue_partial_mrope(kernel, seq_len, rotary_dim, num_cores);
}

static void launch_partial_mrope_qk_spm_kernel(
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
    int num_cores, KernelId id) {
    const auto is_table = [](const at::Tensor& tensor) {
        return tensor.defined() &&
            tensor.device().type() == c10::DeviceType::PrivateUse1 &&
            tensor.scalar_type() == at::kHalf && tensor.dim() == 2 &&
            tensor.is_contiguous();
    };
    TORCH_CHECK(is_table(cos_il) && is_table(sin_il) &&
                    cos_il.sizes() == sin_il.sizes() &&
                    cos_il.device() == sin_il.device() && cos_il.size(1) == 64,
                "fused QK partial M-RoPE requires matching owned FP16 device "
                "cos/sin tables [rows,64]");
    TORCH_CHECK(pos_offset >= 0 && pos_offset <= cos_il.size(0) &&
                    num_tokens > 0 && num_tokens <= cos_il.size(0) - pos_offset &&
                    static_cast<uint64_t>(pos_offset) <= UINT32_MAX,
                "fused QK partial M-RoPE position exceeds table owner capacity");
    TORCH_CHECK(
        pos_offset >= 0 &&
            ((id == KernelId::QWEN3VL_PARTIAL_MROPE_QK_FUSED &&
              (num_tokens == 192 || num_tokens == 1)) ||
             (id == KernelId::WALL_BASE_PARTIAL_MROPE_QK_H32 && num_tokens == 32)) &&
            q_local_heads == 2 && k_local_heads == 1 && head_dim == 128 &&
            rotary_dim == 128 && num_cores == 8,
        "fused QK partial M-RoPE requires Qwen3-VL seq192/1 or Wall H32 "
        "with qh=2,kh=1,dim=128,cores=8");

    auto register_writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
        id,
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

    Kernel_t* kernel = GET_KERNEL(id);
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

void rpu_launch_partial_mrope_qk_spm_kernel(
    uint32_t q_x, uint32_t q_y, uint32_t k_x, uint32_t k_y,
    const at::Tensor& cos, const at::Tensor& sin, int64_t position,
    int64_t tokens, int64_t q_heads, int64_t k_heads,
    int64_t head_dim, int64_t rotary_dim, int cores) {
    launch_partial_mrope_qk_spm_kernel(q_x, q_y, k_x, k_y, cos, sin,
        position, tokens, q_heads, k_heads, head_dim, rotary_dim, cores,
        KernelId::QWEN3VL_PARTIAL_MROPE_QK_FUSED);
}

// Wall Base uses distinct complete Q/K slabs, each updated in place. The
// paired payload reuses the canonical FP16 arithmetic with grid.z selecting Q/K.
void rpu_launch_wall_base_partial_mrope_qk_spm_kernel(
    uint32_t q, uint32_t k, const at::Tensor& cos, const at::Tensor& sin,
    int64_t position) {
    constexpr int kCores = 8;
    TORCH_CHECK(SPM_ALLOC.is_initialized(), "Wall paired RoPE requires initialized SPM");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::pair<uint64_t, uint64_t> slabs[] = {{q, 32 * 2 * 128 * 2}, {k, 32 * 128 * 2}};
    for (const auto& slab : slabs) {
        const auto [address, bytes] = slab;
        TORCH_CHECK(address != 0 && address >= base && address % 32 == 0 &&
                        address - base <= SpmAllocator::SPM_USABLE - bytes,
                    "Wall paired RoPE requires full aligned absolute SPM slabs");
        for (int core = 0; core < kCores; ++core) {
            TORCH_CHECK(static_cast<uint64_t>(SPM_ALLOC.addr(core, 0)) +
                            address - base + bytes - 1 <= UINT32_MAX,
                        "Wall paired RoPE SPM range exceeds register ABI");
        }
    }
    TORCH_CHECK(slabs[0].first + slabs[0].second <= slabs[1].first ||
                    slabs[1].first + slabs[1].second <= slabs[0].first,
                "Wall paired RoPE Q/K slabs must not overlap");
    TORCH_CHECK(KernelCache::instance().has_loaded(KernelId::WALL_BASE_PARTIAL_MROPE_QK_H32),
                "Wall paired RoPE optional payload is unavailable");
    launch_partial_mrope_qk_spm_kernel(q, q, k, k, cos, sin, position,
        32, 2, 1, 128, 128, kCores, KernelId::WALL_BASE_PARTIAL_MROPE_QK_H32);
}

// Exact LingBot Prefill B1/M224/Q4/K1/HD128/TP8. Each of seven 32-token
// blocks writes the original V32 learned-gamma norm to FP16 scratch, waits
// for its SPM stores, then runs the original partial RoPE. Q and K stages
// are unfolded; neither a cross-block barrier nor a new SPM allocation is
// required. This separate optional entrypoint never widens the Qwen2B ABI.
void rpu_launch_lingbot2_prefill_qk_norm_rope_m224_spm_kernel(
    uint32_t q_abs, uint32_t k_abs,
    uint32_t q_norm_abs, uint32_t k_norm_abs,
    uint32_t q_gamma_abs, uint32_t k_gamma_abs,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t position, double eps) {
    constexpr KernelId kId = KernelId::LINGBOT2_PREFILL_QK_NORM_ROPE_M224;
    constexpr uint64_t kQBytes = 224 * 4 * 128 * sizeof(c10::Half);
    constexpr uint64_t kKBytes = 224 * 128 * sizeof(c10::Half);
    const auto table_ok = [](const at::Tensor& table) {
        if (!(table.defined() &&
            table.device().type() == c10::DeviceType::PrivateUse1 &&
            table.scalar_type() == at::kHalf && table.dim() == 2 &&
            table.layout() == c10::Layout::Strided && table.is_contiguous() &&
            table.size(1) == 64 && table.storage_offset() >= 0)) return false;
        const uint64_t storage_bytes = table.storage().nbytes();
        const uint64_t offset = static_cast<uint64_t>(table.storage_offset());
        return offset <= storage_bytes / sizeof(c10::Half) &&
            table.nbytes() <= storage_bytes - offset * sizeof(c10::Half);
    };
    TORCH_CHECK(table_ok(cos) && table_ok(sin) && cos.sizes() == sin.sizes() &&
                    cos.device() == sin.device(),
                "M224 QK norm/RoPE requires matching owned FP16 device tables [rows,64]");
    TORCH_CHECK(position >= 0 && position <= cos.size(0) &&
                    224 <= cos.size(0) - position &&
                    static_cast<uint64_t>(position) + 224 <= (uint64_t{1} << 32) / 128,
                "M224 QK norm/RoPE range exceeds table owner or u32 table-byte offset");
    const c10::Half epsilon(static_cast<float>(eps));
    TORCH_CHECK(std::isfinite(eps) && eps > 0 &&
                    std::isfinite(static_cast<float>(epsilon)) &&
                    static_cast<float>(epsilon) > 0,
                "M224 QK norm/RoPE requires positive finite FP16 epsilon");
    TORCH_CHECK(SPM_ALLOC.is_initialized(), "M224 QK norm/RoPE requires initialized SPM");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::array<uint32_t, 6> addresses{
        q_abs, k_abs, q_norm_abs, k_norm_abs, q_gamma_abs, k_gamma_abs};
    const std::array<uint64_t, 6> bytes{kQBytes, kKBytes, kQBytes, kKBytes, 256, 256};
    for (size_t i = 0; i < addresses.size(); ++i) {
        const uint64_t address = addresses[i];
        TORCH_CHECK(address >= base && (address & 255u) == 0 &&
                        bytes[i] <= SpmAllocator::SPM_USABLE &&
                        address - base <= SpmAllocator::SPM_USABLE - bytes[i] &&
                        address + bytes[i] <= (uint64_t{1} << 32),
                    "M224 QK norm/RoPE requires complete aligned absolute SPM operands");
        for (size_t j = 0; j < i; ++j) {
            TORCH_CHECK(address + bytes[i] <= addresses[j] ||
                            static_cast<uint64_t>(addresses[j]) + bytes[j] <= address,
                        "M224 QK norm/RoPE matrices, scratch and gamma extents overlap");
        }
    }
    const auto ddr_ok = [](const at::Tensor& table) {
        const uint64_t address = rhino_lkn::RpuGetDevAddr(table.data_ptr<c10::Half>());
        return address != 0 && (address & 255u) == 0 &&
            (address >> 8) <= UINT32_MAX &&
            table.nbytes() <= (uint64_t{1} << 40) - address;
    };
    TORCH_CHECK(ddr_ok(cos) && ddr_ok(sin),
                "M224 QK norm/RoPE tables violate the shift8 DDR ABI");
    // Atomic availability is read-only; GET_KERNEL below is graph-aware and
    // may reset its private kernel. Reject missing assets before staging.
    TORCH_CHECK(KernelCache::instance().has_loaded(kId),
                "M224 QK norm/RoPE optional kernel is absent");
    auto* queue = GET_QUEUE(8);
    TORCH_CHECK(queue != nullptr, "M224 QK norm/RoPE eight-core queue is absent");
    auto& graph = RpuKernelGraph::active();
    auto writer = graph.stage_kernel_ddr_registers(kId,
        std::vector<GraphDdrRegisterOperandSpec>{
            {{GraphDdrRegisterRole::SemanticInput, GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 0, 1}, cos},
            {{GraphDdrRegisterRole::SemanticInput, GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, sin}});
    auto* selected = GET_KERNEL(kId);
    TORCH_CHECK(selected != nullptr, "M224 QK norm/RoPE graph kernel is absent");
    auto& kernel = *selected;
    graph.keep_alive(cos);
    graph.keep_alive(sin);
    rpu_ddr_flush(cos.data_ptr<c10::Half>());
    rpu_ddr_flush(sin.data_ptr<c10::Half>());
    kernel.reset_regs();
    const auto u16 = [&kernel](uint32_t index, uint16_t value) {
        TORCH_CHECK(kernel.set_regs(index, value) == rhino_lkn::kKernelRegOk,
                    "M224 QK norm/RoPE u16 parameter rejected");
    };
    const auto u32 = [&u16](uint32_t index, uint32_t value) {
        u16(index, static_cast<uint16_t>(value));
        u16(index + 1, static_cast<uint16_t>(value >> 16));
    };
    u32(4, q_norm_abs); u32(6, q_abs);
    u32(8, static_cast<uint32_t>(position));
    u16(10, 128); u16(11, 4); u16(12, 224); u16(13, 128); u16(14, 1);
    u32(15, k_norm_abs - q_norm_abs); u32(17, k_abs - q_abs);
    u16(19, static_cast<uint16_t>(1 - 4));
    const c10::Half reciprocal(1.0f / 128);
    uint16_t reciprocal_bits = 0, epsilon_bits = 0;
    static_assert(sizeof(c10::Half) == sizeof(uint16_t), "FP16 register width changed");
    std::memcpy(&reciprocal_bits, &reciprocal, sizeof(reciprocal_bits));
    std::memcpy(&epsilon_bits, &epsilon, sizeof(epsilon_bits));
    u16(20, 128); u16(21, reciprocal_bits); u16(22, 256); u16(23, epsilon_bits);
    u32(24, q_gamma_abs); u32(26, k_gamma_abs - q_gamma_abs);
    writer.write(kernel);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(kernel, {1, 7, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

// Exact LingBot T51, Q4/K1, full-rotary HD128, TP8. Absolute in-place
// operands share the installed paired payload; table position is independent
// of the KV-cache row. active() also permits the lazy PASSTHROUGH Graph.
void rpu_launch_lingbot2_qk_rope_t51_spm_kernel(
    uint32_t q_abs, uint32_t k_abs, const at::Tensor& cos,
    const at::Tensor& sin, int64_t position) {
    constexpr KernelId kId = KernelId::QWEN3VL_PARTIAL_MROPE_QK_FUSED;
    constexpr uint64_t kQBytes = 51 * 4 * 128 * 2;
    constexpr uint64_t kKBytes = 51 * 128 * 2;
    const auto table_ok = [](const at::Tensor& table) {
        if (!(table.defined() &&
            table.device().type() == c10::DeviceType::PrivateUse1 &&
            table.scalar_type() == at::kHalf && table.dim() == 2 &&
            table.layout() == c10::Layout::Strided && table.is_contiguous() &&
            table.size(1) == 64 && table.storage_offset() >= 0)) return false;
        const uint64_t storage_bytes = table.storage().nbytes();
        const uint64_t offset = static_cast<uint64_t>(table.storage_offset());
        return offset <= storage_bytes / sizeof(c10::Half) &&
            table.nbytes() <= storage_bytes - offset * sizeof(c10::Half);
    };
    TORCH_CHECK(table_ok(cos) && table_ok(sin) && cos.sizes() == sin.sizes() &&
                    cos.device() == sin.device(),
                "T51 QK RoPE requires matching owned FP16 device tables [rows,64]");
    TORCH_CHECK(position >= 0 && position <= cos.size(0) &&
                    51 <= cos.size(0) - position &&
                    static_cast<uint64_t>(position) + 51 <= (uint64_t{1} << 32) / 128,
                "T51 QK RoPE row range exceeds table owner or u32 table-byte offset");
    TORCH_CHECK(SPM_ALLOC.is_initialized(), "T51 QK RoPE requires initialized SPM");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const auto spm_ok = [base](uint32_t address, uint64_t bytes) {
        return address >= base && (address & 31u) == 0 &&
            bytes <= SpmAllocator::SPM_USABLE &&
            uint64_t{address} - base <= SpmAllocator::SPM_USABLE - bytes &&
            uint64_t{address} + bytes <= (uint64_t{1} << 32);
    };
    TORCH_CHECK(spm_ok(q_abs, kQBytes) && spm_ok(k_abs, kKBytes),
                "T51 QK RoPE requires complete aligned absolute Q/K SPM extents");
    TORCH_CHECK(uint64_t{q_abs} + kQBytes <= k_abs ||
                    uint64_t{k_abs} + kKBytes <= q_abs,
                "T51 QK RoPE Q/K in-place SPM extents overlap");
    const auto ddr_ok = [](const at::Tensor& table) {
        const uint64_t address = rhino_lkn::RpuGetDevAddr(table.data_ptr<c10::Half>());
        return address != 0 && (address & 255u) == 0 &&
            (address >> 8) <= UINT32_MAX &&
            table.nbytes() <= (uint64_t{1} << 40) - address;
    };
    TORCH_CHECK(ddr_ok(cos) && ddr_ok(sin), "T51 QK RoPE tables violate the shift8 DDR ABI");

    auto* queue = GET_QUEUE(8);
    TORCH_CHECK(queue != nullptr, "LingBot2 T51 QK RoPE eight-core queue is absent");

    auto& graph = RpuKernelGraph::active();
    auto writer = graph.stage_kernel_ddr_registers(kId,
        std::vector<GraphDdrRegisterOperandSpec>{
            {{GraphDdrRegisterRole::SemanticInput, GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 0, 1}, cos},
            {{GraphDdrRegisterRole::SemanticInput, GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, sin}});
    // Graph-aware lookup binds the typed census staged above.
    rhino_lkn::Kernel_t* selected = GET_KERNEL(kId);
    TORCH_CHECK(selected != nullptr, "LingBot2 T51 QK RoPE required kernel is absent");
    auto& kernel = *selected;
    graph.keep_alive(cos);
    graph.keep_alive(sin);
    rpu_ddr_flush(cos.data_ptr<c10::Half>());
    rpu_ddr_flush(sin.data_ptr<c10::Half>());
    kernel.reset_regs();
    const auto u16 = [&kernel](uint32_t index, uint16_t value) {
        TORCH_CHECK(kernel.set_regs(index, value) == rhino_lkn::kKernelRegOk,
                    "T51 QK RoPE u16 parameter rejected");
    };
    const auto u32 = [&u16](uint32_t index, uint32_t value) {
        u16(index, static_cast<uint16_t>(value));
        u16(index + 1, static_cast<uint16_t>(value >> 16));
    };
    u32(4, q_abs); u32(6, q_abs);
    u32(8, static_cast<uint32_t>(position));
    u16(10, 128); u16(11, 4); u16(12, 51); u16(13, 128); u16(14, 1);
    // The installed selector performs both additions modulo2^32 / modulo2^16.
    // Unsigned subtraction supports either physical order of the Q/K buffers.
    const uint32_t delta = k_abs - q_abs;
    u32(15, delta); u32(17, delta);
    u16(19, static_cast<uint16_t>(1 - 4));
    writer.write(kernel);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(kernel, {1, 1, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
}

// The K warp retires its own RoPE tile before running the unchanged paired
// writer. Q warps never enter that writer or its core lock. No new SPM.
void rpu_launch_lingbot2_qk_rope_kv_t51_spm_kernel(
    uint32_t q_abs, uint32_t k_off, uint32_t v_off,
    const at::Tensor& cos, const at::Tensor& sin, int64_t rope_position,
    at::Tensor& k_cache, at::Tensor& v_cache,
    int64_t cache_position, int64_t spm_rows) {
    constexpr KernelId kId = KernelId::LINGBOT2_QK_ROPE_KV_T51;
    constexpr uint64_t kQBytes = 51 * 4 * 128 * sizeof(c10::Half);
    constexpr uint64_t kKVBytes = 51 * 128 * sizeof(c10::Half);
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "T51 QK RoPE/KV requires an active Graph context");
    TORCH_CHECK(spm_rows >= 51 && cache_position >= 0 && cache_position % 16 == 0 &&
                    static_cast<uint64_t>(cache_position) <= UINT32_MAX - 51,
                "T51 QK RoPE/KV requires51 rows and an aligned u32 cache position");
    const auto table_ok = [](const at::Tensor& table) {
        if (!(table.defined() && table.device().type() == c10::DeviceType::PrivateUse1 &&
            table.scalar_type() == at::kHalf && table.dim() == 2 &&
            table.layout() == c10::Layout::Strided && table.is_contiguous() &&
            table.size(1) == 64 && table.storage_offset() >= 0)) return false;
        const uint64_t bytes = table.storage().nbytes();
        const uint64_t offset = static_cast<uint64_t>(table.storage_offset());
        return offset <= bytes / sizeof(c10::Half) &&
            table.nbytes() <= bytes - offset * sizeof(c10::Half);
    };
    TORCH_CHECK(table_ok(cos) && table_ok(sin) && cos.sizes() == sin.sizes() &&
                    cos.device() == sin.device(),
                "T51 QK RoPE/KV requires matching owned FP16 [rows,64] tables");
    TORCH_CHECK(rope_position >= 0 && rope_position <= cos.size(0) &&
                    51 <= cos.size(0) - rope_position &&
                    static_cast<uint64_t>(rope_position) + 51 <= (uint64_t{1} << 32) / 128,
                "T51 QK RoPE/KV table window exceeds owner or u32 byte offset");
    const auto cache_ok = [](const at::Tensor& cache) {
        return cache.defined() && cache.device().type() == c10::DeviceType::PrivateUse1 &&
            cache.scalar_type() == at::kHalf && cache.layout() == c10::Layout::Strided &&
            cache.is_contiguous() && cache.storage_offset() == 0 && cache.dim() == 7 &&
            cache.size(0) == 1 && cache.size(1) > 0 && cache.size(2) == 1 &&
            cache.size(3) == 8 && cache.size(4) == 8 && cache.size(5) == 16 &&
            cache.size(6) == 16 && cache.storage().nbytes() == cache.nbytes();
    };
    TORCH_CHECK(cache_ok(k_cache) && cache_ok(v_cache) &&
                    cos.device() == k_cache.device() && cos.device() == v_cache.device() &&
                    k_cache.size(1) == v_cache.size(1) &&
                    (cache_position + 51 + 15) / 16 <= k_cache.size(1),
                "T51 QK RoPE/KV requires matching full FP16 [1,S16,1,8,8,16,16] caches");
    TORCH_CHECK(SPM_ALLOC.is_initialized(), "T51 QK RoPE/KV requires initialized SPM");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const auto spm_ok = [base](uint32_t address, uint64_t bytes) {
        return address >= base && (address & 31u) == 0 &&
            bytes <= SpmAllocator::SPM_USABLE &&
            uint64_t{address} - base <= SpmAllocator::SPM_USABLE - bytes &&
            uint64_t{address} + bytes <= (uint64_t{1} << 32);
    };
    TORCH_CHECK(spm_ok(q_abs, kQBytes), "T51 QK RoPE/KV Q absolute interval is invalid");
    const auto convert = [base, &spm_ok](uint32_t offset) {
        TORCH_CHECK((offset & 255u) == 0 && offset <= SpmAllocator::SPM_USABLE - kKVBytes,
                    "T51 QK RoPE/KV input offset violates alignment or full extent");
        const uint64_t address = SPM_ALLOC.addr(0, offset);
        TORCH_CHECK(address == base + offset && address <= UINT32_MAX &&
                        (address & 255u) == 0 && spm_ok(static_cast<uint32_t>(address), kKVBytes),
                    "T51 QK RoPE/KV input offset-to-absolute conversion failed");
        return static_cast<uint32_t>(address);
    };
    const uint32_t k_abs = convert(k_off), v_abs = convert(v_off);
    const auto disjoint = [](uint64_t a, uint64_t a_bytes, uint64_t b, uint64_t b_bytes) {
        return a <= b ? b - a >= a_bytes : a - b >= b_bytes;
    };
    TORCH_CHECK(disjoint(q_abs, kQBytes, k_abs, kKVBytes) &&
                    disjoint(q_abs, kQBytes, v_abs, kKVBytes) &&
                    disjoint(k_abs, kKVBytes, v_abs, kKVBytes),
                "T51 QK RoPE/KV complete Q/K/V SPM intervals overlap");
    const auto ddr_address = [](const at::Tensor& owner) {
        const uint64_t address = RpuGetDevAddr(owner.data_ptr<c10::Half>());
        TORCH_CHECK(address != 0 && (address & 255u) == 0 &&
                        (address >> 8) <= UINT32_MAX &&
                        owner.nbytes() <= (uint64_t{1} << 40) - address,
                    "T51 QK RoPE/KV full DDR owner violates shift8 addressing");
        return address;
    };
    const uint64_t c_addr = ddr_address(cos), s_addr = ddr_address(sin);
    const uint64_t k_addr = ddr_address(k_cache), v_addr = ddr_address(v_cache);
    TORCH_CHECK(disjoint(k_addr, k_cache.nbytes(), v_addr, v_cache.nbytes()) &&
                    disjoint(k_addr, k_cache.nbytes(), c_addr, cos.nbytes()) &&
                    disjoint(k_addr, k_cache.nbytes(), s_addr, sin.nbytes()) &&
                    disjoint(v_addr, v_cache.nbytes(), c_addr, cos.nbytes()) &&
                    disjoint(v_addr, v_cache.nbytes(), s_addr, sin.nbytes()),
                "T51 QK RoPE/KV cache writes overlap another cache or table reads");
    TORCH_CHECK(KernelCache::instance().has_loaded(kId),
                "T51 QK RoPE/KV optional payload is absent");
    auto* queue = GET_QUEUE(8);
    TORCH_CHECK(queue != nullptr, "T51 QK RoPE/KV eight-core queue is absent");
    auto& graph = RpuKernelGraph::active();
    auto writer = graph.stage_kernel_ddr_registers(kId,
        std::vector<GraphDdrRegisterOperandSpec>{
            {{GraphDdrRegisterRole::SemanticInput, GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 0, 1}, cos},
            {{GraphDdrRegisterRole::SemanticInput, GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, sin},
            {{GraphDdrRegisterRole::KeyCache, GraphDdrRegisterAccess::Write,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 28, 29}, k_cache},
            {{GraphDdrRegisterRole::ValueCache, GraphDdrRegisterAccess::Write,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 32, 33}, v_cache}});
    Kernel_t* selected = GET_KERNEL(kId);
    TORCH_CHECK(selected != nullptr, "T51 QK RoPE/KV optional payload is absent");
    const at::Tensor* owners[] = {&cos, &sin, &k_cache, &v_cache};
    for (const at::Tensor* owner : owners) {
        graph.keep_alive(*owner);
        rpu_ddr_flush(owner->data_ptr<c10::Half>());
    }
    auto& kernel = *selected;
    kernel.reset_regs();
    const auto u16 = [&kernel](uint32_t index, uint16_t value) {
        TORCH_CHECK(kernel.set_regs(index, value) == kKernelRegOk,
                    "T51 QK RoPE/KV rejected parameter ", index);
    };
    const auto u32 = [&u16](uint32_t index, uint32_t value) {
        u16(index, static_cast<uint16_t>(value));
        u16(index + 1, static_cast<uint16_t>(value >> 16));
    };
    u32(4, q_abs); u32(6, q_abs); u32(8, static_cast<uint32_t>(rope_position));
    u16(10, 128); u16(11, 4); u16(12, 51); u16(13, 128); u16(14, 1);
    const uint32_t delta = k_abs - q_abs;
    u32(15, delta); u32(17, delta); u16(19, static_cast<uint16_t>(1 - 4));
    u32(20, static_cast<uint32_t>(cache_position));
    u16(22, 1); u16(23, 128); u16(24, 51);
    u32(26, k_abs); u32(30, v_abs); u16(41, 8);
    writer.write(kernel);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(kernel, {1, 4, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
    rpu_ddr_flush(k_cache.data_ptr<c10::Half>());
    rpu_ddr_flush(v_cache.data_ptr<c10::Half>());
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
                     plan.segment(0).position == 864 || plan.segment(0).position == 896 || plan.segment(0).position == 608 || plan.segment(0).position == 640 ||
                     plan.segment(0).position == 672 || plan.segment(0).position == 928 ||
                     plan.segment(0).position == 960) &&
                    plan.segment(0).token_offset == 0 &&
                    plan.segment(0).rows == 64 && spm_rows == 64,
                "Pi0.5 K-RoPE insert requires the complete M50/P544/P576/P608/P640/P672/P800/P832/P864/P896/P928/P960/PAD16 V16 plan");
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
                "partial_mrope_test: all inputs must be on device");

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
                "partial_rope_1d_test: inputs must be on device");
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
