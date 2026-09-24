// rpu_mrope.cpp — M-RoPE launcher for Qwen3-VL text decoder (R-Phase 1).
//
// Hardware kernel: `llama_mrope_interleave` (KernelId::MROPE).
// Wrapper contract mirrors `rpu_launch_rope_spm_kernel` (direct-address SPM
// variant) but adds (a) a per-forward DDR `position_ids` keepalive read at
// reg[12]/reg[13] and (b) 3 × WARP_SIZE uint16 strobe masks pushed via SCM
// (T/H/W axis selection). cos/sin retain their static DDR table contract.
//
// Register layout:
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
#include <cmath>
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

    // SCM strobe masks: T/H/W × WARP_SIZE uint16 entries at
    // [SCM_REG_OFFSET, SCM_REG_OFFSET + 3 × WARP_SIZE).
    for (int axis = 0; axis < kStrobeMaskCount; ++axis) {
        for (int lane = 0; lane < kWarpSize; ++lane) {
            const int reg_idx = SCM_REG_OFFSET + axis * kWarpSize + lane;
            rpu_set_legacy_scm_u16_checked(
                kernel, reg_idx, strobe_masks[axis][lane], "MRoPE");
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

// Optional, exact ordinary 4B decode leaf. The original BASE reductions and
// FP16 retirement are part of the payload, not an alternative RMSNorm policy.
static void launch_qwen3vl_4b_decode_qk_norm_mrope(
    uint32_t q_spm_addr, uint32_t k_spm_addr,
    uint32_t q_gamma_spm_addr, uint32_t k_gamma_spm_addr,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& position_ids, int64_t position, double eps,
    uint32_t v_spm_addr, const at::Tensor* k_cache, const at::Tensor* v_cache)
{
    const bool insert_kv = k_cache != nullptr;
    TORCH_CHECK(insert_kv == (v_cache != nullptr), "QK fusion needs both cache owners");
    const auto table_ok = [](const at::Tensor& tensor) {
        return tensor.defined() &&
            tensor.device().type() == c10::DeviceType::PrivateUse1 &&
            tensor.scalar_type() == at::kHalf &&
            tensor.layout() == c10::Layout::Strided && tensor.is_contiguous() &&
            tensor.dim() == 2 && tensor.size(0) > 0 && tensor.size(1) == 64;
    };
    TORCH_CHECK(table_ok(cos) && table_ok(sin) && cos.sizes() == sin.sizes(),
                "4B decode QK fusion requires matching contiguous FP16 [rows,64] RPU tables");
    TORCH_CHECK(position_ids.defined() &&
                    position_ids.device().type() == c10::DeviceType::PrivateUse1 &&
                    position_ids.scalar_type() == at::kInt &&
                    position_ids.layout() == c10::Layout::Strided &&
                    position_ids.is_contiguous() && position_ids.dim() == 2 &&
                    position_ids.size(1) == 3 && position >= 0 &&
                    position < position_ids.size(0) && position < cos.size(0) &&
                    static_cast<uint64_t>(position) <= UINT32_MAX,
                "4B decode QK fusion position exceeds actual table/position owner rows");
    // The original M-RoPE loader reads 32 bytes for the three int32 values.
    // Keep logical bounds above; extra backing never extends the admitted row.
    const int64_t offset = position_ids.storage_offset();
    const uint64_t storage_bytes = position_ids.storage().nbytes();
    TORCH_CHECK(offset >= 0 && static_cast<uint64_t>(offset) <= storage_bytes / 4 &&
                    static_cast<uint64_t>(position) * 12 + 32 <=
                        storage_bytes - static_cast<uint64_t>(offset) * 4,
                "4B decode QK fusion requires initialized backing for the 32-byte position load");
    const auto* thw = position_ids.data_ptr<int32_t>() + position * 3;
    for (int axis = 0; axis < 3; ++axis) {
        TORCH_CHECK(thw[axis] >= 0 && thw[axis] < cos.size(0),
                    "4B decode QK fusion T/H/W coordinate exceeds actual RoPE table");
    }
    const c10::Half epsilon(static_cast<float>(eps));
    TORCH_CHECK(std::isfinite(eps) && eps > 0 &&
                    std::isfinite(static_cast<float>(epsilon)) &&
                    static_cast<float>(epsilon) > 0,
                "4B decode QK fusion requires positive finite FP16 epsilon");
    TORCH_CHECK(SPM_ALLOC.is_initialized(), "4B decode QK fusion requires initialized SPM");
    const uint32_t base = SPM_ALLOC.addr(0, 0);
    const std::array<uint32_t, 5> addresses{
        q_spm_addr, k_spm_addr, q_gamma_spm_addr, k_gamma_spm_addr, v_spm_addr};
    const std::array<uint64_t, 5> bytes{1024, 256, 256, 256, 256};
    for (size_t i = 0; i < (insert_kv ? 5u : 4u); ++i) {
        TORCH_CHECK(addresses[i] >= base && addresses[i] % 256 == 0 &&
                        static_cast<uint64_t>(addresses[i]) - base <=
                            SpmAllocator::SPM_USABLE - bytes[i],
                    "4B decode QK fusion requires aligned in-range absolute SPM operands");
        for (size_t j = 0; j < i; ++j) {
            TORCH_CHECK(static_cast<uint64_t>(addresses[i]) + bytes[i] <= addresses[j] ||
                            static_cast<uint64_t>(addresses[j]) + bytes[j] <= addresses[i],
                        "4B decode QK fusion data and gamma SPM regions must be disjoint");
        }
    }
    if (insert_kv) {
        const auto cache_ok = [](const at::Tensor& cache) {
            return cache.defined() &&
                cache.device().type() == c10::DeviceType::PrivateUse1 &&
                cache.scalar_type() == at::kHalf && cache.layout() == c10::Layout::Strided &&
                cache.is_contiguous() && cache.dim() == 7 &&
                cache.size(0) == 1 && cache.size(1) > 0 &&
                cache.size(1) <= UINT32_MAX / 16 && cache.size(2) == 1 &&
                cache.size(3) == 8 && cache.size(4) == 8 &&
                cache.size(5) == 16 && cache.size(6) == 16;
        };
        TORCH_CHECK(cache_ok(*k_cache) && cache_ok(*v_cache) &&
                        k_cache->sizes() == v_cache->sizes() &&
                        position < k_cache->size(1) * 16,
                    "4B decode QK/KV fusion requires matching packed FP16 TP8 caches and an in-capacity row");
    }
    const KernelId id = insert_kv ? KernelId::QWEN3VL_QK_NORM_MROPE_KV_INSERT_D128
                                  : KernelId::QWEN3VL_QK_NORM_MROPE_D128;
    Kernel_t* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel != nullptr,
                "4B decode QK payload missing; cold COMPLETE owner must retain the original route");
    std::vector<GraphDdrRegisterOperandSpec> operands{
            {{GraphDdrRegisterRole::ModelWeight, GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 6, 7}, cos},
            {{GraphDdrRegisterRole::ModelWeight, GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 8, 9}, sin},
            {{GraphDdrRegisterRole::SemanticInput, GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 12, 13}, position_ids}};
    if (insert_kv) {
        operands.push_back({{GraphDdrRegisterRole::KeyCache, GraphDdrRegisterAccess::Write,
                            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25}, *k_cache});
        operands.push_back({{GraphDdrRegisterRole::ValueCache, GraphDdrRegisterAccess::Write,
                            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 26, 27}, *v_cache});
    }
    auto register_writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id, operands);
    if (insert_kv) {
        // Preserve the typed gate before extra device-address queries. Written
        // cache spans must not overlap each other or any table/position input.
        std::array<uint64_t, 5> starts{}, lengths{};
        for (size_t i = 0; i < operands.size(); ++i) {
            const auto& owner = operands[i].tensor;
            starts[i] = RpuGetDevAddr(owner.data_ptr());
            lengths[i] = owner.nbytes();
            TORCH_CHECK(starts[i] && starts[i] < (UINT64_C(1) << 40) &&
                            starts[i] % 256 == 0 && lengths[i] > 0 &&
                            lengths[i] <= (UINT64_C(1) << 40) - starts[i],
                        "4B decode QK/KV DDR span exceeds aligned 40-bit addressing");
            if (i < 3) continue;
            for (size_t j = 0; j < i; ++j) {
                TORCH_CHECK(starts[i] + lengths[i] <= starts[j] ||
                                starts[j] + lengths[j] <= starts[i],
                            "4B decode QK/KV cache writes overlap another DDR operand");
            }
        }
    }
    for (const auto& operand : operands) {
        RpuKernelGraph::active().keep_alive(operand.tensor);
    }
    // Stable position keepalive is updated and sized-flushed by forward prologue.
    rpu_ddr_flush(cos.data_ptr());
    rpu_ddr_flush(sin.data_ptr());
    if (insert_kv) {
        rpu_ddr_flush(k_cache->data_ptr());
        rpu_ddr_flush(v_cache->data_ptr());
    }
    kernel->reset_regs();
    auto scalar = [kernel](int index, uint16_t value) { kernel->set_regs(index, value); };
    auto pair = [&scalar](int index, uint32_t value) {
        scalar(index, static_cast<uint16_t>(value));
        scalar(index + 1, static_cast<uint16_t>(value >> 16));
    };
    pair(0, static_cast<uint32_t>(position));
    scalar(2, 128); scalar(3, 4);
    pair(4, q_spm_addr); pair(10, k_spm_addr);
    pair(14, q_gamma_spm_addr); pair(16, k_gamma_spm_addr);
    if (insert_kv) pair(22, v_spm_addr);
    scalar(18, c10::Half(1.0f / 128).x); scalar(19, epsilon.x);
    scalar(20, 5); scalar(21, 8);
    scalar(64, 5); scalar(65, 1); scalar(66, 1);
    static const auto masks = rpu_compute_mrope_strobe_masks({24, 20, 20}, 128);
    for (int axis = 0; axis < 3; ++axis) {
        for (int lane = 0; lane < 16; ++lane) {
            rpu_set_legacy_scm_u16_checked(kernel, SCM_REG_OFFSET + axis * 16 + lane,
                                         masks[axis][lane], "4B decode QK fusion");
        }
    }
    register_writer.write(*kernel);
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {5, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
    if (insert_kv) {
        rpu_ddr_flush(k_cache->data_ptr());
        rpu_ddr_flush(v_cache->data_ptr());
    }
}

void rpu_launch_qwen3vl_4b_decode_qk_norm_mrope_spm_kernel(
    uint32_t q_spm_addr, uint32_t k_spm_addr,
    uint32_t q_gamma_spm_addr, uint32_t k_gamma_spm_addr,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& position_ids, int64_t position, double eps)
{
    launch_qwen3vl_4b_decode_qk_norm_mrope(q_spm_addr, k_spm_addr,
        q_gamma_spm_addr, k_gamma_spm_addr, cos, sin, position_ids, position, eps,
        0, nullptr, nullptr);
}

void rpu_launch_qwen3vl_4b_decode_qk_norm_mrope_kv_insert_spm_kernel(
    uint32_t q_spm_addr, uint32_t k_spm_addr, uint32_t v_spm_addr,
    uint32_t q_gamma_spm_addr, uint32_t k_gamma_spm_addr,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& position_ids, const at::Tensor& k_cache,
    const at::Tensor& v_cache, int64_t position, double eps)
{
    launch_qwen3vl_4b_decode_qk_norm_mrope(q_spm_addr, k_spm_addr,
        q_gamma_spm_addr, k_gamma_spm_addr, cos, sin, position_ids, position, eps,
        v_spm_addr, &k_cache, &v_cache);
}

// =============================================================================
// compute_mrope_strobe_masks
// =============================================================================
//
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
