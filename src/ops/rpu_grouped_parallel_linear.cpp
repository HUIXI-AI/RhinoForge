// Grouped tensor-parallel Linear for MoE expert projections.
//

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "rpu_ops.h"
#include "rpu_grouped_linear_tiling.h"
#include "rpu_spm_allocator.h"
#include "rpu_linear_tiling.h"

using namespace ::rhino_lkn;

namespace {

constexpr int64_t kModeInt8 =
    static_cast<int64_t>(GroupedParallelLinearWeightMode::INT8);
constexpr int64_t kModeFp16 =
    static_cast<int64_t>(GroupedParallelLinearWeightMode::FP16);
constexpr int64_t kModeFp8E4M3 =
    static_cast<int64_t>(GroupedParallelLinearWeightMode::FP8_E4M3);

void set_reg_u32(Kernel_t* kernel, int lo_reg, uint32_t value) {
    kernel->set_regs(lo_reg, static_cast<uint16_t>(value & 0xFFFF));
    kernel->set_regs(lo_reg + 1, static_cast<uint16_t>(value >> 16));
}

int64_t resolve_weight_mode(const at::Tensor& weight, int64_t requested) {
    if (requested < 0) {
        if (weight.scalar_type() == at::kHalf) return kModeFp16;
        TORCH_CHECK(
            false,
            "grouped_parallel_linear: raw uint8 storage requires explicit "
            "weight_dtype_mode=5 (fp8_e4m3)");
    }

    TORCH_CHECK(requested == kModeFp16 || requested == kModeInt8 || requested == kModeFp8E4M3,
                "grouped_parallel_linear: this Qwen3.5 integration supports "
                "only mode 1(int8), 2(fp16), or 5(fp8_e4m3); got ",
                requested);
    if (requested == kModeFp16) {
        TORCH_CHECK(weight.scalar_type() == at::kHalf,
                    "grouped_parallel_linear: mode 2 requires fp16 weight storage");
    } else if (requested == kModeInt8) {
        TORCH_CHECK(weight.scalar_type() == at::kChar,
                    "grouped_parallel_linear: mode 1 requires signed int8 weight storage");
    } else {
        TORCH_CHECK(weight.scalar_type() == at::kByte,
                    "grouped_parallel_linear: mode 5 requires raw uint8 "
                    "E4M3 storage");
    }
    return requested;
}

uint32_t checked_u32(uint64_t value, const char* what) {
    TORCH_CHECK(value <= std::numeric_limits<uint32_t>::max(),
                "grouped_parallel_linear: ", what, " exceeds u32: ", value);
    return static_cast<uint32_t>(value);
}

}  // namespace

static void launch_grouped_parallel_linear_spm(
    uint32_t input_spm_addr,
    const at::Tensor& packed_weight,
    uint32_t m_sizes_spm_addr,
    const c10::optional<at::Tensor>& scale_opt,
    uint32_t output_spm_addr,
    uint32_t bias_spm_addr,
    int64_t local_n,
    int64_t local_k,
    int64_t group,
    bool is_col_parallel,
    int num_cores,
    int64_t total_m,
    int64_t weight_dtype_mode,
    bool linear_acc32,
    bool expert_parallel_y4,
    bool routed_gate_up,
    uint32_t tokens_spm_addr,
    bool fused_swiglu_route = false,
    uint32_t routes_spm_addr = 0,
    bool scale_shuffle = false,
    const std::array<uint32_t, 4>* ordered_combine_offsets = nullptr) {
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "grouped_parallel_linear: num_cores must be in [1,8], got ",
                num_cores);
    TORCH_CHECK(group > 0 && group <= rpu_grouped_device_m::kMaxGroups,
                "grouped_parallel_linear: group must be in [1,32768] for m_sizes byte offsets, got ", group);
    TORCH_CHECK(local_n > 0 && local_n <= std::numeric_limits<uint32_t>::max() / 2 &&
                    local_n % 16 == 0,
                "grouped_parallel_linear: local_n must be positive, 16-aligned, "
                "and its FP16 row stride must fit u32; got ", local_n);
    TORCH_CHECK(local_k > 0 && local_k <= std::numeric_limits<uint16_t>::max() &&
                    local_k % 16 == 0,
                "grouped_parallel_linear: local_k must be positive, fit u16, "
                "and be divisible by 16; got ", local_k);
    TORCH_CHECK(local_k * static_cast<int64_t>(sizeof(c10::Half)) <=
                    std::numeric_limits<uint32_t>::max(),
                "grouped_parallel_linear: fp16 input row stride exceeds u32");
    TORCH_CHECK(local_n * static_cast<int64_t>(sizeof(c10::Half)) <=
                    std::numeric_limits<uint32_t>::max(),
                "grouped_parallel_linear: fp16 output row stride exceeds u32");

    TORCH_CHECK(packed_weight.defined() &&
                    packed_weight.device().type() == at::kPrivateUse1 &&
                    packed_weight.is_contiguous(),
                "grouped_parallel_linear: packed_weight must be a contiguous device tensor");
    const int64_t mode = resolve_weight_mode(packed_weight, weight_dtype_mode);
    TORCH_CHECK(!ordered_combine_offsets ||
                    ((mode == kModeFp16 || mode == kModeInt8) &&
                     !linear_acc32 && num_cores == 8 && group == 32 &&
                     local_n == 768 && local_k == 64 && !is_col_parallel &&
                     bias_spm_addr == 0 && !expert_parallel_y4 &&
                     !routed_gate_up && !fused_swiglu_route &&
                     scale_shuffle == (mode == kModeInt8)),
                "grouped_parallel_linear: ordered combine requires exact "
                "FP16/W8 ACC16 TP8/G32/N768/K64 Down without bias");
    TORCH_CHECK(!scale_shuffle ||
                    (mode == kModeInt8 && !linear_acc32 && num_cores == 8 &&
                     group == 32 && local_n == 768 && local_k == 64 &&
                     !is_col_parallel && bias_spm_addr == 0 &&
                     !expert_parallel_y4 && !routed_gate_up && !fused_swiglu_route),
                "grouped_parallel_linear: scale_shuffle requires W8 ACC16 "
                "TP8/G32/N768/K64 row-parallel Down without bias");
    TORCH_CHECK(!expert_parallel_y4 ||
                    ((mode == kModeFp16 || mode == kModeInt8) &&
                     !linear_acc32 && num_cores == 8 &&
                     group == 32 && local_n == 128 && local_k == 768 &&
                     is_col_parallel && bias_spm_addr == 0),
                "grouped_parallel_linear: expert_parallel_y4 requires FP16 or INT8 ACC16 "
                "TP8/G32/N128/K768 column GateUp without bias");
    const uint64_t weight_dwidth = packed_weight.element_size();
    const uint64_t weight_group_bytes =
        static_cast<uint64_t>(local_n) * static_cast<uint64_t>(local_k) *
        static_cast<uint64_t>(num_cores) * weight_dwidth;
    // GP34/35 carries the expert stride in 256-byte units. Check before
    // multiplying by group or staging DDR ownership, so large u32-N shapes
    // cannot wrap the total byte extent or reach a truncated address pair.
    TORCH_CHECK(weight_group_bytes % 256 == 0 &&
                    weight_group_bytes / 256 <= std::numeric_limits<uint32_t>::max(),
                "grouped_parallel_linear: weight expert stride exceeds shift8 u32");
    const uint64_t expected_weight_bytes =
        static_cast<uint64_t>(group) * weight_group_bytes;
    TORCH_CHECK(static_cast<uint64_t>(packed_weight.nbytes()) == expected_weight_bytes,
                "grouped_parallel_linear: packed_weight has ", packed_weight.nbytes(),
                " bytes, expected ", expected_weight_bytes,
                " for group/local_n/local_k/tp=", group, "/", local_n, "/",
                local_k, "/", num_cores);

    uint32_t scale_group_bytes = 0;
    if (mode != kModeFp16) {
        TORCH_CHECK(scale_opt.has_value() && scale_opt->defined(),
                    "grouped_parallel_linear: 8-bit weights require fp16 scale");
        const at::Tensor& scale = *scale_opt;
        TORCH_CHECK(scale.device().type() == at::kPrivateUse1 &&
                        scale.scalar_type() == at::kHalf && scale.is_contiguous(),
                    "grouped_parallel_linear: scale must be contiguous fp16 on device");
        const uint64_t global_n = static_cast<uint64_t>(local_n) *
                                  (is_col_parallel ? num_cores : 1);
        const uint64_t expected_scale_bytes =
            static_cast<uint64_t>(group) * ((global_n + 127) / 128 * 128) * sizeof(c10::Half);
        TORCH_CHECK(static_cast<uint64_t>(scale.nbytes()) == expected_scale_bytes,
                    "grouped_parallel_linear: scale has ", scale.nbytes(),
                    " bytes, expected ", expected_scale_bytes,
                    " ([group, global_n] fp16)");
        scale_group_bytes = checked_u32(((global_n + 127) / 128 * 128) * sizeof(c10::Half),
                                        "scale group byte step");
    } else {
        TORCH_CHECK(!scale_opt.has_value() || !scale_opt->defined(),
                    "grouped_parallel_linear: fp16 mode does not consume scale");
    }

    const auto wtype = mode == kModeFp16 ? rpu_pl_tiling::WeightType::Fp16
        : mode == kModeFp8E4M3 ? rpu_pl_tiling::WeightType::Fp8E4M3
                              : rpu_pl_tiling::WeightType::Int8;
    const auto dispatch = rpu_grouped_tiling::select_dispatch(
        total_m, local_n, local_k, static_cast<int>(mode), linear_acc32,
        2, 0, 0, group);
    const auto& tile = dispatch.tile;
    const bool specialized = ordered_combine_offsets || scale_shuffle ||
        fused_swiglu_route || routed_gate_up || expert_parallel_y4;
    const KernelId kernel_id =
        ordered_combine_offsets
            ? (mode == kModeFp16
                ? KernelId::LINGBOT2_GROUPED_DOWN_FP16_ORDERED_COMBINE
                : KernelId::LINGBOT2_GROUPED_DOWN_W8_ORDERED_COMBINE)
            : scale_shuffle
            ? KernelId::LINGBOT2_GROUPED_DOWN_W8A16_ACC16_SCALE_SHUFFLE
            : fused_swiglu_route
            ? (mode == kModeFp16
                ? KernelId::LINGBOT2_GROUPED_GATEUP_ROUTED_FP16_SWIGLU_ROUTE
                : KernelId::LINGBOT2_GROUPED_GATEUP_ROUTED_W8_SWIGLU_ROUTE)
            : routed_gate_up
            ? (mode == kModeFp16
                ? KernelId::LINGBOT2_GROUPED_GATEUP_ROUTED_EXPERT_Y4_ACC16
                : KernelId::LINGBOT2_GROUPED_GATEUP_ROUTED_W8A16_EXPERT_Y4_ACC16)
            : expert_parallel_y4
            ? (mode == kModeFp16
                ? KernelId::LINGBOT2_GROUPED_GATEUP_EXPERT_Y4_ACC16
                : KernelId::LINGBOT2_GROUPED_GATEUP_W8A16_EXPERT_Y4_ACC16)
            : dispatch.kind == rpu_grouped_tiling::DispatchKind::Fp16Acc16DeviceM
            ? KernelId::GROUPED_PL_FP16_ACC16_DEVICE_M
            : dispatch.kind == rpu_grouped_tiling::DispatchKind::W8Acc32DeviceM
            ? KernelId::GROUPED_PL_W8_ACC32_DEVICE_M
            : dispatch.kind == rpu_grouped_tiling::DispatchKind::Fp16Acc32N80DeviceM
            ? KernelId::GROUPED_PL_FP16_ACC32_N80_DEVICE_M
            : dispatch.kind == rpu_grouped_tiling::DispatchKind::Fp16Acc32N32DeviceM
            ? KernelId::GROUPED_PL_FP16_ACC32_N32_DEVICE_M
            : dispatch.kind == rpu_grouped_tiling::DispatchKind::W8Acc16N48DeviceM
            ? KernelId::GROUPED_PL_W8_ACC16_N48_DEVICE_M
            : dispatch.kind == rpu_grouped_tiling::DispatchKind::W8Acc32N48DeviceM
            ? KernelId::GROUPED_PL_W8_ACC32_N48_DEVICE_M
            : grouped_parallel_linear_kernel_id(wtype, linear_acc32, tile);
    Kernel_t* kernel = nullptr;
    if (routed_gate_up || scale_shuffle || ordered_combine_offsets) {
        kernel = GET_KERNEL(kernel_id);
        TORCH_CHECK(kernel != nullptr,
                    ordered_combine_offsets ? "lingbot2_down_ordered_combine: optional precision payload missing"
                    : scale_shuffle ? "lingbot2_down_scale_shuffle: optional payload missing"
                                    : "lingbot2_routed_gateup: optional dtype payload missing");
    }
    std::vector<GraphDdrRegisterOperandSpec> ddr_operands{
        GraphDdrRegisterOperandSpec{
            GraphDdrRegisterAbi{
                mode == kModeFp16
                    ? GraphDdrRegisterRole::ModelWeight
                    : (mode == kModeInt8
                        ? GraphDdrRegisterRole::Int8ModelWeight
                        : GraphDdrRegisterRole::Fp8ModelWeight),
                GraphDdrRegisterAccess::Read,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                2, 3},
            packed_weight}};
    if (mode != kModeFp16) {
        ddr_operands.push_back(GraphDdrRegisterOperandSpec{
            GraphDdrRegisterAbi{
                GraphDdrRegisterRole::PerChannelScale,
                GraphDdrRegisterAccess::Read,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                20, 21},
            *scale_opt});
    }
    auto register_writer =
        RpuKernelGraph::active().stage_kernel_ddr_registers(
            kernel_id, std::move(ddr_operands));
    rpu_ddr_flush(packed_weight.data_ptr());
    if (mode != kModeFp16) {
        rpu_ddr_flush(scale_opt->data_ptr());
    }

    if (!kernel) kernel = GET_KERNEL(kernel_id);
    TORCH_CHECK(kernel != nullptr,
                "grouped_parallel_linear: required kernel missing from rhino op lib");
    kernel->reset_regs();
    // GP9 is unused by both static and device-M grouped programs. total_m is
    // host selection/descriptor metadata; runtime expert M comes from m_sizes.
    rpu_pl_tiling::set_parallel_linear_regs(*kernel, {
        input_spm_addr, 0, output_spm_addr, bias_spm_addr,
        0, static_cast<uint32_t>(local_n), static_cast<uint32_t>(local_k),
        static_cast<uint16_t>(bias_spm_addr != 0), static_cast<uint16_t>(num_cores),
        static_cast<uint32_t>(local_k * sizeof(c10::Half)),
        static_cast<uint32_t>((fused_swiglu_route ? 64 : local_n) * sizeof(c10::Half)),
        0, static_cast<uint16_t>(is_col_parallel), static_cast<uint16_t>(mode)});
    TORCH_CHECK(weight_group_bytes % 256 == 0 && scale_group_bytes % 256 == 0,
                "grouped_parallel_linear: expert strides must be 256-byte aligned");
    kernel->set_regs(30, static_cast<uint16_t>(group));
    set_reg_u32(kernel, 32, m_sizes_spm_addr);
    set_reg_u32(kernel, 34, checked_u32(weight_group_bytes / 256, "weight expert stride"));
    set_reg_u32(kernel, 36, scale_group_bytes / 256);
    set_reg_u32(kernel, 38, 0);
    if (routed_gate_up) set_reg_u32(kernel, 48, tokens_spm_addr);
    if (fused_swiglu_route) set_reg_u32(kernel, 50, routes_spm_addr);
    if (ordered_combine_offsets) {
        for (size_t i = 0; i < ordered_combine_offsets->size(); ++i)
            set_reg_u32(kernel, 48 + 2 * static_cast<int>(i), (*ordered_combine_offsets)[i]);
    }

    const uint64_t grid_x64 = fused_swiglu_route
        ? 1 : (static_cast<uint64_t>(local_n) + (specialized ? 64 : tile.n_tile) - 1) /
                  (specialized ? 64 : tile.n_tile);
    TORCH_CHECK(grid_x64 <= std::numeric_limits<uint16_t>::max(),
                "grouped_parallel_linear: grid.x exceeds u16");
    const uint16_t grid_x = static_cast<uint16_t>(grid_x64);
    const uint16_t grid_y = fused_swiglu_route ? 8 : expert_parallel_y4 ? 4 : 1;
    kernel->set_regs(64, grid_x);
    kernel->set_regs(65, grid_y);
    kernel->set_regs(66, static_cast<uint16_t>(1));
    register_writer.write(*kernel);

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back(static_cast<uint8_t>(i));
    wq->enqueu_kernel(*kernel, {grid_x, grid_y, 1}, cores);
}

void rpu_launch_grouped_parallel_linear_spm_kernel(
    uint32_t input_spm_addr, const at::Tensor& packed_weight,
    uint32_t m_sizes_spm_addr, const c10::optional<at::Tensor>& scale_opt,
    uint32_t output_spm_addr, uint32_t bias_spm_addr,
    int64_t local_n, int64_t local_k, int64_t group, bool is_col_parallel,
    int num_cores, int64_t total_m, int64_t weight_dtype_mode, bool linear_acc32,
    bool expert_parallel_y4) {
    launch_grouped_parallel_linear_spm(
        input_spm_addr, packed_weight, m_sizes_spm_addr, scale_opt,
        output_spm_addr, bias_spm_addr, local_n, local_k, group,
        is_col_parallel, num_cores, total_m, weight_dtype_mode, linear_acc32,
        expert_parallel_y4, false, 0);
}

// Exact T51/R204/G32/N128/K768/TP8 profile. The caller provides canonical
// expert-sorted token IDs [0,51), <=51 rows/expert, and counts summing to204.
// Read the original token rows directly; all K256 ACC16 boundaries and the
// packed FP16/INT8+scale ABI remain the same as the existing expert-y4 path.
void rpu_launch_lingbot2_grouped_gateup_routed_spm_kernel(
    uint32_t input_spm_addr, uint32_t tokens_spm_addr,
    const at::Tensor& packed_weight, uint32_t m_sizes_spm_addr,
    const c10::optional<at::Tensor>& scale_opt, uint32_t output_spm_addr,
    int64_t weight_dtype_mode) {
    TORCH_CHECK(weight_dtype_mode == kModeFp16 || weight_dtype_mode == kModeInt8,
                "lingbot2_routed_gateup: fixed profile requires explicit FP16 or INT8 weights");
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "lingbot2_routed_gateup: uninitialized SPM");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint64_t>, 4> intervals{{
        {input_spm_addr, 51 * 768 * 2}, {tokens_spm_addr, 256 * 2},
        {m_sizes_spm_addr, 128 * 2}, {output_spm_addr, 204 * 128 * 2}}};
    for (size_t i = 0; i < intervals.size(); ++i) {
        const auto [address, bytes] = intervals[i];
        TORCH_CHECK(address >= base && address % 32 == 0 &&
                        bytes <= SpmAllocator::SPM_USABLE &&
                        uint64_t(address) - base <= SpmAllocator::SPM_USABLE - bytes,
                    "lingbot2_routed_gateup: invalid complete aligned absolute SPM interval");
        for (size_t j = 0; j < i; ++j) {
            TORCH_CHECK(uint64_t(address) + bytes <= intervals[j].first ||
                            uint64_t(intervals[j].first) + intervals[j].second <= address,
                        "lingbot2_routed_gateup: SPM operands must be disjoint");
        }
    }
    launch_grouped_parallel_linear_spm(
        input_spm_addr, packed_weight, m_sizes_spm_addr, scale_opt,
        output_spm_addr, 0, 128, 768, 32, true, 8,
        204, weight_dtype_mode, false, true, true, tokens_spm_addr);
}

void rpu_launch_lingbot2_grouped_gateup_routed_swiglu_route_spm_kernel(
    uint32_t input_spm_addr, uint32_t tokens_spm_addr, uint32_t routes_spm_addr,
    const at::Tensor& packed_weight, uint32_t m_sizes_spm_addr,
    const c10::optional<at::Tensor>& scale_opt, uint32_t output_spm_addr,
    int64_t weight_dtype_mode) {
    TORCH_CHECK(weight_dtype_mode == kModeFp16 || weight_dtype_mode == kModeInt8,
                "lingbot2_routed_gateup_activation: fixed profile requires explicit FP16 or INT8 weights");
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "lingbot2_routed_gateup_activation: uninitialized SPM");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint64_t>, 5> intervals{{
        {input_spm_addr, 51 * 768 * 2}, {tokens_spm_addr, 256 * 2},
        {routes_spm_addr, 64 * 32 * 2}, {m_sizes_spm_addr, 128 * 2},
        {output_spm_addr, 204 * 64 * 2}}};
    for (size_t i = 0; i < intervals.size(); ++i) {
        const auto [address, bytes] = intervals[i];
        TORCH_CHECK(address >= base && address % 32 == 0 &&
                        bytes <= SpmAllocator::SPM_USABLE &&
                        uint64_t(address) - base <= SpmAllocator::SPM_USABLE - bytes,
                    "lingbot2_routed_gateup_activation: invalid complete aligned absolute SPM interval");
        for (size_t j = 0; j < i; ++j) {
            TORCH_CHECK(uint64_t(address) + bytes <= intervals[j].first ||
                            uint64_t(intervals[j].first) + intervals[j].second <= address,
                        "lingbot2_routed_gateup_activation: SPM operands must be disjoint");
        }
    }
    launch_grouped_parallel_linear_spm(
        input_spm_addr, packed_weight, m_sizes_spm_addr, scale_opt,
        output_spm_addr, 0, 128, 768, 32, true, 8,
        204, weight_dtype_mode, false, true, true, tokens_spm_addr,
        true, routes_spm_addr);
}

void rpu_launch_lingbot2_grouped_down_w8_scale_shuffle_spm_kernel(
    uint32_t input_spm_addr, const at::Tensor& packed_weight,
    uint32_t m_sizes_spm_addr, const at::Tensor& scale, uint32_t output_spm_addr) {
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "lingbot2_down_scale_shuffle: uninitialized SPM");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint64_t>, 3> intervals{{
        {input_spm_addr, 204 * 64 * 2}, {m_sizes_spm_addr, 128 * 2},
        {output_spm_addr, 204 * 768 * 2}}};
    for (size_t i = 0; i < intervals.size(); ++i) {
        const auto [address, bytes] = intervals[i];
        TORCH_CHECK(address >= base && address % 32 == 0 &&
                        bytes <= SpmAllocator::SPM_USABLE &&
                        uint64_t(address) - base <= SpmAllocator::SPM_USABLE - bytes,
                    "lingbot2_down_scale_shuffle: invalid complete aligned absolute SPM interval");
        for (size_t j = 0; j < i; ++j) {
            TORCH_CHECK(uint64_t(address) + bytes <= intervals[j].first ||
                            uint64_t(intervals[j].first) + intervals[j].second <= address,
                        "lingbot2_down_scale_shuffle: SPM operands must be disjoint");
        }
    }
    launch_grouped_parallel_linear_spm(
        input_spm_addr, packed_weight, m_sizes_spm_addr, scale,
        output_spm_addr, 0, 768, 64, 32, false, 8, 204, kModeInt8,
        false, false, false, 0, false, 0, true);
}

void rpu_launch_lingbot2_grouped_down_ordered_combine_spm_kernel(
    uint32_t input_spm_addr, const at::Tensor& packed_weight,
    uint32_t m_sizes_spm_addr, const c10::optional<at::Tensor>& scale_opt,
    uint32_t taskout_spm_addr, uint32_t inverse_spm_addr,
    uint32_t shared_spm_addr, uint32_t acc_spm_addr,
    int64_t weight_dtype_mode) {
    TORCH_CHECK(weight_dtype_mode == kModeFp16 || weight_dtype_mode == kModeInt8,
                "Lingbot2 ordered Down requires explicit FP16 or INT8 mode");
    const auto complete_owner = [](const at::Tensor& tensor) {
        if (!(tensor.defined() &&
              tensor.device().type() == c10::DeviceType::PrivateUse1 &&
              tensor.layout() == c10::Layout::Strided && tensor.is_contiguous() &&
              tensor.storage_offset() >= 0)) return false;
        const uint64_t storage_bytes = tensor.storage().nbytes();
        const uint64_t item_bytes = tensor.element_size();
        const uint64_t offset = static_cast<uint64_t>(tensor.storage_offset());
        return item_bytes != 0 && offset <= storage_bytes / item_bytes &&
               tensor.nbytes() <= storage_bytes - offset * item_bytes;
    };
    const auto ddr_span = [](const at::Tensor& tensor) {
        const uint64_t address = RpuGetDevAddr(tensor.data_ptr());
        constexpr uint64_t limit = uint64_t{1} << 40;
        return address != 0 && (address & 255u) == 0 && address < limit &&
               tensor.nbytes() <= limit - address;
    };
    TORCH_CHECK(complete_owner(packed_weight) && ddr_span(packed_weight),
                "Lingbot2 ordered Down requires a complete aligned weight owner");
    if (scale_opt.has_value() && scale_opt->defined()) {
        TORCH_CHECK(complete_owner(*scale_opt) && ddr_span(*scale_opt) &&
                        scale_opt->device() == packed_weight.device(),
                    "Lingbot2 ordered Down requires a complete same-device aligned scale owner");
    }
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "Lingbot2 ordered Down requires initialized SPM");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint64_t>, 6> ranges{{
        {input_spm_addr, 204 * 64 * 2}, {m_sizes_spm_addr, 128 * 2},
        {taskout_spm_addr, 204 * 768 * 2}, {inverse_spm_addr, 64 * 32 * 2},
        {shared_spm_addr, 51 * 768 * 2}, {acc_spm_addr, 51 * 768 * 2}}};
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto [address, bytes] = ranges[i];
        TORCH_CHECK(address >= base && (address & 31u) == 0 &&
                        bytes <= SpmAllocator::SPM_USABLE &&
                        uint64_t{address} - base <= SpmAllocator::SPM_USABLE - bytes &&
                        uint64_t{address} + bytes <= (uint64_t{1} << 32),
                    "Lingbot2 ordered Down requires complete aligned absolute SPM intervals");
        for (size_t j = 0; j < i; ++j) {
            TORCH_CHECK(uint64_t{address} + bytes <= ranges[j].first ||
                            uint64_t{ranges[j].first} + ranges[j].second <= address,
                        "Lingbot2 ordered Down SPM operands must be disjoint");
        }
    }
    for (int core = 0; core < 8; ++core) {
        const uint64_t core_base = SPM_ALLOC.addr(core, 0);
        for (const auto& [address, bytes] : ranges) {
            const uint64_t core_address = core_base + address - base;
            TORCH_CHECK((core_address & 31u) == 0 &&
                            core_address + bytes <= (uint64_t{1} << 32),
                        "Lingbot2 ordered Down per-core SPM span exceeds aligned u32 storage");
        }
    }
    const KernelId id = weight_dtype_mode == kModeFp16
        ? KernelId::LINGBOT2_GROUPED_DOWN_FP16_ORDERED_COMBINE
        : KernelId::LINGBOT2_GROUPED_DOWN_W8_ORDERED_COMBINE;
    TORCH_CHECK(KernelCache::instance().has_loaded(id),
                "Lingbot2 ordered Down optional precision payload is absent");
    TORCH_CHECK(GET_QUEUE(8) != nullptr, "Lingbot2 ordered Down eight-core queue is absent");
    const std::array<uint32_t, 4> local_offsets{{
        static_cast<uint32_t>(inverse_spm_addr - base),
        static_cast<uint32_t>(shared_spm_addr - base),
        static_cast<uint32_t>(acc_spm_addr - base),
        static_cast<uint32_t>(taskout_spm_addr - base)}};
    // Reuse the original typed weight/scale lease and Down register setup.
    // GP4 is the absolute output; GP54 is its checked core-local epilogue alias.
    // GP32 carries m_sizes independently of the four GP48..55 epilogue offsets.
    launch_grouped_parallel_linear_spm(
        input_spm_addr, packed_weight, m_sizes_spm_addr, scale_opt,
        taskout_spm_addr, 0, 768, 64, 32, false, 8, 204, weight_dtype_mode,
        false, false, false, 0, false, 0, weight_dtype_mode == kModeInt8,
        &local_offsets);
}

std::array<int64_t, 3> rpu_select_grouped_parallel_linear_uint4_tile(
    int64_t m_total, int64_t num_experts, int64_t local_n, int64_t local_k,
    int64_t group_size, bool force_acc32) {
    if (group_size != 32 && group_size != 64 && group_size != 128)
        throw std::invalid_argument("Grouped UINT4 requires group_size 32, 64, or 128");
    const auto tile = rpu_grouped_tiling::select_uint4(
        m_total, num_experts, local_n, local_k, static_cast<int>(group_size), force_acc32);
    return {tile.n, tile.m, tile.k};
}

void rpu_launch_grouped_parallel_linear_uint4_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& packed_weight,
    uint32_t m_sizes_spm_addr,
    const at::Tensor& scale,
    const at::Tensor& negative_zero,
    uint32_t output_spm_addr,
    int64_t local_n,
    int64_t local_k,
    int64_t num_experts,
    bool is_col_parallel,
    int num_cores,
    int64_t group_size,
    int64_t m_total, bool force_acc32) {
    // Grouped UINT4 uses the unified ParallelLinear registers and N32K32
    // weight storage, followed by the grouped expert descriptor. S/Z metadata
    // is controller-striped Q4/N64, matching the generated per-group payload.
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "grouped UINT4: num_cores must be in [1,8]");
    TORCH_CHECK(num_experts > 0 && num_experts <= rpu_grouped_device_m::kMaxGroups,
                "grouped UINT4: num_experts must be in [1,32768] for m_sizes byte offsets");
    TORCH_CHECK(m_total > 0 && m_total <= UINT16_MAX,
                "grouped UINT4: packed assignment rows must be in [1,65535]");
    TORCH_CHECK(local_n > 0 && local_n <= UINT32_MAX / 2 && local_n % 16 == 0,
                "grouped UINT4: local_n must be positive, 16-aligned and its "
                "FP16 row stride must fit u32");
    TORCH_CHECK(local_k > 0 && local_k <= UINT16_MAX && local_k % 32 == 0,
                "grouped UINT4: local_k must be positive, 32-aligned and fit u16");
    TORCH_CHECK((group_size == 32 || group_size == 64 || group_size == 128) &&
                    local_k % group_size == 0,
                "grouped UINT4: group_size must be 32/64/128 and divide local_k");

    const uint64_t global_n = static_cast<uint64_t>(local_n) *
                              (is_col_parallel ? num_cores : 1);
    const uint64_t global_k = static_cast<uint64_t>(local_k) *
                              (is_col_parallel ? 1 : num_cores);
    const uint32_t weight_expert_bytes = checked_u32(
        static_cast<uint64_t>((local_n + 31) / 32 * 32) * local_k * num_cores / 2,
        "UINT4 N32K32 weight expert byte step");
    const auto metadata = rpu_grouped_tiling::uint4_metadata_layout(
        local_n, local_k, group_size, num_cores);
    const uint32_t scale_group_block_bytes = checked_u32(
        metadata.group_block_bytes, "UINT4 scale Q4 block byte step");
    const uint32_t scale_expert_bytes = checked_u32(
        metadata.expert_bytes, "UINT4 scale expert byte step");
    TORCH_CHECK(packed_weight.defined() &&
                    packed_weight.device().type() == at::kPrivateUse1 &&
                    packed_weight.scalar_type() == at::kByte &&
                    packed_weight.is_contiguous() &&
                    static_cast<uint64_t>(packed_weight.nbytes()) ==
                        static_cast<uint64_t>(num_experts) * weight_expert_bytes,
                "grouped UINT4: weight must be contiguous device uint8 with E*N*K/2 bytes");
    const auto check_scale_owner = [&](const at::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.defined() && tensor.device().type() == at::kPrivateUse1 &&
                        tensor.scalar_type() == at::kHalf && tensor.is_contiguous() &&
                        tensor.dim() == 6 && tensor.size(0) == num_experts &&
                        tensor.size(1) == metadata.q_blocks &&
                        tensor.size(2) == metadata.n_blocks &&
                        tensor.size(3) == num_cores && tensor.size(4) == 4 &&
                        tensor.size(5) == 64 &&
                        static_cast<uint64_t>(tensor.nbytes()) ==
                            static_cast<uint64_t>(num_experts) * scale_expert_bytes,
                    "grouped UINT4: ", name,
                    " must use rhino-ops 135b93fc contiguous device FP16 "
                    "[E,Qblock4,Nblock64,core,4,64] metadata");
    };
    check_scale_owner(scale, "scale");
    check_scale_owner(negative_zero, "negative_zero");
    const auto tile = rpu_select_grouped_parallel_linear_uint4_tile(
        m_total, num_experts, local_n, local_k, group_size, force_acc32);
    const KernelId kernel_id = grouped_parallel_linear_kernel_id(
        rpu_pl_tiling::WeightType::Uint4, force_acc32,
        rpu_pl_tiling::TilePick{static_cast<int>(tile[0]), static_cast<int>(tile[1]),
                               static_cast<int>(tile[2])});
    const uint64_t grid_x64 = (static_cast<uint64_t>(local_n) + tile[0] - 1) / tile[0];
    TORCH_CHECK(grid_x64 <= UINT16_MAX, "grouped UINT4: grid.x must fit u16");
    const uint16_t grid_x = static_cast<uint16_t>(grid_x64);
    // Typed staging validates W256 / S,Z num_cores*512 alignment and retains
    // the exact W/S/Z storage through BUILD/REPLAY. Do not write raw DDR regs.
    auto register_writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
        kernel_id, {
            {{GraphDdrRegisterRole::PackedUint4ModelWeight, GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, packed_weight},
            {{GraphDdrRegisterRole::PerGroupScale, GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale},
            {{GraphDdrRegisterRole::NegativeZeroPoint, GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 22, 23}, negative_zero},
        });
    rpu_ddr_flush(packed_weight.data_ptr());
    rpu_ddr_flush(scale.data_ptr());
    rpu_ddr_flush(negative_zero.data_ptr());
    Kernel_t* kernel = GET_KERNEL(kernel_id);
    TORCH_CHECK(kernel != nullptr, "grouped UINT4: required generated kernel is missing");
    kernel->reset_regs();
    rpu_pl_tiling::set_parallel_linear_w4_group_regs(*kernel, {
        {input_spm_addr, 0, output_spm_addr, 0,
         static_cast<uint32_t>(m_total), static_cast<uint32_t>(local_n),
         static_cast<uint32_t>(local_k), 0, static_cast<uint16_t>(num_cores),
         static_cast<uint32_t>(local_k * 2), static_cast<uint32_t>(local_n * 2),
         0, static_cast<uint16_t>(is_col_parallel), 8},
        static_cast<uint16_t>(group_size), static_cast<uint16_t>(local_k / group_size),
        scale_group_block_bytes, 0, true});
    kernel->set_regs(30, static_cast<uint16_t>(num_experts));
    set_reg_u32(kernel, 32, m_sizes_spm_addr);
    set_reg_u32(kernel, 34, weight_expert_bytes / 256);
    set_reg_u32(kernel, 36, scale_expert_bytes / 256);
    set_reg_u32(kernel, 38, scale_expert_bytes / 256);
    kernel->set_regs(64, grid_x);
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(1));
    register_writer.write(*kernel);
    auto* queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back(static_cast<uint8_t>(i));
    queue->enqueu_kernel(*kernel, {grid_x, 1, 1}, cores);
}

// Conformance-only wrapper. It returns each core's local output (column shard
// or row-parallel partial), isolating this kernel from all-gather/all-reduce.
at::Tensor rpu_grouped_parallel_linear_test(
    const at::Tensor& input_per_core,
    const at::Tensor& packed_weight,
    const at::Tensor& m_sizes,
    const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& bias_per_core,
    int64_t local_n,
    int64_t local_k,
    bool is_col_parallel,
    int64_t num_cores,
    int64_t weight_dtype_mode,
    bool linear_acc32) {
    RECORD_FUNCTION("rpu::grouped_parallel_linear_test", {});
    TORCH_CHECK(RpuKernelGraph::active().state() == RpuKernelGraph::State::PASSTHROUGH,
                "grouped_parallel_linear_test is an immediate conformance wrapper; "
                "it cannot be captured or replayed");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "grouped_parallel_linear_test: num_cores must be in [1,8]");
    TORCH_CHECK(input_per_core.device().type() == at::kPrivateUse1 &&
                    input_per_core.scalar_type() == at::kHalf &&
                    input_per_core.is_contiguous() && input_per_core.dim() == 3,
                "grouped_parallel_linear_test: input_per_core must be contiguous "
                "fp16 device [num_cores,M,local_k]");
    TORCH_CHECK(input_per_core.size(0) == num_cores &&
                    input_per_core.size(2) == local_k,
                "grouped_parallel_linear_test: input shape/dims mismatch");
    TORCH_CHECK(m_sizes.device().type() == at::kPrivateUse1 &&
                    m_sizes.scalar_type() == at::kShort &&
                    m_sizes.is_contiguous() && m_sizes.dim() == 1,
                "grouped_parallel_linear_test: m_sizes must be contiguous int16 "
                "device [group]");
    const int64_t group = m_sizes.numel();
    TORCH_CHECK(group > 0 && group <= rpu_grouped_device_m::kMaxGroups,
                "grouped_parallel_linear_test: experts must be in [1,32768]");
    const int64_t m = input_per_core.size(1);
    TORCH_CHECK(m > 0 && m <= UINT16_MAX,
                "grouped_parallel_linear_test: packed assignment rows must be in [1,65535]");
    const auto counts_cpu = m_sizes.cpu();
    int64_t count_sum = 0;
    for (int64_t i = 0; i < group; ++i) {
        const int16_t count = counts_cpu.data_ptr<int16_t>()[i];
        TORCH_CHECK(count >= 0, "grouped_parallel_linear_test: negative expert row count");
        count_sum += count;
    }
    TORCH_CHECK(count_sum == m, "grouped_parallel_linear_test: sum(m_sizes) must equal M");

    if (bias_per_core.has_value() && bias_per_core->defined()) {
        TORCH_CHECK(bias_per_core->device().type() == at::kPrivateUse1 &&
                        bias_per_core->scalar_type() == at::kHalf &&
                        bias_per_core->is_contiguous() &&
                        bias_per_core->dim() == 2 &&
                        bias_per_core->size(0) == num_cores &&
                        bias_per_core->size(1) == local_n,
                    "grouped_parallel_linear_test: bias_per_core must be contiguous "
                    "fp16 device [num_cores,local_n]");
    }

    // Conformance setup mirrors cold installation. The allocator guarantees
    // 32-byte alignment, but shifted DDR operands need a 256-byte base.
    const auto aligned_test_owner = [](const at::Tensor& source) {
        const uint64_t address = RpuGetDevAddr(source.data_ptr());
        TORCH_CHECK(address != 0, "grouped_parallel_linear_test: missing device address");
        if (address % 256 == 0) return source;
        const int64_t element_bytes = source.element_size();
        auto backing = at::empty({source.numel() + 256 / element_bytes}, source.options());
        const uint64_t base = RpuGetDevAddr(backing.data_ptr());
        TORCH_CHECK(base != 0, "grouped_parallel_linear_test: missing aligned backing");
        const int64_t offset = (256 - base % 256) % 256;
        auto owner = backing.narrow(0, offset / element_bytes, source.numel()).view(source.sizes());
        owner.copy_(source);
        return owner;
    };
    const auto weight_run = aligned_test_owner(packed_weight);
    c10::optional<at::Tensor> scale_run = scale;
    if (scale && scale->defined()) scale_run = aligned_test_owner(*scale);

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    const int64_t input_bytes = m * local_k * sizeof(c10::Half);
    // Grouped kernels read expert counts in vectors of 16. Initialize its masked tail
    // as well as satisfying the immediate DMA's 16-byte granularity.
    const int64_t m_sizes_dma_elements = ((group + 15) / 16) * 16;
    const int64_t m_sizes_bytes = m_sizes_dma_elements * sizeof(int16_t);
    const int64_t output_bytes = m * local_n * sizeof(c10::Half);
    const int64_t bias_bytes = local_n * sizeof(c10::Half);

    at::Tensor m_sizes_dma = m_sizes;
    if (m_sizes_dma_elements != group) {
        m_sizes_dma = at::zeros({m_sizes_dma_elements}, m_sizes.options());
        m_sizes_dma.narrow(0, 0, group).copy_(m_sizes);
    }

    using AR = SpmAllocator::AllocRequest;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{input_bytes, 1, 2},
        AR{m_sizes_bytes, 1, 2},
        AR{bias_bytes, 1, 2},
        AR{output_bytes, 2, 3},
    });
    const uint32_t input_addr = SPM_ALLOC.addr(0, offsets[0]);
    const uint32_t m_sizes_addr = SPM_ALLOC.addr(0, offsets[1]);
    const uint32_t bias_addr = SPM_ALLOC.addr(0, offsets[2]);
    const uint32_t output_addr = SPM_ALLOC.addr(0, offsets[3]);

    try {
        rpu_launch_ddr_scatter_spm_dma_immediate(
            const_cast<c10::Half*>(input_per_core.data_ptr<c10::Half>()),
            m * local_k, input_bytes, input_addr, static_cast<int>(num_cores));
        rpu_launch_ddr_broadcast_spm_dma_immediate(
            reinterpret_cast<c10::Half*>(m_sizes_dma.data_ptr<int16_t>()),
            m_sizes_dma_elements, m_sizes_addr, static_cast<int>(num_cores));

        uint32_t launch_bias_addr = 0;
        if (bias_per_core.has_value() && bias_per_core->defined()) {
            rpu_launch_ddr_scatter_spm_dma_immediate(
                const_cast<c10::Half*>(bias_per_core->data_ptr<c10::Half>()),
                local_n, bias_bytes, bias_addr, static_cast<int>(num_cores));
            launch_bias_addr = bias_addr;
        }

        rpu_launch_grouped_parallel_linear_spm_kernel(
            input_addr, weight_run, m_sizes_addr, scale_run, output_addr,
            launch_bias_addr, local_n, local_k, group, is_col_parallel,
            static_cast<int>(num_cores), m, weight_dtype_mode, linear_acc32);

        auto output = at::empty({num_cores, m, local_n}, input_per_core.options());
        rpu_launch_spm_scatter_ddr_dma_immediate(
            output_addr, output.data_ptr<c10::Half>(), m * local_n,
            output_bytes, static_cast<int>(num_cores));
        SPM_ALLOC.reset_temporary();
        return output;
    } catch (...) {
        SPM_ALLOC.reset_temporary();
        throw;
    }
}

bool rpu_lingbot2_routed_col_available(bool w8) {
    // These current-ABI specializations own a complete N64 expert stripe;
    // their per-precision schedules do not use the ordinary dense selector.
    return KernelCache::instance().has_loaded(w8
            ? KernelId::LINGBOT2_ROUTED_COL_W8_M592N32
            : KernelId::LINGBOT2_ROUTED_COL_FP16_M608N32);
}

void rpu_launch_lingbot2_routed_col_spm_kernel(
    uint32_t input_spm_addr, const at::Tensor& dense_packed_weight,
    uint32_t counts_spm_addr, uint32_t output_spm_addr,
    const at::Tensor& scale) {
    constexpr int64_t rows = 51, experts = 32, local_n = 64, local_k = 768;
    constexpr int num_cores = 8;
    constexpr int64_t output_n = experts * local_n;
    constexpr int64_t global_n = num_cores * output_n;
    TORCH_CHECK(dense_packed_weight.defined() &&
                    dense_packed_weight.device().type() == at::kPrivateUse1 &&
                    dense_packed_weight.is_contiguous() &&
                    dense_packed_weight.dim() == 2 &&
                    dense_packed_weight.size(0) == global_n &&
                    dense_packed_weight.size(1) == local_k &&
                    (dense_packed_weight.scalar_type() == at::kHalf ||
                     dense_packed_weight.scalar_type() == at::kChar),
                "lingbot2_routed_col: requires original dense FP16 or signed "
                "W8 packed weight [16384,768] on device");
    const bool w8 = dense_packed_weight.scalar_type() == at::kChar;
    TORCH_CHECK(dense_packed_weight.nbytes() == global_n * local_k * (w8 ? 1 : 2),
                "lingbot2_routed_col: packed weight byte extent mismatch");
    if (w8) {
        TORCH_CHECK(scale.defined() && scale.device() == dense_packed_weight.device() &&
                        scale.scalar_type() == at::kHalf && scale.is_contiguous() &&
                        scale.dim() == 1 && scale.size(0) == global_n &&
                        scale.nbytes() == global_n * 2,
                    "lingbot2_routed_col: W8 requires original dense FP16 "
                    "scale [16384], flattened [core,E,Ic]");
    } else {
        TORCH_CHECK(!scale.defined(),
                    "lingbot2_routed_col: FP16 does not consume scales");
    }
    TORCH_CHECK(rpu_lingbot2_routed_col_available(w8),
                "lingbot2_routed_col: current precision payload is unavailable");
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "lingbot2_routed_col: SPM allocator is not initialized");
    const uint32_t base = SPM_ALLOC.addr(0, 0);
    const std::vector<std::pair<uint32_t, uint32_t>> ranges{
        {input_spm_addr, static_cast<uint32_t>(rows * local_k * 2)},
        {counts_spm_addr, static_cast<uint32_t>(experts * 2)},
        {output_spm_addr, static_cast<uint32_t>(rows * output_n * 2)}};
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto [address, bytes] = ranges[i];
        TORCH_CHECK(address >= base && address % 32 == 0 &&
                        uint64_t(address) - base <=
                            uint64_t(SpmAllocator::SPM_USABLE) - bytes,
                    "lingbot2_routed_col: invalid 32-byte-aligned absolute "
                    "core-0 SPM interval");
        for (size_t j = 0; j < i; ++j) {
            TORCH_CHECK(uint64_t(address) + bytes <= ranges[j].first ||
                            uint64_t(ranges[j].first) + ranges[j].second <= address,
                        "lingbot2_routed_col: input/counts/output must be disjoint");
        }
    }

    // The existing payload exits on count==0 before any weight/scale load and
    // leaves those output slices untouched. Zero every dense output slice on
    // every BUILD/REPLAY to preserve the subsequent SiLU/weight/Down path.
    rpu_launch_fill_spm_kernel(output_spm_addr, rows * output_n,
                              c10::Half(0.0f), num_cores);
    const KernelId kernel_id = w8
        ? KernelId::LINGBOT2_ROUTED_COL_W8_M592N32
        : KernelId::LINGBOT2_ROUTED_COL_FP16_M608N32;
    std::vector<GraphDdrRegisterOperandSpec> ddr_operands{
        GraphDdrRegisterOperandSpec{
            GraphDdrRegisterAbi{
                w8 ? GraphDdrRegisterRole::Int8ModelWeight
                   : GraphDdrRegisterRole::ModelWeight,
                GraphDdrRegisterAccess::Read,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
            dense_packed_weight}};
    if (w8) {
        ddr_operands.push_back(GraphDdrRegisterOperandSpec{
            GraphDdrRegisterAbi{
                GraphDdrRegisterRole::PerChannelScale,
                GraphDdrRegisterAccess::Read,
                GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale});
    }
    auto register_writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
        kernel_id, std::move(ddr_operands));
    rpu_ddr_flush(dense_packed_weight.data_ptr());
    if (w8) rpu_ddr_flush(scale.data_ptr());
    Kernel_t* kernel = GET_KERNEL(kernel_id);
    TORCH_CHECK(kernel != nullptr, "lingbot2_routed_col: loaded payload is missing");
    kernel->reset_regs();
    set_reg_u32(kernel, 0, input_spm_addr);
    set_reg_u32(kernel, 4, output_spm_addr);
    set_reg_u32(kernel, 6, 0);
  kernel->set_regs(8, uint16_t{1});
    kernel->set_regs(9, static_cast<uint16_t>(rows));
    set_reg_u32(kernel, 10, static_cast<uint32_t>(output_n));
    kernel->set_regs(12, static_cast<uint16_t>(local_k));
    kernel->set_regs(13, static_cast<uint16_t>(local_k / 16));
  kernel->set_regs(14, uint16_t{0});
    kernel->set_regs(15, static_cast<uint16_t>(num_cores));
    set_reg_u32(kernel, 16, static_cast<uint32_t>(local_k * 2));
    set_reg_u32(kernel, 18, static_cast<uint32_t>(output_n * 2));
    if (!w8) set_reg_u32(kernel, 20, 0);
    set_reg_u32(kernel, 22, 0);
    set_reg_u32(kernel, 24, 0);
  kernel->set_regs(26, uint16_t{2});
    kernel->set_regs(27, static_cast<uint16_t>(w8 ? 1 : 2));
    set_reg_u32(kernel, 28, 0);
    set_reg_u32(kernel, 48, counts_spm_addr);
    const uint16_t grid_x = w8 ? 2 : 1;
    kernel->set_regs(64, grid_x);
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(experts));
    register_writer.write(*kernel);
    auto* queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {grid_x, 1, static_cast<uint16_t>(experts)},
                        {0, 1, 2, 3, 4, 5, 6, 7});
}
