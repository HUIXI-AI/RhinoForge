#include "rpu_pi05_owner_norm.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include "rhino_launch_queue.h"
#include <array>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <utility>

using namespace ::rhino_lkn;

namespace {
uint32_t prefill_full_bytes(int64_t rows) {
    TORCH_CHECK(rows == 272 || rows == 288 || rows == 304 || rows == 320 || rows == 400 || rows == 416 || rows == 432 || rows == 448,
                "Pi owner norm requires exact C272/C288/C304/C320/C400/C416/C432/C448");
    return static_cast<uint32_t>(rows) * 2048 * sizeof(c10::Half);
}

uint32_t check_operands(
    std::initializer_list<std::pair<uint32_t,uint32_t>> ranges) {
    TORCH_CHECK(SPM_ALLOC.is_initialized(), "Pi owner norm requires initialized SPM");
    const uint32_t base=SPM_ALLOC.addr(0,0);
    for(auto i=ranges.begin();i!=ranges.end();++i) {
        TORCH_CHECK(i->first>=base && i->first%256==0 &&
                    uint64_t(i->first)-base<=SpmAllocator::SPM_USABLE-i->second,
                    "Pi owner norm invalid absolute SPM interval");
        for(auto j=ranges.begin();j!=i;++j)
            TORCH_CHECK(uint64_t(i->first)+i->second<=j->first ||
                        uint64_t(j->first)+j->second<=i->first,
                        "Pi owner norm requires disjoint SPM operands");
    }
    return base;
}

void launch(KernelId id,uint32_t base,uint32_t partial,uint32_t residual,
            uint32_t raw,uint32_t normalized,uint32_t gamma,bool norm,double eps,
            uint32_t row_scale, int64_t rows) {
    const uint32_t CompactBytes = prefill_full_bytes(rows) / 8;
    // These kernels contain only SPM collectives and SPM norm operands.
    auto& graph=RpuKernelGraph::active();
    if(graph.kernel_register_census_active())
        graph.stage_kernel_no_ddr(id,GraphKernelNoDdrProof::RingAllReduceSpmResidual);
    auto* kernel=GET_KERNEL(id);
    TORCH_CHECK(kernel,"Pi owner norm experimental expansion symbol is missing");
    kernel->reset_regs();
    const auto set=[kernel](uint32_t index,uint16_t value) {
        TORCH_CHECK(kernel->set_regs(index,value)==kKernelRegOk,
                    "Pi owner norm rejected register ",index);
    };
    const auto pair=[&](uint32_t index,uint32_t value) {
        set(index,uint16_t(value));set(index+1,uint16_t(value>>16));
    };
    set(0,8);set(1,2);pair(2,63488);pair(4,residual-base);set(6,0);
    if(norm) {
        set(8,2048);set(9,c10::Half(1.0f/2048).x);set(10,4096);set(11,0);
        pair(12,raw);pair(14,normalized);pair(16,gamma);set(18,c10::Half(eps).x);
    }
    if(row_scale) {
        pair(20,row_scale);
        set(29,2048);set(30,0x57f0);set(31,0xd7f0);
        pair(32,4096);pair(34,2);pair(38,2080);set(40,2048);
    }
    set(64,8);set(65,1);set(66,1);
    // Initial atomic values belong to every BUILD and REPLAY launch. The
    // in-kernel norm bridge preserves the ring epoch rather than resetting it.
    for(uint32_t i=128;i<=255;++i)set(i,0);
    for(uint32_t core=0;core<8;++core) {
        const std::array<std::pair<uint32_t,uint32_t>,5> fields{{
            {0,SPM_ALLOC.addr(core,partial-base)-base},
            {16,SPM_ALLOC.addr(core,raw-base)-base},
            {32,core*CompactBytes},{48,CompactBytes/2},
            {64,norm?SPM_ALLOC.addr(core,normalized-base)-base:0}}};
        for(const auto& field:fields) {
            rpu_set_legacy_scm_u16_checked(kernel,4096+field.first+core,
                                          uint16_t(field.second),"Pi owner norm");
            rpu_set_legacy_scm_u16_checked(kernel,4096+field.first+8+core,
                                          uint16_t(field.second>>16),"Pi owner norm");
        }
    }
    auto* queue=GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel,{8,1,1},{0,1,2,3,4,5,6,7});
}
} // namespace

void rpu_launch_pi05_owner_norm_spm_kernel(
    uint32_t partial,uint32_t residual,uint32_t compact_raw,
    uint32_t normalized,uint32_t gamma,Pi05OwnerNormResidual kind,double eps,
    int64_t rows) {
    const uint32_t FullBytes = prefill_full_bytes(rows), CompactBytes = FullBytes / 8;
    TORCH_CHECK(kind==Pi05OwnerNormResidual::Full || kind==Pi05OwnerNormResidual::Compact,
                "Pi owner norm invalid residual ABI");
    TORCH_CHECK(std::isfinite(eps) && eps==1e-6,
                "Pi owner norm requires original epsilon 1e-6");
    const bool full=kind==Pi05OwnerNormResidual::Full;
    const auto base=check_operands({{partial,FullBytes},
        {residual,full?FullBytes:CompactBytes},{compact_raw,CompactBytes},
        {normalized,FullBytes},{gamma,4096}});
    const auto id = rows == 432
        ? (full ? KernelId::PI05_OWNER_NORM_FULL_RESIDUAL_M432N2048
                : KernelId::PI05_OWNER_NORM_COMPACT_RESIDUAL_M432N2048)
        : rows == 448
        ? (full ? KernelId::PI05_OWNER_NORM_FULL_RESIDUAL_M448N2048
                : KernelId::PI05_OWNER_NORM_COMPACT_RESIDUAL_M448N2048)
        : rows == 416
        ? (full ? KernelId::PI05_OWNER_NORM_FULL_RESIDUAL_M416N2048
                : KernelId::PI05_OWNER_NORM_COMPACT_RESIDUAL_M416N2048)
        : rows == 288
        ? (full ? KernelId::PI05_OWNER_NORM_FULL_RESIDUAL_M288N2048
                : KernelId::PI05_OWNER_NORM_COMPACT_RESIDUAL_M288N2048)
        : rows == 304
        ? (full ? KernelId::PI05_OWNER_NORM_FULL_RESIDUAL_M304N2048
                : KernelId::PI05_OWNER_NORM_COMPACT_RESIDUAL_M304N2048) :
        rows == 320
        ? (full ? KernelId::PI05_OWNER_NORM_FULL_RESIDUAL_M320N2048
                : KernelId::PI05_OWNER_NORM_COMPACT_RESIDUAL_M320N2048)
        : rows == 272
        ? (full ? KernelId::PI05_OWNER_NORM_FULL_RESIDUAL_M272N2048
                : KernelId::PI05_OWNER_NORM_COMPACT_RESIDUAL_M272N2048)
        : (full ? KernelId::PI05_OWNER_NORM_FULL_RESIDUAL_M400N2048
                : KernelId::PI05_OWNER_NORM_COMPACT_RESIDUAL_M400N2048);
    launch(id,base,partial,residual,compact_raw,normalized,gamma,true,eps,0,rows);
}

void rpu_launch_pi05_compact_residual_spm_kernel(
    uint32_t partial,uint32_t compact_residual,uint32_t full_raw,int64_t rows) {
    const uint32_t FullBytes = prefill_full_bytes(rows), CompactBytes = FullBytes / 8;
    const auto base=check_operands({{partial,FullBytes},
        {compact_residual,CompactBytes},{full_raw,FullBytes}});
    launch(rows == 432 ? KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M432N2048
         : rows == 448 ? KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M448N2048
         : rows == 416 ? KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M416N2048
         : rows == 288 ? KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M288N2048
         : rows == 304 ? KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M304N2048
         : rows == 320 ? KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M320N2048
         : rows == 272 ? KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M272N2048
                       : KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M400N2048,
           base,partial,compact_residual,full_raw,0,0,false,0,0,rows);
}

void rpu_launch_pi05_owner_norm_a8_spm_kernel(
    uint32_t partial,uint32_t residual,uint32_t compact_raw,
    uint32_t a8,uint32_t gamma,uint32_t row_scale,double eps,int64_t rows) {
    const uint32_t FullBytes = prefill_full_bytes(rows), CompactBytes = FullBytes / 8;
    // C272/C288/C304/C320/C416/C432/C448 consumers read a vector past the final scalar scale.
    // Initialize the 32B tail even when all M48 tiles are complete.
    const uint32_t scale_bytes = static_cast<uint32_t>(rows) * sizeof(c10::Half) +
        (rows == 272 || rows == 288 || rows == 304 || rows == 320 || rows == 416 || rows == 432 || rows == 448 ? 32 : 0);
    TORCH_CHECK(std::isfinite(eps) && eps==1e-6,
                "Pi A8 owner norm requires original epsilon 1e-6");
    const std::initializer_list<std::pair<uint32_t,uint32_t>> ranges{
        {partial,FullBytes},{residual,CompactBytes},{compact_raw,CompactBytes},
        {a8,static_cast<uint32_t>(rows)*2080},{gamma,4096}};
    const auto base=check_operands(ranges);
    // Each half's row-scale vector starts on a 32B boundary.
    TORCH_CHECK(row_scale>=base && row_scale%32==0 &&
        uint64_t(row_scale)-base<=SpmAllocator::SPM_USABLE-scale_bytes,
        "Pi A8 owner norm invalid scale interval");
    for(const auto& range:ranges)
        TORCH_CHECK(uint64_t(row_scale)+scale_bytes<=range.first ||
                    uint64_t(range.first)+range.second<=row_scale,
                    "Pi A8 owner norm scale aliases a live operand");
    launch(rows == 432 ? KernelId::PI05_OWNER_NORM_COMPACT_A8_M432N2048
         : rows == 448 ? KernelId::PI05_OWNER_NORM_COMPACT_A8_M448N2048
         : rows == 416 ? KernelId::PI05_OWNER_NORM_COMPACT_A8_M416N2048
         : rows == 288 ? KernelId::PI05_OWNER_NORM_COMPACT_A8_M288N2048
         : rows == 304 ? KernelId::PI05_OWNER_NORM_COMPACT_A8_M304N2048
         : rows == 320 ? KernelId::PI05_OWNER_NORM_COMPACT_A8_M320N2048
         : rows == 272 ? KernelId::PI05_OWNER_NORM_COMPACT_A8_M272N2048
                       : KernelId::PI05_OWNER_NORM_COMPACT_A8_M400N2048,base,
           partial,residual,compact_raw,a8,gamma,true,eps,row_scale,rows);
}

namespace {
constexpr uint32_t VisionColumns = 1152;
uint32_t vision_full_bytes(int64_t rows) {
    TORCH_CHECK(rows == 512 || rows == 768,
                "Pi Vision owner LayerNorm requires exact M512 or M768");
    return static_cast<uint32_t>(rows) * VisionColumns * sizeof(c10::Half);
}
constexpr uint32_t VisionAffineBytes = VisionColumns * sizeof(c10::Half);

uint32_t check_vision_operands(
    std::initializer_list<std::pair<uint32_t, uint32_t>> ranges,
    bool allow_partial_normalized_alias = false) {
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "Pi Vision owner LayerNorm requires initialized SPM");
    const uint32_t base = SPM_ALLOC.addr(0, 0);
    for (auto i = ranges.begin(); i != ranges.end(); ++i) {
        TORCH_CHECK(i->first >= base && i->first % 256 == 0 &&
                    uint64_t(i->first) - base + i->second <= SpmAllocator::SPM_USABLE,
                    "Pi Vision owner LayerNorm invalid absolute core-0 SPM interval");
        for (auto j = ranges.begin(); j != i; ++j) {
            // Owner ABI lists partial first and normalized fourth. Only the
            // exact whole-root alias is safe after the reduce-scatter barrier.
            const bool permitted_alias = allow_partial_normalized_alias &&
                i - ranges.begin() == 3 && j == ranges.begin() &&
                i->first == j->first;
            TORCH_CHECK(permitted_alias ||
                        uint64_t(i->first) + i->second <= j->first ||
                        uint64_t(j->first) + j->second <= i->first,
                        "Pi Vision owner LayerNorm requires disjoint SPM operands "
                        "except exact partial == normalized");
        }
    }
    return base;
}

void launch_vision_owner(KernelId id, uint32_t base, uint32_t partial,
                        uint32_t residual, uint32_t raw, uint32_t normalized,
                        uint32_t gamma, uint32_t beta, bool layer_norm, double eps, int64_t rows) {
    const uint32_t VisionCompactBytes = vision_full_bytes(rows) / 8;
    auto& graph = RpuKernelGraph::active();
    if (graph.kernel_register_census_active()) {
        graph.stage_kernel_no_ddr(id, GraphKernelNoDdrProof::RingAllReduceSpmResidual);
    }
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel, "Pi Vision owner LayerNorm expansion symbol is missing");
    kernel->reset_regs();
    const auto set = [kernel](uint32_t index, uint16_t value) {
        TORCH_CHECK(kernel->set_regs(index, value) == kKernelRegOk,
                    "Pi Vision owner LayerNorm rejected register ", index);
    };
    const auto pair = [&](uint32_t index, uint32_t value) {
        set(index, uint16_t(value));
        set(index + 1, uint16_t(value >> 16));
    };
    set(0, 8);
    set(1, 2);
    pair(2, 63488);
    pair(4, residual - base);  // Original ring takes a core-0 local offset.
    set(6, 0);
    if (layer_norm) {
        // Original high-range LayerNorm GP0..21, shifted by eight. The payload
        // runs owner64/96 and keeps inactive blocks in subsequent barriers.
        pair(8, raw);
        pair(10, normalized);
        pair(12, gamma);
        pair(14, beta);
        set(16, static_cast<uint16_t>(rows / 8));
        set(17, VisionColumns);
        const float inverse_columns = 1.0f / VisionColumns;
        uint32_t inverse_columns_bits;
        std::memcpy(&inverse_columns_bits, &inverse_columns, sizeof(inverse_columns_bits));
        pair(18, inverse_columns_bits);
        set(20, c10::Half(static_cast<float>(eps) * VisionColumns).x);
        set(21, 0);  // No skip; residual has already retired to compact raw.
        pair(22, 0);
        pair(24, VisionAffineBytes);
        set(26, 1);
        set(27, 16);
        set(28, c10::Half(std::sqrt(static_cast<float>(VisionColumns))).x);
        set(29, 1);  // Affine gamma and beta.
    }
    set(42, 0);  // Every native launch starts from freshly initialized SCU.
    set(64, 8);
    set(65, 1);
    set(66, 1);
    for (uint32_t i = 128; i <= 255; ++i) set(i, 0);
    for (uint32_t core = 0; core < 8; ++core) {
        const std::array<std::pair<uint32_t, uint32_t>, 5> fields{{
            {0, SPM_ALLOC.addr(core, partial - base) - base},
            {16, SPM_ALLOC.addr(core, raw - base) - base},
            {32, core * VisionCompactBytes},
            {48, VisionCompactBytes / sizeof(c10::Half)},
            {64, layer_norm ? SPM_ALLOC.addr(core, normalized - base) - base : 0}}};
        for (const auto& field : fields) {
            rpu_set_legacy_scm_u16_checked(kernel, 4096 + field.first + core,
                uint16_t(field.second), "Pi Vision owner LayerNorm");
            rpu_set_legacy_scm_u16_checked(kernel, 4096 + field.first + 8 + core,
                uint16_t(field.second >> 16), "Pi Vision owner LayerNorm");
        }
    }
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {8, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}
} // namespace

void rpu_launch_pi05_vision_owner_layernorm_spm_kernel(
    uint32_t partial, uint32_t residual, uint32_t compact_raw,
    uint32_t normalized, uint32_t gamma, uint32_t beta, double eps, int64_t rows) {
    const uint32_t VisionFullBytes = vision_full_bytes(rows), VisionCompactBytes = VisionFullBytes / 8;
    TORCH_CHECK(std::isfinite(eps) && eps == 1e-6,
                "Pi Vision owner LayerNorm requires original epsilon 1e-6");
    const auto base = check_vision_operands({
        {partial, VisionFullBytes}, {residual, VisionFullBytes},
        {compact_raw, VisionCompactBytes}, {normalized, VisionFullBytes},
        {gamma, VisionAffineBytes}, {beta, VisionAffineBytes}}, true);
    launch_vision_owner(rows == 512 ? KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M512N1152
                                    : KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152,
                       base, partial, residual, compact_raw, normalized,
                       gamma, beta, true, eps, rows);
}

void rpu_launch_pi05_vision_compact_residual_spm_kernel(
    uint32_t partial, uint32_t compact_residual, uint32_t full_raw, int64_t rows) {
    const uint32_t VisionFullBytes = vision_full_bytes(rows), VisionCompactBytes = VisionFullBytes / 8;
    const auto base = check_vision_operands({
        {partial, VisionFullBytes}, {compact_residual, VisionCompactBytes},
        {full_raw, VisionFullBytes}});
    launch_vision_owner(rows == 512 ? KernelId::PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M512N1152
                                    : KernelId::PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152,
                       base, partial, compact_residual, full_raw, 0, 0, 0, false, 0, rows);
}
