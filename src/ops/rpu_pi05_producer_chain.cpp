#include "rpu_pi05_producer_chain.h"
#include "rpu_ops.h"
#include "rpu_linear_tiling.h"
#include "rpu_spm_allocator.h"
#include "rhino_launch_buffer.h"
#include "rhino_launch_queue.h"
#include <array>
#include <cmath>
#include <utility>
#include <vector>

using namespace ::rhino_lkn;

void rpu_launch_pi05_prefill_o_chain_spm_kernel(
    uint32_t input, const at::Tensor& weight, uint32_t partial,
    uint32_t residual, uint32_t raw, uint32_t normalized,
    uint32_t gamma, const at::Tensor& scale, double eps) {
    constexpr uint32_t M=400, N=2048, K=256, Cores=8;
    constexpr uint64_t X=M*K*2, Y=M*N*2, G=N*2;
    TORCH_CHECK(SPM_ALLOC.is_initialized(), "Pi O-chain requires initialized SPM");
    const uint32_t base=SPM_ALLOC.addr(0,0);
    const std::array<std::pair<uint32_t,uint64_t>,6> ranges{{
        {input,X},{partial,Y},{residual,Y},{raw,Y},{normalized,Y},{gamma,G}}};
    for(const auto& range:ranges)
        TORCH_CHECK(range.first>=base && range.first%256==0 &&
                    range.second<=SpmAllocator::SPM_USABLE &&
                    uint64_t(range.first)-base<=SpmAllocator::SPM_USABLE-range.second,
                    "Pi O-chain invalid absolute SPM interval");
    // Producer input/partial reads complete before RMSNorm starts.
    // Admit only the two exact mask-v2 overlays, relative to the live arena;
    // raw, residual, gamma and all producer inputs remain mutually disjoint.
    const bool dead_producer_overlay =
        uint64_t(partial) == uint64_t(input) + X &&
        (normalized == input || uint64_t(normalized) + X == input);
    for(size_t i=0;i<ranges.size();++i)for(size_t j=0;j<i;++j) {
        const bool exact_residual_alias=(i==4 && j==2 && normalized==residual);
        const bool final_norm_overlay = i==4 && (j==0 || j==1) &&
            dead_producer_overlay;
        TORCH_CHECK(exact_residual_alias || final_norm_overlay ||
                    uint64_t(ranges[i].first)+ranges[i].second<=ranges[j].first ||
                    uint64_t(ranges[j].first)+ranges[j].second<=ranges[i].first,
                    "Pi O-chain rejects overlap outside residual alias or exact dead-producer mask overlay");
    }
    TORCH_CHECK(weight.defined() && weight.device().type()==at::kPrivateUse1 &&
                weight.scalar_type()==at::kChar && weight.is_contiguous() &&
                weight.sizes()==at::IntArrayRef({N,K*Cores}),
                "Pi O-chain requires original row-striped W8 [2048,2048]");
    TORCH_CHECK(scale.defined() && scale.device().type()==at::kPrivateUse1 &&
                scale.scalar_type()==at::kHalf && scale.is_contiguous() &&
                scale.sizes()==at::IntArrayRef({N}),
                "Pi O-chain requires original per-channel FP16 scale [2048]");
    TORCH_CHECK(std::isfinite(eps) && eps==1e-6,
                "Pi O-chain experiment admits only original epsilon 1e-6");
    const auto tile=rpu_pl_tiling::select_tile_acc16(M,N,K,false);
    TORCH_CHECK(tile.m_tile==400 && tile.n_tile==80,
                "Pi O-chain original M400/N80 tile identity drift");
    constexpr KernelId id=KernelId::PI05_PREFILL_O_LINEAR_XOR3_RMSNORM_M400N2048K256;
    auto writer=RpuKernelGraph::active().stage_kernel_ddr_registers(id,
        std::vector<GraphDdrRegisterOperandSpec>{
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
                GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},weight},
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
                GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},scale}});
    rpu_ddr_flush(weight.data_ptr<int8_t>());
    rpu_ddr_flush(scale.data_ptr<c10::Half>());
    auto* kernel=GET_KERNEL(id);
    TORCH_CHECK(kernel,"Pi O-chain experimental expansion symbol is missing");
    kernel->reset_regs();
    rpu_pl_tiling::set_parallel_linear_regs(*kernel,
        rpu_pl_tiling::ParallelLinearRegisterArgs{input,0,partial,0,M,N,K,0,Cores,K*2,N*2,0,0,1});
    auto set=[kernel](uint32_t index,uint16_t value) {
        TORCH_CHECK(kernel->set_regs(index,value)==kKernelRegOk,
                    "Pi O-chain register setter rejected ",index);
    };
    auto pair=[&](uint32_t index,uint32_t value) {set(index,uint16_t(value));set(index+1,uint16_t(value>>16));};
    set(24,Cores);set(25,2);pair(26,63488);pair(28,residual-base);set(30,0);
    set(32,N);set(33,c10::Half(1.0f/N).x);set(34,N*2);set(35,0);
    pair(36,raw);pair(38,normalized);pair(40,gamma);set(42,c10::Half(eps).x);
    set(64,8);set(65,1);set(66,1);
    // Producer uses local0/1 + global0..7; ring uses local24..31/global24.
    // These initial values belong to every recorded and replayed launch.
    for(uint32_t index=128;index<=255;++index)set(index,0);
    constexpr uint32_t shard=M*N/Cores;
    for(uint32_t core=0;core<Cores;++core) {
        const std::array<std::pair<uint32_t,uint32_t>,4> values{{
            {0,SPM_ALLOC.addr(core,partial-base)-base},
            {16,SPM_ALLOC.addr(core,raw-base)-base},
            {32,core*shard*2},{48,shard}}};
        for(const auto& value:values) {
            rpu_set_legacy_scm_u16_checked(kernel,4096+value.first+core,
                                          uint16_t(value.second),"Pi O-chain");
            rpu_set_legacy_scm_u16_checked(kernel,4096+value.first+8+core,
                                          uint16_t(value.second>>16),"Pi O-chain");
        }
    }
    writer.write(*kernel);
    auto* queue=GET_QUEUE(Cores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel,{8,1,1},{0,1,2,3,4,5,6,7});
}

void rpu_launch_pi05_xor3_gated_residual_spm_kernel(
    uint32_t partial, uint32_t zero, uint32_t skip, uint32_t gate,
    uint32_t output) {
    constexpr uint32_t Cores = 8;
    constexpr uint32_t FullBytes = 50 * 1024 * sizeof(c10::Half);
    constexpr uint32_t GateBytes = 1024 * sizeof(c10::Half);
    constexpr uint32_t ShardElements = 50 * 1024 / Cores;
    constexpr uint32_t ShardBytes = ShardElements * sizeof(c10::Half);
    constexpr uint32_t RoundElements = 63488;

    TORCH_CHECK(
        SPM_ALLOC.is_initialized(),
        "Pi XOR3+gated continuation requires initialized SPM");
    const uint32_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint32_t>, 5> ranges{{
        {partial, FullBytes},
        {zero, FullBytes},
        {skip, FullBytes},
        {gate, GateBytes},
        {output, FullBytes},
    }};
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto [address, bytes] = ranges[i];
        TORCH_CHECK(
            address >= base && (address & 255U) == 0 &&
                uint64_t(address) - base <=
                    uint64_t(SpmAllocator::SPM_USABLE) - bytes,
            "Pi XOR3+gated continuation requires complete 256-byte-aligned "
            "absolute SPM operands");
        for (size_t j = 0; j < i; ++j) {
            const uint64_t lhs = address;
            const uint64_t rhs = ranges[j].first;
            TORCH_CHECK(
                lhs + bytes <= rhs || rhs + ranges[j].second <= lhs,
                "Pi XOR3+gated continuation requires five mutually "
                "disjoint SPM operands");
        }
    }

    constexpr KernelId id = KernelId::PI05_XOR3_GATED_RESIDUAL_M50N1024;
    auto& graph = RpuKernelGraph::active();
    if (graph.kernel_register_census_active()) {
        graph.stage_kernel_no_ddr(
            id, GraphKernelNoDdrProof::RingAllReduceSpmResidual);
    }
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(
        kernel,
        "Pi XOR3+gated continuation expansion symbol is missing");
    kernel->reset_regs();
    const auto set = [kernel](uint32_t index, uint16_t value) {
        TORCH_CHECK(
            kernel->set_regs(index, value) == kKernelRegOk,
            "Pi XOR3+gated continuation rejected register ", index);
    };
    const auto pair = [&set](uint32_t index, uint32_t value) {
        set(index, uint16_t(value));
        set(index + 1, uint16_t(value >> 16));
    };

    // Frozen continuation ABI: preserve the original one-round equal-shard
    // XOR3 reduction, then apply the existing separately-rounded gate/add tail.
    set(0, Cores);
    set(1, 1);
    pair(2, RoundElements);
    pair(4, zero - base);
    set(6, 0);
    pair(8, skip);
    pair(10, gate);
    for (uint32_t index = 56; index <= 58; ++index) set(index, 0x3c00);
    set(59, 0);
    set(60, 0);
    set(61, 1);
    set(62, 0x0492);
    set(63, 0);
    set(64, Cores);
    set(65, 1);
    set(66, 1);
    for (uint32_t index = 128; index <= 255; ++index) set(index, 0);

    for (uint32_t core = 0; core < Cores; ++core) {
        const std::array<std::pair<uint32_t, uint32_t>, 4> fields{{
            {0, SPM_ALLOC.addr(core, partial - base) - base},
            {16, SPM_ALLOC.addr(core, output - base) - base},
            {32, core * ShardBytes},
            {48, ShardElements},
        }};
        for (const auto& [table, value] : fields) {
            rpu_set_legacy_scm_u16_checked(
                kernel, 4096 + table + core, uint16_t(value),
                "Pi XOR3+gated continuation");
            rpu_set_legacy_scm_u16_checked(
                kernel, 4096 + table + 8 + core,
                uint16_t(value >> 16),
                "Pi XOR3+gated continuation");
        }
    }

    auto* queue = GET_QUEUE(Cores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel, {Cores, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}
