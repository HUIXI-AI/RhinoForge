#pragma once
#include "rpu_ops.h"
#include "rpu_kernel_decls.h"
#include "rpu_spm_allocator.h"
#include "rpu_linear_tiling.h"
#include "rpu_specialized_linear_contracts.h"

// Exact HyVLA M51/H1024/I2048/TP8 adapter for Wall W8 and the private N48 trial.
// Resident pair: independent Gate/Up outputs, then the owner's original SiLU*Up.
// DDR fused: direct-store SwiGLU, no external scratch and no unused Up writes.
inline void launch_hyvla_gateup_w8(
    uint32_t input, uint32_t gate, uint32_t up,
    uint32_t gate_spm, uint32_t up_spm, int64_t rows,
    const at::Tensor& gate_weight, const at::Tensor& up_weight,
    const at::Tensor& gate_scale, const at::Tensor& up_scale, bool resident, bool narrow = false,
    bool fused_resident = false) {
    TORCH_CHECK(!(resident && narrow), "HyVLA N48 fusion requires DDR weights");
    TORCH_CHECK(!fused_resident || resident, "HyVLA resident SwiGLU requires SPM weights");
    TORCH_CHECK(rows == 51 && SPM_ALLOC.is_initialized(),
                "HyVLA W8 Gate/Up requires M51 and initialized SPM");
    for (auto* w : {&gate_weight, &up_weight})
        TORCH_CHECK(w->defined() && w->device().type() == c10::DeviceType::PrivateUse1 &&
                    w->scalar_type() == at::kChar && w->is_contiguous() &&
                    w->dim() == 2 && w->size(0) == 2048 && w->size(1) == 1024,
                    "HyVLA W8 Gate/Up requires installed INT8 [2048,1024]");
    for (auto* sc : {&gate_scale, &up_scale})
        TORCH_CHECK(sc->defined() && sc->device() == gate_weight.device() &&
                    sc->scalar_type() == at::kHalf && sc->is_contiguous() &&
                    sc->dim() == 1 && sc->size(0) == 2048,
                    "HyVLA W8 Gate/Up requires owned FP16 [2048] scales");
    TORCH_CHECK(up_weight.device() == gate_weight.device(), "HyVLA W8 device mismatch");
    std::vector<std::pair<uint32_t, uint64_t>> regions{{input, 51*1024*2}, {gate, 51*256*2}};
    if (resident) {
        regions.insert(regions.end(), {{up, 51*256*2}, {gate_spm, 256*1024}, {up_spm, 256*1024}});
    } else {
        TORCH_CHECK(up == 0 && gate_spm == 0 && up_spm == 0,
                    "HyVLA DDR fused Gate/Up has no Up/scratch/SPM weight operand");
    }
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    for (size_t i = 0; i < regions.size(); ++i) {
        const uint64_t start = regions[i].first, bytes = regions[i].second;
        TORCH_CHECK(start >= base && start % 256 == 0 &&
                    start - base <= SpmAllocator::SPM_USABLE - bytes,
                    "HyVLA W8 requires complete aligned SPM regions");
        for (int core = 0; core < 8; ++core)
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + start - base + bytes - 1 <= UINT32_MAX,
                        "HyVLA W8 absolute SPM range exceeds uint32");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(start + bytes <= regions[j].first ||
                        uint64_t(regions[j].first) + regions[j].second <= start,
                        "HyVLA W8 requires disjoint SPM operands");
    }
    const KernelId id = resident
        ? (fused_resident ? KernelId::HYVLA_W8_RESIDENT_SWIGLU_S51
                          : KernelId::HYVLA_GATEUP_RESIDENT_PAIR_W8_V2)
        : KernelId::HYVLA_EXPERT_GATEUP_W8_S51_N48;
    std::vector<GraphDdrRegisterOperandSpec> operands;
    auto add = [&](const at::Tensor& t, GraphDdrRegisterRole role, int lo) {
        TORCH_CHECK(rhino_lkn::RpuGetDevAddr(t.data_ptr()) % 256 == 0, "HyVLA W8 DDR alignment");
        operands.push_back({GraphDdrRegisterAbi{role, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, uint16_t(lo), uint16_t(lo+1)}, t});
    };
    if (!resident) add(gate_weight, GraphDdrRegisterRole::Int8ModelWeight, 2);
    add(gate_scale, GraphDdrRegisterRole::PerChannelScale, 20);
    if (!resident) add(up_weight, GraphDdrRegisterRole::Int8ModelWeight, 32);
    add(up_scale, GraphDdrRegisterRole::PerChannelScale, 34);
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id, operands);
    for (auto* t : {&gate_weight, &up_weight, &gate_scale, &up_scale}) {
        rpu_ddr_flush(t->data_ptr());
        RpuKernelGraph::active().keep_alive(*t);
    }
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel != nullptr, "HyVLA W8 Gate/Up payload missing");
    kernel->reset_regs();
    rpu_pl_tiling::set_parallel_linear_regs(*kernel,
        rpu_pl_tiling::ParallelLinearRegisterArgs{
            input, 0, gate, 0, 51, 256, 1024, 0, 8, 2048, 512, 0, 1, 1});
    if (resident) {
        rpu_pl_tiling::set_u32_reg_pair(*kernel, 2, gate_spm);
        rpu_pl_tiling::set_u32_reg_pair(*kernel, 32, up_spm);
        rpu_pl_tiling::set_u32_reg_pair(*kernel, 36, up);
    }
    writer.write(*kernel);
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, rpu_specialized_linear::grid(id), {0,1,2,3,4,5,6,7});
}
