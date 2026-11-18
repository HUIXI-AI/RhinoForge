#include "rpu_pi05_prefill_pair.h"
#include "rpu_ops.h"
#include "rpu_linear_tiling.h"
#include "rpu_pi_wall_tiling.h"
#include "rpu_spm_allocator.h"
#include "rhino_launch_buffer.h"
#include "rhino_launch_queue.h"
#include <array>
#include <utility>
#include <vector>

using namespace ::rhino_lkn;

void rpu_launch_pi05_prefill_pair_spm_kernel(
    Pi05PrefillPairedProjection kind,uint32_t input0,uint32_t input1,
    const at::Tensor& weight,uint32_t output0,uint32_t output1,const at::Tensor& scale, int64_t rows) {
    TORCH_CHECK(kind==Pi05PrefillPairedProjection::QkvColumn ||
                kind==Pi05PrefillPairedProjection::AttentionOutputRow,
                "Pi paired prefill invalid projection ABI");
    const bool column=kind==Pi05PrefillPairedProjection::QkvColumn;
    TORCH_CHECK(rows == 272 || rows == 288 || rows == 304 || rows == 320 || rows == 336 || rows == 400 || rows == 416 || rows == 432 || rows == 448 || rows == 464 || rows == 480,
                "Pi paired prefill requires C272/C288/C304/C320/C336/C400/C416/C432/C448/C464/C480");
    const bool fp16 = weight.defined() && weight.scalar_type() == at::kHalf;
    TORCH_CHECK(!column || (rows == 400 && !fp16), "Pi paired QKV prototype remains C400/W8");
    const uint32_t M=rows,N=column?256:2048,K=column?2048:256;
    const uint64_t X=uint64_t(M)*K*2,Y=uint64_t(M)*N*2;
    TORCH_CHECK(SPM_ALLOC.is_initialized(),"Pi paired prefill requires initialized SPM");
    const uint32_t base=SPM_ALLOC.addr(0,0);
    const std::array<std::pair<uint32_t,uint64_t>,4> ranges{{
        {input0,X},{input1,X},{output0,Y},{output1,Y}}};
    for(size_t i=0;i<ranges.size();++i) {
        const auto& r=ranges[i];
        TORCH_CHECK(r.first>=base && r.first%256==0 &&
                    uint64_t(r.first)-base<=SpmAllocator::SPM_USABLE-r.second,
                    "Pi paired prefill invalid absolute SPM interval");
        for(size_t j=0;j<i;++j)
            TORCH_CHECK(uint64_t(r.first)+r.second<=ranges[j].first ||
                        uint64_t(ranges[j].first)+ranges[j].second<=r.first,
                        "Pi paired prefill requires four disjoint SPM operands");
    }
    TORCH_CHECK(weight.defined() && weight.device().type()==at::kPrivateUse1 &&
                (fp16 || weight.scalar_type()==at::kChar) && weight.is_contiguous() &&
                weight.sizes()==at::IntArrayRef({2048,2048}),
                "Pi paired prefill requires W8 [2048,2048] in the selected TP layout");
    TORCH_CHECK(fp16 ? !scale.defined() : (scale.defined() && scale.device().type()==at::kPrivateUse1 &&
                scale.scalar_type()==at::kHalf && scale.is_contiguous() &&
                scale.sizes()==at::IntArrayRef({2048})),
                "Pi paired prefill requires original FP16 per-channel scale [2048]");
    const auto selected = rpu_pi_wall::select(
        rpu_pi_wall::Operation::PairedLinear,
        fp16 ? rpu_pl_tiling::WeightType::Fp16 : rpu_pl_tiling::WeightType::Int8,
        false, M, N, K);
    const auto id = pi_wall_kernel_id(selected.ordinal);
    std::vector<GraphDdrRegisterOperandSpec> operands{
            {GraphDdrRegisterAbi{fp16 ? GraphDdrRegisterRole::ModelWeight : GraphDdrRegisterRole::Int8ModelWeight,
                GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},weight}};
    if (!fp16) operands.push_back({GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
                GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},scale});
    auto writer=RpuKernelGraph::active().stage_kernel_ddr_registers(id, operands);
    for(const auto* t:{&weight,&scale}) {
        if (!t->defined()) continue;
        const auto address=RpuGetDevAddr(t->data_ptr());
        TORCH_CHECK(address && address%256==0 && (address>>8)<=UINT32_MAX,
                    "Pi paired prefill DDR operand exceeds aligned shift8 ABI");
        rpu_ddr_flush(t->data_ptr());
    }
    auto* kernel=GET_KERNEL(id);
    TORCH_CHECK(kernel,"Pi paired prefill experimental expansion symbol is missing");
    kernel->reset_regs();
    rpu_pl_tiling::set_parallel_linear_regs(*kernel,
        rpu_pl_tiling::ParallelLinearRegisterArgs{
            input0,0,output0,0,2*M,N,K,/*has_bias=*/0,8,K*2,N*2,0,
            /*partition=*/uint16_t(column?1:0),uint16_t(fp16 ? 2 : 1)});
    const auto pair=[kernel](uint32_t index,uint32_t value) {
        TORCH_CHECK(kernel->set_regs(index,uint16_t(value))==kKernelRegOk &&
                    kernel->set_regs(index+1,uint16_t(value>>16))==kKernelRegOk,
                    "Pi paired prefill rejected extra SPM register pair");
    };
    pair(48,input1);pair(50,output1);
    kernel->set_regs(52, static_cast<uint16_t>(M));
    writer.write(*kernel);
    auto* queue=GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel,{selected.grid_x,selected.grid_y,1},{0,1,2,3,4,5,6,7});
}
