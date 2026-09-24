#include "rpu_pi05_vision_fc.h"
#include "rpu_ops.h"
#include "rpu_linear_tiling.h"
#include "rpu_spm_allocator.h"
#include "rhino_launch_buffer.h"
#include "rhino_launch_queue.h"
#include <array>
#include <utility>
#include <vector>

using namespace ::rhino_lkn;

void rpu_launch_pi05_vision_fc_weight_outer_spm_kernel(
    Pi05VisionFcProjection role,uint32_t input,const at::Tensor& weight,
    uint32_t output,uint32_t bias,const at::Tensor& scale) {
    TORCH_CHECK(role==Pi05VisionFcProjection::Fc1Column ||
        role==Pi05VisionFcProjection::Fc2Row,"Pi Vision FC invalid role");
    const bool column=role==Pi05VisionFcProjection::Fc1Column;
    const uint32_t M=768,N=column?544:1152,K=column?1152:544;
    const int64_t global_n=column?4352:1152,global_k=column?1152:4352;
    TORCH_CHECK(SPM_ALLOC.is_initialized(),"Pi Vision FC requires initialized SPM");
    const uint32_t base=SPM_ALLOC.addr(0,0);
    const std::array<std::pair<uint32_t,uint64_t>,3> ranges{{
        {input,uint64_t(M)*K*2},{output,uint64_t(M)*N*2},{bias,uint64_t(N)*2}}};
    for(size_t i=0;i<ranges.size();++i) {
        const auto& r=ranges[i];
        TORCH_CHECK(r.first>=base && r.first%256==0 && r.second<=SpmAllocator::SPM_USABLE &&
            uint64_t(r.first)-base<=SpmAllocator::SPM_USABLE-r.second,
            "Pi Vision FC invalid absolute FP16 SPM interval");
        for(size_t j=0;j<i;++j)
            TORCH_CHECK(uint64_t(r.first)+r.second<=ranges[j].first ||
                uint64_t(ranges[j].first)+ranges[j].second<=r.first,
                "Pi Vision FC requires disjoint X/Y/original bias");
    }
    TORCH_CHECK(weight.defined() && weight.device().type()==at::kPrivateUse1 &&
        weight.scalar_type()==at::kChar && weight.is_contiguous() &&
        weight.sizes()==at::IntArrayRef({global_n,global_k}) && weight.nbytes()==size_t(global_n*global_k),
        "Pi Vision FC requires the exact TP-swizzled signed W8 weight");
    TORCH_CHECK(scale.defined() && scale.device().type()==at::kPrivateUse1 &&
        scale.scalar_type()==at::kHalf && scale.is_contiguous() &&
        scale.sizes()==at::IntArrayRef({global_n}) && scale.nbytes()==size_t(global_n*2),
        "Pi Vision FC requires the original FP16 per-output-channel scale");
    const auto tile=rpu_pl_tiling::select_tile_acc16(M,N,K,false);
    TORCH_CHECK(tile.m_tile==400 && tile.n_tile==80,"Pi Vision FC original tile identity drift");
    constexpr auto id=KernelId::PI05_VISION_FC_WEIGHT_OUTER_W8A16_M768N80K128;
    auto writer=RpuKernelGraph::active().stage_kernel_ddr_registers(id,
        std::vector<GraphDdrRegisterOperandSpec>{
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
                GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},weight},
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
                GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},scale}});
    for(const auto* t:{&weight,&scale}) {
        const auto address=RpuGetDevAddr(t->data_ptr());
        TORCH_CHECK(address && address%256==0 && (address>>8)<=UINT32_MAX,
            "Pi Vision FC DDR operand exceeds aligned shift8 ABI");
        rpu_ddr_flush(t->data_ptr());
    }
    auto* kernel=GET_KERNEL(id);
    TORCH_CHECK(kernel,"Pi Vision FC experimental expansion symbol is missing");
    kernel->reset_regs();
    rpu_pl_tiling::set_parallel_linear_regs(*kernel,rpu_pl_tiling::ParallelLinearRegisterArgs{
        input,0,output,bias,M,N,K,1,8,K*2,N*2,0,uint16_t(column?1:0),1});
    writer.write(*kernel);
    auto* queue=GET_QUEUE(8);queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel,{uint16_t(column?7:15),1,1},{0,1,2,3,4,5,6,7});
}
