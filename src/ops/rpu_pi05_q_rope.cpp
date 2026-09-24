#include "rpu_pi05_q_rope.h"
#include "rpu_ops.h"
#include "rpu_linear_tiling.h"
#include "rpu_spm_allocator.h"
#include "rhino_launch_buffer.h"
#include "rhino_launch_queue.h"
#include <array>
#include <utility>
#include <vector>

using namespace ::rhino_lkn;

void rpu_launch_pi05_q_rope_spm_kernel(
    uint32_t input,const at::Tensor& weight,uint32_t output,
    const at::Tensor& scale,const at::Tensor& cos,const at::Tensor& sin,
    int64_t logical_position) {
    constexpr uint32_t M=50,N=256,K=1024;
    TORCH_CHECK(SPM_ALLOC.is_initialized(),"Pi Q-RoPE requires initialized SPM");
    const uint32_t base=SPM_ALLOC.addr(0,0);
    for(const auto& r:std::array<std::pair<uint32_t,uint32_t>,2>{{
            {input,M*K*2},{output,M*N*2}}}) {
        TORCH_CHECK(r.first>=base && r.first%256==0 &&
            uint64_t(r.first)-base<=SpmAllocator::SPM_USABLE-r.second,
            "Pi Q-RoPE invalid absolute SPM interval");
    }
    TORCH_CHECK(uint64_t(input)+M*K*2<=output || uint64_t(output)+M*N*2<=input,
                "Pi Q-RoPE requires disjoint input and output");
    const auto tensor=[](const at::Tensor& t,at::ScalarType dtype) {
        return t.defined() && t.device().type()==at::kPrivateUse1 &&
            t.scalar_type()==dtype && t.layout()==c10::Layout::Strided && t.is_contiguous();
    };
    TORCH_CHECK(tensor(weight,at::kChar) && weight.sizes()==at::IntArrayRef({2048,1024}) &&
        tensor(scale,at::kHalf) && scale.sizes()==at::IntArrayRef({2048}),
        "Pi Q-RoPE requires original TP-column signed W8 and FP16 per-channel scale");
    TORCH_CHECK(tensor(cos,at::kHalf) && tensor(sin,at::kHalf) && cos.dim()==2 &&
        cos.sizes()==sin.sizes() && cos.size(1)==128 &&
        cos.size(0)<=UINT32_MAX/256 && logical_position>=0 &&
        logical_position<=cos.size(0) && M<=cos.size(0)-logical_position,
        "Pi Q-RoPE logical position exceeds owned FP16 frequency tables");
    const auto tile=rpu_pl_tiling::select_tile_acc16(M,N,K,false);
    TORCH_CHECK(tile.m_tile==592 && tile.n_tile==32,"Pi Q-RoPE original tile drift");
    for(const auto* t:{&weight,&scale,&cos,&sin}) {
        const uint64_t address=RpuGetDevAddr(t->data_ptr());
        TORCH_CHECK(address && address%256==0 && (address>>8)<=UINT32_MAX,
                    "Pi Q-RoPE DDR operand exceeds aligned shift8 ABI");
        rpu_ddr_flush(t->data_ptr());
    }
    constexpr auto id=KernelId::PI05_DENOISE_Q_PAIR16_ROPE_M50N256K1024;
    auto writer=RpuKernelGraph::active().stage_kernel_ddr_registers(id,
        std::vector<GraphDdrRegisterOperandSpec>{
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
                GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},weight},
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
                GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},scale},
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::SemanticInput,
                GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},cos},
            {GraphDdrRegisterAbi{GraphDdrRegisterRole::SemanticInput,
                GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,26,27},sin}});
    auto* kernel=GET_KERNEL(id);
    TORCH_CHECK(kernel,"Pi Q-RoPE experimental expansion symbol is missing");
    kernel->reset_regs();
    rpu_pl_tiling::set_parallel_linear_regs(*kernel,rpu_pl_tiling::ParallelLinearRegisterArgs{
        input,0,output,0,M,N,K,/*has_bias=*/0,8,K*2,N*2,0,/*partition=*/1,1});
    TORCH_CHECK(kernel->set_regs(28,uint16_t(logical_position))==kKernelRegOk &&
        kernel->set_regs(29,uint16_t(uint64_t(logical_position)>>16))==kKernelRegOk,
        "Pi Q-RoPE rejected logical position registers");
    writer.write(*kernel);
    auto* queue=GET_QUEUE(8);queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel,{8,1,1},{0,1,2,3,4,5,6,7});
}
