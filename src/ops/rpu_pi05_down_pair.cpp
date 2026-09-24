#include "rpu_pi05_down_pair.h"
#include "rpu_ops.h"
#include "rpu_linear_tiling.h"
#include "rpu_spm_allocator.h"
#include "rhino_launch_buffer.h"
#include "rhino_launch_queue.h"
#include <array>
#include <vector>

using namespace ::rhino_lkn;

void rpu_launch_pi05_down_pair_spm_kernel(
    uint32_t input0, uint32_t input1, const at::Tensor& weight,
    uint32_t output0, uint32_t output1, const at::Tensor& scale, int64_t rows) {
    TORCH_CHECK(rows == 272 || rows == 288 || rows == 304 || rows == 320 || rows == 400 || rows == 416 || rows == 432 || rows == 448,
                "Pi Down pair requires C272/C288/C304/C320/C400/C416/C432/C448");
    const bool fp16 = weight.defined() && weight.scalar_type() == at::kHalf;
    const uint32_t M=rows;
    constexpr uint32_t N=2048, K=2048, Cores=8;
    const uint64_t Bytes=uint64_t(M)*N*2;
    TORCH_CHECK(SPM_ALLOC.is_initialized(), "Pi Down pair requires initialized SPM");
    const uint32_t base=SPM_ALLOC.addr(0,0);
    const std::array<uint32_t,4> ranges{input0,input1,output0,output1};
    for(size_t i=0;i<ranges.size();++i) {
        TORCH_CHECK(ranges[i]>=base && ranges[i]%256==0 &&
                    uint64_t(ranges[i])-base<=SpmAllocator::SPM_USABLE-Bytes,
                    "Pi Down pair invalid absolute SPM interval");
        for(size_t j=0;j<i;++j)
            TORCH_CHECK(uint64_t(ranges[i])+Bytes<=ranges[j] ||
                        uint64_t(ranges[j])+Bytes<=ranges[i],
                        "Pi Down pair requires four disjoint SPM intervals");
    }
    TORCH_CHECK(weight.defined() && weight.device().type()==at::kPrivateUse1 &&
                (fp16 || weight.scalar_type()==at::kChar) && weight.is_contiguous() &&
                weight.sizes()==at::IntArrayRef({N,K*Cores}),
                "Pi Down pair requires row-striped W8 [2048,16384]");
    TORCH_CHECK(fp16 ? !scale.defined() : (scale.defined() && scale.device().type()==at::kPrivateUse1 &&
                scale.scalar_type()==at::kHalf && scale.is_contiguous() &&
                scale.sizes()==at::IntArrayRef({N})),
                "Pi Down pair requires FP16 scale [2048]");
    // Each named paired payload owns its fixed tile schedule (C400 uses M160).
    // Generic single-input AUTO selection does not describe this two-slab ABI.
    const KernelId id = rows == 432
        ? (fp16 ? KernelId::PI05_DOWN_WEIGHT_OUTER_FP16_C432X2_M112N80K128
                : KernelId::PI05_DOWN_WEIGHT_OUTER_W8A16_C432X2_M112N80K128) :
        rows == 448
        ? (fp16 ? KernelId::PI05_DOWN_WEIGHT_OUTER_FP16_C448X2_M96N80K128
                : KernelId::PI05_DOWN_WEIGHT_OUTER_W8A16_C448X2_M96N80K128) :
        rows == 416
        ? (fp16 ? KernelId::PI05_DOWN_WEIGHT_OUTER_FP16_C416X2_M128N80K128
                : KernelId::PI05_DOWN_WEIGHT_OUTER_W8A16_C416X2_M128N80K128)
        : rows == 288
        ? (fp16 ? KernelId::PI05_DOWN_WEIGHT_OUTER_FP16_C288X2_M160N80K128
                : KernelId::PI05_DOWN_WEIGHT_OUTER_W8A16_C288X2_M160N80K128)
        : rows == 304
        ? (fp16 ? KernelId::PI05_DOWN_WEIGHT_OUTER_FP16_C304X2_M160N80K128
                : KernelId::PI05_DOWN_WEIGHT_OUTER_W8A16_C304X2_M160N80K128) :
        rows == 320
        ? (fp16 ? KernelId::PI05_DOWN_WEIGHT_OUTER_FP16_C320X2_M160N80K128
                : KernelId::PI05_DOWN_WEIGHT_OUTER_W8A16_C320X2_M160N80K128)
        : fp16 ?
        (rows == 400 ? KernelId::PI05_DOWN_WEIGHT_OUTER_FP16_C400X2_M160N80K128 :
                       KernelId::PI05_DOWN_WEIGHT_OUTER_FP16_C272X2_M160N80K128) :
        (rows == 400 ? KernelId::PI05_DOWN_WEIGHT_OUTER_W8A16_C400X2_M160N80K128 :
                       KernelId::PI05_DOWN_WEIGHT_OUTER_W8A16_C272X2_M160N80K128);
    std::vector<GraphDdrRegisterOperandSpec> operands{
            {GraphDdrRegisterAbi{fp16 ? GraphDdrRegisterRole::ModelWeight : GraphDdrRegisterRole::Int8ModelWeight,
                GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},weight}};
    if (!fp16) operands.push_back({GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
                GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},scale});
    auto writer=RpuKernelGraph::active().stage_kernel_ddr_registers(id, operands);
    for(const auto* t:{&weight,&scale}) {
        if (!t->defined()) continue;
        const uint64_t address=RpuGetDevAddr(t->data_ptr());
        TORCH_CHECK(address && address%256==0 && (address>>8)<=UINT32_MAX,
                    "Pi Down pair DDR operand exceeds aligned shift8 ABI");
        rpu_ddr_flush(t->data_ptr());
    }
    auto* kernel=GET_KERNEL(id);
    TORCH_CHECK(kernel,"Pi Down pair experimental expansion symbol is missing");
    kernel->reset_regs();
    rpu_pl_tiling::set_parallel_linear_regs(*kernel,
        rpu_pl_tiling::ParallelLinearRegisterArgs{
            input0,0,output0,0,M,N,K,0,Cores,K*2,N*2,0,0,uint16_t(fp16 ? 0 : 1)});
    const auto set_pair=[kernel](uint32_t index,uint32_t value) {
        TORCH_CHECK(kernel->set_regs(index,uint16_t(value))==kKernelRegOk &&
                    kernel->set_regs(index+1,uint16_t(value>>16))==kKernelRegOk,
                    "Pi Down pair rejected extra SPM register pair");
    };
    set_pair(24,input1);set_pair(26,output1);
    writer.write(*kernel);
    auto* queue=GET_QUEUE(Cores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel,{26,1,1},{0,1,2,3,4,5,6,7});
}
