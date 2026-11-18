"""Cold row-weight packing and actual resident-O launch boundaries."""
import pytest
import torch

from rpu_backend.adapters.hy_vla.resident_o import (
    pack_expert_o_w8_installed, pack_expert_o_w8_layers,
)
from rpu_backend.runtime.weights import transform_linear_weight
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run


def test_installed_row_pack_matches_each_raw_core_and_retains_carrier_bits():
    raw = ((torch.arange(1024 * 2048, dtype=torch.int32).reshape(1024, 2048) * 101 + 73) % 256 - 128).to(torch.int8)
    installed = transform_linear_weight(raw, partition=0, num_cores=8)
    packed = pack_expert_o_w8_installed(installed)
    assert packed.shape == (8, 262144) and packed.dtype == torch.int8 and packed.is_contiguous()
    for core in range(8):
        expected = transform_linear_weight(raw[:, core * 256:(core + 1) * 256].contiguous(), partition=0, num_cores=1)
        assert torch.equal(packed[core], expected.flatten())
    weights = [(installed.to(torch.int16) + layer).to(torch.int8) for layer in range(12)]
    carrier = pack_expert_o_w8_layers(weights)
    assert carrier.dtype == torch.float16 and carrier.shape == (8, 1572864) and carrier.is_contiguous()
    view = carrier.view(torch.int8).reshape(8, 12, 262144)
    for layer, w in enumerate(weights):
        assert torch.equal(view[:, layer], pack_expert_o_w8_installed(w))
    assert torch.equal(installed, transform_linear_weight(raw, partition=0, num_cores=8))


def test_packing_rejects_wrong_precision_shape_and_layer_count():
    for weight in (None, torch.empty(1024, 2048, dtype=torch.float16),
                   torch.empty(1023, 2048, dtype=torch.int8), torch.empty(1024, 2047, dtype=torch.int8),
                   torch.empty(1, 1024, 2048, dtype=torch.int8)):
        with pytest.raises(ValueError, match='installed INT8'):
            pack_expert_o_w8_installed(weight)
    for count in (0, 6, 11, 13):
        with pytest.raises(ValueError, match='exactly twelve'):
            pack_expert_o_w8_layers([torch.empty(0)] * count)


def test_actual_launcher_ranges_aliases_scales_and_typed_registers(tmp_path):
    launch = block((ROOT/'src/fused/hyvla_w8_o_resident.h').read_text(), 'inline void launch_hyvla_w8_o_resident_spm(')
    source = COMMON + r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId : uint16_t {HYVLA_EXPERT_O_RESIDENT_W8_S51};
enum class GraphDdrRegisterRole {PerChannelScale};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;
 bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}
 uint32_t addr(int c,uint32_t o)const{return base+uint32_t(c)*0x1000000u+o;}
} spm;
#define SPM_ALLOC spm
struct Kernel {std::array<uint16_t,128> regs{};
 void reset_regs(){regs.fill(0);}
 void set_regs(uint32_t i,uint16_t v){regs.at(i)=v;}
 void set_regs(uint32_t,int)=delete;
} kernel;
int fetched=0,queued=0,staged=0,flushed=0;
Kernel* get_kernel(){++fetched;return &kernel;}
#define GET_KERNEL(id) get_kernel()
struct Writer {std::vector<GraphDdrRegisterOperandSpec> ops;
 void write(Kernel& k){for(const auto& o:ops){const uint64_t a=o.tensor.address>>8;
  k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph {static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  assert(id==KernelId::HYVLA_EXPERT_O_RESIDENT_W8_S51 && v.size()==1);++staged;
  assert(v[0].abi.role==GraphDdrRegisterRole::PerChannelScale && v[0].abi.access==GraphDdrRegisterAccess::Read);
  assert(v[0].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi && v[0].abi.lo==20 && v[0].abi.hi==21);
  assert(v[0].tensor.nbytes()==1024*2);return {v};}};
void rpu_ddr_flush(void*){++flushed;}
namespace rhino_lkn {uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}}
struct Queue {bool broadcast=false;
 void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(broadcast && grid==std::vector<uint16_t>({8,1,1}));
  assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++queued;}
} queue;
#define GET_QUEUE(n) (&::queue)
LAUNCH
int main(){
 const uint32_t base=spm.base;
 const std::array<uint32_t,3> good{base+0x10000,base+0x200000,base+0x400000};
 at::Tensor scale;scale.shape={1024};
 auto run=[&](std::array<uint32_t,3> a,bool pass,int rows=51){
  int before=fetched,previous=queued;bool ok=true;
  try{launch_hyvla_w8_o_resident_spm(a[0],scale,a[1],a[2],rows);}
  catch(const std::exception&){ok=false;}
  assert(ok==pass && fetched==before+int(pass) && queued==previous+int(pass));
  assert(staged==queued && flushed==queued);
 };
 run(good,true);
 auto pair=[&](int i){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+1])<<16);};
 assert(pair(0)==good[0] && pair(4)==good[1] && pair(2)==good[2]);
 assert(kernel.regs[8]==0 && kernel.regs[9]==51 && pair(10)==1024 && kernel.regs[12]==256 && kernel.regs[13]==16);
 assert(pair(16)==512 && pair(18)==2048 && pair(22)==0 && pair(24)==0 && kernel.regs[26]==2 && kernel.regs[27]==1);
 assert(pair(20)==scale.address>>8);
 const std::array<uint32_t,3> bytes{51*256*2,51*1024*2,262144};
 for(int i=0;i<3;++i){
  for(auto b:{base-256,good[i]+1,base+SpmAllocator::SPM_USABLE-256}){auto a=good;a[i]=b;run(a,false);}
  for(int j=0;j<3;++j)if(i!=j){auto a=good;a[i]=good[j];run(a,false);
   a[i]=good[j]+256;run(a,false);}
  auto a=good;a[i]=base+SpmAllocator::SPM_USABLE-bytes[i];run(a,true);
  a[i]+=256;run(a,false);
 }
 for(int r:{0,1,50,52,64})run(good,false,r);
 const auto good_scale=scale;
 for(int failure=0;failure<9;++failure){scale=good_scale;
  switch(failure){case 0:scale.present=false;break;case 1:scale.contiguous=false;break;
   case 2:scale.dtype=at::kChar;break;case 3:scale.where.kind=c10::DeviceType::CPU;break;
   case 4:scale.shape={1023};break;case 5:scale.shape={1025};break;
   case 6:scale.shape={1,1024};break;case 7:scale.address+=128;break;
   case 8:scale.address=(uint64_t{1}<<40)-256;break;}
  run(good,false);
 }
 scale=good_scale;
 spm.ready=false;run(good,false);spm.ready=true;
 spm.base=0xf8ff0000;auto high=good;for(auto& a:high)a+=spm.base-base;run(high,false);
}
'''
    compile_run(tmp_path, 'w8_o_resident_launcher', source.replace('LAUNCH', launch))
