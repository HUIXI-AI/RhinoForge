"""Execute actual HyVLA W8 typed-operand and SPM bounds checks on host."""
import re

import pytest
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run

@pytest.mark.parametrize('resident,narrow,fused', [(False, False, False), (True, False, False), (False, True, False), (True, False, True)])
def test_actual_w8_launch_boundaries_and_scale_ownership(tmp_path, resident, narrow, fused):
    launch = block((ROOT/'src/ops/rpu_hyvla_gateup_w8.h').read_text(), 'inline void launch_hyvla_gateup_w8(')
    source = COMMON + r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId : uint16_t {WALL_ACTION_GATE_UP_W8A16_MULTIBASE_M592N32_SPM_WEIGHT,WALL_PREFILL_GATE_UP_SWIGLU_W8A16_M320N112,HYVLA_EXPERT_GATEUP_W8_S51_N48,HYVLA_W8_RESIDENT_SWIGLU_S51};
enum class GraphDdrRegisterRole {Int8ModelWeight,PerChannelScale};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel {std::array<uint16_t,132> regs{};void reset_regs(){regs.fill(0);}
 uint32_t set_regs(uint32_t i,uint16_t x){regs.at(i)=x;return 0;}
 uint32_t set_regs(uint32_t i,uint32_t x){assert(i==128 && x==0);regs.at(i)=x;return 0;}
 uint32_t set_regs(uint32_t,int32_t)=delete;
} kernel;
int queued=0,staged=0,kept=0;
struct Writer {std::vector<GraphDdrRegisterOperandSpec> operands;
 void write(Kernel& k){for(auto& o:operands){auto a=o.tensor.address>>8;
  k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph {
 static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 void keep_alive(const at::Tensor&){++kept;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  assert(id==(RESIDENT?(FUSED?KernelId::HYVLA_W8_RESIDENT_SWIGLU_S51:KernelId::HYVLA_GATEUP_RESIDENT_PAIR_W8_V2):NARROW?KernelId::HYVLA_EXPERT_GATEUP_W8_S51_N48:KernelId::HYVLA_EXPERT_GATEUP_W8_S51_N48));
  assert(v.size()==(RESIDENT?2:4));++staged;
  for(auto& o:v){bool scale=o.abi.lo==20||o.abi.lo==34;
   assert(o.abi.role==(scale?GraphDdrRegisterRole::PerChannelScale:GraphDdrRegisterRole::Int8ModelWeight));
   assert(o.tensor.nbytes()==size_t(scale?4096:2097152));assert(o.abi.hi==o.abi.lo+1);}
  return {v};}
};
struct Queue {void set_broadcast_mode(bool b){assert(b);}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(grid==std::vector<uint16_t>({RESIDENT?(FUSED?2:4):2,1,1}));assert(cores.size()==8);++queued;}
} queue;
#define GET_KERNEL(id) (&::kernel)
#define GET_QUEUE(n) (&::queue)
namespace rhino_lkn {uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}}
void rpu_ddr_flush(void*){}
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;
 uint32_t base=0x20000000;bool is_initialized()const{return ready;}
 uint32_t addr(int c,uint32_t o)const{return base+uint32_t(c)*0x1000000u+o;}} spm;
#define SPM_ALLOC spm
LAUNCH
int main(){
 std::array<at::Tensor,4> ts;
 ts[0].shape=ts[1].shape={2048,1024};ts[0].dtype=ts[1].dtype=at::kChar;
 ts[2].shape=ts[3].shape={2048};
 for(int i=0;i<4;++i)ts[i].address+=i*0x400000;
 std::array<uint32_t,5> a{0x20000000,0x20020000,RESIDENT?0x20030000u:0u,RESIDENT?0x20100000u:0u,RESIDENT?0x20140000u:0u};
 auto run=[&](bool expected,int rows=51){const int q=queued,s=staged,k=kept;bool ok=true;
  try{launch_hyvla_gateup_w8(a[0],a[1],a[2],a[3],a[4],rows,ts[0],ts[1],ts[2],ts[3],RESIDENT,NARROW,FUSED);}
  catch(const std::exception&){ok=false;}assert(ok==expected && queued==q+int(expected));
  assert(staged==s+int(expected) && kept==k+4*int(expected));};
 run(true);auto pair=[&](int i){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+1])<<16);};
 assert(kernel.regs[8]==1 && kernel.regs[9]==51 && pair(10)==256 && pair(20)==ts[2].address>>8 && pair(34)==ts[3].address>>8);
 assert(pair(2)==(RESIDENT?a[3]:ts[0].address>>8));assert(pair(32)==(RESIDENT?a[4]:ts[1].address>>8));
 for(int r:{0,20,50,64})run(false,r);
 for(int i=0;i<(RESIDENT?5:2);++i){auto saved=a[i];a[i]+=1;run(false);a[i]=0x207fff00;run(false);a[i]=saved;}
 auto saved=a[1];a[1]=a[0];run(false);a[1]=saved;
 if(RESIDENT){a[4]=a[3];run(false);a[4]=0x20140000;}
 for(int i=0;i<4;++i){auto t=ts[i];
  ts[i].present=false;run(false);ts[i]=t;ts[i].contiguous=false;run(false);ts[i]=t;
  ts[i].dtype=at::kFloat;run(false);ts[i]=t;ts[i].shape[0]-=1;run(false);ts[i]=t;
  ts[i].where.index=1;run(false);ts[i]=t;
  if(!RESIDENT||i>=2){ts[i].address+=128;run(false);ts[i]=t;}}
 spm.ready=false;run(false);
}
'''
    compile_run(tmp_path, 'w8_gateup', re.sub(r'\b(RESIDENT|NARROW|FUSED)\b', lambda match: str({'RESIDENT': resident, 'NARROW': narrow, 'FUSED': fused}[match[1]]).lower(), source.replace('LAUNCH', launch)))
