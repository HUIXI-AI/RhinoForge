"""Execute VLM W8 launcher ownership and exact-profile failure boundaries."""
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run


def test_actual_vlm_w8_launch_owned_registers_and_complete_regions(tmp_path):
    launch = block((ROOT / 'src/ops/rpu_hyvla_vlm_gateup_w8.h').read_text(),
                   'inline void launch_hyvla_vlm_gateup_w8(')
    source = COMMON + r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId : uint16_t {HYVLA_VLM_GATEUP_SWIGLU_W8_V2};
enum class GraphDdrRegisterRole {Int8ModelWeight,PerChannelScale};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;
 GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel {std::array<uint16_t,68> regs{};void reset_regs(){regs.fill(0);}
 uint32_t set_regs(uint32_t i,uint16_t x){regs.at(i)=x;return 0;}
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
  assert(id==KernelId::HYVLA_VLM_GATEUP_SWIGLU_W8_V2);
  assert(v.size()==4);++staged;
  for(size_t i=0;i<v.size();++i){auto& o=v[i];bool scale=i%2==1;
   assert(o.abi.lo==std::array<int,4>{2,20,32,34}[i]);
   assert(o.abi.role==(scale?GraphDdrRegisterRole::PerChannelScale:GraphDdrRegisterRole::Int8ModelWeight));
   assert(o.tensor.nbytes()==size_t(scale?12288:12582912));assert(o.abi.hi==o.abi.lo+1);
   assert(o.abi.access==GraphDdrRegisterAccess::Read && o.abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);}
  return {v};}
};
struct Queue {void set_broadcast_mode(bool b){assert(b);}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(grid==std::vector<uint16_t>({10,1,1}));
  assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++queued;}
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
 ts[0].shape=ts[1].shape={6144,2048};ts[0].dtype=ts[1].dtype=at::kChar;
 ts[2].shape=ts[3].shape={6144};
 for(int i=0;i<4;++i)ts[i].address+=i*0x1000000;
 uint32_t in=0x20000000,out=0x20200000;
 auto run=[&](bool expected,int rows=240){const int q=queued,s=staged,k=kept;bool ok=true;
  try{launch_hyvla_vlm_gateup_w8(in,out,rows,ts[0],ts[1],ts[2],ts[3]);}
  catch(const std::exception&){ok=false;}
  assert(ok==expected && queued==q+int(expected));
  assert(staged==s+int(expected) && kept==k+4*int(expected));};
 run(true);auto pair=[&](int i){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+1])<<16);};
 assert(pair(0)==in && pair(4)==out && kernel.regs[8]==1 && kernel.regs[9]==240 && pair(10)==768);
 assert(kernel.regs[12]==2048 && kernel.regs[13]==128 && kernel.regs[15]==8);
 assert(pair(16)==4096 && pair(18)==1536 && pair(22)==0 && pair(24)==0 && kernel.regs[26]==2 && kernel.regs[27]==1);
 assert(pair(2)==ts[0].address>>8 && pair(32)==ts[1].address>>8);
 assert(pair(20)==ts[2].address>>8 && pair(34)==ts[3].address>>8);
 for(int r:{0,51,239,241,320})run(false,r);
 for(auto* addr:{&in,&out}){auto saved=*addr;
  for(auto invalid:{0x1fffff00u,0x20000001u,0x207fff00u}){*addr=invalid;run(false);}*addr=saved;}
 out=in+240*2048*2;run(true);out-=256;run(false);
 out=spm.base+SpmAllocator::SPM_USABLE-240*768*2;run(true);out+=256;run(false);out=0x20200000;
 for(int i=0;i<4;++i){auto t=ts[i];
  ts[i].present=false;run(false);ts[i]=t;ts[i].contiguous=false;run(false);ts[i]=t;
  ts[i].dtype=at::kFloat;run(false);ts[i]=t;ts[i].shape[0]-=1;run(false);ts[i]=t;
  ts[i].where.index=1;run(false);ts[i]=t;ts[i].where.kind=c10::DeviceType::CPU;run(false);ts[i]=t;
  ts[i].address+=128;run(false);ts[i]=t;}
 spm.base=0xf8ff0000;in=spm.base;out=spm.base+0x200000;run(false);
 spm.base=0x20000000;in=spm.base;out=0x20200000;spm.ready=false;run(false);
}
'''
    source = source.replace('assert(o.abi.lo==std::array<int,4>{2,20,32,34}[i]);',
                            'assert((o.abi.lo==std::array<int,4>{2,20,32,34}[i]));')
    compile_run(tmp_path, 'vlm_w8_gateup', source.replace('LAUNCH', launch))
