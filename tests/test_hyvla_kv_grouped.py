"""Execute the real paired projection helper, selected pointers and ABI guards."""
import re

import pytest
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run


@pytest.mark.parametrize('w8', [False, True])
def test_actual_helper_owners_registers_and_failure_boundaries(tmp_path, w8):
    launch = block((ROOT / 'src/fused/hyvla_kv_grouped.h').read_text(),
                   'inline void launch_hyvla_kv_grouped_spm(')
    source = COMMON + r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId : uint16_t {HYVLA_EXPERT_KV_FP16_S51,HYVLA_EXPERT_KV_W8_S51};
constexpr auto expected_id=W8?KernelId::HYVLA_EXPERT_KV_W8_S51:KernelId::HYVLA_EXPERT_KV_FP16_S51;
enum class GraphDdrRegisterRole {ModelWeight,Int8ModelWeight,PerChannelScale};
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
struct Kernel {std::array<uint16_t,68> regs{};
 void reset_regs(){regs.fill(0);}
 void set_regs(uint32_t i,uint16_t v){regs.at(i)=v;}
 void set_regs(uint32_t,uint32_t)=delete;void set_regs(uint32_t,int)=delete;
} kernel;
int fetched=0,queued=0,staged=0,flushed=0,kept=0;bool available=true;
Kernel* get_kernel(KernelId id){assert(id==expected_id);++fetched;return available?&kernel:nullptr;}
#define GET_KERNEL(id) get_kernel(id)
struct Writer {std::vector<GraphDdrRegisterOperandSpec> ops;
 void write(Kernel& k){for(const auto& o:ops){const uint64_t a=o.tensor.address>>8;
  k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph {static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 void keep_alive(const at::Tensor& t){assert(t.defined());++kept;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  assert(id==expected_id && v.size()==(W8?4:2));++staged;
  for(size_t i=0;i<v.size();++i){const auto& o=v[i];const bool scale=W8 && i%2;const int p=i/(W8?2:1);
   assert(o.abi.role==(scale?GraphDdrRegisterRole::PerChannelScale:
    (W8?GraphDdrRegisterRole::Int8ModelWeight:GraphDdrRegisterRole::ModelWeight)));
   assert(o.abi.access==GraphDdrRegisterAccess::Read && o.abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
   assert(o.abi.lo==(scale?(p?34:20):(p?32:2)) && o.abi.hi==o.abi.lo+1);
   assert(o.tensor.nbytes()==size_t(scale?2048:1024*1024*(W8?1:2)));}
  return {v};}};
void rpu_ddr_flush(void*){++flushed;}
namespace rhino_lkn {uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}}
struct Queue {bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(broadcast && grid==std::vector<uint16_t>({W8?6:4,1,1}));
  assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++queued;}
} queue;
#define GET_QUEUE(n) (&::queue)
LAUNCH
int main(){
 const uint32_t base=spm.base;uint32_t input=base+0x10000;
 std::vector<uint32_t> out{base+0x100000,base+0x200000};
 std::vector<at::Tensor> weight(2),scale(2);
 for(size_t i=0;i<2;++i){weight[i].shape={1024,1024};weight[i].dtype=W8?at::kChar:at::kHalf;
  weight[i].address+=i*0x400000;scale[i].shape={1024};scale[i].dtype=at::kHalf;
  scale[i].address=weight[i].address+0x200000;scale[i].present=W8;}
 auto run=[&](bool pass,int rows=51,int cores=8){
  const int f=fetched,q=queued;bool ok=true;
  try{launch_hyvla_kv_grouped_spm(input,weight,scale,out,rows,cores);}catch(const std::exception&){ok=false;}
  assert(ok==pass && fetched==f+int(pass) && queued==q+int(pass));
  assert(staged==queued && flushed==(W8?4:2)*queued && kept==(W8?4:2)*queued);
 };
 auto pair=[&](int i){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+1])<<16);};
 run(true);
 assert(pair(0)==input && pair(6)==0 && kernel.regs[8]==1 && kernel.regs[9]==51 && pair(10)==128);
 assert(kernel.regs[12]==1024 && kernel.regs[13]==64 && kernel.regs[14]==0 && kernel.regs[15]==8);
 assert(pair(16)==2048 && pair(18)==256 && pair(22)==0 && pair(24)==0 && kernel.regs[26]==2 && kernel.regs[27]==(W8?1:2));
 assert(kernel.regs[64]==(W8?6:4) && kernel.regs[65]==1 && kernel.regs[66]==1);
 for(size_t i=0;i<2;++i){assert(pair(i?32:2)==weight[i].address>>8 && pair(i?36:4)==out[i]);
  assert(pair(i?34:20)==(W8?scale[i].address>>8:0));}
 for(int rows:{0,50,52,64})run(false,rows);for(int cores:{0,1,4,9})run(false,51,cores);
 for(auto* v:{&weight,&scale}){auto t=v->back();v->pop_back();run(false);v->push_back(t);v->push_back(t);run(false);v->pop_back();}
 {auto t=out.back();out.pop_back();run(false);out.push_back(t);out.push_back(t);run(false);out.pop_back();}
 std::vector<uint32_t*> addresses{&input,&out[0],&out[1]};std::vector<uint32_t> bytes{51*1024*2,51*128*2,51*128*2};
 for(size_t i=0;i<addresses.size();++i){auto& a=*addresses[i];const auto good=a;
  for(uint32_t bad:{base-256,good+1}){a=bad;run(false);}a=good;
  for(size_t j=0;j<addresses.size();++j)if(i!=j){a=*addresses[j];run(false);a=*addresses[j]+256;run(false);a=good;}
  a=base+((SpmAllocator::SPM_USABLE-bytes[i])&~255u);run(true);a+=256;run(false);a=good;
 }
 for(auto* v:{&weight,&scale})for(auto& t:*v){auto good=t;
  if(v==&scale && !W8){t.present=true;run(false);t=good;continue;}
  for(int failure=0;failure<11;++failure){t=good;
   switch(failure){case 0:t.present=false;break;case 1:t.contiguous=false;break;
    case 2:t.dtype=at::kByte;break;case 3:t.where.kind=c10::DeviceType::CPU;break;
    case 4:--t.shape[0];break;case 5:t.shape.push_back(1);break;
    case 6:t.address+=128;break;case 7:t.address=(uint64_t{1}<<40)-256;break;
    case 8:t.where.index=1;break;case 9:t.address=0;break;case 10:t.shape[0]=1;break;}run(false);}
  t=good;t.address=(uint64_t{1}<<40)-t.nbytes();run(true);t.address=uint64_t{1}<<40;run(false);t=good;
 }
 {available=false;const int f=fetched,s=staged,q=queued,k=kept;bool rejected=false;
  try{launch_hyvla_kv_grouped_spm(input,weight,scale,out);}catch(const std::exception&){rejected=true;}
  assert(rejected && fetched==f+1 && staged==s && queued==q && kept==k);available=true;}
 spm.ready=false;run(false);spm.ready=true;
 spm.base=0xf8ff0000;for(auto* a:addresses)*a+=spm.base-base;run(false);
}
'''
    compile_run(tmp_path, 'kv_grouped', re.sub(r'\bW8\b', 'true' if w8 else 'false',
                                             source.replace('LAUNCH', launch)))
