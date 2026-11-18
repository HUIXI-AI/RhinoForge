"""Exercise the actual fused collective launcher at its memory/ABI boundaries."""
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run


def test_actual_launcher_ranges_aliases_typed_registers_and_peer_table(tmp_path):
    launch=block((ROOT/'src/fused/hyvla_o_ring_postnorm.h').read_text(), 'inline void launch_hyvla_o_ring_postnorm_spm(')
    source=COMMON+r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId {HYVLA_EXPERT_O_RING_POSTNORM_S51};
enum class GraphDdrRegisterRole {ModelWeight};
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
struct Kernel {std::array<uint16_t,4160> regs{};uint32_t phase=99;
 void reset_regs(){regs.fill(0);phase=99;}
 void set_regs(uint32_t i,uint16_t v){regs.at(i)=v;}
 void set_regs(uint32_t i,uint32_t v){assert(i==128);phase=v;}
 void set_regs(uint32_t,int)=delete;
} kernel;
void rpu_set_legacy_scm_u16_checked(Kernel* k,uint32_t i,uint16_t v,const char*){k->set_regs(i,v);}
int fetched=0,queued=0,staged=0,flushed=0;
Kernel* get_kernel(){++fetched;return &kernel;}
#define GET_KERNEL(id) get_kernel()
struct Writer {std::vector<GraphDdrRegisterOperandSpec> ops;
 void write(Kernel& k){for(const auto& o:ops){const uint64_t a=o.tensor.address>>8;
  k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph {static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  assert(id==KernelId::HYVLA_EXPERT_O_RING_POSTNORM_S51 && v.size()==1);++staged;
  assert(v[0].abi.role==GraphDdrRegisterRole::ModelWeight && v[0].abi.access==GraphDdrRegisterAccess::Read);
  assert(v[0].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi && v[0].abi.lo==2 && v[0].abi.hi==3);
  assert(v[0].tensor.nbytes()==1024*2048*2);return {v};}};
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
 const std::array<uint32_t,5> good{base+0x10000,base+0x200000,base+0x400000,base,base+0x600000};
 at::Tensor weight;weight.shape={1024,2048};
 auto run=[&](std::array<uint32_t,5> a,bool pass,int rows=51,int cols=1024,int nr=64,double eps=1e-5,int cores=8){
  int before=fetched,previous=queued;bool ok=true;
  try{launch_hyvla_o_ring_postnorm_spm(a[4],weight,a[0],a[1],a[2],a[3],rows,cols,nr,eps,cores);}
  catch(const std::exception&){ok=false;}
  assert(ok==pass && fetched==before+int(pass) && queued==previous+int(pass));
  assert(staged==queued && flushed==queued);
 };
 run(good,true);
 auto pair=[&](int i,int step=1){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+step])<<16);};
 assert(pair(0)==good[4] && pair(4)==good[0] && pair(2)==weight.address>>8);
 assert(kernel.regs[8]==0 && kernel.regs[9]==51 && pair(10)==1024 && kernel.regs[12]==256 && kernel.regs[13]==16);
 assert(pair(16)==512 && pair(18)==2048 && pair(22)==0 && pair(24)==0 && kernel.regs[26]==2 && kernel.regs[27]==2);
 assert(pair(36)==good[1]-base && pair(44)==good[2] && pair(46)==good[1] && pair(48)==good[3]);
 assert(kernel.regs[32]==8 && kernel.regs[33]==1 && pair(34)==63488 && kernel.regs[38]==0);
 assert(kernel.regs[40]==1024 && kernel.regs[41]==c10::Half(1.f/1024).x && kernel.regs[42]==2048);
 assert(kernel.regs[43]==8 && kernel.regs[50]==c10::Half(1e-5).x && kernel.phase==0);
 for(int c=0;c<8;++c){
  assert(pair(4096+c,8)==spm.addr(c,good[0]-base)-base);
  assert(pair(4112+c,8)==spm.addr(c,good[2]-base)-base);
  assert(pair(4128+c,8)==uint32_t(c*6528*2) && pair(4144+c,8)==6528);
 }
 const std::array<uint32_t,5> bytes{51*1024*2,64*1024*2,64*1024*2,2048,51*256*2};
 for(int i=0;i<5;++i){
  for(auto b:{base-256,good[i]+1,base+SpmAllocator::SPM_USABLE-256}){auto a=good;a[i]=b;run(a,false);}
  for(int j=0;j<5;++j)if(i!=j){auto a=good;a[i]=good[j];run(a,false);
   a[i]=good[j]+256;run(a,false);}
  auto a=good;a[i]=base+SpmAllocator::SPM_USABLE-bytes[i];run(a,true);
  a[i]+=256;run(a,false);
 }
 for(int r:{0,50,52,64})run(good,false,r);
 for(int c:{0,128,1023,1025,2048})run(good,false,51,c);
 for(int n:{0,51,63,65})run(good,false,51,1024,n);
 for(double e:{0.,1e-6,-1e-5})run(good,false,51,1024,64,e);
 for(int n:{0,1,4,9})run(good,false,51,1024,64,1e-5,n);
 const auto good_weight=weight;
 for(int failure=0;failure<9;++failure){weight=good_weight;
  switch(failure){case 0:weight.present=false;break;case 1:weight.contiguous=false;break;
   case 2:weight.dtype=at::kChar;break;case 3:weight.where.kind=c10::DeviceType::CPU;break;
   case 4:weight.shape={1024,2047};break;case 5:weight.shape={1023,2048};break;
   case 6:weight.shape={1,1024,2048};break;case 7:weight.address+=128;break;
   case 8:weight.address=(uint64_t{1}<<40)-256;break;}
  run(good,false);
 }
 weight=good_weight;
 spm.ready=false;run(good,false);spm.ready=true;
 spm.base=0xf8ff0000;auto high=good;for(auto& a:high)a+=spm.base-base;run(high,false);
}
'''
    compile_run(tmp_path,'ring_postnorm_launcher',source.replace('LAUNCH',launch))
