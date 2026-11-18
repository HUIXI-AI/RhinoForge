"""Execute production SSPM candidate bodies against typed host DMA/queue records."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_sspm_candidate_dma_owners_absolute_registers_and_admission(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/ops/rpu_pi_nvfp4_sspm.cpp").read_text()
    source = re.sub(r'^#include "[^\n]+\n', '', source, flags=re.MULTILINE)
    harness = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "rpu_pi05_q_kv1_abi.h"
#define TORCH_CHECK(c,...) if (!(c)) { throw std::runtime_error("rejected"); }
namespace rhino_lkn {}
namespace c10 {using Half=uint16_t;}
int flushes=0,preloads=0,stages=0,kept=0,lookups=0,submits=0,counter_writes=0;
bool capturing=true,loaded=true;
namespace at {
constexpr int kByte=1,kHalf=2,kPrivateUse1=3,kChar=4;
using IntArrayRef=std::vector<int64_t>;
struct Device {int type_=kPrivateUse1,index=0;int type()const{return type_;}
 bool operator==(Device b)const{return type_==b.type_&&index==b.index;}};
struct Tensor {IntArrayRef shape;uint64_t address=0x1200000000ULL;Device dev;
 bool present=true,contiguous=true;int dtype=kByte;int64_t offset=0;
 bool defined()const{return present;}Device device()const{return dev;}
 int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}
 int64_t storage_offset()const{return offset;}IntArrayRef sizes()const{return shape;}
 int64_t dim()const{return shape.size();}int64_t size(int64_t i)const{return shape.at(i);}
 uint64_t nbytes()const{uint64_t n=dtype==kHalf?2:1;for(auto d:shape)n*=d;return n;}
 void* data_ptr()const{return reinterpret_cast<void*>(address);}
};
}
struct SpmAllocator {static constexpr uint64_t SPM_USABLE=8*1024*1024-1024;
 bool initialized=true,bad_last_core=false;bool is_initialized()const{return initialized;}
 uint32_t addr(int core,int offset)const{return bad_last_core&&core==7 ? UINT32_MAX-16:
     0x40000000u+uint32_t(core)*0x08000000u+offset;}} spm;
#define SPM_ALLOC spm
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uint64_t>(p);}
void rpu_ddr_flush(void*){++flushes;}
enum class KernelId{PI05_NVFP4_LINEAR_DB_SSPM,PI05_NVFP4_GEGLU_DB_SSPM,PI05_Q_KV1_PAIR_DB_SSPM};
struct KernelCache{static KernelCache&instance(){static KernelCache x;return x;}
 bool has_loaded(KernelId)const{return loaded;}};
enum class GraphDdrRegisterRole{PackedFp4ModelWeight,Fp8BlockScale,Int8ModelWeight,PerChannelScale};
enum class GraphDdrRegisterAccess{Read};
enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct Abi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{Abi abi;at::Tensor tensor;};
struct Kernel_t{std::map<int,uint16_t>regs;
 void reset_regs(){regs.clear();}
 void set_regs(int slot,uint16_t v){regs[slot]=v;}
 void set_regs(int slot,uint32_t v){assert((slot==128||slot==130)&&v==0);regs[slot]=v;regs[slot+1]=v>>16;++counter_writes;}
 uint32_t pair(int s)const{return uint32_t(regs.at(s))|(uint32_t(regs.at(s+1))<<16);}} kernel;
std::vector<GraphDdrRegisterOperandSpec> operands;
KernelId selected;
struct Writer{void write(Kernel_t&k){for(auto&o:operands){
 assert(o.abi.hi==o.abi.lo+1);const uint32_t a=o.tensor.address>>8;
 k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph {static bool has_active(){return capturing;}
 static RpuKernelGraph&active(){static RpuKernelGraph g;return g;}
 void keep_alive(const at::Tensor&t){assert(t.present&&t.dtype==at::kByte);++kept;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> ops){
 assert(preloads>0);selected=id;operands=ops;++stages;return {};}};
enum class GraphOuterFastDmaSide{Source};
namespace graph_dma {bool pending=false;at::Tensor owner;
 bool active(){return capturing;}
 void stage_census_fixed_burst(const at::Tensor&t,GraphOuterFastDmaSide side,size_t begin,size_t bytes,size_t occurrences){
 assert(!pending&&side==GraphOuterFastDmaSide::Source&&begin==0&&bytes==t.nbytes()&&occurrences==8);
 owner=t;pending=true;}}
struct Dma {uint64_t address;uint32_t destination;int64_t bytes;};
std::vector<Dma> dma;
void rpu_launch_ddr_scatter_spm_dma(c10::Half*p,int64_t words,int64_t stride,uint32_t dst,int cores){
 assert(graph_dma::pending&&cores==8&&words*2==stride&&uint64_t(stride)*8==graph_dma::owner.nbytes());
 assert(reinterpret_cast<uint64_t>(p)==graph_dma::owner.address&&dst%256==0);
 dma.push_back({reinterpret_cast<uint64_t>(p),dst,stride});graph_dma::pending=false;++preloads;}
Kernel_t* get_kernel(KernelId id){assert(id==selected);++lookups;return &kernel;}
#define GET_KERNEL(id) get_kernel(id)
struct Queue{bool broadcast=false;std::vector<uint16_t>grid;
 void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel_t&k,std::vector<uint16_t>g,std::vector<uint8_t>cores){
 assert(broadcast&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));
 assert(g==std::vector<uint16_t>({k.regs.at(64),k.regs.at(65),k.regs.at(66)}));grid=g;++submits;}}queue;
Queue*get_queue(int cores){assert(cores==8);return &queue;}
#define GET_QUEUE(c) get_queue(c)
__SOURCE__
template<class F>void reject(F f){const std::array<int,7>before{flushes,preloads,stages,kept,lookups,submits,counter_writes};
 bool failed=false;try{f();}catch(const std::runtime_error&){failed=true;}assert(failed);
 assert((before==std::array<int,7>{flushes,preloads,stages,kept,lookups,submits,counter_writes}));assert(!graph_dma::pending);}
int main(){
 std::array<uint32_t,6>a{0x40010000,0x40040000,0x40080000,0x400a0000,0x400c0000,0x400d0000};
 for(int projection:{0,1,2}){
  int64_t n=projection?1024:2048,k=projection==2?4096:projection?2048:1024;
  int col=!projection;int64_t ln=col?n/8:n,lk=col?k:k/8;
  at::Tensor w{{n,k/2}},s{{8,ln/64,lk/128,512}};s.address=0x1400000000ULL;
  int64_t rows=50;int cores=8;uint16_t layer=17;
  auto launch=[&]{rpu_launch_pi05_nvfp4_sspm_candidate_kernel(a[0],w,a[1],rows,n,k,col,cores,s,a[2],a[4],layer);};
  for(int r:{1,16,50,64})for(int replay=0;replay<2;++replay){rows=r;size_t start=dma.size();launch();
   assert(dma.size()==start+1&&dma.back().address==s.address&&dma.back().destination==a[2]);
   assert(dma.back().bytes==ln*lk/16&&kernel.pair(20)==a[2]&&kernel.pair(2)==w.address/256);
   assert(kernel.pair(24)==a[4]&&kernel.regs[8]==rows&&kernel.pair(10)==ln&&kernel.regs[12]==lk);
   assert(kernel.regs[13]==lk/64&&kernel.regs[15]==8&&kernel.pair(16)==lk*2&&kernel.pair(18)==ln*2);
   assert(kernel.regs[26]==col&&kernel.regs[27]==17&&operands.size()==1&&counter_writes==0);
  }rows=50;
  for(int bad:{0,65}){rows=bad;reject(launch);}rows=50;cores=4;reject(launch);cores=8;layer=18;reject(launch);layer=17;
  capturing=false;reject(launch);capturing=true;loaded=false;reject(launch);loaded=true;
  for(auto*owner:{&w,&s})for(int fault=0;fault<10;++fault){auto saved=*owner;
   if(fault==0)owner->present=false;if(fault==1)owner->dtype=at::kHalf;if(fault==2)owner->dev.type_=0;
   if(fault==3)owner->contiguous=false;if(fault==4)owner->offset=1;if(fault==5)--owner->shape[0];
   if(fault==6)owner->address=0;if(fault==7)++owner->address;if(fault==8)owner->address=1ULL<<40;
   if(fault==9)owner->address=(1ULL<<40)-256;reject(launch);*owner=saved;}
  {auto save=s;s.address=w.address;reject(launch);s=save;s.dev.index=1;reject(launch);s=save;}
  for(int i:{0,1,2,4}){auto save=a[i];for(auto v:{0u,save+1,0x407fff00u}){a[i]=v;reject(launch);}a[i]=save;}
  for(int i:{0,1,2,4})for(int j:{0,1,2,4})if(i>j){auto save=a[i];a[i]=a[j];reject(launch);a[i]=save;}
  spm.initialized=false;reject(launch);spm.initialized=true;spm.bad_last_core=true;reject(launch);spm.bad_last_core=false;
 }
 at::Tensor gw{{4096,512}},uw=gw,gs{{8,8,8,512}},us=gs;
 uw.address=0x1300000000ULL;gs.address=0x1400000000ULL;us.address=0x1500000000ULL;
 auto geglu=[&]{rpu_launch_pi05_nvfp4_geglu_sspm_candidate_kernel(a[0],gw,uw,a[1],gs,us,a[2],a[3],a[4],a[5],17);};
 for(int replay=0;replay<3;++replay){auto start=dma.size();int reset=counter_writes;geglu();
  assert(dma.size()==start+2&&dma[start].address==gs.address&&dma[start+1].address==us.address);
  assert(dma[start].destination==a[2]&&dma[start+1].destination==a[3]&&dma[start].bytes==32768);
  assert(operands.size()==2&&kernel.pair(2)==gw.address/256&&kernel.pair(32)==uw.address/256);
  assert(kernel.pair(20)==a[2]&&kernel.pair(34)==a[3]&&kernel.pair(24)==a[4]&&kernel.pair(36)==a[5]);
  assert(counter_writes==reset+2&&kernel.pair(128)==0&&kernel.pair(130)==0);
 }
 for(int i=0;i<6;++i)for(int j=0;j<i;++j){auto save=a[i];a[i]=a[j];reject(geglu);a[i]=save;}
 for(auto*owner:{&gw,&uw,&gs,&us}){auto save=*owner;owner->dev.index=1;reject(geglu);*owner=save;
  owner->shape.back()-=1;reject(geglu);*owner=save;}
 us.address=gs.address;reject(geglu);
 std::array<at::Tensor,6> owners{{{{2048,512}},{{8,4,8,512}},{{2048,1024}},{{2048}},{{2048,1024}},{{2048}}}};
 for(size_t i=0;i<owners.size();++i) owners[i].address=0x1200000000ULL+i*0x100000000ULL;
 owners[2].dtype=owners[4].dtype=at::kChar;owners[3].dtype=owners[5].dtype=at::kHalf;
 int64_t rows=50;uint16_t layer=17;
 auto fused=[&]{rpu_launch_pi05_q_kv1_sspm_candidate_kernel(a[0],owners[0],owners[1],a[4],a[5],a[1],
     owners[2],owners[3],a[2],owners[4],owners[5],a[3],rows,layer);};
 for(int row:{1,50,64})for(int replay=0;replay<2;++replay){rows=row;auto start=dma.size();int resets=counter_writes;fused();
  assert(selected==KernelId::PI05_Q_KV1_PAIR_DB_SSPM&&dma.size()==start+1&&dma.back().bytes==16384);
  assert(dma.back().destination==a[4]&&dma.back().address==owners[1].address);
  assert(operands.size()==5&&counter_writes==resets);
  const std::array<int,5> slots{2,34,52,54,56},owner_ids{0,2,3,4,5};
  const std::array<GraphDdrRegisterRole,5> roles{GraphDdrRegisterRole::PackedFp4ModelWeight,
    GraphDdrRegisterRole::Int8ModelWeight,GraphDdrRegisterRole::PerChannelScale,
    GraphDdrRegisterRole::Int8ModelWeight,GraphDdrRegisterRole::PerChannelScale};
  for(size_t i=0;i<5;++i){assert(operands[i].abi.lo==slots[i]&&operands[i].abi.role==roles[i]);
    assert(operands[i].tensor.address==owners[owner_ids[i]].address);
    assert(kernel.pair(slots[i])==owners[owner_ids[i]].address/256);}
  assert(kernel.pair(0)==a[0]&&kernel.pair(32)==a[0]&&kernel.pair(4)==a[1]);
  assert(kernel.pair(36)==a[2]&&kernel.pair(58)==a[3]&&kernel.pair(20)==a[4]&&kernel.pair(24)==a[5]);
  assert(kernel.regs[8]==rows&&kernel.regs[40]==rows&&kernel.regs[27]==17&&kernel.regs[28]==4);
  assert(kernel.regs[44]==64&&kernel.regs[45]==0&&kernel.regs[47]==1&&kernel.regs[64]==6);
 }
 rows=50;for(int bad:{0,65}){rows=bad;reject(fused);}rows=50;layer=18;reject(fused);layer=17;
 capturing=false;reject(fused);capturing=true;loaded=false;reject(fused);loaded=true;
 for(auto&owner:owners)for(int fault=0;fault<11;++fault){auto saved=owner;
   if(fault==0)owner.present=false;if(fault==1)owner.dtype=owner.dtype==at::kHalf?at::kChar:at::kHalf;
   if(fault==2)owner.dev.type_=0;if(fault==3)owner.contiguous=false;if(fault==4)owner.offset=1;
   if(fault==5)--owner.shape[0];if(fault==6)owner.address=0;if(fault==7)++owner.address;
   if(fault==8)owner.address=1ULL<<40;if(fault==9)owner.address=(1ULL<<40)-256;
   if(fault==10)owner.dev.index=1;reject(fused);owner=saved;
 }
 for(size_t i=0;i<owners.size();++i)for(size_t j=0;j<i;++j){auto saved=owners[i].address;
   owners[i].address=owners[j].address;reject(fused);owners[i].address=saved;}
 for(int i=0;i<6;++i){auto save=a[i];for(auto bad:{0u,save+1,0x407fff00u}){a[i]=bad;reject(fused);}a[i]=save;}
 for(int i=0;i<6;++i)for(int j=0;j<i;++j){auto save=a[i];a[i]=a[j];reject(fused);a[i]=save;}
 spm.initialized=false;reject(fused);spm.initialized=true;spm.bad_last_core=true;reject(fused);
}
'''.replace("__SOURCE__", source)
    cpp, exe = tmp_path / "sspm.cpp", tmp_path / "sspm"
    cpp.write_text(harness)
    result = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src/ops"), str(cpp), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
