"""Execute Action W8 launchers with the real tile/register helper and owned DDR census."""
from pathlib import Path
import pytest
from test_g05_q_gate_native import _compile_run
from test_siglip_cold_profile import _body
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("linear_acc32", [True, False], ids=["acc32", "acc16"])
def test_action_w8_actual_launchers_scales_pending_kernel_and_resident_preload(tmp_path, linear_acc32):
    source = (ROOT / 'src/ops/rpu_linear.cpp').read_text()
    names = ['rpu_launch_g05_kv_owner_linear_w8a16_acc32_kernel',
             'rpu_launch_g05_q_gate_linear_w8a16_acc32_kernel',
             'rpu_launch_g05_gate_up_linear_w8a16_acc32_kernel',
             'rpu_launch_g05_gateup_resident_w8a16_kernel',
             'rpu_launch_g05_gateup_resident_w8a16_preload', 'check_pi05_resident_spm']
    functions = []
    for name in names:
        start = source.index('void ' + name + '(')
        brace = source.index('{', start)
        functions.append(source[start:brace] + _body(source, 'void ' + name + '('))
    cpp = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>
#include "rpu_linear_tiling.h"
constexpr bool TEST_ACC32=__MODE__;
#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10{using Half=uint16_t;enum class DeviceType{PrivateUse1,CPU};}
namespace at{constexpr int kHalf=1,kChar=2;
struct Tensor{uint64_t address;bool present=true,contiguous=true;int dtype=kChar,rank=2;int64_t rows=2048,cols=1024;c10::DeviceType location=c10::DeviceType::PrivateUse1;
 bool defined()const{return present;}int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}int dim()const{return rank;}int64_t size(int i)const{return i?cols:rows;}
 size_t nbytes()const{return size_t(rows)*(rank==2?cols:1)*(dtype==kHalf?2:1);}
 struct Device{c10::DeviceType value;c10::DeviceType type()const{return value;}};Device device()const{return {location};}template<class T>T*data_ptr()const{return reinterpret_cast<T*>(address);}
};}
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8303616;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int core,int off)const{assert(core==0);return 0x20000000+off;}}allocator;
#define SPM_ALLOC allocator
#define IDS G05_KV_OWNER_W8A16_ACC32_M32N256K1024,G05_Q_GATE_W8A16_ACC32_M32N256K1024,G05_GATE_UP_W8A16_ACC32_M32N512K1024,G05_GATEUP_RESIDENT_W8A16_ACC32_M240N96K128,G05_KV_OWNER_FANOUT_M32D256,G05_KV_OWNER_W8A16_ACC16_M32N256K1024,G05_Q_GATE_W8A16_ACC16_M32N256K1024,G05_GATE_UP_W8A16_ACC16_M32N512K1024,G05_GATEUP_RESIDENT_W8A16_ACC16_M352N96K128
// The unique pending kernel catches a premature GET of the fanout.
enum class KernelId{IDS};
constexpr auto KV=TEST_ACC32?KernelId::G05_KV_OWNER_W8A16_ACC32_M32N256K1024:KernelId::G05_KV_OWNER_W8A16_ACC16_M32N256K1024,QG=TEST_ACC32?KernelId::G05_Q_GATE_W8A16_ACC32_M32N256K1024:KernelId::G05_Q_GATE_W8A16_ACC16_M32N256K1024,GU=TEST_ACC32?KernelId::G05_GATE_UP_W8A16_ACC32_M32N512K1024:KernelId::G05_GATE_UP_W8A16_ACC16_M32N512K1024,RES=TEST_ACC32?KernelId::G05_GATEUP_RESIDENT_W8A16_ACC32_M240N96K128:KernelId::G05_GATEUP_RESIDENT_W8A16_ACC16_M352N96K128,FAN=KernelId::G05_KV_OWNER_FANOUT_M32D256;
struct KernelCache{std::set<KernelId>missing;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId id)const{return !missing.count(id);}};
enum class GraphDdrRegisterRole{Int8ModelWeight,PerChannelScale};enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor owner;};
struct Kernel{std::map<int,uint16_t>regs;void reset_regs(){regs.clear();}void set_regs(int i,uint16_t v){regs[i]=v;}void set_regs(int,uint32_t)=delete;};
struct Submission{KernelId id;std::map<int,uint16_t>regs;std::vector<GraphDdrRegisterOperandSpec>operands;std::vector<uint16_t>grid;};struct Writer{void write(Kernel&);};
struct RpuKernelGraph{std::optional<KernelId>staged,pending;bool written=false;int stages=0,lookups=0;Kernel kernel;std::vector<GraphDdrRegisterOperandSpec>operands;std::vector<Submission>submissions;
 static RpuKernelGraph&active(){static RpuKernelGraph graph;return graph;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec>specs){assert(!staged&&!pending&&specs.size()==(id==RES?1:4));staged=id;written=false;operands=std::move(specs);++stages;return {};}
 Kernel*get(KernelId id){assert(!pending&&KernelCache::instance().has_loaded(id));assert((id==FAN&&!staged)||(staged&&*staged==id));pending=id;written=id==FAN;++lookups;return &kernel;}
};
void Writer::write(Kernel&kernel){auto&g=RpuKernelGraph::active();assert(g.staged&&g.pending&&*g.staged==*g.pending&&!g.written);for(auto&operand:g.operands){uint64_t v=operand.owner.address>>8;kernel.set_regs(operand.abi.lo,static_cast<uint16_t>(v));kernel.set_regs(operand.abi.hi,static_cast<uint16_t>(v>>16));}g.written=true;}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
std::vector<uint64_t>flushed;template<class T>void rpu_ddr_flush(T*p){flushed.push_back(reinterpret_cast<uint64_t>(p));}
struct Queue{bool broadcast=false;void set_broadcast_mode(bool v){broadcast=v;}void enqueu_kernel(Kernel&k,std::vector<uint16_t>grid,std::vector<uint8_t>cores){auto&g=RpuKernelGraph::active();assert(broadcast&&g.pending&&g.written);assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));g.submissions.push_back({*g.pending,k.regs,g.operands,grid});g.operands.clear();g.staged.reset();g.pending.reset();g.written=false;}}queue;
Queue*get_queue(int cores){assert(cores==8);return &queue;}
#define GET_QUEUE(cores) get_queue(cores)
enum class GraphOuterFastDmaSide{Source};
namespace graph_dma{bool active(){return true;}bool staged=false;uint64_t owner=0;size_t bytes=0,packets=0;int count=0;
void stage_census_fixed_burst(const at::Tensor&t,GraphOuterFastDmaSide,size_t begin,size_t extent,size_t n){assert(!staged&&begin==0&&extent==t.nbytes());owner=t.address;bytes=extent;packets=n;staged=true;++count;}}
int dma_launches=0;void rpu_launch_ddr_scatter_spm_dma(c10::Half*p,int64_t elements,int64_t stride,uint32_t spm,int cores){assert(graph_dma::staged&&graph_dma::owner==reinterpret_cast<uint64_t>(p)&&graph_dma::bytes==8388608&&graph_dma::packets==8);assert(elements==524288&&stride==1048576&&cores==8&&spm==allocator.addr(0,0)+0x100000);graph_dma::staged=false;++dma_launches;}
void check_pi05_resident_spm(uint32_t,size_t);
__FUNCTIONS__
uint32_t pair(const Submission&s,int reg){return uint32_t(s.regs.at(reg))|(uint32_t(s.regs.at(reg+1))<<16);}
template<class F>void reject(F f){auto&g=RpuKernelGraph::active();auto before=g.submissions.size(),flushes=flushed.size();int stages=g.stages,lookups=g.lookups,dma=dma_launches,census=graph_dma::count;try{f();assert(false);}catch(const std::runtime_error&){}assert(g.submissions.size()==before&&flushed.size()==flushes&&g.stages==stages&&g.lookups==lookups&&dma==dma_launches&&census==graph_dma::count&&!g.staged&&!g.pending);}
using Launch=void(*)(uint32_t,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,uint32_t,uint32_t,bool);
int main(){
 auto&g=RpuKernelGraph::active();const uint32_t base=allocator.addr(0,0),in=base+0x10000,out=in+65536,out2=out+32768;
 const std::array<Launch,3>launches{rpu_launch_g05_kv_owner_linear_w8a16_acc32_kernel,rpu_launch_g05_q_gate_linear_w8a16_acc32_kernel,rpu_launch_g05_gate_up_linear_w8a16_acc32_kernel};
 for(int mode=0;mode<3;++mode){auto launch=[&](auto&&...args){launches[mode](args...,TEST_ACC32);};const int N=mode==2?4096:2048;const auto id=mode==0?KV:mode==1?QG:GU;const int grid=mode==0?6:mode==1?12:14;
  at::Tensor a{0x1234567800ULL},b{0x2345678900ULL},as{0x3456789a00ULL},bs{0x456789ab00ULL};a.rows=b.rows=N;as.rows=bs.rows=N;as.dtype=bs.dtype=at::kHalf;as.rank=bs.rank=1;
  for(int pass=0;pass<2;++pass){a.address+=256;b.address+=512;as.address+=768;bs.address+=1024;auto begin=g.submissions.size();launch(in,a,b,as,bs,out,out2);assert(g.submissions.size()==begin+(mode==0?2:1));const auto&s=g.submissions[begin];assert(s.id==id&&s.grid==std::vector<uint16_t>({uint16_t(grid),1,1}));
   assert(pair(s,0)==in&&pair(s,2)==a.address>>8&&pair(s,4)==out&&pair(s,32)==b.address>>8&&pair(s,36)==out2&&pair(s,20)==as.address>>8&&pair(s,34)==bs.address>>8);
   assert(pair(s,6)==0&&s.regs.at(8)==1&&s.regs.at(9)==32&&pair(s,10)==uint32_t(N/8)&&s.regs.at(12)==1024&&s.regs.at(13)==64&&s.regs.at(15)==8&&pair(s,16)==2048&&pair(s,18)==uint32_t(N/4)&&pair(s,22)==0&&pair(s,24)==0&&s.regs.at(26)==2&&s.regs.at(27)==1);
   assert(s.operands[0].abi.role==GraphDdrRegisterRole::Int8ModelWeight&&s.operands[1].abi.role==GraphDdrRegisterRole::Int8ModelWeight&&s.operands[2].abi.role==GraphDdrRegisterRole::PerChannelScale&&s.operands[3].abi.role==GraphDdrRegisterRole::PerChannelScale);
   if(mode==0){const auto&f=g.submissions.back();assert(f.id==FAN&&f.operands.empty()&&f.grid==std::vector<uint16_t>({1,1,1})&&pair(f,0)==out-base&&pair(f,2)==out2-base);}
  }
  for(auto missing:{id,FAN}){if(missing==FAN&&mode!=0)continue;KernelCache::instance().missing.insert(missing);reject([&]{launch(in,a,b,as,bs,out,out2);});KernelCache::instance().missing.clear();}
  for(int index=0;index<4;++index)for(int fault=0;fault<7;++fault){auto aa=a,bb=b,sa=as,sb=bs;auto*t=index==0?&aa:index==1?&bb:index==2?&sa:&sb;if(fault==0)t->present=false;if(fault==1)t->dtype=index<2?at::kHalf:at::kChar;if(fault==2)t->contiguous=false;if(fault==3)t->location=c10::DeviceType::CPU;if(fault==4)t->rank=index<2?1:2;if(fault==5)t->rows=1024;if(fault==6){if(index<2)t->cols=512;else t->rows=N+1;}reject([&]{launch(in,aa,bb,sa,sb,out,out2);});}
  for(auto offsets:{std::array<uint32_t,3>{in,in,out2},{in,out,out},{in,out,base+SpmAllocator::SPM_USABLE-256},{in+1,out,out2},{base-256,out,out2}})reject([&]{launch(offsets[0],a,b,as,bs,offsets[1],offsets[2]);});
 }
 at::Tensor scale{0x6789abcd00ULL};scale.dtype=at::kHalf;scale.rank=1;scale.rows=4096;const uint32_t weight=base+0x100000;
 rpu_launch_g05_gateup_resident_w8a16_kernel(in,weight,out,scale);const auto&r=g.submissions.back();assert(r.id==RES&&r.grid==std::vector<uint16_t>({7,1,1})&&r.operands.size()==1&&pair(r,2)==weight&&pair(r,20)==scale.address>>8&&pair(r,18)==1024&&r.regs.at(26)==2&&r.regs.at(27)==1);
 reject([&]{rpu_launch_g05_gateup_resident_w8a16_kernel(in,in,out,scale);});KernelCache::instance().missing.insert(RES);reject([&]{rpu_launch_g05_gateup_resident_w8a16_kernel(in,weight,out,scale);});KernelCache::instance().missing.clear();auto invalid=scale;invalid.rows=512;reject([&]{rpu_launch_g05_gateup_resident_w8a16_kernel(in,weight,out,invalid);});
 at::Tensor packed{0x789abcde00ULL};packed.rows=8;packed.cols=1048576;rpu_launch_g05_gateup_resident_w8a16_preload(packed,weight);assert(dma_launches==1);auto bad=packed;bad.dtype=at::kHalf;reject([&]{rpu_launch_g05_gateup_resident_w8a16_preload(bad,weight);});bad=packed;bad.cols=524288;reject([&]{rpu_launch_g05_gateup_resident_w8a16_preload(bad,weight);});reject([&]{rpu_launch_g05_gateup_resident_w8a16_preload(packed,base+SpmAllocator::SPM_USABLE-524288);});
}
'''
    _compile_run(tmp_path, 'action_w8_launchers', cpp.replace('__MODE__', str(linear_acc32).lower()).replace('__FUNCTIONS__', '\n'.join(functions).replace('bool linear_acc32)', 'bool linear_acc32 = TEST_ACC32)')))
