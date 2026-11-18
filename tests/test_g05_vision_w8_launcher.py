"""Exercise the actual triple-base launcher, typed owners and SPM boundaries."""
from pathlib import Path
import pytest

from test_g05_q_gate_native import _compile_run
from test_siglip_cold_profile import _body

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("linear_acc32", [True, False], ids=["acc32", "acc16"])
def test_vision_w8_triple_actual_launcher_typed_scales_and_spm(tmp_path, linear_acc32):
    source = (ROOT / 'src/ops/rpu_linear.cpp').read_text()
    cpp = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>
#include "rpu_linear_tiling.h"
constexpr bool TEST_ACC32=__MODE__;
#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10{using Half=uint16_t;enum class DeviceType{PrivateUse1,CPU};}
namespace at{constexpr int kHalf=1,kChar=2;
struct Tensor{uint64_t address;bool present=true,contiguous=true;int dtype=kChar,rank=2;int64_t rows=1024,cols=1024;c10::DeviceType location=c10::DeviceType::PrivateUse1;
 bool defined()const{return present;}int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}int dim()const{return rank;}int64_t size(int i)const{return i?cols:rows;}
 struct Device{c10::DeviceType value;c10::DeviceType type()const{return value;}};Device device()const{return {location};}template<class T>T*data_ptr()const{return reinterpret_cast<T*>(address);}
};}
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8303616;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int core,int off)const{assert(core==0);return 0x20000000+off;}}allocator;
#define SPM_ALLOC allocator
enum class KernelId{G05_VISION_QKV_W8A16_ACC32_M256N128K1024,G05_VISION_QKV_W8A16_ACC16_M256N128K1024};constexpr auto ID=TEST_ACC32?KernelId::G05_VISION_QKV_W8A16_ACC32_M256N128K1024:KernelId::G05_VISION_QKV_W8A16_ACC16_M256N128K1024;
struct KernelCache{bool available=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId id)const{assert(id==ID);return available;}};
enum class GraphDdrRegisterRole{Int8ModelWeight,PerChannelScale};enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor owner;};
struct Kernel{std::map<int,uint16_t>regs;void reset_regs(){regs.clear();}void set_regs(int i,uint16_t v){regs[i]=v;}void set_regs(int,uint32_t)=delete;};
struct Submission{std::map<int,uint16_t>regs;std::vector<GraphDdrRegisterOperandSpec>operands;};struct Writer{void write(Kernel&);};
struct RpuKernelGraph{bool staged=false,pending=false,written=false;int stages=0,lookups=0;Kernel kernel;std::vector<GraphDdrRegisterOperandSpec>operands;std::vector<Submission>submissions;
 static RpuKernelGraph&active(){static RpuKernelGraph graph;return graph;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec>specs){assert(id==ID&&!staged&&!pending&&specs.size()==6);staged=true;written=false;operands=std::move(specs);++stages;return {};}
 Kernel*get(KernelId id){assert(id==ID&&staged&&!pending&&KernelCache::instance().available);pending=true;++lookups;return &kernel;}
};
void Writer::write(Kernel&kernel){auto&g=RpuKernelGraph::active();assert(g.staged&&g.pending&&!g.written);for(auto&operand:g.operands){uint64_t v=operand.owner.address>>8;kernel.set_regs(operand.abi.lo,static_cast<uint16_t>(v));kernel.set_regs(operand.abi.hi,static_cast<uint16_t>(v>>16));}g.written=true;}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
std::vector<uint64_t>flushed;template<class T>void rpu_ddr_flush(T*p){flushed.push_back(reinterpret_cast<uint64_t>(p));}
struct Queue{bool broadcast=false;void set_broadcast_mode(bool v){broadcast=v;}void enqueu_kernel(Kernel&k,std::vector<uint16_t>grid,std::vector<uint8_t>cores){auto&g=RpuKernelGraph::active();assert(broadcast&&g.staged&&g.pending&&g.written);assert(grid==std::vector<uint16_t>({9,1,1})&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));g.submissions.push_back({k.regs,g.operands});g.operands.clear();g.staged=g.pending=g.written=false;}}queue;
Queue*get_queue(int cores){assert(cores==8);return &queue;}
#define GET_QUEUE(cores) get_queue(cores)
void launch(uint32_t input,const at::Tensor&query_weight,const at::Tensor&key_weight,const at::Tensor&value_weight,uint32_t query_output,uint32_t key_output,uint32_t value_output,uint32_t query_bias,uint32_t key_bias,uint32_t value_bias,const at::Tensor&query_scale,const at::Tensor&key_scale,const at::Tensor&value_scale,bool linear_acc32=TEST_ACC32)__BODY__
uint32_t pair(const Submission&s,int reg){return uint32_t(s.regs.at(reg))|(uint32_t(s.regs.at(reg+1))<<16);}
using Regions=std::array<uint32_t,7>;using Weights=std::array<at::Tensor,3>;
Weights scales{{{0x456789ab00ULL,true,true,at::kHalf,1},{0x56789abc00ULL,true,true,at::kHalf,1},{0x6789abcd00ULL,true,true,at::kHalf,1}}};
void run(const Regions&r,const Weights&w){launch(r[0],w[0],w[1],w[2],r[1],r[2],r[3],r[4],r[5],r[6],scales[0],scales[1],scales[2]);}
template<class F>void reject(F f){auto&g=RpuKernelGraph::active();auto before=g.submissions.size(),flushes=flushed.size();int stages=g.stages,lookups=g.lookups;try{f();assert(false);}catch(const std::runtime_error&){}assert(g.submissions.size()==before&&flushed.size()==flushes&&g.stages==stages&&g.lookups==lookups&&!g.staged&&!g.pending);}
int main(){
 auto&g=RpuKernelGraph::active();const uint32_t base=allocator.addr(0,0),end=base+SpmAllocator::SPM_USABLE;
 const Regions regions{base,base+524288,base+589824,base+655360,base+720896,base+721408,base+721920};
 const Weights weights{{{0x1234567800ULL},{0x2345678900ULL},{0x3456789a00ULL}}};
 for(uint32_t pass=0;pass<2;++pass){auto r=regions;auto w=weights;for(auto&address:r)address+=pass*0x100000;for(auto&weight:w)weight.address+=pass*0x400000;run(r,w);const auto&s=g.submissions.back();
  assert(pair(s,0)==r[0]&&pair(s,4)==r[1]&&pair(s,36)==r[2]&&pair(s,44)==r[3]);
  assert(pair(s,6)==r[4]&&pair(s,38)==r[5]&&pair(s,46)==r[6]);
  assert(s.regs.at(8)==1&&s.regs.at(9)==256&&pair(s,10)==128&&s.regs.at(12)==1024&&s.regs.at(13)==64&&s.regs.at(14)==1&&s.regs.at(15)==8);
  assert(pair(s,16)==2048&&pair(s,18)==256&&pair(s,22)==0&&pair(s,24)==0&&s.regs.at(26)==2&&s.regs.at(27)==1);
  assert(s.operands.size()==6);const int regs[3]={2,32,40};for(int i=0;i<3;++i){assert(pair(s,regs[i])==w[i].address>>8&&s.operands[i].owner.address==w[i].address&&flushed[pass*6+i]==w[i].address);const auto&a=s.operands[i].abi;assert(a.role==GraphDdrRegisterRole::Int8ModelWeight&&a.access==GraphDdrRegisterAccess::Read&&a.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi&&a.lo==regs[i]&&a.hi==regs[i]+1);}
 }
 for(int i=0;i<3;++i){const auto&s=g.submissions.back();const int regs[3]={20,34,42};assert(pair(s,regs[i])==scales[i].address>>8);const auto&a=s.operands[3+i].abi;assert(a.role==GraphDdrRegisterRole::PerChannelScale&&a.lo==regs[i]&&a.hi==regs[i]+1&&s.operands[3+i].owner.address==scales[i].address&&flushed[9+i]==scales[i].address);}
 for(int which=0;which<3;++which)for(int fault=0;fault<6;++fault){auto saved=scales;auto&t=scales[which];if(fault==0)t.present=false;if(fault==1)t.dtype=at::kChar;if(fault==2)t.contiguous=false;if(fault==3)t.location=c10::DeviceType::CPU;if(fault==4)t.rank=2;if(fault==5)t.rows=1023;reject([&]{run(regions,weights);});scales=saved;}
 // Check each full interval, including the exact final legal bias/cache line.
 const uint32_t sizes[7]={524288,65536,65536,65536,256,256,256};
 for(int i=0;i<7;++i){auto r=regions;r[i]=end-sizes[i];run(r,weights);r[i]+=256;reject([&]{run(r,weights);});r=regions;r[i]+=1;reject([&]{run(r,weights);});r[i]=base-256;reject([&]{run(r,weights);});r[i]=0xffffff00u;reject([&]{run(r,weights);});}
 // Every output/bias/input pairing is rejected before typed staging or GET.
 for(int i=0;i<7;++i)for(int j=i+1;j<7;++j){auto r=regions;r[j]=r[i]+sizes[i]-256;reject([&]{run(r,weights);});}
 allocator.initialized=false;reject([&]{run(regions,weights);});allocator.initialized=true;KernelCache::instance().available=false;reject([&]{run(regions,weights);});KernelCache::instance().available=true;
 for(int which=0;which<3;++which)for(int fault=0;fault<7;++fault){auto w=weights;auto&t=w[which];if(fault==0)t.present=false;if(fault==1)t.dtype=at::kHalf;if(fault==2)t.contiguous=false;if(fault==3)t.location=c10::DeviceType::CPU;if(fault==4)t.rank=1;if(fault==5)t.rows=2048;if(fault==6)t.cols=512;reject([&]{run(regions,w);});}
 run(regions,weights);
}
'''
    cpp = cpp.replace('__BODY__', _body(source, 'void rpu_launch_g05_vision_qkv_linear_w8a16_acc32_kernel('))
    _compile_run(tmp_path, 'vision_qkv_launcher', cpp.replace('__MODE__', str(linear_acc32).lower()))
