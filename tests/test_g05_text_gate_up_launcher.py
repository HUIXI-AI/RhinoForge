"""Run the production Text Gate/Up wrapper with actual auto-tile register helpers."""
from pathlib import Path

import pytest

from test_g05_q_gate_native import _compile_run
from test_siglip_cold_profile import _body

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("w8,linear_acc32", [(False,True),(True,True),(True,False)], ids=["fp16-acc32", "w8a16-acc32", "w8a16-acc16"])
def test_text_gate_up_actual_launcher_ddr_owners_grid_and_spm_boundaries(tmp_path, w8, linear_acc32):
    source = (ROOT / 'src/ops/rpu_linear.cpp').read_text()
    cpp = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>
#include "rpu_linear_tiling.h"
constexpr bool W8=__W8__,TEST_ACC32=__MODE__;
#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10{using Half=uint16_t;enum class DeviceType{PrivateUse1,CPU};}
namespace at{constexpr int kHalf=1,kChar=2;
struct Tensor{uint64_t address;bool present=true,contiguous=true;int dtype=W8?kChar:kHalf,rank=2;int64_t rows=6144,cols=2048;c10::DeviceType location=c10::DeviceType::PrivateUse1;
 bool defined()const{return present;}int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}int dim()const{return rank;}int64_t size(int i)const{return i?cols:rows;}int64_t numel()const{return rank==1?rows:rows*cols;}
 struct Device{c10::DeviceType value;c10::DeviceType type()const{return value;}};Device device()const{return {location};}template<class T>T*data_ptr()const{return reinterpret_cast<T*>(address);}
};}
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8303616;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int core,int off)const{assert(core==0);return 0x20000000+off;}}allocator;
#define SPM_ALLOC allocator
enum class KernelId{G05_TEXT_GATE_UP_FP16_ACC32_M256N768K2048,G05_TEXT_GATE_UP_W8A16_ACC32_M256N768K2048,G05_TEXT_GATE_UP_W8A16_ACC16_M256N768K2048};constexpr auto ID=W8?(TEST_ACC32?KernelId::G05_TEXT_GATE_UP_W8A16_ACC32_M256N768K2048:KernelId::G05_TEXT_GATE_UP_W8A16_ACC16_M256N768K2048):KernelId::G05_TEXT_GATE_UP_FP16_ACC32_M256N768K2048;
struct KernelCache{bool available=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId)const{return available;}};
enum class GraphDdrRegisterRole{ModelWeight,Int8ModelWeight,PerChannelScale};enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor owner;};
struct Kernel{std::map<int,uint16_t>regs;void reset_regs(){regs.clear();}void set_regs(int i,uint16_t v){regs[i]=v;}void set_regs(int,uint32_t)=delete;};
struct Submission{std::map<int,uint16_t>regs;std::vector<GraphDdrRegisterOperandSpec>operands;};struct Writer{void write(Kernel&);};
struct RpuKernelGraph{bool staged=false,pending=false,written=false;int stages=0,lookups=0;Kernel kernel;std::vector<GraphDdrRegisterOperandSpec>operands;std::vector<Submission>submissions;
 static RpuKernelGraph&active(){static RpuKernelGraph graph;return graph;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec>specs){assert(id==ID&&!staged&&!pending&&specs.size()==(W8?4:2));staged=true;written=false;operands=std::move(specs);++stages;return {};}
 Kernel*get(KernelId id){assert(id==ID&&staged&&!pending&&KernelCache::instance().available);pending=true;++lookups;return &kernel;}
};
void Writer::write(Kernel&kernel){auto&g=RpuKernelGraph::active();assert(g.staged&&g.pending&&!g.written);for(auto&operand:g.operands){uint64_t v=operand.owner.address>>8;kernel.set_regs(operand.abi.lo,static_cast<uint16_t>(v));kernel.set_regs(operand.abi.hi,static_cast<uint16_t>(v>>16));}g.written=true;}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
std::vector<uint64_t>flushed;template<class T>void rpu_ddr_flush(T*p){flushed.push_back(reinterpret_cast<uint64_t>(p));}
struct Queue{bool broadcast=false;void set_broadcast_mode(bool v){broadcast=v;}void enqueu_kernel(Kernel&k,std::vector<uint16_t>grid,std::vector<uint8_t>cores){auto&g=RpuKernelGraph::active();assert(broadcast&&g.staged&&g.pending&&g.written);assert(grid==std::vector<uint16_t>({W8?20:24,1,1})&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));g.submissions.push_back({k.regs,g.operands});g.operands.clear();g.staged=g.pending=g.written=false;}}queue;
Queue*get_queue(int cores){assert(cores==8);return &queue;}
#define GET_QUEUE(cores) get_queue(cores)
at::Tensor gs{0x3456789000ULL,true,true,at::kHalf,1,6144,1},us{0x456789a000ULL,true,true,at::kHalf,1,6144,1};
void launch_native(uint32_t input,const at::Tensor&gate_weight,const at::Tensor&up_weight,uint32_t gate_output,uint32_t up_output __SCALE_ARGS__)__BODY__
void launch(uint32_t input,const at::Tensor&gate_weight,const at::Tensor&up_weight,uint32_t gate_output,uint32_t up_output){launch_native(input,gate_weight,up_weight,gate_output,up_output __SCALE_CALL__);}
uint32_t pair(const Submission&s,int reg){return uint32_t(s.regs.at(reg))|(uint32_t(s.regs.at(reg+1))<<16);}
template<class F>void reject(F f){auto&g=RpuKernelGraph::active();auto before=g.submissions.size(),flushes=flushed.size();int stages=g.stages,lookups=g.lookups;try{f();assert(false);}catch(const std::runtime_error&){}assert(g.submissions.size()==before&&flushed.size()==flushes&&g.stages==stages&&g.lookups==lookups&&!g.staged&&!g.pending);}
int main(){
 auto&g=RpuKernelGraph::active();const uint32_t base=allocator.addr(0,0);
 for(uint32_t pass=0;pass<2;++pass){const uint32_t in=base+0x20000+pass*0x200000,q=in+1048576,gate=q+393216;at::Tensor qw{0x1234567800ULL+pass*0x400000},gw{0x2345678900ULL+pass*0x800000};gs.address+=0x40000;us.address+=0x80000;launch(in,qw,gw,q,gate);const auto&s=g.submissions.back();
  assert(pair(s,0)==in&&pair(s,4)==q&&pair(s,36)==gate);assert(pair(s,2)==qw.address>>8&&pair(s,32)==gw.address>>8);
  assert(pair(s,6)==0&&s.regs.at(8)==1&&s.regs.at(9)==256&&pair(s,10)==768&&s.regs.at(12)==2048&&s.regs.at(13)==128&&s.regs.at(14)==0&&s.regs.at(15)==8);
  assert(pair(s,16)==4096&&pair(s,18)==1536&&(W8||pair(s,20)==0)&&pair(s,22)==0&&pair(s,24)==0&&s.regs.at(26)==2&&s.regs.at(27)==(W8?1:2));
  assert(s.operands.size()==(W8?4:2)&&s.operands[0].owner.address==qw.address&&s.operands[1].owner.address==gw.address);
  for(int i=0;i<2;++i){auto&abi=s.operands[i].abi;assert(abi.role==(W8?GraphDdrRegisterRole::Int8ModelWeight:GraphDdrRegisterRole::ModelWeight)&&abi.access==GraphDdrRegisterAccess::Read&&abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi&&abi.lo==(i?32:2)&&abi.hi==(i?33:3));}
  const int fi=pass*(W8?4:2);assert(flushed[fi]==qw.address&&flushed[fi+1]==gw.address);
  if(W8){assert(pair(s,20)==gs.address>>8&&pair(s,34)==us.address>>8);assert(s.operands[2].owner.address==gs.address&&s.operands[3].owner.address==us.address);assert(flushed[fi+2]==gs.address&&flushed[fi+3]==us.address);for(int i=2;i<4;++i){const auto&abi=s.operands[i].abi;assert(abi.role==GraphDdrRegisterRole::PerChannelScale&&abi.access==GraphDdrRegisterAccess::Read&&abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi&&abi.lo==(i==2?20:34)&&abi.hi==(i==2?21:35));}}
 }
 const at::Tensor qw{0x1234567800ULL},gw{0x2345678900ULL};const uint32_t in=base+0x20000,q=in+1048576,gate=q+393216;
 launch(base,qw,gw,base+1048576,base+1441792);
 const uint32_t end=base+SpmAllocator::SPM_USABLE;launch(end-1835008,qw,gw,end-786432,end-393216);
 assert(pair(g.submissions.back(),36)==end-393216);
 for(auto regions:{std::array<uint32_t,3>{in,in,gate},{in,q,q},{in,q,in+256},{in,q-256,gate},{in,q,gate-256},{in+1,q,gate},{base-256,q,gate},{in,q,base+SpmAllocator::SPM_USABLE-8192}})reject([&]{launch(regions[0],qw,gw,regions[1],regions[2]);});
 allocator.initialized=false;reject([&]{launch(in,qw,gw,q,gate);});allocator.initialized=true;KernelCache::instance().available=false;reject([&]{launch(in,qw,gw,q,gate);});KernelCache::instance().available=true;
 for(int which:{0,1})for(int fault=0;fault<7;++fault){auto first=qw,second=gw;auto&w=which?second:first;if(fault==0){w.present=false;}if(fault==1){w.dtype=W8?at::kHalf:at::kChar;}if(fault==2){w.contiguous=false;}if(fault==3){w.location=c10::DeviceType::CPU;}if(fault==4){w.rank=1;}if(fault==5){w.rows=1024;}if(fault==6){w.cols=512;}reject([&]{launch(in,first,second,q,gate);});}
#if __W8__
 for(int which:{0,1})for(int fault=0;fault<6;++fault){auto first=gs,second=us;auto&s=which?second:first;if(fault==0)s.present=false;if(fault==1)s.dtype=at::kChar;if(fault==2)s.contiguous=false;if(fault==3)s.location=c10::DeviceType::CPU;if(fault==4)s.rank=2;if(fault==5)s.rows=6143;reject([&]{launch_native(in,qw,gw,q,gate,first,second);});}
#endif
}
'''
    cpp = cpp.replace('__W8__', '1' if w8 else '0').replace('__MODE__', str(linear_acc32).lower())
    cpp = cpp.replace('__SCALE_ARGS__', ',const at::Tensor&gate_scale,const at::Tensor&up_scale,bool linear_acc32=TEST_ACC32' if w8 else '')
    cpp = cpp.replace('__SCALE_CALL__', ',gs,us' if w8 else '')
    precision = 'w8a16' if w8 else 'fp16'
    cpp = cpp.replace('__BODY__', _body(source, f'void rpu_launch_g05_text_gate_up_linear_{precision}_acc32_kernel('))
    _compile_run(tmp_path, 'launcher', cpp)
