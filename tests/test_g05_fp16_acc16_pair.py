"""Execute ACC16 pair launchers and reject invalid owners and spans."""
from pathlib import Path

import pytest

from test_g05_q_gate_native import _compile_run
from test_siglip_cold_profile import _body

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("text", [False, True], ids=["action", "text"])
def test_fp16_acc16_pair_actual_launcher_owners_and_rejection(tmp_path, text):
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    symbol = "rpu_launch_g05_" + ("text_" if text else "") + "gate_up_linear_fp16_acc16_kernel"
    cpp = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>
#include "rpu_linear_tiling.h"
constexpr int M=__M__,N=__N__,K=__K__,GRID=__GRID__;
constexpr uint32_t INPUT_BYTES=M*K*2,OUTPUT_BYTES=M*N*2;
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10{using Half=uint16_t;enum class DeviceType{PrivateUse1,CPU};}
namespace at{constexpr int kHalf=1,kChar=2;
struct Tensor{
 uint64_t address;bool present=true,contiguous=true;int dtype=kHalf,rank=2;int64_t rows=N*8,cols=K;c10::DeviceType location=c10::DeviceType::PrivateUse1;
 bool defined()const{return present;}int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}int dim()const{return rank;}int64_t size(int i)const{return i?cols:rows;}uint64_t nbytes()const{return rows*cols*2;}
 struct Device{c10::DeviceType value;c10::DeviceType type()const{return value;}};Device device()const{return {location};}
 template<class T>T*data_ptr()const{return reinterpret_cast<T*>(address);}
};}
template<class T>uint64_t RpuGetDevAddr(T*pointer){return reinterpret_cast<uint64_t>(pointer);}
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8303616;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int core,int off)const{assert(core==0);return 0x20000000+off;}}allocator;
#define SPM_ALLOC allocator
enum class KernelId{__ID__};constexpr auto ID=KernelId::__ID__;
struct KernelCache{bool available=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId id)const{assert(id==ID);return available;}};
enum class GraphDdrRegisterRole{ModelWeight};enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor owner;};
struct Kernel{std::map<int,uint16_t>regs;void reset_regs(){regs.clear();}void set_regs(int i,uint16_t v){regs[i]=v;}void set_regs(int,uint32_t)=delete;};
struct Submission{std::map<int,uint16_t>regs;std::vector<GraphDdrRegisterOperandSpec>operands;};struct Writer{void write(Kernel&);};
struct RpuKernelGraph{
 bool staged=false,pending=false,written=false;int stages=0,lookups=0;Kernel kernel;std::vector<GraphDdrRegisterOperandSpec>operands;std::vector<Submission>submissions;
 static RpuKernelGraph&active(){static RpuKernelGraph graph;return graph;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec>specs){assert(id==ID&&!staged&&!pending&&specs.size()==2);staged=true;written=false;operands=std::move(specs);++stages;return {};}
 Kernel*get(KernelId id){assert(id==ID&&staged&&!pending&&KernelCache::instance().available);pending=true;++lookups;return &kernel;}
};
void Writer::write(Kernel&kernel){auto&g=RpuKernelGraph::active();assert(g.staged&&g.pending&&!g.written);for(auto&operand:g.operands){uint64_t v=operand.owner.address>>8;kernel.set_regs(operand.abi.lo,static_cast<uint16_t>(v));kernel.set_regs(operand.abi.hi,static_cast<uint16_t>(v>>16));}g.written=true;}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
std::vector<uint64_t>flushed;template<class T>void rpu_ddr_flush(T*p){flushed.push_back(reinterpret_cast<uint64_t>(p));}
struct Queue{bool broadcast=false;void set_broadcast_mode(bool v){broadcast=v;}void enqueu_kernel(Kernel&k,std::vector<uint16_t>grid,std::vector<uint8_t>cores){auto&g=RpuKernelGraph::active();assert(broadcast&&g.staged&&g.pending&&g.written);assert(grid==std::vector<uint16_t>({GRID,1,1})&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));g.submissions.push_back({k.regs,g.operands});g.operands.clear();g.staged=g.pending=g.written=false;}}queue;
Queue*get_queue(int cores){assert(cores==8);return &queue;}
#define GET_QUEUE(cores) get_queue(cores)
void launch(uint32_t input,const at::Tensor&gate_weight,const at::Tensor&up_weight,uint32_t gate_output,uint32_t up_output)__BODY__
uint32_t pair(const Submission&s,int reg){return uint32_t(s.regs.at(reg))|(uint32_t(s.regs.at(reg+1))<<16);}
template<class F>void reject(F f){auto&g=RpuKernelGraph::active();auto before=g.submissions.size(),flushes=flushed.size();int stages=g.stages,lookups=g.lookups;try{f();assert(false);}catch(const std::runtime_error&){}assert(g.submissions.size()==before&&flushed.size()==flushes&&g.stages==stages&&g.lookups==lookups&&!g.staged&&!g.pending);}
int main(){
 auto&g=RpuKernelGraph::active();const uint32_t base=allocator.addr(0,0);
 for(uint32_t pass=0;pass<2;++pass){const uint32_t in=base+0x20000+pass*0x200000,gate=in+INPUT_BYTES,up=gate+OUTPUT_BYTES;at::Tensor gw{0x1234567800ULL+pass*0x400000},uw{0x2345678900ULL+pass*0x800000};launch(in,gw,uw,gate,up);const auto&s=g.submissions.back();
  assert(pair(s,0)==in&&pair(s,4)==gate&&pair(s,36)==up);assert(pair(s,2)==gw.address>>8&&pair(s,32)==uw.address>>8);
  assert(pair(s,6)==0&&s.regs.at(8)==1&&s.regs.at(9)==M&&pair(s,10)==N&&s.regs.at(12)==K&&s.regs.at(13)==K/16&&s.regs.at(14)==0&&s.regs.at(15)==8);
  assert(pair(s,16)==K*2&&pair(s,18)==N*2&&pair(s,20)==0&&pair(s,22)==0&&pair(s,24)==0&&s.regs.at(26)==2&&s.regs.at(27)==2&&s.regs.at(28)==0&&s.regs.at(29)==0);
  assert(s.operands.size()==2&&s.operands[0].owner.address==gw.address&&s.operands[1].owner.address==uw.address);
  for(int i=0;i<2;++i){auto&abi=s.operands[i].abi;assert(abi.role==GraphDdrRegisterRole::ModelWeight&&abi.access==GraphDdrRegisterAccess::Read&&abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi&&abi.lo==(i?32:2)&&abi.hi==(i?33:3));}
  assert(flushed[pass*2]==gw.address&&flushed[pass*2+1]==uw.address);
 }
 const at::Tensor gw{0x1234567800ULL},uw{0x2345678900ULL};const uint32_t in=base+0x20000,gate=in+INPUT_BYTES,up=gate+OUTPUT_BYTES;
 launch(base,gw,uw,base+INPUT_BYTES,base+INPUT_BYTES+OUTPUT_BYTES);
 const uint32_t end=base+SpmAllocator::SPM_USABLE;launch(end-INPUT_BYTES-2*OUTPUT_BYTES,gw,uw,end-2*OUTPUT_BYTES,end-OUTPUT_BYTES);
 assert(pair(g.submissions.back(),36)==end-OUTPUT_BYTES);
 // Exact final encodable DDR byte is legal for either weight.
 for(int which:{0,1}){auto first=gw,second=uw;auto&w=which?second:first;w.address=(UINT64_C(1)<<40)-w.nbytes();launch(in,first,second,gate,up);}
 for(auto regions:{std::array<uint32_t,3>{in,in,up},{in,gate,gate},{in,gate,in+256},{in,gate-256,up},{in,gate,up-256},{in+1,gate,up},{in,gate+1,up},{in,gate,up+1},{base-256,gate,up},{in,gate,end-OUTPUT_BYTES+256},{end-INPUT_BYTES+256,gate,up}})reject([&]{launch(regions[0],gw,uw,regions[1],regions[2]);});
 allocator.initialized=false;reject([&]{launch(in,gw,uw,gate,up);});allocator.initialized=true;KernelCache::instance().available=false;reject([&]{launch(in,gw,uw,gate,up);});KernelCache::instance().available=true;
 for(int which:{0,1})for(int fault=0;fault<11;++fault){auto first=gw,second=uw;auto&w=which?second:first;
  if(fault==0)w.present=false;
  if(fault==1)w.dtype=at::kChar;
  if(fault==2)w.contiguous=false;
  if(fault==3)w.location=c10::DeviceType::CPU;
  if(fault==4)w.rank=1;
  if(fault==5)--w.rows;
  if(fault==6)--w.cols;
  if(fault==7)w.address=0;
  if(fault==8)++w.address;
  if(fault==9)w.address=UINT64_C(1)<<40;
  if(fault==10)w.address=(UINT64_C(1)<<40)-w.nbytes()+256;
  reject([&]{launch(in,first,second,gate,up);});
 }
}
'''
    values = {"M": 256 if text else 32, "N": 768 if text else 512,
              "K": 2048 if text else 1024, "GRID": 20 if text else 12,
              "ID": "G05_TEXT_GATE_UP_FP16_ACC16_M256N768K2048" if text
              else "G05_GATE_UP_FP16_ACC16_M32N512K1024",
              "BODY": _body(source, "void " + symbol + "(")}
    for key, value in values.items():
        cpp = cpp.replace("__" + key + "__", str(value))
    _compile_run(tmp_path, "fp16_acc16_pair", cpp)
