"""Run the actual combined W8 wrapper against typed-owner and pending-kernel stubs."""
from pathlib import Path
import pytest
from test_g05_q_gate_native import _compile_run
from test_siglip_cold_profile import _body

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("linear_acc32", [True, False], ids=["acc32", "acc16"])
def test_actual_qkv_gate_w8_launcher_validates_all_owners_and_pending_kernel_abi(tmp_path, linear_acc32):
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    marker = "void rpu_launch_g05_qkv_gate_owner_linear_w8a16_acc32_kernel("
    function = ""
    for marker in ("static void launch_g05_qkv_gate_owner_linear_w8a16_acc32(", marker,
                   "void rpu_launch_g05_qkv_gate_owner_only_linear_w8a16_acc32_kernel("):
        start = source.index(marker)
        function += source[start:source.index("{", start)] + _body(source, marker) + "\n"
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
 struct Device{c10::DeviceType value;c10::DeviceType type()const{return value;}};Device device()const{return {location};}void*data_ptr()const{return reinterpret_cast<void*>(address);}template<class T>T*data_ptr()const{return reinterpret_cast<T*>(address);}
};}
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8387584;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int core,int off)const{assert(core==0);return 0x20000000+off;}}allocator;
#define SPM_ALLOC allocator
uint64_t RpuGetDevAddr(void* pointer){return reinterpret_cast<uint64_t>(pointer);}
enum class KernelId{G05_QKV_GATE_OWNER_W8A16_ACC16_M32N256K1024,G05_QKV_GATE_OWNER_W8A16_ACC32_M32N256K1024,G05_KV_OWNER_FANOUT_M32D256};
constexpr auto COMBINED=TEST_ACC32?KernelId::G05_QKV_GATE_OWNER_W8A16_ACC32_M32N256K1024:KernelId::G05_QKV_GATE_OWNER_W8A16_ACC16_M32N256K1024;
constexpr auto FAN=KernelId::G05_KV_OWNER_FANOUT_M32D256;
struct KernelCache{std::set<KernelId>missing;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId id)const{return !missing.count(id);}};
enum class GraphDdrRegisterRole{Int8ModelWeight,PerChannelScale};enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor owner;};
struct Kernel{std::map<int,uint16_t>regs;void reset_regs(){regs.clear();}void set_regs(int i,uint16_t v){regs[i]=v;}void set_regs(int,uint32_t)=delete;};
struct Submission{KernelId id;std::map<int,uint16_t>regs;std::vector<GraphDdrRegisterOperandSpec>operands;std::vector<uint16_t>grid;};
struct Writer{void write(Kernel&);};
struct RpuKernelGraph{
 std::optional<KernelId>staged,pending;bool written=false;int stages=0,lookups=0;Kernel kernel;
 std::vector<GraphDdrRegisterOperandSpec>operands;std::vector<Submission>submissions;
 static RpuKernelGraph&active(){static RpuKernelGraph graph;return graph;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec>specs){
  assert(!staged&&!pending&&id==COMBINED&&specs.size()==8);staged=id;written=false;operands=std::move(specs);++stages;return {};}
 Kernel*get(KernelId id){assert(!pending&&KernelCache::instance().has_loaded(id));
  assert((id==FAN&&!staged)||(staged&&*staged==id));pending=id;written=id==FAN;++lookups;return &kernel;}
};
void Writer::write(Kernel&kernel){auto&g=RpuKernelGraph::active();assert(g.staged&&g.pending&&*g.staged==*g.pending&&!g.written);
 for(auto&operand:g.operands){uint64_t value=operand.owner.address>>8;kernel.set_regs(operand.abi.lo,static_cast<uint16_t>(value));kernel.set_regs(operand.abi.hi,static_cast<uint16_t>(value>>16));}g.written=true;}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
std::vector<uint64_t>flushed;template<class T>void rpu_ddr_flush(T*p){flushed.push_back(reinterpret_cast<uint64_t>(p));}
struct Queue{bool broadcast=false;void set_broadcast_mode(bool v){broadcast=v;}
 void enqueu_kernel(Kernel&k,std::vector<uint16_t>grid,std::vector<uint8_t>cores){
  auto&g=RpuKernelGraph::active();assert(broadcast&&g.pending&&g.written);
  assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));g.submissions.push_back({*g.pending,k.regs,g.operands,grid});
  g.operands.clear();g.staged.reset();g.pending.reset();g.written=false;}}queue;
Queue*get_queue(int cores){assert(cores==8);return &queue;}
#define GET_QUEUE(cores) get_queue(cores)
__FUNCTION__
uint32_t pair(const Submission&s,int reg){return uint32_t(s.regs.at(reg))|(uint32_t(s.regs.at(reg+1))<<16);}
template<class F>void reject(F f){auto&g=RpuKernelGraph::active();auto before=g.submissions.size(),flushes=flushed.size();int stages=g.stages,lookups=g.lookups;
 try{f();assert(false);}catch(const std::runtime_error&){}
 assert(g.submissions.size()==before&&flushed.size()==flushes&&g.stages==stages&&g.lookups==lookups&&!g.staged&&!g.pending);}
void launch(const std::array<at::Tensor,8>& t,const std::array<uint32_t,5>& p){
 rpu_launch_g05_qkv_gate_owner_linear_w8a16_acc32_kernel(p[0],t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7],p[1],p[2],p[3],p[4]);}
int main(){
 auto&g=RpuKernelGraph::active();const uint32_t base=allocator.addr(0,0);
 std::array<uint32_t,5>spm{base+0x10000,base+0x20000,base+0x24000,base+0x28000,base+0x2c000};
 std::array<at::Tensor,8>tensors{};
 for(size_t i=0;i<tensors.size();++i){tensors[i].address=0x1234567800ULL+i*0x200000;
  if(i>=4){tensors[i].dtype=at::kHalf;tensors[i].rank=1;tensors[i].rows=2048;}}
 const std::array<int,8>registers{2,32,40,48,20,34,42,50};
 // A second call changes every address, preventing cached first-call bindings.
 for(int pass=0;pass<2;++pass){for(size_t i=0;i<8;++i)tensors[i].address+=(i+1)*256;
  auto begin=g.submissions.size();auto flush_start=flushed.size();launch(tensors,spm);
  assert(g.submissions.size()==begin+2&&flushed.size()==flush_start+8&&!g.pending&&!g.staged);
  const auto&s=g.submissions[begin];assert(s.id==COMBINED&&s.grid==std::vector<uint16_t>({18,1,1})&&s.operands.size()==8);
  for(size_t i=0;i<8;++i){assert(pair(s,registers[i])==tensors[i].address>>8);
   assert(s.operands[i].owner.address==tensors[i].address&&s.operands[i].abi.lo==registers[i]&&s.operands[i].abi.hi==registers[i]+1);
   assert(s.operands[i].abi.role==(i<4?GraphDdrRegisterRole::Int8ModelWeight:GraphDdrRegisterRole::PerChannelScale));
   assert(s.operands[i].abi.access==GraphDdrRegisterAccess::Read&&s.operands[i].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
   assert(flushed[flush_start+i]==tensors[i].address);}
  assert(pair(s,0)==spm[0]&&pair(s,4)==spm[1]&&pair(s,36)==spm[2]&&pair(s,44)==spm[3]&&pair(s,52)==spm[4]);
  assert(pair(s,6)==0&&s.regs.at(8)==1&&s.regs.at(9)==32&&pair(s,10)==256&&s.regs.at(12)==1024&&s.regs.at(13)==64&&s.regs.at(14)==0&&s.regs.at(15)==8);
  assert(pair(s,16)==2048&&pair(s,18)==512&&pair(s,22)==0&&pair(s,24)==0&&s.regs.at(26)==2&&s.regs.at(27)==1);
  const auto&fan=g.submissions[begin+1];assert(fan.id==FAN&&fan.grid==std::vector<uint16_t>({1,1,1})&&fan.operands.empty());
  assert(pair(fan,0)==spm[3]-base&&pair(fan,2)==spm[4]-base);
 }
 // Owner-only entry uses the identical typed producer and never restores fanout.
 auto begin=g.submissions.size();KernelCache::instance().missing.insert(FAN);
 rpu_launch_g05_qkv_gate_owner_only_linear_w8a16_acc32_kernel(spm[0],tensors[0],tensors[1],tensors[2],tensors[3],tensors[4],tensors[5],tensors[6],tensors[7],spm[1],spm[2],spm[3],spm[4]);
 assert(g.submissions.size()==begin+1&&g.submissions.back().id==COMBINED&&g.submissions.back().operands.size()==8);
 KernelCache::instance().missing.clear();
 for(auto missing:{COMBINED,FAN}){KernelCache::instance().missing.insert(missing);reject([&]{launch(tensors,spm);});KernelCache::instance().missing.clear();}
 allocator.initialized=false;reject([&]{launch(tensors,spm);});allocator.initialized=true;
 // Every owner is independently validated before any stage/flush/GET/enqueue.
 for(size_t index=0;index<8;++index)for(int fault=0;fault<11;++fault){auto bad=tensors;auto&t=bad[index];
  if(fault==0)t.present=false;
  if(fault==1)t.dtype=index<4?at::kHalf:at::kChar;
  if(fault==2)t.contiguous=false;
  if(fault==3)t.location=c10::DeviceType::CPU;
  if(fault==4)t.rank=index<4?1:2;
  if(fault==5)t.rows=2047;
  if(fault==6){if(index<4)t.cols=1023;else t.rows=2049;}
  if(fault==7)t.address=0;
  if(fault==8)t.address+=1;
  if(fault==9)t.address=UINT64_C(1)<<40;
  if(fault==10)t.address=(UINT64_C(1)<<40)-256;
  reject([&]{launch(bad,spm);});}
 // All ten pairs, not only input/output, must be disjoint.
 for(size_t i=0;i<5;++i)for(size_t j=0;j<i;++j){auto bad=spm;bad[i]=bad[j];reject([&]{launch(tensors,bad);});}
 for(size_t i=0;i<5;++i)for(int fault=0;fault<3;++fault){auto bad=spm;
  if(fault==0)bad[i]+=1;
  if(fault==1)bad[i]=base-256;
  if(fault==2)bad[i]=base+SpmAllocator::SPM_USABLE-256;
  reject([&]{launch(tensors,bad);});}
 // Exact endpoint and adjacent intervals are legal; no blanket over-rejection.
 auto edge=tensors;edge[7].address=(UINT64_C(1)<<40)-edge[7].nbytes();
 auto edge_spm=spm;edge_spm[4]=base+SpmAllocator::SPM_USABLE-(32*256*2);launch(edge,edge_spm);
}
'''
    _compile_run(tmp_path, "qkv_gate_w8_launcher", cpp.replace("__MODE__", str(linear_acc32).lower()).replace("__FUNCTION__", function.replace("bool linear_acc32)", "bool linear_acc32 = TEST_ACC32)")))
