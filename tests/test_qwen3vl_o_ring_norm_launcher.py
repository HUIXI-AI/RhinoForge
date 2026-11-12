"""Execute the production O/ring/Norm launcher with SDK-shaped recorders."""
from pathlib import Path

from test_qwen3vl_decode_gateup_launcher import _run
from test_qwen3vl_prefill_gateup_launcher import definition


ROOT = Path(__file__).resolve().parents[1]
NAME = "rpu_launch_qwen3vl_4b_decode_o_ring_norm_spm_kernel"


def test_actual_o_ring_norm_abi_owners_and_rejected_submissions(tmp_path):
    source = (ROOT / "src/ops/rpu_linear_gemv.cpp").read_text()
    body = definition(source, f"void {NAME}(")
    header = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    start = header.index(f"void {NAME}(")
    declaration = header[start:header.index(";", start) + 1]
    harness = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>
template<class... T> void check(bool ok,T&&...){if(!ok)throw std::runtime_error("rejected");}
#define TORCH_CHECK(...) check(__VA_ARGS__)
namespace c10 {
enum class DeviceType{PrivateUse1,CPU}; enum class Layout{Strided,Sparse};
struct Half {uint16_t x; explicit Half(float f):x(f==1.0f/2560?0xe66:f==1e-6f?0x11:0) {}};
}
namespace at {
constexpr int kChar=1,kHalf=2,kFloat=3;
struct Tensor {
 std::vector<int64_t> shape{2560,4096}; int type=kChar;
 bool present=true,contiguous=true;
 c10::DeviceType kind=c10::DeviceType::PrivateUse1;
 c10::Layout layout_kind=c10::Layout::Strided;
 uint64_t address=0x1200000000ULL; int64_t offset=0;
 struct Storage {uint64_t bytes=2560*4096; bool present=true;
   uint64_t nbytes()const{return bytes;} explicit operator bool()const{return present;}} backing;
 struct Device {c10::DeviceType kind; c10::DeviceType type()const{return kind;}};
 bool defined()const{return present;} Device device()const{return {kind};}
 c10::Layout layout()const{return layout_kind;} int scalar_type()const{return type;}
 bool is_contiguous()const{return contiguous;} int64_t dim()const{return shape.size();}
 int64_t size(int i)const{return shape.at(i);} int64_t storage_offset()const{return offset;}
 const Storage& storage()const{return backing;}
 void* data_ptr()const{return reinterpret_cast<void*>(address);}
};
}
struct SpmAllocator {
 static constexpr uint64_t SPM_USABLE=8387584;
 bool initialized=true,drift=false; uint32_t base=0x40000000;
 bool is_initialized()const{return initialized;}
 uint32_t addr(int core,uint32_t offset)const{return base+uint32_t(core)*0x1000000+offset+(drift&&core==3?256:0);}
} SPM_ALLOC;
enum class KernelId{QWEN3VL_O_RING_NORM_M1_H2560_W8};
enum class GraphDdrRegisterRole{Int8ModelWeight,PerChannelScale};
enum class GraphDdrRegisterAccess{Read};
enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct Abi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{Abi abi;at::Tensor tensor;};
int stages=0,submits=0,flushes=0,writes=0,addresses=0;
struct Kernel_t {
 std::array<uint16_t,68> r{};std::array<uint16_t,64> scm{};
 std::array<uint32_t,64> atomic{};int wide=0,resets=0,scm_writes=0;
 void reset_regs(){r.fill(0);scm.fill(0);atomic.fill(0);scm_writes=0;++resets;}
 void set_regs(int i,uint16_t value){r.at(i)=value;}
 void set_regs(int i,int16_t value){set_regs(i,uint16_t(value));}
 void set_regs(int i,uint32_t value){++wide;check(i%2==0);r.at(i)=uint16_t(value);r.at(i+1)=uint16_t(value>>16);}
 void set_regs(int i,int32_t value){set_regs(i,uint32_t(value));}
 uint32_t pair(int i)const{return r.at(i)|uint32_t(r.at(i+1))<<16;}
 uint32_t scm_pair(int i)const{return scm.at(i)|uint32_t(scm.at(i+8))<<16;}
} kernel;
bool missing=false,typed_pending=false;int kernel_gets=0;
Kernel_t* get_kernel(KernelId){assert(typed_pending);typed_pending=false;++kernel_gets;return missing?nullptr:&kernel;}
#define GET_KERNEL(id) get_kernel(id)
void rpu_set_legacy_scm_u16_checked(Kernel_t* k,int i,uint16_t value,const char*){k->scm.at(i-4096)=value;++k->scm_writes;}
uint64_t RpuGetDevAddr(void* p){++addresses;return reinterpret_cast<uint64_t>(p);}
struct Writer {
 std::vector<GraphDdrRegisterOperandSpec> ops;
 void write(Kernel_t& k){++writes;for(const auto& op:ops){auto a=op.tensor.address>>8;k.set_regs(op.abi.lo,uint16_t(a));k.set_regs(op.abi.hi,uint16_t(a>>16));}}
};
struct RpuKernelGraph {
 bool deny=false;std::vector<uint64_t> owners;
 static auto& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> ops){
  ++stages;check(!deny);assert(!typed_pending);assert(id==KernelId::QWEN3VL_O_RING_NORM_M1_H2560_W8&&ops.size()==2);
  for(int i=0;i<2;++i){const auto& a=ops[i].abi;
   assert(a.lo==6+i*2&&a.hi==7+i*2&&a.access==GraphDdrRegisterAccess::Read);
   assert(a.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
   assert(a.role==(i?GraphDdrRegisterRole::PerChannelScale:GraphDdrRegisterRole::Int8ModelWeight));
   check(ops[i].tensor.address&&ops[i].tensor.address%256==0&&ops[i].tensor.address<(1ULL<<40));
  }typed_pending=true;return {ops};
 }
 void keep_alive(const at::Tensor& t){owners.push_back(t.address);}
};
std::vector<uint64_t> flushed;
void rpu_ddr_flush(void* p){++flushes;flushed.push_back(reinterpret_cast<uint64_t>(p));}
struct Queue {
 bool broadcast=false;void set_broadcast_mode(bool v){broadcast=v;}
 void enqueu_kernel(Kernel_t& k,std::vector<int> grid,std::vector<int> cores){
  assert(broadcast&&writes>submits&&k.wide==0&&k.scm_writes==64);
  assert(grid==std::vector<int>({8,1,1})&&cores==std::vector<int>({0,1,2,3,4,5,6,7}));
  for(auto x:k.atomic)assert(x==0);
  k.atomic.fill(17);++submits;
 }
} queue;
Queue* get_queue(int n){assert(n==8);queue.broadcast=false;return &queue;}
#define GET_QUEUE(n) get_queue(n)
__DECL__
__BODY__
using Inputs=std::array<at::Tensor,2>;
using Spm=std::array<uint32_t,5>;
const Spm initial{0x40001000,0x40002000,0x40004000,0x40006000,0x40008000};
Inputs good(int64_t k=4096){Inputs in;in[0].shape[1]=k;in[0].backing.bytes=2560*k;in[1].type=at::kHalf;in[1].shape={2560};in[1].backing.bytes=(2560+256)*2;in[1].address+=0x2000000;return in;}
void launch(const Inputs& in,Spm p=initial,double eps=1e-6,int64_t k=4096){__NAME__(p[0],in[0],in[1],p[1],p[2],p[3],p[4],eps,k);}
void reject(std::function<void()> f,bool before_stage=true){
 const int old=submits,st=stages,fl=flushes,rs=kernel.resets;const auto keeps=RpuKernelGraph::active().owners.size();
 bool rejected=false;try{f();}catch(const std::runtime_error&){rejected=true;}
 assert(rejected&&!typed_pending&&submits==old&&flushes==fl&&kernel.resets==rs&&RpuKernelGraph::active().owners.size()==keeps);
 if(before_stage)assert(stages==st);
}
int main(){
 auto& graph=RpuKernelGraph::active();std::array<uint16_t,68> first{};
 for(int step:{0,1,0}){
  auto in=good();Spm spm=initial;for(auto& t:in)t.address+=step*256;for(auto& p:spm)p+=step*0x10000;
  const int fl=flushes,rs=kernel.resets;const auto keeps=graph.owners.size();launch(in,spm);
  assert(flushes==fl+2&&kernel.resets==rs+1&&graph.owners.size()==keeps+2);
  assert(graph.owners[keeps]==in[0].address&&graph.owners[keeps+1]==in[1].address);
  assert(flushed[fl]==in[0].address&&flushed[fl+1]==in[1].address);
  assert(kernel.r[0]==512&&kernel.r[1]==2560&&kernel.r[2]==0);
  assert(kernel.pair(4)==spm[0]>>8&&kernel.pair(6)==in[0].address>>8&&kernel.pair(8)==in[1].address>>8);
  assert(kernel.pair(10)==spm[1]>>8&&kernel.pair(12)==0x10000&&kernel.pair(14)==0x10000&&kernel.pair(16)==5120);
  assert(kernel.r[18]==320&&kernel.r[19]==320&&kernel.r[21]==8&&kernel.r[22]==1&&kernel.pair(24)==63488);
  assert(kernel.pair(26)==spm[2]-SPM_ALLOC.base&&kernel.r[28]==0&&kernel.r[29]==0xe66&&kernel.r[30]==5120);
  assert(kernel.pair(32)==spm[3]&&kernel.pair(34)==spm[2]&&kernel.pair(36)==spm[4]&&kernel.r[38]==0x11);
  assert(kernel.r[64]==8&&kernel.r[65]==1&&kernel.r[66]==1);
  for(int c=0;c<8;++c){assert(kernel.scm_pair(c)==spm[1]-SPM_ALLOC.base+c*0x1000000);
   assert(kernel.scm_pair(16+c)==spm[3]-SPM_ALLOC.base+c*0x1000000);
   assert(kernel.scm_pair(32+c)==uint32_t(c*640)&&kernel.scm_pair(48+c)==320);}
 if(submits==1)first=kernel.r;else assert((first==kernel.r)==(step==0));
 }
 {auto in=good(9728);launch(in,initial,1e-6,9728);assert(kernel.r[0]==1216&&kernel.pair(16)==12160);}
 for(int64_t k:{0,2048,4097,9729})reject([&]{launch(good(),initial,1e-6,k);});
 for(int i=0;i<2;++i){
  auto in=good();in[i].present=false;reject([&]{launch(in);});
  in=good();in[i].contiguous=false;reject([&]{launch(in);});
  in=good();in[i].kind=c10::DeviceType::CPU;reject([&]{launch(in);});
  in=good();in[i].layout_kind=c10::Layout::Sparse;reject([&]{launch(in);});
  in=good();in[i].type=at::kFloat;reject([&]{launch(in);});
  in=good();in[i].shape[0]--;reject([&]{launch(in);});
  in=good();in[i].shape.push_back(1);reject([&]{launch(in);});
  in=good();in[i].backing.bytes--;reject([&]{launch(in);});
  in=good();in[i].backing.present=false;reject([&]{launch(in);});
  for(int64_t off:{int64_t(-1),int64_t(1),INT64_MAX}){in=good();in[i].offset=off;reject([&]{launch(in);});}
  for(uint64_t a:{0ULL,0x1200000001ULL,1ULL<<40,(1ULL<<40)-256}){in=good();in[i].address=a;reject([&]{launch(in);});}
 }
 {auto in=good();in[0].shape[1]--;reject([&]{launch(in);});}
 {auto in=good();in[1].backing.bytes=(2560+192)*2;reject([&]{launch(in);});}
 {auto in=good();in[1].offset=128;in[1].backing.bytes+=256;in[1].address+=256;launch(in);}
 for(uint64_t delta:{0ULL,256ULL,10485504ULL}){auto in=good();in[1].address=in[0].address+delta;reject([&]{launch(in);});}
 {auto in=good();in[0].address=in[1].address+256;reject([&]{launch(in);});}
 for(int i=0;i<5;++i){
  for(uint32_t a:{0u,0x3fffff00u,0x40001001u,0x407fff00u,0xffffff00u}){auto p=initial;p[i]=a;reject([&]{launch(good(),p);});}
  for(int j=0;j<5;++j)if(i!=j){auto p=initial;p[i]=p[j];reject([&]{launch(good(),p);});p[i]+=256;reject([&]{launch(good(),p);});}
 }
 {auto p=initial;p[1]=p[0]+1024;launch(good(),p);p[1]-=256;reject([&]{launch(good(),p);});}
 {auto p=initial;p[4]=SPM_ALLOC.base+SpmAllocator::SPM_USABLE-5120;launch(good(),p);p[4]+=256;reject([&]{launch(good(),p);});}
 for(double eps:{0.,-1.,1e-5,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})reject([&]{launch(good(),initial,eps);});
 SPM_ALLOC.initialized=false;reject([&]{launch(good());});SPM_ALLOC.initialized=true;
 SPM_ALLOC.drift=true;reject([&]{launch(good());});SPM_ALLOC.drift=false;
 missing=true;reject([&]{launch(good());},false);missing=false;
 graph.deny=true;const int old=kernel_gets;reject([&]{launch(good());},false);assert(kernel_gets==old);graph.deny=false;
}
'''
    harness = harness.replace("__DECL__", declaration).replace("__BODY__", body).replace("__NAME__", NAME)
    _run(tmp_path, harness, "o_ring_norm")
