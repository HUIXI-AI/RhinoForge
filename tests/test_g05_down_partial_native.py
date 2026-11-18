"""Execute actual partial-Down owner, preload and hybrid-register boundaries."""
from pathlib import Path
from test_g05_q_gate_native import _compile_run
from test_siglip_cold_profile import _body
ROOT=Path(__file__).resolve().parents[1]

def test_partial_down_native_owner_and_transport_boundaries(tmp_path):
    source=(ROOT/'src/ops/rpu_linear.cpp').read_text();functions=[]
    for marker in ['void check_pi05_resident_spm(',
                   'at::Tensor rpu_pack_g05_down_partial_w8a16(',
                   'void rpu_launch_g05_down_partial_w8a16_preload(',
                   'void rpu_launch_g05_down_partial_w8a16_acc16_kernel(']:
        start=source.index(marker);functions.append(source[start:source.index('{',start)]+_body(source,marker))
    cpp=r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include "rpu_linear_tiling.h"
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10{using Half=uint16_t;enum class DeviceType{PrivateUse1,CPU};}
int copies=0,allocations=0,flushes=0;
namespace at{constexpr int kHalf=1,kChar=2;
struct Device{c10::DeviceType value=c10::DeviceType::PrivateUse1;int index=0;c10::DeviceType type()const{return value;}bool operator==(Device b)const{return value==b.value&&index==b.index;}};
struct Tensor{uint64_t address=0x1234567800ULL;bool present=true,contig=true;int dtype=kChar,rank=2;int64_t rows=1024,cols=4096,offset=0;Device dev{};
 bool defined()const{return present;}int scalar_type()const{return dtype;}bool is_contiguous()const{return contig;}int dim()const{return rank;}int64_t size(int i)const{return i?cols:rows;}int64_t storage_offset()const{return offset;}
 size_t nbytes()const{return size_t(rows)*(rank==2?cols:1)*(dtype==kHalf?2:1);}Device device()const{return dev;}
 template<class T>T*data_ptr()const{return reinterpret_cast<T*>(address);}
 Tensor cpu()const{++copies;auto t=*this;t.dev.value=c10::DeviceType::CPU;return t;}
 Tensor view(std::initializer_list<int64_t>s)const{auto t=*this;auto v=std::vector<int64_t>(s);assert((v==std::vector<int64_t>({64,16,8,512})||v==std::vector<int64_t>({8,344064})));if(v.size()==2){t.rows=v[0];t.cols=v[1];}return t;}
 Tensor narrow(int d,int64_t begin,int64_t count)const{assert(d==0&&begin==0&&count==42);return *this;}
 Tensor permute(std::initializer_list<int64_t>s)const{assert(std::vector<int64_t>(s)==std::vector<int64_t>({2,0,1,3}));return *this;}
 Tensor contiguous()const{return *this;}Tensor to(Device d)const{++allocations;auto t=*this;t.dev=d;t.address=0x789abcde00ULL;return t;}
};}
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8387584;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int c,int off)const{assert(c==0);return 0x20000000u+off;}}allocator;
#define SPM_ALLOC allocator
uint64_t RpuGetDevAddr(void*p){return reinterpret_cast<uint64_t>(p);}
template<class T>void rpu_ddr_flush(T*){++flushes;}template<class T>void rpu_ddr_flush_force(T*){++flushes;}
enum class KernelId{G05_DOWN_PARTIAL672_W8A16_ACC16_M32N1024K512};constexpr auto ID=KernelId::G05_DOWN_PARTIAL672_W8A16_ACC16_M32N1024K512;
struct KernelCache{bool available=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId id)const{assert(id==ID);return available;}};
enum class GraphDdrRegisterRole{PerChannelScale,Int8ModelWeight};enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor owner;};
struct Kernel{std::map<int,uint16_t>regs;void reset_regs(){regs.clear();}void set_regs(int r,uint16_t v){regs[r]=v;}void set_regs(int,uint32_t)=delete;};
struct Writer{void write(Kernel&);};struct Submission{std::map<int,uint16_t>regs;std::vector<GraphDdrRegisterOperandSpec>owners;};
bool capture=false;
struct RpuKernelGraph{bool staged=false,pending=false,written=false;int stages=0,lookups=0;Kernel kernel;std::vector<GraphDdrRegisterOperandSpec>owners;std::vector<Submission>submissions;
 static bool has_active(){return capture;}static RpuKernelGraph&active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec>s){assert(id==ID&&!staged&&!pending&&s.size()==2);assert(s[0].abi.role==GraphDdrRegisterRole::Int8ModelWeight&&s[0].abi.lo==2&&s[0].abi.hi==3);assert(s[1].abi.role==GraphDdrRegisterRole::PerChannelScale&&s[1].abi.lo==20&&s[1].abi.hi==21);owners=s;staged=true;++stages;return {};}
 Kernel*get(KernelId id){assert(id==ID&&KernelCache::instance().available&&staged&&!pending);pending=true;++lookups;return &kernel;}};
void Writer::write(Kernel&k){auto&g=RpuKernelGraph::active();assert(g.staged&&g.pending&&!g.written);for(auto&s:g.owners){uint64_t a=s.owner.address>>8;k.set_regs(s.abi.lo,uint16_t(a));k.set_regs(s.abi.hi,uint16_t(a>>16));}g.written=true;}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
struct Queue{bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}void enqueu_kernel(Kernel&k,std::vector<uint16_t>grid,std::vector<uint8_t>cores){auto&g=RpuKernelGraph::active();assert(broadcast&&g.pending&&g.written&&grid==std::vector<uint16_t>({22,1,1})&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));g.submissions.push_back({k.regs,g.owners});g.staged=g.pending=g.written=false;}}queue;
Queue*get_queue(int n){assert(n==8);return &queue;}
#define GET_QUEUE(n) get_queue(n)
enum class GraphOuterFastDmaSide{Source};
namespace graph_dma{bool staged=false;uint64_t source=0;int count=0;bool active(){return true;}void stage_census_fixed_burst(const at::Tensor&t,GraphOuterFastDmaSide,size_t begin,size_t bytes,size_t packets){assert(!staged&&begin==0&&bytes==2752512&&packets==8&&t.dtype==at::kChar);source=t.address;staged=true;++count;}}
int dmas=0;
void rpu_launch_ddr_scatter_spm_dma(c10::Half*p,int64_t n,int64_t stride,uint32_t spm,int cores){assert(graph_dma::staged&&reinterpret_cast<uint64_t>(p)==graph_dma::source&&n==172032&&stride==344064&&cores==8&&spm==allocator.addr(0,0)+0x100000);graph_dma::staged=false;++dmas;}
__FUNCTIONS__
template<class F>void reject(F f){auto&g=RpuKernelGraph::active();auto s=g.submissions.size();int c=copies,a=allocations,fl=flushes,st=g.stages,lk=g.lookups,dm=dmas,dc=graph_dma::count;try{f();assert(false);}catch(const std::runtime_error&){}assert(s==g.submissions.size()&&c==copies&&a==allocations&&fl==flushes&&st==g.stages&&lk==g.lookups&&dm==dmas&&dc==graph_dma::count&&!g.staged&&!g.pending);}
uint32_t pair(const Submission&s,int r){return uint32_t(s.regs.at(r))|(uint32_t(s.regs.at(r+1))<<16);}
at::Tensor fault(at::Tensor t,int f){if(f==0)t.present=false;if(f==1)t.dtype=t.dtype==at::kChar?at::kHalf:at::kChar;if(f==2)t.contig=false;if(f==3)t.dev.value=c10::DeviceType::CPU;if(f==4)t.rank=t.rank==1?2:1;if(f==5)--t.rows;if(f==6)t.offset=256;if(f==7)t.address=0;if(f==8)++t.address;if(f==9)t.address=UINT64_C(1)<<40;if(f==10)t.address=(UINT64_C(1)<<40)-256;return t;}
int main(){
 at::Tensor w;auto packed=rpu_pack_g05_down_partial_w8a16(w);assert(packed.rows==8&&packed.cols==344064&&packed.dtype==at::kChar&&copies==1&&allocations==1);
 const uint32_t base=allocator.addr(0,0),x=base+0x10000,sw=base+0x100000,y=base+0x20000;
 rpu_launch_g05_down_partial_w8a16_preload(packed,sw);assert(dmas==1&&!graph_dma::staged);
 capture=true;reject([&]{rpu_pack_g05_down_partial_w8a16(w);});capture=false;
 for(int f=0;f<11;++f){auto bad=fault(w,f);reject([&]{rpu_pack_g05_down_partial_w8a16(bad);});auto p=fault(packed,f);reject([&]{rpu_launch_g05_down_partial_w8a16_preload(p,sw);});}
 {auto bad=w;bad.cols=2048;reject([&]{rpu_pack_g05_down_partial_w8a16(bad);});}
 at::Tensor scale;scale.dtype=at::kHalf;scale.rank=1;scale.rows=1024;scale.address=0x3456789a00ULL;
 auto launch=[&](uint32_t a,uint32_t b,uint32_t c,const at::Tensor&weight,const at::Tensor&s){rpu_launch_g05_down_partial_w8a16_acc16_kernel(a,weight,b,c,s);};
 auto&g=RpuKernelGraph::active();
 for(int pass=0;pass<2;++pass){scale.address+=256;w.address+=256;launch(x,sw,y,w,scale);auto&s=g.submissions.back();assert(pair(s,0)==x&&pair(s,2)==w.address>>8&&pair(s,4)==y&&pair(s,6)==0&&pair(s,20)==scale.address>>8&&pair(s,32)==sw);assert(s.regs.at(8)==0&&s.regs.at(9)==32&&pair(s,10)==1024&&s.regs.at(12)==512&&s.regs.at(13)==32&&s.regs.at(14)==0&&s.regs.at(15)==8&&pair(s,16)==1024&&pair(s,18)==2048&&pair(s,22)==0&&pair(s,24)==0&&s.regs.at(26)==2&&s.regs.at(27)==1&&s.regs.at(28)==0&&s.regs.at(29)==0);}
 for(int f=0;f<11;++f){auto bad=fault(w,f);reject([&]{launch(x,sw,y,bad,scale);});bad=fault(scale,f);reject([&]{launch(x,sw,y,w,bad);});}
 {auto bad=scale;bad.dev.index=1;reject([&]{launch(x,sw,y,w,bad);});bad=scale;bad.address=w.address+256;reject([&]{launch(x,sw,y,w,bad);});}
 for(int index=0;index<3;++index)for(auto bad:{base-256,base+SpmAllocator::SPM_USABLE-256,0xffffff00u,x+1}){std::array<uint32_t,3>p{x,sw,y};p[index]=bad;reject([&]{launch(p[0],p[1],p[2],w,scale);});}
 for(auto p:{std::array<uint32_t,3>{x,x,y},{x,sw,x},{x,y,y},{x,sw,x+256}})reject([&]{launch(p[0],p[1],p[2],w,scale);});
 for(auto bad:{base-256,base+SpmAllocator::SPM_USABLE-256,sw+1})reject([&]{rpu_launch_g05_down_partial_w8a16_preload(packed,bad);});
 auto edge=scale;edge.address=(UINT64_C(1)<<40)-2048;launch(x,sw,base+SpmAllocator::SPM_USABLE-65536,w,edge);
 KernelCache::instance().available=false;reject([&]{launch(x,sw,y,w,scale);});KernelCache::instance().available=true;
 allocator.initialized=false;reject([&]{launch(x,sw,y,w,scale);});reject([&]{rpu_launch_g05_down_partial_w8a16_preload(packed,sw);});allocator.initialized=true;
}
'''
    _compile_run(tmp_path,'g05_down_partial',cpp.replace('__FUNCTIONS__','\n'.join(functions)))
