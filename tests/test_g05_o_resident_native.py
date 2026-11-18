"""Execute the actual cold packer, typed preload and resident O launcher boundaries."""
from pathlib import Path
from test_g05_q_gate_native import _compile_run
from test_siglip_cold_profile import _body
ROOT=Path(__file__).resolve().parents[1]

def test_g05_o_resident_actual_native_owners_preload_and_registers(tmp_path):
    source=(ROOT/'src/ops/rpu_linear.cpp').read_text();functions=[]
    for marker in ['void check_pi05_resident_spm(',
                   'at::Tensor rpu_pack_g05_o_resident_w8a16(',
                   'void rpu_launch_g05_o_resident_w8a16_acc16_kernel(',
                   'void rpu_launch_g05_gateup_resident_w8a16_preload(']:
        start=source.index(marker);functions.append(source[start:source.index('{',start)]+_body(source,marker))
    cpp=r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>
#include "rpu_linear_tiling.h"
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10{using Half=uint16_t;enum class DeviceType{PrivateUse1,CPU};}
int cpu_copies=0,allocations=0,flushes=0;
namespace at{constexpr int kHalf=1,kChar=2;
struct Device{c10::DeviceType value=c10::DeviceType::PrivateUse1;int index=0;c10::DeviceType type()const{return value;}bool operator==(Device b)const{return value==b.value&&index==b.index;}};
struct Tensor{uint64_t address=0x1234567800ULL;bool present=true,contig=true;int dtype=kChar,rank=2;int64_t rows=1024,cols=2048,offset=0;Device dev{};
 bool defined()const{return present;}int scalar_type()const{return dtype;}bool is_contiguous()const{return contig;}int dim()const{return rank;}int64_t size(int i)const{return i?cols:rows;}int64_t storage_offset()const{return offset;}
 size_t nbytes()const{return size_t(rows)*(rank==2?cols:1)*(dtype==kHalf?2:1);}Device device()const{return dev;}
 template<class T>T*data_ptr()const{return reinterpret_cast<T*>(address);}
 Tensor cpu()const{++cpu_copies;auto t=*this;t.dev.value=c10::DeviceType::CPU;return t;}
 Tensor view(std::initializer_list<int64_t>s)const{auto t=*this;auto v=std::vector<int64_t>(s);assert((v==std::vector<int64_t>({64,8,8,512})||v==std::vector<int64_t>({8,262144})));if(v.size()==2){t.rows=v[0];t.cols=v[1];}return t;}
 Tensor permute(std::initializer_list<int64_t>s)const{assert(std::vector<int64_t>(s)==std::vector<int64_t>({2,0,1,3}));return *this;}
 Tensor contiguous()const{return *this;}Tensor to(Device d)const{++allocations;auto t=*this;t.dev=d;t.address=0x789abcde00ULL;return t;}
};using TensorList=std::vector<Tensor>;
Tensor cat(const std::vector<Tensor>&v,int axis){assert(axis==1&&v.size()==4);for(auto&t:v)assert(t.rows==8&&t.cols==262144&&t.dtype==kChar&&t.dev.type()==c10::DeviceType::CPU);auto t=v[0];t.cols=1048576;return t;}}
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8387584;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int c,int off)const{assert(c==0);return 0x20000000u+off;}}allocator;
#define SPM_ALLOC allocator
uint64_t RpuGetDevAddr(void*p){return reinterpret_cast<uint64_t>(p);}
template<class T>void rpu_ddr_flush(T*){++flushes;}template<class T>void rpu_ddr_flush_force(T*){++flushes;}
enum class KernelId{G05_O_RESIDENT_W8A16_ACC16_M32N1024K256};constexpr auto O=KernelId::G05_O_RESIDENT_W8A16_ACC16_M32N1024K256;
struct KernelCache{bool available=true,n128_available=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId id)const{assert(id==O);return available;}};
enum class GraphDdrRegisterRole{PerChannelScale};enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor owner;};
struct Kernel{std::map<int,uint16_t>regs;void reset_regs(){regs.clear();}void set_regs(int r,uint16_t v){regs[r]=v;}void set_regs(int,uint32_t)=delete;};
struct Writer{void write(Kernel&);};struct Submission{std::map<int,uint16_t>regs;at::Tensor scale;std::vector<uint16_t>grid;};
bool capture=false;
struct RpuKernelGraph{bool staged=false,pending=false,written=false;int stages=0,lookups=0;KernelId staged_id=O;Kernel kernel;at::Tensor scale;std::vector<Submission>submissions;
 static bool has_active(){return capture;}static RpuKernelGraph&active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec>s){assert(id==O&&!staged&&!pending&&s.size()==1);staged_id=id;assert(s[0].abi.lo==20&&s[0].abi.hi==21);scale=s[0].owner;staged=true;++stages;return {};}
 Kernel*get(KernelId id){assert(id==staged_id&&KernelCache::instance().has_loaded(id)&&staged&&!pending);pending=true;++lookups;return &kernel;}};
void Writer::write(Kernel&k){auto&g=RpuKernelGraph::active();assert(g.staged&&g.pending&&!g.written);uint64_t a=g.scale.address>>8;k.set_regs(20,uint16_t(a));k.set_regs(21,uint16_t(a>>16));g.written=true;}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
struct Queue{bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}void enqueu_kernel(Kernel&k,std::vector<uint16_t>grid,std::vector<uint8_t>cores){auto&g=RpuKernelGraph::active();assert(broadcast&&g.pending&&g.written&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));g.submissions.push_back({k.regs,g.scale,grid});g.staged=g.pending=g.written=false;}}queue;
Queue*get_queue(int n){assert(n==8);return &queue;}
#define GET_QUEUE(n) get_queue(n)
enum class GraphOuterFastDmaSide{Source};
namespace graph_dma{bool staged=false;uint64_t source=0;int count=0;bool active(){return true;}void stage_census_fixed_burst(const at::Tensor&t,GraphOuterFastDmaSide,size_t begin,size_t bytes,size_t packets){assert(!staged&&begin==0&&bytes==8388608&&packets==8&&t.dtype==at::kChar);source=t.address;staged=true;++count;}}
int dmas=0;
void rpu_launch_ddr_scatter_spm_dma(c10::Half*p,int64_t n,int64_t stride,uint32_t spm,int cores){assert(graph_dma::staged&&reinterpret_cast<uint64_t>(p)==graph_dma::source&&n==524288&&stride==1048576&&cores==8&&spm==allocator.addr(0,0)+0x100000);graph_dma::staged=false;++dmas;}
__FUNCTIONS__
template<class F>void reject(F f){auto&g=RpuKernelGraph::active();auto s=g.submissions.size();int c=cpu_copies,a=allocations,fl=flushes,st=g.stages,lk=g.lookups,dm=dmas,dc=graph_dma::count;try{f();assert(false);}catch(const std::runtime_error&){}assert(s==g.submissions.size()&&c==cpu_copies&&a==allocations&&fl==flushes&&st==g.stages&&lk==g.lookups&&dm==dmas&&dc==graph_dma::count&&!g.staged&&!g.pending);}
uint32_t pair(const Submission&s,int r){return uint32_t(s.regs.at(r))|(uint32_t(s.regs.at(r+1))<<16);}
int main(){
 std::vector<at::Tensor>weights(4);for(int i=0;i<4;++i)weights[i].address+=uint64_t(i)*0x200000;
 auto packed=rpu_pack_g05_o_resident_w8a16(weights);assert(packed.rows==8&&packed.cols==1048576&&packed.dtype==at::kChar&&cpu_copies==4&&allocations==1);
 const uint32_t base=allocator.addr(0,0),x=base+0x10000,w=base+0x100000,y=base+0x20000;
 rpu_launch_g05_gateup_resident_w8a16_preload(packed,w);assert(dmas==1&&!graph_dma::staged);
 for(int count:{0,3,5})reject([&]{rpu_pack_g05_o_resident_w8a16(std::vector<at::Tensor>(count));});
 capture=true;reject([&]{rpu_pack_g05_o_resident_w8a16(weights);});capture=false;
 for(int index=0;index<4;++index)for(int fault=0;fault<13;++fault){auto bad=weights;auto&t=bad[index];
  if(fault==0){t.present=false;}if(fault==1){t.dtype=at::kHalf;}if(fault==2){t.contig=false;}if(fault==3){t.dev.value=c10::DeviceType::CPU;}
  if(fault==4){t.rank=1;}if(fault==5){t.rows=1023;}if(fault==6){t.cols=2047;}if(fault==7){t.offset=256;}
  if(fault==8){t.address=0;}if(fault==9){t.address+=1;}if(fault==10){t.address=UINT64_C(1)<<40;}if(fault==11){t.address=(UINT64_C(1)<<40)-256;}if(fault==12){t.dev.index=1;}
  reject([&]{rpu_pack_g05_o_resident_w8a16(bad);});}
 for(bool n128:{false,true}){
 at::Tensor scale;scale.dtype=at::kHalf;scale.rank=1;scale.rows=1024;scale.address=0x3456789a00ULL;
 auto launch=[&](uint32_t a,uint32_t b,uint32_t c,const at::Tensor&s){rpu_launch_g05_o_resident_w8a16_acc16_kernel(a,b,c,s,n128);};
 auto&g=RpuKernelGraph::active();
 for(int pass=0;pass<2;++pass){scale.address+=256;launch(x,w,y,scale);auto&s=g.submissions.back();assert(g.staged_id==O&&s.grid==std::vector<uint16_t>({13,1,1}));assert(pair(s,0)==x&&pair(s,2)==w&&pair(s,4)==y&&pair(s,6)==0&&pair(s,20)==scale.address>>8&&s.scale.address==scale.address);assert(s.regs.at(8)==0&&s.regs.at(9)==32&&pair(s,10)==1024&&s.regs.at(12)==256&&s.regs.at(13)==16&&s.regs.at(14)==0&&s.regs.at(15)==8&&pair(s,16)==512&&pair(s,18)==2048&&pair(s,22)==0&&pair(s,24)==0&&s.regs.at(26)==2&&s.regs.at(27)==1&&s.regs.at(28)==0&&s.regs.at(29)==0);}
 for(int fault=0;fault<12;++fault){auto s=scale;if(fault==0)s.present=false;if(fault==1)s.dtype=at::kChar;if(fault==2)s.contig=false;if(fault==3)s.dev.value=c10::DeviceType::CPU;if(fault==4)s.rank=2;if(fault==5)s.rows=1023;if(fault==6)s.offset=256;if(fault==7)s.address=0;if(fault==8)s.address+=1;if(fault==9)s.address=UINT64_C(1)<<40;if(fault==10)s.address=(UINT64_C(1)<<40)-256;if(fault==11)s.rows=1025;reject([&]{launch(x,w,y,s);});}
 for(int index=0;index<3;++index)for(auto bad:{base-256,base+SpmAllocator::SPM_USABLE-256,0xffffff00u,x+1}){std::array<uint32_t,3>p{x,w,y};p[index]=bad;reject([&]{launch(p[0],p[1],p[2],scale);});}
 for(auto p:{std::array<uint32_t,3>{x,x,y},{x,w,x},{x,y,y},{x,w,x+256}})reject([&]{launch(p[0],p[1],p[2],scale);});
 auto edge=scale;edge.address=(UINT64_C(1)<<40)-2048;launch(x,w,base+SpmAllocator::SPM_USABLE-65536,edge);
 auto&cache=KernelCache::instance();
 cache.available=false;reject([&]{launch(x,w,y,scale);});cache.available=true;cache.n128_available=false;launch(x,w,y,scale);cache.n128_available=true;
 allocator.initialized=false;reject([&]{launch(x,w,y,scale);});allocator.initialized=true;
 }
}
'''
    _compile_run(tmp_path,'g05_o_resident',cpp.replace('__FUNCTIONS__','\n'.join(functions)))
