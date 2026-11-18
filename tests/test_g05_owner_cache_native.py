"""Execute actual owner continuation wrappers against typed/pending Graph doubles."""
from pathlib import Path
from test_g05_q_gate_native import _compile_run

ROOT = Path(__file__).resolve().parents[1]


def test_owner_norm_rope_and_segmented_cache_actual_wrappers(tmp_path):
    source = (ROOT / 'src/ops/rpu_llama_kvcache.cpp').read_text()
    actual = source[source.index('// Exact G05 W8 Action continuation.'):]
    cpp = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10 {using Half=uint16_t;enum class DeviceType{PrivateUse1,CPU};}
namespace at {constexpr int kHalf=1;
struct Tensor{uint64_t address;std::vector<int64_t> shape;bool present=true,contiguous=true;int dtype=kHalf;int64_t offset=0;c10::DeviceType location=c10::DeviceType::PrivateUse1;
 bool defined()const{return present;}bool is_contiguous()const{return contiguous;}int scalar_type()const{return dtype;}int64_t storage_offset()const{return offset;}int dim()const{return shape.size();}int64_t size(int i)const{return shape.at(i);}const std::vector<int64_t>&sizes()const{return shape;}
 size_t nbytes()const{size_t n=2;for(auto x:shape)n*=x;return n;}
 struct Device{c10::DeviceType value;c10::DeviceType type()const{return value;}bool operator==(const Device&o)const{return value==o.value;}};Device device()const{return {location};}
 void*data_ptr()const{return reinterpret_cast<void*>(address);}template<class T>T*data_ptr()const{return reinterpret_cast<T*>(address);}
};}
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8387584;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int core,uint32_t offset)const{assert(core==0);return 0x40000000+offset;}}allocator;
#define SPM_ALLOC allocator
uint64_t RpuGetDevAddr(void*p){return reinterpret_cast<uint64_t>(p);}
enum class KernelId{G05_OWNER_INSERT_KV_CACHE_M32D256_P0,G05_OWNER_INSERT_KV_CACHE_M32D256_P245,G05_Q_NORM_PARTIAL_ROPE_M32D256R64_GROUP4,G05_K_OWNER_NORM_PARTIAL_ROPE_M32D256R64_GROUP4,G05_K_OWNER_RMS_NORM_M32D256,G05_K_OWNER_PARTIAL_MROPE_M32D256R64,G05_OWNER_INSERT_KCACHE_MULTIWARP,G05_OWNER_INSERT_KCACHE_MULTIWARP_V16,G05_OWNER_INSERT_VCACHE_MULTIWARP,G05_OWNER_INSERT_VCACHE_MULTIWARP_V16};
constexpr auto NORM=KernelId::G05_K_OWNER_RMS_NORM_M32D256;constexpr auto ROPE=KernelId::G05_K_OWNER_PARTIAL_MROPE_M32D256R64;
constexpr auto K2=KernelId::G05_OWNER_INSERT_KCACHE_MULTIWARP;constexpr auto K16=KernelId::G05_OWNER_INSERT_KCACHE_MULTIWARP_V16;
constexpr auto V2=KernelId::G05_OWNER_INSERT_VCACHE_MULTIWARP;constexpr auto V16=KernelId::G05_OWNER_INSERT_VCACHE_MULTIWARP_V16;
struct KernelCache{std::set<KernelId>missing;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId id)const{return !missing.count(id);}};
enum class GraphDdrRegisterRole{SemanticInput,KeyCache,ValueCache};enum class GraphDdrRegisterAccess{Read,Write};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor owner;};
struct Kernel_t{std::map<int,uint16_t>regs;void reset_regs(){regs.clear();}void set_regs(int i,uint16_t v){regs[i]=v;}void set_regs(int,uint32_t)=delete;};
struct Submission{KernelId id;std::map<int,uint16_t>regs;std::vector<GraphDdrRegisterOperandSpec>operands;std::vector<uint16_t>grid;};
struct Writer{void write(Kernel_t&);};
struct RpuKernelGraph{std::optional<KernelId>pending,staged;bool written=false,active_flag=true;int stages=0,lookups=0;Kernel_t kernel;std::vector<GraphDdrRegisterOperandSpec>operands;std::vector<Submission>nodes;
 static RpuKernelGraph&active(){static RpuKernelGraph g;return g;}static bool has_active(){return active().active_flag;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> specs){assert(!staged&&!pending&&id!=NORM);staged=id;operands=std::move(specs);written=false;++stages;return {};}
 Kernel_t*get(KernelId id){assert(!pending);assert(id==NORM?!staged:(staged&&*staged==id));pending=id;written=id==NORM;++lookups;return &kernel;}
};
void Writer::write(Kernel_t&k){auto&g=RpuKernelGraph::active();assert(g.staged&&g.pending&&*g.staged==*g.pending&&!g.written);for(auto&o:g.operands){auto value=o.owner.address>>8;k.set_regs(o.abi.lo,uint16_t(value));k.set_regs(o.abi.hi,uint16_t(value>>16));}g.written=true;}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
std::vector<uint64_t>flushed;void rpu_ddr_flush(c10::Half*p){flushed.push_back(reinterpret_cast<uint64_t>(p));}
struct Queue{bool broadcast=false;void set_broadcast_mode(bool value){broadcast=value;}void enqueu_kernel(Kernel_t&k,std::vector<uint16_t>grid,std::vector<uint8_t>cores){auto&g=RpuKernelGraph::active();assert(broadcast&&g.pending&&g.written&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));g.nodes.push_back({*g.pending,k.regs,g.operands,grid});g.pending.reset();g.staged.reset();g.operands.clear();g.written=false;}}queue;
Queue*get_queue(int count){assert(count==8);return &queue;}
#define GET_QUEUE(n) get_queue(n)
enum class KvInsertKernel{V2,V16};enum class KvInsertRoute{ALIGNED_V16,HYBRID3,V2};
struct Part{KvInsertKernel kernel;int64_t position,token_offset,rows;};
struct KvInsertSegmentPlan{KvInsertRoute kind;std::vector<Part>parts;int64_t logical=32,physical=32;KvInsertRoute route()const{return kind;}size_t segment_count()const{return parts.size();}const Part&segment(size_t i)const{return parts.at(i);}int64_t logical_rows()const{return logical;}int64_t physical_rows()const{return physical;}};
void rpu_validate_kvinsert_segment_plan(const KvInsertSegmentPlan&p,int tp,int kv,int dim){assert(tp==8&&kv==8&&dim==256);TORCH_CHECK(!p.parts.empty());}
__ACTUAL__
uint32_t pair(const Submission&s,int i){return uint32_t(s.regs.at(i))|(uint32_t(s.regs.at(i+1))<<16);}
template<class F>void reject(F f){auto&g=RpuKernelGraph::active();const auto before=g.nodes.size(),flushes=flushed.size();const auto stages=g.stages,lookups=g.lookups;try{f();assert(false);}catch(const std::runtime_error&){}assert(g.nodes.size()==before&&flushed.size()==flushes&&g.stages==stages&&g.lookups==lookups&&!g.staged&&!g.pending);}
int main(){
 auto&g=RpuKernelGraph::active();const uint32_t key=0x40010000,norm=0x40020000;
 at::Tensor cos{0x1234000000,{96,32}},sin{0x1235000000,{96,32}};
 at::Tensor kc{0x1240000000,{1,136,1,16,8,16,16}},vc{0x1250000000,{1,136,1,16,8,16,16}};
 const KvInsertSegmentPlan p0{KvInsertRoute::ALIGNED_V16,{{KvInsertKernel::V16,0,0,32}}};
 const KvInsertSegmentPlan p245{KvInsertRoute::HYBRID3,{{KvInsertKernel::V2,245,0,11},{KvInsertKernel::V16,256,11,16},{KvInsertKernel::V2,272,27,5}}};
 auto launch_norm=[&]{rpu_launch_g05_k_owner_norm_rope_spm_kernel(key,norm,cos,sin,0);};
 auto launch_cache=[&]{rpu_launch_g05_owner_insert_kvcache_spm_with_plan(kc,vc,0x10000,0x20000,p245);};
 for(int pass=0;pass<2;++pass){cos.address+=256;sin.address+=512;auto begin=g.nodes.size();auto pos=pass?64:0;
  rpu_launch_g05_k_owner_norm_rope_spm_kernel(key,norm,cos,sin,pos);
  assert(g.nodes.size()==begin+2);const auto&n=g.nodes[begin];assert(n.id==NORM&&n.operands.empty()&&n.grid==std::vector<uint16_t>({32,1,1}));
  assert(n.regs.at(0)==256&&n.regs.at(1)==0x1c00&&n.regs.at(2)==512&&n.regs.at(3)==0&&n.regs.at(10)==0x11);
  assert(pair(n,4)==key&&pair(n,6)==key&&pair(n,8)==norm);
  const auto&r=g.nodes[begin+1];assert(r.id==ROPE&&r.grid==std::vector<uint16_t>({1,1,1})&&r.operands.size()==2);
  assert(pair(r,0)==cos.address>>8&&pair(r,2)==sin.address>>8&&pair(r,4)==key&&pair(r,6)==key&&pair(r,8)==uint32_t(pos));
  assert(r.regs.at(10)==256&&r.regs.at(11)==1&&r.regs.at(12)==32&&r.regs.at(13)==64&&r.regs.at(14)==1);
  for(int i=0;i<2;++i){const auto&o=r.operands[i];assert(o.abi.role==GraphDdrRegisterRole::SemanticInput&&o.abi.access==GraphDdrRegisterAccess::Read&&o.abi.lo==2*i&&o.abi.hi==2*i+1);}
 }
 for(const auto&plan:{p0,p245}){kc.address+=256;vc.address+=256;auto begin=g.nodes.size();rpu_launch_g05_owner_insert_kvcache_spm_with_plan(kc,vc,0x10000,0x20000,plan);
  assert(g.nodes.size()==begin+2*plan.segment_count());for(int kind=0;kind<2;++kind)for(size_t i=0;i<plan.segment_count();++i){const auto&s=g.nodes[begin+kind*plan.segment_count()+i];const auto&p=plan.segment(i);bool v16=p.kernel==KvInsertKernel::V16;
   assert(s.id==(kind?(v16?V16:V2):(v16?K16:K2)));assert(s.grid==std::vector<uint16_t>({uint16_t(v16?p.rows/16:p.rows),1,1}));
   assert(pair(s,0)==uint32_t(p.position)&&s.regs.at(2)==1&&s.regs.at(3)==256&&s.regs.at(4)==p.rows&&s.regs.at(21)==8);
   assert(pair(s,6)==uint32_t((kind?0x20000:0x10000)+p.token_offset*512));assert(pair(s,8)==(kind?vc.address:kc.address)>>8);
   assert(s.operands.size()==1);const auto&o=s.operands[0];assert(o.abi.role==(kind?GraphDdrRegisterRole::ValueCache:GraphDdrRegisterRole::KeyCache)&&o.abi.access==GraphDdrRegisterAccess::Write&&o.abi.lo==8&&o.abi.hi==9);
  }
 }
 for(auto missing:{NORM,ROPE,K2,K16,V2,V16}){KernelCache::instance().missing.insert(missing);reject(launch_norm);reject(launch_cache);KernelCache::instance().missing.clear();}
 g.active_flag=false;reject(launch_norm);reject(launch_cache);g.active_flag=true;
 allocator.initialized=false;reject(launch_norm);reject(launch_cache);allocator.initialized=true;
 for(int which=0;which<2;++which)for(int fault=0;fault<11;++fault){auto a=cos,b=sin;auto&t=which?b:a;
  if(fault==0)t.present=false;if(fault==1)t.location=c10::DeviceType::CPU;if(fault==2)t.dtype=2;if(fault==3)t.contiguous=false;if(fault==4)t.offset=1;
  if(fault==5)t.shape={31,32};if(fault==6)t.shape={96,31};if(fault==7)t.address+=1;if(fault==8)t.address=0;if(fault==9)t.address=(UINT64_C(1)<<40)-256;if(fault==10)t.shape={96,32,1};
  reject([&]{rpu_launch_g05_k_owner_norm_rope_spm_kernel(key,norm,a,b,0);});}
 reject([&]{rpu_launch_g05_k_owner_norm_rope_spm_kernel(key,norm,cos,cos,0);});
 for(int64_t position:{int64_t{-1},int64_t{65},int64_t{4294967296}})reject([&]{rpu_launch_g05_k_owner_norm_rope_spm_kernel(key,norm,cos,sin,position);});
 for(auto address:{key+1,0x3fffff00u,0x407fff00u})reject([&]{rpu_launch_g05_k_owner_norm_rope_spm_kernel(address,norm,cos,sin,0);});
 reject([&]{rpu_launch_g05_k_owner_norm_rope_spm_kernel(key,key+256,cos,sin,0);});
 for(int which=0;which<2;++which)for(int fault=0;fault<11;++fault){auto a=kc,b=vc;auto&t=which?b:a;
  if(fault==0)t.present=false;if(fault==1)t.location=c10::DeviceType::CPU;if(fault==2)t.dtype=2;if(fault==3)t.contiguous=false;if(fault==4)t.offset=128;
  if(fault==5)t.shape[1]=17;if(fault==6)t.shape[4]=2;if(fault==7)t.address+=1;if(fault==8)t.address=0;if(fault==9)t.address=(UINT64_C(1)<<40)-256;if(fault==10)t.shape.pop_back();
  reject([&]{rpu_launch_g05_owner_insert_kvcache_spm_with_plan(a,b,0x10000,0x20000,p245);});}
 reject([&]{rpu_launch_g05_owner_insert_kvcache_spm_with_plan(kc,kc,0x10000,0x20000,p0);});
 for(auto off:{0x10001u,SpmAllocator::SPM_USABLE-256,0x40010000u})reject([&]{rpu_launch_g05_owner_insert_kvcache_spm_with_plan(kc,vc,off,0x20000,p0);});
 reject([&]{rpu_launch_g05_owner_insert_kvcache_spm_with_plan(kc,vc,0x10000,0x10100,p0);});
 for(int fault=0;fault<7;++fault){auto p=p245;if(fault==0)p.logical=31;if(fault==1)p.physical=48;if(fault==2)p.parts[1].token_offset=12;if(fault==3)p.parts[2].rows=4;if(fault==4)p.parts[0].position=246;if(fault==5)p.kind=KvInsertRoute::V2;if(fault==6)p.parts.pop_back();
  reject([&]{rpu_launch_g05_owner_insert_kvcache_spm_with_plan(kc,vc,0x10000,0x20000,p);});}
 // Real owner capacity governs bounds: exact final legal S18 is accepted.
 kc.shape[1]=vc.shape[1]=18;launch_cache();
 // Exact last SPM interval is accepted, no absolute/offset ambiguity.
 rpu_launch_g05_owner_insert_kvcache_spm_with_plan(kc,vc,0x10000,SpmAllocator::SPM_USABLE-16384,p245);
}
'''
    # Separate compact fault statements for -Wmisleading-indentation under -Werror.
    cpp = cpp.replace(';if(', ';\n  if(')
    _compile_run(tmp_path, 'g05_owner_cache_actual', cpp.replace('__ACTUAL__', actual))
