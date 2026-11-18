"""Execute the actual KV1 pair-owner launcher with and without the fused Q projection."""
from pathlib import Path
from test_siglip_cold_profile import _body
from test_g05_q_gate_native import _compile_run

ROOT = Path(__file__).resolve().parents[1]


def test_fused_q_projection_keeps_both_register_images_and_rejects_hazards(tmp_path):
    source = (ROOT / 'src/ops/rpu_linear.cpp').read_text()
    decls = (ROOT / 'src/core/rpu_kernel_decls.h').read_text()

    def definition(text, marker):
        # The last occurrence: the plan helpers are forward-declared above the pair owner.
        start = text.rindex(marker)
        return text[start:text.index('{', start)] + _body(text[start:], marker)

    # The defaulted DB-candidate argument lives in the forward declaration.
    plan = 'static ActionNvfp4V2Plan plan_action_nvfp4_v2('
    start = source.index(plan)
    functions = [definition(decls, 'struct Pi05Nvfp4QProjection {') + ';',
                 definition(source, 'struct ActionNvfp4V2Plan {') + ';',
                 source[start:source.index(';', start) + 1]]
    for marker in (plan, 'static void launch_pi05_kv1_pair_owner_m50(',
                   'void rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(',
                   'void rpu_launch_pi05_kv_owner_db_candidate_kernel('):
        functions.append(definition(source, marker))
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
#include "rpu_linear_tiling.h"
#include "rpu_pi05_q_kv1_abi.h"
// The launcher's checks name their operand in the message.
template<class...A>void message(const A&...){}
#define TORCH_CHECK(ok,...) do{if(!(ok)){message(__VA_ARGS__);throw std::runtime_error("rejected");}}while(0)
namespace c10 {using Half=uint16_t; enum class DeviceType{PrivateUse1,CPU};}
namespace at {
constexpr int kByte=1,kHalf=2,kChar=3;
constexpr auto kPrivateUse1=c10::DeviceType::PrivateUse1;
using IntArrayRef=std::vector<int64_t>;
struct Device{c10::DeviceType value=kPrivateUse1;int index=0;
 c10::DeviceType type()const{return value;}bool operator==(Device b)const{return value==b.value&&index==b.index;}};
struct Tensor{int dtype=kByte;bool present=false;Device dev{};IntArrayRef shape;uintptr_t address=0;size_t elem=1;
 bool defined()const{return present;}int scalar_type()const{return dtype;}Device device()const{return dev;}
 bool is_contiguous()const{return true;}IntArrayRef sizes()const{return shape;}int64_t dim()const{return shape.size();}
 int64_t size(int i)const{return shape[i];}int64_t storage_offset()const{return 0;}
 size_t nbytes()const{size_t n=elem;for(auto s:shape)n*=s;return n;}void*data_ptr()const{return reinterpret_cast<void*>(address);}};
}
struct SpmAllocator{static constexpr size_t SPM_USABLE=8387584;
 bool is_initialized()const{return true;}uint32_t addr(int,int)const{return 0x20000000;}}allocator;
#define SPM_ALLOC allocator
int flushes=0;
void rpu_ddr_flush(void*){++flushes;}
uint64_t RpuGetDevAddr(void*p){return reinterpret_cast<uintptr_t>(p);}
enum class KernelId {ACC16=0};
KernelId autotile_nvfp4_acc16_kernel_id(int n){return KernelId((128-n)/16);}
// The REF carries every sibling unless a case removes one.
bool linear_db=true,pair_db=true,fused=true,wspm=true;
bool pi05_action_rows_admitted(int64_t rows){return rows==10||rows==50;}
bool pi05_nvfp4_acc16_linear_double_buffered(int64_t rows){return rows<=64&&linear_db;}
bool pi05_kv1_pair_owner_w8_double_buffered(int64_t rows){return rows<=64&&pair_db;}
struct KernelCache{static KernelCache&instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId id)const{
  return (id!=KernelId::PI05_Q_KV1_PAIR_DB_SSPM||fused)&&(id!=KernelId::PI05_Q_KV1_PAIR_DB_WSPM||wspm);}};
enum class GraphDdrRegisterRole{ModelWeight,Int8ModelWeight,PerChannelScale,PackedFp4ModelWeight,Fp8BlockScale};
enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;
 uint16_t reg_lo;uint16_t reg_hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel_t{std::map<int,uint16_t> regs;void reset_regs(){regs.clear();}
 int set_regs(int r,uint16_t v){assert(r>=0&&r<67);regs[r]=v;return 0;}int set_regs(int,uint32_t)=delete;};
struct Writer{void write(Kernel_t&);};
struct RpuKernelGraph{std::vector<GraphDdrRegisterOperandSpec> owners;Kernel_t kernel;KernelId id{};
 static bool has_active(){return true;}
 static RpuKernelGraph&active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId k,std::vector<GraphDdrRegisterOperandSpec> values){id=k;owners=std::move(values);return {};}
 Kernel_t*get(KernelId k){assert(k==id);return &kernel;}};
void Writer::write(Kernel_t&k){for(auto&o:RpuKernelGraph::active().owners){auto value=RpuGetDevAddr(o.tensor.data_ptr())>>8;
 k.set_regs(o.abi.reg_lo,uint16_t(value));k.set_regs(o.abi.reg_hi,uint16_t(value>>16));}}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
struct Queue{bool broadcast=false;int launches=0;std::vector<uint16_t> grid;
 void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel_t&,std::vector<uint16_t> gr,std::vector<uint8_t> cores){assert(broadcast&&cores.size()==8);grid=gr;++launches;}}queue;
Queue*get_queue(int n){assert(n==8);return &queue;}
#define GET_QUEUE(n) get_queue(n)
#include "pi05_execution_topology.h"
__FUNCTIONS__
at::Tensor tensor(int dtype,at::IntArrayRef shape,uintptr_t address,size_t elem){
 at::Tensor t;t.dtype=dtype;t.present=true;t.shape=shape;t.address=address;t.elem=elem;return t;}
template<class F>void reject(F f){const int launches=queue.launches,fl=flushes;try{f();assert(false);}catch(const std::runtime_error&){}
 assert(queue.launches==launches&&flushes==fl);}
uint32_t u32(const std::map<int,uint16_t>&r,int lo){return uint32_t(r.at(lo))|(uint32_t(r.at(lo+1))<<16);}
// Exactly these typed DDR owners, in staging order.
void owners(std::vector<std::pair<GraphDdrRegisterRole,int>> expect){
 const auto&o=RpuKernelGraph::active().owners;assert(o.size()==expect.size());
 for(size_t i=0;i<o.size();++i)assert(o[i].abi.role==expect[i].first&&o[i].abi.reg_lo==expect[i].second&&o[i].abi.reg_hi==expect[i].second+1);}
int main(){
 using Role=GraphDdrRegisterRole;
 auto&g=RpuKernelGraph::active();
 const auto kw=tensor(at::kChar,{2048,1024},0x100000000ULL,1),vw=tensor(at::kChar,{2048,1024},0x100200000ULL,1);
 const auto ks=tensor(at::kHalf,{2048},0x100400000ULL,2),vs=tensor(at::kHalf,{2048},0x100401000ULL,2);
 const auto qw=tensor(at::kByte,{2048,512},0x100500000ULL,1),qs=tensor(at::kByte,{64,2048},0x100600000ULL,1);
 const auto qt=tensor(at::kByte,{64,2048},0x100700000ULL,1);
 constexpr uint32_t base=0x20000000,x=base+0x10000,k=base+0x30000,v=base+0x38000,q=base+0x40000,ts=base+0x60000,slab=base+0x70000;
 const auto w8=rpu_pi_kv_linear::pair(false),f16=rpu_pi_kv_linear::pair(true);
 for(int rows:{10,50}){
  // Without a Q projection: the unified ParallelLinear image of the pair body,
  // V weight/scale/output at 32..37, on the double-buffered W8 owner when the
  // REF carries it and on the selected catalog owner otherwise.
  for(bool db:{true,false}){
   pair_db=db;
   rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(x,kw,ks,k,vw,vs,v,rows,nullptr);
   const auto&r=g.kernel.regs;
   assert(g.id==(db?KernelId::PI05_KV_OWNER_W8_DB:pi_wall_kernel_id(w8.ordinal)));
   assert(queue.grid==(db?std::vector<uint16_t>({2,1,1}):std::vector<uint16_t>({w8.grid_x,w8.grid_y,1})));
   owners({{Role::Int8ModelWeight,2},{Role::Int8ModelWeight,32},{Role::PerChannelScale,20},{Role::PerChannelScale,34}});
   assert(u32(r,0)==x&&u32(r,2)==kw.address>>8&&u32(r,4)==k&&u32(r,6)==0&&r.at(8)==1&&r.at(9)==rows);
   assert(u32(r,10)==32&&r.at(12)==1024&&r.at(13)==64&&r.at(14)==0&&r.at(15)==8);
   assert(u32(r,16)==2048&&u32(r,18)==64&&u32(r,20)==ks.address>>8&&u32(r,22)==0&&u32(r,24)==0);
   assert(r.at(26)==2&&r.at(27)==1&&r.at(28)==0&&r.at(29)==0);
   assert(u32(r,32)==vw.address>>8&&u32(r,34)==vs.address>>8&&u32(r,36)==v&&r.rbegin()->first==37);
  }
  pair_db=true;
  // With one: the Q Linear's registers 0..28, the pair's live slots from 32
  // (rpu_pi05_q_kv1::registers), five DDR owners and two more blocks.
  const Pi05Nvfp4QProjection proj{q,qw,qs,qt,2048,1024,ts,17,slab};
  int fl=flushes;
  rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(x,kw,ks,k,vw,vs,v,rows,&proj);
  const auto&r=g.kernel.regs;
  assert(g.id==KernelId::PI05_Q_KV1_PAIR_DB_SSPM&&queue.grid==std::vector<uint16_t>({6,1,1}));
  owners({{Role::PackedFp4ModelWeight,2},{Role::Int8ModelWeight,34},{Role::Int8ModelWeight,54},
          {Role::PerChannelScale,52},{Role::PerChannelScale,56}});
  const int sspm_flushes=flushes-fl;
  std::map<int,uint32_t> expect{{0,x},{2,qw.address>>8},{4,q},{8,rows},{10,256},{12,1024},{13,16},{15,8},
   {16,2048},{18,512},{20,slab},{22,4096},{24,ts},{26,1},{27,17},{28,4},
   {32,x},{34,kw.address>>8},{36,k},{40,rows},{42,32},{44,64},{46,8},{47,1},{48,2048},{50,64},
   {52,ks.address>>8},{54,vw.address>>8},{56,vs.address>>8},{58,v},{64,6},{65,1},{66,1}};
  // Every slot 0..66 is written; the rest are zero.
  const auto image=[&](const std::map<int,uint32_t>&e){
   assert(r.size()==67&&r.rbegin()->first==66);
   for(const auto&[reg,value]:r){
    const auto lo=e.find(reg),hi=e.find(reg-1);
    const bool wide=hi!=e.end()&&hi->second>0xffff;
    assert(lo!=e.end()?value==uint16_t(lo->second):value==(wide?uint16_t(hi->second>>16):0));
   }
   for(const auto&[reg,value]:e)assert(value<=0xffff||u32(r,reg)==value);};
  image(expect);
  // The SPM-resident Q weight: the _wspm sibling takes the run's absolute SPM
  // address in 2/3, the Q weight is no DDR owner and is not flushed.
  const uint32_t run=base+0x80000;
  Pi05Nvfp4QProjection resident=proj;resident.spm_weight=run;
  fl=flushes;
  rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(x,kw,ks,k,vw,vs,v,rows,&resident);
  assert(g.id==KernelId::PI05_Q_KV1_PAIR_DB_WSPM&&queue.grid==std::vector<uint16_t>({6,1,1}));
  owners({{Role::Int8ModelWeight,34},{Role::Int8ModelWeight,54},{Role::PerChannelScale,52},{Role::PerChannelScale,56}});
  assert(flushes-fl==sspm_flushes-1);
  auto resident_expect=expect;resident_expect[2]=run;
  image(resident_expect);
  // A REF without either sibling, FP16 owners, a layer whose scales stay in
  // DDR and any SPM overlap with the pair's intervals keep the two launches.
  const auto reject_with=[&](const Pi05Nvfp4QProjection&p){
   reject([&]{rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(x,kw,ks,k,vw,vs,v,rows,&p);});};
  fused=false;reject_with(proj);fused=true;
  pair_db=false;reject_with(proj);pair_db=true;
  linear_db=false;reject_with(proj);linear_db=true;
  reject_with({q,qw,qs,qt,2048,1024,ts,17,0});
  reject_with({k,qw,qs,qt,2048,1024,ts,17,slab});
  reject_with({x+0x1000,qw,qs,qt,2048,1024,ts,17,slab});
  reject_with({q,qw,qs,qt,2048,1024,ts,17,v-0x2000});
  reject_with({q,qw,qs,qt,4096,1024,ts,17,slab});
  // The resident run needs its payload and its own aligned SPM interval
  // (128 KiB per core) clear of every output.
  wspm=false;reject_with(resident);wspm=true;
  for(uint32_t bad:{run+0x80,q,k-0x1ff00,v,base+uint32_t(SpmAllocator::SPM_USABLE)-0x10000}){
   auto p=resident;p.spm_weight=bad;reject_with(p);}
  // The pair's own owners and intervals.
  reject([&]{rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(x,kw,ks,k,vw,vs,k,rows,nullptr);});
  reject([&]{rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(x+0x80,kw,ks,k,vw,vs,v,rows,nullptr);});
  reject([&]{rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(x,kw,ks,k,kw,vs,v,rows,nullptr);});
  reject([&]{rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(x,kw,vw,k,vw,vs,v,rows,nullptr);});
  const auto kh=tensor(at::kHalf,{2048,1024},0x100000000ULL,2),vh=tensor(at::kHalf,{2048,1024},0x100400000ULL,2);
  const at::Tensor none;
  rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(x,kh,none,k,vh,none,v,rows,nullptr);
  assert(g.id==pi_wall_kernel_id(f16.ordinal)&&queue.grid==std::vector<uint16_t>({f16.grid_x,f16.grid_y,1}));
  owners({{Role::ModelWeight,2},{Role::ModelWeight,32}});
  assert(g.kernel.regs.at(27)==2&&u32(g.kernel.regs,20)==0&&u32(g.kernel.regs,36)==v);
  reject([&]{rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(x,kh,none,k,vh,none,v,rows,&proj);});
  reject([&]{rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(x,kh,ks,k,vh,vs,v,rows,nullptr);});
 }
 // Unadmitted horizons.
 for(int rows:{1,16,64})
  reject([&]{rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(x,kw,ks,k,vw,vs,v,rows,nullptr);});
 // The DB candidate always takes the W8 DB owner at M50 and rejects FP16.
 pair_db=false;
 rpu_launch_pi05_kv_owner_db_candidate_kernel(x,kw,ks,k,vw,vs,v);
 assert(g.id==KernelId::PI05_KV_OWNER_W8_DB&&queue.grid==std::vector<uint16_t>({2,1,1})&&g.kernel.regs.at(9)==50);
 const auto kh=tensor(at::kHalf,{2048,1024},0x100000000ULL,2),vh=tensor(at::kHalf,{2048,1024},0x100400000ULL,2);
 const at::Tensor none;
 reject([&]{rpu_launch_pi05_kv_owner_db_candidate_kernel(x,kh,none,k,vh,none,v);});
}
'''
    _compile_run(tmp_path, 'pi05_q_kv1_pair_owner_native', cpp.replace('__FUNCTIONS__', '\n'.join(functions)))
