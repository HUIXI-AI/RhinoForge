"""Execute the actual typed NVFP4 wrappers and registers, including Pi limits and lanes."""
from pathlib import Path
from test_siglip_cold_profile import _body
from test_g05_q_gate_native import _compile_run

ROOT = Path(__file__).resolve().parents[1]


def test_nvfp4_typed_launch_preserves_precision_owners_and_model_boundaries(tmp_path):
    source = (ROOT / 'src/ops/rpu_linear.cpp').read_text()

    def definition(marker):
        # The last occurrence: the plan helpers are forward-declared above the pair owner.
        start = source.rindex(marker)
        return source[start:source.index('{', start)] + _body(source[start:], marker)

    functions = [definition('struct ActionNvfp4V2Plan {') + ';']
    for marker in ('void check_pi05_resident_spm(', 'static ActionNvfp4V2Plan plan_action_nvfp4_v2(',
                   'template <typename Kernel>\nstatic void write_action_nvfp4_v2_regs(',
                   'static void launch_action_nvfp4_v2_kernel(',
                   'void rpu_launch_pi05_nvfp4_v2_kernel(', 'void rpu_launch_g05_nvfp4_v2_kernel('):
        functions.append(definition(marker))
    # The Pi wrapper's defaulted tile-scale argument lives in the declaration.
    decls = (ROOT / 'src/core/rpu_kernel_decls.h').read_text()
    start = decls.index('void rpu_launch_pi05_nvfp4_v2_kernel(')
    functions.insert(0, decls[start:decls.index(';', start) + 1])
    # The merge a split-KV attention may leave to the O launch.
    for marker in ('struct Pi05SplitKvMerge {', 'struct Pi05SplitKvPlan {'):
        start = decls.index(marker)
        functions.insert(0, decls[start:decls.index('\n};', start) + 3])
    cpp = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <vector>
#include "rpu_linear_tiling.h"
#include "pi05_execution_topology.h"
#define TORCH_CHECK(ok,...) do{if(!(ok))throw std::runtime_error("rejected");}while(0)
namespace c10 {using Half=uint16_t;}
namespace at {
constexpr int kByte=1,kHalf=2;
enum class kPrivateUse1Type{RPU,CPU}; constexpr auto kPrivateUse1=kPrivateUse1Type::RPU;
using IntArrayRef=std::vector<int64_t>;
struct Device{ kPrivateUse1Type value=kPrivateUse1; int index=0;
 kPrivateUse1Type type()const{return value;} bool operator==(Device b)const{return value==b.value&&index==b.index;}};
struct Tensor{int dtype=kByte;bool contig=true;bool present=false;Device dev{};IntArrayRef shape{2048,512};uintptr_t address=0x1234567800ULL;
 bool defined()const{return present;}int scalar_type()const{return dtype;}Device device()const{return dev;}bool is_contiguous()const{return contig;}
 IntArrayRef sizes()const{return shape;}void*data_ptr()const{return reinterpret_cast<void*>(address);}};
}
struct SpmAllocator{static constexpr size_t SPM_USABLE=8387584;bool initialized=true;
 bool is_initialized()const{return initialized;}uint32_t addr(int,int)const{return 0x20000000;}}allocator;
#define SPM_ALLOC allocator
int flushes=0;
void rpu_ddr_flush(void*){++flushes;}
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}
enum class KernelId {ACC16=0,PARALLEL_LINEAR_NVFP4_V2_M384N48=100,PARALLEL_LINEAR_NVFP4_V2_M320N64=101,
 PI05_NVFP4_LINEAR_DB=102,PI05_NVFP4_LINEAR_DB_TSCALE=103,PI05_NVFP4_LINEAR_DB_SSPM=104,
 PI05_SPLIT_KV_MERGE_O_NVFP4_ACC16_M64N64K128_DB_SSPM=105,PI05_SPLIT_KV_MERGE_ROT_OV_O_NVFP4_ACC16_M64N64K128_DB_SSPM=106};
bool spm_scaled_id(KernelId k){return k==KernelId::PI05_NVFP4_LINEAR_DB_SSPM||
 k==KernelId::PI05_SPLIT_KV_MERGE_O_NVFP4_ACC16_M64N64K128_DB_SSPM||k==KernelId::PI05_SPLIT_KV_MERGE_ROT_OV_O_NVFP4_ACC16_M64N64K128_DB_SSPM;}
KernelId autotile_nvfp4_acc16_kernel_id(int n){assert(n>=32&&n<=128&&n%16==0);return KernelId((128-n)/16);}
// The REF carries the double-buffered M320N64 sibling unless a case removes it.
bool linear_db=true;
bool pi05_nvfp4_acc16_linear_double_buffered(int64_t rows){return rows<=64&&linear_db;}
enum class GraphDdrRegisterRole{PackedFp4ModelWeight,Fp8BlockScale};
enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct Abi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{Abi abi;at::Tensor owner;};
struct Kernel{std::map<int,uint16_t> regs;void reset_regs(){regs.clear();}void set_regs(int r,uint16_t v){regs[r]=v;}
 // Only the atomic counter pairs take 32-bit writes.
 void set_regs(int r,uint32_t v){assert(r==128||r==130);regs[r]=uint16_t(v);regs[r+1]=uint16_t(v>>16);}};
struct Writer{void write(Kernel&);};
struct RpuKernelGraph{std::vector<GraphDdrRegisterOperandSpec> owners;Kernel kernel;KernelId id{};int stages=0;bool pending=false,written=false;
 static RpuKernelGraph&active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId k,std::vector<GraphDdrRegisterOperandSpec> values){assert(!pending&&!written&&values.size()==(spm_scaled_id(k)?1u:2u));assert(values[0].abi.role==GraphDdrRegisterRole::PackedFp4ModelWeight&&values[0].abi.lo==2&&values[0].abi.hi==3);assert(values.size()==1||(values[1].abi.role==GraphDdrRegisterRole::Fp8BlockScale&&values[1].abi.lo==20&&values[1].abi.hi==21));id=k;owners=values;++stages;return {};}
 Kernel*get(KernelId k){assert(k==id&&!pending);pending=true;return &kernel;}};
void Writer::write(Kernel&k){auto&g=RpuKernelGraph::active();assert(g.pending&&!g.written);for(auto&o:g.owners){auto value=o.owner.address>>8;k.set_regs(o.abi.lo,uint16_t(value));k.set_regs(o.abi.hi,uint16_t(value>>16));}g.written=true;}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
struct Queue{bool broadcast=false;int launches=0,lanes=8;std::vector<uint16_t> grid;
 void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> gr,std::vector<uint8_t> cores){auto&g=RpuKernelGraph::active();std::vector<uint8_t> all;for(int c=0;c<lanes;++c)all.push_back(c);assert(g.pending&&g.written&&broadcast&&cores==all);grid=gr;++launches;g.pending=g.written=false;}}queue;
Queue*get_queue(int n){assert(n==4||n==6||n==8);queue.lanes=n;return &queue;}
#define GET_QUEUE(n) get_queue(n)
struct Pi05SplitKvMerge;
// A deferred merge launched alone records the O launch count before it.
std::vector<int> merges_before;
void rpu_launch_pi05_split_kv_merge(const Pi05SplitKvMerge&){merges_before.push_back(queue.launches);}
uint16_t pi05_split_kv_write_merge_o_regs(Kernel*,const Pi05SplitKvMerge&);
__FUNCTIONS__
uint16_t pi05_split_kv_write_merge_o_regs(Kernel*k,const Pi05SplitKvMerge&m){
 const auto grid=uint16_t((m.plan.seq_q+15)/16);k->set_regs(35,grid);k->set_regs(28,uint16_t(m.plan.seq_q));return grid;}
template<class F>void reject(F f){auto&g=RpuKernelGraph::active();const int launches=queue.launches,stages=g.stages,fl=flushes;try{f();assert(false);}catch(const std::runtime_error&){}assert(queue.launches==launches&&g.stages==stages&&flushes==fl&&!g.pending);}
uint32_t pair(int r){auto&v=RpuKernelGraph::active().kernel.regs;return uint32_t(v.at(r))|(uint32_t(v.at(r+1))<<16);}
int main(){
 constexpr uint32_t x=0x20010000,y=0x20030000,ts=0x20050000;
 int tscale_launches=0,merge_o_launches=0;
 auto&g=RpuKernelGraph::active();
 for(auto geometry:{std::array<int,3>{2048,1024,1},{4096,1024,1},{1024,2048,0},{1024,4096,0}}){
  auto n=geometry[0],k=geometry[1],part=geometry[2];at::Tensor w,s;s.address=0x3456789a00ULL;
  w.shape={n,k/2};s.shape={k/16,n};int ln=part?n/8:n,lk=part?k:k/8;
  for(int layer:{0,17,18,23}){
   rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,n,k,part,8,s,ts,layer,false);
   auto tile=rpu_pl_tiling::select_tile_nvfp4_acc16(32,ln,lk);
   assert(int(g.id)==(128-tile.n_tile)/16);
   assert(pair(0)==x&&pair(2)==w.address>>8&&pair(4)==y&&pair(6)==0&&pair(20)==s.address>>8&&pair(24)==ts);
   assert(g.kernel.regs.at(8)==32&&pair(10)==unsigned(ln)&&g.kernel.regs.at(12)==lk&&g.kernel.regs.at(13)==lk/64&&g.kernel.regs.at(27)==layer);
   assert(pair(22)==unsigned(8*ln*2)&&queue.grid==std::vector<uint16_t>({uint16_t((ln+tile.n_tile-1)/tile.n_tile),1,1}));
   rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,n,k,part,8,s,ts,layer,true);
   const bool gate_up=part==1&&n==4096;const int nt=gate_up?48:64;
   assert(g.id==(gate_up?KernelId::PARALLEL_LINEAR_NVFP4_V2_M384N48:KernelId::PARALLEL_LINEAR_NVFP4_V2_M320N64));
   assert(pair(2)==w.address>>8&&pair(20)==s.address>>8&&pair(24)==ts&&g.kernel.regs.at(27)==layer);
   assert(queue.grid==std::vector<uint16_t>({uint16_t((ln+nt-1)/nt),1,1}));
  }
  reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,y,50,n,k,part,8,s,ts,0,false);});
  reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,n,k,part,8,s,ts,24,false);});
  reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,n,k,part,6,s,ts,0,false);});
  reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,x,32,n,k,part,8,s,ts,0,false);});
  reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,n,k,part,8,s,x,0,false);});
  reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x+1,w,y,32,n,k,part,8,s,ts,0,false);});
  reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,0x207fff00,32,n,k,part,8,s,ts,0,false);});
  for(int fault=0;fault<5;++fault){auto bad=s;if(fault==0)bad.dtype=at::kHalf;if(fault==1)bad.contig=false;if(fault==2)--bad.shape[0];if(fault==3)bad.dev.value=at::kPrivateUse1Type::CPU;if(fault==4)bad.dev.index=1;
   reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,n,k,part,8,bad,ts,0,false);});}
  // G05 above never took the sibling although it was loaded; Pi ACC16 takes it
  // only on the M320N64 tile and only when the REF carries it. A tile-scale
  // companion moves that sibling to the tscale payload and its own scale
  // address; every other route keeps reading the striped owner.
  at::Tensor tiled=s;tiled.present=true;tiled.address=0x3456000000ULL;
  for(bool db:{true,false})for(int rows:{10,50})for(bool companion:{false,true}){
   linear_db=db;
   const at::Tensor none;
   const at::Tensor&scale_companion=companion?tiled:none;
   rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,rows,n,k,part,8,s,ts,17,false,scale_companion);
   auto tile=rpu_pl_tiling::select_tile_nvfp4_acc16(rows,ln,lk);
   const bool sibling=db&&tile.n_tile==64;tscale_launches+=sibling&&companion;
   assert(g.id==(sibling?(companion?KernelId::PI05_NVFP4_LINEAR_DB_TSCALE:KernelId::PI05_NVFP4_LINEAR_DB)
                        :autotile_nvfp4_acc16_kernel_id(tile.n_tile)));
   assert(pair(20)==(sibling&&companion?tiled.address:s.address)>>8&&pair(2)==w.address>>8);
   assert(g.kernel.regs.at(8)==rows&&queue.grid==std::vector<uint16_t>({uint16_t((ln+tile.n_tile-1)/tile.n_tile),1,1}));
   rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,rows,n,k,part,8,s,ts,17,true,scale_companion);
   assert(g.id==KernelId::PARALLEL_LINEAR_NVFP4_V2_M384N48||g.id==KernelId::PARALLEL_LINEAR_NVFP4_V2_M320N64);
   assert(pair(20)==s.address>>8);
   if(sibling&&companion)for(int fault=0;fault<4;++fault){auto bad=tiled;if(fault==0)bad.dtype=at::kHalf;if(fault==1)bad.contig=false;if(fault==2)--bad.shape[0];if(fault==3)bad.dev.index=1;
    reject([&]{rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,rows,n,k,part,8,s,ts,17,false,bad);});}
   // An SPM slab moves the tile-scale payload to its SPM-scale sibling: only the
   // weight stays a DDR operand and 20/21 hold the slab. Nothing else takes one.
   constexpr uint32_t slab=0x20060000;
   if(sibling&&companion){
    rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,rows,n,k,part,8,s,ts,17,false,tiled,slab);
    assert(g.id==KernelId::PI05_NVFP4_LINEAR_DB_SSPM&&g.owners.size()==1);
    assert(pair(2)==w.address>>8&&pair(20)==slab&&g.kernel.regs.at(8)==rows);
    for(uint32_t bad:{slab+1,0x1fffff00u,0x207ff000u})
     reject([&]{rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,rows,n,k,part,8,s,ts,17,false,tiled,bad);});
   }else reject([&]{rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,rows,n,k,part,8,s,ts,17,false,scale_companion,slab);});
   reject([&]{rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,rows,n,k,part,8,s,ts,17,true,scale_companion,slab);});
  }
  linear_db=true;
  reject([&]{rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,32,n,k,part,8,s,ts,18,false);});
  // A deferred split-KV merge (its output is the input rows): the one-row-block
  // SPM-scale O takes the merge + O sibling, the merge's blocks first; every
  // other route launches the merge alone first and is otherwise unchanged.
  if(part==0){
   constexpr uint32_t in=0x20000000,slab=0x20060000;
   at::Tensor tiled=s;tiled.present=true;tiled.address=0x3456000000ULL;
   for(int rows:{10,16,50,64})for(bool rot_ov:{false,true})for(int route:{0,1,2}){
    Pi05SplitKvMerge m{};m.plan.seq_q=rows;m.rot_ov=rot_ov;
    const at::Tensor none;
    const auto launch=[&](const Pi05SplitKvMerge*merge){
     if(route==0)rpu_launch_pi05_nvfp4_v2_kernel(in,w,y,rows,n,k,part,8,s,ts,3,false,tiled,slab,merge);
     else rpu_launch_pi05_nvfp4_v2_kernel(in,w,y,rows,n,k,part,8,s,ts,3,route==2,route==2?tiled:none,0,merge);};
    launch(nullptr);
    const KernelId plain=g.id;const auto regs=g.kernel.regs;const auto grid=queue.grid;
    assert(!regs.count(128)&&!regs.count(28)&&!regs.count(35));
    merges_before.clear();const int launches=queue.launches;
    launch(&m);
    if(plain==KernelId::PI05_NVFP4_LINEAR_DB_SSPM){
     const uint16_t mg=uint16_t((rows+15)/16);
     assert(route==0&&merges_before.empty());++merge_o_launches;
     assert(g.id==(rot_ov?KernelId::PI05_SPLIT_KV_MERGE_ROT_OV_O_NVFP4_ACC16_M64N64K128_DB_SSPM
                         :KernelId::PI05_SPLIT_KV_MERGE_O_NVFP4_ACC16_M64N64K128_DB_SSPM));
     assert(queue.grid==std::vector<uint16_t>({uint16_t(mg+16),1,1})&&grid==std::vector<uint16_t>({16,1,1}));
     assert(g.kernel.regs.at(28)==rows&&g.kernel.regs.at(35)==mg&&pair(128)==0&&pair(130)==0);
     // Besides the merge's registers and the counters only the grid moves.
     auto fused=g.kernel.regs,expect=regs;expect[64]=uint16_t(mg+16);
     for(int r:{28,35,128,129,130,131})fused.erase(r);
     assert(fused==expect);
    }else{
     assert(plain!=KernelId::PI05_NVFP4_LINEAR_DB_SSPM&&g.id==plain&&g.kernel.regs==regs&&queue.grid==grid);
     assert(merges_before==std::vector<int>({launches}));
    }
    assert(queue.launches==launches+1);
    Pi05SplitKvMerge other=m;other.plan.seq_q=rows+1;
    reject([&]{rpu_launch_pi05_nvfp4_v2_kernel(in,w,y,rows,n,k,part,8,s,ts,3,false,tiled,slab,&other);});
    reject([&]{rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,rows,n,k,part,8,s,ts,3,false,tiled,slab,&m);});
   }
  }
 }
 assert(tscale_launches==7&&merge_o_launches>=12);
 // Pi reduced lanes: attention TP4, MLP 4 or the six-lane width 4224.
 for(int cores:{4,6}){
  const int mlp=cores==6?4224:4096;
  for(auto geometry:{std::array<int,4>{2048,1024,1,4},{mlp,1024,1,cores},{1024,2048,0,4},{1024,mlp,0,cores}}){
   auto n=geometry[0],k=geometry[1],part=geometry[2],lanes=geometry[3];at::Tensor w,s;
   w.shape={n,k/2};s.shape={k/16,n};int ln=part?n/lanes:n,lk=part?k:k/lanes;
   for(bool acc32:{false,true}){
    rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,50,n,k,part,lanes,s,ts,17,acc32);
    const bool gate_up=part==1&&n==mlp;const int nt=acc32?(gate_up?48:64):rpu_pl_tiling::select_tile_nvfp4_acc16(50,ln,lk).n_tile;
    assert(!acc32||g.id==(gate_up?KernelId::PARALLEL_LINEAR_NVFP4_V2_M384N48:KernelId::PARALLEL_LINEAR_NVFP4_V2_M320N64));
    assert(pair(10)==unsigned(ln)&&g.kernel.regs.at(12)==lk&&g.kernel.regs.at(13)==lk/64&&g.kernel.regs.at(15)==lanes);
    assert(pair(22)==unsigned(lanes*ln*2)&&queue.lanes==lanes&&queue.grid[0]==uint16_t((ln+nt-1)/nt));
   }
  }
  // G05 splits its exact action families over four lanes; six lanes (and the
  // padded MLP width) stay refused. Pi rejects other lane counts and unpadded six-lane MLP.
  if(cores==4)for(auto geometry:{std::array<int,3>{2048,1024,1},{4096,1024,1},{1024,2048,0},{1024,4096,0}}){
   auto n=geometry[0],k=geometry[1],part=geometry[2];at::Tensor gw,gs;gw.shape={n,k/2};gs.shape={k/16,n};int ln=part?n/4:n,lk=part?k:k/4;
   rpu_launch_g05_nvfp4_v2_kernel(x,gw,y,32,n,k,part,4,gs,ts,23,false);
   assert(pair(10)==unsigned(ln)&&g.kernel.regs.at(12)==lk&&g.kernel.regs.at(15)==4&&g.kernel.regs.at(27)==23&&queue.lanes==4);
  }
  at::Tensor w,s;w.shape={mlp,512};s.shape={64,mlp};
  reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,mlp,1024,1,6,s,ts,0,false);});
  reject([&]{rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,50,mlp,1024,1,5,s,ts,0,false);});
  reject([&]{rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,50,mlp,1024,1,cores==6?4:6,s,ts,0,false);});
 }
 {at::Tensor w,s;w.shape={4096,512};s.shape={64,4096};
  reject([&]{rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,50,4096,1024,1,6,s,ts,0,false);});
  w.shape={1024,2048};s.shape={256,1024};
  reject([&]{rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,50,1024,4096,0,6,s,ts,0,false);});
  w.shape={2048,512};s.shape={64,2048};
  reject([&]{rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,50,2048,1024,1,6,s,ts,0,false);});}
 at::Tensor w,s;s.shape={64,2048};reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,1024,1024,1,8,s,ts,0,false);});
 allocator.initialized=false;reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,2048,1024,1,8,s,ts,0,false);});
}
'''
    _compile_run(tmp_path, 'g05_nvfp4_native', cpp.replace('__FUNCTIONS__', '\n'.join(functions)))
