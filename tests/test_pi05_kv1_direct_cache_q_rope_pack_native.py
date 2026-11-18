"""Execute the actual KV1 direct-cache launcher with and without the fused query RoPE pack."""
from pathlib import Path
import re
from test_siglip_cold_profile import _body
from test_g05_q_gate_native import _compile_run

ROOT = Path(__file__).resolve().parents[1]


def test_fused_rope_pack_keeps_both_register_images_and_rejects_hazards(tmp_path):
    sdpa = (ROOT / 'src/ops/rpu_sdpa.cpp').read_text()
    kvcache = (ROOT / 'src/ops/rpu_llama_kvcache.cpp').read_text()
    model = (ROOT / 'src/fused/rpu_pi05_denoise_step_model.cpp').read_text()
    decls = (ROOT / 'src/core/rpu_kernel_decls.h').read_text()

    def definition(text, marker):
        start = text.index(marker)
        return text[start:text.index('{', start)] + _body(text[start:], marker)

    names = ('kSplitKvGroup', 'kSplitKvHeadDim', 'kSplitKvPackRows', 'kSplitKvRopeWarpRows',
             'kSplitKvRopeMaxRows', 'kSplitKvRopeRowBytes', 'kSplitKvRopeGridX')
    constants = [re.search(rf'constexpr int64_t {n} = [^;]*;', sdpa)[0] for n in names]
    functions = [definition(decls, 'struct Pi05SplitKvPlan {') + ';',
                 definition(decls, 'struct Pi05SplitKvRopePack {') + ';']
    for marker in ('int64_t Pi05SplitKvPlan::qp_bytes(', 'bool pi05_split_kv_rope_pack(',
                   'void split_kv_set_scm(', 'void split_kv_check_rope_pack(',
                   'void split_kv_write_rope_pack_regs(', 'void pi05_split_kv_check_rope_pack_spm(',
                   'uint16_t pi05_split_kv_write_rope_pack_spm_regs('):
        functions.append(definition(sdpa, marker))
    for marker in ('bool pi05_kv1_direct_cache_overlapped(', 'bool pi05_kv1_direct_cache_rotated(',
                   'bool pi05_kv1_direct_cache_q_rope_pack(', 'bool pi05_kv1_direct_cache_spm_rope('):
        functions.append(definition(model, marker))
    functions.append(definition(kvcache, 'void rpu_launch_pi05_kv1_direct_cache_m50d256p64_kernel('))
    cpp = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
template<class...A>void message(const A&...){}
#define TORCH_CHECK(ok,...) do{if(!(ok)){message(__VA_ARGS__);throw std::runtime_error("rejected");}}while(0)
#define MAX_CORES 8
#define NUM_CORES 8
#define SCM_REG_OFFSET 4096
#define SCM_PARAMS_PER_CORE 8
template<class T>T CeilDiv(T a,T b){return (a+b-1)/b;}
namespace c10 {using Half=uint16_t; enum class DeviceType{PrivateUse1,CPU};}
namespace at {
constexpr int kHalf=2;
using IntArrayRef=std::vector<int64_t>;
struct Device{c10::DeviceType value=c10::DeviceType::PrivateUse1;int index=0;
 c10::DeviceType type()const{return value;}bool operator==(Device b)const{return value==b.value&&index==b.index;}};
struct Tensor{int dtype=kHalf;bool present=false;Device dev{};IntArrayRef shape;uintptr_t address=0;
 bool defined()const{return present;}int scalar_type()const{return dtype;}Device device()const{return dev;}
 bool is_contiguous()const{return true;}IntArrayRef sizes()const{return shape;}int64_t dim()const{return shape.size();}
 int64_t size(int i)const{return shape[i];}int64_t storage_offset()const{return 0;}
 size_t nbytes()const{size_t n=2;for(auto s:shape)n*=s;return n;}
 template<class T>T*data_ptr()const{return reinterpret_cast<T*>(address);}};
}
// Core c's SPM window; the launchers take core 0's absolute addresses.
struct SpmAllocator{static constexpr size_t SPM_USABLE=8387584;
 bool is_initialized()const{return true;}
 uint64_t addr(int core,uint32_t offset)const{return 0x20000000ULL+uint64_t(core)*0x800000+offset;}}allocator;
#define SPM_ALLOC allocator
int flushes=0;
void rpu_ddr_flush(void*){++flushes;}
uint64_t RpuGetDevAddr(void*p){return reinterpret_cast<uintptr_t>(p);}
enum class KernelId{PI05_SPLIT_KV_PACK_Q_ROPE,PI05_DENOISE_KV1_DIRECT_CACHE_M10D256P16,
 PI05_DENOISE_KV1_DIRECT_CACHE_M50D256P64,PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_OVERLAP,
 PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT,PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64,
 PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_SPM_ROPE_M64};
// The REF carries every payload unless a case removes one, except the
// SPM-RoPE sibling, which the last cases add.
std::map<KernelId,bool> loaded{{KernelId::PI05_SPLIT_KV_PACK_Q_ROPE,true},
 {KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_OVERLAP,true},
 {KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT,true},
 {KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64,true},
 {KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_SPM_ROPE_M64,false}};
struct KernelCache{static KernelCache&instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId id)const{auto it=loaded.find(id);return it!=loaded.end()&&it->second;}};
bool pi05_split_kv_rotated(int64_t rows){return rows>16&&loaded[KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT];}
enum class KvInsertRoute{PAD16_V16,PAD16_V8};enum class KvInsertKernel{V16,V8};
struct KvInsertSegment{KvInsertKernel kernel;int64_t position;int64_t token_offset;int64_t rows;};
struct KvInsertSegmentPlan{KvInsertRoute r;int64_t logical,physical;std::vector<KvInsertSegment> segments;
 KvInsertRoute route()const{return r;}int64_t logical_rows()const{return logical;}
 int64_t physical_rows()const{return physical;}size_t segment_count()const{return segments.size();}
 const KvInsertSegment&segment(size_t i)const{return segments[i];}};
namespace rhino_lkn{
constexpr int kKernelRegOk=0;
struct Kernel_t{std::map<uint32_t,uint16_t> regs;std::map<uint32_t,uint32_t> scm;
 void reset_regs(){regs.clear();scm.clear();}
 int set_regs(uint32_t r,uint16_t v){assert(r<68);regs[r]=v;return 0;}int set_regs(uint32_t,uint32_t)=delete;};
}
using rhino_lkn::Kernel_t;using rhino_lkn::kKernelRegOk;
void rpu_set_scm_u32_checked(Kernel_t*k,uint32_t r,uint32_t v,const char*){
 assert(r>=SCM_REG_OFFSET&&(r-SCM_REG_OFFSET)%2==0);k->scm[r]=v;}
enum class GraphDdrRegisterRole{KeyCache,ValueCache,SemanticInput};
enum class GraphDdrRegisterAccess{Read,Write};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;
 uint32_t reg_lo;uint32_t reg_hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Writer{void write(Kernel_t&);};
struct RpuKernelGraph{std::vector<GraphDdrRegisterOperandSpec> owners;Kernel_t kernel;KernelId id{};int stages=0;
 static bool has_active(){return true;}
 static RpuKernelGraph&active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId k,std::vector<GraphDdrRegisterOperandSpec> values){
  id=k;owners=std::move(values);++stages;return {};}
 void keep_alive(const at::Tensor&){}
 Kernel_t*get(KernelId k){assert(k==id);return loaded[k]?&kernel:nullptr;}};
void Writer::write(Kernel_t&k){for(auto&o:RpuKernelGraph::active().owners){auto value=RpuGetDevAddr(o.tensor.data_ptr<void>())>>8;
 k.set_regs(o.abi.reg_lo,uint16_t(value));k.set_regs(o.abi.reg_hi,uint16_t(value>>16));}}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
struct Queue{bool broadcast=false;int launches=0;std::vector<uint16_t> grid;
 void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel_t&,std::vector<uint16_t> gr,std::vector<uint8_t> cores){assert(broadcast&&cores.size()==8);grid=gr;++launches;}}queue;
Queue*get_queue(int n){assert(n==8);return &queue;}
#define GET_QUEUE(n) get_queue(n)
__CONSTANTS__
__FUNCTIONS__
at::Tensor tensor(at::IntArrayRef shape,uintptr_t address){
 at::Tensor t;t.present=true;t.shape=shape;t.address=address;return t;}
template<class F>void reject(F f){auto&g=RpuKernelGraph::active();
 const int launches=queue.launches,fl=flushes,stages=g.stages;try{f();assert(false);}catch(const std::runtime_error&){}
 assert(queue.launches==launches&&flushes==fl&&g.stages==stages);}
uint32_t u32(const std::map<uint32_t,uint16_t>&r,uint32_t lo){return uint32_t(r.at(lo))|(uint32_t(r.at(lo+1))<<16);}
int main(){
 auto&g=RpuKernelGraph::active();
 const at::IntArrayRef cache_shape{1,128,1,16,8,16,16};
 const auto kc=tensor(cache_shape,0x100000000ULL),vc=tensor(cache_shape,0x101000000ULL);
 const auto cs=tensor({2048,128},0x102000000ULL),sn=tensor({2048,128},0x102100000ULL);
 constexpr uint32_t k=0x30000,v=0x38000,q=0x40000,qp=0x50000,rope=0x70000,spm=0x20000000;
 const KvInsertSegmentPlan h50{KvInsertRoute::PAD16_V16,50,64,{{KvInsertKernel::V16,544,0,64}}};
 const KvInsertSegmentPlan h10{KvInsertRoute::PAD16_V16,10,16,{{KvInsertKernel::V16,544,0,16}}};
 Pi05SplitKvPlan plan;plan.seq_q=50;plan.group=4;plan.rows=200;plan.rows_pad=208;
 const auto launch=[&](const KvInsertSegmentPlan&p,uint16_t mask,const Pi05SplitKvRopePack*pack){
  rpu_launch_pi05_kv1_direct_cache_m50d256p64_kernel(k,v,kc,vc,cs,sn,531,p,mask,pack);};
 // A REF without the fused payload: the rotated writer, registers 0..63, grid 4.
 loaded[KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64]=false;
 launch(h50,0x77,nullptr);
 const auto rot=g.kernel.regs;
 assert(g.id==KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT&&queue.grid==std::vector<uint16_t>({4,1,1}));
 assert(u32(rot,0)==spm+k&&u32(rot,2)==spm+v&&u32(rot,4)==kc.address>>8&&u32(rot,10)==sn.address>>8);
 assert(u32(rot,12)==531&&u32(rot,14)==544&&rot.at(16)==50&&rot.at(25)==0x77&&rot.at(28)==0&&rot.at(64)==4);
 loaded[KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64]=true;
 // The fused payload without the pack: the same image 32 higher, its block
 // count at 28, the pack's registers zero and no pack blocks.
 launch(h50,0x77,nullptr);
 const auto alone=g.kernel.regs;
 assert(g.id==KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64&&queue.grid==std::vector<uint16_t>({4,1,1}));
 assert(g.owners.size()==4&&g.owners[0].abi.reg_lo==36&&g.owners[3].abi.reg_hi==43);
 for(const auto&[reg,value]:rot)if(reg<32)assert(alone.at(reg+32)==value);
 for(uint32_t reg=0;reg<32;++reg)assert(reg==28?alone.at(28)==4:!alone.count(reg));
 assert(alone.at(64)==4&&g.kernel.scm.empty());
 // With the pack: its SPM-row image in 0..14 and SCM slot 0, and two blocks
 // per 16 rows after the writer's four.
 const Pi05SplitKvRopePack pack{q,qp,rope,plan};
 launch(h50,0x77,&pack);
 const auto&r=g.kernel.regs;
 assert(queue.grid==std::vector<uint16_t>({12,1,1})&&r.at(64)==12&&r.at(65)==1&&r.at(66)==1);
 assert(u32(r,0)==spm+rope&&u32(r,2)==spm+rope+50*256&&!r.count(4)&&!r.count(5));
 assert(u32(r,6)==spm+qp&&u32(r,8)==0&&r.at(10)==256&&r.at(11)==4&&r.at(12)==50&&r.at(13)==256&&r.at(14)==0);
 assert(r.at(28)==4);
 for(uint32_t reg=32;reg<68;++reg)if(reg<64)assert(r.at(reg)==alone.at(reg));
 for(int c=0;c<8;++c){const uint32_t base=SCM_REG_OFFSET+c*SCM_PARAMS_PER_CORE;
  assert(g.kernel.scm.at(base)==(allocator.addr(c,q)>>5));
  for(uint32_t slot=2;slot<8;slot+=2)assert(g.kernel.scm.at(base+slot)==0);}
 // M10 has no rotated reader: the overlapped writer's image 32 higher with
 // mask 0, grid 1, and with the pack two blocks after the writer's one. It
 // needs neither the rotated nor the standalone RoPE pack payload.
 loaded[KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64]=false;
 launch(h10,0,nullptr);
 assert(g.id==KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_OVERLAP&&queue.grid==std::vector<uint16_t>({1,1,1}));
 const auto overlap10=g.kernel.regs;
 assert(overlap10.at(16)==10&&overlap10.at(17)==16&&overlap10.at(25)==0&&overlap10.at(64)==1);
 Pi05SplitKvPlan plan10=plan;plan10.seq_q=10;plan10.rows=40;plan10.rows_pad=48;
 const Pi05SplitKvRopePack pack10{q,qp,rope,plan10};
 // Without the fused payload M10 rejects a pack.
 reject([&]{launch(h10,0,&pack10);});
 loaded[KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64]=true;
 for(auto id:{KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT,KernelId::PI05_SPLIT_KV_PACK_Q_ROPE})loaded[id]=false;
 launch(h10,0,nullptr);
 const auto alone10=g.kernel.regs;
 assert(g.id==KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64&&queue.grid==std::vector<uint16_t>({1,1,1}));
 for(const auto&[reg,value]:overlap10)if(reg<32)assert(alone10.at(reg+32)==value);
 for(uint32_t reg=0;reg<32;++reg)assert(reg==28?alone10.at(28)==1:!alone10.count(reg));
 assert(alone10.at(64)==1&&g.kernel.scm.empty());
 launch(h10,0,&pack10);
 assert(queue.grid==std::vector<uint16_t>({3,1,1})&&r.at(64)==3&&r.at(28)==1);
 assert(u32(r,0)==spm+rope&&u32(r,2)==spm+rope+10*256&&r.at(12)==10&&r.at(14)==0);
 for(uint32_t reg=32;reg<64;++reg)assert(r.at(reg)==alone10.at(reg));
 for(auto id:{KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT,KernelId::PI05_SPLIT_KV_PACK_Q_ROPE})loaded[id]=true;
 // A REF without the fused or the RoPE pack payload, another row count, no
 // or misaligned SPM rows, and packed rows over the compact K or V rows all
 // fail before any staging, flush or launch.
 for(auto id:{KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64,KernelId::PI05_SPLIT_KV_PACK_Q_ROPE}){
  loaded[id]=false;reject([&]{launch(h50,0x77,&pack);});loaded[id]=true;}
 Pi05SplitKvPlan plan49=plan;plan49.seq_q=49;
 const Pi05SplitKvRopePack bad[]={{q,qp,rope,plan49},{q,qp,-1,plan},{q,qp,rope+16,plan},
  {q,k-0x1000,rope,plan},{q,v+0x800,rope,plan}};
 for(const auto&p:bad)reject([&]{launch(h50,0x77,&p);});
 // The packed rows may end right where the K rows begin.
 const Pi05SplitKvRopePack edge{q,k-208*512,rope,plan};
 launch(h50,0x77,&edge);
 assert(queue.grid==std::vector<uint16_t>({12,1,1}));
 // With the SPM-RoPE sibling, a launch that packs has its writer read the
 // pack's SPM rows: the same image except 40..45, the rows' absolute SPM
 // addresses and logical row 0, and the DDR tables are no typed operands.
 const auto spm_rope=KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_SPM_ROPE_M64;
 launch(h50,0x77,&pack);
 const auto tables=g.kernel.regs;const auto tables_scm=g.kernel.scm;
 assert(u32(tables,40)==cs.address>>8&&u32(tables,42)==sn.address>>8&&u32(tables,44)==531);
 loaded[spm_rope]=true;
 launch(h50,0x77,&pack);
 assert(g.id==spm_rope&&queue.grid==std::vector<uint16_t>({12,1,1}));
 assert(g.owners.size()==2&&g.owners[0].abi.reg_lo==36&&g.owners[1].abi.reg_hi==39);
 assert(u32(r,40)==spm+rope&&u32(r,42)==spm+rope+50*256&&u32(r,44)==0);
 assert(r.size()==tables.size()&&g.kernel.scm==tables_scm);
 for(const auto&[reg,value]:tables)if(reg<40||reg>45)assert(r.at(reg)==value);
 // A launch without a pack has no SPM rows and keeps the DDR tables.
 launch(h50,0x77,nullptr);
 assert(g.id==KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_ROT_Q_ROPE_PACK_M64&&g.owners.size()==4);
 assert(g.kernel.regs==alone);
 // M10 takes it with the pack's ten rows.
 for(auto id:{KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT,KernelId::PI05_SPLIT_KV_PACK_Q_ROPE})loaded[id]=false;
 launch(h10,0,&pack10);
 assert(g.id==spm_rope&&queue.grid==std::vector<uint16_t>({3,1,1})&&r.at(48)==10);
 assert(u32(r,40)==spm+rope&&u32(r,42)==spm+rope+10*256&&u32(r,44)==0);
 for(auto id:{KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M64D256_ROT,KernelId::PI05_SPLIT_KV_PACK_Q_ROPE})loaded[id]=true;
 for(const auto&p:bad)reject([&]{launch(h50,0x77,&p);});
}
'''
    cpp = cpp.replace('__CONSTANTS__', '\n'.join(constants)).replace('__FUNCTIONS__', '\n'.join(functions))
    _compile_run(tmp_path, 'pi05_kv1_direct_cache_q_rope_pack_native', cpp)
