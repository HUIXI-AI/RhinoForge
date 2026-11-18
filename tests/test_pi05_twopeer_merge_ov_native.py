"""Execute the actual Pi0.5 sibling picks: the two-peer and one-warp fused XOR3 AdaRMS and the overlapped rotated
split-KV merge."""
from pathlib import Path
from test_siglip_cold_profile import _body
from test_g05_q_gate_native import _compile_run

ROOT = Path(__file__).resolve().parents[1]

PRELUDE = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
template<class...A>void message(const A&...){}
#define TORCH_CHECK(ok,...) do{if(!(ok)){message(__VA_ARGS__);throw std::runtime_error("rejected");}}while(0)
#define TORCH_INTERNAL_ASSERT(ok) assert(ok)
namespace c10 {struct Half{uint16_t x;explicit Half(float v):x(uint16_t(std::lround(v*1e6))){}};}
enum class KernelId{PI05_XOR3_GATED_ADARMS_M50N1024,PI05_XOR3_GATED_ADARMS_TWOPEER_M50N1024,
 PI05_XOR3_GATED_ADARMS_ONEWARP_M50N1024,
 PI05_SPLIT_KV_MERGE,PI05_SPLIT_KV_MERGE_ROT,PI05_SPLIT_KV_MERGE_ROT_OV,
 PI05_SPLIT_KV_MERGE_O_NVFP4_ACC16_M64N64K128_DB_SSPM,PI05_SPLIT_KV_MERGE_ROT_OV_O_NVFP4_ACC16_M64N64K128_DB_SSPM};
// The REF carries every payload unless a case removes one.
std::map<KernelId,bool> loaded{{KernelId::PI05_XOR3_GATED_ADARMS_M50N1024,true},
 {KernelId::PI05_XOR3_GATED_ADARMS_TWOPEER_M50N1024,true},{KernelId::PI05_XOR3_GATED_ADARMS_ONEWARP_M50N1024,true},
 {KernelId::PI05_SPLIT_KV_MERGE,true},
 {KernelId::PI05_SPLIT_KV_MERGE_ROT,true},{KernelId::PI05_SPLIT_KV_MERGE_ROT_OV,true},
 {KernelId::PI05_SPLIT_KV_MERGE_O_NVFP4_ACC16_M64N64K128_DB_SSPM,true},
 {KernelId::PI05_SPLIT_KV_MERGE_ROT_OV_O_NVFP4_ACC16_M64N64K128_DB_SSPM,true}};
struct KernelCache{static KernelCache&instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId id)const{return loaded.at(id);}};
struct Kernel_t{std::map<uint32_t,uint16_t> regs;std::map<uint32_t,uint16_t> scm;
 void reset_regs(){regs.clear();}
 int set_regs(uint32_t r,uint16_t v){assert(r<256);regs[r]=v;return 0;}};
Kernel_t kernel;KernelId used{};
Kernel_t*get_kernel(KernelId id){used=id;return loaded.at(id)?&kernel:nullptr;}
#define GET_KERNEL(id) get_kernel(id)
'''


def test_twopeer_and_onewarp_fused_xor3_adarms_keep_the_register_image_and_scm_table(tmp_path):
    source = (ROOT / 'src/ops/rpu_pi05_producer_chain.cpp').read_text()
    helper = 'uint32_t whole_row_first_segment('
    marker = 'void rpu_launch_pi05_xor3_gated_adarms_spm_kernel('
    start = source.index(marker)
    first = source.index(helper)
    launcher = (source[first:source.index('{', first)] + _body(source, helper) + '\n' +
                source[start:source.index('{', start)] + _body(source[start:], marker))
    cpp = PRELUDE + r'''
constexpr int kKernelRegOk=0;
struct SpmAllocator{static constexpr uint64_t SPM_USABLE=8387584;
 bool is_initialized()const{return true;}
 uint32_t addr(int core,uint32_t offset)const{return 0x20000000U+uint32_t(core)*0x800000U+offset;}}allocator;
#define SPM_ALLOC allocator
enum class GraphKernelNoDdrProof{RingAllReduceSpmResidual};
struct RpuKernelGraph{static RpuKernelGraph&active(){static RpuKernelGraph g;return g;}
 bool kernel_register_census_active()const{return false;}
 void stage_kernel_no_ddr(KernelId,GraphKernelNoDdrProof){assert(false);}};
void rpu_set_legacy_scm_u16_checked(Kernel_t*k,uint32_t index,uint16_t value,const char*){k->scm[index]=value;}
struct Queue{bool broadcast=false;int launches=0;std::vector<uint16_t> grid;std::vector<uint8_t> cores;
 void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel_t&,std::vector<uint16_t> g,std::vector<uint8_t> c){assert(broadcast);grid=g;cores=c;++launches;}}queue;
Queue*get_queue(int){return &queue;}
#define GET_QUEUE(n) get_queue(n)
__LAUNCHER__
int main(){
 constexpr uint32_t step=0x20000;
 const auto launch=[&]{rpu_launch_pi05_xor3_gated_adarms_spm_kernel(
  0x20000000,0x20000000+step,0x20000000+2*step,0x20000000+3*step,0x20000000+4*step,0x20000000+5*step,
  0x20000000+6*step,0x20000000+7*step,50,1e-6);};
 const auto grid=[&]{return queue.grid==std::vector<uint16_t>({8,1,1})&&
  queue.cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7});};
 loaded[KernelId::PI05_XOR3_GATED_ADARMS_TWOPEER_M50N1024]=false;
 loaded[KernelId::PI05_XOR3_GATED_ADARMS_ONEWARP_M50N1024]=false;
 launch();
 assert(used==KernelId::PI05_XOR3_GATED_ADARMS_M50N1024&&grid());
 const auto regs=kernel.regs;const auto scm=kernel.scm;
 // The two-peer and one-warp loops need eight peers: p0 is the core count.
 assert(regs.at(0)==8&&regs.at(64)==8);
 // The M50 shards fit one warp's slots: first segments of at most 6,400 elements, the rest at most 1,536.
 for(uint32_t core=0;core<8;++core){
  const uint32_t size=scm.at(4096+48+core),first=scm.at(4096+64+core);
  assert(first<=6400&&size-first<=1536);
 }
 loaded[KernelId::PI05_XOR3_GATED_ADARMS_TWOPEER_M50N1024]=true;
 kernel.scm.clear();launch();
 assert(used==KernelId::PI05_XOR3_GATED_ADARMS_TWOPEER_M50N1024&&grid());
 assert(kernel.regs==regs&&kernel.scm==scm&&queue.launches==2);
 // The one-warp sibling is preferred, alone or with the two-peer one.
 for(bool twopeer:{true,false}){
  loaded[KernelId::PI05_XOR3_GATED_ADARMS_TWOPEER_M50N1024]=twopeer;
  loaded[KernelId::PI05_XOR3_GATED_ADARMS_ONEWARP_M50N1024]=true;
  kernel.scm.clear();launch();
  assert(used==KernelId::PI05_XOR3_GATED_ADARMS_ONEWARP_M50N1024&&grid());
  assert(kernel.regs==regs&&kernel.scm==scm);
 }
 assert(queue.launches==4);
}
'''
    _compile_run(tmp_path, 'pi05_xor3_twopeer_native', cpp.replace('__LAUNCHER__', launcher))


def test_overlapped_merge_only_above_sixteen_rows_and_falls_back_by_ref(tmp_path):
    source = (ROOT / 'src/ops/rpu_sdpa.cpp').read_text()
    marker = 'KernelId split_kv_merge_id(int64_t seq_q)'
    pick = marker + _body(source[source.index(marker):], marker)
    cpp = PRELUDE + r'''
__PICK__
KernelId merge_for(int64_t T){return split_kv_merge_id(T);}
int main(){
 for(int64_t T:{1,10,16})assert(merge_for(T)==KernelId::PI05_SPLIT_KV_MERGE);
 for(int64_t T:{17,50,127})assert(merge_for(T)==KernelId::PI05_SPLIT_KV_MERGE_ROT_OV);
 // A REF without the overlapped sibling keeps the rotated merge; without both, the published one.
 loaded[KernelId::PI05_SPLIT_KV_MERGE_ROT_OV]=false;
 assert(merge_for(50)==KernelId::PI05_SPLIT_KV_MERGE_ROT&&merge_for(10)==KernelId::PI05_SPLIT_KV_MERGE);
 loaded[KernelId::PI05_SPLIT_KV_MERGE_ROT]=false;
 assert(merge_for(50)==KernelId::PI05_SPLIT_KV_MERGE);
 // The overlapped sibling alone is still taken above 16 rows.
 loaded[KernelId::PI05_SPLIT_KV_MERGE_ROT_OV]=true;
 assert(merge_for(50)==KernelId::PI05_SPLIT_KV_MERGE_ROT_OV&&merge_for(16)==KernelId::PI05_SPLIT_KV_MERGE);
}
'''
    _compile_run(tmp_path, 'pi05_merge_ov_native', cpp.replace('__PICK__', pick))


def test_attention_leaves_its_merge_to_a_loaded_merge_o_sibling_only(tmp_path):
    source = (ROOT / 'src/ops/rpu_sdpa.cpp').read_text()
    marker = 'KernelId split_kv_merge_id(int64_t seq_q)'
    pick = marker + _body(source[source.index(marker):], marker)
    begin = source.index('  const KernelId merge_id = split_kv_merge_id(T);')
    step = source[begin:source.index('  return false;', begin) + len('  return false;')]
    cpp = PRELUDE + r'''
struct Pi05SplitKvPlan{int64_t seq_q=0;};
struct Pi05SplitKvMerge{Pi05SplitKvPlan plan;uint32_t part_off=0,output_off=0;bool rot_ov=false;};
std::vector<Pi05SplitKvMerge> launched;
void rpu_launch_pi05_split_kv_merge(const Pi05SplitKvMerge&m){launched.push_back(m);}
__PICK__
bool step(int64_t T,Pi05SplitKvMerge*defer_merge){
 const Pi05SplitKvPlan plan{T};const uint32_t part_off=0x1000,output_off=0x2000;
__STEP__
}
// Deferred or launched, the merge is the same one.
void same(const Pi05SplitKvMerge&m,int64_t T,bool rot_ov){
 assert(m.plan.seq_q==T&&m.part_off==0x1000&&m.output_off==0x2000&&m.rot_ov==rot_ov);}
int main(){
 using K=KernelId;
 for(int64_t T:{10,16,50})for(bool pointer:{false,true})for(int ref=0;ref<4;++ref){
  // ref: 0 everything, 1 no merge + O siblings, 2 no ROT_OV merge, 3 neither ROT_OV payload.
  loaded[K::PI05_SPLIT_KV_MERGE_ROT_OV]=ref<2;
  loaded[K::PI05_SPLIT_KV_MERGE_O_NVFP4_ACC16_M64N64K128_DB_SSPM]=ref!=1;
  loaded[K::PI05_SPLIT_KV_MERGE_ROT_OV_O_NVFP4_ACC16_M64N64K128_DB_SSPM]=ref==0||ref==2;
  const bool rot_ov=T>16&&ref<2;
  // The rotated merge (T > 16 without ROT_OV) has no merge + O sibling.
  const bool defer=pointer&&ref!=1&&(T<=16||ref==0);
  Pi05SplitKvMerge out{};launched.clear();
  assert(step(T,pointer?&out:nullptr)==defer);
  if(defer){assert(launched.empty());same(out,T,rot_ov);}
  else{assert(launched.size()==1);same(launched[0],T,rot_ov);if(pointer)assert(out.plan.seq_q==0);}
 }
}
'''
    _compile_run(tmp_path, 'pi05_merge_o_defer_native',
                 cpp.replace('__PICK__', pick).replace('__STEP__', step))
