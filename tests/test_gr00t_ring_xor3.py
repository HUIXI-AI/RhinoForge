"""Execute the exact GR00T XOR3 launcher against the original paced-ring ABI."""
from pathlib import Path
import re

from test_fmb_standalone_layout_rebuild import _definition
from test_gr00t_ff1_resident_w8 import _run

ROOT = Path(__file__).resolve().parents[1]


def test_ring_xor3_actual_registers_counters_and_rejected_submissions(tmp_path):
    source = (ROOT / "src/core/rpu_runtime_extras.cpp").read_text()
    helper = _definition(source, "void rpu_launch_gr00t_all_reduce_residual_xor3_kernel(")
    ring = "\n".join(_definition(source, marker) for marker in (
        "bool ring_all_reduce_sum_residual_admitted(",
        "void launch_ring_all_reduce_sum_residual(",
    ))
    epilogues = "\n".join(_definition(source, marker) + ";" for marker in (
        "struct RingAdaRmsOperands {", "struct G05RingRtcEpilogue {", "struct G05ActionTailNorm {",
    ))
    constants = "\n".join(re.findall(r"constexpr [^\n]+ kRing\w+ = [^\n]+;", source))
    scm_writer = _definition((ROOT / "src/core/rpu_helpers.h").read_text(),
                             "inline void rpu_set_legacy_scm_u16_checked(")
    ids = sorted(set(re.findall(r"KernelId::(\w+)", helper + ring)))
    decl_source = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    begin = decl_source.index("void rpu_launch_gr00t_all_reduce_residual_xor3_kernel(")
    declaration = decl_source[begin:decl_source.index(";", begin) + 1]
    cpp = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do{if(!(c))throw std::runtime_error(#c);}while(0)
#define TORCH_INTERNAL_ASSERT(c, ...) TORCH_CHECK(c)
#define SCM_REG_OFFSET 4096
namespace c10 {struct Half{uint16_t x;explicit Half(double v):x(v==1.0/1024?0x1400:0x11){}};}
enum class KernelId{__IDS__};
constexpr auto xor3_id=KernelId::GR00T_ALL_REDUCE_RESIDUAL_XOR3_M41N1536;
constexpr auto ring_id=KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_PACED;
struct SpmAllocator{
 static constexpr uint32_t SPM_USABLE=8387584;bool initialized=true;
 std::array<uint32_t,8>bases{0x10000000,0x11800000,0x12700000,0x13600000,0x14a00000,0x15d00000,0x16b00000,0x17f00000};
 bool is_initialized()const{return initialized;}
 uint32_t addr(int core,uint32_t offset)const{return bases.at(core)+offset;}
}SPM_ALLOC;
namespace v3{enum class FmbSharedAllReduceRouteSelector{RING_PI05_XOR3=1};
int64_t fmb_ring_all_reduce_route_selector(int64_t,int64_t,bool,int){return 0;}}
uint64_t RpuGetDevAddr(void*){throw std::runtime_error("unexpected DDR access");}
enum class GraphKernelNoDdrProof{RingAllReduceSpmResidual};
struct RpuKernelGraph{bool census=true,deny=false,pending=false;int stages=0;KernelId staged_id=xor3_id;
 static auto&active(){static RpuKernelGraph graph;return graph;}
 bool kernel_register_census_active()const{return census;}
 void stage_kernel_no_ddr(KernelId id,GraphKernelNoDdrProof proof){
  TORCH_CHECK(!deny&&!pending);assert(proof==GraphKernelNoDdrProof::RingAllReduceSpmResidual);
  assert(id==xor3_id||id==ring_id);staged_id=id;pending=true;++stages;}};
namespace rhino_lkn{
constexpr int kKernelRegOk=0;
struct Kernel_t{
 KernelId id;std::array<uint16_t,256>r{};std::array<uint16_t,64>scm{};
 std::array<bool,256>written{};int resets=0,scm_writes=0,fail_reg=-1;
 // Deliberately preserve dirty counters: reset_regs is not hardware initialization.
 int reset_regs(){std::fill(r.begin(),r.begin()+128,0xdead);written.fill(false);scm.fill(0xbeef);scm_writes=0;++resets;return kKernelRegOk;}
 int set_regs(uint32_t i,uint16_t value){if(int(i)==fail_reg)return 1;
  if(i>=SCM_REG_OFFSET){scm.at(i-SCM_REG_OFFSET)=value;++scm_writes;}
  else{r.at(i)=value;written.at(i)=true;}return kKernelRegOk;}
 uint32_t pair(int i)const{return r.at(i)|(uint32_t(r.at(i+1))<<16);}
 uint32_t scm_pair(int logical)const{return scm.at(logical^1)|(uint32_t(scm.at((logical+8)^1))<<16);}
};}
using namespace rhino_lkn;
Kernel_t candidate{xor3_id},original{ring_id};
bool missing=false;int lookups=0;
struct KernelCache{static KernelCache&instance(){static KernelCache cache;return cache;}
 Kernel_t*get(KernelId id)const{if(id==xor3_id)return missing?nullptr:&candidate;assert(id==ring_id);return &original;}};
Kernel_t*get_kernel(KernelId id){auto&g=RpuKernelGraph::active();++lookups;
 if(g.census)assert(g.pending&&g.staged_id==id);g.pending=false;
 if(id==xor3_id)return missing?nullptr:&candidate;
 assert(id==ring_id);return &original;}
#define GET_KERNEL(id) get_kernel(id)
__SCM_WRITER__
struct Queue{bool broadcast=false;int candidate_launches=0;
 void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel_t&k,std::initializer_list<uint16_t>grid,const std::vector<uint8_t>&cores){
  assert(broadcast&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));
  assert(std::vector<uint16_t>(grid)==std::vector<uint16_t>({8,1,1}));
  if(k.id==xor3_id){assert(k.scm_writes==64);
   for(int i=128;i<256;++i){assert(k.written[i]&&k.r[i]==0);k.r[i]=uint16_t(700+i);}
   for(int i=7;i<64;++i)assert(!k.written[i]);
   for(int i=67;i<128;++i)assert(!k.written[i]);++candidate_launches;}
 }}queue;
Queue*get_queue(int cores){assert(cores==8);queue.broadcast=false;return &queue;}
#define GET_QUEUE(n) get_queue(n)
__CONSTANTS__
__EPILOGUES__
__RING__
__DECLARATION__
__HELPER__
using Spm=std::array<uint32_t,3>;
void launch(const Spm&p){rpu_launch_gr00t_all_reduce_residual_xor3_kernel(p[0],p[1],p[2]);}
template<class F>void rejected(F call,bool before_lookup=true){
 int count=queue.candidate_launches,gets=lookups,resets=candidate.resets,stages=RpuKernelGraph::active().stages;
 bool caught=false;try{call();}catch(const std::runtime_error&){caught=true;}
 assert(caught&&queue.candidate_launches==count&&!RpuKernelGraph::active().pending);
 if(before_lookup)assert(lookups==gets&&candidate.resets==resets&&RpuKernelGraph::active().stages==stages);}
int main(){
 Spm spm{0x10001000,0x10022000,0x10043000};const auto good=spm;
 for(int step:{0,1,0}){spm=good;for(auto&address:spm)address+=step*0x100000;
  // The unchanged BASE launcher produces the reference GP and physical SCM image.
  launch_ring_all_reduce_sum_residual(spm[0],spm[1],nullptr,spm[2],41,1536,8);
  int stages=RpuKernelGraph::active().stages;launch(spm);
  assert(RpuKernelGraph::active().stages==stages+1);
  for(int i=0;i<=6;++i)assert(candidate.r[i]==original.r[i]&&candidate.written[i]);
  for(int i=64;i<=66;++i)assert(candidate.r[i]==original.r[i]);
  assert(candidate.scm==original.scm);
  assert(candidate.r[0]==8&&candidate.r[1]==1&&candidate.pair(2)==63488);
  assert(candidate.pair(4)==spm[1]-SPM_ALLOC.bases[0]&&candidate.r[6]==0);
  for(int core=0;core<8;++core){
   assert(candidate.scm_pair(core)==SPM_ALLOC.addr(core,spm[0]-SPM_ALLOC.bases[0])-SPM_ALLOC.bases[0]);
   assert(candidate.scm_pair(16+core)==SPM_ALLOC.addr(core,spm[2]-SPM_ALLOC.bases[0])-SPM_ALLOC.bases[0]);
   assert(candidate.scm_pair(32+core)==uint32_t(core)*15744&&candidate.scm_pair(48+core)==7872);}
  // Odd shards are 128-byte aligned, not 256-byte aligned; never pad their stride.
  assert(candidate.scm_pair(33)%256==128&&candidate.scm_pair(39)+7872*2==125952);
 }
 assert(queue.candidate_launches==3);
 constexpr uint32_t bytes=125952;
 for(int i=0;i<3;++i){for(auto wrong:{0u,0x10000000u-256,good[i]+1,good[i]+128,
     0x10000000u+SpmAllocator::SPM_USABLE-bytes+256,0xffffff00u}){
   spm=good;spm[i]=wrong;rejected([&]{launch(spm);});}
  for(int j=0;j<i;++j){spm=good;spm[i]=spm[j];rejected([&]{launch(spm);});
   spm=good;spm[i]=spm[j]+bytes-256;rejected([&]{launch(spm);});}}
 spm=good;SPM_ALLOC.initialized=false;rejected([&]{launch(spm);});SPM_ALLOC.initialized=true;
 auto&graph=RpuKernelGraph::active();graph.deny=true;rejected([&]{launch(spm);});graph.deny=false;
 missing=true;rejected([&]{launch(spm);},false);missing=false;
 for(int index:{0,128,255,4096}){candidate.fail_reg=index;rejected([&]{launch(spm);},false);}
 candidate.fail_reg=-1;
 // Immediate launches and exact adjacent/upper-boundary intervals remain legal.
 graph.census=false;spm={SPM_ALLOC.bases[0],SPM_ALLOC.bases[0]+bytes,
                       SPM_ALLOC.bases[0]+SpmAllocator::SPM_USABLE-bytes};
 int stages=graph.stages;launch(spm);assert(graph.stages==stages&&queue.candidate_launches==4);
}
'''
    for marker, value in {"__IDS__": ",".join(ids), "__CONSTANTS__": constants,
                          "__EPILOGUES__": epilogues,
                          "__SCM_WRITER__": scm_writer, "__RING__": ring,
                          "__DECLARATION__": declaration, "__HELPER__": helper}.items():
        cpp = cpp.replace(marker, value)
    _run(tmp_path, cpp)


def test_ring_xor3_append_only_optional_payload():
    cache = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    implementation = (ROOT / "src/core/rpu_kernel_cache.inc").read_text()
    kernel = "GR00T_ALL_REDUCE_RESIDUAL_XOR3_M41N1536"
    symbol = '"gr00t_all_reduce_residual_xor3_m41n1536"'
    assert cache.index(kernel) > cache.index("GR00T_QKV_MULTIBASE_W8A16_M592N32K128")
    assert symbol in cache
    assert symbol in implementation.split("kMixedNames[] = {", 1)[1].split("};", 1)[0]
    loader = _definition(implementation, "for (const char* name : kMixedNames)")
    absent = _definition(loader, "if (mixed_symbols.count(kname) == 0)")
    assert "continue;" in absent
    assert "load_kernel_from_oplib(" not in absent
    assert loader.index(absent) < loader.index("load_kernel_from_oplib(")
    assert "fast_cache_[i] = nullptr;" in _definition(implementation, "void KernelCache::populate_fast_cache()")
