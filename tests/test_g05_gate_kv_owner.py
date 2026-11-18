"""Actual G05 gate/KV owner-only launcher, independent typed owners and no fanout."""
from pathlib import Path
from test_g05_q_gate_native import _compile_run
from test_siglip_cold_profile import _body
ROOT = Path(__file__).resolve().parents[1]

def test_gate_kv_owner_only_native_typed_owners_exact_spans_and_no_fanout(tmp_path):
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    marker = "void rpu_launch_g05_gate_kv_owner_only_w8a16_acc16_kernel("
    start = source.index(marker)
    function = source[start:source.index("{",start)] + _body(source,marker)
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

#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10{using Half=uint16_t;enum class DeviceType{PrivateUse1,CPU};}
namespace at{constexpr int kHalf=1,kChar=2;
struct Tensor{uint64_t address;bool present=true,contiguous=true;int dtype=kChar,rank=2;int64_t rows=2048,cols=1024,offset=0;c10::DeviceType location=c10::DeviceType::PrivateUse1;
 bool defined()const{return present;}int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}int dim()const{return rank;}int64_t storage_offset()const{return offset;}int64_t size(int i)const{return i?cols:rows;}
 size_t nbytes()const{return size_t(rows)*(rank==2?cols:1)*(dtype==kHalf?2:1);}
 struct Device{c10::DeviceType value;c10::DeviceType type()const{return value;}bool operator==(Device b)const{return value==b.value;}};Device device()const{return {location};}void*data_ptr()const{return reinterpret_cast<void*>(address);}template<class T>T*data_ptr()const{return reinterpret_cast<T*>(address);}
};}
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8387584;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int core,int off)const{assert(core==0);return 0x20000000+off;}}allocator;
#define SPM_ALLOC allocator
uint64_t RpuGetDevAddr(void* pointer){return reinterpret_cast<uint64_t>(pointer);}
enum class KernelId{G05_GATE_KV_OWNER_W8A16_ACC16_M32N256K1024_N128};
constexpr auto COMBINED=KernelId::G05_GATE_KV_OWNER_W8A16_ACC16_M32N256K1024_N128;
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
  assert(!staged&&!pending&&id==COMBINED&&specs.size()==6);staged=id;written=false;operands=std::move(specs);++stages;return {};}
 Kernel*get(KernelId id){assert(!pending&&KernelCache::instance().has_loaded(id));
  assert(staged&&*staged==id);pending=id;written=false;++lookups;return &kernel;}
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
void launch(const std::array<at::Tensor,6>& t,const std::array<uint32_t,4>& p){
 rpu_launch_g05_gate_kv_owner_only_w8a16_acc16_kernel(p[0],t[0],t[1],t[2],t[3],t[4],t[5],p[1],p[2],p[3]);}
int main(){
 auto&g=RpuKernelGraph::active();const uint32_t base=allocator.addr(0,0);
 std::array<uint32_t,4>spm{base+0x10000,base+0x20000,base+0x24000,base+0x28000};
 std::array<at::Tensor,6>tensors{};
 for(size_t i=0;i<tensors.size();++i){tensors[i].address=0x1234567800ULL+i*0x200000;
  if(i>=3){tensors[i].dtype=at::kHalf;tensors[i].rank=1;tensors[i].rows=2048;}}
 const std::array<int,6>registers{2,32,40,20,34,42};
 // A second call changes every address, preventing cached first-call bindings.
 for(int pass=0;pass<2;++pass){for(size_t i=0;i<6;++i)tensors[i].address+=(i+1)*256;
  auto begin=g.submissions.size();auto flush_start=flushed.size();launch(tensors,spm);
  assert(g.submissions.size()==begin+1&&flushed.size()==flush_start+6&&!g.pending&&!g.staged);
  const auto&s=g.submissions[begin];assert(s.id==COMBINED&&s.grid==std::vector<uint16_t>({12,1,1})&&s.operands.size()==6);
  for(size_t i=0;i<6;++i){assert(pair(s,registers[i])==tensors[i].address>>8);
   assert(s.operands[i].owner.address==tensors[i].address&&s.operands[i].abi.lo==registers[i]&&s.operands[i].abi.hi==registers[i]+1);
   assert(s.operands[i].abi.role==(i<3?GraphDdrRegisterRole::Int8ModelWeight:GraphDdrRegisterRole::PerChannelScale));
   assert(s.operands[i].abi.access==GraphDdrRegisterAccess::Read&&s.operands[i].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
   assert(flushed[flush_start+i]==tensors[i].address);}
  assert(pair(s,0)==spm[0]&&pair(s,4)==spm[1]&&pair(s,36)==spm[2]&&pair(s,44)==spm[3]);
  assert(pair(s,6)==0&&s.regs.at(8)==1&&s.regs.at(9)==32&&pair(s,10)==256&&s.regs.at(12)==1024&&s.regs.at(13)==64&&s.regs.at(14)==0&&s.regs.at(15)==8);
  assert(pair(s,16)==2048&&pair(s,18)==512&&pair(s,22)==0&&pair(s,24)==0&&s.regs.at(26)==2&&s.regs.at(27)==1&&s.regs.at(28)==0&&s.regs.at(29)==0);
 }
 KernelCache::instance().missing.insert(COMBINED);reject([&]{launch(tensors,spm);});KernelCache::instance().missing.clear();
 allocator.initialized=false;reject([&]{launch(tensors,spm);});allocator.initialized=true;
 // Every owner is independently validated before any stage/flush/GET/enqueue.
 for(size_t index=0;index<6;++index)for(int fault=0;fault<12;++fault){auto bad=tensors;auto&t=bad[index];
  if(fault==0)t.present=false;
  if(fault==1)t.dtype=index<3?at::kHalf:at::kChar;
  if(fault==2)t.contiguous=false;
  if(fault==3)t.location=c10::DeviceType::CPU;
  if(fault==4)t.rank=index<3?1:2;
  if(fault==5)t.rows=2047;
  if(fault==6){if(index<3)t.cols=1023;else t.rows=2049;}
  if(fault==7)t.address=0;
  if(fault==8)t.address+=1;
  if(fault==9)t.address=UINT64_C(1)<<40;
  if(fault==10)t.address=(UINT64_C(1)<<40)-256;
  if(fault==11)t.offset=1;
  reject([&]{launch(bad,spm);});}
 // Independent DDR owner intervals must not overlap, including weights/scales.
 for(size_t i=0;i<6;++i)for(size_t j=0;j<i;++j){auto bad=tensors;bad[i].address=bad[j].address;reject([&]{launch(bad,spm);});}
 // Every SPM input/output pair must be disjoint.
 for(size_t i=0;i<4;++i)for(size_t j=0;j<i;++j){auto bad=spm;bad[i]=bad[j];reject([&]{launch(tensors,bad);});}
 for(size_t i=0;i<4;++i)for(int fault=0;fault<3;++fault){auto bad=spm;
  if(fault==0)bad[i]+=1;
  if(fault==1)bad[i]=base-256;
  if(fault==2)bad[i]=base+SpmAllocator::SPM_USABLE-256;
  reject([&]{launch(tensors,bad);});}
 // Exact endpoint and adjacent intervals are legal; no blanket over-rejection.
 auto edge=tensors;edge[5].address=(UINT64_C(1)<<40)-edge[5].nbytes();
 auto edge_spm=spm;edge_spm[3]=base+SpmAllocator::SPM_USABLE-(32*256*2);launch(edge,edge_spm);
}
'''
    _compile_run(tmp_path, "gate_kv_owner_launcher", cpp.replace("__FUNCTION__", function))


def test_gate_kv_owner_complete_cold_identity_and_caller_fallbacks(tmp_path):
    from test_g05_qkv_gate_native import _qkv_gate_model_cpp
    from test_g05_vision_owner_model import _block
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    cpp = '#include <array>\n' + _qkv_gate_model_cpp().split('int main(){', 1)[0]
    methods = [
        ('void prepare_action_gate_kv_owner()', 'void Qwen3_5Model::prepare_action_gate_kv_owner('),
        ('bool use_action_gate_kv_owner(int64_t rows,int64_t prefix)const', 'bool Qwen3_5Model::use_action_gate_kv_owner('),
        ('std::vector<int64_t>action_gate_kv_owner_arguments(int64_t prefix)const', 'std::vector<int64_t> Qwen3_5Model::action_gate_kv_owner_arguments('),
        ('void launch_action_gate_kv_owner(int layer,const ChunkInfo&chunk)', 'void Qwen3_5Model::launch_action_gate_kv_owner('),
    ]
    stubs = '''
 bool use_action_gate_kv_owner(int64_t,int64_t)const{return false;}
 std::vector<int64_t> action_gate_kv_owner_arguments(int64_t)const{return {};}
 void prepare_action_gate_kv_owner(){}
 template<class...T>void launch_action_gate_kv_owner(T...){assert(false);}
'''
    assert stubs in cpp
    cpp = cpp.replace(stubs, '\n'.join(d + _block(source, m) for d, m in methods))
    cpp = cpp.replace('bool use_action_owner_cache(int64_t,int64_t)const{return false;}',
                      'bool use_action_owner_cache(int64_t rows,int64_t max_prefix)const' +
                      _block(source, 'bool Qwen3_5Model::use_action_owner_cache('))
    cpp = cpp.replace('bool use_action_nvfp4(int64_t)const{return false;}',
                      'bool z2_bound_=false;bool use_action_nvfp4(int64_t rows)const' +
                      _block(source, 'bool Qwen3_5Model::use_action_nvfp4('))
    cpp = cpp.replace('bool enabled()const{return false;}', 'bool enabled()const{return !values.empty();}')
    cpp = cpp.replace('std::vector<int64_t>action_nvfp4_arguments(int)const{assert(false);return {};}', 'std::vector<int64_t>action_nvfp4_arguments(int family)const{return {family};}')
    cpp = cpp.replace('struct Device{c10::DeviceType value;',
                      'struct Device{bool operator!=(Device b)const{return value!=b.value||index!=b.index;}bool operator==(Device b)const{return !(*this!=b);}c10::DeviceType value;int index=0;')
    cpp = cpp.replace('Device device()const{return {location};}', 'int device_index=0;Device device()const{return {location,device_index};}')
    cpp = cpp.replace('int dim()const{return rank;}',
                      'int64_t offset=0;uint64_t address=0;int64_t storage_offset()const{return offset;}size_t nbytes()const{return size_t(rows)*(rank==2?cols:1)*(dtype==at::kHalf?2:1);}void*data_ptr()const{return reinterpret_cast<void*>(address?address:uint64_t(owner)*0x400000);}int dim()const{return rank;}')
    cpp = cpp.replace('struct Weights', 'uint64_t RpuGetDevAddr(void*p){return reinterpret_cast<uint64_t>(p);}\nstruct Weights', 1)
    cpp = cpp.replace('enum class KernelId{', 'enum class KernelId{G05_GATE_KV_OWNER_W8A16_ACC16_M32N256K1024_N128,')
    cpp = cpp.replace('bool q=true,kv=true,fanout=true,joint=true;', 'bool q=true,kv=true,fanout=true,joint=true,gatekv=true;')
    cpp = cpp.replace('bool has_loaded(KernelId id){++queries;',
                      'bool has_loaded(KernelId id){++queries;if(id==KernelId::G05_GATE_KV_OWNER_W8A16_ACC16_M32N256K1024_N128)return gatekv;')
    wrapper = r'''
void rpu_launch_g05_gate_kv_owner_only_w8a16_acc16_kernel(uint32_t input,const at::Tensor&g,const at::Tensor&k,const at::Tensor&v,const at::Tensor&gs,const at::Tensor&ks,const at::Tensor&vs,uint32_t go,uint32_t ko,uint32_t vo){
 assert(input==0x10000&&go==0x24000&&ko==0x28000&&vo==0x2c000);
 assert(g.tag=="gate"&&k.tag=="k"&&v.tag=="v"&&gs.owner==g.owner+10000&&ks.owner==k.owner+10000&&vs.owner==v.owner+10000);
 launches.push_back("gate-kv-owner");
}
'''
    cpp = cpp.replace('struct Model{', wrapper + 'struct Model{')
    cpp += r'''
void ready(Model&m){m.linear_acc32_=false;m.action_nvfp4_.values={1,2,3,4,5};
 for(auto&w:m.layer_weights_)for(auto*t:{&w.q_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w})t->dtype=at::kByte;
 m.install();m.action_owner_cache_available_=true;m.prepare_action_gate_kv_owner();m.loop();}
int main(){
 auto&cache=KernelCache::instance();Model m;ready(m);
 assert(m.action_gate_kv_owner_available_&&m.action_kv_owner_available_&&m.action_kv_owner_w8_&&!m.action_qkv_gate_available_);
 assert(m.use_action_gate_kv_owner(32,0)&&m.use_action_gate_kv_owner(32,245));
 const auto site=QWEN35_ACTION_GATE_KV_OWNER_W8_SITE;
 for(int64_t pos:{0,245,161,193,225,257,321,385,449}){m.position=pos;m.plan({32,0,0});assert(m.routes.size()==2&&m.routes[0].site==site&&m.routes[0].args==m.action_gate_kv_owner_arguments(pos));for(auto a:m.routes[0].args)assert(a>=0);
  for(int layer=0;layer<24;++layer){launches.clear();m.phase2(layer,{32,0,0});assert(launches==std::vector<std::string>({"q","gate-kv-owner"}));}}
 auto identity=m.kvinsert_cost_weight_identity();auto hash=m.subclass_layout_hash();m.position=0;assert(m.subclass_layout_hash()==hash&&m.kvinsert_cost_weight_identity()==identity);m.position=245;
 {auto x=m;x.action_gate_kv_owner_available_=false;assert(x.subclass_layout_hash()!=hash&&x.kvinsert_cost_weight_identity()!=identity);}
 for(int family=0;family<6;++family){auto x=m;auto&w=x.layer_weights_.back();at::Tensor*t[]={&w.attn_gate_w,&w.k_w,&w.v_w,&w.attn_gate_ws,&w.k_ws,&w.v_ws};++t[family]->owner;assert(x.kvinsert_cost_weight_identity()!=identity);}
 // Direct launch rejects stale COMPLETE fields before submitting its projection.
 m.plan({32,0,0});for(size_t i=0;i<m.routes[0].args.size();++i){auto x=m;++x.routes[0].args[i];reject([&]{x.launch_action_gate_kv_owner(23,{32,0,0});});}
 for(int fault=0;fault<8;++fault){auto x=m;if(fault==0)x.complete=false;if(fault==1)x.routes.clear();if(fault==2)++x.routes[0].selector;if(fault==3)++x.routes[0].flags;if(fault==4)++x.routes[0].invocation;if(fault==5)++x.routes[0].site;if(fault==6)x.routes[0].family=FmbRouteFamily::OTHER;if(fault==7)x.position=0;reject([&]{x.launch_action_gate_kv_owner(0,{32,0,0});});}
 for(int layer:{-1,24})reject([&]{m.launch_action_gate_kv_owner(layer,{32,0,0});});
 for(auto c:{ChunkInfo{16,0,0},ChunkInfo{32,1,0},ChunkInfo{32,0,1}})reject([&]{m.launch_action_gate_kv_owner(0,c);});
 for(int64_t pos:{-1,1,244,246,544}){auto x=m;x.position=pos;assert(!x.use_action_gate_kv_owner(32,pos));x.plan({32,0,0});assert(x.routes.size()==1&&x.routes[0].site==QWEN35_ACTION_KV_OWNER_W8_SITE);launches.clear();x.phase2(23,{32,0,0});assert(launches==std::vector<std::string>({"q","kv-w8","gate"}));}
 // Missing optional symbol retains the original mixed producer and fanout.
 cache.gatekv=false;auto old=m;old.prepare_action_gate_kv_owner();assert(!old.action_gate_kv_owner_available_);old.plan({32,0,0});launches.clear();old.phase2(23,{32,0,0});assert(launches==std::vector<std::string>({"q","kv-w8","gate"}));cache.gatekv=true;
 // Every layer and all six independent DDR owners are cold checked.
 for(int layer=0;layer<24;++layer)for(int family=0;family<6;++family)for(int fault=0;fault<12;++fault){auto x=m;auto&w=x.layer_weights_[layer];at::Tensor*owners[]={&w.attn_gate_w,&w.k_w,&w.v_w,&w.attn_gate_ws,&w.k_ws,&w.v_ws};auto&t=*owners[family];if(fault==0)t.present=false;if(fault==1)t.dtype=at::kByte;if(fault==2)t.location=c10::DeviceType::CPU;if(fault==3)t.contiguous=false;if(fault==4)t.offset=1;if(fault==5)t.rank=3;if(fault==6)--t.rows;if(fault==7)t.address=1;if(fault==8)t.address=1ULL<<40;if(fault==9)t.address=(1ULL<<40)-256;if(fault==10)t.owner=0;if(fault==11)t.device_index=1;x.prepare_action_gate_kv_owner();assert(!x.action_gate_kv_owner_available_);}
 // Aligned offset views passed the older replica proof, but the new ABI
 // must disable cold rather than reject after Q has already been emitted.
 for(int family=0;family<4;++family){auto x=m;auto&w=x.layer_weights_.back();at::Tensor*t[]={&w.k_w,&w.v_w,&w.k_ws,&w.v_ws};t[family]->offset=256;t[family]->address=uint64_t(t[family]->owner)*0x400000+256;x.prepare_action_gate_kv_owner();assert(!x.action_gate_kv_owner_available_);x.plan({32,0,0});launches.clear();x.phase2(23,{32,0,0});assert(launches==std::vector<std::string>({"q","kv-w8","gate"}));}
 // Even read-only aliasing retains the older producer path; no late failure.
 for(int family=0;family<6;++family)for(int other=0;other<family;++other){auto x=m;auto&w=x.layer_weights_.back();at::Tensor*t[]={&w.attn_gate_w,&w.k_w,&w.v_w,&w.attn_gate_ws,&w.k_ws,&w.v_ws};t[family]->address=reinterpret_cast<uint64_t>(t[other]->data_ptr());x.prepare_action_gate_kv_owner();assert(!x.action_gate_kv_owner_available_);x.plan({32,0,0});launches.clear();x.phase2(23,{32,0,0});assert(launches==std::vector<std::string>({"q","kv-w8","gate"}));}
 for(int fault=0;fault<7;++fault){auto x=m;if(fault==0)x.action_nvfp4_.values.clear();if(fault==1)x.linear_acc32_=true;if(fault==2)x.action_owner_cache_available_=false;if(fault==3)x.action_kv_owner_w8_=false;if(fault==4)x.layer_weights_.pop_back();if(fault==5)x.action_len_=16;if(fault==6)x.z2_bound_=true;x.prepare_action_gate_kv_owner();assert(!x.action_gate_kv_owner_available_);}
 m.reset();assert(!m.action_gate_kv_owner_available_);Model fp16(false);fp16.install();assert(!fp16.action_gate_kv_owner_available_);Model w8;w8.linear_acc32_=false;w8.install();assert(!w8.action_gate_kv_owner_available_);
}
'''
    _compile_run(tmp_path, 'gate_kv_owner_model', cpp)
