"""Execute the generic production launchers against typed host queue recorders.

No device/runtime build: checks the actual appended catalog, ABI, dtype owners,
SPM spans and failure-before-flush/submit paths. Arithmetic remains OPS CModel
and board work; a passing host recorder is not kernel correctness evidence.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_siglip_attention_dry_live_contract import _function

ROOT = Path(__file__).resolve().parents[1]


def test_generic_grouped_launchers_typed_abi_and_route_contract(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host launcher recorder requires g++")
    cache = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    declarations = []
    for marker in ("enum class KernelId :", "static constexpr const char* KERNEL_ID_NAMES[]"):
        start = cache.index(marker)
        declarations.append(cache[start:cache.index("\n};", start) + 3])
    header = (ROOT / "src/ops/rpu_grouped_indexed_linear.h").read_text()
    header = re.sub(r'^#(?:include|pragma)[^\n]*\n', '', header, flags=re.MULTILINE)
    source = (ROOT / "src/ops/rpu_grouped_indexed_linear.cpp").read_text()
    source = re.sub(r'^#include "(?!rpu_grouped_indexed_linear_(?:(?:current_)?tiles|measurements)\.inc")[^\n]*\n', '', source, flags=re.MULTILINE)
    source = re.sub(r'^#include <ATen/ATen.h>\n', '', source, flags=re.MULTILINE)
    census = (ROOT / "src/graph/graph_register_census.cpp").read_text()
    graph_header = (ROOT / "src/graph/graph_runtime.h").read_text()
    execution = (ROOT / "src/graph/graph_runtime_execute.cpp").read_text()
    # Execute the actual admission, binding, writer/token and enqueue census
    # methods. Only Tensor/SDK memory and queue delivery are host boundaries.
    graph_methods = "\n".join(_function(census, marker) for marker in (
        "uint64_t fnv_byte(", "uint64_t fnv_u64(", "uint64_t fnv_string(",
        "void write_register_pairs(", "void GraphKernelRegisterWriter::write(",
        "void RpuKernelGraph::admit_kernel_register_lookup(KernelId id)",
        "void RpuKernelGraph::bind_pending_kernel_register_lookup(",
        "void RpuKernelGraph::write_pending_kernel_ddr_registers(",
        "void RpuKernelGraph::consume_pending_kernel_register_census(",
        "void RpuKernelGraph::observe_kernel_register_node("))
    graph_methods += "\n" + _function(execution,
        "::rhino_lkn::Kernel_t* RpuKernelGraph::get_kernel_reset(KernelId id)")
    writer_type = _function(graph_header, "class GraphKernelRegisterWriter final {") + ";"
    roles = _function(graph_header, "enum class GraphDdrRegisterRole : uint8_t {") + ";"
    stats_type = _function(graph_header, "struct GraphKernelRegisterCensusStats {") + ";"
    harness = r'''
#include <algorithm>
#include "rpu_linear_tiling.h"
#include "rpu_pi_wall_tiling.h"
#include <array>
#include <cassert>
#include <climits>
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace rhino_lkn {}
namespace c10 {template<class T>using optional=std::optional<T>;using Half=uint16_t;}
template<class... T> [[noreturn]] void fail(T&&... values){
 std::ostringstream message;(message<<...<<values);throw std::runtime_error(message.str());}
// Match c10's bare-if macro so an unbraced outer if/else fails here too.
#define TORCH_CHECK(condition, ...) if (!(condition)) { fail(__VA_ARGS__); }
#define TORCH_INTERNAL_ASSERT(condition) assert(condition)
namespace at {
using ScalarType=int;
constexpr int kByte=1,kChar=2,kHalf=3,kShort=4,kFloat=5,kPrivateUse1=6,kCPU=7;
struct Tensor {
 std::vector<int64_t> shape;
 int dtype=kByte,kind=kPrivateUse1,device_id=0;
 bool present=true,contiguous=true;uint64_t address=0x1200000000ULL;
 struct Device{int kind,id;int type()const{return kind;}
 bool operator==(Device b)const{return kind==b.kind&&id==b.id;}};
 bool defined()const{return present;}Device device()const{return {kind,device_id};}
 int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}
 int64_t dim()const{return shape.size();}int64_t size(int i)const{return shape.at(i);}
 uint64_t nbytes()const{uint64_t n=(dtype==kByte||dtype==kChar)?1:2;for(auto d:shape)n*=d;return n;}
 void* data_ptr()const{return reinterpret_cast<void*>(address);}
};}
__DISPATCH__
__HEADER__
int flushes=0,stages=0,submits=0,fetches=0,availability_queries=0;
void rpu_ddr_flush(void*){++flushes;}
struct SpmAllocator {
 static constexpr uint64_t SPM_USABLE=8*1024*1024-1024;
 bool initialized=true;bool is_initialized()const{return initialized;}
 uint32_t addr(int core,int offset)const{assert(core==0);return 0x40000000u+offset;}
} spm;
#define SPM_ALLOC spm
__ACTUAL_ROLES__
enum class GraphDdrRegisterAccess{Read};
enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct Abi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{Abi abi;at::Tensor tensor;};
struct Kernel_t {
 std::array<uint16_t,68> regs{};uint64_t token=1;
 void reset_regs(){regs.fill(0);++token;}
 int set_regs(int slot,uint16_t value){regs.at(slot)=value;++token;return 0;}
 int query_register_state_token(uint64_t* out)const{*out=token;return 0;}
 void set_regs(int,uint32_t)=delete;
 uint32_t pair(int slot)const{return uint32_t(regs[slot])|(uint32_t(regs[slot+1])<<16);}
} kernel;
namespace rhino_lkn{using Kernel_t=::Kernel_t;constexpr int kKernelRegOk=0,kKernelQueryOk=0;}
KernelId last_id=KernelId::GPL_INDEXED_W4_M256N32K256;
std::set<KernelId> missing;
struct KernelCache{
 static KernelCache& instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId id)const{++availability_queries;return !missing.count(id);}
};
Kernel_t* get_kernel(KernelId id);
#define GET_KERNEL(id) get_kernel(id)
class RpuKernelGraph;
__ACTUAL_WRITER_TYPE__
using Writer=GraphKernelRegisterWriter;
__ACTUAL_STATS_TYPE__
constexpr uint64_t kFnvOffset=UINT64_C(14695981039346656037),kFnvPrime=UINT64_C(1099511628211);
enum class GraphKernelRegisterRuleKind{TypedDdr,IntrinsicNoDdr};
struct GraphKernelRegisterCensus{
 enum class Kind{TypedDdr,NoDdr};Kind kind=Kind::TypedDdr;size_t policy_rule_index=0;
 uint64_t register_state_token=0;std::vector<GraphDdrRegisterOperandSpec> operands;
};
struct Rule{GraphKernelRegisterRuleKind kind=GraphKernelRegisterRuleKind::TypedDdr;
 std::optional<KernelId> kernel_id;std::string kernel_name;};
struct PendingKernelRegisterCensus{
 std::optional<KernelId> kernel_id;std::optional<std::string> kernel_name;
 GraphKernelRegisterCensus census;bool writer_committed=false,kernel_lookup_seen=false;
 bool writer_required=false;uint64_t writer_ticket=0,writer_state_token=0;
 Kernel_t* lookup_kernel=nullptr;Kernel_t* written_kernel=nullptr;
 std::vector<Writer::RegisterPair> expected_pairs;size_t replay_node_idx=0;
};
enum class GraphNodeKind{Kernel,Dma,Barrier};
struct Node{GraphNodeKind kind=GraphNodeKind::Kernel;
 std::optional<KernelId> kernel_id;std::optional<GraphKernelRegisterCensus> register_census;
 size_t kernel_idx=0;
 const Node& as_kernel()const{return *this;}Node& as_kernel(){return *this;}};
using KernelNodeData=Node;
struct RpuKernelGraph{
 enum class State{RECORDING,REPLAYING};State state_=State::RECORDING;
 bool kernel_register_census_active_=true,has_kernel_register_census_nodes_=false;
 bool kernel_register_census_complete_=false,kernel_register_census_poisoned_=false;
 std::optional<PendingKernelRegisterCensus> pending_kernel_register_census_;
 struct{std::vector<Rule> rules{Rule{}};}kernel_register_census_policy_;
 std::vector<Node> nodes_;size_t cursor_=0,pending_kernel_idx_=0;
 std::vector<std::shared_ptr<Kernel_t>> kernels_{std::shared_ptr<Kernel_t>(&kernel,[](auto*){})};
 bool kernel_params_dirty_this_replay_=false;uint64_t next_kernel_register_writer_ticket_=0;
 std::vector<size_t> kernel_register_census_rule_counts_{0};
 GraphKernelRegisterCensusStats kernel_register_census_stats_,kernel_register_census_phase_begin_stats_;
 void bind_pending_kernel_register_lookup(Kernel_t&);
 void write_pending_kernel_ddr_registers(uint64_t,Kernel_t&);
 void consume_pending_kernel_register_census(Kernel_t&,KernelNodeData*,size_t);
 void observe_kernel_register_node(GraphNodeKind,const std::string*);
 void bind_semantic_spm_producer_yield_kernel_lookup(Kernel_t&){}
 Kernel_t* get_kernel(KernelId id){admit_kernel_register_lookup(id);++fetches;last_id=id;
  return missing.count(id)?nullptr:&kernel;}
 Kernel_t* get_kernel_reset(KernelId);
 void check_kernel_register_dma_burst_closed(const char*){}
 const Rule& kernel_register_rule_for(KernelId)const{return kernel_register_census_policy_.rules[0];}
 void admit_kernel_register_lookup(KernelId);
 bool deny=false;std::vector<GraphDdrRegisterOperandSpec> last;
 static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> operands){
  ++stages;TORCH_CHECK(!deny);assert(!pending_kernel_register_census_);assert(operands.size()>=1&&operands.size()<=3);
  for(const auto& op:operands){assert(op.abi.access==GraphDdrRegisterAccess::Read);
   assert(op.abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
   auto alignment=uint64_t(256);
   if(op.abi.role==GraphDdrRegisterRole::PerGroupScale||op.abi.role==GraphDdrRegisterRole::NegativeZeroPoint)
     alignment=op.tensor.size(3)*512;
   TORCH_CHECK(op.tensor.address>0&&op.tensor.address%alignment==0&&op.tensor.address<(1ULL<<40));
  }
  PendingKernelRegisterCensus pending;pending.kernel_id=id;
  pending.writer_required=true;pending.writer_ticket=++next_kernel_register_writer_ticket_;
  pending.census.operands=operands;
  for(const auto& op:operands)pending.expected_pairs.push_back({uint16_t(op.abi.lo),uint16_t(op.abi.hi),uint32_t(op.tensor.address>>8)});
  if(state_==State::REPLAYING)pending.replay_node_idx=cursor_;
  kernel_register_census_policy_.rules[0].kernel_id=id;
  pending_kernel_register_census_=pending;
  last=operands;return Writer(this,pending.writer_ticket,pending.expected_pairs);
 }
};
__ACTUAL_GRAPH_ADMISSION__
Kernel_t* get_kernel(KernelId id){
 // Actual get_kernel_reset also executes binding and reset before the
 // production launcher sets ordinary/grid registers and commits its writer.
 return RpuKernelGraph::active().get_kernel_reset(id);
}
struct Queue{
 bool broadcast=false;int requested_cores=0;std::vector<uint16_t> grid;
 void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel_t& k,std::vector<uint16_t> g,std::vector<uint8_t> cores){
  auto& graph=RpuKernelGraph::active();
  if(graph.state_==RpuKernelGraph::State::RECORDING){
   Node node;node.kernel_id=last_id;
   graph.consume_pending_kernel_register_census(k,&node,0);graph.nodes_.push_back(node);
  }else{
   graph.consume_pending_kernel_register_census(k,nullptr,graph.cursor_);++graph.cursor_;
  }
  assert(broadcast&&int(cores.size())==requested_cores);
  for(int i=0;i<requested_cores;++i)assert(cores[i]==i);
  assert(g==std::vector<uint16_t>({k.regs[64],k.regs[65],k.regs[66]}));grid=g;++submits;
 }
} queue;
Queue* get_queue(int cores){queue.requested_cores=cores;queue.broadcast=false;return &queue;}
#define GET_QUEUE(cores) get_queue(cores)
__SOURCE__
struct Owners{at::Tensor w;c10::optional<at::Tensor>s,z;};
Owners owners(const gil::Shape& s){
 Owners o;const int dtype=int(s.weight_type);auto l=gil::layout(s);
 o.w.dtype=(dtype==8||dtype==5)?at::kByte:dtype==1?at::kChar:at::kHalf;
 o.w.shape={s.E,int64_t(l.weight_expert_bytes/(dtype==2?2:1))};
 if(dtype!=2){at::Tensor scale;scale.dtype=at::kHalf;scale.address=0x1400000000ULL;
  scale.shape=dtype==8?std::vector<int64_t>{s.E,(s.K/s.Q+3)/4,(s.N+63)/64,s.TP,4,64}:
   std::vector<int64_t>{s.E,int64_t(l.scale_expert_bytes/2)};o.s=scale;
  if(dtype==8){o.z=scale;o.z->address=0x1500000000ULL;}
 }
 return o;
}
void launch(const gil::Shape& s,const Owners& o,const gil::Selection& t,
 uint32_t x=0x40010000,uint32_t y=0x40400000,uint32_t c=0x40700000,
 uint32_t i=0x40710000,uint32_t r=0x40720000){
 rpu_launch_grouped_indexed_linear_selected(s,t,x,o.w,c,o.s,o.z,i,r,y);
}
void rejects(std::function<void()> f,bool pre_stage=true){
 int before_submits=submits,before_flushes=flushes,before_stages=stages;
 bool rejected=false;try{f();}catch(const std::runtime_error&){rejected=true;}
 assert(rejected&&submits==before_submits&&flushes==before_flushes);
 if(pre_stage)assert(stages==before_stages);
 // Model Graph abort owns failed-invocation cleanup; the recorder has no SDK.
 RpuKernelGraph::active().pending_kernel_register_census_.reset();
}
void check_abi(const gil::Shape& s,const gil::Selection& t,const Owners& o){
 assert(kernel.pair(0)==0x40010000&&kernel.pair(2)==o.w.address/256&&kernel.pair(4)==0x40400000);
 assert(kernel.pair(6)==0&&kernel.regs[8]==s.col_parallel&&kernel.regs[9]==s.R);
 assert(kernel.pair(10)==s.N&&kernel.regs[12]==s.K&&kernel.regs[13]==s.K/16);
 assert(kernel.regs[14]==0&&kernel.regs[15]==s.TP);
 assert(kernel.pair(16)==s.K*2&&kernel.pair(18)==s.N*2);
 assert(kernel.pair(20)==(o.s?o.s->address/256:0)&&kernel.pair(22)==(o.z?o.z->address/256:0));
 uint64_t qblock=int(s.weight_type)==8?((s.N+63)/64)*s.TP*512:0;
 uint64_t scale=int(s.weight_type)==8?((s.K/s.Q+3)/4)*qblock:
  (int(s.weight_type)==1||int(s.weight_type)==5)?((2*s.N*(s.col_parallel?s.TP:1)+255)/256)*256:0;
 uint64_t weight=s.N*s.K*s.TP*(int(s.weight_type)==2?2:1)/(int(s.weight_type)==8?2:1);
 assert(kernel.pair(24)==qblock&&kernel.regs[26]==2&&kernel.regs[27]==int(s.weight_type));
 assert(kernel.regs[28]==s.Q&&kernel.regs[29]==(s.Q?s.K/s.Q:0)&&kernel.regs[30]==s.E&&kernel.regs[31]==0);
 assert(kernel.pair(32)==0x40700000&&kernel.pair(34)==weight/256&&kernel.pair(36)==scale/256);
 assert(kernel.pair(38)==(o.z?scale/256:0)&&kernel.pair(40)==0x40710000);
 assert(kernel.pair(42)==(s.indexed?0:0x40720000)&&kernel.regs[44]==s.T&&kernel.regs[45]==s.R&&kernel.regs[46]==t.expert_partitions);
 for(int slot=47;slot<64;++slot)assert(kernel.regs[slot]==0);
 assert(queue.grid==std::vector<uint16_t>({uint16_t((s.N+t.n-1)/t.n),uint16_t(t.expert_partitions),1}));
 const auto& ops=RpuKernelGraph::active().last;
 assert(ops.size()==(s.weight_type==gil::WeightType::Uint4?3:s.weight_type==gil::WeightType::Float16?1:2));
 const GraphDdrRegisterRole wrole=s.weight_type==gil::WeightType::Uint4?GraphDdrRegisterRole::PackedUint4ModelWeight:
  s.weight_type==gil::WeightType::Int8?GraphDdrRegisterRole::Int8ModelWeight:
  s.weight_type==gil::WeightType::Float8E4M3?GraphDdrRegisterRole::Fp8ModelWeight:GraphDdrRegisterRole::ModelWeight;
 assert(ops[0].abi.role==wrole&&ops[0].abi.lo==2&&ops[0].abi.hi==3);
 if(o.s)assert(ops[1].abi.lo==20&&ops[1].abi.hi==21&&ops[1].abi.role==
  (o.z?GraphDdrRegisterRole::PerGroupScale:GraphDdrRegisterRole::PerChannelScale));
 if(o.z)assert(ops[2].abi.lo==22&&ops[2].abi.hi==23&&ops[2].abi.role==GraphDdrRegisterRole::NegativeZeroPoint);
}
int main(){
 for(bool indexed:{false,true})for(auto dtype:{gil::WeightType::Float16,gil::WeightType::Int8,
      gil::WeightType::Float8E4M3,gil::WeightType::Uint4}){
  gil::Shape shape{indexed,37,19,17,160,indexed?256:64,dtype==gil::WeightType::Uint4?32:0,8,dtype,true};
  auto choice=gil::canonical(shape);auto own=owners(shape);launch(shape,own,choice);
  auto& graph=RpuKernelGraph::active();const auto recorded=graph.nodes_.size()-1;
  const auto token=graph.nodes_[recorded].register_census->register_state_token;
  for(int replay=0;replay<2;++replay){
   graph.state_=RpuKernelGraph::State::REPLAYING;graph.cursor_=recorded;
   graph.kernel_params_dirty_this_replay_=false;launch(shape,own,choice);check_abi(shape,choice,own);
   assert(graph.cursor_==recorded+1&&graph.kernel_params_dirty_this_replay_);
   assert(graph.nodes_[recorded].register_census->register_state_token>token);
  }
  graph.state_=RpuKernelGraph::State::RECORDING;
 }
 size_t variants=0;
 for(const auto& tile:kGilTiles){
  assert(std::string(KERNEL_ID_NAMES[size_t(tile.id)])==tile.symbol);
  for(int tp:{1,2,4,8})for(bool col:{false,true})for(int q:tile.weight_type==gil::WeightType::Uint4?
       std::vector<int>{32,64,128}:std::vector<int>{0}){
   if(q && tile.k%q)continue;
   const gil::Shape s{tile.indexed,37,19,17,160,tile.k==64?64:320,q,tp,tile.weight_type,col};
   // Physical K includes a complete final quantization group.
   auto physical=s;if(q==128)physical.K=384;
   const auto automatic=gil::canonical(physical);
   // Every catalog entry is force-launched below. Automatic fallback is a
   // separate canonical candidate, not necessarily the forced tile under test.
   assert(gil::selected_tile(physical,automatic).indexed==physical.indexed);
   assert(automatic.expert_partitions==(physical.indexed?8:1));
   gil::Selection t{tile.m,tile.n,tile.k,1};auto o=owners(physical);
   int queries=availability_queries;
   launch(physical,o,t);check_abi(physical,t,o);assert(last_id==tile.id&&availability_queries==queries);
   if(tile.indexed){const auto p1=kernel.regs;t.expert_partitions=8;launch(physical,o,t);check_abi(physical,t,o);
    auto p8=p1;p8[46]=8;p8[65]=8;assert(kernel.regs==p8&&last_id==tile.id);}
   ++variants;
  }
 }
 gil::Shape s{true,768,2048,256,64,2048,128,8,gil::WeightType::Uint4,true};
 auto t=gil::canonical(s);auto o=owners(s);assert(t.m==256&&t.n==32&&t.k==256&&t.expert_partitions==8);
 launch(s,o,t);check_abi(s,t,o);
 for(int n:{32,224,256,288}){auto x=s;x.N=n;assert(gil::canonical(x).expert_partitions==(n<256?8:1));}
 auto resolved=gil::resolve(s);assert(resolved.expert_partitions==8);
 missing.insert(gil::selected_tile(s,t).id);rejects([&]{gil::resolve(s);});rejects([&]{launch(s,o,t);},false);missing.clear();
 auto malformed=t;malformed.m=17;rejects([&]{launch(s,o,malformed);});
 malformed=t;malformed.expert_partitions=2;rejects([&]{launch(s,o,malformed);});
 auto bad=s;bad.indexed=false;rejects([&]{gil::selected_tile(bad,t);});
 for(int tp:{0,3,6,9}){bad=s;bad.TP=tp;rejects([&]{launch(bad,o,t);});}
 for(int64_t r:{0,32768}){bad=s;bad.R=r;rejects([&]{launch(bad,o,t);});}
 for(int64_t tok:{0,65536}){bad=s;bad.T=tok;rejects([&]{launch(bad,o,t);});}
 bad=s;bad.N=48;rejects([&]{launch(bad,o,t);});bad=s;bad.K=96;rejects([&]{launch(bad,o,t);});
 bad=s;bad.N=(1<<20)+32;rejects([&]{launch(bad,o,t);});
 bad=s;bad.Q=16;rejects([&]{launch(bad,o,t);});bad=s;bad.weight_type=static_cast<gil::WeightType>(7);rejects([&]{launch(bad,o,t);});
 auto wrong=o;wrong.w.dtype=at::kChar;rejects([&]{launch(s,wrong,t);});
 wrong=o;wrong.w.shape[1]-=256;rejects([&]{launch(s,wrong,t);});
 wrong=o;wrong.w.contiguous=false;rejects([&]{launch(s,wrong,t);});
 wrong=o;wrong.w.kind=at::kCPU;rejects([&]{launch(s,wrong,t);});
 wrong=o;wrong.s.reset();rejects([&]{launch(s,wrong,t);});
 wrong=o;wrong.z->device_id=1;rejects([&]{launch(s,wrong,t);});
 wrong=o;wrong.s->shape={int64_t(wrong.s->nbytes()/2)};rejects([&]{launch(s,wrong,t);});
 wrong=o;wrong.s->address+=256;rejects([&]{launch(s,wrong,t);},false);
 wrong=o;wrong.w.address=1ULL<<40;rejects([&]{launch(s,wrong,t);},false);
 wrong=o;wrong.w.address+=16;rejects([&]{launch(s,wrong,t);},false);
 RpuKernelGraph::active().deny=true;rejects([&]{launch(s,o,t);},false);RpuKernelGraph::active().deny=false;
 rejects([&]{launch(s,o,t,0x40010001);}); // Alignment.
 rejects([&]{launch(s,o,t,0x10000);}); // Offset instead of absolute address.
 rejects([&]{launch(s,o,t,0x40600000);}); // Whole activation exceeds SPM.
 rejects([&]{launch(s,o,t,0x40010000,0x40010000);}); // Activation/output alias.
 rejects([&]{launch(s,o,t,0x40010000,0x40400000,0x40700000,0x40700000);});
 spm.initialized=false;rejects([&]{launch(s,o,t);});spm.initialized=true;
 // W8 per-channel scale is global N for col-TP, replicated local N for row-TP.
 for(bool col:{false,true}){
  auto a=s;a.weight_type=gil::WeightType::Int8;a.Q=0;a.col_parallel=col;
  auto own=owners(a);auto choice=gil::canonical(a);launch(a,own,choice);check_abi(a,choice,own);
  own.z=o.z;rejects([&]{launch(a,own,choice);});
 }
 // E4M3 keeps the W8 payload and scale layout while preserving raw-U8 owner
 // provenance and mode 5. Exercise both families, TP layouts and the short-K
 // scatter fallback used by Qwen3.5; missing payloads still fail before submit.
 for(bool indexed:{false,true})for(bool col:{false,true})for(int tp:{1,8}){
  gil::Shape a{indexed,37,19,17,80,indexed?288:64,0,tp,gil::WeightType::Float8E4M3,col};
  auto own=owners(a);auto choice=gil::resolve(a);
  const auto& payload=gil::selected_tile(a,choice);
  assert(payload.weight_type==gil::WeightType::Int8);
  launch(a,own,choice);check_abi(a,choice,own);assert(last_id==payload.id);
  const auto current=gil::resolve_current_auto(a);
  launch(a,own,current);check_abi(a,current,own);
  assert(kernel.regs[27]==5&&gil::selected_tile(a,current).weight_type==gil::WeightType::Int8);
  auto invalid=own;invalid.w.dtype=at::kChar;rejects([&]{launch(a,invalid,choice);});
  invalid=own;invalid.s.reset();rejects([&]{launch(a,invalid,choice);});
  invalid=own;invalid.s->shape[1]-=1;rejects([&]{launch(a,invalid,choice);});
  invalid=own;invalid.z=own.s;rejects([&]{launch(a,invalid,choice);});
  missing.insert(payload.id);rejects([&]{launch(a,own,choice);},false);
  const auto canonical_id=gil::selected_tile(a,gil::canonical(a)).id;
  if(payload.id!=canonical_id){
   const auto alternate=gil::resolve(a);
   assert(gil::selected_tile(a,alternate).id!=payload.id);
  }
  missing.clear();missing.insert(canonical_id);rejects([&]{gil::resolve(a);});missing.clear();
 }
 auto fp=s;fp.weight_type=gil::WeightType::Float16;fp.Q=0;auto fpo=owners(fp);auto fpt=gil::canonical(fp);
 fpo.s=o.s;rejects([&]{launch(fp,fpo,fpt);});
 // U16 IDs above 32767 are valid; scatter forbids repeats only within one expert.
 gil::Shape rs{true,65535,3,17,32,32,32,1,gil::WeightType::Uint4,true};
 std::vector<int16_t> counts(17);counts[0]=2;counts[16]=1;
 std::vector<uint16_t> ids{65534,65534,0};gil::validate_routes(rs,counts.data(),ids.data());
 rs.indexed=false;rejects([&]{gil::validate_routes(rs,counts.data(),ids.data());});
 ids={65534,1,65534};gil::validate_routes(rs,counts.data(),ids.data());
 ids[1]=65535;rejects([&]{gil::validate_routes(rs,counts.data(),ids.data());});ids[1]=1;
 counts[0]=-1;rejects([&]{gil::validate_routes(rs,counts.data(),ids.data());});counts[0]=1;
 rejects([&]{gil::validate_routes(rs,counts.data(),ids.data());});
 // Small E/R reserve padded 16-entry spans, not just the logical scalars.
 rs.indexed=true;rs.T=37;rs.R=1;rs.E=1;auto small=owners(rs);auto st=gil::canonical(rs);
 launch(rs,small,st,0x40010000,0x40400000,0x40700000,0x40700020);
 rejects([&]{launch(rs,small,st,0x40010000,0x40400000,0x40700000,0x40700010);});
 std::cout<<"typed_abi_variants="<<variants<<"; submits="<<submits<<"\n";
}
'''
    harness = harness.replace("__ACTUAL_GRAPH_ADMISSION__", graph_methods).replace(
        "__ACTUAL_WRITER_TYPE__", writer_type).replace("__ACTUAL_ROLES__", roles).replace(
        "__ACTUAL_STATS_TYPE__", stats_type).replace("__DISPATCH__", "\n".join(declarations)).replace("__HEADER__", header).replace("__SOURCE__", source)
    cpp = tmp_path / "launcher.cpp"
    cpp.write_text(harness)
    executable = tmp_path / "launcher"
    compiled = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src/core"), "-I", str(ROOT / "src/ops"), str(cpp), "-o", str(executable)], text=True, capture_output=True)
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(executable)], text=True, capture_output=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "typed_abi_variants=" in ran.stdout
    # Negative control: restore only the old GET-before-stage order. Every
    # other production statement and the real Graph admission body stay fixed.
    lookup = """  auto *kernel = GET_KERNEL(tile.id);
  TORCH_CHECK(kernel != nullptr,
              "grouped indexed linear: selected payload unavailable: ", tile.symbol);
"""
    assert harness.count(lookup) == 1
    legacy = harness.replace(lookup, "")
    owner_check = "  gil_check_spans(s, input, counts, ids, coefficients, output);\n"
    assert legacy.count(owner_check) == 1
    legacy = legacy.replace(owner_check, owner_check + lookup)
    negative_cpp = tmp_path / "legacy_order.cpp"
    negative_cpp.write_text(legacy)
    negative_exe = tmp_path / "legacy_order"
    compiled = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src/core"), "-I", str(ROOT / "src/ops"),
                               str(negative_cpp), "-o", str(negative_exe)], text=True, capture_output=True)
    assert compiled.returncode == 0, compiled.stderr
    rejected = subprocess.run([str(negative_exe)], text=True, capture_output=True)
    assert rejected.returncode != 0
    assert "typed/explicit-no-DDR Kernel lookup was not staged" in rejected.stderr
    write = "  writer.write(*kernel);\n"
    grid = "  const auto grid_x = static_cast<uint16_t>((s.N + tile.n - 1) / tile.n);\n"
    reset = "  kernel->reset_regs();\n"
    assert harness.count(write) == harness.count(grid) == harness.count(reset) == 1
    mutations = {
        "writer_before_grid": (harness.replace(write, "").replace(grid, write + grid),
                               "state changed after the staged writer"),
        "writer_before_reset": (harness.replace(write, "").replace(reset, write + reset),
                                "state changed after the staged writer"),
        "writer_missing": (harness.replace(write, ""), "enqueue precedes typed lookup/address writer"),
        "writer_duplicate": (harness.replace(write, write + write), "move-only and single-use"),
        "wrong_lookup_id": (harness.replace("GET_KERNEL(tile.id)",
            "GET_KERNEL(static_cast<KernelId>(static_cast<int>(tile.id)+1))"),
            "KernelId lookup order/identity mismatch"),
    }
    for label, (mutated, message) in mutations.items():
        candidate = tmp_path / (label + ".cpp")
        candidate.write_text(mutated)
        binary = tmp_path / label
        compiled = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT / "src/core"), "-I", str(ROOT / "src/ops"),
                                   str(candidate), "-o", str(binary)], text=True, capture_output=True)
        assert compiled.returncode == 0, (label, compiled.stderr)
        rejected = subprocess.run([str(binary)], text=True, capture_output=True)
        assert rejected.returncode != 0 and message in rejected.stderr, (label, rejected.stdout, rejected.stderr)


def test_generic_fixture_aten_and_dispatcher_signatures_compile(tmp_path):
    """Type-check real fixture code using ATen, with declarations for device APIs.

    No object, native backend, linked binary or device is produced. This catches
    fixture/schema errors that the raw-ABI Tensor recorder cannot represent.
    """
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("fixture syntax check requires g++")
    torch = pytest.importorskip("torch")
    torch_include = Path(torch.__file__).parent / "include"
    if not (torch_include / "ATen/ATen.h").is_file():
        pytest.skip("installed torch does not provide ATen headers")
    header = (ROOT / "src/ops/rpu_grouped_indexed_linear.h").read_text()
    header = re.sub(r'^#(?:include|pragma)[^\n]*\n', '', header, flags=re.MULTILINE)
    fixture = (ROOT / "python/rpu_backend/tests/conformance/_cpp/test_grouped_indexed_linear.cpp").read_text()
    fixture = re.sub(r'^#include "[^\n]*\n', '', fixture, flags=re.MULTILINE)
    graph_header = (ROOT / "src/graph/graph_runtime.h").read_text()
    infra = (ROOT / "src/graph/graph_infra.h").read_text()
    graph_types = "\n".join(_function(graph_header, marker) + ";" for marker in (
        "enum class GraphNodeKind : uint8_t {", "enum class GraphDdrRegisterRole : uint8_t {",
        "enum class GraphDdrRegisterAccess : uint8_t {", "enum class GraphDdrRegisterEncoding : uint8_t {",
        "enum class GraphKernelRegisterRuleKind : uint8_t {", "enum class GraphKernelNoDdrProof : uint8_t {",
        "struct GraphDdrRegisterAbi {", "struct GraphKernelRegisterRule {",
        "struct GraphKernelRegisterCensusStats {", "struct GraphKernelRegisterPhaseExpectation {",
        "struct GraphKernelRegisterPolicy {"))
    graph_types += "\n" + _function(infra, "struct GraphSignature {") + ";"
    source = r'''
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/util/Optional.h>
#include <cstdint>
#include <climits>
#include <vector>
enum class KernelId : uint16_t;
__HEADER__
namespace gil = rpu_grouped_indexed_linear;
struct SpmAllocator {
 bool is_initialized() const;
 void init();
 uint32_t addr(int core, uint32_t offset) const;
 uint32_t alloc_temporary(size_t bytes);
 struct ScopedTemporary {ScopedTemporary();~ScopedTemporary();};
} spm;
#define SPM_ALLOC spm
__GRAPH_TYPES__
struct GraphRuntimePolicy {int execution_core_count=8;};
struct GraphStats {
 bool register_census_present,register_census_build_committed;
 size_t kernel_count,segment_count,hw_batch_submit_total;
 uint64_t build_generation,execution_ordinal;
};
struct RpuKernelGraph {
 enum class State{PASSTHROUGH,RECORDING,REPLAYING,BUILT};
 explicit RpuKernelGraph(GraphRuntimePolicy={});
 State state() const;
 static RpuKernelGraph& active();
 void begin(const GraphSignature&);void end();void abort();void invalidate();
 GraphStats debug_stats()const;
};
struct GraphKernelRegisterCensusGuard {
 GraphKernelRegisterCensusGuard(RpuKernelGraph&,const GraphKernelRegisterPolicy&,uint64_t,uint64_t);
 const GraphKernelRegisterCensusStats& stats() const;
 void complete();
};
struct KernelCache{static KernelCache& instance();bool has_loaded(KernelId) const;};
namespace rhino_lkn { uint64_t RpuGetDevAddr(void*); }
void rpu_launch_ddr_broadcast_spm_dma_immediate(c10::Half*,int64_t,uint32_t,int);
void rpu_launch_ddr_scatter_spm_dma_immediate(c10::Half*,int64_t,int64_t,uint32_t,int);
void rpu_launch_spm_scatter_ddr_dma_immediate(uint32_t,c10::Half*,int64_t,int64_t,int);
__FIXTURE__
'''.replace("__HEADER__", header).replace("__GRAPH_TYPES__", graph_types).replace("__FIXTURE__", fixture)
    cpp = tmp_path / "fixture_syntax.cpp"
    cpp.write_text(source)
    compiled = subprocess.run([
        compiler, "-std=c++17", "-fsyntax-only", "-isystem", str(torch_include),
        "-isystem", str(torch_include / "torch/csrc/api/include"), str(cpp),
    ], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
