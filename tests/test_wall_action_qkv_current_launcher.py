"""Execute the production Action QKV launcher and its real generated contracts."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_qwen3vl_prefill_gateup_launcher import definition


ROOT = Path(__file__).resolve().parents[1]
NAME = "rpu_launch_wall_action_qkv_w8a16_multibase_spm_kernel"


def test_actual_action_qkv_variants_live_abi_and_admission(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("requires a host C++ compiler")
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    launcher = definition(source, f"void {NAME}(")
    helpers = "\n".join(definition(source, marker) for marker in (
        "static GraphKernelRegisterWriter wall_precision_registers(",
        "static void check_wall_action_spm_weight_range(",
    ))
    contract = (ROOT / "src/ops/rpu_pi_wall_action_qkv_contracts.h").read_text()
    contract = contract.replace('#include "rpu_kernel_cache.h"', "").replace("#pragma once", "")
    ids = sorted(set(re.findall(r"KernelId::(\w+)", launcher + contract)))
    enum = "enum class KernelId {" + ",".join(ids + ["Plain"]) + "};"
    harness = r'''
#include "ops/rpu_linear_tiling.h"
#include "ops/rpu_pi_wall_tiling.h"
#include "core/rpu_wall_sorting_config.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>
#define NUM_CORES 8
template<class... A> void check(bool condition,A&&...) {
  if(!condition) throw std::runtime_error("rejected");
}
#define TORCH_CHECK(...) check(__VA_ARGS__)
namespace c10 { using Half=uint16_t; enum class DeviceType {PrivateUse1,CPU}; }
namespace at {
constexpr int kChar=1,kHalf=2;
struct Tensor {
  std::vector<int64_t> shape;
  int dtype=kChar; bool present=true,contiguous=true;
  uint64_t address=0,bytes=0;
  c10::DeviceType kind=c10::DeviceType::PrivateUse1;
  struct Device {c10::DeviceType value;auto type()const{return value;}};
  bool defined()const{return present;} Device device()const{return {kind};}
  int scalar_type()const{return dtype;} bool is_contiguous()const{return contiguous;}
  int64_t dim()const{return shape.size();} int64_t size(int i)const{return shape.at(i);}
  const std::vector<int64_t>& sizes()const{return shape;}
  void* data_ptr()const{return reinterpret_cast<void*>(address);}
};
}
__ENUM__
__CONTRACT__
KernelId pi_wall_kernel_id(uint32_t){return KernelId::Plain;}
struct SpmAllocator {
  static constexpr uint64_t SPM_USABLE=8*1024*1024-1024;
  bool initialized=true;
  bool is_initialized()const{return initialized;}
  uint32_t addr(int core,int off)const{assert(core==0);return 0x40000000u+off;}
} spm;
#define SPM_ALLOC spm
int flushes=0;
void rpu_ddr_flush(void*){++flushes;}
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uint64_t>(p);}
struct Kernel_t {
  std::array<uint16_t,256> regs{};
  int rejected=0,wide=0,atomic_writes=0;
  KernelId id=KernelId::Plain;
  void reset_regs(){
    std::fill(regs.begin(),regs.begin()+68,0);
    rejected=wide=atomic_writes=0;
  }
  uint32_t set_regs(uint32_t slot,uint16_t value){
    if(slot>=regs.size()||(slot>=68&&slot<128)){++rejected;return 3;}
    regs[slot]=value;if(slot>=128)++atomic_writes;return 0;
  }
  uint32_t set_regs(uint32_t slot,uint32_t value){
    ++wide;
    if(slot%2||slot+1>=regs.size()){++rejected;return 3;}
    regs[slot]=uint16_t(value);regs[slot+1]=uint16_t(value>>16);return 0;
  }
  uint32_t pair(int slot)const{return uint32_t(regs[slot])|(uint32_t(regs[slot+1])<<16);}
} projection,fanout;
bool missing=false;
Kernel_t* get_kernel(KernelId id){
  auto& kernel=id==KernelId::WALL_ACTION_KV_OWNER_FANOUT_M20?fanout:projection;
  kernel.id=id;return missing?nullptr:&kernel;
}
#define GET_KERNEL(id) get_kernel(id)
enum class GraphDdrRegisterRole {Int8ModelWeight,ModelWeight,PerChannelScale,SemanticInput};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {
  GraphDdrRegisterRole role;GraphDdrRegisterAccess access;
  GraphDdrRegisterEncoding encoding;int lo,hi;
};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct GraphKernelRegisterWriter {
  std::vector<GraphDdrRegisterOperandSpec> operands;
  void write(Kernel_t& k)const{
    for(const auto& o:operands){
      const uint64_t value=o.tensor.address>>8;
      k.set_regs(o.abi.lo,uint16_t(value));k.set_regs(o.abi.hi,uint16_t(value>>16));
    }
  }
};
struct RpuKernelGraph {
  bool deny=false;std::vector<GraphDdrRegisterOperandSpec> last;
  static RpuKernelGraph& active(){static RpuKernelGraph graph;return graph;}
  GraphKernelRegisterWriter stage_kernel_ddr_registers(
      KernelId,std::vector<GraphDdrRegisterOperandSpec> operands) {
    check(!deny);std::array<bool,68> used{};
    for(const auto& o:operands){
      assert(o.abi.lo%2==0&&o.abi.hi==o.abi.lo+1&&!used.at(o.abi.lo));
      used.at(o.abi.lo)=true;
      check(o.tensor.defined()&&o.tensor.address&&o.tensor.address%256==0
            &&(o.tensor.address>>8)<=UINT32_MAX);
      assert(o.abi.access==GraphDdrRegisterAccess::Read);
      assert(o.abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
      assert(o.tensor.dtype==(o.abi.role==GraphDdrRegisterRole::Int8ModelWeight?at::kChar:at::kHalf));
    }
    last=operands;return {operands};
  }
};
struct Submission {Kernel_t kernel;std::vector<uint16_t> grid;};
struct Queue {
  bool broadcast=false;std::vector<Submission> submitted;
  void set_broadcast_mode(bool value){broadcast=value;}
  void enqueu_kernel(Kernel_t& k,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
    assert(broadcast&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));
    assert(k.rejected==0&&k.wide==0);
    submitted.push_back({k,grid});
  }
} queue;
Queue* get_queue(int n){assert(n==8);return &queue;}
#define GET_QUEUE(n) get_queue(n)
__HELPERS__
__LAUNCHER__
struct Case {
  bool fp16=false,owner=false;
  int64_t rows=20,position=13;
  uint32_t x=0x40001000,q=0x40020000,k=0x40030000,v=0x40040000;
  uint32_t qb=0x40050000,kb=0x40051000,vb=0x40052000;
  uint32_t qw=0,kw=0,vw=0;
  std::array<at::Tensor,3> weight,scale;
  at::Tensor cosine,sine;
  bool cos_present=true,sin_present=true;
  WallSortingConfig sorting;
  Case(bool half=false,int variant=0,int64_t m=20):fp16(half),owner(variant==2),rows(m){
    sorting.fp16_transfer_mask=half?4:0;sorting.silu_mul=m==32;
    for(int i=0;i<3;++i){
      const int n=i==0?2048:1024;
      weight[i]={{n,1024},half?at::kHalf:at::kChar,true,true,
                 0x1200000000ULL+i*0x1000000ULL,uint64_t(n*1024*(half?2:1))};
      scale[i]={{n},at::kHalf,!half,true,0x1300000000ULL+i*0x1000000ULL,uint64_t(n*2)};
    }
    cosine={{80,64},at::kHalf,true,true,0x1400000000ULL,80*64*2};
    sine=cosine;sine.address+=0x1000000;
    if(variant==1){qw=0x40100000;kw=0x40200000;vw=0x40300000;}
  }
  void run(){
    __NAME__(x,weight[0],weight[1],weight[2],q,k,v,qb,kb,vb,
      scale[0],scale[1],scale[2],cos_present?&cosine:nullptr,sin_present?&sine:nullptr,
      position,qw,kw,vw,owner,sorting,rows);
  }
};
void reset(){
  queue.submitted.clear();queue.broadcast=false;missing=false;spm.initialized=true;
  RpuKernelGraph::active().deny=false;RpuKernelGraph::active().last.clear();
  projection.regs.fill(0xabab);fanout.regs.fill(0xcdcd);flushes=0;
}
void rejected(Case value){
  reset();bool failed=false;
  try{value.run();}catch(const std::runtime_error&){failed=true;}
  assert(failed&&queue.submitted.empty()&&RpuKernelGraph::active().last.empty());
}
void positive(Case c){
  reset();c.run();
  assert(queue.submitted.size()==size_t(c.owner?2:1));
  const auto& first=queue.submitted[0];
  const auto& r=first.kernel;const bool compact=c.qw!=0;
  const KernelId expected=c.fp16
    ? (compact?KernelId::WALL_ACTION_QKV_FP16_PARTIAL_MROPE_SPM_WEIGHT_M20N64K128
       :c.owner?KernelId::WALL_ACTION_QKV_FP16_OWNER_PARTIAL_MROPE_M20N64K128
               :KernelId::WALL_ACTION_QKV_FP16_PARTIAL_MROPE_M20N64K128)
    : (compact?KernelId::WALL_ACTION_QKV_PARTIAL_MROPE_W8A16_M20N64_SPM_WEIGHT
       :c.owner?KernelId::WALL_ACTION_QKV_OWNER_PARTIAL_MROPE_W8A16_M20N64
               :KernelId::WALL_ACTION_QKV_PARTIAL_MROPE_W8A16_M20N64);
  assert(r.id==expected);
  assert(first.grid==std::vector<uint16_t>({8,1,1}));
  assert(r.pair(0)==c.x&&r.pair(4)==c.q&&r.pair(6)==c.qb);
  assert(r.regs[8]==1&&r.regs[9]==c.rows&&r.pair(10)==256);
  assert(r.regs[12]==1024&&r.regs[13]==64&&r.regs[14]==1&&r.regs[15]==8);
  assert(r.pair(16)==2048&&r.pair(18)==512);
  assert(r.pair(22)==0&&r.pair(24)==0&&r.regs[26]==2&&r.regs[27]==(c.fp16?2:1));
  assert(r.regs[28]==0&&r.regs[29]==0);
  assert(r.pair(36)==c.k&&r.pair(38)==c.kb&&r.pair(44)==c.v&&r.pair(46)==c.vb);
  const int ws[]={2,32,40},ss[]={20,34,42};
  const uint32_t local[]={c.qw,c.kw,c.vw};
  for(int i=0;i<3;++i){
    assert(r.pair(ws[i])==(compact?local[i]:c.weight[i].address>>8));
    assert(r.pair(ss[i])==(c.fp16?0:c.scale[i].address>>8));
  }
  assert(r.pair(48)==c.cosine.address>>8&&r.pair(50)==c.sine.address>>8);
  assert(r.pair(52)==c.position&&r.atomic_writes==128);
  for(int i=128;i<256;++i)assert(r.regs[i]==0);
  const auto& operands=RpuKernelGraph::active().last;
  assert(operands.size()==size_t((compact?0:3)+(c.fp16?0:3)+2));
  for(const auto& o:operands){
    const int p=o.abi.lo;
    if(p==48||p==50)assert(o.abi.role==GraphDdrRegisterRole::SemanticInput);
    else if(p==20||p==34||p==42)assert(o.abi.role==GraphDdrRegisterRole::PerChannelScale);
    else assert(o.abi.role==(c.fp16?GraphDdrRegisterRole::ModelWeight:GraphDdrRegisterRole::Int8ModelWeight));
  }
  if(c.owner){
    const auto& second=queue.submitted[1];
    assert(second.grid==std::vector<uint16_t>({1,1,1}));
    assert(second.kernel.id==KernelId::WALL_ACTION_KV_OWNER_FANOUT_M20);
    assert(second.kernel.pair(0)==c.k-0x40000000u&&second.kernel.pair(2)==c.v-0x40000000u);
  }
}
int main(){
  for(bool fp16:{false,true})for(int variant:{0,1,2}){
    positive(Case(fp16,variant,20));
    if(variant!=2)positive(Case(fp16,variant,32));
  }
  Case c(false,1,32);
  for(auto& tensor:c.weight)tensor.address+=0x100000;
  for(auto& tensor:c.scale)tensor.address+=0x100000;
  c.cosine.address+=0x100000;c.sine.address+=0x100000;
  c.x+=256;c.q+=256;c.k+=256;c.v+=256;c.qb+=256;c.kb+=256;c.vb+=256;
  c.qw+=256;c.kw+=256;c.vw+=256;positive(c);
  c=Case();c.position=65541;c.cosine.shape={70000,64};c.sine.shape={70000,64};
  c.cosine.bytes=c.sine.bytes=70000*64*2;positive(c);
  c=Case();c.position=60;positive(c);c.position=61;rejected(c);
  c=Case();c.rows=21;rejected(c);
  c=Case(false,0,32);c.sorting.silu_mul=false;rejected(c);
  c=Case(true);c.sorting.fp16_transfer_mask=0;rejected(c);
  c=Case(false,2,32);rejected(c);
  c=Case(false,1);c.owner=true;rejected(c);
  c=Case(false,1);c.kw=0;rejected(c);
  c=Case(false,1);c.qw++;rejected(c);
  c=Case(false,1);c.qw=0x3fffff00;rejected(c);
  c=Case(true,1);c.vw=0x407fff00;rejected(c);
  c=Case(false,1);c.cos_present=c.sin_present=false;rejected(c);
  c=Case();c.sin_present=false;rejected(c);
  c=Case();c.position=-1;rejected(c);
  c=Case();c.position=int64_t(UINT32_MAX)+1;
  c.cosine.shape={int64_t(UINT32_MAX)+64,64};c.sine.shape=c.cosine.shape;rejected(c);
  c=Case();c.cosine.shape={32,64};c.sine.shape=c.cosine.shape;rejected(c);
  c=Case();c.sine.shape={80,32};rejected(c);
  c=Case();c.cosine.dtype=at::kChar;rejected(c);
  c=Case();c.cosine.present=false;rejected(c);
  c=Case();c.cosine.contiguous=false;rejected(c);
  c=Case();c.cosine.address++;rejected(c);
  c=Case();c.sine.address=uint64_t(1)<<40;rejected(c);
  c=Case();c.weight[1].shape={1023,1024};rejected(c);
  c=Case();c.weight[2].dtype=at::kHalf;rejected(c);
  c=Case();c.weight[0].contiguous=false;rejected(c);
  c=Case();c.weight[0].kind=c10::DeviceType::CPU;rejected(c);
  c=Case();c.weight[2].address++;rejected(c);
  c=Case();c.scale[0].shape={2047};rejected(c);
  c=Case();c.scale[1].dtype=at::kChar;rejected(c);
  c=Case(true);c.scale[0].present=true;rejected(c);
  c=Case();c.qb=0;rejected(c);
  reset();missing=true;bool failed=false;
  try{Case().run();}catch(const std::runtime_error&){failed=true;}
  assert(failed&&queue.submitted.empty());
  reset();RpuKernelGraph::active().deny=true;failed=false;
  try{Case().run();}catch(const std::runtime_error&){failed=true;}
  assert(failed&&queue.submitted.empty());
  reset();spm.initialized=false;failed=false;
  try{Case(false,1).run();}catch(const std::runtime_error&){failed=true;}
  assert(failed&&queue.submitted.empty());
  __MEMORY_NEGATIVES__
}
'''
    harness = (harness.replace("__ENUM__", enum).replace("__CONTRACT__", contract)
               .replace("__HELPERS__", helpers).replace("__LAUNCHER__", launcher)
               .replace("__NAME__", NAME).replace("__MEMORY_NEGATIVES__", r'''
  c=Case();c.x++;rejected(c);
  c=Case();c.q=0x3fffff00;rejected(c);
  c=Case(false,2);c.k=0x407fff00;rejected(c);
  c=Case();c.q=c.x;rejected(c);
  c=Case();c.k=c.q+256;rejected(c);
  c=Case();c.v=c.k+256;rejected(c);
  c=Case();c.q=c.kb;rejected(c);
  c=Case(false,1);c.q=c.kw;rejected(c);
  c=Case(true,1);c.v=c.qw+256;rejected(c);
  c=Case();c.qb++;rejected(c);
  c=Case();c.vb=0x407fff00;rejected(c);
  c=Case();c.qb=c.kb;positive(c);
'''))
    cpp, binary = tmp_path / "action_qkv.cpp", tmp_path / "action_qkv"
    cpp.write_text(harness)
    result = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                             "-I" + str(ROOT / "src"), "-I" + str(ROOT / "src/ops"),
                             str(cpp), "-o", str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
