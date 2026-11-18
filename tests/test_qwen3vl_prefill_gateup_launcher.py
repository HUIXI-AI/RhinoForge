"""Execute the production 4B prefill launcher with host register/queue recorders.

This checks admission, typed operand ordering and refreshed registers; it does
not replace the existing REF numeric probe or the COMPLETE full-model gate.
"""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
NAME = "rpu_launch_qwen3vl_4b_prefill_gate_up_swiglu_w8a16_spm_kernel"


def definition(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    end, depth = opening + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def unbound_lingbot_prefill_members(source):
    """Cold sibling metadata for ordinary Qwen/Wall CPU owner facades.

    Use production defaults and pure payload serializers. The optional no-gamma
    and QK emitters are outside these owners; fail if an unbound payload is used.
    """
    import re

    names = (
        "lingbot_prefill_mode_bound_", "lingbot_folded_rmsnorm_",
        "lingbot_prefill_kv_only_", "lingbot_prefill_gateup_shared_a_",
        "lingbot_prefill_qkv_mode_", "lingbot_prefill_kv_pair_",
        "lingbot_prefill_qk_norm_rope_",
    )
    members = "\n".join(re.search(
        rf"(?:bool|int64_t) {name} = [^;]+;", source).group(0) for name in names)
    members += r"""
    struct UnboundNoGammaContract {
        uint32_t capability=0;
        int resolve(int64_t,int64_t)const{throw std::runtime_error("unbound LingBot no-gamma route");}
        std::vector<int64_t> manifest_arguments(int64_t,int64_t)const{throw std::runtime_error("unbound LingBot no-gamma route");}
    } lingbot_nogamma_contract_;
    std::vector<int64_t> lingbot_prefill_qk_norm_rope_arguments()const {
        throw std::runtime_error("unbound LingBot QK payload");
    }
    """
    for name in ("policy", "gateup", "qkv", "kv_pair"):
        members += "\n" + definition(source, f"std::vector<int64_t> lingbot_prefill_{name}_arguments(")
    return members


def test_actual_launcher_bounds_typed_operands_and_live_registers(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("native launcher checks require g++")
    wrapper = definition((ROOT/"src/ops/rpu_linear.cpp").read_text(), f"void {NAME}(")
    header = (ROOT/"src/core/rpu_kernel_decls.h").read_text()
    start = header.index(f"void {NAME}(")
    declaration = header[start:header.index(";",start)+1]
    harness = r'''
#include "rpu_linear_tiling.h"
#include "rpu_pi_wall_tiling.h"
#include <cassert>
#include <functional>
#include <stdexcept>
#include <vector>
#define NUM_CORES 8
template<class... Args> void check(bool condition,Args&&...) {
  if(!condition) throw std::runtime_error("rejected");
}
#define TORCH_CHECK(...) check(__VA_ARGS__)
namespace c10 { using Half=uint16_t; enum class DeviceType {PrivateUse1,CPU}; }
namespace at {
constexpr int kChar=1,kHalf=2,kByte=3,kBFloat16=4;
struct Tensor {
  std::vector<int64_t> shape{9728,2560};
  int dtype=kChar; bool present=true,contiguous=true;
  c10::DeviceType kind=c10::DeviceType::PrivateUse1;
  uint64_t address=0x1200000000ULL;
  struct Device {c10::DeviceType value; c10::DeviceType type() const{return value;}};
  bool defined() const{return present;}
  Device device() const{return {kind};}
  int scalar_type() const{return dtype;}
  bool is_contiguous() const{return contiguous;}
  int64_t dim() const{return shape.size();}
  int64_t size(int i) const{return shape.at(i);}
  void* data_ptr() const{return reinterpret_cast<void*>(address);}
};
}
int flushes=0,addresses=0,stages=0,submits=0,writes=0;
void rpu_ddr_flush(void*) {++flushes;}
uint64_t RpuGetDevAddr(void* p) {++addresses;return reinterpret_cast<uint64_t>(p);}
struct SpmAllocator {
  static constexpr uint64_t SPM_USABLE=8*1024*1024-1024;
  bool initialized=true;
  bool is_initialized() const{return initialized;}
  uint32_t addr(int core,int off) const{assert(core==0);return 0x40000000u+off;}
} spm;
#define SPM_ALLOC spm
enum class KernelId {WALL_PREFILL_GATE_UP_SWIGLU_W8A16_M320N112};
KernelId pi_wall_kernel_id(size_t) { return KernelId::WALL_PREFILL_GATE_UP_SWIGLU_W8A16_M320N112; }
enum class GraphDdrRegisterRole {Int8ModelWeight,PerChannelScale};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {
  GraphDdrRegisterRole role; GraphDdrRegisterAccess access;
  GraphDdrRegisterEncoding encoding; int lo,hi;
};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel_t {
  std::array<uint16_t,64> regs{};
  void reset_regs(){regs.fill(0);}
  void set_regs(int slot,uint16_t value){regs.at(slot)=value;}
  uint32_t pair(int lo) const{return regs[lo]|uint32_t(regs[lo+1])<<16;}
} kernel;
bool missing_kernel=false;
Kernel_t* get_kernel(KernelId){return missing_kernel?nullptr:&kernel;}
#define GET_KERNEL(id) get_kernel(id)
struct Writer {
  std::vector<GraphDdrRegisterOperandSpec> operands;
  void write(Kernel_t& k) {
    ++writes;
    for(const auto& operand:operands) {
      const auto value=RpuGetDevAddr(operand.tensor.data_ptr())>>8;
      k.set_regs(operand.abi.lo,uint16_t(value));k.set_regs(operand.abi.hi,uint16_t(value>>16));
    }
  }
};
struct RpuKernelGraph {
  bool deny=false; std::vector<GraphDdrRegisterOperandSpec> last;
  std::vector<uint64_t> retained;
  static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
  Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> operands) {
    ++stages;check(!deny);
    assert(id==KernelId::WALL_PREFILL_GATE_UP_SWIGLU_W8A16_M320N112);
    assert(operands.size()==4);
    const int slots[]={2,20,32,34};
    for(int i=0;i<4;++i) {
      assert(operands[i].abi.lo==slots[i] && operands[i].abi.hi==slots[i]+1);
      assert(operands[i].abi.role==(i%2==0?GraphDdrRegisterRole::Int8ModelWeight:GraphDdrRegisterRole::PerChannelScale));
      assert(operands[i].abi.access==GraphDdrRegisterAccess::Read);
      assert(operands[i].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
    }
    last=operands;return {operands};
  }
  void keep_alive(const at::Tensor& tensor){retained.push_back(tensor.address);}
};
struct Queue {
  bool broadcast=false;
  void set_broadcast_mode(bool value){broadcast=value;}
  void enqueu_kernel(Kernel_t&,std::vector<int> grid,std::vector<uint8_t> cores) {
    assert(broadcast && writes>submits);
    assert((grid[0]==10 || grid[0]==13 || grid[0]==16 || grid[0]==26) && grid[1]==1 && grid[2]==1);
    assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++submits;
  }
} queue;
Queue* get_queue(int count){assert(count==8);queue.broadcast=false;return &queue;}
#define GET_QUEUE(count) get_queue(count)
__DECLARATION__
__WRAPPER__
using Operands=std::array<at::Tensor,4>;
Operands good() {
  Operands v;
  for(int i=0;i<4;++i)v[i].address+=uint64_t(i)*0x1000000;
  v[2].shape={9728};v[2].dtype=at::kHalf;
  v[3].shape={9728};v[3].dtype=at::kHalf;
  return v;
}
void launch(const Operands& v,int64_t rows=256,uint32_t input=0x40000100,uint32_t output=0x40200000) {
  __NAME__(input,v[0],v[1],output,rows,v[2],v[3]);
}
void rejects(const std::function<void()>& f,bool before_stage=true) {
  int old_submits=submits,old_stages=stages,old_flushes=flushes,old_addresses=addresses;
  bool failed=false;try {f();}catch(const std::runtime_error&) {failed=true;}
  assert(failed && submits==old_submits);
  if(before_stage)assert(stages==old_stages && flushes==old_flushes && addresses==old_addresses);
}
int main() {
  auto& graph=RpuKernelGraph::active();
  for(int m=16;m<=256;m+=16) {
    auto v=good();for(auto& t:v)t.address+=uint64_t(m)*256;
    const int old_retained=graph.retained.size(),old_writes=writes;
    const uint32_t input=0x40000100+uint32_t(m)*256,output=0x40200000+uint32_t(m)*256;
    launch(v,m,input,output);
    assert(writes==old_writes+1 && graph.retained.size()==size_t(old_retained+4));
    assert(kernel.pair(0)==input && kernel.pair(4)==output && kernel.pair(6)==0);
    assert(kernel.regs[8]==1 && kernel.regs[9]==m && kernel.pair(10)==1216);
    assert(kernel.regs[12]==2560 && kernel.regs[13]==160 && kernel.regs[14]==0 && kernel.regs[15]==8);
    assert(kernel.pair(16)==5120 && kernel.pair(18)==2432 && kernel.pair(22)==0 && kernel.pair(24)==0);
    assert(kernel.regs[26]==2 && kernel.regs[27]==1);
    for(int i=28;i<32;++i)assert(kernel.regs[i]==0);
    for(int i=36;i<64;++i)assert(kernel.regs[i]==0);
    for(auto pair:std::vector<std::pair<int,int>>{{0,2},{2,20},{1,32},{3,34}}) {
      assert(kernel.pair(pair.second)==uint32_t(v[pair.first].address>>8));
    }
    assert(graph.last[0].tensor.address==v[0].address && graph.last[1].tensor.address==v[2].address);
    assert(graph.last[2].tensor.address==v[1].address && graph.last[3].tensor.address==v[3].address);
  }
  for(int m:{-1,0,1,15,17,255,257,272,320,640}) rejects([&]{launch(good(),m);});
  for(int i=0;i<4;++i) {
    auto v=good();v[i].present=false;rejects([&]{launch(v);});
    v=good();v[i].contiguous=false;rejects([&]{launch(v);});
    v=good();v[i].kind=c10::DeviceType::CPU;rejects([&]{launch(v);});
    for(int dtype:{at::kByte,at::kBFloat16}) {v=good();v[i].dtype=dtype;rejects([&]{launch(v);});}
    v=good();v[i].dtype=i<2?at::kHalf:at::kChar;rejects([&]{launch(v);});
    v=good();v[i].shape[0]-=16;rejects([&]{launch(v);});
    v=good();v[i].shape.push_back(1);rejects([&]{launch(v);});
    v=good();++v[i].address;rejects([&]{launch(v);},false);
  }
  auto wrong=good();wrong[0].shape[1]=2048;rejects([&]{launch(wrong);});
  for(uint32_t input:{0u,0x100u,0x40000101u,0x40200000u,0x401fff00u,0x407f0000u})
    rejects([&]{launch(good(),256,input);});
  for(uint32_t output:{0u,0x100u,0x40200001u,0x40000100u,0x40000200u,0x407f0000u})
    rejects([&]{launch(good(),256,0x40000100,output);});
  // Exactly adjacent slabs and an output ending at usable SPM are legal.
  launch(good(),256,0x40000100,0x40000100+256*2560*2);
  launch(good(),256,0x40000100,0x40000000+SpmAllocator::SPM_USABLE-256*1216*2);
  spm.initialized=false;rejects([&]{launch(good());});spm.initialized=true;
  graph.deny=true;int old_addresses=addresses,old_flushes=flushes;
  rejects([&]{launch(good());},false);assert(addresses==old_addresses && flushes==old_flushes);graph.deny=false;
  missing_kernel=true;rejects([&]{launch(good());},false);
}
'''
    harness = harness.replace("__DECLARATION__",declaration).replace("__WRAPPER__",wrapper).replace("__NAME__",NAME)
    cpp, executable = tmp_path/"prefill_gateup.cpp", tmp_path/"prefill_gateup"
    cpp.write_text(harness)
    subprocess.run([compiler,"-std=c++17","-Wall","-Wextra","-Werror","-I",str(ROOT/"src/ops"),
                    str(cpp),"-o",str(executable)],check=True,capture_output=True,text=True)
    subprocess.run([str(executable)],check=True,capture_output=True,text=True)
