"""Execute the exact decode launcher with host register/queue recorders.

This gates ABI, backing storage, dynamic addresses and rejected submissions;
current-source arithmetic is covered separately by strict CModel fixtures.
"""
from pathlib import Path
import shutil
import re
import subprocess

import pytest

from test_qwen3vl_prefill_gateup_launcher import definition


ROOT = Path(__file__).resolve().parents[1]
NAME = "rpu_launch_qwen3vl_4b_decode_gate_up_swiglu_w8a16_spm_kernel"


def _run(tmp_path, source, name, *, expect_overload_failure=False):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("native launcher checks require g++")
    if "rpu_qwen3vl_linear::" in source:
        contract = (ROOT / "src/ops/rpu_qwen3vl_linear_contracts.h").read_text()
        contract = contract.replace('#include "rpu_kernel_cache.h"', '').replace('#pragma once', '')
        enum = re.search(r'enum class KernelId\s*\{([^}]*)\};', source)
        assert enum is not None
        present = set(re.findall(r'\b[A-Z][A-Z0-9_]*\b', enum[1]))
        ids = set(re.findall(r'KernelId::(\w+)', contract))
        declaration = 'enum class KernelId {' + enum[1].rstrip().rstrip(',') + ',' + ','.join(sorted(ids - present)) + '};'
        source = source[:enum.start()] + declaration + '\n' + contract + source[enum.end():]
        source = '#include "ops/rpu_linear_tiling.h"\n' + source
    cpp, executable = tmp_path / f"{name}.cpp", tmp_path / name
    cpp.write_text(source)
    compiled = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I" + str(ROOT / "src"), "-I" + str(ROOT / "src/core"), "-I" + str(ROOT / "src/ops"), str(cpp), "-o", str(executable)],
                   check=False, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run([str(executable)], check=False, capture_output=True, text=True)
    if expect_overload_failure:
        assert result.returncode != 0
        assert "kernel.rejected_writes==0" in result.stderr
    else:
        result.check_returncode()


def test_actual_decode_launcher_backing_bounds_and_live_typed_registers(tmp_path):
    wrapper = definition((ROOT / "src/ops/rpu_linear_gemv.cpp").read_text(), f"void {NAME}(")
    header = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    start = header.index(f"void {NAME}(")
    declaration = header[start:header.index(";", start) + 1]
    harness = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>
template<class... Args> void check(bool condition,Args&&...) {
  if(!condition) throw std::runtime_error("rejected");
}
#define TORCH_CHECK(...) check(__VA_ARGS__)
namespace c10 {
using Half=uint16_t;
enum class DeviceType {PrivateUse1,CPU};
enum class Layout {Strided,Sparse};
}
namespace at {
constexpr int kChar=1,kHalf=2,kByte=3,kBFloat16=4;
struct Tensor {
  std::vector<int64_t> shape{9728,2560};
  int dtype=kChar; bool present=true,contiguous=true;
  c10::DeviceType kind=c10::DeviceType::PrivateUse1;
  c10::Layout layout_kind=c10::Layout::Strided;
  uint64_t address=0x1200000000ULL;
  int64_t offset=0;
  struct Storage {
    uint64_t bytes=9728*2560; bool present=true;
    uint64_t nbytes() const{return bytes;}
    explicit operator bool() const{return present;}
  } backing;
  struct Device {c10::DeviceType value; c10::DeviceType type() const{return value;}};
  bool defined() const{return present;}
  Device device() const{return {kind};}
  c10::Layout layout() const{return layout_kind;}
  int scalar_type() const{return dtype;}
  bool is_contiguous() const{return contiguous;}
  int64_t dim() const{return shape.size();}
  int64_t size(int i) const{return shape.at(i);}
  int64_t storage_offset() const{return offset;}
  const Storage& storage() const{return backing;}
  void* data_ptr() const{return reinterpret_cast<void*>(address);}
};
}
int flushes=0,addresses=0,stages=0,submits=0,writes=0;
void rpu_ddr_flush(void*) {++flushes;}
struct SpmAllocator {
  static constexpr uint64_t SPM_USABLE=8*1024*1024-1024;
  bool initialized=true;
  bool is_initialized() const{return initialized;}
  uint32_t addr(int core,int off) const{assert(core==0);return 0x40000000u+off;}
} spm;
#define SPM_ALLOC spm
enum class KernelId {QWEN3VL_GATEUP_SWIGLU_GEMV};
enum class GraphDdrRegisterRole {Int8ModelWeight,PerChannelScale};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {
  GraphDdrRegisterRole role; GraphDdrRegisterAccess access;
  GraphDdrRegisterEncoding encoding; int lo,hi;
};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel_t {
  std::array<uint16_t,68> regs{}; // Actual SDK parameter bank: 0..67.
  int wide_writes=0,rejected_writes=0;
  void reset_regs(){regs.fill(0);}
  uint32_t set_regs(uint32_t slot,uint16_t value){
    if(slot>=regs.size()){++rejected_writes;return 3;}
    regs.at(slot)=value;return 0;
  }
  uint32_t set_regs(uint32_t slot,int16_t value){return set_regs(slot,static_cast<uint16_t>(value));}
  uint32_t set_regs(uint32_t slot,uint8_t value){return set_regs(slot,static_cast<uint16_t>(value));}
  uint32_t set_regs(uint32_t slot,int8_t value){return set_regs(slot,static_cast<uint16_t>(value));}
  uint32_t set_regs(uint32_t slot,uint32_t value){
    ++wide_writes;
    if((slot&1u) || slot+1>=regs.size()){++rejected_writes;return 3;}
    regs.at(slot)=static_cast<uint16_t>(value);
    regs.at(slot+1)=static_cast<uint16_t>(value>>16);return 0;
  }
  uint32_t set_regs(uint32_t slot,int32_t value){return set_regs(slot,static_cast<uint32_t>(value));}
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
      const auto value=operand.tensor.address>>8;
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
    assert(id==KernelId::QWEN3VL_GATEUP_SWIGLU_GEMV);
    assert(operands.size()==4);
    const int slots[]={2,20,32,34};
    for(int i=0;i<4;++i) {
      assert(operands[i].abi.lo==slots[i] && operands[i].abi.hi==slots[i]+1);
      assert(operands[i].abi.role==(i%2==0?GraphDdrRegisterRole::Int8ModelWeight:GraphDdrRegisterRole::PerChannelScale));
      assert(operands[i].abi.access==GraphDdrRegisterAccess::Read);
      assert(operands[i].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
      // Real address resolution is separately executed by the census suite.
      ++addresses;
      const uint64_t address=operands[i].tensor.address;
      check(address!=0 && address%256==0 && (address>>8)<=UINT32_MAX);
    }
    last=operands;return {operands};
  }
  void keep_alive(const at::Tensor& tensor){retained.push_back(tensor.address);}
};
struct Queue {
  bool broadcast=false;
  void set_broadcast_mode(bool value){broadcast=value;}
  void enqueu_kernel(Kernel_t&,std::vector<uint16_t> grid,std::vector<uint8_t> cores) {
    assert(broadcast && writes>submits);
    assert(kernel.rejected_writes==0 && kernel.wide_writes==0);
    assert(grid==std::vector<uint16_t>({8,1,1}));
    const auto c=rpu_qwen3vl_linear::contract(KernelId::QWEN3VL_GATEUP_SWIGLU_GEMV);
    assert(c.m==1 && c.n==160 && c.k==1024);
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
  for(int i=2;i<4;++i) {
    v[i].shape={9728};v[i].dtype=at::kHalf;
    v[i].backing.bytes=9728*2;
  }
  return v;
}
void launch(const Operands& v,uint32_t input=0x40000100,uint32_t output=0x40200000) {
  __NAME__(input,v[0],v[1],output,v[2],v[3]);
}
void rejects(const std::function<void()>& f,bool before_stage=true) {
  int old_submits=submits,old_stages=stages,old_flushes=flushes,old_addresses=addresses;
  const auto old_retained=RpuKernelGraph::active().retained.size();
  bool failed=false;try {f();}catch(const std::runtime_error&) {failed=true;}
  assert(failed && submits==old_submits && flushes==old_flushes);
  assert(RpuKernelGraph::active().retained.size()==old_retained);
  if(before_stage)assert(stages==old_stages && addresses==old_addresses);
}
int main() {
  // Reproduce the real SDK overload distinction: a plain int writes TWO
  // adjacent scalar halves only at an even register; odd writes return 3.
  Kernel_t sdk;
  assert(sdk.set_regs(12,uint32_t(0x12345678))==0);
  assert(sdk.regs[12]==0x5678 && sdk.regs[13]==0x1234);
  const auto old_regs=sdk.regs;
  assert(sdk.set_regs(13,1)==3 && sdk.regs==old_regs);
  assert(sdk.set_regs(13,uint16_t(1))==0 && sdk.regs[13]==1);
  auto& graph=RpuKernelGraph::active();
  std::array<uint16_t,68> original{};
  for(int step:{0,1,0}) { // A/B/A re-resolves every live DDR/SPM address.
    auto v=good();
    for(auto& t:v) {
      t.address+=step*256;
      t.offset=128; t.backing.bytes+=128*(t.dtype==at::kHalf?2:1);
    }
    const int old_retained=graph.retained.size(),old_writes=writes;
    const uint32_t input=0x40000100+step*256,output=0x40200000+step*256;
    launch(v,input,output);
    assert(writes==old_writes+1 && graph.retained.size()==size_t(old_retained+4));
    assert(kernel.pair(0)==input && kernel.pair(4)==output && kernel.pair(6)==0);
    assert(kernel.regs[8]==1 && kernel.regs[9]==1 && kernel.pair(10)==1216);
    assert(kernel.regs[12]==2560 && kernel.regs[13]==160 && kernel.regs[14]==0 && kernel.regs[15]==8);
    assert(kernel.pair(16)==5120 && kernel.pair(18)==2432);
    assert(kernel.pair(22)==0 && kernel.pair(24)==0 && kernel.regs[26]==2 && kernel.regs[27]==1);
    assert(kernel.regs[28]==0 && kernel.regs[29]==0);
    assert(kernel.regs[64]==8 && kernel.regs[65]==1 && kernel.regs[66]==1);
    for(auto pair:std::vector<std::pair<int,int>>{{0,2},{2,20},{1,32},{3,34}})
      assert(kernel.pair(pair.second)==uint32_t(v[pair.first].address>>8));
    assert(graph.last[0].tensor.address==v[0].address && graph.last[1].tensor.address==v[2].address);
    assert(graph.last[2].tensor.address==v[1].address && graph.last[3].tensor.address==v[3].address);
    if(submits==1) original=kernel.regs;
    else assert((kernel.regs==original)==(step==0));
  }
  for(int i=0;i<4;++i) {
    auto v=good();v[i].present=false;rejects([&]{launch(v);});
    v=good();v[i].contiguous=false;rejects([&]{launch(v);});
    v=good();v[i].kind=c10::DeviceType::CPU;rejects([&]{launch(v);});
    v=good();v[i].layout_kind=c10::Layout::Sparse;rejects([&]{launch(v);});
    for(int dtype:{at::kByte,at::kBFloat16}) {v=good();v[i].dtype=dtype;rejects([&]{launch(v);});}
    v=good();v[i].dtype=i<2?at::kHalf:at::kChar;rejects([&]{launch(v);});
    v=good();v[i].shape[0]-=16;rejects([&]{launch(v);});
    v=good();v[i].shape.push_back(1);rejects([&]{launch(v);});
    v=good();--v[i].backing.bytes;rejects([&]{launch(v);});
    v=good();v[i].backing.present=false;rejects([&]{launch(v);});
    for(int64_t offset:{int64_t(-1),int64_t(1),std::numeric_limits<int64_t>::max()}) {
      v=good();v[i].offset=offset;rejects([&]{launch(v);});
    }
    for(uint64_t address:{uint64_t(0),uint64_t(257),uint64_t(1)<<40}) {
      v=good();v[i].address=address;rejects([&]{launch(v);},false);
    }
  }
  auto wrong=good();wrong[0].shape[1]=2048;rejects([&]{launch(wrong);});
  wrong=good();wrong[2].backing.bytes=9728*2;launch(wrong);--wrong[2].backing.bytes;rejects([&]{launch(wrong);});
  for(uint32_t input:{0u,0x100u,0x40000101u,0x40200000u,0x401fff00u,0x407fff00u,0xffffff00u})
    rejects([&]{launch(good(),input);});
  for(uint32_t output:{0u,0x100u,0x40200001u,0x40000100u,0x40000200u,0x407fff00u,0xffffff00u})
    rejects([&]{launch(good(),0x40000100,output);});
  launch(good(),0x40000100,0x40000100+2560*2);
  // Last aligned output whose 2432 bytes fit in the usable core-0 window.
  launch(good(),0x40000100,0x40000000+SpmAllocator::SPM_USABLE-2560);
  spm.initialized=false;rejects([&]{launch(good());});spm.initialized=true;
  graph.deny=true;int old_addresses=addresses;
  rejects([&]{launch(good());},false);assert(addresses==old_addresses);graph.deny=false;
  missing_kernel=true;rejects([&]{launch(good());});
}
'''
    harness = harness.replace("__DECLARATION__", declaration).replace("__WRAPPER__", wrapper).replace("__NAME__", NAME)
    _run(tmp_path, harness, "decode_gateup")
    # Negative control for build17's real failure: the old plain-int constant
    # must compile, select int32_t, and be rejected at the odd register.
    legacy = wrapper.replace("  register_writer.write(*kernel);",
                             "  kernel->set_regs(13, 1);\n  register_writer.write(*kernel);")
    assert legacy != wrapper
    _run(tmp_path, harness.replace(wrapper, legacy), "decode_gateup_legacy_overload",
         expect_overload_failure=True)


def test_optional_expansion_symbols_keep_ids_and_use_current_names(tmp_path):
    header = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    source = (ROOT / "src/core/rpu_kernel_cache.inc").read_text()
    enum = definition(header, "enum class KernelId") + ";"
    names_start = header.index("static constexpr const char* KERNEL_ID_NAMES[]")
    names_open = header.index("{", names_start)
    names = header[names_open:header.index("};", names_open) + 2]
    optional = definition(source, "static bool is_optional_kernel(")
    for name in (
        "qwen3vl_gateup_swiglu_w8_m1_v2",
        "qwen3vl_qk_norm_mrope_d128",
        "qwen3vl_qk_norm_mrope_kv_insert_d128",
        "qwen3vl_o_ring_norm_w8_m1_h2560_v2",
        "rhinovla_newton_norm_shift_h1024",
        "rhinovla_silu_high_mul",
    ):
        assert f'{{"{name}", RHINO_EXPANSION_OP_LIB_PATH}}' in source
    assert "KernelId::QWEN3VL_GATEUP_SWIGLU_GEMV)] != nullptr" in source
    assert "KernelId::QWEN3VL_O_RING_NORM_M1_H2560_W8)] != nullptr" in source
    _run(tmp_path, r'''
#include "ops/rpu_linear_tiling.h"
#include "ops/rpu_pi_wall_tiling.h"
#include <cassert>
#include <cstdint>
#include <string>
#include <unordered_set>
''' + enum + "\nconst char* names[] = " + names + "\n" + optional + r'''
int main() {
  constexpr auto id=static_cast<unsigned>(KernelId::QWEN3VL_GATEUP_SWIGLU_GEMV);
  constexpr auto qk=static_cast<unsigned>(KernelId::QWEN3VL_QK_NORM_MROPE_D128);
  static_assert(id+1==qk);
  constexpr auto qkv=static_cast<unsigned>(KernelId::QWEN3VL_QK_NORM_MROPE_KV_INSERT_D128);
  static_assert(qk+1==qkv);
  constexpr auto orn=static_cast<unsigned>(KernelId::QWEN3VL_O_RING_NORM_M1_H2560_W8);
  static_assert(qkv+1==orn);
  static_assert(orn<static_cast<unsigned>(KernelId::_COUNT));
  static_assert(sizeof(names)/sizeof(*names)==static_cast<unsigned>(KernelId::_COUNT));
  assert(std::string(names[id])=="qwen3vl_gateup_swiglu_w8_m1_v2");
  assert(is_optional_kernel(names[id]));
  assert(std::string(names[qk])=="qwen3vl_qk_norm_mrope_d128");
  assert(is_optional_kernel(names[qk]));
  assert(std::string(names[qkv])=="qwen3vl_qk_norm_mrope_kv_insert_d128");
  assert(is_optional_kernel(names[qkv]));
  assert(std::string(names[orn])=="qwen3vl_o_ring_norm_w8_m1_h2560_v2");
  assert(is_optional_kernel(names[orn]));
  assert(is_optional_kernel("rhinovla_newton_norm_shift_h1024"));
  assert(is_optional_kernel("rhinovla_silu_high_mul"));
  assert(!is_optional_kernel("llama_gemv_wint8"));
  assert(!is_optional_kernel("task_qwen3vl_gateup_swiglu_gemv"));
}
''', "optional_expansion_gateup")
