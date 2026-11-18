"""Execute the unified W4 register writers without RPU."""
from pathlib import Path

import pytest
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_shared_w4_registers_match_runtime_abi(tmp_path):
    compiler = shutil.which("g++")
    assert compiler
    source = tmp_path / "shared_w4.cpp"
    source.write_text(r'''
#include <array>
#include <cassert>
#include "src/ops/rpu_linear_tiling.h"
#include "src/ops/rpu_pi_wall_tiling.h"
struct Kernel {
  std::array<uint16_t, 30> regs{};
  unsigned wide_writes = 0, rejected_writes = 0;
  uint32_t set_regs(uint32_t index, uint16_t value) {
    regs.at(index) = value;
    return 0;
  }
  // Mirror Kernel_t's production integer overloads: an uncast int literal is
  // a 32-bit write, which requires an even halfword register and occupies two.
  uint32_t set_regs(uint32_t index, uint32_t value) {
    ++wide_writes;
    if (index % 2) { ++rejected_writes; return 3; }
    regs.at(index) = static_cast<uint16_t>(value);
    regs.at(index + 1) = static_cast<uint16_t>(value >> 16);
    return 0;
  }
  uint32_t set_regs(uint32_t index, int32_t value) {
    return set_regs(index, static_cast<uint32_t>(value));
  }
};
int main() {
  using namespace rpu_pl_tiling;
  Kernel overload_probe;
  assert(overload_probe.set_regs(27, 8) == 3);
  assert(overload_probe.rejected_writes == 1 && overload_probe.regs[27] == 0);
  // Independent shared ABI fixture, including high words of DDR/SPM/N/stride.
  ParallelLinearW4GroupRegisterArgs args{
    {0x1f120100u, 0x1234567800ull, 0x1f340200u, 0x1f560300u,
     33, 65568, 2048, 1, 8, 4096, 131136, 0x2345678900ull, 1, 1},
    128, 16, 1049088, 0x3456789a00ull, true};
  Kernel u;
  set_parallel_linear_w4_group_regs(u, args);
  assert(u.wide_writes == 0 && u.rejected_writes == 0);
  const std::array<uint16_t, 30> expected{{
    0x0100,0x1f12,0x5678,0x1234,0x0200,0x1f34,0x0300,0x1f56,
    1,33,32,1,2048,128,1,8,4096,0,64,2,0x6789,0x2345,
    0x789a,0x3456,0x0200,0x0010,2,8,128,16}};
  assert(u.regs == expected);
  args.is_uint4 = false;
  Kernel i;
  set_parallel_linear_w4_group_regs(i, args);
  assert(i.wide_writes == 0 && i.rejected_writes == 0);
  for (int r=0; r<30; ++r) {
    const auto want = r == 22 || r == 23 ? 0 : r == 27 ? 9 : expected[r];
    assert(i.regs[r] == want);
  }
  // Signed and unsigned W4 now share the same main ABI; no old signed slots.
  Kernel signed_w4;
  set_parallel_linear_int4_group_regs(signed_w4,
      {args.linear, args.group_size, args.local_groups, args.scale_group_block_byte_step});
  assert(signed_w4.regs == i.regs);
  auto bf16 = args;
  bf16.linear.x_dtype = 3;
  Kernel b;
  set_parallel_linear_w4_group_regs(b, bf16);
  assert(b.regs[26] == 3 && b.regs[27] == 9);
  // The weight backing is N32 padded; runtime stores retain logical N16 lanes.
  // Source ASM's wr14 output loop advances by 16 and uses the supplied stride.
  auto tail = args;
  tail.linear.local_n = 48;
  tail.linear.output_m_byte_step = 96;
  Kernel t;
  set_parallel_linear_w4_group_regs(t, tail);
  assert(t.regs[10] == 48 && t.regs[11] == 0 && t.regs[18] == 96);
  for (bool acc32 : {false, true}) {
    const auto tile = acc32 ? select_tile_int4_acc32(1,768,2048)
                           : select_tile_int4(1,768,2048);
    const auto name = acc32 ? autotile_kernel_name_int4_acc32(tile.m_tile,tile.n_tile,tile.k_tile)
                           : autotile_kernel_name_int4(tile.m_tile,tile.n_tile,tile.k_tile);
    assert(name.find("parallel_linear_w4a16_pgrp_") == 0);
    const auto uname = acc32 ? autotile_kernel_name_uint4_acc32(tile.m_tile,tile.n_tile,tile.k_tile)
                            : autotile_kernel_name_uint4(tile.m_tile,tile.n_tile,tile.k_tile);
    assert(uname.find("parallel_linear_w4a16_pgrp_") == 0);
    assert((uname.find("acc32_") != std::string::npos) == acc32);
    assert((name.find("acc32_") != std::string::npos) == acc32);
  }
}
''')
    binary = tmp_path / "shared_w4"
    subprocess.run([compiler, "-std=c++17", "-I", str(ROOT), str(source),
                    "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_selected_shared_w4_payloads_exist_in_pinned_main():
    """Prevent an ID/role rename from selecting an absent or legacy payload."""
    import re
    catalog = (ROOT / "src/ops/rpu_linear_candidates.inc").read_text()
    names = set(re.findall(r'"((?:grouped_)?parallel_linear_w4a16_pgrp_[^"]+)"', catalog))
    assert names
    ref = ROOT / "src/rhinoOpLib_current.ref"
    if not ref.exists():
        pytest.skip("operator reference is supplied by the runtime package")
    main = ref.read_bytes()
    for name in names:
        assert name.encode() + b"\0" in main, name


def test_native_mlp_zero_install_validation_before_owner_publication(tmp_path):
    """Run the real installation lambda with metadata stand-ins and bad owners."""
    implementation = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    begin = implementation.index("    const auto prepare_zeros =")
    end = implementation.index("    auto gate_zero_owners =", begin)
    validator = implementation[begin:end]
    harness = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error("rejected"); } while(0)
namespace at {
constexpr int kHalf=0, kByte=1, kFloat=2;
using IntArrayRef = std::vector<int64_t>;
struct Tensor {
  bool valid=false, contiguous=true; int kind=kHalf, location=1;
  std::vector<int64_t> shape;
  Tensor()=default;
  Tensor(std::vector<int64_t> dimensions,int dtype,int device=1)
      : valid(true),kind(dtype),location(device),shape(dimensions) {}
  bool defined()const{return valid;}
  int scalar_type()const{return kind;}
  int device()const{return location;}
  bool is_contiguous()const{return contiguous;}
  int64_t dim()const{return shape.size();}
  int64_t size(int index)const{return shape.at(index);}
  const auto& sizes()const{return shape;}
  int64_t numel()const{if(!valid)return 0;int64_t n=1;for(auto s:shape)n*=s;return n;}
};
using TensorList=std::vector<Tensor>;
}
int retained=0;
int num_cores(){return 8;}
at::Tensor rpu_retain_linear_quant_scale(const at::Tensor&,const at::Tensor& z,int64_t cores=8){
 assert(cores==num_cores());++retained;return z;}
template<class F> bool rejected(F fn){try{fn();}catch(const std::runtime_error&){return true;}return false;}
int main(){
  const int64_t N=2;
  bool uint4_mlp=true;
  VALIDATOR
  using at::Tensor;
  at::TensorList weights{Tensor({256,1024},at::kHalf),Tensor({256,512},at::kByte)};
  at::TensorList scales{Tensor({0},at::kFloat,0),Tensor({128,32},at::kHalf)};
  auto zeros=scales;
  auto good=prepare_zeros(weights,scales,zeros,"gate",256,1024);
  assert(!good[0].defined() && good[1].numel()==4096 && retained==1);
  for(int fault=0;fault<7;++fault){
    auto bad=zeros;auto bad_scale=scales;auto bad_weight=weights;
    if(fault==0)bad[1]=Tensor({0},at::kFloat,0);
    if(fault==1)bad[1].kind=at::kByte;
    if(fault==2)bad[1].location=0;
    if(fault==3)bad[1].contiguous=false;
    if(fault==4)bad[0]=zeros[1];
    if(fault==5){bad[1].shape={128,16};bad_scale[1].shape={128,16};}
    if(fault==6)bad_weight[1].kind=at::kHalf;
    const int before=retained;
    assert(rejected([&]{prepare_zeros(bad_weight,bad_scale,bad,"gate",256,1024);}));
    assert(retained==before);
  }
  // Existing signed callers carry no zero lists and keep empty owners.
  uint4_mlp=false;
  auto signed_owners=prepare_zeros(weights,scales,{},"gate",256,1024);
  assert(signed_owners.size()==2 && !signed_owners[1].defined());
}
'''.replace("VALIDATOR", validator)
    source = tmp_path / "zero_install.cpp"
    source.write_text(harness)
    binary = tmp_path / "zero_install"
    subprocess.run([shutil.which("g++"), "-std=c++17", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_qwen35_zero_schema_preserves_existing_optional_tail():
    import torch
    registration = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    line = next(line for line in registration.splitlines()
                if 'm.def("qwen3_5_set_weights(' in line)
    schema = torch._C.parse_schema(line.split('m.def("', 1)[1].rsplit('");', 1)[0])
    assert [arg.name for arg in schema.arguments[-4:]] == [
        "nvfp4_tensor_scales", "gate_negative_zero", "up_negative_zero", "down_negative_zero"]
    for argument in schema.arguments[-4:]:
        assert str(argument.type) == "Optional[List[Tensor]]"
        assert argument.has_default_value() and argument.default_value is None


def test_production_w4_dispatch_preserves_signed_and_validates_uint4(tmp_path):
    """Execute both real W4 branches: payload selection, emitted ABI and rejects."""
    impl = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    start = impl.index('    TORCH_CHECK(scale.scalar_type() == at::kHalf,\n                "wINT4')
    end = impl.index('\n  // Weight in DDR', start)
    body = impl[start:end].removesuffix('  }\n')
    header = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    start = header.index('enum class KernelId')
    enum = header[start:header.index('};', start) + 2]
    start = header.index('inline KernelId parallel_linear_kernel_id(')
    end = header.index('struct CachedKernel', start)
    selectors = header[start:end]
    harness = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#include "src/ops/rpu_linear_tiling.h"
#include "src/ops/rpu_pi_wall_tiling.h"
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while(0)
namespace c10 { struct Half {uint16_t v;}; namespace DeviceType {constexpr int PrivateUse1=1;} }
namespace at {
constexpr int kHalf=0,kByte=1,kFloat=2;
struct Device {int value;int type()const{return value;}bool operator==(Device d)const{return value==d.value;}};
struct Tensor {
  bool valid=false,contiguous=true;int dtype=kHalf,location=1;uintptr_t address=0;
  std::vector<int64_t> shape;
  Tensor()=default;
  Tensor(std::vector<int64_t> s,int d,uintptr_t a):valid(true),dtype(d),address(a),shape(s){}
  bool defined()const{return valid;}int scalar_type()const{return dtype;}Device device()const{return {location};}
  bool is_contiguous()const{return contiguous;}int64_t dim()const{return shape.size();}
  int64_t size(int i)const{return shape.at(i);}const auto& sizes()const{return shape;}
  int64_t numel()const{int64_t n=valid?1:0;for(auto x:shape)n*=x;return n;}
  template<class T=void>T* data_ptr()const{return reinterpret_cast<T*>(address);}
};
}
template<class T>T CeilDiv(T n,T d){return (n+d-1)/d;}
uint64_t RpuGetDevAddr(const void*p){return reinterpret_cast<uintptr_t>(p);}
void rpu_ddr_flush(const void*){}
__ENUM__
__SELECTORS__
struct Kernel_t {
  std::array<uint16_t,67> regs{};
  void reset_regs(){regs.fill(0);}
  void set_regs(unsigned i,uint16_t v){regs.at(i)=v;}
  void set_regs(unsigned i,uint32_t v){assert(i%2==0);regs.at(i)=v;regs.at(i+1)=v>>16;}
} kernel;
KernelId selected;
Kernel_t* get_kernel(KernelId id){selected=id;return &kernel;}
#define GET_KERNEL(id) get_kernel(id)
struct Queue {
  int calls=0,n=0;bool broadcast=false;
  void set_broadcast_mode(bool b){broadcast=b;}
  void enqueu_kernel(const Kernel_t&,std::vector<uint16_t>,std::vector<uint8_t> cores){assert(broadcast&&cores.size()==size_t(n));++calls;}
} queue;
Queue* get_queue(int n){queue.n=n;return &queue;}
#define GET_QUEUE(n) get_queue(n)
enum class GraphDdrRegisterRole{PackedUint4ModelWeight,PackedInt4ModelWeight,PerGroupScale,NegativeZeroPoint};
enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct Operand {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {Operand operand;at::Tensor tensor;};
int staged=0;
struct Writer {
  std::vector<GraphDdrRegisterOperandSpec> specs;
  void write(Kernel_t&k){for(const auto&s:specs){auto a=s.tensor.address>>8;k.set_regs(s.operand.lo,uint16_t(a));k.set_regs(s.operand.hi,uint16_t(a>>16));}}
};
struct RpuKernelGraph {
  static RpuKernelGraph& active(){static RpuKernelGraph graph;return graph;}
  Writer stage_kernel_ddr_registers(KernelId,std::vector<GraphDdrRegisterOperandSpec> specs){
    const bool u4=specs[0].operand.role==GraphDdrRegisterRole::PackedUint4ModelWeight;
    assert(specs.size()==size_t(u4?3:2));
    assert(u4||specs[0].operand.role==GraphDdrRegisterRole::PackedInt4ModelWeight);
    assert(specs[1].operand.role==GraphDdrRegisterRole::PerGroupScale);
    if(u4)assert(specs[2].operand.role==GraphDdrRegisterRole::NegativeZeroPoint);
    ++staged;return {specs};
  }
};
void run(at::Tensor weight,at::Tensor scale,at::Tensor negative_zero,bool use_acc32,int partition,int num_cores,int64_t M=33,int group=128){
  const int64_t N=2048,K=2048;
  const size_t local_m=M,local_n=partition?N/num_cores:N,local_k=partition?K:K/num_cores;
  const bool is_uint4=negative_zero.defined()&&negative_zero.numel()!=0;
  const uint32_t input_spm_addr=0x10000,output_spm_addr=0x20000,bias_spm_addr=partition?0x30000:0;
  const size_t dwidth=2;(void)group;
__BODY__
}
int main(){
  for(int group:{32,64,128})for(int cores:{2,8})for(int partition:{0,1})for(bool acc32:{false,true})for(bool u4:{false,true}){
    const int64_t n=partition?2048/cores:2048,k=partition?2048:2048/cores,g=k/group;
    const int64_t elements=((g+3)/4)*((n+63)/64)*cores*256;
    at::Tensor w({2048,1024},at::kByte,0x100000),s({group,elements/group},at::kHalf,0x200000),z;
    if(u4)z=at::Tensor(s.shape,at::kHalf,0x300000);
    const int before=queue.calls,was_staged=staged;run(w,s,z,acc32,partition,cores);
    assert(queue.calls==before+1&&staged==was_staged+1);
    const auto type=u4?rpu_pl_tiling::WeightType::Uint4:rpu_pl_tiling::WeightType::Int4;
    const auto tile=rpu_pl_tiling::select_tile(33,n,k,type,acc32,rpu_pl_tiling::Family::Parallel,group);
    assert(selected==parallel_linear_kernel_id(type,acc32,tile));
    const auto&r=kernel.regs;assert(r[2]==0x1000&&r[3]==0&&r[20]==0x2000&&r[21]==0);
    assert(r[6]==0&&r[7]==(partition?3:0)&&r[14]==partition);
    assert(r[8]==partition&&r[9]==33&&r[22]==(u4?0x3000:0)&&r[23]==0);
    assert(r[26]==2&&r[27]==(u4?8:9)&&r[28]==group&&r[29]==g);
    assert(r[64]==(n+tile.n_tile-1)/tile.n_tile&&r[65]==(33+tile.m_tile-1)/tile.m_tile);
  }
  at::Tensor w({2048,1024},at::kByte,0x100000),s({128,256},at::kHalf,0x200000),z(s.shape,at::kHalf,0x300000);
  for(int fault=0;fault<9;++fault){
    auto bw=w,bs=s,bz=z;int m=33;
    if(fault==0)bz.dtype=at::kByte;
    if(fault==1)bz.location=0;
    if(fault==2)bz.contiguous=false;
    if(fault==3)bz.shape={128,32};
    if(fault==4)bz.address+=256;
    if(fault==5)bs.shape={128,32};
    if(fault==6)bw.address+=1;
    if(fault==7)bs.location=0;
    if(fault==8)m=65536;
    int before=queue.calls;bool rejected=false;
    try{run(bw,bs,bz,false,1,8,m);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected&&queue.calls==before);
  }
}
'''.replace('__ENUM__', enum).replace('__SELECTORS__', selectors).replace('__BODY__', body)
    cpp = tmp_path / 'w4_dispatch.cpp'
    cpp.write_text(harness)
    binary = tmp_path / 'w4_dispatch'
    result = subprocess.run([shutil.which('g++'), '-std=c++17', '-I', str(ROOT),
                             '-I', str(ROOT / 'src/core'), '-I', str(ROOT / 'src/ops'), str(cpp), '-o', str(binary)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    subprocess.run([str(binary)], check=True)
