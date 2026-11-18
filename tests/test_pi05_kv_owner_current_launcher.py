"""Execute both real KV owner parameter writers, including all prefill rows."""
from pathlib import Path
import shutil
import subprocess

import pytest

from test_qwen3vl_prefill_gateup_launcher import definition


ROOT = Path(__file__).resolve().parents[1]
PREFILL = "rpu_launch_pi05_prefill_kv1_pair_owner_w8a16_m400n32x2k2048_kernel"
DENOISE = "rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel"


def test_real_kv_owner_writers_preserve_current_dtype_and_typed_owners(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("requires C++ compiler")
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    contract = (ROOT / "src/ops/rpu_pi_kv_linear_contract.h").read_text()
    # The denoise wrapper delegates to the shared M50 owner, which may also
    # carry the fused Q projection; its own test executes that path.
    bodies = "\n".join(
        [definition(decls, "struct Pi05Nvfp4QProjection {") + ";",
         definition(source, "struct ActionNvfp4V2Plan {") + ";",
         "namespace rpu_pi_kv_linear {\n"
         + definition(contract, "inline rpu_pi_wall::Selection pair(") + "\n}",
         definition(source, "static void launch_pi05_kv1_pair_owner_m50(")]
        + [definition(source, "void " + name + "(") for name in (PREFILL, DENOISE)])
    harness = r'''
#include "ops/rpu_linear_tiling.h"
#include "ops/rpu_pi_wall_tiling.h"
#include "ops/rpu_pi05_q_kv1_abi.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>
template<class... T> void check(bool value,T&&...) {if(!value)throw std::runtime_error("rejected");}
#define TORCH_CHECK(...) check(__VA_ARGS__)
namespace c10 {using Half=uint16_t;enum class DeviceType {PrivateUse1,CPU};}
namespace at {
constexpr int kChar=1,kHalf=2;
struct Tensor {
  int dtype=kHalf;std::vector<int64_t> shape;uint64_t address=0,bytes=0;
  bool present=true,contiguous=true;int64_t offset=0;
  c10::DeviceType kind=c10::DeviceType::PrivateUse1;
  struct Device {c10::DeviceType value;auto type()const{return value;}
    bool operator==(const Device& rhs)const{return value==rhs.value;}};
  bool defined()const{return present;}int scalar_type()const{return dtype;}
  Device device()const{return {kind};}int64_t dim()const{return shape.size();}
  int64_t size(int i)const{return shape.at(i);}bool is_contiguous()const{return contiguous;}
  int64_t storage_offset()const{return offset;}uint64_t nbytes()const{return bytes;}
  void* data_ptr()const{return reinterpret_cast<void*>(address);}
};}
enum KernelId : uint32_t {PI05_KV_OWNER_W8_DB=1u<<20,PI05_Q_KV1_PAIR_DB_SSPM,PI05_Q_KV1_PAIR_DB_WSPM,PI05_NVFP4_LINEAR_DB_SSPM};
KernelId pi_wall_kernel_id(uint32_t ordinal){return KernelId(ordinal);}
// The REF carries the double-buffered W8 owner only when a case says so.
bool pair_db=false;
bool pi05_action_rows_admitted(int64_t rows){return rows==10||rows==50;}
bool pi05_kv1_pair_owner_w8_double_buffered(int64_t rows){return rows<=64&&pair_db;}
struct KernelCache{static KernelCache& instance(){static KernelCache c;return c;}
  bool has_loaded(KernelId)const{return true;}};
struct SpmAllocator {
  static constexpr uint64_t SPM_USABLE=8*1024*1024-1024;
  bool is_initialized()const{return true;}
  uint32_t addr(int core,int offset)const{assert(core==0);return 0x40000000u+offset;}
} spm;
#define SPM_ALLOC spm
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uint64_t>(p);}
void rpu_ddr_flush(void*){}
constexpr int kKernelRegOk=0;
struct Kernel_t {
  std::array<uint16_t,256> regs{};uint32_t id=0;
  void reset_regs(){regs.fill(0);}
  int set_regs(uint32_t slot,uint16_t value){regs.at(slot)=value;return 0;}
  uint32_t pair(int slot)const{return regs[slot]|(uint32_t(regs[slot+1])<<16);}
} kernel;
Kernel_t* get_kernel(KernelId id){kernel.id=id;return &kernel;}
#define GET_KERNEL(id) get_kernel(id)
enum class GraphDdrRegisterRole{ModelWeight,Int8ModelWeight,PerChannelScale,PackedFp4ModelWeight};
enum class GraphDdrRegisterAccess{Read};
enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;
  GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Writer {
  std::vector<GraphDdrRegisterOperandSpec> operands;
  void write(Kernel_t& k)const{for(const auto& o:operands){
    assert(o.abi.hi==o.abi.lo+1);
    assert(o.tensor.dtype==(o.abi.role==GraphDdrRegisterRole::Int8ModelWeight?at::kChar:at::kHalf));
    const auto value=o.tensor.address>>8;
    k.set_regs(o.abi.lo,uint16_t(value));k.set_regs(o.abi.hi,uint16_t(value>>16));}}
};
struct RpuKernelGraph {
  static bool has_active(){return true;}
  static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
  std::vector<GraphDdrRegisterOperandSpec> last;
  Writer stage_kernel_ddr_registers(KernelId,std::vector<GraphDdrRegisterOperandSpec> ops){last=ops;return {ops};}
  void keep_alive(const at::Tensor&){}
};
struct Queue {
  std::vector<uint16_t> grid;bool broadcast=false;
  void set_broadcast_mode(bool b){broadcast=b;}
  void enqueu_kernel(Kernel_t&,std::vector<uint16_t> g,std::vector<uint8_t> cores){
    assert(broadcast&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));grid=g;}
} queue;
Queue* get_queue(int n){assert(n==8);return &queue;}
#define GET_QUEUE(n) get_queue(n)
struct ActionNvfp4V2Plan;
static ActionNvfp4V2Plan plan_action_nvfp4_v2(const at::Tensor&,const at::Tensor&,int64_t,int64_t,
  int64_t,int,int,bool,bool,const at::Tensor&,uint32_t,int=0);
__BODIES__
static ActionNvfp4V2Plan plan_action_nvfp4_v2(const at::Tensor&,const at::Tensor&,int64_t,int64_t,
  int64_t,int,int,bool,bool,const at::Tensor&,uint32_t,int){throw std::runtime_error("unused");}
int main(){
  for(bool fp16:{false,true})for(int rows:{10,50,272,288,304,320,336,400,416,432,448,464}){
    const bool denoise=rows<=50;int k=denoise?1024:2048;uint64_t bytes=uint64_t(2048)*k*(fp16?2:1);
    at::Tensor kw{fp16?at::kHalf:at::kChar,{2048,k},0x1200000000ULL,bytes};
    auto vw=kw;vw.address+=0x2000000;
    at::Tensor ks{at::kHalf,{2048},0x1300000000ULL,4096,!fp16};
    auto vs=ks;vs.address+=0x2000000;
    const uint32_t x=0x40001000,ko=0x40400000,vo=0x40500000;
    auto call=[&]{if(denoise)__DENOISE__(x,kw,ks,ko,vw,vs,vo,rows,nullptr);
      else __PREFILL__(x,kw,ks,ko,vw,vs,vo,rows);};
    call();
    const auto selected=rpu_pi_wall::select(rpu_pi_wall::Operation::KvOwner,
      fp16?rpu_pl_tiling::WeightType::Fp16:rpu_pl_tiling::WeightType::Int8,false,denoise?50:rows,32,k);
    assert(kernel.id==selected.ordinal&&queue.grid==std::vector<uint16_t>({selected.grid_x,selected.grid_y,1}));
    if(fp16 && denoise){
      assert(std::string(rpu_pi_wall::candidate_names[selected.ordinal])=="pi_kv_owner_w16a16_acc16_m608n32k128");
      assert(selected.tile.m_tile==608 && selected.tile.n_tile==32 && selected.tile.k_tile==128);
      assert(queue.grid==std::vector<uint16_t>({2,1,1}));
    }
    if(fp16 && rows==304){
      assert(std::string(rpu_pi_wall::candidate_names[selected.ordinal])=="pi_kv_owner_w16a16_acc16_m304n32k128");
      assert(selected.tile.m_tile==304 && selected.tile.n_tile==32 && selected.tile.k_tile==128);
      assert(queue.grid==std::vector<uint16_t>({2,1,1}));
    }
    if(fp16 && (rows==272 || rows==288 || rows==320 || rows==336)){
      assert(std::string(rpu_pi_wall::candidate_names[selected.ordinal])=="pi_kv_owner_w16a16_acc16_m336n128k128");
      assert(selected.tile.m_tile==336 && selected.tile.n_tile==128 && selected.tile.k_tile==128);
    }
    assert(kernel.pair(0)==x&&kernel.pair(2)==kw.address>>8&&kernel.pair(4)==ko&&kernel.pair(6)==0);
    assert(kernel.regs[8]==1&&kernel.regs[9]==rows&&kernel.pair(10)==32);
    assert(kernel.regs[12]==k&&kernel.regs[13]==k/16&&kernel.regs[14]==0&&kernel.regs[15]==8);
    assert(kernel.pair(16)==uint32_t(k*2)&&kernel.pair(18)==64);
    assert(kernel.pair(20)==(fp16?0:ks.address>>8)&&kernel.pair(22)==0&&kernel.pair(24)==0);
    assert(kernel.regs[26]==2&&kernel.regs[27]==(fp16?2:1)&&kernel.regs[28]==0&&kernel.regs[29]==0);
    assert(kernel.pair(32)==vw.address>>8&&kernel.pair(34)==(fp16?0:vs.address>>8));
    assert(kernel.pair(36)==vo&&kernel.pair(38)==0);
    assert(RpuKernelGraph::active().last.size()==size_t(fp16?2:4));
    // The double-buffered W8 owner keeps this image on two blocks.
    if(denoise && !fp16){
      const auto image=kernel.regs;pair_db=true;call();pair_db=false;
      assert(kernel.id==KernelId::PI05_KV_OWNER_W8_DB&&kernel.regs==image);
      assert(queue.grid==std::vector<uint16_t>({2,1,1}));
    }
    auto rejects=[&]{bool failed=false;try{call();}catch(const std::runtime_error&){failed=true;}assert(failed);};
    kw.offset=1;rejects();kw.offset=0;
    vw.address=kw.address;rejects();vw.address+=0x2000000;
    if(fp16){ks.present=true;rejects();ks.present=false;}
    else {ks.dtype=at::kChar;rejects();ks.dtype=at::kHalf;}
  }
}
'''.replace("__BODIES__", bodies).replace("__PREFILL__", PREFILL).replace("__DENOISE__", DENOISE)
    cpp = tmp_path / "kv_owner.cpp"
    cpp.write_text(harness)
    exe = tmp_path / "kv_owner"
    subprocess.run([compiler,"-std=c++17","-O0","-I",str(ROOT/"src"),str(cpp),"-o",str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
    # The regression must detect the removed initializer even if sparse
    # numerical fixtures or a future kernel happen to mask its effect.
    marker = "rpu_pl_tiling::set_u32_reg_pair(*kernel, 36, v_output_spm_addr);"
    broken = harness.replace(marker,
        "for(uint32_t slot=24;slot<68;++slot)kernel->set_regs(slot,uint16_t(0));\n" + marker, 1)
    assert broken != harness
    cpp.write_text(broken)
    subprocess.run([compiler,"-std=c++17","-O0","-I",str(ROOT/"src"),str(cpp),"-o",str(exe)],check=True)
    rejected = subprocess.run([str(exe)],capture_output=True)
    assert rejected.returncode != 0 and b"kernel.regs[26]==2" in rejected.stderr
