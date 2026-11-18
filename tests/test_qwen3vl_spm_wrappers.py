"""Host checks of actual RMS/SwiGLU bodies and TM160 admission arithmetic.

These doubles observe register/grid/queue behavior. They do not execute device
instructions or certify arithmetic, cache coherency, or model performance.
"""
from pathlib import Path
from pi_wall_host_catalog import with_actual_linear_contracts
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _function(path, signature):
    source = (ROOT / path).read_text()
    start = source.index(signature)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def spm_wrapper_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    declarations = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    rms_contract = declarations[
        declarations.index("inline constexpr int64_t RPU_RMSNORM_SCM_U16_MAX"):
        declarations.index("enum class RpuMixedRmsNormSpmRoute")
    ]
    bodies = _function("src/ops/rpu_rmsnorm.cpp", "void rpu_launch_rmsnorm_spm_kernel(")
    bodies += "\n" + _function("src/ops/rpu_eltwise_unary.cpp", "void rpu_launch_silu_mul_exact_strided_spm_kernel(")
    sdpa = _function("src/ops/rpu_sdpa.cpp", "void launch_sdpa_spm_unified_impl(")
    kernel_ids = _function("src/core/rpu_kernel_cache.h", "enum class KernelId") + ";"
    sdpa_contract = _function("src/core/rpu_helpers.h", "struct SdpaTiling") + ";\n"
    sdpa_contract += _function(
        "src/ops/rpu_hyvla_sdpa_tiling.h", "inline bool rpu_hyvla_s51_key_rows_supported(")
    sdpa_contract += _function("src/ops/rpu_sdpa.cpp",
        "bool rpu_lingbot2_sdpa_tail_t51_key_rows_supported(")
    sdpa_contract += "\nconstexpr int MASK_2D = 4;\n"
    admission = sdpa[sdpa.index("{") + 1:sdpa.index("size_t dwidth")]
    program = r'''

#include <cassert>
#include <cstdint>
#include <climits>
#include <optional>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <string>
#define TORCH_CHECK(condition, ...) do {if(!(condition)) throw std::runtime_error(#condition);} while(0)
namespace c10 {struct Half {uint16_t bits=0; Half(float) {}};}
uint16_t half_to_u16(c10::Half) {return 0;}
__KERNEL_IDS__
__SDPA_CONTRACT__
struct KernelCache {static KernelCache& instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId)const{throw std::runtime_error("unexpected LingBot2 payload admission");}};
__RMS_CONTRACT__
struct Kernel_t {uint16_t regs[80]{}; void reset_regs(){std::memset(regs,0,sizeof(regs));} void set_regs(int reg,uint16_t value){regs[reg]=value;}} kernel;
KernelId selected;
Kernel_t* get_kernel(KernelId id) {selected=id;return &kernel;}
#define GET_KERNEL(id) get_kernel(id)
struct Queue {int launches=0;bool broadcast=false;std::vector<uint16_t> grid;std::vector<uint8_t> cores;void set_broadcast_mode(bool b){broadcast=b;}void enqueu_kernel(Kernel_t&,std::vector<uint16_t> g,std::vector<uint8_t> c){assert(broadcast);++launches;grid=g;cores=c;}} queue;
Queue* get_queue(int){return &queue;}
#define GET_QUEUE(n) get_queue(n)
constexpr int UNIFIED_NUM_CORES=8;
constexpr int MAX_CORES=8;
void rpu_launch_rmsnorm_spm_kernel(uint32_t,uint32_t,uint32_t,int64_t,int64_t,double,RpuRmsNormSpmRoute,int num_cores=8);
__BODIES__
template<class F> void reject(F f){int before=queue.launches;bool rejected=false;try{f();}catch(const std::runtime_error&){rejected=true;}assert(rejected && queue.launches==before);}

int64_t CeilDiv(int64_t value, int64_t divisor) { return (value+divisor-1)/divisor; }
void sdpa_admit(int num_cores, bool qwen3vl_tm160, bool use_16b=false,
                int64_t seq_q=576, int64_t kv_seq_len=576, int64_t head_dim=64,
                int64_t num_heads=16, int64_t num_kv_heads=16, int virtual_num_cores=8,
                int attn_mask_type=0, int64_t cache_batch_offset_elems=0,
                const SdpaTiling* admitted_tiling=nullptr,
                std::optional<KernelId> admitted_kernel=std::nullopt) {
    const bool lingbot2_tail=false;
    const std::optional<double> scale;
    __SDPA_ADMISSION__
    ++queue.launches;
}
int main(int argc, char** argv) {
 assert(argc==2);const std::string test=argv[1];
 if (test=="rms") {
  auto rms=[](int64_t rows,int64_t cols){rpu_launch_rmsnorm_spm_kernel(0x1000,0x5000,0x9000,rows,cols,1e-6,RpuRmsNormSpmRoute::QWEN3VL_V64);};
  rms(192,2048);assert(selected==KernelId::QWEN3VL_RMS_NORM_MULTIWARP_V64);assert(kernel.regs[0]==2048 && kernel.regs[2]==4096 && kernel.regs[3]==24);assert(queue.grid[0]==8 && queue.cores.size()==8);
  rms(64,128);rms(384,128);rms(64,8192);
  reject([&]{rms(192,130);});reject([&]{rms(191,2048);});reject([&]{rms(0,128);});reject([&]{rms(64,8193);});reject([&]{rms(524288,128);});
  rpu_launch_rmsnorm_spm_kernel(0x1000,0x5000,0x9000,1,128,1e-6,RpuRmsNormSpmRoute::BASE);
  assert(selected==KernelId::RMS_NORM_BF16_SPM && queue.grid[0]==1 && kernel.regs[3]==0);
  rpu_launch_rmsnorm_spm_kernel(0x1000,0x5000,0x9000,192,128,1e-6,RpuRmsNormSpmRoute::V16);
  assert(selected==KernelId::RMS_NORM_SPM_V16 && queue.grid[0]==12 && kernel.regs[3]==16);
  rpu_launch_rmsnorm_spm_kernel(0x1000,0x5000,0x9000,192,128,1e-6,RpuRmsNormSpmRoute::V32);
  assert(selected==KernelId::RMS_NORM_SPM_V32 && queue.grid[0]==6 && kernel.regs[3]==32);
  // New manifests use eight workgroups with rows/core in reg[3]; legacy
  // descriptors above retain their original grid and register semantics.
  for(auto route : {RpuRmsNormSpmRoute::V16_GRID8,RpuRmsNormSpmRoute::V32_GRID8}) {
   rpu_launch_rmsnorm_spm_kernel(0x1000,0x5000,0x9000,192,128,1e-6,route);
   assert(selected==(route==RpuRmsNormSpmRoute::V16_GRID8 ? KernelId::RMS_NORM_SPM_V16 : KernelId::RMS_NORM_SPM_V32));
   assert(queue.grid[0]==8 && kernel.regs[3]==24 && queue.cores.size()==8);
   reject([&]{rpu_launch_rmsnorm_spm_kernel(0x1000,0x5000,0x9000,191,128,1e-6,route);});
   reject([&]{rpu_launch_rmsnorm_spm_kernel(0x1000,0x5000,0x9000,524288,128,1e-6,route);});
  }
 } else if(test=="silu") {
  rpu_launch_silu_mul_exact_strided_spm_kernel(0x123400,0x223400,192,768,8);assert(selected==KernelId::QWEN3VL_SILU_MUL_EXACT_STRIDED);assert(kernel.regs[0]==768 && kernel.regs[1]==768);assert((uint32_t(kernel.regs[7])<<16|kernel.regs[6])==0x123400+1536);assert(queue.grid[0]==192 && queue.cores.size()==8);
  rpu_launch_silu_mul_exact_strided_spm_kernel(0x1000,0x4000,1,768,1);assert(queue.cores.size()==1);
  reject([&]{rpu_launch_silu_mul_exact_strided_spm_kernel(0x1000,0x4000,0,768,8);});
  reject([&]{rpu_launch_silu_mul_exact_strided_spm_kernel(0x1000,0x4000,65536,768,8);});
  reject([&]{rpu_launch_silu_mul_exact_strided_spm_kernel(0x1000,0x4000,1,32768,8);});
  reject([&]{rpu_launch_silu_mul_exact_strided_spm_kernel(0x1000,0x4000,1,769,8);});
  for(int cores : {-1,0,9,256}) reject([&]{rpu_launch_silu_mul_exact_strided_spm_kernel(0x1000,0x4000,192,768,cores);});
 } else if(test=="sdpa") {
  sdpa_admit(8,true);
  // The exact profile must reject zero cores before num_heads / core_num.
  // UBSan makes a regression in statement order fail deterministically.
  for(int cores : {-1,0,4,9}) reject([&]{sdpa_admit(cores,true);});
  reject([&]{sdpa_admit(8,true,true);});
  reject([&]{sdpa_admit(8,true,false,575);});
  reject([&]{sdpa_admit(8,true,false,576,575);});
  reject([&]{sdpa_admit(8,true,false,576,576,128);});
  reject([&]{sdpa_admit(8,true,false,576,576,64,8);});
  reject([&]{sdpa_admit(8,true,false,576,576,64,16,8);});
  reject([&]{sdpa_admit(8,true,false,576,576,64,16,16,4);});
  reject([&]{sdpa_admit(8,true,false,576,576,64,16,16,8,1);});
  reject([&]{sdpa_admit(8,true,false,576,576,64,16,16,8,0,1);});
  // Ordinary profiles do not inherit the TM160 shape restriction.
  sdpa_admit(8,false,true,1,608,128,16,8);
 } else { assert(false); }
}

'''

    program = (program.replace("__RMS_CONTRACT__", rms_contract)
               .replace("__KERNEL_IDS__", kernel_ids)
               .replace("__SDPA_CONTRACT__", sdpa_contract)
               .replace("__BODIES__", bodies)
               .replace("__SDPA_ADMISSION__", admission))
    directory = tmp_path_factory.mktemp("qwen3vl-spm-wrappers")
    source, binary = directory / "wrappers.cpp", directory / "wrappers"
    source.write_text(with_actual_linear_contracts(program))
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-fsanitize=undefined",
                    "-fno-sanitize-recover=undefined", "-I", str(ROOT / "src/core"), str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize("case", ["rms", "silu", "sdpa"])
def test_native_spm_wrapper_default_and_boundary_contract(spm_wrapper_binary, case):
    subprocess.run([str(spm_wrapper_binary), case], check=True)
