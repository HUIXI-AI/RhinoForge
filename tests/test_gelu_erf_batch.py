"""Check bounded ERF GELU batches with the production host schedule."""
from pathlib import Path
import subprocess

import torch


def test_gelu_batches_tail_ownership_and_abort(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "src/ops/rpu_eltwise_unary.cpp").read_text()
    function = "at::Tensor rpu_gelu_erf(" + source.split(
        "at::Tensor rpu_gelu_erf(", 1)[1].split("void rpu_require_high_precision_math_kernels(", 1)[0]
    function = function.replace("at::kPrivateUse1", "at::kCPU")
    harness = r'''
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
enum class ValuOpType {ADD};enum class GeluMode {ERF};
int submissions=0,owners=0,aborts=0;bool fail=false;
namespace graph_dma {bool recording=false;bool active(){return recording;}}
struct RpuKernelGraph {
 void begin(){assert(!graph_dma::recording);graph_dma::recording=true;owners=0;}
 void keep_alive(const at::Tensor&){assert(graph_dma::recording);++owners;}
 void end(){assert(owners==2);++submissions;graph_dma::recording=false;}
 void abort(){++aborts;graph_dma::recording=false;}
 ~RpuKernelGraph(){assert(!graph_dma::recording);}
};
struct SpmAllocator {
 struct ScopedTemporary{};bool is_initialized(){return true;}void init(){}
 uint32_t alloc_temporary(size_t n){assert(n<=51200);return 0;}
 uint32_t addr(int c,uint32_t a){assert(c==0);return a;}
} SPM_ALLOC;
auto memory=at::empty({25600},at::kHalf);
void rpu_launch_ddr_broadcast_spm_dma_immediate(c10::Half*p,int64_t n,uint32_t a,int c){assert(a==0&&c==1);memory.narrow(0,0,n).copy_(at::from_blob(p,{n},at::kHalf));}
void rpu_launch_spm_copy_ddr_dma_immediate(uint32_t a,c10::Half*p,int64_t n){assert(a==0);at::from_blob(p,{n},at::kHalf).copy_(memory.narrow(0,0,n));}
void rpu_launch_ddr_broadcast_spm_dma(c10::Half*p,int64_t n,uint32_t a,int c){assert(graph_dma::active());rpu_launch_ddr_broadcast_spm_dma_immediate(p,n,a,c);}
void rpu_launch_spm_copy_ddr_dma(uint32_t a,c10::Half*p,int64_t n){assert(graph_dma::active());rpu_launch_spm_copy_ddr_dma_immediate(a,p,n);}
void rpu_launch_eltwise_unary_spm_kernel(uint32_t a,uint32_t b,int64_t n,ValuOpType,GeluMode,int c){
 TORCH_CHECK(!fail,"injected");assert(a==0&&b==0&&c==1&&n<=25600);
 memory.narrow(0,0,n).copy_(at::gelu(memory.narrow(0,0,n).to(at::kFloat)).to(at::kHalf));
}
'''
    cases = r'''
int main(){
 at::set_num_threads(2);
 for(int n:{8,25600,25608,204800,204808,409608}){
  auto x=at::linspace(-5,5,n).to(at::kHalf),saved=x.clone();
  auto expected=at::gelu(x.to(at::kFloat)).to(at::kHalf);int before=submissions;
  auto y=rpu_gelu_erf(x);assert(submissions-before==(n+204799)/204800);
  assert(at::equal(y,expected)&&at::equal(x,saved));
  rpu_gelu_erf(-x);assert(at::equal(y,expected));
 }
 for(int mode=0;mode<5;++mode){
  auto x=at::ones({8},at::kHalf);bool rejected=false;int before=submissions;
  graph_dma::recording=mode==0;fail=mode==4;
  try{rpu_gelu_erf(mode==1?x.to(at::kFloat):mode==2?x.narrow(0,0,7):mode==3?x.narrow(0,0,0):x);}
  catch(const c10::Error&){rejected=true;}
  assert(rejected&&submissions==before);graph_dma::recording=false;
 }
 assert(aborts==1);
}
'''
    cpp, exe = tmp_path / "gelu.cpp", tmp_path / "gelu"
    cpp.write_text(harness + function + cases)
    root = Path(torch.__file__).resolve().parent
    subprocess.run(["g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(root / "include"), str(cpp), "-L", str(root / "lib"),
        "-ltorch_cpu", "-lc10", f"-Wl,-rpath,{root / 'lib'}", "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
