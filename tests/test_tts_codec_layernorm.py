"""Check bounded LayerNorm batches without changing the 16-row math tiles."""
from pathlib import Path
import subprocess

import torch


def test_layernorm_batches_tail_freshness_and_ownership(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "src/ops/rpu_linear.cpp").read_text()
    function = "Tensor tts_codec_layernorm_rows(" + source.split(
        "Tensor tts_codec_layernorm_rows(", 1)[1].split("void check_tts_codec_device_half(", 1)[0]
    function = function.replace("at::kPrivateUse1", "at::kCPU")
    harness = r'''
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
using at::Tensor;
constexpr int64_t kTtsCodecLayerNormRows=16;
struct RpuKernelGraph {static inline bool active=false;static bool has_active(){return active;}};
int submits=0,uploads=0;
template<typename Emit>void tts_codec_submit_tile(const Tensor&,const Tensor&,Emit emit){
 assert(!RpuKernelGraph::active);RpuKernelGraph::active=true;emit();++submits;RpuKernelGraph::active=false;
}
struct SpmAllocator {
 static inline uint32_t offset=0;
 struct ScopedTemporary {~ScopedTemporary(){offset=0;}};
 bool is_initialized(){return true;}void init(){}
 uint32_t alloc_temporary(size_t n){auto old=offset;offset+=(n+63)/64*64;assert(offset<70000);return old;}
 uint32_t addr(int c,uint32_t a){assert(c==0);return a;}
} SPM_ALLOC;
auto memory=at::empty({35000},at::kHalf);
Tensor region(uint32_t a,int64_t n){return memory.narrow(0,a/2,n);}
void rpu_launch_ddr_broadcast_spm_dma_immediate(c10::Half*p,int64_t n,uint32_t a,int cores){assert(cores==1);++uploads;region(a,n).copy_(at::from_blob(p,{n},at::kHalf));}
void rpu_launch_spm_copy_ddr_dma_immediate(uint32_t a,c10::Half*p,int64_t n){at::from_blob(p,{n},at::kHalf).copy_(region(a,n));}
void rpu_launch_ddr_broadcast_spm_dma(c10::Half*p,int64_t n,uint32_t a,int cores){assert(RpuKernelGraph::active);rpu_launch_ddr_broadcast_spm_dma_immediate(p,n,a,cores);}
void rpu_launch_spm_copy_ddr_dma(uint32_t a,c10::Half*p,int64_t n){assert(RpuKernelGraph::active);rpu_launch_spm_copy_ddr_dma_immediate(a,p,n);}
void rpu_launch_layernorm_spm_kernel(uint32_t a,uint32_t o,uint32_t w,uint32_t b,int64_t n,int64_t c,double eps,bool rms,uint32_t,int cores){
 assert(n>0&&n<=16&&!rms&&cores==1);
 region(o,n*c).view({n,c}).copy_(at::layer_norm(region(a,n*c).view({n,c}).to(at::kFloat),{c},region(w,c).to(at::kFloat),region(b,c).to(at::kFloat),eps));
}
'''
    cases = r'''
int main(){
 at::manual_seed(82);at::set_num_threads(2);
 for(int n:{1,15,16,17,255,256,257,1025}){
  int c=1024;auto x=at::randn({n,c}).to(at::kHalf),saved=x.clone();
  auto w=at::randn({c}).to(at::kHalf),b=at::randn({c}).to(at::kHalf);
  auto expected=at::layer_norm(x.to(at::kFloat),{c},w.to(at::kFloat),b.to(at::kFloat),1e-6).to(at::kHalf);
  int before=submits,up=uploads;auto y=tts_codec_layernorm_rows(x,w,b,1e-6);
  assert(submits-before==(n+255)/256&&uploads-up==2+(n+15)/16);
  assert(at::equal(y,expected)&&at::equal(x,saved));
  w.add_(1);expected=at::layer_norm(x.to(at::kFloat),{c},w.to(at::kFloat),b.to(at::kFloat),1e-6).to(at::kHalf);
  auto retained=y.clone();assert(at::equal(tts_codec_layernorm_rows(x,w,b,1e-6),expected)&&at::equal(y,retained));
 }
 auto x=at::ones({1,1024},at::kHalf),w=at::ones({1024},at::kHalf);
 for(bool nested:{false,true}){
  int before=submits;bool rejected=false;RpuKernelGraph::active=nested;
  try{tts_codec_layernorm_rows(nested?x:x.narrow(0,0,0),w,w,1e-6);}
  catch(const c10::Error&){rejected=true;}
  assert(rejected&&submits==before&&SpmAllocator::offset==0);RpuKernelGraph::active=false;
 }
}
'''
    cpp, exe = tmp_path / "layernorm.cpp", tmp_path / "layernorm"
    cpp.write_text(harness + function + cases)
    root = Path(torch.__file__).resolve().parent
    subprocess.run(["g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(root / "include"), str(cpp), "-L", str(root / "lib"),
        "-ltorch_cpu", "-lc10", f"-Wl,-rpath,{root / 'lib'}", "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
