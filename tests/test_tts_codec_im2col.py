"""Exercise the production im2col tile schedule with emulated SPM."""
from pathlib import Path
import subprocess

import torch


def test_im2col_batched_tiles_padding_dilation_and_ownership(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "src/ops/rpu_linear.cpp").read_text()
    function = "Tensor tts_codec_im2col_rows(" + source.split(
        "Tensor tts_codec_im2col_rows(", 1)[1].split("Tensor tts_codec_snake_rows(", 1)[0]
    function = function.replace("at::kPrivateUse1", "at::kCPU")
    harness = r'''
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
using at::Tensor;
constexpr size_t kTtsCodecIm2colBudget=4ULL<<20;
struct RpuKernelGraph {static inline bool active=false;static bool has_active(){return active;}};
int submits=0;
template<typename Emit>void tts_codec_submit_tile(const Tensor&,const Tensor&,Emit emit){
 assert(!RpuKernelGraph::active);RpuKernelGraph::active=true;emit();++submits;RpuKernelGraph::active=false;
}
struct SpmAllocator {
 static inline uint32_t offset=0;
 static inline size_t budget=4ULL<<20;
 struct ScopedTemporary {~ScopedTemporary(){offset=0;}};
 bool is_initialized(){return true;}void init(){}
 size_t free_space(){return budget;}
 uint32_t alloc_temporary(size_t n){auto old=offset;offset+=(n+63)/64*64;assert(offset<=budget);return old;}
 uint32_t addr(int c,uint32_t a){assert(c==0);return a;}
} SPM_ALLOC;
auto memory=at::empty({(4<<20)/2},at::kHalf);
Tensor region(uint32_t a,int64_t n){return memory.narrow(0,a/2,n);}
void rpu_launch_fill_spm_kernel(uint32_t a,int64_t n,c10::Half v,int cores){assert(cores==1);region(a,n).fill_(float(v));}
void rpu_launch_ddr_broadcast_spm_dma_immediate(c10::Half*p,int64_t n,uint32_t a,int cores){assert(cores==1);region(a,n).copy_(at::from_blob(p,{n},at::kHalf));}
void rpu_launch_spm_copy_ddr_dma_immediate(uint32_t a,c10::Half*p,int64_t n){at::from_blob(p,{n},at::kHalf).copy_(region(a,n));}
void rpu_launch_ddr_broadcast_spm_dma(c10::Half*p,int64_t n,uint32_t a,int cores){assert(RpuKernelGraph::active);rpu_launch_ddr_broadcast_spm_dma_immediate(p,n,a,cores);}
void rpu_launch_spm_copy_ddr_dma(uint32_t a,c10::Half*p,int64_t n){assert(RpuKernelGraph::active);rpu_launch_spm_copy_ddr_dma_immediate(a,p,n);}
void rpu_launch_im2col_spm(uint32_t a,uint32_t b,int batch,int h,int w,int c,int kh,int kw,int pt,int pb,int pl,int pr,int sh,int sw,int dh,int dw,int cores){
 assert(batch==1&&h==1&&kh==1&&pt==0&&pb==0&&pl==0&&pr==0&&sh==1&&sw==1&&dh==1&&cores==1);
 int n=w-(kw-1)*dw;
 auto input=region(a,w*c).view({w,c}),output=region(b,n*kw*c).view({n,kw,c});
 for(int k=0;k<kw;++k)output.select(1,k).copy_(input.narrow(0,k*dw,n));
}
'''
    cases = r'''
int main(){
 at::manual_seed(81);at::set_num_threads(2);
 for(int c:{96,256,768,1024})for(int k:{1,3,7})for(int d:{1,3,9})for(bool pad:{false,true}){
  int left=pad?(k-1)*d:0,n=pad?57:4097,samples=n+(k-1)*d-left;
  auto x=at::randn({samples,c}).to(at::kHalf),saved=x.clone();
  auto expected=at::zeros({n,k,c},at::kHalf);
  for(int i=0;i<n;++i)for(int j=0;j<k;++j){int pos=i+j*d-left;if(pos>=0&&pos<samples)expected[i][j].copy_(x[pos]);}
  int before=submits;auto y=tts_codec_im2col_rows(x,k,d,left);
  int tile=std::min<int64_t>(n,((4ULL<<20)-(k-1)*d*c*2-512)/((k+1)*c*2));
  assert(submits-before==(n+tile-1)/tile&&!RpuKernelGraph::active);
  assert(at::equal(y,expected.view({n,k*c}))&&at::equal(x,saved));
  tts_codec_im2col_rows(x+1,k,d,left);assert(at::equal(y,expected.view({n,k*c})));
 }
 auto x=at::ones({9,128},at::kHalf);
 for(int mode:{0,1,2,3}){
  int before=submits;bool rejected=false;RpuKernelGraph::active=mode==0;
  SpmAllocator::budget=mode==3?512:4ULL<<20;
  try{tts_codec_im2col_rows(mode==1?x.to(at::kFloat):x,mode==2?17:7,1,0);}
  catch(const c10::Error&){rejected=true;}
  assert(rejected&&submits==before);RpuKernelGraph::active=false;
 }
}
'''
    cpp, exe = tmp_path / "im2col.cpp", tmp_path / "im2col"
    cpp.write_text(harness + function + cases)
    root = Path(torch.__file__).resolve().parent
    subprocess.run(["g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(root / "include"), str(cpp), "-L", str(root / "lib"),
        "-ltorch_cpu", "-lc10", f"-Wl,-rpath,{root / 'lib'}", "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
