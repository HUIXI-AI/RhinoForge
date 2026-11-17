"""Exercise the codec-only ACC32 one-shot route and operand ownership."""
from pathlib import Path
import subprocess

import torch


def test_codec_linear_route_and_fresh_operands(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "src/ops/rpu_linear.cpp").read_text()
    function = "Tensor tts_codec_linear_rows(" + source.split(
        "Tensor tts_codec_linear_rows(", 1)[1].split("}  // namespace", 1)[0]
    harness = r'''
#include <ATen/ATen.h>
#include <cassert>
using at::Tensor;
int eager=0,batches=0,owners=0;
struct RpuKernelGraph {
 static inline bool enabled=false;
 static bool has_active(){return enabled;}
 static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 void keep_alive(const Tensor&t){assert(enabled&&t.defined());++owners;}
};
template<typename Emit>void tts_codec_submit_tile(const Tensor&,const Tensor&,Emit emit){
 assert(!RpuKernelGraph::enabled);RpuKernelGraph::enabled=true;owners=2;
 emit();++batches;RpuKernelGraph::enabled=false;
}
void check_tts_codec_device_half(const Tensor&t,c10::IntArrayRef s,const char*){
 TORCH_CHECK(t.scalar_type()==at::kHalf&&t.is_contiguous()&&t.sizes().equals(s));
}
void validate_parallel_linear_geometry(int64_t m,int64_t n,int64_t k,int p,int c){
 TORCH_CHECK(m>0&&n>0&&n%128==0&&k>0&&k%16==0&&p==1&&c==8);
}
uint64_t RpuGetDevAddr(void*p){return reinterpret_cast<uint64_t>(p);}
Tensor reference(const Tensor&x,const Tensor&w,const c10::optional<Tensor>&b){
 auto y=at::mm(x.to(at::kFloat),w.to(at::kFloat).t());
 if(b)y.add_(b->to(at::kFloat));return y.to(at::kHalf);
}
Tensor rpu_linear_with_accumulation(const Tensor&x,const Tensor&w,const c10::optional<Tensor>&b,
 int p,int c,bool acc,const c10::optional<Tensor>&s){
 assert(!RpuKernelGraph::enabled&&x.size(0)==1&&p==1&&c==8&&acc&&!s);++eager;return reference(x,w,b);
}
void rpu_launch_linear_ddr_kernel(const Tensor&x,const Tensor&w,Tensor&y,const Tensor&b,
 bool has,int p,const uint64_t*dst,int64_t off,const uint64_t*bp,int c,const Tensor&s,bool acc){
 assert(RpuKernelGraph::enabled&&owners==(has?4:3)&&p==1&&c==8&&acc&&!s.defined());
 assert(dst&&*dst==RpuGetDevAddr(y.data_ptr())&&off==0&&!bp);
 y.copy_(reference(x,w,has?c10::optional<Tensor>(b):c10::nullopt));
}
'''
    cases = r'''
int main(){
 at::manual_seed(83);
 for(int n:{1,2,257,1025})for(bool bias:{false,true}){
  auto x=at::randn({n,16}).to(at::kHalf),w=at::randn({128,16}).to(at::kHalf),b=at::randn({128}).to(at::kHalf);
  auto optional=bias?c10::optional<Tensor>(b):c10::nullopt;
  int before=batches,old_eager=eager;
  auto y=tts_codec_linear_rows(x,w,optional),saved=y.clone();
  assert(at::equal(y,reference(x,w,optional))&&batches-before==(n!=1)&&eager-old_eager==(n==1));
  x.add_(1);w.add_(1);b.add_(1);
  assert(at::equal(tts_codec_linear_rows(x,w,optional),reference(x,w,optional))&&at::equal(y,saved));
 }
 auto x=at::ones({2,16},at::kHalf),w=at::ones({128,16},at::kHalf);
 for(int mode=0;mode<5;++mode){
  int before=batches;bool rejected=false;RpuKernelGraph::enabled=mode==0;
  try{tts_codec_linear_rows(mode==1?x.narrow(0,0,0):mode==2?x.to(at::kFloat):x,
       mode==3?w.narrow(0,0,127):w,mode==4?c10::optional<Tensor>(at::ones({1},at::kHalf)):c10::nullopt);}
  catch(const c10::Error&){rejected=true;}
  assert(rejected&&batches==before);RpuKernelGraph::enabled=false;
 }
}
'''
    cpp, exe = tmp_path / "linear.cpp", tmp_path / "linear"
    cpp.write_text(harness + function + cases)
    root = Path(torch.__file__).resolve().parent
    subprocess.run(["g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(root / "include"), str(cpp), "-L", str(root / "lib"),
        "-ltorch_cpu", "-lc10", f"-Wl,-rpath,{root / 'lib'}", "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
