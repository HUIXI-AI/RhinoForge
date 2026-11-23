"""Execute the production SnakeBeta tile schedule against emulated SPM."""
from pathlib import Path
import subprocess

import torch


def test_snake_resident_tiles_tail_and_ownership(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "src/ops/rpu_linear.cpp").read_text()
    function = "Tensor tts_codec_snake_rows(" + source.split(
        "Tensor tts_codec_snake_rows(", 1)[1].split("Tensor tts_codec_layernorm_rows(", 1)[0]
    function += "Tensor tts_codec_bias_rows(" + source.split(
        "Tensor tts_codec_bias_rows(", 1)[1].split("Tensor tts_codec_linear_rows(", 1)[0]
    function = function.replace("at::kPrivateUse1", "at::kCPU")
    harness = r'''
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
using at::Tensor;
bool private_batch=false;
bool tts_codec_private_batch_active(){return private_batch;}
constexpr size_t kTtsCodecIm2colBudget=4ULL<<20;
enum class ValuOpType {ADD,MUL,SIN};
enum class GeluMode {NONE};
int submissions=0,broadcasts=0;
struct RpuKernelGraph {
 static bool has_active(){return false;}
 static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 void keep_alive(const Tensor&t){assert(t.defined());}
};
void check_tts_codec_device_half(const Tensor&t,c10::IntArrayRef shape,const char*){
 TORCH_CHECK(t.scalar_type()==at::kHalf&&t.is_contiguous()&&t.sizes().equals(shape));
}
struct SpmAllocator {
 static inline uint32_t offset=0;
 static inline size_t budget=4ULL<<20;
 struct ScopedTemporary {~ScopedTemporary(){offset=0;}};
 bool is_initialized(){return true;}void init(){}
 size_t free_space(){return budget;}
 uint32_t alloc_temporary(size_t n){auto old=offset;offset+=(n+63)/64*64;assert(offset<free_space());return old;}
 uint32_t addr(int c,uint32_t a){assert(c==0);return a;}
} SPM_ALLOC;
auto memory=at::empty({8,(4<<20)/2},at::kHalf);
Tensor region(int c,uint32_t a,int64_t n){return memory[c].narrow(0,a/2,n);}
template<typename Emit>void tts_codec_submit_tile(const Tensor&,const Tensor&,Emit emit){emit();++submissions;}
void rpu_launch_ddr_broadcast_spm_dma(c10::Half*p,int64_t n,uint32_t a,int cores){
 ++broadcasts;
 for(int c=0;c<cores;++c)region(c,a,n).copy_(at::from_blob(p,{n},at::kHalf));
}
void rpu_launch_ddr_scatter_spm_dma(c10::Half*p,int64_t n,int64_t stride,uint32_t a,int cores){
 for(int c=0;c<cores;++c)region(c,a,n).copy_(at::from_blob(p+c*stride/2,{n},at::kHalf));
}
void rpu_launch_spm_scatter_ddr_dma(uint32_t a,c10::Half*p,int64_t n,int64_t stride,int cores){
 for(int c=0;c<cores;++c)at::from_blob(p+c*stride/2,{n},at::kHalf).copy_(region(c,a,n));
}
void rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(uint32_t a,uint32_t b,uint32_t o,int64_t n,int64_t k,c10::Half,ValuOpType op,bool,int cores){
 for(int c=0;c<cores;++c){auto rows=region(c,b,n*k).view({n,k}),bias=region(c,a,k);
 region(c,o,n*k).view({n,k}).copy_(op==ValuOpType::MUL?rows*bias:rows+bias);}
}
void rpu_launch_eltwise_unary_spm_kernel(uint32_t a,uint32_t b,int64_t n,ValuOpType op,GeluMode,int cores){
 assert(op==ValuOpType::SIN);for(int c=0;c<cores;++c)region(c,b,n).copy_(at::sin(region(c,a,n)));
}
void rpu_launch_eltwise_binary_spm_kernel(uint32_t a,uint32_t b,uint32_t o,int64_t n,ValuOpType op,c10::Half,int cores){
 for(int c=0;c<cores;++c)region(c,o,n).copy_(op==ValuOpType::MUL?region(c,a,n)*region(c,b,n):region(c,a,n)+region(c,b,n));
}
'''
    cases = r'''
int main(){
 at::manual_seed(19);at::set_num_threads(2);
 for(int c:{96,128,192,256,384,768,1536})for(int n:{1,7,9,1025,4105,32769}){
  if(n==32769&&c!=128)continue;
  auto x=at::randn({n,c}).to(at::kHalf),saved=x.clone();
  auto a=at::randn({c}).to(at::kHalf),s=at::randn({c}).to(at::kHalf);
  auto v=at::sin(x*a),expected=x+(v*v)*s;
  int before=submissions,uploads=broadcasts;auto y=tts_codec_snake_rows(x,a,s);
  int tile=std::min<int64_t>(512,((4ULL<<20)-4*c-512)/(6*c)),tiles=0;
  for(int start=0;start<n;++tiles){int cores=std::min(8,n-start);start+=cores*std::min(tile,(n-start)/cores);}
  assert(submissions-before==(tiles+7)/8&&broadcasts-uploads==2&&at::equal(y,expected)&&at::equal(x,saved));
  tts_codec_snake_rows(x+1,a,s);assert(at::equal(y,expected));
  a.add_(1);v=at::sin(x*a);expected=x+(v*v)*s;
  assert(at::equal(tts_codec_snake_rows(x,a,s),expected));
 }
 for(bool batch:{false,true})for(int c:{128,256,768})for(int n:{1,9,4097,32768}){
  private_batch=batch;
  auto x=at::randn({n,c}).to(at::kHalf),b=at::randn({c}).to(at::kHalf),saved=x.clone();
  auto expected=x+b,y=tts_codec_bias_rows(x,b);
  assert(at::equal(y,expected)&&at::equal(x,saved));
  b.add_(1);auto fresh=tts_codec_bias_rows(x+1,b);
  assert(at::equal(fresh,x+1+b)&&at::equal(y,expected));
 }
 private_batch=false;
 auto x=at::ones({9,128},at::kHalf),a=at::ones({128},at::kHalf);
 for(int mode:{0,1,2,3}){
  int before=submissions;bool rejected=false;
  SpmAllocator::budget=mode==3?512:4ULL<<20;
  try{tts_codec_snake_rows(mode==0?x.to(at::kFloat):mode==1?x.narrow(0,0,0):x,
                          mode==2?a.narrow(0,0,127):a,a);}
  catch(const c10::Error&){rejected=true;}
  assert(rejected&&submissions==before);
 }
}
'''
    cpp, exe = tmp_path / "snake.cpp", tmp_path / "snake"
    cpp.write_text(harness + function + cases)
    root = Path(torch.__file__).resolve().parent
    subprocess.run(["g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(root / "include"), str(cpp), "-L", str(root / "lib"),
        "-ltorch_cpu", "-lc10", f"-Wl,-rpath,{root / 'lib'}", "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
