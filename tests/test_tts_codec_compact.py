"""Exercise stage3 device compaction against CPU-backed SPM, including tile tails."""
from pathlib import Path
import subprocess

import torch


def test_stage3_compaction_boundaries_and_owned_output(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "src/ops/rpu_linear.cpp").read_text()
    function = "static Tensor tts_codec_compact_stage3_rows(" + source.split(
        "static Tensor tts_codec_compact_stage3_rows(", 1)[1].split(
        "static Tensor tts_codec_decoder_stage_impl(", 1)[0]
    function = function.replace("at::kPrivateUse1", "at::kCPU")
    harness = r'''
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
using at::Tensor;
int submits=0,barriers=0;
struct RpuKernelGraph {
 static inline bool running=false;
 static bool has_active(){return running;}
 static RpuKernelGraph& active(){assert(running);static RpuKernelGraph g;return g;}
 void record_barrier(int a,int b){assert(!a&&!b);++barriers;}
};
struct SpmAllocator {
 static inline uint32_t offset=0;
 static inline size_t budget=4ULL<<20;
 struct ScopedTemporary {~ScopedTemporary(){offset=0;}};
 bool is_initialized(){return true;}void init(){}
 size_t free_space(){return budget;}
 uint32_t alloc_temporary(size_t n){auto old=offset;offset+=(n+63)/64*64;assert(offset<=budget);return old;}
 uint32_t addr(int c,uint32_t x){assert(!c);return x;}
} SPM_ALLOC;
auto memory=at::empty({(4<<20)/2},at::kHalf);
Tensor region(uint32_t a,int64_t n){assert(a%2==0);return memory.narrow(0,a/2,n);}
template<typename Emit>void tts_codec_submit_tile(const Tensor&,const Tensor&,Emit emit){
 assert(!RpuKernelGraph::running);RpuKernelGraph::running=true;emit();++submits;RpuKernelGraph::running=false;
}
void rpu_launch_ddr_broadcast_spm_dma(c10::Half*p,int64_t n,uint32_t a,int cores){
 assert(cores==1&&RpuKernelGraph::running);region(a,n).copy_(at::from_blob(p,{n},at::kHalf));
}
void rpu_launch_spm_copy_ddr_dma(uint32_t a,c10::Half*p,int64_t n){
 assert(RpuKernelGraph::running);at::from_blob(p,{n},at::kHalf).copy_(region(a,n));
}
void rpu_launch_slice_spm_kernel(uint32_t a,uint32_t b,const std::vector<int64_t>&in,
 const std::vector<int64_t>&out,const std::vector<int64_t>&start,int cores){
 assert(cores==1&&in[1]==256&&out[1]==192&&in[0]==out[0]&&start[0]==0&&start[1]==0);
 region(b,out[0]*out[1]).view(out).copy_(region(a,in[0]*in[1]).view(in).narrow(1,0,192));
}
'''
    cases = r'''
int main(){
 at::manual_seed(21);at::set_num_threads(2);
 for(auto budget:{size_t(4<<20),size_t(1<<20)})for(int count:{1,4095,4096,4097,255*640}){
  SpmAllocator::budget=budget;
  auto x=at::randn({count,256}).to(at::kHalf),saved=x.clone();
  auto expected=x.narrow(1,0,192).contiguous();int before=submits,b=barriers;
  auto y=tts_codec_compact_stage3_rows(x);
  auto tile=std::min<int64_t>(4096,(budget-512)/896);
  assert(submits==before+1&&barriers==b+(count+tile-1)/tile);
  assert(at::equal(y,expected)&&at::equal(x,saved)&&y.is_contiguous()&&y.data_ptr()!=x.data_ptr());
  tts_codec_compact_stage3_rows(x+1);assert(at::equal(y,expected));
 }
 auto x=at::zeros({2,256},at::kHalf);
 for(int bad=0;bad<7;++bad){
  int before=submits;bool rejected=false;RpuKernelGraph::running=bad==0;
  SpmAllocator::budget=bad==6?512:4<<20;
  auto invalid=bad==1?x.to(at::kFloat):bad==2?x.narrow(1,0,192):bad==3?x.t():
    bad==4?x.narrow(0,0,0):bad==5?at::empty({255*640+1,256},at::kHalf):x;
  try{tts_codec_compact_stage3_rows(invalid);}catch(const c10::Error&){rejected=true;}
  assert(rejected&&submits==before);
 }
}
'''
    cpp, exe = tmp_path / "compact.cpp", tmp_path / "compact"
    cpp.write_text(harness + function + cases)
    root = Path(torch.__file__).resolve().parent
    subprocess.run(["g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(root / "include"), str(cpp), "-L", str(root / "lib"),
        "-ltorch_cpu", "-lc10", f"-Wl,-rpath,{root / 'lib'}", "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
