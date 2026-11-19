"""Execute the production bounded overlap schedule against an emulated SPM."""
from pathlib import Path
import subprocess

import torch


def test_tile_graph_completes_before_retirement_and_aborts(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "src/ops/rpu_linear.cpp").read_text()
    helper = "template <typename Emit>" + source.split(
        "template <typename Emit>", 1)[1].split("Tensor tts_codec_cpu_nct_to_device_rows(", 1)[0]
    harness = r'''
#include <cassert>
#include <stdexcept>
struct Tensor{};
struct RpuKernelGraph{
 static inline bool active=false,fail_end=false;
 static inline int owners=0,ends=0,aborts=0;
 void begin(){assert(!active);active=true;owners=0;}
 void keep_alive(const Tensor&){assert(active);++owners;}
 void end(){assert(active&&owners==2);if(fail_end)throw std::runtime_error("end");++ends;active=false;}
 void abort(){assert(active);++aborts;active=false;}
 ~RpuKernelGraph(){assert(!active);}
};
'''
    cases = r'''
int main(){
 Tensor x,y;
 tts_codec_submit_tile(x,y,[]{assert(RpuKernelGraph::active);});
 assert(RpuKernelGraph::ends==1);
 try{tts_codec_submit_tile(x,y,[]{throw std::runtime_error("emit");});assert(false);}catch(const std::runtime_error&){}
 RpuKernelGraph::fail_end=true;
 try{tts_codec_submit_tile(x,y,[]{});assert(false);}catch(const std::runtime_error&){}
 assert(RpuKernelGraph::aborts==2&&!RpuKernelGraph::active);
}
'''
    cpp, exe = tmp_path / "tile.cpp", tmp_path / "tile"
    cpp.write_text(harness + helper + cases)
    subprocess.run(["g++", "-std=c++17", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_overlap_tiles_boundaries_and_ownership(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "src/ops/rpu_linear.cpp").read_text()
    function = "Tensor rpu_tts_codec_transpose_overlap(" + source.split(
        "Tensor rpu_tts_codec_transpose_overlap(", 1)[1].split(
        "void rpu_linear_into(", 1)[0]
    function = function.replace("at::kPrivateUse1", "at::kCPU")
    harness = r'''
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
using at::Tensor;
enum class ValuOpType {ADD};
struct RpuKernelGraph {static inline bool active=false;static bool has_active(){return active;}};
int tile_submits=0;
template<typename Emit>void tts_codec_submit_tile(const Tensor&,const Tensor&,Emit emit){
 assert(!RpuKernelGraph::active);RpuKernelGraph::active=true;
 try{emit();++tile_submits;RpuKernelGraph::active=false;}catch(...){RpuKernelGraph::active=false;throw;}
}
struct SpmAllocator {
 static inline uint32_t offset=0;
 struct ScopedTemporary {~ScopedTemporary(){offset=0;}};
 bool is_initialized(){return true;} void init(){}
 size_t free_space(){return 4ULL<<20;}
 uint32_t alloc_temporary(size_t n){auto old=offset;offset+=(n+63)/64*64;assert(offset<free_space());return old;}
 uint32_t addr(int c,uint32_t x){assert(c==0);return x;}
} SPM_ALLOC;
auto memory=at::empty({(4<<20)/2},at::kHalf);
Tensor region(uint32_t a,int64_t n){assert(a%2==0);return memory.narrow(0,a/2,n);}
void rpu_launch_fill_spm_kernel(uint32_t a,int64_t n,c10::Half v,int cores){assert(cores==1);region(a,n).fill_(float(v));}
void rpu_launch_ddr_broadcast_spm_dma_immediate(c10::Half*p,int64_t n,uint32_t a,int cores){assert(cores==1);region(a,n).copy_(at::from_blob(p,{n},at::kHalf));}
void rpu_launch_spm_copy_ddr_dma_immediate(uint32_t a,c10::Half*p,int64_t n){at::from_blob(p,{n},at::kHalf).copy_(region(a,n));}
void rpu_launch_ddr_broadcast_spm_dma(c10::Half*p,int64_t n,uint32_t a,int cores){assert(RpuKernelGraph::active);rpu_launch_ddr_broadcast_spm_dma_immediate(p,n,a,cores);}
void rpu_launch_spm_copy_ddr_dma(uint32_t a,c10::Half*p,int64_t n){assert(RpuKernelGraph::active);rpu_launch_spm_copy_ddr_dma_immediate(a,p,n);}
void rpu_launch_slice_spm_kernel(uint32_t a,uint32_t b,const std::vector<int64_t>&in,const std::vector<int64_t>&out,const std::vector<int64_t>&start,int cores){
 assert(cores==1&&in.size()==2&&in[0]==out[0]&&start[0]==0);
 region(b,out[0]*out[1]).view(out).copy_(region(a,in[0]*in[1]).view(in).narrow(1,start[1],out[1]));
}
void rpu_launch_eltwise_binary_spm_kernel(uint32_t a,uint32_t b,uint32_t o,int64_t n,ValuOpType,c10::Half,int cores){assert(cores==1);region(o,n).copy_(region(a,n)+region(b,n));}
void rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(uint32_t a,uint32_t b,uint32_t o,int64_t n,int64_t c,c10::Half,ValuOpType,bool,int cores){assert(cores==1);region(o,n*c).view({n,c}).copy_(region(b,n*c).view({n,c})+region(a,c));}
'''
    cases = r'''
int main(){
 at::manual_seed(12);at::set_num_threads(2);
 for(auto geometry:{std::pair<int,int>{1024,2},{768,16},{384,10},{192,8},{96,6}}){
  auto [c,k]=geometry;int s=k==2?2:k/2,p=(c+127)/128*128;
  for(int count:{1,16,17,56,57,512,513})for(bool use_bias:{false,true}){
   auto x=at::randn({count,k*p}).to(at::kHalf),saved=x.clone();
   auto bias=at::randn({p}).to(at::kHalf);
   auto expected=at::zeros({(count+(k>s))*s,p},at::kHalf);
   for(int i=0;i<count;++i)expected.narrow(0,i*s,k).add_(x[i].view({k,p}));
   if(use_bias)expected.add_(bias);
   int before=tile_submits;
   auto y=rpu_tts_codec_transpose_overlap(x,use_bias?c10::optional<Tensor>(bias):c10::nullopt,c,k,s);
   int groups=count+(k>s),block=s*p;
   int tile=std::min<int64_t>(std::min(512,groups),((4ULL<<20)-4*block-2*p-512)/(12*block));
   assert(tile_submits-before==(groups+8*tile-1)/(8*tile)&&!RpuKernelGraph::active);
   assert(at::equal(y,expected)&&at::equal(x,saved)&&y.data_ptr()!=x.data_ptr());
   rpu_tts_codec_transpose_overlap(x+1,c10::nullopt,c,k,s);
   assert(at::equal(y,expected));
  }
 }
 auto x=at::zeros({3,768},at::kHalf);
 for(int mode:{0,1,2}){
  bool rejected=false;
  RpuKernelGraph::active=mode==0;
  try{rpu_tts_codec_transpose_overlap(mode==1?x.to(at::kFloat):x,c10::nullopt,mode==2?95:96,6,3);}
  catch(const c10::Error&){rejected=true;}
  assert(rejected);
 }
}
'''
    cpp, exe = tmp_path / "overlap.cpp", tmp_path / "overlap"
    cpp.write_text(harness + function + cases)
    root = Path(torch.__file__).resolve().parent
    subprocess.run([
        "g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(root / "include"), str(cpp), "-L", str(root / "lib"),
        "-ltorch_cpu", "-lc10", f"-Wl,-rpath,{root / 'lib'}", "-o", str(exe),
    ], check=True)
    subprocess.run([str(exe)], check=True)
