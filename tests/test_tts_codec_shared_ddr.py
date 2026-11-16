"""Production codec staging with real ATen tensors and an emulated device boundary."""
from pathlib import Path
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[1]


def test_complete_residual_keeps_legacy_affine_contract(tmp_path):
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    upload = "Tensor tts_codec_cpu_nct_to_device_rows(" + source.split(
        "Tensor tts_codec_cpu_nct_to_device_rows(", 1)[1].split(
        "Tensor tts_codec_device_rows_to_cpu_nct(", 1)[0]
    function = "static Tensor tts_codec_residual_unit_impl(" + source.split(
        "static Tensor tts_codec_residual_unit_impl(", 1)[1].split(
        "Tensor rpu_tts_codec_snake(", 1)[0]
    harness = r'''
#include <ATen/ATen.h>
#include <cassert>
using at::Tensor;
int64_t CeilDiv(int64_t x,int64_t y){return (x+y-1)/y;}
struct RpuKernelGraph{static bool has_active(){return false;}};
void check_tts_codec_device_half(const Tensor&t,c10::IntArrayRef shape,const char*){
 TORCH_CHECK(t.scalar_type()==at::kHalf && t.is_contiguous() && t.sizes().equals(shape));
}
Tensor rpu_to_cpu_zerocopy_no_graph_sync(const Tensor&t,bool flush){assert(!flush);return t;}
void rpu_ddr_flush_force_sized(void*,size_t){}
Tensor tts_codec_device_rows_to_cpu_nct(const Tensor&t){return t.t().unsqueeze(0).to(at::kFloat).contiguous();}
Tensor tts_codec_snake_rows(const Tensor&x,const Tensor&a,const Tensor&s){auto v=at::sin(x*a);return x+(v*v)*s;}
Tensor tts_codec_im2col_rows(const Tensor&x,int64_t,int64_t d,int64_t pad){
 auto y=at::constant_pad_nd(x.t(),{pad,0},0);
 return y.unfold(1,6*d+1,1).slice(2,0,6*d+1,d).permute({1,0,2}).reshape({x.size(0),x.size(1)*7}).contiguous();
}
Tensor rpu_linear_with_accumulation(const Tensor&x,const Tensor&w,const c10::optional<Tensor>&,
 int64_t,int64_t,bool,const c10::optional<Tensor>&){return at::mm(x.to(at::kFloat),w.to(at::kFloat).t()).to(at::kHalf);}
'''
    cases = r'''
int main(){
 at::manual_seed(73);
 for(int c:{96,192}){
  int n=CeilDiv(c,128)*128;
  auto x=at::randn({1,c,3});auto saved=x.clone();
  auto w1=at::zeros({n,c*7},at::kHalf),w2=at::zeros({n,n},at::kHalf);
  auto a=at::zeros({6,n},at::kHalf);a.select(0,4).fill_(1);
  auto upload=tts_codec_cpu_nct_to_device_rows(x,w1,n);
  assert(at::equal(upload.narrow(1,0,c),x.squeeze(0).t().to(at::kHalf)));
  assert(at::count_nonzero(upload.narrow(1,c,n-c)).item<int64_t>()==0);
  auto y=rpu_tts_codec_residual_unit_complete(x,w1,a,w2,9,c);
  assert(at::equal(y,x.to(at::kHalf).to(at::kFloat)) && at::equal(x,saved));
  auto old=rpu_tts_codec_residual_unit(x,w1,a.narrow(0,0,4),w2,9,c);
  assert(at::equal(old,at::zeros_like(x)));
  bool rejected=false;
  try{rpu_tts_codec_residual_unit(x,w1,a,w2,9,c);}catch(const c10::Error&){rejected=true;}
  assert(rejected);rejected=false;
  try{rpu_tts_codec_residual_unit_complete(x,w1,a.narrow(0,0,4),w2,9,c);}catch(const c10::Error&){rejected=true;}
  assert(rejected);
  rpu_tts_codec_residual_unit_complete(x+1,w1,a,w2,9,c);
  assert(at::equal(y,saved.to(at::kHalf).to(at::kFloat)));
  w1.copy_(at::randn_like(w1)/64);w2.copy_(at::randn_like(w2)/16);
  a.copy_(at::randn_like(a)/10);a.select(0,4).add_(1);
  auto rows=tts_codec_cpu_nct_to_device_rows(x,w1);
  auto h=tts_codec_snake_rows(rows,a.select(0,4).narrow(0,0,c),a.select(0,5).narrow(0,0,c));
  auto patches=tts_codec_im2col_rows(h,7,3,18);
  h=at::mm(patches.to(at::kFloat),w1.to(at::kFloat).t()).to(at::kHalf)+a.select(0,0);
  h=tts_codec_snake_rows(h,a.select(0,1),a.select(0,2));
  h=at::mm(h.to(at::kFloat),w2.to(at::kFloat).t()).to(at::kHalf)+a.select(0,3);
  auto expected=tts_codec_device_rows_to_cpu_nct(h.narrow(1,0,c).contiguous()+rows);
  auto actual=rpu_tts_codec_residual_unit_complete(x,w1,a,w2,3,c);
  assert(at::equal(actual,expected));
 }
}
'''
    cpp, exe = tmp_path / "residual.cpp", tmp_path / "residual"
    cpp.write_text(harness + upload + function + cases)
    torch_root = Path(torch.__file__).resolve().parent
    subprocess.run([
        "g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(torch_root / "include"), str(cpp),
        "-L", str(torch_root / "lib"), "-ltorch_cpu", "-lc10",
        f"-Wl,-rpath,{torch_root / 'lib'}", "-o", str(exe),
    ], check=True)
    subprocess.run([str(exe)], check=True)


def test_codec_staging_conversion_layout_lifetime_and_rejections(tmp_path):
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    function = source.split("Tensor rpu_tts_codec_conv1d(", 1)[1].split("Tensor rpu_tts_codec_conv_pair(", 1)[0]
    function = "Tensor rpu_tts_codec_conv1d(" + function
    assert function.count("weight.device().type() == at::kPrivateUse1") == 1
    function = function.replace("weight.device().type() == at::kPrivateUse1", "weight.device().type() == at::kCPU")
    harness = r'''
#include <ATen/ATen.h>
#include <cassert>
using at::Tensor;
struct RpuKernelGraph {static inline bool captured=false;static bool has_active(){return captured;}};
int calls=0,flushes=0;
Tensor expected_patches;
Tensor rpu_to_cpu_zerocopy_no_graph_sync(const Tensor&t,bool flush){assert(!flush);return t;}
Tensor rpu_to_cpu_zerocopy(const Tensor&t){++flushes;return t;}
void rpu_ddr_flush_force_sized(void*,size_t){++flushes;}
Tensor rpu_linear_with_accumulation(const Tensor&x,const Tensor&w,const c10::optional<Tensor>&b,
 int64_t partition,int64_t cores,bool acc32,const c10::optional<Tensor>&scale){
 assert(!b&&partition==1&&cores==8&&acc32&&!scale);
 assert(x.is_contiguous()&&x.scalar_type()==at::kHalf&&at::equal(x,expected_patches));
 ++calls;return at::mm(x.to(at::kFloat),w.to(at::kFloat).t()).to(at::kHalf);
}
template<class F>void rejects(F fn){int before=calls;bool caught=false;try{fn();}catch(const c10::Error&){caught=true;}assert(caught&&before==calls);}
'''
    cases = r'''
int main(){
 at::manual_seed(19);at::set_num_threads(2);
 for(int kernel:{1,7})for(int dilation:{1,3,9})for(int channels_out:{1,96})for(bool has_bias:{false,true}){
  if(kernel==1&&(dilation!=1||channels_out==1))continue;
  if(channels_out==1&&dilation!=1)continue;
  int width=(kernel-1)*dilation+1,count=11;
  auto base=at::randn({1,96,2*(count+width-1)+2});
  auto input=base.slice(2,1,base.size(2)-1,2);auto saved=input.clone();
  auto weight=at::randn({128,96*kernel}).to(at::kHalf)/8;
  auto bias=at::randn({channels_out});
  expected_patches=input.to(at::kHalf).unfold(2,width,1).slice(3,0,width,dilation)
      .permute({0,2,1,3}).reshape({count,96*kernel}).contiguous();
  auto expected=at::mm(expected_patches.to(at::kFloat),weight.to(at::kFloat).t()).to(at::kHalf)
      .to(at::kFloat).narrow(1,0,channels_out);
  if(has_bias)expected.add_(bias);
  expected=expected.t().unsqueeze(0).contiguous();
  auto output=rpu_tts_codec_conv1d(input,weight,has_bias?c10::optional<Tensor>(bias):c10::nullopt,kernel,dilation,channels_out);
  assert(at::equal(output,expected)&&output.is_contiguous()&&output.scalar_type()==at::kFloat);
  assert(at::equal(input,saved));
  auto changed=input+1;
  expected_patches=changed.to(at::kHalf).unfold(2,width,1).slice(3,0,width,dilation)
      .permute({0,2,1,3}).reshape({count,96*kernel}).contiguous();
  auto next=rpu_tts_codec_conv1d(changed,weight,c10::nullopt,kernel,dilation,channels_out);
  assert(at::equal(output,expected)&&!at::equal(output,next));
  rejects([&]{rpu_tts_codec_conv1d(input.to(at::kHalf),weight,bias,kernel,dilation,channels_out);});
  rejects([&]{rpu_tts_codec_conv1d(input,weight.to(at::kFloat),bias,kernel,dilation,channels_out);});
  rejects([&]{rpu_tts_codec_conv1d(input,weight,bias,2,1,channels_out);});
  rejects([&]{rpu_tts_codec_conv1d(input,weight,bias,kernel,dilation,3);});
  rejects([&]{rpu_tts_codec_conv1d(input,weight,bias.to(at::kHalf),kernel,dilation,channels_out);});
  RpuKernelGraph::captured=true;
  rejects([&]{rpu_tts_codec_conv1d(input,weight,bias,kernel,dilation,channels_out);});
  RpuKernelGraph::captured=false;
 }
 auto weight=at::zeros({128,672},at::kHalf);
 rejects([&]{rpu_tts_codec_conv1d(at::zeros({1,96,6}),weight,c10::nullopt,7,1,96);});
 assert(calls>0&&flushes==calls*2);
}
'''
    cpp, exe = tmp_path / "codec.cpp", tmp_path / "codec"
    cpp.write_text(harness + function + cases)
    torch_root = Path(torch.__file__).resolve().parent
    subprocess.run([
        "g++", "-std=c++17", "-O0", "-include", "ATen/Parallel.h",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(torch_root / "include"), str(cpp),
        "-L", str(torch_root / "lib"), "-ltorch_cpu", "-lc10",
        f"-Wl,-rpath,{torch_root / 'lib'}", "-o", str(exe),
    ], check=True)
    subprocess.run([str(exe)], check=True)


def test_codec_pair_device_math_staging_and_profile_guards(tmp_path):
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    function = "Tensor rpu_tts_codec_conv_pair(" + source.split(
        "Tensor rpu_tts_codec_conv_pair(", 1)[1].split("Tensor rpu_linear_with_core_count(", 1)[0]
    function = function.replace("tensor.device().type() == at::kPrivateUse1", "tensor.device().type() == at::kCPU")
    harness = r'''
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
using at::Tensor;
struct RpuKernelGraph {static inline bool active=false;static bool has_active(){return active;}};
int calls=0,flushes=0;
Tensor rpu_to_cpu_zerocopy_no_graph_sync(const Tensor&t,bool flush){assert(!flush);return t;}
Tensor rpu_to_cpu_zerocopy(const Tensor&t){++flushes;return t;}
void rpu_ddr_flush_force_sized(void*,size_t){++flushes;}
Tensor rpu_linear_with_accumulation(const Tensor&x,const Tensor&w,const c10::optional<Tensor>&bias,
 int64_t partition,int64_t cores,bool acc32,const c10::optional<Tensor>&scale){
 assert(x.scalar_type()==at::kHalf&&x.is_contiguous()&&!bias&&!scale&&partition==1&&cores==8&&acc32);
 ++calls;return at::mm(x.to(at::kFloat),w.to(at::kFloat).t()).to(at::kHalf);
}
template<class F>void rejects(F fn){int before=calls;bool caught=false;try{fn();}catch(const c10::Error&){caught=true;}assert(caught&&calls==before);}
'''
    cases = r'''
int main(){
 at::manual_seed(31);at::set_num_threads(2);
 for(int c:{96,192,384,768})for(int d:{1,3,9}){
  int n=((c+127)/128)*128,count=5;
  auto input=at::randn({1,c,2*(count+6*d)}).slice(2,0,2*(count+6*d),2);
  auto saved=input.clone();auto w1=(at::randn({n,c*7})/64).to(at::kHalf);
  auto w2=(at::randn({n,n})/16).to(at::kHalf);
  auto affine=at::randn({3,n}).to(at::kHalf);auto bias=at::randn({c});
  auto patches=input.to(at::kHalf).unfold(2,6*d+1,1).slice(3,0,6*d+1,d)
      .permute({0,2,1,3}).reshape({count,c*7}).contiguous();
  auto hidden=at::mm(patches.to(at::kFloat),w1.to(at::kFloat).t()).to(at::kHalf)+affine.select(0,0);
  auto sine=at::sin(hidden*affine.select(0,1));
  auto activated=hidden+(sine*sine)*affine.select(0,2);
  auto expected=at::mm(activated.to(at::kFloat),w2.to(at::kFloat).t()).to(at::kHalf)
      .narrow(1,0,c).t().unsqueeze(0).to(at::kFloat).contiguous()+bias.view({1,c,1});
  auto output=rpu_tts_codec_conv_pair(input,w1,affine,w2,bias,d,c);
  assert(at::equal(output,expected)&&at::equal(input,saved)&&output.is_contiguous());
  auto next=rpu_tts_codec_conv_pair(input+1,w1,affine,w2,bias,d,c);
  assert(at::equal(output,expected)&&!at::equal(output,next));
  rejects([&]{rpu_tts_codec_conv_pair(input.to(at::kHalf),w1,affine,w2,bias,d,c);});
  rejects([&]{rpu_tts_codec_conv_pair(input,w1,affine.to(at::kFloat),w2,bias,d,c);});
  rejects([&]{rpu_tts_codec_conv_pair(input,w1,affine.narrow(0,0,2),w2,bias,d,c);});
  rejects([&]{rpu_tts_codec_conv_pair(input,w1,affine,w2,bias.to(at::kHalf),d,c);});
  rejects([&]{rpu_tts_codec_conv_pair(input,w1,affine,w2,bias,2,c);});
  rejects([&]{rpu_tts_codec_conv_pair(input,w1,affine,w2,bias,d,16);});
  rejects([&]{rpu_tts_codec_conv_pair(input.narrow(2,0,6*d),w1,affine,w2,bias,d,c);});
  RpuKernelGraph::active=true;
  rejects([&]{rpu_tts_codec_conv_pair(input,w1,affine,w2,bias,d,c);});
  RpuKernelGraph::active=false;
 }
 assert(calls==48&&flushes==48);
}
'''
    cpp, exe = tmp_path / "pair.cpp", tmp_path / "pair"
    cpp.write_text(harness + function + cases)
    torch_root = Path(torch.__file__).resolve().parent
    subprocess.run([
        "g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(torch_root / "include"), str(cpp),
        "-L", str(torch_root / "lib"), "-ltorch_cpu", "-lc10",
        f"-Wl,-rpath,{torch_root / 'lib'}", "-o", str(exe),
    ], check=True)
    subprocess.run([str(exe)], check=True)


def test_codec_transpose_device_matmul_layout_and_rejections(tmp_path):
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    function = "Tensor rpu_tts_codec_conv_transpose1d(" + source.split(
        "Tensor rpu_tts_codec_conv_transpose1d(", 1)[1].split("Tensor rpu_linear_with_core_count(", 1)[0]
    function = function.replace("weight.device().type() == at::kPrivateUse1", "weight.device().type() == at::kCPU")
    harness = r'''
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
using at::Tensor;
struct RpuKernelGraph {static inline bool active=false;static bool has_active(){return active;}};
int calls=0,flushes=0;
Tensor rpu_to_cpu_zerocopy_no_graph_sync(const Tensor&t,bool flush){assert(!flush);return t;}
Tensor rpu_to_cpu_zerocopy(const Tensor&t){++flushes;return t;}
void rpu_ddr_flush_force_sized(void*,size_t){++flushes;}
Tensor rpu_linear_with_accumulation(const Tensor&x,const Tensor&w,const c10::optional<Tensor>&bias,
 int64_t partition,int64_t cores,bool acc32,const c10::optional<Tensor>&scale){
 assert(x.scalar_type()==at::kHalf&&x.is_contiguous()&&!bias&&!scale&&partition==1&&cores==8&&acc32);
 ++calls;return at::mm(x.to(at::kFloat),w.to(at::kFloat).t()).to(at::kHalf);
}
template<class F>void rejects(F fn){int before=calls;bool caught=false;try{fn();}catch(const c10::Error&){caught=true;}assert(caught&&calls==before);}
'''
    cases = r'''
int main(){
 at::manual_seed(41);at::set_num_threads(2);
 for(auto geometry:{std::tuple<int,int,int,int>{1024,1024,2,2},{1536,768,16,8}}){
  auto [c,n,k,s]=geometry;int count=5,padded=((n*k+127)/128)*128;
  auto input=at::randn({1,c,count});auto saved=input.clone();
  auto weight=(at::randn({padded,c})/32).to(at::kHalf);auto bias=at::randn({n});
  auto projected=at::mm(input.transpose(1,2).reshape({count,c}).to(at::kHalf).to(at::kFloat),
                         weight.to(at::kFloat).t()).to(at::kHalf).narrow(1,0,n*k);
  Tensor expected;
  if(k==s) expected=projected.view({1,count,n,s}).permute({0,2,1,3}).reshape({1,n,count*s}).to(at::kFloat);
  else {
   auto chunks=projected.view({1,count,n,2,s});
   expected=at::empty({1,n,count+1,s});
   expected.select(2,0).copy_(chunks.select(1,0).select(2,0));
   expected.select(2,count).copy_(chunks.select(1,count-1).select(2,1));
   expected.slice(2,1,count).copy_(chunks.slice(1,1,count).select(3,0).permute({0,2,1,3}));
   expected.slice(2,1,count).add_(chunks.slice(1,0,count-1).select(3,1).permute({0,2,1,3}));
   expected=expected.flatten(2);
  }
  expected.add_(bias.view({1,n,1}));
  auto output=rpu_tts_codec_conv_transpose1d(input,weight,bias,n,k,s);
  assert(at::equal(output,expected)&&at::equal(input,saved)&&output.is_contiguous());
  rejects([&]{rpu_tts_codec_conv_transpose1d(input.to(at::kHalf),weight,bias,n,k,s);});
  rejects([&]{rpu_tts_codec_conv_transpose1d(input,weight.to(at::kFloat),bias,n,k,s);});
  rejects([&]{rpu_tts_codec_conv_transpose1d(input,weight,bias,n,k,7);});
  RpuKernelGraph::active=true;
  rejects([&]{rpu_tts_codec_conv_transpose1d(input,weight,bias,n,k,s);});
  RpuKernelGraph::active=false;
 }
 assert(calls==2&&flushes==4);
}
'''
    cpp, exe = tmp_path / "transpose.cpp", tmp_path / "transpose"
    cpp.write_text(harness + function + cases)
    torch_root = Path(torch.__file__).resolve().parent
    subprocess.run([
        "g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(torch_root / "include"), str(cpp),
        "-L", str(torch_root / "lib"), "-ltorch_cpu", "-lc10",
        f"-Wl,-rpath,{torch_root / 'lib'}", "-o", str(exe),
    ], check=True)
    subprocess.run([str(exe)], check=True)
