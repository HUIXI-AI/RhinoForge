"""Production codec staging with real ATen tensors and an emulated device boundary."""
from pathlib import Path
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[1]
LINEAR_BRIDGE = r'''
Tensor tts_codec_linear_rows(const Tensor&x,const Tensor&w,const c10::optional<Tensor>&b){
 return rpu_linear_with_accumulation(x,w,b,1,8,true,c10::nullopt);
}
'''


def test_transpose_device_post_boundary_and_owned_output(tmp_path):
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    helpers = "Tensor tts_codec_cpu_nct_to_device_rows(" + source.split(
        "Tensor tts_codec_cpu_nct_to_device_rows(", 1)[1].split("Tensor tts_codec_im2col_rows(", 1)[0]
    function = "Tensor rpu_tts_codec_conv_transpose1d_device_post(" + source.split(
        "Tensor rpu_tts_codec_conv_transpose1d_device_post(", 1)[1].split(
        "static Tensor tts_codec_decoder_stage_impl(", 1)[0]
    harness = r'''
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <cassert>
using at::Tensor;
int launches=0;
struct RpuKernelGraph{static inline bool active=false;static bool has_active(){return active;}};
void check_tts_codec_device_half(const Tensor&t,c10::IntArrayRef shape,const char*){
 TORCH_CHECK(t.scalar_type()==at::kHalf && t.is_contiguous() && t.sizes().equals(shape));
}
Tensor rpu_to_cpu_zerocopy_no_graph_sync(const Tensor&t,bool){return t;}
Tensor rpu_to_cpu_zerocopy(const Tensor&t){return t;}
void rpu_ddr_flush_force_sized(void*,size_t){}
Tensor rpu_linear_with_accumulation(const Tensor&x,const Tensor&w,const c10::optional<Tensor>&,
 int64_t,int64_t,bool,const c10::optional<Tensor>&){++launches;return at::mm(x.to(at::kFloat),w.to(at::kFloat).t()).to(at::kHalf);}
Tensor rpu_tts_codec_transpose_overlap(const Tensor&x,const c10::optional<Tensor>&bias,int64_t c,int64_t k,int64_t s){
 int64_t p=(c+127)/128*128,t=x.size(0);
 auto y=at::zeros({(t-1)*s+k,p},x.options());
 for(int i=0;i<t;++i)y.narrow(0,i*s,k).add_(x[i].view({k,p}));
 if(bias)y.add_(*bias);return y;
}
'''
    cases = r'''
int main(){
 at::manual_seed(2);at::set_num_threads(2);
 for(int c:{1024,768,384,192,96}){
  int in=c==1024?1024:c*2,k=c==1024?2:c==768?16:c==384?10:c==192?8:6,s=k==2?2:k/2,p=(c+127)/128*128;
  auto x=at::randn({1,in,6}).slice(2,0,6,2),saved=x.clone();
  auto w=(at::randn({k*p,in})/64).to(at::kHalf),bias=at::randn({p}).to(at::kHalf);
  auto rows=x.squeeze(0).t().to(at::kHalf).contiguous();
  auto projected=at::mm(rows.to(at::kFloat),w.to(at::kFloat).t()).to(at::kHalf);
  auto expected=rpu_tts_codec_transpose_overlap(projected,bias,c,k,s).to(at::kFloat).narrow(1,0,c).t().unsqueeze(0).contiguous();
  auto y=rpu_tts_codec_conv_transpose1d_device_post(x,w,bias,c,k,s);
  assert(at::equal(y,expected)&&at::equal(x,saved)&&y.is_contiguous());
  rpu_tts_codec_conv_transpose1d_device_post(x+1,w,bias,c,k,s);assert(at::equal(y,expected));
  for(int bad=0;bad<8;++bad){
   int before=launches;bool rejected=false;RpuKernelGraph::active=bad==0;
   auto invalid=bad==5?x.narrow(2,0,0):bad==6?x.narrow(2,0,1).expand({1,in,255*640+1}):x;
   try{rpu_tts_codec_conv_transpose1d_device_post(bad==1?x.to(at::kHalf):invalid,bad==2?w.to(at::kFloat):w,
       bad==3?bias.narrow(0,0,p-1):bias,bad==7?INT64_MAX:c,k,bad==4?s+1:s);}
   catch(const c10::Error&){rejected=true;}
   assert(rejected&&before==launches);
  }
  RpuKernelGraph::active=false;
 }
}
'''
    cpp, exe = tmp_path / "post.cpp", tmp_path / "post"
    cpp.write_text(harness + LINEAR_BRIDGE + helpers + function + cases)
    root = Path(torch.__file__).resolve().parent
    subprocess.run(["g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(root / "include"), str(cpp), "-L", str(root / "lib"),
        "-ltorch_cpu", "-lc10", f"-Wl,-rpath,{root / 'lib'}", "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_decoder_stage_has_one_host_boundary_and_three_residuals(tmp_path):
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    function = "static Tensor tts_codec_decoder_stage_impl(" + source.split(
        "static Tensor tts_codec_decoder_stage_impl(", 1)[1].split(
        "Tensor rpu_tts_codec_transpose_overlap(", 1)[0]
    function = function.replace("at::kPrivateUse1", "at::kCPU")
    harness = r'''
#include <ATen/ATen.h>
#include <cassert>
using at::Tensor;
int uploads=0,downloads=0,snakes=0,linears=0,overlaps=0,residuals=0;
int64_t CeilDiv(int64_t x,int64_t y){return (x+y-1)/y;}
struct RpuKernelGraph{static inline bool active=false;static bool has_active(){return active;}};
void check_tts_codec_device_half(const Tensor&t,c10::IntArrayRef shape,const char*){
 TORCH_CHECK(t.scalar_type()==at::kHalf&&t.is_contiguous()&&t.sizes().equals(shape));
}
Tensor tts_codec_cpu_nct_to_device_rows(const Tensor&x,const Tensor&,int64_t=0){
 ++uploads;return x.squeeze(0).t().to(at::kHalf).contiguous();
}
Tensor tts_codec_device_rows_to_cpu_nct(const Tensor&x){
 ++downloads;return x.t().unsqueeze(0).to(at::kFloat).contiguous();
}
Tensor tts_codec_snake_rows(const Tensor&x,const Tensor&,const Tensor&){++snakes;return x+1;}
Tensor tts_codec_linear_rows(const Tensor&x,const Tensor&w,const c10::optional<Tensor>&){
 ++linears;return at::zeros({x.size(0),w.size(0)},x.options());
}
Tensor rpu_tts_codec_transpose_overlap(const Tensor&x,const c10::optional<Tensor>&,
 int64_t c,int64_t,int64_t s){++overlaps;return at::zeros({(x.size(0)+1)*s,CeilDiv(c,128)*128},x.options());}
Tensor tts_codec_residual_unit_impl(const Tensor&x,const Tensor&,const Tensor&,const Tensor&,
 int64_t d,int64_t c,bool complete,bool resident,bool padded){
 int step=residuals%3;assert(complete&&resident&&padded==(c==96||c==192)&&d==(step==0?1:step==1?3:9));
 ++residuals;return x+d;
}
'''
    cases = r'''
int main(){
 auto x=at::randn({1,192,3}),saved=x.clone();
 auto snake=at::ones({2,256},at::kHalf),tw=at::zeros({6*128,192},at::kHalf);
 auto bias=at::zeros({128},at::kHalf),w=at::zeros({128,128},at::kHalf);
 std::vector<Tensor> first(3,w),affine(3,at::zeros({6,128},at::kHalf)),second(3,w);
 auto y=rpu_tts_codec_decoder_stage(x,snake,tw,bias,first,affine,second,96,6,3);
 assert(y.sizes().equals({1,96,12})&&uploads==1&&downloads==1&&snakes==1&&
        linears==1&&overlaps==1&&residuals==3&&at::equal(x,saved));
 auto held=y.clone();rpu_tts_codec_decoder_stage(x+1,snake,tw,bias,first,affine,second,96,6,3);
 assert(at::equal(y,held));
 int before=uploads;bool rejected=false;try{
  std::vector<Tensor> short_list(2,w);
  rpu_tts_codec_decoder_stage(x,snake,tw,bias,short_list,affine,second,96,6,3);
 }catch(const c10::Error&){rejected=true;}assert(rejected&&uploads==before);
 rejected=false;try{rpu_tts_codec_decoder_stage(x,snake,tw,bias,first,affine,second,192,6,3);}
 catch(const c10::Error&){rejected=true;}assert(rejected&&uploads==before);
 RpuKernelGraph::active=true;rejected=false;
 try{rpu_tts_codec_decoder_stage(x,snake,tw,bias,first,affine,second,96,6,3);}
 catch(const c10::Error&){rejected=true;}assert(rejected&&uploads==before);
 RpuKernelGraph::active=false;
 auto input=at::randn({1,1536,2});
 auto parameters=[](int in,int out,int kernel){
  int pi=CeilDiv(in,128)*128,po=CeilDiv(out,128)*128;
  return std::vector<Tensor>{at::ones({2,pi},at::kHalf),at::zeros({kernel*po,in},at::kHalf),at::zeros({po},at::kHalf)};
 };
 auto a=parameters(1536,768,16),b=parameters(768,384,10),c=parameters(384,192,8);
 auto run=[&](Tensor value,const std::vector<Tensor>&p,int channels,int kernel,int stride,bool ri,bool ro){
  return rpu_tts_codec_decoder_stage_rows(value,p[0],p[1],p[2],first,affine,second,channels,kernel,stride,ri,ro);
 };
 int u=uploads,d=downloads;
 auto rows=run(input,a,768,16,8,false,true);
 assert(rows.dim()==2&&rows.size(1)==768&&rows.is_contiguous());
 rows=run(rows,b,384,10,5,true,true);
 auto final=run(rows,c,192,8,4,true,false);
 assert(uploads-u==1&&downloads-d==1&&final.size(1)==192);
 auto held_final=final.clone();
 auto reference=rpu_tts_codec_decoder_stage(input,a[0],a[1],a[2],first,affine,second,768,16,8);
 reference=rpu_tts_codec_decoder_stage(reference,b[0],b[1],b[2],first,affine,second,384,10,5);
 reference=rpu_tts_codec_decoder_stage(reference,c[0],c[1],c[2],first,affine,second,192,8,4);
 assert(at::equal(final,reference)&&at::equal(final,held_final));
 int launches=snakes;
 for(int fault=0;fault<3;++fault){
  rejected=false;
  try {
   if(fault==0)run(input,a,768,16,8,true,true);
   if(fault==1)run(rows,c,192,8,4,true,true);
   if(fault==2)run(rows.t(),c,192,8,4,true,false);
  }catch(const c10::Error&){rejected=true;}
  assert(rejected&&snakes==launches);
 }
}
'''
    cpp, exe = tmp_path / "stage.cpp", tmp_path / "stage"
    cpp.write_text(harness + function + cases)
    root = Path(torch.__file__).resolve().parent
    subprocess.run(["g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(root / "include"), str(cpp), "-L", str(root / "lib"),
        "-ltorch_cpu", "-lc10", f"-Wl,-rpath,{root / 'lib'}", "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_complete_residual_keeps_legacy_affine_contract(tmp_path):
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    upload = "Tensor tts_codec_cpu_nct_to_device_rows(" + source.split(
        "Tensor tts_codec_cpu_nct_to_device_rows(", 1)[1].split(
        "Tensor tts_codec_device_rows_to_cpu_nct(", 1)[0]
    function = "static Tensor tts_codec_residual_unit_impl(" + source.split(
        "static Tensor tts_codec_residual_unit_impl(", 1)[1].split(
        "Tensor rpu_tts_codec_snake(", 1)[0]
    function = function.replace("at::kPrivateUse1", "at::kCPU")
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
 return y.unfold(1,6*d+1,1).slice(2,0,6*d+1,d).permute({1,2,0}).reshape({x.size(0),x.size(1)*7}).contiguous();
}
Tensor rpu_linear_with_accumulation(const Tensor&x,const Tensor&w,const c10::optional<Tensor>&,
 int64_t,int64_t,bool,const c10::optional<Tensor>&){return at::mm(x.to(at::kFloat),w.to(at::kFloat).t()).to(at::kHalf);}
'''
    cases = r'''
int main(){
 at::manual_seed(73);
 for(int c:{96,192})for(int t:{1,3,17}){
  int p=CeilDiv(c,128)*128;
  auto x=at::randn({1,c,t})/10;
  auto rows=tts_codec_cpu_nct_to_device_rows(x,at::empty({1},at::kHalf),p);
  auto first=at::constant_pad_nd((at::randn({c,7,c})/64).to(at::kHalf),{0,p-c,0,0,0,p-c}).reshape({p,7*p});
  auto second=at::constant_pad_nd((at::randn({c,c})/16).to(at::kHalf),{0,p-c,0,p-c});
  auto affine=at::constant_pad_nd((at::randn({6,c})/10).to(at::kHalf),{0,p-c});
  affine.select(0,4).narrow(0,c,p-c).fill_(1);
  auto oldfirst=first.view({p,7,p}).narrow(2,0,c).contiguous().view({p,7*c});
  auto reference=x;
  for(int d:{1,3,9}){
   reference=rpu_tts_codec_residual_unit_complete(reference,oldfirst,affine,second,d,c);
   auto saved=rows.clone();
   auto next=rpu_tts_codec_residual_unit_padded_rows(rows,first,affine,second,d,c);
   assert(at::equal(saved,rows));rows=next;
   assert(at::count_nonzero(rows.narrow(1,c,p-c)).item<int64_t>()==0);
   assert(at::allclose(tts_codec_device_rows_to_cpu_nct(rows.narrow(1,0,c)),reference,1e-3,1e-3));
  }
  bool rejected=false;
  try{rpu_tts_codec_residual_unit_padded_rows(rows,oldfirst,affine,second,1,c);}catch(const c10::Error&){rejected=true;}
  assert(rejected);
 }
 for(int c:{384,768}){
  auto input=at::randn({1,c,3})/10;
  auto rows=input.squeeze(0).t().to(at::kHalf).contiguous(),saved=rows.clone();
  auto w1=(at::randn({c,c*7})/64).to(at::kHalf);
  auto w2=(at::randn({c,c})/16).to(at::kHalf);
  auto affine=(at::randn({6,c})/10).to(at::kHalf);
  Tensor expected=input;
  for(int dilation:{1,3,9}){
   expected=rpu_tts_codec_residual_unit_complete(expected,w1,affine,w2,dilation,c);
   rows=rpu_tts_codec_residual_unit_rows(rows,w1,affine,w2,dilation,c);
  }
  assert(at::equal(rows.t().unsqueeze(0).to(at::kFloat),expected));
  assert(at::equal(saved,input.squeeze(0).t().to(at::kHalf)));
  auto retained=rows.clone();
  rpu_tts_codec_residual_unit_rows(rows,w1,affine,w2,1,c);
  assert(at::equal(rows,retained));
  bool rejected=false;
  try{rpu_tts_codec_residual_unit_rows(input,w1,affine,w2,1,c);}catch(const c10::Error&){rejected=true;}
  assert(rejected);
 }
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
    cpp.write_text(harness + LINEAR_BRIDGE + upload + function + cases)
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
    cpp.write_text(harness + LINEAR_BRIDGE + function + cases)
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
    cpp.write_text(harness + LINEAR_BRIDGE + function + cases)
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
    cpp.write_text(harness + LINEAR_BRIDGE + function + cases)
    torch_root = Path(torch.__file__).resolve().parent
    subprocess.run([
        "g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(torch_root / "include"), str(cpp),
        "-L", str(torch_root / "lib"), "-ltorch_cpu", "-lc10",
        f"-Wl,-rpath,{torch_root / 'lib'}", "-o", str(exe),
    ], check=True)
    subprocess.run([str(exe)], check=True)
