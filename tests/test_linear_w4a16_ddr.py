"""Exercise the eager packed-W4 caller and DDR admission without a board."""
import ast
from pathlib import Path
import shutil
import subprocess

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from test_fmb_standalone_layout_rebuild import _definition


ROOT = Path(__file__).resolve().parents[1]


def test_native_w4_caller_preserves_logical_k_and_rejects_before_staging(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    body = _definition(source, "void rpu_launch_linear_ddr_kernel(")
    # Run the actual transport admission before replacing hardware side effects.
    # The recorder checks logical K, scale identity and fresh output storage.
    admission = body[:body.index("  const bool in_graph = graph_dma::active();")]
    program = r'''
#include <cassert>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
#define RECORD_FUNCTION(...)
#define NUM_CORES 8
namespace c10 { template<class T> using optional = std::optional<T>;
using Half = uint16_t;
constexpr auto nullopt=std::nullopt; }
namespace graph_dma { bool capturing=false; bool active(){return capturing;} }
int allocations=0,copies=0,retentions=0,launches=0,last_k=0,last_cores=0;
bool last_acc32=false; int spm_routes=0,fp16_acc32_routes=0;
namespace at {
enum ScalarType {kHalf,kFloat,kByte,kChar}; constexpr int kPrivateUse1=1;
struct Device {int kind=1,index=0; int type()const{return kind;}
 bool operator==(const Device& x)const{return kind==x.kind&&index==x.index;}};
struct Options {ScalarType value=kHalf;Device where; ScalarType dtype()const{return value;}};
struct Sizes:std::vector<int64_t>{using std::vector<int64_t>::vector;
 std::vector<int64_t> vec()const{return {begin(),end()};}};
struct Tensor {
 Sizes shape;Options opts;bool dense=true,present=false;
 std::shared_ptr<int> storage=std::make_shared<int>(0);
 Tensor()=default; Tensor(std::initializer_list<int64_t> d,ScalarType t=kHalf):shape(d),present(true){opts.value=t;}
 int64_t dim()const{return shape.size();} int64_t size(int64_t a)const{return shape.at(a<0?shape.size()+a:a);}
 int64_t numel()const{return std::accumulate(shape.begin(),shape.end(),int64_t{1},std::multiplies<int64_t>());}
 bool defined()const{return present;} bool is_contiguous()const{return dense;}
 Device device()const{return opts.where;} ScalarType scalar_type()const{return opts.value;}
 Sizes sizes()const{return shape;}Options options()const{return opts;}
 template<class T> T* data_ptr()const{return reinterpret_cast<T*>(storage.get());}
 Tensor contiguous()const{++copies;auto t=*this;t.dense=true;return t;}
 Tensor view(std::initializer_list<int64_t> s)const{auto t=*this;t.shape=s;return t;}
};
Tensor empty(const std::vector<int64_t>& shape,Options opts){++allocations;Tensor t;
 t.shape.assign(shape.begin(),shape.end());t.opts=opts;t.present=true;return t;}
}
using at::Tensor;
uint64_t RpuGetDevAddr(void* ptr){return reinterpret_cast<uintptr_t>(ptr);}
struct RpuKernelGraph {
 static bool has_active(){return graph_dma::capturing;}
 void begin(){}
 void keep_alive(const Tensor&){}
 void end(){}
 void abort(){}
};
Tensor rpu_retain_linear_quant_scale(const Tensor&,const Tensor& scale){++retentions;return scale;}
'''
    # Reuse the public declaration so omitted legacy arguments exercise their
    # actual compatibility defaults instead of duplicating those defaults here.
    header = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    begin = header.index("void rpu_launch_linear_ddr_kernel(")
    program += header[begin:header.index(";", begin) + 1]
    program += _definition(source, "static void check_linear_shapes(")
    program += _definition(source, "void validate_parallel_linear_geometry(")
    program += _definition(source, "static void check_linear_w4a16_operands(")
    # Execute the production DDR route decision and forwarding call, replacing
    # only SPM/device work. This catches a dropped or inverted accumulator flag.
    route_begin = body.index("    if (w8a16 || w4a16 || !use_acc32) {")
    route_else = body.index("    } else {", route_begin)
    fp16_tile_end = body.index(";", body.index("const auto tile =", route_else))
    route = body[route_begin:fp16_tile_end + 1] + "\n    }\n"
    program += r'''
void rpu_launch_linear_spm_to_spm_acc16_kernel(
 uint32_t,const Tensor&,uint32_t,int64_t,int64_t,int64_t,int,int cores,
 uint32_t,const Tensor&,int,int,bool acc32){
 ++spm_routes;last_acc32=acc32;last_cores=cores;
}
namespace rpu_pl_tiling {
int select_tile_acc32(int,int,int,bool fp16){
 assert(fp16);++fp16_acc32_routes;last_acc32=true;return 0;
}
}
'''
    program += admission + r'''
  assert(input.size(1)==(w4a16?2*weight.size(1):weight.size(1)));
  assert(output.storage!=input.storage && output.storage!=weight.storage);
  assert(output.storage!=weight_scale.storage);
  assert(input.is_contiguous()&&output.is_contiguous());
  const int64_t cur_m=input.size(0),n=weight.size(0),k=input.size(1);
  const int64_t local_n=partition?n/num_cores:n,local_k=partition?k:k/num_cores;
  const uint32_t x_addr=256,y_addr=512,b_addr=has_bias?768:0;
  last_cores=num_cores;
''' + route + r'''
  last_k=input.size(1); ++launches;*output.storage=launches;
}
'''
    program += _definition(source, "Tensor rpu_linear_with_accumulation(")
    program += _definition(source, "Tensor rpu_linear_with_core_count(")
    program += _definition(source, "Tensor rpu_linear_w4a16(")
    program += r'''
int main(){
 for(int k:{64,2048,2560}) for(int n:{128,151936}){
  Tensor x({2,17,k}),w({n,k/2},at::kByte),b({n});x.dense=false;
  const int elements=((k/32+3)/4)*((n/8+63)/64)*8*4*64;
  Tensor scale({32,elements/32});
  auto first=rpu_linear_w4a16(x,w,scale,b);const int snapshot=*first.storage;
  auto second=rpu_linear_w4a16(x,w,scale,{});
  assert((first.shape==at::Sizes{2,17,n})&&last_k==k&&!last_acc32);
  assert(first.storage!=second.storage&&*first.storage==snapshot);
  auto one=rpu_linear_w4a16(Tensor({k}),w,scale,{});
  assert((one.shape==at::Sizes{n}));
 }
 Tensor x({1,256}),w({128,128},at::kByte),s({32,128}),b({128});
 auto reject=[&](Tensor input,Tensor weight,Tensor scale,c10::optional<Tensor> bias={}){
  int a=allocations,c=copies,r=retentions,l=launches;bool failed=false;
  try{rpu_linear_w4a16(input,weight,scale,bias);}catch(const std::runtime_error&){failed=true;}
  assert(failed&&a==allocations&&c==copies&&r==retentions&&l==launches);
 };
 reject(Tensor({}),w,s);reject(Tensor({0,256}),w,s);
 reject(Tensor({1,255}),w,s);reject(x,Tensor({128,256},at::kByte),s);
 reject(x,Tensor({127,128},at::kByte),s);reject(x,Tensor({128},at::kByte),s);
 auto bad=x;bad.opts.value=at::kFloat;reject(bad,w,s);
 bad=w;bad.opts.value=at::kChar;reject(x,bad,s);
 bad=w;bad.dense=false;reject(x,bad,s);
 bad=w;bad.opts.where.index=1;reject(x,bad,s);
 bad=x;bad.opts.where.kind=0;reject(bad,w,s);
 reject(x,w,{});reject(x,w,Tensor({32,127}));reject(x,w,Tensor({64,64}));
 reject(x,w,Tensor({4096}));reject(x,w,Tensor({32,128},at::kByte));
 bad=s;bad.dense=false;reject(x,w,bad);
 bad=s;bad.opts.where.index=1;reject(x,w,bad);
 reject(x,w,s,Tensor({}));reject(x,w,s,Tensor({127}));
 bad=b;bad.dense=false;reject(x,w,s,bad);
 bad=b;bad.opts.value=at::kFloat;reject(x,w,s,bad);
 bad=b;bad.opts.where.index=1;reject(x,w,s,bad);
 graph_dma::capturing=true;reject(x,w,s);graph_dma::capturing=false;
 // Shared launcher must also reject unsafe metadata from a direct internal caller.
 Tensor output({1,128});int before=launches;bool failed=false;
 try{rpu_launch_linear_ddr_kernel(x,w,output,{},false,0,nullptr,0,nullptr,8,s);}
 catch(const std::runtime_error&){failed=true;} assert(failed&&launches==before);
 failed=false;
 try{rpu_launch_linear_ddr_kernel(x,w,output,{},false,1,nullptr,0,nullptr,4,s);}
 catch(const std::runtime_error&){failed=true;} assert(failed&&launches==before);
 rpu_linear_w4a16(x,w,s,{});assert(launches==before+1&&!last_acc32);
 // Public explicit accumulation must reach the selected DDR/SPM family for
 // FP16, natural-scale W8 and packed G32 W4; reduced W8 uses the actual core count.
 for(auto dtype:{at::kHalf,at::kChar,at::kByte}) for(bool acc32:{false,true})
 for(int cores:{4,8}){
  if(dtype==at::kByte&&cores!=8)continue;
  Tensor input({2,17,256}),weight({128,dtype==at::kByte?128:256},dtype);
  input.dense=false;
  c10::optional<Tensor> scale;
  if(dtype==at::kChar)scale=Tensor({128});
  if(dtype==at::kByte)scale=s;
  const int old_spm=spm_routes,old_acc32=fp16_acc32_routes;
  auto first=rpu_linear_with_accumulation(input,weight,{},1,cores,acc32,scale);
  const int snapshot=*first.storage;
  assert(last_acc32==acc32&&last_cores==cores&&last_k==256);
  assert((first.shape==at::Sizes{2,17,128}));
  assert(fp16_acc32_routes-old_acc32==(dtype==at::kHalf&&acc32));
  assert(spm_routes-old_spm!=(dtype==at::kHalf&&acc32));
  auto second=rpu_linear_with_accumulation(input,weight,Tensor({128}),1,cores,acc32,scale);
  assert(first.storage!=second.storage&&*first.storage==snapshot);
 }
 Tensor fpweight({128,256}),w8weight({128,256},at::kChar);
 rpu_linear_with_core_count(x,fpweight,{},1,4);
 assert(last_acc32&&last_cores==4); // legacy explicit FP16 op remains ACC32
 rpu_launch_linear_ddr_kernel(x,fpweight,output,{},false,1);
 assert(last_acc32); // legacy DDR FP16 omitted flag
 rpu_launch_linear_ddr_kernel(x,w8weight,output,{},false,1,nullptr,0,nullptr,8,Tensor({128}));
 assert(!last_acc32); // legacy DDR W8 omitted flag
 auto reject_explicit=[&](Tensor input,Tensor weight,c10::optional<Tensor> scale,
                           int partition=1,int cores=8){
  int l=launches;bool rejected=false;
  try{rpu_linear_with_accumulation(input,weight,{},partition,cores,true,scale);}
  catch(const std::runtime_error&){rejected=true;}
  assert(rejected&&launches==l);
 };
 reject_explicit(x,w8weight,{});
 reject_explicit(x,w8weight,Tensor({127}));
 reject_explicit(x,w8weight,Tensor({128},at::kFloat));
 auto invalid_scale=Tensor({128});invalid_scale.opts.where.index=1;
 reject_explicit(x,w8weight,invalid_scale);
 invalid_scale=Tensor({128});invalid_scale.dense=false;
 reject_explicit(x,w8weight,invalid_scale);
 reject_explicit(x,w8weight,Tensor({128}),0);
 reject_explicit(x,w8weight,Tensor({128}),1,3);
 reject_explicit(x,fpweight,Tensor({128}));
 reject_explicit(x,w,s,1,4); // G32 stripes remain TP8 on this eager public ABI
 reject_explicit(x,w,{});
 graph_dma::capturing=true;
 reject_explicit(x,fpweight,{});
 reject_explicit(x,w8weight,Tensor({128}));
 reject_explicit(x,w,s);
 graph_dma::capturing=false;
}
'''
    cpp, binary = tmp_path / "w4.cpp", tmp_path / "w4"
    cpp.write_text(program)
    build = subprocess.run([compiler, "-std=c++17", str(cpp), "-o", str(binary)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


def _fake_function():
    tree = ast.parse((ROOT / "python/rpu_backend/graph/fake_dispatch.py").read_text())
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
                "_has_fake", "_require_fake", "_meta_linear_w4a16"}:
            node.decorator_list = []
            functions.append(node)
    namespace = {"torch": torch, "FakeTensor": FakeTensor}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "fake_dispatch.py", "exec"), namespace)
    return namespace["_meta_linear_w4a16"]


@pytest.mark.parametrize("shape", [(256,), (17,256), (2,3,256)])
def test_w4_fake_preserves_selected_rows_and_fp16_output(shape):
    function = _fake_function()
    with FakeTensorMode():
        x = torch.empty(shape, dtype=torch.float16)
        w = torch.empty((128,128), dtype=torch.uint8)
        s = torch.empty((32,128), dtype=torch.float16)
        out = function(x,w,s)
        assert isinstance(out,FakeTensor) and out.shape == (*shape[:-1],128)
        assert out.dtype == torch.float16
        with pytest.raises(RuntimeError, match="scale"):
            function(x,w,s[:,:-1])
        with pytest.raises(RuntimeError, match="packed"):
            function(x,w.half(),s)
    with pytest.raises(RuntimeError, match="non-Fake"):
        function(torch.empty(shape).half(),torch.empty((128,128),dtype=torch.uint8),
                 torch.empty((32,128)).half())
