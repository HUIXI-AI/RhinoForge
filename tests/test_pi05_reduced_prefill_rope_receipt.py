"""Whole native Pi RoPE setter with real ATen and explicit host boundaries.

No board/SDK/Graph or full weight-install certification is provided here.
The cold FMB metadata and weight binding are boundary doubles; actual reduced
projection validation, cold admission expression, full setter, CPU/device dispatch
and copy publication blocks are extracted from the current repository source.
All source payloads, storage/view/flags and inference versions are real ATen.
This fixture is portable: only repository files, Torch, pytest and c++ are used.
"""
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def _function(source, marker):
    start = source.index(marker)
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                    lambda match: " " * len(match[0]), source)
    brace = masked.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return source[start:end]


def _run_native(tmp_path, source, torch):
    compiler = shutil.which("c++")
    assert compiler, "whole native RoPE contract requires c++"
    cpp, binary = tmp_path / "pi_rope.cpp", tmp_path / "pi_rope"
    cpp.write_text(source)
    torch_root = Path(torch.__file__).resolve().parent
    command = [compiler, "-std=c++17", "-O0", "-D_GLIBCXX_USE_CXX11_ABI=" +
               str(int(torch.compiled_with_cxx11_abi())), "-I", str(ROOT / "src"),
               "-I", str(torch_root / "include"),
               "-I", str(torch_root / "include/torch/csrc/api/include"), str(cpp),
               "-L", str(torch_root / "lib"), "-Wl,-rpath," + str(torch_root / "lib"),
               "-ltorch_cpu", "-lc10", "-o", str(binary)]
    built = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stdout + built.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_private_cpu_tp46_rope_receipt_runs_whole_setter_and_copy_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    source = (ROOT / "src/fused/rpu_gemma_model.cpp").read_text()
    dispatch = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    copy_source = (ROOT / "src/core/rpu_tensor_ops.inc").read_text()
    topology_source = (ROOT / "src/core/pi05_execution_topology.h").read_text()
    setter = _function(source, "    void set_prefill_rope(").replace(
        "set_prefill_rope(", "set_prefill_rope_host(").replace("at::Tensor", "HostTensor")
    begin = source.index("    struct Pi05PrefillRopeSpan {")
    fields = source[begin:source.index("    bool prefill_rope_ready_", begin)].replace("at::Tensor", "HostTensor")
    weights = _function(source, "    void set_weights(")
    entry = weights[weights.index("{") + 1:weights.index("        int64_t N =")]
    begin = weights.index("            prefill_cos_ = at::empty(")
    allocation = weights[begin:weights.index("            for (const auto& weights", begin)]
    begin = weights.index("        int64_t physical_intermediate =")
    cold_validation = weights[begin:weights.index("        logical_intermediate_size_ =", begin)]
    begin = weights.index("        pi05_reduced_prefill_rope_receipt_ =\n")
    admission = weights[begin:weights.index("        invalidate_model_state();", begin)]
    install_end = weights[weights.rindex("        invalidate_model_state();"):weights.rindex("    }")]
    topology = "\n".join([re.search(r"enum class Pi05ReducedPrecision \{[^}]*\};",
                                     topology_source).group()] + [
        _function(topology_source, marker) for marker in (
            "inline int64_t pi05_mlp_physical_size(", "inline int64_t validate_pi05_reduced_geometry(",
            "inline void validate_pi05_reduced_projection_shapes(")])
    wrapper = _function(source, "void rpu_gemma_set_prefill_rope(")
    schema = re.search(r'  m.def\("gemma_set_prefill_rope\(.*?;', dispatch).group()
    registrations = []
    for key in ("CPU", "PrivateUse1"):
        body = _function(dispatch, f"TORCH_LIBRARY_IMPL(rpu, {key}, m)")
        entries = re.findall(r'  m.impl\("gemma_set_prefill_rope".*?;', body)
        registrations.append(f"TORCH_LIBRARY_IMPL(pi05_rope_host, {key}, m) {{\n"
                             + "\n".join(entries) + "\n}")
    copy = _function(copy_source, "at::Tensor &rpu_copy_impl(")
    preflight = copy[copy.index("{") + 1:copy.index("  const bool same_dtype")]
    same_device = _function(copy_source, "  if (same_device && same_dtype) {")
    begin = copy_source.index("  // Case 3: Cross-device + Same dtype")
    cross_device = copy_source[begin:copy_source.index("\n  return self;", begin)]
    elementwise = "template <typename SrcT, typename DstT>\n" + _function(
        copy_source, "static void cross_device_elementwise_copy(")
    elementwise = elementwise.replace("const Tensor &", "const HostTensor &")
    code = r'''

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/InferenceMode.h>
#include <torch/library.h>
#include "core/rpu_profile.h"
#include <array>
#include <cassert>
#include <cstring>
#include <limits>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
bool g_rpu_ddr_flush_enabled=true,g_rpu_ddr_flush_force_enabled=true;
// No Graph submission is pending in this harness (graph/graph_async_end.h).
bool rpu_graph_async_end_pending() noexcept{return false;}
void rpu_graph_async_end_wait(const char*) noexcept{}
template<class S> bool rpu_graph_async_end_independent_write(const void*,size_t,const S&) noexcept{return false;}
struct Store{at::Tensor value;std::vector<uint8_t> ddr;bool rpu,device_dirty=false;};
std::vector<std::weak_ptr<Store>> stores;
std::vector<std::pair<void*,size_t>> flushes;
size_t copies=0,scenarios=0;
int allocation_until_failure=-1;
int flushes_until_failure=-1;
struct HostOptions{at::TensorOptions value;c10::Device where;};
struct HostTensor;
HostTensor& transport(HostTensor&,const HostTensor&);
struct HostStorage{
 c10::Storage value;bool override_span;uintptr_t base;size_t bytes;
 auto unsafeGetStorageImpl()const{return value.unsafeGetStorageImpl();}
 struct DataPtr{void* value;void* get()const{return value;}};
 DataPtr data_ptr()const{return {override_span?reinterpret_cast<void*>(base):value.data_ptr().get()};}
 size_t nbytes()const{return override_span?bytes:value.nbytes();}
};
struct HostTensor{
 at::Tensor value;std::shared_ptr<Store> store;c10::Device where{at::kCPU};
 bool override_span=false;uintptr_t storage_base=0;size_t storage_bytes=0;
 bool defined()const{return value.defined();}auto device()const{return where;}
 auto scalar_type()const{return value.scalar_type();}auto dtype()const{return value.dtype();}
 auto dim()const{return value.dim();}auto size(int64_t i)const{return value.size(i);}
 auto stride(int64_t i)const{return value.stride(i);}auto sizes()const{return value.sizes();}
 auto strides()const{return value.strides();}
 auto numel()const{return value.numel();}auto nbytes()const{return value.nbytes();}
 auto storage_offset()const{return value.storage_offset();}
 auto unsafeGetTensorImpl()const{return value.unsafeGetTensorImpl();}
 bool is_contiguous()const{return value.is_contiguous();}
 bool is_inference()const{return value.is_inference();}
 bool is_same(const HostTensor& b)const{return value.is_same(b.value);}
 void* data_ptr()const{return value.data_ptr();}
 template<class T>T* data_ptr()const{return value.data_ptr<T>();}
 HostOptions options()const{return {value.options(),where};}
 HostStorage storage()const{return {value.storage(),override_span,storage_base,storage_bytes};}
 HostTensor narrow(int64_t dim,int64_t start,int64_t rows)const{
  auto result=*this;result.value=value.narrow(dim,start,rows);return result;}
 HostTensor& copy_(const HostTensor& src,bool non_blocking=false){
  ++copies;transport(*this,src);value.copy_(value);return *this;}
};
HostTensor make_tensor(at::IntArrayRef sizes,c10::DeviceType device){
 if(allocation_until_failure==0){allocation_until_failure=-1;TORCH_CHECK(false,"injected allocation failure");}
 if(allocation_until_failure>0)--allocation_until_failure;
 auto s=std::make_shared<Store>();s->value=at::full(sizes,777,at::kHalf);
 auto* p=static_cast<uint8_t*>(s->value.data_ptr());s->ddr.assign(p,p+s->value.nbytes());
 s->rpu=device==at::kPrivateUse1;stores.push_back(s);
 return {s->value,s,c10::Device(device)};
}
namespace at {
 HostTensor empty(IntArrayRef sizes,const HostOptions& options){return make_tensor(sizes,options.where.type());}
 HostTensor empty_like(const HostTensor& t){return make_tensor(t.sizes(),t.device().type());}
}
namespace rhino_lkn{void RpuDdrFlush(void* p,size_t n){
 flushes.emplace_back(p,n);
 if(flushes_until_failure==0){flushes_until_failure=-1;TORCH_CHECK(false,"injected second publication failure");}
 if(flushes_until_failure>0)--flushes_until_failure;
 const auto ptr=reinterpret_cast<uintptr_t>(p);
 for(auto& weak:stores)if(auto s=weak.lock()){
  const auto base=reinterpret_cast<uintptr_t>(s->value.data_ptr());
  if(ptr>=base&&ptr-base<=s->ddr.size()&&n<=s->ddr.size()-(ptr-base)){
   assert(s->rpu);auto* device=s->ddr.data()+ptr-base;
   if(s->device_dirty){std::memcpy(p,device,n);s->device_dirty=false;}else std::memcpy(device,p,n);
   return;
  }
 }assert(false);
}}
void rpu_elementwise_copy_impl(const HostTensor&,const HostTensor&){assert(false);}
__ELEMENTWISE__
HostTensor& transport(HostTensor& self,const HostTensor& src){
 const bool non_blocking=false;
 __PREFLIGHT__
 const bool same_device=self.device()==src.device(),same_dtype=self.dtype()==src.dtype();
 __SAME_DEVICE__
 if(same_device&&same_dtype)return self;
 const bool is_src_rpu=src.device().type()==at::kPrivateUse1,is_dst_rpu=self.device().type()==at::kPrivateUse1;
 const bool is_src_cpu=src.device().is_cpu(),is_dst_cpu=self.device().is_cpu();
 __CROSS_DEVICE__
 return self;
}
HostTensor tagged(const at::Tensor& value){
 if(!value.defined())return {};
 for(auto& weak:stores)if(auto s=weak.lock()){
  if(value.storage().unsafeGetStorageImpl()==s->value.storage().unsafeGetStorageImpl())
   return {value,s,c10::Device(s->rpu?at::kPrivateUse1:at::kCPU)};
 }return {value,{},value.device()};
}
__TOPOLOGY__
struct Model{
 // Reduced TP4/6 RoPE transport never installs the TP8 A8 panel owner.
 bool pi05_prefill_a8_panel512_=false;
 void validate_pi05_a8_panel512_cold()const{assert(!pi05_prefill_a8_panel512_);}
 bool pi05_prefill_kv1_pair_direct_=false,prefill_rope_ready_=false,pi05_prefill_kv1_replica_proof_=false;
 std::vector<int> pi05_a8_scales_,chunk_mask_publications_;
 HostTensor cos_,sin_;int64_t prefix_rows=800;uint64_t generation=9;
 int cores=4,attention_tp=4,mlp=4;int64_t layers=18,logical_intermediate_size_=16384;
 bool reduced_w8a16_=false;
 __FIELDS__
 int num_cores()const{return cores;}int attn_tp()const{return attention_tp;}int mlp_tp()const{return mlp;}
 int64_t num_layers()const{return layers;}int64_t pi05_prefix_rows()const{return prefix_rows;}
 uint64_t installed_model_state_generation()const{return generation;}
 void invalidate_model_state(){++generation;}
 // FMB/device/weight-binding are boundaries. Actual reduced projection metadata
 // validation and cold receipt condition below are extracted without changes.
 // Meta weights avoid materializing an unrelated 4-GiB model in a host unit test.
 void reinstall(int selected_cores,bool w8,int64_t capacity=1600,int bad=0){
  __WEIGHTS_ENTRY__
  cores=selected_cores;attention_tp=cores==8?8:4;mlp=cores;
  const int64_t N=18,num_q_heads=8,num_kv_heads=cores==8?8:4,head_dim=256,hidden_size=2048,intermediate_size=16384;
  auto dtype=w8?at::kChar:at::kHalf;
  auto physical=pi05_mlp_physical_size(intermediate_size,cores,
   w8?Pi05ReducedPrecision::W8A16:Pi05ReducedPrecision::FP16);
  std::vector<at::Tensor> q_w_list,k_w_list,v_w_list,o_w_list,gate_list,up_list,down_list;
  auto meta=[&](int64_t rows,int64_t cols){return at::empty({rows,cols},at::TensorOptions().dtype(dtype).device(at::kMeta));};
  for(int i=0;i<N;++i){
   q_w_list.push_back(meta(2048,hidden_size));k_w_list.push_back(meta(1024,hidden_size));
   v_w_list.push_back(meta(1024,hidden_size));o_w_list.push_back(meta(hidden_size,2048));
   gate_list.push_back(meta(physical,hidden_size));up_list.push_back(meta(physical,hidden_size));
   down_list.push_back(meta(hidden_size,physical));
  }
  if(bad==1)k_w_list[3]=k_w_list[3].to(at::kFloat);
  if(bad==2)down_list[4]=meta(hidden_size,physical-16);
  __COLD_VALIDATION__
  logical_intermediate_size_=intermediate_size;reduced_w8a16_=w8;
  cos_=make_tensor({capacity,128},at::kPrivateUse1);sin_=make_tensor({capacity,128},at::kPrivateUse1);
  if(pi05_prefill_kv1_pair_direct_){__ALLOCATION__}
  __COLD_ADMISSION__
  __INSTALL_END__
 }
 __SETTER__
 void set_prefill_rope(const at::Tensor& cos,const at::Tensor& sin){set_prefill_rope_host(tagged(cos),tagged(sin));}
};
struct GemmaRegistry{static Model* get(int64_t handle,const char*){
 TORCH_CHECK(handle==51,"invalid handle");static Model model;return &model;}};
__WRAPPER__
TORCH_LIBRARY(pi05_rope_host,m){__SCHEMA__}
__REGISTRATIONS__
int64_t version(const HostTensor& t){return t.value.is_inference()?-1:t.value._version();}
void raw_fill(const HostTensor& t,int v){for(int64_t i=0;i<t.numel();++i)t.data_ptr<c10::Half>()[i]=v;}
void raw_bits(const HostTensor& t,uint16_t v){for(int64_t i=0;i<t.numel();++i)static_cast<uint16_t*>(t.data_ptr())[i]=v;}
void check_bits(const HostTensor& dst,const HostTensor& src,bool published){
 assert(std::memcmp(dst.data_ptr(),src.data_ptr(),src.nbytes())==0);
 if(published){const auto offset=reinterpret_cast<uintptr_t>(dst.data_ptr())-
   reinterpret_cast<uintptr_t>(dst.store->value.data_ptr());
  assert(std::memcmp(dst.store->ddr.data()+offset,src.data_ptr(),src.nbytes())==0);}
}
template<class F>void rejects(Model& model,F fn){
 const auto before_c=model.prefill_cos_.value.clone(),before_s=model.prefill_sin_.value.clone();
 const auto f=flushes.size(),c=copies;bool caught=false;
 try{fn();}catch(const c10::Error&){caught=true;}
 assert(caught&&!model.prefill_rope_ready_&&!model.pi05_prefill_rope_cpu_published_);
 assert(flushes.size()==f&&copies==c);
 assert(std::memcmp(before_c.data_ptr(),model.prefill_cos_.data_ptr(),before_c.nbytes())==0);
 assert(std::memcmp(before_s.data_ptr(),model.prefill_sin_.data_ptr(),before_s.nbytes())==0);
}
int main(){
 at::set_num_threads(1);
 auto op=c10::Dispatcher::singleton().findSchemaOrThrow("pi05_rope_host::gemma_set_prefill_rope","")
  .typed<void(int64_t,const at::Tensor&,const at::Tensor&)>();
 assert(op.hasKernelForDispatchKey(c10::DispatchKey::CPU));
 assert(op.hasKernelForDispatchKey(c10::DispatchKey::PrivateUse1));
 auto* shared=GemmaRegistry::get(51,"");
 auto invoke=[&](Model& m,const HostTensor& a,const HostTensor& b,size_t count,size_t publications){
  auto c=copies,f=flushes.size();m.set_prefill_rope_host(a,b);
  assert(copies-c==count&&flushes.size()-f==publications);
  auto expected_a=a.value.contiguous(),expected_b=b.value.contiguous();
  assert(std::memcmp(m.prefill_cos_.data_ptr(),expected_a.data_ptr(),a.nbytes())==0);
  assert(std::memcmp(m.prefill_sin_.data_ptr(),expected_b.data_ptr(),b.nbytes())==0);
  if(g_rpu_ddr_flush_force_enabled){
   assert(std::memcmp(m.prefill_cos_.store->ddr.data(),expected_a.data_ptr(),a.nbytes())==0);
   assert(std::memcmp(m.prefill_sin_.store->ddr.data(),expected_b.data_ptr(),b.nbytes())==0);
  }
 };
 auto pair=[](int64_t rows=800){
  auto a=make_tensor({rows,128},at::kCPU),b=make_tensor({rows,128},at::kCPU);
  auto table=at::arange((rows+8)*128,at::kFloat).reshape({rows+8,128});
  auto positions=at::arange(rows,at::kLong);
  auto cos=at::cos(table/163.0).to(at::kHalf).index_select(0,positions);
  auto sin=at::sin(table/127.0).to(at::kHalf).index_select(0,positions);
  std::memcpy(a.data_ptr(),cos.data_ptr(),a.nbytes());std::memcpy(b.data_ptr(),sin.data_ptr(),b.nbytes());
  assert(a.value.storage_offset()==0&&a.value.nbytes()==a.value.storage().nbytes());
  return std::array<HostTensor,2>{a,b};
 };
 // Four admitted cold profiles, independent normal/force flags and both owner
 // inference modes. Full real gathered payload is checked, not only first/last.
 for(int cores:{4,6})for(bool w8:{false,true})for(bool normal:{false,true})
 for(bool force:{false,true})for(bool inference:{false,true}){
  ++scenarios;c10::InferenceMode mode(inference);g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;
  *shared=Model{};auto& m=*shared;m.reinstall(cores,w8);
  assert(m.pi05_reduced_prefill_rope_receipt_&&!m.prefill_cos_.defined()&&m.generation==10);
  auto inputs=pair();auto a=inputs[0],b=inputs[1];
  auto saved_a=a.value.clone(),saved_b=b.value.clone();const auto av=version(a),bv=version(b);
  auto c=copies,f=flushes.size();op.call(51,a.value,b.value);
  assert(copies-c==2&&flushes.size()-f==(force?2:0));
  assert(m.pi05_prefill_rope_cpu_published_==force&&!m.prefill_rope_ready_);
  assert(m.pi05_reduced_prefill_rope_owner_generation_==m.generation);
  assert(m.pi05_reduced_prefill_rope_keepalive_[0].is_same(m.prefill_cos_));
  assert(m.pi05_reduced_prefill_rope_keepalive_[1].is_same(m.prefill_sin_));
  const auto cp=m.prefill_cos_.data_ptr(),sp=m.prefill_sin_.data_ptr();
  auto fresh=pair();invoke(m,fresh[0],fresh[1],force?0:2,0);
  check_bits(m.prefill_cos_,a,force);check_bits(m.prefill_sin_,b,force);
  // A/B/A writes through the same unversioned CPU source address. Changing only
  // one interior scalar and the sign of zero must invalidate a full-pair hit.
  static_cast<uint16_t*>(a.data_ptr())[173*128+61]^=1;
  invoke(m,a,b,2,force?2:0);invoke(m,a,b,force?0:2,0);
  std::memcpy(a.data_ptr(),saved_a.data_ptr(),a.nbytes());
  invoke(m,a,b,2,force?2:0);invoke(m,a,b,force?0:2,0);
  static_cast<uint16_t*>(b.data_ptr())[603*128+19]=0x8000;
  invoke(m,a,b,2,force?2:0);invoke(m,a,b,force?0:2,0);
  static_cast<uint16_t*>(b.data_ptr())[603*128+19]=0x0000;
  invoke(m,a,b,2,force?2:0);
  raw_bits(a,0x7e11);invoke(m,a,b,2,force?2:0);invoke(m,a,b,force?0:2,0);
  raw_bits(a,0x7e12);invoke(m,a,b,2,force?2:0);
  std::memcpy(a.data_ptr(),saved_a.data_ptr(),a.nbytes());std::memcpy(b.data_ptr(),saved_b.data_ptr(),b.nbytes());
  invoke(m,a,b,2,force?2:0);invoke(m,a,b,force?0:2,0);
  assert(version(a)==av&&version(b)==bv);
  assert(cp==m.prefill_cos_.data_ptr()&&sp==m.prefill_sin_.data_ptr());
  assert(m.prefill_cos_.value.narrow(0,800,800).eq(777).all().item<bool>());
  assert(m.prefill_sin_.value.narrow(0,800,800).eq(777).all().item<bool>());
  a={};b={};inputs={};fresh={};
  assert(std::memcmp(m.prefill_cos_.data_ptr(),saved_a.data_ptr(),saved_a.nbytes())==0);
  assert(std::memcmp(m.prefill_sin_.data_ptr(),saved_b.data_ptr(),saved_b.nbytes())==0);
 }
 g_rpu_ddr_flush_force_enabled=true;g_rpu_ddr_flush_enabled=false;
 for(int cores:{4,6})for(bool w8:{false,true}){
  ++scenarios;*shared=Model{};auto& m=*shared;m.reinstall(cores,w8);
  auto inputs=pair();auto a=inputs[0],b=inputs[1];
  invoke(m,a,b,2,2);invoke(m,a,b,0,0);
  const auto cp=m.prefill_cos_.data_ptr(),sp=m.prefill_sin_.data_ptr();
  auto fallback=[&](const HostTensor& x,const HostTensor& y,size_t n=2){
   invoke(m,x,y,2,n);assert(!m.pi05_prefill_rope_cpu_published_);
   invoke(m,x,y,2,n);assert(!m.pi05_prefill_rope_cpu_published_);
  };
  auto tail=make_tensor({801,128},at::kCPU).narrow(0,0,800);raw_fill(tail,-3);fallback(tail,b);
  auto offset=make_tensor({801,128},at::kCPU).narrow(0,1,800);raw_fill(offset,5);fallback(offset,b);
  auto transposed=tagged(at::arange(800*128,at::kFloat).to(at::kHalf).reshape({128,800}).t());
  assert(!transposed.is_contiguous());fallback(transposed,b);
  check_bits(m.prefill_cos_,tagged(transposed.value.contiguous()),true);
  auto short_inputs=pair(799);fallback(short_inputs[0],short_inputs[1]);
  auto long_inputs=pair(801);fallback(long_inputs[0],long_inputs[1]);
  auto grad=a;grad.value=a.value.clone().set_requires_grad(true);grad.store={};fallback(grad,b);
  // The lazy negative flag is real ATen. This checks that the old copy route is
  // entered, not a claim that the generic backend materializes a lazy view.
  auto neg=a;neg.value=a.value.alias();neg.value.unsafeGetTensorImpl()->_set_neg(true);
  assert(neg.value.is_neg());fallback(neg,b);neg.value.unsafeGetTensorImpl()->_set_neg(false);
  auto conjugate=at::_conj(at::ones({800,128},at::kComplexFloat));
  assert(conjugate.is_conj());auto c=copies,f=flushes.size();bool caught=false;
  try{op.call(51,conjugate,b.value);}catch(const c10::Error&){caught=true;}
  assert(caught&&copies==c&&flushes.size()==f&&!m.pi05_prefill_rope_cpu_published_);
  // Public device values retain their physical input acquire, including a dirty
  // device store whose CPU mapping initially happens to equal the old receipt.
  auto rpu=make_tensor({800,128},at::kPrivateUse1);raw_fill(rpu,7);
  auto* device=reinterpret_cast<c10::Half*>(rpu.store->ddr.data());
  for(int64_t i=0;i<rpu.numel();++i)device[i]=-19;
  rpu.store->device_dirty=true;fallback(rpu,b,3);
  assert(m.prefill_cos_.value.narrow(0,0,800).eq(-19).all().item<bool>());
  fallback(a,rpu,3);fallback(rpu,rpu,4);
  invoke(m,a,b,2,2);invoke(m,a,b,0,0);
  // Force-off never grants authority. Force-on must republish both, regardless
  // of normal flush flag or matching CPU mappings.
  g_rpu_ddr_flush_force_enabled=false;fallback(a,b,0);
  g_rpu_ddr_flush_force_enabled=true;invoke(m,a,b,2,2);invoke(m,a,b,0,0);
  // Real full backing alias must not grant a receipt, even at a distinct
  // TensorImpl and source StorageImpl or unused destination tail.
  auto alias=tagged(at::from_blob(m.prefill_cos_.data_ptr(),{800,128},at::kHalf));fallback(alias,b);
  auto outer=at::from_blob(m.prefill_cos_.data_ptr(),{1600,128},at::kHalf);
  auto alias_tail=tagged(outer.narrow(0,800,800));fallback(alias_tail,b);
  invoke(m,a,b,2,2);invoke(m,a,b,0,0);
  auto owned=m.prefill_cos_;std::weak_ptr<Store> held=owned.store;
  m.prefill_cos_=make_tensor({1600,128},at::kPrivateUse1);
  owned={};assert(!held.expired());fallback(a,b); // strong receipt owner is retained
  m.prefill_cos_=m.pi05_reduced_prefill_rope_keepalive_[0];invoke(m,a,b,2,2);invoke(m,a,b,0,0);
  auto owner=m.prefill_cos_;m.prefill_cos_=owner.narrow(0,0,1600);fallback(a,b);
  m.prefill_cos_=owner;invoke(m,a,b,2,2);invoke(m,a,b,0,0);
  for(int fault:{0,1,2,3}){
   m.prefill_cos_=owner;m.prefill_cos_.override_span=true;
   auto base=reinterpret_cast<uintptr_t>(owner.data_ptr());
   m.prefill_cos_.storage_base=fault==0?0:base;
   m.prefill_cos_.storage_bytes=fault==1?std::numeric_limits<size_t>::max():
       fault==2?owner.nbytes()-1:owner.nbytes()+2;
   fallback(a,b);
  }
  m.prefill_cos_=owner;invoke(m,a,b,2,2);invoke(m,a,b,0,0);
  for(int fail_after:{0,1}){
   raw_fill(a,29+fail_after);raw_fill(b,-29-fail_after);flushes_until_failure=fail_after;caught=false;
   try{op.call(51,a.value,b.value);}catch(const c10::Error&){caught=true;}
   assert(caught&&!m.pi05_prefill_rope_cpu_published_);
   invoke(m,a,b,2,2);invoke(m,a,b,0,0);check_bits(m.prefill_cos_,a,true);check_bits(m.prefill_sin_,b,true);
  }
  // An actual install generation cannot be replaced by the old generic zero.
  ++m.generation;fallback(a,b);
  assert(cp==m.prefill_cos_.data_ptr()&&sp==m.prefill_sin_.data_ptr());
  auto old_c=m.prefill_cos_,old_s=m.prefill_sin_;const auto gen=m.generation;
  try{m.reinstall(cores,w8,1600,1);assert(false);}catch(const c10::Error&){}
  assert(!m.pi05_reduced_prefill_rope_receipt_&&!m.pi05_prefill_rope_private_owned_&&!m.pi05_prefill_rope_cpu_published_);
  assert(m.generation==gen);fallback(a,b);
  m.reinstall(cores,w8);assert(m.generation==gen+1&&m.pi05_reduced_prefill_rope_receipt_);
  assert(old_c.is_same(m.prefill_cos_)&&old_s.is_same(m.prefill_sin_));fallback(a,b);
  assert(!m.pi05_reduced_prefill_rope_keepalive_[0].defined()&&!m.pi05_reduced_prefill_rope_keepalive_[1].defined());
  try{m.reinstall(cores,w8,1600,2);assert(false);}catch(const std::invalid_argument&){}
  assert(!m.pi05_reduced_prefill_rope_receipt_&&!m.pi05_prefill_rope_cpu_published_);
 }
 // Both allocation completion and both forced publications are necessary.
 for(int fail_after:{0,1}){
  *shared=Model{};auto& m=*shared;m.reinstall(4,false);auto inputs=pair();
  allocation_until_failure=fail_after;bool caught=false;
  try{op.call(51,inputs[0].value,inputs[1].value);}catch(const c10::Error&){caught=true;}
  assert(caught&&!m.pi05_prefill_rope_private_owned_&&!m.pi05_prefill_rope_cpu_published_);
  assert(!m.pi05_reduced_prefill_rope_keepalive_[0].defined()&&!m.pi05_reduced_prefill_rope_keepalive_[1].defined());
 }
 // Inference owner mutation restrictions cannot be bypassed by equal bytes.
 HostTensor inference_a,inference_b;
 {
  c10::InferenceMode mode(true);*shared=Model{};shared->reinstall(6,true);auto input=pair();
  inference_a=input[0];inference_b=input[1];invoke(*shared,inference_a,inference_b,2,2);
  invoke(*shared,inference_a,inference_b,0,0);assert(shared->prefill_cos_.is_inference());
 }
 auto c=copies,f=flushes.size();bool caught=false;
 try{op.call(51,inference_a.value,inference_b.value);}catch(const c10::Error&){caught=true;}
 assert(caught&&copies-c==1&&flushes.size()-f==1&&!shared->pi05_prefill_rope_cpu_published_);
 {
  c10::InferenceMode mode(true);invoke(*shared,inference_a,inference_b,2,2);invoke(*shared,inference_a,inference_b,0,0);
 }
 // Zero install generations or unadmitted cold topology retain original copy.
 for(int variant:{0,1,2}){
  *shared=Model{};auto& m=*shared;m.reinstall(4,false);auto input=pair();
  if(variant==0)m.generation=0;
  if(variant==1)m.pi05_reduced_prefill_rope_receipt_=false;
  if(variant==2){m.prefill_cos_=make_tensor({1600,128},at::kPrivateUse1);m.prefill_sin_=make_tensor({1600,128},at::kPrivateUse1);}
  invoke(m,input[0],input[1],2,2);invoke(m,input[0],input[1],2,2);assert(!m.pi05_prefill_rope_cpu_published_);
 }
 // TP8 direct still owns readiness and accepts its original offset CPU ranges.
 for(int64_t rows:{544,800}){
  *shared=Model{};auto& m=*shared;m.pi05_prefill_kv1_pair_direct_=true;m.prefix_rows=rows;m.reinstall(8,false,rows+16);
  assert(!m.pi05_reduced_prefill_rope_receipt_);
  auto a=make_tensor({rows+2,128},at::kCPU).narrow(0,1,rows),b=make_tensor({rows,128},at::kCPU);
  raw_fill(a,-7);raw_fill(b,13);invoke(m,a,b,2,2);invoke(m,a,b,0,0);assert(m.prefill_rope_ready_);
  auto neg=a;neg.value=a.value.alias();neg.value.unsafeGetTensorImpl()->_set_neg(true);
  invoke(m,neg,b,2,2);assert(!m.pi05_prefill_rope_cpu_published_);neg.value.unsafeGetTensorImpl()->_set_neg(false);
 }
 std::cout<<"whole_setter_profile_cases="<<scenarios<<" full payload/private owner/copy controls passed\n";
}

'''
    for marker, value in (("__SETTER__", setter), ("__FIELDS__", fields),
                          ("__WEIGHTS_ENTRY__", entry), ("__ALLOCATION__", allocation),
                          ("__COLD_VALIDATION__", cold_validation), ("__COLD_ADMISSION__", admission),
                          ("__TOPOLOGY__", topology), ("__INSTALL_END__", install_end),
                          ("__WRAPPER__", wrapper), ("__SCHEMA__", schema),
                          ("__REGISTRATIONS__", "\n".join(registrations)),
                          ("__PREFLIGHT__", preflight), ("__SAME_DEVICE__", same_device),
                          ("__CROSS_DEVICE__", cross_device), ("__ELEMENTWISE__", elementwise)):
        assert marker in code
        code = code.replace(marker, value)
    assert not re.search(r"__[A-Z_]+__", code)
    result = _run_native(tmp_path, code, torch)
    assert "full payload/private owner/copy controls passed" in result
