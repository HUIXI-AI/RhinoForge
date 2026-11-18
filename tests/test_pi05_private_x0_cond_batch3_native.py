"""Execute the whole native forward; SDK/cache/Graph/FMB kernels are host doubles.

This is a coherency and lifetime test, not a simulation of Pi mathematics or
hardware. Real ATen owns every ordinary/inference tensor and storage view.
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
    assert compiler, "whole Pi forward publication contract requires C++"
    cpp, binary = tmp_path / "pi_batch3.cpp", tmp_path / "pi_batch3"
    cpp.write_text(source)
    torch_root = Path(torch.__file__).resolve().parent
    command = [compiler, "-std=c++17", "-O0", "-D_GLIBCXX_USE_CXX11_ABI=" +
               str(int(torch.compiled_with_cxx11_abi())),
               "-I", str(torch_root / "include"),
               "-I", str(torch_root / "include/torch/csrc/api/include"), str(cpp),
               "-L", str(torch_root / "lib"), "-Wl,-rpath," + str(torch_root / "lib"),
               "-ltorch_cpu", "-lc10", "-o", str(binary)]
    built = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stdout + built.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_whole_denoise_forward_private_batch3_dirty_alias_and_delayed_graph(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    source = (ROOT / "src/fused/rpu_pi05_denoise_step_model.cpp").read_text()
    profile = (ROOT / "src/core/rpu_profile.h").read_text()
    forward = _function(source, "void Pi05DenoiseStepModel::denoise_loop_forward(")
    # Only device labels, the allocator, cache dirtiness and the unavailable
    # AArch64 barrier instruction are substituted. The COMPLETE function's
    # guard/mode/descriptor/DT/memcpy/bounds/publication/lease/call order runs.
    forward = forward.replace("x0_rpu.device().is_cpu()", "cpu_label(x0_rpu)")
    forward = forward.replace("x0_rpu.device().type() == at::kPrivateUse1",
                              "(!cpu_label(x0_rpu) && x0_rpu.device().type() == at::kCPU)")
    forward = forward.replace("at::kPrivateUse1", "at::kCPU")
    allocation = """at::empty(
            x0_rpu.sizes(), at::TensorOptions().dtype(at::kHalf).device(x_out.device()))"""
    assert forward.count(allocation) == 1
    forward = forward.replace(allocation, "private_bank(x0_rpu.sizes())")
    copy = "std::memcpy(private_x0.data_ptr(), x0_rpu.const_data_ptr(), x0_rpu.nbytes());"
    assert forward.count(copy) == 1
    forward = forward.replace(copy, copy + "\n        mark_dirty(private_x0);")
    forward = forward.replace('#if defined(__aarch64__)\n', '').replace('#endif\n', '')
    forward = forward.replace('asm volatile("dsb sy" ::: "memory");', "host_dsb_sy();")
    macros = "\n".join(profile[profile.index(f"#define {name}("):].split("\n\n", 1)[0]
                       for name in ("rpu_ddr_flush_force", "rpu_ddr_flush_force_sized"))
    cpp = r'''
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/intrusive_ptr.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>
bool g_rpu_ddr_flush_enabled = true, g_rpu_ddr_flush_force_enabled = true;
struct Store {
 at::Tensor backing; void* pointer; size_t bytes;
 std::vector<uint8_t> ddr, dirty;
 c10::weak_intrusive_ptr<c10::StorageImpl> owner;
 explicit Store(at::Tensor t):backing(t),pointer(t.data_ptr()),bytes(t.nbytes()),
  ddr(bytes,0),dirty(bytes,0),owner(t.storage().getWeakStorageImpl()){}
};
std::vector<Store> stores;
struct Publication {void* pointer; size_t bytes; bool unsized;};
std::vector<Publication> publications;
std::vector<size_t> batch_counts;
std::vector<std::string> ordering;
std::vector<std::function<void()>> nodes;
int owner_queries=0, fault=0, fault_target=0;
bool graph_active=false, recording=false, replaying=false;
bool pool_private=false; int64_t private_pool_offset=0;
at::Tensor private_pool;
std::unordered_map<const c10::TensorImpl*,
 c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl>> cpu_inputs;
bool cpu_label(const at::Tensor& t) {
 auto it=cpu_inputs.find(t.unsafeGetTensorImpl());
 return it!=cpu_inputs.end()&&!it->second.expired();
}
void label_cpu(const at::Tensor& t) {
 cpu_inputs.insert_or_assign(t.unsafeGetTensorImpl(),
  c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl>(t.getIntrusivePtr()));
}
Store& locate(void* p,size_t n,size_t& offset) {
 for(auto i=stores.rbegin();i!=stores.rend();++i) {
  const auto b=reinterpret_cast<uintptr_t>(i->pointer),v=reinterpret_cast<uintptr_t>(p);
  if(v>=b && v-b<=i->bytes && n<=i->bytes-(v-b) && !i->owner.expired()) {
   offset=v-b;return *i;
  }
 }
 TORCH_CHECK(false,"invalid or expired host simulated DDR owner");
}
at::Tensor bank(at::IntArrayRef shape) {
 auto t=at::zeros(shape,at::kHalf);stores.emplace_back(t);return t;
}
at::Tensor private_bank(at::IntArrayRef shape) {
 int64_t count=1;for(auto d:shape)count*=d;
 if(pool_private)return private_pool.narrow(0,private_pool_offset,count).view(shape);
 // Extra 64-byte edges exercise the full rounded range against its owner.
 return bank({count+64}).narrow(0,16,count).view(shape);
}
void mark_dirty(const at::Tensor& t) {
 size_t off;auto& s=locate(t.data_ptr(),t.nbytes(),off);
 std::fill(s.dirty.begin()+off,s.dirty.begin()+off+t.nbytes(),1);
}
void fill_cpu_alias(const at::Tensor& t,double value) {
 // Raw CPU alias writes deliberately do not increment the ATen version.
 auto* p=static_cast<c10::Half*>(t.data_ptr());
 for(int64_t i=0;i<t.numel();++i)p[i]=c10::Half(static_cast<float>(value));
}
void fill_dirty(const at::Tensor& t,double value) {
 fill_cpu_alias(t,value);mark_dirty(t);
}
void host_dsb_sy(){ordering.push_back("backend dsb sy");}
namespace rhino_lkn {
struct HostDDR_t {void* base;size_t bytes;
 void* get_cpu_ptr(){return base;}size_t get_memory_size(){return bytes;}};
HostDDR_t* RpuGetHostddr(void* p) {
 const int index=owner_queries++;ordering.push_back("owner query");
 size_t off;auto& s=locate(p,0,off);
 static HostDDR_t owner;owner={s.pointer,s.bytes};
 if(fault && index==fault_target) {
  const uintptr_t address=reinterpret_cast<uintptr_t>(p);
  if(fault==1)return nullptr;
  if(fault==2)owner.base=nullptr;
  if(fault==3)owner.bytes=0;
  if(fault==4)owner.base=reinterpret_cast<void*>(address+1);
  if(fault==5)owner.bytes=off+1; // truncate the current logical range
  if(fault==6)owner.bytes=std::numeric_limits<size_t>::max();
  if(fault==7){owner.base=reinterpret_cast<void*>(address-16);owner.bytes=s.bytes-off+16;}
  if(fault==8){owner.base=reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()-15);owner.bytes=64;}
  if(fault==9)TORCH_CHECK(false,"SDK owner query failed");
 }
 return &owner;
}
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uint64_t>(p);}
void flush_range(void* p,size_t n) {
 size_t off;auto& s=locate(p,n,off);
 const size_t begin=off&~size_t(63),end=(off+n+63)&~size_t(63);
 assert(end<=s.bytes);
 auto* cpu=static_cast<uint8_t*>(s.pointer);
 // Explicit dc-civac model: dirty host bytes publish; clean bytes invalidate
 // from DDR. This exposes missing input publication and dirty output hazards.
 for(size_t i=begin;i<end;++i) {
  if(s.dirty[i])s.ddr[i]=cpu[i];else cpu[i]=s.ddr[i];s.dirty[i]=0;
 }
}
void RpuDdrFlush(void* p,size_t n) {
 publications.push_back({p,n,false});ordering.push_back("scalar sized");flush_range(p,n);
}
void RpuDdrFlush(void* p) {
 size_t off;auto& s=locate(p,0,off);
 publications.push_back({p,s.bytes-off,true});ordering.push_back("scalar unsized");
 flush_range(p,s.bytes-off);
}
void RpuDdrFlushBatch(void** p,size_t* sizes,size_t count) {
 batch_counts.push_back(count);ordering.push_back("SDK Batch");
 for(size_t i=0;i<count;++i){publications.push_back({p[i],sizes[i],false});flush_range(p[i],sizes[i]);}
 ordering.push_back("SDK dsb ish");
}
}
__MACROS__
struct Graph {std::vector<at::Tensor> held;
 void keep_alive(const at::Tensor& t){held.push_back(t);}} graph;
struct RpuKernelGraph {static bool has_active(){return graph_active;}
 static Graph& active(){return graph;}};
at::Tensor read_ddr(void* p,at::IntArrayRef shape) {
 int64_t count=1;for(auto dim:shape)count*=dim;
 size_t off;auto& s=locate(p,count*2,off);
 return at::from_blob(s.ddr.data()+off,shape,at::TensorOptions().dtype(at::kHalf)).clone();
}
bool equal_bytes(const at::Tensor& a,const at::Tensor& b) {
 assert(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type());
 assert(a.is_contiguous()&&b.is_contiguous()&&a.nbytes()==b.nbytes());
 return std::memcmp(a.const_data_ptr(),b.const_data_ptr(),a.nbytes())==0;
}
void write_ddr(void* p,const at::Tensor& values) {
 size_t off;auto& s=locate(p,values.nbytes(),off);
 std::memcpy(s.ddr.data()+off,values.const_data_ptr(),values.nbytes());
}
int64_t version(const at::Tensor& t){return t.is_inference()?-1:t._version();}
struct Pi05DenoiseStepModel {
 int64_t chunk_size_=50,max_action_dim_=32,configured_chunk_size_=0,num_steps_=10;
 int cores=4,layers=1,invalidations=0,forwards=0;
 bool loop_mode_=true,dt_pinned_=false;
 c10::Half dt_=c10::Half(-0.1f);int pi05_graph_dma_policy_=0;
 bool pi05_w4_geglu_m50_opt_in_=false,pi05_xor3_gated_mlp_opt_in_=false;
 bool pi05_geglu_m50_opt_in_=false,pi05_q_rope_epilogue_opt_in_=false;
 at::Tensor adarms_table_,x0_ref_,cond_all_ref_,x_out_ref_,action_emb_stage_;
 uint64_t x0_src_base_=0,cond_all_src_base_=0,x_out_dst_base_=0;
 at::Tensor expected_x0,expected_cond,expected_output,consumed_x0,consumed_cond;
 int num_layers(){return layers;}int64_t hidden_size(){return 1024;}
 int num_cores(){return cores;}
 bool pi05_action_prefix_admitted(int64_t){return true;}
 bool pi05_action_steps_admitted(int64_t){return true;}
 bool pi05_action_rows_admitted(int64_t rows){return rows==50||rows==10;}
 void restore_planned_route_profile(at::IntArrayRef,bool,int64_t,int64_t,int64_t){}
 void invalidate_model_state(bool){++invalidations;}
 void validate_planned_route_profile(bool,int64_t){}
 void validate_pi05_storage_policy(at::IntArrayRef){}
 int prepare_pi05_graph_dma_policy(at::IntArrayRef){return 0;}
 void set_chunk_size_override(int64_t){}
 void prepare_sdpa_mask_cached(const at::Tensor&,int64_t){}
 int run_all_layers(const at::Tensor&,const std::vector<at::Tensor>&,
   const std::vector<at::Tensor>&,std::optional<at::Tensor>,int64_t,bool,int64_t,at::IntArrayRef) {
  ++forwards;
  auto consume=[this]{
   ordering.push_back("first DMA consumer");
   // Independent oracle checks all current logical bytes, including every
   // cond row; SDK/query/publication code has no access to these oracles.
   consumed_x0=read_ddr(reinterpret_cast<void*>(x0_src_base_),{1,50,32});
   assert(equal_bytes(consumed_x0,expected_x0));
   if(!adarms_table_.defined()) {
    consumed_cond=read_ddr(reinterpret_cast<void*>(cond_all_src_base_),{num_steps_,1024});
    assert(equal_bytes(consumed_cond,expected_cond));
   }
   // Kernel arithmetic is explicitly a boundary double, not Pi math proof.
   // Device-written DDR output is the independent CPU oracle supplied by caller.
   write_ddr(reinterpret_cast<void*>(x_out_dst_base_),expected_output);
  };
  if(recording)nodes.push_back(consume);else if(!replaying)consume();
  return 0;
 }
 void denoise_loop_forward(at::Tensor,std::vector<at::Tensor>,std::vector<at::Tensor>,
   at::Tensor,at::Tensor,at::Tensor,double,int64_t,int64_t,at::IntArrayRef);
};
__WHOLE_FORWARD__
void reset() {
 graph.held.clear();nodes.clear();private_pool=at::Tensor();pool_private=false;
 stores.clear();publications.clear();batch_counts.clear();ordering.clear();cpu_inputs.clear();
 fault=owner_queries=0;recording=replaying=graph_active=false;
}
void clear_observation(){publications.clear();batch_counts.clear();ordering.clear();owner_queries=0;}
void oracle(Pi05DenoiseStepModel& m,const at::Tensor& input,const at::Tensor& cond) {
 // ATen independently owns geometry, dtype, finite values, byte equality and
 // clone retention; forceOFF consumes the explicitly initialized stale DDR.
 m.expected_x0=g_rpu_ddr_flush_force_enabled?input.clone():at::zeros_like(input);
 m.expected_cond=g_rpu_ddr_flush_force_enabled?cond.clone():at::zeros_like(cond);
 m.expected_output=(input.to(at::kFloat)+
  (m.adarms_table_.defined()?0.0:cond.to(at::kFloat).sum().item<double>()/cond.numel())).to(at::kHalf);
 assert(at::isfinite(m.expected_output).all().item<bool>());
}
void invoke(Pi05DenoiseStepModel& m,const at::Tensor& input,const at::Tensor& cond,const at::Tensor& out,int steps=10) {
 m.denoise_loop_forward(input,{}, {},cond,at::Tensor(),out,-1.0/steps,0,steps,{1});
}
void check_publication(Pi05DenoiseStepModel& m,const at::Tensor& cond,const at::Tensor& out,
                       bool cpu,bool table,bool admit) {
 const bool force=g_rpu_ddr_flush_force_enabled;
 if(!force){assert(publications.empty()&&batch_counts.empty());}
 else {
  assert(publications.size()==size_t(table?2:3));
  assert(publications[0].pointer==m.x0_ref_.data_ptr()&&publications[0].bytes==3200);
  if(!table) {
   assert(publications[1].pointer==cond.data_ptr());
   assert(publications[1].unsized!=admit);
   if(admit)assert(publications[1].bytes==cond.nbytes());
  }
  assert(publications.back().pointer==out.data_ptr()&&publications.back().bytes==3200);
  if(cpu && (table||admit)) {
   assert(batch_counts==std::vector<size_t>{size_t(table?2:3)});
   auto q=std::find(ordering.begin(),ordering.end(),"SDK Batch");
   assert(std::count(ordering.begin(),q,"owner query")==int(table?2:3));
   assert(std::find(q,ordering.end(),"owner query")==ordering.end());
   assert(std::find(q,ordering.end(),"SDK dsb ish")!=ordering.end());
  } else assert(batch_counts.empty());
 }
 if(cpu) {
  assert(std::count(ordering.begin(),ordering.end(),"backend dsb sy")==1);
  auto first=std::find(ordering.begin(),ordering.end(),"first DMA consumer");
  if(first!=ordering.end())assert(std::find(ordering.begin(),first,"backend dsb sy")!=first);
 }
}
void acquire_and_check(const at::Tensor& out,const at::Tensor& expected) {
 if(!g_rpu_ddr_flush_force_enabled)return; // forceOFF numeric safety is not claimed
 rhino_lkn::flush_range(out.data_ptr(),out.nbytes());assert(equal_bytes(out,expected));
}
int main() {
 int cases=0;
 for(int cores:{4,6,8})for(bool normal:{false,true})for(bool force:{false,true})
  for(bool inference:{false,true})for(bool cpu:{false,true})for(bool table:{false,true}) {
  ++cases;reset();c10::InferenceMode mode(inference);
  g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;
  Pi05DenoiseStepModel m;m.cores=cores;if(table)m.adarms_table_=at::ones({1},at::kHalf);
  auto input=cpu?at::zeros({1,50,32},at::kHalf):bank({1,50,32});if(cpu)label_cpu(input);
  auto cond=bank({12,1024}).narrow(0,1,10); // offset plus unrelated owner tail
  const auto iv=version(input),cv=version(cond);std::vector<at::Tensor> retained,retained_bytes;
  for(int pass=0;pass<3;++pass) {
   const double value=pass==1?-3.5:1.25;
   auto out=bank({1664}).narrow(0,32,1600).view({1,50,32});fill_dirty(out,9);
   if(cpu)fill_cpu_alias(input,value);else fill_dirty(input,value);
   // Include different values in ALL cond rows through raw public CPU aliases.
   auto* cp=static_cast<c10::Half*>(cond.data_ptr());
   for(int64_t i=0;i<cond.numel();++i)cp[i]=c10::Half(float((i/1024+pass)%7-3)*0.25f);
   mark_dirty(cond);oracle(m,input,cond);clear_observation();invoke(m,input,cond,out);
   check_publication(m,cond,out,cpu,table,cpu&&!table);
   acquire_and_check(out,m.expected_output);retained.push_back(out);retained_bytes.push_back(out.clone());
   if(pass>0)assert(retained[0].data_ptr()!=retained[pass].data_ptr());
   for(size_t previous=0;previous<retained.size();++previous)
    assert(equal_bytes(retained[previous],retained_bytes[previous]));
   assert(version(cond)==cv&&version(input)==iv);
  }
 }
 // Actual whole-forward BUILD/REPLAY submission runs after call return. Drop
 // the registry/model strong x0 refs; production Graph.keep_alive must retain
 // the fresh full-copy payload while the original CPU input is dirtied again.
 for(bool table:{false,true})for(bool inference:{false,true}) {
  ++cases;reset();c10::InferenceMode mode(inference);
  g_rpu_ddr_flush_force_enabled=true;g_rpu_ddr_flush_enabled=false;
  Pi05DenoiseStepModel m;if(table)m.adarms_table_=at::ones({1},at::kHalf);
  auto input=at::zeros({1,50,32},at::kHalf);label_cpu(input);auto cond=bank({10,1024});
  graph_active=true;recording=true;std::vector<at::Tensor> saved;
  for(int pass=0;pass<3;++pass) {
   fill_cpu_alias(input,pass==1?-2.5:2);fill_dirty(cond,pass==1?0.75:0.25);
   auto out=bank({1,50,32});fill_dirty(out,9);oracle(m,input,cond);
   clear_observation();graph.held.clear();invoke(m,input,cond,out);
   check_publication(m,cond,out,true,table,!table);assert(graph.held.size()==1);
   c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl> lease(m.x0_ref_.getIntrusivePtr());
   size_t off;locate(m.x0_ref_.data_ptr(),m.x0_ref_.nbytes(),off).backing=at::Tensor();
   m.x0_ref_=at::Tensor();assert(!lease.expired());fill_cpu_alias(input,99);
   if(pass==0){recording=false;replaying=true;assert(nodes.size()==1);}
   for(auto& node:nodes)node();assert(nodes.size()==1);
   acquire_and_check(out,m.expected_output);saved.push_back(out);
   if(pass==1)assert(!equal_bytes(saved[0],saved[1]));
   if(pass==2)assert(equal_bytes(saved[0],saved[2]));
   graph.held.clear();assert(lease.expired());
  }
  assert(m.forwards==3);
 }
 // Same SDK owner is legal: the three 64-rounded intervals are disjoint.
 // Offset32 is also legal if rounded endpoints fit, without requiring exact
 // tensor/storage base identity or a different SDK owner for each operand.
 for(int layout:{0,1,2,3}) {
  ++cases;reset();g_rpu_ddr_flush_force_enabled=true;
  Pi05DenoiseStepModel m;auto input=at::ones({1,50,32},at::kHalf);label_cpu(input);
  private_pool=bank({16384});pool_private=true;private_pool_offset=14000;
  int64_t cond_start=(layout==1||layout==3)?16:32;
  auto cond=private_pool.narrow(0,cond_start,10240).view({10,1024});
  int64_t output_start=11000;
  if(layout==2)output_start=cond_start+10000; // actual overlap
  if(layout==3)output_start=cond_start+10240+8; // disjoint bytes, rounded overlap
  auto out=private_pool.narrow(0,output_start,1600).view({1,50,32});
  fill_dirty(cond,0.5);fill_dirty(out,9);oracle(m,input,cond);clear_observation();invoke(m,input,cond,out);
  check_publication(m,cond,out,true,false,layout<2);acquire_and_check(out,m.expected_output);
 }
 // Fault only fast-path SDK owner metadata; actual ATen owner/current byte
 // buffers remain valid so the unchanged scalar fallback can be checked.
 for(int target:{0,1,2})for(int failure:{1,2,3,4,5,6,7,8}) {
  ++cases;reset();g_rpu_ddr_flush_force_enabled=true;
  Pi05DenoiseStepModel m;auto input=at::full({1,50,32},-3.5,at::kHalf);label_cpu(input);
  auto cond=bank({11264}).narrow(0,16,10240).view({10,1024});
  auto out=bank({1664}).narrow(0,16,1600).view({1,50,32});
  fill_dirty(cond,0.25);fill_dirty(out,9);oracle(m,input,cond);
  fault=failure;fault_target=target;clear_observation();invoke(m,input,cond,out);
 check_publication(m,cond,out,true,false,false);acquire_and_check(out,m.expected_output);
 }
 // SDK query exceptions propagate before any publication or consumer. They
 // must not be converted into scalar fallback or a partial Batch operation.
 for(int target:{0,1,2}) {
  ++cases;reset();g_rpu_ddr_flush_force_enabled=true;graph_active=true;
  Pi05DenoiseStepModel m;auto input=at::ones({1,50,32},at::kHalf);label_cpu(input);
  auto cond=bank({10,1024}),out=bank({1,50,32});fault=9;fault_target=target;
  bool failed=false;
  try{invoke(m,input,cond,out);}catch(const c10::Error& e){
   failed=std::string(e.what()).find("SDK owner query failed")!=std::string::npos;
  }
  assert(failed&&owner_queries==target+1&&publications.empty()&&batch_counts.empty());
  assert(m.forwards==0&&graph.held.empty()&&nodes.empty());
 }
 // Unusual public views retain the original scalar behavior instead of
 // acquiring new fast-path authority. Test their underlying bytes, as the
 // existing native interface does, without claiming lazy-view math support.
 for(int unusual:{0,1,2,3}) {
  ++cases;reset();g_rpu_ddr_flush_force_enabled=true;
  Pi05DenoiseStepModel m;auto input=at::ones({1,50,32},at::kHalf);label_cpu(input);
  auto cond=bank({10,1024}),out=bank({1,50,32});fill_dirty(cond,0.5);fill_dirty(out,9);
  if(unusual==0)cond=at::_neg_view(cond);
  if(unusual==1)out=at::_neg_view(out);
  if(unusual==2)cond.set_requires_grad(true);
  if(unusual==3)out.set_requires_grad(true);
  oracle(m,input,cond);
  m.expected_cond=at::from_blob(cond.data_ptr(),cond.sizes(),at::TensorOptions().dtype(at::kHalf)).clone();
  clear_observation();invoke(m,input,cond,out);check_publication(m,cond,out,true,false,false);
 }
 // Tensor-dependent full byte sizes: do not bake D10's 20480 into the Batch.
 for(int steps:{1,3,10}) {
  ++cases;reset();g_rpu_ddr_flush_force_enabled=true;
  Pi05DenoiseStepModel m;auto input=at::ones({1,50,32},at::kHalf);label_cpu(input);
  auto cond=bank({steps,1024}),out=bank({1,50,32});fill_dirty(cond,0.25);fill_dirty(out,9);
  oracle(m,input,cond);invoke(m,input,cond,out,steps);
  check_publication(m,cond,out,true,false,true);assert(publications[1].bytes==size_t(steps*2048));
 }
 // Errors from the original complete function must precede private allocation,
 // publication, model consumption and Graph lease. Ordinary negative VALUES
 // above remain legal; a lazy negative CPU VIEW is not ordinary current bytes.
 for(int bad=0;bad<10;++bad) {
  ++cases;reset();g_rpu_ddr_flush_force_enabled=true;graph_active=true;
  Pi05DenoiseStepModel m;auto input=at::ones({1,50,32},at::kHalf);label_cpu(input);
  auto cond=bank({10,1024}),out=bank({1,50,32});int steps=10;double dt=-0.1;
  if(bad==0)input=input.to(at::kFloat);
  if(bad==1)input=at::zeros({1,50,64},at::kHalf).slice(2,0,64,2);
  if(bad==2)input=input.narrow(1,0,49);
  if(bad==3)input=at::_neg_view(input);
  if(bad==4)input=input.clone().set_requires_grad(true);
  if(bad==5)cond=cond.to(at::kFloat);
  if(bad==6)out=out.narrow(1,0,49);
  if(bad==7)m.layers=0;
  if(bad==8)dt=0.1;
  if(bad==9){steps=0;cond=bank({0,1024});}
  label_cpu(input);const auto allocated=stores.size();bool failed=false;
  try{m.denoise_loop_forward(input,{}, {},cond,at::Tensor(),out,dt,0,steps,{1});}
  catch(const c10::Error&){failed=true;}
  assert(failed&&stores.size()==allocated&&publications.empty()&&batch_counts.empty());
  assert(m.forwards==0&&graph.held.empty()&&nodes.empty());
 }
 std::cout<<"whole forward Batch3 host contract "<<cases<<" cases\n";
}
'''
    cpp = cpp.replace("__MACROS__", macros).replace("__WHOLE_FORWARD__", forward)
    output = _run_native(tmp_path, cpp, torch)
    assert output.strip() == "whole forward Batch3 host contract 148 cases"
