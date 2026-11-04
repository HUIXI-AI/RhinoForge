"""Run native M-RoPE refresh predicates against real Tensor version semantics."""
import ctypes
from pathlib import Path
import re
import shutil
import subprocess

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def native_refresh(tmp_path_factory):
    source = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    cpp = """
#include <cstdint>
struct Counter {
    bool tracked;
    int64_t version;
    bool enabled() const { return tracked; }
    int64_t current_version() const { return version; }
};
struct Tensor {
    Counter counter;
    uintptr_t address;
    const Tensor* unsafeGetTensorImpl() const { return this; }
    const Counter& version_counter() const { return counter; }
    const void* data_ptr() const { return reinterpret_cast<const void*>(address); }
};
"""
    for name, tensor in (("pos", "pos"), ("cos", "cil"), ("sin", "sil")):
        start = source.index(f"const auto& {name}_version_counter =")
        end = source.index("if (", start)
        # Compile the real version extraction, content-change predicate, and
        # ordinary branch condition. Only the Tensor/SDK boundary is doubled.
        declarations = source[start:end]
        ordinary = re.search(r": \((.*?content_changed)\)", source[end:], re.S).group(1)
        cpp += f"""
extern "C" bool refresh_{name}(bool tracked, int64_t version, uintptr_t address,
    int64_t previous_version, uintptr_t previous_address, bool ka_geom_same) {{
    const Tensor {tensor}{{{{tracked, version}}, address}};
    const int64_t ka_last_{name}_src_version_ = previous_version;
    const void* ka_last_{name}_src_ = reinterpret_cast<const void*>(previous_address);
    {declarations}
    return {ordinary};
}}
"""
    folder = tmp_path_factory.mktemp("mrope-version-native")
    path = folder / "refresh.cpp"
    path.write_text(cpp)
    library = folder / "refresh.so"
    compiler = shutil.which("g++")
    assert compiler, "host native contract tests require g++"
    subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", str(path),
                    "-o", str(library)], check=True, capture_output=True, text=True)
    module = ctypes.CDLL(str(library))
    for name in ("pos", "cos", "sin"):
        function = getattr(module, f"refresh_{name}")
        function.argtypes = [ctypes.c_bool, ctypes.c_int64, ctypes.c_size_t,
                             ctypes.c_int64, ctypes.c_size_t, ctypes.c_bool]
        function.restype = ctypes.c_bool
    return module


@pytest.mark.parametrize("name", ["pos", "cos", "sin"])
@pytest.mark.parametrize("inference", [False, True])
def test_same_geometry_slot_refreshes_changed_content(
    native_refresh, name, inference,
):
    refresh = getattr(native_refresh, f"refresh_{name}")
    with torch.inference_mode(inference):
        dtype = torch.int32 if name == "pos" else torch.float16
        source = torch.full((320, 3), 41, dtype=dtype)
        keepalive = torch.empty_like(source)
        pointer = source.data_ptr()
        previous_version, previous_address = -1, 0
        copies = 0

        def copy_if_needed():
            nonlocal previous_version, previous_address, copies
            version = -1 if torch.is_inference(source) else source._version
            needed = refresh(not torch.is_inference(source), version,
                             source.data_ptr(), previous_version,
                             previous_address, True)
            if needed:
                keepalive.copy_(source)
                previous_version, previous_address = version, source.data_ptr()
                copies += 1
            return needed

        assert copy_if_needed()
        assert torch.equal(keepalive, source)
        # Tracked unchanged sources retain the cache hit. Untracked inference
        # storage cannot prove a hit even when its current bytes happen to match.
        assert copy_if_needed() is inference
        assert copies == (2 if inference else 1)

        # Dual P313 and photo P314 both execute E320 in the same stable slot.
        # A changed M-RoPE payload must refresh before any intervening decode.
        source.fill_(33)
        assert source.data_ptr() == pointer
        assert not torch.equal(keepalive, source)
        assert copy_if_needed()
        assert torch.equal(keepalive, source)
        assert copy_if_needed() is inference


def test_w8_segment_cache_already_requires_tracked_content():
    source = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    hit = source.split("static bool keepalive_segment_hit(", 1)[1].split(
        "static void record_keepalive_segment(", 1)[0]
    assert hit.index("if (!version_counter.enabled()) return false;") < hit.index("cache.find(key)")
    record = source.split("static void record_keepalive_segment(", 1)[1].split(
        "KeepaliveSourceCache ka_pos_sources_;", 1)[0]
    assert record.index("if (version_counter.enabled()) {") < record.index("cache.emplace(key,")


def test_native_cpu_positions_refresh_aliases_and_invalidate_overlapping_sources(tmp_path):
    """Actual prologue, stamps and copy transport; real ATen storage/versions.

    Device labels and SDK visibility use CPU/DDR byte-store doubles. This is
    publication/ownership evidence, not physical SDK or Graph certification.
    """
    from test_siglip_attention_dry_live_contract import _function, _run_native
    source = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    start = source.index('            TORCH_CHECK(position_ids.has_value(),')
    end = source.index('            // partial_mrope:', start)
    prologue = source[start:end]
    declarations = source[source.index('    using KeepaliveSegmentKey ='):
                          source.index('    static bool keepalive_segment_hit(')]
    methods = (_function(source, '    static bool keepalive_segment_hit(') + '\n' +
               _function(source, '    static void record_keepalive_segment('))
    authority_fields = source[source.index('    std::vector<uint8_t> ka_cpu_pos_published_rows_'):
                              source.index('    void invalidate_cpu_position_keepalive(')]
    authority_methods = '\n'.join(_function(source, marker) for marker in (
        '    void invalidate_cpu_position_keepalive(',
        '    bool cpu_position_source_is_disjoint(',
        '    bool cpu_position_keepalive_hit(',
        '    void mark_cpu_position_keepalive(',
    ))
    owner_export = _function(source, '    DecodeAuxiliaryTensorOwners decode_auxiliary_tensor_owners() const')
    export_type = source[source.index('    struct DecodeAuxiliaryTensorOwners {'):
                         source.index('    DecodeAuxiliaryTensorOwners decode_auxiliary_tensor_owners() const')]
    # Compile every real prime's position write/invalidations and publication.
    # Its unrelated planner/profile checks are covered by the existing Z1 tests.
    native = (ROOT / 'src/fused/rpu_qwen3_model.cpp').read_text()
    primes = []
    for name in ('prime_wall_oss_z1_inputs', 'prime_qwen3vl_pooler_z1_inputs',
                 'prime_qwen3vl_multiview_text_inputs'):
        prime = _function(native, f'void CausalDecoderModel::{name}(')
        begin = prime.index('    auto position_dst =')
        end = prime.index('    rpu_ddr_flush_force_sized(', begin)
        # The flush macro invocation ends at its semicolon, not a method brace.
        end = prime.index(';', end) + 1
        block = prime[begin:end]
        block = '\n'.join(line for line in block.splitlines()
                          if 'cos_dst' not in line and 'sin_dst' not in line
                          and 'cos_dst.copy_' not in line and 'sin_dst.copy_' not in line)
        primes.append(f' void {name}(const HostTensor& position_ids,int64_t execution_len){{\n{block}\n }}')
    clear_rows = 'std::fill(ka_cpu_pos_published_rows_.begin(), ka_cpu_pos_published_rows_.end(), 0);'
    setup = source[source.index('        const bool has_mrope = (mrope_section.size() > 0);'):
                   source.index('        if (has_mrope) {', source.index('        const bool has_mrope ='))]
    assert clear_rows in setup
    zero = source[source.index('                position_ids_keepalive_.zero_();'):
                  source.index('                position_ids_keepalive_.zero_();') +
                  len('                position_ids_keepalive_.zero_();')]
    # The Tensor wrapper changes only the device/storage boundary. No source
    # predicate, validation or invalidation logic is rewritten.
    prologue, declarations, methods, authority_methods, owner_export, export_type = [
        s.replace('at::Tensor', 'HostTensor') for s in (
            prologue, declarations, methods, authority_methods, owner_export, export_type)]
    copy_source = (ROOT / "src/core/rpu_tensor_ops.inc").read_text()
    same_device = _function(copy_source, '  if (same_device && same_dtype) {')
    start = copy_source.index('  // Case 3: Cross-device + Same dtype')
    cross_device = copy_source[start:copy_source.index('\n  return self;', start)]
    cpp = r'''
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include "core/rpu_profile.h"
#include <cassert>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <vector>
bool g_rpu_ddr_flush_enabled=true,g_rpu_ddr_flush_force_enabled=true;
struct Store {at::Tensor host;std::vector<uint8_t> ddr;bool device_dirty=false;};
std::vector<std::weak_ptr<Store>> stores;
std::vector<std::pair<void*,size_t>> flushed;
bool fail_next_flush=false;
namespace rhino_lkn {
void RpuDdrFlush(void* ptr,size_t size){
 if(fail_next_flush){fail_next_flush=false;TORCH_CHECK(false,"injected SDK boundary failure");}
 flushed.emplace_back(ptr,size);const auto address=reinterpret_cast<uintptr_t>(ptr);
 for(auto& weak:stores)if(auto store=weak.lock()){
  const auto base=reinterpret_cast<uintptr_t>(store->host.data_ptr());
  if(address>=base && address-base<=store->ddr.size() && size<=store->ddr.size()-(address-base)){
   auto* device=store->ddr.data()+(address-base);
   if(store->device_dirty){std::memcpy(ptr,device,size);store->device_dirty=false;}
   else std::memcpy(device,ptr,size);
   return;
  }
 }
 assert(false && "publication outside live storage");
}
}
struct HostTensor;
HostTensor& copy_transport(HostTensor&,const HostTensor&);
struct HostTensor {
 at::Tensor value;std::shared_ptr<Store> store;c10::Device where{at::kCPU};
 bool defined()const{return value.defined();}
 auto device()const{return where;}auto dim()const{return value.dim();}
 auto size(int64_t axis)const{return value.size(axis);}auto sizes()const{return value.sizes();}
 auto scalar_type()const{return value.scalar_type();}auto dtype()const{return value.dtype();}
 auto nbytes()const{return value.nbytes();}bool is_contiguous()const{return value.is_contiguous();}
 auto unsafeGetTensorImpl()const{return value.unsafeGetTensorImpl();}
 bool is_same(const HostTensor& other)const{return value.is_same(other.value);}
 void* data_ptr()const{return value.data_ptr();}
 template<class T>T* data_ptr()const{return value.data_ptr<T>();}
 HostTensor narrow(int64_t dim,int64_t start,int64_t size)const{
  return {value.narrow(dim,start,size),store,where};
 }
 HostTensor& copy_(const HostTensor& src,bool non_blocking=false){
  copy_transport(*this,src);
  // ATen owns copy_'s shared-version/inference semantics. Its CPU self-copy
  // changes the version without replacing the bytes written by the real body.
  value.copy_(value);
  return *this;
 }
 HostTensor& zero_(){value.zero_();return *this;}
 auto storage()const{return value.storage();}
};
void rpu_elementwise_copy_impl(const HostTensor&,const HostTensor&){assert(false);}
template<class A,class B>void cross_device_elementwise_copy(const HostTensor&,const HostTensor&){assert(false);}
HostTensor& copy_transport(HostTensor& self,const HostTensor& src){
 const bool non_blocking=false;
 const bool same_device=self.device()==src.device(),same_dtype=self.dtype()==src.dtype();
 __SAME_DEVICE__
 if(same_device && same_dtype)return self;
 const bool is_src_rpu=src.device().type()==at::kPrivateUse1;
 const bool is_dst_rpu=self.device().type()==at::kPrivateUse1;
 const bool is_src_cpu=src.device().is_cpu(),is_dst_cpu=self.device().is_cpu();
 __CROSS_DEVICE__
 return self;
}
HostTensor tensor(int rows,int fill,c10::DeviceType device=at::kCPU){
 auto store=std::make_shared<Store>();store->host=at::full({rows,3},fill,at::kInt);
 auto* data=static_cast<uint8_t*>(store->host.data_ptr());
 store->ddr.assign(data,data+store->host.nbytes());stores.push_back(store);
 return {store->host,store,c10::Device(device)};
}
void raw_fill(const HostTensor& value,int x){
 auto* data=value.data_ptr<int32_t>();for(int64_t i=0;i<value.value.numel();++i)data[i]=x;
}
int64_t version(const HostTensor& t){return t.value.is_inference()?-1:t.value._version();}
constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN=128;
constexpr int64_t QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN=256;
struct RpuKernelGraph{static bool has_active(){return false;}};
struct Model {
 bool qwen3vl_2b_w8_profile_=false;
 HostTensor position_ids_keepalive_=tensor(8192,777,at::kPrivateUse1);
 uint64_t generation=1;
 uint64_t installed_model_state_generation()const{return generation;}
 int num_layers()const{return 1;}
 HostTensor qwen3vl_4b_qkv_scale_bank_;
 bool qwen3vl_4b_decode_gateup_fused_=false;
 struct Weights{HostTensor gate_ws,up_ws;};std::vector<Weights> layer_weights_;
 int64_t ka_last_position_=-1,ka_last_seq_=-1,ka_last_pos_src_version_=-1;
 const void* ka_last_pos_src_=nullptr;
 __AUTHORITY_FIELDS__
 Model(int capacity=8192){position_ids_keepalive_=tensor(capacity,777,at::kPrivateUse1);ka_cpu_pos_published_rows_.assign(capacity,0);}
 __AUTHORITY_METHODS__
 __EXPORT_TYPE__
 __OWNER_EXPORT__
 __PRIMES__
 void reinstall_same_storage(){__CLEAR_ROWS__ __ZERO__ ++generation;}
 __DECLARATIONS__
 __METHODS__
 KeepaliveSourceCache ka_pos_sources_;
 void run(const std::optional<HostTensor>& position_ids,int64_t position,int64_t seq_len_in){
  __PROLOGUE__
  ka_last_position_=position;ka_last_seq_=seq_len_in;
 }
};
void expect_rows(const Model& m,int start,int rows,int x,bool published){
 assert(at::all(m.position_ids_keepalive_.value.narrow(0,start,rows)==x).item<bool>());
 auto* ddr=reinterpret_cast<const int32_t*>(m.position_ids_keepalive_.store->ddr.data());
 for(int i=start*3;i<(start+rows)*3;++i)assert(ddr[i]==(published?x:777));
}
template<class F>void rejects(Model& m,F f){
 auto before=m.position_ids_keepalive_.value.clone();auto count=flushed.size();
 const auto ver=version(m.position_ids_keepalive_);bool caught=false;
 try{f();}catch(const c10::Error&){caught=true;}
 assert(caught && at::equal(before,m.position_ids_keepalive_.value));
 assert(flushed.size()==count && version(m.position_ids_keepalive_)==ver);
}
extern "C" int64_t refresh_numpy_cpu_source(void* data,bool reset,bool inference){
 c10::InferenceMode mode(inference);
 static std::unique_ptr<Model> model;
 if(reset)model=std::make_unique<Model>();
 assert(model);g_rpu_ddr_flush_force_enabled=true;
 HostTensor source{at::from_blob(data,{2,3},at::kInt),{},c10::Device(at::kCPU)};
 const auto calls=flushed.size();model->run(source,4,2);
 return flushed.size()-calls;
}
int main(){
 for(int capacity:{8193,16384,32768}){
  Model m(capacity);auto source=tensor(1,17);g_rpu_ddr_flush_force_enabled=true;
  m.run(source,capacity-1,1);expect_rows(m,capacity-1,1,17,true);
  rejects(m,[&]{m.run(source,capacity,1);});
  rejects(m,[&]{m.run(source,-1,1);});
 }
 for(bool normal:{false,true})for(bool force:{false,true})
 for(bool inference:{false,true})for(bool w8:{false,true}){
  c10::InferenceMode mode(inference);
  g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;
  Model m;m.qwen3vl_2b_w8_profile_=w8;
  auto backing=tensor(6,0);auto cpu=backing.narrow(0,1,2);
  const auto src_version=version(cpu),dst_version=version(m.position_ids_keepalive_);
  auto* stable=m.position_ids_keepalive_.data_ptr();
  at::Tensor retained;
  for(int value:{5,9,5}){
   raw_fill(cpu,value);const auto calls=flushed.size();
   m.run(cpu,4,2);
   assert(version(cpu)==src_version && flushed.size()-calls==(force?1:0));
   expect_rows(m,4,2,value,force);
   assert(stable==m.position_ids_keepalive_.data_ptr());
   assert(at::all(m.position_ids_keepalive_.value.narrow(0,0,4)==777).item<bool>());
   assert(at::all(m.position_ids_keepalive_.value.narrow(0,6,58)==777).item<bool>());
   if(retained.defined())assert(at::all(retained==5).item<bool>());
   else retained=m.position_ids_keepalive_.value.narrow(0,4,2).clone();
   const auto repeat_calls=flushed.size();const auto repeat_version=version(m.position_ids_keepalive_);
   m.run(cpu,4,2);
   assert(flushed.size()-repeat_calls==0);
   assert(version(m.position_ids_keepalive_)==repeat_version+((force||inference)?0:1));
  }
  assert(version(m.position_ids_keepalive_)==(inference?-1:dst_version+(force?3:6)));

  // Reusing the same device Tensor/version after a CPU overwrite must refill.
  auto rpu=tensor(2,31,at::kPrivateUse1);const auto rpu_version=version(rpu);
  m.run(rpu,4,2);raw_fill(cpu,47);m.run(cpu,4,2);
  auto calls=flushed.size();m.run(rpu,4,2);
  assert(version(rpu)==rpu_version && flushed.size()-calls==(force?2:0));
  expect_rows(m,4,2,31,force);
  calls=flushed.size();m.run(rpu,4,2);
  assert(flushed.size()-calls==((inference&&force)?2:0));

  // A shorter CPU write invalidates the overlapping larger device stamp only.
  auto other=tensor(1,61,at::kPrivateUse1);m.run(other,20,1);
  auto partial=cpu.narrow(0,0,1);raw_fill(partial,73);m.run(partial,5,1);
  calls=flushed.size();m.run(rpu,4,2);
  assert(flushed.size()-calls==(force?2:0));expect_rows(m,4,2,31,force);
  if(w8&&!inference)assert(Model::keepalive_segment_hit(m.ka_pos_sources_,{20,1},other));

  // Source acquire is still before the memcpy for newly device-written device IDs.
  auto device_source=tensor(2,-7,at::kPrivateUse1);
  auto* device=reinterpret_cast<int32_t*>(device_source.store->ddr.data());
  for(int i=0;i<6;++i)device[i]=81;device_source.store->device_dirty=true;
  calls=flushed.size();m.run(device_source,10,2);
  assert(flushed.size()-calls==(force?2:0));
  expect_rows(m,10,2,force?81:-7,force);

  auto invalid=cpu;invalid.where=c10::Device(at::kCUDA);
  rejects(m,[&]{m.run(invalid,4,2);});
  invalid=cpu;invalid.value=cpu.value.to(at::kHalf);rejects(m,[&]{m.run(invalid,4,2);});
  invalid=cpu;invalid.value=cpu.value.narrow(1,0,2);rejects(m,[&]{m.run(invalid,4,2);});
  invalid=cpu;invalid.value=cpu.value.transpose(0,1).contiguous().transpose(0,1);
  rejects(m,[&]{m.run(invalid,4,2);});
  rejects(m,[&]{m.run(cpu,8191,2);});rejects(m,[&]{m.run(cpu,4,1);});
  rejects(m,[&]{m.run(std::nullopt,4,2);});

  // The destination remains valid after a temporary CPU source is released.
  std::weak_ptr<Store> weak;
  {auto temporary=tensor(1,91);weak=temporary.store;m.run(temporary,30,1);}
  expect_rows(m,30,1,91,force);
  if(w8&&!inference){assert(!weak.expired());m.run(other,30,1);}
  assert(weak.expired());
 }
 // Publication authority, independently of versions and normal-flush policy.
 for(bool inference:{false,true})for(bool w8:{false,true}){
  c10::InferenceMode mode(inference);g_rpu_ddr_flush_force_enabled=true;
  Model m;m.qwen3vl_2b_w8_profile_=w8;auto cpu=tensor(2,11);
  m.run(cpu,4,2);auto calls=flushed.size();m.run(cpu,4,2);
  assert(flushed.size()==calls);
  // A generation change must refill matching bytes; it does not require a
  // source/destination version bump or a changed table address.
  ++m.generation;calls=flushed.size();m.run(cpu,4,2);
  assert(flushed.size()-calls==1);calls=flushed.size();m.run(cpu,4,2);assert(flushed.size()==calls);
  // A synchronous copy failure after the CPU write cannot claim publication.
  raw_fill(cpu,19);fail_next_flush=true;bool failed=false;
  try{m.run(cpu,4,2);}catch(const c10::Error&){failed=true;}
  assert(failed && !m.ka_cpu_pos_published_rows_[4] && !m.ka_cpu_pos_published_rows_[5]);
  calls=flushed.size();m.run(cpu,4,2);assert(flushed.size()-calls==1);expect_rows(m,4,2,19,true);
  // force=False writes cannot seed a later force=True hit, including A/B/A.
  g_rpu_ddr_flush_force_enabled=false;raw_fill(cpu,23);m.run(cpu,4,2);
  raw_fill(cpu,11);m.run(cpu,4,2);g_rpu_ddr_flush_force_enabled=true;
  calls=flushed.size();m.run(cpu,4,2);assert(flushed.size()-calls==1);expect_rows(m,4,2,11,true);
  // A CPU from_blob alias into any part of the backing storage never certifies
  // a hit, even when the exact target bytes already match.
  HostTensor alias{at::from_blob(m.position_ids_keepalive_.data_ptr<int32_t>()+12,
                               {2,3},at::kInt),m.position_ids_keepalive_.store,c10::Device(at::kCPU)};
  const auto alias_version=version(alias);
  for(int value:{37,41,37}){
   raw_fill(alias,value);calls=flushed.size();m.run(alias,4,2);
   assert(version(alias)==alias_version && flushed.size()-calls==1);
   calls=flushed.size();m.run(alias,4,2);assert(flushed.size()-calls==1);
   expect_rows(m,4,2,value,true);
  }
  // Every real composite prime invalidates its written rows without claiming
  // CPU publication authority, so the same CPU value must copy again.
  auto prime_source=tensor(256,53,at::kPrivateUse1);raw_fill(cpu,53);
  for(int route=0;route<3;++route){
   m.run(cpu,4,2);calls=flushed.size();m.run(cpu,4,2);assert(flushed.size()==calls);
   if(route==0)m.prime_wall_oss_z1_inputs(prime_source.narrow(0,0,192),192);
   if(route==1)m.prime_qwen3vl_pooler_z1_inputs(prime_source.narrow(0,0,128),128);
   if(route==2)m.prime_qwen3vl_multiview_text_inputs(prime_source,256);
   calls=flushed.size();m.run(cpu,4,2);assert(flushed.size()-calls==1);
  }
  // Returning an owner permanently disables hits. The alias can publish B and
  // restore CPU A without a version bump: comparing only current dst would lie.
  auto owner=m.decode_auxiliary_tensor_owners().mrope_position_ids;
  raw_fill(cpu,61);m.run(cpu,4,2);
  auto owner_slice=owner.narrow(0,4,2);const auto owner_version=version(owner);
  raw_fill(owner_slice,67);rhino_lkn::RpuDdrFlush(owner_slice.data_ptr(),owner_slice.nbytes());
  raw_fill(owner_slice,61);assert(version(owner)==owner_version);
  calls=flushed.size();m.run(cpu,4,2);assert(flushed.size()-calls==1);expect_rows(m,4,2,61,true);
  auto* stable=m.position_ids_keepalive_.data_ptr();m.reinstall_same_storage();
  assert(stable==m.position_ids_keepalive_.data_ptr());
  for(int i=0;i<2;++i){calls=flushed.size();m.run(cpu,4,2);assert(flushed.size()-calls==1);}
 }
 // The 4B overfetch tail belongs to the same owner, despite being outside its
 // logical 8192-row view. A CPU alias there also cannot seed a cache hit.
 {
  c10::InferenceMode mode(false);Model m;auto backing=tensor(8200,83,at::kPrivateUse1);
  m.position_ids_keepalive_=backing.narrow(0,0,8192);
  HostTensor tail{at::from_blob(backing.data_ptr<int32_t>()+8192*3,{2,3},at::kInt),
                  backing.store,c10::Device(at::kCPU)};
  for(int i=0;i<2;++i){const auto calls=flushed.size();m.run(tail,8190,2);assert(flushed.size()-calls==1);}
  assert(at::all(backing.value.narrow(0,8192,8)==83).item<bool>());
  HostTensor overflow{at::from_blob(reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()-3),
                                  {1,3},at::kInt),{},c10::Device(at::kCPU)};
  assert(!m.cpu_position_source_is_disjoint(overflow));
 }
}
'''
    for key, value in (("__SAME_DEVICE__", same_device), ("__CROSS_DEVICE__", cross_device),
                       ("__DECLARATIONS__", declarations), ("__METHODS__", methods),
                       ("__AUTHORITY_FIELDS__", authority_fields), ("__AUTHORITY_METHODS__", authority_methods),
                       ("__EXPORT_TYPE__", export_type), ("__OWNER_EXPORT__", owner_export),
                       ("__PRIMES__", '\n'.join(primes)), ("__CLEAR_ROWS__", clear_rows), ("__ZERO__", zero),
                       ("__PROLOGUE__", prologue)):
        cpp = cpp.replace(key, value)
    _run_native(tmp_path, cpp, torch)
    # The same actual native prologue consumes a live Python Tensor/NumPy raw
    # alias. Changing its bytes does not increment its ATen version counter.
    torch_root = Path(torch.__file__).resolve().parent
    library = tmp_path / 'mrope_numpy.so'
    built = subprocess.run([
        shutil.which('c++'), '-std=c++17', '-O0', '-shared', '-fPIC',
        '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch.compiled_with_cxx11_abi())),
        '-I', str(ROOT / 'src'), '-I', str(torch_root / 'include'),
        '-I', str(torch_root / 'include/torch/csrc/api/include'),
        str(tmp_path / 'siglip_cold.cpp'), '-L', str(torch_root / 'lib'),
        '-Wl,-rpath,' + str(torch_root / 'lib'), '-ltorch_cpu', '-lc10',
        '-o', str(library),
    ], capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stdout + built.stderr
    module = ctypes.CDLL(str(library))
    refresh = module.refresh_numpy_cpu_source
    refresh.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool]
    refresh.restype = ctypes.c_int64
    for inference in (False, True):
        with torch.inference_mode(inference):
            backing = torch.zeros((6, 3), dtype=torch.int32)
            cpu = backing[1:3]
            alias = cpu.numpy()
            previous_version = -1 if inference else cpu._version
            for index, value in enumerate((5, 9, 5)):
                alias.fill(value)
                assert (-1 if inference else cpu._version) == previous_version
                assert refresh(cpu.data_ptr(), index == 0, inference) == 1
                assert refresh(cpu.data_ptr(), False, inference) == 0
