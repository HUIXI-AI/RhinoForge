"""Host guards for the mask refresh boundary; device outputs live in the ops test."""
from pathlib import Path
from dataclasses import replace

import pytest

from rpu_backend.adapters.gemma4.runtime import compute_side_input as _compute_side_cpu
from rpu_backend.tests.acceptance.test_gemma4_install_transaction import install_runtime

ROOT = Path(__file__).resolve().parents[1]


def _method(source, name):
    start = source.index(name)
    body = source.index("{", start)
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.mark.parametrize("global_fast", [False, True])
def test_gemma_owner_keeps_position_updates_even_with_global_fast(monkeypatch, global_fast):
    from rpu_backend.graph import GraphRuntimePolicy
    from rpu_backend.adapters.gemma4 import _gemma4_graph_runtime_policy

    inherited = replace(GraphRuntimePolicy(), fmb_fast_replay=global_fast,
                        fmb_deep_fast_replay=global_fast, fast_replay_skip_sync=True)
    monkeypatch.setattr(GraphRuntimePolicy, "from_environment", classmethod(lambda cls: inherited))
    effective = _gemma4_graph_runtime_policy()
    assert not effective.fmb_fast_replay and not effective.fmb_deep_fast_replay
    assert effective.fast_replay_skip_sync
    assert effective.execution_core_count == inherited.execution_core_count
    assert "owner:gemma4-replay-position-updates" in effective.provenance


def test_refresh_precedes_skip_and_planning_does_not_upload():
    src = (ROOT / "src/fused/rpu_gemma4_model.cpp").read_text()
    planning = _method(src, "ModelDynamicConfig planning_dynamic_config(")
    dynamic = _method(src, "ModelDynamicConfig dynamic_config(")
    prepare = _method(src, "void prepare_windowed_masks(")
    consume = _method(src, "uint32_t copy_windowed_mask_to_spm(")
    assert "prepare_windowed_masks" not in planning
    assert "preparing_forward_masks_" in dynamic and "prepare_windowed_masks(plan)" in dynamic
    assert "c10::make_scope_exit" in src and "preparing_forward_masks_ = false;" in src
    assert "sdpa_prepare_mask" in prepare and "/*ordinal=*/chunk.idx" in prepare
    assert "sdpa_prepare_mask" not in consume
    assert "sdpa_dma_mask_to_spm(entry.prepared" in consume
    assert "entry.chunk.kv_seq_len == chunk.kv_seq_len" in consume
    assert "entry.start == start && entry.length == length" in consume
    fmb = (ROOT / "src/core/fused_model_base.cpp").read_text()
    run = fmb[fmb.index("// run_all_layers —"):]
    assert run.index("auto dyn_cfg = dynamic_config(plan)") < run.index("const bool fast_skip =")


@pytest.mark.parametrize("inference", [False, True])
def test_ple_cpu_producer_reads_current_ids_and_keeps_retained_outputs(inference):
    import ctypes
    import torch
    from rpu_backend.adapters.gemma4.runtime import compute_side_input

    with torch.inference_mode(inference):
        ids = torch.tensor([[1, 2, 3]])
        weights = dict(
            embed_weight=torch.arange(128, dtype=torch.float32).reshape(16, 8) / 128,
            embed_per_layer_weight=torch.arange(512, dtype=torch.float32).reshape(16, 32) / 256,
            model_projection_weight=torch.arange(256, dtype=torch.float32).reshape(32, 8) / 128,
            projection_norm_weight=torch.ones(16), num_layers=2, hidden_size=8,
            ple_dim=16, eps=1e-6,
        )
        version = None if inference else ids._version
        alias = (ctypes.c_int64 * ids.numel()).from_address(ids.data_ptr())
        outputs = []
        for token in (1, 7, 1):
            alias[0] = token
            output = compute_side_input(ids, **weights)
            assert output.device.type == "cpu"
            assert output.dtype == torch.float16 and output.is_contiguous()
            assert output.shape == (2, 8, 3, 2) and torch.isfinite(output).all()
            assert inference or ids._version == version
            outputs.append(output)
        assert not torch.equal(outputs[0], outputs[1])
        assert torch.equal(outputs[0], outputs[2])
        assert len({out.data_ptr() for out in outputs}) == 3


def test_real_public_gemma4_forward_keeps_both_cpu_producers_current(
        install_runtime, monkeypatch):
    """Actual adapter/CPU producers and output protocol; native decoder is a double."""
    import ctypes
    import torch
    from rpu_backend.tests.acceptance import test_gemma4_install_transaction as installed
    from rpu_backend.adapters.gemma4 import runtime

    h, layers, ple, vocab = 2560, 42, 256, 8

    class Model(installed._Model):
        def __init__(self):
            super().__init__()
            self.embed = torch.arange(vocab * h, dtype=torch.float32).reshape(vocab, h) / 1000000
            self.ple = torch.arange(vocab * layers * ple, dtype=torch.float32).reshape(vocab, -1) / 100000
            self.projection = torch.linspace(0.001, 0.002, layers * ple)[:, None].expand(-1, h)

        def state_dict(self):
            values = super().state_dict()
            values.update({runtime.PFX + "embed_tokens.weight": self.embed,
                           runtime.PFX + "embed_tokens_per_layer.weight": self.ple,
                           runtime.PFX + "per_layer_model_projection.weight": self.projection,
                           runtime.PFX + "per_layer_projection_norm.weight": torch.ones(ple)})
            return values

    source = Model()
    monkeypatch.setattr(runtime, "compute_side_input", _compute_side_cpu)
    observed = []

    def native(handle, hidden, k, v, mask, position, causal, chunk, side, descriptor):
        assert handle == 1 and causal and chunk == 0 and descriptor and mask is None
        assert hidden.device.type == side.device.type == "cpu"
        assert hidden.dtype == side.dtype == torch.float16
        assert hidden.is_contiguous() and side.is_contiguous()
        assert tuple(hidden.shape) == (1, side.shape[2], h)
        assert tuple(side.shape) == (layers, 8, hidden.shape[1], ple // 8)
        observed.append((hidden, side, position))
        return hidden.clone()

    monkeypatch.setattr(torch.ops.rpu, "gemma4_forward", native)
    model = installed._adapter(source).to_rpu()
    cache = model.build_rpu_cache()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    before = ids._version
    alias = (ctypes.c_int64 * ids.numel()).from_address(ids.data_ptr())
    outputs = []
    try:
        for token in (1, 7, 1):
            alias[0] = token
            cache.position = cache._seen_tokens = 0
            result = model.forward(input_ids=ids, past_key_values=cache)
            assert ids._version == before and cache.position == 6
            expected = (torch.nn.functional.embedding(ids, source.embed).float() * h**0.5).half()
            assert torch.equal(observed[-1][0], expected)
            assert torch.equal(result.logits, runtime.softcap_lm_logits(expected.float(), source.embed, 30.0))
            assert result.logits.dtype == torch.float32 and result.logits.device.type == "cpu"
            outputs.append(result.logits)
        assert torch.equal(outputs[0], outputs[2]) and not torch.equal(outputs[0], outputs[1])
        assert torch.equal(observed[0][1], observed[2][1]) and not torch.equal(observed[0][1], observed[1][1])
        assert len({item[0].data_ptr() for item in observed}) == 3
        assert len({item[1].data_ptr() for item in observed}) == 3
        assert len({output.data_ptr() for output in outputs}) == 3
        assert len(model._rpu_gemma4_graph_cache.signatures) == 1
        model.forward(input_ids=torch.tensor([[2]]), past_key_values=cache)
        assert observed[-1][2] == 6 and cache.position == 7
        assert len(model._rpu_gemma4_graph_cache.signatures) == 2
    finally:
        installed._finalize(model)


def test_actual_ple_ingress_preserves_publication_owner_and_dirty_aliases(tmp_path, monkeypatch):
    """Real native ingress and ATen owners; SDK and deferred consumer doubles."""
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    from test_siglip_attention_dry_live_contract import _function, _run_native

    source = (ROOT / "src/fused/rpu_gemma4_model.cpp").read_text()
    kv_span_guard = _function(source, "void validate_gemma4_kv_span(")
    forward = _function(source, "    at::Tensor forward(")
    preflight = forward[forward.index("        TORCH_CHECK(num_layers() > 0"):
                        forward.index("        // dynamic_config also")]
    ingress = _function(forward[forward.index("        // T6 PLE: per-layer"):],
                        "        if (has_ple_) {")
    assert forward.index(ingress) < forward.index("return run_all_layers(")
    planner_source = (ROOT / "src/core/fmb_three_stage_chunk_plan.cpp").read_text()
    planning_guard = _function(planner_source, "void validate_fmb_planning_shape(")
    fmb = (ROOT / "src/core/fused_model_base.cpp").read_text()
    hidden_lease = _function(fmb[fmb.index("// Keep hidden_states alive until deferred"):],
                             "    if (RpuKernelGraph::has_active()) {")
    copy_source = (ROOT / "src/core/rpu_tensor_ops.inc").read_text()
    start = copy_source.index("  // Case 3: Cross-device + Same dtype")
    cross_device = copy_source[start:copy_source.index("\n  return self;", start)]
    cpp = r'''
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include "core/rpu_profile.h"
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
bool g_rpu_ddr_flush_enabled=false,g_rpu_ddr_flush_force_enabled=true;
struct Store {at::Tensor host;std::vector<uint8_t> ddr;};
std::vector<std::weak_ptr<Store>> stores;
struct Call {void* ptr;size_t size;bool sized;bool batch=false;};std::vector<Call> calls;
bool fail_batch=false;int allocations=0,fail_allocation=-1;
namespace rhino_lkn {
void publish(void* ptr,size_t bytes){
 const auto address=reinterpret_cast<uintptr_t>(ptr);
 for(auto& weak:stores)if(auto store=weak.lock()){
  auto base=reinterpret_cast<uintptr_t>(store->host.data_ptr());
  if(address>=base && address-base<=store->ddr.size() && bytes<=store->ddr.size()-(address-base)){
   std::memcpy(store->ddr.data()+(address-base),ptr,bytes);return;
  }
 }
 assert(false && "flush escaped its live owner");
}
void RpuDdrFlush(void* ptr,size_t bytes){calls.push_back({ptr,bytes,true});publish(ptr,bytes);}
void RpuDdrFlush(void* ptr){
 const auto address=reinterpret_cast<uintptr_t>(ptr);
 for(auto& weak:stores)if(auto store=weak.lock()){
  auto base=reinterpret_cast<uintptr_t>(store->host.data_ptr());
  if(address>=base && address<base+store->ddr.size()){
   RpuDdrFlush(ptr,store->ddr.size()-(address-base));calls.back().sized=false;return;
  }
 }
 assert(false);
}
uint64_t RpuGetDevAddr(void* ptr){return reinterpret_cast<uint64_t>(ptr);}
void RpuDdrFlushBatch(void** ptrs,size_t* sizes,size_t count){
 assert(count==2 && sizes[0]>0 && sizes[1]>0);
 if(fail_batch)throw std::runtime_error("batch failure");
 calls.push_back({ptrs[0],sizes[0]+sizes[1],true,true});
 for(size_t i=0;i<count;++i)publish(ptrs[i],sizes[i]);
}
}
using rhino_lkn::RpuGetDevAddr;
struct HostTensor;
HostTensor& copy_transport(HostTensor&,const HostTensor&);
struct HostTensor {
 at::Tensor value;std::shared_ptr<Store> store;c10::Device where{at::kCPU};
 bool defined()const{return value.defined();}auto device()const{return where;}
 auto dim()const{return value.dim();}auto size(int64_t n)const{return value.size(n);}
 bool requires_grad()const{return value.requires_grad();}
 auto scalar_type()const{return value.scalar_type();}auto numel()const{return value.numel();}
 auto sizes()const{return value.sizes();}auto nbytes()const{return value.nbytes();}
 bool is_contiguous()const{return value.is_contiguous();}auto storage()const{return value.storage();}
 void* data_ptr()const{return value.data_ptr();}
 template<class T>T* data_ptr()const{return value.data_ptr<T>();}
 HostTensor contiguous()const{
  if(is_contiguous())return *this;
  // CPU strided staging delegates to real ATen. The unchanged noncontiguous
  // device copy path is outside this boundary test.
  assert(where.is_cpu());return {value.contiguous(),nullptr,where};
 }
 HostTensor to(c10::Device target)const{
  assert(where.is_cpu() && target.type()==at::kPrivateUse1 && is_contiguous());
  auto owner=std::make_shared<Store>();owner->host=at::empty(value.sizes(),value.options());
  owner->ddr.assign(owner->host.nbytes(),0xA5);stores.push_back(owner);
  HostTensor result{owner->host,owner,target};
  // Actual _to_copy allocates a fresh contiguous destination, then directly
  // calls rpu_copy_impl; its same-dtype CPU->device body is compiled below.
  copy_transport(result,*this);return result;
 }
 HostTensor& copy_(const HostTensor&){assert(false);return *this;}
};
HostTensor fake_empty(at::IntArrayRef shape,const at::TensorOptions& options){
 if(allocations++==fail_allocation)throw std::runtime_error("allocation failure");
 assert(options.device().type()==at::kPrivateUse1 && options.dtype().toScalarType()==at::kHalf);
 auto owner=std::make_shared<Store>();owner->host=at::empty(shape,options.device(at::kCPU));
 owner->ddr.assign(owner->host.nbytes(),0xA5);stores.push_back(owner);
 return {owner->host,owner,c10::Device(at::kPrivateUse1)};
}
struct RpuKernelGraph {
 static bool active_;static std::vector<HostTensor> refs;
 static bool has_active(){return active_;}
 static RpuKernelGraph& active(){static RpuKernelGraph graph;return graph;}
 void keep_alive(const HostTensor& tensor){refs.push_back(tensor);}
};
bool RpuKernelGraph::active_=false;std::vector<HostTensor> RpuKernelGraph::refs;
namespace detail {__PLANNING_GUARD__}
__KV_SPAN_GUARD__
template<class A,class B>void cross_device_elementwise_copy(const HostTensor&,const HostTensor&){assert(false);}
HostTensor& copy_transport(HostTensor& self,const HostTensor& src){
 const bool is_src_rpu=src.device().type()==at::kPrivateUse1;
 const bool is_dst_rpu=self.device().type()==at::kPrivateUse1;
 const bool is_src_cpu=src.device().is_cpu(),is_dst_cpu=self.device().is_cpu();
 __CROSS_DEVICE__
 return self;
}
struct Model {
 bool has_ple_=true;int64_t seq_len_=3,ple_dim_=16;HostTensor side_input_;
 HostTensor hidden_input_;int64_t chunk_override=99;
 at::Tensor cos_sliding_=at::zeros({64,1},at::kHalf),cos_global_=at::zeros({64,1},at::kHalf);
 uint64_t side_live_base_=0;int64_t num_layers()const{return 2;}
 int64_t hidden_size()const{return 8;}
 void set_chunk_size_override(int64_t value){chunk_override=value;}
 void prepare(const std::optional<HostTensor>& side_input,
              std::optional<HostTensor> hidden=std::nullopt){
  HostTensor hidden_states=hidden.value_or(HostTensor{
    at::zeros({1,seq_len_,8},at::kHalf),nullptr,c10::Device(at::kPrivateUse1)});
  int64_t position=0,chunk_size=0;std::vector<int64_t> planned_stage_descriptor{1};
  __PREFLIGHT__
  __INGRESS__
  // Execute the actual FMB hidden lease before the deferred consumer. Its
  // live source and retained Tensor are the newly prepared device owner.
  __HIDDEN_LEASE__
  hidden_input_=live_hidden;
 }
};
HostTensor tensor(int64_t seq,c10::DeviceType device,bool offset,bool strided=false){
 const int64_t count=2*seq*16;
 auto owner=std::make_shared<Store>();
 owner->host=at::zeros({count*(strided?2:1)+32},at::kHalf);
 auto view=owner->host.narrow(0,offset?16:0,count*(strided?2:1));
 if(strided)view=view.slice(0,0,view.numel(),2);
 view=view.view({2,8,seq,2});
 owner->ddr.assign(owner->host.nbytes(),0xA5);stores.push_back(owner);
 return {view,owner,c10::Device(device)};
}
HostTensor hidden_tensor(int64_t seq,bool offset){
 auto side=tensor(seq,at::kCPU,offset);
 side.value=side.value.flatten().narrow(0,0,seq*8).view({1,seq,8});return side;
}
int64_t version(const HostTensor& t){return t.value.is_inference()?-1:t.value._version();}
void dirty(const HostTensor& t,int value){
 auto* p=t.data_ptr<c10::Half>();auto step=t.value.stride(-1);
 for(int64_t i=0;i<t.numel();++i)p[i*step]=static_cast<c10::Half>(value);
}
void consume(const Model& m,const at::Tensor& expected,bool published){
 assert(m.side_live_base_==reinterpret_cast<uint64_t>(m.side_input_.data_ptr()));
 const auto& store=*m.side_input_.store;
 auto offset=static_cast<uint8_t*>(m.side_input_.data_ptr())-static_cast<uint8_t*>(store.host.data_ptr());
 const auto* bytes=static_cast<const uint8_t*>(expected.data_ptr());
 // Same core-major flattened extent as PLE scatter: L*8*S*(PD/8).
 for(size_t i=0;i<m.side_input_.nbytes();++i)
  assert(store.ddr[offset+i]==(published?bytes[i]:0xA5));
}
void consume_hidden(const Model& m,const at::Tensor& expected,bool published){
 const auto& live=m.hidden_input_;const auto& store=*live.store;
 auto* bytes=static_cast<const uint8_t*>(expected.data_ptr());
 assert(live.device().type()==at::kPrivateUse1 && at::equal(live.value,expected));
 for(size_t i=0;i<live.nbytes();++i)assert(store.ddr[i]==(published?bytes[i]:0xA5));
}
template<class F>void rejects(F body){bool bad=false;try{body();}catch(const std::exception&){bad=true;}assert(bad);}
int main(){
 for(bool normal:{false,true})for(bool force:{false,true})for(bool inference:{false,true})
 for(bool rpu:{false,true})for(bool offset:{false,true})for(int64_t seq:{1,3}){
  c10::InferenceMode mode(inference);stores.clear();calls.clear();
  g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;
  Model model;model.seq_len_=seq;
  auto input=tensor(seq,rpu?at::kPrivateUse1:at::kCPU,offset,!rpu&&offset);
  auto before=version(input);std::vector<HostTensor> retained;std::vector<at::Tensor> expected;
  for(int fill:{2,7,2}){
   if(input.store)std::fill(input.store->ddr.begin(),input.store->ddr.end(),0xA5);
   dirty(input,fill);assert(version(input)==before);calls.clear();
   auto gold=input.value.contiguous().clone();model.prepare(input);
   assert(model.side_input_.is_contiguous() && model.side_input_.device().type()==at::kPrivateUse1);
   assert(calls.size()==size_t(force));
   if(force){assert(calls[0].sized==!rpu);assert(calls[0].ptr==model.side_input_.data_ptr());
             if(!rpu)assert(calls[0].size==gold.nbytes());}
   assert(version(input)==before);consume(model,gold,force);
   if(!rpu){retained.push_back(model.side_input_);expected.push_back(gold);}
  }
  if(!rpu){
   assert(retained[0].data_ptr()!=retained[1].data_ptr());
   for(size_t i=0;i<retained.size();++i)assert(at::equal(retained[i].value,expected[i]));
  }
  auto* live=model.side_input_.data_ptr();input=HostTensor{};
  assert(model.side_input_.data_ptr()==live && model.side_input_.store);
 }
 // The new pair entry uploads both complete current CPU producers once. Raw
 // aliases do not bump versions; every A/B/A call gets two fresh DDR owners.
 for(bool normal:{false,true})for(bool force:{false,true})for(bool inference:{false,true})
 for(bool offset:{false,true})for(int64_t seq:{1,3}){
  c10::InferenceMode mode(inference);stores.clear();calls.clear();allocations=0;
  RpuKernelGraph::active_=true;RpuKernelGraph::refs.clear();
  g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;
  Model model;auto hidden=hidden_tensor(seq,offset);
  auto input=tensor(seq,at::kCPU,offset,offset);
  auto hidden_version=version(hidden),side_version=version(input);
  std::vector<at::Tensor> hidden_gold,side_gold;
  for(int fill:{2,7,2}){
   dirty(hidden,fill);dirty(input,fill+1);calls.clear();
   auto h=hidden.value.clone(),s=input.value.contiguous().clone();
   model.prepare(input,hidden);
   assert(model.seq_len_==seq && model.chunk_override==0);
   assert(version(hidden)==hidden_version && version(input)==side_version);
   assert(calls.size()==size_t(force));
   if(force)assert(calls[0].batch && calls[0].size==h.nbytes()+s.nbytes());
   consume(model,s,force);consume_hidden(model,h,force);
   assert(RpuKernelGraph::refs.size()==2*(hidden_gold.size()+1));
   assert(RpuKernelGraph::refs.back().data_ptr()==model.hidden_input_.data_ptr());
   hidden_gold.push_back(h);side_gold.push_back(s);
  }
  assert(allocations==6);
  assert(RpuKernelGraph::refs[0].data_ptr()!=RpuKernelGraph::refs[2].data_ptr());
  assert(RpuKernelGraph::refs[1].data_ptr()!=RpuKernelGraph::refs[3].data_ptr());
  hidden=HostTensor{};input=HostTensor{};model.side_input_=HostTensor{};model.hidden_input_=HostTensor{};
  // Lease the owners until the deferred graph consumer finishes, even after
  // Python sources and the model's latest binding have been released.
  for(size_t i=0;i<hidden_gold.size();++i){
   assert(at::equal(RpuKernelGraph::refs[2*i].value,side_gold[i]));
   assert(at::equal(RpuKernelGraph::refs[2*i+1].value,hidden_gold[i]));
   assert(RpuKernelGraph::refs[2*i].store && RpuKernelGraph::refs[2*i+1].store);
  }
  RpuKernelGraph::refs.clear();RpuKernelGraph::active_=false;
  for(auto& weak:stores)assert(weak.expired());
 }
 Model model;auto input=tensor(3,at::kCPU,true);
 calls.clear();auto unchanged=model.side_live_base_;
 rejects([&]{model.prepare(std::nullopt);});
 rejects([&]{model.prepare(HostTensor{});});
 auto wrong=input;wrong.value=wrong.value.to(at::kFloat);rejects([&]{model.prepare(wrong);});
 wrong=input;wrong.where=c10::Device(at::kCUDA);rejects([&]{model.prepare(wrong);});
 wrong=input;wrong.value=wrong.value.flatten().narrow(0,0,5);rejects([&]{model.prepare(wrong);});
 assert(calls.empty() && model.side_live_base_==unchanged && !model.side_input_.defined());
 // The public native contract is flattened numel, not a new rank/shape gate.
 input.value=input.value.flatten();model.prepare(input);assert(model.side_input_.numel()==96);
 auto hidden=hidden_tensor(3,true);model.prepare(input,hidden);
 auto old_side=model.side_input_,old_hidden=model.hidden_input_;
 const auto old_base=model.side_live_base_;const auto old_seq=model.seq_len_;
 const auto old_chunk=model.chunk_override;const auto old_allocations=allocations;
 calls.clear();
 auto unchanged_bindings=[&]{
  assert(model.side_input_.data_ptr()==old_side.data_ptr() && model.side_live_base_==old_base);
  assert(model.hidden_input_.data_ptr()==old_hidden.data_ptr());
  assert(model.seq_len_==old_seq && model.chunk_override==old_chunk);
  assert(calls.empty() && allocations==old_allocations);
 };
 for(int kind=0;kind<7;++kind){
  auto bad=hidden;
  if(kind==0)bad.value=bad.value.to(at::kFloat);
  if(kind==1)bad.value=bad.value.squeeze(0);
  if(kind==2)bad.value=bad.value.expand({2,3,8});
  if(kind==3)bad.value=bad.value.narrow(2,0,4);
  if(kind==4)bad.value=bad.value.slice(1,0,3,2);
  if(kind==5)bad.value=at::empty({1,0,8},at::kHalf);
  if(kind==6)bad.value=bad.value.clone().set_requires_grad(true);
  rejects([&]{model.prepare(input,bad);});unchanged_bindings();
 }
 auto bad_side=input;bad_side.where=c10::Device(at::kPrivateUse1);
 rejects([&]{model.prepare(bad_side,hidden);});unchanged_bindings();
 bad_side=input;bad_side.value=bad_side.value.clone().set_requires_grad(true);
 rejects([&]{model.prepare(bad_side,hidden);});unchanged_bindings();
 rejects([&]{model.prepare(std::nullopt,hidden);});unchanged_bindings();
 model.has_ple_=false;rejects([&]{model.prepare(input,hidden);});unchanged_bindings();
 // The unchanged device-hidden no-PLE branch remains admitted.
 auto rpu_hidden=hidden;rpu_hidden.where=c10::Device(at::kPrivateUse1);
 model.prepare(std::nullopt,rpu_hidden);assert(calls.empty());model.has_ple_=true;
 // Failed paired allocation/publication must not replace the live PLE owner.
 g_rpu_ddr_flush_force_enabled=true;fail_allocation=allocations+1;
 rejects([&]{model.prepare(input,hidden);});fail_allocation=-1;
 assert(model.side_input_.data_ptr()==old_side.data_ptr() && model.side_live_base_==old_base && calls.empty());
 fail_batch=true;rejects([&]{model.prepare(input,hidden);});fail_batch=false;
 assert(model.side_input_.data_ptr()==old_side.data_ptr() && model.side_live_base_==old_base && calls.empty());
}
'''
    cpp = (cpp.replace("__PREFLIGHT__", preflight.replace("at::Tensor", "HostTensor"))
           .replace("__INGRESS__", ingress.replace("at::empty(", "fake_empty("))
           .replace("__HIDDEN_LEASE__", hidden_lease.replace("hidden_states", "live_hidden"))
           .replace("__PLANNING_GUARD__", planning_guard)
           .replace("__KV_SPAN_GUARD__", kv_span_guard)
           .replace("__CROSS_DEVICE__", cross_device))
    _run_native(tmp_path, cpp, torch)
