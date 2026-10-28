"""Pi0.5 RoPE positions: the prefix and the suffix must both be told where they are.

`position` in a fused pi05 subsystem is one number doing two jobs — the RoPE
table index and the KV cache write row. They are equal only while the LOGICAL
position equals the PHYSICAL row, which any padded or holed prefix breaks. Three
sites had it; all three are fixed (2026-08-12):

  1. VLM prefill  — adapters/pi05/gemma.py sends a position_ids-gathered
                    [seq, head_dim/2] RoPE table (the prefix's positions are not
                    contiguous when a modality is masked out, so it needs a table).
  2. action expert — adapters/pi05/adarms.py + runtime.py send a scalar logical
                    start (suffix_pad_masks is all ones, so its positions ARE
                    contiguous and a start offset is enough).
  3. per-chunk masks — rpu_sdpa.cpp keys its stable DDR slot on shape AND an
                    ordinal, so two equal-length chunks of one prefill stop
                    sharing a buffer.

This file gates (1) and (2), the two whose plumbing lives in Python: they must
be sent on EVERY forward.  The suffix start is a replay-mutable kernel register,
so it must not split otherwise identical denoise Graphs. (3) is C++-only and is
covered by the device legs.

Host-only: reads the adapters as source, loads no extension.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_PI05 = _ROOT / "python" / "rpu_backend" / "adapters" / "pi05"


def _fn(module: str, name: str):
    src = (_PI05 / module).read_text()
    tree = ast.parse(src, filename=module)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node, src
    raise AssertionError(f"{name} not found in {module}")


def _calls(fn_node, dotted: str):
    """Every call to `torch.ops.rpu.<dotted>` inside fn_node."""
    out = []
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == dotted:
            out.append(n)
    return out


def _guarding_ifs(fn_node, call):
    """The `if` statements that `call` sits inside, within fn_node."""
    found = []

    def walk(node, stack):
        for child in ast.iter_child_nodes(node):
            if child is call:
                found.extend(stack)
            walk(child, stack + [node] if isinstance(node, ast.If) else stack)

    walk(fn_node, [])
    return found


@pytest.mark.parametrize("module, fn, op", [
    ("gemma.py", "_run_gemma_prefill", "gemma_set_prefill_rope"),
    ("adarms.py", "rpu_adarms_model_forward", "adarms_set_rope_position"),
])
def test_positions_are_sent_unconditionally(module, fn, op):
    """Deleting the call, or hiding it behind `if position_ids is not None`,
    must turn this red — both would leave a graph BUILD keyed on one forward's
    positions being replayed against another's."""
    node, _ = _fn(module, fn)
    calls = _calls(node, op)
    assert len(calls) == 1, f"{fn} must call {op} exactly once, found {len(calls)}"
    assert not _guarding_ifs(node, calls[0]), (
        f"{op} is inside an `if` — it must run on every forward")


def test_the_expert_start_is_replay_mutable_not_a_graph_signature_axis():
    """Changing valid text length must update registers without rebuilding."""
    src = (_PI05 / "runtime.py").read_text()
    tree = ast.parse(src)
    # Both denoise paths set the live value, but do not key the Graph on it.
    for fn in ("_run_denoise_fused", "_run_denoise_loop"):
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == fn)
        text = ast.unparse(node)
        assert "_set_denoise_rope_position(" in text, f"{fn} never sets it"
        sig = text[text.index("GraphSignature("):]
        assert "rope_pos" not in sig[:sig.index("dtypes=")], \
            f"{fn}: replay-mutable rope_pos must not split the Graph"


def test_native_denoise_plan_excludes_mutable_rope_position():
    header = (_ROOT / "src/fused/rpu_pi05_denoise_step_model.h").read_text()
    planner = header[header.index("planner_cache_identity() const") :]
    planner = planner[:planner.index("void set_configured_chunk_size")]
    assert "rope_position_" not in planner

    source = (_ROOT / "src/fused/rpu_pi05_denoise_step_model.cpp").read_text()
    manifest = source[source.index("physical_manifest_for_candidate") :]
    manifest = manifest[:manifest.index("return manifest;")]
    assert "const int64_t rope_base = position;" in manifest

    static_config = source[source.index("Pi05DenoiseStepModel::static_config()") :]
    static_config = static_config[:static_config.index("return cfg;")]
    assert "cfg.replay_reemit_layer_body = !use_pi05_rope_register_patch();" in static_config
    assert "clear_replay_kernel_u32_patches();" in static_config
    assert "KernelId::PARTIAL_MROPE, 8" in static_config
    assert "KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M50D256P64" in static_config
    assert "12, position, expected" in static_config
    assert "num_layers() * num_steps_" in static_config
    assert static_config.count("position, expected") == 2


def test_logical_and_physical_only_agree_without_pad_rows():
    """The premise, in three lines: this is why one number cannot do both jobs."""
    pad = torch.tensor([[1] * 256 + [0] * 512 + [1] * 32])   # one camera of three
    physical = pad.shape[1]                                   # KV write row
    logical = int(pad.sum())                                  # RoPE position
    assert physical == 800 and logical == 288
    assert physical != logical, "a holed prefix must separate the two"
    # ...and with every row real they coincide, which is why production never saw it.
    full = torch.ones(1, 800, dtype=torch.long)
    assert full.shape[1] == int(full.sum())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


@pytest.mark.parametrize("route", [
    "fp16", "w8a16", "w8a16_action_nvfp4",
    "w8a16_prefill_w8a8_action_nvfp4", "cached",
])
@pytest.mark.parametrize("explicit_positions", [False, True])
def test_prefill_rope_preparation_preserves_cpu_and_rpu_routes(
    monkeypatch, route, explicit_positions,
):
    """Current CPU bits reach the real setter call, with the original cache route."""
    from types import SimpleNamespace
    import runpy
    import tomllib
    from rpu_backend.adapters.pi05._rope_cache import PrefillRopeUploadCache

    if route != "cached":
        engine = runpy.run_path(str(_ROOT / "examples/runners/pi05/optimized.py"))
        for cameras in (2, 3):
            config = tomllib.loads((_ROOT / "examples/configs/pi05" /
                                  f"{cameras}cam/{route}.toml").read_text())
            for text_tokens in (32, 64, 128, 160):
                config["input"]["text_tokens"] = text_tokens
                args = engine["_example_arguments"](config)
                env = engine["profile_environment"](
                    args.precision, prefill_w8a8=args.prefill_w8a8,
                    vision_owner_ln=args.vision_owner_ln, cameras=args.cameras,
                    text_tokens=args.text_tokens)
                assert env["RPU_PI05_PREFILL_KV1_PAIR_DIRECT"] == "1"
                assert env["RPU_PI05_PREFILL_ROPE_CACHE"] == "0"
                assert env["RPU_PI05_PREFILL_GATEUP_W8A8"] == str(int(
                    route == "w8a16_prefill_w8a8_action_nvfp4"))
                assert args.public_precision == route
                assert args.precision == ("w4a16" if "nvfp4" in route else route)
                assert env["RPU_PI05_SIGLIP_W8A16"] == str(int(route != "fp16"))

    node, _ = _fn("gemma.py", "_run_gemma_prefill")
    first = next(i for i, stmt in enumerate(node.body)
                 if isinstance(stmt, ast.If)
                 and ast.unparse(stmt.test) == "position_ids is None")
    last = next(i for i, stmt in enumerate(node.body)
                if isinstance(stmt, ast.Expr)
                and _calls(stmt, "gemma_set_prefill_rope"))
    block = compile(ast.Module(body=node.body[first:last + 1], type_ignores=[]),
                    "actual_pi05_rope_preparation", "exec")
    uploads, calls = [], []
    original_to = torch.Tensor.to

    class RpuStorage(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="rpu")

    def transfer(value, *args, **kwargs):
        if args and args[0] == "rpu":
            uploads.append(value.clone())
            return value.clone().as_subclass(RpuStorage)
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", transfer)
    # Actual two-camera T32 prefix axes and the native half-head RoPE width.
    seq_len, width, horizon = 544, 128, 2048
    phase = torch.arange(horizon * width, dtype=torch.float32).reshape(horizon, width) / 257
    cos, sin = phase.cos().half(), phase.sin().half()
    initial_tables = (cos.clone(), sin.clone())
    slots = [torch.full_like(cos, 777), torch.full_like(sin, 777)]
    addresses = [t.data_ptr() for t in slots]
    snapshots = []

    def bits(value):
        return value.contiguous().view(torch.uint8)

    def setter(handle, cos_rows, sin_rows):
        assert handle == 51
        expected_device = "rpu" if route == "cached" else "cpu"
        assert all(t.device.type == expected_device for t in (cos_rows, sin_rows))
        for slot, src in zip(slots, (cos_rows, sin_rows)):
            assert src.shape == (seq_len, width) and src.dtype == torch.float16
            assert src.is_contiguous() and torch.isfinite(src).all()
            slot[:seq_len].copy_(src.as_subclass(torch.Tensor))
        calls.append((cos_rows, sin_rows))

    monkeypatch.setattr(torch.ops.rpu, "gemma_set_prefill_rope", setter, raising=False)
    base_positions = torch.arange(seq_len, dtype=torch.int64) // 2
    positions = base_positions.unsqueeze(0).clone()
    versions = tuple(t._version for t in (positions, cos, sin))
    namespace = dict(torch=torch, position_ids=positions if explicit_positions else None,
        seq_len=seq_len, past_key_values=SimpleNamespace(position=0),
        _cos_cpu=cos, _sin_cpu=sin, gemma_w8a16=route not in ("fp16", "cached"),
        _prefill_rope_cache=PrefillRopeUploadCache(cos, sin) if route == "cached" else None,
        self=SimpleNamespace(_rpu_vlm_decoder_handle=51))
    for ordinal, offset in enumerate((0, 3, 0)):
        # The optional legacy cache retains its unchanged immutable-table contract.
        # Direct CPU ingress must observe raw table and position edits with no bump.
        positions.numpy()[:] = (base_positions + offset).unsqueeze(0).numpy()
        if route != "cached":
            for table, original in zip((cos, sin), initial_tables):
                table.numpy()[:] = ((-original) if ordinal == 1 else original).numpy()
        namespace["past_key_values"].position = offset
        expected = positions[0] if explicit_positions else torch.arange(seq_len) + offset
        exec(block, namespace)
        assert tuple(t._version for t in (positions, cos, sin)) == versions
        assert [t.data_ptr() for t in slots] == addresses
        assert torch.equal(bits(slots[0][:seq_len]), bits(cos[expected]))
        assert torch.equal(bits(slots[1][:seq_len]), bits(sin[expected]))
        assert all(torch.all(t[seq_len:] == 777) for t in slots)
        snapshots.append(tuple(t[:seq_len].clone() for t in slots))
    assert len(calls) == 3
    assert len(uploads) == (6 if route == "cached" else 0)
    assert all(torch.equal(bits(a), bits(b)) for a, b in zip(snapshots[0], snapshots[2]))
    assert not torch.equal(bits(snapshots[0][0]), bits(snapshots[1][0]))
    # Earlier fresh source rows and returned snapshots retain their own CPU bits.
    if route != "cached":
        assert len({src.data_ptr() for pair in calls for src in pair}) == 6
        assert torch.equal(bits(calls[0][0]), bits(snapshots[0][0]))
        assert torch.equal(bits(calls[1][0]), bits(snapshots[1][0]))
    # An identical forward still calls the setter; only the optional upload cache hits.
    exec(block, namespace)
    assert len(calls) == 4
    assert len(uploads) == (6 if route == "cached" else 0)


def test_native_rope_cpu_dispatch_refreshes_stable_owners_and_rejects_bad_inputs(tmp_path):
    """Actual Dispatcher/setter/private receipt and CPU/device copy publications.

    Owners, views, raw bits and versions are real ATen; device device labels,
    allocation and separate CPU/device stores are SDK doubles, not board or
    complete Graph evidence. Weight entry invalidation and fresh-slot allocation
    are extracted from the real installer, without recreating weight validation.
    """
    import re
    from test_siglip_attention_dry_live_contract import _function, _run_native

    source = (_ROOT / "src/fused/rpu_gemma_model.cpp").read_text()
    dispatch = (_ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    copy_source = (_ROOT / "src/core/rpu_tensor_ops.inc").read_text()
    setter = _function(source, "    void set_prefill_rope(").replace(
        "set_prefill_rope(", "set_prefill_rope_host(").replace("at::Tensor", "HostTensor")
    begin = source.index("    struct Pi05PrefillRopeSpan {")
    fields = source[begin:source.index("    bool prefill_rope_ready_", begin)].replace(
        "at::Tensor", "HostTensor")
    weights = _function(source, "    void set_weights(")
    entry = weights[weights.index("{") + 1:weights.index("        int64_t N =")]
    begin = weights.index("            prefill_cos_ = at::empty(")
    allocation = weights[begin:weights.index("            for (const auto& weights", begin)]
    install_end = weights[weights.rindex("        invalidate_model_state();"):weights.rindex("    }")]
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
#include <vector>
bool g_rpu_ddr_flush_enabled=true,g_rpu_ddr_flush_force_enabled=true;
struct Store{at::Tensor value;std::vector<uint8_t> ddr;bool rpu,device_dirty=false;};
std::vector<std::weak_ptr<Store>> stores;
std::vector<std::pair<void*,size_t>> flushes;
size_t copies=0,scenarios=0;
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
template<class A,class B>void cross_device_elementwise_copy(const HostTensor&,const HostTensor&){assert(false);}
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
struct Model{
 bool pi05_prefill_kv1_pair_direct_=true,prefill_rope_ready_=false,pi05_prefill_kv1_replica_proof_=false;
 std::vector<int> pi05_a8_scales_,chunk_mask_publications_;
 HostTensor cos_,sin_;int64_t prefix_rows=4;uint64_t generation=9;
 __FIELDS__
 int64_t pi05_prefix_rows()const{return prefix_rows;}
 uint64_t installed_model_state_generation()const{return generation;}
 void invalidate_model_state(){++generation;}
 void reinstall(int64_t prefix,int64_t capacity,bool fail=false){
  __WEIGHTS_ENTRY__
  TORCH_CHECK(!fail,"injected weight validation failure");
  prefix_rows=prefix;cos_=make_tensor({capacity,128},at::kPrivateUse1);sin_=make_tensor({capacity,128},at::kPrivateUse1);
  if(pi05_prefill_kv1_pair_direct_){__ALLOCATION__}
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
 // Actual direct cold allocation and all four independent publication flags.
 for(int64_t prefix:{544,800})for(bool normal:{false,true})for(bool force:{false,true})
 for(bool inference:{false,true})for(bool cos_rpu:{false,true})for(bool sin_rpu:{false,true}){
  ++scenarios;c10::InferenceMode mode(inference);
  g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;*shared=Model{};auto& m=*shared;
  m.reinstall(prefix,prefix+16);auto a=make_tensor({prefix+2,128},cos_rpu?at::kPrivateUse1:at::kCPU).narrow(0,1,prefix);
  auto b=make_tensor({prefix+2,128},sin_rpu?at::kPrivateUse1:at::kCPU).narrow(0,1,prefix);
  assert(a.storage_offset()==128&&a.is_contiguous());
  const auto av=version(a),bv=version(b),cv=version(m.prefill_cos_),sv=version(m.prefill_sin_);
  const auto cp=m.prefill_cos_.data_ptr(),sp=m.prefill_sin_.data_ptr();int calls=0,last=0;
  at::Tensor retained;
  for(int value:{7,7,-3,-3,7,7}){
   raw_fill(a,value);raw_fill(b,-value);assert(version(a)==av&&version(b)==bv);
   const bool hit=force&&!cos_rpu&&!sin_rpu&&value==last;
   const auto f=flushes.size(),c=copies;op.call(51,a.value,b.value);
   assert(copies-c==(hit?0:2));
   assert(flushes.size()-f==(force&&!hit?2+int(cos_rpu)+int(sin_rpu):0));
   assert(m.prefill_rope_ready_&&m.pi05_prefill_rope_cpu_published_==(force&&!cos_rpu&&!sin_rpu));
   assert(cp==m.prefill_cos_.data_ptr()&&sp==m.prefill_sin_.data_ptr());
   if(!hit)++calls;
   assert(version(m.prefill_cos_)==(inference?-1:cv+calls)&&version(m.prefill_sin_)==(inference?-1:sv+calls));
   check_bits(m.prefill_cos_,a,force);check_bits(m.prefill_sin_,b,force);
   assert(m.prefill_cos_.value.narrow(0,prefix,16).eq(777).all().item<bool>());
   assert(m.prefill_sin_.value.narrow(0,prefix,16).eq(777).all().item<bool>());
   if(!retained.defined())retained=m.prefill_cos_.value.narrow(0,0,prefix).clone();
   assert(retained.eq(7).all().item<bool>());last=value;
  }
  const auto saved=m.prefill_cos_.value.clone();
  rejects(m,[&]{op.call(51,a.value.narrow(0,0,prefix-1),b.value.narrow(0,0,prefix-1));});
  rejects(m,[&]{op.call(51,a.value.to(at::kFloat),b.value.to(at::kFloat));});
  rejects(m,[&]{op.call(51,a.value.flatten(),b.value.flatten());});
  rejects(m,[&]{op.call(51,a.value,b.value.narrow(0,0,prefix-1));});
  auto strided=at::zeros({128,prefix},at::kHalf).t();rejects(m,[&]{op.call(51,strided,b.value);});
  auto foreign=tagged(at::empty({prefix,128},at::TensorOptions().dtype(at::kHalf).device(at::kMeta)));
  rejects(m,[&]{m.set_prefill_rope_host(foreign,foreign);});
  bool caught=false;const auto f=flushes.size(),c=copies;
  try{op.call(99,a.value,b.value);}catch(const c10::Error&){caught=true;}assert(caught&&flushes.size()==f&&copies==c);
  a={};b={};assert(std::memcmp(saved.data_ptr(),m.prefill_cos_.data_ptr(),saved.nbytes())==0);
 }
 // Boundaries that cannot grant or retain a private, completed pair receipt.
 for(bool normal:{false,true})for(bool inference:{false,true}){
  ++scenarios;c10::InferenceMode mode(inference);g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=true;
  *shared=Model{};auto& m=*shared;constexpr int64_t prefix=544,capacity=2*prefix+16;m.reinstall(prefix,capacity);
  auto a=make_tensor({prefix+2,128},at::kCPU).narrow(0,1,prefix),b=make_tensor({prefix,128},at::kCPU);
  const auto av=version(a),bv=version(b);raw_fill(a,7);raw_fill(b,-7);
  auto call=[&](const HostTensor& x,const HostTensor& y,size_t expected_f,size_t expected_c){
   const auto f=flushes.size(),c=copies;op.call(51,x.value,y.value);
   assert(flushes.size()-f==expected_f&&copies-c==expected_c&&m.prefill_rope_ready_);
  };
  call(a,b,2,2);call(a,b,0,0);
  auto other_a=make_tensor({prefix,128},at::kCPU),other_b=make_tensor({prefix,128},at::kCPU);
  raw_fill(other_a,7);raw_fill(other_b,-7);call(other_a,other_b,0,0); // value, not identity
  raw_bits(a,0x7e11);raw_bits(b,0x8000);call(a,b,2,2);call(a,b,0,0);
  check_bits(m.prefill_cos_,a,true);check_bits(m.prefill_sin_,b,true);
  raw_bits(a,0x7e12);call(a,b,2,2);call(a,b,0,0); // a different NaN payload
  raw_bits(b,0x0000);call(a,b,2,2);call(a,b,0,0); // signed zero differs in bits
  assert(version(a)==av&&version(b)==bv);
  auto longer=make_tensor({prefix+8,128},at::kCPU);raw_fill(longer,31);
  call(longer,longer,2,2);call(longer,longer,0,0);
  raw_fill(a,7);raw_fill(b,-7);call(a,b,2,2);call(a,b,0,0);
  assert(m.prefill_cos_.value.narrow(0,prefix,8).eq(31).all().item<bool>());
  assert(m.prefill_cos_.value.narrow(0,prefix+8,capacity-prefix-8).eq(777).all().item<bool>());
  g_rpu_ddr_flush_force_enabled=false;call(a,b,0,2);assert(!m.pi05_prefill_rope_cpu_published_);
  raw_fill(a,23);call(a,b,0,2);check_bits(m.prefill_cos_,a,false);
  g_rpu_ddr_flush_force_enabled=true;call(a,b,2,2);call(a,b,0,0);check_bits(m.prefill_cos_,a,true);
  g_rpu_ddr_flush_enabled=!normal;call(a,b,0,0);g_rpu_ddr_flush_enabled=normal;
  ++m.generation;call(a,b,2,2);call(a,b,0,0);
  auto retired=m.prefill_cos_;bool caught=false;
  try{m.reinstall(prefix,capacity,true);}catch(const c10::Error&){caught=true;}
  assert(caught&&!m.pi05_prefill_rope_private_owned_&&!m.pi05_prefill_rope_cpu_published_&&!m.prefill_rope_ready_);
  call(a,b,2,2);call(a,b,2,2); // failed reinstall has no fresh-owner provenance
  m.reinstall(prefix,capacity);assert(retired.data_ptr()!=m.prefill_cos_.data_ptr());
  call(a,b,2,2);call(a,b,0,0);
  auto installed=m.prefill_cos_;m.prefill_cos_=make_tensor({capacity,128},at::kPrivateUse1);
  call(a,b,2,2);call(a,b,2,2);assert(!m.pi05_prefill_rope_cpu_published_);
  m.prefill_cos_=installed;call(a,b,2,2);call(a,b,0,0);
  // A new view TensorImpl of the same owner also cannot borrow its receipt.
  m.prefill_cos_=installed.narrow(0,0,capacity);call(a,b,2,2);call(a,b,2,2);
  assert(!m.pi05_prefill_rope_cpu_published_);m.prefill_cos_=installed;call(a,b,2,2);call(a,b,0,0);
  // Device-written device input still takes the original acquire/copy route.
  auto rpu=make_tensor({prefix,128},at::kPrivateUse1);raw_fill(rpu,7);
  auto* device=reinterpret_cast<c10::Half*>(rpu.store->ddr.data());for(int64_t i=0;i<rpu.numel();++i)device[i]=19;
  rpu.store->device_dirty=true;call(rpu,b,3,2);assert(!m.pi05_prefill_rope_cpu_published_);
  assert(m.prefill_cos_.value.narrow(0,0,prefix).eq(19).all().item<bool>());
  call(a,b,2,2);call(a,b,0,0);
  raw_fill(a,29);raw_fill(b,-29);flushes_until_failure=1;caught=false;
  try{op.call(51,a.value,b.value);}catch(const c10::Error&){caught=true;}
  assert(caught&&!m.prefill_rope_ready_&&!m.pi05_prefill_rope_cpu_published_);
  check_bits(m.prefill_cos_,a,true);check_bits(m.prefill_sin_,b,false);
  assert(std::memcmp(m.prefill_sin_.store->ddr.data(),b.data_ptr(),b.nbytes())!=0);
  call(a,b,2,2);call(a,b,0,0);check_bits(m.prefill_sin_,b,true);
  // The CPU source's logical range is outside the active destination prefix,
  // but its full backing overlaps that private owner. It must not grant a hit.
  auto backing=at::from_blob(m.prefill_cos_.data_ptr(),{capacity,128},at::kHalf);
  auto tail=tagged(backing.narrow(0,prefix,prefix));raw_fill(tail,29);
  assert(tail.device().is_cpu()&&tail.data_ptr()!=m.prefill_cos_.data_ptr());
  call(tail,b,2,2);call(tail,b,2,2);assert(!m.pi05_prefill_rope_cpu_published_);
  call(a,b,2,2);call(a,b,0,0);
  auto bad=a;bad.override_span=true;bad.storage_base=reinterpret_cast<uintptr_t>(a.store->value.data_ptr());
  bad.storage_bytes=std::numeric_limits<size_t>::max();auto f=flushes.size(),c=copies;
  m.set_prefill_rope_host(bad,b);assert(flushes.size()-f==2&&copies-c==2&&!m.pi05_prefill_rope_cpu_published_);
  call(a,b,2,2);call(a,b,0,0);
  bad.storage_bytes=a.nbytes()-1;f=flushes.size();c=copies;m.set_prefill_rope_host(a,bad);
  assert(flushes.size()-f==2&&copies-c==2&&!m.pi05_prefill_rope_cpu_published_);
  call(a,b,2,2);call(a,b,0,0);
  auto owner=m.prefill_sin_;m.prefill_sin_.override_span=true;
  m.prefill_sin_.storage_base=reinterpret_cast<uintptr_t>(m.prefill_cos_.store->value.data_ptr());
  m.prefill_sin_.storage_bytes=m.prefill_cos_.store->value.nbytes();
  call(a,b,2,2);assert(!m.pi05_prefill_rope_cpu_published_);m.prefill_sin_=owner;call(a,b,2,2);call(a,b,0,0);
  auto out_of_capacity=make_tensor({capacity+1,128},at::kCPU);
  rejects(m,[&]{op.call(51,out_of_capacity.value,out_of_capacity.value);});
  auto empty=make_tensor({0,128},at::kCPU);rejects(m,[&]{op.call(51,empty.value,empty.value);});
  auto old=m.prefill_cos_;m.prefill_cos_={};caught=false;
  try{op.call(51,a.value,b.value);}catch(const c10::Error&){caught=true;}
  assert(caught&&!m.prefill_cos_.defined()&&!m.prefill_rope_ready_&&!m.pi05_prefill_rope_cpu_published_);
  m.prefill_cos_=old;call(a,b,2,2);call(a,b,0,0);
 }
 // Preserve the actual copy_ inference-owner mutation failure outside its mode,
 // even when both private CPU-mapped values match a published pair exactly.
 HostTensor inference_a,inference_b;
 {
  c10::InferenceMode mode(true);*shared=Model{};shared->reinstall(544,560);
  inference_a=make_tensor({544,128},at::kCPU);inference_b=make_tensor({544,128},at::kCPU);
  raw_fill(inference_a,7);raw_fill(inference_b,-7);
  g_rpu_ddr_flush_force_enabled=true;op.call(51,inference_a.value,inference_b.value);
  assert(shared->pi05_prefill_rope_cpu_published_&&shared->prefill_cos_.is_inference());
 }
 assert(!c10::InferenceMode::is_enabled());
 auto f=flushes.size(),c=copies;bool caught=false;
 try{op.call(51,inference_a.value,inference_b.value);}catch(const c10::Error&){caught=true;}
 assert(caught&&copies-c==1&&flushes.size()-f==1&&!shared->pi05_prefill_rope_cpu_published_&&!shared->prefill_rope_ready_);
 {
  c10::InferenceMode mode(true);f=flushes.size();c=copies;op.call(51,inference_a.value,inference_b.value);
  assert(copies-c==2&&flushes.size()-f==2&&shared->pi05_prefill_rope_cpu_published_);
  f=flushes.size();c=copies;op.call(51,inference_a.value,inference_b.value);
  assert(copies==c&&flushes.size()==f&&shared->prefill_rope_ready_);
 }
 // Generic/non-direct Gemma keeps its original fresh lazy slots and copying.
 *shared=Model{};auto& m=*shared;m.pi05_prefill_kv1_pair_direct_=false;m.reinstall(4,16);
 g_rpu_ddr_flush_force_enabled=true;
 auto a=make_tensor({6,128},at::kCPU).narrow(0,1,4),b=make_tensor({4,128},at::kCPU);
 raw_fill(a,7);raw_fill(b,-7);void* cp=nullptr;
 for(int i=0;i<3;++i){const auto f=flushes.size(),c=copies;op.call(51,a.value,b.value);
  assert(flushes.size()-f==2&&copies-c==2&&!m.pi05_prefill_rope_cpu_published_);
  if(!cp)cp=m.prefill_cos_.data_ptr();assert(cp==m.prefill_cos_.data_ptr());check_bits(m.prefill_cos_,a,true);}
 std::cout<<"direct_route_cases="<<scenarios<<" publication/copy/owner controls passed\n";
}
'''
    for marker, value in (("__SETTER__", setter), ("__FIELDS__", fields),
                          ("__WEIGHTS_ENTRY__", entry), ("__ALLOCATION__", allocation),
                          ("__INSTALL_END__", install_end), ("__WRAPPER__", wrapper),
                          ("__SCHEMA__", schema), ("__REGISTRATIONS__", "\n".join(registrations)),
                          ("__PREFLIGHT__", preflight), ("__SAME_DEVICE__", same_device),
                          ("__CROSS_DEVICE__", cross_device)):
        code = code.replace(marker, value)
    _run_native(tmp_path, code, torch)
