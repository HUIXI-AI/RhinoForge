"""Resident and ordinary grouped paths share one typed N32K32 owner."""
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import weakref

import pytest
import torch

from test_qwen35_moe_rtn_weights import _checkpoint, _reader, _inverse_grouped
from test_qwen35_moe_resident_setup import _braced
from rpu_backend.adapters.qwen3_5_moe import weights
from rpu_backend.quant.grouped_uint4_n32k32 import unpack_grouped_uint4_n32k32

ROOT = Path(__file__).resolve().parents[1]


def _mlp(source):
    def linear(n, k):
        return SimpleNamespace(weight=torch.ones(n, k, dtype=torch.float16))
    return SimpleNamespace(
        experts=SimpleNamespace(_qwen3_5_moe_int4_source=source,
                                rpu_weight_mode=6, rpu_quant_profile=source.profile_id),
        gate=linear(256, 128),
        shared_expert=SimpleNamespace(gate_proj=linear(512, 128),
                                     up_proj=linear(512, 128), down_proj=linear(128, 512)),
        shared_expert_gate=linear(1, 128),
    )


@pytest.mark.parametrize("profile", ["rtn", "gptq"])
def test_projection_upload_order_codes_metadata_and_source_release(tmp_path, monkeypatch, profile):
    expected = _checkpoint(tmp_path, profile=profile)
    checkpoint = _reader(tmp_path, profile)
    source = checkpoint.layer_source(0)
    legacy = checkpoint.layer_source(0).prepare()
    events, references = [], []
    original_iterator = source.iter_prepared_projections
    original_to = torch.Tensor.to

    def projections(**kwargs):
        for role, packed, metadata in original_iterator(**kwargs):
            # Previous yielded projection is freed before decoding the next.
            assert all(ref() is None for ref in references)
            references.clear()
            events.append(("projection", role))
            references.append(weakref.ref(packed))
            yield role, packed, metadata
            del packed, metadata

    def upload(tensor, *args, **kwargs):
        if (args and args[0] == "rpu") or kwargs.get("device") == "rpu":
            assert not source.consumed
            events.append(("upload", tensor.dtype, tensor.dim()))
            return tensor.clone()
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(source, "iter_prepared_projections", projections)
    monkeypatch.setattr(torch.Tensor, "to", upload)
    result = weights.prepare_moe_layer_weights(_mlp(source), device="rpu", resident_uint4_n32k32=True)
    assert source.consumed and source.checkpoint is None
    assert result.resident_weight_layout == "grouped_uint4_n32k32_v1"
    for index, (name, role, n, k, partition) in enumerate((
        ("gate", "gate_proj", 512, 128, "col"),
        ("up", "up_proj", 512, 128, "col"),
        ("down", "down_proj", 128, 512, "row"),
    )):
        old = getattr(result, "routed_" + name)
        new = getattr(result, "resident_" + name)
        assert old is new
        assert torch.equal(old, legacy[index])
        codes = torch.stack([entry[0] for entry in expected[role]])
        assert torch.equal(_inverse_grouped(old, n=n, k=k, partition=partition), codes)
        assert torch.equal(unpack_grouped_uint4_n32k32(new, n=n, k=k, partition=partition), codes)
        assert torch.equal(getattr(result, "routed_" + name + "_scale"), legacy[index + 3])
        assert events[index * 3:index * 3 + 3] == [
            ("projection", role), ("upload", torch.uint8, 2),
            ("upload", torch.float16, 6)]
    assert len(events) == 14  # three projection groups plus router/shared uploads


@pytest.mark.parametrize("failure_upload", [2, 4, 7, 11])
def test_failed_partial_upload_keeps_source_retryable(tmp_path, monkeypatch, failure_upload):
    _checkpoint(tmp_path, profile="gptq")
    source = _reader(tmp_path, "gptq").layer_source(0)
    original_to = torch.Tensor.to
    uploads = 0

    def failing_upload(tensor, *args, **kwargs):
        nonlocal uploads
        if args and args[0] == "rpu":
            uploads += 1
            if uploads == failure_upload:
                raise RuntimeError("injected upload failure")
            return tensor.clone()
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", failing_upload)
    with pytest.raises(RuntimeError, match="injected upload"):
        weights.prepare_moe_layer_weights(_mlp(source), device="rpu", resident_uint4_n32k32=True)
    assert not source.consumed and source.checkpoint is not None
    # Retry is a new cold preparation; no partial result or source release escaped.
    result = weights.prepare_moe_layer_weights(_mlp(source), device="cpu", resident_uint4_n32k32=True)
    assert source.consumed and result.resident_gate is not None


def test_default_install_keeps_legacy_only_and_opt_in_requires_uint4(tmp_path):
    _checkpoint(tmp_path)
    source = _reader(tmp_path).layer_source(0)
    result = weights.prepare_moe_layer_weights(_mlp(source), device="cpu")
    assert result.resident_gate is result.resident_up is result.resident_down is None
    assert result.resident_weight_layout is None
    mlp = _mlp(_reader(tmp_path).layer_source(0))
    with pytest.raises(ValueError, match="UINT4 TP8"):
        weights.prepare_moe_layer_weights(mlp, device="cpu", num_cores=4, resident_uint4_n32k32=True)
    del mlp.experts._qwen3_5_moe_int4_source
    with pytest.raises(ValueError, match="UINT4 TP8"):
        weights.prepare_moe_layer_weights(mlp, device="cpu", resident_uint4_n32k32=True)


def test_native_cold_owner_install_atomicity_and_lifetime(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host compiler unavailable")
    source = (ROOT / "src/fused/rpu_qwen3_5_moe_model.cpp").read_text()
    setter = _braced(source, source.index("void Qwen3_5MoeModel::set_resident_uint4_n32k32_weights("))
    header = (ROOT / "src/fused/rpu_qwen3_5_moe_model.h").read_text()
    layer = _braced(header, header.index("    struct MoeLayerWeights")) + ";"
    harness = r'''
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error("rejected");}while(0)
constexpr int64_t kRoutedUint4Mode=6;
namespace at {
constexpr int kByte=1,kHalf=2,kPrivateUse1=3,kCPU=4;
struct Tensor{
 struct Storage{uint64_t address;};std::shared_ptr<Storage> storage;
 int dtype=kByte,kind=kPrivateUse1,device_id=0;bool contiguous=true;
 std::vector<int64_t> shape{256,524288};
 struct Device{int kind,id;int type()const{return kind;}bool operator==(Device b)const{return kind==b.kind&&id==b.id;}};
 bool defined()const{return bool(storage);}Device device()const{return {kind,device_id};}
 int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}
 int64_t dim()const{return shape.size();}int64_t size(int i)const{return shape.at(i);}
 uint64_t nbytes()const{uint64_t n=dtype==kByte?1:2;for(auto s:shape)n*=s;return n;}
 void* data_ptr()const{return reinterpret_cast<void*>(storage?storage->address:0);}
};
using TensorList=std::vector<Tensor>;
}
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uint64_t>(p);}
at::Tensor owner(uint64_t address){at::Tensor t;t.storage=std::make_shared<at::Tensor::Storage>(at::Tensor::Storage{address});return t;}
struct Qwen3_5MoeModel {
__LAYER__
 struct Moe{int64_t num_experts=256,routed_intermediate=512;}moe_;
 bool resident_prefill_=false,resident_uint4_n32k32_installed_=false;
 int64_t resolved=0;int invalidations=0;
 std::vector<MoeLayerWeights> moe_layer_weights_=std::vector<MoeLayerWeights>(40);
 int64_t get_last_resolved_chunk_size()const{return resolved;}
 int64_t num_cores()const{return 8;}int64_t num_layers()const{return 40;}int64_t hidden_size()const{return 2048;}
 void invalidate_model_state(){++invalidations;}
 void set_resident_uint4_n32k32_weights(at::TensorList,at::TensorList,at::TensorList);
 Qwen3_5MoeModel(){for(size_t l=0;l<40;++l){auto& w=moe_layer_weights_[l];w.routed_weight_mode=6;
 w.routed_gate_w=owner(0x1000000000ULL+(l*3+0)*134217728ULL);
 w.routed_up_w=owner(0x1000000000ULL+(l*3+1)*134217728ULL);
 w.routed_down_w=owner(0x1000000000ULL+(l*3+2)*134217728ULL);}}
};
__SETTER__
using Lists=std::vector<at::TensorList>;
Lists owners(){Lists list(3);for(size_t p=0;p<3;++p)for(size_t l=0;l<40;++l)
 list[p].push_back(owner(0x1000000000ULL+(l*3+p)*134217728ULL));return list;}
void install(Qwen3_5MoeModel& m,const Lists& o){m.set_resident_uint4_n32k32_weights(o[0],o[1],o[2]);}
void rejects(Qwen3_5MoeModel& m,const Lists& o){bool rejected=false;try{install(m,o);}catch(const std::runtime_error&){rejected=true;}
 assert(rejected&&!m.resident_uint4_n32k32_installed_&&m.invalidations==0);
 for(const auto& w:m.moe_layer_weights_)assert(!w.resident_gate_w.defined()&&!w.resident_up_w.defined()&&!w.resident_down_w.defined());}
int main(){
 auto o=owners();
 {Qwen3_5MoeModel m;auto bad=o;bad[2][39].dtype=at::kHalf;rejects(m,bad);}
 {Qwen3_5MoeModel m;auto bad=o;bad[2].pop_back();rejects(m,bad);}
 {Qwen3_5MoeModel m;auto bad=o;bad[1][0]=m.moe_layer_weights_[39].routed_down_w;rejects(m,bad);}
 {Qwen3_5MoeModel m;auto bad=o;bad[2][39]=bad[0][0];rejects(m,bad);}
 {Qwen3_5MoeModel m;auto bad=o;bad[2][39]=owner(0x2000000001ULL);rejects(m,bad);}
 {Qwen3_5MoeModel m;auto bad=o;bad[2][39]=owner(1ULL<<40);rejects(m,bad);}
 {Qwen3_5MoeModel m;auto bad=o;bad[1][2].device_id=1;rejects(m,bad);}
 {Qwen3_5MoeModel m;m.moe_layer_weights_[39].routed_weight_mode=5;rejects(m,o);}
 {Qwen3_5MoeModel m;m.resolved=64;rejects(m,o);}
 {Qwen3_5MoeModel m;m.resident_prefill_=true;rejects(m,o);}
 std::weak_ptr<at::Tensor::Storage> weak=o[2][39].storage;
 {Qwen3_5MoeModel m;const auto old=m.moe_layer_weights_[39].routed_down_w.data_ptr();install(m,o);
 assert(m.resident_uint4_n32k32_installed_&&m.invalidations==1);
 assert(m.moe_layer_weights_[39].routed_down_w.data_ptr()==old);
 assert(m.moe_layer_weights_[39].resident_down_w.storage==o[2][39].storage);
 bool rejected=false;try{install(m,o);}catch(const std::runtime_error&){rejected=true;}assert(rejected&&m.invalidations==1);
 o.clear();assert(!weak.expired());}
 assert(weak.expired());
}
'''.replace("__LAYER__", layer).replace("__SETTER__", setter)
    cpp, executable = tmp_path / "owner.cpp", tmp_path / "owner"
    cpp.write_text(harness)
    compiled = subprocess.run([compiler, "-std=c++17", "-O0", str(cpp), "-o", str(executable)], text=True, capture_output=True)
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(executable)], text=True, capture_output=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr


def test_cold_owner_setter_real_aten_and_dispatcher_types(tmp_path):
    """Check actual TensorList references and inferred schema without native build."""
    compiler = shutil.which("g++")
    torch_include = Path(torch.__file__).parent / "include"
    if not compiler or not (torch_include / "ATen/ATen.h").is_file():
        pytest.skip("host syntax check requires g++ and ATen headers")
    source = (ROOT / "src/fused/rpu_qwen3_5_moe_model.cpp").read_text()
    setter = _braced(source, source.index("void Qwen3_5MoeModel::set_resident_uint4_n32k32_weights("))
    entry = _braced(source, source.index("void rpu_qwen3_5_moe_set_resident_uint4_n32k32_weights("))
    header = (ROOT / "src/fused/rpu_qwen3_5_moe_model.h").read_text()
    layer = _braced(header, header.index("    struct MoeLayerWeights")) + ";"
    registrations = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    begin = registrations.index('  m.def("qwen3_5_moe_set_resident_uint4_n32k32_weights",')
    registration = registrations[begin:registrations.index(";", begin) + 1]
    cpp = tmp_path / "owner_syntax.cpp"
    cpp.write_text(r'''
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cstdint>
#include <vector>
#include <utility>
constexpr int64_t kRoutedUint4Mode=6;
uint64_t RpuGetDevAddr(void*);
struct Qwen3_5MoeModel {
__LAYER__
 struct Moe{int64_t num_experts,routed_intermediate;}moe_;
 bool resident_prefill_=false,resident_uint4_n32k32_installed_=false;
 std::vector<MoeLayerWeights> moe_layer_weights_;
 int64_t get_last_resolved_chunk_size() const;
 int64_t num_cores() const; int64_t num_layers() const; int64_t hidden_size() const;
 void invalidate_model_state();
 void set_resident_uint4_n32k32_weights(at::TensorList,at::TensorList,at::TensorList);
};
struct Qwen3_5MoeRegistry {static Qwen3_5MoeModel* get(int64_t,const char*);};
__SETTER__
__ENTRY__
TORCH_LIBRARY(rpu_qwen35_owner_host_syntax,m) {
__REGISTRATION__
}
'''.replace("__LAYER__", layer).replace("__SETTER__", setter)
        .replace("__ENTRY__", entry).replace("__REGISTRATION__", registration))
    compiled = subprocess.run([
        compiler, "-std=c++17", "-fsyntax-only", "-isystem", str(torch_include),
        "-isystem", str(torch_include / "torch/csrc/api/include"), str(cpp),
    ], text=True, capture_output=True)
    assert compiled.returncode == 0, compiled.stderr
