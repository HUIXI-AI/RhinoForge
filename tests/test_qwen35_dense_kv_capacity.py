"""Capacity-sized dense Qwen3.5 text and Vision owners, without a board."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_5 import text


@pytest.mark.parametrize("family", ["dense", "moe"])
@pytest.mark.parametrize("loaded,explicit,expected", [
    (None, None, 8192), (32768, None, 32768), (32768, 16384, 16384),
])
def test_adapter_consumes_cold_loader_capacity(monkeypatch, family, loaded, explicit, expected):
    from rpu_backend.adapters import qwen3_5, qwen3_5_moe

    module = qwen3_5 if family == "dense" else qwen3_5_moe
    adapter_type = module.Qwen3_5Adapter if family == "dense" else module.Qwen3_5MoeAdapter
    model = SimpleNamespace(_rpu_execution={})
    if loaded is not None:
        model._rpu_requested_max_seq_len = loaded
    adapter = adapter_type.__new__(adapter_type)
    adapter.model = model
    adapter._rpu_is_ready = False
    seen = []

    def options(capacity, **kwargs):
        seen.append(capacity)
        raise RuntimeError("cold options boundary")

    monkeypatch.setattr(module, "_resolve_text_install_options", options)
    with pytest.raises(RuntimeError, match="cold options boundary"):
        adapter.to_rpu(**({} if explicit is None else {"max_seq_len": explicit}))
    assert seen == [expected]
    assert not hasattr(model, "_rpu_swizzle_started")


@pytest.mark.parametrize("family", ["dense", "moe"])
def test_ready_adapter_keeps_installed_capacity(monkeypatch, family):
    from rpu_backend.adapters import qwen3_5, qwen3_5_moe

    module = qwen3_5 if family == "dense" else qwen3_5_moe
    adapter_type = module.Qwen3_5Adapter if family == "dense" else module.Qwen3_5MoeAdapter
    state_getter = "_qwen3_5_text_runtime_state" if family == "dense" else "_qwen3_5_moe_text_runtime_state"
    model = SimpleNamespace(_rpu_requested_max_seq_len=32768)
    state = SimpleNamespace(max_seq_len=16384)
    adapter = adapter_type.__new__(adapter_type)
    adapter.model = model
    synchronized = []
    monkeypatch.setattr(module, state_getter, lambda owner: state)
    monkeypatch.setattr(module, "_sync_adapter_text_state", lambda owner, runtime: synchronized.append(runtime))
    monkeypatch.setattr(module, "_resolve_text_install_options", lambda *args, **kwargs: pytest.fail("ready install must not reallocate"))
    assert adapter.to_rpu() is model
    assert synchronized == [state]
    assert state.max_seq_len == 16384


@pytest.mark.parametrize("profile", [(24, 1024), (24, 2048), (32, 2560), (32, 4096), (64, 5120)])
@pytest.mark.parametrize("capacity", [16384, 32768])
def test_installed_dense_capacity_drives_text_image_prefill_and_decode(monkeypatch, profile, capacity):
    from test_execution_planner import _native_candidate_v2
    from test_qwen3_5_install_lifecycle import _text_preflight_owner

    bindings = []
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_chunk_envelope",
                        lambda *args: bindings.append(args), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_planner_cache_identity",
                        lambda handle: (handle, 1), raising=False)
    owner, cache = _text_preflight_owner()
    state = owner._rpu_qwen3_5
    state.max_seq_len = cache.max_seq_len = capacity
    state.num_layers, state.hidden_size = profile
    state.chunk_envelope = text._declare_qwen3_5_chunk_envelope(7, *profile, capacity)
    physical, ceiling = state.chunk_envelope
    cache.allocated_max_seq_len = physical
    assert bindings == [(7, capacity + 128, ceiling)]
    state.exact_chunk_size = ceiling
    state.padding_rows = 1
    calls = []

    def domain(handle, execution, logical):
        calls.append((handle, execution, logical))
        row = _native_candidate_v2(execution, (ceiling,) * 3,
            logical_len=logical, descriptor_version=3, graph_lifecycle=2)
        return [1, 1, len(row), *row]

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_resolve_prefill_stage_domain", domain, raising=False)
    for image in (False, True):
        padded, chunk, result = text._preflight_qwen3_5_text(
            owner, cache, capacity - 1, multimodal_prefill=image)
        assert (padded, chunk) == (capacity, ceiling)
        assert result.selected.padding_rows == 1
    assert calls == [(7, capacity, capacity - 1)] * 2
    cache.position = capacity - 1
    assert text._preflight_qwen3_5_text(owner, cache, 1) == (1, 0, None)
    cache.position = capacity
    with pytest.raises(ValueError, match="KV/rotary capacity"):
        text._preflight_qwen3_5_text(owner, cache, 1)
    cache.position = capacity // 2
    cache.max_seq_len = cache.position
    with pytest.raises(ValueError, match="cache capacity exceeded"):
        text._preflight_qwen3_5_text(owner, cache, 1)


@pytest.mark.parametrize("length,position", [(1048576, 0), (1, 1048560)])
def test_dense_register_capacity_fails_before_planning(monkeypatch, length, position):
    from test_qwen3_5_install_lifecycle import _text_preflight_owner

    owner, cache = _text_preflight_owner()
    owner._rpu_qwen3_5.max_seq_len = cache.max_seq_len = 2097152
    owner._rpu_qwen3_5.chunk_envelope = (2097280, 640)
    cache.position = position
    monkeypatch.setattr(text, "_plan_prefill_execution",
                        lambda *a, **kw: pytest.fail("invalid span reached planning"))
    with pytest.raises(text.UnsupportedModelError, match="kernel register capacity"):
        text._preflight_qwen3_5_text(owner, cache, length)


def test_native_vision_capacity_owns_stable_position_storage(tmp_path):
    from test_siglip_attention_dry_live_contract import _function, _run_native

    root = Path(__file__).resolve().parents[1]
    source = (root / "src/fused/rpu_qwen3_5_vision_model.cpp").read_text()
    capacity = _function(source, "void Qwen3_5VisionModel::configure_cache_capacity(")
    rope = _function(source, "void Qwen3_5VisionModel::set_rope_tables(")
    # Keep the actual allocation, alignment and cold guards; only the device
    # label/address provider is replaced to run against host ATen storage.
    rope = rope.replace("at::kPrivateUse1", "at::kCPU")
    cpp = r'''
#include <ATen/ATen.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
constexpr int64_t G05_TEMPORAL_MAX_SEQ=3*6*256;
uint64_t RpuGetDevAddr(void* p) { return reinterpret_cast<uint64_t>(p); }
struct Qwen3_5VisionModel {
    int64_t cache_capacity_=0, keepalive_capacity_=0, max_hw_=0;
    int64_t rows=0, layers=0, last=0, generation=0, orig_head_dim_=64;
    at::Tensor position_idx_keepalive_,q_ddr_buf_,hidden_buf_,merged_buf_,freq_cos_,freq_sin_;
    bool has_rope_=false;
    int64_t num_layers()const{return layers;}
    int64_t get_last_resolved_chunk_size()const{return last;}
    void set_chunk_envelope(int64_t value,int64_t chunk){assert(chunk==0);rows=value;}
    void invalidate_model_state(){++generation;}
    void configure_cache_capacity(int64_t);
    void set_rope_tables(const at::Tensor&,const at::Tensor&);
};
__CAPACITY__
__ROPE__
template<class F> void rejects(F fn){bool fail=false;try{fn();}catch(const c10::Error&){fail=true;}assert(fail);}
int main(){
    auto cos=at::zeros({128,16},at::kHalf),sin=at::ones_like(cos);
    for(int64_t capacity:{4097,16384,32768}){
        Qwen3_5VisionModel model;
        model.configure_cache_capacity(capacity);
        assert(model.rows==capacity&&model.cache_capacity_==capacity);
        model.layers=24;
        model.set_rope_tables(cos,sin);
        auto& positions=model.position_idx_keepalive_;
        assert(positions.size(0)==std::max(capacity,G05_TEMPORAL_MAX_SEQ));
        assert(positions.size(1)==2&&positions.scalar_type()==at::kShort);
        assert(RpuGetDevAddr(positions.data_ptr())%256==0);
        auto* address=positions.data_ptr();
        positions.fill_(7);
        model.set_rope_tables(cos,sin);
        assert(positions.data_ptr()==address&&positions[capacity-1][1].item<int16_t>()==7);
        rejects([&]{model.configure_cache_capacity(capacity*2);});
        assert(positions.data_ptr()==address&&model.cache_capacity_==capacity);
        auto oversized=at::empty({32769,16},at::kHalf);
        rejects([&]{model.set_rope_tables(oversized,oversized);});
        assert(model.max_hw_==128&&positions.data_ptr()==address);
    }
    for(int fault=0;fault<6;++fault){
        Qwen3_5VisionModel model;
        if(fault==0)model.layers=24;
        if(fault==1)model.last=16;
        if(fault==2)model.position_idx_keepalive_=cos;
        if(fault==3)model.q_ddr_buf_=cos;
        if(fault==4)model.hidden_buf_=cos;
        if(fault==5)model.merged_buf_=cos;
        rejects([&]{model.configure_cache_capacity(16384);});
        assert(model.cache_capacity_==0&&model.rows==0);
    }
    Qwen3_5VisionModel model;
    rejects([&]{model.configure_cache_capacity(0);});
    rejects([&]{model.configure_cache_capacity(std::numeric_limits<int64_t>::max());});
}
'''
    _run_native(tmp_path, cpp.replace("__CAPACITY__", capacity).replace("__ROPE__", rope), torch)
