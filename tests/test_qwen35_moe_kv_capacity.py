"""Capacity-driven MoE allocation and execution admission, without a board."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_5_moe import text
from rpu_backend.adapters.qwen3_5_moe.cores import core_topology
from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache
from test_qwen35_moe_python_integration import _exact_config


@pytest.mark.parametrize("capacity", [1, 8193, 16384, 32768, 1048576])
def test_factory_keeps_requested_capacity_and_physical_padding(monkeypatch, capacity):
    from transformers import Qwen3_5MoeConfig

    config = Qwen3_5MoeConfig.from_dict(_exact_config())
    # Preserve the full forty-layer geometry without reserving multi-GB host
    # storage; generic cache tests exercise real allocations separately.
    empty, zeros = torch.empty, torch.zeros
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: empty(*a, **(kw | {"device": "meta"})))
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: zeros(*a, **(kw | {"device": "meta"})))
    cache = Qwen3_5MoeCache.from_config(config, capacity, device="cpu")
    physical, chunk = text._qwen3_5_moe_prefill_envelope(40, 2048, capacity)
    assert cache.max_seq_len == capacity
    assert cache.allocated_max_seq_len == physical
    assert capacity + 127 <= physical < capacity + 143
    assert cache.k_caches[3].shape[1] * 16 == physical
    assert cache.v_caches[3].shape[1] * 16 == physical
    assert len(cache.k_caches) == 40 and cache.k_caches[0].numel() == 0
    assert chunk == 512


@pytest.mark.parametrize("capacity", [8193, 16384, 32768])
def test_direct_install_passes_larger_capacity_to_weight_owner(monkeypatch, capacity):
    import rpu_backend
    from transformers import Qwen3_5MoeConfig

    config = Qwen3_5MoeConfig.from_dict(_exact_config())
    for name in ("qwen3_5_moe_set_execution_cores", "qwen3_5_moe_get_execution_topology",
                 "qwen3_5_moe_get_execution_topology_v2"):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *a: None, raising=False)
    monkeypatch.setattr(rpu_backend.graph, "GraphCache", lambda **kw: object())
    monkeypatch.setattr(rpu_backend.graph, "Graph", lambda **kw: object())
    captured = {}

    def install(owner, cfg, **options):
        captured.update(options)
        return 91

    monkeypatch.setattr(text, "_install_qwen3_5_moe_text_for_rpu_impl", install)
    assert text.install_qwen3_5_moe_text_for_rpu(
        SimpleNamespace(), config, capacity,
        execution_config={"prefill": {"chunk_size": 512}},
    ) == 91
    assert captured["max_seq_len"] == capacity
    assert captured["exact_chunk_size"] == 512


def _owner(monkeypatch, capacity, *, cache_capacity=None, position=0, padding_rows="auto"):
    physical, chunk = text._qwen3_5_moe_prefill_envelope(40, 2048, capacity)
    state = SimpleNamespace(
        handle=91, execution_topology=core_topology(), max_seq_len=capacity,
        chunk_envelope=(physical, chunk), padding_budget=64,
        padding_rows=padding_rows, exact_chunk_size=512,
    )
    owner = SimpleNamespace(_rpu_qwen3_5_moe=state)
    cache = SimpleNamespace(
        execution_topology=(8, 8, 8, 8, 8), position=position,
        max_seq_len=capacity if cache_capacity is None else cache_capacity,
        allocated_max_seq_len=physical,
    )
    monkeypatch.setattr(text, "_qwen35_moe_retirement_ready", lambda value: value is state)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_planner_cache_identity",
                        lambda handle: (handle, 1), raising=False)
    return owner, cache


@pytest.mark.parametrize("capacity,padding", [(8193, 63), (16384, 64), (32768, 64)])
@pytest.mark.parametrize("multimodal", [False, True])
def test_prefill_planner_uses_installed_capacity_and_full_padding(
    monkeypatch, capacity, padding, multimodal,
):
    from test_execution_planner import _native_candidate_v2

    owner, cache = _owner(monkeypatch, capacity, padding_rows=padding)
    # C448 lets the final chunk contain both real tokens and padding at these
    # lengths; C512 would introduce a forbidden padding-only tail at 16K/32K.
    owner._rpu_qwen3_5_moe.exact_chunk_size = 448
    queried = []

    def domain(handle, execution, logical):
        queried.append((execution, logical))
        descriptor = _native_candidate_v2(
            execution, (448, 448, 448), logical_len=logical,
            descriptor_version=3, graph_lifecycle=2,
        )
        return [1, 1, len(descriptor), *descriptor]

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_resolve_prefill_stage_domain", domain,
                        raising=False)
    physical, chunk, result = text._preflight_qwen3_5_moe_text(
        owner, cache, capacity, multimodal_prefill=multimodal)
    assert (physical, chunk) == (capacity + padding, 448)
    assert queried == [(capacity + padding, capacity)]
    assert result.selected.execution_len - result.selected.padding_rows == capacity
    assert result.selected.padding_rows == padding
    assert cache.position == 0


@pytest.mark.parametrize("capacity", [16384, 32768])
def test_decode_beyond_8192_stops_at_actual_cache_or_rotary_capacity(monkeypatch, capacity):
    owner, cache = _owner(monkeypatch, capacity, position=capacity - 1)
    assert text._preflight_qwen3_5_moe_text(owner, cache, 1) == (1, 0, None)
    cache.position = capacity
    with pytest.raises(ValueError, match="KV/rotary capacity"):
        text._preflight_qwen3_5_moe_text(owner, cache, 1)
    cache.position = 8192
    cache.max_seq_len = 8192
    with pytest.raises(ValueError, match="cache capacity exceeded"):
        text._preflight_qwen3_5_moe_text(owner, cache, 1)


@pytest.mark.parametrize("length,position", [(1048576, 0), (1, 1048560)])
def test_register_overflow_is_rejected_before_native_planning(monkeypatch, length, position):
    owner, cache = _owner(monkeypatch, 2097152, position=position)
    monkeypatch.setattr(text, "_plan_prefill_execution",
                        lambda *a, **kw: pytest.fail("overflow reached planning"))
    with pytest.raises(text.UnsupportedModelError, match="kernel register capacity"):
        text._preflight_qwen3_5_moe_text(owner, cache, length)


def test_required_padding_cannot_cross_kernel_register_capacity(monkeypatch):
    from rpu_backend.runtime.execution_planner import PlannerRejectError

    owner, cache = _owner(monkeypatch, 2097152)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_moe_resolve_prefill_stage_domain",
                        lambda *a: pytest.fail("invalid padded span reached native planning"),
                        raising=False)
    with pytest.raises(PlannerRejectError):
        text._preflight_qwen3_5_moe_text(owner, cache, 1048560)


def test_native_capacity_binding_and_prework_rope_checks(tmp_path):
    from test_qwen35_moe_core_native import NATIVE, _definition, _run_cpp

    source = NATIVE.read_text()
    cap = _definition(source, "int64_t Qwen3_5MoeModel::subclass_chunk_size_cap(")
    install = _definition(source, "void Qwen3_5MoeModel::set_weights(")
    binding = install[install.index("    constexpr int64_t total_padding"):
                      install.index("    has_qk_norm_ =")]
    forward = _definition(source, "at::Tensor Qwen3_5MoeModel::forward_impl(")
    checks = forward[forward.index("    TORCH_CHECK(seq_len == 1 ||"):
                     forward.index("    gdn_states_  =")]
    assert forward.index("rotary position delta exceeds") < forward.index("gdn_states_  =")
    probe = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while(false)
constexpr int64_t kSdpaMaxKvRows = int64_t{UINT16_MAX} * 16;
struct Tensor {
    int64_t rows = 0;
    bool defined() const { return rows > 0; }
    int64_t numel() const { return rows; }
    int64_t dim() const { return 2; }
    int64_t size(int) const { return rows; }
    std::pair<int64_t,int64_t> sizes() const { return {rows,32}; }
};
struct Qwen3_5MoeModel {
    int64_t physical_capacity = 0, chunk = 0, chunk_size_cap_ = 0;
    int64_t mrope_pos_delta_ = 0, work = 0;
    bool has_mrope_ = true;
    Tensor cos_, sin_, prefill_cos_, prefill_sin_;
    void set_chunk_envelope(int64_t maximum, int64_t ceiling) {
        physical_capacity = maximum; chunk = ceiling;
    }
    int64_t enforce_chunk_envelope(int64_t seq, int64_t position, const char*) const {
        TORCH_CHECK(seq <= physical_capacity && position <= physical_capacity-seq);
        return chunk;
    }
    int64_t subclass_chunk_size_cap(int64_t seq_len,int64_t position) const;
    void bind(int64_t rows) {
        Tensor cos{rows}, expert_ids{512};
        const int64_t top_k = 8;
__BINDING__
        cos_ = sin_ = cos;
    }
    void forward(int64_t seq_len,int64_t position) {
__CHECKS__
        ++work;
    }
};
__CAP__
template<class Fn> void rejects(Fn fn) {
    bool rejected=false; try { fn(); } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
}
int main() {
    Qwen3_5MoeModel model;
    for (const auto row : {std::pair<int64_t,int64_t>{1,128}, {8193,8320},
                           {16384,16512}, {32768,32896}}) {
        model.bind(row.first);
        assert(model.physical_capacity==row.second && model.chunk==512);
        model.forward(1,row.first-1);
        const auto work=model.work;
        rejects([&]{model.forward(1,row.first);});
        assert(model.work==work);
    }
    const auto old_capacity=model.physical_capacity;
    rejects([&]{model.bind(std::numeric_limits<int64_t>::max());});
    assert(model.physical_capacity==old_capacity);
    model.bind(32768);
    model.mrope_pos_delta_=-23;
    model.forward(1,32767);
    model.mrope_pos_delta_=1;
    rejects([&]{model.forward(1,32767);});
    model.mrope_pos_delta_=-1;
    rejects([&]{model.forward(1,0);});
    for (auto delta : {std::numeric_limits<int64_t>::min(),
                       std::numeric_limits<int64_t>::max()}) {
        model.mrope_pos_delta_=delta;
        rejects([&]{model.forward(1,32767);});
    }
    model.prefill_cos_=model.prefill_sin_=Tensor{32832};
    // Per-token dynamic tables already encode the positions: no second delta.
    model.forward(32832,0);
    model.prefill_sin_=Tensor{32831};
    rejects([&]{model.forward(32832,0);});
    model.prefill_cos_=model.prefill_sin_=Tensor{32831};
    rejects([&]{model.forward(32832,0);});
    model.prefill_cos_=model.prefill_sin_=Tensor{};
    model.mrope_pos_delta_=0;
    rejects([&]{model.forward(32832,0);});
    rejects([&]{model.forward(1,-1);});
    rejects([&]{model.forward(0,0);});
    rejects([&]{model.forward(65,0);});
    model.bind(2097152);
    assert(model.subclass_chunk_size_cap(1048560,0)==512);
    model.forward(1,1048559);
    rejects([&]{model.forward(1,1048560);});
    rejects([&]{model.subclass_chunk_size_cap(1048576,0);});
    rejects([&]{model.subclass_chunk_size_cap(1,std::numeric_limits<int64_t>::max());});
}
'''
    probe = probe.replace("__BINDING__", binding).replace("__CHECKS__", checks).replace("__CAP__", cap)
    _run_cpp(tmp_path, probe)
