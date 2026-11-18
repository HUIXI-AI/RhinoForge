"""Host contracts for the exact 32B runtime-W8 cold instruction arena plan.

Real adapter/installer/cache decisions run with native and tensor-transfer
boundaries doubled. These tests do not establish SDK capacity or RPU numerics.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace as NS

import pytest

import rpu_backend
from rpu_backend.adapters import qwen3_vl
from rpu_backend.adapters.qwen3_vl import text, vision
from rpu_backend.graph import GraphRuntimePolicy
from rpu_backend.runtime import decoder
from rpu_backend.runtime.chunk_envelope import ChunkEnvelope
from rpu_backend.tests.acceptance.test_decoder_install_transaction import (
    _FakeTensor,
    _finalize as finalize_text,
    _model as text_model,
    native_install as text_native_install,  # noqa: F401
)
from rpu_backend.tests.acceptance.test_qwen3_vl_install_lifecycle import (
    _Model,
    install_runtime,  # noqa: F401
)
from rpu_backend.tests.acceptance.test_vla_vision_install_transaction import (
    _finalize as finalize_vision,
    _model as vision_model,
    native_install as vision_native_install,  # noqa: F401
)
from test_graph_runtime_policy import runtime_module  # noqa: F401
from test_qwen3_vl_large_quantized_adapter import (
    PROFILES,
    quantized_install,  # noqa: F401
    small_quantized_model,
)


@pytest.mark.parametrize("size,bits", PROFILES)
def test_exact_profile_reserves_before_mutation_and_passes_real_cache_limits(
    monkeypatch, quantized_install, size, bits,
):
    state = quantized_install
    model = small_quantized_model(size, bits)
    policies = []
    baseline = GraphRuntimePolicy.from_environment()

    def prepare(policy):
        assert state["claims"] == [model]
        assert state["allocator_enabled"] is True
        assert not getattr(model, "_rpu_swizzle_started", False)
        assert not any(event[0] in ("move_runtime_decoder", "rpu_allocation", "install")
                       for event in state["events"])
        policies.append(policy)
        state["events"].append(("prepare_arenas", policy.graph_arena_count))
        return True

    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", prepare)
    owner = qwen3_vl.Qwen3VLAdapter(model)
    assert owner.to_rpu() is model
    selected = (size, bits) == (32, 8)
    assert len(policies) == int(selected)
    for kwargs in (state["text_kwargs"], state["vision_kwargs"]):
        if selected:
            assert kwargs["_graph_cache_max_entries"] == 8
        else:
            assert "_graph_cache_max_entries" not in kwargs
    if selected:
        policy = policies[0]
        assert owner._graph_arena_policy is policy
        assert policy.graph_arena_count == 3 * 8 + 1
        # Arena placement must not alter SDK ceilings, ordinary segmentation,
        # execution cores or any replay policy captured from the environment.
        for field in fields(policy):
            if field.name not in ("graph_arena_count", "provenance"):
                assert getattr(policy, field.name) == getattr(baseline, field.name)
        events = state["events"]
        assert events.index(("claim", "model")) < events.index(("set_caching_allocator", True))
        assert events.index(("set_caching_allocator", True)) < events.index(("prepare_arenas", 25))
        assert events.index(("prepare_arenas", 25)) < events.index(("move_runtime_decoder",))
    assert owner.to_rpu() is model
    assert len(policies) == int(selected), "ready re-entry must not reserve twice"


def test_ordinary_fp16_owner_does_not_prepare_quantized_arenas(monkeypatch, install_runtime):
    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas",
                        lambda self: pytest.fail("ordinary FP16 reserved quantized arenas"))
    model = _Model()
    assert qwen3_vl.Qwen3VLAdapter(model).to_rpu() is model
    assert "_graph_cache_max_entries" not in install_runtime["text_kwargs"]
    assert "_graph_cache_max_entries" not in install_runtime["vision_kwargs"]


@pytest.mark.parametrize("failure", [False, RuntimeError, KeyboardInterrupt])
def test_arena_failure_releases_claim_without_mutating_weights(
    monkeypatch, quantized_install, failure,
):
    state = quantized_install
    model = small_quantized_model(32, 8)
    original = model.lm_head.weight
    releases = []
    monkeypatch.setattr(qwen3_vl, "_release_live_instance", releases.append)

    def fail_prepare(policy):
        assert policy.graph_arena_count == 25
        if failure is False:
            return False
        raise failure("synthetic cold arena failure")

    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", fail_prepare)
    owner = qwen3_vl.Qwen3VLAdapter(model)
    expected = qwen3_vl.RPUBackendError if failure is False else failure
    with pytest.raises(expected):
        owner.to_rpu()
    assert releases == [model]
    assert model.lm_head.weight is original
    assert not getattr(model, "_rpu_swizzle_started", False)
    assert "_qwen3_vl_retirement_owner" not in vars(model)
    assert not any(event[0] in ("move_runtime_decoder", "rpu_allocation", "install")
                   for event in state["events"])
    assert qwen3_vl._SWIZZLE_LOCK.acquire(blocking=False)
    qwen3_vl._SWIZZLE_LOCK.release()


@pytest.mark.parametrize("boundary", ["claim", "allocator"])
def test_earlier_install_failure_never_attempts_arena_reservation(
    monkeypatch, quantized_install, boundary,
):
    state = quantized_install
    state[boundary + "_error"] = RuntimeError("earlier failure")
    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas",
                        lambda self: pytest.fail("arena reservation preceded ownership"))
    model = small_quantized_model(32, 8)
    with pytest.raises(RuntimeError, match="earlier failure"):
        qwen3_vl.Qwen3VLAdapter(model).to_rpu()
    assert not getattr(model, "_rpu_swizzle_started", False)


def _use_real_cache_constructor(monkeypatch, runtime_module):
    """Run the real Python GraphCache with a non-allocating native cache double."""
    native = runtime_module._cpp_ext.GraphCache
    monkeypatch.setattr(native, "clear", lambda self: None, raising=False)
    monkeypatch.setattr(native, "cache_invariant_ok", lambda self: True, raising=False)
    monkeypatch.setattr(rpu_backend.graph, "GraphCache", runtime_module.GraphCache)
    monkeypatch.setattr(rpu_backend.graph, "GraphRuntimePolicy", runtime_module.GraphRuntimePolicy)


@pytest.mark.parametrize("cap,expected", [(None, 32), (8, 8)])
def test_both_real_mrope_installer_caches_receive_same_bound(
    monkeypatch, text_native_install, runtime_module, cap, expected,
):
    _use_real_cache_constructor(monkeypatch, runtime_module)
    model = text_model(model_type="qwen3")
    model.config.model_type = "qwen3_vl_text"
    model.config.num_hidden_layers = 1
    model.config.rope_scaling = {"mrope_section": [24, 20, 20]}
    table = _FakeTensor()
    monkeypatch.setattr(decoder, "build_mrope_cos_sin_tables", lambda *a, **k: (table, table))
    try:
        decoder.install_mrope_text_decoder_for_rpu(
            model, arch="qwen3_vl_text", chunk_envelope_for=lambda *a: ChunkEnvelope(128),
            _graph_cache_max_entries=cap,
        )
        outer, inner = model._rpu_text_graph_cache, model._rpu_decoder_graph_cache
        assert outer is not inner, "the two public call routes must retain separate graphs"
        assert outer._impl.max_entries == inner._impl.max_entries == expected
        for cache in (outer, inner):
            assert cache.runtime_policy.lkn_max_batch_entries == 262_144
            assert cache.runtime_policy.lkn_kd_buf_mb == 64
            assert cache.runtime_policy.lkn_instr_buf_mb == 512
            assert not cache.runtime_policy.qwen35_legacy_27b_sdk_budget
    finally:
        finalize_text(model)


def test_large_text_adapter_forwards_owner_bound_into_shared_installer(monkeypatch):
    model = small_quantized_model(32, 8).model.language_model
    seen = []
    # The tiny tensor owners cannot satisfy a 32B full inventory. Geometry and
    # arithmetic have separate gates; this case isolates this adapter boundary.
    monkeypatch.setattr(text, "_validate_runtime_quantized_32b_text", lambda *a: None)
    monkeypatch.setattr(text, "install_mrope_text_decoder_for_rpu",
                        lambda *a, **kwargs: seen.append(kwargs) or 19)
    assert text.install_qwen3_vl_text_for_rpu(
        model, _runtime_quantized_32b=True, _graph_cache_max_entries=8,
    ) == 19
    assert seen[0]["_graph_cache_max_entries"] == 8
    assert seen[0]["_kv_cache_layer_bank_size"] == 8


@pytest.mark.parametrize("vision_native_install", [vision], indirect=True)
@pytest.mark.parametrize("cap,expected", [(None, 32), (8, 8)])
def test_real_vision_installer_forwards_bound_to_actual_cache_constructor(
    monkeypatch, vision_native_install, runtime_module, cap, expected,
):
    _use_real_cache_constructor(monkeypatch, runtime_module)
    model = vision_model()
    try:
        vision.install_qwen3_vl_vision_for_rpu(model, _graph_cache_max_entries=cap)
        assert model._rpu_vision_graph_cache._impl.max_entries == expected
        assert not model._rpu_vision_graph_cache.runtime_policy.qwen35_legacy_27b_sdk_budget
    finally:
        finalize_vision(model)


@pytest.mark.parametrize("cap", [True, False, 0, -1, 8.0, "8"])
def test_invalid_private_cache_bound_fails_before_shared_weight_or_native_work(monkeypatch, cap):
    monkeypatch.setattr(decoder, "build_mrope_cos_sin_tables",
                        lambda *a, **k: pytest.fail("invalid capacity allocated RoPE"))
    with pytest.raises((TypeError, ValueError), match="entries|capacity"):
        decoder.install_mrope_text_decoder_for_rpu(
            NS(), arch="qwen3_vl_text", chunk_envelope_for=lambda *a: ChunkEnvelope(128),
            _graph_cache_max_entries=cap,
        )
    with pytest.raises((TypeError, ValueError), match="entries|capacity"):
        decoder._install_causal_decoder_forward(
            NS(), arch="qwen3_vl_text", chunk_envelope_for=lambda *a: ChunkEnvelope(128),
            _graph_cache_max_entries=cap,
        )
    with pytest.raises((TypeError, ValueError), match="entries|capacity"):
        vision.install_qwen3_vl_vision_for_rpu(NS(), _graph_cache_max_entries=cap)


def test_native_cache_capacity_rejects_ninth_signature_before_graph_construction(tmp_path):
    """Execute the real native get_or_create body with no SDK or device code."""
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/graph/graph_infra.cpp").read_text()
    start = source.index("RpuKernelGraph& RpuGraphCache::get_or_create(")
    body = source[start:source.index("\n}", start) + 2]
    program = r'''
#include <cassert>
#include <map>
#include <memory>
#include <stdexcept>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error("capacity"); } while (0)
using GraphSignature = int;
struct RpuKernelGraph {};
int constructions = 0;
std::shared_ptr<RpuKernelGraph> make_registered_rpu_kernel_graph(int, bool) {
    ++constructions;
    return std::make_shared<RpuKernelGraph>();
}
struct RpuGraphCacheEntry { std::shared_ptr<RpuKernelGraph> graph; GraphSignature signature; };
struct RpuGraphCache {
    size_t max_entries_ = 8;
    int runtime_policy_ = 0;
    bool require_single_segment_ = false;
    std::map<GraphSignature, RpuGraphCacheEntry> entries_;
    void add_to_layered_index(const GraphSignature&) {}
    RpuKernelGraph& get_or_create(const GraphSignature& sig);
};
__REAL_BODY__
int main() {
    RpuGraphCache caches[3];
    for (auto& cache : caches) {
        for (int i=0; i<8; ++i) cache.get_or_create(i);
        const int before = constructions;
        for (int i=0; i<8; ++i) cache.get_or_create(i);
        assert(constructions == before);
        bool refused = false;
        try { cache.get_or_create(8); } catch (const std::runtime_error&) { refused = true; }
        assert(refused && constructions == before && cache.entries_.size() == 8);
        cache.entries_.erase(0); // explicit retirement admits a new working-set member
        cache.get_or_create(8);
        assert(constructions == before+1 && cache.entries_.size() == 8);
    }
}
'''.replace("__REAL_BODY__", body)
    cpp = tmp_path / "cache_capacity.cpp"
    cpp.write_text(program)
    binary = tmp_path / "cache_capacity"
    subprocess.run([compiler, "-std=c++17", str(cpp), "-o", str(binary)],
                   check=True, capture_output=True, text=True, timeout=30)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)
