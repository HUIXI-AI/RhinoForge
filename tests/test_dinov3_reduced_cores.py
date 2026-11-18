"""DINOv3 ViT-B (and the InternVLA DINOv2-S towers) at a cold 4/6/8-core budget: host contracts only.

The native topology methods are compiled from production source with small
FMB doubles; the adapter install uses explicit native-op doubles; the weight
layout check runs the real conversion on CPU and recombines per-core slices of
the actual swizzled bytes. None of this is a board run or numerical evidence
for the RPU kernels.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.common import config as COMMON  # noqa: E402
from rpu_backend.adapters import dinov3  # noqa: E402
from rpu_backend.api.cache import RPUCache  # noqa: E402
from rpu_backend.api.errors import RPUBackendError  # noqa: E402
from rpu_backend.tests.acceptance.test_dinov3_install_transaction import (  # noqa: E402,F401
    _model, native_install,
)

NATIVE = ROOT / "src/fused/rpu_dinov3_vision_model.cpp"
TOML = ROOT / "examples/configs/dinov3/vit_b/fp16.toml"


def _block(source: str, signature: str) -> str:
    """Brace-matched body that follows ``signature``."""
    begin = source.index("{", source.index(signature))
    depth, position = 1, begin + 1
    while depth:
        depth += (source[position] == "{") - (source[position] == "}")
        position += 1
    return source[begin:position]


def _vit_b(**overrides):
    values = dict(model_type="dinov3_vit", use_gated_mlp=False, hidden_size=768,
                  intermediate_size=3072, num_attention_heads=12, num_hidden_layers=12,
                  num_register_tokens=4, image_size=224, patch_size=16)
    values.update(overrides)
    return SimpleNamespace(**values)


# --------------------------------------------------------------------------- native


def test_native_topology_admits_only_vit_b_and_dinov2_s_at_reduced_budgets(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required to compile the production topology methods")
    source = NATIVE.read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    weights = _block(source, "void set_weights(")
    start = weights.index("        TORCH_CHECK(num_heads % num_cores() == 0,")
    admission = weights[start:weights.index("set_num_layers(N);", start) + len("set_num_layers(N);")]
    physical = re.search(r"constexpr int DINO_PHYSICAL_KV_CORES = (\d+);", source).group(1)
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "execution_topology.h"
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
using v3::DecoderExecutionTopology;
using v3::decoder_mlp_intermediate_size;
using v3::resolve_decoder_execution_topology;
constexpr int DINO_PHYSICAL_KV_CORES = __PHYSICAL__;
struct FusedModelBase {
    struct Impl {
        int num_cores_ = 8, attn_tp_ = 8, mlp_tp_ = 8;
        int64_t num_q_heads_ = 0, num_kv_heads_ = 0, head_dim_ = 0;
        int64_t hidden_size_ = 0, intermediate_size_ = 0, num_layers_ = 0;
        bool valid_ = false;
    } impl;
    Impl* pimpl_ = &impl;
    int64_t last = 0;
    virtual ~FusedModelBase() = default;
    int num_cores() const { return impl.num_cores_; }
    int attn_tp() const { return impl.attn_tp_; }
    int mlp_tp() const { return impl.mlp_tp_; }
    int64_t num_layers() const { return impl.num_layers_; }
    int64_t num_q_heads() const { return impl.num_q_heads_; }
    int64_t intermediate_size() const { return impl.intermediate_size_; }
    int64_t get_last_resolved_chunk_size() const { return last; }
    void set_num_layers(int64_t n) { impl.num_layers_ = n; }
    void set_execution_core_count(int num_cores) __BASE_SETTER__
    virtual DecoderExecutionTopology resolve_model_execution_topology(
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size) const __BASE_RESOLVE__
    virtual int64_t resolve_model_mlp_intermediate_size(int64_t logical_size) const __BASE_MLP__
    void set_model_params(int64_t num_q_heads, int64_t num_kv_heads,
                          int64_t head_dim, int64_t hidden_size,
                          int64_t intermediate_size) __BASE_PARAMS__
};
struct DINOv3VisionModel : FusedModelBase {
    std::vector<int> layer_weights_;
    void set_execution_cores(int64_t cores) __CORES__
    std::vector<int64_t> execution_topology() const __TOPOLOGY__
    DecoderExecutionTopology resolve_model_execution_topology(
            int64_t nq, int64_t nkv, int64_t hd, int64_t h,
            int64_t intermediate) const override __RESOLVE__
    void install(int64_t num_heads, int64_t head_dim, int64_t hidden_size,
                 int64_t intermediate_size) {
        const int64_t N = 12;
__ADMISSION__
        layer_weights_.assign(N, 0);
    }
};
template <class F> void reject(F body) {
    bool rejected = false;
    try { body(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
int main() {
    // Default eight: unchanged padded 16-head layout and legacy base resolver.
    DINOv3VisionModel eight;
    eight.install(16, 64, 768, 3072);
    assert((eight.execution_topology() == std::vector<int64_t>{8, 8, 8, 8, 16, 3072}));
    reject([] { DINOv3VisionModel m; m.install(12, 64, 768, 3072); });  // unpadded at eight
    reject([&] { eight.set_execution_cores(4); });                       // cold only
    for (int cores : {4, 6}) {
        DINOv3VisionModel m;
        m.set_execution_cores(cores);
        assert(m.num_cores() == cores);
        reject([&] { m.execution_topology(); });                         // no weights yet
        // DINOv2-S is admitted only in its per-budget head layout: 6 unpadded
        // heads on six cores, 8 padded heads on four.
        const int64_t other_small_heads = cores == 6 ? 8 : 6;
        for (auto bad : std::vector<std::vector<int64_t>>{
                 {16, 64, 1024, 3072}, {12, 64, 768, 1536}, {6, 64, 384, 3072},
                 {other_small_heads, 64, 384, 1536}, {12, 64, 384, 1536},
                 {16, 64, 1024, 4096}, {12, 80, 960, 3072}}) {
            reject([&] { m.install(bad[0], bad[1], bad[2], bad[3]); });
        }
        assert(m.num_q_heads() == 0);
        m.install(12, 64, 768, 3072);
        assert((m.execution_topology() ==
                std::vector<int64_t>{cores, cores, cores, 8, 12, 3072}));
        reject([&] { m.set_execution_cores(8); });
        DINOv3VisionModel small;
        small.set_execution_cores(cores);
        const int64_t small_heads = cores == 6 ? 6 : 8;
        small.install(small_heads, 64, 384, 1536);
        assert((small.execution_topology() ==
                std::vector<int64_t>{cores, cores, cores, 8, small_heads, 1536}));
    }
    for (int bad : {0, 1, 2, 3, 5, 7, 9}) {
        reject([bad] { DINOv3VisionModel m; m.set_execution_cores(bad); });
    }
    DINOv3VisionModel busy;
    busy.last = 208;                                                     // planned already
    reject([&] { busy.set_execution_cores(4); });
    assert(busy.num_cores() == 8);
}
'''
    for marker, value in (
        ("__PHYSICAL__", physical),
        ("__BASE_SETTER__", _block(base, "void FusedModelBase::set_execution_core_count(")),
        ("__BASE_RESOLVE__", _block(base, "DecoderExecutionTopology FusedModelBase::resolve_model_execution_topology(")),
        ("__BASE_MLP__", _block(base, "int64_t FusedModelBase::resolve_model_mlp_intermediate_size(")),
        ("__BASE_PARAMS__", _block(base, "void FusedModelBase::set_model_params(")),
        ("__CORES__", _block(source, "void set_execution_cores(int64_t cores)")),
        ("__TOPOLOGY__", _block(source, "std::vector<int64_t> execution_topology() const")),
        ("__RESOLVE__", _block(source, "DecoderExecutionTopology resolve_model_execution_topology(")),
        ("__ADMISSION__", admission),
    ):
        cpp = cpp.replace(marker, value)
    path, binary = tmp_path / "dino_cores.cpp", tmp_path / "dino_cores"
    path.write_text(cpp)
    compiled = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                               "-I", str(ROOT / "src/core"), str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr


def _calls(source: str, name: str) -> list[list[str]]:
    """Top-level arguments of every call to ``name`` (comments removed)."""
    clean = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", source, flags=re.S))
    calls = []
    for match in re.finditer(rf"\b{re.escape(name)}\s*\(", clean):
        depth, position, start = 1, match.end(), match.end()
        args, arg_start = [], start
        while depth:
            char = clean[position]
            if char in "({[":
                depth += 1
            elif char in ")}]":
                depth -= 1
            elif char == "," and depth == 1:
                args.append(clean[arg_start:position].strip())
                arg_start = position + 1
            position += 1
        args.append(clean[arg_start:position - 1].strip())
        calls.append(args)
    return calls


def test_every_multicore_launch_uses_the_owner_budget():
    """A forgotten default-eight launch would leave a 4/6-core Graph budget."""
    source = NATIVE.read_text()
    assert not re.search(r"\bNUM_CORES\b", source)
    assert "constexpr int DINO_PHYSICAL_KV_CORES = 8;" in source
    owner = "num_cores()"
    last_arg = {
        "rpu_launch_memset_spm_multicore": {owner},
        # o/down biases go to core 0 only so the AllReduce adds them once;
        # packed W4 sends them to every core and adds them after the AllReduce.
        "rpu_launch_ddr_broadcast_spm_dma": {owner, "1", "row_bias_cores"},
        "rpu_launch_ddr_scatter_spm_dma": {owner},
        "rpu_launch_spm_scatter_ddr_dma": {owner},
        "rpu_launch_eltwise_binary_1xC_NxC_spm_kernel": {owner},
        "rpu_launch_eltwise_binary_spm_kernel": {owner},
        "rpu_launch_eltwise_unary_spm_kernel": {owner},
        "rpu_launch_layernorm_selected_spm_kernel": {owner},
        "rpu_launch_rope_2d_ddr_kernel": {owner},
        "rpu_launch_v_transpose_spm": {owner},
        "rpu_launch_sdpa_by_mha_spm": {owner},
        "fmb_ring_all_reduce_route_selector": {owner},
    }
    for name, allowed in last_arg.items():
        calls = _calls(source, name)
        assert calls, name
        for args in calls:
            assert args[-1] in allowed, (name, args)
    for args in _calls(source, "rpu_launch_all_reduce_sum_residual_kernel"):
        assert args[-2:] == [owner, owner], args
    for name in ("rpu_launch_linear_spm_to_spm_kernel", "rpu_launch_linear_spm_to_spm_acc16_kernel"):
        for args in _calls(source, name):
            assert args[7] == owner, args
    for args in _calls(source, "rpu_launch_insert_kvcache_spm_unified_with_plan"):
        assert args[6] == owner, args
    sdpa = _calls(source, "rpu_launch_sdpa_spm_unified_kernel_v2")
    assert len(sdpa) == 2
    for args in sdpa:
        # Real cores dispatch; the cache ABI keeps eight virtual (stripe) cores.
        assert args[13:15] == [owner, "DINO_PHYSICAL_KV_CORES"], args
    shared = _calls(source, "append_fmb_shared_runtime_routes")
    assert [args[-2:] for args in shared] == [[owner, "mlp_tp()"]]


def test_core_count_ops_are_declared_and_registered():
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    registrations = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    source = NATIVE.read_text()
    assert "void rpu_dinov3_vision_set_execution_core_count(int64_t handle, int64_t cores);" in decls
    assert "std::vector<int64_t> rpu_dinov3_vision_get_execution_topology(int64_t handle);" in decls
    for name in ("set_execution_core_count", "get_execution_topology"):
        assert f'm.def("dinov3_vision_{name}",\n        TORCH_FN(rpu_dinov3_vision_{name}));' in registrations
        assert f"rpu_dinov3_vision_{name}(int64_t handle" in source


# ------------------------------------------------------------------- adapter admission


@pytest.mark.parametrize("cores", [4, 6, 8])
def test_adapter_admits_vit_b_core_budget(cores):
    model = _model()
    adapter = dinov3.DINOv3Adapter(model, rpu_execution={"model": {"num_cores": cores}})
    assert adapter._rpu_execution["model"] == {"num_cores": cores}
    assert dinov3._dinov3_execution_cores(model) == cores
    assert dinov3._padded_num_heads(12, cores) == (16 if cores == 8 else 12)


def test_adapter_default_and_explicit_eight_keep_the_padded_layout():
    assert dinov3._dinov3_execution_cores(_model()) == 8
    assert dinov3._padded_num_heads(12) == 16
    # Eight admits any registered geometry; the profile preflight decides support.
    assert dinov3._check_execution_cores(_vit_b(hidden_size=1024, num_attention_heads=16), {}) == 8


@pytest.mark.parametrize("cores", [4, 6])
@pytest.mark.parametrize("change", [
    dict(intermediate_size=1536), dict(num_hidden_layers=24),
    dict(hidden_size=1024, num_attention_heads=16), dict(use_gated_mlp=True),
    dict(image_size=256),
])
def test_reduced_budget_rejects_other_geometry_before_mutation(cores, change):
    with pytest.raises(ValueError, match="admits only exact ViT-B"):
        dinov3._check_execution_cores(_vit_b(**change), {"model": {"num_cores": cores}})


@pytest.mark.parametrize("value", [5, 0, True, "6"])
def test_adapter_rejects_invalid_core_budget(value):
    with pytest.raises((TypeError, ValueError), match="num_cores"):
        dinov3.DINOv3Adapter(_model(), rpu_execution={"model": {"num_cores": value}})


def test_core_budget_is_cold_once_weights_are_swizzled():
    model = _model()
    adapter = dinov3.DINOv3Adapter(model, rpu_execution={"model": {"num_cores": 4}})
    adapter.validate_execution_reconfigure({"model": {"num_cores": 6}})  # still cold
    vars(model).update(_rpu_swizzle_started=True, _rpu_vision_num_cores=4)
    adapter.validate_execution_reconfigure({"model": {"num_cores": 4}})
    with pytest.raises(ValueError, match="cold-only"):
        adapter.validate_execution_reconfigure({"model": {"num_cores": 6}})
    with pytest.raises(ValueError, match="cold-only"):
        adapter.validate_execution_reconfigure({})


def test_reduced_budget_refuses_old_native_binary_before_swizzle(monkeypatch):
    model = _model()
    adapter = dinov3.DINOv3Adapter(model, rpu_execution={"model": {"num_cores": 6}})
    monkeypatch.setattr(dinov3, "torch", SimpleNamespace(ops=SimpleNamespace(rpu=SimpleNamespace())))
    with pytest.raises(RPUBackendError, match="rebuild"):
        adapter._to_rpu_locked()
    assert "_rpu_swizzle_started" not in vars(model)


# ------------------------------------------------------------------- native install


@pytest.fixture
def reduced_install(native_install, monkeypatch):
    state = native_install
    state["topology"] = None
    state["kv_args"] = []

    def set_cores(handle, cores):
        state["events"].append(("set_cores", handle, cores))

    def topology(handle):
        if state["topology"] is not None:
            return state["topology"]
        cores = next(event[2] for event in state["events"] if event[:2] == ("set_cores", handle))
        return [cores, cores, cores, 8, 12, 3072]

    class GraphCache:
        def __init__(self, runtime_policy=None):
            self.runtime_policy = runtime_policy
            self.cleared = False
            state["graph_caches"].append(self)

        def clear(self):
            self.cleared = True

        def cache_invariant_ok(self):
            return self.cleared

    def cache(*args):
        state["kv_args"].append(args)
        return SimpleNamespace(max_seq_len=args[1])

    import rpu_backend
    monkeypatch.setattr(rpu_backend.graph, "GraphCache", GraphCache)
    monkeypatch.setattr(dinov3, "_make_dummy_dinov3_kv_caches", cache)
    monkeypatch.setattr(dinov3.torch.ops.rpu, "dinov3_vision_set_execution_core_count",
                        set_cores, raising=False)
    monkeypatch.setattr(dinov3.torch.ops.rpu, "dinov3_vision_get_execution_topology",
                        topology, raising=False)
    return state


def _finalize(model):
    finalizer = getattr(model, "_rpu_dinov3_handle_finalizer", None)
    if finalizer is not None and finalizer.alive:
        finalizer()


@pytest.mark.parametrize("cores", [4, 6])
def test_installer_binds_topology_before_weights(reduced_install, cores):
    model = _model()
    model._rpu_vision_num_cores = cores
    try:
        dinov3.install_dinov3_vision_for_rpu(model)
        events = [event for event in reduced_install["events"] if event[0] != "keepalive"]
        assert events[:3] == [("create", 1), ("set_cores", 1, cores), ("set_weights", 1)]
        assert model._rpu_dinov3_num_heads_padded == 12
        assert reduced_install["kv_args"] == [(1, 256, 12, 64, cores)]
        policy = model._rpu_dinov3_graph_cache.runtime_policy
        assert policy is not None and policy.execution_core_count == cores
    finally:
        _finalize(model)


def test_installer_keeps_eight_core_calls(reduced_install):
    model = _model()
    try:
        dinov3.install_dinov3_vision_for_rpu(model)
        assert not any(event[0] == "set_cores" for event in reduced_install["events"])
        assert model._rpu_dinov3_num_heads_padded == 16
        assert reduced_install["kv_args"] == [(1, 256, 16, 64, 8)]
        assert model._rpu_dinov3_graph_cache.runtime_policy is None
    finally:
        _finalize(model)


def test_installer_rolls_back_on_native_topology_mismatch(reduced_install):
    model = _model()
    model._rpu_vision_num_cores = 6
    reduced_install["topology"] = [6, 4, 6, 8, 12, 3072]
    with pytest.raises(RuntimeError, match="native topology mismatch"):
        dinov3.install_dinov3_vision_for_rpu(model)
    assert reduced_install["alive"] == set()
    assert "_rpu_dinov3_handle" not in vars(model)


@pytest.mark.parametrize("cores, vx", [(4, 3), (6, 2), (8, 2)])
def test_cache_slots_match_the_eight_stripe_sdpa_abi(monkeypatch, cores, vx):
    seen = {}
    monkeypatch.setattr(dinov3, "RPUCache", lambda **kwargs: seen.update(kwargs) or kwargs)
    heads = dinov3._padded_num_heads(12, cores)
    dinov3._make_dummy_dinov3_kv_caches(12, 256, heads, 64, cores)
    assert seen["attn_tp"] == cores
    cache = RPUCache(num_layers=1, batch_size=1, max_seq_len=256, num_kv_heads=heads,
                     head_dim=64, device="cpu", attn_tp=seen["attn_tp"])
    # Eight chunk lanes and one head-vector block per local head.
    assert (cache.physical_kv_cores, cache.nKVHeadChunk, cache.nKVHeadVx) == (8, 8, vx)
    assert heads // cores == vx


# ---------------------------------------------------------- CPU swizzle regression


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(768, 768)
        self.k_proj = nn.Linear(768, 768, bias=False)
        self.v_proj = nn.Linear(768, 768)
        self.o_proj = nn.Linear(768, 768)


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(768, 3072)
        self.down_proj = nn.Linear(3072, 768)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _Attention()
        self.mlp = _Mlp()


class _VitB(nn.Module):
    """One real ViT-B block; the config describes the admitted 12-layer profile."""

    def __init__(self, cores):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.patch_embeddings = nn.Conv2d(3, 768, 16, 16)
        self.layer = nn.ModuleList([_Layer()])
        self.config = _vit_b()
        self._rpu_execution = {"model": {"num_cores": cores}}


def _core_rows(weight, cores, core):
    """Column partition: rows owned by ``core`` read from the swizzled bytes."""
    rows, cols = weight.shape
    view = weight.view(rows // 16 // cores, cols // 16, cores, 16, 16)[:, :, core]
    return view.permute(0, 2, 1, 3).reshape(rows // cores, cols)


def _core_cols(weight, cores, core):
    """Row partition: K columns owned by ``core`` read from the swizzled bytes."""
    rows, cols = weight.shape
    view = weight.view(rows // 16, cols // 16 // cores, cores, 16, 16)[:, :, core]
    return view.permute(0, 2, 1, 3).reshape(rows, cols // cores)


def _attention(q, k, v, heads):
    tokens = q.shape[0]
    q, k, v = (t.view(tokens, heads, 64).transpose(0, 1) for t in (q, k, v))
    scores = (q @ k.transpose(1, 2)) / math.sqrt(64)
    return (scores.softmax(-1) @ v).transpose(0, 1).reshape(tokens, heads * 64)


@pytest.mark.parametrize("cores", [4, 6, 8])
def test_swizzled_core_slices_recombine_to_the_unsplit_block(cores):
    torch.manual_seed(cores)
    model = _VitB(cores).half()
    block = model.layer[0]
    original = {name: param.detach().double().clone() for name, param in block.named_parameters()}
    dinov3._convert_dinov3_weights_for_rpu(model)
    assert model._rpu_vision_num_cores == cores

    attn, mlp = block.attention, block.mlp
    heads = dinov3._padded_num_heads(12, cores)
    assert attn.q_proj.weight.shape == (heads * 64, 768)
    for linear, partition in ((attn.q_proj, 1), (attn.k_proj, 1), (attn.v_proj, 1),
                              (attn.o_proj, 0), (mlp.up_proj, 1), (mlp.down_proj, 0)):
        assert (linear._rpu_linear_partition, linear._rpu_linear_num_cores) == (partition, cores)

    x = torch.randn(21, 768, dtype=torch.float64)
    q = x @ original["attention.q_proj.weight"].T + original["attention.q_proj.bias"]
    k = x @ original["attention.k_proj.weight"].T
    v = x @ original["attention.v_proj.weight"].T + original["attention.v_proj.bias"]
    expected_attn = (_attention(q, k, v, 12) @ original["attention.o_proj.weight"].T
                     + original["attention.o_proj.bias"])
    expected_mlp = (F.gelu(x @ original["mlp.up_proj.weight"].T + original["mlp.up_proj.bias"])
                    @ original["mlp.down_proj.weight"].T + original["mlp.down_proj.bias"])

    local_heads = heads // cores
    local_q = local_heads * 64
    local_inter = 3072 // cores
    attn_sum = torch.zeros(21, 768, dtype=torch.float64)
    mlp_sum = torch.zeros(21, 768, dtype=torch.float64)
    q_bias, v_bias = attn.q_proj.bias.double(), attn.v_proj.bias.double()
    for core in range(cores):
        own = slice(core * local_q, (core + 1) * local_q)
        q_c = x @ _core_rows(attn.q_proj.weight, cores, core).double().T + q_bias[own]
        k_c = x @ _core_rows(attn.k_proj.weight, cores, core).double().T
        v_c = x @ _core_rows(attn.v_proj.weight, cores, core).double().T + v_bias[own]
        local = _attention(q_c, k_c, v_c, local_heads)
        attn_sum += local @ _core_cols(attn.o_proj.weight, cores, core).double().T
        inner = slice(core * local_inter, (core + 1) * local_inter)
        up = (x @ _core_rows(mlp.up_proj.weight, cores, core).double().T
              + mlp.up_proj.bias.double()[inner])
        mlp_sum += F.gelu(up) @ _core_cols(mlp.down_proj.weight, cores, core).double().T
    # Row-partition biases are added once (core 0) by the AllReduce sum.
    attn_sum += attn.o_proj.bias.double()
    mlp_sum += mlp.down_proj.bias.double()
    torch.testing.assert_close(attn_sum, expected_attn, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(mlp_sum, expected_mlp, rtol=1e-9, atol=1e-9)


# ------------------------------------------------------------------ example config


def _with_cores(cores, tmp_path):
    path = tmp_path / "dinov3.toml"
    path.write_text(TOML.read_text(encoding="utf-8")
                    + f"\n[rpu_execution.model]\nnum_cores = {cores}\n", encoding="utf-8")
    return path


def test_example_declares_every_budget_supported():
    assert COMMON.core_support(COMMON.load_config(TOML)) == {4: None, 6: None, 8: None}


@pytest.mark.parametrize("cores", [4, 6, 8])
def test_example_accepts_core_budget(tmp_path, cores):
    config = COMMON.load_config(_with_cores(cores, tmp_path))
    assert config["rpu_execution"]["model"] == {"num_cores": cores}


@pytest.mark.parametrize("setting", ["num_cores = 5", "num_cores = true", "attn_tp = 4"])
def test_example_rejects_invalid_core_setting(tmp_path, setting):
    path = tmp_path / "bad.toml"
    path.write_text(TOML.read_text(encoding="utf-8") + f"\n[rpu_execution.model]\n{setting}\n",
                    encoding="utf-8")
    with pytest.raises(COMMON.ConfigError, match="num_cores|unknown field"):
        COMMON.load_config(path)


@pytest.mark.parametrize("cores", [4, 6])
def test_example_reduced_budget_requires_profile_admission(cores):
    import copy
    profile = copy.deepcopy(COMMON.planner_profiles()["dinov3-vit-b.fp16.vision"])
    assert profile["input_envelope"]["limits"]["allowed_num_cores"] == [4, 6, 8]
    assert profile["hw_status"] == "IMPLEMENTED_UNVERIFIED_HW"
    COMMON._validate_execution({"model": {"num_cores": cores}}, profile=profile)
    profile["input_envelope"]["limits"].pop("allowed_num_cores")
    with pytest.raises(COMMON.ConfigError, match="registered ViT-B vision profile"):
        COMMON._validate_execution({"model": {"num_cores": cores}}, profile=profile)


