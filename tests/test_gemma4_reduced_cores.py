"""Gemma4-E4B at 4/6/8 cores: native lane budgets, padded math and admission.

The native probe compiles the real ``build_layer_subgraph`` body and the real
launcher declarations (with their default core counts) against host recorders.
It checks the emitted compute domains only; kernel numerics stay a board check.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters import gemma4
from rpu_backend.adapters.gemma4 import runtime, topology as g4_topology
from rpu_backend.runtime.weights import transform_linear_weight
from rpu_backend.tests.acceptance import test_gemma4_install_transaction as installed
from rpu_backend.tests.acceptance.test_gemma4_install_transaction import install_runtime  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "src/fused/rpu_gemma4_model.cpp"
DECLS = ROOT / "src/core/rpu_kernel_decls.h"
LAYOUTS = {8: (2, 8, 8, 10240), 6: (2, 6, 4, 10272), 4: (2, 4, 4, 10240)}


# ----------------------------------------------------------------- topology
@pytest.mark.parametrize("cores", (8, 6, 4))
def test_fixed_layout_and_native_readback(cores):
    attention, mlp, ple, physical = LAYOUTS[cores]
    topology = g4_topology.resolve_gemma4_topology(cores)
    assert (topology.attn_tp, topology.mlp_tp, topology.ple_tp,
            topology.physical_intermediate) == (attention, mlp, ple, physical)
    assert topology.native_readback() == (1, cores, attention, mlp, ple, 8, 10240, physical)
    assert topology.physical_intermediate % (16 * mlp) == 0 and 256 % (16 * ple) == 0
    assert topology.mlp_padding == (32 if cores == 6 else 0)


@pytest.mark.parametrize("value", (True, 5, 7, 2, "6", 6.0, None))
def test_core_budget_rejects_everything_but_4_6_8(value):
    with pytest.raises(ValueError, match="4, 6 or 8"):
        g4_topology.resolve_gemma4_topology(value)


def test_reduced_budget_requires_e4b_geometry_and_is_cold_only():
    for override in ({"intermediate_size": 9728}, {"ple_dim": 128}, {"num_kv_heads": 4}):
        with pytest.raises(ValueError, match="E4B"):
            g4_topology.resolve_gemma4_topology(6, **override)
        g4_topology.resolve_gemma4_topology(8, **override)  # eight keeps old behaviour
    g4_topology.require_same_core_budget({}, {"model": {"num_cores": 8}})
    with pytest.raises(ValueError, match="cold-only"):
        g4_topology.require_same_core_budget({"model": {"num_cores": 4}}, {})


# --------------------------------------------- padded MLP / PLE CPU regression
def _lanes(swizzled: torch.Tensor, partition: int, cores: int) -> list[torch.Tensor]:
    """Invert the swizzle into each core's logical shard (rows for col, cols for row)."""
    n, k = swizzled.shape
    if partition == 1:
        whole = (swizzled.reshape(n // 16 // cores, k // 16, cores, 16, 16)
                 .permute(2, 0, 3, 1, 4).reshape(n, k))
        return list(whole.chunk(cores, dim=0))
    whole = (swizzled.reshape(n // 16, k // 16 // cores, cores, 16, 16)
             .permute(0, 3, 2, 1, 4).reshape(n, k))
    return list(whole.chunk(cores, dim=1))


def _gelu(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


@pytest.mark.parametrize("cores", (8, 6, 4))
def test_padded_lane_math_equals_logical_mlp_and_ple(monkeypatch, cores):
    """Actual adapter preparation (pad + swizzle) and per-lane TP equal the HF math."""
    generator = torch.Generator().manual_seed(cores)
    h, inter, ple_dim, rows = 2560, 10240, 256, 3
    names = {}
    prefix = runtime.PFX + "layers.0."

    def weight(*shape):
        return (torch.randn(*shape, generator=generator) / shape[-1] ** 0.5)

    for name, shape in {
        "self_attn.q_proj.weight": (8 * 256, h), "self_attn.k_proj.weight": (2 * 256, h),
        "self_attn.v_proj.weight": (2 * 256, h), "self_attn.o_proj.weight": (h, 8 * 256),
        "mlp.gate_proj.weight": (inter, h), "mlp.up_proj.weight": (inter, h),
        "mlp.down_proj.weight": (h, inter),
        "per_layer_input_gate.weight": (ple_dim, h), "per_layer_projection.weight": (h, ple_dim),
    }.items():
        names[prefix + name] = weight(*shape)
    for name, size in (("self_attn.q_norm.weight", 256), ("self_attn.k_norm.weight", 256),
                       ("input_layernorm.weight", h), ("post_attention_layernorm.weight", h),
                       ("pre_feedforward_layernorm.weight", h), ("post_feedforward_layernorm.weight", h),
                       ("post_per_layer_input_norm.weight", h)):
        names[prefix + name] = torch.ones(size)
    names[prefix + "layer_scalar"] = torch.ones(1)
    # Host stand-in for the RPU upload; the layout and padding are the real ones.
    monkeypatch.setattr(runtime, "_rpu", lambda w: w.contiguous())
    topology = g4_topology.resolve_gemma4_topology(cores)
    geom = gemma4.layer_geometry(installed._text_config())[:1]
    assert not geom[0].is_global and geom[0].kv_source_layer == 0
    lists, ple = runtime.swizzle_decoder_and_ple(lambda name: names[name].float(), geom,
                                                 topology=topology)
    gate, up, down = lists["gate"][0], lists["up"][0], lists["down"][0]
    assert gate.shape == up.shape == (topology.physical_intermediate, h)
    assert down.shape == (h, topology.physical_intermediate)

    x = torch.randn(rows, h, generator=generator).half()
    logical = {key: names[prefix + f"mlp.{key}_proj.weight"].half() for key in ("gate", "up", "down")}
    reference = (_gelu(x.double() @ logical["gate"].double().T)
                 * (x.double() @ logical["up"].double().T)) @ logical["down"].double().T

    # MLP lanes: col-partitioned gate/up, row-partitioned down, ring sum.
    partials, padded_activations = [], []
    for g, u, d in zip(_lanes(gate, 1, topology.mlp_tp), _lanes(up, 1, topology.mlp_tp),
                       _lanes(down, 0, topology.mlp_tp)):
        activation = _gelu(x.double() @ g.double().T) * (x.double() @ u.double().T)
        padded_activations.append(activation)
        partials.append(activation @ d.double().T)
    tail = torch.cat(padded_activations, dim=1)[:, inter:]
    assert tail.numel() == topology.mlp_padding * rows and torch.count_nonzero(tail) == 0
    torch.testing.assert_close(sum(partials), reference, rtol=1e-12, atol=1e-12)

    # PLE lanes: CPU side input in the installed lane layout, gate*side, proj,
    # then a ring with every consumer lane (missing producers were cleared).
    vocab, ids = 5, torch.tensor([[1, 4, 2]])
    embed = torch.randn(vocab, h, generator=generator)
    per_layer = torch.randn(vocab, ple_dim, generator=generator)
    projection = torch.randn(ple_dim, h, generator=generator) / h ** 0.5
    side_inputs = dict(embed_weight=embed, embed_per_layer_weight=per_layer,
                       model_projection_weight=projection, projection_norm_weight=torch.ones(ple_dim),
                       num_layers=1, hidden_size=h, ple_dim=ple_dim, eps=1e-6)
    lane_major = runtime.compute_side_input(ids, **side_inputs, num_cores=topology.ple_tp)
    full = runtime.compute_side_input(ids, **side_inputs, num_cores=1)[:, 0]
    assert lane_major.shape == (1, topology.ple_tp, rows, ple_dim // topology.ple_tp)
    assert torch.equal(torch.cat(list(lane_major[0]), dim=-1), full[0])
    gate_w = names[prefix + "per_layer_input_gate.weight"].half().double()
    proj_w = names[prefix + "per_layer_projection.weight"].half().double()
    ple_reference = (_gelu(x.double() @ gate_w.T) * full[0].double()) @ proj_w.T
    ple_partials = [
        (_gelu(x.double() @ g.double().T) * lane_major[0, lane].double()) @ p.double().T
        for lane, (g, p) in enumerate(zip(_lanes(ple["ple_gate"][0], 1, topology.ple_tp),
                                          _lanes(ple["ple_proj"][0], 0, topology.ple_tp)))]
    cleared = [torch.zeros_like(ple_reference)] * (cores - topology.ple_tp)
    torch.testing.assert_close(sum(ple_partials + cleared), ple_reference, rtol=1e-12, atol=1e-12)

    # Attention keeps its two-lane payload at every budget.
    assert torch.equal(lists["q"][0], transform_linear_weight(
        names[prefix + "self_attn.q_proj.weight"].half(), 1, num_cores=2))


def test_eight_core_preparation_keeps_the_original_payload(monkeypatch):
    monkeypatch.setattr(runtime, "_rpu", lambda w: w.contiguous())
    generator = torch.Generator().manual_seed(0)
    value = torch.randn(10240, 2560, generator=generator)
    padded = g4_topology.pad_mlp_projection(value, role="gate", physical=10240)
    assert padded is value
    assert torch.equal(runtime._swz(value, 1, 8), transform_linear_weight(value.half(), 1, num_cores=8))
    down = g4_topology.pad_mlp_projection(torch.ones(4, 10240), role="down", physical=10272)
    assert down.shape == (4, 10272) and not down[:, 10240:].any() and down[:, :10240].all()
    with pytest.raises(ValueError, match="no wider"):
        g4_topology.pad_mlp_projection(torch.ones(10272, 4), role="up", physical=10240)


# ------------------------------------------------------ native lane probe
def _body(source: str, signature: str) -> str:
    start = source.index(signature)
    position = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[position] == "{") - (source[position] == "}")
        position += 1
    return source[start:position]


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _split_top(text: str) -> list[str]:
    parts, depth, current = [], 0, []
    for char in text:
        depth += char in "(<{" and 1 or (char in ")>}" and -1 or 0)
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if "".join(current).strip():
        parts.append("".join(current))
    return parts


def _declaration(decls: str, name: str, needle: str) -> tuple[str, list[tuple[str, str]]]:
    """The real declaration (defaults included) and its (type, name) parameters."""
    decls = _strip_comments(decls)
    for match in re.finditer(rf"\bvoid {name}\(", decls):
        end = decls.index(";", match.start())
        text = decls[match.start():end + 1]
        if needle in text:
            inner = text[text.index("(") + 1:text.rindex(")")]
            params = []
            for part in _split_top(inner):
                declarator = _split_top(part.replace("=", ",", 1))[0].strip()
                pname = re.search(r"([A-Za-z_]\w*)\s*$", declarator).group(1)
                params.append((declarator[:-len(pname)].strip(), pname))
            return text, params
    raise AssertionError(f"declaration {name} with {needle!r} not found")


_LAUNCHERS = (
    ("rpu_launch_rmsnorm_spm_kernel", "RpuRmsNormSpmRoute"),
    ("rpu_launch_linear_spm_to_spm_acc16_kernel", "int partition"),
    ("rpu_launch_rope_spm_kernel", "uint32_t y_spm_addr"),
    ("rpu_launch_eltwise_unary_spm_kernel", "GeluMode gelu_mode"),
    ("rpu_launch_eltwise_binary_spm_kernel", "c10::Half alpha"),
    ("rpu_launch_eltwise_binary_scalar_spm_kernel", "scalar_val"),
    ("rpu_launch_eltwise_mul_spm_kernel", "num_elements"),
    ("rpu_launch_eltwise_add_spm_kernel", "num_elements"),
    ("rpu_launch_eltwise_sub_spm_kernel", "num_elements"),
    ("rpu_prepare_ring_all_reduce_input", "input_num_cores"),
    ("rpu_launch_all_reduce_sum_residual_kernel", "int output_num_cores = 8)"),
    ("rpu_launch_ddr_scatter_spm_dma_mutable", "const uint64_t* live_src_base"),
    ("rpu_launch_sdpa_spm_dispatch", "virtual_num_cores"),
    ("rpu_launch_insert_kvcache_spm_unified_with_plan", "KvInsertSegmentPlan"),
)


def _recorder(name: str, params: list[tuple[str, str]]) -> str:
    """A definition matching the declaration that records every scalar argument."""
    values = []
    for index, (kind, _pname) in enumerate(params):
        variable = f"p{index}"
        bare = kind.replace("const", "").strip()
        if bare in ("uint32_t", "int64_t", "int", "uint16_t", "bool") or bare in (
                "RpuRmsNormSpmRoute", "ValuOpType", "GeluMode", "RpuUnaryPrecision", "SdpaKernelType"):
            values.append(f"static_cast<int64_t>({variable})")
        elif bare == "double":
            values.append(f"static_cast<int64_t>({variable} * 1e9)")
        elif bare == "c10::Half":
            values.append(f"static_cast<int64_t>({variable}.x)")
        elif bare == "std::optional<double>":
            values.append(f"({variable} ? static_cast<int64_t>(*{variable} * 1e9) : -1)")
        else:
            values.append("0")
    signature = ", ".join(f"{kind} p{index}" for index, (kind, _) in enumerate(params))
    names = ", ".join(f'"{pname}"' for _, pname in params)
    return (f"void {name}({signature}) {{ record(\"{name}\", {{{names}}}, "
            f"{{{', '.join(values)}}}); }}\n")


_HARNESS = r'''
#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "core/gemma4_execution_topology.h"
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
#define DWIDTH 2
#define NUM_CORES 8
namespace c10 { struct Half { uint16_t x = 0; Half(double value = 0.0) : x(static_cast<uint16_t>(value * 1024)) {} }; }
namespace at { struct Tensor {
  int64_t blocks = 64, offset = 0;
  template <class T> T* data_ptr() const { return nullptr; }
  int64_t size(int) const { return blocks; }
  Tensor slice(int, int64_t start) const { Tensor t = *this; t.offset += start; return t; }
}; }
enum class RpuRmsNormSpmRoute : int { BASE = 0 };
enum class ValuOpType : int { ADD = 1, SUB = 2, MUL = 3 };
enum class GeluMode : int { NONE = 0, TANH = 1 };
enum class RpuUnaryPrecision : int { BASE = 0 };
enum class SdpaKernelType : int { FLASH_ATTN_SPM = 0 };
struct WallSortingConfig {};
struct KvInsertSegmentPlan {
  struct Segment { int64_t position = 0; };
  int64_t position = 0, rows = 0;
  Segment segment(int) const { return {position}; }
  int64_t logical_rows() const { return rows; }
  int64_t physical_rows() const { return rows; }
  int route() const { return 2; }
};
struct Call { std::string name; std::vector<std::string> keys; std::vector<int64_t> values; };
std::vector<Call> calls;
void record(const char* name, std::vector<std::string> keys, std::vector<int64_t> values) {
  calls.push_back({name, std::move(keys), std::move(values)});
}
__DECLARATIONS__
__RECORDERS__
namespace v3 {
enum class FmbRouteFamily : int { LINEAR, ROPE, KV_INSERT, ALL_REDUCE, MUTABLE_DMA, GRAPH_SCHEDULE };
enum class FmbLinearRouteSelector : int64_t { AUTO_TILE = 1 };
enum class AttentionExecutionPolicy : int { DDR_KV = 1 };
constexpr int64_t KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION = 1;
struct ChunkInfo { int idx = 0; int64_t offset = 0; int64_t len = 0; int64_t kv_seq_len = 0; };
struct FmbRouteManifestEntry { std::vector<int64_t> arguments; int64_t flags = 0; };
KvInsertSegmentPlan current_plan;
KvInsertSegmentPlan rpu_kvinsert_segment_plan_from_route_arguments(
    const std::vector<int64_t>&, int, int64_t, int64_t) { return current_plan; }
KvInsertSegmentPlan rpu_rebase_kvinsert_segment_plan_position(
    const KvInsertSegmentPlan& plan, int64_t position, int, int64_t, int64_t) {
  KvInsertSegmentPlan next = plan; next.position = position; return next;
}
std::vector<std::pair<int64_t, std::vector<int64_t>>> consumed;
int64_t fmb_ring_all_reduce_route_selector(int64_t rows, int64_t cols, int cores = 8) {  // real default
  record("fmb_ring_all_reduce_route_selector", {"rows", "cols", "num_cores"}, {rows, cols, cores});
  return 1;
}
__CONSTANTS__
struct Context {
  int64_t position = 0;
  bool input_in_spm = false, output_to_spm = false;
  std::vector<at::Tensor>* k_caches = nullptr;
  std::vector<at::Tensor>* v_caches = nullptr;
  FmbRouteManifestEntry kv_route;
  void consume_physical_route(FmbRouteFamily, int64_t site, int64_t, int64_t,
                              std::vector<int64_t> arguments = {}, int64_t = 0) {
    consumed.push_back({site, std::move(arguments)});
  }
  const FmbRouteManifestEntry& find_physical_route(FmbRouteFamily, int64_t, int64_t) const { return kv_route; }
  void consume_physical_attention_route(int64_t site, AttentionExecutionPolicy, int64_t) {
    consumed.push_back({site, {}});
  }
};
struct LayerWeights {
  at::Tensor q_w, k_w, v_w, o_w, q_norm_w, k_norm_w, input_norm_w, post_attn_norm_w,
      pre_ff_norm_w, post_ff_norm_w, gate_w, up_w, down_w, ple_gate_w, ple_proj_w, ple_post_norm_w;
  double layer_scalar = 0.5;
};
struct LayerGeom { int64_t head_dim; int64_t num_kv_heads; int64_t kv_source_layer; bool is_global; bool is_kv_shared; };
struct Probe {
  int cores = 8;
  Context context;
  std::vector<LayerWeights> layer_weights_ = std::vector<LayerWeights>(42);
  std::vector<LayerGeom> geom_;
  std::vector<at::Tensor> k_caches = std::vector<at::Tensor>(42), v_caches = std::vector<at::Tensor>(42);
  at::Tensor cos_sliding_, sin_sliding_, cos_global_, sin_global_;
  double eps_ = 1e-6, qk_scale_ = 1.0;
  int64_t sliding_window_ = 512, ple_dim_ = 256, seq_len_ = 0;
  bool has_ple_ = true, original_is_causal_ = true;
  uint64_t side_live_base_ = 0x1000;
  SdpaKernelType sdpa_kernel_ = SdpaKernelType::FLASH_ATTN_SPM;
  Probe() {
    for (int64_t i = 0; i < 42; ++i) {
      const bool global = (i + 1) % 6 == 0, shared = i >= 24;
      const int64_t source = shared ? (global ? 23 : 22) : i;
      geom_.push_back({global ? 512 : 256, 2, source, global, shared});
    }
    context.k_caches = &k_caches; context.v_caches = &v_caches;
  }
  Context& ctx() { return context; }
  int num_cores() const { return cores; }
  DecoderExecutionTopology topology() const {
    return cores == 8 ? DecoderExecutionTopology{8, 2, 8}
                      : resolve_gemma4_reduced_topology(cores, 8, 2, 512, 2560, 10240);
  }
  int attn_tp() const { return topology().attention_tp; }
  int mlp_tp() const { return topology().mlp_tp; }
  int64_t intermediate_size() const { return gemma4_mlp_physical_size(10240, mlp_tp()); }
  int64_t hidden_size() const { return 2560; }
  int64_t num_q_heads() const { return 8; }
  int64_t num_layers() const { return 42; }
  __PLE_TP__
  uint32_t addr(int, const char*) const { return 0x100; }
  uint32_t layer_addr(int, int, const char*) const { return 0x200; }
  struct Offset { uint32_t value = 0x300; };
  Offset addr_offset(const char*) const { return {}; }
  bool window_mask_dma_for_layer(int, const ChunkInfo&) const { return true; }
  std::pair<int64_t, int64_t> windowed_mask_range(const ChunkInfo& chunk, int64_t) const {
    return {16, chunk.kv_seq_len - 16};
  }
  uint32_t copy_windowed_mask_to_spm(const ChunkInfo&, int64_t, int64_t, bool) const { return 0x400; }
  void emit_layer_input_dma(int, const ChunkInfo&) { record("layer_input_dma", {"num_cores"}, {cores}); }
  void emit_layer_output_dma(int, const ChunkInfo&) { record("layer_output_dma", {}, {}); }
  __BUILD__
};
}  // namespace v3
int main() {
  using namespace v3;
  for (const int budget : {8, 6, 4}) {
    for (const int layer : {0, 5, 24, 41}) {
      for (const auto& chunk : std::vector<ChunkInfo>{{0, 0, 16, 16}, {1, 512, 16, 528}, {0, 0, 1, 600}}) {
        Probe probe; probe.cores = budget; probe.seq_len_ = chunk.offset + chunk.len;
        probe.context.position = chunk.kv_seq_len - chunk.len - chunk.offset;
        current_plan = {probe.context.position + chunk.offset, chunk.len};
        calls.clear(); consumed.clear();
        probe.build_layer_subgraph(layer, chunk);
        std::cout << "CASE " << budget << " " << layer << " " << chunk.idx << " "
                  << chunk.len << " " << chunk.kv_seq_len << "\n";
        for (const auto& call : calls) {
          std::cout << "CALL " << call.name;
          for (size_t i = 0; i < call.keys.size(); ++i)
            std::cout << " " << call.keys[i] << "=" << call.values[i];
          std::cout << "\n";
        }
        for (const auto& route : consumed) {
          std::cout << "ROUTE " << route.first;
          for (const auto value : route.second) std::cout << " " << value;
          std::cout << "\n";
        }
      }
    }
  }
}
'''


def native_trace(root: Path, tmp_path: Path) -> dict[tuple[int, ...], dict]:
    """Compile the owner's layer body at budgets 8/6/4 and parse its launch trace."""
    native = (root / "src/fused/rpu_gemma4_model.cpp").read_text()
    decls = (root / "src/core/rpu_kernel_decls.h").read_text()
    declarations, recorders = [], []
    for name, needle in _LAUNCHERS:
        text, params = _declaration(decls, name, needle)
        declarations.append(text)
        recorders.append(_recorder(name, params))
    constants = "namespace {\n" + native[native.index("// GEMMA4_FIXED_KERNEL_BASIS"):]
    constants = constants[len("namespace {\n"):constants.index("}  // namespace\n")]
    build = _body(native, "void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {")
    build = build.replace(") override {", ") {", 1)
    ple = re.search(r"int ple_tp\(\) const \{[^}]*\}", native)
    source = (_HARNESS.replace("__DECLARATIONS__", "\n".join(declarations))
              .replace("__RECORDERS__", "".join(recorders))
              .replace("__CONSTANTS__", constants)
              .replace("__PLE_TP__", ple.group(0) if ple else "int ple_tp() const { return 8; }")
              .replace("__BUILD__", build))
    cpp, exe = tmp_path / "gemma4_lanes.cpp", tmp_path / "gemma4_lanes"
    cpp.write_text(source)
    compiler = shutil.which("g++") or shutil.which("c++")
    built = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(root / "src"), str(cpp), "-o", str(exe)],
                           text=True, capture_output=True)
    assert built.returncode == 0, built.stderr[-4000:]
    result = subprocess.run([str(exe)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    cases, current = {}, None
    for line in result.stdout.splitlines():
        kind, *rest = line.split()
        if kind == "CASE":
            current = cases.setdefault(tuple(map(int, rest)), {"calls": [], "routes": []})
        elif kind == "CALL":
            current["calls"].append((rest[0], dict(item.split("=") for item in rest[1:])))
        else:
            current["routes"].append(tuple(map(int, rest)))
    return cases


@pytest.fixture(scope="module")
def lane_trace(tmp_path_factory):
    if not (shutil.which("g++") or shutil.which("c++")):
        pytest.skip("C++ compiler unavailable")
    return native_trace(ROOT, tmp_path_factory.mktemp("gemma4_lanes"))


def test_native_layer_body_keeps_every_launch_inside_its_lane_budget(lane_trace):
    ple_prepare = 5712097404687403303
    assert len(lane_trace) == 3 * 4 * 3
    for (budget, layer, _idx, rows, _kv), case in lane_trace.items():
        attention, mlp, ple, physical = LAYOUTS[budget]
        rings, prepares, linear = [], [], {}
        for name, args in case["calls"]:
            values = {key: int(value) for key, value in args.items()}
            for key in ("num_cores", "input_num_cores", "output_num_cores"):
                if key in values:
                    assert 1 <= values[key] <= budget, (budget, name, values)
            if name == "rpu_launch_sdpa_spm_dispatch":
                assert values["num_cores"] == attention and values["virtual_num_cores"] == 8
            elif name in ("rpu_launch_rope_spm_kernel", "rpu_launch_insert_kvcache_spm_unified_with_plan"):
                assert values["num_cores"] == attention
            elif name == "rpu_launch_linear_spm_to_spm_acc16_kernel":
                key = ("ple_gate" if values["N"] == 256 else "ple_proj" if values["K"] == 256
                       else "mlp_up" if values["N"] == physical else "mlp_down" if values["K"] == physical
                       else "attention")
                linear.setdefault(key, set()).add(values["num_cores"])
            elif name == "rpu_launch_ddr_scatter_spm_dma_mutable":
                assert values["num_cores"] == ple and values["elements_per_core"] == rows * 256 // ple
            elif name == "rpu_launch_all_reduce_sum_residual_kernel":
                rings.append((values["input_num_cores"], values["output_num_cores"]))
            elif name == "rpu_prepare_ring_all_reduce_input":
                prepares.append((values["input_num_cores"], values["output_num_cores"]))
            elif name == "rpu_launch_eltwise_unary_spm_kernel":
                lanes = mlp if values["num_elements"] == rows * physical // mlp else ple
                assert values["num_elements"] in (rows * physical // mlp, rows * 256 // ple)
                assert values["num_cores"] == lanes
            elif name == "rpu_launch_eltwise_binary_spm_kernel":
                assert values["num_elements"] == rows * physical // mlp and values["num_cores"] == mlp
            elif name == "rpu_launch_eltwise_mul_spm_kernel":
                assert values["num_elements"] == rows * 256 // ple and values["num_cores"] == ple
            elif name in ("rpu_launch_rmsnorm_spm_kernel", "rpu_launch_eltwise_add_spm_kernel",
                          "rpu_launch_eltwise_sub_spm_kernel", "rpu_launch_eltwise_binary_scalar_spm_kernel",
                          "fmb_ring_all_reduce_route_selector", "layer_input_dma"):
                assert values["num_cores"] == budget, (name, values)
        assert linear == {"attention": {attention}, "mlp_up": {mlp}, "mlp_down": {mlp},
                          "ple_gate": {ple}, "ple_proj": {ple}}
        assert rings == [(attention, budget), (mlp, budget), (ple, budget)]
        assert prepares == [(attention, budget)] + ([(ple, budget)] if ple != budget else [])
        ple_routes = [args for site, *args in case["routes"] if site == ple_prepare]
        assert ple_routes == ([[ple, budget]] if ple != budget else [])
        assert layer in (0, 5, 24, 41)


def test_eight_core_layer_body_keeps_only_the_original_two_and_eight_lanes(lane_trace):
    """Budget 8 keeps attention on 2 lanes and everything else on 8; no PLE ring preparation."""
    for (budget, _layer, _idx, _rows, _kv), case in lane_trace.items():
        if budget != 8:
            continue
        lanes = {int(args[key]) for name, args in case["calls"]
                 for key in ("num_cores", "input_num_cores", "output_num_cores") if key in args
                 and name not in ("rpu_launch_rope_spm_kernel", "rpu_launch_insert_kvcache_spm_unified_with_plan")}
        assert lanes == {2, 8}
        assert all(site != 5712097404687403303 for site, *_ in case["routes"])


# ------------------------------------------------------ adapter admission
def _patch_topology_ops(monkeypatch, state, readback=None):
    def set_cores(handle, cores):
        state["events"].append(("set_cores", handle, cores))

    def get_topology(handle):
        state["events"].append(("get_topology", handle))
        return readback(handle)

    monkeypatch.setattr(torch.ops.rpu, "gemma4_set_execution_core_count", set_cores, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "gemma4_get_execution_topology", get_topology, raising=False)


@pytest.mark.parametrize("cores", (4, 6))
def test_adapter_binds_reduced_budget_before_weights(install_runtime, monkeypatch, cores):
    seen = {}
    monkeypatch.setattr(runtime, "swizzle_decoder_and_ple",
                        lambda raw, geom, **kwargs: seen.update(kwargs) or ({"q": []}, {}))
    expected = g4_topology.resolve_gemma4_topology(cores)
    _patch_topology_ops(monkeypatch, install_runtime, lambda handle: expected.native_readback())
    adapter = installed._adapter(rpu_execution={"model": {"num_cores": cores}})
    model = adapter.to_rpu()
    try:
        events = [event[0] for event in install_runtime["events"]]
        assert events[:4] == ["create", "set_cores", "set_weights", "get_topology"]
        assert install_runtime["events"][1] == ("set_cores", 1, cores)
        assert seen["topology"] == expected == model._rpu_gemma4_topology
        policy = install_runtime["graph_caches"][-1].runtime_policy
        assert policy.execution_core_count == cores
        assert f"owner:gemma4:execution_core_count={cores}" in policy.provenance
        assert model._rpu_execution["model"] == {"num_cores": cores}
        with pytest.raises(ValueError, match="cold-only"):
            adapter.reconfigure_rpu_execution({"model": {"num_cores": 8}})
        adapter.reconfigure_rpu_execution({"prefill": {"chunk_size": 32}})
        assert model._rpu_execution["model"] == {"num_cores": cores}
    finally:
        installed._finalize(model)


def test_adapter_eight_core_install_keeps_the_original_native_calls(install_runtime, monkeypatch):
    _patch_topology_ops(monkeypatch, install_runtime, lambda handle: pytest.fail("no readback at 8"))
    model = installed._adapter(rpu_execution={"model": {"num_cores": 8}}).to_rpu()
    try:
        assert [event[0] for event in install_runtime["events"]][:2] == ["create", "set_weights"]
        policy = install_runtime["graph_caches"][-1].runtime_policy
        assert policy.execution_core_count == 8
        assert not any("execution_core_count" in item for item in policy.provenance)
        assert model._rpu_gemma4_topology == g4_topology.resolve_gemma4_topology(8)
    finally:
        installed._finalize(model)


def test_native_topology_mismatch_cleans_up_before_publication(install_runtime, monkeypatch):
    wrong = g4_topology.resolve_gemma4_topology(4).native_readback()
    _patch_topology_ops(monkeypatch, install_runtime, lambda handle: wrong)
    adapter = installed._adapter(rpu_execution={"model": {"num_cores": 6}})
    with pytest.raises(RuntimeError, match="native topology mismatch"):
        adapter.to_rpu()
    assert ("destroy", 1) in install_runtime["events"] and not install_runtime["alive"]
    assert install_runtime["releases"] == [adapter.model]
    assert not any(hasattr(adapter.model, name) for name in gemma4._INSTALL_ATTRS)


@pytest.mark.parametrize("value", (5, True, "4"))
def test_adapter_rejects_unknown_budgets_before_install(value):
    with pytest.raises((TypeError, ValueError), match="num_cores"):
        installed._adapter(rpu_execution={"model": {"num_cores": value}})


class _EmbeddedModel(installed._Model):
    """The install double with real CPU embedding/PLE tables for the patched forward."""

    def state_dict(self):
        values = super().state_dict()
        values.update({runtime.PFX + "embed_tokens.weight": torch.zeros(8, 2560),
                       runtime.PFX + "embed_tokens_per_layer.weight": torch.zeros(8, 42 * 256),
                       runtime.PFX + "per_layer_model_projection.weight": torch.zeros(42 * 256, 2560),
                       runtime.PFX + "per_layer_projection_norm.weight": torch.ones(256)})
        return values


def test_forward_side_input_uses_the_installed_ple_lanes(install_runtime, monkeypatch):
    lanes = []
    monkeypatch.setattr(runtime, "compute_side_input",
                        lambda *args, **kwargs: lanes.append(kwargs["num_cores"]) or installed._FakeTensor())
    expected = g4_topology.resolve_gemma4_topology(6)
    _patch_topology_ops(monkeypatch, install_runtime, lambda handle: expected.native_readback())
    model = installed._adapter(_EmbeddedModel(), rpu_execution={"model": {"num_cores": 6}}).to_rpu()
    try:
        cache = model.build_rpu_cache(max_seq_len=64)
        model.forward(input_ids=torch.ones(1, 1, dtype=torch.long), past_key_values=cache)
        assert lanes == [4]
    finally:
        installed._finalize(model)


# ------------------------------------------------------ example contract
def test_example_config_admits_gemma4_reduced_budgets():
    from examples.common import config as example_config

    path = ROOT / "examples/configs/gemma4/e4b/fp16.toml"
    config = example_config.load_config(path)
    assert "gemma4" not in example_config._FIXED_EIGHT
    assert example_config.core_support(config) == {4: None, 6: None, 8: None}
    profile = next(p for p in json.loads((ROOT / "examples/configs/profiles.json").read_text())["profiles"]
                   if p["id"] == "gemma4-e4b.fp16.text")
    assert profile["input_envelope"]["limits"]["allowed_num_cores"] == [4, 6, 8]
    assert profile["hw_status"] == "IMPLEMENTED_UNVERIFIED_HW"
    for cores in (4, 6, 8):
        execution = {"prefill": {"chunk_size": "auto"}, "model": {"num_cores": cores}}
        example_config._validate_execution(execution, profile=profile)
    with pytest.raises(Exception, match="num_cores"):
        example_config._validate_execution({"model": {"num_cores": 5}}, profile=profile)
    other = dict(profile, input_envelope={"limits": {}})
    with pytest.raises(example_config.ConfigError, match="Gemma4 core choice"):
        example_config._validate_execution({"model": {"num_cores": 6}}, profile=other)
