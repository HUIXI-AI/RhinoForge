"""Board-free contract for explicit GELU formulas and stable Softplus routing."""

import re
import shutil
import subprocess
from pathlib import Path

from test_high_precision_math_launch import _block


ROOT = Path(__file__).resolve().parents[1]


def test_ultra_erf_actual_launcher_registers_and_cold_native_controls(tmp_path):
    """Execute the production C++ functions against the existing queue ABI."""
    source = _read("src/ops/rpu_eltwise_unary.cpp")
    start = source.index("void rpu_launch_eltwise_unary_spm_kernel(")
    launcher = _block(source, "void rpu_launch_eltwise_unary_spm_kernel(")
    native = _read("src/fused/rpu_qwen3_5_vision_model.cpp")
    start = native.index("void Qwen3_5VisionModel::set_linear_acc32(")
    setters = "\n".join(_block(native, f"void Qwen3_5VisionModel::{name}(")
                        for name in ("set_linear_acc32", "set_gelu_erf_ultra"))
    modes = re.search(r"enum class GeluMode[^;]+;", _read("src/core/rpu_kernel_decls.h")).group()
    precision = re.search(r"enum class RpuUnaryPrecision[^;]+;", _read("src/core/rpu_math_precision.h")).group()
    harness = r'''
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
namespace c10 { struct Half { uint16_t x; Half(double = 0): x(0) {} }; }
enum class ValuOpType { ADD, HARDSIGMOID, SOFTPLUS, SILU, TANH, CLIP };
enum class KernelId { UNARY_SPM, UNARY_GELU_TANH_SPM, UNARY_GELU_ERF_SPM,
                      UNARY_GELU_ERF_ULTRA_SPM, UNARY_SOFTPLUS_SPM,
                      UNARY_SILU_HIGH_PRECISION, UNARY_TANH_HIGH_PRECISION,
                      UNARY_GELU_ERF_HIGH_PRECISION, UNARY_GELU_TANH_HIGH_PRECISION };
constexpr int FP16_RAB_DATA_TYPE_CFG = 0;
struct Info { int op = 0, custom_op = 0; };
Info get_valu_op_info(ValuOpType) { return {}; }
size_t CeilDiv(size_t a, size_t b) { return (a + b - 1) / b; }
struct Kernel_t {
  std::array<uint16_t, 64> regs{};
  void reset_regs() { regs.fill(0); }
  void set_regs(int reg, uint16_t value) { regs.at(reg) = value; }
} kernel;
KernelId selected;
Kernel_t* GET_KERNEL(KernelId id) { selected = id; return &kernel; }
struct Queue {
  bool broadcast = false;
  int cores = 0;
  std::array<uint16_t, 3> grid{};
  void set_broadcast_mode(bool value) { broadcast = value; }
  void enqueu_kernel(Kernel_t&, std::array<uint16_t,3> value, std::vector<uint8_t> ids) {
    grid = value; assert(broadcast && ids.size() == size_t(cores));
    for (int i = 0; i < cores; ++i) assert(ids[i] == i);
  }
} queue;
Queue* GET_QUEUE(int cores) { queue.cores = cores; return &queue; }
struct Qwen3_5VisionModel {
  std::vector<int> layer_weights_;
  int cores = 8, layers = 0, generation = 0;
  bool linear_acc32_ = false, gelu_erf_ultra_ = false, high_precision_ = false;
  int num_cores() { return cores; }
  int num_layers() { return layers; }
  void invalidate_model_state() { ++generation; }
  void set_linear_acc32(bool);
  void set_gelu_erf_ultra(bool);
  void set_high_precision(bool);
};
template<class F> void rejects(F call) {
  bool rejected = false; try { call(); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
}
'''
    checks = r'''
int main() {
  for (const auto mode : {GeluMode::ERF_ULTRA, GeluMode::ERF, GeluMode::TANH}) {
    const size_t block = mode == GeluMode::ERF_ULTRA ? 5888 : 25600;
    for (int cores : {4, 6, 8}) for (const size_t n : {size_t(1), size_t(5887), size_t(5888), size_t(5905), size_t(11776), size_t(25600), size_t(1048576)}) {
      rpu_launch_eltwise_unary_spm_kernel(0x123456, 0x123456, n, ValuOpType::ADD, mode, cores, RpuUnaryPrecision::BASE, 0.0, 0.0);
      assert(kernel.regs[0] == block);
      assert(kernel.regs[1] == n - (CeilDiv(n, block) - 1) * block);
      assert(kernel.regs[8] + (uint32_t(kernel.regs[9]) << 16) == block * 2);
      assert(kernel.regs[4] + (uint32_t(kernel.regs[5]) << 16) == 0x123456);
      assert(kernel.regs[6] == kernel.regs[4] && kernel.regs[7] == kernel.regs[5]);
      assert(kernel.regs[61] == 1 && queue.grid[0] == CeilDiv(n, block));
      assert(selected == (mode == GeluMode::ERF_ULTRA ? KernelId::UNARY_GELU_ERF_ULTRA_SPM
                          : mode == GeluMode::ERF ? KernelId::UNARY_GELU_ERF_SPM : KernelId::UNARY_GELU_TANH_SPM));
    }
  }
  Qwen3_5VisionModel model;
  rejects([&] { model.set_gelu_erf_ultra(true); });
  assert(model.generation == 0 && !model.gelu_erf_ultra_);
  model.set_linear_acc32(true); model.set_gelu_erf_ultra(true);
  assert(model.generation == 2 && model.gelu_erf_ultra_);
  rejects([&] { model.set_linear_acc32(false); });
  model.layer_weights_ = {1};
  rejects([&] { model.set_gelu_erf_ultra(false); });
  rejects([&] { model.set_linear_acc32(true); });
  assert(model.generation == 2);
  Qwen3_5VisionModel reduced; reduced.cores = 4;
  reduced.set_linear_acc32(true); reduced.set_gelu_erf_ultra(true);
  assert(reduced.linear_acc32_ && reduced.gelu_erf_ultra_ && reduced.generation == 2);
  rejects([&] { reduced.set_linear_acc32(false); });
  reduced.set_gelu_erf_ultra(false); reduced.set_linear_acc32(false);
  assert(!reduced.linear_acc32_ && !reduced.gelu_erf_ultra_);
  for (int cores : {0, 1, 2, 3, 5, 6, 7, 9}) {
    Qwen3_5VisionModel invalid; invalid.cores = cores;
    rejects([&] { invalid.set_linear_acc32(true); });
    assert(!invalid.linear_acc32_ && invalid.generation == 0);
    invalid.linear_acc32_ = true;
    rejects([&] { invalid.set_gelu_erf_ultra(true); });
    assert(!invalid.gelu_erf_ultra_ && invalid.generation == 0);
  }
  assert(reduced.generation == 4);
}
'''
    cpp = tmp_path / "ultra_erf.cpp"
    cpp.write_text(harness + modes + precision + launcher + setters + checks)
    binary = tmp_path / "ultra_erf"
    subprocess.run([shutil.which("g++"), "-std=c++17", "-O0", str(cpp), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_spm_unary_selects_the_formula_specific_kernels():
    source = _read("src/ops/rpu_eltwise_unary.cpp")
    for selector, kernel_id in (
        ("gelu_mode == GeluMode::TANH", "KernelId::UNARY_GELU_TANH_SPM"),
        ("gelu_mode == GeluMode::ERF", "KernelId::UNARY_GELU_ERF_SPM"),
        ("op_type == ValuOpType::SOFTPLUS", "KernelId::UNARY_SOFTPLUS_SPM"),
    ):
        match = re.search(
            rf"{re.escape(selector)}.*?kid = {re.escape(kernel_id)};",
            source,
            re.DOTALL,
        )
        assert match is not None

    # rhino-ops 7cb32d44's unary_softplus intentionally retains the generic
    # unary register/block ABI; a second Softplus-specific launch ABI is wrong.
    assert "normal_blk_n = 100 * 256" in source
    assert "rpu_launch_softplus_spm_kernel" not in source

    decls = _read("src/core/rpu_kernel_decls.h")
    assert "is_gelu ? GeluMode::TANH : GeluMode::NONE" in decls
    assert "UnaryKernelPolicy" not in "\n".join((decls, source))
    assert "LEGACY_TANH" not in "\n".join((decls, source))


def test_shared_qwen3vl_vision_uses_tanh_blocks_and_erf_mergers():
    source = _read("src/fused/rpu_qwen3vl_vision_model.cpp")
    assert re.search(
        r'addr\(0, "fc1"\), addr\(0, "fc1"\).*?'
        r'GeluMode::TANH, num_cores\(\)',
        source,
        re.DOTALL,
    )
    assert re.search(
        r'addr\(0, "merger_mid"\), addr\(0, "merger_mid"\).*?'
        r'GeluMode::ERF, num_cores\(\)',
        source,
        re.DOTALL,
    )
    assert "GeluMode::LEGACY_TANH" not in source

    qwen25 = _read("src/fused/rpu_qwen25vl_vision_model.cpp")
    assert re.search(
        r'addr\(0, "merger_mid"\), addr\(0, "merger_mid"\).*?'
        r'GeluMode::ERF, num_cores\(\)',
        qwen25,
        re.DOTALL,
    )
    assert "GeluMode::LEGACY_TANH" not in qwen25


def test_shared_models_keep_their_declared_gelu_formula():
    expected = {
        "src/core/fused_model_base.cpp": "GeluMode::TANH",
        "src/fused/rpu_dinov3_vision_model.cpp": "GeluMode::ERF",
        "src/fused/rpu_gemma2_model.cpp": "GeluMode::TANH",
        "src/fused/rpu_gemma4_model.cpp": "GeluMode::TANH",
        "src/fused/rpu_navdp_model.cpp": "GeluMode::ERF",
        "src/fused/rpu_qwen25vl_vision_model.cpp": "GeluMode::ERF",
        "src/fused/rpu_siglip_model.cpp": "GeluMode::TANH",
        # Eager Qwen3-VL merger fc1 epilogue (linear_with_accumulation gelu_erf).
        "src/ops/rpu_linear.cpp": "gelu_erf ? GeluMode::ERF : GeluMode::NONE",
    }
    for relative, mode in expected.items():
        assert mode in _read(relative), f"{relative} must select {mode}"

    qwen35 = _read("src/fused/rpu_qwen3_5_vision_model.cpp")
    assert "GeluMode::TANH" in qwen35
    assert "GeluMode::ERF" in qwen35


def test_pi05_fused_models_use_canonical_tanh_gelu():
    base = _read("src/core/fused_model_base.cpp")
    header = _read("src/core/fused_model_base.h")
    users = {
        "rpu_adarms_model.cpp",
        "rpu_gemma_model.cpp",
        "rpu_pi05_denoise_step_model.cpp",
    }
    for user in users:
        source = _read(f"src/fused/{user}")
        assert "ActivationKind::GELU" in source
        assert "PI05_SOURCE_GELU" not in source
        assert "source_gelu_w8a16" not in source

    assert "PI05_SOURCE_GELU" not in "\n".join((base, header))
    assert re.search(
        r"case ActivationKind::GELU:.*?GeluMode::TANH",
        base,
        re.DOTALL,
    )


def test_no_fused_model_selects_the_legacy_gelu_symbol():
    fused_sources = list((ROOT / "src" / "fused").glob("*.cpp"))
    fused_sources.append(ROOT / "src" / "core" / "fused_model_base.cpp")
    for path in fused_sources:
        source = path.read_text(encoding="utf-8")
        assert "GeluMode::LEGACY_TANH" not in source, path


def test_eager_gelu_uses_graph_aware_cpu_fallback():
    registrations = _read("src/core/rpu_dispatch_registrations.inc")
    assert re.search(
        r'm\.impl\("gelu",\s*torch::CppFunction::makeFromBoxedFunction'
        r'<&custom_cpu_fallback>\(\)\);',
        registrations,
    )
    assert "m.fallback(torch::CppFunction::makeFromBoxedFunction" in registrations
    assert "custom_cpu_fallback" in registrations

    # There is no callable eager device GELU implementation left behind the
    # dispatcher.  Functional GELU is a Tier2 host callback during capture,
    # so BUILD/REPLAY keeps the CPU fallback in the graph lifecycle.
    assert "rpu_gelu" not in _read("src/core/rpu_tensor_ops.h")
    assert "rpu_gelu" not in _read("src/core/rpu_tensor_ops.inc")
    graph = _read("src/graph/custom_cpu_fallback_dispatch.cpp")
    assert "HostCallbackTier::Tier2" in graph
    assert "capture_host_callback_replay_args" in graph

    # HostCallback capture/execution already owns boundary ordering.  Calling
    # the ordinary zero-copy helper here would sync and recursively execute the
    # HostCallback node currently being recorded.
    fallback = _read("src/graph/graph_cpu_fallback_stub.cpp")
    assert fallback.count("rpu_to_cpu_zerocopy_no_graph_sync(") == 3
    assert "rpu_to_cpu_zerocopy(tensor)" not in fallback
    tensor_ops = _read("src/core/rpu_tensor_ops.inc")
    assert "rpu_to_cpu_zerocopy_impl" in tensor_ops
    assert "if (graph_sync)" in tensor_ops


def test_qwen35_prefill_and_decode_both_route_softplus_to_the_new_symbol():
    qwen35 = _read("src/fused/rpu_qwen3_5_model.cpp")
    recurrent = _read("src/ops/rpu_gdn_recurrent.cpp")
    assert "ValuOpType::SOFTPLUS" in qwen35
    assert "rpu_emit_gdn_recurrent_multihead_seq(" in qwen35
    assert "ValuOpType::SOFTPLUS" in recurrent


def test_gr00t_z2_uses_formula_specific_tanh_gelu():
    for relative in (
        "src/fused/rpu_gr00t_dit_model.cpp",
        "src/fused/rpu_gr00t_vl_encoder_model.cpp",
    ):
        source = _read(relative)
        assert "GeluMode::TANH" in source
        assert "legacy_gelu_for_frozen_graph" not in source
