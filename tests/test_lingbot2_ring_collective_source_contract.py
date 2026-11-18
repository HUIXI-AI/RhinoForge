"""Board-free contract for the retired LingBot2 alias and common ring route."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src/core/rpu_runtime_extras.cpp"
PRIVATE_HEADER = ROOT / "src/core/rpu_lingbot2_ring_collective.h"
PUBLIC_DECLS = ROOT / "src/core/rpu_kernel_decls.h"
VISION = ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp"
TEXT = ROOT / "src/fused/rpu_qwen3_model.h"
MLP = ROOT / "src/core/fused_model_base.cpp"
DENOISE = ROOT / "src/fused/rpu_lingbot_v2_moe_model.cpp"
OUTER = ROOT / "src/fused/rpu_qwen3vl_multiview_spm_z2.cpp"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _ring_body() -> str:
    source = _read(RUNTIME)
    start = source.index("constexpr int kRingCoreNum")
    end = source.index("\n}  // namespace", start)
    return source[start:end]


def test_private_lingbot2_alias_is_deleted_in_favor_of_the_public_ring_route():
    runtime = _read(RUNTIME)
    public_decls = _read(PUBLIC_DECLS)

    assert not PRIVATE_HEADER.exists()
    assert "rpu_launch_lingbot2_ring" not in runtime
    assert "rpu_launch_lingbot2_ring" not in public_decls
    assert "launch_ring_all_reduce_sum_residual(" in runtime
    assert "Kernel_t* kernel = GET_KERNEL(kernel_id);" in runtime
    assert (
        "GraphKernelNoDdrProof::RingAllReduceSpmResidual" in runtime
    )


def test_ring_dispatch_and_register_abi_match_generator_contract():
    body = _ring_body()

    for token in (
        '"llm_all_reduce_residual_nopace"',
        '"llm_all_reduce_residual"',
        "chunk_bytes < kRingNopaceChunkBytes",
        "11U * 1024U",
        "31U * 256U * kRingWarpNum",
        "residual_local_shard ? 2 : (residual_ddr != nullptr)",
        "kernel->set_regs(65, static_cast<uint16_t>(1))",
        "kernel->set_regs(66, static_cast<uint16_t>(1))",
        "kScmBase + 8 + core",
        "kScmBase + 16 + core",
        "kScmBase + 24 + core",
        "kScmBase + 32 + core",
        "kScmBase + 40 + core",
        "kScmBase + 48 + core",
        "kScmBase + 56 + core",
        "queue->set_broadcast_mode(true)",
    ):
        assert token in body

    squashed = "".join(body.split())
    for reg in (0, 1, 2, 3, 4, 5, 64):
        assert f"kernel->set_regs({reg}," in squashed

    assert "total_elements + static_cast<uint64_t>(core_num) - 1" in body
    compact = "".join(body.split())
    assert (
        "ranges_are_disjoint(input_spm,tensor_bytes,output_spm,tensor_bytes)"
        in compact
    )
    assert (
        "ranges_are_disjoint(input_spm,tensor_bytes,residual_spm,residual_bytes)"
        in compact
    )
    assert (
        "ranges_are_disjoint(residual_spm,residual_bytes,output_spm,tensor_bytes)"
        in compact
    )
    assert "SPM_ALLOC.addr(core, input_local)" in body
    assert "SPM_ALLOC.addr(core, output_local)" in body


def test_known_lingbot2_shapes_select_expected_ring_variant():
    def dispatch(rows: int, cols: int) -> tuple[str, int, int]:
        total = rows * cols
        avg = ((total + 7) // 8 + 15) // 16 * 16
        chunk_bytes = avg * 2
        rounds = max(1, (avg + 31 * 256 * 8 - 1) // (31 * 256 * 8))
        return (
            "nopace" if chunk_bytes < 11 * 1024 else "paced",
            chunk_bytes,
            rounds,
        )

    assert dispatch(51, 768) == ("nopace", 9_792, 1)
    assert dispatch(256, 1024) == ("paced", 65_536, 1)
    assert dispatch(225, 2560) == ("paced", 144_000, 2)
    assert dispatch(64, 2560) == ("paced", 40_960, 1)


def test_all_model_routes_use_the_public_admission_wrapper():
    private_call = (
        "rpu_launch_lingbot2_ring_all_reduce_sum_residual_kernel("
    )
    for source in (DENOISE, VISION, TEXT, MLP):
        body = _read(source)
        assert private_call not in body
        assert "rpu_launch_all_reduce_sum_residual_kernel(" in body

    outer_src = _read(OUTER)
    start = outer_src.index('const GraphKernelRegisterPolicy& lingbot2_multiview_w8_register_policy()')
    end = outer_src.index('\nGraphSignature graph_signature(', start)
    assert 'return rpu_pl_census::policy("lingbot2_multiview_w8_register_policy")' in outer_src[start:end]
    policies_src = _read(ROOT / 'src/fused/rpu_parallel_linear_register_policies.h')
    p_start = policies_src.index('inline const GraphKernelRegisterPolicy& lingbot2_multiview_w8_register_policy()')
    p_end = policies_src.index('return value;', p_start)
    p_body = policies_src[p_start:p_end]
    assert 'rule("llm_all_reduce_residual", 228, GraphKernelRegisterRuleKind::ExplicitNoDdr, GraphKernelNoDdrProof::RingAllReduceSpmResidual, {}, false)' in p_body
    for retired in ("llm_all_reduce_sum_residual", "llm_reduce_sum_scatter_residual_v2", "all_gather_multi_core"):
        assert f'"{retired}"' not in p_body


def test_actual_attn_ring_adarms_cold_admission_covers_both_expert_precisions(tmp_path):
    source = _read(DENOISE)
    start = source.index("    const bool adarms_norm_shift_profile =", source.index(
        "void LingbotV2MoeExpertModel::configure_runtime_graph("))
    end = source.index("    const int select_qkv_multibase =", start)
    admission = source[start:end]
    program = r'''
#include <array>
#include <cassert>
#include <functional>
enum class KernelId {
    LINGBOT2_ADARMS_NORM_SHIFT_H768_DIAGONAL,
    PI05_ADARMS_NORM_SHIFT_H1024,
    LINGBOT2_ATTN_ALL_REDUCE_ADARMS_T51_H768,
};
struct KernelCache {
    std::array<bool, 3> loaded{{true, true, true}};
    static KernelCache& instance() { static KernelCache cache; return cache; }
    bool has_loaded(KernelId id) const { return loaded[static_cast<int>(id)]; }
};
struct Selection { bool capable, diagonal, ring; };
struct Model {
    bool sparse_experts_ = true;
    int sparse_weight_mode_ = 2;
    int layers = 36, hidden = 768, chunk_size_ = 51, tp_rows_ = 64, tp = 8, cores = 8;
    double eps_ = 1e-6;
    int num_cores() const { return cores; }
    int num_layers() const { return layers; }
    int hidden_size() const { return hidden; }
    int attn_tp() const { return tp; }
    Selection configure(bool fused_router = true) const {
__ACTUAL_ADMISSION__
        return {select_adarms_norm_shift, select_adarms_diagonal, select_attn_ring_adarms};
    }
};
int main() {
    for (int mode : {1, 2}) {
        KernelCache::instance().loaded = {{true, true, true}};
        Model model; model.sparse_weight_mode_ = mode;
        const auto selected = model.configure();
        assert(selected.capable && selected.diagonal && selected.ring);
        // Published cold selection remains an immutable value after availability changes.
        KernelCache::instance().loaded[2] = false;
        assert(selected.ring && !model.configure().ring);
        KernelCache::instance().loaded[2] = true;
        KernelCache::instance().loaded[0] = false;
        const auto fallback = model.configure();
        assert(fallback.capable && !fallback.diagonal && !fallback.ring);
        KernelCache::instance().loaded[1] = false;
        assert(!model.configure().capable && !model.configure().ring);
        KernelCache::instance().loaded = {{true, true, true}};
        assert(!model.configure(false).ring);
        for (const auto& corrupt : std::array<std::function<void(Model&)>, 8>{{
            [](Model& m) { m.sparse_experts_ = false; },
            [](Model& m) { m.layers = 35; },
            [](Model& m) { m.hidden = 769; },
            [](Model& m) { m.chunk_size_ = 50; },
            [](Model& m) { m.tp_rows_ = 63; },
            [](Model& m) { m.tp = 4; },
            [](Model& m) { m.eps_ = 1e-5; },
            [](Model& m) { m.cores = 4; },
        }}) {
            Model bad = model; corrupt(bad);
            assert(!bad.configure().ring);
        }
    }
    for (int mode : {0, 3, -1}) {
        Model model; model.sparse_weight_mode_ = mode;
        const auto selected = model.configure();
        assert(!selected.capable && !selected.diagonal && !selected.ring);
    }
}
'''.replace("__ACTUAL_ADMISSION__", admission)
    cpp = tmp_path / "attn_ring_admission.cpp"
    executable = tmp_path / "attn_ring_admission"
    cpp.write_text(program, encoding="utf-8")
    built = subprocess.run([
        "g++", "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror",
        str(cpp), "-o", str(executable),
    ], capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr
