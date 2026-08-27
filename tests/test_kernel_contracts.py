from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_5 import text as qwen3_5_text
from rpu_backend.adapters.qwen3_5 import vision as qwen3_5_vision
from rpu_backend.adapters import qwen3
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError


ROOT = Path(__file__).resolve().parents[1]
EAGER_BMM_TILES = {
    (64, 64, 64),
    (64, 64, 128),
    (64, 128, 64),
    (64, 128, 128),
    (128, 128, 64),
    (128, 128, 128),
}


def test_operator_kernel_manifest_is_strict_and_asset_stays_opaque(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")

    source = tmp_path / "manifest_check.cpp"
    source.write_text(
        '#include "rpu_kernel_manifest.h"\n'
        "#include <iostream>\n"
        "int main(int argc, char** argv) {\n"
        "  try { std::cout << rpu_kernel_manifest::load(argv[1]).size(); }\n"
        "  catch (const std::exception& e) { std::cerr << e.what(); return 2; }\n"
        "}\n",
        encoding="utf-8",
    )
    executable = tmp_path / "manifest_check"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT / "src/core"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    asset = tmp_path / "operator.ref"
    asset.write_bytes(b"opaque")
    manifest = Path(str(asset) + ".kernels")

    def parse(contents: str | None) -> subprocess.CompletedProcess[str]:
        if contents is None:
            manifest.unlink(missing_ok=True)
        else:
            manifest.write_text(contents, encoding="utf-8")
        return subprocess.run(
            [str(executable), str(asset)],
            capture_output=True,
            text=True,
            check=False,
        )

    prefix = "rhinoforge-kernels-v1\nasset-size=6\n"
    assert parse(prefix + "alpha\nbeta_2\n").stdout == "2"
    assert parse("rhinoforge-kernels-v1\r\nasset-size=6\r\nalpha\r\n").stdout == "1"
    for invalid in (
        None,
        "wrong-header\nasset-size=6\nalpha\n",
        "rhinoforge-kernels-v1\n",
        "rhinoforge-kernels-v1\nasset-size=x\nalpha\n",
        "rhinoforge-kernels-v1\nasset-size=5\nalpha\n",
        prefix,
        prefix + "alpha\n\n",
        prefix + "alpha-beta\n",
        prefix + "beta\nalpha\n",
        prefix + "alpha\nalpha\n",
        prefix + "a" * (1024 * 1024) + "\n",
    ):
        assert parse(invalid).returncode != 0

    cache = (ROOT / "src/core/rpu_kernel_cache.inc").read_text(encoding="utf-8")
    cache_header = (ROOT / "src/core/rpu_kernel_cache.h").read_text(
        encoding="utf-8"
    )
    mrope = (ROOT / "src/ops/rpu_mrope.cpp").read_text(encoding="utf-8")
    for forbidden in (
        "ref_kernel_names",
        "SYMTAB",
        "STRTAB",
        "ELFCLASS64",
        "byte-scan",
    ):
        assert forbidden not in cache
    for removed in (
        "rpu_kernel_" + "probe_by_name",
        "rpu_rope_" + "kernel_info",
        "rpu_rope_" + "probe",
        "MROPE_" + "SPM_TBL",
    ):
        assert removed not in cache
        assert removed not in cache_header
        assert removed not in mrope

    loader = cache[cache.index("CachedKernel KernelCache::load_kernel_from_oplib") :]
    loader = loader[: loader.index("std::vector<std::string>")]
    assert loader.index("kernel_manifest_names") < loader.index(
        "create_with_binary_file"
    )


def test_eager_bmm_allowlist_matches_preloaded_kernels() -> None:
    bmm = (ROOT / "src/ops/rpu_bmm.cpp").read_text(encoding="utf-8")
    cache = (ROOT / "src/core/rpu_kernel_cache.inc").read_text(
        encoding="utf-8"
    )

    table = bmm.split("kPreloadedEagerBmmTiles[][3] = {", 1)[1].split(
        "};", 1
    )[0]
    assert {
        tuple(map(int, match))
        for match in re.findall(r"\{(\d+), (\d+), (\d+)\}", table)
    } == EAGER_BMM_TILES

    names = {
        (int(m), int(n), int(k), mode)
        for m, n, k, mode in re.findall(
            r'\{"gemm_fp16_spm_16b_w(\d+)x(\d+)_k(\d+)_1core_buf1_'
            r'nt_lpaddr_(peak|univ)"',
            cache,
        )
    }
    assert names == {
        (*tile, mode)
        for tile in EAGER_BMM_TILES
        for mode in ("peak", "univ")
    }

    eager = bmm.split("void rpu_launch_bmm_kernel", 1)[1].split(
        "// ============ BMM SPM Kernel Launch", 1
    )[0]
    assert eager.index("is_preloaded_eager_bmm_tile(") < eager.index(
        "std::string kernel_mode"
    )


@pytest.mark.parametrize("dims", [(64, 128), (128, 64)])
def test_qwen3_5_gdn_head_dims_fail_before_mutation(dims) -> None:
    model = SimpleNamespace(
        norm=SimpleNamespace(weight=torch.ones(1, dtype=torch.float16))
    )
    config = SimpleNamespace(
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_key_head_dim=dims[0],
        linear_value_head_dim=dims[1],
    )

    with pytest.raises(UnsupportedModelError, match="GDN requires"):
        qwen3_5_text._install_qwen3_5_text_for_rpu_impl(
            model,
            config,
            max_seq_len=128,
            chunk_size_cap=0,
            exact_chunk_size=0,
            padding_budget=0,
            padding_rows="auto",
            execution_config={},
            graph_cache=object(),
            prefill_graph=object(),
            cpu_stage_weights=False,
        )

    assert not hasattr(model, "_rpu_qwen3_5_text_install_started")


def test_qwen3_5_gdn_guard_precedes_irreversible_marker() -> None:
    source = Path(qwen3_5_text.__file__).read_text(encoding="utf-8")
    install = source.split("def _install_qwen3_5_text_for_rpu_impl", 1)[1]
    assert install.index("if not all(is_full):") < install.index(
        "inner._rpu_qwen3_5_text_install_started = True"
    )


@pytest.mark.parametrize("value", [None, "0", "true", "01"])
def test_qwen3_5_vision_numeric_gate_fails_before_mutation(
    value, monkeypatch
) -> None:
    if value is None:
        monkeypatch.delenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", raising=False)
    else:
        monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", value)
    model = SimpleNamespace(
        blocks=[SimpleNamespace() for _ in range(24)],
        merger=object(),
        config=SimpleNamespace(
            num_heads=16,
            hidden_size=1024,
            intermediate_size=4096,
            spatial_merge_size=2,
        ),
    )

    with pytest.raises(NotImplementedError, match="numeric-blocked"):
        qwen3_5_vision.install_qwen3_5_vision_for_rpu(model)

    assert not any(name.startswith("_rpu_") for name in vars(model))


def test_qwen3_5_vision_gate_precedes_handle_and_weight_mutation() -> None:
    source = Path(qwen3_5_vision.__file__).read_text(encoding="utf-8")
    install = source.split("def _install_qwen3_5_vision_for_rpu_impl", 1)[1]
    gate = install.index('QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED") != "1"')
    assert gate < install.index("vision_model._rpu_vision_installing = True")
    assert gate < install.index("torch.ops.rpu.qwen3_5_vision_create()")
    assert gate < install.index("_convert_vision_block_weights_for_rpu(")

    example = (ROOT / "examples/qwen3_5_vision.py").read_text(encoding="utf-8")
    assert example.index("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED") < example.index(
        "    import torch"
    )


def test_qwen3_14b_requires_exact_lm_head_quantization_metadata() -> None:
    profile = {
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 40,
        "num_key_value_heads": 8,
    }
    incomplete = SimpleNamespace(
        **profile,
        quant_config={"method": "w8a16"},
    )
    with pytest.raises(UnsupportedModelError, match="untied INT8 lm_head"):
        qwen3._check_profile(incomplete)

    exact = SimpleNamespace(
        **profile,
        quant_config={
            "method": "w8a16",
            "quantized_lm_head": True,
            "lm_head_untied": True,
            "quantized_embed_tokens": False,
        },
    )
    qwen3._check_profile(exact)

    model = torch.nn.Module()
    model.config = exact
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    model.model.embed_tokens = torch.nn.Embedding(2, 2).half()
    model.lm_head = torch.nn.Linear(2, 2).half()
    with pytest.raises(RPUBackendError, match="checkpoint tensors do not match"):
        qwen3.Qwen3Adapter(model)


def test_qwen3_14b_installs_existing_int8_lm_head_path() -> None:
    source = Path(qwen3.__file__).read_text(encoding="utf-8")
    to_rpu = source.split("    def to_rpu(self):", 1)[1].split(
        "\n\nfrom rpu_backend.runtime.registry", 1
    )[0]
    install_tail = to_rpu.split(
        "self._all_layers_once_handle = _install_causal_decoder_forward", 1
    )[1]
    assert install_tail.index("_apply_fused_lm_head_for_rpu(self.model)") < (
        install_tail.index("self.model._rpu_swizzled = True")
    )


def test_dinov3_example_preflights_before_weight_loading() -> None:
    source = (ROOT / "examples/dinov3.py").read_text(encoding="utf-8")
    assert source.index("DINOv3Adapter.preflight(model_hf_config)") < source.index(
        "DINOv3ViTModel.from_pretrained("
    )


def test_hyvla_native_handles_declare_exact_chunk_envelopes() -> None:
    cases = {
        "src/fused/rpu_hyvla_vlm_model.cpp": (240, 240),
        "src/fused/rpu_hyvla_expert_model.cpp": (291, 64),
    }
    for relative, (max_kv_len, chunk) in cases.items():
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert (
            f"set_chunk_envelope(/*max_kv_len=*/{max_kv_len}, "
            f"/*chunk=*/{chunk});"
        ) in source
        validity = source.split("subclass_chunk_size_valid(", 1)[1].split(
            "}", 1
        )[0]
        assert "chunk_within_envelope(cs)" in validity


def test_hyvla_vlm_allows_the_planner_fixed_overhead_probe() -> None:
    source = (ROOT / "src/fused/rpu_hyvla_vlm_model.cpp").read_text(
        encoding="utf-8"
    )
    declarations = source.split("HyVlaVlmModel::declare_buffers", 1)[1].split(
        "HyVlaVlmModel::build_layer_subgraph", 1
    )[0]
    assert "lctx.chunk_size >= moe_chunk_size_" not in declarations
    assert "const int64_t cs      = lctx.chunk_size" in declarations
    assert "chunk.len == moe_chunk_size_" in source


def test_fmb_joint_chunk_budget_and_modes_fail_closed() -> None:
    source = (ROOT / "src/core/fused_model_base.cpp").read_text(
        encoding="utf-8"
    )
    fixed = source.split("static FixedSpmOverhead estimate_fixed_overhead", 1)[1]
    fixed = fixed.split("int64_t detail::estimate_temporary_total", 1)[0]
    assert "Align(d.size" in fixed
    assert "SpmAllocator::ALIGN" in fixed
    assert "aligned_size * d.per_layer" in fixed

    joint = source.split("static void validate_final_chunk_layout_fits", 1)[1]
    joint = joint.split("static void validate_fmb_chunk_mode", 1)[0]
    assert "estimate_fixed_overhead(decls)" in joint
    assert "detail::estimate_temporary_total(decls)" in joint
    assert "available_temporary_spm_after_reset(" in joint
    assert "final joint compute/KV_FIRST layout exceeds SPM" in joint

    modes = source.split("resolve_and_validate_fmb_inter_layer_io", 1)[1]
    modes = modes.split("static void validate_kv_first_chunk_plan", 1)[0]
    assert "dynamic_config returned an invalid inter-layer I/O mode" in modes
    assert "chunk_outer_within_group requires SEQUENTIAL + SPM_RESIDENT" in modes


def test_physical_prepare_is_bound_to_the_exact_stage_plan() -> None:
    header = (ROOT / "src/core/fused_model_base.h").read_text(encoding="utf-8")
    source = (ROOT / "src/core/fused_model_base.cpp").read_text(
        encoding="utf-8"
    )
    assert header.count("const FmbThreeStageChunkPlan& stage_plan") == 2

    bind = source.split("LayoutContext bind_spm_pipeline_stage_plan", 1)[1]
    bind = bind.split("SpmPipelineComponentLayout FusedModelBase::", 1)[0]
    assert "validate_fmb_three_stage_chunk_plan(stage_plan)" in bind
    assert "fmb_three_stage_chunk_plan_fingerprint(stage_plan)" in bind
    assert "exact.stage_plan_fingerprint = fingerprint" in bind

    fused = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src/fused").glob("*.cpp")
    )
    assert re.search(
        r"prepare_spm_pipeline_component(?:_for_cpu_contract)?\(\s*"
        r"(?:layout|shape\.allocation_layout)\s*\)",
        fused,
    ) is None
