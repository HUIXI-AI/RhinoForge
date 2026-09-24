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
    ):
        assert removed not in cache
        assert removed not in cache_header
        assert removed not in mrope

    loader = cache[cache.index("CachedKernel KernelCache::load_kernel_from_oplib") :]
    loader = loader[: loader.index("std::vector<std::string>")]
    assert loader.index("kernel_manifest_names") < loader.index(
        "create_with_binary_file"
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
            control_snapshot=None,
            pending_state=SimpleNamespace(),
        )

    assert not hasattr(model, "_rpu_qwen3_5_text_install_started")


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


def test_qwen3_32b_requires_exact_quantized_metadata_before_admission() -> None:
    config = SimpleNamespace(
        hidden_size=5120,
        intermediate_size=25600,
        num_hidden_layers=64,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
    )
    before = vars(config).copy()

    with pytest.raises(UnsupportedModelError, match="no certified RPU execution"):
        qwen3.Qwen3Adapter.preflight(config)

    assert vars(config) == before
    config.model_type = "qwen3"
    config.architectures = ["Qwen3ForCausalLM"]
    config.rms_norm_eps = 1e-6
    config.vocab_size = 151936
    config.max_position_embeddings = 40960
    config.tie_word_embeddings = False
    config.quant_config = {
        "method": "w8a16", "mode": "per_channel_symmetric", "qaxis": 0,
        "skip_modules": [], "quantized_lm_head": True,
        "quantized_embed_tokens": False, "lm_head_untied": True,
        "embed_tokens_untied": False,
    }
    qwen3.Qwen3Adapter.preflight(config)
    qwen3.Qwen3Adapter.preflight_execution(config, {"prefill": {"chunk_size": 32}})
    with pytest.raises(UnsupportedModelError, match="num_cores=8"):
        qwen3.Qwen3Adapter.preflight_execution(config, {"model": {"num_cores": 4}})
    config.quant_config["qaxis"] = False
    with pytest.raises(UnsupportedModelError, match="no certified RPU execution"):
        qwen3.Qwen3Adapter.preflight(config)


def test_dinov3_example_preflights_before_weight_loading() -> None:
    source = (ROOT / "examples/dinov3.py").read_text(encoding="utf-8")
    assert source.index("DINOv3Adapter.preflight(model_hf_config)") < source.index(
        "DINOv3ViTModel.from_pretrained("
    )


def test_public_release_has_no_activation_export_path() -> None:
    python_debug = (
        ROOT / "python/rpu_backend/runtime/debug.py"
    ).read_text(encoding="utf-8")
    torch_namespace = (
        ROOT / "python/rpu_backend/_torch_rpu.py"
    ).read_text(encoding="utf-8")
    pybind = (ROOT / "src/core/rpu_pybind.inc").read_text(encoding="utf-8")
    runtime_state = (
        ROOT / "src/core/rpu_runtime_state.cpp"
    ).read_text(encoding="utf-8")
    graph_execute = (
        ROOT / "src/graph/graph_runtime_execute.cpp"
    ).read_text(encoding="utf-8")
    pi05 = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "python/rpu_backend/adapters/pi05/runtime.py",
            "python/rpu_backend/adapters/pi05/gemma.py",
            "python/rpu_backend/adapters/pi05/__init__.py",
        )
    )


    for symbol in (
        "set_debug_export",
        "get_debug_tensor",
        "list_debug_tensors",
        "clear_debug_tensors",
    ):
        assert f"def {symbol}" not in python_debug
        assert f'"{symbol}"' not in torch_namespace
        assert f'm.def("{symbol}"' not in pybind
    assert "void set_debug_export(" not in runtime_state
    runtime_header = (ROOT / "src/core/rpu_runtime_state.h").read_text()
    assert "inline constexpr bool get_debug_export() noexcept { return false; }" in runtime_header
    assert "RPU_GRAPH_HCB_CHECKSUM" not in graph_execute
    assert "[HCB-CK]" not in graph_execute
    assert "RPU_PI05_PROBE_DIR" not in pi05
    assert "torch.save(" not in pi05
    # The shared forward implementations retain dormant capture branches, but
    # the public runtime has no switch, operator binding, or environment reader
    # that can activate or retrieve them.
    bindings = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    for name in re.findall(r'm\.(?:def|impl)\("([^"(]+)', bindings):
        assert not any(part in name for part in ("debug_", "_dbg_", "flush_dumps"))
    native_lingbot = (ROOT / "src/fused/rpu_lingbot_v2_moe_model.cpp").read_text()
    assert not re.search(r'getenv\("(?:RPU_L2_(?:CAP|STAGES|DBG)|RPU_LINGBOT2_DEBUG)', native_lingbot)

    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for pattern in ("*.pth", "*.ckpt", "*.safetensors", "*.npz", "*.onnx"):
        assert pattern in ignored


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
