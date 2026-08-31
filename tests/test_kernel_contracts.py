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
from rpu_backend.adapters.wall_oss import llm as wall_llm
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


def test_w8a16_prefix_tile_uses_the_measured_exact_entry(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")

    source = tmp_path / "linear_tiling_check.cpp"
    source.write_text(
        '#include "rpu_linear_tiling.h"\n'
        "int main() {\n"
        "  const auto w8 = rpu_pl_tiling::select_tile_acc16(400, 2048, 2048, false);\n"
        "  const auto nearby = rpu_pl_tiling::select_tile_acc16(399, 2048, 2048, false);\n"
        "  const auto w16 = rpu_pl_tiling::select_tile_acc16(400, 2048, 2048, true);\n"
        "  return w8.m_tile != 400 || w8.n_tile != 80 ||\n"
        "         nearby.m_tile != 512 || nearby.n_tile != 48 ||\n"
        "         w16.m_tile != 480 || w16.n_tile != 64;\n"
        "}\n",
        encoding="utf-8",
    )
    executable = tmp_path / "linear_tiling_check"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT / "src/ops"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


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


def test_wall_oss_exact_multichunk_uses_global_kv_and_ordered_merger() -> None:
    vision = (
        ROOT / "src" / "fused" / "rpu_qwen25vl_vision_model.cpp"
    ).read_text(encoding="utf-8")
    mask_dma = (ROOT / "src" / "ops" / "rpu_sdpa.cpp").read_text(
        encoding="utf-8"
    )
    adapter = (
        ROOT
        / "python"
        / "rpu_backend"
        / "adapters"
        / "wall_oss"
        / "vision.py"
    ).read_text(encoding="utf-8")

    assert "&Qwen25VLVisionModel::emit_kv_first_body" in vision
    assert "cfg.chunk_mode = ChunkMode::KV_FIRST" in vision
    assert "current_num_patches_ == 1024" in vision
    assert "configured_chunk_size_ == 768" in vision
    assert "insert_position = ctx().position + chunk.offset" in vision
    assert "query_row_offset=*/chunk.offset + q_rel" in vision
    assert "full_seq, NUM_CORES, NUM_CORES" in vision
    assert 'addr(0, chunk.idx == 0 ? "merger_out0" : "merger_out1")' in vision
    assert "output_tensor(), chunk.offset * h, chunk.len * h" in vision
    assert "chunk.offset % 4 == 0 && chunk.len % 4 == 0" in vision

    assert "query_row_offset * row_elements" in mask_dma
    assert "mask.ddr_tensor, source_offset, mask_elements" in mask_dma
    assert "packed multi-image multi-chunk Vision is not" in adapter
    assert "per-window SDPA multi-chunk is not implemented" in adapter
    forward_one = adapter.split("def _forward_one", 1)[1].split(
        "def _forward_group", 1
    )[0]
    immediate_reverse = forward_one.split("if self._fused_merger:", 1)[1].split(
        "if self._device_merged:", 1
    )[0]
    assert immediate_reverse.index("spm_alloc_reset_temporary()") < (
        immediate_reverse.index("gather_embedding(merged, ri)")
    )


@pytest.mark.parametrize(
    "prefill_config",
    [{}, {"chunk_size": "auto", "padding_budget": 64}],
)
def test_wall_oss_prefill_uses_shared_multichunk_planner(
    monkeypatch, prefill_config
) -> None:
    assert "causal_decoder_set_equal_two_prefill" not in Path(
        wall_llm.__file__
    ).read_text(encoding="utf-8")
    model = wall_llm.WallOssLLM.__new__(wall_llm.WallOssLLM)
    model.cache = SimpleNamespace(max_seq_len=8192)
    model._handle = 7
    model._prefill_pad16 = True
    model._rpu_execution = {"prefill": prefill_config}

    def resolve(handle, execution_len, position):
        assert (handle, position) == (7, 0)
        if execution_len == 656:
            return 320
        if execution_len == 672:
            return 224
        raise RuntimeError("no legal chunk")

    monkeypatch.setattr(
        torch.ops.rpu,
        "causal_decoder_resolve_prefill_chunk_size",
        resolve,
        raising=False,
    )

    assert model.prefill_execution_plan(643) == (656, 320)
    assert (656 + 320 - 1) // 320 == 3


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


def test_qwen3_5_standalone_pooler_preserves_native_fp16_contract() -> None:
    adapter = Path(qwen3_5_vision.__file__).read_text(encoding="utf-8")
    standalone = adapter.split("def _rpu_vision_forward", 1)[1].split(
        "def fuse_visual_embeds", 1
    )[0]
    native = (
        ROOT / "src" / "fused" / "rpu_qwen3_5_vision_model.cpp"
    ).read_text(encoding="utf-8")

    assert 'device="cpu", dtype=torch.float16' in standalone
    assert "self.merger(last_hidden.detach().cpu().float()).to(" in standalone
    assert "dtype=torch.float16" in standalone
    assert ").detach().cpu().float()" not in standalone
    assert "merged_buf_ = at::empty(" in native
    assert "input.options());" in native
    assert "merged_buf_.data_ptr<c10::Half>()" in native


@pytest.mark.parametrize(
    ("grid", "input_len", "message"),
    [
        ([1, 130, 32], 1, "total patches"),
        ([1, 2, 130], 1, "2D RoPE"),
        ([1, 30, 40], 385, "image-prefill maximum"),
    ],
)
def test_qwen3_5_vision_request_rejects_before_install(
    grid, input_len, message, monkeypatch
) -> None:
    called = False

    def install(_model):
        nonlocal called
        called = True

    monkeypatch.setattr(qwen3_5_vision, "install_qwen3_5_vision_for_rpu", install)
    model = SimpleNamespace(
        config=SimpleNamespace(image_token_id=248056),
        model=SimpleNamespace(
            visual=SimpleNamespace(config=SimpleNamespace(spatial_merge_size=2))
        ),
    )
    patches = grid[0] * grid[1] * grid[2]

    with pytest.raises(ValueError, match=message):
        qwen3_5_vision.fuse_visual_embeds(
            model,
            torch.empty(1, 1, 1),
            torch.zeros(1, input_len, dtype=torch.long),
            torch.empty(patches, 1),
            torch.tensor([grid]),
            mm_token_type_ids=torch.zeros(1, input_len, dtype=torch.long),
        )

    assert not called


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


def test_qwen3_32b_exact_public_profile_rejects_before_admission() -> None:
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


def test_lingbot2_moe_declares_its_native_chunk_envelope() -> None:
    source = (ROOT / "src/fused/rpu_lingbot_v2_moe_model.cpp").read_text(
        encoding="utf-8"
    )
    setter = source.split("LingbotV2MoeExpertModel::set_moe_weights", 1)[1]
    setter = setter.split("LingbotV2MoeExpertModel::declare_buffers", 1)[0]
    assert "tp_rows_        = Align(chunk_size, (int64_t)16);" in setter
    assert (
        "set_chunk_envelope(/*max_kv_len=*/cos_.size(0), "
        "/*chunk=*/tp_rows_);"
    ) in setter


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
    lingbot = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "python/rpu_backend/api/lingbot2.py",
            "src/core/rpu_dispatch_registrations.inc",
            "src/core/rpu_kernel_decls.h",
            "src/fused/rpu_lingbot_v2_moe_model.cpp",
            "src/fused/rpu_lingbot_v2_moe_model.h",
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
    assert "bool get_debug_export() {\n    return false;\n}" in runtime_state
    assert "qwen3_5_vision_get_dbg_" not in lingbot
    assert "RPU_GRAPH_HCB_CHECKSUM" not in graph_execute
    assert "[HCB-CK]" not in graph_execute
    assert "RPU_PI05_PROBE_DIR" not in pi05
    assert "torch.save(" not in pi05
    for marker in (
        "lingbot_v2_moe_debug_",
        "RPU_LINGBOT2_DEBUG_DUMP_ROUTER_H",
        "RPU_L2_CAPTURE_",
        "RPU_L2_CAP_RESID",
        "RPU_L2_DBG_PACKED",
        "RPU_L2_STAGES",
    ):
        assert marker not in lingbot

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


def test_graph_segments_follow_runtime_capacity_and_preserve_dma_fences() -> None:
    source = (ROOT / "src/graph/graph_runtime_execute.cpp").read_text(
        encoding="utf-8"
    )
    backend = (ROOT / "src/core/rpu_backend.cpp").read_text(encoding="utf-8")
    runtime_state = (ROOT / "src/core/rpu_runtime_state.h").read_text(
        encoding="utf-8"
    )
    budget = source.split("static SegmentResourceBudget segment_resource_budget()", 1)[
        1
    ].split("static SegmentResourceUse segment_node_resources", 1)[0]
    assert "rpu_lkn_batch_config_at_load()" in budget
    assert "getenv" not in budget
    assert "LKN_MAX_BATCH_ENTRIES" not in budget
    assert (
        "const LknBatchConfig& rpu_lkn_batch_config_at_load();"
        in runtime_state
    )
    assert re.search(
        r"const LknBatchConfig& rpu_lkn_batch_config_at_load\(\)\s*"
        r"\{\s*return kLknBatchConfigAtLoad;\s*\}",
        backend,
    )
    assert "SegmentResourceBudget{sdk_entries, sdk_kd, sdk_instr}" in budget
    assert "sdk_entries / 2" not in budget
    assert "8192" not in budget
    assert "20000" not in budget
    assert "pi05_denoise_10_step_owns_larger_entry_budget" not in source
    assert "16 + 15 * sizeof(uint64_t)" in source
    assert "fenced_group_resources" in source
    assert "admit_batch_group(i);" in source
    assert "one indivisible batch group exceeds" in source
    assert "barrier.target_stream) != streams.end()" in source
    assert "unterminated cross-stream DMA fence" not in source


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
