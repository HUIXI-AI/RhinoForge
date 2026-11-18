"""Source-level ownership contracts for shared RMSNorm physical routes."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECLS = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
HY_CPP = (ROOT / "src/fused/rpu_hyvla_vlm_model.cpp").read_text()
HY_PY = (ROOT / "python/rpu_backend/adapters/hy_vla/runtime.py").read_text()
QWEN3 = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
WALL_LLM = (ROOT / "python/rpu_backend/adapters/wall_oss/llm.py").read_text()
QWEN25 = (ROOT / "src/fused/rpu_qwen25vl_vision_model.cpp").read_text()
WALL_VISION = (
    ROOT / "python/rpu_backend/adapters/wall_oss/vision.py"
).read_text()
WALL_RUNTIME = (
    ROOT / "python/rpu_backend/adapters/wall_oss/runtime.py"
).read_text()
WALL_API = (ROOT / "python/rpu_backend/api/wall_oss.py").read_text()
RMSNORM_CPP = (ROOT / "src/ops/rpu_rmsnorm.cpp").read_text()


def _between(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    return source[begin : source.index(end, begin)]


def test_fixed32_cannot_fall_through_to_v16_for_m336():
    resolver = _between(
        DECLS,
        "inline RpuRmsNormSpmRoute rpu_resolve_rmsnorm_spm_route(",
        "void rpu_launch_rmsnorm_kernel(",
    )
    assert resolver.index("RPU_RMSNORM_CAP_V32") < resolver.index(
        "RPU_RMSNORM_CAP_V16"
    )
    assert "capability & RPU_RMSNORM_CAP_V16" in resolver
    assert "return RpuRmsNormSpmRoute::BASE;" in resolver
    assert "FIXED32 must not silently select V16 for M336" in (
        ROOT / "tests/test_rmsnorm_route.cpp"
    ).read_text()


def test_hyvla_auto_is_cold_and_every_geometry_has_manifest_authority():
    assert '"RPU_RMSNORM_VWARP": "auto"' in HY_PY
    prepare = _between(
        HY_PY, "def prepare_hy_vla_cold_plan(", "class _DirectHyVlaOwner:"
    )
    assert prepare.index("effective.setdefault(key, value)") < prepare.index(
        "native = _HyVlaNativeColdSnapshot.from_environment(effective)"
    )
    builder = HY_PY[HY_PY.index("def build_hy_vla("):]
    assert builder.index("prepare_hy_vla_cold_plan(num_cores=budget)") < builder.index(
        "_claim_live_instance(lifecycle_owner)"
    )
    assert "_native_cold=cold_plan.native" in builder
    bind = _between(HY_PY, "def bind_vlm(h):", 'vlm = self._sub("vlm"')
    assert bind.index("hyvla_vlm_set_runtime_config") < bind.index(
        "hyvla_vlm_set_weights"
    )
    for site in (
        "kHyVlaVlmRmsNormFullRowsSite",
        "kHyVlaVlmRmsNormQHeadRowsSite",
        "kHyVlaVlmRmsNormKHeadRowsSite",
    ):
        assert HY_CPP.count(site) >= 3
    assert "HyVlaVlmModel non-BASE RMSNorm route requires a COMPLETE" in HY_CPP


def test_wall_text_fixed32_is_profile_scoped_and_descriptor_consumed():
    setter = _between(
        QWEN3, "void set_wall_prefill_routes(", "struct LayerWeights"
    )
    assert "BASE or exact FIXED32" in setter
    assert "set before weights" in setter
    assert "hybrid_w8a16" in QWEN3
    assert "kCausalDecoderRmsNormRowsSite" in QWEN3
    assert "FmbRouteFamily::NORMALIZATION" in QWEN3
    assert "consume_wall_rmsnorm_route" in QWEN3
    configure = _between(
        WALL_LLM, "def _configure_causal_decoder_handle(", "class WallOssLLM"
    )
    assert configure.index("causal_decoder_set_wall_prefill_routes") < configure.index(
        "set_weights(handle"
    )
    assert "rmsnorm_fixed32: bool = False" in WALL_LLM
    assert '{"rmsnorm_fixed32": True}' in WALL_RUNTIME
    # FIXED32 is an eight-lane route; reduced RTC budgets keep BASE.
    assert 'if budget == 8 and (w8a16 or (fp16_transfer_mask & 2)) else {}' in WALL_RUNTIME


def test_wall_vision_fixed32_has_exact_pre_and_compute_receipts():
    setter = _between(
        QWEN25, "void set_execution_routes(", "VisionDispatch resolve_vision_dispatch"
    )
    assert "BASE or exact" in setter and "FIXED32" in setter
    assert "exact MIMC, mixed" in setter
    assert "kQwen25ExactPreRmsNormRowsSite" in QWEN25
    assert "kQwen25ExactComputeRmsNormRowsSite" in QWEN25
    assert "consume_exact_rmsnorm_route" in QWEN25
    configure = _between(
        WALL_VISION,
        "def _configure_qwen25vl_vision_handle(",
        "class WallOssVision",
    )
    assert configure.index("qwen25vl_vision_set_execution_routes") < configure.index(
        "set_weights(handle"
    )
    assert "rmsnorm_fixed32: bool = False" in WALL_VISION
    assert "**quant_kwargs, **rmsnorm_kwargs" in WALL_RUNTIME


def test_wall_mixed_rmsnorm_is_explicitly_planned_and_consumed():
    assert QWEN3.count(
        "rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel("
    ) == QWEN3.count("consume_wall_mixed_rmsnorm_route(") - 1 == 4
    assert QWEN25.count(
        "rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel("
    ) == QWEN25.count("consume_mixed_rmsnorm_route(") - 1 == 5
    for source, sites in (
        (QWEN3, (
            "CAUSAL_DECODER_P640_MIXED_RMSNORM_SITE",
            "CAUSAL_DECODER_INPUT_MIXED_RMSNORM_SITE",
            "CAUSAL_DECODER_POST_MIXED_RMSNORM_SITE",
            "CAUSAL_DECODER_FINAL_MIXED_RMSNORM_SITE",
        )),
        (QWEN25, (
            "QWEN25_EXACT_COMPUTE_MIXED_RMSNORM_SITE",
            "QWEN25_INPUT_MIXED_RMSNORM_SITE",
            "QWEN25_POST_MIXED_RMSNORM_SITE",
            "QWEN25_MERGER_MIXED_RMSNORM_SITE",
            "QWEN25_EXACT_MERGER_MIXED_RMSNORM_SITE",
        )),
    ):
        assert all(source.count(site) >= 3 for site in sites)
    mixed_launcher = RMSNORM_CPP[
        RMSNORM_CPP.index("void rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel("):
    ]
    assert "RpuMixedRmsNormSpmRoute route" in mixed_launcher
    assert "route == expected" in mixed_launcher
    assert "const char* kernel_name" not in mixed_launcher


def test_wall_has_no_dead_rmsnorm_environment_write_or_rollback_claim():
    assert 'os.environ["RPU_RMSNORM_NEWTON"]' not in WALL_RUNTIME
    assert 'setdefault("RPU_RMSNORM_VWARP"' not in WALL_RUNTIME
    assert 'key in {"RPU_RMSNORM_NEWTON", "RPU_RMSNORM_VWARP"}' in WALL_API
