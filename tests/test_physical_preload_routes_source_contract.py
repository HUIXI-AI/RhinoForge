"""Board-free contracts for descriptor-owned conditional preload routes."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / "src/fused" / name).read_text(encoding="utf-8")


def test_conditional_preload_routes_publish_manifest_and_consumer_receipts():
    cases = {
        "rpu_qwen3vl_vision_model.cpp": (
            ("QWEN3VL_ROPE_COS_PRELOAD_PIPELINE_SITE", "7865795941291055524LL"),
            ("QWEN3VL_ROPE_SIN_PRELOAD_PIPELINE_SITE", "7646375323102352740LL"),
            ("QWEN3VL_ROPE_COS_PRELOAD_DIRECT_SITE", "1089927205617401262LL"),
            ("QWEN3VL_ROPE_SIN_PRELOAD_DIRECT_SITE", "9144463814971908982LL"),
            ("QWEN3VL_NVFP4_SCALE_PRELOAD_SITE", "2325782528529170216LL"),
        ),
        "rpu_qwen25vl_vision_model.cpp": (
            ("QWEN25_ROPE_COS_PRELOAD_SITE", "6623618708631996252LL"),
            ("QWEN25_ROPE_SIN_PRELOAD_SITE", "3659110372012461435LL"),
        ),
        "rpu_gr00t_dit_model.cpp": (
            ("GR00T_DIT_DENOISE_TABLE_PRELOAD_SITE", "1729441700529011334LL"),
            ("GR00T_DIT_DENOISE_BIAS_PRELOAD_SITE", "941780559855581702LL"),
        ),
        "rpu_lingbot_v2_moe_model.cpp": (
            ("L2_ROUTER_PADDING_DMA", "8386219696894325645LL"),
        ),
        "rpu_qwen3_5_vision_model.cpp": (
            ("QWEN35_VISION_TEMPORAL_PE_PRELOAD_SITE", "8539869012261735791LL"),
            ("QWEN35_VISION_MERGER_NORM_G_PRELOAD_SITE", "5543483196120457049LL"),
            ("QWEN35_VISION_MERGER_NORM_B_PRELOAD_SITE", "1422421047835217586LL"),
            ("QWEN35_VISION_MERGER_FC1_BIAS_PRELOAD_SITE", "8860010505634381437LL"),
            ("QWEN35_VISION_MERGER_FC2_BIAS_PRELOAD_SITE", "5828121683551566290LL"),
        ),
    }

    for filename, site_ids in cases.items():
        source = _read(filename)
        assert "FmbPhysicalManifestState::COMPLETE" in source
        assert "FmbRouteFamily::MUTABLE_DMA" in source
        assert "consume_physical_route(" in source
        for name, site_id in site_ids:
            assert site_id in source
            assert source.count(name) >= 3

    qwen3vl = _read("rpu_qwen3vl_vision_model.cpp")
    assert "physical_manifest_domain_for_candidate(" in qwen3vl
    assert "FMB_ROUTE_FLAG_ROPE_TABLE_DDR" in qwen3vl
    assert "FMB_ROUTE_FLAG_ROPE_TABLE_SPM" in qwen3vl
    assert "selected_rope_residency(" in qwen3vl
    assert "emit_rope_table_preload" in qwen3vl
    assert "StorageClass::Temp" in qwen3vl
    assert "StorageClass::Persistent" in qwen3vl
    assert "pipeline ? QWEN3VL_ROPE_COS_PRELOAD_PIPELINE_SITE" in qwen3vl
    assert "pipeline ? QWEN3VL_ROPE_SIN_PRELOAD_PIPELINE_SITE" in qwen3vl
    # NVFP4 tables: one record and one callback receipt per projection role.
    assert qwen3vl.count("{static_cast<int64_t>(role), elements, 2 * elements}") == 2

    qwen25vl = _read("rpu_qwen25vl_vision_model.cpp")
    assert qwen25vl.count(
        "{rope_spm_enabled_ ? 1 : 0, max_hw_, head_dim()}"
    ) >= 4

    gr00t = _read("rpu_gr00t_dit_model.cpp")
    assert "{denoise_ready_ ? 1 : 0, num_steps_, elements}" in gr00t
    assert "{denoise_ready_ ? 1 : 0, num_steps_, elems}" in gr00t
    assert "cfg.body_iterations = loop_mode_ ? num_steps_ : 1" in gr00t
    assert "route_invocation" in gr00t

    lingbot = _read("rpu_lingbot_v2_moe_model.cpp")
    assert "{logical_len, tp_rows_}" in lingbot
    assert "{seq_len, Tp}" in lingbot
