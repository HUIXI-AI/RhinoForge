"""Board-free contract for Qwen2.5-VL physical capability digests."""

from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/fused/rpu_qwen25vl_vision_model.cpp"
).read_text(encoding="utf-8")


def _between(begin: str, end: str) -> str:
    start = SOURCE.index(begin)
    return SOURCE[start : SOURCE.index(end, start)]


def test_qwen25vl_route_arguments_digest_all_four_physical_controls() -> None:
    manifest = _between(
        "physical_manifest_for_candidate(", "physical_manifest_forward_capability("
    )
    exact = _between("void emit_exact_mimc_qkv_body", "void build_layer_subgraph")
    generic = _between("void build_layer_subgraph", "void emit_merger_chunk")
    generic_attention = _between(
        "std::vector<int64_t> generic_attention_arguments(",
        "bool generic_attention_raw_valid(",
    )
    preload = _between("void emit_preload_weights()", "void emit_exact_mimc_layer_params")

    rope_arg = "{rope_spm_enabled_ ? 1 : 0}"
    preload_arg = "{rope_spm_enabled_ ? 1 : 0, max_hw_, head_dim()}"
    attention_schedule = "{static_cast<int64_t>(schedule_),"
    attention_window = "per_window_sdpa_enabled_ ? 1 : 0}"
    mixed_arg = "{mixed_precision_ ? 1 : 0}"

    for body in (manifest, exact):
        assert rope_arg in body
        assert attention_schedule in body
        assert attention_window in body
    assert "generic_attention_arguments(site)" in manifest
    assert "consume_generic_attention_route(" in generic
    assert "static_cast<int64_t>(schedule_)" in generic_attention
    assert attention_window in generic_attention
    assert rope_arg in generic
    vision_row = _between(
        "std::vector<int64_t> vision_row_arguments(",
        "std::vector<int64_t> small_image_gate_up_arguments(",
    )
    assert manifest.count("vision_row_arguments(") == 4
    assert vision_row.count("mixed_precision_ ? 1 : 0") >= 2
    assert exact.count("vision_row_arguments(") == 2
    assert generic.count("vision_row_arguments(") == 2
    assert manifest.count(preload_arg) == 2
    assert preload.count(preload_arg) == 2
    assert "FMB_ROUTE_FLAG_ROPE_TABLE_SPM" in manifest
    assert "FMB_ROUTE_FLAG_ROPE_TABLE_DDR" in manifest
    assert "rope_spm_layout" in SOURCE

    assert "resolved_flags,\n            resolved_arguments, invocation" in SOURCE
    assert "std::move(arguments), invocation" in manifest
