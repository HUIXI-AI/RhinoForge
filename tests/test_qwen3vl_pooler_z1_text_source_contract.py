"""Board-free source contract for the bounded Qwen3-VL pooler Z1 Text slice."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src/fused/rpu_qwen3_model.h"
SOURCE = ROOT / "src/fused/rpu_qwen3_model.cpp"


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_z1_text_selected_winner_is_validated_before_mutable_prime():
    source = SOURCE.read_text(encoding="utf-8")
    prepare = _between(source, "CausalDecoderModel::prepare_qwen3vl_pooler_z1_layout(",
                       "SpmDense2DSpec CausalDecoderModel::qwen3vl_pooler_z1_destination_spec(")
    assert 'kvinsert_cost_domain("causal_decoder", selected_descriptor)' in prepare
    assert "kvinsert_cost_owner_prefix(prepared_descriptor)" in prepare
    assert "fmb_three_stage_chunk_plan_fingerprint(candidate.stage_plan)" in prepare
    assert "layout = bind_physical_layout_context(layout, manifest)" in prepare
    assert prepare.index("prepare_stage_candidate(prepared_descriptor)") < prepare.index(
        "bind_physical_layout_context(layout, manifest)")
    assert "qwen3vl_pooler_z1_selected_stage_descriptor_.assign(" in prepare
    prime = _between(source, "void CausalDecoderModel::prime_qwen3vl_pooler_z1_inputs(",
                     "std::vector<int64_t> CausalDecoderModel::qwen3vl_pooler_z1_stage_descriptor(")
    assert prime.index("qwen3vl_pooler_z1_stage_descriptor(partial_mrope)") < prime.index("position_dst.copy_(position_ids)")
    assert "qwen3vl_pooler_z1_stage_descriptor_ = stage_descriptor" in prime
    getter = _between(source, "std::vector<int64_t> CausalDecoderModel::qwen3vl_pooler_z1_stage_descriptor(",
                      "void CausalDecoderModel::rollback_qwen3vl_pooler_z1_inputs()")
    assert getter.index("return qwen3vl_pooler_z1_selected_stage_descriptor_") < getter.index("encode_causal_physical_stage_descriptor(")


def test_exact_qwen3vl_2b_profile_and_real_c128_layout_are_sealed():
    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    prepare = _between(
        source,
        "CausalDecoderModel::prepare_qwen3vl_pooler_z1_layout(",
        "SpmDense2DSpec CausalDecoderModel::qwen3vl_pooler_z1_destination_spec(",
    )
    profile = _between(
        source,
        "void CausalDecoderModel::validate_qwen3vl_pooler_z1_model_profile()",
        "CausalDecoderModel::prepare_qwen3vl_pooler_z1_layout(",
    )

    for declaration in (
        "QWEN3VL_POOLER_Z1_CANARY_LAYERS = 28",
        "QWEN3VL_POOLER_Z1_CANARY_HIDDEN = 2048",
        "QWEN3VL_POOLER_Z1_CANARY_INTERMEDIATE = 6144",
        "QWEN3VL_POOLER_Z1_CANARY_Q_HEADS = 16",
        "QWEN3VL_POOLER_Z1_CANARY_KV_HEADS = 8",
        "QWEN3VL_POOLER_Z1_CANARY_HEAD_DIM = 128",
        "QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN = 128",
        "QWEN3VL_POOLER_Z1_CANARY_REAL_LEN = 72",
        "QWEN3VL_POOLER_Z1_CANARY_CHUNK_SIZE = 128",
        "QWEN3VL_POOLER_Z1_CANARY_TEMPORARY_BYTES =\n        1966080",
    ):
        assert declaration in header
    assert "mrope_section_[0] == 24" in profile
    assert "mrope_section_[1] == 20" in profile
    assert "mrope_section_[2] == 20" in profile
    assert "deepstack_lang_layers_.empty()" in profile
    assert "require_fp16_matrix_min(weights.q_w" in profile
    assert "compose_fmb_default_three_stage_chunk_plan(" in prepare
    assert "ChunkMode::SEQUENTIAL" in prepare
    assert "prepare_spm_pipeline_component(" in prepare
    assert "prepared.temporary_bytes ==" in prepare
    assert "QWEN3VL_POOLER_Z1_CANARY_TEMPORARY_BYTES" in prepare

    destination = _between(
        source,
        "SpmDense2DSpec CausalDecoderModel::qwen3vl_pooler_z1_destination_spec(",
        "void CausalDecoderModel::validate_qwen3vl_pooler_z1_position_ids(",
    )
    assert "spec.rows = QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN" in destination
    assert "spec.cols = QWEN3VL_POOLER_Z1_CANARY_HIDDEN" in destination
    assert "spec.validate();" in destination


def test_position_and_optional_partial_mrope_are_primed_before_adopt():
    source = SOURCE.read_text(encoding="utf-8")
    prime = _between(
        source,
        "void CausalDecoderModel::prime_qwen3vl_pooler_z1_inputs(",
        "void CausalDecoderModel::rollback_qwen3vl_pooler_z1_inputs()",
    )
    adopt = _between(
        source,
        "void CausalDecoderModel::adopt_qwen3vl_pooler_z1_layout(",
        "void CausalDecoderModel::bind_qwen3vl_pooler_z1_destination(",
    )
    forward = _between(
        source,
        "at::Tensor CausalDecoderModel::forward_qwen3vl_pooler_z1(",
        "uint32_t CausalDecoderModel::qwen3vl_pooler_z1_layer_input_residual_addr(",
    )

    assert "!RpuKernelGraph::has_active()" in prime
    assert "position_dst.copy_(position_ids);" in prime
    assert "rope_cos_il.has_value() == rope_sin_il.has_value()" in prime
    assert "cos_dst.copy_(*rope_cos_il);" in prime
    assert "sin_dst.copy_(*rope_sin_il);" in prime
    assert "ka_last_cos_src_ = cos_dst.data_ptr();" in prime
    assert "ka_last_sin_src_ = sin_dst.data_ptr();" in prime
    assert "qwen3vl_pooler_z1_partial_mrope_ = partial_mrope;" in prime
    assert "rpu_ddr_flush_force_sized(" in prime
    assert "ka_last_position_ = 0;" in prime
    assert "ka_last_seq_ = QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN;" in prime
    assert prime.index("qwen3vl_pooler_z1_inputs_primed_ = true") > prime.index(
        "rpu_ddr_flush_force_sized("
    )
    assert adopt.index("qwen3vl_pooler_z1_inputs_primed_") < adopt.index(
        "adopt_spm_pipeline_component(lease, scratch)"
    )
    assert "std::optional<at::Tensor> rope_cos_il = std::nullopt;" in forward
    assert "std::optional<at::Tensor> rope_sin_il = std::nullopt;" in forward
    assert "if (qwen3vl_pooler_z1_partial_mrope_)" in forward
    assert "rope_cos_il = cos_il_keepalive_.narrow(" in forward
    assert "rope_sin_il = sin_il_keepalive_.narrow(" in forward
    assert "/*rope_cos_il=*/rope_cos_il" in forward
    assert "/*rope_sin_il=*/rope_sin_il" in forward
    assert "/*deepstack_dense_visual_embeds=*/std::nullopt" in forward


def test_shared_selector_reads_typed_destination_and_skips_layer0_input_dma():
    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    bind = _between(
        source,
        "void CausalDecoderModel::bind_qwen3vl_pooler_z1_destination(",
        "void CausalDecoderModel::validate_qwen3vl_pooler_z1_layout(",
    )
    selector = _between(
        source,
        "CausalDecoderModel::select_z1_layer_input(",
        "void CausalDecoderModel::check_qwen3vl_pooler_z1_destroy_allowed()",
    )

    assert "destination.spec() == expected" in bind
    assert "destination.resolve_physical_addr(/*core=*/0, lease)" in bind
    assert "qwen3vl_pooler_z1_layer_input_residual_addr" in selector
    assert "wall_z1_layer_input_residual_addr" in selector
    assert "chunk.offset" in source[
        source.index("qwen3vl_pooler_z1_layer_input_residual_addr") :
        source.index("CausalDecoderModel::select_z1_layer_input(")
    ]
    assert header.count("select_z1_layer_input(layer_idx, chunk)") == 2
    assert header.count("!ctx().input_in_spm && !z1_input.external") == 3


def test_ordinary_and_wall_paths_keep_independent_fail_closed_state():
    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")

    assert "prepare_wall_oss_z1_layout" in header
    assert "forward_wall_oss_z1" in header
    assert "wall_z1_layer_input_residual_addr" in source
    assert "wall_z1_prepared_chunk_size_" in header
    assert "qwen3vl_pooler_z1_prepared_chunk_size_" in header
    assert "qwen3vl_pooler_z1_prepared_execution_len_ == 0 ||" in header
    assert "qwen3vl_pooler_z1_dispatch_" in header
    assert "if (wall_z1_dispatch_)" in header
    assert "plan.num_chunks == 1" in header
    assert "cfg.chunk_mode = ChunkMode::SEQUENTIAL" in header
    assert "cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT" in header


def test_rollback_unprepare_destroy_and_free_wrapper_contract_are_complete():
    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    rollback = _between(
        source,
        "void CausalDecoderModel::rollback_qwen3vl_pooler_z1_inputs()",
        "void CausalDecoderModel::unprepare_qwen3vl_pooler_z1_layout()",
    )
    unprepare = _between(
        source,
        "void CausalDecoderModel::unprepare_qwen3vl_pooler_z1_layout()",
        "void CausalDecoderModel::adopt_qwen3vl_pooler_z1_layout(",
    )
    destroy = _between(
        source,
        "void CausalDecoderModel::check_qwen3vl_pooler_z1_destroy_allowed()",
        "}  // namespace v3",
    )
    wrappers = _between(
        source,
        "namespace v3::qwen3vl_pooler_z1_internal {",
        "}  // namespace v3::qwen3vl_pooler_z1_internal",
    )

    assert "qwen3vl_pooler_z1_position_ids_addr_ = 0" in rollback
    assert "qwen3vl_pooler_z1_inputs_primed_ = false" in rollback
    assert "ka_last_position_ = -1" in rollback
    assert "rollback_qwen3vl_pooler_z1_inputs();" in unprepare
    assert "qwen3vl_pooler_z1_prepared_execution_len_ = 0" in unprepare
    assert "qwen3vl_pooler_z1_prepared_real_len_ = 0" in unprepare
    assert "qwen3vl_pooler_z1_prepared_chunk_size_ = 0" in unprepare
    assert "!qwen3vl_pooler_z1_dispatch_" in destroy
    assert "qwen3vl_pooler_z1_prepared_execution_len_ == 0" in destroy
    assert "v3::qwen3vl_pooler_z1_internal::check_text_destroy_allowed(handle);" in source

    for wrapper in (
        "prepare_text(",
        "prime_text_inputs(",
        "rollback_text_inputs(",
        "unprepare_text(",
        "text_destination_spec(",
        "adopt_text(",
        "bind_text_destination(",
        "validate_text(",
        "clear_text(",
        "forward_text_z1(",
        "check_text_destroy_allowed(",
    ):
        assert wrapper in wrappers
    assert "namespace v3::qwen3vl_pooler_z1_internal" in header
