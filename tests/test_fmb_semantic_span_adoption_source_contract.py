"""Source gate for packed sequence-axis users of the common FMB span API."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(name: str) -> str:
    return (ROOT / "src" / "fused" / name).read_text(encoding="utf-8")


def _section(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    return source[begin : source.index(end, begin)]


def _assert_common_span_call(section: str) -> None:
    assert "std::vector<ChunkInfo> input_chunks" in section
    assert "std::vector<FmbExecutionSpan> spans" in section
    assert "input_chunks, spans" in section


def test_qwen3vl_packed_images_publish_one_span_per_image():
    source = _source("rpu_qwen3vl_vision_model.cpp")
    planner = _section(
        source,
        "    std::vector<int64_t> resolve_stage_domain(\n",
        "    // Expose the keepalive base",
    )
    _assert_common_span_call(planner)
    assert "num_patches / image_batch_count_" in planner
    assert "image < image_batch_count_" in planner
    assert "image * patches_per_image" in planner
    assert "resolve_prefill_stage_domain_for_shape(" in planner
    forward = _section(source, "    at::Tensor forward_impl(\n",
                       "    SpmPipelineComponentLayout prepare_pooler_z1_layout(")
    assert "/*is_causal=*/false, /*planned_chunk_size=*/0,\n            planned_stage_descriptor);" in forward
    assert "resolve_stage_domain(" not in forward


def test_hyvit2_packed_images_publish_one_span_per_image():
    source = _source("rpu_hyvit2_vision_model.cpp")
    forward = _section(
        source,
        "    at::Tensor forward_packed(\n",
        "    std::vector<int64_t> resolve_stage_domain(\n",
    )
    _assert_common_span_call(forward)
    assert "seq_len / image_batch_count_" in forward
    assert "image < image_batch_count_" in forward
    assert "input_chunks.push_back(" in forward
    assert "image * rows_per_image" in forward


def test_qwen35_public_vision_publishes_one_checked_image_span():
    source = _source("rpu_qwen3_5_vision_model.cpp")
    planner = _section(
        source,
        "Qwen3_5VisionModel::resolve_execution_plan(int64_t execution_len)",
        "std::vector<std::vector<int64_t>> Qwen3_5VisionModel::resolve_prefill_domain(",
    )
    assert "auto input_chunks = make_vision_chunks(execution_len, input_chunk_size);" in planner
    assert "std::vector<FmbExecutionSpan> spans" in planner
    assert "execution_len / current_camera_batch_count_" in planner
    assert "camera < current_camera_batch_count_" in planner
    assert "camera * rows_per_camera" in planner
    assert "compose_fmb_three_stage_chunk_plan(" in planner
    assert "std::move(compute_chunks), std::move(spans), ChunkMode::KV_FIRST" in planner
    geometry = _section(
        source,
        "int64_t Qwen3_5VisionModel::input_chunk_size_for_geometry(",
        "at::Tensor Qwen3_5VisionModel::forward(\n",
    )
    admission = _section(
        source,
        "void Qwen3_5VisionModel::configure_execution_geometry(",
        "int64_t Qwen3_5VisionModel::input_chunk_size_for_geometry(",
    )
    assert "temporal_num_frames == 1 && camera_batch_count == 1 && !compact_input" in admission
    assert "current_camera_batch_count_ = 1;" in admission
    assert "current_output_patches_ = execution_len;" in admission
    assert "QWEN3_5_VISION_STEP0_CHUNK" in geometry
    assert "input_chunk_size_for_geometry(execution_len)" in planner
    forward = _section(
        source,
        "at::Tensor Qwen3_5VisionModel::forward(\n",
        "SpmPipelineComponentLayout Qwen3_5VisionModel::prepare_z2_layout(",
    )
    assert "execution_plan.stages.input.chunks" in forward
    assert "execution_plan.stages.spans" in forward
    assert "/*planned_chunk_size=*/0, planned_stage_descriptor" in forward
    assert "fmb_three_stage_chunk_plan_fingerprint(ctx().stage_plan) ==" in forward
    assert "fmb_three_stage_chunk_plan_fingerprint(execution_plan.stages)" in forward


def test_physical_component_prepare_requires_the_exact_stage_plan():
    header = (ROOT / "src/core/fused_model_base.h").read_text(encoding="utf-8")
    core = (ROOT / "src/core/fused_model_base.cpp").read_text(encoding="utf-8")
    assert "const FmbThreeStageChunkPlan& stage_plan" in header
    assert "prepare_spm_pipeline_component_impl(" in header
    assert "prepare_spm_pipeline_component_for_cpu_contract_impl(" in header
    assert "validate_fmb_three_stage_chunk_plan(stage_plan);" in core
    assert "exact.stage_plan_fingerprint = fingerprint;" in core
    assert "result.stage_plan = std::move(stage_plan);" in core

    fused = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src/fused").glob("*.cpp")
    )
    bare_prepare = re.compile(
        r"prepare_spm_pipeline_component(?:_for_cpu_contract)?\(\s*"
        r"(?:layout|shape\.allocation_layout)\s*\)"
    )
    assert bare_prepare.search(fused) is None
