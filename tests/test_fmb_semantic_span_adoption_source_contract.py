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
    forward = _section(
        source,
        "    at::Tensor forward_impl(\n",
        "    at::Tensor pop_pooler_merged()",
    )
    _assert_common_span_call(forward)
    assert "num_patches_in / image_batch_count_" in forward
    assert "image < image_batch_count_" in forward
    assert "image * patches_per_image" in forward


def test_hyvit2_packed_images_publish_one_span_per_image():
    source = _source("rpu_hyvit2_vision_model.cpp")
    forward = _section(
        source,
        "    at::Tensor forward_packed(\n",
        "protected:\n    // ================================================",
    )
    _assert_common_span_call(forward)
    assert "seq_len / image_batch_count_" in forward
    assert "image < image_batch_count_" in forward
    assert "input_chunks.push_back(" in forward
    assert "image * rows_per_image" in forward


def test_qwen35_temporal_vision_publishes_one_span_per_camera():
    source = _source("rpu_qwen3_5_vision_model.cpp")
    forward = _section(
        source,
        "at::Tensor Qwen3_5VisionModel::forward(\n",
        "void Qwen3_5VisionModel::emit_step0()",
    )
    _assert_common_span_call(forward)
    assert "layer_rows / current_camera_batch_count_" in forward
    assert "current_compact_step0_" in forward
    assert "G05_COMPACT_STEP0_ROWS" in forward
    assert "current_temporal_num_frames_ * temporal_patches_per_frame_" in forward
    assert "QWEN3_5_VISION_STEP0_CHUNK" in forward
    assert "camera < current_camera_batch_count_" in forward
    assert "camera * rows_per_camera" in forward


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
