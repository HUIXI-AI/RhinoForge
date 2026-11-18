"""Board-free gates for exact multiview Vision Dense/W8 admission."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISION_MODEL = ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp"
FUSED_MODEL_BASE = ROOT / "src/core/fused_model_base.cpp"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _braced_block(source: str, marker: str) -> str:
    start = source.index(marker)
    # Default arguments may contain {} before the actual function body.
    parentheses = 0
    for opening in range(start, len(source)):
        char = source[opening]
        if char == "(":
            parentheses += 1
        elif char == ")":
            parentheses -= 1
        elif char == "{" and parentheses == 0:
            break
    else:
        raise AssertionError(f"missing C++ block after {marker!r}")
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated C++ block after {marker!r}")


def test_braced_block_skips_default_arguments_but_keeps_actual_body():
    for declaration in (
        "void prepare(int n, at::IntArrayRef selected = {})",
        "void prepare(int n) const",
        "enum class Profile",
    ):
        block = declaration + " { body { nested; } end; }"
        assert _braced_block(block + " void next() {}", declaration.split("(")[0]) == block


def _hex_constant(source: str, name: str) -> int:
    match = re.search(
        rf"constexpr uint64_t {name}\s*=\s*UINT64_C\((0x[0-9a-fA-F]+)\)",
        source,
    )
    assert match is not None, f"missing hexadecimal constant {name}"
    return int(match.group(1), 16)


def test_multiview_linear_profile_is_exactly_dense_fp16_or_full_w8a16():
    source = _source(VISION_MODEL)
    profile_enum = _braced_block(source, "enum class MultiviewLinearProfile")
    assert "DenseFp16 = 1" in profile_enum
    assert "W8A16 = 2" in profile_enum
    classify = _braced_block(
        source, "MultiviewLinearProfile classify_multiview_linear_profile() const"
    )

    assert "layer_weights_.size() == 24" in classify
    assert "MultiviewLinearProfile::DenseFp16" in classify
    assert "MultiviewLinearProfile::W8A16" in classify
    assert "const MultiviewLinearProfile expected" in classify
    assert "actual == expected" in classify
    assert "mixed/partial" in classify

    for weight, scale in (
        ("q_w", "q_ws"),
        ("k_w", "k_ws"),
        ("v_w", "v_ws"),
        ("o_w", "o_ws"),
        ("fc1_w", "fc1_ws"),
        ("fc2_w", "fc2_ws"),
    ):
        assert f"weights.{weight}" in classify
        assert f"weights.{scale}" in classify


def test_multiview_w8_owner_and_scale_contract_is_fail_closed():
    source = _source(VISION_MODEL)
    classify = _braced_block(
        source, "MultiviewLinearProfile classify_multiview_linear_profile() const"
    )

    # Exact kChar admits signed W8 and, by equality, excludes kByte/W4 storage.
    assert "weight.scalar_type() == at::kChar" in classify
    assert "weight.scalar_type() == at::kHalf" in classify
    assert "weight.device().type() == at::kPrivateUse1" in classify
    assert "weight.layout() == c10::Layout::Strided" in classify
    assert "weight.is_contiguous()" in classify
    assert "weight.dim() == 2" in classify
    assert "weight.size(0) == out_rows" in classify
    assert "weight.size(1) == in_cols" in classify
    assert "weight.numel() == out_rows * in_cols" in classify

    assert "scale.scalar_type() == at::kHalf" in classify
    assert "scale.device().type() == at::kPrivateUse1" in classify
    assert "scale.layout() == c10::Layout::Strided" in classify
    assert "scale.is_contiguous()" in classify
    assert "scale.dim() == 1" in classify
    assert "scale.numel() == out_rows" in classify
    assert "kByte/W4 and" in classify
    assert "other dtypes are rejected" in classify

    # q/k/v/o are HxH; fc1 is IxH and fc2 is HxI.  These exact row counts
    # also pin every scale to the corresponding output-channel count.
    for weight, scale, rows, cols, role in (
        ("q_w", "q_ws", "attn", "h", "q_w"),
        ("k_w", "k_ws", "attn", "h", "k_w"),
        ("v_w", "v_ws", "attn", "h", "v_w"),
        ("o_w", "o_ws", "h", "attn", "o_w"),
        ("fc1_w", "fc1_ws", "inter", "h", "fc1_w"),
        ("fc2_w", "fc2_ws", "h", "inter", "fc2_w"),
    ):
        pattern = rf"weights\.{weight},\s*weights\.{scale},\s*{rows},\s*{cols},\s*\"{role}\""
        assert re.search(pattern, classify)


def test_w8_scale_inventory_is_all_or_none_before_tensorlist_indexing():
    source = _source(VISION_MODEL)
    start = source.index("void set_weights(")
    setter = source[start : source.index("void set_rope_tables(", start)]

    any_scale = setter.index("const bool any_scale")
    all_or_none_gate = setter.index("if (any_scale)", any_scale)
    attach = setter.index("lw.q_ws")
    assert any_scale < attach
    for scale in (
        "q_scale_list",
        "k_scale_list",
        "v_scale_list",
        "o_scale_list",
        "fc1_scale_list",
        "fc2_scale_list",
    ):
        assert f"!{scale}.empty()" in setter[any_scale:all_or_none_gate]
        assert f'check_list({scale}, "{scale}")' in setter[any_scale:attach]


def test_standalone_pooler_remains_dense_only_while_multiview_admits_w8():
    source = _source(VISION_MODEL)
    standalone = _braced_block(source, "void validate_pooler_z1_profile() const")
    multiview = _braced_block(
        source, "void validate_multiview_composite_profile() const"
    )
    typed = _braced_block(
        source, "MultiviewLinearProfile validate_pooler_z1_profile("
    )

    assert "/*allow_w8a16=*/false" in standalone
    assert "/*allow_w8a16=*/true" in multiview
    assert "allow_w8a16 ||" in typed
    assert "linear_profile == MultiviewLinearProfile::DenseFp16" in typed
    assert "standalone pooler Z1 rejects W8A16" in typed


def test_multiview_typed_validation_precedes_physical_component_prepare():
    source = _source(VISION_MODEL)
    validate = _braced_block(
        source, "void validate_multiview_composite_profile() const"
    )
    prepare = _braced_block(
        source, "SpmPipelineComponentLayout prepare_multiview_composite_layout("
    )
    dry_prepare = _braced_block(
        source, "SpmPipelineComponentLayout prepare_multiview_dry_layout("
    )
    typed = _braced_block(
        source, "MultiviewLinearProfile validate_pooler_z1_profile("
    )

    assert "/*allow_w8a16=*/true" in validate
    assert "classify_multiview_linear_profile()" in typed
    assert prepare.index("validate_multiview_composite_profile();") < prepare.index(
        "compose_fmb_default_three_stage_chunk_plan("
    )
    assert "ChunkMode::KV_FIRST" in prepare
    assert "prepare_spm_pipeline_component(" in prepare
    assert dry_prepare.index(
        "validate_multiview_composite_profile();"
    ) < dry_prepare.index("compose_fmb_default_three_stage_chunk_plan(")
    assert "ChunkMode::KV_FIRST" in dry_prepare
    assert "prepare_spm_pipeline_component_for_cpu_contract(" in dry_prepare


def test_multiview_quant_mode_changes_resolved_profile_hash_identity():
    vision = _source(VISION_MODEL)
    fmb = _source(FUSED_MODEL_BASE)
    fp16_name = "QWEN3VL_MULTIVIEW_FP16_PROFILE_VERSION"
    w8_name = "QWEN3VL_MULTIVIEW_W8A16_PROFILE_VERSION"

    fp16_version = _hex_constant(vision, fp16_name)
    w8_version = _hex_constant(vision, w8_name)
    assert fp16_version != 0
    assert w8_version != 0
    assert fp16_version != w8_version

    resolve = _braced_block(vision, "resolve_multiview_composite_profile()")
    assert "classify_multiview_linear_profile()" in resolve
    assert "request.version" in resolve
    assert "MultiviewLinearProfile::DenseFp16" in resolve
    assert fp16_name in resolve
    assert w8_name in resolve

    hash_builder = _braced_block(
        fmb, "FusedModelBase::resolve_spm_pipeline_execution_profile_impl("
    )
    assert "profile_hash = fmb_hash_u64(profile_hash, request.version);" in hash_builder


def test_multiview_vision_scratch_geometry_is_unchanged_by_weight_storage():
    source = _source(VISION_MODEL)
    prepare = _braced_block(
        source, "SpmPipelineComponentLayout prepare_multiview_composite_layout("
    )

    assert (
        "constexpr size_t QWEN3VL_MULTIVIEW_VISION_SCRATCH_BYTES = 1966080;"
        in source
    )
    assert (
        "prepared.temporary_bytes ==\n"
        "                QWEN3VL_MULTIVIEW_VISION_SCRATCH_BYTES"
        in prepare
    )
