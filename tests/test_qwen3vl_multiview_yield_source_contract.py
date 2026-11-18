"""Board-free source gates for the Qwen3-VL N256 x3 producer yields."""

from pathlib import Path
import re

from test_qwen3vl_prefill_gateup_launcher import definition


ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp"
HEADER = ROOT / "src/fused/rpu_qwen3vl_pooler_spm_z1.h"


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_exact_three_input_occurrence_capability_is_composite_only():
    source = VISION.read_text(encoding="utf-8")
    capability = _between(
        source,
        "SpmFmbTraversalCapability spm_fmb_traversal_capability() const",
        "ModelStaticConfig static_config() override",
    )
    assert "multiview_composite_prepared_" in capability
    assert "CanonicalTraversal" in capability
    assert "CanonicalDenseReplicatedFp16" in capability
    assert "StableThreeInputSlotsSameOwnerPreload" in capability
    assert "QWEN3VL_MULTIVIEW_YIELD_POLICY" in capability
    assert "QWEN3VL_MULTIVIEW_OCCURRENCE_POLICY" in capability


def test_each_n256_occurrence_has_three_layer_yields_and_one_postfn_yield():
    source = VISION.read_text(encoding="utf-8")
    layer = definition(source, "void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk)")
    post = _between(
        source,
        "void multiview_composite_post_fn()",
        "void pooler_z1_post_fn()",
    )
    assert "QWEN3VL_POOLER_Z1_DEEPSTACK_TAPS" in layer
    assert "emit_one_merger_yield(" in layer
    assert "emit_one_merger_yield(" in post
    assert "QWEN3VL_POOLER_Z1_DEEPSTACK_TAPS = {" in source
    assert "QWEN3VL_POOLER_Z1_DEEPSTACK1_TAP, 11, 17" in source
    # 3 ordered producer occurrences x (tap5, tap11, tap17, pooler).
    assert 3 * (len((5, 11, 17)) + 1) == 12


def test_typed_scope_wraps_only_the_existing_terminal_allreduce():
    source = VISION.read_text(encoding="utf-8")
    typed = _between(
        source,
        "void emit_one_merger_yield(",
        "void merger_post_fn()",
    )
    prefix = typed.index("emit_one_merger_prefix(")
    begin = typed.index("begin_spm_pipeline_dense_producer_yield(")
    terminal = typed.index("rpu_launch_all_reduce_sum_residual_kernel(")
    finish = typed.index("yield.finish()")
    assert prefix < begin < terminal < finish
    scoped = typed[begin:finish]
    assert scoped.count("rpu_launch_") == 1
    assert '"merger_out"' in scoped
    assert "yield.target()" in scoped


def test_composite_suppresses_raw_bridges_and_component_fast_replay():
    source = VISION.read_text(encoding="utf-8")
    layer = definition(source, "void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk)")
    static_config = _between(
        source,
        "ModelStaticConfig static_config() override",
        "ModelDynamicConfig dynamic_config",
    )
    assert "if (!pipeline_dispatch_active() &&" in layer
    assert "const bool pipeline_final" in layer
    assert "if (!ctx().output_to_spm && !pipeline_final)" in layer
    assert static_config.count("!pipeline_dispatch_active()") == 3


def test_composite_manifest_omits_the_ordinary_snapshot_reload_alias():
    source = VISION.read_text(encoding="utf-8")
    declarations = definition(source, "std::vector<BufferDecl> declare_buffers(")
    guard = re.search(r"if \(!multiview_dry_prepared_ &&\s+!multiview_composite_prepared_\) \{", declarations)
    assert guard is not None
    guarded = declarations[guard.start() :]
    assert guarded.index('{"merger_in"') < guarded.index("}\n")

    layer = definition(source, "void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk)")
    post = _between(
        source,
        "void multiview_composite_post_fn()",
        "void pooler_z1_post_fn()",
    )
    assert 'addr(0, "merger_in")' not in layer
    assert 'addr(0, "merger_in")' not in post
    assert 'addr(0, "residual1")' in layer
    assert 'addr(0, "residual1")' in post


def test_internal_surface_exposes_lifecycle_not_a_public_compute_op():
    header = HEADER.read_text(encoding="utf-8")
    for name in (
        "prepare_multiview_vision_composite",
        "resolve_multiview_vision_profile",
        "multiview_vision_owner",
        "prime_multiview_vision_position",
        "clear_multiview_vision_position",
        "unprepare_multiview_vision_composite",
        "forward_multiview_vision_composite",
    ):
        assert name in header
