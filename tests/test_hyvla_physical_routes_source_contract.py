"""Board-free contracts for Hy-VLA native physical-route receipts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / "src" / "fused" / name).read_text(encoding="utf-8")


def _assert_manifest_and_consumer(
    source: str, site_name: str, site_id: str
) -> None:
    assert site_id in source
    # declaration + candidate manifest + exact forward receipt
    assert source.count(site_name) >= 3


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start:source.index(end, start)]


def test_hyvla_expert_selectors_are_typed_and_consumed_at_launch_sites():
    source = _read("rpu_hyvla_expert_model.cpp")

    for family, field in (
        ("GRAPH_SCHEDULE", "cold_fast_replay_"),
        ("GRAPH_SCHEDULE", "cold_mask_once_"),
        ("NORMALIZATION", "cold_rmsnorm_pad16_"),
        ("ACTIVATION", "cold_silu_mul_"),
        ("KV_INSERT", "cold_kvpad16_"),
        ("ROPE", "cold_partial_rope_"),
    ):
        assert f"FmbRouteFamily::{family}" in source
        assert field in source

    for name, site_id in (
        ("HYVLA_EXPERT_QKV_LINEAR_SITE", "1236943476379916654LL"),
        ("HYVLA_EXPERT_Q_PARTIAL_ROPE_SITE", "4314852815466214180LL"),
        ("HYVLA_EXPERT_K_PARTIAL_ROPE_SITE", "6204149761964354754LL"),
        ("HYVLA_EXPERT_PRE_MO_MULTI_LINEAR_SITE", "2981602481703720578LL"),
        ("HYVLA_EXPERT_ATTN_ALL_REDUCE_SITE", "5454728429937755845LL"),
        ("HYVLA_EXPERT_MLP_ALL_REDUCE_SITE", "2337931566137374851LL"),
    ):
        _assert_manifest_and_consumer(source, name, site_id)

    assert "{cs, h, action_mlp_cores_, action_mlp_cores_}" in source
    assert "cos_sin_start, tp);" in source


def test_hyvla_vlm_mot_activation_and_bias_preload_have_exact_receipts():
    source = _read("rpu_hyvla_vlm_model.cpp")

    assert "FmbRouteFamily::NORMALIZATION" in source
    assert "static_cast<int64_t>(cold_mot_norm_nomerge_)" in source
    assert "FmbRouteFamily::ACTIVATION" in source
    assert "cold_silu_mul_ ? 1 : 0" in source
    assert "FmbRouteFamily::GRAPH_SCHEDULE" in source
    assert "cold_mask_once_ ? 1 : 0" in source

    _assert_manifest_and_consumer(
        source,
        "HYVLA_VLM_QKV_BIAS_PRELOAD_DMA_SITE",
        "7757347464819546080LL",
    )
    assert "{has_qkv_bias_ ? 1 : 0, bw.numel(), per_core," in source
    assert "bias_invocation" in source
    assert "cos_sin_start, tp);" in source


def test_hyvit2_encoder_routes_cover_q_staging_linears_ring_and_tail():
    source = _read("rpu_hyvit2_vision_model.cpp")

    for name, site_id in (
        ("HYVIT2_KV_Q_LINEAR_SITE", "3821976038473806707LL"),
        ("HYVIT2_KV_K_LINEAR_SITE", "8147072609028639932LL"),
        ("HYVIT2_KV_V_LINEAR_SITE", "8613419147536865054LL"),
        ("HYVIT2_Q_SAVE_DMA_SITE", "4520706683310192342LL"),
        ("HYVIT2_Q_LOAD_DMA_SITE", "2558943548608657790LL"),
        ("HYVIT2_O_LINEAR_SITE", "5411123264736285894LL"),
        ("HYVIT2_ATTN_ALL_REDUCE_SITE", "7296386757428037628LL"),
        ("HYVIT2_FC1_LINEAR_SITE", "816252863600784417LL"),
        ("HYVIT2_FC2_LINEAR_SITE", "2679807282578599669LL"),
        ("HYVIT2_MLP_ALL_REDUCE_SITE", "7190094110879491471LL"),
        ("HYVIT2_MERGER_PERMUTE1_SITE", "1256999960682954151LL"),
        ("HYVIT2_MERGER_PERMUTE2_SITE", "1906212928857069507LL"),
        ("HYVIT2_TAIL_SLOT_DMA_SITE", "3131457000840871027LL"),
        ("HYVIT2_TAIL_OUTPUT_DMA_SITE", "6474014664636894258LL"),
        ("HYVIT2_PROJECTOR_LINEAR_SITE", "7630442159474739795LL"),
    ):
        _assert_manifest_and_consumer(source, name, site_id)

    assert "fmb_ring_all_reduce_route_selector(chunk.len, h, num_cores())" in source
    assert "fmb_ring_all_reduce_route_selector(seq_len, h, num_cores())" in source
    assert "cold_config_.q_inplace ? 1 : 0" in source
    assert "image_batch_count_, seq_len, seq_len_" in source
    assert "merger_member_major_ ? 1 : 0" in source
    assert "proj1_in_merger_ ? 1 : 0" in source
    assert "FmbRouteFamily::GRAPH_SCHEDULE" in source
    assert "FmbRouteFamily::MUTABLE_DMA" in source
    assert "FmbLinearRouteSelector::AUTO_TILE" in source


def test_hyvit2_patch_prologue_shares_the_complete_body_manifest():
    source = _read("rpu_hyvit2_vision_model.cpp")

    for name, site_id in (
        ("HYVIT2_PATCH_INPUT_DMA_SITE", "6327224917790673981LL"),
        ("HYVIT2_PATCH_LINEAR_SITE", "5594838142388648123LL"),
        ("HYVIT2_PATCH_ALL_GATHER_SITE", "2358476529501646058LL"),
        ("HYVIT2_PATCH_POSITION_DMA_SITE", "2702679098023419389LL"),
        ("HYVIT2_PATCH_OUTPUT_DMA_SITE", "1374295323147233269LL"),
    ):
        _assert_manifest_and_consumer(source, name, site_id)

    assert "planning_external_patch_prologue_ = external_patch_prologue" in source
    assert "begin_external_physical_manifest_prologue(" in source
    assert "consume_external_prologue ? &ctx() : nullptr" in source
    assert "/*consume_external_prologue=*/false" in source

    runtime = (
        ROOT / "python" / "rpu_backend" / "adapters" / "hy_vla" / "runtime.py"
    ).read_text(encoding="utf-8")
    assert "bool(self._vit_packed and self._pe_in_graph)" in runtime
    assert "torch.ops.rpu.hyvit2_forward_multi(" in runtime


def test_hyvla_native_cold_selectors_have_one_python_snapshot_authority():
    native_sources = tuple(
        _read(name)
        for name in (
            "rpu_hyvla_expert_model.cpp",
            "rpu_hyvla_vlm_model.cpp",
            "rpu_hyvit2_vision_model.cpp",
        )
    )
    assert all("std::getenv" not in source for source in native_sources)
    assert all("cold_config_bound_" in source for source in native_sources)

    runtime = (
        ROOT / "python" / "rpu_backend" / "adapters" / "hy_vla" / "runtime.py"
    ).read_text(encoding="utf-8")
    assert "@dataclasses.dataclass(frozen=True)" in runtime
    assert "class _HyVlaNativeColdSnapshot:" in runtime
    assert runtime.count("_HyVlaNativeColdSnapshot.from_environment(effective)") == 1
    assert "native = _HyVlaNativeColdSnapshot.from_environment(effective)" in runtime
    assert "native=native" in runtime
    assert "_native_cold=cold_plan.native" in runtime

    for bind_name, config_op, weight_op in (
        ("def bind(h):", "hyvit2_set_runtime_config", "hyvit2_set_weights"),
        (
            "def bind_vlm(h):",
            "hyvla_vlm_set_runtime_config",
            "hyvla_vlm_set_weights",
        ),
        (
            "def bind_exp(h):",
            "hyvla_expert_set_runtime_config",
            "hyvla_expert_set_weights",
        ),
    ):
        start = runtime.index(bind_name)
        assert runtime.index(config_op, start) < runtime.index(weight_op, start)


def test_native_kv_launches_fail_before_side_effects_without_a_descriptor():
    bodies = (
        _between(
            _read("rpu_hyvit2_vision_model.cpp"),
            "void emit_kv_first_body(",
            "void build_layer_subgraph(",
        ),
        _between(
            _read("rpu_hyvla_expert_model.cpp"),
            "void HyVlaExpertModel::build_layer_subgraph(",
            "void HyVlaExpertModel::unroll_forward(",
        ),
        _between(
            _read("rpu_hyvla_vlm_model.cpp"),
            "void HyVlaVlmModel::build_layer_subgraph(",
            "at::Tensor HyVlaVlmModel::step_forward(",
        ),
    )
    for body in bodies:
        assert "requires a COMPLETE physical descriptor" in body
        assert body.index("requires a COMPLETE physical descriptor") < body.index(
            "rpu_launch_"
        )
        assert body.count(
            "rpu_launch_insert_kvcache_spm_unified_with_plan("
        ) == 1
        assert "rpu_resolve_kvinsert_segment_plan_auto(" not in body
        assert "resolve_kvinsert_plan_auto(" not in body
        assert "rpu_kvinsert_segment_plan_from_route_arguments(" not in body
        assert body.index("find_physical_route(") < body.index(
            "restore_kvinsert_plan("
        ) < body.index("consume_physical_route(", body.index("find_physical_route("))


def test_hyvla_shadow_w4_scales_retain_with_projection_tp_domains():
    vlm = _read("rpu_hyvla_vlm_model.cpp")
    expert = _read("rpu_hyvla_expert_model.cpp")
    for proj, ws in (("q_w_list", "q_ws_list"), ("k_w_list", "k_ws_list"),
                     ("v_w_list", "v_ws_list"), ("o_w_list", "o_ws_list")):
        assert vlm.count(f"rpu_retain_linear_quant_scale({proj}[i], {ws}[i], attn_tp())") == 2
        assert expert.count(f"rpu_retain_linear_quant_scale({proj}[i], {ws}[i], attn_tp())") == 1
    for proj, ws in (("gate_list", "gate_ws_list"), ("up_list", "up_ws_list"),
                     ("down_list", "down_ws_list")):
        assert vlm.count(f"rpu_retain_linear_quant_scale({proj}[i], {ws}[i], mlp_tp())") == 2
        assert expert.count(f"rpu_retain_linear_quant_scale({proj}[i], {ws}[i], action_mlp_cores)") == 1

