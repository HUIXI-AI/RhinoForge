from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_wall_action_owns_padded_kv_and_one_replay_mask_upload() -> None:
    action = (ROOT / "src/fused/rpu_wall_oss_action_step_model.cpp").read_text()
    decoder = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()

    assert "kv_insert_spm_rows_ = Align(cs, (int64_t)16);" in action
    assert action.count("b.size = std::max(b.size, kv_bytes);") == 2
    assert "b.phase_start = 0;" in action and "b.phase_end = 9;" in action
    assert "keep_explicit_mask_in_spm_ = true;" in action

    # Launcher/COMPLETE row contracts execute in test_wall_base_paired_kv.py.
    assert "int64_t kv_insert_spm_rows_ = 0;" in decoder
    assert "bool keep_explicit_mask_in_spm_ = false;" in decoder
    assert "ctx().body_iter == 0 && layer_idx == 0" in decoder


def test_wall_action_unroll_guard_matches_the_sdk_entry_limit() -> None:
    action = (ROOT / "src/fused/rpu_wall_oss_action_step_model.cpp").read_text()
    assert "num_steps * 1100 < 65536" in action


def test_wall_action_hoists_row_constant_w2a_across_unrolled_steps() -> None:
    action = (ROOT / "src/fused/rpu_wall_oss_action_step_model.cpp").read_text()
    assert 'd.push_back({"ct_spm",        A(h * DW),                0, 9,' in action

    prehook = action.split("void WallOssActionStepModel::emit_pre_layers_body()", 1)[1]
    prehook = prehook.split("void WallOssActionStepModel::emit_post_layers_body()", 1)[0]
    have_te = prehook.split("if (have_te_ == 1) {", 1)[1].split("} else {", 1)[0]
    body0, every_body = have_te.split("if (bit == 0) {", 1)[1].split("}", 1)

    assert "&b_step_src_base_" in body0
    assert 'addr(0, "ae_spm"), w2_a_, addr(0, "ct_spm")' in body0
    assert "&te_src_base_" not in body0 and "w2_t_" not in body0
    assert "&te_src_base_" in every_body
    assert 'addr(0, "te_spm"), w2_t_, addr(0, "ae_spm")' in every_body
    compact = " ".join(every_body.split())
    assert (
        'addr(0, "ct_spm"), addr(0, "ae_spm"), addr(0, "b_step_spm")'
        in compact
    )


def test_wall_action_embedding_stays_in_spm_through_layer0_ingress() -> None:
    action = (ROOT / "src/fused/rpu_wall_oss_action_step_model.cpp").read_text()
    adapter = (ROOT / "python/rpu_backend/adapters/wall_oss/action.py").read_text()
    decoder = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    prehook = action.split("void WallOssActionStepModel::emit_pre_layers_body()", 1)[1]
    prehook = prehook.split("void WallOssActionStepModel::emit_post_layers_body()", 1)[0]

    assert "pre_layers_residual1_ready_ = true;" in action
    assert 'd.push_back({"emb_core0_spm", A(cs * (h / num_cores()) * DW), 0, 1,' in action
    assert "d.back().size = A(cs * h * DW);" in action
    assert prehook.count("rpu_launch_spm_scatter_spm_dma(") == 2
    assert "/*core_stride_bytes=*/0" in prehook
    assert 'addr(0, "b_step_spm"), num_cores()' in prehook
    assert "/*num_cores=*/eight_cores ? num_cores() : 1" in prehook
    assert "rpu_launch_all_gather_spm_kernel(" in prehook
    assert 'addr(0, "emb_core0_spm"), addr(0, "residual1")' in prehook
    assert "/*chunk_elems=*/h / num_cores()" in prehook
    assert prehook.index('addr(0, "ce_spm"), cs * h') < prehook.index(
        'w3_, addr(0, "emb_core0_spm")'
    ) < prehook.index("rpu_launch_all_gather_spm_kernel(")
    assert "emb_stage_.data_ptr" not in prehook
    assert '"w3":        _to_rpu_half(tp_col_swizzle_mc_weight(' in adapter
    w3_pack = adapter.split(
        '"w3":        _to_rpu_half(tp_col_swizzle_mc_weight(', 1
    )[1].split(")),", 1)[0]
    # Eight cores col-swizzle W3 over 8 lanes; reduced budgets run W3 on core 0 and broadcast.
    assert w3_pack.endswith("1 if num_cores != 8 else num_cores")

    assert "bool pre_layers_residual1_ready_ = false;" in decoder
    assert decoder.count(
        "!(pre_layers_residual1_ready_ && layer_idx == 0)"
    ) == 3
