from pi_wall_host_catalog import assert_current_catalog
"""Board-free contract for Wall's exact staged SwiGLU activation fusion."""

from pathlib import Path

from test_rmsnorm_capability_publication_host import _block


ROOT = Path(__file__).resolve().parents[1]


def test_silu_mul_block_bound_fits_the_kernel_signed_loop_counter():
    source = (ROOT / "src/ops/rpu_eltwise_unary.cpp").read_text()
    launcher = _block(source, "void rpu_launch_silu_mul_spm_kernel(")
    assert 256 * 127 < 2**15
    assert "normal_blk_n = 256 * 127" in launcher
    assert "(uint16_t)blk_cnt" in launcher
    assert launcher.count("enqueu_kernel(") == 1
    assert "for (int64_t off" not in launcher


def test_staged_p640_uses_fused_gate_up_swiglu_for_both_mlp_paths():
    decoder = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    start = decoder.index("void build_wall_p640_staged_layer_subgraph(")
    staged = decoder[start : decoder.index("void build_layer_subgraph(", start)]
    assert staged.count(
        "rpu_launch_wall_prefill_gate_up_swiglu_w8a16_spm_kernel("
    ) == 2
    assert "rpu_launch_silu_mul_spm_kernel(" not in staged
    assert "rpu_launch_eltwise_unary_spm_kernel(" not in staged
    assert "rpu_launch_eltwise_binary_spm_kernel(" not in staged
    assert 'addr(0, "up")' not in staged


def test_staged_p640_gate_up_kernel_uses_current_shared_a_without_scratch_writes():
    assert_current_catalog("SharedSilu")
    cache_h = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    cache_inc = ((ROOT / "src/core/rpu_kernel_cache.inc").read_text() + (ROOT / "src/core/rpu_kernel_cache.h").read_text())
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    linear = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    decoder = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    name = "fused_gate_up_swiglu_w8a16_m320n112k128"
    p640_name = "fused_gate_up_swiglu_w8a16_m128x5n96k128"

    launcher_name = (
        "rpu_launch_wall_prefill_gate_up_swiglu_w8a16_spm_kernel("
    )
    assert launcher_name in decls
    launcher = _block(linear, f"void {launcher_name}")
    launcher_flat = " ".join(launcher.split())
    assert launcher.count("GraphDdrRegisterOperandSpec{") == 4
    assert launcher.count("GraphDdrRegisterRole::Int8ModelWeight") == 2
    assert launcher.count("GraphDdrRegisterRole::PerChannelScale") == 2
    for register_pair in ("2, 3", "20, 21", "32, 33", "34, 35"):
        assert f"DevAddrShift8LoHi, {register_pair}" in launcher
    for guard in (
        "rows == 320 || rows == 640",
        "constexpr int64_t kK = 2048;",
        "constexpr int64_t kN = 11008;",
        "constexpr uint32_t kScratchBytes = 573440;",
        "static_cast<uint32_t>(kK * sizeof(c10::Half))",
        "static_cast<uint32_t>(kNPerCore * sizeof(c10::Half))",
        "queue->set_broadcast_mode(true);",
    ):
        assert guard in launcher

    layout_start = decoder.index("const bool wall_p640_staged_layout")
    layout = decoder[layout_start : decoder.index("// \u2605 Phase 2.5", layout_start)]
    assert 'if (!wall_p640_staged_layout) {' in layout
    assert 'decls.push_back({"up"' in layout
    assert ("const int down_phase_start = (exact_multiview_text_layout || "
            "lingbot_c400_full_layout) ? 8 : 7;") in " ".join(layout.split())
    assert "lingbot_prefill_deepstack_reuses_down(ctx) && ctx.chunk_size == 400" in layout

    start = decoder.index("void build_wall_p640_staged_layer_subgraph(")
    staged = decoder[start : decoder.index("void build_layer_subgraph(", start)]
    staged_flat = " ".join(staged.split())
    assert staged.count(launcher_name) == 2
    assert 'addr(0, "gate"), addr(0, "down"), kHalfRows' in staged
    assert 'addr(0, "gate"), addr(0, "down"), kPackedRows' in staged
    assert "const bool transition_to_bf16 = layer_idx == 2;" in staged
    assert "if (!transition_to_bf16 && scheduled_chunk.idx != 0)" in staged
    assert (
        "if (transition_to_bf16) { const int half = scheduled_chunk.idx;"
        in staged_flat
    )
    assert staged.count('addr(0, "gate"), addr(0, "down")') == 2


def test_staged0_fuses_silu_mul_through_the_shared_mlp_helper():
    decoder = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    decoder_flat = " ".join(decoder.split())
    fuse = decoder_flat.split("/*fuse_silu_mul=*/", 1)[1].split("/*skip_down=*/", 1)[0]
    for route in (
        "use_planned_silu_mul()", "wall_small_prefill_fused",
        "wall_p640_staged0_w8a16", "wall_action_w8a16_silu_mul",
        "(wall_sorting_.silu_mul && use_silu_ && "
        "(rtc_prefill_mixed || wall_action_fuse_silu_mul_))",
    ):
        assert route in fuse
    staged0 = decoder[
        decoder.index("const bool wall_p640_staged0_w8a16") :
        decoder.index("const bool wall_action_fp16_transfer", decoder.index(
            "const bool wall_p640_staged0_w8a16"))
    ]
    for guard in (
        "rtc_prefill_mixed && !wall_prefill_staged_mlp_",
        "seq_len_ == 640", "seq_len == 320", "rows == 320",
        "h == 2048", "nq == 16", "nkv == 8", "hd == 128",
        "tp == NUM_CORES", "has_qkv_bias_", "use_silu_", "!linear_acc32_",
    ):
        assert guard in staged0
    for weight in ("q_w", "k_w", "v_w", "gate_w", "up_w"):
        assert f"lw.{weight}.scalar_type() == at::kChar" in staged0
    mlp = base[base.index("void FusedModelBase::emit_mlp_pipeline(") :]
    assert "if (!fuse_silu_mul && !high_precision_silu_mul)" in mlp
    assert "TORCH_CHECK(act == ActivationKind::SILU," in mlp
    assert "rpu_launch_silu_mul_spm_kernel(" in mlp
    assert "/*bind_silu_mul_route=*/use_planned_silu_mul()" in decoder
    assert "FMB_SHARED_MLP_SILU_MUL_SITE" in mlp


def test_exact_w8_action_fuses_only_the_shared_decoder_mlp():
    decoder = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    action = (
        ROOT / "src/fused/rpu_wall_oss_action_step_model.cpp"
    ).read_text()
    assert "bool wall_action_fuse_silu_mul_ = false;" in decoder
    assert "wall_action_fuse_silu_mul_ = true;" in action
    # The same exact Action selector admits cold-enabled FP16 transfer. Its
    # dtype alias must stay tied to that admission, not accept arbitrary types.
    transfer = decoder[
        decoder.index("bool wall_action_fp16_transfer_enabled() const") :
        decoder.index("static bool wall_action_qkv_weights_compatible(")
    ]
    for guard in (
        "(wall_sorting_.fp16_transfer_mask & 4u)", "!layer_weights_.empty()",
        "layer_weights_.front().q_w.scalar_type() == at::kHalf",
        "!layer_weights_.front().q_ws.defined()",
    ):
        assert guard in transfer
    decoder_flat = " ".join(decoder.split())
    assert (
        "const bool wall_action_fp16_transfer = wall_action_fp16_transfer_enabled(); "
        "const auto wall_action_weight_type = wall_action_fp16_transfer ? at::kHalf : at::kChar;"
        in decoder_flat
    )
    selector = decoder[
        decoder.index("const bool wall_action_w8a16_silu_mul") :
        decoder.index("TORCH_CHECK(!decoder_mixed_precision_", decoder.index(
            "const bool wall_action_w8a16_silu_mul"))
    ]
    for guard in (
        "wall_action_fused_rows_supported(rows, wall_sorting_)", "h == 1024", "intermediate_size() == 2048",
        "num_layers() == 36", "tp == NUM_CORES", "use_silu_", "!linear_acc32_",
        "lw.gate_w.scalar_type() == wall_action_weight_type",
        "lw.up_w.scalar_type() == wall_action_weight_type",
    ):
        assert guard in selector
    assert "wall_p640_staged0_w8a16 || wall_action_w8a16_silu_mul" in decoder_flat
    assert "/*fuse_wall_gate_up=*/wall_action_w8a16_silu_mul" in decoder
    assert "rpu_launch_silu_mul_spm_kernel(" not in action


def test_exact_w8_action_fuses_gate_up_without_new_spm_buffers():
    cache_h = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    cache_inc = ((ROOT / "src/core/rpu_kernel_cache.inc").read_text() + (ROOT / "src/core/rpu_kernel_cache.h").read_text())
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    linear = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    assert_current_catalog("SharedRaw", "SharedRawSpmWeight")

    assert "rpu_launch_wall_action_gate_up_w8a16_multibase_spm_kernel(" in decls
    start = linear.index(
        "void rpu_launch_wall_action_gate_up_w8a16_multibase_spm_kernel("
    )
    launcher = linear[
        start : linear.index(
            "void rpu_launch_wall_action_linear_ring_postnorm_spm_kernel(",
            start,
        )
    ]
    assert launcher.count("GraphDdrRegisterOperandSpec{") == 4
    assert "wall_action_fused_rows_supported(rows, sorting)" in launcher
    assert "const int64_t kM = rows;" in launcher
    assert "constexpr int64_t kK = 1024;" in launcher
    assert "constexpr int64_t kN = 2048;" in launcher
    assert "queue->set_broadcast_mode(true);" in launcher
    assert "{selected.grid_x, selected.grid_y, 1}" in launcher

    mlp = base[base.index("void FusedModelBase::emit_mlp_pipeline(") :]
    selector = mlp[
        mlp.index("const bool wall_action_fuse_gate_up") :
        mlp.index("if (wall_action_fuse_gate_up)")
    ]
    for guard in (
        "fuse_wall_gate_up", "fuse_silu_mul", "wall_action_fused_rows_supported(seq_len, sorting)", "hidden_size_ == 1024",
        "intermediate_size_ == 2048", "!acc32",
        "gate_w.scalar_type() == at::kChar",
        "up_w.scalar_type() == at::kChar",
    ):
        assert guard in selector
    selector_flat = " ".join(selector.split())
    assert (
        "((gate_w.scalar_type() == at::kChar && up_w.scalar_type() == at::kChar "
        "&& gate_ws.defined() && up_ws.defined()) || "
        "((sorting.fp16_transfer_mask & 4u) && gate_w.scalar_type() == at::kHalf "
        "&& up_w.scalar_type() == at::kHalf && !gate_ws.defined() && !up_ws.defined()))"
        in selector_flat
    )
    assert mlp.count(
        "rpu_launch_wall_action_gate_up_w8a16_multibase_spm_kernel("
    ) == 1
    assert 'addr(0, "gate"), addr(0, "up")' in mlp


def test_exact_mimc_w8_vision_fuses_gate_up_silu_with_acc32_fallback():
    vision = (
        ROOT / "src/fused/rpu_qwen25vl_vision_model.cpp"
    ).read_text()
    start = vision.index("void build_exact_mimc_compute_body(")
    exact = vision[start : vision.index("void build_layer_subgraph(", start)]
    compute_selector = exact[
        exact.index("const bool fuse_w8_silu_mul") :
        exact.index("const bool has_next_layer", exact.index(
            "const bool fuse_w8_silu_mul"))
    ]
    assert "exact_mimc_fused_w8_gate_up_enabled()" in compute_selector
    helper_start = vision.index("bool exact_mimc_fused_w8_gate_up_enabled() const")
    qkv_helper_start = vision.index("bool exact_mimc_w8_qkv_enabled() const")
    selector = vision[qkv_helper_start:helper_start] + vision[
        helper_start : vision.index(
            "bool exact_mimc_native_window_sdpa_enabled() const", helper_start
        )
    ]
    for guard in (
        "mixed_precision_",
        "lw.gate_w.scalar_type() == at::kChar",
        "lw.up_w.scalar_type() == at::kChar",
    ):
        assert guard in selector
    assert exact.count(
        "rpu_launch_wall_vision_gate_up_swiglu_w8a16_spm_kernel("
    ) == 1
    assert "if (fuse_w8_silu_mul)" in exact
    assert "rpu_launch_silu_mul_spm_kernel(" not in exact

    cache_h = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    cache_inc = ((ROOT / "src/core/rpu_kernel_cache.inc").read_text() + (ROOT / "src/core/rpu_kernel_cache.h").read_text())
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    linear = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    assert_current_catalog("SharedSiluBias")
    launcher_name = (
        "rpu_launch_wall_vision_gate_up_swiglu_w8a16_spm_kernel("
    )
    assert launcher_name in decls
    start = linear.index(f"void {launcher_name}")
    launcher = linear[
        start : linear.index(
            "void rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(", start
        )
    ]
    assert launcher.count("GraphDdrRegisterOperandSpec{") == 4
    assert launcher.count("GraphDdrRegisterRole::Int8ModelWeight") == 2
    assert launcher.count("GraphDdrRegisterRole::PerChannelScale") == 2
    for register_pair in ("2, 3", "20, 21", "32, 33", "34, 35"):
        assert f"DevAddrShift8LoHi, {register_pair}" in launcher
    for guard in (
        "const int64_t kM = rows;",
        "constexpr int64_t kK = 1280;",
        "constexpr int64_t kN = 3584;",
        "gate_bias_spm_addr != 0 && up_bias_spm_addr != 0",
        "set_u32_reg_pair(*kernel, 38, up_bias_spm_addr);",
        "queue->set_broadcast_mode(true);",
        "{selected.grid_x, selected.grid_y, 1}",
    ):
        assert guard in launcher


def test_exact_mimc_compact_vision_uses_three_image_gate_up_multibase():
    vision = (
        ROOT / "src/fused/rpu_qwen25vl_vision_model.cpp"
    ).read_text()
    start = vision.index("void build_exact_mimc_compute_body(")
    compute = vision[start : vision.index("void build_layer_subgraph(", start)]
    launcher_name = (
        "rpu_launch_wall_vision_gate_up_swiglu_w8a16_spm_kernel("
    )

    assert "if (compact_local_inout && chunk.idx != 0)" in compute
    assert "const int first_image = compact_local_inout ? 0 : chunk.idx;" in compute
    assert (
        "? ExactMimcPlan::kImages : 1;"
        in compute
    )
    first_o_norm = compute.index("for (int image = first_image;")
    gate_up = compute.index(launcher_name)
    packed_down = compute.index("const int64_t down_rows")
    down_ring = compute.index(
        "for (int image = first_image;", first_o_norm + 1
    )
    assert first_o_norm < gate_up < packed_down < down_ring
    for packed_down_contract in (
        "? packed_seq : mlp_rows;",
        "gate_base, lw.down_w, first_packed_h_addr",
        "down_rows, h, is_, /*row=*/0, NUM_CORES",
    ):
        assert packed_down_contract in compute
    ring_body = compute[
        down_ring : compute.index("// Inner layers stay entirely in SPM.")
    ]
    assert "rpu_launch_linear_spm_to_spm_acc16_kernel(" not in ring_body
    assert ring_body.count(
        "rpu_launch_linear_spm_to_spm_acc32_out_bf16_kernel("
    ) == 1
    for address in (
        "first_packed_h_addr + hidden_image_stride",
        "first_packed_h_addr + 2 * hidden_image_stride",
        "gate_base + gate_image_stride",
        "gate_base + 2 * gate_image_stride",
    ):
        assert address in compute

    declarations = vision[
        vision.index("std::vector<BufferDecl> declare_buffers(") : start
    ]
    assert '"gate",         compact_local_inout' in declarations
    assert "? ExactMimcPlan::kImages * mlp_inter" in declarations
    assert "kNativeWindowLocalInoutTempPeakBytes = 6'635'520" in vision
    packed_h = 3 * 576 * 1280 * 2
    compact_residual = 3 * 576 * (1280 // 8) * 2
    gate_image = 576 * (3584 // 8) * 2
    assert packed_h + compact_residual + 3 * gate_image == 6_524_928
    assert packed_h + compact_residual + 3 * gate_image <= 6_635_520

    cache_h = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    cache_inc = ((ROOT / "src/core/rpu_kernel_cache.inc").read_text() + (ROOT / "src/core/rpu_kernel_cache.h").read_text())
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    linear = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    kernel_id = (
        "WALL_VISION_GATE_UP_SWIGLU_W8A16_MULTIBASE_M192X3N96"
    )
    assert_current_catalog("TripletSharedSiluBias")
    assert "uint32_t input1_spm_addr = 0" in decls
    assert "uint32_t output2_spm_addr = 0" in decls

    launcher_start = linear.index(f"void {launcher_name}")
    launcher = linear[
        launcher_start : linear.index(
            "void rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(",
            launcher_start,
        )
    ]
    for register, address in (
        (48, "input1_spm_addr"),
        (54, "input2_spm_addr"),
        (50, "output1_spm_addr"),
        (56, "output2_spm_addr"),
    ):
        assert (
            f"rpu_pl_tiling::set_u32_reg_pair(*kernel, {register}, {address});"
            in launcher
        )
    assert "queue->set_broadcast_mode(true);" in launcher
    assert "{selected.grid_x, selected.grid_y, 1}" in launcher
    assert "set_regs(52, static_cast<uint16_t>(rows))" in launcher
