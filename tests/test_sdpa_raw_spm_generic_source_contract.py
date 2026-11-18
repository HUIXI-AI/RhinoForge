"""Board-free contract for generic raw-SPM V-transpose + by-MHA launchers."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/ops/rpu_sdpa_vctxlen.cpp").read_text(encoding="utf-8")
DECLS = (ROOT / "src/core/rpu_kernel_decls.h").read_text(encoding="utf-8")
CACHE = (ROOT / "src/core/rpu_kernel_cache.h").read_text(encoding="utf-8")
CACHE_INC = (ROOT / "src/core/rpu_kernel_cache.inc").read_text(encoding="utf-8")


def _between(start: str, end: str | None = None) -> str:
    begin = SOURCE.index(f"void {start}(")
    finish = SOURCE.index(f"void {end}(", begin) if end else len(SOURCE)
    return SOURCE[begin:finish]


def test_generic_api_and_wall_compatibility_delegate_are_explicit():
    assert "void rpu_launch_v_transpose_spm(" in DECLS
    assert "void rpu_launch_sdpa_by_mha_spm(" in DECLS
    assert "V_TRANSPOSE_SPM_VCTXLEN" in CACHE
    assert "SDPA_BY_MHA_SPM_VCTXLEN" in CACHE
    assert "V_TRANSPOSE_SPM_BATCH3_576" not in CACHE
    assert "SDPA_BY_MHA_SPM_BATCH3_576" not in CACHE
    assert '"llm_fp16_32b_prefill_trans_prjv_dp_vctxlen"' in CACHE
    assert '"llm_fp16_16b_prefill_flash_attn_by_mha_univ_vctxlen"' in CACHE

    wall_v = _between(
        "rpu_launch_v_transpose_spm_batch3_576",
        "rpu_launch_sdpa_by_mha_spm_batch3_576",
    )
    wall_mha = _between("rpu_launch_sdpa_by_mha_spm_batch3_576")
    assert wall_v.count("rpu_launch_v_transpose_spm(") == 1
    assert wall_mha.count("rpu_launch_sdpa_by_mha_spm(") == 1
    assert "kernel->set_regs" not in wall_v + wall_mha
    for value in (
        "kWallVisionBatch",
        "kWallVisionSeq",
        "kWallVisionHeadsPerCore * kWallVisionCores",
        "kWallVisionHeadDim",
        "kWallVisionCores",
    ):
        assert value in wall_v
        assert value in wall_mha


def test_generator_shape_mask_and_spm_constraints_fail_closed():
    trans = _between("rpu_launch_v_transpose_spm", "rpu_launch_sdpa_by_mha_spm")
    mha = _between(
        "rpu_launch_sdpa_by_mha_spm",
        "rpu_launch_v_transpose_spm_batch3_576",
    )
    for required in (
        "num_cores > 0 && num_cores <= SpmAllocator::NUM_CORES",
        "batch == 1 || seq_k % kV16 == 0",
        "head_dim % kV16 == 0",
        "num_kv_heads % num_cores == 0",
        "num_kv_heads_per_core <= 32",
        "checked_spm_bytes(",
        "checked_spm_region(",
        "check_spm_regions_disjoint(",
        "queue->set_broadcast_mode(true)",
    ):
        assert required in trans

    for required in (
        "attn_mask_type == 0 || attn_mask_type == 1 ||",
        "attn_mask_type == 4",
        "seq_k >= seq_q",
        "num_q_heads % num_kv_heads == 0",
        "num_q_heads % num_cores == 0",
        "num_kv_heads % num_cores == 0",
        "batch == 1 || seq_k % kV16 == 0",
        "seq_q_acc % kV16 == 0 && seq_q_acc % seq_q == 0",
        "kTileKV16 % tile_m_v16 == 0",
        "tile_m_v16 * tile_n_v16 * kTileKV16 <=",
        "sdpa_validate_kernel_call(",
        "grid_dim_y, gqa_group_size",
        '"K", {batch, seq_k_aligned, num_kv_heads_per_core, head_dim}',
        '"attention mask", {seq_q, seq_k_aligned}',
        "check_spm_regions_disjoint(",
        "q_off != output_off",
        "queue->set_broadcast_mode(true)",
    ):
        assert required in mha

    assert "offset + bytes <= SpmAllocator::SPM_USABLE" in SOURCE
    assert "static_cast<uint64_t>(dim) <= SpmAllocator::SPM_USABLE / bytes" in SOURCE


def test_registers_and_grids_match_the_two_generators():
    trans = _between("rpu_launch_v_transpose_spm", "rpu_launch_sdpa_by_mha_spm")
    mha = _between(
        "rpu_launch_sdpa_by_mha_spm",
        "rpu_launch_v_transpose_spm_batch3_576",
    )
    trans_regs = {
        0: "seq_k",
        1: "seq_k >> 16",
        2: "0",
        3: "0",
        6: "seq_k",
        7: "seq_k_v16",
        9: "num_kv_heads_per_core",
        12: "head_dim",
        14: "head_dim_v16",
        15: "head_dim_v16",
        59: "head_dim",
        60: "kTileK",
        62: "head_dim_v16",
        63: "kTileK / kV16",
        64: "1",
        65: "num_kv_heads_per_core",
        66: "batch",
    }
    mha_regs = {
        0: "seq_k",
        1: "seq_k >> 16",
        2: "seq_q",
        3: "seq_q_v16",
        4: "seq_k_v16",
        5: "seq_q_acc_v16",
        6: "num_q_heads",
        7: "num_kv_heads",
        8: "num_q_heads_per_core",
        9: "num_kv_heads_per_core",
        10: "head_dim",
        11: "head_dim",
        12: "head_dim",
        13: "head_dim_v16",
        14: "head_dim_v16",
        15: "head_dim_v16",
        16: "seq_q_acc",
        17: "seq_q_acc >> 16",
        18: "num_cores",
        19: "gqa_group_size",
        40: "scale_bits",
        41: "scale_bits >> 16",
        42: "1",
        56: "attn_mask_type",
        58: "tile_m",
        59: "tile_n",
        60: "tile_k",
        61: "tile_m_v16",
        62: "tile_n_v16",
        63: "kTileKV16",
        64: "num_q_heads_per_core",
        65: "grid_dim_y",
        66: "batch",
    }
    for reg, value in trans_regs.items():
        assert f"set_regs({reg}," in trans and value in trans
    for reg, value in mha_regs.items():
        assert f"set_regs({reg}," in mha and value in mha
    assert "constexpr int kScmStride = 4" in trans
    assert "constexpr int kScmStride = 12" in mha
    assert "{1, static_cast<uint16_t>(num_kv_heads_per_core)," in trans
    assert "{static_cast<uint16_t>(num_q_heads_per_core)," in mha

    # Frozen Wall values produced by the generic formulas: register-equivalent
    # to the former hand-written B3/S576/H80 launchers.
    assert (576 // 16, 80 // 16, 256 // 16) == (36, 5, 16)
    assert (16 // 8, (576 + 80 - 1) // 80, 3) == (2, 8, 3)


def test_wall_native_window_exact_wrapper_matches_generator_abi():
    symbol = "llm_fp16_16b_prefill_flash_attn_by_mha_wall_window_b3s576h16d80"
    kernel_id = "WALL_WINDOW_SDPA_BY_MHA_B3S576H16D80"
    launcher = "rpu_launch_wall_window_sdpa_by_mha_spm"
    assert f'"{symbol}"' in CACHE
    assert f'"{symbol}"' in CACHE_INC
    assert kernel_id in CACHE
    assert f"void {launcher}(" in DECLS

    body = _between(launcher, "rpu_launch_v_transpose_spm_batch3_576")
    assert "kOperandBytes = 552'960" in body
    assert body.count("checked_spm_region(") == 4
    assert "check_spm_regions_disjoint(std::array<SpmRegion, 4>" in body
    assert "q.offset != output.offset" in body
    for reg, value in {
        0: 576, 2: 576, 3: 36, 4: 36, 6: 16, 7: 16, 8: 2, 9: 2,
        10: 80, 11: 80, 12: 80, 13: 5, 14: 5, 15: 5,
        18: 8, 19: 1, 42: 1, 56: 0, 58: 64, 59: 80,
        60: 256, 61: 4, 62: 5, 63: 16, 64: 2, 65: 1, 66: 36,
    }.items():
        assert f"set_regs({reg}, static_cast<uint16_t>({value}))" in body
    assert "set_regs(40, static_cast<uint16_t>(scale_bits))" in body
    assert "set_regs(41, static_cast<uint16_t>(scale_bits >> 16))" in body
    assert "constexpr int kScmStride = 12" in body
    for slot, operand in ((0, "q.offset"), (6, "output.offset"),
                          (8, "k.offset"), (10, "v.offset")):
        base = "base" if slot == 0 else f"base + {slot}"
        assert f"set_scm_spm_ptr(kernel, {base}, core, {operand})" in body
    assert body.count("queue->set_broadcast_mode(true)") == 1
    assert body.count("queue->enqueu_kernel(") == 1
    assert "*kernel, {2, 1, 36}" in body
    assert "rpu_launch_sdpa_by_mha_spm(" not in body
