from pi_wall_host_catalog import assert_current_catalog, CATALOG_INCLUDE
from test_siglip_attention_dry_live_contract import _function
"""Board-free contract for exact Wall P640 O-proj weight reuse."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_H = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
CACHE_INC = (ROOT / "src/core/rpu_kernel_cache.inc").read_text()
DECLS = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
LINEAR = (ROOT / "src/ops/rpu_linear.cpp").read_text()
DECODER = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
SYMBOL = "wall_oproj_weight_outer_w8a16_c320x2_n64k128"
LAUNCHER = "rpu_launch_wall_prefill_oproj_w8a16_c320x2_spm_kernel("


def test_exact_oproj_launcher_preserves_typed_ddr_and_multicore_abi():
    assert_current_catalog("PairedLinear")
    launcher=_function(LINEAR,"void rpu_launch_wall_prefill_oproj_w8a16_c320x2_spm_kernel(")
    assert launcher.count("GraphDdrRegisterOperandSpec{") == 2
    assert launcher.count("GraphDdrRegisterRole::Int8ModelWeight") == 1
    assert launcher.count("GraphDdrRegisterRole::PerChannelScale") == 1
    for pair in ("2, 3","20, 21"):
        assert "DevAddrShift8LoHi, "+pair in launcher
    for contract in ("Operation::PairedLinear", "48, input0_spm_addr + (kM / 2) * kLocalK", "50, output1_spm_addr",
                     "set_regs(52", "register_writer.write(*kernel);", "queue->set_broadcast_mode(true);",
                     "{selected.grid_x, selected.grid_y, 1}"):
        assert contract in launcher


def test_only_layers4_to35_reorder_sdpa_and_use_inplace_carriers():
    start = DECODER.index("void build_wall_p640_staged_layer_subgraph(")
    staged = DECODER[start : DECODER.index("void build_layer_subgraph(", start)]
    assert "const bool normalized_compact_carry = layer_idx > 3;" in staged

    packed_start = staged.index("// Layers 4..35 already carry")
    packed_end = staged.index(
        "        } else {\n            emit_unpacked_attention_half(0);",
        packed_start,
    )
    packed = staged[packed_start:packed_end]
    half1_sdpa = "c10::nullopt, q_half1_off, output_half1_off,"
    half0_sdpa = 'c10::nullopt, addr_offset("q").value, q_half1_off,'
    assert packed.count("rpu_launch_sdpa_spm_dispatch(") == 2
    assert half1_sdpa in packed
    assert half0_sdpa in packed
    assert packed.index(half1_sdpa) < packed.index(half0_sdpa)
    assert "output_half1_off == q_half1_off + kHalfQBytes" in packed
    assert packed.count(LAUNCHER) == 1
    assert 'addr(0, "q") + kHalfQBytes, lw.o_w,' in packed
    assert 'addr(0, "residual1") + kHalfHiddenBytes, lw.o_ws, wall_sorting_);' in packed
    assert packed.count(
        "rpu_launch_wall_ring_postnorm_pair_fp16_partial_bf16_residual_local_spm_kernel("
    ) == 1
    assert (
        'addr(0, "residual1"), addr(0, "residual_local"),\n'
        '                addr(0, "input_norm"),'
    ) in packed
    assert (
        "rpu_launch_wall_ring_postnorm_fp16_partial_bf16_residual_local_spm_kernel("
        not in packed
    )
    assert "emit_attention_half(" not in packed

    fallback = staged[packed_end:]
    assert "emit_unpacked_attention_half(0);" in fallback
    assert "emit_unpacked_attention_half(1);" in fallback
