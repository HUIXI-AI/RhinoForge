from pi_wall_host_catalog import assert_current_catalog, CATALOG_INCLUDE
from test_siglip_attention_dry_live_contract import _function
"""Board-free contract for exact Wall Vision W8A16 fused QKV."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_H = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
CACHE_INC = (ROOT / "src/core/rpu_kernel_cache.inc").read_text()
DECLS = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
LINEAR = (ROOT / "src/ops/rpu_linear.cpp").read_text()
VISION = (ROOT / "src/fused/rpu_qwen25vl_vision_model.cpp").read_text()
KERNEL_NAME = "qkv_parallel_w8a16_multibase_m192x3n96k128"


def test_exact_vision_launcher_is_strict_graph_aware_and_multicore():
    assert_current_catalog("Qkv")
    launcher=_function(LINEAR,"void rpu_launch_wall_vision_qkv_w8a16_multibase_spm_kernel(")
    for geometry in ("kImageM = 576", "kK = 1280", "kNPerCore = 160", "Operation::Qkv"):
        assert geometry in launcher
    assert launcher.count("GraphDdrRegisterOperandSpec{") == 6
    assert launcher.count("GraphDdrRegisterRole::Int8ModelWeight") == 3
    assert launcher.count("GraphDdrRegisterRole::PerChannelScale") == 3
    for pair in ("2, 3","20, 21","32, 33","34, 35","40, 41","42, 43"):
        assert "DevAddrShift8LoHi, "+pair in launcher
    for slot,base in ((36,"k_output_spm_addr"),(44,"v_output_spm_addr"),(38,"k_bias_spm_addr"),(46,"v_bias_spm_addr")):
        assert f"set_u32({slot}, {base});" in launcher
    assert "register_writer.write(*kernel);" in launcher
    assert "queue->set_broadcast_mode(true);" in launcher
    assert "{selected.grid_x, selected.grid_y, 1}" in launcher


def test_only_exact_mixed_w8_uses_fused_qkv_without_an_spm_buffer():
    start = VISION.index("void emit_exact_mimc_qkv_body(")
    compute_start = VISION.index("void build_exact_mimc_compute_body(", start)
    exact = VISION[start:compute_start]
    compute = VISION[compute_start : VISION.index("void build_layer_subgraph(", compute_start)]
    selector_start = VISION.index("bool exact_mimc_w8_qkv_enabled() const")
    selector = VISION[selector_start : VISION.index("bool merger_active() const", selector_start)]
    assert "const bool use_w8_multibase_qkv = exact_mimc_w8_qkv_enabled();" in exact
    assert "const bool use_w8_qkv = exact_mimc_w8_qkv_enabled();" in VISION
    assert (
        "return exact_mimc_w8_qkv_enabled()\n"
        "            && std::all_of("
    ) in selector
    assert "is_exact_mimc() && mixed_precision_" in selector
    assert "rpu_get_linear_acc32_enabled" not in selector
    assert selector.count("lw.q_w.scalar_type() == at::kChar") == 1
    assert selector.count("lw.k_w.scalar_type() == at::kChar") == 1
    assert selector.count("lw.v_w.scalar_type() == at::kChar") == 1
    assert "lw.q_ws.defined() && lw.k_ws.defined()" in selector
    assert "&& lw.v_ws.defined()" in selector
    assert exact.count(
        "rpu_launch_wall_vision_qkv_w8a16_multibase_spm_kernel("
    ) == 1
    assert 'addr(0, "q"), addr(0, "k"), addr(0, "v")' in exact
    assert 'addr(0, "v"), addr(0, "input_norm")' in exact
    assert 'addr(0, "q"), addr(0, "k"), addr(0, "input_norm")' in exact
    assert 'addr(0, "v"), mask_addr, mask_type, attn_scale' in exact
    assert exact.count("rpu_launch_spm_local_copy_dma(") == 1
    assert 'exact_mimc_w8_qkv_enabled()\n                    ? addr(0, "v") : addr(0, "q")' in compute
    assert "attention_output, lw.o_w" in compute

    decl_start = VISION.index("if (is_exact_mimc()) {", VISION.index("declare_buffers("))
    decl = VISION[decl_start : VISION.index("return decls;", decl_start)]
    assert "w8_direct_sdpa_output ? 3 : 4" in decl
    assert "w8_direct_sdpa_output ? 4 : 3" in decl

    # The FP16 branch keeps the original V -> transpose -> K -> Q sequence and
    # the original by-MHA V address. No non-exact function calls the launcher.
    assert exact.count("packed_linear(") == 3
    assert 'addr(0, "q"), addr(0, "v")' in exact
    assert 'addr(0, "q"), addr(0, "k"), addr(0, "v")' in exact
    assert VISION.count(
        "rpu_launch_wall_vision_qkv_w8a16_multibase_spm_kernel("
    ) == 1
    assert '"qkv_fused"' not in VISION
    assert '"qkv_staging"' not in VISION
