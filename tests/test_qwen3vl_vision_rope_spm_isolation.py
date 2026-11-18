from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp").read_text()


def _between(begin: str, end: str) -> str:
    start = SOURCE.index(begin)
    return SOURCE[start : SOURCE.index(end, start)]


def test_ordinary_tp8_rope_uses_high_persistent_spm():
    helper = _between(
        "bool ordinary_rope_persistent_rebalance() const",
        "uint32_t q_bias_addr(",
    )
    for guard in (
        "rope_route_request_ != Qwen3VLVisionRopeRouteRequest::DDR",
        "!pipeline_dispatch_active()",
        "!native_composite_prepare_active_",
        "num_cores() == 8",
        "num_layers() == 24",
        "hidden_size() == 1024",
        "intermediate_size() == 4096",
        "num_q_heads() == 16",
        "head_dim() == 64",
    ):
        assert guard in helper

    residency = _between(
        "bool rope_tables_use_persistent_storage() const",
        "bool spm_rope_route_available() const",
    )
    assert "ordinary_rope_persistent_rebalance()" in residency


def test_rebalanced_q_bias_is_refreshed_per_layer():
    declarations = _between(
        'std::vector<BufferDecl> decls = {',
        "if (ordinary_resident)",
    )
    assert 'ordinary_rope_persistent_rebalance()\n                 ? StorageClass::Temp' in declarations
    assert 'ordinary_rope_persistent_rebalance() ? 0 : nl' in declarations

    body = _between(
        "void emit_kv_first_body(",
        "void build_layer_subgraph(",
    )
    refresh = body.index("if (ordinary_rope_persistent_rebalance())")
    layer_norm = body.index("rpu_launch_layernorm_spm_kernel(")
    assert refresh < layer_norm
    assert 'addr(0, "q_bias")' in body[refresh:layer_norm]
    assert body.count("q_bias_addr(layer_idx)") == 2
