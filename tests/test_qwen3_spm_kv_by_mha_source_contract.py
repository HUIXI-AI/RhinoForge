"""Board-free contract for the bounded Qwen3 raw-SPM attention adopter."""

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
RUNTIME_CONFIG = (ROOT / "docs/runtime_config.md").read_text()


def _section(start: str, end: str) -> str:
    begin = HEADER.index(start)
    return HEADER[begin : HEADER.index(end, begin)]


def test_switch_is_validated_and_snapshotted_per_handle(monkeypatch):
    from rpu_backend.runtime.control import rpu_env_bool

    name = "RPU_QWEN3_SPM_KV_BY_MHA"
    monkeypatch.delenv(name, raising=False)
    assert rpu_env_bool(name, default=True)
    for value, expected in (("0", False), ("1", True), ("false", False), ("on", True)):
        monkeypatch.setenv(name, value)
        assert rpu_env_bool(name, default=True) is expected
    monkeypatch.setenv(name, "maybe")
    with pytest.raises(ValueError):
        rpu_env_bool(name, default=True)

    # Cold Python installation validates once and passes the snapshot into the
    # handle constructor; forward/planning reads the handle field only.
    installer = (ROOT / "python/rpu_backend/runtime/decoder.py").read_text()
    assert 'handle = torch.ops.rpu.causal_decoder_create(\n        rpu_env_bool("RPU_QWEN3_SPM_KV_BY_MHA", default=True),' in installer
    native = (ROOT / "src/fused/rpu_qwen3_model.cpp").read_text()
    create = native[native.index("int64_t rpu_causal_decoder_create(") :
                    native.index("void rpu_causal_decoder_destroy(")]
    assert "->configure_cold_routes(" in create
    configure = _section(
        "    void configure_cold_routes(",
        "    // `position` is the cache position",
    )
    assert "get_last_resolved_chunk_size() == 0" in configure
    assert "qwen3_spm_kv_by_mha_enabled_ = qwen3_spm_kv_by_mha;" in configure
    assert "!cold_routes_configured_ && num_layers() == 0" in configure
    assert "cold_routes_configured_ = true;" in configure
    assert "bool qwen3_spm_kv_by_mha_enabled_ = true;" in HEADER
    assert 'getenv("RPU_QWEN3_SPM_KV_BY_MHA")' not in HEADER
    dynamic = _section(
        "ModelDynamicConfig dynamic_config(",
        "bool subclass_chunk_size_valid(",
    )
    assert "qwen3_spm_kv_by_mha_enabled_" in dynamic
    assert "? AttentionExecutionPolicy::AUTO" in dynamic
    assert ": AttentionExecutionPolicy::DDR_KV" in dynamic
    assert "qwen3_spm_kv_by_mha_enabled()" not in dynamic


def test_switch_documents_auto_and_durable_ddr_fallback():
    assert "`RPU_QWEN3_SPM_KV_BY_MHA`" in RUNTIME_CONFIG
    assert "when the exact plan fits" in RUNTIME_CONFIG
    assert "`0` pins the DDR-cache attention path" in RUNTIME_CONFIG
    assert "unsupported shapes fall back to DDR automatically" in RUNTIME_CONFIG


def test_only_exact_plain_qwen3_short_prefill_is_eligible():
    eligible = _section(
        "bool subclass_spm_kv_by_mha_eligible(",
        "// ═══════════════════════════════════════════════════════════════════════\n"
        "    // static/dynamic config",
    )
    for contract in (
        "typeid(*this) != typeid(CausalDecoderModel)",
        "position != 0",
        "!layout.is_causal",
        "layout.use_attn_mask",
        "layout.batch_size != 1",
        "plan.chunk_mode != ChunkMode::SEQUENTIAL",
        "plan.compute.chunks.size() != 1",
        "seq != 16 && seq != 32 && seq != 64 && seq != 128",
        "num_q_heads() == 16",
        "num_kv_heads() == 8",
        "head_dim() == 128",
        "num_layers() == 28",
        "hidden_size() == 1024 && intermediate_size() == 3072",
        "hidden_size() == 2048 && intermediate_size() == 6144",
        "!has_qk_norm_",
        "has_qkv_bias_",
        "nvfp4_",
        "has_mrope_",
        "!deepstack_lang_layers_.empty()",
        "adarms_",
        "batch_decode_active_",
        "sdpa_kernel_ != SdpaKernelType::FLASH_ATTN_SPM",
        "layer.q_w.scalar_type() == at::kHalf",
        "layer.down_w.scalar_type() == at::kHalf",
    ):
        assert contract in eligible


def test_spm_layout_reuses_non_aliasing_sdpa_tmp_for_v_transpose():
    buffers = _section(
        "std::vector<BufferDecl> declare_buffers(",
        "int64_t subclass_layout_hash() const override",
    )
    assert "const bool spm_kv_by_mha" in buffers
    assert "tmp = std::max(tmp, A(bs * comp_cs * local_kv * DWIDTH));" in buffers
    assert "spm_kv_by_mha ? 4" in buffers
    assert 'decls.push_back({"v",' in buffers
    assert 'decls.push_back({"sdpa_tmp",' in buffers


def test_spm_attention_mirrors_kv_to_ddr_before_raw_dispatch():
    phase4 = _section(
        "// Phase 4: KV cache insert + SDPA + O_proj",
        "// Layer 0 establishes a full residual2",
    )
    consume_plan = phase4.index("const KvInsertSegmentPlan kv_plan = consume_causal_kv_plan(")
    kv_insert = phase4.index("rpu_launch_insert_kvcache_spm_unified_with_plan(")
    consume_attention = phase4.index("CAUSAL_DECODER_RAW_SPM_ATTN_SITE")
    transpose = phase4.index("rpu_launch_v_transpose_spm(")
    by_mha = phase4.index("rpu_launch_sdpa_by_mha_spm(")
    fallback = phase4.index("rpu_launch_sdpa_spm_dispatch(", by_mha)
    assert consume_plan < kv_insert < consume_attention < transpose < by_mha < fallback
    assert "k_cache, v_cache," in phase4[kv_insert:consume_attention]
    assert "kv_insert_spm_rows_, kv_plan);" in phase4[kv_insert:consume_attention]
    assert "mask_type == 1" in phase4
    assert "1.0 / std::sqrt(static_cast<double>(hd))" in phase4
    assert HEADER.count("rpu_launch_v_transpose_spm(") == 1
    assert HEADER.count("rpu_launch_sdpa_by_mha_spm(") == 1
