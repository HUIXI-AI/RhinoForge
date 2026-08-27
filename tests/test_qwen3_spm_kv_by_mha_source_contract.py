"""Board-free contract for the bounded Qwen3 raw-SPM attention adopter."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
RUNTIME_CONFIG = (ROOT / "docs/runtime_config.md").read_text()


def _section(start: str, end: str) -> str:
    begin = HEADER.index(start)
    return HEADER[begin : HEADER.index(end, begin)]


def test_switch_is_strict_and_snapshotted_per_handle():
    parser = _section(
        "inline bool qwen3_spm_kv_by_mha_enabled()",
        "// =============================================================================\n"
        "// CausalDecoderModel",
    )
    assert 'std::getenv("RPU_QWEN3_SPM_KV_BY_MHA")' in parser
    assert 'std::strcmp(e, "0") == 0 || std::strcmp(e, "1") == 0' in parser
    assert "if (!e) return true;" in parser
    assert (
        "bool qwen3_spm_kv_by_mha_enabled_ =\n"
        "        qwen3_spm_kv_by_mha_enabled();"
    ) in HEADER
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
        "// Phase 5: Attention reduce + residual",
    )
    k_insert = phase4.index("rpu_launch_insert_kcache_spm_unified(")
    v_insert = phase4.index("rpu_launch_insert_vcache_spm_unified(")
    transpose = phase4.index("rpu_launch_v_transpose_spm(")
    by_mha = phase4.index("rpu_launch_sdpa_by_mha_spm(")
    fallback = phase4.index("rpu_launch_sdpa_spm_dispatch(", by_mha)
    assert k_insert < v_insert < transpose < by_mha < fallback
    assert "mask_type == 1" in phase4
    assert "1.0 / std::sqrt(static_cast<double>(hd))" in phase4
    assert HEADER.count("rpu_launch_v_transpose_spm(") == 1
    assert HEADER.count("rpu_launch_sdpa_by_mha_spm(") == 1
