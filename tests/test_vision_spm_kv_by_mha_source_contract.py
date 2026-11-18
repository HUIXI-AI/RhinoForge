"""Board-free contracts for Vision/VLA raw-SPM attention adoption."""

from pathlib import Path

from test_siglip_attention_dry_live_contract import _function


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / "src" / "fused" / name).read_text(encoding="utf-8")


SOURCES = {
    "dino": _read("rpu_dinov3_vision_model.cpp"),
    "siglip": _read("rpu_siglip_model.cpp"),
    "gr00t_vl": _read("rpu_gr00t_vl_encoder_model.cpp"),
    "gr00t_dit": _read("rpu_gr00t_dit_model.cpp"),
    "hyvit2": _read("rpu_hyvit2_vision_model.cpp"),
}

RAW_SITES = {
    "dino": (("DINO_ATTENTION_RAW_SPM_SITE", "5876524946555003440LL"),),
    "siglip": (("SIGLIP_RAW_SPM_ATTN_SITE", "1638421364140696436LL"),),
    "gr00t_vl": (
        ("GR00T_VL_RAW_SPM_ATTENTION_SITE", "6949469595549868184LL"),
    ),
    "gr00t_dit": (
        ("GR00T_DIT_SELF_RAW_ATTENTION_SITE", "7562350367563329289LL"),
        ("GR00T_DIT_CROSS_RAW_ATTENTION_SITE", "744437513311785286LL"),
    ),
    "hyvit2": (
        ("HYVIT2_PACKED_RAW_SPM_ATTN_SITE", "2830105912667431460LL"),
        ("HYVIT2_SINGLE_RAW_SPM_ATTN_SITE", "2565594785490903973LL"),
        ("HYVIT2_PER_IMAGE_RAW_SPM_ATTN_SITE", "623508127183406940LL"),
    ),
}


def test_each_owner_exposes_same_chunk_ddr_and_raw_physical_domains():
    for owner, source in SOURCES.items():
        assert "cfg.attention_policy = ctx().has_complete_physical_manifest()" in source
        assert "? AttentionExecutionPolicy::AUTO" in source
        assert ": AttentionExecutionPolicy::DDR_KV;" in source
        assert "physical_manifest_domain_for_candidate(" in source
        assert "spm_layout.attention_policy =" in source
        assert "AttentionExecutionPolicy::SPM_KV_BY_MHA;" in source
        assert "subclass_spm_kv_by_mha_eligible(" in source
        assert "sdpa_by_mha_spm_is_valid(" in source
        # Raw attention selection stays descriptor-owned. A separate cold
        # cross-KV reuse policy may read an environment flag at construction.
        for method in ("physical_manifest_domain_for_candidate(",
                       "subclass_spm_kv_by_mha_eligible("):
            assert "std::getenv" not in _function(source, method), owner
        if owner == "gr00t_dit":
            assert "physical_manifest_for_attention_routes(" in source
            assert "/*self_raw_spm=*/true, /*cross_raw_spm=*/false" in source
            assert "/*self_raw_spm=*/false, /*cross_raw_spm=*/true" in source
            assert "/*self_raw_spm=*/true, /*cross_raw_spm=*/true" in source
        else:
            assert "ddr_layout.attention_policy = AttentionExecutionPolicy::DDR_KV;" in source


def test_every_raw_site_is_manifested_and_consumed_by_the_only_raw_forward():
    expected_launches = {
        "dino": 1,
        "siglip": 1,
        "gr00t_vl": 1,
        "gr00t_dit": 2,
        "hyvit2": 3,
    }
    for owner, sites in RAW_SITES.items():
        source = SOURCES[owner]
        for site, site_id in sites:
            assert site_id in source
            # Constant declaration, candidate manifest, exact forward receipt.
            assert source.count(site) >= 3
        assert source.count("rpu_launch_v_transpose_spm(") == expected_launches[owner]
        assert source.count("rpu_launch_sdpa_by_mha_spm(") == expected_launches[owner]
        assert source.index("rpu_launch_insert_kvcache_spm_unified_with_plan(") < source.index(
            "rpu_launch_v_transpose_spm("
        ) < source.index("rpu_launch_sdpa_by_mha_spm(")

    for owner, ddr_sites in {
        "gr00t_dit": (
            "GR00T_DIT_SELF_DDR_ATTENTION_SITE",
            "GR00T_DIT_CROSS_DDR_ATTENTION_SITE",
        ),
    }.items():
        for site in ddr_sites:
            assert SOURCES[owner].count(site) >= 3


def test_raw_routes_keep_durable_ddr_mirrors_and_stable_fallback_reasons():
    expected = {
        "dino": (
            "DINO_KV_FLAG_DDR_MIRROR",
            "DINO_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED",
            "DINO_HYBRID_FLAG_CPU_SDPA_DDR_REQUIRED",
        ),
        "siglip": (
            "SIGLIP_KV_FLAG_DDR_MIRROR",
            "SIGLIP_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED",
        ),
        "gr00t_vl": (
            "GR00T_VL_KV_FLAG_DDR_MIRROR",
            "GR00T_VL_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED",
        ),
        "gr00t_dit": (
            "GR00T_DIT_KV_FLAG_DDR_MIRROR",
            "GR00T_DIT_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED",
        ),
        "hyvit2": (
            "HYVIT2_KV_FLAG_DDR_MIRROR",
            "HYVIT2_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED",
        ),
    }
    for owner, contracts in expected.items():
        source = SOURCES[owner]
        for contract in contracts:
            assert source.count(contract) >= 2


def test_owner_capabilities_fail_closed_on_their_real_physical_abis():
    dino = SOURCES["dino"]
    for contract in (
        "chunk.len == 201",
        "use_16b_sdpa_)",
        "active_hybrid_layers < num_layers()",
        "num_layers() == 12",
        "hidden_size() == 768 && intermediate_size() == 3072",
        "/*MASK_NONE=*/0",
    ):
        assert contract in dino

    siglip = SOURCES["siglip"]
    for contract in (
        "plan.chunk_mode != ChunkMode::KV_FIRST",
        "seq % image_batch_count_ != 0",
        "image_batch_count_, per_image, per_image",
        "head_dim() == 80 && orig_head_dim_ == 72",
    ):
        assert contract in siglip

    gr00t_vl = SOURCES["gr00t_vl"]
    for contract in (
        "z2_bound_ || z2_prepared_seq_len_ != 0",
        "seq <= 0 || chunk.offset != 0",
        "chunk.kv_seq_len != seq",
        "num_layers() == 4",
        "/*MASK_NONE=*/0",
    ):
        assert contract in gr00t_vl

    gr00t_dit = SOURCES["gr00t_dit"]
    for contract in (
        "z2_bound_ || z2_prepared_seq_len_ != 0",
        "chunk.len != 41",
        "bool self_raw_spm_eligible(",
        "bool cross_raw_spm_eligible(",
        "ctx().physical_attention_policy_for_site(",
        "/*MASK_NONE=*/0",
        "/*MASK_2D=*/4",
    ):
        assert contract in gr00t_dit

    hyvit2 = SOURCES["hyvit2"]
    for contract in (
        "plan.chunk_mode != ChunkMode::KV_FIRST",
        "image_batch_count_ > 8",
        "seq != image_batch_count_ * 196",
        "const int mask_type = image_batch_count_ > 1 ? 4 : 0;",
        "/*batch=*/1, seq, seq",
    ):
        assert contract in hyvit2


def test_raw_layouts_reserve_full_kv_and_disjoint_v_transpose_storage():
    for owner, source in SOURCES.items():
        assert "ctx.attention_policy ==" in source, owner
        assert "AttentionExecutionPolicy::SPM_KV_BY_MHA" in source, owner
        assert "std::max" in source and "sdpa_tmp" in source, owner
        assert '"k"' in source and '"v"' in source and '"sdpa_tmp"' in source

    # KV_FIRST owners must extend K/V lifetime into the compute phase. The
    # sequential owners already keep all attention operands in one phase.
    for owner in ("siglip", "hyvit2"):
        source = SOURCES[owner]
        assert "raw_spm ? ALL : KVIN" in source

    assert "gr00t_pad16(std::max<int64_t>(cs, vl_seq_))" in SOURCES["gr00t_dit"]
    assert "raw_spm ? Align(cs, int64_t{16}) : cs" in SOURCES["dino"]
    assert "raw_spm ? Align(cs, int64_t{16}) : cs" in SOURCES["gr00t_vl"]
