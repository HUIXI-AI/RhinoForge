import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_kvinsert_optional_abi_keeps_spm_rows_after_batch_and_flags() -> None:
    tail = re.compile(
        r"int num_cores(?: = 8)?,\s*"
        r"int64_t cache_batch_offset_elems(?: = 0)?,\s*"
        r"bool allow_non8_v16(?: = false)?,\s*"
        r"bool allow_hybrid_v16(?: = false)?,\s*"
        r"int64_t spm_rows(?: = 0)?"
    )
    header = (ROOT / "src/core/rpu_kernel_decls.h").read_text(encoding="utf-8")
    implementation = re.sub(
        r"//[^\n]*",
        "",
        (ROOT / "src/ops/rpu_llama_kvcache.cpp").read_text(encoding="utf-8"),
    )
    assert len(tail.findall(header)) == 2
    assert len(tail.findall(implementation)) == 2


def test_hyvla_kvinsert_passes_spm_rows_in_its_own_abi_slot() -> None:
    suffixes = {
        "src/fused/rpu_hyvit2_vision_model.cpp": ("HYVIT2_KV_SITE", r"num_cores\(\)", "nq"),
        "src/fused/rpu_hyvla_expert_model.cpp": ("HYVLA_EXPERT_KV_SITE", "tp", "nkv"),
    }
    for path, (site, cores, heads) in suffixes.items():
        source = (ROOT / path).read_text(encoding="utf-8")
        assert source.count(
            "rpu_launch_insert_kvcache_spm_unified_with_plan("
        ) == 1
        # Capability validation may also restore the plan; the launch-site
        # decoder and its exact SPM-rows ABI below are the relevant contract.
        assert "rpu_kvinsert_segment_plan_from_route_arguments(" not in source
        assert f"{site}, manifest.graph_lifecycle," in source
        plan_decode = re.compile(
            rf"restore_kvinsert_plan\(\s*{site},\s*"
            rf"route\.arguments,\s*{cores},\s*{heads},\s*hd\);"
        )
        assert len(plan_decode.findall(source)) == 1
        call_suffix = re.compile(
            rf"{cores},\s*/\*k_cache_batch_offset_elems=\*/0,\s*"
            r"/\*v_cache_batch_offset_elems=\*/0,\s*"
            r"kv_pad_rows_,\s*kv_plan\);"
        )
        assert len(call_suffix.findall(source)) == 1


def test_kvinsert_launcher_does_not_guess_a_hyvla_abi_mismatch() -> None:
    source = (ROOT / "src/ops/rpu_llama_kvcache.cpp").read_text(encoding="utf-8")
    assert "resolve_hyvla_spm_rows_compat" not in source
    assert "cache_batch_offset_elems == padded_rows" not in source
