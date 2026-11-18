"""Board-free source contract for the widened SDPA admission rule.

The native extension is not available on ordinary host runners, so these checks
lock the two layers of the contract without pretending to provide hardware
evidence: the shared generator predicate admits only the structural
multiple-of-eight boundary, while a model capability may retain a narrower
numeric ceiling for a measured-bad geometry.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPERS = (ROOT / "src/core/rpu_helpers.h").read_text(encoding="utf-8")
QWEN3 = (ROOT / "src/fused/rpu_qwen3_model.h").read_text(encoding="utf-8")
QWEN35 = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text(encoding="utf-8")


def _section(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    return source[begin : source.index(end, begin)]


def test_shared_predicate_and_profile_cap_are_distinct():
    """Structural admission is widened; an optional cap remains explicit."""
    assert "int64_t max_sync_product = 0;" in HELPERS
    assert "if (cfg.max_sync_product < 0) return false;" in HELPERS
    assert "if (call_cfg.max_sync_product > 0 && product > call_cfg.max_sync_product)" in HELPERS
    assert "if (product > 8 && product % 8 != 0) return false;" in HELPERS
    assert "product <= 8 || product % 8 == 0 || sync_force_unsafe" in HELPERS


def test_qwen3_06b_numeric_cap_is_geometry_scoped():
    """The known Qwen3-0.6B product-16 failure is not a global rollback."""
    make_cfg = _section(
        QWEN3,
        "SdpaConfig make_sdpa_config(int mask = 1) const",
        "// ── Model state",
    )
    for token in (
        "hidden_size() == 1024 && intermediate_size() == 3072",
        "num_q_heads() == 16 && num_kv_heads() == 8",
        "head_dim() == 128",
        "has_qk_norm_ && !has_qkv_bias_",
        "qwen3_06b_shape ? 8 : 0",
    ):
        assert token in make_cfg
    # Other profiles keep the zero/default capability and therefore use the
    # shared structural rule (the expression has exactly one narrowed branch).
    assert make_cfg.count("qwen3_06b_shape ? 8 : 0") == 1


def test_qwen3_and_qwen35_planner_routes_call_the_shared_validity_helper():
    qwen3_valid = _section(
        QWEN3,
        "bool subclass_chunk_size_valid(int64_t cs, int64_t seq_len, int64_t position)",
        "int64_t subclass_chunk_size_cap",
    )
    qwen35_valid = _section(
        QWEN35,
        "bool Qwen3_5Model::subclass_chunk_size_valid(int64_t cs, int64_t seq_len,",
        "// MR-A: Qwen3.5 had no cap hook",
    )
    assert "sdpa_is_valid_chunk_size(make_sdpa_config" in qwen3_valid
    assert "sdpa_is_valid_chunk_size(make_sdpa_config" in qwen35_valid


def test_unsafe_launcher_bypass_is_diagnostic_only():
    marker = "RPU_CHUNK_FORCE_UNSAFE"
    assert marker in HELPERS
    unsafe = HELPERS[HELPERS.index(marker) : HELPERS.index("if (attn_mask_type == 1)", HELPERS.index(marker))]
    assert "Diagnostic only" in unsafe
    assert "profile-specific certificates" in unsafe


def test_native_ltm_candidate_search_keeps_boundaries(tmp_path):
    """Compile the actual planner and keeper helpers without loading a device."""
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ unavailable")
    pure = _section(HELPERS, "inline int64_t sdpa_tile_m_v16_initial",
                    "// Shared utility: float32")
    pure += _section(HELPERS, "inline bool sdpa_try_compute_tiling",
                     "// Non-throwing capability predicate for the generic raw-SPM")
    pure += _section(HELPERS, "inline void sdpa_validate_kernel_call",
                     "// 清零 kernel")
    source = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
template<class A, class B> int64_t CeilDiv(A a, B b) { return (a+b-1)/b; }
''' + pure + r'''
int main() {
    unsetenv("RPU_CHUNK_FORCE_UNSAFE");
    SdpaConfig cfg{SdpaKernelType::FLASH_ATTN_SPM_VCTXLEN, 256, 8, 4, 4, 1};
    const auto repaired = sdpa_compute_tiling(cfg, 640);
    assert(repaired.tile_m == 80 && repaired.tile_k == 80);
    assert(sdpa_is_valid_chunk_size(cfg, 640, 4096, 0));
    assert(sdpa_compute_tmp_v16_size(cfg, 640) * 32 == 327680);
    for (int64_t offset = 0; offset < 4096; offset += 640) {
        const auto rows = std::min<int64_t>(640, 4096-offset);
        const auto t = sdpa_compute_tiling(cfg, rows);
        assert(sdpa_call_valid(cfg, rows, offset));
        sdpa_validate_kernel_call(rows, offset+rows,
            t.tile_m_v16, t.tile_n_v16, t.tile_k_v16, t.tile_m,
            CeilDiv(rows,t.tile_m), 2, 16,16,16,1);
    }
    // A single-token tail retains the launcher's MASK_NONE dispatch.
    assert(sdpa_is_valid_chunk_size(cfg,640,3841,0));
    assert(!sdpa_is_valid_chunk_size(cfg,640,4095,0));
    assert(!sdpa_is_valid_chunk_size(cfg,640,4096,16));
    cfg.max_sync_product = 8;
    assert(!sdpa_is_valid_chunk_size(cfg,640,4096,0));
    cfg.max_sync_product = -1;
    assert(!sdpa_is_valid_chunk_size(cfg,640,4096,0));
    cfg.max_sync_product = 0;

    // Existing legal tiles are unchanged, including non-power-of-two tiles.
    for (const auto& expected : std::array<std::array<int64_t,3>,7>{{
            {{256,64,128}}, {{320,80,80}}, {{384,96,96}}, {{448,112,112}},
            {{512,64,128}}, {{768,64,128}}, {{1024,64,128}} }}) {
        const auto t = sdpa_compute_tiling(cfg, expected[0]);
        assert(t.tile_m == expected[1] && t.tile_k == expected[2]);
        assert(sdpa_is_valid_chunk_size(cfg,expected[0],2*expected[0],0));
    }
    // The measured-bad accumulated sync product remains rejected.
    SdpaConfig bad{SdpaKernelType::FLASH_ATTN_SPM_VCTXLEN,128,16,8,8,1};
    assert(!sdpa_call_valid(bad,288,288));
    for(const auto& expected:std::array<std::array<int64_t,3>,2>{{
            {{640,80,80}},{{768,96,96}} }}) {
        const auto t=sdpa_compute_tiling(bad,expected[0]);
        assert(t.tile_m==expected[1] && t.tile_k==expected[2]);
        assert(sdpa_is_valid_chunk_size(bad,expected[0],4096,0));
        for(int64_t offset=0;offset<4096;offset+=expected[0]) {
            const auto rows=std::min<int64_t>(expected[0],4096-offset);
            const auto tile=sdpa_compute_tiling(bad,rows);
            assert(sdpa_call_valid(bad,rows,offset));
            sdpa_validate_kernel_call(rows,offset+rows,tile.tile_m_v16,tile.tile_n_v16,
                tile.tile_k_v16,tile.tile_m,CeilDiv(rows,tile.tile_m),2,8,8,8,1);
        }
        bad.max_sync_product=8;
        assert(!sdpa_is_valid_chunk_size(bad,expected[0],4096,0));
        bad.max_sync_product=0;
        assert(!sdpa_is_valid_chunk_size(bad,expected[0],4095,0));
        assert(!sdpa_is_valid_chunk_size(bad,expected[0],4096,16));
    }
    static_assert(kSdpaTilePolicyBase == 3);
}
'''
    path = tmp_path / "sdpa_candidate.cpp"
    path.write_text(source)
    binary = path.with_suffix("")
    subprocess.run([compiler, "-std=c++17", "-O0", str(path), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
