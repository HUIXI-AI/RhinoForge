"""DINO's actual native candidate guard must precede single-chunk consumers."""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_native_dino_feasibility_filters_small_chunks_before_manifest():
    source = (ROOT / "src/fused/rpu_dinov3_vision_model.cpp").read_text()
    guard = re.search(
        r"bool subclass_chunk_size_valid\(\s*int64_t chunk_size, int64_t seq_len,"
        r"\s*int64_t /\*position\*/\) const override \{([^}]+)\}", source)
    assert guard is not None
    assert "dinov3_execution_length_valid(seq_len, identity_rope_)" in guard.group(1)
    assert "seq_len <= chunk_envelope().max_kv_len && chunk_size >= seq_len;" in guard.group(1)
    assert "TORCH_CHECK(plan.num_chunks == 1" in source
    assert "TORCH_CHECK(plan.compute.chunks.size() == 1" in source
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    domain = base[base.index("std::vector<FmbPrefillStageCandidate> result;"):]
    assert domain.index("if (!probe.valid || !probe.fits) continue;") < domain.index(
        "planning_dynamic_config(compute_plan)")
    # 224x224 DINO ViT-B has CLS + 4 registers + 196 patches. Both AUTO's
    # first feasible candidate and external C208 retain one full attention span.
    rows = 201
    candidates = [chunk for chunk in range(16, 225, 16) if chunk >= rows]
    assert candidates == [208, 224]
    assert all((rows + chunk - 1) // chunk == 1 for chunk in candidates)
