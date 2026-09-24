"""Run the production SigLIP eligibility method with board-free geometry doubles."""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_packed_attention_candidate_is_independent_of_previous_forward(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    source = (ROOT / "src/fused/rpu_siglip_model.cpp").read_text()
    method = source[source.index("    bool subclass_spm_kv_by_mha_eligible("):]
    method = method.split("\n    // ========================================================================", 1)[0]
    method = method.replace(") const override {", ") const {")
    program = r'''
#include <cstdint>
#include <vector>
constexpr int NUM_CORES = 8;
enum class ChunkMode { KV_FIRST };
enum class AttentionExecutionPolicy { DDR_KV, SPM_KV_BY_MHA };
struct ChunkInfo { int64_t offset, len, kv_seq_len; };
struct Stage { std::vector<ChunkInfo> chunks; };
struct Span { int64_t offset, len; };
struct FmbThreeStageChunkPlan {
    ChunkMode chunk_mode = ChunkMode::KV_FIRST;
    Stage input, qkv, compute;
    std::vector<Span> spans;
};
struct LayoutContext {
    bool is_causal = false, use_attn_mask = false;
    int64_t batch_size = 1;
    AttentionExecutionPolicy attention_policy = AttentionExecutionPolicy::SPM_KV_BY_MHA;
};
// This test concerns plan ownership. Kernel-geometry validity is a fixed
// accepted capability, so no SDK symbol or device access is needed.
bool sdpa_by_mha_spm_is_valid(int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t) { return true; }
class SiglipGeometry {
public:
    int64_t seq_len_ = 0, image_batch_count_ = 3, orig_head_dim_ = 72;
    std::vector<int> layer_weights_ = std::vector<int>(27);
    int64_t num_layers() const { return 27; }
    int64_t num_cores() const { return 8; }
    int64_t hidden_size() const { return 1152; }
    int64_t intermediate_size() const { return 4352; }
    int64_t num_q_heads() const { return 16; }
    int64_t num_kv_heads() const { return 16; }
    int64_t head_dim() const { return 80; }
''' + method + r'''
};
int main() {
    SiglipGeometry model;
    LayoutContext layout;
    FmbThreeStageChunkPlan plan;
    plan.input.chunks = {{0,256,256}, {256,256,512}, {512,256,768}};
    plan.qkv.chunks = {{0,768,768}};
    plan.compute.chunks = {{0,768,768}};
    plan.spans = {{0,256}, {256,256}, {512,256}};
    // Cold planning, the matching forward, and a previous different shape
    // must all resolve the same physical attention capability.
    for (int64_t previous : {0, 768, 1536}) {
        model.seq_len_ = previous;
        if (!model.subclass_spm_kv_by_mha_eligible(plan, layout, 0)) return 1;
    }
    // Removing the unrelated previous-forward dependency must not weaken
    // the packed coverage, masking, or causal/position guards.
    plan.spans[2].len = 240;
    if (model.subclass_spm_kv_by_mha_eligible(plan, layout, 0)) return 2;
    plan.spans[2].len = 256;
    layout.use_attn_mask = true;
    if (model.subclass_spm_kv_by_mha_eligible(plan, layout, 0)) return 3;
    layout.use_attn_mask = false;
    layout.is_causal = true;
    if (model.subclass_spm_kv_by_mha_eligible(plan, layout, 0)) return 4;
    layout.is_causal = false;
    if (model.subclass_spm_kv_by_mha_eligible(plan, layout, 1)) return 5;
}
'''
    path = tmp_path / "siglip_eligibility.cpp"
    path.write_text(program)
    executable = tmp_path / "siglip_eligibility"
    subprocess.run([compiler, "-std=c++17", str(path), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
