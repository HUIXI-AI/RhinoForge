"""Exercise real scalar and descriptor EXACT admission against sticky policy."""

from pathlib import Path
import os
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _function(source, signature):
    start = source.index(signature)
    end = source.index("(", start) + 1
    parameters = 1
    while parameters:
        parameters += (source[end] == "(") - (source[end] == ")")
        end += 1
    end = source.index("{", end) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def native_exact_probe(tmp_path_factory):
    compiler = shutil.which("g++")
    assert compiler, "native EXACT admission checks require g++"
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    codec = (ROOT / "src/core/fmb_three_stage_chunk_plan.cpp").read_text()
    functions = "\n".join(_function(base, signature) for signature in (
        "static std::vector<ChunkSizeProbe> enumerate_chunk_size_domain_impl(",
        "static std::vector<ChunkInfo> make_fmb_chunks(",
        "static int64_t legacy_chunk_override_for_exact_request(",
        "static std::vector<ChunkInfo> compute_chunks_impl(",
    ))
    shape_checks = "\n".join(_function(codec, signature) for signature in (
        "void validate_fmb_planning_shape(", "void validate_fmb_exact_chunk_size(",
    ))
    schedule = _function(base, "auto require_same_schedule = [](") + ";"
    start = base.index("        const FmbPrefillStageCandidate& candidate =\n            *planned_stage_candidate;")
    admission = base[start:base.index("        // Validate only the exact immutable winner.", start)]
    probe_struct = re.search(r"struct ChunkSizeProbe \{.*?\n\};", base, re.S)[0]
    harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
struct ChunkInfo { int idx; int64_t offset, len, kv_seq_len; };
struct BufferDecl {};
struct LayoutContext { int64_t chunk_size = 0; bool is_causal = true; };
struct FusedModelBase {
    struct Impl { int64_t chunk_size_override_ = 128; bool persistent_allocated_ = false; };
};
__PROBE_STRUCT__
struct { int64_t free_space() const { return 8 << 20; } int64_t temporary_used() const { return 0; } } SPM_ALLOC;
bool log_at(int) { return false; }
namespace detail {
int64_t estimate_temporary_total(const std::vector<BufferDecl>&) { return 16; }
bool prefer_balanced_chunk(int64_t, int64_t candidate, int64_t current) { return candidate > current; }
__SHAPE_CHECKS__
}
int64_t available_temporary_spm_after_reset(const std::vector<BufferDecl>&, bool) { return 8 << 20; }
bool final_chunk_layout_fits(const std::vector<BufferDecl>&, bool) { return true; }
__FUNCTIONS__
struct Stage { std::vector<ChunkInfo> chunks; };
struct FmbPrefillStageCandidate {
    int64_t compute_chunk_size, qkv_chunk_size;
    struct {
        Stage input, compute, qkv;
        std::vector<int> spans;
        struct { int input = 0; } boundary_policies;
    } stage_plan;
};
void validate_fmb_input_stage(const std::vector<ChunkInfo>& chunks,
        const std::vector<int>&, int, int64_t seq_len, int64_t position, const char*) {
    int64_t total = 0;
    for (const auto& chunk : chunks) total += chunk.len;
    assert(total == seq_len && chunks.front().kv_seq_len - chunks.front().len == position);
}
int64_t descriptor_admission(FusedModelBase::Impl& state,
        const FmbPrefillStageCandidate* planned_stage_candidate,
        int64_t seq_len, int64_t position) {
    auto* pimpl_ = &state;
    const bool prepared_candidate = false;
    __SCHEDULE__
    __ADMISSION__
    return candidate.compute_chunk_size;
}
FmbPrefillStageCandidate candidate(int64_t seq_len, int64_t position, int64_t capacity) {
    FmbPrefillStageCandidate out{capacity, capacity, {}};
    // Invalid EXACT capacities still receive valid schedules so rejection
    // must come from the actual capacity checks rather than fixture setup.
    out.stage_plan.input.chunks = make_fmb_chunks(seq_len, position, 16);
    out.stage_plan.compute.chunks = make_fmb_chunks(seq_len, position, std::max<int64_t>(capacity, 1));
    out.stage_plan.qkv.chunks = out.stage_plan.compute.chunks;
    return out;
}
template<class F> void rejects(F&& function) {
    bool failed = false;
    try { function(); } catch (const std::runtime_error&) { failed = true; }
    assert(failed);
}
int main(int argc, char** argv) {
    assert(argc == 2);
    const bool descriptor = std::string(argv[1]) == "descriptor";
    FusedModelBase::Impl state;
    auto request = [&](int64_t seq_len, int64_t position, int64_t capacity, bool causal) {
        const auto before = state.chunk_size_override_;
        int64_t resolved = 0;
        if (descriptor) {
            auto plan = candidate(seq_len, position, capacity);
            resolved = descriptor_admission(state, &plan, seq_len, position);
            assert(plan.compute_chunk_size == capacity && plan.qkv_chunk_size == capacity);
        } else {
            LayoutContext layout; layout.is_causal = causal;
            auto chunks = compute_chunks_impl(state,
                [](const LayoutContext&) { return std::vector<BufferDecl>{}; },
                [](int64_t) { return true; }, 128, seq_len, position, layout,
                &resolved, capacity);
            assert(chunks.front().kv_seq_len - chunks.front().len == position);
            int64_t total = 0; for (const auto& chunk : chunks) total += chunk.len;
            assert(total == seq_len);
        }
        assert(state.chunk_size_override_ == before);
        return resolved;
    };
    for (bool causal : {true, false}) {
        assert(request(128, 0, 128, causal) == 128);
        for (int64_t position : {0, 128, 129}) {
            assert(request(1, position, 16, causal) == 16);
            rejects([&] { request(1, position, 32, causal); });
            rejects([&] { request(1, position, 8, causal); });
            assert(state.chunk_size_override_ == 128);
        }
        // Multi-token EXACT never inherits the legacy clamp, even when its
        // own request would fit the rounded sequence or be a legal subchunk.
        for (int64_t length : {16, 17, 64, 128})
            rejects([&] { request(length, 0, 16, causal); });
        rejects([&] { request(128, 0, 64, causal); });
        rejects([&] { request(128, 0, 144, causal); });
        assert(state.chunk_size_override_ == 128);
        assert(request(128, 0, 128, causal) == 128);
        state.chunk_size_override_ = 0;
        assert(request(1, 128, 16, causal) == 16);
        assert(request(128, 0, 64, causal) == 64);
        state.chunk_size_override_ = 128;
    }
    if (descriptor) {
        auto bad_qkv = candidate(1, 128, 16);
        bad_qkv.qkv_chunk_size = 32;
        rejects([&] { descriptor_admission(state, &bad_qkv, 1, 128); });
        auto bad_schedule = candidate(1, 128, 16);
        ++bad_schedule.stage_plan.compute.chunks.front().kv_seq_len;
        rejects([&] { descriptor_admission(state, &bad_schedule, 1, 128); });
    } else {
        // Compare the retained historical path directly: no canonical EXACT
        // was supplied here, and both causal and noncausal calls clamp to16.
        for (bool causal : {true, false}) {
            LayoutContext layout; layout.is_causal = causal;
            int64_t resolved = 0;
            compute_chunks_impl(state,
                [](const LayoutContext&) { return std::vector<BufferDecl>{}; },
                [](int64_t) { return true; }, 128, 1, 128, layout, &resolved);
            assert(resolved == 16 && state.chunk_size_override_ == 128);
            rejects([&] { compute_chunks_impl(state,
                [](const LayoutContext&) { return std::vector<BufferDecl>{}; },
                [](int64_t) { return false; }, 128, 1, 128, layout, &resolved, 16); });
        }
    }
}
'''
    for marker, text in (("__PROBE_STRUCT__", probe_struct), ("__SHAPE_CHECKS__", shape_checks),
                         ("__FUNCTIONS__", functions), ("__SCHEDULE__", schedule),
                         ("__ADMISSION__", admission)):
        harness = harness.replace(marker, text)
    directory = tmp_path_factory.mktemp("decode-exact-legacy")
    source = directory / "probe.cpp"
    source.write_text(harness)
    binary = directory / "probe"
    compiled = subprocess.run([compiler, "-std=c++17", "-O0", str(source), "-o", str(binary)],
                              capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    return binary


@pytest.mark.parametrize("entry", ["scalar", "descriptor"])
def test_decode_exact_resolves_legacy_policy_without_clamping_descriptor(native_exact_probe, entry):
    env = dict(os.environ)
    env.pop("RPU_CHUNK_FORCE_UNSAFE", None)
    subprocess.run([str(native_exact_probe), entry], env=env,
                   check=True, capture_output=True, text=True)
