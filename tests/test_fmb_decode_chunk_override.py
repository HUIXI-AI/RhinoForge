"""Exercise the actual COMPLETE capacity guard without loading an RPU model."""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _function(source, name):
    start = source.index("void " + name + "(")
    brace = source.index("{", start)
    end, depth = brace + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_complete_decode_preserves_prefill_policy_and_rejects_wrong_capacity(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    stages = (ROOT / "src/core/fmb_three_stage_chunk_plan.cpp").read_text()
    end = base.index("\n\n        // Validate only the exact immutable winner.")
    start = base.rindex("        const int64_t effective_legacy_override =", 0, end)
    guard = base[start:end]
    helper_start = base.index("static int64_t legacy_chunk_override_for_exact_request(")
    helper_end = base.index("\n}\n", helper_start) + len("\n}\n")
    request_local_override = base[helper_start:helper_end]
    validators = "\n".join(_function(stages, name) for name in (
        "validate_fmb_planning_shape", "validate_fmb_exact_chunk_size",
    ))
    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
''' + validators + request_local_override + r'''
struct State { int64_t chunk_size_override_; };
struct Candidate { int64_t compute_chunk_size; };
void dispatch(State& state, int64_t seq_len, int64_t capacity) {
    auto* pimpl_ = &state;
    const Candidate candidate{capacity};
    validate_fmb_exact_chunk_size(capacity, seq_len, "actual COMPLETE guard");
''' + guard + r'''
}
void reject(State& state, int64_t length, int64_t capacity) {
    const auto saved = state.chunk_size_override_;
    bool rejected = false;
    try { dispatch(state, length, capacity); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected && state.chunk_size_override_ == saved);
}
int main() {
    for (int64_t configured : {0, 16, 256, 512}) {
        State state{configured};
        dispatch(state, 1, 16);
        assert(state.chunk_size_override_ == configured);
        for (int64_t bad : {0, 15, 32}) reject(state, 1, bad);
    }
    State fixed{256};
    // Actual failure sequence: explicit prefill -> several decodes -> prefill.
    // Dispatch neither rewrites the incoming capacity nor the sticky policy.
    dispatch(fixed, 256, 256);
    for (int step = 0; step < 4; ++step) dispatch(fixed, 1, 16);
    dispatch(fixed, 256, 256);
    assert(fixed.chunk_size_override_ == 256);
    for (int64_t length : {8, 16, 128, 256}) reject(fixed, length, 16);
    reject(fixed, 256, 128);
    reject(fixed, 128, 256);
    State automatic{0};
    dispatch(automatic, 256, 128);
    State exact{128};
    dispatch(exact, 256, 128);
    reject(exact, 256, 256);
}
'''
    source, executable = tmp_path / "guard.cpp", tmp_path / "guard"
    source.write_text(program)
    subprocess.run([compiler, "-std=c++17", "-O0", str(source), "-o", str(executable)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
