"""Bounded host checks for the 4B W8 causal prefill traversal; no HW claim."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src/fused/rpu_qwen3_model.h"


def _method(source, marker):
    start = source.index(marker)
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_native_prefill_admission_and_causal_traversal(tmp_path):
    """Compile production guards/consumer/traversal and check causal dependencies."""
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ is needed for native admission checks")
    source = HEADER.read_text()
    methods = "\n".join(_method(source, marker) for marker in (
        "bool qwen3vl_4b_w8_resident_prefill(",
        "void consume_qwen3vl_4b_w8_prefill_schedule("))
    fmb = (ROOT / "src/core/fused_model_base.cpp").read_text()
    base = (ROOT / "src/core/fused_model_base.h").read_text()
    traversal = fmb[fmb.index("struct FmbExecutionTraversalSpec {"):
                    fmb.index("int64_t validate_fmb_resolved_chunks(")]
    traversal = ("namespace detail {\n" + _method(base,
        "inline bool fmb_kv_first_pair_carry_eligible(") + "\n}\n" + traversal)
    harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
#define TORCH_INTERNAL_ASSERT(condition, ...) TORCH_CHECK(condition)
enum class ChunkMode { SEQUENTIAL, KV_FIRST };
enum class InterLayerIO { SPM_RESIDENT, DDR_PINGPONG };
enum class CausalDecoderPlanningMode { ORDINARY, OTHER };
enum class FmbRouteFamily { GRAPH_SCHEDULE };
enum class SpmFmbOccurrenceKind { LayerBody, KvInsertBody };
enum class BufferScope { LayerWide, Compute, KvInsert };
using SpmFmbScopeMask = uint32_t;
constexpr SpmFmbScopeMask spm_fmb_scope_bit(BufferScope s) { return 1u << static_cast<int>(s); }
constexpr int64_t CAUSAL_DECODER_4B_W8_PREFILL_SCHEDULE_SITE = 4913775020614323787LL;
struct ChunkInfo { int64_t idx, offset, len, kv_seq_len; };
struct ChunkPlan { int64_t chunk_size = 256; };
struct Compute { ChunkPlan plan; std::vector<ChunkInfo> chunks; };
struct StagePlan { ChunkMode chunk_mode = ChunkMode::SEQUENTIAL; Compute compute; };
struct Context {
    StagePlan stage_plan;
    int64_t seq_len = 512, position = 0, batch_size = 1;
    bool is_causal = true, input_in_spm = false, output_to_spm = true;
    std::optional<int> attention_mask;
    int consumed = 0;
    std::vector<int64_t> args;
    void consume_physical_route(FmbRouteFamily family, int64_t site,
                                int64_t selector, int64_t flags,
                                std::initializer_list<int64_t> values) {
        assert(family == FmbRouteFamily::GRAPH_SCHEDULE);
        assert(site == CAUSAL_DECODER_4B_W8_PREFILL_SCHEDULE_SITE);
        assert(selector == 1 && flags == 0);
        ++consumed; args = values;
    }
};
struct Decoder {
    bool qwen3vl_4b_w8_profile_ = true, linear_acc32_ = false, adarms_ = false;
    bool decoder_mixed_precision_ = false, qwen3vl_pooler_z1_dispatch_ = false;
    bool qwen3vl_multiview_text_composite_dispatch_ = false;
    CausalDecoderPlanningMode planning_mode_ = CausalDecoderPlanningMode::ORDINARY;
    Context context;
    Context& ctx() { return context; }
    int64_t num_layers() const { return 36; }
''' + methods + r'''
};
''' + traversal + r'''
// A causal recurrence stands in for each layer's arithmetic. It requires the
// complete earlier KV prefix and includes three masked, row-dependent DS taps.
// Both schedules must assemble every output row, including a short tail.
std::vector<int64_t> run(int64_t length, bool resident) {
    Decoder d;
    auto& ctx = d.ctx(); ctx.seq_len = length;
    for (int64_t off = 0; off < length; off += 256)
        ctx.stage_plan.compute.chunks.push_back(
            {off / 256, off, std::min<int64_t>(256, length - off), std::min(off + 256, length)});
    const auto& chunks = ctx.stage_plan.compute.chunks;
    std::vector<ChunkInfo> empty;
    FmbExecutionTraversalSpec spec;
    spec.num_layers = 36; spec.cross_batch = 36;
    spec.chunks = &chunks; spec.kv_insert_chunks = &empty;
    spec.inter_layer_io = resident ? InterLayerIO::SPM_RESIDENT : InterLayerIO::DDR_PINGPONG;
    spec.chunk_outer_within_group = resident;
    std::vector<std::vector<int64_t>> kv(36), hidden(37, std::vector<int64_t>(length, -1));
    for (int64_t r = 0; r < length; ++r) hidden[0][r] = r % 29;
    std::vector<int64_t> spm;
    int64_t callbacks = 0;
    for_each_fmb_execution_step(spec, 0, [&](const FmbExecutionStepView& step) {
        const auto& chunk = *step.chunk;
        const int l = step.layer;
        assert(step.kind == SpmFmbOccurrenceKind::LayerBody);
        assert(step.group_id == 0);
        assert(step.traversal_ordinal == callbacks);
        assert(kv[l].size() == static_cast<size_t>(chunk.offset));
        assert(chunk.kv_seq_len == chunk.offset + chunk.len);
        if (resident) {
            assert(l == callbacks % 36 && chunk.idx == callbacks / 36);
            assert(step.updates_io_context);
            ctx.input_in_spm = step.input_in_spm; ctx.output_to_spm = step.output_to_spm;
            d.consume_qwen3vl_4b_w8_prefill_schedule(l, chunk);
        }
        if (!step.input_in_spm)
            spm.assign(hidden[l].begin() + chunk.offset, hidden[l].begin() + chunk.offset + chunk.len);
        for (int64_t r = 0; r < chunk.len; ++r) {
            const int64_t row = chunk.offset + r;
            assert(spm[r] >= 0);
            kv[l].push_back(spm[r]);
            const auto prefix = r + chunk.offset == 0 ? 0 : kv[l][row - 1];
            const auto ds = l < 3 && row % 7 < 3 ? (l + 1) * (row + 1) : 0;
            spm[r] = (spm[r] * 3 + prefix + ds + l) % 1000003;
            // This retained diagnostic records intermediate rows without
            // changing the resident dependency, which continues through spm.
            hidden[l + 1][row] = spm[r];
        }
        ++callbacks;
    });
    assert(callbacks == static_cast<int64_t>(chunks.size()) * 36);
    for (const auto& layer : kv) assert(layer.size() == static_cast<size_t>(length));
    if (resident) {
        assert(ctx.consumed == 1);
        assert((ctx.args == std::vector<int64_t>{length, 256, static_cast<int64_t>(chunks.size()), 36}));
    }
    return hidden.back();
}
int main() {
    Decoder d;
    auto admitted = [&](int64_t len, int64_t pos, bool causal, bool mask, int64_t batch, int64_t count) {
        return d.qwen3vl_4b_w8_resident_prefill(len, pos, causal, mask, batch, count);
    };
    assert(admitted(512, 0, true, false, 1, 2));
    assert(admitted(1024, 0, true, false, 1, 4));
    assert(!admitted(1, 0, true, false, 1, 2));
    assert(!admitted(96, 0, true, false, 1, 1));
    assert(!admitted(512, 1, true, false, 1, 2));
    assert(!admitted(512, 0, false, false, 1, 2));
    assert(!admitted(512, 0, true, true, 1, 2));
    assert(!admitted(512, 0, true, false, 2, 2));
    d.qwen3vl_4b_w8_profile_ = false; // Includes ordinary FP16 and 32B W8.
    assert(!admitted(512, 0, true, false, 1, 2)); d.qwen3vl_4b_w8_profile_ = true;
    for (auto member : {&Decoder::linear_acc32_, &Decoder::adarms_,
                       &Decoder::decoder_mixed_precision_, &Decoder::qwen3vl_pooler_z1_dispatch_,
                       &Decoder::qwen3vl_multiview_text_composite_dispatch_}) {
        d.*member = true; assert(!admitted(512, 0, true, false, 1, 2)); d.*member = false;
    }
    d.planning_mode_ = CausalDecoderPlanningMode::OTHER;
    assert(!admitted(512, 0, true, false, 1, 2));
    for (int64_t len : {512, 560, 1024}) assert(run(len, true) == run(len, false));
    for (int bad = 0; bad < 3; ++bad) {
        Decoder wrong;
        wrong.ctx().stage_plan.compute.chunks = {{0, 0, 256, 256}, {1, 256, 256, 512}};
        if (bad == 0) wrong.ctx().input_in_spm = true;
        if (bad == 1) wrong.ctx().output_to_spm = false;
        if (bad == 2) wrong.ctx().stage_plan.chunk_mode = ChunkMode::KV_FIRST;
        bool rejected = false;
        try { wrong.consume_qwen3vl_4b_w8_prefill_schedule(0, {0, 0, 256, 256}); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
}
'''
    cpp, exe = tmp_path / "resident.cpp", tmp_path / "resident"
    cpp.write_text(harness)
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    str(cpp), "-o", str(exe)], check=True, capture_output=True, text=True)
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)


