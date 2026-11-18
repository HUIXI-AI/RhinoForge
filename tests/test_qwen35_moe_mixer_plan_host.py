"""Compile the model-private planner and real SDPA predicates without Torch.

The resource callback below is deliberately synthetic: these checks establish
search correctness, not whole-model SPM admission or hardware correctness.
"""

from pathlib import Path
import json
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _section(source, start, end):
    begin = source.index(start)
    return source[begin:source.index(end, begin)]


@pytest.fixture(scope="module")
def mixer_plan_binary(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    sdpa = "\n".join((
        _section(helpers, "inline int64_t sdpa_tile_m_v16_initial(",
                 "// Backwards-compat alias"),
        _section(helpers, "#define SDPA_MAX_VLM_SIZE",
                 "// Shared utility: float32"),
        _section(helpers, "inline bool sdpa_try_compute_tiling(",
                 "inline bool sdpa_is_valid_chunk_size("),
    ))
    program = r'''
#include "src/fused/qwen35_moe_mixer_plan.h"
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("check failed"); } while (false)
template<class A, class B> constexpr auto CeilDiv(A a, B b) { return (a + b - 1) / b; }
SDPA_SOURCE
using namespace qwen35_moe_planning;
using Clock = std::chrono::steady_clock;
using Score = std::tuple<int64_t, uint64_t, int64_t, int64_t>;

void verify_calls(const Plan& plan, const CallQuery& query) {
    int64_t offset = 0, calls = 0, actual_workspace = 0;
    for (const auto& window : plan.windows) {
        assert(window.offset == offset);
        assert(window.rows == std::min(plan.request.outer_capacity,
                                       plan.request.execution_rows - offset));
        for (const auto& call : window.calls) {
            assert(call.offset == offset);
            assert(call.absolute_start == plan.request.position + offset);
            assert(call.rows <= plan.mixer_capacity);
            assert(call.rows == std::min(window.uniform_rows,
                                        window.offset + window.rows - offset));
            assert(call.valid_rows == std::min(call.rows, std::max(int64_t{0},
                                        plan.request.logical_rows - offset)));
            const auto need = query(call);
            assert(need && *need == call.workspace_bytes);
            actual_workspace = std::max(actual_workspace, *need);
            offset += call.rows;
            ++calls;
        }
        assert(offset == window.offset + window.rows);
    }
    assert(offset == plan.request.execution_rows);
    assert(calls == plan.total_calls);
    assert(actual_workspace == plan.workspace_bytes);
}

// Independent small-domain oracle: explicitly visit the Cartesian product
// of uniform+tail choices, then test each S with exact actual-call D. This
// expensive approach is only a test oracle, never production search.
std::optional<Score> exhaustive(const Request& req, const CallQuery& query,
                               const LayoutQuery& layout) {
    std::optional<Score> best;
    std::function<void(int64_t, int64_t, int64_t, int64_t)> visit;
    visit = [&](int64_t offset, int64_t calls, int64_t max_rows, int64_t tmp) {
        if (offset == req.execution_rows) {
            for (int64_t s = max_rows; s <= req.outer_capacity; s += req.alignment) {
                const auto fit = layout(s, tmp);
                const Score score{calls, fit.peak_bytes, s, tmp};
                if (fit.fits && (!best || score < *best))
                    best = score;
            }
            return;
        }
        const auto rows = std::min(req.outer_capacity, req.execution_rows - offset);
        for (int64_t c = req.alignment; c <= rows; c += req.alignment) {
            int64_t sub_calls = 0, sub_tmp = tmp;
            bool legal = true;
            for (int64_t r = 0; r < rows; r += c) {
                const auto q = std::min(c, rows - r);
                const auto at = offset + r;
                const Call call{at, q, req.position + at,
                    std::min(q, std::max(int64_t{0}, req.logical_rows - at)), 0};
                const auto need = query(call);
                if (!need || *need < 0) { legal = false; break; }
                sub_tmp = std::max(sub_tmp, *need);
                ++sub_calls;
            }
            if (legal)
                visit(offset + rows, calls + sub_calls,
                      std::max(max_rows, c), sub_tmp);
        }
    };
    visit(0, 0, 0, 0);
    return best;
}

CallQuery real_sdpa_query() {
    const SdpaConfig cfg{SdpaKernelType::FLASH_ATTN_SPM, 256, 16, 8, 8, 1, 0};
    return [cfg](const Call& call) -> std::optional<int64_t> {
        if (!sdpa_call_valid(cfg, call.rows, call.absolute_start))
            return std::nullopt;
        return CeilDiv(sdpa_compute_tmp_v16_size(cfg, call.rows) * 32,
                       int64_t{256}) * 256;
    };
}

void correctness() {
    for (int seed = 0; seed < 240; ++seed) {
        const int64_t align = 1 + seed % 2;
        const int64_t length = (2 + seed % 5) * align;
        const int64_t capacity = (1 + (seed / 5) % (length / align)) * align;
        const Request req{capacity, length, length - seed % 3,
                          (seed % 4) * align, align};
        const CallQuery query = [seed](const Call& c) -> std::optional<int64_t> {
            if ((c.rows * 3 + c.absolute_start * 5 + c.valid_rows + seed) % 7 == 0)
                return std::nullopt;
            return (c.rows * 5 + c.absolute_start * 3 + seed) % 5;
        };
        const LayoutQuery layout = [seed](int64_t s, int64_t d) {
            return LayoutResult{(s * 3 + d * 7 + seed) % 4 != 0,
                static_cast<uint64_t>((s * 11 + d * 13 + seed) % 17)};
        };
        const auto oracle = exhaustive(req, query, layout);
        std::set<std::pair<int64_t, int64_t>> calls_seen, layouts_seen;
        const auto actual = select_mixer_plan(req,
            [&](const Call& c) {
                assert(calls_seen.emplace(c.offset, c.rows).second);
                return query(c);
            },
            [&](int64_t s, int64_t d) {
                assert(layouts_seen.emplace(s, d).second);
                return layout(s, d);
            });
        assert(static_cast<bool>(oracle) == static_cast<bool>(actual.plan));
        assert(actual.statistics.call_queries == calls_seen.size());
        assert(actual.statistics.layout_queries == layouts_seen.size());
        if (oracle) {
            const auto& p = *actual.plan;
            assert((Score{p.total_calls, p.peak_bytes, p.mixer_capacity,
                          p.workspace_bytes} == *oracle));
            verify_calls(p, query);
            const auto repeated = select_mixer_plan(req, query, layout);
            assert(repeated.plan && repeated.plan->windows.size() == p.windows.size());
            for (size_t w = 0; w < p.windows.size(); ++w)
                assert(repeated.plan->windows[w].uniform_rows == p.windows[w].uniform_rows);
        }
    }

    // S is a row capacity, not an actual call: q=6 is never admitted, but
    // a nonmonotone complete-layout callback may admit only capacity S=6.
    const CallQuery small_calls = [](const Call& c) -> std::optional<int64_t> {
        if (c.rows != 2 && c.rows != 4) return std::nullopt;
        return c.rows == 2 ? 30 : 10;  // Tail uses more workspace.
    };
    const auto capacity_only = select_mixer_plan(
        {6, 6, 5, 0, 2}, small_calls,
        [](int64_t s, int64_t d) { return LayoutResult{s == 6 && d == 30, 123}; });
    assert(capacity_only.plan && capacity_only.plan->mixer_capacity == 6);
    assert(capacity_only.plan->windows[0].uniform_rows == 4);
    assert(capacity_only.plan->workspace_bytes == 30);
    verify_calls(*capacity_only.plan, small_calls);

    // Exact D must not be lost by choosing the lower-workspace tied option.
    const auto exact_witness = select_mixer_plan(
        {8, 8, 8, 0, 2},
        [](const Call& c) -> std::optional<int64_t> {
            if (c.rows == 2 || c.rows == 4) return 10;
            if (c.rows == 6) return 20;
            return std::nullopt;
        },
        [](int64_t s, int64_t d) { return LayoutResult{s == 6 && d == 20, 20}; });
    assert(exact_witness.plan && exact_witness.plan->total_calls == 2);
    assert(exact_witness.plan->windows[0].uniform_rows == 6);

    const auto always_fit = [](int64_t, int64_t) { return LayoutResult{true, 0}; };
    const auto no_cover = select_mixer_plan(
        {4, 8, 8, 4, 2},
        [](const Call& c) -> std::optional<int64_t> {
            return c.absolute_start >= 8 ? std::nullopt : std::optional<int64_t>{0};
        }, always_fit);
    assert(!no_cover.plan && no_cover.failure == Failure::NoCallCoverage);
    assert(no_cover.statistics.layout_queries == 0);
    const auto overflow = select_mixer_plan(
        {2, 2, 2, std::numeric_limits<int64_t>::max(), 2},
        small_calls, always_fit);
    assert(!overflow.plan && overflow.failure == Failure::InvalidRequest);

    const auto sdpa = real_sdpa_query();
    // The S bound belongs only to this synthetic resource fixture.
    const auto fixture_layout = [](int64_t s, int64_t d) {
        return LayoutResult{s <= 384, static_cast<uint64_t>(s * 4096 + d)};
    };
    const auto cliff = select_mixer_plan({704, 704, 641, 0, 64}, sdpa, fixture_layout);
    assert(cliff.plan && cliff.plan->windows[0].uniform_rows == 320);
    assert(cliff.plan->total_calls == 3 && cliff.plan->workspace_bytes == 163840);
    assert(!sdpa(Call{384, 320, 384, 257, 0}));
    const auto p753 = select_mixer_plan({768, 768, 753, 0, 64}, sdpa, fixture_layout);
    assert(p753.plan && p753.plan->total_calls == 2);
    assert(p753.plan->mixer_capacity == 384 && p753.plan->workspace_bytes == 196608);
    const auto p4096 = select_mixer_plan({768, 4096, 4096, 0, 64}, sdpa, fixture_layout);
    assert(p4096.plan && p4096.plan->total_calls == 11);
    assert(p4096.plan->workspace_bytes == 262144);
    const auto p7169 = select_mixer_plan({768, 7232, 7169, 0, 64}, sdpa, fixture_layout);
    assert(p7169.plan && p7169.plan->total_calls == 20);
    assert(p7169.plan->windows.back().offset == 6912);
    assert(p7169.plan->windows.back().uniform_rows == 256);
    assert(p7169.plan->windows.back().calls.back().rows == 64);
    for (const auto* result : {&cliff, &p753, &p4096, &p7169})
        verify_calls(*result->plan, sdpa);
    const auto no_caps = select_mixer_plan(
        {1024, 1024, 1024, 0, 64}, sdpa, always_fit);
    assert(no_caps.plan && no_caps.plan->mixer_capacity == 1024);
    assert(no_caps.plan->total_calls == 1);

    // Every aligned execution extent through 8192 gets actual-start coverage.
    // This does not run 8192 models or claim model/device correctness.
    for (int64_t e = 64; e <= 8192; e += 64) {
        const auto result = select_mixer_plan(
            {std::min(int64_t{768}, e), e, e - 1, 0, 64}, sdpa, fixture_layout);
        assert(result.plan);
        verify_calls(*result.plan, sdpa);
    }
    std::cout << "{\"oracle_cases\":240,\"aligned_extents\":128,"
                 "\"status\":\"passed\",\"whole_model_spm\":false}\n";
}

void benchmark() {
    const auto sdpa = real_sdpa_query();
    for (int64_t logical : {641, 753, 4096, 7169, 8192}) {
        const int64_t e = CeilDiv(logical, int64_t{64}) * 64;
        Statistics total;
        int64_t accepted = 0;
        std::chrono::nanoseconds query_elapsed{0}, fit_elapsed{0};
        const auto start = Clock::now();
        for (int64_t c = 64; c <= e; c += 64) {
            const auto result = select_mixer_plan({c, e, logical, 0, 64},
                [&](const Call& call) {
                    const auto before = Clock::now();
                    const auto answer = sdpa(call);
                    query_elapsed += Clock::now() - before;
                    return answer;
                },
                [&](int64_t s, int64_t d) {
                    const auto before = Clock::now();
                    // Synthetic resource cost: not the native allocator.
                    const uint64_t peak = c * 4096 + s * 12288 + d;
                    const LayoutResult answer{peak <= 8 * 1024 * 1024, peak};
                    fit_elapsed += Clock::now() - before;
                    return answer;
                });
            accepted += static_cast<bool>(result.plan);
            total.call_queries += result.statistics.call_queries;
            total.call_cache_hits += result.statistics.call_cache_hits;
            total.legal_window_options += result.statistics.legal_window_options;
            total.resource_candidates += result.statistics.resource_candidates;
            total.layout_queries += result.statistics.layout_queries;
            total.layout_queries_skipped_by_score +=
                result.statistics.layout_queries_skipped_by_score;
        }
        const double wall_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - start).count();
        const double query_ms = std::chrono::duration<double, std::milli>(
            query_elapsed).count();
        const double fit_ms = std::chrono::duration<double, std::milli>(
            fit_elapsed).count();
        std::cout << "{\"P\":" << logical << ",\"E\":" << e
                  << ",\"outer_candidates\":" << e / 64
                  << ",\"synthetic_fit_candidates\":" << accepted
                  << ",\"wall_ms\":" << wall_ms
                  << ",\"query_callback_ms\":" << query_ms
                  << ",\"synthetic_layout_callback_ms\":" << fit_ms
                  << ",\"helper_plus_timing_overhead_ms\":" << wall_ms-query_ms-fit_ms
                  << ",\"call_queries\":" << total.call_queries
                  << ",\"call_cache_hits\":" << total.call_cache_hits
                  << ",\"window_options\":" << total.legal_window_options
                  << ",\"resource_candidates\":" << total.resource_candidates
                  << ",\"layout_queries\":" << total.layout_queries
                  << ",\"layout_queries_skipped_by_score\":"
                  << total.layout_queries_skipped_by_score << "}\n";
    }
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "benchmark")
        benchmark();
    else
        correctness();
}
'''.replace("SDPA_SOURCE", sdpa)
    directory = tmp_path_factory.mktemp("qwen35-moe-mixer-plan")
    cpp, binary = directory / "mixer_plan.cpp", directory / "mixer_plan"
    cpp.write_text(program)
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         "-I", str(ROOT), str(cpp), "-o", str(binary)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    return binary


def test_mixer_search_against_exhaustive_oracle_and_actual_sdpa(mixer_plan_binary):
    result = subprocess.run(
        [str(mixer_plan_binary)], check=True, capture_output=True,
        text=True, timeout=30,
    )
    assert json.loads(result.stdout)["status"] == "passed"


def test_mixer_planning_cost_measurement(mixer_plan_binary):
    result = subprocess.run(
        [str(mixer_plan_binary), "benchmark"], check=True,
        capture_output=True, text=True, timeout=30,
    )
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["P"] for row in rows] == [641, 753, 4096, 7169, 8192]
    for row in rows:
        assert row["layout_queries"] <= row["resource_candidates"]
        print(json.dumps(row, sort_keys=True))
