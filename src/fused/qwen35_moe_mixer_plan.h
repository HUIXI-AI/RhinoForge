#pragma once

// Model-private, host-only geometry/resource selection. This does not change
// the shared outer-window planner or construct a graph. The owner supplies
// actual-call legality and the complete model BufferDecl layout check.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace qwen35_moe_planning {

struct Request {
    int64_t outer_capacity = 0;
    int64_t execution_rows = 0;
    int64_t logical_rows = 0;
    int64_t position = 0;
    int64_t alignment = 0;
};

struct Call {
    int64_t offset = 0;          // Offset within this forward, not KV position.
    int64_t rows = 0;
    int64_t absolute_start = 0;
    int64_t valid_rows = 0;      // Can be zero; legality remains owner-defined.
    int64_t workspace_bytes = 0;
};

struct Window {
    int64_t offset = 0;
    int64_t rows = 0;
    int64_t uniform_rows = 0;
    std::vector<Call> calls;
};

struct LayoutResult {
    bool fits = false;
    uint64_t peak_bytes = 0;
};

struct Plan {
    Request request;
    int64_t mixer_capacity = 0;
    int64_t workspace_bytes = 0;
    int64_t total_calls = 0;
    uint64_t peak_bytes = 0;
    std::vector<Window> windows;
};

struct Statistics {
    uint64_t call_queries = 0;
    uint64_t call_cache_hits = 0;
    uint64_t legal_window_options = 0;
    uint64_t resource_candidates = 0;
    uint64_t layout_queries = 0;
    uint64_t layout_queries_skipped_by_score = 0;
};

enum class Failure { None, InvalidRequest, NoCallCoverage, NoFittingLayout };

struct SearchResult {
    std::optional<Plan> plan;
    Statistics statistics;
    Failure failure = Failure::None;
};

// Return nullopt for an unsupported actual call; otherwise return its already
// aligned workspace requirement (zero is valid). Configuration and resources
// captured by both callbacks must remain immutable for the duration of search.
using CallQuery = std::function<std::optional<int64_t>(const Call&)>;
// Receives mixer row capacity S and exact maximum actual-call workspace D.
// This must check the entire model layout, including outer-window buffers.
// Its result must depend only on S/D and the immutable owner context, not on
// the order of calls or on mutable "last candidate" state.
using LayoutQuery = std::function<LayoutResult(int64_t, int64_t)>;

inline SearchResult select_mixer_plan(
        const Request& request, const CallQuery& query_call,
        const LayoutQuery& query_layout) {
    SearchResult result;
    const int64_t capacity = request.outer_capacity;
    const int64_t length = request.execution_rows;
    const int64_t alignment = request.alignment;
    if (!query_call || !query_layout || alignment <= 0 || capacity <= 0 ||
            length <= 0 || capacity > length || capacity % alignment != 0 ||
            length % alignment != 0 || request.logical_rows < 0 ||
            request.logical_rows > length || request.position < 0 ||
            request.position > std::numeric_limits<int64_t>::max() - length) {
        result.failure = Failure::InvalidRequest;
        return result;
    }

    struct Option {
        int64_t rows;
        int64_t calls;
        int64_t workspace;
    };
    struct WindowOptions {
        int64_t offset;
        int64_t rows;
        std::vector<Option> options;
    };
    // Scope is one immutable request: offset/rows also uniquely determine
    // absolute_start and valid_rows. Failed queries are cached as well.
    std::map<std::pair<int64_t, int64_t>, std::optional<int64_t>> call_cache;
    const auto make_call = [&](int64_t offset, int64_t rows) {
        return Call{offset, rows, request.position + offset,
                    std::min(rows, std::max(int64_t{0},
                                           request.logical_rows - offset)), 0};
    };
    const auto workspace_for = [&](int64_t offset, int64_t rows) {
        const auto key = std::make_pair(offset, rows);
        const auto existing = call_cache.find(key);
        if (existing != call_cache.end()) {
            ++result.statistics.call_cache_hits;
            return existing->second;
        }
        ++result.statistics.call_queries;
        auto workspace = query_call(make_call(offset, rows));
        if (workspace && *workspace < 0)
            workspace.reset();
        call_cache.emplace(key, workspace);
        return workspace;
    };

    std::vector<WindowOptions> windows;
    std::vector<int64_t> workspace_levels;
    for (int64_t offset = 0; offset < length;) {
        const int64_t rows = std::min(capacity, length - offset);
        WindowOptions window{offset, rows, {}};
        for (int64_t uniform = alignment; uniform <= rows;) {
            int64_t workspace = 0;
            int64_t calls = 0;
            bool legal = true;
            for (int64_t local = 0; local < rows;) {
                const int64_t actual = std::min(uniform, rows - local);
                const auto need = workspace_for(offset + local, actual);
                if (!need) {
                    legal = false;
                    break;
                }
                workspace = std::max(workspace, *need);
                ++calls;
                local += actual;
            }
            if (legal) {
                window.options.push_back({uniform, calls, workspace});
                workspace_levels.push_back(workspace);
                ++result.statistics.legal_window_options;
            }
            if (rows - uniform < alignment)
                break;
            uniform += alignment;
        }
        if (window.options.empty()) {
            result.failure = Failure::NoCallCoverage;
            return result;
        }
        windows.push_back(std::move(window));
        offset += rows;
    }
    std::sort(workspace_levels.begin(), workspace_levels.end());
    workspace_levels.erase(
        std::unique(workspace_levels.begin(), workspace_levels.end()),
        workspace_levels.end());

    const auto option_less = [](const Option& lhs, const Option& rhs) {
        return std::tie(lhs.calls, lhs.workspace, lhs.rows) <
               std::tie(rhs.calls, rhs.workspace, rhs.rows);
    };
    struct Candidate {
        int64_t capacity;
        int64_t workspace;
        int64_t calls;
        std::vector<int> choices;
    };
    std::vector<Candidate> candidates;
    const size_t level_count = workspace_levels.size();
    const size_t window_count = windows.size();
    std::vector<std::vector<int>> best(
        window_count, std::vector<int>(level_count, -1));
    std::vector<std::vector<int>> exact(
        window_count, std::vector<int>(level_count, -1));
    std::vector<size_t> next_option(window_count, 0);

    // Incremental prefix tables avoid rescanning all window options for every
    // S/D pair. No cross-window Cartesian product and no fit-monotonicity rule.
    for (int64_t mixer_capacity = alignment; mixer_capacity <= capacity;) {
        for (size_t w = 0; w < window_count; ++w) {
            const auto& options = windows[w].options;
            auto& next = next_option[w];
            while (next < options.size() &&
                   options[next].rows <= mixer_capacity) {
                const auto& option = options[next];
                const size_t first = std::lower_bound(
                    workspace_levels.begin(), workspace_levels.end(),
                    option.workspace) - workspace_levels.begin();
                const auto update = [&](int& chosen) {
                    if (chosen < 0 || option_less(option, options[chosen]))
                        chosen = static_cast<int>(next);
                };
                update(exact[w][first]);
                for (size_t d = first; d < level_count; ++d)
                    update(best[w][d]);
                ++next;
            }
        }
        for (size_t d = 0; d < level_count; ++d) {
            std::vector<int> choices(window_count);
            int64_t calls = 0;
            int64_t actual_workspace = 0;
            bool covered = true;
            for (size_t w = 0; w < window_count; ++w) {
                choices[w] = best[w][d];
                if (choices[w] < 0) {
                    covered = false;
                    break;
                }
                const auto& option = windows[w].options[choices[w]];
                calls += option.calls;
                actual_workspace = std::max(actual_workspace, option.workspace);
            }
            if (!covered)
                continue;

            // D must be the maximum of selected actual calls, not unused
            // reserved capacity. If every free optimum uses less than D,
            // require one witness window whose real requirement is exactly D.
            // One witness suffices; all other windows stay independent.
            if (actual_workspace < workspace_levels[d]) {
                int64_t minimum_extra = std::numeric_limits<int64_t>::max();
                std::vector<int> witnessed;
                for (size_t w = 0; w < window_count; ++w) {
                    const int forced = exact[w][d];
                    if (forced < 0)
                        continue;
                    const auto& options = windows[w].options;
                    const int64_t extra =
                        options[forced].calls - options[choices[w]].calls;
                    auto replacement = choices;
                    replacement[w] = forced;
                    if (extra < minimum_extra ||
                            (extra == minimum_extra &&
                             replacement < witnessed)) {
                        minimum_extra = extra;
                        witnessed = std::move(replacement);
                    }
                }
                if (witnessed.empty())
                    continue;
                choices = std::move(witnessed);
                calls += minimum_extra;
            }
            candidates.push_back({mixer_capacity, workspace_levels[d], calls,
                                  std::move(choices)});
        }
        if (capacity - mixer_capacity < alignment)
            break;
        mixer_capacity += alignment;
    }
    result.statistics.resource_candidates = candidates.size();
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& lhs, const Candidate& rhs) {
        return std::tie(lhs.calls, lhs.capacity, lhs.workspace, lhs.choices) <
               std::tie(rhs.calls, rhs.capacity, rhs.workspace, rhs.choices);
    });

    std::optional<Candidate> winner;
    uint64_t winner_peak = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        if (winner && candidate.calls > winner->calls) {
            result.statistics.layout_queries_skipped_by_score =
                candidates.size() - i;
            break;
        }
        ++result.statistics.layout_queries;
        const auto layout = query_layout(candidate.capacity, candidate.workspace);
        if (!layout.fits)
            continue;
        if (!winner ||
                std::tie(candidate.calls, layout.peak_bytes, candidate.capacity,
                         candidate.workspace, candidate.choices) <
                std::tie(winner->calls, winner_peak, winner->capacity,
                         winner->workspace, winner->choices)) {
            winner = candidate;
            winner_peak = layout.peak_bytes;
        }
    }
    if (!winner) {
        result.failure = Failure::NoFittingLayout;
        return result;
    }
    Plan plan;
    plan.request = request;
    plan.mixer_capacity = winner->capacity;
    plan.workspace_bytes = winner->workspace;
    plan.total_calls = winner->calls;
    plan.peak_bytes = winner_peak;
    for (size_t w = 0; w < window_count; ++w) {
        const auto& source = windows[w];
        const auto& option = source.options[winner->choices[w]];
        Window window{source.offset, source.rows, option.rows, {}};
        for (int64_t local = 0; local < source.rows;) {
            const int64_t actual = std::min(option.rows, source.rows - local);
            auto call = make_call(source.offset + local, actual);
            call.workspace_bytes =
                *call_cache.at(std::make_pair(call.offset, call.rows));
            window.calls.push_back(call);
            local += actual;
        }
        plan.windows.push_back(std::move(window));
    }
    result.plan = std::move(plan);
    return result;
}

}  // namespace qwen35_moe_planning
