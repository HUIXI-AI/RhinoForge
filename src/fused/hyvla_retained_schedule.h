#pragma once

// Frozen production choice from the same-board R25–R38 measurements.
// All candidates may coexist in the expansion library. Publication does not
// select a schedule: the retained chain uses QKV overlap, resident SwiGLU,
// O N128 and Down N64. See the source-backed migration performance report.
namespace hyvla_retained_schedule {
inline constexpr bool qkv_overlap = true;
inline constexpr bool q_n32 = false;
inline constexpr bool gateup_input_reuse = false;
inline constexpr bool down_n128 = false;
inline constexpr bool o_overlap = false;
inline constexpr bool down_overlap = false;
inline constexpr bool gateup_overlap = false;
inline constexpr bool resident_overlap = false;
inline constexpr bool sdpa_k128 = false;
inline constexpr bool sdpa_m16 = false;
}
