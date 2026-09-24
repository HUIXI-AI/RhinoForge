// rpu_runtime_state.cpp — Shared C++ runtime state definitions
//
// See rpu_runtime_state.h for the symbol-ownership contract + initialization
// guarantees (D-03 explicit zero/default-init at definition site).
//
// Bodies in this file were migrated byte-equal from the v2 Qwen3
// fused-decoder file by v5-07-globals-extract (ADR §3 G1/G2/G3); the v2
// file itself was deleted in v5-08 (ADR §2 Q1).
//
// Author: v5-07-globals-extract atomic source commit

#include "rpu_runtime_state.h"
#include "rpu_ops.h"

// =============================================================================
// Globals — explicit zero/default-init per D-03
// =============================================================================

std::unordered_map<std::string, at::Tensor> g_debug_tensors{};     // default-init = empty map

// Cross-layer batch group size (default 12). Header-owned (non-static): the v2
// Qwen3 fused-decoder file (deleted in v5-08) read this directly via
// compute_chunks_impl rather than through accessors, so it must stay extern
// for any future callers following the same pattern. D-03 brace-init.
// Keep one process-wide definition with external linkage.
//
// This one is a genuine process-wide knob (a batching group size, identical for
// every handle and not part of any layout key), which is exactly what the chunk
// override was NOT — see the MR-D note in rpu_runtime_state.h.
int64_t g_cross_layer_batch_size{12};

// =============================================================================
// Cross-layer batch group size
// =============================================================================

void set_cross_layer_batch_size(int64_t size) {
    g_cross_layer_batch_size = size;
}

int64_t get_cross_layer_batch_size() {
    return g_cross_layer_batch_size;
}
