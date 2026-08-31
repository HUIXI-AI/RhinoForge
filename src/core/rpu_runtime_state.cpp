// rpu_runtime_state.cpp — Shared C++ runtime state definitions
//
// See rpu_runtime_state.h for the symbol-ownership contract + initialization
// guarantees, including explicit zero/default initialization.

#include "rpu_runtime_state.h"

// =============================================================================
// Globals — explicit zero/default initialization.
// =============================================================================

std::unordered_map<std::string, at::Tensor> g_debug_tensors{};     // default-init = empty map

// Cross-layer batch group size (default 12). Header-owned and non-static so
// compute_chunks_impl and future callers can access the same process-wide value.
//
// This one is a genuine process-wide knob (a batching group size, identical for
// every handle and not part of any layout key), which is exactly what the chunk
// override is not.
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

bool get_debug_export() {
    return false;
}
