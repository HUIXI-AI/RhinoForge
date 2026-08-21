// rpu_runtime_state.cpp — Shared C++ runtime state definitions
//
// See rpu_runtime_state.h for the symbol-ownership contract + initialization
// guarantees, including explicit zero/default initialization.

#include "rpu_runtime_state.h"
#include "rpu_ops.h"

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

// File-local: enabled flag for debug-export. Read+written ONLY via
// set_debug_export / get_debug_export below; no consumers outside this TU.
static bool g_debug_export_enabled = false;

// =============================================================================
// Cross-layer batch group size
// =============================================================================

void set_cross_layer_batch_size(int64_t size) {
    g_cross_layer_batch_size = size;
}

int64_t get_cross_layer_batch_size() {
    return g_cross_layer_batch_size;
}

// =============================================================================
// Debug export cluster
// =============================================================================

void set_debug_export(bool enabled) {
    g_debug_export_enabled = enabled;
    if (!enabled) {
        g_debug_tensors.clear();
    }
    // This global switch deliberately does not invalidate model-owned graph
    // caches. Any op that branches on get_debug_export() at BUILD time must
    // include that mode in its signature or use a separate debug cache. The
    // G0.5 Qwen3.5 fixed-prefill adapter uses the latter; raw one-shot users
    // naturally record a fresh graph.
}

bool get_debug_export() {
    return g_debug_export_enabled;
}

at::Tensor get_debug_tensor(const std::string& name) {
    auto it = g_debug_tensors.find(name);
    if (it != g_debug_tensors.end()) {
        // Graph-backed debug buffers are written by the RPU during capture or
        // replay. The getter is the post-capture CPU boundary, so synchronize
        // here rather than baking a flush into the captured model forward.
        if (it->second.defined()
            && it->second.device().type()
                == c10::DeviceType::PrivateUse1) {
            rpu_ddr_flush_force_sized(
                it->second.data_ptr(), it->second.nbytes());
        }
        return it->second;
    }
    return at::Tensor();  // Empty tensor if not found
}

void clear_debug_tensors() {
    g_debug_tensors.clear();
}

std::vector<std::string> list_debug_tensors() {
    std::vector<std::string> names;
    for (const auto& kv : g_debug_tensors) {
        names.push_back(kv.first);
    }
    return names;
}
