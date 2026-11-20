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
#include <cstdlib>
#include <cstring>

bool rpu_concurrent_safe_spm_enabled() {
    const char* raw = std::getenv("RPU_CONCURRENT_SAFE_SPM");
    if (raw == nullptr || raw[0] == '\0' || std::strcmp(raw, "0") == 0) {
        return false;
    }
    if (std::strcmp(raw, "1") == 0) {
        return true;
    }
    TORCH_CHECK(false,
                "RPU_CONCURRENT_SAFE_SPM must be unset, empty, '0', or '1'; got '",
                raw, "'");
}

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

// Changed through the debug export accessors.
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
        // Graph-backed debug buffers are written by the device during capture or
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
