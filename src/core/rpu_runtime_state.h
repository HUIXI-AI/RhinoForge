// rpu_runtime_state.h — Shared C++ runtime state declarations
//
// Single declaration owner for shared runtime globals.
//
// Symbols owned here:
//   - g_debug_tensors               — dormant internal debug storage
//   - set/get_cross_layer_batch_size — cross-layer batch group size
//   - get_debug_export              — hard-disabled compatibility query
//   - rpu_lkn_batch_config_at_load  — immutable Launch capacity snapshot
//
// Defined in the sibling TU `src/core/rpu_runtime_extras.cpp`:
//   - set/get_spm_debug                    (kept Python-bound; chat_cli caller)
//   - set/get_cross_layer_batch_prefill    (kept Python-bound; chat_cli caller)
//   - rpu_launch_all_reduce_sum_residual_kernel  (called by 4 surviving v3 files)
//
// Initialization contract:
//   g_debug_tensors is default-initialized to empty (`{}` at definition site).
//   LKN capacity is snapshotted in rpu_backend.cpp during extension load.
//
// Convention: global namespace (matches rpu_helpers.h / rpu_kernel_decls.h);
//             no `namespace rpu` wrapper.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <ATen/ATen.h>

struct LknBatchConfig {
    int64_t max_entries;
    int64_t kd_buf_mb;
    int64_t instr_buf_mb;
};

// Immutable process snapshot taken while rpu_backend and the Launch runtime
// are loaded. Graph planning must use this instead of rereading the env.
const LknBatchConfig& rpu_lkn_batch_config_at_load();

// =============================================================================
// Globals — defined in rpu_runtime_state.cpp with explicit zero/default-init
// =============================================================================

// `g_chunk_size_override` + set_chunk_size / get_chunk_size are gone.
// Chunk size is a PER-HANDLE quantity: it keys the SPM layout (ctx.chunk_size
// feeds compute_params_hash_impl) and the graph signature. A
// process global deciding it meant one handle re-keyed another handle's
// allocation, and changing it after capture silently replayed a graph under a
// layout it was not recorded for ("DMA bytes drift at cursor N").
// The authority is FusedModelBase::set_chunk_size_override(), which was already
// per-handle. Python reaches it through
// rpu_causal_decoder_set_chunk_size_{override,cap} / _set_chunk_envelope and
// model-specific forward arguments.

// Cross-layer batch group size (default 12). Kept header-owned (non-static)
// because compute_chunks_impl reads it directly.
extern int64_t g_cross_layer_batch_size;

// Dormant internal storage retained until model-local debug branches are removed.
// get_debug_export() is hard-disabled, so release execution never populates it.
extern std::unordered_map<std::string, at::Tensor> g_debug_tensors;

// =============================================================================
// Accessor functions — defined in rpu_runtime_state.cpp
// =============================================================================

// Cross-layer batch group size (Python: torch.rpu.set_cross_layer_batch_size /
// get_cross_layer_batch_size)
void    set_cross_layer_batch_size(int64_t size);
int64_t get_cross_layer_batch_size();

bool get_debug_export();
