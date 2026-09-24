// rpu_runtime_state.h — Shared C++ runtime state declarations
//
// Single declaration owner for runtime globals originally defined in the
// v2 Qwen3 fused-decoder file. Migrated by v5-07 (Phase 07-globals-extract;
// ADR §3 G1/G2/G3); the v2 file itself was deleted in v5-08 (Phase 08;
// ADR §2 Q1).
//
// Symbols owned here:
//   - g_debug_tensors              — debug-export registry consumed by Python
//                                    rpu_backend.get_debug_tensor(name)
//   - set/get_cross_layer_batch_size — cross-layer batch group size
//   - set/get_debug_export + get_debug_tensor + clear_debug_tensors
//     + list_debug_tensors          — debug-tensor cluster
//
// Migrated to a sibling TU (NOT here) — see `src/core/rpu_runtime_extras.cpp`
// The sibling translation unit owns these public entry points:
//   - set/get_spm_debug                    (kept Python-bound; chat_cli caller)
//   - set/get_cross_layer_batch_prefill    (kept Python-bound; chat_cli caller)
//   - rpu_launch_all_reduce_sum_residual_kernel  (called by 4 surviving v3 files)
//
// Fully removed (no longer present anywhere in src/):
//   - g_fuse_lm_head + set_fuse_lm_head{,_weights} + get_fuse_lm_head
//     (v5-06 + v5-08; Python tombstones at __init__.py:127-128 via v5-01b D-08)
//   - g_chunk_size_override + set_chunk_size / get_chunk_size (MR-D, see below)
//
// Initialization contract (D-03 explicit zero/default-init):
//   g_debug_tensors is default-initialized to empty (`{}` at definition site).
//
// Author: v5-07-globals-extract atomic source commit
// Convention: global namespace (matches rpu_helpers.h / rpu_kernel_decls.h);
//             no `namespace rpu` wrapper (D-01).

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <ATen/ATen.h>

struct RpuLknBatchConfigSnapshot {
    int64_t max_entries;
    int64_t kd_buf_mb;
    int64_t instr_buf_mb;
};

// Frozen before Queue/SDK initialization. Graph execution copies these typed
// values and never rereads LKN_* from the process environment.
RpuLknBatchConfigSnapshot rpu_lkn_batch_config_at_load();
// Cold arena preflight only: SDK Queue constructors still parse their current
// environment, so reject drift before this profile allocates model weights.
RpuLknBatchConfigSnapshot rpu_lkn_batch_config_at_preflight();

// =============================================================================
// Globals — defined in rpu_runtime_state.cpp with explicit zero/default-init
// =============================================================================

// MR-D: `g_chunk_size_override` + set_chunk_size / get_chunk_size are GONE.
// Chunk size is a PER-HANDLE quantity: it keys the SPM layout (ctx.chunk_size
// feeds compute_params_hash_impl) and, since MR-D, the graph signature. A
// process global deciding it meant one handle re-keyed another handle's
// allocation, and changing it after capture silently replayed a graph under a
// layout it was not recorded for ("DMA bytes drift at cursor N").
// The authority is FusedModelBase::set_chunk_size_override(), which was already
// per-handle. The routes that reach it from Python are
// rpu_causal_decoder_set_chunk_size_{override,cap} / _set_chunk_envelope,
// rpu_halo_image_flow_set_chunk_size_override, and the `chunk_size` forward
// argument the gemma-family models already took.
// Guarded by tests/acceptance/test_chunk_authority_is_per_handle.py.

// Cross-layer batch group size (default 12). The v2 Qwen3 fused-decoder file
// (deleted in v5-08) read this directly via compute_chunks_impl rather than
// through accessors, so it MUST stay header-owned (non-static) for any future
// callers that follow the same pattern. External linkage keeps one shared
// definition across all translation units.
extern int64_t g_cross_layer_batch_size;

// Debug-export tensor registry: populated when set_debug_export(true) by v3
// model forward paths and Python-snapshot helpers; consumed via
// rpu_backend.get_debug_tensor(name) / list_debug_tensors() / clear_debug_tensors().
extern std::unordered_map<std::string, at::Tensor> g_debug_tensors;

// =============================================================================
// Accessor functions — defined in rpu_runtime_state.cpp
// =============================================================================

// Cross-layer batch group size (Python: torch.rpu.set_cross_layer_batch_size /
// get_cross_layer_batch_size)
void    set_cross_layer_batch_size(int64_t size);
int64_t get_cross_layer_batch_size();

// Public builds never capture intermediate activations. Keeping this constant
// lets the compiler remove dormant model probes while shared forward code stays
// aligned with the internal implementation.
inline constexpr bool get_debug_export() noexcept { return false; }
