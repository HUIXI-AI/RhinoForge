// rpu_runtime_state.h — Shared C++ runtime state declarations
//
// Single declaration owner for shared runtime globals.
//
// Symbols owned here:
//   - g_debug_tensors              — debug-export registry consumed by Python
//                                    rpu_backend.get_debug_tensor(name)
//   - set/get_cross_layer_batch_size — cross-layer batch group size
//   - set/get_debug_export + get_debug_tensor + clear_debug_tensors
//     + list_debug_tensors          — debug-tensor cluster
//
// Defined in the sibling TU `src/core/rpu_runtime_extras.cpp`:
//   - set/get_spm_debug                    (kept Python-bound; chat_cli caller)
//   - set/get_cross_layer_batch_prefill    (kept Python-bound; chat_cli caller)
//   - rpu_launch_all_reduce_sum_residual_kernel  (called by 4 surviving v3 files)
//
// Initialization contract:
//   g_debug_tensors is default-initialized to empty (`{}` at definition site).
//
// Convention: global namespace (matches rpu_helpers.h / rpu_kernel_decls.h);
//             no `namespace rpu` wrapper.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <ATen/ATen.h>

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
// per-handle. The routes that reach it from Python are
// rpu_causal_decoder_set_chunk_size_{override,cap} / _set_chunk_envelope,
// rpu_halo_image_flow_set_chunk_size_override, and the `chunk_size` forward
// argument the gemma-family models already took.

// Cross-layer batch group size (default 12). Kept header-owned (non-static)
// because compute_chunks_impl reads it directly.
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

// Debug export cluster (Python: torch.rpu.set_debug_export / get_debug_export /
// list_debug_tensors / get_debug_tensor / clear_debug_tensors)
void                      set_debug_export(bool enabled);
bool                      get_debug_export();
at::Tensor                get_debug_tensor(const std::string& name);
void                      clear_debug_tensors();
std::vector<std::string>  list_debug_tensors();
