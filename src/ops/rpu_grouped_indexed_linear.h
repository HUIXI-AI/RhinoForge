#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include "rpu_kernel_cache.h"

namespace rpu_grouped_indexed_linear {

// Values are the rhino-ops ABI weight-dtype codes. Activations are always FP16.
enum class WeightType : int { Int8 = 1, Float16 = 2, Float8E4M3 = 5, Uint4 = 8 };
struct Shape {
  bool indexed;
  int64_t T, R, E, N, K, Q, TP; // Physical per-core N/K, resident T, routed R.
  WeightType weight_type;
  bool col_parallel;
};
struct Selection {
  int m, n, k, expert_partitions;
};
struct Tile {
  bool indexed;
  WeightType weight_type;
  KernelId id;
  const char *symbol;
  int m, n, k;
};
struct Layout {
  uint64_t weight_expert_bytes, scale_qblock_bytes, scale_expert_bytes;
};
// Offline measured data uses padded physical N/K, just like Shape. Scores are
// normalized measured costs, never generated from the candidate catalog.
struct RankedTile {
  int m, n, k;
  double cost;
};
struct Measurement {
  bool indexed;
  WeightType weight_type;
  std::array<int64_t, 7> shape; // T, R, E, local N, local K, cores, Q.
  int k_tile;
  const RankedTile *ranked;
  size_t ranked_count;
};
enum class SelectionSource { Canonical, Exact, Neighbor, PlExact, PlNeighbor, PlCanonical };
struct SelectionResult {
  Selection selection;
  SelectionSource source;
};
using TileAvailable = bool (*)(KernelId);

void validate_shape(const Shape &shape);
Layout layout(const Shape &shape);
// Pure geometry; does not query the device or require a REF. All selected tiles
// must subsequently be resolved against the installed optional catalog.
Selection canonical(const Shape &shape);
const Tile &selected_tile(const Shape &shape, const Selection &selection);
// Retained measured fused policy: exact -> neighboring measured shape ->
// canonical, using the current ABI. Does not inspect route data. Null availability admits all compiled
// candidates, useful for host parity checks; resolve filters installed payloads.
SelectionResult pick(const Shape &shape, const Measurement *measurements,
                     size_t count, TileAvailable available = nullptr);
Selection resolve(const Shape &shape);

// Explicit current-main ACC16 policy for operator conformance/tuning. Uses
// the ordinary PL M1 rankings, not fused measurements; FP8 borrows INT8 and
// Q64/128 borrow G32. Installed payloads and whole-Q capacity filter first.
// Existing model routes keep resolve() and their measured schedules until a
// matched comparison supports changing them. No environment or model opt-in.
SelectionResult pick_current_auto(const Shape &shape, TileAvailable available = nullptr);
Selection resolve_current_auto(const Shape &shape);

// Counts are nonnegative I16[E] summing to R. IDs are U16[R] stored as raw bits
// in I16 storage, in [0,T). Weighted scatter additionally requires unique IDs
// within each expert; different experts may target the same token. The caller skips R0.
// This CPU proof is used by fixtures, not by device-produced hot routes.
void validate_routes(const Shape &shape, const int16_t *counts, const uint16_t *ids);
} // namespace rpu_grouped_indexed_linear

// All SPM operands use absolute core-0 addresses at identical per-core offsets.
// X: [T,K] indexed or [R,K] scatter; output: [R,N] or initialized [T,N].
// Counts/IDs/coefficients are replicated. Starts are 32B aligned; route arrays
// reserve align(E/R,16) entries. Operands must be disjoint for the entire call.
//
// DDR owners must already use the new main parallel-linear format:
// W4: grouped_uint4_n32k32_v1, U8; W8/W16: N16 K32B swizzle, I8/U8/FP16.
// E4M3 uses raw U8 codes and the W8 payload with weight-dtype register 5;
// it currently borrows INT8 tile rankings; these are not FP8 measurements.
// Each owner contains E contiguous expert payloads. W4 S/Z are FP16
// [E,ceil((K/Q)/4),ceil(N/64),TP,4,64], aligned TP*512. W8 S is FP16
// [E,align(2*(col?N*TP:N),256)/2], aligned 256. W16 has no S/Z.
// No conversion or allocation occurs here. The model retains all owners.
// Typed DDR staging retains provenance in Graph; unsupported tiles/owners fail
// before flush or submission. Forced selections never silently fall back.
void rpu_launch_grouped_indexed_linear_selected(
    const rpu_grouped_indexed_linear::Shape &shape,
    const rpu_grouped_indexed_linear::Selection &selection, uint32_t input_spm_addr,
    const at::Tensor &weight, uint32_t counts_spm_addr, const c10::optional<at::Tensor> &scale,
    const c10::optional<at::Tensor> &negative_zero, uint32_t ids_spm_addr,
    uint32_t coefficients_spm_addr, uint32_t output_spm_addr);
