// Generic indexed grouped projection / weighted scatter-add ABI v2.
// Kernel source, arithmetic, packing specification and tile tuning live in
// rhino-ops. This file owns typed DDR staging and backend launch contracts.
#include <ATen/ATen.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <tuple>
#include <vector>

#include "rpu_ops.h"
#include "rpu_grouped_indexed_linear.h"
#include "rpu_linear_tiling.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;
namespace gil = rpu_grouped_indexed_linear;

namespace {
uint64_t gil_align(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}
const gil::Tile kGilTiles[] = {
#define GIL(indexed, dtype, id, symbol, m, n, k)                                                   \
  {indexed, static_cast<gil::WeightType>(dtype), KernelId::id, symbol, m, n, k},
#include "rpu_grouped_indexed_linear_tiles.inc"
#include "rpu_grouped_indexed_linear_current_tiles.inc"
#undef GIL
};
#include "rpu_grouped_indexed_linear_measurements.inc"

struct GilSpmSpan {
  uint32_t address;
  uint64_t bytes;
  const char *name;
};
void gil_check_spm(std::initializer_list<GilSpmSpan> spans) {
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "grouped indexed linear: SPM is uninitialized");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  for (auto current = spans.begin(); current != spans.end(); ++current) {
    TORCH_CHECK(current->address >= base && current->address % 32 == 0 && current->bytes > 0 &&
                    uint64_t(current->address) - base + current->bytes <=
                        SpmAllocator::SPM_USABLE &&
                    uint64_t(current->address) + current->bytes <= (UINT64_C(1) << 32),
                "grouped indexed linear: invalid complete absolute SPM span for ", current->name);
    for (auto other = spans.begin(); other != current; ++other)
      TORCH_CHECK(uint64_t(current->address) + current->bytes <= other->address ||
                      uint64_t(other->address) + other->bytes <= current->address,
                  "grouped indexed linear: simultaneous operands overlap: ", current->name, " and ",
                  other->name);
  }
}
void gil_check_spans(const gil::Shape &s, uint32_t x, uint32_t counts, uint32_t ids, uint32_t coeff,
                     uint32_t y) {
  const GilSpmSpan xs{x, uint64_t(s.indexed ? s.T : s.R) * s.K * 2, "X"};
  const GilSpmSpan ys{y, uint64_t(s.indexed ? s.R : s.T) * s.N * 2, "Y/partial"};
  const GilSpmSpan cs{counts, gil_align(s.E, 16) * 2, "counts"};
  const GilSpmSpan is{ids, gil_align(s.R, 16) * 2, "IDs"};
  if (s.indexed)
    gil_check_spm({xs, ys, cs, is});
  else
    gil_check_spm({xs, ys, cs, is, {coeff, gil_align(s.R, 16) * 2, "coefficients"}});
}
at::ScalarType gil_weight_dtype(gil::WeightType type) {
  return (type == gil::WeightType::Uint4 || type == gil::WeightType::Float8E4M3) ? at::kByte
         : type == gil::WeightType::Int8 ? at::kChar
                                         : at::kHalf;
}
void gil_check_owners(const gil::Shape &s, const at::Tensor &weight,
                      const c10::optional<at::Tensor> &scale,
                      const c10::optional<at::Tensor> &zero) {
  const auto l = gil::layout(s);
  TORCH_CHECK(
      weight.defined() && weight.device().type() == at::kPrivateUse1 &&
          weight.scalar_type() == gil_weight_dtype(s.weight_type) && weight.is_contiguous() &&
          uint64_t(weight.nbytes()) == uint64_t(s.E) * l.weight_expert_bytes,
      "grouped indexed linear: weight must own complete contiguous new-format expert payloads");
  const auto metadata = [&](const c10::optional<at::Tensor> &owner,
                            const char *name) -> const at::Tensor & {
    TORCH_CHECK(owner.has_value() && owner->defined() && owner->device() == weight.device() &&
                    owner->scalar_type() == at::kHalf && owner->is_contiguous() &&
                    uint64_t(owner->nbytes()) == uint64_t(s.E) * l.scale_expert_bytes,
                "grouped indexed linear: invalid FP16 ", name, " owner");
    return *owner;
  };
  if (s.weight_type == gil::WeightType::Uint4) {
    for (auto *owner : {&metadata(scale, "scale"), &metadata(zero, "negative zero")})
      TORCH_CHECK(
          owner->dim() == 6 && owner->size(0) == s.E && owner->size(1) == (s.K / s.Q + 3) / 4 &&
              owner->size(2) == (s.N + 63) / 64 && owner->size(3) == s.TP && owner->size(4) == 4 &&
              owner->size(5) == 64,
          "grouped indexed linear: W4 metadata layout must be [E,Qblock4,Nblock64,TP,4,64]");
  } else if (s.weight_type == gil::WeightType::Int8 ||
             s.weight_type == gil::WeightType::Float8E4M3) {
    const auto &owner = metadata(scale, "per-channel scale");
    TORCH_CHECK(owner.dim() == 2 && owner.size(0) == s.E &&
                    owner.size(1) == int64_t(l.scale_expert_bytes / 2) && !zero.has_value(),
                "grouped indexed linear: W8 requires padded [E,Nphysical] scale and no zero");
  } else {
    TORCH_CHECK(!scale.has_value() && !zero.has_value(),
                "grouped indexed linear: W16 has no scale/zero operands");
  }
}
void gil_set_u32(Kernel_t *kernel, int slot, uint32_t value) {
  kernel->set_regs(slot, static_cast<uint16_t>(value));
  kernel->set_regs(slot + 1, static_cast<uint16_t>(value >> 16));
}
} // namespace

namespace rpu_grouped_indexed_linear {
void validate_shape(const Shape &s) {
  TORCH_CHECK(s.T > 0 && s.T <= UINT16_MAX && s.R > 0 && s.R <= INT16_MAX && s.E > 0 &&
                  s.E <= INT16_MAX && (s.TP == 1 || s.TP == 2 || s.TP == 4 || s.TP == 8) &&
                  s.N > 0 && s.N <= 16 * 65536 && s.K > 0 && s.K <= UINT16_MAX && s.K % 32 == 0,
              "grouped indexed linear: invalid physical geometry; T is U16, R/E positive I16, TP "
              "is 1/2/4/8, N16 tile base is U16");
  TORCH_CHECK(s.weight_type == WeightType::Uint4 || s.weight_type == WeightType::Int8 ||
                  s.weight_type == WeightType::Float8E4M3 || s.weight_type == WeightType::Float16,
              "grouped indexed linear: unsupported weight precision");
  if (s.weight_type == WeightType::Uint4) {
    TORCH_CHECK(
        s.N % 32 == 0 && (s.Q == 32 || s.Q == 64 || s.Q == 128) && s.K % s.Q == 0,
        "grouped indexed linear: W4 requires physical N32/K32 and complete Q32/64/128 groups");
  } else {
    TORCH_CHECK(s.N % 16 == 0 && s.Q == 0, "grouped indexed linear: W8/W16 require N16 and Q0");
  }
}
Layout layout(const Shape &s) {
  validate_shape(s);
  const uint64_t elements = uint64_t(s.TP) * s.N * s.K;
  const bool w8 = s.weight_type == WeightType::Int8 || s.weight_type == WeightType::Float8E4M3;
  const uint64_t weight = s.weight_type == WeightType::Uint4 ? elements / 2
                          : w8                             ? elements
                                                           : elements * 2;
  const uint64_t qblock =
      s.weight_type == WeightType::Uint4 ? uint64_t((s.N + 63) / 64) * s.TP * 512 : 0;
  const uint64_t scale = s.weight_type == WeightType::Uint4 ? uint64_t((s.K / s.Q + 3) / 4) * qblock
                         : w8
                             ? gil_align(uint64_t(s.N) * (s.col_parallel ? s.TP : 1) * 2, 256)
                             : 0;
  TORCH_CHECK(weight % 256 == 0 && weight / 256 <= UINT32_MAX && qblock <= UINT32_MAX &&
                  scale % 256 == 0 && scale / 256 <= UINT32_MAX,
              "grouped indexed linear: DDR expert stride exceeds ABI");
  return {weight, qblock, scale};
}
Selection canonical(const Shape &s) {
  validate_shape(s);
  const int k = s.indexed || s.K > 64 ? 256 : 64;
  const int n = s.indexed ? 32 : s.K <= 64 ? 128 : 64;
  const int m = !s.indexed && k == 256 && s.weight_type != WeightType::Float16 ? 224 : 256;
  const int partitions = s.indexed && (s.N + n - 1) / n < 8 ? 8 : 1;
  return {m, n, k, partitions};
}
const Tile &selected_tile(const Shape &s, const Selection &selection) {
  validate_shape(s);
  TORCH_CHECK(selection.expert_partitions == 1 || (s.indexed && selection.expert_partitions == 8),
              "grouped indexed linear: indexed allows P1/P8; scatter requires P1");
  // The W8 program decodes INT8/E4M3 using param27. This payload alias never
  // changes the typed DDR owner or the weight-dtype register supplied below.
  const auto payload_type = s.weight_type == WeightType::Float8E4M3 ? WeightType::Int8
                                                                  : s.weight_type;
  for (const auto &tile : kGilTiles) {
    if (tile.indexed == s.indexed && tile.weight_type == payload_type && tile.m == selection.m &&
        tile.n == selection.n && tile.k == selection.k) {
      TORCH_CHECK((!s.Q || tile.k % s.Q == 0) && (s.N + tile.n - 1) / tile.n <= UINT16_MAX,
                  "grouped indexed linear: tile/group or N grid exceeds ABI");
      return tile;
    }
  }
  TORCH_CHECK(false, "grouped indexed linear: forced M/N/K is not in the compiled tile catalog");
}
SelectionResult pick(const Shape &s, const Measurement *measurements,
                     size_t count, TileAvailable available) {
  const auto fallback = canonical(s);
  TORCH_CHECK(measurements != nullptr || count == 0,
              "grouped indexed linear: missing measured table backing");
  if (count == 0) return {fallback, SelectionSource::Canonical};
  // E4M3 provisionally borrows the INT8 tuning policy for the same byte-weight
  // program. These costs were measured with INT8, not with E4M3. Preserve the
  // actual Shape dtype for owner validation and launch register 27.
  const auto policy_type = s.weight_type == WeightType::Float8E4M3 ? WeightType::Int8
                                                                 : s.weight_type;
  const std::array<int64_t, 7> query{s.T, s.R, s.E, s.N, s.K, s.TP, s.Q};
  struct Row {
    const Measurement *measurement;
    std::vector<const RankedTile *> ranked;
  };
  std::vector<Row> rows;
  for (size_t index = 0; index < count; ++index) {
    const auto &entry = measurements[index];
    if (entry.indexed != s.indexed || entry.weight_type != policy_type ||
        entry.shape[6] != s.Q || entry.k_tile != fallback.k)
      continue;
    Row row{&entry, {}};
    for (size_t rank = 0; rank < entry.ranked_count; ++rank) {
      const auto &candidate = entry.ranked[rank];
      for (const auto &tile : kGilTiles)
        if (tile.indexed == s.indexed && tile.weight_type == policy_type &&
            tile.k == fallback.k && tile.m == candidate.m && tile.n == candidate.n &&
            tile.k == candidate.k && (!available || available(tile.id))) {
          row.ranked.push_back(&candidate);
          break;
        }
    }
    if (!row.ranked.empty()) rows.push_back(std::move(row));
  }
  const auto result = [&](const RankedTile &tile, SelectionSource source) {
    const int partitions = s.indexed && (s.N + tile.n - 1) / tile.n < 8 ? 8 : 1;
    return SelectionResult{{tile.m, tile.n, tile.k, partitions}, source};
  };
  for (const auto &row : rows) {
    if (row.measurement->shape != query) continue;
    const auto best = std::min_element(row.ranked.begin(), row.ranked.end(),
        [](const RankedTile *a, const RankedTile *b) {
          return std::make_tuple(a->cost, -a->m, a->n, a->k) <
                 std::make_tuple(b->cost, -b->m, b->n, b->k);
        });
    return result(**best, SelectionSource::Exact);
  }
  const double weights[]{0.5, 1, 1, 4, 2, 0.5};
  std::array<double, 6> features;
  for (size_t axis = 0; axis < features.size(); ++axis)
    features[axis] = std::log2(double(query[axis]));
  const Row *nearest = nullptr;
  std::tuple<bool, double, std::array<int64_t, 7>> nearest_key;
  for (const auto &row : rows) {
    const auto &shape = row.measurement->shape;
    if (shape[5] != s.TP) continue;
    double distance = 0;
    bool bounded = true;
    for (size_t axis = 0; axis < features.size(); ++axis) {
      const double delta = features[axis] - std::log2(double(shape[axis]));
      if (std::abs(delta) > 1) { bounded = false; break; }
      distance += weights[axis] * delta * delta;
    }
    if (!bounded) continue;
    const auto key = std::make_tuple(shape[3] != s.N || shape[4] != s.K, distance, shape);
    if (!nearest || key < nearest_key) { nearest = &row; nearest_key = key; }
  }
  if (!nearest) return {fallback, SelectionSource::Canonical};
  const auto &reference = nearest->measurement->shape;
  const RankedTile *winner = nullptr;
  std::tuple<double, int, int, int> winner_key;
  for (const auto *tile : nearest->ranked) {
    const double mp = std::max(1.0, std::ceil(double(s.R) / s.E / tile->m)) /
        std::max(1.0, std::ceil(double(reference[1]) / reference[2] / tile->m));
    const double np = std::ceil(double(s.N) / tile->n) /
        std::ceil(double(reference[3]) / tile->n);
    const double kp = std::ceil(double(s.K) / tile->k) /
        std::ceil(double(reference[4]) / tile->k);
    const double score = tile->cost * std::pow(mp, 0.5) * std::pow(np * kp, 0.25);
    const auto key = std::make_tuple(score, -tile->m, tile->n, tile->k);
    if (!winner || key < winner_key) { winner = tile; winner_key = key; }
  }
  return result(*winner, SelectionSource::Neighbor);
}
SelectionResult pick_current_auto(const Shape &s, TileAvailable available) {
  validate_shape(s);
  namespace pl = rpu_pl_tiling;
  const auto payload = s.weight_type == WeightType::Float8E4M3 ? WeightType::Int8 : s.weight_type;
  const auto precision = payload == WeightType::Uint4 ? pl::WeightType::Uint4
                         : payload == WeightType::Int8 ? pl::WeightType::Int8
                                                       : pl::WeightType::Fp16;
  std::vector<const Tile *> choices;
  for (const auto &tile : kGilTiles) {
    if (tile.indexed != s.indexed || tile.weight_type != payload ||
        tile.n < 32 || tile.n > 128 || tile.n % (s.Q ? 32 : 16) ||
        pl::k_index(tile.k) < 0 || (!s.Q && tile.k == 96) || (s.Q && tile.k % s.Q))
      continue;
    // Current fused layout adds persistent IDs (indexed) or two scratch rows
    // (scatter) to the ordinary grouped 400-row budget. Only maximal-M auto
    // candidates participate; retained explicit schedules remain separate.
    const auto ref = pl::make_tile(precision, false, tile.n, tile.k, pl::Family::Grouped, 16);
    const int coefficient = (tile.k + tile.n) / 16;
    const int fixed = ref.bias_offset - coefficient;
    const int reserve = tile.n / 16 + 1 + (s.indexed ? 0 : 2);
    const int maximum = (400 - fixed - reserve) / (coefficient + (s.indexed ? 1 : 0)) * 16;
    if (tile.m == maximum && (!available || available(tile.id))) choices.push_back(&tile);
  }
  TORCH_CHECK(!choices.empty(), "grouped indexed linear: no installed current-auto ACC16 tile");
  const auto find = [&](int n, int k) -> const Tile * {
    for (const auto *tile : choices)
      if (tile->n == n && tile->k == k) return tile;
    return nullptr;
  };
  const auto result = [&](const Tile &tile, SelectionSource source) {
    const int partitions = s.indexed && (s.N + tile.n - 1) / tile.n < 8 ? 8 : 1;
    return SelectionResult{{tile.m, tile.n, tile.k, partitions}, source};
  };
  // The exporter verifies this is the exact ranking snapshot copied into the
  // upstream fused module. Reuse data only; ordinary capacity/admission differs.
  const auto &table = pl::detail::measured_table(precision, false);
  for (size_t i = 0; i < table.count; ++i) {
    const auto &entry = table.shapes[i];
    if (entry.m != 1 || entry.n != s.N || entry.k != s.K) continue;
    for (unsigned j = 0; j < entry.ranked_count; ++j) {
      const auto &rank = table.ranked[entry.ranked_offset + j];
      if (const auto *tile = find(rank.n, rank.k)) return result(*tile, SelectionSource::PlExact);
    }
  }
  const std::array<double, 3> feature{std::log2(16.0), std::log2(double(s.N)), std::log2(double(s.K))};
  bool within = true, identical_nk = false;
  for (unsigned axis = 0; axis < 3; ++axis)
    within = within && feature[axis] >= table.lower[axis] - 1 && feature[axis] <= table.upper[axis] + 1;
  for (size_t i = 0; i < table.count; ++i)
    identical_nk = identical_nk || (table.shapes[i].n == s.N && table.shapes[i].k == s.K);
  const pl::detail::MeasuredShape *reference = nullptr;
  double distance = std::numeric_limits<double>::infinity();
  if (within) for (size_t i = 0; i < table.count; ++i) {
    const auto &entry = table.shapes[i];
    if (identical_nk && (entry.n != s.N || entry.k != s.K)) continue;
    bool legal = false;
    for (unsigned j = 0; j < entry.ranked_count && !legal; ++j) {
      const auto &rank = table.ranked[entry.ranked_offset + j];
      legal = find(rank.n, rank.k) != nullptr;
    }
    if (!legal) continue;
    const double dm = feature[0] - entry.feature[0], dn = 4 * (feature[1] - entry.feature[1]);
    const double dk = feature[2] - entry.feature[2], candidate = dm * dm + dn * dn + dk * dk;
    // Exported rows are shape-sorted: strict < preserves upstream's shape tie.
    if (candidate < distance) { distance = candidate; reference = &entry; }
  }
  if (reference) {
    const Tile *winner = nullptr;
    auto best = std::make_tuple(std::numeric_limits<double>::infinity(), 0, 0);
    for (unsigned j = 0; j < reference->ranked_count; ++j) {
      const auto &rank = table.ranked[reference->ranked_offset + j];
      const auto *tile = find(rank.n, rank.k);
      if (!tile) continue;
      const int m = pl::make_tile(precision, false, tile->n, tile->k).m_tile;
      const double cm = 1.0 / pl::ceil_div_positive(reference->m, m);
      const double cn = double((s.N + tile->n - 1) / tile->n) / pl::ceil_div_positive(reference->n, tile->n);
      const double ck = double((s.K + tile->k - 1) / tile->k) / pl::ceil_div_positive(reference->k, tile->k);
      const auto score = std::make_tuple(rank.cost * std::pow(cm, 0.5) * std::pow(cn * ck, 0.25), tile->n, tile->k);
      if (score < best) { best = score; winner = tile; }
    }
    if (winner) return result(*winner, SelectionSource::PlNeighbor);
  }
  if (const auto *tile = find(64, 128)) return result(*tile, SelectionSource::PlCanonical);
  const auto best = std::min_element(choices.begin(), choices.end(), [](const Tile *a, const Tile *b) {
    return std::make_tuple(std::abs(a->k - 128), std::abs(a->n - 64), a->n, a->k) <
           std::make_tuple(std::abs(b->k - 128), std::abs(b->n - 64), b->n, b->k);
  });
  return result(**best, SelectionSource::PlCanonical);
}
Selection resolve_current_auto(const Shape &s) {
  return pick_current_auto(s, [](KernelId id) {
    return KernelCache::instance().has_loaded(id);
  }).selection;
}
Selection resolve(const Shape &s) {
  const auto &tile = selected_tile(s, canonical(s));
  TORCH_CHECK(KernelCache::instance().has_loaded(tile.id),
              "grouped indexed linear: canonical payload unavailable: ", tile.symbol);
  return pick(s, kGilMeasurements, kGilMeasurementCount, [](KernelId id) {
    return KernelCache::instance().has_loaded(id);
  }).selection;
}
void validate_routes(const Shape &s, const int16_t *counts, const uint16_t *ids) {
  validate_shape(s);
  TORCH_CHECK(counts != nullptr && ids != nullptr, "grouped indexed linear: null routes");
  int64_t sum = 0;
  for (int64_t e = 0; e < s.E; ++e) {
    TORCH_CHECK(counts[e] >= 0, "grouped indexed linear: negative expert count");
    sum += counts[e];
  }
  TORCH_CHECK(sum == s.R, "grouped indexed linear: sum(counts) must equal R");
  std::vector<int64_t> last_expert(s.indexed ? 0 : s.T, -1);
  int64_t row = 0;
  for (int64_t e = 0; e < s.E; ++e)
    for (int64_t j = 0; j < counts[e]; ++j, ++row) {
      const auto id = ids[row];
      TORCH_CHECK(id < s.T, "grouped indexed linear: ID outside resident T");
      if (!s.indexed) {
        TORCH_CHECK(last_expert[id] != e,
                    "grouped indexed linear: duplicate ID within scatter expert");
        last_expert[id] = e;
      }
    }
}
} // namespace rpu_grouped_indexed_linear

void rpu_launch_grouped_indexed_linear_selected(const gil::Shape &s,
                                                const gil::Selection &selection, uint32_t input,
                                                const at::Tensor &weight, uint32_t counts,
                                                const c10::optional<at::Tensor> &scale,
                                                const c10::optional<at::Tensor> &zero, uint32_t ids,
                                                uint32_t coefficients, uint32_t output) {
  const auto &tile = gil::selected_tile(s, selection);
  const auto l = gil::layout(s);
  gil_check_owners(s, weight, scale, zero);
  gil_check_spans(s, input, counts, ids, coefficients, output);
  const auto role = s.weight_type == gil::WeightType::Uint4
                        ? GraphDdrRegisterRole::PackedUint4ModelWeight
                    : s.weight_type == gil::WeightType::Int8 ? GraphDdrRegisterRole::Int8ModelWeight
                    : s.weight_type == gil::WeightType::Float8E4M3
                        ? GraphDdrRegisterRole::Fp8ModelWeight
                        : GraphDdrRegisterRole::ModelWeight;
  std::vector<GraphDdrRegisterOperandSpec> operands{
      {{role, GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
       weight}};
  if (scale.has_value())
    operands.push_back(
        {{s.weight_type == gil::WeightType::Uint4 ? GraphDdrRegisterRole::PerGroupScale
                                                  : GraphDdrRegisterRole::PerChannelScale,
          GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21},
         *scale});
  if (zero.has_value())
    operands.push_back({{GraphDdrRegisterRole::NegativeZeroPoint, GraphDdrRegisterAccess::Read,
                         GraphDdrRegisterEncoding::DevAddrShift8LoHi, 22, 23},
                        *zero});
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(tile.id, operands);
  // Typed census must own the DDR operands before graph-aware lookup.
  auto *kernel = GET_KERNEL(tile.id);
  TORCH_CHECK(kernel != nullptr,
              "grouped indexed linear: selected payload unavailable: ", tile.symbol);
  for (const auto &operand : operands)
    rpu_ddr_flush(operand.tensor.data_ptr());
  kernel->reset_regs();
  gil_set_u32(kernel, 0, input);
  gil_set_u32(kernel, 4, output);
  gil_set_u32(kernel, 6, 0); // No bias.
  gil_set_u32(kernel, 10, static_cast<uint32_t>(s.N));
  gil_set_u32(kernel, 16, static_cast<uint32_t>(2 * s.K));
  gil_set_u32(kernel, 18, static_cast<uint32_t>(2 * s.N));
  gil_set_u32(kernel, 24, static_cast<uint32_t>(l.scale_qblock_bytes));
  gil_set_u32(kernel, 32, counts);
  gil_set_u32(kernel, 34, static_cast<uint32_t>(l.weight_expert_bytes / 256));
  gil_set_u32(kernel, 36, static_cast<uint32_t>(l.scale_expert_bytes / 256));
  gil_set_u32(kernel, 38, static_cast<uint32_t>(zero.has_value() ? l.scale_expert_bytes / 256 : 0));
  gil_set_u32(kernel, 40, ids);
  gil_set_u32(kernel, 42, s.indexed ? 0 : coefficients);
  for (const auto &parameter :
       std::initializer_list<std::pair<int, int64_t>>{{8, s.col_parallel},
                                                      {9, s.R},
                                                      {12, s.K},
                                                      {13, s.K / 16},
                                                      {14, 0},
                                                      {15, s.TP},
                                                      {26, 2},
                                                      {27, int(s.weight_type)},
                                                      {28, s.Q},
                                                      {29, s.Q ? s.K / s.Q : 0},
                                                      {30, s.E},
                                                      {44, s.T},
                                                      {45, s.R},
                                                      {46, selection.expert_partitions}})
    kernel->set_regs(parameter.first, static_cast<uint16_t>(parameter.second));
  const auto grid_x = static_cast<uint16_t>((s.N + tile.n - 1) / tile.n);
  const auto grid_y = static_cast<uint16_t>(selection.expert_partitions);
  kernel->set_regs(64, grid_x);
  kernel->set_regs(65, grid_y);
  kernel->set_regs(66, static_cast<uint16_t>(1));
  // Commit after every register, including grid dimensions: the typed
  // Graph writer snapshots the complete Kernel_t register-state token.
  writer.write(*kernel);
  auto *queue = GET_QUEUE(s.TP);
  queue->set_broadcast_mode(true);
  std::vector<uint8_t> cores(s.TP);
  for (int i = 0; i < s.TP; ++i)
    cores[i] = i;
  queue->enqueu_kernel(*kernel, {grid_x, grid_y, 1}, cores);
}
