// Immediate conformance entry points; production library contains no test registration.
#include "rpu_ops.h"
#include "rpu_grouped_indexed_linear.h"
#include "rpu_spm_allocator.h"
#include <torch/library.h>
#include <climits>
#include <tuple>

namespace gil = rpu_grouped_indexed_linear;

namespace {
uint64_t gil_align(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}
void gil_test_tensor(const at::Tensor &tensor, const at::Tensor &reference, at::ScalarType dtype,
                     at::IntArrayRef sizes, const char *name) {
  TORCH_CHECK(tensor.defined() && tensor.device().type() == at::kPrivateUse1 &&
                  tensor.device() == reference.device() && tensor.scalar_type() == dtype &&
                  tensor.is_contiguous() && tensor.sizes() == sizes,
              "grouped indexed conformance: invalid ", name, " shape/dtype/device");
}
at::Tensor gil_test_aligned_owner(const at::Tensor &tensor, uint64_t alignment) {
  const auto address = rhino_lkn::RpuGetDevAddr(tensor.data_ptr());
  TORCH_CHECK(address != 0, "grouped indexed conformance: missing DDR address");
  if (address % alignment == 0)
    return tensor;
  auto backing =
      at::empty({tensor.numel() + int64_t(alignment / tensor.element_size())}, tensor.options());
  const auto base = rhino_lkn::RpuGetDevAddr(backing.data_ptr());
  const auto offset = (alignment - base % alignment) % alignment;
  TORCH_CHECK(base != 0 && offset % tensor.element_size() == 0,
              "grouped indexed conformance: cannot align typed owner");
  auto owner =
      backing.narrow(0, offset / tensor.element_size(), tensor.numel()).view(tensor.sizes());
  owner.copy_(tensor);
  TORCH_CHECK(rhino_lkn::RpuGetDevAddr(owner.data_ptr()) % alignment == 0,
              "grouped indexed conformance: aligned address mismatch");
  return owner;
}
uint32_t gil_test_allocate(uint64_t bytes) {
  return SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(static_cast<size_t>(bytes)));
}
void gil_test_broadcast(const at::Tensor &tensor, uint32_t address, int cores) {
  const int64_t words = gil_align(tensor.numel(), 16);
  auto padded = at::zeros({words}, tensor.options());
  padded.narrow(0, 0, tensor.numel()).copy_(tensor.view({-1}));
  rpu_launch_ddr_broadcast_spm_dma_immediate(reinterpret_cast<c10::Half *>(padded.data_ptr()),
                                             words, address, cores);
}
void gil_test_scatter(const at::Tensor &tensor, uint32_t address, int cores) {
  const int64_t words = tensor.numel() / cores;
  TORCH_CHECK(tensor.numel() % cores == 0 && words % 8 == 0,
              "grouped indexed conformance: DMA requires 16-byte per-core payloads");
  rpu_launch_ddr_scatter_spm_dma_immediate(reinterpret_cast<c10::Half *>(tensor.data_ptr()), words,
                                           words * 2, address, cores);
}
// Exact one-kernel test policy. The scoped SPM owners below outlive BUILD
// and both host-reemitted REPLAYs; no FMB/production policy is relaxed.
GraphKernelRegisterPolicy gil_test_census_policy(const gil::Tile& tile) {
  GraphKernelRegisterRule rule;
  rule.kernel_id = tile.id;
  rule.kind = GraphKernelRegisterRuleKind::TypedDdr;
  rule.expected_count = 1;
  for (const auto& operand : std::initializer_list<std::pair<GraphDdrRegisterRole, uint16_t>>{
           {GraphDdrRegisterRole::PackedUint4ModelWeight, 2},
           {GraphDdrRegisterRole::PerGroupScale, 20},
           {GraphDdrRegisterRole::NegativeZeroPoint, 22}})
    rule.operands.push_back({operand.first, GraphDdrRegisterAccess::Read,
        GraphDdrRegisterEncoding::DevAddrShift8LoHi, operand.second,
        static_cast<uint16_t>(operand.second + 1)});
  constexpr uint64_t offset = UINT64_C(14695981039346656037);
  const auto byte = [](uint64_t h, uint8_t b) { return (h ^ b) * UINT64_C(1099511628211); };
  const std::string symbol(tile.symbol);
  uint64_t kernel_hash = offset;
  for (unsigned shift = 0; shift < 64; shift += 8)
    kernel_hash = byte(kernel_hash, static_cast<uint8_t>(uint64_t(symbol.size()) >> shift));
  for (unsigned char c : symbol) kernel_hash = byte(kernel_hash, c);
  GraphKernelRegisterCensusStats stats;
  stats.node_count = stats.kernel_count = stats.typed_kernel_count = 1;
  stats.operand_count = stats.weight_operand_count = 3;
  stats.ordered_node_kind_hash = byte(offset, static_cast<uint8_t>(GraphNodeKind::Kernel));
  stats.ordered_kernel_hash = kernel_hash;
  GraphKernelRegisterPolicy policy;
  policy.fingerprint = UINT64_C(0x47494c43454e5331); // GILCENS1, test-only exact ABI.
  policy.name = "conformance_grouped_uint4_single_kernel_v1";
  policy.rules.push_back(std::move(rule));
  policy.expected_node_count = policy.expected_kernel_count = policy.expected_typed_kernel_count = 1;
  policy.expected_operand_count = policy.expected_weight_operand_count = 3;
  policy.expected_ordered_node_kind_hash = stats.ordered_node_kind_hash;
  policy.expected_ordered_kernel_hash = stats.ordered_kernel_hash;
  policy.phases.push_back({"projection", stats});
  return policy;
}
at::Tensor gil_test_projection(const gil::Shape &s, const at::Tensor &input,
                               const at::Tensor &weight, const at::Tensor &counts,
                               const c10::optional<at::Tensor> &scale,
                               const c10::optional<at::Tensor> &zero, const at::Tensor &ids,
                               const c10::optional<at::Tensor> &coeff,
                               const c10::optional<at::Tensor> &initial, int64_t m, int64_t n,
                               int64_t k, int64_t partitions,
                               std::vector<int64_t>* census_witness = nullptr) {
  TORCH_CHECK(RpuKernelGraph::active().state() == RpuKernelGraph::State::PASSTHROUGH,
              "grouped indexed conformance: temporary SPM requires PASSTHROUGH");
  gil::validate_shape(s);
  TORCH_CHECK(weight.device() == input.device(),
              "grouped indexed conformance: input/weight device mismatch");
  gil_test_tensor(input, weight, at::kHalf, {s.TP, s.indexed ? s.T : s.R, s.K}, "X");
  gil_test_tensor(counts, weight, at::kShort, {s.E}, "counts");
  gil_test_tensor(ids, weight, at::kShort, {s.R}, "IDs");
  const auto cpu_counts = counts.cpu();
  const auto cpu_ids = ids.cpu();
  gil::validate_routes(s, cpu_counts.data_ptr<int16_t>(),
                       reinterpret_cast<const uint16_t *>(cpu_ids.data_ptr<int16_t>()));
  TORCH_CHECK(at::isfinite(input.cpu()).all().item<bool>(),
              "grouped indexed conformance: input must be finite");
  if (!s.indexed) {
    TORCH_CHECK(coeff.has_value() && initial.has_value(),
                "grouped indexed conformance: missing scatter operands");
    gil_test_tensor(*coeff, weight, at::kHalf, {s.R}, "coefficients");
    gil_test_tensor(*initial, weight, at::kHalf, {s.TP, s.T, s.N}, "initial partial");
    TORCH_CHECK(at::isfinite(coeff->cpu()).all().item<bool>() &&
                    at::isfinite(initial->cpu()).all().item<bool>(),
                "grouped indexed conformance: scatter operands must be finite");
  }
  auto selection = gil::canonical(s);
  TORCH_CHECK((m == 0 && n == 0 && k == 0) ||
                  (m > 0 && m <= INT_MAX && n > 0 && n <= INT_MAX && k > 0 && k <= INT_MAX),
              "grouped indexed conformance: specify complete positive tile M/N/K or all zero");
  if (m)
    selection = {int(m), int(n), int(k), s.indexed && (s.N + n - 1) / n < 8 ? 8 : 1};
  else
    selection = gil::resolve(s);
  TORCH_CHECK(partitions == 0 || partitions == 1 || (s.indexed && partitions == 8),
              "grouped indexed conformance: invalid expert partitions");
  if (partitions)
    selection.expert_partitions = int(partitions);
  const auto &tile = gil::selected_tile(s, selection);
  TORCH_CHECK(KernelCache::instance().has_loaded(tile.id),
              "grouped indexed conformance: selected payload is unavailable");
  const auto w = gil_test_aligned_owner(weight, 256);
  c10::optional<at::Tensor> sc, zp;
  if (scale)
    sc = gil_test_aligned_owner(*scale, s.weight_type == gil::WeightType::Uint4 ? s.TP * 512 : 256);
  if (zero)
    zp = gil_test_aligned_owner(*zero, s.TP * 512);
  if (!SPM_ALLOC.is_initialized())
    SPM_ALLOC.init();
  SpmAllocator::ScopedTemporary temporary;
  const auto x = gil_test_allocate(uint64_t(s.indexed ? s.T : s.R) * s.K * 2);
  const auto y = gil_test_allocate(uint64_t(s.indexed ? s.R : s.T) * s.N * 2);
  const auto c = gil_test_allocate(gil_align(s.E, 16) * 2);
  const auto i = gil_test_allocate(gil_align(s.R, 16) * 2);
  const auto r = s.indexed ? uint32_t(0) : gil_test_allocate(gil_align(s.R, 16) * 2);
  gil_test_scatter(input, x, s.TP);
  gil_test_broadcast(counts, c, s.TP);
  gil_test_broadcast(ids, i, s.TP);
  if (!s.indexed) {
    gil_test_broadcast(*coeff, r, s.TP);
    gil_test_scatter(*initial, y, s.TP);
  }
  const int64_t output_rows = s.indexed ? s.R : s.T;
  const auto read_output = [&]() {
    auto result = at::empty({s.TP, output_rows, s.N}, input.options());
    rpu_launch_spm_scatter_ddr_dma_immediate(y, reinterpret_cast<c10::Half *>(result.data_ptr()),
                                             output_rows * s.N, output_rows * s.N * 2, s.TP);
    return result;
  };
  const auto launch = [&]() {
    rpu_launch_grouped_indexed_linear_selected(s, selection, x, w, c, sc, zp, i, r, y);
  };
  if (census_witness == nullptr) {
    launch();
    return read_output();
  }
  TORCH_CHECK(s.weight_type == gil::WeightType::Uint4 && s.TP == 8,
              "grouped indexed census conformance requires UINT4 TP8");
  GraphRuntimePolicy runtime_policy;
  runtime_policy.execution_core_count = s.TP;
  RpuKernelGraph graph(runtime_policy);
  GraphSignature signature;
  signature.op_id = INT64_C(0x47494c43454e5331);
  signature.shapes = {s.indexed, s.T, s.R, s.E, s.N, s.K, s.Q, s.TP,
                      selection.m, selection.n, selection.k, selection.expert_partitions};
  const auto policy = gil_test_census_policy(tile);
  std::vector<at::Tensor> outputs;
  uint64_t built_generation = 0;
  for (int repeat = 0; repeat < 3; ++repeat) {
    // Refresh scatter's additive input outside the Graph scope, retaining the
    // exact same complete SPM intervals across all three executions.
    if (!s.indexed && repeat != 0) gil_test_scatter(*initial, y, s.TP);
    graph.begin(signature);
    try {
      const bool replay = repeat != 0;
      TORCH_CHECK(graph.state() == (replay ? RpuKernelGraph::State::REPLAYING
                                           : RpuKernelGraph::State::RECORDING),
                  "grouped indexed census did not retain its recorded Graph");
      // This test owns one scoped SPM plan and explicit invocation epochs;
      // it does not mint or impersonate a production FMB physical lease.
      GraphKernelRegisterCensusGuard census(graph, policy, policy.fingerprint, repeat + 1);
      launch();
      const auto observed = census.stats();
      census.complete();
      census_witness->insert(census_witness->end(), {
          replay, int64_t(observed.node_count), int64_t(observed.kernel_count),
          int64_t(observed.typed_kernel_count), int64_t(observed.operand_count),
          int64_t(observed.weight_operand_count)});
      graph.end();
    } catch (...) {
      graph.abort();
      throw;
    }
    TORCH_CHECK(graph.state() == RpuKernelGraph::State::BUILT,
                "grouped indexed census Graph did not complete");
    const auto stats = graph.debug_stats();
    if (repeat == 0) built_generation = stats.build_generation;
    TORCH_CHECK(stats.register_census_present && stats.register_census_build_committed &&
                    stats.kernel_count == 1 && stats.segment_count == 1 &&
                    stats.hw_batch_submit_total == size_t(repeat + 1) &&
                    stats.build_generation == built_generation,
                "grouped indexed census lost its exact retained device execution");
    census_witness->insert(census_witness->end(), {
        int64_t(stats.hw_batch_submit_total), int64_t(stats.build_generation),
        int64_t(stats.execution_ordinal)});
    outputs.push_back(read_output().cpu());
  }
  graph.invalidate();
  return at::stack(outputs);
}
gil::WeightType gil_test_weight_type(int64_t mode) {
  TORCH_CHECK(mode == 1 || mode == 2 || mode == 5 || mode == 8,
              "grouped indexed conformance: INT8/FP16/E4M3/UINT4 mode is 1/2/5/8");
  return static_cast<gil::WeightType>(mode);
}
} // namespace

std::tuple<std::vector<int64_t>, std::string> rpu_grouped_indexed_linear_current_tile_test(
    bool indexed, int64_t tokens, int64_t rows, int64_t experts, int64_t local_n,
    int64_t local_k, int64_t group_size, int64_t cores, int64_t mode, bool col) {
  const gil::Shape shape{indexed, tokens, rows, experts, local_n, local_k, group_size,
                         cores, gil_test_weight_type(mode), col};
  const auto selection = gil::resolve_current_auto(shape);
  const auto &tile = gil::selected_tile(shape, selection);
  return {{selection.m, selection.n, selection.k, selection.expert_partitions}, tile.symbol};
}

at::Tensor rpu_grouped_parallel_linear_indexed_test(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &counts,
    const c10::optional<at::Tensor> &scale, const c10::optional<at::Tensor> &zero,
    const at::Tensor &ids, int64_t local_n, int64_t mode, int64_t group_size, bool col, int64_t m,
    int64_t n, int64_t k, int64_t partitions) {
  TORCH_CHECK(input.dim() == 3 && counts.dim() == 1 && ids.dim() == 1,
              "grouped indexed conformance: expected X[TP,T,K], counts[E], IDs[R]");
  const gil::Shape s{true,
                     input.size(1),
                     ids.numel(),
                     counts.numel(),
                     local_n,
                     input.size(2),
                     group_size,
                     input.size(0),
                     gil_test_weight_type(mode),
                     col};
  return gil_test_projection(s, input, weight, counts, scale, zero, ids, c10::nullopt, c10::nullopt,
                             m, n, k, partitions);
}
at::Tensor rpu_grouped_parallel_linear_weighted_scatter_add_test(
    const at::Tensor &act, const at::Tensor &weight, const at::Tensor &counts,
    const c10::optional<at::Tensor> &scale, const c10::optional<at::Tensor> &zero,
    const at::Tensor &ids, const at::Tensor &coeff, const at::Tensor &initial, int64_t mode,
    int64_t group_size, bool col, int64_t m, int64_t n, int64_t k) {
  TORCH_CHECK(
      act.dim() == 3 && initial.dim() == 3 && counts.dim() == 1 && ids.dim() == 1,
      "grouped scatter conformance: expected Act[TP,R,K], initial[TP,T,N], counts[E], IDs[R]");
  const gil::Shape s{false,
                     initial.size(1),
                     ids.numel(),
                     counts.numel(),
                     initial.size(2),
                     act.size(2),
                     group_size,
                     act.size(0),
                     gil_test_weight_type(mode),
                     col};
  return gil_test_projection(s, act, weight, counts, scale, zero, ids, coeff, initial, m, n, k, 1);
}

std::tuple<at::Tensor, std::vector<int64_t>> rpu_grouped_indexed_linear_census_test(
    bool indexed, const at::Tensor& input, const at::Tensor& weight,
    const at::Tensor& counts, const at::Tensor& scale, const at::Tensor& zero,
    const at::Tensor& ids, const c10::optional<at::Tensor>& coefficients,
    const c10::optional<at::Tensor>& initial, int64_t local_n, int64_t group_size,
    bool col, int64_t m, int64_t n, int64_t k, int64_t partitions) {
  TORCH_CHECK(input.dim() == 3 && counts.dim() == 1 && ids.dim() == 1 &&
                  (indexed || (initial && initial->dim() == 3)),
              "grouped indexed census: malformed fixture");
  const gil::Shape shape{indexed, indexed ? input.size(1) : initial->size(1),
      ids.numel(), counts.numel(), local_n, input.size(2), group_size,
      input.size(0), gil::WeightType::Uint4, col};
  std::vector<int64_t> witness;
  auto outputs = gil_test_projection(shape, input, weight, counts, scale, zero,
      ids, coefficients, initial, m, n, k, partitions, &witness);
  return {outputs, witness};
}

TORCH_LIBRARY_FRAGMENT(rpu, m) {
  m.def("grouped_indexed_linear_current_tile_test(bool indexed, int tokens, int rows, int experts, int local_n, int local_k, int group_size, int cores, int weight_dtype_mode, bool is_col_parallel) -> (int[], str)",
        TORCH_FN(rpu_grouped_indexed_linear_current_tile_test));
  m.def("grouped_indexed_linear_census_test(bool indexed, Tensor input, Tensor weight, Tensor counts, Tensor scale, Tensor negative_zero, Tensor ids, Tensor? coefficients, Tensor? initial, int local_n, int group_size, bool is_col_parallel, int tile_m, int tile_n, int tile_k, int expert_partitions) -> (Tensor, int[])",
        TORCH_FN(rpu_grouped_indexed_linear_census_test));
  m.def("grouped_parallel_linear_indexed_test(Tensor input_per_core, Tensor weight, Tensor counts, Tensor? scale, Tensor? negative_zero, Tensor ids, int local_n, int weight_dtype_mode, int group_size, bool is_col_parallel, int tile_m=0, int tile_n=0, int tile_k=0, int expert_partitions=0) -> Tensor",
        TORCH_FN(rpu_grouped_parallel_linear_indexed_test));
  m.def("grouped_parallel_linear_weighted_scatter_add_test(Tensor act_per_core, Tensor weight, Tensor counts, Tensor? scale, Tensor? negative_zero, Tensor ids, Tensor coefficients, Tensor initial_partial_per_core, int weight_dtype_mode, int group_size, bool is_col_parallel, int tile_m=0, int tile_n=0, int tile_k=0) -> Tensor",
        TORCH_FN(rpu_grouped_parallel_linear_weighted_scatter_add_test));
}
