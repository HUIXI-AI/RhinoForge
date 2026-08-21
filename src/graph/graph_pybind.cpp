// Python bindings for Graph, GraphCache, and GraphSignature. Graph objects are
// shared between Python and their owning cache entries.

#include "graph/graph_pybind.h"

#include "graph/graph_infra.h"
#include "graph/graph_runtime.h"

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace graph_pybind {

void add_bindings(py::module_& m) {
    // GraphSignature admission-key fields.
    py::class_<GraphSignature>(m, "GraphSignature")
        .def(py::init<>())
        .def_readwrite("op_id",        &GraphSignature::op_id)
        .def_readwrite("op_id_str",    &GraphSignature::op_id_str)  // human-readable label
        .def_readwrite("shapes",       &GraphSignature::shapes)
        .def_readwrite("dyn_dims",     &GraphSignature::dyn_dims)
        .def_readwrite("dtypes",       &GraphSignature::dtypes)
        .def_readwrite("flags",        &GraphSignature::flags)
        .def_readwrite("branch_key",   &GraphSignature::branch_key)
        .def_readwrite("segment_key",  &GraphSignature::segment_key)
        .def("__eq__", [](const GraphSignature& a, const GraphSignature& b) {
            return a == b;
        })
        .def("__ne__", [](const GraphSignature& a, const GraphSignature& b) {
            return a != b;
        })
        .def("__hash__", [](const GraphSignature& s) {
            return static_cast<int64_t>(GraphSignatureHash{}(s)
                                        & 0x7fffffffffffffffULL);
        })
        .def("__repr__", [](const GraphSignature& s) {
            return s.to_string();
        });

    // Layered signatures identify which admission-key layer diverged.
    py::enum_<SignatureLayer>(m, "SignatureLayer")
        .value("GmId",         SignatureLayer::GmId)
        .value("TensorStruct", SignatureLayer::TensorStruct)
        .value("ShapeBucket",  SignatureLayer::ShapeBucket)
        .value("DtypeLayout",  SignatureLayer::DtypeLayout)
        .value("ScalarFlags",  SignatureLayer::ScalarFlags)
        .value("BranchKey",    SignatureLayer::BranchKey)
        .value("SegmentKey",   SignatureLayer::SegmentKey)
        .value("kCount",       SignatureLayer::kCount);

    py::class_<LayeredSignature>(m, "LayeredSignature")
        .def(py::init<>())
        .def("layer_hash",     &LayeredSignature::layer_hash,
             py::arg("layer"))
        .def("equals_up_to",   &LayeredSignature::equals_up_to,
             py::arg("other"), py::arg("up_to_inclusive"))
        .def("first_diverge",  &LayeredSignature::first_diverge,
             py::arg("other"))
        .def("with_layer_hash", &LayeredSignature::with_layer_hash,
             py::arg("layer"), py::arg("hash"))
        .def("to_string",      &LayeredSignature::to_string)
        .def("__repr__",       &LayeredSignature::to_string);

    m.def("signature_layer_name",
          [](SignatureLayer layer) {
              return std::string(signature_layer_name(layer));
          },
          py::arg("layer"));
    m.def("layered_from", &layered_from, py::arg("signature"));

    // Graph objects are shared with cache entries.
    py::class_<RpuKernelGraph, std::shared_ptr<RpuKernelGraph>>(m, "Graph")
        .def(py::init([]() { return make_registered_rpu_kernel_graph(); }))
        .def("begin",
             py::overload_cast<>(&RpuKernelGraph::begin),
             "Open scope without signature (always RECORDING)")
        .def("begin",
             py::overload_cast<const GraphSignature&>(&RpuKernelGraph::begin),
             py::arg("signature"),
             "Open scope with signature: BUILT+sig match → REPLAYING,"
             " else RECORDING with sig pending until end()")
        .def("end",        &RpuKernelGraph::end,
             "Close scope: RECORDING → BUILT (replayable) or PASSTHROUGH"
             " (non-replayable); REPLAYING → BUILT")
        .def("abort",      &RpuKernelGraph::abort,
             "Abort scope on exception, drop captured nodes")
        .def("invalidate", &RpuKernelGraph::invalidate,
             "Discard BUILT graph + prepared queues")
        .def("state", [](const RpuKernelGraph& g) -> int {
            return static_cast<int>(g.state());
        }, "0=PASSTHROUGH,1=RECORDING,2=BUILT,3=REPLAYING")
        .def("graph_size", [](const RpuKernelGraph& g) -> int64_t {
            return static_cast<int64_t>(g.graph_size());
        }, "Number of captured nodes")
        .def("replayable",          &RpuKernelGraph::replayable)
        .def("has_built_signature", &RpuKernelGraph::has_built_signature)
        .def("built_signature",     &RpuKernelGraph::built_signature,
             py::return_value_policy::copy)
        .def("debug_stats", [](const RpuKernelGraph& graph) {
            const GraphStats& stats = graph.debug_stats();
            py::dict out;
            out["kernel_count"] = stats.kernel_count;
            out["data_node_count"] = stats.data_node_count;
            out["segment_count"] = stats.segment_count;
            out["per_segment_replay_count"] = stats.per_segment_replay_count;
            out["boundary_flush_count"] = stats.boundary_flush_count;
            out["host_callback_tier1_count"] = stats.host_callback_tier1_count;
            out["host_callback_tier2_count"] = stats.host_callback_tier2_count;
            out["host_callback_tier3_count"] = stats.host_callback_tier3_count;
            out["host_callback_output_bytes"] = stats.host_callback_output_bytes;
            out["tier3_oneshot_count"] = stats.tier3_oneshot_count;
            out["prepared_segment_hit_total"] = stats.prepared_segment_hit_total;
            out["prepared_segment_miss_total"] = stats.prepared_segment_miss_total;
            out["hw_batch_submit_total"] = stats.hw_batch_submit_total;
            out["non_replayable_reason"] = stats.non_replayable_reason;
            return out;
        }, "Return aggregate counters for the most recent graph submission");

    // GraphCache co-owns returned Graph objects with Python.
    py::class_<RpuGraphCache>(m, "GraphCache")
        .def(py::init<>())
        .def(py::init<size_t>(), py::arg("max_entries"))
        .def("get_or_create",
             [](RpuGraphCache& self, const GraphSignature& sig) {
                 return self.lookup_shared(sig)
                     ? self.lookup_shared(sig)
                     : (self.get_or_create(sig), self.lookup_shared(sig));
             },
             py::arg("signature"),
             "Lookup or create a Graph for the signature; returned shared_ptr"
             " is co-owned with the cache")
        .def("lookup",
             &RpuGraphCache::lookup_shared,
             py::arg("signature"),
             "Return shared_ptr<Graph> if present, None otherwise")
        .def("size",
             [](const RpuGraphCache& self) -> int64_t {
                 return static_cast<int64_t>(self.size());
             })
        .def("max_entries",
             [](const RpuGraphCache& self) -> int64_t {
                 return static_cast<int64_t>(self.max_entries());
             })
        .def("evict", &RpuGraphCache::evict, py::arg("signature"))
        .def("clear", &RpuGraphCache::clear)
        // Update replay/recapture counters and LRU order.
        .def("touch_entry", &RpuGraphCache::touch_entry,
             py::arg("signature"), py::arg("replay") = false,
             "Mark an entry as recently accessed and update its counters")
        // Aggregate per-entry diagnostics used by the Python runtime.
        .def("snapshot",            &RpuGraphCache::snapshot)
        .def("cache_invariant_ok",  &RpuGraphCache::cache_invariant_ok)
        .def("debug_bucket_counts", &RpuGraphCache::debug_bucket_counts)
        .def("debug_branch_counts", &RpuGraphCache::debug_branch_counts)
        .def("dump_signature_tree", &RpuGraphCache::dump_signature_tree)
        .def("explain_miss", &RpuGraphCache::explain_miss,
             py::arg("signature"));

    py::class_<RpuGraphCache::Snapshot>(m, "GraphCacheEntrySnapshot")
        .def_readonly("signature",                  &RpuGraphCache::Snapshot::signature)
        .def_readonly("kernel_count",               &RpuGraphCache::Snapshot::kernel_count)
        .def_readonly("segment_count",              &RpuGraphCache::Snapshot::segment_count)
        .def_readonly("local_spm_slot_count",       &RpuGraphCache::Snapshot::local_spm_slot_count)
        .def_readonly("replay_count",               &RpuGraphCache::Snapshot::replay_count)
        .def_readonly("recapture_count",            &RpuGraphCache::Snapshot::recapture_count)
        .def_readonly("non_replayable_reason",      &RpuGraphCache::Snapshot::non_replayable_reason)
        .def_readonly("data_node_count",            &RpuGraphCache::Snapshot::data_node_count)
        .def_readonly("boundary_flush_count",       &RpuGraphCache::Snapshot::boundary_flush_count)
        .def_readonly("prepared_segment_hit_total", &RpuGraphCache::Snapshot::prepared_segment_hit_total)
        .def_readonly("prepared_segment_miss_total",&RpuGraphCache::Snapshot::prepared_segment_miss_total)
        .def_readonly("hw_batch_submit_total",      &RpuGraphCache::Snapshot::hw_batch_submit_total)
        .def_readonly("host_callback_tier1_count",  &RpuGraphCache::Snapshot::host_callback_tier1_count)
        .def_readonly("host_callback_tier2_count",  &RpuGraphCache::Snapshot::host_callback_tier2_count)
        .def_readonly("host_callback_tier3_count",  &RpuGraphCache::Snapshot::host_callback_tier3_count)
        .def_readonly("tier3_oneshot_count",        &RpuGraphCache::Snapshot::tier3_oneshot_count)
        .def_readonly("register_census_present", &RpuGraphCache::Snapshot::register_census_present)
        .def_readonly("register_census_build_committed", &RpuGraphCache::Snapshot::register_census_build_committed)
        .def_readonly("register_census_policy_fingerprint", &RpuGraphCache::Snapshot::register_census_policy_fingerprint)
        .def_readonly("register_census_policy_digest", &RpuGraphCache::Snapshot::register_census_policy_digest)
        .def_readonly("register_census_plan_hash", &RpuGraphCache::Snapshot::register_census_plan_hash)
        .def_readonly("register_census_live_epoch", &RpuGraphCache::Snapshot::register_census_live_epoch)
        .def_readonly("register_census_node_count", &RpuGraphCache::Snapshot::register_census_node_count)
        .def_readonly("register_census_dma_count", &RpuGraphCache::Snapshot::register_census_dma_count)
        .def_readonly("register_census_barrier_count", &RpuGraphCache::Snapshot::register_census_barrier_count)
        .def_readonly("register_census_kernel_count", &RpuGraphCache::Snapshot::register_census_kernel_count)
        .def_readonly("register_census_typed_kernel_count", &RpuGraphCache::Snapshot::register_census_typed_kernel_count)
        .def_readonly("register_census_no_ddr_kernel_count", &RpuGraphCache::Snapshot::register_census_no_ddr_kernel_count)
        .def_readonly("register_census_operand_count", &RpuGraphCache::Snapshot::register_census_operand_count)
        .def_readonly("register_census_weight_operand_count", &RpuGraphCache::Snapshot::register_census_weight_operand_count)
        .def_readonly("register_census_key_cache_operand_count", &RpuGraphCache::Snapshot::register_census_key_cache_operand_count)
        .def_readonly("register_census_value_cache_operand_count", &RpuGraphCache::Snapshot::register_census_value_cache_operand_count)
        .def_readonly("register_census_semantic_input_operand_count", &RpuGraphCache::Snapshot::register_census_semantic_input_operand_count)
        .def_readonly("register_census_ordered_node_kind_hash", &RpuGraphCache::Snapshot::register_census_ordered_node_kind_hash)
        .def_readonly("register_census_ordered_kernel_hash", &RpuGraphCache::Snapshot::register_census_ordered_kernel_hash)
        .def_readonly("register_census_outer_fast_hit", &RpuGraphCache::Snapshot::register_census_outer_fast_hit)
        .def_readonly("register_census_outer_fast_retained_validated", &RpuGraphCache::Snapshot::register_census_outer_fast_retained_validated)
        .def_readonly("register_census_outer_fast_mutable_slot_count", &RpuGraphCache::Snapshot::register_census_outer_fast_mutable_slot_count)
        .def_readonly("register_census_outer_fast_mutable_occurrence_count", &RpuGraphCache::Snapshot::register_census_outer_fast_mutable_occurrence_count)
        .def_readonly("register_census_host_reemitted_node_count", &RpuGraphCache::Snapshot::register_census_host_reemitted_node_count)
        .def_readonly("register_census_host_reemitted_kernel_count", &RpuGraphCache::Snapshot::register_census_host_reemitted_kernel_count);
}

}  // namespace graph_pybind
