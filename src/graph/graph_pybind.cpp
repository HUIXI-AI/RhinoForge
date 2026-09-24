// graph_pybind.cpp — S4 minimal pybind for Graph / GraphCache / GraphSignature
//
// 暴露给 Python 的语义跟 feature-auto-batch 一致:
//   - GraphSignature 是个 POD,Python 端可直接构造 + 赋字段 + 比较 + hash
//   - Graph 通过 shared_ptr 持有,生命周期管理交给 Python(GraphCache 内部用
//     unordered_map<sig, shared_ptr<Graph>> 持有,Python 也拿同一个 ref)
//   - GraphCache 默认无 capacity 上限,get_or_create 返回内部存的 shared_ptr 的引用
//
// 用法示例(decode 单步 capture/replay):
//   import rpu_backend
//   cache = rpu_backend.GraphCache()
//   sig = rpu_backend.GraphSignature()
//   sig.op_id = 1
//   sig.dyn_dims = [16, 1]
//
//   g = cache.get_or_create(sig)
//   g.begin(sig)              # PASSTHROUGH -> RECORDING
//   model(input_ids)          # ops 沿 RpuKernelGraph::active() 路径捕获
//   g.end()                   # RECORDING -> BUILT
//
//   # 第二次同 sig:命中 BUILT,走 REPLAYING -> BUILT
//   g = cache.get_or_create(sig)
//   g.begin(sig)
//   model(input_ids)
//   g.end()

#include "graph/graph_pybind.h"

#include "graph/graph_infra.h"
#include "graph/graph_runtime.h"
#include "core/rpu_kernel_decls.h"

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace graph_pybind {

void add_bindings(py::module_& m) {
    m.def("get_hwperf_evidence_failure_total",
          &rpu_get_hwperf_evidence_failure_total);
    // GraphSignature: A2 admission key 的 7 字段
    py::class_<GraphSignature>(m, "GraphSignature")
        .def(py::init<>())
        .def_readwrite("op_id",        &GraphSignature::op_id)
        .def_readwrite("op_id_str",    &GraphSignature::op_id_str)  // P6 human-readable label
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

    // SignatureLayer / LayeredSignature / helpers — admission policy 用来诊断
    // cache miss 时哪一层(GmId / TensorStruct / ShapeBucket / DtypeLayout /
    // ScalarFlags / BranchKey / SegmentKey)diverge。Port 自 auto-batch
    // src/rpu_backend.cpp:3122 (G5/G6/G7 SignatureTree diagnostics)。
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

    py::enum_<GraphHostOpDeferMode>(m, "GraphHostOpDeferMode")
        .value("Auto", GraphHostOpDeferMode::Auto)
        .value("ForceOff", GraphHostOpDeferMode::ForceOff)
        .value("ForceOn", GraphHostOpDeferMode::ForceOn);

    py::class_<GraphRuntimePolicy>(m, "GraphRuntimePolicy")
        .def(py::init([](
                 bool fast_replay_skip_sync,
                 bool force_oneshot_on_replay,
                 GraphHostOpDeferMode host_op_defer_mode,
                 bool siglip_isolate_patch_embed,
                 bool lingbot2_multiview_spm_z2,
                 bool fmb_fast_replay,
                 bool fmb_deep_fast_replay,
                 size_t lkn_max_batch_entries,
                 size_t lkn_kd_buf_mb,
                 size_t lkn_instr_buf_mb,
                 bool qwen35_legacy_27b_sdk_budget,
                 int execution_core_count,
                 size_t graph_arena_count) {
                 GraphRuntimePolicy result;
                 result.fast_replay_skip_sync = fast_replay_skip_sync;
                 result.force_oneshot_on_replay = force_oneshot_on_replay;
                 result.host_op_defer_mode = host_op_defer_mode;
                 result.siglip_isolate_patch_embed =
                     siglip_isolate_patch_embed;
                 result.lingbot2_multiview_spm_z2 =
                     lingbot2_multiview_spm_z2;
                 result.fmb_fast_replay = fmb_fast_replay;
                 result.fmb_deep_fast_replay = fmb_deep_fast_replay;
                 result.lkn_max_batch_entries = lkn_max_batch_entries;
                 result.lkn_kd_buf_mb = lkn_kd_buf_mb;
                 result.lkn_instr_buf_mb = lkn_instr_buf_mb;
                 result.qwen35_legacy_27b_sdk_budget =
                     qwen35_legacy_27b_sdk_budget;
                 result.execution_core_count = execution_core_count;
                 result.graph_arena_count = graph_arena_count;
                 validate_graph_runtime_policy(result);
                 return result;
             }),
             py::arg("fast_replay_skip_sync"),
             py::arg("force_oneshot_on_replay"),
             py::arg("host_op_defer_mode"),
             py::arg("siglip_isolate_patch_embed"),
             py::arg("lingbot2_multiview_spm_z2"),
             py::arg("fmb_fast_replay"),
             py::arg("fmb_deep_fast_replay"),
             py::arg("lkn_max_batch_entries"),
             py::arg("lkn_kd_buf_mb"),
             py::arg("lkn_instr_buf_mb"),
             py::arg("qwen35_legacy_27b_sdk_budget") = false,
             py::arg("execution_core_count") = 8,
             py::arg("graph_arena_count") = 0)
        .def("prepare_arenas", &prepare_graph_runtime_arenas);

    py::class_<GraphSegmentCensus>(m, "GraphSegmentCensus")
        .def_readonly("segment_id", &GraphSegmentCensus::segment_id)
        .def_readonly("start_idx", &GraphSegmentCensus::start_idx)
        .def_readonly("end_idx", &GraphSegmentCensus::end_idx)
        .def_readonly("core_count", &GraphSegmentCensus::core_count)
        .def_readonly("kernel_count", &GraphSegmentCensus::kernel_count)
        .def_readonly("dma_count", &GraphSegmentCensus::dma_count)
        .def_readonly("barrier_count", &GraphSegmentCensus::barrier_count)
        .def_property_readonly("manifest_metadata_count",
                              &GraphSegmentCensus::manifest_metadata_count);

    py::class_<GraphSegmentExecution>(m, "GraphSegmentExecution")
        .def_readonly("graph_lifetime_id", &GraphSegmentExecution::graph_lifetime_id)
        .def_readonly("build_generation", &GraphSegmentExecution::build_generation)
        .def_readonly("execution_ordinal", &GraphSegmentExecution::execution_ordinal)
        .def_readonly("parent_graph_lifetime_id", &GraphSegmentExecution::parent_graph_lifetime_id)
        .def_readonly("parent_execution_ordinal", &GraphSegmentExecution::parent_execution_ordinal)
        .def_readonly("execution_ancestors", &GraphSegmentExecution::execution_ancestors)
        .def_readonly("graph_name", &GraphSegmentExecution::graph_name)
        .def_readonly("execution_kind", &GraphSegmentExecution::execution_kind)
        .def_readonly("native_composite", &GraphSegmentExecution::native_composite)
        .def_readonly("graph_kernel_count", &GraphSegmentExecution::graph_kernel_count)
        .def_readonly("graph_dma_count", &GraphSegmentExecution::graph_dma_count)
        .def_readonly("graph_barrier_count", &GraphSegmentExecution::graph_barrier_count)
        .def_readonly("graph_child_count", &GraphSegmentExecution::graph_child_count)
        .def_readonly("graph_segment_count", &GraphSegmentExecution::graph_segment_count)
        .def_readonly("segment", &GraphSegmentExecution::segment)
        .def_readonly("hwperf_sequence", &GraphSegmentExecution::hwperf_sequence)
        .def_readonly("hwperf_path", &GraphSegmentExecution::hwperf_path)
        .def_readonly("companion_path", &GraphSegmentExecution::companion_path)
        .def_readonly("hwperf_rc", &GraphSegmentExecution::hwperf_rc)
        .def_readonly("companion_rc", &GraphSegmentExecution::companion_rc);

    py::class_<GraphStats>(m, "GraphStats")
        .def_readonly("graph_lifetime_id", &GraphStats::graph_lifetime_id)
        .def_readonly("build_generation", &GraphStats::build_generation)
        .def_readonly("execution_ordinal", &GraphStats::execution_ordinal)
        .def_readonly("kernel_count", &GraphStats::kernel_count)
        .def_readonly("data_node_count", &GraphStats::data_node_count)
        .def_readonly("dma_count", &GraphStats::dma_count)
        .def_readonly("barrier_count", &GraphStats::barrier_count)
        .def_readonly("child_graph_count", &GraphStats::child_graph_count)
        .def_readonly("child_graph_lifetime_ids", &GraphStats::child_graph_lifetime_ids)
        .def_readonly("hwperf_evidence_failure_total", &GraphStats::hwperf_evidence_failure_total)
        .def_readonly("segment_count", &GraphStats::segment_count)
        .def_readonly("segment_census", &GraphStats::segment_census)
        .def_readonly("last_segment_executions", &GraphStats::last_segment_executions);

    m.def("lingbot2_multiview_spm_z2_graph_stats",
          [](int64_t vision_handle, int64_t text_handle, int64_t expert_handle,
             int64_t plan_hash, const std::vector<int64_t>& descriptor,
             const std::vector<int64_t>& digest) {
              return rpu_lingbot2_multiview_spm_z2_graph_stats(
                  vision_handle, text_handle, expert_handle, plan_hash,
                  descriptor, digest);
          }, py::arg("vision_handle"), py::arg("text_handle"), py::arg("expert_handle"),
          py::arg("plan_hash"), py::arg("planner_descriptor"), py::arg("planner_digest"));

    // Graph (= RpuKernelGraph) — 用 shared_ptr 给 Python 拿 cache 内 entry 的引用
    py::class_<RpuKernelGraph, std::shared_ptr<RpuKernelGraph>>(m, "Graph")
        .def(py::init([]() { return make_registered_rpu_kernel_graph(); }))
        .def(py::init([](const GraphRuntimePolicy& runtime_policy) {
                 return make_registered_rpu_kernel_graph(runtime_policy);
             }),
             py::arg("runtime_policy"))
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
        .def("debug_stats", &RpuKernelGraph::debug_stats,
             py::return_value_policy::copy)
        .def("build_topology_hash", &RpuKernelGraph::build_topology_hash)
        .def("built_signature_identity", &RpuKernelGraph::built_signature_identity)
        .def("dump_replay_plan", &RpuKernelGraph::dump_replay_plan,
             "Return the pointer-free recorded Graph structure")
        .def("dump_tree", &RpuKernelGraph::dump_tree,
             "Return the pointer-free segment and node tree")
        .def("replayable",          &RpuKernelGraph::replayable)
        .def("has_built_signature", &RpuKernelGraph::has_built_signature)
        .def("built_signature",     &RpuKernelGraph::built_signature,
             py::return_value_policy::copy);

    // GraphCache (= RpuGraphCache) — Python 拿 entry 的 shared_ptr,生命周期跟
    // cache 一致(cache 持有 + Python 局部变量持有,双方都释放后才真析构)
    py::class_<RpuGraphCache>(m, "GraphCache")
        .def(py::init<>())
        .def(py::init<size_t>(), py::arg("max_entries"))
        .def(py::init<size_t, GraphRuntimePolicy>(),
             py::arg("max_entries"), py::arg("runtime_policy"))
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
        // P7.1b: Dynamo frontend admission policy 需要 touch_entry
        // (EntryRegistry.touch_after_call) 来跟踪 replay vs recapture 计数 +
        // 更新 LRU 顺序。C++ impl 已存在 graph_infra.cpp:250,P5 漏暴露。
        .def("touch_entry", &RpuGraphCache::touch_entry,
             py::arg("signature"), py::arg("replay") = false,
             "Mark cache entry as recently accessed; replay=True 更新 LRU + "
             "replay 计数,replay=False 标 recapture 计数")
        // Snapshot diagnostic (port auto-batch src/rpu_backend.cpp:3418):
        // 给 dynamo_backend.stats() / RpuDynamoBackend 内部统计用,每 entry
        // 出 replay_count / recapture_count / non_replayable_reason / 各种
        // graph 计数明细。C++ impl 已存在 graph_infra.cpp:270,P5 漏暴露。
        .def("snapshot",            &RpuGraphCache::snapshot)
        .def("cache_invariant_ok",  &RpuGraphCache::cache_invariant_ok);

    py::class_<RpuGraphCache::Snapshot>(m, "GraphCacheEntrySnapshot")
        .def_readonly("signature",                  &RpuGraphCache::Snapshot::signature)
        .def_readonly("kernel_count",               &RpuGraphCache::Snapshot::kernel_count)
        .def_readonly("segment_count",              &RpuGraphCache::Snapshot::segment_count)
        .def_readonly("local_spm_slot_count",       &RpuGraphCache::Snapshot::local_spm_slot_count)
        .def_readonly("replay_count",               &RpuGraphCache::Snapshot::replay_count)
        .def_readonly("recapture_count",            &RpuGraphCache::Snapshot::recapture_count)
        .def_readonly("non_replayable_reason",      &RpuGraphCache::Snapshot::non_replayable_reason)
        .def_readonly("data_node_count",            &RpuGraphCache::Snapshot::data_node_count)
        .def_readonly("dma_count", &RpuGraphCache::Snapshot::dma_count)
        .def_readonly("barrier_count", &RpuGraphCache::Snapshot::barrier_count)
        .def_readonly("child_graph_count", &RpuGraphCache::Snapshot::child_graph_count)
        .def_readonly("child_graph_lifetime_ids", &RpuGraphCache::Snapshot::child_graph_lifetime_ids)
        .def_readonly("hwperf_evidence_failure_total", &RpuGraphCache::Snapshot::hwperf_evidence_failure_total)
        .def_readonly("graph_lifetime_id", &RpuGraphCache::Snapshot::graph_lifetime_id)
        .def_readonly("build_generation", &RpuGraphCache::Snapshot::build_generation)
        .def_readonly("execution_ordinal", &RpuGraphCache::Snapshot::execution_ordinal)
        .def_readonly("segment_census", &RpuGraphCache::Snapshot::segment_census)
        .def_readonly("last_segment_executions", &RpuGraphCache::Snapshot::last_segment_executions)
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
