// graph_infra.cpp — RpuGraphCache and HostCallback classification implementation.
//
// This file backs the non-executing graph infrastructure declared in
// graph_infra.h. RpuGraphCache owns RpuKernelGraph instances but does not execute
// them; HostCallback tiering classifies CPU fallback ops before runtime capture.

#include "graph/graph_infra.h"
#include "graph/graph_runtime.h"

#include <ATen/core/enum_tag.h>

#include <torch/torch.h>

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace {

// 检查给定 graph 当前是否在线程 active stack 上（全栈包含检查）。嵌套
// cache.capture 中只检查栈顶会漏掉仍活跃的 outer graph，导致误删 entry；
// 因此必须检查完整 active stack。
bool is_active_now(RpuKernelGraph& g) {
    return RpuKernelGraph::is_on_active_stack(g);
}

constexpr size_t layer_index(SignatureLayer layer) {
    return static_cast<size_t>(layer);
}

std::string hex_u64(uint64_t value) {
    std::ostringstream os;
    os << "0x" << std::hex << value;
    return os.str();
}

std::string join_i64(const std::vector<int64_t>& values) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            os << ",";
        }
        os << values[i];
    }
    os << "]";
    return os.str();
}

const char* dtype_name(uint8_t dtype) {
    switch (dtype) {
        case 3:  return "int32";
        case 4:  return "int64";
        case 5:  return "half";
        case 6:  return "float";
        case 11: return "bool";
        case 15: return "bfloat16";
        default: return nullptr;
    }
}

std::string join_dtypes(const std::vector<uint8_t>& values) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            os << ",";
        }
        if (const char* name = dtype_name(values[i])) {
            os << name;
        } else {
            os << static_cast<int>(values[i]);
        }
    }
    os << "]";
    return os.str();
}

std::string tensor_struct_text(const GraphSignature& sig) {
    std::ostringstream os;
    os << "TensorStruct(shape_count=" << sig.shapes.size()
       << ", dyn_dim_count=" << sig.dyn_dims.size()
       << ", dtype_count=" << sig.dtypes.size() << ")";
    return os.str();
}

std::string shape_bucket_text(const GraphSignature& sig) {
    std::ostringstream os;
    os << "ShapeBucket(shapes=" << join_i64(sig.shapes)
       << ", dyn_dims=" << join_i64(sig.dyn_dims) << ")";
    return os.str();
}

std::string dtype_layout_text(const GraphSignature& sig) {
    std::ostringstream os;
    os << "DtypeLayout(dtypes=" << join_dtypes(sig.dtypes) << ")";
    return os.str();
}

std::string scalar_flags_text(const GraphSignature& sig) {
    std::ostringstream os;
    os << "ScalarFlags(flags=" << hex_u64(sig.flags) << ")";
    return os.str();
}

std::string branch_key_text(const GraphSignature& sig) {
    std::ostringstream os;
    os << "BranchKey(" << hex_u64(sig.branch_key) << ")";
    return os.str();
}

std::string segment_key_text(const GraphSignature& sig) {
    std::ostringstream os;
    os << "SegmentKey(" << hex_u64(sig.segment_key) << ")";
    return os.str();
}

std::string miss_detail(SignatureLayer layer,
                        const GraphSignature& got,
                        const GraphSignature& expected) {
    std::ostringstream os;
    switch (layer) {
        case SignatureLayer::GmId:
            os << "got op_id=" << got.op_id
               << ", expected op_id=" << expected.op_id;
            break;
        case SignatureLayer::TensorStruct:
            os << "got shape_count=" << got.shapes.size()
               << ", dyn_dim_count=" << got.dyn_dims.size()
               << ", dtype_count=" << got.dtypes.size()
               << "; expected shape_count=" << expected.shapes.size()
               << ", dyn_dim_count=" << expected.dyn_dims.size()
               << ", dtype_count=" << expected.dtypes.size();
            break;
        case SignatureLayer::ShapeBucket:
            os << "got shapes=" << join_i64(got.shapes)
               << ", dyn_dims=" << join_i64(got.dyn_dims)
               << "; expected shapes=" << join_i64(expected.shapes)
               << ", dyn_dims=" << join_i64(expected.dyn_dims);
            break;
        case SignatureLayer::DtypeLayout:
            os << "got dtypes=" << join_dtypes(got.dtypes)
               << "; expected dtypes=" << join_dtypes(expected.dtypes);
            break;
        case SignatureLayer::ScalarFlags:
            os << "got flags=" << hex_u64(got.flags)
               << "; expected flags=" << hex_u64(expected.flags);
            break;
        case SignatureLayer::BranchKey:
            os << "got branch_key=" << hex_u64(got.branch_key)
               << "; expected branch_key=" << hex_u64(expected.branch_key);
            break;
        case SignatureLayer::SegmentKey:
            os << "got segment_key=" << hex_u64(got.segment_key)
               << "; expected segment_key=" << hex_u64(expected.segment_key);
            break;
        case SignatureLayer::kCount:
            os << "all layer hashes match; flat signature still differs";
            break;
    }
    return os.str();
}

struct SignatureTreeDumpEntry {
    GraphSignature sig;
    LayeredSignature layered;
    const RpuGraphCacheEntry* entry = nullptr;
};

bool signature_tree_less(const SignatureTreeDumpEntry& a,
                         const SignatureTreeDumpEntry& b) {
    const SignatureLayer order[] = {
        SignatureLayer::TensorStruct,
        SignatureLayer::ShapeBucket,
        SignatureLayer::DtypeLayout,
        SignatureLayer::ScalarFlags,
        SignatureLayer::BranchKey,
        SignatureLayer::SegmentKey,
    };
    for (SignatureLayer layer : order) {
        const uint64_t ah = a.layered.layer_hash(layer);
        const uint64_t bh = b.layered.layer_hash(layer);
        if (ah != bh) {
            return ah < bh;
        }
    }
    return a.sig.to_string() < b.sig.to_string();
}

}  // namespace

RpuKernelGraph& RpuGraphCache::get_or_create(const GraphSignature& sig) {
    auto it = entries_.find(sig);
    if (it != entries_.end()) {
        return *it->second.graph;
    }
    TORCH_CHECK(entries_.size() < max_entries_,
                "RpuGraphCache: capacity reached (",
                entries_.size(), "/", max_entries_,
                "); manual evict() required (A3 will add LRU)");

    RpuGraphCacheEntry entry;
    entry.graph = make_registered_rpu_kernel_graph();
    entry.signature = sig;
    auto [ins_it, inserted] = entries_.emplace(sig, std::move(entry));
    if (inserted) {
        add_to_layered_index(sig);
    }
    return *ins_it->second.graph;
}

RpuKernelGraph* RpuGraphCache::lookup(const GraphSignature& sig) {
    auto it = entries_.find(sig);
    return it == entries_.end() ? nullptr : it->second.graph.get();
}

std::shared_ptr<RpuKernelGraph> RpuGraphCache::lookup_shared(const GraphSignature& sig) {
    auto it = entries_.find(sig);
    return it == entries_.end() ? nullptr : it->second.graph;
}

void RpuGraphCache::evict(const GraphSignature& sig) {
    RpuExecutionCleanupGuard cleanup("RpuGraphCache::evict");
    auto it = entries_.find(sig);
    if (it == entries_.end()) return;
    TORCH_CHECK(!is_active_now(*it->second.graph),
                "RpuGraphCache: cannot evict a graph that is currently active");
    TORCH_CHECK(it->second.graph.use_count() == 1,
                "RpuGraphCache: cannot evict a graph with external references "
                "(use_count=", it->second.graph.use_count(),
                "); release ChildGraph/lookup_shared references first");
    entries_.erase(it);
    remove_from_layered_index(sig);
}

void RpuGraphCache::clear() {
    RpuExecutionCleanupGuard cleanup("RpuGraphCache::clear");
    for (auto& [sig, entry] : entries_) {
        TORCH_CHECK(!is_active_now(*entry.graph),
                    "RpuGraphCache: cannot clear() while a cached graph is active");
        TORCH_CHECK(entry.graph.use_count() == 1,
                    "RpuGraphCache: cannot clear() while a graph has external "
                    "references (sig=", sig.to_string(),
                    ", use_count=", entry.graph.use_count(), ")");
    }
    entries_.clear();
    by_gm_id_.clear();
    by_gm_branch_.clear();
}

void RpuGraphCache::touch_entry(const GraphSignature& sig, bool replay) {
    auto it = entries_.find(sig);
    if (it == entries_.end()) return;  // 静默:Python 端可能在 evict 后调用
    auto& e = it->second;
    if (replay) {
        e.replay_count += 1;
    } else {
        const GraphStats& stats = e.graph->debug_stats();
        e.built_kernel_count = stats.kernel_count;
        e.built_segment_count = stats.segment_count;
        e.built_local_spm_slot_count = e.graph->local_spm_pool().size();
        // 非 replay 出口:RECORDING 完成 → 第一次 BUILT;若已经 built_once 则
        // 是 invalidate + 重录(recapture)。
        if (e.built_once) {
            e.recapture_count += 1;
        }
        e.built_once = true;
    }
}

std::vector<RpuGraphCache::Snapshot> RpuGraphCache::snapshot() const {
    std::vector<Snapshot> out;
    out.reserve(entries_.size());
    for (const auto& [sig, entry] : entries_) {
        Snapshot s;
        s.signature = sig;
        s.kernel_count = entry.built_kernel_count;
        s.segment_count = entry.built_segment_count;
        s.local_spm_slot_count = entry.built_local_spm_slot_count;
        s.replay_count = entry.replay_count;
        s.recapture_count = entry.recapture_count;
        if (entry.graph) {
            const auto& gs = entry.graph->debug_stats();
            s.non_replayable_reason          = gs.non_replayable_reason;
            s.data_node_count                = gs.data_node_count;
            s.boundary_flush_count           = gs.boundary_flush_count;
            s.prepared_segment_hit_total     = gs.prepared_segment_hit_total;
            s.prepared_segment_miss_total    = gs.prepared_segment_miss_total;
            s.hw_batch_submit_total          = gs.hw_batch_submit_total;
            s.host_callback_tier1_count      = gs.host_callback_tier1_count;
            s.host_callback_tier2_count      = gs.host_callback_tier2_count;
            s.host_callback_tier3_count      = gs.host_callback_tier3_count;
            s.tier3_oneshot_count            = gs.tier3_oneshot_count;
            s.register_census_present = gs.register_census_present;
            s.register_census_build_committed =
                gs.register_census_build_committed;
            s.register_census_policy_fingerprint =
                gs.register_census_policy_fingerprint;
            s.register_census_policy_digest =
                gs.register_census_policy_digest;
            s.register_census_plan_hash = gs.register_census_plan_hash;
            s.register_census_live_epoch = gs.register_census_live_epoch;
            s.register_census_node_count = gs.register_census.node_count;
            s.register_census_dma_count = gs.register_census.dma_count;
            s.register_census_barrier_count = gs.register_census.barrier_count;
            s.register_census_kernel_count = gs.register_census.kernel_count;
            s.register_census_typed_kernel_count =
                gs.register_census.typed_kernel_count;
            s.register_census_no_ddr_kernel_count =
                gs.register_census.no_ddr_kernel_count;
            s.register_census_operand_count =
                gs.register_census.operand_count;
            s.register_census_weight_operand_count =
                gs.register_census.weight_operand_count;
            s.register_census_key_cache_operand_count =
                gs.register_census.key_cache_operand_count;
            s.register_census_value_cache_operand_count =
                gs.register_census.value_cache_operand_count;
            s.register_census_semantic_input_operand_count =
                gs.register_census.semantic_input_operand_count;
            s.register_census_ordered_node_kind_hash =
                gs.register_census.ordered_node_kind_hash;
            s.register_census_ordered_kernel_hash =
                gs.register_census.ordered_kernel_hash;
            s.register_census_outer_fast_hit =
                gs.register_census_outer_fast_hit;
            s.register_census_outer_fast_retained_validated =
                gs.register_census_outer_fast_retained_validated;
            s.register_census_outer_fast_mutable_slot_count =
                gs.register_census_outer_fast_mutable_slot_count;
            s.register_census_outer_fast_mutable_occurrence_count =
                gs.register_census_outer_fast_mutable_occurrence_count;
            s.register_census_host_reemitted_node_count =
                gs.register_census_host_reemitted_node_count;
            s.register_census_host_reemitted_kernel_count =
                gs.register_census_host_reemitted_kernel_count;
        }
        out.push_back(std::move(s));
    }
    return out;
}

bool RpuGraphCache::cache_invariant_ok() const {
    size_t indexed_by_gm = 0;
    for (const auto& [op_id, bucket] : by_gm_id_) {
        indexed_by_gm += bucket.size();
        if (bucket.empty()) {
            return false;
        }
        for (const auto& sig : bucket) {
            if (sig.op_id != op_id) {
                return false;
            }
            if (entries_.find(sig) == entries_.end()) {
                return false;
            }
        }
    }
    if (indexed_by_gm != entries_.size()) {
        return false;
    }

    size_t indexed_by_branch = 0;
    for (const auto& [op_id, branch_buckets] : by_gm_branch_) {
        if (branch_buckets.empty()) {
            return false;
        }
        for (const auto& [branch_key, bucket] : branch_buckets) {
            indexed_by_branch += bucket.size();
            if (bucket.empty()) {
                return false;
            }
            for (const auto& sig : bucket) {
                if (sig.op_id != op_id || sig.branch_key != branch_key) {
                    return false;
                }
                if (entries_.find(sig) == entries_.end()) {
                    return false;
                }
            }
        }
    }
    if (indexed_by_branch != entries_.size()) {
        return false;
    }

    for (const auto& [sig, entry] : entries_) {
        (void)entry;
        auto bucket_it = by_gm_id_.find(sig.op_id);
        if (bucket_it == by_gm_id_.end()) {
            return false;
        }
        bool found = false;
        for (const auto& indexed_sig : bucket_it->second) {
            if (indexed_sig == sig) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }

        auto branch_op_it = by_gm_branch_.find(sig.op_id);
        if (branch_op_it == by_gm_branch_.end()) {
            return false;
        }
        auto branch_bucket_it = branch_op_it->second.find(sig.branch_key);
        if (branch_bucket_it == branch_op_it->second.end()) {
            return false;
        }
        found = false;
        for (const auto& indexed_sig : branch_bucket_it->second) {
            if (indexed_sig == sig) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

std::map<int64_t, size_t> RpuGraphCache::debug_bucket_counts() const {
    std::map<int64_t, size_t> out;
    for (const auto& [op_id, bucket] : by_gm_id_) {
        out[op_id] = bucket.size();
    }
    return out;
}

std::map<int64_t, std::map<uint64_t, size_t>>
RpuGraphCache::debug_branch_counts() const {
    std::map<int64_t, std::map<uint64_t, size_t>> out;
    for (const auto& [op_id, branch_buckets] : by_gm_branch_) {
        auto& by_branch = out[op_id];
        for (const auto& [branch_key, bucket] : branch_buckets) {
            by_branch[branch_key] = bucket.size();
        }
    }
    return out;
}

std::string RpuGraphCache::dump_signature_tree() const {
    if (entries_.empty()) {
        return "<empty>";
    }

    std::ostringstream os;
    bool first_bucket = true;
    std::map<int64_t, std::vector<SignatureTreeDumpEntry>> buckets;
    for (const auto& [op_id, bucket] : by_gm_id_) {
        auto& out_bucket = buckets[op_id];
        out_bucket.reserve(bucket.size());
        for (const auto& sig : bucket) {
            auto entry_it = entries_.find(sig);
            if (entry_it == entries_.end()) {
                continue;
            }
            out_bucket.push_back({sig, layered_from(sig), &entry_it->second});
        }
        std::sort(out_bucket.begin(), out_bucket.end(), signature_tree_less);
    }

    for (const auto& [op_id, bucket] : buckets) {
        if (bucket.empty()) {
            continue;
        }
        if (!first_bucket) {
            os << "\n";
        }
        first_bucket = false;
        os << "GmId(op_id=" << op_id << ")";
        for (size_t i = 0; i < bucket.size(); ++i) {
            const auto& item = bucket[i];
            os << "\n  " << tensor_struct_text(item.sig)
               << "\n    " << shape_bucket_text(item.sig)
               << "\n      " << dtype_layout_text(item.sig)
               << "\n        " << scalar_flags_text(item.sig)
               << "\n          " << branch_key_text(item.sig)
               << "\n            " << segment_key_text(item.sig)
               << " -> entry[" << i
               << "] (replay=" << item.entry->replay_count
               << ", recapture=" << item.entry->recapture_count << ")";
        }
    }

    std::string out = os.str();
    return out.empty() ? std::string("<empty>") : out;
}

std::string RpuGraphCache::explain_miss(const GraphSignature& sig) const {
    if (entries_.find(sig) != entries_.end()) {
        return "hit: exact signature exists";
    }

    auto bucket_it = by_gm_id_.find(sig.op_id);
    if (bucket_it == by_gm_id_.end() || bucket_it->second.empty()) {
        std::ostringstream os;
        os << "miss at GmId: no entry with op_id=" << sig.op_id;
        return os.str();
    }

    const LayeredSignature query = layered_from(sig);
    bool have_best = false;
    SignatureLayer best_layer = SignatureLayer::GmId;
    GraphSignature best_sig;

    std::vector<SignatureTreeDumpEntry> bucket;
    bucket.reserve(bucket_it->second.size());
    for (const auto& candidate : bucket_it->second) {
        auto entry_it = entries_.find(candidate);
        if (entry_it == entries_.end()) {
            continue;
        }
        bucket.push_back({candidate, layered_from(candidate), &entry_it->second});
    }
    std::sort(bucket.begin(), bucket.end(), signature_tree_less);

    for (const auto& item : bucket) {
        const SignatureLayer diverge = query.first_diverge(item.layered);
        if (!have_best || layer_index(diverge) > layer_index(best_layer)) {
            best_layer = diverge;
            best_sig = item.sig;
            have_best = true;
        }
    }

    if (!have_best) {
        return "miss at GmId: no live entry in op_id bucket";
    }

    std::ostringstream os;
    os << "miss at " << signature_layer_name(best_layer)
       << ": closest entry diverges at "
       << signature_layer_name(best_layer)
       << " (" << miss_detail(best_layer, sig, best_sig) << ")";
    return os.str();
}

void RpuGraphCache::add_to_layered_index(const GraphSignature& sig) {
    by_gm_id_[sig.op_id].push_back(sig);
    by_gm_branch_[sig.op_id][sig.branch_key].push_back(sig);
}

void RpuGraphCache::remove_from_layered_index(const GraphSignature& sig) {
    auto bucket_it = by_gm_id_.find(sig.op_id);
    if (bucket_it == by_gm_id_.end()) {
        return;
    }
    auto& bucket = bucket_it->second;
    for (auto it = bucket.begin(); it != bucket.end(); ++it) {
        if (*it == sig) {
            bucket.erase(it);
            break;
        }
    }
    if (bucket.empty()) {
        by_gm_id_.erase(bucket_it);
    }

    auto branch_op_it = by_gm_branch_.find(sig.op_id);
    if (branch_op_it == by_gm_branch_.end()) {
        return;
    }
    auto branch_bucket_it = branch_op_it->second.find(sig.branch_key);
    if (branch_bucket_it == branch_op_it->second.end()) {
        return;
    }
    auto& branch_bucket = branch_bucket_it->second;
    for (auto it = branch_bucket.begin(); it != branch_bucket.end(); ++it) {
        if (*it == sig) {
            branch_bucket.erase(it);
            break;
        }
    }
    if (branch_bucket.empty()) {
        branch_op_it->second.erase(branch_bucket_it);
    }
    if (branch_op_it->second.empty()) {
        by_gm_branch_.erase(branch_op_it);
    }
}

// 线程局部默认实例。
RpuGraphCache& default_graph_cache() {
    static thread_local RpuGraphCache inst;
    return inst;
}

// =============================================================================
// HostCallback tier classification
// =============================================================================

namespace {

// ATen tags do not cover every RNG spelling we need, so keep a small fallback
// table keyed as "name.overload" or just "name" when the overload is empty.
const std::unordered_set<std::string>& rng_op_set() {
    static const std::unordered_set<std::string> s = {
        "aten::multinomial",
        "aten::multinomial.out",
        "aten::randperm",
        "aten::randperm.out",
        "aten::randperm.generator",
        "aten::randperm.generator_out",
        "aten::bernoulli",
        "aten::bernoulli.out",
        "aten::bernoulli.p",
        "aten::bernoulli_.float",
        "aten::bernoulli_.Tensor",
        "aten::normal_",
        "aten::normal.float_float",
        "aten::normal.float_float_out",
        "aten::uniform_",
        "aten::cauchy_",
        "aten::exponential_",
        "aten::log_normal_",
        "aten::geometric_",
        "aten::random_",
        "aten::random_.from",
        "aten::random_.to",
        "aten::dropout",
        "aten::native_dropout",
        "aten::feature_dropout",
        "aten::alpha_dropout",
        "aten::feature_alpha_dropout",
        "aten::poisson",
    };
    return s;
}

std::string opname_key(const c10::OperatorName& name) {
    std::string key = name.name;
    if (!name.overload_name.empty()) {
        key += "." + name.overload_name;
    }
    return key;
}

}  // namespace

bool is_data_dependent_shape(const c10::OperatorName& name) {
    auto handle = c10::Dispatcher::singleton().findOp(name);
    if (!handle.has_value()) return false;
    return handle->hasTag(at::Tag::data_dependent_output);
}

bool is_rng_op(const c10::OperatorName& name) {
    auto handle = c10::Dispatcher::singleton().findOp(name);
    if (handle.has_value() && handle->hasTag(at::Tag::nondeterministic_seeded)) {
        return true;
    }
    return rng_op_set().count(opname_key(name)) > 0;
}

std::vector<size_t> collect_output_arg_indices(const c10::FunctionSchema& schema) {
    std::vector<size_t> out;
    for (size_t i = 0; i < schema.arguments().size(); ++i) {
        if (is_mutable_output_arg(schema, i)) {
            out.push_back(i);
        }
    }
    return out;
}

HostCallbackTier classify_host_op(const c10::OperatorHandle& op) {
    const auto& schema = op.schema();
    const auto& opname = schema.operator_name();

    // HostCallback nodes only handle tensor returns. Scalar-returning ops such
    // as _local_scalar_dense must reject into the sync-point path rather than
    // masquerading as Tier3 dynamic-shape work.
    if (!all_returns_are_tensors(schema)) {
        return HostCallbackTier::Reject;
    }

    if (is_rng_op(opname) || is_data_dependent_shape(opname)) {
        return HostCallbackTier::Tier3;
    }

    bool has_out_arg = false;
    for (size_t i = 0; i < schema.arguments().size(); ++i) {
        if (is_mutable_output_arg(schema, i)) {
            if (is_inplace_self_mutation(schema, i)) {
                return HostCallbackTier::Reject;
            }
            has_out_arg = true;
        }
    }
    if (has_out_arg) {
        return HostCallbackTier::Tier1;
    }

    return HostCallbackTier::Tier2;
}
