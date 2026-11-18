// graph_async_end.cpp — pending record for GraphRuntimePolicy::async_replay_end.
// Contract and wait points: graph_async_end.h.
#include "graph/graph_async_end.h"

#include "core/rpu_dma_endpoint.h"
#include "rpu_ops.h"

#include <ATen/core/Tensor.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/Storage.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <utility>

namespace {

using Clock = std::chrono::steady_clock;

struct PendingSubmission {
    ::rhino_lkn::Queue_t* queue = nullptr;
    const RpuKernelGraph* graph = nullptr;
    std::string graph_id;
    std::vector<RpuDmaSubmissionLease> leases;
    std::vector<void*> post_flush_ptrs;
    std::vector<at::Tensor> tensor_refs;
    Clock::time_point submitted{};
};

struct ReasonStats {
    uint64_t count = 0;
    double overlap_us = 0.0;
    double blocked_us = 0.0;
};

// Bound on the independence marks.  A model marks a handful of storages; the
// cap only stops a misuse from growing the map without limit.  Dropping marks
// is always safe: an unmarked write waits.
constexpr size_t kMaxIndependentMarks = 4096;

struct AsyncEndState {
    std::recursive_mutex mutex;
    PendingSubmission pending;
    std::map<std::string, ReasonStats> stats;
    // Logical DDR allocation id -> ids of the Graphs that never reference it.
    // Ids are process-monotonic and never reused, so a mark cannot outlive its
    // storage into a later allocation at the same address.
    std::map<uint64_t, std::vector<std::string>> independent;
    uint64_t submissions = 0;
    uint64_t failures = 0;
};

std::atomic<bool> g_pending{false};

// Leaked on purpose: a record must never be destroyed during static teardown,
// after the SDK or torch may already be gone.  rpu_shutdown() waits first.
AsyncEndState& state() {
    static AsyncEndState* instance = new AsyncEndState();
    return *instance;
}

void record_stats(AsyncEndState& s, const std::string& graph_id,
                  const char* reason, double overlap_us,
                  double blocked_us) noexcept {
    try {
        auto& entry = s.stats[(graph_id.empty() ? std::string("?") : graph_id) +
                              "|" + (reason != nullptr ? reason : "?")];
        ++entry.count;
        entry.overlap_us += overlap_us;
        entry.blocked_us += blocked_us;
    } catch (...) {
        // Diagnostics only.
    }
}

}  // namespace

bool rpu_graph_async_end_pending() noexcept {
    return g_pending.load(std::memory_order_acquire);
}

bool rpu_graph_async_end_begin(::rhino_lkn::Queue_t* queue,
                               const RpuKernelGraph* graph,
                               RpuDmaSubmissionLease&& lease) noexcept {
    if (queue == nullptr || graph == nullptr) return false;
    try {
        auto& s = state();
        std::lock_guard<std::recursive_mutex> lock(s.mutex);
        // replay_submit waited before the segment loop; a second record means
        // the single-owner contract was broken.  Stay synchronous.
        if (g_pending.load(std::memory_order_acquire)) return false;
        PendingSubmission pending;
        pending.leases.reserve(2);  // attach_lease() must not allocate
        pending.leases.push_back(std::move(lease));
        pending.queue = queue;
        pending.graph = graph;
        pending.submitted = Clock::now();
        s.pending = std::move(pending);
        ++s.submissions;
        g_pending.store(true, std::memory_order_release);
        return true;
    } catch (...) {
        // `lease` is moved only by the noexcept push_back after reserve();
        // on bad_alloc the caller still owns it and finishes synchronously.
        return false;
    }
}

void rpu_graph_async_end_attach_lease(const RpuKernelGraph* graph,
                                      RpuDmaSubmissionLease&& lease) noexcept {
    if (!lease) return;
    auto& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);
    if (!g_pending.load(std::memory_order_acquire) || s.pending.graph != graph) {
        return;
    }
    if (s.pending.leases.size() < s.pending.leases.capacity()) {
        s.pending.leases.push_back(std::move(lease));
        return;
    }
    // Unreachable with the reserved capacity; never drop a lease early.
    rpu_graph_async_end_wait("lease_attach");
}

bool rpu_graph_async_end_seal(const RpuKernelGraph* graph,
                              const char* graph_id,
                              std::vector<void*>& post_flush_ptrs,
                              std::vector<at::Tensor>& tensor_refs) noexcept {
    auto& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);
    if (!g_pending.load(std::memory_order_acquire) || s.pending.graph != graph) {
        return false;
    }
    s.pending.post_flush_ptrs.swap(post_flush_ptrs);
    s.pending.tensor_refs.swap(tensor_refs);
    try {
        s.pending.graph_id = graph_id != nullptr ? graph_id : "";
    } catch (...) {
        // Stats label only.
    }
    return true;
}

void rpu_graph_async_end_wait(const char* reason) noexcept {
    if (!g_pending.load(std::memory_order_acquire)) return;
    auto& s = state();
    std::unique_lock<std::recursive_mutex> lock(s.mutex);
    if (!g_pending.load(std::memory_order_acquire)) return;
    PendingSubmission pending = std::move(s.pending);
    s.pending = PendingSubmission{};

    const auto t0 = Clock::now();
    bool completed = true;
    try {
        pending.queue->finish();
    } catch (...) {
        completed = false;
    }
    const auto t1 = Clock::now();
    // Completion first, then the leases: the same order as a synchronous
    // submission that drops its lease after enqueu_batch(true) returns.
    pending.leases.clear();
    if (completed) {
        try {
            for (void* p : pending.post_flush_ptrs) {
                if (p != nullptr) rhino_lkn::RpuDdrFlush(p);
            }
        } catch (...) {
            completed = false;
        }
    }
    g_pending.store(false, std::memory_order_release);
    // Keep-alive references drop after execution, as in end() (pitfalls C-1).
    pending.tensor_refs.clear();
    if (!completed) {
        ++s.failures;
        std::cerr << "[rpu_graph] async replay end: completion of "
                  << (pending.graph_id.empty() ? "?" : pending.graph_id)
                  << " failed at " << (reason != nullptr ? reason : "?")
                  << "; frees deferred meanwhile are quarantined\n";
    }
    record_stats(
        s, pending.graph_id, reason,
        std::chrono::duration<double, std::micro>(t0 - pending.submitted).count(),
        std::chrono::duration<double, std::micro>(t1 - t0).count());
    // Drain under the record lock (the keep-alive release above already frees
    // under it): a later submission cannot begin until the frees deferred
    // behind this one are released, so none of its own deferred frees can be
    // drained by this wait.
    if (completed) {
        rpu_graph_async_end_drain_deferred_frees();
    } else {
        rpu_graph_async_end_quarantine_deferred_frees();
    }
}

void rpu_graph_async_end_wait_for_graph(const RpuKernelGraph* graph,
                                        const char* reason) noexcept {
    if (!g_pending.load(std::memory_order_acquire)) return;
    auto& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);
    if (!g_pending.load(std::memory_order_acquire) || s.pending.graph != graph) {
        return;
    }
    rpu_graph_async_end_wait(reason);
}

void rpu_graph_async_end_mark_independent(
        const at::Tensor& tensor, const std::vector<std::string>& graph_ids) {
    TORCH_CHECK(tensor.defined() &&
                    tensor.device().type() == c10::DeviceType::PrivateUse1,
                "rpu_graph_async_end_mark_independent: needs a defined RPU "
                "tensor");
    const auto storage = tensor.storage();
    const auto endpoint = rpu_resolve_cpu_dma_endpoint(
        storage.data_ptr().get(), storage.nbytes(),
        "rpu_graph_async_end_mark_independent");
    TORCH_CHECK(endpoint.allocation_id != 0,
                "rpu_graph_async_end_mark_independent: storage is not a "
                "logical DDR allocation");
    auto& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);
    if (s.independent.size() >= kMaxIndependentMarks &&
        s.independent.find(endpoint.allocation_id) == s.independent.end()) {
        s.independent.clear();
    }
    auto& ids = s.independent[endpoint.allocation_id];
    for (const auto& id : graph_ids) {
        if (!id.empty() && std::find(ids.begin(), ids.end(), id) == ids.end()) {
            ids.push_back(id);
        }
    }
}

bool rpu_graph_async_end_independent_write(const void* dst, size_t dst_bytes,
                                           const c10::Storage& src) noexcept {
    if (!g_pending.load(std::memory_order_acquire)) return false;
    if (dst == nullptr || dst_bytes == 0) return false;
    try {
        // A CPU view of device memory the pending Graph may still write (a
        // zero-copy view, a Graph HostDDR slot, a from_blob wrapper) is not
        // the CPU allocator's storage.  No SDK lookup: RpuGetHostddr logs a
        // warning for every ordinary host pointer.
        if (!src || src.data() == nullptr || src.allocator() == nullptr ||
            src.allocator() != c10::GetCPUAllocator()) {
            return false;
        }
        auto& s = state();
        std::string graph_id;
        uint64_t generation = 0;
        {
            std::lock_guard<std::recursive_mutex> lock(s.mutex);
            if (!g_pending.load(std::memory_order_acquire) || s.independent.empty()) {
                return false;
            }
            graph_id = s.pending.graph_id;
            generation = s.submissions;
        }
        // An unsealed record has no Graph id yet; end() has not returned.
        if (graph_id.empty()) return false;
        // Resolve outside the record lock: the allocator's lock is never
        // taken under it on this path.
        const auto endpoint = rpu_resolve_cpu_dma_endpoint(
            dst, dst_bytes, "rpu_graph_async_end_independent_write");
        if (endpoint.allocation_id == 0) return false;
        std::lock_guard<std::recursive_mutex> lock(s.mutex);
        if (!g_pending.load(std::memory_order_acquire) ||
            s.submissions != generation || s.pending.graph_id != graph_id) {
            return false;
        }
        const auto it = s.independent.find(endpoint.allocation_id);
        if (it == s.independent.end() ||
            std::find(it->second.begin(), it->second.end(), graph_id) ==
                it->second.end()) {
            return false;
        }
        record_stats(s, graph_id, "independent_write", 0.0, 0.0);
        return true;
    } catch (...) {
        return false;
    }
}

std::string rpu_graph_async_end_stats() {
    auto& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);
    std::ostringstream out;
    out << "submissions=" << s.submissions << " failures=" << s.failures
        << " pending=" << (g_pending.load(std::memory_order_acquire) ? 1 : 0)
        << "\n";
    for (const auto& [key, entry] : s.stats) {
        out << key << " count=" << entry.count
            << " overlap_us=" << entry.overlap_us
            << " blocked_us=" << entry.blocked_us << "\n";
    }
    return out.str();
}

void rpu_graph_async_end_reset_stats() noexcept {
    auto& s = state();
    std::lock_guard<std::recursive_mutex> lock(s.mutex);
    s.stats.clear();
    s.submissions = 0;
    s.failures = 0;
}
