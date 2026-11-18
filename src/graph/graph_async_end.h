#pragma once
// Asynchronous replay end (GraphRuntimePolicy::async_replay_end, default off).
//
// For a top-level REPLAY of an opted-in Graph, end() submits the final prepared
// segment and returns while the device runs it.  Everything a synchronous end()
// does after its wait moves into rpu_graph_async_end_wait(): completion
// (Queue_t::finish), release of the DMA submission leases, the post-execution
// DDR flush and release of the keep-alive tensors.  Every later entry that can
// touch the device or its memory calls the wait first: the next Graph replay,
// eager kernels, sync points, RPU<->CPU copies, zero-copy views, immediate DMA,
// Graph end/abort/invalidate/destroy and the coordinator's physical, cleanup,
// reconfigure, quiescent and Graph-invalidation entries.
//
// RPU tensor frees issued while a submission is pending are deferred to the
// wait, so no DDR block the device may still read or write is reused.  An
// allocation that fails while frees are deferred waits and retries once.
//
// At most one submission is pending per process.  The execution coordinator
// already admits one execution owner at a time; the record is still guarded by
// a mutex so a deleter or copy on another thread waits instead of racing.
// HWPerf replays, child Graphs and any final segment that does not take the
// prepared sync-only path stay synchronous.

#include <cstddef>
#include <string>
#include <vector>

namespace at { class Tensor; }
namespace c10 { struct Storage; }
namespace rhino_lkn { class Queue_t; }
class RpuKernelGraph;
class RpuDmaSubmissionLease;

// True while a submitted final segment has not been waited.
bool rpu_graph_async_end_pending() noexcept;

// Completes the pending submission, if any, then publishes its post flush and
// releases its leases, keep-alive tensors and deferred frees.  Never throws: a
// failure is reported on stderr and counted in the stats.  `reason` names the
// entry for the stats.
void rpu_graph_async_end_wait(const char* reason) noexcept;

// Same as rpu_graph_async_end_wait() but only when `graph` owns the pending
// submission (Graph destroy, queue release, begin of the same Graph).
void rpu_graph_async_end_wait_for_graph(
    const RpuKernelGraph* graph, const char* reason) noexcept;

// Replay side (graph_runtime_execute.cpp).  After a successful asynchronous
// enqueue the launcher registers the queue and the submission lease; false
// means the bookkeeping failed and the caller must finish the queue itself.
bool rpu_graph_async_end_begin(::rhino_lkn::Queue_t* queue,
                               const RpuKernelGraph* graph,
                               RpuDmaSubmissionLease&& lease) noexcept;
// Moves a further lease of the same submission (the fixed-DMA preflight) into
// the record.  Without a pending record of `graph` it does nothing.
void rpu_graph_async_end_attach_lease(const RpuKernelGraph* graph,
                                      RpuDmaSubmissionLease&& lease) noexcept;

// end() side (graph_runtime.cpp): hands over the post-execution flush set and
// the keep-alive tensors.  Returns false (and leaves both arguments untouched)
// when `graph` has no pending submission.
bool rpu_graph_async_end_seal(const RpuKernelGraph* graph,
                              const char* graph_id,
                              std::vector<void*>& post_flush_ptrs,
                              std::vector<at::Tensor>& tensor_refs) noexcept;

// Allocator side (rpu_caching_allocator.cpp).  defer_free returns true when the
// free was queued for the wait; drain runs the queued frees after a completed
// wait; quarantine drops them (memory stays registered and is never reused)
// when completion could not be confirmed.
bool rpu_graph_async_end_defer_free(void* ptr, bool caching) noexcept;
void rpu_graph_async_end_drain_deferred_frees() noexcept;
void rpu_graph_async_end_quarantine_deferred_frees() noexcept;

// Independent host writes.  The owner of an RPU storage that no Graph listed in
// `graph_ids` ever references (no DMA endpoint, kernel operand or patch of
// those Graphs points into it) marks the storage's logical DDR allocation.
// A host->RPU copy into a marked allocation then skips the wait while one of
// those Graphs is pending: the device cannot read or write that memory, the
// copy's own flush publishes it, and the next submission waits as usual.
// The mark binds the allocation identity, so it ends with the storage.
// Proving the independence is the caller's job; marks are not checked against
// the Graph.  Throws when `tensor` is not a live RPU DDR storage.
void rpu_graph_async_end_mark_independent(
    const at::Tensor& tensor, const std::vector<std::string>& graph_ids);
// True when a submission is pending, sealed, owned by a Graph `dst`'s
// allocation is marked independent of, and `src` is a storage the process CPU
// allocator handed out (ordinary heap memory, never device memory; a zero-copy
// view, Graph HostDDR slot or from_blob wrapper has another or no allocator).
// Counted as "<graph>|independent_write" in the stats (zero overlap and
// blocked time).  False on any lookup failure: the caller waits.
bool rpu_graph_async_end_independent_write(const void* dst, size_t dst_bytes,
                                           const c10::Storage& src) noexcept;

// Diagnostics: one line per (graph, wait reason) with the wait count, the
// summed host time between submission and wait (overlap) and the summed time
// blocked in the wait; plus "submissions=" and "failures=" totals.
std::string rpu_graph_async_end_stats();
void rpu_graph_async_end_reset_stats() noexcept;
