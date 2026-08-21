// rpu_ops.h — Umbrella header for RPU backend operations
//
// Split into focused sub-headers for maintainability:
//   rpu_profile.h       — Profile data structures, macros, global switches, Align/CeilDiv
//   rpu_kernel_cache.h  — KernelId enum, KernelCache, QueueCache
//   rpu_spm_pool.h      — SpmBufferPool (SPM_POOL)
//   rpu_helpers.h       — SDPA tiling, memcpy utilities, HW constants
//   rpu_kernel_decls.h  — All kernel launch and wrapper function declarations
//
// Include this file in .cpp files for full backward compatibility.
#pragma once

#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <string>
#include <vector>
#include <map>
#include <functional>


using namespace at;

#include "rpu_profile.h"
#include "rpu_kernel_cache.h"
#include "rpu_spm_pool.h"
#include "rpu_helpers.h"
#include "rpu_kernel_decls.h"

// graph_runtime.h 末尾会把 GET_KERNEL 覆盖为 graph-aware 版本，并且
// 独占定义 GET_QUEUE：
//   GET_KERNEL(id) → RpuKernelGraph::active().get_kernel_reset(id)
//   GET_QUEUE(N)   → RpuQueueCache::instance().get(N) 返回 RpuQueue*
// op 层用 `auto* wq = GET_QUEUE(N)` 自动 deduce 到 RpuQueue*;wq->enqueu_kernel
// 走 RpuKernelGraph::active().enqueue(...) 路径。rpu_kernel_cache.h 不再暴露
// raw QueueCache 宏，避免直接 include 该 header 绕过 Graph。
#include "graph/graph_runtime.h"
