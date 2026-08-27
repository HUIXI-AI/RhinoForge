// graph_spm_reset_guard.cpp — Graph-aware temporary SPM reset guard.
//
// spm_alloc_reset_temporary 的 graph-aware 入口。从 rpu_backend.cpp 的
// `m.def("spm_alloc_reset_temporary", ...)` 转入这里,按 active() graph state
// 分流:
//
//   PASSTHROUGH / BUILT:
//     直接 SPM_ALLOC.reset_temporary() 推进 host 状态(t_end_=0, generation_++)。
//     原 eager 行为。
//
//   RECORDING:
//     仍执行 SPM_ALLOC.reset_temporary() — Python 侧 ensure_allocated 需要靠
//     gen_changed 触发 Path 2 重新分配临时缓冲区（见 FusedModelBase）。
//     但 RECORDING 期 RPU kernel 还在 nodes_ 没跑,subsystem 边界 SPM 已被 host
//     侧"释放"却仍被后续 graph kernel 引用(同一段 SPM 地址会被下一 subsystem
//     的 ensure_allocated 重 alloc 并 bake 进 kernel set_regs)。kernel 真正执行
//     时,SigLIP / Gemma / AdaRMS 各自的 SPM 区可能 overlap。
//
//     处理:`mark_non_replayable(reason="spm reset during recording")` —
//     end() 走 oneshot 路径(按 nodes_ 顺序同步执行,enqueu_batch wait_finish
//     保证 segment 间 SPM 区不交叠),但 graph 不进 BUILT/REPLAYING cache。
//
//   REPLAYING:
//     mark_non_replayable 已经在 RECORDING 期 prevent 进 REPLAYING,真正进到
//     这里说明 graph 状态机有更深 bug,fail-loud。Pi0.5 e2e Python adapter 在
//     REPLAYING 期也会重跑 forward 函数(包括 reset_temporary 调用)— 但只要
//     RECORDING 期被 mark_non_replayable,后续 begin(sig) 命中 BUILT-not-replayable
//     状态走 RECORDING-rebuild,不会落到这里。
//
// 不直接 include graph_runtime.h 进 rpu_backend.cpp 的理由跟
// custom_cpu_fallback_dispatch.cpp 一致(c10::StorageImpl inline body
// undefined symbol)。

#include "graph/graph_runtime.h"
#include "rpu_spm_allocator.h"
#include "rpu_profile.h"        // log_at(N)

#include <c10/util/Exception.h>
#include <iostream>

void graph_aware_spm_alloc_reset_temporary() {
    RpuExecutionAllocatorMutationGuard mutation(
        "spm_alloc_reset_temporary");
    if (!mutation.graph_claim_active()) {
        SPM_ALLOC.reset_temporary();
        return;
    }

    TORCH_CHECK(
        RpuKernelGraph::has_active(),
        "spm_alloc_reset_temporary: process coordinator/TLS Graph drift");
    auto& g = RpuKernelGraph::active();
    const auto state = g.state();

    switch (state) {
    case RpuKernelGraph::State::PASSTHROUGH:
    case RpuKernelGraph::State::BUILT:
        // 原 eager 行为。BUILT 实际上 begin() 已经退出 active stack,
        // active() 返回 fallback_graph_,state==PASSTHROUGH;保留 BUILT 分支
        // 做防御。
        SPM_ALLOC.reset_temporary();
        return;

    case RpuKernelGraph::State::RECORDING: {
        static thread_local bool warned_once = false;
        if (!warned_once) {
            warned_once = true;
            if (log_at(2)) {
                std::cerr << "[RpuKernelGraph] WARN: spm_alloc_reset_temporary "
                             "called during RECORDING. Graph marked "
                             "non-replayable; sync_point flushes recorded prefix "
                             "(Pi0.5 adapter `output.to(cpu)` would "
                             "otherwise read stale DDR).\n";
            }
        }
        g.mark_non_replayable("spm_alloc_reset_temporary during RECORDING");
        // Pi0.5 adapter 模式是 fused_op → reset_temporary →
        // output.to(cpu)。RPU→CPU copy 走 PyTorch CompositeImplicitAutograd 分解
        // (empty + copy_),copy_ 按 dst 设备分派到 CPU side → 直接 memcpy 共享
        // DDR,完全绕过我们 PrivateUse1 的 rpu_to_copy / rpu_to_cpu_zerocopy 实现。
        // 所以无法在 to_copy 路径上 hook sync。改在 reset_temporary 这里做 sync
        // —— Pi0.5 adapter 出口必经,每个子系统返回 CPU view 前 nodes_ 已经 flush。
        // sync_point() 内部会 execute_graph_oneshot + flush + clear nodes_/kernels_,
        // 保持 RECORDING 态继续录后续。
        if (g.graph_size() > 0) {
            g.sync_point();
        }
        // sync_point 之后 host 才执行 reset_temporary,Python ensure_allocated
        // 看到 gen_changed 触发 Path 2 重 alloc。
        SPM_ALLOC.reset_temporary();
        return;
    }

    case RpuKernelGraph::State::REPLAYING:
        TORCH_CHECK(
            false,
            "spm_alloc_reset_temporary called during graph REPLAYING. "
            "This indicates a graph state machine bug — RECORDING should "
            "have marked the graph non_replayable on the first reset call, "
            "preventing subsequent REPLAYING entry.");
    }
}

void process_guarded_spm_alloc_init() {
    RpuExecutionCleanupGuard cleanup("spm_alloc_init");
    SPM_ALLOC.init();
}

void process_guarded_spm_alloc_reset_all() {
    RpuExecutionCleanupGuard cleanup("spm_alloc_reset_all");
    SPM_ALLOC.reset_all();
}

// RPU→CPU zero-copy 在 graph RECORDING 期需要先 sync。
// rpu_to_cpu_zerocopy 返回的是共享 DDR view,绕过任何等待/同步；Python adapter
// 在 graph
// RECORDING 期调 output.to(cpu) 会拿到 stale DDR (kernels 还在 nodes_ 没跑)。
// Pi0.5 denoise loop 内 action_out_proj(stale) + Euler `x_t += dt*v_t`
// 累积放大成 0.6 量级偏差。
//
// 修法:RECORDING 期且有 pending nodes 时,sync_point() 把已录前缀 oneshot 落
// 地 + 清 nodes_/kernels_ + 保持 RECORDING 状态。`mark_non_replayable` 是
// sync_point() 内部副作用 (graph 必然 non-replayable 才合理 break-then-zerocopy)。
//
// 返回值告诉调用方"我刚 sync 过"用于诊断 trace。
//
// 不直接 include graph_runtime.h 进 rpu_backend.cpp 的理由跟 spm_reset_guard
// 头部一致 (c10::StorageImpl inline undefined symbol)。
bool graph_aware_rpu_to_cpu_zerocopy_sync() {
    auto& g = RpuKernelGraph::active();
    if (g.state() != RpuKernelGraph::State::RECORDING) {
        return false;
    }
    if (g.graph_size() == 0) {
        return false;
    }
    static thread_local size_t sync_count = 0;
    ++sync_count;
    if (sync_count <= 5 && log_at(4)) {
        std::cerr << "[RpuKernelGraph] sync_point triggered by RPU→CPU "
                     "zero-copy during RECORDING (call #" << sync_count
                  << ", graph_size=" << g.graph_size() << ")\n";
    }
    g.sync_point();
    return true;
}
