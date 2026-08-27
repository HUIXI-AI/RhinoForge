// custom_cpu_fallback_dispatch.cpp — graph-aware CPU fallback dispatcher
//
// Graph-state dispatch contract:
//   PASSTHROUGH / BUILT: run_cpu_fallback_zerocopy(alias=has_out_args),原 eager 行为。
//   RECORDING:
//     classify_host_op(op) → Tier:
//       Reject  → g.sync_point() + zerocopy(alias=false) (老 sync_point 兼容)
//       Tier1   → HostCallbackCapture(tier=Tier1) RAII + zerocopy(alias=true)  + adopt_returns
//       Tier2   → HostCallbackCapture(tier=Tier2) RAII + zerocopy(alias=false) + adopt_returns
//       Tier3   → Tier3OneshotCapture RAII + zerocopy(alias=false) + adopt_returns
//   REPLAYING:
//     Tier1/2  → g.capture_host_callback_replay_args(op, *stack)   (不真跑 CPU op)
//     Tier3    → g.capture_tier3_oneshot_replay_args(op, *stack)
//     Reject 在 REPLAYING 不应出现(RECORDING-Reject 已经 sync_point 切 PASSTHROUGH)
//
// 为什么单独放 src/graph/:
//   rpu_backend.cpp 不 include graph_runtime.h,避免 c10::StorageImpl::decref_pyobject
//   等 inline body 在 rpu_backend.cpp 这个 TU 实例化触发 undefined symbol（构建与运行
//   使用不同 torch 库时会出现）。graph/
//   目录下文件已经 include 这些 header 走过链路,把 dispatch fork 放这里 link 没问题。
//
// RECORDING routes CPU fallback through HostCallback nodes so downstream RPU
// operations observe current DDR contents and the same dataflow is preserved
// across RECORDING and REPLAYING.

#include "graph/graph_infra.h"     // HostCallbackTier / classify_host_op / collect_output_arg_indices / run_cpu_fallback_zerocopy
#include "graph/graph_runtime.h"   // RpuKernelGraph / HostCallbackCapture / Tier3OneshotCapture

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <c10/util/Exception.h>

void dispatch_graph_aware_cpu_fallback(const c10::OperatorHandle& op,
                                       torch::jit::Stack* stack) {
    auto& g = RpuKernelGraph::active();
    const auto state = g.state();

    // PASSTHROUGH / BUILT: eager zero-copy (原行为)。alias=true 当 schema 有 .out=
    // writable args 时启用,避免对 zero-copy CPU return view 做 .to(rpu) 触发
    // storage allocator=nullptr heap corruption（详见 run_cpu_fallback_zerocopy
    // 注释）。BUILT 态实际上 begin() 已经退出
    // graph scope,active() 返回 fallback_graph_,state=PASSTHROUGH;留这条分支是
    // defensive(假如调用方在 fallback_graph_ 上 explicit 把 state 设成 BUILT)。
    if (state == RpuKernelGraph::State::PASSTHROUGH ||
        state == RpuKernelGraph::State::BUILT) {
        const bool has_out_args =
            !collect_output_arg_indices(op.schema()).empty();
        run_cpu_fallback_zerocopy(op, stack, /*alias_output_args=*/has_out_args);
        return;
    }

    // RECORDING: Tier 分类 + 三档分派
    if (state == RpuKernelGraph::State::RECORDING) {
        HostCallbackTier tier = classify_host_op(op);
        if (tier == HostCallbackTier::Reject) {
            // 无法建模 (scalar return / in-place self mutation / view-only):走
            // sync_point 老路径打断 graph → state→PASSTHROUGH + replayable=false。
            g.sync_point();
            run_cpu_fallback_zerocopy(op, stack, /*alias_output_args=*/false);
            return;
        }
        if (tier == HostCallbackTier::Tier1 ||
            tier == HostCallbackTier::Tier2) {
            // Tier1 (.out variant): alias=true,CPU op 通过 zero-copy view 写穿到
            // 原 RPU .out= input tensor 的 storage,return slot 替换回原 tensor,
            // 不走 .to(rpu) 避免 heap corruption。adopt_returns 把数据 copy 到
            // graph-owned stable buffer 保持跨 replay dev_addr 稳定。
            // Tier2 (functional): alias=false,CPU op 自己 alloc fresh CPU tensor,
            // .to(rpu) 安全;adopt_returns 同样 stable buffer 化。
            HostCallbackCapture guard(g, op, *stack, tier);
            const bool alias = (tier == HostCallbackTier::Tier1);
            run_cpu_fallback_zerocopy(op, stack, /*alias_output_args=*/alias);
            guard.adopt_returns(*stack);
        } else {  // Tier3 — RNG / data-dependent shape:节点级 oneshot
            // RECORDING capture 期真跑给当前 forward 下游用，并 adopt 到
            // transient buffer；
            // execute_graph_* 时在 segment 边界二次跑保证 dev_addr 内容是当前
            // step 真值。REPLAYING 期由 capture_tier3_oneshot_replay_args 推
            // cursor + 合成 transient return。
            Tier3OneshotCapture t3guard(g, op, *stack);
            run_cpu_fallback_zerocopy(op, stack, /*alias_output_args=*/false);
            t3guard.adopt_returns(*stack);
        }
        return;
    }

    // REPLAYING: 不真跑 CPU op(executor 在 end()/segment 边界二跑),只 push live
    // tensor args 到对应 node 的 replay_args 表,让 cursor 推进 + admission live
    // 表登记本步 fresh tensor 地址。Tier 分类必须跟 RECORDING 一致 (Reject 在
    // RECORDING 已 sync_point 切 PASSTHROUGH,这里看不到)。
    if (state == RpuKernelGraph::State::REPLAYING) {
        HostCallbackTier tier = classify_host_op(op);
        TORCH_CHECK(tier != HostCallbackTier::Reject,
                    "dispatch_graph_aware_cpu_fallback: Reject-tier op '",
                    op.schema().name(),
                    "' in REPLAYING state; this should not occur (RECORDING-"
                    "Reject marks scope non-replayable via sync_point and ends "
                    "as PASSTHROUGH, never REPLAYING). Inspect graph state "
                    "machine + cache admission.");
        if (tier == HostCallbackTier::Tier3) {
            g.capture_tier3_oneshot_replay_args(op, *stack);
        } else {
            g.capture_host_callback_replay_args(op, *stack);
        }
        return;
    }

    TORCH_CHECK(false,
                "dispatch_graph_aware_cpu_fallback: invalid graph state ",
                static_cast<int>(state), " op=", op.schema().name());
}
