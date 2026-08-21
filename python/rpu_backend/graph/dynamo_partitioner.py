"""FX GraphModule partitioner — split materialization ops out of RPU sub-graphs.

Background
==========
RpuKernelGraph RECORDING 是 deferred execution:`g.begin → gm(*args) → g.end`
内部 kernel push 进 `nodes_`,真 launch 在 `g.end()`。但 Dynamo 把
`aten.clone / aten._to_copy / aten.copy_` 这类 host-read / cross-device
materialization 节点 trace 进 FX gm,RECORDING 期跑时 source data 还没填,
materialization 出空 buffer。

Solution
========
在 backend 收到 FX gm 之后 partition:
  - RPU-safe nodes → 各自 sub-GraphModule,backend g.begin/end 包,正常 capture/replay
  - Materialization nodes → 独立 single-node sub-GraphModule,**保持 eager 跑**

  Result:gm 切成 [rpu_seg_0, mat_0, rpu_seg_1, mat_1, ...] chain。Eager 段
  在前一 RPU 段 g.end() 之后跑,data 已 flush,materialization 拿到正确数据。

Materialization detection
=========================
按保守先切的原则:
  - `aten.clone.default`        — 总是切 (clone 出来跟 src 同 device 但
                                       Python 引用 lifetime 不确定)
  - `aten._to_copy.default`     — 当且仅当 device 改变 (跨 device 必 break)
  - `aten.copy_.default`        — 总是切 (in-place,跨 RECORDING 边界语义复杂)

``torch.compile`` on RPU is unsupported
because adapters already own graph capture and nested capture is unsupported.

The lazy-init guard remains available from graph/lazy_init_guard.py because
every patch_*_for_rpu() uses it on the ordinary eager path.
"""
from __future__ import annotations
from typing import Callable, List, Tuple

import torch
import torch.fx
from torch.fx.passes.split_module import split_module
from rpu_backend.runtime import rpu_env_bool


_MATERIALIZATION_TARGETS = frozenset({
    torch.ops.aten.clone.default,
    torch.ops.aten.copy_.default,
})


def _is_to_copy_cross_device(node: torch.fx.Node) -> bool:
    """aten._to_copy is materialization only when device changes
    (or as conservative fallback, any kwarg that triggers a real copy)."""
    if node.target is not torch.ops.aten._to_copy.default:
        return False
    # kwargs may include device / dtype / layout / memory_format
    device_kw = node.kwargs.get("device")
    # If device is changing (e.g. rpu → cpu), this is a cross-device
    # materialization and must be broken out of the RPU graph.
    if device_kw is not None:
        return True
    # If only dtype changes but device stays RPU,_to_copy is still a real
    # copy but the output lives on RPU — can stay inside RPU graph deferred.
    # 保守起见:如果有 dtype 改变,先切(防 Python lifecycle 不可见的 sync)。
    if "dtype" in node.kwargs and node.kwargs["dtype"] is not None:
        return True
    return False


# Dynamo 在早期 FX trace 阶段,把 tensor 上的 method 调用保留成 call_method 节点
# (target 是字符串方法名),不展开成 aten op。例如 `t.clone()` → call_method "clone",
# `t.cpu()` → call_method "cpu"。aten op 形态只在 AOT decomp / Inductor 等下游
# pass 后才出现。Backend 接到的 gm 是 Dynamo 早期形态,所以要 cover call_method。
_MATERIALIZATION_METHOD_NAMES = frozenset({
    "clone",
    "cpu",      # tensor.cpu()
    "numpy",    # tensor.numpy() — 通常已 graph-break,防御性
    "tolist",   # tensor.tolist() — 同上
    "detach",   # detach 跟 graph break 没直接关系,但跨 Python lifetime 复杂,
                #       保守切
    "copy_",    # in-place copy
})


def _is_to_method_cross_device(node: torch.fx.Node) -> bool:
    """`tensor.to('cpu')` / `tensor.to(dtype)` 形态 — call_method target='to'。
    args[0] 是 source tensor (Node),args[1:] / kwargs 含 device/dtype literal。"""
    if node.op != "call_method" or node.target != "to":
        return False
    # args 中除 source 外,literal 包含 device 或 dtype
    for arg in list(node.args[1:]) + list(node.kwargs.values()):
        if isinstance(arg, torch.device):
            return True
        if isinstance(arg, str) and arg in ("cpu", "rpu", "cuda", "meta"):
            return True
        if isinstance(arg, torch.dtype):
            return True
    return False


def is_materialization_node(node: torch.fx.Node) -> bool:
    """True if this FX node must be broken out of an RPU sub-graph."""
    if node.op == "call_function":
        if node.target in _MATERIALIZATION_TARGETS:
            return True
        if _is_to_copy_cross_device(node):
            return True
    if node.op == "call_method":
        if node.target in _MATERIALIZATION_METHOD_NAMES:
            return True
        if _is_to_method_cross_device(node):
            return True
    return False


def find_materialization_nodes(gm: torch.fx.GraphModule) -> List[torch.fx.Node]:
    return [n for n in gm.graph.nodes if is_materialization_node(n)]


def _make_partition_callback(
    materialization_nodes: List[torch.fx.Node],
) -> Callable[[torch.fx.Node], int]:
    """Assign each FX node a partition id:
        - RPU runs of non-materialization nodes share the same id (e.g. 0)
        - Each materialization node gets its own unique id
        - The next RPU run gets a new id (so adjacent rpu/eager/rpu segments
          are separated even if eager partition has only one node)
    """
    mat_set = {id(n) for n in materialization_nodes}
    state = {"current_rpu": 0, "next_avail": 0}

    def callback(node: torch.fx.Node) -> int:
        if id(node) in mat_set:
            state["next_avail"] += 1
            mat_id = state["next_avail"]
            state["next_avail"] += 1
            state["current_rpu"] = state["next_avail"]
            return mat_id
        return state["current_rpu"]

    return callback


def partition_gm(
    gm: torch.fx.GraphModule,
) -> Tuple[torch.fx.GraphModule, List[Tuple[str, bool]]]:
    """Split gm into sub-modules; return (split_gm, [(submod_name, is_rpu), ...]).

    `is_rpu=True` means the sub-module contains no materialization nodes and
    should be wrapped with backend.g.begin/end RAII. `is_rpu=False` means the
    sub-module is a single materialization node that must stay eager.

    If no materialization is found, returns (gm, [("__root__", True)]) signaling
    the caller to use the single-graph fast path unchanged.
    """
    mats = find_materialization_nodes(gm)
    if not mats:
        return gm, [("__root__", True)]

    callback = _make_partition_callback(mats)
    # split_module assigns each unique partition id to a sub-module.
    # partition_affix lets us namespace sub-module names.
    split_gm = split_module(
        gm,
        root_m=gm,
        split_callback=callback,
        keep_original_order=True,
    )

    # Walk split_gm.graph in topological order to find submodule call order
    # and infer whether each is RPU-safe (no materialization inside) or a
    # materialization step (single node).
    info: List[Tuple[str, bool]] = []
    for node in split_gm.graph.nodes:
        if node.op != "call_module":
            continue
        sub = split_gm.get_submodule(node.target)
        sub_mats = find_materialization_nodes(sub)
        is_rpu = not sub_mats
        info.append((node.target, is_rpu))

    return split_gm, info


def materialization_breaks_enabled() -> bool:
    """Env gate. **Default OFF** — experimental opt-in only.

    Set RPU_DYNAMO_MATERIALIZE_BREAKS=1 to enable partitioner. Default off
    because the partitioner is currently experimental: a single Dynamo call
    can match direct capture, but a second call across Dynamo traces may be
    numerically incorrect and miss replay. Enabling it by default could affect
    user workloads,
    必须用户显式 opt-in 才生效。"""
    return rpu_env_bool("RPU_DYNAMO_MATERIALIZE_BREAKS")
