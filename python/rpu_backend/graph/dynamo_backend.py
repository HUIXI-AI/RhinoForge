"""
RPU TorchDynamo backend — torch.compile 入口包装。

用法 (兼容性入口;RPU 上不支持生产使用 ``torch.compile``):
    from rpu_backend.graph.dynamo_backend import RpuDynamoBackend, dynamo_backend
    compiled = torch.compile(model, backend=dynamo_backend)

    # 多个 compile 调用共享 cache:
    be = RpuDynamoBackend()
    m1 = torch.compile(m1, backend=be)
    m2 = torch.compile(m2, backend=be)

工作机制 (per Dynamo-traced FX gm):
    第一次调用:
      planner.make_plan(id(gm), args) → registry.admit_for_call(plan)
      → cache.get_or_create(sig) (registry 内部) → graph.begin(sig)
      → state=RECORDING → gm(*args) (op stream 真跑, push 节点 + capture_data)
      → graph.end() → BUILT → registry.touch_after_call(plan)
    后续同 sig 调用:
      → registry.admit_for_call (LRU 命中,直接返回 graph)
      → graph.begin(sig) → state=REPLAYING (admission 命中)
      → gm(*args) (op stream 重跑, set_regs 自动覆本轮 dev_addr)
      → graph.end() → 复用 prepared_wq launch_prepared_batch

dynamic shape 漂移由 Dynamo guard 在 frontend 处理:guard 失败 → 重 trace
新 gm → 新 id → 新 sig → 新 cache entry。

LRU eviction (max_entries):
    decode loop 里 transformers 内部把 kv_cache.position 当 Python int 读,
    Dynamo 把它纳入 guard,position 每涨
    1 → guard 失败 → retrace 出新 gm → 新 sig → 每步新 entry。每个 BUILT
    graph 持有一份 graph-managed SPM(record-time defer 到 graph 销毁才
    释放)。多 entry 累积下 SPM 池被耗尽,后续 graph acquire 失败 → 错地址
    → KV 写错 → token 错。

    backend 持一个 SignaturePlanner(签名构造)+ EntryRegistry(单 LRU +
    replay/recapture/evict 计数)。`compiled_call` 每步走
    `planner.make_plan(...)` → `registry.admit_for_call(plan)` → graph
    `begin/end` → `registry.touch_after_call(...)`。命中 LRU 容量上限时
    registry 内部 evict 最老 entry → 触发 backend Graph 析构 → SPM 立即释放。

    默认 max_entries=1:每步新 sig 即刻 evict 上一步,SPM 持有量 = 1 份。
    如果同一段会反复在某些 sig 之间切换(比如 prefill_sig + decode_sig 两个
    长期共存),把 max_entries 调到 2-4 让两端都能 replay。

    不要把 max_entries 设成无界;Python-int guard
    根治需要 RPUCache.get_seq_length 返回 0-d tensor 供 Dynamo 作为
    SymInt 处理；当前实现仍需保持 max_entries 有界。


``torch.compile`` on RPU is unsupported because adapters already own graph
capture and nested capture is unsupported.

The backend is intentionally not exported while nested capture remains
unsupported.

The lazy-init guard remains available from graph/lazy_init_guard.py because
every patch_*_for_rpu() uses it on the ordinary eager path.
"""

from typing import Callable, List, Optional
import torch

from .admission import SignaturePlanner
from .admission import EntryRegistry

import hashlib
import logging

_log = logging.getLogger(__name__)


class RpuDynamoBackend:
    """torch.compile 的 backend callable。

    持一个 GraphCache + LRU 队列横跨所有 gm。每个 (gm, arg-sig) 一个 cache
    entry,同 sig 命中走 REPLAYING。LRU 容量满时 evict 最老的 entry,触发
    backend Graph 析构 → graph-managed SPM 立刻释放。

    Args:
        cache: 自定义外部 GraphCache(默认新建一个);多 backend 共享时传入。
               注意:外部 cache 的 max_entries (C++ 层硬限制) 必须 >= LRU
               max_entries,否则 cache.get_or_create 会先于 LRU evict 抛错
        max_entries: LRU 软上限。默认 1 适合每步新 sig 的 decode loop
                     (KV cache position guard 触发的多 gm 场景);若有几个
                     sig 长期共存(prefill+decode 两个 gm),调到 2-4
        debug: True 时每编译一个 gm 打印一行(node 数 + id)
        branch_key_fn: 可选 pure function `(gm, args) -> branch value`。
                       返回 None/bool/int/float/str/tuple/list,backend 稳定
                       hash 到 GraphSignature.branch_key。默认 None 保持旧行为。
    """

    DEFAULT_MAX_ENTRIES = 1

    def __init__(self, cache: Optional["torch.rpu.GraphCache"] = None,
                 *, max_entries: Optional[int] = None,
                 debug: bool = False,
                 branch_key_fn: Optional[Callable] = None):
        self._cache = cache if cache is not None else torch.rpu.GraphCache()
        self._max_entries = (max_entries if max_entries is not None
                              else self.DEFAULT_MAX_ENTRIES)
        self._debug = debug
        self._branch_key_fn = branch_key_fn
        # SignaturePlanner constructs signatures + plans; EntryRegistry
        # holds the cache's raw C++ impl and owns one frontend LRU.
        self._planner = SignaturePlanner()
        self._registry = EntryRegistry(
            self._cache._impl,
            max_entries=self._max_entries,
            planner=self._planner,
        )
        # Recompile sentinel.
        # key = (graph_text_hash, input_shapes, input_dtypes);value = list of gm_id.
        # 同 key 看到 ≥2 个 gm_id → log warning catch-all (lazy attr / training
        # check / Python control flow on tensor 等任何 Dynamo guard hazard).
        self._gm_id_history: dict = {}

    @property
    def cache(self) -> "torch.rpu.GraphCache":
        return self._cache

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def evict_count(self) -> int:
        """累计 evict 次数(LRU 冲队头)。诊断 SPM 复用率用。

        Sums across the prefill, decode, and generic pools owned by
        ``EntryRegistry`` and returns one aggregate integer.
        """
        return self._registry.evict_total()

    def __call__(self, gm: torch.fx.GraphModule,
                 example_inputs: List[torch.Tensor]) -> Callable:
        gm_id = id(gm)

        # Preflight on gm-reachable modules.
        from .lazy_init_guard import _verify_lazy_init_best_effort
        _verify_lazy_init_best_effort(gm, example_inputs)

        # Check for recompilation of an equivalent graph.
        self._sentinel_check(gm, example_inputs)

        if self._debug:
            n = len(list(gm.graph.nodes))
            print(f"[rpu_dynamo_backend] compile gm id={gm_id:x} nodes={n} "
                  f"max_entries={self._max_entries}", flush=True)

        # Materialization break partitioner:
        # Dynamo 把 aten.clone / aten._to_copy (cross-device) / aten.copy_
        # trace 进 FX gm,RECORDING 期 deferred 执行让这些 host-read 节点
        # 读到 empty buffer → silent wrong output。Partition gm 把这些节点
        # 切成 single-node eager partitions,周围 RPU 段各自 g.begin/end。
        from .dynamo_partitioner import (
            partition_gm, materialization_breaks_enabled,
        )
        partitioned_gm = gm
        partition_info = [("__root__", True)]
        if materialization_breaks_enabled():
            partitioned_gm, partition_info = partition_gm(gm)
            if self._debug and len(partition_info) > 1:
                breakdown = ",".join(
                    f"{name}({'RPU' if rpu else 'eager'})"
                    for name, rpu in partition_info
                )
                print(f"[rpu_dynamo_backend] partition: {breakdown}",
                      flush=True)

        planner = self._planner
        registry = self._registry
        branch_key_fn = self._branch_key_fn

        # Single-graph fast path:gm 内没有 materialization,直接走原逻辑
        if len(partition_info) == 1 and partition_info[0][1]:
            def compiled_call_single(*args):
                branch_key = 0
                if branch_key_fn is not None:
                    branch_key = SignaturePlanner.stable_branch_key(
                        branch_key_fn(gm, args))
                plan = planner.make_plan(
                    gm_id, args, branch_key=branch_key)
                g = registry.admit_for_call(plan)
                g.begin(plan.signature._impl)
                enter_state = g.state()
                try:
                    out = gm(*args)
                except Exception:
                    g.abort()
                    raise
                g.end()
                registry.touch_after_call(plan, in_scope_state=enter_state)
                return out
            return compiled_call_single

        # Partitioned path:partitioned_gm 顶层 forward 按拓扑序调子模块。
        # 我们替换每个 RPU 子模块的 forward 走 backend g.begin/end RAII;
        # eager 子模块(单 materialization 节点)保持原样。每个 RPU 子模块
        # 用独立的 gm_id (复合自父 gm_id + 子模块名) 进 planner,这样
        # admission cache 每段独立 LRU entry。
        for sub_name, is_rpu in partition_info:
            if not is_rpu:
                continue
            sub_mod = partitioned_gm.get_submodule(sub_name)
            # sub_id = (gm_id << 16) | hash(sub_name) — **跟 gm_id 关联**,每次
            # Dynamo 新 trace gm_id 不同 → sub_id 不同 → admit_for_call 总走
            # 新 entry,跨 call replay 永不命中。跨 trace replay 要求 sub_id
            # 稳定,而当前 sub_id 取决于 Dynamo trace 和 sub_mod 标识。
            sub_id = (gm_id << 16) | (hash(sub_name) & 0xFFFF)
            self._install_rpu_wrap(
                partitioned_gm, sub_name, sub_mod, sub_id,
                planner, registry, branch_key_fn,
            )

        def compiled_call_partitioned(*args):
            # partitioned_gm 顶层 forward 内部依次调 submod_X (.forward);
            # 已经 in-place 替换 RPU 子模块的 forward 为 backend-wrapped 版本。
            return partitioned_gm(*args)

        return compiled_call_partitioned

    def _install_rpu_wrap(self, parent_gm, sub_name, sub_mod, sub_id,
                          planner, registry, branch_key_fn):
        """In-place wrap 给定 RPU 子模块的 forward 为 backend g.begin/end RAII。
        通过 setattr 替换 sub_mod.forward,parent_gm 顶层 forward 调
        sub_mod(*args) 时实际走 wrapped 版本。"""
        original_forward = sub_mod.forward

        def wrapped_forward(*sub_args, **sub_kwargs):
            sub_args_seq = sub_args + tuple(sub_kwargs.values())
            branch_key = 0
            if branch_key_fn is not None:
                branch_key = SignaturePlanner.stable_branch_key(
                    branch_key_fn(sub_mod, sub_args_seq))
            plan = planner.make_plan(
                sub_id, sub_args_seq, branch_key=branch_key)
            g = registry.admit_for_call(plan)
            g.begin(plan.signature._impl)
            enter_state = g.state()
            try:
                out = original_forward(*sub_args, **sub_kwargs)
            except Exception:
                g.abort()
                raise
            g.end()
            registry.touch_after_call(plan, in_scope_state=enter_state)
            return out

        sub_mod.forward = wrapped_forward

    # ---------------------------------------------------------------- #
    # Recompile sentinel
    # ---------------------------------------------------------------- #
    def _sentinel_key(self, gm, example_inputs):
        """Stable key: (graph_text_hash, shapes, dtypes).

        `gm.graph.python_code('self').src` 是同 callable + 同 shape 下稳定的
        FX trace 文本表示,不依赖 Dynamo internal state field(`compile_subgraph_reason`
        之类不保证存在 / 不稳定)。
        """
        try:
            graph_src = gm.graph.python_code('self').src
            graph_hash = hashlib.md5(
                graph_src.encode(), usedforsecurity=False).hexdigest()[:16]
        except Exception:
            graph_hash = "unknown"
        shapes = tuple(
            tuple(t.shape) for t in example_inputs if hasattr(t, 'shape')
        )
        dtypes = tuple(
            str(t.dtype) for t in example_inputs if hasattr(t, 'dtype')
        )
        return (graph_hash, shapes, dtypes)

    def _sentinel_check(self, gm, example_inputs):
        """Track gm_id history per (graph_hash, shape, dtype). Warn on ≥2."""
        key = self._sentinel_key(gm, example_inputs)
        hist = self._gm_id_history.setdefault(key, [])
        hist.append(id(gm))
        if len(hist) > 1:
            _log.warning(
                "[RpuDynamoBackend] recompile detected for "
                "shape=%s dtype=%s\n"
                "  gm_id history: %s\n"
                "  Likely Dynamo guard hazard. "
                "Run with TORCH_LOGS=recompiles to diagnose.\n"
                "  Common causes: lazy attr init, self.training check, "
                "Python control flow on tensor value.",
                key[1], key[2], hist,
            )


# 默认全局 singleton。简单用法:
#   torch.compile(model, backend=rpu_backend.dynamo_backend)
dynamo_backend = RpuDynamoBackend()


# =============================================================================
# rpu_backend.compile() — 用户面 API,封掉 cache_position 实现细节
# =============================================================================
# Decode 每步进入 forward 时会调 past_key_values.get_seq_length() 拿
# Python int,Dynamo 把这个 int 烘进
# frame guard。compile wrapper 在 traced region 外把它转成 cache_position
# tensor 显式传入,upstream forward 看到 cache_position 非 None 跳过自己
# arange,Dynamo 看到的就只是 tensor 输入,不再 specialize 出 step-varying
# Python int guard。配合 op schema 的 SymInt position 和 dynamic=True,
# decode 全程共享同一 sig,LRU 不动,replay 持续命中。

class _RpuCompiledModule:
    """torch.compile 包装器:在 compiled region 外补齐 cache_position。

    Why:
        decode loop 步进时 RPUCache.position(Python int)在 forward 内被读取,
        Dynamo 把它烘进 frame-local guard,每步 retrace 出新 gm。这里在 compile
        外把它转成 cache_position tensor —— 从 caller 视角 Python int 始终留
        在 compiled region 外,Dynamo trace 看到的是 tensor 而非常量 int。
        Qwen3 forward 看 cache_position 非 None 时跳过内部 arange,从而不再调
        get_seq_length() 拿 int。

    Note:
        本 wrapper 不暴露给用户写 cache_position。用户继续
        按 model.model(input_ids=, past_key_values=, use_cache=True)调,wrapper
        透明补齐。
    """

    def __init__(self, model, compiled_callable, backend=None):
        # 直接走 __dict__ 不经 __getattr__,避免 init 期递归
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_compiled", compiled_callable)
        # 持 backend reference 供 stats() 使用。compile() 总会注入,这里默认
        # None 是为兼容直接构造的情况。
        object.__setattr__(self, "_backend", backend)

    def __call__(self, *args, **kwargs):
        # input_ids 支持 positional 或 keyword
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        past_key_values = kwargs.get("past_key_values")
        if input_ids is not None and hasattr(input_ids, "shape"):
            input_seq_len = (
                int(input_ids.shape[1]) if len(input_ids.shape) >= 2 else 1
            )
        else:
            input_seq_len = 1

        # 仅在用户没显式传 cache_position、且 cache 暴露 .position 时补齐。
        # 失败回退原行为(不补),让上游 forward 自己读 get_seq_length() —— 旧
        # 行为保持兼容,只是不享受 SymInt 路径的 replay 收益。
        if (kwargs.get("cache_position") is None
                and past_key_values is not None
                and input_ids is not None
                and hasattr(past_key_values, "position")):
            # Python int 在 compiled region 外。下方 torch.arange 是 host op,
            # 但产物 cache_position 作为 tensor 进 compiled region。
            pos = int(past_key_values.position)
            seq_len = input_ids.shape[1]
            kwargs["cache_position"] = torch.arange(
                pos, pos + seq_len,
                device=input_ids.device, dtype=torch.long,
            )

        # The compile path is a decode graph path. Full prefill capture is still
        # handled by the dedicated graph-prefill path; recording the whole
        # model prefill through TorchDynamo currently captures unsupported data
        # movement and can corrupt the first token. Keep prefill eager and let
        # single-token decode use the compiled GraphCache path.
        if past_key_values is not None and input_seq_len > 1:
            model = object.__getattribute__(self, "_model")
            return model(*args, **kwargs)

        return self._compiled(*args, **kwargs)

    def stats(self) -> dict:
        """诊断输出:GraphCache 容量 + 累计 replay/recapture/evict 计数。

        decode loop 后调一次,健康指标:
          - size 接近 1 (decode 单 sig)
          - replay_total 随 step 递增
          - recapture_total / evict_count 保持 0 (Python int guard 已修)
          - prepared_segment_miss_total 在稳定 decode 上保持 0
            (>0 说明某步 prepared_wq 被丢失)
          - non_replayable_reasons 列出每个 entry 的 reason(空字符串过滤掉)

        entries 字段给出每个 cache entry 的 graph 计数明细
        (kernel/segment/data_node/boundary_flush/prepared hit-miss/HostCallback
        tier 分桶/Tier3Oneshot),用来在 benchmark 里直接判断"这次跑是稳定
        replay 还是反复 recapture"。
        """
        backend = object.__getattribute__(self, "_backend")
        if backend is None:
            return {}
        snap = backend.cache.snapshot()
        entries = []
        for e in snap:
            entries.append({
                "kernel_count":                 e.kernel_count,
                "segment_count":                e.segment_count,
                "data_node_count":              e.data_node_count,
                "boundary_flush_count":         e.boundary_flush_count,
                "prepared_segment_hit_total":   e.prepared_segment_hit_total,
                "prepared_segment_miss_total":  e.prepared_segment_miss_total,
                "replay_count":                 e.replay_count,
                "recapture_count":              e.recapture_count,
                "host_callback_tier1_count":    e.host_callback_tier1_count,
                "host_callback_tier2_count":    e.host_callback_tier2_count,
                "host_callback_tier3_count":    e.host_callback_tier3_count,
                "tier3_oneshot_count":          e.tier3_oneshot_count,
                "register_census_present":      e.register_census_present,
                "register_census_build_committed":
                    e.register_census_build_committed,
                "register_census_policy_fingerprint":
                    e.register_census_policy_fingerprint,
                "register_census_policy_digest":
                    e.register_census_policy_digest,
                "register_census_plan_hash":    e.register_census_plan_hash,
                "register_census_live_epoch":   e.register_census_live_epoch,
                "register_census_node_count":   e.register_census_node_count,
                "register_census_dma_count":    e.register_census_dma_count,
                "register_census_barrier_count":
                    e.register_census_barrier_count,
                "register_census_kernel_count": e.register_census_kernel_count,
                "register_census_typed_kernel_count":
                    e.register_census_typed_kernel_count,
                "register_census_no_ddr_kernel_count":
                    e.register_census_no_ddr_kernel_count,
                "register_census_operand_count":
                    e.register_census_operand_count,
                "register_census_weight_operand_count":
                    e.register_census_weight_operand_count,
                "register_census_key_cache_operand_count":
                    e.register_census_key_cache_operand_count,
                "register_census_value_cache_operand_count":
                    e.register_census_value_cache_operand_count,
                "register_census_semantic_input_operand_count":
                    e.register_census_semantic_input_operand_count,
                "register_census_ordered_node_kind_hash":
                    e.register_census_ordered_node_kind_hash,
                "register_census_ordered_kernel_hash":
                    e.register_census_ordered_kernel_hash,
                "non_replayable_reason":        e.non_replayable_reason,
            })
        non_replayable_reasons = [
            e.non_replayable_reason for e in snap if e.non_replayable_reason
        ]
        return {
            "size": backend.cache.size(),
            "replay_total":                  sum(e.replay_count for e in snap),
            "recapture_total":               sum(e.recapture_count for e in snap),
            "evict_count":                   backend.evict_count,
            "prepared_segment_hit_total":    sum(
                e.prepared_segment_hit_total for e in snap),
            "prepared_segment_miss_total":   sum(
                e.prepared_segment_miss_total for e in snap),
            "non_replayable_reasons":        non_replayable_reasons,
            "entries":                       entries,
        }

    # 属性透传(model.config / model.lm_head 等)。__init__ 完成后才会触发,
    # 此时 _model 已就位。
    def __getattr__(self, name):
        try:
            model = object.__getattribute__(self, "_model")
        except AttributeError:
            raise AttributeError(name)
        return getattr(model, name)


def compile(model, **compile_kwargs):
    """rpu_backend.compile(model) — 用户面 API,一行搞定。

    封装:
      torch.compile(model, backend=RpuDynamoBackend(...), dynamic=True, ...)
      + cache_position 自动补齐(Python int 留 compiled region 外)。

    用户不用写 backend / max_entries / dynamic / LRU —— 内部默认值已经对
    decode loop 调好。需要 stats 调 returned wrapper 的 .stats()。

    Args:
        model: nn.Module 或 callable。建议传 model.model(backbone),不要传带
               lm_head 的整个 ForCausalLM —— argmax/sampling 是 host op,留
               lm_head 外面跑省一次 graph break。
        **compile_kwargs: 透传给 torch.compile。三个特殊 key:
            - backend=RpuDynamoBackend(...) 自定义 backend 实例（需要控制
              max_entries 或共享 cache 时传）。不传则内部新建默认实例。
            - dynamic=True / fullgraph=False  默认值,如显式传则尊重用户值。
            其它 (mode= / options= 等) 透传给 torch.compile。

    Returns:
        _RpuCompiledModule 实例。callable + 透传 model 属性 + .stats()。
    """
    # 用户传 backend= 走自定义路径（跨多次 compile 共享 cache 或调整
    # max_entries）；否则内部新建默认 backend。
    backend = compile_kwargs.pop("backend", None)
    if backend is None:
        backend = RpuDynamoBackend(max_entries=4)
    # 默认值:dynamic=True(SymInt position 不被 specialize)/
    #         fullgraph=False(argmax/topk 等 host op 自然切段)
    compile_kwargs.setdefault("dynamic", True)
    compile_kwargs.setdefault("fullgraph", False)
    inner = torch.compile(model, backend=backend, **compile_kwargs)
    return _RpuCompiledModule(model, inner, backend=backend)
