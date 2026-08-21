"""HALO action expert 单步驱动（HaloStepRunner）：handle 装载 + prefix KV 灌入 + step。

执行模型（一次 act step）：
  host (fp32) ──组装 x_emb──> RPU fp16 ──halo_action_expert_step_forward──> hidden
  └ in_proj(x_t)+temb+sinpos → 行 1..16；embed_tokens(text_ids) → 行 0/17；
    fp32 → bf16 → fp16（对齐 HALO CUDA bf16 流水的舍入语义）。
  hidden 行 1..16 → host fp32 out_proj → v_t [16,14]。

prefix KV 灌入：CUDA NaiveCache 捕获的 prefix K/V [4706,2,128] bf16（已 RoPE）
直接 host 侧 swizzle 进 7-D RPUCache（slot=head：head 0→chunk 槽 0，head 1→槽 1，
其余 6 个 replica 槽留 0）。该 slot 映射来源于 attn_tp=2 时
``rpu_launch_insert_kcache_spm_unified`` "core c 写槽 c" 的内核行为，与 SDPA
读侧使用相同的 slot/head 映射，因此 prefix 注入与 KV-insert ABI 一致。

图缓存：per-model ``GraphCache``（pi05/wall_oss 先例），每步只在
``capture(GraphSignature)`` 内调一次 ``torch.ops.rpu.halo_action_expert_step_forward``
（step0 BUILD，后续 REPLAY）。GraphSignature 把 (chunk, hidden, layers,
prefix, rope_position) 全部编进 key，prefix/位置改变即重建。

所有权：runner 持有 text/act 权重、cos/sin、掩码与 RPUCache，与 handle 等寿命
（C++ 仅存引用计数副本）。handle 经 ``weakref.finalize`` 回收。
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import weakref
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

import rpu_backend
from rpu_backend.api.cache import RPUCache
from rpu_backend.adapters.halo.weights import (
    HaloBranchWeights,
    HaloConfig,
    HaloHostWeights,
    build_rope_tables,
    build_text_row_mask,
    load_halo_branches,
    sinusoidal_pos_embedding,
    timestep_embedding,
)

_S_CHUNK = 16          # 7-D cache seq / head_dim 块宽（api/cache.py 同源常量）
_KV_SLOTS = 8          # effective_kv_slots = NUM_CORES * nkv / attn_tp = 8*2/2


def _halo_profile_ctx():
    """``HALO_PROFILE=1`` 时开 ``torch.profiler``,否则 nullcontext(零开销)。

    RPU op 在 profiler 里表现为每个高层调用一个 fused 事件；
    ``GraphCache.capture(sig)`` 已把每次 forward 体裹进 ``record_function(sig.op_id)``,故
    ``halo_action_expert_step_cfg`` 等 op_id 自动成 profiler 事件,无需手动埋点。off 时
    ``with`` 的 ``as`` 目标为 ``None`` → 调用方据此跳过 dump。Profile-only 机制,不改算路。
    """
    if os.environ.get("HALO_PROFILE") != "1":
        return contextlib.nullcontext()
    import torch.profiler
    activities = [torch.profiler.ProfilerActivity.CPU]
    if hasattr(torch.profiler.ProfilerActivity, "PrivateUse1"):
        activities.append(torch.profiler.ProfilerActivity.PrivateUse1)
    return torch.profiler.profile(activities=activities)


def _dump_halo_profile(prof, tag: str) -> None:
    """导出 key_averages 表 + chrome trace 到 ``HALO_PROFILE_DIR``(默认 output/halo_profile)。"""
    if prof is None:
        return
    out_dir = Path(os.environ.get("HALO_PROFILE_DIR", "output/halo_profile"))
    out_dir.mkdir(parents=True, exist_ok=True)
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
    (out_dir / f"{tag}.optable.txt").write_text(table)
    prof.export_chrome_trace(str(out_dir / f"{tag}.trace.json"))
    print(f"[HALO_PROFILE] {tag} → {out_dir}/{tag}.{{optable.txt,trace.json}}", flush=True)
    print(table, flush=True)


def euler_action_schedule(num_timesteps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """HALO ``generate_action`` 去噪 schedule。

    固定 Euler 环（无早停）：``timesteps = linspace(1, 0, num_timesteps)``；循环跑
    ``timesteps[:-1]`` 共 ``num_timesteps − 1`` 步；``dts = timesteps[:-1] − timesteps[1:]``
    （均匀 ``dt = 1/(num_timesteps − 1)``，nt=10 → 1/9）；更新式 ``x_t = x_t − v_t·dt[i]``
    （减号，t:1→0 反向 Euler）。返回 ``(timesteps[:-1], dts)``，长度均 ``num_timesteps − 1``。

    标准部署使用 ``num_timesteps=10``。
    **``timestep_shift`` 生成端 no-op**：`generate_action` 接收该形参但函数体不引用，
    schedule 是裸 linspace，无 shift 扭曲。
    """
    if num_timesteps < 2:
        raise ValueError(f"num_timesteps must be >= 2, got {num_timesteps}")
    timesteps = torch.linspace(1, 0, num_timesteps)
    return timesteps[:-1], timesteps[:-1] - timesteps[1:]


@dataclasses.dataclass(frozen=True)
class HaloRollout:
    """整轨去噪结果（:meth:`HaloStepRunner.rollout_cond` 等返回）。

    ``final_action`` [16,14] fp32 = `generate_action` 返回的最终 ``x_t``（验收单位）。
    ``per_step_v_t`` 逐步 v_t [16,14]（diagnostic：跨步漂移曲线）。
    ``timesteps`` / ``dts`` 实际 Euler schedule（长 ``num_timesteps − 1``），冻进结果便于核对。
    """

    final_action: torch.Tensor
    per_step_v_t: List[torch.Tensor]
    timesteps: torch.Tensor
    dts: torch.Tensor


@dataclasses.dataclass(frozen=True)
class HaloCfgBranch:
    """一个 CFG 分支:prefix-KV 银行(已 swizzle 进 cache)+ rope_position 表 + kv_len。

    三分支(cond / text-cfg / img-cfg)共用 :class:`HaloStepRunner` 的 handle / 权重 /
    x_emb,只换这里的 ``cache``(prefix-KV 银行)+ ``cos``/``sin``(rope_position)+
    ``prefix_len``(kv_len)。``branch_id`` 进 GraphSignature 确保各分支独立 BUILD —— 每个
    BUILD 捕获自己 cache / cos-sin 的 DMA 指针(三分支 cache 是不同张量),REPLAY 各自命中。
    (由 :meth:`HaloStepRunner.make_cfg_branch`(经 :func:`build_cfg_branch`)构造。)
    """

    name: str
    branch_id: int
    cache: RPUCache
    cos: torch.Tensor                 # rope_position 的 RoPE 表 [total_len,head_dim/2] fp16 RPU
    sin: torch.Tensor
    prefix_len: int                   # 该分支 kv_len(cond 4706 / text 4698 / img 8)
    rope_position: int
    attn_mask: torch.Tensor           # [1,1,cs,prefix_len+cs] fp16 全 0 —— SDPA mask 宽随 kv_len 变
    # 分配代数：每次新建分支（新 RPUCache/cos/sin = 新 DMA 地址）由
    # ``make_cfg_branch`` 单调递增并写入。它进 ``_forward_branch`` 的 GraphSignature → 同一
    # runner 跨 episode 复用、(branch_id,prefix_len,rope_position) 不变时,新分配仍产新 key →
    # 强制重 BUILD,杜绝旧 graph 硬编码的上轮 cache/cos DMA 指针被陈旧 replay(静默污染动作)。
    epoch: int


def _destroy_handle(handle: int) -> None:
    """weakref.finalize 回调：进程退出/GC 时容忍 backend 已卸载（先例 wall_oss）。"""
    try:
        torch.ops.rpu.halo_action_expert_destroy(handle)
    except Exception:
        pass


def _swizzle_prefix_7d(
    prefix: torch.Tensor, max_seq_len: int, value_layout: bool
) -> torch.Tensor:
    """prefix [P,2,128] → 7-D cache CPU fp16（slot=head，replica 槽 0）。

    K 布局 [1,sVx,1,8, 8,16,16]（…,sChunk,dChunk），V 末两维转置。
    rows [P, max_seq) 留 0：[P, P+18) 由 step_forward 的 insert 覆盖。
    """
    p_len, nkv, hd = prefix.shape
    s_vx = max_seq_len // _S_CHUNK
    d_vx = hd // _S_CHUNK
    pad = torch.zeros(max_seq_len, nkv, hd, dtype=torch.float16)
    pad[:p_len] = prefix.to(torch.float16)
    out = torch.zeros(1, s_vx, 1, d_vx, _KV_SLOTS, _S_CHUNK, _S_CHUNK, dtype=torch.float16)
    for head in range(nkv):
        # [maxS,hd] → [sVx,16,dVx,16] → (sVx,dVx,16,16)；V 交换 (sChunk,dChunk)。
        blocks = pad[:, head, :].view(s_vx, _S_CHUNK, d_vx, _S_CHUNK)
        blocks = blocks.permute(0, 2, 3, 1) if value_layout else blocks.permute(0, 2, 1, 3)
        out[0, :, 0, :, head] = blocks
    return out.contiguous()


@dataclasses.dataclass
class HaloStepRunner:
    """HALO action expert 单步 runner（经 :func:`build_halo_action_expert` 构造）。

    字段所有权：全部 RPU 权重/表/cache 在 runner 存活期内有效；``_handle``
    由 finalize 释放。``rope_position`` 是 query 行共享的常数 position。
    """

    cfg: HaloConfig
    rope_position: int
    text: HaloBranchWeights
    act: HaloBranchWeights
    host: HaloHostWeights
    cache: RPUCache
    _handle: int
    _graph_cache: "rpu_backend.graph.GraphCache"
    _attn_mask: torch.Tensor                 # [1,1,cs,total_len] fp16 CPU，全 0
    # RoPE 表(rope_position 的 cos/sin,[total_len,head_dim/2] fp16 RPU)。改为
    # per-forward 传入 step_forward(对齐 qwenpi05 cos_ref_),CFG 三分支各传自己的表;
    # set_weights 仍收一份作基类契约+默认。runner 持有 cond 分支的表(与 handle 等寿命)。
    _cos: torch.Tensor
    _sin: torch.Tensor
    _x_emb: torch.Tensor = dataclasses.field(init=False)       # [1,cs,H] fp16 RPU
    _handle_finalizer: object | None = dataclasses.field(
        init=False, default=None, repr=False
    )
    _closed: bool = dataclasses.field(init=False, default=False, repr=False)
    # prefix 代数：每次 insert_prefix 换新 RPU 张量（DMA 指针变化），编进
    # GraphSignature 强制重 BUILD，禁止旧 graph 指针重放。（insert_prefix/step 路用。）
    _prefix_epoch: int = dataclasses.field(init=False, default=0)
    # CFG 分支分配代数：rollout_cfg/_forward_branch 路用。每次
    # make_cfg_branch 新建分支银行（新 DMA 地址）单调递增 → 进 _forward_branch 签名 →
    # 跨 episode 复用同 runner 时新分配仍重 BUILD（_prefix_epoch 只随 insert_prefix 动，
    # CFG 路从不调它，故须独立计数；镜像 image_flow._epoch_counter）。
    _cfg_branch_epoch: int = dataclasses.field(init=False, default=0)

    def __post_init__(self) -> None:
        cfg = self.cfg
        # 稳定输入缓冲：BUILD 期烘焙 x_emb 的 DMA 地址，REPLAY 复用（先例 wall_oss
        # x buf）。输出走基类 output_tensor_（registry 跨 forward 稳定），无需独立
        # out buf —— 在 capture 退出后读 step_forward 的返回张量即可。
        self._x_emb = torch.empty(
            1, cfg.chunk_size, cfg.hidden_size, dtype=torch.float16, device="rpu")

    def close(self) -> None:
        """Clear graph ownership, then retire the native handle exactly once."""
        if self._closed:
            return
        self._closed = True
        try:
            self._graph_cache.clear()
        except Exception:
            pass
        finalizer = self._handle_finalizer
        if finalizer is not None:
            if getattr(finalizer, "alive", False):
                finalizer()
        else:
            _destroy_handle(self._handle)

    # ── prefix ────────────────────────────────────────────────────────────
    def insert_prefix(self, k_layers: List[torch.Tensor], v_layers: List[torch.Tensor]) -> None:
        """灌入 28 层 prefix K/V（NaiveCache 捕获，CPU bf16/fp32 [4706,2,128]）。

        K/V 已带原始 RoPE；host swizzle 后整体替换 cache 张量（与 C++ 按
        data_ptr DMA 的借用语义一致：step_forward 每次重新取列表）。
        """
        cfg = self.cfg
        expect = (cfg.prefix_len, cfg.num_kv_heads, cfg.head_dim)
        if len(k_layers) != cfg.num_layers or len(v_layers) != cfg.num_layers:
            raise ValueError(
                f"prefix layers mismatch: got {len(k_layers)}/{len(v_layers)}, "
                f"expect {cfg.num_layers}")
        for i, (k, v) in enumerate(zip(k_layers, v_layers)):
            if tuple(k.shape) != expect or tuple(v.shape) != expect:
                raise ValueError(
                    f"layer {i} prefix shape {tuple(k.shape)}/{tuple(v.shape)} != {expect}")
            self.cache.k_caches[i] = _swizzle_prefix_7d(
                k, self.cache.max_seq_len, value_layout=False).to("rpu")
            self.cache.v_caches[i] = _swizzle_prefix_7d(
                v, self.cache.max_seq_len, value_layout=True).to("rpu")
        self.cache.reset_to_position(cfg.prefix_len)
        self._prefix_epoch += 1

    # ── host boundary ─────────────────────────────────────────────────────
    def _assemble_x_emb(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        action_position_ids: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> torch.Tensor:
        """host fp32 boundary → [1,cs,H] fp16 CPU（行 1..16 action，行 0/17 text）。

        与 HALO `_forward_action_flow` 逐项一致：
        action 行 = in_proj(x_t) + temb(timestep) + sinpos(action_position_ids)，
        text 行 = embed_tokens(text_ids)；fp32 → bf16 → fp16。
        """
        cfg, host = self.cfg, self.host
        x_act = x_t.float() @ host.in_proj_w.t() + host.in_proj_b          # [16,H]
        t_freq = timestep_embedding(timestep, cfg)                          # [16,256]
        temb = torch.nn.functional.silu(
            t_freq @ host.temb0_w.t() + host.temb0_b) @ host.temb2_w.t() + host.temb2_b
        sinpos = sinusoidal_pos_embedding(action_position_ids, cfg)         # [16,H]
        x_act = x_act + temb + sinpos

        x_emb = torch.zeros(cfg.chunk_size, cfg.hidden_size, dtype=torch.float32)
        x_emb[list(cfg.action_rows)] = x_act
        x_emb[list(cfg.text_rows)] = torch.nn.functional.embedding(
            text_ids, self.host.embed_tokens).float()
        return x_emb.to(torch.bfloat16).to(torch.float16).unsqueeze(0)

    # ── step ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        action_position_ids: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> torch.Tensor:
        """一次 act step。I/O 契约（全 CPU）：

          x_t [16,14] fp32；timestep [16] fp32（HALO 断言全行同值）；
          action_position_ids [16] int64；text_ids [2] int64 → v_t [16,14] fp32。

        每步先 reset_to_position(prefix)（K/V [4706,4724) 被本步 insert 覆盖）。
        """
        cfg = self.cfg
        if self._prefix_epoch == 0:
            raise RuntimeError("HaloStepRunner.step: call insert_prefix() first")
        if tuple(x_t.shape) != (len(cfg.action_rows), cfg.action_dim):
            raise ValueError(f"x_t must be [16,{cfg.action_dim}], got {tuple(x_t.shape)}")
        self.cache.reset_to_position(cfg.prefix_len)
        self._x_emb.copy_(self._assemble_x_emb(x_t, timestep, action_position_ids, text_ids))
        sig = rpu_backend.graph.GraphSignature(
            op_id="halo_action_expert_step",
            shapes=[cfg.chunk_size, cfg.hidden_size],
            dyn_dims=[cfg.num_layers, cfg.prefix_len, self.rope_position, self._prefix_epoch],
            dtypes=[torch.float16],
        )
        with self._graph_cache.capture(sig):
            hidden_out = torch.ops.rpu.halo_action_expert_step_forward(
                self._handle, self._x_emb,
                self.cache.k_caches, self.cache.v_caches,
                self._cos, self._sin,                 # per-forward RoPE 表(cond 分支)
                self._attn_mask, cfg.prefix_len)
        # 必须在 capture 作用域**退出后**读：返回的是基类 output_tensor_（graph 写
        # 目标），图在 with 块退出 end() 时才执行写入；作用域内读到的是尚未执行的
        # 全 0；这是 deferred graph 输出的生命周期契约。
        hidden = hidden_out[0].float().cpu()                                # [cs,H]
        act = hidden[list(cfg.action_rows)]                                 # [16,H]
        return act @ self.host.out_proj_w.t() + self.host.out_proj_b        # [16,14]

    @torch.no_grad()
    def rollout_cond(
        self,
        x_t_init: torch.Tensor,
        action_position_ids: torch.Tensor,
        text_ids: torch.Tensor,
        *,
        num_timesteps: int = 10,
    ) -> HaloRollout:
        """cond-only 整轨去噪(无 CFG,单分支)。复刻 `generate_action` 固定 Euler 环。

        I/O 契约(全 CPU)：``x_t_init`` [16,14] fp32(= ``packed_init_noises``);
        ``action_position_ids`` [16] int64、``text_ids`` [2] int64(跨步常数)。每步:
        ``timestep = full(16, t_i)``(全行同值,满足 `_forward_action_flow` 断言),
        ``v_t = step(x_t, t_i, …)``,``x_t = x_t − v_t·dt[i]``;返回 final action +
        逐步 v_t(diagnostic)。

        **cond-only**:仅注入 cond prefix-KV(调用前须 :meth:`insert_prefix`);CFG
        三分支(text-cfg/img-cfg + global renorm)在后续含-CFG rollout 叠加。timestep 经
        mutable ``_x_emb`` 喂入、**不进 GraphSignature** → step0 BUILD、余步 REPLAY
        (build-once/replay-many,与单步 dispatch 测试同 replay 路径;跨步只换 x_emb 缓冲)。

        """
        cfg = self.cfg
        if self._prefix_epoch == 0:
            raise RuntimeError("HaloStepRunner.rollout_cond: call insert_prefix() first")
        expect = (len(cfg.action_rows), cfg.action_dim)
        if tuple(x_t_init.shape) != expect:
            raise ValueError(f"x_t_init must be {list(expect)}, got {tuple(x_t_init.shape)}")
        timesteps, dts = euler_action_schedule(num_timesteps)
        x_t = x_t_init.to(torch.float32).clone()
        per_step_v_t: List[torch.Tensor] = []
        with _halo_profile_ctx() as _prof:                                 # HALO_PROFILE=1 才开
            for i in range(int(timesteps.shape[0])):
                timestep = torch.full(
                    (len(cfg.action_rows),), float(timesteps[i]), dtype=torch.float32)
                v_t = self.step(x_t, timestep, action_position_ids, text_ids)   # [16,14]
                per_step_v_t.append(v_t)
                x_t = x_t - v_t * float(dts[i])
        _dump_halo_profile(_prof, "halo_rollout_cond")
        return HaloRollout(
            final_action=x_t, per_step_v_t=per_step_v_t, timesteps=timesteps, dts=dts)

    # ── CFG 三分支 ─────────────────────────────────────────────────────────
    def make_cfg_branch(
        self,
        name: str,
        branch_id: int,
        k_layers: List[torch.Tensor],
        v_layers: List[torch.Tensor],
        *,
        rope_position: int,
        prefix_len: int,
        max_seq_len: int = 4736,
    ) -> HaloCfgBranch:
        """构造带分配身份(epoch)的 CFG 分支 —— **生产路径唯一入口**。

        每次调用单调递增 ``self._cfg_branch_epoch`` 并注入新分支银行(新 RPUCache/cos/sin =
        新 DMA 地址),使 ``_forward_branch`` 的 GraphSignature 随每次新分配而变 → 同一 runner
        跨 episode 复用、(branch_id,prefix_len,rope_position) 不变时仍重 BUILD,杜绝旧 graph
        硬编码的上轮 DMA 指针被陈旧 replay。镜像 ``image_flow._prepare_branch``。
        """
        self._cfg_branch_epoch += 1
        return build_cfg_branch(
            name, branch_id, k_layers, v_layers,
            rope_position=rope_position, prefix_len=prefix_len,
            epoch=self._cfg_branch_epoch, cfg=self.cfg, max_seq_len=max_seq_len)

    def _forward_branch(self, branch: HaloCfgBranch) -> torch.Tensor:
        """一个 CFG 分支的 native forward → v_t [16,14]。

        ``self._x_emb`` 须已由调用方写入(三分支同一 x_emb;query 跨分支不变)。分支只换
        cache(prefix-KV 银行)/ cos-sin(rope_position)/ prefix_len(kv_len);共用 handle。
        GraphSignature 含 ``branch_id`` → 各分支独立 BUILD(捕获自己 cache/cos DMA 指针),
        step0 BUILD、后续步同分支 REPLAY。返回前必须在 capture 退出**后**读 hidden_out
        (基类 output_tensor_,图在 with 退出才写;三分支共用 output_tensor_,故每分支结果
        须在下个分支 forward 前读走 —— 本方法读后即返回,满足此序)。
        """
        cfg = self.cfg
        branch.cache.reset_to_position(branch.prefix_len)
        sig = rpu_backend.graph.GraphSignature(
            op_id="halo_action_expert_step_cfg",
            shapes=[cfg.chunk_size, cfg.hidden_size],
            # 5 维拓扑契约 = _cfg_branch_dyn_dims（生产与 board-free 负测同源；含 branch.epoch
            # 分配身份,见该函数 docstring）。
            dyn_dims=_cfg_branch_dyn_dims(cfg, branch),
            dtypes=[torch.float16],
        )
        with self._graph_cache.capture(sig):
            hidden_out = torch.ops.rpu.halo_action_expert_step_forward(
                self._handle, self._x_emb,
                branch.cache.k_caches, branch.cache.v_caches,
                branch.cos, branch.sin,
                branch.attn_mask, branch.prefix_len)
        hidden = hidden_out[0].float().cpu()                                # [cs,H]
        act = hidden[list(cfg.action_rows)]                                 # [16,H]
        return act @ self.host.out_proj_w.t() + self.host.out_proj_b        # [16,14]

    @torch.no_grad()
    def rollout_cfg(
        self,
        x_t_init: torch.Tensor,
        action_position_ids: torch.Tensor,
        text_ids: torch.Tensor,
        *,
        cond: HaloCfgBranch,
        text: HaloCfgBranch,
        img: HaloCfgBranch,
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_interval: Sequence[float],
        cfg_renorm_min: float = 0.0,
        num_timesteps: int = 10,
    ) -> HaloRollout:
        """含-CFG 整轨去噪。复刻 `generate_action` + `_forward_action_flow` 三分支 + global renorm。

        每步对同一 x_t 跑 cond/text-cfg/img-cfg 三分支(各自 prefix-KV 银行 + rope_position +
        kv_len,共用 handle/权重/x_emb)→ host global renorm 合并 → ``x_t = x_t − v_t·dt``。
        CFG 判定 ``cfg_interval[0] < t ≤ cfg_interval[1]``；in-interval 步
        跑 3 分支组合,区间外仅 cond(scales=1)。组合公式为：
        ``v_text_ = text + s_text·(cond − text)``;``v_ = img + s_img·(v_text_ − img)``;
        ``v_t = v_ · clamp(‖cond‖/‖v_‖, min, 1)``(整张量标量范数,host)。

        参数由调用方显式提供；函数不隐式选择 CFG profile。
        """
        cfg = self.cfg
        expect = (len(cfg.action_rows), cfg.action_dim)
        if tuple(x_t_init.shape) != expect:
            raise ValueError(f"x_t_init must be {list(expect)}, got {tuple(x_t_init.shape)}")
        timesteps, dts = euler_action_schedule(num_timesteps)
        x_t = x_t_init.to(torch.float32).clone()
        per_step_v_t: List[torch.Tensor] = []
        with _halo_profile_ctx() as _prof:                                 # HALO_PROFILE=1 才开
            for i in range(int(timesteps.shape[0])):
                t = float(timesteps[i])
                timestep = torch.full((len(cfg.action_rows),), t, dtype=torch.float32)
                # x_emb 三分支共用,每步写一次(query 跨分支不变,跨步随 x_t/timestep 变)。
                self._x_emb.copy_(self._assemble_x_emb(x_t, timestep, action_position_ids, text_ids))
                v_cond = self._forward_branch(cond)                         # cond 总跑
                # 区间内 text / image CFG 独立门控。
                in_cfg = (cfg_interval[0] < t <= cfg_interval[1]
                          and (cfg_text_scale > 1.0 or cfg_img_scale > 1.0))
                if in_cfg:
                    v_text_ = v_cond
                    if cfg_text_scale > 1.0:
                        v_text = self._forward_branch(text)
                        v_text_ = v_text + cfg_text_scale * (v_cond - v_text)
                    if cfg_img_scale > 1.0:
                        v_img = self._forward_branch(img)
                        v_ = v_img + cfg_img_scale * (v_text_ - v_img)
                    else:
                        v_ = v_text_
                    scale = (v_cond.norm() / (v_.norm() + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                    v_t = v_ * scale                                       # global renorm
                else:
                    v_t = v_cond                                          # 区间外/无 CFG 仅 cond
                per_step_v_t.append(v_t)
                x_t = x_t - v_t * float(dts[i])
        _dump_halo_profile(_prof, "halo_rollout_cfg")
        return HaloRollout(
            final_action=x_t, per_step_v_t=per_step_v_t, timesteps=timesteps, dts=dts)

    def dispatch_submit_total(self) -> int:
        """图缓存累计的真实 /dev/rpu 批提交次数（GraphStats.hw_batch_submit_total 之和）。

        >0 证明 action 专家的 kernel 批确被同步发射到设备（Queue_t::enqueu_batch
        wait_finish=true），而非仅建图 / 走 replay 指针；用作 native dispatch 硬件
        证据（治理门 hardware_dispatch_proven 的 int>0 路径）。须在 ≥1 次 step() 后读。
        """
        return sum(int(s.hw_batch_submit_total) for s in self._graph_cache.snapshot())


def build_halo_action_expert(
    ckpt_path: "str | Path",
    *,
    rope_position: int,
    cfg: HaloConfig = HaloConfig(),
    max_seq_len: int = 4736,
) -> HaloStepRunner:
    """装载 HALO action expert：权重 → handle → set_weights/set_moe_weights。

    Args:
        ckpt_path: HALO ema.safetensors（bf16）。
        rope_position: query 行共享的常数 position。
        max_seq_len: KV cache 容量，须为 16 的倍数且 ≥ prefix+cs（4724→4736）。
    """
    if max_seq_len % _S_CHUNK != 0 or max_seq_len < cfg.total_len:
        raise ValueError(
            f"max_seq_len {max_seq_len} must be a multiple of {_S_CHUNK} and >= {cfg.total_len}")
    text, act, host = load_halo_branches(Path(ckpt_path), cfg)
    cos, sin = build_rope_tables(cfg, rope_position)
    handle = int(torch.ops.rpu.halo_action_expert_create())
    runner = None
    try:
        torch.ops.rpu.halo_action_expert_set_weights(
            handle,
            text.q_w, text.k_w, text.v_w, text.o_w,
            text.q_norm, text.k_norm, text.input_norm, text.post_norm,
            text.gate_w, text.up_w, text.down_w,
            cos, sin, text.final_norm_w,
            cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
            cfg.hidden_size, cfg.intermediate_size,
            cfg.rms_norm_eps, True,
            text.q_bias, text.k_bias, text.v_bias)
        torch.ops.rpu.halo_action_expert_set_moe_weights(
            handle,
            act.q_w, act.k_w, act.v_w, act.o_w,
            act.q_norm, act.k_norm, act.input_norm, act.post_norm,
            act.gate_w, act.up_w, act.down_w,
            act.q_bias, act.k_bias, act.v_bias,
            act.final_norm_w, build_text_row_mask(cfg), cfg.chunk_size)
        # cache / attn_mask / runner 也必须在 try 内：weakref.finalize 只在
        # HaloStepRunner.__post_init__ 安装，若这几步失败而 handle 已建，孤儿 handle
        # 不会被回收（"一进程一个 live RPU model" 约束下重试会留下注册实例 + RPU 资源）。
        cache = RPUCache(
            num_layers=cfg.num_layers, batch_size=1, max_seq_len=max_seq_len,
            num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim, attn_tp=cfg.attn_tp)
        attn_mask = torch.zeros(
            1, 1, cfg.chunk_size, cfg.total_len, dtype=torch.float16)   # 非因果全注意，全 0
        runner = HaloStepRunner(
            cfg=cfg, rope_position=rope_position, text=text, act=act, host=host,
            cache=cache, _handle=handle, _graph_cache=rpu_backend.graph.GraphCache(),
            _attn_mask=attn_mask, _cos=cos, _sin=sin)
        runner._handle_finalizer = weakref.finalize(
            runner, _destroy_handle, handle
        )
    except BaseException:
        # 构造任一步失败 → 孤儿 handle 立即回收。
        if runner is not None:
            runner.close()
        else:
            _destroy_handle(handle)
        raise
    return runner


def _cfg_branch_dyn_dims(cfg: HaloConfig, branch: HaloCfgBranch) -> List[int]:
    """``_forward_branch`` 的 GraphSignature ``dyn_dims`` —— 生产与 board-free 负测**单一真相源**。

    5 维:``num_layers`` / ``prefix_len``(=kv_len)/ ``rope_position`` / ``branch_id``
    (cache 银行身份)/ ``epoch``(分配代数：新分配 → 新 key → 重 BUILD)。
    ``timestep`` / cfg-scales 故意**不**进 key(不进 RPU 图:timestep 经 mutable ``_x_emb``
    缓冲、scales 在 host renorm;进 key 会破坏 build-once/replay-many + 让 replay 稳定门失效)。
    """
    return [cfg.num_layers, branch.prefix_len, branch.rope_position,
            branch.branch_id, branch.epoch]


def build_cfg_branch(
    name: str,
    branch_id: int,
    k_layers: List[torch.Tensor],
    v_layers: List[torch.Tensor],
    *,
    rope_position: int,
    prefix_len: int,
    epoch: int,
    cfg: HaloConfig = HaloConfig(),
    max_seq_len: int = 4736,
) -> HaloCfgBranch:
    """从一个 CFG 分支的 prefix-KV 银行构造 :class:`HaloCfgBranch`(供 :meth:`HaloStepRunner.rollout_cfg`)。

    ``k_layers``/``v_layers`` = 该分支 CUDA NaiveCache 捕获的 28 层 prefix K/V
    (``[P,2,128]`` bf16/fp32,已带原始 RoPE,P=该分支 kv_len)。swizzle 进新 RPUCache +
    建该 ``rope_position`` 的 cos/sin 表。三分支各调一次(cond/text-cfg/img-cfg)。
    不持 handle —— 共用 :class:`HaloStepRunner` 的句柄/权重(rollout_cfg 调度)。

    ``epoch`` = 分配代数（必填）：标识本次新建的 cache/cos/sin DMA 身份，
    进 ``_forward_branch`` 签名。**生产路径请用 :meth:`HaloStepRunner.make_cfg_branch`**(它单调
    递增 runner 的 ``_cfg_branch_epoch`` 后传入),勿手凑 ``epoch``(同值跨 episode 会撞 key →
    陈旧 replay)。``epoch`` 无默认值即为此 guard —— 漏传 → TypeError(逼调用方走 make_cfg_branch)。
    """
    if len(k_layers) != cfg.num_layers or len(v_layers) != cfg.num_layers:
        raise ValueError(
            f"{name}: prefix layers {len(k_layers)}/{len(v_layers)} != {cfg.num_layers}")
    if not 0 <= prefix_len <= max_seq_len:
        raise ValueError(f"{name}: prefix_len {prefix_len} out of [0,{max_seq_len}]")
    cache = RPUCache(
        num_layers=cfg.num_layers, batch_size=1, max_seq_len=max_seq_len,
        num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim, attn_tp=cfg.attn_tp)
    for i, (k, v) in enumerate(zip(k_layers, v_layers)):
        cache.k_caches[i] = _swizzle_prefix_7d(k, max_seq_len, value_layout=False).to("rpu")
        cache.v_caches[i] = _swizzle_prefix_7d(v, max_seq_len, value_layout=True).to("rpu")
    cache.reset_to_position(prefix_len)
    cos, sin = build_rope_tables(cfg, rope_position)
    # SDPA mask 宽 = 该分支 kv 跨度 prefix_len + cs(非因果全注意,全 0);三分支 kv_len 不同 → 宽不同。
    attn_mask = torch.zeros(1, 1, cfg.chunk_size, prefix_len + cfg.chunk_size, dtype=torch.float16)
    return HaloCfgBranch(
        name=name, branch_id=branch_id, cache=cache, cos=cos, sin=sin,
        prefix_len=prefix_len, rope_position=rope_position, attn_mask=attn_mask, epoch=epoch)
