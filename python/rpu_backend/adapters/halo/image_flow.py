"""HALO image_flow_step（mode="gen" 图像 flow-matching 去噪单步）→ RPU。

``use_goal_image=True`` 路径每步在 host 合成
``x_emb = vae2llm(x_t) + time_embedder(timestep) + latent_pos_embed[pos]``
→ ``forward_inference(mode="gen", is_causal=False)`` 3 个 CFG 分支（cond / cfg_text /
cfg_img，各自 prefix-KV bank + query rope_position）→ ``llm2vae`` → 各 v_t → host 全局/
通道 renorm 合并。完整生成执行 ``num_timesteps-1`` 个
Euler 步：``x_t -= v_t * dt``（timestep_shift 重参数化时间轴）。

``HaloImageFlowModel`` 使用双流 MoT：query = **1378 行 = 2 text
(start/end_of_image, [0,1377]) + 1376 vae**；2 text 行走 und/base
孪生、1376 vae 行走 gen(``*_moe_gen``)孪生，per-twin 投影后行掩码合并（= action expert ⊕
KV_FIRST）。非因果 1378 ⇒ KV_FIRST 多块（Phase 1 插全 K/V，Phase 2 满 KV SDPA）；
image gen 全 attend ⇒ SDPA MASK_NONE（行掩码 ≠ SDPA mask）。组合 Gemma KV_FIRST 骨架 +
HALO Qwen2.5 层体（qk head-norm + SiLU + per-forward cos/sin）。
``vae2llm/time/pos/embed_tokens/llm2vae`` 位于 host 边界。

机制要点：
  * **per-forward cos/sin**（镜像 qwenpi05 / action CFG）：图像 token 共享单一 rope_position
    ，3 个 CFG 分支 position 各异 → 各传自己的位置广播 RoPE 表；每分支
    GraphSignature（含 rope_position + prefix_len + prefix_epoch）独立 → 独立 BUILD →
    各捕各自 cos/sin 指针，防止陈旧 replay。
  * **3 分支独立 cache**：cond/cfg_text/cfg_img 各持自己的 prefix-KV bank。
    prefix 注入一次（rollout 起），query K/V 每步 KV_FIRST Phase 1 重插 ``[P,P+N)`` 覆盖
    上步 → prefix_epoch 跨步稳定 → 49 步 REPLAY（仅首步 BUILD）。
  * **MASK_NONE 全 attend**：``step_forward`` 不传 mask（内部 ``run_all_layers`` nullopt +
    is_causal=false → MASK_NONE）；KV_FIRST 内部多块, 调用方不传 chunk_size。

数值正确性需通过硬件端到端验证。
"""
from __future__ import annotations

import dataclasses
import weakref
from pathlib import Path
from typing import List, Optional, Sequence

import torch

import rpu_backend
from rpu_backend.api.cache import RPUCache
from rpu_backend.adapters.halo.runtime import _S_CHUNK, _swizzle_prefix_7d
from rpu_backend.adapters.halo.weights import (
    HaloBranchWeights,
    HaloConfig,
    HaloGenHostWeights,
    assemble_image_x_emb,
    build_image_text_row_mask,
    build_rope_tables,
    image_v_t,
    load_halo_gen_branch,
)


def _destroy_image_flow(handle: int) -> None:
    """释放 HaloImageFlowModel handle（weakref.finalize 在 GC 时调）。"""
    try:
        torch.ops.rpu.halo_image_flow_destroy(handle)
    except Exception:
        pass


@dataclasses.dataclass
class CFGBranch:
    """一个 CFG 分支的 prefix-KV bank + query rope_position（cond / cfg_text / cfg_img）。

    prefix_k/v: 28 层 ``[P,2,128]`` CUDA NaiveCache（bf16/fp32）；rope_position: 该分支
    query 的共享 1D position（图像 query 全行同值）。
    """

    prefix_k: List[torch.Tensor]
    prefix_v: List[torch.Tensor]
    rope_position: int

    @property
    def prefix_len(self) -> int:
        return int(self.prefix_k[0].shape[0])


@dataclasses.dataclass
class _PreparedBranch:
    """注入完毕的分支运行态：独立 cache + 预建 RoPE 表 + prefix 元数据。"""

    cache: RPUCache
    cos: torch.Tensor
    sin: torch.Tensor
    prefix_len: int
    epoch: int


@dataclasses.dataclass
class HaloImageFlowStep:
    """HALO image flow 去噪-单步 runner（经 :func:`build_halo_image_flow` 构造）。

    单 handle（gen 孪生权重）+ 每 CFG 分支一个 RPUCache。``denoise_step`` 跑 1 步（3 分支
    forward + host renorm）；``rollout`` 跑完整 Euler 循环。
    所有权：RPU 权重/cache 在 runner 存活期有效；``_handle`` 由 finalize 释放。
    """

    cfg: HaloConfig
    und: HaloBranchWeights
    gen: HaloBranchWeights
    host: HaloGenHostWeights
    max_seq_len: int
    _handle: int
    _graph_cache: "rpu_backend.graph.GraphCache"
    _epoch_counter: int = dataclasses.field(init=False, default=0)
    _handle_finalizer: object | None = dataclasses.field(
        init=False, default=None, repr=False
    )
    _closed: bool = dataclasses.field(init=False, default=False, repr=False)

    def close(self) -> None:
        """Clear graph ownership, then retire the image-flow handle once."""
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
            _destroy_image_flow(self._handle)

    # ── prefix 注入（与 text_decode._inject_prefix 同法：7-D swizzle 整体替换） ──
    def _prepare_branch(self, branch: CFGBranch) -> _PreparedBranch:
        cfg = self.cfg
        pk, pv = branch.prefix_k, branch.prefix_v
        if len(pk) != cfg.num_layers or len(pv) != cfg.num_layers:
            raise ValueError(f"prefix layers {len(pk)}/{len(pv)} != {cfg.num_layers}")
        prefix_len = branch.prefix_len
        n = cfg.num_image_query_tokens          # 1378（含 2 text 边界行）
        if prefix_len + n > self.max_seq_len:
            raise ValueError(
                f"prefix_len+N {prefix_len + n} > max_seq_len {self.max_seq_len}")
        cache = RPUCache(
            num_layers=cfg.num_layers, batch_size=1, max_seq_len=self.max_seq_len,
            num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim, attn_tp=cfg.attn_tp)
        expect = (prefix_len, cfg.num_kv_heads, cfg.head_dim)
        for i, (k, v) in enumerate(zip(pk, pv)):
            if tuple(k.shape) != expect or tuple(v.shape) != expect:
                raise ValueError(
                    f"layer {i} prefix {tuple(k.shape)}/{tuple(v.shape)} != {expect}")
            cache.k_caches[i] = _swizzle_prefix_7d(
                k, self.max_seq_len, value_layout=False).to("rpu")
            cache.v_caches[i] = _swizzle_prefix_7d(
                v, self.max_seq_len, value_layout=True).to("rpu")
        cache.reset_to_position(prefix_len)
        # 位置广播 RoPE 表（rows = prefix+N，须 ≥ C++ prefix_len+seq 校验）。
        cos, sin = build_rope_tables(cfg, branch.rope_position, rows=prefix_len + n)
        self._epoch_counter += 1
        return _PreparedBranch(cache=cache, cos=cos, sin=sin,
                               prefix_len=prefix_len, epoch=self._epoch_counter)

    def _forward(self, x_emb: torch.Tensor, pb: _PreparedBranch) -> torch.Tensor:
        """单分支 forward：x_emb [1,N,H] fp16 RPU → hidden [N,H] cpu fp32（post-final-norm）。

        sig 含 prefix_len（KV 落位）+ N + rope_position（经 epoch 唯一 cos/sin 指针隐含）
        + epoch（DMA 指针代数）→ 防陈旧 replay 用错 KV 落位 / 错分支 cos/sin。
        """
        cfg = self.cfg
        sig = rpu_backend.graph.GraphSignature(
            op_id="halo_image_flow_step",
            shapes=[cfg.num_image_query_tokens, cfg.hidden_size],
            dyn_dims=[cfg.num_layers, pb.prefix_len, cfg.num_image_query_tokens, pb.epoch],
            dtypes=[torch.float16],
        )
        with self._graph_cache.capture(sig):
            hidden_out = torch.ops.rpu.halo_image_flow_step_forward(
                self._handle, x_emb, pb.cache.k_caches, pb.cache.v_caches,
                pb.cos, pb.sin, pb.prefix_len)
        # capture 退出**后**读（graph end() 已写；作用域内提前读得全 0）。
        return hidden_out[0].float().cpu()                   # [N,H]

    @staticmethod
    def _cfg_renorm(
        v_t: torch.Tensor, cfg_text_v_t: Optional[torch.Tensor],
        cfg_img_v_t: Optional[torch.Tensor],
        cfg_text_scale: float, cfg_img_scale: float,
        cfg_renorm_min: float, cfg_renorm_type: str
    ) -> torch.Tensor:
        """host CFG renorm（非 text-channel 路径，支持 global/channel）。"""
        # text / image CFG 独立门控；两者都不启用时直接返回 cond。
        if cfg_text_scale <= 1.0 and cfg_img_scale <= 1.0:
            return v_t
        v_t_text_ = v_t
        if cfg_text_scale > 1.0:
            if cfg_text_v_t is None:
                raise ValueError(
                    "cfg_text_scale>1.0 需 cfg_text 分支(cfg_text_v_t),但为 None")
            v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
        if cfg_img_scale > 1.0:
            if cfg_img_v_t is None:
                raise ValueError(
                    "cfg_img_scale>1.0 需 cfg_img 分支(cfg_img_v_t),但为 None")
            v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
        else:
            v_t_ = v_t_text_
        if cfg_renorm_type == "global":
            norm_v_t = torch.norm(v_t)
            norm_v_t_ = torch.norm(v_t_)
        elif cfg_renorm_type == "channel":
            norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
            norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
        else:
            raise NotImplementedError(f"cfg_renorm_type={cfg_renorm_type} 未支持")
        scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
        return v_t_ * scale

    @torch.no_grad()
    def denoise_step(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        vae_position_ids: torch.Tensor,
        cond: CFGBranch,
        cfg_text: Optional[CFGBranch] = None,
        cfg_img: Optional[CFGBranch] = None,
        *,
        cfg_text_scale: float = 1.0,
        cfg_img_scale: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
    ) -> torch.Tensor:
        """一次去噪步（独立注入 3 分支 → forward → renorm）→ v_t [N,64] cpu fp32。

        单步 v_t 硬门用此入口（scales=1 时退化为纯 cond 分支，隔离 RPU forward 正确性）。
        """
        x_emb = assemble_image_x_emb(x_t, timestep, vae_position_ids, self.host, self.cfg)
        pb_cond = self._prepare_branch(cond)
        v_t = image_v_t(self._forward(x_emb, pb_cond), self.host, self.cfg)
        cfg_text_v_t = cfg_img_v_t = None
        if cfg_text_scale > 1.0:
            if cfg_text is None:
                raise ValueError("cfg_text_scale>1 需提供 cfg_text 分支")
            cfg_text_v_t = image_v_t(self._forward(x_emb, self._prepare_branch(cfg_text)), self.host, self.cfg)
        if cfg_img_scale > 1.0:
            if cfg_img is None:
                raise ValueError("cfg_img_scale>1 需提供 cfg_img 分支")
            cfg_img_v_t = image_v_t(self._forward(x_emb, self._prepare_branch(cfg_img)), self.host, self.cfg)
        return self._cfg_renorm(v_t, cfg_text_v_t, cfg_img_v_t, cfg_text_scale,
                                cfg_img_scale, cfg_renorm_min, cfg_renorm_type)

    @torch.no_grad()
    def rollout(
        self,
        init_noise: torch.Tensor,
        vae_position_ids: torch.Tensor,
        cond: CFGBranch,
        cfg_text: Optional[CFGBranch] = None,
        cfg_img: Optional[CFGBranch] = None,
        *,
        num_timesteps: int = 50,
        timestep_shift: float = 1.0,
        cfg_text_scale: float = 1.0,
        cfg_img_scale: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Sequence[float] = (0.0, 1.0),
    ) -> torch.Tensor:
        """完整 Euler 去噪 → final latent [N,64] cpu fp32。

        3 分支 prefix 注入一次（跨步 REPLAY）；每步 host assemble → 3 forward（按 interval
        门控 scales）→ renorm → ``x_t -= v_t * dt``。
        """
        # Fail fast with a clear error instead of a
        # silent no-op / an opaque downstream crash.
        #   - num_timesteps<2 → ts[:-1] empty → 0 Euler steps → init_noise returned
        #     unchanged (silent no-denoise). Require >=2.
        #   - cfg_*_scale>1.0 with the matching branch None → self._forward(x_emb,
        #     None) inside the loop (use_cfg gating makes the scale active). Mirror
        #     denoise_step's up-front guard.
        if num_timesteps < 2:
            raise ValueError(
                f"rollout: num_timesteps must be >= 2 (got {num_timesteps}); "
                "a single timestep performs zero denoising steps")
        if cfg_text_scale > 1.0 and cfg_text is None:
            raise ValueError("rollout: cfg_text_scale>1.0 需提供 cfg_text 分支")
        if cfg_img_scale > 1.0 and cfg_img is None:
            raise ValueError("rollout: cfg_img_scale>1.0 需提供 cfg_img 分支")
        n = self.cfg.num_image_tokens
        x_t = init_noise.float().cpu()                       # [N,64]
        pb_cond = self._prepare_branch(cond)
        pb_text = self._prepare_branch(cfg_text) if cfg_text is not None else None
        pb_img = self._prepare_branch(cfg_img) if cfg_img is not None else None

        ts = torch.linspace(1, 0, num_timesteps)
        ts = timestep_shift * ts / (1 + (timestep_shift - 1) * ts)
        dts = ts[:-1] - ts[1:]
        ts = ts[:-1]

        for i, t in enumerate(ts):
            tval = float(t)
            timestep = torch.full((n,), tval)
            use_cfg = cfg_interval[0] < tval <= cfg_interval[1]
            ts_scale = cfg_text_scale if use_cfg else 1.0
            is_scale = cfg_img_scale if use_cfg else 1.0
            x_emb = assemble_image_x_emb(x_t, timestep, vae_position_ids, self.host, self.cfg)
            v_t = image_v_t(self._forward(x_emb, pb_cond), self.host, self.cfg)
            ctxt = image_v_t(self._forward(x_emb, pb_text), self.host, self.cfg) if ts_scale > 1.0 else None
            cimg = image_v_t(self._forward(x_emb, pb_img), self.host, self.cfg) if is_scale > 1.0 else None
            v_t = self._cfg_renorm(v_t, ctxt, cimg, ts_scale, is_scale,
                                   cfg_renorm_min, cfg_renorm_type)
            x_t = x_t - v_t * float(dts[i])
        return x_t


def build_halo_image_flow(
    ckpt_path: "str | Path",
    *,
    cfg: HaloConfig = HaloConfig(),
    max_seq_len: int = 10976,
) -> HaloImageFlowStep:
    """装载 HALO gen 孪生 + gen host → HaloImageFlowModel handle → set_weights。

    Args:
        ckpt_path: HALO ema.safetensors（bf16）。
        max_seq_len: KV cache 容量，须为 16 的倍数且不小于最长分支的
            ``prefix_len + num_image_query_tokens``；调用方可显式覆盖。
    """
    if max_seq_len % _S_CHUNK != 0:
        raise ValueError(f"max_seq_len {max_seq_len} must be a multiple of {_S_CHUNK}")
    und, gen, host = load_halo_gen_branch(Path(ckpt_path), cfg)
    # set_weights 的 cos/sin 是 base 契约占位（per-forward cos/sin 经 step_forward 覆盖,
    # 镜像 action expert）；给一份 rope_position=0 的表占位即可。
    cos0, sin0 = build_rope_tables(cfg, 0, rows=max_seq_len)
    row_mask = build_image_text_row_mask(cfg)         # [1378] 1=und(text)行 {0,1377}
    handle = int(torch.ops.rpu.halo_image_flow_create())
    runner = None
    try:
        # und/base 孪生 → set_weights（base 路径; 2 个 text 边界行用）。
        torch.ops.rpu.halo_image_flow_set_weights(
            handle,
            und.q_w, und.k_w, und.v_w, und.o_w,
            und.q_norm, und.k_norm,
            und.input_norm, und.post_norm,
            und.gate_w, und.up_w, und.down_w,
            cos0, sin0, und.final_norm_w,
            cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
            cfg.hidden_size, cfg.intermediate_size,
            cfg.rms_norm_eps, True,           # use_silu (SwiGLU)
            und.q_bias, und.k_bias, und.v_bias)
        # gen(*_moe_gen) 孪生 + 行掩码 → set_gen_weights（1376 vae 行用）。
        torch.ops.rpu.halo_image_flow_set_gen_weights(
            handle,
            gen.q_w, gen.k_w, gen.v_w, gen.o_w,
            gen.q_norm, gen.k_norm,
            gen.input_norm, gen.post_norm,
            gen.gate_w, gen.up_w, gen.down_w,
            gen.q_bias, gen.k_bias, gen.v_bias,
            gen.final_norm_w, row_mask, cfg.num_image_query_tokens)
        runner = HaloImageFlowStep(
            cfg=cfg, und=und, gen=gen, host=host, max_seq_len=max_seq_len,
            _handle=handle, _graph_cache=rpu_backend.graph.GraphCache())
        runner._handle_finalizer = weakref.finalize(
            runner, _destroy_image_flow, handle
        )
    except BaseException:
        if runner is not None:
            runner.close()
        else:
            _destroy_image_flow(handle)
        raise
    return runner
