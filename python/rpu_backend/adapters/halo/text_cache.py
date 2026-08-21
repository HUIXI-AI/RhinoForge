"""HALO text_cache_update（mode="und" 文本-prompt KV-prefill）→ RPU。

该路径把文本指令编码进 ``past_key_values``，复用通用
``causal_decoder_*`` op 家族和 HALO und/base 权重。

机制：
  * **显式 2D causal-prefix mask 路径**（``is_causal=False`` + ``[seq, P+seq]`` additive
    mask），**两 regime 统一**：``is_causal=True`` LTM 对 cond 不可用——LTM 要前缀长度
    ``P`` v16 对齐且是 query 倍数，cond ``P=4698`` 不满足。借
    ``is_causal=False`` op 通道，mask 编码因果语义（Wall-OSS ``action.py:_build_action_mask``
    同式：query i 看 [0,P+i]）。
  * **1D RoPE 由 ``position`` arg 驱动**（``mrope_section=[]`` → ``has_mrope_=false``，
    ``position_ids`` 在 1D 路径不参与 kernel 调度）；用**顺序** cos/sin 表
    （行 p = position p 的 rotary 基，theta=1e6），``position=P`` 选起点。**不复用**
    weights.py 的 act 常数-position 表。
  * **embedding 跟 HALO 路径**（``F.embedding(...).float()→bf16→fp16``，保持
    bf16-autocast 的舍入语义），非 Wall-OSS 的 fp32。

两个部署 regime：
  * ``img_uncond``：进入 kv_len=0（空前缀）→ 写 N=8 个文本 token。
  * ``cond``：进入 kv_len=4698 的 image-prefix → 灌前缀 + 写 N 个文本 token。

权重复用：``weights.py:load_halo_branches`` 的 text（act=False）分支采用
``.half()→swizzle→rpu``，格式与 ``causal_decoder_set_weights`` 一致。前缀注入复用
``runtime._swizzle_prefix_7d``（7-D cache 布局，与 kernel DMA 契约一致，禁止重复实现）。
"""
from __future__ import annotations

import dataclasses
import weakref
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

import rpu_backend
from rpu_backend.api.cache import RPUCache
from rpu_backend.adapters.halo.runtime import _S_CHUNK, _swizzle_prefix_7d
from rpu_backend.adapters.halo.weights import (
    HaloBranchWeights,
    HaloConfig,
    HaloHostWeights,
    load_halo_branches,
)


def _destroy_causal_decoder(handle: int) -> None:
    """释放通用 CausalDecoderModel handle（weakref.finalize 在 GC 时调）。"""
    try:
        torch.ops.rpu.causal_decoder_destroy(handle)
    except Exception:
        pass


def _to_rpu_half(t: torch.Tensor) -> torch.Tensor:
    return t.to(dtype=torch.float16, device="rpu").contiguous()


def _dynamic_layer_kv(dc: object, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """跨 transformers 版本取 DynamicCache 第 layer 层 (keys, values)，各 ``[B,nkv,seq,hd]``。

    5.x 用 ``dc.layers[i].keys/.values``（无 ``key_cache``/``to_legacy_cache``）；
    4.x 回落 ``dc.key_cache[i]/.value_cache[i]``。
    """
    layers = getattr(dc, "layers", None)
    if layers is not None:
        return layers[layer].keys, layers[layer].values
    return dc.key_cache[layer], dc.value_cache[layer]  # type: ignore[attr-defined]


def build_seq_rope_tables(cfg: HaloConfig, max_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """**顺序** position 的 1D RoPE 表 ``[max_seq_len, head_dim/2]`` fp16 RPU（kernel 格式）。

    行 p = position p 的 rotary 基（``inv_freq`` theta=1e6，与 transformers Qwen2
    默认 rope、attention_scaling=1.0 一致）。fp64 构造后再降 fp16。
    **与 weights.build_rope_tables 的 act 常数表不同**：文本 prefill position 递增，
    kernel 用 ``cos_sin_start=position`` 索引此表（见模块 docstring 的 1D RoPE 说明）。
    """
    if max_seq_len <= 0 or max_seq_len % _S_CHUNK != 0:
        raise ValueError(
            f"max_seq_len {max_seq_len} must be a positive multiple of {_S_CHUNK}"
        )
    if cfg.head_dim <= 0 or cfg.head_dim % 2 != 0:
        raise ValueError(
            f"head_dim {cfg.head_dim} must be a positive even integer"
        )
    half = cfg.head_dim // 2
    inv_freq = 1.0 / (
        cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float64) / cfg.head_dim)
    )                                                            # [half]
    positions = torch.arange(max_seq_len, dtype=torch.float64)   # [max_seq]
    freqs = positions[:, None] * inv_freq[None, :]               # [max_seq, half]
    cos = freqs.cos().to(torch.float16)
    sin = freqs.sin().to(torch.float16)
    expected_shape = (max_seq_len, half)
    if tuple(cos.shape) != expected_shape or tuple(sin.shape) != expected_shape:
        raise RuntimeError(
            "build_seq_rope_tables produced invalid table shapes: "
            f"cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, "
            f"expected={expected_shape}"
        )
    return _to_rpu_half(cos), _to_rpu_half(sin)


def build_causal_prefix_mask(prefix_len: int, seq: int) -> torch.Tensor:
    """``[seq, prefix_len+seq]`` fp16 additive causal-prefix mask（CPU）。

    query i（绝对 position prefix_len+i）看列 ``[0, prefix_len+i]``（全前缀 + 对新
    seq-block 因果），其余 ``-inf``。prefix_len=0 时退化为纯因果 ``[seq,seq]``。
    与 Wall-OSS ``action.py:_build_action_mask`` 同式；op 的 ``sdpa_prepare_mask``
    把它拷到稳定 RPU DDR 槽（无需手动 flush）。常量（prefix_len/seq 固定）→ BUILD 期烘焙。
    """
    if prefix_len < 0 or seq <= 0:
        raise ValueError(f"prefix_len {prefix_len} >= 0 and seq {seq} > 0 required")
    m = torch.full((seq, prefix_len + seq), float("-inf"), dtype=torch.float16)
    for i in range(seq):
        m[i, : prefix_len + i + 1] = 0.0
    return m


@dataclasses.dataclass
class HaloTextCacheUpdate:
    """HALO und 文本-prompt KV-prefill runner（经 :func:`build_halo_text_cache` 构造）。

    复用通用 ``causal_decoder_*`` op + HALO und/base 权重。``_cos``/``_sin`` 是顺序
    RoPE 表（与 handle 等寿命，set_weights 已绑同一份）。所有权：RPU 权重/表/cache 在
    runner 存活期有效；``_handle`` 由 finalize 释放。
    """

    cfg: HaloConfig
    text: HaloBranchWeights
    host: HaloHostWeights
    cache: RPUCache
    max_seq_len: int
    _handle: int
    _graph_cache: "rpu_backend.graph.GraphCache"
    _cos: torch.Tensor
    _sin: torch.Tensor
    _handle_finalizer: object | None = dataclasses.field(
        init=False, default=None, repr=False
    )
    _closed: bool = dataclasses.field(init=False, default=False, repr=False)
    # GraphSignature epoch changes only when prefix storage is reassigned;
    # stable-address copy_ updates remain replayable.
    _prefix_epoch: int = dataclasses.field(init=False, default=0)

    def close(self) -> None:
        """Clear graph ownership, then retire the decoder handle once."""
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
            _destroy_causal_decoder(self._handle)

    def _embed(self, packed_text_ids: torch.Tensor) -> torch.Tensor:
        """文本 token → ``[1, seq, H]`` fp16 RPU。

        跟 HALO runtime 路径（``runtime._assemble_x_emb``）：``F.embedding`` →
        ``.float()`` → ``.to(bf16).to(fp16)``，保持 bf16-autocast 的舍入语义。
        """
        ids = packed_text_ids.to(torch.long).cpu()
        emb = torch.nn.functional.embedding(ids, self.host.embed_tokens).float()  # [seq,H]
        emb = emb.to(torch.bfloat16).to(torch.float16)
        return emb.unsqueeze(0).to(device="rpu").contiguous()                     # [1,seq,H]

    def _inject_prefix(
        self, prefix_k: List[torch.Tensor], prefix_v: List[torch.Tensor], prefix_len: int
    ) -> None:
        """灌入 28 层 cond 前缀 K/V（CUDA NaiveCache ``[P,2,128]`` bf16/fp32）。

        稳定地址前缀注入（见 ``docs/architecture.md#dma-ownership``）：
        ``_swizzle_prefix_7d`` 输出形状 == ``RPUCache.__init__`` 的原始分配形状（attn_tp=2 →
        ``_KV_SLOTS == nKVHeadChunk == 8``），故**原地 ``copy_`` 进固定地址张量（稳址）**而非换新
        张量。稳址使同 ``(prefix_len, rope_base)`` regime 的 ``GraphSignature`` 不变 → 第 2+ episode
        **命中 replay**，避免每轮重复分配和 H2D。``_prefix_epoch`` 仅在形状/设备
        不匹配回落 reassign（换 DMA 地址）时 bump，强制重 BUILD 防陈旧指针。``reset_to_position(P)``：
        K/V ``[P, P+seq)`` 由本次 forward 覆盖。
        """
        cfg = self.cfg
        if len(prefix_k) != cfg.num_layers or len(prefix_v) != cfg.num_layers:
            raise ValueError(
                f"prefix layers {len(prefix_k)}/{len(prefix_v)} != {cfg.num_layers}")
        expect = (prefix_len, cfg.num_kv_heads, cfg.head_dim)
        reassigned = False
        for i, (k, v) in enumerate(zip(prefix_k, prefix_v)):
            if tuple(k.shape) != expect or tuple(v.shape) != expect:
                raise ValueError(
                    f"layer {i} prefix {tuple(k.shape)}/{tuple(v.shape)} != {expect}")
            swz_k = _swizzle_prefix_7d(k, self.max_seq_len, value_layout=False)
            swz_v = _swizzle_prefix_7d(v, self.max_seq_len, value_layout=True)
            kc, vc = self.cache.k_caches[i], self.cache.v_caches[i]
            if (getattr(kc, "device", None) is not None and kc.device.type == "rpu"
                    and tuple(kc.shape) == tuple(swz_k.shape)
                    and tuple(vc.shape) == tuple(swz_v.shape)):
                kc.copy_(swz_k)                       # in-place 稳址 → replay
                vc.copy_(swz_v)
            else:                                     # 异构/形状不符 → 换址，必重 BUILD
                self.cache.k_caches[i] = swz_k.to("rpu")
                self.cache.v_caches[i] = swz_v.to("rpu")
                reassigned = True
        self.cache.reset_to_position(prefix_len)
        if reassigned:
            self._prefix_epoch += 1

    def _rope_base(self, packed_text_position_ids: torch.Tensor, seq: int) -> int:
        """从 ``packed_text_position_ids`` 取 1D-RoPE 起点（``cos_sin_offset``）。

        HALO MoT 压缩图像 token 的 RoPE position：cond regime 文本插入 cache 偏移
        ``prefix_len``（标准 cond 为 4698），但其 RoPE position 从压缩后的图像
        position（标准 cond 为 3）续起——与插入偏移**解耦**。本函数返回
        ``position_ids[0]`` 作 RoPE 基址，
        并断言这 seq 个 position **连续递增**（否则单 scalar 偏移不成立，须 per-row
        gather；当前部署文本段恒连续，断言失败=遇到未支持的新口径，须扩 C++）。
        """
        pos = packed_text_position_ids.to(torch.long).cpu()
        if tuple(pos.shape) != (seq,):
            raise ValueError(f"packed_text_position_ids shape {tuple(pos.shape)} != ({seq},)")
        base = int(pos[0].item())
        expected = torch.arange(base, base + seq, dtype=torch.long)
        if not torch.equal(pos, expected):
            raise ValueError(
                f"packed_text_position_ids {pos.tolist()} 非连续递增；解耦方案只支持连续"
                f" RoPE 段（须 per-row gather 才能处理非连续）")
        if base + seq > self.max_seq_len:
            raise ValueError(
                f"rope_base+seq {base + seq} > max_seq_len {self.max_seq_len}（RoPE 表行数不足）")
        return base

    @torch.no_grad()
    def prefill(
        self,
        packed_text_ids: torch.Tensor,
        packed_text_position_ids: torch.Tensor,
        prefix_kv: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """一次 und 文本 prefill。返回新文本段 KV {layer: (k,v)}，k/v 各 ``[seq,nkv,hd]`` CPU。

        I/O 契约：
          * ``packed_text_ids`` [seq] int。
          * ``packed_text_position_ids`` [seq] int——驱动 1D RoPE 的
            **语义** position（cond regime 与 cache 插入偏移解耦：[3..10] vs 插入
            [4698..4705]）。img_uncond 时退化为 [0..7]，与插入偏移巧合相等。
          * ``prefix_kv`` = (28 层 k, 28 层 v) 各 ``[P,2,128]``（cond regime 的
            CUDA 前缀）；``None`` = img_uncond（空前缀，P=0）。

        走显式 causal-prefix mask 路径（``is_causal=False``）；KV 插入于 ``position=P``，
        RoPE 由 ``cos_sin_offset=position_ids[0]`` 驱动，与 KV 插入偏移独立。
        """
        cfg = self.cfg
        seq = int(packed_text_ids.shape[0])
        if prefix_kv is not None:
            prefix_k, prefix_v = prefix_kv
            prefix_len = int(prefix_k[0].shape[0])
            self._inject_prefix(prefix_k, prefix_v, prefix_len)
        else:
            prefix_len = 0
            self.cache.reset()                       # 清空（img_uncond 空前缀）
        if prefix_len + seq > self.max_seq_len:
            raise ValueError(
                f"prefix_len+seq {prefix_len + seq} > max_seq_len {self.max_seq_len}")

        rope_base = self._rope_base(packed_text_position_ids, seq)
        ie = self._embed(packed_text_ids)            # [1,seq,H] fp16 RPU
        mask = build_causal_prefix_mask(prefix_len, seq)   # [seq, P+seq] fp16 CPU
        # sig 含 prefix_len（KV 插入偏移）+ seq + rope_base（RoPE 起点，BUILD 期烘焙进
        # cos_sin_start kernel 常量，必须进 key 防陈旧 replay）+ prefix_epoch（DMA 指针
        # 代数）→ 两 regime 独立 BUILD、新前缀强制重 BUILD。
        sig = rpu_backend.graph.GraphSignature(
            op_id="halo_text_cache_prefill",
            shapes=[seq, cfg.hidden_size],
            dyn_dims=[cfg.num_layers, prefix_len, seq, rope_base, self._prefix_epoch],
            dtypes=[torch.float16],
        )
        with self._graph_cache.capture(sig):
            torch.ops.rpu.causal_decoder_forward(
                self._handle, ie, self.cache.k_caches, self.cache.v_caches,
                mask,                # 显式 2D additive causal-prefix mask（CPU fp16）
                prefix_len,          # position：新 K/V 插入于 [P, P+seq)，mask/seq_k 用此
                False,               # is_causal=False → MASK_2D 路径（mask 编码因果）
                None,                # position_ids（M-RoPE 专用，1D 路径忽略）
                [],                  # deepstack_dense_visual_embeds（无）
                cos_sin_offset=rope_base,  # 1D RoPE 表起点（与插入偏移解耦）。
                # op 第 9/10 位是 GR00T 的 rope_cos_il/rope_sin_il,cos_sin_offset 移到第 11 位 → 必须关键字传。
            )
        self.cache.update_position(seq)
        # 只去-swizzle 尾段 [P:P+seq]，免全量 P+seq 去-swizzle + 全量 H2D。
        return self._read_tail_kv(prefix_len, seq)

    def _read_tail_kv(
        self, prefix_len: int, seq: int
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """读出尾段 ``[prefix_len, prefix_len+seq)`` 的逐层 KV（局部去-swizzle）。

        镜像 ``RPUCache.to_dynamic_cache`` 的去-swizzle 数学，但只切覆盖尾 ``seq`` 行的 seq-block
        （7-D 的 dim 1，块宽 ``sKeyChunk=16``）→ 搬运 + 计算 ∝ ``nb/sKeyVx``（部署 2/296），
        免对全 ``P+seq`` 行做去-swizzle + H2D。返回 ``{layer: (k,v)}`` 各 ``[seq,nkv,hd]`` fp16 CPU
        （与原 ``to_dynamic_cache``-切片路逐元素一致）。
        """
        cache = self.cache
        cfg = self.cfg
        chunk = cache.sKeyChunk
        blk0 = prefix_len // chunk
        blk1 = (prefix_len + seq - 1) // chunk + 1
        lo = prefix_len - blk0 * chunk                      # 尾段在去-swizzle 块内的起始行
        nb = blk1 - blk0
        eff = cache.nKVHeadChunk * cache.nKVHeadVx
        out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in range(cfg.num_layers):
            # 切 seq-block 维后才 .to("cpu") → 只搬 nb 块（非全 sKeyVx）。
            k7 = cache.k_caches[layer][:, blk0:blk1].to("cpu")
            v7 = cache.v_caches[layer][:, blk0:blk1].to("cpu")
            b = k7.shape[0]
            k = k7.permute(0, 4, 2, 1, 5, 3, 6).contiguous().reshape(
                b, eff, nb * chunk, cfg.head_dim)
            v = v7.permute(0, 4, 2, 1, 6, 3, 5).contiguous().reshape(
                b, eff, nb * cache.sValChunk, cfg.head_dim)
            k_layer = k[0, : cfg.num_kv_heads, lo:lo + seq, :]                     # [nkv,seq,hd]
            v_layer = v[0, : cfg.num_kv_heads, lo:lo + seq, :]
            out[layer] = (k_layer.permute(1, 0, 2).contiguous(),                   # [seq,nkv,hd]
                          v_layer.permute(1, 0, 2).contiguous())
        return out


def build_halo_text_cache(
    ckpt_path: "str | Path",
    *,
    cfg: HaloConfig = HaloConfig(),
    max_seq_len: int = 4736,
) -> HaloTextCacheUpdate:
    """装载 HALO und/base 权重 → 通用 causal_decoder handle → set_weights（含 qk-norm + qkv bias）。

    Args:
        ckpt_path: HALO ema.safetensors（bf16）。
        max_seq_len: KV cache 容量 + 顺序 RoPE 表行数，须 16 倍数且 ≥ 部署最长 prefill
            （cond 4698+8=4706 → 默认 4736）。
    """
    if max_seq_len % _S_CHUNK != 0:
        raise ValueError(f"max_seq_len {max_seq_len} must be a multiple of {_S_CHUNK}")
    text, _act, host = load_halo_branches(Path(ckpt_path), cfg)
    cos, sin = build_seq_rope_tables(cfg, max_seq_len)
    handle = int(torch.ops.rpu.causal_decoder_create())
    runner = None
    try:
        # 通用 set_weights 参数顺序：HALO und 同时填
        # q_norm/k_norm（Qwen3 式 qk-norm）与 q/k/v_bias（Qwen2 式 qkv bias），
        # mrope_section=[]（1D RoPE）、deepstack_lang_layers=[]。
        torch.ops.rpu.causal_decoder_set_weights(
            handle,
            text.q_w, text.k_w, text.v_w, text.o_w,
            text.q_norm, text.k_norm,
            text.input_norm, text.post_norm,
            text.gate_w, text.up_w, text.down_w,
            cos, sin, text.final_norm_w,
            cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
            cfg.hidden_size, cfg.intermediate_size,
            cfg.rms_norm_eps, True,       # use_silu (SwiGLU)
            [], [],                       # mrope_section, deepstack_lang_layers
            text.q_bias, text.k_bias, text.v_bias)
        # Certified chunk envelope. chunk=0 preserves the auto plan. The
        # length is `max_seq_len` -- the SAME bound this adapter already builds its
        # RPUCache and its RoPE table against, so no new number is invented; it is
        # moved in front of the SPM planner, where an over-picked chunk would
        # otherwise wedge the board.
        # The envelope covers the configured cache and measured prefix length.
        torch.ops.rpu.causal_decoder_set_chunk_envelope(handle, int(max_seq_len), 0)
        cache = RPUCache(
            num_layers=cfg.num_layers, batch_size=1, max_seq_len=max_seq_len,
            num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim, attn_tp=cfg.attn_tp)
        runner = HaloTextCacheUpdate(
            cfg=cfg, text=text, host=host, cache=cache, max_seq_len=max_seq_len,
            _handle=handle, _graph_cache=rpu_backend.graph.GraphCache(), _cos=cos, _sin=sin)
        runner._handle_finalizer = weakref.finalize(
            runner, _destroy_causal_decoder, handle
        )
    except BaseException:
        # 构造任一步失败 → 孤儿 handle 立即回收（同 build_halo_action_expert）。
        if runner is not None:
            runner.close()
        else:
            _destroy_causal_decoder(handle)
        raise
    return runner
