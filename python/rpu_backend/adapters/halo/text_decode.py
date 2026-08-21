"""HALO text_decode_step（mode="und" subtask 文本自回归解码单步）→ RPU。

``forward_inference(mode="und", is_causal=True)`` 产生 post-norm hidden，随后由 host
``lm_head`` 投影为 logits，并在 ``do_sample=False`` 时取 argmax。解码复用通用
``causal_decoder_*`` op 与 HALO und/base 权重。``language_model.lm_head.weight`` 的形状为
``[151933,1536]``，词表维不满足 8-core 列并行对齐，因此词表投影使用 host
``F.linear``。

机制：
  * **seq=1 decode**：query = 1 token，attend 长前缀 P。传 ``is_causal=True`` +
    ``attention_mask=None``。``seq_len==1`` 时内层
    ``sdpa_causal = is_causal && (seq_len > 1)``
    = false → **MASK_NONE**（full-attend ``kv_seq_len = P+1``，tail 因果，**不触发 LTM
    前缀对齐拒绝**），故 C **无需** A 的显式 2D causal-prefix mask（``:1287``）。
  * **RoPE 解耦**：HALO MoT 压缩图像 token 时，KV 长度增加 ``num_img+2``，而 RoPE
    position 只增加 1。因此 decode token 的语义 position ``P_sem`` 与 KV 物理落位
    ``P_idx`` 必须解耦。复用 ``cos_sin_offset``：
    seq=1 时单 chunk ``chunk.offset=0`` ⇒ ``cos_sin_start == cos_sin_offset``，设
    ``cos_sin_offset=P_sem`` / ``position=P_idx``。
  * **lm_head host**：forward 返回 post-final-norm hidden ``[1,1,H]``，**capture 退出
    后**读（graph end() 写入，作用域内提前读取尚未完成的输出）→ host
    ``F.linear(hidden.float(), lm_head_w.float())`` → logits → argmax。

数值正确性需通过硬件端到端验证。

权重复用：``weights.py:load_halo_branches`` 的 text（und/base）分支 + 新增 host
``lm_head_w``。前缀注入复用 ``runtime._swizzle_prefix_7d``（7-D cache 布局，与 kernel
DMA 契约一致）。RoPE 表复用 A 的 ``build_seq_rope_tables``（顺序 1D RoPE，theta=1e6）。
"""
from __future__ import annotations

import dataclasses
import weakref
from pathlib import Path
from typing import List, Tuple

import torch

import rpu_backend
from rpu_backend.api.cache import RPUCache
from rpu_backend.adapters.halo.runtime import _S_CHUNK, _swizzle_prefix_7d
from rpu_backend.adapters.halo.text_cache import build_seq_rope_tables
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


@dataclasses.dataclass
class HaloTextDecodeStep:
    """HALO und 文本自回归解码-单步 runner（经 :func:`build_halo_text_decode` 构造）。

    复用通用 ``causal_decoder_*`` op + HALO und/base 权重 + host lm_head。
    ``_cos``/``_sin`` 是顺序 1D-RoPE 表（与 handle 等寿命，set_weights 已绑同一份）。
    所有权：RPU 权重/表/cache 在 runner 存活期有效；``_handle`` 由 finalize 释放。
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
    # 前缀代数：每次注入新前缀换新 RPU 张量（DMA 指针变化），编进 GraphSignature
    # 强制重 BUILD，禁止旧 graph 指针重放（同 HaloTextCacheUpdate._prefix_epoch）。
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

    def _embed_token(self, token_id: torch.Tensor) -> torch.Tensor:
        """单 decode token → ``[1, 1, H]`` fp16 RPU。

        跟 HALO runtime 路径（``generate_text`` 走 ``embed_tokens`` + 同 A 的
        autocast embed）：``F.embedding`` → ``.float()`` → ``.to(bf16).to(fp16)``。
        """
        ids = token_id.to(torch.long).reshape(1).cpu()
        emb = torch.nn.functional.embedding(ids, self.host.embed_tokens).float()  # [1,H]
        emb = emb.to(torch.bfloat16).to(torch.float16)
        return emb.unsqueeze(0).to(device="rpu").contiguous()                     # [1,1,H]

    def _inject_prefix(
        self, prefix_k: List[torch.Tensor], prefix_v: List[torch.Tensor], prefix_len: int
    ) -> None:
        """灌入 28 层 decode 前缀 K/V（CUDA NaiveCache ``[P,2,128]`` bf16/fp32）。

        前缀 = 图像 prefill + 文本 prompt prefill（标准 cond 配置下 ``P=4706``）。
        复用 ``_swizzle_prefix_7d``（7-D cache 布局，与 kernel DMA 契约一致）整体替换
        cache 张量，再 ``reset_to_position(P)``（新 token K/V 落 ``[P, P+1)``）。
        """
        cfg = self.cfg
        if len(prefix_k) != cfg.num_layers or len(prefix_v) != cfg.num_layers:
            raise ValueError(
                f"prefix layers {len(prefix_k)}/{len(prefix_v)} != {cfg.num_layers}")
        expect = (prefix_len, cfg.num_kv_heads, cfg.head_dim)
        for i, (k, v) in enumerate(zip(prefix_k, prefix_v)):
            if tuple(k.shape) != expect or tuple(v.shape) != expect:
                raise ValueError(
                    f"layer {i} prefix {tuple(k.shape)}/{tuple(v.shape)} != {expect}")
            self.cache.k_caches[i] = _swizzle_prefix_7d(
                k, self.max_seq_len, value_layout=False).to("rpu")
            self.cache.v_caches[i] = _swizzle_prefix_7d(
                v, self.max_seq_len, value_layout=True).to("rpu")
        self.cache.reset_to_position(prefix_len)
        self._prefix_epoch += 1

    @torch.no_grad()
    def decode_step(
        self,
        start_token_id: torch.Tensor,
        query_position_id: torch.Tensor,
        prefix_kv: Tuple[List[torch.Tensor], List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """一次 und 解码单步。返回 ``(hidden [H], logits [vocab], argmax_token)`` CPU fp32。

        I/O 契约：
          * ``start_token_id`` 标量/``[1]`` int —— ``packed_start_tokens``（首步 = BOS）。
          * ``query_position_id`` 标量/``[1]`` int —— ``packed_query_position_ids``，
            驱动 1D RoPE 的**语义** position ``P_sem``，与 KV 物理落位偏移解耦。
          * ``prefix_kv`` = (28 层 k, 28 层 v) 各 ``[P,2,128]`` —— decode 进入时的前缀
            KV（图像 + 文本 prompt prefill）。

        走 ``is_causal=True`` + ``attention_mask=None``（seq=1 → MASK_NONE，过入口守卫，
        无需显式 mask）；KV 插入于 ``position=P``，RoPE 由 ``cos_sin_offset=P_sem`` 驱动
        。
        """
        cfg = self.cfg
        prefix_k, prefix_v = prefix_kv
        prefix_len = int(prefix_k[0].shape[0])
        self._inject_prefix(prefix_k, prefix_v, prefix_len)
        if prefix_len + 1 > self.max_seq_len:
            raise ValueError(
                f"prefix_len+1 {prefix_len + 1} > max_seq_len {self.max_seq_len}")

        pos = query_position_id.to(torch.long).reshape(-1).cpu()
        if pos.numel() != 1:
            raise ValueError(f"query_position_id 须单元素，得 shape {tuple(pos.shape)}")
        rope_base = int(pos[0].item())                       # P_sem（RoPE 表起点）
        if rope_base < 0 or rope_base + 1 > self.max_seq_len:
            raise ValueError(
                f"rope_base {rope_base} 越界 [0,{self.max_seq_len})（RoPE 表行数不足）")

        ie = self._embed_token(start_token_id)               # [1,1,H] fp16 RPU
        # sig 含 prefix_len（KV 落位 P_idx）+ seq=1 + rope_base（RoPE 起点 P_sem，
        # BUILD 期烘焙进 cos_sin_start kernel 常量）+ prefix_epoch（DMA 指针代数）。
        # Decode 每步 P_idx/P_sem 递增，必须进 key 防陈旧 replay 用错
        # KV 落位 / 错 SDPA 长度（kv_seq_len = position+off+len 建图期烘焙）。
        sig = rpu_backend.graph.GraphSignature(
            op_id="halo_text_decode_step",
            shapes=[1, cfg.hidden_size],
            dyn_dims=[cfg.num_layers, prefix_len, 1, rope_base, self._prefix_epoch],
            dtypes=[torch.float16],
        )
        with self._graph_cache.capture(sig):
            hidden_out = torch.ops.rpu.causal_decoder_forward(
                self._handle, ie, self.cache.k_caches, self.cache.v_caches,
                None,                # attention_mask=None（无显式 mask）
                prefix_len,          # position：新 K/V 插入于 [P, P+1)
                True,                # is_causal=True；seq=1 时内层
                                     # sdpa_causal=is_causal&&(seq_len>1)=False → MASK_NONE
                                     # （full-attend P+1，tail 因果），不触发 LTM 前缀对齐检查
                None,                # position_ids（M-RoPE 专用，1D 路径忽略）
                [],                  # deepstack_dense_visual_embeds（无）
                cos_sin_offset=rope_base,  # 1D RoPE 表起点（与落位解耦）。
                # cos_sin_offset 移到 op 第 11 位(9/10=GR00T rope_cos/sin_il)→ 必须关键字传。
            )
        self.cache.update_position(1)
        # capture 退出**后**读 hidden（graph end() 已写；作用域内提前读得全 0）。
        hidden = hidden_out[0, 0].float().cpu()              # [H] post-final-norm
        logits = torch.nn.functional.linear(
            hidden, self.host.lm_head_w.float())             # [vocab] host fp32 投影
        token = int(torch.argmax(logits).item())
        return hidden, logits, token


def build_halo_text_decode(
    ckpt_path: "str | Path",
    *,
    cfg: HaloConfig = HaloConfig(),
    max_seq_len: int = 4736,
) -> HaloTextDecodeStep:
    """装载 HALO und/base 权重 + host lm_head → 通用 causal_decoder handle → set_weights。

    Args:
        ckpt_path: HALO ema.safetensors（bf16）。
        max_seq_len: KV cache 容量 + 顺序 RoPE 表行数，须 16 倍数且 ≥ 部署最长
            decode 落位（cond 前缀 4706 + decode tokens → 默认 4736）。
    """
    if max_seq_len % _S_CHUNK != 0:
        raise ValueError(f"max_seq_len {max_seq_len} must be a multiple of {_S_CHUNK}")
    text, _act, host = load_halo_branches(Path(ckpt_path), cfg, load_lm_head=True)
    # lm_head_w 是 Optional，text_decode 是唯一消费者
    # (host F.linear vocab 投影,见 decode_step)。在 adapter 边界显式报错 → 误用/缺 key 早爆,
    # 而非 decode_step 里 `lm_head_w.float()` 的晚发 AttributeError。
    if host.lm_head_w is None:
        raise ValueError(
            "build_halo_text_decode 需 lm_head,但 load_halo_branches(load_lm_head=True) 未返回 "
            "lm_head_w(ckpt 缺 language_model.lm_head.weight?)。")
    cos, sin = build_seq_rope_tables(cfg, max_seq_len)
    handle = int(torch.ops.rpu.causal_decoder_create())
    runner = None
    try:
        # 与文本 cache 路径使用相同的 und/base 权重（q_norm/k_norm Qwen3 式
        # qk-norm + q/k/v_bias Qwen2 式 qkv bias，
        # mrope_section=[] 1D RoPE、deepstack_lang_layers=[]）。**不** set_lm_head
        # （走 host F.linear；fused 路径 vocab%8 拒绝 151933）。
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
        # RPUCache and RoPE table against, so no new number is invented; it is
        # moved in front of the SPM planner, where an over-picked chunk would
        # otherwise wedge the board.
        # The envelope covers the configured cache and supported prefix length.
        torch.ops.rpu.causal_decoder_set_chunk_envelope(handle, int(max_seq_len), 0)
        cache = RPUCache(
            num_layers=cfg.num_layers, batch_size=1, max_seq_len=max_seq_len,
            num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim, attn_tp=cfg.attn_tp)
        runner = HaloTextDecodeStep(
            cfg=cfg, text=text, host=host, cache=cache, max_seq_len=max_seq_len,
            _handle=handle, _graph_cache=rpu_backend.graph.GraphCache(), _cos=cos, _sin=sin)
        runner._handle_finalizer = weakref.finalize(
            runner, _destroy_causal_decoder, handle
        )
    except BaseException:
        # 构造任一步失败 → 孤儿 handle 立即回收（同 build_halo_text_cache）。
        if runner is not None:
            runner.close()
        else:
            _destroy_causal_decoder(handle)
        raise
    return runner
