"""HALO 组件 B — vit_cache_update(SigLIP-navit ViT prefill → connector → LM und 双向)。

当前执行契约使用 ``use_subtask=False / use_goal_image=False``。五段管线：

  1. host:``embed_tokens(packed_text_ids)`` → 散射进 ``packed_sequence[1566,1536]`` 的
     ``packed_text_indexes`` 行(start/end_of_image,行 0 与 1565)。
  2. native ViT(**rope=False**):host ``patch_embedding``(Linear 588→1152)+ 学习式加性
     ``position_embedding[pos_ids]``(1152 维)→ 喂 3D 预嵌入给原生 ``siglip_forward``
     (26 层,无 rope,MASK_NONE 双向,identity projector)→ post_layernorm → [1564,1152]。
  3. host:``connector``(MLP 1152→1536→1536,gelu_pytorch_tanh)。
  4. host:+ ``vit_pos_embed.pos_embed[pos_ids]``(固定 2D sincos,1536 维)→ 散射进
     ``packed_sequence`` 的 ``packed_vit_token_indexes`` 行(1564 行)。
  5. native LM:``causal_decoder_forward``,**is_causal=False + 无 mask → MASK_NONE**
     (整段双向 full attention;packed_position_ids 全 0 → 常数 position-0 rope = identity)
     → self.norm → ``packed_query_sequence``[1566,1536]。

HALO 配置固定 ``vit_config.rope=False``：SigLIP 走学习式加性位置嵌入，无
2D-rotary。本组件不发 RoPE kernel 或额外置换；ViT 复用现有无-rope
``siglip_*`` ops(3D 预嵌入路径),
LM 复用通用 ``causal_decoder_*``(放松 is_causal=false 无 mask guard → MASK_NONE)。
SigLIP 权重 pad/swizzle 直接**复用** ``adapters.siglip._convert_siglip_weights_for_rpu``
(用 checkpoint 建最小 nn.Module skeleton 喂给真转换器,保证方向字节一致)。
"""

from __future__ import annotations

import dataclasses
import types
import weakref
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

import rpu_backend
from rpu_backend.api.cache import RPUCache
from rpu_backend.adapters.halo.runtime import _S_CHUNK
from rpu_backend.adapters.halo.text_cache import _dynamic_layer_kv
from rpu_backend.adapters.halo.weights import (
    HaloConfig,
    HaloBranchWeights,
    HaloHostWeights,
    _HaloTensorStore,
    load_halo_branches,
)
from rpu_backend.adapters.siglip import (
    _convert_siglip_weights_for_rpu,
    _convert_siglip_projector_for_rpu,
    _siglip_weight_to_rpu,
)

# ── SigLIP-SO400M-14-980-navit vision 超参(HALO drop 末层 → 26 层)─────────────
_VIT_PREFIX = "vit_model.vision_model"
_CONN_PREFIX = "connector"
_VITPOS_KEY = "vit_pos_embed.pos_embed"


@dataclasses.dataclass(frozen=True)
class VitCacheConfig:
    """组件 B 专属 vision/连接超参(LM 段沿用共享 HaloConfig)。"""

    vit_hidden: int = 1152
    vit_layers: int = 26              # SO400M 27 层 drop 末层
    vit_heads: int = 16
    vit_orig_head_dim: int = 72       # → pad 80
    vit_intermediate: int = 4304      # → pad 4352
    vit_eps: float = 1e-6
    vit_num_positions: int = 4900     # 学习式 position_embedding 表 [4900,1152]
    patch_in_dim: int = 588           # 3*14*14(patchify 后,Linear 输入)
    connector_out: int = 1536         # = LM hidden
    vit_pos_rows: int = 4900          # vit_pos_embed 表 [4900,1536]
    n_text: int = 2
    seq_total: int = 1566

    @property
    def vit_padded_head_dim(self) -> int:
        return ((self.vit_orig_head_dim + 15) // 16) * 16          # 72 → 80

    @property
    def vit_attn_tp(self) -> int:
        return min(8, self.vit_heads)                              # 8


def _round_up(n: int, mult: int) -> int:
    return ((n + mult - 1) // mult) * mult


def build_frame_rope_tables(
    cfg: HaloConfig, frame_len: int, max_seq_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """逐帧常数 position 的 step 1D-RoPE 表 ``[max_seq_len, head_dim/2]`` fp16 RPU(kernel 格式)。

    行 i = position ``floor(i / frame_len)`` 的 rotary 基(theta=cfg.rope_theta,与 text 分支
    一致;fp64 构造再降 fp16)。**为何是 step 表**:vit 多帧 prefill 里帧 f 的
    ``frame_len`` 个 token 共享 position f,
    但 LM 长段会被 auto-scan 拆多 chunk,kernel 的 ``cos_sin_start = rope_base + chunk.offset``
    在块内逐 chunk 推进——若用 weights.build_rope_tables 的*连续*表,
    同帧不同 chunk 会拿到不同 rotary,错。本 step 表令 ``cos_sin_offset = f×frame_len`` 选中整段
    ``[f×frame_len, (f+1)×frame_len)`` 全 = rotary(f),块内 chunk.offset 推进仍读 rotary(f)。

    单帧(frame 在 position 0)只用块 0 = rotary(0) = identity，与常数 pos-0 表
    ``build_rope_tables(cfg, 0, ...)`` 逐字节等价。
    """
    if max_seq_len <= 0 or max_seq_len % _S_CHUNK != 0:
        raise ValueError(
            f"max_seq_len {max_seq_len} must be a positive multiple of {_S_CHUNK}"
        )
    if frame_len <= 0:
        raise ValueError(f"frame_len {frame_len} must be > 0")
    if cfg.head_dim <= 0 or cfg.head_dim % 2 != 0:
        raise ValueError(
            f"head_dim {cfg.head_dim} must be a positive even integer"
        )
    half = cfg.head_dim // 2
    inv_freq = 1.0 / (
        cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float64) / cfg.head_dim)
    )                                                                    # [half]
    rows = torch.arange(max_seq_len, dtype=torch.long)
    positions = (rows // frame_len).to(torch.float64)                    # step: 0,0,…,1,1,…,2,…
    freqs = positions[:, None] * inv_freq[None, :]                       # [max_seq, half]
    cos = freqs.cos().to(torch.float16)
    sin = freqs.sin().to(torch.float16)
    expected_shape = (max_seq_len, half)
    if tuple(cos.shape) != expected_shape or tuple(sin.shape) != expected_shape:
        raise RuntimeError(
            "build_frame_rope_tables produced invalid table shapes: "
            f"cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, "
            f"expected={expected_shape}"
        )
    return (cos.to(device="rpu").contiguous(), sin.to(device="rpu").contiguous())


# ═══════════════════════════════════════════════════════════════════════════
# 权重加载
# ═══════════════════════════════════════════════════════════════════════════
def _half(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float16)


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float32)


@dataclasses.dataclass(frozen=True)
class _VitHostWeights:
    """host 侧(CPU)ViT/connector/pos-embed 权重。"""

    patch_w: torch.Tensor       # [1152, 588]  (Linear 权重)
    patch_b: torch.Tensor       # [1152]
    vit_pos_emb_w: torch.Tensor  # [4900, 1152] 学习式加性 pos emb(ViT 内)
    conn_fc1_w: torch.Tensor    # [1536, 1152]
    conn_fc1_b: torch.Tensor    # [1536]
    conn_fc2_w: torch.Tensor    # [1536, 1536]
    conn_fc2_b: torch.Tensor    # [1536]
    vit_pos_embed: torch.Tensor  # [4900, 1536] 固定 2D sincos(connector 后)


def _load_vit_host_weights(store: _HaloTensorStore, vcfg: VitCacheConfig) -> _VitHostWeights:
    """host 数学用的 CPU fp32 权重(patch Linear / 学习式 pos / connector / vit_pos_embed)。"""
    g = store.get
    return _VitHostWeights(
        patch_w=_f32(g(f"{_VIT_PREFIX}.embeddings.patch_embedding.weight")),
        patch_b=_f32(g(f"{_VIT_PREFIX}.embeddings.patch_embedding.bias")),
        vit_pos_emb_w=_f32(g(f"{_VIT_PREFIX}.embeddings.position_embedding.weight")),
        conn_fc1_w=_f32(g(f"{_CONN_PREFIX}.fc1.weight")),
        conn_fc1_b=_f32(g(f"{_CONN_PREFIX}.fc1.bias")),
        conn_fc2_w=_f32(g(f"{_CONN_PREFIX}.fc2.weight")),
        conn_fc2_b=_f32(g(f"{_CONN_PREFIX}.fc2.bias")),
        vit_pos_embed=_f32(g(_VITPOS_KEY)),
    )


def _build_siglip_skeleton(store: _HaloTensorStore, vcfg: VitCacheConfig):
    """从 checkpoint 建最小 SiglipVisionModel-like skeleton(fp16 权重),供真转换器原地
    pad+swizzle。只含转换器/gather 读写的属性,不依赖 HALO 模型类(零跨仓依赖)。"""
    def lin(w_key: str, b_key: str) -> nn.Linear:
        w = _half(store.get(w_key))
        b = _half(store.get(b_key))
        m = nn.Linear(w.shape[1], w.shape[0], bias=True)
        with torch.no_grad():
            m.weight = nn.Parameter(w.contiguous(), requires_grad=False)
            m.bias = nn.Parameter(b.contiguous(), requires_grad=False)
        return m

    def ln(w_key: str, b_key: str):
        return types.SimpleNamespace(
            weight=_half(store.get(w_key)), bias=_half(store.get(b_key)))

    layers = []
    for i in range(vcfg.vit_layers):
        p = f"{_VIT_PREFIX}.encoder.layers.{i}"
        self_attn = types.SimpleNamespace(
            embed_dim=vcfg.vit_hidden,
            num_heads=vcfg.vit_heads,
            q_proj=lin(f"{p}.self_attn.q_proj.weight", f"{p}.self_attn.q_proj.bias"),
            k_proj=lin(f"{p}.self_attn.k_proj.weight", f"{p}.self_attn.k_proj.bias"),
            v_proj=lin(f"{p}.self_attn.v_proj.weight", f"{p}.self_attn.v_proj.bias"),
            out_proj=lin(f"{p}.self_attn.out_proj.weight", f"{p}.self_attn.out_proj.bias"),
        )
        mlp = types.SimpleNamespace(
            fc1=lin(f"{p}.mlp.fc1.weight", f"{p}.mlp.fc1.bias"),
            fc2=lin(f"{p}.mlp.fc2.weight", f"{p}.mlp.fc2.bias"),
        )
        layers.append(types.SimpleNamespace(
            self_attn=self_attn, mlp=mlp,
            layer_norm1=ln(f"{p}.layer_norm1.weight", f"{p}.layer_norm1.bias"),
            layer_norm2=ln(f"{p}.layer_norm2.weight", f"{p}.layer_norm2.bias")))

    encoder = types.SimpleNamespace(layers=layers)
    return types.SimpleNamespace(
        encoder=encoder,
        post_layernorm=ln(f"{_VIT_PREFIX}.post_layernorm.weight",
                          f"{_VIT_PREFIX}.post_layernorm.bias"),
        _rpu_siglip_weights_converted=False)


def _gather_siglip_weights(vit, proj: nn.Linear):
    """从转换后的 SigLIP 模型逐层抽 16 个 weight + 4 个全局张量 → fp16 RPU。"""
    q, k, v, o, fc1, fc2 = [], [], [], [], [], []
    ln1w, ln1b, ln2w, ln2b = [], [], [], []
    qb, kb, vb, ob, fc1b, fc2b = [], [], [], [], [], []
    rpu = dict(dtype=torch.float16, device="rpu")
    for layer in vit.encoder.layers:
        sa, mlp = layer.self_attn, layer.mlp
        q.append(_siglip_weight_to_rpu(sa.q_proj))
        k.append(_siglip_weight_to_rpu(sa.k_proj))
        v.append(_siglip_weight_to_rpu(sa.v_proj))
        o.append(_siglip_weight_to_rpu(sa.out_proj))
        fc1.append(_siglip_weight_to_rpu(mlp.fc1))
        fc2.append(_siglip_weight_to_rpu(mlp.fc2))
        ln1w.append(layer.layer_norm1.weight.to(**rpu))
        ln1b.append(layer.layer_norm1.bias.to(**rpu))
        ln2w.append(layer.layer_norm2.weight.to(**rpu))
        ln2b.append(layer.layer_norm2.bias.to(**rpu))
        qb.append(sa.q_proj.bias.to(**rpu))
        kb.append(sa.k_proj.bias.to(**rpu))
        vb.append(sa.v_proj.bias.to(**rpu))
        ob.append(sa.out_proj.bias.to(**rpu))
        fc1b.append(mlp.fc1.bias.to(**rpu))
        fc2b.append(mlp.fc2.bias.to(**rpu))
    post_ln_w = vit.post_layernorm.weight.to(**rpu)
    post_ln_b = vit.post_layernorm.bias.to(**rpu)
    proj_w = proj.weight.to(**rpu)
    proj_b = proj.bias.to(**rpu)
    return (q, k, v, o, fc1, fc2, ln1w, ln1b, ln2w, ln2b,
            qb, kb, vb, ob, fc1b, fc2b, post_ln_w, post_ln_b, proj_w, proj_b)


# ═══════════════════════════════════════════════════════════════════════════
# 析构
# ═══════════════════════════════════════════════════════════════════════════
def _destroy_siglip(handle: int) -> None:
    try:
        torch.ops.rpu.siglip_destroy(handle)
    except Exception:
        pass


def _destroy_causal_decoder(handle: int) -> None:
    try:
        torch.ops.rpu.causal_decoder_destroy(handle)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# runner
# ═══════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass
class HaloVitCacheUpdate:
    cfg: HaloConfig
    vcfg: VitCacheConfig
    text: HaloBranchWeights
    host: HaloHostWeights
    vit_host: _VitHostWeights
    vit_cache: RPUCache
    lm_cache: RPUCache
    vit_max_seq: int
    lm_max_seq: int
    _vit_handle: int
    _lm_handle: int
    _vit_graph: "rpu_backend.graph.GraphCache"
    _lm_graph: "rpu_backend.graph.GraphCache"
    _cos: torch.Tensor
    _sin: torch.Tensor
    _vit_handle_finalizer: object | None = dataclasses.field(
        init=False, default=None, repr=False
    )
    _lm_handle_finalizer: object | None = dataclasses.field(
        init=False, default=None, repr=False
    )
    _closed: bool = dataclasses.field(init=False, default=False, repr=False)

    def close(self) -> None:
        """Clear both graph owners, then retire LM and ViT handles once."""
        if self._closed:
            return
        self._closed = True
        for graph_cache in (self._lm_graph, self._vit_graph):
            try:
                graph_cache.clear()
            except Exception:
                pass
        if self._lm_handle_finalizer is not None:
            if getattr(self._lm_handle_finalizer, "alive", False):
                self._lm_handle_finalizer()
        elif self._lm_handle >= 0:
            _destroy_causal_decoder(self._lm_handle)
        if self._vit_handle_finalizer is not None:
            if getattr(self._vit_handle_finalizer, "alive", False):
                self._vit_handle_finalizer()
        else:
            _destroy_siglip(self._vit_handle)

    # ── ViT(native,3D 预嵌入,MASK_NONE 双向)────────────────────────────────
    def _run_vit(
        self, packed_vit_tokens: torch.Tensor, packed_vit_position_ids: torch.Tensor
    ) -> torch.Tensor:
        """[1564,588] + pos_ids → ViT → [1564,1152](post_layernorm)。"""
        vcfg = self.vcfg
        vh = self.vit_host
        n_vit = int(packed_vit_tokens.shape[0])
        pos = packed_vit_position_ids.to(torch.long).cpu()
        # host:patch Linear(588→1152) + 学习式加性 position_embedding(fp32 算,fp16 喂 native)。
        x = torch.nn.functional.linear(
            packed_vit_tokens.to(torch.float32).cpu(), vh.patch_w, vh.patch_b)   # [1564,1152]
        x = x + vh.vit_pos_emb_w[pos]                                            # 学习式加性 pos
        hidden = x.to(torch.float16).unsqueeze(0).to(device="rpu").contiguous()  # [1,1564,1152]

        self.vit_cache.reset_to_position(0)
        kc = [self.vit_cache.k_caches[i] for i in range(vcfg.vit_layers)]
        vc = [self.vit_cache.v_caches[i] for i in range(vcfg.vit_layers)]
        sig = rpu_backend.graph.GraphSignature(
            op_id="halo_vit_cache_siglip",
            shapes=[n_vit, vcfg.vit_layers, vcfg.vit_padded_head_dim],
            dyn_dims=[vcfg.vit_heads, vcfg.vit_attn_tp, 1, 0],
            dtypes=[torch.float16],
        )
        with self._vit_graph.capture(sig):
            out = torch.ops.rpu.siglip_forward(self._vit_handle, hidden, kc, vc)
        torch.ops.rpu.spm_alloc_reset_temporary()   # 必须在 capture 外(否则图 non-replayable)
        out = out.clone()                            # projector_output_ 跨 forward 复用 storage
        out = out.float().cpu()
        if out.dim() == 3:
            out = out[0]
        return out[:n_vit]                           # [1564,1152]

    # ── connector + vit_pos_embed(host)──────────────────────────────────────
    def _connect(
        self, vit_out: torch.Tensor, packed_vit_position_ids: torch.Tensor
    ) -> torch.Tensor:
        """[1564,1152] → connector(1152→1536, gelu_tanh)→ + vit_pos_embed → [1564,1536] fp32。"""
        vh = self.vit_host
        pos = packed_vit_position_ids.to(torch.long).cpu()
        h = torch.nn.functional.linear(vit_out, vh.conn_fc1_w, vh.conn_fc1_b)    # [1564,1536]
        h = torch.nn.functional.gelu(h, approximate="tanh")
        h = torch.nn.functional.linear(h, vh.conn_fc2_w, vh.conn_fc2_b)          # [1564,1536]
        h = h + vh.vit_pos_embed[pos]                                            # 固定 2D sincos
        return h

    # ── text embed(host,保持 bf16-autocast 舍入语义)────────────────────────
    def _embed_text(self, packed_text_ids: torch.Tensor) -> torch.Tensor:
        ids = packed_text_ids.to(torch.long).cpu()
        emb = torch.nn.functional.embedding(ids, self.host.embed_tokens).float()
        return emb.to(torch.bfloat16).to(torch.float16)                          # [n_text,1536]

    # ── host glue(段 1-4):text embed + ViT + connector + vit_pos_embed → 散射 ──
    def _assemble_packed_sequence(
        self,
        packed_text_ids: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.Tensor,
        packed_vit_position_ids: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """组装一帧的 ``packed_sequence[seq_len, H]``(fp32 CPU),喂给 LM 前的全部 host 数学。

        (1) text embed → 散射进 ``packed_text_indexes`` 行(start/end_of_image);
        (2-4) native ViT → host connector → + ``vit_pos_embed`` → 散射进 ``packed_vit_token_indexes`` 行。
        """
        cfg = self.cfg
        packed_sequence = torch.zeros(seq_len, cfg.hidden_size, dtype=torch.float32)
        packed_sequence[packed_text_indexes.to(torch.long)] = \
            self._embed_text(packed_text_ids).float()
        vit_out = self._run_vit(packed_vit_tokens, packed_vit_position_ids)      # [1564,1152]
        vit_emb = self._connect(vit_out, packed_vit_position_ids)                # [1564,1536]
        packed_sequence[packed_vit_token_indexes.to(torch.long)] = vit_emb
        return packed_sequence

    # ── 整管线(单帧,位置 0)──────────────────────────────────────────────────
    @torch.no_grad()
    def run(
        self,
        packed_text_ids: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.Tensor,
        packed_vit_position_ids: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = int(torch.sum(packed_seqlens).item())
        if seq_len > self.lm_max_seq:
            raise ValueError(f"seq_len {seq_len} > lm_max_seq {self.lm_max_seq}")
        packed_sequence = self._assemble_packed_sequence(
            packed_text_ids, packed_text_indexes, packed_vit_tokens,
            packed_vit_token_indexes, packed_vit_position_ids, seq_len)
        # (5) native LM:is_causal=False + 无 mask → MASK_NONE 双向。
        return self._run_lm(packed_sequence, packed_position_ids)

    # ── 底层 LM forward(MASK_NONE 双向,attend [0, position+seq);不 reset)─────
    @torch.no_grad()
    def _lm_forward(self, packed_sequence: torch.Tensor, position: int) -> torch.Tensor:
        """单次 LM forward 进 ``lm_cache``(**不** reset;调用方管 reset/position)。

        ``is_causal=False`` + ``attention_mask=None`` → MASK_NONE:query 全双向 attend
        ``[0, position+seq)``(含已在 cache 的前缀；
        ``kv_seq_len = ctx().position + seq_len`` 作 SDPA key 长)。新 K/V 插入于
        ``[position, position+seq)``。
        ``cos_sin_offset = position`` 选中 step-rope 表第 ``position//frame_len`` 帧块
        (块内全 = rotary(帧序号),解多 chunk「整帧同 position」)。返回 hidden ``[seq,H]``;
        KV 留在 cache 供 :meth:`get_lm_kv` 累积读出。
        """
        cfg = self.cfg
        seq_len = int(packed_sequence.shape[0])
        ie = packed_sequence.to(torch.float16).unsqueeze(0).to(device="rpu").contiguous()
        sig = rpu_backend.graph.GraphSignature(
            op_id="halo_vit_cache_lm",
            shapes=[seq_len, cfg.hidden_size],
            # position 进 key:KV 插入偏移 + SDPA kv_seq_len + cos_sin_offset(帧块)三者
            # 皆随它变 → 各帧独立 BUILD,禁止跨帧陈旧 replay。
            dyn_dims=[cfg.num_layers, position, seq_len, position],
            dtypes=[torch.float16],
        )
        with self._lm_graph.capture(sig):
            hidden_out = torch.ops.rpu.causal_decoder_forward(
                self._lm_handle, ie, self.lm_cache.k_caches, self.lm_cache.v_caches,
                None,        # attention_mask=None → 与 is_causal=False 一起 → MASK_NONE
                position,    # position:KV 插入偏移 = 前缀长(前帧累积)
                False,       # is_causal=False → 双向 full attention(MASK_NONE)
                None,        # position_ids(M-RoPE 专用,1D 路径忽略)
                [],          # deepstack_dense_visual_embeds(无)
                cos_sin_offset=position,  # step-rope 表帧块起点(= position,与插入偏移重合)。
                # cos_sin_offset 位于 op 第 11 位，使用关键字传参。
            )
        # capture 退出后才读(图 end() 才写;块内读得全 0)。
        return hidden_out[0].float().cpu()                                       # [seq_len,H]

    # ── forward + 推进 position 的单一真相源(单帧/多帧共用)──────────────────
    @torch.no_grad()
    def _forward_and_advance(self, packed_sequence: torch.Tensor, position: int) -> torch.Tensor:
        """在 ``position`` 跑一次 LM forward,**并**按写入行数推进 ``lm_cache.position``。

        ``_lm_forward`` 把 ``packed_sequence.shape[0]`` 行 KV 写入 ``[position, position+rows)``
        但不动 ``lm_cache.position``;``get_lm_kv`` 经 ``to_dynamic_cache`` 按 ``position`` trim,
        故 forward 后必须推进,否则读出空 KV。把「forward + 推进」收成单一真相源。
        """
        out = self._lm_forward(packed_sequence, position)
        self.lm_cache.update_position(int(packed_sequence.shape[0]))             # 推进 = 本次写入行数
        return out

    # ── 单帧整管线尾(position 0)────────────────────────────────────────────
    @torch.no_grad()
    def _run_lm(
        self, packed_sequence: torch.Tensor, packed_position_ids: torch.Tensor
    ) -> torch.Tensor:
        p0 = self._rope_position(packed_position_ids)             # 单帧要求常数 position 0
        if p0 != 0:
            raise NotImplementedError(
                f"单帧 _run_lm 假设 frame 在 position 0(step-rope 块 0 = identity);"
                f"got p0={p0}。多帧请走 prefill_frames(逐帧块对齐)。")
        self.lm_cache.reset()
        return self._forward_and_advance(packed_sequence, 0)     # 推进 → get_lm_kv 可读出本帧 KV

    # ── 多帧 prefill(逐帧 in-place 累积进 lm_cache → 4698 cfg_text 银行)───────
    @torch.no_grad()
    def prefill_frames(
        self, frames: List[Dict[str, torch.Tensor]]
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """逐帧 vit prefill,KV **in-place 累积**进同一 ``lm_cache``,返回全段逐层 KV。

        当前逐帧调用契约：
        帧 f 的 ``frame_len`` 个 token query 对 ``[已累积前缀 + 自身]`` **全双向**
        (``is_causal=False``，varlen merged-KV),position = 帧序号 f
        (image-only cfg_text regime 的 RoPE 从 0 起)。

        in-place 累积(非逐帧 read-out+reinject):帧 f forward 后 KV 留 ``[f·L, (f+1)·L)``;
        帧 f+1 直接 ``position=(f+1)·L`` forward,kernel SDPA 读 ``[0,(f+1)·L+L)`` 自然含
        前帧，省 swizzle 往返。``cos_sin_offset=position`` 命中 step 表第 f 块 =
        rotary(f)(与插入偏移重合,因 cache 偏移 = 帧块起点)。

        Args:
            frames: ``len==n_frames`` 的逐帧输入,每项 dict 含 ``run`` 的同名字段
                ``packed_text_ids / packed_text_indexes / packed_vit_tokens /
                packed_vit_token_indexes / packed_vit_position_ids / packed_position_ids /
                packed_seqlens``。每帧 ``sum(packed_seqlens)`` 须 == ``vcfg.seq_total``
                (step 表按定长帧块建)。

        Returns:
            ``Dict[layer → (k, v)]``,k/v 各 ``[n_frames·frame_len, nkv, hd]`` fp16 CPU
            (= cfg_text 银行;亦作 cond regime 的 image-prefix 喂 text_cache.prefill)。
        """
        vcfg = self.vcfg
        frame_len = vcfg.seq_total
        n = len(frames)
        total = n * frame_len
        if n == 0:
            raise ValueError("frames 为空")
        if total > self.lm_max_seq:
            raise ValueError(
                f"n_frames·frame_len {total} > lm_max_seq {self.lm_max_seq}"
                f"(请用更大 lm_max_seq 构造 runner)")
        self.lm_cache.reset()
        for f, fr in enumerate(frames):
            seq = int(torch.sum(fr["packed_seqlens"]).item())
            if seq != frame_len:
                raise ValueError(
                    f"帧 {f} seq {seq} != frame_len {frame_len}(step-rope 表按定长帧块建;"
                    "变长帧需逐帧重建 rope 表)")
            self._check_frame_position(fr["packed_position_ids"], f)
            packed_sequence = self._assemble_packed_sequence(
                fr["packed_text_ids"], fr["packed_text_indexes"], fr["packed_vit_tokens"],
                fr["packed_vit_token_indexes"], fr["packed_vit_position_ids"], seq)
            position = self.lm_cache.position                    # = f·frame_len(reset 后递增)
            self._forward_and_advance(packed_sequence, position)  # forward + 推进(单一真相源)
        return self.get_lm_kv()

    def _check_frame_position(self, packed_position_ids: torch.Tensor, frame_idx: int) -> None:
        """断言帧 ``frame_idx`` 的 ``packed_position_ids`` 全 == ``frame_idx``。

        image-only cfg_text regime 的 curr_rope 从 0 起,帧 f 整段同 position f。
        这与
        step-rope 表第 f 块 = rotary(f) 对齐;非此口径(如带 text 前缀的 cond 帧 position
        8/9/10)须改表与 offset,当前不支持。
        """
        p = packed_position_ids.to(torch.long).cpu()
        if not bool((p == frame_idx).all().item()):
            raise ValueError(
                f"帧 {frame_idx} packed_position_ids 期望全 == {frame_idx}"
                f"(image-only cfg_text regime),got min={int(p.min())}/max={int(p.max())}。"
                "cond regime(帧带 text 前缀,position 8/9/10)须另建 step 表块。")

    # ── 读出上次 _run_lm 在 lm_cache 产的逐层 KV(真链拼接用)──────────────────
    def get_lm_kv(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """上次 LM forward 的逐层 KV → ``Dict[layer→(k,v)]`` 每层 ``[cur_len,2,128]`` fp16 CPU。

        镜像 ``text_cache.prefill`` 读出尾段:``to_dynamic_cache`` 去-swizzle 7-D →
        ``[B,nkv,seq,hd]``(已 trim 到 cache.position)→ ``[seq,nkv,hd]``。供
        ``prefill_frames`` 多帧累积 + prefix 组装(action 真全链)。
        """
        dc = self.lm_cache.to_dynamic_cache(device="cpu")
        out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in range(self.cfg.num_layers):
            kf, vf = _dynamic_layer_kv(dc, layer)                  # [B,nkv,seq,hd]
            out[layer] = (kf[0].permute(1, 0, 2).contiguous(),    # [seq,nkv,hd]
                          vf[0].permute(1, 0, 2).contiguous())
        return out

    def _rope_position(self, packed_position_ids: torch.Tensor) -> int:
        p = packed_position_ids.to(torch.long).cpu()
        p0 = int(p[0].item())
        if not bool((p == p0).all().item()):
            raise NotImplementedError(
                "vit_cache LM rope 假设 packed_position_ids 为常数;"
                f"got min={int(p.min())},max={int(p.max())}。"
                "非常数需 per-token position 支持,当前 cos_sin_offset(连续基址)不适用。")
        return p0


# ═══════════════════════════════════════════════════════════════════════════
# 工厂
# ═══════════════════════════════════════════════════════════════════════════
def build_halo_vit_cache(
    ckpt_path: "str | Path",
    *,
    cfg: HaloConfig = HaloConfig(),
    vcfg: VitCacheConfig = VitCacheConfig(),
    vit_max_seq: int = 1792,    # round_up(1564, 256)
    lm_max_seq: int = 4736,     # round_up(3·1566, 16)=4704 → 取 4736(留头,容 3 帧 cfg_text 银行)
) -> HaloVitCacheUpdate:
    for name, v in (("vit_max_seq", vit_max_seq), ("lm_max_seq", lm_max_seq)):
        if v % _S_CHUNK != 0:
            raise ValueError(f"{name} {v} must be a multiple of {_S_CHUNK}")
    ckpt = Path(ckpt_path)
    store = _HaloTensorStore(ckpt)

    # ── LM(text 分支 + host embed)+ ViT host 权重 ──────────────────────────
    text, _act, host = load_halo_branches(ckpt, cfg)
    vit_host = _load_vit_host_weights(store, vcfg)

    # ── ViT native 权重(skeleton → 真转换器 pad+swizzle → gather)────────────
    skeleton = _build_siglip_skeleton(store, vcfg)
    _convert_siglip_weights_for_rpu(skeleton)
    proj = nn.Linear(vcfg.vit_hidden, vcfg.vit_hidden, bias=True)   # identity projector
    with torch.no_grad():
        proj.weight = nn.Parameter(torch.eye(vcfg.vit_hidden, dtype=torch.float16),
                                   requires_grad=False)
        proj.bias = nn.Parameter(torch.zeros(vcfg.vit_hidden, dtype=torch.float16),
                                 requires_grad=False)
    _convert_siglip_projector_for_rpu(proj)
    weights = _gather_siglip_weights(skeleton, proj)

    # ── step rope 表(行 i = position floor(i/frame_len) 的 rotary;帧 f 块全 = rotary(f))──
    # 单帧 run() 只用块 0 = rotary(0) = identity，与常数 pos-0 表逐字节等价。
    cos, sin = build_frame_rope_tables(cfg, vcfg.seq_total, lm_max_seq)

    vit_handle = int(torch.ops.rpu.siglip_create())
    lm_handle = -1
    runner = None
    try:
        torch.ops.rpu.siglip_set_weights(
            vit_handle, *weights,
            vcfg.vit_heads, vcfg.vit_padded_head_dim, vcfg.vit_hidden,
            ((vcfg.vit_intermediate + 127) // 128) * 128,   # padded intermediate 4352
            vcfg.vit_hidden,                                # projection_dim = identity → 1152
            vcfg.vit_eps)
        vit_cache = RPUCache(
            num_layers=vcfg.vit_layers, batch_size=1, max_seq_len=vit_max_seq,
            num_kv_heads=vcfg.vit_heads, head_dim=vcfg.vit_padded_head_dim,
            attn_tp=vcfg.vit_attn_tp)

        lm_handle = int(torch.ops.rpu.causal_decoder_create())
        torch.ops.rpu.causal_decoder_set_weights(
            lm_handle,
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
        # length is `lm_max_seq` -- the SAME bound this adapter already builds its
        # RPUCache and RoPE table against, so no new number is invented; it is
        # moved in front of the SPM planner, where an over-picked chunk would
        # otherwise wedge the board.
        # The envelope covers the configured cache and measured prefix length.
        torch.ops.rpu.causal_decoder_set_chunk_envelope(lm_handle, int(lm_max_seq), 0)
        lm_cache = RPUCache(
            num_layers=cfg.num_layers, batch_size=1, max_seq_len=lm_max_seq,
            num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim, attn_tp=cfg.attn_tp)

        runner = HaloVitCacheUpdate(
            cfg=cfg, vcfg=vcfg, text=text, host=host, vit_host=vit_host,
            vit_cache=vit_cache, lm_cache=lm_cache,
            vit_max_seq=vit_max_seq, lm_max_seq=lm_max_seq,
            _vit_handle=vit_handle, _lm_handle=lm_handle,
            _vit_graph=rpu_backend.graph.GraphCache(), _lm_graph=rpu_backend.graph.GraphCache(),
            _cos=cos, _sin=sin)
        runner._vit_handle_finalizer = weakref.finalize(
            runner, _destroy_siglip, vit_handle
        )
        runner._lm_handle_finalizer = weakref.finalize(
            runner, _destroy_causal_decoder, lm_handle
        )
    except BaseException:
        if runner is not None:
            runner.close()
        else:
            if lm_handle >= 0:
                _destroy_causal_decoder(lm_handle)
            _destroy_siglip(vit_handle)
        raise
    return runner
