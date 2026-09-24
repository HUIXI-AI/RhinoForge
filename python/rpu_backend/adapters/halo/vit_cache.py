"""HALO 组件 B — vit_cache_update(SigLIP-navit ViT prefill → connector → LM und 双向)。

真值契约 = ``extra/inference/export_halo_onnx_bundle.py`` 的 ``VitCacheUpdateWrapper.forward``
(部署口径 use_subtask=False / use_goal_image=False)。五段管线(逐行复现):

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
     → self.norm → ``packed_query_sequence``[1566,1536] = golden。

rope=False 订正(2026-06-23):HALO 部署/golden 硬设 ``vit_config.rope=False``
(test_halo.py:143 / smoke_test_halo.py:257),SigLIP 走学习式加性位置嵌入,无 2D-rotary。
本组件**零 rope C++、零 P 置换**;ViT 复用现有无-rope ``siglip_*`` ops(3D 预嵌入路径),
LM 复用通用 ``causal_decoder_*``(放松 is_causal=false 无 mask guard → MASK_NONE)。
SigLIP 权重 pad/swizzle 直接**复用** ``adapters.siglip._convert_siglip_weights_for_rpu``
(用 checkpoint 建最小 nn.Module skeleton 喂给真转换器,保证方向字节一致)。
"""

from __future__ import annotations

import dataclasses
import types
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

import rpu_backend
from rpu_backend.adapters.halo.runtime import _halo_execution_serialized as execution_serialized
from rpu_backend.api.cache import RPUCache
from rpu_backend.adapters.halo.runtime import (
    HALO_VISION_COMPONENT,
    HALO_VIT_LM_COMPONENT,
    HaloExecutionController,
    _HaloResources,
    _cleanup_halo_build,
    _close_halo_runner,
    _S_CHUNK,
    _coerce_halo_execution_controller,
    _plan_halo_fmb_execution,
    _record_halo_plan,
)
from rpu_backend.adapters.halo.text_cache import _dynamic_layer_kv
from rpu_backend.adapters.halo.weights import (
    validate_halo_core_layout,
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
    一致;fp64 构造再降 fp16 匹配 golden)。**为何是 step 表**:vit 多帧 prefill 里帧 f 的
    ``frame_len`` 个 token 共享 position f(bagel.py:387 ``[curr_position_id]*(num_img+2)``),
    但 LM 长段会被 auto-scan 拆多 chunk,kernel 的 ``cos_sin_start = rope_base + chunk.offset``
    (rpu_qwen3_model.h:1086)在块内逐 chunk 推进——若用 weights.build_rope_tables 的*连续*表,
    同帧不同 chunk 会拿到不同 rotary,错。本 step 表令 ``cos_sin_offset = f×frame_len`` 选中整段
    ``[f×frame_len, (f+1)×frame_len)`` 全 = rotary(f),块内 chunk.offset 推进仍读 rotary(f)。

    向后兼容:单帧(frame 在 position 0)只用块 0 = rotary(0) = identity,与旧常数 pos-0 表
    (``build_rope_tables(cfg, 0, ...)``)逐字节等价,故不回归已板证的单帧 ``run()`` 数值门。
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
    """镜像 adapters/siglip.py:535-578 的 gather:转换后逐层抽 16 weight + 4 全局 → fp16 rpu。"""
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
    torch.ops.rpu.siglip_destroy(handle)


def _destroy_causal_decoder(handle: int) -> None:
    torch.ops.rpu.causal_decoder_destroy(handle)


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
        _close_halo_runner(self, graphs=(self._lm_graph, self._vit_graph), handles={
            "_lm_handle": (self._lm_handle, _destroy_causal_decoder),
            "_vit_handle": (self._vit_handle, _destroy_siglip),
        })

    # ── ViT(native,3D 预嵌入,MASK_NONE 双向)────────────────────────────────
    def _vision_execution_plan(self, length: int):
        """Query only the installed Vision owner; its physical KV position is zero."""
        if type(length) is not int or not 0 < length <= self.vit_max_seq:
            raise ValueError("HALO Vision planning length must fit its actual cache capacity")
        return _plan_halo_fmb_execution(
            self,
            component=HALO_VISION_COMPONENT,
            stage="vision",
            logical_len=length,
            position=0,
            handle=self._vit_handle,
            native_prefix="siglip",
            resolve_stage_domain=lambda execution_len: (
                torch.ops.rpu.siglip_resolve_stage_domain(
                    self._vit_handle, int(execution_len), 1
                )
            ),
            graph_cache=self._vit_graph,
            plan_signature=(1,),
        )

    def _run_vit(
        self, packed_vit_tokens: torch.Tensor, packed_vit_position_ids: torch.Tensor
    ) -> torch.Tensor:
        """[1564,588] + pos_ids → ViT → [1564,1152](post_layernorm)。"""
        vcfg = self.vcfg
        vh = self.vit_host
        n_vit = int(packed_vit_tokens.shape[0])
        plan, planned_stage_descriptor = self._vision_execution_plan(n_vit)
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
            dyn_dims=[vcfg.vit_heads, vcfg.vit_attn_tp, 1, 0,
                      *plan.graph_key_words()],
            dtypes=[torch.float16],
        )
        with self._vit_graph.capture(sig):
            out = torch.ops.rpu.siglip_forward(
                self._vit_handle, hidden, kc, vc,
                planned_stage_descriptor,
            )
        resolved_chunk = int(
            torch.ops.rpu.siglip_get_resolved_chunk_size(self._vit_handle)
        )
        if resolved_chunk != plan.selected.stage_tuple.compute_chunk:
            raise RuntimeError(
                "HALO Vision dry/forward chunk plan drift: "
                f"dry={plan.selected.stage_tuple.compute_chunk}, "
                f"forward={resolved_chunk}, tokens={n_vit}"
            )
        torch.ops.rpu.spm_alloc_reset_temporary()   # 必须在 capture 外(否则图 non-replayable)
        out = out.clone()                            # projector_output_ 跨 forward 复用 storage
        out = out.float().cpu()
        if out.dim() == 3:
            out = out[0]
        _record_halo_plan(
            self, HALO_VISION_COMPONENT, "vision", plan,
            authority="NATIVE_FMB_DESCRIPTOR",
        )
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

    # ── text embed(host,匹配 golden bf16-autocast)───────────────────────────
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
    @execution_serialized
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
        # Admit both children before Vision can execute or reset either cache.
        self._vision_execution_plan(int(packed_vit_tokens.shape[0]))
        self._lm_execution_plan(seq_len, 0)
        packed_sequence = self._assemble_packed_sequence(
            packed_text_ids, packed_text_indexes, packed_vit_tokens,
            packed_vit_token_indexes, packed_vit_position_ids, seq_len)
        # (5) native LM:is_causal=False + 无 mask → MASK_NONE 双向。
        return self._run_lm(packed_sequence, packed_position_ids)

    # ── 底层 LM forward(MASK_NONE 双向,attend [0, position+seq);不 reset)─────
    def _lm_execution_plan(self, length: int, position: int):
        """Query the separate language handle without assembling or advancing a frame."""
        if (type(length) is not int or length <= 0
                or type(position) is not int or position < 0
                or position + length > self.lm_max_seq):
            raise ValueError("HALO Vision LM planning requires positive length and an in-capacity position")
        return _plan_halo_fmb_execution(
            self,
            component=HALO_VIT_LM_COMPONENT,
            stage="prefill",
            logical_len=length,
            position=position,
            handle=self._lm_handle,
            native_prefix="causal_decoder",
            resolve_stage_domain=lambda execution_len: (
                torch.ops.rpu.causal_decoder_resolve_prefill_stage_domain(
                    self._lm_handle, int(execution_len), position, False, 0, 0, 4, length
                )
            ),
            kv_route="DDR_REQUIRED",
            graph_cache=self._lm_graph,
            plan_signature=(False, 0, 0, 4),
        )

    @torch.no_grad()
    def _lm_forward(self, packed_sequence: torch.Tensor, position: int) -> torch.Tensor:
        """单次 LM forward 进 ``lm_cache``(**不** reset;调用方管 reset/position)。

        ``is_causal=False`` + ``attention_mask=None`` → MASK_NONE:query 全双向 attend
        ``[0, position+seq)``(含已在 cache 的前缀;rpu_qwen3_model.h:1274/1304 用
        ``kv_seq_len = ctx().position + seq_len`` 作 SDPA key 长)。新 K/V 插入于
        ``[position, position+seq)``(:1267-1272 ``ctx().position+chunk.offset``)。
        ``cos_sin_offset = position`` 选中 step-rope 表第 ``position//frame_len`` 帧块
        (块内全 = rotary(帧序号),解多 chunk「整帧同 position」)。返回 hidden ``[seq,H]``;
        KV 留在 cache 供 :meth:`get_lm_kv` 累积读出。
        """
        cfg = self.cfg
        seq_len = int(packed_sequence.shape[0])
        a6_plan, planned_stage_descriptor = self._lm_execution_plan(seq_len, position)
        ie = packed_sequence.to(torch.float16).unsqueeze(0).to(device="rpu").contiguous()
        planned_chunk_size = a6_plan.selected.stage_tuple.compute_chunk
        sig = rpu_backend.graph.GraphSignature(
            op_id="halo_vit_cache_lm",
            shapes=[seq_len, cfg.hidden_size],
            # position 进 key:KV 插入偏移 + SDPA kv_seq_len + cos_sin_offset(帧块)三者
            # 皆随它变 → 各帧独立 BUILD,禁止跨帧陈旧 replay。
            dyn_dims=[cfg.num_layers, position, seq_len, position,
                      planned_chunk_size, *a6_plan.graph_key_words()],
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
                planned_chunk_size=0,
                planned_stage_descriptor=planned_stage_descriptor,
                # Pass cos_sin_offset by name; preceding optional arguments are GR00T RoPE tensors.
            )
        # capture 退出后才读(图 end() 才写;块内读得全 0)。
        result = hidden_out[0].float().cpu()                                       # [seq_len,H]
        _record_halo_plan(
            self, HALO_VIT_LM_COMPONENT, "prefill", a6_plan,
            authority="NATIVE_FMB_DESCRIPTOR", kv_route="DDR_REQUIRED",
        )
        return result

    # ── forward + 推进 position 的单一真相源(单帧/多帧共用)──────────────────
    @torch.no_grad()
    def _forward_and_advance(self, packed_sequence: torch.Tensor, position: int) -> torch.Tensor:
        """在 ``position`` 跑一次 LM forward,**并**按写入行数推进 ``lm_cache.position``。

        ``_lm_forward`` 把 ``packed_sequence.shape[0]`` 行 KV 写入 ``[position, position+rows)``
        但不动 ``lm_cache.position``;``get_lm_kv`` 经 ``to_dynamic_cache`` 按 ``position`` trim,
        故 forward 后必须推进,否则读出空 KV。把「forward + 推进」收成单一真相源,杜绝
        单帧路径漏推进(``prefill_frames`` 有推进、``_run_lm`` 曾漏)。
        """
        out = self._lm_forward(packed_sequence, position)
        self.lm_cache.update_position(int(packed_sequence.shape[0]))             # 推进 = 本次写入行数
        return out

    # ── 单帧整管线尾(position 0,bisect / 单帧 golden 用)────────────────────
    @torch.no_grad()
    def _run_lm(
        self, packed_sequence: torch.Tensor, packed_position_ids: torch.Tensor
    ) -> torch.Tensor:
        p0 = self._rope_position(packed_position_ids)             # 常数 position(单帧 golden 全 0)
        if p0 != 0:
            raise NotImplementedError(
                f"单帧 _run_lm 假设 frame 在 position 0(step-rope 块 0 = identity);"
                f"实测 p0={p0}。多帧请走 prefill_frames(逐帧块对齐)。")
        self.lm_cache.reset()
        return self._forward_and_advance(packed_sequence, 0)     # 推进 → get_lm_kv 可读出本帧 KV

    # ── 多帧 prefill(逐帧 in-place 累积进 lm_cache → 4698 cfg_text 银行)───────
    @torch.no_grad()
    @execution_serialized
    def prefill_frames(
        self, frames: List[Dict[str, torch.Tensor]]
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """逐帧 vit prefill,KV **in-place 累积**进同一 ``lm_cache``,返回全段逐层 KV。

        部署真值(``bagel.py:forward_cache_update_vit``,**逐帧调用** inferencer.py:82):
        帧 f 的 ``frame_len`` 个 token query 对 ``[已累积前缀 + 自身]`` **全双向**
        (``is_causal=False``,varlen merged-KV qwen2_navit.py:1634),position = 帧序号 f
        (image-only cfg_text regime:ropes 从 0 起 → 帧 f 在 position f,bagel.py:387/390)。

        in-place 累积(非逐帧 read-out+reinject):帧 f forward 后 KV 留 ``[f·L, (f+1)·L)``;
        帧 f+1 直接 ``position=(f+1)·L`` forward,kernel SDPA 读 ``[0,(f+1)·L+L)`` 自然含
        前帧(:1304),省 swizzle 往返。``cos_sin_offset=position`` 命中 step 表第 f 块 =
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
        # A later frame may have a different native KV domain. Validate the
        # complete request before resetting or executing any earlier frame.
        for f, fr in enumerate(frames):
            seq = int(torch.sum(fr["packed_seqlens"]).item())
            if seq != frame_len:
                raise ValueError(
                    f"帧 {f} seq {seq} != frame_len {frame_len}(step-rope 表按定长帧块建;"
                    "变长帧需逐帧重建 rope 表)")
            self._check_frame_position(fr["packed_position_ids"], f)
            self._vision_execution_plan(int(fr["packed_vit_tokens"].shape[0]))
            self._lm_execution_plan(seq, f * frame_len)
        self.lm_cache.reset()
        for fr in frames:
            packed_sequence = self._assemble_packed_sequence(
                fr["packed_text_ids"], fr["packed_text_indexes"], fr["packed_vit_tokens"],
                fr["packed_vit_token_indexes"], fr["packed_vit_position_ids"], frame_len)
            position = self.lm_cache.position                    # = f·frame_len(reset 后递增)
            self._forward_and_advance(packed_sequence, position)  # forward + 推进(单一真相源)
        return self.get_lm_kv()

    def _check_frame_position(self, packed_position_ids: torch.Tensor, frame_idx: int) -> None:
        """断言帧 ``frame_idx`` 的 ``packed_position_ids`` 全 == ``frame_idx``。

        image-only cfg_text regime:curr_rope 从 0 起,帧 f 整段同 position f(bagel.py:387
        ``[curr_position_id]*(num_img+2)``,:390 ``new_rope=curr_position_id+1``)。这与
        step-rope 表第 f 块 = rotary(f) 对齐;非此口径(如带 text 前缀的 cond 帧 position
        8/9/10)须改表与 offset,当前不支持。
        """
        p = packed_position_ids.to(torch.long).cpu()
        if not bool((p == frame_idx).all().item()):
            raise ValueError(
                f"帧 {frame_idx} packed_position_ids 期望全 == {frame_idx}"
                f"(image-only cfg_text regime),实测 min={int(p.min())}/max={int(p.max())}。"
                "cond regime(帧带 text 前缀,position 8/9/10)须另建 step 表块。")

    # ── 读出上次 _run_lm 在 lm_cache 产的逐层 KV(真链拼接用)──────────────────
    @execution_serialized
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
                "vit_cache LM rope 假设 packed_position_ids 常数(golden 全 0);"
                f"实测非常数(min={int(p.min())},max={int(p.max())})。"
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
    rpu_execution=None,
    execution_controller: HaloExecutionController | None = None,
) -> HaloVitCacheUpdate:
    validate_halo_core_layout(cfg)
    execution_controller = _coerce_halo_execution_controller(
        rpu_execution,
        execution_controller,
        entry_point="build_halo_vit_cache",
        components=(HALO_VISION_COMPONENT, HALO_VIT_LM_COMPONENT),
    )
    for name, v in (("vit_max_seq", vit_max_seq), ("lm_max_seq", lm_max_seq)):
        if v % _S_CHUNK != 0:
            raise ValueError(f"{name} {v} must be a multiple of {_S_CHUNK}")
    for name, rows, capacity in (
        ("vision", vcfg.seq_total - vcfg.n_text, vit_max_seq),
        ("language", vcfg.seq_total, lm_max_seq),
    ):
        if not 0 < rows <= capacity:
            raise ValueError(f"HALO {name} rows {rows} exceed cache capacity {capacity}")
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
    # 单帧 run()/golden 只用块 0 = rotary(0) = identity,与旧常数 pos-0 表逐字节等价(不回归)。
    cos, sin = build_frame_rope_tables(cfg, vcfg.seq_total, lm_max_seq)

    vit_handle = int(torch.ops.rpu.siglip_create())
    resources = _HaloResources(execution_controller, handles={"_vit_handle": (vit_handle, _destroy_siglip)})
    try:
        torch.ops.rpu.siglip_set_weights(
            vit_handle, *weights,
            vcfg.vit_heads, vcfg.vit_padded_head_dim, vcfg.vit_hidden,
            ((vcfg.vit_intermediate + 127) // 128) * 128,   # padded intermediate 4352
            vcfg.vit_hidden,                                # projection_dim = identity → 1152
            vcfg.vit_eps)
        torch.ops.rpu.siglip_set_chunk_envelope(vit_handle, int(vit_max_seq), 0)
        vit_cache = RPUCache(
            num_layers=vcfg.vit_layers, batch_size=1, max_seq_len=vit_max_seq,
            num_kv_heads=vcfg.vit_heads, head_dim=vcfg.vit_padded_head_dim,
            attn_tp=vcfg.vit_attn_tp)

        lm_handle = int(torch.ops.rpu.causal_decoder_create())
        resources.handles = {"_lm_handle": (lm_handle, _destroy_causal_decoder), **resources.handles}
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
        # Use the same lm_max_seq bound as the KV cache and RoPE table.
        # Register it before SPM planning to prevent oversized chunk selection.
        # A zero chunk size preserves automatic planning.
        torch.ops.rpu.causal_decoder_set_chunk_envelope(lm_handle, int(lm_max_seq), 0)
        lm_cache = RPUCache(
            num_layers=cfg.num_layers, batch_size=1, max_seq_len=lm_max_seq,
            num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim, attn_tp=cfg.attn_tp)

        # Fix the shared Persistent floor before either owner's cold search.
        # These calls reserve fixed weight slots, not Temp, DMA, or a Graph;
        # both first forwards remain the original numerically gated BUILDs.
        torch.ops.rpu.siglip_prepare_persistent_spm(
            vit_handle, vcfg.seq_total - vcfg.n_text, 1)
        torch.ops.rpu.causal_decoder_prepare_persistent_spm(
            lm_handle, vcfg.seq_total, 0, False, 0)

        vit_graph = rpu_backend.graph.GraphCache()
        resources.graphs.append(vit_graph)
        lm_graph = rpu_backend.graph.GraphCache()
        resources.graphs.insert(0, lm_graph)
        runner = HaloVitCacheUpdate(
            cfg=cfg, vcfg=vcfg, text=text, host=host, vit_host=vit_host,
            vit_cache=vit_cache, lm_cache=lm_cache,
            vit_max_seq=vit_max_seq, lm_max_seq=lm_max_seq,
            _vit_handle=vit_handle, _lm_handle=lm_handle,
            _vit_graph=vit_graph, _lm_graph=lm_graph,
            _cos=cos, _sin=sin)
        resources.bind(runner)
        execution_controller.attach(
            runner,
            {
                HALO_VISION_COMPONENT: (runner._vit_graph,),
                HALO_VIT_LM_COMPONENT: (runner._lm_graph,),
            },
        )
    except BaseException as build_error:
        _cleanup_halo_build(resources, build_error)
        raise
    return runner
