"""HALO action expert 权重装载：safetensors (ema, bf16) → 预 swizzle fp16 RPU。

权重命名（HALO ema.safetensors）：
  * 每层 text 路：``language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight``
    （q/k/v 另有 ``.bias``，o 无 bias）、``self_attn.{q,k}_norm.weight`` [128]、
    ``input_layernorm.weight`` / ``post_attention_layernorm.weight`` [1536]、
    ``mlp.{gate,up,down}_proj.weight``；
  * act 孪生：模块名加 ``_moe_act``（``q_proj_moe_act`` / ``q_norm_moe_act`` /
    ``input_layernorm_moe_act`` / ``mlp_moe_act.gate_proj`` …），bias 同样有；
  * 模型级：``language_model.model.norm.weight`` 与 ``norm_moe_act.weight``、
    ``language_model.model.embed_tokens.weight`` [151933,1536]、
    ``input_linear_layer`` 14→1536、``output_linear_layer`` 1536→14、
    ``action_time_embedder.mlp.{0,2}``。

swizzle 流程：先 ``.half()``（bf16→fp16），再做布局变换 —
col-partition = q/k/v/gate/up（``tp_col_swizzle_mc_weight``），
row-partition = o/down（``tp_row_swizzle_mc_weight``）；attention 用
``attn_tp=2``（num_kv_heads=2），MLP 用 8 核。norm/bias 不 swizzle，
原样 ``.half().to('rpu')``。

RoPE 表：theta=1e6，行=常数 position（HALO act 模式 packed_position_ids 全行
同值）。``rpu_launch_rope_spm_kernel`` 的 DDR 表 ABI 是
``[max_seq, head_dim/2]``。本模块先按 HF 展开式构造 ``[rows, head_dim]``，再取
前半列；传入 full-dim 表会改变逐行 stride，因此不受支持。

所有权：build 函数返回的张量由调用方（HaloStepRunner）持有；C++
``set_weights_halo``/``set_moe_weights`` 只保存引用计数副本，张量须存活到
handle 销毁。
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open

from rpu_backend.runtime.weights import (
    tp_col_swizzle_mc_weight,
    tp_row_swizzle_mc_weight,
)

# 注意：本模块不做硬件调用，仅构造张量。涉及硬件的入口集中在 runtime.py。

_LAYER_PREFIX = "language_model.model.layers"
_MODEL_PREFIX = "language_model.model"


@dataclasses.dataclass(frozen=True)
class HaloConfig:
    """HALO action expert 超参（来自 qwen_halo_1p5b_config + 冻结 step0 输入）。"""

    num_layers: int = 28
    hidden_size: int = 1536
    num_q_heads: int = 12
    num_kv_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 8960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    chunk_size: int = 18                      # act 模式固定 query 长度
    action_dim: int = 14
    prefix_len: int = 4706                    # KV prefix 长度
    text_rows: Tuple[int, ...] = (0, 17)      # cs=18 中的 text 行
    time_freq_dim: int = 256                  # TimestepEmbedder 频率维
    time_max_period: float = 1e4
    attn_tp: int = 2                          # = num_kv_heads（基类自动选 2 核）
    mlp_cores: int = 8
    # ── gen (image_flow_step) 专用（默认与 action 路径无关）──
    patch_latent_dim: int = 64                # vae latent 维（vae2llm 输入 / llm2vae 输出）
    max_latent_size: int = 32                 # latent_pos_embed 边长（pos_embed 行=32^2=1024）
    num_image_tokens: int = 1376              # vae latent token 数（x_t[1376,64]）
    # 双流 MoT：query = 2 text(start/end_of_image) + 1376 vae = 1378 行
    # packed_text_indexes=[0,1377]，packed_text_ids 对应
    # [start_of_image,end_of_image]=[151652,151653]。
    num_image_query_tokens: int = 1378        # = num_image_tokens + 2（含首尾 text 边界）
    image_text_rows: Tuple[int, ...] = (0, 1377)  # start_of_image=0 / end_of_image=1377
    start_of_image_id: int = 151652           # packed_text_ids[0]
    end_of_image_id: int = 151653             # packed_text_ids[1]

    @property
    def total_len(self) -> int:
        """prefix + chunk（HALO act：4706 + 18 = 4724）。"""
        return self.prefix_len + self.chunk_size

    @property
    def action_rows(self) -> Tuple[int, ...]:
        """action 行索引（text 行的补集，HALO 固定 1..16）。"""
        return tuple(i for i in range(self.chunk_size) if i not in self.text_rows)


@dataclasses.dataclass(frozen=True)
class HaloBranchWeights:
    """一条分支（text 或 act 孪生）的 28 层权重列表，全部预 swizzle fp16 RPU。

    形状契约（i = 层号；swizzle 不改形状只改布局）：
      q_w [1536,1536]  k_w/v_w [256,1536]  o_w [1536,1536]
      q_norm/k_norm [128]  input_norm/post_norm [1536]
      gate_w/up_w [8960,1536]  down_w [1536,8960]
      q_bias [1536]  k_bias/v_bias [256]
      final_norm_w [1536]（text → model.norm，act → model.norm_moe_act）
    """

    q_w: List[torch.Tensor]
    k_w: List[torch.Tensor]
    v_w: List[torch.Tensor]
    o_w: List[torch.Tensor]
    q_norm: List[torch.Tensor]
    k_norm: List[torch.Tensor]
    input_norm: List[torch.Tensor]
    post_norm: List[torch.Tensor]
    gate_w: List[torch.Tensor]
    up_w: List[torch.Tensor]
    down_w: List[torch.Tensor]
    q_bias: List[torch.Tensor]
    k_bias: List[torch.Tensor]
    v_bias: List[torch.Tensor]
    final_norm_w: torch.Tensor


@dataclasses.dataclass(frozen=True)
class HaloHostWeights:
    """boundary host 数学用的 CPU 权重（投影不进 RPU 图）。

    in/out proj 与 time-embedder 转 fp32；大尺寸 embed_tokens 保持 bf16，
    在 ``F.embedding`` 后再转 fp32。
    """

    in_proj_w: torch.Tensor    # [1536, 14] fp32
    in_proj_b: torch.Tensor    # [1536]     fp32
    out_proj_w: torch.Tensor   # [14, 1536] fp32
    out_proj_b: torch.Tensor   # [14]       fp32
    temb0_w: torch.Tensor      # [1536, 256]  fp32 (action_time_embedder.mlp.0)
    temb0_b: torch.Tensor      # [1536]       fp32
    temb2_w: torch.Tensor      # [1536, 1536] fp32 (action_time_embedder.mlp.2)
    temb2_b: torch.Tensor      # [1536]       fp32
    embed_tokens: torch.Tensor  # [vocab, 1536] bf16 CPU
    # Text decode uses a host projection because this vocabulary is not compatible
    # with the fused lm_head core partition. Loaded only when requested.
    lm_head_w: Optional[torch.Tensor]    # [vocab, 1536] bf16 CPU（仅 text_decode；否则 None）


def _to_rpu_half(t: torch.Tensor) -> torch.Tensor:
    return t.to(dtype=torch.float16, device="rpu").contiguous()


class _HaloTensorStore:
    """单文件 safetensors 读取器（HALO ema 无 index 分片）。"""

    def __init__(self, ckpt_path: Path):
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"HALO checkpoint not found: {ckpt_path}")
        # safe_open 句柄缓存（保持 mmap 存活，与 wall_oss._SafeTensorStore 同法）。
        self._handle = safe_open(str(ckpt_path), framework="pt", device="cpu")

    def get(self, name: str) -> torch.Tensor:
        try:
            tensor: torch.Tensor = self._handle.get_tensor(name)
        except Exception as exc:
            raise KeyError(f"missing tensor {name}") from exc
        return tensor


def _branch_keys(layer: int, suffix: str) -> Dict[str, str]:
    """第 layer 层某条孪生的键名。suffix ∈ {"", "_moe_act", "_moe_gen"}
    （text 主路 = ""，act = "_moe_act"，gen = "_moe_gen"）。"""
    s = suffix
    p = f"{_LAYER_PREFIX}.{layer}"
    return {
        "q_w": f"{p}.self_attn.q_proj{s}.weight",
        "k_w": f"{p}.self_attn.k_proj{s}.weight",
        "v_w": f"{p}.self_attn.v_proj{s}.weight",
        "o_w": f"{p}.self_attn.o_proj{s}.weight",
        "q_b": f"{p}.self_attn.q_proj{s}.bias",
        "k_b": f"{p}.self_attn.k_proj{s}.bias",
        "v_b": f"{p}.self_attn.v_proj{s}.bias",
        "q_norm": f"{p}.self_attn.q_norm{s}.weight",
        "k_norm": f"{p}.self_attn.k_norm{s}.weight",
        "input_norm": f"{p}.input_layernorm{s}.weight",
        "post_norm": f"{p}.post_attention_layernorm{s}.weight",
        "gate_w": f"{p}.mlp{s}.gate_proj.weight",
        "up_w": f"{p}.mlp{s}.up_proj.weight",
        "down_w": f"{p}.mlp{s}.down_proj.weight",
    }


def _load_branch(
    store: _HaloTensorStore, cfg: HaloConfig, suffix: str, final_norm_key: str
) -> HaloBranchWeights:
    """装载一条孪生（suffix ∈ {"", "_moe_act", "_moe_gen"}）的 28 层权重，
    按 .half() → swizzle → to('rpu') 的顺序处理。math 与孪生无关（仅键名换 suffix）。"""
    attn_tp = cfg.attn_tp
    mlp_cores = cfg.mlp_cores
    lists: Dict[str, List[torch.Tensor]] = {k: [] for k in (
        "q_w", "k_w", "v_w", "o_w", "q_norm", "k_norm", "input_norm",
        "post_norm", "gate_w", "up_w", "down_w", "q_bias", "k_bias", "v_bias")}
    for layer in range(cfg.num_layers):
        keys = _branch_keys(layer, suffix)
        # col-partition: q/k/v/gate/up；row-partition: o/down。
        lists["q_w"].append(_to_rpu_half(
            tp_col_swizzle_mc_weight(store.get(keys["q_w"]).half(), attn_tp)))
        lists["k_w"].append(_to_rpu_half(
            tp_col_swizzle_mc_weight(store.get(keys["k_w"]).half(), attn_tp)))
        lists["v_w"].append(_to_rpu_half(
            tp_col_swizzle_mc_weight(store.get(keys["v_w"]).half(), attn_tp)))
        lists["o_w"].append(_to_rpu_half(
            tp_row_swizzle_mc_weight(store.get(keys["o_w"]).half(), attn_tp)))
        lists["gate_w"].append(_to_rpu_half(
            tp_col_swizzle_mc_weight(store.get(keys["gate_w"]).half(), mlp_cores)))
        lists["up_w"].append(_to_rpu_half(
            tp_col_swizzle_mc_weight(store.get(keys["up_w"]).half(), mlp_cores)))
        lists["down_w"].append(_to_rpu_half(
            tp_row_swizzle_mc_weight(store.get(keys["down_w"]).half(), mlp_cores)))
        # norm/bias 不 swizzle。
        for src, dst in (("q_norm", "q_norm"), ("k_norm", "k_norm"),
                         ("input_norm", "input_norm"), ("post_norm", "post_norm"),
                         ("q_b", "q_bias"), ("k_b", "k_bias"), ("v_b", "v_bias")):
            lists[dst].append(_to_rpu_half(store.get(keys[src])))
    return HaloBranchWeights(
        final_norm_w=_to_rpu_half(store.get(final_norm_key)), **lists)


def load_halo_branches(
    ckpt_path: Path, cfg: HaloConfig, *, load_lm_head: bool = False
) -> Tuple[HaloBranchWeights, HaloBranchWeights, HaloHostWeights]:
    """装载 (text, act, host) 三组权重。

    I/O 契约：ckpt_path 指向 HALO ema.safetensors（bf16）；返回的 text/act
    分支已在 RPU DDR（fp16 预 swizzle），host 权重留 CPU。

    ``load_lm_head=False`` 时 ``host.lm_head_w`` 为 ``None``，且不要求 checkpoint
    包含 ``language_model.lm_head.weight``；text decode 构建器显式启用它。
    """
    store = _HaloTensorStore(ckpt_path)
    text = _load_branch(store, cfg, "", f"{_MODEL_PREFIX}.norm.weight")
    act = _load_branch(store, cfg, "_moe_act", f"{_MODEL_PREFIX}.norm_moe_act.weight")

    def f32(name: str) -> torch.Tensor:
        return store.get(name).float()

    host = HaloHostWeights(
        in_proj_w=f32("input_linear_layer.weight"),
        in_proj_b=f32("input_linear_layer.bias"),
        out_proj_w=f32("output_linear_layer.weight"),
        out_proj_b=f32("output_linear_layer.bias"),
        temb0_w=f32("action_time_embedder.mlp.0.weight"),
        temb0_b=f32("action_time_embedder.mlp.0.bias"),
        temb2_w=f32("action_time_embedder.mlp.2.weight"),
        temb2_b=f32("action_time_embedder.mlp.2.bias"),
        embed_tokens=store.get(f"{_MODEL_PREFIX}.embed_tokens.weight"),
        # [vocab,1536] bf16 host lm_head —— 仅 text_decode 需要；门控见 docstring。
        lm_head_w=(store.get("language_model.lm_head.weight")
                   if load_lm_head else None),
    )
    return text, act, host


def build_rope_tables(
    cfg: HaloConfig, rope_position: int, rows: "int | None" = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """常数 position 的 RoPE 表 [rows, head_dim/2] fp16 RPU（kernel 格式）。

    ``rows`` 缺省 = ``cfg.total_len``（action 路径 prefix+18）；image_flow 传
    prefix+num_image_tokens（行数须 ≥ C++ step_forward 的 prefix_len+seq 校验）。

    HALO act 模式 packed_position_ids 全 18 行同值，所以表的
    每一行都是该 position 的 cos/sin；fused 模型在 cos_sin_start=prefix_len
    取行，prefix K 来自 CUDA NaiveCache（已 rope），表只服务 query/新 K。
    先按 HF 展开构造 [rows, head_dim]（cat(c,c)），再切 [:, :head_dim/2] —
    kernel 格式 = HF 展开前半（见模块 docstring 的偏离说明）。
    """
    if rope_position < 0:
        raise ValueError(f"rope_position must be >= 0, got {rope_position}")
    half = cfg.head_dim // 2
    inv_freq = 1.0 / (
        cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float64) / cfg.head_dim)
    )                                                            # [half]
    angles = float(rope_position) * inv_freq                     # [half]
    cos_full = torch.cat([angles.cos(), angles.cos()])           # HF 展开 [head_dim]
    sin_full = torch.cat([angles.sin(), angles.sin()])
    n_rows = cfg.total_len if rows is None else rows
    cos = cos_full[:half].to(torch.float16).expand(n_rows, half)
    sin = sin_full[:half].to(torch.float16).expand(n_rows, half)
    return _to_rpu_half(cos), _to_rpu_half(sin)


def build_text_row_mask(cfg: HaloConfig) -> torch.Tensor:
    """text 行掩码 [chunk_size] fp16 RPU：text 行=1，action 行=0（C++ host 侧展开）。"""
    mask = torch.zeros(cfg.chunk_size, dtype=torch.float16)
    for row in cfg.text_rows:
        if not 0 <= row < cfg.chunk_size:
            raise ValueError(f"text row {row} out of range [0,{cfg.chunk_size})")
        mask[row] = 1.0
    return _to_rpu_half(mask)


def timestep_embedding(t: torch.Tensor, cfg: HaloConfig) -> torch.Tensor:
    """HALO TimestepEmbedder.timestep_embedding：[N] fp32 → [N,256] fp32。

    [cos||sin] 拼接、max_period=1e4，与 modeling_utils.py 公式逐项一致。
    """
    half = cfg.time_freq_dim // 2
    freqs = torch.exp(
        -math.log(cfg.time_max_period)
        * torch.arange(half, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([args.cos(), args.sin()], dim=-1)


def sinusoidal_pos_embedding(position_ids: torch.Tensor, cfg: HaloConfig) -> torch.Tensor:
    """HALO apply_sinusoidal_pos_embedding：[N] int → [N,1536] fp32，[sin||cos]。

    分母是 (half-1)（与 timestep_embedding 的 half 不同），theta=1e4。
    """
    half = cfg.hidden_size // 2
    scale = math.log(cfg.time_max_period) / (half - 1)
    freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -scale)
    args = position_ids.float()[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# gen (image_flow_step) host 权重 + 装载。boundary host 数学：
# vae2llm / time_embedder / latent_pos_embed 入图前在 host，llm2vae 出图后在 host。
# ═══════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass(frozen=True)
class HaloGenHostWeights:
    """image_flow boundary host 权重（CPU fp32；投影不进 RPU 图）。

    vae2llm (64→1536) / llm2vae (1536→64) / time_embedder.mlp (256→1536→1536, SiLU) /
    latent_pos_embed.pos_embed [1024,1536]（2D sincos 预算常量, forward = pos_embed[ids]）。
    形状契约见各字段。
    """

    vae2llm_w: torch.Tensor    # [1536, 64]   fp32
    vae2llm_b: torch.Tensor    # [1536]       fp32
    llm2vae_w: torch.Tensor    # [64, 1536]   fp32
    llm2vae_b: torch.Tensor    # [64]         fp32
    temb0_w: torch.Tensor      # [1536, 256]  fp32 (time_embedder.mlp.0)
    temb0_b: torch.Tensor      # [1536]       fp32
    temb2_w: torch.Tensor      # [1536, 1536] fp32 (time_embedder.mlp.2)
    temb2_b: torch.Tensor      # [1536]       fp32
    latent_pos_embed: torch.Tensor  # [max_latent_size^2, 1536] fp32 (PositionEmbedding.pos_embed)
    # 双流 MoT：2 个 text 边界行（start/end_of_image）的 embedding 经 embed_tokens
    # 查表：packed_sequence[text_indexes] = embed_tokens(text_ids)。
    # 保持 bf16 CPU（fp32 约 0.9 GB），assemble 期 .float() 查表。
    embed_tokens: torch.Tensor  # [vocab, 1536] bf16 CPU


def load_halo_gen_branch(
    ckpt_path: Path, cfg: HaloConfig
) -> Tuple[HaloBranchWeights, HaloBranchWeights, HaloGenHostWeights]:
    """装载 image_flow 双流 MoT 的 (und/base 孪生, gen 孪生, gen host) 权重。

    双流 MoT：2 个 text 边界行（start/end_of_image, [0,1377]）走
    **und = 无后缀 base 孪生**（``model.norm.weight`` + 每层无后缀 proj/norm/mlp）;
    1376 vae 行走 **gen = ``*_moe_gen`` 孪生**（``model.norm_moe_gen.weight``）。两孪生
    同结构同 swizzle（``_load_branch`` 仅换 suffix）。gen host = vae2llm/llm2vae/
    time_embedder/latent_pos_embed + embed_tokens（2 text 行查表用）。
    I/O：ckpt = HALO ema.safetensors（bf16）；两孪生在 RPU DDR fp16，host 留 CPU。
    """
    store = _HaloTensorStore(ckpt_path)
    und = _load_branch(store, cfg, "", f"{_MODEL_PREFIX}.norm.weight")
    gen = _load_branch(store, cfg, "_moe_gen", f"{_MODEL_PREFIX}.norm_moe_gen.weight")

    def f32(name: str) -> torch.Tensor:
        return store.get(name).float()

    host = HaloGenHostWeights(
        vae2llm_w=f32("vae2llm.weight"),
        vae2llm_b=f32("vae2llm.bias"),
        llm2vae_w=f32("llm2vae.weight"),
        llm2vae_b=f32("llm2vae.bias"),
        temb0_w=f32("time_embedder.mlp.0.weight"),
        temb0_b=f32("time_embedder.mlp.0.bias"),
        temb2_w=f32("time_embedder.mlp.2.weight"),
        temb2_b=f32("time_embedder.mlp.2.bias"),
        latent_pos_embed=f32("latent_pos_embed.pos_embed"),
        embed_tokens=store.get(f"{_MODEL_PREFIX}.embed_tokens.weight"),
    )
    return und, gen, host


def build_image_text_row_mask(cfg: HaloConfig) -> torch.Tensor:
    """image_flow 行掩码 [num_image_query_tokens] fp16 RPU：text(und)行=1，vae(gen)行=0。

    text 行 = ``cfg.image_text_rows``（start/end_of_image, 默认 {0,1377}）；其余 1376
    行 = vae（gen 孪生）。C++ ``set_gen_weights`` host 侧按 3 个 merge 宽度展开为
    [query_len,W] DDR 常量（und_mask=本掩码, gen_mask=1-本掩码）。
    """
    n = cfg.num_image_query_tokens
    mask = torch.zeros(n, dtype=torch.float16)
    for row in cfg.image_text_rows:
        if not 0 <= row < n:
            raise ValueError(f"image text row {row} out of range [0,{n})")
        mask[row] = 1.0
    return _to_rpu_half(mask)


def gen_time_embed(timestep: torch.Tensor, host: HaloGenHostWeights,
                   cfg: HaloConfig) -> torch.Tensor:
    """TimestepEmbedder(timestep) → [N,1536] fp32：sinusoidal [cos||sin] → mlp(SiLU)。

    与 ``modeling_utils.TimestepEmbedder.forward`` 逐项一致（freq=256, max_period=1e4,
    [cos||sin]；mlp = Linear → SiLU → Linear）。
    """
    t_freq = timestep_embedding(timestep, cfg)                          # [N,256]
    h = torch.nn.functional.linear(t_freq, host.temb0_w, host.temb0_b)  # [N,1536]
    h = torch.nn.functional.silu(h)
    return torch.nn.functional.linear(h, host.temb2_w, host.temb2_b)    # [N,1536]


def _vae_rows(cfg: HaloConfig) -> List[int]:
    """vae(gen)行索引 = [0,query_len) \\ image_text_rows（默认 1..1376）。"""
    text = set(cfg.image_text_rows)
    return [i for i in range(cfg.num_image_query_tokens) if i not in text]


def assemble_image_x_emb(
    x_t: torch.Tensor, timestep: torch.Tensor, vae_position_ids: torch.Tensor,
    host: HaloGenHostWeights, cfg: HaloConfig
) -> torch.Tensor:
    """host 合成 image_flow 双流 x_emb → [1,1378,1536] fp16 RPU。

    query 1378 行 = 2 text(start/end_of_image) + 1376 vae（packed_sequence 散射）:
      * text 行 {0,1377}: ``embed_tokens(start/end_of_image_id)``。
      * vae 行 1..1376: ``vae2llm(x_t) + time_embedder(timestep) + latent_pos_embed[pos]``
    x_t [1376,64]、timestep [1376]（全同值）、vae_position_ids [1376] int。
    在 host fp32 算齐再降 fp16 上板。
    """
    n = cfg.num_image_query_tokens
    seq = torch.zeros(n, cfg.hidden_size, dtype=torch.float32)            # [1378,1536]
    # text 边界行: embed_tokens 查表（embed_tokens bf16 → float）。
    text_ids = torch.tensor([cfg.start_of_image_id, cfg.end_of_image_id], dtype=torch.long)
    text_emb = torch.nn.functional.embedding(text_ids, host.embed_tokens).float()  # [2,1536]
    for row, emb in zip(cfg.image_text_rows, text_emb):
        seq[row] = emb
    # vae 行: vae2llm + time + pos。
    proj = torch.nn.functional.linear(x_t.float(), host.vae2llm_w, host.vae2llm_b)  # [1376,1536]
    t_emb = gen_time_embed(timestep, host, cfg)                            # [1376,1536]
    pos_emb = host.latent_pos_embed[vae_position_ids.long()]               # [1376,1536]
    vae_emb = proj + t_emb + pos_emb                                       # [1376,1536]
    vae_rows = torch.tensor(_vae_rows(cfg), dtype=torch.long)
    seq[vae_rows] = vae_emb
    return seq.to(torch.float16).unsqueeze(0).to(device="rpu").contiguous()  # [1,1378,1536]


def image_v_t(
    hidden: torch.Tensor, host: HaloGenHostWeights, cfg: HaloConfig
) -> torch.Tensor:
    """切 vae 行 → llm2vae → v_t [1376,64] fp32（host 出图投影）。

    hidden [1378,H]（C++ 返回全 query 末隐藏）→ 切 1376 vae 行（丢 2 个 text
    边界行）→ llm2vae [1376,64]。
    """
    vae_rows = torch.tensor(_vae_rows(cfg), dtype=torch.long)
    vae_hidden = hidden[vae_rows]                                          # [1376,H]
    return torch.nn.functional.linear(vae_hidden.float(), host.llm2vae_w, host.llm2vae_b)
