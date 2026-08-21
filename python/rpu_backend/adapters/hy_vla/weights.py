"""Hy-Embodied-0.5-VLA 权重装载：safetensors (bf16) → 预 swizzle fp16 RPU 张量。

三座子系统的权重在这里**一次性**全部就位，这是执行约束：

    所有子系统的权重必须在任何 forward 之前全部 swizzle 上 RPU。

在任一 forward 之后继续上传或转换其他子系统的权重，可能使已 BUILD 的图失效。

每个 swizzle 的顺序都是 `.half()` → swizzle → `.to('rpu')`；反了会走 fp32
分支，布局与 fp16 kernel 不兼容。
"""
from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open

from rpu_backend.quant._common import quantize_linear_per_channel
from rpu_backend.quant.int4_pack import quantize_int4_per_channel, swizzle_pack_int4
from rpu_backend.runtime.weights import (tp_col_swizzle_mc_weight,
                                         tp_row_swizzle_mc_weight,
                                         transform_linear_weight)

# ── ckpt 前缀 ──────────────────────────────────────────────────────────────
_VIT = "model.dual_tower.vlm.model.visual.vision_tower."
_MERGER = "model.dual_tower.vlm.model.visual.merger."
_VLM = "model.dual_tower.vlm.model.language_model.model."
_EXPERT = "model.dual_tower.expert.model."
# embed_tokens 与 lm_head **tied**，checkpoint 中没有独立的 embed_tokens。
_LM_HEAD = "model.dual_tower.vlm.model.language_model.lm_head.weight"


@dataclasses.dataclass(frozen=True)
class HyVlaConfig:
    """Hy-VLA-UMI 的几何；字段来自受支持 checkpoint 的 config.json。"""
    # ViT（HYViT2-400M AnyRes）
    vit_layers: int = 27
    vit_heads: int = 16
    vit_hidden: int = 1152
    vit_head_dim: int = 72            # → padded 80（RPU 要求 16 对齐）
    vit_inter: int = 4304             # → padded 4352（8 核 × 128 对齐）
    vit_seq: int = 196                # 14×14 patch
    num_cameras: int = 3              # 固定工作负载；ViT packed 路径的 KV cache 按它定长
    vit_eps: float = 1e-6
    grid: int = 7                     # merger 输出 7×7 = 49 token/图
    # VLM（HunYuanVL MoT）与 expert 共享的 attention 几何
    layers: int = 32
    num_q_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 128
    eps: float = 1e-5
    vlm_hidden: int = 2048
    vlm_inter: int = 6144
    expert_hidden: int = 1024
    expert_inter: int = 2048
    proj_dim: int = 2048              # merger 输出维 = VLM hidden
    # 序列
    suffix_len: int = 51              # 1 state + 50 action
    n_action: int = 50
    action_dim: int = 32
    num_steps: int = 10               # Euler
    # RoPE：config 写 rope_theta=10000，但 rope_scaling{dynamic, alpha=1000}
    # 使实际 base = 10000 × 1000^(hd/(hd-2)) ≈ 1.115884e7。照抄 rope_theta 会全错。
    rope_alpha: float = 1000.0
    rope_theta: float = 10000.0
    # 特殊 token id（反查得到，不依赖 tokenizer；5 个全部 max|diff| = 0）
    tok_bos: int = 120000
    tok_user: int = 120006
    tok_vision_start: int = 120684
    tok_vision_end: int = 120685
    tok_vision_split: int = 120689

    @property
    def vit_head_dim_padded(self) -> int:
        return ((self.vit_head_dim + 15) // 16) * 16          # 72 → 80

    @property
    def vit_inter_padded(self) -> int:
        return ((self.vit_inter + 127) // 128) * 128          # 4304 → 4352

    @property
    def kv_heads_rpu(self) -> int:
        """**喂给 RPU 的** KV head 数（不是模型真实的 4）。

        `RPU_HY_VLA_ATTN_TP8` 打开后把 4 个 KV head **各复制一份**成 8 个槽位
        （槽 c = 原 head `floor(c/2)`），于是 `attn_tp = min(8, 8) = 8`，
        q/k/v/o 四个投影从 4 核变为 8 核。

        **为什么复制正好对得上**：q 按 `_col(·,8)` 落成核 c ↔ q head 2c,2c+1；
        原始 gqa 分组是 q head h ↔ kv head `h//4` ⇒ 核 c 要的是 `floor(2c/4)`
        = `floor(c/2)` = 槽 c ✅。o 按 `_row(·,8)` 落成核 c ↔ 输入维
        [256c,256c+256) = 同样那两个 q head ✅。

        代价：k/v 权重与 KV cache 的 DDR 各翻一倍（cache 19→38 MB），SDPA 读 KV 翻倍。
        """
        return self.num_kv_heads * 2 if _attn_tp8() else self.num_kv_heads

    @property
    def attn_tp(self) -> int:
        return min(8, self.kv_heads_rpu)                      # 4（默认）或 8（G 打开）

    @property
    def rope_base(self) -> float:
        return self.rope_theta * self.rope_alpha ** (
            self.head_dim / (self.head_dim - 2))


def _rpu16(t: torch.Tensor) -> torch.Tensor:
    return t.to(dtype=torch.float16, device="rpu").contiguous()


def _col(w: torch.Tensor, nc: int) -> torch.Tensor:
    """col-partition swizzle（先 .half() 再 swizzle）。"""
    return _rpu16(tp_col_swizzle_mc_weight(w.half().contiguous(), nc))


def _patch_embed_cores() -> int:
    """patch_embed GEMM 的核数 —— 与 C++ `hyvla_patch_embed_cores()` 同源。

    `RPU_HY_VLA_PATCH_EMBED_MC=0` 时两边一起回单核；其余一律 8 核 col-partition。
    """
    e = os.environ.get("RPU_HY_VLA_PATCH_EMBED_MC", "")
    return 1 if e in ("0", "false", "False") else 8


def _row(w: torch.Tensor, nc: int) -> torch.Tensor:
    return _rpu16(tp_row_swizzle_mc_weight(w.half().contiguous(), nc))


def _action_mlp_cores() -> int:
    """选择 ``action_time_mlp_out`` 在 unroll 中使用的核数。

    `RPU_HY_VLA_ACTION_MLP_MC=0` 时两边一起回单核 col-swizzle（`=1` 是**开**）；其余一律 **8 核
    row(K) 切分**（C++ 侧配 all-reduce）。
    ⚠️ 两边不一致**不会报错，会静默算错** —— 改这里必须同时改那边。
    C++ 执行侧必须使用相同的核数设置。
    """
    e = os.environ.get("RPU_HY_VLA_ACTION_MLP_MC", "").strip()
    return 1 if e in ("0", "off", "false", "False") else 8


def _attn_tp8() -> bool:
    """attention TP8 总开关，默认 OFF。见 `HyVlaConfig.kv_heads_rpu`。"""
    return os.environ.get("RPU_HY_VLA_ATTN_TP8", "0").strip() in (
        "1", "true", "True", "on")


def _dup_kv(w: torch.Tensor, head_dim: int) -> torch.Tensor:
    """`[nkv*hd, K]` → `[2*nkv*hd, K]`，**块 c = 原 head `floor(c/2)`**。

    `repeat_interleave(2, dim=0)` 给出 `[h0,h0,h1,h1,h2,h2,h3,h3]` —— 正是
    `kv_heads_rpu` 里推出来的那个映射。G 关着时原样返回。
    """
    if not _attn_tp8():
        return w
    n = w.shape[0] // head_dim
    assert n * head_dim == w.shape[0], f"_dup_kv: {w.shape} 不是 head_dim {head_dim} 的整数倍"
    return w.view(n, head_dim, -1).repeat_interleave(2, dim=0).reshape(
        2 * w.shape[0], w.shape[1]).contiguous()


# 逐塔 / 逐孪生量化开关认得的全部名字。
_QUANT_NAMES = ("vit", "vlm", "expert", "vlm_text", "vlm_vision")


def _quant_tokens(var: str) -> set:
    """把 `var` 解析成一组名字。`""`/`0`/`off` ⇒ 空集；`1`/`on`/`all` ⇒ 全部。

    ⚠️ **按逗号分词精确匹配，不是子串匹配**。`vlm_text` 与 `vlm_vision`
    必须能独立选择，不能让其中一个隐式命中 `vlm`。

    未知名字**直接报错**，避免拼写错误静默退回 fp16。
    """
    v = os.environ.get(var, "").strip()
    if v in ("", "0", "off", "false", "False"):
        return set()
    if v in ("1", "on", "true", "True", "all"):
        return set(_QUANT_NAMES)
    toks = {t.strip() for t in v.split(",") if t.strip()}
    bad = toks - set(_QUANT_NAMES)
    if bad:
        raise ValueError(
            f"{var}={v!r}: 不认识的名字 {sorted(bad)}；可用 {list(_QUANT_NAMES)}"
            " 或 all/off")
    return toks


def _env_on(var: str, who: str) -> bool:
    """`var` 这个逐塔开关对 `who` 是否生效。**默认 OFF**（量化改数值）。"""
    return who in _quant_tokens(var)


def _qbits(who: str, part: str = "") -> int:
    """这座塔（或这条孪生）的权重位宽：16（fp16）/ 8（W8A16）/ 4（W4A16）。

    `RPU_HY_VLA_W4A16` 优先于 `RPU_HY_VLA_W8A16` —— 两个都点了同一座时按 4 位算，
    因为"我要更省"是明确意图，静默按 8 位跑才是意外。

    `part` 只给 VLM 用（`"text"` / `"vision"`）：先看有没有**逐孪生**点名
    （`vlm_text` / `vlm_vision`），没有再回落到整座 `vlm`。
    ⇒ `=vlm` 与从前完全一样（两条孪生一起量化），`=vlm_vision` 只量化 `_v` 那条。
    两条孪生走的是**两个互相独立的 C++ setter**
    （`set_weights_hyvla` / `set_moe_weights`，各自按"有没有给 scale 列表"
    独立校验 dtype），所以混着来不需要动 C++。

    ⚠️ **ViT 走不了 int4**：int4 的 row-partition swizzle 要求
    `K % (64 × num_cores) == 0`，而 ViT 的 `o_proj`(K=1280) 与 `fc2`(K=4352)
    对 512 都不整除；启用时会在 `swizzle_pack_int4` 中直接抛出 ValueError。
    """
    for var, bits in (("RPU_HY_VLA_W4A16", 4), ("RPU_HY_VLA_W8A16", 8)):
        toks = _quant_tokens(var)
        if part and f"{who}_{part}" in toks:
            return bits
        if who in toks:
            return bits
    return 16


def _q8(w: torch.Tensor, partition: int, nc: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp16 `[N,K]` → (int8 已 swizzle 的 RPU 权重, fp16 `[N]` RPU scale)。

    per-output-channel 对称 int8（`quant/_common.py`），kernel 侧由
    `rpu_linear.cpp` 按 `weight.scalar_type()==at::kChar` 自动派发到
    `PARALLEL_LINEAR_W8A16`，scale 传**全长 N**（partition 由 reg18 告诉 kernel，
    不需要按核切片）。

    三条**不报错、只会静默算错**的次序/口径约束：

    · **先量化再 swizzle。** 量化是逐输出通道（行）取 `amax`，而 swizzle 会把行
      按核打散重排 —— 反过来做，`amax` 就跨了输出通道。
    · **必须走 `transform_linear_weight`，不能用本模块的 `_col`/`_row`。**
      前者按 `element_size()` 取 `num_ele_32B`（int8 = 32，fp16 = 16），
      后两个写死 `dwidth=2` ⇒ 喂 int8 进去布局整个错位。
    · **`_dup_kv` 与量化谁先谁后都行。** per-row 量化对行复制是不变的
      （复制出来的行 `amax` 相同 ⇒ scale 相同 ⇒ int8 相同），所以这里统一放在
      `_dup_kv` **之后**，与 fp16 路径看到的是同一个张量。
    """
    w8, sc = quantize_linear_per_channel(w.half().contiguous())
    return (transform_linear_weight(w8.contiguous(), partition=partition,
                                    num_cores=nc).to("rpu").contiguous(),
            sc.to(torch.float16).to("rpu").contiguous())


def _q4(w: torch.Tensor, partition: int, nc: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp16 `[N,K]` → (nibble-packed uint8 `[N,K/2]` RPU 权重, fp16 `[N]` RPU scale)。

    W4A16 is opt-in and disabled by default.

    与 `_q8` 的三条次序约束完全相同，另加两条 int4 独有的：

    · **swizzle 粒度是 64**（`num_ele_32B = 32 字节 / 0.5 字节`），由
      `quant/int4_pack.py` 显式写死 —— **不能**走 `transform_linear_weight`，
      它按 `element_size()` 推，而 int4 装在 int8 容器里会推成 32。
    · **kernel 收的是 2D `[N, K/2]`**，
      而 `swizzle_pack_int4` 返回的是扁平 1D ⇒ 这里 reshape。
    """
    v4, sc = quantize_int4_per_channel(w.half().contiguous())
    n, k = v4.shape
    packed = swizzle_pack_int4(v4, partition, nc).view(n, k // 2)
    return (packed.to("rpu").contiguous(),
            sc.to(torch.float16).to("rpu").contiguous())


def _quant(w: torch.Tensor, partition: int, nc: int, bits: int):
    """按位宽派发。返回 (RPU 权重, RPU scale)；bits==16 时 scale 为 None。"""
    if bits == 4:
        return _q4(w, partition, nc)
    if bits == 8:
        return _q8(w, partition, nc)
    return (_col if partition else _row)(w, nc), None


@dataclasses.dataclass
class HyVlaWeights:
    """三座子系统的 RPU 权重 + 留在 host 的 boundary 权重。

    所有 RPU 张量在本对象构造完毕时就已全部驻留 DDR —— 见模块 docstring 的顺序约束。
    """
    cfg: HyVlaConfig
    vit: Dict[str, List[torch.Tensor]]          # 27 层，已 pad + swizzle
    vit_proj1_w: torch.Tensor                   # merger.proj1（fused 到 C++ 尾部）
    vit_proj1_b: torch.Tensor
    vit_patch_gemm_w: torch.Tensor              # patch_embed 折成 im2col GEMM
    vit_pos_fused: torch.Tensor                 # pos_embed + conv.bias 预融合
    merger_rest: Dict[str, torch.Tensor]        # DwPooler→GELU→proj2（全 RPU）
    vlm_text: Dict[str, List[torch.Tensor]]     # VLM text 塔
    vlm_vision: Dict[str, List[torch.Tensor]]   # VLM `_v` 塔
    vlm_final_norm: torch.Tensor
    expert: Dict[str, List[torch.Tensor]]       # expert（单塔，只 `_v` 半边）
    expert_final_norm: torch.Tensor
    tok_emb: torch.Tensor                       # host fp32：tied embed_tokens
    host: Dict[str, torch.Tensor]               # host fp32：boundary 投影


def _build_vit(w: Dict[str, torch.Tensor], cfg: HyVlaConfig
               ) -> Dict[str, List[torch.Tensor]]:
    """ViT 27 层：fused-qkv 拆分 → head_dim/intermediate padding → swizzle。

    `scale` 用**原始** head_dim(72) 而非 padded(80) —— padding 只是补零占位，
    不参与 softmax 缩放。这一点由 C++ 侧按 orig head_dim 处理。
    """
    H, D, Dp = cfg.vit_heads, cfg.vit_head_dim, cfg.vit_head_dim_padded
    Hid, I, Ip = cfg.vit_hidden, cfg.vit_inter, cfg.vit_inter_padded

    def split_pad(qkv_w, qkv_b, i):
        # fused [3*H, Hid] 的行布局是 [3, heads, head_dim]
        W = qkv_w.view(3, H, D, Hid)[i]
        B = qkv_b.view(3, H, D)[i]
        W = torch.cat([W, W.new_zeros(H, Dp - D, Hid)], 1)
        B = torch.cat([B, B.new_zeros(H, Dp - D)], 1)
        return W.reshape(H * Dp, Hid).contiguous(), B.reshape(-1).contiguous()

    out = {k: [] for k in ("qw kw vw ow f1w f2w n1w n1b n2w n2b "
                           "qb kb vb ob f1b f2b "
                           "qws kws vws ows f1ws f2ws").split()}
    bits = _qbits("vit")
    for li in range(cfg.vit_layers):
        b = f"blocks.{li}."
        qw, qb = split_pad(w[b + "attn.qkv.weight"], w[b + "attn.qkv.bias"], 0)
        kw, kb = split_pad(w[b + "attn.qkv.weight"], w[b + "attn.qkv.bias"], 1)
        vw, vb = split_pad(w[b + "attn.qkv.weight"], w[b + "attn.qkv.bias"], 2)
        ow = w[b + "attn.proj.weight"].view(Hid, H, D)
        ow = torch.cat([ow, ow.new_zeros(Hid, H, Dp - D)], 2).reshape(Hid, H * Dp)
        f1w = torch.cat([w[b + "mlp.fc1.weight"],
                         w[b + "mlp.fc1.weight"].new_zeros(Ip - I, Hid)], 0)
        f1b = torch.cat([w[b + "mlp.fc1.bias"],
                         w[b + "mlp.fc1.bias"].new_zeros(Ip - I)], 0)
        f2w = torch.cat([w[b + "mlp.fc2.weight"],
                         w[b + "mlp.fc2.weight"].new_zeros(Hid, Ip - I)], 1)
        # col = QKV/fc1（按输出维切核）；row = o_proj/fc2（输出跨核 reduce-sum）
        for n, t, p in (("qw", qw, 1), ("kw", kw, 1), ("vw", vw, 1),
                        ("ow", ow.contiguous(), 0), ("f1w", f1w, 1), ("f2w", f2w, 0)):
            # q/k/v 的 head_dim 补零行、fc1 的 4304→4352 补零行都是整行 0
            # ⇒ amax=0 ⇒ scale 被 clamp 到 fp16 tiny、量化值全 0 ⇒ 输出仍是 0。
            q, sc = _quant(t, p, 8, bits)
            out[n].append(q)
            if sc is not None:
                out[n + "s"].append(sc)
        for n, t in (("qb", qb), ("kb", kb), ("vb", vb),
                     ("ob", w[b + "attn.proj.bias"]), ("f1b", f1b),
                     ("f2b", w[b + "mlp.fc2.bias"]),
                     ("n1w", w[b + "norm1.weight"]), ("n1b", w[b + "norm1.bias"]),
                     ("n2w", w[b + "norm2.weight"]), ("n2b", w[b + "norm2.bias"])):
            out[n].append(_rpu16(t))
    return out


def _build_mot_branch(w: Dict[str, torch.Tensor], cfg: HyVlaConfig, suffix: str
                      ) -> Dict[str, List[torch.Tensor]]:
    """VLM 的一条孪生。suffix "" = text 塔，"_v" = vision 塔。

    q/k head-norm **无 `_v` 版本**（两塔共享），两条孪生取同一张量。
    """
    d = {k: [] for k in ("q_w k_w v_w o_w q_norm k_norm input_norm post_norm "
                         "gate_w up_w down_w "
                         "q_ws k_ws v_ws o_ws gate_ws up_ws down_ws").split()}
    tp, mc = cfg.attn_tp, 8
    # ⚠️ MoT 两条孪生**各量各的 scale** —— text 与 `_v` 是两组不同的权重。
    # 位宽可以逐孪生选择（例如 `RPU_HY_VLA_W8A16=vlm_vision`），因为两条走
    # 独立的 C++ setter。
    bits = _qbits("vlm", "vision" if suffix else "text")
    for li in range(cfg.layers):
        b, sa, mlp = f"layers.{li}.", f"layers.{li}.self_attn.", f"layers.{li}.mlp{suffix}."
        # (影子键, 原始 [N,K], partition: 1=col/0=row, 核数)
        for key, raw, p, nc in (
                ("q_w",    w[f"{sa}q_proj{suffix}.weight"],                        1, tp),
                ("k_w",    _dup_kv(w[f"{sa}k_proj{suffix}.weight"], cfg.head_dim), 1, tp),
                ("v_w",    _dup_kv(w[f"{sa}v_proj{suffix}.weight"], cfg.head_dim), 1, tp),
                ("o_w",    w[f"{sa}o_proj{suffix}.weight"],                        0, tp),
                ("gate_w", w[f"{mlp}gate_proj.weight"],                            1, mc),
                ("up_w",   w[f"{mlp}up_proj.weight"],                              1, mc),
                ("down_w", w[f"{mlp}down_proj.weight"],                            0, mc)):
            q, sc = _quant(raw, p, nc, bits)
            d[key].append(q)
            if sc is not None:
                d[key[:-1] + "ws"].append(sc)
        d["q_norm"].append(_rpu16(w[f"{sa}query_layernorm.weight"]))   # 共享
        d["k_norm"].append(_rpu16(w[f"{sa}key_layernorm.weight"]))     # 共享
        d["input_norm"].append(_rpu16(w[f"{b}input_layernorm{suffix}.weight"]))
        d["post_norm"].append(_rpu16(w[f"{b}post_attention_layernorm{suffix}.weight"]))
    return d


def _build_expert(e: Dict[str, torch.Tensor], v: Dict[str, torch.Tensor],
                  cfg: HyVlaConfig) -> Dict[str, List[torch.Tensor]]:
    """expert 32 层。**单塔**：只取 `_v` 半边。

    两处只会静默算错、不会报错的地方：

    · expert 的 `modality_mask` 恒 True，`mask_apply` 始终选择 `_v` 分支。
    · **q/k head-norm 必须取 VLM 层的那一份**，两塔共享
      `query_layernorm` / `key_layernorm`；expert 自己的同名张量不参与执行。
    """
    d = {k: [] for k in ("q_w k_w v_w o_w q_norm k_norm input_norm post_norm "
                         "gate_w up_w down_w "
                         "q_ws k_ws v_ws o_ws gate_ws up_ws down_ws").split()}
    tp, mc = cfg.attn_tp, 8
    bits = _qbits("expert")
    for li in range(cfg.layers):
        b, sa = f"layers.{li}.", f"layers.{li}.self_attn."
        for key, raw, p, nc in (
                ("q_w",    e[f"{sa}q_proj_v.weight"],                        1, tp),
                ("k_w",    _dup_kv(e[f"{sa}k_proj_v.weight"], cfg.head_dim), 1, tp),
                ("v_w",    _dup_kv(e[f"{sa}v_proj_v.weight"], cfg.head_dim), 1, tp),
                ("o_w",    e[f"{sa}o_proj_v.weight"],                        0, tp),
                ("gate_w", e[f"{b}mlp_v.gate_proj.weight"],                  1, mc),
                ("up_w",   e[f"{b}mlp_v.up_proj.weight"],                    1, mc),
                ("down_w", e[f"{b}mlp_v.down_proj.weight"],                  0, mc)):
            q, sc = _quant(raw, p, nc, bits)
            d[key].append(q)
            if sc is not None:
                d[key[:-1] + "ws"].append(sc)
        d["q_norm"].append(_rpu16(v[f"{sa}query_layernorm.weight"]))   # ← VLM 的
        d["k_norm"].append(_rpu16(v[f"{sa}key_layernorm.weight"]))     # ← VLM 的
        d["input_norm"].append(_rpu16(e[f"{b}input_layernorm_v.weight"]))
        d["post_norm"].append(_rpu16(e[f"{b}post_attention_layernorm_v.weight"]))
    return d


def sample_vit_pos_embedding(pos_embed: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """ckpt 的 `pos_embed` → 本次网格的 `[1, h*w, C]` fp32 位置编码。

    使用模型的 grid 构造与 `sample_positional_embedding` 重采样约定。checkpoint 中
    存储的是
    **128×128** 的 `pos_embed`（`[1, 16384, 1152]`），而 224×224 / patch16 的图只有
    14×14=196 个 patch，所以每帧都要按归一化网格 `grid_sample` 采一次。

    ⚠️ 这是 `grid_sample`（`padding_mode="border"`），**不是** `F.interpolate`。
    `rescale_positional_embedding` 使用 interpolate，`sample_positional_embedding`
    使用 grid_sample；本模型的 ViT 使用后者。两者结果不同，不能互换。

    ⚠️ 返回 **fp32**，避免在重采样结果上引入额外的 bf16 量化误差。
    """
    side = int(round(pos_embed.shape[1] ** 0.5))
    if side * side != pos_embed.shape[1]:
        raise ValueError(f"pos_embed 的 token 数 {pos_embed.shape[1]} 不是完全平方数")
    pe_2d = pos_embed[0].T.contiguous().view(1, -1, side, side)

    # 归一化网格：每格取格心（±margin 把边界内缩半格），meshgrid 默认 "ij"。
    # `stack((meshy, meshx))` 的顺序不能反 —— grid_sample 吃的是 (x=宽, y=高)。
    dh = torch.linspace(-1 + 1.0 / h, 1 - 1.0 / h, steps=h, dtype=torch.float32)
    dw = torch.linspace(-1 + 1.0 / w, 1 - 1.0 / w, steps=w, dtype=torch.float32)
    meshx, meshy = torch.meshgrid(dh, dw, indexing="ij")
    grid = torch.stack((meshy, meshx), 2).reshape(1, h * w, 1, 2)

    out = F.grid_sample(pe_2d.float(), grid, mode="bilinear",
                        align_corners=False, padding_mode="border")
    return out.view(1, -1, h * w).transpose(1, 2).contiguous()


def load_hy_vla_weights(ckpt_path: "str | Path",
                        pos_embedding: torch.Tensor | None = None,
                        cfg: HyVlaConfig = HyVlaConfig()) -> HyVlaWeights:
    """一次性装载全部权重（见模块 docstring 的顺序约束）。

    Args:
        ckpt_path: `model.safetensors`（4.5B，全 bf16）。
        pos_embedding: ViT 的 `[1, 196, 1152]` fp32 位置编码。**默认 `None` =
            从 ckpt 自己的 `pos_embed` 重采样**（`sample_vit_pos_embedding`），
            无需额外的位置编码文件。传入张量则原样使用。
    """
    p = Path(ckpt_path)
    with safe_open(p, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        pick = lambda pre: {k[len(pre):]: f.get_tensor(k).float()
                            for k in keys if k.startswith(pre)}
        vit_w, merger_w = pick(_VIT), pick(_MERGER)
        vlm_w, expert_w = pick(_VLM), pick(_EXPERT)
        tok_emb = f.get_tensor(_LM_HEAD).float()
        host = {k[len("model."):]: f.get_tensor(k).float() for k in keys
                if k.startswith("model.") and "dual_tower" not in k}

    if pos_embedding is None:
        side = int(round(cfg.vit_seq ** 0.5))       # 196 → 14×14
        pos_embedding = sample_vit_pos_embedding(vit_w["pos_embed"], side, side)

    # patch_embed：Conv2d(k16,s16) ≡ im2col + GEMM。cin 3→16 补零满足对齐；
    # conv.bias 预融合进 pos_embedding（C++ 用一次 all_reduce_sum_residual 加上）。
    cw = vit_w["patch_embed.proj.weight"].half()
    cout, cin, kh, kwd = cw.shape
    cw = torch.cat([cw, cw.new_zeros(cout, 16 - cin, kh, kwd)], 1)
    # Weight swizzle and native patch embedding must use the same core count.
    gemm_w = _rpu16(transform_linear_weight(
        cw.permute(0, 2, 3, 1).reshape(cout, 16 * kh * kwd).contiguous(),
        partition=1, num_cores=_patch_embed_cores()))
    pos_fused = _rpu16(pos_embedding
                       + vit_w["patch_embed.proj.bias"].view(1, 1, -1))

    # merger 余部（DwPooler→GELU→proj2）跑在 RPU 的 aten::linear 上，
    # 因此要求 col-swizzled 权重。
    merger_rest = {k: (_col(v, 8) if k.endswith(".weight") else _rpu16(v))
                   for k, v in merger_w.items() if not k.startswith("proj1")}
    # Fused merger splits predictor.0 along K. The pooled half absorbs 1/4
    # because `hyvla_merger_pool` returns the group sum rather than its mean.
    _p0 = merger_w["pooler.predictor.0.weight"]
    _k = _p0.shape[1] // 2
    merger_rest["pooler.predictor.0.weight_a"] = _col(_p0[:, :_k], 8)
    merger_rest["pooler.predictor.0.weight_b"] = _col(_p0[:, _k:] * 0.25, 8)

    return HyVlaWeights(
        cfg=cfg,
        vit=_build_vit(vit_w, cfg),
        vit_proj1_w=_col(merger_w["proj1.weight"], 8),
        vit_proj1_b=_rpu16(merger_w["proj1.bias"]),
        vit_patch_gemm_w=gemm_w,
        vit_pos_fused=pos_fused,
        merger_rest=merger_rest,
        vlm_text=_build_mot_branch(vlm_w, cfg, ""),
        vlm_vision=_build_mot_branch(vlm_w, cfg, "_v"),
        vlm_final_norm=_rpu16(vlm_w["norm.weight"]),
        expert=_build_expert(expert_w, vlm_w, cfg),
        expert_final_norm=_rpu16(expert_w["norm.weight"]),
        tok_emb=tok_emb,
        host=host,
    )


def build_rope_tables(positions: torch.Tensor, cfg: HyVlaConfig,
                      vendor_bf16: bool = True):
    """`[rows, head_dim/2]` fp16 RPU —— kernel 格式是 HF 展开 `cat(f,f)` 的前半。

    第 i 行放 `positions[i]`（**不是** i）：suffix 的 position 从 prefix 的**有效**
    长度起算，与它在 KV 里的绝对下标（S..S+50）不相等。

    `vendor_bf16=True` 使用 checkpoint 所需的 bf16 `inv_freq` 语义；这是生产默认。
    改为 fp32 会改变 RoPE 数值。
    """
    hd = cfg.head_dim
    inv = 1.0 / (cfg.rope_base ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    if vendor_bf16:
        inv = inv.float().to(torch.bfloat16).double()
    ang = torch.outer(positions.double(), inv)
    return _rpu16(ang.cos()), _rpu16(ang.sin())


def _time_bias_list(cfg: HyVlaConfig, host: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """逐 timestep 的 `W1[:, D:] @ te_k + B1`，每项 `[1, D]` fp32。

    10 个 timestep 是固定的 {1.0, .9, …, .1} ⇒ 全预算。单步路径与图内 unroll
    **共用这一份**，避免两条路各写一遍 sinusoidal 而悄悄分叉。
    """
    D = cfg.expert_hidden
    W1, B1 = host["action_time_mlp_in.weight"], host["action_time_mlp_in.bias"]
    frac = torch.linspace(0.0, 1.0, D // 2, dtype=torch.float64)
    period = 4e-3 * (4.0 / 4e-3) ** frac
    scale = (1.0 / period * 2 * math.pi)[None, :]
    dt = -1.0 / cfg.num_steps
    out = []
    for k in range(cfg.num_steps):
        si = scale * torch.tensor([1.0 + dt * k], dtype=torch.float64)[:, None]
        # ⚠️ sin 在前、cos 在后；period 是几何级数插值，不是 1/base^(2i/d)；
        #    频率还要乘 2π。照抄 LLM 的 RoPE 公式会静默算错。
        te = torch.cat([si.sin(), si.cos()], dim=1).float()
        out.append(F.linear(te, W1[:, D:], B1))
    return out


def build_unroll_weights(cfg: HyVlaConfig, host: Dict[str, torch.Tensor]):
    """图内 10 步 unroll 需要的 6 个 RPU 张量。

    ⚠️ swizzle 不是统一的：`mo` 默认 **8 核 row 切分**（见 `_action_mlp_cores()`），
    其余 5 个仍是单核 col-swizzle。

    host 的逐步 encoder 是 `silu((x·W_in + b_in)·W1aᵀ + te_k)·W2ᵀ + b2`。
    前两个 GEMM 之间没有非线性，按结合律折成**一个**与步无关的权重：
        `(x·W_in + b_in)·W1aᵀ + te_k = x·(W1a·W_inᵀ)ᵀ + (b_in·W1aᵀ + te_k)`
    ⇒ `wc = W1a @ action_in_proj.weight`（`[D, action_dim]`），
       `time_all[k] = b_in·W1aᵀ + te_k`（`[num_steps, D]`）。

    ⚠️ 这**不是**恒等重排：① 折叠本身是矩阵重结合（在 fp32 host 上做一次）；
    ② 折完的 GEMM 与 out_proj 都改在 device 上跑 **fp16**，而单步路径是 host
    fp32。两者叠加会产生数值漂移，因此不保证逐位等价。

    `action_dim=32` 已是 16 的倍数 ⇒ 无需 out/K padding（图内 Euler 要求
    encoder 的 K 与 out_proj 的 N 同宽）。
    """
    D = cfg.expert_hidden
    W1 = host["action_time_mlp_in.weight"]
    W1a = W1[:, :D].contiguous()
    W_in0 = host["action_in_proj.weight"]                 # [D, action_dim]
    B_in = host["action_in_proj.bias"]                    # [D]
    time_all = torch.stack([(F.linear(B_in[None], W1a) + tb).reshape(-1)
                            for tb in _time_bias_list(cfg, host)])
    # mo 是这六个权重中唯一的 [D,D] 大矩阵，因此默认切 8 核。
    # ⚠️ **wc 与 mo 要一起改**：C++ 侧接的是
    # 「前一投影列并行 → 后一投影行(K)并行」这一对（与层体里
    # gate/up→down 相同的 TP 模式），
    # 只改一个 = 另一个读到别的核的段 ⇒ 静默算错。两边同源见 `_action_mlp_cores()`。
    amc = _action_mlp_cores()
    mo_w = host["action_time_mlp_out.weight"]
    return (_col(W1a @ W_in0, amc),                        # wc      [D, action_dim] 列并行
            _col(mo_w, 1) if amc == 1 else _row(mo_w, amc),  # mo    [D, D] 行(K)并行
            _rpu16(host["action_time_mlp_out.bias"]),      # mo_bias [D]
            _col(host["action_out_proj.weight"], 1),       # op      [action_dim, D]
            _rpu16(host["action_out_proj.bias"]),          # op_bias [action_dim]
            _rpu16(time_all))                              # time_all[num_steps, D]


def build_time_embed(cfg: HyVlaConfig, host: Dict[str, torch.Tensor]):
    """Precompute fixed timestep terms and return the per-step embed closure.

    The split uses `W1 @ [act; te] + b = W1a @ act + (W1t @ te + b)`.
    """
    D = cfg.expert_hidden
    W_in = host["action_in_proj.weight"].t().contiguous()   # K=32 ⇒ 预转置有利
    B_in = host["action_in_proj.bias"]
    W1 = host["action_time_mlp_in.weight"]
    W1a = W1[:, :D].contiguous()                            # 大 GEMM ⇒ 不转置
    W2, B2 = host["action_time_mlp_out.weight"], host["action_time_mlp_out.bias"]
    bias_te = _time_bias_list(cfg, host)

    def embed(x_t: torch.Tensor, step: int) -> torch.Tensor:
        a = torch.addmm(B_in, x_t.reshape(-1, cfg.action_dim).contiguous(), W_in)
        m = F.linear(a, W1a) + bias_te[step]
        return F.linear(F.silu(m), W2, B2).reshape(1, -1, D)

    return embed
