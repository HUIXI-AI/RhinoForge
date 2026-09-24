"""Hy-Embodied-0.5-VLA 权重装载：safetensors → swizzled FP16 RPU 张量。

checkpoint 源张量逐个读取和释放；三个子系统的权重必须在任何 forward
之前全部就位，避免后续安装扰动已构建的 Graph。
转换顺序为 .half() → swizzle → .to('rpu')，使布局与 FP16 kernel 一致。"""
from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open

from rpu_backend.quant._common import quantize_linear_per_channel
from rpu_backend.quant.int4_pgrp_pack import pack_int4_per_channel_as_pgrp
from rpu_backend.runtime.weights import (tp_col_swizzle_mc_weight,
                                         tp_row_swizzle_mc_weight,
                                         transform_linear_weight)

# ── ckpt 前缀 ──────────────────────────────────────────────────────────────
_VIT = "model.dual_tower.vlm.model.visual.vision_tower."
_MERGER = "model.dual_tower.vlm.model.visual.merger."
_VLM = "model.dual_tower.vlm.model.language_model.model."
_EXPERT = "model.dual_tower.expert.model."
# embed_tokens 与 lm_head 共享权重；checkpoint 没有独立的 embed_tokens。
_LM_HEAD = "model.dual_tower.vlm.model.language_model.lm_head.weight"

_HOST_TENSORS = (
    "action_in_proj.bias", "action_in_proj.weight",
    "action_out_proj.bias", "action_out_proj.weight",
    "action_time_mlp_in.bias", "action_time_mlp_in.weight",
    "action_time_mlp_out.bias", "action_time_mlp_out.weight",
    "state_proj.bias", "state_proj.weight",
)
_VOCAB_SIZE = 120818
_VIT_POS_TOKENS = 128 * 128
_VIT_INPUT_CHANNELS = 3
_VIT_PATCH_SIZE = 16


@dataclasses.dataclass(frozen=True)
class HyVlaConfig:
    """Hy-VLA-UMI 的 checkpoint 几何；字段必须与配置及权重形状一致。"""
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
    # 特殊 token id，直接取自模型词表，不依赖 tokenizer。
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
        """RPU 使用的物理 KV head 数。

        RPU_HY_VLA_ATTN_TP8 将每个逻辑 KV head 复制到两个物理槽，使 attention
        使用 8 核。核 c 的 q head 为 2c、2c+1，对应逻辑 KV head floor(c/2)；
        o 投影的行切分保持同样的 head 对应关系。
        该布局增加 K/V 权重与 cache 存储，VLM 和 expert 必须同时采用。"""
        return self.num_kv_heads * 2 if _attn_tp8() else self.num_kv_heads

    @property
    def attn_tp(self) -> int:
        return min(8, self.kv_heads_rpu)                      # 4（默认）或 8（G 打开）

    @property
    def rope_base(self) -> float:
        return self.rope_theta * self.rope_alpha ** (
            self.head_dim / (self.head_dim - 2))


def _add_tower_shapes(
        shapes: dict[str, tuple[int, ...]], prefix: str, layers: int,
        hidden: int, intermediate: int, cfg: HyVlaConfig) -> None:
    q_width = cfg.num_q_heads * cfg.head_dim
    kv_width = cfg.num_kv_heads * cfg.head_dim
    shapes[prefix + "norm.weight"] = (hidden,)
    for li in range(layers):
        b = f"{prefix}layers.{li}."
        for suffix in ("", "_v"):
            shapes[f"{b}input_layernorm{suffix}.weight"] = (hidden,)
            shapes[f"{b}post_attention_layernorm{suffix}.weight"] = (hidden,)
            shapes[f"{b}self_attn.q_proj{suffix}.weight"] = (q_width, hidden)
            shapes[f"{b}self_attn.k_proj{suffix}.weight"] = (kv_width, hidden)
            shapes[f"{b}self_attn.v_proj{suffix}.weight"] = (kv_width, hidden)
            shapes[f"{b}self_attn.o_proj{suffix}.weight"] = (hidden, q_width)
            mlp = f"{b}mlp{suffix}."
            shapes[mlp + "gate_proj.weight"] = (intermediate, hidden)
            shapes[mlp + "up_proj.weight"] = (intermediate, hidden)
            shapes[mlp + "down_proj.weight"] = (hidden, intermediate)
        # q/k head norms have no `_v` tensor.  VLM shares its copy between the
        # two branches; the expert copy is present in the checkpoint but dead.
        shapes[f"{b}self_attn.query_layernorm.weight"] = (cfg.head_dim,)
        shapes[f"{b}self_attn.key_layernorm.weight"] = (cfg.head_dim,)


def _expected_checkpoint_shapes(cfg: HyVlaConfig) -> dict[str, tuple[int, ...]]:
    """Exact BF16 tensor-shape contract for the supported HyVLA checkpoint."""
    shapes = {
        _LM_HEAD: (_VOCAB_SIZE, cfg.vlm_hidden),
        "model.action_in_proj.bias": (cfg.expert_hidden,),
        "model.action_in_proj.weight": (cfg.expert_hidden, cfg.action_dim),
        "model.action_out_proj.bias": (cfg.action_dim,),
        "model.action_out_proj.weight": (cfg.action_dim, cfg.expert_hidden),
        "model.action_time_mlp_in.bias": (cfg.expert_hidden,),
        "model.action_time_mlp_in.weight": (
            cfg.expert_hidden, 2 * cfg.expert_hidden),
        "model.action_time_mlp_out.bias": (cfg.expert_hidden,),
        "model.action_time_mlp_out.weight": (
            cfg.expert_hidden, cfg.expert_hidden),
        "model.state_proj.bias": (cfg.expert_hidden,),
        "model.state_proj.weight": (cfg.expert_hidden, cfg.action_dim),
        _VIT + "patch_embed.proj.bias": (cfg.vit_hidden,),
        _VIT + "patch_embed.proj.weight": (
            cfg.vit_hidden, _VIT_INPUT_CHANNELS,
            _VIT_PATCH_SIZE, _VIT_PATCH_SIZE),
        _VIT + "pos_embed": (1, _VIT_POS_TOKENS, cfg.vit_hidden),
        _MERGER + "pooler.predictor.0.bias": (cfg.proj_dim,),
        _MERGER + "pooler.predictor.0.weight": (
            cfg.proj_dim, 2 * cfg.proj_dim),
        _MERGER + "pooler.predictor.2.bias": (cfg.proj_dim,),
        _MERGER + "pooler.predictor.2.weight": (cfg.proj_dim, cfg.proj_dim),
        _MERGER + "proj1.bias": (cfg.proj_dim,),
        _MERGER + "proj1.weight": (cfg.proj_dim, cfg.vit_hidden),
        _MERGER + "proj2.bias": (cfg.proj_dim,),
        _MERGER + "proj2.weight": (cfg.proj_dim, cfg.proj_dim),
    }
    qkv_width = 3 * cfg.vit_heads * cfg.vit_head_dim
    for li in range(cfg.vit_layers):
        b = f"{_VIT}blocks.{li}."
        shapes.update({
            b + "attn.proj.bias": (cfg.vit_hidden,),
            b + "attn.proj.weight": (
                cfg.vit_hidden, cfg.vit_heads * cfg.vit_head_dim),
            b + "attn.qkv.bias": (qkv_width,),
            b + "attn.qkv.weight": (qkv_width, cfg.vit_hidden),
            b + "mlp.fc1.bias": (cfg.vit_inter,),
            b + "mlp.fc1.weight": (cfg.vit_inter, cfg.vit_hidden),
            b + "mlp.fc2.bias": (cfg.vit_hidden,),
            b + "mlp.fc2.weight": (cfg.vit_hidden, cfg.vit_inter),
            b + "norm1.bias": (cfg.vit_hidden,),
            b + "norm1.weight": (cfg.vit_hidden,),
            b + "norm2.bias": (cfg.vit_hidden,),
            b + "norm2.weight": (cfg.vit_hidden,),
        })
    _add_tower_shapes(
        shapes, _VLM, cfg.layers, cfg.vlm_hidden, cfg.vlm_inter, cfg)
    _add_tower_shapes(
        shapes, _EXPERT, cfg.layers, cfg.expert_hidden, cfg.expert_inter, cfg)
    return shapes


def _expected_checkpoint_keys(cfg: HyVlaConfig) -> set[str]:
    return set(_expected_checkpoint_shapes(cfg))


def _unused_expert_checkpoint_keys(cfg: HyVlaConfig) -> set[str]:
    """Expert tensors present in the exact checkpoint but unreachable by HyVLA."""
    unused = set()
    for li in range(cfg.layers):
        b = f"{_EXPERT}layers.{li}."
        unused.update((
            f"{b}input_layernorm.weight",
            f"{b}post_attention_layernorm.weight",
            f"{b}self_attn.query_layernorm.weight",
            f"{b}self_attn.key_layernorm.weight",
        ))
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            unused.add(f"{b}self_attn.{proj}.weight")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            unused.add(f"{b}mlp.{proj}.weight")
    return unused


def _expected_materialized_keys(
        cfg: HyVlaConfig, *, load_pos_embedding: bool) -> set[str]:
    keys = _expected_checkpoint_keys(cfg) - _unused_expert_checkpoint_keys(cfg)
    if not load_pos_embedding:
        keys.remove(_VIT + "pos_embed")
    return keys


def _validate_checkpoint_keys(keys, cfg: HyVlaConfig) -> tuple[str, ...]:
    """Reject a different checkpoint inventory before the first RPU upload."""
    actual = tuple(keys)
    expected = _expected_checkpoint_keys(cfg)
    actual_set = set(actual)
    missing = sorted(expected - actual_set)
    unexpected = sorted(actual_set - expected)
    duplicate_count = len(actual) - len(actual_set)
    if missing or unexpected or duplicate_count:
        parts = []
        if missing:
            parts.append(f"missing {len(missing)}: {missing[:8]}")
        if unexpected:
            parts.append(f"unexpected {len(unexpected)}: {unexpected[:8]}")
        if duplicate_count:
            parts.append(f"duplicate names: {duplicate_count}")
        raise ValueError(
            "Hy-VLA checkpoint tensor names do not match the supported profile; "
            + "; ".join(parts))
    return actual


def _validate_checkpoint_schema(handle, cfg: HyVlaConfig) -> tuple[str, ...]:
    """Validate all names, dtypes and shapes without materializing a tensor."""
    keys = _validate_checkpoint_keys(handle.keys(), cfg)
    expected_shapes = _expected_checkpoint_shapes(cfg)
    errors = []
    for name in keys:
        tensor_slice = handle.get_slice(name)
        dtype = tensor_slice.get_dtype()
        shape = tuple(tensor_slice.get_shape())
        expected_shape = expected_shapes[name]
        if dtype != "BF16" or shape != expected_shape:
            errors.append(
                f"{name}: expected BF16 {expected_shape}, got {dtype} {shape}")
    if errors:
        raise ValueError(
            "Hy-VLA checkpoint tensor metadata do not match the supported profile; "
            f"invalid {len(errors)}: {errors[:8]}")
    return keys


def preflight_hy_vla_checkpoint(
        ckpt_path: "str | Path",
        cfg: HyVlaConfig = HyVlaConfig()) -> None:
    """Pure-host exact-profile validation with no tensor or RPU materialization."""
    with safe_open(Path(ckpt_path), framework="pt", device="cpu") as handle:
        _validate_checkpoint_schema(handle, cfg)


def _validate_materialized_keys(
        seen: set[str], cfg: HyVlaConfig, *, load_pos_embedding: bool) -> None:
    """Keep lazy-loading omissions and dead-weight reads fail-closed."""
    expected = _expected_materialized_keys(
        cfg, load_pos_embedding=load_pos_embedding)
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing or unexpected:
        raise RuntimeError(
            "Hy-VLA lazy checkpoint loader violated its materialization contract; "
            f"missing {len(missing)}: {missing[:8]}; "
            f"unexpected {len(unexpected)}: {unexpected[:8]}")


class _LazyFloat32TensorView(Mapping[str, torch.Tensor]):
    """Prefix-stripped safetensors view that never caches materialized values.

    Every access retains the eager loader's ``get_tensor(...).float()`` numeric
    contract.  Builders therefore own only the tensor they are transforming;
    the view itself keeps just the safetensors handle and tensor names.
    """

    def __init__(self, handle, prefix: str, keys: tuple[str, ...],
                 seen: set[str] | None = None):
        self._handle = handle
        self._prefix = prefix
        self._names = tuple(
            key[len(prefix):] for key in keys if key.startswith(prefix))
        self._name_set = frozenset(self._names)
        self._seen = seen

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in self._name_set:
            raise KeyError(f"Hy-VLA checkpoint has no tensor {self._prefix + name!r}")
        full_name = self._prefix + name
        tensor = self._handle.get_tensor(full_name).float()
        if self._seen is not None:
            self._seen.add(full_name)
        return tensor

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


def _rpu16(t: torch.Tensor) -> torch.Tensor:
    return t.to(dtype=torch.float16, device="rpu").contiguous()


def _col(w: torch.Tensor, nc: int) -> torch.Tensor:
    """col-partition swizzle（RC-1：先 .half() 再 swizzle）。"""
    return _rpu16(tp_col_swizzle_mc_weight(w.half().contiguous(), nc))


def _row(w: torch.Tensor, nc: int) -> torch.Tensor:
    return _rpu16(tp_row_swizzle_mc_weight(w.half().contiguous(), nc))


def _attn_tp8() -> bool:
    """Legacy config-view helper; loader execution uses its frozen cold plan."""
    return os.environ.get("RPU_HY_VLA_ATTN_TP8", "0").strip() in (
        "1", "true", "True", "on")


def _dup_kv(
    w: torch.Tensor,
    head_dim: int,
    *,
    attn_tp8: bool,
) -> torch.Tensor:
    """`[nkv*hd, K]` → `[2*nkv*hd, K]`，**块 c = 原 head `floor(c/2)`**。

    `repeat_interleave(2, dim=0)` 给出 `[h0,h0,h1,h1,h2,h2,h3,h3]` —— 正是
    `kv_heads_rpu` 里推出来的那个映射。G 关着时原样返回。
    """
    if not attn_tp8:
        return w
    n = w.shape[0] // head_dim
    assert n * head_dim == w.shape[0], f"_dup_kv: {w.shape} 不是 head_dim {head_dim} 的整数倍"
    return w.view(n, head_dim, -1).repeat_interleave(2, dim=0).reshape(
        2 * w.shape[0], w.shape[1]).contiguous()


# 逐塔 / 逐孪生量化开关认得的全部名字。`vlm_text` / `vlm_vision` 是 R42 加的
# MoT 两条孪生（VLM 的权重字节一半一半，各 3.22 GB/帧）。
_QUANT_NAMES = ("vit", "vlm", "expert", "vlm_text", "vlm_vision")


def _environment_value(
    environment: Mapping[str, str],
    var: str,
    default: str = "",
) -> str:
    return str(environment.get(var, default)).strip()


def _quant_tokens(
    var: str,
    environment: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """把 `var` 解析成一组名字。`""`/`0`/`off` ⇒ 空集；`1`/`on`/`all` ⇒ 全部。

    ⚠️ **按逗号分词精确匹配，不是子串匹配**（R42 改）。原来写的是 `who in v`，
    在只有 `vit`/`vlm`/`expert` 三个名字时两者等价；加了 `vlm_text` /
    `vlm_vision` 之后子串匹配会**互相误命中** —— `="vlm_vision"` 会让
    `"vlm" in "vlm_vision"` 成立，于是 text 那条孪生也被静默量化，
    正好毁掉这个开关存在的意义。原生塔的非量化开关由 runtime
    构建器单次解析后显式绑定，不再有 C++ 环境变量副本。

    未知名字**直接报错**：打错一个字母（`exper`）原来会静默退回 fp16，
    表现为"量化开了但没变快"，是很难查的坑。
    """
    source = os.environ if environment is None else environment
    v = _environment_value(source, var)
    normalized = v.lower()
    if normalized in ("", "0", "off", "false"):
        return frozenset()
    if normalized in ("1", "on", "true", "all"):
        return frozenset(_QUANT_NAMES)
    toks = frozenset(t.strip().lower() for t in v.split(",") if t.strip())
    bad = toks - set(_QUANT_NAMES)
    if bad:
        raise ValueError(
            f"{var}={v!r}: 不认识的名字 {sorted(bad)}；可用 {list(_QUANT_NAMES)}"
            " 或 all/off")
    if not toks:
        raise ValueError(
            f"{var}={v!r}: 无效的空量化选择；可用 {list(_QUANT_NAMES)}"
            " 或 all/off")
    return toks


def _qbits_from_tokens(
    who: str,
    part: str,
    w4_tokens: frozenset[str],
    w8_tokens: frozenset[str],
) -> int:
    for tokens, bits in ((w4_tokens, 4), (w8_tokens, 8)):
        if part and f"{who}_{part}" in tokens:
            return bits
        if who in tokens:
            return bits
    return 16


@dataclasses.dataclass(frozen=True)
class _HyVlaWeightColdPlan:
    """All environment-derived weight layout and quantization decisions."""

    patch_embed_cores: int
    action_mlp_cores: int
    attn_tp8: bool
    vit_bits: int
    vlm_text_bits: int
    vlm_vision_bits: int
    expert_bits: int

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "_HyVlaWeightColdPlan":
        source = os.environ if environment is None else environment
        w4_tokens = _quant_tokens("RPU_HY_VLA_W4A16", source)
        w8_tokens = _quant_tokens("RPU_HY_VLA_W8A16", source)
        vit_bits = _qbits_from_tokens("vit", "", w4_tokens, w8_tokens)
        if vit_bits == 4:
            raise ValueError(
                "RPU_HY_VLA_W4A16 cannot select vit: the certified ViT "
                "o_proj/fc2 layouts are not divisible by the int4 row-partition "
                "contract"
            )
        return cls(
            patch_embed_cores=(
                1 if _environment_value(
                    source, "RPU_HY_VLA_PATCH_EMBED_MC"
                ).lower() in ("0", "false") else 8
            ),
            action_mlp_cores=(
                1 if _environment_value(
                    source, "RPU_HY_VLA_ACTION_MLP_MC"
                ).lower() in ("0", "off", "false") else 8
            ),
            attn_tp8=(
                _environment_value(source, "RPU_HY_VLA_ATTN_TP8").lower()
                in ("1", "true", "on")
            ),
            vit_bits=vit_bits,
            vlm_text_bits=_qbits_from_tokens(
                "vlm", "text", w4_tokens, w8_tokens
            ),
            vlm_vision_bits=_qbits_from_tokens(
                "vlm", "vision", w4_tokens, w8_tokens
            ),
            expert_bits=_qbits_from_tokens(
                "expert", "", w4_tokens, w8_tokens
            ),
        )


def _q8(w: torch.Tensor, partition: int, nc: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp16 `[N,K]` → (int8 已 swizzle 的 RPU 权重, fp16 `[N]` RPU scale)。

    per-output-channel 对称 int8（`quant/_common.py`），kernel 侧由
    `rpu_linear.cpp` 按 `weight.scalar_type()==at::kChar` 自动派发到 generated
    W8A16 ACC16/ACC32 auto-tile family，scale 传**全长 N**（partition 由 reg18
    告诉 kernel，不需要按核切片）。

    三条**不报错、只会静默算错**的次序/口径约束：

    · **先量化再 swizzle。** 量化是逐输出通道（行）取 `amax`，而 swizzle 会把行
      按核打散重排 —— 反过来做，`amax` 就跨了输出通道。
    · **必须走 `transform_linear_weight`，不能用本模块的 `_col`/`_row`。**
      前者按 `element_size()` 取 `num_ele_32B`（int8 = 32，fp16 = 16），
      后两个写死 `dwidth=2` ⇒ 喂 int8 进去布局整个错位。
    · **`_dup_kv` 与量化谁先谁后都行。** per-row 量化对行复制是不变的
      （复制出来的行 `amax` 相同 ⇒ scale 相同 ⇒ int8 相同），所以这里统一放在
      `_dup_kv` **之后**，与 fp16 路径看到的是同一个张量。
      （HANDOFF §14.6 ① 当时担心两份 scale 会不一致，实际不会。）
    """
    w8, sc = quantize_linear_per_channel(w.half().contiguous())
    return (transform_linear_weight(w8.contiguous(), partition=partition,
                                    num_cores=nc).to("rpu").contiguous(),
            sc.to(torch.float16).to("rpu").contiguous())


def _q4(w: torch.Tensor, partition: int, nc: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP16 ``[N,K]`` -> pgrp-packed W4 plus striped FP16 scale.

    Quantization remains per output channel for numerical compatibility. Its
    scale is repeated across K/32 groups before packing the generated pgrp ABI.
    """
    v4, sc = quantize_linear_per_channel(w.half().contiguous(), bits=4)
    n, k = v4.shape
    packed, packed_scale = pack_int4_per_channel_as_pgrp(
        v4, sc.to(torch.float16), partition, nc
    )
    packed = packed.view(n, k // 2)
    return (packed.to("rpu").contiguous(),
            packed_scale.to("rpu").contiguous())


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
    patch_embed_cores: int                    # cold physical layout authority
    action_mlp_cores: int                     # cold physical layout authority
    kv_heads_rpu: int                         # frozen TP8/KV duplication authority
    attn_tp: int                              # frozen attention tensor parallelism
    vit: Dict[str, List[torch.Tensor]]          # 27 层，已 pad + swizzle
    vit_proj1_w: torch.Tensor                   # merger.proj1（fused 到 C++ 尾部）
    vit_proj1_b: torch.Tensor
    vit_patch_gemm_w: torch.Tensor              # patch_embed 折成 im2col GEMM
    vit_pos_fused: torch.Tensor                 # pos_embed + conv.bias 预融合
    merger_rest: Dict[str, torch.Tensor]        # DwPooler→GELU→proj2（P1b，全 RPU）
    vlm_text: Dict[str, List[torch.Tensor]]     # VLM text 塔
    vlm_vision: Dict[str, List[torch.Tensor]]   # VLM `_v` 塔
    vlm_final_norm: torch.Tensor
    expert: Dict[str, List[torch.Tensor]]       # expert（单塔，只 `_v` 半边）
    expert_final_norm: torch.Tensor
    tok_emb: torch.Tensor                       # host fp32：tied embed_tokens
    host: Dict[str, torch.Tensor]               # host fp32：boundary 投影


def _build_vit(
    w: Mapping[str, torch.Tensor],
    cfg: HyVlaConfig,
    cold: _HyVlaWeightColdPlan,
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
    bits = cold.vit_bits
    for li in range(cfg.vit_layers):
        b = f"blocks.{li}."
        # Keep the existing RPU upload order (q/k/v/o/fc1/fc2, then biases and
        # norms), while retaining at most the current source tensor.  qkv is a
        # single checkpoint tensor, so share that one materialization across
        # its three projections.
        qkv_w = w[b + "attn.qkv.weight"]
        qkv_b = w[b + "attn.qkv.bias"]
        padded_biases = {}
        for n, i in (("qw", 0), ("kw", 1), ("vw", 2)):
            linear, bias = split_pad(qkv_w, qkv_b, i)
            q, sc = _quant(linear, 1, 8, bits)
            out[n].append(q)
            if sc is not None:
                out[n + "s"].append(sc)
            padded_biases[n[0] + "b"] = bias
        del qkv_w, qkv_b, linear, bias

        ow = w[b + "attn.proj.weight"].view(Hid, H, D)
        ow = torch.cat([ow, ow.new_zeros(Hid, H, Dp - D)], 2).reshape(Hid, H * Dp)
        q, sc = _quant(ow.contiguous(), 0, 8, bits)
        out["ow"].append(q)
        if sc is not None:
            out["ows"].append(sc)
        del ow

        f1_src = w[b + "mlp.fc1.weight"]
        f1w = torch.cat([f1_src, f1_src.new_zeros(Ip - I, Hid)], 0)
        q, sc = _quant(f1w, 1, 8, bits)
        out["f1w"].append(q)
        if sc is not None:
            out["f1ws"].append(sc)
        del f1_src, f1w
        f1b_src = w[b + "mlp.fc1.bias"]
        padded_biases["f1b"] = torch.cat(
            [f1b_src, f1b_src.new_zeros(Ip - I)], 0)
        del f1b_src

        f2_src = w[b + "mlp.fc2.weight"]
        f2w = torch.cat([f2_src, f2_src.new_zeros(Hid, Ip - I)], 1)
        q, sc = _quant(f2w, 0, 8, bits)
        out["f2w"].append(q)
        if sc is not None:
            out["f2ws"].append(sc)
        del f2_src, f2w

        # q/k/v 的 head_dim 补零行、fc1 的 4304→4352 补零行都是整行 0
        # ⇒ amax=0 ⇒ scale 被 clamp 到 fp16 tiny、量化值全 0 ⇒ 输出仍是 0。
        for n in ("qb", "kb", "vb"):
            out[n].append(_rpu16(padded_biases.pop(n)))
        out["ob"].append(_rpu16(w[b + "attn.proj.bias"]))
        out["f1b"].append(_rpu16(padded_biases.pop("f1b")))
        for n, source in (
                ("f2b", "mlp.fc2.bias"), ("n1w", "norm1.weight"),
                ("n1b", "norm1.bias"), ("n2w", "norm2.weight"),
                ("n2b", "norm2.bias")):
            out[n].append(_rpu16(w[b + source]))
    return out


def _build_mot_branch(
    w: Mapping[str, torch.Tensor],
    cfg: HyVlaConfig,
    suffix: str,
    cold: _HyVlaWeightColdPlan,
) -> Dict[str, List[torch.Tensor]]:
    """VLM 的一条孪生。suffix "" = text 塔，"_v" = vision 塔。

    q/k head-norm **无 `_v` 版本**（两塔共享），两条孪生取同一张量。
    """
    d = {k: [] for k in ("q_w k_w v_w o_w q_norm k_norm input_norm post_norm "
                         "gate_w up_w down_w "
                         "q_ws k_ws v_ws o_ws gate_ws up_ws down_ws").split()}
    tp, mc = (8 if cold.attn_tp8 else min(8, cfg.num_kv_heads)), 8
    # MoT 的 text 与 _v 分支使用不同权重，必须分别计算 scale。
    # 两个独立 native setter 允许分别选择分支位宽。
    bits = cold.vlm_vision_bits if suffix else cold.vlm_text_bits
    for li in range(cfg.layers):
        b, sa, mlp = f"layers.{li}.", f"layers.{li}.self_attn.", f"layers.{li}.mlp{suffix}."
        # (影子键, checkpoint 键, partition: 1=col/0=row, 核数, 是否复制 KV)。
        # 键名元数据先成 tuple，值在循环内逐个物化，避免一层的 7 个大矩阵共存。
        for key, source, p, nc, dup_kv in (
                ("q_w",    f"{sa}q_proj{suffix}.weight",     1, tp, False),
                ("k_w",    f"{sa}k_proj{suffix}.weight",     1, tp, True),
                ("v_w",    f"{sa}v_proj{suffix}.weight",     1, tp, True),
                ("o_w",    f"{sa}o_proj{suffix}.weight",     0, tp, False),
                ("gate_w", f"{mlp}gate_proj.weight",         1, mc, False),
                ("up_w",   f"{mlp}up_proj.weight",           1, mc, False),
                ("down_w", f"{mlp}down_proj.weight",         0, mc, False)):
            raw = w[source]
            if dup_kv:
                raw = _dup_kv(raw, cfg.head_dim, attn_tp8=cold.attn_tp8)
            q, sc = _quant(raw, p, nc, bits)
            d[key].append(q)
            if sc is not None:
                d[key[:-1] + "ws"].append(sc)
            del raw
        d["q_norm"].append(_rpu16(w[f"{sa}query_layernorm.weight"]))   # 共享
        d["k_norm"].append(_rpu16(w[f"{sa}key_layernorm.weight"]))     # 共享
        d["input_norm"].append(_rpu16(w[f"{b}input_layernorm{suffix}.weight"]))
        d["post_norm"].append(_rpu16(w[f"{b}post_attention_layernorm{suffix}.weight"]))
    return d


def _build_expert(
    e: Mapping[str, torch.Tensor],
    v: Mapping[str, torch.Tensor],
    cfg: HyVlaConfig,
    cold: _HyVlaWeightColdPlan,
) -> Dict[str, List[torch.Tensor]]:
    """构造 32 层 expert 的 _v 分支。

    expert 的 modality_mask 恒为 True，因此只使用 _v 权重。
    q/k head-norm 必须取 VLM 对应层：原模型的层循环共享 VLM 的
    query/key_layernorm，不能替换为 expert 中同形状的张量。"""
    d = {k: [] for k in ("q_w k_w v_w o_w q_norm k_norm input_norm post_norm "
                         "gate_w up_w down_w "
                         "q_ws k_ws v_ws o_ws gate_ws up_ws down_ws").split()}
    tp, mc = (8 if cold.attn_tp8 else min(8, cfg.num_kv_heads)), 8
    bits = cold.expert_bits
    for li in range(cfg.layers):
        b, sa = f"layers.{li}.", f"layers.{li}.self_attn."
        for key, source, p, nc, dup_kv in (
                ("q_w",    f"{sa}q_proj_v.weight",       1, tp, False),
                ("k_w",    f"{sa}k_proj_v.weight",       1, tp, True),
                ("v_w",    f"{sa}v_proj_v.weight",       1, tp, True),
                ("o_w",    f"{sa}o_proj_v.weight",       0, tp, False),
                ("gate_w", f"{b}mlp_v.gate_proj.weight", 1, mc, False),
                ("up_w",   f"{b}mlp_v.up_proj.weight",   1, mc, False),
                ("down_w", f"{b}mlp_v.down_proj.weight", 0, mc, False)):
            raw = e[source]
            if dup_kv:
                raw = _dup_kv(raw, cfg.head_dim, attn_tp8=cold.attn_tp8)
            q, sc = _quant(raw, p, nc, bits)
            d[key].append(q)
            if sc is not None:
                d[key[:-1] + "ws"].append(sc)
            del raw
        d["q_norm"].append(_rpu16(v[f"{sa}query_layernorm.weight"]))   # ← VLM 的
        d["k_norm"].append(_rpu16(v[f"{sa}key_layernorm.weight"]))     # ← VLM 的
        d["input_norm"].append(_rpu16(e[f"{b}input_layernorm_v.weight"]))
        d["post_norm"].append(_rpu16(e[f"{b}post_attention_layernorm_v.weight"]))
    return d


def sample_vit_pos_embedding(pos_embed: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """ckpt 的 `pos_embed` → 本次网格的 `[1, h*w, C]` fp32 位置编码。

    照抄 vendor `modeling_hunyuan_vl_mot.py` 的两步（`forward_get_embedding_list`
    的 grid 构造 + `sample_positional_embedding` 的重采样）：ckpt 里存的是
    **128×128** 的 `pos_embed`（`[1, 16384, 1152]`），而 224×224 / patch16 的图只有
    14×14=196 个 patch，所以每帧都要按归一化网格 `grid_sample` 采一次。

    ⚠️ 这是 `grid_sample`（`padding_mode="border"`），**不是** `F.interpolate`。
    vendor 里两条路都存在：`rescale_positional_embedding` 走 interpolate，
    `sample_positional_embedding` 走 grid_sample，本模型的 ViT 只走后者
    （`use_grid_sampling: true`）。两者结果不同，别互换。

    ⚠️ 返回 **fp32**，不跟 vendor 那样 `.bfloat16()`：vendor 在
    `to_bfloat16_like_physical_intelligence()` 里把它连同权重一起降成 bf16，
    沿用那份是白白继承量化误差（`load_hy_vla_weights` 的原契约就要求 fp32）。
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


def load_hy_vla_weights(
    ckpt_path: "str | Path",
    pos_embedding: torch.Tensor | None = None,
    cfg: HyVlaConfig = HyVlaConfig(),
    *,
    _cold_plan: _HyVlaWeightColdPlan | None = None,
) -> HyVlaWeights:
    """在首次 forward 前装载全部 RPU 权重（见模块 docstring 的顺序约束）。

    checkpoint 名称先完整校验，随后源张量按需逐个转成 fp32、变换并释放；不会把
    checkpoint 的全部 fp32 副本同时留在 host。

    Args:
        ckpt_path: `model.safetensors`（4.5B，全 bf16）。
        pos_embedding: ViT 的 `[1, 196, 1152]` fp32 位置编码。**默认 `None` =
            从 ckpt 自己的 `pos_embed` 重采样**（`sample_vit_pos_embedding`）。
            传入张量则原样使用。
    """
    cold = (
        _HyVlaWeightColdPlan.from_environment()
        if _cold_plan is None else _cold_plan
    )
    if not isinstance(cold, _HyVlaWeightColdPlan):
        raise TypeError("_cold_plan must be a _HyVlaWeightColdPlan")
    p = Path(ckpt_path)
    load_pos_embedding = pos_embedding is None
    with safe_open(p, framework="pt", device="cpu") as f:
        # Repeat the pure-host preflight in this handle to close the gap between
        # an earlier lifecycle preflight and actual materialization.
        keys = _validate_checkpoint_schema(f, cfg)
        seen: set[str] = set()
        vit_w = _LazyFloat32TensorView(f, _VIT, keys, seen)
        merger_w = _LazyFloat32TensorView(f, _MERGER, keys, seen)
        vlm_w = _LazyFloat32TensorView(f, _VLM, keys, seen)
        expert_w = _LazyFloat32TensorView(f, _EXPERT, keys, seen)
        host_w = _LazyFloat32TensorView(f, "model.", keys, seen)
        root_w = _LazyFloat32TensorView(f, "", keys, seen)

        if pos_embedding is None:
            side = int(round(cfg.vit_seq ** 0.5))       # 196 → 14×14
            pos_embedding = sample_vit_pos_embedding(
                vit_w["pos_embed"], side, side)

        # The complete layout/quantization plan was resolved before this handle
        # was opened. No environment parsing may occur after tensor access starts.
        patch_embed_cores = cold.patch_embed_cores
        action_mlp_cores = cold.action_mlp_cores

        # patch_embed：Conv2d(k16,s16) ≡ im2col + GEMM。cin 3→16 补零满足对齐；
        # conv.bias 预融合进 pos_embedding（C++ 用一次 all_reduce_sum_residual 加上）。
        cw = vit_w["patch_embed.proj.weight"].half()
        cout, cin, kh, kwd = cw.shape
        cw = torch.cat([cw, cw.new_zeros(cout, 16 - cin, kh, kwd)], 1)
        # The selected core count travels with this swizzled tensor and is bound to
        # the native handle, so weight layout and execution cannot diverge.
        gemm_w = _rpu16(transform_linear_weight(
            cw.permute(0, 2, 3, 1).reshape(cout, 16 * kh * kwd).contiguous(),
            partition=1, num_cores=patch_embed_cores))
        del cw
        pos_fused = _rpu16(pos_embedding
                           + vit_w["patch_embed.proj.bias"].view(1, 1, -1))

        # RPU aten::linear 上的 merger 余部要求 col-swizzled 权重。
        merger_rest = {}
        for key in merger_w:
            if key.startswith("proj1"):
                continue
            value = merger_w[key]
            merger_rest[key] = (
                _col(value, 8) if key.endswith(".weight") else _rpu16(value))
            del value
        # `RPU_HY_VLA_FUSED_MERGER` 路径用的 predictor.0 **K 拆分**两半。
        # 原式是 `cat([nxc, pooled], -1) @ W0^T`（K=4096），拆成
        # `nxc @ Wa^T + pooled @ Wb^T`（各 K=2048）⇒ 省掉那次 4.8 MB 的 `cat` 落地，
        # 且两半输出同形，直接相加即可（**避开首轴隐式广播的静默 NaN**）。
        # `pooled` 那半吸收 1/4：device 侧的 `hyvla_merger_pool` 只做 Σ_m 不除 4。
        # fp16 里乘 0.25 是精确的 2 的幂缩放（Wb 量级 ~1e-2，缩放后远离非规格化）。
        _p0 = merger_w["pooler.predictor.0.weight"]
        _k = _p0.shape[1] // 2
        merger_rest["pooler.predictor.0.weight_a"] = _col(_p0[:, :_k], 8)
        merger_rest["pooler.predictor.0.weight_b"] = _col(_p0[:, _k:] * 0.25, 8)
        del _p0

        # Preserve the established device upload order.  Only the CPU source
        # lifetime changes; all builders still finish before any forward.
        vit = _build_vit(vit_w, cfg, cold)
        vit_proj1_w = _col(merger_w["proj1.weight"], 8)
        vit_proj1_b = _rpu16(merger_w["proj1.bias"])
        vlm_text = _build_mot_branch(vlm_w, cfg, "", cold)
        vlm_vision = _build_mot_branch(vlm_w, cfg, "_v", cold)
        vlm_final_norm = _rpu16(vlm_w["norm.weight"])
        expert = _build_expert(expert_w, vlm_w, cfg, cold)
        expert_final_norm = _rpu16(expert_w["norm.weight"])

        # These are the only persistent CPU fp32 weights.  Materialize them
        # after all RPU uploads so they do not inflate the upload peak.
        tok_emb = root_w[_LM_HEAD]
        host = {name: host_w[name] for name in _HOST_TENSORS}
        _validate_materialized_keys(
            seen, cfg, load_pos_embedding=load_pos_embedding)

        return HyVlaWeights(
            cfg=cfg,
            patch_embed_cores=patch_embed_cores,
            action_mlp_cores=action_mlp_cores,
            kv_heads_rpu=(
                cfg.num_kv_heads * 2 if cold.attn_tp8 else cfg.num_kv_heads
            ),
            attn_tp=(8 if cold.attn_tp8 else min(8, cfg.num_kv_heads)),
            vit=vit,
            vit_proj1_w=vit_proj1_w,
            vit_proj1_b=vit_proj1_b,
            vit_patch_gemm_w=gemm_w,
            vit_pos_fused=pos_fused,
            merger_rest=merger_rest,
            vlm_text=vlm_text,
            vlm_vision=vlm_vision,
            vlm_final_norm=vlm_final_norm,
            expert=expert,
            expert_final_norm=expert_final_norm,
            tok_emb=tok_emb,
            host=host,
        )


def build_rope_tables(positions: torch.Tensor, cfg: HyVlaConfig,
                      vendor_bf16: bool = True):
    """构造 [rows, head_dim/2] FP16 RPU RoPE 表。

    第 i 行对应 positions[i]，不是行号 i。suffix 的 position 从 prefix
    有效长度起算，与 KV 中从 S 开始的绝对下标不同。
    vendor_bf16 保留 checkpoint 将 inv_freq 量化为 BF16 的推理语义；
    改用未经该量化的频率会改变旋转值。"""
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


def build_unroll_weights(cfg: HyVlaConfig, host: Dict[str, torch.Tensor],
                         action_mlp_cores: int):
    """图内 10 步 unroll 需要的 6 个 RPU 张量。

    ⚠️ swizzle 不是统一的：`mo` 默认 **8 核 row 切分**（见 frozen cold plan），
    其余 5 个仍是单核 col-swizzle。

    host 的逐步 encoder 是 `silu((x·W_in + b_in)·W1aᵀ + te_k)·W2ᵀ + b2`。
    前两个 GEMM 之间没有非线性，按结合律折成**一个**与步无关的权重：
        `(x·W_in + b_in)·W1aᵀ + te_k = x·(W1a·W_inᵀ)ᵀ + (b_in·W1aᵀ + te_k)`
    ⇒ `wc = W1a @ action_in_proj.weight`（`[D, action_dim]`），
       `time_all[k] = b_in·W1aᵀ + te_k`（`[num_steps, D]`）。

    ⚠️ 这**不是**恒等重排：① 折叠本身是矩阵重结合（在 fp32 host 上做一次）；
    ② 折完的 GEMM 与 out_proj 都改在 device 上跑 **fp16**，而单步路径是 host
    fp32。因此此路径可能出现由运算重结合及 dtype 转换带来的数值差异。

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
    # wc 与 mo 的核数必须一致：native 使用列并行后接行并行的配对布局。
    # 只改变其中一侧会让消费者读取错误的核分片；核数随权重绑定到 handle。
    amc = action_mlp_cores
    mo_w = host["action_time_mlp_out.weight"]
    return (_col(W1a @ W_in0, amc),                        # wc      [D, action_dim] 列并行
            _col(mo_w, 1) if amc == 1 else _row(mo_w, amc),  # mo    [D, D] 行(K)并行
            _rpu16(host["action_time_mlp_out.bias"]),      # mo_bias [D]
            _col(host["action_out_proj.weight"], 1),       # op      [action_dim, D]
            _rpu16(host["action_out_proj.bias"]),          # op_bias [action_dim]
            _rpu16(time_all))                              # time_all[num_steps, D]


def build_time_embed(cfg: HyVlaConfig, host: Dict[str, torch.Tensor]):
    """预计算固定 timestep 的 boundary 常量并返回逐步 embed 闭包。

    sinusoidal 时间项按固定 Euler 序列预计算。action_in_proj 使用预转置
    权重与 addmm；time_mlp_in 按列拆分：
        W1 @ [act; te] + b = W1[:, :D] @ act + (W1[:, D:] @ te + b)
    右项只依赖 timestep，可复用并省去每步的拼接。
    较大的 GEMM 保留 F.linear 及其惰性转置权重布局。"""
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
