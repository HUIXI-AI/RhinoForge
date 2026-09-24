"""Hy-Embodied-0.5-VLA 的 RPU adapter。

Hy-VLA 由 HYViT2 AnyRes ViT、HunYuanVL MoT 双权重 VLM 和共享 attention
的 flow-matching action expert 组成。native 入口分别为 hyvit2_*、
hyvla_vlm_* 和 hyvla_expert_*。

weights 模块负责 checkpoint 装载、权重 swizzle、RoPE 表和 boundary 常量；
runtime 模块负责子系统生命周期及 image → action 调用。
所有权重须在 forward 前就位。持久 handle 与 GraphCache 由 runner 管理，
执行布局改变时必须按其生命周期规则失效重建。RoPE 保留 checkpoint 的
BF16 inv_freq 语义，不用重新计算的 FP32 频率替代。"""
from __future__ import annotations

from rpu_backend.adapters.hy_vla.runtime import (HyVlaRunner, MASK_NEG,
                                                 build_hy_vla)
from rpu_backend.adapters.hy_vla.weights import (HyVlaConfig, HyVlaWeights,
                                                 build_rope_tables,
                                                 build_time_embed,
                                                 load_hy_vla_weights)

__all__ = [
    "HyVlaConfig",
    "HyVlaRunner",
    "HyVlaWeights",
    "MASK_NEG",
    "build_hy_vla",
    "build_rope_tables",
    "build_time_embed",
    "load_hy_vla_weights",
]
