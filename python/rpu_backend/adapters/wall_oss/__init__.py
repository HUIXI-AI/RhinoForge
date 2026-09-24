"""Wall-OSS-0.5 (Qwen2.5-VL VLA) → rpu_backend adapter package.

P1: text-only Qwen2.5 decoder (expert-0) prefill on RPU.
P2: action expert (expert-1) flow-matching denoise (10-step Euler) sharing P1's prefix KV.
P3: Qwen2.5-VL ViT vision tower (window attention) on RPU.
"""
from __future__ import annotations

from rpu_backend.adapters.wall_oss.llm import WallOssLLM, build_wall_oss_llm
from rpu_backend.adapters.wall_oss.action import WallOssAction, build_wall_oss_action
from rpu_backend.adapters.wall_oss.vision import WallOssVision, build_wall_oss_vision
from rpu_backend.adapters.wall_oss.runtime import WallOssVLA, build_wall_oss_vla

__all__ = [
    "WallOssLLM", "build_wall_oss_llm",
    "WallOssAction", "build_wall_oss_action",
    "WallOssVision", "build_wall_oss_vision",
    "WallOssVLA", "build_wall_oss_vla",
]
