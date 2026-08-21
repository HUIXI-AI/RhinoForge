"""Logical model aliases and local cache resolution for RhinoForge."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "rhinoforge" / "models"

MODELS: Dict[str, str] = {
    # Qwen3 family (HF or ModelScope downloads, flat layout)
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
    # Qwen3 14B is exposed only through its release-scoped W8A16 profile. The
    # FP16-looking alias is intentionally absent.
    "qwen3-14b-w8a16-lmhead-int8": "qwen3_lmhead_int8_ckpts/qwen3-14b-w8a16-lmhead-int8",
    # Qwen3.5 hybrid-attention text profiles.
    "qwen3_5-0.8b": "qwen3_5-0.8b",
    "qwen3_5-2b": "Qwen3.5-2B",      # symmetric GDN 16/16
    "qwen3_5-4b": "Qwen3.5-4B",      # GQA-GDN 16/32 (key-replicated)
    "qwen3_5-9b": "Qwen3.5-9B",      # GQA-GDN 16/32 + untied lm_head
    # Galaxea G0.5 VLA (Qwen3.5-2B VLM + action expert).
    "g05-base": "G05/g05-base",
    # Pi0.5 VLA (LeRobot)
    "pi05-base": "pi05_base",
    "pi05-libero-finetuned": "pi05_libero_finetuned",
    "pi05-libero-finetuned-w8a16-vlm-expert": "pi05_libero_finetuned-w8a16-vlm-expert",
    "pi05-libero-finetuned-w4real-kvint8": "pi05_libero_finetuned-w4real-kvint8",
    # Wall-OSS-0.5 VLA (Qwen2.5-VL-3B)
    "wall-oss-0.5": "wall-oss-0.5",
    "wall-oss-0.5-w8a16": "wall-oss-0.5-w8a16",
    "wall-oss-0.5-w4a16": "wall-oss-0.5-w4a16",
    # Group-wise int4 profile: expert-0 down remains int8; the other seven
    # projections use int4-pgrp with group_size=32.
    "wall-oss-0.5-w4a16-pgrp": "wall-oss-0.5-w4a16-pgrp",
    # GR00T-N1.7-3B VLA (Qwen3-VL-2B backbone + flow-matching DiT action head).
    # RPU runtime = adapters/gr00t (build_gr00t_vla); needs the public Qwen3-VL-2B
    # config/tokenizer alongside ("qwen3-vl-2b") for the backbone scaffold.
    "gr00t-n1d7-3b": "GR00T-N1.7-3B",
    # LingBot-VLA V2 (Qwen3-VL-4B + 32-expert top-4 AdaRMS sparse-MoE
    # action expert). Registry resolution is not a support claim.
    "lingbot-vla-v2-6b": "hf/robbyant/lingbot-vla-v2-6b",
    # InternVLA-N1 controlled-evaluation cache alias. Path resolution is not a
    # support claim: the validated path requires exact cold-start
    # RPU_INTERNVLA_N1_EXACT_TWOSTAGE=1 + RPU_ALLREDUCE_TWOSTAGE=1 plus a
    # caller-pinned SHA-256 manifest; the single-stage opt-in is diagnostic.
    "internvla-n1-navdp": "InternVLA-N1-w-NavDP",
    # Hy-Embodied-0.5-VLA (HunYuanVL dual-tower: HYViT2-400M AnyRes ViT + MoT VLM
    # + flow-matching action expert). RPU runtime = adapters/hy_vla (build_hy_vla).
    # obs supplies pre-tokenized lang_tokens + the prefill/denoise masks, so no
    # tokenizer is needed at inference; the ckpt is a single model.safetensors.
    "hy-embodied-0.5-vla-umi": "Hy-Embodied-0.5-VLA-UMI",
    # DINOv3 ViT-B encoder.
    "dinov3-vit-b": "dinov3-vitb16-pretrain-lvd1689m",
    # Qwen3-VL (vision + text)
    "qwen3-vl-2b": "Qwen3-VL-2B-Instruct",
    "qwen3-vl-4b": "Qwen3-VL-4B-Instruct",
    # Llama-3.2-1B local checkpoint alias.
    "llama-3.2-1b": "Llama-3.2-1B",
    # Gemma 4 E4B source-only adapter. Registry resolution is not a support
    # claim; the first release does not provide a compatible runnable profile.
    "gemma4-e4b": "gemma-4-e4b-it",
    # HALO (BAGEL-MoT VLA) PyTorch weight bundle root under RPU_MODEL_CACHE.
    "halo_action_expert": "HALO/pytorch_weights_current",
}


def cache_root() -> Path:
    """Return the local model cache root.

    Override with the ``RPU_MODEL_CACHE`` environment variable.
    """
    override = os.environ.get("RPU_MODEL_CACHE")
    return Path(override).expanduser() if override else DEFAULT_CACHE_ROOT


def model_path(name: str) -> Path:
    """Resolve a logical model name (e.g. ``"qwen3-0.6b"``) to its on-disk path.

    Raises :class:`KeyError` for unknown names — keeps typos loud.
    """
    if name not in MODELS:
        known = ", ".join(sorted(MODELS))
        raise KeyError(f"Unknown model {name!r}. Known: {known}")
    return cache_root() / MODELS[name]


def hf_cache_root() -> Path:
    """Return the effective Hugging Face cache root."""
    configured = os.environ.get("HF_HOME")
    if configured:
        return Path(configured).expanduser()
    if os.environ.get("RPU_MODEL_CACHE"):
        return cache_root() / ".hf"
    return Path.home() / ".cache" / "huggingface"


def set_hf_home_env() -> None:
    """Route Hugging Face caches under an explicit RPU_MODEL_CACHE override."""
    if not os.environ.get("RPU_MODEL_CACHE"):
        return
    root = os.environ.setdefault("HF_HOME", str(hf_cache_root()))
    hub = str(Path(root) / "hub")
    os.environ.setdefault("HF_HUB_CACHE", hub)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub)
