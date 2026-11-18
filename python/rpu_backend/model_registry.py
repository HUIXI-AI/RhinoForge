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
    # Local conversion alias; runtime admission checks determine support.
    "qwen3-8b-w8a16-lmhead-int8": "qwen3_lmhead_int8_ckpts/qwen3-8b-w8a16-lmhead-int8",
    # Source-only local W8A16 conversion output. This alias does not establish
    # a public immutable derived-checkpoint identity. The source alias below
    # supports explicit on-install recipes, not ordinary FP16 admission.
    "qwen3-14b-w8a16-lmhead-int8": "qwen3_lmhead_int8_ckpts/qwen3-14b-w8a16-lmhead-int8",
    # Qwen3.5 hybrid-attention text profiles.
    "qwen3_5-0.8b": "qwen3_5-0.8b",
    "qwen3_5-27b": "Qwen3.5-27B",
    "qwen3_5-27b-fp16": "Qwen3.5-27B-FP16",
    "qwen3_5-27b-w8a16": "qwen3_5_quant_ckpts/qwen3_5-27b-w8a16",
    "qwen3_5-27b-w4a16": "qwen3_5_quant_ckpts/qwen3_5-27b-w4a16",
    "qwen3_5-27b-w4-mlp-only-g32": "qwen3_5_quant_ckpts/qwen3_5-27b-w4a16-g32-mlp-only",
    "qwen3_5-27b-w4-mlp-w8-attn-gdn-g32": "qwen3_5_quant_ckpts/qwen3_5-27b-w4a16-g32-mlp-w8-attn-gdn",
    "qwen3_5-27b-activation-awq-g128": "qwen3_5_quant_ckpts/qwen3_5-27b-activation-awq-g128",
    "qwen3_5-2b": "Qwen3.5-2B",      # symmetric GDN 16/16
    "qwen3_5-4b": "Qwen3.5-4B",      # GQA-GDN 16/32 (key-replicated)
    "qwen3_5-9b": "Qwen3.5-9B",      # GQA-GDN 16/32 + untied lm_head
    # Exact 35B-A3B TP8 controlled bundle. The adapter additionally validates
    # the complete mixed-E4M3 inventory before materializing model weights.
    "qwen3_5-35b-a3b-mixed": "Qwen3.5-35B-A3B-rpu-fp8-mixed",
    # Routed-only RTN G128; public examples may override the checkpoint with
    # --model PATH. The alias does not certify the current main runtime.
    "qwen3_5-35b-a3b-rtn-w4a16": "Qwen3.5-35B-A3B-RTN-W4A16-G128",
    # Official routed-only GPTQ G128, with the same explicit-path override.
    "qwen3_5-35b-a3b-gptq-w4a16": "Qwen3.5-35B-A3B-GPTQ-Int4",
    # Galaxea G0.5 VLA (Qwen3.5-2B VLM + action expert).
    "g05-base": "G05/g05-base",
    # Pi0.5 VLA (LeRobot)
    "pi05-base": "pi05_base",
    "pi05-libero-finetuned": "pi05_libero_finetuned",
    # Source-only local conversion outputs; aliases are path mappings, not
    # immutable derived-asset identities or support claims.
    "pi05-libero-finetuned-w8a16-vlm-expert": "pi05_libero_finetuned-w8a16-vlm-expert",
    # Per-channel INT4 projections with KV8; the legacy alias is retained.
    "pi05-libero-finetuned-w4real-kvint8": "pi05_libero_finetuned-w4real-kvint8",
    # Action horizon 10: lerobot/pi05_libero_base (openpi pi05_libero, trained
    # with action_horizon=10) with config.json chunk_size/n_action_steps = 10.
    "pi05-libero-h10": "pi05_libero_h10",
    "pi05-libero-h10-w8a16-vlm-expert": "pi05_libero_h10-w8a16-vlm-expert",
    # Wall-OSS-0.5 VLA (Qwen2.5-VL-3B)
    "wall-oss-0.5": "wall-oss-0.5",
    "wall-oss-0.5-w8a16": "wall-oss-0.5-w8a16",
    "wall-oss-0.5-w4a16": "wall-oss-0.5-w4a16",
    # G32 INT4 checkpoints; the runtime also accepts legacy INT8 Down payloads.
    "wall-oss-0.5-w4a16-pgrp": "wall-oss-0.5-w4a16-pgrp",
    # LingBot-VLA V2 (Qwen3-VL-4B + 32-expert top-4 AdaRMS sparse-MoE
    # action expert). Registry resolution is not a support claim.
    "lingbot-vla-v2-6b": "hf/robbyant/lingbot-vla-v2-6b",
    # Public InternRobotics/InternVLA-N1-w-NavDP component-scope cache alias.
    # Path resolution remains separate from the exact component contract; the
    # complete policy is not admitted by this alias.
    "internvla-n1-navdp": "InternVLA-N1-w-NavDP",
    # Hy-Embodied-0.5-VLA (HunYuanVL dual-tower: HYViT2-400M AnyRes ViT + MoT VLM
    # + flow-matching action expert). device runtime = adapters/hy_vla (build_hy_vla).
    # obs supplies pre-tokenized lang_tokens + the prefill/denoise masks, so no
    # tokenizer is needed at inference; the ckpt is a single model.safetensors.
    "hy-embodied-0.5-vla-umi": "Hy-Embodied-0.5-VLA-UMI",
    # DINOv3 ViT-B component-scope feature encoder.
    "dinov3-vit-b": "dinov3-vitb16-pretrain-lvd1689m",
    # Diagnostic assets; loader admission is unchanged.
    "dinov3-vit-s": "dinov3-vits16-pretrain-lvd1689m",
    "dinov3-vit-l": "dinov3-vitl16-pretrain-lvd1689m",
    "paligemma2-3b-pt-224": "paligemma2-3b-pt-224",
    "paligemma2-3b-mix-224": "paligemma2-3b-mix-224",
    # Qwen3-VL (vision + text)
    "qwen3-vl-2b": "Qwen3-VL-2B-Instruct",
    "qwen3-vl-4b": "Qwen3-VL-4B-Instruct",
    "qwen3-vl-8b": "Qwen3-VL-8B-Instruct",
    # Llama-3.2-1B local checkpoint alias.
    "llama-3.2-1b": "Llama-3.2-1B",
    # Additional public example aliases. These resolve caller-managed local
    # assets only; the loader still validates checkpoint format and geometry.
    "halo_action_expert": "HALO/action_expert.safetensors",
    "lingbot-vla-4b": "lingbot-vla-4b",
    "qwen3-0.6b-instruct": "qwen3-0.6b-instruct",
    "qwen3-0.6b-w8a16-lmhead-int8": "qwen3_lmhead_int8_ckpts/qwen3-0.6b-w8a16-lmhead-int8",
    "qwen3-1.7b-w8a16-lmhead-int8": "qwen3_lmhead_int8_ckpts/qwen3-1.7b-w8a16-lmhead-int8",
    "qwen3-14b": "Qwen3-14B",
    "qwen3-32b": "Qwen3-32B",
    "qwen3-32b-w8a16-lmhead-int8": "Qwen3-32B-w8a16",
    "qwen3-4b-instruct": "qwen3-4b-instruct",
    "qwen3-4b-w8a16-lmhead-int8": "qwen3_lmhead_int8_ckpts/qwen3-4b-w8a16-lmhead-int8",
    "qwen3-vl-2b-runtime-w4a16": "Qwen3-VL-2B-Instruct-runtime-w4-awq",
    "qwen3-vl-2b-runtime-w8a16": "Qwen3-VL-2B-Instruct-runtime-w8",
    "qwen3-vl-2b-text-awq-w4a16": "Qwen3-VL-2B-Instruct-AWQ",
    "qwen3-vl-2b-text-w8a16": "Qwen3-VL-2B-Instruct-w8a16-text",
    "qwen3-vl-32b-runtime-w4a16": "Qwen3-VL-32B-Instruct-runtime-w4",
    "qwen3-vl-32b-runtime-w8a16": "Qwen3-VL-32B-Instruct-runtime-w8",
    "qwen3-vl-32b-w8a16": 'Qwen3-VL-32B-Instruct-w8a16-full-20260816',
    "qwen3-vl-4b-runtime-w4a16": "Qwen3-VL-4B-Instruct-runtime-w4-awq",
    "qwen3-vl-4b-runtime-w8a16": "Qwen3-VL-4B-Instruct-runtime-w8",
    "qwen3-vl-4b-text-awq-w4a16": "Qwen3-VL-4B-Instruct-AWQ",
    "qwen3-vl-8b-runtime-w4a16": "Qwen3-VL-8B-Instruct-runtime-w4-awq",
    "qwen3-vl-8b-runtime-w8a16": "Qwen3-VL-8B-Instruct-runtime-w8",
    "qwen3_5-0.8b-w4a16": "qwen3_5_quant_ckpts/qwen3_5-0.8b-w4a16",
    "qwen3_5-0.8b-w8a16": "qwen3_5_quant_ckpts/qwen3_5-0.8b-w8a16",
    "qwen3_5-2b-w4a16": "qwen3_5_quant_ckpts/qwen3_5-2b-w4a16",
    "qwen3_5-2b-w4-mlp-only-g32": "qwen3_5-2b-w4a16-g32-mlp-only",
    "qwen3_5-2b-w4-mlp-w8-attn-gdn-g32": "qwen3_5-2b-w4a16-g32-mlp-w8-attn-gdn",
    "qwen3_5-2b-quanttrio-awq-g128": "Qwen3.5-2B-AWQ",
    "qwen3_5-4b-quanttrio-awq-g128": "Qwen3.5-4B-AWQ",
    "qwen3_5-9b-quanttrio-awq-g128": "Qwen3.5-9B-AWQ",
    "qwen3_5-2b-w8a16": "qwen3_5_quant_ckpts/qwen3_5-2b-w8a16",
    "qwen3_5-35b-a3b-mixed": "Qwen3.5-35B-A3B-rpu-fp8-mixed",
    "qwen3_5-4b-w4a16": "qwen3_5_quant_ckpts/qwen3_5-4b-w4a16",
    "qwen3_5-4b-w8a16": "qwen3_5_quant_ckpts/qwen3_5-4b-w8a16",
    "qwen3_5-9b-w4a16": "qwen3_5_quant_ckpts/qwen3_5-9b-w4a16",
    "qwen3_5-9b-w8a16": "qwen3_5_quant_ckpts/qwen3_5-9b-w8a16",
    "rhinovla-controlled": "rhinovla-controlled",
    # Gemma 4 E4B source-only adapter. Registry resolution is not a support
    # claim; the first release does not provide a compatible runnable profile.
    "gemma4-e4b": "gemma-4-e4b-it",
    'gr00t-n1d7-3b': 'GR00T-N1.7-3B',
    'internvla-n1-controlled': 'internvla_n1-controlled',
}

# Historical aliases are kept outside the active registry so archive tooling can
# resolve an explicitly requested rollback bundle without exposing it as a
# current model choice.
ARCHIVED_MODELS: Dict[str, str] = {
    "qwen3_8-27b-w8a16": "Qwen3.8-27B-w8a16",
}


def cache_root() -> Path:
    """Return the local model cache root.

    Override with the ``RPU_MODEL_CACHE`` environment variable.
    """
    override = os.environ.get("RPU_MODEL_CACHE")
    return Path(override) if override else DEFAULT_CACHE_ROOT


def archived_model_path(name: str) -> Path:
    """Resolve a historical model alias for explicit archive workflows."""
    if name not in ARCHIVED_MODELS:
        known = ", ".join(sorted(ARCHIVED_MODELS))
        raise KeyError(f"Unknown archived model {name!r}. Known: {known}")
    return cache_root() / ARCHIVED_MODELS[name]


def model_path(name: str) -> Path:
    """Resolve a logical model name (e.g. ``"qwen3-0.6b"``) to its on-disk path.

    Raises :class:`KeyError` for unknown names — keeps typos loud.
    """
    if name not in MODELS:
        known = ", ".join(sorted(MODELS))
        raise KeyError(f"Unknown model {name!r}. Known: {known}")
    return cache_root() / MODELS[name]


def hf_cache_root() -> Path:
    """Return the default Hugging Face cache beneath the model cache root."""
    return cache_root() / ".hf"


def set_hf_home_env() -> None:
    """Fill missing Hugging Face cache variables, preserving explicit overrides."""
    root = os.environ.setdefault("HF_HOME", str(hf_cache_root()))
    hub = str(Path(root) / "hub")
    os.environ.setdefault("HF_HUB_CACHE", hub)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub)
