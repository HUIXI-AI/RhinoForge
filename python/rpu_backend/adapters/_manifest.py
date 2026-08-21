"""Pure-data manifest for built-in HF architecture adapters.

The package composition root passes this table to ``runtime.registry``.
Keeping it separate from adapter implementations makes built-in discovery
deterministic without importing model modules or installing their patches.
"""
from __future__ import annotations


BUILTIN_ADAPTERS = (
    (
        "Qwen3ForCausalLM",
        "rpu_backend.adapters.qwen3",
        "Qwen3Adapter",
    ),
    (
        "LlamaForCausalLM",
        "rpu_backend.adapters.llama",
        "LlamaAdapter",
    ),
    (
        "DINOv3ViTModel",
        "rpu_backend.adapters.dinov3",
        "DINOv3Adapter",
    ),
    (
        "Gemma4ForConditionalGeneration",
        "rpu_backend.adapters.gemma4",
        "Gemma4Adapter",
    ),
    (
        "Qwen3_5ForConditionalGeneration",
        "rpu_backend.adapters.qwen3_5",
        "Qwen3_5Adapter",
    ),
    (
        "Qwen3VLForConditionalGeneration",
        "rpu_backend.adapters.qwen3_vl",
        "Qwen3VLAdapter",
    ),
)


__all__ = ["BUILTIN_ADAPTERS"]
