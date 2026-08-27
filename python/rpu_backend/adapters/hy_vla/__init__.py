"""Hy-Embodied-0.5-VLA adapter package.

Hy-VLA combines an HYViT2 vision tower, a dual-weight HunYuanVL MoT prefill
stack, and a flow-matching action expert with ten Euler steps. The native
subsystems are exposed through ``torch.ops.rpu.hyvit2_*``,
``torch.ops.rpu.hyvla_vlm_*`` and ``torch.ops.rpu.hyvla_expert_*``.

``weights`` loads and transforms the complete checkpoint before execution;
``runtime`` owns the three handles, graph caches, KV caches and ``get_action``
lifecycle. RoPE uses the checkpoint's bf16 ``inv_freq`` semantics; changing it
to fp32 changes model output rather than merely increasing arithmetic precision.
"""
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
