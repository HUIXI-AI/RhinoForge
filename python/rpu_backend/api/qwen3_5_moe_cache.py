"""Public cache type for the exact Qwen3.5-35B-A3B fixed core profiles."""
from __future__ import annotations

import torch

from .qwen3_5_cache import (
    QWEN3_5_OPTIONAL_PADDING_CAP,
    QWEN3_5_TOTAL_PADDING_CAP,
    Qwen3_5Cache,
)


QWEN3_5_MOE_OPTIONAL_PADDING_CAP = QWEN3_5_OPTIONAL_PADDING_CAP
QWEN3_5_MOE_TOTAL_PADDING_CAP = QWEN3_5_TOTAL_PADDING_CAP


class Qwen3_5MoeCache(Qwen3_5Cache):
    """Hybrid KV/GDN cache with active compute cores and eight physical stripes."""

    @classmethod
    def from_config(
        cls, config, max_seq_len, device="rpu", dtype=torch.float16, *,
        num_cores=8,
    ):
        from rpu_backend.adapters.qwen3_5_moe.quant_scope import (
            validate_exact_profile,
        )

        if type(max_seq_len) is not int or not 0 < max_seq_len <= 8192:
            raise ValueError(
                "Qwen3.5-35B-A3B cache max_seq_len must be an integer in [1, 8192]"
            )
        if type(num_cores) is not int or num_cores not in (4, 6, 8):
            raise ValueError(
                f"Qwen3.5-35B-A3B cache requires num_cores=4, 6 or 8, got {num_cores!r}"
            )
        try:
            validate_exact_profile(config, require_quant=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        from rpu_backend.adapters.qwen3_5_moe.cores import core_topology
        topology = core_topology({"model": {"num_cores": num_cores}})
        text_config = getattr(config, "text_config", config)
        # Dense Qwen3.5's reduced-profile factory deliberately rejects MoE.
        # Reuse its cache implementation with this already validated geometry.
        return cls(
            num_layers=text_config.num_hidden_layers, batch_size=1,
            max_seq_len=max_seq_len, num_kv_heads=topology.attn_tp,
            head_dim=text_config.head_dim, layer_types=text_config.layer_types,
            num_v_heads=text_config.linear_num_value_heads,
            key_head_dim=text_config.linear_key_head_dim,
            value_head_dim=text_config.linear_value_head_dim,
            conv_dim=(2 * text_config.linear_num_key_heads * text_config.linear_key_head_dim
                      + text_config.linear_num_value_heads * text_config.linear_value_head_dim),
            conv_kernel_dim=text_config.linear_conv_kernel_dim,
            attn_tp=topology.attn_tp, num_cores=topology.num_cores, device=device, dtype=dtype,
        )

    @classmethod
    def from_model(
        cls, model, max_seq_len, device="rpu", dtype=torch.float16
    ):
        from rpu_backend.adapters.qwen3_5.cores import topology_tuple
        from rpu_backend.adapters.qwen3_5_moe.cores import (
            core_topology, validate_native_topology,
        )

        fusion = getattr(model, "model", model)
        inner = getattr(fusion, "language_model", fusion)
        state = getattr(inner, "_rpu_qwen3_5_moe", None)
        topology = getattr(state, "execution_topology", None)
        if (
            state is None
            or getattr(state, "handle", None) is None
            or topology is None
            or getattr(topology, "num_cores", None) not in (4, 6, 8)
        ):
            raise ValueError(
                "Qwen3_5MoeCache.from_model requires a completely installed "
                "35B-A3B model"
            )
        expected = core_topology(getattr(model, "_rpu_execution", None))
        if topology_tuple(topology) != topology_tuple(expected):
            raise ValueError("Qwen3.5-MoE config no longer matches its installed core layout")
        validate_native_topology(state.handle, topology)
        return cls.from_config(
            model.config, max_seq_len=max_seq_len, device=device, dtype=dtype,
            num_cores=topology.num_cores,
        )


__all__ = [
    "QWEN3_5_MOE_OPTIONAL_PADDING_CAP",
    "QWEN3_5_MOE_TOTAL_PADDING_CAP",
    "Qwen3_5MoeCache",
]
