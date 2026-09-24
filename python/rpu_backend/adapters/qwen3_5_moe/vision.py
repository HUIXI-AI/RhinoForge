"""Exact TP4/mixed-TP6/TP8 35B-A3B shared FP16 Vision bridge."""
from __future__ import annotations

from rpu_backend.adapters.qwen3_5.vision import (
    _lazy_vision_graph_options,
    fuse_visual_embeds as _fuse_shared_visual_embeds,
    install_qwen3_5_vision_for_rpu as _install_shared_vision_for_rpu,
    make_qwen3_5_moe_vision_text_bridge,
)


_UNSET = object()


def install_qwen3_5_moe_vision_for_rpu(
    model,
    *,
    execution_config=_UNSET,
    _resolved_options=_UNSET,
    _text_bridge=_UNSET,
    _graph_runtime_policy=_UNSET,
    _cold_numeric_opt_in=_UNSET,
    **kwargs,
):
    """Install Vision against the live fixed-core MoE owner and cold policy."""
    bridge = make_qwen3_5_moe_vision_text_bridge(model)
    state = bridge.state
    options = _lazy_vision_graph_options(
        state, prepared_profile=bridge.prepared_profile
    )
    if execution_config is not _UNSET:
        raise TypeError(
            "Qwen3.5-MoE Vision wrapper owns its cold install controls"
        )
    if _resolved_options is not _UNSET:
        raise TypeError(
            "Qwen3.5-MoE Vision wrapper owns its cold install controls"
        )
    if _text_bridge is not _UNSET:
        raise TypeError(
            "Qwen3.5-MoE Vision wrapper owns its cold install controls"
        )
    if _graph_runtime_policy is not _UNSET:
        raise TypeError(
            "Qwen3.5-MoE Vision wrapper owns its cold install controls"
        )
    if _cold_numeric_opt_in is not _UNSET:
        raise TypeError(
            "Qwen3.5-MoE Vision wrapper owns its cold install controls"
        )
    return _install_shared_vision_for_rpu(
        model,
        execution_config=model._rpu_execution,
        _resolved_options=state.vision_control_options,
        _text_bridge=bridge,
        **options,
        **kwargs,
    )


def fuse_visual_embeds(
    model, hidden, input_ids, pixel_values, image_grid_thw, **kwargs
):
    """Fuse image rows while publishing M-RoPE through the typed MoE bridge."""
    bridge = make_qwen3_5_moe_vision_text_bridge(model)
    return _fuse_shared_visual_embeds(
        model,
        hidden,
        input_ids,
        pixel_values,
        image_grid_thw,
        _text_bridge=bridge,
        **kwargs,
    )


__all__ = ["install_qwen3_5_moe_vision_for_rpu", "fuse_visual_embeds"]
