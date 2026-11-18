"""Exact cold G05 admission shared by Qwen3.5 installers and their G05 caller.

This module owns the profile and payload checks without importing the composite
G05 adapter. It validates existing choices; it does not install or quantize weights.
"""
from __future__ import annotations

import torch


_G05_LAYER_TYPES = tuple(
    "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
    for i in range(24)
)

_G05_VLM_PROFILE = (
    2048, 6144, 24, 8, 2, 256,
    16, 16, 128, 128, 4,
    False, "silu", "embedding", "lm_head", None, True,
    1.0e-6, 10_000_000.0, "default", (11, 11, 10), True, 0.25,
    _G05_LAYER_TYPES,
)

_ACTION_PROFILE = (
    1024, 4096, 24, 8, 2, 256,
    False, "silu", "adaLN", 1024, "linear", 27, "linear", 27, True,
    1.0e-6, 10_000_000.0, "default", (11, 11, 10), True, 0.25,
    ("full_attention",) * 24,
)


def _check_g05_vlm_profile(config) -> None:
    """Accept only the canonical G0.5 Qwen3.5-2B VLM."""
    from rpu_backend.api.errors import UnsupportedModelError

    rope = getattr(config, "rope_parameters", None) or {}
    profile = (
        int(config.hidden_size),
        int(config.intermediate_size),
        int(config.num_hidden_layers),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(config.head_dim),
        int(config.linear_num_value_heads),
        int(config.linear_num_key_heads),
        int(config.linear_value_head_dim),
        int(config.linear_key_head_dim),
        int(config.linear_conv_kernel_dim),
        bool(config.attention_bias),
        str(config.hidden_act),
        str(config.input_type),
        str(config.output_type),
        getattr(config, "adaptive_mode", None),
        bool(config.use_final_norm),
        float(config.rms_norm_eps),
        float(rope.get("rope_theta", 0.0)),
        str(rope.get("rope_type", "")),
        tuple(int(x) for x in rope.get("mrope_section", ())),
        bool(rope.get("mrope_interleaved", False)),
        float(rope.get("partial_rotary_factor", 0.0)),
        tuple(config.layer_types),
    )
    if profile != _G05_VLM_PROFILE:
        raise UnsupportedModelError(
            "G0.5 RPU bring-up supports only the canonical Qwen3.5-2B VLM "
            f"profile; got {profile}."
        )


def _check_g05_action_profile(config) -> None:
    from rpu_backend.api.errors import UnsupportedModelError

    rope = getattr(config, "rope_parameters", None) or {}
    profile = (
        int(config.hidden_size),
        int(config.intermediate_size),
        int(config.num_hidden_layers),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(config.head_dim),
        bool(config.attention_bias),
        str(config.hidden_act),
        str(config.adaptive_mode),
        int(config.time_hidden_size),
        str(config.input_type),
        int(config.input_dim),
        str(config.output_type),
        int(config.output_dim),
        bool(config.use_final_norm),
        float(config.rms_norm_eps),
        float(rope.get("rope_theta", 0.0)),
        str(rope.get("rope_type", "")),
        tuple(int(x) for x in rope.get("mrope_section", ())),
        bool(rope.get("mrope_interleaved", False)),
        float(rope.get("partial_rotary_factor", 0.0)),
        tuple(config.layer_types),
    )
    if profile != _ACTION_PROFILE:
        raise UnsupportedModelError(
            "G0.5 RPU action bring-up supports only the canonical "
            f"Qwen3.5 action-expert profile; got {profile}."
        )


def normalize_math_precision(value):
    if not isinstance(value, str) or value not in ("base", "high"):
        raise ValueError("G0.5 math_precision must be 'base' or 'high'")
    return value


def require_g05_high_math_native(*, vision=False):
    name = ("qwen3_5_vision_set_high_precision" if vision
            else "qwen3_5_set_high_precision")
    if not hasattr(torch.ops.rpu, name):
        raise RuntimeError(f"G0.5 HIGH math requires native {name}; rebuild before installing")


def validate_g05_high_math_install(config, *, cpu_stage_weights):
    """Keep HIGH private to the same exact cold G05 text/action profiles."""
    if not cpu_stage_weights:
        raise ValueError("G0.5 HIGH math requires the private CPU-staged installer")
    if getattr(config, "adaptive_mode", None) == "adaLN":
        _check_g05_action_profile(config)
    else:
        _check_g05_vlm_profile(config)


def g05_w8_quant_config(*, action=False):
    return {
        "format_version": 2,
        "architecture": "qwen3_5",
        "activation_dtype": "float16",
        "default": {"method": "w8a16"},
        "projection_overrides": {} if action else {
            "linear_attn.in_proj_b": {"method": "fp16"},
            "linear_attn.in_proj_a": {"method": "fp16"},
        },
        "storage": "int8",
        "scale_dtype": "float16",
        "profile": "g05_k1_action_w8a16" if action else "g05_k1_text_w8a16",
    }


def validate_g05_w8_install(config, *, cpu_stage_weights):
    """Check private admission independently of tensor dtype validation."""
    if not cpu_stage_weights:
        raise ValueError("G0.5 W8A16 requires the private CPU-staged installer")
    action = getattr(config, "adaptive_mode", None) == "adaLN"
    if action:
        _check_g05_action_profile(config)
    else:
        _check_g05_vlm_profile(config)
    if getattr(config, "quant_config", None) != g05_w8_quant_config(action=action):
        raise ValueError("G0.5 W8A16 requires its exact per-channel quantization scope")


def require_g05_action_nvfp4_native():
    """Fail before mutation when using an archived native library."""
    op = getattr(torch.ops.rpu, "qwen3_5_set_weights", None)
    schemas = getattr(op, "_schemas", {})
    if not isinstance(schemas, dict) or not any(
            any(arg.name == "nvfp4_tensor_scales" and str(arg.type) == "Optional[List[Tensor]]"
                for arg in schema.arguments) for schema in schemas.values()):
        raise RuntimeError("G0.5 Action NVFP4 requires a native qwen3_5_set_weights "
                           "with nvfp4_tensor_scales; rebuild before installing")


def validate_g05_action_nvfp4_payload(payload):
    """Validate physical owners independently of generic INT8 quant specs."""
    if not isinstance(payload, dict) or set(payload) != {
            "abi", "block_size", "weights", "scales", "tensor_scales"}:
        raise ValueError("invalid private G0.5 Action NVFP4 payload")
    if payload["abi"] != "striped_v2" or payload["block_size"] != 16:
        raise ValueError("G0.5 Action NVFP4 requires striped_v2 block16")
    shapes = ((2048, 1024), (1024, 2048), (4096, 1024),
              (4096, 1024), (1024, 4096))
    if any(len(payload[key]) != 5 for key in ("weights", "scales", "tensor_scales")):
        raise ValueError("G0.5 Action NVFP4 requires five projection families")
    for index, (n, k) in enumerate(shapes):
        for key, shape in (("weights", (n, k // 2)), ("scales", (k // 16, n))):
            values = payload[key][index]
            if len(values) != 24 or any(
                    not isinstance(x, torch.Tensor) or x.device.type != "cpu"
                    or x.dtype != torch.uint8 or tuple(x.shape) != shape
                    or not x.is_contiguous() for x in values):
                raise ValueError(f"invalid G0.5 Action NVFP4 {key} family {index}")
            if key == "scales" and any(bool((x > 126).any()) for x in values):
                raise ValueError("G0.5 Action NVFP4 block scales must be finite nonnegative E4M3FN")
        table = payload["tensor_scales"][index]
        if (not isinstance(table, torch.Tensor) or table.device.type != "cpu"
                or table.dtype != torch.float32 or tuple(table.shape) != (24,)
                or not table.is_contiguous() or not bool(torch.isfinite(table).all())
                or not bool((table > 0).all())):
            raise ValueError("G0.5 Action NVFP4 tensor scales must be finite positive CPU FP32[24]")
