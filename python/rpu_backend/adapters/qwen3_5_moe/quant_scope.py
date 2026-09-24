"""Qwen3.5-MoE mixed-FP8 checkpoint scope.

Shared by the offline converter and the adapter-local loader. Keeping the
scope beside the adapter mirrors dense Qwen3.5 and prevents conversion,
loading, and RPU installation from silently selecting different projections.

This module is deliberately stdlib-only so the offline converter can import it
without importing torch or the compiled backend.
"""
from __future__ import annotations

from collections.abc import Mapping

TEXT_PREFIX = "model.language_model.layers."
FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"

ROUTED_EXPERT_ROLES = (
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
)
FULL_ATTN_FP8_ROLES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
)
GDN_FP8_ROLES = (
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.out_proj.weight",
)

# Kept for the converter's public compatibility surface.
ROUTED_EXPERT_SUFFIXES = tuple(f".{role}" for role in ROUTED_EXPERT_ROLES)

FP8_METHOD = "qwen3_5_moe_mixed_fp8"
FP8_FORMAT = "fp8_e4m3_per_output"
FP8_SCOPE = "routed_experts_and_text_mixers"
FP8_PROFILE = "fp8_main_fp8_gdn"
EXACT_PROFILE_ID = "qwen3_5_moe_35b_a3b_tp8_mixed_e4m3"
FP16_PRESERVE = (
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.conv1d",
    "norms/A_log/dt_bias",
    "embed_tokens/lm_head",
    "router/shared_expert/shared_expert_gate",
    "vision",
)

_EXACT_TEXT_FIELDS = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 2048,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "vocab_size": 248320,
    "full_attention_interval": 4,
    "hidden_act": "silu",
    "attention_bias": False,
    "attn_output_gate": True,
    "rms_norm_eps": 1e-6,
}
_EXACT_ROPE_FIELDS = {
    "rope_type": "default",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
}
_EXACT_VISION_FIELDS = {
    "model_type": "qwen3_5_moe",
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_heads": 16,
    "out_hidden_size": 2048,
    "in_channels": 3,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "num_position_embeddings": 2304,
    "hidden_act": "gelu_pytorch_tanh",
}


def _value(owner, name, default=None):
    if isinstance(owner, Mapping):
        return owner.get(name, default)
    return getattr(owner, name, default)


def validate_exact_profile(config, *, require_quant: bool = True) -> dict:
    """Validate the only public sparse-MoE profile before allocation.

    Core topology is deliberately validated by the adapter because it belongs
    to ``rpu_execution`` rather than the checkpoint.  Everything encoded by
    the model/config is checked here so a same-shape semantic variant cannot
    enter the 35B-A3B native owner.
    """
    text = _value(config, "text_config", config)
    vision = _value(config, "vision_config")
    mismatches = {}
    if _value(config, "model_type") != "qwen3_5_moe":
        mismatches["model_type"] = _value(config, "model_type")
    if _value(config, "tie_word_embeddings") is not False:
        mismatches["tie_word_embeddings"] = _value(
            config, "tie_word_embeddings"
        )
    architectures = _value(config, "architectures")
    if architectures not in (None, ["Qwen3_5MoeForConditionalGeneration"],
                              ("Qwen3_5MoeForConditionalGeneration",)):
        mismatches["architectures"] = architectures
    for name, expected in _EXACT_TEXT_FIELDS.items():
        value = _value(text, name)
        if value != expected or (
            isinstance(expected, bool) and type(value) is not bool
        ):
            mismatches[f"text_config.{name}"] = value
    expected_layers = [LINEAR_ATTENTION] * 3 + [FULL_ATTENTION]
    layer_types = _value(text, "layer_types")
    if layer_types != expected_layers * 10:
        mismatches["text_config.layer_types"] = layer_types
    rope = _value(text, "rope_parameters")
    if not isinstance(rope, Mapping):
        mismatches["text_config.rope_parameters"] = rope
    else:
        for name, expected in _EXACT_ROPE_FIELDS.items():
            value = rope.get(name)
            if value != expected or (
                isinstance(expected, bool) and type(value) is not bool
            ):
                mismatches[f"text_config.rope_parameters.{name}"] = value
    if vision is None:
        mismatches["vision_config"] = None
    else:
        for name, expected in _EXACT_VISION_FIELDS.items():
            value = _value(vision, name)
            if value != expected:
                mismatches[f"vision_config.{name}"] = value
        if _value(vision, "deepstack_visual_indexes", []) not in ([], ()):
            mismatches["vision_config.deepstack_visual_indexes"] = _value(
                vision, "deepstack_visual_indexes"
            )
    if mismatches:
        raise ValueError(
            "Qwen3.5-MoE supports only the exact 35B-A3B profile; "
            f"mismatched fields: {sorted(mismatches)}"
        )
    quant = _value(config, "quant_config")
    if require_quant and quant is None:
        raise ValueError(
            "Qwen3.5-35B-A3B requires the controlled mixed-E4M3 checkpoint; "
            "BF16/FP16 and official block-FP8 checkpoints are not admitted"
        )
    if quant is not None:
        quant = validate_quant_config(quant, layer_types)
    return {"profile": EXACT_PROFILE_ID, "quant_config": quant}


def validate_layer_types(layer_types) -> tuple[str, ...]:
    layer_types = tuple(layer_types)
    bad = sorted(set(layer_types) - {FULL_ATTENTION, LINEAR_ATTENTION})
    if bad:
        raise ValueError(f"unsupported Qwen3.5-MoE layer_types {bad}")
    return layer_types


def _layer_and_role(name: str) -> tuple[int, str] | None:
    if not name.startswith(TEXT_PREFIX):
        return None
    layer_text, separator, role = name[len(TEXT_PREFIX):].partition(".")
    if not separator or not layer_text.isdigit():
        return None
    return int(layer_text), role


def is_routed_expert_weight(name: str) -> bool:
    parsed = _layer_and_role(name)
    return parsed is not None and parsed[1] in ROUTED_EXPERT_ROLES


def is_text_mixer_weight(name: str, layer_types) -> bool:
    layer_types = tuple(layer_types)
    parsed = _layer_and_role(name)
    if parsed is None:
        return False
    layer, role = parsed
    if layer >= len(layer_types):
        return False
    expected = (
        FULL_ATTN_FP8_ROLES
        if layer_types[layer] == FULL_ATTENTION
        else GDN_FP8_ROLES
    )
    return role in expected


def is_fp8_weight(name: str, layer_types) -> bool:
    return is_routed_expert_weight(name) or is_text_mixer_weight(
        name, layer_types
    )


def expected_fp8_weight_names(layer_types) -> tuple[str, ...]:
    layer_types = validate_layer_types(layer_types)
    names: list[str] = []
    for layer, layer_type in enumerate(layer_types):
        roles = ROUTED_EXPERT_ROLES + (
            FULL_ATTN_FP8_ROLES
            if layer_type == FULL_ATTENTION
            else GDN_FP8_ROLES
        )
        names.extend(f"{TEXT_PREFIX}{layer}.{role}" for role in roles)
    return tuple(names)


def scale_name(weight_name: str) -> str:
    return weight_name + "_scale"


def validate_quant_config(quant_config, layer_types) -> dict:
    """Validate the exact checkpoint declaration emitted by the converter."""
    if not isinstance(quant_config, Mapping):
        raise TypeError(
            "Qwen3.5-MoE quant_config must be a JSON object, got "
            f"{type(quant_config).__name__}"
        )
    required = {
        "method": FP8_METHOD,
        "format": FP8_FORMAT,
        "weight_mode": 5,
        "qaxis": -1,
        "scale_dtype": "float16",
        "scope": FP8_SCOPE,
        "profile": FP8_PROFILE,
        "official_block_fp8_compatible": False,
    }
    mismatched = {
        key: (quant_config.get(key), expected)
        for key, expected in required.items()
        if quant_config.get(key) != expected
    }
    if mismatched:
        raise ValueError(
            "unsupported Qwen3.5-MoE mixed-FP8 declaration: "
            f"{mismatched}"
        )

    expected_names = set(expected_fp8_weight_names(layer_types))
    declared = quant_config.get("quantized_tensors")
    if not isinstance(declared, list) or not all(
        isinstance(name, str) for name in declared
    ):
        raise ValueError(
            "Qwen3.5-MoE quant_config.quantized_tensors must be a list[str]"
        )
    declared_names = set(declared)
    if len(declared) != len(declared_names) or declared_names != expected_names:
        raise ValueError(
            "Qwen3.5-MoE quantized tensor inventory does not match the model "
            f"scope: missing={sorted(expected_names - declared_names)[:8]}, "
            f"extra={sorted(declared_names - expected_names)[:8]}"
        )
    declared_count = quant_config.get("quantized_tensor_count")
    if declared_count != len(expected_names):
        raise ValueError(
            "Qwen3.5-MoE quantized_tensor_count must be "
            f"{len(expected_names)}, got {declared_count!r}"
        )
    if quant_config.get("fp16_preserve") != list(FP16_PRESERVE):
        raise ValueError(
            "Qwen3.5-MoE fp16_preserve does not match the controlled profile"
        )
    return dict(quant_config)


__all__ = [
    "EXACT_PROFILE_ID",
    "FP8_FORMAT",
    "FP8_METHOD",
    "FP8_PROFILE",
    "FP8_SCOPE",
    "FP16_PRESERVE",
    "FULL_ATTENTION",
    "FULL_ATTN_FP8_ROLES",
    "GDN_FP8_ROLES",
    "LINEAR_ATTENTION",
    "ROUTED_EXPERT_ROLES",
    "ROUTED_EXPERT_SUFFIXES",
    "TEXT_PREFIX",
    "expected_fp8_weight_names",
    "is_fp8_weight",
    "is_routed_expert_weight",
    "is_text_mixer_weight",
    "scale_name",
    "validate_layer_types",
    "validate_exact_profile",
    "validate_quant_config",
]
