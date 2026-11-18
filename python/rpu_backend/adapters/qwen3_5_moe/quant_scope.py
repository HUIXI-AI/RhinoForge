"""Qwen3.5-MoE checkpoint profiles and their execution contracts.

Shared by the offline converter and the adapter-local loader. Keeping the
scope beside the adapter mirrors dense Qwen3.5 and prevents conversion,
loading, and device installation from silently selecting different projections.

This module is deliberately stdlib-only so the offline converter can import it
without importing torch or the compiled backend.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

TEXT_PREFIX = "model.language_model.layers."
FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"
# Transformers 5.19 gives the same Vision configuration its own model type.
VISION_MODEL_TYPES = frozenset({"qwen3_5_moe", "qwen3_5_moe_vision"})

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
RTN_PROFILE_ID = "qwen3_5_moe_35b_a3b_tp8_routed_rtn_w4a16_g128"
RTN_METHOD = "qwen3_5_moe_routed_rtn_uint4"
GPTQ_PROFILE_ID = "qwen3_5_moe_35b_a3b_tp8_routed_gptq_w4a16_g128"
GPTQ_METHOD = "qwen3_5_moe_routed_gptq_uint4"
INT8_PROFILE_ID = "qwen3_5_moe_35b_a3b_tp8_routed_rtn_w8a16"
INT8_METHOD = "qwen3_5_moe_routed_int8"
INT8_FORMAT = "int8_per_output"
INT8_SCOPE = "routed_experts"


@dataclass(frozen=True)
class RuntimeCapabilities:
    routed_weight_mode: int
    mixer_precision: str
    allowed_cores: tuple[int, ...]
    text_only: bool


# Routed quantization does not change the ordinary FP16 Vision tower.  Both
# INT4 sources use the existing exact TP8 Vision/text bridge and owner policy.
_ROUTED_INT4_RUNTIME = RuntimeCapabilities(6, "float16", (4, 6, 8), False)
_MIXED_FP8_RUNTIME = RuntimeCapabilities(5, "fp8_e4m3", (4, 6, 8), False)
# Signed INT8 routed experts use the grouped kernel's mode 1 at the routed
# width E4M3 uses (TP6 pads I512 to 576); mixers stay FP16 as for INT4.
_ROUTED_INT8_RUNTIME = RuntimeCapabilities(1, "float16", (4, 6, 8), False)


def runtime_capabilities(profile_id: str) -> RuntimeCapabilities:
    """Resolve only admitted profile identities, never a tensor dtype/mode hint."""
    if profile_id in (RTN_PROFILE_ID, GPTQ_PROFILE_ID):
        return _ROUTED_INT4_RUNTIME
    if profile_id == EXACT_PROFILE_ID:
        return _MIXED_FP8_RUNTIME
    if profile_id == INT8_PROFILE_ID:
        return _ROUTED_INT8_RUNTIME
    raise ValueError(f"Unknown Qwen3.5-MoE profile: {profile_id!r}")


RTN_RECIPE = {
    "algorithm": "RTN", "bits": 4, "group_size": 128, "sym": True,
    "desc_act": False, "calibration": False, "error_feedback": False,
    "scale_formula": "2*max(abs(group))/15; all-zero group: 2/15",
    "rounding": "nearest_ties_to_even", "quantization_arithmetic": "float32",
    "zero": 8, "q_range": [0, 15], "scale_storage": "float16",
    "format": "gptq_v1", "qzeros_encoding": "zero_minus_one",
    "scale_underflow": "IEEE float16 cast may produce zero; preserve and count, no floor",
}
FP16_PRESERVE = (
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.conv1d",
    "norms/A_log/dt_bias",
    "embed_tokens/lm_head",
    "router/shared_expert/shared_expert_gate",
    "vision",
)
# Routed INT8 quantizes only the experts; the text mixers join the FP16 copy.
INT8_FP16_PRESERVE = ("self_attn/linear_attn projections",) + FP16_PRESERVE

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


def _declares(owner, name):
    return name in owner if isinstance(owner, Mapping) else hasattr(owner, name)


def has_default_vision_rope(config) -> bool:
    """Admit the legacy fixed axial RoPE and its explicit HF representation."""
    rope = _value(config, "rope_parameters")
    return rope is None or (
        isinstance(rope, Mapping)
        and dict(rope) == {"rope_type": "axial", "rope_theta": 10000.0}
    )


def validate_exact_profile(config, *, require_quant: bool = True) -> dict:
    """Validate exact geometry and one supported checkpoint profile before allocation.

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
            matches = (isinstance(value, str) and value in VISION_MODEL_TYPES
                       if name == "model_type" else value == expected)
            if not matches:
                mismatches[f"vision_config.{name}"] = value
        if not has_default_vision_rope(vision):
            mismatches["vision_config.rope_parameters"] = _value(vision, "rope_parameters")
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
    if _declares(config, "rtn_metadata"):
        if _declares(config, "quant_config"):
            raise ValueError("Qwen3.5-MoE RTN cannot also declare mixed-FP8 quant_config")
        return {"profile": RTN_PROFILE_ID,
                "quant_config": validate_rtn_config(config)}
    if _declares(config, "quantization_config"):
        if _declares(config, "quant_config"):
            raise ValueError("Qwen3.5-MoE cannot combine mixed-FP8 and GPTQ declarations")
        return {"profile": GPTQ_PROFILE_ID,
                "quant_config": validate_gptq_config(config)}
    if require_quant and quant is None:
        raise ValueError(
            "Qwen3.5-35B-A3B requires a supported mixed-E4M3, RTN, routed INT8 or official GPTQ checkpoint; "
            "BF16/FP16 and official block-FP8 checkpoints are not admitted"
        )
    if isinstance(quant, Mapping) and quant.get("method") == INT8_METHOD:
        return {"profile": INT8_PROFILE_ID,
                "quant_config": validate_int8_config(quant, layer_types)}
    if quant is not None:
        quant = validate_quant_config(quant, layer_types)
    return {"profile": EXACT_PROFILE_ID, "quant_config": quant}


def validate_rtn_config(config) -> dict:
    """Recognize the audited routed-only RTN artifact, not arbitrary GPTQ."""
    metadata = _value(config, "rtn_metadata")
    expected_metadata = {
        "algorithm": "RTN", "manifest": "derivation_manifest.json",
        "calibration": False,
    }
    if (not isinstance(metadata, Mapping) or dict(metadata) != expected_metadata
            or metadata.get("calibration") is not False):
        raise ValueError("Qwen3.5-MoE RTN requires the exact derivation metadata")
    quant = _value(config, "quantization_config")
    required = {"bits": 4, "group_size": 128, "sym": True,
                "desc_act": False, "quant_method": "gptq"}
    if not isinstance(quant, Mapping) or any(
        quant.get(key) != value or type(quant.get(key)) is not type(value)
        for key, value in required.items()
    ):
        raise ValueError("Qwen3.5-MoE RTN requires symmetric G128 GPTQ-v1 storage")
    if quant.get("checkpoint_format") not in (None, "gptq", "gptq_v1"):
        raise ValueError("Qwen3.5-MoE RTN requires GPTQ-v1 checkpoint_format")
    return {
        "method": RTN_METHOD, "profile": RTN_PROFILE_ID, "weight_mode": 6,
        "group_size": 128, "scope": "routed_experts", "scale_dtype": "float16",
        "format": "gptq_v1", "qzeros_encoding": "zero_minus_one",
    }


def validate_gptq_config(config) -> dict:
    """Admit the official routed-only symmetric layout, not generic GPTQ.

    The checked Qwen release omits checkpoint_format. Its symmetric MoE
    reference uses implicit zero8; every stored zero/g_idx is still checked
    when read. A file with explicit format metadata needs its own assessment.
    """
    if _declares(config, "rtn_metadata") or _declares(config, "quant_config"):
        raise ValueError("Official GPTQ cannot also declare RTN or mixed-FP8 metadata")
    quant = _value(config, "quantization_config")
    required = {"bits": 4, "group_size": 128, "sym": True,
                "desc_act": False, "quant_method": "gptq",
                "static_groups": False, "true_sequential": True,
                "damp_percent": 0.01}
    if not isinstance(quant, Mapping) or any(
        quant.get(key) != value or type(quant.get(key)) is not type(value)
        for key, value in required.items()
    ):
        raise ValueError("Qwen3.5-MoE official GPTQ requires the released symmetric G128 declaration")
    dynamic = {"lm_head": {}, "model.language_model.embed_tokens": {},
               "-:.*attn.*": {}, "-:.*shared_expert.*": {},
               "-:.*mtp.*": {}, "-:.*visual.*": {}}
    if quant.get("dynamic") != dynamic or quant.get("modules_to_not_convert") != []:
        raise ValueError("Qwen3.5-MoE official GPTQ requires its routed-only dynamic scope")
    unknown = set(quant) - set(required) - {"dynamic", "modules_to_not_convert"}
    if unknown:
        raise ValueError(f"Unsupported official GPTQ fields/checkpoint_format: {sorted(unknown)}")
    return {
        "method": GPTQ_METHOD, "profile": GPTQ_PROFILE_ID, "weight_mode": 6,
        "group_size": 128, "scope": "routed_experts", "scale_dtype": "float16",
        "format": "qwen3_5_moe_official_symmetric_gptq",
        "qzeros_encoding": "direct_zero", "effective_zero": 8,
    }


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


def expected_int8_weight_names(layer_types) -> tuple[str, ...]:
    """The routed-only INT8 scope: both fused expert tensors of every layer."""
    layer_types = validate_layer_types(layer_types)
    return tuple(f"{TEXT_PREFIX}{layer}.{role}"
                 for layer in range(len(layer_types)) for role in ROUTED_EXPERT_ROLES)


def validate_int8_config(quant_config, layer_types) -> dict:
    """Validate the routed-only INT8 declaration emitted by the converter."""
    required = {
        "method": INT8_METHOD,
        "format": INT8_FORMAT,
        "weight_mode": 1,
        "qaxis": -1,
        "scale_dtype": "float16",
        "scope": INT8_SCOPE,
        "profile": INT8_PROFILE_ID,
        "symmetric": True,
        "calibration": False,
    }
    mismatched = {
        key: (quant_config.get(key), expected)
        for key, expected in required.items()
        if quant_config.get(key) != expected
        or type(quant_config.get(key)) is not type(expected)
    }
    if mismatched:
        raise ValueError(
            f"unsupported Qwen3.5-MoE routed INT8 declaration: {mismatched}")
    expected_names = list(expected_int8_weight_names(layer_types))
    if (quant_config.get("quantized_tensors") != sorted(expected_names)
            or quant_config.get("quantized_tensor_count") != len(expected_names)):
        raise ValueError(
            "Qwen3.5-MoE routed INT8 must quantize exactly the "
            f"{len(expected_names)} fused routed-expert tensors")
    if quant_config.get("fp16_preserve") != list(INT8_FP16_PRESERVE):
        raise ValueError(
            "Qwen3.5-MoE routed INT8 fp16_preserve does not match the controlled profile")
    unknown = set(quant_config) - set(required) - {
        "quantized_tensors", "quantized_tensor_count", "fp16_preserve"}
    if unknown:
        raise ValueError(f"Unsupported routed INT8 fields: {sorted(unknown)}")
    return dict(quant_config)


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
    "RTN_PROFILE_ID", "RTN_METHOD", "RTN_RECIPE", "validate_rtn_config",
    "GPTQ_PROFILE_ID", "GPTQ_METHOD", "validate_gptq_config", "runtime_capabilities",
    "INT8_FORMAT", "INT8_FP16_PRESERVE", "INT8_METHOD", "INT8_PROFILE_ID", "INT8_SCOPE",
    "expected_int8_weight_names", "validate_int8_config",
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
