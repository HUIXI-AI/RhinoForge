"""Qwen3.5 quantization scope and per-projection precision contract.

Shared by the offline converter (``rpu_backend.quant.convert_qwen3_5``), the
checkpoint loader (``adapters/qwen3_5/loader.py``) and the device install path
(``text.py::_iter_text_projections``). Keeping one list avoids the failure mode
where a checkpoint is converted over one set of modules and loaded over another.

Deliberately stdlib-only: the converter computes the scope from a raw
``config.json`` and must not need torch or the compiled backend.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Mapping

# Checkpoint key prefix of the text backbone inside
# ``Qwen3_5ForConditionalGeneration``.
TEXT_PREFIX = "model.language_model."

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"

FP16 = "fp16"
W8A16 = "w8a16"
W4A16_PGRP = "w4a16_pgrp"
W4A16_AWQ = "w4a16_awq"
NVFP4 = "nvfp4"
QUANTTRIO_AWQ_PROFILE = "quanttrio_awq_g128"
ACTIVATION_AWQ_PROFILE = "activation_awq_g128_v1"
AWQ_PROFILES = (QUANTTRIO_AWQ_PROFILE, ACTIVATION_AWQ_PROFILE)
NAMED_MIXED_PROFILES = ("w4_mlp_only", "w4_mlp_w8_attn_gdn",
                        "w4_main_w8_gdn", "w8_main_w8_gdn")
SUPPORTED_QUANT_METHODS = (FP16, W8A16, W4A16_PGRP, W4A16_AWQ, NVFP4)
PGRP_SUPPORTED_GROUP_SIZES = (32, 64, 128)
DECLARED_DTYPES = (FP16, W8A16, "w4a16")

# The first Qwen3.8-27B W8A16 bundle predates the v1/v2 declaration used by
# ``convert_qwen3_5``.  It quantizes the MLP and full-attention projections,
# leaves all GDN projections in FP16, and stores MTP scales even though the device
# text path intentionally does not instantiate MTP.  Keep this policy explicit
# instead of treating it as the v1 uniform-W8 format (which would silently ask
# for scales on every GDN projection).
LEGACY_W8A16_PROFILE = "w8_main_fp16_gdn_legacy"

# Dense text NVFP4 is quantized on install from the official FP16 checkpoint
# (the Qwen3 / Qwen3-VL precedent); no NVFP4 checkpoint format exists, so the
# loader's storage check refuses this policy on disk. GDN b/a stay FP16: their
# per-core width (16) is below the smallest NVFP4 tile. Vision, MTP, embedding,
# lm_head and conv1d are outside the text projection scope and stay FP16.
NVFP4_TEXT_PROFILE = "nvfp4_main_fp16_ba"
NVFP4_INSTALL_STORAGE = "fp16_install"

# Emission order matches the historical `_iter_text_projections`: the MLP half
# first, then the layer's token mixer. Install-path behaviour is unchanged.
_MLP_PROJECTIONS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
_FULL_ATTN_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
_LINEAR_ATTN_PROJECTIONS = (
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.out_proj",
)

_PROJECTION_ROLES = frozenset(
    _MLP_PROJECTIONS + _FULL_ATTN_PROJECTIONS + _LINEAR_ATTN_PROJECTIONS
)


@dataclass(frozen=True)
class QuantSpec:
    """Logical checkpoint format for one text projection."""

    method: str
    group_size: int | None = None

    def __post_init__(self):
        if self.method not in SUPPORTED_QUANT_METHODS:
            raise ValueError(
                f"unsupported Qwen3.5 quant method {self.method!r}; expected one of "
                f"{SUPPORTED_QUANT_METHODS}"
            )
        if self.method in (W4A16_PGRP, W4A16_AWQ):
            if (not isinstance(self.group_size, int)
                    or isinstance(self.group_size, bool)
                    or self.group_size not in PGRP_SUPPORTED_GROUP_SIZES):
                raise ValueError(
                    f"method={W4A16_PGRP} requires group_size in "
                    f"{PGRP_SUPPORTED_GROUP_SIZES}, got {self.group_size!r}"
                )
            if self.method == W4A16_AWQ and self.group_size != 128:
                raise ValueError("w4a16_awq requires group_size=128")
        elif self.group_size is not None:
            raise ValueError(
                f"method={self.method} does not accept group_size, got "
                f"{self.group_size!r}"
            )


def nvfp4_text_quant_config() -> dict:
    """Return a fresh install-time dense text NVFP4 policy.

    The adapter stamps it on the top-level config of an FP16 model; every
    consumer that branches on ``quant_config`` then takes its quantized path.
    """
    return {
        "format_version": 2,
        "architecture": "qwen3_5",
        "activation_dtype": "float16",
        "profile": NVFP4_TEXT_PROFILE,
        "default": {"method": NVFP4},
        "projection_overrides": {
            "linear_attn.in_proj_b": {"method": FP16},
            "linear_attn.in_proj_a": {"method": FP16},
        },
        "storage": NVFP4_INSTALL_STORAGE,
        "scale_dtype": "float8_e4m3fn",
    }


def is_nvfp4_text_policy(quant_config) -> bool:
    return (isinstance(quant_config, Mapping)
            and quant_config.get("profile") == NVFP4_TEXT_PROFILE)


def is_full_from_layer_types(layer_types):
    """Map ``layer_types`` to the per-layer list[int] used across the adapter."""
    layer_types = list(layer_types)
    bad = sorted(set(layer_types) - {FULL_ATTENTION, LINEAR_ATTENTION})
    if bad:
        raise ValueError(f"unsupported Qwen3.5 layer_types {bad}")
    return [1 if t == FULL_ATTENTION else 0 for t in layer_types]


def quant_relpaths(is_full):
    """Return every supported text projection path, relative to the backbone."""
    relpaths = []
    for i, full in enumerate(is_full):
        relpaths.extend(f"layers.{i}.{proj}" for proj in _MLP_PROJECTIONS)
        mixer = _FULL_ATTN_PROJECTIONS if full else _LINEAR_ATTN_PROJECTIONS
        relpaths.extend(f"layers.{i}.{proj}" for proj in mixer)
    return relpaths


def projection_role(relpath):
    """Return the canonical role (for example ``linear_attn.in_proj_qkv``)."""
    parts = relpath.split(".", 2)
    if (len(parts) != 3 or parts[0] != "layers" or not parts[1].isdigit()
            or parts[2] not in _PROJECTION_ROLES):
        raise ValueError(f"invalid Qwen3.5 projection path {relpath!r}")
    return parts[2]


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Qwen3.5 {name} must be a JSON object, got {type(value).__name__}"
        )
    return value


def _spec(value, name):
    value = _mapping(value, name)
    unknown = sorted(set(value) - {"method", "group_size"})
    if unknown:
        raise ValueError(f"Qwen3.5 {name} has unknown field(s): {unknown}")
    if "method" not in value:
        raise ValueError(f"Qwen3.5 {name} requires method")
    return QuantSpec(value["method"], value.get("group_size"))


def _quanttrio_policy(is_full, profile=QUANTTRIO_AWQ_PROFILE):
    """Versioned per-module declaration; v2 role semantics remain unchanged."""
    return {
        "format_version": 3, "architecture": "qwen3_5",
        "profile": profile, "activation_dtype": "float16",
        "storage": "uint8", "scale_dtype": "float16",
        "default": {"method": FP16},
        "module_overrides": {
            f"layers.{i}.{role}": {"method": W4A16_AWQ, "group_size": 128}
            for i in range(1, len(is_full)) for role in _MLP_PROJECTIONS},
    }


def quanttrio_awq_config(config):
    """Return a fresh exact QuantTrio policy, or None for another profile.

    Accept JSON mappings and HF objects without modifying either. A claimed but
    malformed AWQ declaration raises, including conflicting internal/external
    declarations. This is shared by public preflight, loading and metadata checks.
    """
    get = _config_value
    quant = get(config, "quant_config")
    source = get(config, "quantization_config")
    declared = isinstance(quant, Mapping) and quant.get("profile") in AWQ_PROFILES
    profile = quant["profile"] if declared else QUANTTRIO_AWQ_PROFILE
    if profile == ACTIVATION_AWQ_PROFILE and source is not None:
        raise ValueError("locally calibrated AWQ must not claim an external AWQ source declaration")
    if source is None and not declared:
        return None
    if source is not None:
        source = _mapping(source, "quantization_config")
        expected = {"quant_method": "awq", "bits": 4, "group_size": 128,
                    "version": "gemm", "zero_point": True}
        if (any(source.get(k) != v or (type(v) in (int, bool) and type(source.get(k)) is not type(v))
                for k, v in expected.items())
                or set(source) - set(expected) - {"modules_to_not_convert"}
                or not isinstance(source.get("modules_to_not_convert"), list)
                or len(source["modules_to_not_convert"]) != 5
                or not all(isinstance(name, str) for name in source["modules_to_not_convert"])
                or set(source["modules_to_not_convert"]) != {
                    "visual", "linear_attn", "self_attn", "model.layers.0.", "mtp"}):
            raise ValueError("Qwen3.5 requires exact QuantTrio AWQ GEMM G128 direct-zero metadata")
    tc = get(config, "text_config")
    if tc is None or get(config, "model_type") != "qwen3_5":
        raise ValueError("QuantTrio AWQ requires a Qwen3.5 conditional-generation config")
    h = get(tc, "hidden_size")
    # Keep the accepted geometries explicit.  A matching 27B AWQ checkpoint
    # uses the same GEMM metadata and packed projection layout as the smaller
    # dense checkpoints, but must not be admitted by a loose hidden-size test.
    shapes = {2048: (6144, 24, 8, 2, 16), 2560: (9216, 32, 16, 4, 32),
              4096: (12288, 32, 16, 4, 32),
              5120: (17408, 64, 24, 4, 48)}
    row = shapes.get(h) if type(h) is int else None
    if row is None:
        raise ValueError("QuantTrio AWQ supports exact dense 2B/4B/9B/27B geometry")
    fields = dict(zip(("intermediate_size", "num_hidden_layers", "num_attention_heads",
                       "num_key_value_heads", "linear_num_value_heads"), row))
    fields.update(head_dim=256, linear_num_key_heads=16, linear_key_head_dim=128,
                  linear_value_head_dim=128, linear_conv_kernel_dim=4, vocab_size=248320)
    if any(type(get(tc, k)) is not int or get(tc, k) != v for k, v in fields.items()):
        raise ValueError("QuantTrio AWQ requires the exact dense text geometry")
    semantics = {"model_type": "qwen3_5_text", "hidden_act": "silu", "rms_norm_eps": 1e-6,
                 "attention_bias": False, "attn_output_gate": True, "use_cache": True,
                 "full_attention_interval": 4}
    if (any(get(tc, k) != v or (type(v) is bool and type(get(tc, k)) is not bool)
            for k, v in semantics.items())
            or get(config, "tie_word_embeddings") is not (h not in (4096, 5120))
            or get(config, "image_token_id") != 248056
            or any(get(owner, key) for owner in (config, tc) for key in
                   ("num_experts", "adaptive_mode", "mlp_only_layers"))
            or get(tc, "quant_config") is not None or get(tc, "quantization_config") is not None):
        raise ValueError("QuantTrio AWQ requires the original dense text semantics")
    layers = [FULL_ATTENTION if i % 4 == 3 else LINEAR_ATTENTION for i in range(row[1])]
    if list(get(tc, "layer_types", ())) != layers:
        raise ValueError("QuantTrio AWQ layer_types must match the original dense schedule")
    rope = get(tc, "rope_parameters")
    expected_rope = {"rope_type": "default", "rope_theta": 10_000_000,
                     "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                     "mrope_section": [11, 11, 10]}
    if not isinstance(rope, Mapping) or any(rope.get(k) != v for k, v in expected_rope.items()):
        raise ValueError("QuantTrio AWQ requires the original partial interleaved M-RoPE")
    canonical = _quanttrio_policy(is_full_from_layer_types(layers), profile)
    if quant is not None and quant != canonical:
        raise ValueError("QuantTrio AWQ quant_config conflicts with its exact per-module scope")
    return canonical


def resolve_quant_config(config):
    """Resolve raw checkpoint config to an executable policy without mutation."""
    awq = quanttrio_awq_config(config)
    if awq is not None:
        return awq
    quant = _config_value(config, "quant_config")
    tc = _config_value(config, "text_config", config)
    is_full = is_full_from_layer_types(_config_value(tc, "layer_types", ()))
    quant = normalize_quant_config(is_full, quant)
    resolve_quant_specs(is_full, quant)
    return quant


# Exact metadata of the pre-v1 bundle. Partial or conflicting legacy fields
# must never fall through to the documented v1 uniform-W8 policy.
_LEGACY_METADATA = {
    "method": W8A16,
    "mode": "per_channel_symmetric",
    "qaxis": 0,
    "skip_modules": ["lm_head"],
    "quantized_lm_head": False,
    "quantized_embed_tokens": False,
    "lm_head_untied": False,
    "embed_tokens_untied": False,
}
_LEGACY_MARKERS = frozenset({
    "quantized_lm_head", "quantized_embed_tokens",
    "lm_head_untied", "embed_tokens_untied",
})
_LEGACY_TEXT_GEOMETRY = {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}


def is_legacy_w8a16_config(quant_config) -> bool:
    """Recognize only the shipped pre-v1 declaration, with no conflicting keys.

    This identifies a quantization format, not a model architecture. Call
    ``validate_legacy_text_profile`` before choosing the text-only model loader.
    """
    return (
        isinstance(quant_config, Mapping)
        and set(quant_config) == set(_LEGACY_METADATA)
        and all(type(quant_config[key]) is type(value)
                and quant_config[key] == value
                for key, value in _LEGACY_METADATA.items())
    )


def legacy_w8a16_quant_config() -> dict:
    """Return the canonical scope for the legacy Qwen3.8-27B checkpoint.

    The returned object is a normal v2 policy, so every current consumer can
    validate it with :func:`resolve_quant_specs`.  It is intentionally a fresh
    dict on each call because callers stamp it onto mutable HF config objects.
    """
    quantized_roles = _MLP_PROJECTIONS + _FULL_ATTN_PROJECTIONS
    return {
        "format_version": 2,
        "architecture": "qwen3_5",
        "activation_dtype": "float16",
        "profile": LEGACY_W8A16_PROFILE,
        "default": {"method": FP16},
        "projection_overrides": {
            role: {"method": W8A16} for role in quantized_roles
        },
        "storage": "int8",
        "scale_dtype": "float16",
        # Vision/MTP are absent from the legacy text runtime; embeddings,
        # lm_head and conv1d remain ordinary floating-point parameters.
        "skip_modules": ["lm_head", "embed_tokens", "visual", "mtp", "conv1d"],
        "legacy_source": "qwen3_8_w8a16",
    }


def normalize_quant_config(is_full, quant_config):
    """Expand exact legacy metadata without changing modern v1/v2 policies."""
    if is_legacy_w8a16_config(quant_config):
        return legacy_w8a16_quant_config()
    if isinstance(quant_config, Mapping) and _LEGACY_MARKERS.intersection(quant_config):
        raise ValueError(
            "Qwen3.5 legacy W8A16 metadata is incomplete or conflicting; "
            "expected the exact pre-v1 MLP/full-attention declaration")
    return quant_config


def is_mlp_only_text_profile(config):
    """Identify the named public recipe without constraining generic v2 scopes."""
    quant = _config_value(config, "quant_config")
    return isinstance(quant, Mapping) and quant.get("profile") == "w4_mlp_only"


def validate_mlp_only_text_profile(config):
    """Check named recipe geometry and scope for source-only consumers.

    Runtime admission additionally checks dense text semantics in cores.py.
    Other v2 profiles and generic converter scopes retain their prior behavior.
    """
    if not is_mlp_only_text_profile(config):
        return False
    quant = _config_value(config, "quant_config")
    if "layer_overrides" in quant:
        raise ValueError("Qwen3.5 MLP-only W4 does not support layer_overrides; every layer must use the same MLP scope")
    tc = _config_value(config, "text_config", config)
    geometries = {
        2048: (6144, 24, 8, 2, 16),
        5120: (17408, 64, 24, 4, 48),
    }
    hidden = _config_value(tc, "hidden_size")
    row = geometries.get(hidden)
    if row is None:
        raise ValueError("Qwen3.5 MLP-only W4 requires exact 2B geometry or official 27B geometry")
    expected = dict(hidden_size=hidden, intermediate_size=row[0], num_hidden_layers=row[1],
                    num_attention_heads=row[2], num_key_value_heads=row[3], head_dim=256,
                    linear_num_key_heads=16, linear_num_value_heads=row[4],
                    linear_key_head_dim=128, linear_value_head_dim=128,
                    linear_conv_kernel_dim=4, vocab_size=248320)
    if any(type(_config_value(tc, name)) is not int or _config_value(tc, name) != value
           for name, value in expected.items()):
        raise ValueError("Qwen3.5 MLP-only W4 requires exact 2B geometry or official 27B geometry")
    layers = [FULL_ATTENTION if i % 4 == 3 else LINEAR_ATTENTION for i in range(row[1])]
    if list(_config_value(tc, "layer_types", ())) != layers:
        raise ValueError("Qwen3.5 MLP-only W4 requires the exact hybrid schedule")
    specs = resolve_quant_specs(is_full_from_layer_types(layers),
                                _config_value(config, "quant_config"))
    if any(spec.method != (W4A16_PGRP if ".mlp." in name else FP16)
           or (".mlp." in name and spec.group_size != 32)
           for name, spec in specs.items()):
        raise ValueError("Qwen3.5 MLP-only W4 requires all-layer MLP W4 group32 with FP16 Attention/GDN")
    return True


def _config_value(config, name, default=None):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def validate_legacy_text_profile(config) -> bool:
    """Return False for other policies; validate the exact legacy text profile.

    Accepts raw config dictionaries and HF configs, including the flat text
    config stamped by the loader. Invalid legacy policy or geometry raises
    ValueError before a model skeleton, cache, or native handle is created.
    """
    text_config = _config_value(config, "text_config", config)
    quant = normalize_quant_config(None, _config_value(config, "quant_config"))
    if not isinstance(quant, Mapping) or quant.get("profile") != LEGACY_W8A16_PROFILE:
        return False
    mismatches = {
        key: _config_value(text_config, key)
        for key, expected in _LEGACY_TEXT_GEOMETRY.items()
        if type(_config_value(text_config, key)) is not int
        or _config_value(text_config, key) != expected
    }
    layer_types = _config_value(text_config, "layer_types")
    expected_layers = [LINEAR_ATTENTION] * 3 + [FULL_ATTENTION]
    if layer_types != expected_layers * 16:
        mismatches["layer_types"] = layer_types
    if mismatches:
        raise ValueError(
            "Qwen3.8 legacy text profile requires the exact 27B geometry; "
            f"mismatched fields: {sorted(mismatches)}")
    # These fields alter HF loading or execution while preserving every weight
    # shape. In particular, tie_weights() would overwrite the separately loaded
    # lm_head if a same-shape checkpoint changed tie_word_embeddings to True.
    semantics = {
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "attention_bias": False,
        "attn_output_gate": True,
        "output_gate_type": "swish",
        "rms_norm_eps": 1e-6,
    }
    mismatches = []
    for name, expected in semantics.items():
        value = _config_value(text_config, name, expected)
        if value != expected or (isinstance(expected, bool) and type(value) is not bool):
            mismatches.append(name)
    if _config_value(config, "tie_word_embeddings", False) is not False:
        mismatches.append("tie_word_embeddings")
    # A parsed HF config carries these concrete RoPE parameters. Match the
    # values consumed by the native adapter rather than treating a missing
    # partial factor/section as the original checkpoint's declaration.
    rope = _config_value(text_config, "rope_parameters")
    if not isinstance(rope, Mapping):
        mismatches.append("rope_parameters")
    else:
        expected_rope = {
            "rope_type": "default", "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True, "mrope_section": [11, 11, 10],
        }
        for name, expected in expected_rope.items():
            default = expected if name in ("rope_type", "mrope_interleaved") else None
            value = rope.get(name, default)
            if value != expected or (isinstance(expected, bool) and type(value) is not bool):
                mismatches.append(f"rope_parameters.{name}")
        for name in ("rope_theta", "partial_rotary_factor"):
            if _config_value(text_config, name, expected_rope[name]) != expected_rope[name]:
                mismatches.append(name)
    if mismatches:
        raise ValueError(
            "Qwen3.8 legacy text profile requires the original 27B semantics; "
            f"mismatched fields: {sorted(set(mismatches))}")
    resolve_quant_specs(is_full_from_layer_types(layer_types), quant)
    return True


def validate_modern_27b_profile(config, *, allow_fp16=False) -> bool:
    """Validate official 27B semantics and uniform quantization, without torch.

    Other sizes, named mixed-precision recipes and the independently validated
    legacy policy return False.
    An apparent modern 27B with conflicting metadata raises before allocation.
    ``allow_fp16`` enables the dense FP16 runtime path.  Callers that only
    inspect quantized assets can leave it disabled to retain the stricter
    admission policy.
    """
    tc = _config_value(config, "text_config", config)
    if _config_value(tc, "hidden_size") != 5120:
        return False
    # Named MLP-only and QuantTrio AWQ recipes use their own loader and core
    # contracts.  Do not route a valid 27B mixed-precision declaration through
    # the uniform dense-27B graph policy below.
    quant = _config_value(config, "quant_config")
    named_mixed = isinstance(quant, Mapping) and quant.get("profile") in (
        "w4_mlp_only", "w4_mlp_w8_attn_gdn", "w4_main_w8_gdn", "w8_main_w8_gdn")
    if named_mixed:
        validate_named_27b_mixed_profile(config)
        return False
    if quanttrio_awq_config(config) is not None:
        return False
    if validate_legacy_text_profile(config):
        return False
    bad = []
    geometry = dict(_LEGACY_TEXT_GEOMETRY, vocab_size=248320)
    for name, expected in geometry.items():
        value = _config_value(tc, name)
        if type(value) is not int or value != expected:
            bad.append(name)
    expected_layers = ([LINEAR_ATTENTION] * 3 + [FULL_ATTENTION]) * 16
    if list(_config_value(tc, "layer_types", ()) or ()) != expected_layers:
        bad.append("layer_types")
    for name, expected in (("model_type", "qwen3_5_text"), ("hidden_act", "silu"),
                           ("attention_bias", False), ("attn_output_gate", True),
                           ("rms_norm_eps", 1e-6), ("use_cache", True)):
        value = _config_value(tc, name, True if name == "attn_output_gate" else None)
        if value != expected or (isinstance(expected, bool) and type(value) is not bool):
            bad.append(name)
    if _config_value(tc, "full_attention_interval", 4) != 4:
        bad.append("full_attention_interval")
    if _config_value(tc, "mlp_only_layers", ()):
        bad.append("mlp_only_layers")
    for owner in (config, tc):
        for name in ("quantization_config", "num_experts", "adaptive_mode"):
            if _config_value(owner, name):
                bad.append(name)
    if tc is not config and _config_value(tc, "quant_config") is not None:
        bad.append("text_config.quant_config")
    rope = _config_value(tc, "rope_parameters")
    expected_rope = dict(rope_type="default", rope_theta=10_000_000,
                         partial_rotary_factor=0.25, mrope_interleaved=True,
                         mrope_section=[11, 11, 10])
    if not isinstance(rope, Mapping):
        bad.append("rope_parameters")
    else:
        for name, expected in expected_rope.items():
            value = rope.get(name)
            if name == "mrope_section" and isinstance(value, tuple):
                value = list(value)
            if value != expected or (isinstance(expected, bool) and type(value) is not bool):
                bad.append("rope_parameters." + name)
    if _config_value(config, "tie_word_embeddings") is not False:
        bad.append("tie_word_embeddings")
    if tc is not config:
        if (_config_value(config, "model_type") != "qwen3_5"
                or _config_value(config, "image_token_id") != 248056):
            bad.append("conditional_generation")
        vc = _config_value(config, "vision_config")
        for name, expected in dict(depth=27, num_heads=16, hidden_size=1152,
                intermediate_size=4304, out_hidden_size=5120, in_channels=3,
                patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
                num_position_embeddings=2304).items():
            value = _config_value(vc, name)
            if type(value) is not int or value != expected:
                bad.append("vision_config." + name)
        if (_config_value(vc, "hidden_act") != "gelu_pytorch_tanh"
                or _config_value(vc, "deepstack_visual_indexes", ())
                or _config_value(vc, "quant_config")
                or _config_value(vc, "quantization_config")):
            bad.append("vision_config.semantics")
    if bad:
        raise ValueError("official Qwen3.5-27B requires exact dense semantics; mismatched fields: "
                         + ", ".join(sorted(set(bad))))
    quant = _config_value(config, "quant_config")
    if quant is None:
        if allow_fp16:
            return True
        raise ValueError("official Qwen3.5-27B FP16 requires an explicit runtime admission")
    if is_nvfp4_text_policy(quant):
        resolve_quant_specs(is_full_from_layer_types(expected_layers), quant)
        return True
    specs = set(resolve_quant_specs(is_full_from_layer_types(expected_layers), quant).values())
    if (len(specs) != 1 or next(iter(specs)) not in
            (QuantSpec(W8A16), QuantSpec(W4A16_PGRP, 32))):
        raise ValueError("official Qwen3.5-27B requires uniform W8 or W4 group32 text projections")
    if quant.get("storage", "int8") != "int8" or quant.get("scale_dtype", "float16") != "float16":
        raise ValueError("official Qwen3.5-27B requires int8 raw storage and FP16 scales")
    return True



def validate_named_27b_mixed_profile(config):
    """Bind each named official 27B mixed recipe to its exact role policy."""
    quant = _config_value(config, "quant_config")
    profile = quant.get("profile") if isinstance(quant, Mapping) else None
    tc = _config_value(config, "text_config", config)
    if profile not in NAMED_MIXED_PROFILES or _config_value(tc, "hidden_size") != 5120:
        return False
    source = dict(config) if isinstance(config, Mapping) else copy(config)
    if isinstance(source, dict):
        source["quant_config"] = None
    else:
        source.quant_config = None
    validate_modern_27b_profile(source, allow_fp16=True)
    if (quant.get("format_version") != 2 or "layer_overrides" in quant
            or quant.get("storage", "int8") != "int8"
            or quant.get("scale_dtype", "float16") != "float16"):
        raise ValueError("named official 27B mixed recipes require version 2 raw INT8/FP16 storage")
    is_full = is_full_from_layer_types(_config_value(tc, "layer_types"))
    specs = resolve_quant_specs(is_full, quant)
    for name, spec in specs.items():
        mlp = ".mlp." in name
        if profile == "w4_mlp_only":
            expected = QuantSpec(W4A16_PGRP, 32) if mlp else QuantSpec(FP16)
        elif profile == "w4_mlp_w8_attn_gdn":
            expected = QuantSpec(W4A16_PGRP, 32) if mlp else QuantSpec(W8A16)
        elif name.endswith(("linear_attn.in_proj_b", "linear_attn.in_proj_a")):
            expected = QuantSpec(FP16)
        elif profile == "w8_main_w8_gdn" or ".linear_attn." in name:
            expected = QuantSpec(W8A16)
        else:
            expected = QuantSpec(W4A16_PGRP, 32)
        if spec != expected:
            raise ValueError("named official 27B mixed recipe conflicts with its exact role policy")
    return True


def resolve_quant_specs(is_full, quant_config=None):
    """Resolve ``config.json`` quant policy to ``{relative_path: QuantSpec}``.

    Missing config means FP16. Format v1 keeps the historical uniform ``method``
    contract. Format v2 applies ``default`` and then canonical projection-role
    overrides; arbitrary paths, globs and per-layer overrides are intentionally
    unsupported.
    """
    relpaths = quant_relpaths(is_full)
    if quant_config is None:
        fp16 = QuantSpec(FP16)
        return {relpath: fp16 for relpath in relpaths}

    quant_config = normalize_quant_config(is_full, quant_config)
    quant_config = _mapping(quant_config, "quant_config")
    if quant_config.get("profile") == LEGACY_W8A16_PROFILE:
        canonical = legacy_w8a16_quant_config()
        fields = ("format_version", "architecture", "activation_dtype", "default",
                  "projection_overrides", "storage", "scale_dtype")
        if any(quant_config.get(key) != canonical[key] for key in fields):
            raise ValueError(
                f"Qwen3.5 profile={LEGACY_W8A16_PROFILE!r} requires the exact "
                "MLP/full-attention W8A16 and GDN FP16 policy")
    if (is_nvfp4_text_policy(quant_config)
            and dict(quant_config) != nvfp4_text_quant_config()):
        raise ValueError(
            f"Qwen3.5 profile={NVFP4_TEXT_PROFILE!r} requires the exact install-time "
            "NVFP4 policy with FP16 GDN b/a")
    architecture = quant_config.get("architecture", "qwen3_5")
    if architecture != "qwen3_5":
        raise ValueError(
            f"quant_config.architecture must be 'qwen3_5', got {architecture!r}"
        )
    activation_dtype = quant_config.get("activation_dtype", "float16")
    if activation_dtype != "float16":
        raise ValueError(
            "Qwen3.5 quant_config.activation_dtype must be 'float16', got "
            f"{activation_dtype!r}"
        )

    version = quant_config.get("format_version", 1)
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(
            "Qwen3.5 quant_config.format_version must be integer 1 or 2, got "
            f"{version!r}"
        )
    if version == 3:
        if quant_config.get("profile") not in AWQ_PROFILES or quant_config != _quanttrio_policy(is_full, quant_config.get("profile")):
            raise ValueError("Qwen3.5 v3 requires the exact QuantTrio AWQ per-module scope")
        return {name: _spec(quant_config["module_overrides"].get(name, quant_config["default"]), name)
                for name in relpaths}
    if "module_overrides" in quant_config:
        raise ValueError("Qwen3.5 module_overrides requires format_version=3")
    if version == 1:
        if "default" in quant_config or "projection_overrides" in quant_config:
            raise ValueError(
                "Qwen3.5 quant_config v1 cannot contain v2 "
                "default/projection_overrides fields"
            )
        default = _spec(
            {key: quant_config[key] for key in ("method", "group_size")
             if key in quant_config},
            "quant_config",
        )
        if default.method == W4A16_AWQ:
            raise ValueError("Qwen3.5 AWQ requires the exact version 3 profile")
        if default.method == NVFP4:
            raise ValueError(
                f"Qwen3.5 NVFP4 requires the exact {NVFP4_TEXT_PROFILE!r} profile")
        return {relpath: default for relpath in relpaths}
    if version != 2:
        raise ValueError(
            f"unsupported Qwen3.5 quant_config.format_version {version!r}; "
            "expected 1, 2 or 3"
        )

    if "method" in quant_config or "group_size" in quant_config:
        raise ValueError(
            "Qwen3.5 quant_config v2 uses default/projection_overrides; "
            "top-level method/group_size is ambiguous"
        )
    default = _spec(quant_config.get("default"), "quant_config.default")
    overrides = _mapping(
        quant_config.get("projection_overrides", {}),
        "quant_config.projection_overrides",
    )
    unknown_roles = sorted(set(overrides) - _PROJECTION_ROLES)
    if unknown_roles:
        raise ValueError(
            f"Qwen3.5 quant_config has unknown projection role(s): {unknown_roles}"
        )
    active_roles = {projection_role(relpath) for relpath in relpaths}
    inactive_roles = sorted(set(overrides) - active_roles)
    if inactive_roles:
        raise ValueError(
            "Qwen3.5 quant_config overrides projection role(s) absent from the "
            f"configured layer_types: {inactive_roles}"
        )
    override_specs = {
        role: _spec(value, f"quant_config.projection_overrides[{role!r}]")
        for role, value in overrides.items()
    }
    if any(spec.method == W4A16_AWQ for spec in (default, *override_specs.values())):
        raise ValueError("Qwen3.5 AWQ requires the exact version 3 profile")
    if (not is_nvfp4_text_policy(quant_config)
            and any(spec.method == NVFP4 for spec in (default, *override_specs.values()))):
        raise ValueError(
            f"Qwen3.5 NVFP4 requires the exact {NVFP4_TEXT_PROFILE!r} profile")
    return {
        relpath: override_specs.get(projection_role(relpath), default)
        for relpath in relpaths
    }


def validate_declared_dtype(layer_types, quant_config, declared_dtype):
    """Validate a perf dtype label against the checkpoint's resolved policy.

    Returns reporting metadata derived from the executable policy rather than
    trusting optional converter counters in ``config.json``.
    """
    dtype = str(declared_dtype).lower()
    if dtype not in DECLARED_DTYPES:
        raise ValueError(
            f"Qwen3.5 declared dtype must be one of {DECLARED_DTYPES}, got "
            f"{declared_dtype!r}"
        )
    if dtype == FP16 and quant_config is not None:
        raise ValueError(
            "Qwen3.5 dtype='fp16' requires a dense checkpoint without "
            "quant_config"
        )
    if dtype != FP16 and quant_config is None:
        raise ValueError(
            f"Qwen3.5 dtype={dtype!r} requires config.json.quant_config"
        )

    is_full = is_full_from_layer_types(layer_types)
    quant_config = normalize_quant_config(is_full, quant_config)
    specs = resolve_quant_specs(is_full, quant_config)
    counts = {method: 0 for method in SUPPORTED_QUANT_METHODS}
    for spec in specs.values():
        counts[spec.method] += 1

    if dtype == W8A16 and (
        counts[W8A16] == 0 or counts[W4A16_PGRP] != 0 or counts[W4A16_AWQ] != 0
    ):
        raise ValueError(
            "Qwen3.5 dtype='w8a16' requires at least one W8 projection and no "
            f"W4 projections; resolved counts={counts}"
        )
    if dtype == "w4a16" and counts[W4A16_PGRP] + counts[W4A16_AWQ] == 0:
        raise ValueError(
            "Qwen3.5 dtype='w4a16' requires at least one W4 projection; "
            f"resolved counts={counts}"
        )

    if quant_config is None:
        profile = "dense_fp16"
        format_version = 0
    else:
        format_version = int(quant_config.get("format_version", 1))
        profile = quant_config.get("profile")
        if profile is None:
            profile = (
                quant_config.get("method", "custom_v2")
                if format_version == 1 else "custom_v2"
            )
        if not isinstance(profile, str) or not profile:
            raise ValueError(
                "Qwen3.5 quant_config.profile must be a non-empty string when "
                f"present, got {profile!r}"
            )
    # Keep the existing reporting contract for FP16/W8/signed-W4 consumers.
    for method in (W4A16_AWQ, NVFP4):
        if counts[method] == 0:
            del counts[method]
    return {
        "declared_dtype": dtype,
        "profile": profile,
        "format_version": format_version,
        "projection_counts": counts,
    }


def quant_weight_keys(is_full, prefix=TEXT_PREFIX):
    """Return the checkpoint ``*.weight`` keys covered by W8A16."""
    return [f"{prefix}{rel}.weight" for rel in quant_relpaths(is_full)]


def scale_key(weight_key):
    """Return the ``weight_scale`` key paired with a quantized ``weight`` key."""
    if not weight_key.endswith(".weight"):
        raise ValueError(f"expected a '.weight' key, got {weight_key!r}")
    return weight_key[: -len(".weight")] + ".weight_scale"
