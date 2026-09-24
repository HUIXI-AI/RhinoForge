"""Qwen3.5 quantization scope and per-projection precision contract.

Shared by the offline converter (``rpu_backend.quant.convert_qwen3_5``), the
checkpoint loader (``adapters/qwen3_5/loader.py``) and the RPU install path
(``text.py::_iter_text_projections``). Keeping one list avoids the failure mode
where a checkpoint is converted over one set of modules and loaded over another.

Deliberately stdlib-only: the converter computes the scope from a raw
``config.json`` and must not need torch or the compiled backend.
"""
from __future__ import annotations

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
SUPPORTED_QUANT_METHODS = (FP16, W8A16, W4A16_PGRP)
PGRP_SUPPORTED_GROUP_SIZES = (32, 64, 128)
DECLARED_DTYPES = (FP16, W8A16, "w4a16")

# The first Qwen3.8-27B W8A16 bundle predates the v1/v2 declaration used by
# ``convert_qwen3_5``.  It quantizes the MLP and full-attention projections,
# leaves all GDN projections in FP16, and stores MTP scales even though the RPU
# text path intentionally does not instantiate MTP.  Keep this policy explicit
# instead of treating it as the v1 uniform-W8 format (which would silently ask
# for scales on every GDN projection).
LEGACY_W8A16_PROFILE = "w8_main_fp16_gdn_legacy"

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
        if self.method == W4A16_PGRP:
            if (not isinstance(self.group_size, int)
                    or isinstance(self.group_size, bool)
                    or self.group_size not in PGRP_SUPPORTED_GROUP_SIZES):
                raise ValueError(
                    f"method={W4A16_PGRP} requires group_size in "
                    f"{PGRP_SUPPORTED_GROUP_SIZES}, got {self.group_size!r}"
                )
        elif self.group_size is not None:
            raise ValueError(
                f"method={self.method} does not accept group_size, got "
                f"{self.group_size!r}"
            )


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
        return {relpath: default for relpath in relpaths}
    if version != 2:
        raise ValueError(
            f"unsupported Qwen3.5 quant_config.format_version {version!r}; "
            "expected 1 or 2"
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
        counts[W8A16] == 0 or counts[W4A16_PGRP] != 0
    ):
        raise ValueError(
            "Qwen3.5 dtype='w8a16' requires at least one W8 projection and no "
            f"W4 projections; resolved counts={counts}"
        )
    if dtype == "w4a16" and counts[W4A16_PGRP] == 0:
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
