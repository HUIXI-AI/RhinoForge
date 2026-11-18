"""Host-only tests for the Qwen3.5 per-projection quantization contract."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


_ADAPTER = Path(__file__).resolve().parents[2] / "adapters" / "qwen3_5"
sys.path.insert(0, str(_ADAPTER))
import quant_scope as qs  # noqa: E402


def _layer_mix(n_linear=1, n_full=1):
    return qs.is_full_from_layer_types(
        [qs.LINEAR_ATTENTION] * n_linear + [qs.FULL_ATTENTION] * n_full
    )


def _mixed_config():
    return {
        "format_version": 2,
        "architecture": "qwen3_5",
        "activation_dtype": "float16",
        "profile": "w4_main_w8_gdn",
        "default": {"method": qs.W4A16_PGRP, "group_size": 32},
        "projection_overrides": {
            "linear_attn.in_proj_qkv": {"method": qs.W8A16},
            "linear_attn.in_proj_z": {"method": qs.W8A16},
            "linear_attn.out_proj": {"method": qs.W8A16},
            "linear_attn.in_proj_b": {"method": qs.FP16},
            "linear_attn.in_proj_a": {"method": qs.FP16},
        },
    }


def test_missing_config_resolves_every_projection_to_fp16():
    specs = qs.resolve_quant_specs(_layer_mix(), None)
    assert len(specs) == 15
    assert set(specs.values()) == {qs.QuantSpec(qs.FP16)}


def test_v1_uniform_w8_and_w4_remain_compatible():
    is_full = _layer_mix()

    w8 = qs.resolve_quant_specs(is_full, {"method": qs.W8A16})
    assert set(w8.values()) == {qs.QuantSpec(qs.W8A16)}

    w4 = qs.resolve_quant_specs(
        is_full,
        {"format_version": 1, "method": qs.W4A16_PGRP, "group_size": 32},
    )
    assert set(w4.values()) == {qs.QuantSpec(qs.W4A16_PGRP, 32)}


def test_4b_mixed_profile_resolves_128_w4_72_w8_48_fp16():
    # Qwen3.5-4B config.json repeats 3 GDN layers followed by 1 full-attention
    # layer eight times.
    is_full = qs.is_full_from_layer_types(
        [qs.LINEAR_ATTENTION] * 3 + [qs.FULL_ATTENTION]
    ) * 8
    specs = qs.resolve_quant_specs(is_full, _mixed_config())

    counts = {method: 0 for method in qs.SUPPORTED_QUANT_METHODS}
    for spec in specs.values():
        counts[spec.method] += 1

    assert len(specs) == 248
    assert counts == {qs.FP16: 48, qs.W8A16: 72, qs.W4A16_PGRP: 128, qs.W4A16_AWQ: 0, qs.NVFP4: 0}
    assert specs["layers.0.linear_attn.in_proj_qkv"] == qs.QuantSpec(qs.W8A16)
    assert specs["layers.0.linear_attn.in_proj_b"] == qs.QuantSpec(qs.FP16)
    assert specs["layers.3.self_attn.q_proj"] == qs.QuantSpec(
        qs.W4A16_PGRP, 32
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda config: config["projection_overrides"].update(
                {"linear_attn.not_a_projection": {"method": qs.W8A16}}
            ),
            "unknown projection role",
        ),
        (
            lambda config: config["default"].update({"group_size": 16}),
            "requires group_size",
        ),
        (
            lambda config: config["default"].update({"method": "fp8_e4m3"}),
            "unsupported Qwen3.5 quant method",
        ),
        (
            lambda config: config.update({"format_version": True}),
            "format_version must be integer",
        ),
        (
            lambda config: config.update({"format_version": 1, "method": qs.W8A16}),
            "v1 cannot contain v2",
        ),
    ],
)
def test_invalid_v2_policy_fails_closed(mutate, match):
    config = _mixed_config()
    mutate(config)
    with pytest.raises(ValueError, match=match):
        qs.resolve_quant_specs(_layer_mix(), config)


def test_override_for_absent_layer_type_fails_closed():
    config = _mixed_config()
    with pytest.raises(ValueError, match="absent from the configured layer_types"):
        qs.resolve_quant_specs(
            qs.is_full_from_layer_types([qs.FULL_ATTENTION] * 8), config
        )


def test_declared_dtype_returns_profile_and_counts():
    layer_types = [qs.LINEAR_ATTENTION, qs.FULL_ATTENTION]

    dense = qs.validate_declared_dtype(layer_types, None, "fp16")
    assert dense == {
        "declared_dtype": "fp16",
        "profile": "dense_fp16",
        "format_version": 0,
        "projection_counts": {qs.FP16: 15, qs.W8A16: 0, qs.W4A16_PGRP: 0},
    }

    mixed = qs.validate_declared_dtype(
        layer_types, _mixed_config(), "w4a16")
    assert mixed["profile"] == "w4_main_w8_gdn"
    assert mixed["projection_counts"] == {
        qs.FP16: 2,
        qs.W8A16: 3,
        qs.W4A16_PGRP: 10,
    }


@pytest.mark.parametrize(
    "quant_config,declared_dtype,match",
    [
        (None, "w8a16", "requires config.json.quant_config"),
        ({"method": qs.W8A16}, "fp16", "dense checkpoint"),
        ({"method": qs.W8A16}, "w4a16", "at least one W4"),
        (_mixed_config(), "w8a16", "no W4 projections"),
    ],
)
def test_declared_dtype_rejects_mislabeled_checkpoint(
    quant_config, declared_dtype, match
):
    with pytest.raises(ValueError, match=match):
        qs.validate_declared_dtype(
            [qs.LINEAR_ATTENTION, qs.FULL_ATTENTION],
            quant_config,
            declared_dtype,
        )
