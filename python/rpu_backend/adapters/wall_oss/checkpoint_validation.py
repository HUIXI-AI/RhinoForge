"""Checkpoint-pair validation for experimental Wall-OSS precision modes."""

from __future__ import annotations

import json
from pathlib import Path

from rpu_backend.quant.convert_wall_oss import WALL_OSS_QUANT_SUFFIXES


_QUANTIZATION_CONFIG_KEYS = frozenset(
    {
        "quant_config",
        # The W8 converter records the runtime storage dtype while the source
        # checkpoint may retain its training dtype. These are representation
        # metadata, not architecture/model-identity fields.
        "dtype",
        "torch_dtype",
    }
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _quant_metadata(checkpoint: Path, config: dict) -> dict | None:
    sidecar_path = checkpoint / "rpu_quant_config.json"
    sidecar = _load_json(sidecar_path) if sidecar_path.is_file() else None
    embedded = config.get("quant_config")
    if embedded is not None and not isinstance(embedded, dict):
        raise ValueError(
            f"expected quant_config to be an object in {checkpoint / 'config.json'}"
        )
    if sidecar is not None and embedded is not None:
        sidecar_method = str(sidecar.get("method", "")).lower()
        embedded_method = str(embedded.get("method", "")).lower()
        if sidecar_method != embedded_method:
            raise ValueError(
                f"conflicting quantization metadata in {checkpoint}: "
                f"sidecar={sidecar_method!r}, config={embedded_method!r}"
            )
    return sidecar if sidecar is not None else embedded


def _validate_quantized_checkpoint_pair(
    quant_ckpt_dir: str | Path,
    fp16_ckpt_dir: str | Path,
    *,
    expected_method: str,
    label: str,
) -> tuple[dict, dict, dict]:
    """Reject a mis-typed or unrelated quantized/FP16 checkpoint pair."""
    quant_path = Path(quant_ckpt_dir)
    fp16_path = Path(fp16_ckpt_dir)
    quant_config = _load_json(quant_path / "config.json")
    fp16_config = _load_json(fp16_path / "config.json")

    quant_metadata = _quant_metadata(quant_path, quant_config)
    quant_method = (
        str(quant_metadata.get("method", "")).lower()
        if quant_metadata is not None
        else ""
    )
    if quant_method != expected_method:
        raise ValueError(
            f"quantized ckpt_dir must be a {label} checkpoint with "
            f"method={expected_method!r}; got "
            f"{quant_method or 'unquantized'} at {quant_path}"
        )

    fp16_quant = _quant_metadata(fp16_path, fp16_config)
    if fp16_quant is not None:
        raise ValueError(
            "fp16_ckpt_dir must be an unquantized source "
            f"checkpoint; found method={fp16_quant.get('method')!r} at {fp16_path}"
        )

    source = quant_metadata.get("source")
    if not source:
        raise ValueError(
            f"{label} checkpoint {quant_path} has no source provenance metadata"
        )
    source_path = Path(str(source))
    if not source_path.is_absolute():
        source_path = quant_path.parent / source_path
    try:
        source_resolved = source_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"{label} source provenance cannot be resolved exactly: "
            f"source={source!r}, resolved_candidate={str(source_path)!r}"
        ) from exc
    fp16_resolved = fp16_path.resolve(strict=True)
    if source_resolved != fp16_resolved:
        raise ValueError(
            f"{label} and FP16 checkpoints are not a declared source pair: "
            f"quantized source={str(source_resolved)!r}, "
            f"fp16_ckpt_dir={str(fp16_resolved)!r}"
        )

    config_keys = (
        set(quant_config) | set(fp16_config)
    ) - _QUANTIZATION_CONFIG_KEYS
    mismatched = [
        key
        for key in sorted(config_keys)
        if quant_config.get(key) != fp16_config.get(key)
    ]
    if mismatched:
        raise ValueError(
            f"{label} and FP16 checkpoints are not architecture-compatible; "
            f"mismatched config fields: {mismatched}"
        )
    return quant_config, fp16_config, quant_metadata


def validate_w8_nvfp4_checkpoint_pair(
    w8_ckpt_dir: str | Path,
    fp16_ckpt_dir: str | Path,
) -> tuple[dict, dict]:
    """Reject a mis-typed or unrelated W8/FP16 pair before weight conversion."""
    w8_config, fp16_config, _ = _validate_quantized_checkpoint_pair(
        w8_ckpt_dir,
        fp16_ckpt_dir,
        expected_method="w8a16",
        label="W8A16",
    )
    return w8_config, fp16_config


def validate_rtc_w4_checkpoint_pair(
    w4_ckpt_dir: str | Path,
    fp16_ckpt_dir: str | Path,
) -> tuple[dict, dict]:
    """Validate the exact group-32 W4/INT8/FP16 composition used by RTC."""
    w4_config, fp16_config, quant = _validate_quantized_checkpoint_pair(
        w4_ckpt_dir,
        fp16_ckpt_dir,
        expected_method="w4a16",
        label="RTC W4A16",
    )
    int8_suffixes = quant.get("int8_suffixes")
    required_int8 = "moe.experts.0.down_proj.weight"
    required_int4 = [
        suffix for suffix in WALL_OSS_QUANT_SUFFIXES
        if suffix != required_int8
    ]
    if (
        quant.get("mode") != "group_wise_symmetric"
        or quant.get("group_size") != 32
        or int8_suffixes != [required_int8]
        or quant.get("int4_suffixes") != required_int4
    ):
        raise ValueError(
            "Wall-OSS RTC W4A16 requires the exact seven group-wise INT4 "
            "projection suffixes with group_size=32 and expert-0 down_proj "
            "retained as INT8"
        )
    return w4_config, fp16_config
