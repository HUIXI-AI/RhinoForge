"""Offline W8A16/W4A16 quantization for Qwen3.5 HF checkpoints.

Quantizes the text-backbone projections listed by
``adapters.qwen3_5.quant_scope`` to int8 with a per-output-channel fp16 scale;
every other floating tensor is stored as fp16. Vision, MTP, embeddings, lm_head,
norms and the GDN conv1d are left alone.

Deliberately separate from ``convert_qwen3.py``: Qwen3.5 quantizes the GDN
``linear_attn.in_proj_*`` projections, which the Qwen3 suffix list does not
cover, and its scope is driven by ``text_config.layer_types``.

The fp16 cast before quantization is load-bearing: the retired in-place
prototype quantized after ``model.half()``, so quantizing the bf16 source
directly would produce different int8 values and break bit-identity with the
recorded pre-refactor parity hash.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

try:
    from ._common import quantize_linear_per_channel
    from .int4_pgrp_pack import quantize_int4_group_wise
except ImportError:  # standalone `python .../quant/convert_qwen3_5.py`
    from _common import quantize_linear_per_channel
    from int4_pgrp_pack import quantize_int4_group_wise

try:
    from rpu_backend.adapters.qwen3_5.quant_scope import (
        FP16, FULL_ATTENTION, LINEAR_ATTENTION, PGRP_SUPPORTED_GROUP_SIZES,
        TEXT_PREFIX, W4A16_PGRP, W8A16, LEGACY_W8A16_PROFILE,
        is_full_from_layer_types, legacy_w8a16_quant_config, projection_role,
        resolve_quant_specs, scale_key, validate_legacy_text_profile,
    )
except ImportError:  # standalone: bypass any unrelated editable package install
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[1] / "adapters" / "qwen3_5"),
    )
    from quant_scope import (
        FP16, FULL_ATTENTION, LINEAR_ATTENTION, PGRP_SUPPORTED_GROUP_SIZES,
        TEXT_PREFIX, W4A16_PGRP, W8A16, LEGACY_W8A16_PROFILE,
        is_full_from_layer_types, legacy_w8a16_quant_config, projection_role,
        resolve_quant_specs, scale_key, validate_legacy_text_profile,
    )


ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
FORMAT_VERSION = 1
METHODS = ("w8a16", "w4a16_pgrp")
PROFILES = ("w4_main_w8_gdn", "w8_main_w8_gdn", LEGACY_W8A16_PROFILE)
DEFAULT_GROUP_SIZE = 32
SKIP_MODULES = ["lm_head", "embed_tokens", "visual", "mtp", "conv1d"]


def _as_runtime_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """Cast the way the runtime FP16 load does: bf16-pinned, then fp16.

    ``text_config.dtype`` is bfloat16, so ``from_pretrained`` materializes every
    floating tensor as bf16 before the adapter's ``.half()``. That is a no-op for
    the bf16 bulk of the checkpoint, but the 48 GDN ``A_log`` / ``norm.weight``
    tensors are stored fp32 — casting them straight to fp16 keeps precision the
    runtime never had and shifts the parity hash by one bf16 ulp. Reproduce the
    runtime route so an offline checkpoint and a live FP16 load agree bit-for-bit
    outside the quantized projections.
    """
    return tensor.to(torch.bfloat16).to(torch.float16)


def _layer_types(config: dict) -> list[str]:
    text_config = config.get("text_config", config)
    layer_types = list(text_config.get("layer_types") or [])
    n_layers = int(text_config["num_hidden_layers"])
    if len(layer_types) != n_layers:
        raise ValueError(
            f"layer_types length {len(layer_types)} != num_hidden_layers {n_layers}"
        )
    bad = sorted(set(layer_types) - {FULL_ATTENTION, LINEAR_ATTENTION})
    if bad:
        raise ValueError(f"unsupported layer_types {bad}")
    return layer_types


def _checkpoint_layout(src: Path) -> tuple[bool, list[str], dict[str, str], dict]:
    index_path = src / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as f:
            index = json.load(f)
        weight_map = dict(index["weight_map"])
        return True, sorted(set(weight_map.values())), weight_map, index

    if (src / "model.safetensors").is_file():
        return False, ["model.safetensors"], {}, {}

    raise FileNotFoundError(
        f"no model.safetensors or model.safetensors.index.json in {src}")


def _copy_sidecar_files(src: Path, dst: Path, shards: set[str]) -> None:
    skip = set(shards)
    skip.update({"model.safetensors", "model.safetensors.index.json", "config.json"})
    for path in sorted(src.iterdir()):
        if path.name in skip or not path.is_file():
            continue
        shutil.copy2(path, dst / path.name)


def _quant_config(is_full, method: str | None, group_size: int,
                  profile: str | None = None) -> dict:
    if profile is None:
        pgrp = method == W4A16_PGRP
        config = {
            "method": method,
            "format_version": FORMAT_VERSION,
            "architecture": "qwen3_5",
            "mode": "per_group_symmetric" if pgrp else "per_channel_symmetric",
            "qaxis": 0,
            # int4 values live in an int8 container on disk; the adapter
            # nibble-packs them at install time, when the core count is known.
            "storage": "int8",
            "value_bits": 4 if pgrp else 8,
            "scale_dtype": "float16",
        }
        if pgrp:
            config["group_size"] = group_size
    elif profile == LEGACY_W8A16_PROFILE:
        # Reproducible declaration for the already shipped Qwen3.8-27B bundle:
        # MLP + full-attention W8A16, GDN FP16.  The loader also accepts the
        # original pre-v1 metadata and routes it through the text-only path.
        config = legacy_w8a16_quant_config()
    elif profile in PROFILES:
        if group_size != DEFAULT_GROUP_SIZE:
            raise ValueError(
                f"profile={profile} requires the default --group-size="
                f"{DEFAULT_GROUP_SIZE}, got {group_size}"
            )
        default = ({"method": W4A16_PGRP, "group_size": group_size}
                   if profile == "w4_main_w8_gdn"
                   else {"method": W8A16})
        overrides = {
            "linear_attn.in_proj_b": {"method": FP16},
            "linear_attn.in_proj_a": {"method": FP16},
        }
        if profile == "w4_main_w8_gdn":
            overrides.update({
                "linear_attn.in_proj_qkv": {"method": W8A16},
                "linear_attn.in_proj_z": {"method": W8A16},
                "linear_attn.out_proj": {"method": W8A16},
            })
        config = {
            "format_version": 2,
            "architecture": "qwen3_5",
            "activation_dtype": "float16",
            "profile": profile,
            "default": default,
            "projection_overrides": overrides,
            "storage": "int8",
            "scale_dtype": "float16",
        }
    else:
        raise ValueError(f"unsupported Qwen3.5 quant profile {profile!r}")

    specs = resolve_quant_specs(is_full, config)
    counts = {FP16: 0, W8A16: 0, W4A16_PGRP: 0}
    suffixes = set()
    for relpath, spec in specs.items():
        counts[spec.method] += 1
        if spec.method != FP16:
            suffixes.add(f"{projection_role(relpath)}.weight")
    config.update({
        "quantized_prefix": f"{TEXT_PREFIX}layers.",
        "quantized_weight_suffixes": sorted(suffixes),
        "n_quantized": counts[W8A16] + counts[W4A16_PGRP],
        "projection_counts": counts,
        "skip_modules": list(SKIP_MODULES),
        "producer": "rpu_backend.quant.convert_qwen3_5",
    })
    return config


def convert_checkpoint(src: str | Path, dst: str | Path, *,
                       method: str | None = None,
                       profile: str | None = None,
                       group_size: int = DEFAULT_GROUP_SIZE) -> dict:
    src = Path(src).expanduser().resolve()
    dst = Path(dst).expanduser().resolve()
    if method is not None and profile is not None:
        raise ValueError("choose exactly one of method or profile")
    if method is None and profile is None:
        method = W8A16
    if method is not None and method not in METHODS:
        raise ValueError(f"--method must be one of {list(METHODS)}, got {method!r}")
    if profile is not None and profile not in PROFILES:
        raise ValueError(f"--profile must be one of {list(PROFILES)}, got {profile!r}")
    if (method == W4A16_PGRP
            and group_size not in PGRP_SUPPORTED_GROUP_SIZES):
        raise ValueError(
            "--group-size must be one of "
            f"{PGRP_SUPPORTED_GROUP_SIZES} for w4a16_pgrp, got {group_size}")
    if not src.is_dir():
        raise FileNotFoundError(f"--src is not a directory: {src}")
    if dst.exists():
        raise FileExistsError(f"--dst already exists, refusing to overwrite: {dst}")

    with (src / "config.json").open() as f:
        config = json.load(f)
    architectures = config.get("architectures") or []
    if architectures and architectures[0] != ARCHITECTURE:
        raise ValueError(
            f"expected architectures[0]=={ARCHITECTURE!r}, got {architectures[0]!r}")
    if "quant_config" in config:
        raise ValueError(f"{src} already declares a quant_config; nothing to do")

    is_full = is_full_from_layer_types(_layer_types(config))
    quant_config = _quant_config(is_full, method, group_size, profile)
    if profile == LEGACY_W8A16_PROFILE:
        validate_legacy_text_profile(dict(config, quant_config=quant_config))
    specs = resolve_quant_specs(is_full, quant_config)
    expected = {
        f"{TEXT_PREFIX}{relpath}.weight": spec
        for relpath, spec in specs.items()
    }

    is_sharded, shards, weight_map, index = _checkpoint_layout(src)
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(f"temporary output already exists: {tmp}")

    stats = {
        "n_quantized": 0,
        "n_copied": 0,
        "bytes_in": 0,
        "bytes_out": 0,
        "n_shards": len(shards),
        "projection_counts": {FP16: 0, W8A16: 0, W4A16_PGRP: 0},
    }
    new_weight_map = dict(weight_map) if is_sharded else {}
    seen_projections: set[str] = set()

    try:
        tmp.mkdir(parents=True)
        for shard_idx, shard in enumerate(shards, start=1):
            print(f"[convert_qwen3_5] shard {shard_idx}/{len(shards)}: {shard}",
                  flush=True)
            tensors = load_file(src / shard)
            out: dict[str, torch.Tensor] = {}

            for name in sorted(tensors):
                tensor = tensors[name]
                stats["bytes_in"] += tensor.numel() * tensor.element_size()

                spec = expected.get(name)
                if spec is not None:
                    if not tensor.is_floating_point():
                        raise TypeError(
                            f"cannot quantize non-floating tensor {name!r} "
                            f"with dtype {tensor.dtype} (already quantized?)")
                    seen_projections.add(name)
                    stats["projection_counts"][spec.method] += 1
                    if spec.method == FP16:
                        converted = _as_runtime_fp16(tensor)
                        out[name] = converted
                        stats["n_copied"] += 1
                        stats["bytes_out"] += (
                            converted.numel() * converted.element_size())
                        continue

                    # fp16 FIRST — see the module docstring.
                    src_fp16 = _as_runtime_fp16(tensor)
                    if spec.method == W4A16_PGRP:
                        w_int8, scale, _ = quantize_int4_group_wise(
                            src_fp16, group_size=spec.group_size)
                    else:
                        w_int8, scale = quantize_linear_per_channel(src_fp16)
                    out[name] = w_int8
                    out[scale_key(name)] = scale
                    stats["n_quantized"] += 1
                    stats["bytes_out"] += (
                        w_int8.numel() * w_int8.element_size()
                        + scale.numel() * scale.element_size())
                    if is_sharded:
                        new_weight_map[scale_key(name)] = shard
                    continue

                converted = (_as_runtime_fp16(tensor)
                             if tensor.is_floating_point() else tensor)
                out[name] = converted
                stats["n_copied"] += 1
                stats["bytes_out"] += converted.numel() * converted.element_size()

            save_file(out, tmp / shard, metadata={"format": "pt"})
            del tensors
            del out

        missing = sorted(set(expected) - seen_projections)
        if missing:
            raise KeyError(
                f"checkpoint is missing {len(missing)} projection weight(s) the "
                f"Qwen3.5 quantization scope requires: {missing[:8]}"
                f"{' ...' if len(missing) > 8 else ''}")

        if is_sharded:
            new_index = dict(index)
            metadata = dict(new_index.get("metadata", {}))
            metadata["total_size"] = stats["bytes_out"]
            new_index["metadata"] = metadata
            new_index["weight_map"] = dict(sorted(new_weight_map.items()))
            with (tmp / "model.safetensors.index.json").open("w") as f:
                json.dump(new_index, f, indent=2)

        config["torch_dtype"] = "float16"
        config["dtype"] = "float16"
        config["quant_config"] = quant_config
        with (tmp / "config.json").open("w") as f:
            json.dump(config, f, indent=2)

        _copy_sidecar_files(src, tmp, set(shards))
        tmp.rename(dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="source HF checkpoint directory")
    parser.add_argument("--dst", required=True, help="output quantized checkpoint directory")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--method", choices=METHODS,
                           help="w8a16 = per-output-channel int8; "
                                "w4a16_pgrp = group-wise int4 along K "
                                "(default: w8a16)")
    precision.add_argument("--profile", choices=PROFILES,
                           help="configuration-driven mixed-precision profile")
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE,
                        help="K-group size for w4a16_pgrp (default 32)")
    args = parser.parse_args(argv)

    try:
        stats = convert_checkpoint(args.src, args.dst, method=args.method,
                                   profile=args.profile,
                                   group_size=args.group_size)
    except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"[convert_qwen3_5] error: {exc}", file=sys.stderr)
        return 1

    print(f"[convert_qwen3_5] {Path(args.src).resolve()}")
    print(f"              -> {Path(args.dst).resolve()}")
    print(f"  shards    : {stats['n_shards']}")
    selected = (f"profile={args.profile}" if args.profile
                else f"method={args.method or W8A16}")
    uses_w4 = (args.profile == "w4_main_w8_gdn"
               or args.method == W4A16_PGRP)
    print(f"  precision : {selected}"
          + (f" (group_size={args.group_size})" if uses_w4 else ""))
    print(f"  quantized : {stats['n_quantized']} weights")
    print(f"  projections: {stats['projection_counts']}")
    print(f"  copied    : {stats['n_copied']} tensors")
    print(f"  size      : {stats['bytes_in'] / 1e9:.2f} GB"
          f" -> {stats['bytes_out'] / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
