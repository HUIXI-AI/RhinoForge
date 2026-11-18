"""Offline FP16 export and W8A16/W4A16 quantization for Qwen3.5 checkpoints.

Quantizes the text-backbone projections listed by
``adapters.qwen3_5.quant_scope`` with per-output-channel W8 or group-wise W4
scales. Other floating tensors, including Vision, MTP, embeddings, lm_head,
norms and the GDN conv1d, are stored as fp16 without integer quantization.

Deliberately separate from ``convert_qwen3.py``: Qwen3.5 quantizes the GDN
``linear_attn.in_proj_*`` projections, which the Qwen3 suffix list does not
cover, and its scope is driven by ``text_config.layer_types``.

Floating tensors are cast from their actual stored dtype directly to fp16,
including the inputs to the quantizers. Do not round FP32/FP16 sources through
BF16 to reproduce a historical loader: that loses precision before either
quantization or inference. Existing converted checkpoints must be regenerated
from their source to benefit from this correction.
"""
from __future__ import annotations

import argparse
import hashlib
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
    from .precision_recipes import RECORD, produced, write_record
except ImportError:  # standalone `python .../quant/convert_qwen3_5.py`
    from _common import quantize_linear_per_channel
    from int4_pgrp_pack import quantize_int4_group_wise
    from precision_recipes import RECORD, produced, write_record

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
METHODS = ("fp16", "w8a16", "w4a16_pgrp")
PROFILES = ("w4_main_w8_gdn", "w8_main_w8_gdn", "w4_mlp_only",
            "w4_mlp_w8_attn_gdn", LEGACY_W8A16_PROFILE)
DEFAULT_GROUP_SIZE = 32
FAMILY = "qwen3_5"
# Registered precision recipes this converter produces, as the arguments that produce them.
RECIPES = {
    "fp16": {"method": FP16},
    "w8a16_text": {"method": W8A16},
    "w4a16_text_rtn_g32": {"method": W4A16_PGRP, "group_size": 32},
    "w4a16_mlp_rtn_g32": {"profile": "w4_mlp_only"},
    "w4a16_mlp_rtn_g32_attn_gdn_w8a16": {"profile": "w4_mlp_w8_attn_gdn"},
}
SKIP_MODULES = ["lm_head", "embed_tokens", "visual", "mtp", "conv1d"]


def _as_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """Use the tensor's stored precision, without a lossy BF16 intermediate."""
    return tensor.to(torch.float16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_fp16(tensor: torch.Tensor, name: str) -> torch.Tensor:
    converted = _as_fp16(tensor)
    flat = converted.view(-1)
    for start in range(0, flat.numel(), 1 << 20):
        if not bool(torch.isfinite(flat[start:start + (1 << 20)]).all()):
            raise ValueError(f"nonfinite value or FP16 overflow in {name!r}")
    return converted


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


def _copy_sidecar_files(src: Path, dst: Path, shards: set[str]) -> list[str]:
    skip = set(shards)
    skip.update({"model.safetensors", "model.safetensors.index.json", "config.json",
                 "SHA256SUMS", RECORD})
    copied = []
    for path in sorted(src.iterdir()):
        if (path.name in skip or not path.is_file()
                or path.name.endswith((".bin", ".pt", ".pth", ".safetensors",
                                       ".bin.index.json", ".safetensors.index.json"))):
            continue
        shutil.copy2(path, dst / path.name)
        copied.append(path.name)
    return copied


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
        if profile in ("w4_mlp_only", "w4_mlp_w8_attn_gdn"):
            # Existing v2 roles apply to every layer, including layer zero.
            default = {"method": FP16 if profile == "w4_mlp_only" else W8A16}
            overrides = {
                f"mlp.{role}": {"method": W4A16_PGRP, "group_size": group_size}
                for role in ("gate_proj", "up_proj", "down_proj")
            }
        else:
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


def produced_recipe(method: str | None = None, profile: str | None = None,
                    group_size: int = DEFAULT_GROUP_SIZE) -> str | None:
    """The registered recipe these arguments produce; None for the other profiles and W4 groups."""
    if method is None and profile is None:
        method = W8A16
    arguments = {"profile": profile} if profile is not None else {"method": method}
    if method == W4A16_PGRP:
        arguments["group_size"] = group_size
    return next((name for name, row in RECIPES.items() if row == arguments), None)


def convert_checkpoint(src: str | Path, dst: str | Path, *,
                       method: str | None = None,
                       profile: str | None = None,
                       group_size: int = DEFAULT_GROUP_SIZE,
                       recipe: str | None = None) -> dict:
    """RECIPE, when given, must be the registered recipe the other arguments produce."""
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
    recipe = produced(FAMILY, produced_recipe(method, profile, group_size), recipe)
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
    if any(owner.get("quantization_config") is not None
           or (owner is not config and owner.get("quant_config") is not None)
           for owner in (config, config.get("text_config", {}),
                         config.get("vision_config", {}))):
        raise ValueError("conversion requires original weights without quantization metadata")

    is_full = is_full_from_layer_types(_layer_types(config))
    fp16_export = method == FP16
    quant_config = None if fp16_export else _quant_config(is_full, method, group_size, profile)
    if profile == LEGACY_W8A16_PROFILE:
        validate_legacy_text_profile(dict(config, quant_config=quant_config))
    specs = resolve_quant_specs(is_full, quant_config)
    expected = {
        f"{TEXT_PREFIX}{relpath}.weight": spec
        for relpath, spec in specs.items()
    }

    is_sharded, shards, weight_map, index = _checkpoint_layout(src)
    if fp16_export and any(not isinstance(name, str) or Path(name).name != name
                           or not name.endswith(".safetensors") for name in shards):
        raise ValueError("source shards must be adjacent safetensors files")
    if fp16_export and is_sharded and (src / "model.safetensors").exists():
        raise ValueError("ambiguous source checkpoint layout")
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(f"temporary output already exists: {tmp}")

    stats = {
        "recipe": recipe,
        "n_quantized": 0,
        "n_copied": 0,
        "bytes_in": 0,
        "bytes_out": 0,
        "n_shards": len(shards),
        "projection_counts": {FP16: 0, W8A16: 0, W4A16_PGRP: 0},
    }
    new_weight_map = dict(weight_map) if is_sharded else {}
    seen_projections: set[str] = set()
    source_files = {}
    source_locations = {}
    if fp16_export:
        source_files["config.json"] = _sha256(src / "config.json")
        if is_sharded:
            source_files["model.safetensors.index.json"] = _sha256(src / "model.safetensors.index.json")

    try:
        tmp.mkdir(parents=True)
        for shard_idx, shard in enumerate(shards, start=1):
            print(f"[convert_qwen3_5] shard {shard_idx}/{len(shards)}: {shard}",
                  flush=True)
            if fp16_export:
                before = (src / shard).stat()
                source_files[shard] = _sha256(src / shard)
            tensors = load_file(src / shard)
            out: dict[str, torch.Tensor] = {}

            for name in sorted(tensors):
                tensor = tensors[name]
                stats["bytes_in"] += tensor.numel() * tensor.element_size()

                spec = expected.get(name)
                if fp16_export:
                    if name in source_locations:
                        raise ValueError(f"duplicate source tensor {name!r}")
                    source_locations[name] = shard
                    if name.endswith((".qweight", ".qzeros", ".scales", ".weight_scale", ".weight_zero")):
                        raise ValueError(f"source contains quantized tensor {name!r}")
                    if tensor.is_complex() or (name.endswith(".weight") and not tensor.is_floating_point()):
                        raise TypeError(f"FP16 export requires floating weights: {name!r}")
                    if spec is not None:
                        seen_projections.add(name)
                        stats["projection_counts"][FP16] += 1
                    converted = _checked_fp16(tensor, name) if tensor.is_floating_point() else tensor
                    out[name] = converted
                    stats["n_copied"] += 1
                    stats["bytes_out"] += converted.numel() * converted.element_size()
                    continue
                if spec is not None:
                    if not tensor.is_floating_point():
                        raise TypeError(
                            f"cannot quantize non-floating tensor {name!r} "
                            f"with dtype {tensor.dtype} (already quantized?)")
                    seen_projections.add(name)
                    stats["projection_counts"][spec.method] += 1
                    if spec.method == FP16:
                        converted = _as_fp16(tensor)
                        out[name] = converted
                        stats["n_copied"] += 1
                        stats["bytes_out"] += (
                            converted.numel() * converted.element_size())
                        continue

                    # fp16 FIRST — see the module docstring.
                    src_fp16 = _as_fp16(tensor)
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

                converted = (_as_fp16(tensor)
                             if tensor.is_floating_point() else tensor)
                out[name] = converted
                stats["n_copied"] += 1
                stats["bytes_out"] += converted.numel() * converted.element_size()

            save_file(out, tmp / shard, metadata={"format": "pt"})
            del tensors
            del out
            if fp16_export:
                after = (src / shard).stat()
                if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                    raise ValueError(f"source shard changed during conversion: {shard}")

        missing = sorted(set(expected) - seen_projections)
        if missing:
            raise KeyError(
                f"checkpoint is missing {len(missing)} projection weight(s) the "
                f"Qwen3.5 quantization scope requires: {missing[:8]}"
                f"{' ...' if len(missing) > 8 else ''}")
        if fp16_export and is_sharded and source_locations != weight_map:
            raise ValueError("source shard contents disagree with weight map")

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
        if fp16_export:
            for name in ("text_config", "vision_config"):
                if isinstance(config.get(name), dict):
                    config[name]["dtype"] = "float16"
                    config[name]["torch_dtype"] = "float16"
            copied = _copy_sidecar_files(src, tmp, set(shards))
            source_files.update({name: _sha256(tmp / name) for name in copied})
            for name in ("config.json", "model.safetensors.index.json"):
                if name in source_files and _sha256(src / name) != source_files[name]:
                    raise ValueError(f"source metadata changed during conversion: {name}")
            config["weight_conversion"] = {
                "method": FP16,
                "producer": "rpu_backend.quant.convert_qwen3_5",
                "producer_sha256": _sha256(Path(__file__)),
                "torch_version": torch.__version__,
                "source_files_sha256": dict(sorted(source_files.items())),
                "source_sha256": hashlib.sha256(json.dumps(source_files, sort_keys=True,
                                                          separators=(",", ":")).encode()).hexdigest(),
                "casting": "stored floating dtype directly to float16; integer buffers unchanged",
            }
        else:
            config["quant_config"] = quant_config
        with (tmp / "config.json").open("w") as f:
            json.dump(config, f, indent=2)

        if not fp16_export:
            _copy_sidecar_files(src, tmp, set(shards))
        write_record(tmp, FAMILY, recipe)
        tmp.rename(dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="source HF checkpoint directory")
    parser.add_argument("--dst", required=True, help="output checkpoint directory")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--method", choices=METHODS,
                           help="fp16 = floating-point export without quantization; "
                                "w8a16 = per-output-channel int8; "
                                "w4a16_pgrp = group-wise int4 along K "
                                "(default: w8a16)")
    precision.add_argument("--profile", choices=PROFILES,
                           help="configuration-driven mixed-precision profile")
    precision.add_argument("--recipe", choices=sorted(RECIPES),
                           help="registered precision recipe; sets method/profile and group size")
    parser.add_argument("--group-size", type=int,
                        help=f"K-group size for w4a16_pgrp (default {DEFAULT_GROUP_SIZE})")
    args = parser.parse_args(argv)
    if args.recipe is not None and args.group_size is not None:
        parser.error("--recipe fixes the group size; drop --group-size")
    arguments = (dict(RECIPES[args.recipe]) if args.recipe is not None else
                 {"method": args.method, "profile": args.profile,
                  "group_size": DEFAULT_GROUP_SIZE if args.group_size is None else args.group_size})
    arguments.setdefault("group_size", DEFAULT_GROUP_SIZE)

    try:
        stats = convert_checkpoint(args.src, args.dst, recipe=args.recipe, **arguments)
    except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"[convert_qwen3_5] error: {exc}", file=sys.stderr)
        return 1

    print(f"[convert_qwen3_5] {Path(args.src).resolve()}")
    print(f"              -> {Path(args.dst).resolve()}")
    print(f"  shards    : {stats['n_shards']}")
    profile, method = arguments.get("profile"), arguments.get("method")
    selected = f"profile={profile}" if profile else f"method={method or W8A16}"
    uses_w4 = (profile in ("w4_main_w8_gdn", "w4_mlp_only", "w4_mlp_w8_attn_gdn")
               or method == W4A16_PGRP)
    print(f"  precision : {selected}"
          + (f" (group_size={arguments['group_size']})" if uses_w4 else ""))
    print(f"  recipe    : {stats['recipe'] or 'none registered (no record written)'}")
    print(f"  quantized : {stats['n_quantized']} weights")
    print(f"  projections: {stats['projection_counts']}")
    print(f"  copied    : {stats['n_copied']} tensors")
    print(f"  size      : {stats['bytes_in'] / 1e9:.2f} GB"
          f" -> {stats['bytes_out'] / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
