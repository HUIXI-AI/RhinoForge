"""Offline W8A16 quantization for Wall-OSS-0.5 — every 2D linear weight except skips.

Per-output-channel symmetric int8: each qualifying ``*.weight`` [N, K] becomes an
int8 tensor plus an fp16 ``*.weight_scale`` [N]. Covers the decoder expert
projections, the vision transformer blocks AND the action_preprocessor — i.e. all
2D ``.weight`` tensors except ``embed_tokens`` — matching the tested
``wall-oss-0.5-w8a16`` checkpoint (rpu_quant_config method=w8a16,
per_channel_symmetric, 422 weights). Non-2D / skipped tensors are copied
fp16/original; sidecar files (config, tokenizer, normalizers) are copied through.

CLI: ``python -m`` does NOT work on the compiled wheel module — call
``convert_checkpoint(src, dst)`` (or the packaged ``quantize.py``).
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
except ImportError:  # standalone import (no rpu_backend package)
    from _common import quantize_linear_per_channel

DEFAULT_SKIP_MODULES = ("embed_tokens",)


def _scale_name(weight_name: str) -> str:
    return weight_name[: -len(".weight")] + ".weight_scale"


def _should_quantize(name: str, tensor, skip_modules) -> bool:
    return (
        name.endswith(".weight")
        and tensor.dim() == 2
        and tensor.is_floating_point()
        and not any(skip in name for skip in skip_modules)
    )


def _checkpoint_layout(src: Path):
    index_path = src / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.load(index_path.open())
        weight_map = dict(index["weight_map"])
        return True, sorted(set(weight_map.values())), weight_map, index
    if (src / "model.safetensors").is_file():
        return False, ["model.safetensors"], {}, {}
    raise FileNotFoundError(f"no safetensors in {src}")


def _copy_sidecar_files(src: Path, dst: Path, shards) -> None:
    skip = set(shards) | {
        "model.safetensors", "model.safetensors.index.json",
        "config.json", "rpu_quant_config.json",
    }
    for path in sorted(src.iterdir()):
        if path.name in skip or not path.is_file():
            continue
        shutil.copy2(path, dst / path.name)


def _quant_config(src: Path, skip_modules, quantized_weights) -> dict:
    return {
        "method": "w8a16",
        "mode": "per_channel_symmetric",
        "qaxis": 0,
        "source": Path(src).name,  # basename only — never leak the build-host abs path
        "target": "wall_oss_2d_linear_weights_except_skip_modules",
        "storage": "int8",
        "value_bits": 8,
        "scale": "per_output_channel",
        "skip_modules": list(skip_modules),
        "quantized_weight_count": len(quantized_weights),
        "quantized_weights": sorted(quantized_weights),
    }


def convert_checkpoint(src, dst, *, skip_modules=DEFAULT_SKIP_MODULES):
    src, dst = Path(src).expanduser().resolve(), Path(dst).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"--src not a dir: {src}")
    if dst.exists():
        raise FileExistsError(f"--dst exists: {dst}")
    skip_modules = tuple(skip_modules)
    is_sharded, shards, weight_map, index = _checkpoint_layout(src)
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    new_weight_map = dict(weight_map) if is_sharded else {}
    quantized: list[str] = []
    n_copied = 0
    try:
        tmp.mkdir(parents=True)
        for i, shard in enumerate(shards, 1):
            print(f"[convert_wall_oss_w8a16] shard {i}/{len(shards)}: {shard}", flush=True)
            tensors = load_file(src / shard)
            out = {}
            for name in sorted(tensors):
                t = tensors[name]
                if _should_quantize(name, t, skip_modules):
                    w_int8, scale = quantize_linear_per_channel(t, bits=8)
                    out[name] = w_int8
                    out[_scale_name(name)] = scale
                    quantized.append(name)
                    new_weight_map[_scale_name(name)] = shard
                else:
                    out[name] = t.to(torch.float16) if t.is_floating_point() else t
                    n_copied += 1
            save_file(out, tmp / shard, metadata={"format": "pt"})
            del tensors, out
        if not quantized:
            raise ValueError("no 2D linear weights matched; wrong --src?")
        if is_sharded:
            new_index = dict(index)
            new_index["weight_map"] = dict(sorted(new_weight_map.items()))
            json.dump(new_index, (tmp / "model.safetensors.index.json").open("w"), indent=2)
        config = json.load((src / "config.json").open())
        json.dump(config, (tmp / "config.json").open("w"), indent=2)
        json.dump(_quant_config(src, skip_modules, quantized),
                  (tmp / "rpu_quant_config.json").open("w"), indent=2)
        _copy_sidecar_files(src, tmp, set(shards))
        tmp.rename(dst)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {"n_quantized": len(quantized), "n_copied": n_copied, "n_shards": len(shards)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Wall-OSS-0.5 fp16 -> W8A16 (all 2D linear except skips).")
    ap.add_argument("--src", required=True, help="fp16 Wall-OSS-0.5 checkpoint dir")
    ap.add_argument("--dst", required=True, help="output w8a16 checkpoint dir")
    ap.add_argument("--skip-modules", default=",".join(DEFAULT_SKIP_MODULES),
                    help="comma-separated name substrings to NOT quantize (default: embed_tokens)")
    args = ap.parse_args(argv)
    skip = tuple(s.strip() for s in args.skip_modules.split(",") if s.strip())
    try:
        stats = convert_checkpoint(args.src, args.dst, skip_modules=skip)
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        print(f"[convert_wall_oss_w8a16] error: {exc}", file=sys.stderr)
        return 1
    print(f"[convert_wall_oss_w8a16] {args.src} -> {args.dst}")
    print(f"  quantized (int8): {stats['n_quantized']}  copied: {stats['n_copied']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
