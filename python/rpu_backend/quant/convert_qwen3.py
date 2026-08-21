"""Offline W8A16 quantization for Qwen3 HF checkpoints.

The converter quantizes the seven decoder projection weights to int8 with a
per-output-channel fp16 scale. Other floating tensors are stored as fp16. Both
single-file and sharded safetensors checkpoints are supported.
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
    from ._common import dequantize_linear_per_channel, quantize_linear_per_channel
except ImportError:
    from _common import dequantize_linear_per_channel, quantize_linear_per_channel


QUANT_PROJ_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)

LM_HEAD_WEIGHT_NAME = "lm_head.weight"
EMBED_TOKENS_WEIGHT_NAME = "model.embed_tokens.weight"


def _should_quantize(name: str, skip_modules: list[str]) -> bool:
    if not name.endswith(QUANT_PROJ_SUFFIXES):
        return False
    return not any(skip in name for skip in skip_modules)


def _scale_name(weight_name: str) -> str:
    return weight_name[: -len(".weight")] + ".weight_scale"


def _checkpoint_layout(src: Path) -> tuple[bool, list[str], dict[str, str], dict]:
    index_path = src / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as f:
            index = json.load(f)
        weight_map = dict(index["weight_map"])
        shards = sorted(set(weight_map.values()))
        return True, shards, weight_map, index

    weights_path = src / "model.safetensors"
    if weights_path.is_file():
        return False, ["model.safetensors"], {}, {}

    raise FileNotFoundError(f"no model.safetensors or model.safetensors.index.json in {src}")


def _convert_tensor(
    name: str,
    tensor: torch.Tensor,
    skip_modules: list[str],
    *,
    quant_lm_head: bool = False,
    quant_embed_tokens: bool = False,
) -> tuple[dict[str, torch.Tensor], int, int, int]:
    bytes_in = tensor.numel() * tensor.element_size()
    should_quantize = _should_quantize(name, skip_modules) or (
        quant_lm_head and name == LM_HEAD_WEIGHT_NAME
    ) or (
        quant_embed_tokens and name == EMBED_TOKENS_WEIGHT_NAME
    )
    if should_quantize:
        if tensor.dtype == torch.int8:
            # Accept an existing W8A16 checkpoint as input. Its projection
            # weights are already quantized; the matching .weight_scale tensor
            # will be copied by the generic path below.
            return {name: tensor}, 0, 1, bytes_in
        if not tensor.is_floating_point():
            raise TypeError(
                f"cannot W8A16-quantize non-floating tensor {name!r} "
                f"with dtype {tensor.dtype}"
            )
        w_int8, scale = quantize_linear_per_channel(tensor)
        out = {name: w_int8, _scale_name(name): scale}
        bytes_out = (
            w_int8.numel() * w_int8.element_size()
            + scale.numel() * scale.element_size()
        )
        return out, 1, 0, bytes_out

    out_tensor = tensor.to(torch.float16) if tensor.is_floating_point() else tensor
    bytes_out = out_tensor.numel() * out_tensor.element_size()
    return {name: out_tensor}, 0, 1, bytes_out


def _copy_sidecar_files(src: Path, dst: Path, shards: set[str]) -> None:
    skip = set(shards)
    skip.update({"model.safetensors", "model.safetensors.index.json", "config.json"})
    for path in sorted(src.iterdir()):
        if path.name in skip or not path.is_file():
            continue
        shutil.copy2(path, dst / path.name)


def convert_checkpoint(
    src: str | Path,
    dst: str | Path,
    skip_modules: list[str],
    *,
    quant_lm_head: bool = False,
    quant_embed_tokens: bool = False,
) -> dict:
    src = Path(src).expanduser().resolve()
    dst = Path(dst).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"--src is not a directory: {src}")
    if dst.exists():
        raise FileExistsError(f"--dst already exists, refusing to overwrite: {dst}")

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
    }
    new_weight_map = dict(weight_map) if is_sharded else {}
    source_names = set(weight_map) if is_sharded else set()
    generated_lm_head = False

    try:
        tmp.mkdir(parents=True)
        for shard_idx, shard in enumerate(shards, start=1):
            print(f"[convert_qwen3] shard {shard_idx}/{len(shards)}: {shard}", flush=True)
            tensors = load_file(src / shard)
            if not is_sharded and not source_names:
                source_names = set(tensors)
            out: dict[str, torch.Tensor] = {}

            for name in sorted(tensors):
                converted, n_quant, n_copy, bytes_out = _convert_tensor(
                    name,
                    tensors[name],
                    skip_modules,
                    quant_lm_head=quant_lm_head,
                    quant_embed_tokens=quant_embed_tokens,
                )
                out.update(converted)
                stats["n_quantized"] += n_quant
                stats["n_copied"] += n_copy
                stats["bytes_in"] += tensors[name].numel() * tensors[name].element_size()
                stats["bytes_out"] += bytes_out
                if n_quant:
                    new_weight_map[_scale_name(name)] = shard

                if (
                    quant_lm_head
                    and name == EMBED_TOKENS_WEIGHT_NAME
                    and LM_HEAD_WEIGHT_NAME not in source_names
                    and not generated_lm_head
                ):
                    if quant_embed_tokens:
                        w_int8 = converted[EMBED_TOKENS_WEIGHT_NAME].clone()
                        scale = converted[_scale_name(EMBED_TOKENS_WEIGHT_NAME)].clone()
                    else:
                        w_int8, scale = quantize_linear_per_channel(tensors[name])
                    scale_name = _scale_name(LM_HEAD_WEIGHT_NAME)
                    out[LM_HEAD_WEIGHT_NAME] = w_int8
                    out[scale_name] = scale
                    lm_head_bytes = (
                        w_int8.numel() * w_int8.element_size()
                        + scale.numel() * scale.element_size()
                    )
                    stats["n_quantized"] += 1
                    stats["bytes_out"] += lm_head_bytes
                    new_weight_map[LM_HEAD_WEIGHT_NAME] = shard
                    new_weight_map[scale_name] = shard
                    generated_lm_head = True

            save_file(out, tmp / shard, metadata={"format": "pt"})
            del tensors
            del out

        if is_sharded:
            new_index = dict(index)
            metadata = dict(new_index.get("metadata", {}))
            metadata["total_size"] = stats["bytes_out"]
            new_index["metadata"] = metadata
            new_index["weight_map"] = dict(sorted(new_weight_map.items()))
            with (tmp / "model.safetensors.index.json").open("w") as f:
                json.dump(new_index, f, indent=2)

        with (src / "config.json").open() as f:
            config = json.load(f)
        config["torch_dtype"] = "float16"
        config["dtype"] = "float16"
        if quant_lm_head or quant_embed_tokens:
            config["tie_word_embeddings"] = False
        effective_skip_modules = list(skip_modules)
        if quant_lm_head:
            effective_skip_modules = [
                item for item in effective_skip_modules if item != "lm_head"
            ]
        config["quant_config"] = {
            "method": "w8a16",
            "mode": "per_channel_symmetric",
            "qaxis": 0,
            "skip_modules": effective_skip_modules,
            "quantized_lm_head": bool(quant_lm_head),
            "quantized_embed_tokens": bool(quant_embed_tokens),
            "lm_head_untied": bool(quant_lm_head),
            "embed_tokens_untied": bool(quant_embed_tokens),
        }
        with (tmp / "config.json").open("w") as f:
            json.dump(config, f, indent=2)

        _copy_sidecar_files(src, tmp, set(shards))
        tmp.rename(dst)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="source HF checkpoint directory")
    parser.add_argument("--dst", required=True, help="output W8A16 checkpoint directory")
    parser.add_argument(
        "--skip-modules",
        nargs="+",
        default=["lm_head"],
        help="module-name substrings never quantized",
    )
    parser.add_argument(
        "--quant-lm-head",
        action="store_true",
        help=(
            "Quantize lm_head to int8+fp16 scale. If the source checkpoint is "
            "tied and omits lm_head.weight, derive an untied lm_head from "
            "model.embed_tokens.weight."
        ),
    )
    parser.add_argument(
        "--quant-embed-tokens",
        action="store_true",
        help=(
            "Experimentally quantize model.embed_tokens.weight to int8+fp16 "
            "scale. Requires --quant-lm-head because tied fp16 lm_head and "
            "int8 embeddings cannot share one Parameter safely."
        ),
    )
    args = parser.parse_args(argv)
    if args.quant_embed_tokens and not args.quant_lm_head:
        print(
            "[convert_qwen3] error: --quant-embed-tokens requires --quant-lm-head",
            file=sys.stderr,
        )
        return 1

    try:
        stats = convert_checkpoint(
            args.src,
            args.dst,
            args.skip_modules,
            quant_lm_head=args.quant_lm_head,
            quant_embed_tokens=args.quant_embed_tokens,
        )
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        print(f"[convert_qwen3] error: {exc}", file=sys.stderr)
        return 1

    print(f"[convert_qwen3] {Path(args.src).resolve()}")
    print(f"            -> {Path(args.dst).resolve()}")
    print(f"  shards    : {stats['n_shards']}")
    print(f"  quantized : {stats['n_quantized']} weights")
    print(f"  copied    : {stats['n_copied']} tensors")
    print(f"  lm_head   : {'quantized' if args.quant_lm_head else 'fp16'}")
    print(f"  embed     : {'quantized' if args.quant_embed_tokens else 'fp16'}")
    print(
        f"  size      : {stats['bytes_in'] / 1e9:.2f} GB"
        f" -> {stats['bytes_out'] / 1e9:.2f} GB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
