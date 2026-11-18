"""Offline routed-only W8A16 conversion of official floating Qwen3.5-35B-A3B.

Only the 80 fused routed-expert tensors (``gate_up_proj [E,2I,H]`` and
``down_proj [E,H,I]`` of every layer) are quantized: each expert's output row
gets one FP16 scale (symmetric, ``max|w|/127``) and signed INT8 codes, the
grouped kernel's mode 1. Every other tensor - attention/GDN mixers, router,
shared expert, norms, embeddings, lm_head, MTP and Vision - is copied as FP16,
the same split the routed UINT4 profiles use.
Run this file directly with TORCH_DEVICE_BACKEND_AUTOLOAD=0 on a CPU host.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

if __package__:
    from . import convert_qwen3_5_moe_fp8 as _fp8_converter
    from ._common import quantize_linear_per_channel
else:
    # Load only CPU leaves; importing rpu_backend would initialize its native SO.
    def _leaf(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    _root = Path(__file__).resolve().parent
    _fp8_converter = _leaf("_offline_moe_int8_layout", _root / "convert_qwen3_5_moe_fp8.py")
    quantize_linear_per_channel = _leaf(
        "_offline_moe_int8_common", _root / "_common.py").quantize_linear_per_channel

_scope, _recipes = _fp8_converter._scope, _fp8_converter._recipes

# The one registered precision recipe this converter produces.
FAMILY, RECIPE = "qwen3_5", "w8a16_experts"


def quantize_routed_experts(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Float ``[E,N,K]`` -> (INT8 codes ``[E,N,K]``, FP16 scales ``[E,N]``)."""
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"routed INT8 source must be FP16/BF16/FP32, got {tensor.dtype}")
    if tensor.dim() != 3:
        raise ValueError(f"routed INT8 source must be [E,N,K], got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("routed INT8 source contains nonfinite values")
    codes = torch.empty(tuple(tensor.shape), dtype=torch.int8)
    scales = torch.empty(tuple(tensor.shape[:-1]), dtype=torch.float16)
    for expert in range(int(tensor.shape[0])):
        codes[expert], scales[expert] = quantize_linear_per_channel(tensor[expert], bits=8)
    if not bool(torch.isfinite(scales).all()):
        raise ValueError("routed INT8 source overflows FP16 scales")
    return codes, scales


def convert_checkpoint(src: str | Path, dst: str | Path) -> dict:
    src = Path(src).expanduser().resolve()
    dst = Path(dst).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"source checkpoint is not a directory: {src}")
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")
    if src == dst or src in dst.parents:
        raise ValueError("output must be outside the source checkpoint")
    if not (src / "generation_config.json").is_file():
        raise FileNotFoundError("source needs generation_config.json for the public loader")
    with (src / "config.json").open() as handle:
        config = json.load(handle)
    num_layers, num_experts, routed_i, hidden, layer_types = _fp8_converter._validate_config(config)
    is_sharded, shards, weight_map, index = _fp8_converter._checkpoint_layout(src)
    expected_names = set(_scope.expected_int8_weight_names(layer_types))
    missing = sorted(expected_names - set(weight_map))
    if missing:
        raise ValueError(f"source lacks the complete routed-expert inventory: {missing[:8]}")
    for name in weight_map:
        if (name.startswith(_scope.TEXT_PREFIX) and ".mlp.experts." in name
                and name not in expected_names):
            raise ValueError(f"unsupported routed-expert source key: {name}")
        if name.endswith(("_scale", "_scale_inv", ".qweight", ".scales", ".qzeros", ".g_idx")):
            raise ValueError(f"unexpected quantization key: {name}")
    temp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    if temp.exists():
        raise FileExistsError(f"temporary output already exists: {temp}")
    new_weight_map = dict(weight_map)
    quantized: list[str] = []
    stats = {"n_quantized": 0, "n_copied": 0, "n_shards": len(shards),
             "bytes_in": 0, "bytes_out": 0, "zero_rows": 0}
    try:
        temp.mkdir(parents=True)
        for shard_index, shard in enumerate(shards, start=1):
            print(f"[qwen3_5_moe_int8] shard {shard_index}/{len(shards)}: {shard}", flush=True)
            tensors = load_file(src / shard)
            if set(tensors) != {key for key, value in weight_map.items() if value == shard}:
                raise ValueError(f"source index/shard inventory mismatch: {shard}")
            out: dict[str, torch.Tensor] = {}
            for name in sorted(tensors):
                tensor = tensors[name]
                stats["bytes_in"] += tensor.numel() * tensor.element_size()
                if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise ValueError(f"source tensor must be FP16/BF16/FP32: {name}")
                if name in expected_names:
                    expected = ((num_experts, 2 * routed_i, hidden)
                                if name.endswith("gate_up_proj")
                                else (num_experts, hidden, routed_i))
                    if tuple(tensor.shape) != expected:
                        raise ValueError(f"{name} shape {tuple(tensor.shape)} must be {expected}")
                    codes, scales = quantize_routed_experts(tensor)
                    sname = _scope.scale_name(name)
                    out[name], out[sname] = codes, scales
                    new_weight_map[sname] = shard
                    quantized.append(name)
                    stats["n_quantized"] += 1
                    stats["zero_rows"] += int((codes == 0).all(dim=-1).sum())
                    stats["bytes_out"] += codes.numel() + scales.numel() * 2
                else:
                    copied = tensor.to(torch.float16)
                    if not bool(torch.isfinite(copied).all()):
                        raise ValueError(f"source tensor is nonfinite or overflows FP16: {name}")
                    out[name] = copied
                    stats["n_copied"] += 1
                    stats["bytes_out"] += copied.numel() * copied.element_size()
            save_file(out, temp / shard, metadata={"format": "pt"})
            del tensors, out

        if set(quantized) != expected_names:
            raise ValueError(f"quantized {len(quantized)} tensors, expected {len(expected_names)}")
        if is_sharded:
            new_index = dict(index)
            new_index["metadata"] = {**new_index.get("metadata", {}), "total_size": stats["bytes_out"]}
            new_index["weight_map"] = dict(sorted(new_weight_map.items()))
            with (temp / "model.safetensors.index.json").open("w") as handle:
                json.dump(new_index, handle, indent=2)
        config["torch_dtype"] = config["dtype"] = "float16"
        for name in ("text_config", "vision_config"):
            if isinstance(config.get(name), dict):
                config[name]["dtype"] = "float16"
                if "torch_dtype" in config[name]:
                    config[name]["torch_dtype"] = "float16"
        config["quant_config"] = {
            "method": _scope.INT8_METHOD,
            "format": _scope.INT8_FORMAT,
            "weight_mode": 1,
            "qaxis": -1,
            "scale_dtype": "float16",
            "scope": _scope.INT8_SCOPE,
            "profile": _scope.INT8_PROFILE_ID,
            "symmetric": True,
            "calibration": False,
            "fp16_preserve": list(_scope.INT8_FP16_PRESERVE),
            "quantized_tensor_count": len(quantized),
            "quantized_tensors": sorted(quantized),
        }
        _scope.validate_int8_config(config["quant_config"], layer_types)
        with (temp / "config.json").open("w") as handle:
            json.dump(config, handle, indent=2)
        _fp8_converter._copy_sidecars(src, temp, set(shards))
        _recipes.write_record(temp, FAMILY, RECIPE)
        temp.rename(dst)
    except BaseException:
        # The destination appears only through the rename above.
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="official unquantized floating 35B-A3B directory")
    parser.add_argument("--dst", required=True, help="new output directory; existing paths are rejected")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads (default: 8)")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        stats = convert_checkpoint(args.src, args.dst)
    except Exception as exc:
        print(f"[qwen3_5_moe_int8] error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FAMILY", "RECIPE", "convert_checkpoint", "quantize_routed_experts"]
