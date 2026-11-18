"""Offline Rhino-compatible mixed-FP8 conversion for Qwen3.5-MoE.

Routed experts and the dense Qwen3.5 text-mixer scope are quantized.  Each
logical output row gets one FP16 scale and E4M3 raw uint8 codes.  The explicit
FP16 preserve set mirrors dense Qwen3.5: GDN b/a projections, convolution,
norms, A_log/dt_bias, embeddings, lm_head, router, shared-expert path, and
Vision.

This format intentionally differs from the official Qwen3.5 FP8 checkpoint's
128x128 ``weight_scale_inv`` blocks and dynamic activation scaling.
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
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import Qwen3_5MoeConfig

if __package__:
    from .fp8 import quantize_fp8_e4m3_per_output
    from rpu_backend.adapters.qwen3_5_moe import quant_scope as _scope
else:
    # Load only CPU leaves; importing rpu_backend would initialize its native SO.
    _fp8_spec = importlib.util.spec_from_file_location(
        "_offline_moe_fp8", Path(__file__).resolve().with_name("fp8.py")
    )
    _fp8 = importlib.util.module_from_spec(_fp8_spec)
    sys.modules[_fp8_spec.name] = _fp8
    _fp8_spec.loader.exec_module(_fp8)
    quantize_fp8_e4m3_per_output = _fp8.quantize_fp8_e4m3_per_output
    _scope_spec = importlib.util.spec_from_file_location(
        "_offline_moe_fp8_scope",
        Path(__file__).resolve().parents[1] / "adapters/qwen3_5_moe/quant_scope.py",
    )
    _scope = importlib.util.module_from_spec(_scope_spec)
    sys.modules[_scope_spec.name] = _scope
    _scope_spec.loader.exec_module(_scope)

FP8_FORMAT, FP8_METHOD = _scope.FP8_FORMAT, _scope.FP8_METHOD
FP8_PROFILE, FP8_SCOPE = _scope.FP8_PROFILE, _scope.FP8_SCOPE
FP16_PRESERVE = _scope.FP16_PRESERVE
FULL_ATTN_FP8_ROLES, GDN_FP8_ROLES = _scope.FULL_ATTN_FP8_ROLES, _scope.GDN_FP8_ROLES
ROUTED_EXPERT_SUFFIXES = _scope.ROUTED_EXPERT_SUFFIXES
is_fp8_weight = _scope.is_fp8_weight
is_routed_expert_weight = _scope.is_routed_expert_weight
is_text_mixer_weight = _scope.is_text_mixer_weight
scale_name = _scope.scale_name
expected_fp8_weight_names = _scope.expected_fp8_weight_names
validate_exact_profile = _scope.validate_exact_profile
validate_quant_config = _scope.validate_quant_config


def _checkpoint_layout(src: Path) -> tuple[bool, list[str], dict[str, str], dict]:
    index_path = src / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as handle:
            index = json.load(handle)
        weight_map = index["weight_map"]
        is_sharded = True
    elif (src / "model.safetensors").is_file():
        with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as file:
            weight_map = {key: "model.safetensors" for key in file.keys()}
        index, is_sharded = {}, False
    else:
        raise FileNotFoundError(
            f"no model.safetensors or model.safetensors.index.json in {src}"
        )
    if (not isinstance(weight_map, dict) or not weight_map
            or any(not isinstance(key, str) for key in weight_map)):
        raise ValueError("source needs a nonempty safetensors weight_map")
    for filename in weight_map.values():
        if (not isinstance(filename, str) or Path(filename).name != filename
                or "\\" in filename or not filename.endswith(".safetensors")):
            raise ValueError("source shard names must be local safetensors basenames")
    return is_sharded, sorted(set(weight_map.values())), dict(weight_map), index


def _validate_config(
    config: dict,
) -> tuple[int, int, int, int, tuple[str, ...]]:
    for owner in (config, config.get("text_config", {}), config.get("vision_config", {})):
        if any(key in owner for key in ("quant_config", "quantization_config", "rtn_metadata")):
            raise ValueError("source must be unquantized floating 35B-A3B")
    if config.get("model_type") != "qwen3_5_moe":
        raise ValueError("source must be unquantized floating 35B-A3B")
    # Official JSON omits some HF defaults; use the same admission as the loader.
    resolved = Qwen3_5MoeConfig.from_dict(config)
    validate_exact_profile(resolved, require_quant=False)
    text = resolved.text_config.to_dict()
    required = ("num_hidden_layers", "num_experts", "moe_intermediate_size", "hidden_size")
    missing = [name for name in required if name not in text]
    if missing:
        raise ValueError(f"text config is missing {missing}")
    values = tuple(int(text[name]) for name in required)
    layer_types = tuple(text.get("layer_types") or ())
    if len(layer_types) != values[0]:
        raise ValueError(
            f"layer_types length {len(layer_types)} must equal "
            f"num_hidden_layers={values[0]}"
        )
    bad = sorted(set(layer_types) - {"full_attention", "linear_attention"})
    if bad:
        raise ValueError(f"unsupported layer_types {bad}")
    return (*values, layer_types)


def _quantize_chunked(
    tensor: torch.Tensor, *, expert_chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(
            f"raw FP8 source must be FP16/BF16/FP32, got {tensor.dtype}; "
            "official block-scaled FP8 is not a valid source for this converter"
        )
    if tensor.dim() not in (2, 3):
        raise ValueError(f"FP8 source tensor must be rank-2/3, got {tensor.shape}")
    if expert_chunk < 1:
        raise ValueError("expert_chunk must be positive")
    codes = torch.empty(tuple(tensor.shape), dtype=torch.uint8)
    scales = torch.empty(tuple(tensor.shape[:-1]), dtype=torch.float16)
    for start in range(0, int(tensor.shape[0]), expert_chunk):
        stop = min(int(tensor.shape[0]), start + expert_chunk)
        q, s = quantize_fp8_e4m3_per_output(tensor[start:stop])
        if not bool(torch.isfinite(s).all()) or not bool((s > 0).all()):
            raise ValueError("FP8 source cannot be represented with finite positive FP16 scales")
        codes[start:stop].copy_(q)
        scales[start:stop].copy_(s)
    return codes, scales


def _mixer_shapes(config: dict) -> dict[str, tuple[int, int]]:
    text = Qwen3_5MoeConfig.from_dict(config).text_config
    hidden = text.hidden_size
    query = text.num_attention_heads * text.head_dim
    kv = text.num_key_value_heads * text.head_dim
    linear_key = text.linear_num_key_heads * text.linear_key_head_dim
    linear_value = text.linear_num_value_heads * text.linear_value_head_dim
    return {
        "self_attn.q_proj.weight":
            (query * (2 if getattr(text, "attn_output_gate", True) else 1), hidden),
        "self_attn.k_proj.weight": (kv, hidden),
        "self_attn.v_proj.weight": (kv, hidden),
        "self_attn.o_proj.weight": (hidden, query),
        "linear_attn.in_proj_qkv.weight": (2 * linear_key + linear_value, hidden),
        "linear_attn.in_proj_z.weight": (linear_value, hidden),
        "linear_attn.out_proj.weight": (hidden, linear_value),
    }


def _copy_sidecars(src: Path, dst: Path, shards: set[str]) -> None:
    skip = set(shards) | {
        "model.safetensors", "model.safetensors.index.json", "config.json"
    }
    for path in sorted(src.iterdir()):
        if (path.is_file() and path.name not in skip
                and not path.name.endswith((".safetensors", ".bin"))):
            shutil.copy2(path, dst / path.name)


def convert_checkpoint(
    src: str | Path,
    dst: str | Path,
    *,
    expert_chunk: int = 1,
) -> dict:
    src = Path(src).expanduser().resolve()
    dst = Path(dst).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"source checkpoint is not a directory: {src}")
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")
    if src == dst or src in dst.parents:
        raise ValueError("output must be outside the source checkpoint")
    if expert_chunk < 1:
        raise ValueError("expert_chunk must be positive")
    if not (src / "generation_config.json").is_file():
        raise FileNotFoundError("source needs generation_config.json for the public loader")
    with (src / "config.json").open() as handle:
        config = json.load(handle)
    num_layers, num_experts, routed_i, hidden, layer_types = _validate_config(
        config
    )
    mixer_shapes = _mixer_shapes(config)
    is_sharded, shards, weight_map, index = _checkpoint_layout(src)
    expected_names = set(expected_fp8_weight_names(layer_types))
    missing = sorted(expected_names - set(weight_map))
    if missing:
        raise ValueError(f"source lacks the complete mixed-FP8 inventory: {missing[:8]}")
    for name in weight_map:
        if (is_fp8_weight(name, layer_types) and name not in expected_names
                or name.startswith(_scope.TEXT_PREFIX) and ".mlp.experts." in name
                and not is_routed_expert_weight(name)):
            raise ValueError(f"unsupported mixed-FP8 source key: {name}")
        if name.endswith(("_scale", "_scale_inv", ".qweight", ".scales", ".qzeros", ".g_idx")):
            raise ValueError(f"unexpected quantization key: {name}")
    temp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    if temp.exists():
        raise FileExistsError(f"temporary output already exists: {temp}")
    new_weight_map = dict(weight_map)
    quantized_names: list[str] = []
    stats = {
        "n_quantized": 0,
        "n_copied": 0,
        "n_shards": len(shards),
        "bytes_in": 0,
        "bytes_out": 0,
    }

    try:
        temp.mkdir(parents=True)
        for shard_index, shard in enumerate(shards, start=1):
            print(
                f"[qwen3_5_moe_fp8] shard {shard_index}/{len(shards)}: {shard}",
                flush=True,
            )
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
                    if is_routed_expert_weight(name):
                        expected = (
                            (num_experts, 2 * routed_i, hidden)
                            if name.endswith("gate_up_proj")
                            else (num_experts, hidden, routed_i)
                        )
                        if tuple(tensor.shape) != expected:
                            raise ValueError(
                                f"{name} shape {tuple(tensor.shape)} must be "
                                f"{expected}"
                            )
                    else:
                        role = name[len(_scope.TEXT_PREFIX):].partition(".")[2]
                        expected = mixer_shapes[role]
                        if tuple(tensor.shape) != expected:
                            raise ValueError(
                                f"text mixer {name} shape {tuple(tensor.shape)} "
                                f"must be {expected}"
                            )
                    sname = scale_name(name)
                    if sname in tensors or sname in new_weight_map:
                        raise ValueError(f"source unexpectedly contains scale {sname}")
                    codes, scales = _quantize_chunked(
                        tensor, expert_chunk=expert_chunk
                    )
                    out[name] = codes
                    out[sname] = scales
                    new_weight_map[sname] = shard
                    quantized_names.append(name)
                    stats["n_quantized"] += 1
                    stats["bytes_out"] += (
                        codes.numel() * codes.element_size()
                        + scales.numel() * scales.element_size()
                    )
                else:
                    if not bool(torch.isfinite(tensor).all()):
                        raise ValueError(f"source tensor contains nonfinite values: {name}")
                    copied = tensor.to(torch.float16)
                    if not bool(torch.isfinite(copied).all()):
                        raise ValueError(f"source tensor overflows FP16: {name}")
                    out[name] = copied
                    stats["n_copied"] += 1
                    stats["bytes_out"] += copied.numel() * copied.element_size()
            save_file(out, temp / shard, metadata={"format": "pt"})
            del tensors, out

        if set(quantized_names) != expected_names:
            raise ValueError(
                f"quantized {stats['n_quantized']} tensors, expected "
                f"{len(expected_names)} for the mixed-FP8 scope"
            )
        if is_sharded:
            new_index = dict(index)
            metadata = dict(new_index.get("metadata", {}))
            metadata["total_size"] = stats["bytes_out"]
            new_index["metadata"] = metadata
            new_index["weight_map"] = dict(sorted(new_weight_map.items()))
            with (temp / "model.safetensors.index.json").open("w") as handle:
                json.dump(new_index, handle, indent=2)

        config["torch_dtype"] = "float16"
        config["dtype"] = "float16"
        for name in ("text_config", "vision_config"):
            if isinstance(config.get(name), dict):
                config[name]["dtype"] = "float16"
                if "torch_dtype" in config[name]:
                    config[name]["torch_dtype"] = "float16"
        config["quant_config"] = {
            "method": FP8_METHOD,
            "format": FP8_FORMAT,
            "weight_mode": 5,
            "qaxis": -1,
            "scale_dtype": "float16",
            "scope": FP8_SCOPE,
            "profile": FP8_PROFILE,
            "fp16_preserve": list(FP16_PRESERVE),
            "quantized_tensor_count": stats["n_quantized"],
            "quantized_tensors": sorted(quantized_names),
            "official_block_fp8_compatible": False,
        }
        validate_quant_config(config["quant_config"], layer_types)
        with (temp / "config.json").open("w") as handle:
            json.dump(config, handle, indent=2)
        _copy_sidecars(src, temp, set(shards))
        temp.rename(dst)
    except BaseException:
        # Also retire a partial multi-GB checkpoint on Ctrl-C/KeyboardInterrupt.
        # The final destination is published only by the atomic rename below.
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument(
        "--expert-chunk", type=int, default=1,
        help="leading-dimension rows/experts converted at once; 1 minimizes peak memory",
    )
    parser.add_argument("--threads", type=int, default=8, help="CPU threads (default: 8)")
    args = parser.parse_args(argv)
    if args.expert_chunk < 1:
        parser.error("--expert-chunk must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        stats = convert_checkpoint(
            args.src, args.dst, expert_chunk=args.expert_chunk
        )
    except Exception as exc:
        print(f"[qwen3_5_moe_fp8] error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FP8_FORMAT", "FP8_METHOD", "FP8_SCOPE", "FULL_ATTN_FP8_ROLES",
    "GDN_FP8_ROLES", "ROUTED_EXPERT_SUFFIXES", "convert_checkpoint",
    "is_fp8_weight", "is_routed_expert_weight", "is_text_mixer_weight",
    "scale_name",
]
