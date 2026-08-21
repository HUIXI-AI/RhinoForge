"""Offline W4A16 quantization for Wall-OSS-0.5 decoder expert projections.

int4 (per-output-channel symmetric, values in int8 container) for the fused
qkv_proj_experts / o_proj_experts / gate_up_proj / down_proj of BOTH experts.
expert-0 down_proj (K=intermediate=11008) is ALSO int4: the int4 row swizzle
needs K % (64*cores)==512==0, and 11008%512=256, so the runtime pads INTER to
11264 (next multiple of 512) with zero channels at load (bit-exact: silu(0)*0=0,
down pad cols=0) before swizzling — see adapters/wall_oss/llm.py. All non-decoder
tensors (vision, patch_embed, merger, biases, norms, embeds, action_preprocessor)
are copied as fp16/original. The runtime packs int4 -> uint8 [N,K/2] at load and
dispatches PARALLEL_LINEAR_WINT4_PER_CHANNEL.
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
except ImportError:  # standalone import (no rpu_backend package)
    from _common import quantize_linear_per_channel
    from int4_pgrp_pack import quantize_int4_group_wise

# Fused on-disk decoder projections (see adapters/wall_oss/llm.py weight build).
WALL_OSS_QUANT_SUFFIXES = (
    "self_attn.qkv_proj_experts.0.weight",
    "self_attn.qkv_proj_experts.1.weight",
    "self_attn.o_proj_experts.0.weight",
    "self_attn.o_proj_experts.1.weight",
    "moe.experts.0.gate_up_proj.weight",
    "moe.experts.1.gate_up_proj.weight",
    "moe.experts.0.down_proj.weight",
    "moe.experts.1.down_proj.weight",
)
# No shape-forced INT8 anymore: expert-0 down_proj (K=11008, 11008%512!=0) is made
# int4-legal at load by padding INTER -> 11264 (adapters/wall_oss/llm.py). Kept as an
# empty tuple so the --keep-int8 retreat (extra_int8_suffixes) and config split still work.
WALL_OSS_FORCED_INT8_SUFFIXES = ()


def _scale_name(weight_name: str) -> str:
    return weight_name[: -len(".weight")] + ".weight_scale"


def _is_quant_proj(name: str) -> bool:
    return name.endswith(WALL_OSS_QUANT_SUFFIXES)


def _bits_for_name(name: str, extra_int8_suffixes: tuple[str, ...]) -> int:
    if name.endswith(WALL_OSS_FORCED_INT8_SUFFIXES) or name.endswith(extra_int8_suffixes):
        return 8
    return 4


def _checkpoint_layout(src: Path):
    index_path = src / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.load(index_path.open())
        weight_map = dict(index["weight_map"])
        return True, sorted(set(weight_map.values())), weight_map, index
    if (src / "model.safetensors").is_file():
        return False, ["model.safetensors"], {}, {}
    raise FileNotFoundError(f"no safetensors in {src}")


def _copy_sidecar_files(src: Path, dst: Path, shards: set[str]) -> None:
    skip = set(shards) | {
        "model.safetensors", "model.safetensors.index.json",
        "config.json", "rpu_quant_config.json",
    }
    for path in sorted(src.iterdir()):
        if path.name in skip or not path.is_file():
            continue
        shutil.copy2(path, dst / path.name)


def _convert_tensor(name, tensor, extra_int8_suffixes, group_size=0):
    if _is_quant_proj(name):
        if not tensor.is_floating_point():
            raise TypeError(
                f"cannot quantize non-float {name!r} dtype={tensor.dtype}; use the fp16 ckpt")
        bits = _bits_for_name(name, extra_int8_suffixes)
        # group_size>0 + bits==4 -> group-wise int4 (pgrp): scale is 2D [G, N]
        # (the runtime packs with the pgrp swizzle and the C++ detects the 2D scale).
        # Forced-int8 layers (e.g. expert-0 down_proj) stay per-channel int8.
        if group_size > 0 and bits == 4:
            w_q, scale, _ = quantize_int4_group_wise(tensor, group_size=group_size)
        else:
            w_q, scale = quantize_linear_per_channel(tensor, bits=bits)
        return {name: w_q, _scale_name(name): scale}, 1, 0
    out = tensor.to(torch.float16) if tensor.is_floating_point() else tensor
    return {name: out}, 0, 1


def _quant_config(src: Path, extra_int8_suffixes, group_size=0):
    int8 = list(WALL_OSS_FORCED_INT8_SUFFIXES) + [
        s for s in extra_int8_suffixes if s not in WALL_OSS_FORCED_INT8_SUFFIXES]
    int4 = [s for s in WALL_OSS_QUANT_SUFFIXES if s not in int8]
    gw = group_size > 0
    return {
        "method": "w4a16",
        "mode": "group_wise_symmetric" if gw else "per_channel_symmetric",
        "group_size": group_size if gw else 0,
        "qaxis": 0,
        "source": src.name,  # basename only; paired checkpoints are siblings
        "target": "wall_oss_decoder_experts",
        "storage": "int8",
        "value_bits": 4,
        "kernel": "wint4a16_pgrp" if gw else "wint4a16",
        "scale": "per_group_GxN" if gw else "per_output_channel",
        "int4_suffixes": int4,
        "int8_suffixes": int8,
        "note": ("int4 values in int8 container; runtime packs to uint8 [N,K/2]. "
                 + ("group-wise (pgrp/AWQ): scale is [G=K/group_size, N]; runtime uses the "
                    "pgrp swizzle + tiled parallel_linear_wint4a16_pgrp kernel (available "
                    "in the combined operator asset). " if gw else
                    "per-channel: scale [N], wINT4a16_per_channel kernel. ")
                 + "int8_suffixes (expert-0 down_proj, K=11008) keep the W8A16 path."),
    }


def convert_checkpoint(src, dst, *, extra_int8_suffixes=(), group_size=0):
    src, dst = Path(src).expanduser().resolve(), Path(dst).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"--src not a dir: {src}")
    if dst.exists():
        raise FileExistsError(f"--dst exists: {dst}")
    is_sharded, shards, weight_map, index = _checkpoint_layout(src)
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    stats = {"n_quantized": 0, "n_copied": 0, "n_shards": len(shards), "quant_proj": 0}
    new_weight_map = dict(weight_map) if is_sharded else {}
    try:
        tmp.mkdir(parents=True)
        for i, shard in enumerate(shards, 1):
            print(f"[convert_wall_oss] shard {i}/{len(shards)}: {shard}", flush=True)
            tensors = load_file(src / shard)
            out = {}
            for name in sorted(tensors):
                conv, nq, nc = _convert_tensor(name, tensors[name], extra_int8_suffixes, group_size)
                out.update(conv)
                stats["n_quantized"] += nq
                stats["n_copied"] += nc
                if nq:
                    stats["quant_proj"] += 1
                    new_weight_map[_scale_name(name)] = shard
            save_file(out, tmp / shard, metadata={"format": "pt"})
            del tensors, out
        if stats["quant_proj"] == 0:
            raise ValueError("no wall-oss decoder projection weights matched; wrong --src?")
        if is_sharded:
            new_index = dict(index)
            new_index["weight_map"] = dict(sorted(new_weight_map.items()))
            json.dump(new_index, (tmp / "model.safetensors.index.json").open("w"), indent=2)
        config = json.load((src / "config.json").open())
        json.dump(config, (tmp / "config.json").open("w"), indent=2)
        json.dump(_quant_config(src, tuple(extra_int8_suffixes), group_size),
                  (tmp / "rpu_quant_config.json").open("w"), indent=2)
        _copy_sidecar_files(src, tmp, set(shards))
        tmp.rename(dst)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="fp16 Wall-OSS-0.5 checkpoint dir")
    ap.add_argument("--dst", required=True, help="output w4a16 checkpoint dir")
    ap.add_argument("--keep-int8", default="",
                    help="extra full-suffix tokens to retreat to int8 if the cos gate fails, "
                         "e.g. 'self_attn.qkv_proj_experts.0.weight,self_attn.qkv_proj_experts.1.weight'")
    ap.add_argument("--group-size", type=int, default=0,
                    help="0 (default) = per-channel int4 (wINT4a16_per_channel). "
                         ">0 = group-wise int4 (pgrp/AWQ, scale [G=K/group_size, N]); "
                         "use 32 to target the parallel_linear_wint4a16_pgrp auto-tile kernels. "
                         "Must divide K of every int4 layer.")
    args = ap.parse_args(argv)
    extra = tuple(t.strip() for t in args.keep_int8.split(",") if t.strip())
    bad = [t for t in extra if t not in WALL_OSS_QUANT_SUFFIXES]
    if bad:
        print(f"[convert_wall_oss] error: --keep-int8 unknown {bad}; valid: {WALL_OSS_QUANT_SUFFIXES}",
              file=sys.stderr)
        return 1
    # The wall_oss w4a16 runtime loader does NOT read int8_suffixes — build_wall_oss_llm packs
    # every quant projection as int4. Recording a projection int8 here would be silently re-packed
    # as int4 at load → numeric corruption. Reject the (currently runtime-unsupported) retreat.
    # (The only int8 keep is the automatic expert-0 down_proj via RPU_WALL_OSS_EXPERT0_DOWN_INT4=0.)
    if extra:
        print(f"[convert_wall_oss] error: --keep-int8 {list(extra)} is not honored by the wall_oss "
              f"w4a16 runtime loader (it reads no int8_suffixes) → would be re-packed int4 at load "
              f"→ corruption. Drop --keep-int8.", file=sys.stderr)
        return 2
    try:
        stats = convert_checkpoint(args.src, args.dst, extra_int8_suffixes=extra,
                                   group_size=args.group_size)
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        print(f"[convert_wall_oss] error: {exc}", file=sys.stderr)
        return 1
    print(f"[convert_wall_oss] {args.src} -> {args.dst}")
    print(f"  quantized proj: {stats['quant_proj']}  copied tensors: {stats['n_copied']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
