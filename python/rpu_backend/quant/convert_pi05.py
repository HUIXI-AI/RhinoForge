"""Offline W8A16 / fake-W4 quantization for Pi0.5 Gemma projection weights.

The VLM Gemma decoder and action expert decoder projections are quantized.
SigLIP, AdaRMS dense, action projection, and processor sidecar tensors remain
fp16/original. Fake-W4 stores signed 4-bit values in int8 tensors and still
uses the existing W8A16 runtime kernel; it is a numerical probe only. The
output deliberately omits model_remapped.safetensors so the Pi05 loader
regenerates a remap from the quantized source tensors.
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
except ImportError:
    from _common import quantize_linear_per_channel


PI05_EXPERT_ANCHOR = ".paligemma_with_expert.gemma_expert.model.layers."
PI05_VLM_ANCHOR = ".paligemma_with_expert.paligemma.model.language_model.layers."
PI05_PROJ_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _scale_name(weight_name: str) -> str:
    return weight_name[: -len(".weight")] + ".weight_scale"


def _target_for_name(name: str) -> str | None:
    if name.endswith(PI05_PROJ_SUFFIXES):
        if PI05_EXPERT_ANCHOR in name:
            return "expert"
        if PI05_VLM_ANCHOR in name:
            return "vlm"
    return None


def _is_pi05_quant_proj(name: str) -> bool:
    return _target_for_name(name) is not None


def _bits_for_name(name: str, bits: int, int8_keep_suffixes: tuple[str, ...]) -> int:
    """Per-tensor bit width: keep listed proj suffixes at INT8 under fake-W4.

    Only meaningful when ``bits == 4`` (mixed-precision probe). Suffixes are the
    full ``<proj>.weight`` tokens, e.g. ``self_attn.k_proj.weight``.
    """
    if bits != 4 or not int8_keep_suffixes:
        return bits
    if name.endswith(tuple(int8_keep_suffixes)):
        return 8
    return bits


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


def _copy_sidecar_files(src: Path, dst: Path, shards: set[str]) -> None:
    skip = set(shards)
    skip.update({
        "model.safetensors",
        "model.safetensors.index.json",
        "model_remapped.safetensors",
        "config.json",
        "rpu_quant_config.json",
    })
    for path in sorted(src.iterdir()):
        if path.name in skip or not path.is_file():
            continue
        shutil.copy2(path, dst / path.name)


def _convert_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    bits: int,
    int8_keep_suffixes: tuple[str, ...] = (),
) -> tuple[dict[str, torch.Tensor], int, int, int]:
    bytes_in = tensor.numel() * tensor.element_size()
    if _is_pi05_quant_proj(name):
        eff_bits = _bits_for_name(name, bits, int8_keep_suffixes)
        if tensor.dtype == torch.int8:
            if eff_bits == 4:
                raise TypeError(
                    f"cannot fake-W4-quantize already-int8 tensor {name!r}; "
                    "use the raw fp16 Pi0.5 checkpoint as --src"
                )
            return {name: tensor}, 0, 1, bytes_in
        if not tensor.is_floating_point():
            raise TypeError(
                f"cannot quantize non-floating tensor {name!r} "
                f"with dtype {tensor.dtype}"
            )
        w_int8, scale = quantize_linear_per_channel(tensor, bits=eff_bits)
        bytes_out = (
            w_int8.numel() * w_int8.element_size()
            + scale.numel() * scale.element_size()
        )
        return {name: w_int8, _scale_name(name): scale}, 1, 0, bytes_out

    out_tensor = tensor.to(torch.float16) if tensor.is_floating_point() else tensor
    bytes_out = out_tensor.numel() * out_tensor.element_size()
    return {name: out_tensor}, 0, 1, bytes_out


# Short-form ".weight" suffixes matched via str.endswith (same convention as the
# --keep-int8 CLI). NOT full-path: "self_attn.k_proj.weight".endswith("k_proj.weight")
# is True, but a bare "k_proj" token would NOT match — always include ".weight".
_REAL_W4_REQUIRED_INT8_SUFFIXES = ("k_proj.weight", "v_proj.weight")


def _normalize_keep_int8_suffixes(
    int8_keep_suffixes: tuple[str, ...],
    *,
    real_w4: bool = False,
) -> tuple[str, ...]:
    keep = list(int8_keep_suffixes)
    if real_w4:
        keep.extend(_REAL_W4_REQUIRED_INT8_SUFFIXES)
    # Preserve order while deduplicating.
    return tuple(dict.fromkeys(keep))


def _method_for_stats(*, bits: int, real_w4: bool = False) -> str:
    if real_w4 and bits != 4:
        raise ValueError("real_w4=True requires bits=4")
    if bits == 4:
        return "w4a16" if real_w4 else "w4a16_fake_int8"
    return "w8a16"


def _quant_config(*, bits: int, int8_keep_suffixes: tuple[str, ...] = (),
                  real_w4: bool = False) -> dict:
    int8_keep_suffixes = _normalize_keep_int8_suffixes(
        int8_keep_suffixes, real_w4=real_w4)
    if bits == 4:
        method = "w4a16" if real_w4 else "w4a16_fake_int8"
        kernel = "wint4a16" if real_w4 else "w8a16"
        note = (
            "Real W4: int4 values stored in int8 on disk; runtime packs to uint8 "
            "[N,K/2] and dispatches to the wINT4 kernel. mixed_int8_suffixes stay INT8."
        ) if real_w4 else (
            "Fake W4 numerical probe: signed int4 values stored in int8; "
            "runtime intentionally uses existing W8A16 kernel. "
            "mixed_int8_suffixes (if any) are kept at full INT8."
        )
        return {
            "method": method,
            "mode": "per_channel_symmetric",
            "qaxis": 0,
            "target": "pi05_gemma_vlm_and_expert",
            "quantized_projection_suffixes": list(PI05_PROJ_SUFFIXES),
            "storage": "int8",
            "value_bits": 4,
            "mixed_int8_suffixes": list(int8_keep_suffixes),
            "scale": "per_output_channel",
            "kernel": kernel,
            "note": note,
        }
    if real_w4:
        raise ValueError("real_w4=True requires bits=4")
    if bits == 8:
        return {
            "method": "w8a16",
            "mode": "per_channel_symmetric",
            "qaxis": 0,
            "target": "pi05_gemma_vlm_and_expert",
            "quantized_projection_suffixes": list(PI05_PROJ_SUFFIXES),
            "storage": "int8",
            "value_bits": 8,
            "scale": "per_output_channel",
            "kernel": "w8a16",
        }
    raise ValueError(f"expected bits to be 4 or 8, got {bits}")


def convert_checkpoint(
    src: str | Path,
    dst: str | Path,
    *,
    bits: int = 8,
    int8_keep_suffixes: tuple[str, ...] = (),
    real_w4: bool = False,
) -> dict:
    src = Path(src).expanduser().resolve()
    dst = Path(dst).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"--src is not a directory: {src}")
    if dst.exists():
        raise FileExistsError(f"--dst already exists, refusing to overwrite: {dst}")
    if bits not in (4, 8):
        raise ValueError(f"expected bits to be 4 or 8, got {bits}")
    if int8_keep_suffixes and bits != 4:
        raise ValueError("int8_keep_suffixes is only meaningful with bits=4 (fake-W4)")
    if real_w4 and bits != 4:
        raise ValueError("real_w4=True requires bits=4")
    int8_keep_suffixes = _normalize_keep_int8_suffixes(
        int8_keep_suffixes, real_w4=real_w4)

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
        "expert_projection_weights": 0,
        "vlm_projection_weights": 0,
        "value_bits": bits,
        "method": _method_for_stats(bits=bits, real_w4=real_w4),
        "mixed_int8_suffixes": list(int8_keep_suffixes),
    }
    new_weight_map = dict(weight_map) if is_sharded else {}

    try:
        tmp.mkdir(parents=True)
        for shard_idx, shard in enumerate(shards, start=1):
            print(f"[convert_pi05] shard {shard_idx}/{len(shards)}: {shard}", flush=True)
            tensors = load_file(src / shard)
            out: dict[str, torch.Tensor] = {}

            for name in sorted(tensors):
                target = _target_for_name(name)
                converted, n_quant, n_copy, bytes_out = _convert_tensor(
                    name, tensors[name], bits=bits,
                    int8_keep_suffixes=int8_keep_suffixes,
                )
                out.update(converted)
                stats["n_quantized"] += n_quant
                stats["n_copied"] += n_copy
                stats["bytes_in"] += tensors[name].numel() * tensors[name].element_size()
                stats["bytes_out"] += bytes_out
                if target == "expert":
                    stats["expert_projection_weights"] += 1
                elif target == "vlm":
                    stats["vlm_projection_weights"] += 1
                if n_quant:
                    new_weight_map[_scale_name(name)] = shard

            save_file(out, tmp / shard, metadata={"format": "pt"})
            del tensors
            del out

        if stats["expert_projection_weights"] == 0 or stats["vlm_projection_weights"] == 0:
            raise ValueError(
                "missing Pi05 projection weights: "
                f"expert={stats['expert_projection_weights']} "
                f"vlm={stats['vlm_projection_weights']}"
            )

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
        quant_config = _quant_config(bits=bits, int8_keep_suffixes=int8_keep_suffixes,
                                     real_w4=real_w4)
        with (tmp / "config.json").open("w") as f:
            json.dump(config, f, indent=2)
        with (tmp / "rpu_quant_config.json").open("w") as f:
            json.dump(quant_config, f, indent=2)

        _copy_sidecar_files(src, tmp, set(shards))
        tmp.rename(dst)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="source Pi0.5 checkpoint directory")
    parser.add_argument("--dst", required=True, help="output quantized checkpoint directory")
    parser.add_argument(
        "--fake-w4",
        action="store_true",
        help=(
            "store signed 4-bit values in int8 tensors and mark the checkpoint "
            "as method=w4a16_fake_int8; runtime still uses the W8A16 kernel"
        ),
    )
    parser.add_argument(
        "--keep-int8",
        default="",
        help=(
            "comma-separated projection tokens to keep at full INT8 under "
            "--fake-w4 (mixed precision), e.g. 'k_proj,v_proj'. The rest stay INT4."
        ),
    )
    parser.add_argument(
        "--real-w4",
        action="store_true",
        help=(
            "tag the checkpoint method=w4a16 (real packed INT4): runtime packs to "
            "uint8 [N,K/2] and uses the wINT4 kernel. Requires --fake-w4 quantization "
            "(same int4-in-int8 on-disk storage). k_proj/v_proj are automatically "
            "kept at INT8. Without this, method stays w4a16_fake_int8 (W8A16 kernel)."
        ),
    )
    args = parser.parse_args(argv)
    bits = 4 if args.fake_w4 else 8

    if args.real_w4 and bits != 4:
        print("[convert_pi05] error: --real-w4 requires --fake-w4 (int4 quantization)",
              file=sys.stderr)
        return 1

    valid_tokens = {s.rsplit(".", 2)[-2] for s in PI05_PROJ_SUFFIXES}  # {q_proj, k_proj, ...}
    keep_tokens = [t.strip() for t in args.keep_int8.split(",") if t.strip()]
    bad = [t for t in keep_tokens if t not in valid_tokens]
    if bad:
        print(
            f"[convert_pi05] error: --keep-int8 unknown token(s) {bad}; "
            f"valid: {sorted(valid_tokens)}",
            file=sys.stderr,
        )
        return 1
    int8_keep_suffixes = tuple(f"{t}.weight" for t in keep_tokens)

    try:
        stats = convert_checkpoint(
            args.src, args.dst, bits=bits, int8_keep_suffixes=int8_keep_suffixes,
            real_w4=args.real_w4,
        )
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        print(f"[convert_pi05] error: {exc}", file=sys.stderr)
        return 1

    print(f"[convert_pi05] {Path(args.src).resolve()}")
    print(f"           -> {Path(args.dst).resolve()}")
    print(f"  shards     : {stats['n_shards']}")
    print(f"  vlm        : {stats['vlm_projection_weights']} projection weights")
    print(f"  expert     : {stats['expert_projection_weights']} projection weights")
    print(f"  method     : {stats['method']} (value_bits={stats['value_bits']})")
    effective_keep = stats["mixed_int8_suffixes"]
    if effective_keep:
        print(f"  keep-int8  : {effective_keep} (INT8; rest INT4)")
    print(f"  quantized  : {stats['n_quantized']} weights")
    print(f"  copied     : {stats['n_copied']} tensors")
    print(
        f"  size       : {stats['bytes_in'] / 1e9:.2f} GB"
        f" -> {stats['bytes_out'] / 1e9:.2f} GB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
