"""Validate a W8A16 Qwen3 checkpoint against its source checkpoint."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open


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

_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def _load_weight_map(root: Path) -> dict[str, str]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as f:
            return dict(json.load(f)["weight_map"])

    model_path = root / "model.safetensors"
    if model_path.is_file():
        with safe_open(model_path, framework="pt", device="cpu") as f:
            return {name: "model.safetensors" for name in f.keys()}

    raise FileNotFoundError(f"no safetensors checkpoint found in {root}")


def _load_tensor(root: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    try:
        shard = weight_map[name]
    except KeyError as exc:
        raise KeyError(f"missing tensor {name}") from exc
    with safe_open(root / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _scale_name(weight_name: str) -> str:
    return weight_name[: -len(".weight")] + ".weight_scale"


def _should_quantize(name: str, skip_modules: list[str]) -> bool:
    if not name.endswith(QUANT_PROJ_SUFFIXES):
        return False
    return not any(skip in name for skip in skip_modules)


def _layer_id(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def _sample_names(names: list[str], max_samples: int) -> list[str]:
    by_layer: dict[int, list[str]] = {}
    for name in names:
        layer = _layer_id(name)
        if layer is not None:
            by_layer.setdefault(layer, []).append(name)

    if by_layer:
        layers = sorted(by_layer)
        selected_layers = [layers[0], layers[len(layers) // 2], layers[-1]]
        selected: list[str] = []
        for layer in dict.fromkeys(selected_layers):
            selected.extend(sorted(by_layer[layer]))
        return selected[:max_samples]

    return names[:max_samples]


def verify_checkpoint(
    src: str | Path,
    dst: str | Path,
    skip_modules: list[str],
    max_samples: int,
    min_cosine: float,
    quant_lm_head: bool = False,
    quant_embed_tokens: bool = False,
) -> dict:
    src = Path(src).expanduser().resolve()
    dst = Path(dst).expanduser().resolve()
    src_map = _load_weight_map(src)
    dst_map = _load_weight_map(dst)

    with (dst / "config.json").open() as f:
        config = json.load(f)
    quant_config = config.get("quant_config", {})
    if quant_config.get("method") != "w8a16":
        raise ValueError(f"{dst}/config.json does not declare quant_config.method=w8a16")

    expected = sorted(name for name in src_map if _should_quantize(name, skip_modules))
    expected_pairs = [(name, name) for name in expected]
    lm_head_pair = None
    embed_pair = None
    if quant_lm_head:
        if LM_HEAD_WEIGHT_NAME in src_map:
            src_lm_head = LM_HEAD_WEIGHT_NAME
        elif EMBED_TOKENS_WEIGHT_NAME in src_map:
            src_lm_head = EMBED_TOKENS_WEIGHT_NAME
        else:
            raise KeyError(
                "cannot verify quantized lm_head: source checkpoint has neither "
                f"{LM_HEAD_WEIGHT_NAME} nor {EMBED_TOKENS_WEIGHT_NAME}"
            )
        lm_head_pair = (src_lm_head, LM_HEAD_WEIGHT_NAME)
        expected_pairs.append(lm_head_pair)
    if quant_embed_tokens:
        if EMBED_TOKENS_WEIGHT_NAME not in src_map:
            raise KeyError(
                "cannot verify quantized embed_tokens: source checkpoint has no "
                f"{EMBED_TOKENS_WEIGHT_NAME}"
            )
        embed_pair = (EMBED_TOKENS_WEIGHT_NAME, EMBED_TOKENS_WEIGHT_NAME)
        expected_pairs.append(embed_pair)

    missing = []
    for _, dst_name in expected_pairs:
        if dst_name not in dst_map:
            missing.append(dst_name)
        if _scale_name(dst_name) not in dst_map:
            missing.append(_scale_name(dst_name))
    if missing:
        raise KeyError(f"missing {len(missing)} expected W8A16 tensors, first={missing[0]}")

    sample_names = _sample_names(expected, max_samples)
    sample_pairs = [(name, name) for name in sample_names]
    if lm_head_pair is not None:
        sample_pairs.append(lm_head_pair)
    if embed_pair is not None:
        sample_pairs.append(embed_pair)
    worst = {
        "name": "",
        "cosine": 1.0,
        "rmse": 0.0,
        "max_abs": 0.0,
        "mean_abs": 0.0,
    }

    print(f"[verify_w8a16] source={src}")
    print(f"[verify_w8a16] w8a16 ={dst}")
    print(f"[verify_w8a16] expected quantized weights={len(expected_pairs)}")
    print("| tensor | cosine | rmse | max_abs | mean_abs |")
    print("|---|---:|---:|---:|---:|")

    for src_name, dst_name in sample_pairs:
        display_name = (
            dst_name if src_name == dst_name else f"{src_name} -> {dst_name}"
        )
        src_raw = _load_tensor(src, src_map, src_name)
        q_w = _load_tensor(dst, dst_map, dst_name)
        scale = _load_tensor(dst, dst_map, _scale_name(dst_name))
        if q_w.dtype != torch.int8:
            raise TypeError(f"{dst_name} dtype is {q_w.dtype}, expected int8")
        if scale.dtype != torch.float16:
            raise TypeError(
                f"{_scale_name(dst_name)} dtype is {scale.dtype}, expected float16"
            )

        if src_raw.dtype == torch.int8 and src_name == dst_name:
            src_scale = _load_tensor(src, src_map, _scale_name(src_name))
            if not torch.equal(src_raw, q_w):
                raise AssertionError(f"{dst_name} int8 payload changed while copying")
            if not torch.equal(src_scale.to(torch.float16), scale):
                raise AssertionError(f"{_scale_name(dst_name)} changed while copying")
            print(f"| {display_name} | 1.00000000 | 0 | 0 | 0 |")
            continue

        src_w = src_raw.to(torch.float16).to(torch.float32)

        deq = scale.to(torch.float32).unsqueeze(1) * q_w.to(torch.float32)
        diff = deq - src_w
        rmse = torch.sqrt(torch.mean(diff * diff)).item()
        max_abs = torch.max(torch.abs(diff)).item()
        mean_abs = torch.mean(torch.abs(diff)).item()
        dot = torch.sum(src_w * deq, dtype=torch.float64)
        src_norm = torch.sqrt(torch.sum(src_w * src_w, dtype=torch.float64))
        deq_norm = torch.sqrt(torch.sum(deq * deq, dtype=torch.float64))
        denom = src_norm * deq_norm
        cosine = (dot / denom).item() if denom.item() != 0.0 else 1.0
        cosine = max(-1.0, min(1.0, cosine))

        if not all(math.isfinite(v) for v in (cosine, rmse, max_abs, mean_abs)):
            raise FloatingPointError(
                f"non-finite validation metric for {display_name}"
            )
        if cosine < min_cosine:
            raise AssertionError(
                f"{display_name} cosine={cosine:.8f} below threshold {min_cosine:.8f}"
            )
        if cosine < worst["cosine"]:
            worst = {
                "name": display_name,
                "cosine": cosine,
                "rmse": rmse,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
            }

        print(f"| {display_name} | {cosine:.8f} | {rmse:.6g} | {max_abs:.6g} | {mean_abs:.6g} |")

    return {
        "expected_quantized": len(expected_pairs),
        "sampled": len(sample_pairs),
        "worst": worst,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="source HF checkpoint directory")
    parser.add_argument("--dst", required=True, help="W8A16 checkpoint directory")
    parser.add_argument("--skip-modules", nargs="+", default=["lm_head"])
    parser.add_argument("--max-samples", type=int, default=21)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument(
        "--quant-lm-head",
        action="store_true",
        help="Also verify lm_head.weight, using model.embed_tokens.weight as source if tied.",
    )
    parser.add_argument(
        "--quant-embed-tokens",
        action="store_true",
        help="Also verify model.embed_tokens.weight int8 quantization.",
    )
    args = parser.parse_args(argv)

    try:
        stats = verify_checkpoint(
            args.src,
            args.dst,
            args.skip_modules,
            args.max_samples,
            args.min_cosine,
            args.quant_lm_head,
            args.quant_embed_tokens,
        )
    except Exception as exc:
        print(f"[verify_w8a16] FAIL: {exc}", file=sys.stderr)
        return 1

    worst = stats["worst"]
    print(
        "[verify_w8a16] PASS "
        f"sampled={stats['sampled']} expected_quantized={stats['expected_quantized']} "
        f"worst_cosine={worst['cosine']:.8f} worst_tensor={worst['name']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
