"""CPU-only routed-expert RTN/G128 conversion of official floating 35B-A3B.

Reuses the runtime's existing RTN recipe. This produces a new source-bound
checkpoint; it does not claim byte identity with historical supplied assets.
The shard loop and fused expert split follow derive_moe_rtn_task.py used for
the 2026-10-05 model-zoo RTN derivation; task/NAS/resume machinery is omitted.
Run this file directly with TORCH_DEVICE_BACKEND_AUTOLOAD=0 on a CPU host.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3_5MoeConfig

if __package__:
    from . import qwen3_5_moe_rtn as recipe_module
    from . import precision_recipes
    from rpu_backend.adapters.qwen3_5_moe.quant_scope import (
        RTN_RECIPE, TEXT_PREFIX, validate_exact_profile,
    )
else:
    # Import only CPU leaves, without initializing rpu_backend or its native SO.
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "_offline_moe_quant", root / "__init__.py",
        submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from _offline_moe_quant import qwen3_5_moe_rtn as recipe_module
    from _offline_moe_quant import precision_recipes
    spec = importlib.util.spec_from_file_location(
        "_offline_moe_scope", root.parent / "adapters/qwen3_5_moe/quant_scope.py")
    scope = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scope
    spec.loader.exec_module(scope)
    RTN_RECIPE, TEXT_PREFIX = scope.RTN_RECIPE, scope.TEXT_PREFIX
    validate_exact_profile = scope.validate_exact_profile


# The one registered precision recipe this converter produces.
FAMILY, RECIPE = "qwen3_5", "w4a16_experts_rtn_g128"


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _convert_experts(name, tensor, *, experts, hidden, intermediate):
    gate_up = name.endswith(".gate_up_proj")
    expected = ((experts, 2 * intermediate, hidden) if gate_up
                else (experts, hidden, intermediate))
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name}: expected {expected}, got {tuple(tensor.shape)}")
    prefix = name.rsplit(".", 1)[0]
    result = {}
    for expert in range(experts):
        matrices = (("gate_proj", tensor[expert, :intermediate]),
                    ("up_proj", tensor[expert, intermediate:])) if gate_up else (
                        ("down_proj", tensor[expert]),)
        for role, weight in matrices:
            quartet = recipe_module.quantize_rtn_group128(weight)
            result.update({f"{prefix}.{expert}.{role}.{suffix}": value
                           for suffix, value in quartet.items()})
    return result


def convert_checkpoint(src: str | Path, dst: str | Path) -> dict:
    src, dst = Path(src).expanduser().resolve(), Path(dst).expanduser().resolve()
    if dst.exists():
        raise FileExistsError(f"output already exists: {dst}")
    if src == dst or src in dst.parents:
        raise ValueError("output must be outside the source checkpoint")
    if not (src / "generation_config.json").is_file():
        raise FileNotFoundError("source needs generation_config.json for the public loader")
    source_stats, source_records = {}, {}

    def signature(path):
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino

    def record_source(path):
        before = signature(path)
        digest = _sha(path)
        if signature(path) != before:
            raise ValueError(f"source changed during hashing: {path.name}")
        source_stats[path.name] = before
        source_records[path.name] = {"bytes": before[0], "sha256": digest}

    record_source(src / "config.json")
    config = json.loads((src / "config.json").read_text())
    for owner in (config, config.get("text_config", {}), config.get("vision_config", {})):
        if any(key in owner for key in ("quant_config", "quantization_config", "rtn_metadata")):
            raise ValueError("source must be unquantized floating 35B-A3B")
    if config.get("model_type") != "qwen3_5_moe":
        raise ValueError("source must be unquantized floating 35B-A3B")
    # Official raw JSON omits some HF defaults (for example attn_output_gate).
    # Resolve them through the model's real config class before exact admission.
    validate_exact_profile(Qwen3_5MoeConfig.from_dict(config), require_quant=False)
    tc = config["text_config"]
    layers, experts = tc["num_hidden_layers"], tc["num_experts"]
    hidden, intermediate = tc["hidden_size"], tc["moe_intermediate_size"]
    expected_experts = {f"{TEXT_PREFIX}{layer}.mlp.experts.{role}"
                        for layer in range(layers)
                        for role in ("gate_up_proj", "down_proj")}
    index_path = src / "model.safetensors.index.json"
    if index_path.is_file():
        record_source(index_path)
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as file:
            weight_map = {key: "model.safetensors" for key in file.keys()}
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("source needs a nonempty safetensors weight_map")
    if not expected_experts <= set(weight_map):
        raise ValueError("source lacks the complete fused routed-expert inventory")
    for name in weight_map:
        if (name.startswith(TEXT_PREFIX) and ".mlp.experts." in name
                and name not in expected_experts):
            raise ValueError(f"unsupported routed-expert source key: {name}")
    shards = sorted(set(weight_map.values()))
    for filename in shards:
        if (not isinstance(filename, str) or Path(filename).name != filename
                or not filename.endswith(".safetensors")):
            raise ValueError("source shard names must be safetensors basenames")
    tmp = dst.with_name(f".{dst.name}.partial-{os.getpid()}")
    if tmp.exists():
        raise FileExistsError(f"staging output already exists: {tmp}")
    manifest = {"completed": False, "recipe": RTN_RECIPE,
                "quantizer_sha256": _sha(Path(recipe_module.__file__)),
                "converter_sha256": _sha(Path(__file__)),
                "source_directory": str(src), "source_files": source_records, "output_files": {},
                "quantized_matrices": layers * experts * 3, "zero_stored_scales": 0}
    output_map, total_size = {}, 0
    try:
        tmp.mkdir(parents=True)
        for filename in shards:
            path = src / filename
            record_source(path)
            out = {}
            with safe_open(str(path), framework="pt", device="cpu") as file:
                if set(file.keys()) != {key for key, value in weight_map.items() if value == filename}:
                    raise ValueError(f"source index/shard inventory mismatch: {filename}")
                for name in file.keys():
                    tensor = file.get_tensor(name)
                    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                        raise ValueError(f"source tensor must be F16/BF16/F32: {name}")
                    if name in expected_experts:
                        converted = _convert_experts(name, tensor, experts=experts,
                                                     hidden=hidden, intermediate=intermediate)
                        manifest["zero_stored_scales"] += sum(
                            int((value == 0).sum()) for key, value in converted.items()
                            if key.endswith(".scales"))
                        out.update(converted)
                    else:
                        if name.endswith(("_scale", ".qweight", ".scales", ".qzeros", ".g_idx")):
                            raise ValueError(f"unexpected quantization key: {name}")
                        if not bool(torch.isfinite(tensor).all()):
                            raise ValueError(f"source tensor contains nonfinite values: {name}")
                        out[name] = tensor  # Preserve non-routed source bytes, including MTP.
            save_file(out, tmp / filename, metadata={"format": "pt", "quantization_algorithm": "RTN"})
            total_size += sum(value.numel() * value.element_size() for value in out.values())
            output_map.update({key: filename for key in out})
            del out
            print(f"converted {filename}", flush=True)
        config["rtn_metadata"] = {"algorithm": "RTN", "manifest": "derivation_manifest.json", "calibration": False}
        config["quantization_config"] = {"bits": 4, "group_size": 128, "sym": True,
            "desc_act": False, "quant_method": "gptq", "checkpoint_format": "gptq_v1"}
        validate_exact_profile(Qwen3_5MoeConfig.from_dict(config))
        _json(tmp / "config.json", config)
        _json(tmp / "model.safetensors.index.json", {"metadata": {"total_size": total_size}, "weight_map": output_map})
        skip = set(shards) | {"config.json", "model.safetensors.index.json", "derivation_manifest.json",
                              precision_recipes.RECORD}
        for path in sorted(src.iterdir()):
            if path.is_file() and path.name not in skip and not path.name.endswith((".safetensors", ".bin")):
                record_source(path)
                shutil.copy2(path, tmp / path.name)
        for filename, before in source_stats.items():
            if signature(src / filename) != before:
                raise ValueError(f"source changed during conversion: {filename}")
        precision_recipes.write_record(tmp, FAMILY, RECIPE)
        for path in sorted(tmp.iterdir()):
            manifest["output_files"][path.name] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
        manifest["completed"] = True
        _json(tmp / "derivation_manifest.json", manifest)
        tmp.rename(dst)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="official unquantized floating 35B-A3B directory")
    parser.add_argument("--dst", required=True, help="new output directory; existing paths are rejected")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads (default: 8)")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    convert_checkpoint(args.src, args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
