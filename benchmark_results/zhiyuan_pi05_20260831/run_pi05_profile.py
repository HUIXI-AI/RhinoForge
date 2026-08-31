#!/usr/bin/env python3
"""Calibrate and benchmark the frozen Zhiyuan Pi0.5 profile."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file


IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
NOISE_SEED = 20260831


def make_batch(samples, index: int, device: str) -> dict[str, torch.Tensor]:
    batch = {
        f"observation.images.{key}": torch.from_numpy(
            samples["images"][index : index + 1, image_index]
        ).permute(0, 3, 1, 2).float().div_(255).to(device)
        for image_index, key in enumerate(IMAGE_KEYS)
    }
    batch["observation.state"] = torch.from_numpy(
        samples["state"][index : index + 1]
    ).to(device)
    batch["observation.language.tokens"] = torch.from_numpy(
        samples["tokens"][index : index + 1, :32]
    ).long().to(device)
    batch["observation.language.attention_mask"] = torch.from_numpy(
        samples["token_masks"][index : index + 1, :32]
    ).bool().to(device)
    return batch


def fixed_noise(index: int, device: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(NOISE_SEED + index)
    return torch.randn((1, 50, 32), generator=generator, dtype=torch.float32).to(device)


def target_linears(policy) -> dict[str, nn.Linear]:
    from rpu_backend.adapters.pi05.loader import _pi05_w8a16_target_for_name

    return {
        f"{name}.weight": module
        for name, module in policy._lerobot_policy.named_modules()
        if isinstance(module, nn.Linear)
        and _pi05_w8a16_target_for_name(f"{name}.weight") is not None
    }


def load_policy(checkpoint: Path, device: str):
    from rpu_backend.api import Pi05Policy

    execution = {"vlm_chunk_size": 400}
    if "rpu_execution" in inspect.signature(Pi05Policy.from_pretrained).parameters:
        execution = {
            "rpu_execution": {
                "prefill": {"chunk_size": 400, "padding_budget": 64},
                "vision": {"chunk_size": "auto"},
                "action": {"chunk_size": "auto"},
            }
        }
    policy = Pi05Policy.from_pretrained(
        str(checkpoint),
        dtype=torch.float16,
        trust_remote_code=False,
        **execution,
    )
    return policy.to(device)


def calibrate(args) -> None:
    samples = np.load(args.samples)
    policy = load_policy(args.checkpoint, args.device)
    modules = target_linears(policy)
    if len(modules) != 451:
        raise RuntimeError(f"expected 451 quantized linears, found {len(modules)}")

    sums = {
        name: torch.zeros(module.in_features, dtype=torch.float32, device=args.device)
        for name, module in modules.items()
    }
    counts = dict.fromkeys(modules, 0)
    hooks = []
    for name, module in modules.items():
        def observe(_module, inputs, _name=name):
            value = inputs[0].detach().float()
            sums[_name].add_(value.square().sum(dim=tuple(range(value.ndim - 1))))
            counts[_name] += value.numel() // value.shape[-1]
        hooks.append(module.register_forward_pre_hook(observe))

    started = time.perf_counter()
    with torch.no_grad():
        for index in range(len(samples["images"])):
            policy._lerobot_policy.predict_action_chunk(
                make_batch(samples, index, args.device),
                noise=fixed_noise(index, args.device),
                num_steps=10,
            )
            print(f"[calibrate] {index + 1}/{len(samples['images'])}", flush=True)
    for hook in hooks:
        hook.remove()

    stats = {name: (sums[name] / counts[name]).cpu() for name in sorted(sums)}
    save_file(stats, args.output, metadata={
        "algorithm": "diag_input_second_moment_v1",
        "samples": str(len(samples["images"])),
        "noise_seed": str(NOISE_SEED),
    })
    print(json.dumps({
        "output": str(args.output),
        "target_linears": len(stats),
        "samples": len(samples["images"]),
        "seconds": time.perf_counter() - started,
        "min_observations": min(counts.values()),
        "max_observations": max(counts.values()),
    }, indent=2))


def int8_native_forward(module, value):
    shape = value.shape
    output = torch.ops.aten._weight_int8pack_mm(
        value.reshape(-1, shape[-1]), module.weight, module.weight_scale
    ).reshape(*shape[:-1], module.out_features)
    return output if module.bias is None else output + module.bias


def configure_w8_cuda(policy, execution: str) -> int:
    modules = target_linears(policy)
    for module in modules.values():
        if module.weight.dtype != torch.int8 or not hasattr(module, "weight_scale"):
            raise RuntimeError("W8 checkpoint did not install int8 weight and scale")
        for key, hook in list(module._forward_pre_hooks.items()):
            if getattr(hook, "__name__", "") == "_input_dtype_hook":
                del module._forward_pre_hooks[key]
        if execution == "native":
            module.forward = types.MethodType(int8_native_forward, module)
        elif execution == "torchao":
            import sys

            torchao_runtime = Path(__file__).with_name("torchao_runtime")
            if torchao_runtime.is_dir():
                sys.path.insert(0, str(torchao_runtime))
            from torchao.quantization import Int8Tensor

            overflow_guard = 16.0
            scale = module.weight_scale.reshape(-1, 1) * overflow_guard
            weight = Int8Tensor(
                module.weight,
                scale,
                [1, module.in_features],
                torch.float16,
                zero_point=torch.zeros_like(scale, dtype=torch.int8),
                act_pre_scale=torch.tensor(
                    1.0 / overflow_guard, device=scale.device, dtype=torch.float16
                ),
            )
            module.weight = nn.Parameter(weight, requires_grad=False)
            module._buffers.pop("weight_scale")
        else:
            weight = module.weight.float().mul(module.weight_scale.float().unsqueeze(1)).half()
            module.weight = nn.Parameter(weight, requires_grad=False)
            module._buffers.pop("weight_scale")
    return len(modules)


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run(args) -> None:
    samples = np.load(args.samples)
    policy = load_policy(args.checkpoint, args.device)
    quantized_linears = 0
    if args.precision == "w8a16" and args.device == "cuda":
        quantized_linears = configure_w8_cuda(policy, args.w8_execution)

    graph_report = None
    if args.device == "rpu":
        graph_report = policy.prepare_graphs(
            make_batch(samples, 0, "cpu"), num_steps=args.num_steps
        )
    else:
        for index in range(3):
            policy._lerobot_policy.predict_action_chunk(
                make_batch(samples, index, args.device),
                noise=fixed_noise(index, args.device),
                num_steps=args.num_steps,
            )
        synchronize(args.device)

    outputs = []
    latencies = []
    for index in range(len(samples["images"])):
        batch_device = "cpu" if args.device == "rpu" else args.device
        noise_device = "cpu" if args.device == "rpu" else args.device
        batch = make_batch(samples, index, batch_device)
        noise = fixed_noise(index, noise_device)
        synchronize(args.device)
        started = time.perf_counter()
        action = policy._lerobot_policy.predict_action_chunk(
            batch,
            noise=noise,
            num_steps=args.num_steps,
        )
        synchronize(args.device)
        latencies.append((time.perf_counter() - started) * 1000)
        outputs.append(action.detach().float().cpu().numpy())
        print(f"[run] {index + 1}/{len(samples['images'])} {latencies[-1]:.3f} ms", flush=True)

    output = np.concatenate(outputs)
    if output.shape != (len(samples["images"]), 50, 32):
        raise RuntimeError(f"unexpected output shape {output.shape}")
    if not np.isfinite(output).all():
        raise RuntimeError("model output contains NaN or Inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, actions=output, latencies_ms=np.asarray(latencies))
    report = {
        "device": args.device,
        "precision": args.precision,
        "w8_execution": args.w8_execution if args.precision == "w8a16" else None,
        "quantized_linears": quantized_linears,
        "samples": len(samples["images"]),
        "noise_seed": NOISE_SEED,
        "denoise_steps": args.num_steps,
        "latency_ms": {
            "min": min(latencies),
            "median": statistics.median(latencies),
            "mean": statistics.fmean(latencies),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "graph_prepare": graph_report,
        "torch": torch.__version__,
        "environment": {
            "RPU_PI05_SIGLIP_W8A16": os.environ.get("RPU_PI05_SIGLIP_W8A16"),
            "RPU_PI05_ADARMS_DENSE_W8A16": os.environ.get("RPU_PI05_ADARMS_DENSE_W8A16"),
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--checkpoint", type=Path, required=True)
    calibration.add_argument("--samples", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--device", choices=("cuda",), default="cuda")
    calibration.set_defaults(func=calibrate)

    benchmark = subparsers.add_parser("run")
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--samples", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", choices=("cuda", "rpu"), required=True)
    benchmark.add_argument("--precision", choices=("w16a16", "w8a16"), required=True)
    benchmark.add_argument(
        "--w8-execution", choices=("native", "torchao", "dequant"), default="native"
    )
    benchmark.add_argument("--num-steps", type=int, choices=(5, 10), default=10)
    benchmark.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
