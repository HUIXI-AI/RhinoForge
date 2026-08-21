#!/usr/bin/env python3
"""Run greedy text generation with a supported Qwen3 or Llama checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "qwen3_0_6b.toml"


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    generation = config["generation"]
    if not model.get("alias") and not model.get("checkpoint"):
        raise ValueError("[model] needs alias or checkpoint")
    if not isinstance(generation.get("prompt"), str):
        raise ValueError("[generation].prompt must be a string")
    if not isinstance(generation.get("max_new_tokens"), int) or generation["max_new_tokens"] < 1:
        raise ValueError("[generation].max_new_tokens must be a positive integer")
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)

    model_config = config["model"]
    if args.check_config:
        location = model_config.get("checkpoint") or model_config["alias"]
        print(f"configuration OK: {args.config} ({location})")
        return 0

    import torch
    from transformers import AutoTokenizer
    from rpu_backend import RPUCache, RPUModelForCausalLM
    from rpu_backend.model_registry import model_path

    checkpoint = model_config.get("checkpoint") or str(
        model_path(model_config["alias"])
    )

    local_only = bool(model_config.get("local_files_only", True))
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=local_only, trust_remote_code=False
    )
    model = RPUModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float16,
        device="rpu",
        rpu_execution=config.get("rpu_execution"),
        local_files_only=local_only,
        trust_remote_code=False,
    )
    inputs = tokenizer(config["generation"]["prompt"], return_tensors="pt")
    max_new_tokens = config["generation"]["max_new_tokens"]
    cache = RPUCache.from_model(
        model,
        input_ids=inputs.input_ids,
        max_new_tokens=max_new_tokens,
    )
    rpu_inputs = {name: value.to("rpu") for name, value in inputs.items()}
    outputs = model.generate(
        **rpu_inputs,
        past_key_values=cache,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    completion = outputs[0, inputs.input_ids.shape[1] :].cpu()
    print(tokenizer.decode(completion, skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
