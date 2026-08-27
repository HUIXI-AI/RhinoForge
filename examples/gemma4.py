#!/usr/bin/env python3
"""Run Source-only Gemma4-E4B text generation on RPU."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "gemma4.toml"
ENV_KEYS = {"RPU_LOG_LEVEL", "RPU_WARMUP"}


def _runner_env(config: dict) -> dict:
    env = config.get("runner", {}).get("env", {})
    if not isinstance(env, dict):
        raise ValueError("[runner.env] must be a table")
    unknown = sorted(set(env) - ENV_KEYS)
    if unknown:
        raise ValueError(f"unsupported Gemma4 [runner.env] keys: {unknown}")
    return env


def _env_text(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (str, int)):
        return str(value)
    raise ValueError("[runner.env] values must be strings, integers, or booleans")


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    generation = config["generation"]
    if not model.get("alias") and not model.get("checkpoint"):
        raise ValueError("[model] needs alias or checkpoint")
    if model.get("source_only_acknowledged") is not True:
        raise ValueError("[model].source_only_acknowledged must be true")
    max_seq_len = model.get("max_seq_len", 512)
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len < 1:
        raise ValueError("[model].max_seq_len must be a positive integer")
    if not isinstance(generation.get("prompt"), str) or not generation["prompt"]:
        raise ValueError("[generation].prompt must be a non-empty string")
    max_new_tokens = generation.get("max_new_tokens")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("[generation].max_new_tokens must be a positive integer")
    _runner_env(config)
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
        print(f"configuration OK: {args.config} ({location}; Source-only)")
        return 0

    for name, value in _runner_env(config).items():
        os.environ[name] = _env_text(value)

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from rpu_backend.adapters.gemma4 import Gemma4Adapter
    from rpu_backend.model_registry import model_path

    checkpoint = model_config.get("checkpoint") or str(
        model_path(model_config["alias"])
    )
    local_only = bool(model_config.get("local_files_only", True))
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=local_only, trust_remote_code=False
    )
    input_ids = tokenizer(
        config["generation"]["prompt"], return_tensors="pt"
    ).input_ids
    max_new_tokens = config["generation"]["max_new_tokens"]
    max_seq_len = model_config.get("max_seq_len", 512)
    if input_ids.shape[1] + max_new_tokens > max_seq_len:
        raise SystemExit(
            "prompt plus max_new_tokens exceeds [model].max_seq_len"
        )

    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=local_only,
        trust_remote_code=False,
    ).eval()
    model._rpu_max_seq = max_seq_len
    model = Gemma4Adapter(
        model, rpu_execution=config.get("rpu_execution")
    ).to_rpu()
    cache = model.build_rpu_cache(max_seq_len=max_seq_len)

    generated = []
    current = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(
                input_ids=current,
                past_key_values=cache,
                logits_to_keep=1,
                use_cache=True,
                return_dict=True,
            )
            current = output.logits[:, -1].argmax(dim=-1, keepdim=True).long()
            generated.append(current.cpu())
            if tokenizer.eos_token_id is not None and current.item() == tokenizer.eos_token_id:
                break
    print(tokenizer.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
