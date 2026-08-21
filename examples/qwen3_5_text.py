#!/usr/bin/env python3
"""Run text-only greedy decoding with a supported dense Qwen3.5 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "qwen3_5_0_8b.toml"


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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter
    from rpu_backend.api import Qwen3_5Cache
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
    max_seq_len = input_ids.shape[1] + max_new_tokens
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=local_only,
        trust_remote_code=False,
    ).eval()
    model = Qwen3_5Adapter(
        model, rpu_execution=config.get("rpu_execution")
    ).to_rpu(max_seq_len=max_seq_len)
    text_config = getattr(model.config, "text_config", model.config)
    cache = Qwen3_5Cache.from_config(text_config, max_seq_len=max_seq_len)

    generated = []
    current = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(
                input_ids=current.to("rpu"),
                past_key_values=cache,
                logits_to_keep=1,
                use_cache=True,
                return_dict=True,
            )
            current = output.logits[:, -1].argmax(dim=-1, keepdim=True).long()
            generated.append(current.cpu())
            if tokenizer.eos_token_id is not None and current.item() == tokenizer.eos_token_id:
                break
    print(
        tokenizer.decode(
            torch.cat(generated, dim=1)[0], skip_special_tokens=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
