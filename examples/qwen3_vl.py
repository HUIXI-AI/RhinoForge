#!/usr/bin/env python3
"""Run image-and-text greedy generation with Qwen3-VL."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "qwen3_vl_2b.toml"


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    request = config["request"]
    if not model.get("alias") and not model.get("checkpoint"):
        raise ValueError("[model] needs alias or checkpoint")
    image = request.get("image")
    images = request.get("images")
    has_image = isinstance(image, str) and bool(image)
    has_images = (
        isinstance(images, list)
        and bool(images)
        and all(isinstance(value, str) and value for value in images)
    )
    if has_image == has_images:
        raise ValueError(
            "[request] needs exactly one of string image or non-empty string images"
        )
    if not isinstance(request.get("prompt"), str):
        raise ValueError("[request].prompt must be a string")
    if not isinstance(request.get("max_new_tokens"), int) or request["max_new_tokens"] < 1:
        raise ValueError("[request].max_new_tokens must be a positive integer")
    controlled = config.get("controlled_evaluation")
    if controlled is not None and controlled != {
        "gate": "QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED",
        "enabled": True,
    }:
        raise ValueError(
            "[controlled_evaluation] must enable only the exact Qwen3-VL 32B gate"
        )
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

    controlled = config.get("controlled_evaluation")
    if controlled is not None:
        os.environ[controlled["gate"]] = "1"

    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from rpu_backend.api import RPUCache, RPUModelForConditionalGeneration
    from rpu_backend.model_registry import model_path

    checkpoint = model_config.get("checkpoint") or str(
        model_path(model_config["alias"])
    )

    request = config["request"]
    image_values = request.get("images") or [request["image"]]
    image_paths = [Path(value).expanduser() for value in image_values]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"image file(s) not found: {', '.join(missing)}")
    local_only = bool(model_config.get("local_files_only", True))
    processor = AutoProcessor.from_pretrained(
        checkpoint, local_files_only=local_only, trust_remote_code=False
    )
    images = [Image.open(path).convert("RGB") for path in image_paths]
    messages = [{
        "role": "user",
        "content": [
            *({"type": "image", "image": image} for image in images),
            {"type": "text", "text": request["prompt"]},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    max_new_tokens = request["max_new_tokens"]
    model = RPUModelForConditionalGeneration.from_pretrained(
        checkpoint,
        dtype=torch.float16,
        device="rpu",
        rpu_execution=config.get("rpu_execution"),
        local_files_only=local_only,
        trust_remote_code=False,
    )
    cache = RPUCache.from_model(
        model.model.language_model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
    )
    prefill = {
        name: inputs[name]
        for name in (
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
        )
        if name in inputs
    }

    generated = []
    with torch.no_grad():
        output = model(
            input_ids=input_ids.to("rpu"),
            past_key_values=cache,
            logits_to_keep=1,
            use_cache=True,
            return_dict=True,
            **prefill,
        )
        for step in range(max_new_tokens):
            next_id = output.logits[:, -1].argmax(dim=-1, keepdim=True).long()
            generated.append(next_id.cpu())
            if (
                processor.tokenizer.eos_token_id is not None
                and next_id.item() == processor.tokenizer.eos_token_id
            ):
                break
            if step + 1 == max_new_tokens:
                break
            output = model(
                input_ids=next_id.to("rpu"),
                past_key_values=cache,
                logits_to_keep=1,
                use_cache=True,
                return_dict=True,
            )
    print(
        processor.tokenizer.decode(
            torch.cat(generated, dim=1)[0], skip_special_tokens=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
