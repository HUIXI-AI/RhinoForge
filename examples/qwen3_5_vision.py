#!/usr/bin/env python3
"""Run single-image Qwen3.5 Vision inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "qwen3_5_vision_2b.toml"
def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    request = config["request"]
    if not model.get("alias") and not model.get("checkpoint"):
        raise ValueError("[model] needs alias or checkpoint")
    if not isinstance(request.get("image"), str) or not request["image"]:
        raise ValueError("[request].image must be a non-empty string")
    if not isinstance(request.get("prompt"), str):
        raise ValueError("[request].prompt must be a string")
    max_new_tokens = request.get("max_new_tokens")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise ValueError("[request].max_new_tokens must be a positive integer")
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
    if os.environ.get("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED") != "1":
        raise SystemExit(
            "Qwen3.5 Vision is Experimental/numeric-blocked and disabled by "
            "default. For controlled evaluation only, set exact "
            "QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1 before running this example."
        )

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter
    from rpu_backend.api import Qwen3_5Cache
    from rpu_backend.model_registry import model_path

    checkpoint = model_config.get("checkpoint") or str(
        model_path(model_config["alias"])
    )
    image_path = Path(config["request"]["image"]).expanduser()
    if not image_path.is_file():
        raise SystemExit(f"image file not found: {image_path}")

    local_only = bool(model_config.get("local_files_only", True))
    processor = AutoProcessor.from_pretrained(
        checkpoint, local_files_only=local_only, trust_remote_code=False
    )
    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": config["request"]["prompt"]},
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
    max_new_tokens = config["request"]["max_new_tokens"]
    max_seq_len = input_ids.shape[1] + max_new_tokens
    model = AutoModelForImageTextToText.from_pretrained(
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
    if "pixel_values" in prefill:
        prefill["pixel_values"] = prefill["pixel_values"].to("rpu")

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
