#!/usr/bin/env python3
"""Run one LightNav RPU inference per image in an ordered episode JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

if __package__:
    from . import lightnav_benchmark as benchmark
else:
    import lightnav_benchmark as benchmark


def load_episode(input_path: Path, images_dir: Path | None = None):
    """Resolve image names relative to the JSON (or an explicit image directory)."""
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("episode JSON must be an object")
    instruction = data.get("prompt", data.get("instruction"))
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("episode must contain a nonempty 'prompt' or 'instruction'")
    images = data.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("episode must contain a nonempty ordered 'images' list")
    if data.get("task", "tracking") != "tracking":
        raise ValueError("this script supports task='tracking' only")
    root = images_dir if images_dir is not None else input_path.parent
    paths = []
    for index, name in enumerate(images):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"images[{index}] must be a nonempty file name")
        path = (root / name).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"images[{index}] does not exist: {path}")
        paths.append(path)
    return data, instruction, paths


def load_frames(paths, video_size):
    """Use upstream RGB conversion, bilinear resizing and [-1, 1] normalization."""
    import numpy as np
    from PIL import Image
    import torch
    from lightnav.inference.frame_preprocessing import rgb_frame_to_model_tensor

    frames = torch.empty((len(paths), 3, *video_size), dtype=torch.float32)
    for index, path in enumerate(paths):
        try:
            with Image.open(path) as image:
                rgb = np.array(image.convert("RGB"))
            frames[index].copy_(rgb_frame_to_model_tensor(rgb, video_size))
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot preprocess images[{index}] ({path}): {error}") from error
    return frames


def infer_episode(model, processor, data_processor, bundle, frames, instruction,
                  vision_cache, lm_cache, image_names, history_draft=False,
                  history_frames=9, prefill_graph_cache=None):
    """Slide a window of at most history_frames past frames plus the current one."""
    if type(history_frames) is not int or history_frames < 0:
        raise ValueError("history_frames must be a nonnegative integer")
    results = []
    draft = None
    for index, name in enumerate(image_names):
        frame_started = time.perf_counter()
        # Growing prompt shapes need fresh graphs; the full window safely rebinds inputs.
        if prefill_graph_cache is not None and index <= history_frames:
            prefill_graph_cache.clear()
        graph_clear_ms = (time.perf_counter() - frame_started) * 1000
        # Match the benchmark's history-plus-current window. Once full, advance
        # its start by one to drop the oldest frame. The benchmark offsets the
        # builder's local frame IDs by start, keeping cache keys episode-absolute.
        start = max(0, index - history_frames)
        row = benchmark._run_request(
            model, processor, data_processor, bundle, vision_cache,
            frames[: index + 1], index - start, start, instruction, lm_cache,
            draft_tokens=draft,
            prefetch_latest=False,
        )
        frame_total_ms = (time.perf_counter() - frame_started) * 1000
        tokens = row["token_ids"]
        result = {"frame_index": index, "image": name, "token_ids": tokens}
        result["timing_ms"] = {
            key.removesuffix("_ms"): value
            for key, value in row.items() if key.endswith("_ms")
        }
        result["timing_ms"].update(
            graph_clear=graph_clear_ms, frame_total=frame_total_ms,
        )
        for key in ("input_tokens", "cache_hits", "cache_misses", "draft_accepted",
                    "final_instruction", "model_prompt"):
            if key in row:
                result[key] = row[key]
        results.append(result)
        if history_draft:
            draft = tokens
        if "final_instruction" in row:
            print(f"Final instruction for frame {index}:\n{row['final_instruction']}", flush=True)
        print(
            f"[{index + 1}/{len(image_names)}] {name}: {tokens} "
            f"({frame_total_ms:.1f} ms)", flush=True,
        )
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="episode JSON")
    parser.add_argument("--images-dir", type=Path, help="image root; default: JSON directory")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lightnav-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--history-frames", type=int, default=9,
        help="maximum past frames, excluding the current frame (default: 9; 0: current only)",
    )
    parser.add_argument("--check-config", action="store_true", help="validate paths without loading Torch/RPU")
    parser.add_argument("--constrained-head", action="store_true", help="use the action-token allowlist")
    parser.add_argument("--history-draft", action="store_true", help="verify the previous output as a draft")
    args = parser.parse_args(argv)
    if args.history_frames < 0:
        parser.error("--history-frames must be nonnegative")
    input_path = args.input.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    source_dir = args.lightnav_source.expanduser().resolve() / "src"
    output_path = args.output.expanduser().resolve()
    try:
        data, instruction, paths = load_episode(
            input_path, args.images_dir.expanduser().resolve() if args.images_dir else None,
        )
        if not (checkpoint / "eval_config.json").is_file():
            raise ValueError("--checkpoint must contain eval_config.json")
        for relative in (
            "lightnav/inference/vit_cache.py",
            "lightnav/inference/frame_preprocessing.py",
        ):
            if not (source_dir / relative).is_file():
                raise ValueError(f"--lightnav-source is missing src/{relative}")
        if output_path in [input_path, *paths] or output_path.is_dir():
            raise ValueError("--output must be a file separate from the input JSON and images")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.check_config:
        print(
            f"configuration OK: frames={len(paths)}, inferences={len(paths)}, "
            f"history_frames={args.history_frames}, output_tokens=5"
        )
        return 0

    sys.path.insert(0, str(source_dir))
    if args.constrained_head:
        os.environ["RPU_LIGHTNAV_CONSTRAINED_HEAD"] = "1"

    import torch
    from lightnav.inference.model import (
        _build_data_processor, _make_bundle, _resolve_processing_params,
    )
    from lightnav.inference.vit_cache import VitTubeletCache
    from lightnav.processing import VLNQwen3VLProcessor
    from rpu_backend.api import RPUCache, RPUModelForConditionalGeneration

    def placeholder_positions(**kwargs):
        ids = kwargs["input_ids"]
        return {
            "position_ids": torch.zeros((3, *ids.shape), dtype=torch.long),
            "rope_deltas": None,
        }

    params = _resolve_processing_params(str(checkpoint), task_type="trackvla")
    processor = VLNQwen3VLProcessor.from_pretrained(
        checkpoint, padding_side="left", local_files_only=True,
    )
    data_processor = _build_data_processor(processor, params, placeholder_positions)
    bundle = _make_bundle(
        None, processor, data_processor, torch.device("cpu"), placeholder_positions, params,
    )
    # Decode every image before loading device weights, so corrupt inputs fail early.
    frames = load_frames(paths, tuple(bundle.video_size))
    model = RPUModelForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.float16, device="rpu",
        rpu_execution={"prefill": {"fast_replay": True}},
        local_files_only=True, trust_remote_code=False,
    )
    try:
        if args.constrained_head:
            from rpu_backend.adapters.qwen3_vl import set_lightnav_constrained_lm_head

            set_lightnav_constrained_lm_head(model)
        vision_cache = VitTubeletCache(max_entries=benchmark.VISION_CACHE_ENTRIES, store_device="cpu")
        lm_cache = RPUCache.from_model(
            model.model.language_model,
            max_seq_len=benchmark.MAX_PREFILL_TOKENS + benchmark.OUTPUT_TOKENS,
        )
        results = infer_episode(
            model, processor, data_processor, bundle, frames, instruction,
            vision_cache, lm_cache, data["images"], args.history_draft,
            history_frames=args.history_frames,
            prefill_graph_cache=model.model.language_model._rpu_text_graph_cache,
        )
        report = {
            "schema": 1,
            "episode": data.get("episode"),
            "instruction": instruction,
            "output_tokens": benchmark.OUTPUT_TOKENS,
            "history_frames": args.history_frames,
            "lm_head": "action_allowlist" if args.constrained_head else "full_vocabulary",
            "decode_mode": "verified_history_draft" if args.history_draft else "serial",
            "timing_notes": {
                "unit": "milliseconds",
                "frame_total": "Wall time including graph clearing, request processing and prompt decoding; excludes console printing.",
                "total": "Request wall time including preparation, vision and text generation; excludes graph clearing and prompt decoding.",
                "startup": "Model loading and initial decoding/resizing of all images are excluded from per-frame times.",
                "overlap": "Breakdowns are nested: prepare includes prepare_sample/prepare_processor; vision includes its substeps. Do not sum all fields.",
                "graphs": "Graph construction is included; growing-window text Graphs are cleared before full-window replay.",
            },
            "inferences": results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        from rpu_backend.adapters.qwen3_vl import _is_retirement_owner

        owner = getattr(model, "_qwen3_vl_retirement_owner", None)
        if not _is_retirement_owner(owner, model):
            raise RuntimeError("Qwen3-VL teardown lost its actual composite retirement owner")
        owner.close()
    print(f"Wrote {len(results)} inferences to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
