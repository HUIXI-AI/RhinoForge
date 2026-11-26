#!/usr/bin/env python3
"""Evaluate a LightNav checkpoint against customer PTQ episodes on RPU."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys


HISTORY_FRAMES = 9
OUTPUT_TOKENS = 5
TOKEN_PATTERN = re.compile(r"<[^>]+>")
TOKEN_PREFIXES = ("<apos_", "<opos_", "<act_l0_", "<act_l1_", "<act_l2_")
TIMING_KEYS = ("prepare", "vision", "prefill", "decode_5", "total", "frame_total")


def find_episode_paths(input_root: Path):
    return sorted(
        path for path in input_root.iterdir()
        if path.is_dir() and (path / "action.json").is_file()
    )


def load_ptq_episode(path: Path):
    action_path = path / "action.json"
    rows = json.loads(action_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{action_path} must contain a nonempty list")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{action_path} entries must be objects")
    sequences = [
        sorted(path.glob(f"image_*{suffix}")) for suffix in (".jpg", ".png")
    ]
    sequences = [images for images in sequences if images]
    if len(sequences) != 1:
        raise ValueError(f"{path.name}: expected exactly one image format (.jpg or .png)")
    images = sequences[0]
    suffix = images[0].suffix
    expected_names = [f"image_{index:06d}{suffix}" for index in range(1, len(rows) + 1)]
    if [image.name for image in images] != expected_names:
        raise ValueError(f"{path.name}: images must be contiguous and match action.json")
    instructions = {row.get("instruction") for row in rows}
    if len(instructions) != 1 or not isinstance(next(iter(instructions)), str):
        raise ValueError(f"{path.name}: every action must have one shared instruction")
    if [row.get("step") for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"{path.name}: action steps must be contiguous from 1")
    return next(iter(instructions)), images, rows


def parse_action_tokens(raw_text: str, vocabulary: dict[str, int]) -> list[int]:
    pieces = TOKEN_PATTERN.findall(raw_text) if isinstance(raw_text, str) else []
    if (len(pieces) != OUTPUT_TOKENS + 1 or "".join(pieces) != raw_text
            or any(not pieces[index].startswith(prefix) for index, prefix in enumerate(TOKEN_PREFIXES))
            or pieces[-1] != "<|im_end|>"):
        raise ValueError("raw_text must contain five LightNav action tokens followed by <|im_end|>")
    try:
        return [vocabulary[piece] for piece in pieces[:OUTPUT_TOKENS]]
    except KeyError as error:
        raise ValueError(f"raw_text token is absent from the checkpoint tokenizer: {error.args[0]}") from error


def _percentile(values, percent):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * percent) - 1)]


def summarize_timings(rows):
    if not rows:
        return None
    result = {"frames": len(rows)}
    for key in TIMING_KEYS:
        values = [float(row[key]) for row in rows]
        result[key + "_median_ms"] = round(statistics.median(values), 3)
        result[key + "_mean_ms"] = round(statistics.fmean(values), 3)
        result[key + "_p95_ms"] = round(_percentile(values, 0.95), 3)
    result["mean_fps"] = round(1000.0 / result["frame_total_mean_ms"], 4)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lightnav-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--vision-acc32", action="store_true")
    args = parser.parse_args(argv)
    if args.episode_limit is not None and args.episode_limit < 1:
        parser.error("--episode-limit must be positive")

    input_root = args.input_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    source_root = args.lightnav_source.expanduser().resolve()
    source_dir = source_root / "src"
    output_path = args.output.expanduser().resolve()
    try:
        episodes = find_episode_paths(input_root)
        if args.episode_limit is not None:
            episodes = episodes[:args.episode_limit]
        if not episodes:
            raise ValueError("--input-root contains no episode directories")
        episode_rows = [(path, *load_ptq_episode(path)) for path in episodes]
        if not (checkpoint / "eval_config.json").is_file():
            raise ValueError("--checkpoint must contain eval_config.json")
        if not (source_dir / "lightnav/inference/model.py").is_file():
            raise ValueError("--lightnav-source is missing src/lightnav/inference/model.py")
        if (output_path.suffix != ".json" or output_path.exists()
                or output_path.is_relative_to(input_root)
                or output_path.is_relative_to(checkpoint)
                or output_path.is_relative_to(source_root)):
            raise ValueError("--output must be a new JSON file outside inputs and model sources")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    if args.check_config:
        print(
            f"configuration OK: episodes={len(episode_rows)}, "
            f"frames={sum(len(rows) for _, _, _, rows in episode_rows)}, "
            f"history_frames={HISTORY_FRAMES}, output_tokens={OUTPUT_TOKENS}"
        )
        return 0

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(source_dir))
    os.environ["RPU_LIGHTNAV_CONSTRAINED_HEAD"] = "1"

    import torch
    import lightnav_benchmark as benchmark
    import lightnav_inference as inference
    from lightnav.inference.model import (
        _build_data_processor, _make_bundle, _resolve_processing_params,
    )
    from lightnav.inference.vit_cache import VitTubeletCache
    from lightnav.processing import VLNQwen3VLProcessor
    from rpu_backend.api import RPUCache, RPUModelForConditionalGeneration
    from rpu_backend.adapters.qwen3_vl import set_lightnav_constrained_lm_head

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
    model = RPUModelForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.float16, device="rpu",
        rpu_execution={
            "prefill": {"fast_replay": True},
            "vision": {"linear_acc32": args.vision_acc32},
        },
        local_files_only=True, trust_remote_code=False,
    )
    vocabulary = processor.tokenizer.get_vocab()
    set_lightnav_constrained_lm_head(model)
    lm_cache = RPUCache.from_model(
        model.model.language_model,
        max_seq_len=benchmark.MAX_PREFILL_TOKENS + benchmark.OUTPUT_TOKENS,
    )

    total_frames = 0
    exact_sequences = 0
    matching_tokens = [0] * OUTPUT_TOKENS
    all_timings = []
    steady_timings = []
    steady_cache_patterns = {}
    mismatches = []
    try:
        for episode_number, (path, instruction, images, actions) in enumerate(episode_rows, 1):
            references = [
                parse_action_tokens(row.get("raw_text"), vocabulary) for row in actions
            ]
            frames = inference.load_frames(images, tuple(bundle.video_size))
            results = inference.infer_episode(
                model, processor, data_processor, bundle, frames, instruction,
                VitTubeletCache(max_entries=benchmark.VISION_CACHE_ENTRIES, store_device="cpu"),
                lm_cache, [image.name for image in images], history_frames=HISTORY_FRAMES,
                prefill_graph_cache=model.model.language_model._rpu_text_graph_cache,
                progress=False,
            )
            for frame_index, (actual, expected) in enumerate(zip(results, references)):
                tokens = actual["token_ids"]
                total_frames += 1
                exact_sequences += tokens == expected
                for position, (left, right) in enumerate(zip(tokens, expected)):
                    matching_tokens[position] += left == right
                if tokens != expected and len(mismatches) < 100:
                    mismatches.append({
                        "episode": path.name, "frame_index": frame_index,
                        "expected": expected, "actual": tokens,
                    })
                timing = actual["timing_ms"]
                all_timings.append(timing)
                if frame_index > HISTORY_FRAMES:
                    steady_timings.append(timing)
                    pattern = f"{actual['cache_hits']}/{actual['cache_misses']}"
                    steady_cache_patterns[pattern] = steady_cache_patterns.get(pattern, 0) + 1
            print(
                f"[{episode_number}/{len(episode_rows)}] {path.name}: "
                f"frames={len(results)} exact={sum(result['token_ids'] == reference for result, reference in zip(results, references))}",
                flush=True,
            )
    finally:
        from rpu_backend.adapters.qwen3_vl import _is_retirement_owner

        owner = getattr(model, "_qwen3_vl_retirement_owner", None)
        if not _is_retirement_owner(owner, model):
            raise RuntimeError("Qwen3-VL teardown lost its actual composite retirement owner")
        owner.close()

    steady = summarize_timings(steady_timings)
    report = {
        "schema": 1,
        "episodes": len(episode_rows),
        "frames": total_frames,
        "history_frames": HISTORY_FRAMES,
        "output_tokens": OUTPUT_TOKENS,
        "reference": "customer action.json from original LightNav-0",
        "accuracy": {
            "exact_5_tokens": exact_sequences,
            "exact_5_token_rate": round(exact_sequences / total_frames, 8),
            "matching_tokens_by_position": matching_tokens,
            "token_rate_by_position": [round(value / total_frames, 8) for value in matching_tokens],
            "mismatch_examples": mismatches,
        },
        "timing_ms": {
            "all_frames": summarize_timings(all_timings),
            "steady_full_windows": steady,
        },
        "vision_cache": {
            "steady_hit_miss_patterns": steady_cache_patterns,
            "expected_pattern": "4/1",
        },
        "acceptance": {
            "target_hz": 4.0,
            "maximum_mean_frame_total_ms": 250.0,
            "performance_pass": steady is not None and steady["frame_total_mean_ms"] <= 250.0,
            "closed_loop_reference_success_rate": 0.91,
            "minimum_closed_loop_success_rate": 0.83,
            "closed_loop_evaluated": False,
        },
        "notes": [
            "Beginning frames use all available history; steady timing starts at frame_index=10 after the first full-window Graph build.",
            "Offline token agreement is a regression metric and does not replace closed-loop simulation.",
            "Model loading and initial per-episode JPEG decode/resize are excluded from frame timings.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
