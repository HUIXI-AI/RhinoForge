#!/usr/bin/env python3
"""Benchmark LightNav-0 on RPU with a session-local ViT tubelet cache."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


OUTPUT_TOKENS = 5
MAX_PREFILL_TOKENS = 4176
DEFAULT_HISTORIES = (12, 24, 36, 48, 64)
VISION_CACHE_ATOL = 1e-3


def parse_histories(value: str) -> tuple[int, ...]:
    histories = tuple(int(item) for item in value.split(",") if item.strip())
    if not histories or any(item < 2 or item % 2 for item in histories):
        raise argparse.ArgumentTypeError(
            "history frames must be a comma-separated list of positive even values"
        )
    return histories


def _prepare_prompt(inputs: dict[str, Any], video_token_id: int):
    import torch

    labels = inputs["labels"]
    answer = (labels != -100).nonzero(as_tuple=False)
    if answer.numel() == 0:
        raise RuntimeError("LightNav processor did not mark an answer boundary")
    answer_start = int(answer[0].item())
    input_ids = inputs["input_ids"][:answer_start].unsqueeze(0)
    attention_mask = inputs["attention_mask"][:answer_start].unsqueeze(0)
    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[input_ids == video_token_id] = 2
    return input_ids, attention_mask, mm_token_type_ids


def _cache_vision(
    model,
    processor,
    cache,
    sample: dict[str, Any],
    inputs: dict[str, Any],
):
    import torch
    from lightnav.inference.vit_cache import CachedTubeletEmbeds
    from lightnav.processing import post_vit_spatial_pool

    pixels = inputs["pixel_values_videos"]
    if pixels.ndim == 3:
        pixels = pixels.squeeze(0)
    raw_grid = inputs.get("_original_video_grid_thw", inputs["video_grid_thw"])
    merge_size = int(processor.video_processor.merge_size)
    frame_indices = [
        [int(value) for value in segment["frame_indices"]]
        for segment in sample["video_segments"]
    ]
    plan = cache.collect_misses(
        pixels, raw_grid.tolist(), frame_indices, merge_size=merge_size,
    )

    if plan.missing_units:
        miss_grid = torch.tensor(plan.miss_grid_rows, dtype=torch.long)
        with torch.no_grad():
            output = model.model.visual(
                plan.miss_pixels, grid_thw=miss_grid,
            )
        base = output.pooler_output.detach().to("cpu", dtype=torch.float16)
        deepstack = [
            value.detach().to("cpu", dtype=torch.float16)
            for value in output.deepstack_features
        ]
        split_sizes = [int(unit["out_len"]) for unit in plan.missing_units]
        base_parts = torch.split(base, split_sizes, dim=0)
        deep_parts = [torch.split(value, split_sizes, dim=0) for value in deepstack]
        for index, unit in enumerate(plan.missing_units):
            entry = CachedTubeletEmbeds(
                base_embed=base_parts[index].contiguous(),
                deepstack_embeds=[
                    layer[index].contiguous() for layer in deep_parts
                ],
            )
            cache.put(unit["key"], entry)
            plan.cached_map[unit["key"]] = entry

    pool_factors = inputs.get("_post_vit_pool_factors")
    if pool_factors is None:
        pool_factors = torch.ones(raw_grid.shape[0], dtype=torch.long)
    base_parts: list[Any] = []
    deepstack_parts: list[list[Any]] = [[], [], []]
    pooled_grids: list[Any] = []
    offset = 0
    for segment_index, row in enumerate(raw_grid):
        tubelets = int(row[0].item())
        entries = [
            plan.cached_map[unit["key"]]
            for unit in plan.unit_desc[offset : offset + tubelets]
        ]
        offset += tubelets
        base = torch.cat([entry.base_embed for entry in entries], dim=0)
        deepstack = [
            torch.cat([entry.deepstack_embeds[layer] for entry in entries], dim=0)
            for layer in range(3)
        ]
        pooled, pooled_grid, pooled_deepstack = post_vit_spatial_pool(
            base,
            row.unsqueeze(0),
            spatial_factor=int(pool_factors[segment_index].item()),
            merge_size=merge_size,
            deepstack_embeds=deepstack,
        )
        base_parts.append(pooled.contiguous())
        pooled_grids.append(pooled_grid)
        if pooled_deepstack is None or len(pooled_deepstack) != 3:
            raise RuntimeError("LightNav ViT must return three DeepStack levels")
        for layer, value in enumerate(pooled_deepstack):
            deepstack_parts[layer].append(value.contiguous())

    final_grid = torch.cat(pooled_grids, dim=0)
    expected_grid = inputs["video_grid_thw"]
    if not torch.equal(final_grid, expected_grid):
        raise RuntimeError(
            f"cached pooled grid {final_grid.tolist()} != processor grid "
            f"{expected_grid.tolist()}"
        )
    return {
        "pooler_output_parts": base_parts,
        "deepstack_feature_parts": deepstack_parts,
        "cached_prefix_parts": 0,
    }, plan.hit, plan.miss


def _generate_five(model, input_ids, prefill: dict[str, Any], cache):
    import torch
    from rpu_backend.api import greedy_token_ids

    if input_ids.shape[1] + OUTPUT_TOKENS > cache.max_seq_len:
        raise RuntimeError("LightNav prompt outgrew the stable RPU KV cache")
    cache.reset_to_position(0)
    generated = []
    prefill_started = time.perf_counter()
    with torch.no_grad():
        output = model(
            input_ids=input_ids.to("rpu"),
            past_key_values=cache,
            logits_to_keep=1,
            use_cache=True,
            return_dict=True,
            **prefill,
        )
        prefill_ms = (time.perf_counter() - prefill_started) * 1000
        decode_started = time.perf_counter()
        for step in range(OUTPUT_TOKENS):
            next_id = greedy_token_ids(output.logits)
            generated.append(int(next_id.item()))
            if step + 1 < OUTPUT_TOKENS:
                output = model(
                    input_ids=next_id.to("rpu"),
                    past_key_values=cache,
                    logits_to_keep=1,
                    use_cache=True,
                    return_dict=True,
                )
        decode_ms = (time.perf_counter() - decode_started) * 1000
    return generated, prefill_ms, decode_ms


def _run_request(
    model,
    processor,
    data_processor,
    bundle,
    cache,
    frames,
    history_frames: int,
    offset: int,
    instruction: str,
    lm_cache,
    capture_vision: bool = False,
):
    from lightnav.inference.samples import build_tracking_sample

    started = time.perf_counter()
    window = frames[offset : offset + history_frames]
    sample = build_tracking_sample(window, instruction, None, bundle)
    for segment in sample["video_segments"]:
        segment["frame_indices"] = [
            int(value) + offset for value in segment["frame_indices"]
        ]
    sample["_vit_cached_keys"] = cache.cached_keys()
    inputs = data_processor.process_sample(
        sample, add_generation_prompt=True, validate_video_shapes=False,
    )
    prepare_ms = (time.perf_counter() - started) * 1000

    vision_started = time.perf_counter()
    precomputed, hits, misses = _cache_vision(
        model, processor, cache, sample, inputs,
    )
    vision_ms = (time.perf_counter() - vision_started) * 1000
    input_ids, attention_mask, mm_token_type_ids = _prepare_prompt(
        inputs, model.config.video_token_id,
    )
    if input_ids.shape[1] > MAX_PREFILL_TOKENS:
        raise RuntimeError(
            f"LightNav prompt has {input_ids.shape[1]} tokens; "
            f"RPU Qwen3-VL 4B admits at most {MAX_PREFILL_TOKENS}"
        )
    tokens, prefill_ms, decode_ms = _generate_five(
        model,
        input_ids,
        {
            "attention_mask": attention_mask,
            "video_grid_thw": inputs["video_grid_thw"],
            "mm_token_type_ids": mm_token_type_ids,
            "rpu_precomputed_vision": precomputed,
        },
        lm_cache,
    )
    result = {
        "offset": offset,
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": len(tokens),
        "token_ids": tokens,
        "cache_hits": hits,
        "cache_misses": misses,
        "prepare_ms": prepare_ms,
        "vision_ms": vision_ms,
        "prefill_ms": prefill_ms,
        "decode_5_ms": decode_ms,
        "total_ms": (time.perf_counter() - started) * 1000,
    }
    if capture_vision:
        result["_vision"] = precomputed
    return result


def _vision_feature_max_abs_diff(
    left: dict[str, Any], right: dict[str, Any],
) -> float:
    import torch

    difference = 0.0
    for key in ("pooler_output_parts", "deepstack_feature_parts"):
        left_parts = left[key]
        right_parts = right[key]
        if key == "deepstack_feature_parts":
            left_parts = [part for layer in left_parts for part in layer]
            right_parts = [part for layer in right_parts for part in layer]
        if len(left_parts) != len(right_parts):
            return float("inf")
        for left_part, right_part in zip(left_parts, right_parts):
            if left_part.shape != right_part.shape:
                return float("inf")
            if not torch.isfinite(left_part).all() or not torch.isfinite(right_part).all():
                return float("inf")
            difference = max(
                difference,
                float((left_part.float() - right_part.float()).abs().max().item()),
            )
    return difference


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return round(statistics.median(row[key] for row in rows), 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lightnav-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--history-frames",
        type=parse_histories,
        default=DEFAULT_HISTORIES,
        help="comma-separated even frame counts (default: 12,24,36,48,64)",
    )
    parser.add_argument("--warmups", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--instruction", default="Navigate to the target and report the next action."
    )
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    if args.warmups < 2 or args.repeats < 1:
        parser.error("--warmups must be at least 2 and --repeats must be positive")
    source_dir = args.lightnav_source.expanduser().resolve() / "src"
    if not (source_dir / "lightnav/inference/vit_cache.py").is_file():
        parser.error("--lightnav-source must point to the LightNav-0 repository")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "eval_config.json").is_file():
        parser.error("--checkpoint must contain the LightNav-0 checkpoint")
    sys.path.insert(0, str(source_dir))
    if args.check_config:
        print(
            f"configuration OK: output_tokens={OUTPUT_TOKENS}, "
            f"history_frames={args.history_frames}"
        )
        return 0

    import torch
    from lightnav.inference.model import (
        _build_data_processor,
        _make_bundle,
        _resolve_processing_params,
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
    data_processor = _build_data_processor(
        processor, params, placeholder_positions,
    )
    bundle = _make_bundle(
        None, processor, data_processor, torch.device("cpu"),
        placeholder_positions, params,
    )
    model = RPUModelForConditionalGeneration.from_pretrained(
        checkpoint,
        dtype=torch.float16,
        device="rpu",
        local_files_only=True,
        trust_remote_code=False,
    )
    try:
        max_frames = max(args.history_frames) + args.warmups + args.repeats
        frames = torch.empty((max_frames, 3, 256, 448), dtype=torch.float32)
        for index in range(max_frames):
            frames[index].fill_((index % 17) / 8.0 - 1.0)

        report = {
            "checkpoint": str(checkpoint),
            "output_tokens": OUTPUT_TOKENS,
            "graph_warmup_excluded": True,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "history": [],
        }
        for history_frames in args.history_frames:
            lm_cache = RPUCache.from_model(
                model.model.language_model,
                max_seq_len=MAX_PREFILL_TOKENS + OUTPUT_TOKENS,
            )
            _run_request(
                model, processor, data_processor, bundle,
                VitTubeletCache(max_entries=256, store_device="cpu"), frames,
                history_frames, 0, args.instruction, lm_cache,
            )
            cache = VitTubeletCache(max_entries=256, store_device="cpu")
            cold = _run_request(
                model, processor, data_processor, bundle, cache, frames,
                history_frames, 0, args.instruction, lm_cache,
                capture_vision=True,
            )
            cached_repeat = _run_request(
                model, processor, data_processor, bundle, cache, frames,
                history_frames, 0, args.instruction, lm_cache,
                capture_vision=True,
            )
            cold_vision = cold.pop("_vision")
            cached_repeat_vision = cached_repeat.pop("_vision")
            cache_feature_diff = _vision_feature_max_abs_diff(
                cold_vision, cached_repeat_vision,
            )
            if cache_feature_diff != 0:
                raise RuntimeError(
                    f"history={history_frames}: cached repeat changed the "
                    f"pooler or DeepStack features (max diff={cache_feature_diff})"
                )
            for offset in range(1, args.warmups + 1):
                _run_request(
                    model, processor, data_processor, bundle, cache, frames,
                    history_frames, offset, args.instruction, lm_cache,
                )
            measured = []
            for offset in range(
                args.warmups + 1,
                args.warmups + args.repeats + 1,
            ):
                measured.append(_run_request(
                    model, processor, data_processor, bundle, cache, frames,
                    history_frames, offset, args.instruction, lm_cache,
                    capture_vision=not measured,
                ))
            sliding_cached_vision = measured[0].pop("_vision")
            sliding_fresh = _run_request(
                model, processor, data_processor, bundle,
                VitTubeletCache(max_entries=256, store_device="cpu"), frames,
                history_frames, measured[0]["offset"], args.instruction,
                lm_cache, capture_vision=True,
            )
            sliding_fresh_vision = sliding_fresh.pop("_vision")
            sliding_feature_diff = _vision_feature_max_abs_diff(
                sliding_cached_vision, sliding_fresh_vision,
            )
            if sliding_feature_diff > VISION_CACHE_ATOL:
                raise RuntimeError(
                    f"history={history_frames}: cached sliding window changed "
                    "the pooler or DeepStack features "
                    f"(max diff={sliding_feature_diff}, atol={VISION_CACHE_ATOL})"
                )
            report["history"].append({
                "history_frames": history_frames,
                "cold": cold,
                "cached_repeat": cached_repeat,
                "cache_features_identical": True,
                "cache_feature_max_abs_diff": cache_feature_diff,
                "cache_tokens_identical": (
                    cached_repeat["token_ids"] == cold["token_ids"]
                ),
                "sliding_fresh": sliding_fresh,
                "sliding_features_within_tolerance": True,
                "sliding_feature_max_abs_diff": sliding_feature_diff,
                "sliding_feature_atol": VISION_CACHE_ATOL,
                "sliding_tokens_identical": (
                    sliding_fresh["token_ids"] == measured[0]["token_ids"]
                ),
                "steady_median": {
                    key: _median(measured, key)
                    for key in (
                        "cache_hits", "cache_misses", "prepare_ms", "vision_ms",
                        "prefill_ms", "decode_5_ms", "total_ms",
                    )
                },
                "measured": measured,
            })
            print(
                f"history={history_frames}: cold={cold['total_ms']:.1f}ms, "
                f"steady={report['history'][-1]['steady_median']['total_ms']:.1f}ms, "
                f"hit/miss={report['history'][-1]['steady_median']['cache_hits']}/"
                f"{report['history'][-1]['steady_median']['cache_misses']}"
            )
            model.model.language_model._rpu_text_graph_cache.clear()
            model.model.visual._rpu_vision_graph_cache.clear()
            del lm_cache
    finally:
        from rpu_backend.adapters.qwen3_vl import _is_retirement_owner

        owner = getattr(model, "_qwen3_vl_retirement_owner", None)
        if not _is_retirement_owner(owner, model):
            raise RuntimeError(
                "Qwen3-VL teardown lost its actual composite retirement owner"
            )
        owner.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
