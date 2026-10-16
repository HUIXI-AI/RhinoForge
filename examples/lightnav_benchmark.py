#!/usr/bin/env python3
"""Benchmark LightNav-0 on RPU with a session-local ViT tubelet cache."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
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


def _select_input_window(frames, history_frames: int, offset: int):
    window = frames[offset : offset + history_frames + 1]
    if len(window) != history_frames + 1:
        raise RuntimeError("insufficient frames for history plus current frame")
    return window


class _HistoryOnlyCache:
    """Keep every tubelet except the one containing the current frame."""

    def __init__(self, cache, current_frame_id: int):
        self._cache = cache
        self._current_frame_id = current_frame_id
        self._pooled_segments = OrderedDict()

    def __getattr__(self, name):
        return getattr(self._cache, name)

    def put(self, key, value) -> None:
        if self._current_frame_id not in key[:2]:
            self._cache.put(key, value)

    def get_pooled_segment(self, key):
        value = self._pooled_segments.get(key)
        if value is not None:
            self._pooled_segments.move_to_end(key)
        return value

    def put_pooled_segment(self, key, value) -> None:
        tubelet_keys = key[0]
        if any(self._current_frame_id in tubelet[:2] for tubelet in tubelet_keys):
            return
        self._pooled_segments[key] = value
        self._pooled_segments.move_to_end(key)
        if len(self._pooled_segments) > self._cache.max_entries:
            self._pooled_segments.popitem(last=False)


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
    lookup_started = time.perf_counter()
    plan = cache.collect_misses(
        pixels, raw_grid.tolist(), frame_indices, merge_size=merge_size,
    )
    lookup_ms = (time.perf_counter() - lookup_started) * 1000

    vit_ms = 0.0
    if plan.missing_units:
        miss_grid = torch.tensor(plan.miss_grid_rows, dtype=torch.long)
        vit_started = time.perf_counter()
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
        vit_ms = (time.perf_counter() - vit_started) * 1000

    pool_factors = inputs.get("_post_vit_pool_factors")
    if pool_factors is None:
        pool_factors = torch.ones(raw_grid.shape[0], dtype=torch.long)
    base_parts: list[Any] = []
    deepstack_parts: list[list[Any]] = [[], [], []]
    pooled_grids: list[Any] = []
    offset = 0
    pool_started = time.perf_counter()
    for segment_index, row in enumerate(raw_grid):
        tubelets = int(row[0].item())
        segment_units = plan.unit_desc[offset : offset + tubelets]
        entries = [
            plan.cached_map[unit["key"]]
            for unit in segment_units
        ]
        offset += tubelets
        factor = int(pool_factors[segment_index].item())
        segment_key = (
            tuple(unit["key"] for unit in segment_units), factor, merge_size,
        )
        pooled_segment = (
            cache.get_pooled_segment(segment_key)
            if hasattr(cache, "get_pooled_segment") else None
        )
        if pooled_segment is None:
            base = torch.cat([entry.base_embed for entry in entries], dim=0)
            deepstack = [
                torch.cat(
                    [entry.deepstack_embeds[layer] for entry in entries], dim=0,
                )
                for layer in range(3)
            ]
            pooled_segment = post_vit_spatial_pool(
                base,
                row.unsqueeze(0),
                spatial_factor=factor,
                merge_size=merge_size,
                deepstack_embeds=deepstack,
            )
            if hasattr(cache, "put_pooled_segment"):
                cache.put_pooled_segment(segment_key, pooled_segment)
        pooled, pooled_grid, pooled_deepstack = pooled_segment
        if pooled_deepstack is None or len(pooled_deepstack) != 3:
            raise RuntimeError("LightNav ViT must return three DeepStack levels")
        base_parts.append(pooled.contiguous())
        pooled_grids.append(pooled_grid)
        for layer, value in enumerate(pooled_deepstack):
            deepstack_parts[layer].append(value.contiguous())

    final_grid = torch.cat(pooled_grids, dim=0)
    expected_grid = inputs["video_grid_thw"]
    if not torch.equal(final_grid, expected_grid):
        raise RuntimeError(
            f"cached pooled grid {final_grid.tolist()} != processor grid "
            f"{expected_grid.tolist()}"
        )
    pool_ms = (time.perf_counter() - pool_started) * 1000
    return {
        "pooler_output_parts": base_parts,
        "deepstack_feature_parts": deepstack_parts,
        "cached_prefix_parts": 0,
    }, plan.hit, plan.miss, {
        "vision_lookup_ms": lookup_ms,
        "vision_vit_ms": vit_ms,
        "vision_pool_ms": pool_ms,
    }


def _generate_five(
    model, input_ids, prefill: dict[str, Any], cache,
    draft_tokens: list[int] | None = None,
):
    import torch

    def next_token(logits):
        from rpu_backend.api import greedy_token_ids

        if getattr(model, "_rpu_lightnav_constrained_head_enabled", False):
            candidates = model._rpu_lightnav_candidate_token_ids
            token = greedy_token_ids(logits[..., : candidates.numel()])
            return candidates.index_select(0, token.reshape(-1)).reshape(token.shape)
        return greedy_token_ids(logits)

    if input_ids.shape[1] + OUTPUT_TOKENS > cache.max_seq_len:
        raise RuntimeError("LightNav prompt outgrew the stable RPU KV cache")
    if draft_tokens is not None:
        if len(draft_tokens) != OUTPUT_TOKENS:
            raise ValueError("LightNav history draft must contain five tokens")
        draft_prefix = torch.tensor(
            draft_tokens[:-1], dtype=input_ids.dtype, device=input_ids.device,
        ).unsqueeze(0)
        verify_ids = torch.cat((input_ids, draft_prefix), dim=1)
        verify_prefill = dict(prefill)
        for key, value in (
            ("attention_mask", 1), ("mm_token_type_ids", 0),
        ):
            current = verify_prefill[key]
            suffix = current.new_full((current.shape[0], OUTPUT_TOKENS - 1), value)
            verify_prefill[key] = torch.cat((current, suffix), dim=1)
        cache.reset_to_position(0)
        verify_started = time.perf_counter()
        with torch.no_grad():
            output = model(
                input_ids=verify_ids.to("rpu"),
                past_key_values=cache,
                logits_to_keep=OUTPUT_TOKENS,
                use_cache=True,
                return_dict=True,
                **verify_prefill,
            )
            predicted = [
                int(next_token(output.logits[:, index : index + 1]).item())
                for index in range(output.logits.shape[1])
            ]
        verify_ms = (time.perf_counter() - verify_started) * 1000
        if predicted == draft_tokens:
            return predicted, verify_ms, 0.0, True
        text_graphs = model.model.language_model._rpu_text_graph_cache
        text_graphs.clear()
        generated, prefill_ms, decode_ms, _ = _generate_five(
            model, input_ids, prefill, cache,
        )
        text_graphs.clear()
        return generated, verify_ms + prefill_ms, decode_ms, False

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
            next_id = next_token(output.logits)
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
    return generated, prefill_ms, decode_ms, None


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
    draft_tokens: list[int] | None = None,
):
    from lightnav.inference.samples import build_tracking_sample

    started = time.perf_counter()
    window = _select_input_window(frames, history_frames, offset)
    sample = build_tracking_sample(window, instruction, None, bundle)
    for segment in sample["video_segments"]:
        segment["frame_indices"] = [
            int(value) + offset for value in segment["frame_indices"]
        ]
    sample["_vit_cached_keys"] = cache.cached_keys()
    sample_ms = (time.perf_counter() - started) * 1000
    processor_started = time.perf_counter()
    inputs = data_processor.process_sample(
        sample, add_generation_prompt=True, validate_video_shapes=False,
    )
    processor_ms = (time.perf_counter() - processor_started) * 1000
    prepare_ms = (time.perf_counter() - started) * 1000

    vision_started = time.perf_counter()
    precomputed, hits, misses, vision_parts = _cache_vision(
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
    tokens, prefill_ms, decode_ms, draft_accepted = _generate_five(
        model,
        input_ids,
        {
            "attention_mask": attention_mask,
            "video_grid_thw": inputs["video_grid_thw"],
            "mm_token_type_ids": mm_token_type_ids,
            "rpu_precomputed_vision": precomputed,
        },
        lm_cache,
        draft_tokens,
    )
    result = {
        "offset": offset,
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": len(tokens),
        "token_ids": tokens,
        "cache_hits": hits,
        "cache_misses": misses,
        "prepare_ms": prepare_ms,
        "prepare_sample_ms": sample_ms,
        "prepare_processor_ms": processor_ms,
        "vision_ms": vision_ms,
        **vision_parts,
        "prefill_ms": prefill_ms,
        "decode_5_ms": decode_ms,
        "draft_accepted": draft_accepted,
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


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        key: _median(rows, key)
        for key in (
            "prepare_ms", "prepare_sample_ms", "prepare_processor_ms", "vision_ms",
            "vision_lookup_ms", "vision_vit_ms", "vision_pool_ms", "prefill_ms",
            "decode_5_ms", "total_ms",
        )
    }
    for key in ("input_tokens", "output_tokens", "cache_hits", "cache_misses"):
        values = {row[key] for row in rows}
        if len(values) != 1:
            raise RuntimeError(f"measured requests changed {key}: {sorted(values)}")
        summary[key] = values.pop()
    summary["draft_attempts"] = sum(
        row["draft_accepted"] is not None for row in rows
    )
    summary["draft_accepts"] = sum(
        row["draft_accepted"] is True for row in rows
    )
    return summary


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
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--instruction", default="Navigate to the target and report the next action."
    )
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument(
        "--constrained-head",
        action="store_true",
        help="project only the checkpoint action-token allowlist",
    )
    parser.add_argument(
        "--history-draft",
        action="store_true",
        help="verify the previous request's five tokens in one prefill",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.constrained_head:
        os.environ["RPU_LIGHTNAV_CONSTRAINED_HEAD"] = "1"
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
    candidate_count = 0
    if args.constrained_head:
        from rpu_backend.adapters.qwen3_vl import (
            set_lightnav_constrained_lm_head,
        )

        candidate_count = len(set_lightnav_constrained_lm_head(model))
    try:
        max_frames = max(args.history_frames) + 1
        frames = torch.empty((max_frames, 3, 256, 448), dtype=torch.float32)
        for index in range(max_frames):
            frames[index].fill_((index % 17) / 8.0 - 1.0)

        report = {
            "checkpoint": str(checkpoint),
            "output_tokens": OUTPUT_TOKENS,
            "graph_warmup_excluded": True,
            "repeats": args.repeats,
            "frame_count_semantics": "history_plus_current",
            "lm_head": (
                "action_allowlist" if args.constrained_head else "full_vocabulary"
            ),
            "lm_head_candidates": candidate_count,
            "task_accuracy_evaluated": False,
            "decode_mode": (
                "verified_history_draft" if args.history_draft else "serial"
            ),
            "history": [],
        }
        for history_frames in args.history_frames:
            lm_cache = RPUCache.from_model(
                model.model.language_model,
                max_seq_len=MAX_PREFILL_TOKENS + OUTPUT_TOKENS,
            )
            # Warm the full-window RPU graphs without retaining visual features.
            warm = _run_request(
                model, processor, data_processor, bundle,
                VitTubeletCache(max_entries=256, store_device="cpu"), frames,
                history_frames, 0, args.instruction, lm_cache,
            )
            draft = warm["token_ids"] if args.history_draft else None
            if draft is not None:
                model.model.language_model._rpu_text_graph_cache.clear()
                warm = _run_request(
                    model, processor, data_processor, bundle,
                    VitTubeletCache(max_entries=256, store_device="cpu"), frames,
                    history_frames, 0, args.instruction, lm_cache,
                    draft_tokens=draft,
                )
                draft = warm["token_ids"]
            all_miss = []
            for _ in range(args.repeats):
                row = _run_request(
                    model, processor, data_processor, bundle,
                    VitTubeletCache(max_entries=256, store_device="cpu"), frames,
                    history_frames, 0, args.instruction, lm_cache,
                    capture_vision=not all_miss,
                    draft_tokens=draft,
                )
                all_miss.append(row)
                if args.history_draft:
                    draft = row["token_ids"]
            all_miss_vision = all_miss[0].pop("_vision")
            if any(row["cache_hits"] != 0 for row in all_miss):
                raise RuntimeError(f"history={history_frames}: all-miss cache hit")

            history_cache = _HistoryOnlyCache(
                VitTubeletCache(max_entries=256, store_device="cpu"),
                current_frame_id=history_frames,
            )
            # Populate history entries, then warm the one-current-tubelet graph.
            warm = _run_request(
                model, processor, data_processor, bundle, history_cache, frames,
                history_frames, 0, args.instruction, lm_cache,
                draft_tokens=draft if args.history_draft else None,
            )
            draft = warm["token_ids"] if args.history_draft else None
            warm = _run_request(
                model, processor, data_processor, bundle, history_cache, frames,
                history_frames, 0, args.instruction, lm_cache,
                draft_tokens=draft,
            )
            if args.history_draft:
                draft = warm["token_ids"]
            all_history_hit = []
            for _ in range(args.repeats):
                row = _run_request(
                    model, processor, data_processor, bundle, history_cache, frames,
                    history_frames, 0, args.instruction, lm_cache,
                    capture_vision=not all_history_hit,
                    draft_tokens=draft,
                )
                all_history_hit.append(row)
                if args.history_draft:
                    draft = row["token_ids"]
            history_hit_vision = all_history_hit[0].pop("_vision")
            if any(row["cache_misses"] != 1 for row in all_history_hit):
                raise RuntimeError(
                    f"history={history_frames}: expected only the current "
                    "tubelet to miss"
                )
            feature_diff = _vision_feature_max_abs_diff(
                all_miss_vision, history_hit_vision,
            )
            if feature_diff > VISION_CACHE_ATOL:
                raise RuntimeError(
                    f"history={history_frames}: cached history changed the "
                    "pooler or DeepStack features "
                    f"(max diff={feature_diff}, atol={VISION_CACHE_ATOL})"
                )
            report["history"].append({
                "history_frames": history_frames,
                "current_frames": 1,
                "total_raw_frames": history_frames + 1,
                "all_miss_median": _summarize(all_miss),
                "all_history_hit_median": _summarize(all_history_hit),
                "vision_features_within_tolerance": True,
                "vision_feature_max_abs_diff": feature_diff,
                "vision_feature_atol": VISION_CACHE_ATOL,
                "all_miss_measured": all_miss,
                "all_history_hit_measured": all_history_hit,
            })
            print(
                f"history={history_frames}+current: "
                f"all_miss={report['history'][-1]['all_miss_median']['total_ms']:.1f}ms, "
                f"history_hit={report['history'][-1]['all_history_hit_median']['total_ms']:.1f}ms, "
                f"hit/miss={report['history'][-1]['all_history_hit_median']['cache_hits']}/"
                f"{report['history'][-1]['all_history_hit_median']['cache_misses']}"
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
