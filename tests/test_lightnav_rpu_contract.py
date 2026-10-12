"""Board-free contracts for the LightNav video-cache path."""

from pathlib import Path
import argparse
import importlib.util
import sys
import types

import pytest
import torch

from rpu_backend.runtime.weights import transform_linear_weight


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "python/rpu_backend/adapters/qwen3_vl/__init__.py"
).read_text()
EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples/lightnav_benchmark.py"
SPEC = importlib.util.spec_from_file_location("lightnav_benchmark", EXAMPLE_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_precomputed_vision_selects_the_matching_video_grid_and_token():
    assert "visual_grid_thw = video_grid_thw" in SOURCE
    assert "visual_token_id = self.config.video_token_id" in SOURCE
    assert "visual_grid_thw.detach().cpu()" in SOURCE


def test_row_swizzle_accepts_lightnav_vocabulary_alignment():
    weight = torch.arange(144 * 128, dtype=torch.float16).reshape(144, 128)
    transformed = transform_linear_weight(weight, partition=0)
    assert transformed.shape == weight.shape
    assert transformed.is_contiguous()
    assert torch.equal(transformed.flatten().sort().values, weight.flatten().sort().values)


def test_benchmark_contract_fixes_five_tokens_and_history_set():
    assert BENCHMARK.OUTPUT_TOKENS == 5
    assert BENCHMARK.MAX_PREFILL_TOKENS == 4176
    assert BENCHMARK.VISION_CACHE_ATOL == 1e-3
    assert BENCHMARK.parse_histories("12,24,36,48,64") == (12, 24, 36, 48, 64)
    with pytest.raises(argparse.ArgumentTypeError, match="positive even"):
        BENCHMARK.parse_histories("12,13")


def test_benchmark_adds_current_frame_and_only_caches_history():
    frames = list(range(14))
    assert BENCHMARK._select_input_window(frames, 12, 0) == list(range(13))
    assert BENCHMARK._select_input_window(frames, 12, 1) == list(range(1, 14))
    with pytest.raises(RuntimeError, match="insufficient frames"):
        BENCHMARK._select_input_window(frames, 12, 2)

    class Cache:
        def __init__(self):
            self.values = {}

        def put(self, key, value):
            self.values[key] = value

    cache = Cache()
    history_only = BENCHMARK._HistoryOnlyCache(cache, current_frame_id=12)
    history_only.put((10, 11, 8, 14), "history")
    history_only.put((11, 12, 8, 14), "current")
    assert cache.values == {(10, 11, 8, 14): "history"}


def test_prepare_prompt_trims_answer_and_marks_video_tokens():
    inputs = {
        "input_ids": torch.tensor([7, 151656, 9, 10]),
        "attention_mask": torch.ones(4, dtype=torch.bool),
        "labels": torch.tensor([-100, -100, -100, 10]),
    }
    input_ids, attention_mask, token_types = BENCHMARK._prepare_prompt(
        inputs, 151656,
    )
    assert tuple(input_ids.shape) == (1, 3)
    assert attention_mask.all()
    assert token_types.tolist() == [[0, 2, 0]]


def test_vision_cache_reassembles_repeat_and_sliding_windows(monkeypatch):
    class Entry:
        def __init__(self, base_embed, deepstack_embeds):
            self.base_embed = base_embed
            self.deepstack_embeds = deepstack_embeds

    class Cache:
        def __init__(self):
            self.values = {}

        def put(self, key, value):
            self.values[key] = value

        def collect_misses(self, pixels, grid, frame_indices, merge_size):
            keys = [tuple(frame_indices[0][i : i + 2]) for i in range(0, len(frame_indices[0]), 2)]
            units = [
                {"key": key, "out_len": 1, "pixels": pixels[index : index + 1]}
                for index, key in enumerate(keys)
            ]
            missing = [unit for unit in units if unit["key"] not in self.values]
            return types.SimpleNamespace(
                unit_desc=units,
                cached_map={key: self.values[key] for key in keys if key in self.values},
                missing_units=missing,
                miss_grid_rows=[[1, 1, 1] for _ in missing],
                miss_pixels=torch.cat([unit["pixels"] for unit in missing]) if missing else None,
                hit=len(units) - len(missing),
                miss=len(missing),
            )

    def pool(base, grid, spatial_factor, merge_size, deepstack_embeds):
        return base, grid, deepstack_embeds

    cache_module = types.ModuleType("lightnav.inference.vit_cache")
    cache_module.CachedTubeletEmbeds = Entry
    processing_module = types.ModuleType("lightnav.processing")
    processing_module.post_vit_spatial_pool = pool
    monkeypatch.setitem(sys.modules, "lightnav", types.ModuleType("lightnav"))
    monkeypatch.setitem(sys.modules, "lightnav.inference", types.ModuleType("lightnav.inference"))
    monkeypatch.setitem(sys.modules, "lightnav.inference.vit_cache", cache_module)
    monkeypatch.setitem(sys.modules, "lightnav.processing", processing_module)

    class Visual:
        def __init__(self):
            self.rows = []

        def __call__(self, pixels, grid_thw):
            self.rows.append(len(pixels))
            return types.SimpleNamespace(
                pooler_output=pixels,
                deepstack_features=[pixels + value for value in (100, 200, 300)],
            )

    visual = Visual()
    model = types.SimpleNamespace(model=types.SimpleNamespace(visual=visual))
    processor = types.SimpleNamespace(video_processor=types.SimpleNamespace(merge_size=1))

    def run(cache, frame_ids, values):
        grid = torch.tensor([[2, 1, 1]])
        result, hits, misses = BENCHMARK._cache_vision(
            model,
            processor,
            cache,
            {"video_segments": [{"frame_indices": frame_ids}]},
            {
                "pixel_values_videos": torch.tensor(values, dtype=torch.float16).reshape(2, 1),
                "video_grid_thw": grid,
                "_original_video_grid_thw": grid,
            },
        )
        return result, hits, misses

    cache = Cache()
    cold, cold_hits, cold_misses = run(cache, [0, 1, 2, 3], [10, 20])
    repeat, repeat_hits, repeat_misses = run(cache, [0, 1, 2, 3], [10, 20])
    sliding, sliding_hits, sliding_misses = run(cache, [2, 3, 4, 5], [20, 30])
    fresh, _, _ = run(Cache(), [2, 3, 4, 5], [20, 30])

    assert (cold_hits, cold_misses) == (0, 2)
    assert (repeat_hits, repeat_misses) == (2, 0)
    assert (sliding_hits, sliding_misses) == (1, 1)
    assert visual.rows == [2, 1, 2]
    assert BENCHMARK._vision_feature_max_abs_diff(cold, repeat) == 0
    assert BENCHMARK._vision_feature_max_abs_diff(sliding, fresh) == 0
    invalid_pooler = [part.clone() for part in fresh["pooler_output_parts"]]
    invalid_pooler[0][0, 0] = float("nan")
    invalid = {**fresh, "pooler_output_parts": invalid_pooler}
    assert BENCHMARK._vision_feature_max_abs_diff(sliding, invalid) == float("inf")
    assert torch.equal(cold["pooler_output_parts"][0], repeat["pooler_output_parts"][0])
    assert torch.equal(sliding["pooler_output_parts"][0], fresh["pooler_output_parts"][0])
    for layer in range(3):
        assert torch.equal(
            sliding["deepstack_feature_parts"][layer][0],
            fresh["deepstack_feature_parts"][layer][0],
        )


def test_generate_reuses_and_rewinds_one_rpu_cache(monkeypatch):
    class Value:
        def to(self, device):
            return self

        def item(self):
            return 7

    class Cache:
        def __init__(self):
            self.max_seq_len = 20
            self.rewinds = []

        def reset_to_position(self, position):
            self.rewinds.append(position)

    api = types.ModuleType("rpu_backend.api")
    api.RPUCache = Cache
    api.greedy_token_ids = lambda logits: Value()
    monkeypatch.setitem(sys.modules, "rpu_backend.api", api)

    class Input(Value):
        shape = (1, 10)

    class Model:
        model = types.SimpleNamespace(language_model=object())

        def __call__(self, **kwargs):
            return types.SimpleNamespace(logits=object())

    cache = Cache()
    BENCHMARK._generate_five(Model(), Input(), {}, cache)
    BENCHMARK._generate_five(Model(), Input(), {}, cache)

    assert cache.rewinds == [0, 0]
