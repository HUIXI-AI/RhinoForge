"""Board-free contracts for the LightNav video-cache path."""

from pathlib import Path
import argparse
import importlib.util
import sys
import types

import pytest
import torch

import rpu_backend.adapters.qwen3_vl as qwen3_vl
from rpu_backend.runtime.weights import transform_linear_weight


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "python/rpu_backend/adapters/qwen3_vl/__init__.py"
).read_text()
DECODE_REPLAY_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/fused/rpu_qwen3vl_decode_replay.cpp"
).read_text()
EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples/lightnav_benchmark.py"
SPEC = importlib.util.spec_from_file_location("lightnav_benchmark", EXAMPLE_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)
INFERENCE_PATH = Path(__file__).resolve().parents[1] / "examples/lightnav_inference.py"
INFERENCE_SPEC = importlib.util.spec_from_file_location("lightnav_inference", INFERENCE_PATH)
assert INFERENCE_SPEC is not None and INFERENCE_SPEC.loader is not None
INFERENCE = importlib.util.module_from_spec(INFERENCE_SPEC)
sys.modules["lightnav_benchmark"] = BENCHMARK
INFERENCE_SPEC.loader.exec_module(INFERENCE)


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
    assert BENCHMARK.DEFAULT_HISTORIES == (8,)
    assert BENCHMARK.VISION_CACHE_ENTRIES == 1024
    assert BENCHMARK.MAX_PREFILL_TOKENS == 4176
    assert BENCHMARK.VISION_CACHE_ATOL == 1e-3
    assert BENCHMARK.parse_histories("12,24,36,48,64") == (12, 24, 36, 48, 64)
    assert BENCHMARK.parse_histories("9") == (9,)
    with pytest.raises(argparse.ArgumentTypeError, match="positive values"):
        BENCHMARK.parse_histories("9,0")
    assert 'rpu_execution={"prefill": {"fast_replay": True}}' in EXAMPLE_PATH.read_text()


def test_episode_runner_disables_redundant_nine_frame_prefetch(monkeypatch):
    calls = []

    def run_request(*args, **kwargs):
        calls.append((args[6], args[7], kwargs["prefetch_latest"]))
        return {"token_ids": [1, 2, 3, 4, 5], "total_ms": 1.0}

    monkeypatch.setattr(BENCHMARK, "_run_request", run_request)

    class GraphCache:
        def __init__(self):
            self.clears = 0

        def clear(self):
            self.clears += 1

    graph_cache = GraphCache()
    results = INFERENCE.infer_episode(
        object(), object(), object(), object(), list(range(12)), "track",
        object(), object(), [f"frame-{index}" for index in range(12)],
        history_frames=9, prefill_graph_cache=graph_cache,
    )

    assert len(results) == 12
    assert graph_cache.clears == 10
    assert calls[9:] == [(9, 0, False), (9, 1, False), (9, 2, False)]


def test_lightnav_candidate_head_uses_checked_decode_replay():
    assert "lightnav_candidate_head" in DECODE_REPLAY_SOURCE
    assert "vocab_size_ == 1408" in DECODE_REPLAY_SOURCE
    assert "profile.head_dtype, vocab_size_, hidden_size()" in DECODE_REPLAY_SOURCE
    assert "qwen3vl_4b_w8_profile_" in DECODE_REPLAY_SOURCE


def test_lightnav_position_heads_follow_the_five_token_grammar():
    heads = qwen3_vl._lightnav_position_token_ids()
    assert tuple(len(head) for head in heads) == (1302, 1299, 258, 258, 258)
    assert heads[0][2:] == tuple(range(152443, 153743))
    assert heads[1][2:] == tuple(range(153743, 155040))
    assert heads[2][2:] == tuple(range(151675, 151931))
    assert heads[3][2:] == tuple(range(151931, 152187))
    assert heads[4][2:] == tuple(range(152187, 152443))


def test_lightnav_candidate_head_reconfigures_native_decode_once(monkeypatch):
    class Cache:
        def __init__(self):
            self.events = []

        def begin_warmup(self):
            self.events.append("warmup")

        def clear(self):
            self.events.append("clear")

    cache = Cache()
    text = types.SimpleNamespace(
        _rpu_decoder_handle=7,
        _rpu_text_graph_cache=cache,
        _rpu_decoder_graph_cache=cache,
    )
    token_ids = torch.tensor([1, 2])
    weight = torch.ones((128, 16), dtype=torch.float16)
    model = types.SimpleNamespace(
        config=object(),
        model=types.SimpleNamespace(language_model=text),
        _rpu_lightnav_candidate_token_ids=token_ids,
        _rpu_lightnav_candidate_lm_head_w=weight,
        _rpu_lightnav_position_lm_heads=[weight] * 4,
        _rpu_lightnav_constrained_head_enabled=False,
    )
    native_calls = []
    monkeypatch.setattr(qwen3_vl, "_is_lightnav_profile", lambda _config: True)
    monkeypatch.setattr(
        qwen3_vl, "_text_decode_execution_plan",
        lambda *args, **kwargs: native_calls.append(("plan", args, kwargs)),
    )
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_set_lm_head",
        lambda *args: native_calls.append(("set", args)), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_clear_lm_head",
        lambda *args: native_calls.append(("clear", args)), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_set_lightnav_heads",
        lambda *args: native_calls.append(("heads", args)), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_set_lightnav_head_index",
        lambda *args: native_calls.append(("index", args)), raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "causal_decoder_clear_lightnav_heads",
        lambda *args: native_calls.append(("clear_heads", args)), raising=False,
    )

    assert qwen3_vl.set_lightnav_constrained_lm_head(model) is token_ids
    assert cache.events == ["warmup", "clear"]
    assert [call[0] for call in native_calls] == ["set", "heads", "index", "plan"]
    assert model._rpu_lightnav_constrained_head_enabled is True

    qwen3_vl.set_lightnav_constrained_lm_head(model)
    assert [call[0] for call in native_calls] == ["set", "heads", "index", "plan"]

    qwen3_vl.set_lightnav_constrained_lm_head(model, False)
    assert cache.events == ["warmup", "clear", "warmup", "clear"]
    assert [call[0] for call in native_calls] == [
        "set", "heads", "index", "plan", "clear_heads", "clear", "plan",
    ]
    assert model._rpu_lightnav_constrained_head_enabled is False


def test_benchmark_adds_current_frame_and_only_caches_history():
    frames = list(range(14))
    assert BENCHMARK._select_input_window(frames, 12, 0) == list(range(13))
    assert BENCHMARK._select_input_window(frames, 12, 1) == list(range(1, 14))
    with pytest.raises(RuntimeError, match="insufficient frames"):
        BENCHMARK._select_input_window(frames, 12, 2)

    class Cache:
        def __init__(self):
            self.values = {}
            self.max_entries = 16

        def put(self, key, value):
            self.values[key] = value

    cache = Cache()
    history_only = BENCHMARK._HistoryOnlyCache(cache, current_frame_id=12)
    history_only.put((10, 11, 8, 14), "history")
    history_only.put((11, 12, 8, 14), "current")
    assert cache.values == {(10, 11, 8, 14): "history"}
    history_key = (((10, 11, 8, 14),), 2, 2)
    current_key = (((11, 12, 8, 14),), 1, 2)
    history_only.put_pooled_segment(history_key, "pooled history")
    history_only.put_pooled_segment(current_key, "pooled current")
    assert history_only.get_pooled_segment(history_key) == "pooled history"
    assert history_only.get_pooled_segment(current_key) is None


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

        def cached_keys(self):
            return set(self.values)

        def collect_misses(self, pixels, grid, frame_indices, merge_size):
            keys = [
                (*frame_indices[0][i : i + 2], 1, 1)
                for i in range(0, len(frame_indices[0]), 2)
            ]
            units = [
                {
                    "key": key, "h": 1, "w": 1, "out_len": 1,
                    "pixels": pixels[index : index + 1],
                }
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
    processor = types.SimpleNamespace(video_processor=types.SimpleNamespace(
        merge_size=1, temporal_patch_size=2, patch_size=1,
    ))

    def run(cache, frame_ids, values, *, prefetch=False, pool_factor=1):
        tubelets = len(frame_ids) // 2
        grid = torch.tensor([[tubelets, 1, 1]])
        result, hits, misses, timing = BENCHMARK._cache_vision(
            model,
            processor,
            cache,
            {"video_segments": [{"frame_indices": frame_ids}]},
            {
                "pixel_values_videos": torch.tensor(values, dtype=torch.float16).reshape(tubelets, -1),
                "video_grid_thw": grid,
                "_original_video_grid_thw": grid,
                "_post_vit_pool_factors": torch.tensor([pool_factor]),
            },
            prefetch_latest=prefetch,
        )
        return result, hits, misses, timing

    cache = Cache()
    cold, cold_hits, cold_misses, _ = run(cache, [0, 1, 2, 3], [10, 20])
    repeat, repeat_hits, repeat_misses, _ = run(cache, [0, 1, 2, 3], [10, 20])
    sliding, sliding_hits, sliding_misses, _ = run(cache, [2, 3, 4, 5], [20, 30])
    fresh, _, _, _ = run(Cache(), [2, 3, 4, 5], [20, 30])

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

    factor_cache = Cache()
    _, _, _, cold_pool = run(factor_cache, [0, 1, 2, 3], [10, 20], pool_factor=2)
    _, _, _, repeat_pool = run(factor_cache, [0, 1, 2, 3], [10, 20], pool_factor=2)
    assert cold_pool["vision_pool_async_submits"] == 2
    assert repeat_pool["vision_pool_async_submits"] == 0
    assert repeat_pool["vision_pool_wait_ms"] == 0

    prefetch_cache = Cache()
    _, _, _, first_timing = run(
        prefetch_cache,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [10, 11, 20, 21, 30, 31, 40, 41, 50, 51],
        prefetch=True,
    )
    _, next_hits, next_misses, next_timing = run(
        prefetch_cache,
        [2, 3, 4, 5, 6, 7, 9, 9, 10, 11],
        [20, 21, 30, 31, 40, 41, 51, 51, 60, 61],
        prefetch=True,
    )
    assert (next_hits, next_misses) == (4, 1)
    assert first_timing["vision_prefetches"] == 1
    assert next_timing["vision_prefetches"] == 1
    assert torch.equal(
        prefetch_cache.values[(9, 9, 1, 1)].base_embed,
        torch.tensor([[51, 51]], dtype=torch.float16),
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


def test_generate_verifies_history_draft_and_falls_back(monkeypatch):
    original_to = torch.Tensor.to
    monkeypatch.setattr(
        torch.Tensor, "to",
        lambda tensor, *args, **kwargs: tensor
        if args and args[0] == "rpu" else original_to(tensor, *args, **kwargs),
    )
    api = types.ModuleType("rpu_backend.api")
    api.greedy_token_ids = lambda logits: logits[:, -1, :]
    monkeypatch.setitem(sys.modules, "rpu_backend.api", api)

    class Cache:
        max_seq_len = 32

        def __init__(self):
            self.rewinds = []

        def reset_to_position(self, position):
            self.rewinds.append(position)

    class Graphs:
        clears = 0

        def clear(self):
            self.clears += 1

    class Model:
        def __init__(self, first_output, fallback):
            self.outputs = [first_output]
            if fallback:
                self.outputs.extend([token] for token in draft)
            self.calls = []
            self.graphs = Graphs()
            self.model = types.SimpleNamespace(
                language_model=types.SimpleNamespace(
                    _rpu_text_graph_cache=self.graphs,
                ),
            )

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            output = self.outputs.pop(0)
            return types.SimpleNamespace(
                logits=torch.tensor(output).reshape(1, len(output), 1),
            )

    draft = [153740, 153743, 151680, 152090, 152327]
    for first_output, expected_accept, expected_calls in (
        (draft, True, 1),
        ([0] * 5, False, 6),
    ):
        model = Model(first_output, fallback=not expected_accept)
        cache = Cache()
        tokens, _, decode_ms, accepted = BENCHMARK._generate_five(
            model,
            torch.tensor([[10, 11, 12]]),
            {
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "mm_token_type_ids": torch.zeros((1, 3), dtype=torch.long),
            },
            cache,
            draft,
        )

        assert tokens == draft and accepted is expected_accept and decode_ms >= 0
        if expected_accept:
            assert decode_ms == 0
        assert cache.rewinds == ([0] if expected_accept else [0, 0])
        assert model.outputs == [] and len(model.calls) == expected_calls
        assert model.graphs.clears == (0 if expected_accept else 2)
        assert model.calls[0]["logits_to_keep"] == 5
        assert model.calls[0]["input_ids"].tolist() == [[10, 11, 12, *draft[:-1]]]
        assert model.calls[0]["attention_mask"].tolist() == [[1] * 7]
        assert model.calls[0]["mm_token_type_ids"].tolist() == [[0] * 7]
