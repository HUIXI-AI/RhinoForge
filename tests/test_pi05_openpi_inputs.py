"""openpi pi05 raw-observation inputs: host transforms, checkpoint contract and the runner path."""
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from examples.runners.pi05 import openpi_inputs as openpi
from examples.runners.pi05 import optimized as runner

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


class Tokenizer:
    """Stands in for the PaliGemma SentencePiece model: BOS id 2, then one id per character."""

    def encode(self, text, add_bos=False):
        return ([2] if add_bos else []) + [3 + ord(char) % 997 for char in text]


def _base_config(action=32):
    """pi05_base config.json features: three 224 camera slots, state/action widths."""
    return {"max_action_dim": 32, "image_resolution": [224, 224], "chunk_size": 50,
            "input_features": {**{f"observation.images.{key}": {"type": "VISUAL", "shape": [3, 224, 224]}
                                  for key in CAMERAS},
                               "observation.state": {"type": "STATE", "shape": [32]}},
            "output_features": {"action": {"type": "ACTION", "shape": [action]}}}


def _raw_inputs(tmp_path, *, state=(0.25, -0.5, 40.0), mask=(True, True, False), size=(848, 480)):
    frames = []
    for index in range(3):
        path = tmp_path / f"cam{index}.npy"
        np.save(path, np.full((size[1], size[0], 3), 60 * (index + 1), dtype=np.uint8))
        frames.append(str(path))
    stats = {"norm_stats": {
        "state": {"q01": [0.0, -1.0, 0.0, 0.0], "q99": [1.0, 1.0, 100.0, 1.0]},
        "actions": {"q01": [-0.1, -0.2, 0.0], "q99": [0.1, 0.2, 100.0]}}}
    (tmp_path / "norm_stats.json").write_text(json.dumps(stats))
    inputs = {"images": frames, "frame_size": list(size), "instruction": " pick_up the\nparcel ", "state": list(state),
              "norm_stats": str(tmp_path / "norm_stats.json"), "tokenizer": str(tmp_path / "tokenizer.model"),
              "cameras": 3}
    if mask is not None:
        inputs["delta_action_mask"] = list(mask)
    return inputs


def test_resize_with_pad_keeps_aspect_ratio_and_pads_black():
    frame = np.full((480, 848, 3), 200, dtype=np.uint8)
    resized = openpi.resize_with_pad(frame, 224, 224)
    # 848x480 scales to 224x126 (int truncation as in openpi), centred with 49/49 black rows.
    assert resized.shape == (224, 224, 3) and resized.dtype == np.uint8
    assert (resized[:49] == 0).all() and (resized[175:] == 0).all()
    assert (resized[49:175] == 200).all()
    same = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    assert np.array_equal(openpi.resize_with_pad(same), same)
    with pytest.raises(ValueError, match="uint8"):
        openpi.resize_with_pad(frame.astype(np.float32))


def test_simulation_frames_resize_to_the_same_model_image():
    # 640x480 (Kuavo simulation) scales to 224x168, centred with 28/28 black rows.
    frame = np.full((480, 640, 3), 90, dtype=np.uint8)
    resized = openpi.resize_with_pad(frame, 224, 224)
    assert (resized[:28] == 0).all() and (resized[196:] == 0).all() and (resized[28:196] == 90).all()
    assert openpi.resize_geometry(640, 480) == {"frame_wh": [640, 480], "model_wh": [224, 224],
                                                "resized_wh": [224, 168], "pad_top": 28, "pad_left": 0}
    assert openpi.resize_geometry(848, 480)["resized_wh"] == [224, 126]


def test_frames_of_another_size_are_refused(tmp_path):
    inputs = _raw_inputs(tmp_path)
    loaded = openpi.load_raw_inputs(inputs, processor=Tokenizer())
    small = [np.zeros((224, 224, 3), dtype=np.uint8)] + loaded["frames"][1:]
    with pytest.raises(ValueError, match=r"base_0_rgb frame is \[224, 224, 3\].*\[480, 848, 3\]"):
        openpi.batch_from_frames(small, inputs["state"], "pick", loaded["stats"], Tokenizer(),
                                 text_tokens=128, frame_size=[848, 480])
    with pytest.raises(ValueError, match="640x480"):
        openpi.build_batch({**inputs, "frame_size": [640, 480]}, text_tokens=128, processor=Tokenizer())
    (tmp_path / "sim").mkdir()
    sim = _raw_inputs(tmp_path / "sim", size=(640, 480))
    batch, record = openpi.build_batch(sim, text_tokens=128, processor=Tokenizer())
    assert record["resize_geometry"]["resized_wh"] == [224, 168]
    assert batch["observation.images.base_0_rgb"].shape == (1, 3, 224, 224)


def test_resize_with_pad_preserves_linear_ramps_on_both_axes():
    # The triangle filter keeps a linear ramp linear away from the borders, so each axis
    # (and the channel order) must land where its sample positions say.
    rows = np.broadcast_to(np.arange(240, dtype=np.uint8)[:, None, None], (240, 424, 3))
    cols = np.broadcast_to(np.arange(212, dtype=np.uint8)[None, :, None], (120, 212, 3))
    frame = np.ascontiguousarray(rows) + np.array([0, 5, 10], dtype=np.uint8)
    out = openpi.resize_with_pad(frame)[49:175]  # 424x240 -> 224x126, 49 black rows above
    sample = (np.arange(126) + 0.5) * 240 / 126 - 0.5
    expected = np.round(sample[:, None, None] + [0, 5, 10])
    assert np.abs(out[3:-3].astype(int) - expected[3:-3]).max() <= 1
    assert np.abs(out.astype(int) - out[:, :1]).max() <= 1  # float32 weight sums are not exactly 1
    out = openpi.resize_with_pad(np.ascontiguousarray(cols))[49:175]  # 212x120 -> 224x126 (upscale)
    sample = np.clip((np.arange(224) + 0.5) * 212 / 224 - 0.5, 0, 211)
    assert np.abs(out[0, :, 0].astype(int) - np.round(sample)).max() <= 1
    assert np.abs(out.astype(int) - out[:1]).max() <= 1


def test_prompt_discretizes_the_normalized_state_and_pads_tokens():
    prompt = openpi.pi05_prompt(" pick_up\nbox ", np.array([-1.0, 1.0, 0.0]))
    assert prompt == "Task: pick up box, State: 0 255 128;\nAction: "
    tokens, mask = openpi.tokenize(Tokenizer(), "ab", 5)
    assert tokens == [2, 3 + ord("a"), 3 + ord("b"), 0, 0] and mask == [True, True, True, False, False]
    with pytest.raises(ValueError, match="text_tokens=2 is too small"):
        openpi.tokenize(Tokenizer(), "ab", 2)


def test_build_batch_matches_the_runner_layout(tmp_path):
    inputs = _raw_inputs(tmp_path)
    batch, record = openpi.build_batch(inputs, text_tokens=128, processor=Tokenizer())
    for index, key in enumerate(CAMERAS):
        image = batch[f"observation.images.{key}"]
        assert image.shape == (1, 3, 224, 224) and image.dtype == torch.float32
        assert image.max().item() == pytest.approx(60 * (index + 1) / 255.0)
        assert batch[f"observation.images.{key}_padding_mask"].tolist() == [True]
    # Quantile normalization uses only the state's own dims of the longer statistics.
    expected = (np.array([0.25, -0.5, 40.0]) - [0.0, -1.0, 0.0]) / (np.array([1.0, 2.0, 100.0]) + 1e-6) * 2 - 1
    assert torch.allclose(batch["observation.state"], torch.tensor(expected[None], dtype=torch.float32))
    tokens, mask = batch["observation.language.tokens"], batch["observation.language.attention_mask"]
    assert tokens.shape == mask.shape == (1, 128) and tokens.dtype == torch.int64 and mask.dtype == torch.bool
    valid = len(record["prompt"]) + 1
    assert record["valid_text_tokens"] == int(mask.sum()) == valid and not mask[0, valid:].any()
    assert record["prompt"].startswith("Task: pick up the parcel, State: ")
    assert record["frames"]["base_0_rgb"]["shape"] == [480, 848, 3]
    with pytest.raises(ValueError, match="too small"):
        openpi.build_batch(inputs, text_tokens=32, processor=Tokenizer())


def test_physical_actions_unnormalize_then_add_the_state_on_delta_dims(tmp_path):
    inputs = _raw_inputs(tmp_path)
    actions = torch.zeros(1, 50, 4)
    actions[..., 3] = 0.7  # beyond the action statistics: passes through
    physical = openpi.physical_actions(actions, inputs)
    middle = np.array([0.0, 0.0, 50.0 + 0.5e-6])
    assert torch.allclose(physical[0, :, :3], torch.tensor(middle + [0.25, -0.5, 0.0], dtype=torch.float32))
    assert torch.allclose(physical[0, :, 3], torch.tensor(0.7))
    stats = json.loads(Path(inputs["norm_stats"]).read_text())
    del stats["norm_stats"]["actions"]
    Path(inputs["norm_stats"]).write_text(json.dumps(stats))
    assert openpi.physical_actions(actions, inputs) is None


@pytest.mark.parametrize("change,message", [
    ({"batch": "/batch.pt"}, "already owns"),
    ({"images": ["/a.png", "/b.png"]}, "three frame paths"),
    ({"frame_size": [224, 224]}, "frame_size must be the camera frame size"),
    ({"frame_size": [480, 848]}, "\\[848, 480\\] or \\[640, 480\\]"),
    ({"frame_size": "848x480"}, "frame_size"),
    ({"cameras": 2}, "exactly three cameras"),
    ({"instruction": "  "}, "instruction"),
    ({"state": [0.0, float("nan")]}, "finite"),
    ({"state": []}, "1..32"),
    ({"tokenizer": ""}, "tokenizer"),
    ({"delta_action_mask": [True, False, True, True]}, "no longer than"),
    ({"delta_action_mask": [1, 0]}, "boolean"),
])
def test_raw_input_refusals(tmp_path, change, message):
    inputs = {**_raw_inputs(tmp_path), **change}
    with pytest.raises(ValueError, match=message):
        openpi.validate_input_config(inputs)


def test_raw_and_batch_modes_are_exclusive_and_named():
    assert openpi.validate_input_config({"batch": "/batch.pt"}) is False
    with pytest.raises(ValueError, match="missing"):
        openpi.validate_input_config({"images": ["/a", "/b", "/c"]})


def _example(tmp_path, **inputs):
    return {"pi05": {"precision": "fp16"}, "run": {"warmup": 1, "runs": 1},
            "input": {"checkpoint": "/ckpt", "text_tokens": 128, **_raw_inputs(tmp_path), **inputs}}


@pytest.mark.parametrize("change,message", [
    ({"run": {"reference": "saved"}}, "unsupported keys"),
    ({"input": {"action_dim": 2}}, "longer than \\[input\\].action_dim"),
    ({"input": {"action_dim": 33}}, "1..32"),
    ({"input": {"batch": "/batch.pt"}}, "already owns"),
])
def test_example_config_refuses_raw_input_conflicts(tmp_path, change, message):
    config = _example(tmp_path)
    for section, values in change.items():
        config[section] = {**config[section], **values}
    with pytest.raises(ValueError, match=message):
        runner.validate_example_config(config)


def test_checkpoint_contract_keeps_libero_256_and_admits_declared_224(tmp_path):
    libero = deepcopy(_base_config(7))
    libero["input_features"] = {"observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
                                "observation.images.empty_camera_0": {"type": "VISUAL", "shape": [3, 224, 224]}}
    (tmp_path / "config.json").write_text(json.dumps(libero))
    assert runner.checkpoint_contract(tmp_path) == {"action_dim": 7, "image_shapes": ((1, 3, 256, 256),)}
    (tmp_path / "config.json").write_text(json.dumps(_base_config(16)))
    contract = runner.checkpoint_contract(tmp_path, 16)
    assert contract == {"action_dim": 16, "image_shapes": ((1, 3, 256, 256), (1, 3, 224, 224))}
    with pytest.raises(ValueError, match="action_dim=8"):
        runner.checkpoint_contract(tmp_path, 8)
    batch = {f"observation.images.{key}": torch.zeros(1, 3, 224, 224) for key in CAMERAS}
    runner.validate_image_shapes(batch, contract["image_shapes"])
    batch["observation.images.base_0_rgb"] = torch.zeros(1, 3, 256, 256)
    with pytest.raises(ValueError, match="every camera image"):
        runner.validate_image_shapes(batch, contract["image_shapes"])
    with pytest.raises(ValueError, match="every camera image"):
        runner.validate_image_shapes(batch, ((1, 3, 256, 256),))


def test_checkpoint_view_rewrites_widths_and_shares_only_a_current_remap(tmp_path):
    source = tmp_path / "base"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(_base_config()))
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "policy_preprocessor.json").write_text("{}")
    (source / "model_remapped.safetensors").write_bytes(b"remap")
    os.utime(source / "model.safetensors", ns=(1_000, 1_000))
    view = openpi.checkpoint_view(source, tmp_path / "a8", action_dim=8)
    assert view["linked"] == ["model.safetensors", "model_remapped.safetensors", "policy_preprocessor.json"]
    config = json.loads((tmp_path / "a8/config.json").read_text())
    assert config["output_features"]["action"]["shape"] == [8]
    assert config["input_features"]["observation.state"]["shape"] == [8]
    assert json.loads((source / "config.json").read_text())["output_features"]["action"]["shape"] == [32]
    assert (tmp_path / "a8/model.safetensors").resolve() == (source / "model.safetensors").resolve()
    # A weight newer than the remap: the loader must rebuild it in the view, not reuse the stale one.
    os.utime(source / "model.safetensors", ns=(10**19, 10**19))
    stale = openpi.checkpoint_view(source, tmp_path / "a16", action_dim=16)
    assert "model_remapped.safetensors" not in stale["linked"]
    with pytest.raises(FileExistsError):
        openpi.checkpoint_view(source, tmp_path / "a8", action_dim=8)
    with pytest.raises(ValueError, match="1..32"):
        openpi.checkpoint_view(source, tmp_path / "a40", action_dim=40)


def test_raw_inputs_run_through_the_example_entry(monkeypatch, tmp_path):
    """TOML [input] -> openpi batch -> policy -> [1,50,D] report and physical actions."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(_base_config(3)))
    config = {"pi05": {"precision": "fp16"}, "cpu": {"threads": 1},
              "input": {"checkpoint": str(checkpoint), "text_tokens": 128, "action_dim": 3, **_raw_inputs(tmp_path)},
              "run": {"warmup": 1, "runs": 2, "output_dir": str(tmp_path / "result")}}
    seen, phases, stats_reads = [], {}, []

    class Policy:
        _lerobot_policy = SimpleNamespace(config=SimpleNamespace(chunk_size=50, image_features={}))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            pass

        def prepare_graphs(self, batch, **kwargs):
            return {"phase": "READY"}

        def predict_action_chunk(self, batch, *, num_steps):
            seen.append(batch)
            phases.update(vision_ms=1.0, prefill_ms=1.0, denoise_ms=1.0)
            return torch.zeros(1, 50, 3)

        def close(self):
            pass

    stubs = {"rpu_backend": ModuleType("rpu_backend"), "rpu_backend.api": SimpleNamespace(Pi05Policy=Policy),
             "rpu_backend.model_registry": SimpleNamespace(model_path=lambda alias: checkpoint)}
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)
    load_norm_stats = openpi.load_norm_stats
    monkeypatch.setattr(openpi, "load_norm_stats", lambda path: stats_reads.append(path) or load_norm_stats(path))
    monkeypatch.setattr(openpi, "load_tokenizer", lambda path: Tokenizer())
    monkeypatch.setattr(runner, "bind_cold_environment", lambda env: None)
    monkeypatch.setattr(runner, "profile_libraries", lambda: ({}, {}))
    monkeypatch.setattr(runner, "runtime_payloads", lambda libraries: {})
    monkeypatch.setattr(runner, "configure_text_profile", lambda *args: {})
    monkeypatch.setattr(runner, "validate_camera_batch", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "stage_timings", lambda policy: nullcontext(phases))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)
    monkeypatch.setitem(runner._CPU, "configure_cpu_threads", lambda threads: {"requested_threads": threads})
    assert runner.run_example_config(config) == 0
    report = json.loads((tmp_path / "result/report.json").read_text())
    assert report["action_dim"] == 3 and report["batch"] is None
    assert report["inputs"]["text_tokens"] == 128 and report["inputs"]["state_dim"] == 3
    assert report["inputs"]["resize_geometry"]["resized_wh"] == [224, 126]
    assert report["physical_actions"]["finite"] and report["physical_actions"]["delta_action_mask"] == [True, True, False]
    assert seen and seen[0]["observation.images.base_0_rgb"].shape == (1, 3, 224, 224)
    # Every measured request starts from the native frames: a fresh batch per call, and the
    # resize/prompt and physical-action transforms are timed around the policy phases.
    assert len(seen) == 3 and len({id(batch) for batch in seen}) == 3
    assert all(torch.equal(batch["observation.images.base_0_rgb"], seen[0]["observation.images.base_0_rgb"])
               for batch in seen)
    assert report["timed_phases"] == ["total_ms", "preprocess_ms", "vision_ms", "prefill_ms", "denoise_ms",
                                      "postprocess_ms"]
    assert report["timing_scope"] == runner.RAW_TIMING_SCOPE
    assert report["source_image_sizes_wh"] == [[848, 480]] * 3
    timing = {item["metric"]: item for item in report["timing"]}
    assert {"preprocess_latency", "postprocess_latency"} <= set(timing)
    assert timing["total_latency"]["boundary"] == runner.RAW_TIMING_SCOPE
    # Files are read once before the load, never inside the timed loop.
    assert stats_reads == [config["input"]["norm_stats"]]
    physical = torch.load(tmp_path / "result/actions_physical.pt")
    assert physical.shape == (1, 50, 3)
    assert torch.allclose(physical[0, 0], torch.tensor([0.25, -0.5, 50.0 + 0.5e-6]))
    # The same TOML with a 7-wide expectation is refused before the policy loads.
    config["input"]["action_dim"] = 7
    config["input"]["delta_action_mask"] = [True]
    seen.clear()
    with pytest.raises(ValueError, match="action_dim=7"):
        runner.run_example_config(config)
    assert not seen


def _dense_reference_resize_with_pad(image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    cur_h, cur_w = image.shape[:2]
    if (cur_h, cur_w) == (height, width):
        return image.copy()
    ratio = max(cur_w / width, cur_h / height)
    new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
    x = np.matmul(openpi._resize_weights(cur_h, new_h).T, image.astype(np.float32).reshape(cur_h, -1))
    x = x.reshape(new_h, cur_w, 3).transpose(0, 2, 1).reshape(new_h * 3, cur_w)
    x = np.matmul(x, openpi._resize_weights(cur_w, new_w)).reshape(new_h, 3, new_w).transpose(0, 2, 1)
    x = np.clip(np.round(x), 0, 255).astype(np.uint8)
    pad_h, extra_h = divmod(height - new_h, 2)
    pad_w, extra_w = divmod(width - new_w, 2)
    return np.pad(x, ((pad_h, pad_h + extra_h), (pad_w, pad_w + extra_w), (0, 0)), constant_values=0)


def test_resize_and_batch_are_bit_identical_to_dense_reference():
    rng = np.random.default_rng(20261112)
    stats = {"state": {"q01": [-1.0, -2.0, 0.0], "q99": [1.0, 2.0, 10.0]}}
    prev_threads = torch.get_num_threads()
    for width, height in ((848, 480), (640, 480), (224, 224), (212, 120)):
        frames = [rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8) for _ in range(3)]
        frames[0].flags.writeable = False  # PIL read-only frame contract
        frames[1] = frames[1][:, ::-1][:, ::-1]  # non-contiguous view contract
        for frame in frames:
            got = openpi.resize_with_pad(frame, 224, 224)
            expected = _dense_reference_resize_with_pad(frame, 224, 224)
            assert np.array_equal(got, expected)

        if (width, height) in openpi.FRAME_SIZES:
            batch, _ = openpi.batch_from_frames(
                frames, [0.1, -0.4, 3.0], "flip parcel", stats, Tokenizer(),
                text_tokens=128, frame_size=[width, height],
            )
            assert torch.get_num_threads() == prev_threads
            for key, frame in zip(CAMERAS, frames):
                ref_u8 = _dense_reference_resize_with_pad(frame, 224, 224)
                ref_img = (
                    torch.from_numpy(ref_u8).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
                ).contiguous()
                got_img = batch[f"observation.images.{key}"]
                assert got_img.is_contiguous()
                assert torch.equal(got_img.view(torch.uint8), ref_img.view(torch.uint8))

