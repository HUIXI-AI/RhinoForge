"""CPU caller checks for the opt-in G05 example input preparation."""
from contextlib import contextmanager
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from test_controlled_example_runners import module

ROOT = Path(__file__).resolve().parents[1]
INPUTS = importlib.import_module("examples.runners.g05.inputs")


@pytest.fixture
def release(monkeypatch, tmp_path):
    """Small release-boundary doubles; IDs come from text, never the requested T."""
    class Tokenizer:
        template = "prefix <command_text_!_200> suffix"
        cot_mode = "none"
        hardcode_instruction = "original " * 32

        def __call__(self, sample):
            instruction = self.hardcode_instruction or sample["instruction"]
            return {"samples": {"command_text": instruction, "template": self.template}}

    class Processor:
        def __init__(self):
            self.tokenizer = Tokenizer()
            self.instruction_override = "original " * 32

        def preprocess(self, obs):
            value = self.tokenizer({"instruction": self.instruction_override or obs["task"]})
            value["samples"] = [value["samples"]]
            value["pixel_values"] = copy.deepcopy(obs["pixel_values"])
            value["proprio"] = torch.ones(1, 27)
            if "prev_action" in obs:
                value.update(prev_action=obs["prev_action"]["joints"].unsqueeze(0),
                             inference_interval=torch.tensor([obs["inference_interval"]]),
                             inference_delay=torch.tensor([obs["inference_delay"]]))
            return value

    class ContextProcessor:
        action = False
        calls = []
        def _prepare_segments(self, samples, **kwargs):
            return [[SimpleNamespace(processor_key="action" if self.action else "text", sample_key="text")]]

        def preprocess(self, samples, **kwargs):
            assert kwargs["control_flag"] == "context_only" and not kwargs["training"]
            assert "action" not in samples[0]
            sample = samples[0]
            limit = 256 if "_256>" in sample["template"] else 200
            words = sample["command_text"].split()[:limit]
            # Twenty real template tokens, plus a distinct real token per word.
            text_ids = [9] * 20 + [sum(word.encode()) for word in words]
            self.calls.append((sample["command_text"], limit))
            ids = torch.tensor([[1] * 192 + [2] + text_ids], dtype=torch.int64)
            mask = torch.tensor([[1] * 192 + [2] + [4] * len(text_ids)], dtype=torch.int64)
            torch.rand(1)  # Counting must preserve caller RNG state.
            return ids, None, mask

    class ReplayRunner:
        def __init__(self, policy, processor, **kwargs):
            self.processor = processor
            self.seed = kwargs["seed"]
        def _preprocess_batch(self, obs):
            return self.processor.preprocess(obs)

    module(monkeypatch, "vla_rtc.replay", ReplayRunner=ReplayRunner)
    module(monkeypatch, "vla_rtc.models.g05.g05_policy", _ensure_sample_list=lambda x: x)
    processor = Processor()
    policy = SimpleNamespace(processor=ContextProcessor(),
        model=SimpleNamespace(vlm=SimpleNamespace(_rpu_qwen3_5=SimpleNamespace(
            max_seq_len=16384, prefill_capability_envelope=(16512, 256)))))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"model")
    manifest = {"config.yaml": {"sha256": "config"}, "dataset_stats.json": {"sha256": "stats"},
                "model.pt": {"sha256": "weights"}}
    observation = {"task": "saved task", "coarse_task": "saved task", "pixel_values": {
        name: torch.ones(1, 1, 3, 2, 2) for name in ("face", "left", "right")}}
    golden = {"schema_version": 2, "checkpoint": {"size": 5, "config_sha256": "config", "dataset_stats_sha256": "stats"},
              "steps": [{"obs": copy.deepcopy(observation), "final_action": {"joints": torch.ones(32, 27)}}
                        for _ in range(2)]}
    original_ids, original_types, _ = INPUTS.encode_batch(policy, processor.preprocess(observation))
    for row in golden["steps"]:
        row.update(input_ids=original_ids, attention_mask=original_types)
    path = tmp_path / "golden.pt"
    torch.save(golden, path)
    return SimpleNamespace(policy=policy, processor=processor, checkpoint=checkpoint, manifest=manifest,
                           golden=path, observation=observation, tmp=tmp_path)


def prepare(release, length, name="generated", **options):
    return INPUTS.prepare_inputs(release.policy, release.processor,
        dict(source="synthetic", source_golden=str(release.golden), text_tokens=length, **options),
        {"output_dir": str(release.tmp / name), "seed": 1007}, release.checkpoint, release.manifest)


def test_processor_generated_lengths_preserve_original_restore_state_and_rng(release):
    processor = release.processor
    original_tokenizer = processor.tokenizer
    original_override = processor.instruction_override
    torch.manual_seed(881)
    before = torch.random.get_rng_state().clone()
    original, meta, alternate = prepare(release, 52)
    assert alternate is None and meta["original_instruction_preserved"]
    assert original["samples"][0]["template"] == original_tokenizer.template
    assert original["samples"][0]["command_text"] == original_override
    assert meta["prefix_len"] == 245 and meta["token_counts"] == {
        "image": 192, "state": 1, "padding": 0, "action": 0, "text": 52}
    assert tuple(original["prev_action"].shape) == (1, 32, 27)
    assert meta["seed"] == 1007 and meta["source_step"] == 1
    for length in (32, 64, 256):
        batch, meta, _ = prepare(release, length, name=f"text{length}")
        ids, _, actual = INPUTS.encode_batch(release.policy, batch, length)
        assert actual["text_tokens"] == length and meta["prefix_len"] == length + 193
        assert meta["actual_token_ids"] == ids.tolist()
        assert not meta["original_instruction_preserved"]
        assert "_256>" in batch["samples"][0]["template"]
        assert meta["cpu_reference"] == "not_run" and meta["task_quality"] == "not_evaluated"
        json.dumps(meta, allow_nan=False)
    assert processor.tokenizer is original_tokenizer
    assert processor.instruction_override == original_override
    assert original_tokenizer.hardcode_instruction == original_override
    assert torch.equal(before, torch.random.get_rng_state())
    with pytest.raises(ValueError, match="actual text_tokens"):
        prepare(release, 64, instruction="too short")
    assert processor.tokenizer is original_tokenizer and processor.instruction_override == original_override
    assert original_tokenizer.hardcode_instruction == original_override


def test_saved_wrapper_reencodes_and_rejects_changed_content_seed_or_ids(release):
    batch, generated, _ = prepare(release, 64)
    path = release.tmp / "generated/input_batch.pt"
    options = {"source": "batch", "batch": str(path), "text_tokens": 64}
    run = {"output_dir": str(release.tmp / "replay"), "seed": 1007}
    _, replayed, _ = INPUTS.prepare_inputs(release.policy, release.processor, options, run,
                                           release.checkpoint, release.manifest)
    assert replayed["content_sha256"] == generated["content_sha256"]
    assert replayed["actual_token_ids"] == generated["actual_token_ids"]
    pristine = torch.load(path, weights_only=True)
    for corruption in ("image", "ids", "seed", "checkpoint"):
        broken = copy.deepcopy(pristine)
        if corruption == "image": broken["batch"]["pixel_values"]["face"].add_(1)
        if corruption == "ids": broken["input_ids"][0, -1] += 1
        if corruption == "seed": broken["seed"] += 1
        if corruption == "checkpoint": broken["checkpoint_sha256"]["model.pt"] = "other"
        torch.save(broken, path)
        with pytest.raises(ValueError, match="saved G05 batch"):
            INPUTS.prepare_inputs(release.policy, release.processor, options, run, release.checkpoint, release.manifest)
    # Existing bare tensors remain a supported source and are measured afresh.
    torch.save(batch, path)
    _, bare, _ = INPUTS.prepare_inputs(release.policy, release.processor, options, run,
                                       release.checkpoint, release.manifest)
    assert bare["source_kind"] == "caller_preprocessed_batch"
    options["text_tokens"] = 52
    with pytest.raises(ValueError, match="actual processor count"):
        INPUTS.prepare_inputs(release.policy, release.processor, options, run, release.checkpoint, release.manifest)


def test_invalid_context_and_golden_fail_before_input_publish(release):
    batch, _, _ = prepare(release, 52)
    invalid = copy.deepcopy(batch)
    invalid["pixel_values"].pop("left")
    with pytest.raises(ValueError, match="3"):
        INPUTS.encode_batch(release.policy, invalid)
    invalid = copy.deepcopy(batch)
    invalid["inference_delay"] = torch.tensor([5.5])
    with pytest.raises(ValueError, match="interval/delay"):
        INPUTS.encode_batch(release.policy, invalid)
    release.policy.processor.action = True
    with pytest.raises(ValueError, match="action tokens"):
        INPUTS.encode_batch(release.policy, batch)
    release.policy.processor.action = False
    source = torch.load(release.golden, weights_only=True)
    source["checkpoint"]["config_sha256"] = "other"
    torch.save(source, release.golden)
    with pytest.raises(ValueError, match="verified release"):
        prepare(release, 64, "bad")
    assert not (release.tmp / "bad/input_batch.pt").exists()


def test_example_preserves_installed_capacity_and_checks_prefix(monkeypatch):
    state = SimpleNamespace(handle=44, max_seq_len=16384,
                            prefill_capability_envelope=(16512, 256))
    policy = SimpleNamespace(model=SimpleNamespace(vlm=SimpleNamespace(_rpu_qwen3_5=state)))
    setter = Mock()
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_chunk_envelope", setter, raising=False)
    for prefix in (245, 449):
        metadata = {"prefix_rows": prefix}
        INPUTS.configure_text_envelope(policy, metadata)
        assert metadata["prefill_capability_envelope"] == [16512, 256]
    setter.assert_not_called()
    assert state.prefill_capability_envelope == (16512, 256)
    state.max_seq_len = 256
    with pytest.raises(ValueError, match="installed KV capacity"):
        INPUTS.configure_text_envelope(policy, {"prefix_rows": 449})
    with pytest.raises(ValueError, match="actual T32"):
        INPUTS.configure_text_envelope(policy, {"prefix_rows": 450})


def test_real_runner_opt_in_consumes_prepared_inputs_and_clones_each_inference(monkeypatch, release):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    policy = release.policy
    policy._rpu_g05_fm_only_customer_release = True
    policy.eval = lambda: policy
    def predict(batch):
        result = batch["proprio"].clone()
        batch["proprio"].zero_()
        return {"action": result}
    policy.predict_action = predict
    patch = Mock()
    module(monkeypatch, "rpu_backend.adapters.g05", patch_g05_policy_for_rpu=patch,
           configure_g05_vision_camera_batch=Mock())
    @contextmanager
    def transaction(root): yield lambda: None
    module(monkeypatch, "rpu_backend.adapters.g05.release_replay", _checkpoint_manifest=lambda p: release.manifest,
           _release_import_transaction=transaction, _resolve_release_paths=lambda *a: (release.tmp, release.checkpoint))
    module(monkeypatch, "vla_rtc.loader", __file__=str(release.tmp / "vla_rtc/loader.py"),
           load_for_eval=lambda *a, **kw: (policy, release.processor))
    monkeypatch.setattr(importlib.import_module("examples.runners.g05.runner"), "_checkpoint", lambda _: release.checkpoint)
    config = {"input": {"source": "synthetic", "source_golden": str(release.golden), "text_tokens": 52},
              "run": {"output_dir": str(release.tmp / "caller"), "seed": 1007}, "g05": {}}
    runner = SimpleNamespace(source_config=config, owner=None, _g05_prepare_inputs=True,
                             _g05_patch_options={"weight_dtype": "w8a16"})
    infer = importlib.import_module("examples.runners.g05.runner")._g05(runner)
    first, second = infer(), infer()
    assert torch.equal(first, torch.ones(1, 27)) and torch.equal(first, second)
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(runner._g05_batch["proprio"], first)
    alternative = copy.deepcopy(runner._g05_batch)
    alternative["proprio"].fill_(2)
    assert torch.equal(runner._g05_infer_batch(alternative), torch.full((1, 27), 2.))
    assert torch.equal(alternative["proprio"], torch.full((1, 27), 2.))
    assert runner._g05_input_metadata["text_tokens"] == 52
    assert runner._g05_alternate_batch is None and runner.owner is policy
    patch.assert_called_once_with(policy, rpu_execution=None, weight_dtype="w8a16")
    # A CPU preparation error must not publish an owner to device-only cleanup.
    config["input"]["instruction"] = "too short"
    invalid_runner = SimpleNamespace(source_config=config, owner=None, _g05_prepare_inputs=True)
    patch.reset_mock()
    with pytest.raises(ValueError, match="actual text_tokens"):
        importlib.import_module("examples.runners.g05.runner")._g05(invalid_runner)
    assert invalid_runner.owner is None
    patch.assert_not_called()


def test_two_camera_context_counts_actual_images_and_rejects_reordered_batch(release):
    policy = release.policy
    policy.num_input_images = 2
    policy._rpu_g05_camera_names = ("head_rgb", "left_wrist_rgb")
    def context(samples, **kwargs):
        words = samples[0]["command_text"].split()[:256]
        ids = torch.tensor([[1] * 128 + [2] + [9] * 20 + [sum(w.encode()) for w in words]])
        mask = torch.tensor([[1] * 128 + [2] + [4] * (20 + len(words))])
        return ids, None, mask
    policy.processor.preprocess = context
    batch = release.processor.preprocess(release.observation)
    values = list(batch["pixel_values"].values())
    batch["pixel_values"] = dict(zip(policy._rpu_g05_camera_names, values[:2]))
    _, _, meta = INPUTS.encode_batch(policy, batch, 52)
    assert meta["cameras"] == 2 and meta["prefix_rows"] == 181
    assert meta["token_counts"]["image"] == 128
    assert meta["camera_names"] == ["head_rgb", "left_wrist_rgb"]
    policy.model = SimpleNamespace(vlm=SimpleNamespace(_rpu_qwen3_5=SimpleNamespace(
        max_seq_len=16384, prefill_capability_envelope=(16512, 256))))
    INPUTS.configure_text_envelope(policy, meta)
    assert meta["prefill_capability_envelope"] == [16512, 256]
    with pytest.raises(ValueError, match="actual T32"):
        INPUTS.configure_text_envelope(policy, dict(meta, prefix_rows=245))
    batch["pixel_values"] = dict(reversed(list(batch["pixel_values"].items())))
    with pytest.raises(ValueError, match="camera order"):
        INPUTS.encode_batch(policy, batch, 52)


def test_original_three_camera_policy_rejects_saved_batch_camera_order_drift(release):
    release.policy.num_input_images = 3
    batch = release.processor.preprocess(release.observation)
    batch["pixel_values"] = dict(zip(("head_rgb", "left_wrist_rgb", "right_wrist_rgb"),
                                     batch["pixel_values"].values()))
    INPUTS.encode_batch(release.policy, batch, 52)
    batch["pixel_values"] = dict(reversed(list(batch["pixel_values"].items())))
    with pytest.raises(ValueError, match="camera order"):
        INPUTS.encode_batch(release.policy, batch, 52)
