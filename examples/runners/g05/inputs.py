"""CPU input preparation for the opt-in G05 measurement example.

Uses the installed customer processor; it neither invents token IDs nor runs a
CPU model. Synthetic means task text on the saved real K1 sensor/RTC input.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

SCHEMA = "g05_example_input_v1"


def _digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def content_digest(batch, seed):
    import torch

    checksum = hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            checksum.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
            checksum.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                checksum.update(json.dumps(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            checksum.update(json.dumps([type(value).__name__, len(value)]).encode())
            for item in value:
                visit(item)
        else:
            checksum.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    visit({"batch": batch, "seed": seed})
    return checksum.hexdigest()


def _cpu_tree(value):
    import torch

    if torch.is_tensor(value):
        if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            raise ValueError("G05 example inputs must contain finite CPU tensors")
    elif isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("G05 example mapping keys must be strings")
        for item in value.values():
            _cpu_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _cpu_tree(item)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError("G05 example inputs must be tensor-only safe mappings")


def encode_batch(policy, batch, target=None, *, searching=False):
    """Measure the actual unpadded, action-free release context on CPU."""
    import torch
    from vla_rtc.models.g05.g05_policy import _ensure_sample_list

    if not isinstance(batch, dict) or not {"samples", "pixel_values"} <= batch.keys():
        raise ValueError("G05 batch requires samples and pixel_values")
    _cpu_tree(batch)
    images = batch["pixel_values"]
    cameras = int(getattr(policy, "num_input_images", 3))
    selected_names = getattr(policy, "_rpu_g05_camera_names", None)
    if selected_names is None and hasattr(policy, "num_input_images"):
        selected_names = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")[:cameras]
    if cameras not in (2, 3):
        raise ValueError("G05 K1 camera profile must contain two or three cameras")
    if (selected_names is not None and (not isinstance(images, dict)
            or tuple(images) != tuple(selected_names))):
        raise ValueError("G05 batch camera order differs from the cold camera profile")
    if not isinstance(images, dict) or len(images) != cameras or any(
            not torch.is_tensor(t) or t.ndim != 5 or tuple(t.shape[:3]) != (1, 1, 3)
            for t in images.values()):
        raise ValueError(f"G05 K1 requires {cameras} [1,1,3,H,W] camera tensors")
    samples = copy.deepcopy(_ensure_sample_list(batch["samples"]))
    if len(samples) != 1:
        raise ValueError("G05 example requires batch one")
    proc = policy.processor
    segments = proc._prepare_segments(samples, control_flag="context_only")[0]
    if any(s.processor_key == "action" or s.sample_key == "action" for s in segments):
        raise ValueError("G05 context cannot contain action tokens")
    samples[0].pop("action", None)  # No dead ActionCodec RNG while counting.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        ids, _, mask = proc.preprocess(samples, control_flag="context_only",
                                      device=torch.device("cpu"), training=False)
    if ids.dtype != torch.int64 or ids.ndim != 2 or ids.shape[0] != 1 or mask.shape != ids.shape:
        raise ValueError("release processor produced an invalid context shape")
    counts = {str(kind): int((mask == kind).sum()) for kind in range(7)}
    text = sum(counts[str(kind)] for kind in (4, 5, 6))
    if (counts["1"] != 64 * cameras or counts["2"] != 1 or counts["0"] or counts["3"]
            or sum(counts.values()) != mask.numel() or (not searching and not 32 <= text <= 256)):
        raise ValueError(f"unexpected unpadded G05 K1 context: {counts}")
    if target is not None and (type(target) is not int or text != target):
        raise ValueError(f"text_tokens={target} does not match actual processor count {text}")
    if "prev_action" in batch:
        if not torch.is_tensor(batch["prev_action"]) or tuple(batch["prev_action"].shape) != (1, 32, 27):
            raise ValueError("G05 RTC previous action must be [1,32,27]")
        for key in ("inference_interval", "inference_delay"):
            value = torch.as_tensor(batch.get(key, -1))
            if value.numel() != 1 or value.item() != 5:
                raise ValueError("G05 RTC requires interval/delay 5/5")
    metadata = {"text_tokens": text, "prefix_len": int(mask.numel()), "prefix_rows": int(mask.numel()),
                "token_counts": {"padding": counts["0"], "image": counts["1"], "state": counts["2"],
                                 "action": counts["3"], "text": text}, "token_types": counts,
                "text_length_definition": "TEXT(4)+COT(5)+PRED_TEXT(6), including template/boundary text; no PAD",
                "cameras": cameras, "camera_names": list(images),
                "camera_shapes": {name: list(t.shape) for name, t in images.items()},
                "rtc": "prev_action" in batch}
    return ids.detach().cpu(), mask.detach().cpu(), metadata


class _CommandLimitTokenizer:
    """Same explicit command-only limit used by the existing length benchmark."""
    def __init__(self, wrapped):
        if wrapped.template.count("<command_text_!_200>") != 1 or wrapped.cot_mode != "none":
            raise ValueError("expected exact no-CoT release command template")
        self.wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __call__(self, sample):
        result = self.wrapped(sample)
        nested = result["samples"]
        template = nested["template"]
        if template.count("<command_text_!_200>") != 1:
            raise ValueError("release command template drift")
        nested["template"] = template.replace("<command_text_!_200>", "<command_text_!_256>")
        return result


def _golden(path, checkpoint, manifest):
    import torch

    source = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(source, dict) or source.get("schema_version") != 2
            or not isinstance(source.get("steps"), (list, tuple)) or len(source["steps"]) != 2
            or any(not isinstance(row, dict) or not isinstance(row.get("obs"), dict) for row in source["steps"])):
        raise ValueError("source_golden requires the two-step G05 release observation file")
    recorded = source.get("checkpoint", {})
    if (not isinstance(recorded, dict) or recorded.get("config_sha256") != manifest["config.yaml"]["sha256"] or
            recorded.get("dataset_stats_sha256") != manifest["dataset_stats.json"]["sha256"] or
            recorded.get("size") != (checkpoint / "model.pt").stat().st_size):
        raise ValueError("source_golden does not match the verified release checkpoint")
    return source


def _synthetic(policy, processor, inputs, checkpoint, manifest, seed):
    from vla_rtc.replay import ReplayRunner

    path = Path(inputs["source_golden"]).expanduser()
    source = _golden(path, checkpoint, manifest)
    step = inputs.get("source_step", 1)
    if type(step) is not int or step not in (0, 1):
        raise ValueError("source_step must be 0 or 1")
    target = inputs["text_tokens"]
    if type(target) is not int or not 32 <= target <= 256:
        raise ValueError("text_tokens must be an integer in [32,256]")
    obs = copy.deepcopy(source["steps"][step]["obs"])
    camera_names = getattr(policy, "_rpu_g05_camera_names", None)
    if int(getattr(policy, "num_input_images", 3)) == 2:
        if tuple(camera_names or ()) != ("head_rgb", "left_wrist_rgb"):
            raise ValueError("two-camera G05 input requires the admitted head/left cold profile")
        images = obs.get("images")
        if not isinstance(images, dict) or any(name not in images for name in camera_names):
            raise ValueError("source observation is missing a configured real camera")
        obs["images"] = {name: images[name] for name in camera_names}
    if step:
        obs.update(prev_action=copy.deepcopy(source["steps"][0]["final_action"]),
                   inference_interval=5, inference_delay=5)
    replay = ReplayRunner(policy, processor, device="cpu", use_rtc=True,
                          inference_interval=5, inference_delay=5, seed=seed)
    original_tokenizer = processor.tokenizer
    original_override = processor.instruction_override
    original_hardcode = original_tokenizer.hardcode_instruction
    try:
        def prepare(task, generated):
            processor.instruction_override = None if generated else original_override
            original_tokenizer.hardcode_instruction = None if generated else original_hardcode
            processor.tokenizer = _CommandLimitTokenizer(original_tokenizer) if generated else original_tokenizer
            current = copy.deepcopy(obs)
            if task is not None:
                current.update(task=task, coarse_task=task)
            batch = replay._preprocess_batch(current)
            ids, mask, meta = encode_batch(policy, batch, searching=True)
            return batch, ids, mask, meta
        def find(word):
            # Every candidate is ordinary text encoded by the actual processor.
            # Search token count, never repeat IDs or make padding visible.
            seen = {}
            def measure(n):
                if n not in seen:
                    seen[n] = prepare(" ".join([word] * n), True)
                return seen[n]
            low, high = 0, target + 32
            for _ in range(12):
                if low > high:
                    break
                n = (low + high) // 2
                row = measure(n)
                count = row[3]["text_tokens"]
                if count == target:
                    return " ".join([word] * n), row
                if count < target:
                    low = n + 1
                else:
                    high = n - 1
            for n in range(max(0, high - 8), min(target + 32, low + 8) + 1):
                row = measure(n)
                if row[3]["text_tokens"] == target:
                    return " ".join([word] * n), row
            raise ValueError(f"real processor cannot produce text_tokens={target} with {word!r}")
        instruction = inputs.get("instruction")
        if instruction is not None:
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("instruction must be nonempty text")
            row = prepare(instruction, True)
            kind = "synthetic_instruction_real_images_state"
        else:
            row = prepare(None, False)
            if row[3]["text_tokens"] == target:
                kind = ("original_release_observation" if int(getattr(policy, "num_input_images", 3)) == 3
                        else "original_instruction_two_camera_observation")
            else:
                instruction, row = find("move")
                kind = "synthetic_instruction_real_images_state"
        if row[3]["text_tokens"] != target:
            raise ValueError("instruction's actual text_tokens does not match requested length")
        if instruction is None and int(getattr(policy, "num_input_images", 3)) == 3:
            import torch
            recorded = source["steps"][step]
            if (not torch.is_tensor(recorded.get("input_ids"))
                    or not torch.is_tensor(recorded.get("attention_mask"))
                    or not torch.equal(row[1], recorded["input_ids"])
                    or not torch.equal(row[2], recorded["attention_mask"])):
                raise ValueError("original release observation token IDs/types changed")
        meta = dict(row[3], source_kind=kind, source_golden=str(path.resolve()),
                    source_golden_sha256=_digest(path), source_step=step,
                    instruction=instruction, original_instruction_preserved=instruction is None,
                    command_max_tokens=200 if instruction is None else 256,
                    cpu_reference="not_run", task_quality="not_evaluated")
        return row[0], row[1], row[2], meta
    finally:
        processor.tokenizer = original_tokenizer
        processor.instruction_override = original_override
        original_tokenizer.hardcode_instruction = original_hardcode


def prepare_inputs(policy, processor, inputs, run, checkpoint, checkpoint_manifest):
    """Return main batch, JSON metadata and a reserved alternate-batch slot."""
    import torch

    source = inputs.get("source", "batch")
    seed = run.get("seed", 7)
    if type(seed) is not int or seed < 0:
        raise ValueError("run.seed must be a nonnegative integer")
    file_sha = None
    checkpoint_identity = {name: item["sha256"] for name, item in checkpoint_manifest.items()}
    if source == "synthetic":
        batch, ids, mask, meta = _synthetic(
            policy, processor, inputs, Path(checkpoint), checkpoint_manifest, seed)
    elif source == "batch":
        path = Path(inputs["batch"]).expanduser()
        saved = torch.load(path, map_location="cpu", weights_only=True)
        wrapper = isinstance(saved, dict) and saved.get("schema") == SCHEMA
        batch = saved["batch"] if wrapper else saved
        ids, mask, meta = encode_batch(policy, batch, inputs.get("text_tokens"))
        file_sha = _digest(path)
        if wrapper:
            if (saved.get("seed") != seed or not torch.is_tensor(saved.get("input_ids"))
                    or not torch.is_tensor(saved.get("token_types"))
                    or not torch.equal(saved["input_ids"], ids) or not torch.equal(saved["token_types"], mask)
                    or saved.get("batch_sha256") != content_digest(batch, seed)
                    or saved.get("checkpoint_sha256") != checkpoint_identity):
                raise ValueError("saved G05 batch content, seed or actual token IDs changed")
            original = saved.get("source", {})
            if not isinstance(original, dict):
                raise ValueError("saved G05 source metadata must be a mapping")
            meta = dict(original, **meta)
        else:
            meta.update(source_kind="caller_preprocessed_batch", cpu_reference="not_run", task_quality="not_evaluated")
    else:
        raise ValueError("G05 input.source must be batch or synthetic")
    digest = content_digest(batch, seed)
    meta.update(input_source=source, content_sha256=digest, batch_sha256=digest,
                batch_identity="preprocessed_batch_and_noise_seed_content_sha256",
                seed=seed, actual_token_ids=ids.tolist(), alternate_available=False)
    if file_sha is not None:
        meta["batch_file_sha256"] = file_sha
    directory = Path(run["output_dir"]).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, "batch": batch, "seed": seed, "input_ids": ids,
               "token_types": mask, "batch_sha256": digest, "checkpoint_sha256": checkpoint_identity, "source": meta}
    torch.save(payload, directory / "input_batch.pt")
    return batch, meta, None


def configure_text_envelope(policy, metadata):
    """Record the installed capacity without narrowing it to an example length."""
    prefix = metadata["prefix_rows"]
    cameras = metadata.get("cameras", int(getattr(policy, "num_input_images", 3)))
    if (type(cameras) is not int or cameras not in (2, 3)
            or type(prefix) is not int or not 32 + 64 * cameras + 1 <= prefix <= 256 + 64 * cameras + 1
            or ("text_tokens" in metadata and prefix != metadata["text_tokens"] + 64 * cameras + 1)):
        raise ValueError("G05 example prefix must equal actual T32..256 plus64*cameras+1")
    state = policy.model.vlm._rpu_qwen3_5
    if prefix > state.max_seq_len:
        raise ValueError("G05 example prefix exceeds the installed KV capacity")
    metadata["prefill_capability_envelope"] = list(state.prefill_capability_envelope)
