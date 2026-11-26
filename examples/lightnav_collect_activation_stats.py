#!/usr/bin/env python3
"""Collect LightNav Text input second moments for activation-aware W8 conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECTIONS = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
DEFAULT_FRAME_INDICES = (9, 14, 19, 24, 29, 34)
OUTPUT_TOKENS = 5


def parse_frame_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("frame indices must be comma-separated integers") from error
    if not indices or any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("frame indices must be unique nonnegative integers")
    return indices


def load_teacher_tokens(
    path: Path,
    frame_indices: tuple[int, ...],
    image_names: list[str],
    instruction: str,
    history_frames: int,
):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("inferences"), list):
        raise ValueError("teacher output must contain an 'inferences' list")
    if data.get("instruction") != instruction or data.get("history_frames") != history_frames:
        raise ValueError("teacher output instruction/history_frames do not match the episode")
    if data.get("output_tokens") != OUTPUT_TOKENS:
        raise ValueError("teacher output must declare output_tokens=5")
    rows = {}
    for row in data["inferences"]:
        if not isinstance(row, dict) or type(row.get("frame_index")) is not int:
            raise ValueError("each teacher inference must contain an integer frame_index")
        index = row["frame_index"]
        if index in rows:
            raise ValueError(f"teacher output contains duplicate frame_index={index}")
        rows[index] = row
    result = {}
    for index in frame_indices:
        if index >= len(image_names):
            raise ValueError(f"frame_index={index} exceeds the episode's {len(image_names)} images")
        row = rows.get(index)
        if row is None:
            raise ValueError(f"teacher output is missing frame_index={index}")
        if row.get("image") != image_names[index]:
            raise ValueError(f"teacher frame_index={index} does not match the episode image")
        tokens = row.get("token_ids")
        if (not isinstance(tokens, list) or len(tokens) != OUTPUT_TOKENS
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError(f"teacher frame_index={index} must contain five nonnegative token_ids")
        result[index] = tokens
    return result


class ActivationMoments:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def register(self, model):
        layers = model.model.language_model.layers
        if len(layers) != 36:
            raise ValueError(f"LightNav calibration requires 36 Text layers, got {len(layers)}")
        handles = []
        for layer_index, layer in enumerate(layers):
            for role, path in PROJECTIONS.items():
                module = layer
                for name in path.split("."):
                    module = getattr(module, name)
                key = f"layers.{layer_index}.{role}"
                handles.append(module.register_forward_pre_hook(self._hook(key, module.in_features)))
        return handles

    def _hook(self, key, width):
        def collect(_module, inputs):
            import torch

            value = inputs[0].detach().reshape(-1, width).to(device="cpu", dtype=torch.float32)
            current = value.square().sum(dim=0)
            self.sums[key] = self.sums.get(key, torch.zeros_like(current)) + current
            self.counts[key] = self.counts.get(key, 0) + value.shape[0]

        return collect

    def finish(self):
        import torch

        expected = {f"layers.{layer}.{role}" for layer in range(36) for role in PROJECTIONS}
        if set(self.sums) != expected or any(self.counts.get(key, 0) == 0 for key in expected):
            raise RuntimeError("not every LightNav Text projection observed an activation")
        result = {
            key: (self.sums[key] / self.counts[key]).contiguous()
            for key in sorted(expected)
        }
        if any(
            tensor.dtype != torch.float32 or tensor.device.type != "cpu"
            or not bool(torch.isfinite(tensor).all()) or not bool((tensor > 0).all())
            for tensor in result.values()
        ):
            raise RuntimeError("activation moments must be finite positive CPU float32 vectors")
        return result


def prepare_teacher_inputs(inputs, tokens, model):
    import torch

    labels = inputs.get("labels")
    answer = (labels != -100).nonzero(as_tuple=False) if isinstance(labels, torch.Tensor) else None
    if answer is None or answer.numel() == 0:
        raise RuntimeError("LightNav processor did not mark an answer boundary")
    answer_start = int(answer[0].item())
    prepared = {}
    for key, value in inputs.items():
        if key == "labels" or not isinstance(value, torch.Tensor):
            continue
        if key in ("input_ids", "attention_mask", "image_mask", "video_mask"):
            value = value[:answer_start].unsqueeze(0)
        elif key == "position_ids":
            continue
        elif key == "pixel_values_videos" and value.ndim == 2:
            value = value.unsqueeze(0)
        prepared[key] = value.to(dtype=model.dtype) if value.dtype.is_floating_point else value

    teacher = torch.tensor(tokens, dtype=prepared["input_ids"].dtype).unsqueeze(0)
    prepared["input_ids"] = torch.cat((prepared["input_ids"], teacher), dim=1)
    for key in ("attention_mask", "image_mask", "video_mask"):
        if key in prepared:
            fill = 1 if key == "attention_mask" else 0
            suffix = prepared[key].new_full((prepared[key].shape[0], len(tokens)), fill)
            prepared[key] = torch.cat((prepared[key], suffix), dim=1)

    original_grid = prepared.pop("_original_video_grid_thw", None)
    pool_factors = prepared.pop("_post_vit_pool_factors", None)
    prepared.pop("_post_vit_pool_spatial", None)
    target = getattr(model, "_post_vit_target", None)
    if target is not None and original_grid is not None:
        target._post_vit_original_grid = original_grid
        target._post_vit_pool_factors = pool_factors
    return prepared


def load_model_and_processing(checkpoint: Path):
    import torch
    from transformers import AutoModelForImageTextToText
    from lightnav.inference.model import (
        _build_data_processor, _make_bundle, _resolve_processing_params,
    )
    from lightnav.processing import VLNQwen3VLProcessor

    params = _resolve_processing_params(str(checkpoint), task_type="trackvla")
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=False,
    ).eval()
    processor = VLNQwen3VLProcessor.from_pretrained(
        checkpoint, padding_side="left", local_files_only=True,
    )
    model._post_vit_target = None
    if bool(params.get("pool_enable")) and params.get("pool_stage") == "post_vit":
        from lightnav.processing import enable_post_vit_pool, get_post_vit_target

        merge_size = int(processor.video_processor.merge_size)
        enable_post_vit_pool(
            model, spatial_factor=int(params["pool_spatial"]), merge_size=merge_size,
        )
        model._post_vit_target = get_post_vit_target(model)

    def placeholder_positions(**kwargs):
        ids = kwargs["input_ids"]
        return {
            "position_ids": torch.zeros((3, *ids.shape), dtype=torch.long),
            "rope_deltas": None,
        }

    data_processor = _build_data_processor(processor, params, placeholder_positions)
    bundle = _make_bundle(
        model, processor, data_processor, torch.device("cpu"), placeholder_positions, params,
    )
    bundle.slowfast_tiers = None
    return model, processor, data_processor, bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="episode JSON")
    parser.add_argument("--images-dir", type=Path, help="image root; default: JSON directory")
    parser.add_argument("--teacher-output", type=Path, required=True, help="FP16 inference JSON")
    parser.add_argument("--checkpoint", type=Path, required=True, help="original unquantized checkpoint")
    parser.add_argument("--lightnav-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-frames", type=int, default=9)
    parser.add_argument(
        "--frame-indices", type=parse_frame_indices,
        default=DEFAULT_FRAME_INDICES, help="zero-based comma-separated indices",
    )
    args = parser.parse_args(argv)
    if args.history_frames < 0:
        parser.error("--history-frames must be nonnegative")

    input_path = args.input.expanduser().resolve()
    teacher_path = args.teacher_output.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    source_root = args.lightnav_source.expanduser().resolve()
    source_dir = source_root / "src"
    output_path = args.output.expanduser().resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lightnav_inference import load_episode, load_frames

    try:
        data, instruction, image_paths = load_episode(
            input_path, args.images_dir.expanduser().resolve() if args.images_dir else None,
        )
        if not (checkpoint / "eval_config.json").is_file():
            raise ValueError("--checkpoint must contain eval_config.json")
        if not (source_dir / "lightnav/inference/model.py").is_file():
            raise ValueError("--lightnav-source is missing src/lightnav/inference/model.py")
        if any(index < args.history_frames for index in args.frame_indices):
            raise ValueError("every frame index must have a complete history-plus-current window")
        teacher_tokens = load_teacher_tokens(
            teacher_path, args.frame_indices, data["images"], instruction, args.history_frames,
        )
        if (output_path.suffix != ".safetensors" or output_path.exists()
                or output_path in (input_path, teacher_path, *image_paths)
                or output_path.is_relative_to(checkpoint)
                or output_path.is_relative_to(source_root)):
            raise ValueError("--output must be a new .safetensors file outside inputs and model sources")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    sys.path.insert(0, str(source_dir))
    import torch
    from lightnav.inference.samples import build_tracking_sample
    from safetensors.torch import save_file

    model, _, data_processor, bundle = load_model_and_processing(checkpoint)
    frames = load_frames(image_paths, tuple(bundle.video_size))
    moments = ActivationMoments()
    handles = moments.register(model)
    try:
        with torch.inference_mode():
            for frame_index in args.frame_indices:
                start = frame_index - args.history_frames
                sample = build_tracking_sample(
                    frames[start:frame_index + 1], instruction, None, bundle,
                )
                for segment in sample["video_segments"]:
                    segment["frame_indices"] = [
                        int(value) + start for value in segment["frame_indices"]
                    ]
                inputs = data_processor.process_sample(
                    sample, add_generation_prompt=True, validate_video_shapes=False,
                )
                prepared = prepare_teacher_inputs(inputs, teacher_tokens[frame_index], model)
                model(
                    **prepared, use_cache=False, return_dict=True, logits_to_keep=1,
                )
                print(f"collected frame_index={frame_index}", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    tensors = moments.finish()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path), metadata={
        "source": input_path.name,
        "teacher_output": teacher_path.name,
        "history_frames": str(args.history_frames),
        "frame_indices_zero_based": ",".join(map(str, args.frame_indices)),
        "model_dtype": "bfloat16",
        "teacher_forced_tokens": str(OUTPUT_TOKENS),
    })
    print(f"wrote {len(tensors)} activation moments to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
