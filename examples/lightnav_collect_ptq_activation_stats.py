#!/usr/bin/env python3
"""Collect LightNav Text W8 activation statistics from customer PTQ episodes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


HISTORY_FRAMES = 9


def calibration_frame(image_count: int) -> int:
    if image_count < 1:
        raise ValueError("an episode must contain at least one image")
    return max(0, image_count // 2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lightnav-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    input_root = args.input_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    source_root = args.lightnav_source.expanduser().resolve()
    source_dir = source_root / "src"
    output_path = args.output.expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from lightnav_ptq_evaluate import find_episode_paths, load_ptq_episode, parse_action_tokens

    try:
        episodes = find_episode_paths(input_root)
        if not episodes:
            raise ValueError("--input-root contains no episode directories")
        episode_rows = [(path, *load_ptq_episode(path)) for path in episodes]
        if not (checkpoint / "eval_config.json").is_file():
            raise ValueError("--checkpoint must contain eval_config.json")
        if not (source_dir / "lightnav/inference/model.py").is_file():
            raise ValueError("--lightnav-source is missing src/lightnav/inference/model.py")
        if (output_path.suffix != ".safetensors" or output_path.exists()
                or output_path.is_relative_to(input_root)
                or output_path.is_relative_to(checkpoint)
                or output_path.is_relative_to(source_root)):
            raise ValueError("--output must be a new .safetensors file outside inputs and model sources")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    sys.path.insert(0, str(source_dir))
    import torch
    from lightnav.inference.samples import build_tracking_sample
    from lightnav_collect_activation_stats import (
        ActivationMoments, load_model_and_processing, prepare_teacher_inputs,
    )
    from lightnav_inference import load_frames
    from safetensors.torch import save_file

    model, processor, data_processor, bundle = load_model_and_processing(checkpoint)
    vocabulary = processor.tokenizer.get_vocab()
    moments = ActivationMoments()
    handles = moments.register(model)
    selected = []
    try:
        with torch.inference_mode():
            for number, (path, instruction, images, actions) in enumerate(episode_rows, 1):
                frame_index = calibration_frame(len(images))
                start = max(0, frame_index - HISTORY_FRAMES)
                frames = load_frames(images[start:frame_index + 1], tuple(bundle.video_size))
                sample = build_tracking_sample(frames, instruction, None, bundle)
                for segment in sample["video_segments"]:
                    segment["frame_indices"] = [
                        int(value) + start for value in segment["frame_indices"]
                    ]
                inputs = data_processor.process_sample(
                    sample, add_generation_prompt=True, validate_video_shapes=False,
                )
                tokens = parse_action_tokens(actions[frame_index].get("raw_text"), vocabulary)
                prepared = prepare_teacher_inputs(inputs, tokens, model)
                model(**prepared, use_cache=False, return_dict=True, logits_to_keep=1)
                selected.append(f"{path.name}:{frame_index}")
                print(
                    f"[{number}/{len(episode_rows)}] collected {path.name} "
                    f"frame_index={frame_index}", flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    tensors = moments.finish()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        save_file(tensors, str(temporary), metadata={
            "source": input_root.name,
            "episodes": str(len(episode_rows)),
            "history_frames": str(HISTORY_FRAMES),
            "selection": "middle_frame_per_episode",
            "selected_frames": ",".join(selected),
            "model_dtype": "bfloat16",
            "teacher_forced_tokens": "5",
        })
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {len(tensors)} activation moments to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
