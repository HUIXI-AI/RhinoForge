"""Single-image Qwen3.5 inference through public Dense/MoE loaders."""
from pathlib import Path
from types import SimpleNamespace

def _required_file(inputs, key):
    value = inputs.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"[input].{key} must name an existing caller-owned file")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"[input].{key}: {path}")
    return path

def _images(inputs):
    from PIL import Image

    paths = inputs.get("images")
    if not isinstance(paths, list) or not paths:
        raise ValueError("[input].images must contain at least one image path")
    result = []
    for value in paths:
        path = _required_file({"images": value}, "images")
        with Image.open(path) as image:
            image = image.convert("RGB")
            if "image_size" in inputs:
                image = image.resize((inputs["image_size"], inputs["image_size"]))
            result.append(image)
    return result

def _qwen35_vision(runner):
    from _common import validate_generation_options, validate_qwen35_checkpoint

    validate_generation_options(runner.source_config)
    import torch
    from _text_vision_runners import _extend_multimodal_input, _generate
    from transformers import AutoProcessor
    from rpu_backend.api import RPUModelForConditionalGeneration

    config, inputs = runner.source_config, runner.source_config["input"]
    timing_mode = config.get("run", {}).get("timing_mode", "per_call")
    metadata = validate_qwen35_checkpoint(config, str(_checkpoint(config)))
    images = _images(inputs)
    prompt = inputs.get("prompt", "Describe the image.")
    processor = AutoProcessor.from_pretrained(str(_checkpoint(config)), local_files_only=True)
    messages = [{"role": "user", "content": [
        *({"type": "image"} for _ in images), {"type": "text", "text": prompt}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[text], images=images, return_tensors="pt")
    if "mm_token_type_ids" not in encoded:
        raise ValueError("Qwen3.5 processor must return mm_token_type_ids for image inference")
    encoded = _extend_multimodal_input(encoded, inputs, processor.tokenizer)
    decode_steps = inputs.get("decode_steps", 4)
    required_len = int(encoded["input_ids"].shape[1]) + decode_steps
    max_seq_len = (required_len if timing_mode == "legacy_dense_vl"
                   else max(64, ((required_len + 63) // 64) * 64))
    is_moe = metadata.architectures in (["Qwen3_5MoeForConditionalGeneration"],
                                      ("Qwen3_5MoeForConditionalGeneration",))
    if is_moe:
        from rpu_backend.adapters.qwen3_5_moe import Qwen3_5MoeAdapter
        from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache

        raw = RPUModelForConditionalGeneration.from_pretrained(
            str(_checkpoint(config)), dtype=torch.float16,
            local_files_only=True, rpu_execution=config.get("rpu_execution"),
        )
        runner.owner = raw
        model = Qwen3_5MoeAdapter(
            raw, rpu_execution=config.get("rpu_execution")
        ).to_rpu(max_seq_len=max_seq_len)
        cache = Qwen3_5MoeCache.from_model(model, max_seq_len=max_seq_len)
        request = {
            key: encoded[key]
            for key in (
                "input_ids", "pixel_values", "image_grid_thw",
                "attention_mask", "mm_token_type_ids",
            )
            if key in encoded
        }
    else:
        from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache

        if timing_mode == "legacy_dense_vl":
            from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter

            raw = RPUModelForConditionalGeneration.from_pretrained(
                str(_checkpoint(config)), dtype=torch.float16,
                local_files_only=True, rpu_execution=config.get("rpu_execution"),
            )
            runner.owner = raw
            model = Qwen3_5Adapter(
                raw.half(), rpu_execution=config.get("rpu_execution")
            ).to_rpu(max_seq_len=max_seq_len)
        else:
            model = RPUModelForConditionalGeneration.from_pretrained(
                str(_checkpoint(config)), dtype=torch.float16, device="rpu",
                local_files_only=True, rpu_execution=config.get("rpu_execution"),
            )
        runner.owner = model
        cache = Qwen3_5Cache.from_model(model, max_seq_len=max_seq_len)
        request = {
            key: (
                encoded[key].to("rpu")
                if key == "pixel_values" else encoded[key]
            )
            for key in (
                "input_ids", "pixel_values", "image_grid_thw",
                "mm_token_type_ids",
            )
        }
    runner.owner = model

    input_ids = request.pop("input_ids")
    if timing_mode == "legacy_dense_vl":
        # Historical Dense VL excludes preprocessing and prompt upload.
        input_ids = input_ids.to("rpu")

    def infer(decode_steps=decode_steps):
        return _generate(model, cache, input_ids, decode_steps, request,
                         processor.tokenizer, stop_on_eos=inputs.get("stop_on_eos", False),
                         input_device="rpu", timing_mode=timing_mode)

    return infer

def _checkpoint(config):
    from rpu_backend.model_registry import model_path
    override = config["input"].get("checkpoint")
    return Path(override).expanduser() if override else model_path(config["example"]["registry_alias"])


def prepare_vision(config):
    from _runner import _close
    owner = SimpleNamespace(source_config=config, owner=None)
    try:
        return _qwen35_vision(owner), owner.owner
    except BaseException as error:
        try:
            _close(owner.owner, "qwen3_5_vision")
        except BaseException as cleanup_error:
            error.add_note(f"model setup cleanup failed: {cleanup_error!r}")
        raise
