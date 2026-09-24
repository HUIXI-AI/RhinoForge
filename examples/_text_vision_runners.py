"""Public text/vision inference, shared by exact-profile example configurations."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import time


def _image_inputs(inputs):
    from PIL import Image

    paths = [Path(value).expanduser() for value in inputs["images"]]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"[input].images: {path}")
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def _eos_token_ids(model, tokenizer):
    """Use generation configuration first, then checkpoint/tokenizer defaults."""
    config = getattr(model, "config", None)
    for owner in (getattr(model, "generation_config", None),
                  config, getattr(config, "text_config", None), tokenizer):
        value = getattr(owner, "eos_token_id", None)
        values = (value,) if type(value) is int else value
        if (isinstance(values, (tuple, list)) and values
                and all(type(item) is int and item >= 0 for item in values)):
            return frozenset(values)
    return frozenset()


def _text_input_ids(tokenizer, inputs):
    """Construct the explicitly requested benchmark tokens on the CPU."""
    import torch

    if "prompt_token_ids" in inputs:
        ids = torch.tensor([inputs["prompt_token_ids"]], dtype=torch.long)
    else:
        encoded = tokenizer(inputs["prompt"], return_tensors="pt")
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if "prefill_tokens" not in inputs:
        return ids
    length = inputs["prefill_tokens"]
    if "prefill_fill_token_id" in inputs:
        if length > ids.shape[1]:
            ids = torch.nn.functional.pad(
                ids, (0, length - ids.shape[1]), value=inputs["prefill_fill_token_id"])
        return ids[:, :length].contiguous()
    return ids.repeat(1, (length + ids.shape[1] - 1) // ids.shape[1])[:, :length].contiguous()


def _extend_multimodal_input(encoded, inputs, tokenizer):
    """Extend the last user message, preserving image and assistant tokens."""
    import torch

    if "prefill_tokens" not in inputs:
        return encoded
    length, current = inputs["prefill_tokens"], int(encoded["input_ids"].shape[1])
    if length < current:
        raise ValueError("prefill_tokens cannot truncate a processor's multimodal input")
    if length == current:
        return encoded
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ends = (encoded["input_ids"][0] == end_id).nonzero().reshape(-1)
    if ends.numel() == 0:
        raise ValueError("cannot extend multimodal input without a chat message boundary")
    insertion = int(ends[-1])
    tail_inputs = dict(inputs, prefill_tokens=length - current)
    if "prefill_fill_token_id" in inputs:
        tail_inputs["prompt_token_ids"] = [inputs["prefill_fill_token_id"]]
    if "prefill_filler_prompt" in inputs:
        filler = tokenizer(inputs["prefill_filler_prompt"], add_special_tokens=False,
                           return_tensors="pt")
        filler = filler["input_ids"] if isinstance(filler, dict) else filler.input_ids
        if filler.shape[1] == 0:
            raise ValueError("prefill_filler_prompt must encode at least one token")
        extra = length - current
        tail = filler.repeat(1, (extra + filler.shape[1] - 1) // filler.shape[1])[:, :extra]
    else:
        tail = _text_input_ids(tokenizer, tail_inputs)
    ids = encoded["input_ids"]
    encoded["input_ids"] = torch.cat((ids[:, :insertion], tail, ids[:, insertion:]), dim=1)
    for key, value in (("attention_mask", 1), ("mm_token_type_ids", 0)):
        if key in encoded:
            original = encoded[key]
            extension = original.new_full((1, length - current), value)
            encoded[key] = torch.cat(
                (original[:, :insertion], extension, original[:, insertion:]), dim=1)
    return encoded


def _cpu_fp32_argmax_token(logits):
    """Select from the public Dense model's unchanged CPU FP32 output."""
    import torch

    if logits.device.type != "cpu" or logits.dtype != torch.float32:
        raise ValueError("Qwen3.5 Dense host argmax requires CPU FP32 logits")
    return torch.ops.rpu.argmax_lastdim_host(logits.contiguous()).unsqueeze(-1)


def _generate(model, cache, input_ids, decode_steps, prefill=None, tokenizer=None,
              *, stop_on_eos=True, input_device=None, token_selector=None,
              timing_mode="per_call"):
    """One prompt result plus at most decode_steps decode calls, stopping at EOS."""
    import torch

    if timing_mode == "legacy_dense_vl":
        if stop_on_eos or token_selector is not None or decode_steps < 1:
            raise ValueError("legacy_dense_vl requires fixed decode and ordinary Torch argmax")
        return _generate_legacy_dense_vl(
            model, cache, input_ids, decode_steps, prefill, tokenizer, input_device)
    if timing_mode != "per_call":
        raise ValueError(f"unknown generation timing_mode: {timing_mode}")
    if token_selector is None:
        from rpu_backend.api import greedy_token_ids
        select_token = greedy_token_ids
    else:
        select_token = token_selector
    cache.reset()
    device = input_ids.device if input_device is None else input_device
    start = time.perf_counter()
    output = model(input_ids=input_ids.to(device), past_key_values=cache,
                   logits_to_keep=1, use_cache=True, return_dict=True,
                   **(prefill or {}))
    token = select_token(output.logits[:, -1])
    first_logits = output.logits[:, -1].detach().float().cpu()
    prefill_ms = (time.perf_counter() - start) * 1000
    # Returned CPU tensors can alias backend staging buffers. Retain owned
    # storage rather than pinning one RPU mapping for every generated token.
    generated = [token.cpu().clone()]
    first_logits = first_logits.clone()
    eos_ids = _eos_token_ids(model, tokenizer) if stop_on_eos else frozenset()
    decode_ms = 0.0
    for _ in range(decode_steps):
        if eos_ids and all(value in eos_ids for value in generated[-1].reshape(-1).tolist()):
            break
        start = time.perf_counter()
        # Decode has exactly one input row. Keep the public forward's default
        # logits selection, matching the language report without an extra view.
        output = model(input_ids=token.to(device), past_key_values=cache,
                       use_cache=True, return_dict=True)
        token = select_token(output.logits[:, -1])
        decode_ms += (time.perf_counter() - start) * 1000
        generated.append(token.cpu().clone())
    tokens = torch.cat(generated, dim=1)
    calls = len(generated) - 1
    prefill_tokens = int(input_ids.shape[1])
    result = {"token_ids": tokens, "logits": output.logits.cpu().clone(),
              "first_logits": first_logits,
              "generation": {
                  "prefill_tokens": prefill_tokens, "decode_calls": calls,
                  "output_tokens": calls + 1, "prefill_ms": prefill_ms,
                  "decode_ms": decode_ms,
                  "prefill_tokens_per_second": prefill_tokens * 1000 / prefill_ms,
                  "decode_tokens_per_second": calls * 1000 / decode_ms if calls else None,
                  "timing_scope": "input transfer + forward + argmax; prefill includes FP32 CPU last logits; decode sums calls",
                  "input_ids_sha256": hashlib.sha256(
                      input_ids.cpu().contiguous().numpy().tobytes()).hexdigest(),
              }}
    if tokenizer is not None:
        result["text"] = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return result


def _legacy_checked_logits(output, vocab_size):
    import torch

    value = output.logits.cpu().clone()
    if value.shape != (1, 1, vocab_size) or not bool(torch.isfinite(value).all()):
        raise ValueError("Dense VL logits must be finite with shape (1, 1, vocab_size)")
    return value


def _generate_legacy_dense_vl(model, cache, input_ids, decode_steps, prefill,
                              tokenizer, input_device):
    """Retain the published Dense VL timer boundaries and owning snapshots.

    This explicitly selected mode uses ordinary Torch argmax. Its whole decode
    loop includes token upload/readback and the first/last decode logits checks;
    the first prefill logits check occurs after the prefill timer.
    """
    import torch

    device = input_ids.device if input_device is None else input_device
    # Setup, including the prompt transfer, precedes the historical timer.
    input_ids = input_ids.to(device)
    vocab_size = model.config.text_config.vocab_size
    cache.reset()
    start = time.perf_counter()
    output = model(input_ids=input_ids, past_key_values=cache,
                   logits_to_keep=1, use_cache=True, return_dict=True,
                   **(prefill or {}))
    token = output.logits[:, -1].argmax(dim=-1, keepdim=True).long().to(device)
    token_cpu = token.cpu().clone()
    prefill_ms = (time.perf_counter() - start) * 1000
    first_logits = _legacy_checked_logits(output, vocab_size)
    generated = [token_cpu]
    start = time.perf_counter()
    for step in range(decode_steps):
        output = model(input_ids=token, past_key_values=cache,
                       logits_to_keep=1, use_cache=True, return_dict=True)
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True).long().to(device)
        generated.append(token.cpu().clone())
        if step in (0, decode_steps - 1):
            _legacy_checked_logits(output, vocab_size)
    decode_ms = (time.perf_counter() - start) * 1000
    tokens = torch.cat(generated, dim=1)
    prefill_tokens = int(input_ids.shape[1])
    result = {"token_ids": tokens, "first_logits": first_logits,
              "logits": _legacy_checked_logits(output, vocab_size),
              "generation": {
                  "prefill_tokens": prefill_tokens, "decode_calls": decode_steps,
                  "output_tokens": decode_steps + 1, "prefill_ms": prefill_ms,
                  "decode_ms": decode_ms,
                  "prefill_tokens_per_second": prefill_tokens * 1000 / prefill_ms,
                  "decode_tokens_per_second": decode_steps * 1000 / decode_ms,
                  "timing_mode": "legacy_dense_vl",
                  "timing_scope": "forward + Torch argmax + token upload/readback; whole decode loop includes first/last logits checks; prompt upload excluded",
                  "input_ids_sha256": hashlib.sha256(
                      input_ids.cpu().contiguous().numpy().tobytes()).hexdigest(),
                  "token_ids_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
              }}
    if tokenizer is not None:
        result["text"] = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return result


def prepare_text_vision(config):
    from _runner import _close

    runner = SimpleNamespace(owner=None)
    try:
        return _prepare_text_vision(config, runner)
    except BaseException as error:
        try:
            _close(runner.owner, config["example"]["target"])
        except BaseException as cleanup_error:
            error.add_note(f"model setup cleanup failed: {cleanup_error!r}")
        raise


def _prepare_text_vision(config, runner):
    import torch
    from transformers import AutoTokenizer
    from rpu_backend.model_registry import model_path

    example, inputs = config["example"], config["input"]
    checkpoint = str(Path(inputs["checkpoint"]).expanduser()) if inputs.get("checkpoint") else str(model_path(example["registry_alias"]))
    target = example["target"]
    execution = config.get("rpu_execution")
    decode_steps = inputs.get("decode_steps", 4)
    if target == "qwen3_5":
        from _common import validate_qwen35_checkpoint
        validate_qwen35_checkpoint(config, checkpoint)
    if target == "dinov3":
        from transformers import AutoConfig, AutoImageProcessor, DINOv3ViTModel
        from rpu_backend.adapters.dinov3 import DINOv3Adapter

        images = _image_inputs(inputs)
        hf_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        DINOv3Adapter.preflight(hf_config)
        processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
        pixels = processor(images=images, return_tensors="pt").pixel_values
        model = DINOv3ViTModel.from_pretrained(
            checkpoint, config=hf_config, dtype=torch.float16,
            local_files_only=True).eval()
        model = DINOv3Adapter(model, rpu_execution=execution).to_rpu()
        runner.owner = model

        def infer_vision():
            output = model(pixels)
            return {"last_hidden_state": output.last_hidden_state.cpu(),
                    "pooler_output": output.pooler_output.cpu()}

        return infer_vision, model

    if target == "qwen3_vl":
        from _common import validate_checkpoint_profile
        from transformers import AutoProcessor
        from rpu_backend.api import RPUCache, RPUModelForConditionalGeneration

        validate_checkpoint_profile(config, checkpoint)
        component = inputs.get("component", "multimodal")
        images = _image_inputs(inputs) if component != "text" else []
        processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        if images:
            messages = [{"role": "user", "content": [
                *({"type": "image", "image": image} for image in images),
                {"type": "text", "text": inputs["prompt"]},
            ]}]
            encoded = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
            encoded = _extend_multimodal_input(encoded, inputs, processor.tokenizer)
        else:
            encoded = {"input_ids": _text_input_ids(processor.tokenizer, inputs)}
        model = RPUModelForConditionalGeneration.from_pretrained(
            checkpoint, dtype=torch.float16, device="rpu", rpu_execution=execution,
            local_files_only=True, trust_remote_code=False)
        runner.owner = model
        if component == "vision":
            return lambda: model.model.visual(
                encoded["pixel_values"].to("rpu", dtype=torch.float16),
                grid_thw=encoded["image_grid_thw"]), model
        cache = RPUCache.from_model(model.model.language_model,
                                   input_ids=encoded["input_ids"],
                                   max_new_tokens=decode_steps)
        prefill = {name: encoded[name] for name in (
            "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"
        ) if name in encoded}
        input_ids = encoded["input_ids"]
        return lambda decode_steps=decode_steps: _generate(
            model, cache, input_ids, decode_steps, prefill, processor.tokenizer,
            stop_on_eos=inputs.get("stop_on_eos", True), input_device="rpu"), model

    if target == "qwen3":
        from _common import validate_checkpoint_profile

        validate_checkpoint_profile(config, checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    input_ids = _text_input_ids(tokenizer, inputs)
    # Prefill returns the first token; D subsequent calls consume exactly D
    # further cache positions, even though D+1 output tokens are returned.
    capacity = int(input_ids.shape[1]) + decode_steps
    token_selector = None
    if target == "qwen3_5":
        from rpu_backend.api import RPUModelForConditionalGeneration

        raw = RPUModelForConditionalGeneration.from_pretrained(
            checkpoint, dtype=torch.float16, local_files_only=True,
            rpu_execution=execution)
        runner.owner = raw
        if getattr(raw.config, "architectures", None) == ["Qwen3_5MoeForConditionalGeneration"]:
            from rpu_backend.adapters.qwen3_5_moe import Qwen3_5MoeAdapter
            from rpu_backend.api import Qwen3_5MoeCache

            adapter_type, cache_type = Qwen3_5MoeAdapter, Qwen3_5MoeCache
        else:
            from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter
            from rpu_backend.api import Qwen3_5Cache
            from rpu_backend.adapters.qwen3_5.quant_scope import validate_legacy_text_profile

            if validate_legacy_text_profile(raw.config):
                # The adapter checks the state again before weight upload.
                torch.rpu.set_caching_allocator(False)
            else:
                token_selector = _cpu_fp32_argmax_token
            adapter_type, cache_type = Qwen3_5Adapter, Qwen3_5Cache
        model = adapter_type(raw, rpu_execution=execution).to_rpu(max_seq_len=capacity)
        runner.owner = model
        cache = cache_type.from_model(model, max_seq_len=capacity)
    elif target == "gemma4":
        from transformers import AutoModelForImageTextToText
        from rpu_backend.adapters.gemma4 import Gemma4Adapter

        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, dtype=torch.bfloat16, device_map="cpu",
            local_files_only=True, low_cpu_mem_usage=True).eval()
        model._rpu_max_seq = max(512, capacity)
        model = Gemma4Adapter(model, rpu_execution=execution).to_rpu()
        runner.owner = model
        cache = model.build_rpu_cache(max_seq_len=max(512, capacity))
    else:
        from rpu_backend import RPUCache, RPUModelForCausalLM

        load_kwargs = {}
        if target == "qwen3":
            from _common import example_profiles

            profile = example_profiles()[example["profile_id"]]
            quantization = profile["input_envelope"]["limits"].get("on_install_quantization")
            if quantization is not None:
                load_kwargs["quantization"] = quantization
        model = RPUModelForCausalLM.from_pretrained(
            checkpoint, dtype=torch.float16, device="rpu", rpu_execution=execution,
            local_files_only=True, trust_remote_code=False, **load_kwargs)
        runner.owner = model
        cache = RPUCache.from_model(model, input_ids=input_ids,
                                   max_new_tokens=decode_steps)
    return lambda decode_steps=decode_steps: _generate(
        model, cache, input_ids, decode_steps, tokenizer=tokenizer,
        stop_on_eos=inputs.get("stop_on_eos", True),
        input_device=None if target == "gemma4" else "rpu",
        token_selector=token_selector), model
