"""Public text/vision inference, shared by exact-profile example configurations."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
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
            image = image.convert("RGB")
            # Same square resize as the Qwen3.5 image path (examples.common.inputs).
            if "image_size" in inputs:
                image = image.resize((inputs["image_size"], inputs["image_size"]))
            images.append(image)
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


def _realtext_input_ids(tokenizer, inputs):
    """Validate the frozen MoE prose before any model weights are loaded."""
    import torch

    path = Path(inputs["prompt_file"])
    content = path.read_bytes()
    prompt_hash = hashlib.sha256(content).hexdigest()
    if prompt_hash != inputs["prompt_sha256"]:
        raise ValueError("MoE prompt_file SHA256 does not match the frozen prose")
    encoded = tokenizer(content.decode("utf-8"), add_special_tokens=False, return_tensors="pt")
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    length = inputs["prefill_tokens"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.long:
        raise ValueError("MoE prose tokenizer must return B1 INT64 token IDs")
    source_tokens = int(ids.shape[1])
    if source_tokens < length:
        raise ValueError(f"MoE prose has {source_tokens} tokens, fewer than requested {length}; padding/repetition is forbidden")
    ids = ids[:, :length].contiguous()
    for name in ("eos_token_id", "pad_token_id"):
        special = getattr(tokenizer, name, None)
        if type(special) is int and bool((ids == special).any()):
            raise ValueError(f"MoE performance prose contains {name}")
    ids_hash = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    if ids_hash != inputs["input_ids_sha256"]:
        raise ValueError("MoE tokenizer token IDs SHA256 does not match the frozen input")
    return ids, {"prompt_file": str(path), "prompt_sha256": prompt_hash,
                 "input_ids_sha256": ids_hash, "source_tokens": source_tokens,
                 "add_special_tokens": False, "padding_tokens": 0,
                 "truncation": f"first{length}"}


def _moe_execution_plan(model, stage):
    fusion = getattr(model, "model", model)
    inner = getattr(fusion, "language_model", fusion)
    plan = getattr(inner, "_rpu_last_execution_plan", None)
    if plan is None or plan.get("stage") != stage or not plan.get("physical_descriptor"):
        raise RuntimeError(f"MoE {stage} did not publish an actual execution plan")
    return json.loads(json.dumps(dict(plan), default=dict))


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


def qwen3_vl_messages(images, inputs):
    return [{"role": "user", "content": [
        *({"type": "image", "image": image} for image in images),
        {"type": "text", "text": inputs["prompt"]},
    ]}]


def qwen3_vl_encode(processor, images, inputs):
    """Qwen3-VL chat-template processor input on CPU."""
    encoded = processor.apply_chat_template(
        qwen3_vl_messages(images, inputs), tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt")
    return _extend_multimodal_input(encoded, inputs, processor.tokenizer)


def qwen3_5_vision_messages(images, inputs):
    return [{"role": "user", "content": [
        *({"type": "image"} for _ in images),
        {"type": "text", "text": inputs.get("prompt", "Describe the image.")}
    ]}]


def qwen3_5_vision_encode(processor, images, inputs, *, require_mm_token_type_ids=True):
    """Qwen3.5 image-text processor input on CPU.

    Device inference requires mm_token_type_ids when require_mm_token_type_ids=True;
    when False, callers on older processor versions may derive modality from input_ids.
    """
    text = processor.apply_chat_template(qwen3_5_vision_messages(images, inputs), tokenize=False,
                                         add_generation_prompt=True)
    encoded = processor(text=[text], images=images, return_tensors="pt")
    if require_mm_token_type_ids and "mm_token_type_ids" not in encoded:
        raise ValueError("Qwen3.5 processor must return mm_token_type_ids for image inference")
    return _extend_multimodal_input(encoded, inputs, processor.tokenizer)


def _cpu_fp32_argmax_token(logits):
    """Select from the public Dense model's unchanged CPU FP32 output."""
    import torch

    if logits.device.type != "cpu" or logits.dtype != torch.float32:
        raise ValueError("Qwen3.5 Dense host argmax requires CPU FP32 logits")
    return torch.ops.rpu.argmax_lastdim_host(logits.contiguous()).unsqueeze(-1)


def _reset_generation_cache(model, cache, input_ids, decode_steps):
    architecture = getattr(getattr(model, "config", None), "architectures", None)
    if architecture not in (["Gemma4ForConditionalGeneration"],
                            ["Qwen3_5ForConditionalGeneration"],
                            ["Qwen3_5MoeForConditionalGeneration"]):
        cache.reset()
        return

    from rpu_backend.api.cache import RPUCache
    from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache
    from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache

    if architecture == ["Gemma4ForConditionalGeneration"]:
        import torch
        from rpu_backend.adapters.gemma4.cache import _reset_generation_prefix

        if (isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2
                and input_ids.shape[0] == 1 and input_ids.shape[1] > 0
                and input_ids.dtype in {torch.uint8, torch.int8, torch.int16,
                                        torch.int32, torch.int64}
                and type(decode_steps) is int and decode_steps >= 0
                and _reset_generation_prefix(cache, model, input_ids.shape[1] + decode_steps)):
            return
    hybrid = ((type(cache) is Qwen3_5Cache
               and architecture == ["Qwen3_5ForConditionalGeneration"])
              or (type(cache) is Qwen3_5MoeCache
                  and architecture == ["Qwen3_5MoeForConditionalGeneration"]))
    if (hybrid and input_ids.ndim == 2 and input_ids.shape[0] == 1
            and input_ids.shape[1] > 1
            and not getattr(model, "_rpu_g05_vlm_ready", False)):
        # Prefill writes at least ceil64(P) KV rows and initializes GDN in SPM.
        # Skip KV clearing only when every requested decode's 16-row read block
        # stays within that newly written prefix, independently of EOS/padding.
        prefix = input_ids.shape[1]
        if (type(decode_steps) is int and decode_steps >= 0
                and prefix + decode_steps <= ((prefix + 63) // 64) * 64):
            RPUCache.reset_to_position(cache, 0)
        else:
            RPUCache.reset(cache)
    else:
        cache.reset()


def _generate(model, cache, input_ids, decode_steps, prefill=None, tokenizer=None,
              *, stop_on_eos=True, input_device=None, token_selector=None,
              timing_mode="per_call", input_metadata=None, reuse_greedy_logits=False):
    """One prompt result plus at most decode_steps decode calls, stopping at EOS."""
    import torch

    if timing_mode == "legacy_dense_vl":
        if stop_on_eos or token_selector is not None or decode_steps < 1:
            raise ValueError("legacy_dense_vl requires fixed decode and ordinary Torch argmax")
        return _generate_legacy_dense_vl(
            model, cache, input_ids, decode_steps, prefill, tokenizer, input_device)
    if timing_mode not in ("per_call", "forward_argmax"):
        raise ValueError(f"unknown generation timing_mode: {timing_mode}")
    forward_argmax = timing_mode == "forward_argmax"
    if forward_argmax and (stop_on_eos or token_selector is not None or input_metadata is None):
        raise ValueError("forward_argmax requires frozen prose, fixed decode and the public greedy selector")
    if token_selector is None:
        from rpu_backend.api import greedy_token_ids
        select_token = greedy_token_ids
    else:
        select_token = token_selector
    reuse_logits = reuse_greedy_logits and token_selector is None and not forward_argmax

    def select_output(logits):
        # Reuse the FP16 CPU read already needed for greedy selection.
        # Other selectors, timings and dtype admissions stay intact.
        cpu_logits = (logits.detach().cpu() if reuse_logits
                      and logits.dtype == torch.float16
                      and logits.device.type in {"cpu", "rpu"} else None)
        if cpu_logits is not None and logits.device.type == "rpu":
            # Reuse the original native host reducer on the already-read CPU
            # rows; its CPU entry does not acquire DDR a second time.
            token = torch.ops.rpu.argmax_lastdim_host(cpu_logits[:, -1].contiguous())
            return token.reshape(-1, 1), cpu_logits
        value = logits if cpu_logits is None else cpu_logits
        return select_token(value[:, -1]), cpu_logits

    _reset_generation_cache(model, cache, input_ids, decode_steps)
    device = input_ids.device if input_device is None else input_device
    start = time.perf_counter()
    output = model(input_ids=input_ids.to(device), past_key_values=cache,
                   logits_to_keep=1, use_cache=True, return_dict=True,
                   **(prefill or {}))
    token, cpu_logits = select_output(output.logits)
    if forward_argmax:
        prefill_ms = (time.perf_counter() - start) * 1000
        first_logits = output.logits[:, -1].detach().to(device="cpu", dtype=torch.float32)
        prefill_plan = _moe_execution_plan(model, "prefill")
    else:
        first_logits = (output.logits if cpu_logits is None else cpu_logits)[:, -1].detach().to(
            device="cpu", dtype=torch.float32)
        prefill_ms = (time.perf_counter() - start) * 1000
    # Returned CPU tensors can alias backend staging buffers. Retain owned
    # storage rather than pinning one device mapping for every generated token.
    generated = [token.cpu().clone()]
    first_logits = first_logits.clone()
    eos_ids = _eos_token_ids(model, tokenizer) if stop_on_eos else frozenset()
    decode_step_ms = []
    for _ in range(decode_steps):
        if eos_ids and all(value in eos_ids for value in generated[-1].reshape(-1).tolist()):
            break
        start = time.perf_counter()
        # Decode has exactly one input row. Keep the public forward's default
        # logits selection, matching the language report without an extra view.
        output = model(input_ids=token.to(device), past_key_values=cache,
                       use_cache=True, return_dict=True,
                       **({"logits_to_keep": 1} if forward_argmax else {}))
        token, cpu_logits = select_output(output.logits)
        decode_step_ms.append((time.perf_counter() - start) * 1000)
        generated.append(token.cpu().clone())
    tokens = torch.cat(generated, dim=1)
    calls = len(generated) - 1
    decode_ms = sum(decode_step_ms)
    prefill_tokens = int(input_ids.shape[1])
    result = {"token_ids": tokens,
              "logits": (output.logits.cpu() if cpu_logits is None else cpu_logits).clone(),
              "first_logits": first_logits,
              "generation": {
                  "prefill_tokens": prefill_tokens, "decode_calls": calls,
                  "output_tokens": calls + 1, "prefill_ms": prefill_ms,
                  "decode_ms": decode_ms, "decode_step_ms": decode_step_ms,
                  "prefill_tokens_per_second": prefill_tokens * 1000 / prefill_ms,
                  "decode_tokens_per_second": calls * 1000 / decode_ms if calls else None,
                  "timing_scope": "input transfer + forward + argmax; prefill includes FP32 CPU last logits; decode sums calls",
                  "input_ids_sha256": hashlib.sha256(
                      input_ids.cpu().contiguous().numpy().tobytes()).hexdigest(),
              }}
    if forward_argmax:
        result["generation"].update(
            timing_mode=timing_mode,
            timing_scope="input transfer + forward(logits_to_keep=1) + greedy argmax; decode sums calls; reset/snapshots/checks excluded",
            input=dict(input_metadata),
            token_ids_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest(),
            execution_plans={"prefill": prefill_plan,
                             "decode": _moe_execution_plan(model, "decode") if calls else None},
            cache={"logical_max_seq_len": int(cache.max_seq_len),
                   "allocated_max_seq_len": int(cache.allocated_max_seq_len),
                   "used_positions": int(cache.position)})
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
    _reset_generation_cache(model, cache, input_ids, decode_steps)
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
