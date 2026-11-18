"""Shared text input preparation and the unchanged generation closure."""
from __future__ import annotations

def text_context(config):
    from rpu_backend.model_registry import model_path

    example, inputs = config["example"], config["input"]
    target = example["target"]
    override = config.get("_model_override")
    if override is not None:
        from examples.common.config import planner_profiles

        profile = planner_profiles().get(example.get("profile_id"), {})
        if target != "qwen3_5" or profile.get("family") != "qwen3-5-moe":
            raise ValueError("model path override requires a Qwen3.5 MoE text profile")
    checkpoint = str(override if override is not None else model_path(example["registry_alias"]))
    execution = config.get("rpu_execution")
    decode_steps = inputs.get("decode_steps", 4)
    return example, inputs, target, checkpoint, execution, decode_steps

def text_token_inputs(config, checkpoint):
    """Tokenize the configured prompt on the CPU; shared with the golden producer.

    No checkpoint-profile binding or backend import: the host producer runs
    this against the float parent's tokenizer.
    """
    from transformers import AutoTokenizer
    from examples.common.generation import _text_input_ids, _realtext_input_ids
    inputs = config["input"]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    input_metadata = None
    if "prompt_file" in inputs:
        from examples.common.config import validate_generation_options

        validate_generation_options(config)
        input_ids, input_metadata = _realtext_input_ids(tokenizer, inputs)
    else:
        input_ids = _text_input_ids(tokenizer, inputs)
    return tokenizer, input_ids, input_metadata

def text_inputs(config, checkpoint, decode_steps):
    example = config["example"]
    target = example["target"]

    if target == "qwen3" or (target == "qwen3_5" and "profile_id" in example):
        from examples.common.config import validate_checkpoint_profile

        validate_checkpoint_profile(config, checkpoint)
    tokenizer, input_ids, input_metadata = text_token_inputs(config, checkpoint)
    # Prefill returns the first token; D subsequent calls consume exactly D
    # further cache positions, even though D+1 output tokens are returned.
    capacity = int(input_ids.shape[1]) + decode_steps

    return tokenizer, input_ids, input_metadata, capacity

def text_result(model, cache, input_ids, decode_steps, tokenizer, inputs, target, token_selector, input_metadata):
    from examples.common.generation import _generate
    return lambda decode_steps=decode_steps: _generate(
        model, cache, input_ids, decode_steps, tokenizer=tokenizer,
        stop_on_eos=inputs.get("stop_on_eos", True),
        input_device=("cpu" if target in {"qwen3", "llama", "qwen3_5"}
                      else None if target == "gemma4" else "rpu"),
        token_selector=token_selector,
        reuse_greedy_logits=target in {"qwen3", "llama"},
        **({"timing_mode": "forward_argmax", "input_metadata": input_metadata}
           if input_metadata is not None else {}))
