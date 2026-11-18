"""Public qwen3_5 text model construction."""
from __future__ import annotations

from examples.common.text import text_context, text_inputs, text_result

def prepare(config, runner):
    import torch
    from examples.common.generation import _cpu_fp32_argmax_token
    example, inputs, target, checkpoint, execution, decode_steps = text_context(config)
    tokenizer, input_ids, input_metadata, capacity = text_inputs(config, checkpoint, decode_steps)
    token_selector = None
    from rpu_backend.api import RPUModelForConditionalGeneration
    from examples.common.config import install_quantization

    load_kwargs = {}
    # A dense on-install recipe (NVFP4) binds to the raw model before install.
    quantization = install_quantization(target, example["profile_id"])
    if quantization is not None:
        load_kwargs["quantization"] = quantization
    raw = RPUModelForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.float16, local_files_only=True,
        rpu_execution=execution, **load_kwargs)
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
    return text_result(model, cache, input_ids, decode_steps, tokenizer, inputs, target, token_selector, input_metadata), model
