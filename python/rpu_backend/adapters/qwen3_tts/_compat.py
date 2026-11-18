"""Exact edits for the reviewed source and Transformers 5.3.0/5.19.0.

Weights are loaded with a normal CPU constructor and strict load_state_dict;
none of the HF meta-loading/reinitialization workarounds belong here.
"""


def patch_source(path: str, text: str) -> str:
    def replace(old: str, new: str, count: int) -> None:
        nonlocal text
        if text.count(old) != count:
            raise ValueError(f"Qwen3-TTS compatibility patch does not match {path}")
        text = text.replace(old, new)

    rope_old = '''        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)'''
    rope_new = '''        if self.rope_type != "default":
            raise ValueError("Reviewed TTS profile requires default RoPE")
        dim = config.head_dim
        inv_freq = 1.0 / (config.rope_theta ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / dim
        ))
        self.attention_scaling = 1.0'''
    if path == "core/models/modeling_qwen3_tts.py":
        replace(rope_old, rope_new, 2)
        replace("self.padding_idx = config.pad_token_id",
                'self.padding_idx = getattr(config, "pad_token_id", None)', 2)
        replace('_tied_weights_keys = ["lm_head.weight"]',
                '_tied_weights_keys = {}  # Frozen Base has independent codec heads.', 2)
    elif path == "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py":
        replace(rope_old, rope_new, 1)
        replace("from transformers.utils.generic import check_model_inputs",
                "from transformers.utils.generic import merge_with_config_defaults\n"
                "from transformers.utils.output_capturing import capture_outputs", 1)
        replace("    @check_model_inputs()", "    @merge_with_config_defaults\n    @capture_outputs", 1)
    if path in ("core/models/modeling_qwen3_tts.py",
                "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"):
        replace("logger = logging.get_logger(__name__)",
                "from inspect import signature as _signature\n\n"
                "_MASK_HAS_CACHE_POSITION = \"cache_position\" in _signature(create_causal_mask).parameters\n\n"
                "logger = logging.get_logger(__name__)", 1)
        replace('                "input_embeds": inputs_embeds,',
                '                "inputs_embeds": inputs_embeds,', 1)
        replace('                "cache_position": cache_position,',
                '                **({"cache_position": cache_position} if _MASK_HAS_CACHE_POSITION else {}),', 1)
        if path == "core/models/modeling_qwen3_tts.py":
            replace("            input_embeds=inputs_embeds,", "            inputs_embeds=inputs_embeds,", 1)
            replace("            cache_position=cache_position,\n"
                    "            past_key_values=past_key_values,\n"
                    "            position_ids=text_position_ids,",
                    '            **({"cache_position": cache_position} if _MASK_HAS_CACHE_POSITION else {}),\n'
                    "            past_key_values=past_key_values,\n"
                    "            position_ids=text_position_ids,", 1)
    return text
