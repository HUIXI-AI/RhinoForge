"""PaliGemma2 greedy generation helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch

from rpu_backend.api.errors import RPUBackendError


@dataclass(frozen=True)
class GenerateOptions:
    max_new_tokens: int
    do_sample: bool = False
    num_beams: int = 1
    use_cache: bool = True
    return_dict_in_generate: bool = False


_ALLOWED_KEYS = frozenset({
    "max_new_tokens",
    "do_sample",
    "num_beams",
    "use_cache",
    "return_dict_in_generate",
    "output_scores",
    "output_logits",
    "output_attentions",
    "output_hidden_states",
})

_OUTPUT_FLAGS = (
    "output_scores",
    "output_logits",
    "output_attentions",
    "output_hidden_states",
)

_ALLOWED_STAGES = frozenset({"prefill", "decode"})


def _move_tensor(
    tensor: torch.Tensor,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    return tensor.to(**kwargs)


def collect_baseline_environment(model_name_or_path: str) -> dict[str, Any]:
    import transformers
    from transformers import AutoConfig

    from rpu_backend.model_registry import model_path

    logical_repos = {
        "paligemma2-3b-pt-224": "google/paligemma2-3b-pt-224",
        "paligemma2-3b-mix-224": "google/paligemma2-3b-mix-224",
    }
    if model_name_or_path.startswith("paligemma2-"):
        resolved = model_path(model_name_or_path)
        model_repo = logical_repos.get(model_name_or_path, model_name_or_path)
        resolved_path = str(resolved)
    else:
        resolved = None
        model_repo = model_name_or_path
        resolved_path = model_name_or_path

    config = AutoConfig.from_pretrained(
        resolved_path,
        local_files_only=bool(resolved is not None and resolved.exists()),
    )
    revision = getattr(config, "_name_or_path", None) or model_name_or_path
    commit = getattr(config, "_commit_hash", None)
    return {
        "transformers_version": transformers.__version__,
        "model_name_or_path": model_name_or_path,
        "model_repo": model_repo,
        "model_revision": revision,
        "model_commit": commit,
        "resolved_path": resolved_path,
        "local_snapshot_path": resolved_path if (resolved and resolved.exists()) else None,
        "processor_repo": model_repo,
        "processor_revision": revision,
        "tokenizer_repo": model_repo,
        "tokenizer_revision": revision,
        "processor_tokenizer_different_from_model": False,
        "paligemma_config": _config_summary(config),
        "vision_config": _config_summary(getattr(config, "vision_config", None)),
        "text_config": _config_summary(getattr(config, "text_config", None)),
    }


def run_baseline_semantic_probes(model_name_or_path: str) -> dict[str, str]:
    import inspect

    from transformers.models.gemma2.modeling_gemma2 import (
        Gemma2ForCausalLM,
        Gemma2Model,
    )
    from transformers.models.paligemma.modeling_paligemma import (
        PaliGemmaForConditionalGeneration,
        PaliGemmaModel,
        create_causal_mask_mapping,
    )

    collect_baseline_environment(model_name_or_path)
    image_src = inspect.getsource(PaliGemmaModel.get_image_features)
    causal_mask_mapping_src = inspect.getsource(create_causal_mask_mapping)
    paligemma_lm_forward_src = inspect.getsource(
        PaliGemmaForConditionalGeneration.forward
    )
    paligemma_forward_src = inspect.getsource(PaliGemmaModel.forward)
    paligemma_prepare_src = inspect.getsource(
        PaliGemmaForConditionalGeneration.prepare_inputs_for_generation
    )
    gemma2_src = inspect.getsource(Gemma2Model.forward)
    lm_src = inspect.getsource(Gemma2ForCausalLM.forward)

    image_norm = _compact_source(image_src)
    causal_mask_mapping_norm = _compact_source(causal_mask_mapping_src)
    paligemma_lm_forward_norm = _compact_source(paligemma_lm_forward_src)
    paligemma_forward_norm = _compact_source(paligemma_forward_src)
    paligemma_prepare_norm = _compact_source(paligemma_prepare_src)
    gemma2_norm = _compact_source(gemma2_src)
    lm_norm = _compact_source(lm_src)

    def require(label: str, condition: bool) -> str:
        if not condition:
            raise RuntimeError(
                f"PaliGemma2 HF baseline semantic probe failed: {label}"
            )
        return label

    return {
        "image_feature_scale": require(
            "sqrt(2304)",
            "hidden_size**0.5" in image_norm
            and "image_features=image_features/" in image_norm,
        ),
        "gemma2_embed_scale": require(
            "sqrt(2304)",
            "normalizer=torch.tensor(self.config.hidden_size**0.5"
            in gemma2_norm
            and "hidden_states=hidden_states*normalizer" in gemma2_norm,
        ),
        "position_ids": require(
            "cache_position+1",
            "position_ids=cache_position.unsqueeze(0)+1"
            in paligemma_forward_norm
            and 'model_inputs["position_ids"]=model_inputs["position_ids"]+1'
            in paligemma_prepare_norm,
        ),
        "token_type_reverse": require(
            "1-token_type_ids",
            "token_type_ids=1-token_type_ids" in causal_mask_mapping_norm,
        ),
        "paligemma_lm_head_softcap": require(
            "not_applied",
            "self.lm_head(hidden_states" in paligemma_lm_forward_norm
            and "final_logit_softcapping" not in paligemma_lm_forward_norm
            and "torch.tanh" not in paligemma_lm_forward_norm,
        ),
        "final_logit_softcap": require(
            "recorded",
            "final_logit_softcapping" in lm_norm and "torch.tanh" in lm_norm,
        ),
    }


def _compact_source(source: str) -> str:
    return "".join(source.split())


def _config_summary(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    fields = (
        "model_type",
        "architectures",
        "torch_dtype",
        "image_token_index",
        "image_token_id",
        "num_image_tokens",
        "projection_dim",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_activation",
        "attention_bias",
        "rms_norm_eps",
        "query_pre_attn_scalar",
        "sliding_window",
        "cache_implementation",
        "attn_logit_softcapping",
        "final_logit_softcapping",
        "max_position_embeddings",
        "layer_types",
        "use_bidirectional_attention",
        "image_size",
        "patch_size",
        "num_positions",
        "vision_use_head",
    )
    out: dict[str, Any] = {}
    for field in fields:
        if not hasattr(config, field):
            continue
        value = getattr(config, field)
        if isinstance(value, torch.dtype):
            value = str(value)
        elif isinstance(value, (list, tuple)):
            value = list(value)
        out[field] = value
    return out


def _ensure_rank2(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 2:
        raise RPUBackendError(
            f"PaliGemma2 {name} must be rank 2, got shape "
            f"{tuple(tensor.shape)}"
        )


def _negative_infinity(dtype: torch.dtype) -> float:
    if dtype.is_floating_point:
        return float("-inf")
    raise RPUBackendError("PaliGemma2 additive attention mask dtype must float")


def build_prefill_additive_attention_mask(
    *,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    _ensure_rank2("attention_mask", attention_mask)
    _ensure_rank2("token_type_ids", token_type_ids)
    if tuple(attention_mask.shape) != tuple(token_type_ids.shape):
        raise RPUBackendError(
            "PaliGemma2 token_type_ids shape must match attention_mask shape, "
            f"got {tuple(token_type_ids.shape)} and {tuple(attention_mask.shape)}"
        )

    attn = attention_mask.to("cpu")
    reversed_token_types = (1 - token_type_ids).to("cpu")
    batch_size, seq_len = int(attn.shape[0]), int(attn.shape[1])
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    allowed = causal.unsqueeze(0).expand(batch_size, seq_len, seq_len).clone()

    for batch_idx in range(batch_size):
        row = reversed_token_types[batch_idx]
        pos = 0
        while pos < seq_len:
            if int(row[pos].item()) != 1:
                pos += 1
                continue
            end = pos + 1
            while end < seq_len and int(row[end].item()) == 1:
                end += 1
            allowed[batch_idx, pos:end, pos:end] = True
            pos = end

    key_padding = attn.to(torch.bool).unsqueeze(1)
    allowed &= key_padding
    additive = torch.zeros(
        batch_size,
        seq_len,
        seq_len,
        dtype=dtype,
        device="cpu",
    )
    additive = additive.masked_fill(~allowed, _negative_infinity(dtype))
    return additive.unsqueeze(1).contiguous()


def build_decode_additive_attention_mask(
    *,
    attention_mask: torch.Tensor,
    key_value_len: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    _ensure_rank2("attention_mask", attention_mask)
    if int(attention_mask.shape[1]) != int(key_value_len):
        raise RPUBackendError(
            "PaliGemma2 decode attention_mask length must match key_value_len, "
            f"got {int(attention_mask.shape[1])} and {int(key_value_len)}"
        )
    allowed = attention_mask.to("cpu", dtype=torch.bool).unsqueeze(1)
    additive = torch.zeros(
        int(attention_mask.shape[0]),
        1,
        int(key_value_len),
        dtype=dtype,
        device="cpu",
    )
    additive = additive.masked_fill(~allowed, _negative_infinity(dtype))
    return additive.unsqueeze(1).contiguous()


def _require_exact_bool(
    kwargs: dict[str, Any],
    key: str,
    expected: bool,
) -> None:
    if key in kwargs and kwargs[key] is not expected:
        raise ValueError(
            "PaliGemma2 first-version generate requires "
            f"{key}={expected}"
        )


def _require_int(kwargs: dict[str, Any], key: str, default: int) -> int:
    value = kwargs.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"PaliGemma2 first-version generate requires {key} to be an int"
        )
    return value


def validate_generate_kwargs(kwargs: dict[str, Any]) -> GenerateOptions:
    unknown = sorted(set(kwargs) - _ALLOWED_KEYS)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"PaliGemma2 generate unsupported kwargs: {names}")

    max_new_tokens = _require_int(kwargs, "max_new_tokens", 1)
    if not 1 <= max_new_tokens <= 8:
        raise ValueError(
            "PaliGemma2 first-version generate requires "
            "1 <= max_new_tokens <= 8"
        )

    _require_exact_bool(kwargs, "do_sample", False)

    num_beams = _require_int(kwargs, "num_beams", 1)
    if num_beams != 1:
        raise ValueError(
            "PaliGemma2 first-version generate requires num_beams=1"
        )

    _require_exact_bool(kwargs, "use_cache", True)
    _require_exact_bool(kwargs, "return_dict_in_generate", False)
    for key in _OUTPUT_FLAGS:
        _require_exact_bool(kwargs, key, False)

    return GenerateOptions(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        return_dict_in_generate=False,
    )


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _validate_softcap(final_logit_softcap: Any) -> float | None:
    if final_logit_softcap is None:
        return None
    if isinstance(final_logit_softcap, bool) or not isinstance(
        final_logit_softcap,
        Real,
    ):
        raise ValueError(
            "PaliGemma2 final_logit_softcap must be a positive number"
        )
    cap = float(final_logit_softcap)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError(
            "PaliGemma2 final_logit_softcap must be a positive number"
        )
    return cap


def compute_logits_cpu_fallback(
    lm_head: Any,
    hidden_states: torch.Tensor,
    recorder: Any,
    stage: str,
    final_logit_softcap: Any,
    *,
    apply_final_logit_softcap: bool = False,
) -> torch.Tensor:
    if stage not in _ALLOWED_STAGES:
        raise ValueError(
            "PaliGemma2 lm_head fallback stage must be 'prefill' or 'decode'"
        )
    if not isinstance(apply_final_logit_softcap, bool):
        raise ValueError(
            "PaliGemma2 apply_final_logit_softcap must be a bool"
        )
    cap = _validate_softcap(final_logit_softcap)
    if apply_final_logit_softcap and cap is None:
        raise ValueError(
            "PaliGemma2 apply_final_logit_softcap requires "
            "final_logit_softcap"
        )

    cpu_hidden = hidden_states.detach().to("cpu")
    recorder.record(
        component="lm_head",
        op="cpu_lm_head",
        stage=stage,
        reason="first-version logits fallback",
        shape={
            "hidden_states": list(hidden_states.shape),
            "final_logit_softcap": cap,
            "applied_final_logit_softcap": apply_final_logit_softcap,
        },
        dtype=str(hidden_states.dtype),
        device_from=hidden_states.device.type,
        device_to="cpu",
        bytes_moved=_tensor_bytes(hidden_states),
    )

    logits = lm_head(cpu_hidden)
    if apply_final_logit_softcap:
        logits = torch.tanh(logits / cap) * cap
    return logits
