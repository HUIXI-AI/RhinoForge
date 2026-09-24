"""First-version PaliGemma2 3B 224 profile contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError

IMAGE_TOKEN_INDEX = 257152
NUM_IMAGE_TOKENS = 256
TEXT_HIDDEN_SIZE = 2304
TEXT_VOCAB_SIZE = 257216
OUTER_VOCAB_SIZE = 257152
IMAGE_SIZE = 224
PATCH_SIZE = 14
MAX_SEQ_LEN = 4096
MAX_POSITION_EMBEDDINGS = 8192
SUPPORTED_LOGICAL_NAMES = frozenset({
    "paligemma2-3b-pt-224",
    "paligemma2-3b-mix-224",
})


@dataclass(frozen=True)
class PaliGemma2Profile:
    name: str = "paligemma2-3b-224"
    image_token_index: int = IMAGE_TOKEN_INDEX
    num_image_tokens: int = NUM_IMAGE_TOKENS
    text_hidden_size: int = TEXT_HIDDEN_SIZE
    text_vocab_size: int = TEXT_VOCAB_SIZE
    max_seq_len: int = MAX_SEQ_LEN


PROFILE = PaliGemma2Profile()

_MISSING = object()


def _get(obj: Any, field: str) -> Any:
    value = getattr(obj, field, _MISSING)
    if value is _MISSING:
        _fail(field, "present", "<missing>")
    return value


def _optional_get(obj: Any, field: str) -> Any:
    return getattr(obj, field, _MISSING)


def _architectures(obj: Any, field: str) -> list[str]:
    value = _get(obj, field)
    return list(value) if value is not None else []


def _fail(field: str, expected: Any, actual: Any) -> None:
    raise UnsupportedModelError(
        f"Unsupported PaliGemma2 profile: {field} expected "
        f"{expected!r}, got {actual!r}"
    )


def _expect(obj: Any, field: str, expected: Any) -> None:
    actual = _get(obj, field)
    if actual != expected:
        _fail(field, expected, actual)


def _expect_optional(obj: Any, field: str, expected: Any) -> None:
    actual = _optional_get(obj, field)
    if actual is not _MISSING and actual != expected:
        _fail(field, expected, actual)


def _expect_float(obj: Any, field: str, expected: float) -> None:
    actual = _get(obj, field)
    if float(actual) != expected:
        _fail(field, expected, actual)


def _shape(tensor: Any) -> tuple[int, ...] | None:
    shape = getattr(tensor, "shape", None)
    return tuple(shape) if shape is not None else None


def validate_config_profile(config: Any) -> None:
    """Validate the supported PaliGemma2 3B 224 HuggingFace config tuple."""
    if _architectures(config, "architectures") != [
        "PaliGemmaForConditionalGeneration"
    ]:
        _fail("architectures", ["PaliGemmaForConditionalGeneration"], None)

    _expect(config, "model_type", "paligemma")
    _expect_image_token(config)
    _expect(config, "projection_dim", TEXT_HIDDEN_SIZE)
    _expect_optional(config, "hidden_size", 2048)
    _expect_optional(config, "vocab_size", OUTER_VOCAB_SIZE)
    _expect_optional(config, "_vocab_size", OUTER_VOCAB_SIZE)
    _expect_optional_dtype(config)

    text_config = _get(config, "text_config")
    vision_config = _get(config, "vision_config")
    _validate_text_config(text_config)
    _validate_vision_config(vision_config)


def _validate_text_config(config: Any) -> None:
    if _architectures(config, "architectures") != ["Gemma2ForCausalLM"]:
        _fail("text_config.architectures", ["Gemma2ForCausalLM"], None)

    _expect(config, "model_type", "gemma2")
    _expect(config, "num_hidden_layers", 26)
    _expect(config, "hidden_size", TEXT_HIDDEN_SIZE)
    _expect(config, "intermediate_size", 9216)
    _expect(config, "num_attention_heads", 8)
    _expect(config, "num_key_value_heads", 4)
    _expect(config, "head_dim", 256)
    _expect(config, "hidden_activation", "gelu_pytorch_tanh")
    _expect(config, "attention_bias", False)
    _expect_float(config, "rms_norm_eps", 1e-6)
    _expect(config, "query_pre_attn_scalar", 256)
    _expect(config, "sliding_window", MAX_SEQ_LEN)
    _expect(config, "cache_implementation", "hybrid")
    _expect_float(config, "attn_logit_softcapping", 50.0)
    _expect_float(config, "final_logit_softcapping", 30.0)
    _expect(config, "max_position_embeddings", MAX_POSITION_EMBEDDINGS)
    _expect(
        config,
        "layer_types",
        ["sliding_attention", "full_attention"] * 13,
    )
    _expect(config, "use_bidirectional_attention", True)
    _expect(config, "vocab_size", TEXT_VOCAB_SIZE)
    _expect(config, "num_image_tokens", NUM_IMAGE_TOKENS)


def _validate_vision_config(config: Any) -> None:
    _expect(config, "model_type", "siglip_vision_model")
    _expect(config, "patch_size", PATCH_SIZE)
    _expect(config, "num_positions", NUM_IMAGE_TOKENS)
    _expect_image_size(config)
    _expect(config, "hidden_size", 1152)
    _expect(config, "intermediate_size", 4304)
    _expect(config, "num_attention_heads", 16)
    _expect(config, "num_hidden_layers", 27)
    _expect(config, "projection_dim", TEXT_HIDDEN_SIZE)
    _expect(config, "vision_use_head", False)


def _expect_image_token(config: Any) -> None:
    field = "image_token_index"
    actual = _optional_get(config, field)
    if actual is _MISSING:
        field = "image_token_id"
        actual = _optional_get(config, field)
    if actual is _MISSING:
        _fail("image_token_index/image_token_id", IMAGE_TOKEN_INDEX, "<missing>")
    if actual != IMAGE_TOKEN_INDEX:
        _fail(field, IMAGE_TOKEN_INDEX, actual)


def _expect_optional_dtype(config: Any) -> None:
    actual = _optional_get(config, "torch_dtype")
    if actual is _MISSING:
        return
    if actual not in ("bfloat16", "bf16", "torch.bfloat16", torch.bfloat16):
        _fail("torch_dtype", "bfloat16", actual)


def _expect_image_size(config: Any) -> None:
    actual = _optional_get(config, "image_size")
    if actual is _MISSING:
        num_positions = _get(config, "num_positions")
        patch_size = _get(config, "patch_size")
        actual = int(num_positions**0.5) * patch_size
    if actual != IMAGE_SIZE:
        _fail("image_size", IMAGE_SIZE, actual)


def validate_loaded_weights(model: Any) -> None:
    """Validate loaded PaliGemma2 language-model weights for this profile."""
    embed_tokens, lm_head, embed_field, lm_head_field = _weight_layout(model)

    _expect_weight_shape(
        _runtime_get(embed_tokens, "weight", f"{embed_field}.weight"),
        f"{embed_field}.weight",
        (TEXT_VOCAB_SIZE, TEXT_HIDDEN_SIZE),
    )
    _expect_weight_shape(
        _runtime_get(lm_head, "weight", f"{lm_head_field}.weight"),
        f"{lm_head_field}.weight",
        (TEXT_VOCAB_SIZE, TEXT_HIDDEN_SIZE),
    )

    bias = getattr(lm_head, "bias", None)
    if bias is not None:
        raise RPUBackendError(f"{lm_head_field}.bias must be absent or None")

    if hasattr(model, "named_parameters"):
        named_parameters = model.named_parameters()
    else:
        named_parameters = []
    for name, param in named_parameters:
        if hasattr(param, "is_floating_point") and param.is_floating_point():
            if getattr(param, "dtype", None) != torch.float16:
                raise RPUBackendError(
                    f"{name} must be torch.float16, got {getattr(param, 'dtype', None)}"
                )


def _weight_layout(model: Any) -> tuple[Any, Any, str, str]:
    outer_model = _optional_get(model, "model")
    if outer_model is not _MISSING:
        language_model = _runtime_get(
            outer_model,
            "language_model",
            "model.language_model",
        )
        embed_tokens = _runtime_get(
            language_model,
            "embed_tokens",
            "model.language_model.embed_tokens",
        )
        lm_head = _runtime_get(model, "lm_head", "lm_head")
        return embed_tokens, lm_head, "model.language_model.embed_tokens", "lm_head"

    language_model = _runtime_get(model, "language_model", "language_model")
    inner_model = _runtime_get(
        language_model,
        "model",
        "language_model.model",
    )
    embed_tokens = _runtime_get(
        inner_model,
        "embed_tokens",
        "language_model.model.embed_tokens",
    )
    lm_head = _runtime_get(
        language_model,
        "lm_head",
        "language_model.lm_head",
    )
    return (
        embed_tokens,
        lm_head,
        "language_model.model.embed_tokens",
        "language_model.lm_head",
    )


def _runtime_get(obj: Any, field: str, full_field: str | None = None) -> Any:
    value = getattr(obj, field, _MISSING)
    if value is _MISSING:
        name = full_field if full_field is not None else field
        raise RPUBackendError(f"Missing required PaliGemma2 weight field: {name}")
    return value


def _expect_weight_shape(
    tensor: Any,
    field: str,
    expected: tuple[int, int],
) -> None:
    actual = _shape(tensor)
    if actual != expected:
        raise RPUBackendError(
            f"{field} expected shape {expected!r}, got {actual!r}"
        )
