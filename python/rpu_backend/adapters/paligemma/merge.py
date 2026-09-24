"""PaliGemma2 image/text merge helpers."""
from __future__ import annotations

import torch

from rpu_backend.api.errors import RPUBackendError
from rpu_backend.adapters.paligemma.profile import (
    IMAGE_TOKEN_INDEX,
    NUM_IMAGE_TOKENS,
    TEXT_HIDDEN_SIZE,
)


def scale_image_features(
    image_features: torch.Tensor,
    *,
    hidden_size: int = TEXT_HIDDEN_SIZE,
) -> torch.Tensor:
    """Apply the PaliGemma projector-to-text hidden-size scale."""
    scale = torch.tensor(
        float(hidden_size),
        dtype=image_features.dtype,
        device=image_features.device,
    ).sqrt()
    return image_features / scale


def _count_placeholders(input_ids: torch.Tensor, image_token_index: int) -> int:
    return int((input_ids == image_token_index).sum().item())


def _validate_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RPUBackendError(
            f"PaliGemma2 {name} must be an int, got {type(value).__name__}"
        )
    if value < 0:
        raise RPUBackendError(f"PaliGemma2 {name} must be >= 0, got {value}")
    return value


def merge_image_features(
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_features: torch.Tensor,
    image_token_index: int = IMAGE_TOKEN_INDEX,
    num_image_tokens: int = NUM_IMAGE_TOKENS,
) -> torch.Tensor:
    """Replace image-token placeholders in text embeddings with image features."""
    if input_ids.ndim != 2:
        raise RPUBackendError(
            "PaliGemma2 input_ids must be rank 2, got "
            f"shape {tuple(input_ids.shape)}"
        )
    if inputs_embeds.ndim != 3:
        raise RPUBackendError(
            "PaliGemma2 inputs_embeds must be rank 3, got "
            f"shape {tuple(inputs_embeds.shape)}"
        )
    if image_features.ndim != 3:
        raise RPUBackendError(
            "PaliGemma2 image_features must be rank 3, got "
            f"shape {tuple(image_features.shape)}"
        )

    if tuple(input_ids.shape) != tuple(inputs_embeds.shape[:2]):
        raise RPUBackendError(
            "PaliGemma2 input_ids shape must match inputs_embeds batch/sequence "
            f"shape, got input_ids {tuple(input_ids.shape)} and "
            f"inputs_embeds {tuple(inputs_embeds.shape)}"
        )

    batch_size = int(input_ids.shape[0])
    if int(image_features.shape[0]) != batch_size:
        raise RPUBackendError(
            "PaliGemma2 image_features batch must match input_ids batch, got "
            f"image_features batch {int(image_features.shape[0])} and "
            f"input_ids batch {batch_size}"
        )

    if int(image_features.shape[1]) != int(num_image_tokens):
        raise RPUBackendError(
            "PaliGemma2 image_features token count must match num_image_tokens, "
            f"got {int(image_features.shape[1])} and expected "
            f"{int(num_image_tokens)}"
        )

    hidden_size = int(inputs_embeds.shape[2])
    if int(image_features.shape[2]) != hidden_size:
        raise RPUBackendError(
            "PaliGemma2 image_features hidden size must match inputs_embeds "
            f"hidden size, got {int(image_features.shape[2])} and "
            f"expected {hidden_size}"
        )

    expected_placeholders = batch_size * int(num_image_tokens)
    placeholder_count = _count_placeholders(input_ids, image_token_index)
    if placeholder_count != expected_placeholders:
        raise RPUBackendError(
            "PaliGemma2 placeholder count mismatch: got "
            f"{placeholder_count}, expected {expected_placeholders}"
        )

    placeholder_mask = input_ids == image_token_index
    row_counts = placeholder_mask.sum(dim=1)
    if not bool((row_counts == int(num_image_tokens)).all().item()):
        raise RPUBackendError(
            "PaliGemma2 placeholder count mismatch per row: got "
            f"{[int(count) for count in row_counts.tolist()]}, expected "
            f"{int(num_image_tokens)} per row"
        )

    mask = (
        placeholder_mask
        .to(device=inputs_embeds.device)
        .unsqueeze(-1)
        .expand_as(inputs_embeds)
    )
    src = image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    return inputs_embeds.masked_scatter(mask, src.reshape(-1))


def reverse_token_types(token_type_ids: torch.Tensor | None) -> torch.Tensor:
    if token_type_ids is None:
        raise RPUBackendError("PaliGemma2 token_type_ids are required")
    return 1 - token_type_ids


def make_position_ids(cache_position: torch.Tensor) -> torch.Tensor:
    if cache_position.ndim != 1:
        raise RPUBackendError(
            "PaliGemma2 cache_position must be rank 1, got "
            f"shape {tuple(cache_position.shape)}"
        )
    return (cache_position + 1).unsqueeze(0)


def validate_generate_inputs(
    *,
    pixel_values: torch.Tensor | None,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor | None,
    max_new_tokens: int,
    max_seq_len: int,
) -> None:
    """Validate first-version PaliGemma2 generate input constraints."""
    if pixel_values is None:
        raise RPUBackendError("PaliGemma2 generate requires pixel_values")
    if token_type_ids is None:
        raise RPUBackendError("PaliGemma2 generate requires token_type_ids")

    if input_ids.ndim != 2:
        raise RPUBackendError(
            "PaliGemma2 input_ids must be rank 2, got "
            f"shape {tuple(input_ids.shape)}"
        )
    if int(input_ids.shape[0]) != 1:
        raise RPUBackendError(
            "PaliGemma2 input_ids batch must be 1, got "
            f"{int(input_ids.shape[0])}"
        )

    if tuple(attention_mask.shape) != tuple(input_ids.shape):
        raise RPUBackendError(
            "PaliGemma2 attention_mask shape must match input_ids shape, got "
            f"{tuple(attention_mask.shape)} and {tuple(input_ids.shape)}"
        )
    if tuple(token_type_ids.shape) != tuple(input_ids.shape):
        raise RPUBackendError(
            "PaliGemma2 token_type_ids shape must match input_ids shape, got "
            f"{tuple(token_type_ids.shape)} and {tuple(input_ids.shape)}"
        )

    if pixel_values.ndim != 4:
        raise RPUBackendError(
            "PaliGemma2 pixel_values must be rank 4, got "
            f"shape {tuple(pixel_values.shape)}"
        )
    if int(pixel_values.shape[0]) != 1:
        raise RPUBackendError(
            "PaliGemma2 pixel_values batch must be 1 image, got "
            f"{int(pixel_values.shape[0])}"
        )

    checked_max_new_tokens = _validate_non_negative_int(
        "max_new_tokens",
        max_new_tokens,
    )
    total_tokens = int(input_ids.shape[1]) + checked_max_new_tokens
    if total_tokens > int(max_seq_len):
        raise RPUBackendError(
            "PaliGemma2 max_seq_len overflow: input_ids length "
            f"{int(input_ids.shape[1])} + max_new_tokens "
            f"{checked_max_new_tokens} "
            f"= {total_tokens} exceeds max_seq_len {int(max_seq_len)}"
        )
