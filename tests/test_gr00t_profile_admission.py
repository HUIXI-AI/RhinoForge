from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest
import torch

from rpu_backend.adapters.gr00t import runtime as gr00t_runtime
from rpu_backend import model_registry


ROOT = Path(__file__).resolve().parents[1]


def _inputs(profile: str, sequence_length: int) -> dict[str, torch.Tensor]:
    envelope = gr00t_runtime._GR00T_PROFILE_IDENTITIES[profile]["input_envelope"]
    images = envelope["images_per_request"]
    return {
        "input_ids": torch.zeros((1, sequence_length), dtype=torch.long),
        "attention_mask": torch.ones((1, sequence_length), dtype=torch.long),
        "pixel_values": torch.zeros((images * 256, 1), dtype=torch.float32),
        "image_grid_thw": torch.tensor(
            [envelope["image_grid_thw"]] * images, dtype=torch.long
        ),
        "state": torch.zeros((1, 1, 132), dtype=torch.float32),
    }


def _build_kwargs(profile: str, inputs) -> dict:
    identity = gr00t_runtime._GR00T_PROFILE_IDENTITIES[profile]
    return {
        "profile": profile,
        "checkpoint_repository": identity["checkpoint_repository"],
        "checkpoint_revision": identity["checkpoint_revision"],
        "processor_config_sha256": identity["processor_config_sha256"],
        "statistics_sha256": identity["statistics_sha256"],
        "cosmos_repository": identity["cosmos_repository"],
        "cosmos_revision": identity["cosmos_revision"],
        "cosmos_manifest_sha256": identity["cosmos_manifest_sha256"],
        "embodiment_id": identity["embodiment_id"],
        "input_envelope": identity["input_envelope"],
        "admission_inputs": inputs,
    }


def test_gr00t_rope_index_supports_transformers_457_and_5x_signatures() -> None:
    tensors = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 1]], dtype=torch.int32),
        "image_grid_thw": torch.tensor([[1, 16, 16]], dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }
    calls = []

    def legacy(input_ids, image_grid_thw, video_grid_thw, attention_mask):
        calls.append(("4.57", locals()))
        return "legacy-position-ids", None

    def current(
        input_ids,
        image_grid_thw,
        video_grid_thw,
        attention_mask,
        mm_token_type_ids,
    ):
        calls.append(("5.x", locals()))
        return "current-position-ids", None

    assert gr00t_runtime._get_gr00t_rope_index(legacy, **tensors) == (
        "legacy-position-ids",
        None,
    )
    assert "mm_token_type_ids" not in calls[-1][1]
    assert gr00t_runtime._get_gr00t_rope_index(current, **tensors) == (
        "current-position-ids",
        None,
    )
    assert torch.equal(calls[-1][1]["mm_token_type_ids"], tensors["mm_token_type_ids"])
