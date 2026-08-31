from __future__ import annotations

import hashlib
import json

import pytest
import torch


def _batch(attention_tokens: int) -> dict[str, torch.Tensor]:
    mask = torch.zeros((1, 200), dtype=torch.bool)
    mask[:, :attention_tokens] = True
    return {
        "observation.images.base_0_rgb": torch.zeros((1, 3, 256, 256)),
        "observation.images.left_wrist_0_rgb": torch.zeros((1, 3, 256, 256)),
        "observation.images.right_wrist_0_rgb": torch.zeros((1, 3, 256, 256)),
        "observation.state": torch.zeros((1, 32)),
        "observation.language.tokens": torch.zeros((1, 200), dtype=torch.int64),
        "observation.language.attention_mask": mask,
    }


def _config() -> dict:
    cameras = (
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    )
    return {
        "type": "pi05",
        "input_features": {
            **{
                name: {"type": "VISUAL", "shape": [3, 224, 224]}
                for name in cameras
            },
            "observation.state": {"type": "STATE", "shape": [32]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [32]}},
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_action_dim": 32,
        "max_state_dim": 32,
        "num_inference_steps": 10,
        "image_resolution": [224, 224],
        "tokenizer_max_length": 200,
    }


def _libero_batch(attention_tokens: int) -> dict[str, torch.Tensor]:
    mask = torch.zeros((1, 200), dtype=torch.bool)
    mask[:, :attention_tokens] = True
    return {
        "observation.images.image": torch.zeros((1, 3, 256, 256)),
        "observation.images.image2": torch.zeros((1, 3, 256, 256)),
        "observation.state": torch.zeros((1, 8)),
        "observation.language.tokens": torch.zeros((1, 200), dtype=torch.int64),
        "observation.language.attention_mask": mask,
    }


def test_pi05_base_exact_identity_and_61_token_fail_before_mutation(
    tmp_path, monkeypatch
) -> None:
    from rpu_backend.adapters.pi05 import loader
    from rpu_backend.api import Pi05Policy

    payloads = {
        "config.json": json.dumps(_config()).encode(),
        "model.safetensors": b"public-model",
        "policy_postprocessor.json": b"{}",
        "policy_preprocessor.json": b"{}",
    }
    files = {}
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
        files[name] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(loader, "_PI05_BASE_FILES", files)

    identity = loader._preflight_pi05_public_profile(
        tmp_path,
        profile=loader.PI05_BASE_PUBLIC_PROFILE,
        admission_batch=_batch(60),
    )
    assert identity == {
        "profile": "pi0.5-base",
        "repository": "lerobot/pi05_base",
        "revision": "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba",
        "manifest_sha256": "4133ecc2e9d9ec04afdff7f251857027b20c414952653dbd5022687e40b9ec23",
        "maximum_attention_tokens": 60,
    }

    load_started = False

    def fail_if_loaded(*args, **kwargs):
        nonlocal load_started
        load_started = True
        raise AssertionError("weight loading must not start")

    monkeypatch.setattr(loader, "_load_and_fp16_cast", fail_if_loaded)
    with pytest.raises(ValueError, match=r"got 61.*before model/Graph mutation"):
        Pi05Policy.from_pretrained(
            tmp_path,
            profile=loader.PI05_BASE_PUBLIC_PROFILE,
            admission_batch=_batch(61),
        )
    assert load_started is False

    lerobot_policy = torch.nn.Module()
    lerobot_policy.model = torch.nn.Module()
    monkeypatch.setattr(
        loader, "_load_and_fp16_cast", lambda *args, **kwargs: lerobot_policy
    )
    wrapped = Pi05Policy.from_pretrained(
        tmp_path,
        profile=loader.PI05_BASE_PUBLIC_PROFILE,
        admission_batch=_batch(60),
    )
    assert wrapped._rpu_public_profile == loader.PI05_BASE_PUBLIC_PROFILE
    assert not hasattr(lerobot_policy, "_rpu_public_profile")
    from rpu_backend.runtime.hw_attrs import validate_preinstall
    validate_preinstall(lerobot_policy)

    policy = Pi05Policy.__new__(Pi05Policy)
    policy._rpu_public_profile = loader.PI05_BASE_PUBLIC_PROFILE
    policy._rpu_ready = False
    policy._adapter = object()
    with pytest.raises(ValueError, match=r"got 61.*before model/Graph mutation"):
        policy.prepare_graphs(_batch(61), num_steps=10)

    out_of_range = _batch(60)
    out_of_range["observation.state"][0, 0] = 1.01
    with pytest.raises(ValueError, match=r"state must stay in \[-1, 1\]"):
        policy.prepare_graphs(out_of_range, num_steps=10)


def test_pi05_libero_exact_identity_and_request_envelope(tmp_path, monkeypatch) -> None:
    from rpu_backend.adapters.pi05 import loader

    payloads = {
        "config.json": b"{}",
        "model.safetensors": b"public-model",
        "model_remapped.safetensors": b"public-remap",
    }
    files = {}
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
        files[name] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(loader, "_PI05_LIBERO_FILES", files)

    identity = loader._preflight_pi05_public_profile(
        tmp_path,
        profile=loader.PI05_LIBERO_PUBLIC_PROFILE,
        admission_batch=_libero_batch(60),
    )
    assert identity["profile"] == "pi0.5-libero-v044"
    assert identity["repository"] == "lerobot/pi05_libero_finetuned_v044"
    assert identity["manifest_sha256"] == loader._PI05_LIBERO_MANIFEST_SHA256

    wrong_profile_batch = _libero_batch(60)
    wrong_profile_batch["observation.state"] = torch.zeros((1, 32))
    with pytest.raises(ValueError, match=r"observation.state must be \(1, 8\)"):
        loader._validate_pi05_public_request(
            wrong_profile_batch,
            profile=loader.PI05_LIBERO_PUBLIC_PROFILE,
            entry_point="test",
            num_steps=10,
        )
