from __future__ import annotations

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


def test_gr00t_droid_profiles_are_distinct_and_embodiment_24() -> None:
    identities = gr00t_runtime._GR00T_PROFILE_IDENTITIES
    assert set(identities) == {
        "gr00t-n1.7-3b-droid-zero-shot",
        "gr00t-n1.7-droid-finetuned",
    }
    base = identities["gr00t-n1.7-3b-droid-zero-shot"]
    tuned = identities["gr00t-n1.7-droid-finetuned"]
    assert base["checkpoint_revision"] != tuned["checkpoint_revision"]
    assert base["embodiment_id"] == tuned["embodiment_id"] == 24
    assert base["input_envelope"]["video_delta_indices"] == [-15, 0]
    assert base["input_envelope"]["images_per_request"] == 4
    assert tuned["input_envelope"]["video_delta_indices"] == [0]
    assert tuned["input_envelope"]["images_per_request"] == 2
    assert base["input_envelope"]["processor_input_oracle_sequence_length"] == 277
    assert tuned["input_envelope"]["processor_input_oracle_sequence_length"] == 145
    assert base["input_envelope"]["kvpad16_min_rows"] == 288
    assert tuned["input_envelope"]["kvpad16_min_rows"] == 160
    assert base["input_envelope"]["rhinoforge_cache_capacity"] == 320
    assert tuned["input_envelope"]["rhinoforge_cache_capacity"] == 256


def test_gr00t_base_cache_admits_320_and_rejects_321_before_assets(
    monkeypatch, tmp_path: Path
) -> None:
    profile = "gr00t-n1.7-3b-droid-zero-shot"
    entered = []
    monkeypatch.setattr(
        gr00t_runtime,
        "_normalize_gr00t_execution",
        lambda value: entered.append(value),
    )
    with pytest.raises(ValueError, match="sequence_length=321.*capacity 320"):
        gr00t_runtime.build_gr00t_vla(
            tmp_path, None, **_build_kwargs(profile, _inputs(profile, 321))
        )
    assert entered == []

    def stop_at_asset(_path: Path) -> str:
        raise RuntimeError("asset boundary")

    monkeypatch.setattr(gr00t_runtime, "_sha256_file", stop_at_asset)
    with pytest.raises(RuntimeError, match="asset boundary"):
        gr00t_runtime.build_gr00t_vla(
            tmp_path, None, **_build_kwargs(profile, _inputs(profile, 320))
        )


def test_gr00t_base_requires_local_cosmos_assets_after_checkpoint_checks(
    monkeypatch, tmp_path: Path
) -> None:
    profile = "gr00t-n1.7-3b-droid-zero-shot"
    identity = gr00t_runtime._GR00T_PROFILE_IDENTITIES[profile]
    seen = []

    def digest(path: Path) -> str:
        seen.append(path.name)
        return identity["checkpoint_files"][path.name]

    monkeypatch.setattr(gr00t_runtime, "_sha256_file", digest)
    with pytest.raises(ValueError, match="Cosmos asset directory is required"):
        gr00t_runtime.build_gr00t_vla(
            tmp_path, None, **_build_kwargs(profile, _inputs(profile, 128))
        )
    assert seen == list(identity["checkpoint_files"])


def test_gr00t_finetuned_verifies_exact_cosmos_assets_before_build(
    monkeypatch, tmp_path: Path
) -> None:
    profile = "gr00t-n1.7-droid-finetuned"
    identity = gr00t_runtime._GR00T_PROFILE_IDENTITIES[profile]
    cosmos_dir = tmp_path / "cosmos"
    cosmos_dir.mkdir()
    seen = []

    def digest(path: Path) -> str:
        scope = "cosmos" if path.parent == cosmos_dir else "checkpoint"
        seen.append((scope, path.name))
        if scope == "cosmos":
            return identity["cosmos_files"][path.name]
        return identity["checkpoint_files"][path.name]

    entered = []
    monkeypatch.setattr(gr00t_runtime, "_sha256_file", digest)

    def stop_at_build(value):
        entered.append(value)
        raise RuntimeError("build boundary")

    monkeypatch.setattr(
        gr00t_runtime,
        "_normalize_gr00t_execution",
        stop_at_build,
    )
    with pytest.raises(RuntimeError, match="build boundary"):
        gr00t_runtime.build_gr00t_vla(
            tmp_path, cosmos_dir, **_build_kwargs(profile, _inputs(profile, 128))
        )
    assert seen == [
        ("checkpoint", name) for name in identity["checkpoint_files"]
    ] + [("cosmos", name) for name in identity["cosmos_files"]]
    assert entered == [None]


def test_gr00t_identity_and_input_mismatches_reject_before_assets(
    monkeypatch, tmp_path: Path
) -> None:
    profile = "gr00t-n1.7-droid-finetuned"
    inputs = _inputs(profile, 128)
    kwargs = _build_kwargs(profile, inputs)
    monkeypatch.setattr(
        gr00t_runtime,
        "_sha256_file",
        lambda path: pytest.fail(f"asset read after static mismatch: {path.name}"),
    )

    wrong = dict(kwargs, embodiment_id=20)
    with pytest.raises(ValueError, match="embodiment_id=24"):
        gr00t_runtime.build_gr00t_vla(tmp_path, None, **wrong)

    wrong = dict(kwargs, processor_config_sha256="0" * 64)
    with pytest.raises(ValueError, match="processor_config_sha256"):
        gr00t_runtime.build_gr00t_vla(tmp_path, None, **wrong)

    wrong_envelope = dict(kwargs["input_envelope"], images_per_request=4)
    wrong = dict(kwargs, input_envelope=wrong_envelope)
    with pytest.raises(ValueError, match="input_envelope"):
        gr00t_runtime.build_gr00t_vla(tmp_path, None, **wrong)


def test_gr00t_public_config_and_registry_match_runtime_identity() -> None:
    namespace = runpy.run_path(str(ROOT / "examples" / "gr00t.py"))
    config = namespace["load_config"](ROOT / "examples" / "configs" / "gr00t.toml")
    model = config["model"]
    assert model["profile"] == "gr00t-n1.7-3b-droid-zero-shot"
    assert model["embodiment_id"] == 24
    assert "qwen3vl_alias" not in model
    assert model["cosmos_revision"] == (
        "9ce19a195e423419c349abfc86fd07178b230561"
    )
    assert model["statistics_sha256"] == (
        "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9"
    )

    identity = gr00t_runtime._GR00T_PROFILE_IDENTITIES[model["profile"]]
    for key in (
        "checkpoint_revision",
        "processor_config_sha256",
        "statistics_sha256",
        "cosmos_revision",
        "cosmos_manifest_sha256",
    ):
        assert model[key] == identity[key]
    assert model["input_envelope"] == identity["input_envelope"]

    assert "gr00t-n1d7-3b" not in model_registry.MODELS
    assert model_registry.MODELS["gr00t-n1d7-3b-droid-zero-shot"] == "GR00T-N1.7-3B"
    assert model_registry.MODELS["gr00t-n1d7-droid-finetuned"] == "GR00T-N1.7-DROID"
