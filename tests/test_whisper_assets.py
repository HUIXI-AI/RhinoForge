"""Host-only identity checks for the frozen Whisper base asset set."""

import hashlib
import json
from pathlib import Path

import pytest

from rpu_backend.api.errors import UnsupportedModelError
from rpu_backend.adapters.whisper import assets


FIXTURE = Path(__file__).parent / "fixtures" / "whisper_base" / "manifest.json"


def _identity(data: bytes) -> assets.AssetIdentity:
    return assets.AssetIdentity(len(data), hashlib.sha256(data).hexdigest())


def test_frozen_manifest_matches_reviewed_fixture():
    reviewed = json.loads(FIXTURE.read_text())
    assert reviewed["model_id"] == assets.MODEL_ID
    assert reviewed["revision"] == assets.REVISION
    assert reviewed["profile"] == assets.PROFILE
    assert reviewed["assets"] == {
        name: {"bytes": identity.bytes, "sha256": identity.sha256}
        for name, identity in assets.WHISPER_BASE_ASSETS.items()
    }


def test_reference_transcriptions_are_frozen():
    references = json.loads(FIXTURE.read_text())["reference_transcriptions"]
    assert references["engine"] == "sherpa-onnx 1.12.26"
    assert references["test_wavs/0.wav"] == {
        "language": "en",
        "text": " After early nightfall, the yellow lamps would light up here and there, "
                "the squalid quarter of the brothels.",
        "token_ids": [2381, 2440, 1818, 6691, 11, 264, 5566, 34887, 576, 1442,
                      493, 510, 293, 456, 11, 264, 2339, 31509, 6555, 295, 264,
                      18872, 1625, 13],
    }


def test_checkpoint_identity_is_checked_without_reading_whole_files(tmp_path, monkeypatch):
    expected = {"base-encoder.onnx": b"encoder", "base-decoder.onnx": b"decoder",
                "base-tokens.txt": b"tokens"}
    monkeypatch.setattr(assets, "WHISPER_BASE_ASSETS",
                        {name: _identity(data) for name, data in expected.items()})
    for name, data in expected.items():
        (tmp_path / name).write_bytes(data)
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("whole-file read"))

    assert assets.validate_checkpoint(tmp_path) == {
        name: tmp_path / name for name in expected
    }


@pytest.mark.parametrize("case", ["missing-directory", "missing-file", "wrong-size", "wrong-hash"])
def test_checkpoint_mismatch_fails_closed(tmp_path, monkeypatch, case):
    expected = {"base-encoder.onnx": b"encoder", "base-decoder.onnx": b"decoder",
                "base-tokens.txt": b"tokens"}
    monkeypatch.setattr(assets, "WHISPER_BASE_ASSETS",
                        {name: _identity(data) for name, data in expected.items()})
    checkpoint = tmp_path / "missing" if case == "missing-directory" else tmp_path
    if case != "missing-directory":
        for name, data in expected.items():
            (tmp_path / name).write_bytes(data)
        if case == "missing-file":
            (tmp_path / "base-tokens.txt").unlink()
        elif case == "wrong-size":
            (tmp_path / "base-encoder.onnx").write_bytes(b"short")
        elif case == "wrong-hash":
            (tmp_path / "base-encoder.onnx").write_bytes(b"encodeR")

    with pytest.raises(UnsupportedModelError, match="Whisper base"):
        assets.validate_checkpoint(checkpoint)
