"""Frozen asset identity for the supported sherpa-onnx Whisper base export."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from rpu_backend.runtime import UnsupportedModelError


MODEL_ID = "csukuangfj/sherpa-onnx-whisper-base"
REVISION = "bb53ee204431c90d314c1cc08d28d23e5b7927cc"


@dataclass(frozen=True)
class AssetIdentity:
    bytes: int
    sha256: str


WHISPER_BASE_ASSETS = {
    "base-encoder.onnx": AssetIdentity(
        95087154, "5a6b87cb313993f6c9fefec9e7027556f6cb30becddf49655bee36c50ecc12d7"),
    "base-decoder.onnx": AssetIdentity(
        196548998, "8a12c3f6ad65bb5b86d7e6eccc302378f20f9fb2df6cb10747c62895da7ac194"),
    "base-tokens.txt": AssetIdentity(
        816730, "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126"),
}

PROFILE = {
    "batch": 1,
    "precision": "fp16",
    "sample_rate": 16000,
    "max_audio_samples": 480000,
    "mel_bins": 80,
    "mel_frames": 3000,
    "audio_context": 1500,
    "hidden_size": 512,
    "attention_heads": 8,
    "head_dim": 64,
    "encoder_layers": 6,
    "decoder_layers": 6,
    "ffn_size": 2048,
    "vocab_size": 51865,
    "text_context": 448,
    "eot": 50257,
    "sot": 50258,
    "transcribe": 50359,
    "no_timestamps": 50363,
    "language_token_count": 99,
    "language_pairs_sha256": "da890f14c0e20a3fb18ca7c2aa5fc698f0b566d25304788c3f749e83fe5ba058",
}


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_checkpoint(checkpoint: str | Path) -> dict[str, Path]:
    """Reject anything other than the exact reviewed upstream export."""
    root = Path(checkpoint)
    if not root.is_dir():
        raise UnsupportedModelError("Whisper base requires a local checkpoint directory")
    resolved = {}
    for name, identity in WHISPER_BASE_ASSETS.items():
        path = root / name
        if not path.is_file():
            raise UnsupportedModelError(f"Whisper base checkpoint mismatch: {name} is missing")
        if path.stat().st_size != identity.bytes:
            raise UnsupportedModelError(f"Whisper base checkpoint mismatch: {name} size")
        if _sha256_file(path) != identity.sha256:
            raise UnsupportedModelError(f"Whisper base checkpoint mismatch: {name} SHA256")
        resolved[name] = path
    return resolved
