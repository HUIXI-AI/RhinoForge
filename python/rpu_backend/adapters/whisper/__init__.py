"""Frozen Whisper base assets; the RPU runtime is added after capability admission."""

from rpu_backend.adapters.whisper.assets import (
    MODEL_ID,
    PROFILE,
    REVISION,
    validate_checkpoint,
)

__all__ = ["MODEL_ID", "PROFILE", "REVISION", "validate_checkpoint"]
