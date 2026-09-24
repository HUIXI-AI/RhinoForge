"""PaliGemma2 loader: logical name resolution, profile preflight, CPU FP16 load."""
from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoConfig, PaliGemmaForConditionalGeneration

from rpu_backend.api.errors import RPUUnsupportedDtypeError, UnsupportedModelError
from rpu_backend.model_registry import MODELS, model_path
from rpu_backend.adapters.paligemma.profile import (
    SUPPORTED_LOGICAL_NAMES,
    validate_config_profile,
    validate_loaded_weights,
)

_RESERVED_HF_KWARGS = frozenset({
    "config",
    "torch_dtype",
    "device_map",
    "device",
    "tp_plan",
    "low_cpu_mem_usage",
    "load_in_4bit",
    "load_in_8bit",
    "quantization_config",
})

_CONFIG_KWARGS = frozenset({
    "cache_dir",
    "revision",
    "subfolder",
    "local_files_only",
    "trust_remote_code",
    "token",
    "force_download",
    "proxies",
})

_LOGICAL_HF_REPOS = {
    "paligemma2-3b-pt-224": "google/paligemma2-3b-pt-224",
    "paligemma2-3b-mix-224": "google/paligemma2-3b-mix-224",
}


def _resolve_name_or_path(name_or_path: str) -> str:
    if name_or_path in SUPPORTED_LOGICAL_NAMES:
        path = model_path(name_or_path)
        if path.exists():
            return str(path)
        return _LOGICAL_HF_REPOS[name_or_path]
    if name_or_path in MODELS:
        path = model_path(name_or_path)
        return str(path) if path.exists() else name_or_path
    return os.fspath(name_or_path)


def _validate_dtype_and_kwargs(dtype: torch.dtype, hf_kwargs: dict[str, Any]) -> None:
    if dtype is not torch.float16:
        raise RPUUnsupportedDtypeError(
            f"PaliGemmaPolicy requires torch.float16, got {dtype}."
        )
    hf_keys = set(hf_kwargs)
    collisions = _RESERVED_HF_KWARGS & hf_keys
    if collisions:
        raise ValueError(
            "PaliGemmaPolicy.from_pretrained: hf_kwargs cannot include "
            f"reserved key(s) {sorted(collisions)}; RPU loads on CPU as FP16 "
            "and owns placement, dtype, and quantization decisions."
        )
    unsupported = hf_keys - _CONFIG_KWARGS
    if unsupported:
        raise ValueError(
            "PaliGemmaPolicy.from_pretrained: unsupported hf_kwargs "
            f"{sorted(unsupported)}; the first-version PaliGemma loader only "
            "accepts config/download locating kwargs and rejects config "
            "override kwargs so preflight matches the loaded model."
        )


def load_and_construct_paligemma_policy(
    pretrained_name_or_path: str,
    *,
    dtype: torch.dtype = torch.float16,
    **hf_kwargs: Any,
):
    # Internal callers must observe the same downgrade as the public facade;
    # fail before config resolution or multi-gigabyte HF weight loading.
    from rpu_backend.api.paligemma import _raise_unsupported

    _raise_unsupported()

    # Retained below as recertification scaffold. It is intentionally
    # unreachable while PaliGemma2 remains numerical-blocked.
    _validate_dtype_and_kwargs(dtype, hf_kwargs)
    resolved = _resolve_name_or_path(pretrained_name_or_path)
    cfg_kwargs = {k: v for k, v in hf_kwargs.items() if k in _CONFIG_KWARGS}
    try:
        config = AutoConfig.from_pretrained(resolved, **cfg_kwargs)
    except (OSError, ValueError) as exc:
        raise UnsupportedModelError(
            f"Could not load PaliGemma2 config for {pretrained_name_or_path!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    validate_config_profile(config)

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        resolved,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        **hf_kwargs,
    )
    if hasattr(model, "half"):
        model = model.half()
    model.eval()
    model._paligemma_source_name_or_path = resolved
    validate_loaded_weights(model)

    from rpu_backend.api.paligemma import PaliGemmaPolicy
    return PaliGemmaPolicy.from_hf_model(model)
