"""Minimal CausalLM adapter template for adding a new HF architecture.

Copy this file to ``python/rpu_backend/adapters/<arch>.py`` and fill the
configuration block below. The ``_template`` package is skipped by in-tree
adapter discovery, so this file is safe to keep as a copy source.

This is intentionally smaller than the production Qwen3/Llama adapters. It
shows the required order: fail-fast profile guard, instance-level
``.to("rpu")`` interception, irreversible swizzle, CausalDecoder install,
hardware-attribute validator, and registry binding.
"""
from __future__ import annotations

import os
import threading
from typing import Any

import torch
import torch.nn as nn

from rpu_backend.api.causal_lm import _claim_live_instance
from rpu_backend.runtime.device import extract_to_device_target, is_rpu_device_target
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError
from rpu_backend.runtime.decoder import (
    _causal_lm_runtime_complete,
    _cleanup_causal_decoder_install,
    _install_causal_decoder_forward,
    _install_rmsnorm_class_swap,
    _install_rotary_class_swap,
)
from rpu_backend.runtime.registry import register_adapter
from rpu_backend.runtime.weights import swizzle_model_inplace
from rpu_backend.runtime.chunk_envelope import ChunkEnvelope, make_lookup

# ── Certified chunk envelope — TEMPLATE ──
# Every CausalDecoderModel handle must declare one before its first prefill, or
# the planner rejects the request. Replace the row below with the target
# geometry and measured envelope; see
# ``docs/model_porting.md#8-verification-gates``.
# chunk=0 means "auto is certified up to max_kv_len" — only claim a length you
# have actually run.
_CHUNK_ENVELOPE = {
    ("llama", 2, 128): ChunkEnvelope(128),
}
lookup_causal_decoder = make_lookup(
    _CHUNK_ENVELOPE,
    "rpu_backend/adapters/_template/minimal_causal_lm.py::_CHUNK_ENVELOPE")


# ---------------------------------------------------------------------------
# 1. Fill these constants after copying this file to adapters/<arch>.py.
# ---------------------------------------------------------------------------

# Must match ``AutoConfig.from_pretrained(repo).architectures[0]`` exactly.
HF_ARCH = "YourArchForCausalLM"

# Current runtime decoder accepts "llama" (no q/k norm) or "qwen3" (q/k norm).
# If neither branch describes your model, extend runtime.decoder or C++ first.
DECODER_ARCH = "llama"

# Profile tuple:
# (hidden_size, intermediate_size, num_hidden_layers,
#  num_attention_heads, num_key_value_heads)
SUPPORTED_PROFILES: frozenset[tuple[int, int, int, int, int]] = frozenset({
    # Example only. Replace with the target model's real profile.
    # (2048, 8192, 16, 32, 8),
})

# Fill with classes from the model's HF modeling module, for example:
# from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
# RMSNORM_CLASS = LlamaRMSNorm
# ROTARY_CLASS = LlamaRotaryEmbedding
RMSNORM_CLASS: type | None = None
ROTARY_CLASS: type | None = None

# Plain CausalLM ports usually swizzle every leaf Linear. Add fully qualified
# names here only for specialized Linear modules that must not be transformed.
SKIP_LINEAR_NAMES: set[str] = set()

_SWIZZLE_LOCK = threading.Lock()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _profile(config: Any) -> tuple[int, int, int, int, int]:
    return (
        int(config.hidden_size),
        int(config.intermediate_size),
        int(config.num_hidden_layers),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
    )


def _check_template_filled() -> None:
    if HF_ARCH == "YourArchForCausalLM":
        raise UnsupportedModelError(
            "minimal_causal_lm template is not configured: set HF_ARCH to "
            "config.architectures[0] after copying it to adapters/<arch>.py."
        )
    if DECODER_ARCH not in {"llama", "qwen3"}:
        raise UnsupportedModelError(
            f"{HF_ARCH}: DECODER_ARCH must be 'llama' or 'qwen3' for the "
            f"current runtime decoder, got {DECODER_ARCH!r}."
        )
    if not SUPPORTED_PROFILES:
        raise UnsupportedModelError(
            f"{HF_ARCH}: SUPPORTED_PROFILES is empty. Add the exact config "
            "profile before registering the adapter."
        )
    if RMSNORM_CLASS is None or ROTARY_CLASS is None:
        raise UnsupportedModelError(
            f"{HF_ARCH}: RMSNORM_CLASS and ROTARY_CLASS must be filled with "
            "classes from the target model's HF modeling module."
        )


def _check_profile(config: Any) -> None:
    _check_template_filled()
    profile = _profile(config)
    if profile not in SUPPORTED_PROFILES:
        raise UnsupportedModelError(
            f"{HF_ARCH} profile {profile} is not supported. "
            f"Supported profiles: {sorted(SUPPORTED_PROFILES)}."
        )


def _install_class_patches_once() -> None:
    if RMSNORM_CLASS is not None and not getattr(
        RMSNORM_CLASS, "_rpu_patched_rmsnorm", False
    ):
        _install_rmsnorm_class_swap(RMSNORM_CLASS)
        RMSNORM_CLASS._rpu_patched_rmsnorm = True
    if ROTARY_CLASS is not None and not getattr(
        ROTARY_CLASS, "_rpu_patched_rotary", False
    ):
        _install_rotary_class_swap(ROTARY_CLASS)
        ROTARY_CLASS._rpu_patched_rotary = True


def _apply_runtime_attrs(model: nn.Module) -> None:
    if not hasattr(model, "_rpu_spm_mode"):
        model._rpu_spm_mode = False
    if not hasattr(model, "_rpu_warmup"):
        model._rpu_warmup = _env_int("RPU_WARMUP", 0)
    if not hasattr(model, "_rpu_debug_export"):
        model._rpu_debug_export = False


class MinimalCausalLMAdapter:
    """Smallest adapter shape for a decoder-only HF CausalLM."""

    @classmethod
    def preflight(cls, config: Any) -> None:
        """Reject unsupported profiles before the full HF weight load."""
        _check_profile(config)

    @classmethod
    def preflight_execution(cls, config: Any, execution_config) -> None:
        requested = execution_config.get("prefill", {}).get(
            "chunk_size", "auto"
        )
        if not isinstance(requested, int):
            return
        env = lookup_causal_decoder(
            DECODER_ARCH,
            int(config.num_hidden_layers),
            int(config.hidden_size),
        )
        if env.chunk <= 0 or requested > env.chunk:
            raise UnsupportedModelError(
                f"{HF_ARCH} prefill chunk_size={requested} has no certified "
                f"exact profile; ceiling={env.chunk}, max certified KV "
                f"length={env.max_kv_len}."
            )

    def __init__(self, model: nn.Module) -> None:
        _check_profile(model.config)
        self.model = model
        self._all_layers_once_handle: int | None = None
        self._rpu_is_ready = _causal_lm_runtime_complete(model)

        if getattr(model.to, "__rpu_wrapped__", False):
            return

        original_to = model.to

        def rpu_aware_to(*args: Any, **kwargs: Any):
            target = extract_to_device_target(args, kwargs)

            if target is not None and is_rpu_device_target(
                target, entry_point=f"{HF_ARCH}.model.to"
            ):
                rejected = set(kwargs) - {"device"}
                if rejected:
                    raise ValueError(
                        f"model.to('rpu', ...) does not accept extra kwargs "
                        f"{sorted(rejected)}. Pass execution controls through "
                        "the loader's cold `rpu_execution` argument instead."
                    )
                if len(args) > 1:
                    raise ValueError(
                        "model.to('rpu', *args) accepts only the device target; "
                        "the library loads and swizzles fp16 weights."
                    )
                return self.to_rpu()

            if (
                self._rpu_is_ready
                or getattr(self.model, "_rpu_swizzled", False)
                or getattr(self.model, "_rpu_swizzle_started", False)
            ):
                raise RPUBackendError(
                    f"model.to({target!r}) rejected: weights have been swizzled "
                    "for RPU and cannot be moved back safely. Reload the model "
                    "if you need a CPU copy."
                )

            return original_to(*args, **kwargs)

        rpu_aware_to.__rpu_wrapped__ = True
        model.to = rpu_aware_to

    def to_rpu(self) -> nn.Module:
        if _causal_lm_runtime_complete(self.model):
            self._rpu_is_ready = True
            return self.model
        if self._rpu_is_ready or getattr(self.model, "_rpu_swizzled", False):
            self._rpu_is_ready = False
            raise RPUBackendError(
                f"{HF_ARCH}.to_rpu(): the ready marker exists but decoder "
                "runtime ownership is incomplete. Reload the model instead of "
                "accepting a partial install."
            )

        if getattr(self.model, "_rpu_swizzle_started", False):
            raise RPUBackendError(
                f"{HF_ARCH}.to_rpu(): a prior swizzle failed mid-way. "
                "The model instance is in an undefined state; reload it before "
                "retrying."
            )

        if not _SWIZZLE_LOCK.acquire(blocking=False):
            raise RPUBackendError(
                f"{HF_ARCH}.to_rpu(): another swizzle is in progress in this "
                "process. Gate the caller so .to('rpu') runs once per model."
            )

        try:
            from rpu_backend.runtime.hw_attrs import (
                install_hw_attr_validator,
                validate_postinstall,
                validate_preinstall,
            )

            validate_preinstall(self.model)

            _claim_live_instance(self.model)
            self.model._rpu_swizzle_started = True
            inner = self.model.model
            inner_state = vars(inner)
            had_instance_forward = "forward" in inner_state
            original_instance_forward = inner_state.get("forward")
            try:
                _install_class_patches_once()
                swizzle_model_inplace(
                    self.model, skip_names=SKIP_LINEAR_NAMES
                )
                original_to = type(self.model).to
                original_to(self.model, "rpu")
                self._all_layers_once_handle = _install_causal_decoder_forward(
                    inner,
                    arch=DECODER_ARCH,
                    chunk_envelope_for=lookup_causal_decoder,
                    execution_config=getattr(
                        self.model, "_rpu_execution", None
                    ),
                )

                validate_postinstall(self.model)
                install_hw_attr_validator(self.model)
                _apply_runtime_attrs(self.model)
                self._rpu_is_ready = True
                self.model._rpu_swizzled = True
            except BaseException:
                self._rpu_is_ready = False
                self._all_layers_once_handle = None
                _cleanup_causal_decoder_install(
                    inner,
                    had_instance_forward=had_instance_forward,
                    original_instance_forward=original_instance_forward,
                )
                raise

            return self.model
        finally:
            _SWIZZLE_LOCK.release()


def register() -> None:
    """Register this adapter after the template constants are filled."""
    _check_template_filled()
    register_adapter(HF_ARCH, MinimalCausalLMAdapter)


# Keep the checked-in template inert. In a copied real adapter, setting HF_ARCH
# to the real HF architecture string enables import-time registration.
if HF_ARCH != "YourArchForCausalLM":
    register()
