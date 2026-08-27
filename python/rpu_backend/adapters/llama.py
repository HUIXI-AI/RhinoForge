"""Per-instance Llama-3.2-1B adapter for the shared causal decoder.

Lock contention raises ``RPUBackendError``. Supported profiles use the tuple
``(hidden, intermediate, layers, attention_heads, kv_heads)``;
Llama-3.2-1B is ``(2048, 8192, 16, 32, 8)``.
"""
from __future__ import annotations
import os
import threading
from typing import Any

import torch
import torch.nn as nn

# Install LlamaRMSNorm and LlamaRotaryEmbedding
# class patches at module-load time. Idempotency sentinels guard against
# repeated module-level execution.
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from rpu_backend.runtime.decoder import (
    _install_rmsnorm_class_swap,
    _install_rotary_class_swap,
)


def _idempotent_patch_llama_rmsnorm(rmsnorm_class) -> None:
    if getattr(rmsnorm_class, "_rpu_patched_rmsnorm", False):
        return
    _install_rmsnorm_class_swap(rmsnorm_class)
    rmsnorm_class._rpu_patched_rmsnorm = True


def _idempotent_patch_llama_rotary(rotary_emb_class) -> None:
    if getattr(rotary_emb_class, "_rpu_patched_rotary", False):
        return
    _install_rotary_class_swap(rotary_emb_class)
    rotary_emb_class._rpu_patched_rotary = True


_idempotent_patch_llama_rmsnorm(LlamaRMSNorm)
_idempotent_patch_llama_rotary(LlamaRotaryEmbedding)

# `_install_causal_decoder_forward` lives in `runtime/decoder.py`. The thin
# `patch_llama_model_for_rpu_all_layers_once` wrapper is folded into
# the call site here:
#     _install_causal_decoder_forward(model, arch="llama")
from rpu_backend.runtime.weights import swizzle_model_inplace
from rpu_backend.runtime.decoder import (
    _causal_lm_runtime_complete,
    _cleanup_causal_decoder_install,
    _install_causal_decoder_forward,
)

from rpu_backend.api.errors import UnsupportedModelError, RPUBackendError
from rpu_backend.api.causal_lm import _claim_live_instance
from rpu_backend.runtime.device import extract_to_device_target, is_rpu_device_target
from rpu_backend.runtime.chunk_envelope import ChunkEnvelope, make_lookup

# Certified chunk envelope.
_CHUNK_ENVELOPE = {
    ("llama", 16, 2048): ChunkEnvelope(64, 64),      # Llama-3.2-1B
}
lookup_causal_decoder = make_lookup(
    _CHUNK_ENVELOPE, "rpu_backend/adapters/llama.py::_CHUNK_ENVELOPE")


# Module-level lock serializes swizzle across all Llama models in the process.
_SWIZZLE_LOCK = threading.Lock()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def apply_rpu_runtime(model: nn.Module) -> None:
    """Cold-set hardware-attribute defaults on `model`.

    Called by `LlamaAdapter.to_rpu()` AFTER `.to('rpu')` and AFTER
    `install_hw_attr_validator` so writes flow through the validator.
    Only sets attributes that have NOT been pre-set by the user.
    """
    if not hasattr(model, "_rpu_spm_mode"):
        model._rpu_spm_mode = False
    if not hasattr(model, "_rpu_warmup"):
        model._rpu_warmup = _env_int("RPU_WARMUP", 0)
    if not hasattr(model, "_rpu_debug_export"):
        model._rpu_debug_export = False


# Five-field profile guard.
# (hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads)
# Llama-3.2-1B HF config: (2048, 8192, 16, 32, 8).
# Both num_hidden_layers and num_attention_heads are guarded.
_SUPPORTED_PROFILES: frozenset[tuple[int, int, int, int, int]] = frozenset({
    (2048, 8192, 16, 32, 8),  # Llama-3.2-1B
})


def _config_profile(config) -> tuple[int, int, int, int, int]:
    """Extract the 5-tuple profile from an HF Llama config object."""
    return (
        int(config.hidden_size),
        int(config.intermediate_size),
        int(config.num_hidden_layers),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
    )


def _check_profile(config) -> None:
    """Raise UnsupportedModelError if the 5-tuple profile is not supported."""
    profile = _config_profile(config)
    if profile not in _SUPPORTED_PROFILES:
        raise UnsupportedModelError(
            f"Llama with profile (hidden_size={profile[0]}, "
            f"intermediate_size={profile[1]}, num_hidden_layers={profile[2]}, "
            f"num_attention_heads={profile[3]}, num_key_value_heads={profile[4]}) "
            f"is not supported. Supported profiles: "
            f"{sorted(_SUPPORTED_PROFILES)}. See "
            "docs/api_reference.md#causal-language-models."
        )


class LlamaAdapter:
    """Per-instance adapter for HF `LlamaForCausalLM`.

    It uses the shared all-layers-once runtime without importing model-specific
    code into the runtime layer.
    """

    @classmethod
    def preflight(cls, config) -> None:
        """Config-only fail-fast preflight (called by `from_pretrained`)."""
        _check_profile(config)

    @classmethod
    def preflight_execution(cls, config, execution_config) -> None:
        requested = execution_config.get("prefill", {}).get(
            "chunk_size", "auto"
        )
        if not isinstance(requested, int):
            return
        env = lookup_causal_decoder(
            "llama", int(config.num_hidden_layers), int(config.hidden_size)
        )
        if requested > env.chunk:
            raise UnsupportedModelError(
                f"Llama prefill chunk_size={requested} exceeds this profile's "
                f"certified ceiling {env.chunk}; max certified KV length is "
                f"{env.max_kv_len}. Refusing before loading model weights."
            )

    def __init__(self, model: nn.Module) -> None:
        # Belt-and-suspenders: catches direct-instantiation paths that bypass
        # `RPUModelForCausalLM.from_pretrained.preflight`.
        _check_profile(model.config)
        self.model = model
        self._all_layers_once_handle: int | None = None
        # Mirror Qwen3Adapter pattern: readiness lives on the model so a second
        # adapter on the same instance does not swizzle a second time.
        self._rpu_is_ready: bool = _causal_lm_runtime_complete(model)

        # Idempotent re-wrap guard.
        if getattr(model.to, "__rpu_wrapped__", False):
            return

        original_to = model.to

        def rpu_aware_to(*args: Any, **kwargs: Any):
            target = extract_to_device_target(args, kwargs)
            if target is not None and is_rpu_device_target(
                target, entry_point="LlamaAdapter.model.to"
            ):
                rejected = set(kwargs) - {"device"}
                if rejected:
                    raise ValueError(
                        f"model.to('rpu', ...) does not accept extra kwargs {sorted(rejected)}; "
                        "pass `rpu_execution={'prefill': {'chunk_size': N}}` to "
                        "`RPUModelForCausalLM.from_pretrained(...)` before "
                        "`.to('rpu')` for a cold chunk override."
                    )
                if len(args) > 1:
                    raise ValueError(
                        f"model.to('rpu', *args) does not accept extra positional args ({args[1:]!r}); "
                        "library forces fp16 at load time."
                    )
                return self.to_rpu()
            # Reject non-rpu .to() after swizzle — convert_linear_weights_inplace
            # is irreversible.
            if (
                self._rpu_is_ready
                or getattr(self.model, "_rpu_swizzled", False)
                or getattr(self.model, "_rpu_swizzle_started", False)
            ):
                raise RPUBackendError(
                    f"model.to({target!r}) rejected: weights have been swizzled for RPU "
                    "(irreversible). Reload via `RPUModelForCausalLM.from_pretrained(...)` "
                    "if you need a CPU copy."
                )
            return original_to(*args, **kwargs)

        rpu_aware_to.__rpu_wrapped__ = True
        model.to = rpu_aware_to

    def to_rpu(self) -> nn.Module:
        """Swizzle weights, claim single-handle gate, move to RPU,
        install all-layers-once patch + validator.

        Lock contention raises ``RPUBackendError`` instead of returning a
        partially installed model.
        """
        if _causal_lm_runtime_complete(self.model):
            self._rpu_is_ready = True
            return self.model
        if self._rpu_is_ready or getattr(self.model, "_rpu_swizzled", False):
            self._rpu_is_ready = False
            raise RPUBackendError(
                "LlamaAdapter.to_rpu(): the ready marker exists but decoder "
                "runtime ownership is incomplete. Reload the model instead of "
                "accepting a partial install."
            )

        # Preserve the _rpu_swizzle_started failure-state guard. A prior swizzle that crashed
        # mid-way leaves `_rpu_swizzle_started=True` and `_rpu_swizzled=False`;
        # a naive retry would re-swizzle already-mutated weights (double-swizzle).
        if getattr(self.model, "_rpu_swizzle_started", False):
            raise RPUBackendError(
                "LlamaAdapter.to_rpu(): prior swizzle attempt failed mid-way; "
                "weights are in an undefined state. Reload via "
                "`RPUModelForCausalLM.from_pretrained(...)` before retrying."
            )

        # Raise on lock contention; never return a partial installation.
        if not _SWIZZLE_LOCK.acquire(blocking=False):
            raise RPUBackendError(
                "LlamaAdapter.to_rpu(): another swizzle is in progress in this "
                "process. Gate your caller so .to('rpu') runs once per model."
            )
        try:
            from rpu_backend.runtime.hw_attrs import (
                install_hw_attr_validator, validate_postinstall, validate_preinstall,
            )
            validate_preinstall(self.model)

            _claim_live_instance(self.model)

            self.model._rpu_swizzle_started = True
            inner = self.model.model
            inner_state = vars(inner)
            had_instance_forward = "forward" in inner_state
            original_instance_forward = inner_state.get("forward")
            try:
                # Plain Llama has no specialized Linear leaves, so an empty
                # skip set is intentional.
                swizzle_model_inplace(self.model, skip_names=set())
                original_to = type(self.model).to
                original_to(self.model, "rpu")
                self._all_layers_once_handle = _install_causal_decoder_forward(
                    inner, arch="llama",
                    chunk_envelope_for=lookup_causal_decoder,
                    execution_config=getattr(
                        self.model, "_rpu_execution", None
                    ),
                )

                validate_postinstall(self.model)
                install_hw_attr_validator(self.model)
                apply_rpu_runtime(self.model)
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


from rpu_backend.runtime.registry import register_adapter
register_adapter("LlamaForCausalLM", LlamaAdapter)


# ---------------------------------------------------------------------------
# Adapter-facing entry for Llama discovery and library callers. It delegates
# to the shared causal-decoder implementation.
# ---------------------------------------------------------------------------


def patch_llama_for_causal_lm(model) -> int:
    """Adapter-facing entry — patches a LlamaModel for RPU all-layers-once execution.

    This public adapter API delegates to
    ``runtime.decoder._install_causal_decoder_forward(model, arch='llama')``.

    Prerequisites (mirror Qwen3 path):
      - model.to("rpu") must have been called (typically via `LlamaAdapter.to_rpu()`)
      - swizzle_model_inplace(model) must have been called

    Args:
        model: A LlamaModel instance (base model, NOT LlamaForCausalLM — pass `causal_lm.model`
            if you start from the outer wrapper).

    Returns:
        int: The handle for this model, also stored as `model._rpu_decoder_handle`.
    """
    return _install_causal_decoder_forward(model, arch="llama", chunk_envelope_for=lookup_causal_decoder)
