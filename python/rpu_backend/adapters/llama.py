"""Llama-3.2-1B per-instance adapter (D-6-17 / Phase 06.1 generic CausalLM fusion).

R2-HIGH-3 (round-2): the lock-contention failure branch RAISES RPUBackendError
(mirrors transformers/qwen3/adapter.py:273-279 + transformers/pi05/policy.py:183-185
+ the now-deleted models/llama_3p2_1b/loader.py:138-142 precedent the round-1 plan
incorrectly proposed to weaken to `return self.model`).

R2-MEDIUM-3 (round-2): 5-tuple profile (hidden, intermediate, num_hidden_layers,
num_attention_heads, num_key_value_heads); Llama-3.2-1B = (2048, 8192, 16, 32, 8).

v5-02 / B2: merged from transformers/llama/adapter.py + models/llama_3p2_1b/runtime.py.
`apply_rpu_runtime` is now a private module helper; the to_rpu() lazy import
collapses to a direct same-module call (G5 / G7-acc-b).
"""
from __future__ import annotations
import os
import threading
from typing import Any

import torch
import torch.nn as nn

from rpu_backend._native_loader import require_native_backend
# v5-11 NS-03b (D-03e): inline class-patch block absorbed from the deleted
# `_internal/patches/llama.py` (~49 LOC). Fires LlamaRMSNorm + LlamaRotaryEmbedding
# class patches at module-load time. Idempotency sentinels guard against
# repeated module-level execution (matches the absorbed module's behavior +
# adapters/qwen3.py mirror pattern + R1 M-2 idempotency invariant).
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from rpu_backend._native_loader import require_native_backend
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

# Shared decoder ownership:
# `_install_causal_decoder_forward` (was `patch_causal_decoder_for_rpu` in
# `_internal/patches/__init__.py:393`) lives in `runtime/decoder.py`. The thin
# `patch_llama_model_for_rpu_all_layers_once` wrapper (D-03d) is folded into
# the call site here:
#     _install_causal_decoder_forward(model, arch="llama")
from rpu_backend.runtime.weights import swizzle_model_inplace
from rpu_backend.runtime.decoder import (
    _bind_causal_decoder_execution_session,
    _causal_lm_runtime_complete,
    _cleanup_causal_decoder_install,
    _enable_causal_decoder_execution_reconfigure,
    _install_causal_decoder_forward,
)

from rpu_backend.api.errors import UnsupportedModelError, RPUBackendError
from rpu_backend.api.causal_lm import _claim_live_instance
from rpu_backend.api._execution import _LLAMA_EXECUTION_SUPPORTED, llama_core_profile
from rpu_backend.runtime.device import extract_to_device_target, is_rpu_device_target
from rpu_backend.runtime.chunk_envelope import ChunkEnvelope, make_lookup
from rpu_backend.runtime.topology import (
    decoder_mlp_intermediate_size, execution_core_count, resolve_decoder_topology,
)

# Explicit chunk bounds for the supported Llama-3.2-1B configuration.
_CHUNK_ENVELOPE = {
    ("llama", 16, 2048): ChunkEnvelope(0, 64),      # Llama-3.2-1B
}
lookup_causal_decoder = make_lookup(
    _CHUNK_ENVELOPE, "rpu_backend/adapters/llama.py::_CHUNK_ENVELOPE")


# Module-level lock serializes swizzle across all Llama models in the process.
# Mirrors the Qwen3Adapter precedent (adapters/qwen3.py:48).
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
    """Cold-set A9 hardware-attribute defaults on `model`.

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


# R2-MEDIUM-3 (round-2): 5-tuple profile guard.
# (hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads)
# Llama-3.2-1B HF config: (2048, 8192, 16, 32, 8).
# Both num_hidden_layers AND num_attention_heads are guarded — the round-1 plan
# dropped one of them and would have admitted unsupported Q-head shapes.
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
            f"is not supported in v4.0. "
            f"Supported profiles: {sorted(_SUPPORTED_PROFILES)}. "
            "See docs/api_reference.md#Migration. "
            "Track progress: v4.0.x backlog."
        )


def _execution_topology(config, execution_config):
    """Cold topology for a reduced budget; ``None`` keeps the eight-core install.

    Budget 8 (explicit or omitted) passes no topology to the shared installer,
    so its native create/weights/Graph identity stays as before. Budgets 4/6
    bind the shared reduced template (attention/MLP/head 4/4/4 or 4/6/4).
    """
    try:
        cores = execution_core_count(execution_config)
        if cores == 8:
            return None
        profile = llama_core_profile(config, cores)
        return resolve_decoder_topology(num_cores=cores, **profile.geometry())
    except ValueError as exc:
        raise UnsupportedModelError(str(exc)) from exc


def _validate_reduced_model_structure(model, topology, *, before_swizzle=False) -> None:
    """Check the actual FP16 modules; ``before_swizzle`` adds the CPU MLP leaves.

    The constructor also sees meta/CPU models; ``to_rpu`` repeats the complete
    check, including the padding inputs, before claiming or mutating anything.
    """
    if topology is None:
        return
    from rpu_backend.runtime.weights import (
        validate_decoder_mlp_padding, validate_decoder_weight_structure,
    )
    try:
        profile = llama_core_profile(model.config, topology.num_cores)
        if resolve_decoder_topology(num_cores=topology.num_cores,
                                    **profile.geometry()) != topology:
            raise ValueError("decoder geometry changed its bound cold topology")
        if any(parameter.dtype != torch.float16 for parameter in model.parameters()):
            raise ValueError("reduced-core Llama requires plain FP16 model weights")
        validate_decoder_weight_structure(model, **profile.weight_geometry(), qk_norm=False)
        if before_swizzle:
            validate_decoder_mlp_padding(
                model, logical_size=profile.intermediate_size,
                physical_size=decoder_mlp_intermediate_size(profile.intermediate_size,
                                                            topology.mlp_tp))
    except ValueError as exc:
        raise UnsupportedModelError(str(exc)) from exc


class LlamaAdapter:
    """Per-instance adapter for HF `LlamaForCausalLM`.

    Phase 06.1 / D-6-17: this adapter lives under `rpu_backend.adapters.llama`
    (v5-02 relocation; previously `rpu_backend.transformers.llama`) so the
    all-layers-once C++ patch can be invoked via `_internal.patches` without
    violating the import-graph contract.
    """

    EXECUTION_SUPPORTED = _LLAMA_EXECUTION_SUPPORTED

    @classmethod
    def preflight(cls, config) -> None:
        """Config-only fail-fast preflight (called by `from_pretrained`)."""
        _check_profile(config)

    @classmethod
    def preflight_execution(cls, config, execution_config) -> None:
        _execution_topology(config, execution_config)
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
                f"chunk ceiling {env.chunk}. Refusing before loading model weights."
            )

    def __init__(self, model: nn.Module) -> None:
        # Belt-and-suspenders: catches direct-instantiation paths that bypass
        # `RPUModelForCausalLM.from_pretrained.preflight`.
        _check_profile(model.config)
        self.model = model
        self._all_layers_once_handle: int | None = None
        self._topology = _execution_topology(
            model.config, getattr(model, "_rpu_execution", None))
        if not _causal_lm_runtime_complete(model):
            _validate_reduced_model_structure(model, self._topology)
        # Mirror Qwen3Adapter pattern: readiness lives on the model so a second
        # adapter on the same instance does NOT re-swizzle (Pitfall 1).
        self._rpu_is_ready: bool = _causal_lm_runtime_complete(model)
        self._execution_session = _bind_causal_decoder_execution_session(
            self, entry_point="LlamaAdapter"
        )

        # Idempotent re-wrap guard (Qwen3 WR-04 pattern).
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
            # is irreversible (Qwen3 iter2 HIGH-3 pattern).
            if (
                self._rpu_is_ready
                or getattr(self.model, "_rpu_swizzled", False)
                or getattr(self.model, "_rpu_swizzle_started", False)
            ):
                raise RPUBackendError(
                    f"model.to({target!r}) rejected: weights have been swizzled for device "
                    "(irreversible). Reload via `RPUModelForCausalLM.from_pretrained(...)` "
                    "if you need a CPU copy."
                )
            return original_to(*args, **kwargs)

        rpu_aware_to.__rpu_wrapped__ = True
        model.to = rpu_aware_to

    def to_rpu(self) -> nn.Module:
        """Swizzle weights, claim single-handle gate, move to device,
        install all-layers-once patch + validator.

        R2-HIGH-3 (round-2): the lock-contention branch RAISES RPUBackendError
        (does NOT silently return self.model — that was the round-1 skeleton bug).
        Mirrors adapters/qwen3.py + adapters/pi05/__init__.py.
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

        # R2-HIGH-3 (round-2): preserve the _rpu_swizzle_started failure-state guard
        # (mirrors the now-deleted loader.py:127-136). A prior swizzle that crashed
        # mid-way leaves `_rpu_swizzle_started=True` and `_rpu_swizzled=False`;
        # a naive retry would re-swizzle already-mutated weights (double-swizzle).
        if getattr(self.model, "_rpu_swizzle_started", False):
            raise RPUBackendError(
                "LlamaAdapter.to_rpu(): prior swizzle attempt failed mid-way; "
                "weights are in an undefined state. Reload via "
                "`RPUModelForCausalLM.from_pretrained(...)` before retrying."
            )

        # R2-HIGH-3 (round-2): RAISE on lock contention (do NOT silently return).
        # Mirrors adapters/qwen3.py + adapters/pi05/__init__.py
        # + the now-deleted models/llama_3p2_1b/loader.py:138-142.
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
            if (_execution_topology(self.model.config, getattr(self.model, "_rpu_execution", None))
                    != self._topology):
                raise UnsupportedModelError("Llama core budget changed after adapter binding")
            _validate_reduced_model_structure(self.model, self._topology, before_swizzle=True)

            require_native_backend()
            _claim_live_instance(self.model)

            self.model._rpu_swizzle_started = True
            if self._topology is not None:
                self.model._rpu_decoder_topology = self._topology
            inner = self.model.model
            inner_state = vars(inner)
            had_instance_forward = "forward" in inner_state
            original_instance_forward = inner_state.get("forward")
            try:
                # Plain Llama has no specialized Linear leaves, so an empty
                # skip set is intentional.
                if self._topology is None:
                    swizzle_model_inplace(self.model, skip_names=set())
                else:
                    # MLP6 needs 8192 -> 8256 (8192 % 96 == 32): zero rows/
                    # columns are added once on CPU before the single swizzle.
                    from rpu_backend.runtime.weights import pad_decoder_mlp_weights
                    logical = self.model.config.intermediate_size
                    pad_decoder_mlp_weights(self.model, logical_size=logical,
                        physical_size=decoder_mlp_intermediate_size(
                            logical, self._topology.mlp_tp))
                    swizzle_model_inplace(
                        self.model, skip_names=set(),
                        attn_num_cores=self._topology.attn_tp,
                        mlp_num_cores=self._topology.mlp_tp,
                        lm_head_num_cores=self._topology.lm_head_tp,
                        execution_core_count=self._topology.num_cores,
                    )
                original_to = type(self.model).to
                original_to(self.model, "rpu")
                self._all_layers_once_handle = _install_causal_decoder_forward(
                    inner, arch="llama",
                    chunk_envelope_for=lookup_causal_decoder,
                    execution_config=getattr(
                        self.model, "_rpu_execution", None
                    ),
                    **({"topology": self._topology}
                       if self._topology is not None else {}),
                )
                _enable_causal_decoder_execution_reconfigure(inner)
                self._execution_session = _bind_causal_decoder_execution_session(
                    self, entry_point="LlamaAdapter"
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
# v5-06 D-02: adapter-facing entry for Llama HF-discovery + library callers.
# Mirrors `_apply_fused_lm_head_for_rpu` on the Qwen3 side. Body is a single
# delegation to the shared `patch_causal_decoder_for_rpu` per D-06 (single
# common body; no parallel implementations).
# ---------------------------------------------------------------------------
# v5-11 NS-03b (Step-4): _install_causal_decoder_forward already imported at top
# of file from rpu_backend.runtime.decoder. The local _patch_causal_decoder_for_rpu
# alias dropped — direct call to the runtime helper.


def patch_llama_for_causal_lm(model) -> int:
    """Adapter-facing entry — patches a LlamaModel for device all-layers-once execution.

    D-02 + D-06: this is the public adapter API. Body is a 1-line delegation to
    `runtime.decoder._install_causal_decoder_forward(model, arch='llama')` — the
    shared body collapsed in v5-06 from the prior `patch_llama_model_for_rpu_all_layers_once`
    sibling and now owned by runtime/decoder.py to share the implementation
    across supported causal-decoder architectures.

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
