"""Qwen3 adapter for the shared all-layers-once causal decoder.

It provides CPU-first loading with a ``.to('rpu')`` shortcut, per-instance
installation, deny-by-default profile validation, and single-handle ownership.
"""
from __future__ import annotations
import threading

import torch

from rpu_backend.runtime.log import _LOG

# Weight transformation and decoder installation use their owning modules.
from rpu_backend.runtime.weights import convert_linear_weights_inplace
from rpu_backend.quant.convert_qwen3 import QUANT_PROJ_SUFFIXES
# `_install_causal_decoder_forward` lives in `runtime/decoder.py`. The thin
# `patch_qwen3_model_for_rpu_all_layers_once`
# wrapper is folded into the call site here:
#     _install_causal_decoder_forward(model, arch="qwen3")
from rpu_backend.runtime.decoder import (
    _causal_lm_runtime_complete,
    _canonicalize_plain_text_controls,
    _cleanup_causal_decoder_install,
    _install_causal_decoder_forward,
    _reject_unconsumed_text_padding,
    _text_prefill_execution_plan,
    chunk_policy_key,
    prefill_position_key,
)

from rpu_backend.api.errors import UnsupportedModelError, RPUBackendError
from rpu_backend.api.causal_lm import _claim_live_instance
from rpu_backend.runtime.device import extract_to_device_target, is_rpu_device_target

# Class swaps are guarded below so repeated imports remain idempotent.
from rpu_backend.runtime.decoder import (
    _install_rmsnorm_class_swap,
    _install_rotary_class_swap,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding
from rpu_backend.runtime.chunk_envelope import ChunkEnvelope, make_lookup

# Certified chunk envelope.
# (arch, num_hidden_layers, hidden_size) -> (max_kv_len, safe chunk ceiling).
# A positive ceiling bounds both auto and exact public requests. It is a memory
# safety bound, not proof that every 16-aligned value below it is legal for a
# particular attention geometry/input shape; the native exact planner remains
# the final authority. The framework rejects exact requests on a zero/auto-only
# row. Deny-by-default:
# a size with no row here
# cannot prefill, because the C++ planner refuses an undeclared handle.
# 14B remains W8A16-only; its conservative row is
# intentionally short and pinned while the larger-length sweep stays pending.
_CHUNK_ENVELOPE = {
    ("qwen3", 28, 1024): ChunkEnvelope(1024, 512),   # 0.6b  fp16 + w8a16
    ("qwen3", 28, 2048): ChunkEnvelope(1024, 512),   # 1.7b  fp16
    ("qwen3", 36, 2560): ChunkEnvelope(1024, 256),   # 4b    fp16
    # 8b — 128 is the largest currently selectable/certified chunk; cs=224 is
    # not legal under the current SDPA validity predicate.
    ("qwen3", 36, 4096): ChunkEnvelope(4096, 128),   # 8b    fp16 + w8a16
    ("qwen3", 40, 5120): ChunkEnvelope(192, 32),     # 14b   w8a16 only
}

# Exact values whose attention geometry is admissible at each profile's
# certificate input. The shared runtime still validates the concrete logical
# length, padding and cache position before executing: admissibility can change
# with the tail/position. Keeping this small table prevents a multi-GB load for
# values such as 8B/cs=224 or cs=48 which are 16-aligned and below the nominal
# memory cap but cannot be launched by the current SDPA geometry.
_EXACT_CHUNKS = {
    ("qwen3", 36, 4096): frozenset((16, 32, 64, 128)),
    ("qwen3", 40, 5120): frozenset((16, 32)),
}
lookup_causal_decoder = make_lookup(
    _CHUNK_ENVELOPE, "rpu_backend/adapters/qwen3.py::_CHUNK_ENVELOPE")

# Wrap direct calls in idempotent guards. Sentinel attrs `_rpu_patched_*` make
# repeated module-level execution (e.g., importlib.reload, multi-adapter import
# orderings) a no-op after the first patch.
def _idempotent_patch_qwen3_rmsnorm(rmsnorm_class) -> None:
    if getattr(rmsnorm_class, "_rpu_patched_rmsnorm", False):
        return
    _install_rmsnorm_class_swap(rmsnorm_class)
    rmsnorm_class._rpu_patched_rmsnorm = True


def _idempotent_patch_qwen3_rotary(rotary_emb_class) -> None:
    if getattr(rotary_emb_class, "_rpu_patched_rotary", False):
        return
    _install_rotary_class_swap(rotary_emb_class)
    rotary_emb_class._rpu_patched_rotary = True


_idempotent_patch_qwen3_rmsnorm(Qwen3RMSNorm)
_idempotent_patch_qwen3_rotary(Qwen3RotaryEmbedding)


# The module-level lock serializes swizzle across all
# Qwen3 models in the process. Two racing threads calling .to('rpu')
# on the same (or different) Qwen3 instances can no longer both enter
# the irreversible convert_linear_weights_inplace path — the second
# thread either blocks briefly and short-circuits on the model's
# _rpu_swizzled flag when the first finishes, or (for same-model
# re-entry within one thread) hits the fast-path check above the lock.
#
# The single-handle policy restricts the process to one live RPU model
# at a time, so cross-model serialization adds near-zero contention in
# practice (users never run two concurrent .to('rpu') intentionally).
_SWIZZLE_LOCK = threading.Lock()


# Qwen3 supported envelope.
# Multi-dimensional check: hidden_size alone is too weak (a future Qwen3 release
# could ship hidden=4096 with much larger intermediate_size or num_hidden_layers
# and silently pass the guard). Use a profile registry keyed on
# (hidden_size, intermediate_size, num_hidden_layers, num_key_value_heads).
# Values come from each model's config.json (model_cache/qwen3-*/config.json).
#
_SUPPORTED_PROFILES: frozenset[tuple[int, int, int, int]] = frozenset({
    # (hidden_size, intermediate_size, num_hidden_layers, num_key_value_heads)
    (1024,  3072, 28,  8),  # Qwen3-0.6B
    (2048,  6144, 28,  8),  # Qwen3-1.7B
    (2560,  9728, 36,  8),  # Qwen3-4B
    (4096, 12288, 36,  8),  # Qwen3-8B
})

_W8A16_ONLY_PROFILES: frozenset[tuple[int, int, int, int]] = frozenset({
    # 14B is only admitted for local W8A16 checkpoints. Plain fp16 14B must
    # still fail during config preflight so from_pretrained never materializes
    # the full HF model.
    (5120, 17408, 40,  8),  # Qwen3-14B W8A16
})

# Convenience set used by the lighter-weight unknown-profile fast-fail check.
_SUPPORTED_HIDDEN_SIZES: frozenset[int] = frozenset({
    p[0] for p in (_SUPPORTED_PROFILES | _W8A16_ONLY_PROFILES)
})
_W8A16_PROJ_NAMES: frozenset[str] = frozenset(
    suffix.split(".")[0] for suffix in QUANT_PROJ_SUFFIXES
)


def _config_profile(config) -> tuple[int, int, int, int]:
    """Extract the 4-dim envelope profile from an HF Qwen3 config object."""
    return (
        int(config.hidden_size),
        int(config.intermediate_size),
        int(config.num_hidden_layers),
        int(config.num_key_value_heads),
    )


def _config_is_w8a16(config) -> bool:
    qc = getattr(config, "quant_config", None)
    return isinstance(qc, dict) and qc.get("method") == "w8a16"


def _check_profile(config) -> None:
    """Raise `UnsupportedModelError` if the config's 4-tuple profile is not in
    the supported fp16 or W8A16-only envelope. Used by both
    `Qwen3Adapter.preflight(config)` (the fail-fast preflight before HF
    load) AND `Qwen3Adapter.__init__(model)` (belt-and-suspenders — catches
    direct instantiation paths that bypass `RPUModelForCausalLM.from_pretrained`).
    """
    profile = _config_profile(config)
    if profile in _W8A16_ONLY_PROFILES:
        quant_config = getattr(config, "quant_config", None)
        exact_w8a16 = (
            _config_is_w8a16(config)
            and quant_config.get("quantized_lm_head") is True
            and quant_config.get("lm_head_untied") is True
            and quant_config.get("quantized_embed_tokens") is False
        )
        if not exact_w8a16:
            raise UnsupportedModelError(
                "Qwen3-14B requires the release W8A16 profile with an "
                "untied INT8 lm_head and FP16 embeddings."
            )
    supported = profile in _SUPPORTED_PROFILES or profile in _W8A16_ONLY_PROFILES
    if not supported:
        raise UnsupportedModelError(
            f"Qwen3 with profile (hidden_size={profile[0]}, "
            f"intermediate_size={profile[1]}, num_hidden_layers={profile[2]}, "
            f"num_key_value_heads={profile[3]}) has no certified RPU execution "
            "profile in this build. This deny-by-default result means the "
            "profile/precision/envelope combination has not completed hardware "
            "certification; it is not a claim that a future bounded or quantized "
            "profile cannot be supported. "
            f"Supported fp16 profiles: {sorted(_SUPPORTED_PROFILES)}. "
            f"W8A16-only profiles: {sorted(_W8A16_ONLY_PROFILES)}."
        )


def _detect_and_validate_w8a16(model) -> bool:
    """Return True for local W8A16 Qwen3 models and fail on partial loads."""
    layers = getattr(getattr(model, "model", None), "layers", [])
    expected = len(layers) * len(_W8A16_PROJ_NAMES)
    projections = [
        module for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in _W8A16_PROJ_NAMES
    ]
    int8_count = sum(1 for module in projections if module.weight.dtype == torch.int8)
    if int8_count == 0:
        return False
    scale_count = sum(
        1 for module in projections
        if hasattr(module, "weight_scale")
        and module.weight_scale.dtype == torch.float16
    )
    if expected and len(projections) != expected:
        raise RPUBackendError(
            f"W8A16 Qwen3 validation failed: found {len(projections)} projection "
            f"modules, expected {expected}"
        )
    if int8_count != len(projections) or scale_count != len(projections):
        raise RPUBackendError(
            f"W8A16 Qwen3 validation failed: int8={int8_count}/{len(projections)}, "
            f"fp16 weight_scale={scale_count}/{len(projections)}"
        )
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, torch.nn.Linear) and lm_head.weight.dtype == torch.int8:
        scale = getattr(lm_head, "weight_scale", None)
        if scale is None or scale.dtype != torch.float16:
            raise RPUBackendError(
                "W8A16 Qwen3 validation failed: int8 lm_head.weight requires "
                "fp16 lm_head.weight_scale"
            )
        if scale.numel() != lm_head.weight.size(0):
            raise RPUBackendError(
                f"W8A16 Qwen3 validation failed: lm_head.weight_scale "
                f"numel={scale.numel()} != vocab_size={lm_head.weight.size(0)}"
            )
    embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
    if (
        isinstance(embed_tokens, torch.nn.Embedding)
        and embed_tokens.weight.dtype == torch.int8
    ):
        scale = getattr(embed_tokens, "weight_scale", None)
        if scale is None or scale.dtype != torch.float16:
            raise RPUBackendError(
                "W8A16 Qwen3 validation failed: int8 embed_tokens.weight "
                "requires fp16 model.embed_tokens.weight_scale"
            )
        if scale.numel() != embed_tokens.weight.size(0):
            raise RPUBackendError(
                f"W8A16 Qwen3 validation failed: "
                f"model.embed_tokens.weight_scale numel={scale.numel()} "
                f"!= vocab_size={embed_tokens.weight.size(0)}"
            )
    return True


class Qwen3Adapter:
    """Per-instance adapter for HF `Qwen3ForCausalLM`."""

    @classmethod
    def preflight(cls, config) -> None:
        """Config-only fail-fast preflight.

        Called by `RPUModelForCausalLM.from_pretrained` BEFORE the full HF
        model load so unsupported dtype/profile combinations raise within
        milliseconds (one JSON parse) rather than minutes (full weight load).
        """
        _check_profile(config)

    @classmethod
    def preflight_execution(cls, config, execution_config) -> None:
        requested = execution_config.get("prefill", {}).get("chunk_size", "auto")
        if not isinstance(requested, int):
            return
        key = ("qwen3", int(config.num_hidden_layers), int(config.hidden_size))
        env = lookup_causal_decoder(*key)
        if env.chunk > 0 and requested > env.chunk:
            raise UnsupportedModelError(
                f"Qwen3 prefill chunk_size={requested} exceeds this profile's "
                f"certified ceiling {env.chunk}; max certified KV length is "
                f"{env.max_kv_len}. Refusing before loading model weights."
            )
        admissible = _EXACT_CHUNKS.get(key)
        if admissible is not None and requested not in admissible:
            choices = ", ".join(str(value) for value in sorted(admissible))
            raise UnsupportedModelError(
                f"Qwen3 prefill chunk_size={requested} is 16-aligned and within "
                f"the memory ceiling, but is not admissible for this profile's "
                f"current SDPA geometry. Choose one of {{{choices}}} or 'auto'. "
                "The concrete input length and cache position are validated "
                "again by the exact planner. Refusing before loading model weights."
            )

    def __init__(self, model):
        # Validate the envelope before any
        # mutation (so an unsupported profile raises before swizzle), even on direct adapter
        # instantiation paths that bypass RPUModelForCausalLM.from_pretrained.
        _check_profile(model.config)
        self.model = model
        self._all_layers_once_handle: int | None = None
        self._is_w8a16 = _detect_and_validate_w8a16(model)
        if _config_profile(model.config) in _W8A16_ONLY_PROFILES:
            lm_head = getattr(model, "lm_head", None)
            embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
            if (
                not self._is_w8a16
                or not isinstance(lm_head, torch.nn.Linear)
                or lm_head.weight.dtype != torch.int8
                or not isinstance(embed_tokens, torch.nn.Embedding)
                or embed_tokens.weight.dtype != torch.float16
            ):
                raise RPUBackendError(
                    "Qwen3-14B checkpoint tensors do not match the release "
                    "W8A16 profile: decoder projections and lm_head must be "
                    "INT8, and embeddings must remain FP16."
                )
        # Readiness must live on the model, not the
        # adapter. Otherwise a second `Qwen3Adapter(model)` construction on
        # an already-swizzled model creates a fresh adapter with
        # `_rpu_is_ready=False`; `to_rpu()` then calls
        # `convert_linear_weights_inplace` on already-swizzled weights →
        # double-swizzle. Adopt the existing per-model flag
        # if present; otherwise start False.
        self._rpu_is_ready: bool = _causal_lm_runtime_complete(model)

        # Install a `.to('rpu')` interceptor on the
        # model instance so user code `model.to('rpu')` triggers `to_rpu()`.
        # This monkey-patches the bound method on this single instance only — does
        # NOT mutate the HF class, does NOT affect other model instances.
        #
        # Detect an already-wrapped `model.to` (from a
        # prior `Qwen3Adapter(model)` on the same instance) via a sentinel
        # attribute on the wrapper function. Without this guard, repeated
        # construction stacks interceptors: the second __init__ captures the
        # first's `rpu_aware_to` as `original_to`, so `model.to('cpu')` would
        # chain through two interceptors. The single-handle gate only
        # catches two DIFFERENT live models on RPU — not double-wrap on the
        # same instance. Short-circuit the `model.to` rewiring; the rest of
        # __init__ runs normally (cheap rebind of self.model / self._rpu_is_ready
        # is safe — to_rpu()'s own `_rpu_is_ready` guard prevents re-swizzle).
        if getattr(model.to, "__rpu_wrapped__", False):
            return

        original_to = model.to

        # The interceptor handles device routing only. Execution planning is a
        # cold constructor concern: pass ``rpu_execution`` to the public loader.
        def rpu_aware_to(*args, **kwargs):
            # Detect target device: positional `to('rpu')`, `to(torch.device('rpu'))`,
            # or kwarg `device=`. ANY other positional/keyword args (e.g. dtype)
            # forward to the original `.to`.
            target = extract_to_device_target(args, kwargs)
            if target is not None and is_rpu_device_target(
                target, entry_point="Qwen3Adapter.model.to"
            ):
                # Reject unknown kwargs the user may expect us to honor (e.g.
                # `chunk_size=`) so silent-no-op bugs surface immediately.
                rejected = set(kwargs) - {"device"}
                if rejected:
                    raise ValueError(
                        f"model.to('rpu', ...) does not accept extra kwargs {sorted(rejected)}; "
                        "pass `rpu_execution={'prefill': {'chunk_size': N}}` to "
                        "`RPUModelForCausalLM.from_pretrained(...)` before "
                        "`.to('rpu')` for a cold chunk override."
                    )
                # Drop positional after 'rpu' too (e.g. `to('rpu', torch.float16)`):
                # we already loaded as fp16 in from_pretrained — dtype changes here
                # would invalidate the swizzle.
                if len(args) > 1:
                    raise ValueError(
                        f"model.to('rpu', *args) does not accept extra positional args ({args[1:]!r}); "
                        "library forces fp16 at load time. Use .to('rpu') alone."
                    )
                return self.to_rpu()
            # Once the adapter has swizzled
            # weights for RPU, `convert_linear_weights_inplace` is IRREVERSIBLE —
            # moving the model back to CPU and then to RPU again would NOT
            # re-swizzle (no-op via _rpu_is_ready) and the CPU forward path
            # would crash on swizzled weights. Reject any non-rpu target after
            # to_rpu() so the failure surfaces at the .to() call site.
            # Raise the typed `RPUBackendError` so callers can
            # catch with `except RPUBackendError` rather than the broader
            # `RuntimeError`.
            # This closure captures the creating adapter's
            # `self._rpu_is_ready`. If a SECOND Qwen3Adapter was
            # constructed on the same model and its `to_rpu()` swizzled
            # the weights, adapter1's flag stays False (closure is stale)
            # but `self.model._rpu_swizzled` is True. Check BOTH so the
            # reject-after-swizzle gate fires regardless of which adapter
            # drove the swizzle.
            if (
                self._rpu_is_ready
                or getattr(self.model, "_rpu_swizzled", False)
                or getattr(self.model, "_rpu_swizzle_started", False)
            ):
                from rpu_backend.api.errors import RPUBackendError as _RPUBackendError
                raise _RPUBackendError(
                    f"model.to({target!r}) rejected: weights have been swizzled for RPU "
                    "(an irreversible operation per `convert_linear_weights_inplace` semantics). "
                    "Reload the model fresh via `RPUModelForCausalLM.from_pretrained(...)` if you "
                    "need a CPU copy."
                )
            return original_to(*args, **kwargs)

        # Tag the wrapper so a subsequent `Qwen3Adapter(model)`
        # on the same instance detects the existing wrap via
        # `getattr(model.to, "__rpu_wrapped__", False)` and short-circuits.
        rpu_aware_to.__rpu_wrapped__ = True

        # Bind onto the instance (NOT the class) — survives only as long as `model` lives.
        # `nn.Module.__setattr__` falls through to `super().__setattr__` for plain
        # functions, so this assignment is safe.
        model.to = rpu_aware_to

    def to_rpu(self):
        """Step A → B → C: convert weights, all-layers-once patch, claim live-handle, return RPU-ready model.

        Idempotent for the same model instance. If
        `to_rpu()` was already called on this adapter, returns `self.model`
        without re-running weight swizzle (which would corrupt already-swizzled
        weights) or re-claiming the live-handle gate (which would raise against
        ourself).

        Double-swizzle guard: SKIP_LINEAR_NAMES.
        Qwen3 has NO specialized `.dense` layers (unlike AdaRMS / QwenPI05). We pass
        `skip_names=set()` explicitly so future readers see the deliberate choice.

        The default `skip_names=None` in `convert_linear_weights_inplace`
        merges `{'dense'}` into the skip set as a double-swizzle guard. For Qwen3 with no `.dense`
        leaves, this would be harmless either way, but explicit `set()` documents intent and
        keeps adapter-level intent explicit alongside Pi0.5 (`skip_names={'dense'}`)
        and QwenPI05 (`skip_names={'cond', 'dense'}`).
        """
        if _causal_lm_runtime_complete(self.model):
            self._rpu_is_ready = True
            return self.model
        if self._rpu_is_ready or getattr(self.model, "_rpu_swizzled", False):
            self._rpu_is_ready = False
            raise RPUBackendError(
                "Qwen3Adapter.to_rpu(): the ready marker exists but decoder "
                "runtime ownership is incomplete. Reload the model instead of "
                "accepting a partial install."
            )

        # Fail loud if a prior swizzle attempt started mutation
        # but crashed mid-way. `convert_linear_weights_inplace` mutates
        # Linear.weight.data in place. If `.to('rpu')` or
        # `patch_qwen3_model_for_rpu_all_layers_once` raised AFTER that,
        # `_rpu_swizzled` stays False but weights are already swizzled. A
        # naive retry would re-swizzle already-mutated weights (double-
        # swizzle corruption). `_rpu_swizzle_started` is stamped BEFORE
        # mutation and never cleared — a retry with started=True AND
        # swizzled=False means the model is in an undefined partial state
        # and the caller MUST reload from_pretrained.
        if getattr(self.model, "_rpu_swizzle_started", False):
            raise RPUBackendError(
                "Qwen3Adapter.to_rpu(): prior swizzle attempt on this model failed "
                "mid-way; weights are in an undefined state (some Linear.weight.data "
                "tensors were mutated by convert_linear_weights_inplace before the "
                "error). Reload the model fresh via "
                "`RPUModelForCausalLM.from_pretrained(...)` before retrying; do NOT "
                "call .to('rpu') on the broken instance."
            )

        # The module-level lock serializes the entire swizzle critical section
        # across all Qwen3 models. A check-then-set flag alone has a race window
        # between the read and write — two threads could both pass the
        # check before either write landed, then both reach the irreversible
        # convert_linear_weights_inplace. Holding _SWIZZLE_LOCK for the
        # whole body eliminates that window.
        #
        # Non-blocking acquire: if another thread is mid-swizzle, fail loud
        # rather than silently queueing. Users never intentionally run two
        # concurrent swizzles; a contended lock indicates a caller bug
        # (double-dispatch, background .to('rpu'), etc.) best surfaced
        # immediately.
        if not _SWIZZLE_LOCK.acquire(blocking=False):
            raise RPUBackendError(
                "Qwen3Adapter.to_rpu(): another Qwen3 swizzle is currently in "
                "progress in this process (module-level serialization). Gate your "
                "caller so .to('rpu') runs once per model and does not race "
                "with another RPU model load."
            )
        try:
            from rpu_backend.runtime.hw_attrs import (
                install_hw_attr_validator,
                validate_postinstall,
                validate_preinstall,
            )
            validate_preinstall(self.model)

            # Claim the single-handle gate before mutation so a second concurrent
            # .to('rpu') on a DIFFERENT model raises before we touch any weights.
            # _claim_live_instance is idempotent for the same model; only a
            # different live model triggers RPUSingleHandleError.
            _claim_live_instance(self.model)

            # Stamp started BEFORE any mutation. If anything below raises,
            # this flag stays set and the HIGH-#2 check at function-entry
            # above raises on retry telling the caller to reload.
            self.model._rpu_swizzle_started = True
            inner = self.model.model
            inner_state = vars(inner)
            had_instance_forward = "forward" in inner_state
            original_instance_forward = inner_state.get("forward")
            try:
                # Step A: weight conversion and device migration are
                # irreversible on this instance.
                _stash_int8_lm_head_cpu_reference(self.model)
                _patch_int8_embedding_forward(self.model)
                convert_linear_weights_inplace(
                    self.model, skip_names=set()
                )
                original_to = type(self.model).to
                original_to(self.model, "rpu")

                scale_lists = None
                if self._is_w8a16:
                    num_layers = len(inner.layers)
                    scale_lists = (
                        [inner.layers[i].self_attn.q_proj.weight_scale for i in range(num_layers)],
                        [inner.layers[i].self_attn.k_proj.weight_scale for i in range(num_layers)],
                        [inner.layers[i].self_attn.v_proj.weight_scale for i in range(num_layers)],
                        [inner.layers[i].self_attn.o_proj.weight_scale for i in range(num_layers)],
                        [inner.layers[i].mlp.gate_proj.weight_scale for i in range(num_layers)],
                        [inner.layers[i].mlp.up_proj.weight_scale for i in range(num_layers)],
                        [inner.layers[i].mlp.down_proj.weight_scale for i in range(num_layers)],
                    )
                self._all_layers_once_handle = _install_causal_decoder_forward(
                    inner, arch="qwen3", scale_lists=scale_lists,
                    chunk_envelope_for=lookup_causal_decoder,
                    execution_config=getattr(
                        self.model, "_rpu_execution", None
                    ),
                )

                # Readiness is published only after the full A9 install. A
                # failure before this point remains poisoned by
                # `_rpu_swizzle_started` and cannot be retried.
                validate_postinstall(self.model)
                install_hw_attr_validator(self.model)
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
            # Release the lock regardless of outcome. `_rpu_swizzle_started`
            # is NOT cleared here — on success it stays True alongside
            # _rpu_swizzled; on failure it stays True WITHOUT
            # _rpu_swizzled, tripping the partial-state check on retry.
            _SWIZZLE_LOCK.release()


from rpu_backend.runtime.registry import register_adapter
register_adapter("Qwen3ForCausalLM", Qwen3Adapter)


# =====================================================================
# CausalLM-specific helpers for the fused-lm-head path use
# `_rpu_decoder_handle` as the shared handle attribute.
# =====================================================================
from rpu_backend.runtime.weights import transform_linear_weight
from rpu_backend.runtime.decoder import (
    _is_batched_prefill,
    _run_batched_prefill,
    _run_causal_decoder_forward,
)


def _stash_int8_lm_head_cpu_reference(causal_lm) -> None:
    """Keep an unswizzled CPU int8 lm_head copy for generate prefill logits."""
    lm_head = getattr(causal_lm, "lm_head", None)
    if not isinstance(lm_head, torch.nn.Linear):
        return
    lm_w = lm_head.weight.detach()
    if lm_w.dtype != torch.int8:
        return
    if hasattr(causal_lm, "_rpu_lm_head_int8_cpu_ref"):
        return
    scale = getattr(lm_head, "weight_scale", None)
    if scale is None or scale.dtype != torch.float16:
        raise RPUBackendError(
            "int8 lm_head.weight requires fp16 lm_head.weight_scale"
        )
    if scale.numel() != lm_w.size(0):
        raise RPUBackendError(
            f"lm_head.weight_scale numel={scale.numel()} must match "
            f"vocab_size={lm_w.size(0)}"
        )
    causal_lm._rpu_lm_head_int8_cpu_ref = lm_w.to("cpu").contiguous()
    causal_lm._rpu_lm_head_scale_cpu_ref = scale.detach().to(
        device="cpu", dtype=torch.float16
    ).contiguous()


def _int8_lm_head_prefill_logits(causal_lm, hidden: torch.Tensor) -> torch.Tensor:
    """Reference first-token lm_head for int8 lm_head prefill.

    The optimized path is decode-only and runs inside C++ while seq_len==1.
    For prefill, we support the generate path (`logits_to_keep=1`) by using
    the unswizzled CPU int8 reference stashed before RPU weight swizzling.
    """
    w_int8 = getattr(causal_lm, "_rpu_lm_head_int8_cpu_ref", None)
    scale = getattr(causal_lm, "_rpu_lm_head_scale_cpu_ref", None)
    if w_int8 is None or scale is None:
        raise RPUBackendError(
            "int8 lm_head prefill logits require an unswizzled CPU reference; "
            "reload the model and call .to('rpu') once before generation"
        )
    hidden_cpu = hidden.to("cpu", dtype=torch.float16).contiguous()
    weight = getattr(causal_lm, "_rpu_lm_head_fp16_cpu_ref", None)
    if weight is None:
        weight = (
            w_int8.to(torch.float32)
            * scale.to(torch.float32).unsqueeze(1)
        ).to(torch.float16).contiguous()
        causal_lm._rpu_lm_head_fp16_cpu_ref = weight
    logits_cpu = torch.nn.functional.linear(hidden_cpu, weight)
    return logits_cpu.to(hidden.device)


def _stash_int8_embedding_cpu_reference(causal_lm) -> None:
    """Keep raw CPU int8 embedding rows for dequantized embedding lookup."""
    embed_tokens = getattr(getattr(causal_lm, "model", None), "embed_tokens", None)
    if not isinstance(embed_tokens, torch.nn.Embedding):
        return
    embed_w = embed_tokens.weight.detach()
    if embed_w.dtype != torch.int8:
        return
    if hasattr(causal_lm, "_rpu_embed_tokens_int8_cpu_ref"):
        return
    scale = getattr(embed_tokens, "weight_scale", None)
    if scale is None or scale.dtype != torch.float16:
        raise RPUBackendError(
            "int8 embed_tokens.weight requires fp16 "
            "model.embed_tokens.weight_scale"
        )
    if scale.numel() != embed_w.size(0):
        raise RPUBackendError(
            f"model.embed_tokens.weight_scale numel={scale.numel()} must "
            f"match vocab_size={embed_w.size(0)}"
        )
    causal_lm._rpu_embed_tokens_int8_cpu_ref = embed_w.to("cpu").contiguous()
    causal_lm._rpu_embed_tokens_scale_cpu_ref = scale.detach().to(
        device="cpu", dtype=torch.float16
    ).contiguous()


def _patch_int8_embedding_forward(causal_lm) -> None:
    """Patch int8 Qwen3 embeddings to gather rows and dequantize to fp16."""
    embed_tokens = getattr(getattr(causal_lm, "model", None), "embed_tokens", None)
    if not isinstance(embed_tokens, torch.nn.Embedding):
        return
    if embed_tokens.weight.dtype != torch.int8:
        return
    _stash_int8_embedding_cpu_reference(causal_lm)
    if getattr(embed_tokens, "__rpu_int8_embedding_forward_patched__", False):
        return

    def int8_embedding_forward(input_ids):
        w_int8 = getattr(causal_lm, "_rpu_embed_tokens_int8_cpu_ref", None)
        scale = getattr(causal_lm, "_rpu_embed_tokens_scale_cpu_ref", None)
        if w_int8 is None or scale is None:
            raise RPUBackendError(
                "int8 embed_tokens forward requires a raw CPU reference; "
                "reload the model and call .to('rpu') through Qwen3Adapter"
            )
        ids_cpu = input_ids.to(device="cpu", dtype=torch.long)
        rows = w_int8[ids_cpu]
        row_scale = scale[ids_cpu].to(torch.float32).unsqueeze(-1)
        out = (rows.to(torch.float32) * row_scale).to(torch.float16)
        return out.to(input_ids.device)

    embed_tokens.forward = int8_embedding_forward
    embed_tokens.__rpu_int8_embedding_forward_patched__ = True


def _w8a16_scale_lists_for_qwen3_inner(inner):
    layers = getattr(inner, "layers", [])
    if not layers:
        return None
    first_q = layers[0].self_attn.q_proj
    if getattr(first_q.weight, "dtype", None) != torch.int8:
        return None
    num_layers = len(layers)
    return (
        [layers[i].self_attn.q_proj.weight_scale for i in range(num_layers)],
        [layers[i].self_attn.k_proj.weight_scale for i in range(num_layers)],
        [layers[i].self_attn.v_proj.weight_scale for i in range(num_layers)],
        [layers[i].self_attn.o_proj.weight_scale for i in range(num_layers)],
        [layers[i].mlp.gate_proj.weight_scale for i in range(num_layers)],
        [layers[i].mlp.up_proj.weight_scale for i in range(num_layers)],
        [layers[i].mlp.down_proj.weight_scale for i in range(num_layers)],
    )


def _push_lm_head_weight(causal_lm, handle: int) -> None:
    """Push causal_lm.lm_head.weight to C++ via causal_decoder_set_lm_head.

    The weight must be col-partition swizzled for the SPM GEMM kernel
    (rpu_launch_linear_spm_to_spm_kernel with partition=1). Two cases:

    (a) Weight already on RPU → assumed pre-swizzled (i.e., caller ran
        convert_linear_weights_inplace on the full CausalLM INCLUDING
        lm_head BEFORE .to("rpu")). If lm_head was moved to RPU WITHOUT
        swizzling, the caller has a bug — we cannot detect this case.
    (b) Weight still on CPU → explicitly swizzled here via
        transform_linear_weight(..., partition=1) before moving to RPU.

    CPU weights are transformed here before being uploaded.
    """
    lm_w = causal_lm.lm_head.weight.detach()
    if lm_w.dtype == torch.int8:
        if lm_w.device.type == "rpu" and not hasattr(
            causal_lm, "_rpu_lm_head_int8_cpu_ref"
        ):
            raise RPUBackendError(
                "int8 lm_head is already on RPU but no unswizzled CPU "
                "reference was saved; reload the model and move it through "
                "the Qwen3 RPU adapter before enabling fused lm_head"
            )
        _stash_int8_lm_head_cpu_reference(causal_lm)
        lm_scale = causal_lm.lm_head.weight_scale.detach()
        if lm_w.device.type == "rpu":
            lm_w = lm_w.contiguous()
            lm_scale = lm_scale.to(torch.float16).contiguous()
        else:
            lm_w = transform_linear_weight(lm_w.contiguous(), partition=1).to("rpu")
            lm_scale = lm_scale.to(torch.float16).contiguous().to("rpu")
        causal_lm._rpu_lm_head_w_keepalive = lm_w
        causal_lm._rpu_lm_head_scale_keepalive = lm_scale
        torch.ops.rpu.causal_decoder_set_lm_head(handle, lm_w, lm_scale)
        return

    if lm_w.device.type == "rpu":
        # Already swizzled + on RPU (common path: convert_linear_weights_inplace
        # was called on the full CausalLM before .to("rpu")).
        lm_w = lm_w.to(torch.float16).contiguous()
    else:
        # Still on CPU — need explicit col-partition swizzle before moving.
        lm_w = transform_linear_weight(
            lm_w.to(torch.float16).contiguous(), partition=1
        ).to("rpu")
    causal_lm._rpu_lm_head_w_keepalive = lm_w
    torch.ops.rpu.causal_decoder_set_lm_head(handle, lm_w)

# patch-reason: (e) CausalLM lm-head fuse — §3a (e) module-method-set
def _apply_fused_lm_head_for_rpu(causal_lm) -> int:
    """Patch a Qwen3ForCausalLM instance for decode-only fused lm_head.

    Installs an outer `fused_forward` on `causal_lm` that:
      - decode (seq_len=1): C++ fuses lm_head GEMM into the main graph's last
        layer batch (SPM col-partition). Returns [B, 1, vocab_size] logits.
      - prefill (seq_len>1): C++ returns hidden states, Python runs
        self.lm_head. logits_to_keep=0 → full-sequence logits (HF default);
        logits_to_keep=1 → last-token only (HF generate() fast path).
      - labels / output_attentions / output_hidden_states → NotImplementedError

    Pushes lm_head weight (col-partition swizzled) to C++ once during patching.
    _push_lm_head_weight() handles the swizzle automatically: if the weight is
    already on RPU it's assumed pre-swizzled; if on CPU it's explicitly
    transform_linear_weight(..., partition=1)'d.

    Prerequisites:
      - `causal_lm.model.to("rpu")` called
      - `convert_linear_weights_inplace(causal_lm)` called on the FULL
        CausalLM (including lm_head), OR lm_head.weight still on CPU
        (in which case _push_lm_head_weight swizzles it on the fly)

    Idempotent: re-calling on the same instance destroys the old C++ handle
    (via the base helper's guard) and reinstalls fused_forward.

    Args:
        causal_lm: A Qwen3ForCausalLM instance

    Returns:
        int: The C++ handle (= causal_lm.model._rpu_decoder_handle)
    """
    try:
        from transformers.modeling_outputs import CausalLMOutputWithPast
    except ImportError as e:
        raise RuntimeError(
            "_apply_fused_lm_head_for_rpu requires transformers "
            "with CausalLMOutputWithPast"
        ) from e

    # 1. Base patch. If `.to("rpu")` already installed the inner decoder,
    #    reuse that handle so W8A16 scale_lists are preserved. Otherwise install
    #    it here, deriving W8A16 scale_lists from the already-loaded modules.
    if not hasattr(causal_lm.model, "_rpu_decoder_handle"):
        _install_causal_decoder_forward(
            causal_lm.model,
            arch="qwen3",
            scale_lists=_w8a16_scale_lists_for_qwen3_inner(causal_lm.model),
            chunk_envelope_for=lookup_causal_decoder,
        )
    handle = causal_lm.model._rpu_decoder_handle  # single authoritative handle

    # 2. Push lm_head weight to the C++ side (state tracking for future
    #    C++ post_graph; currently used only as API symmetry).
    _push_lm_head_weight(causal_lm, handle)

    from contextlib import nullcontext as _nullcontext
    import rpu_backend as _rb
    from rpu_backend.api.cache import RPUCache as _RPUCache_for_sig
    _GraphSignature = _rb.graph.GraphSignature

    # 3. Install fused_forward on the Qwen3ForCausalLM.
    #    Signature explicitly lists labels / logits_to_keep / output_*
    #    / return_dict so HF generate() can inspect.signature it correctly.
    def fused_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ):
        # ── fail-fast on unsupported HF forward paths ──────────────────────
        if labels is not None:
            raise NotImplementedError(
                "fused_lm_head path does not support `labels` (no loss computation). "
                "Use the unfused path (_install_causal_decoder_forward(arch='qwen3')) "
                "for training/labels workflows."
            )
        if output_attentions:
            raise NotImplementedError(
                "fused_lm_head path: output_attentions=True not supported."
            )
        if output_hidden_states:
            raise NotImplementedError(
                "fused_lm_head path: output_hidden_states=True not supported."
            )
        # HF generate() inspects `logits_to_keep` in the forward signature and
        # passes 1 for the last-token-only optimization. Our fused path only
        # produces last-token logits, so accept 0 (HF default) and 1 (generate)
        # and reject anything else.
        if logits_to_keep not in (0, 1):
            raise NotImplementedError(
                f"fused_lm_head path only supports logits_to_keep in (0, 1) "
                f"(last-token only), got {logits_to_keep}. "
                f"Use the unfused path for multi-token logits."
            )

        # Single authoritative handle lookup each call, no mirror field.
        h = self.model._rpu_decoder_handle

        prefill_plan = None
        batched_prefill = _is_batched_prefill(input_ids, inputs_embeds)
        if isinstance(past_key_values, _RPUCache_for_sig):
            if inputs_embeds is not None:
                _batch, _seq_len = (int(inputs_embeds.shape[0]),
                                    int(inputs_embeds.shape[1]))
            elif input_ids is not None:
                _batch, _seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
            else:
                _batch, _seq_len = 1, 0
            if not batched_prefill:
                attention_mask, position_ids, cache_position = (
                    _canonicalize_plain_text_controls(
                        _seq_len,
                        past_key_values,
                        attention_mask,
                        position_ids,
                        cache_position,
                        batch_size=_batch,
                    )
                )
                _reject_unconsumed_text_padding(
                    self.model,
                    _seq_len,
                    attention_mask,
                    position_ids,
                )
            else:
                if (attention_mask is not None
                        and attention_mask.device.type != "rpu"):
                    attention_mask = attention_mask.to("rpu")
            _execution_len = _seq_len
            _planned_chunk_size = 0
            if (_seq_len > 1
                    and position_ids is None
                    and (attention_mask is None or batched_prefill)):
                prefill_plan = _text_prefill_execution_plan(
                    self.model,
                    h,
                    past_key_values,
                    _seq_len,
                )
                _execution_len, _planned_chunk_size = prefill_plan

            def _capture(batch, seq_len, plan=None):
                execution_len, planned_chunk_size = (
                    (int(plan[0]), int(plan[1]))
                    if plan is not None else (int(seq_len), 0)
                )
                return self.model._rpu_decoder_graph_cache.capture(
                    _GraphSignature(
                        op_id="rpu_causal_decoder_lm_head",
                        shapes=[
                            int(batch),
                            int(seq_len),
                            execution_len,
                            self.model._rpu_decoder_hidden_size,
                        ],
                        dyn_dims=[
                            self.model._rpu_decoder_num_layers,
                            self.model._rpu_decoder_deepstack_hash,
                            int(self.config.vocab_size),
                            chunk_policy_key(
                                self.model._rpu_decoder_handle
                            ),
                            planned_chunk_size,
                            prefill_position_key(
                                execution_len,
                                past_key_values.position,
                            ),
                        ],
                        dtypes=[torch.float16],
                    )
                )
        else:
            # Preserve the runner's precise invalid-cache/input error while
            # keeping any CPU→RPU transfer outside capture.
            if attention_mask is not None and attention_mask.device.type != "rpu":
                attention_mask = attention_mask.to("rpu")
            if position_ids is not None and position_ids.device.type != "rpu":
                position_ids = position_ids.to("rpu")
            def _capture(batch, seq_len, plan=None):
                return _nullcontext()
            _batch, _seq_len = 1, 0

        if _batch > 1 and not bool(self.model._rpu_batch_decode_enabled):
            raise AssertionError(
                "RPU batch > 1 requires the Qwen3 batch-decode capability"
            )

        _runner_kwargs = dict(
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        if _is_batched_prefill(input_ids, inputs_embeds):
            # [B>1, S>1]: B captured single-sequence prefills, one KV slot each.
            # Every sequence returns hidden states (the fused lm_head is decode-
            # only), so the Python lm_head below runs once on the [B, S, H] cat.
            raw, pkv = _run_batched_prefill(
                self.model, h, _capture,
                past_key_values=past_key_values,
                input_ids=input_ids, inputs_embeds=inputs_embeds,
                prefill_plan=prefill_plan,
                **_runner_kwargs,
            )
        else:
            with _capture(_batch, _seq_len, prefill_plan):
                raw, pkv = _run_causal_decoder_forward(
                    self.model,
                    h,
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    prefill_plan=prefill_plan,
                    **_runner_kwargs,
                )

        # Decode-only fused lm_head:
        #   decode (seq_len=1) + fuse_lm_head enabled → C++ fuses lm_head GEMM
        #   in main graph last layer, returns [1, 1, vocab_size] logits directly.
        #   prefill (seq_len>1) → C++ returns [1, seq, hidden_size] hidden states,
        #   Python runs self.lm_head. logits_to_keep controls the slice:
        #     logits_to_keep=0 → full-sequence logits (HF default, e.g. teacher forcing)
        #     logits_to_keep=1 → last-token only (HF generate() fast path)
        hidden = self.config.hidden_size
        vocab = self.config.vocab_size
        if raw.size(-1) == vocab:
            # Decode fused: C++ returned [B, 1, vocab] logits via SPM lm_head GEMM
            logits = raw
        elif raw.size(-1) == hidden:
            # Prefill or non-fused decode: Python-side lm_head
            if self.lm_head.weight.dtype == torch.int8:
                if logits_to_keep == 0 and raw.size(1) != 1:
                    raise NotImplementedError(
                        "int8 fused_lm_head prefill only supports last-token "
                        "logits; pass logits_to_keep=1 for generate/decode. "
                        "Full-sequence int8 lm_head would require a DDR W8A16 "
                        "linear path, which is intentionally out of scope."
                    )
                logits = _int8_lm_head_prefill_logits(self, raw[:, -1:, :])
            elif logits_to_keep == 0:
                # Full-sequence logits (HF default semantics for logits_to_keep=0)
                logits = self.lm_head(raw)
            else:
                # Last-token only (HF generate() passes logits_to_keep=1)
                logits = self.lm_head(raw[:, -logits_to_keep:, :])
        else:
            raise RuntimeError(
                f"fused_forward: unexpected raw last-dim {raw.size(-1)}, "
                f"expected hidden_size={hidden} or vocab_size={vocab}"
            )

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=pkv,
            hidden_states=None,
            attentions=None,
        )

    import types
    causal_lm.forward = types.MethodType(fused_forward, causal_lm)

    _LOG.info("Patched Qwen3ForCausalLM (instance) with fused lm_head, "
              "handle=%d, vocab_size=%d",
              handle, causal_lm.config.vocab_size)

    return handle


def _run_fused_lm_head_decode_top1(
    causal_lm,
    *,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    **kwargs,
):
    """Run one fused-lm_head decode step and return greedy token ids.

    The fused decoder remains graph-captured. The top1 reduction runs after the
    graph scope through a tiny C++ zero-copy CPU scan over the graph-written DDR
    logits. It avoids the outer Python CausalLMOutput/logits/argmax path without
    claiming to be the final SPM-local top1 kernel.
    """
    from contextlib import nullcontext as _nullcontext
    import rpu_backend as _rb
    from rpu_backend.api.cache import RPUCache as _RPUCache_for_sig

    if not hasattr(causal_lm.model, "_rpu_decoder_handle"):
        raise RuntimeError(
            "_run_fused_lm_head_decode_top1 requires a model patched by "
            "_apply_fused_lm_head_for_rpu"
        )

    if (input_ids is None) == (inputs_embeds is None):
        raise AssertionError(
            "fused lm_head top1 decode requires exactly one of input_ids or "
            "inputs_embeds"
        )
    _src = inputs_embeds if inputs_embeds is not None else input_ids
    batch, seq_len = int(_src.shape[0]), int(_src.shape[1])
    if batch > 1 and not bool(
        getattr(causal_lm.model, "_rpu_batch_decode_enabled", False)
    ):
        raise AssertionError(
            "RPU batch > 1 requires the Qwen3 batch-decode capability"
        )
    if seq_len != 1:
        raise NotImplementedError(
            f"fused lm_head top1 helper is decode-only (seq_len=1), got {seq_len}"
        )

    h = causal_lm.model._rpu_decoder_handle
    if isinstance(past_key_values, _RPUCache_for_sig):
        attention_mask, position_ids, cache_position = (
            _canonicalize_plain_text_controls(
                seq_len,
                past_key_values,
                attention_mask,
                position_ids,
                cache_position,
                batch_size=batch,
            )
        )
        sig = _rb.graph.GraphSignature(
            op_id="rpu_causal_decoder_lm_head_top1",
            shapes=[batch, seq_len, causal_lm.model._rpu_decoder_hidden_size],
            dyn_dims=[
                causal_lm.model._rpu_decoder_num_layers,
                causal_lm.model._rpu_decoder_deepstack_hash,
                int(causal_lm.config.vocab_size),
                # Same decoder forward, same SPM-layout hazard.
                chunk_policy_key(causal_lm.model._rpu_decoder_handle),
                prefill_position_key(seq_len, past_key_values.position),
            ],
            dtypes=[torch.float16],
        )
        capture_ctx = causal_lm.model._rpu_decoder_graph_cache.capture(sig)
    else:
        # Preserve the runner's precise invalid-cache error. Transfers happen
        # outside capture and are relevant only to that error path.
        if attention_mask is not None and attention_mask.device.type != "rpu":
            attention_mask = attention_mask.to("rpu")
        if position_ids is not None and position_ids.device.type != "rpu":
            position_ids = position_ids.to("rpu")
        capture_ctx = _nullcontext()

    if input_ids is not None and input_ids.device.type != "rpu":
        input_ids = input_ids.to("rpu")

    with capture_ctx:
        raw, pkv = _run_causal_decoder_forward(
            causal_lm.model,
            h,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        vocab = causal_lm.config.vocab_size
        if raw.size(-1) != vocab:
            raise RuntimeError(
                "fused lm_head top1 helper expected C++ fused logits with "
                f"last-dim vocab_size={vocab}, got {raw.size(-1)}"
            )

    next_id = torch.ops.rpu.lm_head_logits_top1(raw)

    return next_id, pkv
