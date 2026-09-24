"""Qwen3 adapter: wraps the existing convert + all-layers-once patch sequence
behind a uniform interface. D-13 (v4.0): imported unchanged helpers from the
model_converter shim. v4.1 Phase 12 D-A1 / D-H2: migrated to canonical homes
(core.weights for weight primitives + _internal.patches for
patch_qwen3_model_for_rpu_all_layers_once). Zero edits to behavior.

v5-02 / B1: relocated from transformers/qwen3/adapter.py. The cross-surface
idempotent monkey-patcher in the legacy transformers/qwen3/__init__.py was
DROPPED per G4-C — the legacy harness `tests/model/test_qwen3_prefill.py` is
script-style (not pytest-collected) and already broken on v5-01b, so the
70-LOC idempotency belt is moot. The two raw class-patch calls
(`patch_rmsnorm_class(Qwen3RMSNorm)` + `patch_rotary_embedding(Qwen3RotaryEmbedding)`)
remain — see below.

This adapter implements decisions D-01 (two-step + .to('rpu') shortcut),
D-02 (thin wrapper, NOT subclass), D-03 (per-instance patches at .to),
D-04 (build_rpu_cache contract — caller passes input_ids; no runtime model attr),
D-12 (deny-by-default model-profile envelope), D-13 (additive only),
D-17 (single-handle).
"""
from __future__ import annotations
import gc
import threading

import torch

from rpu_backend.runtime.log import _LOG

# Phase 12 D-H2: was the model_converter shim multi-import; split per canonical home.
from rpu_backend.runtime.weights import (
    _collect_embedding_data_ptrs,
    convert_linear_weights_inplace,
    _validate_bounded_linear_move,
)
from rpu_backend.quant.convert_qwen3 import QUANT_PROJ_SUFFIXES
# Shared decoder ownership:
# `_install_causal_decoder_forward` lives in `runtime/decoder.py` (was
# `patch_causal_decoder_for_rpu` in `_internal/patches/__init__.py:393`,
# deleted in Commit 2 of this phase). The thin `patch_qwen3_model_for_rpu_all_layers_once`
# wrapper (D-03d) is folded into the call site here:
#     _install_causal_decoder_forward(model, arch="qwen3")
from rpu_backend.runtime.decoder import (
    _bind_causal_decoder_execution_session,
    _causal_lm_runtime_complete,
    _canonicalize_plain_text_controls,
    _cleanup_causal_decoder_install,
    _enable_causal_decoder_execution_reconfigure,
    _install_causal_decoder_forward,
    _reject_unconsumed_text_padding,
    _text_decode_execution_plan,
    _text_prefill_execution_plan,
    chunk_policy_key,
    prefill_position_key,
)

from rpu_backend.runtime.causal_append import continuation_segments, run_continuation_segments

from rpu_backend.api.errors import UnsupportedModelError, RPUBackendError
from rpu_backend.api._execution import (
    execution_serialized, _QWEN3_EXECUTION_SUPPORTED, validate_qwen3_core_profile,
    qwen3_core_profile,
    is_qwen3_17b_w8a16_core_config,
)
from rpu_backend.runtime.topology import (
    execution_core_count, resolve_decoder_topology, validate_decoder_cache_topology,
    decoder_topology_for_model, DECODER_GEOMETRY_PROFILES, decoder_mlp_intermediate_size,
)
from rpu_backend.quant.qwen3_profiles import (
    qwen3_dense_quant_profile, is_qwen3_32b_w8a16_config,
)
from rpu_backend.api.causal_lm import _claim_live_instance, _release_live_instance
from rpu_backend.runtime.device import extract_to_device_target, is_rpu_device_target

# v5-02 / B1: lifted from the deleted transformers/qwen3/__init__.py — the two
# raw class-patch invocations (was wrapped in a 70-LOC idempotent helper per
# G4-C; helper dropped because the legacy harness is no longer pytest-collected).
# v5-04 V2D-06: V2D-06 helpers relocated to runtime/decoder.py; renamed to
# _install_*_class_swap to make the side-effecting "install once" semantics
# explicit at the call site.
from rpu_backend.runtime.decoder import (
    _install_rmsnorm_class_swap,
    _install_rotary_class_swap,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding
from rpu_backend.runtime.chunk_envelope import ChunkEnvelope, make_lookup

# (arch, num_hidden_layers, hidden_size) -> (max_kv_len, safe chunk ceiling).
# The length bounds position + physical execution length for multi-token
# prefill; single-token decode retains its separate cache/RoPE bounds. These
# geometry-based limits are shared by FP16 and W8A16. Keep chunk ceilings fixed
# as context grows: the native planner still checks padding, tails and SPM.
# A geometry without a row cannot prefill.
_CHUNK_ENVELOPE = {
    ("qwen3", 28, 1024): ChunkEnvelope(4096, 512),   # 0.6b
    ("qwen3", 28, 2048): ChunkEnvelope(4096, 512),   # 1.7b
    ("qwen3", 36, 2560): ChunkEnvelope(4096, 256),   # 4b
    # 8b — 128 is the largest currently selectable/certified chunk. Historical
    # cs=224 measurements predate the current SDPA validity predicate; 224 is
    # no longer a legal exact request and therefore must not be advertised as
    # the public ceiling. The large certificate covers 64/128 through P4096.
    ("qwen3", 36, 4096): ChunkEnvelope(4096, 128),   # 8b    fp16 + w8a16
    ("qwen3", 40, 5120): ChunkEnvelope(192, 32),     # 14b   w8a16 only
}

# Exact values whose attention geometry is admissible at each profile's
# certificate input. The shared runtime still validates the concrete logical
# length, padding and cache position before executing: admissibility can change
# with the tail/position. Keeping this small table prevents a multi-GB load for
# values such as 8B/cs=224 or cs=48 which are 16-aligned and below a historical
# memory cap but cannot be launched by the current SDPA geometry.
_EXACT_CHUNKS = {
    ("qwen3", 36, 4096): frozenset((16, 32, 64, 128)),
    ("qwen3", 40, 5120): frozenset((16, 32)),
}
lookup_causal_decoder = make_lookup(
    _CHUNK_ENVELOPE, "rpu_backend/adapters/qwen3.py::_CHUNK_ENVELOPE")
# Quant-only candidate bounds; source admission stays exact and native planning
# remains authoritative for each request. This adds no FP16 geometry profile.
_QUANT_CHUNK_ENVELOPE = {
    **_CHUNK_ENVELOPE,
    ("qwen3", 64, 5120): ChunkEnvelope(4096, 64),
}
lookup_quantized_causal_decoder = make_lookup(
    _QUANT_CHUNK_ENVELOPE, "rpu_backend/adapters/qwen3.py::_QUANT_CHUNK_ENVELOPE")

# v5-11 NS-03b (Step-4): wrap the direct calls in idempotent guards (matches
# the absorbed _internal/patches/llama.py:34-45 pattern; codex sidebar #2 + R1 M-2
# class-patch idempotency invariant). Sentinel attrs `_rpu_patched_*` make
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


# Round-5 hardening: module-level lock serializes swizzle across ALL
# Qwen3 models in the process. Two racing threads calling .to('rpu')
# on the same (or different) Qwen3 instances can no longer both enter
# the irreversible convert_linear_weights_inplace path — the second
# thread either blocks briefly and short-circuits on the model's
# _rpu_swizzled flag when the first finishes, or (for same-model
# re-entry within one thread) hits the fast-path check above the lock.
#
# Note: D-17 single-handle restricts the process to one LIVE RPU model
# at a time, so cross-model serialization adds near-zero contention in
# practice (users never run two concurrent .to('rpu') intentionally).
_SWIZZLE_LOCK = threading.Lock()


# D-12 + MEDIUM-10 (codex iter1): Qwen3 supported envelope.
# Multi-dimensional check: hidden_size alone is too weak (a future Qwen3 release
# could ship hidden=4096 with much larger intermediate_size or num_hidden_layers
# and silently pass the guard). Use a profile registry keyed on
# (hidden_size, intermediate_size, num_hidden_layers, num_key_value_heads).
# Values come from each model's config.json (model_cache/qwen3-*/config.json).
#
_SUPPORTED_PROFILES: frozenset[tuple[int, int, int, int]] = frozenset(
    (profile.hidden_size, profile.intermediate_size, profile.num_layers,
     profile.num_kv_heads) for profile in DECODER_GEOMETRY_PROFILES
)

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


def _execution_topology(config, execution_config):
    try:
        cores = execution_core_count(execution_config)
        if is_qwen3_32b_w8a16_config(config):
            if cores != 8:
                raise ValueError("Qwen3-32B W8A16 requires model.num_cores=8")
        elif not validate_qwen3_core_profile(config, cores):
            return None
        return resolve_decoder_topology(
            num_cores=cores, hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_q_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim, vocab_size=config.vocab_size)
    except ValueError as exc:
        raise UnsupportedModelError(str(exc)) from exc


def _validate_reduced_model_structure(model, topology):
    if topology is None:
        return
    from rpu_backend.runtime.weights import validate_decoder_weight_structure
    try:
        profile = qwen3_core_profile(model.config, topology.num_cores)
        # Exact 8B FP16 also needs whole-tree admission at budget8 before its
        # bounded CPU-to-RPU migration. Other default8 paths stay unchanged.
        if (topology.num_cores == 8 and model.config.hidden_size == 4096
                and profile is None):
            raise ValueError("bounded 8B weight migration requires its exact plain FP16 profile")
        if topology.num_cores == 8 and (profile is None or profile.hidden_size != 4096):
            return
        if resolve_decoder_topology(num_cores=topology.num_cores,
                                    **profile.geometry()) != topology:
            raise ValueError("decoder geometry changed its bound cold topology")
        geometry = profile.weight_geometry()
        if getattr(model, "_rpu_swizzled", False):
            geometry["intermediate_size"] = decoder_mlp_intermediate_size(
                profile.intermediate_size, topology.mlp_tp)
        w8 = is_qwen3_17b_w8a16_core_config(model.config)
        if w8:
            validate_decoder_weight_structure(model, **geometry,
                projection_dtype=torch.int8, lm_head_dtype=torch.int8)
            expected_device = "rpu" if getattr(model, "_rpu_swizzled", False) else "cpu"
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    scale = getattr(module, "weight_scale", None)
                    if (not isinstance(scale, torch.Tensor) or scale.dtype != torch.float16
                            or scale.device.type != expected_device or not scale.is_contiguous()
                            or tuple(scale.shape) != (module.out_features,)):
                        raise ValueError(f"decoder W8A16 requires its FP16 per-channel scale: {name}")
                elif getattr(module, "weight_scale", None) is not None:
                    raise ValueError(f"decoder W8A16 does not quantize this module: {name}")
                if any(getattr(module, attr, None) is not None for attr in (
                        "weight_scale_inv", "qweight")):
                    raise ValueError(f"decoder W8A16 rejects alternate quantized storage: {name}")
                if not getattr(model, "_rpu_swizzled", False) and hasattr(module, "_rpu_linear_partition"):
                    raise ValueError(f"decoder W8A16 requires original unswizzled weights: {name}")
            for name, tensor in (*model.named_parameters(), *model.named_buffers()):
                if tensor.device.type != expected_device or not tensor.is_contiguous():
                    raise ValueError(f"decoder W8A16 requires contiguous {expected_device} tensors: {name}")
        else:
            validate_decoder_weight_structure(model, **geometry)
    except ValueError as exc:
        raise UnsupportedModelError(str(exc)) from exc


def _validate_w4a16_source(model, execution_config, quantization):
    """Admit the original FP16 tree before the bounded on-install G32 recipe."""
    from rpu_backend.runtime.weights import validate_decoder_weight_structure

    profile = Qwen3Adapter.preflight_quantization(
        model.config, execution_config, quantization)
    try:
        validate_decoder_weight_structure(model, **profile.weight_geometry())
        for name, tensor in model.named_parameters():
            if (tensor.dtype != torch.float16 or tensor.device.type != "cpu"
                    or not tensor.is_contiguous()):
                raise ValueError(f"W4A16 requires original contiguous CPU FP16 weights: {name}")
        for name, module in model.named_modules():
            if (hasattr(module, "_rpu_linear_partition") or any(
                    getattr(module, attr, None) is not None for attr in (
                        "weight_scale", "weight_scale_inv", "qweight"))):
                raise ValueError(f"W4A16 requires original unquantized, unswizzled weights: {name}")
    except ValueError as exc:
        raise UnsupportedModelError(str(exc)) from exc


def _check_profile(config) -> None:
    """Raise `UnsupportedModelError` if the config's 4-tuple profile is not in
    the supported fp16 or W8A16-only envelope. Used by both
    `Qwen3Adapter.preflight(config)` (the iter7 fail-fast preflight before HF
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
    supported = profile in _SUPPORTED_PROFILES or (
        _config_is_w8a16(config) and profile in _W8A16_ONLY_PROFILES
    ) or is_qwen3_32b_w8a16_config(config)
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
            f"W8A16-only profiles: {sorted(_W8A16_ONLY_PROFILES)}. "
            "The exact 32B symmetric W8A16 decoder/head checkpoint is also admitted. "
            "See docs/api_reference.md §Qwen3 size matrix for deferral rationale. "
            "Track progress: v3.x backlog."
        )


def _detect_and_validate_w8a16(model) -> bool:
    """Return True for local W8A16 Qwen3 models and fail on partial loads."""
    # An INT8 output head needs the fused route even if decoder projections
    # remain FP16. Validate it before any irreversible adapter work.
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


def _stage_large_qwen3_weights_for_rpu(model) -> None:
    """Convert/transfer one decoder layer at a time before the remaining tree."""
    inner = model.model
    embedding_ptrs = _collect_embedding_data_ptrs(model)
    for index, layer in enumerate(inner.layers):
        convert_linear_weights_inplace(
            layer, prefix=f"model.layers.{index}",
            _embedding_ptrs=embedding_ptrs, skip_names=set(),
        )
        layer.to("rpu")
        gc.collect()

    _convert_qwen3_remaining_weights(model, embedding_ptrs)


def _convert_qwen3_remaining_weights(model, embedding_ptrs=None):
    inner = model.model
    if embedding_ptrs is None:
        embedding_ptrs = _collect_embedding_data_ptrs(model)
    # Walk the rest through the same converter, including tied lm_head and
    # any other Linear children, without revisiting already-staged layers.
    # These temporary parent modules share the original children; the actual
    # model registration and parameter ownership remain intact throughout.
    remaining = torch.nn.Module()
    for name, child in model.named_children():
        if child is inner:
            remaining_inner = torch.nn.Module()
            for inner_name, inner_child in inner.named_children():
                if inner_child is not inner.layers:
                    remaining_inner.add_module(inner_name, inner_child)
            remaining.add_module(name, remaining_inner)
        else:
            remaining.add_module(name, child)
    convert_linear_weights_inplace(
        remaining, _embedding_ptrs=embedding_ptrs, skip_names=set(),
    )


class Qwen3Adapter:
    """Per-instance adapter for HF `Qwen3ForCausalLM`."""

    EXECUTION_SUPPORTED = _QWEN3_EXECUTION_SUPPORTED

    @classmethod
    def preflight_quantization(cls, config, execution_config, quantization):
        """Admit an exact dense FP16 source for the two TP8/G32 recipes."""
        try:
            if quantization not in ("w4a16", "w4a16_lm_head"):
                raise ValueError("unknown on-install W4 recipe")
            profile = qwen3_dense_quant_profile(config)
            if execution_core_count(execution_config) != 8:
                raise ValueError("W4 requires model.num_cores=8")
        except ValueError as exc:
            raise UnsupportedModelError(
                "on-install W4 requires exact dense Qwen3 FP16 source weights, "
                "G32 and model.num_cores=8; embedding and norms remain FP16: " + str(exc)
            ) from exc
        return profile

    @classmethod
    def preflight(cls, config) -> None:
        """iter7 HIGH (codex): config-only D-12 fail-fast preflight.

        Called by `RPUModelForCausalLM.from_pretrained` BEFORE the full HF
        model load so unsupported dtype/profile combinations raise within
        milliseconds (one JSON parse) rather than minutes (full weight load).
        """
        _check_profile(config)

    @classmethod
    def preflight_execution(cls, config, execution_config) -> None:
        _execution_topology(config, execution_config)
        requested = execution_config.get("prefill", {}).get("chunk_size", "auto")
        if not isinstance(requested, int):
            return
        key = ("qwen3", int(config.num_hidden_layers), int(config.hidden_size))
        env = lookup_quantized_causal_decoder(*key)
        if env.chunk > 0 and requested > env.chunk:
            raise UnsupportedModelError(
                f"Qwen3 prefill chunk_size={requested} exceeds this profile's "
                f"certified ceiling {env.chunk}; max certified KV length is "
                f"{env.max_kv_len}. Refusing before loading model weights."
            )
        admissible = (frozenset((16, 32, 64)) if key == ("qwen3", 64, 5120)
                      else _EXACT_CHUNKS.get(key))
        if admissible is not None and requested not in admissible:
            choices = ", ".join(str(value) for value in sorted(admissible))
            raise UnsupportedModelError(
                f"Qwen3 prefill chunk_size={requested} is 16-aligned and within "
                f"the memory ceiling, but is not admissible for this profile's "
                f"current SDPA geometry. Choose one of {{{choices}}} or 'auto'. "
                "The concrete input length and cache position are validated "
                "again by the exact planner. Refusing before loading model weights."
            )

    def __init__(self, model, *, quantization=None):
        # D-12 + MEDIUM-10 belt-and-suspenders: validate envelope BEFORE any
        # mutation (so an unsupported profile raises before swizzle), even on direct adapter
        # instantiation paths that bypass RPUModelForCausalLM.from_pretrained.
        if quantization is None:
            _check_profile(model.config)
        if (getattr(model.to, "__rpu_wrapped__", False)
                and getattr(model.to, "__rpu_quantization__", None) != quantization):
            raise UnsupportedModelError("Qwen3 on-install quantization is cold-only; reload the model to change it")
        self._quantization = quantization
        ready = _causal_lm_runtime_complete(model)
        quant_profile = None
        self._staged_w4 = False
        if quantization is not None:
            quant_profile = self.preflight_quantization(
                model.config, getattr(model, "_rpu_execution", None), quantization)
            self._staged_w4 = quant_profile.num_layers == 64
            if not ready:
                if self._staged_w4:
                    from rpu_backend.quant.load_qwen3_quantized import validate_staged_qwen3_w4a16
                    validate_staged_qwen3_w4a16(model, quantization=quantization)
                else:
                    _validate_w4a16_source(model, getattr(model, "_rpu_execution", None), quantization)
        elif any(parameter.dtype == torch.uint8 for parameter in model.parameters()):
            raise UnsupportedModelError(
                "Packed UINT8 Qwen3 weights require the explicit on-install W4 "
                "recipe from original FP16 weights"
            )
        self.model = model
        self._all_layers_once_handle: int | None = None
        self._is_w8a16 = _detect_and_validate_w8a16(model)
        self._large_quant = self._staged_w4 or is_qwen3_32b_w8a16_config(model.config)
        self._topology = (resolve_decoder_topology(num_cores=8, **quant_profile.geometry())
            if quant_profile is not None else _execution_topology(
                config=model.config, execution_config=getattr(model, "_rpu_execution", None)))
        self._bound_profile = _config_profile(model.config)
        if self._topology is not None:
            if (quantization is None and not self._large_quant
                    and not is_qwen3_17b_w8a16_core_config(model.config)
                    and (self._is_w8a16 or any(parameter.dtype != torch.float16
                    for parameter in model.parameters()))):
                raise UnsupportedModelError("fixed-core Qwen3 requires plain FP16 model weights")
        if quantization is None and not self._large_quant:
            _validate_reduced_model_structure(model, self._topology)
        if quantization is None and _config_profile(model.config) in _W8A16_ONLY_PROFILES:
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
        # Round-2 P1 hardening: readiness MUST live on the model, not the
        # adapter. Otherwise a second `Qwen3Adapter(model)` construction on
        # an already-swizzled model creates a fresh adapter with
        # `_rpu_is_ready=False`; `to_rpu()` then calls
        # `convert_linear_weights_inplace` on already-swizzled weights →
        # double-swizzle (Pitfall 1). Adopt the existing per-model flag
        # if present; otherwise start False.
        self._rpu_is_ready: bool = _causal_lm_runtime_complete(model)
        self._execution_session = _bind_causal_decoder_execution_session(
            self, entry_point="Qwen3Adapter"
        )

        # Per D-01 two-step canonical: install a `.to('rpu')` interceptor on the
        # model instance so user code `model.to('rpu')` triggers `to_rpu()`.
        # This monkey-patches the bound method on this single instance only — does
        # NOT mutate the HF class, does NOT affect other model instances.
        #
        # WR-04 (codex iter9): detect an already-wrapped `model.to` (from a
        # prior `Qwen3Adapter(model)` on the same instance) via a sentinel
        # attribute on the wrapper function. Without this guard, repeated
        # construction stacks interceptors: the second __init__ captures the
        # first's `rpu_aware_to` as `original_to`, so `model.to('cpu')` would
        # chain through two interceptors. The single-handle gate (D-17) only
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
            # iter2 HIGH-3 + iter3 MEDIUM (codex): once the adapter has swizzled
            # weights for RPU, `convert_linear_weights_inplace` is IRREVERSIBLE —
            # moving the model back to CPU and then to RPU again would NOT
            # re-swizzle (no-op via _rpu_is_ready) and the CPU forward path
            # would crash on swizzled weights. Reject any non-rpu target after
            # to_rpu() so the failure surfaces at the .to() call site.
            # iter3: raise the typed `RPUBackendError` (D-18 base) so callers can
            # catch with `except RPUBackendError` rather than the broader
            # `RuntimeError`.
            # Round-3 hardening: this closure captures adapter1's
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

        # WR-04 sentinel: tag the wrapper so a subsequent `Qwen3Adapter(model)`
        # on the same instance detects the existing wrap via
        # `getattr(model.to, "__rpu_wrapped__", False)` and short-circuits.
        rpu_aware_to.__rpu_wrapped__ = True
        rpu_aware_to.__rpu_quantization__ = quantization

        # Bind onto the instance (NOT the class) — survives only as long as `model` lives.
        # `nn.Module.__setattr__` falls through to `super().__setattr__` for plain
        # functions (verified — module.py:2072), so this assignment is safe.
        model.to = rpu_aware_to

    def to_rpu(self):
        """Step A → B → C: convert weights, all-layers-once patch, claim live-handle, return RPU-ready model.

        HIGH-3 (codex iter1): IDEMPOTENT for the same model instance. If
        `to_rpu()` was already called on this adapter, returns `self.model`
        without re-running weight swizzle (which would corrupt already-swizzled
        weights) or re-claiming the live-handle gate (which would raise against
        ourself).

        Pitfall 1 (P1 — CLAUDE.md §"Critical: Known Pitfalls"): SKIP_LINEAR_NAMES.
        Plain/W8 Qwen3 has no specialized `.dense` layers, so it uses an empty
        skip set. The on-install W4 recipe packs its seven decoder projections
        once and excludes them from the subsequent generic FP16 head swizzle.

        The default `skip_names=None` in `convert_linear_weights_inplace` (model_converter.py:195)
        merges `{'dense'}` into the skip set as a P1 safety belt. For Qwen3 with no `.dense`
        leaves, this would be harmless either way, but explicit `set()` documents intent and
        keeps adapter-level discipline visible alongside Phase 3 Pi05 (`skip_names={'dense'}`)
        and Phase 4 QwenPI05 (`skip_names={'cond', 'dense'}`).
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

        # Round-5 HIGH #2: fail loud if a prior swizzle attempt started mutation
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

        # Round-5 HIGH #1: module-level lock serializes the entire swizzle
        # critical section across all Qwen3 models in the process. Round-4's
        # check-then-set on `_rpu_swizzle_in_progress` had a race window
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
            if getattr(self.model, "_rpu_execution", None) != self._execution_session.config:
                raise RPUBackendError("Qwen3 execution configuration changed outside its bound session")

            if self._topology is not None and _config_profile(self.model.config) != self._bound_profile:
                raise UnsupportedModelError("decoder geometry changed its bound cold topology")
            if self._quantization is not None:
                if self._staged_w4:
                    from rpu_backend.quant.load_qwen3_quantized import validate_staged_qwen3_w4a16
                    validate_staged_qwen3_w4a16(self.model, quantization=self._quantization)
                else:
                    _validate_w4a16_source(self.model, self._execution_session.config, self._quantization)
            elif not self._large_quant:
                _validate_reduced_model_structure(self.model, self._topology)
            if self._topology is not None:
                from rpu_backend.runtime.weights import validate_decoder_mlp_padding
                logical = self.model.config.intermediate_size
                validate_decoder_mlp_padding(self.model, logical_size=logical,
                    physical_size=decoder_mlp_intermediate_size(logical, self._topology.mlp_tp))
            bounded_move = (self._topology is not None and not self._is_w8a16
                            and self._quantization is None
                            and self.model.config.hidden_size == 4096)
            if bounded_move:
                _validate_bounded_linear_move(
                    self.model, False, self._topology.attn_tp,
                    self._topology.mlp_tp, self._topology.lm_head_tp,
                    {"dense"})

            # D-17: claim single-handle gate BEFORE mutation so a second concurrent
            # .to('rpu') on a DIFFERENT model raises before we touch any weights.
            # _claim_live_instance is idempotent for the same model id (HIGH-3 fix
            # in 02-01); only a different live model triggers RPUSingleHandleError.
            _claim_live_instance(self.model)
            try:
                # Reuse HostDDR mappings for request temporaries before the
                # first RPU allocation freezes the process allocator policy.
                # A rejected second model must not change the live owner's policy.
                torch.rpu.set_caching_allocator(True)
                if self._large_quant:
                    from rpu_backend.graph import GraphRuntimePolicy
                    arena_policy = GraphRuntimePolicy.from_environment(graph_arena_count=9)
                    if not arena_policy.prepare_arenas():
                        raise RPUBackendError("Qwen3-32B quantization requires cold SDK graph arenas")
                    self._graph_arena_policy = arena_policy
            except BaseException:
                _release_live_instance(self.model)
                raise

            # Stamp started BEFORE any mutation. If anything below raises,
            # this flag stays set and the HIGH-#2 check at function-entry
            # above raises on retry telling the caller to reload.
            self.model._rpu_swizzle_started = True
            if self._topology is not None:
                self.model._rpu_decoder_topology = self._topology
            inner = self.model.model
            if self._large_quant:
                self.model._rpu_kv_cache_layer_bank_size = 8
                inner._rpu_kv_cache_layer_bank_size = 8
            inner_state = vars(inner)
            had_instance_forward = "forward" in inner_state
            original_instance_forward = inner_state.get("forward")
            outer_state = vars(self.model)
            had_outer_forward = "forward" in outer_state
            original_outer_forward = outer_state.get("forward")
            try:
                # Step A: weight conversion and device migration are
                # irreversible on this instance.
                _stash_int8_lm_head_cpu_reference(self.model)
                _patch_int8_embedding_forward(self.model)
                if self._staged_w4:
                    from rpu_backend.quant.load_qwen3_quantized import materialize_staged_qwen3_w4a16_for_rpu
                    materialize_staged_qwen3_w4a16_for_rpu(self.model, quantization=self._quantization)
                elif self._quantization is not None:
                    from rpu_backend.quant.convert_qwen3_w4a16 import (
                        DEFAULT_GROUP_SIZE, SKIP_LINEAR_NAMES, _PROJECTIONS,
                        convert_qwen3_to_w4a16_,
                    )
                    quantize_head = self._quantization == "w4a16_lm_head"
                    convert_qwen3_to_w4a16_(self.model, group_size=DEFAULT_GROUP_SIZE,
                                          quantize_lm_head=quantize_head)
                    skip = set(SKIP_LINEAR_NAMES) | ({"lm_head"} if quantize_head else set())
                    convert_linear_weights_inplace(self.model, skip_names=skip)
                    if quantize_head:
                        self.model.lm_head._rpu_linear_partition = 1
                        self.model.lm_head._rpu_linear_num_cores = 8
                    # The generic skip marker describes only that converter.
                    # These projections already carry the packer's TP8 layout.
                    for layer in inner.layers:
                        for role, partition in _PROJECTIONS:
                            projection = layer.get_submodule(role)
                            projection._rpu_linear_partition = partition
                            projection._rpu_linear_num_cores = 8
                elif self._large_quant:
                    from rpu_backend.quant.load_qwen3_quantized import move_qwen3_w8a16_decoder_for_rpu
                    move_qwen3_w8a16_decoder_for_rpu(self.model)
                    _convert_qwen3_remaining_weights(self.model)
                elif self._topology is not None:
                    from rpu_backend.runtime.weights import pad_decoder_mlp_weights
                    logical = self.model.config.intermediate_size
                    pad_decoder_mlp_weights(self.model, logical_size=logical,
                        physical_size=decoder_mlp_intermediate_size(logical, self._topology.mlp_tp))
                    convert_linear_weights_inplace(
                        self.model, skip_names=set(),
                        **({"move_to_device": "rpu"} if bounded_move else {}),
                        attn_num_cores=self._topology.attn_tp,
                        mlp_num_cores=self._topology.mlp_tp,
                        lm_head_num_cores=self._topology.lm_head_tp,
                        execution_core_count=self._topology.num_cores,
                    )
                elif int(self.model.config.hidden_size) >= 4096:
                    # A full converted 8B/14B CPU decoder plus its RPU copy
                    # can exceed the staging room of a 29-GiB host. Release
                    # each converted CPU layer before converting the next.
                    _stage_large_qwen3_weights_for_rpu(self.model)
                else:
                    convert_linear_weights_inplace(
                        self.model, skip_names=set()
                    )
                original_to = type(self.model).to
                original_to(self.model, "rpu")

                scale_lists = None
                if self._is_w8a16 or self._quantization is not None:
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
                    chunk_envelope_for=(lookup_quantized_causal_decoder
                        if self._quantization is not None or self._large_quant
                        else lookup_causal_decoder),
                    **({"_graph_cache_max_entries": 8} if self._large_quant else {}),
                    execution_config=getattr(
                        self.model, "_rpu_execution", None
                    ),
                    **({"topology": self._topology}
                       if self._topology is not None else {}),
                )
                lm_head = getattr(self.model, "lm_head", None)
                if isinstance(lm_head, torch.nn.Linear) and lm_head.weight.dtype in (torch.int8, torch.uint8):
                    _apply_fused_lm_head_for_rpu(self.model)
                else:
                    _bind_fp16_lm_head_accumulation(self.model)
                    _apply_plain_lm_head_causal_append_for_rpu(self.model)
                _enable_causal_decoder_execution_reconfigure(inner)
                self._execution_session = _bind_causal_decoder_execution_session(
                    self, entry_point="Qwen3Adapter"
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
                # A failed retirement keeps the installed forward and its
                # weights under the poisoned session. Restore the original
                # outer entry only after native ownership is gone.
                if getattr(inner, "_rpu_decoder_handle", None) is None:
                    if had_outer_forward:
                        outer_state["forward"] = original_outer_forward
                    else:
                        outer_state.pop("forward", None)
                    outer_state.pop("_rpu_lm_head_w_keepalive", None)
                    outer_state.pop("_rpu_lm_head_scale_keepalive", None)
                raise

            return self.model
        finally:
            # Release the lock regardless of outcome. `_rpu_swizzle_started`
            # is NOT cleared here — on success it stays True alongside
            # _rpu_swizzled; on failure it stays True WITHOUT
            # _rpu_swizzled, tripping the HIGH-#2 check on retry.
            _SWIZZLE_LOCK.release()


from rpu_backend.runtime.registry import register_adapter
register_adapter("Qwen3ForCausalLM", Qwen3Adapter)


# =====================================================================
# Phase 04 / Q1 lift (ADR §6.4 + Ledger Q1): CausalLM-specific helpers
# for the fused-lm-head path; lifted byte-equal from
# _internal/patches/__init__.py (deleted in v5-11). HND-01 rename: the
# legacy per-arch handle attribute is replaced by `_rpu_decoder_handle`
# in the lifted body.
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
    """Quantized DDR head using the same cold accumulation policy as decode."""
    topology = decoder_topology_for_model(causal_lm)
    cores = topology.lm_head_tp if topology is not None else 8
    acc32 = bool(causal_lm._rpu_execution.get("prefill", {}).get("linear_acc32", False))
    output = torch.ops.rpu.linear_with_accumulation(
        hidden.reshape(-1, hidden.shape[-1]).contiguous(),
        causal_lm._rpu_lm_head_w_keepalive, None, 1, cores, acc32,
        causal_lm._rpu_lm_head_scale_keepalive)
    return output.reshape(*hidden.shape[:-1], output.shape[-1])


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
    if getattr(first_q.weight, "dtype", None) not in (torch.int8, torch.uint8):
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

    This mirrors legacy setup_fuse_lm_head() (model_converter.py:1421-1430).
    """
    topology = decoder_topology_for_model(causal_lm)
    lm_head_tp = topology.lm_head_tp if topology is not None else 8
    lm_w = causal_lm.lm_head.weight.detach()
    if topology is not None and topology.num_cores != 8:
        try:
            profile = qwen3_core_profile(causal_lm.config, topology.num_cores)
            if resolve_decoder_topology(num_cores=topology.num_cores,
                                        **profile.geometry()) != topology:
                raise ValueError("fused lm_head geometry changed its bound cold topology")
        except ValueError as exc:
            raise RPUBackendError(str(exc)) from exc
        shape = (profile.vocab_size, profile.hidden_size)
        dtype = torch.int8 if is_qwen3_17b_w8a16_core_config(causal_lm.config) else torch.float16
        if lm_w.dtype != dtype or tuple(lm_w.shape) != shape:
            label = "INT8" if dtype == torch.int8 else "plain FP16"
            raise RPUBackendError(f"reduced-core fused lm_head requires {label} {shape} weights")
    if topology is not None and lm_w.device.type == "rpu":
        if (getattr(causal_lm.lm_head, "_rpu_linear_num_cores", None) != lm_head_tp
                or getattr(causal_lm.lm_head, "_rpu_linear_partition", None) != 1):
            raise RPUBackendError("fused lm_head weight layout differs from the cold model topology")
    if lm_w.dtype == torch.uint8:
        if getattr(causal_lm.to, "__rpu_quantization__", None) != "w4a16_lm_head":
            raise RPUBackendError("packed W4 lm_head requires its cold w4a16_lm_head recipe")
        profile = Qwen3Adapter.preflight_quantization(
            causal_lm.config, causal_lm._rpu_execution, "w4a16_lm_head")
        scale = getattr(causal_lm.lm_head, "weight_scale", None)
        scale_elems = ((profile.hidden_size // 32 + 3) // 4
                       * ((profile.vocab_size // 8 + 63) // 64) * 8 * 4 * 64)
        if (tuple(lm_w.shape) != (profile.vocab_size, profile.hidden_size // 2)
                or not lm_w.is_contiguous()
                or getattr(causal_lm.lm_head, "_rpu_linear_partition", None) != 1
                or getattr(causal_lm.lm_head, "_rpu_linear_num_cores", None) != 8
                or not isinstance(scale, torch.Tensor) or scale.dtype != torch.float16
                or tuple(scale.shape) != (32, scale_elems // 32)
                or not scale.is_contiguous()):
            raise RPUBackendError("packed W4 lm_head requires its TP8/G32 weight and scale layout")
        # Both payloads are already packed. A dtype cast or generic swizzle
        # would reinterpret bytes and corrupt the independent FP16 embedding.
        lm_w = lm_w.to("rpu")
        lm_scale = scale.detach().to("rpu")
        causal_lm._rpu_lm_head_w_keepalive = lm_w
        causal_lm._rpu_lm_head_scale_keepalive = lm_scale
        resource = getattr(causal_lm.model, "_rpu_decoder_retirement_state", None)
        if resource is not None:
            resource.keepalive = (resource.keepalive, lm_w, lm_scale)
        torch.ops.rpu.causal_decoder_set_lm_head(handle, lm_w, lm_scale)
        return
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
            lm_w = transform_linear_weight(lm_w.contiguous(), partition=1,
                **({"num_cores": lm_head_tp} if topology is not None else {})).to("rpu")
            lm_scale = lm_scale.to(torch.float16).contiguous().to("rpu")
        causal_lm._rpu_lm_head_w_keepalive = lm_w
        causal_lm._rpu_lm_head_scale_keepalive = lm_scale
        resource = getattr(causal_lm.model, "_rpu_decoder_retirement_state", None)
        if resource is not None:
            resource.keepalive = (resource.keepalive, lm_w, lm_scale)
        torch.ops.rpu.causal_decoder_set_lm_head(handle, lm_w, lm_scale)
        return

    if lm_w.device.type == "rpu":
        # Already swizzled + on RPU (common path: convert_linear_weights_inplace
        # was called on the full CausalLM before .to("rpu")).
        lm_w = lm_w.to(torch.float16).contiguous()
    else:
        # Still on CPU — need explicit col-partition swizzle before moving.
        lm_w = transform_linear_weight(
            lm_w.to(torch.float16).contiguous(), partition=1,
            **({"num_cores": lm_head_tp} if topology is not None else {}),
        ).to("rpu")
    causal_lm._rpu_lm_head_w_keepalive = lm_w
    resource = getattr(causal_lm.model, "_rpu_decoder_retirement_state", None)
    if resource is not None:
        resource.keepalive = (resource.keepalive, lm_w)
    torch.ops.rpu.causal_decoder_set_lm_head(handle, lm_w)


def _bind_fp16_lm_head_accumulation(causal_lm) -> None:
    """Keep the eager HF head while binding the model's cold Linear policy."""
    from types import MethodType
    from rpu_backend.runtime.weights import _check_linear_shapes

    head = causal_lm.lm_head
    partition = getattr(head, "_rpu_linear_partition", None)
    num_cores = getattr(head, "_rpu_linear_num_cores", None)
    if partition not in (0, 1) or num_cores not in (4, 6, 8):
        raise RPUBackendError("Qwen3 lm_head requires its recorded swizzle layout")
    acc32 = bool(causal_lm._rpu_execution.get("prefill", {}).get("linear_acc32", False))
    original_forward = head.forward

    def forward(self, x):
        if x.device.type != "rpu":
            return original_forward(x)
        if (getattr(self, "_rpu_linear_partition", None),
                getattr(self, "_rpu_linear_num_cores", None)) != (partition, num_cores):
            raise RuntimeError("Qwen3 lm_head swizzle layout changed after installation")
        _check_linear_shapes(x, self.weight, self.bias)
        y = torch.ops.rpu.linear_with_accumulation(
            x.reshape(-1, x.shape[-1]).contiguous(), self.weight, self.bias,
            partition, num_cores, acc32)
        return y.reshape(*x.shape[:-1], y.shape[-1])

    head.forward = MethodType(forward, head)


def _apply_plain_lm_head_causal_append_for_rpu(causal_lm) -> None:
    """Add segmented continuation to the ordinary FP16 HF outer forward.

    The shared decoder owns native execution and returns hidden states; the
    original HF ``Qwen3ForCausalLM.forward`` must remain responsible for the
    FP16 lm_head and output object.  Only the unpadded, auto-chunk continuation
    route is intercepted.  Every other request delegates byte-for-byte through
    the pre-install outer method.
    """
    from types import MethodType, SimpleNamespace
    from rpu_backend.api.cache import RPUCache

    original_forward = causal_lm.forward

    def segmented_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ):
        source = inputs_embeds if inputs_embeds is not None else input_ids
        if (
            isinstance(past_key_values, RPUCache)
            and isinstance(source, torch.Tensor)
            and source.ndim >= 2
            and int(source.shape[0]) == 1
            and int(source.shape[1]) > 1
        ):
            seq_len = int(source.shape[1])
            spans = continuation_segments(
                past_key_values.position,
                seq_len,
                getattr(self.model, "_rpu_execution", {}).get("prefill", {}),
                capacity=past_key_values.max_seq_len,
            )
            if spans is not None:
                if (input_ids is None) == (inputs_embeds is None):
                    raise AssertionError(
                        "RPU continuation requires exactly one of input_ids or inputs_embeds"
                    )
                if labels is not None:
                    raise NotImplementedError(
                        "RPU segmented continuation does not support labels/loss"
                    )
                if use_cache is False or kwargs.get("return_dict") is False:
                    raise NotImplementedError(
                        "RPU continuation requires use_cache=True and return_dict=True"
                    )
                attention_mask, position_ids, cache_position = (
                    _canonicalize_plain_text_controls(
                        seq_len,
                        past_key_values,
                        attention_mask,
                        position_ids,
                        cache_position,
                        batch_size=1,
                    )
                )
                handle = self.model._rpu_decoder_handle
                for offset, count in spans:
                    if count > 1:
                        prospective = SimpleNamespace(
                            position=int(past_key_values.position) + offset,
                            max_seq_len=past_key_values.max_seq_len,
                        )
                        _text_prefill_execution_plan(
                            self.model, handle, prospective, count
                        )
                _text_decode_execution_plan(self.model, handle)
                arguments = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    logits_to_keep=logits_to_keep,
                    **kwargs,
                )
                return run_continuation_segments(
                    lambda **call: original_forward(**call),
                    past_key_values,
                    self.model,
                    spans,
                    arguments,
                )

        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    segmented_forward = execution_serialized(segmented_forward)
    segmented_forward.__rpu_segmented_append__ = True
    causal_lm.forward = MethodType(segmented_forward, causal_lm)

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
        topology = decoder_topology_for_model(causal_lm)
        _install_causal_decoder_forward(
            causal_lm.model,
            arch="qwen3",
            scale_lists=_w8a16_scale_lists_for_qwen3_inner(causal_lm.model),
            chunk_envelope_for=lookup_causal_decoder,
            **({"topology": topology,
                "execution_config": getattr(causal_lm, "_rpu_execution", None)}
               if topology is not None else {}),
        )
    handle = causal_lm.model._rpu_decoder_handle  # single authoritative handle
    topology = getattr(causal_lm.model, "_rpu_decoder_topology", None)

    # 2. Push lm_head weight to the C++ side (state tracking for future
    #    C++ post_graph; currently used only as API symmetry).
    _push_lm_head_weight(causal_lm, handle)
    # The shared installer observed the topology before lm_head was enabled.
    # fused_forward calls the runner directly and needs the final descriptor.
    _text_decode_execution_plan(causal_lm.model, handle)

    from contextlib import nullcontext as _nullcontext
    import rpu_backend as _rb
    from rpu_backend.api.cache import RPUCache as _RPUCache_for_sig
    _GraphSignature = _rb.graph.GraphSignature

    # 3. Install fused_forward on the Qwen3ForCausalLM.
    #    Signature explicitly lists labels / logits_to_keep / output_*
    #    / return_dict so HF generate() can inspect.signature it correctly.
    @execution_serialized
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
        decode_plan_result = None
        batched_prefill = _is_batched_prefill(input_ids, inputs_embeds)
        if isinstance(past_key_values, _RPUCache_for_sig):
            validate_decoder_cache_topology(topology, past_key_values)
            if inputs_embeds is not None:
                _batch, _seq_len = (int(inputs_embeds.shape[0]),
                                    int(inputs_embeds.shape[1]))
            elif input_ids is not None:
                _batch, _seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
            else:
                _batch, _seq_len = 1, 0
            validate_decoder_cache_topology(topology, past_key_values, batch_size=_batch)
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
            if _batch == 1 and _seq_len > 1:
                spans = continuation_segments(
                    past_key_values.position, _seq_len,
                    getattr(self.model, "_rpu_execution", {}).get("prefill", {}),
                    capacity=past_key_values.max_seq_len,
                )
                if spans is not None:
                    if (input_ids is None) == (inputs_embeds is None):
                        raise AssertionError("must provide exactly one of input_ids or inputs_embeds")
                    if use_cache is False or return_dict is False:
                        raise NotImplementedError("RPU continuation requires use_cache=True and return_dict=True")
                    if self.lm_head.weight.dtype == torch.int8 and logits_to_keep != 1:
                        raise NotImplementedError("int8 continuation requires logits_to_keep=1")
                    from types import SimpleNamespace
                    for offset, count in spans:
                        if count > 1:
                            prospective = SimpleNamespace(
                                position=int(past_key_values.position) + offset,
                                max_seq_len=past_key_values.max_seq_len)
                            _text_prefill_execution_plan(self.model, h, prospective, count)
                    _text_decode_execution_plan(self.model, h)
                    return run_continuation_segments(
                        lambda **call: fused_forward(self, **call),
                        past_key_values, self.model, spans,
                        dict(input_ids=input_ids, inputs_embeds=inputs_embeds,
                             past_key_values=past_key_values, attention_mask=attention_mask,
                             position_ids=position_ids, cache_position=cache_position,
                             logits_to_keep=logits_to_keep, use_cache=use_cache,
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states,
                             return_dict=return_dict, **kwargs),
                    )
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
                _execution_len, _planned_chunk_size = prefill_plan[:2]
            elif _seq_len == 1:
                decode_plan_result = _text_decode_execution_plan(self.model, h)

            def _capture(batch, seq_len, plan=None):
                execution_len, planned_chunk_size = (
                    (int(plan[0]), int(plan[1]))
                    if plan is not None else (int(seq_len), 0)
                )
                plan_key = (
                    plan[2].graph_key_words()
                    if plan is not None
                    else (decode_plan_result.graph_key_words()
                          if int(seq_len) == 1 else ())
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
                            *(topology.identity() if topology is not None else ()),
                            int(getattr(
                                self.model, "_rpu_execution_generation", 0
                            )),
                            int(self.config.vocab_size),
                            chunk_policy_key(
                                self.model._rpu_decoder_handle
                            ),
                            planned_chunk_size,
                            prefill_position_key(
                                execution_len,
                                past_key_values.position,
                            ),
                            *plan_key,
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

        # Phase 2.5 decode-only fused lm_head:
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
            elif self.lm_head.weight.dtype == torch.uint8:
                selected = raw if logits_to_keep == 0 else raw[:, -logits_to_keep:, :]
                logits = _int8_lm_head_prefill_logits(self, selected)
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
    fused_forward = execution_serialized(fused_forward)
    causal_lm.forward = types.MethodType(fused_forward, causal_lm)

    _LOG.info("Patched Qwen3ForCausalLM (instance) with fused lm_head, "
              "handle=%d, vocab_size=%d",
              handle, causal_lm.config.vocab_size)

    return handle


@execution_serialized
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
    topology = getattr(causal_lm.model, "_rpu_decoder_topology", None)
    if isinstance(past_key_values, _RPUCache_for_sig):
        validate_decoder_cache_topology(topology, past_key_values)
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
        decode_plan_result = _text_decode_execution_plan(causal_lm.model, h)
        sig = _rb.graph.GraphSignature(
            op_id="rpu_causal_decoder_lm_head_top1",
            shapes=[batch, seq_len, causal_lm.model._rpu_decoder_hidden_size],
            dyn_dims=[
                causal_lm.model._rpu_decoder_num_layers,
                causal_lm.model._rpu_decoder_deepstack_hash,
                *(topology.identity() if topology is not None else ()),
                int(getattr(
                    causal_lm.model, "_rpu_execution_generation", 0
                )),
                int(causal_lm.config.vocab_size),
                # MR-D: same decoder forward, same SPM-layout hazard.
                chunk_policy_key(causal_lm.model._rpu_decoder_handle),
                prefill_position_key(seq_len, past_key_values.position),
                *decode_plan_result.graph_key_words(),
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
