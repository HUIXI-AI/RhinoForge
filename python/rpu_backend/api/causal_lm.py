"""RPUModelForCausalLM: thin HF-compatible entry point.

v5-01b (Phase 01.3 D-13 + D-14): `build_cache` and `build_rpu_cache` factories
deleted; v5 canonical factory is `RPUCache.from_model(model, ...)` (ADR §2.5).
v5-02 D-04: relocated this module from `transformers/_causal_lm.py` per the
api/ ownership contract — the api package is the canonical home for public
entry points.

D-01: two-step canonical (`from_pretrained(...)` → `.to('rpu')`); `device='rpu'`
shortcut combines both. D-02: thin wrapper, NOT a subclass of any HF class.
"""
from __future__ import annotations
import threading
import weakref

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from rpu_backend.api.errors import RPUUnsupportedDtypeError, RPUSingleHandleError, UnsupportedModelError
from rpu_backend.api._execution import normalize_rpu_execution
from rpu_backend.api._loading import resolve_config_architecture, resolve_loader_device
from rpu_backend.runtime.registry import get_adapter
# D-40 A2 DAG (Plan B Task 2 Step 2.1): import canonical home directly (not via
# model_converter shim). top -> core.weights is an allowed inward edge; the
# reverse (core -> transformers) is forbidden and check_import_graph.py confirms.
from rpu_backend.api.cache import RPUCache

# D-17 single-handle gate. WR-03 (codex iter9): store a weakref to the live
# instance rather than `id(model)`. `id()` is ONLY unique among live objects;
# a GC'd model's id can be reused by a freshly-allocated one, which would
# collide the idempotent re-entry branch (skipping the swizzle) AND cause the
# queued finalize to clear the gate for the NEW instance. Using a weakref
# decouples identity from liveness: ref() returns None once the target is
# GC'd, and identity is compared via `ref() is model` on the dereferenced
# object rather than on an integer address.
_LIVE_REF: "weakref.ref | None" = None
_LIVE_LOCK = threading.RLock()
_LIVE_TERMINAL_REASON: str | None = None


def _release_live_instance(model=None) -> None:
    """Release the D-17 single-handle claim.

    This never clears a process-terminal latch installed by an irreversible
    owner; that state intentionally survives explicit release and owner GC.

    CR-R7 BLOCKER A signature upgrade: accepts optional `model` arg for
    model-scoped release (Pi05Adapter exception path); preserves zero-arg form
    for the existing weakref GC callback at line 69
    (``lambda _ref: _release_live_instance()``).

    Args:
        model: Optional model. If provided, only releases when the live ref
               points to this specific model (safe against double-release
               races from finalizer + explicit cleanup). If None (the weakref
               GC callback path), uses the legacy "release if target is
               GC'd" behavior — this preserves WR-03 semantics.

    Behavior:
      - ``model is None``:    legacy zero-arg path. Release if ``_LIVE_REF()``
                               returned None (target was GC'd). This is the
                               weakref finalize callback.
      - ``model is not None``: adapter explicit-release path. Release iff
                               ``_LIVE_REF()`` returns the passed ``model``
                               (or None — the target was already GC'd between
                               claim and release).
    """
    global _LIVE_REF
    with _LIVE_LOCK:
        if _LIVE_REF is None:
            return
        live = _LIVE_REF()
        if model is None:
            # Legacy WR-03 callback path: only clear if the target is gone.
            if live is None:
                _LIVE_REF = None
            return
        # Explicit-release path: clear iff our model owns the claim
        # (or target is already gone).
        if live is model or live is None:
            _LIVE_REF = None


def _claim_live_instance(model) -> int:
    """Atomically claim the process-wide RPU slot for a model or policy.

    Idempotent for the same owner instance.

    HIGH-3 (codex iter1): same-instance re-entry MUST be a no-op so that
    `model.to('rpu').to('rpu')` and `model.to('rpu'); model.to('cpu'); model.to('rpu')`
    do not raise against the model itself.

    WR-03 (codex iter9): liveness is determined by the weakref, not by
    `id(model)`. If the previous claim's target has been GC'd (`ref() is
    None`), we quietly take over. If it's still live AND is not `model`, we
    raise `RPUSingleHandleError`. Same-instance re-entry returns immediately
    with the same id (the adapter's `to_rpu()` uses `_rpu_is_ready` to skip
    re-swizzle).

    Public API unchanged: returns `id(model)` as before (the integer is used
    only for debug/logging by callers; no caller compares it across claims).
    A process-terminal owner is stricter: after its first materialization,
    only idempotent re-entry by that still-live owner is accepted; a new owner
    always requires a fresh Python process.
    """
    global _LIVE_REF
    with _LIVE_LOCK:
        from rpu_backend.api._execution import _require_execution_process_safe

        _require_execution_process_safe()
        if _LIVE_TERMINAL_REASON is not None:
            live = _LIVE_REF() if _LIVE_REF is not None else None
            if live is model:
                return id(model)
            raise RPUSingleHandleError(
                "This process cannot create another RPU model/policy after a "
                f"process-terminal owner: {_LIVE_TERMINAL_REASON}. Restart "
                "the Python process before loading another RPU model/policy."
            )
        if _LIVE_REF is not None:
            live = _LIVE_REF()
            if live is None:
                # Previous model was GC'd — release the slot and fall through to claim.
                _LIVE_REF = None
            elif live is model:
                return id(model)  # idempotent re-entry — adapter's to_rpu() detects via _rpu_is_ready
            else:
                raise RPUSingleHandleError(
                    "An RPU model/policy is already live in this process "
                    f"(type={type(live).__name__}, id={id(live)}). "
                    "Call `del model; import gc; gc.collect()` to release it "
                    "before loading a new RPU model/policy. (If gc.collect() "
                    "does not release the handle, the previous owner is held "
                    "by another reference or cycle; release it or restart the "
                    "Python process.)"
                )
        _LIVE_REF = weakref.ref(model, lambda _ref: _release_live_instance())
        return id(model)


def _poison_live_instance(model, reason: str, *, unsafe: bool = False) -> None:
    """Make the current claim process-terminal after irreversible setup."""
    global _LIVE_TERMINAL_REASON
    with _LIVE_LOCK:
        live = _LIVE_REF() if _LIVE_REF is not None else None
        if live is not model:
            raise RPUSingleHandleError(
                "Cannot mark a non-owner as process-terminal."
            )
        if _LIVE_TERMINAL_REASON is None:
            _LIVE_TERMINAL_REASON = str(reason)
        if unsafe:
            from rpu_backend.api import _execution

            _execution._mark_execution_process_unsafe(reason)


class RPUModelForCausalLM:
    """Stateless entry point. NOT a subclass of any HF class (D-02).

    Use `RPUModelForCausalLM.from_pretrained(repo, dtype=torch.float16)` to load
    on CPU; call `.to('rpu')` on the returned model to perform weight swizzle +
    per-instance patches. Or pass `device='rpu'` to combine both (D-01 shortcut).
    `quantization='w4a16'` selects TP8/G32 dense Qwen3 decoder quantization;
    'w4a16_lm_head' also quantizes the head. Embedding and norms stay FP16.
    The 32B recipes require the one-step device='rpu' streaming loader.
    """

    # Reserved hf_kwargs the library manages internally — passing any of these
    # via **hf_kwargs collides with the library's explicit args (which produces
    # a confusing duplicate-keyword TypeError) OR bypasses the CPU-first invariant.
    _RESERVED_HF_KWARGS = frozenset({
        "torch_dtype", "dtype",          # library passes torch_dtype=dtype
        "device_map", "device",          # library forces device_map='cpu'
        "low_cpu_mem_usage",             # library passes True
        "tp_plan",                       # HF placement knob — bypasses CPU load
        "load_in_8bit", "load_in_4bit",  # quantization — incompatible with swizzle
        "quantization_config",
    })
    _SUPPORTED_BUILTIN_ARCHITECTURES = frozenset({
        "LlamaForCausalLM",
        "Qwen3ForCausalLM",
    })

    @classmethod
    def from_pretrained(
        cls,
        hf_repo_or_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device | None = None,
        rpu_execution=None,
        quantization: str | None = None,
        **hf_kwargs,
    ):
        if hf_kwargs.get("trust_remote_code", False) is not False:
            raise ValueError("RhinoForge requires trust_remote_code=False")
        if quantization not in (None, "w4a16", "w4a16_lm_head"):
            raise UnsupportedModelError(
                "quantization must be None, 'w4a16' or 'w4a16_lm_head' (dense Qwen3, G32, TP8)"
            )
        execution_config = normalize_rpu_execution(
            rpu_execution,
            entry_point="RPUModelForCausalLM.from_pretrained",
            supported={
                "prefill": ("chunk_size", "padding_rows", "padding_budget", "linear_acc32"),
                "model": ("num_cores",),
            },
        )
        model_scope_requested = rpu_execution is not None and "model" in rpu_execution
        # D-19: dtype enforcement BEFORE HF load
        if dtype is not torch.float16:
            raise RPUUnsupportedDtypeError(
                f"RPU kernels require torch.float16, got {dtype}. "
                "Load the CPU reference separately if you need an fp32 baseline for comparison."
            )

        # MEDIUM-08 (codex iter1): explicitly reject reserved kwargs BEFORE the
        # call to AutoModelForCausalLM. Avoids both the unhelpful Python
        # duplicate-key TypeError AND the case where a user passes `tp_plan` /
        # `device_map={...}` and bypasses the CPU-first load invariant.
        collisions = cls._RESERVED_HF_KWARGS & set(hf_kwargs)
        if collisions:
            raise ValueError(
                f"RPUModelForCausalLM.from_pretrained: hf_kwargs cannot include "
                f"reserved key(s) {sorted(collisions)}; the library manages these "
                f"internally (load on CPU as fp16; user calls .to('rpu') next). "
                f"Remove these keys from your call."
            )

        move_to_rpu = resolve_loader_device(
            device,
            entry_point="RPUModelForCausalLM.from_pretrained",
        )

        # iter7 HIGH (codex): D-12 fail-fast preflight via AutoConfig BEFORE the
        # full HF model load. Without this, `from_pretrained("Qwen/Qwen3-14B")`
        # downloads + materializes ~30GB of weights and only THEN raises
        # UnsupportedModelError when the adapter's `__init__` runs the profile
        # guard. AutoConfig is a tiny JSON parse that costs milliseconds.
        # Pass only repo-locating kwargs through; loader-specific ones are
        # already filtered by the reserved-kwargs check above.
        _CONFIG_KWARGS = {"cache_dir", "revision", "subfolder", "local_files_only",
                           "trust_remote_code", "token", "force_download", "proxies"}
        cfg_kwargs = {k: v for k, v in hf_kwargs.items() if k in _CONFIG_KWARGS}
        # WR-01 (codex iter9): HF's AutoConfig.from_pretrained may raise OSError
        # (missing/unreachable repo), ValueError (malformed entries), or
        # json.JSONDecodeError (which IS-A ValueError — corrupted config.json).
        # Without this wrap, those escape the D-18 RPUBackendError family and
        # callers relying on `except RPUBackendError:` see a raw HF loader error.
        # Re-raise as UnsupportedModelError with chained cause (preserves
        # traceback via `from e`).
        try:
            config = AutoConfig.from_pretrained(hf_repo_or_path, **cfg_kwargs)
        except (OSError, ValueError) as e:
            raise UnsupportedModelError(
                f"Could not load HF config for {hf_repo_or_path!r}: "
                f"{type(e).__name__}: {e}. "
                f"See docs/api_reference.md §Supported Models."
            ) from e
        arch = resolve_config_architecture(
            config,
            source=hf_repo_or_path,
            entry_point="RPUModelForCausalLM.from_pretrained",
            supported_builtins=cls._SUPPORTED_BUILTIN_ARCHITECTURES,
        )
        adapter_cls = get_adapter(arch)  # D-06: raises UnsupportedModelError if no adapter
        # The first pass admits the union of public loader fields; the resolved
        # adapter must approve its own subset before any checkpoint is loaded.
        declared_execution = dict(execution_config)
        if model_scope_requested:
            declared_execution.setdefault("model", {})
        execution_config = normalize_rpu_execution(
            declared_execution,
            entry_point="RPUModelForCausalLM.from_pretrained",
            supported=getattr(adapter_cls, "EXECUTION_SUPPORTED", {
                "prefill": ("chunk_size", "padding_rows", "padding_budget"),
            }),
        )
        # D-12 fail-fast: if the adapter exposes a preflight classmethod, run it
        # with the config-only object. Adapters are expected to validate
        # config-derived constraints (envelope, dtype, etc.) here.
        staged_w4 = False
        if quantization is not None:
            if arch != "Qwen3ForCausalLM":
                raise UnsupportedModelError("on-install W4 requires a plain Qwen3 FP16 source")
            profile = adapter_cls.preflight_quantization(config, execution_config, quantization)
            staged_w4 = profile.num_layers == 64
            if staged_w4 and not move_to_rpu:
                raise UnsupportedModelError(
                    "Qwen3-32B W4 requires device='rpu' for bounded source staging; "
                    "CPU-first loading would materialize the complete FP16 checkpoint"
                )
        elif hasattr(adapter_cls, "preflight"):
            adapter_cls.preflight(config)
        if hasattr(adapter_cls, "preflight_execution"):
            adapter_cls.preflight_execution(config, execution_config)

        # Profile validated — safe to do the full load. W8A16 checkpoints need
        # a custom loader because HF's generic path casts int8 projection weights
        # back into fp16 Parameters.
        from rpu_backend.quant.load import is_w8a16_config, load_w8a16_model
        if staged_w4:
            from rpu_backend.quant.load_qwen3_quantized import stage_qwen3_w4a16_for_rpu
            model = stage_qwen3_w4a16_for_rpu(
                config, hf_repo_or_path, quantization=quantization, dtype=dtype, **cfg_kwargs)
        elif is_w8a16_config(config):
            model = load_w8a16_model(
                config,
                hf_repo_or_path,
                dtype=dtype,
                **cfg_kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                hf_repo_or_path,
                torch_dtype=dtype,
                device_map="cpu",
                low_cpu_mem_usage=True,
                **hf_kwargs,
            )
        model.eval()
        model._rpu_execution = execution_config

        # D-02: instantiate adapter on the loaded model. The adapter's __init__
        # re-checks the profile (belt-and-suspenders — preflight already ran).
        adapter = adapter_cls(model, **({"quantization": quantization}
                                     if quantization is not None else {}))
        if not hasattr(adapter, "_rpu_execution"):
            adapter._rpu_execution = model._rpu_execution

        if move_to_rpu:
            # D-01 shortcut: combine .to('rpu') in one call
            return adapter.to_rpu()

        # D-01 canonical two-step: return CPU-resident HF model with the adapter
        # bound onto the instance for `.to('rpu')` to find. Adapter is responsible
        # for monkey-patching `.to` to intercept "rpu" target. (Wired in 02-02;
        # 02-01 ships no adapter so this path is exercised by the smoke test only
        # via the missing-adapter error.)
        return model


# v5-01b (Phase 01.3 D-13 + D-14): `build_cache` and `build_rpu_cache` factories
# deleted. The v5 canonical factory is `RPUCache.from_model(model, ...)` (ADR §2.5);
# the v4-era `_REQUIRED` sentinel migrated alongside (api/cache.py:32 already uses
# the same pattern). Karpathy meta-rule 4 — goal-driven: contract relocated to its
# v5 home; this file no longer exposes the legacy wrappers.
