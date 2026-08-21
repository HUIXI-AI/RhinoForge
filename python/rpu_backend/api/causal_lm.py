"""RPUModelForCausalLM: thin HF-compatible entry point.

Load on CPU with ``from_pretrained(...)`` and then call ``.to('rpu')``, or
pass ``device='rpu'`` for the one-step shortcut. Build the cache with
``RPUCache.from_model(model, ...)``. This entry point is not an HF subclass.
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
# Import the cache from its public owning module.
from rpu_backend.api.cache import RPUCache

# Store a weak reference to the live instance rather than an object id. Object
# ids can be reused after collection; dereferencing preserves identity and
# liveness for the single-handle gate.
_LIVE_REF: "weakref.ref | None" = None
_LIVE_LOCK = threading.RLock()
_LIVE_TERMINAL_REASON: str | None = None


def _release_live_instance(model=None) -> None:
    """Release the single-handle claim when its owner is gone or matches.

    This never clears a process-terminal latch installed by an irreversible
    owner; that state intentionally survives explicit release and owner GC.

    Args:
        model: Optional model. If provided, only releases when the live ref
               points to this specific model (safe against double-release
               races from finalizer + explicit cleanup). If None, release only
               after the weak-reference target has been collected.

    Behavior:
      - ``model is None``: release if ``_LIVE_REF()`` returned None.
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
            # The weakref callback clears only after the target is gone.
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

    Same-instance re-entry is a no-op so that
    `model.to('rpu').to('rpu')` and `model.to('rpu'); model.to('cpu'); model.to('rpu')`
    do not raise against the model itself.

    Liveness is determined by the weakref, not by `id(model)`. If the previous
    claim's target has been GC'd (``ref() is None``), we quietly take over. If
    it is still live and is not `model`, we
    raise `RPUSingleHandleError`. Same-instance re-entry returns immediately
    with the same id (the adapter's `to_rpu()` uses `_rpu_is_ready` to skip
    re-swizzle).

    Returns `id(model)` for debug/logging; callers do not compare it across
    claims.
    A process-terminal owner is stricter: after its first materialization,
    only idempotent re-entry by that still-live owner is accepted; a new owner
    always requires a fresh Python process.
    """
    global _LIVE_REF
    with _LIVE_LOCK:
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


def _poison_live_instance(model, reason: str) -> None:
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


class RPUModelForCausalLM:
    """Stateless entry point that is not a subclass of any HF class.

    Use `RPUModelForCausalLM.from_pretrained(repo, dtype=torch.float16)` to load
    on CPU; call `.to('rpu')` on the returned model to perform weight swizzle +
    per-instance patches. Or pass `device='rpu'` to combine both.
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
        **hf_kwargs,
    ):
        execution_config = normalize_rpu_execution(
            rpu_execution,
            entry_point="RPUModelForCausalLM.from_pretrained",
            supported={
                "prefill": ("chunk_size", "padding_rows", "padding_budget"),
            },
        )
        # Reject unsupported dtypes before loading weights.
        if dtype is not torch.float16:
            raise RPUUnsupportedDtypeError(
                f"RPU kernels require torch.float16, got {dtype}. "
                "Load the CPU reference separately if you need an fp32 baseline for comparison."
            )

        # Reject reserved kwargs before calling AutoModelForCausalLM. This
        # avoids both the unhelpful Python
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
        trust_remote_code = hf_kwargs.pop("trust_remote_code", False)
        if trust_remote_code is not False:
            raise ValueError(
                "RPUModelForCausalLM.from_pretrained requires "
                "trust_remote_code=False; custom model code is not supported."
            )

        move_to_rpu = resolve_loader_device(
            device,
            entry_point="RPUModelForCausalLM.from_pretrained",
        )

        # Run the config-only preflight before the full model load. Without
        # this, an unsupported large checkpoint could download and materialize
        # its weights before raising
        # UnsupportedModelError when the adapter's `__init__` runs the profile
        # guard. AutoConfig is a tiny JSON parse that costs milliseconds.
        # Pass only repo-locating kwargs through; loader-specific ones are
        # already filtered by the reserved-kwargs check above.
        _CONFIG_KWARGS = {"cache_dir", "revision", "subfolder", "local_files_only",
                           "token", "force_download", "proxies"}
        cfg_kwargs = {k: v for k, v in hf_kwargs.items() if k in _CONFIG_KWARGS}
        # AutoConfig.from_pretrained may raise OSError
        # (missing/unreachable repo), ValueError (malformed entries), or
        # json.JSONDecodeError (which IS-A ValueError — corrupted config.json).
        # Wrap those failures in the public backend error family so callers
        # relying on `except RPUBackendError:` do not see a raw HF loader error.
        # Re-raise as UnsupportedModelError with chained cause (preserves
        # traceback via `from e`).
        try:
            config = AutoConfig.from_pretrained(
                hf_repo_or_path, trust_remote_code=False, **cfg_kwargs
            )
        except (OSError, ValueError) as e:
            raise UnsupportedModelError(
                f"Could not load HF config for {hf_repo_or_path!r}: "
                f"{type(e).__name__}: {e}. "
                "See docs/api_reference.md#causal-language-models."
            ) from e
        arch = resolve_config_architecture(
            config,
            source=hf_repo_or_path,
            entry_point="RPUModelForCausalLM.from_pretrained",
            supported_builtins=cls._SUPPORTED_BUILTIN_ARCHITECTURES,
        )
        adapter_cls = get_adapter(arch)
        # If the adapter exposes a preflight classmethod, run it with the
        # config-only object. Adapters are expected to validate
        # config-derived constraints (envelope, dtype, etc.) here.
        if hasattr(adapter_cls, "preflight"):
            adapter_cls.preflight(config)  # rejects unsupported dtype/profile combinations
        if hasattr(adapter_cls, "preflight_execution"):
            adapter_cls.preflight_execution(config, execution_config)

        # Profile validated — safe to do the full load. W8A16 checkpoints need
        # a custom loader because HF's generic path casts int8 projection weights
        # back into fp16 Parameters.
        from rpu_backend.quant.load import is_w8a16_config, load_w8a16_model
        if is_w8a16_config(config):
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
                trust_remote_code=False,
                **hf_kwargs,
            )
        model.eval()
        model._rpu_execution = execution_config

        # Bind the adapter to the loaded model. Its constructor re-checks the
        # profile before installing any runtime state.
        adapter = adapter_cls(model)
        adapter._rpu_execution = execution_config

        if move_to_rpu:
            # Combine loading and .to('rpu') when requested.
            return adapter.to_rpu()

        # Return the CPU-resident model with an adapter-bound ``.to('rpu')``.
        return model
