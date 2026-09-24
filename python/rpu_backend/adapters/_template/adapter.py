"""Template ``adapter.py`` — v5 canonical entry point for a new model port.

Copy this directory to ``python/rpu_backend/adapters/<your_arch>/`` and edit
the placeholders, OR extract this file's body into a single
``python/rpu_backend/adapters/<your_arch>.py`` per the v5 flat-layout
convention. See ``docs/porting_guide.md#hello-world-skeleton`` for the full
walkthrough.

This consolidated template replaces the v4 5-file pattern
(``loader.py + model.py + weights.py + patches.py + runtime.py``) with a
single canonical entry point. The five concerns are still separated as
sections within this file:

  1. **Loader** — HF config validation + AutoModelForCausalLM load.
  2. **Weights** — per-Linear partition selection + swizzle.
  3. **Patches** — §3a (a)/(b) residual patches only; (d) class swaps go
     through ``runtime.decoder._install_*_class_swap``.
  4. **Runtime** — per-instance hardware attributes + first-forward setup.
  5. **Adapter class** — registration via
     ``runtime.registry.register_adapter("<HF_ARCH>", <ArchName>Adapter)``.

References:

- v3 framework contract: ``docs/architecture.md#fusedmodelbase-v3--framework-contract``
- 5-step porting checklist: ``docs/porting_guide.md#5-step-porting-checklist``
- Pitfall traps: ``docs/pitfalls.md``
- Concrete examples in tree:
  - ``python/rpu_backend/adapters/qwen3.py`` (Qwen3 reference)
  - ``python/rpu_backend/adapters/llama.py`` (Llama reference; proves shared
    ``causal_decoder_*`` op family per REN-03)
  - ``python/rpu_backend/adapters/pi05/`` (Pi0.5 multi-component VLA)
"""
from __future__ import annotations
from typing import Any, Optional

import torch
import torch.nn as nn


# =============================================================================
# Section 1 — Loader: HF config validation + weight load
# =============================================================================

# Replace this frozenset with your model's profile tuples. Each tuple is:
#   (hidden_size, intermediate_size, num_attention_heads, num_key_value_heads)
# Add one row per supported size variant. Mirror
# ``adapters/qwen3.py``'s _SUPPORTED_PROFILES for the canonical pattern.
_SUPPORTED_PROFILES: frozenset[tuple[int, int, int, int]] = frozenset({
    # Example for Llama-3.2-1B: (2048, 8192, 32, 8)
    # Example for Qwen3-0.6B:   (1024, 3072, 16, 8)
})


def load(repo_id: str, dtype: torch.dtype = torch.float16,
         **hf_kwargs: Any) -> nn.Module:
    """Return CPU-resident HF model ready for ``.to("rpu")``.

    The 6-step loader contract (D-3-03 canonical pattern, derived from
    ``adapters/pi05/loader.py:270-300`` and ``adapters/qwen3.py:47-87``):

      1. Validate the HF config tuple is in ``_SUPPORTED_PROFILES``. Raise
         ``UnsupportedModelError`` BEFORE the weight download.
      2. Probe optional dependencies (e.g., lerobot for VLA models). On
         missing, raise ``ImportError`` with ``"pip install <dep>"``
         actionable hint per D-4-09.
      3. Validate the user-supplied dtype (most ports require fp16; raise
         ``RPUUnsupportedDtypeError`` otherwise).
      4. Call ``AutoModelForCausalLM.from_pretrained(repo_id, **safe_hf_kwargs)``
         forcing ``device_map="cpu"`` so the swizzle in Section 2 runs on CPU.
      5. Return a CPU-resident ``nn.Module`` ready for the caller's
         ``.to("rpu")``.
      6. The returned module is wrapped in your ``<ArchName>Adapter`` class
         (Section 5) which ``runtime.registry.register_adapter("<HF_ARCH>",
         <ArchName>Adapter)`` binds into the registry.

    Args:
        repo_id: HF Hub repo or local path.
        dtype: target dtype; most RPU ports require fp16.
        **hf_kwargs: forwarded to AutoModelForCausalLM.from_pretrained;
                     ``device_map`` / ``torch_dtype`` are owned by this loader.

    Raises:
        UnsupportedModelError: HF config not in ``_SUPPORTED_PROFILES``.
        RPUUnsupportedDtypeError: dtype != fp16 (port-dependent).
        ImportError: missing optional dependency (port-dependent).

    See ``docs/porting_guide.md`` Step 1 for the full walkthrough.
    """
    raise NotImplementedError(
        "see docs/porting_guide.md Step 1 — Port HF model body to C++"
    )


# =============================================================================
# Section 2 — Weights: per-Linear partition selection + swizzle
# =============================================================================
#
# P1 SKIP_LINEAR_NAMES rule (CLAUDE.md "Pitfall red flags" +
# docs/pitfalls.md#p1--double-swizzle): specialized Linear modules (e.g.,
# ``PiGemmaRMSNorm.dense``) MUST be in ``SKIP_LINEAR_NAMES`` OR save/restore
# weights around ``swizzle_model_inplace``. For plain CausalLM ports the 7
# standard Linears (q/k/v/o/gate/up/down_proj) all use the recursive walk —
# no skip set needed.
#
# Per-layer naming convention reminder:
#   - Attention block: ``model.layers.<L>.self_attn.{q,k,v,o}_proj.weight``
#   - MLP block:       ``model.layers.<L>.mlp.{gate,up,down}_proj.weight``
#   - LayerNorms:      handled by Section 3 (patches) / Section 4 (runtime)
#   - Embeddings:      handled by Section 1 (loader); tied weights auto-untied

# Set of fully-qualified attribute names (relative to model root) to SKIP
# during the recursive Linear-swizzle walk. Empty set = swizzle every leaf
# Linear (the common case for plain CausalLM ports).
SKIP_LINEAR_NAMES: set[str] = set()


def swizzle(model: nn.Module,
            partition_overrides: Optional[dict] = None) -> None:
    """Swizzle every leaf Linear in ``model`` to RPU layout in place.

    Uses ``rpu_backend.runtime.weights.swizzle_model_inplace`` as the canonical
    recursive walker. Auto-detects row vs col partition:

    - **Row partition** — split K across cores (used for ``o_proj``,
      ``down_proj``).
    - **Col partition** — split N across cores (used for QKV, gate/up,
      projectors).

    Args:
        model: CPU-resident HF model returned by ``load()``.
        partition_overrides: optional dict {fully_qualified_name: int}
                             where int is 0 (row-partition) or 1 (col-partition).
                             Most ports leave this as None (auto-pick).

    Side effects:
        Mutates ``model`` parameters in place. Caller MUST NOT call this twice
        on the same model — Pitfall 1 (double-swizzle) corrupts weights with
        no exception raised.

    See ``docs/api_reference.md#weight-conversion-api`` for the canonical
    swizzle entry point + RPUCache 7-D layout reference.

    See ``docs/porting_guide.md`` Step 2 for the full walkthrough.
    """
    raise NotImplementedError(
        "see docs/porting_guide.md Step 2 — Swizzle weights"
    )


# =============================================================================
# Section 3 — Patches: §3a (a)/(b) residual patches only
# =============================================================================
#
# §3a triage criteria (design notes 2026-04-22 §3a):
#   (a) Forward-path op-tensor layout mismatch (RPU [b,s,h,d] vs torch
#       [b,h,s,d]) where upstream HF exposes no seam.
#       → Keep here; "# patch-reason: (a) <where + why>".
#   (b) Model-specific hardware-constraint attribute injection (for example,
#       the Gemma-family compatibility `_rpu_chunk_size` path).
#       → Keep here; "# patch-reason: (b) <where + why>".
#   (c) Entire forward() rewritten.
#       → NOT a patch. Subclass via v3 FusedModelBase.
#   (d) Single class swap (e.g., RMSNorm → RPURMSNorm).
#       → Call runtime.decoder._install_rmsnorm_class_swap from Section 5.
#   (e) Pure weight transformation.
#       → Move to Section 2 (weights).
#
# scripts/check_patch_reasons.py enforces 100% comment coverage in CI; every
# def patch_* MUST have a "# patch-reason: (a)" or "(b)" comment on the line
# above. Categories (c)/(d)/(e) MUST NOT appear in this section.
#
# Idempotent class-level installer pattern (from adapters/pi05/patches.py):
#
#     def _install_<thing>_class_patch() -> None:
#         '''CLASS-level idempotent patch.'''
#         try:
#             from <some_lib> import <SomeClass>
#         except ImportError:
#             return
#         if getattr(<SomeClass>, "_rpu_<thing>_patched", False):
#             return
#         # ... apply the patch + set the sentinel attribute ...
#         <SomeClass>._rpu_<thing>_patched = True


# patch-reason: (a) example placeholder — REPLACE with your real (a)/(b) patch
# OR delete this stub if your port needs no patches.
def patch_template_forward_path_layout(*args: Any, **kwargs: Any) -> None:
    """Placeholder for a §3a (a) op-tensor layout patch.

    Most ports do NOT need this. Delete this stub for ports without (a)
    patches. See ``docs/porting_guide.md`` Step 3 for the full walkthrough.
    """
    raise NotImplementedError(
        "see docs/porting_guide.md Step 3 — Declare SPM + register handle"
    )


# =============================================================================
# Section 4 — Runtime: per-instance hardware attributes + first-forward setup
# =============================================================================
#
# Execution planning is cold: public entry points normalize ``rpu_execution``
# before loading weights and adapters pass it to their native handle installer.
# It must not be reimplemented as a hot attribute. Remaining A9 hw-attribute
# slots are set on ``model`` AFTER ``.to("rpu")``:
#
# | Slot                  | Type           | Cold-set | Hot-set | Default                  |
# |-----------------------|----------------|----------|---------|--------------------------|
# | _rpu_chunk_size       | int (0=auto)   | yes      | model-specific | Gemma-family compatibility only |
# | _rpu_spm_mode         | bool           | yes      | NO (full rebuild) | False                       |
# | _rpu_warmup           | int            | yes      | yes (no reset needed) | env RPU_WARMUP → 0     |
# | _rpu_debug_export     | bool           | yes      | yes (no reset needed) | False                  |
#
# Cold-set = before first forward post-``.to("rpu")``.
# Hot-set = between forwards; requires ``rpu_backend.reset_graph_cache()``.
# Unknown ``_rpu_*`` attrs → ``RPUConfigError``.
#
# Pitfall 5 (``docs/pitfalls.md#p5--per-instance-_rpu_chunk_size``) applies to
# Gemma-family adapters that explicitly read ``_rpu_chunk_size``. Plain
# CausalDecoder adapters use the cold ``rpu_execution`` mapping instead.
#
# VLA models compose components (Pi0.5: siglip → reset → gemma → reset →
# adarms → reset) in this section. CausalLM models (Qwen3, Llama, Phi,
# Mistral) use the all-layers-once direct path — this section is ≤ 100 LOC
# for plain CausalLM.

# A9 supported attributes (used by the runtime to validate user-set
# ``_rpu_*`` attrs at .to("rpu") cold-set time).
_A9_SUPPORTED_ATTRS: frozenset[str] = frozenset({
    "_rpu_chunk_size",
    "_rpu_execution",
    "_rpu_spm_mode",
    "_rpu_warmup",
    "_rpu_debug_export",
})


def apply_rpu_runtime(model: nn.Module,
                      hw_attrs: Optional[dict[str, Any]] = None) -> None:
    """Apply per-instance hw attribute defaults + first-forward hooks.

    Args:
        model: RPU-resident model returned by ``<ArchName>Adapter.to_rpu()``.
        hw_attrs: optional dict {name: value} for cold-set attribute injection.
                  Each name must be in ``_A9_SUPPORTED_ATTRS`` or
                  ``RPUConfigError`` raises.

    Side effects:
        - Sets adapter-picked defaults for any attr the user did NOT set.
        - Validates user-set ``_rpu_*`` attrs against ``_A9_SUPPORTED_ATTRS``.
        - Registers a first-forward hook that finalizes graph compilation
          (so subsequent forwards REPLAY the cached graph; hot-set requires
          ``rpu_backend.reset_graph_cache()``).

    See ``docs/porting_guide.md`` Step 3 for the full walkthrough.
    """
    raise NotImplementedError(
        "see docs/porting_guide.md Step 3 — Declare SPM + register handle"
    )


# =============================================================================
# Section 5 — Adapter class + registry binding
# =============================================================================
#
# CANONICAL FORM (mirrors adapters/qwen3.py / adapters/llama.py):
#
#     class <ArchName>Adapter:
#         """Adapter for <HF_ARCH>ForCausalLM."""
#
#         @staticmethod
#         def supported(repo_id: str) -> bool:
#             # config validation — see Section 1
#             return ...
#
#         @staticmethod
#         def load(repo_id: str, dtype, **kwargs):
#             return load(repo_id, dtype, **kwargs)
#
#         @staticmethod
#         def to_rpu(model, hw_attrs=None):
#             # 1. patch model classes (idempotent via _rpu_patched flags)
#             # 2. swizzle weights (Section 2)
#             # 3. move to RPU
#             # 4. apply runtime hw attrs (Section 4)
#             return model
#
#     # Register
#     from rpu_backend.runtime.registry import register_adapter
#     register_adapter("<HF_ARCH>ForCausalLM", <ArchName>Adapter)
#
# Replace ``<ArchName>``, ``<HF_ARCH>`` with your model's identifiers. See
# ``adapters/qwen3.py:47-87`` for the canonical reference port.
#
# When extracting this template into the v5 flat layout, the resulting
# ``adapters/<your_arch>.py`` typically ends with the ``register_adapter(...)``
# call at module scope.


__all__ = [
    "_SUPPORTED_PROFILES",
    "SKIP_LINEAR_NAMES",
    "_A9_SUPPORTED_ATTRS",
    "load",
    "swizzle",
    "patch_template_forward_path_layout",
    "apply_rpu_runtime",
]
