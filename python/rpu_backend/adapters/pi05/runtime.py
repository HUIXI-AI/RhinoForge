"""Pi0.5 runtime helpers: handle map + denoise driver (decomposed) + destroyer (PI05-01 reshape Plan 03-01).

Per D-3-02, runtime.py owns the denoise loop. The 270-LOC
_install_sample_actions_class_patch from _adapter.py:483-751 is decomposed
into 6 nested helpers per D-3-02 (helper boundaries match the COMMENTED
sections in the source: 507-540, 565-586, 588-607, 609-657, 666-747).

v5-02 D-11 (G6-B) + v5-03 C1: the 3 stale `core.components.*` F401 side-effect
imports (SiglipComponent / GemmaComponent / AdaRMSComponent) and the 8-LOC
explanatory comment block were dropped. The classes were never bound to
model instances — the legacy sibling-converter installers continued to
monkey-patch `model.forward = types.MethodType(...)`. v5-03 C1 deleted
`core/components/` (deleted in v5-03 atomic commit); this file only ever
held them as side-effect imports, never invoked.

v5-02 / B3: relocated from transformers/pi05/runtime.py to adapters/pi05/runtime.py.

Task 4.3: lifted helpers 2-5 to top-level, added _build_attention_mask_4d,
_kv_prefix_full_hash, _run_denoise_fused, _run_denoise_python_baseline,
and replaced nested _run_denoise with dispatcher.
"""
from __future__ import annotations
import os
import threading
import weakref
from typing import Any

import torch

from rpu_backend.runtime import rpu_env_bool
from rpu_backend.runtime.masks import versioned_cpu_mask
from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
from rpu_backend.runtime.log import _LOG
from rpu_backend.api.errors import RPUBackendError
from rpu_backend.api._execution import (
    execution_serialized,
    normalize_rpu_execution,
    resolve_component_rpu_execution,
)
from rpu_backend.runtime.execution_planner import (
    A6PlanResult,
    GRAPH_COMPOSITE_CHILD,
    plan_fixed_component_execution,
)


from rpu_backend.api._execution import (
    PI05_VISION_COMPONENT, PI05_TEXT_COMPONENT, PI05_ACTION_COMPONENT,
    PI05_EXECUTION_COMPONENTS, resolve_pi05_execution_components,
)


def _component_execution_plan(
    owner: Any,
    *,
    component: str,
    stage: str,
    logical_len: int,
    execution_len: int,
    chunk_size: int,
    position: int = 0,
    queue_owner_id: int = 0,
) -> A6PlanResult:
    components = getattr(owner, "_pi05_execution_components", {})
    component_config = components.get(component, {})
    component_generations = getattr(
        owner, "_pi05_execution_component_generations", {}
    )
    return plan_fixed_component_execution(
        logical_len,
        execution_len=execution_len,
        chunk_size=chunk_size,
        component_id=component,
        stage=stage,
        generation=int(component_generations.get(component, 0)),
        position=position,
        stage_config=component_config.get(stage, {}),
        queue_owner_id=queue_owner_id,
    )


def _plan_pi05_denoise_action_execution(
    owner,
    *,
    handle: int,
    logical_len: int,
    position: int,
    kv_len: int,
    cache_capacity: int,
    loop_mode: bool,
    num_steps: int,
    _cost_request=None,
    graph_cache=None,
) -> A6PlanResult:
    """Resolve the native denoise descriptor before Graph admission."""
    components = getattr(owner, "_pi05_execution_components", {})
    stage = components.get(PI05_ACTION_COMPONENT, {}).get("action", {})
    requested = stage.get("chunk_size", "auto")
    if _cost_request is not None:
        from rpu_backend.runtime.decoder import _cold_text_cost_request

        if type(handle) is not int or handle != owner._rpu_fused_denoise_handle:
            raise ValueError("cold Pi0.5 action planning requires its actual handle")
        request, _ = _cold_text_cost_request(
            owner, ("pi05_denoise_step", handle), None, _cost_request, position,
            component=PI05_ACTION_COMPONENT, stage="action")
        if request.padding_rows != 0 or request.padding_budget != 0:
            raise ValueError("cold Pi0.5 action planning cannot add execution padding")
        requested = request.chunk_size or "auto"
    exact_chunk = requested if isinstance(requested, int) else None
    result_box = {}
    resolved_execution, resolved_chunk = plan_bounded_prefill_execution(
        logical_len,
        logical_len,
        0,
        execution_owner=owner,
        execution_component=PI05_ACTION_COMPONENT,
        execution_stage="action",
        execution_native=("pi05_denoise_step", int(handle)),
        position=position,
        alignment=1,
        padding_rows=0,
        exact_chunk_size=exact_chunk,
        resolve_stage_domain=lambda length: (
            torch.ops.rpu.pi05_denoise_step_resolve_action_stage_domain(
                int(handle),
                int(length),
                int(logical_len),
                int(position),
                int(kv_len),
                int(cache_capacity),
                0 if exact_chunk is None else int(exact_chunk),
                _cold_kvinsert_pad16_request(owner),
                bool(loop_mode),
                int(num_steps),
            )
        ),
        request_id="pi05:action_expert:denoise",
        graph_mode=GRAPH_COMPOSITE_CHILD,
        queue_owner_id=int(handle),
        physical_metadata=(
            ("component:action_expert", 1),
            ("execution_generation", int(getattr(
                owner, "_pi05_execution_component_generations", {}
            ).get(PI05_ACTION_COMPONENT, 0))),
            ("prefix_len", int(position)),
            ("kv_len", int(kv_len)),
            ("legacy_pad16_request", int(
                _cold_kvinsert_pad16_request(owner)
            )),
            ("loop_mode", int(loop_mode)),
            ("num_steps", int(num_steps)),
        ),
        plan_result_sink=lambda result: result_box.__setitem__("result", result),
        plan_signature=(int(kv_len), int(cache_capacity), bool(_cold_kvinsert_pad16_request(owner)), bool(loop_mode), int(num_steps)),
        graph_cache=graph_cache,
    )
    result = result_box["result"]
    selected = result.selected
    if (
        selected is None
        or resolved_execution != logical_len
        or resolved_chunk != selected.stage_tuple.compute_chunk
        or not selected.stage_tuple.physical_descriptor
    ):
        raise RuntimeError(
            "Pi0.5 denoise planner returned no consumable native descriptor"
        )
    return result


def _fixed_plan_receipt(
    plan: A6PlanResult, *, component: str, stage: str, position: int,
) -> dict[str, Any]:
    selected = plan.selected
    if selected is None:  # pragma: no cover - plan helper rejects first
        raise RuntimeError("fixed component planner returned no selection")
    receipt = plan.as_dict(include_candidates=False)
    receipt.update({
        "component": component,
        "stage": stage,
        "logical_len": selected.execution_len - selected.padding_rows,
        "execution_len": selected.execution_len,
        "chunk_size": selected.stage_tuple.compute_chunk,
        "padding_rows": selected.padding_rows,
        "position": int(position),
        "authority": (
            "NATIVE_A6_STAGE_DESCRIPTOR"
            if selected.stage_tuple.physical_descriptor
            else "FIXED_ABI_SINGLETON"
        ),
        "execution_generation": dict(
            selected.stage_tuple.physical_metadata
        )["execution_generation"],
        "physical_descriptor": {
            "stage_tuple": selected.stage_tuple.as_tuple(),
            "physical_metadata": dict(
                selected.stage_tuple.physical_metadata
            ),
        },
    })
    if component == PI05_ACTION_COMPONENT and selected.stage_tuple.physical_descriptor:
        receipt.update({
            "attention_policy": "DDR_REQUIRED",
            "attention_reason": (
                "PREFIX_HISTORY_DDR_REQUIRED_NO_RAW_RESIDENCY_ABI"
            ),
            "dry_forward_agreement": True,
        })
    return receipt


# Finding 3: per-instance handle map (from _adapter.py:66).
_PI05_HANDLES: "weakref.WeakKeyDictionary[Any, int]" = weakref.WeakKeyDictionary()
_PI05_PREPARED_GRAPH_PROFILES: "weakref.WeakKeyDictionary[Any, dict]" = (
    weakref.WeakKeyDictionary()
)

# Normalized-action implementation-parity ceilings. This internal
# [B, horizon, max_action_dim] check does not replace validation of cropped,
# denormalized actions or task rollouts. Cosine remains diagnostic.
_PI05_FUSED_PARITY_MSE_MAX = 1.0e-4
_PI05_FUSED_PARITY_ROW_MSE_MAX = 5.0e-4
_PI05_FUSED_PARITY_MAX_ABS = 5.0e-2
_PI05_FUSED_OUTPUT_SHAPE = (1, 50, 32)


def _validate_fused_parity(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    where: str = "Pi0.5 fused parity",
    expected_shape: tuple[int, ...] | None = _PI05_FUSED_OUTPUT_SHAPE,
) -> dict[str, float]:
    """Gate RPU orchestration outputs materialized as CPU float32."""
    if tuple(actual.shape) != tuple(reference.shape):
        raise RuntimeError(
            f"{where}: shape mismatch {tuple(actual.shape)} != "
            f"{tuple(reference.shape)}"
        )
    if expected_shape is not None and tuple(actual.shape) != expected_shape:
        raise RuntimeError(
            f"{where}: expected shape {expected_shape}, got "
            f"{tuple(actual.shape)}"
        )
    if actual.device.type != "cpu" or reference.device.type != "cpu":
        raise RuntimeError(
            f"{where}: device mismatch from CPU boundary "
            f"({actual.device} vs {reference.device})"
        )
    if actual.dtype != torch.float32 or reference.dtype != torch.float32:
        raise RuntimeError(
            f"{where}: dtype mismatch from torch.float32 boundary "
            f"({actual.dtype} vs {reference.dtype})"
        )

    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    actual_finite = bool(torch.isfinite(actual_fp32).all().item())
    reference_finite = bool(torch.isfinite(reference_fp32).all().item())
    if not actual_finite or not reference_finite:
        raise RuntimeError(
            f"{where}: non-finite tensor "
            f"(actual={actual_finite}, reference={reference_finite})"
        )

    diff = actual_fp32 - reference_fp32
    reference_norm = float(torch.linalg.vector_norm(reference_fp32).item())
    actual_rows = actual_fp32.reshape(-1, actual_fp32.shape[-1])
    reference_rows = reference_fp32.reshape(-1, reference_fp32.shape[-1])
    diff_rows = diff.reshape(-1, diff.shape[-1])
    row_cosine = torch.nn.functional.cosine_similarity(
        actual_rows,
        reference_rows,
        dim=-1,
    )
    actual_row_zero = torch.linalg.vector_norm(actual_rows, dim=-1) == 0
    reference_row_zero = torch.linalg.vector_norm(reference_rows, dim=-1) == 0
    both_row_zero = actual_row_zero & reference_row_zero
    single_row_zero = actual_row_zero ^ reference_row_zero
    row_cosine = torch.where(
        both_row_zero,
        torch.ones_like(row_cosine),
        row_cosine,
    )
    row_mse = torch.mean(diff_rows.square(), dim=-1)
    metrics = {
        "mse": float(torch.mean(diff.square()).item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual_fp32.flatten(),
                reference_fp32.flatten(),
                dim=0,
            ).item()
        ),
        "row_cosine_mean": float(torch.mean(row_cosine).item()),
        "row_cosine_p01": float(torch.quantile(row_cosine, 0.01).item()),
        "row_cosine_min": float(torch.min(row_cosine).item()),
        "row_both_zero_count": float(torch.sum(both_row_zero).item()),
        "row_single_zero_count": float(torch.sum(single_row_zero).item()),
        "row_mse_p99": float(torch.quantile(row_mse, 0.99).item()),
        "row_mse_max": float(torch.max(row_mse).item()),
        "relative_l2": float(torch.linalg.vector_norm(diff).item())
        / max(reference_norm, torch.finfo(torch.float32).tiny),
        "max_abs": float(torch.max(torch.abs(diff)).item()),
    }
    violations = []
    if metrics["row_single_zero_count"] > 0:
        violations.append(
            "row_single_zero_count="
            f"{metrics['row_single_zero_count']:.0f} > 0"
        )
    if metrics["mse"] > _PI05_FUSED_PARITY_MSE_MAX:
        violations.append(
            f"mse={metrics['mse']:.6e} > "
            f"{_PI05_FUSED_PARITY_MSE_MAX:.6e}"
        )
    if metrics["row_mse_max"] > _PI05_FUSED_PARITY_ROW_MSE_MAX:
        violations.append(
            f"row_mse_max={metrics['row_mse_max']:.6e} > "
            f"{_PI05_FUSED_PARITY_ROW_MSE_MAX:.6e}"
        )
    if metrics["max_abs"] > _PI05_FUSED_PARITY_MAX_ABS:
        violations.append(
            f"max_abs={metrics['max_abs']:.6e} > "
            f"{_PI05_FUSED_PARITY_MAX_ABS:.6e}"
        )
    if violations:
        raise RuntimeError(
            f"{where}: {'; '.join(violations)}; "
            f"cosine={metrics['cosine']:.8f}, "
            f"row_cosine_p01={metrics['row_cosine_p01']:.8f}, "
            f"row_cosine_min={metrics['row_cosine_min']:.8f}, "
            f"row_single_zero_count={metrics['row_single_zero_count']:.0f}, "
            f"row_mse_p99={metrics['row_mse_p99']:.6e}, "
            f"row_mse_max={metrics['row_mse_max']:.6e}, "
            f"relative_l2={metrics['relative_l2']:.6e}, "
            f"mse={metrics['mse']:.6e}, max_abs={metrics['max_abs']:.6e}"
        )
    return metrics


def _denoise_graph_enabled() -> bool:
    """Parse the legacy selector for a new Pi0.5 component installation."""
    return rpu_env_bool("RPU_PI05_DENOISE_GRAPH", default=True)


def _euler_fp16_enabled() -> bool:
    """A/B probe for the Diff-B fix. When set, round the Euler accumulator to
    fp16 each step — models an on-device fp16 Euler (e.g. porting the Wall-OSS
    in-graph denoise unroll). Default off = CPU fp32 accumulator (Diff-B fix).
    Read per-call so a single process can A/B both precisions."""
    return rpu_env_bool("RPU_PI05_EULER_FP16")


def _denoise_unroll_enabled() -> bool:
    """Parse the legacy unroll selector for a new component installation."""
    return rpu_env_bool("RPU_PI05_DENOISE_UNROLL")


def _prefix_pad16_enabled() -> bool:
    """Pad the prefix up to a multiple of 16 (see _pad_prefix_to_16). Default on;
    it is a no-op whenever the prefix is already aligned, which every shipped
    profile is (3*256 + a 32-token prompt)."""
    return rpu_env_bool("RPU_PI05_PREFIX_PAD16", default=True)


def _kvinsert_pad16_enabled() -> bool:
    """Legacy cold-input parser retained for checked-in delivery profiles."""
    return rpu_env_bool(
        "RPU_PI05_KVINSERT_PAD16",
    )


def _cold_kvinsert_pad16_request(owner) -> bool:
    """Read the legacy knob frozen by Pi05Adapter before installation."""
    name = "_rpu_legacy_kvinsert_pad16_request"
    if not hasattr(owner, name):
        raise RuntimeError("Pi0.5 KV-insert PAD16 cold snapshot is missing")
    return bool(getattr(owner, name))


def _cold_prefix_pad16_request(owner) -> bool:
    """Read the legacy prefix policy frozen before any weight mutation."""
    name = "_rpu_legacy_prefix_pad16_request"
    if not hasattr(owner, name):
        raise RuntimeError("Pi0.5 prefix PAD16 cold snapshot is missing")
    return bool(getattr(owner, name))


def _pi05_graph_runtime_profile(self, num_steps: int) -> dict:
    """Return the runtime knobs that determine Pi0.5 graph admission."""
    pawe = self.paligemma_with_expert
    vlm = pawe.paligemma.model.language_model
    expert = pawe.gemma_expert.model
    vision_tower = pawe.paligemma.model.vision_tower
    vision = getattr(vision_tower, "vision_model", vision_tower)
    execution = getattr(self, "_rpu_execution", {})
    return {
        "num_steps": int(num_steps),
        "vlm_chunk_size": int(getattr(vlm, "_rpu_chunk_size", 0)),
        "siglip_batch_n": int(getattr(self, "_rpu_siglip_batch_n", 1)),
        "siglip_graph": bool(vision._siglip_graph_enabled),
        "gemma_graph": bool(vlm._gemma_graph_enabled),
        "adarms_graph": bool(expert._adarms_graph_capture_enabled),
        "fused_denoise": getattr(
            self, "_rpu_fused_denoise_handle", None
        ) is not None,
        "denoise_graph": bool(self._pi05_denoise_graph_enabled),
        "denoise_unroll": bool(self._pi05_denoise_unroll_enabled),
        "kvinsert_pad16": _cold_kvinsert_pad16_request(self),
        "prefix_pad16": _cold_prefix_pad16_request(self),
        "execution": tuple(
            (stage, tuple(sorted(fields.items())))
            for stage, fields in sorted(execution.items())
        ),
    }


def _validate_prepared_graph_profile(self, num_steps: int) -> None:
    """Reject runtime-profile drift before any frozen GraphCache is entered."""
    prepared = _PI05_PREPARED_GRAPH_PROFILES.get(self)
    if prepared is None:
        return
    expected = prepared["runtime"]
    actual = _pi05_graph_runtime_profile(self, num_steps)
    changed = {
        key: (expected[key], actual[key])
        for key in expected
        if expected[key] != actual[key]
    }
    if changed:
        raise RuntimeError(
            "Pi0.5 request/runtime differs from the prepared READY profile "
            f"(expected, actual): {changed}. Online graph BUILD is disabled."
        )


# CR-R5 BLOCKER 9: allowlist of permitted `**kwargs` to the RPU-path sample_actions
# (from _adapter.py:480). Unknown kwargs raise instead of being silently dropped.
_RPU_ALLOWED_SAMPLE_ACTIONS_KWARGS = {"num_steps"}


# ===========================================================================
# B3·A·2 (2026-05-27): prefix 4D attention mask LRU-1 cache + helpers.
# Spec §5.2 + Rev 4 corrections C3/C4/C5/C6/R3-* / R4-3.
# ===========================================================================

# Atomic tuple snapshot — readers grab `snap = _PREFIX_MASK_SNAPSHOT`
# locally before comparing key. Writers serialize via the lock.
_PREFIX_MASK_SNAPSHOT: "tuple | None" = None  # (key, mask4d, pos_ids) or None
_PREFIX_MASK_LOCK = threading.Lock()
_PREFIX_MASK_CACHE_HITS: int = 0
_PREFIX_MASK_CACHE_MISSES: int = 0

# Rev 4 R4-3: once-per-device sentinel to avoid spam under
# warnings.simplefilter("always").
_HASH_MASK_NONCPU_WARNED: set = set()


def _prefix_mask_cache_enabled() -> bool:
    """Runtime env check (per-call, NOT module const).

    Keep the diagnostic opt-out live so callers can disable the cache without
    constructing a new policy.
    """
    return rpu_env_bool("RPU_PI05_PREFIX_MASK_CACHE", default=True)


def _hash_mask(t: "torch.Tensor") -> bytes:
    """Hash a mask tensor to bytes for cache-key construction.

    The production path keeps prefix masks on CPU. A device mask requires a
    CPU readback for the content key; warn once per device when that occurs.
    """
    if t.device.type != 'cpu' and str(t.device) not in _HASH_MASK_NONCPU_WARNED:
        import warnings
        warnings.warn(
            f"_hash_mask: tensor on {t.device}, .cpu() sync per cache lookup",
            RuntimeWarning, stacklevel=2)
        _HASH_MASK_NONCPU_WARNED.add(str(t.device))
    return t.contiguous().cpu().numpy().tobytes()


def _make_cache_key(prefix_pad_masks, prefix_att_masks):
    """8-field key (4 per mask × 2 masks) — Rev 3 C5a.

    Includes shape/dtype/device + content bytes for BOTH masks to avoid
    collisions across episodes / dtype changes / device migrations.
    """
    return (
        # pad
        tuple(prefix_pad_masks.shape),
        prefix_pad_masks.dtype,
        str(prefix_pad_masks.device),
        _hash_mask(prefix_pad_masks),
        # att
        tuple(prefix_att_masks.shape),
        prefix_att_masks.dtype,
        str(prefix_att_masks.device),
        _hash_mask(prefix_att_masks),
    )


def _cached_build_prefix_mask_4d(
    prefix_pad_masks, prefix_att_masks, prepare_4d_fn, *, output_dtype=None,
):
    """Build (mask_4d, position_ids) with LRU-1 cache + fast-path guard.

    Spec §5.2 Rev 4:
    - C3: helper accepts prefix_att_masks; only fast-paths when att is truly
      all-zero. Otherwise calls upstream make_att_2d_masks.
    - C4: atomic tuple snapshot for thread safety (read is GIL-atomic; write
      via lock).
    - C6: env opt-out runtime per-call.
    - Optional output dtype is part of the cache key. Conversion preserves
      Gemma's original to(dtype).cpu().contiguous() order on a miss or opt-out.
    """
    global _PREFIX_MASK_CACHE_HITS, _PREFIX_MASK_CACHE_MISSES, _PREFIX_MASK_SNAPSHOT

    if not _prefix_mask_cache_enabled():
        # Honest passthrough — bit-identical to lerobot upstream behavior.
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        mask4d = prepare_4d_fn(att_2d)
        if output_dtype is not None:
            mask4d = mask4d.to(dtype=output_dtype).cpu().contiguous()
        return (mask4d, torch.cumsum(prefix_pad_masks, dim=1) - 1)

    key = _make_cache_key(prefix_pad_masks, prefix_att_masks) + (output_dtype,)
    snap = _PREFIX_MASK_SNAPSHOT  # GIL-atomic read of tuple/None
    if snap is not None and snap[0] == key:
        _PREFIX_MASK_CACHE_HITS += 1
        return snap[1], snap[2]

    _PREFIX_MASK_CACHE_MISSES += 1
    # Fast-path guard: att must be truly all-zero for outer-product shortcut.
    if torch.count_nonzero(prefix_att_masks) == 0:
        att_2d = prefix_pad_masks[:, None, :] * prefix_pad_masks[:, :, None]
    else:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    mask4d = prepare_4d_fn(att_2d)
    if output_dtype is not None:
        mask4d = mask4d.to(dtype=output_dtype).cpu().contiguous()
    pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    with _PREFIX_MASK_LOCK:
        _PREFIX_MASK_SNAPSHOT = (key, mask4d, pos_ids)  # atomic re-assign

    return mask4d, pos_ids


def _reset_prefix_mask_cache():
    """Reset cache state — test fixture helper (Rev 3 C7)."""
    global _PREFIX_MASK_SNAPSHOT, _PREFIX_MASK_CACHE_HITS, _PREFIX_MASK_CACHE_MISSES
    with _PREFIX_MASK_LOCK:
        _PREFIX_MASK_SNAPSHOT = None
        _PREFIX_MASK_CACHE_HITS = 0
        _PREFIX_MASK_CACHE_MISSES = 0


def _load_pi05_probe_tensor(
    path: str,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    """Load one local debug override without enabling arbitrary pickle."""
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{name} override must contain a tensor, got "
            f"{type(value).__name__}"
        )
    shape = tuple(value.shape)
    if shape != expected_shape:
        raise ValueError(
            f"{name} override must have shape {expected_shape}, got {shape}"
        )
    if value.is_complex():
        raise TypeError(f"{name} override must be real-valued")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} override must contain only finite values")
    return value


# =============================================================================
# Handle destroyer (from _adapter.py:754-766).
# =============================================================================
def _pi05_destroy_handle_and_unmap(handle: int, model_ref: Any) -> None:
    """Finding 3 finalizer: pop the WeakKeyDictionary entry + destroy C++ handle."""
    _PI05_HANDLES.pop(model_ref, None)
    _PI05_PREPARED_GRAPH_PROFILES.pop(model_ref, None)
    try:
        torch.ops.rpu.pi05_destroy(handle)
    except Exception:
        pass   # Swallow — interpreter shutdown may have destroyed the op.


# =============================================================================
# Top-level helpers (lifted from install_sample_actions_patch nested closures).
# These had no meaningful captures — they only call other module-scope helpers.
# =============================================================================

def _resolve_rpu_device(self):
    """Helper 2: resolve RPU device from VLM layer 0 (from _adapter.py:565-586)."""
    cached = getattr(self, "_rpu_runtime_device", None)
    if cached is not None:
        return cached

    vlm = self.paligemma_with_expert.paligemma.model.language_model
    # Iteration 6: read device from VLM's first decoder layer (not
    # next(vlm.parameters()) which may return embed_tokens — that stays
    # on CPU by design; see Step D's `vlm.embed_tokens.to('cpu')`).
    rpu_device = next(vlm.layers[0].parameters()).device
    if rpu_device.type != 'rpu':
        raise RPUBackendError(
            f"Pi05 RPU path: vlm is not on RPU (device={rpu_device}). "
            "adapter.to_rpu() Step D failed silently. Reload the policy.")
    return rpu_device


def _sample_noise(self, bsize, noise):
    """Helper 3: sample / override noise (from _adapter.py:588-607)."""
    actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
    if noise is None:
        seed = int(getattr(self, '_rpu_noise_seed', 0))
        gen = torch.Generator(device='cpu').manual_seed(seed)
        x_t = torch.randn(actions_shape, generator=gen, dtype=torch.float32, device='cpu')
    else:
        # Caller-provided noise: move to CPU fp32 for the Python loop.
        x_t = noise.to(device='cpu', dtype=torch.float32)

    # Probe: bisection harness may override noise via env var to force
    # identical inputs between legacy and library runs.
    _override = os.environ.get("RPU_PI05_LOAD_NOISE")
    if _override and os.path.exists(_override):
        x_t = _load_pi05_probe_tensor(
            _override,
            name="RPU_PI05_LOAD_NOISE",
            expected_shape=actions_shape,
        ).to(dtype=torch.float32)
    return x_t


def _prepare_prefill_siglip_inputs(
    self, images, img_masks, rpu_device
):
    """Upload SigLIP inputs without splitting a marked multi-camera slab.

    The batch-preprocessing path returns CPU dim-0 views of one camera slab.
    Keep those views intact until `_siglip_forward_images`, which performs one
    packed upload. Dynamic/reference inputs retain the established per-image
    upload. Masks stay on CPU: they only feed prefix-mask construction, and the
    fused Gemma path normalizes its attention mask on CPU.
    """
    if images is None:
        return images, img_masks

    batch_n = getattr(self, "_rpu_siglip_batch_n", 1)
    from rpu_backend.adapters.siglip import _shared_siglip_image_slab

    shared_batch = (
        batch_n > 1
        and len(images) == batch_n
        and _shared_siglip_image_slab(images) is not None
    )
    if not shared_batch:
        images = [image.to(rpu_device) for image in images]
    return images, img_masks



def _pad_prefix_rows(
    prefix_embs, prefix_pad_masks, prefix_att_masks, padding_rows: int
):
    """Append an exact number of masked, zero-valued physical prefix rows."""
    padding_rows = int(padding_rows)
    if padding_rows < 0:
        raise ValueError(f"padding_rows must be non-negative, got {padding_rows}")
    if padding_rows == 0:
        return prefix_embs, prefix_pad_masks, prefix_att_masks
    b = prefix_pad_masks.shape[0]
    embs = torch.cat([
        prefix_embs,
        prefix_embs.new_zeros(b, padding_rows, prefix_embs.shape[2]),
    ], dim=1)
    pads = torch.cat([
        prefix_pad_masks,
        prefix_pad_masks.new_zeros(b, padding_rows),
    ], dim=1)
    atts = torch.cat([
        prefix_att_masks,
        prefix_att_masks.new_zeros(b, padding_rows),
    ], dim=1)
    return embs, pads, atts


def _pad_prefix_to_16(prefix_embs, prefix_pad_masks, prefix_att_masks):
    """Round the prefix up to a multiple of 16 with TRAILING pad rows.

    KV-insert v16 requires a 16-aligned insertion position, which equals the
    physical prefix length. Trailing rows are masked from prefix and suffix
    attention. RoPE uses logical positions while KV storage uses physical
    rows, so padding does not change the positions of valid tokens.
    """
    n = int(prefix_pad_masks.shape[1])
    pad = (-n) % 16
    return _pad_prefix_rows(
        prefix_embs, prefix_pad_masks, prefix_att_masks, pad
    )


def _plan_pi05_prefix_execution(self, logical_len: int, *, _cost_request=None,
                               _validation_request=None):
    """Jointly choose Pi0.5's physical prefix length and Gemma chunk size.

    The shared planner owns SPM/kernel feasibility and the fixed A6 score.
    Pi0.5's default 16-aligned physical prefix remains an explicit capability
    choice; it is not a private comparator term. Padding remains trailing and
    masked, and the RoPE index/cache-row split keeps it semantics-preserving.
    """
    vlm = self.paligemma_with_expert.paligemma.model.language_model
    handle = vlm._rpu_vlm_decoder_handle
    requested_chunk = int(getattr(vlm, "_rpu_chunk_size", 0))
    component_views = getattr(self, "_pi05_execution_components", None)
    if component_views is None:
        prefill = getattr(self, "_rpu_execution", {}).get("prefill", {})
    else:
        prefill = component_views.get("language_model", {}).get("prefill", {})
    if _validation_request is not None:
        if _cost_request is not None:
            raise ValueError("Pi0.5 cannot mix hot validation and cold cost inspection")
        prefill = _validation_request
        requested_chunk = prefill.get("chunk_size", "auto")
        requested_chunk = 0 if requested_chunk == "auto" else requested_chunk
    if _cost_request is not None:
        from rpu_backend.runtime.decoder import _cold_text_cost_request

        request, _ = _cold_text_cost_request(
            self, ("gemma", handle), vlm._rpu_cache, _cost_request, 0,
            component=PI05_TEXT_COMPONENT)
        requested_chunk = request.chunk_size or 0
        prefill = {"chunk_size": requested_chunk or "auto",
                   "padding_rows": "auto" if request.padding_rows is None else request.padding_rows,
                   "padding_budget": request.padding_budget}
    prefix_pad16 = _cold_prefix_pad16_request(self)
    explicit_padding_rows = type(prefill.get("padding_rows")) is int
    has_padding_policy = bool(requested_chunk) or any(
        field in prefill for field in ("padding_rows", "padding_budget")
    )

    if has_padding_policy:
        padding_rows = prefill.get("padding_rows", "auto")
        padding_budget = int(prefill.get("padding_budget", 15))
    else:
        # Preserve the existing default exactly, but send it through the same
        # planner/receipt path as caller-supplied controls.
        padding_rows = (-logical_len) % 16 if prefix_pad16 else 0
        padding_budget = 0
    # Leave the configured action horizon addressable after the physical prefix.
    physical_limit = (
        int(getattr(vlm, "_rpu_max_seq_len", 2048))
        - int(self.config.chunk_size)
    )

    plan_box = {}
    component_generation = int(getattr(
        self, "_pi05_execution_component_generations", {}
    ).get("language_model", 0))
    execution_len, chunk_size = plan_bounded_prefill_execution(
        logical_len,
        physical_limit,
        padding_budget,
        execution_owner=self if _validation_request is None else None,
        execution_component=PI05_TEXT_COMPONENT,
        execution_stage="prefill",
        execution_native=("gemma", int(handle)) if _validation_request is None else None,
        # KV-insert v16 is a route feasibility constraint, not a comparator
        # preference. The explicit A/B switch removes that capability.
        alignment=1 if explicit_padding_rows else (16 if prefix_pad16 else 1),
        padding_rows=padding_rows,
        exact_chunk_size=(requested_chunk or None),
        resolve_stage_domain=lambda execution_len: (
            torch.ops.rpu.gemma_resolve_prefill_stage_domain(
                handle, int(execution_len), requested_chunk, int(logical_len)
            )
        ),
        request_id="pi05:language_model:prefill",
        plan_result_sink=lambda result: plan_box.__setitem__("result", result),
        graph_mode=GRAPH_COMPOSITE_CHILD,
        queue_owner_id=int(handle),
        physical_metadata=(
            ("component:language_model", 1),
            ("execution_generation", component_generation),
        ),
        plan_signature=(),
        graph_cache=vlm._rpu_gemma_graph_cache if _validation_request is None else None,
    )
    return execution_len, chunk_size, plan_box["result"]


def _prefill_prefix_embs(self, images, img_masks, tokens, masks, rpu_device):
    """Helper 4: VLM prefill — embed_prefix -> paligemma_with_expert.forward(use_cache=True)
    (from _adapter.py:609-657).
    """
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    # Iteration 6: VLM's embed_tokens now lives on CPU (matches legacy
    # pi05_converter.py:1230-1231), so tokens MUST stay on CPU; they'd
    # otherwise hit a dtype/device mismatch inside torch.embedding. masks
    # stay on CPU for lerobot mask ops. Images go to RPU only when SigLIP
    # is on RPU — lerobot's patched `embed_image` then passes them to the
    # RPU vision_tower.
    _siglip_param = next(
        self.paligemma_with_expert.paligemma.model.vision_tower.parameters(),
        None,
    )
    _siglip_on_rpu = (
        _siglip_param is not None and _siglip_param.device.type == 'rpu'
    )
    if _siglip_on_rpu and images is not None:
        images, img_masks = _prepare_prefill_siglip_inputs(
            self, images, img_masks, rpu_device
        )

    # --- VLM prefill (matches lerobot sample_actions:812-825) ---
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, tokens, masks)
    if images is not None:
        vision_tower = (
            self.paligemma_with_expert.paligemma.model.vision_tower
        )
        vision_inner = getattr(vision_tower, "vision_model", vision_tower)
        vision_handle = getattr(vision_inner, "_rpu_vision_handle", None)
        if vision_handle is not None:
            vision_plan = getattr(
                vision_inner, "_fmb_last_execution_plan", None
            )
            if vision_plan is None:
                raise RuntimeError(
                    "Pi0.5 Vision forward did not publish its fixed component plan"
                )
            selected = vision_plan.selected
            if selected is None:
                raise RuntimeError("Pi0.5 Vision fixed component plan is empty")
            resolved = int(
                torch.ops.rpu.siglip_get_resolved_chunk_size(vision_handle)
            )
            if resolved != selected.stage_tuple.compute_chunk:
                raise RuntimeError(
                    "Pi0.5 Vision dry/forward chunk mismatch: "
                    f"planned={selected.stage_tuple.compute_chunk}, "
                    f"resolved={resolved}"
                )
            plans = vars(self).setdefault("_rpu_last_execution_plan", {})
            plans["vision"] = _fixed_plan_receipt(
                vision_plan,
                component=PI05_VISION_COMPONENT,
                stage="vision",
                position=0,
            )
    # Debug: optionally equalize prefix_embs to legacy's probe
    _pe_override = os.environ.get("RPU_PI05_LOAD_PREFIX_EMBS")
    if _pe_override and os.path.exists(_pe_override):
        prefix_embs = _load_pi05_probe_tensor(
            _pe_override,
            name="RPU_PI05_LOAD_PREFIX_EMBS",
            expected_shape=tuple(prefix_embs.shape),
        ).to(dtype=prefix_embs.dtype, device=prefix_embs.device)
    logical_prefix_len = int(prefix_pad_masks.shape[1])
    (
        execution_prefix_len,
        planned_chunk_size,
        a6_plan,
    ) = _plan_pi05_prefix_execution(self, logical_prefix_len)
    prefix_embs, prefix_pad_masks, prefix_att_masks = _pad_prefix_rows(
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        execution_prefix_len - logical_prefix_len,
    )
    # B3·A·2 (2026-05-27): cache hit returns last (mask4d, pos_ids) verbatim;
    # cache miss with all-zero att uses pad-outer-product fast-path; otherwise
    # calls upstream make_att_2d_masks. Spec §5.2.
    prefix_att_2d_masks_4d, prefix_position_ids = _cached_build_prefix_mask_4d(
        prefix_pad_masks, prefix_att_masks, self._prepare_attention_masks_4d,
        output_dtype=(torch.float16 if getattr(
            self, "_pi05_prefill_mask_fp16", False
        ) else None),
    )
    self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    vlm = self.paligemma_with_expert.paligemma.model.language_model
    vlm._rpu_planned_prefill_plan = a6_plan
    try:
        kv_only_entry = getattr(vlm, "_rpu_prefill_kv_only", None)
        if kv_only_entry is not None:
            past_key_values = kv_only_entry(
                inputs_embeds=prefix_embs,
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
            )
        else:
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
    finally:
        vars(vlm).pop("_rpu_planned_prefill_plan", None)
    resolved_chunk_size = int(torch.ops.rpu.gemma_get_resolved_chunk_size(
        vlm._rpu_vlm_decoder_handle
    ))
    if resolved_chunk_size != planned_chunk_size:
        raise RuntimeError(
            "Pi0.5 prefill dry/forward chunk mismatch: "
            f"planned={planned_chunk_size}, resolved={resolved_chunk_size}, "
            f"logical_len={logical_prefix_len}, "
            f"execution_len={execution_prefix_len}"
        )
    plans = vars(self).setdefault("_rpu_last_execution_plan", {})
    plans["prefill"] = {
        "component": PI05_TEXT_COMPONENT,
        "stage": "prefill",
        "logical_len": logical_prefix_len,
        "execution_len": execution_prefix_len,
        "chunk_size": resolved_chunk_size,
        "padding_rows": execution_prefix_len - logical_prefix_len,
        "position": 0,
        "authority": "NATIVE_A6_STAGE_DESCRIPTOR",
        "physical_descriptor": tuple(
            a6_plan.selected.stage_tuple.physical_descriptor
        ),
        "physical_plan_digest": a6_plan.physical_plan_digest,
        "plan_digest": a6_plan.plan_digest,
        "execution_generation": int(getattr(
            self, "_pi05_execution_component_generations", {}
        ).get(PI05_TEXT_COMPONENT, 0)),
    }
    plans["prefill"].update({
        "a6_plan_digest": a6_plan.plan_digest,
        "a6_domain_digest": a6_plan.domain_digest,
        "a6_optimality": a6_plan.optimality,
        "a6_search_complete": a6_plan.search_complete,
    })
    prefix_len = int(prefix_pad_masks.shape[1])
    past_key_values._prefix_len = prefix_len
    return past_key_values, prefix_pad_masks, prefix_len


def _apply_kv_cache(self, past_key_values):
    """Helper 5: apply shared KV cache to expert (from _adapter.py:660-665)."""
    # Gap G3 (iter-3 closure): zero-copy shared cache. Set expert._rpu_cache
    # to the VLM's RPUCache so `rpu_adarms_model_forward` sees it during the
    # denoise loop and skips its own RPUCache auto-creation.
    expert = self.paligemma_with_expert.gemma_expert.model
    expert._rpu_cache = past_key_values
    return expert


# =============================================================================
# New top-level helpers (Task 4.3 additions)
# =============================================================================

def _build_attention_mask_4d(self, prefix_pad_masks, prefix_len, bsize,
                              chunk_size, dtype, device):
    """Build the [B, 1, S, S] 4D attention mask used by the denoise step.

    Spec §2.3 — prefix_len is constant across calls, so this mask is invariant
    across denoise steps in one sample_actions call. Precompute it once.
    """
    cache_enabled = hasattr(self, "_rpu_denoise_mask_cache")
    cache_key = None
    if cache_enabled:
        cache_key = (
            tuple(prefix_pad_masks.shape),
            str(prefix_pad_masks.dtype),
            str(prefix_pad_masks.device),
            _hash_mask(prefix_pad_masks),
            int(prefix_len),
            int(bsize),
            int(chunk_size),
            str(dtype),
            str(device),
        )
        snapshot = self._rpu_denoise_mask_cache
        if snapshot is not None and snapshot[0] == cache_key:
            return snapshot[1]

    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    suffix_pad_masks = torch.ones(bsize, chunk_size, dtype=torch.bool, device=device)
    suffix_att_masks = torch.tensor(
        [1] + [0] * (chunk_size - 1), dtype=dtype, device=device,
    )[None, :].expand(bsize, chunk_size)
    suffix_len = suffix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
        bsize, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
    if cache_enabled:
        full_att_2d_masks_4d = versioned_cpu_mask(full_att_2d_masks_4d)
        self._rpu_denoise_mask_cache = (cache_key, full_att_2d_masks_4d)
    return full_att_2d_masks_4d


def _kv_prefix_full_hash(past_key_values, prefix_len: int) -> bytes:
    """Spec §6.3 — blake2b over all-layer K/V prefix slices.

    Walks the 7-D swizzled cache layout (python/rpu_backend/api/cache.py:45-108):
      K[batch, sKeyVx, nKVHeadVx, headDimVx, nKVHeadChunk, sKeyChunk, headDimChunk]
      V[batch, sValVx, nKVHeadVx, headDimVx, nKVHeadChunk, headDimChunk, sValChunk]

    For prefix of length P with sKeyChunk = sValChunk = 16:
      - vx_full = P // 16
      - tail    = P % 16
      Full vx-blocks: cache[:, :vx_full, ..., :, :]  → contiguous
      Tail slice:     cache[:, vx_full:vx_full+1, ..., :, :tail] (K) /
                                                       :, :, :tail (V)
    """
    import hashlib
    h = hashlib.blake2b()
    s_chunk_k = past_key_values.sKeyChunk
    s_chunk_v = past_key_values.sValChunk
    for kc in past_key_values.k_caches:
        vx_full = prefix_len // s_chunk_k
        tail    = prefix_len %  s_chunk_k
        if vx_full > 0:
            h.update(kc[:, :vx_full].contiguous().cpu().numpy().tobytes())
        if tail > 0:
            h.update(kc[:, vx_full:vx_full + 1, :, :, :, :tail, :]
                       .contiguous().cpu().numpy().tobytes())
    for vc in past_key_values.v_caches:
        vx_full = prefix_len // s_chunk_v
        tail    = prefix_len %  s_chunk_v
        if vx_full > 0:
            h.update(vc[:, :vx_full].contiguous().cpu().numpy().tobytes())
        if tail > 0:
            # V last dim is sValChunk (dim 6), not dim 5 as in K.
            h.update(vc[:, vx_full:vx_full + 1, :, :, :, :, :tail]
                       .contiguous().cpu().numpy().tobytes())
    return h.digest()


def _run_denoise_python_baseline(self, x_t, prefix_pad_masks, past_key_values,
                                  num_steps, prefix_len, bsize):
    """Legacy denoise loop — CPU/RPU AdaRMS-per-step path (from _adapter.py:666-748).

    Preserves all existing behavior verbatim including Diff-B fix (CPU fp32
    Euler accumulator) and PERF E3/E4/E5 hoisting.
    """
    from lerobot.policies.pi05.modeling_pi05 import (
        make_att_2d_masks, create_sinusoidal_pos_embedding)
    import torch.nn.functional as F

    dt = -1.0 / num_steps
    cfg = self.config
    chunk_size = cfg.chunk_size
    device = x_t.device  # CPU for Pi0.5 denoise

    # PERF E3-B1 (2026-05-07): _attn_implementation is invariant — set
    # ONCE before the loop instead of every step.
    self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
    # PERF E3-B2: take_len = min(chunk_size, suffix_out.shape[1]); the
    # Action Expert always returns chunk_size tokens, so take_len ==
    # chunk_size (invariant, no per-step computation).
    take_len = chunk_size

    # PERF E5 (2026-05-07): precompute sinusoidal pos embeddings for all
    # `num_steps` denoise time values. The 5 time values are deterministic
    # (1.0 + step * dt). Building the LUT once replaces num_steps× calls
    # to create_sinusoidal_pos_embedding (linspace + outer-product +
    # sin/cos on dim=action_in_proj.out_features).
    sin_dim = self.action_in_proj.out_features
    sin_lut: list[torch.Tensor] = []
    for s in range(num_steps):
        tv = 1.0 + s * dt
        t_step = torch.tensor(tv, dtype=torch.float32, device=device).expand(bsize)
        emb = create_sinusoidal_pos_embedding(
            t_step, sin_dim, cfg.min_period, cfg.max_period, device=device)
        sin_lut.append(emb.type(t_step.dtype))

    # PERF E4 (2026-05-07): suffix_pad_masks and suffix_att_masks from
    # embed_suffix are shape-only (modeling_pi05.py:717 torch.ones,
    # :721 [1]+[0]*(chunk-1)) — bit-identical across all denoise steps.
    # Build them ONCE outside the loop. The dtype of suffix_att_masks
    # matches embs.dtype (fp16 after action_in_proj's post-hook cast);
    # we materialize it lazily at step 0 once we know action_emb.dtype.
    suffix_pad_masks = torch.ones(
        bsize, chunk_size, dtype=torch.bool, device=device)
    suffix_att_masks: torch.Tensor | None = None

    # L2 (2026-05-07): mask construction also hoisted to step 0; the
    # SAME Python tensor flows into all 5 forwards so the C++ mask cache
    # (keyed on data_ptr) hits on steps 1-4. See PERF L2 in adarms.py +
    # rpu_adarms_model.cpp.
    full_att_2d_masks_4d: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None

    for step in range(num_steps):
        # Gap G3: reset shared cache position to prefix_len before each
        # step. AdaRMS will write suffix K/V at position..position+suffix_len;
        # this reset ensures prefix KV at [0, prefix_len) stays intact across
        # denoise steps. Matches legacy pi05_converter.py:318.
        past_key_values.reset_to_position(prefix_len)

        # PERF E4+E5: inline `embed_suffix` using the precomputed sinusoidal
        # LUT (E5) and shape-only mask cache (E4). `_apply_checkpoint` is
        # a no-op in eval (gradient_checkpointing_enabled=False), so direct
        # module calls are bit-identical to the original embed_suffix path.
        # action_in_proj / time_mlp_* hooks (fp32→fp16 cast on input,
        # fp32→fp16 on output) still fire — they live on the modules.
        action_emb = self.action_in_proj(x_t)        # fp16 via hooks
        time_emb_pre = sin_lut[step]                 # fp32 (from LUT)
        x = self.time_mlp_in(time_emb_pre)           # fp16 via hooks
        x = F.silu(x)
        x = self.time_mlp_out(x)                     # fp16
        time_emb = F.silu(x)                         # fp16
        suffix_embs = action_emb
        adarms_cond = time_emb
        if suffix_att_masks is None:
            # First step only: materialize using action_emb's dtype.
            # att_masks dtype = embs.dtype per modeling_pi05.py:725.
            suffix_att_masks = torch.tensor(
                [1] + [0] * (chunk_size - 1),
                dtype=suffix_embs.dtype, device=suffix_embs.device,
            )[None, :].expand(bsize, chunk_size)

        # L2 mask construction once at step 0 (data_ptr stable across
        # 5 forwards → C++ AdaRMSModel mask cache hits on steps 1-4).
        if step == 0:
            suffix_len = suffix_pad_masks.shape[1]
            # Matches legacy _patched_denoise_step:345-356
            # and lerobot denoise_step:876-886.
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                bsize, suffix_len, prefix_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        # Expert forward via paligemma_with_expert — routes through
        # `rpu_adarms_model_forward` (patched in Step E). That function:
        #   - auto-routes CPU inputs to RPU
        #   - invokes torch.ops.rpu.adarms_forward (fused AdaRMS C++ op)
        #   - applies final PiGemmaRMSNorm(output, adarms_cond)  ← Diff-A fix
        #   - returns `BaseModelOutputWithPast(last_hidden_state=...)` on CPU
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]

        # Match legacy semantics (pi05_converter.py:369-372):
        #   - take last chunk_size tokens (take_len precomputed above)
        #   - cast to fp32 (Diff-C fix — action_out_proj runs fp32 via hook)
        #   - action_out_proj on CPU fp32 -> v_t is fp16 (post-hook cast)
        suffix_out = suffix_out[:, -take_len:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        # Euler integrator (Diff-B fix): Python type promotion upgrades
        # `x_t + dt * v_t` to fp32 because x_t is fp32 and (dt*v_t) is fp32
        # (Python float * fp16 -> fp32).
        x_t = x_t + dt * v_t

    action_plan = getattr(
        self.paligemma_with_expert.gemma_expert.model,
        "_fmb_last_execution_plan",
        None,
    )
    if action_plan is None:
        raise RuntimeError(
            "Pi0.5 AdaRMS forward did not publish its fixed component plan"
        )
    plans = vars(self).setdefault("_rpu_last_execution_plan", {})
    plans["action"] = _fixed_plan_receipt(
        action_plan,
        component=PI05_ACTION_COMPONENT,
        stage="action",
        position=int(prefix_len),
    )
    return x_t


def _run_denoise_fused(self, x_t_cpu_fp32, prefix_pad_masks, past_key_values,
                        num_steps, prefix_len, bsize):
    """Spec §2.4 fused denoise loop.

    Mirrors _run_denoise_python_baseline at the boundary (x_t CPU fp32 in/out);
    inside the loop x_t lives RPU fp16, Euler accumulator stays CPU fp32
    (Diff-B fix preserved per spec §10).
    """
    import torch.nn.functional as F
    from rpu_backend.adapters.pi05.weights import _precompute_adarms_cond_all
    from rpu_backend.graph import GraphSignature

    handle = getattr(self, "_rpu_fused_denoise_handle", None)
    if handle is None:
        raise RuntimeError("_run_denoise_fused requires fused handle")
    cache = getattr(self, "_rpu_fused_denoise_graph_cache", None)
    if cache is None:
        raise RuntimeError("_run_denoise_fused requires fused graph cache")
    use_graph = bool(self._pi05_denoise_graph_enabled)

    # In-graph unroll: ONE pi05_denoise_loop_forward (on-device fp16 Euler)
    # replaces the host per-step loop. Requires graph mode (the loop runs
    # 1 BUILD + 1 REPLAY inside cache.capture).
    if self._pi05_denoise_unroll_enabled and use_graph:
        return _run_denoise_loop(self, x_t_cpu_fp32, prefix_pad_masks,
                                 past_key_values, num_steps, prefix_len, bsize)

    cfg = self.config
    cs   = cfg.chunk_size
    mad  = cfg.max_action_dim
    dt   = -1.0 / num_steps

    # Bound check (echoes the C++ TORCH_CHECK; fail fast in Python).
    k0 = past_key_values.k_caches[0]
    cache_max_seq = k0.size(1) * k0.size(5)  # sKeyVx * sKeyChunk
    if prefix_len + cs > cache_max_seq:
        raise RuntimeError(
            f"_run_denoise_fused: prefix_len({prefix_len}) + chunk_size({cs}) "
            f"exceeds k_cache.max_seq_len={cache_max_seq}.")

    adarms_cond_all_rpu = _precompute_adarms_cond_all(self, num_steps)

    x_t_rpu = torch.empty(bsize, cs, mad, dtype=torch.float16, device='rpu')
    v_t_buf = torch.empty(bsize, cs, mad, dtype=torch.float16, device='rpu')
    x_t_rpu.copy_(x_t_cpu_fp32.to(torch.float16))

    mask_4d = _build_attention_mask_4d(
        self, prefix_pad_masks, prefix_len, bsize, cs,
        dtype=torch.float16, device='rpu')

    expert_model = self.paligemma_with_expert.gemma_expert.model
    ec = expert_model.config
    effective_num_kv_heads = int(getattr(
        expert_model, "_rpu_effective_num_kv_heads", ec.num_key_value_heads
    ))
    rope_pos = _set_denoise_rope_position(handle, prefix_pad_masks)
    action_plan = _plan_pi05_denoise_action_execution(
        self,
        handle=int(handle),
        logical_len=int(cs),
        position=int(prefix_len),
        kv_len=int(prefix_len + cs),
        cache_capacity=int(cache_max_seq),
        loop_mode=False,
        num_steps=1,
        graph_cache=cache,
    )
    sig = GraphSignature(
        op_id="pi05_denoise_step",
        shapes=[bsize, cs, mad],
        dyn_dims=[len(past_key_values.k_caches),
                  ec.num_attention_heads, effective_num_kv_heads,
                  ec.head_dim, prefix_len, num_steps,
                  self.action_in_proj.out_features,
                  rope_pos,
                  *action_plan.graph_key_words()],  # physical child authority
        dtypes=[torch.float16],
    )

    for step in range(num_steps):
        cond_step = adarms_cond_all_rpu.select(0, step).contiguous()
        if use_graph:
            with cache.capture(sig):
                torch.ops.rpu.pi05_denoise_step_forward(
                    handle, x_t_rpu,
                    past_key_values.k_caches, past_key_values.v_caches,
                    cond_step, mask_4d, v_t_buf, prefix_len,
                    action_plan.selected.stage_tuple.physical_descriptor)
        else:
            torch.ops.rpu.pi05_denoise_step_forward(
                handle, x_t_rpu,
                past_key_values.k_caches, past_key_values.v_caches,
                cond_step, mask_4d, v_t_buf, prefix_len,
                [])
        # Euler CPU fp32 (Diff-B fix preserved):
        x_t_cpu_fp32 = x_t_cpu_fp32 + dt * v_t_buf.to('cpu', torch.float32)
        if _euler_fp16_enabled():  # A/B: model an on-device fp16 Euler
            x_t_cpu_fp32 = x_t_cpu_fp32.to(torch.float16).to(torch.float32)
        x_t_rpu.copy_(x_t_cpu_fp32.to(torch.float16))  # auto-flushes
    plans = vars(self).setdefault("_rpu_last_execution_plan", {})
    forwarded_chunk = int(
        torch.ops.rpu.pi05_denoise_step_get_resolved_chunk_size(handle)
    )
    if forwarded_chunk != action_plan.selected.stage_tuple.compute_chunk:
        raise RuntimeError(
            "Pi0.5 action dry/forward chunk mismatch: "
            f"planned={action_plan.selected.stage_tuple.compute_chunk}, "
            f"resolved={forwarded_chunk}"
        )
    plans["action"] = _fixed_plan_receipt(
        action_plan,
        component=PI05_ACTION_COMPONENT,
        stage="action",
        position=int(prefix_len),
    )
    return x_t_cpu_fp32



def _set_denoise_rope_position(handle, prefix_pad_masks) -> int:
    """Give the fused denoise expert the suffix's LOGICAL RoPE start.

    `prefix_len` (= prefix_pad_masks.shape[1]) is the PHYSICAL cache row the
    suffix K/V is written at, and the fused expert used it for RoPE too. The
    reference derives the suffix positions from `sum(prefix_pad_masks)` — the
    LOGICAL count — so the two agree only while the prefix has no pad rows.
    The suffix itself is never padded (suffix_pad_masks is all ones), so a
    scalar start is enough here; the PREFIX needs a gathered table instead
    (adapters/pi05/gemma.py).

    Returns the value so callers can put it in the graph signature — it is
    baked into the kernel at BUILD.
    """
    pos = int(prefix_pad_masks[0].sum())
    torch.ops.rpu.pi05_denoise_step_set_rope_position(handle, pos)
    return pos

def _run_denoise_loop(self, x_t_cpu_fp32, prefix_pad_masks, past_key_values,
                      num_steps, prefix_len, bsize):
    """In-graph N-step unroll: ONE pi05_denoise_loop_forward call (on-device
    fp16 Euler; 1 BUILD + 1 REPLAY). Same boundary contract as
    _run_denoise_fused (x_t CPU fp32 in, final action CPU fp32 out). The cond
    stack [num_steps, h_ada] is passed whole; C++ indexes it per body_iter."""
    from rpu_backend.adapters.pi05.weights import _precompute_adarms_cond_all
    from rpu_backend.graph import GraphSignature

    handle = getattr(self, "_rpu_fused_denoise_handle", None)
    if handle is None:
        raise RuntimeError("_run_denoise_loop requires fused handle")
    cache = getattr(self, "_rpu_fused_denoise_graph_cache", None)
    if cache is None:
        raise RuntimeError("_run_denoise_loop requires fused graph cache")

    cfg = self.config
    cs   = cfg.chunk_size
    mad  = cfg.max_action_dim
    dt   = -1.0 / num_steps

    k0 = past_key_values.k_caches[0]
    cache_max_seq = k0.size(1) * k0.size(5)
    if prefix_len + cs > cache_max_seq:
        raise RuntimeError(
            f"_run_denoise_loop: prefix_len({prefix_len}) + chunk_size({cs}) "
            f"exceeds k_cache.max_seq_len={cache_max_seq}.")

    cond_all = _precompute_adarms_cond_all(self, num_steps)  # [num_steps, h_ada]
    x0 = torch.empty(bsize, cs, mad, dtype=torch.float16, device='rpu')
    x0.copy_(x_t_cpu_fp32.to(torch.float16))
    x_out = torch.empty(bsize, cs, mad, dtype=torch.float16, device='rpu')

    mask_4d = _build_attention_mask_4d(
        self, prefix_pad_masks, prefix_len, bsize, cs,
        dtype=torch.float16, device='rpu')

    expert_model = self.paligemma_with_expert.gemma_expert.model
    ec = expert_model.config
    effective_num_kv_heads = int(getattr(
        expert_model, "_rpu_effective_num_kv_heads", ec.num_key_value_heads
    ))
    rope_pos = _set_denoise_rope_position(handle, prefix_pad_masks)
    action_plan = _plan_pi05_denoise_action_execution(
        self,
        handle=int(handle),
        logical_len=int(cs),
        position=int(prefix_len),
        kv_len=int(prefix_len + cs),
        cache_capacity=int(cache_max_seq),
        loop_mode=True,
        num_steps=int(num_steps),
        graph_cache=cache,
    )
    sig = GraphSignature(
        op_id="pi05_denoise_loop",
        shapes=[bsize, cs, mad],
        dyn_dims=[len(past_key_values.k_caches),
                  ec.num_attention_heads, effective_num_kv_heads,
                  ec.head_dim, prefix_len, num_steps,
                  self.action_in_proj.out_features,
                  rope_pos,
                  *action_plan.graph_key_words()],  # physical child authority
        dtypes=[torch.float16],
    )
    with cache.capture(sig):
        torch.ops.rpu.pi05_denoise_loop_forward(
            handle, x0,
            past_key_values.k_caches, past_key_values.v_caches,
            cond_all, mask_4d, x_out, dt, prefix_len, num_steps,
            action_plan.selected.stage_tuple.physical_descriptor)
    plans = vars(self).setdefault("_rpu_last_execution_plan", {})
    forwarded_chunk = int(
        torch.ops.rpu.pi05_denoise_step_get_resolved_chunk_size(handle)
    )
    if forwarded_chunk != action_plan.selected.stage_tuple.compute_chunk:
        raise RuntimeError(
            "Pi0.5 action-loop dry/forward chunk mismatch: "
            f"planned={action_plan.selected.stage_tuple.compute_chunk}, "
            f"resolved={forwarded_chunk}"
        )
    plans["action"] = _fixed_plan_receipt(
        action_plan,
        component=PI05_ACTION_COMPONENT,
        stage="action",
        position=int(prefix_len),
    )
    return x_out.to('cpu', torch.float32)


# =============================================================================
# Decomposed sample_actions class patch (D-3-02): the 270-LOC monolith at
# _adapter.py:483-751 split into 6 nested helpers + OUTER wrapper.
# =============================================================================

def install_sample_actions_patch() -> None:
    """BLOCKER 1 + CR-R3 Findings 1/3/6 + CR-R5 BLOCKER 9 closure.

    CLASS-level idempotent patch of `PI05Pytorch.sample_actions`. Installed
    ONCE per process (sentinel-guarded). The patched body:
      - Uses the EXACT installed lerobot signature (Finding 6).
      - Looks up pi05_handle from `_PI05_HANDLES[self]` at call time (Finding 3).
      - Precomputes ONLY adarms_cond_list (safe — no x_t dep) + attention_mask + prefix K/V.
        Passes initial noise to pi05_forward (Finding 1).
      - CR-R5 BLOCKER 2 cascade: no position_ids Tensor precompute.
      - CR-R5 BLOCKER 9: validates kwargs against allowlist; unknown kwargs raise.
      - Routes `num_steps` kwarg through to pi05_forward (Finding 8).
      - If RTC is enabled, falls back to the original CPU path (Phase 3 deferral).

    D-3-02 decomposition: helpers 2-5 lifted to module top-level (Task 4.3).
    Helper 1 (_resolve_handle_and_validate) stays nested: closes over
    `_orig_sample_actions` + `_RPU_ALLOWED_SAMPLE_ACTIONS_KWARGS`.
    _run_denoise stays nested: dispatcher that references top-level
    _run_denoise_python_baseline and _run_denoise_fused.
    """
    try:
        import lerobot  # noqa: F401 — probe only
    except ImportError as e:
        raise ImportError("Pi0.5 requires lerobot. Install with: pip install lerobot\nSee docs/api_reference.md#Migration for Pi0.5 setup.") from e
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
    if getattr(PI05Pytorch, "_rpu_patched_sample_actions", False):
        return

    _orig_sample_actions = PI05Pytorch.sample_actions

    # -------------------------------------------------------------------
    # Helper 1: validate handle + RTC fallback + BLOCKER 9 kwarg allowlist
    # + batch-size guard (from _adapter.py:507-540). Returns a SENTINEL tuple
    # `(proceed, fallback_kwargs)` — if proceed=False, caller returns
    # `_orig_sample_actions(..., **fallback_kwargs)` immediately.
    # STAYS nested: closes over `_orig_sample_actions`.
    # -------------------------------------------------------------------
    def _resolve_handle_and_validate(self, images, img_masks, tokens, masks,
                                      noise, num_steps, kwargs):
        handle = _PI05_HANDLES.get(self)
        if handle is None:
            # Not RPU-swizzled for this instance -> CPU original fallback.
            return (False, _orig_sample_actions(
                self, images, img_masks, tokens, masks,
                noise=noise, num_steps=num_steps, **kwargs))

        # RTC fallback — Phase 3 deferral.
        if hasattr(self, "_rtc_enabled") and self._rtc_enabled():
            return (False, _orig_sample_actions(
                self, images, img_masks, tokens, masks,
                noise=noise, num_steps=num_steps, **kwargs))

        # CR-R5 BLOCKER 9: fail-loud on unexpected kwargs instead of silent drop.
        unexpected = set(kwargs) - _RPU_ALLOWED_SAMPLE_ACTIONS_KWARGS
        if unexpected:
            raise RPUBackendError(
                f"Pi05Policy RPU-path sample_actions received unsupported kwargs: "
                f"{sorted(unexpected)}. Either extend "
                f"_RPU_ALLOWED_SAMPLE_ACTIONS_KWARGS in runtime.py or disable the "
                f"RPU path for this inference call.")

        bsize = tokens.shape[0]
        if bsize != 1:
            raise RPUBackendError(
                f"Pi05 RPU path requires batch_size=1, got {bsize}. Phase 3 is "
                "single-batch only (see CONTEXT §deferred).")

        return (True, None)

    # -------------------------------------------------------------------
    # Helper 6: denoise dispatcher (Task 4.3).
    # STAYS nested: sits alongside _patched_sample_actions; references
    # top-level _run_denoise_python_baseline and _run_denoise_fused.
    # -------------------------------------------------------------------
    def _run_denoise(self, x_t, prefix_pad_masks, past_key_values, num_steps,
                     prefix_len, bsize):
        """Dispatcher: fused (default) -> baseline + safety-net on first call."""
        handle = getattr(self, "_rpu_fused_denoise_handle", None)
        if handle is None:
            return _run_denoise_python_baseline(
                self, x_t, prefix_pad_masks, past_key_values,
                num_steps, prefix_len, bsize)

        if getattr(self, "_rpu_fuse_validated", False):
            return _run_denoise_fused(
                self, x_t, prefix_pad_masks, past_key_values,
                num_steps, prefix_len, bsize)

        # First call: safety net (spec §6.3).
        prefix_hash_before = _kv_prefix_full_hash(past_key_values, prefix_len)

        final_baseline = _run_denoise_python_baseline(
            self, x_t.clone(), prefix_pad_masks, past_key_values,
            num_steps, prefix_len, bsize)
        prefix_hash_after_baseline = _kv_prefix_full_hash(past_key_values, prefix_len)
        if prefix_hash_before != prefix_hash_after_baseline:
            raise RuntimeError(
                "[Pi05Fused] safety net: baseline path corrupted prefix.")

        final_fused = _run_denoise_fused(
            self, x_t.clone(), prefix_pad_masks, past_key_values,
            num_steps, prefix_len, bsize)
        prefix_hash_after_fused = _kv_prefix_full_hash(past_key_values, prefix_len)
        if prefix_hash_before != prefix_hash_after_fused:
            raise RuntimeError(
                "[Pi05Fused] zero-copy violated: fused wrote into prefix KV. "
                "Disable via RPU_PI05_FUSED_DENOISE=0 and file a bug.")

        try:
            parity = _validate_fused_parity(
                final_fused,
                final_baseline,
                where="[Pi05Fused] safety net mismatch",
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc}. Disable via RPU_PI05_FUSED_DENOISE=0 and file a bug."
            ) from exc
        _LOG.info(
            "[Pi05Fused] safety net PASS: mse=%.6e max_abs=%.6e "
            "cosine=%.8f row_cosine_p01=%.8f row_cosine_min=%.8f "
            "row_zero_pair/single=%.0f/%.0f row_mse_p99=%.6e "
            "row_mse_max=%.6e relative_l2=%.6e",
            parity["mse"],
            parity["max_abs"],
            parity["cosine"],
            parity["row_cosine_p01"],
            parity["row_cosine_min"],
            parity["row_both_zero_count"],
            parity["row_single_zero_count"],
            parity["row_mse_p99"],
            parity["row_mse_max"],
            parity["relative_l2"],
        )

        self._rpu_fuse_validated = True
        return final_fused

    # -------------------------------------------------------------------
    # OUTER wrapper — matches lerobot signature; calls helpers 1..6 in order.
    # -------------------------------------------------------------------
    @execution_serialized
    @torch.no_grad()
    def _patched_sample_actions(self, images, img_masks, tokens, masks,
                                 noise=None, num_steps=None, **kwargs):
        """CR-R3 Finding 6: EXACT signature match with lerobot modeling_pi05.py:786-795."""
        # Helper 1: resolve handle + RTC + allowlist + bsize guard
        proceed, fallback = _resolve_handle_and_validate(
            self, images, img_masks, tokens, masks, noise, num_steps, kwargs)
        if not proceed:
            return fallback

        vars(self)["_rpu_last_execution_plan"] = {}

        if num_steps is None:
            num_steps = self.config.num_inference_steps
        _validate_prepared_graph_profile(self, num_steps)

        # The fused denoise loop is the previous prediction's final subsystem
        # and intentionally keeps its graph-owned SPM plan alive. Reset the
        # global temporary allocator at the next top-level boundary, before
        # entering any per-subsystem capture, so consecutive policy calls
        # share the same deterministic allocator boundary.
        torch.ops.rpu.spm_alloc_reset_temporary()

        bsize = tokens.shape[0]

        # =====================================================================
        # ITERATION 5 (2026-04-21) — Python denoise loop matching legacy
        # `_patched_denoise_step` semantics in-tree (no env-guarded fallback).
        # =====================================================================

        # Helper 2: resolve RPU device (top-level)
        rpu_device = _resolve_rpu_device(self)

        # Helper 3: noise sampling (top-level)
        x_t = _sample_noise(self, bsize, noise)

        # Helper 4: VLM prefill (top-level)
        past_key_values, prefix_pad_masks, prefix_len = _prefill_prefix_embs(
            self, images, img_masks, tokens, masks, rpu_device)

        # Helper 5: apply shared KV cache to expert (top-level)
        _apply_kv_cache(self, past_key_values)

        # Helper 6: denoise dispatcher
        x_t = _run_denoise(self, x_t, prefix_pad_masks, past_key_values,
                           num_steps, prefix_len, bsize)

        return x_t

    PI05Pytorch.sample_actions = _patched_sample_actions
    PI05Pytorch._rpu_patched_sample_actions = True
