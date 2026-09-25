"""Pi0.5 internal orchestrator (Pi05Adapter) + Q7 module-level swizzle lock.

v5-02 / B3: this file is the canonical home for Pi05Adapter — the internal
orchestrator that performs the irreversible CPU→RPU mutation triggered by
``Pi05Policy.to('rpu')``. The class was previously co-located with
``Pi05Policy`` in ``transformers/pi05/policy.py``; v5-01a-2 split ownership
so ``api/policy.py`` (Pi05Policy) is a thin lazy-import wrapper while this
module holds the swizzle / patch / handle bookkeeping. v5-02 relocates the
package from ``transformers/pi05/`` to ``adapters/pi05/`` per ADR §8 / B3.

``_SWIZZLE_LOCK`` (Q7) is a module-level non-reentrant guard that serializes
concurrent ``Pi05Adapter.to_rpu()`` calls — see CONTEXT §domain "G1-A" verdict
for why the lock lives with the class instead of in api/policy.py.
"""
from __future__ import annotations
from dataclasses import replace
import os
import threading
from typing import Any

import torch

from rpu_backend.runtime import rpu_env_bool
from rpu_backend.api.errors import RPUBackendError, RPUUnsupportedDtypeError
from rpu_backend.api.causal_lm import _claim_live_instance, _release_live_instance
from rpu_backend.api._execution import (
    bind_execution_session,
    bind_rpu_execution,
    execution_serialized,
    native_execution_reconfigure,
)
from rpu_backend.runtime.execution_planner import GRAPH_COMPOSITE_CHILD
from .runtime import (
    PI05_ACTION_COMPONENT,
    PI05_EXECUTION_COMPONENTS,
    PI05_TEXT_COMPONENT,
    PI05_VISION_COMPONENT,
    _denoise_graph_enabled,
    _denoise_unroll_enabled,
    _kvinsert_pad16_enabled,
    _prefix_pad16_enabled,
    resolve_pi05_execution_components,
)


# Q7: Pi05-SPECIFIC module-level swizzle lock (from _adapter.py:57).
_SWIZZLE_LOCK = threading.Lock()


# Spec §5.5: env switch for opt-out of the fused denoise step path.
# Default ON; set RPU_PI05_FUSED_DENOISE=0 to fall back to legacy.
_FUSED_ENV_VAR = "RPU_PI05_FUSED_DENOISE"
_MISSING = object()
_FUSED_DENOISE_INSTALL_ATTRS = (
    "_pi05_denoise_graph_enabled",
    "_pi05_denoise_unroll_enabled",
    "_rpu_fused_denoise_handle",
    "_rpu_fused_denoise_handle_finalizer",
    "_rpu_fused_denoise_retirement_state",
    "_rpu_fused_denoise_graph_cache",
)


from rpu_backend.api._execution import PI05_EXECUTION_SUPPORTED


def _fused_denoise_enabled() -> bool:
    return rpu_env_bool(_FUSED_ENV_VAR, default=True)


def _pi05_graph_runtime_policy():
    """Freeze Pi's Graph choices without mutating process-wide defaults."""
    from rpu_backend.graph import GraphRuntimePolicy

    # Vision, prefill, denoise and the first-call AdaRMS safety graph stay
    # resident; device prefix assembly also needs one transient queue.
    inherited = GraphRuntimePolicy.from_environment(graph_arena_count=5)
    fast_replay = rpu_env_bool("RPU_WALL_OSS_FAST_REPLAY", default=True)
    deep_replay = rpu_env_bool("RPU_DEEP_FAST_REPLAY", default=True)
    return replace(
        inherited,
        fmb_fast_replay=fast_replay,
        fmb_deep_fast_replay=deep_replay,
        provenance=inherited.provenance + (
            f"owner:pi05:fmb_fast_replay={int(fast_replay)}",
            f"owner:pi05:fmb_deep_fast_replay={int(deep_replay)}",
        ),
    )


def _validate_execution_geometry(adapter: "Pi05Adapter") -> None:
    """Reject model-specific fixed chunks before irreversible RPU mutation."""
    model = adapter._lerobot_policy.model
    execution, vision, _text, action = adapter._resolve_execution(
        adapter._rpu_execution,
        entry_point="Pi05Adapter.to_rpu",
    )

    action_chunk = action.get("action", {}).get("chunk_size", "auto")
    action_horizon = int(model.config.chunk_size)
    action_required = ((action_horizon + 15) // 16) * 16
    if isinstance(action_chunk, int) and action_chunk != action_required:
        raise ValueError(
            f"Pi0.5 action chunk_size={action_chunk} must equal the "
            f"single-chunk capacity {action_required} for the full "
            f"{action_horizon}-row action horizon; use 'auto' or exactly "
            f"{action_required}."
        )

    image_features = getattr(model.config, "image_features", None)
    from .cores import vision_batch_enabled
    batch_vision = vision_batch_enabled(execution)
    num_vision_chunks = (
        len(image_features) if batch_vision and image_features else 1
    )
    vision_required = num_vision_chunks * 256
    vision_chunk = vision.get("vision", {}).get("chunk_size", "auto")
    if isinstance(vision_chunk, int) and vision_chunk != vision_required:
        mode = (
            f"packed {num_vision_chunks}-camera"
            if num_vision_chunks > 1
            else "single-image"
        )
        raise ValueError(
            f"Pi0.5 {mode} vision chunk_size={vision_chunk} must equal the "
            f"fixed single-chunk capacity {vision_required}; use 'auto' or "
            f"exactly {vision_required}."
        )


def _preflight_pi05_cold_model(lerobot_policy: Any, *, attn_tp: int = 8):
    """Resolve existing host-only dtype/config gates before allocator claim."""
    quant_config = getattr(lerobot_policy, "_pi05_quant_config", {}) or {}
    nvfp4_scales = set()
    if quant_config.get("method") == "nvfp4a16" and quant_config.get("nvfp4_abi") == "striped_v2":
        if attn_tp != 8:
            raise RPUBackendError("Pi05 NVFP4 striped_v2 requires TP8 before swizzle")
        for module_name, module in lerobot_policy.named_modules():
            ts = getattr(module, "tensor_scale", None)
            if (getattr(module, "_pi05_nvfp4_abi", None) == "striped_v2"
                    and ts is not None and ts.dtype == torch.float32 and ts.numel() == 1
                    and torch.isfinite(ts).all() and (ts > 0).all()):
                nvfp4_scales.add(module_name + ".tensor_scale")
    bad_dtypes = []
    for name, parameter in lerobot_policy.named_parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.float16:
            bad_dtypes.append(f"param {name}: {parameter.dtype}")
            if len(bad_dtypes) >= 5:
                break
    if len(bad_dtypes) < 5:
        for name, buffer in lerobot_policy.named_buffers():
            if (buffer.is_floating_point() and buffer.dtype != torch.float16
                    and name not in nvfp4_scales):
                bad_dtypes.append(f"buffer {name}: {buffer.dtype}")
                if len(bad_dtypes) >= 5:
                    break
    if bad_dtypes:
        raise RPUUnsupportedDtypeError(
            "Pi05Adapter.to_rpu(): floating params/buffers must be "
            "torch.float16 before swizzle. Offenders (first 5): "
            f"{bad_dtypes}. Reload with dtype=torch.float16 or "
            "call .half() on the lerobot policy."
        )

    try:
        pi05_pytorch = lerobot_policy.model
        pawe = pi05_pytorch.paligemma_with_expert
        paligemma_model = pawe.paligemma.model
        vlm = paligemma_model.language_model
        expert = pawe.gemma_expert.model
        siglip = paligemma_model.vision_tower
        projector = paligemma_model.multi_modal_projector
        siglip_inner = (
            siglip.vision_model if hasattr(siglip, "vision_model") else siglip
        )
        for label, decoder in (("vlm", vlm), ("expert", expert)):
            config = decoder.config
            if not decoder.layers:
                raise RPUBackendError(
                    f"Pi05Adapter.to_rpu(): {label}.layers is empty"
                )
            original_nkv = int(config.num_key_value_heads)
            num_q_heads = int(config.num_attention_heads)
            head_dim = int(config.head_dim)
            if (
                original_nkv <= 0
                or head_dim <= 0
                or attn_tp < original_nkv
                or attn_tp % original_nkv
                or num_q_heads % attn_tp
            ):
                raise RPUBackendError(
                    "Pi05Adapter.to_rpu(): invalid KV replication config for "
                    f"{label}: nkv={original_nkv}, q_heads={num_q_heads}, "
                    f"head_dim={head_dim}"
                )
        if int(vlm.config.head_dim) != int(expert.config.head_dim):
            raise RPUBackendError(
                "Pi05Adapter.to_rpu(): VLM/expert cache sharing requires "
                "matching head_dim, got "
                f"vlm={vlm.config.head_dim}, expert={expert.config.head_dim}"
            )
        vlm_q_dtype = vlm.layers[0].self_attn.q_proj.weight.dtype
    except RPUBackendError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise RPUBackendError(
            "Pi05Adapter.to_rpu(): incomplete or malformed lerobot model"
        ) from exc

    from rpu_backend.adapters.pi05.w4pack import int4_child_names

    quant_config = getattr(lerobot_policy, "_pi05_quant_config", {}) or {}
    int4_children = (int4_child_names(quant_config, component="vlm")
                     | int4_child_names(quant_config, component="expert"))
    group_size = quant_config.get("int4_group_size", quant_config.get("group_size", 32))
    if type(group_size) is not int or group_size not in (32, 64, 128):
        raise ValueError("Pi0.5 int4_group_size must be 32, 64, or 128")
    return (
        pi05_pytorch,
        pawe,
        paligemma_model,
        vlm,
        expert,
        siglip_inner,
        projector,
        attn_tp,
        attn_tp,
        int4_children,
        vlm_q_dtype in (torch.int8, torch.uint8),
    )


def _apply_vlm_chunk_size(adapter: "Pi05Adapter", vlm: Any) -> None:
    """Apply the resolved text-child authority to the legacy model field."""
    vars(vlm)["_rpu_chunk_size"] = adapter._vlm_chunk_size


def _destroy_fused_denoise_handle(handle: int) -> None:
    """Raw destroy; the installed resource owns retirement failures."""
    torch.ops.rpu.pi05_denoise_step_destroy(handle)


def _clear_component_graph_cache(owner: Any, attr: str) -> None:
    """Best-effort release of a partially installed component GraphCache."""
    if owner is None:
        return
    cache = getattr(owner, attr, None)
    if cache is not None:
        try:
            cache.clear()
        except Exception:
            pass
    vars(owner).pop(attr, None)


def _cleanup_component_handle(
    owner: Any,
    *,
    handle_attr: str,
    finalizer_attr: str,
    destroy,
    install_attrs: tuple[str, ...] = (),
) -> None:
    """Destroy one published component handle and UNINSTALL what it published.

    `install_attrs` is the component's OWN install-attribute tuple — the exact
    list its patch function writes and its local rollback pops
    (`_GEMMA_RUNTIME_INSTALL_ATTRS`, `_ADARMS_RUNTIME_INSTALL_ATTRS`,
    `_SIGLIP_RUNTIME_INSTALL_ATTRS`, `_FUSED_DENOISE_INSTALL_ATTRS`). Those
    tuples existed all along and this orchestrator never used them: it popped
    only `(handle_attr, finalizer_attr)`, so when a LATER component failed the
    earlier ones were left holding `_rpu_cache`, their GraphCache, the
    lazy-init markers and — the load-bearing one — their patched instance
    `forward`, all still bound to a handle this function had just destroyed.
    """
    if owner is None:
        return
    finalizer = getattr(owner, finalizer_attr, None)
    handle = getattr(owner, handle_attr, None)
    if finalizer is not None and getattr(finalizer, "alive", False):
        try:
            finalizer()
        except Exception:
            pass
    elif finalizer is None and handle is not None:
        try:
            destroy(handle)
        except Exception:
            pass
    owner_state = vars(owner)
    for attr in (handle_attr, finalizer_attr, *install_attrs):
        owner_state.pop(attr, None)


def _release_after_failed_install(owner: Any, *, pre_swizzle: bool) -> None:
    """Release only when failure preceded irreversible Pi0.5 weight mutation."""
    if not pre_swizzle:
        return
    try:
        _release_live_instance(owner)
    except Exception:
        pass


def _closed_pi05_forward(*args, **kwargs):
    raise RPUBackendError("Pi0.5 is closed; reload the policy before forward.")


def _retirement_owner_owns(parent, child, handle_attr, finalizer, cache) -> bool:
    if not isinstance(parent, Pi05Adapter):
        return False
    model = parent._lerobot_policy.model
    session = parent._execution_session
    return bool(
        getattr(model, "_pi05_retirement_owner", None) is parent
        and session is getattr(model, "_execution_session", None)
        and session._owner is model
        and not getattr(parent, "_retirement_failed", None)
        and not parent._closed
        and any(
            owner is child and name == handle_attr
            and handle == getattr(child, name, None)
            and tracked is finalizer and graph is cache
            and resource is getattr(child, name.replace("_handle", "_retirement_state"), None)
            for owner, name, _graph_name, _destroy, _attrs, handle, tracked, graph, resource
            in getattr(parent, "_retirement_components", ())
        )
    )


def _bound_component_runtime(
    owner: Any,
    *,
    handle_attr: str,
    finalizer_attr: str,
    graph_cache_attr: str,
    forward_name: str,
) -> tuple[Any, Any] | None:
    """Return a component's live handle/finalizer pair when fully callable."""
    if owner is None:
        return None
    handle = getattr(owner, handle_attr, None)
    finalizer = getattr(owner, finalizer_attr, None)
    if handle is None or finalizer is None:
        return None
    if not getattr(finalizer, "alive", False) and not _retirement_owner_owns(
        getattr(owner, "_pi05_retirement_owner", None), owner, handle_attr,
        finalizer, getattr(owner, graph_cache_attr, None),
    ):
        return None
    if getattr(owner, graph_cache_attr, None) is None:
        return None
    if getattr(owner, "_rpu_cache", None) is None:
        return None
    installed_forward = vars(owner).get("forward")
    forward_function = getattr(installed_forward, "__func__", None)
    if not (
        getattr(installed_forward, "__self__", None) is owner
        and getattr(forward_function, "__name__", None) == forward_name
    ):
        return None
    return handle, finalizer


def _pi05_runtime_state(lerobot_policy: Any) -> dict[str, Any] | None:
    """Return complete Pi0.5 ownership state, never a marker-only install."""
    if getattr(lerobot_policy, "_rpu_swizzled", False) is not True:
        return None
    if getattr(lerobot_policy, "_rpu_swizzle_started", False) is not True:
        return None
    try:
        pi05_pytorch = lerobot_policy.model
        pawe = pi05_pytorch.paligemma_with_expert
        vlm = pawe.paligemma.model.language_model
        expert = pawe.gemma_expert.model
        vision_tower = pawe.paligemma.model.vision_tower
        siglip_inner = getattr(vision_tower, "vision_model", vision_tower)
    except AttributeError:
        return None

    vlm_state = _bound_component_runtime(
        vlm,
        handle_attr="_rpu_vlm_decoder_handle",
        finalizer_attr="_rpu_vlm_decoder_handle_finalizer",
        graph_cache_attr="_rpu_gemma_graph_cache",
        forward_name="rpu_gemma_model_forward",
    )
    expert_state = _bound_component_runtime(
        expert,
        handle_attr="_rpu_action_handle",
        finalizer_attr="_rpu_action_handle_finalizer",
        graph_cache_attr="_rpu_adarms_graph_cache",
        forward_name="rpu_adarms_model_forward",
    )
    siglip_state = _bound_component_runtime(
        siglip_inner,
        handle_attr="_rpu_vision_handle",
        finalizer_attr="_rpu_vision_handle_finalizer",
        graph_cache_attr="_rpu_siglip_graph_cache",
        forward_name="rpu_siglip_forward",
    )
    if vlm_state is None or expert_state is None or siglip_state is None:
        return None
    chunk_size = getattr(vlm, "_rpu_chunk_size", None)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        return None
    if chunk_size < 0 or (chunk_size and chunk_size % 16 != 0):
        return None

    fused_attrs = (
        "_rpu_fused_denoise_handle",
        "_rpu_fused_denoise_handle_finalizer",
        "_rpu_fused_denoise_graph_cache",
    )
    if any(not hasattr(pi05_pytorch, name) for name in fused_attrs):
        return None
    fused_handle = pi05_pytorch._rpu_fused_denoise_handle
    fused_finalizer = pi05_pytorch._rpu_fused_denoise_handle_finalizer
    fused_graph_cache = pi05_pytorch._rpu_fused_denoise_graph_cache
    if fused_handle is None:
        if fused_finalizer is not None or fused_graph_cache is not None:
            return None
    elif (
        fused_finalizer is None
        or (not getattr(fused_finalizer, "alive", False)
            and not _retirement_owner_owns(
                getattr(pi05_pytorch, "_pi05_retirement_owner", None),
                pi05_pytorch, "_rpu_fused_denoise_handle", fused_finalizer,
                fused_graph_cache))
        or fused_graph_cache is None
    ):
        return None

    required_runtime_attrs = (
        "_rpu_runtime_device",
        "_rpu_adarms_cond_cache",
        "_rpu_denoise_mask_cache",
        "_pi05_language_prefix_cache",
        "_pi05_prefix_assembly_ok",
        "_pi05_prefix_assembly_enabled",
        "_rpu_siglip_batch_n",
        "_rpu_legacy_prefix_pad16_request",
        "_rpu_legacy_kvinsert_pad16_request",
    )
    if any(not hasattr(pi05_pytorch, name) for name in required_runtime_attrs):
        return None
    if getattr(pi05_pytorch._rpu_runtime_device, "type", None) != "rpu":
        return None
    batch_n = pi05_pytorch._rpu_siglip_batch_n
    if isinstance(batch_n, bool) or not isinstance(batch_n, int) or batch_n < 1:
        return None
    if pi05_pytorch._pi05_prefix_assembly_enabled is not True:
        return None

    from .runtime import _PI05_HANDLES

    try:
        pi05_handle = _PI05_HANDLES.get(pi05_pytorch)
    except TypeError:
        return None
    if pi05_handle is None:
        return None
    return {
        "pi05_handle": pi05_handle,
        "vlm_handle": vlm_state[0],
        "expert_handle": expert_state[0],
        "siglip_handle": siglip_state[0],
        "fused_finalizer": fused_finalizer,
        "vlm_chunk_size": chunk_size,
    }


def _reset_adapter_runtime_state(adapter: Any) -> None:
    adapter._rpu_is_ready = False
    adapter._pi05_handle = None
    adapter._siglip_handle = None
    adapter._vlm_handle = None
    adapter._expert_handle = None
    adapter._finalizers = []


def _sync_adapter_runtime_state(adapter: Any, state: dict[str, Any]) -> None:
    adapter._rpu_is_ready = True
    adapter._pi05_handle = state["pi05_handle"]
    adapter._siglip_handle = state["siglip_handle"]
    adapter._vlm_handle = state["vlm_handle"]
    adapter._expert_handle = state["expert_handle"]
    adapter._vlm_chunk_size = state["vlm_chunk_size"]
    fused_finalizer = state["fused_finalizer"]
    adapter._finalizers = (
        [fused_finalizer] if fused_finalizer is not None else []
    )


# =============================================================================
# Pi05Adapter — internal orchestrator for `Pi05Policy.to('rpu')` mutation.
# Relocated verbatim from _adapter.py:767-1207 with Step A / Step C / Step E
# delegating to patches.py / weights.py / runtime.py per D-3-01.
# =============================================================================

def _selected_prefill_pair_rows(adapter) -> int:
    """Resolve only the explicitly selected cold text bucket, never tokenizer defaults."""
    text_tokens = getattr(adapter, "_rpu_prefill_text_tokens", 32)
    if type(text_tokens) is not int or text_tokens not in (32, 64, 96, 128):
        raise ValueError("Pi0.5 cold text bucket must be 32, 64, 96, or 128")
    config = getattr(adapter._lerobot_policy, "config", None)
    if config is None:
        config = adapter._lerobot_policy.model.config
    cameras = len(config.image_features)
    if text_tokens != 32:
        if cameras not in (2, 3):
            raise ValueError(f"Pi0.5 T{text_tokens} requires exactly two or three cameras")
        if getattr(adapter, "_rpu_execution", {}).get("model", {}).get("num_cores", 8) != 8:
            raise ValueError(f"Pi0.5 T{text_tokens} currently requires exactly eight cores")
        return (256 * cameras + text_tokens) // 2
    return 272 if cameras == 2 else 400


class Pi05Adapter:
    """Internal orchestrator for `Pi05Policy.to('rpu')` mutation."""

    def __init__(self, lerobot_policy: Any) -> None:
        self._lerobot_policy = lerobot_policy
        self._rpu_prefill_text_tokens = 32
        model = getattr(lerobot_policy, "model", lerobot_policy)
        model_state = vars(model)
        state = _pi05_runtime_state(lerobot_policy)
        if "_pi05_prefill_mask_fp16" not in model_state:
            model_state["_pi05_prefill_mask_fp16"] = rpu_env_bool(
                "RPU_PI05_PREFILL_MASK_FP16", default=False
            )
        existing_session = getattr(model, "_execution_session", None)
        # A fresh model must reach A9 preinstall with public attributes only.
        # Keep these legacy environment requests on the adapter/session until
        # installation owns the model; mirrors reuse that exact cold snapshot.
        cold_pad16 = getattr(existing_session, "_pi05_legacy_pad16_requests", None)
        if cold_pad16 is None:
            cold_pad16 = (
                bool(model_state["_rpu_legacy_prefix_pad16_request"])
                if "_rpu_legacy_prefix_pad16_request" in model_state
                else bool(_prefix_pad16_enabled()),
                bool(model_state["_rpu_legacy_kvinsert_pad16_request"])
                if "_rpu_legacy_kvinsert_pad16_request" in model_state
                else bool(_kvinsert_pad16_enabled()),
            )
        self._legacy_pad16_requests = cold_pad16
        (self._legacy_prefix_pad16_request,
         self._legacy_kvinsert_pad16_request) = cold_pad16
        inherited_generation = int(getattr(existing_session, "generation", 0))
        inherited_component_generations = getattr(
            model, "_pi05_execution_component_generations", None
        )
        existing_execution = getattr(model, "_rpu_execution", None)
        self._rpu_execution = bind_rpu_execution(
            model,
            existing_execution,
            entry_point="Pi05Adapter",
            supported=PI05_EXECUTION_SUPPORTED,
            supported_components=PI05_EXECUTION_COMPONENTS,
        )
        # Zero lets the native planner select a chunk for the current shape.
        # Pi05Policy.from_pretrained can bind an explicit admitted chunk.
        self._vlm_chunk_size = 0
        self._legacy_vlm_chunk_size = self._discover_legacy_vlm_chunk_size()
        self._execution_generation = inherited_generation
        self._component_generations = (
            {
                PI05_VISION_COMPONENT: inherited_generation,
                PI05_TEXT_COMPONENT: inherited_generation,
                PI05_ACTION_COMPONENT: inherited_generation,
            }
            if inherited_component_generations is None
            else {
                component: int(inherited_component_generations[component])
                for component in PI05_EXECUTION_COMPONENTS
            }
        )
        self._execution_reconfigure_journal = None
        self._closed = (
            existing_session is not None
            and existing_session.stats()["state"] in {"CLOSED", "POISONED"}
        )
        _reset_adapter_runtime_state(self)
        state = _pi05_runtime_state(lerobot_policy)
        if state is not None:
            _sync_adapter_runtime_state(self, state)
            self._rpu_execution = getattr(
                lerobot_policy.model, "_rpu_execution", {}
            )
            self._legacy_vlm_chunk_size = state["vlm_chunk_size"]
        self._publish_execution_views(
            self._resolve_execution(
                self._rpu_execution,
                entry_point="Pi05Adapter",
            ),
            generation=inherited_generation,
            component_generations=self._component_generations,
        )

        session_callbacks = {}
        if getattr(model, "_execution_session", None) is None:
            session_callbacks = {
                "validate": self.validate_execution_reconfigure,
                "apply": self.apply_execution_reconfigure,
                "rollback": self.rollback_execution_reconfigure,
            }
        self._execution_session = bind_execution_session(
            model,
            self._rpu_execution,
            entry_point="Pi05Adapter",
            supported=PI05_EXECUTION_SUPPORTED,
            supported_components=PI05_EXECUTION_COMPONENTS,
            graph_mode=GRAPH_COMPOSITE_CHILD,
            **session_callbacks,
        )
        self._execution_session._pi05_legacy_pad16_requests = self._legacy_pad16_requests
        self._execution_session.register_config_view(self)
        # Prefix/denoise methods live on the policy model; standalone child
        # forwards use their real module objects. Both address the same cold
        # component context without a global handle-to-cost map.
        try:
            pawe = model.paligemma_with_expert
            tower = pawe.paligemma.model.vision_tower
            children = (
                (pawe.paligemma.model.language_model, PI05_TEXT_COMPONENT),
                (pawe.gemma_expert.model, PI05_ACTION_COMPONENT),
                (getattr(tower, "vision_model", tower), PI05_VISION_COMPONENT),
            )
        except AttributeError:
            # Keep incomplete cold models on the existing installer diagnostic
            # path; no physical child identity exists to bind yet.
            children = ()
        if children:
            for component in (PI05_TEXT_COMPONENT, PI05_ACTION_COMPONENT):
                self._execution_session._bind_planner_owner(model, component)
        for child, component in children:
            self._execution_session._bind_planner_owner(child, component)

    def _validate_cold_pad16_requests(self) -> None:
        """Validate the adapter/session snapshot without publishing state."""
        model = self._lerobot_policy.model
        if (getattr(model, "_execution_session", None) is not self._execution_session
                or self._legacy_pad16_requests != self._execution_session._pi05_legacy_pad16_requests):
            raise RPUBackendError("Pi0.5 PAD16 cold snapshot lost its execution owner")

    def _publish_cold_pad16_requests(self) -> None:
        """Publish the already validated snapshot after allocator selection."""
        model = self._lerobot_policy.model
        prefix, kvinsert = self._legacy_pad16_requests
        model._rpu_legacy_prefix_pad16_request = prefix
        model._rpu_legacy_kvinsert_pad16_request = kvinsert

    def _retirement_inventory(self):
        """The existing four native owners, including the optional fused loop."""
        pi05_pytorch = self._lerobot_policy.model
        pawe = pi05_pytorch.paligemma_with_expert
        vlm = pawe.paligemma.model.language_model
        expert = pawe.gemma_expert.model
        vision_tower = pawe.paligemma.model.vision_tower
        vision = getattr(vision_tower, "vision_model", vision_tower)

        from .adarms import _ADARMS_RUNTIME_INSTALL_ATTRS
        from .gemma import _GEMMA_RUNTIME_INSTALL_ATTRS
        from rpu_backend.adapters.siglip import _SIGLIP_RUNTIME_INSTALL_ATTRS

        return tuple(
            (owner, handle_name, graph_name, destroy, attrs,
             getattr(owner, handle_name, None),
             getattr(owner, handle_name + "_finalizer", None),
             getattr(owner, graph_name, None),
             getattr(owner, handle_name.replace("_handle", "_retirement_state"), None))
            for owner, handle_name, graph_name, destroy, attrs in (
                (pi05_pytorch, "_rpu_fused_denoise_handle",
                 "_rpu_fused_denoise_graph_cache", "pi05_denoise_step_destroy",
                 _FUSED_DENOISE_INSTALL_ATTRS),
                (vision, "_rpu_vision_handle", "_rpu_siglip_graph_cache",
                 "siglip_destroy", _SIGLIP_RUNTIME_INSTALL_ATTRS),
                (expert, "_rpu_action_handle", "_rpu_adarms_graph_cache",
                 "adarms_destroy", _ADARMS_RUNTIME_INSTALL_ATTRS),
                (vlm, "_rpu_vlm_decoder_handle", "_rpu_gemma_graph_cache",
                 "gemma_destroy", _GEMMA_RUNTIME_INSTALL_ATTRS),
            )
        )

    def _take_retirement_ownership(self) -> None:
        model = self._lerobot_policy.model
        parent = getattr(model, "_pi05_retirement_owner", None)
        if parent is self:
            return
        if parent is not None:
            raise RuntimeError("Pi0.5 already has a different retirement owner")
        components = self._retirement_inventory()
        self._retirement_components = components
        for _owner, _name, _graph_name, _destroy, _attrs, handle, finalizer, graph, _resource in components:
            if handle is not None and graph is None:
                raise RuntimeError("Pi0.5 native retirement lost its GraphCache")
            if handle is not None and finalizer is not None and not finalizer.alive:
                raise RuntimeError("Pi0.5 cannot adopt a consumed native finalizer")
        # Publication and partial rollback adopt the same installed leaves.
        # Pending configure failures remain retained by their leaf resource.
        for owner, name, _graph_name, _destroy, _attrs, handle, finalizer, _graph, resource in components:
            if resource is not None and handle is not None:
                if resource.owner() is not owner or resource.finalizer is not finalizer:
                    raise RuntimeError("Pi0.5 component lost its actual retirement resource")
                resource.take_ownership(self)
            vars(owner)["_pi05_retirement_owner"] = self
            if finalizer is not None and finalizer.alive:
                finalizer.detach()
        self._gc_retirement_enabled = True

    def _retain_retirement_failure(self, error) -> None:
        from rpu_backend.api._execution import _mark_execution_process_unsafe
        from rpu_backend.runtime._native_retirement import _FAILED_RETIREMENTS, _InstalledNativeResource

        if getattr(self, "_retirement_failed", None) is None:
            self._retirement_failed = error
            _FAILED_RETIREMENTS.append(self)
        # A failure while adopting legacy callbacks must also disarm their
        # interpreter-exit path; retaining the parent alone does not do that.
        for component in getattr(self, "_retirement_components", ()):
            current_resource = getattr(component[0], component[1].replace(
                "_handle", "_retirement_state"), None)
            for resource in (component[8], current_resource):
                if isinstance(resource, _InstalledNativeResource) and resource.handle is not None:
                    resource.retain_failure(error, self)
                    if resource.finalizer.alive:
                        resource.finalizer.detach()
            for finalizer in (component[6], getattr(
                    component[0], component[1] + "_finalizer", None)):
                if finalizer is not None and finalizer.alive:
                    finalizer.detach()
        self._execution_session.poison()
        _mark_execution_process_unsafe(f"Pi0.5 retirement failed: {error}")

    def _close_without_session(self, *, release=True) -> None:
        """Clear every graph before any raw destroy; retain the first failure."""
        if release and not self._rpu_is_ready and not getattr(self, "_retirement_components", ()):
            self._closed = True
            return
        try:
            self._take_retirement_ownership()
            from rpu_backend.api._execution import _require_execution_process_safe

            _require_execution_process_safe()
            inventory = self._retirement_inventory()
            for current, owned in zip(inventory, self._retirement_components):
                if any(current[i] is not owned[i] for i in (0, 6, 7, 8)) or current[5] != owned[5]:
                    raise RuntimeError("Pi0.5 retirement owner/handle/Graph identity changed")
                resource = owned[8]
                if resource is not None and current[5] is not None and (
                        resource.handle != current[5] or resource.failed is not None
                        or resource.finalizer is not current[6]
                        or resource.graphs != (current[7],)
                        or (resource.owner() is not None and resource.owner() is not current[0])):
                    raise RuntimeError("Pi0.5 native retirement handle identity changed")
            seen = set()
            for _owner, _name, _graph_name, _destroy, _attrs, _handle, _finalizer, graph, _resource in inventory:
                if graph is not None and id(graph) not in seen:
                    seen.add(id(graph))
                    graph.clear()
                    if not graph.cache_invariant_ok():
                        raise RuntimeError("Pi0.5 GraphCache invariant failed during close")
            for owner, name, _graph_name, destroy, _attrs, handle, _finalizer, _graph, resource in inventory:
                if handle is not None:
                    if resource is not None:
                        resource.destroy(handle)
                        resource._native_destroyed(handle)
                    else:
                        getattr(torch.ops.rpu, destroy)(handle)
                    vars(owner)[name] = None
        except BaseException as error:
            self._retain_retirement_failure(error)
            raise

        from .runtime import _PI05_HANDLES, _PI05_PREPARED_GRAPH_PROFILES

        for owner, _name, _graph_name, _destroy, attrs, _handle, _finalizer, _graph, _resource in inventory:
            for attr in attrs:
                vars(owner).pop(attr, None)
            if release:
                vars(owner)["forward"] = _closed_pi05_forward
        model = self._lerobot_policy.model
        if release:
            vars(model)["sample_actions"] = _closed_pi05_forward
        _PI05_HANDLES.pop(model, None)
        _PI05_PREPARED_GRAPH_PROFILES.pop(model, None)
        self._retirement_components = ()
        _reset_adapter_runtime_state(self)
        self._closed = True
        if release:
            _release_live_instance(self._lerobot_policy)

    def close(self) -> None:
        """Release resources through the shared stop-the-world lifecycle."""
        if getattr(self, "_closed", False) and not getattr(self, "_retirement_components", ()):
            return
        model = self._lerobot_policy.model
        parent = getattr(model, "_pi05_retirement_owner", None)
        if parent is not None and parent is not self:
            if (not isinstance(parent, Pi05Adapter)
                    or parent._lerobot_policy is not self._lerobot_policy
                    or parent._execution_session is not self._execution_session
                    or not getattr(parent, "_gc_retirement_enabled", False)):
                error = RuntimeError("Pi0.5 close lost its actual retirement owner")
                self._retain_retirement_failure(error)
                raise error
            parent.close()
            _reset_adapter_runtime_state(self)
            self._closed = True
            return
        with self._execution_session._lock:
            if self._execution_session._active:
                raise RuntimeError("Pi0.5 cannot close during a forward")
            try:
                from rpu_backend.api import causal_lm

                if (self._execution_session is not getattr(model, "_execution_session", None)
                        or self._execution_session._owner is not model):
                    raise RuntimeError("Pi0.5 retirement Session identity changed")
                if self._execution_session._poisoned:
                    raise RuntimeError("Pi0.5 cannot retire resources in a poisoned Session")
                with causal_lm._LIVE_LOCK:
                    live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                    if live is not None and live is not self._lerobot_policy:
                        raise RuntimeError("Pi0.5 retirement after live-owner handoff")
                    self._execution_session.shutdown(self._close_without_session)
                if not self._closed:
                    raise RuntimeError("Pi0.5 closed Session still owns native resources")
            except BaseException as error:
                self._retain_retirement_failure(error)
                raise

    def __del__(self):
        if (not getattr(self, "_gc_retirement_enabled", False)
                or getattr(self, "_closed", False)
                or getattr(self, "_retirement_failed", None) is not None):
            return
        try:
            self.close()
        except BaseException as error:
            self._retain_retirement_failure(error)

    def _vlm_model(self):
        try:
            return (
                self._lerobot_policy.model.paligemma_with_expert
                .paligemma.model.language_model
            )
        except AttributeError:
            return None

    def _discover_legacy_vlm_chunk_size(self) -> int:
        vlm = self._vlm_model()
        value = 0 if vlm is None else getattr(vlm, "_rpu_chunk_size", 0)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (value and value % 16)
        ):
            raise ValueError(
                "Pi05Adapter: existing VLM _rpu_chunk_size must be 0 or a "
                f"positive multiple of 16, got {value!r}"
            )
        return int(value)

    def _resolve_execution(self, value, *, entry_point: str):
        return resolve_pi05_execution_components(
            value,
            entry_point=entry_point,
            legacy_prefill_chunk=self._legacy_vlm_chunk_size,
        )

    def _publish_execution_views(
        self, resolved, *, generation: int, component_generations=None,
    ) -> None:
        root, vision, text, action = resolved
        self._rpu_execution = root
        self._vision_execution = vision
        self._vlm_execution = text
        self._action_execution = action
        chunk = text.get("prefill", {}).get("chunk_size", "auto")
        self._vlm_chunk_size = 0 if chunk == "auto" else int(chunk)
        self._execution_generation = int(generation)
        if component_generations is not None:
            self._component_generations = dict(component_generations)
        model = getattr(self._lerobot_policy, "model", None)
        if model is not None:
            vars(model)["_pi05_execution_components"] = {
                PI05_VISION_COMPONENT: vision,
                PI05_TEXT_COMPONENT: text,
                PI05_ACTION_COMPONENT: action,
            }
            vars(model)["_pi05_execution_generation"] = int(generation)
            vars(model)["_pi05_execution_component_generations"] = dict(
                self._component_generations
            )
            try:
                pawe = model.paligemma_with_expert
                vision_tower = pawe.paligemma.model.vision_tower
                children = (
                    (
                        getattr(vision_tower, "vision_model", vision_tower),
                        PI05_VISION_COMPONENT,
                        vision,
                    ),
                    (
                        pawe.paligemma.model.language_model,
                        PI05_TEXT_COMPONENT,
                        text,
                    ),
                    (pawe.gemma_expert.model, PI05_ACTION_COMPONENT, action),
                )
            except AttributeError:
                children = ()
            for child, component, config in children:
                child_state = vars(child)
                child_state["_fmb_execution_component_id"] = component
                child_state["_fmb_execution_component_config"] = config
                child_state["_fmb_execution_generation"] = int(
                    self._component_generations[component]
                )

    def validate_execution_reconfigure(self, execution_config) -> None:
        """Validate one hot geometry generation without mutating runtime."""
        if not self._rpu_is_ready:
            raise RuntimeError(
                "Pi05Adapter.reconfigure requires Pi05Policy.to('rpu') first"
            )
        resolved = self._resolve_execution(
            execution_config,
            entry_point="Pi05Adapter.reconfigure",
        )
        from .cores import core_topology
        if core_topology(resolved[0]) != core_topology(self._rpu_execution):
            raise ValueError("Pi0.5 model.num_cores is cold-only; reload the policy")
        previous = self._rpu_execution
        old_views = self._resolve_execution(previous, entry_point="Pi05Adapter.reconfigure")
        for old, new, stage in zip(old_views[1:], resolved[1:], ("vision", "prefill", "action")):
            if bool(old.get(stage, {}).get("linear_acc32", False)) != bool(new.get(stage, {}).get("linear_acc32", False)):
                raise ValueError(f"Pi0.5 {stage}.linear_acc32 is cold-only; close and reload the policy")
        try:
            self._rpu_execution = resolved[0]
            _validate_execution_geometry(self)
        finally:
            self._rpu_execution = previous
        model = self._lerobot_policy.model
        previous_prefill = getattr(model, "_rpu_last_execution_plan", {}).get("prefill")
        if previous_prefill and resolved[2] != self._vlm_execution:
            from .runtime import _plan_pi05_prefix_execution

            # Read only: do not retire graphs, install cost state or rewrite
            # the live child config to discover an impossible hot request.
            _plan_pi05_prefix_execution(
                model, int(previous_prefill["logical_len"]),
                _validation_request=resolved[2].get("prefill", {}),
            )
        required = (
            "execution_reconfigure_begin",
            "execution_reconfigure_commit",
            "execution_reconfigure_abort",
            "execution_reconfigure_abort_attempt",
        )
        missing = [name for name in required if not hasattr(torch.ops.rpu, name)]
        if missing:
            raise RuntimeError(
                "Pi0.5 binary lacks native execution-reconfigure op(s): "
                + ", ".join(missing)
            )

    @staticmethod
    def _reset_graph_owner(cache, *, label: str) -> None:
        if cache is None:
            return
        cache.begin_warmup()
        cache.clear()
        if not cache.cache_invariant_ok():
            raise RuntimeError(
                f"Pi0.5 {label} GraphCache invariant failed during reconfigure"
            )

    def _reset_prefill_graph_owners(self) -> None:
        """Retire only graphs whose physical identity depends on prefill."""
        model = self._lerobot_policy.model
        pawe = model.paligemma_with_expert
        vlm = pawe.paligemma.model.language_model
        expert = pawe.gemma_expert.model
        self._reset_graph_owner(
            getattr(vlm, "_rpu_gemma_graph_cache", None), label="prefill"
        )
        self._reset_graph_owner(
            getattr(expert, "_rpu_adarms_graph_cache", None),
            label="action safety",
        )
        self._reset_graph_owner(
            getattr(model, "_rpu_fused_denoise_graph_cache", None),
            label="action",
        )
        model_state = vars(model)
        model_state["_rpu_denoise_mask_cache"] = None
        model_state["_pi05_language_prefix_cache"] = None
        model_state["_pi05_prefix_assembly_ok"] = None

    def _reset_vision_graph_owner(self) -> None:
        pawe = self._lerobot_policy.model.paligemma_with_expert
        tower = pawe.paligemma.model.vision_tower
        vision = getattr(tower, "vision_model", tower)
        self._reset_graph_owner(
            getattr(vision, "_rpu_siglip_graph_cache", None), label="vision"
        )

    def _reset_action_graph_owners(self) -> None:
        model = self._lerobot_policy.model
        expert = model.paligemma_with_expert.gemma_expert.model
        self._reset_graph_owner(
            getattr(expert, "_rpu_adarms_graph_cache", None),
            label="action safety",
        )
        self._reset_graph_owner(
            getattr(model, "_rpu_fused_denoise_graph_cache", None),
            label="action",
        )

    def _apply_execution_state(
        self,
        resolved,
        *,
        generation: int,
        component_generations=None,
        force_components=(),
    ) -> None:
        previous = (
            self._vision_execution,
            self._vlm_execution,
            self._action_execution,
        )
        _root, vision, text, action = resolved
        current = (vision, text, action)
        component_ids = (
            PI05_VISION_COMPONENT,
            PI05_TEXT_COMPONENT,
            PI05_ACTION_COMPONENT,
        )
        changed = {
            component
            for component, before, after in zip(component_ids, previous, current)
            if before != after
        }
        changed.update(force_components)
        if component_generations is None:
            component_generations = dict(self._component_generations)
            for component in changed:
                component_generations[component] += 1
        vlm = self._vlm_model()
        if vlm is None:
            raise RuntimeError("Pi0.5 VLM runtime is unavailable")
        chunk = text.get("prefill", {}).get("chunk_size", "auto")
        vars(vlm)["_rpu_chunk_size"] = 0 if chunk == "auto" else int(chunk)
        if PI05_TEXT_COMPONENT in changed:
            self._reset_prefill_graph_owners()
        if PI05_VISION_COMPONENT in changed:
            self._reset_vision_graph_owner()
        if (
            PI05_ACTION_COMPONENT in changed
            and PI05_TEXT_COMPONENT not in changed
        ):
            self._reset_action_graph_owners()
        from .runtime import _PI05_PREPARED_GRAPH_PROFILES

        _PI05_PREPARED_GRAPH_PROFILES.pop(self._lerobot_policy.model, None)
        vars(self._lerobot_policy.model)["_rpu_last_execution_plan"] = {}
        self._publish_execution_views(
            resolved,
            generation=generation,
            component_generations=component_generations,
        )

    def apply_execution_reconfigure(
        self, old_config, new_config, generation: int, *, force_rebuild=False
    ) -> None:
        """Apply one quiescent generation under the native STW claim."""
        self._execution_reconfigure_journal = {"mutation_started": False}
        old_resolved = self._resolve_execution(
            old_config, entry_point="Pi05Adapter.reconfigure rollback"
        )
        new_resolved = self._resolve_execution(
            new_config, entry_point="Pi05Adapter.reconfigure"
        )
        self._execution_reconfigure_journal.update({
            "resolved": old_resolved,
            "generation": self._execution_generation,
            "component_generations": dict(self._component_generations),
            "changed": {
                component
                for component, old, new in zip(
                    (
                        PI05_VISION_COMPONENT,
                        PI05_TEXT_COMPONENT,
                        PI05_ACTION_COMPONENT,
                    ),
                    old_resolved[1:],
                    new_resolved[1:],
                )
                if old != new
            },
        })
        if force_rebuild:
            self._execution_reconfigure_journal["changed"].update((
                PI05_VISION_COMPONENT, PI05_TEXT_COMPONENT, PI05_ACTION_COMPONENT,
            ))
        with native_execution_reconfigure(torch.ops.rpu):
            self._execution_reconfigure_journal["mutation_started"] = True
            self._apply_execution_state(
                new_resolved, generation=generation,
                force_components=self._execution_reconfigure_journal["changed"],
            )
        # Retain undo state through the shared session's config publication.

    def rollback_execution_reconfigure(
        self, old_config, _new_config, generation: int
    ) -> None:
        """Restore prior geometry; failure poisons the shared session."""
        journal = self._execution_reconfigure_journal
        if journal is not None and not journal.get("mutation_started", False):
            self._execution_reconfigure_journal = None
            return
        if journal is None:
            resolved = self._resolve_execution(
                old_config, entry_point="Pi05Adapter.reconfigure rollback"
            )
            force_components = {
                component
                for component, current, previous in zip(
                    (
                        PI05_VISION_COMPONENT,
                        PI05_TEXT_COMPONENT,
                        PI05_ACTION_COMPONENT,
                    ),
                    (
                        self._vision_execution,
                        self._vlm_execution,
                        self._action_execution,
                    ),
                    resolved[1:],
                )
                if current != previous
            }
            component_generations = dict(self._component_generations)
        else:
            resolved = journal["resolved"]
            generation = journal["generation"]
            force_components = journal["changed"]
            component_generations = journal["component_generations"]
        with native_execution_reconfigure(torch.ops.rpu):
            self._apply_execution_state(
                resolved,
                generation=generation,
                component_generations=component_generations,
                force_components=force_components,
            )
        self._execution_reconfigure_journal = None

    @execution_serialized
    @torch.no_grad()
    def prepare_graphs(
        self,
        batch: dict,
        *,
        num_steps: "int | None" = None,
        precompute_adarms: bool = False,
        noise: torch.Tensor | None = None,
    ) -> dict:
        """Prebuild the finite Pi0.5 production signature and enter READY."""
        if not self._rpu_is_ready:
            raise RPUBackendError(
                "Pi05Adapter.prepare_graphs requires Pi05Policy.to('rpu') first."
            )

        pi05_pytorch = self._lerobot_policy.model
        from .runtime import (
            _PI05_PREPARED_GRAPH_PROFILES,
            _pi05_graph_runtime_profile,
            _validate_fused_parity,
        )

        if pi05_pytorch in _PI05_PREPARED_GRAPH_PROFILES:
            raise RuntimeError(
                "Pi0.5 graphs are already prepared and frozen for this policy."
            )

        steps = (
            pi05_pytorch.config.num_inference_steps
            if num_steps is None
            else num_steps
        )
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError(
                f"num_steps must be a positive integer, got {steps!r}"
            )
        if type(precompute_adarms) is not bool:
            raise TypeError("precompute_adarms must be bool")

        runtime_profile = _pi05_graph_runtime_profile(pi05_pytorch, steps)
        required_graph_modes = (
            "siglip_graph",
            "gemma_graph",
            "fused_denoise",
            "denoise_graph",
        )
        disabled = [
            name for name in required_graph_modes
            if not runtime_profile[name]
        ]
        if disabled:
            raise RuntimeError(
                "Pi0.5 prepare_graphs requires the production fused graph path; "
                f"disabled modes: {disabled}"
            )

        pawe = pi05_pytorch.paligemma_with_expert
        vision = pawe.paligemma.model.vision_tower
        vision = vision.vision_model if hasattr(vision, "vision_model") else vision
        vlm = pawe.paligemma.model.language_model
        expert = pawe.gemma_expert.model
        table_metadata = None
        if precompute_adarms:
            handle = pi05_pytorch._rpu_fused_denoise_handle
            if (not runtime_profile["denoise_unroll"]
                    or torch.ops.rpu.pi05_denoise_step_get_resolved_chunk_size(handle) != 0):
                raise RuntimeError(
                    "precompute_adarms requires in-graph unroll and a fresh policy before inference")
            from .weights import _precompute_adarms_cond_all, _precompute_adarms_table
            # Read the installed final dense weight, independent of optional
            # scale tables appended to the native set_weights signature.
            installed_weights = pi05_pytorch._rpu_fused_denoise_retirement_state.keepalive[0]
            dense_w8a16 = installed_weights[11].dtype == torch.int8
            table = _precompute_adarms_table(
                expert, _precompute_adarms_cond_all(pi05_pytorch, steps),
                dense_w8a16=dense_w8a16)
            torch.ops.rpu.pi05_denoise_step_set_adarms_table(handle, table.to("rpu"))
            table_metadata = dict(shape=list(table.shape), bytes=table.numel() * 2,
                                  dense_w8a16=dense_w8a16, accumulation="cpu_fp32")
        caches = [
            ("vision", vision._rpu_siglip_graph_cache),
            ("prefill", vlm._rpu_gemma_graph_cache),
            ("denoise", pi05_pytorch._rpu_fused_denoise_graph_cache),
        ]

        # The first fused call also runs the legacy AdaRMS path once as a
        # correctness safety net. If that safety net still needs to run, make
        # its graph part of the same finite transaction.
        safety_adarms = (
            not getattr(pi05_pytorch, "_rpu_fuse_validated", False)
            and runtime_profile["adarms_graph"]
        )
        clear_cache_names = {"vision", "prefill", "denoise"}
        safety_cache = expert._rpu_adarms_graph_cache
        if safety_adarms:
            caches.append(("safety_adarms", safety_cache))
            clear_cache_names.add("safety_adarms")
        elif safety_cache.size() > 0:
            # prepare_graphs() is also valid after an exploratory forward. The
            # one-shot safety graph is no longer on the production hot path,
            # but freeze the already-built entry so the policy has no
            # build-capable GraphCache left behind.
            caches.append(("safety_adarms", safety_cache))

        for name, cache in caches:
            cache.begin_warmup()
            if name in clear_cache_names:
                cache.clear()

        _PI05_PREPARED_GRAPH_PROFILES[pi05_pytorch] = {
            "runtime": dict(runtime_profile),
        }

        def _stats(cache) -> dict:
            snapshots = cache.snapshot()
            return {
                "entries": int(cache.size()),
                "replays": sum(
                    int(getattr(entry, "replay_count", 0))
                    for entry in snapshots
                ),
                "recaptures": sum(
                    int(getattr(entry, "recapture_count", 0))
                    for entry in snapshots
                ),
                "invariant_ok": bool(cache.cache_invariant_ok()),
            }

        try:
            build_value = self._lerobot_policy.predict_action_chunk(
                batch, num_steps=steps, **({"noise": noise} if noise is not None else {})
            )
            if not isinstance(build_value, torch.Tensor):
                raise RuntimeError(
                    "Pi0.5 prepare_graphs expected predict_action_chunk to "
                    f"return Tensor, got {type(build_value).__name__}"
                )
            build_action = build_value.detach().cpu().clone()

            graph_counts = {
                name: int(cache.size())
                for name, cache in caches
                if cache.size() > 0
            }
            mandatory_counts = {
                name: graph_counts.get(name, 0)
                for name in ("vision", "prefill", "denoise")
            }
            expected_counts = {"vision": 1, "prefill": 1, "denoise": 1}
            if mandatory_counts != expected_counts:
                raise RuntimeError(
                    "Pi0.5 graph prewarm produced "
                    f"{mandatory_counts}, expected {expected_counts}"
                )

            frozen_caches = [
                (name, cache)
                for name, cache in caches
                if cache.size() > 0
            ]
            # Discard the RECORDING result and exercise one WARMING replay
            # before freezing. This primes every mutable DMA base under the
            # exact production call without admitting a new signature.
            before_warming_replay = {
                name: _stats(cache) for name, cache in frozen_caches
            }
            warm_value = self._lerobot_policy.predict_action_chunk(
                batch, num_steps=steps, **({"noise": noise} if noise is not None else {})
            )
            warm_action = warm_value.detach().cpu().clone()
            after_warming_replay = {
                name: _stats(cache) for name, cache in frozen_caches
            }
            before_warming_invariants = {
                name: (stats["entries"], stats["recaptures"])
                for name, stats in before_warming_replay.items()
            }
            after_warming_invariants = {
                name: (stats["entries"], stats["recaptures"])
                for name, stats in after_warming_replay.items()
            }
            if after_warming_invariants != before_warming_invariants:
                raise RuntimeError(
                    "Pi0.5 WARMING replay grew or recaptured a graph: "
                    f"before={before_warming_replay}, "
                    f"after={after_warming_replay}"
                )

            # Check the public CPU output boundary and record replay differences.
            # Graph counts, recapture and invariants remain hard requirements.
            # Floating-point metrics and exact equality are diagnostics.
            build_replay_bit_equal = bool(torch.equal(build_action, warm_action))
            build_replay_parity = _validate_fused_parity(
                warm_action,
                build_action,
                where="Pi05 WARMING replay parity",
                expected_shape=None,
            )
            build_replay_max_abs = build_replay_parity["max_abs"]
            for _, cache in frozen_caches:
                cache.freeze()

            before_ready = {
                name: _stats(cache) for name, cache in frozen_caches
            }
            ready_value = self._lerobot_policy.predict_action_chunk(
                batch, num_steps=steps, **({"noise": noise} if noise is not None else {})
            )
            ready_action = ready_value.detach().cpu().clone()
            after_ready = {
                name: _stats(cache) for name, cache in frozen_caches
            }
            before_invariants = {
                name: (stats["entries"], stats["recaptures"])
                for name, stats in before_ready.items()
            }
            after_invariants = {
                name: (stats["entries"], stats["recaptures"])
                for name, stats in after_ready.items()
            }
            if after_invariants != before_invariants:
                raise RuntimeError(
                    "Pi0.5 READY replay grew or recaptured a graph: "
                    f"before={before_ready}, after={after_ready}"
                )
            mandatory_ready = {
                name: after_ready.get(name, {})
                for name in ("vision", "prefill", "denoise")
            }
            if not all(
                int(stats.get("replays", 0)) >= 2
                for stats in mandatory_ready.values()
            ):
                raise RuntimeError(
                    "Pi0.5 READY mandatory graph did not complete two "
                    f"replays: {mandatory_ready}"
                )
            if not all(
                int(stats.get("entries", 0)) == 1
                and int(stats.get("recaptures", -1)) == 0
                and stats.get("invariant_ok") is True
                for stats in after_ready.values()
            ):
                raise RuntimeError(
                    "Pi0.5 READY graph cache invariant failed: "
                    f"{after_ready}"
                )
            ready_replay_bit_equal = bool(torch.equal(warm_action, ready_action))
            ready_replay_parity = _validate_fused_parity(
                ready_action,
                warm_action,
                where="Pi05 READY replay parity",
                expected_shape=None,
            )
        except Exception:
            for _, cache in caches:
                cache.begin_warmup()
            _PI05_PREPARED_GRAPH_PROFILES.pop(pi05_pytorch, None)
            raise

        result = {
            "phase": "READY",
            "num_steps": steps,
            "adarms_table": table_metadata,
            "graph_counts": graph_counts,
            "runtime": dict(runtime_profile),
            "ready_replay_validated": True,
            "action_bit_equal": ready_replay_bit_equal,
            "build_replay_bit_equal": build_replay_bit_equal,
            "replay_parity": {
                "build_vs_warming": build_replay_parity,
                "warming_vs_ready": ready_replay_parity,
                "contract": "pi05-runtime-boundary-and-numeric-diagnostics-v2",
                "numerical_status": "DIAGNOSTIC",
                "task_quality": "NOT_EVALUATED",
            },
            "build_replay_max_abs": build_replay_max_abs,
            "ready_stats": after_ready,
        }
        _PI05_PREPARED_GRAPH_PROFILES[pi05_pytorch] = result
        return dict(result)

    def _install_fused_denoise_handle(self, *, runtime_policy=None) -> None:
        """Build a Pi05DenoiseStepModel handle and bind weights.

        Called from to_rpu() AFTER Step G (after the expert is swizzled and
        per-layer norm._rpu_dense_w_rp exist). Skipped (handle stays None) when
        RPU_PI05_FUSED_DENOISE=0.

        CRITICAL OBJECT PATH (Rev 3 §C13/C26 round 2 fix):
          The patched sample_actions is bound to `PI05Pytorch.sample_actions`
          (runtime.py:432). Inside that patched body, `self` is a PI05Pytorch
          instance — i.e. `self._lerobot_policy.model` from this adapter's
          POV. The fused handle MUST be stored on that PI05Pytorch so the
          patched dispatcher can find it via plain `getattr(self, ...)`.

          Existing precedent: _PI05_HANDLES is keyed on
          `self._lerobot_policy.model` at adapters/pi05/__init__.py:262-268.
        """
        from rpu_backend.api._execution import _require_execution_process_safe
        from rpu_backend.runtime._native_retirement import _InstalledNativeResource
        from rpu_backend.adapters.pi05.weights import (
            _prepare_final_norm_weights, _swizzle_action_proj_weights,
        )

        pi05_pytorch = self._lerobot_policy.model           # PI05Pytorch
        _require_execution_process_safe()
        old_resource = getattr(pi05_pytorch, "_rpu_fused_denoise_retirement_state", None)
        if old_resource is not None:
            old_resource.require_replaceable()
        elif getattr(pi05_pytorch, "_rpu_fused_denoise_handle", None) is not None:
            raise RuntimeError("Pi0.5 denoise replacement requires its actual retirement resource")
        model_state = vars(pi05_pytorch)
        install_snapshot = {
            name: model_state.get(name, _MISSING)
            for name in _FUSED_DENOISE_INSTALL_ATTRS
        }
        old_finalizer = install_snapshot[
            "_rpu_fused_denoise_handle_finalizer"]
        if old_finalizer is _MISSING:
            old_finalizer = None
        denoise_graph_enabled = install_snapshot[
            "_pi05_denoise_graph_enabled"]
        if denoise_graph_enabled is _MISSING:
            denoise_graph_enabled = bool(_denoise_graph_enabled())
        denoise_unroll_enabled = install_snapshot[
            "_pi05_denoise_unroll_enabled"]
        if denoise_unroll_enabled is _MISSING:
            denoise_unroll_enabled = bool(_denoise_unroll_enabled())
        adapter_state = vars(self)
        old_adapter_finalizers = adapter_state.get("_finalizers", _MISSING)

        def restore_publication() -> None:
            state = vars(pi05_pytorch)
            for name, value in install_snapshot.items():
                if value is _MISSING:
                    state.pop(name, None)
                else:
                    state[name] = value
            if old_adapter_finalizers is _MISSING:
                vars(self).pop("_finalizers", None)
            else:
                vars(self)["_finalizers"] = old_adapter_finalizers

        if not _fused_denoise_enabled():
            committed = False
            try:
                # Publish the explicit opt-out completion state before touching
                # the old graph/handle, so a Python publication interruption is
                # fully reversible.
                pi05_pytorch._rpu_fused_denoise_handle = None
                pi05_pytorch._rpu_fused_denoise_handle_finalizer = None
                pi05_pytorch._rpu_fused_denoise_retirement_state = None
                pi05_pytorch._rpu_fused_denoise_graph_cache = None
                pi05_pytorch._pi05_denoise_graph_enabled = denoise_graph_enabled
                pi05_pytorch._pi05_denoise_unroll_enabled = denoise_unroll_enabled
                self._finalizers = [
                    finalizer
                    for finalizer in self._finalizers
                    if finalizer is not old_finalizer and finalizer.alive
                ]

                if old_resource is not None:
                    old_resource.retire()
                committed = True
            except BaseException:
                if not committed:
                    restore_publication()
                raise

            if old_finalizer is not None and old_finalizer.alive:
                old_finalizer.detach()
            return

        pawe   = pi05_pytorch.paligemma_with_expert
        expert = pawe.gemma_expert.model
        # Denoising shares the prefix VLM cache, not the expert's private cache.
        prefix_cache = pawe.paligemma.model.language_model._rpu_cache
        ec     = expert.config
        effective_expert_nkv = int(getattr(
            expert, "_rpu_effective_num_kv_heads", ec.num_key_value_heads
        ))

        # Swizzle final_norm + action projections (idempotent under sentinel).
        from .cores import model_topology, graph_cache_options, validate_native_topology
        topology = model_topology(expert)
        _prepare_final_norm_weights(expert, num_cores=topology.attn_tp)
        _swizzle_action_proj_weights(pi05_pytorch.action_in_proj,
                                      pi05_pytorch.action_out_proj)

        # Reuse cos/sin stashed by patch_gemma_model_for_rpu_all_layers_once
        # (gemma.py:model._gemma_rope_cos/sin). RPU DDR is full after all weights
        # load; allocating 2×1 MB new cos/sin tensors fails with std::bad_alloc.
        # The stashed tensors are the same data already held by gemma_set_weights.
        cos_c = getattr(expert, "_gemma_rope_cos", None)
        sin_c = getattr(expert, "_gemma_rope_sin", None)
        if cos_c is None or sin_c is None:
            raise RuntimeError(
                "_install_fused_denoise_handle: expert missing _gemma_rope_cos/"
                "_gemma_rope_sin — patch_gemma_model_for_rpu_all_layers_once must "
                "run before _install_fused_denoise_handle."
            )

        # Gather weights.
        layers = expert.layers
        L = len(layers)
        H = pi05_pytorch.action_in_proj.out_features
        q_w  = [l.self_attn.q_proj.weight  for l in layers]
        k_w  = [l.self_attn.k_proj.weight  for l in layers]
        v_w  = [l.self_attn.v_proj.weight  for l in layers]
        o_w  = [l.self_attn.o_proj.weight  for l in layers]
        gate = [l.mlp.gate_proj.weight     for l in layers]
        up   = [l.mlp.up_proj.weight       for l in layers]
        down = [l.mlp.down_proj.weight     for l in layers]
        attn_w = [l.input_layernorm._rpu_dense_w_rp        for l in layers]
        attn_b = [l.input_layernorm._rpu_dense_b_rp        for l in layers]
        mlp_w  = [l.post_attention_layernorm._rpu_dense_w_rp for l in layers]
        mlp_b  = [l.post_attention_layernorm._rpu_dense_b_rp for l in layers]
        from rpu_backend.adapters.pi05.w8a16 import pi05_expert_scale_lists, pi05_nvfp4_tensor_scale_tables
        scale_lists = pi05_expert_scale_lists(expert, require_rpu=True)

        # AdaRMS dense W8A16: quantize the (fp16) dense modulation weights on the
        # fly when the model is w8a16. The conditioning GEMV is M=1 and purely
        # weight-bandwidth-bound, so int8 weights roughly halve it. Independent
        # of the decoder-projection scales above (has_dense_scale is separate in
        # C++). Defaults on for w8a16; opt out with RPU_PI05_ADARMS_DENSE_W8A16=0.
        # Quantizes the original dense.weight and re-swizzles int8 with the SAME
        # row-partition layout as the fp16 _rpu_dense_w_rp buffers (which are left
        # intact, so the standalone AdaRMS path is unaffected).
        _is_w8a16 = bool(scale_lists and scale_lists[0])
        _dense_w8a16 = rpu_env_bool(
            "RPU_PI05_ADARMS_DENSE_W8A16", default=_is_w8a16)
        final_dense_w = expert.norm._rpu_dense_w_rp
        attn_dense_ws, mlp_dense_ws = [], []
        final_dense_ws = expert.norm._rpu_dense_w_rp.new_empty(0)
        if _dense_w8a16:
            from rpu_backend.quant._common import quantize_linear_per_channel
            from rpu_backend.runtime.weights import (
                transform_linear_weight as _tlw)

            def _q_dense(dense):
                orig = dense.weight.detach().cpu().to(torch.float16).contiguous()
                w8, sc = quantize_linear_per_channel(orig)
                w_rp = _tlw(w8.contiguous(), partition=0,
                            num_cores=topology.attn_tp).to('rpu').contiguous()
                return w_rp, sc.to(torch.float16).to('rpu').contiguous()

            _aw = [_q_dense(l.input_layernorm.dense) for l in layers]
            _mw = [_q_dense(l.post_attention_layernorm.dense) for l in layers]
            attn_w = [p[0] for p in _aw]; attn_dense_ws = [p[1] for p in _aw]
            mlp_w  = [p[0] for p in _mw]; mlp_dense_ws  = [p[1] for p in _mw]
            final_dense_w, final_dense_ws = _q_dense(expert.norm.dense)

        # Prepare GraphCache before creating a native handle. A cache allocation
        # failure must not orphan a native entry or overwrite a live install.
        import rpu_backend
        denoise_graph_cache = (
            rpu_backend.graph.GraphCache(**graph_cache_options(topology.num_cores))
            if runtime_policy is None
            else rpu_backend.graph.GraphCache(runtime_policy=runtime_policy)
        )

        set_weights_args = (
            q_w, k_w, v_w, o_w, gate, up, down,
            attn_w, attn_b, mlp_w, mlp_b,
            final_dense_w, expert.norm._rpu_dense_b_rp,
            pi05_pytorch.action_in_proj._rpu_w_num_cores_1,
            pi05_pytorch.action_in_proj._rpu_bias,
            pi05_pytorch.action_out_proj._rpu_w_num_cores_1,
            pi05_pytorch.action_out_proj._rpu_bias,
            cos_c, sin_c, H,
            pi05_pytorch.config.max_action_dim,
            pi05_pytorch.config.chunk_size,
            ec.num_attention_heads, effective_expert_nkv, ec.head_dim,
            L, float(ec.rms_norm_eps), *scale_lists,
            attn_dense_ws, mlp_dense_ws, final_dense_ws, pi05_nvfp4_tensor_scale_tables(expert))
        resource = _InstalledNativeResource(
            pi05_pytorch, None, _destroy_fused_denoise_handle,
            graphs=(denoise_graph_cache,), keepalive=(set_weights_args, prefix_cache),
            label="Pi0.5 fused denoise", handle_name="_rpu_fused_denoise_handle")
        handle_finalizer = resource.finalizer
        committed = False
        # Schema arg order (Task 1.5): (... hidden_size, max_action_dim,
        # chunk_size, num_q_heads, num_kv_heads, head_dim, num_layers, eps).
        # Rev 3 §C26 fix: snippet now matches verbatim.
        _action_execution = getattr(self, "_action_execution", self._rpu_execution)
        try:
            resource.handle = handle = torch.ops.rpu.pi05_denoise_step_create(
                bool(_action_execution.get("action", {}).get("linear_acc32", False)))
            if topology.num_cores != 8:
                torch.ops.rpu.pi05_denoise_step_set_execution_core_count(handle, topology.num_cores)
            torch.ops.rpu.pi05_denoise_step_set_weights(handle, *set_weights_args)
            validate_native_topology("pi05_denoise_step", handle, topology, ec.intermediate_size, gate[0].dtype)
            torch.ops.rpu.pi05_denoise_step_set_chunk_envelope(
                handle, int(prefix_cache.max_seq_len), 0)
            _action_chunk = _action_execution.get(
                "action", {}
            ).get("chunk_size", "auto")
            torch.ops.rpu.pi05_denoise_step_set_chunk_size(
                handle,
                0 if _action_chunk == "auto" else int(_action_chunk),
            )

            pi05_pytorch._rpu_fused_denoise_handle = handle
            pi05_pytorch._rpu_fused_denoise_retirement_state = resource
            pi05_pytorch._rpu_fused_denoise_handle_finalizer = handle_finalizer
            pi05_pytorch._rpu_fused_denoise_graph_cache = denoise_graph_cache
            pi05_pytorch._pi05_denoise_graph_enabled = denoise_graph_enabled
            pi05_pytorch._pi05_denoise_unroll_enabled = denoise_unroll_enabled
            self._finalizers = [
                finalizer
                for finalizer in self._finalizers
                if finalizer is not old_finalizer and finalizer.alive
            ] + [handle_finalizer]

            if old_resource is not None:
                old_resource.retire()
            committed = True
        except BaseException as error:
            if not committed:
                cleanup_ok = resource.cleanup_failure(error, pi05_pytorch, install_snapshot)
                if resource.handle is None and handle_finalizer.alive:
                    handle_finalizer.detach()
                if cleanup_ok:
                    restore_publication()
                else:
                    vars(pi05_pytorch).update(
                        _rpu_fused_denoise_handle=resource.handle,
                        _rpu_fused_denoise_handle_finalizer=handle_finalizer,
                        _rpu_fused_denoise_retirement_state=resource,
                        _rpu_fused_denoise_graph_cache=denoise_graph_cache)
            raise

        if old_finalizer is not None and old_finalizer.alive:
            old_finalizer.detach()

    def to_rpu(self) -> Any:
        """Install all policy subsystems under one swizzle lock."""
        if self._closed:
            raise RPUBackendError(
                "Pi05Adapter.to_rpu(): this policy is closed; reload it"
            )
        # HIGH-3 idempotent re-entry.
        state = _pi05_runtime_state(self._lerobot_policy)
        if state is not None:
            _sync_adapter_runtime_state(self, state)
            return self._lerobot_policy
        if self._rpu_is_ready or getattr(
            self._lerobot_policy, "_rpu_swizzled", False
        ):
            _reset_adapter_runtime_state(self)
            raise RPUBackendError(
                "Pi05Adapter.to_rpu(): the ready marker exists but composite "
                "runtime ownership is incomplete; reload the policy."
            )

        # Round-5 HIGH #2: partial-failure retry guard.
        if getattr(self._lerobot_policy, "_rpu_swizzle_started", False):
            raise RPUBackendError(
                "Pi05Adapter.to_rpu(): prior swizzle attempt failed mid-way; "
                "submodel weights in undefined state. Reload via "
                "Pi05Policy.from_pretrained(...).")

        prefill_pair_rows = _selected_prefill_pair_rows(self)
        _validate_execution_geometry(self)
        from .cores import core_topology, validate_reduced_policy, pad_decoder_mlp_inplace
        topology = core_topology(self._rpu_execution)
        validate_reduced_policy(self._lerobot_policy, topology)

        # Reject the retired mixed CPU/RPU scaffold before claiming the live
        # instance, installing patches, or irreversibly transforming weights.
        _keep_cpu_raw = os.environ.get("RPU_PI05_KEEP_CPU", "").strip()
        if _keep_cpu_raw:
            raise RuntimeError(
                "RPU_PI05_KEEP_CPU is a v3.0 bisection scaffold; "
                "it is not supported under v4.0 fused-decoder path. "
                "Set RPU_PI05_KEEP_CPU= (empty) to use the canonical v4 path. "
                "Mixed CPU/RPU bisection is deferred to v4.1 or later."
            )

        if not _SWIZZLE_LOCK.acquire(blocking=False):
            raise RPUBackendError(
                "Pi05Adapter.to_rpu(): another Pi0.5 swizzle in progress (Q7).")
        try:
            # Re-check all ownership under the lock before preinstall rejects
            # the completed runtime's internal attributes as stale state.
            state = _pi05_runtime_state(self._lerobot_policy)
            if state is not None:
                _sync_adapter_runtime_state(self, state)
                return self._lerobot_policy
            if self._rpu_is_ready or getattr(
                self._lerobot_policy, "_rpu_swizzled", False
            ):
                _reset_adapter_runtime_state(self)
                raise RPUBackendError(
                    "Pi05Adapter.to_rpu(): the ready marker exists but "
                    "composite runtime ownership is incomplete; reload the "
                    "policy."
                )
            if getattr(self._lerobot_policy, "_rpu_swizzle_started", False):
                raise RPUBackendError(
                    "Pi05Adapter.to_rpu(): prior swizzle attempt failed "
                    "mid-way; reload the policy."
                )

            # D-32 A9 wire-point #1 (PRE-install scan).
            from rpu_backend.runtime.hw_attrs import validate_preinstall
            validate_preinstall(self._lerobot_policy)

            self._validate_cold_pad16_requests()
            cold_model = _preflight_pi05_cold_model(self._lerobot_policy, attn_tp=topology.attn_tp)
            graph_runtime_policy = _pi05_graph_runtime_policy()
            if topology.num_cores != 8:
                graph_runtime_policy = replace(
                    graph_runtime_policy, execution_core_count=topology.num_cores,
                    provenance=graph_runtime_policy.provenance +
                    (f"owner:pi05:execution_core_count={topology.num_cores}",),
                )

            _claim_live_instance(self._lerobot_policy)  # WARNING 14 ordering
            try:
                # Pi retains thousands of weights and request buffers. Reuse
                # their HostDDR mappings through the common, immutable tensor
                # allocator policy before the first RPU tensor is materialized.
                torch.rpu.set_caching_allocator(True)
                if not graph_runtime_policy.prepare_arenas():
                    raise RPUBackendError("Pi0.5 requires its cold Graph arena plan")
            except BaseException:
                _release_live_instance(self._lerobot_policy)
                raise

            # CR-R5 BLOCKER 4: outer transaction from the mutation marker to
            # Step G. Once Step C starts, the broken owner may still hold RPU
            # tensors after native cleanup; keep the slot until owner GC.
            release_live_on_failure = True
            pi05_pytorch = None
            vlm = None
            expert = None
            siglip_inner = None
            try:
                self._lerobot_policy._rpu_swizzle_started = True
                self._publish_cold_pad16_requests()

                # Step A: class-level patches.
                from .patches import install_class_patches
                install_class_patches(self._lerobot_policy)

                (
                    pi05_pytorch,
                    pawe,
                    paligemma_model,
                    vlm,
                    expert,
                    siglip_inner,
                    projector,
                    expected_vlm_effective_nkv,
                    expected_expert_effective_nkv,
                    _int4,
                    siglip_w8a16_default,
                ) = cold_model

                # SigLIP W8A16 defaults ON for the w8a16 path (the VLM gemma is
                # loaded as int8), OFF for fp16. RPU_PI05_SIGLIP_W8A16 overrides
                # either way. Probed from the VLM q_proj dtype (int8 ⇒ w8a16
                # bundle); read before convert_linear_weights swizzles (which
                # preserves the int8 dtype, so the probe is order-independent).
                # Per-dtype best-perf switch defaults — one VLM-dtype probe makes
                # the chosen dtype the only knob for optimal Pi05 inference:
                #   int8 (w8a16) / uint8 (w4a16 int4-packed) -> quantized VLM ->
                #     SigLIP-W8A16 is the best default (int8 vision Linears).
                #   fp16 -> SigLIP stays fp16 and uses the generator-selected
                #     FP16 ACC16 auto-tile family.
                # Each child's cold linear_acc32 config chooses ACC16 (default)
                # or ACC32. KV-insert-v16 is C++ default-ON; reductions retain
                # the generator-scheduled ring path.
                # Fast replay remains a per-GraphCache choice.  The frozen Pi
                # policy defaults these three generic execution capabilities on
                # while preserving explicit environment A/B overrides; it does
                # not change defaults seen by GraphCaches created for other
                # models later in this process.
                # Unroll denoise into one graph with on-device FP16 Euler,
                # avoiding per-step CPU/RPU round-trips. Preserve an explicit
                # RPU_PI05_DENOISE_UNROLL=0 opt-out.
                os.environ.setdefault("RPU_PI05_DENOISE_UNROLL", "1")

                # --- Step C: subsystem swizzles (Pitfall 1 belt on Expert) ---
                release_live_on_failure = False
                # The fused path transforms all three subsystems through the
                # shared runtime weight utilities.
                from rpu_backend.runtime import weights as _mc

                from rpu_backend.adapters.pi05.w8a16 import (
                    replicate_pi05_decoder_kv_heads_for_attention_tp,
                )
                vlm_effective_nkv = replicate_pi05_decoder_kv_heads_for_attention_tp(
                    vlm, effective_num_kv_heads=topology.attn_tp, label="vlm"
                )
                expert_effective_nkv = replicate_pi05_decoder_kv_heads_for_attention_tp(
                    expert, effective_num_kv_heads=topology.attn_tp, label="expert"
                )
                if (
                    vlm_effective_nkv != expected_vlm_effective_nkv
                    or expert_effective_nkv != expected_expert_effective_nkv
                ):
                    raise RPUBackendError(
                        "Pi05Adapter.to_rpu(): KV replication result changed "
                        "after cold preflight"
                    )

                from rpu_backend.adapters.pi05.w4pack import (
                    int4_child_names, pack_int4_projections_inplace,
                )
                # loader.py sets `<lerobot_policy>._pi05_quant_config`; the adapter
                # holds that lerobot policy as self._lerobot_policy.
                _quant_cfg = getattr(self._lerobot_policy, "_pi05_quant_config", {}) or {}
                _vlm_int4 = int4_child_names(_quant_cfg, component="vlm")
                _expert_int4 = int4_child_names(_quant_cfg, component="expert")
                _int4_group_size = _quant_cfg.get("int4_group_size", _quant_cfg.get("group_size", 32))

                pad_decoder_mlp_inplace(vlm, topology)
                pad_decoder_mlp_inplace(expert, topology)
                vlm._rpu_decoder_topology = topology
                expert._rpu_decoder_topology = topology
                vlm_attn_tp = topology.attn_tp
                _mc.convert_linear_weights_inplace(
                    vlm, attn_num_cores=vlm_attn_tp, skip_names=set(_vlm_int4),
                    mlp_num_cores=topology.mlp_tp, lm_head_num_cores=topology.attn_tp,
                    execution_core_count=topology.num_cores)
                if _vlm_int4:
                    pack_int4_projections_inplace(vlm, _vlm_int4, attn_num_cores=vlm_attn_tp, group_size=_int4_group_size)

                expert_attn_tp = topology.attn_tp
                _mc.convert_linear_weights_inplace(
                    expert, skip_names={'dense'} | set(_expert_int4), attn_num_cores=expert_attn_tp,
                    mlp_num_cores=topology.mlp_tp, lm_head_num_cores=topology.attn_tp,
                    execution_core_count=topology.num_cores)
                if _expert_int4:
                    pack_int4_projections_inplace(expert, _expert_int4, attn_num_cores=expert_attn_tp, group_size=_int4_group_size)

                # SigLIP projector swizzle — delegate to weights.py
                # (preserves Pitfall-1 belt via _rpu_weights_converted sentinel).
                from .weights import swizzle_projector_inplace
                swizzle_projector_inplace(projector, num_cores=topology.attn_tp)

                # v5-05 D-01: SigLIP encoder swizzle runs on CPU before move-to-RPU.
                # Helper lifted from _internal/patches/siglip_pi05.py to
                # adapters/pi05/siglip.py (Task 2). Idempotence guard on
                # `vit._rpu_siglip_weights_converted` short-circuits the inner call
                # inside patch_siglip_model_for_rpu_all_layers_once (siglip.py:108).
                from rpu_backend.adapters.siglip import _convert_siglip_weights_for_rpu
                _convert_siglip_weights_for_rpu(
                    siglip_inner, w8a16_default=siglip_w8a16_default,
                    num_cores=topology.attn_tp)

                # --- Step D: move subsystems to RPU ---
                vlm._rpu_attn_tp = vlm_attn_tp
                vlm._rpu_max_seq_len = 2048
                vlm.to('rpu')
                # Iteration 6: embed_tokens MUST stay on CPU.
                if getattr(vlm, "embed_tokens", None) is not None:
                    vlm.embed_tokens = vlm.embed_tokens.to('cpu')

                expert._rpu_attn_tp = expert_attn_tp
                expert._rpu_max_seq_len = 2048
                expert.to('rpu')

                # BLOCKER 4: in-place .to('cpu').
                if getattr(expert, "embed_tokens", None) is not None:
                    expert.embed_tokens.to('cpu')
                if getattr(expert, "norm", None) is not None:
                    expert.norm.to('cpu')

                # Iteration 7: Move ONLY encoder + post_layernorm + projector to RPU;
                # keep vit.embeddings on CPU (match legacy pi05_converter.py:1409-1411).
                siglip_inner.encoder.to('rpu')
                siglip_inner.post_layernorm.to('rpu')
                projector.to('rpu')

                # --- Step E: install all-layers-once patches ---
                # v5-03 D-11: bodies relocated to adapters/pi05/{gemma,adarms}.py per ADR §6.2.
                from rpu_backend.adapters.pi05.gemma import patch_gemma_model_for_rpu_all_layers_once
                from rpu_backend.adapters.pi05.adarms import patch_adarms_model_for_rpu_all_layers_once

                # Iteration 8: patch order VLM -> Expert -> SigLIP (match legacy).
                self._vlm_handle = patch_gemma_model_for_rpu_all_layers_once(
                    vlm, runtime_policy=graph_runtime_policy,
                    prefill_pair_rows=prefill_pair_rows)
                self._expert_handle = patch_adarms_model_for_rpu_all_layers_once(
                    expert, runtime_policy=graph_runtime_policy)

                # Pitfall 5: set the configured per-instance value only after
                # the VLM move. Generic API default is 0 = auto (was 160);
                # delivery profiles pass their certified value through
                # Pi05Policy.from_pretrained.
                _apply_vlm_chunk_size(self, vlm)

                # SigLIP bookkeeping failures join the composite rollback below;
                # no child can destroy before all sibling graphs retire.
                from rpu_backend.adapters.siglip import patch_siglip_model_for_rpu_all_layers_once
                # SigLIP-batch (2026-05-31): pack all camera images into one
                # seq=N*256 fused SigLIP forward (collapses 27-layer weight DDR
                # + LN/Linear/MLP/KV launches N→1; per-image attention isolation
                # via the minibatch SDPA). N = number of expected image features
                # (present + zero-padded missing cameras; constant per config,
                # so cache/GraphSignature shapes stay Dynamo-stable). Verified
                # bit-exact vs N× single-image (test_siglip_batch_3image.py).
                # Opt out with RPU_PI05_SIGLIP_BATCH=0 (→ per-image path).
                _pi05_model = self._lerobot_policy.model
                _img_feats = getattr(_pi05_model.config, 'image_features', None)
                from .cores import vision_batch_enabled
                _batch_on = vision_batch_enabled(self._rpu_execution)
                _num_cam = len(_img_feats) if (_batch_on and _img_feats) else 1
                _vision_execution = getattr(
                    self, "_vision_execution", self._rpu_execution
                )
                self._siglip_handle = patch_siglip_model_for_rpu_all_layers_once(
                    siglip_inner, projector, num_images=_num_cam,
                    w8a16_default=siglip_w8a16_default,
                    num_cores=topology.attn_tp,
                    runtime_policy=graph_runtime_policy,
                    linear_acc32=bool(_vision_execution.get("vision", {}).get("linear_acc32", False)))
                _vision_chunk = _vision_execution.get(
                    "vision", {}
                ).get("chunk_size", "auto")
                torch.ops.rpu.siglip_set_chunk_size(
                    self._siglip_handle,
                    0 if _vision_chunk == "auto" else int(_vision_chunk),
                )
                # _patched_embed_prefix (patches.py) reads this to decide
                # stack-batch vs per-image loop. Must match num_images.
                _pi05_model._rpu_siglip_batch_n = _num_cam

                # Double-projection guard — skip projector call in get_image_features.
                paligemma_model_cls = type(paligemma_model)
                if not getattr(paligemma_model_cls, "_rpu_get_image_features_patched", False):
                    from .runtime import (
                        _load_pi05_probe_tensor,
                    )
                    def _rpu_get_image_features(self, pixel_values, **kwargs):
                        _pix_override = os.environ.get("RPU_PI05_LOAD_PIXEL_VALUES")
                        if _pix_override and os.path.exists(_pix_override):
                            pixel_values = _load_pi05_probe_tensor(
                                _pix_override,
                                name="RPU_PI05_LOAD_PIXEL_VALUES",
                                expected_shape=tuple(pixel_values.shape),
                            ).to(
                                dtype=pixel_values.dtype,
                                device=pixel_values.device,
                            )
                        image_outputs = self.vision_tower(pixel_values, return_dict=True)
                        image_outputs.pooler_output = image_outputs.last_hidden_state
                        return image_outputs
                    paligemma_model_cls.get_image_features = _rpu_get_image_features
                    paligemma_model_cls._rpu_get_image_features_patched = True

                # embed_image: skip fp32 + *sqrt(hs) on RPU (they cancel).
                pwe_cls = type(pawe)
                if not getattr(pwe_cls, "_rpu_embed_image_patched", False):
                    def _rpu_embed_image(self, image):
                        image_outputs = self.paligemma.model.get_image_features(image)
                        out = image_outputs.pooler_output.half()
                        return out
                    pwe_cls.embed_image = _rpu_embed_image
                    pwe_cls._rpu_embed_image_patched = True

                # Step F (Iteration 5): keep 4 root Linears on CPU fp16.
                # The denoise loop routes via rpu_adarms_model_forward which
                # auto-routes CPU inputs to RPU + handles final PiGemmaRMSNorm.
                # Step G: install sample_actions patch (runtime.py).
                from .runtime import install_sample_actions_patch, _PI05_HANDLES
                install_sample_actions_patch()

                runtime_device = next(vlm.layers[0].parameters()).device
                if runtime_device.type != "rpu":
                    raise RPUBackendError(
                        "Pi05Adapter.to_rpu(): VLM layer 0 did not remain on RPU "
                        f"after swizzle (device={runtime_device})."
                    )
                pi05_pytorch._rpu_runtime_device = runtime_device
                pi05_pytorch._rpu_adarms_cond_cache = {}
                pi05_pytorch._rpu_denoise_mask_cache = None
                pi05_pytorch._pi05_language_prefix_cache = None
                pi05_pytorch._pi05_prefix_assembly_ok = None
                pi05_pytorch._pi05_prefix_assembly_enabled = True

                _PI05_HANDLES[pi05_pytorch] = 1  # sentinel; only non-None checked
                pi05_pytorch._rpu_execution = self._rpu_execution
                self._install_fused_denoise_handle(
                    runtime_policy=graph_runtime_policy)
                # LeRobot calls eval() on every action prediction. Perform the
                # recursive transition once; patches.py makes later already-eval
                # calls O(1) while still honoring a future explicit train().
                self._lerobot_policy.eval()

                # D-32 A9 wire-point #2 (POST-install + rollback).
                from rpu_backend.runtime.hw_attrs import (
                    validate_postinstall, install_hw_attr_validator,
                )
                validate_postinstall(self._lerobot_policy)
                install_hw_attr_validator(self._lerobot_policy)

                # Publish the model ready marker last. A failure here is still
                # caught by the outer transaction and retires every component.
                self._pi05_handle = 1
                self._rpu_is_ready = True
                self._lerobot_policy._rpu_swizzled = True

            except BaseException as error:
                if pi05_pytorch is not None:
                    try:
                        # Snapshot even a partial install before Session can
                        # reject an already-poisoned pending leaf's cleanup.
                        self._take_retirement_ownership()
                        self._execution_session.shutdown(
                            lambda: self._close_without_session(release=False))
                    except BaseException as cleanup_error:
                        self._retain_retirement_failure(cleanup_error)
                        error.add_note(f"Pi0.5 cleanup failed: {cleanup_error!r}")
                else:
                    _reset_adapter_runtime_state(self)

                _release_after_failed_install(
                    self._lerobot_policy,
                    pre_swizzle=release_live_on_failure,
                )
                raise  # Leave _rpu_swizzle_started=True per CR-P1-r5

            try:
                self._take_retirement_ownership()
            except BaseException as error:
                self._retain_retirement_failure(error)
                raise
            return self._lerobot_policy

        finally:
            _SWIZZLE_LOCK.release()
