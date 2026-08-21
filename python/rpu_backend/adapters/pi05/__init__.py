"""Pi0.5 adapter orchestration and process-wide swizzle serialization.

``Pi05Adapter`` performs the irreversible CPU→RPU mutation triggered by
``Pi05Policy.to('rpu')``. ``_SWIZZLE_LOCK`` serializes concurrent installs.
"""
from __future__ import annotations
import os
import threading
from typing import Any

import torch

from rpu_backend.runtime import rpu_env_bool
from rpu_backend.api.errors import RPUBackendError, RPUUnsupportedDtypeError
from rpu_backend.api.causal_lm import _claim_live_instance, _release_live_instance


# Pi0.5-specific module-level swizzle lock.
_SWIZZLE_LOCK = threading.Lock()


# Default ON; set RPU_PI05_FUSED_DENOISE=0 to use the per-step AdaRMS path.
_FUSED_ENV_VAR = "RPU_PI05_FUSED_DENOISE"
_MISSING = object()
_FUSED_DENOISE_INSTALL_ATTRS = (
    "_rpu_fused_denoise_handle",
    "_rpu_fused_denoise_handle_finalizer",
    "_rpu_fused_denoise_graph_cache",
)


def _fused_denoise_enabled() -> bool:
    return rpu_env_bool(_FUSED_ENV_VAR, default=True)


def _validate_execution_geometry(adapter: "Pi05Adapter") -> None:
    """Reject model-specific fixed chunks before irreversible RPU mutation."""
    model = adapter._lerobot_policy.model
    execution = adapter._rpu_execution

    if execution.get("action") and not _fused_denoise_enabled():
        raise ValueError(
            "Pi0.5 action execution configuration requires the fused denoise "
            "path; RPU_PI05_FUSED_DENOISE=0 selects per-step AdaRMS, which does "
            "not consume action.chunk_size."
        )
    action_chunk = execution.get("action", {}).get("chunk_size", "auto")
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
    batch_vision = rpu_env_bool("RPU_PI05_SIGLIP_BATCH", default=True)
    num_vision_chunks = (
        len(image_features) if batch_vision and image_features else 1
    )
    vision_required = num_vision_chunks * 256
    vision_chunk = execution.get("vision", {}).get("chunk_size", "auto")
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


def _apply_vlm_chunk_size(adapter: "Pi05Adapter", vlm: Any) -> None:
    """Apply public prefill authority, otherwise preserve the current value."""
    configured = "chunk_size" in adapter._rpu_execution.get("prefill", {})
    if configured or not hasattr(vlm, "_rpu_chunk_size"):
        vlm._rpu_chunk_size = adapter._vlm_chunk_size


def _destroy_fused_denoise_handle(handle: int) -> None:
    """Best-effort finalizer callback for a Pi0.5 fused denoise handle."""
    try:
        torch.ops.rpu.pi05_denoise_step_destroy(handle)
    except Exception:
        # The op may already be unavailable during interpreter shutdown.
        pass


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
    if not getattr(finalizer, "alive", False):
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
        or not getattr(fused_finalizer, "alive", False)
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
# Pi05Adapter orchestrates the `Pi05Policy.to('rpu')` mutation.
# =============================================================================

class Pi05Adapter:
    """Internal orchestrator for `Pi05Policy.to('rpu')` mutation."""

    def __init__(self, lerobot_policy: Any) -> None:
        self._lerobot_policy = lerobot_policy
        self._rpu_execution = {}
        # 0 lets the exact planner pick per shape. A deployment profile may pin
        # a certified value through Pi05Policy.from_pretrained.
        self._vlm_chunk_size = 0
        _reset_adapter_runtime_state(self)
        state = _pi05_runtime_state(lerobot_policy)
        if state is not None:
            _sync_adapter_runtime_state(self, state)
            self._rpu_execution = getattr(
                lerobot_policy.model, "_rpu_execution", {}
            )

    @torch.no_grad()
    def prepare_graphs(
        self,
        batch: dict,
        *,
        num_steps: "int | None" = None,
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
        caches = [
            ("vision", vision._rpu_siglip_graph_cache),
            ("prefill", vlm._rpu_gemma_graph_cache),
            ("denoise", pi05_pytorch._rpu_fused_denoise_graph_cache),
        ]

        # The first fused call also runs the standalone AdaRMS path once as a
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
            }

        try:
            build_value = self._lerobot_policy.predict_action_chunk(
                batch, num_steps=steps
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
                batch, num_steps=steps
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

            build_replay_bit_equal = torch.equal(build_action, warm_action)
            build_replay_max_abs = float(
                (build_action.float() - warm_action.float()).abs().max()
            )
            if not build_replay_bit_equal:
                raise RuntimeError(
                    "Pi0.5 WARMING replay differs from the RECORDING action "
                    f"(max_abs={build_replay_max_abs:.7g})"
                )
            for _, cache in frozen_caches:
                cache.freeze()

            before_ready = {
                name: _stats(cache) for name, cache in frozen_caches
            }
            ready_value = self._lerobot_policy.predict_action_chunk(
                batch, num_steps=steps
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
            if not torch.equal(warm_action, ready_action):
                max_abs = float(
                    (warm_action.float() - ready_action.float()).abs().max()
                )
                raise RuntimeError(
                    "Pi0.5 READY replay differs from the WARMING action "
                    f"(max_abs={max_abs:.7g})"
                )
        except Exception:
            for _, cache in caches:
                cache.begin_warmup()
            _PI05_PREPARED_GRAPH_PROFILES.pop(pi05_pytorch, None)
            raise

        result = {
            "phase": "READY",
            "num_steps": steps,
            "graph_counts": graph_counts,
            "runtime": dict(runtime_profile),
            "ready_replay_validated": True,
            "action_bit_equal": True,
            "build_replay_bit_equal": build_replay_bit_equal,
            "build_replay_max_abs": build_replay_max_abs,
            "ready_stats": after_ready,
        }
        _PI05_PREPARED_GRAPH_PROFILES[pi05_pytorch] = result
        return dict(result)

    def _install_fused_denoise_handle(self) -> None:
        """Build a Pi05DenoiseStepModel handle and bind weights.

        Called from to_rpu() AFTER Step G (after the expert is swizzled and
        per-layer norm._rpu_dense_w_rp exist). Skipped (handle stays None) when
        RPU_PI05_FUSED_DENOISE=0.

        Object ownership:
          The patched sample_actions is bound to `PI05Pytorch.sample_actions`.
          Inside that patched body, `self` is a PI05Pytorch
          instance — i.e. `self._lerobot_policy.model` from this adapter's
          POV. The fused handle MUST be stored on that PI05Pytorch so the
          patched dispatcher can find it via plain `getattr(self, ...)`.

          `_PI05_HANDLES` is keyed on the same `self._lerobot_policy.model`.
        """
        import weakref
        from rpu_backend.adapters.pi05.weights import (
            _prepare_final_norm_weights, _swizzle_action_proj_weights,
        )

        pi05_pytorch = self._lerobot_policy.model           # PI05Pytorch
        model_state = vars(pi05_pytorch)
        install_snapshot = {
            name: model_state.get(name, _MISSING)
            for name in _FUSED_DENOISE_INSTALL_ATTRS
        }
        old_handle = install_snapshot["_rpu_fused_denoise_handle"]
        old_finalizer = install_snapshot[
            "_rpu_fused_denoise_handle_finalizer"]
        if old_finalizer is _MISSING:
            old_finalizer = None
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
            old_cache = getattr(
                pi05_pytorch, "_rpu_fused_denoise_graph_cache", None)
            committed = False
            try:
                # Publish the explicit opt-out completion state before touching
                # the old graph/handle, so a Python publication interruption is
                # fully reversible.
                pi05_pytorch._rpu_fused_denoise_handle = None
                pi05_pytorch._rpu_fused_denoise_handle_finalizer = None
                pi05_pytorch._rpu_fused_denoise_graph_cache = None
                self._finalizers = [
                    finalizer
                    for finalizer in self._finalizers
                    if finalizer is not old_finalizer and finalizer.alive
                ]

                if old_cache is not None:
                    try:
                        old_cache.clear()
                    except Exception:
                        pass
                if (
                    (old_finalizer is None or old_finalizer.alive)
                    and old_handle is not _MISSING
                    and old_handle is not None
                ):
                    torch.ops.rpu.pi05_denoise_step_destroy(old_handle)
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
        ec     = expert.config
        effective_expert_nkv = int(getattr(
            expert, "_rpu_effective_num_kv_heads", ec.num_key_value_heads
        ))

        # Swizzle final_norm + action projections (idempotent under sentinel).
        _prepare_final_norm_weights(expert)
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
        from rpu_backend.adapters.pi05.w8a16 import pi05_expert_scale_lists
        scale_lists = pi05_expert_scale_lists(expert, require_rpu=True)

        # AdaRMS dense W8A16 follows its own env/default selector and scale list.
        # The int8 weights use the same row-partition layout; the original FP16
        # buffers remain available to the standalone AdaRMS path.
        _is_w8a16 = bool(scale_lists and scale_lists[0])
        _dense_w8a16 = rpu_env_bool(
            "RPU_PI05_ADARMS_DENSE_W8A16", default=_is_w8a16)
        final_dense_w = expert.norm._rpu_dense_w_rp
        attn_dense_ws, mlp_dense_ws = [], []
        final_dense_ws = expert.norm._rpu_dense_w_rp.new_empty(0)
        if _dense_w8a16:
            from rpu_backend.quant._common import quantize_linear_per_channel
            from rpu_backend.runtime.weights import (
                transform_linear_weight as _tlw, NUM_CORES as _NC)

            def _q_dense(dense):
                orig = dense.weight.detach().cpu().to(torch.float16).contiguous()
                w8, sc = quantize_linear_per_channel(orig)
                w_rp = _tlw(w8.contiguous(), partition=0,
                            num_cores=_NC).to('rpu').contiguous()
                return w_rp, sc.to(torch.float16).to('rpu').contiguous()

            _aw = [_q_dense(l.input_layernorm.dense) for l in layers]
            _mw = [_q_dense(l.post_attention_layernorm.dense) for l in layers]
            attn_w = [p[0] for p in _aw]; attn_dense_ws = [p[1] for p in _aw]
            mlp_w  = [p[0] for p in _mw]; mlp_dense_ws  = [p[1] for p in _mw]
            final_dense_w, final_dense_ws = _q_dense(expert.norm.dense)

        # Prepare GraphCache before creating a native handle. A cache allocation
        # failure must not orphan a native entry or overwrite a live install.
        import rpu_backend
        denoise_graph_cache = rpu_backend.graph.GraphCache()

        handle = torch.ops.rpu.pi05_denoise_step_create()
        handle_finalizer = None
        committed = False
        # Schema arg order: (... hidden_size, max_action_dim,
        # chunk_size, num_q_heads, num_kv_heads, head_dim, num_layers, eps).
        try:
            handle_finalizer = weakref.finalize(
                pi05_pytorch, _destroy_fused_denoise_handle, handle)
            torch.ops.rpu.pi05_denoise_step_set_weights(
                handle, q_w, k_w, v_w, o_w, gate, up, down,
                attn_w, attn_b, mlp_w, mlp_b,
                final_dense_w, expert.norm._rpu_dense_b_rp,
                pi05_pytorch.action_in_proj._rpu_w_num_cores_1,
                pi05_pytorch.action_in_proj._rpu_bias,
                pi05_pytorch.action_out_proj._rpu_w_num_cores_1,
                pi05_pytorch.action_out_proj._rpu_bias,
                cos_c, sin_c,
                H,
                pi05_pytorch.config.max_action_dim,
                pi05_pytorch.config.chunk_size,
                ec.num_attention_heads, effective_expert_nkv, ec.head_dim,
                L, float(ec.rms_norm_eps),
                *scale_lists,
                attn_dense_ws, mlp_dense_ws, final_dense_ws)
            _action_chunk = self._rpu_execution.get(
                "action", {}
            ).get("chunk_size", "auto")
            torch.ops.rpu.pi05_denoise_step_set_chunk_size(
                handle,
                0 if _action_chunk == "auto" else int(_action_chunk),
            )

            pi05_pytorch._rpu_fused_denoise_handle = handle
            pi05_pytorch._rpu_fused_denoise_handle_finalizer = handle_finalizer
            pi05_pytorch._rpu_fused_denoise_graph_cache = denoise_graph_cache
            self._finalizers = [
                finalizer
                for finalizer in self._finalizers
                if finalizer is not old_finalizer and finalizer.alive
            ] + [handle_finalizer]

            if (
                (old_finalizer is None or old_finalizer.alive)
                and old_handle is not _MISSING
                and old_handle is not None
            ):
                torch.ops.rpu.pi05_denoise_step_destroy(old_handle)
            committed = True
        except BaseException:
            if not committed:
                restore_publication()
                if handle_finalizer is None:
                    _destroy_fused_denoise_handle(handle)
                elif handle_finalizer.alive:
                    handle_finalizer()
            raise

        if old_finalizer is not None and old_finalizer.alive:
            old_finalizer.detach()

    def to_rpu(self) -> Any:
        """Install all Pi0.5 RPU components under one swizzle lock."""
        # Idempotent re-entry.
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

        # Partial-failure retry guard.
        if getattr(self._lerobot_policy, "_rpu_swizzle_started", False):
            raise RPUBackendError(
                "Pi05Adapter.to_rpu(): prior swizzle attempt failed mid-way; "
                "submodel weights in undefined state. Reload via "
                "Pi05Policy.from_pretrained(...).")

        _validate_execution_geometry(self)

        # RPU_PI05_KEEP_CPU is unsupported. Check before any irreversible state
        # mutation while rejection can still leave the model pristine.
        _keep_cpu_raw = os.environ.get("RPU_PI05_KEEP_CPU", "").strip()
        if _keep_cpu_raw:
            raise RuntimeError(
                "RPU_PI05_KEEP_CPU is unsupported. Set RPU_PI05_KEEP_CPU= "
                "(empty) to use the fused decoder path."
            )

        if not _SWIZZLE_LOCK.acquire(blocking=False):
            raise RPUBackendError(
                "Pi05Adapter.to_rpu(): another Pi0.5 swizzle is in progress.")
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

            # Pre-install hardware-attribute scan.
            from rpu_backend.runtime.hw_attrs import validate_preinstall
            validate_preinstall(self._lerobot_policy)

            _claim_live_instance(self._lerobot_policy)  # WARNING 14 ordering

            # The outer transaction covers the dtype gate through final publish.
            # Before irreversible weight/device mutation, a failed preflight
            # can release the process slot immediately. Once Step C starts, the
            # broken owner may still hold RPU tensors even after native handles
            # are cleaned; keep the slot claimed until that owner is GC'd.
            release_live_on_failure = True
            pi05_pytorch = None
            vlm = None
            expert = None
            siglip_inner = None
            try:
                self._lerobot_policy._rpu_swizzle_started = True

                # Step A: class-level patches.
                from .patches import install_class_patches
                install_class_patches(self._lerobot_policy)

                # Step B: pre-swizzle dtype gate (params + buffers).
                bad_dtypes = []
                for name, p in self._lerobot_policy.named_parameters():
                    if p.is_floating_point() and p.dtype != torch.float16:
                        bad_dtypes.append(f"param {name}: {p.dtype}")
                        if len(bad_dtypes) >= 5:
                            break
                if len(bad_dtypes) < 5:
                    for name, buf in self._lerobot_policy.named_buffers():
                        if buf.is_floating_point() and buf.dtype != torch.float16:
                            bad_dtypes.append(f"buffer {name}: {buf.dtype}")
                            if len(bad_dtypes) >= 5:
                                break
                if bad_dtypes:
                    raise RPUUnsupportedDtypeError(
                        "Pi05Adapter.to_rpu(): floating params/buffers must be "
                        f"torch.float16 before swizzle. Offenders (first 5): "
                        f"{bad_dtypes}. Reload with dtype=torch.float16 or "
                        "call .half() on the lerobot policy.")

                # Reach into the lerobot submodel tree.
                pawe = self._lerobot_policy.model.paligemma_with_expert
                vlm    = pawe.paligemma.model.language_model
                expert = pawe.gemma_expert.model
                siglip = pawe.paligemma.model.vision_tower
                projector = pawe.paligemma.model.multi_modal_projector
                pi05_pytorch = self._lerobot_policy.model
                siglip_inner = (
                    siglip.vision_model
                    if hasattr(siglip, 'vision_model')
                    else siglip
                )

                # SigLIP W8A16 follows the VLM weight dtype by default;
                # RPU_PI05_SIGLIP_W8A16 remains authoritative. Probe before
                # weight conversion.
                _vlm_layers = getattr(vlm, "layers", None)
                # Pi0.5 enables its validated two-stage route by default.
                # setdefault preserves explicit environment overrides.
                os.environ.setdefault("RPU_ALLREDUCE_TWOSTAGE", "1")
                # Pi0.5 graphs have no post_fn, so full replay-skip is supported.
                os.environ.setdefault("RPU_WALL_OSS_FAST_REPLAY", "1")
                os.environ.setdefault("RPU_FASTREPLAY_SKIP_SYNC", "1")
                os.environ.setdefault("RPU_DEEP_FAST_REPLAY", "1")
                # Default to graph-mode denoise unroll with device-side FP16 Euler.
                os.environ.setdefault("RPU_PI05_DENOISE_UNROLL", "1")
                _vlm_q_dtype = (_vlm_layers[0].self_attn.q_proj.weight.dtype
                                if _vlm_layers else None)
                siglip_w8a16_default = _vlm_q_dtype in (torch.int8, torch.uint8)

                # Subsystem swizzles; Expert skips its specialized dense modules.
                release_live_on_failure = False
                # The supported path always swizzles all three subsystems.
                from rpu_backend.runtime import weights as _mc

                from rpu_backend.adapters.pi05.w8a16 import (
                    replicate_pi05_decoder_kv_heads_for_attention_tp,
                )
                vlm_effective_nkv = replicate_pi05_decoder_kv_heads_for_attention_tp(
                    vlm, effective_num_kv_heads=8, label="vlm"
                )
                expert_effective_nkv = replicate_pi05_decoder_kv_heads_for_attention_tp(
                    expert, effective_num_kv_heads=8, label="expert"
                )
                if int(vlm.config.head_dim) != int(expert.config.head_dim):
                    raise RPUBackendError(
                        "Pi05Adapter.to_rpu(): VLM/expert cache sharing "
                        "requires matching head_dim after KV replication, got "
                        f"vlm={vlm.config.head_dim}, expert={expert.config.head_dim}"
                    )
                if int(vlm_effective_nkv) != int(expert_effective_nkv):
                    raise RPUBackendError(
                        "Pi05Adapter.to_rpu(): VLM/expert cache sharing "
                        "requires matching effective KV heads, got "
                        f"vlm={vlm_effective_nkv}, expert={expert_effective_nkv}"
                    )

                from rpu_backend.adapters.pi05.w4pack import (
                    int4_child_names, pack_int4_projections_inplace,
                )
                # loader.py sets `<lerobot_policy>._pi05_quant_config`; the adapter
                # holds that lerobot policy as self._lerobot_policy.
                _quant_cfg = getattr(self._lerobot_policy, "_pi05_quant_config", {}) or {}
                _int4 = int4_child_names(_quant_cfg)   # empty unless method=w4a16

                vlm_attn_tp = min(8, vlm_effective_nkv)
                _mc.convert_linear_weights_inplace(
                    vlm, attn_num_cores=vlm_attn_tp, skip_names=set(_int4))
                if _int4:
                    pack_int4_projections_inplace(vlm, _int4, attn_num_cores=vlm_attn_tp)

                expert_attn_tp = min(8, expert_effective_nkv)
                _mc.convert_linear_weights_inplace(
                    expert, skip_names={'dense'} | set(_int4), attn_num_cores=expert_attn_tp)
                if _int4:
                    pack_int4_projections_inplace(expert, _int4, attn_num_cores=expert_attn_tp)

                # SigLIP projector swizzle — delegate to weights.py
                # (_rpu_weights_converted prevents a second swizzle).
                from .weights import swizzle_projector_inplace
                swizzle_projector_inplace(projector)

                # SigLIP encoder swizzle runs on CPU before move-to-RPU. The
                # idempotence guard on
                # `vit._rpu_siglip_weights_converted` short-circuits the inner call
                # inside patch_siglip_model_for_rpu_all_layers_once.
                from rpu_backend.adapters.siglip import _convert_siglip_weights_for_rpu
                _convert_siglip_weights_for_rpu(
                    siglip_inner, w8a16_default=siglip_w8a16_default)

                # --- Step D: move subsystems to RPU ---
                vlm._rpu_attn_tp = vlm_attn_tp
                vlm._rpu_max_seq_len = 2048
                vlm.to('rpu')
                # embed_tokens must stay on CPU.
                if getattr(vlm, "embed_tokens", None) is not None:
                    vlm.embed_tokens = vlm.embed_tokens.to('cpu')

                expert._rpu_attn_tp = expert_attn_tp
                expert._rpu_max_seq_len = 2048
                expert.to('rpu')

                # Keep Expert embedding and final norm on CPU.
                if getattr(expert, "embed_tokens", None) is not None:
                    expert.embed_tokens.to('cpu')
                if getattr(expert, "norm", None) is not None:
                    expert.norm.to('cpu')

                # Move only encoder + post_layernorm + projector to RPU;
                # keep vit.embeddings on CPU.
                siglip_inner.encoder.to('rpu')
                siglip_inner.post_layernorm.to('rpu')
                projector.to('rpu')

                # Install all-layers-once patches.
                from rpu_backend.adapters.pi05.gemma import patch_gemma_model_for_rpu_all_layers_once
                from rpu_backend.adapters.pi05.adarms import patch_adarms_model_for_rpu_all_layers_once

                # Patch order: VLM -> Expert -> SigLIP.
                self._vlm_handle    = patch_gemma_model_for_rpu_all_layers_once(vlm)
                self._expert_handle = patch_adarms_model_for_rpu_all_layers_once(expert)

                # Set the configured per-instance value only after
                # the VLM move. Generic API default is 0 = auto (was 160);
                # deployment profiles pass their certified value through
                # Pi05Policy.from_pretrained.
                _apply_vlm_chunk_size(self, vlm)

                # Direct-handle SigLIP is wrapped for orphan-handle cleanup.
                # Same-instance retry is not supported because validate_preinstall
                # rejects the non-public _rpu_*
                # attrs left behind by earlier VLM/expert/projector swizzles. Fresh-
                # instance retry IS supported (lifecycle Case 8 verifies that path).
                from rpu_backend.adapters.siglip import patch_siglip_model_for_rpu_all_layers_once
                try:
                    # Pack all camera images into one
                    # seq=N*256 fused SigLIP forward (collapses 27-layer weight DDR
                    # + LN/Linear/MLP/KV launches N→1; per-image attention isolation
                    # via the minibatch SDPA). N = number of expected image features
                    # (present + zero-padded missing cameras; constant per config,
                    # so cache/GraphSignature shapes stay Dynamo-stable).
                    # Opt out with RPU_PI05_SIGLIP_BATCH=0 (→ per-image path).
                    _pi05_model = self._lerobot_policy.model
                    _img_feats = getattr(_pi05_model.config, 'image_features', None)
                    _batch_on = rpu_env_bool(
                        "RPU_PI05_SIGLIP_BATCH", default=True)
                    _num_cam = len(_img_feats) if (_batch_on and _img_feats) else 1
                    self._siglip_handle = patch_siglip_model_for_rpu_all_layers_once(
                        siglip_inner, projector, num_images=_num_cam,
                        w8a16_default=siglip_w8a16_default)
                    _vision_chunk = self._rpu_execution.get(
                        "vision", {}
                    ).get("chunk_size", "auto")
                    torch.ops.rpu.siglip_set_chunk_size(
                        self._siglip_handle,
                        0 if _vision_chunk == "auto" else int(_vision_chunk),
                    )
                    # _patched_embed_prefix (patches.py) reads this to decide
                    # stack-batch vs per-image loop. Must match num_images.
                    _pi05_model._rpu_siglip_batch_n = _num_cam
                except BaseException:
                    # The shared SigLIP installer cleans an in-flight handle
                    # itself. This outer guard still owns a successfully
                    # published handle if the following adapter bookkeeping
                    # raises; invoke its tracked finalizer immediately. Keep the
                    # direct destroy fallback for custom patchers that do
                    # not publish a finalizer.
                    self._siglip_handle = None
                    if 'siglip_inner' in locals() and hasattr(siglip_inner, "_rpu_vision_handle"):
                        finalizer = getattr(
                            siglip_inner, "_rpu_vision_handle_finalizer", None)
                        if finalizer is not None and finalizer.alive:
                            finalizer()
                        elif finalizer is None:
                            try:
                                torch.ops.rpu.siglip_destroy(
                                    siglip_inner._rpu_vision_handle)
                            except Exception:
                                pass
                        for attr in (
                            "_rpu_vision_handle",
                            "_rpu_vision_handle_finalizer",
                        ):
                            vars(siglip_inner).pop(attr, None)
                    raise

                # Double-projection guard — skip projector call in get_image_features.
                paligemma_model = pawe.paligemma.model
                paligemma_model_cls = type(paligemma_model)
                if not getattr(paligemma_model_cls, "_rpu_get_image_features_patched", False):
                    from .runtime import (
                        _load_pi05_probe_tensor,
                        _pi05_probe_save,
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
                        _pi05_probe_save("get_image_features_input", pixel_values)
                        image_outputs = self.vision_tower(pixel_values, return_dict=True)
                        _pi05_probe_save("vision_tower_last_hidden_state", image_outputs.last_hidden_state)
                        image_outputs.pooler_output = image_outputs.last_hidden_state
                        return image_outputs
                    paligemma_model_cls.get_image_features = _rpu_get_image_features
                    paligemma_model_cls._rpu_get_image_features_patched = True

                # Keep the pooled vision features on RPU and return FP16.
                pwe_cls = type(pawe)
                if not getattr(pwe_cls, "_rpu_embed_image_patched", False):
                    from .runtime import _pi05_probe_save
                    def _rpu_embed_image(self, image):
                        _pi05_probe_save("embed_image_input", image)
                        image_outputs = self.paligemma.model.get_image_features(image)
                        out = image_outputs.pooler_output.half()
                        _pi05_probe_save("embed_image_out", out)
                        return out
                    pwe_cls.embed_image = _rpu_embed_image
                    pwe_cls._rpu_embed_image_patched = True

                # Keep four root Linear modules on CPU fp16.
                # The denoise loop routes via rpu_adarms_model_forward which
                # auto-routes CPU inputs to RPU + handles final PiGemmaRMSNorm.
                # Install the sample_actions patch.
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
                self._install_fused_denoise_handle()
                # LeRobot calls eval() on every action prediction. Perform the
                # recursive transition once; patches.py makes later already-eval
                # calls O(1) while still honoring a future explicit train().
                self._lerobot_policy.eval()

                # Post-install hardware-attribute validation and rollback support.
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

                return self._lerobot_policy

            except BaseException:
                # Clear private queues before their native component handles.
                _clear_component_graph_cache(
                    pi05_pytorch, "_rpu_fused_denoise_graph_cache")
                _clear_component_graph_cache(
                    siglip_inner, "_rpu_siglip_graph_cache")
                _clear_component_graph_cache(
                    expert, "_rpu_adarms_graph_cache")
                _clear_component_graph_cache(
                    vlm, "_rpu_gemma_graph_cache")

                # Each component is retired with ITS OWN install-attribute
                # tuple, so the rollback deletes everything that component's
                # patch published (`forward` included) instead of only the
                # handle pair.
                from rpu_backend.adapters.pi05.adarms import (
                    _ADARMS_RUNTIME_INSTALL_ATTRS,
                )
                from rpu_backend.adapters.pi05.gemma import (
                    _GEMMA_RUNTIME_INSTALL_ATTRS,
                )
                from rpu_backend.adapters.siglip import (
                    _SIGLIP_RUNTIME_INSTALL_ATTRS,
                )

                _cleanup_component_handle(
                    pi05_pytorch,
                    handle_attr="_rpu_fused_denoise_handle",
                    finalizer_attr="_rpu_fused_denoise_handle_finalizer",
                    destroy=lambda handle: (
                        torch.ops.rpu.pi05_denoise_step_destroy(handle)),
                    install_attrs=_FUSED_DENOISE_INSTALL_ATTRS,
                )
                _cleanup_component_handle(
                    siglip_inner,
                    handle_attr="_rpu_vision_handle",
                    finalizer_attr="_rpu_vision_handle_finalizer",
                    destroy=lambda handle: torch.ops.rpu.siglip_destroy(handle),
                    install_attrs=_SIGLIP_RUNTIME_INSTALL_ATTRS,
                )
                _cleanup_component_handle(
                    expert,
                    handle_attr="_rpu_action_handle",
                    finalizer_attr="_rpu_action_handle_finalizer",
                    destroy=lambda handle: torch.ops.rpu.adarms_destroy(handle),
                    install_attrs=_ADARMS_RUNTIME_INSTALL_ATTRS,
                )
                _cleanup_component_handle(
                    vlm,
                    handle_attr="_rpu_vlm_decoder_handle",
                    finalizer_attr="_rpu_vlm_decoder_handle_finalizer",
                    destroy=lambda handle: torch.ops.rpu.gemma_destroy(handle),
                    install_attrs=_GEMMA_RUNTIME_INSTALL_ATTRS,
                )
                if pi05_pytorch is not None:
                    try:
                        from .runtime import (
                            _PI05_HANDLES,
                            _PI05_PREPARED_GRAPH_PROFILES,
                        )
                        _PI05_HANDLES.pop(pi05_pytorch, None)
                        _PI05_PREPARED_GRAPH_PROFILES.pop(pi05_pytorch, None)
                    except Exception:
                        pass
                _reset_adapter_runtime_state(self)

                _release_after_failed_install(
                    self._lerobot_policy,
                    pre_swizzle=release_live_on_failure,
                )
                raise  # Leave _rpu_swizzle_started=True to block unsafe retry.

        finally:
            _SWIZZLE_LOCK.release()
