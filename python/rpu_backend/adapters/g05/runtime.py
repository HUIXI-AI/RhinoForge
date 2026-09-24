"""Galaxea G0.5 Qwen3.5 VLM runtime on RPU.

The canonical G0.5 components reuse the shipping Qwen3.5 fused runtime.
Single-frame vision uses its experimental vision path. Action I/O projections and FP16 Euler run on RPU while time
conditioning remains on the host. BAR and training are intentionally unsupported.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import functools
import importlib
import os
import sys
import threading
import types
import weakref

import torch

from rpu_backend.api._execution import execution_serialized
from rpu_backend.api.causal_lm import _claim_live_instance
from rpu_backend.adapters.g05.action import (
    _check_g05_action_profile,
    _g05_action_runtime_ready,
    _rpu_g05_inference_fm,
    patch_g05_action_expert_for_rpu,
)


_G05_LAYER_TYPES = tuple(
    "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
    for i in range(24)
)
_G05_VLM_PROFILE = (
    2048, 6144, 24, 8, 2, 256,
    16, 16, 128, 128, 4,
    False, "silu", "embedding", "lm_head", None, True,
    1.0e-6, 10_000_000.0, "default", (11, 11, 10), True, 0.25,
    _G05_LAYER_TYPES,
)
_INSTALL_LOCK = threading.Lock()
_VISION_PATCH_LOCK = threading.Lock()
_POLICY_INSTALL_LOCK = threading.Lock()
_FAILED_POLICY_RETIREMENTS = []
_GRAPH_COMPOSITE_CHILD = "COMPOSITE_CHILD"

_G05_VISION_COMPONENT = "vision_encoder"
_G05_TEXT_COMPONENT = "language_model"
_G05_ACTION_COMPONENT = "action_expert"
_G05_EXECUTION_COMPONENTS = {
    _G05_VISION_COMPONENT: {
        "vision": ("chunk_size", "padding_rows", "padding_budget"),
    },
    _G05_TEXT_COMPONENT: {
        "prefill": ("chunk_size", "padding_rows", "padding_budget"),
    },
    _G05_ACTION_COMPONENT: {"action": ("chunk_size",)},
}


def _resolve_g05_execution(value, *, max_seq_len: int, entry_point: str):
    """Resolve one root request into three stable composite-child views."""
    from rpu_backend.api._execution import (
        normalize_rpu_execution,
        resolve_component_rpu_execution,
    )

    root = normalize_rpu_execution(
        value,
        entry_point=entry_point,
        supported_components=_G05_EXECUTION_COMPONENTS,
    )
    profile_auto = {
        _G05_VISION_COMPONENT: {"vision": {"chunk_size": "auto"}},
        _G05_TEXT_COMPONENT: {
            "prefill": {"chunk_size": "auto", "padding_budget": 64},
        },
        _G05_ACTION_COMPONENT: {"action": {"chunk_size": "auto"}},
    }
    components = {
        component: resolve_component_rpu_execution(
            root,
            component,
            entry_point=entry_point,
            supported_components=_G05_EXECUTION_COMPONENTS,
            profile_auto=profile_auto[component],
        )
        for component in _G05_EXECUTION_COMPONENTS
    }
    text_chunk = components[_G05_TEXT_COMPONENT]["prefill"]["chunk_size"]
    if isinstance(text_chunk, int) and text_chunk % 64:
        raise ValueError(
            f"{entry_point}: language_model prefill.chunk_size must be a "
            f"multiple of 64, got {text_chunk}"
        )
    action_chunk = components[_G05_ACTION_COMPONENT]["action"]["chunk_size"]
    if isinstance(action_chunk, int) and action_chunk != 32:
        raise ValueError(
            f"{entry_point}: action_expert action.chunk_size must be 'auto' "
            f"or exactly 32, got {action_chunk}"
        )
    if (
        isinstance(max_seq_len, bool)
        or not isinstance(max_seq_len, int)
        or max_seq_len < 1
    ):
        raise ValueError(
            f"{entry_point}: max_seq_len must be a positive integer, got "
            f"{max_seq_len!r}"
        )
    text_prefill = components[_G05_TEXT_COMPONENT]["prefill"]
    if int(text_prefill.get("padding_budget", 0)) > 64:
        raise ValueError(f"{entry_point}: language padding_budget must be <= 64")
    vision = components[_G05_VISION_COMPONENT]["vision"]
    if vision.get("padding_rows", "auto") not in ("auto", 0) or int(
        vision.get("padding_budget", 0)
    ):
        raise ValueError(
            f"{entry_point}: vision padding is unsupported without a "
            "bidirectional padding-mask route"
        )
    return root, components


def _snapshot_instance_attrs(owner, names):
    if owner is None or not hasattr(owner, "__dict__"):
        return ()
    state = vars(owner)
    return tuple((owner, name, name in state, state.get(name)) for name in names)


def _restore_instance_attrs(snapshots) -> None:
    for owner, name, had_value, value in reversed(snapshots):
        if had_value:
            vars(owner)[name] = value
        else:
            vars(owner).pop(name, None)


def _retire_new_qwen3_5_state(owner, state_before) -> None:
    """Clear a composite-installed text runtime before destroying its handle."""
    state = getattr(owner, "_rpu_qwen3_5", None)
    if state is None or state is state_before:
        return
    from rpu_backend.adapters.qwen3_5.text import _retire_qwen3_5_text_state

    _retire_qwen3_5_text_state(owner, state=state)


def _retire_new_g05_vision_runtime(vision, runtime_before) -> None:
    """Delegate composite Vision teardown to the shared transactional installer."""
    if vision is None:
        return
    handle_before, finalizer_before = runtime_before
    handle = getattr(vision, "_rpu_vision_installing_handle", None)
    if handle is None:
        handle = getattr(vision, "_rpu_vision_handle", None)
    finalizer = getattr(vision, "_rpu_vision_installing_finalizer", None)
    if finalizer is None:
        finalizer = getattr(vision, "_rpu_vision_handle_finalizer", None)
    if handle == handle_before and finalizer is finalizer_before:
        return
    from rpu_backend.adapters.qwen3_5.vision import _rollback_qwen3_5_vision_install

    _rollback_qwen3_5_vision_install(vision)


def _instance_method_is(owner, name: str, function) -> bool:
    """Check an instance-published bound method without class fallback."""
    if owner is None or not hasattr(owner, "__dict__"):
        return False
    method = vars(owner).get(name)
    return bool(
        getattr(method, "__self__", None) is owner
        and getattr(method, "__func__", None) is function
    )


def _transactional_g05_policy_install(function):
    """Serialize composite publication and restore host hooks on failure."""
    @functools.wraps(function)
    def wrapped(policy, *args, **kwargs):
        if not _POLICY_INSTALL_LOCK.acquire(blocking=False):
            raise RuntimeError("another G0.5 policy installation is in progress")
        snapshots = ()
        perf_env_names = (
            "RPU_FASTREPLAY_SKIP_SYNC",
        )
        perf_env_snapshot = {
            name: os.environ.get(name) for name in perf_env_names
        }
        try:
            if getattr(policy, "_rpu_g05_policy_ready", False):
                if _g05_policy_runtime_ready(policy):
                    requested = kwargs.get("rpu_execution")
                    if requested is not None:
                        max_seq_len = kwargs.get("max_seq_len", 2048)
                        normalized, _components = _resolve_g05_execution(
                            requested,
                            max_seq_len=max_seq_len,
                            entry_point="patch_g05_policy_for_rpu",
                        )
                        if normalized != getattr(policy, "_rpu_execution", None):
                            raise ValueError(
                                "G0.5 policy execution config conflicts with "
                                "the installed runtime"
                            )
                    return policy
                raise RuntimeError(
                    "G0.5 policy ready marker has incomplete runtime state; "
                    "reload the checkpoint"
                )
            if getattr(policy, "_rpu_g05_policy_install_started", False):
                raise RuntimeError(
                    "G0.5 policy installation was already attempted; reload "
                    "the checkpoint to avoid reusing partial wrapper state"
                )
            model = getattr(policy, "model", None)
            processor = getattr(policy, "processor", None)
            vision = getattr(model, "vision_tower", None)
            vlm = getattr(model, "vlm", None)
            expert = getattr(model, "action_expert", None)
            vision_runtime_before = (
                getattr(vision, "_rpu_vision_handle", None),
                getattr(vision, "_rpu_vision_handle_finalizer", None),
            )
            vlm_state_before = getattr(vlm, "_rpu_qwen3_5", None)
            expert_state_before = getattr(expert, "_rpu_qwen3_5", None)
            snapshots = (
                _snapshot_instance_attrs(
                    vision,
                    (
                        "forward",
                        "_rpu_g05_vision_forward_impl",
                        "_rpu_execution",
                        "_rpu_execution_component_id",
                        "_rpu_execution_generation",
                        "_execution_generation",
                        "_execution_session",
                    ),
                )
                + _snapshot_instance_attrs(
                    model,
                    ("_rpu_g05_vision_ready", "_rpu_execution", "_execution_session"),
                )
                + _snapshot_instance_attrs(
                    vlm,
                    (
                        "forward",
                        "decode",
                        "_rpu_g05_max_seq_len",
                        "_rpu_g05_owner",
                        "_rpu_g05_original_decode",
                        "_rpu_execution",
                        "_rpu_execution_component_id",
                        "_rpu_execution_generation",
                        "_execution_session",
                    ),
                )
                + _snapshot_instance_attrs(
                    model,
                    (
                        "build_causal_mask_and_position_ids",
                        "_build_prefix_action_kv",
                        "_rpu_g05_prefix_bridge_installed",
                        "_rpu_g05_original_build_causal_mask_and_position_ids",
                        "_rpu_g05_vlm_ready",
                    ),
                )
                + _snapshot_instance_attrs(
                    expert,
                    (
                        "encode_time",
                        "forward",
                        "_rpu_g05_adaptive_mod_cache",
                        "_rpu_g05_adaptive_mod_stack_cache",
                        "_rpu_g05_time_cond_cache",
                        "_rpu_g05_fixed_time_schedule",
                        "_rpu_g05_rtc_time_schedules",
                        "_rpu_g05_original_encode_time",
                        "_rpu_g05_original_forward",
                        "_rpu_g05_action_ready",
                        "_rpu_execution",
                        "_rpu_execution_component_id",
                        "_rpu_execution_generation",
                        "_execution_session",
                    ),
                )
                + _snapshot_instance_attrs(
                    model,
                    (
                        "_rpu_g05_embed_w_rpu",
                        "_rpu_g05_proprio_mlp",
                        "_forward_embed",
                        "_rpu_g05_original_inference_fm",
                        "inference_fm",
                        "_rpu_execution",
                        "_execution_session",
                    ),
                )
                + _snapshot_instance_attrs(
                    processor,
                    (
                        "_rpu_g05_original_encode_inference",
                        "_rpu_g05_action_rvq",
                        "_rpu_g05_action_rng_rows",
                        "_rpu_g05_action_rng_codebooks",
                        "encode_inference",
                    ),
                )
                + _snapshot_instance_attrs(
                    policy,
                    (
                        "_rpu_g05_original_predict_action",
                        "predict_action",
                        "_rpu_g05_policy_ready",
                        "_rpu_execution",
                        "_rpu_execution_generation",
                        "_rpu_g05_execution_components",
                        "_rpu_g05_execution_component_generations",
                        "_rpu_g05_execution_reconfigure_journal",
                        "_execution_session",
                        "reconfigure_rpu_execution",
                        "close_rpu_execution",
                    ),
                )
            )
            try:
                return function(policy, *args, **kwargs)
            except BaseException as error:
                if vars(policy).get("_rpu_g05_policy_install_started", False):
                    session = vars(policy).get("_execution_session")
                    def retire():
                        from rpu_backend.adapters.qwen3_5.text import _clear_qwen35_graphs

                        states = tuple(state for child, before in ((vlm, vlm_state_before), (expert, expert_state_before))
                                       if (state := getattr(child, "_rpu_qwen3_5", None)) is not before)
                        _clear_qwen35_graphs(states, vision)
                        _retire_new_qwen3_5_state(expert, expert_state_before)
                        _retire_new_qwen3_5_state(vlm, vlm_state_before)
                        _retire_new_g05_vision_runtime(vision, vision_runtime_before)
                    try:
                        if session is not None:
                            session.shutdown(retire)
                        else:
                            retire()
                        for child, before in ((expert, expert_state_before), (vlm, vlm_state_before)):
                            remaining = getattr(child, "_rpu_qwen3_5", None)
                            if remaining is not None and remaining is not before:
                                raise RuntimeError("G0.5 policy cleanup left a new text runtime installed")
                        if (getattr(vision, "_rpu_vision_handle", None),
                            getattr(vision, "_rpu_vision_handle_finalizer", None)) != vision_runtime_before or getattr(vision, "_rpu_vision_installing_handle", None) is not None:
                            raise RuntimeError("G0.5 policy cleanup left a new Vision runtime installed")
                    except BaseException as cleanup_error:
                        error.add_note(f"G0.5 policy install cleanup failed: {cleanup_error!r}")
                        if session is not None:
                            session.poison()
                        from rpu_backend.api.causal_lm import _poison_live_instance

                        _poison_g05_retirement(policy, cleanup_error)
                        vars(policy)["_rpu_g05_policy_ready"] = False
                    else:
                        if all(item[2].handle is None for item in vars(policy).get("_qwen35_retirement_children", ())):
                            vars(policy).pop("_qwen35_retirement_children", None)
                        _restore_instance_attrs(snapshots)
                for name, value in perf_env_snapshot.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
                raise
        finally:
            _POLICY_INSTALL_LOCK.release()

    return wrapped


def _g05_token_index(model):
    """Resolve the token-type enum from the loaded G0.5 implementation."""
    module_names = []
    mask_helper = getattr(model, "mask_helper", None)
    if mask_helper is not None:
        module_names.append(type(mask_helper).__module__)
    module_names.extend(
        (type(model).__module__, "g05.models.g05.io.input_preprocessor")
    )
    for module_name in dict.fromkeys(module_names):
        try:
            token_index = getattr(importlib.import_module(module_name), "TOKEN_INDEX")
        except (AttributeError, ModuleNotFoundError):
            continue
        if all(
            hasattr(token_index, name)
            for name in ("IMAGE_TOKEN_INDEX", "PROPRIO_TOKEN_INDEX")
        ):
            return token_index
    raise TypeError(
        "G0.5 RPU direct fusion could not resolve the model's TOKEN_INDEX enum"
    )


class _G05SharedImages(dict):
    """Keep read-only raw images out of the processor's action-only deepcopy."""

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


def _check_g05_vlm_profile(config) -> None:
    """Accept only the canonical G0.5 Qwen3.5-2B VLM."""
    from rpu_backend.api.errors import UnsupportedModelError

    rope = getattr(config, "rope_parameters", None) or {}
    profile = (
        int(config.hidden_size),
        int(config.intermediate_size),
        int(config.num_hidden_layers),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(config.head_dim),
        int(config.linear_num_value_heads),
        int(config.linear_num_key_heads),
        int(config.linear_value_head_dim),
        int(config.linear_key_head_dim),
        int(config.linear_conv_kernel_dim),
        bool(config.attention_bias),
        str(config.hidden_act),
        str(config.input_type),
        str(config.output_type),
        getattr(config, "adaptive_mode", None),
        bool(config.use_final_norm),
        float(config.rms_norm_eps),
        float(rope.get("rope_theta", 0.0)),
        str(rope.get("rope_type", "")),
        tuple(int(x) for x in rope.get("mrope_section", ())),
        bool(rope.get("mrope_interleaved", False)),
        float(rope.get("partial_rotary_factor", 0.0)),
        tuple(config.layer_types),
    )
    if profile != _G05_VLM_PROFILE:
        raise UnsupportedModelError(
            "G0.5 RPU bring-up supports only the canonical Qwen3.5-2B VLM "
            f"profile; got {profile}."
        )


def _preflight_g05_vlm(model):
    """Validate the complete G0.5 VLM contract before any irreversible install."""
    vlm = getattr(model, "vlm", None)
    if vlm is None or not hasattr(vlm, "layers") or not hasattr(vlm, "norm"):
        raise TypeError("expected G05ModelQwen35 with model.vlm.layers and model.vlm.norm")
    if getattr(model, "training", True):
        raise NotImplementedError(
            "G0.5 RPU bring-up supports inference only; call model.eval()"
        )
    _check_g05_vlm_profile(vlm.config)
    if getattr(getattr(model, "cfg", None), "position_ids_type", None) != "pi0fast":
        raise NotImplementedError(
            "G0.5 RPU q1 decode currently requires position_ids_type=pi0fast"
        )
    return vlm


def _g05_cache_num_items(cache) -> int:
    return int(cache._rpu_qwen3_5_cache.position)


def _check_causal_mask(mask, *, query_len: int, past_len: int) -> None:
    if mask is None:
        return
    mask = mask.detach().to("cpu")
    expected_shape = (1, 1, query_len, past_len + query_len)
    if tuple(mask.shape) != expected_shape:
        raise NotImplementedError(
            f"G0.5 RPU expected causal mask shape {expected_shape}, got {tuple(mask.shape)}"
        )
    q = torch.arange(query_len).view(query_len, 1)
    k = torch.arange(past_len + query_len).view(1, past_len + query_len)
    expected_zeros = (k <= past_len + q).view(expected_shape)
    if not torch.equal(mask.eq(0), expected_zeros):
        raise NotImplementedError(
            "G0.5 RPU supports only the standard causal mask; BAR/block masks are not ported"
        )


def _rpu_g05_build_causal_mask_and_position_ids(
    self,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    kv_len: int = 0,
    dtype: torch.dtype = torch.float32,
    is_action_block: bool = False,
):
    """Build only M-RoPE positions for the certified causal prefill path."""
    original = self._rpu_g05_original_build_causal_mask_and_position_ids
    if (
        self.training
        or kv_len != 0
        or is_action_block
        or input_ids.device.type != "cpu"
        or attention_mask.device.type != "cpu"
        or input_ids.shape != attention_mask.shape
    ):
        return original(
            input_ids,
            attention_mask,
            kv_len=kv_len,
            dtype=dtype,
            is_action_block=is_action_block,
        )

    helper = self.mask_helper
    grid = self._cached_image_grid_thw
    merge_size = int(self.vision_tower.spatial_merge_size)
    helper._is_training = False
    helper._image_grid_thw = grid
    helper._spatial_merge_size = merge_size

    cached = getattr(self, "_rpu_g05_position_cache", None)
    same_grid = cached is not None and (
        (cached.grid is None and grid is None)
        or (
            cached.grid is not None
            and grid is not None
            and torch.equal(cached.grid, grid)
        )
    )
    if (
        cached is not None
        and cached.merge_size == merge_size
        and same_grid
        and torch.equal(cached.attention_mask, attention_mask)
    ):
        helper._mrope_position_deltas = (
            None if cached.mrope_delta is None else cached.mrope_delta.clone()
        )
        if hasattr(helper, "_last_T_pos"):
            helper._last_T_pos = (
                None if cached.last_t is None else cached.last_t.clone()
            )
        return None, cached.position_ids.clone()

    position_ids = helper._build_mrope_position_ids(
        attention_mask,
        attention_mask.device,
    )
    self._rpu_g05_position_cache = types.SimpleNamespace(
        attention_mask=attention_mask.clone(),
        grid=None if grid is None else grid.clone(),
        merge_size=merge_size,
        position_ids=position_ids.clone(),
        mrope_delta=(
            None
            if helper._mrope_position_deltas is None
            else helper._mrope_position_deltas.clone()
        ),
        last_t=(
            None
            if getattr(helper, "_last_T_pos", None) is None
            else helper._last_T_pos.clone()
        ),
    )
    return None, position_ids


def _rpu_g05_vlm_decode(self, hidden: torch.Tensor) -> torch.Tensor:
    """Run the still-host-owned lm_head after the RPU text decoder."""
    weight = self.output_proj.weight
    hidden = hidden.to(device=weight.device, dtype=weight.dtype)
    with torch.autocast(weight.device.type, enabled=False):
        return self._rpu_g05_original_decode(hidden)


@execution_serialized
def _rpu_g05_vision_forward(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    num_frames: int = 1,
    bsz: int = 1,
):
    """Adapt the Qwen3.5 vision result to G0.5's tuple contract."""
    if int(num_frames) != 1 or int(bsz) != 1:
        raise NotImplementedError(
            "G0.5 RPU vision supports only single-frame calls with bsz=1"
        )
    output = self._rpu_g05_vision_forward_impl(hidden_states, grid_thw)
    plan = getattr(self, "_rpu_vision_last_a6_plan", None)
    if isinstance(plan, Mapping):
        receipt = dict(plan)
        receipt.update({
            "stage": "vision",
            "component": _G05_VISION_COMPONENT,
            "generation": int(getattr(self, "_rpu_execution_generation", 0)),
            "authority": "NATIVE_A6_STAGE_DESCRIPTOR",
            "dry_forward_agreement": True,
        })
        vars(self)["_rpu_last_execution_plan"] = receipt
    return output.last_hidden_state, output.pooler_output


def _preflight_g05_vision(model):
    """Validate the complete canonical tower before FP16 cast or swizzle."""
    if getattr(model, "training", True):
        raise NotImplementedError(
            "G0.5 RPU vision supports inference only; call model.eval()"
        )
    vision = getattr(model, "vision_tower", None)
    if vision is None:
        raise TypeError("expected G05ModelQwen35 with model.vision_tower")

    cfg = vision.config
    vision_profile = (
        int(getattr(cfg, "depth", 0)),
        int(getattr(cfg, "num_heads", 0)),
        int(getattr(cfg, "hidden_size", 0)),
        int(getattr(cfg, "intermediate_size", 0)),
        int(getattr(cfg, "patch_size", 0)),
        int(getattr(cfg, "temporal_patch_size", 0)),
        int(getattr(cfg, "spatial_merge_size", 0)),
        int(getattr(cfg, "in_channels", 0)),
        int(getattr(cfg, "out_hidden_size", 0)),
        str(getattr(cfg, "hidden_act", "")),
    )
    if vision_profile != (
        24,
        16,
        1024,
        4096,
        16,
        2,
        2,
        3,
        2048,
        "gelu_pytorch_tanh",
    ):
        raise NotImplementedError(
            "G0.5 RPU vision requires the canonical Qwen3.5 tower profile; "
            f"got {vision_profile}"
        )
    temporal_profile = (
        int(getattr(cfg, "temporal_freq", 0)),
        str(getattr(cfg, "spacetime_mode", "")),
        int(getattr(cfg, "token_drop_layer", None) or cfg.depth),
        getattr(cfg, "temporal_pe_pretrain_frames", None),
        bool(getattr(cfg, "batch_all_cameras", False)),
    )
    if temporal_profile != (0, "factorized", 24, None, False):
        raise NotImplementedError(
            f"G0.5 RPU vision requires a single-frame profile; got {temporal_profile}"
        )
    return vision, cfg, temporal_profile


def _g05_vision_runtime_ready(model) -> bool:
    """Return whether both shared Vision state and the G0.5 wrapper are live."""
    if getattr(model, "_rpu_g05_vision_ready", False) is not True:
        return False
    if getattr(model, "_rpu_g05_vision_install_started", False) is not True:
        return False
    vision = getattr(model, "vision_tower", None)
    if vision is None or not hasattr(vision, "__dict__"):
        return False
    implementation = vars(vision).get("_rpu_g05_vision_forward_impl")
    if not callable(implementation):
        return False
    try:
        vision_adapter = sys.modules.get(
            "rpu_backend.adapters.qwen3_5.vision"
        )
        if vision_adapter is None:
            return False
        runtime_complete = getattr(
            vision_adapter, "_qwen3_5_vision_runtime_complete", None
        )
        if not callable(runtime_complete) or not runtime_complete(
            vision, installed_forward=implementation
        ):
            return False
    except Exception:
        return False
    if int(getattr(getattr(vision, "config", None), "temporal_freq", 0)) != 0:
        return False
    return _instance_method_is(vision, "forward", _rpu_g05_vision_forward)


def patch_g05_vision_for_rpu(model, *, _execution_config=None):
    """Patch the canonical G0.5 single-frame vision tower for RPU."""
    if getattr(model, "_rpu_g05_vision_ready", False):
        if _g05_vision_runtime_ready(model):
            vision = model.vision_tower
            if (
                _execution_config is not None
                and getattr(vision, "_rpu_execution", None)
                != _execution_config
            ):
                raise ValueError(
                    "G0.5 Vision execution config conflicts with the installed runtime"
                )
            return model
        raise RuntimeError(
            "G0.5 Vision ready marker has incomplete runtime state; reload "
            "the checkpoint"
        )
    if getattr(model, "_rpu_g05_vision_install_started", False):
        raise RuntimeError(
            "G0.5 Vision installation was already attempted; reload the "
            "checkpoint to avoid reusing partial wrapper state"
        )
    if not _VISION_PATCH_LOCK.acquire(blocking=False):
        raise RuntimeError("another G0.5 Vision installation is in progress")

    vision = None
    snapshots = ()
    shared_installed = False
    rollback_shared = None
    try:
        if getattr(model, "_rpu_g05_vision_ready", False):
            if _g05_vision_runtime_ready(model):
                return model
            raise RuntimeError(
                "G0.5 Vision ready marker has incomplete runtime state; reload "
                "the checkpoint"
            )
        if getattr(model, "_rpu_g05_vision_install_started", False):
            raise RuntimeError(
                "G0.5 Vision installation was already attempted; reload the "
                "checkpoint to avoid reusing partial wrapper state"
            )
        vision, cfg, _ = _preflight_g05_vision(model)
        if os.environ.get("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED") != "1":
            raise NotImplementedError(
                "generic G0.5 RPU Vision is numeric-blocked; controlled "
                "evaluation requires QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1"
            )
        parameters = getattr(vision, "parameters", None)
        if callable(parameters) and any(
            parameter.device.type != "cpu" for parameter in parameters()
        ):
            raise NotImplementedError(
                "G0.5 RPU Vision patch requires a CPU-hosted checkpoint"
            )
        vision_adapter = importlib.import_module(
            "rpu_backend.adapters.qwen3_5.vision"
        )

        install_vision = vision_adapter.install_qwen3_5_vision_for_rpu
        runtime_complete = getattr(
            vision_adapter,
            "_qwen3_5_vision_runtime_complete",
            lambda _vision: False,
        )
        rollback_shared = getattr(
            vision_adapter,
            "_rollback_qwen3_5_vision_install",
            None,
        )
        if runtime_complete(vision):
            raise RuntimeError(
                "G0.5 Vision found an existing Qwen3.5 runtime; reload the "
                "checkpoint instead of wrapping pre-installed weights"
            )

        snapshots = (
            _snapshot_instance_attrs(
                vision,
                (
                    "forward",
                    "_rpu_g05_vision_forward_impl",
                    "_rpu_execution",
                    "_rpu_execution_component_id",
                    "_rpu_execution_generation",
                    "_execution_generation",
                ),
            )
            + _snapshot_instance_attrs(model, ("_rpu_g05_vision_ready",))
        )
        # Claim and poison before the first FP16 cast/handle creation. The
        # same-model policy/VLM claims are idempotent.
        _claim_live_instance(model)
        model._rpu_g05_vision_install_started = True

        # Cast before the in-place multi-core FP16 swizzle (DWIDTH=2).
        vision.half()
        install_kwargs = {"vision_config": cfg}
        if _execution_config is not None:
            install_kwargs["execution_config"] = _execution_config
        handle = install_vision(vision, **install_kwargs)
        shared_installed = True
        vision._rpu_g05_vision_forward_impl = vision.forward
        vision.forward = types.MethodType(_rpu_g05_vision_forward, vision)
        if _execution_config is not None:
            vision._rpu_execution = _execution_config
        vision._rpu_execution_component_id = _G05_VISION_COMPONENT
        vision._rpu_execution_generation = 0
        vision._execution_generation = 0
        model._rpu_g05_vision_ready = True
        return model
    except BaseException as error:
        if callable(rollback_shared) and (
            shared_installed
            or getattr(vision, "_rpu_vision_installing_handle", None) is not None
            or getattr(vision, "_rpu_vision_handle", None) is not None
        ):
            try:
                rollback_shared(vision)
            except BaseException as cleanup_error:
                error.add_note(f"G0.5 Vision cleanup failed: {cleanup_error!r}")
                raise error
        _restore_instance_attrs(snapshots)
        raise
    finally:
        _VISION_PATCH_LOCK.release()


def _pack_g05_vision(pixel_values, vision):
    """Prepare batch-one single-frame cameras in their declared order."""
    if not isinstance(pixel_values, dict) or not pixel_values:
        raise TypeError("G0.5 RPU vision expects a non-empty camera dict")

    patch_size = int(vision.patch_size)
    temporal_patch_size = int(vision.config.temporal_patch_size)
    merge_size = int(vision.spatial_merge_size)
    camera_info, grids, split_sizes = [], [], []
    num_frames = None
    for camera in pixel_values.values():
        if camera.dim() != 5 or camera.shape[0] != 1 or camera.shape[1] != 1:
            raise NotImplementedError(
                "G0.5 RPU direct vision fusion supports only [1,1,C,H,W]"
            )
        _, camera_frames, channels, height, width = camera.shape
        if num_frames is None:
            num_frames = int(camera_frames)
        elif num_frames != int(camera_frames):
            raise NotImplementedError("all G0.5 cameras must use the same frame count")
        if height % patch_size or width % patch_size:
            raise ValueError("camera H/W must be divisible by the vision patch size")
        grid_h, grid_w = height // patch_size, width // patch_size
        if grid_h % merge_size or grid_w % merge_size:
            raise ValueError("camera patch grid must be divisible by spatial_merge_size")

        camera_info.append(
            (camera.reshape(camera_frames, channels, height, width), grid_h, grid_w)
        )
        grids.append(
            torch.tensor(
                [[1, grid_h, grid_w]] * camera_frames,
                dtype=torch.long,
                device=camera.device,
            )
        )
        split_sizes.append((grid_h // merge_size) * (grid_w // merge_size))

    same_grid = all(
        info[0].shape[1:] == camera_info[0][0].shape[1:]
        for info in camera_info[1:]
    )
    merged_images = None
    if same_grid:
        image_views = [info[0] for info in camera_info]
        first = image_views[0]
        storage_base = first.untyped_storage().data_ptr()
        adjacent = first.is_contiguous() and all(
            image.is_contiguous()
            and image.dtype == first.dtype
            and image.device == first.device
            and image.untyped_storage().data_ptr() == storage_base
            and image.storage_offset()
            == first.storage_offset() + index * first.numel()
            for index, image in enumerate(image_views)
        )
        merged_images = (
            first.as_strided(
                (len(image_views) * first.shape[0], *first.shape[1:]),
                first.stride(),
            )
            if adjacent
            else torch.cat(image_views, dim=0)
        )

    if same_grid:
        pack_groups = [(merged_images, camera_info[0][1], camera_info[0][2])]
    else:
        pack_groups = camera_info

    patches = []
    for images, grid_h, grid_w in pack_groups:
        # The RPU STEP0 consumes FP16. Casting before the two layout
        # materializations preserves the final FP16 bytes and halves both copies.
        if images.dtype == torch.float32:
            images = images.to(torch.float16)
        frames, channels = images.shape[:2]
        images = images.unsqueeze(2).expand(
            -1, -1, temporal_patch_size, -1, -1
        )
        patches.append(
            images.reshape(
                frames,
                temporal_patch_size,
                channels,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .reshape(frames * grid_h * grid_w, -1)
        )

    grid_thw = torch.cat(grids, dim=0)
    packed = patches[0] if len(patches) == 1 else torch.cat(patches, dim=0)
    return packed, grid_thw, split_sizes, num_frames


def _rpu_g05_forward_embed(
    self,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    pixel_values,
    proprio=None,
):
    """Build text/state embeds on RPU and let the merger fill image runs."""
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise NotImplementedError("G0.5 RPU direct fusion supports batch_size == 1")
    if not bool(attention_mask.ne(0).all().item()):
        raise NotImplementedError("G0.5 RPU direct fusion does not support padding")

    token_index = _g05_token_index(self)

    image_tokens = input_ids.eq(int(self.image_token_index))
    if not torch.equal(
        image_tokens.cpu(),
        attention_mask.eq(token_index.IMAGE_TOKEN_INDEX).cpu(),
    ):
        raise ValueError("G0.5 image token IDs and token-type mask disagree")

    torch.ops.rpu.spm_alloc_reset_temporary()
    # One exact entry lives with this inference-only model; its embedding weight
    # is frozen after install. Clone below preserves the fresh-output contract.
    cached = getattr(self, "_rpu_g05_embedding_cache", None)
    if cached is None or not torch.equal(cached[0], input_ids):
        ids = input_ids.reshape(-1).to(dtype=torch.int32, device="rpu").contiguous()
        cached = (
            input_ids.detach().clone(),
            torch.ops.rpu.gather_embedding(
                self._rpu_g05_embed_w_rpu, ids
            ).unsqueeze(0),
        )
        self._rpu_g05_embedding_cache = cached
    # Vision mutates image rows in-place and callers may retain this tensor.
    hidden = cached[1].clone()

    state_token_id = getattr(self.proprio_embedder, "state_token_id", None)
    if state_token_id is not None:
        if not torch.equal(
            input_ids.eq(int(state_token_id)).cpu(),
            attention_mask.eq(token_index.PROPRIO_TOKEN_INDEX).cpu(),
        ):
            raise ValueError("G0.5 state token IDs and token-type mask disagree")
        positions = (input_ids[0].cpu() == int(state_token_id)).nonzero(
            as_tuple=True
        )[0].tolist()
        if positions and proprio is None:
            raise ValueError("G0.5 RPU direct fusion requires proprio for state tokens")
        if positions:
            state = self._rpu_g05_proprio_mlp(
                proprio.to(dtype=torch.float16)
            ).reshape(-1, hidden.shape[-1])
            if len(positions) != state.shape[0] or positions != list(
                range(positions[0], positions[0] + len(positions))
            ):
                raise NotImplementedError(
                    "G0.5 RPU direct fusion requires one contiguous state-token run"
                )
            hidden[0].narrow(0, positions[0], len(positions)).copy_(
                state.to(device="rpu", dtype=torch.float16).contiguous()
            )

    patches, grid_thw, split_sizes, _ = _pack_g05_vision(
        pixel_values, self.vision_tower,
    )
    self._cached_image_grid_thw = grid_thw.contiguous()
    from rpu_backend.adapters.qwen3_5.vision import _visual_run_starts

    starts = _visual_run_starts(
        input_ids.cpu(), int(self.image_token_index), split_sizes
    )
    if starts is None:
        raise NotImplementedError(
            "G0.5 RPU direct fusion requires contiguous tokens for each camera"
        )
    self.vision_tower._rpu_g05_vision_forward_impl(
        patches,
        grid_thw,
        _rpu_fusion_target=hidden,
        _rpu_fusion_run_starts=starts,
    )
    return hidden


def _rpu_g05_build_prefix_action_kv(self, vlm_kv, split_index: int):
    """Expose the stable RPU VLM prefix cache to the action expert."""
    mode = getattr(getattr(self, "cfg", None), "ae_vlm_condition_mode", None)
    if mode != "cross_attn_only":
        raise NotImplementedError(
            "G0.5 RPU action inference requires "
            "ae_vlm_condition_mode=cross_attn_only"
        )
    if getattr(getattr(self, "fm_helper", None), "action_causal", True):
        raise NotImplementedError(
            "G0.5 RPU action inference requires action_causal=false"
        )
    if not hasattr(vlm_kv, "num_items"):
        raise TypeError("G0.5 RPU action prefix must be a SparseKVCache facade")
    prefix_len = int(vlm_kv.num_items())
    if int(split_index) != prefix_len:
        raise NotImplementedError(
            "G0.5 RPU action inference supports only the complete read-only "
            f"VLM prefix; split_index={split_index}, cache_len={prefix_len}"
        )
    if getattr(vlm_kv, "_rpu_qwen3_5_cache", None) is None:
        raise TypeError("G0.5 RPU action prefix is missing its physical VLM cache")
    generation = getattr(vlm_kv, "_rpu_qwen3_5_generation", None)
    current_generation = getattr(self.vlm, "_rpu_g05_cache_generation", None)
    current_cache = getattr(self.vlm, "_rpu_g05_cache", None)
    if (
        generation is None
        or generation != current_generation
        or vlm_kv._rpu_qwen3_5_cache is not current_cache
    ):
        raise RuntimeError(
            "G0.5 RPU action prefix belongs to an older turn; interleaved "
            "InferenceState objects are not supported"
        )
    return types.SimpleNamespace(
        _rpu_g05_prefix_len=prefix_len,
        _rpu_g05_prefix_cache=vlm_kv,
    )


def _ensure_g05_text_cache(vlm):
    """Allocate the single production KV cache, also used by cold planning."""
    from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache

    cache = getattr(vlm, "_rpu_g05_cache", None)
    if cache is None:
        cache = Qwen3_5Cache.from_config(vlm.config, max_seq_len=vlm._rpu_g05_max_seq_len)
        vlm._rpu_g05_cache = cache
    return cache


@execution_serialized
def _rpu_g05_vlm_forward(
    self,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    kv_cache=None,
    past_key_values=None,
    time_cond=None,
    return_kv_cache: bool = False,
    attn_implementation: str = "eager",
    mixture_name=None,
    split_idx=None,
    padding_mask=None,
):
    """Canonical G0.5 VLM prefill/q1-decode through Qwen3.5 fused text."""
    del attn_implementation, mixture_name
    owner = self._rpu_g05_owner()
    if owner is None:
        raise RuntimeError("G0.5 RPU model owner no longer exists")
    if owner.training:
        raise NotImplementedError("G0.5 RPU bring-up supports inference only")
    if inputs_embeds.shape[0] != 1:
        raise NotImplementedError("G0.5 RPU VLM currently supports batch_size == 1")
    if past_key_values is not None or kv_cache is None:
        raise TypeError("G0.5 RPU VLM requires its SparseKVCache via kv_cache")
    if not return_kv_cache:
        raise NotImplementedError("G0.5 RPU VLM requires return_kv_cache=True")
    if time_cond is not None or split_idx is not None:
        raise NotImplementedError("G0.5 RPU VLM bring-up supports inference prefill/decode only")
    if padding_mask is not None and not bool(padding_mask.bool().all().item()):
        raise NotImplementedError("G0.5 RPU VLM does not support padded tokens")

    from rpu_backend.api.cache import RPUCache
    from rpu_backend.adapters.qwen3_5.text import (
        _normalize_prefill_position_ids,
        run_qwen3_5_text,
    )

    cache = getattr(kv_cache, "_rpu_qwen3_5_cache", None)
    new_turn = cache is None
    if cache is None:
        if kv_cache.num_items() != 0:
            raise ValueError("G0.5 RPU cache must be installed before it contains tokens")
        cache = getattr(self, "_rpu_g05_cache", None)
        if cache is None:
            cache = _ensure_g05_text_cache(self)
        else:
            # Decode GraphCache replay bakes cache DDR addresses. Keep one cache
            # allocation per installed model. Prefill overwrites full-attention
            # KV before reading it. A multi-token prefill zeros recurrent/conv
            # state in the first C++ chunk; decode-first keeps the host fallback.
            RPUCache.reset_to_position(cache, 0)
            if inputs_embeds.shape[1] == 1:
                for state in cache.recurrent_states:
                    state.zero_()
                for state in cache.conv_states:
                    state.zero_()
        generation = getattr(self, "_rpu_g05_cache_generation", 0) + 1
        self._rpu_g05_cache_generation = generation
        kv_cache._rpu_qwen3_5_cache = cache
        kv_cache._rpu_qwen3_5_generation = generation
        kv_cache.num_items = types.MethodType(_g05_cache_num_items, kv_cache)
    elif kv_cache._rpu_qwen3_5_generation != self._rpu_g05_cache_generation:
        raise RuntimeError(
            "G0.5 RPU cache belongs to an older turn; interleaved InferenceState "
            "objects are not supported"
        )

    start = int(cache.position)
    _check_causal_mask(
        attention_mask,
        query_len=int(inputs_embeds.shape[1]),
        past_len=start,
    )
    if new_turn:
        torch.ops.rpu.qwen3_5_set_mrope_position_delta(
            self._rpu_qwen3_5.handle,
            0,
        )
    caller_device = inputs_embeds.device
    if caller_device.type == "rpu":
        hidden = inputs_embeds.to(dtype=torch.float16).contiguous()
    else:
        hidden = inputs_embeds.to(device="cpu", dtype=torch.float16)
        hidden = hidden.to(device="rpu").contiguous()
    hidden = run_qwen3_5_text(
        self,
        hidden,
        cache,
        position_ids=position_ids,
    )
    plan = getattr(self, "_rpu_last_execution_plan", None)
    if isinstance(plan, Mapping):
        receipt = dict(plan)
        receipt.update({
            "component": _G05_TEXT_COMPONENT,
            "generation": int(getattr(self, "_rpu_execution_generation", 0)),
            "authority": "NATIVE_A6_STAGE_DESCRIPTOR",
            "dry_forward_agreement": True,
        })
        vars(self)["_rpu_last_execution_plan"] = receipt

    # pi0fast decode advances all three M-RoPE lanes from the largest prefill T.
    if start == 0 and inputs_embeds.shape[1] > 1 and position_ids is not None:
        pid = _normalize_prefill_position_ids(
            position_ids,
            real_len=inputs_embeds.shape[1],
            padded_len=inputs_embeds.shape[1],
        )
        delta = int(pid[0].max().item()) + 1 - int(cache.position)
        torch.ops.rpu.qwen3_5_set_mrope_position_delta(
            self._rpu_qwen3_5.handle,
            delta,
        )

    # FusedModelBase owns a shape-keyed reusable output buffer. Return a fresh
    # caller-device tensor because G0.5 keeps last_hidden across later forwards.
    if caller_device.type == "rpu":
        hidden = hidden.clone()
    else:
        hidden = hidden.to("cpu")
        if caller_device.type != "cpu":
            hidden = hidden.to(caller_device)
    return hidden, kv_cache


def _g05_vlm_runtime_ready(model) -> bool:
    """Return whether the G0.5 VLM publication points to a live text runtime."""
    if getattr(model, "_rpu_g05_vlm_ready", False) is not True:
        return False
    if getattr(model, "_rpu_g05_vlm_install_started", False) is not True:
        return False
    vlm = getattr(model, "vlm", None)
    if vlm is None or not hasattr(vlm, "__dict__"):
        return False
    if getattr(vlm, "_rpu_qwen3_5_text_install_started", False) is not True:
        return False
    state = getattr(vlm, "_rpu_qwen3_5", None)
    if state is None or getattr(state, "handle", None) is None:
        return False
    from rpu_backend.adapters.qwen3_5.text import _qwen35_retirement_ready

    if not _qwen35_retirement_ready(state):
        return False
    for name in ("graph_cache", "prefill_graph", "prefill_graph_cache"):
        if getattr(state, name, None) is None:
            return False
    for name in (
        "prefill_graph_sig",
        "prefill_debug_graph_cache",
        "prefill_debug_graph_sig",
        "prefill_rope_cache",
        "rotary_dim",
        "rope_theta",
        "mrope_section",
        "padding_budget",
    ):
        if not hasattr(state, name):
            return False
    max_seq_len = getattr(vlm, "_rpu_g05_max_seq_len", None)
    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        return False
    if getattr(state, "max_seq_len", None) != max_seq_len:
        return False
    try:
        metadata_valid = (
            int(getattr(state, "num_layers", 0)) > 0
            and int(getattr(state, "hidden_size", 0)) > 0
            and int(state.rotary_dim) > 0
            and float(state.rope_theta) > 0.0
            and len(tuple(state.mrope_section)) > 0
            and int(state.padding_budget) >= 0
        )
    except (TypeError, ValueError):
        return False
    if not metadata_valid:
        return False
    owner = vars(vlm).get("_rpu_g05_owner")
    if not isinstance(owner, weakref.ReferenceType) or owner() is not model:
        return False
    if not callable(vars(vlm).get("_rpu_g05_original_decode")):
        return False
    if not callable(
        vars(model).get("_rpu_g05_original_build_causal_mask_and_position_ids")
    ):
        return False
    if not _instance_method_is(vlm, "forward", _rpu_g05_vlm_forward):
        return False
    if not _instance_method_is(vlm, "decode", _rpu_g05_vlm_decode):
        return False
    if not _instance_method_is(
        model,
        "build_causal_mask_and_position_ids",
        _rpu_g05_build_causal_mask_and_position_ids,
    ):
        return False
    if "_rpu_g05_prefix_bridge_installed" not in vars(model):
        return False
    if vars(model)["_rpu_g05_prefix_bridge_installed"] is False:
        return True
    if vars(model)["_rpu_g05_prefix_bridge_installed"] is not True:
        return False
    return _instance_method_is(
        model, "_build_prefix_action_kv", _rpu_g05_build_prefix_action_kv
    )


def patch_g05_vlm_for_rpu(
    model,
    *,
    max_seq_len: int = 2048,
    _execution_config=None,
):
    """Patch ``G05ModelQwen35.vlm`` for FP16 RPU prefill and q1 decode.

    The caller keeps vision, embedding assembly, lm_head, and policy glue on
    their original device. The mutation is irreversible; reload the checkpoint
    for a CPU/CUDA reference model.
    """
    if getattr(model, "_rpu_g05_vlm_ready", False):
        if _g05_vlm_runtime_ready(model):
            if (
                _execution_config is not None
                and getattr(model.vlm, "_rpu_execution", None)
                != _execution_config
            ):
                raise ValueError(
                    "G0.5 VLM execution config conflicts with the installed runtime"
                )
            return model
        raise RuntimeError(
            "G0.5 VLM ready marker has incomplete runtime state; reload the "
            "checkpoint"
        )
    if getattr(model, "_rpu_g05_vlm_install_started", False):
        raise RuntimeError(
            "G0.5 VLM installation was already attempted; reload the checkpoint "
            "to avoid double-swizzle"
        )
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    parameters = getattr(getattr(model, "vlm", None), "parameters", None)
    if callable(parameters) and any(
        parameter.device.type != "cpu" for parameter in parameters()
    ):
        raise NotImplementedError(
            "G0.5 RPU VLM patch requires a CPU-hosted checkpoint"
        )
    from rpu_backend.adapters.qwen3_5.text import install_qwen3_5_text_for_rpu

    if not _INSTALL_LOCK.acquire(blocking=False):
        raise RuntimeError("another G0.5 VLM installation is in progress")

    vlm = None
    state_before = None
    snapshots = ()
    try:
        vlm = _preflight_g05_vlm(model)
        state_before = getattr(vlm, "_rpu_qwen3_5", None)
        if (
            state_before is not None
            or getattr(vlm, "_rpu_qwen3_5_text_install_started", False)
        ):
            raise RuntimeError(
                "G0.5 VLM has an existing or previously attempted Qwen3.5 "
                "runtime; reload the checkpoint"
            )
        snapshots = tuple(
            (owner, name, name in vars(owner), vars(owner).get(name))
            for owner, names in (
                (
                    vlm,
                    (
                        "forward",
                        "decode",
                        "_rpu_g05_max_seq_len",
                        "_rpu_g05_owner",
                        "_rpu_g05_original_decode",
                        "_rpu_execution",
                        "_rpu_execution_component_id",
                        "_rpu_execution_generation",
                    ),
                ),
                (
                    model,
                    (
                        "build_causal_mask_and_position_ids",
                        "_build_prefix_action_kv",
                        "_rpu_g05_prefix_bridge_installed",
                        "_rpu_g05_original_build_causal_mask_and_position_ids",
                        "_rpu_g05_vlm_ready",
                    ),
                ),
            )
            for name in names
        )
        _claim_live_instance(model)
        model._rpu_g05_vlm_install_started = True

        # RC-1/P1: dtype first, swizzle inside the shared installer second.
        # Restrict the cast to the fused backbone so a direct caller cannot
        # enqueue eager casts on unrelated VLM modules already placed on RPU.
        vlm.layers.to(device="cpu", dtype=torch.float16)
        vlm.norm.to(device="cpu", dtype=torch.float16)
        handle = install_qwen3_5_text_for_rpu(
            vlm,
            vlm.config,
            max_seq_len,
            execution_config=_execution_config,
            _cpu_stage_weights=True,
        )
        torch.ops.rpu.qwen3_5_set_linear_acc32(handle, True)
        torch.ops.rpu.qwen3_5_set_fast_replay(handle, True)
        torch.ops.rpu.qwen3_5_set_retained_prefill_graph(handle, True)
        torch.ops.rpu.qwen3_5_enable_execution_reconfigure(handle)
        import rpu_backend as _rb

        vlm._rpu_qwen3_5.prefill_graph_cache = _rb.graph.GraphCache(max_entries=1)
        vlm._rpu_qwen3_5.prefill_graph_sig = None
        # Debug prefill is lazy and uses a separate cache because its graph has
        # diagnostic-only fixed-address SPM→DDR taps. Never replay it as the
        # production prefill graph (or vice versa).
        vlm._rpu_qwen3_5.prefill_debug_graph_cache = None
        from rpu_backend.adapters.qwen3_5.text import _refresh_qwen35_text_retirement

        _refresh_qwen35_text_retirement(vlm._rpu_qwen3_5)
        vlm._rpu_qwen3_5.prefill_debug_graph_sig = None
        vlm._rpu_qwen3_5.prefill_rope_cache = None

        vlm._rpu_g05_max_seq_len = int(max_seq_len)
        vlm._rpu_g05_owner = weakref.ref(model)
        if _execution_config is not None:
            vlm._rpu_execution = _execution_config
        vlm._rpu_execution_component_id = _G05_TEXT_COMPONENT
        vlm._rpu_execution_generation = 0
        vlm._rpu_qwen3_5.execution_generation = 0
        vlm._rpu_g05_original_decode = vlm.decode
        vlm.forward = types.MethodType(_rpu_g05_vlm_forward, vlm)
        vlm.decode = types.MethodType(_rpu_g05_vlm_decode, vlm)
        model._rpu_g05_original_build_causal_mask_and_position_ids = (
            model.build_causal_mask_and_position_ids
        )
        model.build_causal_mask_and_position_ids = types.MethodType(
            _rpu_g05_build_causal_mask_and_position_ids,
            model,
        )
        prefix_bridge_installed = hasattr(model, "_build_prefix_action_kv")
        if prefix_bridge_installed:
            model._build_prefix_action_kv = types.MethodType(
                _rpu_g05_build_prefix_action_kv,
                model,
            )
        model._rpu_g05_prefix_bridge_installed = prefix_bridge_installed
        model._rpu_g05_vlm_ready = True
        return model
    except BaseException as error:
        state = getattr(vlm, "_rpu_qwen3_5", None) if vlm is not None else None
        if state is not None and state is not state_before:
            from rpu_backend.adapters.qwen3_5.text import _retire_failed_qwen3_5_text_install

            try:
                _retire_failed_qwen3_5_text_install(vlm, state=state)
            except BaseException as cleanup_error:
                error.add_note(f"G0.5 VLM install cleanup failed: {cleanup_error!r}")
        for owner, name, had_value, value in reversed(snapshots):
            if had_value:
                vars(owner)[name] = value
            else:
                vars(owner).pop(name, None)
        # Swizzle/install mutation is irreversible. Keep both poison markers
        # and the claimed process owner so another live policy cannot enter.
        raise
    finally:
        _INSTALL_LOCK.release()


def _rpu_g05_encode_inference(
    self,
    samples,
    device,
    mode: str = "fm",
    training: bool = False,
    **kwargs,
):
    """Skip the unused ActionCodec pass in continuous AR-context prefill."""
    original = self._rpu_g05_original_encode_inference
    if (
        mode != "ar"
        or training
        or kwargs.get("template") is not None
        or not bool(self.batchify_action)
        or not isinstance(device, torch.device)
        or device.type != "cpu"
        or not isinstance(samples, list)
        or len(samples) != 1
    ):
        return original(samples, device=device, mode=mode, training=training, **kwargs)

    sample = samples[0]
    if not isinstance(sample, dict):
        return original(samples, device=device, mode=mode, training=training, **kwargs)
    action = sample.get("action")
    if not isinstance(action, dict) or not all(
        isinstance(action.get(key), torch.Tensor)
        for key in ("value", "action_op_mask", "action_dim_is_pad")
    ):
        return original(samples, device=device, mode=mode, training=training, **kwargs)
    retained = self._prepare_segments(samples, control_flag="context_only")[0]
    if any(
        segment.sample_key == "action" or segment.processor_key == "action"
        for segment in retained
    ):
        return original(samples, device=device, mode=mode, training=training, **kwargs)

    if self._rpu_g05_action_rvq.training:
        rows = self._rpu_g05_action_rng_rows
        torch.rand(rows, device=device)
        torch.randint(
            1,
            self._rpu_g05_action_rng_codebooks + 1,
            (rows,),
            device=device,
        )
    stripped = [dict(sample)]
    stripped[0].pop("action")
    return original(stripped, device=device, mode=mode, training=training, **kwargs)


def _rpu_g05_data_preprocess(self, data):
    """Avoid copying read-only raw images while deriving the unused gt_action."""
    images = data.get("images") if isinstance(data, dict) else None
    if (
        self.is_train
        or not isinstance(data, dict)
        or "action" not in data
        or not isinstance(images, dict)
        or not all(isinstance(image, torch.Tensor) for image in images.values())
    ):
        return self._rpu_g05_original_preprocess(data)

    data["images"] = _G05SharedImages(images)
    try:
        return self._rpu_g05_original_preprocess(data)
    finally:
        data["images"] = images


def _rpu_g05_to_tensor(self, image):
    """Official uint8 conversion without its redundant FP32 division output."""
    if image.dtype != torch.uint8:
        raise AssertionError("G0.5 ToTensor expects uint8 input")
    return image.to(torch.float32).div_(255.0)


def _rpu_g05_process_images(self, data):
    """Run identical camera transforms once over the canonical merged batch."""
    images = data.get("images") if isinstance(data, dict) else None
    values = (
        [images.get(meta["key"]) for meta in self._rpu_g05_image_meta]
        if isinstance(images, dict)
        else []
    )
    if (
        self.is_train
        or len(values) != 3
        or any(not isinstance(image, torch.Tensor) for image in values)
        or any(
            image.device.type != "cpu"
            or image.dtype != torch.uint8
            or image.dim() != 4
            or int(image.shape[0]) != int(self.num_obs_steps)
            or not image.is_contiguous()
            for image in values
        )
        or len({tuple(image.shape) for image in values}) != 1
    ):
        return self._rpu_g05_original_process_images(data)

    merged = torch.cat(values, dim=0)
    for transform in self._rpu_g05_image_transforms:
        merged = transform(merged)
    frames = int(self.num_obs_steps)
    return {
        meta["camera_type"]: merged.narrow(0, index * frames, frames)
        for index, meta in enumerate(self._rpu_g05_image_meta)
    }


def patch_g05_data_processor_for_rpu(processor):
    """Install the exact, eval-only G0.5 host preprocessing fast path."""
    if getattr(processor, "_rpu_g05_data_processor_ready", False):
        return processor
    processor_type = (type(processor).__module__, type(processor).__name__)
    if processor_type != (
        "g05.data_processor.processor.galaxea_cot_processor",
        "GalaxeaCoTProcessor",
    ):
        raise TypeError(
            "G0.5 RPU host preprocessing requires GalaxeaCoTProcessor; "
            f"got {processor_type}"
        )
    if getattr(processor, "is_train", None) is not False:
        raise NotImplementedError("G0.5 RPU host preprocessing requires processor.eval()")
    if (
        int(getattr(processor, "num_obs_steps", 0)) != 1
        or int(getattr(processor, "num_output_cameras", 0))
        != 3 * int(processor.num_obs_steps)
        or int(getattr(processor, "action_horizon", 0)) != 32
    ):
        raise NotImplementedError(
            "G0.5 RPU host preprocessing supports only three single-frame cameras, "
            "and action_horizon=32"
        )

    image_meta = list(getattr(processor, "shape_meta", {}).get("images", ()))
    transforms = getattr(processor, "val_transforms", None)
    if (
        len(image_meta) != 3
        or len({meta.get("key") for meta in image_meta}) != 3
        or len({meta.get("camera_type") for meta in image_meta}) != 3
        or not isinstance(transforms, dict)
    ):
        raise NotImplementedError(
            "G0.5 RPU host preprocessing requires three canonical image transforms"
        )
    if not callable(getattr(processor, "process_images", None)):
        raise TypeError("G0.5 data processor is missing process_images()")
    transform_signature = None
    expected_types = (
        ("torchvision.transforms.transforms", "Resize"),
        ("g05.data_processor.transforms.image", "ToTensor"),
        ("torchvision.transforms.transforms", "Normalize"),
    )
    for meta in image_meta:
        chain = transforms.get(meta.get("key"), ())
        actual_types = tuple(
            (type(transform).__module__, type(transform).__name__)
            for transform in chain
        )
        if actual_types != expected_types:
            raise NotImplementedError(
                "G0.5 RPU host preprocessing requires "
                "Resize -> ToTensor -> Normalize for every camera"
            )
        resize, to_tensor, normalize = chain
        signature = (
            tuple(int(value) for value in resize.size),
            getattr(resize, "interpolation", None),
            getattr(resize, "max_size", None),
            getattr(resize, "antialias", None),
        )
        if (
            signature[0] != tuple(int(value) for value in meta["shape"][-2:])
            or tuple(float(value) for value in normalize.mean) != (0.5, 0.5, 0.5)
            or tuple(float(value) for value in normalize.std) != (0.5, 0.5, 0.5)
            or not callable(getattr(to_tensor, "forward", None))
        ):
            raise NotImplementedError(
                "G0.5 RPU host preprocessing requires canonical resize and normalization"
            )
        if transform_signature is not None and signature != transform_signature:
            raise NotImplementedError(
                "G0.5 RPU host preprocessing requires identical camera resize transforms"
            )
        transform_signature = signature

    fast_transforms = tuple(transforms[image_meta[0]["key"]])
    processor._rpu_g05_original_preprocess = processor.preprocess
    processor._rpu_g05_original_process_images = processor.process_images
    processor._rpu_g05_image_meta = tuple(image_meta)
    processor._rpu_g05_image_transforms = fast_transforms
    processor.preprocess = types.MethodType(_rpu_g05_data_preprocess, processor)
    processor.process_images = types.MethodType(_rpu_g05_process_images, processor)
    fast_transforms[1].forward = types.MethodType(
        _rpu_g05_to_tensor, fast_transforms[1]
    )
    fast_transforms[2].inplace = True
    processor._rpu_g05_data_processor_ready = True
    return processor


def _g05_dead_action_pipeline(processor):
    """Return cold pipeline objects when padded actions cannot affect context."""
    if getattr(processor, "is_train", True):
        return None
    if (
        (type(processor.action_filter).__module__, type(processor.action_filter).__name__)
        not in {
            ("g05.data_processor.transforms.action_filter", "BaseActionFilter"),
            ("g05.data_processor.transforms.action_filter", "DummyActionFilter"),
        }
        or getattr(processor, "_vlm_input_action_normalizer", None) is not None
    ):
        return None

    transforms = tuple(getattr(processor, "action_state_transforms", ()) or ())
    if len(transforms) != 1 or (
        type(transforms[0]).__module__, type(transforms[0]).__name__
    ) != (
        "g05.data_processor.transforms.relative_action",
        "RelativeJointTransform",
    ) or not bool(getattr(transforms[0], "fast_forward", False)):
        return None
    normalizer = getattr(processor, "_normalizer", None)
    merger = getattr(processor, "action_state_merger", None)
    if (
        (type(normalizer).__module__, type(normalizer).__name__)
        != ("g05.utils.data.normalizer", "LinearNormalizer")
        or (type(merger).__module__, type(merger).__name__)
        != (
            "g05.data_processor.transforms.action_state_merger",
            "GroupedPaddingMerger",
        )
    ):
        return None

    builder = getattr(processor, "samples_builder", None)
    builder_type = (type(builder).__module__, type(builder).__name__)
    if builder_type == (
        "g05.data_processor.processor.samples_builder",
        "MixedSamplesBuilder",
    ):
        if getattr(builder, "_training", True):
            return None
        active_builder = getattr(builder, "_eval_builder", None)
    else:
        active_builder = builder
    if (type(active_builder).__module__, type(active_builder).__name__) != (
        "g05.data_processor.processor.samples_builder",
        "BaseSamplesBuilder",
    ):
        return None

    template = active_builder.template
    if not isinstance(template, str):
        return None
    merger_signature = (
        repr(getattr(merger, "_raw_max_action_shape_meta", None)),
        repr(getattr(merger, "merge_spec", None)),
        bool(getattr(merger, "merge", False)),
    )
    refs = (
        processor,
        processor.action_filter,
        transforms[0],
        normalizer,
        merger,
        builder,
        active_builder,
    )
    return refs, (template, merger_signature)


def _g05_padded_action_schema(obs_dict):
    action = obs_dict.get("action") if isinstance(obs_dict, dict) else None
    action_is_pad = obs_dict.get("action_is_pad") if isinstance(obs_dict, dict) else None
    if (
        not isinstance(action, dict)
        or not action
        or not isinstance(action_is_pad, torch.Tensor)
        or action_is_pad.device.type != "cpu"
        or action_is_pad.dtype != torch.bool
        or tuple(action_is_pad.shape) != (32,)
        or not bool(action_is_pad.all())
        or any(
            key in obs_dict
            for key in (
                "action_dim_is_pad",
                "action_op_mask",
                "action_parts_meta",
                "gt_action",
                "vlm_action",
            )
        )
    ):
        return None

    schema = []
    for key, value in sorted(action.items()):
        if (
            not isinstance(key, str)
            or not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or not value.dtype.is_floating_point
            or value.ndim != 2
            or int(value.shape[0]) != 32
            or not bool(torch.isfinite(value).all())
        ):
            return None
        schema.append((key, tuple(value.shape), value.dtype))
    return tuple(schema)


def _g05_action_prepare_payload(model_processor, prepared):
    sample = getattr(prepared, "sample", None)
    nested = sample.get("samples") if isinstance(sample, dict) else None
    action = nested.get("action") if isinstance(nested, dict) else None
    if (
        not isinstance(action, dict)
        or not all(
            isinstance(action.get(key), torch.Tensor)
            for key in ("value", "action_dim_is_pad", "action_op_mask")
        )
        or not isinstance(sample.get("action_dim_is_pad"), torch.Tensor)
        or not isinstance(sample.get("action_op_mask"), torch.Tensor)
    ):
        return None
    retained = model_processor._prepare_segments(
        [nested], control_flag="context_only"
    )[0]
    if any(
        segment.sample_key == "action" or segment.processor_key == "action"
        for segment in retained
    ):
        return None

    fields = {
        key: copy.deepcopy(sample[key])
        for key in ("action_dim_is_pad", "action_op_mask", "action_parts_meta")
        if key in sample
    }
    return fields, copy.deepcopy(action)


def _g05_context_ignores_action(model_processor, sample):
    retained = model_processor._prepare_segments(
        [sample], control_flag="context_only"
    )[0]
    return not any(
        segment.sample_key == "action" or segment.processor_key == "action"
        for segment in retained
    )


def _rpu_g05_prepare(self, obs_dict):
    """Reuse only dead, fully-padded action metadata around official preprocess."""
    original = self._rpu_g05_original_prepare
    policy = getattr(self, "policy", None)
    model_processor = getattr(policy, "processor", None)
    encode = getattr(model_processor, "encode_inference", None)
    if (
        not getattr(policy, "_rpu_g05_policy_ready", False)
        or not getattr(self, "_rpu_g05_prepare_cache_allowed", False)
        or not bool(getattr(model_processor, "batchify_action", False))
        or getattr(encode, "__func__", None) is not _rpu_g05_encode_inference
        or not callable(getattr(model_processor, "_prepare_segments", None))
    ):
        return original(obs_dict)

    from g05.models.g05.inferencer import resolve_processor

    sub_processor = resolve_processor(self.processor, obs_dict)
    pipeline = _g05_dead_action_pipeline(sub_processor)
    schema = _g05_padded_action_schema(obs_dict)
    if pipeline is None or schema is None:
        return original(obs_dict)
    refs, pipeline_signature = pipeline
    if not _g05_context_ignores_action(
        model_processor, {"template": pipeline_signature[0]}
    ):
        return original(obs_dict)
    prepare_segments = model_processor._prepare_segments
    refs = refs + (
        model_processor,
        getattr(prepare_segments, "__func__", prepare_segments),
    )
    key = (pipeline_signature, schema)

    cached = self._rpu_g05_action_prepare_cache
    if (
        cached is not None
        and cached[1] == key
        and len(cached[0]) == len(refs)
        and all(actual is expected for actual, expected in zip(cached[0], refs))
    ):
        stripped = dict(obs_dict)
        stripped.pop("action")
        stripped.pop("action_is_pad")
        prepared = original(stripped)
        sample = getattr(prepared, "sample", None)
        nested = sample.get("samples") if isinstance(sample, dict) else None
        if getattr(prepared, "sub_processor", None) is sub_processor and isinstance(
            nested, dict
        ) and nested.get("template") == pipeline_signature[0]:
            fields, action = copy.deepcopy(cached[2])
            sample.update(fields)
            nested["action"] = action
            prepared.obs_dict = obs_dict
            return prepared
        self._rpu_g05_action_prepare_cache = None
        raise RuntimeError("G0.5 padded-action prepare cache contract drifted")

    self._rpu_g05_action_prepare_cache = None
    prepared = original(obs_dict)
    if getattr(prepared, "sub_processor", None) is sub_processor:
        payload = _g05_action_prepare_payload(model_processor, prepared)
        if payload is not None:
            self._rpu_g05_action_prepare_cache = (refs, key, payload)
    return prepared


def _rpu_g05_infer_with_timing(self, obs_dicts):
    """Scope the prepare cache to the fused path's batch-one contract."""
    previous = self._rpu_g05_prepare_cache_allowed
    self._rpu_g05_prepare_cache_allowed = (
        isinstance(obs_dicts, list) and len(obs_dicts) == 1
    )
    try:
        return self._rpu_g05_original_infer_with_timing(obs_dicts)
    finally:
        self._rpu_g05_prepare_cache_allowed = previous


def _g05_collate_one(value):
    if isinstance(value, dict):
        return {
            nested_key: _g05_collate_one(nested_value)
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, torch.Tensor):
        return value.unsqueeze(0)
    try:
        return torch.utils.data.default_collate([value])
    except Exception:
        return [value]


def _rpu_g05_collate(self, samples, padding_input_id):
    first = samples[0] if len(samples) == 1 else None
    if (
        not isinstance(first, dict)
        or "samples" not in first
        or "gt_action" in first
        or not isinstance(first.get("pixel_values"), dict)
        or {"input_ids", "labels", "attention_mask"} <= set(first)
    ):
        return self._rpu_g05_original_collate(samples, padding_input_id)

    collated = {
        key: _g05_collate_one(value)
        for key, value in first.items()
        if key != "samples"
    }
    collated["samples"] = [first["samples"]]
    sample_meta_keys = ("idx", "task", "embodiment", "dataset_locator", "frequency")
    collated["sample_meta"] = [
        {key: first.get(key) for key in sample_meta_keys if key in first}
    ]
    return collated


def _rpu_g05_forward_batch(self, batch):
    """Skip the official CPU-to-CPU tensor walks for the patched policy."""
    if not getattr(self.policy, "_rpu_g05_policy_ready", False):
        return self._rpu_g05_original_forward_batch(batch)
    with torch.no_grad():
        return self.policy.predict_action(batch)


def patch_g05_inferencer_for_rpu(inferencer):
    """Use tensor views for the canonical batch-1 G0.5 collation path."""
    if getattr(inferencer, "_rpu_g05_inferencer_ready", False):
        return inferencer
    inferencer_type = (type(inferencer).__module__, type(inferencer).__name__)
    if inferencer_type != (
        "g05.models.g05.inferencer",
        "PolicyInferencer",
    ):
        raise TypeError(
            "G0.5 RPU collation requires PolicyInferencer; "
            f"got {inferencer_type}"
        )
    if torch.device(inferencer.device).type != "cpu":
        raise NotImplementedError("G0.5 RPU collation requires a CPU host inferencer")
    registered = getattr(inferencer.processor, "processors", None)
    processors = tuple(registered.values()) if isinstance(registered, dict) else (
        inferencer.processor,
    )
    if not processors or not all(
        getattr(processor, "_rpu_g05_data_processor_ready", False)
        for processor in processors
    ):
        raise RuntimeError(
            "patch_g05_data_processor_for_rpu() must run before patching the inferencer"
        )

    if not callable(getattr(inferencer, "_prepare", None)) or not callable(
        getattr(inferencer, "infer_with_timing", None)
    ):
        raise TypeError("G0.5 RPU inferencer is missing prepare/timing methods")
    inferencer._rpu_g05_original_prepare = inferencer._prepare
    inferencer._rpu_g05_original_infer_with_timing = inferencer.infer_with_timing
    inferencer._rpu_g05_action_prepare_cache = None
    inferencer._rpu_g05_prepare_cache_allowed = True
    inferencer._prepare = types.MethodType(_rpu_g05_prepare, inferencer)
    inferencer.infer_with_timing = types.MethodType(
        _rpu_g05_infer_with_timing, inferencer
    )
    inferencer._rpu_g05_original_collate = inferencer._collate
    inferencer._rpu_g05_original_forward_batch = inferencer.forward_batch
    inferencer._collate = types.MethodType(_rpu_g05_collate, inferencer)
    inferencer.forward_batch = types.MethodType(_rpu_g05_forward_batch, inferencer)
    inferencer._rpu_g05_inferencer_ready = True
    return inferencer


def _clear_g05_graph(cache) -> None:
    if cache is None:
        return
    cache.begin_warmup()
    cache.clear()
    invariant = getattr(cache, "cache_invariant_ok", None)
    if callable(invariant) and not invariant():
        raise RuntimeError("G0.5 GraphCache invariant failed during reconfigure")


def _publish_g05_execution_views(
    policy, resolved, *, generation: int, component_generations
) -> None:
    """Publish one root generation and independent child generations."""
    root, components = resolved
    model = policy.model
    policy._rpu_execution = root
    model._rpu_execution = root
    policy._rpu_execution_generation = int(generation)
    policy._rpu_g05_execution_components = dict(components)
    policy._rpu_g05_execution_component_generations = dict(
        component_generations
    )
    children = {
        _G05_VISION_COMPONENT: getattr(model, "vision_tower", None),
        _G05_TEXT_COMPONENT: getattr(model, "vlm", None),
        _G05_ACTION_COMPONENT: getattr(model, "action_expert", None),
    }
    for component, child in children.items():
        if child is None:
            continue
        child_state = vars(child)
        child_state["_rpu_execution"] = components[component]
        child_state["_rpu_execution_component_id"] = component
        child_state["_rpu_execution_generation"] = int(
            component_generations[component]
        )

    text = children[_G05_TEXT_COMPONENT]
    text_state = getattr(text, "_rpu_qwen3_5", None)
    if text_state is not None:
        text_state.execution_config = components[_G05_TEXT_COMPONENT]
        text_state.execution_generation = int(
            component_generations[_G05_TEXT_COMPONENT]
        )
    vision = children[_G05_VISION_COMPONENT]
    if getattr(vision, "_rpu_vision_handle", None) is not None:
        vision._execution_generation = int(
            component_generations[_G05_VISION_COMPONENT]
        )
    action = children[_G05_ACTION_COMPONENT]
    action_state = getattr(action, "_rpu_qwen3_5", None)
    if action_state is not None:
        action_state.execution_config = components[_G05_ACTION_COMPONENT]
        action_state.execution_generation = int(
            component_generations[_G05_ACTION_COMPONENT]
        )


def _validate_g05_execution_reconfigure(policy, execution_config) -> None:
    """Validate every child and required native transaction API."""
    _resolve_g05_execution(
        execution_config,
        max_seq_len=policy.model.vlm._rpu_g05_max_seq_len,
        entry_point="G0.5 policy reconfigure",
    )
    required = (
        "execution_reconfigure_begin",
        "execution_reconfigure_commit",
        "execution_reconfigure_abort",
        "execution_reconfigure_abort_attempt",
        "qwen3_5_stage_prefill_execution_controls",
        "qwen3_5_stage_action_chunk_size",
    )
    if getattr(policy.model.vision_tower, "_rpu_vision_handle", None) is not None:
        required += ("qwen3_5_vision_stage_prefill_execution_controls",)
    missing = [name for name in required if not hasattr(torch.ops.rpu, name)]
    if missing:
        raise RuntimeError(
            "G0.5 binary lacks execution-reconfigure op(s): "
            + ", ".join(missing)
        )


def _apply_g05_execution_config(
    policy,
    execution_config,
    generation: int,
    *,
    force_native: bool = False,
    forced_component_generations=None,
) -> None:
    """Atomically stage native child controls, then rotate their graph owners."""
    policy._rpu_g05_execution_reconfigure_journal = {"mutation_started": False}
    from rpu_backend.api._execution import native_execution_reconfigure

    model = policy.model
    max_seq_len = model.vlm._rpu_g05_max_seq_len
    resolved = _resolve_g05_execution(
        execution_config,
        max_seq_len=max_seq_len,
        entry_point="G0.5 policy reconfigure",
    )
    _root, components = resolved
    old_components = getattr(policy, "_rpu_g05_execution_components", {})
    changed = {
        component
        for component in _G05_EXECUTION_COMPONENTS
        if force_native or old_components.get(component) != components[component]
    }
    old_generations = dict(getattr(
        policy,
        "_rpu_g05_execution_component_generations",
        {component: 0 for component in _G05_EXECUTION_COMPONENTS},
    ))
    if forced_component_generations is None:
        next_generations = dict(old_generations)
        for component in changed:
            next_generations[component] = old_generations[component] + 1
    else:
        next_generations = dict(forced_component_generations)

    text = model.vlm
    text_state = text._rpu_qwen3_5
    vision = model.vision_tower
    vision_handle = getattr(vision, "_rpu_vision_handle", None)
    action = model.action_expert
    action_state = action._rpu_qwen3_5
    from rpu_backend.adapters.qwen3_5.text import _resolve_text_install_options
    from rpu_backend.adapters.qwen3_5.vision import _resolve_vision_install_options

    text_options = _resolve_text_install_options(
        max_seq_len,
        execution_config=components[_G05_TEXT_COMPONENT],
        cold_snapshot=text_state.control_snapshot,
    )
    vision_options = _resolve_vision_install_options(
        execution_config=components[_G05_VISION_COMPONENT],
        cold_snapshot=getattr(
            vision, "_rpu_vision_control_snapshot", None
        ),
    )
    text_requested = components[_G05_TEXT_COMPONENT]["prefill"]["chunk_size"]
    text_exact = 0 if text_requested == "auto" else int(text_requested)
    action_requested = components[_G05_ACTION_COMPONENT]["action"]["chunk_size"]
    action_exact = 0 if action_requested == "auto" else int(action_requested)

    policy._rpu_g05_execution_reconfigure_journal = {
        "mutation_started": False,
        "component_generations": old_generations,
    }
    if changed:
        with native_execution_reconfigure(torch.ops.rpu) as token:
            policy._rpu_g05_execution_reconfigure_journal[
                "mutation_started"
            ] = True
            if _G05_TEXT_COMPONENT in changed:
                max_kv_len, envelope_chunk = getattr(
                    text_state,
                    "prefill_capability_envelope",
                    text_state.chunk_envelope,
                )
                torch.ops.rpu.qwen3_5_stage_prefill_execution_controls(
                    text_state.handle,
                    token,
                    int(text_options[1]),
                    text_exact,
                    int(max_kv_len),
                    int(envelope_chunk),
                )
            if _G05_VISION_COMPONENT in changed and vision_handle is not None:
                torch.ops.rpu.qwen3_5_vision_stage_prefill_execution_controls(
                    vision_handle,
                    token,
                    int(vision_options[0]),
                    int(vision_options[1] or 0),
                )
            if _G05_ACTION_COMPONENT in changed:
                torch.ops.rpu.qwen3_5_stage_action_chunk_size(
                    action_state.handle, token, action_exact
                )

    if _G05_TEXT_COMPONENT in changed:
        text_state.padding_budget = int(text_options[2])
        text_state.padding_rows = components[_G05_TEXT_COMPONENT]["prefill"].get(
            "padding_rows", "auto"
        )
        text_state.control_snapshot = text_options[3]
        text_state.exact_chunk_size = text_exact or None
        text_state.prefill_plan_key = None
        text_state.prefill_plan = None
        text_state.last_a6_plan = None
        text_state.decode_stage_descriptor = None
        _clear_g05_graph(text_state.graph_cache)
        _clear_g05_graph(getattr(text_state, "prefill_graph_cache", None))
        _clear_g05_graph(getattr(text_state, "prefill_debug_graph_cache", None))
        text_state.prefill_graph.invalidate()
    if _G05_VISION_COMPONENT in changed and vision_handle is not None:
        (
            vision._rpu_vision_chunk_size_cap,
            vision._rpu_vision_exact_chunk_size,
            vision._rpu_vision_padding_rows,
            vision._rpu_vision_padding_budget,
            vision._rpu_vision_control_snapshot,
        ) = vision_options
        _clear_g05_graph(vision._rpu_vision_graph_cache)
        vision._rpu_vision_debug_graph.invalidate()
        vision._rpu_vision_graph_key = None
        vision._rpu_vision_graph_sig = None
        vision._rpu_vision_last_a6_plan = None
        vision._rpu_vision_a6_plans = {}
    if _G05_ACTION_COMPONENT in changed:
        _clear_g05_graph(action_state.graph_cache)
        _clear_g05_graph(action_state.action_graph_cache)
        _clear_g05_graph(action_state.action_trace_graph_cache)
        action_state.last_a6_plan = None
        vars(action).pop("_rpu_last_execution_plan", None)

    policy._rpu_g05_execution_reconfigure_journal["mutation_started"] = True
    _publish_g05_execution_views(
        policy,
        resolved,
        generation=generation,
        component_generations=next_generations,
    )
    # Retain undo state through the shared session's config publication.


def _rollback_g05_execution_config(
    policy, old_config, generation: int
) -> None:
    journal = getattr(policy, "_rpu_g05_execution_reconfigure_journal", None)
    if journal is not None and not journal.get("mutation_started", False):
        policy._rpu_g05_execution_reconfigure_journal = None
        return
    old_generations = (
        None if journal is None else journal.get("component_generations")
    )
    _apply_g05_execution_config(
        policy,
        old_config,
        generation,
        force_native=True,
        forced_component_generations=old_generations,
    )
    policy._rpu_g05_execution_reconfigure_journal = None


def _retire_g05_policy_runtime(policy) -> None:
    """Clear all child Graph owners before retiring their native handles."""
    model = policy.model
    vision = getattr(model, "vision_tower", None)
    text = getattr(model, "vlm", None)
    action = getattr(model, "action_expert", None)
    from rpu_backend.adapters.qwen3_5.text import (
        _retire_qwen3_5_text_state, _clear_qwen35_graphs, _validate_qwen35_retirement_children)
    from rpu_backend.api._execution import ExecutionSession

    session = policy._execution_session
    if session._owner is not policy:
        raise RuntimeError("G0.5 retirement requires its actual parent Session")
    _validate_qwen35_retirement_children(policy, tuple(
        (child, getattr(child, "_rpu_qwen3_5", None) or getattr(child, "_rpu_vision_retirement_state", None))
        for child in (text, action, vision)))
    for child in (text, action, vision):
        state = getattr(child, "_rpu_qwen3_5", None) or getattr(child, "_rpu_vision_retirement_state", None)
        if child is vision and any(getattr(vision, name, None) is not None for name in (
                "_rpu_vision_handle", "_rpu_vision_installing_handle")) and state is None:
            raise RuntimeError("G0.5 Vision retirement has no resource owner")
        if state is None or getattr(state, "handle", None) is None:
            continue
        resource = getattr(state, "retirement", None)
        if (resource is None or resource.failed is not None or resource.handle != state.handle
                or (resource.owner() is not None and resource.owner() is not state)
                or resource.parent is None
                or (resource.parent() is not None and resource.parent() is not policy)
                or getattr(state, "_execution_session", None) is not session
                or getattr(state, "_retirement_close", None) is not _rpu_g05_close_execution):
            raise RuntimeError("G0.5 retirement child belongs to a different parent")
        try:
            guarded = getattr(child, "_execution_session", None)
            if not isinstance(guarded, ExecutionSession) or vars(guarded) is not vars(session):
                raise RuntimeError("G0.5 retirement child belongs to a different Session")
        except ReferenceError:
            if resource.owner() is not None or resource.parent() is not None:
                raise
    _clear_qwen35_graphs(tuple(getattr(owner, "_rpu_qwen3_5", None)
                             for owner in (text, action)), vision)

    for owner in (action, text):
        if owner is not None:
            _retire_qwen3_5_text_state(owner, _graphs_retired=True)
    if vision is not None:
        from rpu_backend.adapters.qwen3_5.vision import _rollback_qwen3_5_vision_install

        _rollback_qwen3_5_vision_install(vision, _graphs_retired=True)
    from rpu_backend.api.causal_lm import _release_live_instance

    _release_live_instance(model)
    policy._rpu_g05_policy_ready = False


def _poison_g05_retirement(policy, error):
    from rpu_backend.api import causal_lm
    from rpu_backend.api._execution import _mark_execution_process_unsafe

    if not getattr(policy, "_rpu_g05_retirement_failed", None):
        _FAILED_POLICY_RETIREMENTS.append(policy)
        vars(policy)["_rpu_g05_retirement_failed"] = repr(error)
    session = getattr(policy, "_execution_session", None)
    if session is not None:
        session.poison()
    with causal_lm._LIVE_LOCK:
        live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
        if live is None:
            causal_lm._claim_live_instance(policy.model)
            live = policy.model
        causal_lm._poison_live_instance(live, f"G0.5 cleanup failed: {error!r}", unsafe=True)
    _mark_execution_process_unsafe(f"G0.5 cleanup failed: {error!r}")
    for child in (getattr(policy.model, "vlm", None), getattr(policy.model, "action_expert", None),
                  getattr(policy.model, "vision_tower", None)):
        state = getattr(child, "_rpu_qwen3_5", None) or getattr(child, "_rpu_vision_retirement_state", None)
        resource = getattr(state, "retirement", None)
        if resource is not None and resource.handle is not None:
            resource.retain_failure(error, policy)
    for _child, _state, resource, _session in vars(policy).get("_qwen35_retirement_children", ()):
        if resource.handle is not None:
            resource.retain_failure(error, policy)


def _rpu_g05_reconfigure_execution(self, value):
    return self._execution_session.reconfigure(value)


def _rpu_g05_close_execution(self) -> None:
    if getattr(self, "_rpu_g05_retirement_failed", None):
        raise RuntimeError("G0.5 retirement already failed; restart the process")
    started = False
    def retire():
        nonlocal started
        started = True
        from rpu_backend.api import causal_lm

        with causal_lm._LIVE_LOCK:
            live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
            if live is not None and live is not self.model:
                raise RuntimeError("G0.5 retirement lost the live-owner slot")
            _retire_g05_policy_runtime(self)
    try:
        self._execution_session.shutdown(retire)
        if any(item[2].handle is not None for item in vars(self).get("_qwen35_retirement_children", ())):
            started = True
            raise RuntimeError("G0.5 closed Session still owns native resources")
        for child in (getattr(self.model, "vlm", None), getattr(self.model, "action_expert", None),
                      getattr(self.model, "vision_tower", None)):
            if (getattr(getattr(child, "_rpu_qwen3_5", None), "handle", None) is not None
                    or getattr(child, "_rpu_vision_handle", None) is not None
                    or getattr(child, "_rpu_vision_installing_handle", None) is not None):
                started = True
                raise RuntimeError("G0.5 closed Session still owns native resources")
        vars(self).pop("_qwen35_retirement_children", None)
    except BaseException as error:
        if started:
            _poison_g05_retirement(self, error)
        raise


def _install_g05_execution_session(policy, resolved) -> None:
    from rpu_backend.api._execution import bind_execution_session

    generations = {component: 0 for component in _G05_EXECUTION_COMPONENTS}
    _publish_g05_execution_views(
        policy,
        resolved,
        generation=0,
        component_generations=generations,
    )
    session = bind_execution_session(
        policy,
        resolved[0],
        entry_point="G0.5 policy",
        supported_components=_G05_EXECUTION_COMPONENTS,
        validate=lambda config: _validate_g05_execution_reconfigure(
            policy, config
        ),
        apply=lambda _old, new, generation, *, force_rebuild=False: _apply_g05_execution_config(
            policy, new, generation, force_native=force_rebuild
        ),
        rollback=lambda old, _new, generation: _rollback_g05_execution_config(
            policy, old, generation
        ),
        graph_mode=_GRAPH_COMPOSITE_CHILD,
    )
    session.register_config_view(policy.model)
    vars(policy.model)["_execution_session"] = weakref.proxy(session)
    for component, child in (
        (_G05_VISION_COMPONENT, getattr(policy.model, "vision_tower", None)),
        (_G05_TEXT_COMPONENT, getattr(policy.model, "vlm", None)),
        (_G05_ACTION_COMPONENT, getattr(policy.model, "action_expert", None)),
    ):
        if child is not None:
            session._bind_planner_owner(child, component)
            vars(child)["_execution_session"] = weakref.proxy(session)
            state = getattr(child, "_rpu_qwen3_5", None) or getattr(child, "_rpu_vision_retirement_state", None)
            if state is not None:
                from rpu_backend.adapters.qwen3_5.text import _bind_qwen35_retirement

                _bind_qwen35_retirement(state, policy, _rpu_g05_close_execution, child)
    policy.reconfigure_rpu_execution = types.MethodType(
        _rpu_g05_reconfigure_execution, policy
    )
    policy.close_rpu_execution = types.MethodType(
        _rpu_g05_close_execution, policy
    )


@execution_serialized
def _rpu_g05_predict_action(self, batch):
    """Run the already-eval RPU policy without recursive mode walks."""
    if self.training or self.model.training:
        raise NotImplementedError("G0.5 RPU policy supports inference only")
    samples = batch["samples"]
    generated = self.forward_inference(
        samples=samples,
        pixel_values=batch["pixel_values"],
        actions=batch.get("action"),
        action_dim_is_pad=batch.get("action_dim_is_pad"),
        prev_action=batch.get("prev_action"),
        inference_interval=batch.get("inference_interval"),
        inference_delay=batch.get("inference_delay"),
    )
    batch.update(generated)
    return batch


def _g05_policy_runtime_ready(policy) -> bool:
    """Return whether every component published by the policy patch is live."""
    if getattr(policy, "_rpu_g05_policy_ready", False) is not True:
        return False
    if getattr(policy, "_rpu_g05_policy_install_started", False) is not True:
        return False
    if not hasattr(policy, "__dict__"):
        return False
    model = getattr(policy, "model", None)
    if model is None or not hasattr(model, "__dict__"):
        return False
    if not _g05_vlm_runtime_ready(model):
        return False
    expert = getattr(model, "action_expert", None)
    if expert is None or not _g05_action_runtime_ready(expert):
        return False
    if getattr(model, "_rpu_g05_prefix_bridge_installed", False) is not True:
        return False
    if not callable(vars(model).get("_rpu_g05_original_inference_fm")):
        return False
    if not _instance_method_is(model, "inference_fm", _rpu_g05_inference_fm):
        return False
    if not callable(vars(policy).get("_rpu_g05_original_predict_action")):
        return False
    if not _instance_method_is(policy, "predict_action", _rpu_g05_predict_action):
        return False
    if not callable(getattr(policy, "forward_inference", None)):
        return False
    session = vars(policy).get("_execution_session")
    if session is None:
        return False
    try:
        if session.graph_mode != _GRAPH_COMPOSITE_CHILD:
            return False
        if session.stats()["state"] != "QUIESCENT":
            return False
    except Exception:
        return False
    if not _instance_method_is(
        policy, "reconfigure_rpu_execution", _rpu_g05_reconfigure_execution
    ) or not _instance_method_is(
        policy, "close_rpu_execution", _rpu_g05_close_execution
    ):
        return False
    for component, child in (
        (_G05_VISION_COMPONENT, getattr(model, "vision_tower", None)),
        (_G05_TEXT_COMPONENT, getattr(model, "vlm", None)),
        (_G05_ACTION_COMPONENT, expert),
    ):
        if child is None:
            return False
        if getattr(child, "_rpu_execution_component_id", None) != component:
            return False
        if getattr(child, "_execution_session", None) is None:
            return False

    vision_required = (
        getattr(model, "_rpu_g05_vision_install_started", False) is True
        or getattr(model, "_rpu_g05_vision_ready", False) is True
    )
    if vision_required:
        if not _g05_vision_runtime_ready(model):
            return False
        if getattr(model.vision_tower, "_rpu_vision_has_merger", False) is not True:
            return False
        if not _instance_method_is(model, "_forward_embed", _rpu_g05_forward_embed):
            return False
        if getattr(model, "_rpu_g05_embed_w_rpu", None) is None:
            return False
        if getattr(model, "_rpu_g05_proprio_mlp", None) is None:
            return False

    processor = getattr(policy, "processor", None)
    processor_state = vars(processor) if hasattr(processor, "__dict__") else {}
    if "_rpu_g05_original_encode_inference" in processor_state:
        if not callable(processor_state["_rpu_g05_original_encode_inference"]):
            return False
        if not _instance_method_is(
            processor, "encode_inference", _rpu_g05_encode_inference
        ):
            return False
        if processor_state.get("_rpu_g05_action_rvq") is None:
            return False
        try:
            if int(processor_state.get("_rpu_g05_action_rng_rows", 0)) <= 0:
                return False
            if int(processor_state.get("_rpu_g05_action_rng_codebooks", 0)) <= 0:
                return False
        except (TypeError, ValueError):
            return False

    return True


@_transactional_g05_policy_install
def patch_g05_policy_for_rpu(
    policy, *, max_seq_len: int = 2048, rpu_execution=None
):
    """Patch a canonical G0.5 policy for continuous action inference.

    Load the official policy on CPU and call ``eval()`` before this function.
    The VLM text stack and action-expert decoder core move to RPU. Vision
    requires the controlled Qwen3.5 numeric-blocked opt-in. Tokenization and
    time conditioning stay on the host; the action I/O projections and FP16
    Euler run inside the RPU action graph.
    """
    if getattr(policy, "_rpu_g05_policy_ready", False):
        if _g05_policy_runtime_ready(policy):
            return policy
        raise RuntimeError(
            "G0.5 policy ready marker has incomplete runtime state; reload "
            "the checkpoint"
        )
    if getattr(policy, "_rpu_g05_policy_install_started", False):
        raise RuntimeError(
            "G0.5 policy installation was already attempted; reload the "
            "checkpoint to avoid reusing partial wrapper state"
        )
    resolved_execution = _resolve_g05_execution(
        rpu_execution,
        max_seq_len=max_seq_len,
        entry_point="patch_g05_policy_for_rpu",
    )
    if getattr(policy, "training", True):
        raise NotImplementedError(
            "G0.5 RPU policy supports inference only; call policy.eval()"
        )
    continuous = bool(getattr(policy, "continuous_action", False))
    discrete = bool(getattr(policy, "discrete_action", False))
    model = getattr(policy, "model", None)
    if model is None:
        raise TypeError("expected G05PolicyQwen35 with policy.model")
    if not continuous or discrete:
        raise NotImplementedError(
            "G0.5 RPU bring-up supports only continuous_action=true with "
            "discrete_action=false; discrete AR/CoT is not certified"
        )
    if bool(getattr(policy, "predict_cot", False)):
        raise NotImplementedError(
            "G0.5 RPU bring-up requires predict_cot=false; AR/CoT is not certified"
        )
    if max_seq_len <= 0 or (continuous and max_seq_len < 32):
        raise ValueError(
            "max_seq_len must be positive and cover the 32-token action horizon"
        )
    parameters = getattr(policy, "parameters", None)
    if callable(parameters) and any(
        parameter.device.type != "cpu" for parameter in parameters()
    ):
        raise NotImplementedError(
            "G0.5 RPU policy requires a CPU-hosted checkpoint before patching"
        )

    if continuous:
        expert = getattr(model, "action_expert", None)
        if expert is None:
            raise TypeError("continuous G0.5 policy is missing model.action_expert")
        fm_helper = getattr(model, "fm_helper", None)
        horizon = getattr(fm_helper, "horizon_steps", None)
        if horizon != 32:
            raise NotImplementedError(
                "G0.5 RPU action inference requires horizon_steps=32"
            )
        if (
            getattr(fm_helper, "action_dim", None) != 27
            or getattr(fm_helper, "num_inference_steps", None) != 10
            or getattr(fm_helper, "time_convention", None) != "pi_convention"
            or (
                not callable(getattr(fm_helper, "_sample_noise", None))
                and bool(getattr(fm_helper, "use_correlated_noise", False))
            )
        ):
            raise NotImplementedError(
                "G0.5 RPU action-step fusion requires action_dim=27, "
                "num_inference_steps=10, and time_convention=pi_convention"
            )
        _check_g05_action_profile(expert.config)
        mode = getattr(getattr(model, "cfg", None), "ae_vlm_condition_mode", None)
        if mode != "cross_attn_only":
            raise NotImplementedError(
                "G0.5 RPU action inference requires "
                "ae_vlm_condition_mode=cross_attn_only"
            )
        if getattr(getattr(model, "fm_helper", None), "action_causal", True):
            raise NotImplementedError(
                "G0.5 RPU action inference requires action_causal=false"
            )
        if getattr(expert, "training", True):
            raise NotImplementedError(
                "G0.5 RPU action expert supports inference only; call expert.eval()"
            )
    vlm = _preflight_g05_vlm(model)
    vision_enabled = os.environ.get("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED") == "1"
    proprio_mlp = None
    if vision_enabled:
        if os.environ.get("RPU_VISION_MERGER", "1") == "0":
            raise NotImplementedError(
                "G0.5 RPU direct vision fusion requires RPU_VISION_MERGER"
            )
        source_mlp = getattr(getattr(model, "proprio_embedder", None), "mlp", None)
        if not isinstance(source_mlp, torch.nn.Module):
            raise TypeError(
                "G0.5 RPU direct vision fusion requires model.proprio_embedder.mlp"
            )
        if not callable(getattr(model, "_forward_embed", None)):
            raise TypeError("G0.5 RPU direct vision fusion requires model._forward_embed")
        embed_weight = getattr(getattr(vlm, "input_proj", None), "weight", None)
        if not isinstance(embed_weight, torch.Tensor) or embed_weight.ndim != 2:
            raise TypeError("G0.5 RPU direct fusion requires vlm.input_proj.weight")
        _preflight_g05_vision(model)
        # Build the private copy before any irreversible vision/text mutation.
        proprio_mlp = copy.deepcopy(source_mlp).half().eval().requires_grad_(False)
    predict_action = getattr(policy, "predict_action", None)
    if not callable(predict_action) or not callable(
        getattr(policy, "forward_inference", None)
    ):
        raise TypeError("expected G05PolicyQwen35 inference methods")
    if not callable(getattr(model, "inference_fm", None)):
        raise TypeError("expected G05ModelQwen35.inference_fm")
    # The exact profile installs Vision before Text. Claim the process-wide
    # owner before either subsystem mutates weights or creates a handle.
    _claim_live_instance(model)
    policy._rpu_g05_policy_install_started = True
    os.environ.setdefault("RPU_FASTREPLAY_SKIP_SYNC", "1")
    if vision_enabled:
        patch_g05_vision_for_rpu(
            model,
            _execution_config=resolved_execution[1][_G05_VISION_COMPONENT],
        )
    vlm_install_kwargs = {
        "max_seq_len": max_seq_len,
        "_execution_config": resolved_execution[1][_G05_TEXT_COMPONENT],
    }
    patch_g05_vlm_for_rpu(model, **vlm_install_kwargs)
    if getattr(model, "_rpu_g05_vision_ready", False):
        model._rpu_g05_embed_w_rpu = (
            model.vlm.input_proj.weight.detach().to(torch.float16).contiguous().to("rpu")
        )
        object.__setattr__(
            model,
            "_rpu_g05_proprio_mlp",
            proprio_mlp,
        )
        model._forward_embed = types.MethodType(_rpu_g05_forward_embed, model)
    patch_g05_action_expert_for_rpu(
        model.action_expert,
        max_seq_len=max_seq_len,
        _execution_config=resolved_execution[1][_G05_ACTION_COMPONENT],
    )
    model._rpu_g05_original_inference_fm = model.inference_fm
    model.inference_fm = types.MethodType(_rpu_g05_inference_fm, model)
    processor = getattr(policy, "processor", None)
    action_tokenizer = getattr(processor, "action_tokenizer", None)
    codec = getattr(action_tokenizer, "action_tokenizer", None)
    codec_model = getattr(codec, "model", None)
    rvq = getattr(codec_model, "rvq", None)
    partitioner = getattr(codec, "_partitioner", None)
    get_metadata = getattr(codec, "get_decode_metadata", None)
    metadata = get_metadata() if callable(get_metadata) else None
    codec_parameters = (
        tuple(codec.parameters()) if isinstance(codec, torch.nn.Module) else ()
    )
    quantizers = tuple(getattr(rvq, "quantizers", ()))
    if (
        bool(getattr(processor, "batchify_action", False))
        and callable(getattr(processor, "encode_inference", None))
        and callable(getattr(processor, "_prepare_segments", None))
        and type(action_tokenizer).__name__ == "VQActionTokenizer"
        and type(codec).__name__ == "ActionCodecV2Wrapper"
        and type(codec_model).__name__ == "ActionCodecV2Model"
        and type(rvq).__name__ == "ResidualVectorQuantize"
        and getattr(codec, "key_dims", None)
        == {
            "left_control": 9,
            "left_gripper": 1,
            "right_control": 9,
            "right_gripper": 1,
            "lower_body": 7,
        }
        and partitioner is not None
        and getattr(partitioner, "merge_preprocessor", None) is None
        and int(getattr(metadata, "nn_chunked_key_count", 0)) == 3
        and int(getattr(rvq, "n_codebooks", 0)) == 4
        and bool(codec_parameters)
        and all(parameter.device.type == "cpu" for parameter in codec_parameters)
        and len(quantizers) == 4
        and all(
            type(quantizer).__name__ == "EMAVectorQuantize"
            and bool(getattr(quantizer, "inited", torch.tensor(False)).item())
            for quantizer in quantizers
        )
    ):
        processor._rpu_g05_original_encode_inference = processor.encode_inference
        processor._rpu_g05_action_rvq = rvq
        processor._rpu_g05_action_rng_rows = int(metadata.nn_chunked_key_count)
        processor._rpu_g05_action_rng_codebooks = int(rvq.n_codebooks)
        processor.encode_inference = types.MethodType(
            _rpu_g05_encode_inference,
            processor,
        )
    policy._rpu_g05_original_predict_action = predict_action
    policy.predict_action = types.MethodType(_rpu_g05_predict_action, policy)
    _install_g05_execution_session(policy, resolved_execution)
    policy._rpu_g05_policy_ready = True
    return policy


__all__ = [
    "patch_g05_action_expert_for_rpu",
    "patch_g05_data_processor_for_rpu",
    "patch_g05_inferencer_for_rpu",
    "patch_g05_policy_for_rpu",
    "patch_g05_vision_for_rpu",
    "patch_g05_vlm_for_rpu",
]
