"""G0.5 Qwen3.5 action expert FP16 bring-up."""
from __future__ import annotations

import math
import threading
import types

import torch
import torch.nn.functional as F

from rpu_backend.api._execution import execution_serialized

_ACTION_INSTALL_LOCK = threading.Lock()
_SHARED_PREFIX_LAYERS = (3, 7, 11, 15, 19, 23)
_ACTION_DIM_PAD = 32
_ACTION_HORIZON = 32
_G05_ACTION_COMPONENT = "action_expert"
_GRAPH_BOUNDED_ONESHOT = "BOUNDED_ONESHOT"
_GRAPH_COMPOSITE_CHILD = "COMPOSITE_CHILD"
_ACTION_ROUTE_IO = 1 << 0
_ACTION_ROUTE_RTC = 1 << 1
_ACTION_ROUTE_TRACE = 1 << 2
_ACTION_PROFILE = (
    1024, 4096, 24, 8, 2, 256,
    False, "silu", "adaLN", 1024, "linear", 27, "linear", 27, True,
    1.0e-6, 10_000_000.0, "default", (11, 11, 10), True, 0.25,
    ("full_attention",) * 24,
)


def _run_action_cold_prime(graph_api, retained_cache, plan, dispatch):
    """Prime persistent action state under one audited bounded one-shot."""
    from rpu_backend.adapters.qwen3_5.text import _oneshot_scope

    size_before = int(retained_cache.size())
    if not retained_cache.cache_invariant_ok():
        raise RuntimeError("G0.5 action cache invariant failed before cold-prime")
    graph = graph_api.Graph()
    with _oneshot_scope(graph):
        dispatch()
    graph_nodes = int(graph.graph_size())
    if graph.state() != 0:
        raise RuntimeError(
            "G0.5 action cold-prime one-shot did not return to PASSTHROUGH"
        )
    if graph_nodes <= 1:
        raise RuntimeError("G0.5 action cold-prime recorded no non-trivial Graph")
    size_after = int(retained_cache.size())
    if size_after != size_before or not retained_cache.cache_invariant_ok():
        raise RuntimeError(
            "G0.5 action cold-prime changed the retained GraphCache"
        )
    receipt = plan.as_dict(include_candidates=False)
    receipt.update({
        "stage": "action_cold_prime",
        "component": _G05_ACTION_COMPONENT,
        "graph_nodes": graph_nodes,
        "post_state": "PASSTHROUGH",
        "retained_cache_size_before": size_before,
        "retained_cache_size_after": size_after,
        "retained_cache_invariant": True,
    })
    return receipt


def _evict_action_graph_entries(cache):
    """Retire previous shape graphs without discarding still-legal plans.

    Prefix/resource and configuration invalidation continue to use clear().
    READY eviction is rejected by the original GraphCache boundary.
    """
    for entry in cache.snapshot():
        cache.evict(types.SimpleNamespace(_impl=entry.signature))


def _plan_g05_action_execution(
    expert, *, prefix_len: int, include_action_io: bool,
    action_route_flags: int = 0,
    graph_mode: str = _GRAPH_COMPOSITE_CHILD,
    num_steps: int = 1,
    graph_cache=None,
):
    """Resolve the fixed-horizon action descriptor before Graph admission."""
    from rpu_backend.runtime.execution_planner import GRAPH_MODES
    from rpu_backend.runtime.decoder import plan_bounded_prefill_execution

    if graph_mode not in (_GRAPH_BOUNDED_ONESHOT, _GRAPH_COMPOSITE_CHILD):
        raise ValueError(f"unsupported G0.5 action graph mode: {graph_mode}")

    state = expert._rpu_qwen3_5
    action_config = getattr(
        expert, "_rpu_execution", {"action": {"chunk_size": "auto"}}
    )["action"]
    requested = action_config.get("chunk_size", "auto")
    exact_chunk = int(requested) if isinstance(requested, int) else None
    generation = int(getattr(expert, "_rpu_execution_generation", 0))
    result_box = {}
    execution_len, chunk_size = plan_bounded_prefill_execution(
        _ACTION_HORIZON,
        _ACTION_HORIZON,
        0,
        position=int(prefix_len),
        alignment=1,
        padding_rows=0,
        exact_chunk_size=exact_chunk,
        resolve_stage_domain=lambda length: (
            torch.ops.rpu.qwen3_5_resolve_action_stage_domain(
                state.handle,
                int(length),
                int(prefix_len),
                0 if exact_chunk is None else exact_chunk,
                bool(include_action_io),
                int(action_route_flags),
                GRAPH_MODES.index(graph_mode) + 1,
                num_steps,
            )
        ),
        request_id="g05:action_expert:action",
        graph_mode=graph_mode,
        queue_owner_id=int(state.handle),
        execution_owner=expert,
        execution_stage="action",
        execution_native=("qwen3_5", int(state.handle)),
        physical_metadata=(
            ("component:action_expert", 1),
            ("execution_generation", generation),
            ("prefix_len", int(prefix_len)),
            ("prefix_history_ddr_required", 1),
            ("action_io", int(include_action_io)),
            ("action_route_flags", int(action_route_flags)),
        ),
        plan_result_sink=lambda result: result_box.__setitem__("result", result),
        plan_signature=(bool(include_action_io), int(action_route_flags), int(num_steps)),
        graph_cache=graph_cache,
    )
    result = result_box["result"]
    selected = result.selected
    if (
        selected is None
        or int(execution_len) != _ACTION_HORIZON
        or int(chunk_size) != _ACTION_HORIZON
        or not selected.stage_tuple.physical_descriptor
    ):
        raise RuntimeError(
            "G0.5 action planner returned no complete horizon-32 descriptor"
        )
    return result


def _publish_g05_action_receipt(
    expert, plan, *, prefix_len: int, cold_prime=None,
) -> None:
    """Publish dry/forward agreement for the action composite child."""
    selected = plan.selected
    if selected is None:
        raise RuntimeError("G0.5 action planner selected no plan")
    state = expert._rpu_qwen3_5
    resolved = int(torch.ops.rpu.qwen3_5_get_resolved_chunk_size(state.handle))
    if resolved != int(selected.stage_tuple.compute_chunk):
        raise RuntimeError(
            "G0.5 action dry/forward chunk plan drift: "
            f"dry={selected.stage_tuple.compute_chunk}, forward={resolved}"
        )
    receipt = plan.as_dict(include_candidates=False)
    receipt.update({
        "stage": "action",
        "component": _G05_ACTION_COMPONENT,
        "generation": int(getattr(expert, "_rpu_execution_generation", 0)),
        "logical_len": _ACTION_HORIZON,
        "execution_len": int(selected.execution_len),
        "chunk_size": resolved,
        "padding_rows": int(selected.padding_rows),
        "position": int(prefix_len),
        "authority": "NATIVE_A6_STAGE_DESCRIPTOR",
        "attention_policy": "DDR_REQUIRED",
        "route_authority": {
            "attention_call_site": 9137328812466618373,
            "kv_insert_call_site": 8503568375190929130,
            "local_kv_insert_invocation": 0,
            "prefix_kv_insert_invocation": 1,
        },
        "dry_forward_agreement": True,
    })
    if cold_prime is not None:
        receipt["cold_prime"] = cold_prime
    vars(expert)["_rpu_last_execution_plan"] = receipt


def _check_g05_action_profile(config) -> None:
    from rpu_backend.api.errors import UnsupportedModelError

    rope = getattr(config, "rope_parameters", None) or {}
    profile = (
        int(config.hidden_size),
        int(config.intermediate_size),
        int(config.num_hidden_layers),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(config.head_dim),
        bool(config.attention_bias),
        str(config.hidden_act),
        str(config.adaptive_mode),
        int(config.time_hidden_size),
        str(config.input_type),
        int(config.input_dim),
        str(config.output_type),
        int(config.output_dim),
        bool(config.use_final_norm),
        float(config.rms_norm_eps),
        float(rope.get("rope_theta", 0.0)),
        str(rope.get("rope_type", "")),
        tuple(int(x) for x in rope.get("mrope_section", ())),
        bool(rope.get("mrope_interleaved", False)),
        float(rope.get("partial_rotary_factor", 0.0)),
        tuple(config.layer_types),
    )
    if profile != _ACTION_PROFILE:
        raise UnsupportedModelError(
            "G0.5 RPU action bring-up supports only the canonical "
            f"Qwen3.5 action-expert profile; got {profile}."
        )


def _g05_action_runtime_ready(expert) -> bool:
    """Return whether the published action runtime is complete and live."""
    if not hasattr(expert, "__dict__"):
        return False
    if getattr(expert, "_rpu_g05_action_ready", False) is not True:
        return False
    if getattr(expert, "_rpu_g05_action_install_started", False) is not True:
        return False
    if getattr(expert, "_rpu_qwen3_5_text_install_started", False) is not True:
        return False
    state = getattr(expert, "_rpu_qwen3_5", None)
    if state is None or getattr(state, "handle", None) is None:
        return False
    from rpu_backend.adapters.qwen3_5.text import _qwen35_retirement_ready

    if not _qwen35_retirement_ready(state):
        return False
    for name in (
        "graph_cache",
        "action_graph_cache",
        "action_trace_graph_cache",
        "action_cache",
        "action_k_caches",
        "action_v_caches",
    ):
        if getattr(state, name, None) is None:
            return False
    for name in (
        "action_prefix_binding",
        "action_prefix_key",
        "action_prefix_source",
        "action_prefix_lens",
        "action_rope_cache",
        "action_layout_cache",
        "action_keep_mask_cache",
        "action_dim",
        "action_dim_pad",
    ):
        if not hasattr(state, name):
            return False
    if getattr(state, "action_dim", None) != 27:
        return False
    if getattr(state, "action_dim_pad", None) != _ACTION_DIM_PAD:
        return False
    if getattr(expert, "_rpu_execution_component_id", None) != _G05_ACTION_COMPONENT:
        return False
    if getattr(expert, "_rpu_execution_generation", None) is None:
        return False
    for name in (
        "_rpu_g05_adaptive_mod_cache",
        "_rpu_g05_adaptive_mod_stack_cache",
        "_rpu_g05_time_cond_cache",
        "_rpu_g05_fixed_time_schedule",
        "_rpu_g05_rtc_time_schedules",
    ):
        if not hasattr(expert, name):
            return False
    for name in (
        "_rpu_g05_adaptive_mod_cache",
        "_rpu_g05_time_cond_cache",
        "_rpu_g05_rtc_time_schedules",
    ):
        if not isinstance(getattr(expert, name), dict):
            return False
    if not callable(vars(expert).get("_rpu_g05_original_encode_time")):
        return False
    if not callable(vars(expert).get("_rpu_g05_original_forward")):
        return False
    encode_time = vars(expert).get("encode_time")
    forward = vars(expert).get("forward")
    return bool(
        getattr(encode_time, "__self__", None) is expert
        and getattr(encode_time, "__func__", None) is _rpu_g05_encode_time
        and getattr(forward, "__self__", None) is expert
        and getattr(forward, "__func__", None) is _rpu_g05_action_forward
    )


def _logical_prefix(cache, layer_idx: int):
    if hasattr(cache, "has_item") and cache.has_item(layer_idx):
        return cache.get(layer_idx)
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and layer_idx in key_cache:
        return key_cache[layer_idx], value_cache[layer_idx]
    return None


def _copy_logical_prefix(dst, layer_idx: int, key: torch.Tensor,
                         value: torch.Tensor, prefix_len: int) -> None:
    if tuple(key.shape) != tuple(value.shape):
        raise ValueError("G0.5 action prefix K/V shapes must match")
    if key.ndim != 4 or key.shape[0] != 1 or key.shape[2] != prefix_len:
        raise ValueError(
            "G0.5 action prefix K/V must be [1, num_kv_heads, prefix_len, head_dim]"
        )
    effective_heads = dst.nKVHeadChunk * dst.nKVHeadVx
    if effective_heads % key.shape[1] != 0:
        raise ValueError(
            f"cannot expand {key.shape[1]} prefix KV heads to {effective_heads}"
        )
    rep = effective_heads // key.shape[1]
    key = key.repeat_interleave(rep, dim=1)
    value = value.repeat_interleave(rep, dim=1)
    padded = math.ceil(prefix_len / dst.sKeyChunk) * dst.sKeyChunk
    if padded != prefix_len:
        key = F.pad(key, (0, 0, 0, padded - prefix_len))
        value = F.pad(value, (0, 0, 0, padded - prefix_len))
    key = key.to(device="rpu", dtype=torch.float16).contiguous()
    value = value.to(device="rpu", dtype=torch.float16).contiguous()

    bsz = dst.batch_size
    skv = padded // dst.sKeyChunk
    svv = padded // dst.sValChunk
    key = key.reshape(
        bsz, dst.nKVHeadChunk, dst.nKVHeadVx, skv, dst.sKeyChunk,
        dst.headDimVx, dst.headDimChunk,
    ).permute(0, 3, 2, 5, 1, 4, 6).contiguous()
    value = value.reshape(
        bsz, dst.nKVHeadChunk, dst.nKVHeadVx, svv, dst.sValChunk,
        dst.headDimVx, dst.headDimChunk,
    ).permute(0, 3, 2, 5, 1, 6, 4).contiguous()
    dst.k_caches[layer_idx][:, :skv].copy_(key)
    dst.v_caches[layer_idx][:, :svv].copy_(value)


def _prepare_prefix(expert, prefix_cache, prefix_len: int) -> tuple[list[int], bool]:
    state = expert._rpu_qwen3_5
    dst = state.action_cache
    generation = getattr(prefix_cache, "_rpu_qwen3_5_generation", None)
    key = (
        int(generation) if generation is not None else id(prefix_cache),
        int(prefix_len),
    )
    same_source = state.action_prefix_source is prefix_cache
    same_key = state.action_prefix_key == key and (
        generation is not None or same_source
    )

    physical = getattr(prefix_cache, "_rpu_qwen3_5_cache", None)
    if physical is not None:
        k_caches = list(dst.k_caches)
        v_caches = list(dst.v_caches)
        binding = ["physical"]
        for layer_idx in _SHARED_PREFIX_LAYERS:
            src_k = physical.k_caches[layer_idx]
            src_v = physical.v_caches[layer_idx]
            if src_k.ndim != 7 or src_v.ndim != 7:
                raise ValueError(
                    f"G0.5 VLM prefix cache layer {layer_idx} is not physical KV"
                )
            k_caches[layer_idx] = src_k
            v_caches[layer_idx] = src_v
            binding.extend((int(src_k.data_ptr()), int(src_v.data_ptr())))
        binding = tuple(binding)
    else:
        k_caches = dst.k_caches
        v_caches = dst.v_caches
        binding_parts = ["copy"]
        for layer_idx in _SHARED_PREFIX_LAYERS:
            binding_parts.extend(
                (
                    int(k_caches[layer_idx].data_ptr()),
                    int(v_caches[layer_idx].data_ptr()),
                )
            )
        binding = tuple(binding_parts)

    if same_key and state.action_prefix_binding == binding:
        return state.action_prefix_lens, False
    if (
        state.action_prefix_binding is not None
        and (state.action_prefix_binding != binding
             or (state.action_prefix_lens is not None
                 and state.action_prefix_lens[_SHARED_PREFIX_LAYERS[0]] != prefix_len))
    ):
        trace_cache = getattr(state, "action_trace_graph_cache", None)
        if state.action_graph_cache.is_frozen() or (trace_cache is not None and trace_cache.is_frozen()):
            raise RuntimeError("G0.5 READY prefix layout changed; call begin_warmup() before rebuilding")
        state.action_graph_cache.clear()
        if trace_cache is not None:
            trace_cache.clear()

    if physical is None:
        for layer_idx in _SHARED_PREFIX_LAYERS:
            pair = _logical_prefix(prefix_cache, layer_idx)
            if pair is None:
                raise ValueError(
                    f"G0.5 action prefix is missing full-attention layer {layer_idx}"
                )
            _copy_logical_prefix(dst, layer_idx, pair[0], pair[1], prefix_len)

    state.action_k_caches = k_caches
    state.action_v_caches = v_caches
    state.action_prefix_binding = binding

    prefix_lens = [
        prefix_len if layer_idx in _SHARED_PREFIX_LAYERS else 0
        for layer_idx in range(state.num_layers)
    ]
    state.action_prefix_key = key
    # Retain the source when it has no explicit generation. This prevents an
    # object-id ABA from treating a new same-length SparseKVCache as the old one.
    state.action_prefix_source = prefix_cache
    state.action_prefix_lens = prefix_lens
    return prefix_lens, True


def _adaptive_modulation_cpu(expert, time_cond: torch.Tensor) -> torch.Tensor:
    if (
        time_cond is None
        or time_cond.ndim != 2
        or time_cond.shape[0] not in (1, 2)
        or time_cond.shape[1] != expert.config.hidden_size
    ):
        raise ValueError(
            "G0.5 action time_cond must have shape "
            f"[1|2, {expert.config.hidden_size}]"
        )
    cond = time_cond.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    rows = []
    with torch.inference_mode(), torch.autocast("cpu", enabled=False):
        for layer in expert.layers:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                modulation = F.linear(
                    cond, norm.dense.weight.float(), norm.dense.bias.float()
                )
                scale, shift, gate = modulation.chunk(3, dim=-1)
                rows.append(torch.cat((scale + 1.0, shift, gate), dim=-1))
        modulation = F.linear(
            cond, expert.norm.dense.weight.float(), expert.norm.dense.bias.float()
        )
        scale, shift, gate = modulation.chunk(3, dim=-1)
        rows.append(torch.cat((scale + 1.0, shift, gate), dim=-1))
    stacked = torch.stack(rows)
    return stacked[:, 0] if cond.shape[0] == 1 else stacked


def _adaptive_modulation(expert, time_cond: torch.Tensor) -> torch.Tensor:
    if time_cond is None or tuple(time_cond.shape) != (1, expert.config.hidden_size):
        raise ValueError(
            "G0.5 action time_cond must have shape "
            f"[1, {expert.config.hidden_size}]"
        )
    cond = time_cond.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    key = cond.numpy().tobytes()
    cached = expert._rpu_g05_adaptive_mod_cache.get(key)
    if cached is not None:
        return cached
    modulation = _adaptive_modulation_cpu(expert, cond).to(
        device="rpu", dtype=torch.float16
    ).contiguous()
    # ponytail: canonical G0.5 has 10 timesteps; use an LRU if that expands.
    if len(expert._rpu_g05_adaptive_mod_cache) < 10:
        expert._rpu_g05_adaptive_mod_cache[key] = modulation
    return modulation


def _adaptive_modulation_stack(expert, time_conditions) -> torch.Tensor:
    conds = [
        value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for value in time_conditions
    ]
    key = b"".join(value.numpy().tobytes() for value in conds)
    cached = expert._rpu_g05_adaptive_mod_stack_cache
    if cached is not None and cached[0] == key:
        return cached[1]
    modulation = torch.stack([
        _adaptive_modulation_cpu(expert, value) for value in conds
    ]).to(device="rpu", dtype=torch.float16).contiguous()
    expert._rpu_g05_adaptive_mod_stack_cache = (key, modulation)
    return modulation


def _rpu_g05_encode_time(self, timestep: torch.Tensor) -> torch.Tensor:
    if tuple(timestep.shape) != (1,):
        raise ValueError("G0.5 RPU time conditioning requires batch_size=1")
    timestep = timestep.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    key = timestep.numpy().tobytes()
    cached = self._rpu_g05_time_cond_cache.get(key)
    if cached is not None:
        return cached
    with torch.inference_mode():
        time_cond = self._rpu_g05_original_encode_time(timestep).contiguous()
    # ponytail: canonical G0.5 has 10 timesteps; use an LRU if that expands.
    if len(self._rpu_g05_time_cond_cache) < 10:
        self._rpu_g05_time_cond_cache[key] = time_cond
    return time_cond


def _fixed_action_time_schedule(expert, num_steps: int, dtype: torch.dtype):
    """Build the canonical deterministic time/modulation stack once."""
    if num_steps != 10 or dtype != torch.float32:
        raise ValueError(
            "G0.5 fixed action schedule requires 10 FP32 timesteps"
        )
    cached = expert._rpu_g05_fixed_time_schedule
    if cached is not None:
        return cached

    delta_t = 1.0 / num_steps
    t = torch.ones(1, device="cpu", dtype=dtype)
    time_conditions = []
    for _ in range(num_steps):
        with torch.autocast("cpu", enabled=False):
            time_conditions.append(expert.encode_time(t).detach().clone())
        t -= delta_t
    cached = (
        tuple(time_conditions),
        _adaptive_modulation_stack(expert, time_conditions),
    )
    expert._rpu_g05_fixed_time_schedule = cached
    return cached


def _fixed_action_rtc_time_schedule(
    expert,
    num_steps: int,
    dtype: torch.dtype,
    inference_delay: int,
):
    """Build the exact clean/noise AdaLN table for one RTC prefix length."""
    if (
        num_steps != 10
        or dtype != torch.float32
        or not 0 < inference_delay < 32
    ):
        raise ValueError(
            "G0.5 RTC schedule requires 10 FP32 timesteps and "
            "0 < inference_delay < 32"
        )
    cached = expert._rpu_g05_rtc_time_schedules.get(inference_delay)
    if cached is not None:
        return cached

    delta_t = 1.0 / num_steps
    t = torch.ones(1, 32, device="cpu", dtype=dtype)
    time_conditions = []
    for _ in range(num_steps):
        t[:, :inference_delay] = 0.0
        with torch.autocast("cpu", enabled=False):
            full = expert._rpu_g05_original_encode_time(t).detach()
        if tuple(full.shape[:2]) != (1, 32):
            raise ValueError(
                "G0.5 RTC time encoder must preserve the [1, 32] token axes"
            )
        time_conditions.append(
            torch.stack((full[0, 0], full[0, inference_delay])).contiguous()
        )
        t -= delta_t
    cached = (
        tuple(time_conditions),
        _adaptive_modulation_stack(expert, time_conditions),
    )
    expert._rpu_g05_rtc_time_schedules[inference_delay] = cached
    return cached


def _unique_scalar(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().unique()
        if values.numel() != 1:
            raise ValueError("G0.5 RTC scalar inputs must be uniform within the batch")
        return int(values.item())
    return int(value)


def _check_action_mask(mask, *, action_len: int, prefix_len: int) -> None:
    if mask is None:
        return
    mask = mask.detach().to("cpu")
    expected = (1, 1, action_len, prefix_len + action_len)
    if tuple(mask.shape) != expected:
        raise NotImplementedError(
            f"G0.5 RPU action mask must have shape {expected}, got {tuple(mask.shape)}"
        )
    if not bool(mask.eq(0).all().item()):
        raise NotImplementedError(
            "G0.5 RPU action bring-up supports only unpadded, non-causal action attention"
        )


def _cached_action_position(
    model,
    state,
    attention_mask: torch.Tensor,
    position_ids,
    *,
    action_len: int,
    prefix_len: int,
    dtype: torch.dtype,
    action_causal: bool,
):
    """Build and validate the exact action layout once per input value."""
    offset = getattr(model.mask_helper, "action_position_offset", None)
    key = None
    if offset is None:
        values = (attention_mask, position_ids)
        key = (
            action_len,
            prefix_len,
            int(attention_mask.size(1)),
            dtype,
            action_causal,
            tuple(
                None
                if value is None
                else (
                    str(value.device),
                    value.dtype,
                    tuple(value.shape),
                    value.detach().cpu().contiguous().numpy().tobytes(),
                )
                for value in values
            ),
        )
        cached = state.action_layout_cache
        if cached is not None and cached[0] == key:
            return cached[1]

    action_mask, action_pos = model.build_action_mask_and_position_ids(
        attention_mask,
        action_len=action_len,
        position_ids_prefix=position_ids,
        split_index=attention_mask.size(1),
        dtype=dtype,
        action_causal=action_causal,
    )
    _check_action_mask(
        action_mask, action_len=action_len, prefix_len=prefix_len
    )
    if key is not None:
        state.action_layout_cache = (key, action_pos)
    return action_pos


def _cached_action_keep_mask(state, pad_mask):
    """Keep the cold feature mask and its RPU allocation stable across REPLAY."""
    feature_mask = None
    if pad_mask is not None:
        if (
            pad_mask.device.type != "cpu"
            or pad_mask.dtype != torch.bool
            or tuple(pad_mask.shape) != (1, 1, state.action_dim)
        ):
            raise ValueError(
                "G0.5 action feature pad mask must be CPU bool [1, 1, action_dim]"
            )
        feature_mask = pad_mask[0, 0].bool().contiguous()
    key = None if feature_mask is None else feature_mask.numpy().tobytes()
    cached = state.action_keep_mask_cache
    if cached is not None and cached[0] == key:
        return cached[1]

    keep_mask = torch.zeros(state.action_dim_pad, dtype=torch.float16)
    keep_mask[: state.action_dim] = 1
    if feature_mask is not None:
        keep_mask[: state.action_dim].masked_fill_(feature_mask, 0)
    keep_mask = keep_mask.to("rpu").contiguous()
    state.action_keep_mask_cache = (key, keep_mask)
    return keep_mask


def _ensure_action_rope(state, position_ids, action_len: int) -> None:
    """Refresh action M-RoPE only when the exact canonical positions change."""
    cached = state.action_rope_cache
    if (
        cached is not None
        and cached[0] is position_ids
        and torch.equal(cached[1], position_ids)
    ):
        return

    from rpu_backend.adapters.qwen3_5.text import (
        _build_interleaved_mrope_cos_sin,
        _normalize_prefill_position_ids,
    )

    pid = _normalize_prefill_position_ids(
        position_ids, real_len=action_len, padded_len=action_len
    )
    if cached is not None and torch.equal(cached[2], pid):
        cos, sin = cached[3], cached[4]
    else:
        cos, sin = _build_interleaved_mrope_cos_sin(
            pid, state.rotary_dim, state.rope_theta, state.mrope_section
        )
        torch.ops.rpu.qwen3_5_set_prefill_rope(state.handle, cos, sin)
    state.action_rope_cache = (
        position_ids,
        position_ids.detach().clone(),
        pid,
        cos,
        sin,
    )


@execution_serialized
def _rpu_g05_action_forward(
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
    del attn_implementation, mixture_name
    if self.training:
        raise NotImplementedError("G0.5 RPU action expert supports inference only")
    if inputs_embeds.device.type != "cpu" or inputs_embeds.shape[0] != 1:
        raise NotImplementedError(
            "G0.5 RPU action expert requires batch_size=1 with host inputs"
        )
    if inputs_embeds.shape[1] != 32:
        raise NotImplementedError("G0.5 RPU action expert requires horizon_steps=32")
    if past_key_values is not None or kv_cache is None:
        raise TypeError("G0.5 RPU action expert requires a sparse prefix kv_cache")
    if return_kv_cache or split_idx is not None or padding_mask is not None:
        raise NotImplementedError(
            "G0.5 RPU action expert supports read-only prefix inference only"
        )
    prefix_len = int(getattr(kv_cache, "_rpu_g05_prefix_len", -1))
    prefix_cache = getattr(kv_cache, "_rpu_g05_prefix_cache", kv_cache)
    if prefix_len < 0:
        logical = getattr(prefix_cache, "num_items", None)
        prefix_len = int(logical() if callable(logical) else -1)
    if prefix_len < 0:
        raise TypeError("G0.5 RPU action prefix cache is missing its prefix length")
    _check_action_mask(
        attention_mask,
        action_len=int(inputs_embeds.shape[1]),
        prefix_len=prefix_len,
    )

    import rpu_backend as _rb

    state = self._rpu_qwen3_5
    plan = _plan_g05_action_execution(
        self, prefix_len=prefix_len, include_action_io=False,
        action_route_flags=0,
        graph_cache=state.action_graph_cache,
    )
    selected = plan.selected
    assert selected is not None
    descriptor = selected.stage_tuple.physical_descriptor
    sig = _rb.graph.GraphSignature(
        op_id="rpu_g05_action_expert",
        shapes=[int(inputs_embeds.shape[1]), state.hidden_size, prefix_len],
        dyn_dims=[
            state.num_layers,
            int(getattr(self, "_rpu_execution_generation", 0)),
            *plan.graph_key_words(),
        ],
        dtypes=[torch.float16],
    )
    # READY admission is lookup-only and must precede KV/SPM preparation,
    # one-entry eviction and the bounded cold-prime, not just capture itself.
    if state.action_graph_cache.is_frozen():
        state.action_graph_cache.get_or_create(sig)
    prefix_lens, _ = _prepare_prefix(self, prefix_cache, prefix_len)
    _ensure_action_rope(state, position_ids, int(inputs_embeds.shape[1]))
    hidden_input = inputs_embeds.to(device="rpu", dtype=torch.float16).contiguous()
    modulation = _adaptive_modulation(self, time_cond)
    cold = state.action_graph_cache.lookup(sig) is None
    cold_prime_receipt = None
    if cold:
        if state.action_graph_cache.size() != 0:
            _evict_action_graph_entries(state.action_graph_cache)
        # Prime the newly allocated persistent SPM and action-cache state in
        # PASSTHROUGH. The following captured BUILD is then a valid measured
        # result; later Euler steps and observations are one-launch REPLAYs.
        cold_plan = _plan_g05_action_execution(
            self, prefix_len=prefix_len, include_action_io=False,
            action_route_flags=0, graph_mode=_GRAPH_BOUNDED_ONESHOT,
            graph_cache=None,
        )
        assert cold_plan.selected is not None
        cold_descriptor = cold_plan.selected.stage_tuple.physical_descriptor
        cold_prime_receipt = _run_action_cold_prime(
            _rb.graph,
            state.action_graph_cache,
            cold_plan,
            lambda: torch.ops.rpu.qwen3_5_action_forward(
                state.handle,
                hidden_input,
                modulation,
                state.action_k_caches,
                state.action_v_caches,
                prefix_lens,
                cold_descriptor,
            ),
        )
    with state.action_graph_cache.capture(sig):
        hidden = torch.ops.rpu.qwen3_5_action_forward(
            state.handle,
            hidden_input,
            modulation,
            state.action_k_caches,
            state.action_v_caches,
            prefix_lens,
            descriptor,
        )
    _publish_g05_action_receipt(
        self, plan, prefix_len=prefix_len, cold_prime=cold_prime_receipt,
    )
    return hidden.to("cpu").clone()


def _rpu_g05_action_step(
    expert,
    action_input: torch.Tensor,
    action_keep_mask: torch.Tensor,
    action_output: torch.Tensor,
    delta_t: float,
    time_cond: torch.Tensor,
    prefix_lens: list[int],
    prefix_len: int,
) -> torch.Tensor:
    """Run one RPU input-projection → decoder → output-projection → Euler step."""
    import rpu_backend as _rb

    state = expert._rpu_qwen3_5
    expected = (1, 32, state.action_dim_pad)
    if (
        tuple(action_input.shape) != expected
        or tuple(action_output.shape) != expected
        or action_input.device.type != "rpu"
        or action_output.device.type != "rpu"
        or action_input.data_ptr() == action_output.data_ptr()
    ):
        raise ValueError(
            "G0.5 RPU action step expects distinct FP16 RPU input/output "
            f"buffers with shape {expected}"
        )
    if (
        tuple(action_keep_mask.shape) != (state.action_dim_pad,)
        or action_keep_mask.device.type != "rpu"
    ):
        raise ValueError(
            "G0.5 RPU action keep mask must be a padded 1D RPU tensor"
        )
    plan = _plan_g05_action_execution(
        expert, prefix_len=prefix_len, include_action_io=True,
        action_route_flags=_ACTION_ROUTE_IO,
        graph_cache=state.action_graph_cache,
    )
    selected = plan.selected
    assert selected is not None
    descriptor = selected.stage_tuple.physical_descriptor
    sig = _rb.graph.GraphSignature(
        op_id="rpu_g05_action_euler_step",
        shapes=[32, state.hidden_size, prefix_len, state.action_dim_pad],
        dyn_dims=[
            state.num_layers,
            int(getattr(expert, "_rpu_execution_generation", 0)),
            *plan.graph_key_words(),
        ],
        dtypes=[torch.float16],
    )
    if state.action_graph_cache.is_frozen():
        state.action_graph_cache.get_or_create(sig)
    modulation = _adaptive_modulation(expert, time_cond)
    cold_prime_receipt = None
    if state.action_graph_cache.lookup(sig) is None:
        if state.action_graph_cache.size() != 0:
            _evict_action_graph_entries(state.action_graph_cache)
        cold_plan = _plan_g05_action_execution(
            expert, prefix_len=prefix_len, include_action_io=True,
            action_route_flags=_ACTION_ROUTE_IO,
            graph_mode=_GRAPH_BOUNDED_ONESHOT,
            graph_cache=None,
        )
        assert cold_plan.selected is not None
        cold_descriptor = cold_plan.selected.stage_tuple.physical_descriptor
        cold_prime_receipt = _run_action_cold_prime(
            _rb.graph,
            state.action_graph_cache,
            cold_plan,
            lambda: torch.ops.rpu.qwen3_5_action_step_forward(
                state.handle,
                action_input,
                action_keep_mask,
                action_output,
                delta_t,
                modulation,
                state.action_k_caches,
                state.action_v_caches,
                prefix_lens,
                1,
                cold_descriptor,
            ),
        )
    with state.action_graph_cache.capture(sig):
        velocity = torch.ops.rpu.qwen3_5_action_step_forward(
            state.handle,
            action_input,
            action_keep_mask,
            action_output,
            delta_t,
            modulation,
            state.action_k_caches,
            state.action_v_caches,
            prefix_lens,
            1,
            descriptor,
        )
    _publish_g05_action_receipt(
        expert, plan, prefix_len=prefix_len,
        cold_prime=cold_prime_receipt,
    )
    return velocity


def _rpu_g05_action_loop(
    expert,
    action_input: torch.Tensor,
    action_keep_mask: torch.Tensor,
    action_output: torch.Tensor,
    delta_t: float,
    time_conditions,
    prefix_lens: list[int],
    prefix_len: int,
    precomputed_modulation: torch.Tensor | None = None,
    rtc_prefix: torch.Tensor | None = None,
    rtc_prefix_len: int = 0,
) -> torch.Tensor:
    """Run the complete Euler trajectory in one captured RPU graph."""
    import rpu_backend as _rb

    state = expert._rpu_qwen3_5
    num_steps = len(time_conditions)
    expected = (1, 32, state.action_dim_pad)
    if (
        num_steps != 10
        or tuple(action_input.shape) != expected
        or tuple(action_output.shape) != expected
        or action_input.device.type != "rpu"
        or action_output.device.type != "rpu"
        or action_input.data_ptr() == action_output.data_ptr()
    ):
        raise ValueError(
            "G0.5 RPU action loop expects 10 steps and distinct FP16 "
            f"RPU input/output buffers with shape {expected}"
        )
    if (
        tuple(action_keep_mask.shape) != (state.action_dim_pad,)
        or action_keep_mask.device.type != "rpu"
    ):
        raise ValueError(
            "G0.5 RPU action keep mask must be a padded 1D RPU tensor"
        )
    rtc_active = rtc_prefix is not None
    if rtc_active:
        if (
            rtc_prefix_len <= 0
            or tuple(rtc_prefix.shape)
            != (1, rtc_prefix_len, state.action_dim_pad)
            or rtc_prefix.device.type != "rpu"
            or rtc_prefix.dtype != torch.float16
            or not rtc_prefix.is_contiguous()
        ):
            raise ValueError(
                "G0.5 RTC prefix must be contiguous FP16 RPU "
                "[1, inference_delay, action_dim_pad]"
            )
    elif rtc_prefix_len != 0:
        raise ValueError("G0.5 RTC prefix length requires an RTC prefix tensor")
    plan = _plan_g05_action_execution(
        expert, prefix_len=prefix_len, include_action_io=True,
        action_route_flags=(
            _ACTION_ROUTE_IO | (_ACTION_ROUTE_RTC if rtc_active else 0)
        ),
        num_steps=num_steps,
        graph_cache=state.action_graph_cache,
    )
    selected = plan.selected
    assert selected is not None
    descriptor = selected.stage_tuple.physical_descriptor
    sig = _rb.graph.GraphSignature(
        op_id=(
            "rpu_g05_action_rtc_euler_loop"
            if rtc_active
            else "rpu_g05_action_euler_loop"
        ),
        shapes=[
            32, state.hidden_size, prefix_len, state.action_dim_pad,
            rtc_prefix_len,
        ],
        dyn_dims=[
            state.num_layers,
            num_steps,
            int(getattr(expert, "_rpu_execution_generation", 0)),
            *plan.graph_key_words(),
        ],
        dtypes=[torch.float16],
    )
    if state.action_graph_cache.is_frozen():
        state.action_graph_cache.get_or_create(sig)
    modulation = (
        precomputed_modulation
        if precomputed_modulation is not None
        else _adaptive_modulation_stack(expert, time_conditions)
    )
    if state.action_graph_cache.lookup(sig) is None:
        if state.action_graph_cache.size() != 0:
            _evict_action_graph_entries(state.action_graph_cache)
    with state.action_graph_cache.capture(sig):
        if rtc_active:
            velocity = torch.ops.rpu.qwen3_5_action_rtc_step_forward(
                state.handle,
                action_input,
                action_keep_mask,
                action_output,
                rtc_prefix,
                rtc_prefix_len,
                delta_t,
                modulation,
                state.action_k_caches,
                state.action_v_caches,
                prefix_lens,
                num_steps,
                descriptor,
            )
        else:
            velocity = torch.ops.rpu.qwen3_5_action_step_forward(
                state.handle,
                action_input,
                action_keep_mask,
                action_output,
                delta_t,
                modulation,
                state.action_k_caches,
                state.action_v_caches,
                prefix_lens,
                num_steps,
                descriptor,
            )
    _publish_g05_action_receipt(expert, plan, prefix_len=prefix_len)
    return velocity


def _rpu_g05_action_rtc_trace(
    expert,
    action_input: torch.Tensor,
    action_keep_mask: torch.Tensor,
    action_output: torch.Tensor,
    rtc_prefix: torch.Tensor,
    rtc_prefix_len: int,
    delta_t: float,
    time_conditions,
    prefix_lens: list[int],
    prefix_len: int,
    precomputed_modulation: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Capture the five canonical taps from the actual 10-body RTC graph."""
    import rpu_backend as _rb

    state = expert._rpu_qwen3_5
    num_steps = len(time_conditions)
    expected = (1, 32, state.action_dim_pad)
    tensors = (action_input, action_output, action_keep_mask, rtc_prefix)
    if (
        num_steps != 10
        or tuple(action_input.shape) != expected
        or tuple(action_output.shape) != expected
        or any(tensor.device.type != "rpu" for tensor in tensors)
        or any(tensor.dtype != torch.float16 for tensor in tensors)
        or any(not tensor.is_contiguous() for tensor in tensors)
        or action_input.data_ptr() == action_output.data_ptr()
    ):
        raise ValueError(
            "G0.5 RTC trace expects 10 steps, contiguous FP16 RPU inputs, "
            f"and distinct action buffers with shape {expected}"
        )
    if tuple(action_keep_mask.shape) != (state.action_dim_pad,):
        raise ValueError(
            "G0.5 RTC trace keep mask must be [action_dim_pad]"
        )
    if (
        not 0 < rtc_prefix_len < 32
        or tuple(rtc_prefix.shape)
        != (1, rtc_prefix_len, state.action_dim_pad)
    ):
        raise ValueError(
            "G0.5 RTC trace prefix must be "
            "[1, inference_delay, action_dim_pad]"
        )

    plan = _plan_g05_action_execution(
        expert, prefix_len=prefix_len, include_action_io=True,
        action_route_flags=(
            _ACTION_ROUTE_IO | _ACTION_ROUTE_RTC | _ACTION_ROUTE_TRACE
        ),
        num_steps=num_steps,
        graph_cache=state.action_trace_graph_cache,
    )
    selected = plan.selected
    assert selected is not None
    descriptor = selected.stage_tuple.physical_descriptor
    sig = _rb.graph.GraphSignature(
        op_id="rpu_g05_action_rtc_trace_loop",
        shapes=[
            num_steps,
            32,
            state.hidden_size,
            prefix_len,
            state.action_dim_pad,
            rtc_prefix_len,
        ],
        dyn_dims=[
            state.num_layers,
            int(getattr(expert, "_rpu_execution_generation", 0)),
            *plan.graph_key_words(),
        ],
        dtypes=[torch.float16],
    )
    cache = state.action_trace_graph_cache
    if cache.is_frozen():
        cache.get_or_create(sig)
    modulation = (
        precomputed_modulation
        if precomputed_modulation is not None
        else _adaptive_modulation_stack(expert, time_conditions)
    )
    trace_shapes = {
        "pre_action": (num_steps, 1, 32, state.action_dim_pad),
        "action_embed": (num_steps, 1, 32, state.hidden_size),
        "final_hidden": (num_steps, 1, 32, state.hidden_size),
        "velocity": (num_steps, 1, 32, state.action_dim_pad),
        "post_action": (num_steps, 1, 32, state.action_dim_pad),
    }
    traces = {
        name: torch.empty(
            shape, dtype=torch.float16, device=action_input.device
        )
        for name, shape in trace_shapes.items()
    }
    if cache.lookup(sig) is None and cache.size() != 0:
        _evict_action_graph_entries(cache)
    with cache.capture(sig):
        torch.ops.rpu.qwen3_5_action_rtc_trace_forward(
            state.handle,
            action_input,
            action_keep_mask,
            action_output,
            rtc_prefix,
            rtc_prefix_len,
            delta_t,
            modulation,
            state.action_k_caches,
            state.action_v_caches,
            prefix_lens,
            traces["pre_action"],
            traces["action_embed"],
            traces["final_hidden"],
            traces["velocity"],
            traces["post_action"],
            descriptor,
        )
    _publish_g05_action_receipt(expert, plan, prefix_len=prefix_len)
    return {
        name: tensor.to("cpu").float().clone()
        for name, tensor in traces.items()
    }


def _rpu_g05_action_step_velocity(
    expert,
    action: torch.Tensor,
    time_cond: torch.Tensor,
    prefix_lens: list[int],
    prefix_len: int,
) -> torch.Tensor:
    """Diagnostic one-step velocity with the production Euler graph."""
    state = expert._rpu_qwen3_5
    if tuple(action.shape) != (1, 32, state.action_dim):
        raise ValueError(
            "G0.5 RPU action step expects action shape "
            f"[1, 32, {state.action_dim}]"
        )
    action_input = F.pad(
        action, (0, state.action_dim_pad - state.action_dim)
    ).to(device="rpu", dtype=torch.float16).contiguous()
    action_output = torch.empty_like(action_input)
    action_keep_mask = _cached_action_keep_mask(state, None)
    velocity = _rpu_g05_action_step(
        expert,
        action_input,
        action_keep_mask,
        action_output,
        0.1,
        time_cond,
        prefix_lens,
        prefix_len,
    )
    return velocity.to("cpu").float()[:, :, : state.action_dim].clone()


@execution_serialized
def _rpu_g05_inference_fm(
    self,
    attention_mask: torch.Tensor,
    pixel_values,
    past_key_values,
    action_dim_is_pad=None,
    position_ids_override=None,
    embodiment_types: list | None = None,
    prev_actions: torch.Tensor | None = None,
    inference_interval=None,
    inference_delay=None,
    **kwargs,
) -> torch.Tensor:
    """Canonical G0.5 FM loop with RPU-fused action I/O projections."""
    if prev_actions is None:
        prev_actions = kwargs.pop("prev_action", None)
    del kwargs
    helper = self.fm_helper
    expert = self.action_expert
    state = expert._rpu_qwen3_5
    first_image = next(iter(pixel_values.values()))
    if first_image.device.type != "cpu" or first_image.size(0) != 1:
        raise NotImplementedError(
            "G0.5 RPU FM inference requires batch_size=1 with host inputs"
        )
    dtype = first_image.dtype
    if dtype != torch.float32:
        raise NotImplementedError("G0.5 RPU FM inference requires FP32 host state")

    pad_mask = (
        action_dim_is_pad.bool().unsqueeze(1)
        if not helper.zero_pad_action_target and action_dim_is_pad is not None
        else None
    )
    dummy = torch.zeros(
        1, helper.horizon_steps, helper.action_dim,
        device="cpu", dtype=dtype,
    )
    sample_noise = getattr(helper, "_sample_noise", None)
    if callable(sample_noise):
        action = sample_noise(dummy, dtype, embodiment_types)
    else:
        if bool(getattr(helper, "use_correlated_noise", False)):
            raise NotImplementedError(
                "G0.5 RPU release inference does not support correlated noise"
            )
        action = torch.randn(
            (1, helper.horizon_steps, helper.action_dim),
            device="cpu",
            dtype=dtype,
        )
    if pad_mask is not None:
        action.masked_fill_(pad_mask, 0.0)

    use_rtc = bool(getattr(self, "use_training_rtc", False)) and (
        prev_actions is not None
    )
    rtc_prefix = None
    rtc_prefix_len = 0
    rtc_prefix_cpu = None
    if use_rtc:
        interval = _unique_scalar(inference_interval)
        delay = _unique_scalar(inference_delay)
        if interval is None or delay is None:
            raise ValueError(
                "G0.5 RTC inference requires inference_interval and inference_delay"
            )
        if delay < 0 or delay > helper.horizon_steps:
            raise ValueError(
                "G0.5 RTC inference_delay must be within the action horizon"
            )
        if (
            not isinstance(prev_actions, torch.Tensor)
            or prev_actions.device.type != "cpu"
            or tuple(prev_actions.shape)
            != (1, helper.horizon_steps, helper.action_dim)
            or not prev_actions.dtype.is_floating_point
            or not bool(torch.isfinite(prev_actions).all())
        ):
            raise ValueError(
                "G0.5 RTC prev_actions must be finite CPU "
                "[1, horizon_steps, action_dim]"
            )
        prefix_end = interval + delay
        if interval < 0 or prefix_end > prev_actions.shape[1]:
            raise ValueError(
                f"G0.5 RTC prefix slice [{interval}:{prefix_end}] exceeds "
                f"prev_actions horizon {prev_actions.shape[1]}"
            )
        if delay > 0:
            rtc_prefix_len = delay
            rtc_prefix_cpu = prev_actions[:, interval:prefix_end].to(
                device="cpu", dtype=dtype
            ).contiguous()
            rtc_prefix = F.pad(
                rtc_prefix_cpu,
                (0, state.action_dim_pad - state.action_dim),
            ).to(device="rpu", dtype=torch.float16).contiguous()

    prefix = self._build_prefix_action_kv(
        past_key_values, past_key_values.num_items()
    )
    prefix_len = int(prefix._rpu_g05_prefix_len)
    prefix_cache = prefix._rpu_g05_prefix_cache
    action_pos = _cached_action_position(
        self,
        state,
        attention_mask,
        position_ids_override,
        action_len=helper.horizon_steps,
        prefix_len=prefix_len,
        dtype=dtype,
        action_causal=helper.action_causal,
    )
    prefix_lens, _ = _prepare_prefix(expert, prefix_cache, prefix_len)
    _ensure_action_rope(state, action_pos, helper.horizon_steps)

    action_input = F.pad(
        action, (0, state.action_dim_pad - state.action_dim)
    ).to(device="rpu", dtype=torch.float16).contiguous()
    action_output = torch.empty_like(action_input)
    action_keep_mask = _cached_action_keep_mask(state, pad_mask)

    delta_t = 1.0 / helper.num_inference_steps
    if rtc_prefix is None:
        time_conditions, modulation = _fixed_action_time_schedule(
            expert, helper.num_inference_steps, dtype
        )
    else:
        time_conditions, modulation = _fixed_action_rtc_time_schedule(
            expert,
            helper.num_inference_steps,
            dtype,
            rtc_prefix_len,
        )
    _rpu_g05_action_loop(
        expert,
        action_input,
        action_keep_mask,
        action_output,
        delta_t,
        time_conditions,
        prefix_lens,
        prefix_len,
        precomputed_modulation=modulation,
        rtc_prefix=rtc_prefix,
        rtc_prefix_len=rtc_prefix_len,
    )

    action = action_output.to("cpu").float()[:, :, : state.action_dim].clone()
    if rtc_prefix_cpu is not None:
        action[:, :rtc_prefix_len].copy_(rtc_prefix_cpu)
    if pad_mask is not None:
        action.masked_fill_(pad_mask, 0.0)

    if helper.final_action_clip_value is not None:
        action = torch.clamp(
            action,
            -helper.final_action_clip_value,
            helper.final_action_clip_value,
        )
    return action


def patch_g05_action_expert_for_rpu(
    expert, *, max_seq_len: int = 2048, _execution_config=None
):
    """Patch G0.5 action inference for diagnostic one-step and ten-step RPU graphs."""
    from rpu_backend.api._execution import normalize_rpu_execution

    execution_config = normalize_rpu_execution(
        (
            {"action": {"chunk_size": "auto"}}
            if _execution_config is None else _execution_config
        ),
        entry_point="patch_g05_action_expert_for_rpu",
        supported={"action": ("chunk_size",)},
    )
    requested_chunk = execution_config["action"].get("chunk_size", "auto")
    if isinstance(requested_chunk, int) and requested_chunk != _ACTION_HORIZON:
        raise ValueError(
            "G0.5 action chunk_size must be 'auto' or exactly 32; got "
            f"{requested_chunk}"
        )
    if getattr(expert, "_rpu_g05_action_ready", False):
        if _g05_action_runtime_ready(expert):
            if getattr(expert, "_rpu_execution", None) != execution_config:
                raise ValueError(
                    "G0.5 action runtime execution config conflicts with the "
                    "installed component config"
                )
            return expert
        raise RuntimeError(
            "G0.5 action ready marker has incomplete runtime state; reload "
            "the checkpoint"
        )
    if getattr(expert, "_rpu_g05_action_install_started", False):
        raise RuntimeError(
            "G0.5 action installation was already attempted; reload the checkpoint "
            "to avoid double-swizzle"
        )
    if expert.training:
        raise NotImplementedError(
            "G0.5 RPU action expert supports inference only; call expert.eval()"
        )
    if max_seq_len < 32:
        raise ValueError("max_seq_len must cover the 32-token action horizon")
    parameters = getattr(expert, "parameters", None)
    if callable(parameters) and any(
        parameter.device.type != "cpu" for parameter in parameters()
    ):
        raise NotImplementedError(
            "G0.5 RPU action patch requires a CPU-hosted checkpoint"
        )
    _check_g05_action_profile(expert.config)
    if not _ACTION_INSTALL_LOCK.acquire(blocking=False):
        raise RuntimeError("another G0.5 action installation is in progress")

    state_before = None
    snapshots = ()
    try:
        state_before = getattr(expert, "_rpu_qwen3_5", None)
        if (
            state_before is not None
            or getattr(expert, "_rpu_qwen3_5_text_install_started", False)
        ):
            raise RuntimeError(
                "G0.5 action expert has an existing or previously attempted "
                "Qwen3.5 runtime; reload the checkpoint"
            )
        snapshots = tuple(
            (name, name in vars(expert), vars(expert).get(name))
            for name in (
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
            )
        )
        expert._rpu_g05_action_install_started = True
        # Prepare attention/MLP weights in CPU FP16; the shared installer
        # uploads only their final swizzled layouts. AdaLN and time
        # conditioning stay FP32 on the host.
        for layer in expert.layers:
            layer.self_attn.to("cpu").half()
            layer.mlp.to("cpu").half()
            layer.input_layernorm.to("cpu").float()
            layer.post_attention_layernorm.to("cpu").float()
        expert.norm.to("cpu").float()
        for name in (
            "input_proj", "output_proj", "time_embedding",
            "time_mlp_in", "time_mlp_out",
        ):
            module = getattr(expert, name, None)
            if module is not None:
                module.to("cpu").float()

        from rpu_backend.adapters.qwen3_5.text import install_qwen3_5_text_for_rpu
        from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache

        handle = install_qwen3_5_text_for_rpu(
            expert,
            expert.config,
            max_seq_len=max_seq_len,
            _cpu_stage_weights=True,
        )
        torch.ops.rpu.qwen3_5_set_linear_acc32(handle, True)
        torch.ops.rpu.qwen3_5_enable_action_mode(handle)
        torch.ops.rpu.qwen3_5_set_fast_replay(handle, True)
        from rpu_backend.runtime.weights import tp_col_swizzle_mc_weight

        input_w = tp_col_swizzle_mc_weight(
            F.pad(
                expert.input_proj.weight.detach().half(),
                (0, _ACTION_DIM_PAD - expert.config.input_dim),
            ),
            num_cores=1,
        ).contiguous().to("rpu")
        input_b = expert.input_proj.bias.detach().half().contiguous().to("rpu")
        output_w = tp_col_swizzle_mc_weight(
            F.pad(
                expert.output_proj.weight.detach().half(),
                (0, 0, 0, _ACTION_DIM_PAD - expert.config.output_dim),
            ),
            num_cores=1,
        ).contiguous().to("rpu")
        output_b = F.pad(
            expert.output_proj.bias.detach().half(),
            (0, _ACTION_DIM_PAD - expert.config.output_dim),
        ).contiguous().to("rpu")
        expert._rpu_qwen3_5.retirement.keepalive += (input_w, input_b, output_w, output_b)
        torch.ops.rpu.qwen3_5_set_action_io_weights(
            handle,
            input_w,
            input_b,
            output_w,
            output_b,
            expert.config.input_dim,
            _ACTION_DIM_PAD,
            32,
        )
        torch.ops.rpu.qwen3_5_set_action_chunk_size(
            handle,
            0 if requested_chunk == "auto" else int(requested_chunk),
        )
        torch.ops.rpu.qwen3_5_enable_execution_reconfigure(handle)
        import rpu_backend as _rb

        state = expert._rpu_qwen3_5
        state.action_graph_cache = _rb.graph.GraphCache(max_entries=1)
        state.action_trace_graph_cache = _rb.graph.GraphCache(max_entries=1)
        from rpu_backend.adapters.qwen3_5.text import _refresh_qwen35_text_retirement

        _refresh_qwen35_text_retirement(state)
        state.action_cache = Qwen3_5Cache.from_config(
            expert.config, max_seq_len=max_seq_len
        )
        state.action_k_caches = state.action_cache.k_caches
        state.action_v_caches = state.action_cache.v_caches
        state.retirement.keepalive += (state.action_cache,)
        state.action_prefix_binding = None
        state.action_prefix_key = None
        state.action_prefix_source = None
        state.action_prefix_lens = None
        state.action_rope_cache = None
        state.action_layout_cache = None
        state.action_keep_mask_cache = None
        state.action_dim = int(expert.config.input_dim)
        state.action_dim_pad = _ACTION_DIM_PAD
        state.execution_config = execution_config
        expert._rpu_execution = execution_config
        expert._rpu_execution_component_id = _G05_ACTION_COMPONENT
        expert._rpu_execution_generation = 0
        expert._rpu_g05_adaptive_mod_cache = {}
        expert._rpu_g05_adaptive_mod_stack_cache = None
        expert._rpu_g05_time_cond_cache = {}
        expert._rpu_g05_fixed_time_schedule = None
        expert._rpu_g05_rtc_time_schedules = {}
        expert._rpu_g05_original_encode_time = expert.encode_time
        expert.encode_time = types.MethodType(_rpu_g05_encode_time, expert)
        expert._rpu_g05_original_forward = expert.forward
        expert.forward = types.MethodType(_rpu_g05_action_forward, expert)
        expert._rpu_g05_action_ready = True
        return expert
    except BaseException as error:
        state = getattr(expert, "_rpu_qwen3_5", None)
        if state is not None and state is not state_before:
            from rpu_backend.adapters.qwen3_5.text import _retire_failed_qwen3_5_text_install

            try:
                _retire_failed_qwen3_5_text_install(expert, state=state)
            except BaseException as cleanup_error:
                error.add_note(f"G0.5 action install cleanup failed: {cleanup_error!r}")
        for name, had_value, value in reversed(snapshots):
            if had_value:
                vars(expert)[name] = value
            else:
                vars(expert).pop(name, None)
        # The shared text install may already have swizzled weights. Preserve
        # both install-started markers so retry requires a fresh checkpoint.
        raise
    finally:
        _ACTION_INSTALL_LOCK.release()


__all__ = ["patch_g05_action_expert_for_rpu"]
