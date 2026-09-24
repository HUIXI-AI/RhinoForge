"""RhinoVLA fused forward (P1).

Patches a RhinoVLA action expert (depth-18 / 72-D) to run the fused C++ expert
via the DEDICATED RhinoVLA ops `torch.ops.rpu.rhino_vla_{create,set_weights,
forward,destroy}` (src/fused/rpu_rhino_vla_model.cpp — an independent fork of the
qwenpi05 expert so RhinoVLA can be optimized in isolation). The expert math is
identical to qwenpi05's; RhinoVLA specifics (72-D IO, mask_condition,
instance-LoRA merge, 0→1 denoise) live in the adapter / test harness, and LoRA
is merged into base weights before set_weights so the C++ op never sees it.

Prerequisites (required caller order):
    expert = convert.convert_expert_for_rpu(expert)   # half() -> swizzle -> rpu
    patch_rhino_vla_for_rpu(expert, prefix_len=P, suffix_len=S, cpu_rotary_emb=...)

action_io + the final AdaRMSNorm stay on CPU fp32 (suffix construction + decode
run in PyTorch), so the qwenpi05 dtype monkey-patches are not needed here.
"""

import copy
import dataclasses
import sys
import types
from numbers import Integral

import torch

from rpu_backend.runtime import rpu_env_bool
from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
from rpu_backend.runtime.execution_planner import GRAPH_COMPOSITE_CHILD


def _rhino_action_math(expert):
    """Use the helpers belonging to the loaded expert, not a different checkout."""
    names = ("make_suffix_attn_mask", "bool_mask_to_attention_bias", "get_cache_layer")
    for cls in type(expert).__mro__:
        module = sys.modules.get(cls.__module__)
        helpers = tuple(getattr(module, name, None) for name in names)
        if all(callable(helper) for helper in helpers):
            return helpers
    raise RuntimeError("RhinoVLA expert source does not expose its attention/cache helpers")


def _cached_rhino_attention_bias(owner, prefix_mask, suffix_mask,
                                 make_suffix_attn_mask, bool_mask_to_attention_bias):
    """One content-owned CPU mask shared by step and unroll callers.

    Tensor identity is not a semantic key: request masks may be mutated in
    place or be inference tensors without versions. Snapshot their actual
    binary values, retaining physical lengths and padded suffix rows.
    """
    prefix = prefix_mask.detach().to(device="cpu").contiguous()
    suffix = suffix_mask.detach().to(device="cpu").contiguous()
    key = (str(prefix.dtype), tuple(prefix.shape), tuple(prefix.reshape(-1).tolist()),
           str(suffix.dtype), tuple(suffix.shape), tuple(suffix.reshape(-1).tolist()),
           make_suffix_attn_mask, bool_mask_to_attention_bias)
    cached = getattr(owner, "_rhino_attention_bias_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    bias = bool_mask_to_attention_bias(make_suffix_attn_mask(prefix, suffix), torch.float16)
    pad_k = (-bias.shape[-1]) % 16
    if pad_k:
        bias = torch.nn.functional.pad(bias, (0, pad_k), value=torch.finfo(bias.dtype).min)
    result = versioned_cpu_mask(bias)
    owner._rhino_attention_bias_cache = (key, result)
    return result


def _aligned_rhino_action_prefix(prefix_mask, *, prefix_len, suffix_len, capacity):
    """Pad only the expert's KV prefix; text position and logical RoPE stay live."""
    if prefix_mask.ndim != 2 or prefix_mask.shape[1] != prefix_len:
        raise ValueError("RhinoVLA aligned KV prefix mask must match physical text rows")
    _validate_rhino_prefix_mask(
        runtime_prefix_len=prefix_len, batch_size=prefix_mask.shape[0],
        prefix_mask=prefix_mask,
    )
    position = ((prefix_len + 15) // 16) * 16
    if suffix_len != 31 or position + 32 > capacity:
        raise ValueError("RhinoVLA aligned KV requires M31 and capacity for its 32-row write")
    if position == prefix_len:
        return position, prefix_mask
    return position, torch.nn.functional.pad(prefix_mask, (0, position - prefix_len), value=0)


@dataclasses.dataclass(frozen=True)
class _RhinoVLANativeColdConfig:
    """One immutable Python-owned selector snapshot for a native handle."""

    denoise_static_context_cache: bool = False
    fused_adarms: bool = False
    skip_adarms_gemv: bool = False
    precompute_adarms: bool = False
    precompute_time_proj: bool = False
    fold_action_time_in: bool = False
    gated_no_sub: bool = False
    fused_silu_mul: bool = False
    expert_fusions: bool = False
    adarms_resident: bool = False
    packed_qkv: bool = False
    aligned_kv: bool = False

    @classmethod
    def from_env(cls) -> "_RhinoVLANativeColdConfig":
        config = cls(
            denoise_static_context_cache=rpu_env_bool(
                "RPU_RHINOVLA_DENOISE_STATIC_CONTEXT_CACHE"
            ),
            fused_adarms=rpu_env_bool("RPU_RHINOVLA_FUSED_ADARMS_GEMV"),
            skip_adarms_gemv=rpu_env_bool(
                "RPU_RHINOVLA_SKIP_ADARMS_GEMV"
            ),
            precompute_adarms=rpu_env_bool(
                "RPU_RHINOVLA_PRECOMPUTE_ADARMS"
            ),
            precompute_time_proj=rpu_env_bool(
                "RPU_RHINOVLA_PRECOMPUTE_TIME_PROJ"
            ),
            fold_action_time_in=rpu_env_bool(
                "RPU_RHINOVLA_FOLD_ACTION_TIME_IN"
            ),
            gated_no_sub=rpu_env_bool("RPU_RHINOVLA_GATED_NO_SUB"),
            fused_silu_mul=rpu_env_bool("RPU_RHINOVLA_FUSED_SILU_MUL"),
            expert_fusions=rpu_env_bool("RPU_RHINOVLA_EXPERT_FUSIONS"),
            adarms_resident=rpu_env_bool("RPU_RHINOVLA_ADARMS_RESIDENT"),
            packed_qkv=rpu_env_bool("RPU_RHINOVLA_PACKED_QKV"),
            aligned_kv=rpu_env_bool("RPU_RHINOVLA_ALIGNED_KV"),
        )
        if config.expert_fusions or config.adarms_resident:
            if not config.precompute_adarms or config.skip_adarms_gemv:
                raise ValueError("RhinoVLA expert transfer requires PRECOMPUTE_ADARMS and rejects SKIP_ADARMS_GEMV")
            if config.expert_fusions and not config.gated_no_sub:
                raise ValueError("RhinoVLA EXPERT_FUSIONS requires GATED_NO_SUB")
        return config


def plan_rhino_action_execution(
    expert,
    *,
    logical_len,
    execution_len,
    prefix_len,
    kv_len,
    use_attention_mask,
    is_causal,
    prefix_plan_key_words=(),
):
    """Resolve the exact native action descriptor before Graph admission."""
    component = str(getattr(
        expert, "_fmb_execution_component_id", "action_expert"
    ))
    generation = int(getattr(expert, "_fmb_execution_generation", 0))
    action_config = getattr(expert, "_rpu_execution", {}).get("action", {})
    requested_chunk = action_config.get("chunk_size", "auto")
    exact_chunk = requested_chunk if isinstance(requested_chunk, int) else None
    prefix_plan_key_words = tuple(int(word) for word in prefix_plan_key_words)
    handle = int(expert._rpu_handle)
    plan_box = {}
    resolved_execution, resolved_chunk = plan_bounded_prefill_execution(
        int(logical_len),
        int(execution_len),
        int(execution_len) - int(logical_len),
        execution_owner=expert,
        execution_stage="action",
        execution_native=("rhino_vla", handle),
        plan_signature=(int(kv_len), bool(use_attention_mask), bool(is_causal)),
        graph_cache=getattr(expert, "_rpu_graph_cache", None),
        position=int(prefix_len),
        alignment=1,
        padding_rows=int(execution_len) - int(logical_len),
        exact_chunk_size=exact_chunk,
        resolve_stage_domain=lambda length: (
            torch.ops.rpu.rhino_vla_resolve_action_stage_domain(
                handle,
                int(length),
                int(logical_len),
                int(prefix_len),
                int(kv_len),
                bool(use_attention_mask),
                bool(is_causal),
                0 if exact_chunk is None else int(exact_chunk),
            )
        ),
        request_id=f"rhinovla:{component}:action",
        plan_result_sink=lambda result: plan_box.__setitem__("result", result),
        graph_mode=GRAPH_COMPOSITE_CHILD,
        queue_owner_id=handle,
        physical_metadata=(
            (f"component:{component}", 1),
            ("execution_generation", generation),
            ("prefix_len", int(prefix_len)),
            ("kv_len", int(kv_len)),
            ("direct_prefix_plan", int(bool(prefix_plan_key_words))),
            *(tuple(
                (f"prefix_plan_digest_{index}", word)
                for index, word in enumerate(prefix_plan_key_words)
            )),
        ),
    )
    result = plan_box["result"]
    selected = result.selected
    if (
        selected is None
        or resolved_execution != int(execution_len)
        or resolved_chunk != selected.stage_tuple.compute_chunk
        or not selected.stage_tuple.physical_descriptor
    ):
        raise RuntimeError(
            "RhinoVLA action dry planner returned no consumable native "
            "stage descriptor"
        )
    return result


def publish_rhino_action_execution_receipt(
    expert,
    plan,
    *,
    logical_len,
    prefix_plan=None,
    prefix_route_argument_digest="",
):
    """Publish one standard receipt after native dry/forward agreement."""
    selected = plan.selected
    if selected is None:
        raise RuntimeError("RhinoVLA action planner selected no plan")
    resolved_chunk = int(
        torch.ops.rpu.rhino_vla_get_resolved_chunk_size(expert._rpu_handle)
    )
    if resolved_chunk != selected.stage_tuple.compute_chunk:
        raise RuntimeError(
            "RhinoVLA action dry/forward chunk plan drift: dry="
            f"{selected.stage_tuple.compute_chunk}, forward={resolved_chunk}"
        )
    receipt = plan.as_dict(include_candidates=False)
    receipt.update({
        "stage": "action",
        "component": getattr(
            expert, "_fmb_execution_component_id", "action_expert"
        ),
        "generation": int(getattr(
            expert, "_fmb_execution_generation", 0
        )),
        "logical_len": int(logical_len),
        "execution_len": int(selected.execution_len),
        "chunk_size": resolved_chunk,
        "padding_rows": int(selected.padding_rows),
        "position": dict(
            selected.stage_tuple.physical_metadata
        )["prefix_len"],
        "authority": "NATIVE_A6_STAGE_DESCRIPTOR",
        "attention_policy": "DDR_REQUIRED",
        "dry_forward_agreement": True,
    })
    if prefix_plan is not None:
        receipt["prefix_kv"] = {
            **prefix_plan.as_dict(include_candidates=False),
            "stage": "prefix_kv",
            "authority": "TYPED_KV_INSERT_ROUTE",
            "route_argument_digest": str(prefix_route_argument_digest),
        }
    vars(expert)["_rpu_last_execution_plan"] = receipt
    return receipt


_RHINO_RUNTIME_INSTALL_ATTRS = (
    "_rpu_graph_cache",
    "_rpu_cos_cached",
    "_rpu_sin_cached",
    "_rpu_rope_max_seq_len",
    "_rpu_suffix_len",
    "_rpu_kv_cache",
    "_rpu_prefix_source_snapshot",
    "_rpu_execution",
    "_fmb_execution_component_id",
    "_fmb_execution_generation",
    "_rpu_native_cold_config",
    "_rpu_full_w8a16",
    "_rpu_last_execution_plan",
    "_rpu_handle",
    "_rpu_handle_finalizer",
    "_rpu_retirement_state",
    "forward",
)


def _rhino_tensor_version(tensor):
    """Return a mutation counter when PyTorch exposes one.

    Inference tensors may deliberately omit version counters.  Identity still
    protects those tensors from being replaced; ordinary tensors additionally
    fail closed after an in-place update.
    """
    try:
        return int(tensor._version)
    except RuntimeError:
        return None


def _snapshot_rhino_prefix_source(
    prefix_key_values,
    prefix_layer_offset,
    prefix_pairs,
    *,
    route_argument_digest="",
    execution_generation=0,
):
    """Keep strong refs plus versions for the exact K/V layers copied to RPU."""
    return (
        prefix_key_values,
        int(prefix_layer_offset),
        str(route_argument_digest),
        int(execution_generation),
        tuple(
            (
                prefix_k,
                _rhino_tensor_version(prefix_k),
                prefix_v,
                _rhino_tensor_version(prefix_v),
            )
            for prefix_k, prefix_v in prefix_pairs
        ),
    )


def _rhino_prefix_source_matches(snapshot, candidate) -> bool:
    """Compare snapshots without invoking tensor value equality."""
    if snapshot is None or candidate is None:
        return snapshot is candidate
    old_owner, old_offset, old_route, old_generation, old_pairs = snapshot
    new_owner, new_offset, new_route, new_generation, new_pairs = candidate
    if (
        old_owner is not new_owner
        or old_offset != new_offset
        or old_route != new_route
        or old_generation != new_generation
        or len(old_pairs) != len(new_pairs)
    ):
        return False
    return all(
        old_k is new_k
        and old_k_version == new_k_version
        and old_v is new_v
        and old_v_version == new_v_version
        for (
            old_k,
            old_k_version,
            old_v,
            old_v_version,
        ), (
            new_k,
            new_k_version,
            new_v,
            new_v_version,
        ) in zip(old_pairs, new_pairs)
    )


def _validate_rhino_prefix_rope_contract(
    *,
    runtime_prefix_len,
    rope_max_seq_len,
    suffix_len,
    batch_size,
    position_ids,
    prefix_mask,
    suffix_mask,
    prefix_rope_deltas=None,
):
    """Return the suffix RoPE start, independent of physical prefix KV rows."""
    runtime_prefix_len = int(runtime_prefix_len)
    rope_max_seq_len = int(rope_max_seq_len)
    suffix_len = int(suffix_len)
    batch_size = int(batch_size)
    if runtime_prefix_len < 0 or suffix_len <= 0 or batch_size <= 0:
        raise ValueError(
            "RhinoVLA fused: prefix_len must be non-negative and suffix_len/"
            "batch_size must be positive"
        )

    logical_prefix_len = _rhino_suffix_rope_start(
        runtime_prefix_len=runtime_prefix_len,
        batch_size=batch_size,
        prefix_mask=prefix_mask,
        prefix_rope_deltas=prefix_rope_deltas,
    )

    if suffix_mask is not None:
        if not isinstance(suffix_mask, torch.Tensor):
            raise TypeError("RhinoVLA fused: suffix_mask must be a tensor or None")
        suffix_cpu = suffix_mask.detach().to("cpu")
        if suffix_cpu.numel() != batch_size * suffix_len:
            raise RuntimeError(
                "RhinoVLA fused: suffix_mask has "
                f"{suffix_cpu.numel()} entries, expected {batch_size * suffix_len}."
            )
        if suffix_cpu.numel() and not bool(
            ((suffix_cpu == 0) | (suffix_cpu == 1)).all()
        ):
            raise ValueError("RhinoVLA fused: suffix_mask must be binary")
        if suffix_cpu.numel() and not bool(suffix_cpu.bool().all()):
            raise NotImplementedError(
                "RhinoVLA fused: padded suffix rows are not supported"
            )

    if position_ids is None:
        raise NotImplementedError(
            "RhinoVLA fused: explicit contiguous position_ids are required"
        )
    integer_dtypes = (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    if (
        not isinstance(position_ids, torch.Tensor)
        or position_ids.dtype not in integer_dtypes
    ):
        raise TypeError(
            "RhinoVLA fused: position_ids must be an integer tensor"
        )
    actual = position_ids.detach().to("cpu", dtype=torch.long)
    expected = torch.arange(
        logical_prefix_len,
        logical_prefix_len + suffix_len,
        dtype=torch.long,
    ).view(1, 1, suffix_len).expand(3, batch_size, suffix_len)
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        raise NotImplementedError(
            "RhinoVLA fused: position_ids must be the contiguous range "
            f"[{logical_prefix_len}, {logical_prefix_len + suffix_len}) on all "
            f"three M-RoPE axes; got shape={tuple(actual.shape)}."
        )
    if logical_prefix_len + suffix_len > rope_max_seq_len:
        raise RuntimeError(
            "RhinoVLA fused: logical prefix + suffix exceeds the installed "
            f"RoPE table: {logical_prefix_len}+{suffix_len}>{rope_max_seq_len}"
        )
    return logical_prefix_len


def _rhino_suffix_rope_start(
    *, runtime_prefix_len, batch_size, prefix_mask, prefix_rope_deltas=None,
):
    """Apply the model's continuation delta, not a planner search setting."""
    logical_rows = _validate_rhino_prefix_mask(
        runtime_prefix_len=runtime_prefix_len,
        batch_size=batch_size,
        prefix_mask=prefix_mask,
    )
    delta = prefix_rope_deltas
    if delta is None:
        delta = 0
    if isinstance(delta, torch.Tensor):
        if delta.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("RhinoVLA prefix_rope_deltas must be an integer tensor")
        if tuple(delta.shape) not in ((batch_size,), (batch_size, 1)):
            raise ValueError("RhinoVLA prefix_rope_deltas must have shape [B] or [B,1]")
        values = delta.detach().cpu().reshape(-1)
        if not torch.equal(values, values[:1].expand_as(values)):
            raise ValueError("RhinoVLA prefix RoPE deltas must agree across batch rows")
        delta = int(values[0])
    if isinstance(delta, bool) or not isinstance(delta, Integral):
        raise TypeError("RhinoVLA prefix_rope_deltas must be an integer or integer tensor")
    position = logical_rows + int(delta)
    if position < 0:
        raise ValueError("RhinoVLA logical prefix + RoPE delta must be non-negative")
    return position


def _validate_rhino_prefix_mask(
    *, runtime_prefix_len, batch_size, prefix_mask
):
    """Resolve logical rows while preserving a separate physical cache row."""
    runtime_prefix_len = int(runtime_prefix_len)
    batch_size = int(batch_size)
    if runtime_prefix_len < 0 or batch_size <= 0:
        raise ValueError(
            "RhinoVLA fused: prefix_len must be non-negative and batch_size "
            "must be positive"
        )
    if prefix_mask is None:
        return runtime_prefix_len
    if not isinstance(prefix_mask, torch.Tensor):
        raise TypeError("RhinoVLA fused: prefix_mask must be a tensor or None")
    mask_cpu = prefix_mask.detach().to("cpu")
    if mask_cpu.numel() != batch_size * runtime_prefix_len:
        raise RuntimeError(
            "RhinoVLA fused: prefix_mask has "
            f"{mask_cpu.numel()} entries, expected "
            f"{batch_size * runtime_prefix_len}."
        )
    if mask_cpu.numel() and not bool(
        ((mask_cpu == 0) | (mask_cpu == 1)).all()
    ):
        raise ValueError("RhinoVLA fused: prefix_mask must be binary")
    mask_cpu = mask_cpu.reshape(batch_size, runtime_prefix_len).bool()
    # Only right padding preserves the prefix's logical token order.
    if runtime_prefix_len > 1 and bool(
        (mask_cpu[:, 1:] & ~mask_cpu[:, :-1]).any()
    ):
        raise ValueError(
            "RhinoVLA fused: prefix_mask must contain valid rows followed "
            "only by right-padding rows"
        )
    logical_lengths = mask_cpu.sum(dim=1)
    if not bool((logical_lengths == logical_lengths[0]).all()):
        raise ValueError(
            "RhinoVLA fused: all batch items must have the same logical "
            "prefix length"
        )
    return int(logical_lengths[0])


def patch_rhino_vla_for_rpu(expert, *, prefix_len, suffix_len, max_seq_len=2048,
                            cpu_rotary_emb=None, full_w8a16=False):
    """Patch a RhinoVLA action expert transactionally onto its fused C++ op.

    Weight/M-RoPE/KV/GraphCache state is prepared before native handle creation.
    A failed replacement cleans only its pending handle and preserves the
    currently published handle/finalizer/cache/forward.

    Args:
        expert: action expert, already converted via convert_expert_for_rpu.
        prefix_len: Initial prefix length; retained for constructor compatibility.
            Runtime prefixes may vary within ``max_seq_len``.
        suffix_len: suffix token count = action_horizon + (1 if use_state_token).
        max_seq_len: RPUCache allocation length.
        cpu_rotary_emb: a CPU clone of the Qwen3-VL rotary (cloned BEFORE .to('rpu')).

    Returns:
        int handle (also stored as expert._rpu_handle).
    """
    from rpu_backend.adapters.rhinovla.convert import (
        _gather_expert_weights, _gather_expert_scales,
        _precompute_mrope_cos_sin, _create_kv_cache,
    )

    for name, value, minimum in (
        ("prefix_len", prefix_len, 0),
        ("suffix_len", suffix_len, 1),
        ("max_seq_len", max_seq_len, 1),
    ):
        if (isinstance(value, bool) or not isinstance(value, Integral)
                or value < minimum):
            raise ValueError(
                f"patch_rhino_vla_for_rpu: {name} must be an integer >= "
                f"{minimum}, got {value!r}")
    prefix_len = int(prefix_len)
    suffix_len = int(suffix_len)
    max_seq_len = int(max_seq_len)
    if prefix_len + suffix_len > max_seq_len:
        raise ValueError(
            "patch_rhino_vla_for_rpu: initial prefix + suffix exceeds "
            f"max_seq_len ({prefix_len}+{suffix_len}>{max_seq_len})")

    if not hasattr(expert, 'layers'):
        raise RuntimeError("patch_rhino_vla_for_rpu: expert has no .layers")
    if not hasattr(expert, 'config'):
        raise RuntimeError("patch_rhino_vla_for_rpu: expert has no .config")
    if len(expert.layers) == 0:
        raise RuntimeError("patch_rhino_vla_for_rpu: expert has no layers")

    native_cold = _RhinoVLANativeColdConfig.from_env()

    expert_state = vars(expert)
    install_snapshot = {
        name: expert_state[name]
        for name in _RHINO_RUNTIME_INSTALL_ATTRS
        if name in expert_state
    }
    had_old_handle = "_rpu_handle" in install_snapshot
    old_handle = install_snapshot.get("_rpu_handle")
    old_finalizer = install_snapshot.get("_rpu_handle_finalizer")
    old_resource = install_snapshot.get("_rpu_retirement_state")
    if old_resource is not None:
        old_resource.require_replaceable()
    else:
        from rpu_backend.api._execution import _require_execution_process_safe
        _require_execution_process_safe()

    import rpu_backend as _rb
    graph_cache = _rb.graph.GraphCache()
    _GraphSignature = _rb.graph.GraphSignature

    # -- gather per-layer weight lists --
    scale_lists = _gather_expert_scales(expert)
    if full_w8a16 and (not scale_lists or not native_cold.precompute_adarms
                       or native_cold.skip_adarms_gemv):
        raise ValueError("RhinoVLA full W8 requires expert W8 and PRECOMPUTE_ADARMS without SKIP_ADARMS_GEMV")
    if scale_lists and native_cold.packed_qkv:
        raise ValueError("RhinoVLA EXPERT_W8A16 is incompatible with PACKED_QKV")
    weights = (_gather_expert_weights(expert, full_w8a16=True)
               if full_w8a16 else _gather_expert_weights(expert))
    cond_owners = getattr(expert, "_rpu_full_w8_cond_owners", None) if full_w8a16 else None

    # -- Stable full M-RoPE table. RoPE uses a logical per-forward start while
    #    KV insert keeps using the physical prefix row (Pi0.5's proven split).
    #    The raw Qwen3-VL rotary computes on CPU when fed CPU tensors, so no
    #    wrapper is needed; just ensure it lives on CPU.
    if cpu_rotary_emb is not None:
        cpu_rotary = cpu_rotary_emb.cpu()
    else:
        rotary_src = expert.qwen_rotary_emb
        if hasattr(rotary_src, '_cpu_emb'):
            rotary_src = rotary_src._cpu_emb
        cpu_rotary = copy.deepcopy(rotary_src).cpu()
    cos_cached, sin_cached = _precompute_mrope_cos_sin(
        cpu_rotary, 0, max_seq_len
    )
    kv_cache = _create_kv_cache(expert.config, max_seq_len)
    print(f"[Rhino] M-RoPE precomputed: cos/sin [{max_seq_len}, {cos_cached.shape[-1]}], "
          f"KV cache max_seq={max_seq_len}")

    # -- prepare set_weights args (reused qwenpi05 op) --
    config = expert.config
    set_weights_args = (
        *weights,  # q,k,v,o,gate,up,down, adarms dense/pair dense, q_norm,k_norm
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.width,
        config.mlp_dim,
        config.rms_norm_eps,
    )

    # -- final AdaRMSNorm runs in Python on CPU fp32 (D-402) --
    if getattr(expert, 'norm', None) is not None:
        expert.norm = expert.norm.to('cpu').to(dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Forward replacement
    # ------------------------------------------------------------------ #
    def rpu_rhino_vla_fused_forward(
        self,
        suffix_embeds,
        *,
        prefix_key_values=None,
        prefix_mask=None,
        suffix_mask=None,
        position_ids=None,
        adarms_cond=None,
        prefix_layer_offset=0,
        prefix_rope_deltas=None,
    ):
        """Fused velocity forward: routes through the reused C++ qwenpi05_forward op."""
        make_suffix_attn_mask, bool_mask_to_attention_bias, get_cache_layer = _rhino_action_math(self)
        from rpu_backend.adapters.rhinovla.convert import _prefill_kv_cache
        from rpu_backend.adapters.rhinovla.convert import (
            _plan_rhino_prefix_kv_execution,
        )

        orig_device = suffix_embeds.device
        if suffix_embeds.shape[0] != 1:
            raise AssertionError(
                "RhinoVLA fused: batch_size must be 1, got "
                f"{suffix_embeds.shape[0]}"
            )

        # Resolve the physical prefix row independently from the suffix's
        # logical RoPE start before any cache fill or graph dispatch.
        cache = self._rpu_kv_cache
        prefix_source_snapshot = None
        prefix_plan = None
        prefix_route_arguments = ()
        prefix_route_argument_digest = ""
        if prefix_key_values is not None:
            prefix_lengths = []
            prefix_pairs = []
            for layer_idx in range(len(self.layers)):
                prefix_k, prefix_v = get_cache_layer(
                    prefix_key_values,
                    prefix_layer_offset + layer_idx,
                )
                if prefix_k is None or prefix_v is None:
                    raise RuntimeError(
                        "RhinoVLA fused: prefix KV is missing layer "
                        f"{prefix_layer_offset + layer_idx}"
                    )
                if prefix_k.shape[2] != prefix_v.shape[2]:
                    raise RuntimeError(
                        "RhinoVLA fused: prefix K/V length mismatch at layer "
                        f"{prefix_layer_offset + layer_idx}: "
                        f"K={prefix_k.shape[2]}, V={prefix_v.shape[2]}"
                    )
                prefix_lengths.append(int(prefix_k.shape[2]))
                prefix_pairs.append((prefix_k, prefix_v))
            runtime_prefix_len = prefix_lengths[0] if prefix_lengths else 0
            if any(length != runtime_prefix_len for length in prefix_lengths):
                raise RuntimeError(
                    "RhinoVLA fused: prefix KV lengths differ across layers: "
                    f"{prefix_lengths}"
                )
            generation = int(getattr(self, "_fmb_execution_generation", 0))
            component = str(getattr(
                self, "_fmb_execution_component_id", "action_expert"
            ))
            (
                prefix_plan,
                prefix_route_arguments,
                prefix_route_argument_digest,
            ) = _plan_rhino_prefix_kv_execution(
                prefix_pairs,
                component=component,
                generation=generation,
                queue_owner_id=int(self._rpu_handle),
            )
            prefix_source_snapshot = _snapshot_rhino_prefix_source(
                prefix_key_values,
                prefix_layer_offset,
                prefix_pairs,
                route_argument_digest=prefix_route_argument_digest,
                execution_generation=generation,
            )
            if cache.position != 0:
                if int(cache.position) != runtime_prefix_len:
                    raise RuntimeError(
                        "RhinoVLA fused: cached physical prefix length "
                        f"{int(cache.position)} differs from the supplied "
                        f"prefix length {runtime_prefix_len}; call "
                        "cache.reset_to_position(0) before a new prefix"
                    )
                if not _rhino_prefix_source_matches(
                    self._rpu_prefix_source_snapshot,
                    prefix_source_snapshot,
                ):
                    raise RuntimeError(
                        "RhinoVLA fused: the supplied prefix source, layer "
                        "offset, or selected K/V tensors changed while the "
                        "previous prefix is still cached; call "
                        "cache.reset_to_position(0) before a new prefix"
                    )
        elif prefix_mask is not None:
            # Shared-cache mode preloads the expert cache before this call.
            runtime_prefix_len = int(cache.position)
        else:
            runtime_prefix_len = 0
        actual_suffix_len = int(suffix_embeds.shape[1])
        target_suffix_len = int(self._rpu_suffix_len)
        if actual_suffix_len > target_suffix_len:
            raise RuntimeError(
                f"RhinoVLA fused: runtime suffix_len {actual_suffix_len} "
                "exceeds the precomputed cos/sin table length "
                f"{target_suffix_len} — re-patch with "
                "patch_rhino_vla_for_rpu(suffix_len >= this value)."
            )
        logical_prefix_len = _validate_rhino_prefix_rope_contract(
            runtime_prefix_len=runtime_prefix_len,
            rope_max_seq_len=self._rpu_rope_max_seq_len,
            suffix_len=actual_suffix_len,
            batch_size=int(suffix_embeds.shape[0]),
            position_ids=position_ids,
            prefix_mask=prefix_mask,
            suffix_mask=suffix_mask,
            prefix_rope_deltas=prefix_rope_deltas,
        )
        if (
            runtime_prefix_len + target_suffix_len > cache.max_seq_len
            or logical_prefix_len + target_suffix_len
            > self._rpu_rope_max_seq_len
        ):
            raise RuntimeError(
                "RhinoVLA fused: prefix plus padded suffix exceeds installed "
                f"capacity (physical={runtime_prefix_len}, logical="
                f"{logical_prefix_len}, suffix={target_suffix_len}, "
                f"cache={cache.max_seq_len})"
            )
        torch.ops.rpu.rhino_vla_set_rope_position(
            self._rpu_handle, logical_prefix_len
        )

        # -- hidden_states -> RPU fp16 --
        hidden_states = suffix_embeds
        if hidden_states.device.type != 'rpu':
            hidden_states = hidden_states.to(dtype=torch.float16, device='rpu')
        if hidden_states.dtype != torch.float16:
            hidden_states = hidden_states.to(torch.float16)

        # -- pad suffix to the precomputed target length (SDPA alignment) --
        # The M-RoPE cos/sin table + KV/SDPA geometry are baked at patch time for
        # target_suffix_len; a longer suffix reads the table out of bounds (silent
        # wrong output / illegal DDR access). Shorter is padded below; longer must
        # re-patch (patch_rhino_vla_for_rpu is keyed on suffix_len).
        if actual_suffix_len < target_suffix_len:
            pad_len = target_suffix_len - actual_suffix_len
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_len))
            if suffix_mask is not None:
                suffix_mask = torch.nn.functional.pad(suffix_mask, (0, pad_len), value=False)
        hidden_states = hidden_states.contiguous()

        # -- cond -> [hidden_size] fp16 RPU contiguous --
        hidden_size = self.config.width
        if adarms_cond is None:
            raise AssertionError("RhinoVLA fused: adarms_cond is required")
        cond_rpu = adarms_cond.to(dtype=torch.float16, device='rpu')
        if cond_rpu.dim() == 2 and cond_rpu.shape[0] == 1:
            cond_rpu = cond_rpu.squeeze(0)
        if cond_rpu.dim() != 1 or cond_rpu.shape[0] != hidden_size:
            raise AssertionError(
                f"RhinoVLA fused: cond must be [{hidden_size}] or [1,{hidden_size}], "
                f"got {tuple(cond_rpu.shape)}")
        cond_rpu = cond_rpu.contiguous()

        # -- prefix KV pre-fill (first step / after reset) --
        if prefix_key_values is not None and cache.position == 0:
            _prefill_kv_cache(
                cache, prefix_pairs, prefix_route_arguments
            )
            self._rpu_prefix_source_snapshot = prefix_source_snapshot
        elif prefix_key_values is None:
            # A shared-cache caller owns the prefix bytes. Do not later accept
            # an old direct-prefix identity against storage it may have replaced.
            self._rpu_prefix_source_snapshot = None
        # Phase2 shared-cache path preloads the expert RPUCache directly from
        # the text RPUCache and passes prefix_key_values=None. In that mode the
        # cache position itself is the prefix contract.
        has_prefix = prefix_key_values is not None or prefix_mask is not None
        prefix_len_actual = cache.position if has_prefix else 0

        # -- attention mask --
        if prefix_mask is not None and suffix_mask is not None:
            bool_mask = make_suffix_attn_mask(prefix_mask, suffix_mask)
            attention_mask = bool_mask_to_attention_bias(bool_mask, torch.float16)
            # Keep RhinoVLA SPM SDPA on v16-aligned K length; dummy cache lanes
            # are masked below and the C++ launcher reads the aligned length.
            pad_k = (-attention_mask.shape[-1]) % 16
            if pad_k:
                attention_mask = torch.nn.functional.pad(
                    attention_mask,
                    (0, pad_k),
                    value=torch.finfo(attention_mask.dtype).min,
                )
            attention_mask = attention_mask.cpu().contiguous()
            is_causal = False
        else:
            attention_mask = None
            is_causal = True

        action_plan = plan_rhino_action_execution(
            self,
            logical_len=actual_suffix_len,
            execution_len=target_suffix_len,
            prefix_len=prefix_len_actual,
            kv_len=(
                int(attention_mask.shape[-1])
                if attention_mask is not None
                else prefix_len_actual + target_suffix_len
            ),
            use_attention_mask=attention_mask is not None,
            is_causal=is_causal,
            prefix_plan_key_words=(
                prefix_plan.graph_key_words()
                if prefix_plan is not None else ()
            ),
        )
        selected_action_plan = action_plan.selected
        if selected_action_plan is None:
            raise RuntimeError("RhinoVLA action planner selected no plan")

        # The native handle owns the frozen selector snapshot bound at install;
        # the resulting COMPLETE descriptor digest is the graph-key authority.
        sig = _GraphSignature(
            op_id="rhino_vla_forward",
            shapes=[int(hidden_states.shape[1]), int(hidden_states.shape[2])],
            dyn_dims=[
                int(len(self.layers)),
                int(self.config.num_attention_heads),
                int(self.config.num_key_value_heads),
                int(self.config.head_dim),
                int(prefix_len_actual),
                int(logical_prefix_len),
                _validate_rhino_prefix_mask(
                    runtime_prefix_len=runtime_prefix_len,
                    batch_size=int(suffix_embeds.shape[0]),
                    prefix_mask=prefix_mask,
                ),
                int(is_causal),
                1 if attention_mask is not None else 0,
                *action_plan.graph_key_words(),
            ],
            dtypes=[torch.float16],
        )
        try:
            with self._rpu_graph_cache.capture(sig):
                output = torch.ops.rpu.rhino_vla_forward(
                    self._rpu_handle,
                    hidden_states,
                    cond_rpu,
                    cache.k_caches,
                    cache.v_caches,
                    self._rpu_cos_cached,
                    self._rpu_sin_cached,
                    attention_mask,
                    prefix_len_actual,
                    is_causal,
                    list(
                        selected_action_plan.stage_tuple.physical_descriptor
                    ),
                )
        finally:
            torch.ops.rpu.spm_alloc_reset_temporary()

        # -- strip suffix padding --
        if actual_suffix_len < target_suffix_len:
            output = output[:, :actual_suffix_len, :]

        # -- final norm in Python on CPU fp32 (D-402) --
        if getattr(self, "norm", None) is not None:
            norm_w = getattr(self.norm.norm, 'weight', None)
            norm_device = norm_w.device if norm_w is not None else torch.device('cpu')
            norm_dtype = norm_w.dtype if norm_w is not None else torch.float32
            output_for_norm = output.to(device=norm_device, dtype=norm_dtype)
            output, _ = self.norm(output_for_norm, adarms_cond)

        if orig_device.type != output.device.type:
            output = output.to(orig_device)

        publish_rhino_action_execution_receipt(
            self,
            action_plan,
            logical_len=actual_suffix_len,
            prefix_plan=prefix_plan,
            prefix_route_argument_digest=prefix_route_argument_digest,
        )

        return output

    handle = torch.ops.rpu.rhino_vla_create()
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource
    resource = _InstalledNativeResource(
        expert, handle, torch.ops.rpu.rhino_vla_destroy,
        graphs=(graph_cache,), keepalive=(set_weights_args, scale_lists, cond_owners, cos_cached, sin_cached, kv_cache),
        label="RhinoVLA action", handle_name="_rpu_handle")
    handle_finalizer = resource.finalizer
    try:
        torch.ops.rpu.rhino_vla_set_runtime_config(
            handle,
            native_cold.denoise_static_context_cache,
            native_cold.fused_adarms,
            native_cold.skip_adarms_gemv,
            native_cold.precompute_adarms,
            native_cold.precompute_time_proj,
            native_cold.fold_action_time_in,
            native_cold.gated_no_sub,
            native_cold.fused_silu_mul,
            native_cold.expert_fusions,
            native_cold.adarms_resident,
            native_cold.packed_qkv,
            native_cold.aligned_kv,
        )
        if full_w8a16:
            torch.ops.rpu.rhino_vla_set_weights_full_w8a16(
                handle, *set_weights_args, *scale_lists, *cond_owners["scales"])
        elif scale_lists:
            torch.ops.rpu.rhino_vla_set_weights_w8a16(handle, *set_weights_args, *scale_lists)
        else:
            torch.ops.rpu.rhino_vla_set_weights(handle, *set_weights_args)
        torch.ops.rpu.rhino_vla_set_chunk_envelope(handle, int(kv_cache.max_seq_len), 0)
        torch.ops.rpu.rhino_vla_set_rope_position(handle, int(prefix_len))

        expert._rpu_graph_cache = graph_cache
        expert._rpu_cos_cached = cos_cached
        expert._rpu_sin_cached = sin_cached
        expert._rpu_rope_max_seq_len = int(max_seq_len)
        expert._rpu_suffix_len = int(suffix_len)
        expert._rpu_kv_cache = kv_cache
        expert._rpu_prefix_source_snapshot = None
        expert._rpu_execution = {"action": {"chunk_size": "auto"}}
        expert._fmb_execution_component_id = "action_expert"
        expert._fmb_execution_generation = 0
        expert._rpu_native_cold_config = native_cold
        expert._rpu_full_w8a16 = bool(full_w8a16)
        expert._rpu_handle = handle
        expert._rpu_handle_finalizer = handle_finalizer
        expert._rpu_retirement_state = resource
        expert.forward = types.MethodType(
            rpu_rhino_vla_fused_forward, expert)

        if old_resource is not None:
            old_resource.retire()
            if old_resource.parent is not None:
                resource.take_ownership(old_resource.parent())
        elif had_old_handle and old_handle is not None:
            raise RuntimeError("RhinoVLA action replacement requires its actual retirement state")
    except BaseException as error:
        resource.cleanup_failure(error, expert, install_snapshot)
        expert_state = vars(expert)
        for name in _RHINO_RUNTIME_INSTALL_ATTRS:
            expert_state.pop(name, None)
        expert_state.update(install_snapshot)
        raise

    if old_finalizer is not None and getattr(old_finalizer, "alive", False):
        old_finalizer.detach()
    print(f"[Rhino] weights set, handle={handle}, layers={len(expert.layers)}")
    print(f"[Rhino] patched fused forward, handle={handle}, layers={len(expert.layers)}")
    return handle
