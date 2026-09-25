"""Qwen3.5 vision tower (== HF ``Qwen3_5VisionModel``) → rpu_backend.

Based on the Qwen3-VL vision adapter (``qwen3_vl/vision.py``), without DeepStack.
Qwen3.5's vision config carries ``deepstack_visual_indexes=[]``
and the HF ``Qwen3_5VisionModel`` deletes the deepstack members, so the encoder is
byte-for-byte the qwen3vl tower minus deepstack — profile-sized ViT (fused QKV +
2D RoPE + GELU MLP), output ``BaseModelOutputWithPooling(last_hidden_state,
pooler_output)`` (no ``deepstack_features``).

Wires ``Qwen3_5VisionModel`` into the C++ vision subsystem registered as
``torch.ops.rpu.qwen3_5_vision_*`` (see ``src/fused/rpu_qwen3_5_vision_model.cpp``).

Public surface:
  - ``install_qwen3_5_vision_for_rpu(model)`` — swizzle vision blocks, create the
    C++ handle on ``model.model.visual``, set weights + rope tables, replace the
    vision-tower forward. Generic 2B/4B image inference is numeric-blocked and
    requires an explicit controlled-evaluation opt-in. Idempotent; called lazily
    on the first image forward.
  - ``fuse_visual_embeds(model, hidden, input_ids, pixel_values, image_grid_thw, ...)``
    — vision→text fusion: run the tower and merger on RPU, DMA merged rows directly
    into ``hidden``, compute 3D M-RoPE ``position_ids``, return
    ``(hidden, position_ids[3, N])``.

Architecture decisions:
  - Patch embed + addition of an HF-exact position tensor run as in-graph STEP0
    by default. HF computes that tensor once per whole grid; the graph uploads
    it through a mutable-source DMA.
  - The patch merger runs in-graph and uses mutable-destination DMA for direct
    text fusion; standalone vision still returns a normal pooler tensor.
  - 2D RoPE: position_idx built per forward, copy_in to the model-owned keepalive
    ``[MAX_KEEPALIVE_SEQ, 2] int16``.
"""
from __future__ import annotations

import math
import os
import threading
import types
from collections.abc import Mapping
from numbers import Integral
from types import MappingProxyType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import rpu_backend
from rpu_backend.adapters.qwen3_5.text import _clear_qwen35_graphs, _oneshot_scope
from rpu_backend.runtime import rpu_env_bool
from rpu_backend.runtime.log import _LOG
from rpu_backend.runtime.weights import (
    tp_col_swizzle_mc_weight,
    tp_row_swizzle_mc_weight,
)
from rpu_backend.api.cache import RPUCache


QWEN3_5_VISION_ARCH = "qwen3_5_vision"
QWEN3_5_MOE_35B_VISION_PROFILE = "qwen3_5_moe_35b_tp8"
QWEN3_5_MOE_35B_TP4_VISION_PROFILE = "qwen3_5_moe_35b_multimodal_tp4"
QWEN3_5_MOE_35B_TP6_VISION_PROFILE = "qwen3_5_moe_35b_multimodal_tp6"
# Each value is (authoritative text topology, actual Vision execution cores).
# The text tuple is root/attention/MLP/head/physical-KV.  GDN/router/routing
# remain text-native profile axes and must be attested before this bridge state
# is published; Vision does not infer them from the five-axis ABI.
_QWEN3_5_MOE_35B_VISION_PROFILE_LAYOUTS = MappingProxyType({
    QWEN3_5_MOE_35B_VISION_PROFILE: ((8, 8, 8, 8, 8), 8),
    QWEN3_5_MOE_35B_TP4_VISION_PROFILE: ((4, 4, 4, 4, 8), 4),
    QWEN3_5_MOE_35B_TP6_VISION_PROFILE: ((6, 4, 6, 4, 8), 4),
})
_VISION_INSTALL_LOCK = threading.RLock()
_QWEN3_5_VISION_MODEL_TP = 8
_QWEN3_5_VISION_PROFILES = frozenset({
    (24, 16, 1024, 4096),
})
_QWEN3_5_VISION_CANDIDATE_PROFILES = frozenset({
    (12, 12, 768, 3072),
    (27, 16, 1152, 4304),
})

_VISION_PADDING_CAP = 64


def _qwen3_5_moe_vision_profile_layout(
    prepared_profile: str,
) -> tuple[tuple[int, int, int, int, int], int]:
    """Return the exact text topology and actual Vision owner count."""
    try:
        return _QWEN3_5_MOE_35B_VISION_PROFILE_LAYOUTS[prepared_profile]
    except (KeyError, TypeError) as exc:
        raise ValueError("unknown Qwen3.5 Vision prepared profile") from exc


def _qwen3_5_moe_vision_profile_cores(prepared_profile: str) -> int:
    """Return the actual Vision execution cores, independent of root cores."""
    return _qwen3_5_moe_vision_profile_layout(prepared_profile)[1]


def _qwen3_5_moe_vision_profile_root_cores(prepared_profile: str) -> int:
    """Return the text residual/root owner count for the prepared profile."""
    return _qwen3_5_moe_vision_profile_layout(prepared_profile)[0][0]


def _validate_qwen3_5_moe_text_topology(
    state, prepared_profile: str, topology
) -> int:
    """Attest the installed Python and complete native owners for Vision."""
    expected, vision_cores = _qwen3_5_moe_vision_profile_layout(
        prepared_profile
    )
    root_cores = expected[0]
    actual_python = tuple(
        getattr(topology, name, None)
        for name in (
            "num_cores",
            "attn_tp",
            "mlp_tp",
            "lm_head_tp",
            "physical_kv_cores",
        )
    )
    if actual_python != expected:
        raise RuntimeError(
            "Qwen3.5-MoE Vision text bridge requires prepared "
            f"root{root_cores}/Vision{vision_cores} topology {expected}, "
            f"got {actual_python}"
        )
    for name in (
        "qwen3_5_moe_get_execution_topology",
        "qwen3_5_moe_get_execution_topology_v2",
    ):
        if not callable(getattr(torch.ops.rpu, name, None)):
            raise RuntimeError(f"Qwen3.5-MoE binary lacks {name}")
    from rpu_backend.adapters.qwen3_5_moe.cores import (
        validate_native_topology,
    )

    # The legacy five-axis result remains the prepared-profile selector.  The
    # versioned result additionally attests GDN, router and every expert route.
    validate_native_topology(state.handle, topology)
    return root_cores


def _validate_qwen3_5_moe_vision_policies(state, prepared_profile: str):
    """Attest root and Vision graph policies against one four-arena plan."""
    from dataclasses import replace

    root_cores = _qwen3_5_moe_vision_profile_root_cores(prepared_profile)
    vision_cores = _qwen3_5_moe_vision_profile_cores(prepared_profile)
    root_policy = getattr(
        getattr(state, "graph_cache", None), "runtime_policy", None
    )
    if (
        not isinstance(root_policy, rpu_backend.graph.GraphRuntimePolicy)
        or root_policy.execution_core_count != root_cores
        or root_policy.graph_arena_count != 4
        or root_policy.qwen35_legacy_27b_sdk_budget
        or getattr(
            getattr(state, "prefill_graph", None), "runtime_policy", None
        )
        is not root_policy
    ):
        raise RuntimeError(
            f"Qwen3.5-MoE Vision requires its prepared root{root_cores} "
            "four-arena text policy"
        )

    vision_policy = getattr(state, "vision_graph_policy", None)
    expected_vision_policy = (
        root_policy
        if vision_cores == root_cores
        else replace(root_policy, execution_core_count=vision_cores)
    )
    if (
        not isinstance(vision_policy, rpu_backend.graph.GraphRuntimePolicy)
        or vision_policy.execution_core_count != vision_cores
        or vision_policy.graph_arena_count != 4
        or vision_policy.qwen35_legacy_27b_sdk_budget
        or (
            vision_policy is not root_policy
            if vision_cores == root_cores
            else vision_policy != expected_vision_policy
        )
    ):
        raise RuntimeError(
            f"Qwen3.5-MoE Vision requires its prepared Vision{vision_cores}/"
            f"root{root_cores} four-arena policy pair"
        )
    return vision_policy


class Qwen3_5MoeVisionTextBridge:
    """Verified MoE text owner used by the shared Vision implementation.

    The old MoE wrapper passed an arbitrary ``_mrope_setter`` keyword.  A
    shared ``**kw`` accepted that spelling without consuming it, so image
    prefill could silently leave the dense/MoE decode offset unchanged.  This
    bridge binds the exact installed state, top-level Session owner, native op
    and close callback.  Every use revalidates the mutable model tree before it
    touches the native handle.
    """

    __slots__ = (
        "model", "fusion", "state", "close_callback", "prepared_profile"
    )

    def __init__(self, *, model, fusion, state, close_callback):
        self.model = model
        self.fusion = fusion
        self.state = state
        self.close_callback = close_callback
        self.prepared_profile = getattr(state, "prepared_profile", None)

    def set_mrope_position_delta(self, delta: int) -> None:
        if isinstance(delta, bool) or not isinstance(delta, Integral):
            raise TypeError("Qwen3.5-MoE M-RoPE delta must be an integer")
        state = _validate_qwen3_5_moe_vision_text_bridge(self, self.model)
        setter = getattr(
            torch.ops.rpu, "qwen3_5_moe_set_mrope_position_delta", None
        )
        if setter is None:
            raise RuntimeError(
                "Qwen3.5-MoE binary lacks qwen3_5_moe_set_mrope_position_delta"
            )
        setter(state.handle, int(delta))


def _validate_qwen3_5_moe_vision_text_bridge(
    bridge: Qwen3_5MoeVisionTextBridge, model, *, require_vision_policy=False
):
    """Return the live state or fail before using a stale/wrong text handle."""
    if type(bridge) is not Qwen3_5MoeVisionTextBridge:
        raise TypeError(
            "Qwen3.5-MoE Vision requires a Qwen3_5MoeVisionTextBridge"
        )
    try:
        root_cores = _qwen3_5_moe_vision_profile_root_cores(
            bridge.prepared_profile
        )
        vision_cores = _qwen3_5_moe_vision_profile_cores(
            bridge.prepared_profile
        )
    except ValueError as exc:
        raise RuntimeError(
            "Qwen3.5-MoE Vision bridge profile identity changed"
        ) from exc
    if bridge.model is not model:
        raise RuntimeError("Qwen3.5-MoE Vision bridge top-level owner changed")
    fusion = getattr(model, "model", model)
    if bridge.fusion is not fusion:
        raise RuntimeError("Qwen3.5-MoE Vision bridge fusion owner changed")
    language = getattr(fusion, "language_model", fusion)
    state = getattr(language, "_rpu_qwen3_5_moe", None)
    if state is None or state is not bridge.state:
        raise RuntimeError("Qwen3.5-MoE Vision text state was removed or replaced")
    if getattr(state, "prepared_profile", None) != bridge.prepared_profile:
        raise RuntimeError(
            "Qwen3.5-MoE Vision text state lacks its prepared profile"
        )
    handle = getattr(state, "handle", None)
    if handle is None:
        raise RuntimeError("Qwen3.5-MoE Vision text state has no live handle")
    session = getattr(state, "_execution_session", None)
    if (
        session is None
        or getattr(session, "_owner", None) is not model
        or getattr(model, "_execution_session", None) is not session
    ):
        raise RuntimeError("Qwen3.5-MoE Vision text Session owner changed")
    if (
        type(getattr(state, "execution_generation", None)) is not int
        or state.execution_generation != getattr(session, "generation", None)
    ):
        raise RuntimeError(
            "Qwen3.5-MoE Vision text execution generation changed"
        )
    close = getattr(state, "_retirement_close", None)
    from rpu_backend.adapters.qwen3_5_moe import _close_qwen3_5_moe_model

    if (
        close is not _close_qwen3_5_moe_model
        or close is not bridge.close_callback
    ):
        raise RuntimeError("Qwen3.5-MoE Vision text close callback changed")
    from rpu_backend.adapters.qwen3_5.text import _qwen35_retirement_ready

    if not _qwen35_retirement_ready(state):
        raise RuntimeError(
            "Qwen3.5-MoE Vision text state is not retirement-ready"
        )
    resource = state.retirement
    if not any(
        child is language
        and recorded_state is state
        and recorded_resource is resource
        and recorded_session is session
        for child, recorded_state, recorded_resource, recorded_session in vars(
            model
        ).get("_qwen35_retirement_children", ())
    ):
        raise RuntimeError(
            "Qwen3.5-MoE Vision text retirement child identity changed"
        )
    topology = getattr(state, "execution_topology", None)
    installed_root_cores = _validate_qwen3_5_moe_text_topology(
        state, bridge.prepared_profile, topology
    )
    if installed_root_cores != root_cores:
        raise RuntimeError("Qwen3.5-MoE Vision bridge profile identity changed")
    _validate_qwen3_5_moe_vision_policies(state, bridge.prepared_profile)
    if (
        require_vision_policy
        and getattr(state, "vision_arena_admitted", None) is not True
    ):
        raise RuntimeError(
            f"Qwen3.5-MoE Vision requires its prepared Vision{vision_cores}/"
            f"root{root_cores} four-arena policy"
        )
    return state


def _validate_qwen3_5_moe_vision_retirement(
    vision_model, bridge: Qwen3_5MoeVisionTextBridge, text_state
):
    """Prove an already-installed Vision handle still belongs to this owner."""
    vision_state = getattr(vision_model, "_rpu_vision_retirement_state", None)
    from rpu_backend.adapters.qwen3_5.text import _qwen35_retirement_ready

    if vision_state is None or not _qwen35_retirement_ready(vision_state):
        raise RuntimeError(
            "Qwen3.5-MoE Vision installed state is not retirement-ready"
        )
    session = text_state._execution_session
    resource = getattr(vision_state, "retirement", None)
    owned = vars(bridge.model).get("_qwen35_retirement_children", ())
    if (
        getattr(vision_state, "_execution_session", None) is not session
        or getattr(vision_state, "_retirement_close", None)
        is not bridge.close_callback
        or resource is None
        or resource.parent is None
        or resource.parent() is not bridge.model
        or not any(
            child is vision_model
            and state is vision_state
            and item is resource
            and original_session is session
            for child, state, item, original_session in owned
        )
    ):
        raise RuntimeError(
            "Qwen3.5-MoE Vision installed state belongs to a different owner"
        )
    return vision_state


def make_qwen3_5_moe_vision_text_bridge(model) -> Qwen3_5MoeVisionTextBridge:
    """Bind shared Vision to the installed 35B-A3B MoE text owner."""
    fusion = getattr(model, "model", model)
    language = getattr(fusion, "language_model", fusion)
    state = getattr(language, "_rpu_qwen3_5_moe", None)
    if state is None:
        raise RuntimeError("Qwen3.5-MoE text must be installed before Vision")
    bridge = Qwen3_5MoeVisionTextBridge(
        model=model,
        fusion=fusion,
        state=state,
        close_callback=getattr(state, "_retirement_close", None),
    )
    _validate_qwen3_5_moe_vision_text_bridge(bridge, model)
    return bridge


def _validate_qwen3_5_moe_35b_vision_profile(
    vision_model,
    config,
    num_cores: int,
    prepared_profile: str = QWEN3_5_MOE_35B_VISION_PROFILE,
) -> None:
    """Admit the exact 35B-A3B FP16 tower on its prepared owner count."""
    expected_cores = _qwen3_5_moe_vision_profile_cores(prepared_profile)
    geometry = tuple(
        getattr(config, name, None)
        for name in (
            "depth",
            "num_heads",
            "hidden_size",
            "intermediate_size",
            "out_hidden_size",
            "in_channels",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
            "num_position_embeddings",
        )
    )
    expected = (27, 16, 1152, 4304, 2048, 3, 16, 2, 2, 2304)
    valid = (
        type(num_cores) is int
        and num_cores == expected_cores
        and geometry == expected
        and len(getattr(vision_model, "blocks", ())) == 27
        and getattr(config, "model_type", None) == "qwen3_5_moe"
        and getattr(config, "hidden_act", None) == "gelu_pytorch_tanh"
        and not getattr(config, "deepstack_visual_indexes", ())
        and not any(
            getattr(config, name, None)
            for name in ("quant_config", "quantization_config")
        )
    )
    if not valid:
        raise ValueError(
            f"Qwen3.5-MoE Vision requires exact TP{expected_cores} 35B-A3B "
            "L27/H1152/I4304/out2048 FP16 geometry for prepared profile "
            f"{prepared_profile!r}"
        )
    if (
        _qwen3_5_vision_padded_head_dim(1152 // 16) != 80
        or _qwen3_5_vision_padded_intermediate_size(4304) != 4352
    ):
        raise RuntimeError("Qwen3.5-MoE Vision cold padding contract changed")


def _validate_qwen3_5_moe_35b_vision_weights(
    vision_model,
    config,
    num_cores: int = 8,
    prepared_profile: str = QWEN3_5_MOE_35B_VISION_PROFILE,
) -> None:
    """Validate raw Vision shapes before any irreversible TP swizzle."""
    _validate_qwen3_5_moe_35b_vision_profile(
        vision_model, config, num_cores, prepared_profile
    )

    def shape(owner, name, expected):
        value = getattr(owner, name, None)
        if value is None or tuple(value.shape) != expected:
            raise ValueError(
                f"Qwen3.5-MoE Vision weight {name} requires shape {expected}"
            )

    for block in vision_model.blocks:
        for owner, out_size, in_size in (
            (block.attn.qkv, 3456, 1152),
            (block.attn.proj, 1152, 1152),
            (block.mlp.linear_fc1, 4304, 1152),
            (block.mlp.linear_fc2, 1152, 4304),
        ):
            shape(owner, "weight", (out_size, in_size))
            shape(owner, "bias", (out_size,))
        for norm in (block.norm1, block.norm2):
            shape(norm, "weight", (1152,))
            shape(norm, "bias", (1152,))
    shape(vision_model.patch_embed.proj, "weight", (1152, 3, 2, 16, 16))
    shape(vision_model.patch_embed.proj, "bias", (1152,))
    shape(vision_model.pos_embed, "weight", (2304, 1152))
    shape(vision_model.merger.norm, "weight", (1152,))
    shape(vision_model.merger.norm, "bias", (1152,))
    for owner, out_size in (
        (vision_model.merger.linear_fc1, 4608),
        (vision_model.merger.linear_fc2, 2048),
    ):
        shape(owner, "weight", (out_size, 4608))
        shape(owner, "bias", (out_size,))
    from rpu_backend.adapters.qwen3_5.cores import validate_cold_weight_dtype

    validate_cold_weight_dtype(vision_model, fp16=True)


def _resolve_vision_install_options(
    *, execution_config=None, resolved_options=None, cold_snapshot=None
) -> tuple[int, int | None, str | int, int, Mapping[str, object]]:
    """Resolve vision controls once, before handle creation or weight mutation.

    A canonical ``vision.chunk_size`` is an exact planner constraint.  The old
    ``QWEN3_5_VISION_CHUNK`` variable remains a cap alias only when the
    canonical field is absent; canonical ``auto`` therefore clears a stale env
    cap just like the text path.
    """
    if resolved_options is not None:
        if not isinstance(resolved_options, tuple) or len(resolved_options) != 5:
            raise TypeError("resolved_options must be a five-item tuple")
        return (
            int(resolved_options[0]),
            resolved_options[1],
            resolved_options[2],
            int(resolved_options[3]),
            resolved_options[4],
        )
    if execution_config is None:
        execution_config = {}
    if not isinstance(execution_config, Mapping):
        raise TypeError(
            "Qwen3.5 vision execution_config must be a mapping, got "
            f"{type(execution_config).__name__}"
        )
    vision = execution_config.get("vision", {})
    if not isinstance(vision, Mapping):
        raise TypeError("Qwen3.5 execution_config['vision'] must be a mapping")

    if cold_snapshot is not None and not isinstance(cold_snapshot, Mapping):
        raise TypeError("cold_snapshot must be a mapping")
    if cold_snapshot is None:
        env_present = "QWEN3_5_VISION_CHUNK" in os.environ
        env_raw = os.environ.get("QWEN3_5_VISION_CHUNK", "0")
    else:
        env_present = bool(cold_snapshot.get("legacy_env_present", False))
        env_raw = str(cold_snapshot.get("legacy_env_raw", "0"))
    try:
        env_chunk = int(env_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("QWEN3_5_VISION_CHUNK must be an integer") from exc
    if env_chunk < 0 or (env_chunk and env_chunk % 16):
        raise ValueError(
            "QWEN3_5_VISION_CHUNK must be 0 or a positive multiple of 16"
        )

    canonical_present = "chunk_size" in vision
    canonical = vision.get("chunk_size", "auto")
    exact: int | None = None
    if canonical_present:
        if canonical == "auto":
            cap = 0
        elif (
            isinstance(canonical, Integral)
            and not isinstance(canonical, bool)
            and int(canonical) > 0
            and int(canonical) % 16 == 0
        ):
            exact = int(canonical)
            if env_chunk > 0 and env_chunk != exact:
                raise ValueError(
                    "QWEN3_5_VISION_CHUNK conflicts with canonical "
                    f"rpu_execution vision chunk_size: {env_chunk} != {exact}"
                )
            cap = exact
        else:
            raise ValueError(
                "Qwen3.5 vision chunk_size must be 'auto' or a positive "
                f"multiple of 16, got {canonical!r}"
            )
    else:
        cap = env_chunk

    if "padding_rows" in vision and "padding_budget" in vision:
        raise ValueError(
            "Qwen3.5 vision padding_rows conflicts with padding_budget"
        )
    raw_rows = vision.get("padding_rows", "auto")
    if raw_rows != "auto" and (
        isinstance(raw_rows, bool)
        or not isinstance(raw_rows, Integral)
        or int(raw_rows) < 0
    ):
        raise ValueError(
            "Qwen3.5 vision padding_rows must be 'auto' or a non-negative integer"
        )
    padding_rows: str | int = (
        "auto" if raw_rows == "auto" else int(raw_rows)
    )
    raw_budget = vision.get("padding_budget", 0)
    if (
        isinstance(raw_budget, bool)
        or not isinstance(raw_budget, Integral)
        or int(raw_budget) < 0
        or int(raw_budget) > _VISION_PADDING_CAP
    ):
        raise ValueError(
            "Qwen3.5 vision padding_budget must be in 0.."
            f"{_VISION_PADDING_CAP}"
        )
    budget = int(raw_budget)
    if (padding_rows != "auto" and padding_rows != 0) or budget:
        raise ValueError(
            "Qwen3.5 vision padding is unsupported because bidirectional "
            "attention has no padding-mask route"
        )
    snapshot = MappingProxyType({
        "chunk_source": (
            "canonical" if canonical_present
            else "legacy_env" if env_present else "default"
        ),
        "canonical_chunk_present": canonical_present,
        "canonical_chunk": canonical,
        "legacy_env_present": env_present,
        "legacy_env_raw": env_raw,
        "chunk_effective_cap": cap,
        "padding_rows": padding_rows,
        "padding_budget": budget,
    })
    return cap, exact, padding_rows, budget, snapshot


def _resolve_qwen3_5_vision_attention_tp(num_heads: int, num_cores: int = 8) -> int:
    """Resolve the physical MHA core domain while model/MLP TP stays at 8."""
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError(
            f"Qwen3.5 vision num_heads must be a positive integer, got {num_heads!r}"
        )
    if type(num_cores) is not int or num_cores not in (4, 8):
        raise ValueError("Qwen3.5 Vision num_cores must be 4 or 8")
    return math.gcd(num_heads, num_cores)


def _qwen3_5_vision_padded_head_dim(head_dim: int) -> int:
    """RPU attention layout width; RoPE/softmax still use the real head dim."""
    if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError(
            f"Qwen3.5 vision head_dim must be a positive integer, got {head_dim!r}"
        )
    return ((head_dim + 15) // 16) * 16


def _qwen3_5_vision_padded_intermediate_size(intermediate_size: int) -> int:
    """Apply the fp16 vision MLP alignment (model TP * 32)."""
    if (
        isinstance(intermediate_size, bool)
        or not isinstance(intermediate_size, int)
        or intermediate_size <= 0
    ):
        raise ValueError(
            "Qwen3.5 vision intermediate_size must be a positive integer, got "
            f"{intermediate_size!r}"
        )
    alignment = _QWEN3_5_VISION_MODEL_TP * 32
    return ((intermediate_size + alignment - 1) // alignment) * alignment


def build_vision_rope_tables(
    head_dim: int,
    max_hw: int,
    *,
    device: str | torch.device = "rpu",
    dtype: torch.dtype = torch.float16,
    theta: float = 10000.0,
    inv_freq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FreqCos / FreqSin ``[max_hw, head_dim/4]`` fp16 tables for the
    ``rope_2d_spm`` kernel.

    Mirrors ``Qwen3_5VisionRotaryEmbedding(head_dim // 2).forward(max_hw)``
    followed by element-wise cos/sin. Pass the model's ``inv_freq`` buffer when
    available: loading the model in fp16 rounds that buffer, and HF then performs
    the position outer-product in that same dtype. Upcasting it before the
    multiply changes many table entries. The kernel uses the same table for BOTH
    row and col axes — it indexes once per axis via the per-token position_idx
    int16 pair. ``head_dim`` is the REAL attention head dim (64 for 0.8B/2B/4B,
    72 for 9B), not the v16-padded layout width.
    """
    if head_dim <= 0 or (head_dim % 4) != 0:
        raise ValueError(
            f"build_vision_rope_tables: head_dim must be positive multiple of 4, "
            f"got {head_dim}"
        )
    if max_hw <= 0:
        raise ValueError(f"build_vision_rope_tables: max_hw must be positive, got {max_hw}")

    half_axis = head_dim // 4
    if inv_freq is None:
        inv_freq_cpu = 1.0 / (
            theta ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float64) / (head_dim // 2))
        )
    else:
        inv_freq_cpu = inv_freq.detach().to(device="cpu").flatten()
        if inv_freq_cpu.numel() != half_axis:
            raise ValueError(
                f"build_vision_rope_tables: inv_freq must have {half_axis} elements, "
                f"got {inv_freq_cpu.numel()}"
            )
    positions = torch.arange(max_hw, dtype=inv_freq_cpu.dtype)
    freqs = positions[:, None] * inv_freq_cpu[None, :]  # [max_hw, head_dim/4]
    expected_shape = (max_hw, half_axis)
    if tuple(freqs.shape) != expected_shape:
        raise RuntimeError(
            "build_vision_rope_tables produced an invalid frequency shape: "
            f"expected {expected_shape}, got {tuple(freqs.shape)}"
        )

    cos = freqs.cos().to(dtype=dtype).contiguous()
    sin = freqs.sin().to(dtype=dtype).contiguous()
    return cos.to(device=device), sin.to(device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Weight conversion: split fused QKV, swizzle per partition
# ─────────────────────────────────────────────────────────────────────────────

def _convert_vision_block_weights_for_rpu(
    block, num_heads: int, hidden_size: int, *, num_cores: int = 8
) -> None:
    """In-place swizzle of a single Qwen3_5VisionBlock's weights.

    Splits the fused HF ``attn.qkv`` Linear ``[3*dim, dim]`` into three Linears
    q/k/v ``[dim, dim]`` (stashed as ``_rpu_q_w``, ``_rpu_q_b``, etc. on the block),
    then col-swizzles each for the resolved attention core domain. o_proj uses
    the same domain; fc1/fc2 stay on the 8-core model domain. Idempotent via
    ``_rpu_qwen3_5_vision_weights_converted`` marker.
    """
    _resolve_qwen3_5_vision_attention_tp(num_heads, num_cores)
    if getattr(block, "_rpu_qwen3_5_vision_weights_converted", False):
        if getattr(block, "_rpu_vision_num_cores", 8) != num_cores:
            raise RuntimeError("Qwen3.5 Vision weights were swizzled for a different core count")
        return
    if getattr(block, "_rpu_qwen3_5_vision_conversion_started", False):
        raise RuntimeError(
            "Qwen3.5 vision weight conversion previously failed after mutation; "
            "reload the model before retrying."
        )

    with torch.no_grad():
        attention_tp = _resolve_qwen3_5_vision_attention_tp(num_heads, num_cores)
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Qwen3.5 vision hidden_size={hidden_size} must be divisible by "
                f"num_heads={num_heads}"
            )
        head_dim = hidden_size // num_heads
        padded_head_dim = _qwen3_5_vision_padded_head_dim(head_dim)
        padded_attention_size = num_heads * padded_head_dim

        qkv_w = block.attn.qkv.weight.data  # [3*hidden, hidden]
        qkv_b = block.attn.qkv.bias.data    # [3*hidden]
        expected_weight = (3 * hidden_size, hidden_size)
        if tuple(qkv_w.shape) != expected_weight:
            raise ValueError(
                "Qwen3.5 vision fused QKV weight must have shape "
                f"{expected_weight}, got {tuple(qkv_w.shape)}"
            )
        expected_bias = (3 * hidden_size,)
        if tuple(qkv_b.shape) != expected_bias:
            raise ValueError(
                "Qwen3.5 vision fused QKV bias must have shape "
                f"{expected_bias}, got {tuple(qkv_b.shape)}"
            )

        # Validate every original HF shape before publishing the poison marker.
        o_w = block.attn.proj.weight.data
        o_b = block.attn.proj.bias.data
        fc1_w = block.mlp.linear_fc1.weight.data
        fc1_b = block.mlp.linear_fc1.bias.data
        fc2_w = block.mlp.linear_fc2.weight.data
        fc2_b = block.mlp.linear_fc2.bias.data
        intermediate_size = fc1_w.shape[0]
        original_shapes = {
            "attention output weight": (tuple(o_w.shape), (hidden_size, hidden_size)),
            "attention output bias": (tuple(o_b.shape), (hidden_size,)),
            "MLP fc1 weight": (tuple(fc1_w.shape), (intermediate_size, hidden_size)),
            "MLP fc1 bias": (tuple(fc1_b.shape), (intermediate_size,)),
            "MLP fc2 weight": (tuple(fc2_w.shape), (hidden_size, intermediate_size)),
            "MLP fc2 bias": (tuple(fc2_b.shape), (hidden_size,)),
        }
        for name, (actual, expected) in original_shapes.items():
            if actual != expected:
                raise ValueError(
                    f"Qwen3.5 vision {name} must have shape {expected}, got {actual}"
                )

        # Everything below is irreversible. Keep a poison marker until the
        # complete block is published so a failed partial assignment can never
        # be followed by a double swizzle.
        block._rpu_qwen3_5_vision_conversion_started = True

        q_w = qkv_w[0:hidden_size].contiguous()
        k_w = qkv_w[hidden_size : 2 * hidden_size].contiguous()
        v_w = qkv_w[2 * hidden_size : 3 * hidden_size].contiguous()
        q_b = qkv_b[0:hidden_size].contiguous()
        k_b = qkv_b[hidden_size : 2 * hidden_size].contiguous()
        v_b = qkv_b[2 * hidden_size : 3 * hidden_size].contiguous()

        # Pad each head independently. Padding only the fused tensor tail would
        # shift head boundaries and make RoPE/SDPA interpret real lanes as pad.
        if padded_head_dim != head_dim:
            head_pad = padded_head_dim - head_dim

            def pad_qkv_output(weight, bias):
                weight = F.pad(
                    weight.view(num_heads, head_dim, hidden_size),
                    (0, 0, 0, head_pad),
                ).reshape(padded_attention_size, hidden_size)
                bias = F.pad(
                    bias.view(num_heads, head_dim), (0, head_pad)
                ).reshape(padded_attention_size)
                return weight.contiguous(), bias.contiguous()

            q_w, q_b = pad_qkv_output(q_w, q_b)
            k_w, k_b = pad_qkv_output(k_w, k_b)
            v_w, v_b = pad_qkv_output(v_w, v_b)

        q_w = tp_col_swizzle_mc_weight(q_w, num_cores=attention_tp)
        k_w = tp_col_swizzle_mc_weight(k_w, num_cores=attention_tp)
        v_w = tp_col_swizzle_mc_weight(v_w, num_cores=attention_tp)

        block._rpu_q_w, block._rpu_k_w, block._rpu_v_w = q_w, k_w, v_w
        block._rpu_q_b, block._rpu_k_b, block._rpu_v_b = q_b, k_b, v_b

        o_w = o_w.contiguous()  # [hidden, hidden]
        if padded_head_dim != head_dim:
            o_w = F.pad(
                o_w.view(hidden_size, num_heads, head_dim),
                (0, padded_head_dim - head_dim),
            ).reshape(hidden_size, padded_attention_size).contiguous()
        o_w = tp_row_swizzle_mc_weight(o_w, num_cores=attention_tp)
        block.attn.proj.weight = nn.Parameter(o_w, requires_grad=False)
        block._rpu_o_b = o_b.contiguous()

        padded_intermediate_size = _qwen3_5_vision_padded_intermediate_size(
            intermediate_size
        )
        intermediate_pad = padded_intermediate_size - intermediate_size
        fc1_w = fc1_w.contiguous()  # [intermediate, hidden]
        fc1_b = fc1_b.contiguous()
        fc2_w = fc2_w.contiguous()  # [hidden, intermediate]
        if intermediate_pad:
            fc1_w = F.pad(fc1_w, (0, 0, 0, intermediate_pad)).contiguous()
            fc1_b = F.pad(fc1_b, (0, intermediate_pad)).contiguous()
            fc2_w = F.pad(fc2_w, (0, intermediate_pad)).contiguous()
        fc1_w = tp_col_swizzle_mc_weight(fc1_w, **({"num_cores": num_cores} if num_cores != 8 else {}))
        block.mlp.linear_fc1.weight = nn.Parameter(fc1_w, requires_grad=False)
        block._rpu_fc1_b = fc1_b

        fc2_w = tp_row_swizzle_mc_weight(fc2_w, **({"num_cores": num_cores} if num_cores != 8 else {}))
        block.mlp.linear_fc2.weight = nn.Parameter(fc2_w, requires_grad=False)
        block._rpu_fc2_b = fc2_b.contiguous()

    block._rpu_vision_num_cores = num_cores
    block._rpu_qwen3_5_vision_weights_converted = True
    del block._rpu_qwen3_5_vision_conversion_started


def _fold_conv3d_to_linear_weight(conv3d_weight: torch.Tensor) -> torch.Tensor:
    """``[embed_dim, cin, tp, ps, ps]`` Conv3d weight → ``[embed_dim, cin*tp*ps*ps]``
    Linear weight.

    HF Qwen3_5VisionPatchEmbed processes flattened input ``[N, cin*tp*ps*ps]`` by
    ``view(-1, cin, tp, ps, ps)`` then Conv3d with kernel=stride=[tp, ps, ps]. This
    is mathematically equivalent to a Linear ``[embed_dim, cin*tp*ps*ps]`` because
    the reshape preserves memory order (contiguous strides match).
    """
    if conv3d_weight.dim() != 5:
        raise ValueError(
            f"_fold_conv3d_to_linear_weight: expected 5D weight, got {conv3d_weight.dim()}D"
        )
    embed_dim, cin, tp, ps_h, ps_w = conv3d_weight.shape
    return conv3d_weight.reshape(embed_dim, cin * tp * ps_h * ps_w).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Per-forward CPU compute: position_idx
# ─────────────────────────────────────────────────────────────────────────────

def _compute_vision_position_idx_cpu(
    grid_thw_cpu: torch.Tensor,
    spatial_merge_size: int,
    max_hw: int | None = None,
) -> torch.Tensor:
    """Build ``[num_patches, 2]`` int16 (row_idx, col_idx) on CPU from grid_thw.

    Mirrors HF ``Qwen3_5VisionModel.rot_pos_emb`` body — same ``coords`` layout
    (row, col stacked along the trailing dim of size 2). The RPU rope_2d kernel
    does the ``freq_table[pos_ids]`` lookup internally via position_idx +
    FreqCos/Sin table indexing. ``max_hw`` rejects indices outside that table.
    """
    grid_thw_list, merge_size = _validate_vision_grid(
        grid_thw_cpu, spatial_merge_size, "_compute_vision_position_idx_cpu")

    total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.int64)

    offset = 0
    for _, height, width in grid_thw_list:
        if max_hw is not None and (height > max_hw or width > max_hw):
            raise ValueError(
                f"_compute_vision_position_idx_cpu: image grid HxW={height}x{width} "
                f"exceeds 2D RoPE table max_hw={max_hw}"
            )
        merged_h = height // merge_size
        merged_w = width // merge_size

        block_rows = torch.arange(merged_h)
        block_cols = torch.arange(merged_w)
        intra_row = torch.arange(merge_size)
        intra_col = torch.arange(merge_size)

        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    if pos_ids.max().item() >= 2**15:
        raise ValueError(
            f"_compute_vision_position_idx_cpu: max idx {pos_ids.max().item()} "
            f"exceeds int16 range"
        )
    return pos_ids.to(torch.int16).contiguous()


def _cached_vision_position_idx_cpu(
    vision_model,
    grid_thw_cpu: torch.Tensor,
    spatial_merge_size: int,
    max_hw: int | None = None,
) -> torch.Tensor:
    """Keep one exact, shape-only position layout per vision instance."""
    cache_key = (int(spatial_merge_size), None if max_hw is None else int(max_hw))
    cached_grid = getattr(vision_model, "_rpu_vision_position_idx_grid", None)
    if (
        getattr(vision_model, "_rpu_vision_position_idx_cache_key", None) != cache_key
        or cached_grid is None
        or not torch.equal(cached_grid, grid_thw_cpu)
    ):
        vision_model._rpu_vision_position_idx_cpu = _compute_vision_position_idx_cpu(
            grid_thw_cpu,
            spatial_merge_size,
            max_hw=max_hw,
        )
        vision_model._rpu_vision_position_idx_grid = grid_thw_cpu.clone()
        vision_model._rpu_vision_position_idx_cache_key = cache_key
    return vision_model._rpu_vision_position_idx_cpu


def _validate_vision_grid(grid_thw_cpu, spatial_merge_size, caller):
    """Return a non-empty positive grid compatible with spatial merging."""
    if grid_thw_cpu.dim() != 2 or grid_thw_cpu.size(-1) != 3:
        raise ValueError(
            f"{caller}: grid_thw must be [n, 3], got {tuple(grid_thw_cpu.shape)}")
    if grid_thw_cpu.size(0) == 0:
        raise ValueError(f"{caller}: grid_thw must contain at least one image")
    merge_size = int(spatial_merge_size)
    if merge_size <= 0:
        raise ValueError(f"{caller}: spatial_merge_size must be positive")
    grid = [[int(x) for x in row] for row in grid_thw_cpu.tolist()]
    for t, h, w in grid:
        if min(t, h, w) <= 0:
            raise ValueError(f"{caller}: T/H/W must be positive, got {(t, h, w)}")
        if t != 1:
            raise ValueError(f"{caller}: video grids are not supported, got T={t}")
        if h % merge_size or w % merge_size:
            raise ValueError(
                f"{caller}: H/W must be divisible by spatial_merge_size="
                f"{merge_size}, got {(t, h, w)}")
    return grid, merge_size


def _visual_run_starts(input_ids_cpu, image_token_id, split_sizes):
    """Return one contiguous image-token start per image, or ``None`` if fragmented."""
    if input_ids_cpu.dim() != 2 or input_ids_cpu.size(0) != 1:
        raise ValueError("Qwen3.5 vision fusion supports batch_size == 1 only")
    positions = (input_ids_cpu[0] == image_token_id).nonzero(as_tuple=True)[0].tolist()
    if len(positions) != sum(split_sizes):
        raise RuntimeError(
            f"Qwen3.5 vision fuse: image token count {len(positions)} "
            f"!= image embeds rows {sum(split_sizes)}")
    starts = []
    offset = 0
    for size in split_sizes:
        run = positions[offset:offset + size]
        if not run or run != list(range(run[0], run[0] + size)):
            return None
        starts.append(run[0])
        offset += size
    return starts


def _qwen3_5_vision_destroy_handle(h: int) -> None:
    try:
        from rpu_backend.api import causal_lm

        with causal_lm._LIVE_LOCK:
            if causal_lm._LIVE_TERMINAL_REASON is not None:
                return  # Unsafe process: GC must not retry explicit retirement.
        torch.ops.rpu.qwen3_5_vision_destroy(h)
    except Exception:
        pass


def _make_dummy_vision_kv_caches(
    num_layers: int,
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
    device: str = "rpu",
    *, num_cores: int = 8,
) -> RPUCache:
    """Create an RPUCache sized for bidirectional vision SDPA.

    Vision encoder always inserts at position=0 and reads all tokens once — no
    autoregressive growth. Size the cache to ``max_seq_len`` (worst-case
    num_patches) so any forward shape fits. ``reset_to_position(0)`` per forward.
    """
    return RPUCache(
        num_layers=num_layers,
        batch_size=1,
        max_seq_len=max_seq_len,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        device=device,
        attn_tp=_resolve_qwen3_5_vision_attention_tp(num_heads, num_cores),
    )


def _resolve_vision_model(model_or_vision):
    """Accept either the top ``Qwen3_5ForConditionalGeneration`` / ``Qwen3_5Model``
    or the ``Qwen3_5VisionModel`` directly and return the vision tower."""
    if hasattr(model_or_vision, "blocks") and hasattr(model_or_vision, "merger"):
        return model_or_vision                       # already the vision tower
    inner = getattr(model_or_vision, "model", model_or_vision)  # ForCondGen → Qwen3_5Model
    return inner.visual


def _decode_qwen3_5_vision_plan_descriptor(
    raw_descriptor,
    *,
    logical_len,
    temporal_num_frames,
    camera_batch_count,
    compact_input,
):
    from rpu_backend.runtime.execution_planner import (
        GRAPH_RETAINED_CACHE,
        StageTuple,
        _decode_native_stage_domain,
    )

    raw_descriptor = tuple(raw_descriptor)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in raw_descriptor
    ):
        raise RuntimeError(
            "Qwen3.5 vision planner descriptor must contain only integers"
        )
    descriptor = tuple(int(value) for value in raw_descriptor)
    if len(descriptor) < 17 or descriptor[0] != 3:
        raise RuntimeError("Qwen3.5 vision planner returned an invalid descriptor")
    span_count = descriptor[11]
    stage_offset = 17 + 2 * span_count
    if span_count <= 0 or stage_offset >= len(descriptor):
        raise RuntimeError(
            "Qwen3.5 vision planner descriptor geometry is inconsistent"
        )
    stage_descriptor = descriptor[stage_offset:]
    stage_rows = _decode_native_stage_domain(
        (1, 1, len(stage_descriptor), *stage_descriptor),
        execution_len=int(logical_len),
        logical_len=int(logical_len),
        position=0,
        graph_mode=GRAPH_RETAINED_CACHE,
    )
    if len(stage_rows) != 1:
        raise RuntimeError(
            "Qwen3.5 vision planner returned no native stage descriptor"
        )
    stage = stage_rows[0]
    stage_metadata = dict(stage.physical_metadata)
    stage_fingerprint = int(stage_metadata["stage_plan_fingerprint"])
    if (
        descriptor[1] != int(logical_len)
        or span_count != int(camera_batch_count)
        or stage.as_tuple() != tuple(descriptor[2:5])
        or stage_metadata.get("span_count") != span_count
        or stage_metadata.get("manifest_state") != 1
        or descriptor[14] != int(temporal_num_frames)
        or descriptor[15] != int(camera_batch_count)
        or descriptor[16] != int(bool(compact_input))
        or descriptor[12] not in (0, 1)
        or descriptor[13] not in (0, 1)
        or not 0 <= descriptor[7] <= 0xFFFFFFFF
        or not 0 <= descriptor[8] <= 0xFFFFFFFF
        or (descriptor[7] == 0 and descriptor[8] == 0)
        or ((descriptor[7] << 32) | descriptor[8]) != stage_fingerprint
        or not 0 <= descriptor[9] <= 0xFFFFFFFF
        or not 0 <= descriptor[10] <= 0xFFFFFFFF
        or (descriptor[9] == 0 and descriptor[10] == 0)
    ):
        raise RuntimeError(
            "Qwen3.5 vision planner descriptor geometry is inconsistent"
        )
    if (
        descriptor[13] == 1
        and (descriptor[5] <= 0 or descriptor[6] <= 0)
    ) or (
        descriptor[13] == 0
        and (descriptor[5] != 0 or descriptor[6] != 0)
    ):
        raise RuntimeError(
            "Qwen3.5 vision planner descriptor merger schedule is inconsistent"
        )

    stage_descriptor_version = stage_metadata.pop("descriptor_version")
    stage_metadata.pop("span_count")
    metadata = [
        ("descriptor_version", descriptor[0]),
        ("merger_chunk", descriptor[5]),
        ("merger_chunks_per_span", descriptor[6]),
        ("stage_plan_fingerprint_hi", descriptor[7]),
        ("stage_plan_fingerprint_lo", descriptor[8]),
        ("layout_hash_hi", descriptor[9]),
        ("layout_hash_lo", descriptor[10]),
        ("span_count", span_count),
        ("has_step0", descriptor[12]),
        ("has_merger", descriptor[13]),
        ("temporal_num_frames", descriptor[14]),
        ("camera_batch_count", descriptor[15]),
        ("compact_input", descriptor[16]),
        ("stage_descriptor_version", stage_descriptor_version),
        ("stage_descriptor_words", len(stage_descriptor)),
        *stage_metadata.items(),
    ]
    # The shared decoder validated this native schedule; the custom prefix must
    # describe those same spans, not merely another contiguous partition.
    native_span_offset = 15 + 4 * sum(stage_descriptor[11:14])
    next_offset = 0
    for ordinal in range(span_count):
        offset = descriptor[17 + 2 * ordinal]
        length = descriptor[18 + 2 * ordinal]
        if offset != next_offset or length <= 0:
            raise RuntimeError(
                "Qwen3.5 vision planner descriptor spans are not contiguous"
            )
        native_span = native_span_offset + 3 * ordinal
        if (offset, length) != stage_descriptor[native_span:native_span + 2]:
            raise RuntimeError(
                "Qwen3.5 vision planner outer/inner descriptor spans disagree"
            )
        next_offset += length
        metadata.extend((
            (f"span_{ordinal}_offset", offset),
            (f"span_{ordinal}_len", length),
        ))
    if next_offset != int(logical_len):
        raise RuntimeError(
            "Qwen3.5 vision planner descriptor spans do not cover execution_len"
        )
    # Sort projected candidates using the same canonical metadata identity as
    # the shared planner, including when DDR/SPM alternatives share chunks.
    return StageTuple.from_value(StageTuple(
        *stage.as_tuple(), tuple(metadata), stage.physical_descriptor,
    ))


def _plan_qwen3_5_vision_group(
    vision_model,
    handle,
    logical_len,
    *,
    group_id,
    temporal_num_frames=1,
    camera_batch_count=1,
    compact_input=False,
):
    """Resolve one vision child through the shared A6 facade.

    Vision attention is bidirectional and has no safe attention-mask/padding
    route today, so its capability advertises an exact physical length.  A
    non-zero padding request is rejected before dispatch rather than silently
    changing the image attention graph.
    """
    from rpu_backend.runtime.execution_planner import PlannerRejectError, native_chunk_reject
    from rpu_backend.runtime.decoder import plan_bounded_prefill_execution

    exact = getattr(vision_model, "_rpu_vision_exact_chunk_size", None)
    padding_rows = getattr(vision_model, "_rpu_vision_padding_rows", "auto")
    padding_budget = int(getattr(vision_model, "_rpu_vision_padding_budget", 0))
    requested_padding = padding_budget if padding_rows == "auto" else int(padding_rows)
    if requested_padding:
        raise PlannerRejectError(
            "UNSUPPORTED", stage="VISION", requested=requested_padding,
            detail="vision bidirectional attention has no padding-mask route",
        )
    resolver = getattr(torch.ops.rpu, "qwen3_5_vision_resolve_prefill_domain", None)
    if resolver is None:
        raise RuntimeError(
            "Qwen3.5 vision binary lacks the canonical A6 domain resolver; "
            "rebuild rpu_backend before dispatch"
        )
    logical_len = int(logical_len)

    def resolve_domain(execution_len):
        try:
            descriptors = resolver(
                handle, execution_len, int(temporal_num_frames),
                int(camera_batch_count), bool(compact_input),
            )
        except RuntimeError as exc:
            reject = native_chunk_reject(
                exc, stage="VISION",
                requested=int(exact) if exact is not None else logical_len,
            )
            if reject is None:
                raise
            raise reject from exc
        rows = [
            _decode_qwen3_5_vision_plan_descriptor(
                descriptor, logical_len=execution_len,
                temporal_num_frames=temporal_num_frames,
                camera_batch_count=camera_batch_count,
                compact_input=compact_input,
            )
            for descriptor in descriptors
        ]
        # The custom geometry prefix adds metadata fields to the native rows;
        # normalize their projected order without dropping any alternative.
        return tuple(sorted(rows, key=lambda row: row.identity()))

    diagnostic_oneshot = bool(
        getattr(vision_model, "_rpu_vision_graph_disable", False))
    plan_box = {}
    plan_bounded_prefill_execution(
        logical_len,
        min(
            4096, int(getattr(vision_model._rpu_vision_kv_cache,
                              "max_seq_len", logical_len))),
        0,
        alignment=1, padding_rows=0, exact_chunk_size=exact,
        resolve_stage_domain=resolve_domain,
        request_id=f"vision:{group_id}",
        graph_mode="RETAINED_CACHE", queue_owner_id=int(handle),
        execution_owner=vision_model, execution_stage="vision",
        execution_native=("qwen3_5_vision", int(handle)),
        plan_signature=(int(temporal_num_frames), int(camera_batch_count),
                        bool(compact_input), diagnostic_oneshot),
        graph_cache=(None if diagnostic_oneshot else
                     getattr(vision_model, "_rpu_vision_graph_cache", None)),
        plan_result_sink=lambda result: plan_box.__setitem__("result", result),
    )
    return plan_box["result"]


def _vision_group_graph_signature(key, vision_plan):
    (n_i, hidden_size, num_layers, temporal_num_frames, camera_batch_count,
     direct_fusion, compact_input, embed_shape, _digest, generation) = key
    metadata = dict(vision_plan.selected.stage_tuple.physical_metadata)
    return rpu_backend.graph.GraphSignature(
        op_id="qwen3_5_vision",
        shapes=[n_i, hidden_size, *embed_shape],
        dyn_dims=[
            num_layers, temporal_num_frames, camera_batch_count,
            int(direct_fusion), int(compact_input),
            int(vision_plan.selected.stage_tuple.input_chunk),
            int(vision_plan.selected.stage_tuple.qkv_chunk),
            int(vision_plan.selected.stage_tuple.compute_chunk),
            int(metadata["merger_chunk"]),
            int(metadata["merger_chunks_per_span"]),
            int(metadata["stage_plan_fingerprint_hi"]),
            int(metadata["stage_plan_fingerprint_lo"]),
            int(metadata["layout_hash_hi"]),
            int(metadata["layout_hash_lo"]),
            generation, *vision_plan.graph_key_words(),
        ],
        dtypes=[torch.float16],
    )


def _evict_vision_graph(model, signature=None):
    if signature is None:
        signature = model._rpu_vision_graph_sig
    if signature is not None:
        model._rpu_vision_graph_cache.evict(signature)
    model._rpu_vision_graph_key = None
    model._rpu_vision_graph_sig = None


def _run_vision_diagnostic_oneshot(model, op_args):
    """Run a diagnostic capture without admitting it to the retained cache."""
    retained_entries = model._rpu_vision_graph_cache.size()
    with _oneshot_scope(model._rpu_vision_debug_graph):
        output = torch.ops.rpu.qwen3_5_vision_forward(*op_args)
    if model._rpu_vision_debug_graph.state() != 0:
        raise RuntimeError(
            "Qwen3.5 Vision diagnostic one-shot did not return to PASSTHROUGH"
        )
    if model._rpu_vision_graph_cache.size() != retained_entries:
        raise RuntimeError(
            "Qwen3.5 Vision diagnostic one-shot changed retained GraphCache"
        )
    return output


_QWEN3_5_VISION_READY_ATTRS = (
    "_rpu_vision_handle",
    "_rpu_vision_handle_finalizer",
    "_rpu_vision_freq_cos",
    "_rpu_vision_freq_sin",
    "_rpu_vision_position_idx_keepalive",
    "_rpu_vision_step0",
    "_rpu_vision_patch_embed_w",
    "_rpu_vision_patch_embed_b",
    "_rpu_vision_has_merger",
    "_rpu_vision_kv_cache",
    "_rpu_vision_graph_disable",
    "_rpu_vision_graph_cache",
    "_rpu_vision_graph_key",
    "_rpu_vision_graph_sig",
    "_rpu_vision_debug_graph",
    "_rpu_vision_spatial_merge_size",
    "_rpu_vision_num_layers",
    "_rpu_vision_hidden_size",
    "_rpu_vision_linear_acc32",
    "_rpu_vision_gelu_erf_mode",
    "_rpu_vision_chunk_size_cap",
    "_rpu_vision_exact_chunk_size",
    "_rpu_vision_padding_rows",
    "_rpu_vision_padding_budget",
    "_rpu_vision_control_snapshot",
    "_rpu_vision_last_a6_plan",
    "_rpu_vision_a6_plans",
    "_rpu_qwen3_5_had_instance_forward",
    "_rpu_qwen3_5_original_forward",
)

_QWEN3_5_VISION_PREINSTALL_ATTRS = frozenset({
    "_rpu_vision_graph_disable",
    "_rpu_vision_rope_disable",
})


def _qwen3_5_vision_runtime_complete(
    vision_model, *, installed_forward=None
) -> bool:
    """Return whether the controlled-evaluation Vision runtime is complete.

    ``installed_forward`` lets a model-specific wrapper validate the shared
    forward it retained underneath its own instance-level wrapper. Normal
    Qwen3.5 callers continue to validate the currently published forward.
    """
    if getattr(vision_model, "_rpu_vision_installing", False):
        return False
    if any(not hasattr(vision_model, name)
           for name in _QWEN3_5_VISION_READY_ATTRS):
        return False
    handle = getattr(vision_model, "_rpu_vision_handle", None)
    finalizer = getattr(
        vision_model, "_rpu_vision_handle_finalizer", None
    )
    if handle is None or finalizer is None:
        return False
    from rpu_backend.adapters.qwen3_5.text import _qwen35_retirement_ready

    state = getattr(vision_model, "_rpu_vision_retirement_state", None)
    if (not _qwen35_retirement_ready(state) or state.handle != handle
            or state.handle_finalizer is not finalizer):
        return False
    from rpu_backend.api import causal_lm

    with causal_lm._LIVE_LOCK:
        if causal_lm._LIVE_TERMINAL_REASON is not None:
            return False
    if getattr(vision_model, "_rpu_vision_graph_cache", None) is None:
        return False
    if getattr(vision_model, "_rpu_vision_debug_graph", None) is None:
        return False
    for name in (
        "_rpu_vision_freq_cos",
        "_rpu_vision_freq_sin",
        "_rpu_vision_position_idx_keepalive",
        "_rpu_vision_patch_embed_w",
        "_rpu_vision_kv_cache",
    ):
        if getattr(vision_model, name, None) is None:
            return False
    blocks = getattr(vision_model, "blocks", None)
    if not blocks:
        return False
    if getattr(vision_model, "_rpu_vision_num_layers", None) != len(blocks):
        return False
    if getattr(vision_model, "_rpu_vision_step0", False):
        for name in (
            "_rpu_vision_pe_w_swz",
            "_rpu_vision_pe_b_rpu",
            "_rpu_vision_step0_pos_grid",
            "_rpu_vision_step0_pos",
            "_rpu_vision_step0_pos_temporal_num_frames",
            "_rpu_vision_step0_pos_camera_batch_count",
        ):
            if not hasattr(vision_model, name):
                return False
        if getattr(vision_model, "_rpu_vision_pe_w_swz", None) is None:
            return False
        if getattr(vision_model, "_rpu_vision_pe_b_rpu", None) is None:
            return False
    if getattr(vision_model, "_rpu_vision_has_merger", False):
        if getattr(vision_model, "_rpu_vision_merger_refs", None) is None:
            return False
    for block in blocks:
        if getattr(
            block, "_rpu_qwen3_5_vision_weights_converted", False
        ) is not True:
            return False
        if getattr(block, "_rpu_qwen3_5_vision_conversion_started", False):
            return False
    if installed_forward is None:
        installed_forward = vars(vision_model).get("forward")
    return bool(
        getattr(installed_forward, "__self__", None) is vision_model
        and getattr(installed_forward, "__func__", None)
        is _rpu_vision_forward
    )


def _rollback_qwen3_5_vision_install(vision_model, *, _graphs_retired=False) -> None:
    """Retire resources strictly; the caller owns the Session boundary."""
    state = getattr(vision_model, "_rpu_vision_retirement_state", None)
    resource = getattr(state, "retirement", None)
    if _graphs_retired:
        from rpu_backend.api._execution import ExecutionSession

        binding = getattr(vision_model, "_planner_cost_session", None)
        session = getattr(state, "_execution_session", None)
        if session is None:
            session = getattr(vision_model, "_execution_session", None) or (binding[0]() if binding else None)
        if not isinstance(session, ExecutionSession) or not session._shutting_down or session._active:
            raise RuntimeError("Qwen3.5 precleared Vision retirement requires its exclusive Session shutdown")
    finalizer = getattr(
        vision_model, "_rpu_vision_installing_finalizer", None)
    if finalizer is None:
        finalizer = getattr(
            vision_model, "_rpu_vision_handle_finalizer", None)
    handle = getattr(vision_model, "_rpu_vision_installing_handle", None)
    if handle is None:
        handle = getattr(vision_model, "_rpu_vision_handle", None)
    from rpu_backend.api import causal_lm

    # A prior failed release is process-terminal, including a pending handle
    # whose finalizer could not yet be constructed. Never guess a safe retry.
    with causal_lm._LIVE_LOCK:
        if causal_lm._LIVE_TERMINAL_REASON is not None:
            raise RuntimeError("Qwen3.5 Vision retirement blocked by unsafe process state")
    try:
        if resource is not None and (
                (resource.owner() is not None and resource.owner() is not state) or resource.handle != handle
                or state.handle != handle or resource.failed is not None):
            raise RuntimeError("Qwen3.5 Vision retirement handle identity changed")
        if handle is None and finalizer is not None and finalizer.alive:
            raise RuntimeError("Qwen3.5 Vision live finalizer has no recorded handle")
        if handle is not None and resource is None and finalizer is not None and not finalizer.alive:
            raise RuntimeError("Qwen3.5 Vision handle has a dead finalizer without a confirmed retirement")
        if not _graphs_retired:
            graph_cache = getattr(vision_model, "_rpu_vision_graph_cache", None)
            if graph_cache is not None:
                graph_cache.clear()
                if not graph_cache.cache_invariant_ok():
                    raise RuntimeError("Qwen3.5 Vision GraphCache invariant failed")
            graph = getattr(vision_model, "_rpu_vision_debug_graph", None)
            if graph is not None:
                graph.invalidate()
        if handle is not None:
            torch.ops.rpu.qwen3_5_vision_destroy(handle)
            if resource is not None:
                resource._native_destroyed(handle)
            elif finalizer is not None:
                finalizer.detach()
    except BaseException as cleanup_error:
        session = getattr(state, "_execution_session", None)
        if session is None:
            session = getattr(vision_model, "_execution_session", None)
        binding = getattr(vision_model, "_planner_cost_session", None)
        if session is None and binding is not None:
            session = binding[0]()
        if session is not None:
            session.poison()
        with causal_lm._LIVE_LOCK:
            if causal_lm._LIVE_TERMINAL_REASON is None:
                owner = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                if owner is None:
                    owner = vision_model
                    causal_lm._claim_live_instance(owner)
                causal_lm._poison_live_instance(
                    owner, f"Qwen3.5 Vision cleanup failed: {cleanup_error!r}", unsafe=True)
            from rpu_backend.api._execution import _mark_execution_process_unsafe

            _mark_execution_process_unsafe(f"Qwen3.5 Vision cleanup failed: {cleanup_error!r}")
        if resource is not None:
            resource.retain_failure(cleanup_error, vision_model)
        raise

    model_state = vars(vision_model)
    for name in tuple(model_state):
        if name.startswith("_rpu_vision_"):
            model_state.pop(name, None)
    if _graphs_retired:
        # A successful parent close is terminal, not failed-install rollback:
        # never expose the original HF forward over irreversibly swizzled weights.
        return
    had_instance_forward = getattr(
        vision_model, "_rpu_qwen3_5_had_instance_forward", None)
    original_forward = getattr(
        vision_model, "_rpu_qwen3_5_original_forward", None)
    if had_instance_forward is True:
        model_state["forward"] = original_forward
    elif had_instance_forward is False:
        model_state.pop("forward", None)
    elif had_instance_forward is None and original_forward is not None:
        # Compatibility with an in-process install started by older code.
        model_state["forward"] = original_forward


def _qwen35_vision_precision(
    execution_config, *, prepared_profile=None,
):
    """Resolve independent MoE controls while preserving other profiles."""
    stage = (execution_config or {}).get("vision", {})
    if prepared_profile is not None:
        _qwen3_5_moe_vision_profile_layout(prepared_profile)
        acc32 = stage.get("linear_acc32", False)
        ultra_erf = stage.get("gelu_erf_ultra", False)
        for name, value in (("linear_acc32", acc32), ("gelu_erf_ultra", ultra_erf)):
            if not isinstance(value, bool):
                raise TypeError(f"Qwen3.5-MoE vision.{name} must be bool")
        return acc32, ultra_erf
    if "gelu_erf_ultra" in stage:
        raise ValueError("vision.gelu_erf_ultra is supported only by Qwen3.5-MoE")
    acc32 = stage.get("linear_acc32", False)
    if not isinstance(acc32, bool):
        raise TypeError("Qwen3.5 vision.linear_acc32 must be bool")
    # Explicit ACC32 selects the same-formula ultra kernel; tanh blocks stay tanh.
    ultra_erf = stage.get("linear_acc32") is True
    return acc32, ultra_erf


def install_qwen3_5_vision_for_rpu(
    model,
    *,
    vision_config: Any | None = None,
    execution_config=None,
    _resolved_options=None,
    _graph_runtime_policy=None,
    _cold_numeric_opt_in: bool = False,
    _text_bridge: Qwen3_5MoeVisionTextBridge | None = None,
    max_hw: int = 128,        # 每维 patch 上限(2D rope 表 [max_hw,hd/4] 行数,~4KB 近乎免费)；128→单边 ≤2048px、~4:1 长宽比。原 48 卡 >768px
    max_seq_len: int = 4096,  # Generic vision KV cache cap.
    num_cores: int | None = None,
) -> int:
    """Install a controlled-evaluation Qwen3.5 vision profile transactionally.

    A live install is an idempotent no-op. Failures before weight mutation can
    be retried; failures after an irreversible block transform retain a poison
    marker and require a model reload. Only successful resource retirement
    removes the in-flight owners and restores the pre-install forward; failed
    retirement preserves them and requires a fresh process.
    """
    from rpu_backend.api import causal_lm

    with causal_lm._LIVE_LOCK:
        if causal_lm._LIVE_TERMINAL_REASON is not None:
            raise RuntimeError("Qwen3.5 Vision installation blocked by unsafe process state")
    vision_model = _resolve_vision_model(model)
    config = vision_config or vision_model.config
    moe_35b_geometry = tuple(
        getattr(config, name, None)
        for name in (
            "depth", "num_heads", "hidden_size", "intermediate_size",
            "out_hidden_size",
        )
    ) == (27, 16, 1152, 4304, 2048)
    prepared_profile = None
    if _text_bridge is not None:
        _validate_qwen3_5_moe_vision_text_bridge(
            _text_bridge, model, require_vision_policy=True
        )
        prepared_profile = _text_bridge.prepared_profile
    elif moe_35b_geometry or (
        getattr(config, "model_type", None) == "qwen3_5_moe"
    ):
        raise ValueError(
            "Qwen3.5-MoE Vision requires its verified text bridge"
        )
    from rpu_backend.adapters.qwen3_5.cores import vision_core_count, validate_vision_core_profile
    selected_cores = vision_core_count(execution_config)
    if prepared_profile is not None:
        prepared_root_cores = _qwen3_5_moe_vision_profile_root_cores(
            prepared_profile
        )
        prepared_vision_cores = _qwen3_5_moe_vision_profile_cores(
            prepared_profile
        )
        configured_root_cores = (execution_config or {}).get(
            "model", {}
        ).get("num_cores", 8)
        if configured_root_cores != prepared_root_cores:
            raise ValueError(
                "Qwen3.5-MoE Vision execution config differs from its "
                f"prepared root{prepared_root_cores} profile"
            )
        if selected_cores != prepared_vision_cores:
            raise ValueError(
                "Qwen3.5-MoE Vision owner selection differs from its "
                f"prepared Vision{prepared_vision_cores} profile"
            )
    if num_cores is None:
        num_cores = selected_cores
    elif execution_config and "model" in execution_config and num_cores != selected_cores:
        raise ValueError("Qwen3.5 Vision core count conflicts with root model budget")
    if prepared_profile is None:
        validate_vision_core_profile(config, num_cores)
    else:
        _validate_qwen3_5_moe_35b_vision_profile(
            vision_model, config, num_cores, prepared_profile
        )
    if _graph_runtime_policy is not None:
        if not isinstance(_graph_runtime_policy, rpu_backend.graph.GraphRuntimePolicy):
            raise TypeError("Qwen3.5 Vision graph policy must be a GraphRuntimePolicy")
        if (_graph_runtime_policy.execution_core_count != num_cores
                or _graph_runtime_policy.graph_arena_count != 4
                or _graph_runtime_policy.qwen35_legacy_27b_sdk_budget
                or _cold_numeric_opt_in is not True):
            raise ValueError("Qwen3.5 Vision requires its prepared cold graph policy and numeric opt-in")
        if prepared_profile is not None:
            if _graph_runtime_policy is not _text_bridge.state.vision_graph_policy:
                raise RuntimeError(
                    "Qwen3.5-MoE Vision graph policy differs from its cold owner"
                )
            _validate_qwen3_5_moe_35b_vision_profile(
                vision_model, config, num_cores, prepared_profile
            )
        else:
            from rpu_backend.adapters.qwen3_5.cores import validate_dense_9b_vision_profile

            validate_dense_9b_vision_profile(config)
    elif _cold_numeric_opt_in:
        raise ValueError("Qwen3.5 Vision cold numeric opt-in requires its prepared graph policy")
    elif prepared_profile is not None:
        raise ValueError(
            "Qwen3.5-MoE Vision requires its prepared cold graph policy"
        )
    requested_acc32, requested_ultra = _qwen35_vision_precision(
        execution_config,
        prepared_profile=prepared_profile)
    if num_cores != 8:
        for name in ("qwen3_5_vision_set_execution_cores", "qwen3_5_vision_get_execution_topology"):
            if not hasattr(torch.ops.rpu, name):
                raise RuntimeError(f"Qwen3.5 binary lacks reduced-core op {name}; rebuild before installing")
    resolved_options = _resolve_vision_install_options(
        execution_config=execution_config,
        resolved_options=_resolved_options,
    )
    with _VISION_INSTALL_LOCK:
        if getattr(vision_model, "_rpu_vision_installing", False):
            raise RuntimeError(
                "Qwen3.5 vision installation re-entered before commit"
            )
        if _qwen3_5_vision_runtime_complete(vision_model):
            if _graph_runtime_policy is not None:
                installed_policy = (
                    vision_model._rpu_vision_graph_cache.runtime_policy
                )
                policy_changed = (
                    installed_policy is not _graph_runtime_policy
                    if _text_bridge is not None
                    else installed_policy != _graph_runtime_policy
                )
                if policy_changed:
                    raise RuntimeError(
                        "Qwen3.5 Vision graph policy is immutable after installation"
                    )
            if getattr(vision_model, "_rpu_vision_num_cores", 8) != num_cores:
                raise RuntimeError("Qwen3.5 Vision core count is immutable after installation")
            if vision_model._rpu_vision_linear_acc32 != requested_acc32:
                raise RuntimeError(
                    "Qwen3.5 vision Linear accumulation is immutable after "
                    "installation"
                )
            if getattr(vision_model, "_rpu_vision_gelu_erf_mode", "standard") != (
                    "ultra" if requested_ultra else "standard"):
                raise RuntimeError("Qwen3.5 vision GELU precision is immutable after installation")
            installed = (
                vision_model._rpu_vision_chunk_size_cap,
                vision_model._rpu_vision_exact_chunk_size,
                vision_model._rpu_vision_padding_rows,
                vision_model._rpu_vision_padding_budget,
                vision_model._rpu_vision_control_snapshot,
            )
            if installed != resolved_options:
                raise RuntimeError(
                    "Qwen3.5 vision execution controls are immutable after "
                    "installation; use the planner reconfigure lifecycle"
                )
            if _text_bridge is not None:
                state = _validate_qwen3_5_moe_vision_text_bridge(
                    _text_bridge, model, require_vision_policy=True
                )
                _validate_qwen3_5_moe_vision_retirement(
                    vision_model, _text_bridge, state
                )
            return vision_model._rpu_vision_handle
        stale_runtime_attrs = {
            name for name in vars(vision_model)
            if name.startswith("_rpu_vision_")
            and name not in _QWEN3_5_VISION_PREINSTALL_ATTRS
        }
        if stale_runtime_attrs:
            raise RuntimeError(
                "Qwen3.5 vision runtime state is incomplete; reload the "
                "model instead of reinstalling mutated weights."
            )
        try:
            if _graph_runtime_policy is not None:
                if prepared_profile is not None:
                    _validate_qwen3_5_moe_35b_vision_weights(
                        vision_model, config, num_cores, prepared_profile
                    )
                else:
                    from rpu_backend.adapters.qwen3_5.cores import (
                        validate_dense_9b_vision_weights, validate_cold_weight_dtype,
                    )
                    validate_dense_9b_vision_weights(vision_model, config)
                    validate_cold_weight_dtype(vision_model, fp16=True)
            # Vision may install lazily inside an inference-mode request.
            # Its position backing and cache buffers must still support
            # in-place updates when a later request uses no_grad instead.
            with torch.inference_mode(False), torch.no_grad():
                handle = _install_qwen3_5_vision_for_rpu_impl(
                    model,
                    vision_config=vision_config,
                    execution_config=execution_config,
                    resolved_options=resolved_options,
                    max_hw=max_hw,
                    max_seq_len=max_seq_len,
                    num_cores=num_cores,
                    _graph_runtime_policy=_graph_runtime_policy,
                    _cold_numeric_opt_in=_cold_numeric_opt_in,
                    _prepared_profile=prepared_profile,
                )
            if _text_bridge is not None:
                state = _validate_qwen3_5_moe_vision_text_bridge(
                    _text_bridge, model, require_vision_policy=True
                )
                session = state._execution_session
                parent = _text_bridge.model
                close_callback = _text_bridge.close_callback
            else:
                binding = getattr(vision_model, "_planner_cost_session", None)
                session = binding[0]() if binding else None
                parent = session._owner if session is not None else None
                close_callback = None
            if session is not None:
                from rpu_backend.adapters.qwen3_5.text import _bind_qwen35_retirement

                if close_callback is None:
                    from rpu_backend.adapters.qwen3_5 import _close_qwen3_5_model

                    close_callback = _close_qwen3_5_model
                _bind_qwen35_retirement(vision_model._rpu_vision_retirement_state,
                                       parent, close_callback, vision_model)
            return handle
        except BaseException as error:
            if any(
                hasattr(vision_model, name)
                for name in (
                    "_rpu_vision_installing",
                    "_rpu_vision_installing_handle",
                    "_rpu_vision_handle",
                )
            ):
                try:
                    _rollback_qwen3_5_vision_install(vision_model)
                except BaseException as cleanup_error:
                    error.add_note(f"Qwen3.5 Vision cleanup failed: {cleanup_error!r}")
            raise


def _install_qwen3_5_vision_for_rpu_impl(
    model,
    *,
    vision_config: Any | None = None,
    execution_config=None,
    resolved_options=None,
    max_hw: int = 128,
    max_seq_len: int = 4096,
    num_cores: int = 8,
    _graph_runtime_policy=None,
    _cold_numeric_opt_in: bool = False,
    _prepared_profile: str | None = None,
) -> int:
    """Install RPU all-layers-once forward on the ``Qwen3_5VisionModel``.

    ``model`` may be the top ``Qwen3_5ForConditionalGeneration``, the middle
    ``Qwen3_5Model`` fusion, or the ``Qwen3_5VisionModel`` itself — the vision
    tower is resolved via ``model.model.visual`` (or used directly). Idempotence
    is handled by the public wrapper; this implementation performs one install.

    Returns the C++ handle (also stashed at ``vision_model._rpu_vision_handle``).
    """
    vision_model = _resolve_vision_model(model)
    cfg = vision_config or vision_model.config
    from rpu_backend.adapters.qwen3_5.cores import validate_vision_core_profile
    if _prepared_profile is None:
        validate_vision_core_profile(cfg, num_cores)
    else:
        _validate_qwen3_5_moe_35b_vision_profile(
            vision_model, cfg, num_cores, _prepared_profile
        )
    if (
        isinstance(max_seq_len, bool)
        or not isinstance(max_seq_len, Integral)
        or not 0 < int(max_seq_len) <= 4096
    ):
        raise ValueError(
            "Qwen3.5 generic vision max_seq_len must be an integer in [1, 4096]"
        )
    max_seq_len = int(max_seq_len)
    (
        chunk_size_cap,
        exact_chunk_size,
        padding_rows,
        padding_budget,
        control_snapshot,
    ) = _resolve_vision_install_options(
        execution_config=execution_config,
        resolved_options=resolved_options,
    )

    num_layers = len(vision_model.blocks)
    if num_cores != 8 and num_layers != cfg.depth:
        raise ValueError("Qwen3.5 Vision layer count disagrees with config")
    num_heads = cfg.num_heads
    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"Qwen3.5 vision hidden_size={hidden_size} must be divisible by "
            f"num_heads={num_heads}"
        )
    head_dim = hidden_size // num_heads
    padded_head_dim = _qwen3_5_vision_padded_head_dim(head_dim)
    spatial_merge_size = cfg.spatial_merge_size
    eps = 1e-6  # Qwen3_5VisionBlock LayerNorm eps (hardcoded in HF)

    # Shape support is not production admission.  Candidate geometries may run
    # only behind the same exact controlled-evaluation opt-in as 2B/4B; each
    # profile still needs its own numerical, Graph, and lifecycle evidence.
    vision_profile = (num_layers, num_heads, hidden_size, intermediate_size)
    controlled_profiles = (
        _QWEN3_5_VISION_PROFILES | _QWEN3_5_VISION_CANDIDATE_PROFILES
    )
    if vision_profile not in controlled_profiles:
        raise ValueError(
            f"Qwen3.5 vision profile {vision_profile} is unsupported; "
            "runtime-capable controlled profiles are "
            f"{sorted(controlled_profiles)}."
        )
    if (
        not _cold_numeric_opt_in
        and os.environ.get("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED") != "1"
    ):
        raise NotImplementedError(
            "Qwen3.5 Vision is experimental/numeric-blocked: profile-local "
            "official real-image correctness gates are incomplete or fail, "
            "and no production-safe "
            "input envelope is certified. Image inference is disabled by "
            "default; Qwen3.5 text remains supported. For controlled "
            "evaluation only, set QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1 "
            "before the first image forward."
        )
    set_cap = getattr(
        torch.ops.rpu, "qwen3_5_vision_set_chunk_size_cap", None
    )
    resolver = getattr(
        torch.ops.rpu, "qwen3_5_vision_resolve_prefill_domain", None
    )
    set_exact = getattr(
        torch.ops.rpu, "qwen3_5_vision_set_prefill_chunk_size", None
    )
    set_linear_acc32 = getattr(
        torch.ops.rpu, "qwen3_5_vision_set_linear_acc32", None
    )
    linear_acc32, ultra_erf = _qwen35_vision_precision(
        execution_config,
        prepared_profile=_prepared_profile)
    set_moe_precision = getattr(torch.ops.rpu, "qwen3_5_vision_set_moe_precision", None)
    if _prepared_profile is not None and set_moe_precision is None:
        raise RuntimeError("Qwen3.5-MoE vision binary lacks qwen3_5_vision_set_moe_precision; rebuild before installing")
    set_gelu_erf_ultra = getattr(torch.ops.rpu, "qwen3_5_vision_set_gelu_erf_ultra", None)
    if _prepared_profile is None and ultra_erf and set_gelu_erf_ultra is None:
        raise RuntimeError("Qwen3.5 vision binary lacks qwen3_5_vision_set_gelu_erf_ultra; rebuild before installing")
    if (
        set_cap is None or set_exact is None or resolver is None
        or set_linear_acc32 is None
    ):
        missing = [
            name
            for name, value in (
                ("qwen3_5_vision_set_chunk_size_cap", set_cap),
                ("qwen3_5_vision_set_prefill_chunk_size", set_exact),
                ("qwen3_5_vision_resolve_prefill_domain", resolver),
                ("qwen3_5_vision_set_linear_acc32", set_linear_acc32),
            )
            if value is None
        ]
        raise RuntimeError(
            "Qwen3.5 vision binary lacks required planner op(s): "
            + ", ".join(missing)
            + "; rebuild rpu_backend before installing Vision"
        )
    graph_options = {}
    if _graph_runtime_policy is not None:
        graph_options["runtime_policy"] = _graph_runtime_policy
    elif num_cores != 8:
        graph_options["runtime_policy"] = rpu_backend.graph.GraphRuntimePolicy.from_environment(
            execution_core_count=num_cores)
    if not hasattr(vision_model, "_rpu_qwen3_5_original_forward"):
        model_vars = vars(vision_model)
        vision_model._rpu_qwen3_5_had_instance_forward = "forward" in model_vars
        vision_model._rpu_qwen3_5_original_forward = model_vars.get("forward")
    vision_model._rpu_vision_installing = True

    # ------------------------------------------------------------------ #
    # Step 1: create handle + GC finalizer
    # ------------------------------------------------------------------ #
    handle = torch.ops.rpu.qwen3_5_vision_create()
    # Publish pending resource ownership without user __setattr__ interceptors;
    # a failed ready-field publication below must still find both resources.
    vars(vision_model)["_rpu_vision_installing_handle"] = handle
    from rpu_backend.adapters.qwen3_5.text import _Qwen35NativeState
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource

    retirement_state = _Qwen35NativeState(handle=handle)
    vars(vision_model)["_rpu_vision_retirement_state"] = retirement_state
    retirement_state.retirement = resource = _InstalledNativeResource(
        retirement_state, handle, torch.ops.rpu.qwen3_5_vision_destroy,
        graphs=(), keepalive=(), label="Qwen3.5 Vision", handle_name="handle")
    handle_finalizer = retirement_state.handle_finalizer = resource.finalizer
    vars(vision_model)["_rpu_vision_installing_finalizer"] = handle_finalizer
    # Bind both MoE choices atomically, before any weight transform/upload.
    # Dense explicit ACC32 retains the ordinary setter semantics.
    if num_cores != 8:
        torch.ops.rpu.qwen3_5_vision_set_execution_cores(handle, num_cores)
    if _prepared_profile is not None:
        set_moe_precision(handle, linear_acc32, ultra_erf)
    else:
        set_linear_acc32(handle, linear_acc32)
        if ultra_erf:
            set_gelu_erf_ultra(handle, True)
    # Latch the per-handle cap before any forward.  Passing zero explicitly is
    # important: it prevents the native legacy getenv fallback from leaking a
    # process-wide environment value into a canonical ``auto`` request.
    set_cap(handle, chunk_size_cap)
    set_exact(handle, exact_chunk_size or 0)

    # ------------------------------------------------------------------ #
    # Step 2: convert per-block encoder weights (split fused QKV + swizzle)
    # ------------------------------------------------------------------ #
    for block in vision_model.blocks:
        _convert_vision_block_weights_for_rpu(block, num_heads, hidden_size,
                                            **({"num_cores": num_cores} if num_cores != 8 else {}))

    # ------------------------------------------------------------------ #
    # Step 3: gather per-layer weights into 16 lists (one per arg)
    # ------------------------------------------------------------------ #
    q_w_list, k_w_list, v_w_list, o_w_list = [], [], [], []
    fc1_w_list, fc2_w_list = [], []
    ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list = [], [], [], []
    q_b_list, k_b_list, v_b_list, o_b_list = [], [], [], []
    fc1_b_list, fc2_b_list = [], []

    for block in vision_model.blocks:
        q_w_list.append(block._rpu_q_w.to(dtype=torch.float16, device="rpu"))
        k_w_list.append(block._rpu_k_w.to(dtype=torch.float16, device="rpu"))
        v_w_list.append(block._rpu_v_w.to(dtype=torch.float16, device="rpu"))
        o_w_list.append(block.attn.proj.weight.to(dtype=torch.float16, device="rpu"))

        fc1_w_list.append(block.mlp.linear_fc1.weight.to(dtype=torch.float16, device="rpu"))
        fc2_w_list.append(block.mlp.linear_fc2.weight.to(dtype=torch.float16, device="rpu"))

        ln1_w_list.append(block.norm1.weight.to(dtype=torch.float16, device="rpu"))
        ln1_b_list.append(block.norm1.bias.to(dtype=torch.float16, device="rpu"))
        ln2_w_list.append(block.norm2.weight.to(dtype=torch.float16, device="rpu"))
        ln2_b_list.append(block.norm2.bias.to(dtype=torch.float16, device="rpu"))

        q_b_list.append(block._rpu_q_b.to(dtype=torch.float16, device="rpu"))
        k_b_list.append(block._rpu_k_b.to(dtype=torch.float16, device="rpu"))
        v_b_list.append(block._rpu_v_b.to(dtype=torch.float16, device="rpu"))
        o_b_list.append(block._rpu_o_b.to(dtype=torch.float16, device="rpu"))

        fc1_b_list.append(block._rpu_fc1_b.to(dtype=torch.float16, device="rpu"))
        fc2_b_list.append(block._rpu_fc2_b.to(dtype=torch.float16, device="rpu"))

    # ------------------------------------------------------------------ #
    # Step 4: set_weights (no deepstack_visual_indexes — Qwen3.5 has no DeepStack)
    # ------------------------------------------------------------------ #
    resource.keepalive = (
        q_w_list, k_w_list, v_w_list, o_w_list, fc1_w_list, fc2_w_list,
        ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list,
        q_b_list, k_b_list, v_b_list, o_b_list, fc1_b_list, fc2_b_list)
    torch.ops.rpu.qwen3_5_vision_set_weights(
        handle,
        q_w_list, k_w_list, v_w_list, o_w_list,
        fc1_w_list, fc2_w_list,
        ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list,
        q_b_list, k_b_list, v_b_list, o_b_list,
        fc1_b_list, fc2_b_list,
        num_heads, head_dim, hidden_size, intermediate_size,
        float(eps),
    )
    if num_cores != 8:
        actual = tuple(torch.ops.rpu.qwen3_5_vision_get_execution_topology(handle))
        if actual != (num_cores, num_cores, num_cores, 8):
            raise RuntimeError(f"Qwen3.5 Vision native/Python topology mismatch: {actual}")

    # ------------------------------------------------------------------ #
    # Step 5: build + set rope tables, retrieve keepalive view
    # ------------------------------------------------------------------ #
    freq_cos, freq_sin = build_vision_rope_tables(
        head_dim,
        max_hw,
        device="rpu",
        dtype=torch.float16,
        inv_freq=vision_model.rotary_pos_emb.inv_freq,
    )
    vision_model._rpu_vision_freq_cos = freq_cos
    vision_model._rpu_vision_freq_sin = freq_sin
    resource.keepalive += (freq_cos, freq_sin)
    torch.ops.rpu.qwen3_5_vision_set_rope(handle, freq_cos, freq_sin)

    position_idx_keepalive = torch.ops.rpu.qwen3_5_vision_position_idx_keepalive(handle)
    vision_model._rpu_vision_position_idx_keepalive = position_idx_keepalive

    # ------------------------------------------------------------------ #
    # Step 6: fold patch_embed Conv3d → Linear weight.
    #
    # STEP 0 (patch_embed + exact HF position upload) runs in the fused graph unless
    # RPU_VISION_STEP0=0 retains the CPU fallback. The device path must use
    # column-swizzled patch weights and preserve the HF grid's token order.
    # Prepare the full-grid position tensor before entering the graph;
    # wrong weight layout or token order can silently change the output.
    # ------------------------------------------------------------------ #
    pe = vision_model.patch_embed
    pe_w_folded = _fold_conv3d_to_linear_weight(pe.proj.weight.data)
    pe_b = pe.proj.bias.data if pe.proj.bias is not None else None

    use_rpu_step0 = rpu_env_bool("RPU_VISION_STEP0", default=True)
    vision_model._rpu_vision_step0 = use_rpu_step0

    # CPU copies — the fallback path reads these, and they cost ~5MB.
    vision_model._rpu_vision_patch_embed_w = pe_w_folded.to(dtype=torch.float16, device="cpu").contiguous()
    vision_model._rpu_vision_patch_embed_b = (
        pe_b.to(dtype=torch.float16, device="cpu").contiguous() if pe_b is not None else None
    )

    if use_rpu_step0:
        if pe_b is None:
            raise RuntimeError(
                "Qwen3.5 vision STEP 0: patch_embed has no bias, but the RPU path "
                "assumes one (set RPU_VISION_STEP0=0 to fall back to CPU)"
            )

        pe_w_swz = tp_col_swizzle_mc_weight(
            pe_w_folded.to(dtype=torch.float16, device="cpu").contiguous(),
            **({"num_cores": num_cores} if num_cores != 8 else {}),
        ).to("rpu")
        pe_b_rpu = pe_b.to(dtype=torch.float16, device="cpu").contiguous().to("rpu")
        # Hold refs: these back in-graph DMAs, so they must outlive every forward.
        vision_model._rpu_vision_pe_w_swz = pe_w_swz
        vision_model._rpu_vision_pe_b_rpu = pe_b_rpu
        resource.keepalive += (pe_w_swz, pe_b_rpu)

        torch.ops.rpu.qwen3_5_vision_set_patch_embed(handle, pe_w_swz, pe_b_rpu)
        vision_model._rpu_vision_step0_pos_grid = None
        vision_model._rpu_vision_step0_pos_temporal_num_frames = None
        vision_model._rpu_vision_step0_pos_camera_batch_count = None
        vision_model._rpu_vision_step0_pos = None

    # ------------------------------------------------------------------ #
    # Step 7: pos_embed stays on CPU for HF-exact whole-grid interpolation and
    # for the full CPU fallback. The patch merger goes IN-GRAPH as a
    # post_layers_fn when RPU_VISION_MERGER != 0. RPU_VISION_MERGER=0 keeps
    # merger arithmetic on CPU fp32 for A/B. The CPU module stays
    # resident either way.
    # No deepstack_merger_list on Qwen3.5.
    # ------------------------------------------------------------------ #
    vision_model.pos_embed.to(device="cpu", dtype=torch.float16)
    vision_model.merger.to(device="cpu", dtype=torch.float32)

    use_rpu_merger = rpu_env_bool("RPU_VISION_MERGER", default=True)
    if use_rpu_merger:
        merger = vision_model.merger
        # HF Qwen3_5VisionPatchMerger: norm=LayerNorm(hidden, eps=1e-6) BEFORE the
        # 2x2 merge (use_postshuffle_norm=False), then fc1 [4h,4h] / GELU / fc2
        # [out_hidden,4h]. fc1 is COL-parallel (splits the 4h output), fc2 is
        # ROW-parallel (splits the 4h K then all_reduce).
        #
        # ⚠️ Deliberately the col/row primitives, NOT transform_linear_weight(w,
        # partition): 4h and out_hidden satisfy BOTH the col (N%(16*tp)) and row
        # (N%16 / K%128) asserts, so a wrong partition would raise nothing and just
        # produce a wrong result. fc1=col, fc2=row is load-bearing (see emit_merger).
        fc1_w = merger.linear_fc1.weight.data.to(dtype=torch.float16, device="cpu").contiguous()  # [4h, 4h]
        fc1_w_swz = tp_col_swizzle_mc_weight(fc1_w, **({"num_cores": num_cores} if num_cores != 8 else {})).to("rpu")
        fc1_b = merger.linear_fc1.bias.data.to(dtype=torch.float16, device="cpu").contiguous().to("rpu")
        fc2_w = merger.linear_fc2.weight.data.to(dtype=torch.float16, device="cpu").contiguous()  # [out_hidden, 4h]
        fc2_w_swz = tp_row_swizzle_mc_weight(fc2_w, **({"num_cores": num_cores} if num_cores != 8 else {})).to("rpu")
        fc2_b = merger.linear_fc2.bias.data.to(dtype=torch.float16, device="cpu").contiguous().to("rpu")
        norm_w = merger.norm.weight.data.to(dtype=torch.float16, device="cpu").contiguous().to("rpu")
        norm_b = merger.norm.bias.data.to(dtype=torch.float16, device="cpu").contiguous().to("rpu")
        out_hidden_size = int(fc2_w.size(0))
        # Hold refs — these back in-graph DMAs, so they must outlive every forward
        # (same contract as the STEP 0 pe_w_swz refs above).
        vision_model._rpu_vision_merger_refs = (
            fc1_w_swz, fc1_b, fc2_w_swz, fc2_b, norm_w, norm_b)
        resource.keepalive += vision_model._rpu_vision_merger_refs
        torch.ops.rpu.qwen3_5_vision_set_merger(
            handle, fc1_w_swz, fc1_b, fc2_w_swz, fc2_b, norm_w, norm_b, out_hidden_size)
    vision_model._rpu_vision_has_merger = use_rpu_merger

    # ------------------------------------------------------------------ #
    # Step 8: allocate dummy KV caches and the optional debug graph.
    # ------------------------------------------------------------------ #
    vision_model._rpu_vision_kv_cache = _make_dummy_vision_kv_caches(
        num_layers, max_seq_len, num_heads, padded_head_dim,
        **({"num_cores": num_cores} if num_cores != 8 else {})
    )
    # The ordinary dense 2B pilot may repeat two image shapes in one request.
    # Retain that bounded working set; specialized/large profiles keep one slot.
    graph_entries = 2 if (
        _prepared_profile is None
        and (cfg.depth, hidden_size, intermediate_size, num_heads,
             getattr(cfg, "out_hidden_size", None)) == (24, 1024, 4096, 16, 2048)
    ) else 1
    vision_model._rpu_vision_graph_disable = (
        os.environ.get("QWEN3_5_VISION_GRAPH_DISABLE", "0") != "0"
    )
    vision_model._rpu_vision_graph_cache = rpu_backend.graph.GraphCache(
        max_entries=graph_entries, **graph_options)
    resource.graphs = (vision_model._rpu_vision_graph_cache,)
    vision_model._rpu_vision_graph_key = None
    vision_model._rpu_vision_graph_sig = None
    vision_model._rpu_vision_debug_graph = rpu_backend.graph.Graph(**graph_options)
    resource.raw_graphs = (vision_model._rpu_vision_debug_graph,)
    resource.keepalive += (vision_model._rpu_vision_kv_cache, position_idx_keepalive)

    # Stash config for forward replacement.
    vision_model._rpu_vision_spatial_merge_size = spatial_merge_size
    vision_model._rpu_vision_num_layers = num_layers
    vision_model._rpu_vision_num_cores = num_cores
    vision_model._rpu_vision_hidden_size = hidden_size
    vision_model._rpu_vision_linear_acc32 = linear_acc32
    vision_model._rpu_vision_gelu_erf_mode = "ultra" if ultra_erf else "standard"
    vision_model._rpu_vision_chunk_size_cap = chunk_size_cap
    vision_model._rpu_vision_exact_chunk_size = exact_chunk_size
    vision_model._rpu_vision_padding_rows = padding_rows
    vision_model._rpu_vision_padding_budget = padding_budget
    vision_model._rpu_vision_control_snapshot = control_snapshot
    vision_model._rpu_vision_has_dispatched = False
    vision_model._rpu_vision_last_a6_plan = None
    vision_model._rpu_vision_a6_plans = {}

    # ------------------------------------------------------------------ #
    # Step 9: forward replacement
    # ------------------------------------------------------------------ #
    vision_model.forward = types.MethodType(_rpu_vision_forward, vision_model)

    # Publish the handle only after every dependent field and the replacement
    # forward are ready. Before this point rollback owns the temporary handle.
    vision_model._rpu_vision_handle = handle
    vision_model._rpu_vision_handle_finalizer = handle_finalizer
    del vision_model._rpu_vision_installing_handle
    del vision_model._rpu_vision_installing_finalizer
    del vision_model._rpu_vision_installing

    _LOG.info("Patched Qwen3_5VisionModel: handle=%d, num_layers=%d, "
              "hidden=%d, num_heads=%d, head_dim=%d/%d, "
              "intermediate=%d/%d, max_hw=%d",
              handle, num_layers, hidden_size, num_heads, head_dim,
              padded_head_dim, intermediate_size,
              _qwen3_5_vision_padded_intermediate_size(intermediate_size), max_hw)

    return handle


# ─────────────────────────────────────────────────────────────────────────────
# Forward replacement (instance method bound by install_qwen3_5_vision_for_rpu)
# ─────────────────────────────────────────────────────────────────────────────

def _rpu_vision_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
    """RPU-dispatching forward for Qwen3_5VisionModel.

    Args (mirror HF):
        hidden_states: ``[seq_len, cin*tp*ps*ps]`` pre-flattened pixel patches.
        grid_thw: ``[n_images, 3]`` int (T, H, W) per image. Read on CPU to
            build position indices and the exact whole-grid position tensor.

    Returns ``BaseModelOutputWithPooling(last_hidden_state, pooler_output)`` (no
    deepstack). Multi-image input runs once per image group.
    """
    from rpu_backend.api import causal_lm

    with causal_lm._LIVE_LOCK:
        if causal_lm._LIVE_TERMINAL_REASON is not None:
            raise RuntimeError("Qwen3.5 Vision forward blocked by unsafe process state")
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    handle = getattr(self, "_rpu_vision_handle", None)
    if handle is None:
        raise RuntimeError("Qwen3.5 Vision runtime is closed or not installed")
    spatial_merge_size = self._rpu_vision_spatial_merge_size
    num_layers = self._rpu_vision_num_layers
    cache = self._rpu_vision_kv_cache
    fusion_target = kwargs.pop("_rpu_fusion_target", None)
    fusion_run_starts = kwargs.pop("_rpu_fusion_run_starts", None)
    temporal_num_frames = int(kwargs.pop("_rpu_temporal_num_frames", 1))
    camera_batch_count = int(kwargs.pop("_rpu_camera_batch_count", 1))
    if temporal_num_frames != 1:
        raise ValueError("public Qwen3.5 Vision requires _rpu_temporal_num_frames=1")
    if camera_batch_count != 1:
        raise ValueError("public Qwen3.5 Vision requires _rpu_camera_batch_count=1")
    grid_thw_cpu = grid_thw.detach().cpu() if grid_thw.device.type != "cpu" else grid_thw
    _validate_vision_grid(
        grid_thw_cpu, spatial_merge_size, "Qwen3.5 vision forward")
    step0_on = getattr(self, "_rpu_vision_step0", False)
    if hidden_states.dim() != 2:
        raise ValueError("Qwen3.5 Vision requires flattened pixel patches")
    compact_input = False
    num_cores = getattr(self, "_rpu_vision_num_cores", 8)
    patches_per_image = grid_thw_cpu.prod(-1).tolist()  # [n_0, n_1, ...]
    total_patches = sum(patches_per_image)
    if hidden_states.size(0) != total_patches:
        raise RuntimeError(
            f"Qwen3.5 vision forward: patch input has {hidden_states.size(0)} "
            f"tokens but grid_thw implies {total_patches}"
        )
    groups = [(index, count, count) for index, count in enumerate(patches_per_image)]

    merger_on = getattr(self, "_rpu_vision_has_merger", False)
    direct_fusion = fusion_target is not None
    if direct_fusion:
        if not merger_on:
            raise RuntimeError("Qwen3.5 direct vision fusion requires the RPU merger")
        expected_starts = len(groups)
        if fusion_run_starts is None or len(fusion_run_starts) != expected_starts:
            raise RuntimeError(
                "Qwen3.5 direct vision fusion requires one run start per image")
        fusion_run_starts = tuple(int(x) for x in fusion_run_starts)

    # Admission is request-atomic: reject any later image group before the
    # first group can mutate KV, position keepalives, graphs, or fusion_target.
    planned_groups = tuple(
        (
            group,
            _plan_qwen3_5_vision_group(
                self,
                handle,
                group[1],
                group_id=i,
                temporal_num_frames=temporal_num_frames,
                camera_batch_count=camera_batch_count,
                compact_input=compact_input,
            ),
        )
        for i, group in enumerate(groups)
    )
    if planned_groups:
        (_, last_n_i, _), last_plan = planned_groups[-1]
        plan_key = (
            int(last_n_i), int(temporal_num_frames), int(camera_batch_count),
            bool(compact_input),
            getattr(self, "_rpu_vision_exact_chunk_size", None),
            getattr(self, "_rpu_vision_padding_rows", "auto"),
            getattr(self, "_rpu_vision_padding_budget", 0),
        )
        # Keep only the latest immutable planning receipt.
        self._rpu_vision_a6_plans = {plan_key: last_plan}
        self._rpu_vision_last_a6_plan = last_plan.as_dict(
            include_candidates=False
        )

    if step0_on:
        # ----- STEP 0 runs in the fused graph: hand the tower RAW folded patches.
        # No CPU patch-embedding GEMM and no RPU→CPU→RPU round trip. HF computes
        # the small position tensor once per grid; pre_layers_fn uploads and adds
        # it after patch_embed + all_gather.
        embed_packed = (
            hidden_states.to(device="rpu", dtype=torch.float16).contiguous()
            if (hidden_states.device.type != "rpu" or hidden_states.dtype != torch.float16)
            else hidden_states.contiguous()
        )
    else:
        # ----- Fallback: patch_embed + pos_embed on CPU (the pre-STEP0 path).
        # Kept for A/B against the in-graph path — set RPU_VISION_STEP0=0.
        if hidden_states.device.type != "cpu":
            hidden_states_cpu = hidden_states.to(device="cpu", dtype=torch.float16).contiguous()
        elif hidden_states.dtype != torch.float16:
            hidden_states_cpu = hidden_states.to(dtype=torch.float16).contiguous()
        else:
            hidden_states_cpu = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()

        pe_w = self._rpu_vision_patch_embed_w  # CPU fp16
        pe_b = self._rpu_vision_patch_embed_b  # CPU fp16
        embed_cpu = torch.nn.functional.linear(hidden_states_cpu, pe_w, pe_b)  # CPU fp16 [N_total, hidden]
        embed_packed = embed_cpu.to(device="rpu", dtype=torch.float16, non_blocking=False).contiguous()

        # ----- fast_pos_embed_interpolate (CPU fp16 → RPU + add) (F32) --------
        pos_embeds_cpu = type(self).fast_pos_embed_interpolate(self, grid_thw_cpu)  # [N_total, hidden] CPU fp16
        pos_embeds = pos_embeds_cpu.to(device=embed_packed.device, dtype=embed_packed.dtype).contiguous()
        embed_packed = embed_packed + pos_embeds  # rpu_add

    # ----- Per-image encoder loop -----------------------------------------
    keepalive = self._rpu_vision_position_idx_keepalive
    k_caches = [cache.k_caches[i] for i in range(num_layers)]
    v_caches = [cache.v_caches[i] for i in range(num_layers)]

    last_hidden_per_image: list[torch.Tensor] = []
    graph_requests = []
    if not getattr(self, "_rpu_vision_graph_disable", False):
        for (_, n_i, _), vision_plan in planned_groups:
            embed_shape = (n_i, *tuple(embed_packed.shape[1:]))
            key = (
                n_i, self._rpu_vision_hidden_size, num_layers,
                temporal_num_frames, camera_batch_count, direct_fusion,
                compact_input, embed_shape, vision_plan.physical_plan_digest,
                int(getattr(self, "_execution_generation", 0)),
            )
            sig = (self._rpu_vision_graph_sig if self._rpu_vision_graph_key == key
                   else _vision_group_graph_signature(key, vision_plan))
            # A later missing child must reject before any image writes KV,
            # position keepalives or the caller's fusion target.
            if self._rpu_vision_graph_cache.is_frozen():
                self._rpu_vision_graph_cache.get_or_create(sig)
            graph_requests.append((key, sig))
    # In-graph merger runs as a post_layers_fn inside each image's forward; we pull
    # its per-image output via the getter and cat (merge-then-cat). RPU_VISION_MERGER=0
    # leaves this off and falls back to the CPU merger over the cat'd last_hidden.
    step0_pos_all = None
    if step0_on:
        cached_grid = self._rpu_vision_step0_pos_grid
        cached_frames = self._rpu_vision_step0_pos_temporal_num_frames
        cached_camera_batch = self._rpu_vision_step0_pos_camera_batch_count
        if (
            cached_grid is None
            or cached_frames != temporal_num_frames
            or cached_camera_batch != camera_batch_count
            or not torch.equal(cached_grid, grid_thw_cpu)
        ):
            pos_cpu = type(self).fast_pos_embed_interpolate(self, grid_thw_cpu)
            self._rpu_vision_step0_pos = pos_cpu.to(
                device="rpu", dtype=torch.float16
            ).contiguous()
            self._rpu_vision_step0_pos_grid = grid_thw_cpu.clone()
            self._rpu_vision_step0_pos_temporal_num_frames = temporal_num_frames
            self._rpu_vision_step0_pos_camera_batch_count = camera_batch_count
        step0_pos_all = self._rpu_vision_step0_pos

    merged_per_image: list[torch.Tensor] = []

    offset = 0
    for i, ((grid_start, n_i, output_patches), vision_plan) in enumerate(
        planned_groups
    ):
        plan_metadata = dict(
            vision_plan.selected.stage_tuple.physical_metadata
        )
        embed_i = embed_packed.narrow(0, offset, n_i).contiguous()

        pos_idx_cpu = _cached_vision_position_idx_cpu(
            self,
            grid_thw_cpu[
                grid_start : grid_start + temporal_num_frames * camera_batch_count
            ],
            spatial_merge_size,
            max_hw=int(self._rpu_vision_freq_cos.size(0)),
        )
        if pos_idx_cpu.size(0) != n_i:
            raise RuntimeError(
                f"position_idx rows {pos_idx_cpu.size(0)} != "
                f"image_{i} patches {n_i}"
            )
        if getattr(self, "_rpu_vision_rope_disable", False):
            keepalive.narrow(0, 0, n_i).zero_()
        else:
            keepalive.narrow(0, 0, n_i).copy_(pos_idx_cpu.to(keepalive.device))

        cache.reset_to_position(0)

        embed_i_3d = embed_i.unsqueeze(0).contiguous()
        if step0_pos_all is None:
            step0_pos_i = None
        else:
            step0_pos_i = step0_pos_all.narrow(0, offset, n_i)
        op_args = (
            handle, embed_i_3d, k_caches, v_caches, n_i,
            step0_pos_i,
            fusion_target if direct_fusion else None,
            [int(fusion_run_starts[i])] if direct_fusion else [],
            temporal_num_frames,
            camera_batch_count,
            int(plan_metadata["stage_plan_fingerprint_hi"]),
            int(plan_metadata["stage_plan_fingerprint_lo"]),
            int(plan_metadata["layout_hash_hi"]),
            int(plan_metadata["layout_hash_lo"]),
            vision_plan.selected.stage_tuple.physical_descriptor,
        )

        active_sig = None
        if getattr(self, "_rpu_vision_graph_disable", False):
            # Diagnostic fallback: bypass the retained GraphCache but keep the
            # complete physical manifest under an active one-shot lifecycle.
            try:
                out_3d = _run_vision_diagnostic_oneshot(self, op_args)
            except BaseException:
                _evict_vision_graph(self)
                raise
        else:
            key, active_sig = graph_requests[i]
            graph_cache = self._rpu_vision_graph_cache
            if (graph_cache.lookup(active_sig) is None
                    and graph_cache.size() >= graph_cache.max_entries()):
                # Capacity is bounded. A new working set is a warmup transition,
                # never an eviction/online BUILD under a frozen READY cache.
                if graph_cache.is_frozen():
                    raise RuntimeError("Qwen3.5 Vision GraphCache READY miss")
                if graph_cache.max_entries() == 1:
                    _evict_vision_graph(self)
                else:
                    # The complete request was already planned. Retire only
                    # graphs, retaining those still-valid bounded plan memos.
                    for entry in graph_cache.snapshot():
                        graph_cache.evict(types.SimpleNamespace(_impl=entry.signature))
            self._rpu_vision_graph_key = key
            self._rpu_vision_graph_sig = active_sig
            try:
                with self._rpu_vision_graph_cache.capture(active_sig):
                    out_3d = torch.ops.rpu.qwen3_5_vision_forward(*op_args)
            except BaseException:
                _evict_vision_graph(self, active_sig)
                raise

        if vision_plan.selected is not None:
            resolved = int(
                torch.ops.rpu.qwen3_5_vision_get_resolved_chunk_size(handle)
            )
            planned = int(vision_plan.selected.stage_tuple.compute_chunk)
            if resolved != planned:
                if active_sig is not None:
                    _evict_vision_graph(self, active_sig)
                raise RuntimeError(
                    "Qwen3.5 vision dispatch chunk drifted from its A6 plan: "
                    f"planned={planned}, resolved={resolved}"
                )
        self._rpu_vision_has_dispatched = True

        if not direct_fusion:
            # .clone() is REQUIRED, not defensive. Graph mode returns a per-shape
            # model buffer, so a second same-size image overwrites the first view.
            # The direct path does not consume last_hidden_state and skips this copy.
            live = out_3d.squeeze(0)
            last_hidden_per_image.append(live.clone())

        if merger_on and not direct_fusion:
            # The in-graph merger already ran (post_layers_fn); pull this image's
            # merged rows. .clone() is REQUIRED for the SAME reason as out_3d above —
            # merged_buf_ is a stable per-model buffer overwritten by the next image's
            # forward, so appending a view would let image i+1 clobber image i.
            merged_i = torch.ops.rpu.qwen3_5_vision_get_merger_out(
                handle)  # [1, n_i/sm², out_hidden] RPU fp16
            merged_per_image.append(merged_i.squeeze(0).clone())
        offset += n_i

    # ----- Concatenate per-image outputs ----------------------------------
    if direct_fusion:
        last_hidden = None
        merged = None
    else:
        last_hidden = (last_hidden_per_image[0] if len(last_hidden_per_image) == 1
                       else torch.cat(last_hidden_per_image, dim=0))
        if merger_on:
            # merge-then-cat: every n_i is a multiple of sm², so groups never
            # straddle images. Keep the standalone pooler output on CPU fp32.
            merged = (merged_per_image[0] if len(merged_per_image) == 1
                      else torch.cat(merged_per_image, dim=0)).detach().cpu().float()
        else:
            merged = self.merger(last_hidden.detach().cpu().float())

    torch.ops.rpu.spm_alloc_reset_temporary()

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden,
        pooler_output=merged,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vision → text fusion (§7.9) — called by _rpu_qwen3_5_forward when pixel_values
# is not None. Mirrors HF Qwen3_5Model.forward's fusion semantics + 3D M-RoPE,
# minus DeepStack; contiguous image spans use direct-to-final-DDR merger output.
# ─────────────────────────────────────────────────────────────────────────────

def _lazy_vision_graph_options(text_state, *, prepared_profile=None):
    """Carry the admitted cold owner policy without consulting the environment."""
    admitted = getattr(text_state, "vision_arena_admitted", None)
    if admitted is None:
        if prepared_profile is not None:
            raise RuntimeError(
                "Qwen3.5-MoE Vision has no cold arena admission record"
            )
        return {}
    if admitted is not True:
        if prepared_profile is not None:
            raise RuntimeError(
                "Qwen3.5-MoE Vision requires its prepared cold graph arenas "
                "before to_rpu; reload the model"
            )
        raise RuntimeError(
            "Qwen3.5 9B VL requires QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1 "
            "before to_rpu; reload the model to prepare its cold graph arenas")
    if prepared_profile is None:
        graph_cache = getattr(text_state, "graph_cache", None)
        policy = getattr(graph_cache, "runtime_policy", None)
        topology = text_state.execution_topology
        if (policy.graph_arena_count != 4
                or policy.execution_core_count != topology.num_cores
                or policy.qwen35_legacy_27b_sdk_budget):
            raise RuntimeError(
                "Qwen3.5 9B VL cold graph arena ownership is incomplete"
            )
    else:
        _validate_qwen3_5_moe_text_topology(
            text_state, prepared_profile, text_state.execution_topology
        )
        vision_policy = _validate_qwen3_5_moe_vision_policies(
            text_state, prepared_profile
        )
        return {"_graph_runtime_policy": vision_policy,
                "_cold_numeric_opt_in": True}
    vision_policy = getattr(text_state, "vision_graph_policy", None)
    if vision_policy is None:
        from dataclasses import replace

        vision_policy = (policy if policy.execution_core_count == topology.attn_tp
                         else replace(policy, execution_core_count=topology.attn_tp))
        text_state.vision_graph_policy = vision_policy
    return {"_graph_runtime_policy": vision_policy,
            "_cold_numeric_opt_in": True}


def _install_lazy_qwen3_5_vision(model, text_state, *, prepared_profile=None,
                               _text_bridge=None):
    """Reuse the transactional install and immutable cold controls."""
    handle = install_qwen3_5_vision_for_rpu(
        model,
        execution_config=getattr(model, "_rpu_execution", None),
        _resolved_options=getattr(text_state, "vision_control_options", None),
        _text_bridge=_text_bridge,
        **_lazy_vision_graph_options(text_state, prepared_profile=prepared_profile),
    )
    model.model.visual._execution_generation = int(
        getattr(text_state, "execution_generation", 0))
    return handle


def _prepare_qwen3_5_multimodal_spm(model, input_ids, pixel_values,
                                   image_grid_thw, execution_len, *, _text_bridge=None):
    """Fix both owners' Persistent floor before their final cold plans.

    Only an image request installs the vision tower. No vision forward, Temp
    allocation, preload or Graph runs here. A text owner that already executed
    retains its existing fixed slots.
    """
    fusion = model.model
    vision = fusion.visual
    if _text_bridge is None:
        state = fusion.language_model._rpu_qwen3_5
        text_prefix = "qwen3_5"
    else:
        state = _validate_qwen3_5_moe_vision_text_bridge(
            _text_bridge, model, require_vision_policy=True)
        text_prefix = "qwen3_5_moe"
    identity = (int(state.handle), getattr(vision, "_rpu_vision_handle", None))
    if identity == getattr(state, "multimodal_spm_identity", None):
        return False
    grid, merge = _validate_vision_grid(
        image_grid_thw.detach().cpu(), vision.config.spatial_merge_size,
        "Qwen3.5 vision preparation")
    patches = [t * h * w for t, h, w in grid]
    if pixel_values.ndim != 2 or pixel_values.shape[0] != sum(patches):
        raise ValueError("Qwen3.5 vision preparation: patch input does not match grid_thw")
    _visual_run_starts(input_ids.detach().cpu(), int(model.config.image_token_id),
                       [n // (merge * merge) for n in patches])
    # A new peer changes the feasible domain; the caller must leave READY
    # before installing it, just as for a new physical graph shape.
    for owner, names in ((state, ("graph_cache", "prefill_graph_cache",
                                 "prefill_debug_graph_cache")),
                         (vision, ("_rpu_vision_graph_cache",))):
        for name in names:
            graph = getattr(owner, name, None)
            if graph is not None and graph.is_frozen():
                raise RuntimeError("Qwen3.5 READY cannot install a new vision SPM peer; begin warmup first")
    vision_options = ({} if _text_bridge is None else {
        "prepared_profile": _text_bridge.prepared_profile,
        "_text_bridge": _text_bridge,
    })
    vision_handle = _install_lazy_qwen3_5_vision(model, state, **vision_options)
    # Existing graphs retain absolute temporary SPM addresses. Retire them
    # before a new peer lowers the floor, including the earlier text-only
    # decode graph and every normal/debug prefill owner.
    _clear_qwen35_graphs((state,), vision)
    from rpu_backend.api._execution import prepared_execution_plan_cache

    for owner, prefix, stages in ((fusion.language_model, text_prefix, ("prefill", "decode")),
                                  (vision, "qwen3_5_vision", ("vision",))):
        for stage in stages:
            prepared_execution_plan_cache(owner, stage, None, prefix).clear()
    state.prefill_graph_sig = state.prefill_debug_graph_sig = None
    state.prefill_plan = state.last_a6_plan = state.decode_plan = None
    state.decode_stage_descriptor = None
    vision._rpu_vision_graph_key = vision._rpu_vision_graph_sig = None
    vision._rpu_vision_last_a6_plan = None
    vision._rpu_vision_a6_plans = {}
    if torch.ops.rpu.qwen3_5_vision_get_resolved_chunk_size(vision_handle) == 0:
        torch.ops.rpu.qwen3_5_vision_prepare_persistent_spm(
            vision_handle, max(patches))
    if _text_bridge is not None:
        if not getattr(state, "has_dispatched", False):
            torch.ops.rpu.qwen3_5_moe_prepare_persistent_spm(state.handle, execution_len)
    elif torch.ops.rpu.qwen3_5_get_resolved_chunk_size(state.handle) == 0:
        torch.ops.rpu.qwen3_5_prepare_persistent_spm(state.handle, execution_len)
    # Publish only after both preparations succeed; ordinary replay does not
    # repeat the barrier or clear any graphs/plans.
    state.multimodal_spm_identity = (int(state.handle), int(vision_handle))
    return True


def fuse_visual_embeds(model, hidden, input_ids, pixel_values, image_grid_thw,
                       *, attention_mask=None, video_grid_thw=None,
                       pixel_values_videos=None, mm_token_type_ids=None,
                       past_key_values=None,
                       _text_bridge: Qwen3_5MoeVisionTextBridge | None = None,
                       **kw):
    """Run the vision tower + fuse into ``hidden``; return ``(hidden, position_ids)``.

    ``model`` is the top ``Qwen3_5ForConditionalGeneration``; ``model.model`` is the
    ``Qwen3_5Model`` fusion (owns ``visual`` + ``get_image_features`` +
    ``compute_3d_position_ids``). ``hidden`` is ``[1, seq, hidden]`` embeds (fp16,
    RPU). Replicates HF ``Qwen3_5Model.forward`` (modeling L1652-1683) minus
    DeepStack:
      1. vision encode and write the merger output directly into the contiguous
         image-token runs of ``hidden`` on RPU.
      2. Fall back to CPU ``masked_scatter`` only for fragmented token layouts or
         when the RPU merger is disabled.
      3. 3D ``position_ids`` via ``compute_3d_position_ids`` (get_rope_index) →
         return ``position_ids[:, 0, :]`` = ``[3, seq]`` for the prefill M-RoPE.
    """
    if kw:
        raise TypeError(
            "Qwen3.5 vision fuse received unsupported keyword argument(s): "
            + ", ".join(sorted(kw))
        )
    if image_grid_thw is None:
        raise ValueError("Qwen3.5 vision fuse: image_grid_thw must accompany pixel_values.")
    if video_grid_thw is not None or pixel_values_videos is not None:
        raise NotImplementedError("Qwen3.5 vision fuse: video inputs not supported yet.")
    if input_ids is None:
        raise ValueError("Qwen3.5 vision fuse: input_ids required to locate image tokens.")
    if pixel_values is None:
        raise ValueError("Qwen3.5 vision fuse: pixel_values is required.")
    if mm_token_type_ids is None:
        raise ValueError(
            "Qwen3.5 vision fuse: mm_token_type_ids is required for M-RoPE "
            "(returned by the processor alongside input_ids).")

    fusion = model.model  # Qwen3_5Model
    vision_model = fusion.visual
    if _text_bridge is None:
        language = getattr(fusion, "language_model", fusion)
        text_state = getattr(language, "_rpu_qwen3_5", None)
        prepared_profile = None
    else:
        text_state = _validate_qwen3_5_moe_vision_text_bridge(
            _text_bridge, model, require_vision_policy=True
        )
        prepared_profile = _text_bridge.prepared_profile

    image_token_id = int(model.config.image_token_id)
    input_ids_cpu = input_ids.to("cpu")
    visual_mask = input_ids_cpu == image_token_id
    n_visual_tokens = int(visual_mask.sum())
    split_sizes = (
        image_grid_thw.detach().cpu().prod(-1)
        // int(vision_model.config.spatial_merge_size) ** 2
    ).tolist()
    if sum(split_sizes) != n_visual_tokens:
        raise RuntimeError(
            f"Qwen3.5 vision fuse: image embeds rows {sum(split_sizes)} "
            f"!= image token count {n_visual_tokens}"
        )

    # Lazy install: first image forward swizzles + installs the vision tower.
    # Keeps text-only to_rpu byte-identical (BC — spec §9.1); vision touches the
    # RPU only when an image actually arrives.
    # The installer is a cheap locked no-op after commit and also validates
    # that an apparent existing handle still has a live finalizer.
    # Vision is lazy, but its immutable cold controls belong to the same
    # model-level execution snapshot as text; do not silently fall back to
    # environment/default values on the first image forward.
    _install_lazy_qwen3_5_vision(
        model, text_state, prepared_profile=prepared_profile,
        _text_bridge=_text_bridge)

    run_starts = _visual_run_starts(input_ids_cpu, image_token_id, split_sizes)
    direct_fusion = getattr(vision_model, "_rpu_vision_has_merger", False) and run_starts is not None
    if direct_fusion:
        hidden = hidden.to(device="rpu", dtype=torch.float16).contiguous()
        with torch.no_grad():
            vision_model(
                pixel_values.type(vision_model.dtype), grid_thw=image_grid_thw,
                return_dict=True, _rpu_fusion_target=hidden,
                _rpu_fusion_run_starts=run_starts)
    else:
        with torch.no_grad():
            image_outputs = fusion.get_image_features(
                pixel_values, image_grid_thw, return_dict=True)
        image_embeds = torch.cat(list(image_outputs.pooler_output), dim=0)
        hidden_cpu = hidden.to("cpu", dtype=torch.float16)
        visual_mask_expanded = visual_mask.unsqueeze(-1).expand_as(hidden_cpu)
        hidden_cpu = hidden_cpu.masked_scatter(
            visual_mask_expanded, image_embeds.to("cpu", dtype=torch.float16))
        hidden = hidden_cpu.to(device="rpu", dtype=torch.float16).contiguous()

    # ----- 3. 3D position_ids via compute_3d_position_ids (get_rope_index) ----
    # Image input is admitted only at cache position 0, so this takes the
    # get_rope_index path and stashes fusion.rope_deltas for decode.
    position_ids = fusion.compute_3d_position_ids(
        input_ids=input_ids_cpu,
        inputs_embeds=hidden,
        image_grid_thw=image_grid_thw.detach().cpu(),
        video_grid_thw=None,
        attention_mask=attention_mask.to("cpu") if attention_mask is not None else None,
        past_key_values=past_key_values,
        mm_token_type_ids=mm_token_type_ids.to("cpu"),
    )  # [3, batch, seq]
    # batch=1 ⇒ [3, seq] token-indexed T/H/W for the prefill partial M-RoPE.
    position_ids = position_ids[:, 0, :].contiguous()

    # W2: push the decode M-RoPE offset (HF rope_deltas, set by compute_3d_position_ids)
    # to the C++ text model. DECODE after this image indexes the static cos_ at
    # position + delta (prefill uses the per-token prefill_cos_ table, no delta). Set
    # once here during the image prefill; it persists on the C++ model for the decodes.
    _publish_qwen3_5_vision_mrope_delta(fusion, _text_bridge=_text_bridge)

    return hidden, position_ids


def _publish_qwen3_5_vision_mrope_delta(
    fusion, *, _text_bridge: Qwen3_5MoeVisionTextBridge | None = None
) -> None:
    """Publish one image-prefill delta to the matching installed text owner."""
    rope_deltas = getattr(fusion, "rope_deltas", None)
    if rope_deltas is None:
        if _text_bridge is not None:
            raise RuntimeError(
                "Qwen3.5-MoE image prefill produced no M-RoPE delta"
            )
        return
    flattened = rope_deltas.flatten()
    if _text_bridge is not None:
        if flattened.numel() != 1:
            raise RuntimeError("Qwen3.5-MoE Vision supports batch-one M-RoPE only")
        if _text_bridge.fusion is not fusion:
            raise RuntimeError("Qwen3.5-MoE Vision bridge fusion owner changed")
        _text_bridge.set_mrope_position_delta(int(flattened[0].item()))
        return
    inner = getattr(fusion, "language_model", fusion)
    state = getattr(inner, "_rpu_qwen3_5", None)
    if state is not None:
        torch.ops.rpu.qwen3_5_set_mrope_position_delta(
            state.handle, int(flattened[0].item())
        )
