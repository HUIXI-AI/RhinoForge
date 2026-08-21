"""RPU runtime for the standalone InternVLA-N1 Lumina NextDiT action core.

The host keeps the small timestep/caption glue while RPU owns the 12
transformer blocks and final continuous LayerNorm/projection. The boundary
matches the reference trajectory semantics and retains per-block numerical
probes for correctness runs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any
import weakref

import torch

from rpu_backend.runtime.weights import tp_col_swizzle_mc_weight, tp_row_swizzle_mc_weight
from rpu_backend.runtime import UnsupportedModelError
from rpu_backend.runtime.control import rpu_env_bool

from ._policy import require_controlled_evaluation, verify_asset_manifest


NUM_LAYERS = 12
NUM_HEADS = 6
HEAD_DIM = 64
HIDDEN = 384
FF_INTERMEDIATE = 1536
SUPPORTED_FF_INTERMEDIATES = (1024, FF_INTERMEDIATE)
ACTION_SEQ = 32
ACTION_DIM = 3
ACTION_PAD = 16
ATTN_TP = 6
MLP_TP = 8
EPS = 1e-5
CONDITION_CACHE_SIZE = 8


def _destroy_nextdit_handle(handle: int) -> None:
    try:
        torch.ops.rpu.internvla_nextdit_destroy(handle)
    except Exception:
        pass


def _configure_nextdit_handle(configure) -> int:
    """Create/configure one native handle, releasing it on failure."""
    handle = torch.ops.rpu.internvla_nextdit_create()
    configured = False
    try:
        configure(handle)
        configured = True
        return handle
    finally:
        if not configured:
            _destroy_nextdit_handle(handle)


class _FixedStepPrepare(torch.nn.Module):
    """Traceable fixed-width CPU glue for a batch of scheduler steps."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(
        self,
        time_embedding: torch.Tensor,
        caption_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_step = torch.nn.functional.linear(
            torch.nn.functional.silu(time_embedding + caption_embedding),
            self.weight,
            self.bias,
        )
        modulation_elements = NUM_LAYERS * 4 * HIDDEN
        modulation = raw_step[:, :modulation_elements].view(
            -1, NUM_LAYERS, 4, HIDDEN
        )
        modulation[:, :, 0].add_(1)
        modulation[:, :, 1].tanh_()
        modulation[:, :, 2].add_(1)
        modulation[:, :, 3].tanh_()
        final_scale = raw_step[:, modulation_elements:].add_(1)
        return modulation, final_scale.contiguous()


def _lumina_model(traj_dit: torch.nn.Module) -> torch.nn.Module:
    model = getattr(traj_dit, "model", traj_dit)
    required = ("caption_projection", "time_caption_embed", "layers", "norm_out")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(f"not a Lumina NextDiT model; missing {missing}")
    return model


def _validate_geometry(traj_dit: torch.nn.Module) -> torch.nn.Module:
    model = _lumina_model(traj_dit)
    if len(model.layers) != NUM_LAYERS:
        raise ValueError(f"RPU slice requires {NUM_LAYERS} layers, got {len(model.layers)}")
    layer0 = model.layers[0]
    geometry = (
        layer0.attn1.to_q.in_features,
        layer0.attn1.heads,
        layer0.head_dim,
    )
    expected = (HIDDEN, NUM_HEADS, HEAD_DIM)
    if geometry != expected:
        raise ValueError(f"RPU slice requires geometry {expected}, got {geometry}")
    ff_intermediates = {
        layer.feed_forward.linear_1.out_features for layer in model.layers
    }
    if (
        len(ff_intermediates) != 1
        or next(iter(ff_intermediates)) not in SUPPORTED_FF_INTERMEDIATES
    ):
        raise ValueError(
            f"RPU slice requires FF intermediate in {SUPPORTED_FF_INTERMEDIATES}, "
            f"got {sorted(ff_intermediates)}"
        )
    return model


@torch.no_grad()
def prepare_nextdit_condition_cpu(
    traj_dit: torch.nn.Module,
    condition: torch.Tensor,
    context_norm_weight: torch.Tensor | None = None,
    cross_kv_weight: torch.Tensor | None = None,
    cross_norm_weight: torch.Tensor | None = None,
    cross_norm_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Project one condition and produce exact head-major cross K/V on CPU."""

    model = _validate_geometry(traj_dit)
    if condition.device.type != "cpu" or condition.ndim != 3 or condition.shape[0] != 1:
        raise ValueError("condition must be a CPU tensor with shape [1,S,384]")
    parameter = next(model.parameters())
    condition = condition.to(dtype=parameter.dtype)
    caption = model.caption_projection(condition)

    seq = caption.shape[1]
    cross_k: list[torch.Tensor] = []
    cross_v: list[torch.Tensor] = []
    if context_norm_weight is not None and cross_kv_weight is not None:
        norm = model.layers[0].norm1_context
        variance = caption.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normalized = caption * torch.rsqrt(variance + norm.eps)
        contexts = normalized.expand(NUM_LAYERS, -1, -1) * context_norm_weight[:, None, :]
        key_values = torch.bmm(contexts, cross_kv_weight.transpose(1, 2))
        if cross_norm_weight is not None and cross_norm_bias is not None:
            keys = torch.vmap(
                lambda value, weight, bias: torch.nn.functional.layer_norm(
                    value,
                    (HIDDEN,),
                    weight,
                    bias,
                    model.layers[0].attn2.norm_k.eps,
                )
            )(key_values[:, :, :HIDDEN], cross_norm_weight, cross_norm_bias)
            cross_k = list(
                keys.view(NUM_LAYERS, seq, NUM_HEADS, HEAD_DIM)
                .permute(0, 2, 1, 3)
                .contiguous()
                .unbind(0)
            )
            cross_v = list(
                key_values[:, :, HIDDEN:]
                .view(NUM_LAYERS, seq, NUM_HEADS, HEAD_DIM)
                .permute(0, 2, 1, 3)
                .contiguous()
                .unbind(0)
            )
            return caption, cross_k, cross_v
        for layer_index, layer in enumerate(model.layers):
            key = key_values[layer_index : layer_index + 1, :, :HIDDEN]
            if layer.attn2.norm_k is not None:
                key = layer.attn2.norm_k(key)
            value = key_values[layer_index, :, HIDDEN:]
            cross_k.append(
                key.view(1, seq, NUM_HEADS, HEAD_DIM)[0]
                .permute(1, 0, 2)
                .contiguous()
            )
            cross_v.append(
                value.view(seq, NUM_HEADS, HEAD_DIM)
                .permute(1, 0, 2)
                .contiguous()
            )
    else:
        for layer in model.layers:
            context = layer.norm1_context(caption)
            key = layer.attn2.to_k(context)
            if layer.attn2.norm_k is not None:
                key = layer.attn2.norm_k(key)
            value = layer.attn2.to_v(context)
            cross_k.append(
                key.view(1, seq, NUM_HEADS, HEAD_DIM)[0]
                .permute(1, 0, 2)
                .contiguous()
            )
            cross_v.append(
                value.view(1, seq, NUM_HEADS, HEAD_DIM)[0]
                .permute(1, 0, 2)
                .contiguous()
            )
    return caption, cross_k, cross_v


@torch.no_grad()
def prepare_nextdit_step_cpu(
    traj_dit: torch.nn.Module,
    timestep: torch.Tensor,
    caption: torch.Tensor,
    caption_embedding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``temb``, block modulation and final-norm scale for one step."""

    model = _validate_geometry(traj_dit)
    if caption.device.type != "cpu" or caption.ndim != 3 or caption.shape[0] != 1:
        raise ValueError("caption must be a CPU tensor with shape [1,S,384]")
    if timestep.numel() != 1:
        raise ValueError("the first RPU slice supports one trajectory per call")
    timestep = timestep.reshape(1).to(device="cpu")
    if caption_embedding is None:
        encoder_mask = torch.ones((1, caption.shape[1]), device="cpu", dtype=torch.long)
        temb = model.time_caption_embed(timestep, caption, encoder_mask)
    else:
        time_embedder = model.time_caption_embed
        time_freq = time_embedder.time_proj(timestep)
        time_embed = time_embedder.timestep_embedder(
            time_freq.to(dtype=caption.dtype)
        )
        temb = time_embed + caption_embedding

    packed = []
    for layer in model.layers:
        scale_msa, gate_msa, scale_mlp, gate_mlp = layer.norm1.linear(
            layer.norm1.silu(temb)
        ).chunk(4, dim=1)
        packed.append(
            torch.stack(
                (
                    1 + scale_msa[0],
                    gate_msa[0].tanh(),
                    1 + scale_mlp[0],
                    gate_mlp[0].tanh(),
                ),
                dim=0,
            )
        )
    final_scale = 1 + model.norm_out.linear_1(model.norm_out.silu(temb))[0]
    return temb, torch.stack(packed, dim=0).contiguous(), final_scale.contiguous()


@torch.no_grad()
def prepare_nextdit_steps_cpu(
    traj_dit: torch.nn.Module,
    timesteps: torch.Tensor,
    caption: torch.Tensor,
    caption_embedding: torch.Tensor,
    modulation_weight: torch.Tensor | None = None,
    modulation_bias: torch.Tensor | None = None,
    time_embedding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare all scheduler-step modulation tensors in one CPU batch."""

    model = _validate_geometry(traj_dit)
    timesteps = timesteps.reshape(-1).to(device="cpu")
    if time_embedding is None:
        time_embedder = model.time_caption_embed
        time_freq = time_embedder.time_proj(timesteps)
        time_embedding = time_embedder.timestep_embedder(
            time_freq.to(dtype=caption.dtype)
        )
    temb = time_embedding + caption_embedding

    if modulation_weight is None or modulation_bias is None:
        modulation_weight = torch.cat(
            [layer.norm1.linear.weight for layer in model.layers]
            + [model.norm_out.linear_1.weight],
            dim=0,
        )
        modulation_bias = torch.cat(
            [layer.norm1.linear.bias for layer in model.layers]
            + [model.norm_out.linear_1.bias],
            dim=0,
        )
    raw_step = torch.nn.functional.linear(
        model.layers[0].norm1.silu(temb),
        modulation_weight,
        modulation_bias,
    )
    modulation_elements = NUM_LAYERS * 4 * HIDDEN
    raw_modulation = raw_step[:, :modulation_elements].view(
        timesteps.numel(), NUM_LAYERS, 4, HIDDEN
    )
    raw_modulation[:, :, 0].add_(1)
    raw_modulation[:, :, 1].tanh_()
    raw_modulation[:, :, 2].add_(1)
    raw_modulation[:, :, 3].tanh_()
    final_scale = raw_step[:, modulation_elements:].add_(1)
    return raw_modulation, final_scale.contiguous()


def _to_rpu_half(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float16).to("rpu").contiguous()


def _col(weight: torch.Tensor, cores: int) -> torch.Tensor:
    return _to_rpu_half(tp_col_swizzle_mc_weight(weight.detach().cpu().half(), cores))


def _row(weight: torch.Tensor, cores: int) -> torch.Tensor:
    return _to_rpu_half(tp_row_swizzle_mc_weight(weight.detach().cpu().half(), cores))


@dataclass
class _ConditionState:
    source: torch.Tensor
    fingerprint: tuple[Any, ...]
    caption: torch.Tensor
    caption_embedding: torch.Tensor
    cross_context_rpu: torch.Tensor


@dataclass
class _StepState:
    modulation_rpu: torch.Tensor
    final_scale_rpu: torch.Tensor


@dataclass
class _StepBatchState:
    modulation_rpu: torch.Tensor
    final_scale_rpu: torch.Tensor


class NextDiTRPURuntime:
    """Own one live RPU NextDiT policy and its graph/cache state."""

    def __init__(
        self,
        traj_dit: torch.nn.Module,
        *,
        max_cross_seq_len: int = 320,
        collect_layer_outputs: bool = False,
        action_encoder: torch.nn.Linear | None = None,
        action_decoder: torch.nn.Linear | None = None,
        action_position: torch.Tensor | None = None,
        asset_manifest: Mapping[str | Path, str] | None = None,
        asset_paths: Sequence[str | Path] | None = None,
    ) -> None:
        require_controlled_evaluation(asset_manifest)
        if not asset_paths:
            raise UnsupportedModelError(
                "InternVLA-N1 NextDiT RPU requires asset_paths for SHA-256 verification"
            )
        verify_asset_manifest(asset_paths, asset_manifest)
        fold_norm_modulation = rpu_env_bool(
            "RPU_INTERNVLA_NEXTDIT_FOLD_NORM_MOD",
            cpp_mirror="src/fused/rpu_internvla_nextdit_model.cpp",
        )
        step_prepare_on_rpu = rpu_env_bool(
            "RPU_INTERNVLA_NEXTDIT_STEP_PREP_RPU"
        )
        fused_qkq = rpu_env_bool("RPU_INTERNVLA_NEXTDIT_FUSED_QKQ")
        model = _validate_geometry(traj_dit)
        if next(model.parameters()).device.type != "cpu":
            raise ValueError("build the RPU runtime from an unswizzled CPU NextDiT model")
        if getattr(traj_dit, "_rpu_nextdit_runtime", None) is not None:
            raise RuntimeError("this NextDiT instance already has an RPU runtime")

        import rpu_backend
        from rpu_backend.api import RPUCache

        if not hasattr(torch.ops.rpu, "internvla_nextdit_create"):
            raise RuntimeError("rpu_backend was not rebuilt with the InternVLA NextDiT operator")

        self._model = model
        self._ff_intermediate = model.layers[0].feed_forward.linear_1.out_features
        self._owner = traj_dit
        self._graph_cache = rpu_backend.graph.GraphCache()
        self._self_cache = RPUCache(
            num_layers=NUM_LAYERS,
            batch_size=1,
            max_seq_len=ACTION_SEQ,
            num_kv_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            attn_tp=ATTN_TP,
        )
        self._packed_self_cache = RPUCache(
            num_layers=NUM_LAYERS,
            batch_size=1,
            max_seq_len=4 * ACTION_SEQ,
            num_kv_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            attn_tp=ATTN_TP,
        )
        self._cross_cache = RPUCache(
            num_layers=NUM_LAYERS,
            batch_size=1,
            max_seq_len=max_cross_seq_len,
            num_kv_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            attn_tp=ATTN_TP,
        )
        self._max_cross_seq_len = max_cross_seq_len
        self._collect_layer_outputs = bool(collect_layer_outputs)
        self._action_encoder_enabled = action_encoder is not None or action_decoder is not None
        if self._action_encoder_enabled:
            if action_encoder is None or action_decoder is None:
                raise ValueError("RPU action I/O requires both encoder and decoder")
            if action_encoder.in_features != ACTION_DIM or action_encoder.out_features != HIDDEN:
                raise ValueError("RPU action encoder requires Linear(3,384)")
            if action_decoder.in_features != HIDDEN or action_decoder.out_features != ACTION_DIM:
                raise ValueError("RPU action decoder requires Linear(384,3)")
            if action_position is None or action_position.shape != (ACTION_SEQ, HIDDEN):
                raise ValueError("RPU action position table must have shape [32,384]")
            self._action_encoder = action_encoder
            self._action_decoder = action_decoder
            self._action_position = action_position.detach().cpu()
        else:
            self._action_encoder = None
            self._action_decoder = None
            self._action_position = torch.zeros(ACTION_SEQ, HIDDEN)
        self._conditions: list[_ConditionState] = []
        self._next_condition_slot = 0
        self._steps: dict[tuple[int, int], _StepState] = {}
        self._step_batches: dict[
            tuple[int, tuple[int, ...], bool], _StepBatchState
        ] = {}
        self._step_time_embeddings: dict[tuple[int, ...], torch.Tensor] = {}
        self._condition_batch_source: torch.Tensor | None = None
        self._condition_batch_version = -1
        self._condition_batch_bindings: list[tuple[int, _ConditionState]] = []
        self._graph_signatures: dict[tuple[Any, ...], Any] = {}
        self._active_cross_condition_index: int | None = None
        cross_context_elements = max_cross_seq_len * HIDDEN
        self._cross_context_slots = [
            torch.empty(
                cross_context_elements, dtype=torch.float16, device="rpu"
            )
            for _ in range(CONDITION_CACHE_SIZE)
        ]
        self._step_modulation_slots = [
            torch.empty(
                (10, NUM_LAYERS, 4, HIDDEN),
                dtype=torch.float16,
                device="rpu",
            )
            for _ in range(CONDITION_CACHE_SIZE)
        ]
        self._step_final_scale_slots = [
            torch.empty((10, HIDDEN), dtype=torch.float16, device="rpu")
            for _ in range(CONDITION_CACHE_SIZE)
        ]
        self._step_activated_slots = [
            torch.empty((10, HIDDEN), dtype=torch.float16, device="rpu")
            for _ in range(CONDITION_CACHE_SIZE)
        ]
        self._step_modulation_weight = torch.cat(
            [layer.norm1.linear.weight for layer in model.layers]
            + [model.norm_out.linear_1.weight],
            dim=0,
        )
        self._step_modulation_bias = torch.cat(
            [layer.norm1.linear.bias for layer in model.layers]
            + [model.norm_out.linear_1.bias],
            dim=0,
        )
        with torch.no_grad():
            step_prepare = _FixedStepPrepare(
                self._step_modulation_weight,
                self._step_modulation_bias,
            ).eval()
            self._step_prepare_jit = torch.jit.freeze(
                torch.jit.trace(
                    step_prepare,
                    (torch.zeros(10, HIDDEN), torch.zeros(1, HIDDEN)),
                    check_trace=False,
                ).eval()
            )
        self._fold_norm_weights = None
        if fold_norm_modulation:
            self._fold_norm_weights = torch.stack(
                [
                    torch.stack(
                        (
                            layer.norm1.norm.weight,
                            layer.norm2.weight,
                            layer.ffn_norm1.weight,
                            layer.ffn_norm2.weight,
                        )
                    )
                    for layer in model.layers
                ]
            ).detach().cpu().to(dtype=torch.float16)
        self._rpu_step_prepare = fold_norm_modulation and step_prepare_on_rpu
        self._fused_qkq = fused_qkq
        self._keep: dict[str, list[torch.Tensor]] = {}
        self._layer_outputs_rpu = torch.empty(
            ((NUM_LAYERS + 3) if self._collect_layer_outputs else 1, ACTION_SEQ, HIDDEN),
            dtype=torch.float16,
            device="rpu",
        )
        self._raw_actions_dummy_rpu = torch.zeros(
            (1, ACTION_SEQ, ACTION_PAD), dtype=torch.float16, device="rpu"
        )
        self._raw_actions_input_rpu = torch.empty(
            (4, ACTION_SEQ, ACTION_PAD), dtype=torch.float16, device="rpu"
        )
        self._raw_actions_input_cpu = torch.zeros(
            (4, ACTION_SEQ, ACTION_PAD), dtype=torch.float16, device="cpu"
        )
        self._hidden_input_rpu = torch.zeros(
            (1, ACTION_SEQ, HIDDEN), dtype=torch.float16, device="rpu"
        )
        self._packed_hidden_inputs_rpu = {
            batch: torch.zeros(
                (1, batch * ACTION_SEQ, HIDDEN),
                dtype=torch.float16,
                device="rpu",
            )
            for batch in range(2, 5)
        }
        self._packed_layer_outputs_rpu = {
            batch: torch.empty(
                (1, batch * ACTION_SEQ, HIDDEN),
                dtype=torch.float16,
                device="rpu",
            )
            for batch in range(2, 5)
        }
        self._hidden_input_cpu = torch.zeros(1, ACTION_SEQ, HIDDEN)
        self.last_layer_outputs: torch.Tensor | None = None
        self.last_debug_outputs: dict[str, torch.Tensor] = {}
        self._handle: int | None = _configure_nextdit_handle(
            self._install_weights
        )
        try:
            self._handle_finalizer = weakref.finalize(
                self, _destroy_nextdit_handle, self._handle
            )
        except BaseException:
            _destroy_nextdit_handle(self._handle)
            self._handle = None
            raise
        traj_dit._rpu_nextdit_runtime = self

    def _install_weights(self, handle: int) -> None:
        lists: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "self_q self_k self_v cross_q cross_k cross_v out ff1 ff2 ff3 norm1 norm2 "
                "ffn_norm1 ffn_norm2 self_qn_w self_qn_b self_kn_w self_kn_b "
                "cross_qn_w cross_qn_b cross_kn_w cross_kn_b context_norm_w cross_gate"
            ).split()
        }
        for layer in self._model.layers:
            # Q/K are head-partitioned, gathered to core 0 for exact H-wide LN,
            # then scattered back to the six attention cores.
            lists["self_q"].append(_col(layer.attn1.to_q.weight, ATTN_TP))
            lists["self_k"].append(_col(layer.attn1.to_k.weight, ATTN_TP))
            lists["self_v"].append(_col(layer.attn1.to_v.weight, ATTN_TP))
            lists["cross_q"].append(_col(layer.attn2.to_q.weight, ATTN_TP))
            lists["cross_k"].append(_col(layer.attn2.to_k.weight, ATTN_TP))
            lists["cross_v"].append(_col(layer.attn2.to_v.weight, ATTN_TP))
            lists["out"].append(_row(layer.attn2.to_out[0].weight, ATTN_TP))
            lists["ff1"].append(_col(layer.feed_forward.linear_1.weight, MLP_TP))
            lists["ff2"].append(_row(layer.feed_forward.linear_2.weight, MLP_TP))
            lists["ff3"].append(_col(layer.feed_forward.linear_3.weight, MLP_TP))
            lists["norm1"].append(_to_rpu_half(layer.norm1.norm.weight))
            lists["norm2"].append(_to_rpu_half(layer.norm2.weight))
            lists["ffn_norm1"].append(_to_rpu_half(layer.ffn_norm1.weight))
            lists["ffn_norm2"].append(_to_rpu_half(layer.ffn_norm2.weight))
            lists["self_qn_w"].append(_to_rpu_half(layer.attn1.norm_q.weight))
            lists["self_qn_b"].append(_to_rpu_half(layer.attn1.norm_q.bias))
            lists["self_kn_w"].append(_to_rpu_half(layer.attn1.norm_k.weight))
            lists["self_kn_b"].append(_to_rpu_half(layer.attn1.norm_k.bias))
            lists["cross_qn_w"].append(_to_rpu_half(layer.attn2.norm_q.weight))
            lists["cross_qn_b"].append(_to_rpu_half(layer.attn2.norm_q.bias))
            lists["cross_kn_w"].append(_to_rpu_half(layer.attn2.norm_k.weight))
            lists["cross_kn_b"].append(_to_rpu_half(layer.attn2.norm_k.bias))
            lists["context_norm_w"].append(_to_rpu_half(layer.norm1_context.weight))
            gate = layer.gate.detach().tanh().view(NUM_HEADS, 1).expand(NUM_HEADS, HEAD_DIM)
            lists["cross_gate"].append(_to_rpu_half(gate.contiguous()))

        if self._action_encoder is None:
            action_weight = torch.zeros(HIDDEN, ACTION_PAD)
            action_bias = torch.zeros(HIDDEN)
            action_decoder_weight = torch.zeros(ACTION_PAD, HIDDEN)
            action_decoder_bias = torch.zeros(ACTION_PAD)
        else:
            action_weight = torch.zeros(HIDDEN, ACTION_PAD)
            action_weight[:, :ACTION_DIM] = self._action_encoder.weight.detach().cpu()
            action_bias = self._action_encoder.bias.detach().cpu()
            action_decoder_weight = torch.zeros(ACTION_PAD, HIDDEN)
            action_decoder_weight[:ACTION_DIM] = self._action_decoder.weight.detach().cpu()
            action_decoder_bias = torch.zeros(ACTION_PAD)
            action_decoder_bias[:ACTION_DIM] = self._action_decoder.bias.detach().cpu()

        torch.ops.rpu.internvla_nextdit_set_weights(
            handle,
            lists["self_q"], lists["self_k"], lists["self_v"],
            lists["cross_q"], lists["cross_k"], lists["cross_v"], lists["out"],
            lists["ff1"], lists["ff2"], lists["ff3"],
            lists["norm1"], lists["norm2"], lists["ffn_norm1"], lists["ffn_norm2"],
            lists["self_qn_w"], lists["self_qn_b"],
            lists["self_kn_w"], lists["self_kn_b"],
            lists["cross_qn_w"], lists["cross_qn_b"],
            lists["cross_kn_w"], lists["cross_kn_b"], lists["context_norm_w"],
            lists["cross_gate"],
            _col(self._model.norm_out.linear_2.weight, 1),
            _to_rpu_half(self._model.norm_out.linear_2.bias),
            _to_rpu_half(torch.ones(HIDDEN)),
            _to_rpu_half(torch.zeros(HIDDEN)),
            _col(action_weight, 1),
            _to_rpu_half(action_bias),
            _to_rpu_half(self._action_position),
            _col(action_decoder_weight, 1),
            _to_rpu_half(action_decoder_bias),
            NUM_HEADS, HEAD_DIM, HIDDEN, self._ff_intermediate, EPS,
            float(self._model.norm_out.norm.eps),
        )
        self._keep = lists
        if self._fused_qkq:
            fused_qkq = [
                _col(
                    torch.cat(
                        (layer.attn1.to_q.weight,
                         layer.attn1.to_k.weight,
                         layer.attn2.to_q.weight),
                        dim=0,
                    ),
                    ATTN_TP,
                )
                for layer in self._model.layers
            ]
            torch.ops.rpu.internvla_nextdit_set_fused_qkq_weights(
                handle, fused_qkq)
            self._keep["fused_qkq"] = fused_qkq
        if self._rpu_step_prepare:
            modulation_rows = NUM_LAYERS * 4 * HIDDEN
            planar_weight = torch.cat(
                (
                    self._step_modulation_weight[:modulation_rows]
                    .view(NUM_LAYERS, 4, HIDDEN, HIDDEN)
                    .permute(1, 0, 2, 3)
                    .reshape(modulation_rows, HIDDEN),
                    self._step_modulation_weight[modulation_rows:],
                ),
                dim=0,
            ).contiguous()
            planar_bias = torch.cat(
                (
                    self._step_modulation_bias[:modulation_rows]
                    .view(NUM_LAYERS, 4, HIDDEN)
                    .permute(1, 0, 2)
                    .reshape(modulation_rows),
                    self._step_modulation_bias[modulation_rows:],
                ),
                dim=0,
            ).contiguous()
            assert self._fold_norm_weights is not None
            step_prepare_keep = [
                _col(planar_weight, MLP_TP),
                _to_rpu_half(planar_bias),
                _to_rpu_half(
                    self._fold_norm_weights.permute(1, 0, 2).contiguous()
                ),
            ]
            torch.ops.rpu.internvla_nextdit_set_step_prepare_weights(
                handle, *step_prepare_keep
            )
            self._keep["step_prepare"] = step_prepare_keep

    def _condition_state(
        self,
        condition: torch.Tensor,
        *,
        condition_is_caption: bool = False,
        reuse_condition: bool = True,
    ) -> tuple[int, _ConditionState]:
        condition = condition.detach().to(device="cpu")
        fingerprint: tuple[Any, ...] = ()
        if reuse_condition:
            flat = condition.reshape(-1)
            fingerprint = (
                tuple(condition.shape),
                condition.dtype,
                condition_is_caption,
                float(flat[0]),
                float(flat[flat.numel() // 3]),
                float(flat[(2 * flat.numel()) // 3]),
                float(flat[-1]),
            )
            for index, state in enumerate(self._conditions):
                if (
                    state.fingerprint == fingerprint
                    and torch.equal(state.source, condition)
                ):
                    return index, state
        if condition.shape[1] > self._max_cross_seq_len:
            raise ValueError(
                f"condition length {condition.shape[1]} exceeds RPU cache {self._max_cross_seq_len}"
            )
        if len(self._conditions) >= CONDITION_CACHE_SIZE:
            condition_slot = self._next_condition_slot
            self._next_condition_slot = (
                self._next_condition_slot + 1
            ) % CONDITION_CACHE_SIZE
            for key in tuple(self._steps):
                if key[0] == condition_slot:
                    del self._steps[key]
            for key in tuple(self._step_batches):
                if key[0] == condition_slot:
                    del self._step_batches[key]
            if self._active_cross_condition_index == condition_slot:
                self._active_cross_condition_index = None
            self._condition_batch_source = None
            self._condition_batch_version = -1
            self._condition_batch_bindings = []
        else:
            condition_slot = len(self._conditions)
        caption = condition if condition_is_caption else self._model.caption_projection(condition)
        caption_feats_pool = caption.mean(dim=1)
        caption_embedding = self._model.time_caption_embed.caption_embedder(
            caption_feats_pool
        )
        cross_context_rpu = self._cross_context_slots[condition_slot][
            : caption.numel()
        ].view(caption.shape)
        # PrivateUse1 copy performs the same FP32→FP16 conversion as the
        # explicit host cast, without materializing a full temporary caption.
        cross_context_rpu.copy_(caption)
        state = _ConditionState(
            source=condition.clone() if reuse_condition else condition,
            fingerprint=fingerprint,
            caption=caption,
            caption_embedding=caption_embedding,
            cross_context_rpu=cross_context_rpu,
        )
        if condition_slot == len(self._conditions):
            self._conditions.append(state)
        else:
            self._conditions[condition_slot] = state
        return condition_slot, state

    def _condition_bindings(
        self, condition: torch.Tensor
    ) -> list[tuple[int, _ConditionState]]:
        version = condition._version
        if (
            self._condition_batch_source is condition
            and self._condition_batch_version == version
        ):
            return self._condition_batch_bindings
        first = condition[:1]
        uniform = condition.shape[0] > 1 and torch.equal(
            condition[1:], first.expand(condition.shape[0] - 1, -1, -1)
        )
        if uniform:
            binding = self._condition_state(first)
            bindings = [binding] * condition.shape[0]
        else:
            if (
                condition.shape[0] <= CONDITION_CACHE_SIZE
                and len(self._conditions) + condition.shape[0] > CONDITION_CACHE_SIZE
            ):
                self._conditions.clear()
                self._next_condition_slot = 0
                self._steps.clear()
                self._step_batches.clear()
                self._active_cross_condition_index = None
            bindings = [
                self._condition_state(condition[index : index + 1])
                for index in range(condition.shape[0])
            ]
        self._condition_batch_source = condition
        self._condition_batch_version = version
        self._condition_batch_bindings = bindings
        return bindings

    def _x_batch_to_rpu(self, x: torch.Tensor) -> torch.Tensor:
        return _to_rpu_half(x)

    def _actions_batch_to_rpu(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[0] <= self._raw_actions_input_rpu.shape[0]:
            padded = self._raw_actions_input_cpu[: actions.shape[0]]
            padded[:, :, :ACTION_DIM].copy_(actions)
            target = self._raw_actions_input_rpu[: actions.shape[0]]
            target.copy_(padded)
            return target
        padded = torch.nn.functional.pad(
            actions, (0, ACTION_PAD - ACTION_DIM)
        ).to(dtype=torch.float16)
        return padded.to("rpu").contiguous()

    def _step_state(
        self,
        condition_index: int,
        condition: _ConditionState,
        timestep: torch.Tensor,
    ) -> _StepState:
        timestep_value = int(timestep.reshape(-1)[0].item())
        key = (condition_index, timestep_value)
        state = self._steps.get(key)
        if state is None:
            _, modulation, final_scale = prepare_nextdit_step_cpu(
                self._model,
                timestep,
                condition.caption,
                condition.caption_embedding,
            )
            state = _StepState(
                modulation_rpu=_to_rpu_half(modulation),
                final_scale_rpu=_to_rpu_half(final_scale),
            )
            self._steps[key] = state
        return state

    @torch.no_grad()
    def _prepare_step_batch_cpu(
        self,
        timesteps: torch.Tensor,
        condition_index: int,
        condition_state: _ConditionState,
        *,
        fold_for_unroll: bool = False,
    ) -> tuple[list[int], tuple[int, ...], torch.Tensor | None, torch.Tensor | None]:
        timestep_values = [int(value.item()) for value in timesteps.reshape(-1)]
        timestep_key = tuple(timestep_values)
        batch_key = (condition_index, timestep_key, fold_for_unroll)
        if batch_key in self._step_batches:
            return timestep_values, timestep_key, None, None
        selected = timesteps.reshape(-1)
        time_embedding = self._step_time_embeddings.get(timestep_key)
        if time_embedding is None:
            time_embedder = self._model.time_caption_embed
            time_freq = time_embedder.time_proj(selected)
            time_embedding = time_embedder.timestep_embedder(
                time_freq.to(dtype=condition_state.caption.dtype)
            )
            self._step_time_embeddings[timestep_key] = time_embedding
        if self._rpu_step_prepare and fold_for_unroll:
            modulation = torch.nn.functional.silu(
                time_embedding + condition_state.caption_embedding
            ).to(dtype=torch.float16)
            final_scale = None
        else:
            modulation, final_scale = self._step_prepare_jit(
                time_embedding,
                condition_state.caption_embedding,
            )
        if fold_for_unroll and not self._rpu_step_prepare:
            assert self._fold_norm_weights is not None
            modulation = modulation.to(dtype=torch.float16).mul_(
                self._fold_norm_weights.unsqueeze(0)
            )
        return timestep_values, timestep_key, modulation, final_scale

    @torch.no_grad()
    def _publish_step_batch(
        self,
        condition_index: int,
        timestep_values: list[int],
        timestep_key: tuple[int, ...],
        modulation: torch.Tensor | None,
        final_scale: torch.Tensor | None,
        *,
        fold_for_unroll: bool = False,
    ) -> None:
        rpu_step_prepare = self._rpu_step_prepare and fold_for_unroll
        if modulation is None or (final_scale is None and not rpu_step_prepare):
            return
        batch_key = (condition_index, timestep_key, fold_for_unroll)
        condition_has_step_batch = any(
            key[0] == condition_index for key in self._step_batches
        )
        if rpu_step_prepare:
            modulation_rpu = self._step_activated_slots[condition_index][
                : len(timestep_values)
            ]
            modulation_rpu.copy_(modulation)
            final_scale_rpu = self._step_final_scale_slots[condition_index][
                : len(timestep_values)
            ]
        elif not condition_has_step_batch and len(timestep_values) <= 10:
            modulation_rpu = self._step_modulation_slots[condition_index][
                : len(timestep_values)
            ]
            final_scale_rpu = self._step_final_scale_slots[condition_index][
                : len(timestep_values)
            ]
            modulation_rpu.copy_(modulation.to(dtype=torch.float16))
            final_scale_rpu.copy_(final_scale.to(dtype=torch.float16))
        else:
            modulation_rpu = _to_rpu_half(modulation)
            final_scale_rpu = _to_rpu_half(final_scale)
        self._step_batches[batch_key] = _StepBatchState(
            modulation_rpu=modulation_rpu,
            final_scale_rpu=final_scale_rpu,
        )
        if not fold_for_unroll:
            for batch_index, timestep_value in enumerate(timestep_values):
                self._steps[(condition_index, timestep_value)] = _StepState(
                    modulation_rpu=modulation_rpu[batch_index],
                    final_scale_rpu=final_scale_rpu[batch_index],
                )

    @torch.no_grad()
    def prepare_steps(
        self,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        *,
        condition_is_caption: bool = False,
        _fold_for_unroll: bool = False,
        _reuse_condition: bool = True,
    ) -> None:
        """Batch and cache fixed scheduler-step tensors for one condition."""

        condition_index, condition_state = self._condition_state(
            condition,
            condition_is_caption=condition_is_caption,
            reuse_condition=_reuse_condition,
        )
        self._condition_batch_source = condition
        self._condition_batch_version = condition._version
        self._condition_batch_bindings = [(condition_index, condition_state)]
        prepared = self._prepare_step_batch_cpu(
            timesteps,
            condition_index,
            condition_state,
            fold_for_unroll=_fold_for_unroll,
        )
        self._publish_step_batch(
            condition_index,
            *prepared,
            fold_for_unroll=_fold_for_unroll,
        )

    @torch.no_grad()
    def _forward_one(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        *,
        condition_binding: tuple[int, _ConditionState] | None = None,
        x_rpu: torch.Tensor | None = None,
        raw_actions_rpu: torch.Tensor | None = None,
        use_action_encoder: bool = False,
    ) -> torch.Tensor:
        if self._handle is None:
            raise RuntimeError("NextDiTRPURuntime has been destroyed")
        if x.device.type != "cpu" or x.shape != (1, ACTION_SEQ, HIDDEN):
            raise ValueError(f"x must be CPU [1,{ACTION_SEQ},{HIDDEN}], got {tuple(x.shape)}")
        if condition_binding is None:
            condition_index, condition_state = self._condition_state(condition)
        else:
            condition_index, condition_state = condition_binding
        step_state = self._step_state(condition_index, condition_state, timestep)
        if x_rpu is None:
            x_rpu = _to_rpu_half(x)
        if raw_actions_rpu is None:
            raw_actions_rpu = self._raw_actions_dummy_rpu
        populate_cross_cache = condition_index != self._active_cross_condition_index
        sig = self._graph_signature(
            condition.shape[1], use_action_encoder, populate_cross_cache
        )
        with self._graph_cache.capture(sig):
            hidden_rpu = torch.ops.rpu.internvla_nextdit_forward(
                self._handle,
                x_rpu,
                step_state.modulation_rpu,
                step_state.final_scale_rpu,
                raw_actions_rpu,
                use_action_encoder,
                populate_cross_cache,
                False,
                False,
                [],
                condition_state.cross_context_rpu,
                self._self_cache.k_caches,
                self._self_cache.v_caches,
                self._cross_cache.k_caches,
                self._cross_cache.v_caches,
                self._layer_outputs_rpu,
                self._collect_layer_outputs,
            )
        self._active_cross_condition_index = condition_index
        # The fused-model output is registry-stable.  Copy immediately before
        # another trajectory or step can reuse its stable DDR storage.
        hidden_cpu = hidden_rpu.cpu().float()
        self.last_debug_outputs = {}
        self.last_layer_outputs = None
        if self._collect_layer_outputs:
            probes = self._layer_outputs_rpu.cpu().float()
            self.last_layer_outputs = probes[:NUM_LAYERS]
            if torch.rpu.get_debug_export():
                self.last_debug_outputs = {
                    "internvla_nextdit_L0_q_projected": probes[NUM_LAYERS],
                    "internvla_nextdit_L0_q_normalized": probes[NUM_LAYERS + 1],
                    "internvla_nextdit_L0_q_headmajor": probes[NUM_LAYERS + 2]
                    .view(NUM_HEADS, ACTION_SEQ, HEAD_DIM),
                }
        return hidden_cpu[..., :ACTION_DIM] if use_action_encoder else hidden_cpu

    def _graph_signature(
        self,
        cross_seq: int,
        use_action_encoder: bool = False,
        populate_cross_cache: bool = True,
        action_batch: int = 1,
        packed_action_batch: bool = False,
        denoise_steps: int = 0,
        prepare_modulation_on_rpu: bool = False,
        euler_step_bits: tuple[int, ...] = (),
    ) -> Any:
        key = (
            cross_seq,
            use_action_encoder,
            populate_cross_cache,
            action_batch,
            packed_action_batch,
            denoise_steps,
            prepare_modulation_on_rpu,
            euler_step_bits,
        )
        cached = self._graph_signatures.get(key)
        if cached is not None:
            return cached
        import rpu_backend

        signature = rpu_backend.graph.GraphSignature(
            op_id="internvla_nextdit_blocks",
            shapes=[
                ACTION_SEQ * action_batch if packed_action_batch else ACTION_SEQ,
                HIDDEN,
                cross_seq,
                action_batch,
            ],
            dyn_dims=[
                NUM_LAYERS,
                NUM_HEADS,
                HEAD_DIM,
                self._ff_intermediate,
                int(self._collect_layer_outputs),
                int(use_action_encoder),
                int(populate_cross_cache),
                action_batch,
                int(packed_action_batch),
                denoise_steps,
                int(prepare_modulation_on_rpu),
                *euler_step_bits,
            ],
            dtypes=[torch.float16],
        )
        self._graph_signatures[key] = signature
        return signature

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Run one or more hidden-state trajectories over the RPU runtime."""

        if x.ndim != 3 or condition.ndim != 3 or x.shape[0] != condition.shape[0]:
            raise ValueError("x and condition must be [B,T,C] with the same batch")
        if timestep.numel() == 1:
            timestep = timestep.reshape(1).expand(x.shape[0])
        if timestep.numel() != x.shape[0]:
            raise ValueError("timestep must be scalar or have one value per trajectory")
        if x.shape[0] > CONDITION_CACHE_SIZE:
            return torch.cat(
                [
                    self._forward_one(
                        x[index : index + 1],
                        timestep[index : index + 1],
                        condition[index : index + 1],
                    )
                    for index in range(x.shape[0])
                ],
                dim=0,
            )
        bindings = self._condition_bindings(condition)
        x_rpu = self._x_batch_to_rpu(x)
        outputs = [
            self._forward_one(
                x[i : i + 1],
                timestep[i : i + 1],
                condition[i : i + 1],
                condition_binding=bindings[i],
                x_rpu=x_rpu[i : i + 1],
            )
            for i in range(x.shape[0])
        ]
        return torch.cat(outputs, dim=0)

    def _forward_actions_uniform_batch(
        self,
        actions_rpu: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        condition_binding: tuple[int, _ConditionState],
    ) -> torch.Tensor:
        condition_index, condition_state = condition_binding
        step_state = self._step_state(
            condition_index, condition_state, timestep[:1]
        )
        populate_cross_cache = condition_index != self._active_cross_condition_index
        action_batch = actions_rpu.shape[0]
        sig = self._graph_signature(
            condition.shape[1], True, populate_cross_cache, action_batch, True
        )
        with self._graph_cache.capture(sig):
            velocity_rpu = torch.ops.rpu.internvla_nextdit_forward(
                self._handle,
                self._packed_hidden_inputs_rpu[action_batch],
                step_state.modulation_rpu,
                step_state.final_scale_rpu,
                actions_rpu,
                True,
                populate_cross_cache,
                False,
                False,
                [],
                condition_state.cross_context_rpu,
                self._packed_self_cache.k_caches,
                self._packed_self_cache.v_caches,
                self._cross_cache.k_caches,
                self._cross_cache.v_caches,
                self._packed_layer_outputs_rpu[action_batch],
                False,
            )
        self._active_cross_condition_index = condition_index
        self.last_layer_outputs = None
        self.last_debug_outputs = {}
        return velocity_rpu.cpu().float()[..., :ACTION_DIM]

    @torch.no_grad()
    def denoise_actions(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        euler_steps: torch.Tensor,
        condition: torch.Tensor,
        *,
        condition_is_caption: bool = False,
        reuse_condition: bool = True,
    ) -> torch.Tensor:
        """Run a fixed flow-matching Euler schedule as one RPU graph."""

        if not self._action_encoder_enabled:
            raise RuntimeError("this runtime was built without an RPU action encoder")
        if self._handle is None:
            raise RuntimeError("NextDiTRPURuntime has been destroyed")
        if (
            actions.device.type != "cpu"
            or actions.ndim != 3
            or actions.shape[1:] != (ACTION_SEQ, ACTION_DIM)
            or not 1 <= actions.shape[0] <= 4
            or condition.device.type != "cpu"
            or condition.ndim != 3
            or condition.shape[0] != 1
        ):
            raise ValueError(
                "actions must be CPU [B,32,3] with B in [1,4], and condition "
                "must be CPU [1,S,C]"
            )
        step_count = timesteps.numel()
        if step_count < 1 or step_count > 10 or euler_steps.numel() != step_count:
            raise ValueError("timesteps and euler_steps must have the same length in [1,10]")

        timesteps = timesteps.reshape(-1)
        euler_steps = euler_steps.reshape(-1).to(device="cpu", dtype=torch.float32)
        fold_for_unroll = self._fold_norm_weights is not None
        self.prepare_steps(
            timesteps,
            condition,
            condition_is_caption=condition_is_caption,
            _fold_for_unroll=fold_for_unroll,
            _reuse_condition=reuse_condition,
        )
        condition_index, condition_state = self._condition_batch_bindings[0]
        timestep_key = tuple(int(value.item()) for value in timesteps)
        step_batch = self._step_batches[
            (condition_index, timestep_key, fold_for_unroll)
        ]
        actions_rpu = self._actions_batch_to_rpu(actions)
        action_batch = actions.shape[0]
        packed_action_batch = action_batch > 1
        self_cache = self._packed_self_cache if packed_action_batch else self._self_cache
        hidden_input = (
            self._packed_hidden_inputs_rpu[action_batch]
            if packed_action_batch
            else self._hidden_input_rpu
        )
        layer_outputs = (
            self._packed_layer_outputs_rpu[action_batch]
            if packed_action_batch
            else self._layer_outputs_rpu
        )
        populate_cross_cache = condition_index != self._active_cross_condition_index
        euler_values = [float(value.item()) for value in euler_steps]
        euler_step_bits = tuple(
            struct.unpack("<i", struct.pack("<f", value))[0]
            for value in euler_values
        )

        sig = self._graph_signature(
            condition.shape[1],
            True,
            populate_cross_cache,
            action_batch,
            packed_action_batch,
            step_count,
            self._rpu_step_prepare and fold_for_unroll,
            euler_step_bits,
        )
        with self._graph_cache.capture(sig):
            actions_out_rpu = torch.ops.rpu.internvla_nextdit_forward(
                self._handle,
                hidden_input,
                step_batch.modulation_rpu,
                step_batch.final_scale_rpu,
                actions_rpu,
                True,
                populate_cross_cache,
                True,
                self._rpu_step_prepare and fold_for_unroll,
                euler_values,
                condition_state.cross_context_rpu,
                self_cache.k_caches,
                self_cache.v_caches,
                self._cross_cache.k_caches,
                self._cross_cache.v_caches,
                layer_outputs,
                False,
            )
        self._active_cross_condition_index = condition_index
        self.last_layer_outputs = None
        self.last_debug_outputs = {}
        return actions_out_rpu.cpu().float()[..., :ACTION_DIM]

    @torch.no_grad()
    def forward_actions(
        self,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Encode raw 3D actions on RPU, then run the fused NextDiT core."""

        if not self._action_encoder_enabled:
            raise RuntimeError("this runtime was built without an RPU action encoder")
        if (
            actions.device.type != "cpu"
            or actions.ndim != 3
            or actions.shape[1:] != (ACTION_SEQ, ACTION_DIM)
            or condition.ndim != 3
            or condition.shape[0] not in (1, actions.shape[0])
        ):
            raise ValueError("actions must be CPU [B,32,3] with condition batch 1 or B")
        scalar_timestep = timestep.numel() == 1
        if not scalar_timestep and timestep.numel() != actions.shape[0]:
            raise ValueError("timestep must be scalar or have one value per trajectory")
        if actions.shape[0] > CONDITION_CACHE_SIZE:
            return torch.cat(
                [
                    self.forward_actions(
                        actions[index : index + 1],
                        timestep[:1] if scalar_timestep else timestep[index : index + 1],
                        condition if condition.shape[0] == 1 else condition[index : index + 1],
                    )
                    for index in range(actions.shape[0])
                ],
                dim=0,
            )
        bindings = self._condition_bindings(condition)
        if condition.shape[0] == 1:
            bindings = bindings * actions.shape[0]
        actions_rpu = self._actions_batch_to_rpu(actions)
        uniform_batch = (
            1 < actions.shape[0] <= 4
            and not self._collect_layer_outputs
            and all(index == bindings[0][0] for index, _ in bindings[1:])
            and (
                scalar_timestep
                or torch.equal(timestep, timestep[:1].expand_as(timestep))
            )
        )
        if uniform_batch:
            return self._forward_actions_uniform_batch(
                actions_rpu, timestep.reshape(-1), condition[:1], bindings[0]
            )
        outputs = [
            self._forward_one(
                self._hidden_input_cpu,
                timestep[:1] if scalar_timestep else timestep[index : index + 1],
                condition if condition.shape[0] == 1 else condition[index : index + 1],
                condition_binding=bindings[index],
                x_rpu=self._hidden_input_rpu,
                raw_actions_rpu=actions_rpu[index : index + 1],
                use_action_encoder=True,
            )
            for index in range(actions.shape[0])
        ]
        return torch.cat(outputs, dim=0)

    def destroy(self) -> None:
        if getattr(self, "_handle", None) is not None:
            graph_cache = getattr(self, "_graph_cache", None)
            if graph_cache is not None:
                graph_cache.clear()
            finalizer = getattr(self, "_handle_finalizer", None)
            if finalizer is not None and getattr(finalizer, "alive", False):
                torch.ops.rpu.internvla_nextdit_destroy(self._handle)
                finalizer.detach()
            elif finalizer is None:
                torch.ops.rpu.internvla_nextdit_destroy(self._handle)
            self._handle = None
        owner = getattr(self, "_owner", None)
        if getattr(owner, "_rpu_nextdit_runtime", None) is self:
            owner._rpu_nextdit_runtime = None

    def graph_stats(self) -> dict[str, Any]:
        """Return a serializable retained-Graph lifecycle receipt."""

        entries = list(self._graph_cache.snapshot())
        return {
            "size": int(self._graph_cache.size()),
            "invariant": bool(self._graph_cache.cache_invariant_ok()),
            "entries": [
                {
                    "signature": repr(entry.signature),
                    "kernel_count": int(entry.kernel_count),
                    "data_node_count": int(entry.data_node_count),
                    "replay_count": int(entry.replay_count),
                    "recapture_count": int(entry.recapture_count),
                }
                for entry in entries
            ],
        }


def build_nextdit_runtime(
    traj_dit: torch.nn.Module,
    *,
    max_cross_seq_len: int = 320,
    collect_layer_outputs: bool = False,
    action_encoder: torch.nn.Linear | None = None,
    action_decoder: torch.nn.Linear | None = None,
    action_position: torch.Tensor | None = None,
    asset_manifest: Mapping[str | Path, str] | None = None,
    asset_paths: Sequence[str | Path] | None = None,
) -> NextDiTRPURuntime:
    """Build the first-slice RPU runtime from a loaded CPU NextDiT instance."""

    return NextDiTRPURuntime(
        traj_dit,
        max_cross_seq_len=max_cross_seq_len,
        collect_layer_outputs=collect_layer_outputs,
        action_encoder=action_encoder,
        action_decoder=action_decoder,
        action_position=action_position,
        asset_manifest=asset_manifest,
        asset_paths=asset_paths,
    )


__all__ = [
    "NextDiTRPURuntime",
    "build_nextdit_runtime",
    "prepare_nextdit_condition_cpu",
    "prepare_nextdit_step_cpu",
]
