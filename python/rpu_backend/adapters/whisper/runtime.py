"""Resident RPU runtime for the frozen Whisper base encoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import rpu_backend
from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime._native_retirement import _InstalledNativeResource
from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
from rpu_backend.runtime.execution_planner import GRAPH_RETAINED_CACHE
from rpu_backend.runtime.weights import transform_linear_weight

from .weights import load_encoder_weights


_LAYERS = 6
_HIDDEN = 512
_INTERMEDIATE = 2048
_HEADS = 8
_HEAD_DIM = 64
_ENCODER_TOKENS = 1500


@dataclass(frozen=True)
class _PreparedEncoder:
    weight_args: tuple
    final_args: tuple
    frontend_args: tuple
    keepalive: tuple


@dataclass(frozen=True)
class _EncoderPlan:
    descriptor: list[int]
    signature: object


def _device(array) -> torch.Tensor:
    return torch.from_numpy(array).half().contiguous().to("rpu")


def _swizzle(array, partition: int) -> torch.Tensor:
    tensor = torch.from_numpy(array).half().contiguous()
    return transform_linear_weight(tensor, partition, 8).to("rpu")


def _prepare_encoder_tensors(weights: dict[str, object]) -> _PreparedEncoder:
    layers = weights["layers"]
    q_weights = [_swizzle(layer["q_w"], 1) for layer in layers]
    k_weights = [_swizzle(layer["k_w"], 1) for layer in layers]
    v_weights = [_swizzle(layer["v_w"], 1) for layer in layers]
    o_weights = [_swizzle(layer["o_w"], 0) for layer in layers]
    up_weights = [_swizzle(layer["up_w"], 1) for layer in layers]
    down_weights = [_swizzle(layer["down_w"], 0) for layer in layers]
    columns = tuple(
        [_device(layer[name]) for layer in layers]
        for name in (
            "ln1_w", "ln1_b", "ln2_w", "ln2_b", "q_b", "v_b",
            "o_b", "up_b", "down_b",
        )
    )
    gammas = [_device(np.ones(_HIDDEN, dtype=np.float32)) for _ in layers]
    weight_args = (
        q_weights, k_weights, v_weights, o_weights, up_weights, down_weights,
        *columns, gammas, gammas, _HEADS, _HEAD_DIM, _HIDDEN,
        _INTERMEDIATE, 1e-5,
    )

    def conv(name: str) -> torch.Tensor:
        tensor = torch.from_numpy(weights[name]).half().permute(0, 2, 1)
        return F.pad(tensor.unsqueeze(1).contiguous(), (0, 16)).to("rpu")

    frontend_args = (
        conv("conv1_w"), F.pad(_device(weights["conv1_b"]), (0, 16)),
        conv("conv2_w"), F.pad(_device(weights["conv2_b"]), (0, 16)),
        _device(weights["position"]),
    )
    final_args = (_device(weights["final_w"]), _device(weights["final_b"]))
    return _PreparedEncoder(
        weight_args=weight_args,
        final_args=final_args,
        frontend_args=frontend_args,
        keepalive=(weight_args, final_args, frontend_args),
    )


def _make_encoder_cache() -> RPUCache:
    return RPUCache(
        _LAYERS, 1, 1536, _HEADS, _HEAD_DIM,
        dtype=torch.float16, attn_tp=8,
    )


def _make_mel_staging() -> torch.Tensor:
    return torch.empty((1, 1, 3000, 96), dtype=torch.float16, device="rpu")


def _copy_mel_to_staging(staging: torch.Tensor, mel: torch.Tensor) -> None:
    if (not isinstance(mel, torch.Tensor) or mel.device.type != "cpu"
            or mel.dtype is not torch.float16 or tuple(mel.shape) != (1, 80, 3000)):
        raise ValueError("Whisper encoder requires one CPU FP16 mel tensor [1, 80, 3000]")
    formatted = F.pad(
        mel.unsqueeze(2).permute(0, 2, 3, 1).contiguous(), (0, 16)
    )
    staging.copy_(formatted)


def _plan_whisper_encoder(handle: int, graph) -> _EncoderPlan:
    box = {}
    _, _ = plan_bounded_prefill_execution(
        _ENCODER_TOKENS,
        _ENCODER_TOKENS,
        0,
        resolve_stage_domain=lambda length: (
            torch.ops.rpu.whisper_encoder_resolve_stage_domain(handle, int(length))
        ),
        position=0,
        alignment=1,
        padding_rows=0,
        exact_chunk_size=None,
        request_id="whisper:encoder",
        execution_owner=None,
        execution_stage="encoder",
        execution_native=None,
        queue_owner_id=int(handle),
        plan_result_sink=lambda plan: box.__setitem__("plan", plan),
        graph_mode=GRAPH_RETAINED_CACHE,
        plan_signature=(),
        graph_cache=graph,
    )
    plan = box["plan"]
    descriptor = list(plan.selected.stage_tuple.physical_descriptor)
    if not descriptor:
        raise RuntimeError("Whisper encoder planner returned no native descriptor")
    signature = rpu_backend.graph.GraphSignature(
        op_id="whisper_encoder",
        shapes=[1, 80, 3000],
        dyn_dims=[_LAYERS, *plan.graph_key_words()],
        dtypes=[torch.float16],
    )
    return _EncoderPlan(descriptor, signature)


class WhisperEncoderRuntime:
    """Own one installed encoder, its fixed staging address and retained graph."""

    def __init__(self, encoder_path: str | Path):
        self._handle = None
        self._resource = None
        self._closed = True
        prepared = _prepare_encoder_tensors(load_encoder_weights(encoder_path))
        cache = _make_encoder_cache()
        staging = _make_mel_staging()
        graph = rpu_backend.graph.GraphCache(max_entries=1)
        try:
            handle = torch.ops.rpu.whisper_encoder_create()
        except BaseException:
            graph.clear()
            raise
        self._handle = handle
        resource = _InstalledNativeResource(
            self,
            handle,
            torch.ops.rpu.whisper_encoder_destroy,
            graphs=(graph,),
            keepalive=(prepared.keepalive, cache, staging),
            label="Whisper encoder",
            handle_name="_handle",
        )
        self._resource = resource
        try:
            torch.ops.rpu.whisper_encoder_set_weights(
                handle, *prepared.weight_args
            )
            torch.ops.rpu.whisper_encoder_set_final_norm(
                handle, *prepared.final_args
            )
            torch.ops.rpu.whisper_encoder_set_frontend(
                handle, *prepared.frontend_args
            )
            torch.ops.rpu.whisper_encoder_set_chunk_envelope(
                handle, cache.max_seq_len, 0
            )
            plan = _plan_whisper_encoder(handle, graph)
        except BaseException as error:
            resource.cleanup_failure(error, prepared, cache, staging)
            raise
        self._cache = cache
        self._staging = staging
        self._graph = graph
        self._plan = plan
        self._closed = False

    @torch.no_grad()
    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("Whisper encoder is closed")
        _copy_mel_to_staging(self._staging, mel)
        self._cache.reset_to_position(0)
        try:
            with self._graph.capture(self._plan.signature):
                output = torch.ops.rpu.whisper_encoder_forward(
                    self._handle,
                    self._staging,
                    list(self._cache.k_caches),
                    list(self._cache.v_caches),
                    _ENCODER_TOKENS,
                    self._plan.descriptor,
                )
        finally:
            torch.ops.rpu.spm_alloc_reset_temporary()
        if not self._graph.is_frozen():
            self._graph.freeze()
        return output.clone()

    def close(self) -> None:
        if self._closed:
            return
        self._resource.retire()
        self._closed = True
        self._resource = None
        self._cache = self._staging = self._graph = self._plan = None
