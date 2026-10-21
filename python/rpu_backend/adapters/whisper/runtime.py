"""Resident RPU runtime for the frozen Whisper base encoder and decoder."""

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

from .weights import load_decoder_weights, load_encoder_weights


_LAYERS = 6
_HIDDEN = 512
_INTERMEDIATE = 2048
_HEADS = 8
_HEAD_DIM = 64
_ENCODER_TOKENS = 1500
_TEXT_TOKENS = 448
_VOCAB = 51865
_PADDED_VOCAB = 51968


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


@dataclass(frozen=True)
class _PreparedDecoder:
    weight_args: tuple
    cross_args: tuple
    io_args: tuple
    token_embedding: torch.Tensor
    keepalive: tuple


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


def _prepare_decoder_tensors(
    encoder_weights: dict[str, object], decoder_weights: dict[str, object]
) -> _PreparedDecoder:
    layers = decoder_weights["layers"]
    weight_args = tuple(
        [[_swizzle(layer[name], partition) for layer in layers]
         for name, partition in (
             ("sq_w", 1), ("sk_w", 1), ("sv_w", 1), ("so_w", 0),
             ("cq_w", 1), ("co_w", 0), ("ff1_w", 1), ("ff2_w", 0),
         )]
        + [[_device(layer[name]) for layer in layers]
           for name in (
               "n1_w", "n1_b", "n2_w", "n2_b", "n3_w", "n3_b",
               "sq_b", "sv_b", "so_b", "cq_b", "co_b", "ff1_b", "ff2_b",
           )]
    )
    cross_layers = encoder_weights["cross_layers"]
    cross_args = (
        [_swizzle(layer["k_w"], 1) for layer in cross_layers],
        [_swizzle(layer["v_w"], 1) for layer in cross_layers],
        [_device(layer["v_b"]) for layer in cross_layers],
    )
    token_embedding = _device(decoder_weights["token_embedding"])
    padded_head = np.pad(
        decoder_weights["token_embedding"], ((0, _PADDED_VOCAB - _VOCAB), (0, 0))
    )
    io_args = (
        _device(decoder_weights["position"]),
        _device(decoder_weights["final_w"]),
        _device(decoder_weights["final_b"]),
        _swizzle(padded_head, 1),
    )
    return _PreparedDecoder(
        weight_args, cross_args, io_args, token_embedding,
        (weight_args, cross_args, io_args, token_embedding),
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


def _make_decoder_cache(max_seq_len: int) -> RPUCache:
    return RPUCache(
        _LAYERS, 1, max_seq_len, _HEADS, _HEAD_DIM,
        dtype=torch.float16, attn_tp=8,
    )


def _make_decoder_staging():
    return (
        torch.empty((1, _ENCODER_TOKENS, _HIDDEN),
                    dtype=torch.float16, device="rpu"),
        [torch.empty(1, dtype=torch.int32, device="rpu") for _ in range(2)],
        [torch.empty((1, 1, _HIDDEN), dtype=torch.float16, device="rpu")
         for _ in range(2)],
        [torch.empty((1, _PADDED_VOCAB), dtype=torch.float16, device="rpu")
         for _ in range(2)],
    )


class WhisperDecoderRuntime:
    """Own the six-layer decoder, both KV banks and fixed Graph addresses."""

    def __init__(self, encoder_path: str | Path, decoder_path: str | Path):
        self._handle = None
        self._resource = None
        self._closed = True
        prepared = _prepare_decoder_tensors(
            load_encoder_weights(encoder_path), load_decoder_weights(decoder_path)
        )
        self_cache = _make_decoder_cache(_TEXT_TOKENS)
        cross_cache = _make_decoder_cache(1536)
        cross_staging, ids, hidden, outputs = _make_decoder_staging()
        graphs = tuple(rpu_backend.graph.GraphCache(max_entries=1) for _ in range(3))
        try:
            handle = torch.ops.rpu.whisper_decoder_create()
        except BaseException:
            for graph in graphs:
                graph.clear()
            raise
        self._handle = handle
        resource = _InstalledNativeResource(
            self, handle, torch.ops.rpu.whisper_decoder_destroy,
            graphs=graphs,
            keepalive=(prepared.keepalive, self_cache, cross_cache, cross_staging,
                       ids, hidden, outputs),
            label="Whisper decoder", handle_name="_handle",
        )
        self._resource = resource
        try:
            torch.ops.rpu.whisper_decoder_set_weights(handle, *prepared.weight_args)
            torch.ops.rpu.whisper_decoder_set_cross_weights(handle, *prepared.cross_args)
            torch.ops.rpu.whisper_decoder_set_io(handle, *prepared.io_args)
        except BaseException as error:
            resource.cleanup_failure(error, prepared, self_cache, cross_cache)
            raise
        self._token_embedding = prepared.token_embedding
        self._self_cache = self_cache
        self._cross_cache = cross_cache
        self._cross_staging = cross_staging
        self._ids = ids
        self._hidden = hidden
        self._outputs = outputs
        self._cross_graph, self._prefill_graph, self._decode_graph = graphs
        self._position = 0
        self._prepared = False
        self._closed = False

    def prepare_audio(self, encoder_hidden: torch.Tensor) -> None:
        if self._closed:
            raise RuntimeError("Whisper decoder is closed")
        if (not isinstance(encoder_hidden, torch.Tensor)
                or encoder_hidden.device.type != "rpu"
                or encoder_hidden.dtype is not torch.float16
                or tuple(encoder_hidden.shape) != (1, _ENCODER_TOKENS, _HIDDEN)):
            raise ValueError(
                "Whisper decoder requires one RPU FP16 encoder tensor [1, 1500, 512]"
            )
        self._self_cache.reset()
        self._cross_cache.reset()
        self._cross_staging.copy_(encoder_hidden)
        torch.ops.rpu.whisper_decoder_reset(self._handle, 0)
        signature = rpu_backend.graph.GraphSignature(
            op_id="whisper_decoder_cross", shapes=[1, _ENCODER_TOKENS, _HIDDEN],
            dyn_dims=[_LAYERS], dtypes=[torch.float16],
        )
        try:
            with self._cross_graph.capture(signature):
                torch.ops.rpu.whisper_decoder_prepare_cross(
                    self._handle, self._cross_staging,
                    list(self._cross_cache.k_caches),
                    list(self._cross_cache.v_caches),
                )
        finally:
            torch.ops.rpu.spm_alloc_reset_temporary()
        if not self._cross_graph.is_frozen():
            self._cross_graph.freeze()
        self._position = 0
        self._prepared = True

    def _embed(self, token_id: int, slot: int) -> torch.Tensor:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("Whisper token_id must be an integer")
        if not 0 <= token_id < _VOCAB:
            raise ValueError(f"Whisper token_id must be in [0, {_VOCAB})")
        self._ids[slot].copy_(torch.tensor([token_id], dtype=torch.int32))
        gathered = torch.ops.rpu.gather_embedding(
            self._token_embedding, self._ids[slot]
        )
        self._hidden[slot].copy_(gathered.reshape(1, 1, _HIDDEN))
        return self._hidden[slot]

    @torch.no_grad()
    def prefill(self, token_id: int) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("Whisper decoder is closed")
        if not self._prepared or self._position != 0:
            raise RuntimeError("Whisper decoder requires freshly prepared audio")
        return self._forward(token_id, prefill=True)

    @torch.no_grad()
    def decode(self, token_id: int) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("Whisper decoder is closed")
        if not self._prepared or self._position == 0:
            raise RuntimeError("Whisper decoder requires prefill before decode")
        if self._position >= _TEXT_TOKENS:
            raise RuntimeError("Whisper decoder reached the 448-token limit")
        return self._forward(token_id, prefill=False)

    def _forward(self, token_id: int, *, prefill: bool) -> torch.Tensor:
        slot = 0 if prefill else 1
        hidden = self._embed(token_id, slot)
        graph = self._prefill_graph if prefill else self._decode_graph
        output = self._outputs[slot]
        signature = rpu_backend.graph.GraphSignature(
            op_id="whisper_decoder_prefill" if prefill else "whisper_decoder_decode",
            shapes=[1, 1, _HIDDEN], dyn_dims=[_LAYERS], dtypes=[torch.float16],
        )
        try:
            with graph.capture(signature):
                if prefill:
                    torch.ops.rpu.whisper_decoder_prefill_logits(
                        self._handle, hidden, output,
                        list(self._self_cache.k_caches),
                        list(self._self_cache.v_caches),
                        list(self._cross_cache.k_caches),
                        list(self._cross_cache.v_caches),
                    )
                else:
                    torch.ops.rpu.whisper_decoder_decode_logits(
                        self._handle, hidden, output,
                        list(self._self_cache.k_caches),
                        list(self._self_cache.v_caches),
                        list(self._cross_cache.k_caches),
                        list(self._cross_cache.v_caches), self._position,
                    )
        finally:
            torch.ops.rpu.spm_alloc_reset_temporary()
        if not graph.is_frozen():
            graph.freeze()
        self._position += 1
        self._self_cache.update_position(1)
        return output.cpu()[:, :_VOCAB].clone()

    def close(self) -> None:
        if self._closed:
            return
        self._resource.retire()
        self._closed = True
        self._resource = None
        self._token_embedding = self._self_cache = self._cross_cache = None
        self._cross_staging = self._ids = self._hidden = self._outputs = None
        self._cross_graph = self._prefill_graph = self._decode_graph = None
