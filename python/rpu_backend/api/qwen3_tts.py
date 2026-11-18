"""Bounded Qwen3-TTS Base speech inference with device autoregressive components."""
from __future__ import annotations

import hashlib
import copy
from contextlib import nullcontext
from functools import partial
import operator
import re
from pathlib import Path
import threading
import time
import types

import numpy as np
import torch

from ._qwen3_audio_stages import StageTimer
from .errors import RPUUnsupportedDtypeError


def _rescale_mlp_payload(weight, scale, tensor_scale, multiplier):
    """Balance Up/Down after quantization; keep integer codes unchanged."""
    if multiplier == 1:
        return weight, scale, tensor_scale
    target = tensor_scale if tensor_scale is not None else scale if scale is not None else weight
    adjusted = target * multiplier
    if not torch.isfinite(adjusted).all():
        raise ValueError("TTS MLP range balance produced nonfinite weights or scales")
    if target is not weight and ((target != 0) & (adjusted == 0)).any():
        raise ValueError("TTS MLP range balance underflowed a quantization scale")
    if tensor_scale is not None:
        return weight, scale, adjusted
    if scale is not None:
        return weight, adjusted, None
    return adjusted, None, None


def _validate_request(text, language, max_new_tokens):
    if not isinstance(text, str) or not text.strip() or len(text) > 1024:
        raise ValueError("TTS text must contain 1–1024 characters")
    if language not in ("Chinese", "English"):
        raise ValueError("TTS profile admits Chinese or English")
    if isinstance(max_new_tokens, bool):
        raise ValueError("max_new_tokens must be an integer in [2, 256]")
    try:
        count = operator.index(max_new_tokens)
    except TypeError as exc:
        raise ValueError("max_new_tokens must be an integer in [2, 256]") from exc
    if not 2 <= count <= 256:
        raise ValueError("max_new_tokens must be an integer in [2, 256]")
    return count


def _split_text(text, tokenizer):
    if len(text) <= 128 and len(tokenizer(text)["input_ids"]) <= 32:
        return [text]
    # Shorter long-form segments leave room for EOS and avoid long-prefix drift.
    remaining = text.strip()
    segments = []
    while remaining:
        end = min(len(remaining), 128)
        while end and len(tokenizer(remaining[:end])["input_ids"]) > 12:
            end -= 1
        if not end:
            raise ValueError("TTS text cannot fit a long-form segment within 12 tokenizer tokens")
        if end < len(remaining):
            for pattern in (r"[。！？；\n]|[.!?;](?=\s|$)", r"[,，:：、]|\s"):
                boundaries = [match for match in re.finditer(pattern, remaining) if match.end() <= end]
                if boundaries:
                    boundary = boundaries[-1].end()
                    if len(tokenizer(remaining[:boundary])["input_ids"]) <= 12:
                        end = boundary
                        break
        segment = remaining[:end].strip()
        # Keep tiny sentence endings attached to their preceding clause.
        if (segments and segments[-1].endswith((",", "，", ":", "：", "、"))
                and segment.endswith(("。", "！", "？", ".", "!", "?"))
                and len(segment.rstrip("。！？.!?")) <= 8
                and len(segments[-1] + segment) <= 128
                and len(tokenizer(segments[-1] + segment)["input_ids"]) <= 16):
            segments[-1] += segment
        else:
            segments.append(segment)
        remaining = remaining[end:].lstrip()
    return segments


def _read_reference(path):
    import soundfile as sf

    path = Path(path)
    info = sf.info(path)
    if info.samplerate != 24000 or info.channels != 1 or not 12000 <= info.frames <= 480000:
        raise ValueError("TTS reference must be mono 24 kHz audio, 0.5–20 seconds")
    samples, rate = sf.read(path, dtype="float32")
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("TTS reference must contain finite mono samples")
    if not np.any(samples):
        raise ValueError("TTS reference must not be silent")
    return samples, rate


def _predictor_generate(owner, inputs_embeds, *, input_width, max_new_tokens, do_sample,
                        top_k=50, top_p=1.0, temperature=0.9, part=None, **kwargs):
    # shortcut: only the frozen checkpoint's 15-step sampling recipe is supported.
    if (max_new_tokens != 15 or do_sample is not True
            or (top_k, top_p, temperature) != (50, 1.0, 0.9)
            or input_width not in (1024, 2048)
            or inputs_embeds.shape != (1, 2, input_width)
            or owner.generation_config.eos_token_id is not None
            or owner.generation_config.repetition_penalty not in (None, 1)):
        raise ValueError("TTS Predictor requires the fixed 15-step sampling profile")
    # One op for temperature + top-k + softmax + draw. Spelled in Python that is
    # five dispatches per step against a 2048-wide row whose arithmetic is a few
    # microseconds, so the dispatches, not the math, set the Predictor's floor.
    # The profile pins top_p to 1.0, so there is no nucleus stage to carry.
    cache = None
    tokens = []
    embeddings = owner.model.get_input_embeddings()
    fused_head = part is not None and part.get("head_bank")
    for index in range(15):
        hidden = inputs_embeds if index == 0 else embeddings[index - 1](ids)
        if fused_head:
            # Steps differ only in which head they apply. Selecting moves the
            # fused head among tensors this handle already owns, which reaches
            # the retained decode stream as register patches, not a recapture.
            torch.ops.rpu.causal_decoder_select_lm_head(part["handle"], index)
        result = owner.model(inputs_embeds=owner.small_to_mtp_projection(hidden),
            past_key_values=cache, use_cache=True, output_attentions=False,
            output_hidden_states=False, return_dict=True)
        scores = owner.lm_head[index](result.last_hidden_state)[:, -1].float()
        ids = torch.ops.rpu.sample_topk_lastdim_host(
            scores.contiguous(), temperature, top_k).reshape(-1, 1)
        tokens.append(ids)
        cache = result.past_key_values
    return types.SimpleNamespace(sequences=torch.cat(tokens, dim=1))


def _decoder_geometry(core, name):
    config = core.config
    fields = ("num_hidden_layers", "hidden_size", "intermediate_size",
              "num_attention_heads", "num_key_value_heads", "head_dim")
    geometry = tuple(getattr(config, field) for field in fields)
    allowed = {(28, 1024, 3072, 16, 8, 128), (28, 2048, 6144, 16, 8, 128)} if name == "talker" else {
        (5, 1024, 3072, 16, 8, 128)}
    if (geometry not in allowed or len(core.layers) != geometry[0]
            or config.rms_norm_eps != 1e-6 or config.hidden_act != "silu"
            or config.attention_bias or config.sliding_window is not None
            or config.vocab_size != (3072 if name == "talker" else 2048)):
        raise ValueError(f"unsupported TTS {name} decoder profile")
    return geometry


# Request stages reported by execution_stats()["stage_ms"], in pipeline order.
# The timer itself is shared with the ASR pipeline; see _qwen3_audio_stages.py
# for why nested stages are charged to the innermost one.
_STAGE_NAMES = ("speaker_embedding", "talker", "predictor", "codec_decode")


class Qwen3TTSModel:
    """Reviewed 0.6B/1.7B Base, bounded voice cloning, batch one.

    Talker, Code Predictor and their output heads use the selected weight
    precision and Linear accumulation with FP16 activations on device.
    NVFP4 keeps output heads in FP16 with the same accumulation mode.
    The codec frontend also uses device FP16 when the native RoPE contract is available.
    Embedding, speaker encoder and composition use CPU FP32. The codec's twelve
    residual convolutions default to RPU FP16/ACC32; the rest stays CPU.
    Only speech output is exposed; internal per-layer traces are not an API.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._closed = True
        self._wrapper = None
        self._components = {}
        self._restore = []
        self._restore_generate = None
        self._restore_talker_generate = None
        self._last_generation = None
        self._stages = StageTimer("TTS", _STAGE_NAMES)
        self._stage_ms = None
        self._head_weights = []
        self._head_calls = [0] * 16
        self._retirement_error = None
        self._reference_key = self._voice_clone_prompt = None
        self._codec_convolutions = {"backend": "cpu", "layers": 0, "calls": 0}
        self._precision, self._linear_acc32 = "fp16", False
        self._num_cores = 8

    @classmethod
    def from_pretrained(cls, checkpoint, *, source, dtype=torch.float16, codec_backend="rpu",
                        precision="fp16", rpu_execution=None, max_seq_len=512):
        """Load reviewed Base weights; precision/ACC affect Talker, Predictor and heads."""
        from rpu_backend.adapters.qwen3_tts import load_cpu_model
        from .causal_lm import _claim_live_instance
        from ._qwen3_audio_precision import normalize_audio_precision

        if dtype is not torch.float16:
            raise RPUUnsupportedDtypeError("TTS device components require torch.float16; CPU components use float32")
        if codec_backend not in ("cpu", "rpu"):
            raise ValueError("codec_backend must be cpu or rpu (experimental residual convolutions)")
        if type(max_seq_len) is not int or max_seq_len < 1:
            raise ValueError("max_seq_len must be a positive integer")
        precision, acc32 = normalize_audio_precision(precision, rpu_execution, default_acc32=False)
        model = cls()
        model._max_seq_len = max_seq_len
        model._precision, model._linear_acc32 = precision, acc32
        from rpu_backend.runtime.topology import execution_core_count
        model._num_cores = execution_core_count(rpu_execution)
        model._codec_convolutions["backend"] = codec_backend
        _claim_live_instance(model)
        model._closed = False
        try:
            model._wrapper = load_cpu_model(checkpoint, source)
            torch.rpu.set_caching_allocator(True)
            model._install_rpu()
        except BaseException as error:
            try:
                model.close()
            except BaseException as cleanup_error:
                error.add_note(f"TTS cleanup failed: {cleanup_error!r}")
            raise
        return model

    def _install_head_bank(self):
        """Fuse the Predictor's 15 MTP heads into its retained decode stream.

        Each of the fixed 15 sampling steps applies a different `lm_head[i]` to a
        single row. As eager ops those were 15 launches per codec token at 812 us
        each, of which a two-point fit against the Talker's larger head attributed
        736-751 us to launch overhead rather than to reading the 2.10 MB weight.
        Installed as a bank, the selected head is the one trailing projection of
        the decode graph the Predictor already replays, and selecting a step
        rewrites its weight/scale address registers instead of rebuilding.

        The route is declined, leaving the eager heads in place, whenever the
        checked decode replay that carries the patches cannot be admitted.
        """
        part = self._components["predictor"]
        heads = self._head_weights[1:]
        if (self._num_cores != 8 or part["topology"].lm_head_tp != 8
                or len(heads) != 15
                or any(weight.dtype != torch.int8 or scale is None
                       or scale.dtype != torch.float16 for weight, scale in heads)):
            return
        shapes = {(tuple(weight.shape), tuple(scale.shape)) for weight, scale in heads}
        if len(shapes) != 1:
            return
        (weight_shape, _), = shapes
        if weight_shape != (part["vocab_size"], part["hidden_size"]):
            return
        torch.ops.rpu.causal_decoder_set_lm_head_bank(
            part["handle"], [weight for weight, _ in heads], [scale for _, scale in heads])
        # The install ends in one invalidate_model_state(), so re-resolve the
        # decode descriptor and signature the fused head now belongs to.
        from rpu_backend.graph import GraphSignature

        descriptor = list(torch.ops.rpu.causal_decoder_resolve_decode_stage_descriptor(part["handle"]))
        part["descriptor"] = descriptor
        part["signature"] = GraphSignature(op_id="qwen3_tts_predictor", shapes=[1, 1, part["hidden_size"]],
            dyn_dims=[part["handle"], *descriptor], dtypes=[torch.float16])
        part["head_bank"] = True

    def _install_rpu(self):
        from rpu_backend.graph import GraphCache, GraphSignature, GraphRuntimePolicy
        from rpu_backend.runtime._native_retirement import _InstalledNativeResource
        from ._qwen3_audio_precision import (
            pack_audio_linear, pack_audio_nvfp4, pack_audio_tensor_scales,
        )
        from rpu_backend.adapters.qwen3_tts._manifest import CHECKPOINT_PROFILES
        from rpu_backend.runtime.topology import DecoderTopology

        talker = self._wrapper.model.talker
        identity = self._wrapper._rpu_checkpoint_identity
        predictor_divisors = CHECKPOINT_PROFILES[identity["profile"]]["predictor_mlp_up_divisors"]
        geometry = {name: _decoder_geometry(core, name) for name, core in (
            ("talker", talker.model), ("predictor", talker.code_predictor.model))}
        # Four retained Graphs plus one serialized immediate-launch arena.
        self._graph_runtime_policy = GraphRuntimePolicy.from_environment(
            graph_arena_count=5, execution_core_count=self._num_cores)
        if not self._graph_runtime_policy.prepare_arenas():
            raise RuntimeError("TTS requires its cold five-arena Graph plan")
        for name, core in (("talker", talker.model), ("predictor", talker.code_predictor.model)):
            layers, width, intermediate, query_heads, kv_heads, head_dim = geometry[name]
            topology = DecoderTopology(self._num_cores,
                8 if self._num_cores == 8 else 4, self._num_cores,
                8 if self._num_cores == 8 else 4)
            graph = GraphCache(runtime_policy=self._graph_runtime_policy)
            handle = torch.ops.rpu.causal_decoder_create(
                num_cores=topology.num_cores, vocab_size=core.config.vocab_size)
            part = {"handle": handle, "graph": graph, "calls": 0, "tokens": 0,
                    "capacity": getattr(self, "_max_seq_len", 512) if name == "talker" else 32, "layers": layers,
                    "hidden_size": width, "kv_heads": kv_heads, "head_dim": head_dim,
                    "topology": topology,
                    "name": name, "generation_limit": 256, "device_head_only": name == "predictor",
                    "vocab_size": core.config.vocab_size,
                    "mlp_up_divisors": dict(predictor_divisors) if name == "predictor" else {}}
            self._components[name] = part
            handle_name = "_native_" + name + "_handle"
            setattr(self, handle_name, handle)
            part["resource"] = _InstalledNativeResource(
                self, handle, torch.ops.rpu.causal_decoder_destroy, graphs=(graph,),
                keepalive=part, label="TTS " + name, handle_name=handle_name,
            )
            part["resource"].take_ownership(self)
            groups, scales, tensor_scales = [], [], []
            def prepare_projection(path, partition):
                weights, weight_scales, layer_tensor_scales = [], [], []
                projection_cores = topology.mlp_tp if path.startswith("mlp.") else topology.attn_tp
                for layer_index, layer in enumerate(core.layers):
                    source = layer.get_submodule(path).weight.detach()
                    tensor_scale = None
                    if self._precision == "nvfp4":
                        weight, scale, tensor_scale = pack_audio_nvfp4(source, partition, num_cores=projection_cores)
                    else:
                        weight, scale = pack_audio_linear(source, partition, self._precision,
                                                          num_cores=projection_cores)
                    # This checkpoint has a large SwiGLU channel. Up/s and
                    # Down*s preserve the real-valued function while keeping
                    # the intermediate product representable in FP16.
                    divisor = part["mlp_up_divisors"].get(layer_index, 1)
                    multiplier = (1 / divisor if path == "mlp.up_proj" else
                                  divisor if path == "mlp.down_proj" else 1)
                    weight, scale, tensor_scale = _rescale_mlp_payload(
                        weight, scale, tensor_scale, multiplier)
                    if tensor_scale is not None:
                        layer_tensor_scales.append(tensor_scale)
                    weights.append(weight.to("rpu"))
                    if scale is not None:
                        weight_scales.append(scale.to("rpu"))
                groups.append(weights)
                scales.append(weight_scales)
                if layer_tensor_scales:
                    tensor_scales.append(pack_audio_tensor_scales(layer_tensor_scales).to("rpu"))

            for path, partition in (("self_attn.q_proj", 1), ("self_attn.k_proj", 1),
                                    ("self_attn.v_proj", 1), ("self_attn.o_proj", 0)):
                prepare_projection(path, partition)
            for path in ("self_attn.q_norm", "self_attn.k_norm", "input_layernorm", "post_attention_layernorm"):
                groups.append([layer.get_submodule(path).weight.detach().half().to("rpu") for layer in core.layers])
            for path, partition in (("mlp.gate_proj", 1), ("mlp.up_proj", 1), ("mlp.down_proj", 0)):
                prepare_projection(path, partition)
            positions = torch.arange(part["capacity"]).view(1, -1)
            if name == "talker":
                positions = positions.unsqueeze(0).expand(3, -1, -1)
            # Replay binds immutable table/norm owners through their version counters.
            with torch.inference_mode(False), torch.no_grad():
                cos, sin = core.rotary_emb(torch.zeros(1, 1, width, dtype=torch.float16), positions)
                if name == "talker":
                    if not all(torch.equal(value[0], value[index]) for value in (cos, sin) for index in (1, 2)):
                        raise ValueError("TTS requires coincident position axes")
                    cos, sin = cos[0], sin[0]
                cos = cos.squeeze(0)[:, :64].contiguous().to("rpu")
                sin = sin.squeeze(0)[:, :64].contiguous().to("rpu")
                norm = core.norm.weight.detach().half().to("rpu")
            part["storage"] = (groups, cos, sin, norm, scales)
            part["tensor_scales"] = tensor_scales
            part["planner"] = core
            core._rpu_decoder_handle = handle
            core._rpu_decoder_graph_cache = graph
            core._rpu_decoder_topology = topology
            args = [handle, *groups, cos, sin, norm, query_heads, kv_heads, head_dim,
                    width, intermediate, 1e-6, True]
            if self._precision == "fp16":
                torch.ops.rpu.causal_decoder_set_weights(*args)
            elif self._precision == "nvfp4":
                torch.ops.rpu.causal_decoder_set_weights_nvfp4(*args, [], [], *scales, *tensor_scales)
            else:
                torch.ops.rpu.causal_decoder_set_weights_w8a16(*args, [], [], *scales)
            actual = tuple(torch.ops.rpu.causal_decoder_topology(handle))
            if actual != topology.identity()[1:]:
                raise RuntimeError(f"TTS {name} native topology differs from its weight layout: {actual}")
            part["native_topology"] = actual
            torch.ops.rpu.causal_decoder_set_linear_acc32(handle, self._linear_acc32)
            if name == "predictor" and self._precision != "nvfp4":
                torch.ops.rpu.causal_decoder_set_fast_replay(handle, True)
            if name == "talker":
                from rpu_backend.adapters.qwen3 import lookup_causal_decoder

                envelope = lookup_causal_decoder("qwen3", part["layers"], width)
                torch.ops.rpu.causal_decoder_set_chunk_envelope(
                    handle, part["capacity"], envelope.chunk)
            descriptor = list(torch.ops.rpu.causal_decoder_resolve_decode_stage_descriptor(handle))
            part["descriptor"] = descriptor
            part["signature"] = GraphSignature(op_id="qwen3_tts_" + name, shapes=[1, 1, width],
                dyn_dims=[handle, *descriptor], dtypes=[torch.float16])
            self._restore.append((core, core.forward))

            def forward(owner, inputs_embeds=None, past_key_values=None, attention_mask=None,
                        position_ids=None, cache_position=None, input_ids=None,
                        output_attentions=False, output_hidden_states=False, part=part, **kwargs):
                if input_ids is not None or output_attentions:
                    raise ValueError("TTS speech components require embeddings and no attention trace")
                return _decoder_forward(part, inputs_embeds, past_key_values, attention_mask,
                                        position_ids, cache_position, output_hidden_states)
            core.forward = types.MethodType(forward, core)

        calls = self._head_calls
        head_precision = "fp16" if self._precision == "nvfp4" else self._precision
        for index, head in enumerate([talker.codec_head, *talker.code_predictor.lm_head]):
            part = self._components["talker" if index == 0 else "predictor"]
            weight, scale = pack_audio_linear(head.weight.detach(), 1, head_precision,
                bounded_head=True, num_cores=part["topology"].lm_head_tp)
            weight = weight.to("rpu")
            scale = scale.to("rpu") if scale is not None else None
            self._head_weights.append((weight, scale))
            self._restore.append((head, head.forward))
            part = self._components["talker" if index == 0 else "predictor"]

            def linear(owner, hidden, weight=weight, scale=scale, index=index, part=part):
                if part.get("head_bank"):
                    # The fused head already applied this step's projection inside
                    # the captured decode graph. One eager launch of this GEMV cost
                    # 812 us, of which only ~70 us was the weight read.
                    logits = part.pop("head_logits", None)
                    if logits is None:
                        raise RuntimeError("TTS head bank produced no device logits")
                    result = logits.cpu().float()
                    if not torch.isfinite(result).all():
                        raise FloatingPointError("non-finite TTS device logits")
                    calls[index] += 1
                    return result.reshape(*hidden.shape[:-1], owner.out_features)
                device_hidden = _take_head_input(part, hidden)
                if device_hidden is None:
                    device_hidden = hidden.half().contiguous().to("rpu")
                logical_shape = ({"logical_out_features": owner.out_features}
                                 if weight.dtype == torch.uint8 else {})
                result = torch.ops.rpu.linear_with_accumulation(
                    device_hidden.reshape(-1, part["hidden_size"]), weight, None, 1,
                    part["topology"].lm_head_tp,
                    self._linear_acc32, scale, **logical_shape).cpu().float()
                if not torch.isfinite(result).all():
                    raise FloatingPointError("non-finite TTS device logits")
                calls[index] += 1
                return result.reshape(*hidden.shape[:-1], owner.out_features)
            head.forward = types.MethodType(linear, head)
        self._install_head_bank()
        predictor = talker.code_predictor
        self._restore_generate = (predictor, predictor.generate)
        sample = partial(_predictor_generate, input_width=geometry["talker"][1],
                         part=self._components["predictor"])

        def timed_predictor(owner, *args, **kwargs):
            # Runs inside the Talker's loop; the stage timer keeps the two disjoint.
            with self._stages.stage("predictor"):
                return sample(owner, *args, **kwargs)
        predictor.generate = types.MethodType(timed_predictor, predictor)
        self._restore_talker_generate = (talker, talker.generate)
        talker.generate = self._generate_talker
        from rpu_backend.adapters.qwen3_tts.codec import (
            install_codec_activations, install_codec_convolutions,
        )

        decoder = self._wrapper.model.speech_tokenizer.model.decoder
        install_codec_convolutions(decoder, self._restore)
        install_codec_activations(decoder, self._restore)
        from rpu_backend.adapters.qwen3_tts.codec_transformer import install_codec_transformer

        install_codec_transformer(self, decoder.pre_transformer)
        if self._codec_convolutions["backend"] == "rpu":
            from rpu_backend.adapters.qwen3_tts.codec import install_rpu_convolutions

            install_rpu_convolutions(decoder, self._restore, self._codec_convolutions,
                                     num_cores=self._num_cores)
        # Outermost codec wrap: the whole code-to-waveform decode, including the
        # transformer frontend and every convolution the installs above replaced.
        decode = decoder.forward
        self._restore.append((decoder, decode))

        def timed_decode(owner, *args, **kwargs):
            with self._stages.stage("codec_decode"):
                return decode(*args, **kwargs)
        decoder.forward = types.MethodType(timed_decode, decoder)

    def _generate_talker(self, *args, **kwargs):
        with self._stages.stage("talker"):
            result = self._restore_talker_generate[1](*args, **kwargs)
        ids = result.sequences
        eos = self._wrapper.model.config.talker_config.codec_eos_token_id
        finished = bool(ids.shape[1] and ids[0, -1].item() == eos)
        self._last_generation = {"steps": ids.shape[1], "eos": finished}
        if not finished:
            raise RuntimeError("TTS reached max_new_tokens without EOS; shorten the text or increase the limit (up to 256)")
        return result

    def stream_synthesize(self, text, *, reference_audio, language="Chinese", max_new_tokens=256):
        """Return a generator yielding independent float32 mono chunks at 24 kHz.

        Chunks follow internal text segments. Close the generator when stopping early.
        """
        limit = _validate_request(text, language, max_new_tokens)
        return self._stream_synthesize(text, reference_audio, language, limit)

    def _stream_synthesize(self, text, reference_audio, language, limit):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("TTS requests, stats and close must be serial")
        steps = []
        generation_started = completed = False
        active_ns = 0
        try:
            if self._closed:
                raise RuntimeError("TTS model is closed")
            if self._retirement_error is not None:
                raise RuntimeError("TTS retirement failed; restart the process") from self._retirement_error
            self._stages.reset()
            self._stage_ms = None
            step_start = time.perf_counter_ns()
            segments = _split_text(text, self._wrapper.processor.tokenizer)
            reference = _read_reference(reference_audio)
            self._components["talker"]["generation_limit"] = limit
            self._last_generation = None
            for part in self._components.values():
                part["request_prefills"], part["request_input_lengths"] = [], {}
                part.pop("last_execution_plan", None)
            with torch.inference_mode():
                key = hashlib.sha256(reference[0].tobytes()).digest()
                if key != self._reference_key:
                    if reference[1] != self._wrapper.model.speaker_encoder_sample_rate:
                        raise ValueError("TTS frozen speaker encoder requires 24 kHz audio")
                    with self._stages.stage("speaker_embedding"):
                        embedding = self._wrapper.model.extract_speaker_embedding(
                            audio=reference[0], sr=reference[1])
                    # x-vector-only mode discards reference codec tokens entirely.
                    self._voice_clone_prompt = dict(ref_code=[None], ref_spk_embedding=[embedding],
                        x_vector_only_mode=[True], icl_mode=[False])
                    self._reference_key = key
            segment_rng = torch.random.get_rng_state() if len(segments) > 1 else None
            generation_started = True
            active_ns += time.perf_counter_ns() - step_start
            for index, segment in enumerate(segments):
                seg_start = time.perf_counter_ns()
                try:
                    self._last_generation = None
                    utterance = segment
                    if len(segments) > 1 and not segment.endswith(("。", "！", "？", ".", "!", "?")):
                        utterance = segment.rstrip(",，;；:：、") + ("。" if language == "Chinese" else ".")
                    with torch.inference_mode():
                        # A segment's sampling must not depend on earlier segments' lengths.
                        with torch.random.fork_rng(devices=[]) if index else nullcontext():
                            if index:
                                torch.random.set_rng_state(segment_rng)
                            waves, rate = self._wrapper.generate_voice_clone(text=utterance, language=language,
                                voice_clone_prompt=self._voice_clone_prompt, non_streaming_mode=True,
                                do_sample=True, subtalker_dosample=True, max_new_tokens=limit)
                    if self._last_generation is None or not self._last_generation["eos"]:
                        raise RuntimeError("TTS segment did not complete generation")
                    if len(waves) != 1 or rate != 24000:
                        raise RuntimeError("TTS output shape or sample rate differs from the profile")
                    waveform = np.asarray(waves[0], dtype=np.float32)
                    if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
                        raise FloatingPointError("TTS produced empty or non-finite audio")
                    if not np.any(waveform):
                        raise FloatingPointError("TTS produced silent audio")
                    chunk = waveform.copy()
                    steps.append(self._last_generation["steps"])
                    active_ns += time.perf_counter_ns() - seg_start
                except Exception as error:
                    error.add_note(
                        f"TTS segment {index + 1} of {len(segments)} failed after "
                        f"{len(steps)} completed audio chunk(s)")
                    raise
                yield chunk, rate
            self._last_generation = {"steps": sum(steps), "eos": True,
                                     "segment_steps": tuple(steps)}
            self._stage_ms = self._stages.milliseconds(active_ns)
            completed = True
        finally:
            if generation_started and not completed:
                self._last_generation = {"steps": sum(steps), "eos": False,
                                         "segment_steps": tuple(steps)}
            self._lock.release()

    def synthesize(self, text, *, reference_audio, language="Chinese", max_new_tokens=256):
        """Return an independent float32 mono waveform and sample rate (24 kHz).

        Sampling follows the frozen checkpoint. Set torch.manual_seed for repeatability.
        Long text uses 12-token segments, up to 16 tokens to retain short sentence endings.
        max_new_tokens applies to each segment; no partial waveform is returned on failure.
        Reaching the cap without EOS raises before waveform decoding. Reference audio
        is a local mono 24 kHz file; cloning uses speaker embedding without a transcript.
        """
        stream = self.stream_synthesize(text, reference_audio=reference_audio,
                                        language=language, max_new_tokens=max_new_tokens)
        outputs = []
        try:
            for waveform, rate in stream:
                outputs.append(waveform)
        except Exception as error:
            error.add_note("TTS synthesis failed; no partial audio returned")
            raise
        return (outputs[0] if len(outputs) == 1 else np.concatenate(outputs)), rate

    def execution_stats(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("TTS requests, stats and close must be serial")
        try:
            if self._closed:
                raise RuntimeError("TTS model is closed")
            if self._retirement_error is not None:
                raise RuntimeError("TTS retirement failed; restart the process") from self._retirement_error
            result = {"head_calls": list(self._head_calls), "components": {},
                      "text_precision": self._precision, "decoder_precision": self._precision,
                      "num_cores": self._num_cores,
                      "head_precision": "fp16" if self._precision == "nvfp4" else self._precision,
                      "text_linear_acc32": self._linear_acc32,
                      "generation": dict(self._last_generation) if self._last_generation else None,
                      # Exclusive per-stage wall time of the last completed request.
                      "stage_ms": dict(self._stage_ms) if self._stage_ms else None}
            identity = getattr(self._wrapper, "_rpu_checkpoint_identity", None)
            if identity is not None:
                result["model_profile"] = identity["profile"]
            codec_rpu = "codec_transformer" in self._components
            device = self._components["codec_transformer"].get("last_device", "rpu") if codec_rpu else "cpu"
            result["codec_frontend"] = {"device": device,
                "fallback_reason": (None if device == "rpu" else "native codec frontend unavailable")
                    if codec_rpu else "native lacks explicit TTS codec mode"}
            result["codec_convolutions"] = dict(self._codec_convolutions)
            for name, part in self._components.items():
                graph = part["graph"]
                result["components"][name] = {"calls": part["calls"], "tokens": part["tokens"],
                    "graph_size": graph.size(), "graph_replays": sum(e.replay_count for e in graph.snapshot()),
                    "graph_invariant_ok": graph.cache_invariant_ok(),
                    "native_topology": part.get("native_topology"),
                    "request_input_lengths": dict(part.get("request_input_lengths", {})),
                    "execution_plans": {
                        "prefill": [_execution_receipt(row) for row in part.get("request_prefills", [])],
                        "last": _execution_receipt(part.get("last_execution_plan"))},
                    "graph_core_budget": getattr(getattr(graph, "runtime_policy", None),
                                                 "execution_core_count", None)}
                if name in ("talker", "predictor"):
                    groups, _, _, _, scales = part["storage"]
                    result["components"][name].update({"hidden_size": part["hidden_size"],
                        "weight_dtype": str(groups[0][0].dtype), "has_weight_scales": bool(scales[0]),
                        "tensor_scale_tables": len(part["tensor_scales"]),
                        "mlp_up_divisors": dict(part["mlp_up_divisors"]),
                        "linear_acc32": self._linear_acc32})
            return result
        finally:
            self._lock.release()

    def close(self):
        from .causal_lm import _release_live_instance

        if not self._lock.acquire(blocking=False):
            raise RuntimeError("TTS requests, stats and close must be serial")
        try:
            if self._closed:
                return
            if self._retirement_error is not None:
                raise RuntimeError("TTS retirement failed; restart the process") from self._retirement_error
            try:
                # Retire every captured queue before either component's weights.
                for part in self._components.values():
                    part["graph"].clear()
                for part in reversed(list(self._components.values())):
                    part["resource"].retire()
                    part["handle"] = None
            except BaseException as error:
                self._retirement_error = error
                # Retain the whole parent, including the other component and all
                # heads. Native cleanup is uncertain and must never be retried.
                for part in self._components.values():
                    part["resource"].retain_failure(error, self)
                raise
            for module, original in reversed(self._restore):
                module.forward = original
            self._restore.clear()
            if self._restore_generate is not None:
                predictor, original = self._restore_generate
                predictor.generate = original
                self._restore_generate = None
            if self._restore_talker_generate is not None:
                talker, original = self._restore_talker_generate
                talker.generate = original
                self._restore_talker_generate = None
            self._components.clear()
            self._head_weights.clear()
            self._wrapper = None
            self._reference_key = self._voice_clone_prompt = None
            # Drop the last loop's storage before admitting a new resident model.
            part = None
            self._closed = True
            _release_live_instance(self)
        finally:
            self._lock.release()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("TTS model is closed")
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _take_head_input(part, hidden):
    previous = part.pop("head_hidden", None)
    device_hidden = part.pop("head_device", None)
    return device_hidden if hidden is previous and hidden.shape[1] == 1 else None


def _execution_receipt(receipt):
    if receipt is None:
        return None
    result = {key: copy.deepcopy(value) for key, value in receipt.items() if key != "selected"}
    selected = receipt.get("selected")
    if selected is not None:
        result["selected"] = selected.as_dict()
    return result


def _decoder_forward(part, hidden, cache, attention_mask, position_ids, cache_position, output_hidden_states):
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from .cache import RPUCache

    width = part["hidden_size"]
    if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != width:
        raise ValueError(f"TTS decoder requires [1, tokens, {width}] inputs")
    if hidden.device.type != "cpu" or hidden.dtype != torch.float32:
        raise ValueError("TTS decoder requires finite CPU FP32 inputs")
    # One FP16 scan both admits the input and clears the cast the device needs: a
    # non-finite FP32 value is still non-finite after .half(), so a value that only
    # fails here was finite and left the FP16 range. Scanning FP32 as well repeated
    # the same traversal over twice the bytes on every Predictor step.
    half_hidden = hidden.half()
    if not torch.isfinite(half_hidden).all():
        if not torch.isfinite(hidden).all():
            raise ValueError("TTS decoder requires finite CPU FP32 inputs")
        raise FloatingPointError("TTS input exceeds FP16 range")
    if attention_mask is not None and not bool(attention_mask.all()):
        raise ValueError("TTS padding is outside this profile")
    if not isinstance(cache, RPUCache):
        if cache is not None and cache.get_seq_length() != 0:
            raise ValueError("cannot replace a populated CPU TTS cache")
        if "cache" not in part:
            part["cache"] = RPUCache(part["layers"], 1, part["capacity"], part["kv_heads"],
                part["head_dim"], attn_tp=part["topology"].attn_tp if "topology" in part else 8)
        cache = part["cache"]
        cache.reset_to_position(0)
    if "topology" in part:
        from rpu_backend.runtime.topology import validate_decoder_cache_topology
        validate_decoder_cache_topology(part["topology"], cache, batch_size=1)
    count = hidden.shape[1]
    if count < 1 or cache.position + count > part["capacity"]:
        raise ValueError("TTS cache capacity exceeded")
    if part["name"] == "talker" and cache.position == 0 and count + part["generation_limit"] > part["capacity"]:
        raise ValueError("TTS prompt plus requested generation exceeds cache capacity")
    # Build the contiguous range only for a caller that supplies positions to check
    # against it. The Predictor supplies neither, so this was a per-step allocation
    # whose only consumer was two skipped comparisons.
    if position_ids is not None or cache_position is not None:
        expected = torch.arange(cache.position, cache.position + count)
        if position_ids is not None:
            if position_ids.ndim == 3:
                if position_ids.shape[0] != 3 or not torch.equal(position_ids[0], position_ids[1]) or not torch.equal(position_ids[0], position_ids[2]):
                    raise ValueError("TTS requires coincident position axes")
                position_ids = position_ids[0]
            if not torch.equal(position_ids.cpu().reshape(-1), expected.to(position_ids.dtype)):
                raise ValueError("TTS positions must match the contiguous cache range")
        if cache_position is not None and not torch.equal(cache_position.cpu().reshape(-1), expected):
            raise ValueError("TTS cache_position differs from the contiguous cache range")
    part.pop("head_hidden", None)
    part.pop("head_device", None)
    chunks = []
    head_only = part.get("device_head_only", False) and not output_hidden_states
    if part.get("head_bank") and not head_only:
        # The fused head consumed the final hidden state inside the graph and
        # returned logits in its place, so there is no hidden state to hand back.
        raise ValueError("the fused TTS head returns logits, not a hidden state")
    # shortcut: Predictor keeps decode until its five-layer prefill envelope is measured.
    sizes = [count] if count > 1 and part["name"] == "talker" else [1] * count
    offset = 0
    for size in sizes:
        half_input = half_hidden[:, offset:offset + size].contiguous()
        descriptor, signature = part["descriptor"], part["signature"]
        execution_len, selected = size, None
        if size > 1:
            from rpu_backend.graph import GraphSignature
            from rpu_backend.runtime.decoder import _text_prefill_execution_plan, prefill_position_key

            execution_len, _, plan = _text_prefill_execution_plan(
                part["planner"], part["handle"], cache, size)
            descriptor = plan.selected.stage_tuple.physical_descriptor
            selected = plan.selected
            signature = GraphSignature(op_id="qwen3_tts_prefill_" + part["name"],
                shapes=[1, size, execution_len, width],
                dyn_dims=[part["handle"], prefill_position_key(execution_len, cache.position),
                          *plan.graph_key_words()], dtypes=[torch.float16])
            previous = part.get("prefill_signature")
            if previous is not None and previous != signature:
                part["graph"].evict(previous)
            if execution_len > size:
                half_input = torch.nn.functional.pad(half_input, (0, 0, 0, execution_len - size))
        device_input = half_input.to("rpu")
        with part["graph"].capture(signature):
            raw = torch.ops.rpu.causal_decoder_forward(part["handle"], device_input,
                cache.k_caches, cache.v_caches, None, cache.position, True,
                planned_stage_descriptor=descriptor)
        receipt = {"logical_len": size, "execution_len": execution_len,
            "padding_rows": execution_len - size, "position": int(cache.position),
            "physical_descriptor": descriptor, "selected": selected,
            "torch_num_threads": torch.get_num_threads(),
            "inference_mode": torch.is_inference_mode_enabled()}
        part["last_execution_plan"] = receipt
        counts = part.setdefault("request_input_lengths", {})
        counts[str(size)] = counts.get(str(size), 0) + 1
        if size > 1:
            part.setdefault("request_prefills", []).append(receipt)
        if size > 1:
            part["prefill_signature"] = signature
        if head_only:
            borrowed = part["vocab_size"] if part.get("head_bank") else width
            if raw.shape != (1, size, borrowed) or raw.dtype != torch.float16:
                raise RuntimeError("invalid TTS Predictor device hidden state")
            cache.update_position(size)
            offset += size
            continue
        value = raw[:, :size].cpu().float()
        if value.shape != (1, size, width) or not torch.isfinite(value).all():
            raise FloatingPointError("non-finite or invalid TTS device hidden state")
        chunks.append(value)
        cache.update_position(size)
        offset += size
    part["calls"] += 1
    part["tokens"] += count
    if head_only:
        # Borrow only until the frozen Predictor immediately consumes its head;
        # the head publishes independent finite-checked CPU logits.
        result = raw[:, -1:]
        if part.get("head_bank"):
            # The fused head already projected this row inside the same graph, so
            # the borrowed tensor is the step's logits. The Predictor reaches them
            # through its head module, which reads this slot instead of launching.
            part["head_logits"] = result
            return BaseModelOutputWithPast(last_hidden_state=result, past_key_values=cache)
        part["head_hidden"], part["head_device"] = result, result
        return BaseModelOutputWithPast(last_hidden_state=result, past_key_values=cache)
    result = torch.cat(chunks, dim=1)
    if count == 1:
        # The head immediately consumes this exact result before another decoder replay.
        part["head_hidden"], part["head_device"] = result, raw
    # The upstream speech wrapper consumes only the final normalized hidden state.
    return BaseModelOutputWithPast(last_hidden_state=result, past_key_values=cache,
        hidden_states=(result,) if output_hidden_states else None)
