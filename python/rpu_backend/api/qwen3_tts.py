"""Bounded Qwen3-TTS Base speech inference with device autoregressive components."""
from __future__ import annotations

import hashlib
from contextlib import nullcontext
import operator
import re
from pathlib import Path
import threading
import types

import numpy as np
import torch

from .errors import RPUUnsupportedDtypeError


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
        if (segments and (segments[-1].endswith((",", "，", ":", "：", "、"))
                          or not segment.rstrip("。！？.!?"))
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


def _predictor_generate(owner, inputs_embeds, *, max_new_tokens, do_sample,
                        top_k=50, top_p=1.0, temperature=0.9, **kwargs):
    # shortcut: only the frozen checkpoint's 15-step sampling recipe is supported.
    if (max_new_tokens != 15 or do_sample is not True
            or (top_k, top_p, temperature) != (50, 1.0, 0.9)
            or inputs_embeds.shape != (1, 2, 1024)
            or owner.generation_config.eos_token_id is not None
            or owner.generation_config.repetition_penalty not in (None, 1)):
        raise ValueError("TTS Predictor requires the fixed 15-step sampling profile")
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper

    warpers = (TemperatureLogitsWarper(temperature), TopKLogitsWarper(top_k))
    cache = None
    tokens = []
    embeddings = owner.model.get_input_embeddings()
    for index in range(15):
        hidden = inputs_embeds if index == 0 else embeddings[index - 1](ids)
        result = owner.model(inputs_embeds=owner.small_to_mtp_projection(hidden),
            past_key_values=cache, use_cache=True, output_attentions=False,
            output_hidden_states=False, return_dict=True)
        scores = owner.lm_head[index](result.last_hidden_state)[:, -1].float()
        for warper in warpers:
            scores = warper(None, scores)
        ids = torch.multinomial(scores.softmax(-1), num_samples=1)
        tokens.append(ids)
        cache = result.past_key_values
    return types.SimpleNamespace(sequences=torch.cat(tokens, dim=1))


def _tts_mlp_weight(weight, component, layer_index, path):
    # The frozen Predictor's third-layer SiLU*up exceeds FP16; balance its two
    # linear projections before packing so every token can stay on device.
    if component == "predictor" and layer_index == 2:
        if path == "mlp.up_proj":
            weight = weight / 8
        elif path == "mlp.down_proj":
            weight = weight * 8
    return weight.half()


class Qwen3TTSModel:
    """Frozen 0.6B Base, bounded non-streaming voice cloning, batch one.

    Talker, Code Predictor and their output heads execute FP16 on device.
    The codec frontend also uses device FP16 when the native RoPE contract is available.
    Embedding, speaker encoder and composition use CPU FP32. The codec's 31 decoding
    Conv1d layers default to RPU FP16/ACC32; native pairs also offload 12 intermediate
    SnakeBeta layers. Transposed convolutions and remaining activations stay CPU.
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
        self._head_weights = []
        self._head_calls = [0] * 16
        self._retirement_error = None
        self._reference_key = self._voice_clone_prompt = None
        self._codec_convolutions = {"backend": "cpu", "layers": 0, "calls": 0}

    @classmethod
    def from_pretrained(cls, checkpoint, *, source, dtype=torch.float16, codec_backend="rpu", max_seq_len=512):
        """Load frozen weights/source; codec_backend=cpu restores CPU FP32 Conv1d."""
        from rpu_backend.adapters.qwen3_tts import load_cpu_model
        from .causal_lm import _claim_live_instance

        if dtype is not torch.float16:
            raise RPUUnsupportedDtypeError("TTS device components require torch.float16; CPU components use float32")
        if codec_backend not in ("cpu", "rpu"):
            raise ValueError("codec_backend must be cpu or rpu (experimental codec Conv1d)")
        if type(max_seq_len) is not int or max_seq_len < 1:
            raise ValueError("max_seq_len must be a positive integer")
        model = cls()
        model._max_seq_len = max_seq_len
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

    def _install_rpu(self):
        from rpu_backend.graph import GraphCache, GraphSignature, GraphRuntimePolicy
        from rpu_backend.runtime.weights import transform_linear_weight
        from rpu_backend.runtime._native_retirement import _InstalledNativeResource

        # Five retained Graphs plus one serialized immediate-launch arena.
        self._graph_runtime_policy = GraphRuntimePolicy.from_environment(graph_arena_count=6)
        if not self._graph_runtime_policy.prepare_arenas():
            raise RuntimeError("TTS requires its cold six-arena Graph plan")
        talker = self._wrapper.model.talker
        for name, core in (("talker", talker.model), ("predictor", talker.code_predictor.model)):
            graph = GraphCache(runtime_policy=self._graph_runtime_policy)
            handle = torch.ops.rpu.causal_decoder_create()
            part = {"handle": handle, "graph": graph, "calls": 0, "tokens": 0,
                    "capacity": getattr(self, "_max_seq_len", 512) if name == "talker" else 32, "layers": len(core.layers),
                    "name": name, "generation_limit": 256, "device_head_only": name == "predictor"}
            self._components[name] = part
            handle_name = "_native_" + name + "_handle"
            setattr(self, handle_name, handle)
            part["resource"] = _InstalledNativeResource(
                self, handle, torch.ops.rpu.causal_decoder_destroy, graphs=(graph,),
                keepalive=part, label="TTS " + name, handle_name=handle_name,
            )
            part["resource"].take_ownership(self)
            groups = []
            for path, partition in (("self_attn.q_proj", 1), ("self_attn.k_proj", 1),
                                    ("self_attn.v_proj", 1), ("self_attn.o_proj", 0)):
                groups.append([transform_linear_weight(layer.get_submodule(path).weight.detach().half(), partition, 8).to("rpu")
                               for layer in core.layers])
            for path in ("self_attn.q_norm", "self_attn.k_norm", "input_layernorm", "post_attention_layernorm"):
                groups.append([layer.get_submodule(path).weight.detach().half().to("rpu") for layer in core.layers])
            for path, partition in (("mlp.gate_proj", 1), ("mlp.up_proj", 1), ("mlp.down_proj", 0)):
                groups.append([transform_linear_weight(_tts_mlp_weight(
                    layer.get_submodule(path).weight.detach(), name, index, path), partition, 8).to("rpu")
                    for index, layer in enumerate(core.layers)])
            positions = torch.arange(part["capacity"]).view(1, -1)
            if name == "talker":
                positions = positions.unsqueeze(0).expand(3, -1, -1)
            # Replay binds immutable table/norm owners through their version counters.
            with torch.inference_mode(False), torch.no_grad():
                cos, sin = core.rotary_emb(torch.zeros(1, 1, 1024, dtype=torch.float16), positions)
                if name == "talker":
                    if not all(torch.equal(value[0], value[index]) for value in (cos, sin) for index in (1, 2)):
                        raise ValueError("TTS requires coincident position axes")
                    cos, sin = cos[0], sin[0]
                cos = cos.squeeze(0)[:, :64].contiguous().to("rpu")
                sin = sin.squeeze(0)[:, :64].contiguous().to("rpu")
                norm = core.norm.weight.detach().half().to("rpu")
            part["storage"] = (groups, cos, sin, norm)
            part["planner"] = core
            core._rpu_decoder_handle = handle
            core._rpu_decoder_graph_cache = graph
            torch.ops.rpu.causal_decoder_set_weights(handle, *groups, cos, sin, norm, 16, 8, 128, 1024, 3072, 1e-6, True)
            torch.ops.rpu.causal_decoder_set_linear_acc32(handle, True)
            if name == "predictor":
                torch.ops.rpu.causal_decoder_set_fast_replay(handle, True)
            if name == "talker":
                from rpu_backend.adapters.qwen3 import lookup_causal_decoder

                envelope = lookup_causal_decoder("qwen3", part["layers"], 1024)
                torch.ops.rpu.causal_decoder_set_chunk_envelope(
                    handle, part["capacity"], envelope.chunk)
            else:
                torch.ops.rpu.causal_decoder_set_chunk_envelope(handle, 32, 32)
            descriptor = list(torch.ops.rpu.causal_decoder_resolve_decode_stage_descriptor(handle))
            part["descriptor"] = descriptor
            part["signature"] = GraphSignature(op_id="qwen3_tts_" + name, shapes=[1, 1, 1024],
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
        for index, head in enumerate([talker.codec_head, *talker.code_predictor.lm_head]):
            weight = transform_linear_weight(head.weight.detach().half(), 1, 8).to("rpu")
            self._head_weights.append(weight)
            self._restore.append((head, head.forward))
            part = self._components["talker" if index == 0 else "predictor"]

            def linear(owner, hidden, weight=weight, index=index, part=part):
                fused = _take_predictor_logits(part, hidden, index)
                device_hidden = _take_head_input(part, hidden)
                if fused is None:
                    if device_hidden is None:
                        device_hidden = hidden.half().contiguous().to("rpu")
                    fused = torch.ops.rpu.linear_with_accumulation(
                        device_hidden.reshape(-1, 1024), weight, None, 1, 8, False)
                result = fused.cpu().float()
                if not torch.isfinite(result).all():
                    raise FloatingPointError("non-finite TTS device logits")
                calls[index] += 1
                return result.reshape(*hidden.shape[:-1], owner.out_features)
            head.forward = types.MethodType(linear, head)
        if hasattr(torch.ops.rpu, "causal_decoder_set_tts_predictor_heads"):
            if not hasattr(torch.ops.rpu, "causal_decoder_tts_predictor_logits"):
                raise RuntimeError("native TTS Predictor head fusion is incomplete")
            part = self._components["predictor"]
            torch.ops.rpu.causal_decoder_set_tts_predictor_heads(part["handle"], self._head_weights[1:])
            part["fused_heads"] = True
            part["fused_head_calls"] = 0
            descriptor = list(torch.ops.rpu.causal_decoder_resolve_decode_stage_descriptor(part["handle"]))
            part["descriptor"] = descriptor
            part["signature"] = GraphSignature(op_id="qwen3_tts_predictor", shapes=[1, 1, 1024],
                dyn_dims=[part["handle"], *descriptor], dtypes=[torch.float16])
        predictor = talker.code_predictor
        self._restore_generate = (predictor, predictor.generate)
        predictor.generate = types.MethodType(_predictor_generate, predictor)
        self._restore_talker_generate = (talker, talker.generate)
        talker.generate = self._generate_talker
        from rpu_backend.adapters.qwen3_tts.codec import install_codec_activations, install_codec_convolutions

        install_codec_convolutions(self._wrapper.model.speech_tokenizer.model.decoder, self._restore)
        install_codec_activations(self._wrapper.model.speech_tokenizer.model.decoder, self._restore)
        from rpu_backend.adapters.qwen3_tts.codec_transformer import install_codec_transformer

        install_codec_transformer(self, self._wrapper.model.speech_tokenizer.model.decoder.pre_transformer)
        if self._codec_convolutions["backend"] == "rpu":
            from rpu_backend.adapters.qwen3_tts.codec import install_rpu_convolutions

            install_rpu_convolutions(self._wrapper.model.speech_tokenizer.model.decoder,
                                            self._restore, self._codec_convolutions)

    def _generate_talker(self, *args, **kwargs):
        result = self._restore_talker_generate[1](*args, **kwargs)
        ids = result.sequences
        eos = self._wrapper.model.config.talker_config.codec_eos_token_id
        finished = bool(ids.shape[1] and ids[0, -1].item() == eos)
        self._last_generation = {"steps": ids.shape[1], "eos": finished}
        if not finished:
            raise RuntimeError("TTS reached max_new_tokens without EOS; shorten the text or increase the limit (up to 256)")
        return result

    def synthesize(self, text, *, reference_audio, language="Chinese", max_new_tokens=256):
        """Return an independent float32 mono waveform and sample rate (24 kHz).

        Sampling follows the frozen checkpoint. Set torch.manual_seed for repeatability.
        Long text uses 12-token segments, up to 16 tokens to retain short sentence endings.
        max_new_tokens applies to each segment; no partial waveform is returned on failure.
        Reaching the cap without EOS raises before waveform decoding. Reference audio
        is a local mono 24 kHz file; cloning uses speaker embedding without a transcript.
        """
        limit = _validate_request(text, language, max_new_tokens)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("TTS requests, stats and close must be serial")
        try:
            if self._closed:
                raise RuntimeError("TTS model is closed")
            if self._retirement_error is not None:
                raise RuntimeError("TTS retirement failed; restart the process") from self._retirement_error
            segments = _split_text(text, self._wrapper.processor.tokenizer)
            reference = _read_reference(reference_audio)
            self._components["talker"]["generation_limit"] = limit
            self._last_generation = None
            with torch.inference_mode():
                key = hashlib.sha256(reference[0].tobytes()).digest()
                if key != self._reference_key:
                    if reference[1] != self._wrapper.model.speaker_encoder_sample_rate:
                        raise ValueError("TTS frozen speaker encoder requires 24 kHz audio")
                    embedding = self._wrapper.model.extract_speaker_embedding(
                        audio=reference[0], sr=reference[1])
                    # x-vector-only mode discards reference codec tokens entirely.
                    self._voice_clone_prompt = dict(ref_code=[None], ref_spk_embedding=[embedding],
                        x_vector_only_mode=[True], icl_mode=[False])
                    self._reference_key = key
                outputs = []
                steps = []
                segment_rng = torch.random.get_rng_state() if len(segments) > 1 else None
                for index, segment in enumerate(segments):
                    self._last_generation = None
                    utterance = segment
                    if len(segments) > 1 and not segment.endswith(("。", "！", "？", ".", "!", "?")):
                        utterance = segment.rstrip(",，;；:：、") + ("。" if language == "Chinese" else ".")
                    try:
                        # A segment's sampling must not depend on earlier segments' lengths.
                        with torch.random.fork_rng(devices=[]) if index else nullcontext():
                            if index:
                                torch.random.set_rng_state(segment_rng)
                            waves, rate = self._wrapper.generate_voice_clone(text=utterance, language=language,
                                voice_clone_prompt=self._voice_clone_prompt, non_streaming_mode=True,
                                do_sample=True, subtalker_dosample=True, max_new_tokens=limit)
                    except RuntimeError as error:
                        error.add_note(f"TTS segment {index + 1} of {len(segments)} failed; no partial audio returned")
                        raise
                    if self._last_generation is None or not self._last_generation["eos"]:
                        raise RuntimeError("TTS segment did not complete generation")
                    if len(waves) != 1 or rate != 24000:
                        raise RuntimeError("TTS output shape or sample rate differs from the profile")
                    waveform = np.asarray(waves[0], dtype=np.float32)
                    if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
                        raise FloatingPointError("TTS produced empty or non-finite audio")
                    if not np.any(waveform):
                        raise FloatingPointError("TTS produced silent audio")
                    outputs.append(waveform.copy())
                    steps.append(self._last_generation["steps"])
                self._last_generation = {"steps": sum(steps), "eos": True,
                                         "segment_steps": tuple(steps)}
            return (outputs[0] if len(outputs) == 1 else np.concatenate(outputs)), rate
        finally:
            self._lock.release()

    def execution_stats(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("TTS requests, stats and close must be serial")
        try:
            if self._closed:
                raise RuntimeError("TTS model is closed")
            if self._retirement_error is not None:
                raise RuntimeError("TTS retirement failed; restart the process") from self._retirement_error
            result = {"head_calls": list(self._head_calls), "components": {},
                      "generation": dict(self._last_generation) if self._last_generation else None}
            codec_rpu = "codec_transformer" in self._components
            device = self._components["codec_transformer"].get("last_device", "rpu") if codec_rpu else "cpu"
            result["codec_frontend"] = {"device": device,
                "fallback_reason": (None if device == "rpu" else "sequence exceeds 63 device codec frames")
                    if codec_rpu else "native lacks explicit TTS codec mode"}
            result["codec_convolutions"] = dict(self._codec_convolutions)
            for name, part in self._components.items():
                graph = part["graph"]
                result["components"][name] = {"calls": part["calls"], "tokens": part["tokens"],
                    "graph_size": graph.size(), "graph_replays": sum(e.replay_count for e in graph.snapshot()),
                    "graph_invariant_ok": graph.cache_invariant_ok()}
                if name == "predictor":
                    result["components"][name]["fused_head_calls"] = part.get("fused_head_calls", 0)
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


def _take_predictor_logits(part, hidden, index):
    logits = part.pop("head_logits", None)
    head_index = part.pop("head_index", None)
    if hidden is part.get("head_hidden") and index == head_index and hidden.shape[1] == 1:
        if logits is not None:
            part["fused_head_calls"] += 1
        return logits
    return None


def _take_head_input(part, hidden):
    previous = part.pop("head_hidden", None)
    device_hidden = part.pop("head_device", None)
    return device_hidden if hidden is previous and hidden.shape[1] == 1 else None


def _decoder_forward(part, hidden, cache, attention_mask, position_ids, cache_position, output_hidden_states):
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from .cache import RPUCache

    if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != 1024:
        raise ValueError("TTS decoder requires [1, tokens, 1024] inputs")
    if hidden.device.type != "cpu" or hidden.dtype != torch.float32 or not torch.isfinite(hidden).all():
        raise ValueError("TTS decoder requires finite CPU FP32 inputs")
    if attention_mask is not None and not bool(attention_mask.all()):
        raise ValueError("TTS padding is outside this profile")
    if not isinstance(cache, RPUCache):
        if cache is not None and cache.get_seq_length() != 0:
            raise ValueError("cannot replace a populated CPU TTS cache")
        if "cache" not in part:
            part["cache"] = RPUCache(part["layers"], 1, part["capacity"], 8, 128)
        cache = part["cache"]
        cache.reset_to_position(0)
    count = hidden.shape[1]
    if count < 1 or cache.position + count > part["capacity"]:
        raise ValueError("TTS cache capacity exceeded")
    if part["name"] == "talker" and cache.position == 0 and count + part["generation_limit"] > part["capacity"]:
        raise ValueError("TTS prompt plus requested generation exceeds cache capacity")
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
    part.pop("head_logits", None)
    part.pop("head_index", None)
    if part.get("fused_heads", False) and not (
            (cache.position == 0 and count == 2) or
            (2 <= cache.position <= 15 and count == 1)):
        raise ValueError("fused TTS Predictor requires two conditioning tokens then fourteen decode steps")
    chunks = []
    head_only = part.get("device_head_only", False) and not output_hidden_states
    # Only the two known conditioning tokens can run before the first sample.
    prefill = count > 1 and (part["name"] == "talker" or
                            (count == 2 and cache.position == 0))
    sizes = [count] if prefill else [1] * count
    offset = 0
    for size in sizes:
        half_input = hidden[:, offset:offset + size].half().contiguous()
        if not torch.isfinite(half_input).all():
            raise FloatingPointError("TTS input exceeds FP16 range")
        descriptor, signature = part["descriptor"], part["signature"]
        if size > 1:
            from rpu_backend.graph import GraphSignature
            from rpu_backend.runtime.decoder import _text_prefill_execution_plan, prefill_position_key

            execution_len, _, plan = _text_prefill_execution_plan(
                part["planner"], part["handle"], cache, size)
            descriptor = plan.selected.stage_tuple.physical_descriptor
            signature = GraphSignature(op_id="qwen3_tts_prefill_" + part["name"],
                shapes=[1, size, execution_len, 1024],
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
        if size > 1:
            part["prefill_signature"] = signature
        if head_only:
            physical_size = execution_len if size > 1 else size
            if raw.shape != (1, physical_size, 1024) or raw.dtype != torch.float16:
                raise RuntimeError("invalid TTS Predictor device hidden state")
            raw = raw[:, :size]
            cache.update_position(size)
            offset += size
            continue
        value = raw[:, :size].cpu().float()
        if value.shape != (1, size, 1024) or not torch.isfinite(value).all():
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
        part["head_hidden"], part["head_device"] = result, result
        if part.get("fused_heads", False):
            logits = torch.ops.rpu.causal_decoder_tts_predictor_logits(part["handle"])
            if logits.shape != (1, 1, 2048) or logits.dtype != torch.float16:
                raise RuntimeError("invalid fused TTS Predictor logits")
            part["head_logits"] = logits
            part["head_index"] = cache.position - 1
        return BaseModelOutputWithPast(last_hidden_state=result, past_key_values=cache)
    result = torch.cat(chunks, dim=1)
    if count == 1:
        # The head immediately consumes this exact result before another decoder replay.
        part["head_hidden"], part["head_device"] = result, raw
    # The upstream speech wrapper consumes only the final normalized hidden state.
    return BaseModelOutputWithPast(last_hidden_state=result, past_key_values=cache,
        hidden_states=(result,) if output_hidden_states else None)
