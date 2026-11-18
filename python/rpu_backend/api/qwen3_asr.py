"""Frozen Qwen3-ASR 0.6B/1.7B: resident device audio blocks and text decoder, CPU glue.

Audio length and placeholder semantics follow QwenLM/Qwen3-ASR commit
7c6daf77a2421100f5fb066495372c00129d39ff (Apache-2.0).
"""
from __future__ import annotations

import hashlib
import copy
import json
import operator
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ._qwen3_audio_stages import StageTimer
from .errors import RPUUnsupportedDtypeError, UnsupportedModelError


_IDENTITIES = {
    "preprocessor_config.json": "45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d",
    "tokenizer_config.json": "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c",
    "chat_template.json": "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
}
# Official Qwen checkpoint revisions; the index and every shard are immutable.
_PROFILES = {
    "76d3ae4601ce939830b2517f4a6cadb86cc51316c3900af6b020b051c21a478c": {
        "name": "Qwen3-ASR-0.6B", "revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        "index": None,
        "weights": {"model.safetensors": (1876091704,
            "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea")},
    },
    "2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f": {
        "name": "Qwen3-ASR-1.7B", "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5",
        "index": "f994739fe38e5210b9e3e8ce6c6307315e2ceac3cb630e7b7414d69dce520f60",
        "weights": {
            "model-00001-of-00002.safetensors": (4220320824,
                "a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6"),
            "model-00002-of-00002.safetensors": (478200688,
                "6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc"),
        },
    },
}
_DECODER_ROLES = (("self_attn.q_proj", 1), ("self_attn.k_proj", 1),
    ("self_attn.v_proj", 1), ("self_attn.o_proj", 0), ("mlp.gate_proj", 1),
    ("mlp.up_proj", 1), ("mlp.down_proj", 0))
_AUDIO_TOKEN = "<|audio_pad|>"
_AUDIO_TOKEN_ID = 151676
_CACHE_LENGTH = 512
# Request stages reported by execution_stats()["stage_ms"], in pipeline order.
# `decode` accumulates every step after the prefill call, so a long transcript
# is visible as decode rather than as an unexplained total.
_STAGE_NAMES = ("feature_extract", "audio_encoder", "prefill", "decode")


@dataclass(frozen=True)
class ASRTranscription:
    """Independent host output; raw_text retains the model's language metadata."""

    text: str
    raw_text: str
    token_ids: tuple[int, ...]


def _audio_token_length(mel_frames: int) -> int:
    # Upstream chunks every 100 mel frames before three stride-2 convolutions.
    remainder = mel_frames % 100
    return (remainder + 7) // 8 + (mel_frames // 100) * 13


def _validate_audio(audio, sample_rate: int, max_new_tokens: int) -> np.ndarray:
    if isinstance(sample_rate, bool) or sample_rate != 16000:
        raise ValueError("Qwen3-ASR requires mono 16 kHz audio")
    if isinstance(max_new_tokens, bool):
        raise ValueError("max_new_tokens must be an integer in [1, 64]")
    try:
        count = operator.index(max_new_tokens)
    except TypeError as exc:
        raise ValueError("max_new_tokens must be an integer in [1, 64]") from exc
    if not 1 <= count <= 64:
        raise ValueError("max_new_tokens must be an integer in [1, 64]")
    if torch.is_tensor(audio):
        if audio.device.type != "cpu" or not audio.is_floating_point():
            raise ValueError("audio tensor must contain CPU floating-point samples")
        audio = audio.detach().float().numpy()
    waveform = np.asarray(audio)
    if waveform.ndim != 1 or waveform.dtype.kind != "f":
        raise ValueError("audio must be one mono floating-point waveform")
    if not 8000 <= waveform.size <= 128000 or not np.isfinite(waveform).all():
        raise ValueError("audio must be finite and 0.5–8 seconds long")
    waveform = np.asarray(waveform, dtype=np.float32)
    if not np.isfinite(waveform).all():
        raise ValueError("audio must remain finite in float32")
    return waveform


def _checkpoint_profile(checkpoint: Path) -> dict:
    config = checkpoint / "config.json"
    digest = hashlib.sha256(config.read_bytes()).hexdigest() if config.is_file() else None
    if digest not in _PROFILES:
        raise UnsupportedModelError("Qwen3-ASR frozen checkpoint mismatch: config.json")
    return _PROFILES[digest]


def _preflight(checkpoint: Path, dtype) -> dict:
    if dtype is not torch.float16:
        raise RPUUnsupportedDtypeError("Qwen3-ASR activations require torch.float16")
    from importlib.metadata import version

    if version("transformers") not in ("5.3.0", "5.19.0"):
        raise UnsupportedModelError("Qwen3-ASR requires transformers==5.3.0 or ==5.19.0")
    if not checkpoint.is_dir():
        raise UnsupportedModelError("Qwen3-ASR requires a local frozen checkpoint directory")
    profile = _checkpoint_profile(checkpoint)
    identities = dict(_IDENTITIES)
    if profile["index"] is not None:
        identities["model.safetensors.index.json"] = profile["index"]
    for name, expected in identities.items():
        path = checkpoint / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise UnsupportedModelError(f"Qwen3-ASR frozen checkpoint mismatch: {name}")
    for name, (size, _) in profile["weights"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != size:
            raise UnsupportedModelError(f"Qwen3-ASR requires its original weight shard: {name}")
    if set(path.name for path in checkpoint.glob("*.safetensors")) != set(profile["weights"]):
        raise UnsupportedModelError("Qwen3-ASR checkpoint has unexpected weight shards")
    if profile["index"] is None and (checkpoint / "model.safetensors.index.json").exists():
        raise UnsupportedModelError("Qwen3-ASR checkpoint has an unexpected weight index")
    if any((checkpoint / name).exists() for name in ("tokenizer.json", "special_tokens_map.json", "added_tokens.json")):
        raise UnsupportedModelError("Qwen3-ASR frozen profile uses the original tokenizer files")
    if any(not hasattr(torch.ops.rpu, name) for name in (
            "siglip_set_audio_mode", "siglip_set_audio_projection", "assemble_inputs_embeds")):
        raise UnsupportedModelError("Qwen3-ASR resident audio requires the rebuilt native backend")
    return json.loads((checkpoint / "config.json").read_text())["thinker_config"]


def _read_frozen_weights(checkpoint: Path, profile: dict, expected: dict) -> dict:
    from safetensors import safe_open

    weight_map = None
    if profile["index"] is not None:
        weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        if set(weight_map) != set(expected) or set(weight_map.values()) != set(profile["weights"]):
            raise UnsupportedModelError("Qwen3-ASR weight index differs from the frozen profile")
    states = {}
    for filename, (size, digest) in profile["weights"].items():
        path = checkpoint / filename
        with path.open("rb") as stream:
            if path.stat().st_size != size or hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise UnsupportedModelError(f"Qwen3-ASR weight identity differs: {filename}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if (keys - set(expected) or keys & states.keys()
                    or (weight_map is not None and keys != {k for k, v in weight_map.items() if v == filename})):
                raise UnsupportedModelError("Qwen3-ASR weight keys differ from the frozen profile")
            for name in keys:
                view = handle.get_slice(name)
                if tuple(view.get_shape()) != expected[name] or view.get_dtype() != "BF16":
                    raise UnsupportedModelError(f"Qwen3-ASR weight metadata mismatch: {name}")
                value = handle.get_tensor(name).to(torch.float16)
                if not torch.isfinite(value).all():
                    raise UnsupportedModelError(f"Qwen3-ASR weight is not finite in FP16: {name}")
                states[name] = value
    if set(states) != set(expected):
        raise UnsupportedModelError("Qwen3-ASR weight keys differ from the frozen profile")
    return states


def _load_processor(checkpoint: Path):
    from transformers import Qwen2TokenizerFast, WhisperFeatureExtractor

    tokenizer = Qwen2TokenizerFast.from_pretrained(checkpoint, local_files_only=True)
    # Transformers 5.3 passes this argument twice if supplied to from_pretrained.
    tokenizer._patch_mistral_regex(
        tokenizer, str(checkpoint), fix_mistral_regex=True, local_files_only=True, is_local=True,
    )
    template = json.loads((checkpoint / "chat_template.json").read_text())["chat_template"]
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": ""},
         {"role": "user", "content": [{"type": "audio", "audio": ""}]}],
        chat_template=template, add_generation_prompt=True, tokenize=False,
    )
    if prompt.count(_AUDIO_TOKEN) != 1 or tokenizer.convert_tokens_to_ids(_AUDIO_TOKEN) != _AUDIO_TOKEN_ID:
        raise UnsupportedModelError("Qwen3-ASR audio placeholder contract differs")
    return tokenizer, WhisperFeatureExtractor.from_pretrained(checkpoint, local_files_only=True), prompt


def _prepare_inputs(tokenizer, feature_extractor, prompt: str, waveform: np.ndarray) -> dict:
    features = feature_extractor(
        [waveform], sampling_rate=16000, padding=True, truncation=False,
        return_attention_mask=True, return_tensors="pt",
    )
    lengths = features["attention_mask"].sum(-1)
    count = _audio_token_length(int(lengths.item()))
    tokens = tokenizer(
        [prompt.replace(_AUDIO_TOKEN, _AUDIO_TOKEN * count)],
        padding=True, padding_side="left", return_tensors="pt",
    )
    if tokens["input_ids"].shape[0] != 1 or not bool(tokens["attention_mask"].all()):
        raise ValueError("Qwen3-ASR requires one unpadded prompt")
    if int((tokens["input_ids"] == _AUDIO_TOKEN_ID).sum()) != count:
        raise ValueError("Qwen3-ASR audio placeholder count differs from encoder length")
    if tokens["input_ids"].shape[1] + 64 > _CACHE_LENGTH:
        raise ValueError("Qwen3-ASR prompt exceeds the cache envelope")
    return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"],
            "input_features": features["input_features"].to(torch.float16), "feature_lens": lengths}


def _load_components(checkpoint: Path, config: dict):
    from transformers import Qwen3Config, Qwen3Model
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

    audio_config = Qwen3OmniMoeAudioEncoderConfig(**config["audio_config"])
    audio_config._attn_implementation = "eager"
    values = dict(config["text_config"])
    values.pop("rope_scaling")
    values["rope_parameters"] = {"rope_type": "default", "rope_theta": values.pop("rope_theta")}
    text_config = Qwen3Config(**values)
    text_config._attn_implementation = "eager"
    with torch.device("meta"):
        audio = Qwen3OmniMoeAudioEncoder(audio_config)
        text = Qwen3Model(text_config)
    expected = {"thinker.audio_tower." + k: tuple(v.shape) for k, v in audio.state_dict().items()}
    expected.update({"thinker.model." + k: tuple(v.shape) for k, v in text.state_dict().items()})
    expected["thinker.lm_head.weight"] = (text_config.vocab_size, text_config.hidden_size)
    states = _read_frozen_weights(checkpoint, _checkpoint_profile(checkpoint), expected)
    # Construct CPU buffers normally: meta .to_empty would lose rotary/sinusoid values.
    audio = Qwen3OmniMoeAudioEncoder(audio_config).half().eval()
    text = Qwen3Model(text_config).eval()
    # from_pretrained(dtype=FP16) preserves the explicitly FP32 RoPE frequencies.
    # A blanket .half() would round these before cos/sin generation.
    inv_freq = text.rotary_emb.inv_freq.clone()
    text.half()
    text.rotary_emb.inv_freq = inv_freq
    text.rotary_emb.original_inv_freq = inv_freq.clone()
    audio.load_state_dict({k.removeprefix("thinker.audio_tower."): v for k, v in states.items()
                           if k.startswith("thinker.audio_tower.")}, strict=True)
    text.load_state_dict({k.removeprefix("thinker.model."): v for k, v in states.items()
                          if k.startswith("thinker.model.")}, strict=True)
    # The official files store both tensors. Never replace the original head
    # with the embedding, even when the config declares tied embeddings.
    return audio, text, states["thinker.lm_head.weight"]


class Qwen3ASRModel:
    """One resident device ASR model; requests are serial and outputs live on CPU.

    Use ``with Qwen3ASRModel.from_pretrained(local_checkpoint) as model:``.
    Frozen Qwen3-ASR 0.6B/1.7B use FP16 activations and FP16/W8/W4/NVFP4 decoder
    weights; NVFP4 keeps the output head in FP16. Audio must be mono float samples at 16 kHz, 0.5–8 seconds. No timestamps or streaming.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._closed = True
        self._audio = self._text = self._embedding_weight = None
        self._head_weight = self._head_scale = self._head_cpu_weight = None
        self._language_model = None
        self._audio_linear_calls = {}
        self._audio_encoder = None
        self._decoder_calls = 0
        self._restore_forwards = []
        self._graph_prompt_length = None
        self._cache = None
        self._stages = StageTimer("ASR", _STAGE_NAMES)
        self._stage_ms = None
        self._retirement_error = None
        self._precision, self._head_precision, self._linear_acc32 = "fp16", "fp16", True
        self._nvfp4_tensor_scales = None
        self._hidden_size = 1024
        self._num_layers, self._num_kv_heads, self._head_dim = 28, 8, 128
        self._num_cores = 8
        self._topology = None

    @classmethod
    def from_pretrained(cls, checkpoint, *, dtype=torch.float16, precision="fp16",
                        rpu_execution=None) -> "Qwen3ASRModel":
        from .causal_lm import _claim_live_instance
        from ._qwen3_audio_precision import normalize_audio_precision
        from rpu_backend.runtime.topology import execution_core_count

        precision, acc32 = normalize_audio_precision(precision, rpu_execution, True)

        path = Path(checkpoint)
        config = _preflight(path, dtype)
        model = cls()
        model._precision, model._linear_acc32 = precision, acc32
        model._num_cores = execution_core_count(rpu_execution)
        model._head_precision = "fp16" if precision == "nvfp4" else precision
        _claim_live_instance(model)
        model._closed = False
        try:
            model._tokenizer, model._feature_extractor, model._prompt = _load_processor(path)
            model._audio, model._text, model._head_cpu_weight = _load_components(path, config)
            torch.rpu.set_caching_allocator(True)
            model._install_rpu()
        except BaseException as error:
            try:
                model.close()
            except BaseException as cleanup_error:
                error.add_note(f"ASR cleanup failed: {cleanup_error!r}")
            raise
        return model

    def _install_rpu(self):
        from rpu_backend.adapters.qwen3 import lookup_causal_decoder, _apply_fused_lm_head_for_rpu
        from rpu_backend.runtime.decoder import _install_causal_decoder_forward
        from rpu_backend.runtime.topology import resolve_decoder_topology
        from rpu_backend.runtime.weights import transform_linear_weight
        from ._qwen3_audio_precision import (pack_audio_linear, pack_audio_nvfp4,
                                            pack_audio_tensor_scales)

        config = self._text.config
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_hidden_layers
        self._num_kv_heads, self._head_dim = config.num_key_value_heads, config.head_dim
        topology = resolve_decoder_topology(num_cores=self._num_cores, hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size, num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads, head_dim=config.head_dim, vocab_size=config.vocab_size)
        self._topology = topology
        with torch.inference_mode():
            cos, sin = self._text.rotary_emb(torch.zeros(1, 1, config.hidden_size, dtype=torch.float16),
                                            torch.arange(_CACHE_LENGTH).view(1, -1))
        tables = (cos[0, :, :config.head_dim // 2].contiguous(),
                  sin[0, :, :config.head_dim // 2].contiguous())
        self._embedding_weight = self._text.embed_tokens.weight.detach()
        # Validate and pack on CPU before creating a native decoder. Quantized
        # Linears retain their logical in/out dimensions and actual scale buffer.
        tensor_scales = [[] for _ in _DECODER_ROLES] if self._precision == "nvfp4" else None
        for layer in self._text.layers:
            for index, (role, partition) in enumerate(_DECODER_ROLES):
                linear = layer.get_submodule(role)
                cores = topology.mlp_tp if role.startswith("mlp.") else topology.attn_tp
                if tensor_scales is not None:
                    weight, scale, scalar = pack_audio_nvfp4(linear.weight, partition, num_cores=cores)
                    tensor_scales[index].append(scalar)
                else:
                    weight, scale = pack_audio_linear(linear.weight, partition, self._precision, num_cores=cores)
                linear.weight = torch.nn.Parameter(weight, requires_grad=False)
                if scale is not None:
                    linear.register_buffer("weight_scale", scale)
                linear._rpu_linear_num_cores, linear._rpu_linear_partition = cores, partition
        head_weight, head_scale = pack_audio_linear(
            self._head_cpu_weight, 1, self._head_precision, bounded_head=True,
            num_cores=topology.lm_head_tp)
        self._head_cpu_weight = None
        self._text.to("rpu")
        if self._embedding_weight.device.type != "cpu" or self._embedding_weight.dtype != torch.float16:
            raise RuntimeError("ASR host embedding storage moved during device preparation")
        scale_lists = None if self._precision == "fp16" else tuple(
            [layer.get_submodule(role).weight_scale for layer in self._text.layers]
            for role, _ in _DECODER_ROLES)
        if tensor_scales is not None:
            self._nvfp4_tensor_scales = tuple(
                pack_audio_tensor_scales(values).to("rpu") for values in tensor_scales)
        execution = {"model": {"num_cores": self._num_cores}, "prefill": {"linear_acc32": self._linear_acc32}}
        self._text._rpu_execution = execution
        _install_causal_decoder_forward(self._text, arch="qwen3", topology=topology,
            cos_sin=tables, chunk_envelope_for=lookup_causal_decoder, scale_lists=scale_lists,
            execution_config=execution, _audio_component="asr", _graph_cache_max_entries=2,
            nvfp4_tensor_scales=self._nvfp4_tensor_scales)
        self._head_weight = head_weight.to("rpu")
        self._head_scale = head_scale.to("rpu") if head_scale is not None else None
        # Reuse the fused decode head and separate last-token prefill projection.
        self._language_model = torch.nn.Module()
        self._language_model.model = self._text
        self._language_model.config = config
        self._language_model._rpu_execution = self._text._rpu_execution
        with torch.device("meta"):
            head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        head.weight = torch.nn.Parameter(self._head_weight, requires_grad=False)
        if self._head_scale is not None:
            head.register_buffer("weight_scale", self._head_scale)
        head._rpu_linear_num_cores, head._rpu_linear_partition = topology.lm_head_tp, 1
        head.forward = self._project_logits
        self._language_model.lm_head = head
        _apply_fused_lm_head_for_rpu(self._language_model, _audio_precision=self._head_precision)
        counters = self._audio_linear_calls
        audio_cores = 8 if self._num_cores == 8 else 4
        for name, module in self._audio.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self._restore_forwards.append((module, module.forward))
                weight = module.weight.detach().float().contiguous(memory_format=torch.channels_last)
                bias = module.bias.detach().float() if module.bias is not None else None

                def convolution(owner, hidden, weight=weight, bias=bias):
                    # Use CPU FP32 convolution with FP16 input/output boundaries.
                    hidden = hidden.float().contiguous(memory_format=torch.channels_last)
                    return torch.nn.functional.conv2d(hidden, weight, bias,
                        owner.stride, owner.padding, owner.dilation, owner.groups).half()

                module.forward = types.MethodType(convolution, module)
                continue
            if not isinstance(module, torch.nn.Linear):
                continue
            counters[name] = 0
            if name.startswith("layers.") or name in ("proj1", "proj2"):
                continue
            if module.in_features % 16 or module.out_features % 128:
                raise UnsupportedModelError(f"unaligned ASR audio Linear: {name}")
            weight = transform_linear_weight(module.weight.detach().contiguous(), 1, audio_cores).to("rpu")
            bias = module.bias.detach().contiguous().to("rpu") if module.bias is not None else None
            counters[name] = 0
            self._restore_forwards.append((module, module.forward))

            def forward(owner, hidden, weight=weight, bias=bias, name=name):
                if hidden.device.type not in ("cpu", "rpu") or hidden.dtype != torch.float16:
                    raise ValueError("ASR audio Linear requires CPU or device FP16 input")
                result = torch.ops.rpu.linear_with_core_count(hidden.contiguous().to("rpu"), weight, bias, 1, audio_cores)
                counters[name] += 1
                return result.cpu() if hidden.device.type == "cpu" else result

            module.forward = types.MethodType(forward, module)
        if len(counters) != len(self._audio.layers) * 6 + 3:
            raise UnsupportedModelError("ASR audio projection count differs")
        from ._qwen3_asr_audio import ResidentAudioEncoder

        self._audio_encoder = ResidentAudioEncoder(self._audio, counters, num_cores=audio_cores)
        self._audio_encoder.install(self)
        self._restore_forwards.append((self._audio, self._audio.forward))
        self._audio.forward = self._audio_encoder.forward

    def transcribe(self, audio, *, sample_rate=16000, max_new_tokens=64) -> ASRTranscription:
        """Transcribe one waveform; accepts CPU float Tensor or numpy samples."""
        waveform = _validate_audio(audio, sample_rate, max_new_tokens)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Qwen3-ASR requests and close must be serial")
        try:
            if self._closed:
                raise RuntimeError("Qwen3-ASR model is closed")
            if self._retirement_error is not None:
                raise RuntimeError("ASR retirement failed; restart the process") from self._retirement_error
            # A failed request leaves no stale breakdown behind for execution_stats().
            self._stages.reset()
            self._stage_ms = None
            request_start = time.perf_counter_ns()
            with torch.inference_mode():
                with self._stages.stage("feature_extract"):
                    inputs = _prepare_inputs(self._tokenizer, self._feature_extractor, self._prompt, waveform)
                prompt_length = inputs["input_ids"].shape[1]
                if self._graph_prompt_length is not None and prompt_length != self._graph_prompt_length:
                    # Keep only the current shape's prefill/decode pair, leaving
                    # launch arena capacity for the next audio eager Linear.
                    self._text._rpu_decoder_graph_cache.clear()
                self._graph_prompt_length = prompt_length
                transcription = self._generate(inputs, max_new_tokens)
            self._stage_ms = self._stages.milliseconds(time.perf_counter_ns() - request_start)
            return transcription
        finally:
            self._lock.release()

    def _generate(self, inputs, max_new_tokens):
        from .cache import RPUCache

        if self._cache is None:
            self._cache = RPUCache(self._num_layers, 1, _CACHE_LENGTH, self._num_kv_heads,
                                   self._head_dim, dtype=torch.float16, attn_tp=self._topology.attn_tp)
        cache = self._cache
        cache.reset_to_position(0)
        length = inputs["feature_lens"]
        with self._stages.stage("audio_encoder"):
            features = self._audio(inputs["input_features"][0, :, :int(length.item())],
                                   feature_lens=length).last_hidden_state
        ids = inputs["input_ids"]
        mask = ids == _AUDIO_TOKEN_ID
        if features.shape != (int(mask.sum()), self._hidden_size) or not torch.isfinite(features.cpu()).all():
            raise ValueError("ASR audio features violate the placeholder/finite contract")
        hidden = self._embed_tokens(ids, features)
        attention_mask = inputs["attention_mask"]
        generated = []
        self._last_request_execution = {"prefill_tokens": int(ids.shape[1]),
            "input_ids_sha256": hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest(),
            "audio_tokens": int(features.shape[0]), "decode_calls": 0,
            "torch_num_threads": torch.get_num_threads(),
            "inference_mode": torch.is_inference_mode_enabled(), "execution_plans": {}}
        for _ in range(max_new_tokens):
            start = cache.get_seq_length()
            positions = torch.arange(start, start + hidden.shape[1])
            stage = "prefill" if start == 0 else "decode"
            with self._stages.stage(stage):
                output = self._language_model(inputs_embeds=hidden.contiguous().to("rpu"),
                    position_ids=positions.unsqueeze(0), cache_position=positions,
                    attention_mask=attention_mask, past_key_values=cache, use_cache=True,
                    logits_to_keep=1, return_dict=True)
            self._decoder_calls += 1
            self._last_request_execution["execution_plans"][stage] = getattr(
                self._text, "_rpu_last_execution_plan", None)
            if start:
                self._last_request_execution["decode_calls"] += 1
            logits = output.logits.cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite ASR logits")
            token = logits[:, -1].argmax(-1, keepdim=True)
            generated.append(int(token.item()))
            if generated[-1] in (151645, 151643):
                break
            hidden = self._embed_tokens(token)
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)), dim=1)
        raw = self._tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # Preserve raw output for diagnostics; expose the text after ASR metadata.
        text = raw.split("<asr_text>", 1)[-1].strip()
        return ASRTranscription(text=text, raw_text=raw, token_ids=tuple(generated))

    def _embed_tokens(self, ids, features=None):
        weight = self._text.embed_tokens.weight
        if (ids.device.type != "cpu" or ids.dtype != torch.long or ids.ndim != 2 or
                ids.shape[0] != 1 or ids.numel() == 0 or
                bool((ids < 0).any()) or bool((ids >= weight.shape[0]).any())):
            raise ValueError("ASR embedding requires one row of valid CPU token IDs")
        # Use CPU embedding lookup for single-token requests.
        if features is None:
            return torch.nn.functional.embedding(ids, self._embedding_weight).to("rpu")
        positions = (ids[0] == _AUDIO_TOKEN_ID).nonzero().flatten()
        if (positions.numel() == 0 or features.shape != (positions.numel(), self._hidden_size) or
                not torch.equal(positions, torch.arange(int(positions[0]), int(positions[0]) + positions.numel()))):
            raise ValueError("ASR requires one contiguous audio placeholder block")
        device_ids = ids.flatten().to(torch.int32).to("rpu")
        hidden = torch.ops.rpu.assemble_inputs_embeds(
            weight, device_ids, features.contiguous().to("rpu"), int(positions[0]))
        return hidden.unsqueeze(0)

    def _project_logits(self, hidden):
        logical_shape = ({"logical_out_features": self._text.config.vocab_size}
                         if self._head_weight.dtype == torch.uint8 else {})
        return torch.ops.rpu.linear_with_accumulation(
            hidden.contiguous(), self._head_weight, None, 1, self._topology.lm_head_tp,
            self._linear_acc32, self._head_scale, **logical_shape)

    def execution_stats(self) -> dict:
        """Return execution-route counters."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Qwen3-ASR requests, statistics and close must be serial")
        try:
            if self._closed:
                raise RuntimeError("Qwen3-ASR model is closed")
            if self._retirement_error is not None:
                raise RuntimeError("ASR retirement failed; restart the process") from self._retirement_error
            graph = self._text._rpu_decoder_graph_cache
            audio_graph = self._audio_encoder.graph
            return {"precision": self._precision, "decoder_precision": self._precision,
                    "head_precision": self._head_precision, "linear_acc32": self._linear_acc32,
                    "hidden_size": self._hidden_size, "num_cores": self._num_cores,
                    "decoder_topology": self._topology.identity(),
                    "audio_num_cores": self._audio_encoder.num_cores,
                    "native_decoder_topology": list(torch.ops.rpu.causal_decoder_topology(
                        self._text._rpu_decoder_handle)),
                    "native_audio_topology": list(torch.ops.rpu.siglip_get_execution_topology(
                        self._audio_encoder.handle)),
                    "request_execution": copy.deepcopy(getattr(self, "_last_request_execution", None)),
                    "stage_ms": dict(self._stage_ms) if self._stage_ms else None,
                    "audio_input_geometry": copy.deepcopy(getattr(self._audio_encoder, "last_input_geometry", None)),
                    "audio_execution_plan": (self._audio_encoder.last_execution_plan.as_dict(include_candidates=False)
                        if getattr(self._audio_encoder, "last_execution_plan", None) is not None else None),
                    "audio_linear_calls": dict(self._audio_linear_calls), "decoder_calls": self._decoder_calls,
                    "audio_resident_calls": self._audio_encoder.calls,
                    "audio_graph_size": audio_graph.size(),
                    "audio_graph_replays": sum(e.replay_count for e in audio_graph.snapshot()),
                    "audio_graph_invariant_ok": audio_graph.cache_invariant_ok(),
                    "graph_size": graph.size(), "graph_replays": sum(e.replay_count for e in graph.snapshot()),
                    "graph_invariant_ok": graph.cache_invariant_ok()}
        finally:
            self._lock.release()

    def close(self):
        """Retire the decoder before releasing device weights and process ownership."""
        from .causal_lm import _release_live_instance

        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Qwen3-ASR requests and close must be serial")
        try:
            if self._closed:
                return
            if self._retirement_error is not None:
                raise RuntimeError("ASR retirement failed; restart the process") from self._retirement_error
            resource = getattr(self._text, "_rpu_decoder_retirement_state", None)
            audio_resource = getattr(self._audio_encoder, "_resource", None)
            try:
                if resource is not None:
                    resource.retire()
                if self._audio_encoder is not None:
                    self._audio_encoder.close()
            except BaseException as error:
                self._retirement_error = error
                for installed in (resource, audio_resource):
                    if installed is not None:
                        installed.retain_failure(error, self)
                raise
            self._audio_encoder = None
            for module, forward in reversed(self._restore_forwards):
                module.forward = forward
            self._restore_forwards.clear()
            if self._text is not None:
                vars(self._text).pop("forward", None)
            self._audio = self._text = self._embedding_weight = None
            self._head_weight = self._head_scale = self._head_cpu_weight = None
            self._nvfp4_tensor_scales = None
            self._language_model = None
            self._cache = None
            self._closed = True
            _release_live_instance(self)
        finally:
            self._lock.release()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Qwen3-ASR model is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
