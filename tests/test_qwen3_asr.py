"""Host-only admission, preprocessing and ownership checks for the ASR profile."""
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest
import torch

from rpu_backend.api import Qwen3ASRModel
from rpu_backend.api import qwen3_asr as asr
from rpu_backend.api.errors import RPUSingleHandleError, RPUUnsupportedDtypeError, UnsupportedModelError


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, "fp16"])
def test_dtype_rejected_before_checkpoint_access(tmp_path, dtype):
    with pytest.raises(RPUUnsupportedDtypeError):
        Qwen3ASRModel.from_pretrained(tmp_path / "missing", dtype=dtype)


def test_unreviewed_config_rejected_before_loading(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_asr"}')
    monkeypatch.setattr(asr, "_load_components", lambda *a: pytest.fail("weights loaded"))
    with pytest.raises(UnsupportedModelError, match="config.json"):
        Qwen3ASRModel.from_pretrained(tmp_path)


@pytest.mark.parametrize("version", ["5.3.0", "5.19.0"])
def test_reviewed_transformers_versions_reach_checkpoint_checks(tmp_path, monkeypatch, version):
    monkeypatch.setattr("importlib.metadata.version", lambda _: version)
    with pytest.raises(UnsupportedModelError, match="config.json"):
        asr._preflight(tmp_path, torch.float16)


@pytest.mark.parametrize("version", ["4.57.3", "5.18.0", "5.19.1"])
def test_unreviewed_transformers_rejected_before_checkpoint(tmp_path, monkeypatch, version):
    monkeypatch.setattr("importlib.metadata.version", lambda _: version)
    with pytest.raises(UnsupportedModelError, match="transformers==5.3.0 or ==5.19.0"):
        asr._preflight(tmp_path / "missing", torch.float16)


def _empty_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"thinker_config":{}}')
    (tmp_path / "model.safetensors").touch()
    digest = hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    monkeypatch.setattr(asr, "_IDENTITIES", {})
    monkeypatch.setattr(asr, "_PROFILES", {digest: {
        "index": None, "weights": {"model.safetensors": (0, hashlib.sha256(b"").hexdigest())}}})


@pytest.mark.parametrize("extra", ["tokenizer.json", "added_tokens.json", "special_tokens_map.json"])
def test_tokenizer_overrides_rejected_before_loading(tmp_path, monkeypatch, extra):
    _empty_checkpoint(tmp_path, monkeypatch)
    (tmp_path / extra).write_text("{}")
    with pytest.raises(UnsupportedModelError, match="original tokenizer files"):
        asr._preflight(tmp_path, torch.float16)


def test_old_native_rejected_before_loading(tmp_path, monkeypatch):
    _empty_checkpoint(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.ops, "rpu", SimpleNamespace())
    with pytest.raises(UnsupportedModelError, match="rebuilt native backend"):
        asr._preflight(tmp_path, torch.float16)


@pytest.mark.parametrize("extra", ["unreviewed.safetensors", "model.safetensors.index.json"])
def test_extra_weight_inputs_rejected(tmp_path, monkeypatch, extra):
    _empty_checkpoint(tmp_path, monkeypatch)
    (tmp_path / extra).write_text("{}")
    with pytest.raises(UnsupportedModelError, match="unexpected weight"):
        asr._preflight(tmp_path, torch.float16)


@pytest.mark.parametrize("precision,execution", [
    ("w4a16", None), ("bf16", None), ("fp16", {"model": {"num_cores": 5}}),
    ("w8a16", {"model": {"num_cores": True}}), ("w4a16_g32", {"prefill": {"linear_acc32": 1}}),
])
def test_invalid_precision_rejects_before_checkpoint(tmp_path, monkeypatch, precision, execution):
    monkeypatch.setattr(asr, "_preflight", lambda *args: pytest.fail("checkpoint accessed"))
    with pytest.raises(ValueError):
        Qwen3ASRModel.from_pretrained(tmp_path, precision=precision, rpu_execution=execution)


def _write_weight_shards(tmp_path, shards):
    from safetensors.torch import save_file

    profile = {"index": "test-index", "weights": {}}
    weight_map = {}
    for filename, tensors in shards.items():
        path = tmp_path / filename
        save_file(tensors, path)
        profile["weights"][filename] = (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        weight_map.update({name: filename for name in tensors})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return profile


def test_original_embedding_and_head_remain_distinct_across_shards(tmp_path):
    embedding = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
    head = embedding + 8
    profile = _write_weight_shards(tmp_path, {
        "first.safetensors": {"thinker.model.embed_tokens.weight": embedding},
        "second.safetensors": {"thinker.lm_head.weight": head},
    })
    states = asr._read_frozen_weights(tmp_path, profile, {
        "thinker.model.embed_tokens.weight": (4, 2), "thinker.lm_head.weight": (4, 2)})
    torch.testing.assert_close(states["thinker.model.embed_tokens.weight"], embedding.half())
    torch.testing.assert_close(states["thinker.lm_head.weight"], head.half())
    assert states["thinker.model.embed_tokens.weight"].data_ptr() != states["thinker.lm_head.weight"].data_ptr()


def test_component_loader_preserves_original_head_and_float32_rope(tmp_path, monkeypatch):
    import transformers
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as audio_module
    from transformers.models.qwen3 import modeling_qwen3 as text_module

    class Audio(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2, bias=False)

    class Text(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
            self.rotary_emb = torch.nn.Module()
            self.rotary_emb.register_buffer("inv_freq", torch.tensor([1 / 8193]), persistent=False)

    monkeypatch.setattr(text_module, "Qwen3Model", Text)
    monkeypatch.setattr(transformers, "Qwen3Model", Text)
    monkeypatch.setattr(audio_module, "Qwen3OmniMoeAudioEncoder", Audio)
    embedding = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
    head = embedding + 8
    profile = _write_weight_shards(tmp_path, {
        "first.safetensors": {"thinker.model.embed_tokens.weight": embedding,
                              "thinker.audio_tower.proj.weight": torch.ones(2, 2, dtype=torch.bfloat16)},
        "second.safetensors": {"thinker.lm_head.weight": head},
    })
    monkeypatch.setattr(asr, "_checkpoint_profile", lambda path: profile)
    _, text, loaded_head = asr._load_components(tmp_path, {"audio_config": {}, "text_config": {
        "hidden_size": 2, "vocab_size": 4, "rope_scaling": {}, "rope_theta": 1000000,
        "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 2}})
    torch.testing.assert_close(text.embed_tokens.weight, embedding.half())
    torch.testing.assert_close(loaded_head, head.half())
    assert loaded_head.data_ptr() != text.embed_tokens.weight.data_ptr()
    assert text.rotary_emb.inv_freq.dtype == torch.float32
    torch.testing.assert_close(text.rotary_emb.inv_freq, torch.tensor([1 / 8193]), rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["missing", "wrong-shard", "duplicate", "shape", "dtype", "nonfinite", "hash"])
def test_frozen_shards_reject_corrupt_payloads(tmp_path, failure):
    first = torch.ones(4, 2, dtype=torch.bfloat16)
    second = torch.zeros_like(first)
    if failure == "shape":
        second = second.reshape(2, 4)
    elif failure == "dtype":
        second = second.half()
    elif failure == "nonfinite":
        second[0, 0] = float("inf")
    profile = _write_weight_shards(tmp_path, {
        "first.safetensors": {"embedding": first},
        "second.safetensors": {"head": second, **({"embedding": first} if failure == "duplicate" else {})},
    })
    expected = {"embedding": (4, 2), "head": (4, 2)}
    if failure == "missing":
        expected["layer"] = (2, 2)
    elif failure == "wrong-shard":
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"embedding": "second.safetensors", "head": "first.safetensors"}}))
    elif failure == "hash":
        size, digest = profile["weights"]["second.safetensors"]
        profile["weights"]["second.safetensors"] = (size, "0" * 64)
    with pytest.raises(UnsupportedModelError):
        asr._read_frozen_weights(tmp_path, profile, expected)


@pytest.mark.parametrize("audio,rate,tokens", [
    (np.zeros(7999, dtype=np.float32), 16000, 64),
    (np.zeros(128001, dtype=np.float32), 16000, 64),
    (np.zeros((1, 16000), dtype=np.float32), 16000, 64),
    (np.zeros(16000, dtype=np.int16), 16000, 64),
    (np.full(16000, np.nan), 16000, 64),
    (np.full(16000, np.inf), 16000, 64),
    (np.zeros(16000, dtype=np.float32), 24000, 64),
    (np.zeros(16000, dtype=np.float32), 16000, 65),
    (np.zeros(16000, dtype=np.float32), 16000, True),
    (np.zeros(16000, dtype=np.float32), 16000, 1.5),
])
def test_invalid_audio_rejected_without_inference(audio, rate, tokens):
    with pytest.raises(ValueError):
        asr._validate_audio(audio, rate, tokens)


@pytest.mark.parametrize("samples", [8000, 67263, 128000])
def test_valid_audio_boundaries(samples):
    value = asr._validate_audio(torch.zeros(samples), 16000, 64)
    assert value.shape == (samples,)
    assert value.dtype == np.float32


@pytest.mark.parametrize("frames,expected", [(50, 7), (99, 13), (100, 13), (101, 14), (420, 55), (800, 104)])
def test_audio_tokens_respect_chunk_boundary(frames, expected):
    assert asr._audio_token_length(frames) == expected


def test_preprocessor_checks_feature_placeholder_agreement():
    def extractor(*args, **kwargs):
        return {"attention_mask": torch.ones(1, 420, dtype=torch.long),
                "input_features": torch.zeros(1, 128, 420)}

    def tokenizer(prompts, **kwargs):
        assert prompts == ["prefix" + "<|audio_pad|>" * 55]
        return {"input_ids": torch.tensor([[151676] * 54]), "attention_mask": torch.ones(1, 54)}

    with pytest.raises(ValueError, match="placeholder count"):
        asr._prepare_inputs(tokenizer, extractor, "prefix<|audio_pad|>", np.zeros(67263, dtype=np.float32))


def test_failed_load_releases_owner_and_does_not_install(tmp_path, monkeypatch):
    from rpu_backend.api import causal_lm

    events = []
    monkeypatch.setattr(asr, "_preflight", lambda *args: {})
    monkeypatch.setattr(asr, "_load_processor", lambda *args: (None, None, ""))
    monkeypatch.setattr(causal_lm, "_claim_live_instance", lambda owner: events.append("claim"))
    monkeypatch.setattr(causal_lm, "_release_live_instance", lambda owner: events.append("release"))
    monkeypatch.setattr(Qwen3ASRModel, "_install_rpu", lambda self: pytest.fail("device accessed"))

    def fail(*args):
        events.append("load")
        raise ValueError("invalid weights")

    monkeypatch.setattr(asr, "_load_components", fail)
    with pytest.raises(ValueError, match="invalid weights"):
        Qwen3ASRModel.from_pretrained(tmp_path)
    assert events == ["claim", "load", "release"]


def test_occupied_owner_rejects_before_processor_or_weights(tmp_path, monkeypatch):
    from rpu_backend.api import causal_lm

    monkeypatch.setattr(asr, "_preflight", lambda *args: {})
    monkeypatch.setattr(asr, "_load_processor", lambda *args: pytest.fail("processor loaded"))
    monkeypatch.setattr(asr, "_load_components", lambda *args: pytest.fail("weights loaded"))

    def occupied(owner):
        raise RPUSingleHandleError("occupied")

    monkeypatch.setattr(causal_lm, "_claim_live_instance", occupied)
    with pytest.raises(RPUSingleHandleError, match="occupied"):
        Qwen3ASRModel.from_pretrained(tmp_path)


def test_close_retires_before_releasing_and_is_idempotent(monkeypatch):
    from rpu_backend.api import causal_lm

    events = []
    monkeypatch.setattr(causal_lm, "_release_live_instance", lambda owner: events.append("release"))
    model = Qwen3ASRModel()
    model._closed = False
    model._text = SimpleNamespace(_rpu_decoder_retirement_state=SimpleNamespace(retire=lambda: events.append("retire")))
    model._head_weight = object()
    model._language_model = object()
    model.close()
    model.close()
    assert events == ["retire", "release"]
    assert model._text is None
    assert model._head_weight is None
    assert model._language_model is None
    with pytest.raises(RuntimeError, match="closed"):
        model.transcribe(np.zeros(8000, dtype=np.float32))
    with pytest.raises(RuntimeError, match="closed"):
        model.execution_stats()


def test_close_cannot_release_an_active_request():
    model = Qwen3ASRModel()
    model._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="serial"):
            model.close()
        with pytest.raises(RuntimeError, match="serial"):
            model.execution_stats()
    finally:
        model._lock.release()


def test_failed_retirement_does_not_release_owner(monkeypatch):
    from rpu_backend.api import causal_lm

    events = []
    monkeypatch.setattr(causal_lm, "_release_live_instance", lambda owner: events.append("release"))
    model = Qwen3ASRModel()
    model._closed = False

    def fail():
        raise RuntimeError("retirement failed")

    model._text = SimpleNamespace(_rpu_decoder_retirement_state=SimpleNamespace(
        retire=fail, retain_failure=lambda error, parent: events.append("retain")))
    head = model._head_weight = object()
    with pytest.raises(RuntimeError, match="retirement failed"):
        model.close()
    assert events == ["retain"] and not model._closed and model._text is not None
    assert model._head_weight is head
    # No real native resource exists in this host test.
    model._closed = True


def test_decoder_install_bounds_graphs_to_leave_audio_launch_capacity(monkeypatch):
    from rpu_backend.runtime import decoder, topology, weights

    captured = {}
    model = Qwen3ASRModel()
    table = torch.zeros(1, 512, 128, dtype=torch.float16)
    model._text = SimpleNamespace(
        config=SimpleNamespace(hidden_size=1024, intermediate_size=3072, num_attention_heads=16,
            num_hidden_layers=28, num_key_value_heads=8, head_dim=128, vocab_size=151936),
        rotary_emb=lambda *args: (table, table), layers=[],
        embed_tokens=SimpleNamespace(weight=torch.zeros(1, 1024, dtype=torch.float16)),
        to=lambda device: None,
    )
    model._head_cpu_weight = torch.zeros(1, 1024, dtype=torch.float16)
    from rpu_backend.api import _qwen3_audio_precision as precision
    monkeypatch.setattr(precision, "pack_audio_linear", lambda weight, *args, **kwargs: (weight, None))
    monkeypatch.setattr(topology, "resolve_decoder_topology", lambda **kwargs:
        SimpleNamespace(num_cores=8, attn_tp=8, mlp_tp=8, lm_head_tp=8))

    def install(model, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before audio installation")

    monkeypatch.setattr(decoder, "_install_causal_decoder_forward", install)
    with pytest.raises(RuntimeError, match="stop before audio"):
        model._install_rpu()
    assert captured["_graph_cache_max_entries"] == 2


@pytest.mark.parametrize("precision,dtype", [("fp16", torch.float16), ("w8a16", torch.int8),
                                           ("w4a16_g32", torch.uint8), ("nvfp4", torch.uint8)])
@pytest.mark.parametrize("acc32", [False, True])
def test_decoder_packing_keeps_embedding_and_binds_actual_scales(monkeypatch, precision, dtype, acc32):
    from rpu_backend.adapters import qwen3
    from rpu_backend.api import _qwen3_audio_precision as packing
    from rpu_backend.runtime import decoder, topology

    text = torch.nn.Module()
    text.config = SimpleNamespace(hidden_size=1024, intermediate_size=3072, num_attention_heads=16,
        num_hidden_layers=28, num_key_value_heads=8, head_dim=128, vocab_size=151936)
    text.embed_tokens = torch.nn.Embedding(16, 512, dtype=torch.float16)
    layer = torch.nn.Module()
    layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
    for group, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                         (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
        for name in names:
            # Small real matrices satisfy both column and row packing geometry.
            setattr(group, name, torch.nn.Linear(512, 128, bias=False, dtype=torch.float16))
    if precision == "nvfp4":
        from copy import deepcopy
        text.layers = torch.nn.ModuleList([layer, *[deepcopy(layer) for _ in range(27)]])
    else:
        text.layers = torch.nn.ModuleList([layer])
    tables = torch.zeros(1, 512, 128, dtype=torch.float16)
    text.rotary_emb = lambda *args: (tables, tables)
    text.to = lambda device: text
    embedding = text.embed_tokens.weight.detach().clone()
    original_head = torch.full((128, 512), 0.25, dtype=torch.float16)
    model = Qwen3ASRModel()
    model._text, model._head_cpu_weight = text, original_head
    model._precision, model._linear_acc32 = precision, acc32
    model._head_precision = "fp16" if precision == "nvfp4" else precision
    head_dtype = torch.float16 if precision == "nvfp4" else dtype
    packed_head = []
    real_pack = packing.pack_audio_linear

    def pack(weight, partition, precision, *, bounded_head=False, num_cores=8):
        if bounded_head:
            assert weight is original_head
        result = real_pack(weight, partition, precision, bounded_head=bounded_head, num_cores=num_cores)
        if bounded_head:
            packed_head.append(result)
        return result

    monkeypatch.setattr(packing, "pack_audio_linear", pack)
    monkeypatch.setattr(topology, "resolve_decoder_topology", lambda **kwargs:
        SimpleNamespace(num_cores=8, attn_tp=8, mlp_tp=8, lm_head_tp=8))
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *args, **kwargs:
        tensor if args == ("rpu",) else original_to(tensor, *args, **kwargs))
    captured = {}

    def install(inner, **kwargs):
        assert inner is text
        captured.update(kwargs)
        # The real installer takes a new canonical snapshot, including its
        # padding policy. The outer head must use this completed snapshot.
        inner._rpu_execution = {"model": {"num_cores": 8},
                                "prefill": {"linear_acc32": acc32, "padding_budget": 17}}

    def fused(outer, *, _audio_precision):
        assert _audio_precision == model._head_precision
        assert outer._rpu_execution is text._rpu_execution
        assert outer.lm_head.weight.dtype == head_dtype
        assert outer.lm_head._rpu_linear_partition == 1
        assert getattr(outer.lm_head, "weight_scale", None) is model._head_scale
        raise RuntimeError("host packing complete")

    monkeypatch.setattr(decoder, "_install_causal_decoder_forward", install)
    monkeypatch.setattr(qwen3, "_apply_fused_lm_head_for_rpu", fused)
    with pytest.raises(RuntimeError, match="host packing complete"):
        model._install_rpu()
    assert captured["_audio_component"] == "asr"
    assert captured["execution_config"] == {"model": {"num_cores": 8}, "prefill": {"linear_acc32": acc32}}
    torch.testing.assert_close(text.embed_tokens.weight, embedding)
    assert text.embed_tokens.weight.dtype == torch.float16
    assert packed_head[0][0].dtype == head_dtype
    if precision == "nvfp4":
        assert captured["nvfp4_tensor_scales"] is model._nvfp4_tensor_scales
        assert len(model._nvfp4_tensor_scales) == 7
        for scale in model._nvfp4_tensor_scales:
            assert scale.dtype == torch.float32 and scale.shape == (32,)
            assert torch.all(scale[28:] == 1)
    roles = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    for index, role in enumerate(roles):
        linear = layer.get_submodule(role)
        assert linear.weight.dtype == dtype
        if precision == "fp16":
            assert captured["scale_lists"] is None
        else:
            assert captured["scale_lists"][index][0] is linear.weight_scale
            assert linear.weight_scale.dtype == (torch.uint8 if precision == "nvfp4" else torch.float16)


def test_shape_switch_releases_graphs_before_audio_and_same_shape_replays(monkeypatch):
    events = []
    graphs = set()

    def clear():
        events.append("clear")
        graphs.clear()

    model = Qwen3ASRModel()
    model._closed = False
    model._tokenizer = model._feature_extractor = model._prompt = None
    model._text = SimpleNamespace(_rpu_decoder_graph_cache=SimpleNamespace(clear=clear))
    monkeypatch.setattr(asr, "_prepare_inputs", lambda *args: {
        "input_ids": torch.zeros(1, args[-1].size // 100, dtype=torch.long)})

    def generate(inputs, max_new_tokens):
        events.append("audio")
        graphs.add(("prefill", inputs["input_ids"].shape[1]))
        graphs.add(("decode", 1))
        assert len(graphs) <= 2
        return asr.ASRTranscription("", "", ())

    model._generate = generate
    try:
        for samples in (8000, 8000, 16000, 8000):
            model.transcribe(np.zeros(samples, dtype=np.float32))
        assert events == ["audio", "audio", "clear", "audio", "clear", "audio"]
    finally:
        model.close()


def test_request_reuses_kv_storage_and_resets_position_after_failure(monkeypatch):
    from rpu_backend.api import cache as cache_module

    allocations = []
    cache = SimpleNamespace(position=0)
    cache.reset_to_position = lambda pos: setattr(cache, "position", pos)
    def allocate(*args, **kwargs):
        allocations.append(cache)
        return cache
    monkeypatch.setattr(cache_module, "RPUCache", allocate)
    model = Qwen3ASRModel()
    model._topology = SimpleNamespace(attn_tp=8)
    def audio(*args, **kwargs):
        assert cache.position == 0
        cache.position = 17
        raise RuntimeError("interrupted request")
    model._audio = audio
    inputs = {"feature_lens": torch.tensor([50]), "input_features": torch.zeros(1, 128, 50)}
    for _ in range(2):
        with pytest.raises(RuntimeError, match="interrupted request"):
            model._generate(inputs, 64)
    assert len(allocations) == 1


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf")])
def test_causal_lm_logits_keep_finite_rejection_and_eos(monkeypatch, invalid):
    from rpu_backend.api import cache as cache_module

    cache = SimpleNamespace(reset_to_position=lambda pos: None, get_seq_length=lambda: 0)
    monkeypatch.setattr(cache_module, "RPUCache", lambda *args, **kwargs: cache)
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *args, **kwargs:
        tensor if args == ("rpu",) else original_to(tensor, *args, **kwargs))
    monkeypatch.setattr(Qwen3ASRModel, "_embed_tokens", lambda self, ids, features=None:
        torch.zeros(*ids.shape, 1024, dtype=torch.float16))
    model = Qwen3ASRModel()
    model._topology = SimpleNamespace(attn_tp=8)
    model._audio = lambda *args, **kwargs: SimpleNamespace(
        last_hidden_state=torch.zeros(1, 1024, dtype=torch.float16))
    logits = torch.zeros(1, 1, 151646, dtype=torch.float16)
    logits[0, 0, 151645] = 1
    if invalid is not None:
        logits[0, 0, 0] = invalid
    def forward(**kwargs):
        assert kwargs["logits_to_keep"] == 1 and kwargs["return_dict"]
        assert kwargs["past_key_values"] is cache
        return SimpleNamespace(logits=logits)
    model._language_model = forward
    model._tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "<asr_text>done")
    inputs = dict(feature_lens=torch.tensor([1]), input_features=torch.zeros(1, 128, 1),
                  input_ids=torch.tensor([[151676]]), attention_mask=torch.ones(1, 1))
    if invalid is None:
        result = model._generate(inputs, 64)
        assert result.token_ids == (151645,) and result.text == "done"
    else:
        with pytest.raises(FloatingPointError, match="non-finite ASR logits"):
            model._generate(inputs, 64)


def test_attention_head_padding_preserves_real_heads_and_projection():
    from rpu_backend.api._qwen3_asr_audio import _pad_attention

    torch.manual_seed(1)
    hidden = torch.randn(3, 896)
    qkv = []
    padded_qkv = []
    for _ in range(3):
        weight, bias = torch.randn(896, 896) * 0.02, torch.randn(896) * 0.02
        padded_weight, padded_bias = _pad_attention(weight, bias)
        qkv.append(torch.nn.functional.linear(hidden, weight, bias).view(3, 14, 64).transpose(0, 1))
        padded_qkv.append(torch.nn.functional.linear(hidden, padded_weight, padded_bias).view(3, 16, 64).transpose(0, 1))
        assert torch.count_nonzero(padded_weight[896:]) == 0
        assert torch.count_nonzero(padded_bias[896:]) == 0
    weight, bias = torch.randn(896, 896) * 0.02, torch.randn(896) * 0.02
    padded_weight, padded_bias = _pad_attention(weight, bias, output=True)
    reference = torch.nn.functional.scaled_dot_product_attention(*qkv).transpose(0, 1).reshape(3, 896)
    padded = torch.nn.functional.scaled_dot_product_attention(*padded_qkv).transpose(0, 1).reshape(3, 1024)
    torch.testing.assert_close(torch.nn.functional.linear(padded, padded_weight, padded_bias),
                               torch.nn.functional.linear(reference, weight, bias))


def test_audio_retirement_failure_preserves_owner(monkeypatch):
    from rpu_backend.api import causal_lm

    released = []
    monkeypatch.setattr(causal_lm, "_release_live_instance", lambda owner: released.append(owner))
    model = Qwen3ASRModel()
    model._closed = False
    def fail():
        raise RuntimeError("audio retirement failed")
    model._audio_encoder = SimpleNamespace(close=fail)
    with pytest.raises(RuntimeError, match="audio retirement"):
        model.close()
    assert not released and not model._closed and model._audio_encoder is not None
    model._closed = True


def test_resident_audio_rejects_multiple_utterances_before_execution():
    from rpu_backend.api._qwen3_asr_audio import ResidentAudioEncoder

    encoder = ResidentAudioEncoder(None, {})
    with pytest.raises(ValueError, match="one CPU FP16 mel"):
        encoder.forward(torch.zeros(128, 100, dtype=torch.float16), feature_lens=torch.tensor([50, 50]))


@pytest.mark.parametrize("width,output_width", [(896, 1024), (1024, 2048)])
def test_resident_graph_returns_final_projection_and_counts_both_projections(monkeypatch, width, output_width):
    from contextlib import nullcontext
    from rpu_backend.adapters import siglip
    from rpu_backend.api._qwen3_asr_audio import ResidentAudioEncoder

    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda value, *args, **kwargs:
        value if args == ("rpu",) else original_to(value, *args, **kwargs))
    cleared = []
    audio = SimpleNamespace(config=SimpleNamespace(d_model=width))
    encoder = ResidentAudioEncoder(audio, {"layers.0.fc1": 0, "proj1": 0, "proj2": 0, "conv_out": 0})
    encoder.handle = 1
    encoder.graph = SimpleNamespace(clear=lambda: cleared.append(True), capture=lambda signature: nullcontext())
    encoder.cache = SimpleNamespace(k_caches=[], v_caches=[])
    plan = SimpleNamespace(selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=[1])),
                           graph_key_words=lambda: [1])
    monkeypatch.setattr(siglip, "_siglip_component_execution_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(torch.rpu, "GraphSignature", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(torch.ops, "rpu", SimpleNamespace(
        siglip_forward=lambda handle, inputs, *args: inputs[:, :, :1].expand(-1, -1, output_width).clone(),
        spm_alloc_reset_temporary=lambda: None))
    first = encoder.encode(torch.ones(7, width, dtype=torch.float16))
    second = encoder.encode(torch.full((7, width), 2, dtype=torch.float16))
    assert first.shape == second.shape == (7, output_width)
    assert torch.all(first == 1) and torch.all(second == 2)
    assert cleared == [True] and encoder.calls == 2
    assert encoder.counters == {"layers.0.fc1": 2, "proj1": 2, "proj2": 2, "conv_out": 0}


@pytest.mark.parametrize("hidden_size", [1024, 2048])
def test_device_embedding_assembly_and_decode(monkeypatch, hidden_size):
    monkeypatch.setattr(asr, "_AUDIO_TOKEN_ID", 4)
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda value, *args, **kwargs:
        value if args == ("rpu",) else original_to(value, *args, **kwargs))
    weight = torch.arange(8 * hidden_size, dtype=torch.float32).reshape(8, hidden_size).half()
    model = Qwen3ASRModel()
    model._hidden_size = hidden_size
    model._text = SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight))
    model._embedding_weight = weight
    calls = []
    def gather(table, ids):
        assert table is weight and ids.dtype == torch.int32
        calls.append("gather")
        return table[ids.long()].clone()
    def assemble(table, ids, features, start):
        result = gather(table, ids)
        result[start:start + len(features)] = features
        calls.append("assemble")
        return result
    monkeypatch.setattr(torch.ops, "rpu", SimpleNamespace(
        gather_embedding=gather, assemble_inputs_embeds=assemble))
    ids = torch.tensor([[1, 4, 4, 7]])
    features = torch.full((2, hidden_size), -3, dtype=torch.float16)
    value = model._embed_tokens(ids, features)
    expected = weight[ids].clone()
    expected[:, 1:3] = features
    assert torch.equal(value, expected)
    saved = value.clone()
    assert torch.equal(model._embed_tokens(torch.tensor([[3]])), weight[3:4].unsqueeze(0))
    assert torch.equal(value, saved)
    assert calls == ["gather", "assemble"]

    for invalid in (torch.tensor([[-1]]), torch.tensor([[8]]), torch.tensor([[1.0]])):
        with pytest.raises(ValueError, match="valid CPU token IDs"):
            model._embed_tokens(invalid)
    for invalid in (torch.tensor([[4, 1, 4]]), torch.tensor([[1, 2, 3]])):
        with pytest.raises(ValueError, match="contiguous audio placeholder"):
            model._embed_tokens(invalid, features)
    assert calls == ["gather", "assemble"]


@pytest.mark.parametrize("frames", [50, 99, 100, 101, 420, 800])
@pytest.mark.parametrize("hidden_size", [896, 1024])
def test_resident_frontend_keeps_only_the_valid_chunk_prefix(monkeypatch, frames, hidden_size):
    from rpu_backend.api._qwen3_asr_audio import ResidentAudioEncoder

    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda value, *args, **kwargs:
        value if args == ("rpu",) else original_to(value, *args, **kwargs))
    monkeypatch.setattr(torch.nn.functional, "gelu", lambda value: value)
    def convolution(value):
        batch, width = value.shape[0], (value.shape[-1] + 7) // 8
        return torch.arange(batch * width, dtype=torch.float16).reshape(batch, 1, 1, width).expand(-1, hidden_size, 1, -1)
    audio = SimpleNamespace(config=SimpleNamespace(d_model=hidden_size), conv2d1=lambda value: value, conv2d2=lambda value: value,
                            conv2d3=convolution, conv_out=lambda value: value, proj2=lambda value: value)
    encoder = ResidentAudioEncoder(audio, {})
    encoder.positions = torch.arange(13, dtype=torch.float16).reshape(1, 13, 1).expand(8, 13, hidden_size).contiguous()
    encoder.encode = lambda value: value
    result = encoder.forward(torch.zeros(128, frames, dtype=torch.float16),
                             feature_lens=torch.tensor([frames])).last_hidden_state
    width = (min(frames, 100) + 7) // 8
    expected = [i * width + 2 * j for i, start in enumerate(range(0, frames, 100))
                for j in range((min(100, frames - start) + 7) // 8)]
    assert torch.equal(result, torch.tensor(expected, dtype=torch.float16)[:, None].expand(-1, hidden_size))
