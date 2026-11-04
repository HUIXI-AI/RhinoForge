"""Board-free rejection and lifetime checks for the bounded speech API."""
from types import SimpleNamespace
import wave

import numpy as np
import pytest
import torch

from rpu_backend.api import Qwen3TTSModel
from rpu_backend.api import qwen3_tts as tts
from rpu_backend.api.errors import RPUUnsupportedDtypeError


def test_fixed_predictor_sampling_steps_and_cache():
    class Predictor:
        generation_config = SimpleNamespace(eos_token_id=None, repetition_penalty=None)

        def __init__(self):
            self.calls = 0
            self.model = self
            self.small_to_mtp_projection = lambda value: value + 1
            self.lm_head = [self.head(index) for index in range(15)]

        def get_input_embeddings(self):
            def embed(index):
                def forward(ids):
                    assert ids.item() == index
                    return torch.full((1, 1, 1024), index, dtype=torch.float32)
                return forward
            return [embed(index) for index in range(14)]

        def head(self, index):
            def forward(hidden):
                assert hidden.item() == index
                logits = torch.zeros(1, 1, 16)
                logits[0, 0, index] = 1000
                return logits
            return forward

        def __call__(self, **kwargs):
            index = self.calls
            assert kwargs["past_key_values"] == (None if index == 0 else index)
            assert kwargs["use_cache"] and not kwargs["output_hidden_states"]
            assert not kwargs["output_attentions"]
            if index == 0:
                assert kwargs["inputs_embeds"].shape == (1, 2, 1024)
                assert torch.all(kwargs["inputs_embeds"] == 1)
            else:
                assert kwargs["inputs_embeds"].shape == (1, 1, 1024)
                assert torch.all(kwargs["inputs_embeds"] == index)
            self.calls += 1
            return SimpleNamespace(last_hidden_state=torch.tensor(index), past_key_values=self.calls)

    owner = Predictor()
    result = tts._predictor_generate(owner, torch.zeros(1, 2, 1024),
        max_new_tokens=15, do_sample=True)
    assert result.sequences.tolist() == [list(range(15))]
    for count, sampling in ((14, True), (15, False)):
        with pytest.raises(ValueError, match="15-step"):
            tts._predictor_generate(owner, torch.zeros(1, 2, 1024),
                max_new_tokens=count, do_sample=sampling)
    assert owner.calls == 15


def test_predictor_uses_checkpoint_sampling_and_rejects_other_recipes():
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper

    class Predictor:
        generation_config = SimpleNamespace(eos_token_id=None, repetition_penalty=None)

        def __init__(self):
            self.model = self
            self.small_to_mtp_projection = torch.nn.Identity()
            self.lm_head = [lambda hidden, index=index: self.scores(index) for index in range(15)]

        def get_input_embeddings(self):
            return [lambda ids: torch.zeros(1, 1, 1024)] * 14

        def scores(self, index):
            return torch.linspace(-2, 2, 128).roll(index).view(1, 1, -1)

        def __call__(self, **kwargs):
            return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"], past_key_values=None)

    owner = Predictor()
    torch.manual_seed(17)
    actual = tts._predictor_generate(owner, torch.zeros(1, 2, 1024),
        max_new_tokens=15, do_sample=True).sequences
    torch.manual_seed(17)
    expected = []
    for index in range(15):
        scores = owner.scores(index)[:, -1]
        scores = TemperatureLogitsWarper(0.9)(None, scores)
        scores = TopKLogitsWarper(50)(None, scores)
        expected.append(torch.multinomial(scores.softmax(-1), 1))
    assert torch.equal(actual, torch.cat(expected, dim=1))
    for recipe in ({"top_k": 0}, {"top_p": 0.8}, {"temperature": 1.0}):
        with pytest.raises(ValueError, match="sampling profile"):
            tts._predictor_generate(owner, torch.zeros(1, 2, 1024),
                max_new_tokens=15, do_sample=True, **recipe)


@pytest.mark.parametrize("text,language,limit", [
    ("", "English", 32), ("  ", "Chinese", 32), ("x" * 1025, "English", 32),
    ("hello", "French", 32), ("hello", "English", 1), ("hello", "English", 257),
    ("hello", "English", True), ("hello", "English", 1.5),
])
def test_request_rejection(text, language, limit):
    with pytest.raises(ValueError):
        tts._validate_request(text, language, limit)


def test_supported_request_boundaries():
    assert tts._validate_request("hello", "English", 2) == 2
    assert tts._validate_request("你好。", "Chinese", 256) == 256
    assert tts._validate_request("字" * 1024, "Chinese", 256) == 256


@pytest.mark.parametrize("text", [
    "大家好，欢迎收听今天的工作安排。" * 12,
    "We measured a distance of 3.14 meters today. " * 4,
    "没有标点的长文本" * 20,
])
def test_text_splitting_preserves_content_and_component_limits(text):
    tokenizer = lambda value: {"input_ids": list(value)}
    segments = tts._split_text(text, tokenizer)
    assert len(segments) > 1
    assert all(0 < len(segment) <= 16 for segment in segments)
    assert "".join("".join(segment.split()) for segment in segments) == "".join(text.split())
    if "3.14" in text:
        assert sum(segment.count("3.14") for segment in segments) == 4
    assert tts._split_text("  Hello.  ", tokenizer) == ["  Hello.  "]


def test_text_splitting_keeps_tiny_sentence_endings_with_their_clause():
    tokenizer = lambda value: {"input_ids": list(range((len(value) * 2 + 2) // 3))}
    sentence = "甲" * 14 + "，" + "乙" * 7 + "。"
    assert tts._split_text(sentence * 4, tokenizer) == [sentence] * 4


def test_text_splitting_does_not_synthesize_a_standalone_sentence_stop():
    tokenizer = lambda value: {"input_ids": list(value)}
    text = ("甲" * 12 + "。") * 3
    segments = tts._split_text(text, tokenizer)
    assert segments == ["甲" * 12 + "。"] * 3


def test_predictor_mlp_balance_prevents_product_overflow_without_changing_mlp():
    # Captured RPU layer-2 channel: both operands fit FP16; their product does not.
    gate = torch.tensor([[351.75, -2.]])
    up = torch.tensor([[440.25], [1.]])
    down = torch.tensor([[0.341796875, 0.5]])
    original_up, original_down = up.clone(), down.clone()
    activation = torch.nn.functional.silu(gate)
    assert torch.isinf(activation.half() * up.T.half()).any()
    balanced_up = tts._tts_mlp_weight(up, "predictor", 2, "mlp.up_proj")
    balanced_down = tts._tts_mlp_weight(down, "predictor", 2, "mlp.down_proj")
    product = activation.half() * balanced_up.T
    assert torch.isfinite(product).all()
    expected = (activation * up.T) @ down.T
    actual = product.float() @ balanced_down.float().T
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    # The paired power-of-two transform preserves FP32 reference semantics.
    torch.testing.assert_close((activation * balanced_up.float().T) @ balanced_down.float().T,
                               expected, rtol=0, atol=0)
    assert torch.equal(up, original_up) and torch.equal(down, original_down)


@pytest.mark.parametrize("component,index,path", [
    ("talker", 2, "mlp.up_proj"), ("predictor", 1, "mlp.up_proj"),
    ("predictor", 3, "mlp.down_proj"), ("predictor", 2, "mlp.gate_proj"),
])
def test_mlp_balance_leaves_other_weights_unchanged(component, index, path):
    weight = torch.tensor([[0.1, -0.25], [0.375, 1.]])
    assert torch.equal(tts._tts_mlp_weight(weight, component, index, path), weight.half())


def test_text_splitting_also_bounds_characters():
    segments = tts._split_text("字" * 300, lambda value: {"input_ids": [1]})
    assert list(map(len, segments)) == [128, 128, 44]


@pytest.mark.parametrize("fail_second", [False, True])
@pytest.mark.parametrize("punctuation", ["。", "，"])
def test_long_synthesis_owns_chunks_and_never_returns_partial_audio(monkeypatch, fail_second, punctuation):
    model = Qwen3TTSModel()
    model._closed = False
    model._components = {"talker": {}}
    calls = []
    draws = []
    torch.manual_seed(17)
    torch.rand(1)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(17)
    buffer = np.ones(8, dtype=np.float32)
    def generate(**kwargs):
        calls.append(kwargs["text"])
        if fail_second and len(calls) == 2:
            raise RuntimeError("segment failure")
        draws.append(torch.rand(1).item())
        buffer[:] = len(calls)
        model._last_generation = {"steps": len(calls) + 1, "eos": True}
        return [buffer], 24000
    model._wrapper = SimpleNamespace(
        model=SimpleNamespace(speaker_encoder_sample_rate=24000,
            extract_speaker_embedding=lambda **kwargs: torch.ones(1)),
        processor=SimpleNamespace(tokenizer=lambda value: {"input_ids": list(value)}),
        generate_voice_clone=generate)
    monkeypatch.setattr(tts, "_read_reference", lambda path: (np.ones(12000, dtype=np.float32), 24000))
    try:
        text = ("甲" * 10 + punctuation) * 3
        if fail_second:
            with pytest.raises(RuntimeError, match="segment failure"):
                model.synthesize(text, reference_audio="unused")
            assert len(calls) == 2
        else:
            result, rate = model.synthesize(text, reference_audio="unused")
            assert rate == 24000 and np.array_equal(result, np.repeat([1, 2, 3], 8))
            assert "".join(calls) == text.replace("，", "。")
            assert model._last_generation == {"steps": 9, "eos": True, "segment_steps": (2, 3, 4)}
        assert len(set(draws)) == 1
        assert torch.equal(torch.random.get_rng_state(), expected_rng)
        assert model._lock.acquire(blocking=False)
        model._lock.release()
    finally:
        model._closed = True


@pytest.mark.parametrize("rate,channels,frames", [(16000, 1, 16000), (24000, 2, 24000), (24000, 1, 11999), (24000, 1, 480001)])
def test_reference_rejected_before_decoding(monkeypatch, rate, channels, frames):
    import soundfile

    monkeypatch.setattr(soundfile, "info", lambda path: SimpleNamespace(samplerate=rate, channels=channels, frames=frames))
    monkeypatch.setattr(soundfile, "read", lambda *a, **kw: pytest.fail("unsupported audio decoded"))
    with pytest.raises(ValueError, match="reference"):
        tts._read_reference("local.wav")


def test_reference_is_real_float32_audio(tmp_path):
    path = tmp_path / "ref.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(np.ones(12000, dtype="<i2").tobytes())
    samples, rate = tts._read_reference(path)
    assert rate == 24000 and samples.shape == (12000,) and samples.dtype == np.float32


def test_silent_reference_rejected(tmp_path):
    import soundfile

    path = tmp_path / "silence.wav"
    soundfile.write(path, np.zeros(12000), 24000)
    with pytest.raises(ValueError, match="silent"):
        tts._read_reference(path)


@pytest.mark.parametrize("ids,finished", [([1, 2, 2150], True), ([1, 2, 3], False), ([], False)])
def test_generation_requires_eos_before_codec(ids, finished):
    model = Qwen3TTSModel()
    result = SimpleNamespace(sequences=torch.tensor([ids], dtype=torch.long))
    model._restore_talker_generate = (None, lambda **kw: result)
    model._wrapper = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
        talker_config=SimpleNamespace(codec_eos_token_id=2150))))
    if finished:
        assert model._generate_talker() is result
    else:
        with pytest.raises(RuntimeError, match="without EOS"):
            model._generate_talker()
    assert model._last_generation == {"steps": len(ids), "eos": finished}


def test_dtype_rejected_before_loading(monkeypatch):
    from rpu_backend.adapters import qwen3_tts

    monkeypatch.setattr(qwen3_tts, "load_cpu_model", lambda *a: pytest.fail("weights loaded"))
    with pytest.raises(RPUUnsupportedDtypeError):
        Qwen3TTSModel.from_pretrained("missing", source="missing", dtype=torch.bfloat16)


def test_tts_cache_capacity_reaches_cold_install_and_rejects_invalid_before_load(monkeypatch):
    from rpu_backend.adapters import qwen3_tts
    from rpu_backend.api import causal_lm

    loaded, installed = [], []
    monkeypatch.setattr(qwen3_tts, "load_cpu_model", lambda *args: loaded.append(args) or object())
    monkeypatch.setattr(torch, "rpu", SimpleNamespace(set_caching_allocator=lambda enabled: None), raising=False)
    monkeypatch.setattr(causal_lm, "_claim_live_instance", lambda model: None)
    monkeypatch.setattr(causal_lm, "_release_live_instance", lambda model: None)
    monkeypatch.setattr(Qwen3TTSModel, "_install_rpu", lambda model: installed.append(model._max_seq_len))
    with Qwen3TTSModel.from_pretrained("checkpoint", source="source", max_seq_len=16384):
        assert installed == [16384]
    for invalid in (True, 0, -1, 512.0):
        with pytest.raises(ValueError, match="positive integer"):
            Qwen3TTSModel.from_pretrained("must-not-load", source="source", max_seq_len=invalid)
    assert loaded == [("checkpoint", "source")]


@pytest.mark.parametrize("options,expected", [({}, "rpu"), ({"codec_backend": "cpu"}, "cpu"),
                                             ({"codec_backend": "rpu"}, "rpu")])
def test_codec_backend_default_and_explicit_override(monkeypatch, options, expected):
    from rpu_backend.adapters import qwen3_tts
    from rpu_backend.api import causal_lm

    installed = []
    monkeypatch.setattr(qwen3_tts, "load_cpu_model", lambda *a: object())
    monkeypatch.setattr(torch, "rpu", SimpleNamespace(set_caching_allocator=lambda enabled: None), raising=False)
    monkeypatch.setattr(causal_lm, "_claim_live_instance", lambda model: None)
    monkeypatch.setattr(causal_lm, "_release_live_instance", lambda model: None)
    monkeypatch.setattr(Qwen3TTSModel, "_install_rpu",
                        lambda model: installed.append(model._codec_convolutions["backend"]))
    with Qwen3TTSModel.from_pretrained("checkpoint", source="source", **options):
        assert installed == [expected]


def test_failed_load_releases_owner(monkeypatch):
    from rpu_backend.adapters import qwen3_tts
    from rpu_backend.api import causal_lm

    calls = []
    monkeypatch.setattr(causal_lm, "_claim_live_instance", lambda model: calls.append("claim"))
    monkeypatch.setattr(causal_lm, "_release_live_instance", lambda model: calls.append("release"))
    def fail(*args):
        calls.append("load")
        raise ValueError("wrong identity")
    monkeypatch.setattr(qwen3_tts, "load_cpu_model", fail)
    with pytest.raises(ValueError, match="identity"):
        Qwen3TTSModel.from_pretrained("missing", source="missing")
    assert calls == ["claim", "load", "release"]


def test_close_and_stats_are_serial():
    model = Qwen3TTSModel()
    model._lock.acquire()
    try:
        for method in (model.close, model.execution_stats):
            with pytest.raises(RuntimeError, match="serial"):
                method()
    finally:
        model._lock.release()
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.synthesize("你好", reference_audio="missing")


def test_close_restores_predictor_generation(monkeypatch):
    from rpu_backend.api import causal_lm

    monkeypatch.setattr(causal_lm, "_release_live_instance", lambda owner: None)
    original = object()
    predictor = SimpleNamespace(generate=object())
    model = Qwen3TTSModel()
    model._closed = False
    model._restore_generate = (predictor, original)
    talker = SimpleNamespace(generate=object())
    model._restore_talker_generate = (talker, original)
    model.close()
    assert predictor.generate is original and model._restore_generate is None
    assert talker.generate is original and model._restore_talker_generate is None


def test_decoder_rejects_bad_shapes_and_nonfinite_without_device_work():
    for hidden in (torch.zeros(2, 1, 1024), torch.zeros(1, 1, 1000), torch.full((1, 1, 1024), float("nan"))):
        with pytest.raises(ValueError):
            tts._decoder_forward({}, hidden, None, None, None, None, False)


def test_graph_creation_failure_does_not_create_native_handle(monkeypatch):
    from rpu_backend import graph

    model = Qwen3TTSModel()
    core = SimpleNamespace(layers=[])
    model._wrapper = SimpleNamespace(model=SimpleNamespace(
        talker=SimpleNamespace(model=core, code_predictor=SimpleNamespace(model=core))))

    prepared = []
    monkeypatch.setattr(graph.GraphRuntimePolicy, "prepare_arenas",
                        lambda policy: prepared.append(policy.graph_arena_count) or True)
    def fail_graph(*, runtime_policy):
        assert prepared == [6] and runtime_policy.graph_arena_count == 6
        raise RuntimeError("graph allocation failed")

    monkeypatch.setattr(graph, "GraphCache", fail_graph)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_create", lambda: pytest.fail("untracked handle created"), raising=False)
    with pytest.raises(RuntimeError, match="graph allocation failed"):
        model._install_rpu()
    assert model._components == {}


@pytest.mark.parametrize("failure", ["graph", "destroy"])
def test_retirement_failure_retains_parent_and_blocks_new_owner_after_gc(monkeypatch, failure):
    import gc
    import weakref
    from rpu_backend.api import causal_lm, _execution
    from rpu_backend.runtime import _native_retirement

    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])
    events = []
    model = Qwen3TTSModel()
    model._closed = False
    causal_lm._claim_live_instance(model)
    for name, handle in (("talker", 1), ("predictor", 2)):
        def clear(name=name):
            events.append(("clear", name))
            if failure == "graph":
                raise RuntimeError("graph retirement failed")

        def destroy(value):
            events.append(("destroy", value))
            if failure == "destroy":
                raise RuntimeError("native retirement failed")

        cache = SimpleNamespace(clear=clear, cache_invariant_ok=lambda: True)
        part = {"handle": handle, "graph": cache}
        handle_name = "_native_" + name + "_handle"
        setattr(model, handle_name, handle)
        resource = _native_retirement._InstalledNativeResource(
            model, handle, destroy, graphs=(cache,), keepalive=part,
            label="test TTS", handle_name=handle_name)
        part["resource"] = resource
        model._components[name] = part
        resource.take_ownership(model)
        assert not resource.finalizer.alive
    with pytest.raises(RuntimeError, match="retirement failed"):
        model.close()
    previous = list(events)
    with pytest.raises(RuntimeError, match="restart"):
        model.close()
    with pytest.raises(RuntimeError, match="restart"):
        model.synthesize("hello", language="English", reference_audio="unused")
    assert events == previous  # No retry after uncertain Graph/native cleanup.
    if failure == "destroy":
        assert events[:2] == [("clear", "talker"), ("clear", "predictor")]
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is not None  # Quarantine retains both components and heads.
    with pytest.raises(RuntimeError, match="unsafe"):
        causal_lm._claim_live_instance(Qwen3TTSModel())
    assert len(_native_retirement._FAILED_RETIREMENTS) == 2


def test_install_failure_preserves_original_error_when_cleanup_also_fails(monkeypatch):
    from rpu_backend.adapters import qwen3_tts
    from rpu_backend.api import causal_lm

    monkeypatch.setattr(causal_lm, "_claim_live_instance", lambda owner: None)
    monkeypatch.setattr(qwen3_tts, "load_cpu_model", lambda *args: object())
    def install(self):
        raise ValueError("original install failure")
    def close(self):
        raise RuntimeError("cleanup failure")
    monkeypatch.setattr(Qwen3TTSModel, "_install_rpu", install)
    monkeypatch.setattr(Qwen3TTSModel, "close", close)
    with pytest.raises(ValueError, match="original install failure") as captured:
        Qwen3TTSModel.from_pretrained("unused", source="unused")
    assert "cleanup failure" in captured.value.__notes__[0]


@pytest.mark.parametrize("token_count,silent", [(32, False), (33, False), (32, True)])
def test_text_token_limit_precedes_reference_work_and_returns_owned_output(monkeypatch, token_count, silent):
    source_wave = np.full(24, 0 if silent else 0.5, dtype=np.float32)
    calls = []
    def reference(path):
        calls.append("reference")
        return np.zeros(12000, dtype=np.float32), 24000
    def generate(**kwargs):
        calls.append("generate")
        model._last_generation = {"steps": 2, "eos": True}
        assert kwargs["do_sample"] and kwargs["subtalker_dosample"]
        assert kwargs["max_new_tokens"] == 256
        return [source_wave], 24000
    def speaker(**kwargs):
        calls.append("speaker")
        return torch.zeros(1024)
    model = Qwen3TTSModel()
    model._closed = False
    model._components = {"talker": {}}
    model._wrapper = SimpleNamespace(
        model=SimpleNamespace(speaker_encoder_sample_rate=24000, extract_speaker_embedding=speaker),
        processor=SimpleNamespace(tokenizer=lambda text: {"input_ids": list(range(token_count))}),
        generate_voice_clone=generate)
    monkeypatch.setattr(tts, "_read_reference", reference)
    try:
        if token_count == 33:
            with pytest.raises(ValueError, match="12 tokenizer tokens"):
                model.synthesize("hello", reference_audio="unused")
            assert calls == []
        elif silent:
            with pytest.raises(FloatingPointError, match="silent"):
                model.synthesize("hello", reference_audio="unused")
            assert calls == ["reference", "speaker", "generate"]
        else:
            result, rate = model.synthesize("hello", reference_audio="unused")
            source_wave[:] = 1
            assert calls == ["reference", "speaker", "generate"]
            assert rate == 24000 and np.array_equal(result, np.full(24, 0.5))
    finally:
        # This host fixture has no claimed owner or native resources.
        model._closed = True


def test_speaker_cache_tracks_audio_content_and_never_runs_reference_codec(monkeypatch):
    wave = np.zeros(12000, dtype=np.float32)
    embeddings = []
    prompts = []
    def speaker(**kwargs):
        value = torch.tensor([float(kwargs["audio"].sum())])
        embeddings.append(value)
        return value
    def generate(**kwargs):
        assert "ref_audio" not in kwargs
        prompt = kwargs["voice_clone_prompt"]
        assert prompt["ref_code"] == [None] and prompt["icl_mode"] == [False]
        assert prompt["x_vector_only_mode"] == [True]
        prompts.append(prompt)
        model._last_generation = {"steps": 2, "eos": True}
        return [np.ones(24, dtype=np.float32)], 24000
    model = Qwen3TTSModel()
    model._closed = False
    model._components = {"talker": {}}
    model._wrapper = SimpleNamespace(
        model=SimpleNamespace(speaker_encoder_sample_rate=24000, extract_speaker_embedding=speaker),
        processor=SimpleNamespace(tokenizer=lambda text: {"input_ids": [1]}),
        generate_voice_clone=generate)
    monkeypatch.setattr(tts, "_read_reference", lambda path: (wave, 24000))
    try:
        model.synthesize("first", reference_audio="same.wav")
        model.synthesize("second", reference_audio="same.wav")
        wave[0] = 1
        model.synthesize("third", reference_audio="same.wav")
        assert len(embeddings) == 2
        assert prompts[0] is prompts[1] and prompts[2] is not prompts[1]
        assert prompts[2]["ref_spk_embedding"][0].item() == 1
    finally:
        model._closed = True


@pytest.mark.parametrize("head_only,hidden_states", [(False, False), (True, True), (True, False)])
@pytest.mark.parametrize("fused_heads", [False, True])
def test_predictor_reuses_kv_with_fresh_positions_inputs_and_owned_outputs(monkeypatch, head_only, hidden_states, fused_heads):
    from contextlib import nullcontext
    from rpu_backend.api import cache as cache_module
    from rpu_backend.runtime import decoder

    allocations, positions = [], []
    class Cache:
        def __init__(self, *args):
            self.position = 0
            self.k_caches = self.v_caches = []
            allocations.append(self)
        def reset_to_position(self, pos):
            self.position = pos
        def update_position(self, count):
            self.position += count
    monkeypatch.setattr(cache_module, "RPUCache", Cache)
    original_to = torch.Tensor.to
    def to(tensor, *args, **kwargs):
        if args == ("rpu",):
            return tensor
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", to)
    plan = SimpleNamespace(selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(7,))),
                           graph_key_words=lambda: (7,))
    monkeypatch.setattr(decoder, "_text_prefill_execution_plan", lambda *args: (16, 16, plan))
    output = torch.empty(1, 16, 1024, dtype=torch.float16)
    def forward(handle, hidden, k, v, mask, position, causal, **kwargs):
        positions.append(position)
        result = output[:, :hidden.shape[1]].copy_(hidden)
        if hidden.shape[1] == 16:
            assert position == 0 and hidden[:, 2:].eq(0).all()
            assert kwargs["planned_stage_descriptor"] == (7,)
            result[:, 2:] = 17  # Padded rows must never become the head input.
        return result
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", forward, raising=False)
    part = dict(layers=5, capacity=32, name="predictor", handle=1, descriptor=[],
                signature=None, graph=SimpleNamespace(capture=lambda sig: nullcontext()), calls=0, tokens=0,
                device_head_only=head_only, planner=object(), fused_heads=fused_heads, fused_head_calls=0)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_tts_predictor_logits",
        lambda handle: torch.full((1, 1, 2048), positions[-1], dtype=torch.float16), raising=False)
    borrowed = head_only and not hidden_states
    if borrowed:
        monkeypatch.setattr(torch.Tensor, "cpu", lambda tensor: pytest.fail("Predictor hidden copied to CPU"))
    first = tts._decoder_forward(part, torch.ones(1, 2, 1024), None, None, None, None, hidden_states)
    if borrowed:
        assert first.last_hidden_state.shape == (1, 1, 1024)
        if fused_heads:
            assert part["head_index"] == 1 and part["head_logits"].eq(0).all()
        first_head = tts._take_head_input(part, first.last_hidden_state).clone()
    else:
        first_head = first.last_hidden_state
    tts._decoder_forward(part, torch.full((1, 1, 1024), 2.), first.past_key_values,
                         None, None, None, hidden_states)
    if borrowed and fused_heads:
        assert part["head_index"] == 2 and part["head_logits"].eq(2).all()
    second = tts._decoder_forward(part, torch.zeros(1, 2, 1024), None, None, None, None, hidden_states)
    if borrowed and fused_heads:
        assert part["head_index"] == 1 and part["head_logits"].eq(0).all()
        assert part["fused_head_calls"] == 0
    else:
        assert "head_logits" not in part
    assert len(allocations) == 1 and positions == [0, 2, 0]
    assert first.past_key_values.position == 2
    assert first_head.eq(1).all() and second.last_hidden_state.eq(0).all()


def test_head_reuses_only_current_single_token_device_output():
    hidden = torch.zeros(1, 1, 1024)
    device = object()
    part = {"head_hidden": hidden, "head_device": device}
    assert tts._take_head_input(part, hidden.clone()) is None
    assert part == {}
    part.update(head_hidden=hidden, head_device=device)
    assert tts._take_head_input(part, hidden) is device
    assert tts._take_head_input(part, hidden) is None
    prompt = torch.zeros(1, 2, 1024)
    part.update(head_hidden=prompt, head_device=device)
    assert tts._take_head_input(part, prompt) is None
    assert part == {}


def test_talker_prefill_padding_positions_shape_switch_and_owned_outputs(monkeypatch):
    from contextlib import nullcontext
    import rpu_backend.graph as graph_module
    from rpu_backend.api import cache as cache_module
    from rpu_backend.runtime import decoder

    monkeypatch.setattr(graph_module, "GraphSignature", lambda **kwargs: kwargs)

    class Cache:
        def __init__(self, *args):
            self.position = 0
            self.k_caches = self.v_caches = []
        def reset_to_position(self, position):
            self.position = position
        def update_position(self, count):
            self.position += count
    monkeypatch.setattr(cache_module, "RPUCache", Cache)
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *args, **kwargs:
        tensor if args == ("rpu",) else original_to(tensor, *args, **kwargs))
    captures, evicted, calls = [], [], []
    def capture(signature):
        captures.append(signature)
        return nullcontext()
    graph = SimpleNamespace(capture=capture, evict=evicted.append,
                            clear=lambda: pytest.fail("decode Graph must be retained"))
    plan = SimpleNamespace(selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(7,))),
                           graph_key_words=lambda: (7,))
    monkeypatch.setattr(decoder, "_text_prefill_execution_plan", lambda *args: (16, 16, plan))
    buffer = torch.empty(1, 16, 1024, dtype=torch.float16)
    def forward(handle, hidden, k, v, mask, position, causal, **kwargs):
        calls.append((position, hidden.clone(), kwargs["planned_stage_descriptor"]))
        return buffer[:, :hidden.shape[1]].copy_(hidden)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", forward, raising=False)
    part = dict(layers=28, capacity=128, generation_limit=64, name="talker", handle=1,
                descriptor=(1,), signature=None, graph=graph, planner=object(), calls=0, tokens=0)
    first = tts._decoder_forward(part, torch.ones(1, 3, 1024), None, None, None, None, True)
    cache = first.past_key_values
    assert cache.position == 3 and first.last_hidden_state.shape == (1, 3, 1024)
    assert calls[0][1][:, 3:].eq(0).all() and calls[0][2] == (7,)
    second = tts._decoder_forward(part, torch.full((1, 1, 1024), 9.), cache, None, None, None, False)
    assert cache.position == 4 and calls[1][0] == 3 and calls[1][2] == (1,)
    assert first.last_hidden_state.eq(1).all() and second.last_hidden_state.eq(9).all()
    for _ in range(2):
        third = tts._decoder_forward(part, torch.zeros(1, 5, 1024), None, None, None, None, False)
        assert third.past_key_values is cache and cache.position == 5
    assert evicted == [captures[0]] and calls[2][0] == calls[3][0] == 0
    assert captures[0] != captures[2] and captures[2] == captures[3]


@pytest.mark.parametrize("kernel,stride", [(2, 2), (16, 8)])
def test_codec_convolution_paths_match_fp32(kernel, stride):
    from rpu_backend.adapters.qwen3_tts.codec import install_codec_convolutions

    torch.manual_seed(12)
    module = torch.nn.ConvTranspose1d(8, 8, kernel, stride=stride).eval()
    inputs = [torch.randn(1, 8, n) for n in (70, 91)]
    with torch.inference_mode():
        expected = [module(x) for x in inputs]
        restore = []
        original = module.forward
        install_codec_convolutions(module, restore)
        for x, reference in zip(inputs, expected):
            result = module(x)
            assert result.dtype == torch.float32 and torch.isfinite(result).all()
            torch.testing.assert_close(result, reference, rtol=1e-5, atol=2e-6)
        assert restore == [(module, original)]
        module.forward = restore[0][1]
        assert torch.equal(module(inputs[0]), expected[0])


def test_codec_geometry_rejected_before_any_patch():
    from rpu_backend.adapters.qwen3_tts.codec import install_codec_convolutions

    modules = torch.nn.Sequential(torch.nn.ConvTranspose1d(4, 4, 2, stride=2), torch.nn.ConvTranspose1d(4, 4, 3, padding=1))
    originals = [m.forward for m in modules]
    restore = []
    with pytest.raises(ValueError, match="geometry"):
        install_codec_convolutions(modules, restore)
    assert restore == [] and [m.forward for m in modules] == originals


@pytest.mark.parametrize("same_hidden,same_index", [(True, True), (False, True), (True, False)])
def test_fused_predictor_logits_are_single_use_and_match_hidden_and_head(same_hidden, same_index):
    hidden = torch.zeros(1, 1, 1024)
    logits = torch.ones(1, 1, 2048, dtype=torch.float16)
    part = dict(head_hidden=hidden, head_logits=logits, head_index=3, fused_head_calls=0)
    result = tts._take_predictor_logits(part, hidden if same_hidden else hidden.clone(), 3 if same_index else 4)
    assert (result is logits) == (same_hidden and same_index)
    assert part["fused_head_calls"] == int(same_hidden and same_index)
    assert tts._take_predictor_logits(part, hidden, 3) is None
