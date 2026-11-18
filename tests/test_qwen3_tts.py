"""Board-free rejection and lifetime checks for the bounded speech API."""
from types import SimpleNamespace
import wave

import numpy as np
import pytest
import torch

from rpu_backend.api import Qwen3TTSModel
from rpu_backend.api import _qwen3_audio_stages as stages
from rpu_backend.api import qwen3_tts as tts
from rpu_backend.api.errors import RPUUnsupportedDtypeError


def install_reference_sampler(monkeypatch, seen=None):
    """Stand the replaced warper chain in the fused op's place.

    These checks are board-free, so the native op is not built here. Driving the
    seam with the chain it replaced keeps them checking the Predictor's own
    behaviour — the arguments it hands over and what it does with the draw —
    while test_host_topk_sample checks the op's distribution for real.
    """
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper

    def chain(logits, temperature, top_k, generator=None):
        if seen is not None:
            seen.append((logits.clone(), temperature, top_k))
        scores = TemperatureLogitsWarper(temperature)(None, logits)
        scores = TopKLogitsWarper(top_k)(None, scores)
        return torch.multinomial(scores.softmax(-1), 1).reshape(-1)

    monkeypatch.setattr(torch.ops.rpu, "sample_topk_lastdim_host", chain, raising=False)
    return chain


@pytest.mark.parametrize("input_width", [1024, 2048])
def test_fixed_predictor_sampling_steps_and_cache(input_width, monkeypatch):
    install_reference_sampler(monkeypatch)
    class Predictor:
        generation_config = SimpleNamespace(eos_token_id=None, repetition_penalty=None)

        def __init__(self):
            self.calls = 0
            self.model = self
            self.small_to_mtp_projection = lambda value: value[..., :1024] + 1
            self.lm_head = [self.head(index) for index in range(15)]

        def get_input_embeddings(self):
            def embed(index):
                def forward(ids):
                    assert ids.item() == index
                    return torch.full((1, 1, input_width), index, dtype=torch.float32)
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
    result = tts._predictor_generate(owner, torch.zeros(1, 2, input_width),
        input_width=input_width, max_new_tokens=15, do_sample=True)
    assert result.sequences.tolist() == [list(range(15))]
    for count, sampling in ((14, True), (15, False)):
        with pytest.raises(ValueError, match="15-step"):
            tts._predictor_generate(owner, torch.zeros(1, 2, input_width),
                input_width=input_width, max_new_tokens=count, do_sample=sampling)
    assert owner.calls == 15


def test_predictor_uses_checkpoint_sampling_and_rejects_other_recipes(monkeypatch):
    """The fused op stands exactly where the warper chain stood.

    Driving the seam with that chain has to reproduce the historical sequence,
    which pins the arguments the Predictor hands over; the op's own draw is a
    different number of random values and is checked in test_host_topk_sample.
    """
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

    owner, seen = Predictor(), []
    install_reference_sampler(monkeypatch, seen)
    torch.manual_seed(17)
    actual = tts._predictor_generate(owner, torch.zeros(1, 2, 1024),
        input_width=1024, max_new_tokens=15, do_sample=True).sequences
    torch.manual_seed(17)
    expected = []
    for index in range(15):
        scores = owner.scores(index)[:, -1]
        scores = TemperatureLogitsWarper(0.9)(None, scores)
        scores = TopKLogitsWarper(50)(None, scores)
        expected.append(torch.multinomial(scores.softmax(-1), 1))
    assert torch.equal(actual, torch.cat(expected, dim=1))
    # Pre-temperature logits, the checkpoint's recipe, once per step.
    assert len(seen) == 15 and {(step[1], step[2]) for step in seen} == {(0.9, 50)}
    for index, (logits, _, _) in enumerate(seen):
        assert logits.dtype is torch.float32 and logits.shape == (1, 128)
        assert torch.equal(logits, owner.scores(index)[:, -1].float())
    for recipe in ({"top_k": 0}, {"top_p": 0.8}, {"temperature": 1.0}):
        with pytest.raises(ValueError, match="sampling profile"):
            tts._predictor_generate(owner, torch.zeros(1, 2, 1024),
                input_width=1024, max_new_tokens=15, do_sample=True, **recipe)


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
        assert torch.is_inference_mode_enabled()
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


def test_stream_synthesis_yields_owned_chunks_and_releases_lock_on_close(monkeypatch):
    model = Qwen3TTSModel()
    model._closed = False
    model._components = {"talker": {}}
    calls = []
    buffer = np.ones(8, dtype=np.float32)
    def generate(**kwargs):
        calls.append(kwargs["text"])
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
        with pytest.raises(ValueError, match="1–1024"):
            model.stream_synthesize("", reference_audio="unused")
        unstarted = model.stream_synthesize("甲", reference_audio="unused")
        unstarted.close()
        assert model._last_generation is None
        stream = model.stream_synthesize(("甲" * 10 + "。") * 3, reference_audio="unused")
        first, rate = next(stream)
        assert not torch.is_inference_mode_enabled()
        second, _ = next(stream)
        assert rate == 24000 and first.tolist() == [1] * 8 and second.tolist() == [2] * 8
        assert not np.shares_memory(first, second)
        with pytest.raises(RuntimeError, match="serial"):
            model.execution_stats()
        stream.close()
        assert model._last_generation == {"steps": 5, "eos": False, "segment_steps": (2, 3)}
        assert model._lock.acquire(blocking=False)
        model._lock.release()
    finally:
        model._closed = True


def test_stream_synthesis_keeps_yielded_chunk_when_later_segment_fails(monkeypatch):
    model = Qwen3TTSModel()
    model._closed = False
    model._components = {"talker": {}}
    calls = 0
    def generate(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("segment failure")
        model._last_generation = {"steps": 2, "eos": True}
        return [np.ones(8, dtype=np.float32)], 24000
    model._wrapper = SimpleNamespace(
        model=SimpleNamespace(speaker_encoder_sample_rate=24000,
            extract_speaker_embedding=lambda **kwargs: torch.ones(1)),
        processor=SimpleNamespace(tokenizer=lambda value: {"input_ids": list(value)}),
        generate_voice_clone=generate)
    monkeypatch.setattr(tts, "_read_reference", lambda path: (np.ones(12000, dtype=np.float32), 24000))
    try:
        stream = model.stream_synthesize(("甲" * 10 + "。") * 3, reference_audio="unused")
        first, rate = next(stream)
        with pytest.raises(RuntimeError, match="segment failure") as captured:
            next(stream)
        assert rate == 24000 and first.tolist() == [1] * 8
        assert "after 1 completed audio chunk(s)" in captured.value.__notes__[0]
        assert model._last_generation == {"steps": 2, "eos": False, "segment_steps": (2,)}
        assert model._lock.acquire(blocking=False)
        model._lock.release()
    finally:
        model._closed = True


def _fake_clock(monkeypatch):
    """Replace the clock so stage arithmetic is exact, not wall-clock flaky.

    The request total is read in this module and the per-stage intervals in the
    shared timer, so both clocks move together or the remainder goes negative.
    """
    now = [0]
    clock = SimpleNamespace(perf_counter_ns=lambda: now[0])
    monkeypatch.setattr(tts, "time", clock)
    monkeypatch.setattr(stages, "time", clock)
    return lambda ms: now.__setitem__(0, now[0] + int(ms * 1e6))


def test_stage_timer_charges_nested_time_to_the_inner_stage_only(monkeypatch):
    """The Predictor runs inside the Talker's loop, so plain elapsed time would double-count."""
    spend = _fake_clock(monkeypatch)
    timer = stages.StageTimer("TTS", tts._STAGE_NAMES)
    with timer.stage("talker"):
        spend(3)
        with timer.stage("predictor"):
            spend(10)
        spend(2)
    spend(1)  # outside every stage
    with timer.stage("codec_decode"):
        spend(4)
    assert timer.milliseconds(total_ns=20_000_000) == {
        "speaker_embedding": 0.0, "talker": 5.0, "predictor": 10.0, "codec_decode": 4.0, "other": 1.0}


def test_stage_timer_closes_a_stage_that_raised_and_rejects_unknown_names(monkeypatch):
    spend = _fake_clock(monkeypatch)
    timer = stages.StageTimer("TTS", tts._STAGE_NAMES)
    with pytest.raises(ValueError):
        with timer.stage("talker"):
            spend(7)
            raise ValueError("segment failed")
    assert timer.milliseconds(total_ns=7_000_000)["talker"] == 7.0
    # A typo in a stage name would otherwise hide its time inside the remainder.
    with timer.stage("taker"):
        spend(1)
    with pytest.raises(RuntimeError, match="unknown stages"):
        timer.milliseconds(total_ns=8_000_000)


def test_synthesize_records_exclusive_stage_times_and_caches_the_speaker_stage(monkeypatch):
    """stage_ms must locate the cost: the autoregression or the codec decode."""
    spend = _fake_clock(monkeypatch)
    model = Qwen3TTSModel()
    model._closed = False
    model._components = {"talker": {}}

    def generate(**kwargs):
        # The real installs open these; the Predictor's sampling nests in the Talker's loop.
        with model._stages.stage("talker"):
            spend(3)
            with model._stages.stage("predictor"):
                spend(10)
            spend(2)
        with model._stages.stage("codec_decode"):
            spend(4)
        model._last_generation = {"steps": 2, "eos": True}
        return [np.ones(8, dtype=np.float32)], 24000

    def embed(**kwargs):
        spend(6)
        return torch.ones(1)

    model._wrapper = SimpleNamespace(
        model=SimpleNamespace(speaker_encoder_sample_rate=24000, extract_speaker_embedding=embed),
        processor=SimpleNamespace(tokenizer=lambda value: {"input_ids": list(value)}),
        generate_voice_clone=generate)

    def read_reference(path):
        spend(1)  # stage-external work: reading and validating the reference
        return np.ones(12000, dtype=np.float32), 24000
    monkeypatch.setattr(tts, "_read_reference", read_reference)
    try:
        model.synthesize("甲" * 10 + "。", reference_audio="unused")
        assert model._stage_ms == {"speaker_embedding": 6.0, "talker": 5.0, "predictor": 10.0,
                                   "codec_decode": 4.0, "other": 1.0}
        # Disjoint stages plus the remainder are the whole request, so nothing hides.
        assert sum(model._stage_ms.values()) == 26.0
        model.synthesize("甲" * 10 + "。", reference_audio="unused")
        # The embedding is cached per reference audio; the stage stays visible at 0.
        assert model._stage_ms["speaker_embedding"] == 0.0 and model._stage_ms["talker"] == 5.0
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
            tts._decoder_forward({"hidden_size": 1024}, hidden, None, None, None, None, False)


def test_graph_creation_failure_does_not_create_native_handle(monkeypatch):
    from rpu_backend import graph

    model = Qwen3TTSModel()
    def core(layers, vocab):
        return SimpleNamespace(layers=[None] * layers, config=SimpleNamespace(
            num_hidden_layers=layers, hidden_size=1024, intermediate_size=3072,
            num_attention_heads=16, num_key_value_heads=8, head_dim=128,
            rms_norm_eps=1e-6, hidden_act="silu", attention_bias=False,
            sliding_window=None, vocab_size=vocab))
    model._wrapper = SimpleNamespace(_rpu_checkpoint_identity={"profile": "0.6b"}, model=SimpleNamespace(
        talker=SimpleNamespace(model=core(28, 3072),
            code_predictor=SimpleNamespace(model=core(5, 2048)))))

    prepared = []
    monkeypatch.setattr(graph.GraphRuntimePolicy, "prepare_arenas",
                        lambda policy: prepared.append(policy.graph_arena_count) or True)
    def fail_graph(*, runtime_policy):
        assert prepared == [5] and runtime_policy.graph_arena_count == 5
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
def test_predictor_reuses_kv_with_fresh_positions_inputs_and_owned_outputs(monkeypatch, head_only, hidden_states):
    from contextlib import nullcontext
    from rpu_backend.api import cache as cache_module

    allocations, positions = [], []
    class Cache:
        def __init__(self, *args, attn_tp=8):
            self.attn_tp, self.physical_kv_cores, self.batch_size = attn_tp, 8, 1
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
    output = torch.empty(1, 1, 1024, dtype=torch.float16)
    def forward(handle, hidden, k, v, mask, position, causal, **kwargs):
        positions.append(position)
        return output.copy_(hidden)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", forward, raising=False)
    part = dict(layers=5, capacity=32, name="predictor", hidden_size=1024, kv_heads=8, head_dim=128, handle=1, descriptor=[],
                signature=None, graph=SimpleNamespace(capture=lambda sig: nullcontext()), calls=0, tokens=0,
                device_head_only=head_only)
    borrowed = head_only and not hidden_states
    if borrowed:
        monkeypatch.setattr(torch.Tensor, "cpu", lambda tensor: pytest.fail("Predictor hidden copied to CPU"))
    first = tts._decoder_forward(part, torch.ones(1, 2, 1024), None, None, None, None, hidden_states)
    if borrowed:
        assert first.last_hidden_state.shape == (1, 1, 1024)
        first_head = tts._take_head_input(part, first.last_hidden_state).clone()
    else:
        first_head = first.last_hidden_state
    second = tts._decoder_forward(part, torch.zeros(1, 1, 1024), None, None, None, None, hidden_states)
    assert len(allocations) == 1 and positions == [0, 1, 0]
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


@pytest.mark.parametrize("width", [1024, 2048])
def test_talker_prefill_padding_positions_shape_switch_and_owned_outputs(monkeypatch, width):
    from contextlib import nullcontext
    import rpu_backend.graph as graph_module
    from rpu_backend.api import cache as cache_module
    from rpu_backend.runtime import decoder

    monkeypatch.setattr(graph_module, "GraphSignature", lambda **kwargs: kwargs)

    class Cache:
        def __init__(self, *args, attn_tp=8):
            self.attn_tp, self.physical_kv_cores, self.batch_size = attn_tp, 8, 1
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
    buffer = torch.empty(1, 16, width, dtype=torch.float16)
    def forward(handle, hidden, k, v, mask, position, causal, **kwargs):
        calls.append((position, hidden.clone(), kwargs["planned_stage_descriptor"]))
        return buffer[:, :hidden.shape[1]].copy_(hidden)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", forward, raising=False)
    part = dict(layers=28, capacity=128, hidden_size=width, kv_heads=8, head_dim=128, generation_limit=64, name="talker", handle=1,
                descriptor=(1,), signature=None, graph=graph, planner=object(), calls=0, tokens=0)
    first = tts._decoder_forward(part, torch.ones(1, 3, width), None, None, None, None, True)
    cache = first.past_key_values
    assert cache.position == 3 and first.last_hidden_state.shape == (1, 3, width)
    assert calls[0][1][:, 3:].eq(0).all() and calls[0][2] == (7,)
    second = tts._decoder_forward(part, torch.full((1, 1, width), 9.), cache, None, None, None, False)
    assert cache.position == 4 and calls[1][0] == 3 and calls[1][2] == (1,)
    assert first.last_hidden_state.eq(1).all() and second.last_hidden_state.eq(9).all()
    for _ in range(2):
        third = tts._decoder_forward(part, torch.zeros(1, 5, width), None, None, None, None, False)
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


@pytest.mark.parametrize("precision,acc32", [(p, a) for p in ("fp16", "w8a16", "w4a16_g32", "nvfp4") for a in (False, True)])
@pytest.mark.parametrize("width", [1024, 2048])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_decoder_and_heads_bind_the_selected_precision_and_keep_scale_owners(
        monkeypatch, precision, acc32, width, cores):
    """Exercise the installer/head consumer boundary without allocating device memory."""
    from rpu_backend import graph
    from rpu_backend.api import _qwen3_audio_precision as precision_module
    from rpu_backend.api import causal_lm
    from rpu_backend.adapters import qwen3
    from rpu_backend.adapters.qwen3_tts import codec, codec_transformer

    class Core(torch.nn.Module):
        def __init__(self, count, hidden, vocab):
            super().__init__()
            self.config = SimpleNamespace(num_hidden_layers=count, hidden_size=hidden,
                intermediate_size=hidden * 3, num_attention_heads=16, num_key_value_heads=8,
                head_dim=128, rms_norm_eps=1e-6, hidden_act="silu", attention_bias=False,
                sliding_window=None, vocab_size=vocab)
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(layer.self_attn, name, torch.nn.Linear(2, 2, bias=False))
            layer.self_attn.q_norm = layer.self_attn.k_norm = torch.nn.LayerNorm(2)
            layer.input_layernorm = layer.post_attention_layernorm = torch.nn.LayerNorm(2)
            layer.mlp = torch.nn.Module()
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(layer.mlp, name, torch.nn.Linear(2, 2, bias=False))
            self.layers = torch.nn.ModuleList([layer] * count)
            self.norm = torch.nn.LayerNorm(2)
        def rotary_emb(self, hidden, positions):
            values = torch.zeros(*positions.shape, 128)
            return values.cos(), values.sin()

    talker_core, predictor_core = Core(28, width, 3072), Core(5, 1024, 2048)
    talker_head = torch.nn.Linear(width, 8, bias=False)
    predictor_heads = [torch.nn.Linear(1024, 8, bias=False) for _ in range(15)]
    predictor = SimpleNamespace(model=predictor_core, lm_head=predictor_heads, generate=lambda: None)
    talker = SimpleNamespace(model=talker_core, code_predictor=predictor,
        codec_head=talker_head, generate=lambda: None)
    model = Qwen3TTSModel()
    model._precision, model._linear_acc32 = precision, acc32
    model._num_cores = cores
    decode = lambda *args, **kwargs: "waveform"  # noqa: E731 - identity is what the test checks
    codec_decoder = SimpleNamespace(pre_transformer=None, forward=decode)
    model._wrapper = SimpleNamespace(_rpu_checkpoint_identity={"profile": "0.6b" if width == 1024 else "1.7b"}, model=SimpleNamespace(talker=talker,
        speech_tokenizer=SimpleNamespace(model=SimpleNamespace(decoder=codec_decoder))))
    sources = [parameter.detach().clone() for parameter in talker_core.parameters()]
    dtype = {"fp16": torch.float16, "w8a16": torch.int8, "w4a16_g32": torch.uint8,
             "nvfp4": torch.uint8}[precision]
    packed = []
    packing_cores = []
    def pack(weight, partition, selected, *, bounded_head=False, num_cores=8):
        packing_cores.append(num_cores)
        assert selected == ("fp16" if precision == "nvfp4" else precision) and partition in (0, 1)
        physical = torch.ones(2, 2, dtype=torch.float16 if selected == "fp16" else dtype)
        scale = None if selected == "fp16" else torch.ones(2, dtype=torch.float16)
        packed.append((physical, scale, bounded_head))
        return physical, scale
    def pack_nvfp4(weight, partition, *, num_cores=8):
        packing_cores.append(num_cores)
        assert precision == "nvfp4" and partition in (0, 1)
        physical = torch.ones(2, 2, dtype=torch.uint8)
        scale = torch.ones(2, 2, dtype=torch.uint8)
        packed.append((physical, scale, False))
        return physical, scale, torch.tensor(0.5, dtype=torch.float32)
    monkeypatch.setattr(precision_module, "pack_audio_linear", pack)
    monkeypatch.setattr(precision_module, "pack_audio_nvfp4", pack_nvfp4)
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda value, *args, **kwargs:
        value if args == ("rpu",) else original_to(value, *args, **kwargs))
    policy = SimpleNamespace(prepare_arenas=lambda: True)
    def graph_policy(**kwargs):
        assert kwargs == {"graph_arena_count": 5, "execution_core_count": cores}
        return policy
    monkeypatch.setattr(graph, "GraphRuntimePolicy", SimpleNamespace(from_environment=graph_policy))
    monkeypatch.setattr(graph, "GraphCache", lambda **kwargs: SimpleNamespace(
        clear=lambda: None, cache_invariant_ok=lambda: True, size=lambda: 0, snapshot=lambda: []))
    monkeypatch.setattr(graph, "GraphSignature", lambda **kwargs: kwargs)
    monkeypatch.setattr(qwen3, "lookup_causal_decoder", lambda *args: SimpleNamespace(max_kv_len=512, chunk=64))
    monkeypatch.setattr(codec, "install_codec_convolutions", lambda *args: None)
    monkeypatch.setattr(codec, "install_codec_activations", lambda *args: None)
    monkeypatch.setattr(codec_transformer, "install_codec_transformer", lambda *args: None)
    monkeypatch.setattr(causal_lm, "_release_live_instance", lambda owner: None)
    events, bound = [], {}
    handles = iter((10, 20))
    created = []
    def create(**kwargs):
        created.append(kwargs)
        return next(handles)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_create", create, raising=False)
    attention_cores = 8 if cores == 8 else 4
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_topology",
        lambda handle: (cores, attention_cores, cores, attention_cores, 8), raising=False)
    def bind(*args):
        bound[args[0]] = args
        events.append(("weights", args[0]))
    for name in ("causal_decoder_set_weights", "causal_decoder_set_weights_w8a16",
                 "causal_decoder_set_weights_nvfp4"):
        monkeypatch.setattr(torch.ops.rpu, name, bind, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_linear_acc32",
        lambda handle, selected: events.append(("acc", handle, selected)), raising=False)
    replay_handles = []
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_fast_replay", lambda handle, enabled:
        replay_handles.append(handle), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_chunk_envelope", lambda *args: None, raising=False)
    def descriptor(handle):
        assert events[-1] == ("acc", handle, acc32)
        return [handle]
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_resolve_decode_stage_descriptor", descriptor, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_destroy", lambda handle: None, raising=False)
    consumed = []
    def linear(hidden, weight, bias, partition, active_cores, selected_acc, scale, *, logical_out_features=-1):
        if precision == "w4a16_g32":
            assert logical_out_features == 8  # Tiny test owner retains its logical output width.
        else:
            assert logical_out_features == -1
        assert active_cores == attention_cores
        consumed.append((weight, scale, selected_acc))
        assert hidden.shape[-1] in (1024, 2048)
        return torch.zeros(hidden.shape[0], 8)
    monkeypatch.setattr(torch.ops.rpu, "linear_with_accumulation", linear, raising=False)
    model._closed = False
    try:
        model._install_rpu()
        assert created == [{"num_cores": cores, "vocab_size": vocab} for vocab in (3072, 2048)]
        assert packing_cores == ([attention_cores] * (4 * 28) + [cores] * (3 * 28)
                                + [attention_cores] * (4 * 5) + [cores] * (3 * 5)
                                + [attention_cores] * 16)
        assert len(packed) == 7 * (28 + 5) + 16
        assert sum(item[2] for item in packed) == 16
        for name, handle, count in (("talker", 10, 28), ("predictor", 20, 5)):
            part = model._components[name]
            groups, _, _, _, scales = part["storage"]
            assert groups[0][0].dtype == dtype
            assert all(len(group) == (0 if precision == "fp16" else count) for group in scales)
            if precision == "nvfp4":
                assert all(actual is retained for actual, retained in zip(bound[handle][-14:-7], scales))
                tables = part["tensor_scales"]
                assert len(tables) == 7
                for projection, (actual, retained) in enumerate(zip(bound[handle][-7:], tables)):
                    assert actual is retained and retained.dtype == torch.float32
                    assert retained.shape == ((count + 7) // 8 * 8,)
                    expected_table = torch.full((count,), 0.5)
                    if name == "predictor" and width == 1024 and projection in (5, 6):
                        expected_table[2] *= 1 / 8 if projection == 5 else 8
                    assert torch.equal(retained[:count], expected_table) and torch.isfinite(retained[count:]).all()
            elif precision != "fp16":
                assert all(actual is retained for actual, retained in zip(bound[handle][-7:], scales))
            assert part["resource"].keepalive is part
            assert part["mlp_up_divisors"] == ({2: 8} if name == "predictor" and width == 1024 else {})
            if precision != "nvfp4":
                for projection, group in enumerate((9, 10)):
                    actual = groups[group] if precision == "fp16" else scales[projection + 5]
                    for layer, payload in enumerate(actual):
                        expected = (1 / 8 if projection == 0 else 8) if (
                            name == "predictor" and width == 1024 and layer == 2) else 1
                        assert payload.eq(expected).all()
        assert replay_handles == ([] if precision == "nvfp4" else [20])
        for index, head in enumerate([talker_head, *predictor_heads]):
            hidden_width = width if index == 0 else 1024
            assert head(torch.zeros(1, 1, hidden_width)).shape == (1, 1, 8)
            weight, scale = model._head_weights[index]
            assert consumed[-1][0] is weight and consumed[-1][1] is scale
            assert consumed[-1][2] is acc32
            if precision == "nvfp4":
                assert weight.dtype == torch.float16 and scale is None
        for original, actual in zip(sources, talker_core.parameters()):
            assert torch.equal(original, actual)
        # The codec decode is wrapped for stage timing and still returns the real result.
        assert codec_decoder.forward is not decode and codec_decoder.forward() == "waveform"
        assert model._stages.milliseconds(total_ns=10 ** 9)["codec_decode"] > 0
        stats = model.execution_stats()
        assert stats["num_cores"] == cores
        assert stats["stage_ms"] is None  # no request has completed on this instance
        assert stats["components"]["talker"]["native_topology"] == (cores, attention_cores, cores, attention_cores, 8)
        assert stats["text_precision"] == precision and stats["text_linear_acc32"] is acc32
        assert stats["head_precision"] == ("fp16" if precision == "nvfp4" else precision)
        assert stats["components"]["talker"]["hidden_size"] == width
        assert stats["components"]["predictor"]["mlp_up_divisors"] == ({2: 8} if width == 1024 else {})
    finally:
        model.close()


@pytest.mark.parametrize("precision", ["fp16", "w8a16", "w4a16_g32", "nvfp4"])
def test_mlp_balance_scales_actual_payload_without_requantizing(precision):
    from rpu_backend.api._qwen3_audio_precision import pack_audio_linear, pack_audio_nvfp4

    source = torch.randn(512, 256, generator=torch.Generator().manual_seed(7)) * 0.04
    if precision == "nvfp4":
        weight, scale, tensor_scale = pack_audio_nvfp4(source, 1)
    else:
        weight, scale = pack_audio_linear(source, 1, precision)
        tensor_scale = None
    originals = [value.clone() if value is not None else None for value in (weight, scale, tensor_scale)]
    actual = tts._rescale_mlp_payload(weight, scale, tensor_scale, 1 / 8)
    index = 2 if precision == "nvfp4" else 0 if precision == "fp16" else 1
    for i, (before, after) in enumerate(zip((weight, scale, tensor_scale), actual)):
        if i == index:
            assert torch.equal(after, before / 8) and after.dtype == before.dtype
        else:
            assert after is before
        if before is not None:
            assert torch.equal(before, originals[i])


def test_mlp_balance_rejects_nonfinite_and_lost_quantization_scales():
    codes = torch.ones(2, dtype=torch.int8)
    with pytest.raises(ValueError, match="nonfinite"):
        tts._rescale_mlp_payload(codes, torch.tensor([65504.], dtype=torch.float16), None, 8)
    with pytest.raises(ValueError, match="underflowed"):
        tts._rescale_mlp_payload(codes, torch.tensor([2 ** -24], dtype=torch.float16), None, 1 / 8)
    with pytest.raises(ValueError, match="nonfinite"):
        tts._rescale_mlp_payload(torch.tensor([65504.], dtype=torch.float16), None, None, 8)


def _predictor_bank_part(handle=20, vocab=2048, hidden=1024):
    from rpu_backend.runtime.topology import DecoderTopology

    return dict(handle=handle, hidden_size=hidden, vocab_size=vocab, name="predictor",
                topology=DecoderTopology(8, 8, 8, 8), device_head_only=True,
                kv_heads=8, head_dim=128, calls=0, tokens=0, descriptor=[], signature=None)


def _bank_payload(count=15, vocab=2048, hidden=1024):
    return [(torch.ones(vocab, hidden, dtype=torch.int8) * (index + 1),
             torch.full((vocab,), index + 1, dtype=torch.float16)) for index in range(count)]


def test_head_bank_installs_the_predictor_heads_in_one_transaction(monkeypatch):
    installed, resolved = [], []
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_lm_head_bank",
        lambda handle, weights, scales: installed.append((handle, weights, scales)), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_resolve_decode_stage_descriptor",
        lambda handle: resolved.append(handle) or [handle, 4], raising=False)
    model = Qwen3TTSModel()
    model._num_cores = 8
    part = _predictor_bank_part()
    model._components = {"predictor": part}
    heads = _bank_payload()
    model._head_weights = [(torch.ones(8, 1024, dtype=torch.int8), torch.ones(8, dtype=torch.float16)), *heads]
    model._install_head_bank()
    assert part["head_bank"] is True and len(installed) == 1
    handle, weights, scales = installed[0]
    # The talker head stays eager; only the Predictor's 15 steps enter the bank,
    # in step order, so select_lm_head(i) is lm_head[i].
    assert handle == 20 and len(weights) == 15
    assert all(actual is expected[0] for actual, expected in zip(weights, heads))
    assert all(actual is expected[1] for actual, expected in zip(scales, heads))
    # The install invalidates model state, so the decode descriptor it captured
    # before the fused head existed must not be reused.
    assert resolved == [20] and part["descriptor"] == [20, 4]
    assert part["signature"].op_id == "qwen3_tts_predictor"


@pytest.mark.parametrize("reason", ["cores", "dtype", "unscaled", "vocab", "count"])
def test_head_bank_declines_and_leaves_the_eager_heads_in_place(monkeypatch, reason):
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_lm_head_bank",
        lambda *args: pytest.fail("declined head bank was installed"), raising=False)
    model = Qwen3TTSModel()
    model._num_cores = 4 if reason == "cores" else 8
    part = _predictor_bank_part()
    if reason == "cores":
        from rpu_backend.runtime.topology import DecoderTopology

        part["topology"] = DecoderTopology(4, 4, 4, 4)
    model._components = {"predictor": part}
    heads = _bank_payload(count=14 if reason == "count" else 15,
                          vocab=1024 if reason == "vocab" else 2048)
    if reason == "dtype":
        heads = [(weight.half(), scale) for weight, scale in heads]
    if reason == "unscaled":
        heads = [(weight, None) for weight, _ in heads]
    model._head_weights = [(torch.ones(8, 1024, dtype=torch.int8), torch.ones(8, dtype=torch.float16)), *heads]
    model._install_head_bank()
    assert "head_bank" not in part and part["descriptor"] == []


def test_fused_head_publishes_each_step_logits_for_the_selected_head(monkeypatch):
    from contextlib import nullcontext
    from rpu_backend.api import cache as cache_module

    class Cache:
        def __init__(self, *args, attn_tp=8):
            self.attn_tp, self.physical_kv_cores, self.batch_size = attn_tp, 8, 1
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
    part = dict(_predictor_bank_part(vocab=16), layers=5, capacity=32, head_bank=True,
                graph=SimpleNamespace(capture=lambda signature: nullcontext()))
    selected, forwards = [], []
    def forward(handle, hidden, k, v, mask, position, causal, **kwargs):
        # The fused head of the retained stream applies whichever bank entry the
        # step selected, so the logits a replay produces carry that index.
        forwards.append((handle, position, selected[-1]))
        return torch.full((1, hidden.shape[1], 16), float(selected[-1]), dtype=torch.float16)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", forward, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_select_lm_head",
        lambda handle, index: selected.append(index), raising=False)
    selected.append(0)
    result = tts._decoder_forward(part, torch.ones(1, 1, 1024), None, None, None, None, False)
    assert result.last_hidden_state.shape == (1, 1, 16)
    assert part["head_logits"] is result.last_hidden_state
    # A fused head replaces the borrowed hidden state; nothing may re-launch it.
    assert "head_hidden" not in part and "head_device" not in part
    assert tts._take_head_input(part, result.last_hidden_state) is None
    with pytest.raises(ValueError, match="returns logits, not a hidden state"):
        tts._decoder_forward(part, torch.ones(1, 1, 1024), None, None, None, None, True)
    part["head_logits"] = result.last_hidden_state
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward",
        lambda *args, **kwargs: torch.ones(1, 1, 1024, dtype=torch.float16), raising=False)
    with pytest.raises(RuntimeError, match="invalid TTS Predictor device hidden state"):
        tts._decoder_forward(part, torch.ones(1, 1, 1024), None, None, None, None, False)


def test_fused_head_bank_selects_this_step_before_its_decode(monkeypatch):
    install_reference_sampler(monkeypatch)
    selected = []
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_select_lm_head",
        lambda handle, index: selected.append((handle, index)), raising=False)

    class Predictor:
        generation_config = SimpleNamespace(eos_token_id=None, repetition_penalty=None)

        def __init__(self, fused):
            self.model = self
            self.fused, self.step = fused, 0
            self.small_to_mtp_projection = torch.nn.Identity()
            # In the installed route the head reads what the decode published;
            # here it reads the selection, which must already be this step's.
            self.lm_head = [self.head(index) for index in range(15)]

        def get_input_embeddings(self):
            return [lambda ids: torch.zeros(1, 1, 1024)] * 14

        def head(self, index):
            def forward(hidden):
                expected = [(20, step) for step in range(index + 1)] if self.fused else []
                assert selected == expected
                logits = torch.zeros(1, 1, 16)
                logits[0, 0, index] = 1000
                return logits
            return forward

        def __call__(self, **kwargs):
            # Selection must already have happened for this step, not the last.
            assert len(selected) == (self.step + 1 if self.fused else 0)
            self.step += 1
            return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"], past_key_values=None)

    part = {"head_bank": True, "handle": 20}
    actual = tts._predictor_generate(Predictor(True), torch.zeros(1, 2, 1024), part=part,
        input_width=1024, max_new_tokens=15, do_sample=True).sequences
    assert actual.tolist() == [list(range(15))]
    assert selected == [(20, index) for index in range(15)]
    # Without a bank the heads stay eager and nothing is selected.
    selected.clear()
    tts._predictor_generate(Predictor(False), torch.zeros(1, 2, 1024), part=None,
        input_width=1024, max_new_tokens=15, do_sample=True)
    assert selected == []


@pytest.mark.parametrize("dilation", [1, 3, 9])
def test_tap_major_weight_over_a_compact_slab_is_the_residual_convolution(dilation):
    """The layout the device operator consumes, checked against Conv1d itself.

    `conv1d_slab_with_accumulation` never sees a patch matrix: it reads tap t of
    window m from row `m + t*dilation` of the compact `[L, C]` activation and
    multiplies by weight column `t*C + c`. Both halves of that — the tap-major K
    axis and the row-offset gather — are asserted here against the frozen
    module's own output, because a channel-major weight has the SAME shape and
    would simply return wrong audio.
    """
    torch.manual_seed(7)
    module = torch.nn.Conv1d(16, 32, 7, dilation=dilation).eval()
    hidden = torch.randn(1, 16, 6 * dilation + 12)
    with torch.inference_mode():
        reference = module(hidden)

    tap_major = module.weight.detach().permute(0, 2, 1).flatten(1)
    compact = hidden[0].t().contiguous()
    count = compact.shape[0] - 6 * dilation
    windows = torch.stack([compact[m:m + 6 * dilation + 1:dilation].reshape(-1)
                           for m in range(count)])
    assert windows.shape == (count, 7 * 16)
    # Dilation 1: the gather is literally seven consecutive slab rows, which is
    # what `input_m_byte_step = C*2` makes the kernel's own addressing produce.
    if dilation == 1:
        assert torch.equal(windows[3], compact[3:10].reshape(-1))

    result = windows @ tap_major.t() + module.bias.detach()
    torch.testing.assert_close(result.t().unsqueeze(0), reference, rtol=1e-5, atol=2e-5)
    # The channel-major weight the host im2col path uses is NOT interchangeable.
    channel_major = module.weight.detach().flatten(1)
    assert not torch.allclose(windows @ channel_major.t(), windows @ tap_major.t())


def install_residual_reference_ops(monkeypatch, with_slab=True):
    """CPU stand-ins for the device operators, the swizzle and the RPU upload."""
    from rpu_backend.runtime import weights as weights_module

    def slab(compact, weight, bias, taps, dilation, num_cores):
        count = compact.shape[0] - (taps - 1) * dilation
        windows = compact.unfold(0, (taps - 1) * dilation + 1, 1)[..., ::dilation]
        windows = windows.permute(0, 2, 1).reshape(count, -1)   # tap-major k
        out = windows.float() @ weight.float().t()
        return (out if bias is None else out + bias.float()).half()

    def linear(patches, weight, bias, partition, num_cores, acc32):
        out = patches.float() @ weight.float().t()
        return (out if bias is None else out + bias.float()).half()

    calls = {"slab": [], "linear": []}

    def record(name, fn):
        def wrapper(*args):
            calls[name].append(tuple(a.shape for a in args if torch.is_tensor(a)))
            return fn(*args)
        return wrapper

    namespace = SimpleNamespace(linear_with_accumulation=record("linear", linear))
    if with_slab:
        namespace.conv1d_slab_with_accumulation = record("slab", slab)
    monkeypatch.setattr(torch.ops, "rpu", namespace)
    tensor_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs:
        self if args == ("rpu",) else tensor_to(self, *args, **kwargs))
    # The col swizzle is a device-layout permutation; keep the logical weight so
    # the stand-ins above can be ordinary matmuls, and hand back something whose
    # .to("rpu") yields that weight.
    prepared = []

    def swizzle(weight, partition, num_cores):
        prepared.append(weight)
        return SimpleNamespace(to=lambda device: weight)
    monkeypatch.setattr(weights_module, "transform_linear_weight", swizzle)
    return calls, prepared


def residual_profile():
    """The frozen twelve-layer residual stack, at the channel counts it uses."""
    return torch.nn.Sequential(*[
        torch.nn.Conv1d(channels, channels, 7, dilation=dilation)
        for channels in (768, 384, 192, 96) for dilation in (1, 3, 9)])


def test_dilation_one_residual_layers_take_the_compact_operator_route(monkeypatch):
    calls, prepared = install_residual_reference_ops(monkeypatch)
    torch.manual_seed(3)
    decoder = residual_profile()
    from rpu_backend.adapters.qwen3_tts.codec import install_rpu_residual_convolutions

    stats = {"backend": "rpu", "layers": 0, "calls": 0}
    restore = []
    install_rpu_residual_convolutions(decoder, restore, stats, num_cores=8)
    assert stats["layers"] == 12 and stats["slab_layers"] == 12 and len(restore) == 12

    # Every prepared weight is tap-major; the operator never sees channel-major.
    for module, weight in zip(decoder, prepared):
        channels = module.in_channels
        assert weight.shape == (channels + (-channels) % 128, 7 * channels)
        expected = module.weight.detach().permute(0, 2, 1).flatten(1).half()
        torch.testing.assert_close(weight[:channels], expected, rtol=0, atol=0)

    with torch.inference_mode():
        for module in decoder:
            # 12 windows, so every residue class of every dilation is non-empty.
            hidden = torch.randn(1, module.in_channels, 6 * module.dilation[0] + 12)
            result = module(hidden)
            assert result.shape == (1, module.out_channels, 12)
            assert result.dtype == torch.float32 and torch.isfinite(result).all()
    # One operator call per LAYER, whatever the dilation: the residue classes are
    # concatenated into a single slab rather than staged one call each.
    assert len(calls["slab"]) == 12 and calls["linear"] == []
    assert stats["calls"] == 12 and stats["slab_calls"] == 12
    # What the operator is handed: the activation itself, never a patch matrix.
    # d=1 stages all 18 rows; d=3 stages ceil(30/3)=10 rows per class and d=9
    # ceil(66/9)=8, so the slab is the activation plus at most d-1 padding rows.
    assert [shape[0] for shape, _ in calls["slab"]] == [18, 30, 72] * 4
    assert [shape[1] for shape, _ in calls["slab"]] == (
        [768] * 3 + [384] * 3 + [192] * 3 + [96] * 3)
    for module, (shape, _) in zip(decoder, calls["slab"]):
        dilation, rows = module.dilation[0], shape[0] // module.dilation[0]
        assert shape[0] % dilation == 0
        # Every class must fit its own windows plus the 6-row halo, which is what
        # keeps the six garbage windows at each class boundary unread.
        for offset in range(dilation):
            assert -(-(12 - offset) // dilation) + 6 <= rows
        # ...and the last class's last window must still be an emitted row.
        assert (dilation - 1) * rows + -(-(12 - dilation + 1) // dilation) <= shape[0] - 6


def test_residual_slab_skips_residue_classes_with_no_output_row(monkeypatch):
    """Fewer output rows than the dilation: the empty classes yield no row."""
    calls, _ = install_residual_reference_ops(monkeypatch)
    torch.manual_seed(17)
    decoder = residual_profile()
    from rpu_backend.adapters.qwen3_tts.codec import install_rpu_residual_convolutions

    module = decoder[2]                      # c768, dilation 9
    assert module.dilation[0] == 9
    hidden = torch.randn(1, 768, 6 * 9 + 4)  # only 4 windows for 9 classes
    with torch.inference_mode():
        expected = module(hidden)
        install_rpu_residual_convolutions(decoder, [], {"layers": 0, "calls": 0}, num_cores=8)
        torch.testing.assert_close(module(hidden), expected, rtol=2e-2, atol=2e-2)
    # Still one call: ceil(58/9) = 7 rows per class over 9 classes. Five classes
    # own no output row at all, and the scatter must leave them out rather than
    # read past the operator's 63 - 6 = 57 emitted rows.
    assert len(calls["slab"]) == 1 and calls["slab"][0][0] == (63, 768)


def test_residual_slab_route_matches_the_host_im2col_route(monkeypatch):
    """Both routes must compute the same convolution, layer for layer."""
    calls, _ = install_residual_reference_ops(monkeypatch)
    torch.manual_seed(11)
    decoder = residual_profile()
    from rpu_backend.adapters.qwen3_tts.codec import install_rpu_residual_convolutions

    inputs = [torch.randn(1, m.in_channels, 6 * m.dilation[0] + 9) for m in decoder]
    with torch.inference_mode():
        reference = [m(x) for m, x in zip(decoder, inputs)]
        stats = {"backend": "rpu", "layers": 0, "calls": 0}
        install_rpu_residual_convolutions(decoder, [], stats, num_cores=8)
        for module, x, expected in zip(decoder, inputs, reference):
            torch.testing.assert_close(module(x), expected, rtol=2e-2, atol=2e-2)
    assert len(calls["slab"]) == 12 and stats["slab_calls"] == 12


def test_residual_install_keeps_the_host_route_when_the_operator_is_absent(monkeypatch):
    """A .so without the operator must not be handed a tap-major weight."""
    calls, prepared = install_residual_reference_ops(monkeypatch, with_slab=False)
    torch.manual_seed(5)
    decoder = residual_profile()
    from rpu_backend.adapters.qwen3_tts.codec import install_rpu_residual_convolutions

    stats = {"backend": "rpu", "layers": 0, "calls": 0}
    install_rpu_residual_convolutions(decoder, [], stats, num_cores=8)
    assert stats["slab_layers"] == 0
    for module, weight in zip(decoder, prepared):
        torch.testing.assert_close(
            weight[:module.in_channels],
            module.weight.detach().flatten(1).half(), rtol=0, atol=0)
    with torch.inference_mode():
        hidden = torch.randn(1, 768, 12)
        assert decoder[0](hidden).shape == (1, 768, 6)
    assert calls["slab"] == [] and len(calls["linear"]) == 1 and stats["slab_calls"] == 0


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
