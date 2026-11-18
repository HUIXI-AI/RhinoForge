"""The speech stage timer shared by the ASR and TTS request paths.

The breakdown reaches the performance pages through
``execution_stats()["stage_ms"]`` -> the runner's ``stage_ms_runs`` ->
``scripts/perf/collect_reports.py``, so a stage that silently disappears or
double-counts nested time changes published numbers. The clock is replaced so
the arithmetic is exact rather than wall-clock flaky; no device is opened.

test_qwen3_tts.py owns the TTS-specific nesting and synthesize() checks. This
file covers what both pipelines share and the ASR stage set.
"""
from types import SimpleNamespace

import pytest

from rpu_backend.api import _qwen3_audio_stages as stages
from rpu_backend.api.qwen3_asr import _STAGE_NAMES as ASR_STAGES
from rpu_backend.api.qwen3_tts import _STAGE_NAMES as TTS_STAGES


@pytest.fixture
def spend(monkeypatch):
    """Advance the shared timer's clock by a whole number of milliseconds."""
    now = [0]
    monkeypatch.setattr(stages, "time", SimpleNamespace(perf_counter_ns=lambda: now[0]))
    return lambda ms: now.__setitem__(0, now[0] + int(ms * 1e6))


@pytest.mark.parametrize("names", [ASR_STAGES, TTS_STAGES], ids=["asr", "tts"])
def test_reports_every_declared_stage_in_pipeline_order(names, spend):
    timer = stages.StageTimer("speech", names)
    with timer.stage(names[0]):
        spend(10)
    out = timer.milliseconds(total_ns=12_000_000)
    assert list(out) == [*names, "other"]
    # A stage that did not run stays visible as 0.0 rather than vanishing, so
    # the reported stage set does not change with the request.
    assert out == {names[0]: 10.0, **{name: 0.0 for name in names[1:]}, "other": 2.0}


@pytest.mark.parametrize("names", [ASR_STAGES, TTS_STAGES], ids=["asr", "tts"])
def test_stages_and_the_remainder_are_the_whole_request(names, spend):
    timer = stages.StageTimer("speech", names)
    for index, name in enumerate(names):
        with timer.stage(name):
            spend(index + 1)
        spend(1)  # glue between the stages
    out = timer.milliseconds(total_ns=(sum(range(1, len(names) + 1)) + len(names)) * 1_000_000)
    assert [out[name] for name in names] == [float(index + 1) for index in range(len(names))]
    assert out["other"] == float(len(names))
    assert sum(out.values()) == pytest.approx(sum(range(1, len(names) + 1)) + len(names))


def test_repeated_calls_of_one_stage_accumulate(spend):
    # ASR decode is one call per generated token; a transcript's cost has to
    # land in `decode` rather than in the unattributed remainder.
    timer = stages.StageTimer("ASR", ASR_STAGES)
    with timer.stage("prefill"):
        spend(8)
    for _ in range(4):
        with timer.stage("decode"):
            spend(3)
    out = timer.milliseconds(total_ns=21_000_000)
    assert out["prefill"] == 8.0 and out["decode"] == 12.0 and out["other"] == 1.0


def test_reset_drops_the_previous_request(spend):
    timer = stages.StageTimer("ASR", ASR_STAGES)
    with timer.stage("prefill"):
        spend(5)
    timer.reset()
    with timer.stage("decode"):
        spend(4)
    out = timer.milliseconds(total_ns=4_000_000)
    assert out["prefill"] == 0.0 and out["decode"] == 4.0 and out["other"] == 0.0


def test_an_unknown_stage_name_is_rejected_per_pipeline(spend):
    # A TTS stage name reaching the ASR timer is a wiring mistake; letting it
    # pass would hide that time inside `other`.
    timer = stages.StageTimer("ASR", ASR_STAGES)
    with timer.stage("talker"):
        spend(1)
    with pytest.raises(RuntimeError, match="ASR stage timer recorded unknown stages"):
        timer.milliseconds(total_ns=1_000_000)


def test_an_unclosed_stage_is_rejected(spend):
    timer = stages.StageTimer("TTS", TTS_STAGES)
    entered = timer.stage("talker")
    entered.__enter__()
    spend(3)
    with pytest.raises(RuntimeError, match="TTS stage timer has an unclosed stage"):
        timer.milliseconds(total_ns=3_000_000)


def test_the_two_pipelines_declare_different_stages():
    assert ASR_STAGES == ("feature_extract", "audio_encoder", "prefill", "decode")
    assert TTS_STAGES == ("speaker_embedding", "talker", "predictor", "codec_decode")
    # `other` is the computed remainder, never a declared stage.
    assert "other" not in set(ASR_STAGES) | set(TTS_STAGES)
