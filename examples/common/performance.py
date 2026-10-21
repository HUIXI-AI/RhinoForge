"""Format measured example performance without importing model dependencies.

The report remains authoritative: no requested token count, inferred device
time, or sum of separately timed components is substituted for a measurement.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import statistics


_VLA_TARGETS = {"pi05", "g05", "hy_vla", "lingbot", "lingbot2", "wall_oss",
                "wall_oss_sorting", "wall_oss_rtc", "gr00t", "rhinovla", "internvla_n1"}


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _counts(rows, name):
    values = [row.get(name) for row in rows]
    if not values or any(type(value) is not int or value < 0 for value in values):
        return None
    if min(values) == max(values):
        return str(values[0])
    return f"{min(values)}..{max(values)} (varies)"


def _identity(config, report):
    for target, label in (("qwen3_asr", "ASR"), ("qwen3_tts", "TTS"),
                          ("whisper_asr", "Whisper ASR")):
        if target in config:
            return label, False
    example = _mapping(config.get("example"))
    target = example.get("target") or report.get("model")
    if not target:
        target = next((name for name in _VLA_TARGETS if name in config), "inference")
    if str(target).startswith("wall_oss"):
        target = "wall_oss"
    inputs = _mapping(config.get("input"))
    if target == "halo":
        return "Action expert component (single step)", False
    if target == "dinov3" or inputs.get("component") == "vision":
        return "Vision component", False
    if target in _VLA_TARGETS:
        return "VLA E2E", False
    multimodal = target == "qwen3_5_vision" or (
        target == "qwen3_vl" and inputs.get("component", "multimodal") != "text")
    return ("VLM" if multimodal else "LLM"), multimodal


def _latency(item, *, median=False):
    value, statistic = item.get("value"), item.get("statistic", "measured")
    samples = item.get("samples")
    if median and isinstance(samples, (list, tuple)) and samples and all(map(_number, samples)):
        value, statistic = statistics.median(samples), "median"
    if not _number(value):
        return None
    text = f"{statistic}={value:.3f} ms"
    count = item.get("sample_count")
    if type(count) is int and count > 0:
        text += f"; n={count}"
    return text


def summary(config, report) -> list[str]:
    """Return untagged performance lines for a completed measured invocation.

    E2E display defaults to the median of the existing formal samples. Token
    throughput keeps the report's mean aggregate or median-per-run convention.
    No report/config fields, timing boundaries, or persisted statistics change.
    """
    report = _mapping(report)
    if _mapping(report.get("execution")).get("status") != "COMPLETED":
        return []
    config = _mapping(config if config is not None else report.get("config"))
    label, multimodal = _identity(config, report)
    timings = [item for item in report.get("timing", ()) if isinstance(item, Mapping)]
    by_metric = {item.get("metric"): item for item in timings}
    lines = []
    if label in {"ASR", "TTS", "Whisper ASR"} and _number(report.get("median_rtf")):
        lines.append(f"{label} RTF median={report['median_rtf']:.3f}; "
                     "RTF = request time / audio duration")
    total = by_metric.get("total_latency")
    if total and total.get("unit") == "ms":
        measured = _latency(total, median=True)
        if measured:
            name = label if "component" in label or label == "VLA E2E" else f"{label} inference call"
            lines.append(f"{name}: {measured}; load/warmup excluded; boundary={total.get('boundary', 'not recorded')}")

    rows = [row for row in report.get("generation_runs", ()) if isinstance(row, Mapping)]
    generation = _mapping(report.get("generation_summary"))
    if rows:
        counts = [(name, _counts(rows, key)) for name, key in (
            ("prompt_tokens", "prefill_tokens"), ("decode_calls", "decode_calls"),
            ("output_tokens", "output_tokens"))]
        actual = "; ".join(f"{name}={value}" for name, value in counts if value is not None)
        if actual:
            lines.append(f"{label} actual per run: {actual}; measured_runs={len(rows)}")
    for phase in ("prefill", "decode") if generation or rows else ():
        rate = by_metric.get(phase + "_throughput", {})
        value = rate.get("value", generation.get(phase + "_tokens_per_second"))
        statistic = rate.get("statistic")
        if statistic is None:
            selected = generation.get("summary_statistic", _mapping(config.get("run")).get("summary_statistic", "mean"))
            statistic = "median_per_run_rate" if selected == "median" else "aggregate_tokens_over_total_time"
        parts = []
        if _number(value):
            parts.append(f"{value:.3f} tok/s ({statistic})")
        elif phase == "decode" and rows and _counts(rows, "decode_calls") == "0":
            parts.append("tok/s=n/a (no decode calls)")
        latency = by_metric.get(phase + "_latency", {})
        if latency.get("unit") == "ms":
            measured = _latency(latency)
            if measured:
                parts.append(measured)
        else:
            for stat in ("mean", "median"):
                value = generation.get(f"{phase}_ms_{stat}")
                if _number(value):
                    parts.append(f"{stat}={value:.3f} ms")
                    break
        if parts:
            boundary = rate.get("boundary") or latency.get("boundary")
            if not boundary and rows:
                boundaries = {row.get("timing_scope") for row in rows if row.get("timing_scope")}
                if len(boundaries) == 1:
                    boundary = next(iter(boundaries))
            if boundary:
                parts.append(f"boundary={boundary}")
            if multimodal and phase == "prefill" and "vision_latency" not in by_metric:
                parts.append("multimodal prefill; Vision not separately timed")
            lines.append(f"{label} {phase}: " + "; ".join(parts))

    for item in timings:
        metric = item.get("metric", "")
        if (metric == "total_latency" or item.get("unit") != "ms"
                or not isinstance(metric, str) or not metric.endswith("_latency")
                or (rows or generation) and metric in {"prefill_latency", "decode_latency"}):
            continue
        measured = _latency(item)
        if measured:
            component = metric.removesuffix("_latency").replace("_", " ")
            lines.append(f"{component} component: {measured}; boundary={item.get('boundary', 'not recorded')}")
    return lines
