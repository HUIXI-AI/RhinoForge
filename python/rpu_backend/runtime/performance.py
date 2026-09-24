"""Shared inference profiling and HWPerf reports (no backend initialization)."""
from __future__ import annotations

import contextlib
import json
import math
import os
import re
from typing import Any

def _profile_ctx(enabled: bool) -> Any:
    if not enabled:
        return contextlib.nullcontext()
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if hasattr(torch.profiler.ProfilerActivity, "PrivateUse1"):
        activities.append(torch.profiler.ProfilerActivity.PrivateUse1)
    return torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
    )


_HWPERF_NAME_SUFFIX_RE = re.compile(r"_\d+$")


def _hwperf_aggregate_file(path: str) -> dict:
    """Parse one HW perf trace JSON and aggregate Compute-event durations.

    Returns a dict matching the requested print format:
      {
        "file": <path>,
        "latency_ms": <max_ts_us - min_ts_us> / 1000,
        "rows": [{"name", "count", "total_us", "avg_us", "pct"}, ...],
        "error": str | None,
      }

    DMA events are deliberately excluded from the kernel table — the
    requested user-facing format lists compute kernels only, matching the
    `parallel_linear_w8a16 / parallel_linear_wint4a16_pgrp / ...` example.
    """
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {"file": path, "latency_ms": 0.0, "rows": [],
                "error": f"{type(exc).__name__}: {exc}"}
    events = doc.get("traceEvents", []) or []
    ts_min = None
    ts_max = None
    per_kernel: dict[str, list[float]] = {}
    for e in events:
        ph = e.get("ph")
        ts = e.get("ts")
        if ts is None:
            continue
        if ph == "B":
            ts_min = ts if ts_min is None else min(ts_min, ts)
        elif ph == "E":
            ts_max = ts if ts_max is None else max(ts_max, ts)
            if e.get("cat") == "Compute":
                dur = e.get("args", {}).get("duration_us")
                if dur is None:
                    continue
                base = _HWPERF_NAME_SUFFIX_RE.sub("", e.get("name", ""))
                per_kernel.setdefault(base, []).append(float(dur))
    latency_ms = 0.0 if (ts_min is None or ts_max is None) \
        else (ts_max - ts_min) / 1000.0
    total_us = sum(sum(v) for v in per_kernel.values())
    rows = []
    for name, durs in per_kernel.items():
        s = sum(durs)
        rows.append({
            "name": name,
            "count": len(durs),
            "total_us": s,
            "avg_us": s / len(durs) if durs else 0.0,
            "pct": (s / total_us * 100.0) if total_us > 0 else 0.0,
        })
    rows.sort(key=lambda r: r["total_us"], reverse=True)
    return {"file": path, "latency_ms": latency_ms, "rows": rows, "error": None}


def _hwperf_collect_files(directory: str) -> list[str]:
    """Glob and sort by filename so build/replay/seq ordering is stable."""
    if not directory or not os.path.isdir(directory):
        return []
    out = []
    for p in os.listdir(directory):
        if p.startswith("rpu_hwperf_") and p.endswith(".json"):
            out.append(os.path.join(directory, p))
    out.sort()
    return out


def _hwperf_device_sample_ns(doc: dict) -> int:
    """Validate one provenance-aware writer segment for strict cost ingestion.

    The required writer contract validates raw work/completion identity and
    unwraps a completed capture shorter than one counter period. This function
    checks the exported segment, not the writer binary or a whole invocation:
    callers must independently bind build provenance, planned Compute/DMA
    counts, complete Graph/segment coverage and correctness gates. The marker
    alone cannot certify costs. Ordinary report aggregation deliberately remains
    permissive so historical traces can still be inspected.
    """
    if not isinstance(doc, dict) or not isinstance(doc.get("otherData"), dict):
        raise ValueError("HWPerf sample requires otherData metadata")
    metadata = doc["otherData"]
    if metadata.get("source") != "rhino-launch-kernel HW perf trace":
        raise ValueError("HWPerf sample requires the launch-kernel writer source")
    if metadata.get("record_validation") != "work-completion-v1":
        raise ValueError("HWPerf cost sample requires work-completion-v1 record_validation")

    def metadata_integer(field, maximum=(1 << 63) - 1, minimum=1):
        value = metadata.get(field)
        if (not isinstance(value, str) or
                re.fullmatch(r"0|[1-9][0-9]{0,18}", value) is None or
                not minimum <= int(value) <= maximum):
            raise ValueError(f"HWPerf {field} must be a decimal string in {minimum}..{maximum}")
        return int(value)

    frequency = metadata_integer("frequency_hz")
    metadata_integer("base_cycle", (1 << 32) - 1, minimum=0)
    batch_events = metadata_integer("batch_events")
    records_captured = metadata_integer("records_captured")
    max_time_us = ((1 << 32) - 1) * 1_000_000 / frequency

    def time_us(value):
        if (type(value) not in (int, float) or
                not 0 <= value <= max_time_us + 0.000500001):
            raise ValueError("HWPerf time must be a finite nonnegative cycle-range number")
        return value

    events = doc.get("traceEvents")
    if not isinstance(events, list) or not events or len(events) % 2:
        raise ValueError("HWPerf sample requires non-empty paired traceEvents")
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("HWPerf events must be objects")
        if (type(event.get("pid")) is not int or event["pid"] != 0 or
                not isinstance(event.get("tid"), str) or
                re.fullmatch(r"stream_(?:[0-9]|1[0-5])", event["tid"]) is None or
                not isinstance(event.get("name"), str) or not event["name"]):
            raise ValueError("HWPerf event requires writer-format process/stream/name")
        time_us(event.get("ts"))

    starts, ends = set(), set()
    work_count = flow_count = 0
    for begin, end in zip(events[::2], events[1::2]):
        # dump_hw_perf_chrome emits adjacent B/E pairs, followed by s/f pairs.
        category = begin.get("cat")
        if category == "SYNC":
            identity = str(flow_count)
            if (begin.get("ph") != "s" or end.get("ph") != "f" or
                    end.get("cat") != "SYNC" or end.get("bp") != "e" or
                    begin.get("id") != identity or end.get("id") != identity or
                    begin["name"] != f"barrier_{identity}" or end["name"] != begin["name"] or
                    (begin["tid"], begin["ts"]) not in ends or
                    (end["tid"], end["ts"]) not in starts or end["ts"] < begin["ts"]):
                raise ValueError("HWPerf barrier flow must pair existing work endpoints")
            flow_count += 1
            continue
        if (flow_count or category not in ("Compute", "DMA") or
                begin.get("ph") != "B" or end.get("ph") != "E" or
                any(end.get(field) != begin[field] for field in ("cat", "name", "tid"))):
            raise ValueError("HWPerf Compute/DMA requires matching adjacent B/E pairs")
        args, end_args = begin.get("args"), end.get("args")
        if not isinstance(args, dict) or not isinstance(end_args, dict):
            raise ValueError("HWPerf work requires input and duration metadata")
        # build_batch increments perf_batch_idx only for Compute/DMA, not barriers.
        if (type(args.get("batch_index")) is not int or args["batch_index"] != work_count or
                type(args.get("stream")) is not int or begin["tid"] != f"stream_{args['stream']}"):
            raise ValueError("HWPerf work requires consecutive batch indices and matching stream")
        if category == "Compute":
            cores = args.get("cores")
            if (not isinstance(cores, list) or
                    any(type(core) is not int or not 0 <= core < 8 for core in cores) or
                    len(set(cores)) != len(cores)):
                raise ValueError("HWPerf Compute requires valid core metadata")
        else:
            if (type(args.get("channel")) is not int or not 0 <= args["channel"] < 8 or
                    args["stream"] != args["channel"] * 2 or
                    begin["name"] != f"dma_{work_count}_ch{args['channel']}" or
                    type(args.get("size_bytes")) is not int or
                    not 0 < args["size_bytes"] < (1 << 32) or args["size_bytes"] % 16 or
                    args.get("direction") not in ("DDR->SPM", "SPM->DDR", "DDR->DDR", "SPM->SPM")):
                raise ValueError("HWPerf DMA requires valid channel/size/direction metadata")
        duration = time_us(end_args.get("duration_us"))
        cycles = end_args.get("duration_cycles")
        if (type(cycles) is not int or not 0 < cycles < (1 << 32) or duration <= 0 or
                end["ts"] <= begin["ts"] or
                not math.isclose(duration, cycles * 1_000_000 / frequency,
                                 rel_tol=0, abs_tol=0.000500001) or
                not math.isclose(end["ts"] - begin["ts"], duration,
                                 rel_tol=0, abs_tol=0.001500001)):
            raise ValueError("HWPerf duration must match positive B/E elapsed and frequency-scaled cycles")
        starts.add((begin["tid"], begin["ts"]))
        ends.add((end["tid"], end["ts"]))
        work_count += 1
    # The validated writer counts actual records, including the completion pair.
    # Requiring the exact count rejects missing exported work and clipped traces.
    if (not work_count or records_captured != 2 * (work_count + 1) or
            batch_events < work_count + flow_count or min(ts for _, ts in starts) != 0):
        raise ValueError("HWPerf metadata disagrees with complete zero-based work capture")
    duration = round(max(ts for _, ts in ends) * 1000)
    if not 1 <= duration < (1 << 63):
        raise ValueError("HWPerf sample must contain positive finite device work")
    return duration


def _hwperf_format_block(agg: dict) -> str:
    """Render one aggregated JSON file's table in the user-specified layout.

    Show the file name and latency, followed by per-kernel count, total time,
    average time and percentage columns. Values come from the supplied record.
    """
    lines = [f"File: {agg['file']}",
             f"Latency: {agg['latency_ms']:.6f} ms"]
    if agg.get("error"):
        lines.append(f"(error: {agg['error']})")
        return "\n".join(lines)
    # Column widths chosen so a typical 40-char kernel name (e.g.
    # "llm_fp16_32b_prefill_flash_attn_univ_dp") doesn't wrap.
    header = (f"{'Kernel':<52}{'Cnt':>5}{'Total(us)':>15}"
              f"{'Avg(us)':>11}{'Pct':>8}")
    lines.append(header)
    lines.append("-" * len(header))
    for row in agg["rows"]:
        lines.append(
            f"{row['name']:<52}"
            f"{row['count']:>5d}"
            f"{row['total_us']:>15,.1f}"
            f"{row['avg_us']:>11.1f}"
            f"{row['pct']:>7.1f}%"
        )
    return "\n".join(lines)
