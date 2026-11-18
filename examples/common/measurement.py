"""Shared measurement/output around public inference, not a certification harness."""
from __future__ import annotations

import contextlib
from dataclasses import fields, is_dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time


from examples.common import results
_RESULTS = vars(results)

def _materialize_geometry(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _materialize_geometry(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_materialize_geometry(v) for v in value]
    return value


ALL_TARGETS = ("qwen3", "llama", "qwen3_5", "qwen3_vl", "dinov3", "gemma4",
               "wall_oss", "gr00t", "hy_vla", "lingbot",
               "internvla_n1", "qwen3_5_vision", "lingbot2", "wall_oss_rtc",
               "halo", "g05", "rhinovla")


def _provenance(config):
    """Record checkout/config and declared dependencies, not hardware qualification."""
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=True)
        return result.stdout

    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        delta = git("diff", "--binary", "HEAD")
        untracked = git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        patch_digest = hashlib.sha256(delta)
        for raw in sorted(item for item in untracked if item):
            path = root / os.fsdecode(raw)
            patch_digest.update(raw + b"\0")
            patch_digest.update(hashlib.sha256(path.read_bytes()).digest())
        source = {"commit": commit, "dirty": bool(delta or any(untracked)),
                  "worktree_delta_sha256": patch_digest.hexdigest()}
    except (OSError, subprocess.CalledProcessError) as exc:
        source = {"commit": None, "status": "UNAVAILABLE", "reason": type(exc).__name__}
    manifest_path = root / "release/runtime-dependencies.json"
    if manifest_path.is_file():
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        dependencies = {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                        "scope": "declared_build_inputs",
                        "launch_kernel": manifest["launch_kernel"]["commit"],
                        "rhino_ops": {name: manifest["artifacts"][role]["commit"]
                                      for name, role in (("main", "rhino_ops_main"),
                                                         ("expansion", "rhino_ops_expansion"))}}
    else:
        dependencies = {"status": "UNAVAILABLE"}
    from examples.common.overrides import config_sha256
    return {"source": source, "config_sha256": config_sha256(config),
            "dependencies": dependencies}


def _prepare(config):
    from examples.common.preparation import prepare
    return prepare(config)


def _cpu_output(value):
    """Materialize caller-owned output after timing, before the next forward."""
    import torch
    from collections.abc import Mapping

    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().clone()
        if not torch.isfinite(result).all():
            raise ValueError("inference output contains NaN/Inf")
        return result
    if isinstance(value, Mapping):
        return {key: _cpu_output(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_cpu_output(item) for item in value]
    if is_dataclass(value):
        return {field.name: _cpu_output(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("inference output contains NaN/Inf")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported inference output type: {type(value).__name__}")


def _close(owner, target=None):
    if owner is None:
        return
    for name in ("close", "destroy", "close_rpu_execution"):
        method = getattr(owner, name, None)
        if callable(method):
            method()
            return
    if target in {"qwen3_5", "qwen3_5_vision"}:
        if getattr(owner, "_qwen3_5_moe_profile", None) is not None:
            from rpu_backend.adapters.qwen3_5_moe import (
                _close_qwen3_5_moe_model,
            )
            _close_qwen3_5_moe_model(owner)
        else:
            from rpu_backend.adapters.qwen3_5 import _close_qwen3_5_model
            _close_qwen3_5_model(owner)
    elif target == "qwen3_vl" or getattr(owner, "_qwen3_vl_retirement_owner", None) is not None:
        from rpu_backend.adapters.qwen3_vl import _is_retirement_owner
        parent = getattr(owner, "_qwen3_vl_retirement_owner", None)
        if not _is_retirement_owner(parent, owner):
            raise RuntimeError("Qwen3-VL teardown lost its actual composite retirement owner")
        parent.close()
    elif target == "dinov3":
        from rpu_backend.adapters.dinov3 import _close_dinov3_model
        _close_dinov3_model(owner)
    elif target in {"qwen3", "llama", "gemma4"}:
        from rpu_backend.runtime.decoder import _close_causal_lm_model
        _close_causal_lm_model(owner)
    else:
        raise RuntimeError(f"{target}: inference owner has no explicit retirement entry")


def _generation_summary(samples, statistic="mean"):
    if statistic == "median":
        # Median of the per-run rates, as in the historical Dense VL report;
        # do not substitute a rate computed from mean time.
        decode_rates = [item["decode_tokens_per_second"] for item in samples]
        return {
            "summary_statistic": "median",
            "prefill_ms_median": statistics.median(item["prefill_ms"] for item in samples),
            "decode_ms_median": statistics.median(item["decode_ms"] for item in samples),
            "prefill_tokens_per_second": statistics.median(
                item["prefill_tokens_per_second"] for item in samples),
            "decode_tokens_per_second": (statistics.median(decode_rates)
                                         if all(rate is not None for rate in decode_rates) else None),
        }
    if statistic != "mean":
        raise ValueError(f"unknown generation summary_statistic: {statistic}")
    prefill_total = sum(item["prefill_ms"] for item in samples)
    decode_total = sum(item["decode_ms"] for item in samples)
    decode_calls = sum(item["decode_calls"] for item in samples)
    return {
        "prefill_ms_mean": prefill_total / len(samples),
        "decode_ms_mean": decode_total / len(samples),
        "prefill_tokens_per_second": sum(item["prefill_tokens"] for item in samples) * 1000 / prefill_total,
        "decode_tokens_per_second": decode_calls * 1000 / decode_total if decode_calls else None,
    }


def _result_timings(report, options):
    timings = _RESULTS["latency_timing"](report["wall_ms_runs"],
        "public inference call; caller-owned CPU output snapshot, load and warmup excluded")
    samples, summary = report.get("generation_runs", []), report.get("generation_summary")
    if summary is not None:
        statistic = options.get("summary_statistic", "mean")
        mode = options.get("timing_mode", "per_call")
        for phase in ("prefill", "decode"):
            field = phase + "_tokens_per_second"
            boundary = f"generation {phase}; timing_mode={mode}; see generation_runs for actual tokens/calls"
            timings += _RESULTS["latency_timing"]((row[phase + "_ms"] for row in samples),
                boundary, metric=phase + "_latency", statistic=statistic)
            timings.append({"metric": phase + "_throughput", "unit": "tokens/s", "boundary": boundary,
                "statistic": "aggregate_tokens_over_total_time" if statistic == "mean" else "median_per_run_rate",
                "value": summary[field], "samples": [row[field] for row in samples], "sample_count": len(samples)})
    timings += _RESULTS["stage_phase_timings"](report.get("stage_runs", []),
        "host wall time of the runner-declared stage calls inside each measured public inference call; "
        "glue between stages is not attributed")
    return timings


def run(config):
    from examples.common.config import cpu_runtime, example_profiles

    example, options = config["example"], config["run"]
    if (set(options) & {"timing_mode", "summary_statistic"}
            or set(config["input"]) & {"image_size", "prefill_filler_prompt"}):
        from examples.common.config import validate_generation_options
        validate_generation_options(config)
    hw_profile = config.get("runner", {}).get("hw_profile", {})
    hw_enabled = options.get("hwperf", False) or hw_profile.get("enabled", False)
    max_dumps = hw_profile.get("max_dumps", 32)
    runtime_env = {}
    if example["target"] == "gr00t":
        from examples.runners.gr00t.config import configure_environment
        runtime_env = configure_environment(config)
    for name, value in example.get("opt_in", {}).items():
        if name in os.environ and os.environ[name] != value:
            raise ValueError(f"inherited {name} conflicts with the exact profile opt-in")
        os.environ[name] = value
    cpu = cpu_runtime()
    effective_cpu = cpu.configure_cpu_threads(cpu.resolve_cpu_threads(
        config, default=8 if example["target"] == "g05" else None))
    import torch
    import rpu_backend
    from rpu_backend.runtime.performance import (
        _profile_ctx, _hwperf_aggregate_file, _hwperf_collect_files, _hwperf_format_block,
    )

    parent = Path(options.get("output_dir", "perf_results/examples")).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=example["profile_id"] + "-", dir=parent))
    hwperf_dir = directory / "hwperf"
    seed = options.get("seed", 0)
    warmup, runs = options.get("warmup", 2), options.get("runs", 1)
    profile = example_profiles()[example["profile_id"]]
    support = profile.get("support", {"status": "NOT_EVALUATED",
                                     "reason_code": "CALLER_FACTORY_TEMPLATE"})
    report = {"runner": "public_example", "profile_id": profile["id"],
              "support": support,
              "evaluation_only": profile.get("example_only", False) or support["status"] in {
                  "CONTROLLED_EVALUATION", "IMPLEMENTED_UNVERIFIED_HW", "EXPERIMENTAL"},
              "formal_certification": False, "correctness": "not_run",
              "config": config, "warmup": warmup, "runs": runs,
              "warmup_wall_ms_runs": [], "wall_ms_runs": [], "status": "RUNNING"}
    report["cpu_runtime"] = dict(effective_cpu,
                                inference_mode=options.get("inference_mode", True))
    if runtime_env:
        report["runtime_env"] = runtime_env
    owner = None
    retirement_attempted = False
    primary_error = None
    try:
        report["provenance"] = _provenance(config)
        # Enable before model setup/Graph BUILD. Toggling after BUILD invalidates graphs.
        hw_context = (torch.rpu.hw_perf_trace(str(hwperf_dir), max_dumps=max_dumps)
                      if hw_enabled else contextlib.nullcontext())
        grad_context = torch.inference_mode if options.get("inference_mode", True) else torch.no_grad
        with grad_context(), hw_context:
            torch.manual_seed(seed)
            # Keep persistent weights and mutable Graph inputs versioned.
            with torch.inference_mode(False), torch.no_grad():
                start = time.perf_counter()
                infer, owner = _prepare(config)
                # Model load and install through the runner's prepare(); a lazy
                # Graph BUILD falls in the first warmup call instead.
                report["load_seconds"] = time.perf_counter() - start
            input_geometry = vars(infer).get("input_geometry") if hasattr(infer, "__dict__") else None
            if isinstance(input_geometry, dict):
                report["input_geometry"] = _materialize_geometry(input_geometry)
            # device PASSTHROUGH and Graph BUILD/REPLAY/one-shot launches wait for
            # completion before returning (enqueu_batch(wait_finish=true)).
            # The actual torch.rpu namespace has no CUDA-style synchronize().
            # A lazy Graph needs both BUILD and an actual REPLAY before timing.
            # Keep warmup observations separate, including the first replay.
            for _ in range(warmup):
                torch.manual_seed(seed)
                start = time.perf_counter()
                if "warmup_decode_steps" in options:
                    value = infer(decode_steps=options["warmup_decode_steps"])
                else:
                    value = infer()
                report["warmup_wall_ms_runs"].append((time.perf_counter() - start) * 1000)
                del value
            output = None
            declared = vars(infer).get("stage_boundaries", ()) if hasattr(infer, "__dict__") else ()
            boundaries, missing = _RESULTS["available_stage_boundaries"](declared)
            if declared:
                report["stage_boundaries"] = [f"{stage}: {type(target).__name__}.{name}"
                                              for stage, target, name in boundaries]
                report["stage_boundaries_missing"] = missing
            if boundaries:
                report["stage_runs"] = []
            with _RESULTS["stage_timings"](boundaries) as phases:
                for _ in range(runs):
                    torch.manual_seed(seed)
                    phases.clear()
                    start = time.perf_counter()
                    value = infer()
                    report["wall_ms_runs"].append((time.perf_counter() - start) * 1000)
                    if boundaries:
                        report["stage_runs"].append(dict(phases))
                    output = _cpu_output(value)
                    # The caller-owned CPU snapshot survives. Retire the original
                    # output now, not at `value = infer()` inside the next timer.
                    del value
                    if isinstance(output, dict) and "generation" in output:
                        if "input_geometry" in output["generation"]:
                            output["generation"]["input_geometry"] = _materialize_geometry(
                                output["generation"]["input_geometry"])
                        report.setdefault("generation_runs", []).append(output["generation"])
            torch.save(output, directory / "output.pt")
            if options.get("profile", False):
                # Separate diagnostic invocation: profiler overhead never enters timings.
                torch.manual_seed(seed)
                with _profile_ctx(True) as profiler:
                    infer()
                profiler.export_chrome_trace(str(directory / "profile.trace.json"))
                (directory / "profile.txt").write_text(
                    profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=100),
                    encoding="utf-8")
        report["hwperf"] = [_hwperf_aggregate_file(path)
                            for path in _hwperf_collect_files(str(hwperf_dir))]
        if hw_enabled and (
            not report["hwperf"] or any(item["error"] for item in report["hwperf"])
        ):
            raise RuntimeError("HWPerf was requested but its traces are missing or unreadable")
        report["hwperf_max_dumps"] = max_dumps if hw_enabled else 0
        values = report["wall_ms_runs"]
        report.update(avg_ms=statistics.mean(values), median_ms=statistics.median(values),
                      min_ms=min(values), max_ms=max(values))
        if "generation_runs" in report:
            report["generation_summary"] = _generation_summary(
                report["generation_runs"], options.get("summary_statistic", "mean"))
        # Never retry an uncertain native retirement, including from finally.
        retirement_attempted = True
        try:
            _close(owner, example["target"])
        except BaseException as exc:
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            raise
        text = [f"# {profile['id']}", "", "Inference only; correctness/certification not run.", "",
                f"Warmup calls (ms; excluded): {report['warmup_wall_ms_runs']}", "",
                f"Median: {report['median_ms']:.3f} ms; samples: {values}", ""]
        if "generation_runs" in report:
            text += [f"Generation summary statistic: {options.get('summary_statistic', 'mean')}; "
                     f"timing mode: {options.get('timing_mode', 'per_call')}.", ""]
            text += ["| Run | P tokens | D calls | Prefill token/s | Decode token/s |",
                     "|---|---:|---:|---:|---:|"]
            for index, sample in enumerate(report["generation_runs"]):
                decode_rate = sample["decode_tokens_per_second"]
                decode_text = "n/a" if decode_rate is None else f"{decode_rate:.3f}"
                text.append(f"| {index + 1} | {sample['prefill_tokens']} | {sample['decode_calls']} | "
                            f"{sample['prefill_tokens_per_second']:.3f} | "
                            f"{decode_text} |")
            text.append("")
        text.extend(_hwperf_format_block(item) for item in report["hwperf"])
        (directory / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
        report["status"] = "PASS"
    except BaseException as exc:
        primary_error = exc
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if not retirement_attempted:
                _close(owner, example["target"])
        except BaseException as exc:
            report.update(status="FAIL", cleanup_error=f"{type(exc).__name__}: {exc}")
            if primary_error is None:
                raise
            primary_error.add_note(f"model cleanup failed: {exc!r}")
        finally:
            _RESULTS["add_result_fields"](report, directory, completed=report["status"] == "PASS",
                timing=_result_timings(report, options),
                runtime_checks=("public inference and caller-owned output validation",),
                output_names=("output.pt", "report.md") +
                    (("profile.trace.json", "profile.txt") if options.get("profile", False) else ()))
            _RESULTS["write_report"](report, directory, allow_nan=False)
    if isinstance(output, dict) and "text" in output:
        logging.getLogger("examples.output").info("%s", output["text"])
    return 0
