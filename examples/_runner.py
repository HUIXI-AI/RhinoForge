"""Shared measurement/output around public inference, not a certification harness."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time


ALL_TARGETS = ("qwen3", "llama", "qwen3_5", "qwen3_vl", "dinov3", "gemma4",
               "wall_oss", "hy_vla", "qwen3_5_vision", "lingbot2", "halo", "rhinovla", "pi05")


def _provenance(config):
    """Record the configuration and checkout used by this inference request."""
    root = Path(__file__).resolve().parents[1]

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
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"source": source, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "runtime_assets": "selected by the installed runtime environment"}


def _prepare(config):
    from _common import config_metadata
    target = config_metadata(config)["target"]
    if target in {"qwen3", "llama", "qwen3_5", "qwen3_vl", "dinov3", "gemma4"}:
        from _text_vision_runners import prepare_text_vision
        return prepare_text_vision(config)
    if target == "qwen3_5_vision":
        from _qwen35_vision import prepare_vision
        return prepare_vision(config)
    if target == "pi05":
        from _policy_examples import prepare_pi05
        return prepare_pi05(config)
    from _policy_runners import prepare_policy
    return prepare_policy(config)


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


def run(config):
    from _common import config_metadata

    example, options = config_metadata(config), config["run"]
    if (set(options) & {"timing_mode", "summary_statistic"}
            or set(config.get("input", {})) & {"image_size", "prefill_filler_prompt"}):
        from _common import validate_generation_options
        validate_generation_options(config)
    hw_profile = config.get("runner", {}).get("hw_profile", {})
    hw_enabled = options.get("hwperf", False) or hw_profile.get("enabled", False)
    for name, value in example.get("opt_in", {}).items():
        if name in os.environ and os.environ[name] != value:
            raise ValueError(f"inherited {name} conflicts with the exact profile opt-in")
        os.environ[name] = value
    threads = options.get("torch_num_threads", 8 if example["target"] == "pi05" else None)
    if threads is not None:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = str(threads)
    import torch
    if threads is not None:
        torch.set_num_threads(threads)
    import rpu_backend
    from rpu_backend.runtime.performance import _profile_ctx

    parent = Path(options.get("output_dir", "perf_results/examples")).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=example["profile_id"] + "-", dir=parent))
    seed = options.get("seed", 0)
    warmup = options.get("warmup", 2 if example["target"] == "pi05" else 1)
    runs = options.get("runs", 6 if example["target"] == "pi05" else 1)
    if hw_enabled:
        raise ValueError("hardware trace export is unavailable in the public runtime")
    report = {"runner": "public_example", "profile_id": example["profile_id"],
              "formal_certification": False, "correctness": "not_run",
              "config": config, "warmup": warmup, "runs": runs,
              "wall_ms_runs": [], "status": "RUNNING"}
    report["cpu_runtime"] = {
        "torch_version": torch.__version__, "torch_num_threads": torch.get_num_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "inference_mode": options.get("inference_mode", True),
    }
    owner = None
    retirement_attempted = False
    primary_error = None
    try:
        report["provenance"] = _provenance(config)
        grad_context = torch.inference_mode if options.get("inference_mode", True) else torch.no_grad
        with grad_context():
            torch.manual_seed(seed)
            infer, owner = _prepare(config)
            # RPU PASSTHROUGH and Graph BUILD/REPLAY/one-shot launches wait for
            # completion before returning (enqueu_batch(wait_finish=true)).
            # The actual torch.rpu namespace has no CUDA-style synchronize().
            for _ in range(warmup):
                torch.manual_seed(seed)
                if "warmup_decode_steps" in options:
                    infer(decode_steps=options["warmup_decode_steps"])
                else:
                    infer()
            output = None
            for _ in range(runs):
                torch.manual_seed(seed)
                start = time.perf_counter()
                value = infer()
                report["wall_ms_runs"].append((time.perf_counter() - start) * 1000)
                output = _cpu_output(value)
                if isinstance(output, dict) and "generation" in output:
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
        text = [f"# {example['profile_id']}", "", "Inference only; correctness/certification not run.", "",
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
            (directory / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"inference PASS: median={report['median_ms']:.3f} ms; artifacts={directory}")
    if isinstance(output, dict) and "text" in output:
        print(output["text"])
    return 0
