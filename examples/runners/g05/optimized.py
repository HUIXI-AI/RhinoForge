#!/usr/bin/env python3
"""Shared G05 K1 precision/input examples, following the HYVLA report format.

Inference/performance only: this entry does not certify new precision profiles.
"""
from __future__ import annotations

from examples.common import cpu as _cpu_helpers
from examples.common import cold_libraries as _cold_libraries
from examples.common import results as _result_helpers

import copy
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

_RESULTS = vars(_result_helpers)

# Each spelling is a registered recipe selector plus its legacy default accumulation.
_SPELLINGS = {"fp16": ("fp16", "acc32"), "fp16_acc16": ("fp16", "acc16"),
              "w8a16_acc32": ("w8a16", "acc32"), "w8a16": ("w8a16", "acc16"),
              "w8a16_nvfp4": ("w8a16_nvfp4", "acc16"), "w8a16_nvfp4_acc32": ("w8a16_nvfp4", "acc32")}
_PRECISIONS = tuple(_SPELLINGS)
# Weight dtypes (backbone, Action) by the resolved per-module specs of the selector's recipe.
_ROUTE_PROBES = (("vision",), ("text",), ("text", "in_proj_a"), ("text", "in_proj_b"), ("action",))
_ROUTES = {("fp16",) * 5: ("fp16", None),
           ("w8a16", "w8a16", "fp16", "fp16", "w8a16"): ("w8a16", None),
           ("w8a16", "w8a16", "fp16", "fp16", "nvfp4"): ("w8a16", "nvfp4a16")}
PRECISION_ALIASES = {
    "fp16_acc32": "fp16", "w8a16_acc16": "w8a16",
    "w8a16_nvfp4_acc16": "w8a16_nvfp4",
    "w8a16_action_nvfp4": "w8a16_nvfp4",
    "w8a16_action_nvfp4_acc16": "w8a16_nvfp4",
    "w8a16_action_nvfp4_acc32": "w8a16_nvfp4_acc32",
}


_LOG = logging.getLogger(__name__)

def canonical_precision(precision):
    return PRECISION_ALIASES.get(precision, precision)


def library_helpers():
    # Configuration validation imports this module before backend initialization.
    return vars(_cold_libraries)


def cpu_helpers():
    return vars(_cpu_helpers)


def precision_options(precision, linear_accumulation=None, math_precision="base"):
    precision = canonical_precision(precision)
    if precision not in _PRECISIONS:
        raise ValueError(f"[g05].precision must be one of {_PRECISIONS}")
    from examples.common.config import owner_route
    selector, accumulation = _SPELLINGS[precision]
    weight, action = owner_route("G0.5", "g05", _ROUTE_PROBES, _ROUTES, selector=selector)
    result = {"math_precision": math_precision, "weight_dtype": weight, "linear_accumulation": accumulation}
    if math_precision not in ("base", "high"):
        raise ValueError("[g05].math_precision must be base or high")
    if action is not None:
        result["action_weight_dtype"] = action
    if linear_accumulation is not None:
        if linear_accumulation not in ("acc16", "acc32"):
            raise ValueError("[g05].linear_accumulation must be acc16 or acc32")
        result["linear_accumulation"] = linear_accumulation
    return result


def input_camera_names(inputs):
    canonical = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
    cameras = inputs.get("cameras", len(inputs.get("camera_names", canonical)))
    names = inputs.get("camera_names", canonical[:cameras] if type(cameras) is int else ())
    if (type(cameras) is not int or cameras not in (2, 3)
            or not isinstance(names, (list, tuple)) or tuple(names) != canonical[:cameras]):
        raise ValueError("[input].camera_names must be head_rgb/left_wrist_rgb with optional right_wrist_rgb, matching cameras=2 or 3")
    return tuple(names)


def core_budget(config):
    """The cold rpu_execution.model.num_cores (8 when omitted)."""
    model = config.get("rpu_execution", {}).get("model", {})
    return model.get("num_cores", 8) if isinstance(model, dict) else 8


def resolve_row_config(config, catalog_path=None):
    """Apply BASE schedules; HIGH may borrow them without a performance claim."""
    result = copy.deepcopy(config)
    options = precision_options(config["g05"]["precision"], config["g05"].get("linear_accumulation"),
                                config["g05"].get("math_precision", "base"))
    base = ("fp16" if options["weight_dtype"] == "fp16" else
            "w8a16_action_nvfp4" if options.get("action_weight_dtype") == "nvfp4a16" else "w8a16")
    key = {"cameras": len(input_camera_names(config["input"])),
           "precision": base + "_" + options["linear_accumulation"],
           "text_length": config["input"].get("text_tokens")}
    path = Path(catalog_path) if catalog_path is not None else (
        Path(__file__).resolve().parents[3] / "examples/configs/g05/selected_profiles.json")
    selection = {"key": key, "catalog": str(path), "matched": False,
                 "requested_math_precision": options["math_precision"],
                 "schedule_source_math_precision": "base",
                 "applied_parameters": {}, "libraries": "explicit process binding; never replaced by catalog"}
    candidate = None
    if core_budget(config) != 8:
        # Every catalog row was measured on eight cores; a reduced budget keeps
        # its own auto choices (one camera per Vision forward, planner chunk).
        selection["reason"] = (f"selected_profiles.json rows are eight-core measurements; "
                               f"model.num_cores={core_budget(config)} uses auto")
    elif path.is_file():
        catalog = json.loads(path.read_text())
        if catalog.get("schema_version") == 2:
            rows = [row for row in catalog.get("profiles", [])
                    if all(row.get(field) == value for field, value in key.items())
                    and tuple(row.get("camera_names", ())) == input_camera_names(config["input"])]
            if len(rows) > 1:
                raise ValueError("G05 best catalog contains duplicate camera/precision/text rows")
            if rows:
                candidate, = rows
                if options["math_precision"] == "base":
                    selection.update(matched=True, selected_build=candidate.get("selected_build"))
                else:
                    selection.update(borrowed_base_schedule=True,
                        reason="HIGH borrows the BASE schedule; this catalog contains no measured HIGH optimum")
    if not selection["matched"]:
        selection.setdefault("reason", "no measured catalog row for the exact camera/precision/text key")
    if config["g05"].get("vision_camera_batch", 1) == "auto":
        batch = candidate.get("vision_camera_batch", 1) if candidate else 1
        if type(batch) is not int or batch not in (1, 2, 3):
            raise ValueError("G05 best catalog camera batch must be 1, 2 or 3")
        result["g05"]["vision_camera_batch"] = batch
        selection["applied_parameters"]["vision_camera_batch"] = batch
    prefill = config.get("rpu_execution", {}).get("components", {}).get(
        "language_model", {}).get("prefill", {})
    if prefill.get("chunk_size", "auto") == "auto" and candidate:
        chunk = candidate.get("text_chunk", {}).get("requested", "auto")
        if chunk == 0:
            chunk = "auto"
        if chunk != "auto" and (type(chunk) is not int or chunk not in (64, 128, 192, 256)):
            raise ValueError("G05 best catalog requested text chunk must be auto/0 or 64/128/192/256")
        actual = result.setdefault("rpu_execution", {}).setdefault("components", {}).setdefault(
            "language_model", {}).setdefault("prefill", {})
        actual["chunk_size"] = chunk
        selection["applied_parameters"]["text_chunk"] = chunk
    validate_example_config(result)
    return result, selection


def validate_example_config(config):
    required = {"g05", "input", "run"}
    if not required <= set(config) or set(config) - required - {"rpu_execution", "libraries", "cpu", "logging"}:
        raise ValueError("G05 requires [g05], [input], [run], optional [cpu]/[rpu_execution]/[libraries]/[logging]")
    from examples.common.config import validate_logging_config
    validate_logging_config(config)
    allowed = {"g05": {"precision", "linear_accumulation", "vision_camera_batch", "math_precision"},
               "input": {"checkpoint", "release_root", "source", "batch", "source_golden",
                         "source_step", "instruction", "text_tokens", "text_lengths",
                         "cameras", "camera_names", "num_steps"},
               "run": {"warmup", "runs", "seed", "output_dir", "profile", "hwperf",
                       "cpu_threads", "torch_num_threads"}}
    for name, keys in allowed.items():
        if not isinstance(config[name], dict) or set(config[name]) - keys:
            raise ValueError(f"[{name}] has unsupported keys or is not a table")
    precision = precision_options(config["g05"].get("precision"),
                                  config["g05"].get("linear_accumulation"),
                                  config["g05"].get("math_precision", "base"))
    names = input_camera_names(config["input"])
    camera_batch = config["g05"].get("vision_camera_batch", 1)
    if camera_batch != "auto" and (type(camera_batch) is not int or camera_batch not in (1, 2, 3)):
        raise ValueError("[g05].vision_camera_batch must be auto, 1, 2 or 3")
    if camera_batch != "auto" and camera_batch > 1:
        if camera_batch != len(names):
            raise ValueError("G05 K1 camera group must match the selected camera count")
        if precision["weight_dtype"] != "w8a16":
            raise ValueError("G05 K1 camera grouping requires W8A16 Vision")
    inputs, run = config["input"], config["run"]
    for key in ("checkpoint", "release_root"):
        if not isinstance(inputs.get(key), str) or not inputs[key].strip():
            raise ValueError(f"[input].{key} must be a nonempty path")
    source = inputs.get("source", "batch")
    if source not in ("batch", "synthetic"):
        raise ValueError("[input].source must be batch or synthetic")
    path_key = "batch" if source == "batch" else "source_golden"
    if not isinstance(inputs.get(path_key), str) or not inputs[path_key].strip():
        raise ValueError(f"[input].{path_key} must be a nonempty path for source={source}")
    if source == "batch" and any(key in inputs for key in ("source_golden", "source_step", "instruction")):
        raise ValueError("source=batch takes its observation and instruction from the saved batch")
    if source == "synthetic" and "batch" in inputs:
        raise ValueError("[input].batch requires source=batch")
    lengths = inputs.get("text_lengths")
    if "text_tokens" in inputs and "text_lengths" in inputs:
        raise ValueError("[input].text_tokens and text_lengths are mutually exclusive")
    if "text_lengths" in inputs and (not isinstance(lengths, list) or not lengths
            or any(type(value) is not int or not 32 <= value <= 256 for value in lengths)
            or len(set(lengths)) != len(lengths)):
        raise ValueError("[input].text_lengths must be unique integers from 32 to 256")
    if lengths is not None and source != "synthetic":
        raise ValueError("[input].text_lengths requires source=synthetic; a saved batch has one actual context")
    text = inputs.get("text_tokens")
    if (text is not None or (source == "synthetic" and lengths is None)) and (type(text) is not int or not 32 <= text <= 256):
        raise ValueError("[input].text_tokens must be an integer from 32 to 256")
    if lengths is not None and len(lengths) > 1 and "instruction" in inputs:
        raise ValueError("one explicit instruction cannot specify multiple actual text_lengths")
    if "instruction" in inputs and (not isinstance(inputs["instruction"], str) or not inputs["instruction"].strip()):
        raise ValueError("[input].instruction must be nonempty text")
    if type(inputs.get("source_step", 1)) is not int or inputs.get("source_step", 1) not in (0, 1):
        raise ValueError("[input].source_step must be 0 or 1")
    for key, expected in (("num_steps", 10),):
        if type(inputs.get(key, expected)) is not int or inputs.get(key, expected) != expected:
            raise ValueError(f"G05 exact K1 examples require {key}={expected}")
    for key, default, minimum in (("warmup", 2, 0), ("runs", 6, 1), ("seed", 7, 0)):
        value = run.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"[run].{key} must be an integer >= {minimum}")
    for key in ("profile", "hwperf"):
        if type(run.get(key, False)) is not bool:
            raise ValueError(f"[run].{key} must be boolean")
    if not isinstance(run.get("output_dir"), str) or not run["output_dir"].strip():
        raise ValueError("[run].output_dir must be a nonempty path")
    if "rpu_execution" in config and not isinstance(config["rpu_execution"], dict):
        raise ValueError("[rpu_execution] must be a table")
    model = config.get("rpu_execution", {}).get("model", {})
    if (not isinstance(model, dict) or set(model) - {"num_cores"}
            or model.get("num_cores", 8) not in (4, 6, 8)
            or type(model.get("num_cores", 8)) is not int):
        raise ValueError("[rpu_execution.model] accepts only num_cores = 4, 6 or 8")
    cores = core_budget(config)
    if cores != 8:
        # Vision and the action expert run on four cores; HIGH math and camera
        # grouping have eight-core kernels only.
        if precision["math_precision"] == "high":
            raise ValueError(f"math_precision = 'high' is eight-core only; got model.num_cores={cores}")
        if camera_batch not in ("auto", 1):
            raise ValueError(
                f"vision_camera_batch = {camera_batch} groups cameras into one Vision dispatch, "
                f"which is eight-core only; use auto or 1 at model.num_cores={cores}")
    cpu_helpers()["resolve_cpu_threads"](config, default=8)
    library_helpers()["validate_libraries"](config.get("libraries"))


def registered_example(config):
    validate_example_config(config)
    # New precision/asset combinations do not inherit registered FP16 evidence.
    return None


def graph_caches(policy):
    model = policy.model
    return {"vision": model.vision_tower._rpu_vision_graph_cache,
            "text": model.vlm._rpu_qwen3_5.prefill_graph_cache,
            "action": model.action_expert._rpu_qwen3_5.action_graph_cache}


def graph_summary(policy):
    return {name: {"size": int(cache.size()),
        "invariant": bool(cache.cache_invariant_ok()), "entries": [
        {"signature": repr(entry.signature), **{key: int(getattr(entry, key)) for key in (
            "graph_lifetime_id", "build_generation", "replay_count", "recapture_count",
            "kernel_count", "segment_count", "dma_count", "barrier_count",
            "register_census_present", "register_census_build_committed")}}
        for entry in cache.snapshot()]} for name, cache in graph_caches(policy).items()}


def require_replay(before, after, count, camera_batch, cameras=3):
    if cameras not in (2, 3) or camera_batch not in (1, cameras):
        raise RuntimeError("G05 replay camera count and grouping disagree")
    if set(before) != {"vision", "text", "action"} or set(before) != set(after):
        raise RuntimeError("G05 requires three retained Graph owners")
    for name, old in before.items():
        new = after[name]
        if (not old["invariant"] or not new["invariant"] or old["size"] != 1 or new["size"] != 1
                or len(old["entries"]) != 1 or len(new["entries"]) != 1):
            raise RuntimeError(f"{name}: expected one invariant retained Graph")
        previous, current = old["entries"][0], new["entries"][0]
        identity = ("signature", "graph_lifetime_id", "build_generation", "kernel_count",
                    "segment_count", "dma_count", "barrier_count")
        if any(previous[key] != current[key] for key in identity):
            raise RuntimeError(f"{name}: Graph owner/layout changed")
        factor = cameras // camera_batch if name == "vision" else 1
        if (previous["kernel_count"] <= 0 or previous["segment_count"] <= 0
                or previous["recapture_count"] or current["recapture_count"]
                or current["replay_count"] != previous["replay_count"] + count * factor):
            raise RuntimeError(f"{name}: expected nontrivial stable REPLAY without recapture")


@contextmanager
def stage_timings(policy):
    """Diagnostic replays only: wrappers are outside the policy latency samples."""
    model, phases, originals = policy.model, {}, []
    try:
        for owner, name, label in ((model, "_forward_embed", "vision_ms"),
                                  (model.vlm, "forward", "prefill_ms"),
                                  (model, "inference_fm", "denoise_ms")):
            original = getattr(owner, name)
            originals.append((owner, name, original))
            def measured(*args, _fn=original, _label=label, **kwargs):
                start = time.perf_counter()
                value = _fn(*args, **kwargs)
                phases[_label] = phases.get(_label, 0.0) + (time.perf_counter() - start) * 1000
                return value
            object.__setattr__(owner, name, measured)
        yield phases
    finally:
        for owner, name, original in reversed(originals):
            object.__setattr__(owner, name, original)


def summarize(rows):
    return {key: {"mean": statistics.mean(row[key] for row in rows),
                  "median": statistics.median(row[key] for row in rows),
                  "min": min(row[key] for row in rows), "max": max(row[key] for row in rows)}
            for key in rows[0]}


def checked_output(value):
    import torch
    from examples.common.measurement import _cpu_output

    output = _cpu_output(value)
    if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 32, 27):
        raise RuntimeError("G05 requires action output [1,32,27]")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("G05 action output contains nonfinite values")
    return output


def byte_equal(a, b):
    import torch
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def execution_receipts(policy):
    model = policy.model
    receipts = {name: copy.deepcopy(getattr(owner, "_rpu_last_execution_plan", None))
                for name, owner in (("vision", model.vision_tower), ("text", model.vlm),
                                    ("action", model.action_expert))}
    # Direct embedding fusion calls the native Vision implementation without
    # the G05 tower wrapper that publishes _rpu_last_execution_plan. Read its
    # actual plan, also consumed by the B3 fallback check, for both B1 and B3.
    vision_plan = getattr(model.vision_tower, "_rpu_vision_last_a6_plan", None)
    if vision_plan is not None:
        receipts["vision"] = copy.deepcopy(vision_plan)
    return receipts


def _run_row_process(config):
    """Give each independently loaded policy a fresh SDK device lifetime."""
    output_dir = Path(config["run"]["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    # A child can abort inside the SDK before Python can write its final report.
    # Start this attempt explicitly so an old COMPLETE cannot survive that abort.
    _RESULTS["write_report"]({"model": "g05", "status": "RUNNING", "config": config},
                             output_dir, default=str)
    root = str(Path(__file__).resolve().parents[3])
    command = [sys.executable, "-B", "-c",
               f"import json, sys; sys.path.insert(0, {root!r})\n"
               "from examples.runners.g05.optimized import _run_single_example_config\n"
               "from examples.common.logging import logging_session\n"
               "config = json.load(sys.stdin)\n"
               "with logging_session(config, keep_until_exit=True):\n"
               "    code = _run_single_example_config(config)\n"
               "raise SystemExit(code)\n"]
    try:
        with (output_dir / "stdout.log").open("w") as log:
            completed = subprocess.run(command, input=json.dumps(config), text=True,
                                       stdout=log, stderr=subprocess.STDOUT, check=False)
    except BaseException as error:
        report = json.loads(report_path.read_text())
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        _RESULTS["add_result_fields"](report, output_dir, completed=False)
        _RESULTS["write_report"](report, output_dir, default=str)
        raise
    report = json.loads(report_path.read_text())
    report["process_exit_code"] = completed.returncode
    failed = completed.returncode != 0 or report.get("status") != "COMPLETE"
    if failed:
        report["status"] = "FAILED"
        report.setdefault("error", f"G05 row process exited {completed.returncode}; see stdout.log")
        _RESULTS["add_result_fields"](report, output_dir, completed=False)
    _RESULTS["write_report"](report, output_dir, default=str)
    if failed:
        raise RuntimeError(f"G05 T{config['input']['text_tokens']}: {report['error']}")


def run_example_config(config):
    """Measure one actual context or a list of separately prepared contexts."""
    validate_example_config(config)
    lengths = config["input"].get("text_lengths")
    if lengths is None:
        return _run_single_example_config(config)
    output_dir = Path(config["run"]["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"model": "g05", "status": "RUNNING", "formal_certification": False,
              "math_precision": config["g05"].get("math_precision", "base"),
              "evaluation_only": True, "cpu_reference": "not_run",
              "task_quality": "not_evaluated", "config": config, "rows": {},
              "row_execution": "one fresh process per text length"}
    try:
        # Validate and bind one library selection, inherited by every child.
        # Rm/Free do not restore the current SDK/driver's cumulative mapping
        # budget. Independent policies must close their device process between
        # rows; ordinary Python/native retirement alone cannot restore it.
        helpers = library_helpers()
        env, libraries = helpers["bind_libraries"](config)
        helpers["bind_cold_environment"](env)
        cpu = cpu_helpers()
        report["cpu_runtime"] = cpu["configure_cpu_threads"](
            cpu["resolve_cpu_threads"](config, default=8))
        for length in lengths:
            row = copy.deepcopy(config)
            row["input"].pop("text_lengths")
            row["input"]["text_tokens"] = length
            row_dir = output_dir / f"text_{length}"
            row["run"]["output_dir"] = str(row_dir)
            try:
                _run_row_process(row)
            finally:
                path = row_dir / "report.json"
                if path.is_file():
                    report["rows"][str(length)] = json.loads(path.read_text())
        report["status"] = "COMPLETE"
    except BaseException as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _RESULTS["add_result_fields"](report, output_dir, completed=report["status"] == "COMPLETE",
            runtime_checks=("all requested text-length rows executed and retired",))
        _RESULTS["write_report"](report, output_dir, default=str)
    return 0


def _run_single_example_config(config, *, libraries=None):
    validate_example_config(config)
    requested_config = config
    config, selection = resolve_row_config(config)
    from examples.runners.g05.runner import _g05
    from examples.common.measurement import _close, _provenance

    options = config["run"]
    output_dir = Path(options["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    internal = copy.deepcopy(config)
    internal["example"] = {"target": "g05"}
    runner = SimpleNamespace(source_config=internal, owner=None,
                             _g05_prepare_inputs=True,
                             _g05_patch_options=precision_options(config["g05"]["precision"],
                                 config["g05"].get("linear_accumulation"),
                                 config["g05"].get("math_precision", "base")))
    warmup, runs, seed = options.get("warmup", 2), options.get("runs", 6), options.get("seed", 7)
    # Even a legacy warmup=0 request needs two settling calls to verify REPLAY.
    effective_warmup = max(2, warmup)
    camera_batch = config["g05"].get("vision_camera_batch", 1)
    cameras = len(input_camera_names(config["input"]))
    precision = canonical_precision(config["g05"]["precision"])
    report = {"model": "g05", "precision": precision,
              "math_precision": runner._g05_patch_options["math_precision"],
              "requested_precision": config["g05"]["precision"],
              "effective_precision": runner._g05_patch_options,
              "status": "RUNNING", "formal_certification": False,
              "correctness": "not_run", "evaluation_only": True,
              "cpu_reference": "not_run", "task_quality": "not_evaluated",
              "numerical_notes": ["This example does not run a reference comparison or robot task."],
              "config": config, "requested_config": requested_config, "selection": selection,
              "warmup": warmup, "effective_warmup": effective_warmup,
              "runs": runs, "seed": seed, "cameras": cameras,
              "camera_names": list(input_camera_names(config["input"])), "vision_camera_batch": camera_batch,
              "timing_scope": "copy of preprocessed batch + policy.predict_action; CPU output copy excluded",
              "phase_timing_scope": "separate instrumented replays; Vision includes embedding; "
                                    "prefill and denoise are host wall time, not HWPerf device time",
              "wall_ms_runs": [], "samples": [], "provenance": _provenance(config)}
    failure = None
    try:
        helpers = library_helpers()
        if libraries is None:
            env, libraries = helpers["bind_libraries"](config)
            helpers["bind_cold_environment"](env)
        report["libraries"] = libraries
        cpu = cpu_helpers()
        report["cpu_runtime"] = cpu["configure_cpu_threads"](
            cpu["resolve_cpu_threads"](config, default=8))
        import torch
        import rpu_backend  # noqa: F401
        from rpu_backend.runtime.performance import _profile_ctx

        report["runtime_payloads"] = helpers["runtime_payloads"](libraries)
        with torch.inference_mode():
            torch.manual_seed(seed)
            infer = _g05(runner)
            report["input"] = runner._g05_input_metadata
            settling = []
            for _ in range(effective_warmup):
                torch.manual_seed(seed)
                checked_output(infer())
                settling.append(graph_summary(runner.owner))
            require_replay(settling[-2], settling[-1], 1, camera_batch, cameras)
            report["settling_graphs"] = settling
            for cache in graph_caches(runner.owner).values():
                cache.freeze()
            report["graphs_before"] = graph_summary(runner.owner)
            for _ in range(runs):
                torch.manual_seed(seed)
                start = time.perf_counter()
                value = infer()
                report["wall_ms_runs"].append((time.perf_counter() - start) * 1000)
                report["samples"].append({"total_ms": report["wall_ms_runs"][-1]})
                output = checked_output(value)
            report["graphs_after"] = graph_summary(runner.owner)
            require_replay(report["graphs_before"], report["graphs_after"], runs, camera_batch, cameras)
            report["execution_receipts"] = execution_receipts(runner.owner)
            # G05 draws FM noise inside policy.predict_action. Changing the RNG
            # seed exercises the same frozen owners with a different real input.
            torch.manual_seed(seed + 1)
            changed_value = infer()
            changed = checked_output(changed_value)
            if byte_equal(output, changed):
                raise RuntimeError("A/B/A: changed FM noise did not change actions")
            if not byte_equal(output, checked_output(value)):
                raise RuntimeError("G05 overwrote a previous returned action tensor")
            torch.manual_seed(seed)
            restored = checked_output(infer())
            if not byte_equal(output, restored) or not byte_equal(changed, checked_output(changed_value)):
                raise RuntimeError("A/B/A: restored noise or fresh-output contract failed")
            report["graphs_after_aba"] = graph_summary(runner.owner)
            require_replay(report["graphs_after"], report["graphs_after_aba"], 2, camera_batch, cameras)
            report["aba"] = {"input_change": "FM RNG seed", "a_seed": seed, "b_seed": seed + 1,
                             "a_vs_b_byte_equal": False, "a_restore_byte_equal": True,
                             "previous_outputs_unchanged": True}
            phase_rows = []
            with stage_timings(runner.owner) as phases:
                for _ in range(2):
                    phases.clear()
                    torch.manual_seed(seed)
                    checked_output(infer())
                    if set(phases) != {"vision_ms", "prefill_ms", "denoise_ms"}:
                        raise RuntimeError("incomplete G05 phase timings")
                    phase_rows.append(dict(phases))
            require_replay(report["graphs_after_aba"], graph_summary(runner.owner), 2, camera_batch, cameras)
            report.update(phase_samples=phase_rows, phase_summary=summarize(phase_rows),
                          summary=summarize(report["samples"]))
            # Trace separate stable replays so the reported latency excludes
            # device trace collection and JSON export, as in the G05 benchmark.
            if options.get("hwperf", False):
                with torch.rpu.hw_perf_trace(str(output_dir / "hwperf"), max_dumps=16):
                    for _ in range(2):
                        torch.manual_seed(seed)
                        infer()
                report["hwperf_replays"] = 2
            if options.get("profile", False):
                torch.manual_seed(seed)
                with _profile_ctx(True) as profiler:
                    infer()
                profiler.export_chrome_trace(str(output_dir / "profile.trace.json"))
            torch.save(output, output_dir / "actions.pt")
            report["graphs_final"] = graph_summary(runner.owner)
            diagnostic_calls = 4 + (2 if options.get("hwperf", False) else 0) + int(options.get("profile", False))
            require_replay(report["graphs_after"], report["graphs_final"], diagnostic_calls, camera_batch, cameras)
            report.update(status="COMPLETE", mean_ms=statistics.mean(report["wall_ms_runs"]),
                          median_ms=statistics.median(report["wall_ms_runs"]),
                          output_shape=list(output.shape), finite=True,
                          correctness="runtime_checks_only")
    except BaseException as error:
        failure = error
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            try:
                _close(runner.owner, "g05")
            except BaseException as error:
                report.update(status="FAILED", cleanup_error=f"{type(error).__name__}: {error}")
                if failure is None:
                    raise
        finally:
            _RESULTS["add_result_fields"](report, output_dir, completed=report["status"] == "COMPLETE",
                timing=_RESULTS["latency_timing"](report["wall_ms_runs"], report["timing_scope"])
                    + _RESULTS["phase_timings"](report.get("phase_samples", []), report["phase_timing_scope"],
                        metric_prefix="diagnostic_"),
                runtime_checks=("finite action output shape", "Graph REPLAY/no recapture",
                                "same-seed A/B/A and returned-output ownership"),
                output_names=("actions.pt", "input_batch.pt") +
                    (("profile.trace.json",) if options.get("profile", False) else ()))
            _RESULTS["write_report"](report, output_dir, default=str)
    _LOG.debug("report=%s", report)
    return 0
