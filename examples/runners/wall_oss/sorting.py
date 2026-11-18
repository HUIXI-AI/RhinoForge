#!/usr/bin/env python3
"""Explicit precision examples for parcel-sorting WRC v728 (source checkout)."""

from __future__ import annotations

from examples.common import cpu as _cpu_helpers
from examples.common import libraries as _library_helpers
from examples.common import results as _result_helpers
from examples.common.logging import prepare_output_directory

import json
import logging
import os
from pathlib import Path
import sys
import time


_LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
_CPU = vars(_cpu_helpers)
PRECISIONS = {"fp16": "fp16", "w8a16": "w8", "w8a16_action_w4a16": "action-w4"}
CAMERAS = ("face_view", "left_wrist_view", "right_wrist_view")

def validation_runner():
    from importlib import import_module
    return import_module("examples.runners.wall_oss.sorting_engine")


def validate_example_config(config: dict) -> None:
    allowed = {
        "wall_oss_sorting": {"precision"},
        "input": {"fp16_checkpoint", "w8_checkpoint", "action_w4_checkpoint", "batch", "inference_delay"},
        "run": {"warmup", "runs", "output_dir", "cpu_threads", "torch_num_threads"},
    }
    if not set(allowed) <= set(config) or set(config) - (set(allowed) | {"cpu", "logging"}):
        raise ValueError("sorting examples require [wall_oss_sorting], [input], [run], optional [cpu]/[logging]")
    from examples.common.config import validate_logging_config
    validate_logging_config(config)
    for name, keys in allowed.items():
        if not isinstance(config[name], dict) or set(config[name]) - keys:
            raise ValueError(f"[{name}] contains unsupported keys or is not a table")
    _CPU["resolve_cpu_threads"](config)
    profile, inputs, run = (config[key] for key in allowed)
    precision = profile.get("precision")
    if not isinstance(precision, str) or precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {tuple(PRECISIONS)}")
    required = {"fp16_checkpoint", "batch"}
    if precision != "fp16":
        required.add("w8_checkpoint")
    if precision == "w8a16_action_w4a16":
        required.add("action_w4_checkpoint")
    for key in required:
        if not isinstance(inputs.get(key), str) or not inputs[key].strip():
            raise ValueError(f"[input].{key} must be a nonempty path")
    if set(inputs) - required - {"inference_delay"}:
        raise ValueError("checkpoint paths must match the selected precision; unused weights are not loaded")
    delay = inputs.get("inference_delay", 0)
    if type(delay) is not int or delay not in (0, 1, 3, 10):
        raise ValueError("inference_delay must be 0, 1, 3, or 10")
    for key, default, minimum in (("warmup", 2, 1), ("runs", 6, 1)):
        value = run.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"[run].{key} must be an integer >= {minimum}")
    if not isinstance(run.get("output_dir"), str) or not run["output_dir"].strip():
        raise ValueError("[run].output_dir must be a nonempty path")


def cold_environment(precision: str) -> tuple[dict, dict]:
    env = validation_runner().exact_environment(ROOT, PRECISIONS[precision])
    # A wheel owns its native extension. Only override it for an explicitly
    # selected library or a build actually present in this source checkout.
    checkout_native = Path(env.pop("RPU_BACKEND_SO"))
    selected_native = os.environ.get("RPU_BACKEND_SO")
    if selected_native is not None:
        env["RPU_BACKEND_SO"] = str(Path(selected_native).expanduser().resolve())
    elif checkout_native.is_file():
        env["RPU_BACKEND_SO"] = str(checkout_native.resolve())
    env["RPU_LINEAR_ACC32"] = "0"
    resolver = vars(_library_helpers)
    libraries, records = resolver["current_libraries"](ROOT)
    env.update(libraries)
    return env, records


def bind_cold_environment(env: dict) -> None:
    if "rpu_backend" in sys.modules:
        raise ValueError("sorting examples require a fresh process before importing rpu_backend")
    inherited = dict(os.environ)
    if "RPU_BACKEND_SO" in inherited:
        inherited["RPU_BACKEND_SO"] = str(Path(inherited["RPU_BACKEND_SO"]).expanduser().resolve())
    conflicts = [key for key, value in env.items()
                 if key in inherited and inherited[key] != value]
    conflicts += [key for key in os.environ if key.startswith("RPU_WALL_OSS_") and key not in env]
    if conflicts:
        raise ValueError("conflicting cold environment: " + ", ".join(sorted(set(conflicts))))
    if os.environ.get("RPU_CHUNK_FORCE_UNSAFE", "0").strip() not in ("", "0"):
        raise ValueError("RPU_CHUNK_FORCE_UNSAFE is incompatible with this profile")
    if os.environ.get("WALL_OSS_PROFILE_PREFIX") or os.environ.get("RPU_RMSNORM_VWARP"):
        raise ValueError("unset WALL_OSS_PROFILE_PREFIX and RPU_RMSNORM_VWARP before running")
    os.environ.update(env)


def observation(batch: dict, delay: int) -> dict:
    """Check the CPU observation before creating or mutating a model."""
    import torch
    from PIL import Image

    required = {"images", "instruction", "state", "state_mask", "noise", "dof_mask"}
    if not isinstance(batch, dict) or not required <= set(batch) or set(batch) - required - {"prefix_actions"}:
        raise ValueError("batch requires images, instruction, state, state_mask, noise, dof_mask; optional prefix_actions")
    if not isinstance(batch["instruction"], str) or not batch["instruction"].strip():
        raise ValueError("batch instruction must be nonempty text")
    shapes = {"images": (3, 252, 448, 3), "state": (26,), "state_mask": (26,),
              "noise": (1, 20, 26), "dof_mask": (26,), "prefix_actions": (20, 26)}
    for key, shape in shapes.items():
        if key not in batch:
            continue
        value = batch[key]
        if not torch.is_tensor(value) or value.device.type != "cpu" or tuple(value.shape) != shape:
            raise ValueError(f"batch {key} must be a CPU tensor with shape {shape}")
        dtype_ok = (value.dtype == torch.uint8 if key == "images" else
                    value.dtype == torch.bool if key.endswith("mask") else
                    value.dtype in (torch.float16, torch.float32))
        if not dtype_ok or not bool(torch.isfinite(value).all()):
            raise ValueError(f"invalid dtype or nonfinite values in batch {key}")
    mask = torch.tensor([True] * 20 + [False] * 6)
    if not torch.equal(batch["state_mask"], mask) or not torch.equal(batch["dof_mask"], mask):
        raise ValueError("state_mask and dof_mask must be [true]*20+[false]*6 for WRC v728")
    if delay and "prefix_actions" not in batch:
        raise ValueError("nonzero inference_delay requires batch prefix_actions in physical action units")
    if not delay and "prefix_actions" in batch:
        raise ValueError("prefix_actions requires nonzero inference_delay")
    return {
        "image": [Image.fromarray(frame.numpy()) for frame in batch["images"]],
        "instruction": batch["instruction"], "proprioception": batch["state"].float(),
        "state_mask": batch["state_mask"].float(), "noise": batch["noise"].float(),
        "dof_mask": batch["dof_mask"].float()[None, None, :].expand(1, 20, 26).contiguous(),
        "num_steps": 6, "use_rtc": True, "inference_delay": delay,
        "rtc_prefix_actions": batch["prefix_actions"].float() if delay else None,
    }


def checkpoint_arguments(inputs: dict, precision: str) -> tuple[str, dict]:
    fp16 = Path(inputs["fp16_checkpoint"]).expanduser().resolve(strict=True)
    from rpu_backend.adapters.wall_oss.checkpoint_validation import (
        _load_json, _quant_metadata,
        validate_rtc_w4_checkpoint_pair, validate_w8_nvfp4_checkpoint_pair)

    if _quant_metadata(fp16, _load_json(fp16 / "config.json")) is not None:
        raise ValueError("fp16_checkpoint must be the unquantized WRC v728 source")
    w8 = Path(inputs["w8_checkpoint"]).expanduser().resolve(strict=True) if precision != "fp16" else None
    w4 = (Path(inputs["action_w4_checkpoint"]).expanduser().resolve(strict=True)
          if precision == "w8a16_action_w4a16" else None)
    if w8 is not None:
        validate_w8_nvfp4_checkpoint_pair(w8, fp16)
    if w4 is not None:
        validate_rtc_w4_checkpoint_pair(w4, fp16)
    return str(w8 or fp16), {
        "fp16_ckpt_dir": str(fp16), "w8a16": w8 is not None,
        "action_w4a16_ckpt_dir": str(w4) if w4 else None,
        "rtc": True, "dataset_key": "ex_normal", "camera_names": CAMERAS,
        "delta_action": True, "state_bins": 512, "max_seq_len": 2048,
        # Wall's builders are TP8; unlike Pi0.5 they do not accept model.num_cores.
        "rpu_execution": {},
    }


_RESULTS = vars(_result_helpers)


def run_example_config(config: dict) -> int:
    validate_example_config(config)
    precision = config["wall_oss_sorting"]["precision"]
    inputs, run = config["input"], config["run"]
    output_dir = prepare_output_directory(run["output_dir"])
    env, libraries = cold_environment(precision)
    bind_cold_environment(env)
    # Use the activated environment's wheel/editable installation. Validate its
    # identity below without redirecting a wheel to the source snapshot.
    effective_cpu = _CPU["configure_cpu_threads"](_CPU["resolve_cpu_threads"](config))
    import torch
    import rpu_backend
    from rpu_backend import _cpp_ext
    from rpu_backend.adapters.wall_oss import build_wall_oss_vla
    from rpu_backend.graph import GraphRuntimePolicy

    _library_helpers.verify_backend_source(ROOT, rpu_backend.__file__)
    actual = _cpp_ext.get_runtime_payload_paths()
    for role, key in (("main", "RPU_KERNEL_LIB_PATH"), ("expansion", "RPU_KERNEL_LIB_PATH_EXPANSION")):
        path = actual.get("oplib_roles", {}).get(role)
        if path != libraries[key]["path"] or actual.get("oplib_payloads", {}).get(path, {}).get("sha256") != libraries[key]["sha256"]:
            raise RuntimeError(f"loaded {role} library does not match the sorting profile")
    batch_path = Path(inputs["batch"]).expanduser().resolve(strict=True)
    common = observation(torch.load(batch_path, map_location="cpu", weights_only=True), inputs.get("inference_delay", 0))
    checkpoint, kwargs = checkpoint_arguments(inputs, precision)
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_ddr_flush(False)
    torch.rpu.set_ddr_flush_force(True)
    helpers = validation_runner()
    report = {"model": "wall_oss_parcel_sorting_wrc_v728", "precision": precision,
              "config": config, "checkpoint": checkpoint, "batch": str(batch_path),
              "warmup": run.get("warmup", 2), "runs": run.get("runs", 6),
              "evaluation_only": True, "formal_certification": False}
    with _RESULTS["record_result"](report, output_dir,
        timing=lambda: _RESULTS["latency_timing"](report.get("e2e", {}).get("samples_ms", []),
            report.get("timing_scope", "predict(debug=False) including CPU actions"))
            + _RESULTS["stage_phase_timings"](report.get("stage_runs", []),
                "host wall time of the Vision, Prefill and Denoise calls inside each measured predict; "
                "glue between stages is not attributed"),
        runtime_checks=("finite action output shape [1,20,26]", "fixed-input replay",
                        "Graph READY/REPLAY/no recapture") +
                       (("supplied RTC prefix preserved",) if common["inference_delay"] else ()),
        output_names=("actions.pt",)):
        vla = None
        try:
            # The P640 profile retains Vision, Prefill and twelve denoise
            # graphs (non-RTC plus delays 0..10). Patch embedding also needs
            # one transient queue. Reserve their arenas before installing weights.
            report["graph_arena_count"] = 15
            if not GraphRuntimePolicy.from_environment(graph_arena_count=15).prepare_arenas():
                raise RuntimeError("sorting could not prepare fifteen Graph arenas")
            vla = build_wall_oss_vla(checkpoint, **kwargs)
            if precision == "w8a16_action_w4a16" and not (
                vla.action.llm is vla.llm and not vla.action._owns_llm
                and vla.action._precision_mode == 2 and vla._runtime_complete()
            ):
                raise RuntimeError("Action-only W4 shared-KV composition was not built")
            prepared = vla.prepare_graphs(
                images=common["image"], **{key: common[key] for key in (
                    "instruction", "proprioception", "state_mask", "noise", "dof_mask", "num_steps")},
                prefix_length_range=(599, 640))
            if prepared.get("phase") != "READY" or not prepared.get("ready_replay_validated"):
                raise RuntimeError("sorting graphs did not reach validated READY")
            before = helpers.cache_snapshot(vla)
            samples, stage_rows, reference = [], [], None
            report["e2e"] = {"samples_ms": samples}
            report["stage_runs"] = stage_rows
            from examples.runners.wall_oss.runner import wall_stage_boundaries
            boundaries, report["stage_boundaries_missing"] = _RESULTS["available_stage_boundaries"](
                wall_stage_boundaries(vla))
            with _RESULTS["stage_timings"](boundaries) as phases:
                for index in range(run.get("warmup", 2) + run.get("runs", 6)):
                    phases.clear()
                    started = time.perf_counter()
                    output = vla.predict(**common, debug=False)
                    action = output["action"].detach().cpu()
                    elapsed = (time.perf_counter() - started) * 1000
                    stage = dict(phases)
                    if tuple(action.shape) != (1, 20, 26) or not bool(torch.isfinite(action).all()):
                        raise RuntimeError("expected finite actions [1,20,26]")
                    if output["prefix_len"] != 640 or not 599 <= output["real_prefix_len"] <= 640:
                        raise RuntimeError("observation does not use the P640 sorting profile")
                    if reference is None:
                        reference = action.clone()
                    elif not torch.equal(action, reference):
                        raise RuntimeError("fixed-input sorting replay changed actions")
                    delay = common["inference_delay"]
                    if delay and not torch.equal(action[0, :delay], common["rtc_prefix_actions"][:delay]):
                        raise RuntimeError("RTC replay did not preserve the supplied action prefix")
                    if index >= run.get("warmup", 2):
                        samples.append(elapsed)
                        stage_rows.append(stage)
            final = helpers.cache_snapshot(vla)
            if not helpers.graph_gate(final, before, prepared["graph_counts"]):
                raise RuntimeError("sorting Graph READY/replay invariant failed")
            report.update({
                "model": "wall_oss_parcel_sorting_wrc_v728", "precision": precision,
                "evaluation_only": True, "formal_certification": False,
                "num_cores": 8, "camera_names": CAMERAS,
                "num_steps": 6, "action_horizon": 20, "inference_delay": common["inference_delay"],
                "real_prefix_len": int(output["real_prefix_len"]), "prefix_len": int(output["prefix_len"]),
                "checkpoint": checkpoint, "checkpoint_arguments": kwargs, "batch": str(batch_path),
                "cold_environment": env, "libraries": libraries, "runtime_payloads": actual,
                "graph_prepare": prepared, "cache_before_timing": before, "cache_final": final,
                "warmup": run.get("warmup", 2), "runs": run.get("runs", 6),
                "cpu_runtime": effective_cpu,
                "timing_scope": "predict(debug=False) including CPU actions; excludes load, prepare and warmup; tracing disabled",
                "e2e": helpers.perf_summary(samples), "output_shape": list(action.shape),
                "fixed_input_replay_equal": True,
                "numerical_notes": "CPU parity and parcel-sorting task quality are not evaluated by this example.",
            })
        finally:
            _RESULTS["close_with_report"](vla, report)
        torch.save(reference, output_dir / "actions.pt")
    _LOG.debug("report=%s", report)
    return 0
