#!/usr/bin/env python3
"""Independent parcel-sorting WRC v728 measurement: fixed full workload, numeric metrics recorded, runtime invariants enforced."""

from __future__ import annotations

from examples.common import libraries as _library_helpers

import argparse
import contextlib
import io
import importlib.util
import json
import logging
import hashlib
import subprocess
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback


_LOG = logging.getLogger(__name__)

DELAYS = (0, 1, 3, 10)
CORE_COUNTS = (4, 6, 8)
# The parcel-sorting schedules above have eight-lane owners only. Four/six
# cores run the ordinary routes with the fused denoiser, SPM Vision RoPE,
# four-lane fused Vision merger and residual-scaled FP16 expert 0 / Vision
# (build_wall_oss_vla(rtc=True) selects the scale).
_REDUCED_OFF = (
    "RPU_WALL_OSS_ATTN_TP8", "RPU_WALL_OSS_BATCH_VISION",
    "RPU_WALL_OSS_VISION_WEIGHT_OUTER",
    "RPU_WALL_OSS_VISION_ROW_WEIGHT_OUTER", "RPU_WALL_OSS_VISION_MIXED_PRECISION",
    "RPU_WALL_OSS_VISION_NATIVE_WINDOW_SDPA", "RPU_WALL_OSS_PREFILL_MIXED_PRECISION",
    "RPU_WALL_OSS_PREFILL_STAGED_MLP", "RPU_WALL_OSS_PREFILL_OWNER_DIRECT_KV",
    "RPU_WALL_OSS_ACTION_WEIGHT_RESIDENCY", "RPU_WALL_OSS_ACTION_KV_OWNER_FANOUT",
    "RPU_WALL_OSS_W16_REUSE_MASK", "RPU_WALL_OSS_W16_NEXT3_MASK",
    "RPU_WALL_OSS_W16_NEXT4_MASK", "RPU_WALL_OSS_FP16_TRANSFER_MASK",
    "RPU_WALL_OSS_FP16_MLP_M224", "RPU_WALL_OSS_SORTING_SILU_MUL",
    "RPU_WALL_OSS_FUSED_ASSEMBLE",
)
DEFAULT_BACKEND = Path(__file__).resolve().parents[3]
_LIBRARIES = vars(_library_helpers)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fp16", "w8", "action-w4"), required=True)
    parser.add_argument("--num-cores", type=int, choices=CORE_COUNTS, default=8,
                        help="Wall model.num_cores; 4/6 run the ordinary residual-scaled routes")
    parser.add_argument("--backend-root", type=Path, default=DEFAULT_BACKEND)
    parser.add_argument("--site-packages", type=Path,
                        help="Optional explicit Python dependency directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fp16", type=Path, required=True)
    parser.add_argument("--w8", type=Path)
    parser.add_argument("--w4", type=Path)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--so", type=Path, required=True)
    parser.add_argument("--main-ref", type=Path)
    parser.add_argument("--expansion", type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--post-profile", action=argparse.BooleanOptionalAction, default=True,
                        help="Collect a separate device profile after unprofiled timings")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate paths and print the startup bundle without importing device")
    args = parser.parse_args(argv)
    if args.mode != "fp16" and args.w8 is None:
        parser.error("--w8 is required for W8 Vision/prefill")
    if args.mode == "action-w4" and args.w4 is None:
        parser.error("--w4 is required for Action-only W4")
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be nonnegative and --repeat must be positive")
    if args.mode == "action-w4" and args.num_cores != 8:
        parser.error("Action-only W4 has an eight-core owner only")
    args.main_ref = args.main_ref or args.backend_root / "src/rhinoOpLib_current.ref"
    args.expansion = args.expansion or args.backend_root / "src/rhinoExpansionOpLib_current.ref"
    return args


def required_artifact(explicit: Path | None, env_name: str) -> Path:
    value = explicit if explicit is not None else os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"provide the matching CLI path or {env_name}")
    path = Path(value).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def exact_environment(root: Path, mode: str, num_cores: int = 8) -> dict[str, str]:
    """Pure cold defaults shared by the example and the validation CLI."""
    if mode not in ("fp16", "w8", "action-w4"):
        raise ValueError(f"unknown parcel-sorting mode: {mode}")
    if num_cores not in CORE_COUNTS:
        raise ValueError(f"parcel sorting accepts model.num_cores {CORE_COUNTS}, got {num_cores!r}")
    if mode == "action-w4" and num_cores != 8:
        raise ValueError("Action-only W4 parcel sorting has an eight-core owner only")
    values = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "RPU_WALL_OSS_RTC_ALLOW_NUMERIC_BLOCKED": "1",

        "RPU_WALL_OSS_ACTION_FP32_TAIL": "0",
        "RPU_WALL_OSS_FUSED_DENOISE": "1",
        "RPU_WALL_OSS_DENOISE_UNROLL": "1",
        "RPU_WALL_OSS_ATTN_TP8": "1",
        "RPU_WALL_OSS_VISION_FUSED_MERGER": "1",
        "RPU_WALL_OSS_VISION_WEIGHT_OUTER": "1",
        "RPU_WALL_OSS_VISION_LAYER_GROUP": "0",
        "RPU_WALL_OSS_VISION_MIXED_PRECISION": "1",
        "RPU_WALL_OSS_VISION_NATIVE_WINDOW_SDPA": "1",
        "RPU_WALL_OSS_PREFILL_MIXED_PRECISION": "1",
        "RPU_WALL_OSS_PREFILL_STAGED_MLP": "1",
        "RPU_WALL_OSS_ACTION_WEIGHT_RESIDENCY": "1",
        "RPU_WALL_OSS_ACTION_KV_OWNER_FANOUT": "1",
        "RPU_WALL_OSS_FAST_REPLAY": "1",
        "RPU_FASTREPLAY_SKIP_SYNC": "1",
        "RPU_DEEP_FAST_REPLAY": "1",
        "RPU_WALL_OSS_FUSED_ASSEMBLE": "0",
        "RPU_WALL_OSS_VISION_ROPE_SPM": "1",
        "RPU_WALL_OSS_PARTIAL_MROPE": "1",
        "RPU_WALL_OSS_FAST_PROCESSOR": "1",
        "RPU_WALL_OSS_FAST_IMGPROC": "1",
        "RPU_WALL_OSS_PE_NORM_FOLD": "1",
        "RPU_WALL_OSS_ONDEVICE_EMBED": "1",
        "RPU_WALL_OSS_HOST_CACHE": "1",
        "RPU_WALL_OSS_INSTRUMENT": "0",
        "LKN_KD_BUF_MB": "8",
        "LKN_INSTR_BUF_MB": "64",
        "LKN_MAX_BATCH_ENTRIES": "65536",
        "RPU_KERNEL_LIB_PATH": str(root / "src/rhinoOpLib_current.ref"),
        "RPU_KERNEL_LIB_PATH_EXPANSION": str(
            root / "src/rhinoExpansionOpLib_current.ref"
        ),
        "RPU_BACKEND_SO": str(root / "build/rpu_backend/rpu_backend.so"),
    }
    values.update(
        RPU_WARMUP="0", RPU_WALL_OSS_DEVICE_PATCH_EMBED="1",
        RPU_WALL_OSS_BATCH_VISION="1", RPU_WALL_OSS_KVINSERT_PAD16="1",
        RPU_WALL_OSS_PREFILL_OWNER_DIRECT_KV="1",
        RPU_WALL_OSS_VISION_ROW_WEIGHT_OUTER="1",
        RPU_WALL_OSS_W16_REUSE_MASK="8" if mode != "action-w4" else "0",
        RPU_WALL_OSS_W16_NEXT3_MASK="1",
        RPU_WALL_OSS_W16_NEXT4_MASK="3",
        RPU_WALL_OSS_FP16_TRANSFER_MASK="7" if mode == "fp16" else "0",
        RPU_WALL_OSS_FP16_MLP_M224="1" if mode == "fp16" else "0",
        RPU_WALL_OSS_SORTING_SILU_MUL="1",
    )
    if mode == "action-w4":
        values.update(
            RPU_WALL_OSS_ACTION_WEIGHT_RESIDENCY="0",
            RPU_WALL_OSS_ACTION_KV_OWNER_FANOUT="0",
            RPU_WALL_OSS_EXPERT0_DOWN_INT4="0",
        )
    if num_cores != 8:
        values.update(dict.fromkeys(_REDUCED_OFF, "0"))
    return values


def configure_exact_all_on(root: Path, args) -> dict[str, str]:
    unsafe = os.environ.get("RPU_CHUNK_FORCE_UNSAFE", "").strip()
    if unsafe not in ("", "0"):
        raise RuntimeError(
            "parcel-sorting validation rejects RPU_CHUNK_FORCE_UNSAFE; "
            "unset it before running this entrypoint"
        )
    values = exact_environment(root, args.mode, args.num_cores)
    libraries, _ = _LIBRARIES["verify_libraries"](root, {
        "RPU_BACKEND_SO": args.so,
        "RPU_KERNEL_LIB_PATH": args.main_ref,
        "RPU_KERNEL_LIB_PATH_EXPANSION": args.expansion,
    })
    values.update(libraries)
    # These historical global selectors are replaced by cold builder capabilities.
    os.environ.pop("RPU_RMSNORM_VWARP", None)
    os.environ.update(values)
    return values


def provenance(root: Path, rpu_backend, torch, selectors: dict[str, str]) -> dict:
    refs = {
        "main": str(Path(os.environ["RPU_KERNEL_LIB_PATH"]).resolve()),
        "expansion": str(
            Path(os.environ["RPU_KERNEL_LIB_PATH_EXPANSION"]).resolve()
        ),
    }
    paths = {"so": os.environ["RPU_BACKEND_SO"], **refs}
    return {"backend_root": str(root),
            "backend_head": subprocess.check_output(["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "backend_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "-c", f"safe.directory={root}", "-C", str(root), "diff", "HEAD"])).hexdigest(),
            "rpu_backend_file": str(Path(rpu_backend.__file__).resolve()),
            "torch_file": str(Path(torch.__file__).resolve()),
            "python": sys.executable, "selectors": {k: os.environ.get(k) for k in selectors},
            "native": {k: {"path": v, "sha256": hashlib.sha256(Path(v).read_bytes()).hexdigest()} for k, v in paths.items()},
            "loaded_libraries": sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if any(x in line for x in ("rpu_backend.so", "librhino", "libhxrpu"))})}


def cosine(torch, a, b) -> float:
    aa = a.detach().cpu().float().reshape(-1)
    bb = b.detach().cpu().float().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(aa, bb, dim=0))


def row_cosine_summary(torch, a, b) -> dict[str, float]:
    rows = torch.nn.functional.cosine_similarity(
        a.detach().cpu().float(), b.detach().cpu().float(), dim=-1
    ).reshape(-1)
    return {
        "mean": float(rows.mean()),
        "p01": float(torch.quantile(rows, 0.01)),
        "min": float(rows.min()),
    }


def cache_snapshot(vla) -> dict:
    result = {}
    for name, component in (
        ("vision", vla.vision), ("prefill", vla.llm), ("denoise", vla.action)
    ):
        cache = component._graph_cache
        entries = cache.snapshot()
        result[name] = {
            "size": cache.size(),
            "phase": cache.phase,
            "invariant": cache.cache_invariant_ok(),
            "replay": sum(int(entry.replay_count) for entry in entries),
            "recapture": sum(int(entry.recapture_count) for entry in entries),
        }
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def perf_summary(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "samples_ms": values,
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "p90_ms": percentile(values, 0.9),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def graph_gate(final: dict, prepared: dict, expected_counts: dict) -> bool:
    return all(
        item["phase"] == "READY"
        and item["invariant"]
        and item["recapture"] == 0
        and item["size"] == expected_counts[name]
        and item["replay"] > prepared[name]["replay"]
        for name, item in final.items()
    )


def numerical_metrics(torch, a, b):
    a, b = a.detach().cpu().float(), b.detach().cpu().float()
    assert a.shape == b.shape
    diff = a-b
    return {"cos": cosine(torch, a, b), "rows": row_cosine_summary(torch, a, b),
            "mse": float(diff.square().mean()), "mae": float(diff.abs().mean()),
            "max_abs": float(diff.abs().max()),
            "relative_l2": float(diff.norm()/b.norm().clamp_min(1e-20))}


def run(args: argparse.Namespace, fp16: Path, w8: Path, w4: Path, golden_path: Path):
    root = args.backend_root.resolve()
    selectors = configure_exact_all_on(root, args)
    site_packages = args.site_packages.resolve() if args.site_packages else None
    if site_packages is not None and not site_packages.is_dir():
        raise FileNotFoundError(site_packages)
    for env_name in (
        "RPU_KERNEL_LIB_PATH", "RPU_KERNEL_LIB_PATH_EXPANSION", "RPU_BACKEND_SO"
    ):
        selected_file = Path(os.environ[env_name]).resolve()
        if not selected_file.is_file():
            raise FileNotFoundError(f"{env_name}={selected_file}")
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if not finder.__class__.__module__.startswith("_rpu_backend_editable")
    ]
    sys.path[:0] = [str(root / "python")] + ([str(site_packages)] if site_packages else [])
    backend_spec = importlib.util.find_spec("rpu_backend")
    if (
        backend_spec is None
        or backend_spec.origin is None
    ):
        raise RuntimeError(f"conda/import overlay before import: {backend_spec}")
    _library_helpers.verify_backend_source(root, backend_spec.origin)
    import numpy as np
    import torch
    from PIL import Image
    import rpu_backend
    from rpu_backend.adapters.wall_oss import build_wall_oss_vla
    from rpu_backend.adapters.wall_oss.checkpoint_validation import (
        validate_rtc_w4_checkpoint_pair,
        validate_w8_nvfp4_checkpoint_pair,
    )

    loaded_native = {
        Path(line.split()[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if line.split()[-1].endswith("/rpu_backend.so")
    }
    if loaded_native != {args.so.resolve()}:
        raise RuntimeError(f"unexpected native extension mapping: {sorted(map(str, loaded_native))}")
    _library_helpers.verify_backend_source(root, rpu_backend.__file__)
    torch_origin = Path(torch.__file__).resolve()
    if site_packages is not None and not torch_origin.is_relative_to(site_packages):
        raise RuntimeError(
            f"conda/import overlay: torch={torch_origin}, expected below {site_packages}"
        )
    if w8 is not None:
        validate_w8_nvfp4_checkpoint_pair(w8, fp16)
    if w4 is not None:
        validate_rtc_w4_checkpoint_pair(w4, fp16)
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_ddr_flush(False)
    torch.rpu.set_ddr_flush_force(True)

    golden = torch.load(golden_path, map_location="cpu", weights_only=False)
    images = [
        Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
        for image in golden["images"]
    ]
    common = {
        "image": images,
        "instruction": golden["instruction"],
        "proprioception": golden["state"],
        "state_mask": golden["state_mask"],
        "noise": golden["noise"],
        "dof_mask": golden["dof_mask"],
        "num_steps": 6,
    }
    build_kwargs = {
        "dataset_key": "ex_normal",
        "camera_names": golden["camera_names"],
        "delta_action": True,
        "state_bins": 512,
        "max_seq_len": 2048,
        "rtc": True,
        "w8a16": args.mode != "fp16",
        "fp16_ckpt_dir": str(fp16),
        "action_w4a16_ckpt_dir": str(w4) if args.mode == "action-w4" else None,
        **({"rpu_execution": {"model": {"num_cores": args.num_cores}}}
           if args.num_cores != 8 else {}),
    }

    _LOG.info("building mode=%s", args.mode)
    vla = build_wall_oss_vla(str(fp16 if args.mode == "fp16" else w8), **build_kwargs)
    tensors = {"mode": args.mode, "delays": {}}
    try:
        if args.mode == "action-w4" and not (
            vla.action.llm is vla.llm and not vla.action._owns_llm
            and vla.action._precision_mode == 2 and vla._runtime_complete()
        ):
            raise AssertionError("Action-only W4 cold composition/shared-KV contract failed")

        if args.num_cores != 8:
            if (vla.vision._weight_outer or vla.vision._mixed_precision
                    or vla.llm._prefill_mixed_precision or vla.llm.residual_scale != 4):
                raise AssertionError("reduced-core ordinary residual-scaled RTC was not built")
        elif not (
            (bool(vla.vision._weight_outer) == (args.mode != "fp16" or bool(int(os.environ.get("RPU_WALL_OSS_FP16_TRANSFER_MASK", "0")) & 1)))
            and bool(vla.vision._mixed_precision)
            and bool(vla.llm._prefill_mixed_precision)
        ):
            raise AssertionError("W8 Vision/Prefill all-on contract was not built")

        cold_before = cache_snapshot(vla)
        cold = vla.predict(**common, use_rtc=True, debug=True)
        cold_tensors = {k: cold[k].detach().cpu().clone() for k in ("action", "x_norm", "merged", "prefix_last")}
        cold_built = cache_snapshot(vla)
        replay = vla.predict(**common, use_rtc=True, debug=True)
        cold_equal = {k: torch.equal(value.view(torch.uint8), replay[k].detach().cpu().view(torch.uint8)) for k,value in cold_tensors.items()}
        assert all(cold_equal.values()), cold_equal
        cold_replayed = cache_snapshot(vla)
        assert all(cold_before[k]["size"] == 0 and cold_built[k]["size"] == 1 and cold_replayed[k]["size"] == 1 and cold_replayed[k]["replay"] > cold_built[k]["replay"] and cold_replayed[k]["recapture"] == 0 for k in cold_before)
        graphs = []
        for name, model in (("vision", vla.vision), ("prefill", vla.llm), ("denoise", vla.action)):
            for index, entry in enumerate(model._graph_cache.snapshot()):
                graph = model._graph_cache._impl.lookup(entry.signature)
                item = {"component": name, "signature": str(entry.signature), "nodes": int(graph.graph_size()), "kernels": int(entry.kernel_count)}
                for kind, dump in (("nodes", graph.dump_nodes_jsonl), ("kernel_args", graph.dump_kernel_args_jsonl)):
                    path = args.output_dir / f"{name}_{index}_{kind}.jsonl"
                    path.write_text(dump())
                    item[kind] = str(path)
                graphs.append(item)
        # Both the graph wrapper and its last bound dump method retain the
        # native shared_ptr. Release them before prepare_graphs clears caches.
        del graph, dump, entry
        _LOG.debug("cold: equal=%s replayed=%s", cold_equal, cold_replayed)
        profile = vla.prepare_graphs(
            images=images,
            instruction=golden["instruction"],
            proprioception=golden["state"],
            state_mask=golden["state_mask"],
            noise=golden["noise"],
            dof_mask=golden["dof_mask"],
            prefix_length_range=(599, 640),
            num_steps=6,
        )
        prepared = cache_snapshot(vla)
        expected_counts = {"vision": 1, "prefill": 1, "denoise": 12}
        structural_errors = []
        if (
            profile.get("phase") != "READY"
            or not profile.get("ready_replay_validated")
            or profile.get("graph_counts") != expected_counts
            or tuple(profile.get("prefix_length_range", ())) != (599, 640)
        ):
            structural_errors.append(f"unexpected prepared profile: {profile}")

        cpu_metrics = {}
        hard_errors = list(structural_errors)
        for delay in DELAYS:
            out = vla.predict(
                **common,
                use_rtc=True,
                rtc_prefix_actions=(None if delay == 0 else golden["prefix_physical"]),
                inference_delay=delay,
                debug=True,
            )
            reference = golden["results"][str(delay)]
            finite_names = (
                ("action", "x_norm", "merged", "prefix_last")
                if delay == 0 else ("action", "x_norm")
            )
            finite = all(bool(torch.isfinite(out[name]).all()) for name in finite_names)
            hard_pin = delay == 0 or torch.equal(
                out["action"][:, :delay],
                golden["prefix_physical"][:delay].unsqueeze(0),
            )
            metrics = {
                "action": numerical_metrics(torch, out["action"], reference["action"]),
                "xnorm": numerical_metrics(torch, out["x_norm"], reference["x_norm"]),
                "finite": finite,
                "hard_pin_exact": hard_pin,
            }
            tensors["delays"][str(delay)] = {
                "action": out["action"].detach().cpu().clone(),
                "x_norm": out["x_norm"].detach().cpu().clone(),
            }
            if not finite or not hard_pin:
                hard_errors.append(f"D{delay} finite={finite} hard_pin={hard_pin}")
            if delay == 0:
                real = int(out["real_prefix_len"])
                prefix = out["prefix_last"][:, :real].detach().cpu().clone()
                metrics.update({
                    "merged_cos": cosine(torch, out["merged"], golden["merged"]),
                    "prefix_flat_cos": cosine(torch, prefix, golden["prefix_last"]),
                    "prefix_rows": row_cosine_summary(torch, prefix, golden["prefix_last"]),
                    "real_prefix_len": real,
                    "execution_len": int(out["prefix_len"]),
                })
                tensors["merged"] = out["merged"].detach().cpu().clone()
                tensors["prefix_last"] = prefix
                if real != 639 or int(out["prefix_len"]) != 640:
                    hard_errors.append(
                        f"unexpected REAL/P={real}/{int(out['prefix_len'])}"
                    )
            cpu_metrics[str(delay)] = metrics
            _LOG.info("numeric D%s: %s", delay, metrics)

        changed = dict(common, image=list(reversed(images)), noise=-golden["noise"])
        out_b = vla.predict(**changed, use_rtc=True, debug=True)
        tensors["changed_input"] = {k: out_b[k].detach().cpu().clone() for k in ("action", "x_norm", "merged", "prefix_last")}
        assert all(torch.isfinite(t).all() for t in tensors["changed_input"].values())
        assert not torch.equal(tensors["changed_input"]["action"], tensors["delays"]["0"]["action"])
        out_a = vla.predict(**common, use_rtc=True, debug=True)
        for key in ("action", "x_norm"):
            assert torch.equal(out_a[key].detach().cpu(), tensors["delays"]["0"][key]), ("A-B-A replay", key)
        current_provenance = provenance(root, rpu_backend, torch, selectors)
        for _ in range(args.warmup):
            vla.predict(**common, use_rtc=True, debug=False)
        e2e_samples = []
        d0_actions = []
        for index in range(args.repeat):
            started = time.perf_counter()
            out = vla.predict(**common, use_rtc=True, debug=False)
            elapsed = (time.perf_counter() - started) * 1000.0
            e2e_samples.append(elapsed)
            d0_actions.append(out["action"].detach().cpu().clone())
            _LOG.info("run=%s/%s e2e_ms=%.3f", index + 1, args.repeat, elapsed)
        d0_bitwise = all(torch.equal(d0_actions[0], value) for value in d0_actions[1:])
        if not d0_bitwise:
            hard_errors.append("D0 repeated output is not bitwise stable")

        action_samples = []
        host_samples = []
        os.environ["RPU_WALL_OSS_INSTRUMENT"] = "1"
        for index in range(args.repeat):
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                vla.predict(**common, use_rtc=True, debug=False)
            text = captured.getvalue()
            _LOG.debug("%s", text)
            action_match = re.search(
                r"9_denoise\.with_mask\(wrap\+hw\)=([0-9.]+)", text
            )
            host_match = re.search(r"host_sum=([0-9.]+)ms", text)
            if action_match is None or host_match is None:
                raise AssertionError(f"instrument parse failed on run {index + 1}")
            action_samples.append(float(action_match.group(1)))
            host_samples.append(float(host_match.group(1)))
        os.environ["RPU_WALL_OSS_INSTRUMENT"] = "0"

        final = cache_snapshot(vla)
        graph_ready = graph_gate(final, prepared, expected_counts)
        if not graph_ready:
            hard_errors.append(f"Graph READY/invariant/replay gate failed: {final}")

        report = {
            "mode": args.mode,
            "num_cores": args.num_cores,
            "formal_certification": False,
            "profile_enabled": False,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "cold_equal": cold_equal,
            "changed_image_noise_aba": True,
            "cold_lifecycle": {"before": cold_before, "built": cold_built, "replayed": cold_replayed},
            "graphs": graphs,
            "composition": (
                "W8 Vision + W8 expert0/shared-KV + W4 expert1 Action"
                if args.mode == "action-w4" else args.mode
            ),
            "provenance": current_provenance,
            "artifacts": {
                "fp16": str(fp16), "w8": str(w8), "w4": str(w4),
                "golden": str(golden_path),
            },
            "profile": profile,
            "cache_after_prepare": prepared,
            "cache_final": final,
            "cpu_golden": cpu_metrics,
            "timing": {
                "e2e": perf_summary(e2e_samples),
                "action_instrumented": perf_summary(action_samples),
                "host_instrumented": perf_summary(host_samples),
                "d0_bitwise": d0_bitwise,
            },
            "hard_errors": hard_errors,
        }
        if args.post_profile:
            (args.output_dir / "unprofiled.json").write_text(json.dumps(report, indent=2, default=list) + "\n")
            torch.save(tensors, args.output_dir / "unprofiled.pt")
            # Timing and the complete READY invariant snapshot above are already
            # finished. Explicitly thaw/clear only our caches for diagnostic
            # BUILD + REPLAY; no profiled calls enter headline samples.
            for component in (vla.vision, vla.llm, vla.action):
                component._graph_cache.begin_warmup()
                component._graph_cache.clear()
            torch.rpu.set_hw_perf_trace(True, str(args.output_dir / "hwperf"),
                                        6)
            profile_out = None
            for _ in range(2):
                profile_out = vla.predict(**common, use_rtc=True, debug=True)
            report["post_timing_profile"] = {
                "scope": "Explicit diagnostic rebuild after unprofiled timings and READY validation; excluded from headline",
                "cache": cache_snapshot(vla),
                "action_vs_unprofiled": numerical_metrics(torch, profile_out["action"], tensors["delays"]["0"]["action"])}
            assert torch.isfinite(profile_out["action"]).all()
            assert torch.equal(profile_out["action"], tensors["delays"]["0"]["action"])
            torch.rpu.set_hw_perf_trace(False, "", 0)
            from examples.runners.wall_oss.sorting_profile import summarize_profile
            report["device_profile"] = summarize_profile(args.output_dir)
        return report, tensors
    finally:
        vla.close()


def main() -> None:
    args = parse_args()
    fp16 = required_artifact(args.fp16, "WALL_RTC_GATE_FP16_CKPT")
    w8 = required_artifact(args.w8, "WALL_RTC_GATE_W8_CKPT") if args.w8 else None
    w4 = required_artifact(args.w4, "WALL_RTC_GATE_W4_CKPT") if args.w4 else None
    golden = required_artifact(args.golden, "WALL_RTC_GATE_GOLDEN")
    for name in ("so", "main_ref", "expansion"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.check_only:
        _LOG.info("configuration=%s", configure_exact_all_on(args.backend_root.resolve(), args))
        return
    if os.geteuid() != 0:
        raise RuntimeError("run under the project's exclusive device device-lock wrapper as root")
    from examples.common.logging import prepare_output_directory
    prepare_output_directory(args.output_dir)
    try:
        report, tensors = run(args, fp16, w8, w4, golden)
    except Exception as exc:
        failure_path = args.output_dir / f"{args.mode}.failure.json"
        failure_path.write_text(json.dumps({
            "mode": args.mode,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "artifacts": {
                "fp16": str(fp16), "w8": str(w8), "w4": str(w4),
                "golden": str(golden),
            },
        }, indent=2, sort_keys=True) + "\n")
        _LOG.error("failure artifact: %s", failure_path)
        raise

    import torch
    report["runtime_checks_passed"] = not report["hard_errors"]
    json_path = args.output_dir / f"{args.mode}.json"
    tensor_path = args.output_dir / f"{args.mode}.pt"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=list) + "\n")
    torch.save(tensors, tensor_path)
    _LOG.info("artifacts: %s %s", json_path, tensor_path)
    _LOG.info("runtime checks: %s", "PASS" if report["runtime_checks_passed"] else "FAIL")
    if report["hard_errors"]:
        raise AssertionError("; ".join(report["hard_errors"]))


if __name__ == "__main__":
    main()
