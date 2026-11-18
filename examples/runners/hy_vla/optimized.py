#!/usr/bin/env python3
"""HyVLA TOML execution behind the shared examples entry (one to three cameras, 4/6/8 cores)."""
from __future__ import annotations

from examples.common import cpu as _cpu_helpers
from examples.common import cold_libraries as _cold_libraries
from examples.common import results as _result_helpers

from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace


_LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
_CPU = vars(_cpu_helpers)
_LIBRARIES = vars(_cold_libraries)
PRECISIONS = ("fp16", "w8a16", "w8a16_nvfp4", "w4a16", "w4a16_vision")
PRECISION_ALIASES = {"w8a16_action_nvfp4": "w8a16_nvfp4"}
# Admission by the resolved per-module specs of the entry's precision recipe.
ROUTE_PROBES = (("vision",), ("vlm",), ("action",))
ROUTES = {("fp16",) * 3: "fp16", ("w8a16",) * 3: "w8a16", ("w8a16", "w8a16", "nvfp4"): "w8a16_nvfp4",
          ("w8a16", "w4a16-g32-rtn", "w4a16-g32-rtn"): "w4a16", ("w4a16-g32-rtn",) * 3: "w4a16_vision"}
DEFAULT_INSTRUCTION = (
    "Pick up the bottle from the table with the right gripper, keep it upright, and move it toward "
    "the empty target area. Avoid the cup and the other objects near the bottle. Use the left "
    "gripper to stabilize the target container if necessary. Lower the bottle gently into the "
    "target area, release the gripper when it is stable, and move both hands away from the objects. "
    "Keep the wrists level during the transfer and leave enough space between the arms. "
    "After placing the bottle, return the hands to a neutral position without moving the cup."
)


def canonical_precision(precision):
    return PRECISION_ALIASES.get(precision, precision)


def precision_route(precision):
    """The PRECISIONS mode the recipe behind PRECISION (a selector or alias) resolves to."""
    from examples.common.config import owner_route
    return owner_route("HY-VLA", "hy_vla", ROUTE_PROBES, ROUTES, selector=canonical_precision(precision))


# The action expert admits its resident and fused routes for prefix rows 128-240
# (optimized_prefix_supported in rpu_hyvla_expert_model.cpp). Two or three cameras
# always need at least 128 rows; one camera (at most 124 real rows) runs at that floor.
EXPERT_MIN_PREFIX = 128


def prefix_rows(cameras, text_tokens):
    return max(EXPERT_MIN_PREFIX, ((2 + cameras * 58 + text_tokens + 15) // 16) * 16)


def core_budget(config):
    """The cold rpu_execution.model.num_cores (8 when omitted)."""
    model = (config.get("rpu_execution") or {}).get("model", {})
    return model.get("num_cores", 8) if isinstance(model, dict) else 8


def validate_example_config(config):
    required = {"hy_vla", "input", "run"}
    if not required <= set(config) or set(config) - (required | {"libraries", "rpu_execution", "cpu", "logging"}):
        raise ValueError("HyVLA requires [hy_vla], [input], [run], optional [cpu]/[libraries]/[rpu_execution]/[logging]")
    from examples.common.config import validate_logging_config
    validate_logging_config(config)
    allowed = {
        "hy_vla": {"precision", "resident_fp8_scale_layers", "rmsnorm_vwarp"},
        "input": {"checkpoint", "source", "batch", "instruction", "cameras", "text_tokens", "prefix_len", "num_steps", "reference"},
        "run": {"warmup", "runs", "seed", "output_dir", "profile", "hwperf", "cpu_threads", "torch_num_threads"},
        "libraries": {"native", "main", "expansion"},
        "cpu": {"threads"},
    }
    for section, keys in allowed.items():
        table = config.get(section, {})
        if not isinstance(table, dict) or set(table) - keys:
            raise ValueError(f"[{section}] has unsupported fields")
    _CPU["resolve_cpu_threads"](config)
    precision = precision_route(config["hy_vla"].get("precision"))
    if config["hy_vla"].get("rmsnorm_vwarp", "auto") not in ("auto", "16"):
        raise ValueError('[hy_vla].rmsnorm_vwarp must be "auto" or "16"')
    execution = config.get("rpu_execution", {})
    if not isinstance(execution, dict) or set(execution) - {"model"}:
        raise ValueError("this HyVLA entry admits automatic component planning; only rpu_execution.model.num_cores")
    model = execution.get("model", {})
    if (not isinstance(model, dict) or set(model) - {"num_cores"}
            or type(model.get("num_cores", 8)) is not int or model.get("num_cores", 8) not in (4, 6, 8)):
        raise ValueError("[rpu_execution.model] accepts only num_cores = 4, 6 or 8")
    cores = core_budget(config)
    # At 4/6 cores the VLM runs on the whole budget and Vision/Action on four
    # cores; "auto" resolves to the BASE RMSNorm there because the vector
    # payloads and the resident FP8-scale layout are eight-core only.
    if cores != 8 and config["hy_vla"].get("rmsnorm_vwarp", "auto") != "auto":
        raise ValueError(f'[hy_vla].rmsnorm_vwarp = "16" is eight-core only; use "auto" at model.num_cores={cores}')
    inputs, run = config["input"], config["run"]
    if type(inputs.get("cameras")) is not int or inputs["cameras"] not in (1, 2, 3):
        raise ValueError("HyVLA requires cameras=1, 2 or 3")
    text = inputs.get("text_tokens")
    if type(text) is not int or not 1 <= text <= 64:
        raise ValueError("text_tokens must be an integer from 1 to 64: the checkpoint's tokenizer_max_length "
                         "is 64, and its preprocessing truncates the instruction to 64 language slots")
    minimum = prefix_rows(inputs["cameras"], text)
    maximum = prefix_rows(inputs["cameras"], 64)
    prefix = inputs.get("prefix_len", "auto")
    if prefix != "auto" and (type(prefix) is not int or prefix % 16 or not minimum <= prefix <= maximum):
        raise ValueError(f"prefix_len must be auto or a 16-row bucket covering this input, from {minimum} to {maximum}")
    scale_layers = config["hy_vla"].get("resident_fp8_scale_layers", 0)
    if type(scale_layers) is not int or scale_layers not in (0, 4, 8):
        raise ValueError("[hy_vla].resident_fp8_scale_layers must be integer 0, 4 or 8")
    if scale_layers and precision != "w8a16_nvfp4":
        raise ValueError(f"resident_fp8_scale_layers={scale_layers} requires Action NVFP4")
    if scale_layers and cores != 8:
        raise ValueError(f"resident_fp8_scale_layers is eight-core only; got model.num_cores={cores}")
    if type(inputs.get("num_steps", 10)) is not int or inputs.get("num_steps", 10) != 10:
        raise ValueError("HyVLA requires ten denoise steps")
    if not isinstance(inputs.get("checkpoint"), str) or not inputs["checkpoint"].strip():
        raise ValueError("[input].checkpoint must be a nonempty path or checkpoint alias")
    source = inputs.get("source", "batch" if "batch" in inputs else "synthetic")
    if source not in ("synthetic", "batch"):
        raise ValueError("[input].source must be synthetic or batch")
    if source == "batch":
        if not isinstance(inputs.get("batch"), str) or not inputs["batch"].strip():
            raise ValueError("[input].batch must name a saved observation for source=batch")
        if "instruction" in inputs:
            raise ValueError("source=batch takes its instruction from the saved batch")
    elif "batch" in inputs:
        raise ValueError("[input].batch requires source=batch")
    if "instruction" in inputs and (not isinstance(inputs["instruction"], str) or not inputs["instruction"].strip()):
        raise ValueError("[input].instruction must be a nonempty prompt")
    if "reference" in inputs and (not isinstance(inputs["reference"], str) or not inputs["reference"].strip()):
        raise ValueError("[input].reference must name a matching saved reference.pt")
    for key, default, minimum in (("warmup", 3, 3), ("runs", 6, 1), ("seed", 0, 0)):
        value = run.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"[run].{key} must be integer >= {minimum}")
    if not isinstance(run.get("output_dir"), str) or not run["output_dir"].strip():
        raise ValueError("[run].output_dir is required")
    for key in ("profile", "hwperf"):
        if type(run.get(key, False)) is not bool:
            raise ValueError(f"[run].{key} must be boolean")
    libraries = config.get("libraries")
    if libraries is not None and (set(libraries) != allowed["libraries"] or
            any(not isinstance(v, str) or not v.strip() for v in libraries.values())):
        raise ValueError("[libraries] must explicitly name native, main, and expansion together")


def example_arguments(config):
    validate_example_config(config)
    inputs, run = config["input"], config["run"]
    prefix = inputs.get("prefix_len", "auto")
    cores = core_budget(config)
    return SimpleNamespace(precision=precision_route(config["hy_vla"]["precision"]), checkpoint=inputs["checkpoint"],
        num_cores=cores,
        resident_fp8_scale_layers=config["hy_vla"].get("resident_fp8_scale_layers", 0),
        # "auto" (also when omitted) picks V16/V32 by geometry at eight cores and
        # the BASE route ("0") at 4/6, where the vector payloads have no owner.
        rmsnorm_vwarp=config["hy_vla"].get("rmsnorm_vwarp", "auto") if cores == 8 else "0",
        source=inputs.get("source", "batch" if "batch" in inputs else "synthetic"),
        batch=Path(inputs["batch"]).expanduser() if "batch" in inputs else None,
        instruction=inputs.get("instruction", DEFAULT_INSTRUCTION),
        cameras=inputs["cameras"], text_tokens=inputs["text_tokens"],
        prefix_len=prefix_rows(inputs["cameras"], inputs["text_tokens"]) if prefix == "auto" else prefix,
        warmup=run.get("warmup", 3), runs=run.get("runs", 6), seed=run.get("seed", 0),
        cpu_threads=_CPU["resolve_cpu_threads"](config, default=12),
        output_dir=Path(run["output_dir"]).expanduser(), profile=run.get("profile", False),
        hwperf=run.get("hwperf", False), libraries=config.get("libraries"),
        reference=Path(inputs["reference"]).expanduser() if "reference" in inputs else None,
        # Stage scopes stay automatic; the cold core budget is the only model scope.
        rpu_execution={**{stage: {"chunk_size": "auto"} for stage in ("vision", "prefill", "action")},
                       **({} if cores == 8 else {"model": {"num_cores": cores}})})


def registered_example(config):
    # The mixed precision and newly measured language inputs do not inherit the
    # immutable-v2 FP16 fixture's numerical/support status.
    validate_example_config(config)
    return None


def profile_environment(args):
    nv4 = canonical_precision(args.precision) == "w8a16_nvfp4"
    w4 = args.precision == "w4a16"
    w4v = args.precision == "w4a16_vision"
    env = {
        "RPU_HY_VLA_W8A16": "vit,vlm" if nv4 else "vit" if w4 else ("all" if args.precision == "w8a16" else "0"),
        "RPU_HY_VLA_W4A16": "vit,vlm,expert" if w4v else "vlm,expert" if w4 else "0",
        "RPU_HY_VLA_EXPERT_NVFP4": str(int(nv4)),
        "RPU_HY_VLA_ATTN_TP8": "1", "RPU_HY_VLA_ACTION_MLP_MC": "1",
        "RPU_HY_VLA_RMSNORM_PAD16": "expert", "RPU_HY_VLA_PREFILL_KV_ONLY": "1",
        "RPU_HY_VLA_VISION_KV_SPM_ONLY": "1", "RPU_HY_VLA_EXPERT_GATEUP_RESIDENT": "1",
        "RPU_HY_VLA_EXPERT_GATEUP_RESIDENT_LAYERS": str(24 if nv4 else (12 if args.precision == "w8a16" else 6)),
        "RPU_HY_VLA_EXPERT_O_RESIDENT_LAYERS": "0",
        "RPU_HY_VLA_EXPERT_RESIDENT_FP8_SCALE_LAYERS": str(args.resident_fp8_scale_layers),
        "RPU_RMSNORM_VWARP": args.rmsnorm_vwarp,
    }
    if w4 or w4v:
        # SPM-resident Gate/Up has FP16/W8/NVFP4 layouts only; packed W4 keeps
        # the generated auto-tile DDR Linear at every core count.
        env.update({"RPU_HY_VLA_EXPERT_GATEUP_RESIDENT": "0",
                    "RPU_HY_VLA_EXPERT_GATEUP_RESIDENT_LAYERS": "6"})
    if w4v:
        # Packed W4 Vision runs on four lanes under every budget (padded o/fc2
        # rows do not split into eight W4 cores), so its KV stays in DDR.
        env["RPU_HY_VLA_VISION_KV_SPM_ONLY"] = "0"
    if getattr(args, "num_cores", 8) != 8:
        # 4/6 cores: native KV4 attention, DDR Vision KV and no SPM-resident
        # expert weights (eight-core layouts); the Action MLP takes four cores.
        env.update({
            "RPU_HY_VLA_ATTN_TP8": "0", "RPU_HY_VLA_RMSNORM_PAD16": "0",
            "RPU_HY_VLA_VISION_KV_SPM_ONLY": "0", "RPU_HY_VLA_EXPERT_GATEUP_RESIDENT": "0",
            "RPU_HY_VLA_EXPERT_GATEUP_RESIDENT_LAYERS": "6",
        })
    return env


def digest(path):
    return _LIBRARIES["_digest"](path)


def bind_libraries(args, root=ROOT):
    return _LIBRARIES["bind_libraries"]({"libraries": args.libraries}, root)


def bind_cold_environment(env):
    if "rpu_backend" in sys.modules:
        raise ValueError("HyVLA configuration requires a fresh process before backend import")
    conflicts = [key for key, value in env.items() if key in os.environ and os.environ[key] != value]
    if conflicts:
        raise ValueError("conflicting cold environment: " + ", ".join(conflicts))
    if os.environ.get("RPU_CHUNK_FORCE_UNSAFE", "0") != "0":
        raise ValueError("RPU_CHUNK_FORCE_UNSAFE is incompatible with this entry")
    os.environ.update(env)


def runtime_payloads(libraries):
    from rpu_backend import _cpp_ext
    actual = _cpp_ext.get_runtime_payload_paths()
    for role, key in (("main", "RPU_KERNEL_LIB_PATH"), ("expansion", "RPU_KERNEL_LIB_PATH_EXPANSION")):
        path = actual.get("oplib_roles", {}).get(role)
        if path != libraries[key]["path"] or actual.get("oplib_payloads", {}).get(path, {}).get("sha256") != libraries[key]["sha256"]:
            raise RuntimeError(f"loaded {role} payload does not match configuration")
    extension = str(Path(actual["extension"]).resolve(strict=True))
    native = {"path": extension, "sha256": digest(extension)}
    if libraries.get("RPU_BACKEND_SO", native) != native:
        raise RuntimeError("loaded native extension does not match configuration")
    libraries["RPU_BACKEND_SO"] = native
    return actual


def language_fields(cameras, mask):
    """Use vendor masks, adding only inaccessible physical tail rows."""
    import torch
    from rpu_backend.adapters.hy_vla import preprocess as prep

    pad, att, modality, ranges, full = prep.prefix_masks(cameras, 7, mask)
    tail = prefix_rows(cameras, 64) - pad.shape[1]
    if tail:
        pad = torch.cat((pad, pad.new_zeros((1, tail))), 1)
        att = torch.cat((att, att.new_zeros((1, tail))), 1)
        modality = torch.cat((modality, modality.new_zeros((1, tail))), 1)
    prefill = prep.make_att_2d_masks(pad, att)
    prep.apply_visual_segment_mask(prefill, ranges, full)
    suffix_pad, suffix_att, _ = prep.suffix_masks()
    return {
        "prefill_mask": prefill[0], "prefill_pos": (pad.cumsum(1)-1)[0],
        "modality_mask": modality[0], "n_valid": int(pad.sum()),
        "denoise_mask": torch.cat((pad[:, None, :].expand(1, 51, pad.shape[1]),
            prep.make_att_2d_masks(suffix_pad, suffix_att)), 2)[0],
        "suffix_pos": (pad.sum(-1)[:, None] + suffix_pad.cumsum(1)-1)[0],
    }


def synthetic_language(tokenizer, instruction, text_tokens):
    """Truncate real prompt tokens; never repeat IDs or mark padding valid."""
    import torch
    from rpu_backend.adapters.hy_vla.preprocess import prepare_language

    short, short_mask = prepare_language(instruction, tokenizer, max_length=text_tokens)
    if int(short_mask.sum()) != text_tokens:
        raise ValueError("synthetic instruction must tokenize to at least text_tokens valid tokens")
    # Preserve the checkpoint's physical padding ID; padding is never valid.
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise ValueError("synthetic input requires the checkpoint tokenizer pad_token_id")
    tokens = torch.full((1, 64), pad_id, dtype=short.dtype)
    tokens[:, :text_tokens] = short
    mask = torch.arange(64)[None] < text_tokens
    return tokens[0], mask


def make_synthetic_batch(args, tokenizer):
    import torch
    from rpu_backend.adapters.hy_vla import preprocess as prep

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    # Keep the same sensors when changing only camera count or text length.
    frames = [torch.rand((1, 3, 224, 224), generator=generator) for _ in range(3)]
    state = torch.randn(20, generator=generator) * 0.05
    noise = torch.randn((1, 50, 32), generator=generator)
    tokens, mask = synthetic_language(tokenizer, args.instruction, args.text_tokens)
    _, full_mask = prep.prepare_language(args.instruction, tokenizer, max_length=65)
    observation = {
        "images": prep.prepare_images(frames[:args.cameras]),
        "state": prep.prepare_state(state), "noise": noise,
        "lang_tokens": tokens, **language_fields(args.cameras, mask),
    }
    return {"observation": observation, "instruction": args.instruction, "source": {
        "kind": "synthetic", "seed": args.seed,
        "camera_names": list(prep.CAMERA_ORDER[:args.cameras]),
        "images": "seeded synthetic RGB pixels; not robot camera captures",
        "state": "seeded synthetic normalized state; not a measured robot state",
        "tokenization": "checkpoint tokenizer truncation, then right padding to 64 slots",
        "tokenizer_max_length": args.text_tokens,
        "valid_text_tokens": int(mask.sum()),
        "actual_token_ids": tokens[:args.text_tokens].tolist(),
        "prompt_truncated": int(full_mask.sum()) > args.text_tokens,
        "assistant_suffix": "may be removed by prompt truncation",
        "cpu_reference": "not_generated", "task_quality": "not_evaluated",
    }}


def observation_digest(batch):
    """Content identity for generated inputs, independent of torch.save paths."""
    import torch

    checksum = hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            checksum.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
            checksum.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key, item in sorted(value.items()):
                checksum.update(json.dumps(key).encode())
                visit(item)
        elif isinstance(value, (tuple, list)):
            checksum.update(str(len(value)).encode())
            for item in value:
                visit(item)
        else:
            checksum.update(json.dumps(value, sort_keys=True).encode())
    visit({"instruction": batch["instruction"], "observation": batch["observation"]})
    return checksum.hexdigest()


def validate_observation(batch, args):
    import torch
    from rpu_backend.api.hy_embodied import _validate_runner_observation

    if not isinstance(batch, dict) or set(batch) - {"observation", "instruction", "source"}:
        raise ValueError("batch must contain observation, instruction and optional source metadata")
    if not isinstance(batch.get("instruction"), str) or not batch["instruction"].strip():
        raise ValueError("batch instruction is required to identify the actual text")
    if "source" in batch and not isinstance(batch["source"], dict):
        raise ValueError("batch source metadata must be a mapping")
    obs = batch.get("observation")
    if not isinstance(obs, dict):
        raise ValueError("batch observation must be a tensor mapping")
    cameras = getattr(args, "cameras", 3)
    _validate_runner_observation(obs, n_cameras=cameras, prefix_len=args.prefix_len)
    for value in [*obs["images"], *(v for v in obs.values() if torch.is_tensor(v))]:
        if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            raise ValueError("batch tensors must be finite and CPU-owned")
    if obs["lang_tokens"].dtype != torch.int64 or bool((obs["lang_tokens"] < 0).any()):
        raise ValueError("lang_tokens must be nonnegative int64 IDs")
    mask = torch.arange(64)[None] < args.text_tokens
    expected = language_fields(cameras, mask)
    if obs["n_valid"] != expected.pop("n_valid") or any(not torch.equal(obs[k], v) for k, v in expected.items()):
        raise ValueError("text_tokens must match actual right-padded language masks and all positions")
    return obs


def validate_text_tokens(batch, tokenizer, args=None):
    import torch
    from rpu_backend.adapters.hy_vla.preprocess import prepare_language
    cameras = getattr(args, "cameras", 3)
    source = batch.get("source", {})
    if source.get("kind") == "synthetic":
        text_tokens = source.get("tokenizer_max_length")
        if type(text_tokens) is not int or not 1 <= text_tokens <= 64 or text_tokens != getattr(args, "text_tokens", text_tokens):
            raise ValueError("saved synthetic tokenizer_max_length must match configured text_tokens")
        tokens, mask = synthetic_language(tokenizer, batch["instruction"], text_tokens)
    else:
        tokens, mask = prepare_language(batch["instruction"], tokenizer)
        tokens = tokens[0]
    obs = batch["observation"]
    if not torch.equal(tokens, obs["lang_tokens"]) or int(mask.sum()) != obs["n_valid"] - (2 + cameras * 58):
        raise ValueError("batch tokens/mask do not match its instruction and checkpoint tokenizer")


def graph_snapshot(runner):
    return {name: {"handle": sub.handle, "invariant": bool(sub.gc.cache_invariant_ok()),
        "entries": {str(entry.signature): {"replays": entry.replay_count, "recaptures": entry.recapture_count,
            "kernels": entry.kernel_count} for entry in sub.gc.snapshot()}} for name, sub in runner._psubs.items()}


def require_replay(before, after, count=1):
    if set(before) != {"vit", "vlm", "expert"} or set(before) != set(after):
        raise RuntimeError("HyVLA requires three retained Graph owners")
    for name, old in before.items():
        new = after[name]
        if old["handle"] != new["handle"] or not old["invariant"] or not new["invariant"] or not old["entries"] or old["entries"].keys() != new["entries"].keys():
            raise RuntimeError(f"{name}: Graph owner/layout changed")
        for signature, previous in old["entries"].items():
            current = new["entries"][signature]
            if current["recaptures"] or previous["recaptures"] or current["kernels"] != previous["kernels"] or current["kernels"] <= 0 or current["replays"] != previous["replays"] + count:
                raise RuntimeError(f"{name}: expected nontrivial stable REPLAY without recapture")


@contextmanager
def stage_timings(runner):
    phases, originals = {}, {}
    try:
        for name, label in (("_run_vit", "vision_ms"), ("_run_vlm_prefill", "prefill_ms"), ("_run_denoise", "denoise_ms")):
            original = getattr(runner, name)
            originals[name] = original
            def measured(*a, _fn=original, _label=label, **kw):
                started = time.perf_counter()
                result = _fn(*a, **kw)
                phases[_label] = (time.perf_counter()-started)*1000
                return result
            setattr(runner, name, measured)
        yield phases
    finally:
        for name, original in originals.items():
            setattr(runner, name, original)


def metrics(actual, reference):
    import torch
    if actual.shape != reference.shape or not torch.isfinite(reference).all():
        raise ValueError("reference must be finite and match normalized [50,20] output")
    a, b = actual.double(), reference.double()
    byte_equal = actual.dtype == reference.dtype and torch.equal(
        actual.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8))
    return {"cosine": torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
        "min_cosine": torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item(),
        "mse": (a-b).square().mean().item(), "max_abs": (a-b).abs().max().item(), "byte_equal": byte_equal}


def measure_policy(policy, obs, args, *, progress=None):
    import torch
    runner, settling, rows = policy._runner, [], []
    if progress is None:
        progress = {}
    progress.update(samples=rows, settling_graphs=settling)
    def infer(value):
        output = policy.infer(obs=value).actions_normalized.detach().cpu().clone()
        if tuple(output.shape) != (50, 20) or not bool(torch.isfinite(output).all()):
            raise RuntimeError("expected finite normalized actions [50,20]")
        return output
    with stage_timings(runner) as phases:
        for _ in range(args.warmup):
            infer(obs)
            settling.append(graph_snapshot(runner))
        require_replay(settling[-2], settling[-1])
        for sub in runner._psubs.values():
            sub.gc.freeze()
        before = graph_snapshot(runner)
        progress["graphs_before"] = before
        for _ in range(args.runs):
            phases.clear()
            start = time.perf_counter()
            output = infer(obs)
            elapsed = (time.perf_counter()-start)*1000
            if set(phases) != {"vision_ms", "prefill_ms", "denoise_ms"}:
                raise RuntimeError("incomplete HyVLA phase timings")
            rows.append({"total_ms": elapsed, **phases})
        after = graph_snapshot(runner)
        progress["graphs_after"] = after
        require_replay(before, after, args.runs)
        # Last measured A, changed-noise B, restored A. These two extra calls
        # are outside latency samples and use the same frozen owner signatures.
        alternate = dict(obs, noise=-obs["noise"])
        if torch.equal(alternate["noise"], obs["noise"]):
            raise ValueError("A/B/A requires a nonzero fixed noise tensor")
        changed = infer(alternate)
        restored = infer(obs)
        require_replay(after, graph_snapshot(runner), 2)
        if torch.equal(output, changed):
            raise RuntimeError("A/B/A: changed noise did not change the action output")
        if not torch.equal(output.contiguous().view(torch.uint8), restored.contiguous().view(torch.uint8)):
            raise RuntimeError("A/B/A: restored input did not restore the action bytes")
    report = {"samples": rows, "summary": {key: {"mean": statistics.mean(row[key] for row in rows),
        "median": statistics.median(row[key] for row in rows), "min": min(row[key] for row in rows),
        "max": max(row[key] for row in rows)} for key in rows[0]},
        "graphs_before": before, "graphs_after": after, "settling_graphs": settling,
        "aba": {"input_change": "negated fixed noise", "a_vs_b": metrics(output, changed), "a_restore": metrics(output, restored)},
        "execution_receipts": runner.last_rpu_execution_plan}
    return output, report


_RESULTS = vars(_result_helpers)


def run_example_config(config):
    args = example_arguments(config)
    env, libraries = bind_libraries(args)
    env.update(profile_environment(args))
    bind_cold_environment(env)
    effective_cpu = _CPU["configure_cpu_threads"](args.cpu_threads)
    import torch
    from rpu_backend.api import HyEmbodiedPolicy
    from rpu_backend.model_registry import model_path

    torch.manual_seed(args.seed)
    payloads = runtime_payloads(libraries)
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_dir():
        checkpoint = model_path(args.checkpoint)
    batch = torch.load(args.batch, map_location="cpu", weights_only=True) if args.batch is not None else None
    if batch is not None:
        validate_observation(batch, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"model": "hy_vla", "precision": args.precision, "text_tokens": args.text_tokens,
        "rmsnorm_vwarp": args.rmsnorm_vwarp,
        "language_slots": 64, "prefix_rows": args.prefix_len, "cameras": args.cameras, "num_steps": 10,
        "input_source": args.source,
        "libraries": libraries, "runtime_payloads": payloads, "runtime_env": env, "config": config,
        "rpu_execution": args.rpu_execution,
        "cpu_runtime": effective_cpu,
        "warmup": args.warmup, "runs": args.runs, "formal_certification": False,
        "cpu_reference": "not_run", "correctness": "not_run",
        "input_boundary": "preprocessed CPU observation; normalized actions; preprocessing and physical decoding excluded",
        "hwperf_enabled": args.hwperf}
    with _RESULTS["record_result"](report, args.output_dir,
        timing=lambda: _RESULTS["phase_timings"](report.get("samples", []),
            "host policy.infer and normalized CPU action snapshot; preprocessing, physical decoding, load and warmup excluded",
            phase_boundary="instrumented Vision/Prefill/Denoise host calls during measured inference; CPU action snapshot excluded"),
        runtime_checks=("finite normalized action output", "Graph REPLAY/no recapture", "noise A/B/A"),
        reference_keys=("reference",),
        output_names=("actions.pt", "reference.pt") + (("input_batch.pt",) if args.source == "synthetic" else ()) +
                     (("profile.trace.json",) if args.profile else ())):
        trace = torch.rpu.hw_perf_trace(str(args.output_dir / "hwperf"), max_dumps=32) if args.hwperf else nullcontext()
        policy = None
        with torch.inference_mode(), trace:
            try:
                policy = HyEmbodiedPolicy.from_checkpoint(checkpoint,
                    dtype={"w4a16_vision": "w4a16-vision"}.get(args.precision, args.precision)
                          if args.precision in ("fp16", "w4a16", "w4a16_vision") else "w8a16", prefix_len=args.prefix_len,
                    n_cameras=args.cameras, threads=args.cpu_threads, runtime_env=profile_environment(args), rpu_execution=args.rpu_execution)
                tokenizer = policy._tokenizer()
                if batch is None:
                    batch = make_synthetic_batch(args, tokenizer)
                obs = validate_observation(batch, args)
                validate_text_tokens(batch, tokenizer, args)
                synthetic = batch.get("source", {}).get("kind") == "synthetic"
                batch_sha = observation_digest(batch) if synthetic else digest(args.batch)
                report.update(instruction=batch["instruction"], batch_sha256=batch_sha, source=batch.get("source"),
                    valid_text_tokens=obs["n_valid"] - (2 + args.cameras * 58),
                    batch_identity="observation_content_sha256" if synthetic else "file_sha256")
                if args.batch is not None:
                    report["batch_file_sha256"] = digest(args.batch)
                if args.source == "synthetic":
                    torch.save(batch, args.output_dir / "input_batch.pt")
                policy.to("rpu")
                report["native_cold_config"] = asdict(policy._runner._native_cold)
                report["effective_rpu_environment"] = {key: value for key, value in os.environ.items() if key.startswith("RPU_")}
                output, measured = measure_policy(policy, obs, args, progress=report)
                report.update(measured)
                report["correctness"] = "runtime_checks_only"
                report["expert_nvfp4_metadata"] = vars(policy._runner.w.expert_nvfp4) if policy._runner.w.expert_nvfp4 is not None else None
                if args.reference is not None:
                    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
                    if not isinstance(reference, dict) or reference.get("batch_sha256") != batch_sha:
                        raise ValueError("reference input identity differs; never reuse the short-prompt CPU golden for other text")
                    report["reference"] = {"path": str(args.reference), "precision": reference.get("precision"),
                        "prefix_rows": reference.get("prefix_rows"), "metrics": metrics(output, reference["actions"])}
                if args.profile:
                    from rpu_backend.runtime.performance import _profile_ctx
                    with _profile_ctx(True) as profiler:
                        policy.infer(obs=obs)
                    profiler.export_chrome_trace(str(args.output_dir / "profile.trace.json"))
                torch.save(output, args.output_dir / "actions.pt")
                torch.save({"actions": output, "batch_sha256": batch_sha, "precision": args.precision,
                    "prefix_rows": args.prefix_len}, args.output_dir / "reference.pt")
            finally:
                if policy is not None:
                    _RESULTS["close_with_report"](policy, report)
        report["status"] = "completed"
    _LOG.debug("report=%s", report)
    return 0
