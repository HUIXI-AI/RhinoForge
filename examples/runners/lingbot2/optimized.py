"""LingBot v2 adapter examples with separate clean timing.

Configuration validation is stdlib-only. This entry calls the existing adapter
builder directly because its public-case mask contract differs from the older
exact-Z2 facade. It does not certify robot behavior or enable private kernels.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys
import time

from examples.common import cpu, libraries, results

_LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
COMPONENTS = {"vision_encoder": {"vision": ("chunk_size",)},
              "language_model": {"prefill": ("chunk_size", "padding_rows", "padding_budget")},
              "action_expert": {"action": ("chunk_size",)}}
STAGES = {"vision_ms": "_build_prefix", "prefill_ms": "_vlm_fill", "action_ms": "_denoise_unroll_run"}


def expanded_path(value, *, strict=True):
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if re.search(r"\$[A-Za-z_{]", expanded):
        raise ValueError(f"unresolved environment variable in path: {value}")
    return Path(expanded).resolve(strict=strict)


def execution_config(value):
    from examples.common.config import _backend_leaf
    normalized = _backend_leaf("api/_execution.py").normalize_rpu_execution(
        value, entry_point="LingBot v2 example", supported_components=COMPONENTS,
        supported={"model": ("num_cores",), "vision": ("chunk_size",),
            "prefill": ("chunk_size", "padding_rows", "padding_budget"), "action": ("chunk_size",)})
    def plain(value):
        from collections.abc import Mapping
        return {str(k): plain(v) for k, v in value.items()} if isinstance(value, Mapping) else value
    result = plain(normalized)
    cores = result.get("model", {}).get("num_cores", 8)
    if type(cores) is not int or cores not in (4, 6, 8):
        raise ValueError("[rpu_execution.model] accepts only num_cores = 4, 6 or 8")
    return result


# The W8A16 cold switches by the resolved per-module specs of the entry's precision recipe.
ROUTE_PROBES = (("vision",), ("vlm_prefill",), ("action",), ("action", "router"))
ROUTES = {("fp16",) * 4: False, ("w8a16", "w8a16", "w8a16", "fp16"): True}


def precision_w8(precision) -> bool:
    from examples.common.config import owner_route
    return owner_route("LingBot v2", "lingbot2", ROUTE_PROBES, ROUTES, selector=precision)


def core_budget(execution):
    """The cold rpu_execution.model.num_cores (8 when omitted)."""
    return execution.get("model", {}).get("num_cores", 8)


def default_cpu_threads(execution):
    """[cpu].threads when omitted: 1 at eight cores, 12 at 4/6 cores.

    Below eight cores the Vision merger LayerNorm and GELU stay on the host
    CPU. While the patch embed and merger GEMMs also ran there, one thread
    cost about 0.8 s per camera; the default is kept. An explicit
    [cpu].threads always wins.
    """
    return 1 if core_budget(execution) == 8 else 12


def validate_example_config(config):
    required = {"lingbot2", "input", "run"}
    if not isinstance(config, dict) or not required <= config.keys() or set(config) - (required | {"cpu", "rpu_execution", "logging"}):
        raise ValueError("LingBot v2 requires [lingbot2], [input], [run], optional [cpu]/[rpu_execution]/[logging]")
    from examples.common.config import validate_logging_config
    validate_logging_config(config)
    allowed = {"lingbot2": {"precision", "instruction_residency"},
        "input": {"checkpoint", "training_config", "case", "observation", "images", "instruction", "state", "noise", "cameras", "expected_language_tokens", "max_language_tokens"},
        "run": {"warmup", "runs", "seed", "output_dir", "profile", "hwperf"}}
    for section, keys in allowed.items():
        if not isinstance(config[section], dict) or set(config[section]) - keys:
            raise ValueError(f"[{section}] contains unsupported keys or is not a table")
    profile, inputs, run = (config[k] for k in ("lingbot2", "input", "run"))
    precision_w8(profile.get("precision"))
    if profile.get("instruction_residency", "auto") not in ("auto", "off"):
        raise ValueError("[lingbot2].instruction_residency must be auto or off")
    from .inputs import validate_input_config
    validate_input_config(inputs)
    for key, default, minimum in (("warmup", 2, 1), ("runs", 5, 1), ("seed", 0, 0)):
        value = run.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"[run].{key} must be an integer >= {minimum}")
    if not isinstance(run.get("output_dir"), str) or not run["output_dir"].strip():
        raise ValueError("[run].output_dir is required")
    for key in ("profile", "hwperf"):
        if type(run.get(key, False)) is not bool:
            raise ValueError(f"[run].{key} must be boolean")
    cpu.resolve_cpu_threads(config, default=default_cpu_threads(execution_config(config.get("rpu_execution"))))


def registered_example(config):
    validate_example_config(config)
    # The catalog's exact-Z2 mask/REAL215..225 profile is a different contract.
    return None


# Template selectors with eight-core owners only (runtime Graph, Vision batch,
# fused merger, device patch embed, direct prefix, fused AdaRMS broadcast). The
# adapter refuses them below eight cores; the remaining template selectors fall
# back or are core-agnostic, and the sparse expert route is required there.
EIGHT_CORE_SELECTORS = (
    "RPU_ADARMS_FUSED_BCAST", "RPU_QWEN3VL_VISION_BATCH", "RPU_QWEN3VL_VISION_FUSED_MERGER",
    "RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE", "RPU_LINGBOT2_RUNTIME_GRAPH",
    "RPU_LINGBOT2_VISION_DIRECT_PREFIX", "RPU_LINGBOT2_VISION_DIRECT_PREFIX_NO_OUTPUT")


def profile_environment(precision, cores=8):
    on = """RPU_FASTREPLAY_SKIP_SYNC RPU_KVINSERT_HYBRID_V16 RPU_ADARMS_FUSED_BCAST
RPU_QWEN3VL_VISION_ROPE_SPM RPU_QWEN3VL_VISION_BATCH RPU_QWEN3VL_VISION_FUSED_MERGER
RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE RPU_LINGBOT2_PREFILL_FAST_REPLAY
RPU_LINGBOT2_VISION_FAST_REPLAY RPU_LINGBOT2_DENOISE_UNROLL RPU_LINGBOT2_SPARSE_EXPERTS
RPU_LINGBOT2_RUNTIME_GRAPH RPU_LINGBOT2_FP16_TOP4 RPU_LINGBOT2_EXPERT_REPLAY
RPU_LINGBOT2_HOST_PREFIX_OPT RPU_LINGBOT2_VISION_DIRECT_PREFIX
RPU_LINGBOT2_VISION_DIRECT_PREFIX_NO_OUTPUT RPU_LINGBOT2_VLM_REPLAY RPU_LINGBOT2_ALLOW_UNVALIDATED""".split()
    off = """RPU_LINGBOT2_GROUPED_EXPERTS RPU_WALL_OSS_FAST_REPLAY RPU_L2_RCHUNK
RPU_LINGBOT2_EXPERT_W4A16 RPU_LINGBOT2_PREFILL_W4A16 RPU_LINGBOT2_LEGACY_PREPROC
RPU_L2_CAPTURE_L0 RPU_L2_CAPTURE_INNORM RPU_L2_CAPTURE_GATE RPU_L2_CAP_RESID
RPU_L2_STAGES RPU_L2_ROUTED_ONLY RPU_L2_DOWN_ACC16 RPU_L2_DBG_PACKED RPU_L2_SCHUNK
RPU_LINGBOT2_ROUTED_GEMM RPU_LINGBOT2_DEBUG_DENSE_SOFT_ROUTER
RPU_LINGBOT2_DEBUG_DUMP_ROUTER_H RPU_LINGBOT2_MULTIVIEW_SPM_Z2 RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE
RPU_DEEP_FAST_REPLAY RPU_KVINSERT_HYBRID3_V16 RPU_QWEN3VL_VISION_HOST_FP32_PATCH
RPU_LINGBOT2_PREFILL_KV_PAIR RPU_LINGBOT2_PREFILL_QK_NORM_ROPE
LKN_BATCH_INSTR_REUSE LKN_SKIP RPU_CHUNK_FORCE_UNSAFE""".split()
    env = dict.fromkeys(off, "0") | dict.fromkeys(on, "1")
    if cores != 8:
        env.update(dict.fromkeys(EIGHT_CORE_SELECTORS, "0"))
    w8 = precision_w8(precision)
    for name in ("EXPERT", "PREFILL", "BASE", "VISION"):
        env["RPU_LINGBOT2_" + name + "_W8A16"] = str(int(w8))
    env["RPU_WALL_OSS_FAST_REPLAY"] = str(int(not w8))
    env.update(LKN_MAX_BATCH_ENTRIES="131072", LKN_KD_BUF_MB="32", LKN_INSTR_BUF_MB="128",
               TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    return env


def profile_libraries():
    # Use exactly the checkout's two published REF roles and their manifest.
    return libraries.current_libraries(ROOT)


def bind_cold_environment(env):
    if "rpu_backend" in sys.modules:
        raise ValueError("LingBot v2 example requires a fresh process before importing rpu_backend")
    def same(key, value):
        if os.environ[key] == value: return True
        if key in {item[0] for item in libraries.LIBRARY_ROLES}:
            return expanded_path(os.environ[key]) == expanded_path(value)
        return False
    conflicts = [k for k, v in env.items() if k in os.environ and not same(k, v)]
    unknown = [k for k in os.environ if k.startswith(("RPU_LINGBOT2_", "RPU_L2_")) and k not in env]
    if conflicts or unknown:
        raise ValueError("conflicting or unsupported cold environment: " + ", ".join(sorted(set(conflicts + unknown))))
    os.environ.update(env)


def runtime_payloads(expected):
    from rpu_backend import _cpp_ext
    actual = _cpp_ext.get_runtime_payload_paths()
    for role, key in (("main", "RPU_KERNEL_LIB_PATH"), ("expansion", "RPU_KERNEL_LIB_PATH_EXPANSION")):
        path = actual.get("oplib_roles", {}).get(role)
        if (path != expected[key]["path"] or actual.get("oplib_payloads", {}).get(path, {}).get("sha256") != expected[key]["sha256"]):
            raise RuntimeError(f"loaded {role} library differs from selected role")
    path = Path(actual["launch"]).resolve(strict=True)
    loaded = {"path": str(path), "sha256": libraries.file_sha256(path)}
    actual["loaded_launch_identity"] = loaded
    actual["extension_sha256"] = libraries.file_sha256(actual["extension"])
    return actual


def metrics(actual, reference):
    import torch
    if actual.shape != reference.shape or not torch.isfinite(reference).all():
        raise ValueError("CPU reference shape/finite contract differs")
    a, b = actual.double().reshape(-1), reference.double().reshape(-1)
    d = a - b
    return {"max_abs": float(d.abs().max()), "mse": float(d.square().mean()),
        "relative_l2": float(d.norm() / b.norm().clamp_min(1e-30)),
        "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        "actual_rms": float(a.square().mean().sqrt()), "reference_rms": float(b.square().mean().sqrt())}


@contextmanager
def stage_timings(policy):
    samples = {key: [] for key in STAGES}
    originals = {}
    try:
        for key, name in STAGES.items():
            original = getattr(policy, name); originals[name] = original
            def timed(*a, _key=key, _original=original, **kw):
                begin = time.perf_counter()
                try: return _original(*a, **kw)
                finally: samples[_key].append((time.perf_counter() - begin) * 1000)
            setattr(policy, name, timed)
        yield samples
    finally:
        for name, original in originals.items(): setattr(policy, name, original)


def graph_stats(policy):
    caches = {"vision": policy._visual._rpu_vision_graph_cache, "prefill": policy._vgc, "action": policy._dgc}
    result = {}
    for name, cache in caches.items():
        if not cache.is_frozen() or not cache.cache_invariant_ok() or cache.size() <= 0:
            raise RuntimeError(f"{name} Graph not frozen or invalid")
        rows = []
        for entry in cache.snapshot():
            if entry.kernel_count <= 0 or entry.recapture_count:
                raise RuntimeError(f"{name} Graph empty or recaptured")
            rows.append({"replays": entry.replay_count, "recaptures": entry.recapture_count,
                "segments": entry.segment_count, "prepared_hits": entry.prepared_segment_hit_total,
                "prepared_misses": entry.prepared_segment_miss_total,
                "hw_batch_submits": entry.hw_batch_submit_total,
                "graph_lifetime_id": entry.graph_lifetime_id, "build_generation": entry.build_generation,
                "instruction_residency": {key: getattr(entry, key, None) for key in (
                    "batch_snapshot_mode", "batch_snapshot_available", "batch_snapshot_fallback_reason",
                    "batch_snapshot_effective_contracts")}})
        result[name] = rows
    return result


def check_replay(before, after):
    if before.keys() != after.keys(): raise RuntimeError("Graph stages changed")
    for stage, rows in before.items():
        if len(rows) != len(after[stage]): raise RuntimeError("Graph topology changed")
        for a, b in zip(rows, after[stage]):
            # OFF and conservative AUTO fallback may rebuild the shared queue's
            # prepared segment; that diagnostic miss is not a graph recapture.
            if (b["replays"] <= a["replays"] or b["hw_batch_submits"] <= a["hw_batch_submits"] or
                    b["recaptures"] or b["graph_lifetime_id"] != a["graph_lifetime_id"] or
                    b["build_generation"] != a["build_generation"]):
                raise RuntimeError(f"{stage} clean replay changed Graph lifetime/build or did not submit")


def load_runtime():
    from rpu_backend.adapters.lingbot_vla_v2 import runtime
    from rpu_backend.adapters.lingbot_vla_v2.prefix import AlignConfig
    return runtime, AlignConfig


def installed_precision_evidence(policy, profile_env):
    """Inspect existing Python owners only; no tensor reads, DMA or native calls.

    This records installed dtype metadata, not a kernel-dispatch/quality claim.
    Unknown owner layouts are reported instead of introducing a new run gate.
    """
    import dataclasses
    from collections.abc import Mapping
    from enum import Enum
    import torch

    def config_snapshot(value):
        # Execution configs implement Mapping but are intentionally immutable;
        # json's default=str would lose their contents and print object addresses.
        if isinstance(value, Enum):
            return config_snapshot(value.value)
        if isinstance(value, Mapping):
            return {str(key): config_snapshot(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [config_snapshot(item) for item in value]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        return {"unavailable_type": type(value).__name__}

    def tensors(values):
        values = list(values)
        found = [value for value in values if isinstance(value, torch.Tensor)]
        return {"entries": len(values), "tensors": len(found),
                "dtypes": sorted({str(value.dtype) for value in found}),
                "devices": sorted({str(value.device) for value in found}),
                "shapes": sorted({tuple(value.shape) for value in found})}

    def observe(reader):
        try:
            return reader()
        except Exception as error:
            return {"unavailable": type(error).__name__ + ": " + str(error)}

    cold = getattr(policy, "_lingbot2_moe_cold_config", None)
    text, vision = getattr(policy, "_text", None), getattr(policy, "_visual", None)
    result = {
        "scope": "Python installed-owner metadata; no tensor values or native dispatch inspection",
        "effective_env": {key: os.environ.get(key) for key in sorted(profile_env)},
        "moe_cold_config": dataclasses.asdict(cold) if dataclasses.is_dataclass(cold) else None,
        "resolved_execution_config": config_snapshot(getattr(policy, "_rpu_execution", None)),
        "resolved_component_configs": config_snapshot(getattr(policy, "_execution_components", None)),
        "flags": {"language_norm_folded": getattr(text, "_lingbot_runtime_norm_folded", None),
                  "vision_fused_merger": getattr(vision, "_rpu_vision_fused_merger", None),
                  "vision_fused_merger_w8a16": getattr(vision, "_rpu_vision_fused_merger_w8a16", None),
                  "vision_aux_w8a16": getattr(vision, "_rpu_vision_aux_w8a16", None),
                  "vision_patch_on_device": getattr(vision, "_rpu_patch_embed_on_device", None),
                  "direct_prefix": getattr(policy, "_vision_direct_prefix", None),
                  "direct_prefix_no_output": getattr(policy, "_vision_direct_prefix_no_output", None),
                  "denoise_unroll": getattr(policy, "_denoise_unroll", None)},
    }

    def language():
        names = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
        return {name: {"weight": tensors(layer.get_submodule(name).weight for layer in text.layers),
                       "scale": tensors(getattr(layer.get_submodule(name), "weight_scale", None)
                                        for layer in text.layers)} for name in names}

    def vision_encoder():
        # vision.py retains exactly the native set_weights arguments and scale
        # lists, then wraps that keepalive with the position-index owner.
        bundle = vision._rpu_vision_retirement_state.keepalive[0]
        args = bundle[0]
        names = ("q", "k", "v", "o", "fc1", "fc2")
        if len(args) != 22 or len(bundle) != 12:
            return {"unavailable": "unrecognized Vision native keepalive layout"}
        return {name: {"weight": tensors(args[index]), "scale": tensors(bundle[index + 1])}
                for index, name in enumerate(names)}

    def action():
        # convert.py documents the first twenty keepalive entries as stable;
        # Optional base scales, sparse shared-GU owners and routed banks follow.
        keep = policy._exp_keep
        base = {name: tensors(keep[index]) for name, index in
                (("q", 0), ("k", 1), ("v", 2), ("o", 3),
                 ("shared_gate", 9), ("shared_up", 10), ("shared_down", 11), ("router", 12))}
        sparse = getattr(cold, "sparse_experts", False)
        if sparse:
            names = ("gate_up", "down", "gate_up_scale", "down_scale")
        elif profile_env.get("RPU_LINGBOT2_GROUPED_EXPERTS") == "1":
            names = ("gate", "up", "down", "gate_scale", "up_scale", "down_scale")
        else:
            names = ()
        extra = len(keep) - 20 - len(names)
        if extra not in ((0, 2, 7, 9) if sparse else (0, 7)):
            return {"base_weights": base, "routed": {"unavailable": "unrecognized Action keepalive layout"}}
        base_quantized = extra in (7, 9)
        joint_shared = sparse and extra in (2, 9)
        routed = {name: tensors(keep[len(keep) - len(names) + index])
                  for index, name in enumerate(names)} if names else {
                      name: tensors(keep[index]) for name, index in
                      (("gate", 14), ("up", 15), ("down", 16))}
        result = {"base_weights": base, "routed": routed,
                "base_scales": {name: tensors(keep[20 + index]) for index, name in
                                enumerate(("q", "k", "v", "o", "shared_gate", "shared_up", "shared_down"))}
                               if base_quantized else {}}
        if joint_shared:
            offset = 20 + (7 if base_quantized else 0)
            result["shared_gate_up"] = {
                "weight": tensors(keep[offset]), "scale": tensors(keep[offset + 1])}
        return result

    result["language_projections"] = observe(language)
    result["vision_encoder_native_weights"] = observe(vision_encoder)
    result["action_native_weights"] = observe(action)
    result["vision_patch_weight"] = tensors([getattr(vision, "_rpu_patch_embed_w_rpu", None)])
    result["action_io_weights"] = observe(lambda: {
        name: tensors([policy._denoise_keep[index]]) for name, index in
        (("state", 0), ("wc", 2), ("mo", 3), ("output", 5))})
    result["limitations"] = (
        "Vision merger precision is an installer flag, not a native weight readback. "
        "Resident preload completion and selected kernel routes are checked separately by execution/Graph evidence. "
        "The full public W8 profile still retains FP16 Vision patch/merger and Action IO weights.")
    return result



def source_identity():
    from examples.common.config import ConfigError, _backend_leaf_path
    identity = {"git_head": None, "tracked_dirty": None}
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                check=True, capture_output=True, text=True).stdout
        identity.update(git_head=head, tracked_dirty=bool(status))
    except (OSError, subprocess.CalledProcessError) as error:
        # Source copies deployed to a board need not include Git metadata or git.
        identity["git_status_unavailable"] = type(error).__name__
    paths = (Path(__file__), Path(__file__).with_name("inputs.py"))
    identity["source_sha256"] = {str(p.resolve().relative_to(ROOT)): libraries.file_sha256(p) for p in paths}
    for relative in ("adapters/lingbot_vla_v2/runtime.py", "adapters/lingbot_vla_v2/prefix.py"):
        source = ROOT / "python/rpu_backend" / relative
        if source.is_file():
            identity["source_sha256"][str(source.relative_to(ROOT))] = libraries.file_sha256(source)
            continue
        try:
            installed = _backend_leaf_path(relative).resolve(strict=True)
            identity.setdefault("installed_backend_artifacts", {})[relative] = {
                "path": str(installed), "sha256": libraries.file_sha256(installed),
                "kind": "python_source" if installed.suffix == ".py" else "compiled_extension"}
        except (OSError, ConfigError) as error:
            identity.setdefault("unavailable_backend_sources", {})[relative] = str(error)
    return identity


def _diagnostics(policy, infer, first, config, output_dir, report):
    import torch
    run = config["run"]
    report["diagnostic_calls"] = 0
    if run.get("profile", False):
        from rpu_backend.runtime.performance import _profile_ctx
        with _profile_ctx(True) as profiler:
            value = infer()
        report["diagnostic_calls"] += 1
        if not torch.equal(value, first): raise RuntimeError("CPU profiler changed action output")
        profiler.export_chrome_trace(str(output_dir / "profile.trace.json"))
        (output_dir / "profile.txt").write_text(profiler.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=100))
    if run.get("hwperf", False):
        from rpu_backend.runtime.performance import _hwperf_aggregate_file, _hwperf_collect_files
        directory = output_dir / "hwperf"
        with torch.rpu.hw_perf_trace(str(directory), max_dumps=16):
            value = infer()
        report["diagnostic_calls"] += 1
        if not torch.equal(value, first): raise RuntimeError("HWF replay changed action output")
        report["hwperf"] = [_hwperf_aggregate_file(path) for path in _hwperf_collect_files(str(directory))]
        if not report["hwperf"] or any(item.get("error") for item in report["hwperf"]):
            raise RuntimeError("requested HWF traces are missing or unreadable")
    if report["diagnostic_calls"]:
        report["graphs_after_diagnostics"] = graph_stats(policy)
    report["diagnostic_scope"] = "separate calls after clean timing; HWF may rebuild prepared segments; excluded from clean samples"


def run_example_config(config):
    validate_example_config(config)
    profile, inputs, run = (config[k] for k in ("lingbot2", "input", "run"))
    output_dir = expanded_path(run["output_dir"], strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    execution = execution_config(config.get("rpu_execution"))
    report = {"model": "lingbot2", "precision": profile["precision"],
        "entry_point": "rpu_backend.adapters.lingbot_vla_v2.runtime.build_lingbot_vla_v2",
        "registered_profile": None, "support_scope": "public generic controlled evaluation; not the exact-Z2 catalog profile",
        "formal_certification": False, "correctness": "not_run", "num_steps": 10,
        "top_k": 4, "routed_experts": 32, "action_layers": 36, "num_cores": core_budget(execution),
        "rpu_execution": execution, "warmup": run.get("warmup", 2), "runs": run.get("runs", 5),
        "seed": run.get("seed", 0), "samples_ms": [], "profile": run.get("profile", False),
        "hwperf_enabled": run.get("hwperf", False), "cpu_reference": {},
        "timing_scope": "host get_action -> float -> CPU clone; load, prepare, warmup and diagnostics excluded",
        "numerical_notes": "CPU FP32 differences are diagnostics, without a cosine acceptance floor; robot task quality is not evaluated"}
    owner = None
    with results.record_result(report, output_dir,
        timing=lambda: results.phase_timings(report["samples_ms"], report["timing_scope"],
            phase_boundary="existing synchronous Vision/prefix, language prefill and native Action host methods; CPU output copy excluded"),
        runtime_checks=("finite full action shape [1,50,55]", "same-runtime warm/measurement exact replay",
                        "adapter prepare A/B/A and READY validation", "frozen nonempty Graph replay/submission without recapture or generation changes"),
        reference_keys=("cpu_reference",),
        output_names=("actions.pt",) + (("profile.trace.json", "profile.txt") if run.get("profile", False) else ())):
        try:
            hwperf_dir = output_dir / "hwperf"
            if run.get("hwperf", False) and hwperf_dir.exists() and any(hwperf_dir.iterdir()):
                raise ValueError("requested HWPerf directory is nonempty; choose a fresh output_dir")
            env = profile_environment(profile["precision"], core_budget(execution))
            lib_env, bound = profile_libraries(); env.update(lib_env)
            bind_cold_environment(env)
            report.update(effective_env=dict(env), libraries=bound,
                library_binding="current_libraries: exactly two manifest-published REF roles",
                instruction_residency=profile.get("instruction_residency", "auto"))
            report["source_identity"] = source_identity()
            report["cpu_runtime"] = cpu.configure_cpu_threads(cpu.resolve_cpu_threads(config, default=default_cpu_threads(execution)))
            import torch
            torch.manual_seed(report["seed"])
            from .inputs import load_inputs
            started = time.perf_counter()
            obs, golden, assets, align_values = load_inputs(inputs, seed=report["seed"])
            report["input_prepare_ms"] = (time.perf_counter() - started) * 1000
            report["input_timing_scope"] = "CPU input loading/tokenization/image processing performed once before model loading; excluded from clean inference"
            report["assets"] = assets
            runtime, AlignConfig = load_runtime()
            report["runtime_payloads"] = runtime_payloads(bound)
            from rpu_backend.graph import GraphRuntimePolicy
            graph_policy = GraphRuntimePolicy.from_environment(batch_snapshot_mode=profile.get("instruction_residency", "auto"))
            started = time.perf_counter()
            # The adapter reads the cold model.num_cores itself; an explicit
            # 8 is the default and is dropped so the eight-core request is unchanged.
            adapter_execution = ({k: v for k, v in execution.items() if k != "model"}
                                 if core_budget(execution) == 8 else execution)
            report["adapter_rpu_execution"] = adapter_execution
            owner = runtime.build_lingbot_vla_v2(assets["checkpoint"],
                training_config_path=assets["training_config"],
                align_cfg=AlignConfig(**align_values),
                rpu_execution=adapter_execution, graph_runtime_policy=graph_policy)
            report["load_ms"] = (time.perf_counter() - started) * 1000
            report["installed_precision"] = installed_precision_evidence(owner, env)
            geometry = owner._request_prefix_geometry(obs["images"], obs["img_masks"],
                obs["lang_tokens"], obs["lang_masks"], obs["image_grid_thw"])
            report["prefix_geometry"] = {k: v for k, v in geometry.items() if k != "lang_ids"}
            started = time.perf_counter()
            prepared = owner.prepare_graphs(obs, prefix_length_range=(geometry["real_len"], geometry["real_len"]))
            report["prepare_ms"] = (time.perf_counter() - started) * 1000
            if not isinstance(prepared, dict) or prepared.get("phase") != "READY" or not prepared.get("ready_replay_validated"):
                raise RuntimeError("adapter prepare did not validate A/B/A and READY")
            report["graph_prepare"] = {key: prepared.get(key) for key in (
                "phase", "ready_replay_validated", "graph_counts", "example_real_prefix_len", "num_steps")}
            report["graphs_before"] = graph_stats(owner)
            first = None
            def infer():
                value = owner.get_action(obs).float().cpu().clone()
                if value.shape != (1, 50, 55) or not torch.isfinite(value).all():
                    raise RuntimeError("LingBot v2 returned invalid full action output")
                return value
            with torch.no_grad():
                for _ in range(report["warmup"]):
                    value = infer()
                    if first is None: first = value
                    elif not torch.equal(first, value): raise RuntimeError("warm replay changed actions")
                with stage_timings(owner) as phases:
                    for index in range(report["runs"]):
                        started = time.perf_counter(); value = infer()
                        total = (time.perf_counter() - started) * 1000
                        if not torch.equal(first, value): raise RuntimeError("measured replay changed actions")
                        if any(len(rows) != index + 1 for rows in phases.values()):
                            raise RuntimeError("incomplete measured stage boundaries")
                        report["samples_ms"].append({"total_ms": total, **{k: rows[-1] for k, rows in phases.items()}})
                report["graphs_after"] = graph_stats(owner)
                check_replay(report["graphs_before"], report["graphs_after"])
                if golden is not None:
                    report["cpu_reference"] = {"metrics": metrics(first, golden["normalized_actions"]),
                        "valid16_metrics": metrics(first[..., :16], golden["normalized_actions"][..., :16]),
                        "policy": "record-only CPU FP32 comparison; no inferred model/task acceptance"}
                report["execution_plan"] = owner._last_execution_plan
                report["output_shape"] = list(first.shape)
                torch.save(first, output_dir / "actions.pt")
                report["mean_ms"] = {k: statistics.mean(row[k] for row in report["samples_ms"]) for k in report["samples_ms"][0]}
                report["median_ms"] = {k: statistics.median(row[k] for row in report["samples_ms"]) for k in report["samples_ms"][0]}
                report["runtime_calls"] = {"prepare": 5, "warmup": report["warmup"], "measured": report["runs"]}
                report["ddr_flush_enabled"] = torch.rpu.get_ddr_flush()
                report["ddr_flush_force_enabled"] = torch.rpu.get_ddr_flush_force()
                _diagnostics(owner, infer, first, config, output_dir, report)
                report["runtime_calls"]["diagnostic"] = report["diagnostic_calls"]
                report["runtime_calls"]["total"] = sum(report["runtime_calls"].values())
                report["correctness"] = "runtime_checked_reference_measured" if golden is not None else "runtime_checked_no_reference"
        finally:
            results.close_with_report(owner, report)
    _LOG.debug("report=%s", report)
    return 0
