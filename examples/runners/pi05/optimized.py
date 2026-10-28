#!/usr/bin/env python3
"""Shared Pi0.5 example engine and historical diagnostic CLI.

T64, T96 and T128 are cold two/three-camera opt-ins.
T32 remains the default.

Use a fresh process for each checkpoint/precision. See examples/configs/pi05/README.md.
"""
from __future__ import annotations

from examples.common import cpu as _cpu_helpers
from examples.common import libraries as _library_helpers
from examples.common import results as _result_helpers

import argparse
from contextlib import contextmanager, nullcontext
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

_CPU = vars(_cpu_helpers)

_PRECISIONS = ("fp16", "w8a16", "w4a16", "int4_g128", "legacy_int4")
_EXAMPLE_PRECISIONS = ("fp16", "w8a16", "w4a16_kvint8", "w8a16_action_nvfp4", "w8a16_prefill_w8a8_action_nvfp4")
_REGISTERED_ASSETS = {
    "fp16": "pi05-libero-finetuned",
    "w8a16": "pi05-libero-finetuned-w8a16-vlm-expert",
    "w4a16_kvint8": "pi05-libero-finetuned-w4real-kvint8",
}
_ON = """
RING_XOR3 K_ROPE_INSERT KVINSERT_PAD16 DENOISE_UNROLL PREFILL_MASK_FP16 PREFILL_KV_ONLY
DENOISE_XOR3_GATED_MLP VISION_MERGE_SEGMENT VISION_Q_RESIDENT PREFILL_CARRY_AB
PREFILL_MASK_RESIDENT PREFILL_DOWN_PAIR PREFILL_OWNER_NORM PREFILL_O_PAIR
DENOISE_XOR3_GATED_ATTN VISION_KV_SPM_ONLY DENOISE_KV1_DIRECT_CACHE
DENOISE_KV1_PAIR_OWNER PREFILL_KV1_PAIR_DIRECT PREFILL_GATEUP_PAIR FUSED_DENOISE SIGLIP_BATCH
""".split()

# Generic routes retained for legacy INT4 two-camera experiments.
_GENERIC_ON = frozenset("""
KVINSERT_PAD16 DENOISE_UNROLL PREFILL_MASK_FP16 FUSED_DENOISE SIGLIP_BATCH
""".split())
_OFF = """
GEGLU_PREFETCH SDPA_SINGLE_GROUP VISION_FC_WEIGHT_OUTER PREFILL_CARRY_A
PREFILL_ROPE_CACHE DENOISE_GATEUP_RESIDENT DENOISE_ELIDE_UNUSED_DMA
DENOISE_SPM_ACTION_STAGING DENOISE_Q_ROPE_EPILOGUE PREFILL_O_CHAIN
PREFILL_QKV_PAIR DENOISE_ADARMS_BODY_STREAM DENOISE_GATEUP_RESIDENT_LAYERS
DENOISE_KV1_GATHER
DENOISE_NVFP4_GEGLU_ACC16_M50
""".split()


_LOG = logging.getLogger(__name__)

def profile_environment(
    precision: str, *, prefill_w8a8: bool | None = None,
    vision_owner_ln: bool | None = None, cameras: int = 3, text_tokens: int = 32,
) -> dict[str, str]:
    """Return cold settings; explicit opt-ins default off in the CLI.

    None retains compatibility with existing one-argument experiment callers
    that bind these two flags separately. It never enables either flag.
    """
    if precision not in _PRECISIONS:
        raise ValueError("precision must be fp16, w8a16, w4a16 (NVFP4), int4_g128, or legacy_int4")
    if type(cameras) is not int or cameras not in (2, 3):
        raise ValueError("cameras must be 2 or 3")
    _validate_text_profile(text_tokens, cameras=cameras, precision=precision)
    for name, value in (("prefill_w8a8", prefill_w8a8),
                        ("vision_owner_ln", vision_owner_ln)):
        if value is not None and type(value) is not bool:
            raise TypeError(f"{name} must be bool or None")
    if prefill_w8a8 and precision != "w4a16":
        raise ValueError("Prefill W8A8 is admitted only with Action NVFP4 (--precision w4a16)")
    result = {"RPU_PI05_" + name: str(int(cameras == 3 or precision in ("fp16", "w8a16", "w4a16") or name in _GENERIC_ON))
              for name in _ON}
    result.update({"RPU_PI05_" + name: "0" for name in _OFF})
    # Pin the two on-load quantizers so a nominal FP16 run cannot inherit W8.
    for name in ("SIGLIP_W8A16", "ADARMS_DENSE_W8A16"):
        result["RPU_PI05_" + name] = str(int(precision != "fp16"))
    result["RPU_PI05_SIGLIP_W8A16_SCOPE"] = "all"
    result["RPU_PI05_DENOISE_GEGLU_M50"] = str(int(precision in ("fp16", "w8a16")))
    for name in ("DENOISE_W4_FASTPATH", "DENOISE_W4_GEGLU_M50", "DENOISE_W4_GEGLU_N64"):
        result["RPU_PI05_" + name] = str(int(precision in ("int4_g128", "legacy_int4") and cameras == 3))
    result["RPU_PI05_DENOISE_NVFP4_GEGLU_M50"] = "0"
    if precision == "w4a16":
        # Preserve NVFP4 storage with W8 K/V; the cold action config selects
        # ACC16/ACC32. run_profile enables the uniform ACC16 GeGLU fusion only
        # after resolving the Action component's cold accumulation setting.
        result["RPU_PI05_DENOISE_W4_FASTPATH"] = "1"
    for name, value in (("PREFILL_GATEUP_W8A8", prefill_w8a8),
                        ("VISION_OWNER_LN", vision_owner_ln)):
        if value is not None or precision == "fp16" or cameras == 2:
            result["RPU_PI05_" + name] = str(int(bool(value)))
    return result


def _validate_text_profile(text_tokens: int, *, cameras: int, precision: str | None = None) -> None:
    if type(text_tokens) is not int or text_tokens not in (32, 64, 96, 128, 160):
        raise ValueError("text_tokens must be 32, 64, 96, 128, or 160")
    if text_tokens != 32:
        if cameras not in (2, 3):
            raise ValueError(f"T{text_tokens} requires exactly two or three cameras")
        if precision is not None and precision not in ("fp16", "w8a16", "w4a16"):
            raise ValueError(f"T{text_tokens} requires FP16, W8A16, or Action NVFP4 precision")


def _cpu_interop_threads(*, paired: bool, precision: str, prefill_w8a8: bool,
                         cameras: int, text_tokens: int) -> int:
    return 1 if (paired and precision == "w4a16" and prefill_w8a8
                 and cameras == 3 and text_tokens == 160) else 24


def resolve_prefill_chunk_size(*, cameras: int, prefill_w8a8: bool, requested: int | None,
                               text_tokens: int = 32) -> int:
    """Bind the selected A8 ABI to its exact chunk before importing the backend."""
    if type(cameras) is not int or cameras not in (2, 3):
        raise ValueError("cameras must be 2 or 3")
    if type(prefill_w8a8) is not bool:
        raise TypeError("prefill_w8a8 must be bool")
    _validate_text_profile(text_tokens, cameras=cameras)
    expected = (cameras * 256 + text_tokens) // 2
    chunk = expected if requested is None else requested
    if type(chunk) is not int or chunk <= 0 or chunk % 16:
        raise ValueError("--prefill-chunk-size must be a positive multiple of 16")
    if (prefill_w8a8 or text_tokens != 32) and chunk != expected:
        route = f"T{text_tokens}" if text_tokens != 32 else "Prefill W8A8"
        raise ValueError(f"{cameras}-camera {route} requires exactly --prefill-chunk-size {expected}")
    return chunk


def validate_checkpoint(config: dict, precision: str) -> None:
    if precision == "fp16":
        if config:
            raise ValueError("requested fp16 requires an unquantized checkpoint without rpu_quant_config")
        return
    method = {"w8a16": "w8a16", "w4a16": "nvfp4a16",
              "int4_g128": "w4a16", "legacy_int4": "w4a16"}[precision]
    if config.get("method") != method:
        raise ValueError(f"requested {precision} requires method={method}, checkpoint method={config.get('method')!r}")
    if precision != "w8a16":
        keep = {name.removesuffix(".weight").rsplit(".", 1)[-1]
                for name in config.get("mixed_int8_suffixes", [])}
        if config.get("int4_components") != ["expert"] or keep != {"k_proj", "v_proj"}:
            raise ValueError("optimized W4 requires Action-only Q/O/Gate/Up/Down; K/V and VLM W8")
        if precision == "w4a16":
            if config.get("nvfp4_abi") != "striped_v2" or config.get("nvfp4_block_size") != 16:
                raise ValueError("w4a16 defaults to NVFP4 striped_v2 block16, not legacy INT4")
        elif (config.get("int4_group_size") != 128 or
              (config.get("mode") == "group_symmetric") != (precision == "int4_g128")):
            raise ValueError("INT4 group metadata does not match requested precision")


def validate_camera_batch(batch: dict, image_features, *, cameras: int, text_tokens: int = 32) -> None:
    """Require the selected physical token width and configured image slots."""
    import torch

    if type(cameras) is not int or cameras not in (2, 3):
        raise ValueError("cameras must be 2 or 3")
    _validate_text_profile(text_tokens, cameras=cameras)
    configured = set(image_features or ())
    present = {key for key in batch if key.startswith("observation.images.")
               and not key.endswith("_padding_mask")}
    label = "three" if cameras == 3 else "two"
    if len(configured) != cameras or present != configured:
        raise ValueError(f"this Pi0.5 entry requires all {label} configured cameras and no additional image slots")
    if cameras == 2 and any(key.rsplit(".", 1)[-1].startswith("empty_camera_") for key in configured):
        raise ValueError("two-camera input requires two real VISUAL features, without empty_camera slots")
    orphan_masks = {key.removesuffix("_padding_mask") for key in batch
                    if key.startswith("observation.images.") and key.endswith("_padding_mask")} - configured
    if orphan_masks:
        raise ValueError("image padding masks must refer to configured cameras")
    for key in configured:
        image = batch[key]
        if not torch.is_tensor(image) or image.ndim != 4 or tuple(image.shape[:2]) != (1, 3):
            raise ValueError(f"{key} must be a batch-one [1,3,H,W] image tensor")
        mask = batch.get(key + "_padding_mask")
        if mask is not None and (not torch.is_tensor(mask) or mask.dtype != torch.bool
                                 or tuple(mask.shape) != (1,) or not bool(mask.all())):
            raise ValueError(f"all {label} cameras must have present (true) padding masks")
    tokens = batch.get("observation.language.tokens")
    if not torch.is_tensor(tokens) or tuple(tokens.shape) != (1, text_tokens):
        raise ValueError(f"this Pi0.5 entry requires exactly {text_tokens} language tokens")
    if text_tokens != 32:
        mask = batch.get("observation.language.attention_mask")
        if (not torch.is_tensor(mask) or mask.dtype != torch.bool or tuple(mask.shape) != (1, text_tokens)
                or not bool(mask.any()) or bool((~mask[:, :-1] & mask[:, 1:]).any())):
            raise ValueError(f"T{text_tokens} requires a nonempty boolean [1,{text_tokens}] language mask with right padding")


def validate_three_camera_batch(batch: dict, image_features) -> None:
    """Compatibility entry for existing P800 experiment probes."""
    validate_camera_batch(batch, image_features, cameras=3)


def _require_cold_shared_policy(policy, scope: str):
    lerobot = policy._lerobot_policy
    adapter = policy._adapter
    model = lerobot.model
    config = lerobot.config
    if adapter._lerobot_policy is not lerobot or model.config is not config:
        raise ValueError(f"{scope} configuration requires one shared cold policy/model config")
    for owner, flags in ((policy, ("_rpu_ready",)), (adapter, ("_rpu_is_ready",)),
                         (lerobot, ("_rpu_swizzled", "_rpu_swizzle_started"))):
        if any(getattr(owner, flag, False) for flag in flags):
            raise ValueError(f"{scope} configuration must precede device installation/swizzle")
    # A partially installed or already-used CPU wrapper is not a cold model.
    installed = ("_rpu_vision_handle", "_rpu_vlm_decoder_handle", "_rpu_action_handle",
                 "_rpu_fused_denoise_handle", "_rpu_siglip_graph_cache",
                 "_rpu_gemma_graph_cache", "_rpu_adarms_graph_cache",
                 "_rpu_fused_denoise_graph_cache", "_rpu_cache", "_rpu_siglip_batch_n",
                 "_rpu_last_execution_plan")
    if any(getattr(module, name, None) is not None
           for module in lerobot.modules() for name in installed):
        raise ValueError(f"{scope} configuration requires a cold model without installed handles/caches")
    if any(parameter.device.type != "cpu" for parameter in lerobot.parameters()):
        raise ValueError(f"{scope} configuration requires CPU checkpoint weights before installation")
    return lerobot, adapter, config


def configure_text_profile(policy, text_tokens: int) -> dict:
    """Bind the physical text bucket on the adapter before swizzle/installation.

    The checkpoint tokenizer_max_length may be 200; it never selects this
    execution profile. A prepared policy cannot change its bucket in place.
    """
    lerobot, adapter, config = _require_cold_shared_policy(policy, "text-profile")
    _validate_text_profile(text_tokens, cameras=len(config.image_features))
    quant = getattr(lerobot, "_pi05_quant_config", {}) or {}
    if text_tokens != 32 and quant.get("method") not in (None, "w8a16", "nvfp4a16"):
        raise ValueError(f"T{text_tokens} requires FP16, W8A16, or Action NVFP4 precision")
    previous = getattr(adapter, "_rpu_prefill_text_tokens", 32)
    adapter._rpu_prefill_text_tokens = text_tokens
    return {"previous_text_tokens": previous, "text_tokens": text_tokens,
            "prefill_pair_rows": (256 * len(config.image_features) + text_tokens) // 2,
            "configured_before_rpu_install": True}


def configure_two_camera_policy(policy, batch: dict, *, text_tokens: int = 32) -> dict:
    """Select two real VISUAL fields on the shared config before device install.

    LeRobot reads config.image_features dynamically during preprocessing; the
    adapter reads the same object when choosing the packed Vision row count.
    No weights or checkpoint quantization metadata are changed here.
    """
    lerobot, adapter, config = _require_cold_shared_policy(policy, "two-camera")
    before = dict(config.image_features)
    selected = {key: feature for key, feature in before.items()
                if key in batch and not key.rsplit(".", 1)[-1].startswith("empty_camera_")}
    validate_camera_batch(batch, selected, cameras=2, text_tokens=text_tokens)
    previous_empty = config.empty_cameras
    config.input_features = {key: feature for key, feature in config.input_features.items()
                             if key not in before or key in selected}
    config.empty_cameras = 0
    if set(config.image_features) != set(selected):
        raise RuntimeError("shared config did not retain exactly the selected two VISUAL features")
    return {"before_image_features": list(before), "image_features": list(config.image_features),
            "before_empty_cameras": previous_empty, "empty_cameras": 0,
            "shared_config": True, "configured_before_rpu_install": True}


def _execution_config(value):
    # Read the public parser without importing the backend before library binding.
    from examples.common.config import _backend_leaf
    root, vision, text, action = _backend_leaf("api/_execution.py").resolve_pi05_execution_components(
        value, entry_point="Pi0.5 example")
    result = {stage: dict(fields) for stage, fields in root.items() if stage != "components"}
    if "components" in root:
        result["components"] = {name: {stage: dict(fields) for stage, fields in scopes.items()}
                                for name, scopes in root["components"].items()}
    return result, vision, text, action


def resolve_execution(args):
    execution, vision, text, action = _execution_config(getattr(args, "rpu_execution", None))
    if not hasattr(args, "public_precision"):
        chunk = resolve_prefill_chunk_size(
            cameras=args.cameras, prefill_w8a8=args.prefill_w8a8,
            requested=args.prefill_chunk_size, text_tokens=args.text_tokens)
        execution["model"] = {"num_cores": 8}
        execution.setdefault("prefill", {})["chunk_size"] = chunk
        return execution, True, chunk
    cores = execution.get("model", {}).get("num_cores", 8)
    if cores not in (4, 6, 8):
        raise ValueError("Pi0.5 num_cores must be 4, 6 or 8")
    if cores != 8 and (args.precision not in ("fp16", "w8a16")
                       or args.cameras != 3 or args.text_tokens != 32):
        raise ValueError("Pi0.5 TP4/TP6 requires FP16/W8A16, three cameras and T32")
    if args.precision == "w4a16_kvint8" and (cores != 8 or args.cameras != 3 or args.text_tokens != 32):
        raise ValueError("legacy whole-model INT4 requires TP8, three cameras and T32")
    pair = (256 * args.cameras + args.text_tokens) // 2
    vision_chunk = vision.get("vision", {}).get("chunk_size", "auto")
    action_chunk = action.get("action", {}).get("chunk_size", "auto")
    allowed_vision = ("auto", 256, 256 * args.cameras) if cores == 8 else ("auto", 256)
    if vision_chunk not in allowed_vision:
        raise ValueError(f"Pi0.5 Vision chunk must be one of {allowed_vision}")
    if action_chunk not in ("auto", 64):
        raise ValueError("Pi0.5 Action chunk must be auto or 64 for horizon 50")
    prefill = text.get("prefill", {})
    requested = prefill.get("chunk_size", "auto")
    diagnostic_chunk = getattr(args, "prefill_chunk_size", None)
    if diagnostic_chunk is not None:
        if requested != "auto" and requested != diagnostic_chunk:
            raise ValueError("conflicting prefill chunk controls")
        requested = diagnostic_chunk
    paired = (cores == 8 and args.precision in ("fp16", "w8a16", "w4a16")
              and requested in ("auto", pair)
              and vision_chunk in ("auto", 256 * args.cameras)
              and action_chunk in ("auto", 64)
              and not any(prefill.get(key, 0) not in (0, "auto")
                          for key in ("padding_rows", "padding_budget")))
    if args.prefill_w8a8 and not paired:
        raise ValueError("Prefill W8A8 requires TP8 and the camera/text paired prefill geometry")
    if args.precision == "w4a16" and not paired:
        raise ValueError("Action NVFP4 example requires its TP8 paired prefill geometry")
    execution.setdefault("model", {})["num_cores"] = cores
    if paired:
        execution.setdefault("prefill", {})["chunk_size"] = pair
        # A component's AUTO overrides its root; resolve it to the same pair.
        child = execution.get("components", {}).get("language_model", {}).get("prefill")
        if child is not None:
            child["chunk_size"] = pair
        requested = pair
    elif diagnostic_chunk is not None:
        execution.setdefault("prefill", {})["chunk_size"] = diagnostic_chunk
    return execution, paired, requested


def validate_example_config(config: dict) -> None:
    """Validate physical choices, never a user-supplied certification flag."""
    required = {"pi05", "input", "run"}
    if not required <= set(config) or set(config) - (required | {"rpu_execution", "cpu", "logging"}):
        raise ValueError("Pi0.5 examples require [pi05], [input], [run], and optional [cpu]/[rpu_execution]/[logging]")
    from examples.common.config import validate_logging_config
    validate_logging_config(config)
    allowed = {
        "pi05": {"precision", "instruction_residency"},
        "input": {"checkpoint", "batch", "noise", "cameras", "text_tokens", "num_steps"},
        "run": {"warmup", "runs", "seed", "output_dir", "profile", "hwperf", "cpu_threads", "torch_num_threads"},
    }
    for name, keys in allowed.items():
        if not isinstance(config[name], dict) or set(config[name]) - keys:
            raise ValueError(f"[{name}] contains unsupported keys or is not a table")
    _CPU["resolve_cpu_threads"](config)
    profile, inputs, run = (config[key] for key in allowed)
    if profile.get("precision") not in _EXAMPLE_PRECISIONS:
        raise ValueError(f"[pi05].precision must be one of {_EXAMPLE_PRECISIONS}")
    if "instruction_residency" in profile and profile["instruction_residency"] not in ("auto", "off"):
        raise ValueError("[pi05].instruction_residency must be auto or off")
    if type(inputs.get("cameras")) is not int or inputs["cameras"] not in (2, 3):
        raise ValueError("[input].cameras must be 2 or 3 actual computed images")
    _validate_text_profile(inputs.get("text_tokens", 32), cameras=inputs["cameras"])
    if type(inputs.get("num_steps", 10)) is not int or inputs.get("num_steps", 10) != 10:
        raise ValueError("Pi0.5 examples require num_steps=10")
    for key in ("checkpoint", "batch"):
        if not isinstance(inputs.get(key), str) or not inputs[key].strip():
            raise ValueError(f"[input].{key} must be a nonempty path or checkpoint alias")
    if "noise" in inputs and (not isinstance(inputs["noise"], str) or not inputs["noise"].strip()):
        raise ValueError("[input].noise must be a nonempty path to a CPU noise tensor")
    for key, default, minimum in (("warmup", 2, 0), ("runs", 6, 1), ("seed", 0, 0)):
        value = run.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"[run].{key} must be an integer >= {minimum}")
    if "output_dir" in run and (not isinstance(run["output_dir"], str) or not run["output_dir"].strip()):
        raise ValueError("[run].output_dir must be a nonempty path")
    for key in ("profile", "hwperf"):
        if type(run.get(key, False)) is not bool:
            raise ValueError(f"[run].{key} must be boolean")
    if (run.get("profile") or run.get("hwperf")) and not run.get("output_dir"):
        raise ValueError("Pi0.5 diagnostics require [run].output_dir")
    resolve_execution(_example_arguments(config))


def _example_arguments(config):
    profile, inputs, run = (config[key] for key in ("pi05", "input", "run"))
    precision = profile["precision"]
    return SimpleNamespace(
        # The diagnostic CLI's historical w4a16 selector denotes Action NVFP4.
        precision="w4a16" if "nvfp4" in precision else precision,
        public_precision=precision,
        instruction_residency=profile.get("instruction_residency"),
        profile=run.get("profile", False), hwperf=run.get("hwperf", False),
        cameras=inputs["cameras"], text_tokens=inputs.get("text_tokens", 32),
        prefill_chunk_size=None, prefill_w8a8=precision == "w8a16_prefill_w8a8_action_nvfp4",
        vision_owner_ln=True, checkpoint=inputs["checkpoint"],
        batch=Path(inputs["batch"]).expanduser(), warmup=run.get("warmup", 2),
        noise=Path(inputs["noise"]).expanduser() if "noise" in inputs else None,
        runs=run.get("runs", 6), seed=run.get("seed", 0),
        cpu_threads=_CPU["resolve_cpu_threads"](config, default=8),
        output_dir=Path(run["output_dir"]).expanduser() if "output_dir" in run else None,
        rpu_execution=config.get("rpu_execution"),
    )


def example_arguments(config: dict) -> SimpleNamespace:
    validate_example_config(config)
    return _example_arguments(config)


def registered_example(config):
    """Resolve only the exact existing asset/geometry profiles for the catalog.

    Two cameras, longer text and NVFP4 have no inherited three-camera profile.
    This identifies the registered scope, not a new numerical certification.
    """
    args = example_arguments(config)
    alias = _REGISTERED_ASSETS.get(args.public_precision)
    if alias is None or args.checkpoint != alias or args.cameras != 3 or args.text_tokens != 32:
        return None
    execution, _, _ = resolve_execution(args)
    cores = execution["model"]["num_cores"]
    if cores == 8:
        _, vision, text, _ = _execution_config(execution)
        prefill = text.get("prefill", {})
        if (prefill.get("chunk_size", "auto") not in ("auto", 400)
                or vision.get("vision", {}).get("chunk_size", "auto") not in ("auto", 768)
                or any(prefill.get(key, 0) not in (0, "auto")
                       for key in ("padding_rows", "padding_budget"))):
            return None
    precision = "w4a16-kvint8" if args.public_precision == "w4a16_kvint8" else args.public_precision
    suffix = f"-tp{cores}" if cores != 8 else ""
    return {"profile_id": f"{alias}.{precision}.vla{suffix}",
            "registry_alias": alias, "target": "pi05"}


def profile_libraries(root: Path | None = None) -> tuple[dict, dict]:
    """Bind the manifest-selected REF pair before importing the backend."""
    resolver = vars(_library_helpers)
    return resolver["current_libraries"](root or Path(__file__).resolve().parents[3])


def runtime_payloads(libraries: dict) -> dict:
    from rpu_backend import _cpp_ext

    actual = _cpp_ext.get_runtime_payload_paths()
    for role, key in (("main", "RPU_KERNEL_LIB_PATH"),
                      ("expansion", "RPU_KERNEL_LIB_PATH_EXPANSION")):
        if key not in libraries:
            continue
        expected = libraries[key]
        path = actual.get("oplib_roles", {}).get(role)
        if (path != expected["path"] or
                actual.get("oplib_payloads", {}).get(path, {}).get("sha256") != expected["sha256"]):
            raise RuntimeError(f"loaded {role} library does not match the optimized profile")
    return actual


def bind_cold_environment(env: dict[str, str]) -> None:
    if "rpu_backend" in sys.modules:
        raise ValueError("optimized profiles require a fresh process before importing rpu_backend")
    conflicts = [key for key, value in env.items() if key in os.environ and os.environ[key] != value]
    if conflicts:
        raise ValueError("conflicting cold environment: " + ", ".join(conflicts))
    if os.environ.get("RPU_CHUNK_FORCE_UNSAFE", "0") != "0":
        raise ValueError("RPU_CHUNK_FORCE_UNSAFE is incompatible with this profile")
    os.environ.update(env)


@contextmanager
def stage_timings(policy):
    """Measure the same host call boundaries used by the optimization reports.

    No extra synchronization or trace is inserted inside the three stages.
    These are host wall times around synchronous runtime calls, not device cycles.
    """
    from rpu_backend.adapters.pi05 import runtime
    from rpu_backend.adapters import siglip

    vlm = policy._lerobot_policy.model.paligemma_with_expert.paligemma.model.language_model
    phases, restore = {}, []

    def observe(owner, name, phase):
        original = getattr(owner, name)
        if not callable(original):
            raise RuntimeError(f"missing optimized timing boundary: {name}")
        had_local = name in vars(owner)

        def call(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                phases[phase] = phases.get(phase, 0.0) + (time.perf_counter() - start) * 1000

        setattr(owner, name, call)
        restore.append((owner, name, original, had_local))

    try:
        observe(runtime, "_run_denoise_fused", "denoise_ms")
        observe(siglip, "_siglip_forward_images", "vision_ms")
        entry = "_rpu_prefill_kv_only" if callable(getattr(vlm, "_rpu_prefill_kv_only", None)) else "forward"
        observe(vlm, entry, "prefill_ms")
        yield phases
    finally:
        for owner, name, original, had_local in reversed(restore):
            if had_local:
                setattr(owner, name, original)
            else:
                delattr(owner, name)


def numerical_notes(*, cameras, text_tokens, valid_tokens, precision, prefill_w8a8):
    return ["This example does not run a reference comparison or robot task."]


_RESULTS = vars(_result_helpers)


def run_profile(args, *, bind_libraries=False) -> dict:
    execution, paired, chunk_size = resolve_execution(args)
    cores = execution["model"]["num_cores"]
    if paired:
        env = profile_environment(args.precision, prefill_w8a8=args.prefill_w8a8,
                                  vision_owner_ln=args.vision_owner_ln, cameras=args.cameras,
                                  text_tokens=args.text_tokens)
    elif hasattr(args, "public_precision"):
        # TP4/TP6, legacy whole-model INT4 and explicitly different chunk plans
        # use the existing generic kernels. Never feed them TP8-only fusions.
        env = {"RPU_PI05_" + name: "0" for name in (*_ON, *_OFF)}
        _, vision, _, _ = _execution_config(execution)
        packed_vision = cores == 8 and vision.get("vision", {}).get("chunk_size", "auto") != 256
        quantized = args.precision != "fp16"
        env.update(RPU_PI05_FUSED_DENOISE="1", RPU_PI05_KVINSERT_PAD16=str(int(cores == 8)),
                   RPU_PI05_DENOISE_UNROLL="1",
                   RPU_PI05_SIGLIP_BATCH=str(int(packed_vision)),
                   RPU_PI05_SIGLIP_W8A16=str(int(quantized)),
                   RPU_PI05_ADARMS_DENSE_W8A16=str(int(quantized)),
                   RPU_PI05_DENOISE_GEGLU_M50="0", RPU_PI05_DENOISE_W4_FASTPATH="0",
                   RPU_PI05_DENOISE_W4_GEGLU_M50="0", RPU_PI05_DENOISE_W4_GEGLU_N64="0",
                   RPU_PI05_DENOISE_NVFP4_GEGLU_M50="0", RPU_PI05_PREFILL_GATEUP_W8A8="0",
                   RPU_PI05_VISION_OWNER_LN="0")
    _, _, _, action = _execution_config(execution)
    # The uniform ACC16 fusion has a bounded M50/T32-or-T160/ten-step ABI.
    # Component overrides take precedence over the root accumulation mode.
    env["RPU_PI05_DENOISE_NVFP4_GEGLU_ACC16_M50"] = str(int(
        paired and args.precision == "w4a16" and args.text_tokens in (32, 160)
        and not action.get("action", {}).get("linear_acc32", False)))
    libraries = {}
    if bind_libraries:
        library_env, libraries = profile_libraries()
        env.update(library_env)
        env.update(MIMALLOC_PURGE_DELAY="1000", MIMALLOC_ARENA_PURGE_MULT="0")
        env.update(RPU_LINEAR_ACC32="0", RPU_PI05_PREFIX_MASK_CACHE="1",
                   RPU_PI05_GEMMA_GRAPH="1", RPU_PI05_DENOISE_GRAPH="1")
        if os.environ.get("RPU_PI05_LOAD_PREFIX_EMBS"):
            raise ValueError("Pi0.5 examples reject RPU_PI05_LOAD_PREFIX_EMBS input replacement")
    if args.noise is not None:
        noise_path = args.noise.resolve(strict=True)
        env["RPU_PI05_LOAD_NOISE"] = str(noise_path)
    bind_cold_environment(env)

    # All cold selectors are installed before imports and before checkpoint mutation.
    effective_cpu = _CPU["configure_cpu_threads"](getattr(args, "cpu_threads", 8))
    import torch
    from rpu_backend.api import Pi05Policy
    from rpu_backend.model_registry import model_path

    actual_payloads = runtime_payloads(libraries)
    torch.set_num_interop_threads(_cpu_interop_threads(
        paired=paired, precision=args.precision, prefill_w8a8=args.prefill_w8a8,
        cameras=args.cameras, text_tokens=args.text_tokens))
    effective_cpu["torch_num_interop_threads"] = torch.get_num_interop_threads()
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_dir():
        checkpoint = model_path(args.checkpoint)
    quant_path = checkpoint / "rpu_quant_config.json"
    quant = json.loads(quant_path.read_text()) if quant_path.is_file() else {}
    if args.precision == "w4a16_kvint8":
        # Old per-channel INT4 covers VLM + expert; absent scope/group metadata
        # is valid. The loader validates storage and projection scales.
        keep = {name.removesuffix(".weight").rsplit(".", 1)[-1]
                for name in quant.get("mixed_int8_suffixes", ())}
        if (quant.get("method") != "w4a16" or quant.get("mode") == "group_symmetric"
                or keep != {"k_proj", "v_proj"}
                or set(quant.get("int4_components", ("vlm", "expert"))) != {"vlm", "expert"}):
            raise ValueError("w4a16_kvint8 requires the legacy whole-model per-channel INT4 checkpoint")
    else:
        validate_checkpoint(quant, args.precision)
    batch = torch.load(args.batch, map_location="cpu", weights_only=True)
    if not isinstance(batch, dict):
        raise ValueError("--batch must contain a tensor dictionary")
    if bind_libraries:
        for key, value in batch.items():
            if key.startswith("observation.images.") and not key.endswith("_padding_mask"):
                if not torch.is_tensor(value) or tuple(value.shape) != (1, 3, 256, 256):
                    raise ValueError(f"Pi0.5 example requires {key} shape [1,3,256,256]")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    hwperf_dir = args.output_dir / "hwperf" if args.output_dir is not None else None
    report = {"model": "pi05", "precision": getattr(args, "public_precision", args.precision),
              "checkpoint": str(checkpoint), "batch": str(args.batch),
              "warmup": args.warmup, "runs": args.runs, "cameras": args.cameras,
              "text_tokens": args.text_tokens, "cpu_runtime": effective_cpu,
              "rpu_execution": execution, "libraries": libraries,
              "formal_certification": False, "correctness": "not_run"}
    with _RESULTS["record_result"](report, args.output_dir,
        timing=lambda: _RESULTS["phase_timings"](report.get("samples_ms", []),
            report.get("timing_scope", "host predict_action_chunk including CPU output copy"), statistic="mean",
            phase_boundary="instrumented Vision/Prefill/Denoise host calls during measured inference; CPU output copy excluded"),
        runtime_checks=("finite action output shape [1,50,7]",),
        output_names=("actions.pt",) +
                     (("profile.trace.json", "profile.txt") if getattr(args, "profile", False) else ())):
        trace = (torch.rpu.hw_perf_trace(str(hwperf_dir), max_dumps=32)
                 if getattr(args, "hwperf", False) else nullcontext())
        with trace:
            policy = None
            try:
                policy_options = {}
                if getattr(args, "instruction_residency", None) is not None:
                    from rpu_backend.graph import GraphRuntimePolicy
                    policy_options["graph_runtime_policy"] = GraphRuntimePolicy.from_environment(
                        batch_snapshot_mode=args.instruction_residency)
                policy = Pi05Policy.from_pretrained(str(checkpoint), dtype=torch.float16,
                    trust_remote_code=False, rpu_execution=execution, **policy_options)
                camera_configuration = configure_two_camera_policy(policy, batch, text_tokens=args.text_tokens) if args.cameras == 2 else None
                text_configuration = configure_text_profile(policy, args.text_tokens)
                validate_camera_batch(batch, policy._lerobot_policy.config.image_features,
                                      cameras=args.cameras, text_tokens=args.text_tokens)
                if int(policy._lerobot_policy.config.chunk_size) != 50:
                    raise ValueError("Pi0.5 profile requires checkpoint action horizon/chunk_size=50")
                mask = batch.get("observation.language.attention_mask")
                valid_tokens = int(mask.sum()) if torch.is_tensor(mask) else args.text_tokens
                notes = numerical_notes(cameras=args.cameras, text_tokens=args.text_tokens,
                                        valid_tokens=valid_tokens, precision=args.precision,
                                        prefill_w8a8=args.prefill_w8a8)
                policy.to("rpu")
                prepared = policy.prepare_graphs(batch, num_steps=10, precompute_adarms=paired)
                if args.precision == "fp16" and (prepared.get("adarms_table") or {}).get("dense_w8a16", False):
                    raise RuntimeError("FP16 profile unexpectedly installed quantized AdaRMS dense weights")
                rows = []
                report["samples_ms"] = rows
                with (stage_timings(policy) if paired else nullcontext({})) as phases:
                    for index in range(args.warmup + args.runs):
                        torch.manual_seed(args.seed)
                        phases.clear()
                        start = time.perf_counter()
                        output = policy.predict_action_chunk(batch, num_steps=10).cpu()
                        total_ms = (time.perf_counter() - start) * 1000
                        if tuple(output.shape) != (1, 50, 7) or not torch.isfinite(output).all():
                            raise RuntimeError("expected finite action output [1,50,7]")
                        if paired and set(phases) != {"vision_ms", "prefill_ms", "denoise_ms"}:
                            raise RuntimeError(f"incomplete optimized timing boundaries: {sorted(phases)}")
                        if index >= args.warmup:
                            rows.append({"total_ms": total_ms, **phases})
                if getattr(args, "profile", False):
                    from rpu_backend.runtime.performance import _profile_ctx
                    with _profile_ctx(True) as profiler:
                        torch.manual_seed(args.seed)
                        policy.predict_action_chunk(batch, num_steps=10).cpu()
                    profiler.export_chrome_trace(str(args.output_dir / "profile.trace.json"))
                    (args.output_dir / "profile.txt").write_text(
                        profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=100))
                report.update({
                    "precision": getattr(args, "public_precision", args.precision), "num_steps": 10, "num_cores": cores,
                    "execution_route": "paired_tp8" if paired else "generic",
                    "timed_phases": list(rows[0]),
                    "profile": getattr(args, "profile", False), "hwperf_enabled": getattr(args, "hwperf", False),
                    "rpu_execution": execution,
                    "instruction_residency": getattr(args, "instruction_residency", None),
                    "cameras": args.cameras, "vision_rows": 256 * args.cameras,
                    "prefix_rows": 256 * args.cameras + args.text_tokens,
                    "text_tokens": args.text_tokens, "valid_text_tokens": valid_tokens,
                    "prefill_chunk_size": chunk_size, "camera_configuration": camera_configuration,
                    "text_configuration": text_configuration, "prefill_w8a8": args.prefill_w8a8,
                    "vision_owner_ln": args.vision_owner_ln if paired else False, "adarms_table": prepared.get("adarms_table"),
                    "action_nvfp4_geglu_acc16": env["RPU_PI05_DENOISE_NVFP4_GEGLU_ACC16_M50"] == "1",
                    "checkpoint": str(checkpoint), "batch": str(args.batch), "libraries": libraries,
                    "runtime_payloads": actual_payloads,
                    "graph_prepare": {key: prepared.get(key) for key in (
                        "phase", "build_replay_bit_equal", "ready_replay_validated", "ready_stats")},
                    "noise": os.environ.get("RPU_PI05_LOAD_NOISE"),
                    "warmup": args.warmup, "runs": args.runs, "seed": args.seed,
                    "cpu_runtime": effective_cpu,
                    "timing_scope": "host wall time; total includes CPU output copy; load, prepare, and warmup excluded",
                    "mean_ms": {key: statistics.mean(row[key] for row in rows) for key in rows[0]},
                    "median_ms": {key: statistics.median(row[key] for row in rows) for key in rows[0]},
                    "mean_ms_including_output_copy": statistics.mean(row["total_ms"] for row in rows),
                    "samples_ms": rows, "output_shape": list(output.shape),
                    "registered_profile": getattr(args, "registered_profile", None),
                    "formal_certification": False, "correctness": "not_run",
                    "numerical_notes": notes,
                })
                if args.output_dir is not None:
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(output, args.output_dir / "actions.pt")
            finally:
                _RESULTS["close_with_report"](policy, report)
        if getattr(args, "hwperf", False):
            from rpu_backend.runtime.performance import _hwperf_aggregate_file, _hwperf_collect_files
            report["hwperf"] = [_hwperf_aggregate_file(path) for path in _hwperf_collect_files(str(hwperf_dir))]
            if not report["hwperf"] or any(item["error"] for item in report["hwperf"]):
                raise RuntimeError("HWPerf was requested but its traces are missing or unreadable")
    _LOG.debug("report=%s", report)
    return report


def run_example_config(config: dict) -> int:
    args = example_arguments(config)
    example = registered_example(config)
    args.registered_profile = None
    if example is not None:
        from examples.common.config import example_profiles

        profile = example_profiles()[example["profile_id"]]
        args.registered_profile = {key: profile[key] for key in ("id", "support", "input_envelope")}
    run_profile(args, bind_libraries=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=_PRECISIONS, required=True)
    parser.add_argument("--instruction-residency", choices=("auto", "off"),
                        help="optional cold instruction snapshot policy; absent preserves runtime defaults")
    parser.add_argument("--cameras", type=int, choices=(2, 3), default=3,
                        help="exact present camera count; two cameras use a new controlled profile")
    parser.add_argument("--text-tokens", type=int, choices=(32, 64, 96, 128, 160), default=32,
                        help="cold physical text bucket; T64/T96/T128/T160 require two or three cameras")
    parser.add_argument("--prefill-chunk-size", type=int,
                        help="cold prefill chunk; T32 defaults 400/272 for three/two cameras; T64 requires 416/288; T96 requires 432/304; T128 requires 448/320; T160 requires 464/336")
    parser.add_argument("--prefill-w8a8", action="store_true",
                        help="opt in to Prefill GateUp W8A8 with Action NVFP4 only")
    parser.add_argument("--vision-owner-ln", action="store_true",
                        help="opt in to FP16/W8 Vision owner-reduce/LayerNorm for two or three cameras")
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory or model_registry alias")
    parser.add_argument("--batch", type=Path, required=True, help="CPU tensor dictionary saved with torch.save")
    parser.add_argument("--noise", type=Path, help="optional fixed CPU noise tensor saved with torch.save")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 1 or args.seed < 0:
        parser.error("--runs must be positive, --warmup >= 1, and --seed >= 0")
    try:
        run_profile(args)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
