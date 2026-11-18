
from examples.common import cpu as _cpu_helpers
from examples.common import libraries as _library_helpers
from examples.common import results as _result_helpers
from examples.common.logging import prepare_output_directory
#!/usr/bin/env python3
"""Wall base H32/10-step validation and benchmark. Invoke inside a device lock.

Example: python benchmark.py --mode candidate --precision fp16 --fp16 /models/base
         --so /build/rpu_backend.so --output-dir /results/candidate
The public infer timer includes its CPU physical-action result. Stage timers are
ordinary synchronous call durations; no extra device synchronization is added.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


_LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
_CPU = vars(_cpu_helpers)
_RESULTS = vars(_result_helpers)
_LIBRARIES = vars(_library_helpers)
CAMERAS = ["face_view", "left_wrist_view", "right_wrist_view"]
INSTRUCTION = "Pick up the object and place it in the bin."


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--precision", choices=("fp16", "w8"), required=True)
    parser.add_argument("--components", choices=("action", "all", "small", "small-prefill"), default="action")
    parser.add_argument("--fp16", type=Path, required=True)
    parser.add_argument("--w8", type=Path)
    parser.add_argument("--so", type=Path)
    parser.add_argument("--main-ref", type=Path)
    parser.add_argument("--expansion-ref", type=Path)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--compare", type=Path, help="saved board outputs/input tensors for A/B diagnostics")
    parser.add_argument("--batch", type=Path, help="CPU observation mapping; uses supplied images and instruction without rewriting")
    parser.add_argument("--text-tokens", type=int, help="require this exact tokenizer length in the selected scope")
    parser.add_argument("--text-length-scope", choices=("instruction", "non_image"), default="instruction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction-residency", choices=("auto", "off"),
                        help="optional cold instruction snapshot policy; absent preserves runtime defaults")
    parser.add_argument("--action-residency", action="store_true")
    parser.add_argument("--action-nvfp4", action="store_true",
                        help="experimental W8 Vision/Prefill with only Action NVFP4A16")
    parser.add_argument("--vision-window", action="store_true")
    parser.add_argument("--vision-pair", action="store_true")
    parser.add_argument("--cameras", type=int, choices=(1, 2, 3), default=3,
                        help="ordered cameras: face_view, left_wrist_view, then right_wrist_view")
    parser.add_argument("--image-size", type=int, nargs=2, default=(448, 252), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-range", type=int, nargs=2,
                        help="default: prepare the observed real prefix length only")
    parser.add_argument("--hw-perf-trace", action="store_true",
                        help="diagnostic kernel trace; resulting wall times are not deployment performance")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--cpu-threads", type=int,
                        help="default: 8; public TOML profiles supply their selected CPU count")
    parser.add_argument("--cpu-thread-sweep", type=int, nargs="+", choices=(1, 2, 4, 8, 12, 16, 24),
                        default=[], help="additional thread counts on the same live model")
    args = parser.parse_args(argv)
    if args.text_tokens is not None and (args.text_tokens <= 0 or args.batch is None):
        parser.error("--text-tokens requires a positive length and --batch")
    if args.batch is not None and args.golden is not None:
        parser.error("--batch and --golden describe different input sources")
    if args.cpu_threads is None:
        args.cpu_threads = 8
    if args.precision == "w8" and args.w8 is None:
        parser.error("--precision w8 requires --w8")
    if args.action_nvfp4 and (args.precision != "w8" or args.mode != "candidate"
                             or not args.components.startswith("small")
                             or args.action_residency):
        parser.error("--action-nvfp4 requires a small W8 candidate without --action-residency")
    if args.components != "action" and args.mode != "candidate":
        parser.error("full/small component optimization requires --mode candidate")
    if (args.action_residency or args.vision_window) and args.mode != "candidate":
        parser.error("experimental optimizations require --mode candidate")
    if args.vision_window and tuple(args.image_size) != (224, 224):
        parser.error("--vision-window candidate requires two/three224 images")
    if args.vision_window and args.components == "small-prefill":
        parser.error("--vision-window requires Vision row reuse (--components small)")
    if args.vision_pair and (args.mode != "candidate" or args.vision_window
                            or tuple(args.image_size) != (256, 256)
                            or not args.components.startswith("small")):
        parser.error("--vision-pair requires the small two/three256 candidate")
    if min(args.image_size) <= 0:
        parser.error("image dimensions must be positive")
    if tuple(args.image_size) != (448, 252) and args.components == "all":
        parser.error("--components all is specific to 448x252")
    if args.components.startswith("small") and tuple(args.image_size) not in ((224, 224), (256, 256)):
        parser.error("--components small requires 224x224 or 256x256")
    if not 1 <= args.warmup <= 10 or not 1 <= args.runs <= 30:
        parser.error("require 1..10 warmups and 1..30 measured calls")
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be a positive integer")
    if len(set([args.cpu_threads, *args.cpu_thread_sweep])) != 1 + len(args.cpu_thread_sweep):
        parser.error("CPU thread counts must be distinct")
    if args.runs * (1 + len(args.cpu_thread_sweep)) > 30:
        parser.error("at most 30 measured calls across CPU thread counts")
    for name in ("fp16", "w8", "so", "main_ref", "expansion_ref", "golden", "compare", "batch"):
        path = getattr(args, name)
        if path is not None and not path.exists():
            parser.error(f"missing {name}: {path}")
    try:
        prepare_output_directory(args.output_dir)
    except FileExistsError as error:
        parser.error(str(error))
    return args


def cold_environment(args, root=ROOT):
    # Pin the base runtime independently of test fixtures or local experiments.
    env = {
        'RPU_WALL_OSS_ACTION_FP32_TAIL': '0',
        'RPU_WALL_OSS_SHORT_PROMPT': '0',
        'RPU_WALL_OSS_GENERIC_PROLOGUE': '0',
        'RPU_WALL_OSS_MULTISUITE_PROLOGUE': '0',
        'RPU_WALL_OSS_PROLOGUE_FILLER': '',
        'RPU_WALL_OSS_VISION_DEVICE_MERGED': '0',
        'RPU_WALL_OSS_ATTN_TP8': '1',
        'RPU_WALL_OSS_PARTIAL_MROPE': '1',
        'RPU_WALL_OSS_VISION_ROPE_SPM': '1',
        'RPU_WALL_OSS_VISION_FUSED_MERGER': '1',
        'RPU_WALL_OSS_FAST_REPLAY': '1',
        'RPU_FASTREPLAY_SKIP_SYNC': '1',
        'RPU_DEEP_FAST_REPLAY': '1',
        'RPU_WALL_OSS_DENOISE_UNROLL': '1',
        'RPU_WALL_OSS_FUSED_DENOISE': '1',
        'RPU_WALL_OSS_KVINSERT_PAD16': '1',
        'RPU_WALL_OSS_BATCH_VISION': '1',
        'RPU_WALL_OSS_BATCH_MAX_SEQ': '768',
        'RPU_WALL_OSS_VISION_LAYER_GROUP': '0',
        'RPU_WALL_OSS_VISION_WEIGHT_OUTER': '0',
        'RPU_WALL_OSS_DEVICE_PATCH_EMBED': '1',
        'RPU_WALL_OSS_FAST_PROCESSOR': '1',
        'RPU_WALL_OSS_FAST_IMGPROC': '1',
        'RPU_WALL_OSS_PE_NORM_FOLD': '1',
        'RPU_WALL_OSS_ONDEVICE_EMBED': '1',
        'RPU_WALL_OSS_HOST_CACHE': '1',
        'RPU_WALL_OSS_FUSED_ASSEMBLE': '0',
    }
    env.update({
        "RPU_WALL_OSS_FP16_TRANSFER_MASK": "4" if args.mode == "candidate" and args.precision == "fp16" else "0",
        "RPU_WALL_OSS_SORTING_SILU_MUL": "1" if args.mode == "candidate" else "0",
        "RPU_WALL_OSS_ACTION_WEIGHT_RESIDENCY": "0",
        "RPU_WALL_OSS_ACTION_KV_OWNER_FANOUT": "0",
        "RPU_WALL_OSS_W16_REUSE_MASK": "0",
        "RPU_WALL_OSS_VISION_MIXED_PRECISION": "0",
        "RPU_WALL_OSS_PREFILL_MIXED_PRECISION": "0",
        "RPU_WALL_OSS_PREFILL_STAGED_MLP": "0",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    if args.vision_pair:
        env["RPU_WALL_OSS_VISION_PAIR_252"] = "1"
    if args.components == "all":
        env.update({
            "RPU_WALL_OSS_FP16_TRANSFER_MASK": "7" if args.precision == "fp16" else "0",
            "RPU_WALL_OSS_VISION_WEIGHT_OUTER": "1",
            "RPU_WALL_OSS_VISION_ROW_WEIGHT_OUTER": "1",
            "RPU_WALL_OSS_VISION_MIXED_PRECISION": "1",
            "RPU_WALL_OSS_VISION_NATIVE_WINDOW_SDPA": "1",
            "RPU_WALL_OSS_PREFILL_MIXED_PRECISION": "1",
            "RPU_WALL_OSS_PREFILL_STAGED_MLP": "1",
            "RPU_WALL_OSS_PREFILL_OWNER_DIRECT_KV": "1",
        })
    elif args.components.startswith("small"):
        env.update({
            "RPU_WALL_OSS_FP16_TRANSFER_MASK": (
                "6" if args.components == "small-prefill" else "7") if args.precision == "fp16" else "0",
            "RPU_WALL_OSS_VISION_ROW_WEIGHT_OUTER": "0" if args.components == "small-prefill" else "1",
            "RPU_WALL_OSS_FUSED_ASSEMBLE": "1",
        })
    if args.action_residency:
        env["RPU_WALL_OSS_ACTION_WEIGHT_RESIDENCY"] = "1"
    if args.vision_window:
        env["RPU_WALL_OSS_VISION_NATIVE_WINDOW_SDPA"] = "1"
    if (getattr(args, "rpu_execution", None) or {}).get("model", {}).get("num_cores", 8) != 8:
        # The candidate selectors above have eight-core owners only. Both reduced
        # budgets run the ordinary routes with fused Action, SPM RoPE and the
        # four-lane fused merger; --vision-window selects per-window SDPA.
        env.update({
            "RPU_WALL_OSS_ATTN_TP8": "0",
            "RPU_WALL_OSS_BATCH_VISION": "0",
            "RPU_WALL_OSS_VISION_ROW_WEIGHT_OUTER": "0",
            "RPU_WALL_OSS_FUSED_ASSEMBLE": "0",
            "RPU_WALL_OSS_FP16_TRANSFER_MASK": "0",
            "RPU_WALL_OSS_SORTING_SILU_MUL": "0",
        })
    selected = {}
    for argument, key in (
        (args.main_ref, "RPU_KERNEL_LIB_PATH"),
        (args.expansion_ref, "RPU_KERNEL_LIB_PATH_EXPANSION"),
    ):
        selected[key] = argument or os.environ.get(key)
    missing = [key for key, value in selected.items() if value is None]
    if missing:
        defaults, _ = _LIBRARIES["current_libraries"](Path(root))
        selected.update({key: defaults[key] for key in missing})
    if args.so is not None:
        selected["RPU_BACKEND_SO"] = args.so
    libraries, _ = _LIBRARIES["verify_libraries"](Path(root), selected)
    env.update(libraries)
    if os.environ.get("RPU_CHUNK_FORCE_UNSAFE", "0") not in ("", "0"):
        raise ValueError("RPU_CHUNK_FORCE_UNSAFE is forbidden for this benchmark")
    for key in list(os.environ):
        if key.startswith("RPU_WALL_OSS_"):
            del os.environ[key]
    # Production accumulation is bound by the native handle (default ACC16).
    os.environ.pop("RPU_LINEAR_ACC32", None)
    os.environ.update(env)
    return env


def summarize(values):
    return {"samples_ms": values, "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values), "min_ms": min(values), "max_ms": max(values)}


def load_observation_batch(path, image_size, cameras=3):
    """Validate caller-owned CPU inputs before loading model weights."""
    import torch
    from PIL import Image

    if type(cameras) is not int or cameras not in (1, 2, 3):
        raise ValueError("cameras must be 1, 2 or 3")
    batch = torch.load(path, map_location="cpu", weights_only=True)
    required = {"images", "instruction", "proprioception", "state_mask", "noise", "dof_mask"}
    if not isinstance(batch, dict) or set(batch) != required:
        raise ValueError(f"batch must contain exactly {sorted(required)}")
    if not isinstance(batch["instruction"], str) or not batch["instruction"].strip():
        raise ValueError("batch instruction must be nonempty text")
    width, height = image_size
    shapes = {"images": (cameras, height, width, 3), "proprioception": (26,),
              "state_mask": (26,), "noise": (1, 32, 26), "dof_mask": (1, 32, 26)}
    for key, shape in shapes.items():
        tensor = batch[key]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tuple(tensor.shape) != shape:
            raise ValueError(f"batch {key} must be a CPU tensor with shape {shape}")
        dtype_ok = (tensor.dtype == torch.uint8 if key == "images" else
                    tensor.dtype == torch.bool if key == "state_mask" else
                    tensor.dtype in (torch.float16, torch.float32))
        if not dtype_ok or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"batch {key} has an invalid dtype or nonfinite values")
    mask = torch.arange(26) < 7
    if not torch.equal(batch["state_mask"], mask) or not torch.equal(
            batch["dof_mask"].float(), mask.float().reshape(1, 1, 26).expand(1, 32, 26)):
        raise ValueError("base benchmark requires the seven active berkeley_autolab_ur5 slots")
    return ([Image.fromarray(frame.contiguous().numpy()) for frame in batch["images"]],
            batch["instruction"], batch["proprioception"].float(), batch["state_mask"],
            batch["noise"].float(), batch["dof_mask"].float())


def validate_golden_metadata(golden, image_size, cameras=3, instruction=INSTRUCTION):
    if (type(cameras) is not int or cameras not in (1, 2, 3)
            or golden["instruction"] != instruction or int(golden["horizon"]) != 32
            or int(golden["num_steps"]) != 10 or int(golden["num_images"]) != cameras
            or tuple(golden["image_size"]) != tuple(image_size)
            or list(golden["camera_names"]) != CAMERAS[:cameras]
            or golden["dataset_key"] != "berkeley_autolab_ur5"
            or golden["delta_action"] or int(golden["state_bins"]) != 256):
        raise ValueError("CPU golden does not match the exact base fixture")


def changed_camera_images(images, index):
    """Keep camera order and change one supplied image without adding cameras."""
    from PIL import ImageOps
    changed = list(images)
    changed[index] = ImageOps.invert(images[index])
    return changed


def text_lengths(tokenizer, instruction, input_ids, grid):
    """Separate task text from scaffold/state, image expansion and padding."""
    instruction_tokens = len(tokenizer.encode(instruction, add_special_tokens=False))
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_tokens = int((input_ids == image_id).sum().item())
    expected_vision = int(grid.prod(dim=1).sum().item()) // 4
    if vision_tokens != expected_vision or vision_tokens <= 0:
        raise ValueError("processor image token count differs from the actual grid")
    real_prefix = int(input_ids.size(1))
    return {"instruction_tokens": instruction_tokens,
            "non_image_tokens": real_prefix - vision_tokens,
            "vision_tokens": vision_tokens, "real_prefix_len": real_prefix}


def build_task_policy(policy, action_nvfp4):
    """Use the existing internal hybrid builder without adding a public mode.

    Keep the ordinary Policy.to lifecycle, Session ownership and cleanup. Only
    this single-threaded task's cold builder invocation changes composition.
    The public NVFP4 mode also quantizes Prefill and is a different experiment.
    """
    if not action_nvfp4:
        return policy.to("rpu")
    from unittest.mock import patch
    from rpu_backend.adapters import wall_oss
    original = wall_oss.build_wall_oss_vla

    def hybrid(*args, **kwargs):
        if (kwargs.get("w8a16") is not True or kwargs.get("rtc")
                or kwargs.get("w4a16") or kwargs.get("nvfp4_wint4_scope")):
            raise ValueError("task Action NVFP4 requires ordinary W8 source composition")
        kwargs.update(w8a16=False, w8_nvfp4a16=True)
        return original(*args, **kwargs)

    with patch.object(wall_oss, "build_wall_oss_vla", hybrid):
        result = policy.to("rpu")
    result._precision = "w8a16_action_nvfp4a16_experimental"
    if result._vla.action._precision_mode != 3:
        raise RuntimeError("Action NVFP4 builder did not install NVFP4 weights")
    return result


def main(args=None):
    args = arguments() if args is None else args
    cameras = CAMERAS[:args.cameras]
    report = {"mode": args.mode, "precision": args.precision, "components": args.components, "horizon": 32,
              "num_steps": 10, "warmup": args.warmup, "runs": args.runs,
              "prefix_range": args.prefix_range,
              "checkpoint": str((args.w8 if args.precision == "w8" else args.fp16).resolve()),
              "action_precision": "nvfp4a16" if args.action_nvfp4 else args.precision,
              "experimental_action_nvfp4": args.action_nvfp4,
              "status": "error", "formal_certification": False}
    report["source_config"] = getattr(args, "source_config", None)
    report["rpu_execution"] = getattr(args, "rpu_execution", {})
    report["instruction_residency"] = getattr(args, "instruction_residency", None)
    report["requested_linear_accumulation"] = {
        stage: "ACC32" if report["rpu_execution"].get(stage, {}).get("linear_acc32", False) else "ACC16"
        for stage in ("vision", "prefill", "action")
    }
    report["seed"] = args.seed
    report["reference_correctness"] = "not_run" if args.golden is None and args.compare is None else "saved_reference_diagnostics"
    policy = None
    try:
        report["cold_environment"] = cold_environment(args)
        report["cpu_runtime"] = _CPU["configure_cpu_threads"](args.cpu_threads)
        import torch
        import transformers
        from PIL import Image, ImageOps
        import rpu_backend
        from rpu_backend.api import WallOssPolicy

        imported = _library_helpers.verify_backend_source(ROOT, rpu_backend.__file__)
        report["python_source"] = str(imported)
        report["packages"] = {"torch": torch.__version__, "transformers": transformers.__version__}
        loaded_native = sorted({str(Path(line.split()[-1]).resolve())
            for line in Path("/proc/self/maps").read_text().splitlines()
            if line.split()[-1].endswith("/rpu_backend.so")})
        report["loaded_native"] = loaded_native
        if args.so is not None and loaded_native != [str(args.so.resolve())]:
            raise RuntimeError(f"unexpected native extension mapping: {loaded_native}")
        report["artifacts"] = {}
        paths = {key: os.environ.get(key) for key in
                 ("RPU_BACKEND_SO", "RPU_KERNEL_LIB_PATH", "RPU_KERNEL_LIB_PATH_EXPANSION")}
        for key, path in paths.items():
            if path:
                report["artifacts"][key] = {"path": str(Path(path).resolve()),
                    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        report["cpu_threads"] = torch.get_num_threads()
        report["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        torch.rpu.set_caching_allocator(True)
        torch.rpu.set_ddr_flush(False)
        torch.rpu.set_ddr_flush_force(True)
        if args.hw_perf_trace:
            # Bind tracing before any Graph BUILD. Keep its flag unchanged
            # through READY replay; dump budget bounds file IO, not execution.
            torch.rpu.set_hw_perf_trace(True, str(args.output_dir / "hwperf"), 16)
            report["timing_scope"] = "hardware_trace_diagnostic"
            report["hw_perf_trace"] = {"directory": str(args.output_dir / "hwperf"), "max_dumps": 16}
        image_path = ROOT / "assets/test_frame.png"
        base = Image.open(image_path).convert("RGB").resize(tuple(args.image_size), Image.Resampling.BICUBIC)
        images_a = [base, ImageOps.mirror(base), ImageOps.flip(base)][:args.cameras]
        state = torch.linspace(0.05, 0.95, 26)
        state_mask = torch.arange(26) < 7
        dof = state_mask.float().reshape(1, 1, 26).expand(1, 32, 26).contiguous()
        noise = torch.randn(1, 32, 26, generator=torch.Generator().manual_seed(args.seed))
        instruction = INSTRUCTION
        if args.batch is not None:
            images_a, instruction, state, state_mask, noise, dof = load_observation_batch(
                args.batch, args.image_size, args.cameras)
        images_b = changed_camera_images(images_a, 0)
        observation = {"instruction": instruction, "proprioception": state,
                      "state_mask": state_mask, "dof_mask": dof, "noise_seed": 0}
        report["input"] = {"image": str(image_path),
                           "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                           "image_size": list(args.image_size), "cameras": cameras,
                           "num_images": args.cameras,
                           "dataset_key": "berkeley_autolab_ur5", "state_bins": 256,
                           "instruction": instruction, "active_slots": list(range(7)),
                           "changed_input": "invert RGB values of first camera only"}
        if args.vision_pair:
            report["input"]["changed_last_camera"] = cameras[-1]
        if args.batch is not None:
            report["input"].update(batch=str(args.batch.resolve()),
                batch_sha256=hashlib.sha256(args.batch.read_bytes()).hexdigest(),
                image="caller_batch", image_sha256=None)
        reference = None
        if args.golden is not None:
            golden = torch.load(args.golden, map_location="cpu", weights_only=False)
            # CPU RNG and floating construction can differ between x86 and ARM.
            # Consume the saved inputs instead of regenerating a reference by seed.
            noise = golden["noise"].float().reshape(1, 32, 26).clone()
            state = golden["proprioception"].float().reshape(26).clone()
            state_mask = golden["state_mask"].bool().reshape(26).clone()
            dof = golden["dof_mask"].float().reshape(1, 32, 26).clone()
            observation.update(proprioception=state, state_mask=state_mask, dof_mask=dof)
            report["input"]["tensor_source"] = str(args.golden.resolve())
            validate_golden_metadata(golden, args.image_size, args.cameras)
            processor = transformers.AutoProcessor.from_pretrained(args.fp16, local_files_only=True)
            processed = processor(text=[golden["text"]], images=images_a, return_tensors="pt")
            input_checks = {key: torch.equal(processed[key], golden[key])
                            for key in ("pixel_values", "input_ids", "image_grid_thw")}
            if not all(input_checks.values()):
                raise ValueError(f"CPU golden processor inputs differ: {input_checks}")
            report["golden_input_checks"] = input_checks
            del processor, processed
            reference = golden["action"].float().reshape(32, 26)
        previous_outputs = None
        if args.compare is not None:
            previous_outputs = torch.load(args.compare, map_location="cpu", weights_only=False)
            saved = {"noise": noise, "proprioception": state,
                     "state_mask": state_mask, "dof_mask": dof}
            for key, tensor in saved.items():
                if not torch.equal(tensor, previous_outputs[key].reshape_as(tensor)):
                    raise ValueError(f"comparison input differs: {key}")
            report["comparison_input_exact"] = True
        policy_options = {}
        if getattr(args, "instruction_residency", None) is not None:
            from rpu_backend.graph import GraphRuntimePolicy
            policy_options["graph_runtime_policy"] = GraphRuntimePolicy.from_environment(
                batch_snapshot_mode=args.instruction_residency)
        policy = WallOssPolicy.from_checkpoint(
            args.w8 if args.precision == "w8" else args.fp16,
            fp16_ckpt_dir=args.fp16, w8a16=args.precision == "w8",
            dataset_key="berkeley_autolab_ur5", camera_names=cameras,
            action_horizon=32, num_steps=10, active_slots=list(range(7)),
            rpu_execution=getattr(args, "rpu_execution", None), **policy_options,
        )
        policy = build_task_policy(policy, args.action_nvfp4)
        vla = policy._vla
        report["actual_precision"] = policy.precision
        report["action_precision_mode"] = vla.action._precision_mode
        from rpu_backend.adapters.wall_oss.runtime import _build_flow_prompt
        text = _build_flow_prompt(instruction, cameras, vla.dataset_key,
                                 vla.delta_action, vla._proprioception_string(state, state_mask))
        input_ids, _, grid = vla._process_inputs(text, images_a)
        if int(grid.size(0)) != args.cameras:
            raise ValueError("processor camera count differs from the supplied images")
        real_prefix_len = int(input_ids.size(1))
        lengths = text_lengths(vla._processor.tokenizer, instruction, input_ids, grid)
        report["input"].update(lengths)
        report["input"]["text_length_scope"] = args.text_length_scope
        report["input"]["requested_text_tokens"] = args.text_tokens
        length_key = "instruction_tokens" if args.text_length_scope == "instruction" else "non_image_tokens"
        if args.text_tokens is not None and lengths[length_key] != args.text_tokens:
            raise ValueError(f"requested {args.text_length_scope} tokens={args.text_tokens}, actual={lengths[length_key]}; input is never truncated or padded")
        prefix_range = tuple(args.prefix_range or (real_prefix_len, real_prefix_len))
        report["prefix_range"] = prefix_range
        report["input"].update({"image_grid_thw": grid.tolist(),
                               "real_prefix_len": real_prefix_len,
                               "vision_tokens": int(grid.prod(dim=1).sum()) // 4})
        phase_times = {}
        for label, owner, name in (("vision", vla.vision, "forward"),
                                   ("prefill", vla.llm, "forward_embeds"),
                                   ("denoise", vla.action, "denoise_with_mask")):
            original = getattr(owner, name)
            def timed(*values, _original=original, _label=label, **keywords):
                started = time.perf_counter()
                try:
                    return _original(*values, **keywords)
                finally:
                    phase_times[_label] = phase_times.get(_label, 0.0) + (time.perf_counter() - started) * 1000
            setattr(owner, name, timed)

        def snapshot():
            result = {}
            for name, component in (("vision", vla.vision), ("prefill", vla.llm), ("denoise", vla.action)):
                cache = component._graph_cache
                entries = cache.snapshot()
                result[name] = {"size": cache.size(), "phase": cache.phase,
                    "invariant": cache.cache_invariant_ok(),
                    "kernel_count": sum(int(entry.kernel_count) for entry in entries),
                    "replay": sum(int(entry.replay_count) for entry in entries),
                    "recapture": sum(int(entry.recapture_count) for entry in entries),
                    "instruction_snapshots": [{field: getattr(entry, field, None) for field in (
                        "batch_snapshot_mode", "batch_snapshot_available",
                        "batch_snapshot_effective_contracts", "batch_snapshot_fallback_reason")}
                        for entry in entries]}
            return result

        policy.prepare_graphs(images=images_a, prefix_length_range=prefix_range, **observation)
        report["graph_profile"] = policy.graph_profile
        report["processor_paths"] = {
            "fast_image_validated": vla._fast_imgproc_ok,
            "normalize_fold_enabled": vla._pe_norm_fold,
            "fused_assembly_used": vla.vision._pending_rev is not None,
        }
        expected_profile = next(item for item in policy.graph_profile["execution_profiles"]
                                if item["real_min"] <= real_prefix_len <= item["real_max"])
        report["input"].update(execution_len=expected_profile["execution_len"],
            execution_padding_rows=expected_profile["execution_len"] - real_prefix_len,
            prefill_chunk_size=expected_profile["chunk_size"])
        prepared = snapshot()
        report["graphs_prepared"] = prepared
        # Preserve the actual native winner for route audits; capacity padding
        # alone does not reveal whether attention selected DDR or raw SPM.
        report["vision_execution_plan"] = dict(vla.vision._rpu_last_execution_plan)
        report["prefill_execution_plan"] = dict(vla.llm._rpu_last_execution_plan)
        report["action_execution_plan"] = dict(vla.action._rpu_last_execution_plan)
        if args.vision_pair:
            report["vision_invocation_plans"] = list(vla.vision._last_forward_plans)
            report["graph_arena_count"] = vla._graph_arena_policy.graph_arena_count
        previous = prepared
        samples, phases, outputs = [], {name: [] for name in prepared}, {}

        def infer(images):
            nonlocal previous
            phase_times.clear()
            started = time.perf_counter()
            result = policy.infer(images=images, noise=noise, **observation)
            elapsed = (time.perf_counter() - started) * 1000
            if result.actions.shape != (32, 26) or not torch.isfinite(result.actions).all():
                raise RuntimeError("action must be finite with shape (32, 26)")
            current = snapshot()
            for name, state_now in current.items():
                if not (state_now["size"] == prepared[name]["size"] > 0
                        and state_now["phase"] == "READY" and state_now["invariant"]
                        and state_now["kernel_count"] > 0
                        and state_now["recapture"] == 0
                        and state_now["replay"] > previous[name]["replay"]):
                    raise RuntimeError(f"unstable {name} Graph replay: {previous[name]} -> {state_now}")
            previous = current
            if (result.extra["prefix_len"] != expected_profile["execution_len"]
                    or result.extra["real_prefix_len"] != real_prefix_len
                    or result.extra["action_chunk_size_resolved"] != 32):
                raise RuntimeError(f"unexpected execution geometry: {result.extra}")
            return result.actions.clone(), elapsed, dict(phase_times), result.extra

        for _ in range(args.warmup):
            warm, _, _, _ = infer(images_a)
        for _ in range(args.runs):
            action, elapsed, timing, extra = infer(images_a)
            if not torch.equal(action, warm):
                raise RuntimeError("identical A input changed action during REPLAY")
            outputs["A"] = action
            samples.append(elapsed)
            for name in phases:
                phases[name].append(timing.get(name, 0.0))
        outputs["B"], _, _, _ = infer(images_b)
        outputs["B_repeat"], _, _, _ = infer(images_b)
        outputs["A_restored"], _, _, _ = infer(images_a)
        checks = {"B_repeat_exact": torch.equal(outputs["B"], outputs["B_repeat"]),
                  "A_restore_exact": torch.equal(outputs["A"], outputs["A_restored"])}
        if args.vision_pair:
            # Exercise the pair's second camera, or the third-camera single group.
            images_c = changed_camera_images(images_a, -1)
            outputs["C"], _, _, _ = infer(images_c)
            outputs["C_repeat"], _, _, _ = infer(images_c)
            outputs["A_after_C"], _, _, _ = infer(images_a)
            checks.update(C_changes_action=not torch.equal(outputs["A"], outputs["C"]),
                          C_repeat_exact=torch.equal(outputs["C"], outputs["C_repeat"]),
                          A_after_C_exact=torch.equal(outputs["A"], outputs["A_after_C"]))
        report.update({"latency": summarize(samples),
                       "phases": {name: summarize(values) for name, values in phases.items()},
                       "graphs_final": snapshot(), "checks": checks, "last_extra": extra,
                       "B_changes_action": not torch.equal(outputs["A"], outputs["B"])})
        torch.save({**outputs, "noise": noise, "proprioception": state, "state_mask": state_mask,
                    "dof_mask": dof}, args.output_dir / "outputs.pt")
        if not all(checks.values()):
            raise RuntimeError(f"changed-input replay failed: {checks}")
        report["status"] = "pass"
        def metrics(a, b):
            delta = a.float() - b.float()
            return {"mse": float(delta.square().mean()), "max_abs": float(delta.abs().max()),
                    "cosine": float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)),
                    "exact": torch.equal(a, b)}
        if previous_outputs is not None:
            report["previous_outputs"] = {"path": str(args.compare.resolve()), "metrics": {
                name: {"all": metrics(outputs[name], previous_outputs[name]),
                       "active": metrics(outputs[name][:, :7], previous_outputs[name][:, :7])}
                for name in outputs if name in previous_outputs}}
        if args.golden is not None:
            golden_metrics = {}
            for name, count in (("all", 26), ("active", 7)):
                a, b = outputs["A"][:, :count], reference[:, :count]
                golden_metrics[name] = metrics(a, b)
            report["cpu_golden"] = {"path": str(args.golden.resolve()), "metrics": golden_metrics}
            if args.precision == "fp16" and tuple(args.image_size) == (448, 252):
                passed = golden_metrics["all"]["mse"] <= 1e-4
                report["cpu_golden"].update({"mse_budget": 1e-4, "mse_pass": passed})
                if not passed:
                    report["status"] = "numeric_failed"
        if args.cpu_thread_sweep:
            report["cpu_thread_sweep"] = {}
            thread_outputs = {}
            for threads in args.cpu_thread_sweep:
                effective_cpu = _CPU["configure_cpu_threads"](threads)
                for _ in range(args.warmup):
                    sweep_warm, _, _, _ = infer(images_a)
                sweep_times, sweep_phases = [], {name: [] for name in phases}
                for _ in range(args.runs):
                    a, elapsed, timing, _ = infer(images_a)
                    if not torch.equal(a, sweep_warm):
                        raise RuntimeError(f"CPU{threads} identical A changed during REPLAY")
                    sweep_times.append(elapsed)
                    for name in sweep_phases:
                        sweep_phases[name].append(timing.get(name, 0.0))
                out = {"A": a}
                out["B"], _, _, _ = infer(images_b)
                out["B_repeat"], _, _, _ = infer(images_b)
                out["A_restored"], _, _, _ = infer(images_a)
                sweep_checks = {"B_changes_action": not torch.equal(a, out["B"]),
                                "B_repeat_exact": torch.equal(out["B"], out["B_repeat"]),
                                "A_restore_exact": torch.equal(a, out["A_restored"])}
                if args.vision_pair:
                    out["C"], _, _, _ = infer(images_c)
                    out["C_repeat"], _, _, _ = infer(images_c)
                    out["A_after_C"], _, _, _ = infer(images_a)
                    sweep_checks.update(C_changes_action=not torch.equal(a, out["C"]),
                                        C_repeat_exact=torch.equal(out["C"], out["C_repeat"]),
                                        A_after_C_exact=torch.equal(a, out["A_after_C"]))
                entry = {"cpu_threads": torch.get_num_threads(),
                         "cpu_runtime": effective_cpu,
                         "latency": summarize(sweep_times),
                         "phases": {k: summarize(v) for k, v in sweep_phases.items()},
                         "checks": sweep_checks, "graphs_final": snapshot(),
                         "vs_primary": {k: metrics(v, outputs[k]) for k, v in out.items()}}
                report["cpu_thread_sweep"][str(threads)] = entry
                thread_outputs[str(threads)] = out
                if not all(sweep_checks.values()):
                    raise RuntimeError(f"CPU{threads} changed-input replay failed: {sweep_checks}")
            torch.save(thread_outputs, args.output_dir / "cpu_thread_outputs.pt")
            _CPU["configure_cpu_threads"](args.cpu_threads)
    except Exception:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
        _LOG.error("%s", report["error"])
    finally:
        if policy is not None:
            try:
                policy.close()
            except Exception:
                report["status"] = "error"
                report["cleanup_error"] = traceback.format_exc()
        _RESULTS["add_result_fields"](report, args.output_dir,
            completed=report["status"] in {"pass", "numeric_failed"},
            timing=_RESULTS["latency_timing"](report.get("latency", {}).get("samples_ms", []),
                report.get("timing_scope", "public infer including CPU physical actions; load, Graph preparation and warmup excluded")),
            runtime_checks=("finite output shape", "Graph REPLAY/no recapture", "changed-input deterministic replay"),
            reference_keys=("previous_outputs", "cpu_golden"),
            output_names=(("outputs.pt", "cpu_thread_outputs.pt")
                          if args.cpu_thread_sweep else ("outputs.pt",)))
        _RESULTS["write_report"](report, args.output_dir)
    _LOG.debug("report=%s", report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
