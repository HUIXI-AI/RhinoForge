"""Pi0.5 catalog adapter using the public policy and cold execution profiles."""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy


_PRECISIONS = {
    "fp16": "fp16", "w8a16": "w8a16",
    "w8a16_action_nvfp4": "w8_action_nvfp4",
    "w8a16_prefill_w8a8_action_nvfp4": "w8_prefill_a8_action_nvfp4",
}


def _leaf(relative):
    path = Path(__file__).resolve().parents[1] / "python/rpu_backend" / relative
    if not path.is_file():
        from importlib.metadata import distribution
        package = distribution("rhinoforge")
        path = Path(package.locate_file("rpu_backend/" + relative))
    return runpy.run_path(str(path))


def _execution(config):
    resolve = _leaf("api/_execution.py")["resolve_pi05_execution_components"]
    root, vision, text, action = resolve(config.get("rpu_execution"), entry_point="Pi0.5 example")
    # The resolver returns immutable views. Detach them before deriving a pair.
    execution = {key: {field: dict(values) if isinstance(values, dict) else values
                       for field, values in value.items()}
                 for key, value in root.items() if key != "components"}
    if "components" in root:
        execution["components"] = {name: {stage: dict(values) for stage, values in scopes.items()}
                                   for name, scopes in root["components"].items()}
    inputs = config["input"]
    cores = root.get("model", {}).get("num_cores", 8)
    cameras, tokens = inputs["cameras"], inputs.get("text_tokens", 32)
    precision = config["pi05"]["precision"]
    if cores != 8 and (precision not in ("fp16", "w8a16") or cameras != 3 or tokens != 32):
        raise ValueError("Pi0.5 TP4/TP6 requires FP16/W8A16, three cameras and T32")
    vision_chunk = vision.get("vision", {}).get("chunk_size", "auto")
    action_chunk = action.get("action", {}).get("chunk_size", "auto")
    pair = (256 * cameras + tokens) // 2
    prefill = text.get("prefill", {})
    if action_chunk not in ("auto", 64):
        raise ValueError("Pi0.5 Action chunk must be auto or 64")
    if cores == 8:
        if (vision_chunk not in ("auto", 256 * cameras)
                or prefill.get("chunk_size", "auto") not in ("auto", pair)
                or prefill.get("padding_rows", "auto") not in ("auto", 0)
                or prefill.get("padding_budget", 0) != 0):
            raise ValueError("Pi0.5 TP8 catalog requires the paired camera/text geometry without extra padding")
        execution.setdefault("prefill", {})["chunk_size"] = pair
        child = execution.get("components", {}).get("language_model", {}).get("prefill")
        if child is not None:
            child["chunk_size"] = pair
    elif vision_chunk not in ("auto", 256):
        raise ValueError("Pi0.5 TP4/TP6 requires serial Vision chunk 256 or auto")
    execution.setdefault("model", {})["num_cores"] = cores
    return execution


def validate_pi05_config(config):
    required = {"pi05", "input", "run"}
    if not required <= set(config) or set(config) - (required | {"rpu_execution"}):
        raise ValueError("Pi0.5 examples require [pi05], [input], [run], and optional [rpu_execution]")
    allowed = {
        "pi05": {"precision"},
        "input": {"checkpoint", "batch", "cameras", "text_tokens", "num_steps"},
        "run": {"warmup", "runs", "seed", "output_dir", "profile", "hwperf", "torch_num_threads", "inference_mode"},
    }
    for name, keys in allowed.items():
        if not isinstance(config[name], dict) or set(config[name]) - keys:
            raise ValueError(f"[{name}] contains unsupported keys or is not a table")
    if config["pi05"].get("precision") not in _PRECISIONS:
        raise ValueError(f"[pi05].precision must be one of {tuple(_PRECISIONS)}")
    inputs, options = config["input"], config["run"]
    if type(inputs.get("cameras")) is not int or inputs["cameras"] not in (2, 3):
        raise ValueError("[input].cameras must be 2 or 3")
    if type(inputs.get("text_tokens", 32)) is not int or inputs.get("text_tokens", 32) not in (32, 64, 96, 128):
        raise ValueError("[input].text_tokens must be 32, 64, 96, or 128")
    if type(inputs.get("num_steps", 10)) is not int or inputs.get("num_steps", 10) != 10:
        raise ValueError("Pi0.5 examples require num_steps=10")
    for key in ("checkpoint", "batch"):
        if not isinstance(inputs.get(key), str) or not inputs[key].strip():
            raise ValueError(f"[input].{key} must be a nonempty path or checkpoint alias")
    for key, default, minimum in (("warmup", 2, 0), ("runs", 6, 1), ("seed", 0, 0), ("torch_num_threads", 8, 1)):
        if type(options.get(key, default)) is not int or options.get(key, default) < minimum:
            raise ValueError(f"[run].{key} must be an integer >= {minimum}")
    for key in ("profile", "hwperf", "inference_mode"):
        if type(options.get(key, False)) is not bool:
            raise ValueError(f"[run].{key} must be boolean")
    if options.get("hwperf", False):
        raise ValueError("hardware trace collection is not available from public examples")
    if "output_dir" in options and (not isinstance(options["output_dir"], str) or not options["output_dir"].strip()):
        raise ValueError("[run].output_dir must be a nonempty path")
    _execution(config)


def config_metadata(config):
    inputs = config["input"]
    precision = config["pi05"]["precision"]
    return {"target": "pi05", "profile_id": f"pi05.{inputs['cameras']}cam.t{inputs.get('text_tokens', 32)}.{precision}",
            "registry_alias": inputs["checkpoint"], "dtype": precision, "opt_in": {}}


def _generic_environment(precision):
    profile = _leaf("adapters/pi05/optimized.py")
    env = {"RPU_PI05_" + name: "0" for name in (*profile["_ON"], *profile["_OFF"])}
    env.update(RPU_PI05_FUSED_DENOISE="1", RPU_PI05_DENOISE_UNROLL="1",
               RPU_PI05_SIGLIP_W8A16=str(int(precision != "fp16")),
               RPU_PI05_ADARMS_DENSE_W8A16=str(int(precision != "fp16")),
               RPU_PI05_SIGLIP_W8A16_SCOPE="all", RPU_PI05_DENOISE_GEGLU_M50="0",
               RPU_PI05_DENOISE_W4_FASTPATH="0", RPU_PI05_DENOISE_W4_GEGLU_M50="0",
               RPU_PI05_DENOISE_W4_GEGLU_N64="0", RPU_PI05_DENOISE_NVFP4_GEGLU_M50="0",
               RPU_PI05_PREFILL_GATEUP_W8A8="0", RPU_PI05_VISION_OWNER_LN="0")
    return env


class _PolicyOwner:
    def __init__(self, environment):
        conflicts = [name for name, value in environment.items()
                     if name in os.environ and os.environ[name] != value]
        if conflicts:
            raise ValueError("environment conflicts with Pi0.5 generic profile: " + ", ".join(conflicts))
        self.previous = {name: os.environ.get(name) for name in environment}
        self.policy = None
        self.closed = False
        os.environ.update(environment)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.policy is not None:
                self.policy.close()
        finally:
            for name, value in self.previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def prepare_pi05(config):
    validate_pi05_config(config)
    for name in ("RPU_PI05_LOAD_NOISE", "RPU_PI05_LOAD_PREFIX_EMBS"):
        if os.environ.get(name):
            raise ValueError(f"public Pi0.5 examples reject diagnostic input replacement {name}")
    import torch
    from rpu_backend.api import Pi05Policy
    from rpu_backend.model_registry import model_path
    from rpu_backend.adapters.pi05.optimized import validate_checkpoint, validate_camera_batch

    inputs = config["input"]
    execution = _execution(config)
    paired = execution["model"]["num_cores"] == 8
    precision = config["pi05"]["precision"]
    checkpoint = Path(inputs["checkpoint"]).expanduser()
    if not checkpoint.is_dir():
        checkpoint = Path(model_path(inputs["checkpoint"]))
    quant_path = checkpoint / "rpu_quant_config.json"
    quant = json.loads(quant_path.read_text()) if quant_path.is_file() else {}
    validate_checkpoint(quant, "w4a16" if "nvfp4" in precision else precision)
    batch = torch.load(Path(inputs["batch"]).expanduser(), map_location="cpu", weights_only=True)
    if not isinstance(batch, dict):
        raise ValueError("Pi0.5 batch must contain a tensor dictionary")
    profile = {"precision": _PRECISIONS[precision], "num_cameras": inputs["cameras"],
               "text_tokens": inputs.get("text_tokens", 32)} if paired else None
    owner = _PolicyOwner({} if paired else _generic_environment(precision))
    try:
        owner.policy = Pi05Policy.from_pretrained(str(checkpoint), dtype=torch.float16,
            trust_remote_code=False, rpu_execution=execution, optimized_profile=profile)
        policy = owner.policy
        if not paired:
            model_config = policy._lerobot_policy.config
            validate_camera_batch(batch, model_config.image_features, cameras=3, text_tokens=32)
            if (model_config.chunk_size != 50 or model_config.max_action_dim != 32
                    or model_config.num_inference_steps != 10):
                raise ValueError("Pi0.5 requires horizon 50, max_action_dim 32 and 10 denoise steps")
        policy.to("rpu")
        policy.prepare_graphs(batch, num_steps=10, precompute_adarms=paired)
    except BaseException as error:
        try:
            owner.close()
        except BaseException as cleanup_error:
            error.add_note(f"Pi0.5 setup cleanup failed: {cleanup_error!r}")
        raise
    def infer():
        # Pi's total-latency convention includes a caller-owned CPU action.
        return policy.predict_action_chunk(batch, num_steps=10).detach().float().cpu().clone()
    return infer, owner
