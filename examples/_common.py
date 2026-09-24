#!/usr/bin/env python3
"""Hardware-free validation and shared execution for the model examples."""

from __future__ import annotations

import argparse
import ast
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
MODEL_REGISTRY = ROOT / "python" / "rpu_backend" / "model_registry.py"
PROFILE_CATALOG = ROOT / "examples/configs/profiles.json"
RUNNER = ROOT / "examples" / "_runner.py"


# Architecture names bind configuration labels; runtime admission remains authoritative.
ARCH_FAMILIES = {
    "DINOv3ViTModel": "dinov3",
    "Gemma4ForConditionalGeneration": "gemma4",
    "LlamaForCausalLM": "llama",
    "Qwen3ForCausalLM": "qwen3",
    "Qwen3VLForConditionalGeneration": "qwen3-vl",
    "Qwen3_5ForConditionalGeneration": "qwen3-5",
    "Qwen3_5MoeForConditionalGeneration": "qwen3-5-moe",
}
# Profile names predate two runner target names. Keep the translation here;
# the catalog binds configuration identity, not model certification.
RUNNER_TARGETS = {"qwen3_5_text": "qwen3_5"}
_EXAMPLE_KEYS = frozenset({"profile_id", "architecture", "registry_alias", "target",
                           "size", "dtype", "opt_in"})
_RUN_KEYS = frozenset({"warmup", "runs", "hwperf", "profile", "output_dir", "seed",
                       "warmup_decode_steps", "torch_num_threads", "inference_mode",
                       "timing_mode", "summary_statistic"})


class ConfigError(ValueError):
    """The example cannot be safely mapped to a documented public profile."""


@lru_cache(maxsize=1)
def registry_aliases() -> frozenset[str]:
    """Read MODELS as literal data; importing rpu_backend may require a device."""
    tree = ast.parse(MODEL_REGISTRY.read_text(encoding="utf-8"), MODEL_REGISTRY.name)
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(target, ast.Name) and target.id == "MODELS":
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict) or not all(
                isinstance(key, str) and isinstance(item, str)
                for key, item in value.items()
            ):
                break
            return frozenset(value)
    raise ConfigError(f"literal MODELS mapping missing from {MODEL_REGISTRY}")


@lru_cache(maxsize=1)
def runner_targets() -> frozenset[str]:
    """Read the shared runner target tuple without importing torch."""
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), RUNNER.name)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "ALL_TARGETS"
                    for target in node.targets)
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
                return frozenset(value)
            break
    raise ConfigError(f"literal ALL_TARGETS tuple missing from {RUNNER}")


@lru_cache(maxsize=4)
def example_profiles(path: Path = PROFILE_CATALOG) -> dict[str, dict]:
    """Read example identity and configuration constraints, not certification."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ConfigError(f"profiles list missing from {path}")
    profiles: dict[str, dict] = {}
    for row in rows:
        profile_id = row.get("id") if isinstance(row, dict) else None
        if not isinstance(profile_id, str) or profile_id in profiles:
            raise ConfigError(f"invalid or duplicate profile id in {path}: {profile_id!r}")
        profiles[profile_id] = row
    return profiles


def profile_runner_target(profile: dict) -> str:
    example = profile.get("example")
    runner_target = example.get("runner_target") if isinstance(example, dict) else None
    target = RUNNER_TARGETS.get(runner_target, runner_target)
    if not isinstance(target, str):
        raise ConfigError(f"profile {profile.get('id')!r} has no example runner target")
    return target


def validate_qwen35_checkpoint(config: dict, checkpoint: str):
    """Bind the Qwen3.5 report label to actual metadata before loading weights."""
    from transformers import AutoConfig

    example = config["example"]
    profile = example_profiles()[example["profile_id"]]
    architecture = {"qwen3-5": "Qwen3_5ForConditionalGeneration",
                    "qwen3-5-moe": "Qwen3_5MoeForConditionalGeneration"}.get(profile["family"])
    if architecture is None or example.get("architecture", architecture) != architecture:
        raise ConfigError("Qwen3.5 architecture label does not match the selected profile")
    metadata = AutoConfig.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    if getattr(metadata, "architectures", None) not in ([architecture], (architecture,)):
        raise ConfigError(f"Qwen3.5 checkpoint requires architecture={architecture}; check {checkpoint}")
    declared_dtype = example.get("dtype", profile["precision"])
    if declared_dtype != profile["precision"]:
        raise ConfigError("Qwen3.5 dtype label does not match the selected profile")
    try:
        if profile["family"] == "qwen3-5-moe":
            if declared_dtype != "mixed-e4m3":
                raise ValueError("Qwen3.5-MoE requires declared dtype=mixed-e4m3")
            _backend_leaf("adapters/qwen3_5_moe/quant_scope.py").validate_exact_profile(metadata, require_quant=True)
        else:
            text = getattr(metadata, "text_config", metadata)
            _backend_leaf("adapters/qwen3_5/quant_scope.py").validate_declared_dtype(
                getattr(text, "layer_types", ()), getattr(metadata, "quant_config", None), declared_dtype)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Qwen3.5 checkpoint precision does not match dtype={declared_dtype!r}: {exc}") from exc
    return metadata


def validate_checkpoint_profile(config: dict, checkpoint: str) -> None:
    """Bind quantized examples to their recipe before loading weights.

    The public model loader still validates the complete geometry and tensor
    inventory. This check prevents another valid recipe/size in the alias's
    directory from silently running under the selected example's report label.
    """
    profile = example_profiles()[config["example"]["profile_id"]]
    limits = profile["input_envelope"]["limits"]
    expected = limits.get("checkpoint_quant_profile")
    checkpoint_format = limits.get("checkpoint_format")
    on_install = limits.get("on_install_quantization")
    if expected is None and checkpoint_format is None and on_install is None:
        return
    if on_install is not None or checkpoint_format == "qwen3_32b_w8a16":
        from transformers import AutoConfig

        metadata = AutoConfig.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False)
        fields = ("hidden_size", "intermediate_size", "num_hidden_layers",
                  "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size")
        matches = all(getattr(metadata, name, None) == limits[name] for name in fields)
        if checkpoint_format == "qwen3_32b_w8a16":
            from rpu_backend.quant.qwen3_profiles import is_qwen3_32b_w8a16_config

            matches = matches and is_qwen3_32b_w8a16_config(metadata)
        if not matches:
            raise ConfigError(
                f"profile {profile['id']} requires its exact Qwen3 geometry and "
                f"checkpoint format={checkpoint_format or 'dense source'}; check {checkpoint}")
        return
    if checkpoint_format is not None:
        from transformers import AutoConfig
        from rpu_backend.quant.load import (
            is_qwen3_vl_2b_w8a16_config, is_qwen3_vl_awq_config,
        )

        predicates = {
            "qwen3_vl_text_w8": is_qwen3_vl_2b_w8a16_config,
            "qwen3_vl_text_awq": is_qwen3_vl_awq_config,
        }
        predicate = predicates.get(checkpoint_format)
        metadata = AutoConfig.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False)
        if (predicate is None or not predicate(metadata)
                or metadata.text_config.hidden_size != limits["text_hidden_size"]):
            raise ConfigError(
                f"profile {profile['id']} requires checkpoint format={checkpoint_format} "
                f"and text hidden_size={limits['text_hidden_size']}; check {checkpoint}")
        return
    path = Path(checkpoint) / "config.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    quant = metadata.get("quant_config")
    if (not isinstance(quant, dict) or quant.get("profile") != expected
            or quant.get("method") != profile["precision"]):
        raise ConfigError(
            f"profile {profile['id']} requires checkpoint quant_config.profile={expected} "
            f"and method={profile['precision']}; check {path}")


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=2)
def _backend_leaf(relative):
    """Load a stdlib-only public helper without bootstrapping the backend."""
    return _load_file("_example_" + Path(relative).stem,
                      ROOT / "python/rpu_backend" / relative)


def _integer(value, label, minimum):
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{label} must be an integer >= {minimum}")


def validate_generation_options(config):
    """Reject incompatible measurement/input options before importing the backend."""
    example, inputs, run = config["example"], config["input"], config.get("run", {})
    target = example["target"]
    mode = run.get("timing_mode", "per_call")
    if mode not in ("per_call", "legacy_dense_vl"):
        raise ConfigError("[run].timing_mode must be per_call or legacy_dense_vl")
    if run.get("summary_statistic", "mean") not in ("mean", "median"):
        raise ConfigError("[run].summary_statistic must be mean or median")
    if "summary_statistic" in run and (target not in {
        "qwen3", "llama", "qwen3_5", "qwen3_vl", "qwen3_5_vision", "gemma4"
    } or (target == "qwen3_vl" and inputs.get("component") == "vision")):
        raise ConfigError("summary_statistic requires a text-generation example")
    if "image_size" in inputs:
        _integer(inputs["image_size"], "[input].image_size", 1)
        if target != "qwen3_5_vision" or not inputs.get("images"):
            raise ConfigError("image_size requires a Qwen3.5 image-generation input")
    if "prefill_filler_prompt" in inputs:
        if (target != "qwen3_5_vision" or "prefill_tokens" not in inputs
                or not isinstance(inputs["prefill_filler_prompt"], str)
                or not inputs["prefill_filler_prompt"]
                or "prefill_fill_token_id" in inputs or "prompt_token_ids" in inputs):
            raise ConfigError("prefill_filler_prompt requires Qwen3.5 VL prefill_tokens and no token-ID filler")
    if mode == "legacy_dense_vl":
        if (target != "qwen3_5_vision"
                or example.get("architecture") != "Qwen3_5ForConditionalGeneration"
                or example.get("dtype") != "fp16"
                or example.get("registry_alias") not in {
                    "qwen3_5-0.8b", "qwen3_5-2b", "qwen3_5-4b", "qwen3_5-9b"}):
            raise ConfigError("legacy_dense_vl timing requires a Dense FP16 Qwen3.5 VL profile")
        if (type(inputs.get("prefill_tokens")) is not int or inputs["prefill_tokens"] < 1
                or type(inputs.get("decode_steps")) is not int or inputs["decode_steps"] < 1
                or inputs.get("stop_on_eos") is not False):
            raise ConfigError("legacy_dense_vl timing requires fixed prefill_tokens/decode_steps and stop_on_eos=false")
        if run.get("inference_mode", True) is not True:
            raise ConfigError("legacy_dense_vl timing requires inference_mode=true")
        if run.get("warmup_decode_steps", inputs["decode_steps"]) != inputs["decode_steps"]:
            raise ConfigError("legacy_dense_vl timing requires full decode warmup")


def _validate_execution(value, *, profile=None):
    # The same normalizer as the public constructors, without package bootstrap.
    normalizer = _backend_leaf("api/_execution.py")
    if not isinstance(value, dict):
        raise ConfigError("[rpu_execution] must be a mapping")
    scopes = {key: item for key, item in value.items() if key != "components"}
    syntax_fields = {stage: normalizer._FIELDS for stage in normalizer._STAGES}
    root_fields = dict(syntax_fields)
    if profile is not None and profile["family"] in {"qwen3", "qwen3-vl", "qwen3-5", "qwen3-5-moe"}:
        root_fields["model"] = ("num_cores",)
    normalized = normalizer.normalize_rpu_execution(scopes, entry_point="example", supported=root_fields)
    cores = normalized.get("model", {}).get("num_cores", 8)
    if (profile is not None and profile["family"] == "qwen3-5"
            and profile["precision"] == "fp16" and profile["component"] == "text"
            and profile["variant"] in ("tp4", "tp6")):
        limits = profile["input_envelope"]["limits"]
        if (profile["id"] != f"{profile['asset_alias']}.fp16.text-tp{cores}"
                or limits.get("num_cores") != cores):
            raise ConfigError("Qwen3.5 text core choice must match its exact fixed profile")
    if profile is not None and profile["family"] == "qwen3-5-moe":
        limits = profile["input_envelope"]["limits"]
        component = profile["component"]
        runner_target = {"text": "qwen3_5", "multimodal": "qwen3_5_vision"}.get(component)
        if (cores not in (4, 6, 8)
                or runner_target is None
                or profile_runner_target(profile) != runner_target
                or profile["id"] != f"qwen3_5-35b-a3b.mixed-e4m3.{component}-tp{cores}"
                or profile["precision"] != "mixed-e4m3"
                or profile["asset_alias"] != "qwen3_5-35b-a3b-mixed"
                or limits.get("num_cores") != cores):
            raise ConfigError("Qwen3.5-MoE core choice must match its exact fixed TP4/TP6/TP8 profile")
    elif profile is not None and profile["family"] == "qwen3-5" and profile["precision"] == "w8a16" and cores != 8:
        limits = profile["input_envelope"]["limits"]
        if (profile["asset_alias"] != "qwen3_8-27b-w8a16"
                or profile["component"] != "text"
                or profile["id"] != f"qwen3_8-27b.w8a16.text-controlled-tp{cores}"
                or profile_runner_target(profile) != "qwen3_5"
                or limits.get("num_cores") != cores):
            raise ConfigError("legacy Qwen3.8-27B model.num_cores requires its exact W8 text profile")
    elif cores != 8:
        # This validates the selected registered profile without opening model
        # weights. The loader checks the actual AutoConfig before materializing.
        if profile["family"] == "qwen3-5":
            aliases = _backend_leaf("adapters/qwen3_5/cores.py").REDUCED_CORE_ASSET_ALIASES
        else:
            aliases = (normalizer.QWEN3_VL_REDUCED_CORE_ASSET_ALIASES
                       if profile["family"] == "qwen3-vl" and profile["component"] == "multimodal"
                       else normalizer.QWEN3_REDUCED_CORE_ASSET_ALIASES)
        if profile["precision"] != "fp16" or profile["asset_alias"] not in aliases:
            raise ConfigError("model.num_cores=4/6 requires an admitted dense FP16 Qwen3, Qwen3-VL or Qwen3.5 profile")
    components = value.get("components", {})
    if not isinstance(components, dict):
        raise ConfigError("[rpu_execution.components] must be a mapping")
    for name, scope in components.items():
        if not isinstance(name, str) or not name:
            raise ConfigError("execution component IDs must be non-empty strings")
        normalizer.normalize_rpu_execution(
            scope, entry_point=f"example component {name}", supported=syntax_fields)
    # Actual component IDs, capabilities, physical envelopes and exact-plan
    # feasibility are checked by the selected public constructor before mutation.


def load_config(path: Path) -> dict:
    """Check profile identity and syntax; assets/capability require the runtime."""
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if "pi05" in config:
        from _policy_examples import validate_pi05_config
        validate_pi05_config(config)
        return config
    if not {"example", "run", "input"} <= set(config) or set(config) - {
        "example", "run", "input", "rpu_execution", "runner"
    }:
        raise ConfigError("expected [example], [input], [run], optional [rpu_execution]/[runner.hw_profile]")
    for name, allowed in (("example", _EXAMPLE_KEYS), ("run", _RUN_KEYS)):
        if not isinstance(config[name], dict) or set(config[name]) - allowed:
            raise ConfigError(f"[{name}] has unsupported keys or is not a table")
    example, run, inputs = config["example"], config["run"], config["input"]
    if example.get("target") == "pi05":
        raise ConfigError("Pi0.5 uses [pi05]; migrate the old [example] configuration")
    if not isinstance(inputs, dict):
        raise ConfigError("[input] must be a table")
    profile = example_profiles().get(example.get("profile_id"))
    if profile is None:
        raise ConfigError(f"unknown profile_id: {example.get('profile_id')!r}")
    alias = example.get("registry_alias")
    if alias not in registry_aliases() or alias != profile["asset_alias"]:
        raise ConfigError(f"registry_alias {alias!r} does not match profile {profile['id']}")
    target = profile_runner_target(profile)
    if example.get("target") != target or target not in runner_targets():
        raise ConfigError(f"profile {profile['id']} requires runner target {target}")
    if "runtime_env" in inputs:
        raise ConfigError("[input].runtime_env is not accepted; use the profile's exact opt_in")
    if "dtype" in example:
        allowed_dtype = {"w4a16-kvint8": "w4a16"}.get(profile["precision"], profile["precision"])
        if example["dtype"] != allowed_dtype:
            raise ConfigError(f"profile {profile['id']} requires dtype {allowed_dtype}")
    architecture = example.get("architecture")
    if architecture is not None and (
        architecture not in ARCH_FAMILIES
        or ARCH_FAMILIES.get(architecture) != profile.get("family")
    ):
        raise ConfigError(f"architecture {architecture!r} does not match profile family")
    gates = {gate["name"]: gate["value"] for gate in profile["controlled_gates"]
             if gate["kind"] == "env"}
    if example.get("opt_in", {}) != gates:
        raise ConfigError(f"profile {profile['id']} requires exact opt_in={gates!r}")
    for key in ("hwperf", "profile", "inference_mode"):
        if type(run.get(key, False)) is not bool:
            raise ConfigError(f"[run].{key} must be boolean")
    for key, default, minimum in (("warmup", 1, 0), ("runs", 1, 1), ("seed", 0, 0)):
        _integer(run.get(key, default), f"[run].{key}", minimum)
    if "output_dir" in run and (not isinstance(run["output_dir"], str) or not run["output_dir"]):
        raise ConfigError("[run].output_dir must be a non-empty path")
    runner = config.get("runner", {})
    if not isinstance(runner, dict) or set(runner) - {"hw_profile"}:
        raise ConfigError("[runner] only accepts hw_profile")
    hw_profile = runner.get("hw_profile", {})
    if not isinstance(hw_profile, dict) or set(hw_profile) - {"enabled", "max_dumps"}:
        raise ConfigError("[runner.hw_profile] only accepts enabled/max_dumps")
    if type(hw_profile.get("enabled", False)) is not bool:
        raise ConfigError("[runner.hw_profile].enabled must be boolean")
    _integer(hw_profile.get("max_dumps", 32), "[runner.hw_profile].max_dumps", 1)
    if run.get("hwperf", False) or hw_profile.get("enabled", False):
        raise ConfigError("hardware trace export is unavailable in the public runtime; use run.profile for Torch profiling")
    for key in ("decode_steps", "inference_delay"):
        if key in inputs:
            _integer(inputs[key], f"[input].{key}", 0)
    for key in ("prefill_tokens", "num_steps"):
        if key in inputs:
            _integer(inputs[key], f"[input].{key}", 1)
    text_targets = {"qwen3", "llama", "qwen3_5", "qwen3_vl", "qwen3_5_vision", "gemma4"}
    vision_only = profile.get("component") == "vision" and target != "qwen3_5_vision"
    if "prefill_tokens" in inputs and (
        target not in text_targets or vision_only
    ):
        raise ConfigError("prefill_tokens requires a text-generation example")
    if "torch_num_threads" in run:
        _integer(run["torch_num_threads"], "[run].torch_num_threads", 1)
    if "warmup_decode_steps" in run:
        _integer(run["warmup_decode_steps"], "[run].warmup_decode_steps", 0)
        if target not in text_targets or vision_only:
            raise ConfigError("warmup_decode_steps requires a text-generation example")
        if run["warmup_decode_steps"] > inputs.get("decode_steps", 4):
            raise ConfigError("warmup_decode_steps must not exceed input.decode_steps")
    if "stop_on_eos" in inputs and type(inputs["stop_on_eos"]) is not bool:
        raise ConfigError("[input].stop_on_eos must be boolean")
    if "prompt_token_ids" in inputs:
        ids = inputs["prompt_token_ids"]
        if (not isinstance(ids, list) or not ids
                or any(type(value) is not int or value < 0 for value in ids)):
            raise ConfigError("[input].prompt_token_ids must be a non-empty list of non-negative integers")
        if target not in text_targets or profile.get("component") != "text":
            raise ConfigError("prompt_token_ids requires a text-only example")
    if "prefill_fill_token_id" in inputs:
        _integer(inputs["prefill_fill_token_id"], "[input].prefill_fill_token_id", 0)
        if "prefill_tokens" not in inputs:
            raise ConfigError("prefill_fill_token_id requires input.prefill_tokens")
    for key in ("prompt", "instruction", "checkpoint"):
        if key in inputs and (not isinstance(inputs[key], str) or not inputs[key]):
            raise ConfigError(f"[input].{key} must be non-empty text")
    if "images" in inputs and (
        not isinstance(inputs["images"], list) or not inputs["images"]
        or any(not isinstance(item, str) or not item for item in inputs["images"])
    ):
        raise ConfigError("[input].images must be a non-empty list of paths")
    if target == "qwen3_vl":
        component = profile["component"]
        expected = component if component in {"vision", "text"} else "multimodal"
        if inputs.get("component", "multimodal") != expected:
            raise ConfigError(f"profile {profile['id']} requires input.component={expected}")
    if target in {"wall_oss", "hy_vla", "lingbot2", "rhinovla", "halo"}:
        from _policy_runners import validate_policy_config
        validate_policy_config(config)
    validate_generation_options(config)
    try:
        _validate_execution(config.get("rpu_execution", {}), profile=profile)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    return config


def config_metadata(config):
    if "pi05" in config:
        from _policy_examples import config_metadata as pi_metadata
        return pi_metadata(config)
    return config["example"]


def run_example(expected_target, default_config):
    parser = argparse.ArgumentParser(description="Run one model configuration.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--check-config", action="store_true",
                        help="Check configuration without importing torch or accessing an RPU.")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    try:
        config = load_config(args.config.expanduser().resolve())
        example = config_metadata(config)
        if expected_target is not None and example["target"] != expected_target:
            raise ConfigError(f"entry expects target {expected_target!r}")
    except (ConfigError, OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        parser.error(str(exc))
    if args.check_config:
        print(f"configuration OK: {args.config} ({example['profile_id']}; configuration only)")
        return 0
    config["run"]["profile"] = bool(args.profile or config["run"].get("profile", False))
    from _runner import run
    return run(config)


def maybe_run_catalog(expected_target, default_config):
    """Keep existing public entry points compatible with both TOML layouts."""
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path, default=default_config)
    args, _ = probe.parse_known_args()
    with args.config.expanduser().open("rb") as stream:
        config = tomllib.load(stream)
    if "example" in config or "pi05" in config:
        return run_example(expected_target, default_config)
    if set(config) & {"input", "run"}:
        probe.error("[input]/[run] require the [example] or [pi05] configuration format")
    return None
