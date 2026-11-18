#!/usr/bin/env python3
"""Hardware-free validation and shared execution for the model examples."""

from __future__ import annotations

import ast
from functools import lru_cache
import importlib.util
from importlib import import_module, machinery, metadata
import json
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[2]
API_REFERENCE = ROOT / "docs" / "api_reference.md"
MODEL_REGISTRY = ROOT / "python" / "rpu_backend" / "model_registry.py"
PLANNER_REGISTRY = ROOT / "examples/configs/profiles.json"
EXAMPLE_REGISTRY = ROOT / "examples/configs/profiles.json"
RUNNER = ROOT / "examples" / "common" / "measurement.py"


# Public architecture names group exact runnable profiles; support still comes
# from each profile rather than from this presentation mapping.
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
# the profile remains the support authority.
RUNNER_TARGETS = {"qwen3_5_text": "qwen3_5", "lingbot_v1": "lingbot"}
_EXAMPLE_KEYS = frozenset({"profile_id", "architecture", "registry_alias", "target",
                           "size", "dtype", "opt_in"})
_RUN_KEYS = frozenset({"warmup", "runs", "hwperf", "profile", "output_dir", "seed",
                       "warmup_decode_steps", "torch_num_threads", "cpu_threads", "inference_mode",
                       "timing_mode", "summary_statistic"})
_TABLE_START = "<!-- supported-models-start -->"
_TABLE_END = "<!-- supported-models-end -->"


class ConfigError(ValueError):
    """The example cannot be safely mapped to a documented public profile."""


# Unified layout: one [model] table names the runner target, precision and
# model assets. Profile-bound entries add profile_id; dedicated runners do not.
# Legacy [example] and per-model identity tables still load unchanged.
DEDICATED_TARGETS = frozenset({"pi05", "g05", "hy_vla", "lingbot2", "wall_oss",
                               "wall_oss_sorting", "qwen3_asr", "qwen3_tts"})
LEGACY_IDENTITY_TABLES = DEDICATED_TARGETS | {"example"}
# Model assets that the runners read from [input].
MODEL_ASSET_KEYS = frozenset({
    "checkpoint", "fp16_checkpoint", "w8_checkpoint", "action_w4_checkpoint",
    "fp16_registry_alias", "qwen3vl_registry_alias", "release_root", "training_config",
    "runtime_factory", "factory_config", "asset_manifest", "source_dir",
})
# Speech runners keep checkpoint/source in their own model table.
_ASSETS_IN_MODEL_TABLE = frozenset({"qwen3_asr", "qwen3_tts"})


def normalize_config(raw: dict) -> dict:
    """Return the runner-facing layout of a parsed TOML file.

    A [model] file is mapped to the legacy layout the runners validate:
    profile-bound entries become [example] (precision -> dtype), dedicated
    entries become their named table, model assets move to [input], and
    G0.5 linear_acc32 becomes linear_accumulation. Legacy files are returned
    unchanged. Pure and import-light, so tools that read TOML directly can
    call it before indexing tables.
    """
    if "model" not in raw:
        return raw
    config = dict(raw)
    model = config.pop("model")
    if not isinstance(model, dict):
        raise ConfigError("[model] must be a table")
    mixed = sorted(LEGACY_IDENTITY_TABLES & set(config))
    if mixed:
        raise ConfigError(f"[model] cannot be combined with legacy table(s) {mixed}")
    model = dict(model)
    target = model.get("target")
    if not isinstance(target, str) or not target:
        raise ConfigError("[model].target is required")
    profile_bound = "profile_id" in model
    if not profile_bound and target not in DEDICATED_TARGETS:
        raise ConfigError(f"[model].target={target!r} requires [model].profile_id; "
                          f"targets without a profile are {sorted(DEDICATED_TARGETS)}")
    inputs = config.get("input", {})
    if not isinstance(inputs, dict):
        raise ConfigError("[input] must be a table")
    inputs = dict(inputs)
    if profile_bound or target not in _ASSETS_IN_MODEL_TABLE:
        for key in sorted(MODEL_ASSET_KEYS & set(model)):
            if key in inputs:
                raise ConfigError(f"{key} belongs in [model]; remove it from [input]")
            inputs[key] = model.pop(key)
    if profile_bound:
        if "dtype" in model:
            raise ConfigError("[model] uses precision; dtype is the legacy [example] name")
        if "precision" in model:
            model["dtype"] = model.pop("precision")
        config["example"] = model
    else:
        del model["target"]
        if target == "g05" and "linear_acc32" in model:
            if "linear_accumulation" in model:
                raise ConfigError("[model] uses linear_acc32; linear_accumulation is the legacy G0.5 name")
            value = model.pop("linear_acc32")
            if type(value) is not bool:
                raise ConfigError("[model].linear_acc32 must be boolean")
            model["linear_accumulation"] = "acc32" if value else "acc16"
        config[target] = model
    if "input" in config or inputs:
        config["input"] = inputs
    return config


def validate_logging_config(config: dict) -> dict:
    """Resolve diagnostics and summary settings without mutating the config."""
    options = config.get("logging", {})
    if not isinstance(options, dict) or set(options) - {"level", "summary"}:
        raise ConfigError("[logging] only accepts level/summary and must be a table")
    level = options.get("level", "ERROR")
    levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    if not isinstance(level, str) or level not in levels:
        raise ConfigError(f"[logging].level must be one of {levels}")
    summary = options.get("summary", True)
    if type(summary) is not bool:
        raise ConfigError("[logging].summary must be boolean")
    return {"level": level, "summary": summary}


def supported_rows(path: Path = API_REFERENCE) -> list[tuple[str, ...]]:
    """Parse the five-column fenced support table without importing the package."""
    text = path.read_text(encoding="utf-8")
    if text.count(_TABLE_START) != 1 or text.count(_TABLE_END) != 1:
        raise ConfigError(f"{path} must contain exactly one supported-models fence pair")
    body = text.split(_TABLE_START, 1)[1].split(_TABLE_END, 1)[0]
    rows: list[tuple[str, ...]] = []
    for line in body.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        if not cells or cells[0] == "HF arch" or set(cells[0]) <= {"-", ":"}:
            continue
        if len(cells) != 5 or not all(cells):
            raise ConfigError(f"malformed supported-model row in {path}: {line}")
        rows.append(cells)
    if not rows or len({row[0] for row in rows}) != len(rows):
        raise ConfigError(f"{path} has an empty or duplicate supported-model table")
    return rows


@lru_cache(maxsize=1)
def registry_aliases() -> frozenset[str]:
    """Read MODELS as literal data; importing rpu_backend may require a device."""
    if not MODEL_REGISTRY.is_file():
        # Closed wheels compile this leaf module; load it without package bootstrap.
        return frozenset(_backend_leaf("model_registry.py").MODELS)
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
def planner_profiles(path: Path = PLANNER_REGISTRY) -> dict[str, dict]:
    """Read the machine support authority without importing the backend."""
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
    return {key: row for key, row in profiles.items()
            if row.get("example_only") is not True}


def profile_runner_target(profile: dict) -> str:
    example = profile.get("example")
    runner_target = example.get("runner_target") if isinstance(example, dict) else None
    target = RUNNER_TARGETS.get(runner_target, runner_target)
    if not isinstance(target, str):
        raise ConfigError(f"profile {profile.get('id')!r} has no example runner target")
    return target


@lru_cache(maxsize=1)
def example_profiles() -> dict[str, dict]:
    """Add caller-factory identities without changing planner support admission."""
    profiles = dict(planner_profiles())
    data = json.loads(EXAMPLE_REGISTRY.read_text(encoding="utf-8"))
    rows = data.get("profiles") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ConfigError(f"profiles list missing from {EXAMPLE_REGISTRY}")
    for row in rows:
        if isinstance(row, dict) and row.get("example_only") is not True:
            continue
        profile_id = row.get("id") if isinstance(row, dict) else None
        if (not isinstance(profile_id, str) or not profile_id
                or profile_id in profiles):
            raise ConfigError(f"invalid or duplicate example profile: {profile_id!r}")
        if (row.get("example_only") is not True
                or row.get("family") != "rhinovla"
                or row.get("example", {}).get("runner_target") != "rhinovla"
                or "support" in row or "hw_status" in row):
            raise ConfigError("caller-factory templates cannot declare planner support")
        profiles[profile_id] = row
    return profiles


def _validate_unbound_qwen_checkpoint(profile: dict, checkpoint: str) -> None:
    """Bind older example labels whose catalog predates checkpoint-format fields.

    Match model metadata, not a directory name. The public loader remains the
    authority for complete tensor inventories and runtime admission.
    """
    family = profile["family"]
    if family not in {"qwen3", "qwen3-vl"}:
        return
    from transformers import AutoConfig

    metadata = AutoConfig.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False)
    multimodal = family == "qwen3-vl"
    prefix = "qwen3-vl-" if multimodal else "qwen3-"
    size = profile["asset_alias"].removeprefix(prefix).split("-")[0]
    # These bind the catalog's size label, independently of loader admission.
    geometries = ({"2b": (28, 2048, 6144), "4b": (36, 2560, 9728),
                   "8b": (36, 4096, 12288), "32b": (64, 5120, 25600)}
                  if multimodal else
                  {"0.6b": (28, 1024, 3072), "1.7b": (28, 2048, 6144),
                   "4b": (36, 2560, 9728), "8b": (36, 4096, 12288),
                   "14b": (40, 5120, 17408), "32b": (64, 5120, 25600)})
    text = getattr(metadata, "text_config", None) if multimodal else metadata
    fields = ("num_hidden_layers", "hidden_size", "intermediate_size")
    actual = tuple(getattr(text, key, None) for key in fields)
    architecture = "Qwen3VLForConditionalGeneration" if multimodal else "Qwen3ForCausalLM"
    if (size not in geometries or actual != geometries[size]
            or any(type(value) is not int for value in actual)
            or getattr(metadata, "model_type", None) != ("qwen3_vl" if multimodal else "qwen3")
            or tuple(getattr(metadata, "architectures", ()) or ()) != (architecture,)):
        raise ConfigError(f"profile {profile['id']} requires its declared Qwen3 architecture and size; check {checkpoint}")

    precision = profile["precision"]
    if precision == "fp16":
        matches = all(getattr(owner, key, None) is None
                      for owner in (metadata, text)
                      for key in ("quant_config", "quantization_config"))
    elif not multimodal and precision == "w8a16":
        from rpu_backend.quant.qwen3_profiles import _W8_METADATA

        quant = getattr(metadata, "quant_config", None)
        matches = (isinstance(quant, dict) and quant == _W8_METADATA
                   and all(type(quant[key]) is type(value) for key, value in _W8_METADATA.items())
                   and getattr(metadata, "quantization_config", None) is None
                   and getattr(metadata, "tie_word_embeddings", None) is False)
    elif profile["asset_alias"] == "qwen3-vl-32b-w8a16":
        from rpu_backend.quant.load import is_qwen3_vl_32b_w8a16_config

        matches = is_qwen3_vl_32b_w8a16_config(metadata)
    else:
        matches = False
    if not matches:
        raise ConfigError(
            f"profile {profile['id']} checkpoint precision/scope does not match "
            f"{precision}; check {checkpoint}")


def _is_official_qwen35_27b_profile(profile):
    return (profile.get("family") == "qwen3-5"
            and profile.get("asset_alias") in {
                "qwen3_5-27b", "qwen3_5-27b-fp16",
                "qwen3_5-27b-w8a16", "qwen3_5-27b-w4a16"})


def validate_checkpoint_profile(config: dict, checkpoint: str) -> None:
    """Bind quantized examples to their recipe before loading weights.

    The public model loader still validates the complete geometry and tensor
    inventory. This check prevents another valid recipe/size in the alias's
    directory from silently running under the selected example's report label.
    """
    profile = planner_profiles()[config["example"]["profile_id"]]
    limits = profile["input_envelope"]["limits"]
    if _is_official_qwen35_27b_profile(profile):
        metadata = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
        scope = _backend_leaf("adapters/qwen3_5/quant_scope.py")
        try:
            matches = (metadata.get("architectures") == ["Qwen3_5ForConditionalGeneration"]
                       and isinstance(metadata.get("text_config"), dict)
                       and scope.validate_modern_27b_profile(
                           metadata, allow_fp16=profile["precision"] == "fp16"))
            if not matches:
                raise ValueError("requires the official dense 27B geometry and uniform text quantization")
            specs = scope.resolve_quant_specs(
                scope.is_full_from_layer_types(metadata["text_config"]["layer_types"]),
                metadata.get("quant_config"))
            expected = {"fp16": scope.FP16, "w8a16": scope.W8A16,
                        "w4a16": scope.W4A16_PGRP}[profile["precision"]]
            if {spec.method for spec in specs.values()} != {expected}:
                raise ValueError(f"checkpoint precision does not match {profile['precision']}")
        except (TypeError, ValueError, KeyError) as exc:
            raise ConfigError(f"profile {profile['id']} rejects checkpoint: {exc}") from exc
        return
    if profile["family"] == "qwen3-5-moe" and (
            "checkpoint_profile" in limits or "_model_override" in config):
        # This stdlib-only admission uses the same checkpoint recipe as the
        # public loader, before tokenizer/weight loading. A path override must
        # not silently relabel another supported MoE checkpoint as RTN.
        metadata = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
        scope = _backend_leaf("adapters/qwen3_5_moe/quant_scope.py")
        expected_profile = limits.get("checkpoint_profile", scope.EXACT_PROFILE_ID)
        try:
            actual = scope.validate_exact_profile(metadata)["profile"]
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"profile {profile['id']} rejects checkpoint: {exc}") from exc
        if actual != expected_profile:
            raise ConfigError(
                f"profile {profile['id']} requires checkpoint profile "
                f"{expected_profile}, got {actual}")
        return
    expected = limits.get("checkpoint_quant_profile")
    checkpoint_format = limits.get("checkpoint_format")
    awq_geometries = {
        "qwen3_5_2b_quanttrio_awq_g128": (24, 2048, 6144),
        "qwen3_5_4b_quanttrio_awq_g128": (32, 2560, 9216),
        "qwen3_5_9b_quanttrio_awq_g128": (32, 4096, 12288),
        "qwen3_5_27b_activation_awq_g128": (64, 5120, 17408),
    }
    if checkpoint_format in awq_geometries:
        metadata = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
        scope = _backend_leaf("adapters/qwen3_5/quant_scope.py")
        try:
            quant = scope.resolve_quant_config(metadata)
            text = metadata.get("text_config", {})
            geometry = tuple(text.get(k) for k in
                             ("num_hidden_layers", "hidden_size", "intermediate_size"))
            expected_profile = ("activation_awq_g128_v1" if "activation_awq" in checkpoint_format
                                else "quanttrio_awq_g128")
            valid = (quant is not None and quant.get("profile") == expected_profile
                     and geometry == awq_geometries[checkpoint_format])
        except (TypeError, ValueError, KeyError) as exc:
            raise ConfigError(f"invalid QuantTrio AWQ G128 checkpoint: {exc}") from exc
        if not valid:
            raise ConfigError(f"profile requires its exact {checkpoint_format} checkpoint")
        return
    if checkpoint_format in ("qwen3_5_2b_w4_mlp_only_g32", "qwen3_5_27b_w4_mlp_only_g32"):
        metadata = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
        scope = _backend_leaf("adapters/qwen3_5/quant_scope.py")
        try:
            text = metadata.get("text_config", {})
            expected_geometry = ((24, 2048, 6144) if "_2b_" in checkpoint_format
                                 else (64, 5120, 17408))
            valid = (metadata.get("model_type") == "qwen3_5"
                     and metadata.get("architectures") == ["Qwen3_5ForConditionalGeneration"]
                     and isinstance(metadata.get("text_config"), dict)
                     and scope.validate_mlp_only_text_profile(metadata)
                     and tuple(text.get(k) for k in ("num_hidden_layers", "hidden_size",
                                                    "intermediate_size")) == expected_geometry)
            if valid and "_27b_" in checkpoint_format:
                valid = scope.validate_named_27b_mixed_profile(metadata)
        except (TypeError, ValueError, KeyError) as exc:
            raise ConfigError(f"invalid MLP-only G32 checkpoint: {exc}") from exc
        if not valid:
            raise ConfigError("profile requires its exact Qwen3.5 MLP-only G32 size and recipe")
        return
    if checkpoint_format in ("qwen3_5_2b_w4_mlp_w8_attn_gdn_g32",
                              "qwen3_5_27b_w4_mlp_w8_attn_gdn_g32"):
        metadata = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
        scope = _backend_leaf("adapters/qwen3_5/quant_scope.py")
        # Bind this example to its measured recipe; generic v2 runtime admission
        # remains unchanged. A profile label alone cannot establish the scope.
        row = ((2048, 6144, 24, 8, 2, 16) if "_2b_" in checkpoint_format
               else (5120, 17408, 64, 24, 4, 48))
        geometry = dict(zip(("hidden_size", "intermediate_size", "num_hidden_layers",
                             "num_attention_heads", "num_key_value_heads", "linear_num_value_heads"), row))
        geometry.update(head_dim=256, linear_num_key_heads=16,
                        linear_key_head_dim=128, linear_value_head_dim=128,
                        linear_conv_kernel_dim=4, vocab_size=248320)
        layers = [scope.LINEAR_ATTENTION] * 3 + [scope.FULL_ATTENTION]
        try:
            quant = scope.resolve_quant_config(metadata)
            text = metadata.get("text_config")
            valid = (metadata.get("model_type") == "qwen3_5"
                     and metadata.get("architectures") == ["Qwen3_5ForConditionalGeneration"]
                     and isinstance(text, dict)
                     and all(type(text.get(k)) is int and text[k] == v
                             for k, v in geometry.items())
                     and text.get("layer_types") == layers * (row[2] // 4)
                     and text.get("quant_config") is None
                     and text.get("quantization_config") is None
                     and isinstance(quant, dict)
                     and quant.get("format_version") == 2
                     and quant.get("profile") == "w4_mlp_w8_attn_gdn"
                     and "layer_overrides" not in quant
                     and quant.get("storage", "int8") == "int8"
                     and quant.get("scale_dtype", "float16") == "float16")
            if valid:
                specs = scope.resolve_quant_specs(
                    scope.is_full_from_layer_types(text["layer_types"]), quant)
                valid = all(
                    spec.method == (scope.W4A16_PGRP if ".mlp." in name else scope.W8A16)
                    and (".mlp." not in name or spec.group_size == 32)
                    for name, spec in specs.items())
            if valid and "_27b_" in checkpoint_format:
                valid = scope.validate_named_27b_mixed_profile(metadata)
        except (TypeError, ValueError, KeyError) as exc:
            raise ConfigError(f"invalid MLP W4/G32 + Attention/GDN W8 checkpoint: {exc}") from exc
        if not valid:
            raise ConfigError("profile requires its exact Qwen3.5 MLP W4/G32 + Attention/GDN W8 size and recipe")
        return
    on_install = limits.get("on_install_quantization")
    if expected is None and checkpoint_format is None and on_install is None:
        _validate_unbound_qwen_checkpoint(profile, checkpoint)
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


def required_example(profile: dict) -> bool:
    status = profile.get("support", {}).get("status")
    example = profile.get("example", {})
    internal_evaluation = (
        example.get("reason_code") == "INTERNAL_CONTROLLED_EVALUATION"
        and status == "CONTROLLED_EVALUATION"
        and profile.get("hw_status") == "IMPLEMENTED_UNVERIFIED_HW"
        and profile.get("input_envelope", {}).get("kind") == "EXACT"
        and example.get("eligible") is False and example.get("runner_target") is None
        and any(gate.get("kind") == "env" and gate.get("value") == "1"
                for gate in profile.get("controlled_gates", []))
        # Private evidence paths are omitted from the public catalog.
    )
    return (status not in {"UNSUPPORTED", "ASSET_ONLY"}
            and not internal_evaluation
            and example.get("reason_code") not in {
                "INTERNAL_COMPONENT_HARDWARE_CASE", "COMPOSITE_COMPONENT_HARDWARE_CASE"})


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cpu_runtime():
    from examples.common import cpu
    return cpu


def _backend_leaf_path(relative):
    """Locate a source/compiled backend leaf without importing the backend."""
    path = ROOT / "python/rpu_backend" / relative
    if not path.is_file():
        try:
            dist = metadata.distribution("rpu_backend")
        except metadata.PackageNotFoundError as exc:
            raise ConfigError("release examples require the matching installed rpu_backend wheel") from exc
        stem = "rpu_backend/" + relative.removesuffix(".py")
        names = {stem + suffix for suffix in [".py", *machinery.EXTENSION_SUFFIXES]}
        matches = [dist.locate_file(item) for item in (dist.files or ())
                   if str(item) in names]
        if len(matches) != 1:
            raise ConfigError(f"installed wheel must provide exactly one {relative} leaf")
        path = Path(matches[0])
    if not path.is_file():
        raise ConfigError(f"backend leaf is missing: {relative}")
    return path


@lru_cache(maxsize=2)
def _backend_leaf(relative):
    """Load one stdlib-only leaf from source or the installed closed wheel."""
    path = _backend_leaf_path(relative)
    # Compiled leaf init symbols use the filename stem, not a synthetic alias.
    return _load_file(Path(relative).stem, path)


def _integer(value, label, minimum):
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{label} must be an integer >= {minimum}")


def validate_generation_options(config):
    """Reject incompatible measurement/input options before importing the backend."""
    example, inputs, run = config["example"], config["input"], config.get("run", {})
    target = example["target"]
    prepared_fields = {"prepared_input_safetensors", "prepared_input_sha256"}
    if set(inputs) & prepared_fields:
        if target != "qwen3_vl" or inputs.get("component") != "multimodal":
            raise ConfigError("prepared_input_safetensors requires a Qwen3-VL multimodal profile")
        if not prepared_fields <= set(inputs):
            raise ConfigError("prepared_input_safetensors and prepared_input_sha256 must be supplied together")
        if (not isinstance(inputs["prepared_input_safetensors"], str)
                or not inputs["prepared_input_safetensors"]):
            raise ConfigError("[input].prepared_input_safetensors must be a non-empty path")
        digest = inputs["prepared_input_sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ConfigError("[input].prepared_input_sha256 must be a SHA256 hex digest")
        extra = set(inputs) - prepared_fields - {"component", "decode_steps", "stop_on_eos"}
        if extra:
            raise ConfigError("prepared Qwen3-VL input cannot also specify prompt/images or input extensions: "
                              + ", ".join(sorted(extra)))
    if (target == "qwen3_5_vision"
            and example.get("architecture") == "Qwen3_5MoeForConditionalGeneration"
            and set(inputs) & {"video", "videos", "pixel_values_videos", "video_grid_thw"}):
        raise ConfigError("Qwen3.5-MoE image profiles do not support video inputs")
    mode = run.get("timing_mode", "per_call")
    if mode not in ("per_call", "legacy_dense_vl", "forward_argmax"):
        raise ConfigError("[run].timing_mode must be per_call, legacy_dense_vl or forward_argmax")
    profile = planner_profiles().get(example.get("profile_id"), {})
    moe_text_perf = (target == "qwen3_5" and profile.get("family") == "qwen3-5-moe"
                     and "prefill_tokens" in inputs)
    file_fields = {"prompt_file", "prompt_sha256", "input_ids_sha256"}
    if moe_text_perf or mode == "forward_argmax" or set(inputs) & file_fields:
        if (not moe_text_perf or profile.get("component") != "text"
                or mode != "forward_argmax"):
            raise ConfigError("frozen prose and forward_argmax require a Qwen3.5-MoE text performance profile")
        if (set(inputs) & {"prompt", "prompt_token_ids", "prefill_fill_token_id", "prefill_filler_prompt"}
                or not isinstance(inputs.get("prompt_file"), str) or not inputs["prompt_file"]):
            raise ConfigError("MoE text performance requires prompt_file; synthetic token/filler inputs are forbidden")
        for field in ("prompt_sha256", "input_ids_sha256"):
            value = inputs.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ConfigError(f"[input].{field} must be a SHA256 hex digest")
        _integer(inputs.get("prefill_tokens"), "[input].prefill_tokens", 1)
        _integer(inputs.get("decode_steps"), "[input].decode_steps", 0)
        if inputs.get("stop_on_eos") is not False:
            raise ConfigError("MoE text performance requires stop_on_eos=false and fixed decode_steps")
    if run.get("summary_statistic", "mean") not in ("mean", "median"):
        raise ConfigError("[run].summary_statistic must be mean or median")
    if "summary_statistic" in run and (target not in {
        "qwen3", "llama", "qwen3_5", "qwen3_vl", "qwen3_5_vision", "gemma4"
    } or (target == "qwen3_vl" and inputs.get("component") == "vision")):
        raise ConfigError("summary_statistic requires a text-generation example")
    if "image_size" in inputs:
        _integer(inputs["image_size"], "[input].image_size", 1)
        if target != "gr00t" and (target != "qwen3_5_vision" or not inputs.get("images")):
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


_CORE_BUDGETS = (4, 6, 8)
_FIXED_EIGHT = {
    "g05": "原生实现只有 8 核：动作专家只支持 8 核，W8 视觉和多相机批处理也只实现了 8 核",
    "g05_nvfp4": "原生实现只有 8 核：NVFP4 动作 GEMM 按 8 核切分，动作专家只支持 8 核",
    "gr00t": "原生实现只有 8 核：DiT 的时间表、常驻 FF1 kernel 和交叉注意力 KV 都按 8 核布局",
    "hy_vla": "原生实现只有 8 核：HyVLA 专用融合 kernel 要求 8 核，视觉塔 16 个注意力头也不能分到 6 核",
    "internvla": "原生实现只有 8 核：视觉、DINOv2 和 NavDP owner 按 8 核编写，骨干的 QKV 偏置和 RoPE 布局不满足少核解码器的准入",
    "lingbot2": "原生实现只有 8 核：运行器、权重转换和多视角 KV 语言路径都要求 8 核",
    "lingbot2_exact_z2": "exact-Z2 的 SPM 驻留计划按 8 核切分（每核 8,303,616 字节），少核需要新的驻留计划",
    "rhinovla": "原生实现只有 8 核：视觉 merger 融合 kernel 和动作专家 owner 按 8 核编写",
    "wall_oss": "原生实现只有 8 核：语言模型的注意力和 MLP 固定 8 核，动作专家和 Qwen2.5-VL 视觉 owner 也按 8 核编写",
    "pi05_nvfp4": "原生实现只有 8 核：Action NVFP4 的 striped_v2 权重和张量缩放表只有 8 核布局，少核准入只接受 FP16/W8A16 权重",
    "pi05_prefill_w8a8": "原生实现只有 8 核：Prefill W8A8 依赖只支持 8 核的配对 prefill 路径，Action 仍为 NVFP4",
    "halo": "延期模型：拓扑固定为注意力 2 核、MLP 8 核",
    "lingbot": "延期模型：只有 8 核实现",
}
_PI05_REDUCED_PRECISION = "Pi0.5 的 4/6 核只接受 FP16/W8A16 权重：其他精度（如旧版整模型 INT4 w4a16_kvint8）只有 8 核实现"
_PI05_REDUCED_TEXT = "Pi0.5 的 4/6 核只接受 T32：T64–T160 只有 8 核的配对 prefill 布局，T192 的 P960 前缀也只按 8 核绑定"
_MODEL_SCOPE_TABLES = frozenset({"pi05", "hy_vla", "lingbot2", "qwen3_asr", "qwen3_tts"})


def _fixed_reason(config: dict, target: str | None, profile: dict | None) -> str | None:
    if target == "pi05":
        precision = config["pi05"]["precision"]
        if precision == "w8a16_prefill_w8a8_action_nvfp4":
            return _FIXED_EIGHT["pi05_prefill_w8a8"]
        if "nvfp4" in precision:
            return _FIXED_EIGHT["pi05_nvfp4"]
        if precision not in ("fp16", "w8a16"):
            return _PI05_REDUCED_PRECISION
        if config.get("input", {}).get("text_tokens", 32) != 32:
            return _PI05_REDUCED_TEXT
        return None
    if target == "g05":
        precision = (config.get("g05") or config.get("example") or {}).get("precision", "")
        return _FIXED_EIGHT["g05_nvfp4" if "nvfp4" in str(precision) else "g05"]
    if target == "lingbot2" and profile is not None and profile["id"].endswith(".exact-z2"):
        return _FIXED_EIGHT["lingbot2_exact_z2"]
    if target in ("wall_oss", "wall_oss_rtc", "wall_oss_sorting"):
        return _FIXED_EIGHT["wall_oss"]
    if target == "internvla_n1":
        return _FIXED_EIGHT["internvla"]
    return _FIXED_EIGHT.get(target)


def _declared_core_support(config: dict, target: str | None, profile: dict | None, validate) -> dict[int, str | None]:
    reason = _fixed_reason(config, target, profile)
    if reason is not None:
        return {4: reason, 6: reason, 8: None}
    if target in ("qwen3_asr", "qwen3_tts", "pi05"):
        return dict.fromkeys(_CORE_BUDGETS)
    support = {}
    for cores in _CORE_BUDGETS:
        try:
            validate(cores)
        except (TypeError, ValueError) as error:
            support[cores] = str(error)
        else:
            support[cores] = None
    return support


def _core_identity(config):
    target = "rhinovla" if "env" in config else config_target(config)
    # Only profile-bound [example] entries need their profile here; dedicated
    # runners are declared by their own table.
    profile_id = config.get("example", {}).get("profile_id")
    return target, example_profiles().get(profile_id) if profile_id else None


def core_support(config):
    """Map 4/6/8 to None when this configuration accepts the budget, else the reason."""
    declared = _declared_core_support
    target, profile = _core_identity(config)

    def validate(cores):
        execution = json.loads(json.dumps(config.get("rpu_execution", {})))
        execution.setdefault("model", {})["num_cores"] = cores
        _validate_execution(execution, profile=profile)

    return declared(config, target, profile, validate)


def _check_core_budget(config):
    """Report a declared refusal with its reason; drop an explicit 8 a fixed runner does not read.

    Runner-decided budgets (Qwen profiles) keep their own validation below.
    """
    MODEL_SCOPE_TABLES, fixed_reason = _MODEL_SCOPE_TABLES, _fixed_reason
    execution = config.get("rpu_execution")
    model = execution.get("model") if isinstance(execution, dict) else None
    if not isinstance(model, dict) or model.get("num_cores") not in (4, 6, 8):
        return
    try:
        reason = fixed_reason(config, *_core_identity(config))
    except (KeyError, TypeError, ValueError):
        return  # The runner's own validation reports the malformed table.
    if reason is None:
        return
    if model["num_cores"] != 8:
        raise ConfigError(f"rpu_execution.model.num_cores = {model['num_cores']} is not supported here: {reason}")
    if set(model) == {"num_cores"} and not MODEL_SCOPE_TABLES & set(config):
        del execution["model"]
        if not execution:
            del config["rpu_execution"]


def _validate_execution(value, *, profile=None):
    # The same normalizer as the public constructors, without package bootstrap.
    normalizer = _backend_leaf("api/_execution.py")
    if not isinstance(value, dict):
        raise ConfigError("[rpu_execution] must be a mapping")
    scopes = {key: item for key, item in value.items() if key != "components"}
    syntax_fields = {stage: normalizer._FIELDS for stage in normalizer._STAGES}
    root_fields = dict(syntax_fields)
    if profile is not None and profile["family"] in {"qwen3", "qwen3-vl", "qwen3-5", "qwen3-5-moe", "llama", "dinov3", "gemma4"}:
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
    if profile is not None and profile["family"] == "qwen3-5" and profile.get("variant") == "legacy-text":
        limits = profile["input_envelope"]["limits"]
        allowed = tuple(limits.get("allowed_num_cores", (4, 6, 8)))
        if (profile["id"] != "qwen3_5-27b.w8a16.text-legacy"
                or profile["asset_alias"] != "qwen3_8-27b-w8a16"
                or profile["precision"] != "w8a16"
                or profile["component"] != "text"
                or profile_runner_target(profile) != "qwen3_5"
                or cores not in allowed
                or limits.get("num_cores") is not None):
            raise ConfigError("Qwen3.5 legacy 27B accepts the unified 4/6/8 core profile")
    elif profile is not None and _is_official_qwen35_27b_profile(profile):
        limits = profile["input_envelope"]["limits"]
        precision, component = profile["precision"], profile["component"]
        suffix = "" if cores == 8 else f"-tp{cores}"
        allowed_cores = tuple(limits.get("allowed_num_cores", (8,)))
        suffix = "" if limits.get("num_cores") is None else suffix
        expected_alias = f"qwen3_5-27b-{precision}"
        if (precision not in ("fp16", "w8a16", "w4a16") or cores not in allowed_cores
                or component not in ("text", "vision")
                or profile["id"] != f"qwen3_5-27b.{precision}.{component}-controlled{suffix}"
                or profile["asset_alias"] != expected_alias
                or profile_runner_target(profile) != ("qwen3_5" if component == "text" else "qwen3_5_vision")
                or limits.get("num_cores") not in (None, cores) or limits.get("batch_size") != 1
                or (precision == "w4a16" and limits.get("group_size") != 32)):
            raise ConfigError("official Qwen3.5-27B model.num_cores must match its registered precision profile")
    elif profile is not None and profile["family"] == "qwen3-5-moe":
        limits = profile["input_envelope"]["limits"]
        component = profile["component"]
        runner_target = {"text": "qwen3_5", "multimodal": "qwen3_5_vision"}.get(component)
        int4_profiles = {
            "qwen3_5-35b-a3b.rtn-w4a16-g128.text": (
                "qwen3_5-35b-a3b-rtn-w4a16", "qwen3_5_moe_35b_a3b_tp8_routed_rtn_w4a16_g128", "text"),
            "qwen3_5-35b-a3b.gptq-w4a16-g128.text": (
                "qwen3_5-35b-a3b-gptq-w4a16", "qwen3_5_moe_35b_a3b_tp8_routed_gptq_w4a16_g128", "text"),
            "qwen3_5-35b-a3b.rtn-w4a16-g128.multimodal": (
                "qwen3_5-35b-a3b-rtn-w4a16", "qwen3_5_moe_35b_a3b_tp8_routed_rtn_w4a16_g128", "multimodal"),
            "qwen3_5-35b-a3b.gptq-w4a16-g128.multimodal": (
                "qwen3_5-35b-a3b-gptq-w4a16", "qwen3_5_moe_35b_a3b_tp8_routed_gptq_w4a16_g128", "multimodal"),
        }
        if profile["id"] in int4_profiles:
            alias, checkpoint_profile, expected_component = int4_profiles[profile["id"]]
            if (cores not in limits.get("allowed_num_cores", (8,)) or component != expected_component
                    or profile_runner_target(profile) != runner_target
                    or profile["precision"] != "w4a16" or profile["asset_alias"] != alias
                    or limits.get("num_cores") not in (None, cores) or limits.get("batch_size") != 1
                    or limits.get("group_size") != 128
                    or limits.get("checkpoint_profile") != checkpoint_profile):
                raise ConfigError("Qwen3.5-MoE INT4 requires its registered routed-only G128/B1 profile")
            if normalized.get("prefill", {}).get("linear_acc32", False):
                raise ConfigError("Qwen3.5-MoE INT4 text supports ACC16 only; prefill.linear_acc32 must be false")
        elif (cores not in (4, 6, 8)
                or runner_target is None
                or profile_runner_target(profile) != runner_target
                or profile["id"] != f"qwen3_5-35b-a3b.mixed-e4m3.{component}"
                or profile["precision"] != "mixed-e4m3"
                or profile["asset_alias"] != "qwen3_5-35b-a3b-mixed"
                or cores not in limits.get("allowed_num_cores", ())):
            raise ConfigError("Qwen3.5-MoE core choice must match its registered profile")
    elif (profile is not None and profile["family"] == "qwen3"
          and profile["precision"] == "w4a16" and cores in (4, 6)):
        limits = profile["input_envelope"]["limits"]
        recipe = limits.get("on_install_quantization")
        variant = "w4a16-g32-lm-head" if recipe == "w4a16_lm_head" else "w4a16-g32"
        if (recipe not in ("w4a16", "w4a16_lm_head") or limits.get("group_size") != 32
                or profile["component"] != "text"
                or profile["asset_alias"] not in normalizer.QWEN3_REDUCED_CORE_ASSET_ALIASES
                or profile["id"] != f"{profile['asset_alias']}.{variant}.text"
                or cores not in limits.get("allowed_num_cores", ())):
            raise ConfigError("Qwen3 reduced-core W4 requires its registered on-install G32 recipe")
    elif (profile is not None and profile["family"] == "qwen3"
          and profile["precision"] == "w8a16" and profile["component"] == "text"
          and cores in (4, 6)):
        aliases = {f"qwen3-{size}b-w8a16-lmhead-int8" for size in ("0.6", "1.7", "4", "8", "14", "32")}
        if (profile["asset_alias"] not in aliases
                or profile["id"] != f"{profile['asset_alias']}.w8a16.text"):
            raise ConfigError("Qwen3 reduced-core W8 requires a registered decoder/head checkpoint")
    elif (profile is not None and profile["family"] == "qwen3-5"
          and profile["precision"] != "fp16" and cores != 8):
        if cores not in profile["input_envelope"]["limits"].get("allowed_num_cores", ()):
            raise ConfigError("A reduced core budget requires a profile with explicit admission")
    elif profile is not None and profile["family"] == "dinov3" and cores != 8:
        # The adapter re-checks the actual AutoConfig (exact ViT-B) before swizzle.
        if (profile["precision"] != "fp16" or profile["component"] != "vision"
                or cores not in profile["input_envelope"]["limits"].get("allowed_num_cores", ())):
            raise ConfigError("DINOv3 model.num_cores=4/6 requires the registered FP16 ViT-B vision profile")
    elif profile is not None and profile["family"] == "gemma4" and cores != 8:
        # Gemma4 E4B FP16 text: attention 2 at every budget; MLP/PLE 4/4 or 6/4.
        limits = profile["input_envelope"]["limits"]
        if (profile["id"] != "gemma4-e4b.fp16.text" or profile["precision"] != "fp16"
                or profile_runner_target(profile) != "gemma4"
                or cores not in limits.get("allowed_num_cores", (8,))):
            raise ConfigError("Gemma4 core choice requires its registered E4B FP16 text profile")
    elif profile is not None and profile["family"] == "qwen3-vl" and cores != 8:
        limits = profile["input_envelope"]["limits"]
        if (profile["precision"] not in ("fp16", "w8a16", "w4a16")
                or profile["component"] not in ("text", "vision", "multimodal")
                or profile_runner_target(profile) != "qwen3_vl"
                or cores not in limits.get("allowed_num_cores", ())
                or limits.get("num_cores") not in (None, cores)):
            raise ConfigError("Qwen3-VL core choice requires its registered dense precision profile")
    elif profile is not None and profile["family"] == "llama" and cores != 8:
        limits = profile["input_envelope"]["limits"]
        if (profile["precision"] != "fp16" or profile["component"] != "text"
                or profile["asset_alias"] not in normalizer.LLAMA_REDUCED_CORE_ASSET_ALIASES
                or profile["id"] != profile["asset_alias"] + ".fp16.text"
                or cores not in limits.get("allowed_num_cores", ())):
            raise ConfigError("A reduced Llama core budget requires the registered Llama-3.2-1B FP16 text profile")
    elif cores != 8:
        # This validates the selected registered profile without opening model
        # weights. The loader checks the actual AutoConfig before materializing.
        if profile["family"] == "qwen3-5":
            aliases = _backend_leaf("adapters/qwen3_5/cores.py").REDUCED_CORE_ASSET_ALIASES
        else:
            aliases = normalizer.QWEN3_REDUCED_CORE_ASSET_ALIASES
        if profile["precision"] != "fp16" or profile["asset_alias"] not in aliases:
            raise ConfigError("A reduced core budget requires an admitted dense FP16 Qwen3, Qwen3-VL or Qwen3.5 profile")
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
        config = normalize_config(tomllib.load(stream))
    validate_logging_config(config)
    try:
        cpu_runtime().resolve_cpu_threads(config)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    _check_core_budget(config)
    if "qwen3_asr" in config or "qwen3_tts" in config:
        from examples.runners.qwen3_audio.runner import validate_example_config, resolve_paths
        try:
            validate_example_config(config)
            resolve_paths(config, path.parent)
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
        return config
    if "env" in config:
        # RhinoVLA v3 factory/validation fragments have no public profile ID.
        # Validate their syntax without importing the device-facing factory.
        if set(config) - {"logging"} != {"env", "rpu_execution"} or not isinstance(config["env"], dict):
            raise ConfigError("RhinoVLA runtime config requires [env] and [rpu_execution], optional [logging]")
        for name, value in config["env"].items():
            if not name.startswith("RPU_RHINOVLA_") or value not in ("0", "1"):
                raise ConfigError("RhinoVLA [env] requires RPU_RHINOVLA_ switches with string 0/1 values")
        try:
            _validate_execution(config["rpu_execution"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
        return config
    if "wall_oss" in config:
        try:
            _wall_base_runner().validate_example_config(config)
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
        return config
    if "wall_oss_sorting" in config:
        try:
            _wall_sorting_runner().validate_example_config(config)
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
        return config
    if "g05" in config:
        try:
            _g05_runner().validate_example_config(config)
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
        return config
    if "pi05" in config:
        try:
            _pi05_runner().validate_example_config(config)
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
        return config
    if "lingbot2" in config:
        try:
            _lingbot2_runner().validate_example_config(config)
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
        return config
    if "hy_vla" in config:
        try:
            _hyvla_runner().validate_example_config(config)
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
        return config
    if not {"example", "run", "input"} <= set(config) or set(config) - {
        "example", "run", "input", "rpu_execution", "runner", "cpu", "logging"
    }:
        raise ConfigError("expected [example], [input], [run], optional [cpu]/[rpu_execution]/[runner.hw_profile]/[logging]")
    if isinstance(config["example"], dict) and "expansion_ref" in config["example"]:
        raise ConfigError("example.expansion_ref is retired; use the shared main/expansion runtime libraries")
    for name, allowed in (("example", _EXAMPLE_KEYS), ("run", _RUN_KEYS)):
        if not isinstance(config[name], dict) or set(config[name]) - allowed:
            raise ConfigError(f"[{name}] has unsupported keys or is not a table")
    example, run, inputs = config["example"], config["run"], config["input"]
    if example.get("target") == "pi05":
        raise ConfigError('Pi0.5 is a dedicated entry: use [model] target = "pi05" without profile_id')
    if not isinstance(inputs, dict):
        raise ConfigError("[input] must be a table")
    profile = example_profiles().get(example.get("profile_id"))
    if profile is None:
        raise ConfigError(f"unknown profile_id: {example.get('profile_id')!r}")
    if not required_example(profile) or profile.get("example", {}).get("eligible") is not True:
        raise ConfigError(f"profile {profile['id']} is not example.eligible=true")
    alias = example.get("registry_alias")
    if alias not in registry_aliases() or alias != profile["asset_alias"]:
        raise ConfigError(f"registry_alias {alias!r} does not match profile {profile['id']}")
    target = profile_runner_target(profile)
    if example.get("target") != target or target not in runner_targets():
        raise ConfigError(f"profile {profile['id']} requires runner target {target}")
    if target == "gr00t":
        try:
            import_module("examples.runners.gr00t.config").validate_config(config)
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
    if ("checkpoint" in inputs and target not in {"wall_oss_rtc", "g05"}
            and not profile.get("example_only")):
        raise ConfigError("[input].checkpoint is only accepted for exact Wall RTC/G0.5 releases")
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
    for key, default, minimum in (("warmup", 2, 0), ("runs", 1, 1), ("seed", 0, 0)):
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
    if "warmup_decode_steps" in run:
        _integer(run["warmup_decode_steps"], "[run].warmup_decode_steps", 0)
        if target not in text_targets or vision_only:
            raise ConfigError("warmup_decode_steps requires a text-generation example")
        if run["warmup_decode_steps"] > inputs.get("decode_steps", 4):
            raise ConfigError("warmup_decode_steps must not exceed input.decode_steps")
        if (run.get("warmup", 2) >= 2
                and run["warmup_decode_steps"] != inputs.get("decode_steps", 4)):
            raise ConfigError(
                "warmup >= 2 requires full decode warmup: omit warmup_decode_steps "
                "or set it equal to input.decode_steps; use warmup=0 or 1 only "
                "for cold-path diagnostics")
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
    if target == "rhinovla" and (
        inputs.get("num_steps", 10) != 10 or inputs.get("trust_model_code") is not True
    ):
        raise ConfigError("official RhinoVLA example requires num_steps=10 and trust_model_code=true")
    if profile.get("example_only"):
        try:
            import_module("examples.runners.rhinovla.runner").validate_rhinovla_config(config)
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
    validate_generation_options(config)
    if "prompt_file" in inputs:
        prompt_path = Path(inputs["prompt_file"]).expanduser()
        inputs["prompt_file"] = str((path.parent / prompt_path).resolve())
    if "prepared_input_safetensors" in inputs:
        prepared_path = Path(inputs["prepared_input_safetensors"]).expanduser()
        inputs["prepared_input_safetensors"] = str((path.parent / prepared_path).resolve())
    try:
        _validate_execution(config.get("rpu_execution", {}), profile=profile)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    return config


def _g05_runner():
    return import_module("examples.runners.g05.optimized")



def _pi05_runner():
    return import_module("examples.runners.pi05.optimized")


def _lingbot2_runner():
    return import_module("examples.runners.lingbot2.optimized")


def _hyvla_runner():
    return import_module("examples.runners.hy_vla.optimized")


def config_target(config):
    """Return the existing runner target of a validated, runnable configuration."""
    for target in ("wall_oss", "wall_oss_sorting", "g05", "pi05", "hy_vla", "lingbot2", "qwen3_asr", "qwen3_tts"):
        if target in config:
            return target
    return config.get("example", {}).get("target")


def example_metadata(config):
    if "lingbot2" in config:
        return _lingbot2_runner().registered_example(config)
    if "g05" in config:
        return _g05_runner().registered_example(config)
    if "wall_oss" in config:
        # New resolution/text/hybrid measurements do not inherit old support.
        return None
    if "pi05" in config:
        return _pi05_runner().registered_example(config)
    if "hy_vla" in config:
        return _hyvla_runner().registered_example(config)
    return config.get("example")


def _wall_sorting_runner():
    return import_module("examples.runners.wall_oss.sorting")


def _wall_base_runner():
    return import_module("examples.runners.wall_oss.base")
