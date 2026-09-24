"""Qwen3.5-MoE checkpoint loader.

Like the dense Qwen3.5 loader, this adapter-local module owns the model-specific
quant declaration and tensor contract. Quantized checkpoints are built on
meta, populated from safetensors with ``assign=True``, and keep their file-backed
storage. Raw FP8 Parameters and scales are installed only after ``model.half``
so Hugging Face never casts the byte codes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText, GenerationConfig

from rpu_backend.adapters.qwen3_5_moe.quant_scope import (
    EXACT_PROFILE_ID,
    expected_fp8_weight_names,
    scale_name,
    validate_exact_profile,
    validate_layer_types,
)
from rpu_backend.api.errors import (
    RPUBackendError,
    RPUUnsupportedDtypeError,
    UnsupportedModelError,
)
from rpu_backend.runtime.log import _LOG

_CONFIG_KWARGS = frozenset({
    "cache_dir", "revision", "subfolder", "local_files_only",
    "trust_remote_code", "token", "force_download", "proxies",
})


def _layer_types(config) -> tuple[str, ...]:
    text_config = getattr(config, "text_config", config)
    layer_types = tuple(getattr(text_config, "layer_types", None) or ())
    n_layers = int(text_config.num_hidden_layers)
    if len(layer_types) != n_layers:
        raise UnsupportedModelError(
            f"Qwen3.5-MoE layer_types length {len(layer_types)} != "
            f"num_hidden_layers {n_layers}."
        )
    try:
        return validate_layer_types(layer_types)
    except ValueError as exc:
        raise UnsupportedModelError(str(exc)) from exc


def parse_quant_config(config):
    """Return the exact controlled mixed-E4M3 declaration."""
    try:
        profile = validate_exact_profile(config, require_quant=True)
        return profile["quant_config"]
    except (TypeError, ValueError) as exc:
        raise UnsupportedModelError(str(exc)) from exc


def _shard_paths(ckpt_dir: Path) -> list[Path]:
    single = ckpt_dir / "model.safetensors"
    if single.is_file():
        return [single]
    index_path = ckpt_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RPUBackendError(
            "Qwen3.5-MoE quantized checkpoint has neither model.safetensors "
            f"nor model.safetensors.index.json: {ckpt_dir}"
        )
    with index_path.open() as handle:
        weight_map = json.load(handle)["weight_map"]
    shards = [ckpt_dir / name for name in sorted(set(weight_map.values()))]
    missing = [str(shard) for shard in shards if not shard.is_file()]
    if missing:
        raise RPUBackendError(
            f"Qwen3.5-MoE quantized checkpoint shard(s) missing: {missing}"
        )
    return shards


def _rotary_buffer(module, name):
    """Rebuild non-persistent text/vision rotary buffers left on meta."""
    if name not in ("inv_freq", "original_inv_freq"):
        return None

    config = getattr(module, "config", None)
    if config is not None and hasattr(module, "compute_default_rope_parameters"):
        rope_type = getattr(module, "rope_type", None) or "default"
        if rope_type == "default":
            rope_init_fn = module.compute_default_rope_parameters
        else:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
        inv_freq, attention_scaling = rope_init_fn(
            config, device=torch.device("cpu")
        )
        module.attention_scaling = attention_scaling
        inv_freq = inv_freq.detach().to(device="cpu").contiguous()
        return inv_freq.clone() if name == "original_inv_freq" else inv_freq

    dim = getattr(module, "dim", None)
    theta = getattr(module, "theta", None)
    if dim is not None and theta is not None and name == "inv_freq":
        return 1.0 / (
            theta
            ** (torch.arange(0, int(dim), 2, dtype=torch.float32) / int(dim))
        )
    return None


def _materialize_meta_buffers(model) -> None:
    unresolved: list[str] = []
    for module_name, module in model.named_modules():
        for buffer_name, buffer in list(module._buffers.items()):
            if buffer is None or buffer.device.type != "meta":
                continue
            replacement = _rotary_buffer(module, buffer_name)
            if replacement is None:
                unresolved.append(
                    f"{module_name}.{buffer_name}" if module_name else buffer_name
                )
                continue
            module.register_buffer(buffer_name, replacement, persistent=False)
    if unresolved:
        raise RPUBackendError(
            f"Qwen3.5-MoE loader left {len(unresolved)} buffer(s) "
            f"unmaterialized: {sorted(unresolved)[:8]}"
        )


def _pop_fp8_tensors(state_dict, expected, parameters):
    """Pop raw-FP8 tensors for post-``half`` installation without cloning."""
    consumed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    seen: set[str] = set()
    for weight_name in sorted(name for name in state_dict if name in expected):
        weight = state_dict.pop(weight_name)
        sname = scale_name(weight_name)
        if sname not in state_dict:
            raise RPUBackendError(
                f"Qwen3.5-MoE mixed-FP8 checkpoint has {weight_name} but no "
                f"{sname}"
            )
        scale = state_dict.pop(sname)
        if weight.dtype != torch.uint8:
            raise RPUBackendError(
                f"Qwen3.5-MoE FP8 tensor {weight_name} must be raw uint8, got "
                f"{weight.dtype}"
            )
        expected_scale_shape = tuple(weight.shape[:-1])
        if scale.dtype != torch.float16 or tuple(scale.shape) != expected_scale_shape:
            raise RPUBackendError(
                f"Qwen3.5-MoE {sname} must be FP16 {expected_scale_shape}, got "
                f"{scale.dtype} {tuple(scale.shape)}"
            )
        if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
            raise RPUBackendError(
                f"Qwen3.5-MoE {sname} must contain only finite positive scales"
            )
        parameter = parameters.get(weight_name)
        if parameter is None:
            raise RPUBackendError(
                f"Qwen3.5-MoE checkpoint has no model Parameter for {weight_name}"
            )
        if tuple(parameter.shape) != tuple(weight.shape):
            raise RPUBackendError(
                f"Qwen3.5-MoE {weight_name} has shape {tuple(weight.shape)}, "
                f"model expects {tuple(parameter.shape)}"
            )
        # Converter output is contiguous, so this retains safetensors storage.
        consumed[weight_name] = (weight.contiguous(), scale.contiguous())
        seen.add(weight_name)

    stray = sorted(name for name in state_dict if name.endswith("_scale"))
    if stray:
        raise RPUBackendError(
            f"Qwen3.5-MoE checkpoint carries {len(stray)} scale tensor(s) "
            f"outside the mixed-FP8 scope: {stray[:5]}"
        )
    return consumed, seen


def _install_fp8_tensors(consumed, model) -> None:
    for weight_name, (weight, scale) in consumed.items():
        module_name, parameter_name = weight_name.rsplit(".", 1)
        module = model.get_submodule(module_name)
        module._parameters[parameter_name] = nn.Parameter(
            weight, requires_grad=False
        )
        scale_attr = parameter_name + "_scale"
        if scale_attr in module._buffers:
            module._buffers[scale_attr] = scale
        else:
            module.register_buffer(scale_attr, scale)
        module.rpu_fp8_format = "fp8_e4m3"
        module.rpu_weight_mode = 5
        module.rpu_fp8_scale_axis = -1
        module.rpu_fp8_profile = EXACT_PROFILE_ID


def _assert_expected_unexpected(unexpected, model) -> None:
    patterns = [
        re.compile(pattern)
        for pattern in (
            getattr(model, "_keys_to_ignore_on_load_unexpected", None) or []
        )
    ]
    unknown = [
        name for name in unexpected
        if not any(pattern.search(name) for pattern in patterns)
    ]
    if unknown:
        raise RPUBackendError(
            f"Qwen3.5-MoE checkpoint has {len(unknown)} tensor(s) the model "
            f"does not accept: {sorted(unknown)[:8]}"
        )


def load_qwen3_5_moe_model(
    pretrained_name_or_path,
    *,
    config=None,
    dtype: torch.dtype = torch.float16,
    **hf_kwargs,
):
    """Stream the exact 35B-A3B mixed-E4M3 checkpoint onto a CPU skeleton."""
    if hf_kwargs.get("trust_remote_code", False) is not False:
        raise ValueError("RhinoForge requires trust_remote_code=False")
    if dtype is not torch.float16:
        raise RPUUnsupportedDtypeError(
            f"Qwen3.5-MoE requires dtype=torch.float16, got {dtype}."
        )

    if config is None:
        cfg_kwargs = {
            key: value for key, value in hf_kwargs.items()
            if key in _CONFIG_KWARGS
        }
        config = AutoConfig.from_pretrained(
            pretrained_name_or_path, **cfg_kwargs
        )
    quant = parse_quant_config(config)

    unsupported = sorted(set(hf_kwargs) - _CONFIG_KWARGS)
    if unsupported:
        raise ValueError(
            "Qwen3.5-MoE scope-aware loader accepts only checkpoint/config "
            f"location kwargs, got {unsupported}"
        )

    # AutoConfig already resolves subfolder; use that same location for the
    # generation sidecar and every weight shard, including supplied configs.
    ckpt_dir = Path(pretrained_name_or_path).expanduser() / hf_kwargs.get("subfolder", "")
    if not ckpt_dir.is_dir():
        raise RPUBackendError(
            "Qwen3.5-MoE quantized checkpoints must be a local directory, got "
            f"{str(ckpt_dir)!r}. Download it first."
        )

    layer_types = _layer_types(config)
    expected = set(expected_fp8_weight_names(layer_types))
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config, trust_remote_code=False)
    # from_config only derives defaults from the model/text config. The exact
    # checkpoint adds <|im_end|> to the EOS set in this sidecar; preserve it for
    # both public generation and callers performing their own greedy loop.
    model.generation_config = GenerationConfig.from_pretrained(
        ckpt_dir, local_files_only=True
    )
    # Inference installation performs irreversible storage replacement and must
    # not retain autograd views of streamed checkpoint tensors.
    model.requires_grad_(False)
    parameters = dict(model.named_parameters())
    absent_from_model = sorted(expected - set(parameters))
    if absent_from_model:
        raise RPUBackendError(
            "Qwen3.5-MoE skeleton is missing mixed-FP8 Parameter(s): "
            f"{absent_from_model[:8]}"
        )

    consumed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    seen: set[str] = set()
    unexpected: list[str] = []
    for shard in _shard_paths(ckpt_dir):
        state_dict = load_file(str(shard))
        shard_consumed, shard_seen = _pop_fp8_tensors(
            state_dict, expected, parameters
        )
        duplicate = sorted(seen & shard_seen)
        if duplicate:
            raise RPUBackendError(
                f"Qwen3.5-MoE checkpoint repeats tensor(s) across shards: {duplicate[:8]}"
            )
        consumed.update(shard_consumed)
        seen |= shard_seen
        result = model.load_state_dict(state_dict, assign=True, strict=False)
        unexpected.extend(result.unexpected_keys)
        del state_dict

    not_consumed = sorted(expected - seen)
    if not_consumed:
        raise RPUBackendError(
            "Qwen3.5-MoE checkpoint declares mixed FP8 but "
            f"{len(not_consumed)} scope tensor(s) were not consumed: "
            f"{not_consumed[:8]}"
        )
    _assert_expected_unexpected(unexpected, model)

    model.tie_weights()
    _materialize_meta_buffers(model)
    model.half()
    _install_fp8_tensors(consumed, model)

    meta_params = [
        name for name, parameter in model.named_parameters()
        if parameter.device.type == "meta"
    ]
    meta_buffers = [
        name for name, buffer in model.named_buffers()
        if buffer.device.type == "meta"
    ]
    if meta_params or meta_buffers:
        raise RPUBackendError(
            "Qwen3.5-MoE loader finished with meta tensors — "
            f"params={meta_params[:3]} buffers={meta_buffers[:3]}"
        )

    model._qwen3_5_moe_quant_config = dict(quant)
    model._qwen3_5_moe_profile = EXACT_PROFILE_ID
    _LOG.info(
        "Qwen3.5-MoE %s checkpoint loaded (%d FP8 tensors)",
        quant["profile"],
        len(consumed),
    )
    return model.eval()


__all__ = ["load_qwen3_5_moe_model", "parse_quant_config"]
