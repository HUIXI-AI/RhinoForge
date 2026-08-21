"""W8A16 checkpoint loading for Qwen3.

HF's generic loader copies checkpoint tensors into existing Parameters and casts
them to that Parameter's dtype. A W8A16 checkpoint needs the projection weights
to stay int8, so this loader builds the model skeleton first and installs those
Parameters directly.
"""
from __future__ import annotations

import json
import os
import gc
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from rpu_backend.quant.convert_qwen3 import (
    EMBED_TOKENS_WEIGHT_NAME,
    LM_HEAD_WEIGHT_NAME,
    QUANT_PROJ_SUFFIXES,
)


_QWEN3_VL_32B_W8A16_QUANT_CONFIG = {
    "method": "w8a16",
    "mode": "per_channel_symmetric",
    "qaxis": 0,
    "skip_modules": ["lm_head"],
    "quantized_lm_head": False,
    "quantized_embed_tokens": False,
    "lm_head_untied": False,
    "embed_tokens_untied": False,
}

_W8A16_IMAGETEXT_STAGE_ATTR = "_rpu_w8a16_imagetext_load_plan"
_SAFE_FLOAT_DTYPES = frozenset({"F64", "F32", "F16", "BF16"})


@dataclass(frozen=True)
class _W8A16ImageTextLoadPlan:
    model_id: int
    tensor_shards: tuple[tuple[str, str], ...]
    layer_tensor_names: tuple[tuple[str, ...], ...]
    nondecoder_tensor_names: tuple[str, ...]
    parameter_names: tuple[str, ...]


def is_w8a16_config(config) -> bool:
    """Return True when an HF config declares the local W8A16 quant format."""
    qc = getattr(config, "quant_config", None)
    return isinstance(qc, dict) and qc.get("method") == "w8a16"


def is_qwen3_vl_32b_w8a16_config(config) -> bool:
    """Return True only for the controlled Qwen3-VL-32B W8A16 profile."""
    text = getattr(config, "text_config", None)
    vision = getattr(config, "vision_config", None)
    try:
        return (
            tuple(config.architectures) == ("Qwen3VLForConditionalGeneration",)
            and config.model_type == "qwen3_vl"
            and config.tie_word_embeddings is False
            and config.quant_config == _QWEN3_VL_32B_W8A16_QUANT_CONFIG
            and (
                text.model_type,
                int(text.hidden_size),
                int(text.intermediate_size),
                int(text.num_hidden_layers),
                int(text.num_attention_heads),
                int(text.num_key_value_heads),
                int(text.head_dim),
                int(text.vocab_size),
                text.hidden_act,
                float(text.rms_norm_eps),
            ) == (
                "qwen3_vl_text", 5120, 25600, 64, 64, 8, 128, 151936,
                "silu", 1e-6,
            )
            and (
                vision.model_type,
                int(vision.hidden_size),
                int(vision.intermediate_size),
                int(vision.depth),
                int(vision.num_heads),
                int(vision.patch_size),
                int(vision.temporal_patch_size),
                int(vision.spatial_merge_size),
                tuple(int(x) for x in vision.deepstack_visual_indexes),
                int(vision.out_hidden_size),
                vision.hidden_act,
            ) == (
                "qwen3_vl", 1152, 4304, 27, 16, 16, 2, 2,
                (8, 16, 24), 5120, "gelu_pytorch_tanh",
            )
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _safetensor_shards(ckpt_dir: Path) -> list[Path]:
    single = ckpt_dir / "model.safetensors"
    if single.is_file():
        return [single]

    index_path = ckpt_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"W8A16 checkpoint has neither model.safetensors nor "
            f"model.safetensors.index.json: {ckpt_dir}"
        )
    with index_path.open() as f:
        index = json.load(f)
    return [ckpt_dir / name for name in sorted(set(index["weight_map"].values()))]


def _resolve_checkpoint_dir(
    ckpt_dir: str,
    *,
    cache_dir: str | None = None,
    revision: str | None = None,
    subfolder: str = "",
    local_files_only: bool = False,
    token: str | bool | None = None,
    force_download: bool = False,
    proxies: dict | None = None,
    trust_remote_code: bool | None = None,
) -> Path:
    del trust_remote_code  # Config-only option; kept for from_pretrained kwarg symmetry.
    del proxies  # AutoConfig accepts it; snapshot_download in this env does not.
    subfolder = subfolder or ""

    local_path = Path(os.path.abspath(os.path.expanduser(ckpt_dir)))
    if local_path.is_dir():
        return local_path / subfolder if subfolder else local_path

    snapshot_path = Path(snapshot_download(
        repo_id=ckpt_dir,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
        token=token,
        force_download=force_download,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
    ))
    return snapshot_path / subfolder if subfolder else snapshot_path


def _parameter_owner(model: nn.Module, name: str) -> tuple[nn.Module, str, nn.Parameter]:
    if "." in name:
        module_name, param_name = name.rsplit(".", 1)
        module = model.get_submodule(module_name)
    else:
        module = model
        param_name = name
    param = module._parameters.get(param_name)
    if param is None:
        raise KeyError(f"W8A16 checkpoint tensor {name!r} has no matching model parameter")
    return module, param_name, param


def _install_parameter(model: nn.Module, name: str, tensor: torch.Tensor) -> None:
    module, param_name, old_param = _parameter_owner(model, name)
    if tuple(old_param.shape) != tuple(tensor.shape):
        raise ValueError(
            f"W8A16 checkpoint tensor {name!r} has shape {tuple(tensor.shape)}, "
            f"expected {tuple(old_param.shape)}"
        )
    requires_grad = bool(old_param.requires_grad and tensor.is_floating_point())
    module._parameters[param_name] = nn.Parameter(
        tensor.contiguous(), requires_grad=requires_grad)


def _materialize_qwen3_rotary_buffer(module: nn.Module, name: str) -> torch.Tensor | None:
    if name not in {"inv_freq", "original_inv_freq"}:
        return None
    config = getattr(module, "config", None)
    if config is None or not hasattr(module, "compute_default_rope_parameters"):
        return None

    rope_type = getattr(module, "rope_type", None)
    rope_params = getattr(config, "rope_parameters", None)
    if rope_type is None and isinstance(rope_params, dict):
        rope_type = rope_params.get("rope_type", "default")
    rope_type = rope_type or "default"

    if rope_type == "default":
        rope_init_fn = module.compute_default_rope_parameters
    else:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]

    inv_freq, attention_scaling = rope_init_fn(config, device=torch.device("cpu"))
    module.attention_scaling = attention_scaling
    inv_freq = inv_freq.detach().to(device="cpu").contiguous()
    return inv_freq.clone() if name == "original_inv_freq" else inv_freq


def _materialize_meta_buffers(model: nn.Module) -> None:
    """Populate non-persistent buffers omitted from safetensors checkpoints."""
    unresolved: list[str] = []
    for module_name, module in model.named_modules():
        for buffer_name, buffer in list(module._buffers.items()):
            if buffer is None or not getattr(buffer, "is_meta", False):
                continue
            replacement = _materialize_qwen3_rotary_buffer(module, buffer_name)
            full_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
            if replacement is None:
                unresolved.append(full_name)
                continue
            module.register_buffer(buffer_name, replacement, persistent=False)

    if unresolved:
        raise KeyError(
            f"W8A16 checkpoint left {len(unresolved)} meta buffer(s) without "
            f"materialization support: {sorted(unresolved)[:8]}"
            f"{' ...' if len(unresolved) > 8 else ''}"
        )


def load_w8a16_model(
    config,
    ckpt_dir: str,
    *,
    dtype: torch.dtype = torch.float16,
    cache_dir: str | None = None,
    revision: str | None = None,
    subfolder: str = "",
    local_files_only: bool = False,
    token: str | bool | None = None,
    force_download: bool = False,
    proxies: dict | None = None,
    trust_remote_code: bool | None = None,
) -> nn.Module:
    """Load a local W8A16 checkpoint without casting int8 weights to fp16."""
    ckpt_dir_path = _resolve_checkpoint_dir(
        ckpt_dir,
        cache_dir=cache_dir,
        revision=revision,
        subfolder=subfolder,
        local_files_only=local_files_only,
        token=token,
        force_download=force_download,
        proxies=proxies,
        trust_remote_code=trust_remote_code,
    )
    shards = _safetensor_shards(ckpt_dir_path)

    # Skeleton only. Construct on meta so 8B/14B W8A16 checkpoints do not first
    # allocate a full random fp32/fp16 model before the int8 weights are installed.
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    modules = dict(model.named_modules())
    loaded_names: set[str] = set()

    for shard in shards:
        if not shard.is_file():
            raise FileNotFoundError(f"W8A16 checkpoint shard missing: {shard}")
        for name, tensor in load_file(str(shard)).items():
            if name.endswith(".weight_scale"):
                mod = modules.get(name[: -len(".weight_scale")])
                if mod is None:
                    raise KeyError(f"W8A16 checkpoint has no module for scale {name!r}")
                mod.register_buffer("weight_scale", tensor.to(torch.float16))
                continue

            if name.endswith(QUANT_PROJ_SUFFIXES):
                if tensor.dtype != torch.int8:
                    raise TypeError(
                        f"W8A16 projection {name!r} must be int8, got {tensor.dtype}"
                    )
                mod = modules.get(name[: -len(".weight")])
                if not isinstance(mod, nn.Linear):
                    raise KeyError(f"W8A16 checkpoint has no Linear for {name!r}")
                _install_parameter(model, name, tensor)
                loaded_names.add(name)
                continue

            if name == LM_HEAD_WEIGHT_NAME:
                lm_head = getattr(model, "lm_head", None)
                if not isinstance(lm_head, nn.Linear):
                    raise KeyError(f"W8A16 checkpoint has no Linear for {name!r}")
                if tensor.dtype == torch.int8:
                    _install_parameter(model, name, tensor)
                elif tensor.is_floating_point():
                    _install_parameter(model, name, tensor.to(dtype))
                else:
                    raise TypeError(
                        f"W8A16 lm_head.weight must be floating or int8, "
                        f"got {tensor.dtype}"
                    )
                loaded_names.add(name)
                continue

            if name == EMBED_TOKENS_WEIGHT_NAME:
                embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
                if not isinstance(embed_tokens, nn.Embedding):
                    raise KeyError(f"W8A16 checkpoint has no Embedding for {name!r}")
                if tensor.dtype == torch.int8:
                    _install_parameter(model, name, tensor)
                elif tensor.is_floating_point():
                    _install_parameter(model, name, tensor.to(dtype))
                else:
                    raise TypeError(
                        f"W8A16 embed_tokens.weight must be floating or int8, "
                        f"got {tensor.dtype}"
                    )
                loaded_names.add(name)
                continue

            _install_parameter(model, name, tensor.to(dtype))
            loaded_names.add(name)

    missing = [
        name for name, _param in model.named_parameters()
        if name not in loaded_names
    ]
    if missing:
        raise KeyError(
            f"W8A16 checkpoint did not provide {len(missing)} parameter(s): "
            f"{sorted(missing)[:8]}{' ...' if len(missing) > 8 else ''}"
        )
    _materialize_meta_buffers(model)
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, nn.Linear) and lm_head.weight.dtype == torch.int8:
        scale = getattr(lm_head, "weight_scale", None)
        if scale is None or scale.dtype != torch.float16:
            raise KeyError(
                "W8A16 checkpoint has int8 lm_head.weight but no fp16 "
                "lm_head.weight_scale"
            )
        if scale.numel() != lm_head.weight.size(0):
            raise ValueError(
                f"lm_head.weight_scale numel={scale.numel()} must match "
                f"vocab_size={lm_head.weight.size(0)}"
            )
    embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
    if isinstance(embed_tokens, nn.Embedding) and embed_tokens.weight.dtype == torch.int8:
        scale = getattr(embed_tokens, "weight_scale", None)
        if scale is None or scale.dtype != torch.float16:
            raise KeyError(
                "W8A16 checkpoint has int8 embed_tokens.weight but no fp16 "
                "model.embed_tokens.weight_scale"
            )
        if scale.numel() != embed_tokens.weight.size(0):
            raise ValueError(
                f"model.embed_tokens.weight_scale numel={scale.numel()} must "
                f"match vocab_size={embed_tokens.weight.size(0)}"
            )
    return model


def _validate_imagetext_tensor_metadata(
    name: str,
    shape: tuple[int, ...],
    safe_dtype: str,
    *,
    model: nn.Module,
    modules: dict[str, nn.Module],
    expected_quant_weights: set[str],
    parameter_names: set[str],
) -> None:
    if name.endswith(".weight_scale"):
        weight_name = name[: -len(".weight_scale")] + ".weight"
        module = modules.get(name[: -len(".weight_scale")])
        if weight_name not in expected_quant_weights or not isinstance(module, nn.Linear):
            raise KeyError(f"W8A16 checkpoint has unexpected scale {name!r}")
        if safe_dtype != "F16":
            raise TypeError(f"W8A16 scale {name!r} must be fp16, got {safe_dtype}")
        if shape != (module.out_features,):
            raise ValueError(
                f"W8A16 scale {name!r} has shape {shape}, "
                f"expected {(module.out_features,)}"
            )
        return

    if name not in parameter_names:
        raise KeyError(f"W8A16 checkpoint tensor {name!r} has no matching model parameter")
    _module, _param_name, old_param = _parameter_owner(model, name)
    expected_shape = tuple(old_param.shape)
    if shape != expected_shape:
        raise ValueError(
            f"W8A16 checkpoint tensor {name!r} has shape {shape}, "
            f"expected {expected_shape}"
        )
    if name in expected_quant_weights:
        if safe_dtype != "I8":
            raise TypeError(f"W8A16 projection {name!r} must be int8, got {safe_dtype}")
    elif safe_dtype not in _SAFE_FLOAT_DTYPES:
        raise TypeError(
            f"W8A16 non-projection tensor {name!r} must be floating, got {safe_dtype}"
        )


def _build_w8a16_imagetext_plan(
    model: nn.Module,
    ckpt_dir: Path,
    shards: list[Path],
) -> _W8A16ImageTextLoadPlan:
    modules = dict(model.named_modules())
    parameter_names = {name for name, _parameter in model.named_parameters()}
    expected_quant_weights = {
        f"{name}.weight" for name, module in modules.items()
        if isinstance(module, nn.Linear)
        and f"{name}.weight".endswith(QUANT_PROJ_SUFFIXES)
    }
    expected_scales = {
        name[: -len(".weight")] + ".weight_scale"
        for name in expected_quant_weights
    }

    tensor_shards: dict[str, str] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle.keys():
                if name in tensor_shards:
                    raise KeyError(
                        f"W8A16 checkpoint tensor {name!r} appears more than once"
                    )
                tensor_slice = handle.get_slice(name)
                _validate_imagetext_tensor_metadata(
                    name,
                    tuple(int(dim) for dim in tensor_slice.get_shape()),
                    tensor_slice.get_dtype(),
                    model=model,
                    modules=modules,
                    expected_quant_weights=expected_quant_weights,
                    parameter_names=parameter_names,
                )
                tensor_shards[name] = str(shard)

    seen_names = set(tensor_shards)
    loaded_quant_weights = seen_names & expected_quant_weights
    if loaded_quant_weights != expected_quant_weights:
        missing = sorted(expected_quant_weights - loaded_quant_weights)
        extra = sorted(loaded_quant_weights - expected_quant_weights)
        raise KeyError(
            f"W8A16 projection inventory mismatch: missing={missing[:8]}, extra={extra[:8]}"
        )
    loaded_scales = seen_names & expected_scales
    if loaded_scales != expected_scales:
        missing = sorted(expected_scales - loaded_scales)
        extra = sorted(loaded_scales - expected_scales)
        raise KeyError(
            f"W8A16 scale inventory mismatch: missing={missing[:8]}, extra={extra[:8]}"
        )
    missing_params = sorted(parameter_names - seen_names)
    if missing_params:
        raise KeyError(
            f"W8A16 checkpoint did not provide {len(missing_params)} parameter(s): "
            f"{missing_params[:8]}{' ...' if len(missing_params) > 8 else ''}"
        )

    index_path = ckpt_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as handle:
            weight_map = json.load(handle).get("weight_map")
        if not isinstance(weight_map, dict):
            raise KeyError("W8A16 checkpoint index has no object-valued weight_map")
        if set(weight_map) != seen_names:
            missing = sorted(seen_names - set(weight_map))
            extra = sorted(set(weight_map) - seen_names)
            raise KeyError(
                f"W8A16 checkpoint index inventory mismatch: "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )
        for name, shard in tensor_shards.items():
            actual = Path(shard).relative_to(ckpt_dir).as_posix()
            declared = Path(weight_map[name]).as_posix()
            if actual != declared:
                raise KeyError(
                    f"W8A16 checkpoint index maps {name!r} to {declared!r}, "
                    f"but the tensor is in {actual!r}"
                )

    text_model = model.model.language_model
    layers = text_model.layers
    decoder_prefix = next(
        (name for name, module in model.named_modules() if module is layers),
        None,
    )
    if not decoder_prefix:
        raise KeyError("W8A16 checkpoint model has no named decoder layer container")
    layer_tensor_names = tuple(
        tuple(sorted(
            name for name in seen_names
            if name.startswith(f"{decoder_prefix}.{layer_index}.")
        ))
        for layer_index in range(len(layers))
    )
    empty_layers = [index for index, names in enumerate(layer_tensor_names) if not names]
    if empty_layers:
        raise KeyError(
            f"W8A16 checkpoint has no tensors for decoder layer(s) {empty_layers[:8]}"
        )
    decoder_names = {name for names in layer_tensor_names for name in names}
    nondecoder_names = tuple(sorted(seen_names - decoder_names))
    return _W8A16ImageTextLoadPlan(
        model_id=id(model),
        tensor_shards=tuple(sorted(tensor_shards.items())),
        layer_tensor_names=layer_tensor_names,
        nondecoder_tensor_names=nondecoder_names,
        parameter_names=tuple(sorted(parameter_names)),
    )


def _prepare_w8a16_imagetext(
    config,
    ckpt_dir: str,
    *,
    dtype: torch.dtype,
    **resolve_kwargs,
) -> tuple[nn.Module, _W8A16ImageTextLoadPlan]:
    if dtype is not torch.float16:
        raise TypeError(f"Qwen3-VL-32B W8A16 loader requires torch.float16, got {dtype}")
    if not is_qwen3_vl_32b_w8a16_config(config):
        raise ValueError(
            "W8A16 image-text loading is restricted to the exact "
            "Qwen3-VL-32B-Instruct controlled profile"
        )

    ckpt_dir_path = _resolve_checkpoint_dir(ckpt_dir, **resolve_kwargs)
    shards = _safetensor_shards(ckpt_dir_path)
    missing_shards = [str(shard) for shard in shards if not shard.is_file()]
    if missing_shards:
        raise FileNotFoundError(
            f"W8A16 checkpoint shard(s) missing: {missing_shards[:8]}"
        )

    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)
    plan = _build_w8a16_imagetext_plan(model, ckpt_dir_path, shards)
    return model, plan


def _expected_imagetext_quant_weights(model: nn.Module) -> set[str]:
    return {
        f"{name}.weight" for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and f"{name}.weight".endswith(QUANT_PROJ_SUFFIXES)
    }


def _install_imagetext_tensor(
    model: nn.Module,
    modules: dict[str, nn.Module],
    expected_quant_weights: set[str],
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> None:
    if name.endswith(".weight_scale"):
        weight_name = name[: -len(".weight_scale")] + ".weight"
        module = modules.get(name[: -len(".weight_scale")])
        if weight_name not in expected_quant_weights or not isinstance(module, nn.Linear):
            raise KeyError(f"W8A16 checkpoint has unexpected scale {name!r}")
        if tensor.dtype != torch.float16:
            raise TypeError(f"W8A16 scale {name!r} must be fp16, got {tensor.dtype}")
        if tuple(tensor.shape) != (module.out_features,):
            raise ValueError(
                f"W8A16 scale {name!r} has shape {tuple(tensor.shape)}, "
                f"expected {(module.out_features,)}"
            )
        module.register_buffer("weight_scale", tensor.clone().contiguous())
        return

    if name in expected_quant_weights:
        if tensor.dtype != torch.int8:
            raise TypeError(f"W8A16 projection {name!r} must be int8, got {tensor.dtype}")
        _install_parameter(model, name, tensor.clone())
        return

    if not tensor.is_floating_point():
        raise TypeError(
            f"W8A16 non-projection tensor {name!r} must be floating, got {tensor.dtype}"
        )
    _install_parameter(model, name, tensor.to(dtype=dtype, copy=True).contiguous())


def _load_imagetext_plan_names(
    model: nn.Module,
    plan: _W8A16ImageTextLoadPlan,
    names: tuple[str, ...],
    *,
    dtype: torch.dtype,
) -> None:
    modules = dict(model.named_modules())
    expected_quant_weights = _expected_imagetext_quant_weights(model)
    name_to_shard = dict(plan.tensor_shards)
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(name_to_shard[name], []).append(name)
    for shard, shard_names in by_shard.items():
        with safe_open(shard, framework="pt") as handle:
            for name in shard_names:
                tensor = handle.get_tensor(name)
                _install_imagetext_tensor(
                    model, modules, expected_quant_weights, name, tensor, dtype=dtype
                )
                del tensor


def _materialize_imagetext_meta_buffers(model: nn.Module) -> None:
    # Computed non-persistent buffers are intentionally absent from safetensors.
    # Materialize only the two exact Qwen3-VL rotary forms.
    for mod_name, mod in model.named_modules():
        for buf_name, buf in list(mod._buffers.items()):
            if buf is None or not getattr(buf, "is_meta", False):
                continue
            if buf_name == "inv_freq" and hasattr(mod, "dim") and hasattr(mod, "theta"):
                dim = int(mod.dim)
                theta = float(mod.theta)
                inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
                mod.register_buffer("inv_freq", inv.contiguous(), persistent=False)
                continue
            replacement = _materialize_qwen3_rotary_buffer(mod, buf_name)
            if replacement is not None:
                mod.register_buffer(buf_name, replacement, persistent=False)
                continue
            raise KeyError(
                f"W8A16 VLM checkpoint left meta buffer {mod_name}.{buf_name!r} "
                "without materialization support"
            )


def _raise_on_imagetext_meta(model: nn.Module) -> None:
    meta = [name for name, param in model.named_parameters() if param.is_meta]
    meta += [name for name, buf in model.named_buffers() if buf.is_meta]
    if meta:
        raise KeyError(
            f"W8A16 VLM checkpoint left {len(meta)} tensor(s) on meta: "
            f"{sorted(meta)[:8]}{' ...' if len(meta) > 8 else ''}"
        )


def _validate_staged_w8a16_imagetext(model: nn.Module) -> _W8A16ImageTextLoadPlan:
    plan = vars(model).get(_W8A16_IMAGETEXT_STAGE_ATTR)
    if not isinstance(plan, _W8A16ImageTextLoadPlan) or plan.model_id != id(model):
        raise RuntimeError(
            "Qwen3-VL-32B W8A16 staged load plan is missing or belongs to another model"
        )
    parameter_names = tuple(sorted(name for name, _param in model.named_parameters()))
    if parameter_names != plan.parameter_names:
        raise RuntimeError(
            "Qwen3-VL-32B W8A16 staged model hierarchy changed after "
            "metadata validation"
        )
    non_meta = [name for name, param in model.named_parameters() if not param.is_meta]
    if non_meta:
        raise RuntimeError(
            "Qwen3-VL-32B W8A16 staged model was materialized before RPU ownership: "
            f"{non_meta[:8]}"
        )
    non_host_buffers = [
        name for name, buf in model.named_buffers()
        if buf.device.type not in {"cpu", "meta"}
    ]
    if non_host_buffers:
        raise RuntimeError(
            "Qwen3-VL-32B W8A16 staged model owns non-host buffers before RPU ownership: "
            f"{non_host_buffers[:8]}"
        )
    if len(model.model.language_model.layers) != len(plan.layer_tensor_names):
        raise RuntimeError("Qwen3-VL-32B W8A16 staged decoder layer count changed")
    checkpoint_names = {name for name, _shard in plan.tensor_shards}
    expected_quant_weights = _expected_imagetext_quant_weights(model)
    if not expected_quant_weights.issubset(checkpoint_names):
        raise RuntimeError(
            "Qwen3-VL-32B W8A16 staged quantized module inventory changed"
        )
    return plan


def stage_w8a16_imagetext_for_rpu(
    config,
    ckpt_dir: str,
    *,
    dtype: torch.dtype = torch.float16,
    **resolve_kwargs,
) -> nn.Module:
    """Validate checkpoint metadata and return an unmaterialized exact 32B model.

    Tensor loading is deliberately deferred until the Qwen3-VL adapter owns the
    process-wide RPU slot and has published its irreversible-install poison.
    """
    model, plan = _prepare_w8a16_imagetext(
        config, ckpt_dir, dtype=dtype, **resolve_kwargs
    )
    vars(model)[_W8A16_IMAGETEXT_STAGE_ATTR] = plan
    model.eval()
    return model


def _swizzle_and_move_w8a16_decoder_layer(layer: nn.Module) -> None:
    from rpu_backend.runtime.weights import convert_linear_weights_inplace

    convert_linear_weights_inplace(layer, skip_names=set())
    layer.to("rpu")


def _materialize_staged_w8a16_imagetext_for_rpu(model: nn.Module) -> None:
    """Consume a metadata-only plan after the adapter owns and poisons RPU state."""
    plan = _validate_staged_w8a16_imagetext(model)
    if vars(model).get("_rpu_swizzle_started") is not True:
        raise RuntimeError(
            "Qwen3-VL-32B W8A16 streaming requires adapter ownership/poison first"
        )

    _load_imagetext_plan_names(
        model, plan, plan.nondecoder_tensor_names, dtype=torch.float16
    )
    layers = model.model.language_model.layers
    for layer, names in zip(layers, plan.layer_tensor_names, strict=True):
        _load_imagetext_plan_names(model, plan, names, dtype=torch.float16)
        layer_meta = [name for name, param in layer.named_parameters() if param.is_meta]
        if layer_meta:
            raise KeyError(
                f"W8A16 decoder layer remains incomplete before swizzle: {layer_meta[:8]}"
            )
        _swizzle_and_move_w8a16_decoder_layer(layer)
        gc.collect()

    _materialize_imagetext_meta_buffers(model)
    _raise_on_imagetext_meta(model)
    vars(model).pop(_W8A16_IMAGETEXT_STAGE_ATTR, None)
    model.eval()


def load_w8a16_imagetext(
    config,
    ckpt_dir: str,
    *,
    dtype: torch.dtype = torch.float16,
    **resolve_kwargs,
) -> nn.Module:
    """Strictly load the controlled Qwen3-VL-32B W8A16 checkpoint on CPU."""
    model, plan = _prepare_w8a16_imagetext(
        config, ckpt_dir, dtype=dtype, **resolve_kwargs
    )
    all_names = tuple(name for name, _shard in plan.tensor_shards)
    _load_imagetext_plan_names(model, plan, all_names, dtype=dtype)
    gc.collect()

    _materialize_imagetext_meta_buffers(model)
    _raise_on_imagetext_meta(model)
    non_cpu = [
        name for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.device.type != "cpu"
    ]
    if non_cpu:
        raise RuntimeError(
            f"W8A16 VLM CPU loader materialized non-CPU tensor(s): {non_cpu[:8]}"
        )
    model.eval()
    return model
