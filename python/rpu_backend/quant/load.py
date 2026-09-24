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
from rpu_backend.quant.convert_qwen3_vl import (
    QUANT_CONFIG as _QWEN3_VL_4B_W8A16_QUANT_CONFIG,
    matches_4b_geometry,
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

_QWEN3_VL_2B_W8A16_QUANT_CONFIG = {
    **_QWEN3_VL_32B_W8A16_QUANT_CONFIG,
    "skip_modules": ["model.visual", "lm_head"],
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


def is_qwen3_vl_4b_w8a16_config(config) -> bool:
    """Exact controlled 4B profile: offline W8 text/head, install-time W8 Vision."""
    text = getattr(config, "text_config", None)
    vision = getattr(config, "vision_config", None)
    return (
        matches_4b_geometry(config)
        and getattr(config, "tie_word_embeddings", None) is False
        and getattr(text, "tie_word_embeddings", None) is False
        and getattr(config, "quant_config", None) == _QWEN3_VL_4B_W8A16_QUANT_CONFIG
        and getattr(config, "quantization_config", None) is None
        and all(getattr(owner, key, None) is None for owner in (text, vision)
                for key in ("quant_config", "quantization_config"))
    )


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


def is_qwen3_vl_2b_w8a16_config(config) -> bool:
    """Exact 2B text-only per-channel INT8; Vision/embed/head remain FP16.

    This is the ``w8a16-text-v1`` checkpoint format, not the separate hybrid
    delivery profile or a generic admission of quantized Qwen3-VL models.
    """
    from rpu_backend.api._execution import qwen3_vl_text_core_profile

    try:
        text, vision = config.text_config, config.vision_config
        profile = qwen3_vl_text_core_profile(text, 4)
        return (
            tuple(config.architectures) == ("Qwen3VLForConditionalGeneration",)
            and config.model_type == "qwen3_vl"
            and config.tie_word_embeddings is True
            and text.tie_word_embeddings is True
            and config.quant_config == _QWEN3_VL_2B_W8A16_QUANT_CONFIG
            and getattr(config, "quantization_config", None) is None
            and all(getattr(vision, key, None) is None
                    for key in ("quant_config", "quantization_config"))
            and (profile.hidden_size, profile.intermediate_size,
                 profile.num_layers, profile.num_q_heads) == (2048, 6144, 28, 16)
            and (vision.model_type, vision.hidden_size, vision.intermediate_size,
                 vision.depth, vision.num_heads, vision.patch_size,
                 vision.temporal_patch_size, vision.spatial_merge_size,
                 tuple(vision.deepstack_visual_indexes), vision.out_hidden_size,
                 vision.hidden_act, vision.in_channels, vision.num_position_embeddings)
                == ("qwen3_vl", 1024, 4096, 24, 16, 16, 2, 2,
                    (5, 11, 17), 2048, "gelu_pytorch_tanh", 3, 2304)
            and tuple(getattr(config, key) for key in (
                "image_token_id", "video_token_id", "vision_start_token_id",
                "vision_end_token_id")) == (151655, 151656, 151652, 151653)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def is_qwen3_vl_w8a16_config(config) -> bool:
    return (is_qwen3_vl_2b_w8a16_config(config)
            or is_qwen3_vl_4b_w8a16_config(config)
            or is_qwen3_vl_32b_w8a16_config(config))


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
    expected_quant_weights = _expected_imagetext_quant_weights(model)
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
        raise TypeError(f"Qwen3-VL W8A16 loader requires torch.float16, got {dtype}")
    if not is_qwen3_vl_w8a16_config(config):
        raise ValueError(
            "W8A16 image-text loading is restricted to the exact "
            "Qwen3-VL-2B-Instruct text-only, 4B runtime-alignment or 32B controlled profile"
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
    expected = {
        f"{name}.weight" for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and f"{name}.weight".endswith(QUANT_PROJ_SUFFIXES)
    }
    if is_qwen3_vl_4b_w8a16_config(getattr(model, "config", None)):
        expected.add("lm_head.weight")
    return expected


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
        if is_qwen3_vl_4b_w8a16_config(getattr(model, "config", None)) and (
                not bool(torch.isfinite(tensor).all()) or not bool((tensor > 0).all())):
            raise ValueError(f"W8A16 runtime-alignment scale {name!r} must be finite and positive")
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
            "Qwen3-VL W8A16 staged load plan is missing or belongs to another model"
        )
    parameter_names = tuple(sorted(name for name, _param in model.named_parameters()))
    if parameter_names != plan.parameter_names:
        raise RuntimeError(
            "Qwen3-VL W8A16 staged model hierarchy changed after "
            "metadata validation"
        )
    non_meta = [name for name, param in model.named_parameters() if not param.is_meta]
    if non_meta:
        raise RuntimeError(
            "Qwen3-VL W8A16 staged model was materialized before RPU ownership: "
            f"{non_meta[:8]}"
        )
    non_host_buffers = [
        name for name, buf in model.named_buffers()
        if buf.device.type not in {"cpu", "meta"}
    ]
    if non_host_buffers:
        raise RuntimeError(
            "Qwen3-VL W8A16 staged model owns non-host buffers before RPU ownership: "
            f"{non_host_buffers[:8]}"
        )
    if len(model.model.language_model.layers) != len(plan.layer_tensor_names):
        raise RuntimeError("Qwen3-VL W8A16 staged decoder layer count changed")
    checkpoint_names = {name for name, _shard in plan.tensor_shards}
    expected_quant_weights = _expected_imagetext_quant_weights(model)
    if not expected_quant_weights.issubset(checkpoint_names):
        raise RuntimeError(
            "Qwen3-VL W8A16 staged quantized module inventory changed"
        )
    return plan


def stage_w8a16_imagetext_for_rpu(
    config,
    ckpt_dir: str,
    *,
    dtype: torch.dtype = torch.float16,
    **resolve_kwargs,
) -> nn.Module:
    """Validate metadata and return an unmaterialized controlled 4B/32B model.

    Tensor loading is deliberately deferred until the Qwen3-VL adapter owns the
    process-wide RPU slot and has published its irreversible-install poison.
    """
    if not (is_qwen3_vl_32b_w8a16_config(config) or is_qwen3_vl_4b_w8a16_config(config)):
        raise ValueError("W8A16 metadata-only staging requires exact Qwen3-VL-4B/32B; use the CPU loader for 2B")
    model, plan = _prepare_w8a16_imagetext(
        config, ckpt_dir, dtype=dtype, **resolve_kwargs
    )
    vars(model)[_W8A16_IMAGETEXT_STAGE_ATTR] = plan
    model.eval()
    return model


def _allocate_decoder_projection_views(
    model: nn.Module, layer_tensor_names: tuple[tuple[str, ...], ...],
    *, dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """One model-owned storage, with byte-aligned independent projection views.

    Exact 2B FP16/text-W8 and controlled 4B W8 callers use this layout; 4B
    FP16 requests one layer at a time to keep each allocation below 4 GiB. Q/K/V
    bytes are adjacent in that order so decode can form a read-only packed
    view without copying weights or allocating another DDR mapping. Parameters retain
    the storage after this temporary view dictionary dies; no global arena or
    allocator policy is involved. Build offsets from validated tensor geometry
    before any projection is swizzled.
    """
    if dtype not in (torch.uint8, torch.int8, torch.float16):
        raise ValueError("Decoder projection storage requires packed INT4, INT8 or FP16")
    element_size = 2 if dtype == torch.float16 else 1
    layout = []
    size = 0
    for names in layer_tensor_names:
        projections = [name for name in names if name.endswith(QUANT_PROJ_SUFFIXES)]
        qkv_order = (".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                     ".self_attn.v_proj.weight")
        # Stable sorting keeps every other projection's existing relative order.
        projections.sort(key=lambda name: next(
            (index for index, suffix in enumerate(qkv_order) if name.endswith(suffix)), 3))
        for name in projections:
            weight = model.get_parameter(name)
            offset = (size + 255) // 256 * 256
            layout.append((name, tuple(weight.shape), offset // element_size, weight.numel()))
            size = offset + weight.numel() * element_size
    if not layout:
        raise ValueError("Decoder projection storage requires nonempty weights")
    storage = torch.empty(size // element_size, dtype=dtype, device="rpu")
    return {name: storage.narrow(0, offset, count).view(shape)
            for name, shape, offset, count in layout}


def _allocate_w8a16_decoder_projection_views(
    model: nn.Module, layer_tensor_names: tuple[tuple[str, ...], ...],
) -> dict[str, torch.Tensor]:
    return _allocate_decoder_projection_views(model, layer_tensor_names, dtype=torch.int8)


def _swizzle_and_move_decoder_layer(
    layer: nn.Module, *, dtype: torch.dtype,
    projection_views: dict[str, torch.Tensor] | None = None,
) -> None:
    from rpu_backend.runtime.weights import convert_linear_weights_inplace

    convert_linear_weights_inplace(layer, skip_names=set())
    if projection_views is not None:
        for name, target in projection_views.items():
            module = layer.get_submodule(name)
            source = module.weight.detach()
            if (source.dtype != dtype or source.device.type != "cpu"
                    or source.shape != target.shape or not source.is_contiguous()
                    or target.dtype != dtype or not target.is_contiguous()):
                raise ValueError(f"Invalid swizzled projection storage: {name}")
            # copy_ publishes only this view's bytes through the existing sized
            # CPU->RPU flush. Do not create a per-projection RPU allocation.
            target.copy_(source)
            module.weight = nn.Parameter(target, requires_grad=False)
    # Packed weights already reside on RPU; only FP16 norms/scales still move.
    layer.to("rpu")


def _swizzle_and_move_w8a16_decoder_layer(
    layer: nn.Module, *, projection_views: dict[str, torch.Tensor] | None = None,
) -> None:
    _swizzle_and_move_decoder_layer(layer, dtype=torch.int8, projection_views=projection_views)


def _move_materialized_w8a16_decoder_for_rpu(model: nn.Module) -> None:
    """Give the public CPU-load then .to('rpu') path the same 4B storage."""
    if not is_qwen3_vl_4b_w8a16_config(getattr(model, "config", None)):
        raise ValueError("Shared W8 decoder storage requires the exact controlled 4B profile")
    _move_materialized_decoder_for_rpu(model, dtype=torch.int8)


def _move_materialized_decoder_for_rpu(
    model: nn.Module, *, dtype: torch.dtype, per_layer: bool = False,
    decoder: nn.Module | None = None, decoder_prefix: str = "model.language_model",
) -> None:
    """Pack decoder weights after the adapter admits the exact eight-core profile."""
    from rpu_backend.runtime.weights import _release_cpu_weight_pages

    if dtype not in (torch.int8, torch.float16):
        raise ValueError("Decoder migration requires INT8 or FP16")
    if vars(model).get("_rpu_swizzle_started") is not True:
        raise RuntimeError("Decoder migration requires adapter ownership/poison first")
    decoder = model.get_submodule(decoder_prefix) if decoder is None else decoder
    if model.get_submodule(decoder_prefix) is not decoder:
        raise ValueError("Decoder bank prefix must identify the supplied model-owned decoder")
    layers = decoder.layers
    layer_tensor_names = tuple(
        tuple(f"{decoder_prefix}.layers.{index}.{name}"
              for name, _ in sorted(layer.named_parameters()))
        for index, layer in enumerate(layers)
    )
    # Check the entire inventory before allocating or mutating the first layer.
    # The adapter already validates exact shapes and the matching scale tensors.
    for names in layer_tensor_names:
        for name in names:
            if not name.endswith(QUANT_PROJ_SUFFIXES):
                continue
            module = model.get_submodule(name[:-len(".weight")])
            weight = module.weight
            if (weight.device.type != "cpu" or weight.dtype != dtype
                    or not weight.is_contiguous()
                    or getattr(module, "_rpu_linear_partition", None) is not None):
                label = "INT8" if dtype == torch.int8 else "FP16"
                raise ValueError(f"Decoder migration requires unswizzled CPU {label}: {name}")
            # Validation must not pin the final CPU Parameter during migration.
            del weight
    views = (None if per_layer else
             _allocate_decoder_projection_views(model, layer_tensor_names, dtype=dtype))
    for layer, names in zip(layers, layer_tensor_names, strict=True):
        if per_layer:
            views = _allocate_decoder_projection_views(model, (names,), dtype=dtype)
        relative_views = {
            name.rsplit(".layers.", 1)[1].split(".", 1)[1][:-len(".weight")]: views[name]
            for name in names if name in views
        }
        _swizzle_and_move_decoder_layer(layer, dtype=dtype, projection_views=relative_views)
        # HostDDR shares physical RAM with CPU weights. GC alone can leave
        # freed CPU arenas resident while the next device bank is allocated.
        _release_cpu_weight_pages()


def _materialize_staged_w8a16_imagetext_for_rpu(model: nn.Module) -> None:
    """Consume a metadata-only plan after the adapter owns and poisons RPU state."""
    plan = _validate_staged_w8a16_imagetext(model)
    if vars(model).get("_rpu_swizzle_started") is not True:
        raise RuntimeError(
            "Qwen3-VL W8A16 streaming requires adapter ownership/poison first"
        )

    projection_views = (
        _allocate_w8a16_decoder_projection_views(model, plan.layer_tensor_names)
        if is_qwen3_vl_4b_w8a16_config(getattr(model, "config", None)) else None
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
        if projection_views is None:
            _swizzle_and_move_w8a16_decoder_layer(layer)
        else:
            relative_views = {
                name.rsplit(".layers.", 1)[1].split(".", 1)[1][:-len(".weight")]: projection_views[name]
                for name in names if name in projection_views
            }
            _swizzle_and_move_w8a16_decoder_layer(layer, projection_views=relative_views)
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
    """Strictly load a controlled Qwen3-VL W8A16 checkpoint on CPU."""
    model, plan = _prepare_w8a16_imagetext(
        config, ckpt_dir, dtype=dtype, **resolve_kwargs
    )
    all_names = tuple(name for name, _shard in plan.tensor_shards)
    _load_imagetext_plan_names(model, plan, all_names, dtype=dtype)
    if is_qwen3_vl_2b_w8a16_config(config):
        # Safetensors omits the tied head. Installing a new embedding Parameter
        # leaves the original alias on meta, so restore this exact HF tie before
        # validation. The adapter unties it once before col-swizzling the head.
        model.lm_head.weight = model.model.language_model.embed_tokens.weight
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


# Exact compressed-tensors Text-W4 loader; independent of the W8 staging path.
from rpu_backend.quant.load_qwen3_vl_awq import (  # noqa: E402,F401
    is_qwen3_vl_awq_config,
    load_qwen3_vl_awq_imagetext,
    _validate_qwen3_vl_awq_model,
    _move_materialized_awq_decoder_for_rpu,
)
