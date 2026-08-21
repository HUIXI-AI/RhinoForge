"""RPUModelForConditionalGeneration: HF-compatible entry for vision-language models.

Mirrors `RPUModelForCausalLM.from_pretrained` but routes to multimodal HF
classes through `AutoModelForImageTextToText`. It shares
the same single-handle gate (`_LIVE_REF` in causal_lm.py) so a CausalLM and a
ConditionalGeneration model cannot be live on RPU simultaneously.

Supported: `Qwen3VLForConditionalGeneration` (registered by
`rpu_backend.adapters.qwen3_vl`). Other built-in conditional-generation
architectures use their documented model-specific adapters.
"""
from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForImageTextToText

from rpu_backend.api.errors import (
    RPUUnsupportedDtypeError,
    UnsupportedModelError,
)
from rpu_backend.api._execution import normalize_rpu_execution
from rpu_backend.api._loading import resolve_config_architecture, resolve_loader_device
from rpu_backend.runtime.registry import get_adapter


class RPUModelForConditionalGeneration:
    """Stateless thin wrapper for vision-language models.

    Usage::

        from rpu_backend.api import RPUModelForConditionalGeneration
        model = RPUModelForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", dtype=torch.float16
        ).to("rpu")
        out = model(input_ids=..., pixel_values=..., image_grid_thw=...,
                    mm_token_type_ids=..., past_key_values=rpu_cache)
        logits = out.logits

    Notes:
      - Like `RPUModelForCausalLM`, this is NOT a subclass of any HF class.
      - `from_pretrained(..., device='rpu')` is the one-step shortcut.
      - Video frames, batch > 1, and 30B-A3B MoE are rejected at preflight or
        adapter __init__.
    """

    _RESERVED_HF_KWARGS = frozenset({
        "torch_dtype", "dtype",
        "device_map", "device",
        "low_cpu_mem_usage",
        "tp_plan",
        "load_in_8bit", "load_in_4bit",
        "quantization_config",
    })
    _SUPPORTED_BUILTIN_ARCHITECTURES = frozenset({
        "Qwen3VLForConditionalGeneration",
    })

    @classmethod
    def from_pretrained(
        cls,
        hf_repo_or_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device | None = None,
        rpu_execution=None,
        **hf_kwargs,
    ):
        execution_config = normalize_rpu_execution(
            rpu_execution,
            entry_point="RPUModelForConditionalGeneration.from_pretrained",
            supported={
                "prefill": ("chunk_size", "padding_rows", "padding_budget"),
                "vision": ("chunk_size",),
            },
        )
        if dtype is not torch.float16:
            raise RPUUnsupportedDtypeError(
                f"RPU kernels require torch.float16, got {dtype}. "
                "Load the CPU reference separately for fp32 comparison."
            )

        collisions = cls._RESERVED_HF_KWARGS & set(hf_kwargs)
        if collisions:
            raise ValueError(
                f"RPUModelForConditionalGeneration.from_pretrained: hf_kwargs "
                f"cannot include reserved key(s) {sorted(collisions)}; the "
                "library manages these internally."
            )
        trust_remote_code = hf_kwargs.pop("trust_remote_code", False)
        if trust_remote_code is not False:
            raise ValueError(
                "RPUModelForConditionalGeneration.from_pretrained requires "
                "trust_remote_code=False; custom model code is not supported."
            )

        move_to_rpu = resolve_loader_device(
            device,
            entry_point="RPUModelForConditionalGeneration.from_pretrained",
        )

        _CONFIG_KWARGS = {
            "cache_dir", "revision", "subfolder", "local_files_only",
            "token", "force_download", "proxies",
        }
        cfg_kwargs = {k: v for k, v in hf_kwargs.items() if k in _CONFIG_KWARGS}
        try:
            config = AutoConfig.from_pretrained(
                hf_repo_or_path, trust_remote_code=False, **cfg_kwargs
            )
        except (OSError, ValueError) as e:
            raise UnsupportedModelError(
                f"Could not load HF config for {hf_repo_or_path!r}: "
                f"{type(e).__name__}: {e}."
            ) from e

        arch = resolve_config_architecture(
            config,
            source=hf_repo_or_path,
            entry_point="RPUModelForConditionalGeneration.from_pretrained",
            supported_builtins=cls._SUPPORTED_BUILTIN_ARCHITECTURES,
        )
        adapter_cls = get_adapter(arch)
        if hasattr(adapter_cls, "preflight"):
            adapter_cls.preflight(config)
        if hasattr(adapter_cls, "preflight_execution"):
            adapter_cls.preflight_execution(config, execution_config)

        from rpu_backend.quant.load import (
            is_qwen3_vl_32b_w8a16_config,
            is_w8a16_config,
            load_w8a16_imagetext,
            stage_w8a16_imagetext_for_rpu,
        )
        if is_w8a16_config(config):
            if not is_qwen3_vl_32b_w8a16_config(config):
                raise UnsupportedModelError(
                    "RPUModelForConditionalGeneration: W8A16 image-text loading "
                    "is restricted to the exact Qwen3-VL-32B-Instruct controlled "
                    "profile; 2B/4B W8A16 are not enabled and 8B remains unsupported."
                )
            unsupported = sorted(set(hf_kwargs) - _CONFIG_KWARGS)
            if unsupported:
                raise ValueError(
                    "RPUModelForConditionalGeneration: the exact W8A16 loader "
                    f"does not support hf_kwargs {unsupported}; it accepts only "
                    "config/checkpoint location kwargs and always returns a model."
                )
            loader = (
                stage_w8a16_imagetext_for_rpu
                if move_to_rpu else load_w8a16_imagetext
            )
            model = loader(config, hf_repo_or_path, dtype=dtype, **cfg_kwargs)
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                hf_repo_or_path,
                torch_dtype=dtype,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=False,
                **hf_kwargs,
            )
        model.eval()
        model._rpu_execution = execution_config

        adapter = adapter_cls(model)
        adapter._rpu_execution = execution_config

        if move_to_rpu:
            return adapter.to_rpu()
        return model
