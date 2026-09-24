"""RPUModelForConditionalGeneration: HF-compatible entry for vision-language models.

R-Phase 4. Mirrors `RPUModelForCausalLM.from_pretrained` (api/causal_lm.py)
but routes to multimodal HF classes via `AutoModelForImageTextToText`. Shares
the same single-handle gate (`_LIVE_REF` in causal_lm.py) so a CausalLM and a
ConditionalGeneration model cannot be live on RPU simultaneously.

Supported: `Qwen3VLForConditionalGeneration` and
`Qwen3_5ForConditionalGeneration`, routed through their registered adapters.
"""
from __future__ import annotations

from collections.abc import Mapping

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
    """Stateless entry point for vision-language models (D-02 thin wrapper).

    Usage::

        from rpu_backend.api import RPUModelForConditionalGeneration, greedy_token_ids
        model = RPUModelForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", dtype=torch.float16
        ).to("rpu")
        out = model(input_ids=..., pixel_values=..., image_grid_thw=...,
                    mm_token_type_ids=..., past_key_values=rpu_cache)
        logits = out.logits
        next_ids = greedy_token_ids(logits)  # CPU int64 [batch, 1]

    Notes:
      - Like `RPUModelForCausalLM`, this is NOT a subclass of any HF class.
      - `from_pretrained(..., device='rpu')` is the one-step shortcut (D-01).
      - Video inputs have a controlled implementation path but are outside the
        certified Qwen3-VL 8B envelope. Batch > 1 and 30B-A3B MoE are rejected.
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
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
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
        execution_config = None
        if not (
            isinstance(rpu_execution, Mapping)
            and "components" in rpu_execution
        ):
            execution_config = normalize_rpu_execution(
                rpu_execution,
                entry_point="RPUModelForConditionalGeneration.from_pretrained",
                supported={
                    "model": ("num_cores",),
                    "prefill": (
                        "chunk_size", "padding_rows", "padding_budget", "linear_acc32",
                        "fast_replay",
                    ),
                    # Structure/types only; after resolving the actual model
                    # adapter, tighten this to its declared fields below.
                    "vision": (
                        "chunk_size", "padding_rows", "padding_budget", "linear_acc32",
                        "gelu_erf_ultra",
                        "fast_replay",
                    ),
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

        move_to_rpu = resolve_loader_device(
            device,
            entry_point="RPUModelForConditionalGeneration.from_pretrained",
        )

        if hf_kwargs.get("trust_remote_code", False) is not False:
            raise ValueError("RhinoForge requires trust_remote_code=False")
        _CONFIG_KWARGS = {
            "cache_dir", "revision", "subfolder", "local_files_only",
            "trust_remote_code", "token", "force_download", "proxies",
        }
        cfg_kwargs = {k: v for k, v in hf_kwargs.items() if k in _CONFIG_KWARGS}
        try:
            config = AutoConfig.from_pretrained(hf_repo_or_path, **cfg_kwargs)
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
        execution_supported = getattr(adapter_cls, "_EXECUTION_SUPPORTED", None)
        component_capabilities = getattr(
            adapter_cls, "_rpu_execution_components", None
        )
        if component_capabilities is not None:
            execution_config = normalize_rpu_execution(
                rpu_execution,
                entry_point=(
                    "RPUModelForConditionalGeneration.from_pretrained"
                ),
                supported_components=component_capabilities,
                supported=execution_supported,
            )
        else:
            execution_config = normalize_rpu_execution(
                rpu_execution,
                entry_point=(
                    "RPUModelForConditionalGeneration.from_pretrained"
                ),
                supported=execution_supported if execution_supported is not None else {
                    "prefill": (
                        "chunk_size", "padding_rows", "padding_budget"
                    ),
                    "vision": ("chunk_size",),
                },
            )
        if hasattr(adapter_cls, "preflight"):
            adapter_cls.preflight(config)
        if hasattr(adapter_cls, "preflight_execution"):
            adapter_cls.preflight_execution(config, execution_config)

        # Model-specific checkpoint formats get first refusal. In particular,
        # the Qwen3.5 loader preserves per-projection int8 tensors and scales;
        # the generic HF loader would silently cast them into fp16 Parameters.
        adapter_loader = getattr(adapter_cls, "load_pretrained", None)
        if adapter_loader is not None:
            model = adapter_loader(
                hf_repo_or_path, config=config, dtype=dtype, **hf_kwargs)
        else:
            from rpu_backend.quant.load import (
                is_qwen3_vl_4b_w8a16_config,
                is_qwen3_vl_32b_w8a16_config,
                is_qwen3_vl_w8a16_config,
                is_qwen3_vl_awq_config,
                is_w8a16_config,
                load_qwen3_vl_awq_imagetext,
                load_w8a16_imagetext,
                stage_w8a16_imagetext_for_rpu,
            )
            from rpu_backend.quant.load_qwen3_vl_runtime import (
                is_qwen3_vl_runtime_quant_config,
                load_qwen3_vl_runtime_imagetext,
            )
            if is_qwen3_vl_runtime_quant_config(config):
                unsupported = sorted(set(hf_kwargs) - _CONFIG_KWARGS)
                if unsupported:
                    raise ValueError(
                        "Qwen3-VL runtime quantized loader accepts only "
                        f"config/checkpoint location kwargs; unsupported {unsupported}."
                    )
                runtime_load_kwargs = dict(cfg_kwargs)
                if (move_to_rpu and config.text_config.hidden_size == 5120
                        and config.quant_config["method"] == "w8a16"):
                    runtime_load_kwargs["_mmap_w8_weights"] = True
                model = load_qwen3_vl_runtime_imagetext(
                    config, hf_repo_or_path, dtype=dtype, **runtime_load_kwargs)
            elif is_qwen3_vl_awq_config(config):
                unsupported = sorted(set(hf_kwargs) - _CONFIG_KWARGS)
                if unsupported:
                    raise ValueError(
                        "RPUModelForConditionalGeneration: the exact Qwen3-VL "
                        "AWQ loader accepts only config/checkpoint location "
                        f"kwargs; unsupported {unsupported}."
                    )
                model = load_qwen3_vl_awq_imagetext(
                    config, hf_repo_or_path, dtype=dtype, **cfg_kwargs)
            elif is_w8a16_config(config):
                if not is_qwen3_vl_w8a16_config(config):
                    raise UnsupportedModelError(
                        "RPUModelForConditionalGeneration: W8A16 image-text "
                        "loading requires the exact Qwen3-VL-2B-Instruct "
                        "text-only profile, exact Qwen3-VL-4B runtime-alignment "
                        "or Qwen3-VL-32B-Instruct controlled profile; 8B W8A16 is not enabled."
                    )
                unsupported = sorted(set(hf_kwargs) - _CONFIG_KWARGS)
                if unsupported:
                    raise ValueError(
                        "RPUModelForConditionalGeneration: the exact W8A16 "
                        "loader does not support hf_kwargs "
                        f"{unsupported}; it accepts only config/checkpoint "
                        "location kwargs and always returns a model."
                    )
                loader = (
                    stage_w8a16_imagetext_for_rpu
                    if move_to_rpu and (is_qwen3_vl_32b_w8a16_config(config)
                                       or is_qwen3_vl_4b_w8a16_config(config))
                    else load_w8a16_imagetext
                )
                model = loader(
                    config, hf_repo_or_path, dtype=dtype, **cfg_kwargs)
            else:
                model = AutoModelForImageTextToText.from_pretrained(
                    hf_repo_or_path,
                    torch_dtype=dtype,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    **hf_kwargs,
                )
        model.eval()
        model._rpu_execution = execution_config

        adapter = adapter_cls(model)
        adapter._rpu_execution = execution_config

        if move_to_rpu:
            return adapter.to_rpu()
        return model
