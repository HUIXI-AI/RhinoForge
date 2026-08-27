"""Shared validation for the public Hugging Face loader entry points."""
from __future__ import annotations

from collections.abc import Collection

import torch

from rpu_backend.api.errors import UnsupportedModelError


_BUILTIN_ROUTES = {
    "DINOv3ViTModel": "use DINOv3Adapter with the direct vision-model path",
    "Gemma4ForConditionalGeneration": "use Gemma4Adapter(model).to_rpu()",
    "Qwen3_5ForConditionalGeneration": (
        "use Qwen3_5Adapter(model).to_rpu() with Qwen3_5Cache"
    ),
    "Qwen3VLForConditionalGeneration": (
        "use RPUModelForConditionalGeneration.from_pretrained(...)"
    ),
    "LlamaForCausalLM": "use RPUModelForCausalLM.from_pretrained(...) and .to('rpu')",
    "Qwen3ForCausalLM": "use RPUModelForCausalLM.from_pretrained(...) and .to('rpu')",
}


def resolve_loader_device(device, *, entry_point: str) -> bool:
    """Validate a loader target and return whether the RPU shortcut is requested."""
    if device is None:
        return False
    if not isinstance(device, (str, torch.device)):
        raise ValueError(
            f"{entry_point}: device must be None, cpu, cpu:0, rpu, or rpu:0; "
            f"got {device!r}"
        )
    try:
        resolved = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{entry_point}: invalid device {device!r}") from exc
    if resolved.type not in {"cpu", "rpu"} or resolved.index not in {None, 0}:
        raise ValueError(
            f"{entry_point}: device must be None, cpu, cpu:0, rpu, or rpu:0; "
            f"got {device!r}"
        )
    return resolved.type == "rpu"


def resolve_config_architecture(
    config,
    *,
    source: str,
    entry_point: str,
    supported_builtins: Collection[str],
) -> str:
    """Return the primary HF architecture after validating loader ownership."""
    # api/ must not import adapters at module initialization. The manifest is
    # pure data and already registered by the package composition root; import
    # it only when an explicit model load asks for route validation.
    from rpu_backend.adapters._manifest import BUILTIN_ADAPTERS

    architectures = getattr(config, "architectures", None)
    if (
        not isinstance(architectures, (list, tuple))
        or not architectures
        or any(not isinstance(arch, str) or not arch.strip() for arch in architectures)
    ):
        raise UnsupportedModelError(
            f"HF config at {source!r} must define `architectures` as a non-empty "
            f"list of non-empty strings; got {architectures!r}. Cannot dispatch "
            f"through {entry_point}. See docs/model_support.md."
        )

    arch = architectures[0]
    builtin_architectures = frozenset(entry[0] for entry in BUILTIN_ADAPTERS)
    if arch in builtin_architectures and arch not in supported_builtins:
        route = _BUILTIN_ROUTES.get(arch, "use its model-specific adapter")
        raise UnsupportedModelError(
            f"{entry_point} does not support built-in HF architecture {arch!r}; "
            f"{route}. See docs/model_support.md."
        )
    return arch


__all__ = [
    "resolve_config_architecture",
    "resolve_loader_device",
]
