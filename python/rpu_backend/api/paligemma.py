"""Compatibility tombstone for the unsupported PaliGemma2 RPU policy.

The canonical user import is:

    from rpu_backend.api import PaliGemmaPolicy

The adapter scaffold remains private for numerical investigation, but the
public constructors fail before model loading or RPU mutation until the model
passes the repository's support gates again.
"""
from __future__ import annotations

from typing import Any, NoReturn

import torch


_UNSUPPORTED_MESSAGE = (
    "PaliGemma2 RPU execution is unsupported: the 3B pt-224 and mix-224 "
    "profiles have not passed the required prefill/decode/logits numerical "
    "gates. PaliGemmaPolicy remains importable for API compatibility, but "
    "construction is disabled until the model is fixed and recertified."
)


def _raise_unsupported() -> NoReturn:
    from rpu_backend.api.errors import UnsupportedModelError

    raise UnsupportedModelError(_UNSUPPORTED_MESSAGE)


class PaliGemmaPolicy:
    """Reserved public name for a currently unsupported PaliGemma2 policy."""

    def __init__(self) -> None:
        _raise_unsupported()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        **hf_kwargs: Any,
    ) -> "PaliGemmaPolicy":
        del pretrained_name_or_path, dtype, hf_kwargs
        _raise_unsupported()

    @classmethod
    def from_hf_model(cls, hf_model: Any) -> "PaliGemmaPolicy":
        del hf_model
        _raise_unsupported()

    def to(self, device: Any) -> "PaliGemmaPolicy":
        del device
        _raise_unsupported()

    @torch.no_grad()
    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        _raise_unsupported()

    @torch.no_grad()
    def _forward_rpu(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        _raise_unsupported()
