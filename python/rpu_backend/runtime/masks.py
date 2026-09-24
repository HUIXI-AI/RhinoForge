"""Owned CPU masks for adapters that cache invariant attention layouts."""

import torch


def versioned_cpu_mask(mask: torch.Tensor) -> torch.Tensor:
    """Copy an internal additive mask into a versioned CPU FP16 owner.

    Call only when an adapter's content-keyed cache creates a new mask. Native
    mask preparation can then reuse the same tensor while checking its mutation
    version, including when the caller runs under ``torch.inference_mode()``.
    The copy also separates the cache from an inference tensor's untracked
    storage. Arbitrary caller-owned masks still use native conservative checks.
    Keeping this owner on CPU avoids a device readback during preparation.
    """
    with torch.inference_mode(False), torch.no_grad():
        return mask.detach().to(
            device="cpu", dtype=torch.float16, copy=True
        ).contiguous()
