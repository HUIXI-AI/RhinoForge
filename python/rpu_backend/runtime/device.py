"""Model-neutral device parsing shared by public policies and adapters."""
from __future__ import annotations

import torch


def extract_to_device_target(args, kwargs):
    """Return the device-shaped target from an ``nn.Module.to`` call, if any."""
    if args and isinstance(args[0], (str, torch.device, int)):
        return args[0]
    if "device" in kwargs:
        return kwargs["device"]
    return None


def is_rpu_device_target(device, *, entry_point: str) -> bool:
    """Classify a ``.to`` target without treating malformed RPU strings as RPU."""
    if isinstance(device, int):
        raise ValueError(
            f"{entry_point}: integer device indices are ambiguous; use "
            "'rpu' or 'rpu:0' explicitly"
        )
    if not isinstance(device, (str, torch.device)):
        return False
    try:
        resolved = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raw = str(device).strip().lower()
        if raw.startswith(("rpu", "privateuseone")):
            raise ValueError(
                f"{entry_point}: invalid RPU device target {device!r}; use "
                "'rpu' or 'rpu:0'"
            ) from exc
        return False
    if resolved.type != "rpu":
        return False
    if resolved.index not in {None, 0}:
        raise ValueError(
            f"{entry_point}: only RPU device 0 is supported, got {device!r}"
        )
    return True


__all__ = ["extract_to_device_target", "is_rpu_device_target"]
