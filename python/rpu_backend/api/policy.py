"""Public Pi0.5 policy facade.

Adapter and LeRobot imports remain method-local so importing the public API
does not load optional model dependencies.

Class lifecycle:
  - ``Pi05Policy.from_pretrained(...)`` — checkpoint constructor; lazy-imports
    ``rpu_backend.adapters.pi05.loader.load_and_construct_pi05_policy``.
  - ``Pi05Policy.from_lerobot_policy(...)`` — convenience wrapper that lazy-imports
    ``rpu_backend.adapters.pi05.Pi05Adapter``.
  - ``Pi05Policy.to('rpu')`` — irreversible CPU→RPU mutation routed through
    ``self._adapter.to_rpu()``.
  - ``Pi05Policy.select_action(batch)`` — public inference API that keeps the
    action queue on CPU.
  - ``Pi05Policy._forward_rpu(batch, ...)`` — internal direct-forward path.
"""
from __future__ import annotations
import os
import threading  # noqa: F401
from typing import Any

import torch


def _validate_batch(batch: dict, *, entry_point: str) -> dict:
    if not isinstance(batch, dict):
        raise TypeError(
            f"{entry_point}: batch must be a dict, got {type(batch).__name__}"
        )
    return batch


def _validate_num_steps(num_steps: int | None, *, entry_point: str) -> int | None:
    if num_steps is None:
        return None
    if (
        isinstance(num_steps, bool)
        or not isinstance(num_steps, int)
        or num_steps < 1
    ):
        raise ValueError(
            f"{entry_point}: num_steps must be a positive integer, got "
            f"{num_steps!r}"
        )
    return num_steps


class Pi05Policy:
    """Thin public wrapper around a Pi0.5 policy and its RPU adapter."""

    def __init__(self) -> None:
        raise RuntimeError(
            "Pi05Policy() is not a public constructor. Use "
            "Pi05Policy.from_pretrained(...) or Pi05Policy.from_lerobot_policy(...)."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = False,
        vlm_chunk_size: int | None = None,
        rpu_execution=None,
        **lerobot_kwargs: Any,
    ) -> "Pi05Policy":
        """Load a Pi0.5 checkpoint through the adapter loader.

        The loader is imported lazily from
        ``rpu_backend.adapters.pi05.loader``.
        """
        if not isinstance(trust_remote_code, bool):
            raise TypeError(
                "Pi05Policy.from_pretrained: trust_remote_code must be bool, "
                f"got {trust_remote_code!r}"
            )
        if trust_remote_code is not False:
            raise ValueError(
                "Pi05Policy.from_pretrained requires trust_remote_code=False; "
                "custom model code is not supported."
            )
        from rpu_backend.api._execution import normalize_rpu_execution
        execution_explicit = rpu_execution is not None or vlm_chunk_size is not None
        execution_config = normalize_rpu_execution(
            rpu_execution,
            entry_point="Pi05Policy.from_pretrained",
            vlm_chunk_size=vlm_chunk_size,
            supported={
                "prefill": ("chunk_size", "padding_rows", "padding_budget"),
                "vision": ("chunk_size",),
                "action": ("chunk_size",),
            },
        )
        # Keep this optional model dependency out of package import.
        from rpu_backend.adapters.pi05.loader import load_and_construct_pi05_policy
        policy = load_and_construct_pi05_policy(
            pretrained_name_or_path,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            **(
                {"rpu_execution": execution_config}
                if execution_explicit
                else {}
            ),
            **lerobot_kwargs,
        )
        # A real loader returns an already-bound Pi05Policy.  Keep this
        # assignment for small loader doubles and assert one canonical object
        # for both public and adapter views.
        if not hasattr(policy, "_rpu_execution"):
            policy._rpu_execution = execution_config
        policy._adapter._rpu_execution = policy._rpu_execution
        chunk_size = policy._rpu_execution.get("prefill", {}).get(
            "chunk_size", "auto"
        )
        policy._vlm_chunk_size = 0 if chunk_size == "auto" else chunk_size
        policy._adapter._vlm_chunk_size = policy._vlm_chunk_size
        return policy

    @classmethod
    def from_lerobot_policy(
        cls,
        lerobot_policy: Any,
        *,
        rpu_execution=None,
    ) -> "Pi05Policy":
        """Wrap an existing LeRobot policy.

        ``Pi05Adapter`` is imported lazily from
        ``rpu_backend.adapters.pi05``.
        """
        # Keep this optional model dependency out of package import.
        from rpu_backend.adapters.pi05 import Pi05Adapter
        from rpu_backend.api._execution import bind_rpu_execution
        supported = {
            "prefill": ("chunk_size", "padding_rows", "padding_budget"),
            "vision": ("chunk_size",),
            "action": ("chunk_size",),
        }
        inst = cls.__new__(cls)
        inst._lerobot_policy = lerobot_policy
        inst._adapter = Pi05Adapter(lerobot_policy)
        owner = getattr(lerobot_policy, "model", lerobot_policy)
        execution_config = bind_rpu_execution(
            owner,
            rpu_execution,
            entry_point="Pi05Policy.from_lerobot_policy",
            supported=supported,
        )
        inst._rpu_ready = inst._adapter._rpu_is_ready
        inst._rpu_execution = execution_config
        inst._adapter._rpu_execution = execution_config
        configured_chunk = execution_config.get("prefill", {}).get("chunk_size")
        if configured_chunk is not None:
            inst._vlm_chunk_size = (
                0 if configured_chunk == "auto" else configured_chunk
            )
            inst._adapter._vlm_chunk_size = inst._vlm_chunk_size
        else:
            inst._vlm_chunk_size = inst._adapter._vlm_chunk_size
        return inst

    def to(self, device: Any) -> "Pi05Policy":
        """Device routing."""
        from rpu_backend.api.errors import RPUBackendError
        from rpu_backend.runtime.device import is_rpu_device_target

        if is_rpu_device_target(device, entry_point="Pi05Policy.to"):
            try:
                self._adapter.to_rpu()
            except BaseException:
                self._rpu_ready = False
                raise
            self._rpu_ready = getattr(
                self._adapter, "_rpu_is_ready", True
            ) is True
            return self
        if (
            self._rpu_ready
            or getattr(self._adapter, "_rpu_is_ready", False)
            or getattr(self._lerobot_policy, "_rpu_swizzled", False)
            or getattr(self._lerobot_policy, "_rpu_swizzle_started", False)
        ):
            raise RPUBackendError(
                f"Pi05Policy.to({device!r}) rejected: weights are swizzled for "
                "RPU (irreversible). Reload a fresh policy for another device."
            )
        self._lerobot_policy.to(device)
        return self

    @torch.no_grad()
    def prepare_graphs(
        self,
        batch: dict,
        *,
        num_steps: "int | None" = None,
    ) -> dict:
        """Prebuild one finite Pi0.5 signature, freeze it, and verify READY."""
        if not getattr(self, "_rpu_ready", False):
            from rpu_backend.api.errors import RPUBackendError
            raise RPUBackendError(
                "prepare_graphs requires Pi05Policy.to('rpu') to complete first."
            )
        batch = _validate_batch(batch, entry_point="Pi05Policy.prepare_graphs")
        num_steps = _validate_num_steps(
            num_steps, entry_point="Pi05Policy.prepare_graphs"
        )
        from rpu_backend.runtime.hw_attrs import mark_first_forward_done
        mark_first_forward_done(self._lerobot_policy)
        return self._adapter.prepare_graphs(batch, num_steps=num_steps)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict,
        *,
        num_steps: "int | None" = None,
    ) -> torch.Tensor:
        """Run one full Pi0.5 inference and return the complete action chunk."""
        if not getattr(self, "_rpu_ready", False):
            from rpu_backend.api.errors import RPUBackendError
            raise RPUBackendError(
                "predict_action_chunk requires Pi05Policy.to('rpu') to "
                "complete first."
            )
        batch = _validate_batch(
            batch, entry_point="Pi05Policy.predict_action_chunk"
        )
        num_steps = _validate_num_steps(
            num_steps, entry_point="Pi05Policy.predict_action_chunk"
        )
        from rpu_backend.runtime.hw_attrs import mark_first_forward_done
        mark_first_forward_done(self._lerobot_policy)
        if num_steps is None:
            return self._lerobot_policy.predict_action_chunk(batch)
        return self._lerobot_policy.predict_action_chunk(
            batch, num_steps=num_steps)

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        """Run inference and return a CPU tensor.

        After LeRobot populates ``_action_queue``, move each queued tensor to
        CPU so no RPU tensors survive between calls.
        """
        batch = _validate_batch(batch, entry_point="Pi05Policy.select_action")

        # Record the first forward once before dispatch.
        from rpu_backend.runtime.hw_attrs import mark_first_forward_done
        mark_first_forward_done(self._lerobot_policy)

        result = self._lerobot_policy.select_action(batch)

        # Keep queued actions on CPU.
        queue = getattr(self._lerobot_policy, '_action_queue', None)
        if queue is None:
            queue = getattr(getattr(self._lerobot_policy, 'model', None),
                             '_action_queue', None)
        if queue is not None:
            from collections import deque
            from rpu_backend.api.errors import RPUBackendError

            if not isinstance(queue, deque):
                raise RPUBackendError(
                    "Pi05Policy expected LeRobot _action_queue to be a deque, "
                    f"got {type(queue).__name__}"
                )
            pinned = []
            for entry in queue:
                try:
                    if isinstance(entry, torch.Tensor) and entry.device.type != "cpu":
                        entry = entry.cpu()
                except Exception as exc:
                    raise RPUBackendError(
                        "Pi05Policy failed to move an _action_queue entry to CPU"
                    ) from exc
                pinned.append(entry)
            queue.clear()
            queue.extend(pinned)

        if not isinstance(result, torch.Tensor):
            from rpu_backend.api.errors import RPUBackendError

            raise RPUBackendError(
                "Pi05Policy.select_action runtime must return a torch.Tensor, "
                f"got {type(result).__name__}"
            )
        return result.cpu() if result.device.type != "cpu" else result

    @torch.no_grad()
    def _forward_rpu(self, batch: dict, *, return_chunk: bool = True,
                     num_steps: "int | None" = None) -> torch.Tensor:
        """Internal direct-forward path.

        return_chunk=True -> predict_action_chunk (RPU tensor).
        return_chunk=False -> self.select_action (CPU-pinned).
        Requires ``_rpu_ready`` set by ``Pi05Policy.to('rpu')``.
        """
        # ``_rpu_ready`` is a plain Python attribute on Pi05Policy.
        if not getattr(self, '_rpu_ready', False):
            from rpu_backend.api.errors import RPUBackendError
            raise RPUBackendError(
                "_forward_rpu requires Pi05Policy.to('rpu') to have completed first.")

        if not isinstance(return_chunk, bool):
            raise TypeError(
                "Pi05Policy._forward_rpu: return_chunk must be bool, got "
                f"{return_chunk!r}"
            )

        if return_chunk:
            return self.predict_action_chunk(batch, num_steps=num_steps)
        # Route through the wrapper that keeps queued and returned actions on CPU.
        return self.select_action(batch)
