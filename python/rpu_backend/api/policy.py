"""Pi0.5 policy entry point and device lifecycle.

Adapter and LeRobot imports are deferred until use to avoid importing
optional dependencies at package load time.

``from_pretrained`` loads the policy; ``from_lerobot_policy`` wraps an
existing policy. The adapter owns weight conversion and its mutation lock.
``to('rpu')`` performs the irreversible device conversion. ``select_action``
keeps queued actions on CPU. ``_forward_rpu`` requires conversion to have
completed and can return a device action chunk or a CPU action.
"""
from __future__ import annotations
import os
import threading
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


def _validate_noise(
    noise: torch.Tensor, batch: dict, config: Any, *, entry_point: str,
) -> None:
    if (
        not isinstance(noise, torch.Tensor)
        or noise.device.type != "cpu"
        or noise.dtype != torch.float32
    ):
        raise ValueError(f"{entry_point}: noise must be a CPU float32 tensor")
    state = batch.get("observation.state")
    if not isinstance(state, torch.Tensor) or state.ndim != 2:
        raise ValueError(
            f"{entry_point}: explicit noise requires observation.state "
            "[batch, features]"
        )
    expected = (state.shape[0], config.chunk_size, config.max_action_dim)
    if tuple(noise.shape) != expected or not bool(torch.isfinite(noise).all()):
        raise ValueError(f"{entry_point}: noise must be finite with shape {expected}")


class Pi05Policy:
    """Public Pi0.5 entry (D-02 thin wrapper)."""

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
        graph_runtime_policy=None,
        max_seq_len: int | None = None,
        **lerobot_kwargs: Any,
    ) -> "Pi05Policy":
        """D-01 canonical (delegates to loader.py per D-3-03).

        Lazy import per ADR §6.2: ``loader`` lives in
        ``rpu_backend.adapters.pi05.loader`` (v5-02 D-12 relocated from
        ``transformers/pi05/``).
        """
        if max_seq_len is not None and (type(max_seq_len) is not int or max_seq_len < 1):
            raise ValueError("max_seq_len must be a positive integer")
        if graph_runtime_policy is not None:
            from rpu_backend.graph import GraphRuntimePolicy
            if not isinstance(graph_runtime_policy, GraphRuntimePolicy):
                raise TypeError("graph_runtime_policy must be GraphRuntimePolicy or None")
        if not isinstance(trust_remote_code, bool):
            raise TypeError(
                "Pi05Policy.from_pretrained: trust_remote_code must be bool, "
                f"got {trust_remote_code!r}"
            )
        from rpu_backend.api._execution import normalize_rpu_execution
        from rpu_backend.api._execution import PI05_EXECUTION_COMPONENTS, PI05_EXECUTION_SUPPORTED
        execution_explicit = rpu_execution is not None or vlm_chunk_size is not None
        execution_config = normalize_rpu_execution(
            rpu_execution,
            entry_point="Pi05Policy.from_pretrained",
            vlm_chunk_size=vlm_chunk_size,
            supported=PI05_EXECUTION_SUPPORTED,
            supported_components=PI05_EXECUTION_COMPONENTS,
        )

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
            **({"graph_runtime_policy": graph_runtime_policy}
               if graph_runtime_policy is not None else {}),
            **({"max_seq_len": max_seq_len} if max_seq_len is not None else {}),
            **lerobot_kwargs,
        )


        if not hasattr(policy, "_rpu_execution"):
            policy._rpu_execution = execution_config
        policy._adapter._rpu_execution = policy._rpu_execution
        text_config = getattr(policy._adapter, "_vlm_execution", {})
        chunk_size = text_config.get("prefill", {}).get(
            "chunk_size",
            policy._rpu_execution.get("prefill", {}).get("chunk_size", "auto"),
        )
        policy._vlm_chunk_size = (
            0 if chunk_size == "auto" else int(chunk_size)
        )
        policy._adapter._vlm_chunk_size = policy._vlm_chunk_size
        return policy

    @classmethod
    def from_lerobot_policy(
        cls,
        lerobot_policy: Any,
        *,
        rpu_execution=None,
        graph_runtime_policy=None,
        max_seq_len: int | None = None,
    ) -> "Pi05Policy":
        """D-01 convenience.

        Lazy import per ADR §6.2 / codex G1-A: ``Pi05Adapter`` (the internal
        swizzle orchestrator) lives in ``rpu_backend.adapters.pi05`` (v5-02 B3
        relocated from ``transformers/pi05/adapter.py``; the class body now
        sits in the package's ``__init__.py``).
        """
        if max_seq_len is not None and (type(max_seq_len) is not int or max_seq_len < 1):
            raise ValueError("max_seq_len must be a positive integer")
        if graph_runtime_policy is not None:
            from rpu_backend.graph import GraphRuntimePolicy
            if not isinstance(graph_runtime_policy, GraphRuntimePolicy):
                raise TypeError("graph_runtime_policy must be GraphRuntimePolicy or None")

        from rpu_backend.adapters.pi05 import Pi05Adapter
        from rpu_backend.api._execution import PI05_EXECUTION_COMPONENTS, PI05_EXECUTION_SUPPORTED
        from rpu_backend.api._execution import bind_rpu_execution
        inst = cls.__new__(cls)
        inst._lerobot_policy = lerobot_policy
        owner = getattr(lerobot_policy, "model", lerobot_policy)
        execution_config = bind_rpu_execution(
            owner,
            rpu_execution,
            entry_point="Pi05Policy.from_lerobot_policy",
            supported=PI05_EXECUTION_SUPPORTED,
            supported_components=PI05_EXECUTION_COMPONENTS,
        )
        inst._adapter = Pi05Adapter(
            lerobot_policy,
            **({"max_seq_len": max_seq_len} if max_seq_len is not None else {}),
            **({"graph_runtime_policy": graph_runtime_policy}
                              if graph_runtime_policy is not None else {}))
        inst._rpu_ready = inst._adapter._rpu_is_ready
        inst._rpu_execution = execution_config
        text_config = getattr(inst._adapter, "_vlm_execution", {})
        chunk_size = text_config.get("prefill", {}).get(
            "chunk_size",
            execution_config.get("prefill", {}).get("chunk_size", "auto"),
        )
        inst._vlm_chunk_size = 0 if chunk_size == "auto" else int(chunk_size)
        inst._adapter._vlm_chunk_size = inst._vlm_chunk_size
        session = getattr(inst._adapter, "_execution_session", None)
        if session is not None:
            inst._execution_session = session
            session.register_config_view(inst)
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
                "device (irreversible). Reload a fresh policy for another device."
            )
        self._lerobot_policy.to(device)
        return self

    def close(self) -> None:
        """Retire Pi0.5 resources through its shared execution session."""
        self._adapter.close()
        self._rpu_ready = False

    @torch.no_grad()
    def prepare_graphs(
        self,
        batch: dict,
        *,
        num_steps: "int | None" = None,
        precompute_adarms: bool = False,
        noise: torch.Tensor | None = None,
    ) -> dict:
        """Prebuild one finite Pi0.5 signature, freeze it, and verify READY."""
        if not getattr(self, "_rpu_ready", False):
            from rpu_backend.api.errors import RPUBackendError
            raise RPUBackendError(
                "prepare_graphs requires Pi05Policy.to('rpu') to complete first."
            )
        batch = _validate_batch(batch, entry_point="Pi05Policy.prepare_graphs")
        if noise is not None:
            _validate_noise(
                noise, batch, self._lerobot_policy.config,
                entry_point="Pi05Policy.prepare_graphs",
            )
        num_steps = _validate_num_steps(
            num_steps, entry_point="Pi05Policy.prepare_graphs"
        )
        if type(precompute_adarms) is not bool:
            raise TypeError("precompute_adarms must be bool")
        from rpu_backend.runtime.hw_attrs import mark_first_forward_done
        mark_first_forward_done(self._lerobot_policy)
        return self._adapter.prepare_graphs(
            batch, num_steps=num_steps, precompute_adarms=precompute_adarms,
            **({"noise": noise} if noise is not None else {}),
        )

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict,
        *,
        num_steps: "int | None" = None,
        noise: torch.Tensor | None = None,
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
        if noise is not None:
            _validate_noise(
                noise, batch, self._lerobot_policy.config,
                entry_point="Pi05Policy.predict_action_chunk",
            )
        num_steps = _validate_num_steps(
            num_steps, entry_point="Pi05Policy.predict_action_chunk"
        )
        from rpu_backend.runtime.hw_attrs import mark_first_forward_done
        mark_first_forward_done(self._lerobot_policy)
        kwargs = {}
        if num_steps is not None:
            kwargs["num_steps"] = num_steps
        if noise is not None:
            kwargs["noise"] = noise
        return self._lerobot_policy.predict_action_chunk(batch, **kwargs)

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        """Public inference API (D-04). Returns CPU tensor (D-CR3-SELECT1).

        CR-R5 BLOCKER 7: after lerobot.select_action populates `_action_queue`
        from predict_action_chunk (device-resident on device path), walk the queue
        and `.cpu()` each entry IN-PLACE so zero device tensors survive.
        """
        batch = _validate_batch(batch, entry_point="Pi05Policy.select_action")


        from rpu_backend.runtime.hw_attrs import mark_first_forward_done
        mark_first_forward_done(self._lerobot_policy)

        result = self._lerobot_policy.select_action(batch)


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
        """Return a device action chunk or a CPU action after device setup.

        ``return_chunk=True`` uses ``predict_action_chunk``; otherwise this
        delegates to ``select_action``. Both require ``_rpu_ready``.
        """


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

        return self.select_action(batch)
