"""Pi0.5 public entry class (Pi05Policy) — owning module per ADR §10 #16.

Phase 01.2-01a-2-policy-split (REQ SKEL-03 + REQ SKEL-01b). This file is the
canonical home for ``Pi05Policy``. Module-top-level imports are limited to
``torch / typing / threading / os`` per ADR §6.2 — every adapters / lerobot
reference goes through method-body lazy imports to keep ``api/`` free of
reverse-direction DAG edges and to avoid importing lerobot at package load time.

v5-02 D-12: lazy-import paths swung from ``rpu_backend.transformers.pi05.*``
to ``rpu_backend.adapters.pi05.*``. The transition shim at
``rpu_backend.transformers.pi05.policy`` is gone (transformers/pi05/ deleted
entirely with v5-02); the v5-01a-2 splits remain — Pi05Policy here, Pi05Adapter
in adapters/pi05/__init__.py.

Class lifecycle:
  - ``Pi05Policy.from_pretrained(...)`` — canonical constructor; lazy-imports
    ``rpu_backend.adapters.pi05.loader.load_and_construct_pi05_policy``.
  - ``Pi05Policy.from_lerobot_policy(...)`` — convenience wrapper that lazy-imports
    ``rpu_backend.adapters.pi05.Pi05Adapter`` (the internal orchestrator owns
    the swizzle / patch lock per G1-A).
  - ``Pi05Policy.to('rpu')`` — irreversible CPU→RPU mutation routed through
    ``self._adapter.to_rpu()``.
  - ``Pi05Policy.select_action(batch)`` — public inference API; CR-R5 BLOCKER 7
    queue-pin loop CPU-pins ``_action_queue`` entries.
  - ``Pi05Policy._forward_rpu(batch, ...)`` — internal MSE-harness path; AM-7
    requires ``_rpu_ready`` before claiming the RPU path.
"""
from __future__ import annotations
import os
import threading  # noqa: F401  (intentional; matches policy.py:14 for byte-equal lift discipline. ADR §6.2 allows it.)
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
        optimized_profile=None,
        **lerobot_kwargs: Any,
    ) -> "Pi05Policy":
        """D-01 canonical (delegates to loader.py per D-3-03).

        Lazy import per ADR §6.2: ``loader`` lives in
        ``rpu_backend.adapters.pi05.loader`` (v5-02 D-12 relocated from
        ``transformers/pi05/``).
        """
        if not isinstance(trust_remote_code, bool):
            raise TypeError(
                "Pi05Policy.from_pretrained: trust_remote_code must be bool, "
                f"got {trust_remote_code!r}"
            )
        from rpu_backend.adapters.pi05.optimized import (
            normalize_profile, execution_for_profile, profile_environment_scope,
            bind_profile, validate_checkpoint, require_profile_assets,
            _PRECISION_MAP,
        )
        profile = normalize_profile(optimized_profile)
        if profile is not None:
            from rpu_backend.adapters.pi05.loader import _load_pi05_rpu_quant_config
            validate_checkpoint(_load_pi05_rpu_quant_config(pretrained_name_or_path),
                                _PRECISION_MAP[profile["precision"]])
            require_profile_assets(profile, rpu_execution)
        rpu_execution = execution_for_profile(profile, rpu_execution)
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
        # ADR §6.2 / codex G2-A literal: lazy import, NOT module top-level.
        from rpu_backend.adapters.pi05.loader import load_and_construct_pi05_policy
        with profile_environment_scope(profile, rpu_execution):
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
        bind_profile(policy, profile)
        # A real loader returns an already-bound Pi05Policy.  Keep this
        # assignment for small loader doubles and assert one canonical object
        # for both public and adapter views.
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
        optimized_profile=None,
    ) -> "Pi05Policy":
        """D-01 convenience.

        Lazy import per ADR §6.2 / codex G1-A: ``Pi05Adapter`` (the internal
        swizzle orchestrator) lives in ``rpu_backend.adapters.pi05`` (v5-02 B3
        relocated from ``transformers/pi05/adapter.py``; the class body now
        sits in the package's ``__init__.py``).
        """
        # ADR §6.2 / codex G1-A literal: lazy import; lock + class both in adapters/pi05/__init__.py.
        from rpu_backend.adapters.pi05 import Pi05Adapter
        from rpu_backend.api._execution import PI05_EXECUTION_COMPONENTS, PI05_EXECUTION_SUPPORTED
        from rpu_backend.api._execution import bind_rpu_execution
        from rpu_backend.adapters.pi05.optimized import (
            normalize_profile, execution_for_profile, profile_environment_scope, bind_profile,
        )
        profile = normalize_profile(optimized_profile)
        rpu_execution = execution_for_profile(profile, rpu_execution)
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
        with profile_environment_scope(profile, rpu_execution):
            inst._adapter = Pi05Adapter(lerobot_policy)
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
        bind_profile(inst, profile)
        return inst

    def to(self, device: Any) -> "Pi05Policy":
        """Device routing."""
        from rpu_backend.api.errors import RPUBackendError
        from rpu_backend.runtime.device import is_rpu_device_target

        if is_rpu_device_target(device, entry_point="Pi05Policy.to"):
            try:
                from rpu_backend.adapters.pi05.optimized import profile_environment_scope, require_profile_assets
                profile = getattr(self, "_optimized_profile", None)
                require_profile_assets(profile, self._rpu_execution)
                with profile_environment_scope(profile, self._rpu_execution):
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
        precompute_adarms: bool | None = None,
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
        from rpu_backend.adapters.pi05.optimized import validate_profile_batch, profile_environment_scope
        profile = getattr(self, "_optimized_profile", None)
        validate_profile_batch(self, batch, num_steps)
        if precompute_adarms is None:
            precompute_adarms = profile is not None
        if type(precompute_adarms) is not bool:
            raise TypeError("precompute_adarms must be bool")
        from rpu_backend.runtime.hw_attrs import mark_first_forward_done
        mark_first_forward_done(self._lerobot_policy)
        with profile_environment_scope(profile, self._rpu_execution):
            return self._adapter.prepare_graphs(
                batch, num_steps=num_steps, precompute_adarms=precompute_adarms)

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
        from rpu_backend.adapters.pi05.optimized import validate_profile_batch
        validate_profile_batch(self, batch, num_steps)
        # Installation already froze the profile on the Python/native owners.
        # Inference must not republish cold flags into the process environment.
        if num_steps is None:
            return self._lerobot_policy.predict_action_chunk(batch)
        return self._lerobot_policy.predict_action_chunk(batch, num_steps=num_steps)

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        """Public inference API (D-04). Returns CPU tensor (D-CR3-SELECT1).

        CR-R5 BLOCKER 7: after lerobot.select_action populates `_action_queue`
        from predict_action_chunk (RPU-resident on RPU path), walk the queue
        and `.cpu()` each entry IN-PLACE so zero RPU tensors survive.
        """
        batch = _validate_batch(batch, entry_point="Pi05Policy.select_action")

        # D-32 A9 wire-point #3 (idempotent first-forward marker).
        from rpu_backend.runtime.hw_attrs import mark_first_forward_done
        mark_first_forward_done(self._lerobot_policy)

        from rpu_backend.adapters.pi05.optimized import validate_profile_batch
        validate_profile_batch(self, batch)
        result = self._lerobot_policy.select_action(batch)

        # CR-R5 BLOCKER 7: CPU-pin the queue.
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
        """D-701 internal surface — MSE harness path.

        return_chunk=True -> predict_action_chunk (RPU tensor).
        return_chunk=False -> self.select_action (CPU-pinned, AM-2).
        AM-7: requires `_rpu_ready` (set by Pi05Policy.to('rpu')).
        """
        # AM-7: _rpu_ready is a plain Python attr on Pi05Policy (NOT nn.Module),
        # so A9 validator does not intercept; no whitelist extension required.
        if not getattr(self, '_rpu_ready', False):
            from rpu_backend.api.errors import RPUBackendError   # ADR §6.2 lazy
            raise RPUBackendError(
                "_forward_rpu requires Pi05Policy.to('rpu') to have completed first.")

        if not isinstance(return_chunk, bool):
            raise TypeError(
                "Pi05Policy._forward_rpu: return_chunk must be bool, got "
                f"{return_chunk!r}"
            )

        if return_chunk:
            return self.predict_action_chunk(batch, num_steps=num_steps)
        # AM-2 fix: route through self.select_action(batch) CPU-pin wrapper.
        return self.select_action(batch)
