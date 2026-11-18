"""Live openpi pi05 inference on RPU for one raw-observation Pi0.5 example TOML.

``OpenpiPi05Server`` loads the checkpoint, precision and text width of a Kuavo
TOML (``examples/configs/pi05/kuavo``), prepares its Graphs once and then serves
requests of three uint8 camera frames, the physical state and an instruction.
It returns physical actions ``[50, action_dim]`` (openpi Unnormalize plus
AbsoluteActions), the same transforms as the example runner.

The fused denoise bakes the suffix RoPE start, which is the valid prefix length,
at Graph BUILD. A live state changes the length: each state bin is written as a
1–3 digit number. Startup therefore prepares every length the configured
instructions reach over all states. A request outside that set is refused before
any device work; no Graph is built online.

Non-finite guard: a non-finite state is refused. With ``nonfinite="hold"`` (the
default), a non-finite model output is replaced by a hold chunk, and the event
is counted and logged. A hold chunk repeats the current physical state, so every
action dimension must be an absolute position target in the state's layout.
``nonfinite="error"`` raises instead.

The process must be fresh: every cold selector is bound before rpu_backend is
imported, and swizzling is irreversible (one live policy per process).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from examples.runners.pi05 import openpi_inputs as _openpi
from examples.runners.pi05 import optimized as _optimized

NONFINITE_MODES = ("hold", "error")
NUM_STEPS = 10


class NonFiniteActionError(RuntimeError):
    """The model returned NaN/Inf actions and the guard is in ``error`` mode."""


def load_example_config(path) -> dict:
    """A Pi0.5 raw-observation example TOML in the runner layout."""
    import tomllib

    from examples.common.config import normalize_config

    path = Path(path).expanduser()
    config = normalize_config(tomllib.loads(path.read_text(encoding="utf-8")))
    if "pi05" not in config:
        raise ValueError(f"{path} is not a Pi0.5 example configuration")
    return config


class OpenpiPi05Server:
    """One prepared openpi pi05 policy on RPU serving live observations."""

    def __init__(self, config: dict, *, instructions=None, nonfinite: str = "hold",
                 noise: str = "per_request", seed: int | None = None, log=print):
        if nonfinite not in NONFINITE_MODES:
            raise ValueError(f"nonfinite must be one of {NONFINITE_MODES}")
        if noise not in ("per_request", "fixed"):
            raise ValueError("noise must be per_request or fixed")
        args = _optimized.example_arguments(config)
        raw = args.raw_inputs
        if raw is None:
            raise ValueError("serving needs a raw-observation TOML ([input].images/instruction/state/"
                             "norm_stats/tokenizer), not [input].batch")
        if args.cameras != 3:
            raise ValueError("openpi pi05 serving uses exactly three cameras")
        if args.noise is not None:
            raise ValueError("serving samples its own noise; remove [input].noise")
        self._log = log
        self._lock = threading.Lock()
        self.text_tokens = args.text_tokens
        self.nonfinite = nonfinite
        self.noise_mode = noise
        self.seed = args.seed if seed is None else seed
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.stats = _openpi.load_norm_stats(raw["norm_stats"])
        if "actions" not in self.stats:
            raise ValueError("serving returns physical actions; the norm stats need action statistics")
        self.processor = _openpi.load_tokenizer(raw["tokenizer"])
        self.state_dim = len(raw["state"])
        # Requests carry the camera frames at this size ([width, height]); others are refused.
        self.frame_size = list(raw["frame_size"])
        self.delta_action_mask = raw.get("delta_action_mask")
        instructions = [raw["instruction"]] if instructions is None else list(dict.fromkeys(instructions))
        if not instructions or any(not isinstance(item, str) or not item.strip() for item in instructions):
            raise ValueError("instructions must be nonempty strings")
        self.default_instruction = instructions[0]
        self.instructions = tuple(instructions)
        counts = set()
        for instruction in self.instructions:
            counts.update(_openpi.reachable_token_counts(
                self.processor, instruction, self.state_dim, self.text_tokens))
        self.language_token_counts = tuple(sorted(counts))

        started = time.perf_counter()
        setup = _optimized.cold_setup(args, bind_libraries=True)
        import torch

        self._torch = torch
        checkpoint, contract = _optimized.resolve_checkpoint(args)
        self.action_dim = contract["action_dim"]
        if nonfinite == "hold" and self.action_dim != self.state_dim:
            raise ValueError(f"a hold chunk repeats the state; action_dim={self.action_dim} differs from "
                             f"state_dim={self.state_dim} (use nonfinite='error')")
        # The warmup request uses the TOML frames and state with the first instruction,
        # so its valid length is one of the prepared lengths.
        batch, record = _openpi.build_batch(dict(raw, instruction=self.default_instruction),
                                            text_tokens=self.text_tokens, processor=self.processor)
        # Every request is resized to the same shape, so one check covers them all.
        _optimized.validate_image_shapes(batch, contract["image_shapes"])
        self._generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.policy = _optimized.load_policy(args, checkpoint, setup.execution,
                                             language_counts=len(self.language_token_counts))
        try:
            _optimized.install_policy(self.policy, args, batch)
            loaded = time.perf_counter()
            _, _, _, action_cfg = _optimized._execution_config(setup.execution)
            precompute_adarms = setup.paired or (
                setup.cores in (4, 6)
                and args.precision in ("fp16", "w8a16", "w4a16")
                and not action_cfg.get("action", {}).get("linear_acc32", False)
            )
            prepared = self.policy.prepare_graphs(
                batch, num_steps=NUM_STEPS, precompute_adarms=precompute_adarms,
                noise=self._noise(), language_token_counts=self.language_token_counts)
        except BaseException:
            self.policy.close()
            raise
        self.prepared = {key: prepared.get(key) for key in (
            "phase", "graph_counts", "language_token_counts", "ready_replay_validated",
            "build_replay_bit_equal")}
        self.checkpoint = str(checkpoint)
        self.precision = args.public_precision
        self.execution_route = "paired_tp8" if setup.paired else "generic"
        self.startup = {"load_s": round(loaded - started, 3),
                        "prepare_s": round(time.perf_counter() - loaded, 3),
                        "representative_valid_tokens": record["valid_text_tokens"]}
        self.counters = {"requests": 0, "nonfinite_outputs": 0, "refused": 0}
        self.last_timing_ms: dict = {}
        self._generator.manual_seed(self.seed)  # the first request draws the seed's first noise
        self._log(f"[pi05-rpu] READY {self.precision} T{self.text_tokens} {self.execution_route}: "
                  f"{len(self.language_token_counts)} valid lengths "
                  f"{self.language_token_counts[0]}..{self.language_token_counts[-1]}, "
                  f"load {self.startup['load_s']} s, prepare {self.startup['prepare_s']} s")

    def _noise(self):
        """Initial noise [1,50,32]: fresh per request (openpi), or the seed's draw every time."""
        if self.noise_mode == "fixed":
            self._generator.manual_seed(self.seed)
        return self._torch.randn((1, 50, 32), generator=self._generator, dtype=self._torch.float32)

    def metadata(self) -> dict:
        return {"checkpoint": self.checkpoint, "precision": self.precision,
                "text_tokens": self.text_tokens, "execution_route": self.execution_route,
                "frame_size": list(self.frame_size),
                "state_dim": self.state_dim, "action_dim": self.action_dim, "horizon": 50,
                "num_steps": NUM_STEPS, "instructions": list(self.instructions),
                "language_token_counts": list(self.language_token_counts),
                "nonfinite": self.nonfinite, "noise": self.noise_mode, "seed": self.seed,
                "prepared": dict(self.prepared), "startup": dict(self.startup),
                "counters": dict(self.counters)}

    def reset(self) -> None:
        """Restart the per-request noise sequence (the policy itself keeps no episode state)."""
        with self._lock:
            self._generator.manual_seed(self.seed)

    def infer(self, frames, state, instruction: str | None = None):
        """Three uint8 [H,W,3] frames + physical state -> (physical actions [50, D] float32, info)."""
        import numpy as np

        torch = self._torch
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        instruction = self.default_instruction if instruction is None else str(instruction)
        with self._lock:
            self.counters["requests"] += 1
            try:
                if state.shape[0] != self.state_dim:
                    raise ValueError(f"state has {state.shape[0]} values, the policy expects {self.state_dim}")
                if not np.isfinite(state).all():
                    raise ValueError("state contains NaN/Inf; refusing to run the policy")
                if not instruction.strip():
                    raise ValueError("instruction must be nonempty")
                started = time.perf_counter()
                batch, record = _openpi.batch_from_frames(
                    frames, state, instruction, self.stats, self.processor, text_tokens=self.text_tokens,
                    frame_size=self.frame_size)
                valid = record["valid_text_tokens"]
                if valid not in self.language_token_counts:
                    raise ValueError(
                        f"the request has {valid} valid language tokens; this server prepared "
                        f"{self.language_token_counts[0]}..{self.language_token_counts[-1]} for its "
                        "configured instructions (add the instruction at startup)")
            except ValueError:
                self.counters["refused"] += 1
                raise
            prepared_at = time.perf_counter()
            output = self.policy.predict_action_chunk(batch, num_steps=NUM_STEPS, noise=self._noise())
            output = output.detach().cpu()
            inferred_at = time.perf_counter()
            if tuple(output.shape) != (1, 50, self.action_dim):
                raise RuntimeError(f"expected action output [1,50,{self.action_dim}], got {tuple(output.shape)}")
            normalized_finite = bool(torch.isfinite(output).all())
            physical = None
            if normalized_finite:
                with np.errstate(over="ignore", invalid="ignore"):  # an overflow is held below
                    physical = _openpi.absolute_actions(
                        output[0].double().numpy(), self.stats["actions"], state, self.delta_action_mask)
            guarded = physical is None or not np.isfinite(physical).all()
            if guarded:
                self.counters["nonfinite_outputs"] += 1
                message = (f"[pi05-rpu] non-finite actions (request {self.counters['requests']}, "
                           f"normalized finite={normalized_finite})")
                if self.nonfinite == "error":
                    self._log(message + "; returning an error")
                    raise NonFiniteActionError(message)
                self._log(message + "; returning a hold chunk at the current state")
                physical = np.repeat(state[None, :], 50, axis=0)
            actions = physical.astype(np.float32)
            self.last_timing_ms = {
                "host_prepare_ms": (prepared_at - started) * 1000,
                "policy_ms": (inferred_at - prepared_at) * 1000,
                "total_ms": (time.perf_counter() - started) * 1000}
            info = {"valid_text_tokens": valid, "guarded": guarded, "timing_ms": dict(self.last_timing_ms)}
            return actions, info

    def close(self) -> None:
        policy, self.policy = getattr(self, "policy", None), None
        if policy is not None:
            policy.close()
