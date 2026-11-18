"""kuavo_server adapter: openpi pi05 on the Rhino RPU through RhinoForge.

Install into a LeTools-Learning checkout (see examples/configs/pi05/README.md#kuavo-server):

1. copy this file to ``kuavo_server/adapters/openpi_pi05_rpu.py`` at the repo root
   (or ``src/kuavo_server/adapters/openpi_pi05_rpu.py`` in older layouts);
2. add ``"openpi_pi05_rpu": "kuavo_server.adapters.openpi_pi05_rpu"`` to
   ``kuavo_server/builtin_adapters.py:ADAPTER_MODULES`` (or ``src/kuavo_server/...``).

The model code stays in the RhinoForge checkout given by ``--rpu_backend_root``
(or ``RPU_BACKEND_ROOT``). Keep this module free of torch and rpu_backend imports:
the RPU process binds CPU threads and runtime selectors before torch loads.
"""
from __future__ import annotations

import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from ..runtime import register_adapter
from .base import ModelServerAdapter


def _rpu_backend_body(root: str):
    """Put the rpu_backend checkout on sys.path and import its adapter body."""
    root = (root or os.environ.get("RPU_BACKEND_ROOT", "")).strip()
    if not root or not (Path(root).expanduser() / "examples" / "runners" / "pi05" / "kuavo.py").is_file():
        raise FileNotFoundError("--rpu_backend_root (or RPU_BACKEND_ROOT) must name the rpu_backend checkout "
                                "that contains examples/runners/pi05/kuavo.py")
    root_path = Path(root).expanduser().resolve()
    loaded = sys.modules.get("examples")
    if loaded is not None and Path(getattr(loaded, "__file__", "") or ".").resolve().parent != root_path / "examples":
        raise ImportError(f"another `examples` package is already imported ({loaded!r}); "
                          "start the server in a fresh process")
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))
    from examples.runners.pi05 import kuavo

    return kuavo


@register_adapter
class OpenpiPi05RpuAdapter(ModelServerAdapter):
    """openpi pi05 (Kuavo example TOML) served on the RPU."""

    name = "openpi_pi05_rpu"

    def __init__(self, body: Any) -> None:
        self._body = body

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--rpu_backend_root", type=str, default="",
                            help="rpu_backend checkout; default $RPU_BACKEND_ROOT")
        parser.add_argument("--rpu_config", type=str, required=True,
                            help="rpu_backend examples/configs/pi05/kuavo/<layout>/<precision>.toml "
                                 "(checkpoint, precision, norm stats, tokenizer, warmup frames and state)")
        parser.add_argument("--prompt", type=str, default="",
                            help="deployment task prompt; default the TOML instruction")
        parser.add_argument("--prompts_file", type=str, default="",
                            help="more prompts to prepare, one per line")
        parser.add_argument("--which_arm", type=str, default="both", choices=["left", "right", "both"])
        parser.add_argument("--arm_joint_dim", type=int, default=7)
        parser.add_argument("--end_effector_dim", type=int, default=1)
        parser.add_argument("--execution_horizon", type=int, default=0,
                            help="steps returned per chunk; 0 returns all 50")
        parser.add_argument("--nonfinite", type=str, default="hold", choices=["hold", "error"],
                            help="NaN/Inf model output: hold = return the current state as a hold chunk; "
                                 "error = return an error to the client")
        parser.add_argument("--noise", type=str, default="per_request", choices=["per_request", "fixed"],
                            help="per_request draws fresh initial noise from a seeded generator (openpi); "
                                 "fixed reuses the seed's noise every call")
        parser.add_argument("--seed", type=int, default=None, help="noise seed; default the TOML [run].seed")

    @classmethod
    def from_args(cls, args: Namespace) -> "OpenpiPi05RpuAdapter":
        kuavo = _rpu_backend_body(args.rpu_backend_root)
        prompts = [args.prompt.strip()] if args.prompt.strip() else []
        if args.prompts_file:
            lines = Path(args.prompts_file).expanduser().read_text(encoding="utf-8").splitlines()
            prompts += [line.strip() for line in lines if line.strip()]
        return cls(kuavo.KuavoOpenpiPi05.from_config(
            args.rpu_config, prompts=prompts, nonfinite=args.nonfinite, noise=args.noise, seed=args.seed,
            which_arm=args.which_arm, arm_joint_dim=args.arm_joint_dim,
            end_effector_dim=args.end_effector_dim, execution_horizon=args.execution_horizon))

    def metadata(self) -> dict[str, Any]:
        return self._body.metadata()

    def reset(self) -> dict[str, Any]:
        return self._body.reset()

    def select_action(self, obs: dict[str, Any]) -> Any:
        return self._body.select_action(obs)

    def select_action_chunk(self, obs: dict[str, Any]) -> Any:
        return self._body.select_action_chunk(obs)
