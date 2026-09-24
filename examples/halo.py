#!/usr/bin/env python3
from pathlib import Path

from _common import run_example


if __name__ == "__main__":
    raise SystemExit(run_example(
        "halo", Path(__file__).with_name("configs") / "halo/action_expert/fp16.toml"))
