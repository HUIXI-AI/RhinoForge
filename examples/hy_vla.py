#!/usr/bin/env python3
from pathlib import Path

from _common import run_example


if __name__ == "__main__":
    raise SystemExit(
        run_example(
            "hy_vla",
            Path(__file__).with_name("configs") / "hy_embodied/umi/fp16.toml",
        )
    )
