#!/usr/bin/env python3
from pathlib import Path

from _common import run_example


if __name__ == "__main__":
    raise SystemExit(
        run_example(
            "llama",
            Path(__file__).with_name("configs") / "llama/3_2_1b/fp16.toml",
        )
    )
