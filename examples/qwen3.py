#!/usr/bin/env python3
from pathlib import Path

from _common import run_example


if __name__ == "__main__":
    raise SystemExit(
        run_example(
            "qwen3",
            Path(__file__).with_name("configs") / "qwen3/text/0_6b/fp16.toml",
        )
    )
