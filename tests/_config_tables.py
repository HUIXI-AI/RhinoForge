"""Read one TOML's fields from a generated model-page configuration section.

Group tables list only the columns that differ between their configurations;
values shared by the whole group follow under "本组配置相同的项", and a
single-configuration group is an item/value list. Tests read a configuration's
complete field set through this helper instead of assuming one wide row.
"""
from __future__ import annotations

import re

SHARED_HEADING = "本组配置相同的项："


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def _items(lines: list[str]) -> dict[str, str]:
    return {_cells(line)[0]: _cells(line)[1] for line in lines
            if line.startswith("| ") and not line.startswith(("| 项目 |", "|---"))}


def config_cells(page: str, relative: str) -> dict[str, str]:
    """All fields a model page shows for ``relative``: its row plus shared items."""
    link = f"[{relative}]({relative})"
    for section in page.split("\n### ")[1:]:
        lines = section.splitlines()
        if f"| 配置 | {link} |" in lines:
            return _items(lines)
        row = next((line for line in lines if line.startswith(f"| {link} |")), None)
        if row is None:
            continue
        header = next(line for line in lines if line.startswith("| 配置 |"))
        result = dict(zip(_cells(header), _cells(row), strict=True))
        if SHARED_HEADING in lines:
            result.update(_items(lines[lines.index(SHARED_HEADING):]))
        return result
    raise AssertionError(f"{relative} not found in the configuration section")
