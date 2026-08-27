#!/usr/bin/env python3
"""Generate the v1.0.0 kernel allowlist from reviewed host source."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAZY_KERNELS = {
    "llama_gather_embedding",
    "scatter_elements_v2_spm_smallC",
    "tile_Nx1_NxC_ddr",
    "tile_Nx1_NxC_v16_ddr",
    "tile_general_largeC_ddr",
    "tile_general_smallC_ddr",
    "topk_by_select_fp16",
    "transpose_cbn_c16",
    "transpose_cnb_c16",
    "unary_cast_uint16_int32",
    "unary_cast_uint8_fp16",
}


def _block(text: str, start: str, end: str) -> str:
    try:
        return text.split(start, 1)[1].split(end, 1)[0]
    except IndexError as exc:
        raise RuntimeError(f"source contract marker missing: {start!r}") from exc


def _autotile_names() -> set[str]:
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("c++ is required to evaluate the host autotile table")
    source = (
        '#include "rpu_linear_tiling.h"\n#include <iostream>\n'
        "int main(){for(const auto& n:rpu_pl_tiling::autotile_kernel_names())"
        'std::cout<<n<<"\\n";}\n'
    )
    with tempfile.TemporaryDirectory() as directory:
        executable = Path(directory) / "autotile_names"
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                f"-I{ROOT / 'src/ops'}",
                "-x",
                "c++",
                "-",
                "-o",
                str(executable),
            ],
            input=source,
            text=True,
            check=True,
        )
        return set(
            subprocess.check_output([str(executable)], text=True).splitlines()
        )


def _reachable_names() -> set[str]:
    cache = (ROOT / "src/core/rpu_kernel_cache.inc").read_text(encoding="utf-8")
    header = (ROOT / "src/core/rpu_kernel_cache.h").read_text(encoding="utf-8")
    preloaded = set(
        re.findall(
            r'\{"([A-Za-z_][A-Za-z0-9_]*)",\s*'
            r'(?:RHINO_OP|GEMM|SOFTMAX)_LIB_PATH\}',
            _block(cache, "KERNEL_LIST = {", "};\n\nstatic void print_tensor_info"),
        )
    )
    ids = set(
        re.findall(
            r'^\s*"([A-Za-z_][A-Za-z0-9_]*)"',
            _block(header, "KERNEL_ID_NAMES[] = {", "};\nstatic_assert"),
            re.MULTILINE,
        )
    )
    autotile = _autotile_names()
    if (len(preloaded), len(ids), len(autotile)) != (156, 159, 42):
        raise RuntimeError("v1.0.0 fixed kernel inventory changed; review it")
    if ids - preloaded != autotile:
        raise RuntimeError("KernelId/autotile inventory drifted")

    bmm = (ROOT / "src/ops/rpu_bmm.cpp").read_text(encoding="utf-8")
    tile_block = _block(bmm, "kPreloadedEagerBmmTiles[][3] = {", "};")
    tiles = {
        tuple(map(int, match))
        for match in re.findall(r"\{(\d+),\s*(\d+),\s*(\d+)\}", tile_block)
    }
    eager_bmm = {
        f"gemm_fp16_spm_16b_w{m}x{n}_k{k}_1core_buf1_nt_lpaddr_{mode}"
        for m, n, k in tiles
        for mode in ("peak", "univ")
    }
    if len(tiles) != 6 or not eager_bmm <= preloaded:
        raise RuntimeError("eager BMM allowlist drifted")

    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (ROOT / "src").rglob("*")
        if path.suffix in {".cc", ".cpp", ".cxx", ".h", ".hpp", ".inc"}
    )
    direct = set(
        re.findall(
            r'\b(?:get_kernel(?:_reset)?|graph_kernel_by_name)\s*'
            r'\(\s*"([A-Za-z_][A-Za-z0-9_]*)"',
            source_text,
        )
    )
    names = preloaded | ids | autotile | LAZY_KERNELS
    if not LAZY_KERNELS <= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source_text)):
        raise RuntimeError("lazy kernel reachability changed")
    if not direct <= names or len(names) != 209:
        raise RuntimeError("v1.0.0 reachable kernel inventory changed; review it")
    return names


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} OPERATOR_ASSET OUTPUT.kernels")
    asset = Path(sys.argv[1])
    output = Path(sys.argv[2])
    names = sorted(_reachable_names())
    manifest = (
        "rhinoforge-kernels-v1\n"
        f"asset-size={asset.stat().st_size}\n"
        + "\n".join(names)
        + "\n"
    )
    output.write_text(manifest, encoding="ascii")


if __name__ == "__main__":
    main()
