#!/usr/bin/env python3
"""Generate an opaque-asset sidecar from host reachability and vendor names."""

from __future__ import annotations

import argparse
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
    if not preloaded or not ids or not autotile:
        raise RuntimeError("host kernel inventory is empty")

    # The eager and fused BMM launchers derive these names from concrete
    # M/N/K tiling and the requested transpose layout.
    eager_bmm = {
        f"gemm_fp16_spm_16b_w{m}x{n}_k{k}_1core_buf1_{layout}_lpaddr_{mode}"
        for m in (64, 96, 128, 192, 256)
        for n in (64, 96, 128, 192, 256)
        for k in (64, 96, 128)
        for layout in ("nt", "nn", "tn", "tt")
        for mode in ("peak", "univ")
    }

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
    mixed = set(re.findall(
        r'"([A-Za-z_][A-Za-z0-9_]*)"',
        _block(cache, "kMixedNames[] = {", "};"),
    ))
    names = preloaded | ids | autotile | LAZY_KERNELS | mixed | direct | eager_bmm
    if not LAZY_KERNELS <= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source_text)):
        raise RuntimeError("lazy kernel reachability changed")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--available-kernels", type=Path, required=True,
        help="Vendor-provided plain-text kernel names; the asset is never parsed",
    )
    args = parser.parse_args()
    available = args.available_kernels.read_text(encoding="ascii").splitlines()
    if not available or len(available) != len(set(available)) or any(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,254}", name) is None
        for name in available
    ):
        raise ValueError("vendor kernel names must be nonempty, unique identifiers")
    names = _reachable_names() & set(available)
    if not names:
        raise ValueError("vendor asset has no host-reachable kernels")
    args.output.write_text(
        "rhinoforge-kernels-v1\n"
        f"asset-size={args.asset.stat().st_size}\n"
        + "\n".join(sorted(names)) + "\n",
        encoding="ascii",
    )
    print(f"wrote {len(names)} host-reachable kernel names to {args.output}")


if __name__ == "__main__":
    main()
