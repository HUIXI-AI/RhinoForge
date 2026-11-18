"""Production Grouped wrappers against independent logical CPU one-hot goldens.

Run under the shared board lock with the integration build and formal BASE REF.
The nonuniform counts, empty experts, biases and per-expert scale tails exercise
new BASE addressing. EXP ACC32 is a separate retained ABI case. No performance
or model numerical claim is made by these immediate kernel checks.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


def run_case(mode: int, partition: str, acc32: bool = False) -> dict:
    from rpu_backend.quant.fp8 import pack_grouped_weight

    cores, experts, local_n, local_k = 8, 5, 48, 96
    col = partition == "col"
    n, k = (cores * local_n, local_k) if col else (local_n, cores * local_k)
    counts = torch.tensor([2, 0, 1, 0, 3], dtype=torch.int16)
    m = int(counts.sum())
    # Logical source values do not depend on any runtime packer or unpacker.
    logical = ((torch.arange(experts * n * k).reshape(experts, n, k) * 7 // 13) % 5 - 2)
    scale = torch.pow(2., -((torch.arange(experts * n).reshape(experts, n) % 4) + 2)).half()
    if mode == 2:
        source = (logical.float() / 4).half()
        decoded = source.float()
        physical_scale = None
    elif mode == 1:
        source = logical.to(torch.int8)
        decoded = logical.float() * scale.float().unsqueeze(-1)
        physical_scale = scale if acc32 else F.pad(scale, (0, (-n) % 128))
    else:
        source = (logical.float() / 2).to(torch.float8_e4m3fn).view(torch.uint8)
        decoded = (logical.float() / 2) * scale.float().unsqueeze(-1)
        physical_scale = scale if acc32 else F.pad(scale, (0, (-n) % 128))
    x = torch.zeros((cores, m, local_k), dtype=torch.float16)
    bias = ((torch.arange(cores * local_n) % 3 - 1).reshape(cores, local_n).float() / 8).half()
    expected = torch.empty((cores, m, local_n), dtype=torch.float16)
    row_experts = torch.repeat_interleave(torch.arange(experts), counts.long()).tolist()
    for core in range(cores):
        for row, expert in enumerate(row_experts):
            kk = (row * 31 + core * 13) % local_k
            x[core, row, kk] = -0.5 if row % 2 else 0.5
            ns = slice(core * local_n, (core + 1) * local_n) if col else slice(None)
            ks = kk if col else core * local_k + kk
            expected[core, row] = (decoded[expert, ns, ks] * x[core, row, kk] + bias[core]).half()
    weight = pack_grouped_weight(source, partition=partition, num_cores=cores).to("rpu")
    scale_rpu = None if physical_scale is None else physical_scale.contiguous().to("rpu")
    xr, cr, br = x.to("rpu"), counts.to("rpu"), bias.to("rpu")
    actual = torch.ops.rpu.grouped_parallel_linear_test(
        xr, weight, cr, scale_rpu, br, local_n, local_k, col, cores, mode, acc32).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The legacy unpadded owner must fail admission before BASE launch.
    if mode != 2 and not acc32 and n % 128:
        try:
            torch.ops.rpu.grouped_parallel_linear_test(
                xr, weight, cr, scale.to("rpu"), br,
                local_n, local_k, col, cores, mode, acc32)
        except RuntimeError as error:
            assert "256-byte-aligned expert scale stride" in str(error)
        else:
            raise AssertionError("BASE accepted an EXP scale owner with the wrong expert stride")
    bad_counts = counts.clone()
    bad_counts[0] += 1
    try:
        torch.ops.rpu.grouped_parallel_linear_test(
            xr, weight, bad_counts.to("rpu"), scale_rpu, br,
            local_n, local_k, col, cores, mode, acc32)
    except RuntimeError as error:
        assert "sum(m_sizes) must equal M" in str(error)
    else:
        raise AssertionError("invalid expert counts were accepted")
    report = dict(mode=mode, partition=partition, acc32=acc32, experts=experts,
                  rows=m, local_n=local_n, local_k=local_k,
                  scale_expert_bytes=0 if physical_scale is None else physical_scale.shape[1] * 2,
                  unequal_elements=int(torch.count_nonzero(actual != expected)))
    print("GROUPED_BASE", json.dumps(report), flush=True)
    return report


def run_all():
    reports = [run_case(mode, partition) for mode in (2, 1, 5) for partition in ("col", "row")]
    reports.append(run_case(5, "row", acc32=True))
    return reports


def test_grouped_base_linear():
    import pytest
    if os.environ.get("RPU_HW_AVAILABLE") != "1":
        pytest.skip("requires an exclusively reserved RPU board")
    run_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import rpu_backend  # noqa: F401
    report = {"scope": "production grouped BASE and retained EXP ACC32 wrappers", "cases": run_all()}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
