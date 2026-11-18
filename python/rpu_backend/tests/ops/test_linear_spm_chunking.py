"""Actual eager DDR Linear staging, chunk offsets and final partial chunk.

Keep the two established row/col geometries. FP16 eager Linear delegates each
SPM chunk to the production ACC32 Linear launcher. Row output is host-reduced;
col output is all-gathered before DDR write-back. This is eager A/B/A repetition,
not retained Graph REPLAY (the row path explicitly rejects Graph capture).

The sparse integer/power-of-two fixture has an independent exact oracle. Every
product and partial sum is exactly representable, so equality checks layout and
ownership; it is not a floating-point error budget for general GEMM or models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn as nn

SPM_USABLE = 8 * 1024 * 1024 - 1024
SPM_ALIGNMENT = 256
CORES = 8


def _fixture(m, k, n, partition):
    rows = torch.arange(m)
    shards = CORES if partition == 0 else 1
    local_k = k // shards
    selected_k = torch.stack([
        core * local_k + (rows * 7 + core * 3) % local_k
        for core in range(shards)
    ], dim=1)
    amplitudes = torch.stack([
        torch.where((rows + core) % 2 == 0, 1.0, -0.5)
        for core in range(shards)
    ], dim=1).half()
    x = torch.zeros((m, k), dtype=torch.float16)
    x[rows[:, None], selected_k] = amplitudes
    # Define logical weights before the production swizzle, independently of
    # its address calculation. Values are integers in [-6, 6].
    weight = ((torch.arange(n)[:, None] * 3 + torch.arange(k)[None, :]) % 13 - 6).half()
    return x, weight, selected_k, amplitudes


def _check_output(output, selected_k, amplitudes, n, sign):
    m = selected_k.shape[0]
    assert output.device.type == "rpu", output.device
    assert output.dtype == torch.float16 and output.shape == (m, n)
    actual = output.to("cpu")
    assert torch.isfinite(actual).all(), "non-finite eager DDR Linear output"
    channels = torch.arange(n)
    max_abs = 0.0
    mismatches = 0
    # Bound host scratch independently of the large DDR output width. No GEMM,
    # packed-weight decoder, or backend tile/register helper supplies the oracle.
    for row0 in range(0, m, 8):
        chosen = selected_k[row0:row0 + 8]
        amp = amplitudes[row0:row0 + 8]
        expected = torch.zeros((len(chosen), n), dtype=torch.float32)
        for core in range(chosen.shape[1]):
            values = (channels[None, :] * 3 + chosen[:, core, None]) % 13 - 6
            expected += values.float() * amp[:, core, None].float() * sign
        block = actual[row0:row0 + len(chosen)]
        mismatches += int(torch.count_nonzero(block.float() != expected))
        max_abs = max(max_abs, float((block.float() - expected).abs().max()))
    metrics = {"max_abs_error": max_abs, "unequal_elements": mismatches}
    print(json.dumps(metrics), flush=True)
    assert mismatches == 0, f"exact sparse DDR layout/oracle mismatch: {metrics}"
    return metrics


def check(m, k, n, expect_partition):
    import rpu_backend  # noqa: F401; native import only in the hardware entry
    from rpu_backend.runtime.weights import get_linear_partition, transform_linear_weight

    started = time.monotonic()
    partition = get_linear_partition(k, n)
    assert partition == expect_partition, (k, n, partition, expect_partition)
    local_k = k // CORES if partition == 0 else k
    local_n = n if partition == 0 else n // CORES
    # Include the col all-gather landing zone and the actual three allocation
    # round-ups. Runtime occupancy can reduce this upper bound, never enlarge it.
    row_bytes = 2 * (local_k + local_n + (n if partition else 0))
    chunk_rows_upper_bound = (SPM_USABLE - 3 * SPM_ALIGNMENT) // row_bytes
    chunks_lower_bound = -(-m // chunk_rows_upper_bound)
    assert chunks_lower_bound > 1, "fixture must cross an SPM chunk boundary"

    x, logical_weight, selected_k, amplitudes = _fixture(m, k, n, partition)
    linear = nn.Linear(k, n, bias=False).half().eval()
    linear.weight.data = transform_linear_weight(logical_weight, partition)
    linear = linear.to("rpu")
    inputs = [x.to("rpu"), (-x).to("rpu")]
    outputs = []
    reports = []
    with torch.inference_mode():
        for input_index, sign in ((0, 1), (1, -1), (0, 1)):
            result = linear(inputs[input_index])
            # All prior outputs remain live. A cached staging/output alias would
            # violate the public eager Linear ownership contract.
            assert all(result.data_ptr() != old.data_ptr() for old in outputs)
            outputs.append(result)
            reports.append(_check_output(result, selected_k, amplitudes, n, sign))
        # Later calls must also leave the first returned tensor's contents intact.
        retained = _check_output(outputs[0], selected_k, amplitudes, n, 1)
    report = {
        "M": m, "K": k, "N": n, "partition": "row" if partition == 0 else "col",
        "dtype": "float16", "accumulation": "production FP16 default ACC32",
        "chunk_rows_upper_bound": chunk_rows_upper_bound,
        "chunks_lower_bound": chunks_lower_bound, "calls": "A/B/A",
        "fresh_output_storage": True, "retained_first_output": retained,
        "call_metrics": reports, "elapsed_s": time.monotonic() - started,
    }
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [
        check(300, 128, 32752, expect_partition=0),  # >=3 chunks
        check(300, 16, 262016, expect_partition=1),  # >=22 gathered chunks
    ]
    report = {"completed": True, "cases": results}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("[summary] 2/2 eager DDR Linear cases passed", flush=True)


if __name__ == "__main__":
    main()
