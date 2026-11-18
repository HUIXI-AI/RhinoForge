"""Formal kvsort_dir_v1 through the production full-sort launcher, with CPU golden.

Use the integration build under the shared board lock. The C++ shim checks all
four SPM guards and initialized input padding at their actual capacity, then
returns logical keys and opaque 16-bit values. Unique finite FP16 keys make
value ordering unambiguous without assuming a stable tie-breaking algorithm.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def logical_case(c: int, cores: int = 8):
    if not 1 <= c <= 24000:
        raise ValueError("bounded golden fixture needs C in [1,24000]")
    rng = np.random.default_rng(317 + c)
    keys = np.empty((cores, c), dtype=np.float16)
    values = np.empty((cores, c), dtype=np.uint16)
    for core in range(cores):
        # Distinct positive normals with exact binary16 encodings. No codec
        # or hardware sorting implementation participates in this oracle.
        key_bits = np.arange(0x0400, 0x0400 + c, dtype=np.uint16)
        permutation = rng.permutation(c)
        keys[core] = key_bits.view(np.float16)[permutation]
        values[core] = ((np.arange(c, dtype=np.uint32) * 40503 + core * 5011 + 0x8000)
                        & 0xffff).astype(np.uint16)
    return torch.from_numpy(keys), torch.from_numpy(values.view(np.int16))


def cpu_golden(keys, values, descending):
    order = np.argsort(keys.numpy().astype(np.float32), axis=1)
    if descending:
        order = order[:, ::-1]
    return (torch.from_numpy(np.take_along_axis(keys.numpy(), order, axis=1).copy()),
            torch.from_numpy(np.take_along_axis(values.numpy(), order, axis=1).copy()))


def run_all():
    cases = []
    # SPM tail / 4096 transition / VLM-only padding / multi-warp merge capacity.
    for c in (17, 4096, 4097, 6144, 12288, 19320):
        keys, values = logical_case(c)
        kr, vr = keys.to("rpu"), values.to("rpu")
        for descending in (False, True):
            expected_keys, expected_values = cpu_golden(keys, values, descending)
            actual_keys, actual_values = torch.ops.rpu.test_moe_full_kvsort(kr, vr, descending)
            actual_keys, actual_values = actual_keys.cpu(), actual_values.cpu()
            assert actual_keys.dtype == torch.float16 and actual_values.dtype == torch.int16
            assert torch.equal(actual_keys.view(torch.int16), expected_keys.view(torch.int16))
            assert torch.equal(actual_values, expected_values)
            item = dict(c=c, cores=8, descending=descending,
                        key_bit_exact=True, opaque_u16_exact=True, spm_guards=True)
            print("FULL_KVSORT", json.dumps(item), flush=True)
            cases.append(item)
    # Rejections occur before temporary allocation or kernel submission.
    for bad_keys, bad_values, message in (
        (kr[:, :0].contiguous(), vr[:, :0].contiguous(), "C in [1,131072]"),
        (kr, vr.float(), "raw int16"),
        (kr[:, :-1].contiguous(), vr, "matching contiguous"),
    ):
        try:
            torch.ops.rpu.test_moe_full_kvsort(bad_keys, bad_values, False)
        except RuntimeError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"invalid KVSort input accepted: {message}")
    return cases


def test_full_kvsort():
    import pytest
    if os.environ.get("RPU_HW_AVAILABLE") != "1":
        pytest.skip("requires an exclusively reserved RPU board")
    run_all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import rpu_backend  # noqa: F401
    torch.ops.load_library(os.environ["RPU_CONFORMANCE_SO"])
    args.output.write_text(json.dumps({"scope": "production full KVSort", "cases": run_all()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
