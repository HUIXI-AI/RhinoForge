"""Two TP8 UINT4 independent fixtures through real typed Graph BUILD/REPLAY.

Run only under the repository device reservation with --fixtures pointing to
existing immutable CModel exports. No task path or alternate REF is embedded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from test_grouped_indexed_linear import load_fixtures, _check_payloads


def select_fixtures(manifests):
    selected = [fixture for fixture in load_fixtures(manifests, 8)
                if fixture[0]["mode"] == 8]
    if (len(selected) != 2 or
            {item[0]["family"] for item in selected} != {"indexed", "weighted_scatter_add"}):
        raise ValueError("census probe requires exactly one TP8 UINT4 fixture per family")
    return selected


def run_census(fixtures):
    import torch
    import rpu_backend  # noqa: F401
    import rpu_backend._cpp_ext as ext

    library = Path(os.environ["RPU_CONFORMANCE_SO"]).resolve()
    torch.ops.load_library(str(library))
    runtime = dict(ext.get_runtime_payload_paths())
    refs = _check_payloads(fixtures, runtime)
    reports = []
    for case, arrays, manifest, manifest_sha in fixtures:
        tensors, backing, original = {}, {}, {}
        for key, array in arrays.items():
            if key == "expected":
                continue
            guard = 256 // array.dtype.itemsize
            raw = np.full(array.size + 2 * guard, 7, dtype=array.dtype)
            raw[guard:guard + array.size] = array.reshape(-1)
            original[key] = raw.tobytes()
            backing[key] = torch.from_numpy(raw).to("rpu")
            tensors[key] = backing[key].narrow(0, guard, array.size).view(array.shape)
        shape, expected = case["shape"], arrays["expected"]
        with torch.no_grad():
            outputs, witness = torch.ops.rpu.grouped_indexed_linear_census_test(
                case["family"] == "indexed", tensors["x"], tensors["weight"],
                tensors["counts"], tensors["scale"], tensors["negative_zero"], tensors["ids"],
                tensors.get("coefficients"), tensors.get("initial"), shape["N"], shape["Q"],
                case["partition"] == "col", *case["tile"], case["expert_partitions"])
            actual = outputs.cpu().contiguous().numpy()
        assert actual.shape == (3, *expected.shape) and actual.dtype == np.float16
        assert np.isfinite(actual).all()
        counts = np.asarray(witness, dtype=np.int64).reshape(3, 9)
        assert counts[:, :6].tolist() == [[0, 1, 1, 1, 3, 3], [1, 1, 1, 1, 3, 3], [1, 1, 1, 1, 3, 3]]
        assert counts[:, 6].tolist() == [1, 2, 3]  # actual synchronous SDK batch submissions
        assert counts[0, 7] > 0 and (counts[:, 7] == counts[0, 7]).all()
        calls = []
        for repeat in range(3):
            mismatches = int(np.count_nonzero(actual[repeat].view(np.uint16) != expected.view(np.uint16)))
            assert mismatches == 0, f"{case['case']}: {mismatches} CModel bit mismatches"
            calls.append({"phase": "BUILD" if repeat == 0 else "REPLAY", "bit_mismatch": mismatches,
                          "output_sha256": hashlib.sha256(actual[repeat].tobytes()).hexdigest(),
                          "census": counts[repeat].tolist()})
        for key, owner in backing.items():
            assert owner.cpu().numpy().tobytes() == original[key], f"input/DDR guard changed: {key}"
        report = {"case": case["case"], "family": case["family"], "mode": case["mode"],
                  "shape": shape, "symbol": case["symbol"], "tile": case["tile"],
                  "expert_partitions": case["expert_partitions"], "source_case": case["source_case"],
                  "fixture_manifest": manifest, "fixture_manifest_sha256": manifest_sha,
                  "fixture_sha256": case["sha256"], "calls": calls,
                  "ddr_inputs_and_guards_unchanged": True}
        reports.append(report)
        print(json.dumps({"case": case["case"], "typed_graph_build": 1, "replays": 2, "pass": True}), flush=True)
    return {"completed": True, "runtime": runtime, "refs": refs,
            "conformance_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "reference": "independent same-tile CModel output",
            "scope": "one exact typed kernel per Graph; scoped stable SPM owners; no model/FMB lease claim",
            "witness_columns": ["replay", "nodes", "kernels", "typed_kernels", "operands", "weight_operands",
                                "hw_batch_submits", "build_generation", "execution_ordinal"], "cases": reports}


def test_grouped_indexed_linear_census_fixtures():
    import pytest
    manifests = os.environ.get("RPU_GROUPED_INDEXED_FIXTURES")
    if os.environ.get("RPU_HW_AVAILABLE") != "1" or not manifests:
        pytest.skip("requires reserved RPU and RPU_GROUPED_INDEXED_FIXTURES")
    run_census(select_fixtures(manifests.split(os.pathsep)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    fixtures = select_fixtures(args.fixtures)
    report = {"hardware_executed": False, "preflight_only": True,
              "cases": [case for case, _, _, _ in fixtures]} if args.preflight_only else run_census(fixtures)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
