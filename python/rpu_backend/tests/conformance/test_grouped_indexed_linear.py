"""Production indexed/scatter launchers against explicit frozen CModel fixtures.

Run under the existing board lock with --fixtures manifest.json (repeatable),
RPU_CONFORMANCE_SO and the formal REF pair. Defaults to TP8. NPZ inputs and
same-tile CModel output remain external evidence; no task path is embedded here.
--preflight-only validates fixtures without importing the native backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy

import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fixtures(manifests, cores=8):
    fixtures = []
    names = set()
    for manifest_path in map(Path, manifests):
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("completed") or not manifest.get("cases"):
            raise ValueError(f"incomplete fixture export: {manifest_path}")
        for case in manifest["cases"]:
            if case["shape"]["TP"] != cores:
                continue
            if case["case"] in names:
                raise ValueError(f"duplicate fixture: {case['case']}")
            names.add(case["case"])
            path = manifest_path.parent / case["file"]
            if _sha(path) != case["sha256"]:
                raise ValueError(f"fixture identity differs: {path}")
            if case["family"] not in ("indexed", "weighted_scatter_add") or case["mode"] not in (1, 2, 5, 8):
                raise ValueError(f"unsupported family/dtype: {case['case']}")
            if not case.get("source_case") or not case.get("source_files"):
                raise ValueError(f"missing independent reference provenance: {case['case']}")
            if not case.get("cmodel_spm_guards") or not all(g["exact"] for g in case["cmodel_spm_guards"]):
                raise ValueError(f"source CModel guard failure: {case['case']}")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            if set(arrays) != set(case["arrays"]):
                raise ValueError(f"fixture array inventory differs: {path}")
            for key, array in arrays.items():
                identity = case["arrays"][key]
                if (list(array.shape) != identity["shape"] or str(array.dtype) != identity["dtype"] or
                        hashlib.sha256(array.tobytes()).hexdigest() != identity["sha256"]):
                    raise ValueError(f"fixture array identity differs: {path}:{key}")
            shape = case["shape"]
            rows = shape["R"] if case["family"] == "indexed" else shape["T"]
            expected = arrays["expected"]
            if (expected.shape != (cores, rows, shape["N"]) or expected.dtype != np.float16 or
                    not np.isfinite(expected).all()):
                raise ValueError(f"invalid independent output: {path}")
            fixtures.append((case, arrays, str(manifest_path), _sha(manifest_path)))
    if not fixtures:
        raise ValueError(f"no fixtures selected for TP{cores}")
    return fixtures


def _check_payloads(fixtures, runtime):
    # The parser is a pure file tool; no alternate source library supplies bytes.
    parser = Path(__file__).parents[2] / "debug/refdiff_kernels.py"
    parse_ref = runpy.run_path(str(parser))["parse_ref"]
    roles = runtime["oplib_roles"]
    refs = {role: parse_ref(path) for role, path in roles.items()}
    for case, _, _, _ in fixtures:
        symbol, instruction = case["symbol"], case["instruction"]
        if symbol not in refs["main"] or symbol in refs["expansion"]:
            raise AssertionError(f"expected exactly one BASE generic payload: {symbol}")
        if refs["main"][symbol] != (instruction["bytes"], instruction["sha1"]):
            raise AssertionError(f"BASE payload differs from independent CModel: {symbol}")
    return {role: {"path": path, "sha256": _sha(path)} for role, path in roles.items()}


def run_fixtures(fixtures, *, current_auto=False):
    import torch
    import rpu_backend  # noqa: F401
    import rpu_backend._cpp_ext as ext

    conformance = Path(os.environ.get("RPU_CONFORMANCE_SO", Path(__file__).parent / "_cpp/librpu_conformance.so"))
    torch.ops.load_library(str(conformance.resolve()))
    runtime = dict(ext.get_runtime_payload_paths())
    ref_identity = _check_payloads(fixtures, runtime)
    reports = []
    for case, arrays, manifest, manifest_sha in fixtures:
        shape, expected = case["shape"], arrays["expected"]
        if current_auto:
            selected, symbol = torch.ops.rpu.grouped_indexed_linear_current_tile_test(
                case["family"] == "indexed", shape["T"], shape["R"], shape["E"],
                shape["N"], shape["K"], shape["Q"], shape["TP"], case["mode"],
                case["partition"] == "col")
            if (list(selected) != [*case["tile"], case["expert_partitions"]] or
                    symbol != case["symbol"]):
                raise ValueError(f"{case['case']}: current auto needs its own same-tile CModel fixture")
        tensors, backing, host = {}, {}, {}
        for key, array in arrays.items():
            if key == "expected":
                continue
            guard = 256 // array.dtype.itemsize
            raw = np.full(array.size + 2 * guard, 7, dtype=array.dtype)
            raw[guard:guard + array.size] = array.reshape(-1)
            host[key] = raw
            backing[key] = torch.from_numpy(raw.copy()).to("rpu")
            tensors[key] = backing[key].narrow(0, guard, array.size).view(array.shape)
        record = {"case": case["case"], "family": case["family"], "mode": case["mode"],
                  "shape": shape, "tile": case["tile"], "expert_partitions": case["expert_partitions"],
                  "symbol": case["symbol"], "source_case": case["source_case"],
                  "fixture_manifest": manifest, "fixture_manifest_sha256": manifest_sha,
                  "fixture_sha256": case["sha256"], "current_auto": current_auto, "calls": []}
        outputs = []
        for repeat in range(3):
            common = (tensors["x"], tensors["weight"], tensors["counts"], tensors.get("scale"),
                      tensors.get("negative_zero"), tensors["ids"])
            m, n, k = case["tile"]
            with torch.no_grad():
                if case["family"] == "indexed":
                    result = torch.ops.rpu.grouped_parallel_linear_indexed_test(
                        *common, shape["N"], case["mode"], shape["Q"], case["partition"] == "col",
                        m, n, k, case["expert_partitions"])
                else:
                    result = torch.ops.rpu.grouped_parallel_linear_weighted_scatter_add_test(
                        *common, tensors["coefficients"], tensors["initial"], case["mode"],
                        shape["Q"], case["partition"] == "col", m, n, k)
                actual = result.cpu().contiguous().numpy()
            assert actual.dtype == np.float16 and actual.shape == expected.shape
            assert np.isfinite(actual).all()
            mismatch = int(np.count_nonzero(actual.view(np.uint16) != expected.view(np.uint16)))
            # Exact replay of identical instructions/inputs/forced tile; this
            # is a CModel parity contract, not a floating-point accuracy budget.
            assert mismatch == 0, f"{case['case']}: {mismatch} CModel bit mismatches"
            assert all(result.data_ptr() != previous.data_ptr() for previous in outputs)
            outputs.append(result)
            for key, owner in backing.items():
                assert owner.cpu().numpy().tobytes() == host[key].tobytes(), f"input/DDR guard changed: {key}"
            record["calls"].append({"repeat": repeat, "bit_mismatch": mismatch,
                                    "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest()})
        assert all(output.cpu().numpy().tobytes() == expected.tobytes() for output in outputs)
        record.update(finite=True, ddr_inputs_and_guards_unchanged=True, retained_outputs_unchanged=True)
        reports.append(record)
        print(json.dumps({"case": case["case"], "mode": case["mode"], "calls": 3, "pass": True}), flush=True)
    return {"completed": True, "reference": "independent same-tile CModel output",
            "scope": "native DDR guards; source CModel SPM guards; no native SPM canary claim",
            "runtime": runtime, "refs": ref_identity, "cases": reports}


def test_grouped_indexed_linear_fixtures():
    import pytest
    manifests = os.environ.get("RPU_GROUPED_INDEXED_FIXTURES")
    if os.environ.get("RPU_HW_AVAILABLE") != "1" or not manifests:
        pytest.skip("requires reserved RPU and RPU_GROUPED_INDEXED_FIXTURES manifest paths")
    run_fixtures(load_fixtures(manifests.split(os.pathsep)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, action="append", required=True)
    parser.add_argument("--cores", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--current-auto", action="store_true",
                        help="verify explicit current-main selection matches each independent fixture")
    args = parser.parse_args()
    fixtures = load_fixtures(args.fixtures, args.cores)
    report = {"preflight_only": True, "hardware_executed": False,
              "cases": [{"case": c["case"], "family": c["family"], "mode": c["mode"],
                         "shape": c["shape"], "manifest_sha256": sha}
                        for c, _, _, sha in fixtures]} if args.preflight_only else run_fixtures(
                            fixtures, current_auto=args.current_auto)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
