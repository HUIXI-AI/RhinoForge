"""Frontend signature keys must match the native GraphSignature identity."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
ADMISSION = ROOT / "python" / "rpu_backend" / "graph" / "admission.py"


def _load_admission():
    name = "rpu_backend_graph_admission_contract"
    spec = importlib.util.spec_from_file_location(name, ADMISSION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _signature(**overrides):
    fields = {
        "op_id": 11,
        "shapes": [1, 16, 128],
        "dyn_dims": [320, 160, -1, 7],
        "dtypes": [5, 5],
        "flags": 3,
        "branch_key": 5,
        "segment_key": 9,
    }
    fields.update(overrides)
    return SimpleNamespace(_impl=SimpleNamespace(**fields))


def test_signature_key_includes_complete_native_identity():
    planner = _load_admission().SignaturePlanner
    sig = _signature()
    assert planner.signature_key(sig) == (
        11,
        (1, 16, 128),
        (320, 160, -1, 7),
        (5, 5),
        3,
        5,
        9,
    )


def test_different_dyn_dims_cannot_collide_in_frontend_key():
    planner = _load_admission().SignaturePlanner
    assert planner.signature_key(_signature(dyn_dims=[320, 160])) != (
        planner.signature_key(_signature(dyn_dims=[320, 161]))
    )


def test_build_plan_key_matches_signature_identity():
    module = _load_admission()
    first = module.build_plan_for_call(_signature(dyn_dims=[320, 160]))
    second = module.build_plan_for_call(_signature(dyn_dims=[320, 161]))
    assert first.entry_key != second.entry_key
    assert first.guard_key != second.guard_key
