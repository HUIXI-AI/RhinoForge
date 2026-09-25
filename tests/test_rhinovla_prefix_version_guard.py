"""Copied prefix reuse needs observable mutation versions, without native code."""
import ast
from pathlib import Path

import torch


def _helpers():
    path = (Path(__file__).parents[1] / "python/rpu_backend/adapters/rhinovla/fused.py")
    names = {"_rhino_tensor_version", "_snapshot_rhino_prefix_source",
             "_rhino_prefix_source_matches"}
    tree = ast.parse(path.read_text())
    module = ast.Module(body=[node for node in tree.body
                             if isinstance(node, ast.FunctionDef) and node.name in names],
                        type_ignores=[])
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return (namespace["_snapshot_rhino_prefix_source"],
            namespace["_rhino_prefix_source_matches"])


def test_normal_prefix_reuse_rejects_mutation_and_replacement():
    snapshot, matches = _helpers()
    owner = object()
    key, value = torch.zeros(2), torch.ones(2)
    before = snapshot(owner, 0, [(key, value)])
    assert matches(before, snapshot(owner, 0, [(key, value)]))
    assert not matches(before, snapshot(owner, 0, [(key.clone(), value)]))
    value.add_(1)
    assert not matches(before, snapshot(owner, 0, [(key, value)]))


def test_inference_prefix_identity_cannot_hide_in_place_mutation():
    snapshot, matches = _helpers()
    owner = object()
    with torch.inference_mode():
        key, value = torch.zeros(2), torch.ones(2)
        before = snapshot(owner, 0, [(key, value)])
        # Even an unchanged inference tensor has no reusable version witness.
        assert not matches(before, snapshot(owner, 0, [(key, value)]))
        key.add_(1)
        assert not matches(before, snapshot(owner, 0, [(key, value)]))
