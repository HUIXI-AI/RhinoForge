"""Exercise the actual four SSPM board fixtures without loading the device module."""
import ast
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFORMANCE = ROOT / "python/rpu_backend/tests/conformance"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("projection,per_core_bytes", [
    ("q", 16384), ("o", 16384), ("down", 32768), ("geglu", 32768),
])
def test_board_fixture_keeps_math_and_installs_distinct_full_core_scales(
        monkeypatch, projection, per_core_bytes):
    pack = load_module("sspm_fixture_pack", ROOT / "python/rpu_backend/quant/nvfp4_pack.py")
    monkeypatch.setitem(sys.modules, "rpu_backend.quant.nvfp4_pack", pack)
    # Execute the real input and byte-view helpers, avoiding unrelated backend
    # imports and without replacing either helper with a test implementation.
    source = ast.parse((CONFORMANCE / "production_linear_launchers.py").read_text())
    nodes = [node for node in source.body
             if isinstance(node, ast.FunctionDef) and node.name in ("activation", "raw")]
    assert len(nodes) == 2
    common = types.ModuleType("production_linear_launchers")
    common.torch = torch
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CONFORMANCE), "exec"), common.__dict__)
    monkeypatch.setitem(sys.modules, "production_linear_launchers", common)
    candidates = load_module("sspm_fixture_candidates", CONFORMANCE / "production_pi_db_candidates.py")

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        actual = candidates.nvfp4_fixture(f"pi_nvfp4_{projection}_db_sspm", upload=lambda x: x)
        prior = candidates.nvfp4_fixture(f"pi_nvfp4_{projection}_db_tscale", upload=lambda x: x)
    finally:
        torch.set_num_threads(threads)
    assert f"pi_nvfp4_{projection}_db_sspm" in candidates.CASES
    assert actual[0] == f"pi_nvfp4_{'geglu' if projection == 'geglu' else 'linear'}_db_sspm"
    assert actual[1:4] == prior[1:4] and actual[6] == prior[6] == [1]
    assert actual[8] == prior[8] == (projection != "geglu")
    count = 2 if projection == "geglu" else 1
    owners, previous_owners = actual[4], prior[4]
    assert len(owners) == len(previous_owners) == count * 2
    for i in range(count):
        torch.testing.assert_close(owners[i], previous_owners[i], rtol=0, atol=0)
        scale = owners[count + i]
        torch.testing.assert_close(scale, previous_owners[count + i].permute(2, 0, 1, 3), rtol=0, atol=0)
        assert scale.dtype == torch.uint8 and scale.shape[0] == 8
        assert scale.is_contiguous() and scale.storage_offset() == 0
        assert scale.numel() == 8 * per_core_bytes
        # Wrong core selection and Gate/Up confusion must be observable.
        assert torch.count_nonzero(scale != scale.roll(1, 0)).item() > scale.numel() // 2
    if count == 2:
        assert torch.count_nonzero(owners[2] != owners[3]).item() > owners[2].numel() // 2
    for arm in range(2):
        image, old_image = actual[5][arm], prior[5][arm]
        assert len(image) == len(old_image) + count
        for before, after in zip(old_image, image):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        for poison in image[len(old_image):]:
            assert poison.shape == (8, per_core_bytes) and poison.dtype == torch.uint8
            assert torch.all(poison == 0x5a + arm)
        torch.testing.assert_close(actual[7][arm][0], prior[7][arm][0], rtol=0, atol=0)
    assert not torch.equal(actual[7][0][0], actual[7][1][0])
