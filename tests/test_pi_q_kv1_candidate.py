"""Host checks for the exact fused candidate ABI and board numeric fixtures."""
import ast
import importlib.util
from numbers import Integral
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "python/rpu_backend/tests/conformance"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def actual_functions(path, names, globals_):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    module = types.ModuleType(path.stem)
    module.__dict__.update(globals_)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("rows,layer", [(1, 0), (50, 3), (64, 17)])
def test_fixture_uses_current_owner_stripes_and_full_scale_preload(monkeypatch, rows, layer):
    common = actual_functions(CONF / "production_linear_launchers.py", ("activation", "raw"), {"torch": torch})
    weights = actual_functions(ROOT / "python/rpu_backend/runtime/weights.py",
        ("_positive_int", "_validate_swizzle_tensor", "tp_col_swizzle_mc_weight", "transform_linear_weight"),
        {"torch": torch, "Integral": Integral, "NUM_CORES": 8})
    monkeypatch.setitem(sys.modules, "production_linear_launchers", common)
    monkeypatch.setitem(sys.modules, "rpu_backend.runtime.weights", weights)
    monkeypatch.setitem(sys.modules, "rpu_backend.quant.nvfp4_pack",
        load("qkv_pack", ROOT / "python/rpu_backend/quant/nvfp4_pack.py"))
    db = load("production_pi_db_candidates", CONF / "production_pi_db_candidates.py")
    monkeypatch.setitem(sys.modules, "production_pi_db_candidates", db)
    fused = load("qkv_fixture", CONF / "production_pi_q_kv1_candidate.py")
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        value = fused.fixture(f"pi_q_kv1_m{rows}_l{layer}", upload=lambda x: x)
        q = db.nvfp4_fixture("pi_nvfp4_q_db_sspm", rows=rows, upload=lambda x: x)
    finally:
        torch.set_num_threads(previous)
    assert value[:4] == ("pi_q_kv1_pair_sspm", rows, 8, layer)
    assert value[6] == [1, 2, 3] and value[8] is True
    owners = value[4]
    assert [tuple(x.shape) for x in owners] == [(2048, 512), (8, 4, 8, 512),
        (2048, 1024), (2048,), (2048, 1024), (2048,)]
    assert [x.dtype for x in owners] == [torch.uint8, torch.uint8, torch.int8, torch.float16,
                                        torch.int8, torch.float16]
    assert all(x.is_contiguous() and x.storage_offset() == 0 for x in owners)
    assert not torch.equal(owners[1], owners[1].roll(1, 0))
    # Independently unswizzle complete physical K/V owners and select the
    # actual pair's two stripes. Matrix multiplication is the semantic oracle.
    dense = [owners[i].reshape(16, 32, 8, 16, 32).permute(2, 0, 3, 1, 4)
             .reshape(8, 256, 1024).float() for i in (2, 4)]
    for arm in range(2):
        images, expected = value[5][arm], value[7][arm]
        assert len(images) == 6
        assert images[4].shape == (8, 16384) and torch.all(images[4] == 0x5a + arm)
        table = images[5].view(torch.float32)
        assert table.shape == (8, 64) and torch.unique(table).numel() == 3
        torch.testing.assert_close(expected[0], (q[7][arm][0].float() * (table[0, layer] / .25)).half(), rtol=0, atol=0)
        x = images[0].view(torch.float16).reshape(8, rows, 1024).float()
        for p in range(2):
            for core in range(8):
                ids = torch.cat((torch.arange(core * 16, core * 16 + 16),
                                 torch.arange(core * 16 + 128, core * 16 + 144)))
                oracle = (x[core] @ dense[p][core, ids].T * owners[3 + 2 * p].reshape(8, 256)[core, ids]).half()
                torch.testing.assert_close(expected[p + 1][core], oracle, rtol=0, atol=0)
        for i, width in enumerate((256, 32, 32), 1):
            assert images[i].shape == (8, rows * width * 2)
            assert torch.isnan(images[i].view(torch.float16)).all()
            assert torch.isfinite(expected[i - 1]).all()
    assert all(not torch.equal(a, b) for a, b in zip(value[7][0], value[7][1]))
    with pytest.raises(ValueError):
        fused.fixture("pi_q_kv1_m65_l0", upload=lambda x: x)


def test_real_abi_header_preserves_high_halves_and_rejects_invalid_geometry(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    code = r'''
#include "rpu_pi05_q_kv1_abi.h"
#include <cassert>
#include <utility>
int main() {
  using namespace rpu_pi05_q_kv1;
  SpmArguments a{0x41230100,0x42340200,0x43450300,0x44560400,0x45670500,0x46780600,64,17};
  const auto p = registers(a);
  for (const auto& pair : {std::pair{0, a.input}, {32,a.input}, {4,a.query_output},
                         {36,a.key_output},{58,a.value_output},{20,a.query_block_scale},{24,a.query_tensor_scale}})
    assert((uint32_t(p[pair.first]) | uint32_t(p[pair.first+1]) << 16) == pair.second);
  for (int slot : {2,3,34,35,52,53,54,55,56,57,38,39,45,60,61}) assert(p[slot] == 0);
  assert(p[8]==64 && p[40]==64 && p[27]==17 && p[28]==4 && p[64]==6);
  for (auto row : {0,65,65535}) {
    a.rows=row; bool failed=false;
    try { (void)registers(a); } catch (const std::invalid_argument&) { failed=true; }
    assert(failed);
  }
  a.rows=1;
  for (auto layer : {18,65535}) {
    a.layer=layer; bool failed=false;
    try { (void)registers(a); } catch (const std::invalid_argument&) { failed=true; }
    assert(failed);
  }
}
'''
    source = tmp_path / "probe.cpp"
    source.write_text(code)
    binary = tmp_path / "probe"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "src/ops"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_production_cache_oracle_preserves_all_outside_bytes_and_zeroes_pad64():
    fused = load("qkv_cache_oracle", CONF / "production_pi_q_kv1_candidate.py")
    compact = torch.arange(8 * 50 * 32).reshape(8, 50, 32).remainder(17).half() / 16
    poison = torch.arange(256).byte().repeat(32768)
    cos = torch.ones((1024, 128), dtype=torch.float16)
    sin = torch.zeros_like(cos)
    lo, hi = 34 * 65536, 38 * 65536
    for key in (True, False):
        result = fused.production_cache_expected(poison, compact, cos, sin, key=key)
        assert result.dtype == torch.uint8 and result.numel() == 8 * 1024 * 1024
        assert torch.equal(result[:lo], poison[:lo]) and torch.equal(result[hi:], poison[hi:])
        blocks = result[lo:hi].view(torch.float16).reshape(4, 16, 8, 16, 16)
        final = blocks[3]
        assert torch.all(final[:, :, 2:, :] == 0) if key else torch.all(final[:, :, :, 2:] == 0)
        # Verify both halves, core ownership, all replicated head slots and
        # token placement independently of the vectorized oracle layout.
        for token, core, lane, half in ((0, 0, 3, 0), (17, 7, 9, 1), (49, 4, 15, 0)):
            sb, row = divmod(token, 16)
            for head in range(8):
                actual = blocks[sb, core + 8 * half, head, row, lane] if key else blocks[sb, core + 8 * half, head, lane, row]
                assert actual == compact[core, token, lane + half * 16]
    with pytest.raises(ValueError):
        fused.production_cache_expected(poison, compact[:, :1], cos, sin, key=True)
