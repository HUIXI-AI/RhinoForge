"""Exercise cold precision and real table construction without native loading."""
import ast
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _functions(relative, names, scope):
    tree = ast.parse((ROOT / relative).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), relative, "exec"), scope)
    return scope


@pytest.fixture
def resolve(monkeypatch):
    path = ROOT / "python/rpu_backend/runtime/control.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if
                isinstance(node, ast.FunctionDef) and node.name == "rpu_env_bool" or
                isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                {"_ENV_BOOL_TRUE", "_ENV_BOOL_FALSE"} for t in node.targets)]
    env = {"os": os}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), env)
    runtime = ModuleType("rpu_backend.runtime")
    runtime.rpu_env_bool = env["rpu_env_bool"]
    monkeypatch.setitem(sys.modules, "rpu_backend.runtime", runtime)
    for name in ("RPU_RHINOVLA_FULL_W8A16", "RPU_RHINOVLA_EXPERT_W8A16"):
        monkeypatch.delenv(name, raising=False)
    scope = _functions("python/rpu_backend/adapters/rhinovla/runtime.py",
                       ["resolve_rhinovla_quantization"], {})
    return scope["resolve_rhinovla_quantization"]


def test_explicit_expert_modes_do_not_enable_pipeline_quantization(resolve, monkeypatch):
    assert resolve() == (False, False, False)
    assert resolve(expert_w8a16=True, full_expert_w8a16=False) == (True, False, False)
    assert resolve(expert_w8a16=True, full_expert_w8a16=True) == (True, True, False)
    monkeypatch.setenv("RPU_RHINOVLA_FULL_W8A16", "1")
    assert resolve() == (True, True, True)
    with pytest.raises(ValueError, match="conflicts"):
        resolve(expert_w8a16=True, full_expert_w8a16=False)


@pytest.mark.parametrize("bad", [0, 1, "true", [], {}])
def test_flags_are_strict_booleans(resolve, bad):
    with pytest.raises(TypeError):
        resolve(expert_w8a16=bad)
    with pytest.raises(TypeError):
        resolve(full_expert_w8a16=bad)


def test_inconsistent_expert_flags_and_ambient_precision_fail_closed(resolve, monkeypatch):
    with pytest.raises(ValueError, match="requires"):
        resolve(expert_w8a16=False, full_expert_w8a16=True)
    monkeypatch.setenv("RPU_RHINOVLA_EXPERT_W8A16", "on")
    with pytest.raises(ValueError, match="conflicts"):
        resolve(expert_w8a16=False)


@pytest.fixture
def table_builder(monkeypatch):
    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            args = ("cpu", *args[1:])
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    scope = _functions("python/rpu_backend/adapters/rhinovla/convert.py",
                       ["_prepare_full_w8_adarms_tables", "_fold_final_norm_dense"],
                       {"torch": torch})
    return scope["_prepare_full_w8_adarms_tables"]


def _owners():
    expert = NS(_rpu_full_w8_cond_owners={
        "cold_weights": [torch.tensor([[i]], dtype=torch.int8) for i in range(18)],
        "cold_scales": [torch.ones(1, dtype=torch.float16) for _ in range(18)],
        "cold_biases": [torch.zeros(1, dtype=torch.float16) for _ in range(18)],
    })
    norm = NS(norm=NS(weight=torch.ones(1024)), cond=torch.nn.Linear(1024, 3072))
    with torch.no_grad():
        norm.cond.weight.zero_()
        norm.cond.bias.copy_(torch.arange(3072).float() / 1000)
    expert.norm = norm
    return expert


@pytest.mark.parametrize("io_w8", [False, True])
def test_exact_18_and_legacy_19_table_owners_keep_final_precision(table_builder, monkeypatch, io_w8):
    expert = _owners()
    loop = {}
    if io_w8:
        loop = {"final_norm_w_col": torch.tensor([[18]], dtype=torch.int8),
                "io_scales": [torch.ones(1, dtype=torch.float16) for _ in range(6)],
                "final_norm_b": torch.zeros(3072, dtype=torch.float16)}
    calls = []

    def linear(cond, weight, scale, bias):
        assert cond.dtype == torch.float16
        assert weight.dtype == torch.int8 and scale.dtype == torch.float16
        index = len(calls)
        calls.append((weight, scale, bias))
        return torch.full((10, 3072 if index == 18 else 6144), index / 8, dtype=torch.float16)

    monkeypatch.setattr(torch.ops.rpu, "linear_w8a16", linear, raising=False)
    pair, final, weights, scales, biases = table_builder(
        expert, torch.zeros(10, 1024), loop, quantize_final_norm=io_w8)
    assert len(calls) == len(weights) == len(scales) == len(biases) == (19 if io_w8 else 18)
    assert pair.shape == (10, 18, 6144) and final.shape == (10, 3072)
    assert pair.dtype == final.dtype == torch.float16
    assert loop["adarms_cold_owners"] == (weights, scales, biases)
    assert all(a is b for call, owner in zip(calls, zip(weights, scales, biases)) for a, b in zip(call, owner))
    if not io_w8:
        expected = expert.norm.cond.bias.detach().expand(10, -1).clone()
        expected[:, :1024] += 1
        assert torch.equal(final, expected.half())
        assert expert.norm.cond.weight.dtype == torch.float32
        assert "io_scales" not in loop and "final_norm_w_col" not in loop
    else:
        assert torch.equal(final[:, :1024], torch.full((10, 1024), 3.25, dtype=torch.float16))


@pytest.mark.parametrize("damage", ["missing_owner", "extra_owner", "w8_io", "high", "tanh", "steps", "final_dtype", "final_nonfinite"])
def test_expert_only_bad_precision_or_owner_scope_rejected_before_dispatch(table_builder, monkeypatch, damage):
    expert, loop, cond = _owners(), {}, torch.zeros(10, 1024)
    kwargs = {"quantize_final_norm": False}
    if damage == "missing_owner":
        expert._rpu_full_w8_cond_owners["cold_scales"].pop()
    elif damage == "extra_owner":
        expert._rpu_full_w8_cond_owners["cold_weights"].append(torch.ones(1))
    elif damage == "w8_io":
        loop["io_scales"] = [torch.ones(1)] * 6
    elif damage == "high":
        kwargs["high_precision"] = True
    elif damage == "tanh":
        kwargs["precompute_gate_tanh"] = True
    elif damage == "final_dtype":
        expert.norm.cond.weight = torch.nn.Parameter(expert.norm.cond.weight.to(torch.int8), requires_grad=False)
    elif damage == "final_nonfinite":
        with torch.no_grad():
            expert.norm.cond.weight[0, 0] = float("nan")
    else:
        cond = cond[:9]
    monkeypatch.setattr(torch.ops.rpu, "linear_w8a16", lambda *a: pytest.fail("bad scope dispatched"), raising=False)
    with pytest.raises(ValueError):
        table_builder(expert, cond, loop, **kwargs)
    assert "adarms_cold_owners" not in loop
