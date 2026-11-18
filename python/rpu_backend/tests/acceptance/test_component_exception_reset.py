"""SPM-reset postcondition AST-structural test for component adapters.

The legacy ComponentBase try/finally exception contract was retired with the
deletion of core/components/; wrapping each rpu_*_forward in try/finally
remains outside this structural contract.

This test is **structural** — it parses the source of the lifted
`patch_*_model_for_rpu_all_layers_once` functions and asserts the
subsystem-boundary postcondition: each `patch_*` function's nested
`rpu_*_forward` closure MUST
call `torch.ops.rpu.<fused_op>` AND `torch.ops.rpu.spm_alloc_reset_temporary`
with the reset call lineno strictly greater than the fused-op call lineno
(mirrors `scripts/check_component_spm_contract.py` so the unit-acceptance
test stays in lock-step with the CI verifier).

The test does NOT invoke the patch functions at runtime — exercising the
emitted closures end-to-end requires real `transformers` + RPU hardware,
which the `tests/model/test_*_all_layers_once.py` hardware suites cover.
"""
from __future__ import annotations
import ast
import importlib
import inspect
import pytest


# (module_path, patch_fn_name, fused_op_attr) — fused_op_attr matches the
# `torch.ops.rpu.<suffix>(...)` Attribute chain emitted by the lifted
# closure. Mirrors `scripts/check_component_spm_contract.py:TARGET_FILES`.
_LIFTED_PATCHES = [
    ("rpu_backend.adapters.pi05.gemma",  "patch_gemma_model_for_rpu_all_layers_once",  "gemma_forward"),
    ("rpu_backend.adapters.pi05.adarms", "patch_adarms_model_for_rpu_all_layers_once", "adarms_forward"),
    ("rpu_backend.adapters.siglip",      "patch_siglip_model_for_rpu_all_layers_once", "siglip_forward"),
    ("rpu_backend.adapters.paligemma.gemma2", "patch_gemma2_model_for_rpu_all_layers_once", "gemma2_forward"),
]


def _is_torch_ops_rpu_call(node: ast.AST, expected_attr: str) -> bool:
    """True iff `node` is a Call whose `.func` is the Attribute chain
    `torch.ops.rpu.<expected_attr>`. Mirrors the verifier in
    `scripts/check_component_spm_contract.py`."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    a3 = node.func
    if a3.attr != expected_attr:
        return False
    a2 = a3.value
    if not isinstance(a2, ast.Attribute) or a2.attr != "rpu":
        return False
    a1 = a2.value
    if not isinstance(a1, ast.Attribute) or a1.attr != "ops":
        return False
    a0 = a1.value
    return isinstance(a0, ast.Name) and a0.id == "torch"


def _first_call_lineno(fn: ast.FunctionDef, expected_attr: str) -> int | None:
    for node in ast.walk(fn):
        if _is_torch_ops_rpu_call(node, expected_attr):
            return node.lineno
    return None


def _first_op_reference_lineno(fn: ast.FunctionDef, expected_attr: str) -> int | None:
    """Return the call line or an aliased operator reference line."""
    direct = _first_call_lineno(fn, expected_attr)
    if direct is not None:
        return direct
    for node in ast.walk(fn):
        if not isinstance(node, ast.Attribute) or node.attr != expected_attr:
            continue
        parent = node.value
        if not isinstance(parent, ast.Attribute) or parent.attr != "rpu":
            continue
        parent = parent.value
        if not isinstance(parent, ast.Attribute) or parent.attr != "ops":
            continue
        if isinstance(parent.value, ast.Name) and parent.value.id == "torch":
            return node.lineno
    return None


@pytest.mark.parametrize("module_path,patch_fn_name,fused_op_attr", _LIFTED_PATCHES)
def test_patch_function_body_contains_spm_reset_call(module_path, patch_fn_name, fused_op_attr):
    """Each lifted patch function MUST
    define at least one nested `rpu_*_forward` whose body calls both
    `torch.ops.rpu.<fused_op>` AND `torch.ops.rpu.spm_alloc_reset_temporary`,
    with the reset lineno strictly greater than the fused-op lineno.
    Structural-only check; no runtime invocation.
    """
    mod = importlib.import_module(module_path)
    src = inspect.getsource(mod)
    tree = ast.parse(src)
    patch_fn = next(
        (n for n in ast.iter_child_nodes(tree)
         if isinstance(n, ast.FunctionDef) and n.name == patch_fn_name),
        None,
    )
    assert patch_fn is not None, (
        f"{patch_fn_name} not found at top level of {module_path}"
    )

    # Find every nested FunctionDef named rpu_*_forward inside the patch fn.
    # The lifted v3 closures are bound via types.MethodType, so they live as
    # nested FunctionDef nodes (not as class methods).
    nested = [
        n for n in ast.walk(patch_fn)
        if isinstance(n, ast.FunctionDef)
        and n is not patch_fn
        and n.name.startswith("rpu_")
        and n.name.endswith("forward")
    ]
    assert nested, (
        f"{module_path}.{patch_fn_name} defines no nested `rpu_*_forward` "
        f"closure; v3 contract requires at least one (lifted from "
        f"pi05_all_layers_once.py per B4)"
    )

    for inner in nested:
        contract_body = inner
        fused_lineno = _first_op_reference_lineno(contract_body, fused_op_attr)
        reset_lineno = _first_call_lineno(contract_body, "spm_alloc_reset_temporary")
        if fused_lineno is None and module_path == "rpu_backend.adapters.pi05.gemma":
            # The Gemma model and KV-only entry points share one lifted body.
            # Keep the structural check on that body so the forwarding
            # closures cannot bypass the postcondition.
            helper = next(
                (n for n in ast.walk(patch_fn)
                 if isinstance(n, ast.FunctionDef) and n.name == "_run_gemma_prefill"),
                None,
            )
            assert helper is not None, "Gemma forwarding closure has no shared prefill body"
            assert any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_run_gemma_prefill"
                for node in ast.walk(inner)
            ), "Gemma forwarding closure does not call the shared prefill body"
            contract_body = helper
            fused_lineno = _first_op_reference_lineno(contract_body, fused_op_attr)
            reset_lineno = _first_call_lineno(contract_body, "spm_alloc_reset_temporary")
        assert fused_lineno is not None, (
            f"{module_path}.{patch_fn_name}::{inner.name} body does not call "
            f"`torch.ops.rpu.{fused_op_attr}(...)` — fused-op emission missing"
        )
        assert reset_lineno is not None, (
            f"{module_path}.{patch_fn_name}::{inner.name} body does not call "
            f"`torch.ops.rpu.spm_alloc_reset_temporary()` — v3 SPM-reset "
            f"postcondition violated"
        )
        assert reset_lineno > fused_lineno, (
            f"{module_path}.{patch_fn_name}::{inner.name} calls "
            f"spm_alloc_reset_temporary() at line {reset_lineno} which is "
            f"NOT after the fused `torch.ops.rpu.{fused_op_attr}` call at "
            f"line {fused_lineno} — v3 subsystem-boundary postcondition "
            f"violated (reset must fire AFTER the forward op)"
        )
