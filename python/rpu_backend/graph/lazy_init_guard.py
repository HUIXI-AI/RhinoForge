"""Lazy-init guard — verify an RPU-patched module finished its install.

Every `patch_*_for_rpu()` calls this guard on the ordinary eager path. It
catches forward-time lazy initialization (`if not hasattr(self, "_rpu_xxx")`)
at patch time instead of letting state appear later. Dynamo can bake that
attribute test into a guard and repeatedly recompile, but the install contract
applies with or without Dynamo: finish module initialization before returning.

Guard surface:
- `DynamoUnsafeLazyStateError` — adapter 漏 eager init 时的 sentinel exception
- `_verify_lazy_init(model)` — preflight 实现,walk model.modules() 验所有
  `_rpu_lazy_init_checked=True` 子模块的 `_rpu_required_attrs` 全存在
- `freeze_for_dynamo(model)` — 公共 API,patch_* 外部用户手动 freeze 入口
- `_verify_lazy_init_best_effort(gm, example_inputs)` — RpuDynamoBackend 入口
  hook,扫 gm._modules best-effort

上述列表定义了该 guard 的完整约束。
"""
from __future__ import annotations

import torch.nn as nn


class DynamoUnsafeLazyStateError(RuntimeError):
    """Raised when an RPU-patched module has lazy state not initialized
    before Dynamo trace.

    Root cause: forward-time `if not hasattr(self, '_rpu_xxx'): ...` lazy init
    makes Dynamo bake `not hasattr == True` into the guard cache;Call 2 sees
    attr present → guard fail → recompile → backend admission装新 entry →
    旧 RpuKernelGraph entry 永不命中(replay miss)。修法:把 init 挪到
    patch_*_for_rpu() / to_rpu() 阶段。
    """


# Adapters stamp one of these marker attributes to opt into the preflight.
# Both spellings are accepted for compatibility with out-of-tree adapters.
# The compatibility name is also allowed by `INTERNAL_HW_ATTRS_TRANSITIONAL`
# so the hardware-attribute validator permits adapters to set it.
_LAZY_INIT_MARKERS = ("_rpu_lazy_init_checked", "_rpu_dynamo_safe")


def _verify_lazy_init(model: nn.Module) -> None:
    """Walk model.modules(), verify every `_rpu_lazy_init_checked=True` submodule
    has all attrs listed in its `_rpu_required_attrs`. Raise on missing.

    The legacy marker `_rpu_dynamo_safe` is accepted as an alias.

    Called automatically at the end of each `patch_*_for_rpu()` function and
    also at RpuDynamoBackend entry (best-effort).
    """
    failures: list[tuple[str, list[str]]] = []
    for sub in model.modules():
        if not any(getattr(sub, m, False) for m in _LAZY_INIT_MARKERS):
            continue
        required = getattr(sub, '_rpu_required_attrs', ())
        missing = [a for a in required if not hasattr(sub, a)]
        if missing:
            failures.append((type(sub).__name__, missing))
    if failures:
        msg = "\n".join(
            f"  {name}: missing {miss}" for name, miss in failures
        )
        raise DynamoUnsafeLazyStateError(
            "RPU module(s) have lazy state not initialized before Dynamo trace.\n"
            "Initialize in patch_*_for_rpu() / to_rpu(), not in forward.\n"
            "Forward-time lazy init causes Dynamo `hasattr` guard hazards →\n"
            "Call 2 recompile → backend cache miss → replay never hits.\n"
            f"{msg}"
        )


def verify_lazy_init(model: nn.Module) -> None:
    """Public API. Use when not going through the standard patch_*_for_rpu path
    (hand-rolled forward, non-adapter integration).

    Equivalent to the auto-call at the end of each patch_*_for_rpu(), exposed
    here so external callers who skip the adapter convention can still trigger
    the preflight check explicitly.
    """
    _verify_lazy_init(model)


# ── Backwards-compatible aliases ─────────────────────────────────────────────
# `freeze_for_dynamo` and `DynamoUnsafeLazyStateError` are exported from the
# packaged runtime, so the old spellings keep working. They are misnomers (this has
# never been Dynamo-specific); prefer the names above in new code.
freeze_for_dynamo = verify_lazy_init
UninitializedLazyStateError = DynamoUnsafeLazyStateError


def _verify_lazy_init_best_effort(gm, example_inputs) -> None:
    """Backend-entry hook called from `RpuDynamoBackend.__call__`.

    Walks `gm._modules` and runs `_verify_lazy_init` on each submodule.
    Closure-only references (modules captured via Python closure into the
    compiled callable that didn't end up as gm submodules) are not visible
    here — that's the "best-effort" part. Standard adapter path stamps marker
    at patch time so this is mostly belt-and-suspenders.

    Args:
        gm: torch.fx.GraphModule (or anything with ._modules dict)
        example_inputs: currently ignored because modules linked through tensor
                        metadata are not cheaply reachable without private
                        PyTorch introspection
    """
    modules_dict = getattr(gm, '_modules', None)
    if not modules_dict:
        return
    seen = set()
    for name, sub in modules_dict.items():
        if sub is None:
            continue
        if id(sub) in seen:
            continue
        seen.add(id(sub))
        _verify_lazy_init(sub)
