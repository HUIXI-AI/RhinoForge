# Backend registration boundary

RhinoForge keeps dispatcher schemas, tensor/device integration, launch-cache
bindings, and Python bindings in reviewable source fragments. Those fragments
are included by the main backend source and compiled as one C++ translation
unit; they are not independent libraries or initialization owners.

## Why this matters

Registration and process-lifetime runtime state have an explicit startup and
shutdown order. Moving a fragment into a separately compiled source file can
change static initialization and teardown ordering even when compilation and
linking succeed. A refactor therefore needs lifecycle and Graph execution
evidence, not only a build check.

## Contributor rule

- Add a dispatcher schema and `PrivateUse1` implementation through
  `src/core/rpu_dispatch_registrations.inc`.
- Keep model math and launch validation in the owning `src/ops/` or
  `src/fused/` file.
- Do not compile the registration fragments separately.
- After a registration-boundary change, verify clean import/shutdown and at
  least one retained BUILD-to-REPLAY path.

## Sources

- [Model porting: add an operation](../../docs/model_porting.md#5-add-or-reuse-a-public-operator)
- [Source layout](../../docs/architecture.md#source-layout)
- [Main backend source](../../src/core/rpu_backend.cpp)
- [Dispatcher registrations](../../src/core/rpu_dispatch_registrations.inc)
