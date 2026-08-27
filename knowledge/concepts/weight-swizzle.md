# Weight swizzle

RhinoForge transforms supported weights in place into the layout expected by
RPU operators. The public helpers are `get_linear_partition`, `swizzle_linear`,
and `swizzle_model_inplace`. [Weight-layout API](../../docs/api_reference.md#weight-layout-api-for-adapter-authors)

## Rules

- Validate the exact model profile before loading all weights or starting any
  in-place transformation.
  [Model porting](../../docs/model_porting.md#1-assess-the-capability-gap)
- Transform one model instance exactly once. The operation is irreversible;
  do not transform it again or move the transformed instance back to CPU.
  [Architecture](../../docs/architecture.md#python-api-and-adapters)
- A specialized `Linear` with its own authoritative conversion path belongs in
  `skip_names`. The companion path, not the generic converter, owns its RPU
  layout. [Adapter template](../../python/rpu_backend/adapters/_template/adapter.py)
- The converter records the chosen partition on each transformed module so
  eager dispatch cannot silently assume a different layout.
  [Weight helpers](../../python/rpu_backend/runtime/weights.py)
- If a transformation fails after it starts, discard the instance and reload a
  clean model; do not retry a partially transformed object.
  [Model porting](../../docs/model_porting.md#3-reuse-the-causal-decoder-when-it-matches)

The common failure pattern is applying both the generic conversion and a
component-specific conversion to the same projection. Prevent it at the shared
`skip_names` boundary instead of patching each call site.
[Weight helpers](../../python/rpu_backend/runtime/weights.py)

## Sources

- [API reference: weight-layout API](../../docs/api_reference.md#weight-layout-api-for-adapter-authors)
- [Public weight helpers](../../python/rpu_backend/runtime/weights.py)
- [Annotated adapter template](../../python/rpu_backend/adapters/_template/adapter.py)
- [Model porting: conversion order](../../docs/model_porting.md#3-reuse-the-causal-decoder-when-it-matches)
