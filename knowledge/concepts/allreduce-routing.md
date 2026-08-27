# Automatic residual reduction

Residual all-reduce has one shared production route. Model call sites provide
the validated tensor geometry and residual ownership; the common wrapper emits
the generated eight-core ring schedule. There is no public environment or
per-model selector for choosing another route.

## Contract

- Inputs use the declared FP16 or model-approved mixed-precision boundary.
- The output geometry, source shards, residual storage, and participating
  producer cores are validated before launch.
- A producer using fewer than all cores clears inactive shards before writing
  its active shards, so the same eight-core reduction consumes a complete
  logical input.
- The wrapper selects the generated scheduling variant from the per-core data
  size. Callers do not duplicate that choice.
- Immediate multi-core launch still requires broadcast mode; Graph and batch
  paths apply it centrally.
- BUILD and REPLAY must use the same admitted geometry and live residual
  ownership.

Mixed-precision model boundaries may use a dedicated checked wrapper, but that
does not restore a global route switch or authorize another model profile.

## Porting consequence

Do not copy retired route, chunk, or two-stage selectors into a TOML or adapter.
A new producer should prepare its shards through the shared helper, call the
common residual wrapper, and verify same-semantic parity plus stable Graph
lifecycle at the maximum admitted shape.

## Sources

- [Architecture](../../docs/architecture.md#graph-and-batch-execution)
- [Runtime configuration](../../docs/runtime_config.md)
- [Runtime wrapper](../../src/core/rpu_runtime_extras.cpp)
- [Shared kernel cache declarations](../../src/core/rpu_kernel_cache.h)
- [Multi-core broadcast](multicore-broadcast.md)
- [GraphCache capture](graphcache-capture.md)
