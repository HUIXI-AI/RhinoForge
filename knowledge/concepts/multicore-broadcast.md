# Multi-core broadcast

RhinoForge targets eight cores, but each launched operation retains its own
participating-core contract. A one-core operation and an eight-core operation
may coexist in the same batch. [Architecture](../../docs/architecture.md#graph-and-batch-execution)

## Launch rule

- An immediate multi-core launch must enable broadcast mode so every
  participating core receives the host parameters.
  [Architecture](../../docs/architecture.md#graph-and-batch-execution)
- Batch and `GraphCache` paths apply broadcast centrally; use those paths for a
  production adapter instead of duplicating launch setup.
  [Model porting](../../docs/model_porting.md#4-add-a-fused-subsystem-only-when-required)
- Do not infer core count from the surrounding model. Pass the count required
  by the selected operator and keep weight partitioning consistent with it.
  [Weight helpers](../../python/rpu_backend/runtime/weights.py)
- Mixed-core batching is valid because core selection belongs to each node, not
  to the batch as a whole.
  [Architecture](../../docs/architecture.md#graph-and-batch-execution)

Verification still requires numerical comparison and the normal graph
BUILD-to-REPLAY lifecycle gate; a successful launch alone is not a model
support result. [Model porting](../../docs/model_porting.md#8-verification-gates)

## Sources

- [Architecture: graph and batch execution](../../docs/architecture.md#graph-and-batch-execution)
- [Model porting: fused subsystem contract](../../docs/model_porting.md#4-add-a-fused-subsystem-only-when-required)
- [Public multi-core launch call sites](../../src/core/rpu_runtime_extras.cpp)
- [Public weight partition helpers](../../python/rpu_backend/runtime/weights.py)
