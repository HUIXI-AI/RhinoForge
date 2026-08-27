# Live model and policy lifecycle

Fused RhinoForge execution permits one live RPU-resident model or policy owner
per process. Installation changes weights in place, creates model handles and
Graph caches, and binds cold execution settings; it is a transaction, not a
reversible device copy.

## Installation transaction

1. Validate the complete profile and claim process ownership.
2. Mark irreversible transformation as started.
3. Transform each weight exactly once and move the admitted state to RPU.
4. Create handles, caches, Graph ownership, and cold execution configuration.
5. Mark the instance ready only after every step succeeds.

A completed second `.to("rpu")` may be an idempotent no-op for the same
instance. A partial failure must release claimed runtime resources but must not
retry transformation on the mutated instance. Reload a clean CPU model.

## Teardown

The owning facade or adapter releases its model handles and Graph caches. It
does not undo process-cold registration or restore transformed weights to their
original layout. Start a fresh process to switch a fused model, plugin set,
runtime asset set, or import/native-bound profile.

## Sources

- [Architecture: Python API and adapters](../../docs/architecture.md#python-api-and-adapters)
- [Model porting installation order](../../docs/model_porting.md#3-reuse-the-causal-decoder-when-it-matches)
- [Runtime profiles](runtime-profiles.md)
- [Weight swizzle](weight-swizzle.md)
- [Public policy APIs](../../docs/api_reference.md#policy-apis)
