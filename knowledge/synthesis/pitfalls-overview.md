# Porting pitfalls overview

Use this table before adding model-specific workarounds. Each risk already has
a shared contract in RhinoForge.

| Risk | First check | Public contract |
|---|---|---|
| A weight is transformed twice | Put companion-owned projections in `skip_names`; transform one instance once | [Weight swizzle](../concepts/weight-swizzle.md) |
| Replay-persistent SPM state uses temporary storage | Move it into the `BufferDecl` manifest with the correct lifetime | [SPM allocation](../concepts/spm-allocation.md) |
| An operator receives an offset where it expects an absolute SPM address | Use `addr()`/`layer_addr()` unless the wrapper explicitly accepts offsets | [FusedModelBase contract](../../docs/architecture.md#fusedmodelbase-contract) |
| Retained Python outputs share one backing tensor | Allocate independent output storage per forward | [FusedModelBase contract](../../docs/architecture.md#fusedmodelbase-contract) |
| Fixed DMA captures a tensor that moves between forwards | Use a mutable wrapper and update the live address | [DMA ownership](../concepts/ddr-dma-wrappers.md) |
| A graph signature omits a semantic input | Add it to the signature or update it through the mutable-data contract | [Graph invariants](../concepts/graphcache-capture.md) |
| A setter changes layout or bound weights without invalidation | End the setter with `invalidate_model_state()` | [Fused subsystem guide](../../docs/model_porting.md#4-add-a-fused-subsystem-only-when-required) |
| A cold execution option changes after model construction | Create a new model handle with the final `rpu_execution` mapping | [Runtime controls](../../docs/runtime_config.md#cold-per-handle-rpu_execution) |

When the first check fails to explain a numerical mismatch, compare the CPU
reference and RPU result at the smallest public model boundary, then widen one
phase at a time. Do not claim support from an end-to-end run alone.
[Verification gates](../../docs/model_porting.md#8-verification-gates)

## Sources

- [Porting pitfalls](../../docs/pitfalls.md)
- [Architecture](../../docs/architecture.md)
- [Model porting](../../docs/model_porting.md)
- [Runtime configuration](../../docs/runtime_config.md)
- [Public weight helpers](../../python/rpu_backend/runtime/weights.py)
