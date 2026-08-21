# SPM allocation

SPM is RhinoForge's per-core scratchpad for fused execution. The runtime plans
against eight 8 MiB SPM instances. Its `SPM_PLANNING_BUDGET` is 99% of usable
SPM, about 7.9 MiB per core. Capacity is a hard constraint, not a tuning hint.
[Allocator contract](../../src/core/rpu_spm_allocator.h)

## Contract

- A `FusedModelBase` subclass declares its complete layout through
  `declare_buffers(const LayoutContext&)`. The declaration must be deterministic
  and must not launch work because several candidate chunk sizes may be tested.
  [Model porting](../../docs/model_porting.md#4-add-a-fused-subsystem-only-when-required)
- `Temp` and `TempPerLayer` storage may be reused after its declared phase;
  `Persistent` and `PersistentPerLayer` storage survives graph replay.
  [Architecture](../../docs/architecture.md#spm-design)
- An alias is valid only when the producer and consumer lifetimes do not
  overlap. The planner and allocator use the same alias-aware layout rules.
  [Architecture](../../docs/architecture.md#spm-design)
- Replay-persistent data belongs in the manifest, with a preload callback when
  initialization is required. Temporary storage is reset at subsystem
  boundaries before compute ownership is released.
  [Architecture](../../docs/architecture.md#spm-design)
- Ordinary operator wrappers receive absolute SPM addresses from `addr()` or
  `layer_addr()`. Offset access is limited to wrappers whose public contract
  explicitly asks for an offset, such as attention and KV-cache insertion.
  [FusedModelBase contract](../../docs/architecture.md#fusedmodelbase-contract)
- A setter that changes weights, layout, or graph-relevant state must finish
  with `invalidate_model_state()`.
  [Model porting](../../docs/model_porting.md#4-add-a-fused-subsystem-only-when-required)

Before admitting a chunk, verify that the exact `LayoutContext` is represented,
the manifest fits, aliases do not overlap, and every replay-persistent value is
declared. [Public C++ contract](../../src/core/fused_model_base.h)

## Sources

- [Architecture: FusedModelBase and SPM](../../docs/architecture.md)
- [Model porting: fused subsystem contract](../../docs/model_porting.md#4-add-a-fused-subsystem-only-when-required)
- [Public `BufferDecl` and `FusedModelBase` declarations](../../src/core/fused_model_base.h)
