# DDR/SPM DMA wrappers

RhinoForge uses explicit DDR-to-SPM broadcast and SPM-to-DDR copy wrappers.
The wrapper choice is an ownership decision because graph replay may reuse the
address captured during the first build. [Architecture](../../docs/architecture.md#dma-ownership)

| Wrapper kind | Use when | Replay rule |
|---|---|---|
| Fixed | The DDR tensor has stable storage for the cache entry's lifetime | The build-time address remains valid |
| Mutable | A caller input or newly allocated output may move between forwards | Update the live address before replay |
| Immediate | The transfer intentionally runs outside graph capture | Execute one synchronous transfer |

These three contracts are declared by the public host wrappers.
[DMA declarations](../../src/core/rpu_kernel_decls.h)

## Rules

- Do not use fixed DMA for caller-owned inputs or fresh outputs unless their
  storage address is guaranteed stable across forwards.
  [Architecture](../../docs/architecture.md#dma-ownership)
- A mutable wrapper changes the replay address; it does not remove the physical
  transfer. The caller also owns coherency for the live buffer.
  [DMA declarations](../../src/core/rpu_kernel_decls.h)
- Python-visible outputs that callers may retain or concatenate need independent
  storage on every forward.
  [FusedModelBase contract](../../docs/architecture.md#fusedmodelbase-contract)
- Copying directly from SPM into the final DDR tensor can remove host staging,
  but it is not zero-copy.
  [Architecture](../../docs/architecture.md#dma-ownership)

## Sources

- [Architecture: DMA ownership](../../docs/architecture.md#dma-ownership)
- [Public DMA host declarations](../../src/core/rpu_kernel_decls.h)
- [Model porting: DMA choice](../../docs/model_porting.md#4-add-a-fused-subsystem-only-when-required)
