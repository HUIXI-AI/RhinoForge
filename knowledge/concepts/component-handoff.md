# Multi-component ownership and handoff

Vision-language and action policies move execution through several model
components. Each component owns its handle, Graph cache, temporary SPM, live
inputs, and returned outputs. A handoff is therefore a lifetime boundary, not
just the next function call.

## Ownership contract

- Bind one exact runtime profile before constructing any component. A component
  must not silently consume execution settings intended for another stage.
- Give each model-owned component an explicit Graph cache. Do not use a
  process-global cache to bridge vision, text, or action ownership.
- Declare state that must survive replay as persistent. Reset temporary SPM and
  release the current compute owner before the next subsystem claims it.
- Use fixed DMA only for storage that remains stable for the complete replay
  lifetime. Use mutable DMA for live caller inputs and freshly allocated
  outputs.
- Give Python-visible results independent storage when a caller may retain
  them across another component or forward.
- Treat direct SPM-to-final-DDR output as a transfer with explicit destination
  ownership. It removes staging, not the copy or lifetime contract.

## Handoff sequence

1. Validate the next component's logical inputs, geometry, dtype, and profile.
2. Finish or synchronize the producing component according to its public
   Graph/DMA contract.
3. Preserve only declared persistent state; retire temporary allocations and
   stale live-address bindings.
4. Transfer or bind the produced value through the wrapper that matches its
   address lifetime.
5. Enter the consuming component through its adapter-owned capture scope and
   build or replay the matching signature.
6. Before returning to an earlier component, repeat its public admission and
   ownership path; do not assume its previous temporary state remains live.

## Failure signals

- The first forward is correct but a repeated or multi-image forward changes:
  check replay-persistent SPM and fixed DMA addresses.
- A retained Python result changes after the next component runs: check output
  storage ownership.
- A component passes alone but the composed policy fails: check the boundary
  tensor's dtype, layout, logical length, synchronization, and Graph signature.
- A fresh process works while switching profiles in one process fails: check
  one-live-owner, cold configuration, teardown, and persistent-generation
  ownership.

## Sources

- [Architecture: SPM, Graph, and DMA ownership](../../docs/architecture.md)
- [Policy API contracts](../../docs/api_reference.md#policy-apis)
- [GraphCache capture](graphcache-capture.md)
- [SPM allocation](spm-allocation.md)
- [DDR/SPM DMA wrappers](ddr-dma-wrappers.md)
- [Pi0.5 adapter](../../python/rpu_backend/adapters/pi05/__init__.py)
- [Wall-OSS adapter](../../python/rpu_backend/adapters/wall_oss/__init__.py)
- [RhinoVLA facade](../../python/rpu_backend/api/rhinovla.py)
