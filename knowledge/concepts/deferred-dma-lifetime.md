# Deferred DMA tensor lifetime

A device address is a number, not ownership. During Graph capture a wrapper may
record an address and return before the Graph executes. If the tensor that owns
that storage dies first, BUILD or REPLAY can read or overwrite reused memory.

## Rule

Every tensor whose address is consumed by deferred execution must remain owned
until that execution completes. This applies to both DMA sources and
destinations.

- Keep a tensor reference in the Graph or model object for the required
  lifetime.
- Use mutable DMA when a caller input or fresh output changes address across
  forwards.
- Use fixed DMA only for model-owned storage proven stable for every replay.
- Allocate independent storage for outputs a caller may retain.
- Test the cold first BUILD as well as warm REPLAY; warmup can hide a
  first-allocation lifetime defect.

## Distinguishing check

Run the first call in a fresh process, repeat the same signature, then retain
one output while running a different input of the same shape. A cold-only
failure points to deferred ownership or initialization; a replay-only failure
points to live-address refresh or persistent state; mutation of the retained
output points to destination reuse.

## Sources

- [DMA ownership](../../docs/architecture.md#dma-ownership)
- [DDR/SPM DMA wrappers](ddr-dma-wrappers.md)
- [GraphCache capture](graphcache-capture.md)
- [Porting pitfalls](../../docs/pitfalls.md)
- [Graph runtime](../../src/graph/graph_runtime.h)
