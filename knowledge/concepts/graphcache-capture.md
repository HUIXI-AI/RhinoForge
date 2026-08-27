# GraphCache capture

`GraphSignature` is the key for one admitted execution shape and semantic mode.
`GraphCache.capture(signature)` builds the graph on the first call and reuses it
on later calls with the same signature. [Graph API](../../docs/api_reference.md#graph-api)

## Correct reuse

- A production adapter normally owns a dedicated per-model `GraphCache` and
  wraps its top-level native forward in `capture`.
  [Architecture](../../docs/architecture.md#graph-and-batch-execution)
- Every value that can change the result must be encoded in the signature or
  updated at a stable contract point through a mutable transfer.
  [Model porting](../../docs/model_porting.md#7-graph-and-cache-invariants)
- A repeated signature is healthy when it builds once, later replay counters
  increase, cache size stays fixed, no second build occurs, and
  `cache_invariant_ok()` remains true.
  [Architecture](../../docs/architecture.md#graph-and-batch-execution)
- `freeze()` changes a warmed cache to lookup-only mode; a missing or invalid
  entry is rejected instead of being built online.
  [GraphCache implementation](../../python/rpu_backend/graph/_runtime.py)
- Cache ownership includes teardown. Prefer an explicit per-model cache over a
  shared default when writing a new adapter.
  [Graph API](../../docs/api_reference.md#graph-api)

A bounded one-shot graph is an explicit model-contract exception for a
variable-shape path. It must execute one non-trivial graph, return to
passthrough state, and leave retained-cache size unchanged.
[Model porting](../../docs/model_porting.md#7-graph-and-cache-invariants)

## Sources

- [Architecture: graph and batch execution](../../docs/architecture.md#graph-and-batch-execution)
- [Graph API](../../docs/api_reference.md#graph-api)
- [Canonical Python runtime](../../python/rpu_backend/graph/_runtime.py)
- [Model porting: graph invariants](../../docs/model_porting.md#7-graph-and-cache-invariants)
