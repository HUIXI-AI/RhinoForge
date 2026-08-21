# Attention layout boundaries

RhinoForge's public unified attention boundary uses
`[batch, sequence, heads, head_dim]`. The common PyTorch attention convention
uses `[batch, heads, sequence, head_dim]`. Both describe the same logical
tensor, but silently treating one as the other changes which values belong to
each token and head.

## Boundary rules

- Reshape a contiguous projection from `[batch, sequence, hidden]` to
  `[batch, sequence, heads, head_dim]` when `hidden == heads * head_dim`.
- Crossing from the unified boundary to the common PyTorch convention requires
  swapping the sequence and head dimensions. Materialize contiguous storage
  when the next consumer requires it; do not rely on a transposed stride being
  accepted implicitly.
- Keep query, key, value, and returned attention output in one declared layout
  throughout a wrapper. Convert once at the boundary instead of adding
  compensating transposes at individual call sites.
  [Public attention schema](../../src/core/rpu_dispatch_registrations.inc)
- Treat `RPUCache` as the owner of cached K/V storage. Callers size it and
  update its logical position through the public cache API; they do not
  reinterpret its storage as a regular PyTorch attention tensor.
  [RPUCache](../../docs/api_reference.md#rpucache)
- Mask shape, causal mode, cache position, and head geometry are semantic
  inputs. They belong in profile validation and, where applicable, the Graph
  signature.
  [Model porting](../../docs/model_porting.md#7-graph-and-cache-invariants)
- A leading batch dimension does not imply that every model path supports
  general batching. Use only the envelope named for the exact release profile.
  [Model support](../../docs/model_support.md#general-runtime-limits)

If attention output is numerically wrong while projection norms look
reasonable, verify the layout at every public boundary before changing model
math or adding a model-specific workaround.

## Sources

- [Public attention registration](../../src/core/rpu_dispatch_registrations.inc)
- [Public tensor-operation boundary](../../src/core/rpu_tensor_ops.inc)
- [Runtime decoder](../../python/rpu_backend/runtime/decoder.py)
- [Cache implementation](../../python/rpu_backend/api/cache.py)
- [Architecture](../../docs/architecture.md)
