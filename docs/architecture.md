# Architecture

RhinoForge is a PyTorch `PrivateUse1` backend for inference on the Rhino
Processing Unit. Importing `rpu_backend` registers the device name `rpu`, loads
the native extension when the board device is accessible, installs the
`torch.rpu` namespace, and registers model adapters lazily.

```text
Hugging Face model or policy API
              |
        Python adapter
  profile guard, weight layout,
  cache and graph orchestration
              |
       torch.ops.rpu.*
              |
 C++ operators / FusedModelBase
 SPM plan, multi-core schedule,
 graph, batch and DMA nodes
              |
       Rhino Launch 1.0.0
              |
             RPU
```

The combined operator asset is loaded through the host launch interface. The
asset is opaque to RhinoForge: the backend selects named operators and supplies
the documented host ABI parameters, but does not parse the asset format or
interpret device instructions.

## Runtime layers

### Python API and adapters

The public loaders and policy facades live under `rpu_backend.api`. A loader
first reads the model configuration, resolves its exact Hugging Face
architecture, and asks the adapter registry for an implementation. The adapter
then:

1. rejects unsupported profiles before loading or modifying weights;
2. transforms supported weights into the layout consumed by RPU operators;
3. moves tensors to `rpu`;
4. creates native model handles and caches;
5. installs the model-specific forward path; and
6. owns graph signatures and lifecycle.

Weight transformation is irreversible in place. Do not transform one model
instance twice or move a transformed instance back to CPU. The main fused
CausalLM path also permits only one live RPU-resident model or policy per
process.

Built-in Hugging Face adapters are registered from a static manifest and
imported only when selected. Third-party adapters use the same registry without
overriding built-in architecture names. See [API reference](api_reference.md).

### Operators and CPU fallback

Native operators register implementations for the PyTorch `PrivateUse1`
dispatch key and model-specific schemas under `torch.ops.rpu`. Supported tensor
operations launch an RPU operator directly or contribute nodes to an active
graph. An operation without an RPU implementation may use the checked CPU
fallback path; a model profile is not considered supported merely because all
of its calls return.

The public operator boundary is the host wrapper: tensor shapes, dtypes,
addresses, lengths, core selection, synchronization, and parameter packing may
be represented in source. Device program source, instruction semantics,
instruction streams, and operator-asset internals are outside this repository.

## FusedModelBase contract

`v3::FusedModelBase` is the shared C++ framework for fused model subsystems. It
keeps repeated decoder or encoder intermediates in per-core SPM and centralizes
chunk planning, allocation, graph construction, DMA, and model-state
invalidation.

A new fused subclass implements four required methods:

- `declare_buffers(const LayoutContext&)` returns the SPM buffer manifest and
  lifetimes. It must be deterministic and free of launch side effects because
  the planner may call it for multiple candidate chunk sizes.
- `static_config()` declares model-wide layer and traversal configuration.
- `dynamic_config(const ChunkPlan&)` selects the execution order and
  inter-layer storage mode for the resolved plan.
- `build_layer_subgraph(layer_idx, chunk)` emits the operator sequence for one
  layer and chunk through existing launch wrappers.

After a subclass changes bound weights or any state that affects the layout or
graph, its setter must finish with `invalidate_model_state()`. Framework-owned
outputs that may outlive a forward use `allocate_tracked_output`; Python-visible
outputs accumulated by a caller must have independent storage.

Use `addr()` and `layer_addr()` for absolute SPM addresses consumed by ordinary
operators. Offset-oriented access is limited to operators whose host contract
explicitly requires an SPM offset, such as attention and KV-cache insertion.

### Planning and invalidation lifecycle

A fused handle keeps model-state invalidation, SPM allocation identity, and
Graph identity separate. Model setters bind weights or configuration and call
`invalidate_model_state()`. The next admitted forward resolves a chunk plan,
evaluates the deterministic buffer manifest, allocates the matching layout, and
contributes operations to the adapter-owned Graph BUILD. Later calls may REPLAY
only when the model state, allocation identity, Graph signature, and live DMA
inputs still satisfy the same contract.

`invalidate_model_state()` marks model-owned weights and preload work dirty; it
does not invent a new allocation identity. If `declare_buffers()` sizes storage
from subclass-owned state outside `LayoutContext` and the standard model
parameters—for example an image grid, action horizon, or expert count—the
subclass must include that state in `subclass_layout_hash()`. Otherwise a new
shape can incorrectly reuse an older SPM layout.

Use `subclass_chunk_size_cap()` to bound automatic chunk search and
`subclass_chunk_size_valid()` for constraints that every automatic or explicit
candidate must satisfy. Both hooks, like `declare_buffers()`, must be pure and
deterministic; rejecting a candidate is preferable to emitting a graph for an
unproven layout.

## SPM design

The current runtime targets eight cores with 8 MiB of SPM per core and plans to
a conservative usable budget of about 7.5 MiB per core. FP16 is the primary
execution dtype, and many operator dimensions require 16-element alignment.

`BufferDecl` expresses storage and lifetime:

- `Temp` and `TempPerLayer` can be reused after their declared phase ends;
- `Persistent` and `PersistentPerLayer` survive graph replay; and
- `alias_of` permits explicit reuse only when producer and consumer lifetimes
  do not overlap.

The chunk planner evaluates the same alias-aware first-fit plan used by the
allocator. SPM capacity is a hard admission constraint. Among feasible
16-aligned candidates, the generic planner first minimizes chunk count and then
minimizes the final-chunk deficit. Model-specific logical padding, when
allowed, remains in the adapter and is removed from outputs and cache position
before returning to the caller.

Persistent data needed after a replay must be declared in the manifest, with a
preload callback when necessary. Temporary SPM is reset at subsystem boundaries;
the corresponding model handle releases its compute ownership before the next
subsystem starts.

## Graph and batch execution

`GraphSignature` identifies one admitted execution shape and semantic mode.
`GraphCache.capture(signature)` records the first call as BUILD and reuses the
retained graph for later calls with the same signature as REPLAY. Each retained
cache entry owns its launch queue. Production adapters normally wrap their
top-level native forward in this scope.

A repeated signature is valid only when:

- the first call builds once;
- later calls replay without a second build;
- replay counters increase while cache size remains fixed; and
- `cache_invariant_ok()` remains true.

Some variable-shape paths may define a bounded one-shot graph instead. Such a
path must record a non-trivial graph, execute it once, return to passthrough
state, and leave retained-cache size unchanged. This is an explicit
model-contract exception, not the default porting pattern.

Batching groups kernel, DMA, and synchronization nodes into launch segments.
One-core and eight-core operators may coexist in a batch; each node retains its
own core-count contract. Immediate multi-core launches must enable broadcast
mode so every participating core receives the host parameters. Batch and graph
paths do this centrally.

## DMA ownership

RhinoForge uses explicit DDR-to-SPM broadcast and SPM-to-DDR copy wrappers.
Their replay behavior is part of graph correctness:

- a fixed wrapper captures its DDR source or destination address at BUILD and
  is safe only for storage whose address stays stable across forwards;
- a mutable wrapper updates the live DDR address before REPLAY and is required
  for caller inputs or newly allocated outputs; and
- an immediate wrapper performs a one-shot transfer outside graph capture.

Mutable DMA changes the replay address; it does not remove the physical data
transfer. A direct SPM-to-final-DDR path removes host staging but is not
zero-copy.

## Source layout

| Path | Responsibility |
|---|---|
| `python/rpu_backend/api/` | Public model, cache, and policy interfaces |
| `python/rpu_backend/adapters/` | Model-profile guards and orchestration |
| `python/rpu_backend/graph/` | Graph signatures, caches, and capture scopes |
| `python/rpu_backend/runtime/` | Shared device, registry, weight, and decoder helpers |
| `python/rpu_backend/quant/` | Offline quantization and checkpoint loading |
| `src/core/` | Backend state, memory, launch cache, and FusedModelBase |
| `src/graph/` | Capture, segmentation, replay, and fallback |
| `src/ops/` | Public host wrappers for tensor operations |
| `src/fused/` | Model-specific fused subsystems |

Supported behavior is defined by an exact model profile and release asset set,
not by source-file presence. See [Model support](model_support.md) and
[Model porting](model_porting.md).
