# Model porting

This guide covers inference-only ports that use RhinoForge's public adapter,
operator, graph, SPM, and host-launch interfaces. A port is complete only for
an exact model profile: architecture, checkpoint revision, precision, input
envelope, execution settings, Rhino Launch version, and operator-asset version.
Code reuse or a matching Hugging Face architecture name is not a support claim.

Before changing code, compare the target with [Model support](model_support.md)
and write down the exact profile to admit. Unsupported profiles must fail
before the full weight load or any in-place weight transformation.

For AI-assisted work, use the checked-in
[`rhinoforge-port` skill](../.agents/skills/rhinoforge-port/SKILL.md). Start in
assessment mode, review its outcome and certification, then enter port mode
only for an accepted, certified, non-Blocked result. The skill loads a public
capability map and one family reference; the CausalLM reference routes
Adapter-only work to the step-by-step checklist.

## 1. Assess the capability gap

Trace one reference forward from input preprocessing through the returned
outputs. Record each operation's shapes, dtypes, state updates, cache behavior,
and numerical reference. Then map every operation to an existing public
RhinoForge path.

Choose the smallest applicable outcome:

| Outcome | Use when |
|---|---|
| Existing path | A supported adapter and exact profile already cover the model |
| Adapter-only | Existing public operators express the model; only configuration, weights, cache, or forward orchestration differs |
| Runtime extension | A new host-side operator wrapper or shared runtime behavior is required |
| New fused subsystem | The model needs a new repeated fused execution path and SPM plan |
| Blocked | A required operator asset, shape, precision, memory envelope, or numerical result is unavailable |

Do not approximate missing model math merely to make the forward complete. If
the restricted operator asset lacks a required operation, request a
release-matched asset update with the operation's mathematical definition,
tensor shapes, dtypes, and CPU reference. Device-program implementation details
are outside the public porting workflow.

## 2. Start from the adapter template

For a decoder-only CausalLM, copy:

```text
python/rpu_backend/adapters/_template/minimal_causal_lm.py
```

`python/rpu_backend/adapters/_template/adapter.py` is the annotated version for
ports that need more structure. Fill these fields before registering the
adapter:

- `HF_ARCH`: exactly `config.architectures[0]`;
- `DECODER_ARCH`: an existing decoder branch only when its math matches;
- `SUPPORTED_PROFILES`: exact configuration tuples, not family-wide ranges;
- `RMSNORM_CLASS` and `ROTARY_CLASS`: the target Hugging Face classes;
- `SKIP_LINEAR_NAMES`: specialized linear layers with a separate, authoritative
  conversion path; and
- the profile's admitted chunk envelope.

The adapter should keep these responsibilities in Python:

1. configuration and dependency preflight;
2. CPU-first checkpoint loading;
3. exactly-once weight transformation;
4. class swaps or small layout-boundary patches;
5. cache and native-handle ownership;
6. graph signatures and forward orchestration; and
7. teardown and process-ownership checks.

Use C++ only when existing public operations cannot express the required model
math or lifetime. Do not replace an entire model forward with a collection of
unreviewed patches.

## 3. Reuse the causal decoder when it matches

The minimal template installs the shared causal decoder after profile
validation and weight transformation. Preserve its ordering:

1. validate the exact profile and cold `rpu_execution` settings;
2. claim the live model instance;
3. mark the irreversible transformation as started;
4. transform each weight exactly once;
5. move parameters and buffers to `rpu`;
6. create and configure the native handle;
7. bind weights and install the forward path; and
8. mark the runtime complete only after every prior step succeeds.

Do not move a transformed model back to CPU or retry a partially failed
transformation. Reload a clean model instance instead. Python-visible outputs
that callers may retain or concatenate need fresh storage on every forward.

## 4. Add a fused subsystem only when required

New repeated fused paths derive from `v3::FusedModelBase`. A subclass provides:

```cpp
std::vector<BufferDecl> declare_buffers(const LayoutContext&);
ModelStaticConfig static_config();
ModelDynamicConfig dynamic_config(const ChunkPlan&);
void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk);
```

Keep `declare_buffers` deterministic and free of launch side effects because
the planner may evaluate more than one candidate. Declare replay-persistent SPM
state as `Persistent` or `PersistentPerLayer`; do not allocate it through a
temporary path. Use `addr()` or `layer_addr()` for operators whose host contract
takes an absolute SPM address, and use offsets only where the operator contract
explicitly calls for one.

If the manifest depends on subclass-owned sizing state that is not already in
`LayoutContext` or the standard model parameters, follow the
[planning and invalidation lifecycle](architecture.md#planning-and-invalidation-lifecycle)
and implement `subclass_layout_hash()`. Model-state invalidation alone does not
change the allocation identity.

Every weight or configuration setter that changes layout, planning, or graph
state must end with `invalidate_model_state()`. Immediate multi-core launches
must enable broadcast mode. Prefer the framework's batch and graph paths, which
apply this requirement centrally.

For DDR/SPM transfers, choose ownership explicitly:

- fixed DMA is valid only when the captured DDR address remains stable;
- mutable DMA updates a caller or fresh-output address before replay; and
- immediate DMA is a one-shot transfer outside capture.

## 5. Add or reuse a public operator

When an existing operator is sufficient, call its public host wrapper rather
than duplicating it. A new host-side operator normally requires these source
changes:

1. declare the host wrapper in `src/core/rpu_kernel_decls.h`;
2. implement argument validation and launch orchestration in `src/ops/`;
3. add its schema and `PrivateUse1` registration through
   `src/core/rpu_dispatch_registrations.inc`;
4. add the source file to `src/CMakeLists.txt`; and
5. add a FakeTensor implementation only if the public profile promises
   `torch.compile` support.

The wrapper may expose the host ABI needed to select and launch a named
operation: tensor metadata, memory locations, lengths, core selection,
synchronization, and parameter packing. Keep restricted operator assets outside
the repository and use only the release's documented lookup interface.

## 6. Register the adapter

An in-tree adapter is added to
`python/rpu_backend/adapters/_manifest.py` as an architecture, module, and class
tuple. Its module should also call `register_adapter(HF_ARCH, AdapterClass)` so
direct imports and lazy manifest loading agree.

An external package can avoid an in-tree change by registering the same
adapter contract through the `rpu_backend.plugins` entry-point group. Plugins
cannot replace a built-in or previously registered architecture. See
[API reference](api_reference.md#adapter-registry-and-plugins).

## 7. Graph and cache invariants

Production top-level forwards normally run inside
`GraphCache.capture(signature)`. For a repeated signature, verify one BUILD
followed by stable REPLAY: the replay counter increases, cache size remains
fixed, no second BUILD occurs, and `cache_invariant_ok()` stays true.

A bounded one-shot graph is an explicit model-contract exception for a
variable-shape path. It must record and execute a non-trivial graph once, return
to passthrough state, and leave the retained `GraphCache` size unchanged.

Every semantic input that can affect output belongs in the signature or must be
updated through a mutable transfer at a stable contract point. Same-signature,
different-value cases must match the uncaptured path.

## 8. Verification gates

Run board-free checks first:

- unknown architecture and unsupported-profile rejection occurs before weight
  loading or transformation;
- registry and plugin discovery are deterministic;
- execution configuration rejects malformed or out-of-envelope values;
- cache sizing and output shapes match the reference contract; and
- the source build and package import complete cleanly.

Then run the exact release profile on an RPU board:

- run the hard semantic, same-dtype implementation-parity, CPU FP32 anchor,
  and task-quality gates defined by the exact profile in
  [Model validation policy](validation_policy.md);
- verify `RPU_WARMUP=0`, `1`, and `3` produce bit-identical outputs;
- prove the applicable graph lifecycle invariant;
- run multiple inputs, and multiple images for a vision model, to catch reused
  output storage;
- exercise the maximum admitted sequence, batch, image, and cache envelope;
  and
- run the public end-to-end example from a clean process with the exact
  checkpoint, TOML, launch library, and operator asset.

Stop and classify the port as Blocked when a required operation, memory plan,
profile asset, or hard semantic gate is missing. Incomplete representative
task evidence can remain Experimental or Source-only, but cannot be promoted to
Supported or Limited. Do not broaden the accepted input or model table based on
a smaller passing case.

Record the exact checkpoint revision and hashes, configuration hash, software
versions, operator-asset hash, input envelope, precision, and results in the
release evidence. Only then add or change a row in
[Model support](model_support.md).
