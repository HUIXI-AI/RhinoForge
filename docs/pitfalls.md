# Porting pitfalls

Use this page when a port runs but is numerically wrong, changes after warmup,
or fails only with repeated or multi-input inference. Start with P1–P5 before
adding diagnostics; each has one small distinguishing check.

The contracts below follow [Architecture](architecture.md) and the public
adapter and `FusedModelBase` source. They do not replace the full gates in
[Model porting](model_porting.md).

## P1 — Double-swizzle

**Symptom:** transformed weights have a plausible norm, but projection outputs
point in the wrong direction; a fresh model works until a second conversion or
installation attempt.

**Cause:** `swizzle_model_inplace` touched a weight twice, or both the generic
walker and a fused subsystem transformed the same specialized linear layer.
The transformation is irreversible in place and cannot detect every repeated
application from tensor shape alone.

The transformed layout also depends on tensor element width. A weight
transformed as FP32 and cast to FP16 afterward is not equivalent to a weight
cast to FP16 first and then transformed for the FP16 profile.

**Fix:**

- put the clean CPU model in the profile's final admitted dtype before the
  first transformation, and do not cast transformed storage afterward;
- keep one authoritative transformation path per weight;
- put specialized linear modules in `SKIP_LINEAR_NAMES`;
- set the transformation-started marker before the first mutation;
- make class-level patches idempotent; and
- discard and reload the model after any partial failure.

**Gate:** install a clean model once, confirm a completed second `.to("rpu")`
does not transform weights again, and confirm a simulated partial install fails
instead of retrying.

## P2 — Out-of-band SPM state

**Symptom:** the first forward is correct, while a later REPLAY or a run with
warmup is corrupted.

**Cause:** data needed across REPLAY was placed in temporary SPM or created
outside the subclass's deterministic `declare_buffers()` manifest. Reset or
reuse then invalidates the hidden lifetime assumption.

**Fix:** declare replay-persistent data as `Persistent` or
`PersistentPerLayer`, and use its `preload_callback` when data must be restored
after allocation-state changes. Keep `declare_buffers()` free of launch side
effects and describe aliases only when lifetimes do not overlap.

**Gate:** compare `RPU_WARMUP=0`, `1`, and `3`, then repeat one signature until
the cache shows stable REPLAY. Values must remain bit-identical and
`cache_invariant_ok()` must stay true.

## P3 — Absolute address versus offset

**Symptom:** output is deterministically wrong even though shapes, weights, and
SPM capacity are correct.

**Cause:** a host wrapper received an SPM offset where its public contract
requires an absolute address, or the reverse.

**Fix:** use `addr()` and `layer_addr()` for ordinary operators. Use
`addr_offset()` or `layer_addr_offset()` only when the wrapper contract
explicitly requires the typed `SpmOffset`, such as attention or KV-cache
insertion. Do not cast between the two to make a call compile.

**Gate:** review every SPM argument against its declaration in
`src/core/rpu_kernel_decls.h`, then compare the affected operation with its CPU
reference before rerunning end to end.

## P4 — Reused output storage

**Symptom:** a single call passes, but a caller that retains results, processes
multiple inputs, or accumulates image outputs observes earlier values changing.

**Cause:** two forwards return views of one reusable DDR tensor, or Graph replay
writes through an address captured for an earlier output.

**Fix:** allocate independent Python-visible output storage per forward. A
`FusedModelBase` subclass can use `allocate_tracked_output`; other paths can use
a fresh `at::empty(...)`. Use mutable DMA when that fresh destination address
changes between BUILD and REPLAY.

**Gate:** retain the first output, clone its bytes, run a different second
input, and verify that the first output is unchanged and does not share storage
with the second.

## P5 — Cold per-handle execution settings

**Symptom:** a direct adapter run and the public loader choose different chunk
plans, or changing a setting after the first forward has no effect or reuses an
incompatible Graph.

**Cause:** chunk or padding state was set through an obsolete process-global
path, changed after model installation, or not forwarded into the native
handle.

**Fix:** pass public chunk and padding choices through the loader's immutable
`rpu_execution` mapping before `.to("rpu")` and the first BUILD. Treat
model-specific `_rpu_chunk_size` as a compatibility attribute only where that
adapter documents it; new integrations use `rpu_execution`. Recreate the model
to change a cold setting.

**Gate:** prove an invalid setting fails before weight mutation, and prove each
admitted setting in a fresh process reports and executes the expected resolved
plan.

## Other high-value checks

| Check | Failure pattern | Required correction |
|---|---|---|
| Top-level capture | Every call stays outside retained Graph execution | Wrap the production native forward in the adapter-owned `GraphCache.capture(signature)` unless the reviewed model contract defines a bounded one-shot |
| Complete signature | Same shape but different mask, grid, or lookup data reuses stale results | Add the semantic value to the signature, keep it at a proven stable address, or update it through mutable DMA |
| DMA address ownership | BUILD passes, REPLAY reads or writes an earlier tensor | Use fixed DMA only for stable model-owned storage; use mutable DMA for caller inputs and fresh outputs |
| Multi-core broadcast | An immediate multi-core operation updates only one core | Call `set_broadcast_mode(true)`; batch and Graph paths already apply it centrally |
| Model invalidation | New weights or layout settings reuse an old plan | End every layout-, weight-, or Graph-affecting setter with `invalidate_model_state()` |
| CPU fallback | End-to-end execution completes but a required model component did not run on RPU | List the fallback explicitly and keep the profile out of Supported status until the intended execution contract is validated |
| Profile inheritance | One size or short input passes and nearby profiles are accepted | Guard exact configuration tuples and maximum envelopes before weight loading |
| Process ownership | A second live fused model or transformed model move causes state conflicts | Use one live RPU-resident fused model or policy per process and reload a clean CPU instance for another placement |

## Fast triage order

1. Compare the first call with the first repeated call. A repeat-only failure
   points first to P2, signature completeness, or DMA ownership.
2. Retain output one while running a different input. Mutation points to P4.
3. Re-run from a fresh CPU model and isolate the first wrong projection. Check
   P1, then P3.
4. Compare the requested and resolved execution plans. A mismatch points to
   P5 or profile preflight.
5. Only when top-level parity or task evidence still fails, bisect model phases
   against the exact same-dtype reference and use CPU FP32 as the precision
   anchor.

Finish with the complete semantic, parity, task, warmup, Graph-lifecycle,
multi-input, and maximum-envelope gates in
[Model validation policy](validation_policy.md).
