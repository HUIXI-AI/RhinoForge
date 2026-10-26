# Porting pitfalls

[简体中文](pitfalls.zh.md) | English

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

## Pi0.5 v1.2.0 denoise known issues (updated 2026-10-26)

This section records two related failures confirmed during a v1.2.0 customer
investigation. They affect the default Graph loop and the Graph-disabled host
loop respectively. The current RhinoForge source contains compatible host and
native-extension fixes without changing the v1.2.0 Runtime/Launch package or operator
assets. Unpatched v1.2.0 RhinoForge builds remain affected.

### Changing logical prefixes retain Graphs until the arena pool is exhausted

**Symptom:** `pi05_denoise_loop` repeatedly BUILDs across control frames and
fails with `graph arena pool capacity exhausted` after the number of retained
Graphs reaches the cold arena count. This is one new complete loop Graph per
top-level action call, not one rebuild per denoise step within a frame.

**Cause:** [`_set_denoise_rope_position`](../python/rpu_backend/adapters/pi05/runtime.py)
derives the logical RoPE start from the prefix mask's valid-token count. State
is part of the tokenized prompt, so that value can change across control
frames. The RoPE start is baked into the kernel at BUILD, and
[`_run_denoise_loop`](../python/rpu_backend/adapters/pi05/runtime.py) therefore
correctly includes it and the physical execution plan in the `GraphSignature`.
Different values miss the cache, while Python
[`GraphCache`](../python/rpu_backend/graph/_runtime.py) and native
[`RpuGraphCache::get_or_create`](../src/graph/graph_infra.cpp) enforce capacity
without automatic eviction. Old Graphs retain the finite physical arenas until
the pool is exhausted. This “leak” is an unbounded retained-Graph lifecycle,
not an ordinary host-heap leak.

**Fix:**

- The bounded-cache fix remains a safety net: retain only the current denoise
  signature and use `GraphCache.evict()` before creating a different one.
- The production performance fix makes the logical RoPE start replay-mutable.
  It is removed from the Python Graph signature and native plan identity. Before
  REPLAY, the validated TP8/T160/10-step path updates only the two 32-bit position
  registers in `PARTIAL_MROPE` and KV1 direct-cache nodes and verifies the exact
  node count; other paths retain the safe host-layer-body re-emission fallback.
  Physical route validation remains keyed by the canonical padded prefix length.
  Vision and prefill retain their existing full fast-replay path.
- Invalid fixes: increasing `graph_arena_count` only delays failure; deleting
  the RoPE start from the signature without a mutable semantic path silently
  replays stale values.

**Gate:** run multiple frames whose valid-token counts change and prove cache
size and arena use remain bounded. The production fix must also prove one
BUILD followed by stable REPLAY, `cache_invariant_ok()` throughout, and
numerical parity at different logical positions. Eviction must retain the
active/use-count guards.

**R1 evidence:** with the supplied six T160 frames (149--154 valid tokens), two
warmups and six measured W8A16 iterations used one denoise Graph with ten
replays, zero recaptures and a valid cache invariant. Mean end-to-end latency
was 168.27 ms (median 168.54 ms, P95 169.72 ms), down from 209.14 ms before the
mutable replay fix. Mean denoise time fell from 101.94 ms to 60.01 ms. Against
the Graph-disabled reference, per-frame action RMSE was 0.00049--0.00055,
maximum absolute error was at most 0.00340, and cosine similarity was at least
0.99999988. The v1.2.0 Runtime/Launch package and operator assets were unchanged.

With the exact register update and four CPU threads, the same W8A16 workload
averaged 153.69 ms end to end (median 153.93 ms, P95 154.25 ms) and 50.91 ms in
denoise; all six actions were bit-identical to the correct layer-body re-emission
implementation. The formal T160 Action NVFP4 + Prefill W8A8 profile averaged
142.97 ms (median 142.76 ms, P95 143.93 ms), with Vision/Prefill/denoise means of
22.36/67.69/44.85 ms. Four and eight CPU threads produced bit-identical outputs,
while four threads improved on the eight-thread 145.88 ms mean, so the T160 NVFP4
templates cap CPU threads at four to reduce contention with colocated applications.
Prefill W8A8 adds quantization differences and still requires task-success
acceptance; it is not an unconditional default replacement.

### Disabling the denoise Graph corrupts steady-state actions

**Symptom:** setting `RPU_PI05_DENOISE_GRAPH=0` avoids arena exhaustion, but
later fixed-input, fixed-noise outputs change substantially after a first-frame
`[Pi05Fused] runtime checks PASS`; robot actions diverge within seconds.

**Cause:** [`_run_denoise_fused`](../python/rpu_backend/adapters/pi05/runtime.py)
already computes a complete `action_plan`. The Graph branch passes the selected
`physical_descriptor`, but the host branch passes an empty list. Native
[`restore_planned_route_profile`](../src/fused/rpu_pi05_denoise_step_model.cpp)
then enters the legacy immediate path without Graph proof. Outside Graph,
[`FusedModelBase::run_preload_callbacks_`](../src/core/fused_model_base.cpp) may
skip preloads when the generation appears unchanged and the dirty flag is
false. Intervening SigLIP/Gemma Graphs can reuse and overwrite physical SPM, so
denoise Persistent biases such as `action_in_proj_b_spm` and
`action_out_proj_b_spm` can read clobbered data. The first frame can pass after
a correct preload, while `_rpu_fuse_validated` is only a one-time check and
does not prove steady-state correctness.

In a three-call fixed-input, fixed-noise diagnostic, the original path changed
from the first to second call with RMSE `0.8649037` and maximum difference
`2.6068361`. Retaining and passing the same COMPLETE descriptor made all three
outputs bit-identical, with zero RMSE and maximum difference. This excludes
host FP32 Euler as the cause and isolates descriptor loss and incomplete
preload/route authority.

**Fix:**

- Current fix: pass the selected COMPLETE `physical_descriptor` in the host
  branch too, so immediate execution follows the current physical manifest and
  performs the required preloads.
- Reject a Graph-only optimized descriptor early and explicitly when Graph is
  disabled; never fall through silently to the empty-descriptor legacy path.
- Do not globally force every legacy `FusedModelBase` forward to preload. That
  has a much wider blast radius than using the exact descriptor already
  computed for this call.

**Gate:** run at least three fixed-input, fixed-noise calls with intervening
SigLIP/Gemma Graph replays and require stable output. Then compare the host loop
against the reference/Graph loop across multiple frames. Graph-only profiles
must fail closed when `RPU_PI05_DENOISE_GRAPH=0`. A one-time first-frame
`runtime checks PASS` is not a steady-state gate.

### v1.2.0 operational guidance

For a v1.2.0 RhinoForge build without the source patch, do not work around these
failures by adding arenas or disabling the denoise Graph. Use a validated v1.0
combination or rebuild RhinoForge with both adapter fixes. In representative
W8A16/R1 validation, the denoise cache stayed at one entry across ten changing-
prefix frames and the final repeated signature REPLAYed; with Graph disabled,
three fixed-input/fixed-noise outputs were bit-identical with zero maximum
difference. The customer's original checkpoint still requires the same focused
acceptance run before delivery. Prefix-composition fallback after arena failure
and a shutdown-time `sys.meta_path is None` cleanup error are consequences or
secondary issues, not the causes of these failures.

## Other high-value checks

| Check | Failure pattern | Required correction |
|---|---|---|
| Top-level capture | Every call stays outside retained Graph execution | Wrap the production native forward in the adapter-owned `GraphCache.capture(signature)` unless the reviewed model contract defines a bounded one-shot |
| Tensor views | A view created across inference-mode contexts loses its base state or raises a version-counter error | Use native PyTorch view construction to preserve base inference state, aliasing, storage offsets, bounds and version counters |
| Complete signature | Same shape but different mask, grid, or lookup data reuses stale results | Add the semantic value to the signature, keep it at a proven stable address, or update it through mutable DMA |
| DMA address ownership | BUILD passes, REPLAY reads or writes an earlier tensor | Use fixed DMA only for stable model-owned storage; use mutable DMA for caller inputs and fresh outputs |
| Multi-core broadcast | An immediate multi-core operation updates only one core | Call `set_broadcast_mode(true)`; batch and Graph paths already apply it centrally |
| Model invalidation | New weights or layout settings reuse an old plan | End every layout-, weight-, or Graph-affecting setter with `invalidate_model_state()` |
| CPU fallback | End-to-end execution completes but a required model component did not run on device | List the fallback explicitly and keep the profile out of Supported status until the intended execution contract is validated |
| Profile inheritance | One size or short input passes and nearby profiles are accepted | Guard exact configuration tuples and maximum envelopes before weight loading |
| Process ownership | A second live fused model or transformed model move causes state conflicts | Use one live device-resident fused model or policy per process and reload a clean CPU instance for another placement |

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
