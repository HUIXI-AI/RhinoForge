# Model porting playbook

This is the shortest supported path from a Hugging Face model to a RhinoForge
release profile.

1. **Define the exact profile.** Record architecture, checkpoint revision,
   precision, input envelope, execution settings, launch version, and operator
   asset version. [Capability review](../../docs/model_porting_capability_review.md)
2. **Trace the numerical references.** Freeze the exact same-dtype executable
   for implementation parity and a CPU FP32 anchor, then map shapes, dtypes,
   state, cache behavior, and outputs to existing public paths.
   [Capability review](../../docs/model_porting_capability_review.md)
3. **Choose the smallest outcome.** Prefer an existing path, then adapter-only;
   add shared runtime behavior or a fused subsystem only when the model math or
   lifetime requires it. Existing path ends here; Blocked stops; only
   Adapter-only continues with the decoder checklist. Other outcomes use the
   broader porting guide. [Capability review](../../docs/model_porting_capability_review.md)
4. **For Adapter-only, start from the public adapter template.** Validate the profile before
   mutation, keep orchestration in Python, and reuse the causal decoder only
   when its math matches. [Decoder checklist](../../docs/new_model_step_by_step.md)
5. **Close the state contracts.** Apply weight conversion once, size the KV
   cache for the full envelope, declare all SPM lifetimes, select fixed or
   mutable DMA by ownership, and encode every semantic graph input.
   [Architecture](../../docs/architecture.md)
6. **Register one architecture binding.** In-tree adapters use the static
   manifest; external packages use the public plugin entry point.
   [Adapter registry](../../docs/model_porting.md#6-register-the-adapter)
7. **Run board-free gates, then exact-profile board gates.** Check early
   rejection, deterministic discovery, sizing and imports first; then verify
   CPU/RPU numerics, warmup stability, graph lifecycle, multiple inputs, and
   the maximum admitted envelope. [Verification gates](../../docs/model_porting.md#8-verification-gates)
8. **Publish evidence or narrow the claim.** A missing operation, unsafe memory
   plan, or hard semantic gate blocks that profile. Incomplete parity or task
   evidence keeps it Experimental or Source-only; it is not a reason to widen
   the support table. [Validation policy](../../docs/validation_policy.md)

Concept references: [SPM allocation](../concepts/spm-allocation.md),
[GraphCache capture](../concepts/graphcache-capture.md),
[DMA wrappers](../concepts/ddr-dma-wrappers.md),
[KV cache](../concepts/kv-cache.md),
[weight swizzle](../concepts/weight-swizzle.md), and
[multi-core broadcast](../concepts/multicore-broadcast.md).

## Sources

- [Capability review](../../docs/model_porting_capability_review.md)
- [Decoder step-by-step](../../docs/new_model_step_by_step.md)
- [Porting pitfalls](../../docs/pitfalls.md)
- [Model porting](../../docs/model_porting.md)
- [Architecture](../../docs/architecture.md)
- [Adapter template](../../python/rpu_backend/adapters/_template/adapter.py)
- [Adapter manifest](../../python/rpu_backend/adapters/_manifest.py)
