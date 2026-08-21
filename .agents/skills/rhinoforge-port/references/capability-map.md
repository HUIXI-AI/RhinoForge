# Public capability map

Use this map during assessment to route one exact model profile and to prevent
an architecture-name match from becoming a support claim. The canonical
contracts remain [model support](../../../../docs/model_support.md),
[capability review](../../../../docs/model_porting_capability_review.md), and
[model porting](../../../../docs/model_porting.md).

## Evidence rule

A requirement is covered only when all three pieces agree:

1. the candidate source establishes the exact mathematical and lifecycle
   requirement;
2. public RhinoForge policy admits that requirement for the profile; and
3. public source implements the same tensor, state, and lifetime contract.

Candidate documentation is not RhinoForge implementation evidence. Source-file
presence alone is not a support claim. Missing or conflicting evidence makes
the assessment `uncertified` and records the affected row as needing review.

## Family routing

| Family | Recognition signal | Required capability checks | Working reference |
|---|---|---|---|
| Decoder CausalLM | decoder-only `*ForCausalLM`, one autoregressive stream, KV cache, no vision tower | attention, cache, position, norm, MLP, linear/weights, logits/generation, runtime/Graph | [CausalLM](families/causal-lm.md) |
| Vision encoder | patch/image embedding, bidirectional encoder blocks, returned image features, no generation cache | attention, position when present, norm, MLP, linear/weights, runtime/Graph | [Vision](families/vision.md) |
| VLM | vision tower plus text decoder and a projector, merger, or image/text composition step | every decoder and vision check plus multimodal glue | [VLM](families/vlm.md) |
| VLA | VLM path plus state/action inputs, an action path, and an iterative or scheduled policy update | every VLM check plus action-state lifecycle and final policy output | [VLA](families/vla.md) |

If no row matches, keep the family `unknown`, check the union of relevant
categories, and do not certify an Adapter-only result until the ambiguity is
resolved.

## Nine capability checks

For each required category, cite candidate evidence, public policy, public
source, status, and the unresolved question. Skip a category only when the
frozen profile proves it does not apply.

| Category | What must match | Applies to |
|---|---|---|
| `attention` | causal or bidirectional semantics, masks, head/KV geometry, scaling, external context, and returned layout | all families |
| `cache` | prefill/decode or recurrent-state reads and writes, capacity, logical position, reset/rewind behavior, and ownership | CausalLM, VLM, VLA |
| `position` | rotary, multimodal, absolute, or bias semantics; position construction; padding behavior; Graph visibility | any profile with position-dependent math |
| `norm` | normalization type, epsilon, learned-weight convention, placement, dtype, and residual ordering | all families |
| `mlp` | dense or gated projections, activation, bias, expert/routing behavior, and residual ordering | all families |
| `linear_weights` | projection shapes, tied weights, bias, quantization metadata, patch-embedding conversion, and exactly-once layout transformation | all families |
| `multimodal_glue` | component ownership, projector/merger placement, image-feature count, image/text composition, and state/action inputs | VLM and VLA |
| `logits_generate` | output head, cache return, generation mode, sampling boundary, output post-processing, and public return type | CausalLM, VLM, generative VLA components |
| `runtime_graph` | exact-profile admission, SPM plan, Graph signature, DMA address ownership, output lifetime, component handoff, and teardown | all families |

Use the public [adapter registry and plugins](../../../../docs/api_reference.md#adapter-registry-and-plugins),
[Graph API](../../../../docs/api_reference.md#graph-api), and checked-in adapter
source as evidence. A generic CPU fallback is never evidence that the category
runs on RPU.

## Review-trigger screen

These signals require a focused check before choosing a port class. A signal is
not automatically supported or rejected; it prevents reuse of a nearby profile
without exact public evidence.

| Signal | Identification hint | Required assessment action |
|---|---|---|
| Sparse experts or routing | nonzero expert counts, router/gate plus top-k selection, expert collections, data-dependent dispatch | Trace routing and every expert operation; use an existing path only if its exact contract is public, otherwise classify the missing work or mark Blocked. |
| Recurrent or scan-style state | model configuration or forward carries recurrent tensors, scan/update operations, or convolutional state | Trace initialization, per-step update, reset, and returned state; do not map it to a stateless dense decoder. |
| True sliding attention | configured window is smaller than the admitted sequence and changes visible cache history | Prove mask and cache-visibility semantics for the maximum envelope; full attention is not a substitute. |
| Encoder-decoder or cross-attention | `is_encoder_decoder`, `encoder_hidden_states`, or a separate cross-attention block | Require an exact external-key/value attention path and ownership contract; causal self-attention is not equivalent. |
| Additive position bias | ALiBi, learned attention bias, slopes, or another bias injected into attention scores | Verify the public attention path accepts and applies the exact bias; rotary position support does not imply this support. |
| Batch requirement beyond the profile | the reference forward requires more examples than the admitted public profile | Check the exact model entry and state/cache/output ownership. Generic batch submission does not widen a model profile. |
| Different primary precision | checkpoint or reference requires a dtype outside the admitted profile, or profile inputs overflow at the selected dtype | Require a reviewed precision path and profile-specific numerical evidence; casting is not a correctness argument. |
| Non-cache generation | generation disables or omits incremental state and recomputes a different forward lifecycle | Trace that lifecycle independently; do not reuse a cache-oriented route by approximation. |
| Training or loss | backward, optimizer, gradient state, or a required training/loss return | Stop as Blocked: RhinoForge's public scope is inference only. |

## Explicit CPU glue boundaries

CPU glue is allowed only when the exact public profile names the boundary, the
math remains identical to the upstream reference, and the boundary is included
in end-to-end verification. It preserves correctness but does not count as RPU
execution evidence.

| Boundary | Publicly reviewable contract |
|---|---|
| Image/text composition | Validate placeholder and feature counts, then perform the exact composition once. See the [Qwen3-VL adapter](../../../../python/rpu_backend/adapters/qwen3_vl/__init__.py). |
| Multimodal position construction | Use the upstream-equivalent position helper and treat its result as semantic Graph input. See [multimodal positions](../../../../knowledge/concepts/mrope.md). |
| Patch-embedding preparation | An exact convolution-to-linear weight fold or a checked host patch embed may remain outside the fused encoder when the profile documents it. See the [DINOv3 adapter](../../../../python/rpu_backend/adapters/dinov3.py). |
| Exact scalar feature adjustment | A model-defined scale or normalization between components may remain host glue only when it is applied exactly once and verified at that component boundary. |
| Iterative policy update | A model-defined host update such as `x_t = x_t + dt * v_t` may remain explicit host math when the profile fixes its precision and verifies every step plus the final action. See the [Pi0.5 runtime](../../../../python/rpu_backend/adapters/pi05/runtime.py). |

Any other fallback remains a capability gap until public policy and the exact
profile admit it. Follow [operators and CPU fallback](../../../../docs/architecture.md#operators-and-cpu-fallback)
and report every fallback in the assessment.
