# Capability assessment reference

Use this reference only for assessment mode. The canonical phase definitions
remain in [the capability review](../../../../docs/model_porting_capability_review.md).

## Evidence rules

- Freeze one exact model profile, not a family name.
- Define numerical evidence through the public
  [validation policy](../../../../docs/validation_policy.md); do not substitute
  a single global cosine minimum for semantic and task gates.
- Close the source set needed to trace preprocessing, the complete forward,
  cache or recurrent state, and returned outputs. Missing source makes the
  assessment uncertified.
- Treat a requirement as supported only when public RhinoForge policy permits
  it and public source implements the exact tensor and lifecycle contract.
- Candidate model documentation describes the requirement; it is not evidence
  that RhinoForge implements it.
- Record failed or incomplete probes as `needs human review`. Do not silently
  drop them or convert them into confident support or rejection.
- CPU fallback preserves correctness only when the public profile explicitly
  admits that boundary. It is not evidence of RPU execution.

## Family and capability map

Classify the profile before checking individual operations:

| Family | Required trace |
|---|---|
| Decoder CausalLM | attention, position handling, KV cache, normalization, MLP, logits, generation, Graph lifecycle |
| Vision encoder | patch or image embedding, bidirectional attention, position handling, normalization, MLP, returned features, Graph lifecycle |
| VLM | complete vision path, projection or merger ownership, image/text composition, decoder and cache, returned logits or embeddings |
| VLA | complete VLM path plus state/action inputs, action expert, iterative update rule, returned action and policy state |

For every traced step, record one mapping: existing adapter, shared runtime,
existing operation, admitted CPU boundary, runtime extension, new fused
subsystem, asset change, or blocker. Use the public registrations and source;
source-file presence alone is not a capability claim.

## Report contract

Write a compact report with these sections:

1. exact candidate identity, revision, configuration hashes, precision, and
   input envelope;
2. source closure and missing sources;
3. semantic blockers;
4. operation and lifecycle gap table with candidate requirement, public policy
   evidence, source evidence, status, and unresolved review item;
5. SPM, alignment, dtype, Graph, DMA, cache, output-lifetime, and asset fit;
6. verification gates defined before implementation; and
7. outcome and certification.

Use exactly one outcome: `Existing path`, `Adapter-only`, `Runtime extension`,
`New fused subsystem`, or `Blocked`. Record certification independently as
`certified` or `uncertified`; a plausible port with missing evidence remains
uncertified. Only a certified, non-Blocked outcome may enter port mode.

## Portable independent review

When the host supports isolated reviewers or subagents, separate the assessment
from certification:

1. the primary reviewer traces requirements and produces the evidence table;
2. a second reviewer receives the frozen candidate identity, source set, and
   evidence table, but not the draft outcome, and tries to refute every
   `supported` row; and
3. the final synthesis resolves each challenge and records any remaining
   conflict or missing source.

Certification is `certified` only when source closure is complete, every
required capability has both policy and source evidence, and all independent
challenges are resolved. If the host cannot provide an independent reviewer,
or the evidence cannot be isolated, report `uncertified`; do not simulate
independence by asking the same context to approve its own verdict. This
protocol is tool-neutral and does not require a particular agent runtime.
