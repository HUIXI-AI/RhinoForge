# Capability assessment report template

Copy this template for one exact model profile. Keep the outcome and evidence
certification independent. Unknown or unavailable evidence stays explicit.

## 1. Candidate identity

| Field | Value |
|---|---|
| Model source and immutable revision | |
| Architecture and family | |
| Configuration and preprocessing hashes | |
| Precision / quantization | |
| Input, cache, image, and action envelope | |
| Returned-output and persistent-state contract | |
| RhinoForge / PyTorch / Launch / operator-asset versions | |

## 2. Source closure

- Candidate files inspected:
- RhinoForge policy and source inspected:
- Missing or inaccessible sources:
- Categories affected by missing evidence:

## 3. Semantic review triggers

Record every applicable trigger from
[the capability map](capability-map.md#review-trigger-screen), the exact
candidate evidence, and whether it is covered, a gap, or blocked.

## 4. Capability table

| Category / requirement | Candidate evidence | Public policy | Public source | Mapping | Status / open question |
|---|---|---|---|---|---|
| | | | | existing adapter / shared runtime / operation / admitted CPU boundary / runtime extension / fused subsystem / asset change / blocker | |

Every required category needs candidate evidence plus both public-policy and
public-source evidence before it can be marked covered.

## 5. Resource and lifecycle fit

- Maximum-envelope SPM and alignment plan:
- Graph signature and BUILD/REPLAY or bounded-one-shot contract:
- Fixed/mutable/immediate DMA ownership:
- Cache, recurrent state, and output-storage ownership:
- Exactly-once weight transformation and process lifetime:
- Runtime and model asset availability:

## 6. Verification contract

List the exact inputs, references, metrics, thresholds, maximum envelope, Graph
lifecycle, warmup/repeatability, multi-input, safe-rejection, and public-entry
checks defined before implementation. Use
[the validation policy](../../../../docs/validation_policy.md); do not replace
the profile contract with one repository-wide cosine threshold.

## 7. Decision

```text
outcome:       Existing path | Adapter-only | Runtime extension | New fused subsystem | Blocked
certification: certified | uncertified
```

- Assumptions:
- Blockers or unresolved evidence:
- Smallest next step:
- Independent-review challenges and resolutions:

Only an accepted, certified, non-Blocked report may enter port mode.
