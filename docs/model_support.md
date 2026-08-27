# Model support

[简体中文](model_support.zh.md) | English

Model support is defined for an exact profile, not just a model family. A
checkpoint revision, input envelope, precision, execution configuration, and
runtime asset set together form that profile. Status does not transfer between
model sizes or quantization formats.

> **v1.0.0 release:** the statuses below are the current release claims. They
> apply only to the immutable public identities in
> [`release/public-models-v1.0.0.json`](../release/public-models-v1.0.0.json)
> and the matching v1.0.0 runtime set. A generated or neighboring checkpoint
> does not inherit a status.

## Status definitions

| Status | Meaning |
|---|---|
| Supported | The exact public profile passes checkpoint/input, numerical, warmup, graph-lifecycle, and end-to-end validation |
| Limited | The same quality bar as Supported passes, but only for the exact named, narrower envelope; nearby profiles do not inherit the result |
| Experimental | Hard semantic, required finite-output, declared-repeatability, and safe rejection gates pass, while parity coverage or representative task evidence may remain incomplete |
| Component-only | Only the named component API and component validation are provided, not a complete product model |
| Source-only | Source is included without a runnable support commitment; known-unsupported configurations fail before model mutation |
| Unsupported | The profile is not registered as runnable and has no dedicated public runtime; shared loaders fail fast when applicable |

See [Model validation policy](validation_policy.md) for the independent hard,
implementation-parity, task-quality, and runtime-lifecycle gates. `Limited` is
not a waiver for lower numerical quality, and `Experimental` does not by itself
mean that a model is numerically wrong.

## Supported and Limited profiles

| Model/profile | Public entry | Status | Exact scope |
|---|---|---|---|
| Qwen3 0.6B / 1.7B / 4B / 8B | `rpu_backend.RPUModelForCausalLM` | Supported | FP16 inference; release-specific batch, prefill, and decode envelope |
| Llama-3.2-1B | `rpu_backend.RPUModelForCausalLM` | Supported | Causal language-model inference |
| Qwen3.5 text 2B / 9B | `rpu_backend.adapters.qwen3_5.Qwen3_5Adapter` with `rpu_backend.api.Qwen3_5Cache` | Supported | Dense FP16, text only; `torch.compile` is not supported |
| Qwen3-VL 2B | `rpu_backend.api.RPUModelForConditionalGeneration` | Supported | Image and text; video is outside the first-release scope |
| Qwen3-VL 4B | `rpu_backend.api.RPUModelForConditionalGeneration` | Supported | Within the validated single-image, multi-image, prefill, decode, and chunk envelope only |
| Qwen3-VL 8B | `rpu_backend.api.RPUModelForConditionalGeneration` | Limited | Exact dense-FP16 padded profile: 4/4 numeric/task/Graph legs pass, including dual-full; base 5/5 and long-text 9/9 tokens are exact; nearby lengths, video, and quantized paths do not inherit |
| Pi0.5 Libero FP16 | `rpu_backend.api.Pi05Policy` | Supported | Exact Libero FP16 profile: 9/9 numeric/task legs and 27/27 execution-plan records pass; an independent fresh-process Graph lifecycle cell also passes |
| Hy-Embodied-0.5-VLA | `rpu_backend.api.HyEmbodiedPolicy` | Limited | Exact FP16/W16 input profile only; numerical inference support is not robot-readiness certification |
| DINOv3 ViT-B | `rpu_backend.adapters.dinov3.DINOv3Adapter` | Supported | ViT-B only |

## Experimental, Component-only, and Source-only profiles

| Model/profile | Public entry | Status | Exact scope |
|---|---|---|---|
| Qwen3 W8A16 (0.6B / 1.7B / 4B / 8B / 14B) | `rpu_backend.RPUModelForCausalLM` | Source-only | Converter and loader source are included; v1.0.0 binds no public immutable derived checkpoint identity or hash, and aliases resolve local outputs only |
| Qwen3 32B | `rpu_backend.RPUModelForCausalLM` | Source-only | Known unsupported configuration; preflight remains fail-fast |
| Qwen3.5 text 0.8B / 4B | `rpu_backend.adapters.qwen3_5.Qwen3_5Adapter` with `rpu_backend.api.Qwen3_5Cache` | Experimental | The 192-token validation's last-prefill full-vocabulary relative-L2 gate exceeds `0.01`; these profiles are numeric-blocked and controlled evaluation only |
| Qwen3.5 Vision 2B / 4B | `rpu_backend.adapters.qwen3_5.Qwen3_5Adapter` | Experimental | Official real-image standalone gates fail for both sizes; 2B real-image end-to-end also fails and 4B real-image end-to-end remains unrun. Image inference is fail-closed by default; exact `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1` enables controlled evaluation only |
| Qwen3-VL 32B W8A16 | `rpu_backend.api.RPUModelForConditionalGeneration` | Source-only | Retained Graph replay is known to produce all-zero logits from the third decode step. The default remains fail-closed; the explicit opt-in is only for controlled diagnostics and does not pass the hard Graph/lifecycle gate |
| Pi0.5 Libero W8A16 / W4A16 | `rpu_backend.api.Pi05Policy` | Source-only | Converter and runtime source are included; v1.0.0 binds no public immutable derived checkpoint identity or hash |
| Pi0.5 reduced-step / closed-loop | `rpu_backend.api.Pi05Policy` | Source-only | Policy source only; validation remains pending and these execution profiles do not inherit evidence or status from the exact FP16 or quantized paths |
| Wall-OSS-0.5 | `rpu_backend.api.WallOssPolicy` | Source-only | Public `wall-x` integration only; the pinned public checkpoint has not completed the release validation gates |
| Wall-OSS quantized profiles | `rpu_backend.api.WallOssPolicy` | Source-only | Public converters are included; each derived asset still needs converter provenance, output identity, and fresh validation |
| SigLIP | `rpu_backend.adapters.siglip.patch_siglip_model_for_rpu_all_layers_once` | Component-only | Vision encoder component only |
| GR00T-N1.7-3B | `rpu_backend.adapters.gr00t.build_gr00t_vla` | Source-only | Public loader, checkpoint, TOML, and end-to-end asset flow are not yet complete |
| Gemma4-E4B text | `rpu_backend.adapters.gemma4.Gemma4Adapter` | Source-only | Dependency compatibility is resolved; checkpoint, runtime-asset, and exact-profile validation remain pending |
| RhinoVLA | `rpu_backend.api.RhinoVLAPolicy` | Source-only | Integration candidate; checkpoint composition, preprocessing, and the live-input contract are owned by the separate model repository |
| LingBot-VLA-V2 | `rpu_backend.api.Lingbot2Policy` | Source-only | Public upstream integration; the pinned public checkpoint has not completed the release validation gates |
| Galaxea G0.5 continuous | `rpu_backend.adapters.g05.patch_g05_policy_for_rpu` | Source-only | Public `G05PolicyQwen35` continuous-policy integration only; other upstream policy modes fail before weight mutation |
| InternVLA-N1 + NavDP | `rpu_backend.adapters.internvla_n1` and `rpu_backend.adapters.navdp` | Source-only | Public upstream integration; exact public asset closure and validation remain pending |
| Hy-Embodied W8/W4 | `rpu_backend.api.HyEmbodiedPolicy` | Source-only | Runtime conversion source is included; v1.0.0 binds no public immutable derived checkpoint identity or hash, and FP16/W16 evidence does not transfer |

## Profile-specific validation contracts

Validation has three independent layers: hard semantics plus Graph/runtime
lifecycle, same-semantics implementation parity, and task or end-to-end quality
on the consumed output. A strong aggregate cosine cannot waive a hard failure,
and no repository-wide cosine threshold is applied to every model. Quantized
profiles first compare against the same quantization semantics, then against an
FP16 or FP32 task-quality anchor.

The v1.0.0 release record binds each claimed profile to its reference identity,
inputs, metrics, and thresholds. The current contracts are:

| Profile | Hard semantics and lifecycle | Implementation parity | Task or release-quality evidence |
|---|---|---|---|
| Qwen3 FP16 | Valid positions, cache slots, warmup `0/1/3`, long-context bounds, and BUILD followed by stable REPLAY | CPU-FP32 versus RPU per-position cosine `>=0.99995`; every non-exact greedy near-tie is reported | Frozen perplexity/task accuracy and fixed-prompt generation for each named size and envelope |
| Qwen3-VL 2B / 4B | Image/text ordering, corrected DeepStack routing, single- and multi-image inputs, and stable Vision/prefill/decode Graph lifecycles | Consumed text rows require mean cosine `>0.99` for single-image and `>0.98` for dual-image cases, with exact final argmax and ordered top-5; image-token metrics remain separate diagnostics | Frozen image-conditioned and multi-image task cases over only the named prompt, decode, and chunk envelope |
| Qwen3-VL 8B dense FP16 | The same semantic and lifecycle gates as 2B/4B, plus exact padded-profile shape and physical-padding preflight; all 4/4 named legs pass their Graph lifecycle | All 4/4 named legs pass calibrated numeric and task gates, including dual-full; exact-token results are base 5/5 and long-text 9/9 | The Limited claim covers only the exact padded profile; nearby lengths, video, and quantized paths do not inherit |
| Qwen3.5 text 2B / 9B | Prefill uses its declared bounded one-shot and decode uses stable REPLAY | Text/decode cosine `>=0.999` and last-prefill full-vocabulary relative L2 `<=0.01`; accepted same/adjacent-FP16-bin near-ties are marked `greedy_exact=false` | Frozen dense-FP16 text cases pass for the named sizes |
| Qwen3.5 text 0.8B / 4B | The same bounded-one-shot prefill and stable-REPLAY decode lifecycle is exercised | The 192-token validation's last-prefill full-vocabulary relative L2 exceeds `0.01` | Numeric-blocked; controlled evaluation does not constitute release support |
| Qwen3.5 Vision 2B / 4B | Default image admission fails closed before the Vision handle or weight transformation; controlled evaluation uses a bounded retained cache and accepts images only | The declared row-mean cosine `>=0.999`, loose minimum `>=0.90`, and maximum relative error `<0.40` gate is not met by official real images for either size | No production-safe normal-image envelope is certified; controlled results do not constitute image or video support |
| Pi0.5 Libero FP16 | Exact action shape/dtype, finite output, multi-camera and length behavior, and 27/27 action/prefill/vision execution-plan records; an independent fresh process proves READY stable Graph replay without recapture or invariant failure | All 9/9 same-noise numeric legs pass the source MSE ceiling `0.0072` | The exact 9/9 action/task matrix and independent Graph cell pass; quantized and reduced-step profiles do not inherit |
| Hy-Embodied FP16/W16 | Exact profile admission, finite outputs, four-call repeatability spread `<=1e-4`, stable Graph lifecycle, and A/B/A input refresh | Final action cosine `>=0.999` and MSE `<=0.01`; named intermediate stages use their separately frozen `0.99995` or `0.999` floors | Checkpoint-owned offline action evidence; robot safety and W8/W4 are outside the claim |

See [Model validation policy](validation_policy.md) for reference identity,
near-zero rows, quantized-reference ordering, task-metric retention, and status
decisions. An unrun gate remains pending and is never inferred from another
model size or precision.

## Unsupported and excluded profiles

This is an exclusion list, not a runnable model matrix. These profiles are
intentionally absent from runnable aliases and examples:

| Profile | Status | Public entry |
|---|---|---|
| Gemma2 and PaliGemma2 | Unsupported | None |
| DINOv3 ViT-S and ViT-L | Unsupported | None; DINOv3 ViT-B remains in scope |
| LingBot-VLA v1 | Unsupported | None; LingBot-VLA-V2 remains only in the scope listed above |
| Training, fine-tuning, dataset distribution, and checkpoint redistribution | Unsupported | None |

## Registry alias classification

The registry only maps a stable name to a local cache directory. Each retained
alias is nevertheless assigned to a release profile so a path alias cannot be
mistaken for an extra support claim.

| Registry alias(es) | Family/profile | Status | Classification |
|---|---|---|---|
| `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b` | Qwen3 FP16 | Supported | Asset aliases for the exact size named by each alias |
| `qwen3-14b-w8a16-lmhead-int8` | Qwen3 14B W8A16 | Source-only | Local conversion-output alias; v1.0.0 binds no public immutable derived checkpoint identity or hash |
| `qwen3_5-2b`, `qwen3_5-9b` | Qwen3.5 text FP16 | Supported | Asset aliases for the exact size named by each alias |
| `qwen3_5-0.8b`, `qwen3_5-4b` | Qwen3.5 text FP16 | Experimental | Asset aliases for the exact numeric-blocked size; alias resolution does not imply release support |
| `qwen3-vl-2b` | Qwen3-VL 2B Instruct | Supported | Asset alias |
| `qwen3-vl-4b` | Qwen3-VL 4B Instruct | Supported | Asset alias for the validated image, prefill, decode, and chunk envelope only |
| `qwen3-vl-8b` | Qwen3-VL 8B Instruct | Limited | Asset alias for the exact dense-FP16 padded profile whose 4/4 numeric/task/Graph legs include dual-full; nearby lengths, video, and quantized paths do not inherit |
| `llama-3.2-1b` | Llama-3.2-1B | Supported | Asset alias |
| `pi05-libero-finetuned` | Pi0.5 Libero FP16 | Supported | Asset alias |
| `pi05-libero-finetuned-w8a16-vlm-expert` | Pi0.5 Libero W8A16 | Source-only | Local conversion-output alias; no public immutable derived checkpoint identity or hash is bound |
| `pi05-libero-finetuned-w4real-kvint8` | Pi0.5 Libero W4A16 | Source-only | Local conversion-output alias; no public immutable derived checkpoint identity or hash is bound |
| `pi05-base` | Pi0.5 base | Source-only | Asset alias with no independent task-profile support claim |
| `wall-oss-0.5`, `wall-oss-0.5-w8a16`, `wall-oss-0.5-w4a16`, `wall-oss-0.5-w4a16-pgrp` | Wall-OSS-0.5 | Source-only | Public source/convert paths; aliases do not establish derived-asset identity or support |
| `dinov3-vit-b` | DINOv3 ViT-B | Supported | Asset alias; ViT-S/L remain absent |
| `hy-embodied-0.5-vla-umi` | Hy-Embodied-0.5-VLA | Limited | Public FP16/W16 asset alias; runtime-derived W8/W4 paths remain Source-only |
| `g05-base` | Galaxea G0.5 | Source-only | Public upstream asset alias; access and non-commercial terms apply |
| `lingbot-vla-v2-6b` | LingBot-VLA-V2 | Source-only | Public upstream asset alias |
| `internvla-n1-navdp` | InternVLA-N1 + NavDP | Source-only | Public upstream asset alias |
| `gr00t-n1d7-3b` | GR00T-N1.7-3B | Source-only | Asset alias |
| `gemma4-e4b` | Gemma4-E4B text | Source-only | Asset alias |

The former `qwen3-0.6b-instruct` and `qwen3-4b-instruct` local aliases are not
published because they lack an independently pinned public asset record. A
future release may add them only with an exact source, revision, hash, and
profile classification.

A shared loader that recognizes an excluded configuration must reject it before
loading or modifying model weights.

## General runtime limits

- RhinoForge is an inference backend; training and fine-tuning are not
  supported.
- FP16 is the primary execution dtype. Quantized support is stated separately
  for each exact profile.
- A registry alias resolves a checkpoint path; it is not a support claim.
- Input shape, batch, context, image count, warmup, and graph behavior are part
  of the release profile. Values not named by a release do not inherit its
  status.
- VLA numerical output validation does not certify coordinate frames, actuator
  limits, safety policies, or robot readiness.

Use [Model assets](model_assets.md) for checkpoint and configuration records,
and [Getting started](getting_started.md) for the source-build flow.
