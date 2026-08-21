# Model support

Model support is defined for an exact profile, not just a model family. A
checkpoint revision, input envelope, precision, execution configuration, and
runtime asset set together form that profile. Status does not transfer between
model sizes or quantization formats.

> **Candidate notice:** every status below is the planned release status. The
> final release must bind the model table, examples, hashes, and validation
> results to the same immutable package and asset set. Until that is complete,
> `Supported` and `Limited` are not signed release claims.

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

## Planned Supported and Limited profiles

| Model/profile | Public entry | Status | Exact scope |
|---|---|---|---|
| Qwen3 0.6B / 1.7B / 4B / 8B | `rpu_backend.RPUModelForCausalLM` | Supported | FP16 inference; release-specific batch, prefill, and decode envelope |
| Qwen3 14B | `rpu_backend.RPUModelForCausalLM` | Limited | Exact W8A16 profile with an untied INT8 `lm_head` and FP16 embeddings; FP16 is not included |
| Llama-3.2-1B | `rpu_backend.RPUModelForCausalLM` | Supported | Causal language-model inference |
| Qwen3.5 text 0.8B / 2B / 4B / 9B | `rpu_backend.adapters.qwen3_5.Qwen3_5Adapter` with `rpu_backend.api.Qwen3_5Cache` | Supported | Dense FP16, text only; `torch.compile` is not supported |
| Qwen3-VL 2B | `rpu_backend.api.RPUModelForConditionalGeneration` | Supported | Image and text; video is outside the first-release scope |
| Qwen3-VL 4B | `rpu_backend.api.RPUModelForConditionalGeneration` | Limited | Only single- and multi-image envelopes named in the final release manifest |
| Pi0.5 Libero FP16 | `rpu_backend.api.Pi05Policy` | Supported | Exact Libero FP16 profile |
| Pi0.5 Libero W8A16 | `rpu_backend.api.Pi05Policy` | Limited | Only the final-release direct-inference profile |
| Wall-OSS-0.5 | `rpu_backend.api.WallOssPolicy` | Limited | Only camera, prefix, precision, and execution values named in the final release manifest |
| Hy-Embodied-0.5-VLA | `rpu_backend.api.HyEmbodiedPolicy` | Limited | Exact input profile; numerical inference support is not robot-readiness certification |
| DINOv3 ViT-B | `rpu_backend.adapters.dinov3.DINOv3Adapter` | Supported | ViT-B only |
| SigLIP | `rpu_backend.adapters.siglip.patch_siglip_model_for_rpu_all_layers_once` | Component-only | Vision encoder component only |

## Planned Experimental, Component-only, and Source-only profiles

| Model/profile | Public entry | Status | Exact scope |
|---|---|---|---|
| Qwen3 0.6B / 1.7B / 4B / 8B W8A16 | `rpu_backend.RPUModelForCausalLM` | Source-only | Converter and loader source are included, but no quantized runnable profile or registry alias is released |
| Qwen3 32B | `rpu_backend.RPUModelForCausalLM` | Source-only | Known unsupported configuration; preflight remains fail-fast |
| Qwen3.5 Vision 2B / 4B | `rpu_backend.adapters.qwen3_5.Qwen3_5Adapter` | Experimental | Named controlled evaluation only; default fail-fast remains enabled and broader image or video inputs are not covered |
| Qwen3-VL 32B W8A16 | `rpu_backend.api.RPUModelForConditionalGeneration` | Experimental | Controlled evaluation; asset and host preflight do not constitute a complete numerical or graph verdict |
| Pi0.5 W4 / reduced-step / closed-loop | `rpu_backend.api.Pi05Policy` | Experimental | Each precision and execution mode is evaluated independently |
| Wall-OSS RTC | `rpu_backend.api.WallOssPolicy` | Experimental | Only exact controlled profiles named by the release are eligible |
| Wall-OSS W4 and other quantized profiles | `rpu_backend.api.WallOssPolicy` | Experimental | Evaluation only; results do not transfer from FP16 or W8A16 |
| GR00T-N1.7-3B | `rpu_backend.adapters.gr00t.build_gr00t_vla` | Source-only | Public loader, checkpoint, TOML, and end-to-end asset flow are not yet complete |
| Gemma4-E4B text | `rpu_backend.adapters.gemma4.Gemma4Adapter` | Source-only | Dependency compatibility is resolved; checkpoint, runtime-asset, and exact-profile validation remain pending |
| RhinoVLA | `rpu_backend.api.RhinoVLAPolicy` | Source-only | Integration candidate; checkpoint composition, preprocessing, and the live-input contract are owned by the separate model repository |
| LingBot-VLA-V2 exact-Z2 W8A16 | `rpu_backend.api.Lingbot2Policy` | Experimental | Exact controlled profile only; other precisions and generic or robot use are not included |
| Galaxea G0.5 exact K1 RTC/FM W0 | `rpu_backend.adapters.g05.patch_g05_policy_for_rpu` | Experimental | Exact K1 W0 controlled profile only |
| InternVLA-N1 + NavDP | `rpu_backend.adapters.internvla_n1` and `rpu_backend.adapters.navdp` | Experimental | Exact caller-pinned asset manifest and controlled evaluation are required |

## Unsupported and excluded profiles

This is an exclusion list, not a runnable model matrix. These profiles are
intentionally absent from runnable aliases and examples:

| Profile | Status | Public entry |
|---|---|---|
| Gemma2 and PaliGemma2 | Unsupported | None |
| DINOv3 ViT-S and ViT-L | Unsupported | None; DINOv3 ViT-B remains in scope |
| LingBot-VLA v1 | Unsupported | None; LingBot-VLA-V2 remains only in the scope listed above |
| Qwen3-VL 8B | Unsupported | None; shared Qwen3-VL code remains for the 2B and 4B profiles |
| Training, fine-tuning, dataset distribution, and checkpoint redistribution | Unsupported | None |

## Registry alias classification

The registry only maps a stable name to a local cache directory. Each retained
alias is nevertheless assigned to a release profile so a path alias cannot be
mistaken for an extra support claim.

| Registry alias(es) | Family/profile | Status | Classification |
|---|---|---|---|
| `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b` | Qwen3 FP16 | Supported | Asset aliases for the exact size named by each alias |
| `qwen3-14b-w8a16-lmhead-int8` | Qwen3 14B W8A16 | Limited | Exact untied INT8-`lm_head`, FP16-embedding asset alias; no FP16 inheritance |
| `qwen3_5-0.8b`, `qwen3_5-2b`, `qwen3_5-4b`, `qwen3_5-9b` | Qwen3.5 text FP16 | Supported | Asset aliases for the exact size named by each alias |
| `qwen3-vl-2b` | Qwen3-VL 2B Instruct | Supported | Asset alias |
| `qwen3-vl-4b` | Qwen3-VL 4B Instruct | Limited | Asset alias for the named image envelope only |
| `llama-3.2-1b` | Llama-3.2-1B | Supported | Asset alias |
| `pi05-libero-finetuned` | Pi0.5 Libero FP16 | Supported | Asset alias |
| `pi05-libero-finetuned-w8a16-vlm-expert` | Pi0.5 Libero W8A16 | Limited | Quantized asset alias |
| `pi05-libero-finetuned-w4real-kvint8` | Pi0.5 Libero W4 evaluation | Experimental | Quantized asset alias |
| `pi05-base` | Pi0.5 base | Source-only | Asset alias with no independent task-profile support claim |
| `wall-oss-0.5`, `wall-oss-0.5-w8a16` | Wall-OSS-0.5 FP16 / W8A16 | Limited | Asset aliases; only final-manifest envelopes qualify |
| `wall-oss-0.5-w4a16`, `wall-oss-0.5-w4a16-pgrp` | Wall-OSS-0.5 W4 | Experimental | Quantized asset aliases |
| `dinov3-vit-b` | DINOv3 ViT-B | Supported | Asset alias; ViT-S/L remain absent |
| `hy-embodied-0.5-vla-umi` | Hy-Embodied-0.5-VLA | Limited | Asset alias |
| `g05-base` | Galaxea G0.5 | Experimental | Controlled-profile asset alias |
| `lingbot-vla-v2-6b` | LingBot-VLA-V2 | Experimental | Controlled-profile asset alias |
| `internvla-n1-navdp` | InternVLA-N1 + NavDP | Experimental | Caller-manifest-bound asset alias |
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
