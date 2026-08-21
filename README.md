<div align="center">

# RhinoForge

**PyTorch inference and model deployment for the Rhino Processing Unit**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

**English** | [简体中文](README_CN.md)

</div>

RhinoForge exposes the Rhino Processing Unit (RPU) as `torch.rpu`. It combines
model adapters, fused model runtimes, graph capture and replay, SPM memory
management, multi-core execution, offline quantization, and a model-porting
template in one PyTorch backend.

The distribution name is `rhinoforge`; the Python import remains
`rpu_backend`. RhinoForge is inference-only and does not support training or
fine-tuning.

## Highlights

- **PyTorch integration:** use the RPU through the standard custom-device path
  and `torch.rpu` APIs.
- **Model deployment:** run text, vision-language, vision, and VLA profiles
  through model-specific adapters and TOML examples.
- **Extensible runtime:** reuse the public fused-model, graph, SPM, multi-core,
  batch, and adapter mechanisms when porting another model.
- **Profile-scoped support:** every status is tied to an exact checkpoint,
  precision, input envelope, configuration, and runtime asset set.

## AI-assisted model porting

The checked-in [`rhinoforge-port` skill](.agents/skills/rhinoforge-port/SKILL.md)
supports two modes. **Assess** produces a read-only capability review with an
outcome and independent certification; **Port** starts only from an accepted,
certified, non-Blocked assessment and loads the applicable public family
playbook. See [Model porting](docs/model_porting.md) for the canonical workflow.

## Model matrix

> **Release-candidate status:** the table below is the planned release status.
> `Supported` and `Limited` become release claims only after final-package
> validation on one immutable software and asset set. That validation is still
> pending for this source candidate.

| Model | Public entry | Status | Limits |
|---|---|---|---|
| Qwen3 0.6B / 1.7B / 4B / 8B | `RPUModelForCausalLM` | Supported | FP16 inference; exact execution envelope is release-specific |
| Qwen3 0.6B / 1.7B / 4B / 8B W8A16 | `RPUModelForCausalLM` | Source-only | Converter and loader source are included without a runnable release claim |
| Qwen3 14B | `RPUModelForCausalLM` | Limited | Exact W8A16 profile with untied INT8 `lm_head` and FP16 embeddings only |
| Qwen3 32B | `RPUModelForCausalLM` | Source-only | No runnable profile; preflight rejects unsupported configurations |
| Llama-3.2-1B | `RPUModelForCausalLM` | Supported | Causal language-model inference |
| Qwen3.5 text 0.8B / 2B / 4B / 9B | `Qwen3_5Adapter` | Supported | Dense FP16, text only; `torch.compile` is not supported |
| Qwen3.5 Vision 2B / 4B | `Qwen3_5Adapter` | Experimental | Controlled evaluation only; default fail-fast remains enabled |
| Qwen3-VL 2B | `RPUModelForConditionalGeneration` | Supported | Image and text; video is outside the release scope |
| Qwen3-VL 4B | `RPUModelForConditionalGeneration` | Limited | Only release-named single- and multi-image envelopes |
| Qwen3-VL 32B W8A16 | `RPUModelForConditionalGeneration` | Experimental | Controlled evaluation; no full support verdict |
| Gemma4-E4B text | `Gemma4Adapter` | Source-only | Runtime dependency and release profile are not yet frozen |
| Pi0.5 Libero FP16 | `Pi05Policy` | Supported | Exact Libero profile |
| Pi0.5 Libero W8A16 | `Pi05Policy` | Limited | Exact direct-inference profile |
| Pi0.5 W4 / reduced-step / closed-loop | `Pi05Policy` | Experimental | Each profile requires separate validation |
| Wall-OSS-0.5 | `WallOssPolicy` | Limited | Only release-named camera, prefix, and precision profiles |
| Wall-OSS RTC | `WallOssPolicy` | Experimental | Controlled evaluation for exact named profiles |
| Wall-OSS W4 and other quantized profiles | `WallOssPolicy` | Experimental | Evaluation only; status is profile-specific |
| Hy-Embodied-0.5-VLA | `HyEmbodiedPolicy` | Limited | Exact input profile; numerical inference is not robot-readiness certification |
| DINOv3 ViT-B | `DINOv3Adapter` | Supported | ViT-B only |
| SigLIP | `rpu_backend.adapters.siglip` | Component-only | Vision encoder component only |
| GR00T-N1.7-3B | `build_gr00t_vla` | Source-only | Public asset and end-to-end setup are not yet complete |
| [RhinoVLA](https://github.com/HUIXI-AI/RhinoVLA) | `RhinoVLAPolicy` | Source-only | Integration candidate; the model repository owns checkpoint composition and preprocessing |
| LingBot-VLA-V2 exact-Z2 W8A16 | `Lingbot2Policy` | Experimental | Controlled evaluation only |
| Galaxea G0.5 exact K1 RTC/FM W0 | `patch_g05_policy_for_rpu` | Experimental | Exact K1 W0 profile only |
| InternVLA-N1 + NavDP | `rpu_backend.adapters.internvla_n1` / `navdp` | Experimental | Exact asset manifest and controlled evaluation required |

See [Model support](docs/model_support.md) for status definitions, exact public
entries, registry aliases, and excluded profiles.

## Quick start

The first release is installed from source. It requires a preconfigured RPU
board environment, Python 3.12, the Rhino Launch 1.0.0 binary development
package, and one compatible combined operator asset named
`rhinoOpLib_current.ref`. Rhino Launch and the operator asset are distributed
separately through authorized channels and are not included in this repository.
Before building, follow [Restricted runtime assets](docs/runtime_assets.md) to
select one active compatibility set and verify its signed BOM and status.

If Rhino Launch is outside the system CMake search path, expose its prefix,
then build and verify RhinoForge:

```bash
export CMAKE_PREFIX_PATH="/opt/rhino-launch${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
python -m pip install . --no-build-isolation

export RPU_KERNEL_LIB_PATH="$HOME/.local/share/rhinoforge/rhinoOpLib_current.ref"
python examples/verify_install.py
```

Obtain Qwen3-0.6B from the source listed in
[Model assets](docs/model_assets.md), copy its TOML profile, and set the local
checkpoint path. A published release manifest pins the exact revision and
hashes used for release validation.

```bash
export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
cp examples/configs/qwen3_0_6b.toml qwen3.local.toml
# Edit qwen3.local.toml to use the downloaded checkpoint.
python examples/causal_lm.py --config qwen3.local.toml --check-config
python examples/causal_lm.py --config qwen3.local.toml
```

Model checkpoints and generated quantized checkpoints are not distributed by
RhinoForge. Read [Getting started](docs/getting_started.md) for the complete
installation, asset verification, model download, quantization, and inference
flow. The checked-in [examples](examples/) provide concise TOML-driven entry
points; [Model execution and profiling](docs/model_testing.md) documents the
full TOML runner, Torch profiler, hardware profiler, and runnable or scoped
model entry paths.

## Documentation

| Topic | Document |
|---|---|
| Install and run | [Getting started](docs/getting_started.md) |
| Supported and excluded profiles | [Model support](docs/model_support.md) |
| Restricted Launch and operator assets | [Restricted runtime assets](docs/runtime_assets.md) |
| Model checkpoints | [Model assets](docs/model_assets.md) |
| Runtime environment variables and profiling | [Runtime configuration](docs/runtime_config.md) |
| Model execution, testing, and profiling | [Model execution and profiling](docs/model_testing.md) |
| Reproducible performance measurement | [Performance measurement](docs/performance.md) |
| Model validation and status gates | [Model validation policy](docs/validation_policy.md) |
| Runtime design | [Architecture](docs/architecture.md) |
| Python and C++ APIs | [API reference](docs/api_reference.md) |
| Port another model | [Model porting](docs/model_porting.md) |
| Offline quantization | [Quantization](docs/quantization.md) |
| Report a vulnerability | [Security policy](SECURITY.md) |
| Contribute | [Contributing](CONTRIBUTING.md) |

## License

RhinoForge source code is licensed under the
[Apache License 2.0](LICENSE). Model checkpoints, Rhino Launch, the combined
operator asset, and other third-party materials are governed by their own terms
and are not licensed by this repository's `LICENSE` file.
