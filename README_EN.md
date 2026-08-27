<div align="center">

<a href="https://github.com/HUIXI-AI/RhinoVLA"><img src="https://raw.githubusercontent.com/HUIXI-AI/RhinoVLA/fce05e5859104c94544edc1f4380a677c3ebac4c/assets/huixi_logo_cropped.png" alt="Huixi Intelligence" height="72" /></a>

# RhinoForge

**PyTorch inference and model deployment for the Rhino Processing Unit**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

**English** | [简体中文](README.md)

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

> **v1.0.0 release status:** the table below is the current release scope.
> `Supported` and `Limited` apply only to the immutable public model identities
> in [`release/public-models-v1.0.0.json`](release/public-models-v1.0.0.json)
> and the matching v1.0.0 runtime set; generated or neighboring assets do not
> inherit those claims.

| Model | Public entry | Status | Limits |
|---|---|---|---|
| Qwen3 0.6B / 1.7B / 4B / 8B | `RPUModelForCausalLM` | Supported | FP16 inference; exact execution envelope is release-specific |
| Qwen3 0.6B / 1.7B / 4B / 8B W8A16 | `RPUModelForCausalLM` | Source-only | Converter and loader source are included without a runnable release claim |
| Qwen3 14B W8A16 | `RPUModelForCausalLM` | Source-only | Converter and loader source only; v1.0.0 binds no public immutable derived checkpoint identity or hash |
| Qwen3 32B | `RPUModelForCausalLM` | Source-only | No runnable profile; preflight rejects unsupported configurations |
| Llama-3.2-1B | `RPUModelForCausalLM` | Supported | Causal language-model inference |
| Qwen3.5 text 2B / 9B | `Qwen3_5Adapter` | Supported | Dense FP16, text only; `torch.compile` is not supported |
| Qwen3.5 text 0.8B / 4B | `Qwen3_5Adapter` | Experimental | The 192-token validation's last-prefill full-vocabulary relative L2 exceeds `0.01`; numeric-blocked and controlled evaluation only |
| Qwen3.5 Vision 2B / 4B | `Qwen3_5Adapter` | Experimental | Official real-image numerical gates fail; image inference is rejected by default and exact `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1` is controlled evaluation only |
| Qwen3-VL 2B | `RPUModelForConditionalGeneration` | Supported | Image and text; video is outside the release scope |
| Qwen3-VL 4B | `RPUModelForConditionalGeneration` | Supported | Only the validated single-image, multi-image, prefill, decode, and chunk envelope |
| Qwen3-VL 8B | `RPUModelForConditionalGeneration` | Limited | Exact dense-FP16 padded profile passes 4/4 numeric/task/Graph legs, including dual-full; base 5/5 and long-text 9/9 tokens are exact; nearby lengths, video, and quantized paths do not inherit |
| Qwen3-VL 32B W8A16 | `RPUModelForConditionalGeneration` | Source-only | Known all-zero-logit hard failure from the third decode step; explicit opt-in remains for controlled diagnostics only |
| Gemma4-E4B text | `Gemma4Adapter` | Source-only | Runtime dependency and release profile are not yet frozen |
| Pi0.5 Libero FP16 | `Pi05Policy` | Supported | Exact Libero FP16 profile passes 9/9 numeric/task legs and 27/27 execution-plan records; an independent fresh-process Graph lifecycle cell also passes |
| Pi0.5 Libero W8A16 | `Pi05Policy` | Source-only | Converter/runtime source only; v1.0.0 binds no public immutable derived checkpoint identity or hash |
| Pi0.5 Libero W4A16 | `Pi05Policy` | Source-only | Converter/runtime source only; v1.0.0 binds no public immutable derived checkpoint identity or hash |
| Pi0.5 reduced-step / closed-loop | `Pi05Policy` | Source-only | Policy source only; validation remains pending and these profiles do not inherit evidence or status from the exact FP16 or quantized paths |
| Wall-OSS-0.5 | `WallOssPolicy` | Source-only | Public `wall-x` integration; the pinned public checkpoint has not completed the release validation gates |
| Wall-OSS quantized profiles | `WallOssPolicy` | Source-only | Public converters are included; derived checkpoint identity and validation do not inherit from FP16 |
| Hy-Embodied-0.5-VLA FP16/W16 | `HyEmbodiedPolicy` | Limited | Exact public FP16/W16 profile only; numerical inference is not robot-readiness certification |
| Hy-Embodied-0.5-VLA W8/W4 | `HyEmbodiedPolicy` | Source-only | Runtime conversion source only; v1.0.0 binds no public immutable derived checkpoint identity or hash |
| DINOv3 ViT-B | `DINOv3Adapter` | Supported | ViT-B only |
| SigLIP | `rpu_backend.adapters.siglip` | Component-only | Vision encoder component only |
| GR00T-N1.7-3B | `build_gr00t_vla` | Source-only | Public asset and end-to-end setup are not yet complete |
| [RhinoVLA](https://github.com/HUIXI-AI/RhinoVLA) | `RhinoVLAPolicy` | Source-only | Integration candidate; the model repository owns checkpoint composition and preprocessing |
| LingBot-VLA-V2 | `Lingbot2Policy` | Source-only | Public upstream integration; derived W8 validation is not a public release claim |
| Galaxea G0.5 continuous | `patch_g05_policy_for_rpu` | Source-only | Public upstream continuous-policy integration; gated non-commercial checkpoint |
| InternVLA-N1 + NavDP | `rpu_backend.adapters.internvla_n1` / `navdp` | Source-only | Public upstream integration; exact public asset closure and validation remain pending |

See [Model support](docs/model_support.md) for status definitions, exact public
entries, registry aliases, and excluded profiles.

RhinoForge does not apply one repository-wide cosine threshold. Each exact
profile has three independent gate layers: hard semantics and Graph/runtime
lifecycle, same-semantics implementation parity, and task or end-to-end quality
on the consumed output. Quantized paths first compare against the same
quantization semantics, then against an FP16/FP32 task-quality anchor. An unrun
gate is never inferred from another model size. See the model-specific metrics
in [Model support](docs/model_support.md#profile-specific-validation-contracts)
and the general [Model validation policy](docs/validation_policy.md).

## Quick start

RhinoForge v1.0.0 is installed from source and requires a preconfigured RPU
board environment and Python 3.12. The installation path for an authorized
recipient is:

1. Download `RhinoForge-runtime-v1.0.0-r4.tar.gz` and its `.sha256` from the
   versioned bundle URL supplied by the distributor. Verify the outer archive
   before extracting it.
2. Confirm `download_enabled=true`,
   `compatible_rhinoforge_release=v1.0.0`, and
   `checksum_scope=all_runtime_payloads` in `RELEASE.txt`, then verify the
   three runtime payloads with the inner `SHA256SUMS`.
3. Read `launch_package_file`, `operator_asset_file`, and
   `operator_kernel_manifest_file` from `RELEASE.txt`. Install Rhino Launch,
   the opaque operator asset, and its adjacent sidecar under user-owned
   directories; root access is not required.
4. Run a regular, non-editable `pip install` from the RhinoForge v1.0.0 source
   root. Dependencies, including Accelerate, are installed from
   `pyproject.toml`; do not use `--no-deps`.
5. Verify the installation, then download Qwen3-0.6B at the exact
   repository/revision in the public identity ledger to its registry path and
   run inference with the unchanged TOML alias.

See [Restricted runtime assets](docs/runtime_assets.md) for copyable outer and
inner verification, safe payload-name handling, and rootless installation
commands. [Getting started](docs/getting_started.md) provides the complete
copyable path from a clean environment to Qwen3-0.6B output. Compatibility is
defined by the RhinoForge `v1.0.0` tag/version, not by matching a particular
source commit.

After installing the runtime assets and retaining the documented environment
variables, run from the source root:

```bash
python -m pip install . --no-build-isolation
python examples/verify_install.py --check-config
python examples/verify_install.py

export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
# Download Qwen3-0.6B to "$RPU_MODEL_CACHE/Qwen3-0.6B" as documented.
cp examples/configs/qwen3_0_6b.toml qwen3.local.toml
python examples/run_model.py --config qwen3.local.toml --check-config
python examples/run_model.py --config qwen3.local.toml
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
| Runtime and TOML parameters | [Runtime configuration](docs/runtime_config.md) |
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
