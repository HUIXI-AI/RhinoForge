<div align="center">

<a href="https://github.com/HUIXI-AI/RhinoVLA"><img src="https://raw.githubusercontent.com/HUIXI-AI/RhinoVLA/3bfdbbfe8370be42f3a936eec616d065219e1825/assets/huixi_logo_cropped.png" alt="Huixi Intelligence" height="72" /></a>

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

## Examples

[`examples/`](examples/README.md) provides the unified runner and TOML
configurations organized by model for text, image-text, vision components, and
VLA workflows. See the [configuration index](examples/configs/README.md) for
the directory layout, precision choices, and input requirements.

Supply your own model weights and requests. Configurations and runtime admission
checks constrain each checkpoint, precision, core count, and input envelope.
Controlled and Source-only examples retain their restrictions; a template is
not model certification.

## Quick start

The current source requires Python 3.12, a configured RPU board environment,
and matching Rhino Launch and operator assets. Follow
[Build the current development source](docs/development_build.md) to create an
isolated environment and install the dependencies defined in
[pyproject.toml](pyproject.toml). Use the `vision` or `vla` extra for those entry
points.

1. Configure the external runtime paths and install the current source using
   the build guide.
2. Prepare the checkpoint as described in [Model assets](docs/model_assets.md),
   set `RPU_MODEL_CACHE`, and select the corresponding TOML example.
3. Confirm that the board is idle, check the installation and configuration,
   then run inference:

```bash
python examples/verify_install.py --check-config
python examples/verify_install.py
cp examples/configs/qwen3/text/0_6b/fp16.toml qwen3.local.toml
python examples/run_model.py --config qwen3.local.toml --check-config
python examples/run_model.py --config qwen3.local.toml
```

[Model execution and profiling](docs/model_testing.md) describes the TOML
runner, quantization, and profiling entry points. Obtain model checkpoints,
Rhino Launch, and operator assets from their respective distributors.

See [Getting started](docs/getting_started.md) for installation and a first run.

## Documentation

| Topic | Document |
|---|---|
| Install current source | [Development build](docs/development_build.md) |
| Installation and first run | [Getting started](docs/getting_started.md) |
| Examples | [Example directory and entry points](examples/README.md) |
| Restricted Launch and operator assets | [Restricted runtime assets](docs/runtime_assets.md) |
| Model checkpoints | [Model assets](docs/model_assets.md) |
| Runtime and TOML parameters | [Runtime configuration](docs/runtime_config.md) |
| Model execution, testing, and profiling | [Model execution and profiling](docs/model_testing.md) |
| Reproducible performance measurement | [Performance measurement](docs/performance.md) |
| Proportionate validation | [Model validation policy](docs/validation_policy.md) |
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
