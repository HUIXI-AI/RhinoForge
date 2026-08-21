# Third-party notices

RhinoForge is licensed under Apache-2.0. That license does not replace the
licenses or distribution terms of the software and assets used with it.

The following dependencies are declared by `pyproject.toml` or linked by the
checked-in CMake build. Consult the linked upstream license text for the exact
version you install.

| Component | Use | License or terms |
|---|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) | Python and C++ tensor runtime | [Upstream license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| [NumPy](https://github.com/numpy/numpy) | Array utilities | [Upstream license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| [safetensors](https://github.com/safetensors/safetensors) | Checkpoint I/O | [Upstream license](https://github.com/safetensors/safetensors/blob/main/LICENSE) |
| [Transformers](https://github.com/huggingface/transformers) | Hugging Face model APIs | [Upstream license](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Checkpoint retrieval used by quantized loaders | [Upstream license](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) |
| [Pillow](https://github.com/python-pillow/Pillow) | Optional vision input support | [Upstream license](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| [TorchVision](https://github.com/pytorch/vision) | Optional vision transforms | [Upstream license](https://github.com/pytorch/vision/blob/main/LICENSE) |
| [Diffusers](https://github.com/huggingface/diffusers) | Optional NavDP scheduling | [Upstream license](https://github.com/huggingface/diffusers/blob/main/LICENSE) |
| [LeRobot](https://github.com/huggingface/lerobot) | Optional Pi0.5/VLA integration | [Upstream license](https://github.com/huggingface/lerobot/blob/main/LICENSE) |
| [OmegaConf](https://github.com/omry/omegaconf) | Optional VLA configuration | [Upstream license](https://github.com/omry/omegaconf/blob/main/LICENSE) |
| [PyYAML](https://github.com/yaml/pyyaml) | Optional VLA configuration loading | [Upstream license](https://github.com/yaml/pyyaml/blob/main/LICENSE) |
| [SciPy](https://github.com/scipy/scipy) | Optional VLA action conversion | [Upstream license](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| [scikit-build-core](https://github.com/scikit-build/scikit-build-core) | Python/CMake build backend | [Upstream license](https://github.com/scikit-build/scikit-build-core/blob/main/LICENSE) |
| [build](https://github.com/pypa/build), [pytest](https://github.com/pytest-dev/pytest) | Optional development tools | See each upstream repository's license |

Rhino Launch 1.0.0 is a separately distributed restricted prerequisite. It is
not included in this repository and is governed by the terms delivered with
that package. The combined `rhinoOpLib_current.ref` operator asset is also
distributed separately under its accompanying terms.

RhinoForge assumes a preconfigured board environment. System components
provided by that environment remain governed by the platform terms and are not
delivered by this repository.

Model checkpoints, tokenizers, processors, datasets, and generated quantized
checkpoints are not covered by RhinoForge's Apache-2.0 license. See
[`docs/model_assets.md`](docs/model_assets.md) and the relevant model owner.

This notice is an engineering inventory, not a completed license-clearance
record. Final release publication remains subject to legal review of the exact
dependency lock, binary packages, operator asset, model assets, and
redistribution terms.
