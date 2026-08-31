# Third-party notices

RhinoForge is licensed under Apache-2.0. That license does not replace the
licenses or distribution terms of the software and assets used with it.

The following dependencies and adapted upstream sources are declared by
`pyproject.toml`, linked by the checked-in CMake build, or cited by adapter
source. Consult the linked upstream license text for the exact version used.

| Component | Use | License or terms |
|---|---|---|
| [Hugging Face Accelerate](https://github.com/huggingface/accelerate) | Device-aware Hugging Face model loading | [Apache License 2.0](https://github.com/huggingface/accelerate/blob/main/LICENSE) |
| [PyTorch](https://github.com/pytorch/pytorch) | Python and C++ tensor runtime | [Upstream license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| [NumPy](https://github.com/numpy/numpy) | Array utilities | [Upstream license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| [safetensors](https://github.com/safetensors/safetensors) | Checkpoint I/O | [Upstream license](https://github.com/safetensors/safetensors/blob/main/LICENSE) |
| [Transformers](https://github.com/huggingface/transformers) | Hugging Face model APIs | [Upstream license](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Checkpoint retrieval used by quantized loaders | [Upstream license](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) |
| [Pillow](https://github.com/python-pillow/Pillow) | Optional vision input support | [Upstream license](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| [TorchVision](https://github.com/pytorch/vision) | Optional vision transforms | [Upstream license](https://github.com/pytorch/vision/blob/main/LICENSE) |
| [Diffusers](https://github.com/huggingface/diffusers) | Optional NavDP scheduling | [Upstream license](https://github.com/huggingface/diffusers/blob/main/LICENSE) |
| [LeRobot](https://github.com/huggingface/lerobot) | Optional Pi0.5/VLA integration | [Upstream license](https://github.com/huggingface/lerobot/blob/main/LICENSE) |
| [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/23ace64f17aa5015259b8609d371eb61a357c776) | Optional GR00T adapter reference and public preprocessing contract | [Apache License 2.0](https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/LICENSE) |
| [NVIDIA GR00T-N1.7 checkpoints](https://huggingface.co/nvidia/GR00T-N1.7-3B) | User-supplied model assets; not redistributed by RhinoForge | NVIDIA Open Model License and the access terms delivered by the model owner |
| [NVIDIA Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | Gated, user-supplied GR00T processor/reference dependency; not redistributed by RhinoForge | NVIDIA Open Model License; access and license acceptance are required from the model owner |
| [OpenGalaxea G05 checkpoint](https://huggingface.co/OpenGalaxea/G05/tree/e312be81e90c56a55bcb26b57429bd39a335b449) / [GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA/tree/89f2322b4ad016e192437adc1a2c253b05bab246) | Optional public G0.5 continuous/discrete/CoT adapter reference and user-supplied checkpoint | G0.5 Community License Agreement (Non-Commercial + Limited Patent License); see the pinned upstream [LICENSE-G0.5](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/LICENSE-G0.5) and [NOTICE](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/NOTICE). Not redistributed by RhinoForge |
| [LingBot-VLA-V2 Normalizer](https://github.com/Robbyant/lingbot-vla-v2/blob/951475ae1b1d87553e7dc47c97b53a3d695c0d13/lingbotvla/data/vla_data/transform.py) | Adapted action de-normalization formulas | [Apache License 2.0](https://github.com/Robbyant/lingbot-vla-v2/blob/951475ae1b1d87553e7dc47c97b53a3d695c0d13/LICENSE) |
| [InternNav](https://github.com/InternRobotics/InternNav/tree/7a5c62400ac45b313d9b709c740b64191556a242) | Optional InternVLA/NavDP integration | [MIT, Copyright 2025 Intern Robotics](https://github.com/InternRobotics/InternNav/blob/7a5c62400ac45b313d9b709c740b64191556a242/LICENSE) |
| [OmegaConf](https://github.com/omry/omegaconf) | Optional VLA configuration | [Upstream license](https://github.com/omry/omegaconf/blob/main/LICENSE) |
| [PyYAML](https://github.com/yaml/pyyaml) | Optional VLA configuration loading | [Upstream license](https://github.com/yaml/pyyaml/blob/main/LICENSE) |
| [SciPy](https://github.com/scipy/scipy) | Optional VLA action conversion | [Upstream license](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| [scikit-build-core](https://github.com/scikit-build/scikit-build-core) | Python/CMake build backend | [Upstream license](https://github.com/scikit-build/scikit-build-core/blob/main/LICENSE) |
| [build](https://github.com/pypa/build), [pytest](https://github.com/pytest-dev/pytest) | Optional development tools | See each upstream repository's license |

Rhino Launch 1.0.0 is a separately distributed restricted prerequisite. It is
not included in this repository and is governed by the terms delivered with
that package. The versioned combined operator asset named by the matching
runtime set is also distributed separately under its accompanying terms.

RhinoForge assumes a preconfigured board environment. System components
provided by that environment remain governed by the platform terms and are not
delivered by this repository.

Model checkpoints, tokenizers, processors, datasets, and generated quantized
checkpoints are not covered by RhinoForge's Apache-2.0 license. See
[`docs/model_assets.md`](docs/model_assets.md) and the relevant model owner.

This notice is an engineering inventory, not legal advice or a grant of rights
to separately distributed software, runtime binaries, operator assets, model
assets, or datasets. Recipients must follow the terms supplied by each owner
for the exact versions and assets they obtain.
