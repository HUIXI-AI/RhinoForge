<div align="center">

<a href="https://github.com/HUIXI-AI/RhinoVLA"><img src="https://raw.githubusercontent.com/HUIXI-AI/RhinoVLA/fce05e5859104c94544edc1f4380a677c3ebac4c/assets/huixi_logo_cropped.png" alt="辉羲智能" height="72" /></a>

# RhinoForge

**面向 Rhino Processing Unit 的 PyTorch 推理与模型部署工具**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

[English](README_EN.md) | **简体中文**

</div>

RhinoForge 将 Rhino Processing Unit（RPU）接入为 `torch.rpu`。项目在同一个
PyTorch 后端中提供模型适配器、融合模型运行时、图捕获与重放、SPM 内存管理、
多核执行、离线量化和模型移植模板。

发行包名为 `rhinoforge`，Python 导入名仍为 `rpu_backend`。RhinoForge 仅支持
推理，不支持训练和微调。

## 主要能力

- **PyTorch 集成：** 通过标准自定义设备路径和 `torch.rpu` API 使用 RPU。
- **模型部署：** 通过模型适配器和 TOML 示例运行文本、视觉语言、视觉和 VLA
  配置。
- **可扩展运行时：** 移植新模型时可复用公开的融合模型、图、SPM、多核、批量
  提交和适配器机制。
- **按配置界定支持范围：** 每个状态都绑定到具体 checkpoint、精度、输入范围、
  配置和运行时资产组合。

## AI 辅助模型移植

仓库内置的 [`rhinoforge-port` skill](.agents/skills/rhinoforge-port/SKILL.md)
提供两种模式：**Assess** 只读分析能力差距，分别给出 outcome 与 certification；
**Port** 只从已接受、certified 且非 Blocked 的评估开始，并按模型类型加载公开
playbook。规范流程见[模型移植](docs/model_porting.zh.md)。

## 模型矩阵

> **v1.0.0 发布状态：** 下表是当前发布范围。`Supported` 和 `Limited`
> 仅适用于 [`release/public-models-v1.0.0.json`](release/public-models-v1.0.0.json)
> 中的不可变公开模型身份及配套 v1.0.0 runtime set；生成的派生资产和
> 相邻配置不继承这些声明。

| 模型 | 公开入口 | 状态 | 限制 |
|---|---|---|---|
| Qwen3 0.6B / 1.7B / 4B / 8B | `RPUModelForCausalLM` | Supported | FP16 推理；准确执行范围以发布版本为准 |
| Qwen3 0.6B / 1.7B / 4B / 8B W8A16 | `RPUModelForCausalLM` | Source-only | 包含转换器和加载器源码，不声明可运行发布配置 |
| Qwen3 14B W8A16 | `RPUModelForCausalLM` | Source-only | 仅包含 converter/loader 源码；v1.0.0 未绑定公开不可变派生 checkpoint 身份或 hash |
| Qwen3 32B | `RPUModelForCausalLM` | Source-only | 无可运行配置；预检会拒绝不支持的配置 |
| Llama-3.2-1B | `RPUModelForCausalLM` | Supported | 因果语言模型推理 |
| Qwen3.5 text 2B / 9B | `Qwen3_5Adapter` | Supported | 仅 dense FP16 文本；不支持 `torch.compile` |
| Qwen3.5 text 0.8B / 4B | `Qwen3_5Adapter` | Experimental | 本次 192-token 验证的 last-prefill full-vocabulary relative-L2 超过 `0.01`；numeric-blocked，仅供受控评估 |
| Qwen3.5 Vision 2B / 4B | `Qwen3_5Adapter` | Experimental | 官方真实图数值门未通过；默认拒绝图像推理，精确 `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1` 仅供受控评估 |
| Qwen3-VL 2B | `RPUModelForConditionalGeneration` | Supported | 图像和文本；视频不在发布范围内 |
| Qwen3-VL 4B | `RPUModelForConditionalGeneration` | Supported | 仅限已验证的单图、多图、prefill、decode 和 chunk 范围 |
| Qwen3-VL 8B | `RPUModelForConditionalGeneration` | Limited | 精确 dense-FP16 padded profile 的 4/4 numeric/task/Graph 已通过，包含 dual-full；base 5/5、long-text 9/9 token 精确一致；相邻长度、视频和量化路径不继承 |
| Qwen3-VL 32B W8A16 | `RPUModelForConditionalGeneration` | Source-only | 第三个 decode step 起存在全零 logits 硬失败；只保留显式放行的受控诊断入口 |
| Gemma4-E4B text | `Gemma4Adapter` | Source-only | 运行时依赖和发布配置尚未冻结 |
| Pi0.5 Libero FP16 | `Pi05Policy` | Supported | 精确 Libero FP16 profile 的 9/9 numeric/task 与 27/27 execution-plan record 已通过；独立 fresh-process Graph 生命周期门也已通过 |
| Pi0.5 Libero W8A16 | `Pi05Policy` | Source-only | 仅包含 converter/runtime 源码；v1.0.0 未绑定公开不可变派生 checkpoint 身份或 hash |
| Pi0.5 Libero W4A16 | `Pi05Policy` | Source-only | 仅包含 converter/runtime 源码；v1.0.0 未绑定公开不可变派生 checkpoint 身份或 hash |
| Pi0.5 reduced-step / closed-loop | `Pi05Policy` | Source-only | 仅保留策略源码；验证仍待完成，不继承精确 FP16 profile 或量化路径的证据与状态 |
| Wall-OSS-0.5 | `WallOssPolicy` | Source-only | 接入公开 `wall-x`；固定的公开 checkpoint 尚未完成发布验证门禁 |
| Wall-OSS 量化配置 | `WallOssPolicy` | Source-only | 包含公开 converter；派生 checkpoint 身份和验证不从 FP16 继承 |
| Hy-Embodied-0.5-VLA FP16/W16 | `HyEmbodiedPolicy` | Limited | 仅精确公开 FP16/W16 profile；数值推理不代表机器人就绪认证 |
| Hy-Embodied-0.5-VLA W8/W4 | `HyEmbodiedPolicy` | Source-only | 仅包含 runtime 转换源码；v1.0.0 未绑定公开不可变派生 checkpoint 身份或 hash |
| DINOv3 ViT-B | `DINOv3Adapter` | Supported | 仅 ViT-B |
| SigLIP | `rpu_backend.adapters.siglip` | Component-only | 仅视觉编码器组件 |
| GR00T-N1.7-3B | `build_gr00t_vla` | Source-only | 公开资产和端到端设置尚未完成 |
| [RhinoVLA](https://github.com/HUIXI-AI/RhinoVLA) | `RhinoVLAPolicy` | Source-only | 集成候选；checkpoint 组成和预处理由模型仓库定义 |
| LingBot-VLA-V2 | `Lingbot2Policy` | Source-only | 公开上游接入；派生 W8 验证不构成公开发布声明 |
| Galaxea G0.5 continuous | `patch_g05_policy_for_rpu` | Source-only | 公开上游 continuous policy 接入；checkpoint gated 且非商业 |
| InternVLA-N1 + NavDP | `rpu_backend.adapters.internvla_n1` / `navdp` | Source-only | 公开上游接入；精确公开资产闭包和验证仍待完成 |

状态定义、完整公开入口、registry alias 和排除配置见
[模型支持](docs/model_support.zh.md)。

模型验证不使用全仓统一的 cosine 阈值。每个精确 profile 分别通过三层门禁：
硬语义与 Graph/runtime 生命周期、同语义实现对齐，以及面向最终输出的任务或
端到端质量。量化路径先与相同量化语义的 reference 对齐，再与 FP16/FP32 anchor
比较任务质量；未运行的门禁不会从其他模型尺寸推断为通过。各模型指标与阈值见
[模型支持](docs/model_support.zh.md#分模型验证合同)和
[模型验证策略](docs/validation_policy.zh.md)。

## 快速开始

RhinoForge v1.0.0 从源码安装，需要已配置好的 RPU 板卡环境和 Python 3.12。
授权接收方的安装路径如下：

1. 从分发方提供的版本化 bundle URL 下载
   `RhinoForge-runtime-v1.0.0-r4.tar.gz` 及其 `.sha256`，先校验外层归档，再解包。
2. 检查 `RELEASE.txt` 中 `download_enabled=true`、
   `compatible_rhinoforge_release=v1.0.0` 和
   `checksum_scope=all_runtime_payloads`，然后使用内层 `SHA256SUMS` 校验三个
   runtime payload。
3. 从 `RELEASE.txt` 读取 `launch_package_file`、`operator_asset_file` 和
   `operator_kernel_manifest_file`，将 Rhino Launch、opaque 算子资产及相邻
   sidecar 安装到用户目录；全程不需要 root。
4. 在 RhinoForge v1.0.0 源码根目录执行普通（非 editable）`pip install`。依赖
   （包括 Accelerate）会按 `pyproject.toml` 自动安装，不要使用 `--no-deps`。
5. 运行安装检查，再按公开身份台账中的准确 repository/revision 将 Qwen3-0.6B
   下载到 registry 约定目录，直接使用原 TOML alias 执行推理。

外层与内层校验、三个 payload 的安全取名和无 root 安装命令见
[受限运行时资产](docs/runtime_assets.zh.md)；从干净环境到 Qwen3-0.6B 输出的完整
可复制步骤见[入门指南](docs/getting_started.zh.md)。兼容关系按 RhinoForge
`v1.0.0` tag/version 判断，不要求匹配某个源码 commit。

完成运行时资产安装并保持文档要求的环境变量后，在源码根目录执行：

```bash
python -m pip install . --no-build-isolation
python examples/verify_install.py --check-config
python examples/verify_install.py

export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
# 按入门指南将 Qwen3-0.6B 下载到 "$RPU_MODEL_CACHE/Qwen3-0.6B"。
cp examples/configs/qwen3_0_6b.toml qwen3.local.toml
python examples/run_model.py --config qwen3.local.toml --check-config
python examples/run_model.py --config qwen3.local.toml
```

RhinoForge 不分发模型 checkpoint 和生成的量化 checkpoint。完整的安装、资产
校验、模型下载、量化和推理流程见[入门指南](docs/getting_started.zh.md)。仓库中的
[示例](examples/)提供简洁的 TOML 驱动入口；[模型运行与性能分析](docs/model_testing.zh.md)
说明完整 TOML runner、Torch profiler、硬件 profiler，以及可运行或受限的模型入口。

## 文档

| 主题 | 文档 |
|---|---|
| 完整文档导航 | [文档索引](docs/README.md) |
| 安装与运行 | [入门指南](docs/getting_started.zh.md) |
| 支持和排除的配置 | [模型支持](docs/model_support.zh.md) |
| 受限 Launch 与算子资产 | [受限运行时资产](docs/runtime_assets.zh.md) |
| 模型 checkpoint | [模型资产](docs/model_assets.zh.md) |
| Runtime/TOML 参数与性能分析 | [运行时配置](docs/runtime_config.zh.md) |
| 模型运行、测试与性能分析 | [模型运行与性能分析](docs/model_testing.zh.md) |
| 可复现性能测量 | [性能测量](docs/performance.zh.md) |
| 模型验证与状态门禁 | [模型验证策略](docs/validation_policy.zh.md) |
| 运行时设计 | [架构](docs/architecture.zh.md) |
| Python 与 C++ API | [API 参考](docs/api_reference.zh.md) |
| 移植新模型 | [模型移植](docs/model_porting.zh.md) |
| 离线量化 | [量化](docs/quantization.zh.md) |
| 报告安全问题 | [安全策略](SECURITY.md) |
| 参与贡献 | [贡献指南](CONTRIBUTING.md) |

## 许可证

RhinoForge 源码采用 [Apache License 2.0](LICENSE)。模型 checkpoint、Rhino
Launch、合并算子资产及其他第三方材料遵循各自条款，不属于本仓库 `LICENSE`
文件的授权范围。
