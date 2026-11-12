<div align="center">

<a href="https://github.com/HUIXI-AI/RhinoVLA"><img src="https://raw.githubusercontent.com/HUIXI-AI/RhinoVLA/3bfdbbfe8370be42f3a936eec616d065219e1825/assets/huixi_logo_cropped.png" alt="辉羲智能" height="72" /></a>

# RhinoForge

**面向 Rhino RPU 的 PyTorch 推理与模型部署工具**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

[English](README_EN.md) | **简体中文**

</div>

RhinoForge 将 Rhino RPU 接入为 `torch.rpu`。项目在同一个
PyTorch 后端中提供模型适配器、融合模型运行时、图捕获与重放、SPM 内存管理、
多核执行、离线量化和模型移植模板。

发行包名为 `rhinoforge`，Python 导入名仍为 `rpu_backend`。RhinoForge 仅支持
推理，不支持训练和微调。

当前源码版本为 **v1.3.0.dev0**（开发版），依赖已归档的 **Rhino Launch v1.2.0**；
使用[运行时资产指南](docs/runtime_assets.zh.md)指定的正式版组件。版本规则见[发布流程](docs/release_policy.md)。更新范围见[发行说明](CHANGELOG.md)，
已有环境请先阅读[升级指南](docs/upgrading_1_2.zh.md)。

## 主要能力

- **PyTorch 集成：** 通过标准自定义设备路径和 `torch.rpu` API 使用设备。
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

## 示例

[`examples/`](examples/README.md) 提供统一运行入口和按模型归档的 TOML 配置，覆盖
文本、图文、视觉组件和 VLA 的公开使用方式。目录结构、精度和输入要求见
[配置索引](examples/configs/README.md)。

调用方准备模型权重和请求数据；配置与运行时共同检查 checkpoint、精度、核数和输入范围。

Qwen3 ASR/TTS 使用独立语音入口，下载和运行步骤见[语音指南](docs/qwen3_audio.zh.md)。

## 快速开始

先由平台方提供并确认板卡 OS、驱动、SDK 和匹配的运行资产，再按
[统一安装、运行与开发指南](docs/getting_started.zh.md)选择一条应用部署路径：

- [从 Miniconda3 开始构建](docs/getting_started.zh.md#source-build)：安装管理器、创建 Python 3.12 环境、安装依赖并编译 RhinoForge。
- [使用依赖环境包](docs/getting_started.zh.md#bundle-deploy)：从 GitHub Releases 获取匹配的第三方依赖环境，随后从 GitHub 获取并编译 RhinoForge。

两条路径都使用一个 `RHINOFORGE_ENV`，完成后继续相同的模型准备、Qwen3-0.6B FP16
运行和后续开发步骤。依赖包不包含 RhinoForge、SDK 用户态库、Launch 或算子；
SDK、运行资产和模型仍单独准备。推荐 SDK 版本和平台适配要求见
[安装指南](docs/getting_started.zh.md#前置条件由平台方提供)。

## 文档

| 主题 | 文档 |
|---|---|
| v1.2 更新与安装 | [发行说明](CHANGELOG.md) · [升级指南](docs/upgrading_1_2.zh.md) |
| 完整文档导航 | [文档索引](docs/README.md) |
| 安装、首次运行与开发 | [统一指南](docs/getting_started.zh.md) |
| 语音识别与生成 | [Qwen3 ASR/TTS](docs/qwen3_audio.zh.md) |
| 示例 | [示例目录与运行入口](examples/README.md) |
| 受限 Launch 与算子资产 | [受限运行时资产](docs/runtime_assets.zh.md) |
| 模型 checkpoint | [模型资产](docs/model_assets.zh.md) |
| Runtime/TOML 参数 | [运行时配置](docs/runtime_config.zh.md) |
| 模型运行、测试与调试 | [模型运行与调试](docs/model_testing.zh.md) |
| 适量验证 | [模型验证策略](docs/validation_policy.zh.md) |
| 运行时设计 | [架构](docs/architecture.zh.md) |
| Python 与 C++ API | [API 参考](docs/api_reference.zh.md) |
| 移植新模型 | [模型移植](docs/model_porting.zh.md) |
| 离线量化 | [量化](docs/quantization.zh.md) |
| 报告安全问题 | [安全策略](SECURITY.md) |
| 参与贡献 | [贡献指南](CONTRIBUTING.md) |

## 许可证

RhinoForge 源码采用 [Apache License 2.0](LICENSE)。模型 checkpoint、Rhino
Launch、主算子资产与扩展算子资产及其他第三方材料遵循各自条款，不属于本仓库 `LICENSE`
文件的授权范围。
