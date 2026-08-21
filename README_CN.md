<div align="center">

# RhinoForge

**面向 Rhino Processing Unit 的 PyTorch 推理与模型部署工具**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

[English](README.md) | **简体中文**

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
playbook。规范流程见[模型移植](docs/model_porting.md)。

## 模型矩阵

> **候选发布状态：** 下表是计划中的发布状态。只有在同一套不可变软件与资产上
> 完成最终包验证后，`Supported` 和 `Limited` 才构成正式发布声明。当前源码候选
> 仍在等待该验证。

| 模型 | 公开入口 | 状态 | 限制 |
|---|---|---|---|
| Qwen3 0.6B / 1.7B / 4B / 8B | `RPUModelForCausalLM` | Supported | FP16 推理；准确执行范围以发布版本为准 |
| Qwen3 0.6B / 1.7B / 4B / 8B W8A16 | `RPUModelForCausalLM` | Source-only | 包含转换器和加载器源码，不声明可运行发布配置 |
| Qwen3 14B | `RPUModelForCausalLM` | Limited | 仅限 untied INT8 `lm_head`、FP16 embedding 的确定 W8A16 配置 |
| Qwen3 32B | `RPUModelForCausalLM` | Source-only | 无可运行配置；预检会拒绝不支持的配置 |
| Llama-3.2-1B | `RPUModelForCausalLM` | Supported | 因果语言模型推理 |
| Qwen3.5 text 0.8B / 2B / 4B / 9B | `Qwen3_5Adapter` | Supported | 仅 dense FP16 文本；不支持 `torch.compile` |
| Qwen3.5 Vision 2B / 4B | `Qwen3_5Adapter` | Experimental | 仅用于受控评估；默认保持提前拒绝 |
| Qwen3-VL 2B | `RPUModelForConditionalGeneration` | Supported | 图像和文本；视频不在发布范围内 |
| Qwen3-VL 4B | `RPUModelForConditionalGeneration` | Limited | 仅限发布版本明确列出的单图和多图输入范围 |
| Qwen3-VL 32B W8A16 | `RPUModelForConditionalGeneration` | Experimental | 受控评估；尚无完整支持结论 |
| Gemma4-E4B text | `Gemma4Adapter` | Source-only | 运行时依赖和发布配置尚未冻结 |
| Pi0.5 Libero FP16 | `Pi05Policy` | Supported | 确定的 Libero 配置 |
| Pi0.5 Libero W8A16 | `Pi05Policy` | Limited | 确定的直接推理配置 |
| Pi0.5 W4 / reduced-step / closed-loop | `Pi05Policy` | Experimental | 每种配置需要独立验证 |
| Wall-OSS-0.5 | `WallOssPolicy` | Limited | 仅限发布版本明确列出的相机、前缀和精度配置 |
| Wall-OSS RTC | `WallOssPolicy` | Experimental | 仅用于确定配置的受控评估 |
| Wall-OSS W4 and other quantized profiles | `WallOssPolicy` | Experimental | 仅用于评估；状态按配置分别判定 |
| Hy-Embodied-0.5-VLA | `HyEmbodiedPolicy` | Limited | 确定输入配置；数值推理结果不代表机器人就绪认证 |
| DINOv3 ViT-B | `DINOv3Adapter` | Supported | 仅 ViT-B |
| SigLIP | `rpu_backend.adapters.siglip` | Component-only | 仅视觉编码器组件 |
| GR00T-N1.7-3B | `build_gr00t_vla` | Source-only | 公开资产和端到端设置尚未完成 |
| [RhinoVLA](https://github.com/HUIXI-AI/RhinoVLA) | `RhinoVLAPolicy` | Source-only | 集成候选；checkpoint 组成和预处理由模型仓库定义 |
| LingBot-VLA-V2 exact-Z2 W8A16 | `Lingbot2Policy` | Experimental | 仅用于受控评估 |
| Galaxea G0.5 exact K1 RTC/FM W0 | `patch_g05_policy_for_rpu` | Experimental | 仅限确定的 K1 W0 配置 |
| InternVLA-N1 + NavDP | `rpu_backend.adapters.internvla_n1` / `navdp` | Experimental | 需要确定的资产清单和受控评估 |

状态定义、完整公开入口、registry alias 和排除配置见
[模型支持](docs/model_support.md)。

## 快速开始

首个版本从源码安装，需要预先配置好的 RPU 板卡环境、Python 3.12、Rhino
Launch 1.0.0 二进制开发包，以及一个名为 `rhinoOpLib_current.ref` 的兼容合并
算子资产。Rhino Launch 和算子资产通过授权渠道单独分发，不包含在本仓库中。
构建前，请按[受限运行时资产](docs/runtime_assets.md)选择同一个有效兼容集，并校验其
签名 BOM 和发布状态。

如果 Rhino Launch 不在系统的 CMake 搜索路径中，请先设置其安装前缀，然后构建
并验证 RhinoForge：

```bash
export CMAKE_PREFIX_PATH="/opt/rhino-launch${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
python -m pip install . --no-build-isolation

export RPU_KERNEL_LIB_PATH="$HOME/.local/share/rhinoforge/rhinoOpLib_current.ref"
python examples/verify_install.py
```

从[模型资产](docs/model_assets.md)列出的来源获取 Qwen3-0.6B，复制其 TOML 配置，
并填写本地 checkpoint 路径。正式发布时，release manifest 会固定用于发布验证的
准确 revision 和 hash。

```bash
export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
cp examples/configs/qwen3_0_6b.toml qwen3.local.toml
# 编辑 qwen3.local.toml，填入已下载的 checkpoint 路径。
python examples/causal_lm.py --config qwen3.local.toml --check-config
python examples/causal_lm.py --config qwen3.local.toml
```

RhinoForge 不分发模型 checkpoint 和生成的量化 checkpoint。完整的安装、资产
校验、模型下载、量化和推理流程见[入门指南](docs/getting_started.md)。仓库中的
[示例](examples/)提供简洁的 TOML 驱动入口；[模型运行与性能分析](docs/model_testing.md)
说明完整 TOML runner、Torch profiler、硬件 profiler，以及可运行或受限的模型入口。

## 文档

| 主题 | 文档 |
|---|---|
| 安装与运行 | [入门指南](docs/getting_started.md) |
| 支持和排除的配置 | [模型支持](docs/model_support.md) |
| 受限 Launch 与算子资产 | [受限运行时资产](docs/runtime_assets.md) |
| 模型 checkpoint | [模型资产](docs/model_assets.md) |
| 运行时环境变量与性能分析 | [运行时配置](docs/runtime_config.md) |
| 模型运行、测试与性能分析 | [模型运行与性能分析](docs/model_testing.md) |
| 可复现性能测量 | [性能测量（英文）](docs/performance.md) |
| 模型验证与状态门禁 | [模型验证策略](docs/validation_policy.md) |
| 运行时设计 | [架构](docs/architecture.md) |
| Python 与 C++ API | [API 参考](docs/api_reference.md) |
| 移植新模型 | [模型移植](docs/model_porting.md) |
| 离线量化 | [量化](docs/quantization.md) |
| 报告安全问题 | [安全策略](SECURITY.md) |
| 参与贡献 | [贡献指南](CONTRIBUTING.md) |

## 许可证

RhinoForge 源码采用 [Apache License 2.0](LICENSE)。模型 checkpoint、Rhino
Launch、合并算子资产及其他第三方材料遵循各自条款，不属于本仓库 `LICENSE`
文件的授权范围。
