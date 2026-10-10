# RhinoForge 文档

中文是默认阅读入口；主要指南提供中英文切换。英文文件保留原名，
中文版使用 `.zh.md` 后缀。

## 安装、运行与发布配置

| 主题 | 中文 | English |
|---|---|---|
| v1.2.0 变更与升级 | [升级指南](upgrading_1_2.zh.md) · [发行说明](../CHANGELOG.md) | [Upgrade guide](upgrading_1_2.md) · [Changelog](../CHANGELOG.md) |
| 从源码安装并跑通首个模型 | [入门指南](getting_started.zh.md) | [Getting started](getting_started.md) |
| 语音识别与生成 | [Qwen3 ASR/TTS](qwen3_audio.zh.md) | [Qwen3 ASR/TTS](qwen3_audio.md) |
| 示例与运行范围 | [示例](model_support.zh.md) | [Examples](model_support.md) |
| LightNav-0 R1 迁移与缓存评测 | [迁移与评测](lightnav_0_r1.zh.md) | [Port and benchmark](lightnav_0_r1.md) |
| Rhino Launch 和主算子资产与扩展算子资产 | [受限运行时资产](runtime_assets.zh.md) | [Restricted runtime assets](runtime_assets.md) |
| Checkpoint 来源、hash 和转换 | [模型资产](model_assets.zh.md) | [Model assets](model_assets.md) |
| TOML、环境变量和运行时参数 | [运行时配置](runtime_config.zh.md) | [Runtime configuration](runtime_config.md) |
| 模型入口、配置检查与调试 | [模型运行与调试](model_testing.zh.md) | [Run and debug models](model_testing.md) |
| 改动验证 | [模型验证策略](validation_policy.zh.md) | [Model validation policy](validation_policy.md) |

## 开发、扩展与调试

| 主题 | 中文 | English |
|---|---|---|
| Runtime、Graph、SPM、多核和 DMA | [架构](architecture.zh.md) | [Architecture](architecture.md) |
| Python、Graph、policy 和 device API | [API 参考](api_reference.zh.md) | [API reference](api_reference.md) |
| 冷态核数、组件拓扑与缓存布局 | [计算核拓扑](planner_core_topology.zh.md) | [Compute-core topology](planner_core_topology.md) |
| 完整模型移植流程 | [模型移植](model_porting.zh.md) | [Model porting](model_porting.md) |
| 写代码前的六阶段能力评审 | [模型移植能力评审](model_porting_capability_review.zh.md) | [Capability review](model_porting_capability_review.md) |
| Adapter-only decoder 清单 | [新模型逐步指南](new_model_step_by_step.zh.md) | [New decoder step by step](new_model_step_by_step.md) |
| 数值、Graph、SPM、DMA 常见错误 | [模型移植常见陷阱](pitfalls.zh.md) | [Porting pitfalls](pitfalls.md) |
| 离线 W8/W4 checkpoint 转换 | [量化](quantization.zh.md) | [Quantization](quantization.md) |

## 面向 AI 的项目知识

- [`knowledge/INDEX.md`](../knowledge/INDEX.md) 是简洁、可检索、带公开源码链接的概念与
  playbook 层；用户文档和源码仍是事实源。
- [`.agents/skills/rhinoforge-port`](../.agents/skills/rhinoforge-port/README.zh.md)
  是通用 AI agent 模型评估与移植工作流，不依赖 `.claude` 或外部插件。
