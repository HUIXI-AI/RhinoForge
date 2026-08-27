# RhinoForge 文档

中文是默认阅读入口；每份用户和开发者文档都提供同页中英文切换。英文文件保留原名，
中文版使用 `.zh.md` 后缀。

## 安装、运行与发布配置

| 主题 | 中文 | English |
|---|---|---|
| 从源码安装并跑通首个模型 | [入门指南](getting_started.zh.md) | [Getting started](getting_started.md) |
| 模型状态、入口和边界 | [模型支持](model_support.zh.md) | [Model support](model_support.md) |
| Rhino Launch 和合并算子资产 | [受限运行时资产](runtime_assets.zh.md) | [Restricted runtime assets](runtime_assets.md) |
| Checkpoint 来源、hash 和转换 | [模型资产](model_assets.zh.md) | [Model assets](model_assets.md) |
| TOML、环境变量和运行时参数 | [运行时配置](runtime_config.zh.md) | [Runtime configuration](runtime_config.md) |
| 模型入口、配置检查和 profiling | [模型运行与性能分析](model_testing.zh.md) | [Model execution and profiling](model_testing.md) |
| 可复现性能测量 | [性能测量](performance.zh.md) | [Performance measurement](performance.md) |
| 支持状态与数值门禁 | [模型验证策略](validation_policy.zh.md) | [Model validation policy](validation_policy.md) |

## 开发、扩展与调试

| 主题 | 中文 | English |
|---|---|---|
| Runtime、Graph、SPM、多核和 DMA | [架构](architecture.zh.md) | [Architecture](architecture.md) |
| Python、Graph、policy 和 device API | [API 参考](api_reference.zh.md) | [API reference](api_reference.md) |
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

本目录不保留开发时间线、内部机器路径、受限设备程序说明、算子资产内部格式或原始
profiling/debug payload。仍有复用价值的结论会提炼为公开契约，而不是复制历史记录。
