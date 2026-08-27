# 性能测量

简体中文 | [English](performance.md)

RhinoForge 的性能结果只对一个确定的模型与运行时配置有意义。适用的
[模型验证策略](validation_policy.zh.md)门禁通过前，不要比较或发布延迟数据。

仓库刻意不保存持续变化的内部板卡成绩表。Release notes 可以发布绑定到不可变软件、
资产、模型、输入和测量记录的结果。

## 固定被测配置

运行前记录：

- RhinoForge 源码 revision 和安装包版本；
- Rhino Launch 包与合并算子资产的兼容标识；
- 模型 checkpoint revision、文件 hash、精度和量化元数据；
- 板卡/runtime 版本及相关 host 软件版本；
- 输入形状或请求范围、生成或动作参数和 batch；
- TOML hash 与实际生效的公开 runtime 配置；
- warmup 次数、测量样本数和 profiler 状态。

每个新进程只运行一个模型配置。checkpoint、精度、输入范围、Graph 计划或 runtime
资产集合中任一项变化，都构成另一个 benchmark。为什么不能孤立比较单个环境变量，见
[运行时配置](../knowledge/concepts/runtime-profiles.md)。

## 分开报告启动与稳态

- **启动阶段**可包含包导入、模型加载、权重转换、设备搬运、Graph BUILD 和显式
  Graph 准备。报告必须说明边界。
- **稳态阶段**只从规定的 warmup 和 Graph 准备完成后开始。复用同一输入范围，并确认
  重复 signature 使用文档规定的 replay 生命周期。

不要从端到端时间中选择性扣除 host 工作。单独报告组件时，应定义其输入输出边界，
并避免把组件耗时之和冒充端到端延迟。

## 测量流程

1. 固定确定性或已记录的输入。随机 policy 的正确性和计时比较应复用相同初始噪声或
   随机状态。
2. 在关闭 profiler 的情况下完成正确性和生命周期门禁。
3. 启动新进程，应用完全相同的配置，完成规定的 warmup 或 Graph 准备。
4. 测量完整公开 API 调用，包括调用方需要的输出物化。报告样本数、至少中位数和一个
   尾部百分位，不要只发布最快值。
5. 评估小优化时，以交替或其他温度受控的顺序重复 baseline/candidate，同时报告绝对值
   与差值。
6. Torch 或硬件 profiling 在单独的诊断进程中运行；带 profiler 的延迟不是 release
   性能结果。

语言模型应分别报告 prefill 吞吐、decode 吞吐和端到端请求延迟；统计逻辑 token，
不统计 padding 执行行。VLM/VLA 应报告图像/视角数、prompt 范围、denoise 步数、action
horizon 和完整 policy 调用延迟；只有说明 action chunk 定义时才使用 `action chunks/s`。

## 最小发布记录

| 字段 | 必需内容 |
|---|---|
| 配置 | 模型 revision、精度/量化、输入范围和 TOML hash |
| Runtime | RhinoForge revision、Launch 包、算子资产和板卡/runtime 版本 |
| 正确性 | 门禁名称、参考身份、结果及任何 pending 范围 |
| 计时 | 边界、warmup、样本数、中位数、尾部百分位和单位 |
| 生命周期 | 冷启动/稳态；适用时附 Graph BUILD/REPLAY 证据 |
| 条件 | profiler 开关、相关 host 配置和日期 |

Torch trace 可能暴露应用形状和源码路径。不要提交到仓库，只发布审查后的摘要。TOML profiling 入口见
[模型运行与性能分析](model_testing.zh.md#torch-profile)。
