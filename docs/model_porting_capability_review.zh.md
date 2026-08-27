# 模型移植能力评审

简体中文 | [English](model_porting_capability_review.md)

写 adapter 前先完成本评审。目标是判断一个精确模型配置是否适合 RhinoForge 当前公开
接口，并在 checkpoint 加载或不可逆权重转换前暴露 blocker。

先读[模型支持](model_support.zh.md)、[架构](architecture.zh.md)和
[模型移植](model_porting.zh.md)。每次只评审一个精确配置；家族名称或相同 Hugging Face
architecture 不是配置。

## 评审记录

在 port proposal 中记录：

- 模型来源、精确 revision、配置 hash 和 `config.architectures[0]`；
- hidden、intermediate、layer、attention head、KV head 和 vision geometry；
- 精度和量化元数据；
- batch、sequence、cache、image、video 和 action horizon 上限；
- 预处理与返回输出契约；
- RhinoForge、PyTorch、Rhino Launch 和合并算子资产版本；
- 下文五种 outcome 中的一个。

未知值是评审缺口，不是继承相邻配置的许可。

## 阶段 1：固定配置

逐行比较[模型支持](model_support.zh.md)中的状态和排除项。定义有用且可测试的最小输入
范围，并包含所有会改变模型数学、tensor shape、Graph topology、cache layout 或精度
的选项。

出现下列任一情况时提前失败：精确配置明确不支持；所需精度不可用；目标需要训练或
微调；所需模型代码无法在 `trust_remote_code=False` 下运行；或无法固定 checkpoint、
预处理、输出契约。

**完成标准：** 两名 reviewer 只依赖记录即可选择同一 checkpoint，并构造相同输入和
期望输出。

## 阶段 2：追踪参考 forward

以 eval 模式运行 upstream 模型。按[验证策略](validation_policy.zh.md)分别固定精确同
dtype 参考和 CPU FP32 anchor。从预处理到返回输出完整追踪一次请求，并按执行顺序记录：

- module 或数学运算；
- 输入输出 shape 和 dtype；
- layout 变化；
- mask、position、lookup table 等语义输入；
- cache 或 recurrent state 读写；
- 用于实现一致性的同 dtype 值，以及作为高精度 anchor 的 CPU FP32 值。

语言模型分别覆盖 prefill 和 decode。Vision/VLA 覆盖每个组件边界和至少一个多输入
用例。不得用近似数学补齐缺失 forward。

**完成标准：** 每个返回值和持久状态更新都能追溯到 upstream producer。

## 阶段 3：映射到公开能力

每个参考步骤映射到最小的现有 RhinoForge 路径：

| 映射 | 必需证据 |
|---|---|
| 现有 adapter | 精确配置和输入范围通过其 preflight |
| 共享 causal decoder | 目标数学精确匹配模板使用的 `llama` 或 `qwen3` 分支 |
| 现有 RPU operation | 公开 schema 覆盖准确 shape、dtype、layout 和状态更新 |
| 已审查 CPU fallback | 保持正确性且明确列出，但不算 RPU 支持证据 |
| 新 host operation | 已定义数学、tensor 契约、验证、注册和 release 匹配的算子可用性 |
| 新 fused subsystem | 重复执行需要新的 `FusedModelBase` SPM manifest 和 layer subgraph |

检查 `python/rpu_backend/adapters/_template/`、
`src/core/rpu_dispatch_registrations.inc` 的公开注册和 `src/fused/` 实现。文件存在本身
不是能力声明。

每个缺口标为 reuse、adapter change、runtime change、asset change 或 blocker。
Release 匹配资产缺少必需 operation 时，必须在资产更新并验证前保持 blocker。

**完成标准：** 阶段 2 每一行都只有一个映射和 owner。

## 阶段 4：证明资源和生命周期可行

按公开 runtime 契约检查完整配置：

- 八核、每核 8 MiB SPM，以及约为可用 SPM 99% 的 `SPM_PLANNING_BUDGET`；
- FP16 主 dtype，以及要求处的 16-element 对齐；
- fused subsystem 的确定性、alias-aware `BufferDecl` 规划；
- replay 持久数据声明为 `Persistent` 或 `PersistentPerLayer`；
- fixed DMA 只用于稳定 storage，caller input/新 output 使用 mutable DMA；
- 每个语义输入进入 `GraphSignature`，或由验证过的 mutable transfer 刷新；
- Python 可见输出跨调用保留时使用独立 storage；
- 权重恰好转换一次，并遵守[架构](architecture.zh.md)中的进程 ownership 限制。

SPM 容量是硬 admission gate。更小 shape 通过不能证明最大范围；无法安全规划时应缩小
配置或标记 blocked。

**完成标准：** 最大允许输入有可行 memory plan，以及明确的 Graph、DMA、cache 和
output lifetime 契约。

## 阶段 5：实现前定义验证

先写出 port 必须通过的门禁：

1. 不支持配置在权重加载或转换前失败；
2. adapter registry 和执行配置通过无板卡测试；
3. 精确配置定义[验证策略](validation_policy.zh.md)要求的硬语义、同 dtype、FP32 anchor
   和代表性任务门禁；
4. `RPU_WARMUP=0`、`1`、`3` 默认输出 bit-identical；精确配置有已提交的有界
   repeatability 合同时按该合同验证；
5. 重复 signature 一次 BUILD 后稳定 REPLAY，cache size 固定且
   `cache_invariant_ok()` 为 true；
6. 相同 shape 的不同语义输入与 uncaptured 路径一致；
7. 多输入及适用时多图不会 alias 返回 storage；
8. 最大范围和公开 TOML example 在新进程运行。

运行入口见[模型运行与性能分析](model_testing.zh.md)。Profiler 输出是诊断证据，不是
数值门禁。

**完成标准：** 每个门禁明确输入、参考、期望结果和结果记录位置。

## 阶段 6：给出一个 outcome

| Outcome | 决策 |
|---|---|
| Existing path | 不需要移植，使用已有支持的精确配置 |
| Adapter-only | 现有公开 operation/runtime 覆盖完整 trace |
| Runtime extension | 缺少共享 host operation 或 runtime 行为 |
| New fused subsystem | 需要新的重复 SPM-resident 执行路径 |
| Blocked | operation、asset、shape、precision、memory plan、dependency 或数值门禁不可用 |

Outcome 旁列出假设和 blocker。只有批准的 decoder-only `Adapter-only` 才继续
[新模型逐步指南](new_model_step_by_step.zh.md)。`Existing path` 直接使用已有配置；
runtime extension、fused subsystem、vision 和多组件 policy 使用完整
[模型移植](model_porting.zh.md)流程。

所有门禁未在同一不可变源码、依赖、checkpoint、配置和资产集合上完成前，不得添加
support matrix 行。
