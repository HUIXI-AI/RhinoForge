# 模型验证策略

简体中文 | [English](validation_policy.md)

RhinoForge 验证的是精确模型配置，而不是模型家族名称。配置由 checkpoint revision、
预处理、精度、输入范围、执行设置、Rhino Launch 包、合并算子资产和 RhinoForge commit
共同确定。一个尺寸、量化格式或输入范围的结论不能自动转移到另一个配置。

即使数学计算等价，不同设备与实现的浮点结果也可能不同。单一全局 cosine 阈值，特别
是所有 token 或 feature 行中的最小值，不足以决定产品质量。RhinoForge 将语义、实现
一致性、任务质量和运行时生命周期作为彼此独立的门禁。

## 1. 固定证据身份

评估前记录：

- 源码、依赖、Launch 包和算子资产版本；
- checkpoint、tokenizer/processor 和配置 hash；
- 精度和量化流程；
- 输入数据集或 fixture revision、随机种子，以及适用时抽样得到的噪声；
- 允许的 sequence、image、batch、cache 和执行设置。

只要参考侧身份和输入 hash 全部不变，CPU FP32 golden 可以复用。仅 backend 源码变化时
需要重新比较 RPU，不必重做 CPU golden。

## 2. 硬语义与运行时门禁

以下检查是二元结果，不能用较好的端到端分数豁免：

- 输出形状、dtype、顺序、mask、position、cache 更新和逻辑长度符合模型契约；
- 公共契约要求 finite 的输出与状态不存在意外 NaN/Inf，调用方保留的输出拥有独立
  storage；显式 mask sentinel 仍遵守算子自身契约；
- 不支持的配置在不可逆权重转换前失败；
- 权重只转换一次，Graph signature、DMA ownership 和持久状态生命周期正确；
- 重复运行符合配置声明的重复性契约；随机配置比较相同 seed、noise 或抽样状态；
- 适用路径证明 BUILD 后稳定 REPLAY，或完整证明受限 one-shot 契约。

如果实现存在已知的层映射、索引、cache 或状态生命周期错误，即使聚合分数看似正常，
也必须在本门禁失败。

## 3. 实现一致性

两个参考回答不同问题：

| 参考 | 问题 |
|---|---|
| 同 dtype 参考 | RPU 是否保持所选 FP16 或量化执行路径的语义？ |
| CPU FP32 anchor | 相对更高精度模型损失了多少准确性？ |

“同 dtype”仍不是完整身份。必须固定参考设备/backend、框架和库版本、累加 dtype、融合
设置、模型模式、确定性设置和输入 hash；任一项变化都先视为不同参考，直到证明等价。

顶层一致性或任务证据失败时，先定位第一个分歧的算子或组件。逐层比对是诊断升级，
不是所有已通过模型的发布必跑项。高维 tensor 应报告分布，而非只报一个极值：

- 行 cosine 的 mean、低百分位和 minimum；
- relative L2 与 maximum absolute error；
- 被消费 logits 的 top-1 或 top-k 一致率；
- 非 finite 或语义不匹配的数量和位置。

零行或近常量行应使用绝对/相对容差而非 cosine。图像 placeholder logits、padding 行和
其他公共输出不会消费的值，应与被消费的 text、decode、feature 或 action 输出分开报告。

仓库刻意不定义统一的逐行 cosine 通过值。每个配置在候选运行前固定：

```text
reference identity: <backend、版本、dtype、累加与融合>
consumed tensors:   <tensor 名和行/token 选择>
parity metrics:     <cosine 分布、relative-L2、max-abs、一致率>
thresholds:         <每项阈值，以及零值/near-tie 处理>
rationale:          <参考波动与下游敏感性>
```

算子门禁可以比完整低精度模型更严格。看到失败结果后不得直接放宽阈值；需要审查说明和
新的证据版本。尚无校准契约的家族应先用指标分布定位错误，并保持 `Experimental` 或
`Source-only`，直到任务关联契约被批准。

量化配置做两次比较：RPU 对精确量化同 dtype 参考用于实现一致性；量化结果对 FP16 或
FP32 anchor 用于保留任务质量。

## 4. 任务和端到端质量

`Supported` 和 `Limited` 配置需要固定、有代表性的评估集，以及与公共输出对应的指标：

| 家族 | 主要证据示例 |
|---|---|
| Causal LM | perplexity 或任务准确率、teacher-forced token 一致性、固定 prompt 生成 |
| VLM | 文本任务质量，加图像条件和多图用例 |
| Vision encoder | 下游检索、分类或 checkpoint 自有 feature 指标 |
| VLA | 归一化 action error/一致性，加 checkpoint 自有离线或闭环 benchmark |

任务目标在候选运行前固定。某些准确率指标可以用 99% 保留率，单独命名的高精度配置
也可能用 99.9%；这些只是例子，不是仓库默认值。其他任务可要求不同相对目标或绝对
误差界。对正的 higher-is-better 指标，保留率为 `candidate / reference`；对正的
lower-is-better 指标，为 `reference / candidate`。接近零、有符号、安全上限或物理单位
action 应使用绝对界，而不是比值。

greedy token 精确一致是有用证据，但固定任务指标通过时，一个数值稳定的 near-tie 不必
使整个模型失败。所有非精确结果仍要计数和解释，不能悄悄转成 exact pass。VLA 数值验证
不认证机器人安全、坐标系、执行器限制或部署就绪。

## 5. 状态决策

| 状态 | 验证含义 |
|---|---|
| Supported | 精确公开配置的全部硬语义、一致性、任务、生命周期和最大范围门禁通过 |
| Limited | 与 Supported 同样质量标准通过，但承诺范围明确更窄 |
| Experimental | 硬语义、要求的 finite 输出、声明的重复性和安全拒绝通过；一致性或代表性任务证据仍可不完整 |
| Component-only | 指定组件通过组件契约，不声明完整产品模型 |
| Source-only | 提供源码但不承诺可运行或质量；适用时已知不支持路径安全拒绝 |
| Unsupported | 不提供可运行的公开配置契约 |

一个模型专用验证失败通常只阻塞该配置的支持声明、资产或 example。只有共享路径存在
未隔离的语义错误、数据损坏、非 finite 输出、不安全 mutation，或无法在使用前拒绝受
影响配置时，才阻塞整个源码发布。

## 6. 最小验证记录

`Supported` 或 `Limited` 的证据至少包含：

1. 第 1 节的完整证据身份；
2. 硬门禁结果和最大允许范围；
3. 同 dtype 与 FP32 anchor 指标分布；
4. 任务指标、参考值、候选值和保留率计算；
5. warmup、重复性、Graph 生命周期和多输入结果；
6. 所有例外、未运行门禁和 owner 批准。

Smoke 和 profiler trace 仅是诊断证据。未运行的门禁保持 pending，不能从另一模型尺寸或
更小输入推断。

背景参考：PyTorch 的[数值精度](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
与[可复现性](https://docs.pytorch.org/docs/stable/notes/randomness.html)。
[MLCommons Inference](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
展示了按任务选择相对/绝对质量目标的做法；RhinoForge 同样要求每个配置选择并说明
适合该任务的指标和目标。
