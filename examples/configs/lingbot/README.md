# LingBot 配置

[配置总览](../README.md) · [全部配置](#configurations) · [核数支持矩阵](../../../docs/planner_core_topology.zh.md) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

先确认权重版本，再选配置。v2 常规输入、v1 混合执行路径和历史 exact-Z2 使用不同的输入约束，不能只换权重目录复用配置。

<a id="配置矩阵"></a>

| 路径 | 输入与执行方式 | 入口 |
|---|---|---|
| v2 | 2 或 3 路相机；FP16/W8A16；文本长度通过参数设置 | [v2 指南](v2/README.md) |
| v1 | 配套 observation；CPU FP32 Vision + RPU FP16 | [4B 模板](v1/4b/fp16_cpu_vision.toml) |
| v2 Legacy exact-Z2 | 历史专用 observation，W8A16 | [exact-Z2 指南](legacy/exact_z2/README.md) |

v1 的后续测试目前暂缓，已有实现和登记记录保留；这项安排不涉及 v2。具体输入范围和核心数见[全部配置](#configurations)。

<a id="运行"></a>

## 运行 v2 示例

按 [v2 指南](v2/README.md)准备 checkpoint、training config 和 observation，从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/lingbot/v2/fp16.toml --check-config
python examples/run.py --config examples/configs/lingbot/v2/fp16.toml
```

`--check-config` 只检查配置和输入参数范围；实际 observation 的 token 数与相机数量在运行时核对。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：LingBot。

### v1 observation · 4B

| 项目 | 值 |
|---|---|
| 配置 | [v1/4b/fp16_cpu_vision.toml](v1/4b/fp16_cpu_vision.toml) |
| 支持状态 | 受限支持（`SUPPORTED_LIMITED`）；未实测：缺少 observation 输入（CPU FP32 golden 需用 internal validation notes 的 lingbotvla 环境生成），不是硬件限制 |
| 精度 | fp16 |
| 量化范围 | Vision CPU FP32 |
| 输入类型 | tensor observation |
| num_steps 默认值 | 10 |
| 文本可调范围 | 登记logical prefix≤178；max_seq默认256 |
| 图像/音频可调范围 | 登记1/2图；Vision使用CPU FP32 |
| 生成可调范围 | num_steps 必须为 10 |
| 其他输入形状 | 匹配原v1 observation |
| 联动限制 | 后续测试暂缓；不继承v2的相机、token和shape范围。 |
| 默认核数 | 8 |
| 4 核 | ✕ 不支持 |
| 6 核 | ✕ 不支持 |
| 8 核 | ○ 未实测 |
| CPU线程数 | 未设置（沿用环境） |
| 权重 | `lingbot-vla-4b` |
| 入口 | [run.py](../../run.py) |

- ✕ 4/6 核不支持：延期模型：只有 8 核实现。

输入校验来源：[runner.py](../../../examples/runners/lingbot_v1/runner.py) · [runtime.py](../../../python/rpu_backend/adapters/lingbot_vla/runtime.py)。

### v2 图文动作

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 |
|---|---|---|---|---|---|---|
| [v2/fp16.toml](v2/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测 | ● 已实测 | ● 已实测 |
| [v2/w8a16.toml](v2/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / VLM prefill / 动作专家 W8A16（动作专家 router FP16） | ● 已实测 | ● 已实测 | ● 已实测 |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | case / observation / 原始图文+state |
| expected_language_tokens 默认值 | 32 |
| max_language_tokens 默认值 | 72 |
| cameras 默认值 | 3 |
| 文本可调范围 | 1≤L≤max_language_tokens≤160；默认cap72；expected_language_tokens可选 |
| 图像/音频可调范围 | 1/2/3相机（1相机 FP16 6/8 核、W8A16 6 核已在板上通过）；处理后Float32[C,256,1536]，grid每行[1,16,16] |
| 生成可调范围 | 固定动作生成流程 |
| 其他输入形状 | language/mask [1,L]；原始state=1..55个有限数→[1,55]；noise文件可选→[1,50,55] |
| 联动限制 | 三种输入互斥。expected_language_tokens设定时有效mask计数须严格相等；改T需同步cap和真实输入；原始指令不静默截断。case/observation自带state/noise/instruction。prefix为1cam 66+T、2cam 132+T、3cam 198+T行（training_config含align_params时各加16行），对齐到16行后最多再补prefill.padding_budget行；模板为64，T1–160都能补到可行长度；改小时对齐后272/304/336/352/368/384行没有可行布局（2026-10-23 8核板测）。model.num_cores为4/6/8（省略为8）；4/6核时Vision和动作专家用4核、文本骨干Attention 4/MLP全部核（6核时MLP补到9792），Vision逐图、动作专家只走稀疏专家路由；4/6核的prefill可行长度尚无板测。 |
| 默认核数 | 8 |
| CPU线程数 | 8 核 1，4/6 核 12（入口默认） |
| 权重 | `${RPU_MODEL_CACHE}/lingbot-vla-v2-6b`；准备方法见[权重准备](../WEIGHTS.md) |
| 入口 | [run.py](../../run.py) |

输入校验来源：[inputs.py](../../../examples/runners/lingbot2/inputs.py) · [optimized.py](../../../examples/runners/lingbot2/optimized.py)。

### 历史 exact-Z2 · 6B

| 项目 | 值 |
|---|---|
| 配置 | [legacy/exact_z2/6b/w8a16.toml](legacy/exact_z2/6b/w8a16.toml) |
| 支持状态 | 支持（`SUPPORTED`） |
| 精度 | w8a16 |
| 量化范围 | Vision / VLM prefill / 动作专家 W8A16（动作专家 router FP16） |
| 输入类型 | 专用 observation |
| 文本可调范围 | 由 exact-Z2 fixture 给定 |
| 图像/音频可调范围 | 专用W8视觉布局 |
| 生成可调范围 | 固定 exact-Z2 流程 |
| 其他输入形状 | 模板 prefix_length_range=[215,225] |
| 联动限制 | 只用于历史exact-Z2复现，不继承常规v2输入范围。 |
| 默认核数 | 8 |
| 4 核 | ✕ 不支持 |
| 6 核 | ✕ 不支持 |
| 8 核 | ● 已实测 |
| CPU线程数 | 未设置（沿用环境） |
| 权重 | `lingbot-vla-v2-6b` |
| 入口 | [run.py](../../run.py) |

- ✕ 4/6 核不支持：exact-Z2 的 SPM 驻留计划按 8 核切分（每核 8,303,616 字节），少核需要新的驻留计划。

输入校验来源：[runner.py](../../../examples/runners/lingbot2/runner.py) · [runtime.py](../../../python/rpu_backend/adapters/lingbot_vla_v2/runtime.py)。

<!-- model-configurations-end -->
