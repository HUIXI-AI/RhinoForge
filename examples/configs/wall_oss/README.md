# Wall-OSS 配置

[配置总览](../README.md) · [全部配置](#configurations) · [核数支持矩阵](../../../docs/planner_core_topology.zh.md) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

按任务和输入资产选择入口。Base 使用预处理 batch；Legacy、RTC 接收三路图像与状态；Sorting 需要分拣模型的配套 batch 和权重。

<a id="配置矩阵"></a>

| 用途 | 输入 | 精度/配方 | 入口 |
|---|---|---|---|
| Base | 2 或 3 路相机的 batch，224×224 或 256×256 | FP16、W8A16、W8A16 + Action NVFP4 | [Base 指南](base/README.md) |
| Legacy | 三路图像、指令、状态和状态掩码 | FP16、W4A16、W8A16、W4A16 G32 | [Legacy 指南](legacy/README.md) |
| RTC | 三路图像、指令、状态和推理延迟 | FP16、W8A16 | [RTC 指南](rtc/README.md) |
| Sorting | 分拣任务的配套 batch | FP16、W8A16、W8A16 + Action W4A16 G32 | [Sorting 指南](sorting/README.md) |

<a id="运行"></a>

## 运行 Base 示例

按 [Base 指南](base/README.md)填写权重和 batch 路径，从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/wall_oss/base/2cam/224/fp16.toml --check-config
python examples/run.py --config examples/configs/wall_oss/base/2cam/224/fp16.toml
```

输入文本长度、动作范围、核心布局与权重别名见[全部配置](#configurations)。各入口的资产和登记状态独立，不能把 Base 的运行结果用于判断 RTC 或 Sorting 的支持情况。


<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：Wall-OSS。

### Base 动作

| 配置 | 支持状态 | 精度 | 量化范围 | cameras 默认值 | image_size 默认值 | 4 核 | 6 核 | 8 核 | CPU线程数 | 权重 |
|---|---|---|---|---|---|---|---|---|---|---|
| [base/2cam/224/fp16.toml](base/2cam/224/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 2 | [224,224] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | 4 | `/path/to/wall-oss-0.5`；准备方法见[权重准备](../WEIGHTS.md) |
| [base/2cam/224/w8a16.toml](base/2cam/224/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision、decoder 与 action_preprocessor 的二维 Linear W8（embed_tokens 除外） | 2 | [224,224] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | 8 | fp16_checkpoint `/path/to/wall-oss-0.5`、w8_checkpoint `/path/to/wall-oss-0.5-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [base/2cam/224/w8a16_action_nvfp4.toml](base/2cam/224/w8a16_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_action_nvfp4 | Vision / Prefill W8；Action NVFP4 | 2 | [224,224] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | 4 | fp16_checkpoint `/path/to/wall-oss-0.5`、w8_checkpoint `/path/to/wall-oss-0.5-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [base/2cam/256/fp16.toml](base/2cam/256/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 2 | [256,256] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | 4 | `/path/to/wall-oss-0.5`；准备方法见[权重准备](../WEIGHTS.md) |
| [base/2cam/256/w8a16.toml](base/2cam/256/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision、decoder 与 action_preprocessor 的二维 Linear W8（embed_tokens 除外） | 2 | [256,256] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | 8 | fp16_checkpoint `/path/to/wall-oss-0.5`、w8_checkpoint `/path/to/wall-oss-0.5-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [base/2cam/256/w8a16_action_nvfp4.toml](base/2cam/256/w8a16_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_action_nvfp4 | Vision / Prefill W8；Action NVFP4 | 2 | [256,256] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | 8 | fp16_checkpoint `/path/to/wall-oss-0.5`、w8_checkpoint `/path/to/wall-oss-0.5-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [base/3cam/224/fp16.toml](base/3cam/224/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 3 | [224,224] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | 4 | `/path/to/wall-oss-0.5`；准备方法见[权重准备](../WEIGHTS.md) |
| [base/3cam/224/w8a16.toml](base/3cam/224/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision、decoder 与 action_preprocessor 的二维 Linear W8（embed_tokens 除外） | 3 | [224,224] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | 12 | fp16_checkpoint `/path/to/wall-oss-0.5`、w8_checkpoint `/path/to/wall-oss-0.5-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [base/3cam/224/w8a16_action_nvfp4.toml](base/3cam/224/w8a16_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_action_nvfp4 | Vision / Prefill W8；Action NVFP4 | 3 | [224,224] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | 8 | fp16_checkpoint `/path/to/wall-oss-0.5`、w8_checkpoint `/path/to/wall-oss-0.5-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [base/3cam/256/fp16.toml](base/3cam/256/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 3 | [256,256] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | 8 | `/path/to/wall-oss-0.5`；准备方法见[权重准备](../WEIGHTS.md) |
| [base/3cam/256/w8a16.toml](base/3cam/256/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision、decoder 与 action_preprocessor 的二维 Linear W8（embed_tokens 除外） | 3 | [256,256] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | 12 | fp16_checkpoint `/path/to/wall-oss-0.5`、w8_checkpoint `/path/to/wall-oss-0.5-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [base/3cam/256/w8a16_action_nvfp4.toml](base/3cam/256/w8a16_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_action_nvfp4 | Vision / Prefill W8；Action NVFP4 | 3 | [256,256] | ✕ 不支持 | ✕ 不支持 | ● 已实测 | 12 | fp16_checkpoint `/path/to/wall-oss-0.5`、w8_checkpoint `/path/to/wall-oss-0.5-w8a16`；准备方法见[权重准备](../WEIGHTS.md) |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | saved batch / batch_template |
| text_tokens 默认值 | 32 |
| num_steps 默认值 | 10 |
| action_horizon 默认值 | 32 |
| 文本可调范围 | text_tokens正整数，需匹配实际instruction或non_image计数；instruction范围下真实prefix为2cam224 244+T、2cam256 278+T、3cam224 319+T、3cam256 370+T行，须≤640，即T≤396、362、321、270 |
| 图像/音频可调范围 | 2/3相机；image_size=[224,224]或[256,256] |
| 生成可调范围 | num_steps=10；horizon=32 |
| 其他输入形状 | batch含对应state与noise |
| 联动限制 | batch/template二选一；template只允许cameras/image_size/text_tokens占位符。vision_window仅224，vision_pair仅256；指令不截断/填充。prefill执行行数：真实prefix≤384行补到16的倍数（单chunk），385–448、449–512、513–640行执行448、512、640行（等分两chunk），超过640行报错。 |
| 默认核数 | 8 |
| 入口 | [run.py](../../run.py) |

- ✕ 4/6 核不支持：原生实现只有 8 核：语言模型的注意力和 MLP 固定 8 核，动作专家和 Qwen2.5-VL 视觉 owner 也按 8 核编写。

输入校验来源：[base.py](../../../examples/runners/wall_oss/base.py) · [base_engine.py](../../../examples/runners/wall_oss/base_engine.py)。

### 历史动作 profile

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|
| [legacy/fp16.toml](legacy/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `wall-oss-0.5` |
| [legacy/w4a16_channel.toml](legacy/w4a16_channel.toml) | 支持（`SUPPORTED`） | w4a16 | decoder 专家投影逐通道 W4 | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `wall-oss-0.5-w4a16` |
| [legacy/w4a16_g32.toml](legacy/w4a16_g32.toml) | 支持（`SUPPORTED`） | w4a16-pgrp | decoder 专家投影 W4 G32（expert-0 down 保持 INT8） | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `wall-oss-0.5-w4a16-pgrp` |
| [legacy/w8a16.toml](legacy/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision、decoder 与 action_preprocessor 的二维 Linear W8（embed_tokens 除外） | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `wall-oss-0.5-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | images + instruction + state |
| 文本可调范围 | 匹配精确profile的prefix范围 |
| 图像/音频可调范围 | 非空images；camera_names数量一致 |
| 生成可调范围 | 匹配原动作profile |
| 其他输入形状 | 4 个 profile 都要求 3 相机、每图 448×252（宽×高），真实 prefix 545–640 |
| 联动限制 | 精度仅FP16/W8/W4；量化需浮点父权重。保留原profile输入合同，不继承新Base的224/256范围。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

- ✕ 4/6 核不支持：原生实现只有 8 核：语言模型的注意力和 MLP 固定 8 核，动作专家和 Qwen2.5-VL 视觉 owner 也按 8 核编写。

输入校验来源：[runner.py](../../../examples/runners/wall_oss/runner.py) · [runtime.py](../../../python/rpu_backend/adapters/wall_oss/runtime.py)。

### RTC release

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|
| [rtc/fp16.toml](rtc/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/wrc_sf_v728`（profile 登记名 `wall-oss-0.5`）；准备方法见[权重准备](../WEIGHTS.md) |
| [rtc/w8a16.toml](rtc/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action W8（配套 W8 checkpoint） | ✕ 不支持 | ✕ 不支持 | ● 已实测 | checkpoint `/path/to/wrc_sf_v728-w8a16`、fp16_checkpoint `/path/to/wrc_sf_v728`（profile 登记名 `wall-oss-0.5-w8a16`）；准备方法见[权重准备](../WEIGHTS.md) |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 图片 + instruction + proprioception |
| inference_delay 默认值 | 0 |
| 文本可调范围 | 实际 prefix 599–671：≤640 时执行 P640/chunk 320，641–671 时执行 P672/chunk 336；模板的 prefix_length_range 为 [599, 640]，要用 641–671 须同时把它扩到 [599, 671] |
| 图像/音频可调范围 | 固定3图；图像尺寸由release处理器决定 |
| 生成可调范围 | 按RTC release运行；inference_delay 0–10，>0 时须提供 rtc_prefix_actions |
| 其他输入形状 | release专用proprioception与动作历史 |
| 联动限制 | W8还需匹配浮点checkpoint提供CPU glue；Graph与权重必须匹配RTC精确profile。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

- ✕ 4/6 核不支持：原生实现只有 8 核：语言模型的注意力和 MLP 固定 8 核，动作专家和 Qwen2.5-VL 视觉 owner 也按 8 核编写。

输入校验来源：[rtc.py](../../../examples/runners/wall_oss/rtc.py) · [runtime.py](../../../python/rpu_backend/adapters/wall_oss/runtime.py)。

### 分拣动作

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|
| [sorting/fp16.toml](sorting/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ✕ 不支持 | ✕ 不支持 | ● 已实测 | `/path/to/wrc_sf_v728_0_15000`；准备方法见[权重准备](../WEIGHTS.md) |
| [sorting/w8a16.toml](sorting/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision / Prefill / Action W8（配套 W8 checkpoint） | ✕ 不支持 | ✕ 不支持 | ● 已实测 | fp16_checkpoint `/path/to/wrc_sf_v728_0_15000`、w8_checkpoint `/path/to/wrc_v728_w8a16`；准备方法见[权重准备](../WEIGHTS.md) |
| [sorting/w8a16_action_w4a16_g32.toml](sorting/w8a16_action_w4a16_g32.toml) | 支持（`SUPPORTED`） | w8a16_action_w4a16 | Vision / Prefill（expert0，含共享 KV）W8；Action（expert1）W4A16 G32 | ✕ 不支持 | ✕ 不支持 | ● 已实测 | fp16_checkpoint `/path/to/wrc_sf_v728_0_15000`、w8_checkpoint `/path/to/wrc_v728_w8a16`、action_w4_checkpoint `/path/to/wrc_v728_action_w4a16_pgrp`；准备方法见[权重准备](../WEIGHTS.md) |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 专用 tensor batch |
| inference_delay 默认值 | 0 |
| 文本可调范围 | 非空instruction；长度受exact runtime prefix约束 |
| 图像/音频可调范围 | Uint8 images[3,252,448,3] |
| 生成可调范围 | num_steps=6；inference_delay∈{0,1,3,10} |
| 其他输入形状 | state/state_mask/dof_mask [26]；noise[1,20,26] |
| 联动限制 | mask前20真、后6假；非零delay需prefix_actions[20,26]，零delay禁止携带；浮点tensor必须有限。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

- ✕ 4/6 核不支持：原生实现只有 8 核：语言模型的注意力和 MLP 固定 8 核，动作专家和 Qwen2.5-VL 视觉 owner 也按 8 核编写。

输入校验来源：[sorting.py](../../../examples/runners/wall_oss/sorting.py)。

<!-- model-configurations-end -->
