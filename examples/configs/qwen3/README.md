# Qwen3 配置

[配置总览](../README.md) · [全部配置](#configurations) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md) · 目录字段

按模型尺寸选择目录，再匹配已有权重的精度。32B 目前只有量化模板；需要 Instruct 行为时，使用明确标注 `fp16_instruct` 的配置及其对应 tokenizer。

<a id="快速选择"></a>

<a id="配置矩阵"></a>

0.6B、1.7B、4B、8B、14B 提供 FP16、W8、W4/G32；32B 提供 W8、W4/G32。0.6B 与 4B 另有 FP16 Instruct 配置。

<a id="选择规则"></a>

## 匹配权重、输入和核心数

W8 模板量化文本投影和输出头；W4/G32 分为文本投影，以及文本投影加输出头两类。按[全部配置](#configurations)中的量化范围和权重别名选择，避免混用产物。输入长度、解码长度和累加选项直接修改当前 TOML，同一配方共用权重。

模板默认 8 核，全部配置可选择 4/6/8 核；每个配置在各核数下的状态和 4/6 核时各组件实际核数见[配置表](#configurations)。

<a id="权重与运行"></a>

## 准备与运行

需要最少改动时，从对应尺寸的 FP16 模板开始；32B 只有量化模板，按现有配方准备权重。

1. 按[权重准备](../WEIGHTS.md)配置所选尺寸和配方的目录。
2. 打开对应 TOML，检查提示词或固定输入、解码步数及核心数。
3. 从仓库根目录先检查配置，再运行。下面使用 4B FP16：

```bash
python examples/run.py --config examples/configs/qwen3/text/4b/fp16.toml --check-config
python examples/run.py --config examples/configs/qwen3/text/4b/fp16.toml
```

检查通过会输出 `configuration OK`，只说明语法和登记 profile 匹配。更改输入长度、核心数或累加选项后，重新检查并启动新进程；输出格式见[结果说明](../../RESULTS.md)。

## 0.6B FP16 独立环境短例

首次安装按[从独立环境构建到运行](../../../docs/runtime_assets.zh.md)操作，使用固定
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` 的完整 checkpoint。
官方 BF16 存储权重由 FP16 profile 在加载时转换，不需要离线转换。

`text/0_6b/fp16.toml` 为 W2/R5（预热 2 次、正式 5 次），原始提示词在此 tokenizer 下为 P7，D4 表示 prefill
首 token 之后四次 decode，最多生成 5 个 token（EOS 可提前结束）。不使用 chat template；
它是执行短例，不能据此判断完整问答质量或长上下文性能。在本地配置设置
`[logging] level = "INFO"` 可显示生成文本；默认模型/性能摘要仍有独立开关。

`report.json` 和最后一次测量的 `output.pt` 位于结果子目录。`PASS` 说明执行完成，
普通 runner 不比较 CPU、不验证 Graph invariant；`correctness=not_run`、
`formal_certification=false` 保留这一边界。`--hwperf` 实际采集硬件 trace，但采样会
改变开销，正常计时比较保持关闭。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：Qwen3。

P 为处理后的输入 token 数，D 为 decode 调用数；生成首 token 来自 prefill。

### 文本生成 · 0.6B

| 配置 | 支持状态 | 精度 | 量化范围 | prefill_tokens 默认值 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|---|
| [text/0_6b/fp16.toml](text/0_6b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-0.6b` |
| [text/0_6b/fp16_instruct.toml](text/0_6b/fp16_instruct.toml) | 支持（`SUPPORTED`） | fp16 | — | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-0.6b-instruct` |
| [text/0_6b/w4a16_text_head_rtn_g32.toml](text/0_6b/w4a16_text_head_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text + output head | 128 | 7 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-0.6b` |
| [text/0_6b/w4a16_text_rtn_g32.toml](text/0_6b/w4a16_text_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text | 128 | 7 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-0.6b` |
| [text/0_6b/w8a16_text_head.toml](text/0_6b/w8a16_text_head.toml) | 支持（`SUPPORTED`） | w8a16 | text + output head | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-0.6b-w8a16-lmhead-int8` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / token IDs |
| prompt 默认示例 | Explain what a compiler does. |
| 文本可调范围 | 非空文本或非负 token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | P 由分词结果决定；显式 prefill_tokens 会截取/重复文本。cache 按 P + max(D, 补齐行) 分配（补齐最多 64 行），另受位置容量、原生执行形状和内存约束。 此尺寸多token prefill物理执行包络≤4096，chunk≤512。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[config.py](../../../examples/common/config.py) · [generation.py](../../../examples/common/generation.py) · [text.py](../../../examples/common/text.py) · [qwen3.py](../../../python/rpu_backend/adapters/qwen3.py)。

### 文本生成 · 1.7B

| 配置 | 支持状态 | 精度 | 量化范围 | prefill_tokens 默认值 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|---|
| [text/1_7b/fp16.toml](text/1_7b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-1.7b` |
| [text/1_7b/w4a16_text_head_rtn_g32.toml](text/1_7b/w4a16_text_head_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text + output head | 128 | 7 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-1.7b` |
| [text/1_7b/w4a16_text_rtn_g32.toml](text/1_7b/w4a16_text_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text | 128 | 7 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-1.7b` |
| [text/1_7b/w8a16_text_head.toml](text/1_7b/w8a16_text_head.toml) | 支持（`SUPPORTED`） | w8a16 | text + output head | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-1.7b-w8a16-lmhead-int8` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / token IDs |
| prompt 默认示例 | Explain what a compiler does. |
| 文本可调范围 | 非空文本或非负 token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | P 由分词结果决定；显式 prefill_tokens 会截取/重复文本。cache 按 P + max(D, 补齐行) 分配（补齐最多 64 行），另受位置容量、原生执行形状和内存约束。 此尺寸多token prefill物理执行包络≤4096，chunk≤512。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[config.py](../../../examples/common/config.py) · [generation.py](../../../examples/common/generation.py) · [text.py](../../../examples/common/text.py) · [qwen3.py](../../../python/rpu_backend/adapters/qwen3.py)。

### 文本生成 · 4B

| 配置 | 支持状态 | 精度 | 量化范围 | prefill_tokens 默认值 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|---|
| [text/4b/fp16.toml](text/4b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-4b` |
| [text/4b/fp16_instruct.toml](text/4b/fp16_instruct.toml) | 支持（`SUPPORTED`） | fp16 | — | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-4b-instruct` |
| [text/4b/w4a16_text_head_rtn_g32.toml](text/4b/w4a16_text_head_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text + output head | 128 | 7 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-4b` |
| [text/4b/w4a16_text_rtn_g32.toml](text/4b/w4a16_text_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text | 128 | 7 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-4b` |
| [text/4b/w8a16_text_head.toml](text/4b/w8a16_text_head.toml) | 支持（`SUPPORTED`） | w8a16 | text + output head | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-4b-w8a16-lmhead-int8` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / token IDs |
| prompt 默认示例 | Explain what a compiler does. |
| 文本可调范围 | 非空文本或非负 token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | P 由分词结果决定；显式 prefill_tokens 会截取/重复文本。cache 按 P + max(D, 补齐行) 分配（补齐最多 64 行），另受位置容量、原生执行形状和内存约束。 此尺寸多token prefill物理执行包络≤4096，chunk≤256。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[config.py](../../../examples/common/config.py) · [generation.py](../../../examples/common/generation.py) · [text.py](../../../examples/common/text.py) · [qwen3.py](../../../python/rpu_backend/adapters/qwen3.py)。

### 文本生成 · 8B

| 配置 | 支持状态 | 精度 | 量化范围 | prefill_tokens 默认值 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|---|
| [text/8b/fp16.toml](text/8b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-8b` |
| [text/8b/w4a16_text_head_rtn_g32.toml](text/8b/w4a16_text_head_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text + output head | 128 | 7 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-8b` |
| [text/8b/w4a16_text_rtn_g32.toml](text/8b/w4a16_text_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text | 128 | 7 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-8b` |
| [text/8b/w8a16_text_head.toml](text/8b/w8a16_text_head.toml) | 支持（`SUPPORTED`） | w8a16 | text + output head | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-8b-w8a16-lmhead-int8` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / token IDs |
| prompt 默认示例 | Explain what a compiler does. |
| 文本可调范围 | 非空文本或非负 token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | P 由分词结果决定；显式 prefill_tokens 会截取/重复文本。cache 按 P + max(D, 补齐行) 分配（补齐最多 64 行），另受位置容量、原生执行形状和内存约束。 此尺寸多token prefill物理执行包络≤4096，chunk≤128。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[config.py](../../../examples/common/config.py) · [generation.py](../../../examples/common/generation.py) · [text.py](../../../examples/common/text.py) · [qwen3.py](../../../python/rpu_backend/adapters/qwen3.py)。

### 文本生成 · 14B

| 配置 | 支持状态 | 精度 | 量化范围 | prefill_tokens 默认值 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|---|
| [text/14b/fp16.toml](text/14b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-14b` |
| [text/14b/w4a16_text_head_rtn_g32.toml](text/14b/w4a16_text_head_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text + output head | 128 | 7 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-14b` |
| [text/14b/w4a16_text_rtn_g32.toml](text/14b/w4a16_text_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text | 128 | 7 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-14b` |
| [text/14b/w8a16_text_head.toml](text/14b/w8a16_text_head.toml) | 支持（`SUPPORTED`） | w8a16 | text + output head | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-14b-w8a16-lmhead-int8` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / token IDs |
| prompt 默认示例 | Explain what a compiler does. |
| 文本可调范围 | 非空文本或非负 token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | P 由分词结果决定；显式 prefill_tokens 会截取/重复文本。cache 按 P + max(D, 补齐行) 分配（补齐最多 64 行），另受位置容量、原生执行形状和内存约束。 此尺寸多token prefill物理执行包络≤192，chunk≤32。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[config.py](../../../examples/common/config.py) · [generation.py](../../../examples/common/generation.py) · [text.py](../../../examples/common/text.py) · [qwen3.py](../../../python/rpu_backend/adapters/qwen3.py)。

### 文本生成 · 32B

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|
| [text/32b/w4a16_text_head_rtn_g32.toml](text/32b/w4a16_text_head_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text + output head | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-32b` |
| [text/32b/w4a16_text_rtn_g32.toml](text/32b/w4a16_text_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-32b` |
| [text/32b/w8a16_text_head.toml](text/32b/w8a16_text_head.toml) | 支持（`SUPPORTED`） | w8a16 | text + output head | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3-32b-w8a16-lmhead-int8` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / token IDs |
| prompt 默认示例 | Explain what a compiler does. |
| prefill_tokens 默认值 | 128 |
| decode_steps 默认值 | 7 |
| 文本可调范围 | 非空文本或非负 token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | P 由分词结果决定；显式 prefill_tokens 会截取/重复文本。cache 按 P + max(D, 补齐行) 分配（补齐最多 64 行），另受位置容量、原生执行形状和内存约束。 此尺寸多token prefill物理执行包络≤4096，chunk≤64。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[config.py](../../../examples/common/config.py) · [generation.py](../../../examples/common/generation.py) · [text.py](../../../examples/common/text.py) · [qwen3.py](../../../python/rpu_backend/adapters/qwen3.py)。

<!-- model-configurations-end -->
