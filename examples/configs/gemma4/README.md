# Gemma4 配置

[配置总览](../README.md) · [全部配置](#configurations) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

<a id="配置矩阵"></a>

[E4B FP16](e4b/fp16.toml) 提供文本生成入口。支持状态与 4/6/8 核见[配置表](#configurations)；“支持”表示该配置在板上通过运行时检查，数值误差见性能报告。

[W8A16](e4b/w8a16_text.toml) 和 [W4A16 G32](e4b/w4a16_text_rtn_g32.toml) 读取同一个浮点权重目录，在加载时量化 decoder 每层的七个投影，其余保持原精度；两者 2026-11-06 首次上板并通过运行时检查（8 核；W4A16 另跑 6 核，耗时见下段），细节见[量化指南](../../../docs/quantization.zh.md)。

解码层和与词嵌入共享的输出头在 RPU 上运行（输出头 FP16 权重、FP32 累加，8 核时用 8 核，4/6 核时用 4 核），logits 回到 CPU 做最后的 softcap；词嵌入和 PLE 侧输入在 CPU 上计算。核数改变解码层的切分（注意力固定 2 核，MLP 用 4/6/8 核）和输出头的核数。2026-10-21 输出头还在 CPU 上时，实测 4/6/8 核的 TPOT 都约 1.7 s。2026-11-06 输出头移到 RPU 后，同一输入（P6 / D4，预热 2 次、正式 5 次）实测 TPOT：FP16 8 核约 106 ms，W8A16 8 核约 73 ms，W4A16 6 核约 73 ms。

<a id="运行"></a>

## 准备与运行

按权重准备页配置 `gemma4-e4b` 目录，在 TOML 中填写 `input.prompt` 和 `input.decode_steps`；默认生成 4 步。模型与 tokenizer 从本地权重目录加载。

从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/gemma4/e4b/fp16.toml --check-config
python examples/run.py --config examples/configs/gemma4/e4b/fp16.toml
```

`--check-config` 通过只说明配置与登记 profile 匹配；生成文本、token 和运行记录见[结果说明](../../RESULTS.md)。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：Gemma4。

P 为处理后的输入 token 数，D 为 decode 调用数；生成首 token 来自 prefill。

### E4B 文本

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 |
|---|---|---|---|---|---|---|
| [e4b/fp16.toml](e4b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测（注意力 2 · MLP 4） | ● 已实测（注意力 2 · MLP 6） | ● 已实测 |
| [e4b/w4a16_text_rtn_g32.toml](e4b/w4a16_text_rtn_g32.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | w4a16 | 文本层 W4A16 G32 RTN | ● 已实测（注意力 2 · MLP 4） | ● 已实测（注意力 2 · MLP 6） | ● 已实测 |
| [e4b/w8a16_text.toml](e4b/w8a16_text.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | w8a16 | 文本层 W8A16 | ● 已实测（注意力 2 · MLP 4） | ● 已实测（注意力 2 · MLP 6） | ● 已实测 |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / token IDs |
| prompt 默认示例 | Explain what a compiler does. |
| decode_steps 默认值 | 4 |
| 文本可调范围 | 非空文本/token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | 安装和分配max(512,P+D)，不得超过checkpoint max_position_embeddings；cache与RoPE horizon共同校验。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 权重 | `gemma4-e4b` |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/gemma4/runner.py) · [__init__.py](../../../python/rpu_backend/adapters/gemma4/__init__.py)。

<!-- model-configurations-end -->
