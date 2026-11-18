# Llama 配置

[配置总览](../README.md) · [全部配置](#configurations) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

<a id="配置矩阵"></a>

[Llama 3.2 1B FP16](3_2_1b/fp16.toml) 用于文本生成。先准备 `llama-3.2-1b` 对应权重，再在 TOML 中设置提示词和解码步数。

## 运行

从仓库根目录先检查配置，再运行：

```bash
python examples/run.py --config examples/configs/llama/3_2_1b/fp16.toml --check-config
python examples/run.py --config examples/configs/llama/3_2_1b/fp16.toml
```

输入长度、解码长度和核心范围见[全部配置](#configurations)。调整这些参数后，在新进程中重新加载配置；输出字段见[结果说明](../../RESULTS.md)。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：Llama。

P 为处理后的输入 token 数，D 为 decode 调用数；生成首 token 来自 prefill。

### Llama 文本

| 项目 | 值 |
|---|---|
| 配置 | [3_2_1b/fp16.toml](3_2_1b/fp16.toml) |
| 支持状态 | 支持（`SUPPORTED`） |
| 精度 | fp16 |
| 量化范围 | — |
| 输入类型 | 文本 / token IDs |
| prompt 默认示例 | Explain what a compiler does. |
| decode_steps 默认值 | 4 |
| 文本可调范围 | 非空文本或非负 token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | Llama3.2-1B 当前物理 prefill 位置+执行长度≤64，chunk≤64；P+D 另受逻辑 cache 与 checkpoint 位置范围约束。 |
| 默认核数 | 8 |
| 4 核 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） |
| 6 核 | ● 已实测（注意力 4 · MLP 6 · 输出头 4） |
| 8 核 | ● 已实测 |
| CPU线程数 | 未设置（沿用环境） |
| 权重 | `llama-3.2-1b` |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/llama/runner.py) · [llama.py](../../../python/rpu_backend/adapters/llama.py) · [decoder.py](../../../python/rpu_backend/runtime/decoder.py) · [fused_model_base.cpp](../../../src/core/fused_model_base.cpp)。

<!-- model-configurations-end -->
