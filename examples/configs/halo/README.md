# HALO Action Expert 配置

<a id="halo-配置"></a>

<a id="配置矩阵"></a>

[配置总览](../README.md) · [全部配置](#configurations) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

[FP16 配置](action_expert/fp16.toml) 只执行一次 Action Expert step，需要预先准备的张量输入和 KV cache。完整机器人策略不在此入口的支持范围内。

HALO 的后续测试目前暂缓，保留此配置用于已有路径的查阅和复现。单步 profile 的登记状态见[模型矩阵](../../../docs/model_matrix.zh.md)。

<a id="运行"></a>

## 输入与运行

准备 `halo_action_expert` 权重和匹配的 `input.fixture`。fixture 中的文本/动作行数、prefix 长度和 `packed_position_ids` 必须满足单步约束，KV cache 必须包含张量形式的 `key_cache` 和 `value_cache`。

具备配套资产、需要运行这一已登记单步路径时，从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/halo/action_expert/fp16.toml --check-config
python examples/run.py --config examples/configs/halo/action_expert/fp16.toml
```

`--check-config` 只检查配置语法和 profile；fixture 内容在运行时检查。结果不能用于声明完整 HALO 支持或机器人任务验收通过。

仓库内没有生成 `input.fixture` 的脚本，也没有随仓库发布匹配的 fixture。已有的原始捕获把 KV cache 存为 Python 对象，不能用 `weights_only=True` 读取，因此不能直接作为本入口的输入。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：HALO。

### 保留Action fixture

| 项目 | 值 |
|---|---|
| 配置 | [action_expert/fp16.toml](action_expert/fp16.toml) |
| 支持状态 | 受限支持（`SUPPORTED_LIMITED`）；未实测：仓库内没有本入口可读的单步 fixture（现有捕获的 KV cache 是 Python 对象，不能以 weights_only 读取），不是硬件限制 |
| 精度 | fp16 |
| 量化范围 | — |
| 输入类型 | 单步Action tensor fixture |
| 文本可调范围 | text/action rows与prefix匹配HaloConfig |
| 图像/音频可调范围 | 由fixture给定 |
| 生成可调范围 | 单Action step |
| 其他输入形状 | past_key_values tensor字典；packed_position_ids单常数 |
| 联动限制 | 后续测试暂缓；保留现有输入合同。 |
| 默认核数 | 8 |
| 4 核 | ✕ 不支持 |
| 6 核 | ✕ 不支持 |
| 8 核 | ○ 未实测 |
| CPU线程数 | 未设置（沿用环境） |
| 权重 | `halo_action_expert` |
| 入口 | [run.py](../../run.py) |

- ✕ 4/6 核不支持：延期模型：拓扑固定为注意力 2 核、MLP 8 核。

输入校验来源：[runner.py](../../../examples/runners/halo/runner.py) · [weights.py](../../../python/rpu_backend/adapters/halo/weights.py)。

<!-- model-configurations-end -->
