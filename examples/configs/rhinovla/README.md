# RhinoVLA 配置

[配置总览](../README.md) · [全部配置](#configurations) · [核数支持矩阵](../../../docs/planner_core_topology.zh.md) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

先区分官方 v1、v3 调用者工厂和 v3 源码入口。三类配置都用 `examples/run.py` 运行，但需要的输入与权重资产不同。

<a id="配置矩阵"></a>

| 路径 | 配方 | 所需资产与入口 |
|---|---|---|
| v1 官方 non-RTC | FP16 | 固定版本源码、checkpoint 和 CPU fixture；[v1 指南](v1/README.md) |
| v3 调用者工厂 | FP16、W8A16 Action、W8A16 pipeline | 调用者提供 runtime factory、request 与 factory config；[v3 指南](v3/README.md) |
| v3 源码入口 | FP16、W8A16 Action、W8A16 pipeline 及融合变体 | v3 steps_150000 non-RTC 源码、checkpoint 和 CPU fixture；也可交给 `scripts/validate_rhinovla_v3.py` 校验；[controlled 指南](v3/controlled/README.md) |

<a id="运行"></a>

## 运行 v1 配置

按 [v1 指南](v1/README.md)准备完整资产目录，从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/rhinovla/v1/fp16.toml --check-config
python examples/run.py --config examples/configs/rhinovla/v1/fp16.toml
```

v3 工厂模板的执行范围由调用者实现决定。支持状态与 4/6/8 核见下方配置表；“支持”只表示运行时检查通过，数值误差见性能报告。`--check-config` 检查配置，输入资产在运行时校验。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：RhinoVLA。

### 自定义factory

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 | CPU线程数 |
|---|---|---|---|---|---|---|---|
| [v3/fp16.toml](v3/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测 | ● 已实测 | ● 已实测 | 未设置（沿用环境） |
| [v3/w8a16_action.toml](v3/w8a16_action.toml) | 支持（`SUPPORTED`） | w8a16 | Action expert W8A16 | ● 已实测 | ● 已实测 | ● 已实测 | 未设置（沿用环境） |
| [v3/w8a16_action_adarms.toml](v3/w8a16_action_adarms.toml) | 支持（`SUPPORTED`） | w8a16 | Action expert / AdaRMS 条件投影 W8A16 | ● 已实测 | ● 已实测 | ● 已实测 | 未设置（沿用环境） |
| [v3/w8a16_pipeline.toml](v3/w8a16_pipeline.toml) | 支持（`SUPPORTED`） | w8a16 | Vision（patch / merger） / Prefix / Action expert / AdaRMS 条件投影 / 动作 IO W8A16 | ● 已实测 | ● 已实测 | ● 已实测 | 12 |
| [v3/w8a16_pipeline_high_fused_vector_qk.toml](v3/w8a16_pipeline_high_fused_vector_qk.toml) | 支持（`SUPPORTED`） | w8a16 | Vision（patch / merger） / Prefix / Action expert / AdaRMS 条件投影 / 动作 IO W8A16 | ● 已实测 | ● 已实测 | ● 已实测 | 12 |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 调用者 request tensor mapping |
| num_steps 默认值 | 10 |
| 文本可调范围 | 由factory的request合同决定 |
| 图像/音频可调范围 | 由factory的request合同决定 |
| 生成可调范围 | num_steps=10 |
| 其他输入形状 | checkpoint/runtime_factory/request由调用者提供 |
| 联动限制 | trust_model_code=true；factory.predict(**request)校验具体shape，量化选择核对实际owner。 |
| 默认核数 | 8 |
| 权重 | `/path/to/rhinovla-v3/checkpoint`（profile 登记名 `rhinovla-controlled`）；准备方法见[权重准备](../WEIGHTS.md) |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/rhinovla/runner.py)。

### 官方固定fixture

| 项目 | 值 |
|---|---|
| 配置 | [v1/fp16.toml](v1/fp16.toml) |
| 支持状态 | 支持（`SUPPORTED`） |
| 精度 | fp16 |
| 量化范围 | — |
| 输入类型 | 冻结non-RTC fixture |
| num_steps 默认值 | 10 |
| 文本可调范围 | 固定prefix171 |
| 图像/音频可调范围 | 两路vision grid[1,16,16] |
| 生成可调范围 | num_steps=10；rtc=false |
| 其他输入形状 | B=1；action[30,72] |
| 联动限制 | 源码提交、checkpoint SHA、fixture类型固定；此TOML不提供任意request替换。 |
| 默认核数 | 8 |
| 4 核 | ✕ 不支持 |
| 6 核 | ✕ 不支持 |
| 8 核 | ● 已实测 |
| CPU线程数 | 未设置（沿用环境） |
| 权重 | `rhinovla-controlled` |
| 入口 | [run.py](../../run.py) |

- ✕ 4/6 核不支持：原生实现只有 8 核：v1 动作专家的几何没有少核 owner，4/6 核只覆盖 v3。

输入校验来源：[runner.py](../../../examples/runners/rhinovla/runner.py)。

### v3 源码入口

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 |
|---|---|---|---|---|---|---|
| [v3/controlled/fp16.toml](v3/controlled/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测 | ● 已实测 | ● 已实测 |
| [v3/controlled/w8a16_action.toml](v3/controlled/w8a16_action.toml) | 支持（`SUPPORTED`） | w8a16 | Action expert W8A16 | ● 已实测 | ● 已实测 | ● 已实测 |
| [v3/controlled/w8a16_pipeline.toml](v3/controlled/w8a16_pipeline.toml) | 支持（`SUPPORTED`） | w8a16 | Vision（patch / merger） / Prefix / Action expert / AdaRMS 条件投影 / 动作 IO W8A16 | ● 已实测 | ● 已实测 | ● 已实测 |
| [v3/controlled/w8a16_pipeline_high.toml](v3/controlled/w8a16_pipeline_high.toml) | 支持（`SUPPORTED`） | w8a16 | Vision（patch / merger） / Prefix / Action expert / AdaRMS 条件投影 / 动作 IO W8A16 | ● 已实测 | ● 已实测 | ● 已实测 |
| [v3/controlled/w8a16_pipeline_high_fused.toml](v3/controlled/w8a16_pipeline_high_fused.toml) | 支持（`SUPPORTED`） | w8a16 | Vision（patch / merger） / Prefix / Action expert / AdaRMS 条件投影 / 动作 IO W8A16 | ● 已实测 | ● 已实测 | ● 已实测 |
| [v3/controlled/w8a16_pipeline_high_fused_vector_k.toml](v3/controlled/w8a16_pipeline_high_fused_vector_k.toml) | 支持（`SUPPORTED`） | w8a16 | Vision（patch / merger） / Prefix / Action expert / AdaRMS 条件投影 / 动作 IO W8A16 | ● 已实测 | ● 已实测 | ● 已实测 |
| [v3/controlled/w8a16_pipeline_high_fused_vector_qk.toml](v3/controlled/w8a16_pipeline_high_fused_vector_qk.toml) | 支持（`SUPPORTED`） | w8a16 | Vision（patch / merger） / Prefix / Action expert / AdaRMS 条件投影 / 动作 IO W8A16 | ● 已实测 | ● 已实测 | ● 已实测 |
| [v3/controlled/w8a16_pipeline_high_fused_vector_qk_partial_rope.toml](v3/controlled/w8a16_pipeline_high_fused_vector_qk_partial_rope.toml) | 支持（`SUPPORTED`） | w8a16 | Vision（patch / merger） / Prefix / Action expert / AdaRMS 条件投影 / 动作 IO W8A16 | ● 已实测 | ● 已实测 | ● 已实测 |
| [v3/controlled/w8a16_pipeline_input_cache.toml](v3/controlled/w8a16_pipeline_input_cache.toml) | 支持（`SUPPORTED`） | w8a16 | Vision（patch / merger） / Prefix / Action expert / AdaRMS 条件投影 / 动作 IO W8A16 | ● 已实测 | ● 已实测 | ● 已实测 |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | scripts/validate_rhinovla_v3.py prepare 生成的 CPU FP32 fixture（第 1 组输入），或原始图片+指令+state |
| num_steps 默认值 | 10 |
| 文本可调范围 | fixture：固定 input_ids；原始输入：非空instruction，prefix行数（模板+每图64+指令token）由源码processor计算 |
| 图像/音频可调范围 | fixture：固定图像网格；原始输入：1/2/3图，任意尺寸，转RGB后缩放到256×256（原始输入 FP16 1/2/3 图 8 核、1 图 6 核已在板上通过） |
| 生成可调范围 | num_steps=10；rtc=false |
| 其他输入形状 | source_dir/checkpoint由用户提供；B=1；action[30,96]；原始输入另需state 1–96个有限值（补零到96），可选view_roles（省略取前C个top_head、hand_left、hand_right）与action_dims 1–96（省略同len(state)）；x0由[run].seed生成 |
| 联动限制 | 源码提交、checkpoint SHA256 和 fixture 元数据须匹配 v3 steps_150000 non-RTC；fixture与images二选一；view_roles取源码训练配置的view_role_vocab；[model.opt_in] 须与登记 profile 一致；trust_model_code=true。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 权重 | `/path/to/rhinovla-v3/source/models/rhinovla_v3/steps_150000/model.pt`；准备方法见[权重准备](../WEIGHTS.md) |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/rhinovla/runner.py) · [validate_rhinovla_v3.py](../../../scripts/validate_rhinovla_v3.py)。

<!-- model-configurations-end -->
