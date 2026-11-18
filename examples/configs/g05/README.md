# G0.5 配置

[配置总览](../README.md) · [全部配置](#configurations) · [核数支持矩阵](../../../docs/planner_core_topology.zh.md) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

先按相机数量选择模板。`2cam/` 和 `3cam/` 默认从配套 golden 取图像和状态，并合成目标长度的指令；已有 K1 RTC batch 则使用 [Legacy 入口](legacy/README.md)。

<a id="配置矩阵"></a>

2 路相机为 head、left wrist；3 路另加 right wrist。两组均提供 FP16、W8A16 和 W8A16 + Action NVFP4，Legacy K1 RTC batch 使用 FP16。

<a id="运行"></a>

## 准备与运行

填写 TOML 的 `model.checkpoint`、`model.release_root`、与 checkpoint 配套的 `input.source_golden`，以及 `[libraries]` 中的 backend、main REF 和 expansion REF 路径。默认 `input.source="synthetic"` 也会读取 `source_golden`，复用对应 step 的真实图像和状态，只合成指定长度的指令。`input.source_step` 选择 0 或 1，模板默认值为 1。

相机模板默认运行 32 个文本 token，动作去噪为 10 步。要一次测多个长度，把 `text_tokens` 换成 `text_lengths` 列表（如 2026-10-06 矩阵用的 `[32, 52, 64, 128, 192, 256]`），每个长度在独立进程中运行。

从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/g05/2cam/fp16.toml --check-config
python examples/run.py --config examples/configs/g05/2cam/fp16.toml
```

`--check-config` 只检查配置和输入参数范围，允许路径占位符。实际图像尺寸、动作输出与核心范围见[全部配置](#configurations)；运行前需替换占位路径。

<a id="input-files"></a>

## 输入文件准备

| 字段 | 内容要求 | 获取方式 |
|---|---|---|
| `model.checkpoint`、`model.release_root` | 客户 K1 RTC/FM release 目录及其中的 checkpoint（`model.pt`、`config.yaml`、`dataset_stats.json`） | 由客户 release 提供 |
| `input.source_golden` | CPU 参考文件：`schema_version=2`，`steps` 含两个真实观测（`obs`、`final_action`、`input_ids`、`attention_mask`）；文件记录的 checkpoint 摘要必须与当前 checkpoint 一致 | 用仓库脚本从 release 自带的 MCAP 录制在 CPU 上生成（加载约 11 GB 的完整模型，耗时较长）：<br>`python python/rpu_backend/tests/model/g05/test_g05_rtc_release_e2e.py --generate-cpu-golden --release-root <vla_rtc_release> --mcap <vla_rtc_release>/data/<录制>.mcap --golden <输出目录>/g05_k1_rtc_cpu_golden.pt`<br>不写 `--golden` 时默认输出名为 `cpu_golden.pt` |
| `input.batch`（`source = "batch"` 时） | 本入口自己保存的输入文件 | 每次 synthetic 运行都会把实际输入写到 `<run.output_dir>/input_batch.pt`，可直接作为 batch 重放；种子、token 和 checkpoint 身份必须与当次一致 |

`--check-config` 不读取上述文件；摘要不一致时运行时报 `source_golden does not match the verified release checkpoint`。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：G0.5。

### K1 RTC/FM

| 配置 | 支持状态 | 精度 | 量化范围 | cameras 默认值 | 4 核 | 6 核 | 8 核 |
|---|---|---|---|---|---|---|---|
| [2cam/fp16.toml](2cam/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 2 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [2cam/w8a16.toml](2cam/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision 与文本投影 W8（GDN in_proj_a/b 保持 FP16）；Action 投影 W8 | 2 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [2cam/w8a16_action_nvfp4.toml](2cam/w8a16_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_nvfp4 | Vision 与文本投影 W8（GDN in_proj_a/b 保持 FP16）；Action NVFP4 | 2 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [3cam/fp16.toml](3cam/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 3 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [3cam/w8a16.toml](3cam/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision 与文本投影 W8（GDN in_proj_a/b 保持 FP16）；Action 投影 W8 | 3 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [3cam/w8a16_action_nvfp4.toml](3cam/w8a16_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_nvfp4 | Vision 与文本投影 W8（GDN in_proj_a/b 保持 FP16）；Action NVFP4 | 3 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 相机 + 文本 + state |
| text_tokens 默认值 | 32 |
| num_steps 默认值 | 10 |
| 文本可调范围 | 有效 text_tokens=32–256整数；或不重复 text_lengths |
| 图像/音频可调范围 | 2/3相机；head、left wrist、可加right wrist |
| 生成可调范围 | num_steps=10 |
| 其他输入形状 | B=1；每图[1,1,3,H,W]；RTC previous action [1,32,27] |
| 联动限制 | batch 与 synthetic 二选一；batch 自带 instruction。synthetic 使用 source_golden 的step0/1，tokenization 必须匹配所选T；RTC interval/delay=5/5；图像H/W由release processor限制。 |
| 默认核数 | 8 |
| CPU线程数 | 8 |
| 权重 | `/path/to/vla_rtc_release/checkpoint`；准备方法见[权重准备](../WEIGHTS.md) |
| 入口 | [run.py](../../run.py) |

- ✕ 4/6 核不支持（`2cam/fp16.toml`、`2cam/w8a16.toml`、`3cam/fp16.toml`、`3cam/w8a16.toml`）：原生实现只有 8 核：动作专家只支持 8 核，W8 视觉和多相机批处理也只实现了 8 核。
- ✕ 4/6 核不支持（`2cam/w8a16_action_nvfp4.toml`、`3cam/w8a16_action_nvfp4.toml`）：原生实现只有 8 核：NVFP4 动作 GEMM 按 8 核切分，动作专家只支持 8 核。

输入校验来源：[optimized.py](../../../examples/runners/g05/optimized.py) · [inputs.py](../../../examples/runners/g05/inputs.py)。

### 历史 K1 RTC/FM

| 项目 | 值 |
|---|---|
| 配置 | [legacy/fp16.toml](legacy/fp16.toml) |
| 支持状态 | 支持（`SUPPORTED`） |
| 精度 | fp16 |
| 量化范围 | — |
| 输入类型 | 固定release tensor batch |
| 文本可调范围 | 由匹配release的batch给定 |
| 图像/音频可调范围 | 由匹配release的processor给定 |
| 生成可调范围 | 固定release去噪设置 |
| 其他输入形状 | batch含samples/pixel_values |
| 联动限制 | checkpoint、源码和release身份匹配；不继承新K1运行器的2/3相机和32–256文本范围。 |
| 默认核数 | 8 |
| 4 核 | ✕ 不支持 |
| 6 核 | ✕ 不支持 |
| 8 核 | ● 已实测 |
| CPU线程数 | 8（入口默认） |
| 权重 | `/path/to/vla_rtc_release/checkpoint`（profile 登记名 `g05-base`）；准备方法见[权重准备](../WEIGHTS.md) |
| 入口 | [run.py](../../run.py) |

- ✕ 4/6 核不支持：原生实现只有 8 核：动作专家只支持 8 核，W8 视觉和多相机批处理也只实现了 8 核。

输入校验来源：[runner.py](../../../examples/runners/g05/runner.py) · [runtime.py](../../../python/rpu_backend/adapters/g05/runtime.py)。

<!-- model-configurations-end -->
