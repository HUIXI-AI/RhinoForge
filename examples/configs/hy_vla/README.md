# HyVLA 配置

[配置总览](../README.md) · [全部配置](#configurations) · [核数支持矩阵](../../../docs/planner_core_topology.zh.md) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

按相机数量选择目录，再选择精度。各模板默认使用合成输入：32 个文本 token、10 步动作去噪，`prefix_len` 由输入自动计算。

<a id="配置矩阵"></a>

2/3 路相机均提供 FP16、W8A16 和 W8A16 + Action NVFP4 配方。

<a id="运行"></a>

## 准备与运行

在 `model.checkpoint` 中填写 `hy-embodied-0.5-vla-umi` 权重别名或目录。核心范围和输入限制见[全部配置](#configurations)与[核数支持矩阵](../../../docs/planner_core_topology.zh.md)。需要直接输入三张图像、指令和状态时，另见 [UMI FP16 入口](../hy_embodied/README.md)。

从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/hy_vla/2cam/fp16.toml --check-config
python examples/run.py --config examples/configs/hy_vla/2cam/fp16.toml
```

`--check-config` 检查配置和输入参数范围；合成输入的运行结果不能代表真实机器人任务质量。

模板末尾 `[libraries]` 中的 `/path/to/...` 是占位：填写当前 backend 构建和仓库自带的两个算子库 REF，或删除整个表，改用环境变量或仓库默认文件，规则见[通用参数](../PARAMETERS.md#模型专用参数)。

<a id="input-files"></a>

## 输入文件准备

模板默认 `input.source = "synthetic"`，不需要任何输入文件：入口按 `run.seed` 生成随机图像、状态和噪声，并用 checkpoint 的 tokenizer 构造指令。每次运行都会在 `run.output_dir` 下保存：

- `input_batch.pt`：本次实际输入。设 `input.source = "batch"`、`input.batch` 指向该文件，即可原样重放。
- `reference.pt`：本次输出动作。下次用同一输入运行时填入 `input.reference`，可对比两次输出；它不是 CPU 参考。

需要用真实观测按 16/32/48/64 个文本 token 生成 batch 时（仅 3 路相机），可用 `scripts/prepare_hyvla_text_batches.py`：

```bash
python scripts/prepare_hyvla_text_batches.py --source-observation <原始观测.pt> \
  --checkpoint <hy-embodied-0.5-vla-umi 本地目录> --output <输出目录> [--instruction-file <指令.txt>]
```

输出 `hyvla_3cam_t{16,32,48,64}_batch.pt` 和 `inputs.json`。batch 在生成 `report.json` 之前读取，格式不符时只在终端报错。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：HyVLA。

### HyVLA 动作

| 配置 | 支持状态 | 精度 | 量化范围 | cameras 默认值 | 4 核 | 6 核 | 8 核 |
|---|---|---|---|---|---|---|---|
| [2cam/fp16.toml](2cam/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 2 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [2cam/w8a16.toml](2cam/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | ViT、VLM 与动作专家投影 W8 | 2 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [2cam/w8a16_action_nvfp4.toml](2cam/w8a16_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_nvfp4 | ViT 与 VLM 投影 W8；动作专家 NVFP4 | 2 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [3cam/fp16.toml](3cam/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 3 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [3cam/w8a16.toml](3cam/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | ViT、VLM 与动作专家投影 W8 | 3 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [3cam/w8a16_action_nvfp4.toml](3cam/w8a16_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_nvfp4 | ViT 与 VLM 投影 W8；动作专家 NVFP4 | 3 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | batch / synthetic 图文+state |
| text_tokens 默认值 | 32 |
| num_steps 默认值 | 10 |
| 文本可调范围 | 有效 text_tokens=1–64整数 |
| 图像/音频可调范围 | 2/3相机；处理后每图[1,3,224,224] |
| 生成可调范围 | num_steps=10 |
| 其他输入形状 | state[1,32]或state_emb[1,1,1024]；noise[1,50,32]；lang_tokens[64] |
| 联动限制 | 总核数8；prefix_len为auto或16对齐，ceil16(2+58C+T)≤prefix_len≤ceil16(2+58C+64)。batch拥有instruction；synthetic真实token数必须匹配。 |
| 默认核数 | 8 |
| CPU线程数 | 12（入口默认） |
| 权重 | `hy-embodied-0.5-vla-umi` |
| 入口 | [run.py](../../run.py) |

- ✕ 4/6 核不支持：原生实现只有 8 核：HyVLA 专用融合 kernel 要求 8 核，视觉塔 16 个注意力头也不能分到 6 核。

输入校验来源：[optimized.py](../../../examples/runners/hy_vla/optimized.py) · [hy_embodied.py](../../../python/rpu_backend/api/hy_embodied.py)。

<!-- model-configurations-end -->
