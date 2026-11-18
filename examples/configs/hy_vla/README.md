# HyVLA 配置

[配置总览](../README.md) · [全部配置](#configurations) · [核数支持矩阵](../../../docs/planner_core_topology.zh.md) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

按精度选择模板。各模板默认使用合成输入：3 路相机、32 个文本 token、10 步动作去噪，2 路相机在命令行加 `--input 2cam_t32`（预设见 [vla.toml](../../inputs/vla.toml)）；`prefix_len` 由输入自动计算（取值见 [prefix_len 与 RMSNorm 的实际取值](#auto-values)）。

<a id="配置矩阵"></a>

FP16、W8A16、W8A16 + Action NVFP4、W4A16（VLM 与动作专家 G32 RTN，Vision W8A16）和全 W4A16（`w4a16_rtn_g32.toml`，Vision 也为 G32 RTN）五个模板都可用 2 路或 3 路相机。全 W4A16 模板的 Vision 在任何核数预算下都用 4 核（8 核时文本骨干和动作专家仍用 8 核），Vision KV 放 DDR。两个 W4A16 模板 8 核已上板通过运行时检查（2026-11-06），4/6 核尚未上板。8 核实测全 W4A16 的 Vision 为 52.0 ms，Vision W8A16 的模板为 20.0 ms，整次请求 144.4 ms 对 112.3 ms（3 相机、T32、10 步）。量化方式见[量化指南](../../../docs/quantization.zh.md)。

<a id="运行"></a>

## 准备与运行

在 `model.checkpoint` 中填写 `hy-embodied-0.5-vla-umi` 权重别名或目录。核心范围和输入限制见[全部配置](#configurations)与[核数支持矩阵](../../../docs/planner_core_topology.zh.md)。需要直接输入三张图像、指令和状态时，另见 [UMI FP16 入口](../hy_embodied/README.md)。

从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/hy_vla/fp16.toml --check-config
python examples/run.py --config examples/configs/hy_vla/fp16.toml
# 2 路相机
python examples/run.py --config examples/configs/hy_vla/fp16.toml --input 2cam_t32
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

<a id="auto-values"></a>

## prefix_len 与 RMSNorm 的实际取值

prefix_len 不随核数变化。RMSNorm 的 V16/V32 只有 8 核实现：`rpu_execution.model.num_cores` 为 4/6 时 `rmsnorm_vwarp = "auto"` 走 BASE RMSNorm，写 `"16"` 报错；下面的 RMSNorm 取值只适用于 8 核。

- **`prefix_len = "auto"`**：取能覆盖真实行数的最小 16 行 bucket，即 ceil16(2 + 58×相机数 + text_tokens)。显式值须 16 对齐，范围从 auto 值到 ceil16(2 + 58×相机数 + 64)，即 2 路相机不超过 192、3 路不超过 240；多出的行是补齐。
- **`rmsnorm_vwarp = "auto"`**：只作用于 VLM，Vision 与 Action 保持 BASE。整行与 K 头的 RMSNorm 行数等于 prefix_len，能被 32 整除用 V32，否则用 V16。Q 头的行数是 2×prefix_len，总是 V32；KV-only 尾层的 Q 头用 BASE。
- **`rmsnorm_vwarp = "16"`**：不用 V32，16 对齐的行用 V16。

| 相机 | text_tokens | auto 的 prefix_len | 整行与 K 头 RMSNorm（auto） |
|---|---|---:|---|
| 2 | 1–10 | 128 | V32 |
| 2 | 11–26 | 144 | V16 |
| 2 | 27–42（含模板 T32） | 160 | V32 |
| 2 | 43–58 | 176 | V16 |
| 2 | 59–64 | 192 | V32 |
| 3 | 1–16 | 192 | V32 |
| 3 | 17–32（含模板 T32） | 208 | V16 |
| 3 | 33–48 | 224 | V32 |
| 3 | 49–64 | 240 | V16 |

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：HyVLA。

### HyVLA 动作

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 |
|---|---|---|---|---|---|---|
| [fp16.toml](fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测 | ● 已实测 | ● 已实测 |
| [w4a16_rtn_g32.toml](w4a16_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | ViT / VLM / 动作专家 W4A16 G32 RTN | ● 已实测 | ● 已实测 | ● 已实测 |
| [w4a16_rtn_g32_vision_w8a16.toml](w4a16_rtn_g32_vision_w8a16.toml) | 支持（`SUPPORTED`） | w4a16 | ViT W8A16；VLM / 动作专家 W4A16 G32 RTN | ● 已实测 | ● 已实测 | ● 已实测 |
| [w8a16.toml](w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | ViT / VLM / 动作专家 W8A16 | ● 已实测 | ● 已实测 | ● 已实测 |
| [w8a16_action_nvfp4.toml](w8a16_action_nvfp4.toml) | 支持（`SUPPORTED`） | w8a16_action_nvfp4 | ViT / VLM W8A16；动作专家 NVFP4 | ● 已实测 | ● 已实测 | ● 已实测 |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | batch / synthetic 图文+state |
| text_tokens 默认值 | 32 |
| cameras 默认值 | 3 |
| num_steps 默认值 | 10 |
| 文本可调范围 | 有效 text_tokens=1–64整数（上限来自 checkpoint 的 tokenizer_max_length=64：预处理把指令截断并补齐到 64 个语言槽位） |
| 图像/音频可调范围 | 1/2/3相机（1相机 FP16 6/8 核已在板上通过）；处理后每图[1,3,224,224] |
| 生成可调范围 | num_steps=10 |
| 其他输入形状 | state[1,32]或state_emb[1,1,1024]；noise[1,50,32]；lang_tokens[64] |
| 联动限制 | model.num_cores为4/6/8（省略为8）；4/6核时Vision和动作专家用4核、文本骨干Attention 4/MLP全部核，rmsnorm_vwarp只能auto（走BASE），resident_fp8_scale_layers须为0；全W4配方（w4a16_rtn_g32）的Vision在8核时也用4核。prefix_len为auto或16对齐，max(128,ceil16(2+58C+T))≤prefix_len≤ceil16(2+58C+64)，auto取下限；动作专家的常驻与融合路线只放行prefix 128–240，所以单相机固定128。batch拥有instruction；synthetic真实token数必须匹配。 |
| 默认核数 | 8 |
| CPU线程数 | 12（入口默认） |
| 权重 | `hy-embodied-0.5-vla-umi` |
| 入口 | [run.py](../../run.py) |

输入校验来源：[optimized.py](../../../examples/runners/hy_vla/optimized.py) · [hy_embodied.py](../../../python/rpu_backend/api/hy_embodied.py)。

<!-- model-configurations-end -->
