# HY Embodied UMI 配置

<a id="hy-embodied-配置"></a>

<a id="配置矩阵"></a>

[配置总览](../README.md) · [全部配置](#configurations) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

[UMI FP16](umi/fp16.toml) 使用一到三路相机图像、文本指令和机器人状态，返回归一化动作。同一入口另有 [W8A16](umi/w8a16.toml) 和 [W4A16](umi/w4a16_rtn_g32_vision_w8a16.toml)（VLM 与动作专家 G32 RTN，Vision W8A16）两个量化配置，读取同一份浮点权重、装载时量化，尚未上板实测，量化方式见[量化指南](../../../docs/quantization.zh.md)。要用合成输入扫相机数和文本长度时，查看 [HyVLA 模板](../hy_vla/README.md)。

<a id="运行"></a>

## 准备与运行

准备 `hy-embodied-0.5-vla-umi` 权重，并在 TOML 中填写 `input.images`、`input.instruction` 和 `input.state`。图像填一到三张；模板的 `prefix_len = "auto"` 按图数取 240 / 192 / 128。

从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/hy_embodied/umi/fp16.toml --check-config
python examples/run.py --config examples/configs/hy_embodied/umi/fp16.toml
```

配置检查只校验语法和 profile，图像数量等内容由运行入口检查。输出文件见[结果说明](../../RESULTS.md)。

此入口不需要预先准备 tensor 文件：`input.images` 按 face、left_wrist、right_wrist 的顺序填一到三张普通图片路径，`instruction` 填文本，`state` 直接在 TOML 中写最多 32 个已归一化的状态值。图片数量不在 1–3 之间时运行时报错。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：Hy-Embodied。

### Hy-Embodied 图文

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 |
|---|---|---|---|---|---|---|
| [umi/fp16.toml](umi/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测 | ● 已实测 | ● 已实测 |
| [umi/w4a16_rtn_g32_vision_w8a16.toml](umi/w4a16_rtn_g32_vision_w8a16.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | w4a16 | ViT W8A16；VLM / 动作专家 W4A16 G32 RTN | ● 已实测 | ● 已实测 | ● 已实测 |
| [umi/w8a16.toml](umi/w8a16.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | w8a16 | ViT / VLM / 动作专家 W8A16 | ● 已实测 | ● 已实测 | ● 已实测 |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 原始图片 + instruction + state |
| 文本可调范围 | 指令文本，分词后≤64 token（checkpoint的tokenizer_max_length=64，超出部分截断）；有效prefix 2+58C+T由processor计算 |
| 图像/音频可调范围 | 1/2/3图，按face、left_wrist、right_wrist顺序取前C张，任意尺寸等比缩放补边到224（1/2图 FP16 6/8 核已在板上通过） |
| 生成可调范围 | 按模型动作入口 |
| 其他输入形状 | prefix_len为auto（省略同auto）或16的倍数，范围[128, ceil16(2+58C+64)]：3图240、2图192、1图128；auto取上限 |
| 联动限制 | 实际prefix向上16对齐后不得超过prefix_len；动作专家的常驻与融合路线只放行prefix 128–240，所以单图固定128。 |
| 默认核数 | 8 |
| CPU线程数 | 12（入口默认） |
| 权重 | `hy-embodied-0.5-vla-umi` |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/hy_vla/runner.py) · [hy_embodied.py](../../../python/rpu_backend/api/hy_embodied.py)。

<!-- model-configurations-end -->
