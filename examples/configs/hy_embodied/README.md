# HY Embodied UMI 配置

<a id="hy-embodied-配置"></a>

<a id="配置矩阵"></a>

[配置总览](../README.md) · [全部配置](#configurations) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

[UMI FP16](umi/fp16.toml) 使用三路相机图像、文本指令和机器人状态，返回归一化动作。需要两路相机或量化配置时，查看 [HyVLA 模板](../hy_vla/README.md)，按各自输入要求准备资产。

<a id="运行"></a>

## 准备与运行

准备 `hy-embodied-0.5-vla-umi` 权重，并在 TOML 中填写 `input.images`、`input.instruction` 和 `input.state`。此入口要求恰好三张图像；模板将 `prefix_len` 设为 240。

从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/hy_embodied/umi/fp16.toml --check-config
python examples/run.py --config examples/configs/hy_embodied/umi/fp16.toml
```

配置检查只校验语法和 profile，图像数量等内容由运行入口检查。输出文件见[结果说明](../../RESULTS.md)。

此入口不需要预先准备 tensor 文件：`input.images` 填三张普通图片路径，`instruction` 填文本，`state` 直接在 TOML 中写最多 32 个已归一化的状态值。图片数量不是 3 时运行时报错。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：Hy-Embodied。

### Hy-Embodied 图文

| 项目 | 值 |
|---|---|
| 配置 | [umi/fp16.toml](umi/fp16.toml) |
| 支持状态 | 支持（`SUPPORTED`） |
| 精度 | fp16 |
| 量化范围 | — |
| 输入类型 | 原始图片 + instruction + state |
| 文本可调范围 | 有效prefix由processor计算 |
| 图像/音频可调范围 | 当前runner固定3图 |
| 生成可调范围 | 按模型动作入口 |
| 其他输入形状 | prefix_len为16的倍数，三相机上限240 |
| 联动限制 | 实际prefix向上16对齐后不得超过安装prefix_len；API另有2图功能不属于此runner。 |
| 默认核数 | 8 |
| 4 核 | ✕ 不支持 |
| 6 核 | ✕ 不支持 |
| 8 核 | ● 已实测 |
| CPU线程数 | 12（入口默认） |
| 权重 | `hy-embodied-0.5-vla-umi` |
| 入口 | [run.py](../../run.py) |

- ✕ 4/6 核不支持：原生实现只有 8 核：HyVLA 专用融合 kernel 要求 8 核，视觉塔 16 个注意力头也不能分到 6 核。

输入校验来源：[runner.py](../../../examples/runners/hy_vla/runner.py) · [hy_embodied.py](../../../python/rpu_backend/api/hy_embodied.py)。

<!-- model-configurations-end -->
