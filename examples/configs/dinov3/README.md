# DINOv3 配置

[配置总览](../README.md) · [全部配置](#configurations) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

<a id="配置矩阵"></a>

[ViT-B FP16](vit_b/fp16.toml) 接收单张图像，返回视觉特征 `last_hidden_state` 和池化特征 `pooler_output`。这一路径用于特征提取。

[W8A16](vit_b/w8a16_vision.toml) 和 [W4A16 G32](vit_b/w4a16_vision_rtn_g32.toml) 读取同一个浮点权重目录，在加载时量化编码器每层的六个投影，其余保持 FP16；两者都还没上板，细节见[量化指南](../../../docs/quantization.zh.md)。

<a id="运行"></a>

## 准备与运行

按权重准备页配置 `dinov3-vit-b` 目录，将 TOML 的 `input.images` 替换为实际图片路径。图像预处理器从同一权重目录加载；输入处理和输出形状受该模型配置约束。

从仓库根目录先检查配置，再运行：

```bash
python examples/run.py --config examples/configs/dinov3/vit_b/fp16.toml --check-config
python examples/run.py --config examples/configs/dinov3/vit_b/fp16.toml
```

配置检查通过会输出 `configuration OK`，只说明语法和 profile 匹配。运行输出及文件位置见[结果说明](../../RESULTS.md)。

核数默认 8。需要 4 或 6 核时，在 TOML 中加入 `[rpu_execution.model]` 并写 `num_cores = 4` 或 `6`：注意力和 MLP 都用所选核数（每核 3 或 2 个头，不补头），需要包含 DINOv3 核数接口的重编译 native 扩展。各核数的实测见下表。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：DINOv3。

### 视觉特征

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 |
|---|---|---|---|---|---|---|
| [vit_b/fp16.toml](vit_b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测 | ● 已实测 | ● 已实测 |
| [vit_b/w4a16_vision_rtn_g32.toml](vit_b/w4a16_vision_rtn_g32.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | w4a16 | Vision W4A16 G32 RTN | ● 已实测 | ● 已实测 | ● 已实测 |
| [vit_b/w8a16_vision.toml](vit_b/w8a16_vision.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | w8a16 | Vision W8A16 | ● 已实测 | ● 已实测 | ● 已实测 |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 单张RGB图片 |
| 文本可调范围 | 不使用 |
| 图像/音频可调范围 | 一次1图；processor后H/W整除16，默认224×224 |
| 生成可调范围 | 不生成 |
| 其他输入形状 | tensor[1,3,H,W] |
| 联动限制 | 处理后patch grid不得超过安装max_hw和KV容量；默认ViT-B max_hw=14。原始文件尺寸经processor调整，不等同模型tensor尺寸。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 权重 | `dinov3-vit-b` |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/dinov3/runner.py) · [dinov3.py](../../../python/rpu_backend/adapters/dinov3.py)。

<!-- model-configurations-end -->
