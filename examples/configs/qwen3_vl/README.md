# Qwen3-VL 配置

[配置总览](../README.md) · [全部配置](#configurations) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md) · 目录字段

按输入和所需输出选择入口：`text/` 只运行文本路径，`vl/` 先处理图像再生成文本，`vision/` 只返回视觉输出。随后选择尺寸和精度；32B 没有 FP16 模板。

<a id="快速选择"></a>

<a id="配置矩阵"></a>

2B、4B、8B 的 Text/VL 提供 FP16、W8 与 W4 配方，32B 仅有 W8/W4；Vision-only 提供 2B、4B、8B FP16。

<a id="选择规则"></a>

## 匹配权重和组件配置

量化模板分别指定文本投影、输出头和视觉组件的精度，不能只按文件名中的 W8 或 W4 选择权重。确认[全部配置](#configurations)中的量化范围，再按[权重准备](../WEIGHTS.md)配置目录。Text、VL、Vision 是否共用权重，由各自 TOML 的权重别名决定。

模板默认 8 核，全部配置可选择 4/6/8 核。6 核预算下，Vision、文本 Attention/head 使用 4 核，文本 MLP 按配方使用 4 或 6 核；每个配置的实际分配见[配置表](#configurations)。

<a id="权重与运行"></a>

## 准备与运行

有 FP16 模板的尺寸，优先从 FP16 开始，再按完整索引选择 W8、W4/G32 或 AWQ。32B 使用现有量化模板。

1. 选择输入路径、模型尺寸和配方，准备对应权重。
2. 填写 TOML 的文本和图像路径，确认图像规格、文本长度、解码步数及核心数。
3. 从仓库根目录先检查配置，再运行。下面使用 VL 4B FP16：

```bash
python examples/run.py --config examples/configs/qwen3_vl/vl/4b/fp16.toml --check-config
python examples/run.py --config examples/configs/qwen3_vl/vl/4b/fp16.toml
```

`--check-config` 只检查语法和 profile，不加载图像或权重。更改图片、输入长度、核心数或累加选项后，重新检查并启动新进程；输出见[结果说明](../../RESULTS.md)。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：Qwen3-VL。

P 为处理后的输入 token 数，D 为 decode 调用数；生成首 token 来自 prefill。

### 图文 · 2B

| 配置 | 支持状态 | 精度 | 量化范围 | prompt 默认示例 | prefill_tokens 默认值 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|---|---|
| [vl/2b/fp16.toml](vl/2b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | Describe the image. | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-2b` |
| [vl/2b/w4a16_text_awq_g32.toml](vl/2b/w4a16_text_awq_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text | Describe this image in one sentence. | 430 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-2b-text-awq-w4a16` |
| [vl/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](vl/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | 支持（`SUPPORTED`） | w4a16 | text + vision | Describe this image in one sentence. | 430 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-2b-runtime-w4a16` |
| [vl/2b/w8a16_text.toml](vl/2b/w8a16_text.toml) | 支持（`SUPPORTED`） | w8a16 | text | Describe this image in one sentence. | 430 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-2b-text-w8a16` |
| [vl/2b/w8a16_text_head_vision.toml](vl/2b/w8a16_text_head_vision.toml) | 支持（`SUPPORTED`） | w8a16 | text + output head + vision | Describe this image in one sentence. | 430 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-2b-runtime-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 图像 + 文本 |
| 文本可调范围 | 非空提示词；P 包含图像和聊天格式 |
| 图像/音频可调范围 | 非空 RGB 图片列表；不接收视频；每图 processor 网格≤4096 patch（patch 16 px，如 1024×1024、2048×512 px）；Vision RoPE 表与 KV 容量按所配图像冷态安装，网格边长≤48且≤2048 patch 时保持默认 48 / 2048，否则随最大图像增大；板上实测到 1024×1024 px（4096 patch，8 核）、2048×512 px（32×128 网格，4096 patch，8 核，RoPE 表与 KV 容量按该网格扩大）与 896×896 px（3136 patch，4/6/8 核）；其余网格可以安装，但尚未上板运行 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1；prepared input 只用于 multimodal |
| 联动限制 | prepared input 要求 SHA、单图 grid、全1 mask、有限像素及匹配 M-RoPE。2/4/8B 普通文本 owner 的物理 prefill 上界4176，chunk 上限320/256/128；32B 的运行时量化 W8/W4 与旧 W8 文本配方同为物理 prefill 上界4176、chunk 上限64；其余量化及固定交付 profile 另按其准入检查。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/qwen3_vl/runner.py) · [inputs.py](../../../examples/runners/qwen3_vl/inputs.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_vl/text.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_vl/vision.py) · [__init__.py](../../../python/rpu_backend/adapters/qwen3_vl/__init__.py)。

### 图文 · 4B

| 配置 | 支持状态 | 精度 | 量化范围 | prompt 默认示例 | prefill_tokens 默认值 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|---|---|
| [vl/4b/fp16.toml](vl/4b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | Describe the image. | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-4b` |
| [vl/4b/w4a16_text_awq_g32.toml](vl/4b/w4a16_text_awq_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text | Describe this image in one sentence. | 430 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-4b-text-awq-w4a16` |
| [vl/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](vl/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | 支持（`SUPPORTED`） | w4a16 | text + vision | Describe this image in one sentence. | 430 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-4b-runtime-w4a16` |
| [vl/4b/w8a16_text_head_vision.toml](vl/4b/w8a16_text_head_vision.toml) | 支持（`SUPPORTED`） | w8a16 | text + output head + vision | Describe this image in one sentence. | 430 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-4b-runtime-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 图像 + 文本 |
| 文本可调范围 | 非空提示词；P 包含图像和聊天格式 |
| 图像/音频可调范围 | 非空 RGB 图片列表；不接收视频；每图 processor 网格≤4096 patch（patch 16 px，如 1024×1024、2048×512 px）；Vision RoPE 表与 KV 容量按所配图像冷态安装，网格边长≤48且≤2048 patch 时保持默认 48 / 2048，否则随最大图像增大；板上实测到 1024×1024 px（4096 patch，8 核）、2048×512 px（32×128 网格，4096 patch，8 核，RoPE 表与 KV 容量按该网格扩大）与 896×896 px（3136 patch，4/6/8 核）；其余网格可以安装，但尚未上板运行 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1；prepared input 只用于 multimodal |
| 联动限制 | prepared input 要求 SHA、单图 grid、全1 mask、有限像素及匹配 M-RoPE。2/4/8B 普通文本 owner 的物理 prefill 上界4176，chunk 上限320/256/128；32B 的运行时量化 W8/W4 与旧 W8 文本配方同为物理 prefill 上界4176、chunk 上限64；其余量化及固定交付 profile 另按其准入检查。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/qwen3_vl/runner.py) · [inputs.py](../../../examples/runners/qwen3_vl/inputs.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_vl/text.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_vl/vision.py) · [__init__.py](../../../python/rpu_backend/adapters/qwen3_vl/__init__.py)。

### 图文 · 8B

| 配置 | 支持状态 | 精度 | 量化范围 | prompt 默认示例 | prefill_tokens 默认值 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|---|---|
| [vl/8b/fp16.toml](vl/8b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | Describe the image. | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-8b` |
| [vl/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](vl/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | 支持（`SUPPORTED`） | w4a16 | text + vision | Describe this image in one sentence. | 430 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-8b-runtime-w4a16` |
| [vl/8b/w8a16_text_head_vision.toml](vl/8b/w8a16_text_head_vision.toml) | 支持（`SUPPORTED`） | w8a16 | text + output head + vision | Describe this image in one sentence. | 430 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-8b-runtime-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 图像 + 文本 |
| 文本可调范围 | 非空提示词；P 包含图像和聊天格式 |
| 图像/音频可调范围 | 非空 RGB 图片列表；不接收视频；每图 processor 网格≤4096 patch（patch 16 px，如 1024×1024、2048×512 px）；Vision RoPE 表与 KV 容量按所配图像冷态安装，网格边长≤48且≤2048 patch 时保持默认 48 / 2048，否则随最大图像增大；板上实测到 1024×1024 px（4096 patch，8 核）、2048×512 px（32×128 网格，4096 patch，8 核，RoPE 表与 KV 容量按该网格扩大）与 896×896 px（3136 patch，4/6/8 核）；其余网格可以安装，但尚未上板运行 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1；prepared input 只用于 multimodal |
| 联动限制 | prepared input 要求 SHA、单图 grid、全1 mask、有限像素及匹配 M-RoPE。2/4/8B 普通文本 owner 的物理 prefill 上界4176，chunk 上限320/256/128；32B 的运行时量化 W8/W4 与旧 W8 文本配方同为物理 prefill 上界4176、chunk 上限64；其余量化及固定交付 profile 另按其准入检查。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/qwen3_vl/runner.py) · [inputs.py](../../../examples/runners/qwen3_vl/inputs.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_vl/text.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_vl/vision.py) · [__init__.py](../../../python/rpu_backend/adapters/qwen3_vl/__init__.py)。

### 图文 · 32B

| 配置 | 支持状态 | 精度 | 量化范围 | prompt 默认示例 | prefill_tokens 默认值 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|---|---|
| [vl/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml](vl/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml) | 支持（`SUPPORTED`） | w4a16 | text + output head + vision | Describe this image in one sentence. | 430 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-32b-runtime-w4a16` |
| [vl/32b/w8a16_text.toml](vl/32b/w8a16_text.toml) | 支持（`SUPPORTED`） | w8a16 | text | Describe the image. | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-32b-w8a16` |
| [vl/32b/w8a16_text_head_vision.toml](vl/32b/w8a16_text_head_vision.toml) | 支持（`SUPPORTED`） | w8a16 | text + output head + vision | Describe this image in one sentence. | 430 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-32b-runtime-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 图像 + 文本 |
| 文本可调范围 | 非空提示词；P 包含图像和聊天格式 |
| 图像/音频可调范围 | 非空 RGB 图片列表；不接收视频；每图 processor 网格≤4096 patch（patch 16 px，如 1024×1024、2048×512 px）；Vision RoPE 表与 KV 容量按所配图像冷态安装，网格边长≤48且≤2048 patch 时保持默认 48 / 2048，否则随最大图像增大；板上实测到 1024×1024 px（4096 patch，8 核）、2048×512 px（32×128 网格，4096 patch，8 核，RoPE 表与 KV 容量按该网格扩大）与 896×896 px（3136 patch，4/6/8 核）；其余网格可以安装，但尚未上板运行 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1；prepared input 只用于 multimodal |
| 联动限制 | prepared input 要求 SHA、单图 grid、全1 mask、有限像素及匹配 M-RoPE。2/4/8B 普通文本 owner 的物理 prefill 上界4176，chunk 上限320/256/128；32B 的运行时量化 W8/W4 与旧 W8 文本配方同为物理 prefill 上界4176、chunk 上限64；其余量化及固定交付 profile 另按其准入检查。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/qwen3_vl/runner.py) · [inputs.py](../../../examples/runners/qwen3_vl/inputs.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_vl/text.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_vl/vision.py) · [__init__.py](../../../python/rpu_backend/adapters/qwen3_vl/__init__.py)。

### 文本 · 2B

| 配置 | 支持状态 | 精度 | 量化范围 | prompt 默认示例 | prefill_tokens 默认值 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|---|---|
| [text/2b/fp16.toml](text/2b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | Describe the image. | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-2b` |
| [text/2b/w4a16_text_awq_g32.toml](text/2b/w4a16_text_awq_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text | Explain what a compiler does. | 512 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-2b-text-awq-w4a16` |
| [text/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](text/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | 支持（`SUPPORTED`） | w4a16 | text + vision | Explain what a compiler does. | 512 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-2b-runtime-w4a16` |
| [text/2b/w8a16_text.toml](text/2b/w8a16_text.toml) | 支持（`SUPPORTED`） | w8a16 | text | Explain what a compiler does. | 512 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-2b-text-w8a16` |
| [text/2b/w8a16_text_head_vision.toml](text/2b/w8a16_text_head_vision.toml) | 支持（`SUPPORTED`） | w8a16 | text + output head + vision | Explain what a compiler does. | 512 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-2b-runtime-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / token IDs |
| 文本可调范围 | 非空文本或token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | 2/4/8B 普通文本 owner 的物理 prefill 上界4176，chunk 上限320/256/128；32B 的运行时量化 W8/W4 与旧 W8 文本配方同为物理 prefill 上界4176、chunk 上限64；其余量化及固定交付 profile 另按其准入检查。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/qwen3_vl/runner.py) · [inputs.py](../../../examples/runners/qwen3_vl/inputs.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_vl/text.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_vl/vision.py) · [__init__.py](../../../python/rpu_backend/adapters/qwen3_vl/__init__.py)。

### 文本 · 4B

| 配置 | 支持状态 | 精度 | 量化范围 | prompt 默认示例 | prefill_tokens 默认值 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|---|---|
| [text/4b/fp16.toml](text/4b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | Describe the image. | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-4b` |
| [text/4b/w4a16_text_awq_g32.toml](text/4b/w4a16_text_awq_g32.toml) | 支持（`SUPPORTED`） | w4a16 | text | Explain what a compiler does. | 512 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-4b-text-awq-w4a16` |
| [text/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](text/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | 支持（`SUPPORTED`） | w4a16 | text + vision | Explain what a compiler does. | 512 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-4b-runtime-w4a16` |
| [text/4b/w8a16_text_head_vision.toml](text/4b/w8a16_text_head_vision.toml) | 支持（`SUPPORTED`） | w8a16 | text + output head + vision | Explain what a compiler does. | 512 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-4b-runtime-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / token IDs |
| 文本可调范围 | 非空文本或token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | 2/4/8B 普通文本 owner 的物理 prefill 上界4176，chunk 上限320/256/128；32B 的运行时量化 W8/W4 与旧 W8 文本配方同为物理 prefill 上界4176、chunk 上限64；其余量化及固定交付 profile 另按其准入检查。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/qwen3_vl/runner.py) · [inputs.py](../../../examples/runners/qwen3_vl/inputs.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_vl/text.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_vl/vision.py) · [__init__.py](../../../python/rpu_backend/adapters/qwen3_vl/__init__.py)。

### 文本 · 8B

| 配置 | 支持状态 | 精度 | 量化范围 | prompt 默认示例 | prefill_tokens 默认值 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|---|---|
| [text/8b/fp16.toml](text/8b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | Describe the image. | — | 4 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-8b` |
| [text/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](text/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | 支持（`SUPPORTED`） | w4a16 | text + vision | Explain what a compiler does. | 512 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-8b-runtime-w4a16` |
| [text/8b/w8a16_text_head_vision.toml](text/8b/w8a16_text_head_vision.toml) | 支持（`SUPPORTED`） | w8a16 | text + output head + vision | Explain what a compiler does. | 512 | 31 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-8b-runtime-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / token IDs |
| 文本可调范围 | 非空文本或token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | 2/4/8B 普通文本 owner 的物理 prefill 上界4176，chunk 上限320/256/128；32B 的运行时量化 W8/W4 与旧 W8 文本配方同为物理 prefill 上界4176、chunk 上限64；其余量化及固定交付 profile 另按其准入检查。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/qwen3_vl/runner.py) · [inputs.py](../../../examples/runners/qwen3_vl/inputs.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_vl/text.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_vl/vision.py) · [__init__.py](../../../python/rpu_backend/adapters/qwen3_vl/__init__.py)。

### 文本 · 32B

| 配置 | 支持状态 | 精度 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|
| [text/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml](text/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml) | 支持（`SUPPORTED`） | w4a16 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-32b-runtime-w4a16` |
| [text/32b/w8a16_text_head_vision.toml](text/32b/w8a16_text_head_vision.toml) | 支持（`SUPPORTED`） | w8a16 | ● 已实测（注意力 4 · MLP 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4 · 视觉 4） | ● 已实测 | `qwen3-vl-32b-runtime-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 量化范围 | text + output head + vision |
| 输入类型 | 文本 / token IDs |
| prompt 默认示例 | Explain what a compiler does. |
| prefill_tokens 默认值 | 512 |
| decode_steps 默认值 | 31 |
| 文本可调范围 | 非空文本或token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | 2/4/8B 普通文本 owner 的物理 prefill 上界4176，chunk 上限320/256/128；32B 的运行时量化 W8/W4 与旧 W8 文本配方同为物理 prefill 上界4176、chunk 上限64；其余量化及固定交付 profile 另按其准入检查。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/qwen3_vl/runner.py) · [inputs.py](../../../examples/runners/qwen3_vl/inputs.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_vl/text.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_vl/vision.py) · [__init__.py](../../../python/rpu_backend/adapters/qwen3_vl/__init__.py)。

### 视觉特征 · 2B

| 项目 | 值 |
|---|---|
| 配置 | [vision/2b/fp16.toml](vision/2b/fp16.toml) |
| 支持状态 | 支持（`SUPPORTED`） |
| 精度 | fp16 |
| 量化范围 | — |
| 输入类型 | 图像（视觉特征） |
| 文本可调范围 | 不使用 |
| 图像/音频可调范围 | 非空 RGB 图片列表；不接收视频；每图 processor 网格≤4096 patch（patch 16 px，如 1024×1024、2048×512 px）；Vision RoPE 表与 KV 容量按所配图像冷态安装，网格边长≤48且≤2048 patch 时保持默认 48 / 2048，否则随最大图像增大；板上实测到 1024×1024 px（4096 patch，8 核）、2048×512 px（32×128 网格，4096 patch，8 核，RoPE 表与 KV 容量按该网格扩大）与 896×896 px（3136 patch，4/6/8 核）；其余网格可以安装，但尚未上板运行 |
| 生成可调范围 | 不生成 |
| 其他输入形状 | processor 生成 pixel_values 和 image_grid_thw |
| 联动限制 | 图像 grid 与视觉特征布局匹配；图数与像素受 processor、视觉 plan 和内存约束。 |
| 默认核数 | 8 |
| 4 核 | ● 已实测（视觉 4） |
| 6 核 | ● 已实测（视觉 4） |
| 8 核 | ● 已实测 |
| CPU线程数 | 未设置（沿用环境） |
| 权重 | `qwen3-vl-2b` |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/qwen3_vl/runner.py) · [inputs.py](../../../examples/runners/qwen3_vl/inputs.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_vl/text.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_vl/vision.py) · [__init__.py](../../../python/rpu_backend/adapters/qwen3_vl/__init__.py)。

### 视觉特征 · 4B

| 项目 | 值 |
|---|---|
| 配置 | [vision/4b/fp16.toml](vision/4b/fp16.toml) |
| 支持状态 | 支持（`SUPPORTED`） |
| 精度 | fp16 |
| 量化范围 | — |
| 输入类型 | 图像（视觉特征） |
| 文本可调范围 | 不使用 |
| 图像/音频可调范围 | 非空 RGB 图片列表；不接收视频；每图 processor 网格≤4096 patch（patch 16 px，如 1024×1024、2048×512 px）；Vision RoPE 表与 KV 容量按所配图像冷态安装，网格边长≤48且≤2048 patch 时保持默认 48 / 2048，否则随最大图像增大；板上实测到 1024×1024 px（4096 patch，8 核）、2048×512 px（32×128 网格，4096 patch，8 核，RoPE 表与 KV 容量按该网格扩大）与 896×896 px（3136 patch，4/6/8 核）；其余网格可以安装，但尚未上板运行 |
| 生成可调范围 | 不生成 |
| 其他输入形状 | processor 生成 pixel_values 和 image_grid_thw |
| 联动限制 | 图像 grid 与视觉特征布局匹配；图数与像素受 processor、视觉 plan 和内存约束。 |
| 默认核数 | 8 |
| 4 核 | ● 已实测（视觉 4） |
| 6 核 | ● 已实测（视觉 4） |
| 8 核 | ● 已实测 |
| CPU线程数 | 未设置（沿用环境） |
| 权重 | `qwen3-vl-4b` |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/qwen3_vl/runner.py) · [inputs.py](../../../examples/runners/qwen3_vl/inputs.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_vl/text.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_vl/vision.py) · [__init__.py](../../../python/rpu_backend/adapters/qwen3_vl/__init__.py)。

### 视觉特征 · 8B

| 项目 | 值 |
|---|---|
| 配置 | [vision/8b/fp16.toml](vision/8b/fp16.toml) |
| 支持状态 | 支持（`SUPPORTED`） |
| 精度 | fp16 |
| 量化范围 | — |
| 输入类型 | 图像（视觉特征） |
| 文本可调范围 | 不使用 |
| 图像/音频可调范围 | 非空 RGB 图片列表；不接收视频；每图 processor 网格≤4096 patch（patch 16 px，如 1024×1024、2048×512 px）；Vision RoPE 表与 KV 容量按所配图像冷态安装，网格边长≤48且≤2048 patch 时保持默认 48 / 2048，否则随最大图像增大；板上实测到 1024×1024 px（4096 patch，8 核）、2048×512 px（32×128 网格，4096 patch，8 核，RoPE 表与 KV 容量按该网格扩大）与 896×896 px（3136 patch，4/6/8 核）；其余网格可以安装，但尚未上板运行 |
| 生成可调范围 | 不生成 |
| 其他输入形状 | processor 生成 pixel_values 和 image_grid_thw |
| 联动限制 | 图像 grid 与视觉特征布局匹配；图数与像素受 processor、视觉 plan 和内存约束。 |
| 默认核数 | 8 |
| 4 核 | ● 已实测（视觉 4） |
| 6 核 | ● 已实测（视觉 4） |
| 8 核 | ● 已实测 |
| CPU线程数 | 未设置（沿用环境） |
| 权重 | `qwen3-vl-8b` |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/qwen3_vl/runner.py) · [inputs.py](../../../examples/runners/qwen3_vl/inputs.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_vl/text.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_vl/vision.py) · [__init__.py](../../../python/rpu_backend/adapters/qwen3_vl/__init__.py)。

<!-- model-configurations-end -->
