# Qwen3.5 配置

[配置总览](../README.md) · [全部配置](#configurations) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

<a id="快速选择"></a>

## 选择模型与配方

Dense 包括 **0.8B、2B、4B、9B、27B**，MoE 为 **35B-A3B**；每个尺寸按 `text/` 文本或 `vl/` 图文入口选择。全部配方、默认输入和可调范围列在[本页配置表](#configurations)。

Dense 提供 FP16、W8、RTN W4 G32；部分尺寸另有 MLP-only、混合 W4/W8 和 AWQ。五个尺寸的文本入口另有 NVFP4（`nvfp4_text`）：加载时由官方浮点权重转换一次，GDN `in_proj_a`/`in_proj_b`、输出头、嵌入和归一化保持 FP16，尚未上板实测。2B/4B/9B 的 AWQ 使用原始 AWQ 包，27B 使用本仓转换的激活感知 AWQ G128。MoE 提供 mixed E4M3、RTN W4 G128、GPTQ W4 G128 和路由专家 W8A16（`w8a16_experts`：离线转换上游浮点权重，只量化路由专家，ACC16，不接受常驻 prefill，尚未上板实测）。权重须匹配量化范围和转换方法。

文本入口在 8 核下的实测如下（性能页，默认输入 P7 / D4）。每格为 TPOT（ms，每个输出 token 的 decode 时间）/ 末 token logits 相对浮点权重的 rel L2；误差只作记录，不是质量结论。全部配方的输出头保持 FP16。

| 尺寸 | FP16 | W8 `w8a16_text` | W4 `w4a16_text_rtn_g32` | 其他 W4 |
|---|---|---|---|---|
| 0.8B | 17.2 / 0.63% | 14.0 / 4.6% | 12.7 / 31% | — |
| 2B | 32.4 / 0.86% | 24.4 / 4.9% | 20.3 / 37% | MLP W4 + Attention/GDN W8 21.5 / 33%；MLP RTN 24.9 / 34%；MLP AWQ 25.8 / 37% |
| 4B | 67.8 / 0.56% | 44.3 / 3.4% | 35.3 / 29% | MLP AWQ 47.4 / 25% |
| 9B | 117.4 / 0.48% | 73.5 / 2.5% | 55.5 / 26% | MLP AWQ 75.3 / 16% |
| 27B | 387.9 / 0.52% | 220.2 / 2.0% | 155.0 / 17% | MLP W4 + Attention/GDN W8 175.7 / 16%；MLP RTN 228.4 / 16%；激活感知 AWQ 228.9 / 10% |

W8 的 TPOT 约为 FP16 的 57%–81%，误差 2%–5%；整体 W4 再快 9%–30%，误差升到 17%–37%。MLP-only 与 AWQ 配方的误差低于整体 W4：MLP RTN 与 AWQ 的速度接近 W8，MLP W4 + Attention/GDN W8 介于 W8 与整体 W4 之间。35B-A3B MoE（P13 / D31）：FP8 54.8 / 4.5%，RTN W4 G128 59.2 / 8.4%，GPTQ W4 G128 60.4 / 9.1%。图文入口的 TPOT 与同配方文本入口相近，见性能页。

## 调整输入与核心

每个配方只保留一份配置，直接调整提示词、`input.prefill_tokens`、`input.decode_steps` 和 `rpu_execution`。默认 8 核；每个配置的 4/6/8 核状态和联动限制见[配置表](#configurations)。修改核心数、累加模式或输入形状后，使用新进程运行。

- **Dense 文本**：不设置 `prefill_tokens` 时使用 tokenizer 的实际长度；固定 P 时保留原始 token IDs 或原文本。0.8B/2B/4B/9B FP16 的长 prefill profile 用同一 `fp16.toml`，把 `model.profile_id` 改为 `qwen3_5-<尺寸>.fp16.text-long-prefill`、`input.prefill_tokens` 设为 4096；2B 的既有显式 chunk 为 640，也可保留 `auto`。覆盖值见 profile_variants.json。这 4 个长 prefill profile 已支持（`SUPPORTED`）：2026-10-24 用当前模板加上述覆盖值在板上跑 P4096 / D4，4/6/8 核的运行时检查均通过（见各尺寸性能页）；这些输入没有同输入 CPU 参考，误差未记录。chunk 取 `auto` 或不超过尺寸上限（512/640/448/256）的 64 倍数；`auto` 实测选中 0.8B 512（4/6/8 核）、2B 512（4 核）、4B 448（8 核）/256（4/6 核）、9B 256（8 核）/192（4/6 核）。2B 的固定 640 只能在 6 或 8 核运行：4 核没有可行的 640 组合，规划器按精确请求拒绝，4 核时把 chunk 设为 `auto`。长 prefill 是 profile 覆盖值，不是单独的 TOML，所以不在配置表中。当前模板（`*.fp16.text`）不改 profile 时的实测最长输入为 P2048。
- **MoE 固定文本**：设置 `prefill_tokens` 时使用 `prompt_file`，同时填写 `prompt_sha256` 和 `input_ids_sha256`，删除自然 `prompt`；`run.timing_mode="forward_argmax"`、`input.stop_on_eos=false`，按需要设置 D。原文必须至少有 P 个 token，不重复或补齐。历史测量的原文和两项摘要须来自对应记录。
- **图文**：P 包含图像占位和聊天格式，不能短于 processor 的完整输出。图像数、分辨率、P/D 还受视觉 plan 和内存共同限制。历史 Dense VL 长输入计时使用 `run.timing_mode="legacy_dense_vl"`、`run.inference_mode=true`、`run.summary_statistic="median"`、`input.image_size=448`、原 `input.prefill_filler_prompt` 及 12 个 CPU 线程；复现时一并恢复这些条件。

27B 与其他 Dense 使用同样的参数组织。其已测输入边界与权重转换方法见下文。

<a id="权重与运行"></a>

## 准备与运行

1. 先在[权重准备](../WEIGHTS.md)中选择对应尺寸和配方。
2. 在[全部配置](#configurations)中打开 TOML，确认输入类型、精度、核心范围和权重别名。
3. 从仓库根目录先检查配置，再运行统一示例入口：

```bash
python examples/run.py --config examples/configs/qwen3_5/text/4b/fp16.toml --check-config
python examples/run.py --config examples/configs/qwen3_5/text/4b/fp16.toml
```

`--check-config` 通过只说明配置与登记 profile 匹配。输入路径、输入档、核心数和累加模式改变后，使用新进程加载配置。输出文件格式见[日志与结果](../../RESULTS.md)。

## 27B 权重配方

27B 的量化范围分别为全部文本投影、MLP-only、MLP W4 + Attention/GDN W8，以及首层之外 MLP 的激活感知 AWQ。各范围对应独立转换产物；NVFP4 没有转换产物，加载 `qwen3_5-27b-fp16` 时转换。具体 TOML 见[全部配置](#configurations)。

各配方的短输入、固定前缀和长输入通过同一 TOML 调参。8 核为默认，可选择 4/6/8 核；27B FP16 的既有验证覆盖短输入，长输入优先使用量化权重。

<a id="device-validation"></a>

<a id="设备验证范围"></a>

### 27B 核数与已测输入

各 27B 配置的 4/6/8 核状态见下方[配置表](#configurations)：2026-10-19 起，每个配置都以模板默认输入在 4/6/8 核上运行并通过运行时检查。此前用其他输入（64 个输入 token、单张 224×224 图像）的测量见27B 性能报告。

<a id="weight-recipes"></a>
<a id="text-w4-recipes"></a>
<a id="quantization"></a>

### 转换官方 27B 权重

转换命令统一列在[权重准备](../WEIGHTS.md#cmd-qwen35-dense)，每个配方输出到独立目录。FP16 配置使用预先导出的 `qwen3_5-27b-fp16`（`Qwen3.5-27B-FP16`），减少加载时的权重转换内存；运行还需足够的连续设备内存。`qwen3_5-27b` 保留官方原始权重。

激活感知 AWQ 产物使用 `activation_awq_g128_v1`，与下载的原始 AWQ 包分开登记；校准文件应来自同一浮点权重，转换记录随产物保存。MLP-only、混合 W4/W8 和激活感知 AWQ 当前只登记文本入口。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：Qwen3.5。

P 为处理后的输入 token 数，D 为 decode 调用数；生成首 token 来自 prefill。

### MoE 文本 · 35B-A3B

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 | CPU线程数 | 权重 |
|---|---|---|---|---|---|---|---|---|
| [text/35b_a3b/fp8_e4m3_experts_attn_gdn.toml](text/35b_a3b/fp8_e4m3_experts_attn_gdn.toml) | 支持（`SUPPORTED`） | fp8_e4m3 | 路由专家 / Attention / GDN FP8 E4M3 | ● 已实测（注意力 4 · 输出头 4） | ● 已实测（注意力 4 · 输出头 6） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-35b-a3b-mixed` |
| [text/35b_a3b/w4a16_experts_gptq_g128.toml](text/35b_a3b/w4a16_experts_gptq_g128.toml) | 支持（`SUPPORTED`） | w4a16 | 路由专家 W4A16 G128 GPTQ | ● 已实测（注意力 4 · 输出头 4） | ● 已实测（注意力 4 · 输出头 4） | ● 已实测 | 24 | `qwen3_5-35b-a3b-gptq-w4a16` |
| [text/35b_a3b/w4a16_experts_rtn_g128.toml](text/35b_a3b/w4a16_experts_rtn_g128.toml) | 支持（`SUPPORTED`） | w4a16 | 路由专家 W4A16 G128 RTN | ● 已实测（注意力 4 · 输出头 4） | ● 已实测（注意力 4 · 输出头 4） | ● 已实测 | 24 | `qwen3_5-35b-a3b-rtn-w4a16` |
| [text/35b_a3b/w8a16_experts.toml](text/35b_a3b/w8a16_experts.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | w8a16 | 路由专家 W8A16 | ○ 未实测 | ○ 未实测 | ○ 未实测 | 24 | `qwen3_5-35b-a3b-rtn-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / 真实文本文件 |
| prompt 默认示例 | A compiler translates source code into executable instructions. Its main stages are |
| decode_steps 默认值 | 31 |
| 文本可调范围 | P≥1；固定 P 时原文至少有 P 个 token |
| 生成可调范围 | D≥0；P+D≤8192 |
| 其他输入形状 | B=1 |
| 联动限制 | 指定 prefill_tokens 时使用 prompt_file，并绑定文本 SHA 与 token SHA；forward_argmax、固定 D、stop_on_eos=false。不重复或补齐原文；RTN/GPTQ 使用 ACC16。 |
| 默认核数 | 8 |
| 入口 | [run.py](../../run.py) |

输入校验来源：[config.py](../../../examples/common/config.py) · [generation.py](../../../examples/common/generation.py) · [text.py](../../../examples/common/text.py) · [qwen3_5_moe_cache.py](../../../python/rpu_backend/api/qwen3_5_moe_cache.py) · [__init__.py](../../../python/rpu_backend/adapters/qwen3_5_moe/__init__.py)。

### MoE 图文 · 35B-A3B

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 | CPU线程数 | 权重 |
|---|---|---|---|---|---|---|---|---|
| [vl/35b_a3b/fp8_e4m3_experts_attn_gdn.toml](vl/35b_a3b/fp8_e4m3_experts_attn_gdn.toml) | 支持（`SUPPORTED`） | fp8_e4m3 | 路由专家 / Attention / GDN FP8 E4M3 | ● 已实测（注意力 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · 输出头 4 · 视觉 4） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-35b-a3b-mixed` |
| [vl/35b_a3b/w4a16_experts_gptq_g128.toml](vl/35b_a3b/w4a16_experts_gptq_g128.toml) | 支持（`SUPPORTED`） | w4a16 | 路由专家 W4A16 G128 GPTQ | ● 已实测（注意力 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · 输出头 4 · 视觉 4） | ● 已实测 | 24 | `qwen3_5-35b-a3b-gptq-w4a16` |
| [vl/35b_a3b/w4a16_experts_rtn_g128.toml](vl/35b_a3b/w4a16_experts_rtn_g128.toml) | 支持（`SUPPORTED`） | w4a16 | 路由专家 W4A16 G128 RTN | ● 已实测（注意力 4 · 输出头 4 · 视觉 4） | ● 已实测（注意力 4 · 输出头 4 · 视觉 4） | ● 已实测 | 24 | `qwen3_5-35b-a3b-rtn-w4a16` |
| [vl/35b_a3b/w8a16_experts.toml](vl/35b_a3b/w8a16_experts.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | w8a16 | 路由专家 W8A16 | ○ 未实测 | ○ 未实测 | ○ 未实测 | 24 | `qwen3_5-35b-a3b-rtn-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 图像 + 文本 |
| prompt 默认示例 | Describe the image. |
| decode_steps 默认值 | 4 |
| 文本可调范围 | 非空提示词；P 包含图像和聊天格式 |
| 图像/音频可调范围 | 非空 RGB 图片列表；不接收视频 |
| 生成可调范围 | D≥0；ceil64(P+D)≤8192 |
| 其他输入形状 | B=1；T=1，H/W 整除 merge_size |
| 联动限制 | 指定 P 不可截断完整图文输入；图像 token 与特征匹配；chunk 上限512。图数和像素需同时满足 processor、视觉 plan、cache 与内存约束。 |
| 默认核数 | 8 |
| 入口 | [run.py](../../run.py) |

输入校验来源：[vision.py](../../../examples/runners/qwen3_5/vision.py) · [config.py](../../../examples/common/config.py) · [qwen3_5_moe_cache.py](../../../python/rpu_backend/api/qwen3_5_moe_cache.py) · [__init__.py](../../../python/rpu_backend/adapters/qwen3_5_moe/__init__.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_5/vision.py)。

### Dense 文本 · 0.8B

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|
| [text/0_8b/fp16.toml](text/0_8b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3_5-0.8b` |
| [text/0_8b/nvfp4_text.toml](text/0_8b/nvfp4_text.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | nvfp4 | Attention / GDN / MLP NVFP4（GDN in_proj_a/in_proj_b FP16） | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3_5-0.8b` |
| [text/0_8b/w4a16_text_rtn_g32.toml](text/0_8b/w4a16_text_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | Attention / GDN / MLP W4A16 G32 RTN | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3_5-0.8b-w4a16` |
| [text/0_8b/w8a16_text.toml](text/0_8b/w8a16_text.toml) | 支持（`SUPPORTED`） | w8a16 | Attention / GDN / MLP W8A16 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | `qwen3_5-0.8b-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / token IDs |
| prompt 默认示例 | Explain what a compiler does. |
| decode_steps 默认值 | 4 |
| 文本可调范围 | 非空文本或非负 token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | prefill 的物理位置+执行长度≤8192。chunk 上限按尺寸为512/640/448/256/128（0.8/2/4/9/27B）；padding、cache 和设备内存共同约束。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[config.py](../../../examples/common/config.py) · [text.py](../../../examples/common/text.py) · [generation.py](../../../examples/common/generation.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_5/text.py) · [qwen3_5_cache.py](../../../python/rpu_backend/api/qwen3_5_cache.py) · [cores.py](../../../python/rpu_backend/adapters/qwen3_5/cores.py)。

### Dense 文本 · 2B

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 | CPU线程数 | 权重 |
|---|---|---|---|---|---|---|---|---|
| [text/2b/fp16.toml](text/2b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-2b` |
| [text/2b/nvfp4_text.toml](text/2b/nvfp4_text.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | nvfp4 | Attention / GDN / MLP NVFP4（GDN in_proj_a/in_proj_b FP16） | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 6） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-2b` |
| [text/2b/w4a16_mlp_awq_g128.toml](text/2b/w4a16_mlp_awq_g128.toml) | 支持（`SUPPORTED`） | w4a16 | MLP W4A16 G128 AWQ（第 0 层 FP16） | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 4 | `qwen3_5-2b-quanttrio-awq-g128` |
| [text/2b/w4a16_mlp_rtn_g32.toml](text/2b/w4a16_mlp_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | MLP W4A16 G32 RTN | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 4 | `qwen3_5-2b-w4-mlp-only-g32` |
| [text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml](text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml) | 支持（`SUPPORTED`） | w4a16 | MLP W4A16 G32 RTN；Attention / GDN W8A16 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 4 | `qwen3_5-2b-w4-mlp-w8-attn-gdn-g32` |
| [text/2b/w4a16_text_rtn_g32.toml](text/2b/w4a16_text_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | Attention / GDN / MLP W4A16 G32 RTN | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-2b-w4a16` |
| [text/2b/w8a16_text.toml](text/2b/w8a16_text.toml) | 支持（`SUPPORTED`） | w8a16 | Attention / GDN / MLP W8A16 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-2b-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / token IDs |
| prompt 默认示例 | Explain what a compiler does. |
| decode_steps 默认值 | 4 |
| 文本可调范围 | 非空文本或非负 token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | prefill 的物理位置+执行长度≤8192。chunk 上限按尺寸为512/640/448/256/128（0.8/2/4/9/27B）；padding、cache 和设备内存共同约束。 |
| 默认核数 | 8 |
| 入口 | [run.py](../../run.py) |

输入校验来源：[config.py](../../../examples/common/config.py) · [text.py](../../../examples/common/text.py) · [generation.py](../../../examples/common/generation.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_5/text.py) · [qwen3_5_cache.py](../../../python/rpu_backend/api/qwen3_5_cache.py) · [cores.py](../../../python/rpu_backend/adapters/qwen3_5/cores.py)。

### Dense 文本 · 4B

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 | CPU线程数 | 权重 |
|---|---|---|---|---|---|---|---|---|
| [text/4b/fp16.toml](text/4b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-4b` |
| [text/4b/nvfp4_text.toml](text/4b/nvfp4_text.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | nvfp4 | Attention / GDN / MLP NVFP4（GDN in_proj_a/in_proj_b FP16） | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-4b` |
| [text/4b/w4a16_mlp_awq_g128.toml](text/4b/w4a16_mlp_awq_g128.toml) | 支持（`SUPPORTED`） | w4a16 | MLP W4A16 G128 AWQ（第 0 层 FP16） | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 4 | `qwen3_5-4b-quanttrio-awq-g128` |
| [text/4b/w4a16_text_rtn_g32.toml](text/4b/w4a16_text_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | Attention / GDN / MLP W4A16 G32 RTN | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-4b-w4a16` |
| [text/4b/w8a16_text.toml](text/4b/w8a16_text.toml) | 支持（`SUPPORTED`） | w8a16 | Attention / GDN / MLP W8A16 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-4b-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / token IDs |
| prompt 默认示例 | Explain what a compiler does. |
| decode_steps 默认值 | 4 |
| 文本可调范围 | 非空文本或非负 token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | prefill 的物理位置+执行长度≤8192。chunk 上限按尺寸为512/640/448/256/128（0.8/2/4/9/27B）；padding、cache 和设备内存共同约束。 |
| 默认核数 | 8 |
| 入口 | [run.py](../../run.py) |

输入校验来源：[config.py](../../../examples/common/config.py) · [text.py](../../../examples/common/text.py) · [generation.py](../../../examples/common/generation.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_5/text.py) · [qwen3_5_cache.py](../../../python/rpu_backend/api/qwen3_5_cache.py) · [cores.py](../../../python/rpu_backend/adapters/qwen3_5/cores.py)。

### Dense 文本 · 9B

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 | CPU线程数 | 权重 |
|---|---|---|---|---|---|---|---|---|
| [text/9b/fp16.toml](text/9b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-9b` |
| [text/9b/nvfp4_text.toml](text/9b/nvfp4_text.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | nvfp4 | Attention / GDN / MLP NVFP4（GDN in_proj_a/in_proj_b FP16） | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 6） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-9b` |
| [text/9b/w4a16_mlp_awq_g128.toml](text/9b/w4a16_mlp_awq_g128.toml) | 支持（`SUPPORTED`） | w4a16 | MLP W4A16 G128 AWQ（第 0 层 FP16） | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 4 | `qwen3_5-9b-quanttrio-awq-g128` |
| [text/9b/w4a16_text_rtn_g32.toml](text/9b/w4a16_text_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | Attention / GDN / MLP W4A16 G32 RTN | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-9b-w4a16` |
| [text/9b/w8a16_text.toml](text/9b/w8a16_text.toml) | 支持（`SUPPORTED`） | w8a16 | Attention / GDN / MLP W8A16 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-9b-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / token IDs |
| prompt 默认示例 | Explain what a compiler does. |
| decode_steps 默认值 | 4 |
| 文本可调范围 | 非空文本或非负 token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | prefill 的物理位置+执行长度≤8192。chunk 上限按尺寸为512/640/448/256/128（0.8/2/4/9/27B）；padding、cache 和设备内存共同约束。 |
| 默认核数 | 8 |
| 入口 | [run.py](../../run.py) |

输入校验来源：[config.py](../../../examples/common/config.py) · [text.py](../../../examples/common/text.py) · [generation.py](../../../examples/common/generation.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_5/text.py) · [qwen3_5_cache.py](../../../python/rpu_backend/api/qwen3_5_cache.py) · [cores.py](../../../python/rpu_backend/adapters/qwen3_5/cores.py)。

### Dense 文本 · 27B

| 配置 | 支持状态 | 精度 | 量化范围 | 4 核 | 6 核 | 8 核 | CPU线程数 | 权重 |
|---|---|---|---|---|---|---|---|---|
| [text/27b/fp16.toml](text/27b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-27b-fp16` |
| [text/27b/nvfp4_text.toml](text/27b/nvfp4_text.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | nvfp4 | Attention / GDN / MLP NVFP4（GDN in_proj_a/in_proj_b FP16） | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-27b-fp16` |
| [text/27b/w4a16_mlp_activation_awq_g128.toml](text/27b/w4a16_mlp_activation_awq_g128.toml) | 支持（`SUPPORTED`） | w4a16 | MLP W4A16 G128 activation AWQ（第 0 层 FP16） | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 4 | `qwen3_5-27b-activation-awq-g128` |
| [text/27b/w4a16_mlp_rtn_g32.toml](text/27b/w4a16_mlp_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | MLP W4A16 G32 RTN | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 4 | `qwen3_5-27b-w4-mlp-only-g32` |
| [text/27b/w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml](text/27b/w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml) | 支持（`SUPPORTED`） | w4a16 | MLP W4A16 G32 RTN；Attention / GDN W8A16 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 4 | `qwen3_5-27b-w4-mlp-w8-attn-gdn-g32` |
| [text/27b/w4a16_text_rtn_g32.toml](text/27b/w4a16_text_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | Attention / GDN / MLP W4A16 G32 RTN | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 8 | `qwen3_5-27b-w4a16` |
| [text/27b/w8a16_text.toml](text/27b/w8a16_text.toml) | 支持（`SUPPORTED`） | w8a16 | Attention / GDN / MLP W8A16 | ● 已实测（注意力 4 · MLP 4 · 输出头 4） | ● 已实测（注意力 4 · MLP 6 · 输出头 4） | ● 已实测 | 8 | `qwen3_5-27b-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 文本 / token IDs |
| prompt 默认示例 | Explain what a compiler does. |
| decode_steps 默认值 | 4 |
| 文本可调范围 | 非空文本或非负 token IDs；P≥1 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1 |
| 联动限制 | prefill 的物理位置+执行长度≤8192。chunk 上限按尺寸为512/640/448/256/128（0.8/2/4/9/27B）；padding、cache 和设备内存共同约束。 |
| 默认核数 | 8 |
| 入口 | [run.py](../../run.py) |

输入校验来源：[config.py](../../../examples/common/config.py) · [text.py](../../../examples/common/text.py) · [generation.py](../../../examples/common/generation.py) · [text.py](../../../python/rpu_backend/adapters/qwen3_5/text.py) · [qwen3_5_cache.py](../../../python/rpu_backend/api/qwen3_5_cache.py) · [cores.py](../../../python/rpu_backend/adapters/qwen3_5/cores.py)。

### Dense 图文 · 0.8B

| 配置 | 支持状态 | 精度 | 量化范围 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|
| [vl/0_8b/fp16.toml](vl/0_8b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 4 | ● 已实测（视觉 4） | ● 已实测（视觉 4） | ● 已实测 | `qwen3_5-0.8b` |
| [vl/0_8b/w4a16_text_rtn_g32.toml](vl/0_8b/w4a16_text_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | Attention / GDN / MLP W4A16 G32 RTN | 32 | ● 已实测（视觉 4） | ● 已实测（视觉 4） | ● 已实测 | `qwen3_5-0.8b-w4a16` |
| [vl/0_8b/w8a16_text.toml](vl/0_8b/w8a16_text.toml) | 支持（`SUPPORTED`） | w8a16 | Attention / GDN / MLP W8A16 | 32 | ● 已实测（视觉 4） | ● 已实测（视觉 4） | ● 已实测 | `qwen3_5-0.8b-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 图像 + 文本 |
| prompt 默认示例 | Describe the image. |
| 文本可调范围 | 非空提示词；P 包含图像和聊天格式 |
| 图像/音频可调范围 | 非空图片列表；RGB；可指定正方形 image_size；不接收视频 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1；每图 grid 的 T=1，H/W 为 merge_size 的正整数倍 |
| 联动限制 | 显式 P 不能短于 processor 完整输出；图像占位数须匹配视觉特征。数量、像素和文本共同受视觉 plan、cache 与内存约束，没有独立统一的图数或像素上限。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[vision.py](../../../examples/runners/qwen3_5/vision.py) · [inputs.py](../../../examples/common/inputs.py) · [generation.py](../../../examples/common/generation.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_5/vision.py) · [cores.py](../../../python/rpu_backend/adapters/qwen3_5/cores.py)。

### Dense 图文 · 2B

| 配置 | 支持状态 | 精度 | 量化范围 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|
| [vl/2b/fp16.toml](vl/2b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 4 | ● 已实测（视觉 4） | ● 已实测（视觉 4） | ● 已实测 | `qwen3_5-2b` |
| [vl/2b/w4a16_text_rtn_g32.toml](vl/2b/w4a16_text_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | Attention / GDN / MLP W4A16 G32 RTN | 32 | ● 已实测（视觉 4） | ● 已实测（视觉 4） | ● 已实测 | `qwen3_5-2b-w4a16` |
| [vl/2b/w8a16_text.toml](vl/2b/w8a16_text.toml) | 支持（`SUPPORTED`） | w8a16 | Attention / GDN / MLP W8A16 | 32 | ● 已实测（视觉 4） | ● 已实测（视觉 4） | ● 已实测 | `qwen3_5-2b-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 图像 + 文本 |
| prompt 默认示例 | Describe the image. |
| 文本可调范围 | 非空提示词；P 包含图像和聊天格式 |
| 图像/音频可调范围 | 非空图片列表；RGB；可指定正方形 image_size；不接收视频 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1；每图 grid 的 T=1，H/W 为 merge_size 的正整数倍 |
| 联动限制 | 显式 P 不能短于 processor 完整输出；图像占位数须匹配视觉特征。数量、像素和文本共同受视觉 plan、cache 与内存约束，没有独立统一的图数或像素上限。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[vision.py](../../../examples/runners/qwen3_5/vision.py) · [inputs.py](../../../examples/common/inputs.py) · [generation.py](../../../examples/common/generation.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_5/vision.py) · [cores.py](../../../python/rpu_backend/adapters/qwen3_5/cores.py)。

### Dense 图文 · 4B

| 配置 | 支持状态 | 精度 | 量化范围 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|
| [vl/4b/fp16.toml](vl/4b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 4 | ● 已实测（视觉 4） | ● 已实测（视觉 4） | ● 已实测 | `qwen3_5-4b` |
| [vl/4b/w4a16_text_rtn_g32.toml](vl/4b/w4a16_text_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | Attention / GDN / MLP W4A16 G32 RTN | 32 | ● 已实测（视觉 4） | ● 已实测（视觉 4） | ● 已实测 | `qwen3_5-4b-w4a16` |
| [vl/4b/w8a16_text.toml](vl/4b/w8a16_text.toml) | 支持（`SUPPORTED`） | w8a16 | Attention / GDN / MLP W8A16 | 32 | ● 已实测（视觉 4） | ● 已实测（视觉 4） | ● 已实测 | `qwen3_5-4b-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 图像 + 文本 |
| prompt 默认示例 | Describe the image. |
| 文本可调范围 | 非空提示词；P 包含图像和聊天格式 |
| 图像/音频可调范围 | 非空图片列表；RGB；可指定正方形 image_size；不接收视频 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1；每图 grid 的 T=1，H/W 为 merge_size 的正整数倍 |
| 联动限制 | 显式 P 不能短于 processor 完整输出；图像占位数须匹配视觉特征。数量、像素和文本共同受视觉 plan、cache 与内存约束，没有独立统一的图数或像素上限。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[vision.py](../../../examples/runners/qwen3_5/vision.py) · [inputs.py](../../../examples/common/inputs.py) · [generation.py](../../../examples/common/generation.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_5/vision.py) · [cores.py](../../../python/rpu_backend/adapters/qwen3_5/cores.py)。

### Dense 图文 · 9B

| 配置 | 支持状态 | 精度 | 量化范围 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | 权重 |
|---|---|---|---|---|---|---|---|---|
| [vl/9b/fp16.toml](vl/9b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 4 | ● 已实测（视觉 4） | ● 已实测（视觉 4） | ● 已实测 | `qwen3_5-9b` |
| [vl/9b/w8a16_text.toml](vl/9b/w8a16_text.toml) | 支持（`SUPPORTED`） | w8a16 | Attention / GDN / MLP W8A16 | 32 | ● 已实测（视觉 4） | ● 已实测（视觉 4） | ● 已实测 | `qwen3_5-9b-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 图像 + 文本 |
| prompt 默认示例 | Describe the image. |
| 文本可调范围 | 非空提示词；P 包含图像和聊天格式 |
| 图像/音频可调范围 | 非空图片列表；RGB；可指定正方形 image_size；不接收视频 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1；每图 grid 的 T=1，H/W 为 merge_size 的正整数倍 |
| 联动限制 | 显式 P 不能短于 processor 完整输出；图像占位数须匹配视觉特征。数量、像素和文本共同受视觉 plan、cache 与内存约束，没有独立统一的图数或像素上限。 |
| 默认核数 | 8 |
| CPU线程数 | 未设置（沿用环境） |
| 入口 | [run.py](../../run.py) |

输入校验来源：[vision.py](../../../examples/runners/qwen3_5/vision.py) · [inputs.py](../../../examples/common/inputs.py) · [generation.py](../../../examples/common/generation.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_5/vision.py) · [cores.py](../../../python/rpu_backend/adapters/qwen3_5/cores.py)。

### Dense 图文 · 27B

| 配置 | 支持状态 | 精度 | 量化范围 | decode_steps 默认值 | 4 核 | 6 核 | 8 核 | CPU线程数 | 权重 |
|---|---|---|---|---|---|---|---|---|---|
| [vl/27b/fp16.toml](vl/27b/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 4 | ● 已实测（视觉 4） | ● 已实测（视觉 4） | ● 已实测 | 未设置（沿用环境） | `qwen3_5-27b-fp16` |
| [vl/27b/w4a16_text_rtn_g32.toml](vl/27b/w4a16_text_rtn_g32.toml) | 支持（`SUPPORTED`） | w4a16 | Attention / GDN / MLP W4A16 G32 RTN | 32 | ● 已实测（视觉 4） | ● 已实测（视觉 4） | ● 已实测 | 8 | `qwen3_5-27b-w4a16` |
| [vl/27b/w8a16_text.toml](vl/27b/w8a16_text.toml) | 支持（`SUPPORTED`） | w8a16 | Attention / GDN / MLP W8A16 | 32 | ● 已实测（视觉 4） | ● 已实测（视觉 4） | ● 已实测 | 8 | `qwen3_5-27b-w8a16` |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 图像 + 文本 |
| prompt 默认示例 | Describe the image. |
| 文本可调范围 | 非空提示词；P 包含图像和聊天格式 |
| 图像/音频可调范围 | 非空图片列表；RGB；可指定正方形 image_size；不接收视频 |
| 生成可调范围 | D≥0；EOS 可提前结束 |
| 其他输入形状 | B=1；每图 grid 的 T=1，H/W 为 merge_size 的正整数倍 |
| 联动限制 | 显式 P 不能短于 processor 完整输出；图像占位数须匹配视觉特征。数量、像素和文本共同受视觉 plan、cache 与内存约束，没有独立统一的图数或像素上限。 |
| 默认核数 | 8 |
| 入口 | [run.py](../../run.py) |

输入校验来源：[vision.py](../../../examples/runners/qwen3_5/vision.py) · [inputs.py](../../../examples/common/inputs.py) · [generation.py](../../../examples/common/generation.py) · [vision.py](../../../python/rpu_backend/adapters/qwen3_5/vision.py) · [cores.py](../../../python/rpu_backend/adapters/qwen3_5/cores.py)。

<!-- model-configurations-end -->
