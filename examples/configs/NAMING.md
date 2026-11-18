# 配置命名

[模型目录](README.md) · [配置总览](README.md) · [参数说明](PARAMETERS.md) · [路径对照](config_names.json)

## 目录与文件名

目录按模型族、版本或任务、模型大小、输入规格分组，省略不需要的层级。例如 `qwen3/text/4b/`、`wall_oss/base/3cam/224/`。尺寸和相机规格已由目录说明，文件名只写同组配置之间的区别。

文件名使用小写字母、数字和下划线，顺序为：

```text
<精度与模块范围>[_量化方法][_g分组大小][_独立执行配方].toml
```

开头的精度与模块范围就是 TOML 中 `precision` 写的精度配方名（登记在 [manifests/precision_recipes.v1.json](../../manifests/precision_recipes.v1.json)）。文件名等于配方名，或是配方名加 `_<执行变体>`，如 `fp16_instruct.toml`、`w8a16_pipeline_high_fused.toml`；`scripts/check_examples.py` 拒绝文件名与 `precision` 不符的模板。

只写区分配置所需的部分。`fp16.toml` 是浮点基准；存在多个量化范围时写明 `text`、`mlp`、`experts`、`action` 或 `pipeline`。混合精度组合按组件写明，不用笼统的 `mixed` 代替已有明确含义。

| 名称片段 | 含义 |
|---|---|
| `w8a16` / `w4a16` | 所选投影的 INT8 / INT4 权重，FP16 激活 |
| `text` / `head` / `vision` | 文本投影 / 输出头 / 视觉投影；具体层范围见模型页。Qwen3、Qwen3-VL、Qwen3.5 的配方带 `head` 时量化输出头，不带时输出头保持 FP16；Qwen3-ASR/TTS 不按此命名，输出头精度见其量化范围 |
| `mlp` / `experts` | MLP 投影 / MoE 路由专家，不等于全部文本模块 |
| `nvfp4_text` | Qwen3、Qwen3-VL 文本的七个解码投影使用 NVFP4；Qwen3.5 dense 文本的 Attention、GDN、MLP 投影使用 NVFP4，GDN `in_proj_a`/`in_proj_b` 保持 FP16；输出头、嵌入、归一化和 Vision 均保持 FP16 |
| `action_nvfp4` | 动作组件中指定投影使用 NVFP4，范围见该配方 |
| `fp8_e4m3` | 指定投影使用 FP8 E4M3，对应权重页的 mixed E4M3 配方 |
| `attn` / `gdn` | Attention / Gated DeltaNet；具体投影范围见模型页 |
| `rtn` / `awq` / `gptq` | 量化方法或所读取量化包的方法 |
| `g32` / `g128` | 量化分组大小；`channel` 表示逐输出通道 |
| `high_fused_vector_qk` | RhinoVLA 的高精度非线性、融合及 vector Q/K 归一化组合 |

核心数、累加精度、chunk、P/D、图像或文本长度通过 TOML 字段调整，不用这些参数复制配置文件。独立量化范围、权重布局或执行入口才保留不同文件；复现历史测量时还须恢复原输入与计时参数。

<a id="camera-image-size-exceptions"></a>

### 按相机数或图像尺寸分开的模板（例外）

下列 VLA 仍按相机数或图像尺寸分目录保存模板。修改这些维度时，直接选用对应目录的模板，不要只改 `cameras` 或 `image_size`。

| 模型 | 分组 | 组间差别（除相机数或尺寸本身外） |
|---|---|---|
| Wall-OSS Base | `2cam` / `3cam`，`224` / `256` | 224 用 `vision_window`，256 用 `vision_pair`；部分组合的 `padding_rows` 和 `[cpu].threads` 不同，`[cpu].threads` 按相机数分别调过，所以相机数也不合并 |
| GR00T | `224` / `256` | 只差 `image_size` 和 `observation` 文件；输入预设还不含图像尺寸，两种尺寸的输入范围也不同（256 另允许 1 路相机），暂按尺寸分目录 |

Pi0.5、G0.5、HyVLA、LingBot v2 和 GR00T 的 2 相机模板与 3 相机模板只差输入字段（相机数、文本长度和固定输入文件，G0.5 另有 `camera_names`），已合并为默认 3 相机的同名模板。2 相机用 `--input 2cam_*` 预设（见 [examples/inputs](../inputs/)），需要输入文件的族同时用 `--set input.<键>=<路径>` 给出该形状的文件。

[config_names.json](config_names.json) 的 `paths` 记录改名，`merged` 记录并入另一模板的旧路径：`config` 为现在的模板，`input` 为旧模板默认输入对应的预设（`null` 表示旧模板的默认输入就是新模板的默认输入）。旧路径下的板上记录按这张表归到新模板的对应输入格，支持状态和性能按（模板、核数、输入）绑定。

## 示例

| 配置 | 如何阅读 |
|---|---|
| `qwen3/text/4b/w4a16_text_rtn_g32.toml` | 4B 纯文本，文本主干 RTN W4/G32，输出头保持 FP16 |
| `qwen3/text/4b/w4a16_text_head_rtn_g32.toml` | 同时量化文本主干与输出头 |
| `qwen3_5/text/2b/w4a16_mlp_awq_g128.toml` | 2B 纯文本，指定 MLP 范围使用 AWQ/G128 |
| `qwen3_5/text/2b/w4a16_mlp_rtn_g32_attn_gdn_w8a16.toml` | MLP 使用 RTN W4/G32，Attention/GDN 使用逐输出通道 W8；G32 只属于 MLP |
| `qwen3_5/text/35b_a3b/w4a16_experts_gptq_g128.toml` | MoE 路由专家使用 GPTQ/G128 |
| `qwen3_vl/vl/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml` | 文本主干 AWQ、输出头 RTN，均为 W4/G32；Vision W8 |
| `gr00t/n1_7_droid/256/fp16.toml` | N1.7 DROID、256×256 输入、FP16 |
| `internvla/navdp/w8a16_system2.toml` | NavDP 中 System2 的 decoder 使用 W8，其余组件精度见模型页 |

目录（`catalog.json`）的 `recipe` 列是配方名；精度标签和“量化范围”一列由登记表展开，不另行手写。精度标签为 `fp16`、`fp8_e4m3`、`nvfp4`、`w8a16`、`w4a16`、`w8a16_action_nvfp4`、`w8a16_action_w4a16`、`w8a16_prefill_w8a8_action_nvfp4`、`w8a16_vision_w4a16_action_nvfp4`；量化范围列出各组件的格式、分组和方法，以及按投影或层的例外。

每个配方还登记运行器使用的精度写法（如 G0.5 的 `w8a16_nvfp4`、Wall-OSS 的 `w4a16-pgrp`、Qwen3.5 MoE 的 `mixed-e4m3`）、可绑定的 profile、profile 须一致的限制，以及由配方设置的冷启动开关（RhinoVLA v3 的 W8A16 开关）。这些写法不写进 TOML；`precision` 写运行器写法或未登记的名称都会被拒绝。新的混合精度先在登记表加配方（须有对应的实现 owner），不在 TOML 里逐模块组合。权重目录、registry alias 和 profile ID 仍用各自的登记名称，不能从文件名推导。比较性能时，还要核对实际输入和计时条件。

`controlled/` 表示带固定冷启动开关组合的入口（如 RhinoVLA v3 源码入口，也可交给 `validate_v3.py` 校验）；`legacy/` 表示保留独立资产、输入或调用约定的配置。

## 定位已有配置记录

[路径对照](config_names.json)按仓库根目录列出原路径到当前路径的对应关系。原始测量 JSON/CSV 保留当时的文件名、哈希和配置值；查找运行模板时使用当前路径。
