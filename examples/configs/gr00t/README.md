# GR00T N1.7 Droid 配置

<a id="gr00t-配置"></a>

[配置总览](../README.md) · [全部配置](#configurations) · [核数支持矩阵](../../../docs/planner_core_topology.zh.md) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

当前模板覆盖 2 路或 3 路相机、224×224 或 256×256 图像，每组均提供 FP16 和 W8A16。选择模板时，`observation` 的相机数量、图像尺寸和文本 token 数必须一起匹配。

运行器还保留 256×256 输入的单相机兼容路径；224×224 仅允许 2/3 路相机。使用兼容路径时仍须匹配 `observation` 并通过运行时容量检查，实测范围以性能报告为准。

<a id="配置矩阵"></a>

两路相机模板的默认文本长度为 128，三路为 32；修改 `input.text_tokens` 时，observation 也要匹配。

<a id="运行"></a>

## 准备与运行

准备 `gr00t-n1d7-3b` 权重和 `model.qwen3vl_registry_alias` 指定的 Qwen3-VL 资产，将 `input.observation` 替换为配套张量文件。模板设置 `action_horizon = 40`、`num_steps = 4`、`embodiment_id = 20`。其他输入参数和核心范围见[全部配置](#configurations)。

从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/gr00t/n1_7_droid/224/2cam/fp16.toml --check-config
python examples/run.py --config examples/configs/gr00t/n1_7_droid/224/2cam/fp16.toml
```

更改 `text_tokens` 时需同步更换 observation；不能只改数值复用原张量。配置检查不会加载该文件，运行结果及测量范围见GR00T 性能记录。

<a id="input-files"></a>

## 输入文件准备

`input.observation` 是 `torch.save` 保存的 tensor 字典（可以外包一层 `{"inputs": ..., "meta": ...}`）。设 C 为 `cameras`、E 为 `image_size / 16`：

| 键 | 形状与类型 | 说明 |
|---|---|---|
| `pixel_values` | 浮点 `[C·E², 1536]` | Qwen3-VL processor 的输出；224 图为 `[C·196,1536]`，256 图为 `[C·256,1536]` |
| `image_grid_thw` | `[C,3]`，每行 `[1,E,E]` | 与 `cameras`、`image_size` 一致 |
| `input_ids` | int32 / int64 `[1,L]` | 必须是 Qwen 用户模板：`im_start` + `user` 换行，C 段图像占位，**恰好 `text_tokens` 个指令 token**，再接 `im_end` 和 `assistant` 起始段（特殊 token 均为 Qwen tokenizer 的标准 token） |
| `attention_mask` | 与 `input_ids` 同形状，全为 1 | 不允许补齐 |
| `state` | 浮点 `[1,1,132]`，有限值 | 机器人状态 |
| `mm_token_type_ids`、`embodiment_id`（可选） | — | 若提供，须分别等于图像占位位置和 `input.embodiment_id` |

仓库内没有能生成上述文件的公开脚本。需要用 Qwen3-VL processor 按以上模板自行构造，并保证指令恰好是 `text_tokens` 个 token；格式不符时运行时报 `GR00T observation must contain ... text_tokens instruction tokens in the Qwen user template`。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：GR00T。

### GR00T observation

| 配置 | 支持状态 | 精度 | 量化范围 | text_tokens 默认值 | cameras 默认值 | image_size 默认值 | 4 核 | 6 核 | 8 核 |
|---|---|---|---|---|---|---|---|---|---|
| [n1_7_droid/224/2cam/fp16.toml](n1_7_droid/224/2cam/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 128 | 2 | 224 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [n1_7_droid/224/2cam/w8a16.toml](n1_7_droid/224/2cam/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision ViT、文本 backbone、VL encoder、DiT 投影与步间 glue W8；merger、AdaLN、归一化保持 FP16 | 128 | 2 | 224 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [n1_7_droid/224/fp16.toml](n1_7_droid/224/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 32 | 3 | 224 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [n1_7_droid/224/w8a16.toml](n1_7_droid/224/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision ViT、文本 backbone、VL encoder、DiT 投影与步间 glue W8；merger、AdaLN、归一化保持 FP16 | 32 | 3 | 224 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [n1_7_droid/256/2cam/fp16.toml](n1_7_droid/256/2cam/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 128 | 2 | 256 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [n1_7_droid/256/2cam/w8a16.toml](n1_7_droid/256/2cam/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision ViT、文本 backbone、VL encoder、DiT 投影与步间 glue W8；merger、AdaLN、归一化保持 FP16 | 128 | 2 | 256 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [n1_7_droid/256/fp16.toml](n1_7_droid/256/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 32 | 3 | 256 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |
| [n1_7_droid/256/w8a16.toml](n1_7_droid/256/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision ViT、文本 backbone、VL encoder、DiT 投影与步间 glue W8；merger、AdaLN、归一化保持 FP16 | 32 | 3 | 256 | ✕ 不支持 | ✕ 不支持 | ● 已实测 |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 预处理图文 + state |
| num_steps 默认值 | 4 |
| action_horizon 默认值 | 40 |
| 文本可调范围 | text_tokens ∈ {32,64,128,160,192,224,256} |
| 图像/音频可调范围 | image_size=224或256；224允许2/3图，256允许1/2/3图 |
| 生成可调范围 | num_steps=4或10；horizon=40 |
| 其他输入形状 | state[1,1,132]；pixels[C×(H/16)^2,1536]；grid每图[1,H/16,H/16] |
| 联动限制 | B=1；attention_mask全1；图片之后恰为指定T个instruction token。尺寸、相机与离散T还需通过原生容量检查。 |
| 默认核数 | 8 |
| CPU线程数 | 8 |
| 权重 | `gr00t-n1d7-3b` |
| 入口 | [run.py](../../run.py) |

- ✕ 4/6 核不支持：原生实现只有 8 核：DiT 的时间表、常驻 FF1 kernel 和交叉注意力 KV 都按 8 核布局。

输入校验来源：[config.py](../../../examples/runners/gr00t/config.py)。

<!-- model-configurations-end -->
