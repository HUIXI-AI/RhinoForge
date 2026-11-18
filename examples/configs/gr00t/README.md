# GR00T N1.7 Droid 配置

<a id="gr00t-配置"></a>

[配置总览](../README.md) · [全部配置](#configurations) · [核数支持矩阵](../../../docs/planner_core_topology.zh.md) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

当前模板覆盖 2 路或 3 路相机、224×224 或 256×256 图像，每组均提供 FP16、W8A16 和两种 W4A16（G32）：`w4a16_rtn_g32_vision_glue_adaln_w8a16` 的 Vision 为 W8，`w4a16_rtn_g32_glue_adaln_w8a16` 的 Vision 也是 W4；4/6/8 核默认模板均已上板实测（两种 W4A16 配方尚未完成正式数值认证，状态为 `IMPLEMENTED_UNVERIFIED_HW`）。选择模板时，`observation` 的相机数量、图像尺寸和文本 token 数必须一起匹配。

运行器同样支持 224×224 与 256×256 输入的单相机路径（预设 `1cam_*`）；使用单相机路径时仍须匹配 `observation` 并通过运行时容量检查，实测范围以性能报告为准。

<a id="配置矩阵"></a>

模板默认 3 路相机、32 个文本 token；2 路相机用 `--input 2cam_t128`（原两路相机模板的默认输入，2 路、128 个 token）或其他 `2cam_*` 预设（见 [vla.toml](../../inputs/vla.toml)），并用 `--set input.observation=<路径>` 给出对应 observation。修改 `input.text_tokens` 时，observation 也要匹配。

<a id="运行"></a>

## 准备与运行

准备 `gr00t-n1d7-3b` 权重和 `model.qwen3vl_registry_alias` 指定的 Qwen3-VL 资产，将 `input.observation` 替换为配套张量文件。模板设置 `action_horizon = 40`、`num_steps = 4`、`embodiment_id = 20`。其他输入参数和核心范围见[全部配置](#configurations)。

从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/gr00t/n1_7_droid/224/fp16.toml --check-config
python examples/run.py --config examples/configs/gr00t/n1_7_droid/224/fp16.toml
# 2 路相机、128 个文本 token
python examples/run.py --config examples/configs/gr00t/n1_7_droid/224/fp16.toml --input 2cam_t128 \
  --set input.observation=/path/to/gr00t_2cam_t128_observation.pt
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

仓库内没有公开的生成脚本。测试代码 `python/rpu_backend/tests/library/test_v5_e2e_perf.py` 中的 `Gr00tRunner._build_3img_inputs(state, embodiment_id, image_size=..., text_len=..., num_images=...)` 按上述模板构造同样的字典，可作参考：

1. 用 checkpoint 的 Qwen3-VL processor 处理 C 张图，并套用 chat 模板（`add_generation_prompt=True`）。
2. 指令按 token 截到恰好 `text_tokens` 个，再解码、重新编码，确认数量不变。
3. 224 图须给 image processor 传 `size={"shortest_edge": 224*224, "longest_edge": 224*224}`，否则 processor 会自行缩放，`image_grid_thw` 不再是 `[1,14,14]`。
4. 用 `torch.save` 保存；加载时用 `weights_only=True`，文件里只能有 tensor 和基本类型。

prefix 长度 = text_tokens + C×((E/2)²+2) + 8，`input.max_prefix_len` 须 16 对齐、覆盖它且不超过 512，例如 3 路 256 图、64 个 token 为 270，取 272。格式不符时运行时报 `GR00T observation must contain ... text_tokens instruction tokens in the Qwen user template`。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：GR00T。

### GR00T observation

| 配置 | 支持状态 | 精度 | 量化范围 | image_size 默认值 | 4 核 | 6 核 | 8 核 |
|---|---|---|---|---|---|---|---|
| [n1_7_droid/224/fp16.toml](n1_7_droid/224/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 224 | ● 已实测 | ● 已实测 | ● 已实测 |
| [n1_7_droid/224/w4a16_rtn_g32_glue_adaln_w8a16.toml](n1_7_droid/224/w4a16_rtn_g32_glue_adaln_w8a16.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | w4a16 | Vision ViT / 文本 backbone / VL encoder / DiT W4A16 G32 RTN；步间 glue / AdaLN W8A16；merger FP16 | 224 | ● 已实测 | ● 已实测 | ● 已实测 |
| [n1_7_droid/224/w4a16_rtn_g32_vision_glue_adaln_w8a16.toml](n1_7_droid/224/w4a16_rtn_g32_vision_glue_adaln_w8a16.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | w4a16 | Vision ViT / 步间 glue / AdaLN W8A16；文本 backbone / VL encoder / DiT W4A16 G32 RTN；merger FP16 | 224 | ● 已实测 | ● 已实测 | ● 已实测 |
| [n1_7_droid/224/w8a16.toml](n1_7_droid/224/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision ViT / 文本 backbone / VL encoder / DiT / 步间 glue / AdaLN W8A16；merger FP16 | 224 | ● 已实测 | ● 已实测 | ● 已实测 |
| [n1_7_droid/256/fp16.toml](n1_7_droid/256/fp16.toml) | 支持（`SUPPORTED`） | fp16 | — | 256 | ● 已实测 | ● 已实测 | ● 已实测 |
| [n1_7_droid/256/w4a16_rtn_g32_glue_adaln_w8a16.toml](n1_7_droid/256/w4a16_rtn_g32_glue_adaln_w8a16.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | w4a16 | Vision ViT / 文本 backbone / VL encoder / DiT W4A16 G32 RTN；步间 glue / AdaLN W8A16；merger FP16 | 256 | ● 已实测 | ● 已实测 | ● 已实测 |
| [n1_7_droid/256/w4a16_rtn_g32_vision_glue_adaln_w8a16.toml](n1_7_droid/256/w4a16_rtn_g32_vision_glue_adaln_w8a16.toml) | 已实现，未认证（`IMPLEMENTED_UNVERIFIED_HW`） | w4a16 | Vision ViT / 步间 glue / AdaLN W8A16；文本 backbone / VL encoder / DiT W4A16 G32 RTN；merger FP16 | 256 | ● 已实测 | ● 已实测 | ● 已实测 |
| [n1_7_droid/256/w8a16.toml](n1_7_droid/256/w8a16.toml) | 支持（`SUPPORTED`） | w8a16 | Vision ViT / 文本 backbone / VL encoder / DiT / 步间 glue / AdaLN W8A16；merger FP16 | 256 | ● 已实测 | ● 已实测 | ● 已实测 |

本组配置相同的项：

| 项目 | 值 |
|---|---|
| 输入类型 | 预处理图文 + state |
| text_tokens 默认值 | 32 |
| cameras 默认值 | 3 |
| num_steps 默认值 | 4 |
| action_horizon 默认值 | 40 |
| 文本可调范围 | text_tokens ∈ {32,64,128,160,192,224,256} |
| 图像/音频可调范围 | image_size=224或256；都允许1/2/3图（224 与 256 的 1 图 FP16 6/8 核、256 的 1 图 W8A16 6 核已在板上通过） |
| 生成可调范围 | num_steps=4或10；horizon=40 |
| 其他输入形状 | state[1,1,132]；pixels[C×(H/16)^2,1536]；grid每图[1,H/16,H/16] |
| 联动限制 | B=1；attention_mask全1；图片之后恰为指定T个instruction token。尺寸、相机与离散T还需通过原生容量检查。 |
| 默认核数 | 8 |
| CPU线程数 | 8 |
| 权重 | `gr00t-n1d7-3b` |
| 入口 | [run.py](../../run.py) |

输入校验来源：[config.py](../../../examples/runners/gr00t/config.py)。

<!-- model-configurations-end -->
