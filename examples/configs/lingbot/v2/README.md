# LingBot v2 配置

[模型页及全部配置](../README.md#configurations) · [权重准备](../../WEIGHTS.md) · [参数说明](../../PARAMETERS.md)

<a id="配置矩阵"></a>

2 路和 3 路相机分别使用 [2cam/](2cam/) 与 [3cam/](3cam/)，各有 FP16、W8A16 两份配置，默认 8 核。文本长度通过参数调整，不按 token 数另建文件。

## 输入范围

`input.expected_language_tokens` 声明实际有效 token 数，默认 32；运行时必须与 observation 的 mask 计数一致。`input.max_language_tokens` 是物理上限，默认 72，可增至 160，满足 `1 ≤ 实际有效 token 数 ≤ max_language_tokens ≤ 160`。例如使用 128 个有效 token，两个字段都设为 128，并提供对应 observation；不能靠修改声明或补齐 token 改变输入身份。

每路图像预处理为 `[256,1536]`，grid 为 `[1,16,16]`；state 为 `[1,55]`，noise 为 `[1,50,55]`。相机数为 2 或 3，动作去噪固定 10 步、horizon 固定 50。

## 准备与运行

填写 `model.checkpoint` 与 `model.training_config`，从下面三种输入方式选一种：

- `input.observation`：准备好的 observation 文件。
- `input.case`：配套 case 目录，由加载器构造 observation。
- `input.images` 加 `input.instruction`：图像列表与实际提示词，由 processor 处理；`input.state` 为 1–55 个有限数字，处理后补零为 `[1,55]`；可选 `input.noise` 指向 `[1,50,55]` tensor 文件，不填则按 `run.seed` 生成。

三种方式不能混用。模板的路径环境变量可直接替换成实际路径；实际输入必须与相机和语言参数一致。

```bash
python examples/run.py --config examples/configs/lingbot/v2/2cam/fp16.toml --check-config
python examples/run.py --config examples/configs/lingbot/v2/2cam/fp16.toml
```

配置检查不加载 observation。实际相机数和有效 token 数在运行时检查。

<a id="input-files"></a>

## 输入文件准备

模板里的 `${RPU_MODEL_CACHE}`、`${LINGBOT2_TRAINING_CONFIG}`、`${LINGBOT2_2CAM_T32_OBSERVATION}` 等是环境变量占位，运行前需要自行导出，或直接改成实际路径；变量未定义时运行时报 `unresolved environment variable in path`，`--check-config` 不展开变量，不会提前发现。

| 方式 | 需要准备什么 | 获取方式 |
|---|---|---|
| `input.images` + `instruction` + `state`（推荐） | 2 或 3 张普通图片、指令文本、1–55 个状态值；可选 `noise` 文件 `[1,50,55]` | 不需要预先生成文件：入口把图片缩放到 256×256，用 checkpoint 的 processor 和 tokenizer 处理，不填 `noise` 时按 `run.seed` 生成。删除模板中的 `observation` 行，改写这三个字段 |
| `input.observation` | `.pt` 或 `.safetensors` 字典，键恰好为 `images` float32 `[C,256,1536]`、`image_grid_thw` int64 `[C,3]`（每行 `[1,16,16]`）、`img_masks` bool `[C]`、`lang_tokens` int64 / `lang_masks` bool `[1,L]`、`state` float32 `[1,55]`、`noise` float32 `[1,50,55]`；`pixel_values`、`initial_noise` 会自动改名为 `images`、`noise` | 仓库内没有生成脚本；用于复现已有 observation |
| `input.case` | 已有 CPU 参考 case 目录（含 `case.json` 及其登记的 input、golden 文件） | 仓库内没有生成脚本；用于对照已有 CPU 参考 |

三种方式只能选一种。`lang_masks` 的有效计数必须等于 `expected_language_tokens`。
