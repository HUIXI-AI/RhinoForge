# LingBot v2 配置

[模型页及全部配置](../README.md#configurations) · [权重准备](../../WEIGHTS.md) · [参数说明](../../PARAMETERS.md)

<a id="配置矩阵"></a>

[FP16](fp16.toml) 与 [W8A16](w8a16.toml) 两份配置默认 3 路相机、8 核，`rpu_execution.model.num_cores` 可改为 4 或 6（冷配置，各组件核数见配置文件注释）；2 路相机在命令行加 `--input 2cam_t32`（预设见 [vla_lingbot2.toml](../../../inputs/vla_lingbot2.toml)），并给出 2 路相机的输入。相机数和文本长度都通过参数调整，不另建文件。

## 输入范围

`input.expected_language_tokens` 声明实际有效 token 数，默认 32；运行时必须与 observation 的 mask 计数一致。`input.max_language_tokens` 是物理上限，默认 72，可增至 160，满足 `1 ≤ 实际有效 token 数 ≤ max_language_tokens ≤ 160`。例如使用 128 个有效 token，两个字段都设为 128，并提供对应 observation；不能靠修改声明或补齐 token 改变输入身份。

每路图像预处理为 `[256,1536]`，grid 为 `[1,16,16]`；state 为 `[1,55]`，noise 为 `[1,50,55]`。相机数为 2 或 3，动作去噪固定 10 步、horizon 固定 50。

## 准备与运行

填写 `model.checkpoint` 与 `model.training_config`（见[训练配置](#training-config)），从下面三种输入方式选一种：

- `input.observation`：准备好的 observation 文件。
- `input.case`：配套 case 目录，由加载器构造 observation。
- `input.images` 加 `input.instruction`：图像列表与实际提示词，由 processor 处理；`input.state` 为 1–55 个有限数字，处理后补零为 `[1,55]`；可选 `input.noise` 指向 `[1,50,55]` tensor 文件，不填则按 `run.seed` 生成。

三种方式不能混用。模板的路径环境变量可直接替换成实际路径；实际输入必须与相机和语言参数一致。

```bash
python examples/run.py --config examples/configs/lingbot/v2/fp16.toml --check-config
python examples/run.py --config examples/configs/lingbot/v2/fp16.toml
# 2 路相机
python examples/run.py --config examples/configs/lingbot/v2/fp16.toml --input 2cam_t32 \
  --set input.observation=/path/to/lingbot2_2cam_t32_observation.pt
```

配置检查不加载 observation。实际相机数和有效 token 数在运行时检查。

<a id="training-config"></a>

### 训练配置（`model.training_config`）

必填，指向 LingBot v2 上游训练用的 YAML 文件。公开 checkpoint `lingbot-vla-v2-6b` 不带这个文件，入口也不会到 checkpoint 目录里查找。入口只读取文件顶层的 `train:` 段，缺少该段时报 `KeyError: 'train'`。

| `train` 中的字段 | 取值与作用 |
|---|---|
| `align_params` | 非空时启用对齐分支，prefix 多 16 行 task query；省略或为空时关闭。`mode` 只能是 `query`（省略时为 `query`） |
| `token_top_k`、`token_num_experts`、`chunk_size` | 省略，或分别等于 4、32、50，其他值报错 |
| `token_moe_layers` | 省略，或为 0–35 全部 36 层，其他值报错 |
| `vlm_causal` | 布尔，省略时为 `true` |

prefix 行数 = 每路相机 66 行 +（对齐开启时 16 行）+ T：对齐关闭时 2 路相机 132+T、3 路 198+T；对齐开启时 2 路 148+T、3 路 214+T。模板注释里的可行长度表是用对齐关闭的文件在板上测的。

用过的两种文件：

- 上游仓库 `Robbyant/lingbot-vla-v2` 的 `configs/vla/real_robot/real_robot.yaml`：对齐开启，2026-10-09 文本矩阵用的是它。
- 只含 `train:` 几何字段、不含 `align_params` 的文件：对齐关闭，模板板测使用的是仅保留上述几何字段的配置。

`report.json` 的 `training_config_sha256` 与 `alignment` 记录实际使用的文件和对齐状态。对齐开关会改变 prefix 内容，换文件后的结果不能直接与另一种文件的记录比较。本仓库没有关于哪一种更适合某个机器人任务的结论。

<a id="input-files"></a>

## 输入文件准备

模板里的 `${RPU_MODEL_CACHE}`、`${LINGBOT2_TRAINING_CONFIG}`、`${LINGBOT2_2CAM_T32_OBSERVATION}` 等是环境变量占位，运行前需要自行导出，或直接改成实际路径；变量未定义时运行时报 `unresolved environment variable in path`，`--check-config` 不展开变量，不会提前发现。

| 方式 | 需要准备什么 | 获取方式 |
|---|---|---|
| `input.images` + `instruction` + `state`（推荐） | 2 或 3 张普通图片、指令文本、1–55 个状态值；可选 `noise` 文件 `[1,50,55]` | 不需要预先生成文件：入口把图片缩放到 256×256，用 checkpoint 的 processor 和 tokenizer 处理，不填 `noise` 时按 `run.seed` 生成。删除模板中的 `observation` 行，改写这三个字段 |
| `input.observation` | `.pt` 或 `.safetensors` 字典，键恰好为 `images` float32 `[C,256,1536]`、`image_grid_thw` int64 `[C,3]`（每行 `[1,16,16]`）、`img_masks` bool `[C]`、`lang_tokens` int64 / `lang_masks` bool `[1,L]`、`state` float32 `[1,55]`、`noise` float32 `[1,50,55]`；`pixel_values`、`initial_noise` 会自动改名为 `images`、`noise` | 仓库内没有生成脚本；用于复现已有 observation |
| `input.case` | 已有 CPU 参考 case 目录（含 `case.json` 及其登记的 input、golden 文件） | 仓库内没有生成脚本；用于对照已有 CPU 参考 |

三种方式只能选一种。`lang_masks` 的有效计数必须等于 `expected_language_tokens`。三种方式都需要 `model.checkpoint` 和 `model.training_config`。换相机数或指令长度（如 3 路、T64）时，相机数用模板默认值（2 路用 `--input 2cam_t32`），把 `expected_language_tokens` 设为 64（不超过 72，不需要改上限），推荐用 `images` 方式；observation 方式须按上表重新准备张量。
