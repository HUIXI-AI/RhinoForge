# Runtime 配置

简体中文 | [English](runtime_config.md)

> 英文版是机器校验和配置契约的事实源。中文版完整翻译用户说明；变量名、默认值、
> accepted grammar、读取阶段、作用域与风险必须与英文版保持一致。

RhinoForge 提供少量面向用户的设置，以及数量更多的精确模型配置和诊断控制项。
请在导入 `rpu_backend` 前设置完整环境；除非某一行明确说明按调用读取，否则不支持在
存活进程中更改配置。

本页记录 `python/rpu_backend` 和 `src` 中当前保留的所有环境变量读取项，以及 Launch
容量输入。变量列在这里并不代表包含该变量的实验性组合已获支持。

## 如何阅读表格

解析器缩写：

- **PB(default)** 表示共享 Python 布尔解析器。它会去除首尾空白，不区分大小写地接受
  `1`/`true`/`on` 和 `0`/`false`/`off`/空值；其他值会引发 `ValueError`。
- **B01(default)** 表示 Python 和原生代码都会读取的布尔值。只有字面值 `0` 和 `1`
  具有可移植性，并被 Python 镜像接受。
- **E1** 表示未设置时关闭，只有精确字面值 `1` 会启用该路径。
- **N1** 表示未设置时关闭，精确值 `1` 或小写 `true` 会启用原生路径。使用不同原生
  解析器的行会单独说明。

读取/生命周期缩写：

- **IMPORT**：包或原生扩展初始化；更改后需要启动新进程。
- **NATIVE**：首次使用原生功能时缓存；更改后需要启动新进程。
- **MODEL**：构造模型、policy 或权重时绑定；需要创建新模型。由于 RhinoForge 每个
  进程只允许一个存活的融合 policy，启动新进程是安全的替换方式。
- **BUILD**：构建 handle 或 Graph 时绑定；应清除并重建所有受影响的 Graph 及其所属
  模型，或使用新进程。
- **CALL**：每次适用的调用都会读取。如果变化会改变执行形状或 Graph 拓扑，该行仍会
  要求重建 Graph。

即使文档说明解析器接受更多形式，部署 manifest 也应使用 `0`/`1`。下文的 facade
默认值只由指定的公共 facade 应用；直接使用 adapter 时采用所列源码原始默认值。

## 优先级与可复现配置

使用 `examples/run_model.py` 时，`[runner.env]` 中每个 allowlist 值都会在导入 `torch`
或 `rpu_backend` 前覆盖继承环境中的同名值。表中省略的变量保留 shell 中的值。随后，
模型 facade 使用 `setdefault` 设置其负责的默认值，因此进程环境中的显式值仍优先；
直接使用 adapter 时采用下文记录的源码原始默认值。

部署负责的路径和凭据不应写入 TOML。尤其应在经过验证的部署环境中设置
`RPU_KERNEL_LIB_PATH`，并通过 loader 的显式 `rpu_execution` 参数传入冷态、按 handle
生效的规划。模型 preflight 始终具有最终决定权，并可能拒绝超出精确配置范围的环境值
或执行设置。

为了可复现运行，请记录 TOML 哈希，以及每个显式设置的 runtime 变量的有效值。更改
IMPORT、NATIVE、MODEL 或 BUILD 设置时应使用新进程；不要认为在构造完成后修改
`os.environ` 会重新配置已有 handle 或 Graph。

## TOML 参数目录

[配置索引](../examples/configs/README.md) 列出全部当前模板路径。每个模型只使用一套目录：
Qwen3 在 `qwen3/text`，Qwen3-VL 分为 `text/vision/vl`，Qwen3.5 分为 `text/vl/legacy`
（legacy 是独立 checkpoint 格式）。LingBot 只保留 `v2`，RhinoVLA 按 `v1/v3`，
Wall-OSS 使用 `base`，Pi0.5 按 `2cam/3cam` 且只提供八核配置。根目录不放散落的模型 TOML。

### 模型与策略配置

| 表 | 字段 | 作用 |
|---|---|---|
| `[example]` | `profile_id`、`target`、`registry_alias`、`dtype`、可选 `architecture` | 选择精确配置身份，实际 checkpoint 元数据仍由加载器检查。 |
| `[example.opt_in]` | profile 规定的环境值 | 导入 backend 前设置，拒绝继承环境中的冲突值；必须匹配所选 profile。 |
| `[input]` | `checkpoint` 及模型对应的 prompt/images/tensor、生成或策略选项 | 显式 checkpoint 优先于 alias。请求与输入文件由调用方提供，见各模型 README。 |
| `[run]` | `warmup`、`runs`、`seed`、`torch_num_threads`、`inference_mode`、`output_dir` | 控制重复推理和主机设置；权重加载与 Graph 准备在计时样本之前。 |
| `[run]` | `profile`，以及入口支持的 `timing_mode`、`summary_statistic`、`warmup_decode_steps` | Torch profiler 单独执行一次诊断调用；生成计时字段按模型检查。 |
| `[rpu_execution.model]` | `num_cores` | 冷态计算预算，由模型/profile 准入；Pi 示例固定八核。 |
| `[rpu_execution.<stage>]` | `chunk_size`、支持的布尔控制和 prefill padding 字段 | 冷态按 handle 规划；构造器检查实际 stage、字段和形状。 |
| `[rpu_execution.components.<component>.<stage>]` | 支持的逐组件设置 | 复合模型覆盖项；组件 ID 与字段在安装前检查。 |

`example.profile_id` 对应 [profiles.json](../examples/configs/profiles.json)，只包含当前模板
实际引用的身份和约束，用于绑定精度、组件和 opt-in，不代表模型质量或性能认证。
Qwen 的 `input.decode_steps` 是 prefill 首 token 之后的 decode 调用数；
`input.prefill_tokens` 是显式构造的输入长度，不能当作旧文件的 `max_new_tokens` 使用。

Pi0.5 使用 `[pi05].precision`，其 `[input]` 包含 `checkpoint`、`batch`、`cameras`、
`text_tokens` 和 `num_steps`。runner 绑定对应公开优化 profile 并检查 paired prefill
几何，见 [Pi 配置](../examples/configs/pi05/README.md)。

RhinoVLA 的 `input.runtime_factory`、`factory_config`、`request` 分别指定调用方信任的实现、
JSON factory 配置和 tensor-only 请求。模型字段由所选版本的 factory 定义；RhinoForge
绑定 checkpoint、step 数和冷态执行设置，见 [RhinoVLA 配置](../examples/configs/rhinovla/README.md)。

### 独立集成模板与调用方自有文件

G0.5、GR00T、NavDP、SigLIP 使用各自的 `[runner/model/request]` 集成模板。调用方已有的
同格式文件仍可运行，仓库不为同一模型额外保存一份兼容模板。

| 表 | 字段 | 作用 |
|---|---|---|
| `[runner]` | `target` | 选择独立集成入口。 |
| `[runner.env]` | 本页记录的非诊断变量，不包含运行时资产路径或凭据 | 在导入 Torch/backend 前设置，省略项继续继承环境。 |
| `[runner.torch_profile]` | `enabled`、`output`、`record_shapes`、`profile_memory`、`with_stack` | 整条命令的 Torch trace。 |
| `[model]`、`[request]`、`[generation]` | 所选直接 example 接受的字段 | 模型位置和调用方输入。 |
| `[runtime]` | 调用方旧 RhinoVLA 文件中由可信 factory 定义的字段 | 经公开 factory 合同传递。 |

不要把 `[input]/[run]` 混进 `[runner/model/request]` 文件，检查时会拒绝。
`--check-config` 不导入 Torch 或访问板卡，只检查结构和冷态约束，不检查本地权重或认证请求。

`padding_rows` 为 `"auto"` 或非负整数；精确值不能与 `padding_budget` 同时启用。
chunk size 为 `"auto"` 或正 16 倍数，并受各 loader 的实际 profile 进一步限制。
完整冷态规划见[非环境变量 runtime 控制](#非环境变量-runtime-控制)。

## Host 执行设置

统一 examples runner 在 `no_grad()` 和 `inference_mode(False)` 下准备所有模型，
使持久权重和可变 Graph 输入保留版本计数；`run.inference_mode` 仍控制 warmup
和计时推理。延迟安装组件也需要局部使用相同的准备上下文。依赖 tensor 版本的缓存
遇到 inference tensor 时，应刷新派生数据或比较独立保存的内容；不能把同一个
tensor 对象当作内容未变的证明。

这些参数属于应用进程及其安装的 CPU runtime，不属于 RPU planner。标准 `[pi05]`
配置通过 `run_model.py` 或 `pi05.py` 运行时，都会在导入 Torch 前绑定下列 allocator
值，`run.torch_num_threads` 默认八线程。库内推理不会修改这些进程设置；更改后使用
新进程。旧版直接 example 格式不会代替统一 runner 应用 `[runner.env]`。

| Variable | Pi 建议值 | 读取 / 更改 | 作用与影响 |
|---|---:|---|---|
| `OMP_NUM_THREADS` | `8` | CPU runtime 初始化 / **IMPORT** | 限制 OpenMP host 工作线程，实际效果取决于 Torch 构建；显式 Torch 线程设置优先。 |
| `MKL_NUM_THREADS` | `8` | CPU runtime 初始化 / **IMPORT** | 安装的 CPU runtime 使用 MKL 时限制其线程数。 |
| `MIMALLOC_PURGE_DELAY` | `1000` | mimalloc 初始化 / **IMPORT** | 对齐上游 Pi host 设置；在含 mimalloc 的构建中延迟释放空闲页，可能增加保留内存。 |
| `MIMALLOC_ARENA_PURGE_MULT` | `0` | mimalloc 初始化 / **IMPORT** | 对齐上游 Pi host 设置；调整 arena purge 策略，具体行为取决于 runtime 版本。 |

Qwen3、Qwen3-VL、Qwen3.5 和 Pi0.5 接受冷态 `model.num_cores` 与分组件
`linear_acc32`。核数是整数 `4/6/8`，实际准入仍由 checkpoint、量化与输入 profile
收窄；Pi 优化 profile 和量化 Qwen 配置保持八核约束。`linear_acc32` 只接受布尔值，
`false`（默认）选择 ACC16，`true` 选择 ACC32。Qwen3、Qwen3.5 文本入口在 `prefill`
设置，图文入口还可在 `vision` 设置，Pi 分别在 `prefill/vision/action` 设置。
Qwen3-VL 另接受冷态布尔值 `prefill.fast_replay`，不接受 `vision.fast_replay`
或独立 `decode` 表。普通 FP16 和量化 API 入口省略时默认 `false`；适用的 FP16
和量化示例模板显式设为 `true`。专用 profile 策略及原生 owner/Graph 检查仍具最终决定权。配置在安装权重前绑定；改变后应关闭模型、重新加载。

## 通用和 cache 设置

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_KERNEL_LIB_PATH` | 扩展旁的 combined operator asset；现有文件路径及相邻 `.kernels` manifest | 首次访问 asset / **IMPORT** | 进程级 asset 选择。资产或 manifest 缺失、不兼容或不受信任都会阻止安全执行。 |
| `RPU_MODEL_CACHE` | `~/.cache/rhinoforge/models`；目录路径 | 为 HF 默认值执行包 bootstrap / **IMPORT**，alias 解析 / **CALL** | RhinoForge 模型 alias 的根目录。显式值也会提供 HF cache 默认值；导入后修改会影响之后的 alias，但无法可靠地重新配置已导入的 HF 组件。 |
| `RPU_LOG_LEVEL` | `3`；十进制整数 `0..5`；无效值会告警并回退到 `3` | 原生扩展加载 / **IMPORT** | 进程日志级别：`0` 静默，最高 `5` trace。较高级别会增加输出，并可能暴露路径或请求 metadata。 |
| `RPU_WARMUP` | `0`；整数；无效值变为 `0`，负数收窄为 `0` | Adapter 构造 / **MODEL** | 支持 warmup 的 adapter 的 warmup forward 次数。会增加启动工作，并可能消耗诊断预算。 |
| `RPU_CAUSAL_PREFILL_PADDING_BUDGET` | `64`；非负十进制整数 | 符合条件的 causal prefill 规划 / **CALL**；新 plan 需重建 Graph | 通用 Qwen3/Llama planner 可选 padding 的最大行数。`0` 禁用可选 padding；plan 变化会改变 Graph signature。 |
| `LKN_MAX_BATCH_ENTRIES` | `65536`；正十进制数；无效/空/零使用默认值；上限 `4194304` | Launch 初始化与 Graph 规划 / **IMPORT** | batch submission 的进程容量。过小会拒绝 Graph；过大则预留更多 host 内存。 |
| `LKN_KD_BUF_MB` | `8`；正 MiB；无效/空/零使用默认值；上限 `256` | Launch 初始化与 Graph 规划 / **IMPORT** | 进程 command-buffer 容量。过小会拒绝 Graph；过大则增加内存占用。 |
| `LKN_INSTR_BUF_MB` | `64`；正 MiB；无效/空/零使用默认值；上限 `1024` | Launch 初始化与 Graph 规划 / **IMPORT** | 进程 execution-buffer 容量。过小会拒绝 Graph；过大则增加内存占用。 |
| `HF_HOME` | Hugging Face 默认值；目录路径 | 包 bootstrap 与 HF 初始化 / **IMPORT** | 标准 HF cache 根目录。`RPU_MODEL_CACHE` 只通过 `setdefault` 提供它；调用者值优先。 |
| `HF_HUB_CACHE` | Hugging Face 默认值；目录路径 | 包 bootstrap 与 HF 初始化 / **IMPORT** | 标准 Hub cache。仅在未设置时，显式 `RPU_MODEL_CACHE` 才提供 `<HF_HOME>/hub`。 |
| `HUGGINGFACE_HUB_CACHE` | Hugging Face 默认值；目录路径 | 包 bootstrap 与 HF 初始化 / **IMPORT** | 兼容性 Hub cache 变量，处理方式与 `HF_HUB_CACHE` 相同；两者应保持一致。 |

导入 backend 前应设置全部三个 `LKN_*` 值。facade 可在首次导入 adapter 前选择更大的
默认值，但无法调整已经加载的 Launch runtime。增大容量不会让不支持的模型配置获得支持。

<a id="model-specific-public-settings"></a>
## 模型特定公共设置

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_QWEN3_SPM_KV_BY_MHA` | `1`；仅接受精确 `0` 或 `1` | Qwen3 fused handle 构造 / **MODEL** | 允许已认证的短 prefill 配置在精确 plan 可容纳时将临时 K/V 保留在 SPM。`0` 固定使用 DDR cache attention；不符合条件的 shape 会自动使用 DDR。 |
| `QWEN3_5_QUANT_ALLOW_UNCERTIFIED` | `0`；只有精确字符串 `1` 启用 | Qwen3.5 量化 preflight / **MODEL** | 允许未认证的 W8/W4 受控评估；不会跳过 checkpoint 格式、投影范围或数值检查。 |
| `QWEN3_5_TEXT_CHUNK` | `0` auto；十进制 `0` 或至少为 `64` 的整数 | Qwen3.5 text 安装 / **MODEL** | 冷态、按 handle 生效的 prefill 上限。无效值或更小的正值会失败；新集成使用 `rpu_execution`。 |
| `QWEN3_5_TEXT_PADDING_BUDGET` | `64`；十进制整数 `0..64` | Qwen3.5 text 安装 / **MODEL** | 冷态可选 padding budget。它会改变 prefill plan 和 Graph signature。 |
| `RPU_QWEN3_5_FREE_HF_WEIGHTS` | **PB(true)** | Qwen3.5 权重安装 / **MODEL** | 已存在转换副本后，`1` 释放原始 HF 权重；`0` 保留并增加 host 内存。 |
| `RPU_PI05_SIGLIP_W8A16` | int8/uint8 bundle 时为 **PB(true)**，否则为 **PB(false)** | Pi0.5 SigLIP 安装 / **MODEL** | 选择现有量化 projection 路径。必须匹配 checkpoint dtype 和数值验证。 |
| `RPU_PI05_SIGLIP_W8A16_SCOPE` | `all`；`all`/`full`、`row`/`safe`、`out`/`out_proj`/`o`、`fc2`/`mlp.fc2`，或 `none`/`0`/`off`/`false`/空 | Pi0.5 SigLIP 安装 / **MODEL** | 选择量化 projection 分组。未知文本会失败；每种 scope 都是独立数值配置。 |
| `RPU_PI05_ADARMS_DENSE_W8A16` | 量化 expert 时为 **PB(true)**，否则为 **PB(false)** | Pi0.5 AdaRMS 安装 / **MODEL** | 选择量化 AdaRMS dense 权重。与加载配置不匹配会失败或改变数值。 |

chunk 和 padding 选择应优先使用公共 loader 或 policy 的 `rpu_execution` 参数。该参数会在
加载权重前校验，并在绑定后保持不可变。

## Fail-closed 评估门

这些 gate 默认关闭，并且只接受精确字面值 `1`。它们在 preflight 或构造期间读取
（**MODEL**），更改时需要新模型/新进程。启用 gate 只会允许受控评估路径；不会认证该
路径、放宽数值门禁或扩大模型支持矩阵。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED` | `0`；**E1** | Qwen3.5/G0.5 Vision preflight / **MODEL** | 启用 numeric-blocked Qwen3.5 Vision 2B/4B 和通用 G0.5 Vision 的受控评估。它不认证图像输出。 |
| `QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED` | `0`；**E1** | Preflight / **MODEL** | 为兼容保留旧名称，仅用于精确 Source-only Qwen3-VL 32B W8A16 候选。它要求匹配的预留 buffer runtime 和未修改的精确 LKN 默认值；不认证数值、任务、性能或发布 runtime 门。 |
| `RPU_LINGBOT2_ALLOW_UNVALIDATED` | `0`；**E1** | Preflight / **MODEL** | LingBot-VLA-V2 受控配置；输出没有机器人认证。 |
| `RPU_INTERNVLA_N1_ALLOW_NUMERIC_BLOCKED` | `0`；**E1** | Preflight / **MODEL** | InternVLA-N1 legacy 受控评估。 |
| `RPU_S2_SDPA_BF16` | 只接受未设置/空/`0`；字面值 `1` 和其他所有值都会被拒绝 | Preflight / **MODEL** | InternVLA policy 对不可用 asset 的 tripwire；保持关闭。 |

## 共享高级执行 selector

这些 selector 由配置负责。facade 可以在模型构造前设置经过验证的值；环境覆盖可能破坏
Graph、内存或数值契约。

### Collective 执行

Residual reduction 不再提供公开 route、chunk、ring 或 two-stage selector。共享 wrapper
统一使用 generated 八核 ring，并根据已验证 geometry 自动选择 schedule。Partial
producer 在 reduction 前清理 inactive shard。见
[自动 residual reduction](../knowledge/concepts/allreduce-routing.md)。

### Graph 与 replay

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_DEEP_FAST_REPLAY` | 关闭；首字符不是 `0` 的任意非空值会启用 | Fused handle 构造 / **MODEL** | full-body replay 时跳过已审计的 setup 工作。配置必须证明每个被跳过的 input/layout 仍有效。 |
| `RPU_FASTREPLAY_SKIP_SYNC` | 开启；空值、`0`/`false`/`off` 禁用 | Graph BUILD / **BUILD** | 仅当本轮 replay 未触碰 Graph 所有的 kernel 参数，且每个 Launch register-state token 仍与 prepared batch 一致时，通用 Graph executor 才省略 mutable-kernel scan。mutable DMA endpoint 每轮独立解析和更新，提交前会发布全部 command 写入；通用脏标记、token 变化或 token 查询失败都会保守执行完整 scan。该变量只用于诊断退出，不是模型 route。 |
| `RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN` | 关闭；精确 `1`、`true`、`True` 或 `on` 会启用 | 首次原生 coexistence 使用 / **NATIVE** | 在配置负责的 subsystem handoff 之间保留 persistent SPM generation。归属错误可能破坏后续执行。 |
| `RPU_GRAPH_DEFER_TO_COPY` | 没有独立默认值；legacy alias 接受 `auto`/空、`0`/`off`/`false`，其他任意非空值强制开启 | 首次 host-op gate 使用 / **NATIVE** | 仅当未设置 `RPU_GRAPH_HOST_OP_DEFER_GATE` 时查询。避免同时设置两者。 |
| `RPU_GRAPH_HOST_OP_DEFER_GATE` | `auto`；`auto`/空、`0`/`off`/`false`，其他任意非空值强制开启 | 首次 host-op gate 使用 / **NATIVE** | 控制 capture 期间 stable host input 的 deferral。对不稳定 storage 强制开启会产生过期数据。 |
| `RPU_SKIP_IDLE_RECORD_FUNCTION` | 开启；`0`/`off`/`false` 会禁用 | 首次 Python graph scope / **MODEL** | 只在 profiler 关闭时省略 idle profiler scope。profiling 时保留 scope，其他情况下减少 host 开销。 |

### KV、linear、normalization 与 scheduling

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_ADARMS_FUSED_BCAST` | 关闭；除 `0`/`false`/`False` 外的非空值 | Handle/Graph build / **BUILD** | 融合配置负责的 AdaRMS broadcast。归属 layout 必须连续且稳定。 |
| `RPU_KVINSERT_HYBRID3_V16` | 关闭；除 `0`/`false`/`False` 外的非空值 | 每次 KV insertion / **CALL**；重建 Graph | 选择三段 aligned/tail schedule。必须重新验证 shape 和 cache signature。 |
| `RPU_KVINSERT_HYBRID_V16` | 关闭；除 `0`/`false`/`False` 外的非空值 | 每次 KV insertion / **CALL**；重建 Graph | 选择 aligned bulk/tail schedule。facade 可以负责该值；任意启用会改变 Graph 拓扑。 |
| `RPU_KVINSERT_V16` | 开启；精确 `0` 或 `false` 会禁用，其他所有值会启用 | 首次原生使用 / **NATIVE** | 在允许时选择 aligned KV-insert 路径。禁用会改变 scheduling 和性能，但不改变 logical cache length。 |
| `RPU_KVINSERT_V16_ANY_TP` | 关闭；除 `0`/`false`/`False` 外的非空值 | 首次原生使用 / **NATIVE** | 允许其他 whole-head parallel factor 使用 aligned 路径。不支持的 geometry 可能无法通过 admission。 |
| `RPU_LINEAR_ACC32` | 关闭；严格接受 `0`/`1`、`false`/`true` 或 `off`/`on` 及列出的大小写变体；无效值失败 | 首次原生使用 / **NATIVE** | 为 tiled linear 家族选择 FP32 accumulation。它会改变 tile、内存使用、性能和数值。 |
| `RPU_RMSNORM_NEWTON` | 关闭；严格接受 `0`/`1`、`false`/`true` 或 `off`/`on` 及列出的大小写变体；无效值失败 | 首次原生使用 / **NATIVE** | 选择 refined normalization。它会改变数值，也可能改变可用 schedule。 |
| `RPU_RMSNORM_VWARP` | 兼容输入；Hy-VLA 严格接受空/`0`/`off`/`false`、`16`、`32` 或 `auto`，Wall-OSS 拒绝该变量 | owner cold-plan admission / **NATIVE** | 归属 adapter 将其翻译成冻结的 per-handle capability。Hy-VLA 无效文本在进程 claim 前失败；原生 handle 不会重新读取。 |

<a id="qwen35-vision-selectors"></a>
### Qwen3.5 Vision selector

这些 selector 只在 numeric-blocked 受控评估门允许 Qwen3.5 Vision 后生效，
不会扩大其 Source-only 状态。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `QWEN3_5_VISION_CHUNK` | Auto；十进制整数；空值/非正值表示 auto | 首次原生 vision plan / **NATIVE** | 限制 Qwen3.5 Vision chunk。正数 override 必须仍处于允许的 shape 和 memory envelope 内。 |
| `RPU_VISION_MERGER` | **PB(true)** | Qwen3.5 Vision setup / **MODEL** | 启用 RPU merger 路径。`0` 选择 fallback，并改变性能/数值。 |
| `RPU_VISION_STEP0` | **PB(true)** | Qwen3.5 Vision setup / **MODEL** | 启用 RPU first-stage 路径。`0` 选择 fallback，并改变性能/数值。 |

<a id="gr00t-profile"></a>
## GR00T 配置

builder 将 base-zero-shot DROID 与 finetuned DROID 作为两个不同的精确 profile。
前者是两相机、时间索引 `[-15,0]`（四图），后者是两相机、时间索引 `[0]`
（双图）；两者都要求 embodiment 24。profile、checkpoint revision、
processor/statistics 身份和 CPU 输入范围都在模型构造前检查。gated Cosmos 的精确
八文件 manifest 也会在构造前校验，通用 Qwen3-VL-2B alias 不能替代。冻结的
processor input-oracle 为 S277/S145，KVPAD16 floor 为 288/160，profile-local
候选 cache 为 320/256。共享 512 行分配与 upstream 1024 行 action-position table
都不是输入支持上限；仍须通过 single-chunk SPM 与生命周期验证。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_GR00T_KVPAD16` | 原始默认关闭；facade 为 `1`；原生解析器中 `1` 或以 `t`/`T` 开头的值会启用 | 首次原生 KV setup / **NATIVE** | 为 GR00T expert padding 并 mask KV row。它会改变 cache layout 和 Graph signature。 |
| `RPU_GR00T_PARTIAL_MROPE` | 原始为 **PB(false)**；facade 为 **PB(true)** | GR00T 构造 / **MODEL** | 使用预计算的 partial multimodal position table。 |
| `RPU_GR00T_W8A16` | **PB(false)** | GR00T 权重安装 / **MODEL** | 量化允许的 backbone 和 action projection，并改变数值。 |

<a id="hy-embodied-profile"></a>
## Hy-Embodied 配置

直接 builder 或 `HyEmbodiedPolicy` 获取进程归属前，会先把 builder 默认值、facade
dtype 和 `runtime_env` override 合并成一个不可变 cold plan。下面标为严格解析的 RMS、
component、persist、bool 和量化 selector 若格式错误，会在 allocator policy 选择、
terminal poison 和 safetensors tensor 读取前失败。权重 layout、原生 handle、cache
geometry 和 Graph signature 只从该 plan 获取决定，不会再从环境变量重新读取这些
selector。下面的严格 token 都会去除首尾空白并忽略大小写，component list 以逗号
分隔；显式空 component selector 会禁用全部 component，仅含逗号和空 token 的 list
会失败。切换 policy 配置必须使用新进程。固定三相机 facade 的 `prefix_len` 仅接受
`{192, 208, 224, 240}`。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_HY_VLA_ACTION_MLP_MC` | 未设置表示 8 cores；`0`/`off`/`false` 表示 1，其他值表示 8 | 进程 claim 前的 cold-plan admission / **MODEL** | 选择 multi-core action-MLP 执行。它会改变 reduction order，且必须继续满足配置数值门禁。 |
| `RPU_HY_VLA_ATTN_TP8` | 原始默认关闭；只有 `1`/`true`/`on` 会启用；builder 默认 `1` | 进程 claim 前的 cold-plan admission / **MODEL** | 为 eight-core attention 复制 KV head。cache geometry 和权重使用冻结值。 |
| `RPU_HY_VLA_CACHING_ALLOC` | 开启；严格接受 `0`/`off`/`false` 或 `1`/`on`/`true`；显式空值表示开启；其他值失败 | 进程 claim 前的 cold-plan admission / **PROCESS** | 在物化前选择通用的冷态 tensor allocation policy；进程中的第一次 policy 声明不可更改。它会改变内存保留；测量 free memory 前调用 `empty_cache()`。 |
| `RPU_HY_VLA_DENOISE_UNROLL` | builder 默认 `1`；严格接受 `0`/`off`/`false` 或 `1`/`on`/`true`；显式空值表示关闭；其他值失败 | 进程 claim 前的 cold-plan admission / **MODEL** | 将 denoise loop 记录为一个配置负责的 Graph。它会改变 Euler-state precision 和数值结果。 |
| `RPU_HY_VLA_FAST_REPLAY` | builder 默认 `1`；严格接受开关别名或 `vit`、`vlm`、`expert` 的精确 CSV subset；`all` 表示全部；未知 component 失败 | 进程 claim 前的 cold-plan admission / **NATIVE** | replay 时跳过已审计的 layer-body emission。错误使用可能使 dynamic input 过期。 |
| `RPU_HY_VLA_FAST_REPLAY_PRELOAD` | builder 默认 `1`；使用与 `RPU_HY_VLA_FAST_REPLAY` 相同的严格 component grammar | 进程 claim 前的 cold-plan admission / **NATIVE** | replay 时也跳过已审计的 preload 工作。需要稳定的 persistent state。 |
| `RPU_HY_VLA_FUSED_MERGER` | builder 默认 `1`；严格接受 `0`/`off`/`false` 或 `1`/`on`/`true`；显式空值表示关闭；其他值失败 | 进程 claim 前的 cold-plan admission / **MODEL** | 将 vision merger 折叠进 Graph；runner 复用冻结后的 Python 决定。 |
| `RPU_HY_VLA_KVPAD16` | builder 默认 `1`；严格接受开关别名或 `vit`、`vlm`、`expert` 的精确 CSV subset；未知 component 失败 | 进程 claim 前的 cold-plan admission / **NATIVE** | 对 expert KV row 进行 padding/mask。它会改变 cache layout 和 signature。 |
| `RPU_HY_VLA_MASK_ONCE` | builder 默认 `1`；严格接受开关别名或 `vit`、`vlm`、`expert` 的精确 CSV subset；未知 component 失败 | 进程 claim 前的 cold-plan admission / **NATIVE** | 复用 invariant mask upload。mask 内容或 storage 可能变化时不安全。 |
| `RPU_HY_VLA_MERGER_IN_GRAPH` | builder 默认 `1`；严格接受 `0`/`off`/`false` 或 `1`/`on`/`true`；显式空值表示关闭；其他值失败 | 进程 claim 前的 cold-plan admission / **MODEL** | 将启用的 fused merger 放进 capture。要求 `RPU_HY_VLA_FUSED_MERGER=1`。 |
| `RPU_HY_VLA_MOT_NORM_NOMERGE` | 默认 `both`；严格接受关闭别名、`1`/`on`/`true`/`both`，或 `qkv`、`mlp` 的精确 CSV subset；显式空值表示 `both`；其他文本失败 | 进程 claim 前的 cold-plan admission / **NATIVE** | 选择配置特定的 norm/merge scheduling。它会改变 temporary-memory 和 Graph 结构。 |
| `RPU_HY_VLA_PARTIAL_ROPE` | builder 默认 `expert,vlm`；严格接受开关别名或 `vit`、`vlm`、`expert` 的精确 CSV subset；未知 component 失败 | 进程 claim 前的 cold-plan admission / **NATIVE** | 为指定 subsystem 使用预计算的 partial position table。scope 不匹配会改变 position semantics。 |
| `RPU_HY_VLA_PATCH_EMBED_IN_GRAPH` | builder 默认 `1`；严格接受 `0`/`off`/`false` 或 `1`/`on`/`true`；显式空值表示开启；其他值失败 | 进程 claim 前的 cold-plan admission / **MODEL** | capture device patch embedding。禁用会选择不同的 host/device boundary。 |
| `RPU_HY_VLA_PATCH_EMBED_MC` | 未设置表示 8 cores；`0`/`false` 表示 1，其他值表示 8；builder 默认 `1` | 进程 claim 前的 cold-plan admission / **MODEL** | 选择 multi-core patch embedding。权重 layout 与原生执行使用冻结 core 数。 |
| `RPU_HY_VLA_PERSIST_HANDLES` | builder 默认 `1`；严格关闭别名或显式空值禁用全部，开启别名/`all` 启用全部，`vit`、`vlm`、`expert` 的精确 CSV subset 选择 owner；未知或仅含空 token 的 CSV 失败 | 进程 claim 前的 cold-plan admission / **MODEL** | 跨调用保持指定 handle 存活。会增加保留内存，并实施单 policy 归属。 |
| `RPU_HY_VLA_PREFIX_TEMPLATE` | 开启；严格接受 `0`/`off`/`false` 或 `1`/`on`/`true`；显式空值表示开启；其他值失败 | 进程 claim 前的 cold-plan admission / **MODEL** | 启用配置 prompt template。改变它会改变 token input，而不只是性能。 |
| `RPU_HY_VLA_PROJ1_IN_MERGER` | 开启；严格接受 `0`/`off`/`false` 或 `1`/`on`/`true`；显式空值表示开启；其他值失败 | 进程 claim 前的 cold-plan admission / **MODEL** | 在 merger 路径中包含 first projection；runner 对 layout 和 ownership 复用冻结决定。 |
| `RPU_HY_VLA_Q_INPLACE` | builder 默认 `1`；严格接受 `0`/`off`/`false` 或 `1`/`on`/`true`；显式空值表示关闭；其他值失败 | 进程 claim 前的 cold-plan admission / **NATIVE** | 原地复用 query buffer。仅当配置证明旧值已死亡时才安全。 |
| `RPU_HY_VLA_RMSNORM_PAD16` | 原始默认关闭；严格接受开关别名或 `vit`、`vlm`、`expert` 的精确 CSV subset；未知 component 失败 | 进程 claim 前的 cold-plan admission / **NATIVE** | 对指定 normalization row 做 padding。它会改变 layout 和 Graph census。 |
| `RPU_HY_VLA_SILU_MUL` | builder 默认 `expert`；严格接受开关别名或 `vit`、`vlm`、`expert` 的精确 CSV subset；未知 component 失败 | 进程 claim 前的 cold-plan admission / **NATIVE** | 融合指定的 activation/multiply schedule。每个 scope 都需要数值验证。 |
| `RPU_HY_VLA_VIT_PACKED` | builder 默认 `1`；严格接受 `0`/`off`/`false` 或 `1`/`on`/`true`；显式空值表示关闭；其他值失败 | 进程 claim 前的 cold-plan admission / **MODEL** | 打包 vision-tower 执行配置。它会改变 shape/layout 契约，并要求重建权重/Graph。 |
| `RPU_HY_VLA_W4A16` | 关闭；严格接受开关别名或 `vit`、`vlm`、`expert`、`vlm_text`、`vlm_vision` 的精确 CSV subset；未知或仅空 CSV token 失败 | 进程 claim 前的 cold-plan admission / **MODEL** | 按 scope 选择 W4A16；重叠时 W4 优先于 W8。不支持的 ViT W4 会在归属和 tensor 读取前失败。数值配置会改变。 |
| `RPU_HY_VLA_W8A16` | 原始默认关闭；facade 映射 `fp16`→`0`、`w8a16`→`all`、`w8a16-expert`→`expert`、`w8a16-expert-vlmv`→`expert,vlm_vision`、`w8a16-vlm`→`vlm`、`w8a16-vit`→`vit`、`w8a16-no-vlm`→`vit,expert`；其他值使用 W4 行的严格 grammar | 进程 claim 前的 cold-plan admission / **MODEL** | 按 scope 选择 W8A16。它必须匹配指定 precision profile；格式错误会在归属和 tensor 读取前失败。 |
| `RPU_RMSNORM_VWARP` | Hy-VLA builder 默认 `auto`；严格接受空/`0`/`off`/`false`、`16`、`32` 或 `auto`；其他文本失败 | 进程 claim 前的 cold-plan admission / **NATIVE** | 冻结允许的 VLM RMSNorm capability；原生 handle 不再读取环境变量。 |

<a id="internvla-n1-and-navdp-profiles"></a>
## InternVLA-N1 与 NavDP 配置

受控 InternVLA 路径还要求通过其 Python API 提供由调用者以 SHA256 固定的 asset
manifest；不能用环境路径替代。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_INTERNVLA_DINO_FINAL_NORM_RPU` | **PB(true)** | DINO tower 构造 / **MODEL** | 在 RPU 上运行 final normalization。`0` 会改变 host/device boundary 和 timing。 |
| `RPU_INTERNVLA_DINO_NORM_FOLD` | **PB(false)** | DINO tower 构造 / **MODEL** | 将 input normalization 折叠进 patch 权重。它会改变已安装权重，必须重建。 |
| `RPU_INTERNVLA_HOST_CACHE` | **PB(true)** | Backbone 构造 / **MODEL** | 启用 host memoization 和 stable owner。禁用会增加重复 host 工作。 |
| `RPU_NAVDP_DENOISE_UNROLL` | 关闭；去除空白后，`1`/`true`/`True` 会启用，其他值禁用 | NavDP action 调用 / **CALL**；重建 Graph | capture 固定 denoise loop。它会改变 Graph 拓扑和 step 执行。 |
| `RPU_NAVDP_BATCH` | 开启；去除空白后，`1`/`true`/`True` 会启用，其他值禁用 | 首次 unrolled action dispatch / **CALL**；重建 Graph/model | batch trajectory。开启时，首次 dispatch 后 sample count 固定；之后不匹配会失败。 |
| `RPU_NAVDP_SPM_KV_BY_MHA` | `1`；仅接受精确 `0` 或 `1` | NavDP fused handle 构造 / **MODEL** | 允许已认证的初始 self-attention 配置在精确联合 layout 可容纳时使用临时 SPM K/V。`0` 固定使用 DDR；不符合条件的调用仍使用 DDR。 |

<a id="lingbot-vla-v2-profile"></a>
## LingBot-VLA-V2 配置

优先使用 `Lingbot2Policy.from_checkpoint(..., runtime_env=...)`。facade 会校验精确
allowlist，在导入 adapter 前应用完整 snapshot，并在 close 时恢复 shell 环境；原生状态
仍是 process-cold，因此另一个配置应使用新进程。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_LINGBOT2_ADARMS_DIRECT_SCHEDULE` | 原始/facade 为 `0`；`1`/`true`/`True`/`on` | Expert 构造 / **MODEL** | 使用 direct indexed AdaRMS schedule。它与 denoise unroll 冲突，并要求 expert replay。 |
| `RPU_LINGBOT2_BASE_W8A16` | 原始为 `0`；**E1**；W8/W4 facade 为 `1` | 权重转换 / **MODEL** | 量化 shared/base expert projection。它是完整 precision 配置的一部分，不是独立 toggle。 |
| `RPU_LINGBOT2_DENOISE_UNROLL` | 原始为 `0`；facade 为 `1`；facade 只把精确 `1` 规范为开启 | Policy 构造 / **MODEL** | 记录固定 ten-step denoise Graph，并使用 FP16 Euler state。`0` 恢复 host loop 并改变数值。 |
| `RPU_LINGBOT2_ENCODER_1THREAD` | `1`；`1`/`true`/`True`/`on` 会启用 | Policy 构造 / **MODEL** | 使用单线程运行小型 host encoder 工作。`0` 使用调用者进程级 thread 设置。 |
| `RPU_LINGBOT2_EXPERT_REPLAY` | Facade 为 `1`；`1`/`true`/`True`/`on` | Policy 构造 / **MODEL** | 启用稳定 expert Graph replay。要求稳定 input owner；禁用会增加 build/dispatch 工作。 |
| `RPU_LINGBOT2_EXPERT_W4A16` | 原始为 `0`；**E1**；W4 facade 为 `1` | 权重转换 / **MODEL** | 选择 W4A16 routed-expert 权重。若 W4/W8 同时设置则 W4 优先；仅使用完整 W4 配置。 |
| `RPU_LINGBOT2_EXPERT_W8A16` | 原始为 `0`；**E1**；W8 facade 为 `1` | 权重转换 / **MODEL** | W4 关闭时选择 W8A16 routed-expert 权重。数值配置会改变。 |
| `RPU_LINGBOT2_FP16_TOP4` | 除非启用 dense-router 诊断，否则原生默认开启；facade 为 `1`；精确 `0`/`1` | Expert 权重安装 / **MODEL** | 保持严格 FP16 top-4 routing。没有 dense diagnostic 时禁用会失败；routing 变化对精度敏感。 |
| `RPU_LINGBOT2_GROUPED_EXPERTS` | 原始为 `0`；**E1**；W8/W4/FP16 facade 为 `1` | 权重转换和 handle build / **MODEL** | 选择 packed grouped expert。必须与权重格式和 row-chunk 配置一致。 |
| `RPU_LINGBOT2_HOST_PREFIX_OPT` | Facade 为 `1`；`1`/`true`/`True`/`on` | Policy 构造 / **MODEL** | 保持稳定 prompt-prefix owner 和 memoization。VLM replay 与 direct-prefix output 需要它。 |
| `RPU_LINGBOT2_LEGACY_PREPROC` | `0`；**E1** | Policy 构造 / **MODEL** | 强制 legacy preprocessing 的兼容性 override。它会改变模型输入，不是仅性能开关。 |
| `RPU_LINGBOT2_PREFILL_FAST_REPLAY` | 原始为 `0`；`1`/`true`/`True`/`on`；facade 为 `1` | Prefill handle setup / **MODEL** | 启用已审计的 prefill body replay skip。要求 VLM replay 和稳定 prefix storage。 |
| `RPU_LINGBOT2_PREFILL_W4A16` | 原始/facade 为 `0`；`1`/`true`/`True`/`on` | 权重转换 / **MODEL** | 只把允许的 prefill projection 量化为 W4A16 的受限选项。普通 W4 使用 prefill W8A16。 |
| `RPU_LINGBOT2_PREFILL_W8A16` | 原始为 `0`；`1`/`true`/`True`/`on`；W8/W4 facade 为 `1` | 权重转换 / **MODEL** | 选择 W8A16 prefill projection。必须属于完整 precision 配置。 |
| `RPU_LINGBOT2_PREPROC` | `exact`；小写 `exact`、`fast` 或 `legacy`；其他 facade input 在 build 前规范为 `exact` | Policy 构造 / **MODEL** | 选择 preprocessing semantics，因此会改变模型输入。 |
| `RPU_LINGBOT2_QWEN3VL_BASE` | Packaged/registry fallback；包含有效 config 的非空本地目录 | Policy 构造 / **MODEL** | 覆盖 Qwen3-VL base asset。错误或不受信任内容无法通过校验；它不是下载凭据。 |
| `RPU_LINGBOT2_VISION_DIRECT_PREFIX` | Facade 为 `0`；`1`/`true`/`True`/`on` | Policy 构造 / **MODEL** | 将 fused vision 结果写入 stable prefix owner。要求 host-prefix optimization、VLM replay、batching 和 fused merger。 |
| `RPU_LINGBOT2_VISION_DIRECT_PREFIX_NO_OUTPUT` | Facade 为 `0`；`1`/`true`/`True`/`on` | Policy 构造 / **MODEL** | 仅在启用 direct-prefix 时抑制额外 vision return。配对错误会失败。 |
| `RPU_LINGBOT2_VISION_FAST_REPLAY` | 原始为 `0`；`1`/`true`/`True`/`on`；facade 为 `1` | Vision handle setup / **MODEL** | 启用已审计的 vision replay skip。要求指定 vision topology 和稳定 dynamic input。 |
| `RPU_LINGBOT2_VISION_W8A16` | 原始为 `0`；`1`/`true`/`True`/`on`；W8/W4 facade 为 `1` | Vision 权重转换 / **MODEL** | 选择 W8A16 vision projection。它会改变数值，并必须匹配完整 checkpoint 配置。 |
| `RPU_LINGBOT2_VLM_REPLAY` | host-prefix optimization 开启时默认为 `1`，否则为 `0`；`1`/`true`/`True`/`on` | Policy 构造 / **MODEL** | 启用跨调用 VLM prefill replay。prefix storage 不稳定时会拒绝 `1`，以防止过期读取。 |

<a id="pi05-profile"></a>
## Pi0.5 配置

Pi0.5 loader 负责这些设置。Graph selector 必须在模型构造和首次 capture 前固定。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_PI05_ADARMS_GRAPH` | **PB(true)** | AdaRMS 构造 / **MODEL** | 启用 AdaRMS GraphCache 路径。`0` 使用 eager fallback，并改变 timing/lifecycle。 |
| `RPU_PI05_ADARMS_W8A16_GRAPH` | **PB(true)** | 量化 AdaRMS 构造 / **MODEL** | 为 W8A16 AdaRMS 路径启用 GraphCache。必须匹配量化权重。 |
| `RPU_PI05_DENOISE_GRAPH` | **PB(true)** | Action 构造 / **MODEL** | capture denoise step。禁用会改变 dispatch 行为，并使 replay 测量失效。 |
| `RPU_PI05_DENOISE_UNROLL` | 原始为 **PB(false)**；Pi adapter 提供 `1` | Action 构造 / **MODEL** | 在一个 Graph 中记录固定 denoise loop，并使用 FP16 Euler state。相对 host loop 会改变数值。 |
| `RPU_PI05_EMBED_PREFIX_PATCH` | **PB(true)** | Adapter class-patch 安装 / **MODEL** | 启用支持的 prefix-embedding patch。class hook 是进程状态，因此只能在新进程中改变。 |
| `RPU_PI05_EULER_FP16` | **PB(false)** | 每次 action 调用 / **CALL**；重建 prepared Graph 配置 | 使用 FP16 而不是 host-FP32 Euler state。它会直接改变 action 数值。 |
| `RPU_PI05_FUSED_DENOISE` | **PB(true)** | Action 构造 / **MODEL** | 使用 fused denoise subsystem。`0` 选择 legacy per-step 路径，并改变拓扑/性能。 |
| `RPU_PI05_GEMMA_GRAPH` | **PB(true)** | VLM 构造 / **MODEL** | 启用 Gemma GraphCache 路径。需要稳定 prefix/cache owner。 |
| `RPU_PI05_KEEP_CPU` | 只接受未设置/空；任意非空值（包括 `0`）都会引发错误 | Pi0.5 安装 / **MODEL** | retired-path tripwire。刻意不支持通过该变量保留 CPU 权重。 |
| `RPU_PI05_KVINSERT_PAD16` | **B01(false)** | Expert 构造 / **MODEL** | 对 KV row 做 padding/mask。它会改变 cache layout 和 Graph signature。 |
| `RPU_PI05_PREFIX_MASK_CACHE` | **PB(true)** | 每次 action 调用 / **CALL** | 复用 one-entry host mask cache。`0` 会重新计算；正确性仍要求稳定 Graph input。 |
| `RPU_PI05_PREFIX_PAD16` | **PB(true)** | Prefix 规划 / **CALL**；新 plan 需重建 Graph | 为允许的 shape 增加 masked prefix padding。它改变 execution length/signature，不改变 logical prefix length。 |
| `RPU_PI05_SIGLIP_BATCH` | **PB(true)** | SigLIP 构造 / **MODEL** | 打包兼容 camera input。禁用会改变 Graph shape 和 host/device traffic。 |
| `RPU_PI05_SIGLIP_GRAPH` | **PB(true)** | SigLIP 构造 / **MODEL** | 启用 SigLIP GraphCache 路径。`0` 是 timing 不同的诊断 fallback。 |

<a id="qwen3-vl-shared-vision-profile"></a>
## Qwen3-VL 共享 Vision 配置

这些变量由 Qwen3-VL Vision 以及嵌入该 tower 的 VLA facade 消费。facade 可以在模型
构造前设置精确默认值。

精确普通 dense FP16 Qwen3-VL-2B、4B 和 8B 配置默认在 RPU 执行 patch 和
merger GEMM，merger LayerNorm 保持 CPU FP32。安装前设置
`RPU_QWEN3VL_VISION_HOST_FP32_PATCH=1` 可保留 CPU FP32 patch projection 和
CPU FP32 merger。该选择在安装时固定；之后修改环境变量需要重新创建模型。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_QWEN3VL_VISION_BATCH` | **PB(false)** | Vision forward / **CALL**；重建 Graph | 打包兼容 image/view。它会改变 Graph signature 和内存；facade 约束仍会限制 shape。 |
| `RPU_QWEN3VL_VISION_BATCH_CAP` | `3`；十进制整数，至少收窄为 `1` | Vision forward / **CALL**；重建 Graph | 每个 packed group 的最大兼容 image 数。更大值可能超过配置内存 envelope。 |
| `RPU_QWEN3VL_VISION_FUSED_MERGER` | 原始默认关闭；**B01(false)** | Vision 构造 / **MODEL** | 将 merger 折叠进 Vision Graph。只有字面值 `0`/`1` 能保持 Python 与原生 reader 一致。 |
| `RPU_QWEN3VL_VISION_HOST_FP32_PATCH` | **PB(false)** | Vision 安装 / **MODEL**；在 `to_rpu()` 前设置，修改需新建模型 | 固定 CPU FP32 patch 配置；精确普通 dense FP16 VL2/VL4/VL8 同时保留 CPU FP32 merger。它会改变数值和 transfer cost。 |
| `RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE` | `0`；**E1** | Vision 构造 / **MODEL** | 将 patch embedding 移到 RPU。它会改变已安装权重、数值和 Graph 拓扑。 |
| `RPU_QWEN3VL_VISION_ROPE_SPM` | 关闭；原生值以 `1`、`t` 或 `T` 开头时启用 | 首次原生 Vision 使用 / **NATIVE** | 将 Vision position table 保存在 SPM。它会改变 persistent memory 使用，并要求重建 Vision handle。 |

<a id="rhinovla-profile"></a>
## RhinoVLA 配置

这些是源码级集成控制项。请使用模型仓库的 runtime factory 和精确、已验证的配置；
单独一个 RhinoForge 开关并不构成端到端契约。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_RHINOVLA_DENOISE_STATIC_CONTEXT_CACHE` | 关闭；除 `0`/`false`/`False` 外的非空值 | 首次原生 action 使用 / **NATIVE** | cache 由 factory 证明为静态的 denoise context。错误使用会复用过期 observation。 |
| `RPU_RHINOVLA_FOLD_ACTION_TIME_IN` | 关闭；除 `0`/`false`/`False` 外的非空值 | 首次原生 action 使用 / **NATIVE** | 使用配置的 folded action/time input projection。要求匹配已安装权重。 |
| `RPU_RHINOVLA_FUSED_ADARMS_GEMV` | 关闭；**B01(false)** | Action 构造（**MODEL**）和首次原生使用（**NATIVE**）；新进程 | 启用 fused AdaRMS projection。Python/原生状态必须一致，并会改变 Graph census。 |
| `RPU_RHINOVLA_FUSED_SILU_MUL` | 关闭；**B01(false)** | Action 构造（**MODEL**）和首次原生使用（**NATIVE**）；新进程 | 启用 fused activation/multiply。它会改变数值顺序，需要配置验证。 |
| `RPU_RHINOVLA_GATED_NO_SUB` | 关闭；**B01(false)** | Action 构造（**MODEL**）和首次原生使用（**NATIVE**）；新进程 | 使用配置的 subtraction-free gated formulation。只在已验证等价性的配置中启用。 |
| `RPU_RHINOVLA_KVINSERT_HYBRID_V16` | 关闭；除 `0`/`false`/`False` 外的非空值 | 每次 KV insertion / **CALL**；重建 Graph | RhinoVLA scope 的 alias，用于选择共享 hybrid KV schedule。它会改变 cache Graph 拓扑。 |
| `RPU_RHINOVLA_PRECOMPUTE_ADARMS` | 关闭；除 `0`/`false`/`False` 外的非空值 | 首次原生 action 使用 / **NATIVE** | 使用 factory 绑定的预计算 AdaRMS table。table 缺失/不匹配会失败或产生过期 conditioning。 |
| `RPU_RHINOVLA_PRECOMPUTE_TIME_PROJ` | 关闭；除 `0`/`false`/`False` 外的非空值 | 首次原生 action 使用 / **NATIVE** | 使用 factory 绑定的预计算 time projection。只对绑定的 timestep schedule 有效。 |
| `RPU_RHINOVLA_SKIP_ADARMS_GEMV` | 关闭；**B01(false)** | Action 构造（**MODEL**）和首次原生使用（**NATIVE**）；新进程 | 只有绑定预计算值时才跳过 AdaRMS projection。错误组合会产生无效 conditioning。 |
| `RPU_RHINOVLA_VISION_BATCH_MERGERS` | **PB(false)** | Vision 安装 / **MODEL** | batch 兼容 merger 调用。它会改变 Graph shape 和内存使用。 |
| `RPU_RHINOVLA_VISION_BATCH_VIEWS` | **PB(false)** | Vision forward / **CALL**；重建 Graph | 打包等尺寸 view，部分 batched merger 配置要求启用。混合 geometry 会回退或失败。 |
| `RPU_RHINOVLA_VISION_CPU_MERGER_NORM_FP16` | **PB(false)** | Vision 安装 / **MODEL** | 为 CPU merger-normalization fallback 使用 FP16。会直接改变数值结果。 |
| `RPU_RHINOVLA_VISION_PREP_CACHE` | **PB(false)** | Vision 安装 / **MODEL** | memoize stable preprocessing artifact。会保留 host 内存，并要求正确 cache key。 |
| `RPU_RHINOVLA_VISION_RPU_MERGER_NORM` | **PB(false)** | Vision 安装 / **MODEL** | 在 RPU 上运行 merger normalization。它会改变 boundary 和数值配置。 |
| `RPU_RHINOVLA_VISION_RPU_MERGER_OUTPUT_RPU` | **PB(false)** | Vision 安装 / **MODEL** | 将 merger output 保留在 RPU 上。consumer 必须接受 device-resident output 和稳定 ownership。 |
| `RPU_RHINOVLA_VISION_SKIP_RAW_SNAPSHOTS` | **PB(false)** | Vision 安装 / **MODEL** | `1` 抑制 raw debug snapshot。保持关闭会增加内存/同步，并可能保留敏感模型 input/intermediate。 |

### RhinoVLA expert-only W8 冷安装

可信 factory 可向 `RhinoVLAOnRPU` 传入 `expert_w8a16` 与 `full_expert_w8a16`。
前者选择 expert 七类投影，后者还选择其 AdaRMS 条件投影并要求前者为真。
两者只接受布尔值或 `None`（沿用现有环境选择）。公开示例的
`input.full_w8a16` 对应构造参数 `full_expert_w8a16`，不对应全流水线环境开关。

exact full-expert 模式绑定 18 组 W8 条件 owner，final norm 保持浮点预计算，
最终表及 action IO 保持 FP16。既有 `RPU_RHINOVLA_FULL_W8A16` 路径仍要求
19 组 W8 owner 与 6 个 IO scale，HIGH 和 gate-TANH 预计算仍限该路径。
精度、IO owner 和表都在冷安装时固定，改变时须重建模型；数值门槛不变。
FP16、expert W8 和 full-expert W8 模板的九项公开缓存/融合环境控制见
[配置说明](../examples/configs/rhinovla/v3/README.md)，不会隐式扩大精度范围。


独立的 full-pipeline 示例显式启用 `RPU_RHINOVLA_FULL_W8A16`，推理前检查
实际安装的 text、vision、action IO 与冷 AdaRMS owner；其范围不同于 expert-only，
基础版显式关闭 HIGH、gate-TANH 预计算与 vector Q/K norm，HIGH 版则启用它们。
模板本身不代表数值或任务质量认证。

RhinoVLA 在 `components.language_model.prefill` 与
`components.vision_encoder.vision` 接受冷态布尔 `linear_acc32`。
默认 `false` 使用 ACC16，`true` 选择现有 text Linear 或 vision block/融合 merger 的
ACC32 路径。patch embedding、action 与冷 AdaRMS 保持既有累加策略。
构造后不能改变这些字段，action 组件不接受该字段；HIGH 非线性/归一化选项
与 Linear 累加精度互相独立。详见 [v3 模板](../examples/configs/rhinovla/v3/README.md)。

<a id="wall-oss-profile"></a>
## Wall-OSS 配置

`WallOssPolicy` 会在模型构造前提供经过验证的 facade 默认值。prompt selector 会改变
实际模型输入，绝不能当作仅影响性能的开关。所有 selector 均未设置时，固定公开
checkpoint 按 `use_embodied_system_prompt_ratio=0.0` 使用公开 `wall-x` 中精确的
`You are a helpful assistant.` system prompt。

Source-only facade 固定 Vision `chunk_size=768`、关闭 packed Vision，并在 checkpoint
权重加载前调用 `preflight_images(images)`。CPU processor 冻结的 grid 必须精确为两张
`[1,32,32]` 图片；每张图独立使用共享的 `768+256` KV-first plan。`auto`、任何其他
grid 和之后变化的 grid 分别会在进程级 claim 前或任何 Vision 工作前拒绝。这是准入
契约，不是板上验证证据。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_WALL_OSS_ACTION_FP32_TAIL` | **PB(false)** | 每次 denoise 调用 / **CALL**；重建 prepared Graph 配置 | Euler tail 使用 host FP32。它会改变 action 数值，并与 fused/unrolled 执行冲突。 |
| `RPU_WALL_OSS_ATTN_TP8` | 原始为 **PB(false)**；facade 为 `1` | 权重安装 / **MODEL** | 为 eight-core attention 复制 KV head。重建权重、cache 和 Graph。 |
| `RPU_WALL_OSS_BATCH_MAX_SEQ` | `768`；十进制整数 | Vision grouping / **CALL**；重建 Graph | 限制 packed vision sequence length。过大可能超出内存；所选公开配置可能进一步约束。 |
| `RPU_WALL_OSS_BATCH_VISION` | 原始 **PB(true)**；Source-only facade 为 `0` | Vision forward / **CALL**；重建 Graph | 直接 builder 可在上限内打包 equal-grid image。公开 facade 固定逐图预检，并拒绝启用该 selector。 |
| `RPU_WALL_OSS_DENOISE_UNROLL` | **PB(true)** | Action 构造和调用 / **MODEL** | 使用 fused denoise loop。`0` 选择 host loop，并改变拓扑和 timing。 |
| `RPU_WALL_OSS_DEVICE_PATCH_EMBED` | **PB(true)** | Vision 构造 / **MODEL** | 在 RPU 上运行 folded patch embedding。`0` 使用 CPU projection，并改变 boundary/性能。 |
| `RPU_WALL_OSS_EXPERT0_DOWN_INT4` | **PB(true)** | 量化 LLM 权重安装 / **MODEL** | 使用允许的 compact expert-0 down projection。部分 W4 配置强制关闭；不匹配会校验失败。 |
| `RPU_WALL_OSS_FAST_IMGPROC` | **PB(true)** | Policy 构造 / **MODEL** | 首次调用 parity 检查后使用 direct image-preprocessing 路径。`0` 强制使用 HF processor。 |
| `RPU_WALL_OSS_FAST_PROCESSOR` | **PB(true)** | Policy 构造 / **MODEL** | 首次调用 parity 检查后绕过 combined processor wrapper。`0` 保留 wrapper。 |
| `RPU_WALL_OSS_FAST_REPLAY` | 原始默认关闭；不以 `0` 或 `f` 开头的非空值；facade 为 `1` | Fused handle 构造 / **MODEL** | replay 时跳过已审计的 layer-body emission。错误使用可能 replay 过期 dynamic state。 |
| `RPU_WALL_OSS_FUSED_ASSEMBLE` | 原始为 **PB(false)**；facade 为 `1` | Vision 构造 / **MODEL** | 将 Vision reorder 与 input-embedding 构造合并。需要 on-device embedding 路径。 |
| `RPU_WALL_OSS_FUSED_DENOISE` | Facade 为 **PB(true)** | Action 构造 / **MODEL** | 使用 fused action subsystem。`0` 选择 legacy per-step 路径，并改变拓扑/timing。 |
| `RPU_WALL_OSS_GENERIC_PROLOGUE` | **PB(false)** | Prompt 构造 / **CALL** | 选择 generic prompt schema。与其他 prologue selector 互斥，并改变 token。 |
| `RPU_WALL_OSS_HOST_CACHE` | **PB(true)** | Component 构造 / **MODEL** | 启用 host layout/prompt memoization 和 stable owner。`0` 增加重复 host 工作。 |
| `RPU_WALL_OSS_KVINSERT_PAD16` | **PB(true)** | LLM 构造 / **MODEL** | 对 prefill KV row 做 padding/mask。它会改变 execution length、cache layout 和 Graph signature。 |
| `RPU_WALL_OSS_MULTISUITE_PROLOGUE` | **PB(false)** | Prompt 构造 / **CALL** | 选择 multisuite prompt schema；与 generic/short mode 互斥。改变 token。 |
| `RPU_WALL_OSS_ONDEVICE_EMBED` | **PB(true)** | Policy 构造 / **MODEL** | 在 RPU 上构造 token/vision embedding。`0` 使用 CPU fallback，并禁用 fused construction 路径。 |
| `RPU_WALL_OSS_PARTIAL_MROPE` | **PB(true)**；facade 设置 `1` | LLM 构造 / **MODEL** | 使用必需的 Wall-OSS multimodal position-table 路径。已准入的 Wall-OSS input 会拒绝显式 `0`。 |
| `RPU_WALL_OSS_PE_NORM_FOLD` | **PB(true)** | Policy 构造 / **MODEL** | fast preprocessing 开启时，将 image normalization 折叠进 patch 权重。改变时需重建已安装权重。 |
| `RPU_WALL_OSS_SHORT_PROMPT` | **PB(false)** | Prompt 构造 / **CALL** | 选择 short prompt schema；与其他 prologue mode 互斥。改变 token。 |
| `RPU_WALL_OSS_VISION_FUSED_MERGER` | 原始为 **B01(false)**；facade 为 `1` | Vision 构造 / **MODEL** | 将 merger 折叠进 Vision Graph。只有字面值 `0`/`1` 能保持 Python/原生状态一致。 |
| `RPU_WALL_OSS_VISION_ROPE_SPM` | 原始默认关闭；**N1**；facade 为 `1` | 首次原生 Vision 使用 / **NATIVE** | 将 position table 保存在 SPM。会改变 persistent-memory 使用，并要求重建 Vision handle。 |

## 仅诊断项清单

本节中的任何变量出现在 TOML 中时，完整公共 runner 都会拒绝。开发者仍可在 shell 中
显式设置某一变量，用于受控本地调查。这些控制项可能改变执行、dump 模型数据、暴露 tensor summary
或 timing 结构，或者使正确性/性能声明失效。生产环境中应保持未设置，分享前审查每个
生成的 artifact。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `QWEN3_5_VISION_GRAPH_DISABLE` | `0`；除精确 `0` 外的任意值都会禁用 | Vision forward / **CALL**；重建/清除 Graph | 绕过 Qwen3.5 Vision GraphCache，用于受控比较。replay 和 latency 结论不再适用。 |
| `RPU_ALL_GATHER_FORCE_MULTI_CORE` | 关闭；**N1** | 首次原生使用 / **NATIVE** | 强制 multi-core schedule，用于 A/B 比较。它不是普遍支持的性能 selector。 |
| `RPU_CHUNK_FORCE_UNSAFE` | 关闭；**N1** | 首次适用的原生 planner/launcher 使用 / **NATIVE** | 绕过 planner safety check。可能超过执行约束，绝不能产出可部署输出。 |
| `RPU_DYNAMO_MATERIALIZE_BREAKS` | **PB(false)** | Dynamo partitioning / **CALL**；重新编译 | materialize partition break。它会改变 Graph boundary 并增加 transfer，仅用于 compiler 诊断。 |
| `RPU_GRAPH_DDR_SPM_LOG` | 关闭；首字符不是 `0` 的非空值 | 首次 data-node 执行 / **NATIVE** | 记录 node index、byte count 和有界 source checksum。输出由模型数据派生，且日志会改变 timing。 |
| `RPU_GRAPH_FORCE_ONESHOT_ON_REPLAY` | 关闭；首字符不是 `0` 的非空值 | 首次 replay 检查 / **NATIVE** | 执行 one-shot 路径而不是普通 replay。会使 Graph lifecycle 和性能结论失效。 |
| `RPU_KVINSERT_V16_TRACE` | 关闭；除精确 `0` 或 `false` 外，任何出现的值都会启用；export 的空值也会启用 | 首次原生使用 / **NATIVE** | 记录 KV route selection 和 shape。它是诊断输出，不是 cache 正确性证明。 |
| `RPU_L2_BUFONLY` | 未出现时关闭；任意出现（包括 `0`）都会启用 | Grouped-expert Graph emission / **BUILD** | 声明 grouped buffer，但运行 per-expert 路径进行 bisection。会改变 Graph 和性能。 |
| `RPU_L2_DOWN_ACC16` | 关闭；**E1** | Expert 权重安装 / **MODEL** | 强制 diagnostic ACC16 grouped-down 路径。它会改变数值结果，不是受支持 precision 配置。 |
| `RPU_L2_RCHUNK` | Runtime fallback 为 `256`；正十进制数；受支持 grouped 配置要求精确 `1632` | Weight/config 校验（**MODEL**）和 Graph emission（**BUILD**）；新进程 | 覆盖 grouped scaling row chunk。错误值会使密封配置失败，或改变 Graph/timing。 |
| `RPU_L2_ROUTED_ONLY` | 关闭；**E1** | Expert 权重安装 / **MODEL** | 隔离 routed-expert contribution 用于 bisection。输出不是完整模型输出。 |
| `RPU_L2_SCHUNK` | `32512`；正十进制数；非正值使用默认值 | Graph emission / **BUILD** | 覆盖 grouped activation host chunk。它会改变 Graph census，并可能降低安全性/性能。 |
| `RPU_LINGBOT2_DEBUG_DENSE_SOFT_ROUTER` | 关闭；**E1** | Expert 权重安装 / **MODEL** | 用 dense soft routing 替换严格 top-4 routing。精度未验证，输出不用于生产。 |
| `RPU_PI05_LOAD_NOISE` | 未设置；现有 tensor `.pt` 的路径；路径不存在时忽略 | Noise preparation / **CALL** | 在严格 shape/finite 检查后使用 `weights_only=True` 替换 sampled noise。只应使用受信任本地文件。 |
| `RPU_PI05_LOAD_PIXEL_VALUES` | 未设置；现有 tensor `.pt` 的路径；路径不存在时忽略 | Image feature 调用 / **CALL** | 在严格检查后替换 processed pixel value。它会改变模型输入，并可能加载敏感测试数据。 |
| `RPU_PI05_LOAD_PREFIX_EMBS` | 未设置；现有 tensor `.pt` 的路径；路径不存在时忽略 | Prefix preparation / **CALL** | 在严格检查后替换 prefix embedding。它会绕过普通 upstream 值，并使 E2E 声明失效。 |
| `RPU_PI05_LOG_CONVERSION` | 未出现时关闭；任意非空值（包括 `0`）都会启用 | Gemma prefill / **CALL** | 记录 conversion/handle/chunk 诊断。增加输出，并可能暴露模型 shape/configuration。 |
| `RPU_RHINOVLA_VISION_RPU_MERGERS_MEM_DEBUG` | **PB(false)** | Vision 安装/materialization / **MODEL** | 在 merger materialization 前后打印 allocator summary。增加同步，并暴露内存结构。 |
| `RPU_SIGLIP_ISOLATE_PATCH_EMBED` | 关闭；除 `0`/`false`/`False` 外的非空值 | 首次 segment 规划 / **NATIVE** | 隔离 Pi0.5 patch-embedding segment，诊断 Graph finalization failure。改变 Graph segmentation。 |
| `RPU_WALL_OSS_INSTRUMENT` | **PB(false)** | Wall-OSS forward stage / **CALL** | 打印 host stage timing。instrumentation 开销会使同一次运行不适合报告干净 latency。 |
| `RPU_WALL_OSS_PROLOGUE_FILLER` | 空；任意字符串 | Prompt 构造 / **CALL** | 向 prompt 追加诊断内容。它会改变 token，使输出不适合正确性声明。 |
| `RPU_WALL_OSS_VISION_DEVICE_MERGED` | **PB(false)** | Vision 构造 / **MODEL** | 将 eager merged result 保留在 RPU 上，作为普通 fused 路径的诊断替代。改变 ownership/topology。 |
| `WALL_OSS_PROFILE_PREFIX` | 未设置；十进制整数 target length | 每次 prediction / **CALL**；prepared production Graph 禁止使用 | 保留 image token 后截断尾部 text，用于 profiling shape。所得 action 没有意义。 |

## 非环境变量 Runtime 控制

应优先使用这些公共 API，而不是增加新的环境开关。

### 冷态、按 handle 生效的 `rpu_execution`

公共 loader 和 policy 接受不可变的 `rpu_execution` mapping。stage 表为
`prefill`、`vision` 和 `action`；支持该能力的入口另有 `model` 控制表；每个入口都会公布其支持的 subset，并在加载权重前拒绝
其他所有 stage 或字段。

| 字段 | 接受值 | 含义与生命周期 |
|---|---|---|
| `model.num_cores` | 整数 `4/6/8` | 仅限入口声明并由模型 profile 准入的冷态计算预算。 |
| `<stage>.linear_acc32` | 布尔值 | 上述组件的 ACC16（`false`，默认）或 ACC32（`true`）；不改变量化权重格式。 |
| `prefill.fast_replay` | 布尔值 | 仅 Qwen3-VL；普通 API 默认 `false`，适用的 FP16/量化示例显式启用。 |
| `chunk_size` | `"auto"` 或 16 的正整数倍 | 指定 stage 的 planner cap/choice。绑定到模型或 policy；改变时应构造新实例。 |
| `padding_rows` | `"auto"` 或非负整数 | 指定 stage 的精确/自动 execution padding。精确整数与 `padding_budget` 互斥。 |
| `padding_budget` | 非负整数 | stage planner 考虑的最大可选 padding。不能与精确整数 `padding_rows` 同时使用。 |

即使某字段只影响一个 Graph，全部三个 stage mapping 仍是冷态配置：应在 RPU 权重安装和
首次 Graph BUILD 前传给 loader/factory。adapter 可以把请求收窄到其精确 capability
envelope；如果公共 policy API 有记录，它也会公开 resolved execution plan。

### 内存与 coherency 控制

- `torch.rpu.set_caching_allocator(bool)` 选择进程 tensor allocator（默认 direct/关闭）。
  第一次显式选择或非空 RPU tensor allocation 会冻结该选择；同值重复调用是幂等的，
  冲突值会报错。切换模式必须使用新进程。`torch.rpu.empty_cache()` 释放已缓存、未使用的
  block；无法释放 live tensor 或 Graph-owned storage。需要保留大量 tensor 的 adapter
  会在首次 RPU allocation 前选择同一个通用 caching policy。
- `torch.rpu.memory_stats()` 和 `torch.rpu.get_memory_stats()` 返回不含地址的
  allocator counter。`caching_allocator_policy_frozen` 表示选择是否已经不可更改。
  当 `caching_allocator_enabled` 为 true 时，
  `caching_allocator_mapping` 只统计该 allocator 持有的 HostDDR segment，
  `cached_idle_mapping` 统计其中完全空闲的 segment；它们不包含 Launch、SPM 或
  direct allocation，因此不是进程级 driver mapping 总数。
  `reset_peak_memory_stats()` 重置 peak counter，`reset_accumulated_memory_stats()` 重置
  累计 allocate/free counter。重置 counter 不会释放内存。
- `torch.rpu.set_ddr_flush(bool)` 控制内部 RPU-to-RPU flush point（默认关闭）。模型
  adapter 可在其 cross-component 契约需要时设置。
- `torch.rpu.set_ddr_flush_force(bool)` 控制 CPU/RPU boundary coherency（默认开启）。
  普通推理应保持启用；除隔离 microbenchmark 外，禁用是不安全的。对应 `get_*` 函数
  报告当前进程状态。

### Profiling 机制

| 机制 | 测量内容 | 启用时机 | 输出/风险 |
|---|---|---|---|
| `torch.rpu.set_profile(True)` | Backend wrapper 和累计 timing counter | 如果只需要 steady-state counter，在 warmup 后启用；测量窗口前调用 `reset_profile_accumulators()` | 轻量 console/counter 诊断；它不是 PyTorch operation trace。 |
| `torch.profiler.profile(...)` | CPU 和 `PrivateUse1` operation scope，包括 Graph capture/replay range | 只包围要测量的调用；按 benchmark protocol 单独包含 warmup | 生成 operation-level trace/table，并增加 profiler 开销。 |

## 清单维护

表格为每个当前具体环境/配置名称保留一行。新增 reader 或 facade preset 时，必须在正确
tier 增加一行；删除最后一个 reader/preset 时，必须删除对应行。自动化文档检查应将行
集合与源码 reader 双向比较，并将诊断变量保持在稳定的 `Diagnostic-only inventory`
heading 之后。
