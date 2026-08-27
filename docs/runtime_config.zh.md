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

完整 runner 将四个配置层分开：

| 表 | 公共字段 | 归属 |
|---|---|---|
| `[runner]` | `target` | 选择 `examples/run_model.py --list-targets` 报告的一个 target。 |
| `[runner.env]` | 本页记录的任意非诊断变量，但不包括 `RPU_KERNEL_LIB_PATH` 和类似凭据的名称 | 在导入 PyTorch 或 RhinoForge 前应用。省略的值继续从 shell 继承。只有在提供完整、已验证配置时才应更改模型配置 selector。 |
| `[runner.torch_profile]` | `enabled`, `output`, `record_shapes`, `profile_memory`, `with_stack` | 配置覆盖整条命令的 Torch trace。 |
| `[rpu_execution.<stage>]` | `chunk_size`；`prefill` 还接受 `padding_rows` 和 `padding_budget` | 冷态、按 handle 生效的规划。下面的 target 矩阵是 allowlist；不支持的 stage 或字段会在配置校验时失败。 |
| `[model]`, `[generation]`, `[request]` | 下文列出的 target 特定字段 | 由所选直接 example 消费。路径和输入仍由部署方负责。 |
| `[runtime]` | 仅限 RhinoVLA factory 定义的字段 | 非空 mapping 会原样传给受信任的外部 runtime factory。RhinoForge 无法安全地发明或校验模型仓库特定的名称。 |

`padding_rows` 可以是 `"auto"` 或精确的非负整数。精确整数和 `padding_budget` 是二选一，
不能同时生效。所有 chunk size 都必须是 `"auto"` 或 16 的正整数倍。loader 或 policy
还可以将每个值进一步收窄到模型允许的配置范围。

`--check-config` 会校验 `[runner.env]` 名称和 TOML 标量类型，但不会重复实现每个变量的
解析器。下表中的值 grammar 和精确配置兼容性仍由构造或 preflight 阶段最终决定。

### 公共和模型特定参数映射

每个模型配置只保留一个模板。同一个文件既可交给 direct example，也包含公共 runner、
profiler 字段和所有适用的 `rpu_execution` stage。

| Target / 模型家族 | 模板 | 接受的规划 stage | 模型和请求字段 | 模型负责的 runtime 分组 |
|---|---|---|---|---|
| `causal_lm` · Qwen3 | [`qwen3_0_6b.toml`](../examples/configs/qwen3_0_6b.toml) | `prefill` | `model.alias` 或 `model.checkpoint`，`model.local_files_only`；`generation.prompt`，`generation.max_new_tokens` | 通用/cache 设置；精确量化配置可以增加其已记录的 selector |
| `causal_lm` · Llama | [`llama_3_2_1b.toml`](../examples/configs/llama_3_2_1b.toml) | `prefill` | 与 CausalLM 相同的字段 | 通用/cache 设置 |
| `qwen3_5_text` | [`qwen3_5_0_8b.toml`](../examples/configs/qwen3_5_0_8b.toml) | `prefill` | 与 CausalLM 相同的字段 | [模型特定公共设置](#model-specific-public-settings)；新集成优先使用 `rpu_execution` |
| `qwen3_5_vision` | [`qwen3_5_vision_2b.toml`](../examples/configs/qwen3_5_vision_2b.toml) | `prefill` | Qwen3.5 模型字段、图像、prompt 和生成长度 | [Qwen3.5 Vision selector](#qwen35-vision-selectors)；状态按精确配置判定 |
| `qwen3_vl` | [`qwen3_vl_2b.toml`](../examples/configs/qwen3_vl_2b.toml) | `prefill`, `vision` | `model.alias` 或 `model.checkpoint`，`model.local_files_only`；`request.image` 和 `request.images` 必须二选一，另含 prompt 和生成长度 | [Qwen3-VL 共享 Vision 配置](#qwen3-vl-shared-vision-profile) |
| `dinov3` | [`dinov3_vit_b.toml`](../examples/configs/dinov3_vit_b.toml) | `vision` | `model.alias` 或 `model.checkpoint`，`model.local_files_only`；`request.image` | 通用/cache 设置 |
| `siglip` | [`siglip.toml`](../examples/configs/siglip.toml) | `vision` | Pi0.5 bundle alias/checkpoint 和一张或多张图像路径 | [Pi0.5 配置](#pi05-profile)；仅 Component-only 输出 |
| `pi05` | [`pi05_libero.toml`](../examples/configs/pi05_libero.toml) | `prefill`, `vision`, `action` | `model.alias` 或 `model.checkpoint`；`request.batch_file`，`request.num_steps`，`request.prepare_graphs` | [Pi0.5 配置](#pi05-profile)和模型特定公共设置 |
| `wall_oss` | [`wall_oss.toml`](../examples/configs/wall_oss.toml) | `prefill`, `vision`, `action` | checkpoint/alias、精度与 FP16 来源、dataset/camera、state/action 字段；images、instruction、proprioception 和 noise | [Wall-OSS 配置](#wall-oss-profile) |
| `hy_embodied` | [`hy_embodied.toml`](../examples/configs/hy_embodied.toml) | 无；非空 `rpu_execution` 会被拒绝 | checkpoint/alias，`dtype`，`prefix_len`；三张 images、instruction、state、noise seed | [Hy-Embodied 配置](#hy-embodied-profile) |
| `gemma4` | [`gemma4.toml`](../examples/configs/gemma4.toml) | `prefill`（仅 `chunk_size`） | checkpoint/alias、local-only、最大序列长度和 Source-only 确认；prompt 和生成长度 | 通用/cache 设置；Source-only |
| `gr00t` | [`gr00t.toml`](../examples/configs/gr00t.toml) | `prefill`, `vision`, `action` | GR00T 与 Qwen3-VL checkpoint/alias、embodiment ID 和 Source-only 确认；调用方预处理 tensor 文件、seed 和 step 数 | [GR00T 配置](#gr00t-profile)；Source-only |
| `lingbot2` | [`lingbot2.toml`](../examples/configs/lingbot2.toml) | 无 | checkpoint/alias、精确 dtype/camera/image/token 配置和受控确认；三张 images、instruction、proprioception 和 seed | [LingBot-VLA-V2 配置](#lingbot-vla-v2-profile) |
| `g05` | [`g05.toml`](../examples/configs/g05.toml) | 无 | checkpoint/alias、最大序列长度和受控确认 | 仅集成检查；由官方模型仓库构造 policy |
| `internvla_navdp` | [`internvla_navdp.toml`](../examples/configs/internvla_navdp.toml) | 无 | 根 alias/checkpoint、NavDP checkpoint、SHA256 manifest 和受控确认；调用方 embeddings、sample 数和 seed | [InternVLA-N1 与 NavDP 配置](#internvla-n1-and-navdp-profiles)；仅 component |
| `rhinovla` | [`rhinovla.toml`](../examples/configs/rhinovla.toml) | 仅限受信任 runtime factory 声明的 stage 和字段 | `model.checkpoint`，`model.runtime_factory`；`request.request_json`；factory 定义的 `[runtime]` | [RhinoVLA 配置](#rhinovla-profile)和外部 factory 契约 |

下面穷举记录模型负责的 runtime 分组，因为它们确实是源码读取项；但其中大多数是密封的
配置 selector，而不是可以独立调优的用户旋钮。因此，模板展示稳定的公共配置面，
不会为每个 selector 猜测一个值后全部启用。把所有 selector 默认值复制到 TOML 中
可能覆盖 facade 负责的配置，并悄然产生未验证组合。

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
| `QWEN3_VL_32B_ALLOW_GRAPH_BLOCKED` | `0`；**E1** | Preflight / **MODEL** | 仅用于精确 Qwen3-VL 32B W8A16 graph-blocked 评估。 |
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
| `RPU_FASTREPLAY_SKIP_SYNC` | 关闭；不以 `0` 开头的非空值会启用 | Graph BUILD / **BUILD** | 仅在完全跳过的 replay 中省略冗余 mutable-parameter scan。错误使用可能 replay 过期参数。 |
| `RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN` | 关闭；精确 `1`、`true`、`True` 或 `on` 会启用 | 首次原生 coexistence 使用 / **NATIVE** | 在配置负责的 subsystem handoff 之间保留 persistent SPM generation。归属错误可能破坏后续执行。 |
| `RPU_GRAPH_DEFER_TO_COPY` | 没有独立默认值；legacy alias 接受 `auto`/空、`0`/`off`/`false`，其他任意非空值强制开启 | 首次 host-op gate 使用 / **NATIVE** | 仅当未设置 `RPU_GRAPH_HOST_OP_DEFER_GATE` 时查询。避免同时设置两者。 |
| `RPU_GRAPH_HOST_OP_DEFER_GATE` | `auto`；`auto`/空、`0`/`off`/`false`，其他任意非空值强制开启 | 首次 host-op gate 使用 / **NATIVE** | 控制 capture 期间 stable host input 的 deferral。对不稳定 storage 强制开启会产生过期数据。 |
| `RPU_SKIP_IDLE_RECORD_FUNCTION` | 关闭；`1`/`on`/`true`/`True` 会启用 | 首次 Python graph scope / **MODEL** | 只在 profiler 关闭时省略 idle profiler scope。profiling 时保留 scope，其他情况下减少 host 开销。 |

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
| `RPU_RMSNORM_VWARP` | `0`；`0`、`16`、`32` 或 `auto`；未知文本回退到 `0` | 首次原生使用 / **NATIVE** | 选择允许的 vector row schedule。不支持的 divisibility 会回退；timing 会变化。 |

<a id="qwen35-vision-selectors"></a>
### Qwen3.5 Vision selector

这些 selector 只在 numeric-blocked 受控评估门允许 Qwen3.5 Vision 后生效，
不会扩大其 Experimental 状态。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `QWEN3_5_VISION_CHUNK` | Auto；十进制整数；空值/非正值表示 auto | 首次原生 vision plan / **NATIVE** | 限制 Qwen3.5 Vision chunk。正数 override 必须仍处于允许的 shape 和 memory envelope 内。 |
| `RPU_VISION_MERGER` | **PB(true)** | Qwen3.5 Vision setup / **MODEL** | 启用 RPU merger 路径。`0` 选择 fallback，并改变性能/数值。 |
| `RPU_VISION_STEP0` | **PB(true)** | Qwen3.5 Vision setup / **MODEL** | 启用 RPU first-stage 路径。`0` 选择 fallback，并改变性能/数值。 |

<a id="gr00t-profile"></a>
## GR00T 配置

公共 GR00T builder 会在构造前提供其 facade 默认值。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_GR00T_KVPAD16` | 原始默认关闭；facade 为 `1`；原生解析器中 `1` 或以 `t`/`T` 开头的值会启用 | 首次原生 KV setup / **NATIVE** | 为 GR00T expert padding 并 mask KV row。它会改变 cache layout 和 Graph signature。 |
| `RPU_GR00T_PARTIAL_MROPE` | 原始为 **PB(false)**；facade 为 **PB(true)** | GR00T 构造 / **MODEL** | 使用预计算的 partial multimodal position table。 |
| `RPU_GR00T_W8A16` | **PB(false)** | GR00T 权重安装 / **MODEL** | 量化允许的 backbone 和 action projection，并改变数值。 |

<a id="hy-embodied-profile"></a>
## Hy-Embodied 配置

`HyEmbodiedPolicy` 会在构造前应用完整的 scoped 配置。下面的原始默认值描述直接使用
adapter 的行为；行中列出的 facade 默认值是普通公共 policy 值。切换 policy 配置时应
使用新进程。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_HY_VLA_ACTION_MLP_MC` | 原始未设置表示 8 cores；`0`/`off`/`false`/`False` 表示 1，其他所有值表示 8；facade 为 `1` | Weight/build setup / **MODEL** | 选择 multi-core action-MLP 执行。它会改变 reduction order，且必须继续满足配置数值门禁。 |
| `RPU_HY_VLA_ATTN_TP8` | 原始默认关闭；`1`/`true`/`True`/`on`；facade 为 `1` | 权重安装 / **MODEL** | 为 eight-core attention 复制 KV head。必须重建 cache geometry 和权重。 |
| `RPU_HY_VLA_CACHING_ALLOC` | 开启；`0`/`false`/`False`/`off` 会禁用 | Hy-VLA 构造 / **MODEL** | 为该 policy 启用进程 caching allocator。它会改变内存保留；测量 free memory 前调用 `empty_cache()`。 |
| `RPU_HY_VLA_DENOISE_UNROLL` | 原始默认关闭；`1`/`true`/`True`/`on`；facade 为 `1` | Action 构造 / **MODEL** | 将 denoise loop 记录为一个配置负责的 Graph。它会改变 Euler-state precision 和数值结果。 |
| `RPU_HY_VLA_FAST_REPLAY` | 原始默认关闭；`0`/`off`/`false`、`1`/`on`/`true`，或命名 `vit`、`vlm` 和/或 `expert` 的值；facade 为 `1` | 首次原生 subsystem 使用 / **NATIVE** | replay 时跳过已审计的 layer-body emission。错误使用可能使 dynamic input 过期。 |
| `RPU_HY_VLA_FAST_REPLAY_PRELOAD` | 原始默认关闭；`0`/`off`/`false`、`1`/`on`/`true`，或命名 `vit`、`vlm` 和/或 `expert` 的值；facade 为 `1` | 首次原生 subsystem 使用 / **NATIVE** | replay 时也跳过已审计的 preload 工作。需要稳定的 persistent state。 |
| `RPU_HY_VLA_FUSED_MERGER` | 原始默认关闭；Python 接受 `1`/`true`/`True`/`on`；原生镜像要求字面值 `0`/`1`；facade 为 `1` | Vision 构造 / **MODEL** | 将 vision merger 折叠进 Graph。只使用 `0`/`1` 可避免解析器不一致。 |
| `RPU_HY_VLA_KVPAD16` | 原始默认关闭；`0`/`off`/`false`、`1`/`on`/`true`，或命名 `vit`、`vlm` 和/或 `expert` 的值；facade 为 `1` | 首次原生 subsystem 使用 / **NATIVE** | 对 expert KV row 进行 padding/mask。它会改变 cache layout 和 signature。 |
| `RPU_HY_VLA_MASK_ONCE` | 原始默认关闭；`0`/`off`/`false`、`1`/`on`/`true`，或命名 `vit`、`vlm` 和/或 `expert` 的值；facade 为 `1` | 首次原生 subsystem 使用 / **NATIVE** | 复用 invariant mask upload。mask 内容或 storage 可能变化时不安全。 |
| `RPU_HY_VLA_MERGER_IN_GRAPH` | 原始默认关闭；`1`/`true`/`True`/`on`；facade 为 `1` | Vision 构造 / **MODEL** | 将启用的 fused merger 放进 capture。要求 `RPU_HY_VLA_FUSED_MERGER=1`。 |
| `RPU_HY_VLA_MOT_NORM_NOMERGE` | `both`；`0`/`off`/`false`、`1`/`on`/`true`/`both`，或包含 `qkv` 和/或 `mlp` 的值 | 首次原生 VLM 使用 / **NATIVE** | 选择配置特定的 norm/merge scheduling。它会改变 temporary-memory 和 Graph 结构。 |
| `RPU_HY_VLA_PARTIAL_ROPE` | 原始默认关闭；`0`/`off`/`false`、`1`/`on`/`true`，或 CSV scope `vit`、`vlm`、`expert`；facade 为 `expert,vlm` | 首次原生 subsystem 使用 / **NATIVE** | 为指定 subsystem 使用预计算的 partial position table。scope 不匹配会改变 position semantics。 |
| `RPU_HY_VLA_PATCH_EMBED_IN_GRAPH` | 开启；`0`/`false`/`False`/`off` 会禁用；facade 为 `1` | Vision 构造 / **MODEL** | capture device patch embedding。禁用会选择不同的 host/device boundary。 |
| `RPU_HY_VLA_PATCH_EMBED_MC` | 原始未设置表示 8 cores；`0`/`false`/`False` 表示 1；facade 为 `1` | 权重和 Vision 构造 / **MODEL** | 选择 multi-core patch embedding。Python/native state 必须一致；需重建权重和 Graph。 |
| `RPU_HY_VLA_PERSIST_HANDLES` | 原始默认关闭；`0`/`off`/`false`/空，`1`/`on`/`true` 表示全部，或 CSV subset `vit`、`vlm`、`expert`；facade 为 `1` | Policy 构造 / **MODEL** | 跨调用保持指定 handle 存活。会增加保留内存，并实施单 policy 归属。 |
| `RPU_HY_VLA_PREFIX_TEMPLATE` | 开启；`0`/`false`/`False`/`off` 会禁用 | Prompt 构造 / **MODEL** | 启用配置 prompt template。改变它会改变 token input，而不只是性能。 |
| `RPU_HY_VLA_PROJ1_IN_MERGER` | 开启；`0`/`off`/`false`/`False` 会禁用；原生镜像使用同一变量 | Vision 构造 / **MODEL** | 在 merger 路径中包含 first projection。使用字面值 `0`/`1`；状态不一致会破坏 output shape/ownership。 |
| `RPU_HY_VLA_Q_INPLACE` | 原始默认关闭；除 `0`/`off`/`false` 外的非空值；facade 为 `1` | 首次原生 Vision 使用 / **NATIVE** | 原地复用 query buffer。仅当配置证明旧值已死亡时才安全。 |
| `RPU_HY_VLA_RMSNORM_PAD16` | 原始默认关闭；`0`/`off`/`false`、`1`/`on`/`true`，或命名 `vit`、`vlm` 和/或 `expert` 的值 | 首次原生 subsystem 使用 / **NATIVE** | 对指定 normalization row 做 padding。它会改变 layout 和 Graph census。 |
| `RPU_HY_VLA_SILU_MUL` | 原始默认关闭；`0`/`off`/`false`、`1`/`on`/`true`，或 CSV scope `vit`、`vlm`、`expert`；facade 为 `expert` | 首次原生 subsystem 使用 / **NATIVE** | 融合指定的 activation/multiply schedule。每个 scope 都需要数值验证。 |
| `RPU_HY_VLA_VIT_PACKED` | 原始默认关闭；`1`/`true`/`True`/`on`；facade 为 `1` | Vision 构造 / **MODEL** | 打包 vision-tower 执行配置。它会改变 shape/layout 契约，并要求重建权重/Graph。 |
| `RPU_HY_VLA_W4A16` | 关闭；`0`/`off`/`false`/`False`/空，`1`/`on`/`true`/`True`/`all`，或 CSV subset `vit`、`vlm`、`expert`、`vlm_text`、`vlm_vision` | 权重安装 / **MODEL** | 按 scope 选择 W4A16；重叠时 W4 优先于 W8，不支持的 scope 会失败。数值配置会改变。 |
| `RPU_HY_VLA_W8A16` | 原始默认关闭；facade 映射 `fp16`→`0`、`w8a16`→`all`、`w8a16-expert`→`expert`、`w8a16-expert-vlmv`→`expert,vlm_vision`、`w8a16-vlm`→`vlm`、`w8a16-vit`→`vit`、`w8a16-no-vlm`→`vit,expert`；也接受 W4 行的精确 enum/CSV grammar | 权重安装 / **MODEL** | 按 scope 选择 W8A16。它必须匹配 checkpoint/source 权重和指定 precision 配置。 |

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

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_QWEN3VL_VISION_BATCH` | **PB(false)** | Vision forward / **CALL**；重建 Graph | 打包兼容 image/view。它会改变 Graph signature 和内存；facade 约束仍会限制 shape。 |
| `RPU_QWEN3VL_VISION_BATCH_CAP` | `3`；十进制整数，至少收窄为 `1` | Vision forward / **CALL**；重建 Graph | 每个 packed group 的最大兼容 image 数。更大值可能超过配置内存 envelope。 |
| `RPU_QWEN3VL_VISION_FUSED_MERGER` | 原始默认关闭；**B01(false)** | Vision 构造 / **MODEL** | 将 merger 折叠进 Vision Graph。只有字面值 `0`/`1` 能保持 Python 与原生 reader 一致。 |
| `RPU_QWEN3VL_VISION_HOST_FP32_PATCH` | `0`；**E1** | Vision forward / **CALL**；重建 Graph/model | 使用 host FP32 patch projection，而不是默认 FP16/device 配置。它会改变数值和 transfer cost。 |
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

<a id="wall-oss-profile"></a>
## Wall-OSS 配置

`WallOssPolicy` 会在模型构造前提供经过验证的 facade 默认值。prompt selector 会改变
实际模型输入，绝不能当作仅影响性能的开关。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_WALL_OSS_ACTION_FP32_TAIL` | **PB(false)** | 每次 denoise 调用 / **CALL**；重建 prepared Graph 配置 | Euler tail 使用 host FP32。它会改变 action 数值，并与 fused/unrolled 执行冲突。 |
| `RPU_WALL_OSS_ATTN_TP8` | 原始为 **PB(false)**；facade 为 `1` | 权重安装 / **MODEL** | 为 eight-core attention 复制 KV head。重建权重、cache 和 Graph。 |
| `RPU_WALL_OSS_BATCH_MAX_SEQ` | `768`；十进制整数 | Vision grouping / **CALL**；重建 Graph | 限制 packed vision sequence length。过大可能超出内存；所选公开配置可能进一步约束。 |
| `RPU_WALL_OSS_BATCH_VISION` | **PB(true)** | Vision forward / **CALL**；重建 Graph | 在上限内打包 equal-grid image。它会改变 signature 和内存。 |
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
| `RPU_WALL_OSS_VISION_LAYER_GROUP` | `0`；精确 `0` 或 `32` | Vision 构造（**MODEL**）和首次原生 schedule claim（**NATIVE**）；新进程 | 受限精确 three-image scheduling 实验；与 per-window SDPA 不兼容。 |
| `RPU_WALL_OSS_VISION_ROPE_SPM` | 原始默认关闭；**N1**；facade 为 `1` | 首次原生 Vision 使用 / **NATIVE** | 将 position table 保存在 SPM。会改变 persistent-memory 使用，并要求重建 Vision handle。 |

## 仅诊断项清单

本节中的任何变量出现在 TOML 中时，完整公共 runner 都会拒绝。开发者仍可在 shell 中
显式设置某一变量，用于受控本地调查。这些控制项可能改变执行、dump 模型数据、暴露 tensor summary
或 timing 结构，或者使正确性/性能声明失效。生产环境中应保持未设置，分享前审查每个
生成的 artifact。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `QWEN3_5_VISION_DBG_Q` | 关闭；精确 `1` 或小写 `true` | 首次原生 debug 检查 / **NATIVE** | 启用 Qwen3.5 Vision query probe 路径。会增加 capture/同步，并可能暴露 intermediate。 |
| `QWEN3_5_VISION_GRAPH_DISABLE` | `0`；除精确 `0` 外的任意值都会禁用 | Vision forward / **CALL**；重建/清除 Graph | 绕过 Qwen3.5 Vision GraphCache，用于受控比较。replay 和 latency 结论不再适用。 |
| `RPU_ALL_GATHER_FORCE_MULTI_CORE` | 关闭；**N1** | 首次原生使用 / **NATIVE** | 强制 multi-core schedule，用于 A/B 比较。它不是普遍支持的性能 selector。 |
| `RPU_CHUNK_FORCE_UNSAFE` | 关闭；**N1** | 首次适用的原生 planner/launcher 使用 / **NATIVE** | 绕过 planner safety check。可能超过执行约束，绝不能产出可部署输出。 |
| `RPU_DYNAMO_MATERIALIZE_BREAKS` | **PB(false)** | Dynamo partitioning / **CALL**；重新编译 | materialize partition break。它会改变 Graph boundary 并增加 transfer，仅用于 compiler 诊断。 |
| `RPU_GRAPH_DDR_SPM_LOG` | 关闭；首字符不是 `0` 的非空值 | 首次 data-node 执行 / **NATIVE** | 记录 node index、byte count 和有界 source checksum。输出由模型数据派生，且日志会改变 timing。 |
| `RPU_GRAPH_FORCE_ONESHOT_ON_REPLAY` | 关闭；首字符不是 `0` 的非空值 | 首次 replay 检查 / **NATIVE** | 执行 one-shot 路径而不是普通 replay。会使 Graph lifecycle 和性能结论失效。 |
| `RPU_GRAPH_HCB_CHECKSUM` | 关闭；首字符不是 `0` 的非空值 | 首次 host-callback 执行 / **NATIVE** | 记录 live/stable tensor metadata 和有界 value summary。将输出视为敏感模型数据。 |
| `RPU_KVINSERT_V16_TRACE` | 关闭；除精确 `0` 或 `false` 外，任何出现的值都会启用；export 的空值也会启用 | 首次原生使用 / **NATIVE** | 记录 KV route selection 和 shape。它是诊断输出，不是 cache 正确性证明。 |
| `RPU_L2_BUFONLY` | 未出现时关闭；任意出现（包括 `0`）都会启用 | Grouped-expert Graph emission / **BUILD** | 声明 grouped buffer，但运行 per-expert 路径进行 bisection。会改变 Graph 和性能。 |
| `RPU_L2_CAPTURE_GATE` | 关闭；**E1** | Expert 权重安装 / **MODEL** | 分配并导出 gate 相关 layer-0 intermediate。增加内存/Graph 工作，并暴露模型数据。 |
| `RPU_L2_CAPTURE_INNORM` | 关闭；**E1** | Expert 权重安装 / **MODEL** | capture layer-0 post-normalization input。tensor 可能包含用户派生的 activation。 |
| `RPU_L2_CAPTURE_L0` | 关闭；**E1** | Expert 权重安装 / **MODEL** | capture layer-0 output 用于比较。增加 persistent storage 和一次 copy。 |
| `RPU_L2_CAP_RESID` | 关闭；**E1** | Expert 权重安装 / **MODEL** | 跨 layer capture residual-stream tensor。内存成本高；输出可能包含请求派生的 activation。 |
| `RPU_L2_DBG_PACKED` | 关闭；每次 debug getter 都检查 **E1** | Debug getter / **CALL** | 解锁 packed 模型权重供本地检查。绝不能公开返回 tensor。 |
| `RPU_L2_DOWN_ACC16` | 关闭；**E1** | Expert 权重安装 / **MODEL** | 强制 diagnostic ACC16 grouped-down 路径。它会改变数值结果，不是受支持 precision 配置。 |
| `RPU_L2_RCHUNK` | Runtime fallback 为 `256`；正十进制数；受支持 grouped 配置要求精确 `1632` | Weight/config 校验（**MODEL**）和 Graph emission（**BUILD**）；新进程 | 覆盖 grouped scaling row chunk。错误值会使密封配置失败，或改变 Graph/timing。 |
| `RPU_L2_ROUTED_ONLY` | 关闭；**E1** | Expert 权重安装 / **MODEL** | 隔离 routed-expert contribution 用于 bisection。输出不是完整模型输出。 |
| `RPU_L2_SCHUNK` | `32512`；正十进制数；非正值使用默认值 | Graph emission / **BUILD** | 覆盖 grouped activation host chunk。它会改变 Graph census，并可能降低安全性/性能。 |
| `RPU_L2_STAGES` | 关闭；**E1** | Expert 权重安装 / **MODEL** | capture intermediate grouped stage。增加内存/copy，并暴露 activation。 |
| `RPU_LINGBOT2_DEBUG_DENSE_SOFT_ROUTER` | 关闭；**E1** | Expert 权重安装 / **MODEL** | 用 dense soft routing 替换严格 top-4 routing。精度未验证，输出不用于生产。 |
| `RPU_LINGBOT2_DEBUG_DUMP_ROUTER_H` | 关闭；**E1** | Expert 权重安装 / **MODEL** | capture per-layer router input。dump 可能包含请求派生 activation，并占用大量内存。 |
| `RPU_PI05_LOAD_NOISE` | 未设置；现有 tensor `.pt` 的路径；路径不存在时忽略 | Noise preparation / **CALL** | 在严格 shape/finite 检查后使用 `weights_only=True` 替换 sampled noise。只应使用受信任本地文件。 |
| `RPU_PI05_LOAD_PIXEL_VALUES` | 未设置；现有 tensor `.pt` 的路径；路径不存在时忽略 | Image feature 调用 / **CALL** | 在严格检查后替换 processed pixel value。它会改变模型输入，并可能加载敏感测试数据。 |
| `RPU_PI05_LOAD_PREFIX_EMBS` | 未设置；现有 tensor `.pt` 的路径；路径不存在时忽略 | Prefix preparation / **CALL** | 在严格检查后替换 prefix embedding。它会绕过普通 upstream 值，并使 E2E 声明失效。 |
| `RPU_PI05_LOG_CONVERSION` | 未出现时关闭；任意非空值（包括 `0`）都会启用 | Gemma prefill / **CALL** | 记录 conversion/handle/chunk 诊断。增加输出，并可能暴露模型 shape/configuration。 |
| `RPU_PI05_PROBE_DIR` | 未设置；非空目录路径 | Pi0.5 conversion 和 forward / **CALL** | 将具名 input、activation 和 KV tensor dump 为 `.pt`。artifact 可能包含模型权重和用户数据。 |
| `RPU_RHINOVLA_VISION_RPU_MERGERS_MEM_DEBUG` | **PB(false)** | Vision 安装/materialization / **MODEL** | 在 merger materialization 前后打印 allocator summary。增加同步，并暴露内存结构。 |
| `RPU_SIGLIP_ISOLATE_PATCH_EMBED` | 关闭；除 `0`/`false`/`False` 外的非空值 | 首次 segment 规划 / **NATIVE** | 隔离 Pi0.5 patch-embedding segment，诊断 Graph finalization failure。改变 Graph segmentation。 |
| `RPU_WALL_OSS_INSTRUMENT` | **PB(false)** | Wall-OSS forward stage / **CALL** | 打印 host stage timing。instrumentation 开销会使同一次运行不适合报告干净 latency。 |
| `RPU_WALL_OSS_PROLOGUE_FILLER` | 空；任意字符串 | Prompt 构造 / **CALL** | 向 prompt 追加诊断内容。它会改变 token，使输出不适合正确性声明。 |
| `RPU_WALL_OSS_VISION_DEVICE_MERGED` | **PB(false)** | Vision 构造 / **MODEL** | 将 eager merged result 保留在 RPU 上，作为普通 fused 路径的诊断替代。改变 ownership/topology。 |
| `WALL_OSS_PROFILE_PREFIX` | 未设置；十进制整数 target length | 每次 prediction / **CALL**；prepared production Graph 禁止使用 | 保留 image token 后截断尾部 text，用于 profiling shape。所得 action 没有意义。 |

## 非环境变量 Runtime 控制

应优先使用这些公共 API，而不是增加新的环境开关。

### 冷态、按 handle 生效的 `rpu_execution`

公共 loader 和 policy 接受不可变的 `rpu_execution` mapping。三个可能的 stage 是
`prefill`、`vision` 和 `action`；每个入口都会公布其支持的 subset，并在加载权重前拒绝
其他所有 stage 或字段。

| 字段 | 接受值 | 含义与生命周期 |
|---|---|---|
| `chunk_size` | `"auto"` 或 16 的正整数倍 | 指定 stage 的 planner cap/choice。绑定到模型或 policy；改变时应构造新实例。 |
| `padding_rows` | `"auto"` 或非负整数 | 指定 stage 的精确/自动 execution padding。精确整数与 `padding_budget` 互斥。 |
| `padding_budget` | 非负整数 | stage planner 考虑的最大可选 padding。不能与精确整数 `padding_rows` 同时使用。 |

即使某字段只影响一个 Graph，全部三个 stage mapping 仍是冷态配置：应在 RPU 权重安装和
首次 Graph BUILD 前传给 loader/factory。adapter 可以把请求收窄到其精确 capability
envelope；如果公共 policy API 有记录，它也会公开 resolved execution plan。

### 内存与 coherency 控制

- `torch.rpu.set_caching_allocator(bool)` 切换进程 caching allocator（默认关闭）。
  `torch.rpu.empty_cache()` 释放已缓存、未使用的 block；无法释放 live tensor 或
  Graph-owned storage。
- `torch.rpu.memory_stats()` 和 `torch.rpu.get_memory_stats()` 返回 allocator counter。
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
