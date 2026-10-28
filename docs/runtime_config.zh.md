# Runtime 配置

简体中文 | [English](runtime_config.md)

> 英文版是机器校验和配置契约的事实源。中文版完整翻译用户说明；变量名、默认值、
> accepted grammar、读取阶段、作用域与风险必须与英文版保持一致。

RhinoForge 提供少量面向用户的设置，以及数量更多的精确模型配置和诊断控制项。
请在导入 `rpu_backend` 前设置完整环境，按表中生命周期更改环境变量。支持热修改的 stage 通过下述执行 API 更新。

本页记录 `python/rpu_backend` 和 `src` 中当前保留的所有环境变量读取项，以及 Launch
容量输入。

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

## 配置与执行

统一入口为 `python -m examples.run --config PATH`。配置结构、模型目录与输入要求见[模型示例](../examples/README.md)。通用模型使用 `[example]/[input]/[run]`，专用策略配置使用对应模板的模型表。`example.opt_in` 在导入 Torch 前绑定配置允许的控制；继承环境冲突会被拒绝。`[cpu].threads` 在首次 Torch 导入前生效，线程别名必须取值一致。

运行时资产通过 `RPU_KERNEL_LIB_PATH` 与 `RPU_KERNEL_LIB_PATH_EXPANSION` 分别指定主库与扩展库。资产路径、凭据和本机信息只放在部署环境中。`rpu_execution` 管理模型与组件配置。冷态设置需新模型/进程；支持热修改的 stage 在请求之间通过[执行 API](api_reference.zh.md#执行配置与运行中重配)更新。

## Host 执行设置

所有统一 runner 入口的 `run.warmup` 默认值都是两次推理调用。对延迟捕获的固定
signature，这覆盖 BUILD 和一次 REPLAY。显式次数（含冷路径诊断的零次或一次）原样
执行，不插入隐藏调用。每次预热耗时保存在 `warmup_wall_ms_runs`，与 `wall_ms_runs`
分开。`warmup >= 2` 时，`warmup_decode_steps` 必须省略或等于 `input.decode_steps`；
缩短预热会在加载模型前被拒绝。只有显式 `warmup=0` 或 `1` 才允许缩短 decode 的冷路径诊断。

统一 examples runner 在 `no_grad()` 和 `inference_mode(False)` 下准备所有模型，
使持久权重和可变 Graph 输入保留版本计数；`run.inference_mode` 仍控制 warmup
和计时推理。延迟安装组件也需要局部使用相同的准备上下文。依赖 tensor 版本的缓存
遇到 inference tensor 时，应刷新派生数据或比较独立保存的内容；不能把同一个
tensor 对象当作内容未变的证明。

这些参数属于应用进程及其安装的 CPU runtime，不属于 设备 planner。标准 `[pi05]`
配置通过 `run.py` 运行时，都会在导入 Torch 前绑定下列 allocator
值，`run.torch_num_threads` 默认八线程。库内推理不会修改这些进程设置；更改后使用
新进程。

| Variable | Pi 建议值 | 读取 / 更改 | 作用与影响 |
|---|---:|---|---|
| `OMP_NUM_THREADS` | `8` | CPU runtime 初始化 / **IMPORT** | 限制 OpenMP host 工作线程，实际效果取决于 Torch 构建；显式 Torch 线程设置优先。 |
| `MKL_NUM_THREADS` | `8` | CPU runtime 初始化 / **IMPORT** | 安装的 CPU runtime 使用 MKL 时限制其线程数。 |
| `MIMALLOC_PURGE_DELAY` | `1000` | mimalloc 初始化 / **IMPORT** | 对齐上游 Pi host 设置；在含 mimalloc 的构建中延迟释放空闲页，可能增加保留内存。 |
| `MIMALLOC_ARENA_PURGE_MULT` | `0` | mimalloc 初始化 / **IMPORT** | 对齐上游 Pi host 设置；调整 arena purge 策略，具体行为取决于 runtime 版本。 |

Qwen3、Qwen3-VL、Qwen3.5 和 Pi0.5 接受冷态 `model.num_cores` 与分组件
`linear_acc32`。核数是整数 `4/6/8`，实际准入仍由 checkpoint、量化与输入 profile
收窄；Pi 优化配置使用八核，通用 Pi 缩核配置与量化 Qwen 配置按各自明确准入的范围选择。`linear_acc32` 只接受布尔值，
`false`（默认）选择 ACC16，`true` 选择 ACC32。Qwen3、Qwen3.5 文本入口在 `prefill`
设置，图文入口还可在 `vision` 设置，Pi 分别在 `prefill/vision/action` 设置。
Qwen3-VL 另接受冷态布尔值 `prefill.fast_replay`，不接受 `vision.fast_replay`
或独立 `decode` 表。普通 FP16 和量化 API 入口省略时默认 `false`；适用的 FP16
和量化示例模板显式设为 `true`。专用 profile 策略及原生 owner/Graph 检查仍具最终决定权。配置在安装权重前绑定；改变后应关闭模型、重新加载。

`model.num_cores` 省略时为八核。获准使用缩核的 dense Qwen3、Qwen3-VL、Qwen3.5 配置，四核档的 attention/MLP/head 均使用四核；六核档的 attention/head 使用四核、MLP 使用六核，图像塔使用四核。通过模型的 cache factory 创建缓存，以匹配已选拓扑。具体模型大小与量化变体见[配置指南](../examples/configs/CONFIGURATIONS.md)。

模型开关放在对应配置或启动进程中。显式环境值优先于 adapter 的 `setdefault`；示例 runner 在导入 adapter 前应用所选配置。日志、分配器及 Launch 初始化设置须在导入 backend 前准备。使用模板的 opt-in 与默认值，避免把模型优化开关导出到所有进程。

Qwen3.5-35B-A3B mixed-E4M3 权重按
[官方源下载与 CPU 转换命令](../examples/configs/qwen3_5/README.md#cmd-moe-mixed)准备。
Text/Vision、核数与 ACC 变体共用转换产物，运行参数不改变权重配方。

## 通用和 cache 设置

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_BACKEND_SO` | 默认未设置；原生扩展文件路径 | 扩展解析 / **IMPORT** | 优先于包内和源码构建候选；文件必须存在并匹配当前 Python/PyTorch ABI，只指定可信构建。 |
| `RPU_KERNEL_LIB_PATH` | 主库 `rhinoOpLib_current.ref` 路径及相邻 `.kernels` | 首次访问 / **IMPORT** | 选择匹配源码的主算子资产。 |
| `RPU_KERNEL_LIB_PATH_EXPANSION` | 扩展库路径；省略时取主库旁的 `rhinoExpansionOpLib_current.ref` | 首次访问 / **IMPORT** | 选择配套扩展资产及相邻 `.kernels`。 |
| `RPU_MODEL_CACHE` | `~/.cache/rhinoforge/models`；目录路径 | 为 HF 默认值执行包 bootstrap / **IMPORT**，alias 解析 / **CALL** | RhinoForge 模型 alias 的根目录。显式值也会提供 HF cache 默认值；导入后修改会影响之后的 alias，但无法可靠地重新配置已导入的 HF 组件。 |
| `RPU_LOG_LEVEL` | `3`；十进制整数 `0..5`；无效值会告警并回退到 `3` | 原生扩展加载 / **IMPORT** | 进程日志级别：`0` 静默，最高 `5` trace。较高级别会增加输出，并可能暴露路径或请求 metadata。 |
| `RPU_WARMUP` | `0`；整数；无效值变为 `0`，负数收窄为 `0` | Adapter 构造 / **MODEL** | 支持 warmup 的 adapter 的 warmup forward 次数。会增加启动工作，并可能消耗诊断预算。 |
| `RPU_CAUSAL_PREFILL_PADDING_BUDGET` | `64`；非负十进制整数 | 符合条件的 causal prefill 规划 / **CALL**；新 plan 需重建 Graph | 通用 Qwen3/Llama planner 可选 padding 的最大行数。`0` 禁用可选 padding；plan 变化会改变 Graph signature。 |
| `LKN_MAX_BATCH_ENTRIES` | `65536`；正十进制数；无效/空/零使用默认值；上限 `4194304` | Launch 初始化与 Graph 规划 / **IMPORT** | batch submission 的进程容量。过小会拒绝 Graph；过大则预留更多 host 内存。 |
| `LKN_KD_BUF_MB` | `8`；正 MiB；无效/空/零使用默认值；上限 `256` | Launch 初始化与 Graph 规划 / **IMPORT** | 进程 command-buffer 容量。过小会拒绝 Graph；过大则增加内存占用。 |
| `LKN_INSTR_BUF_MB` | `64`；正 MiB；无效/空/零使用默认值；上限 `1024` | Launch 初始化与 Graph 规划 / **IMPORT** | 进程 execution-buffer 容量。过小会拒绝 Graph；过大则增加内存占用。 |
| `HF_HOME` | `<model-cache>/.hf`；目录路径 | 包 bootstrap 与 HF 初始化 / **IMPORT** | HF cache 根目录；保留显式设置的值。 |
| `HF_HUB_CACHE` | Hugging Face 默认值；目录路径 | 包 bootstrap 与 HF 初始化 / **IMPORT** | 标准 Hub cache；未设置时默认为 `<HF_HOME>/hub`。 |
| `HUGGINGFACE_HUB_CACHE` | Hugging Face 默认值；目录路径 | 包 bootstrap 与 HF 初始化 / **IMPORT** | 兼容性 Hub cache 变量，处理方式与 `HF_HUB_CACHE` 相同；两者应保持一致。 |

导入 backend 前应设置三个 `LKN_*` 容量值。facade 可在首次导入 adapter 前选择更大的
默认值，但无法调整已经加载的 Launch runtime。增大容量不会让不支持的模型配置获得支持。

`RPU_LOG_LEVEL` 在导入时控制 backend 日志，`LKN_LOG_LEVEL` 单独控制 Launch 日志；
启动输出需在导入前配置。运行时 `set_debug_level()` 只改变 backend 日志。
fork 子进程继承当前级别；spawn 子进程读取环境中的 `RPU_LOG_LEVEL`，传播运行时修改前
需先更新该变量。

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
加载权重前校验，并以不可变快照暴露；支持的修改通过执行 API 替换快照。

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

Residual reduction 不提供公开 route、chunk、ring 或 two-stage selector。普通 FP16 ring
使用已安装的消费核数，默认八核；缩核配置使用绑定的拓扑。Local-SPM、mixed-precision
和固定布局路线仍限定八核。Partial producer 先清空所有消费分片，再写有效前缀。见
[自动 residual reduction](../knowledge/concepts/allreduce-routing.md)。

### Graph 与 replay

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_DEEP_FAST_REPLAY` | 关闭；首字符不是 `0` 的任意非空值会启用 | Fused handle 构造 / **MODEL** | full-body replay 时跳过已审计的 setup 工作。配置必须证明每个被跳过的 input/layout 仍有效。 |
| `RPU_FASTREPLAY_SKIP_SYNC` | 开启；空值、`0`/`false`/`off` 禁用 | Graph BUILD / **BUILD** | 仅当本轮 replay 未触碰 Graph 所有的 kernel 参数，且每个 Launch register-state token 仍与 prepared batch 一致时，通用 Graph executor 才省略 mutable-kernel scan。mutable DMA endpoint 每轮独立解析和更新，提交前会发布全部 command 写入；通用脏标记、token 变化或 token 查询失败都会保守执行完整 scan。该变量只用于诊断退出，不是模型 route。 |
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
| `RPU_LINEAR_ACC32` | 已停用 | 生产 launcher 不读取 | 在 `rpu_execution` 中设置组件的 `linear_acc32`；Wall-OSS 的 `runtime_env` 拒绝旧名称。 |
| `RPU_RMSNORM_NEWTON` | 已停用 | 生产 launcher 不读取 | Wall-OSS 的 `runtime_env` 拒绝此名称；应从生产配置中移除。 |
| `RPU_RMSNORM_VWARP` | 兼容输入；Hy-VLA 严格接受空/`0`/`off`/`false`、`16`、`32` 或 `auto`，Wall-OSS 拒绝该变量 | owner cold-plan admission / **NATIVE** | 归属 adapter 将其翻译成冻结的 per-handle capability。Hy-VLA 无效文本在进程 claim 前失败；原生 handle 不会重新读取。 |

普通 direct-SPM RMSNorm 按 handle 冻结可用算子，使用统一的 `(rows, cols)` 路由：
行数能被 32 整除时优先 V32，其次是能被 16 整除的 V16，否则使用 BASE。V16/V32 使用八核 grid。
`RPU_RMSNORM_VWARP` 仅在 Hy-VLA VLM 冷态配置时转换为能力选择；显式要求但不可用的路由在安装时失败。
Mixed RMSNorm 读取 BF16、输出 FP16。

<a id="qwen35-vision-selectors"></a>
### Qwen3.5 Vision selector

这些 selector 只在 numeric-blocked 受控评估门允许 Qwen3.5 Vision 后生效，
不会扩大其 Source-only 状态。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `QWEN3_5_VISION_CHUNK` | Auto；十进制整数；空值/非正值表示 auto | 首次原生 vision plan / **NATIVE** | 限制 Qwen3.5 Vision chunk。正数 override 必须仍处于允许的 shape 和 memory envelope 内。 |
| `RPU_VISION_MERGER` | **PB(true)** | Qwen3.5 Vision setup / **MODEL** | 启用 设备 merger 路径。`0` 选择 fallback，并改变性能/数值。 |
| `RPU_VISION_STEP0` | **PB(true)** | Qwen3.5 Vision setup / **MODEL** | 启用 设备 first-stage 路径。`0` 选择 fallback，并改变性能/数值。 |

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
| `RPU_HY_VLA_PATCH_EMBED_IN_GRAPH` | builder 默认 `1`；严格接受 `0`/`off`/`false` 或 `1`/`on`/`true`；显式空值表示开启；其他值失败 | 进程 claim 前的 cold-plan admission / **MODEL** | capture 设备 patch embedding。禁用会选择不同的 host/设备 boundary。 |
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
| `RPU_INTERNVLA_DINO_FINAL_NORM_RPU` | **PB(true)** | DINO tower 构造 / **MODEL** | 在 设备 上运行 final normalization。`0` 会改变 host/设备 boundary 和 timing。 |
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
| `RPU_LINGBOT2_RUNTIME_GRAPH` | `0`；当前示例为 `1`；字面量 `0/1` | 构建 / **MODEL** | 保留 Vision/Prefill/Action Graph 和原生去噪；拒绝 exact-Z2、W4 和 debug routing 组合。 |
| `RPU_LINGBOT2_SPARSE_EXPERTS` | `0`；当前示例为 `1`；字面量 `0/1` | 构建 / **MODEL** | 使用紧凑 Top4 FP16/W8 专家；拒绝 dense grouped、W4、exact-Z2 和 debug routing/chunk override。 |
| `RPU_LINGBOT2_ROUTED_GEMM` | `0`；字面量 `0/1` | 构建 / **MODEL** | dense grouped-expert 替代路由；要求 grouped storage，拒绝 sparse storage。 |
| `RPU_LINGBOT2_PREFILL_KV_PAIR` / `RPU_LINGBOT2_PREFILL_QK_NORM_ROPE` | **PB(false)** | 构建 / **MODEL** | 实验性 Prefill 融合，要求 runtime Graph 和匹配的 native 能力；当前模板均关闭。 |
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
| `RPU_PI05_DENOISE_GRAPH` | **PB(true)** | Action 构造 / **MODEL** | capture denoise step。未包含当前 RhinoForge 源码补丁的 v1.2.0 构建中，变化的逻辑 prefix 会让 Graph 持续 BUILD 并耗尽 arena；设为 `0` 的 host loop 又会丢失物理 descriptor，导致稳态数值错误。补丁前两者都不应作为生产规避方案，见[已知故障与修复](pitfalls.zh.md#pi05-v120-denoise-已知故障2026-10-22)。 |
| `RPU_PI05_DENOISE_UNROLL` | 原始为 **PB(false)**；Pi adapter 提供 `1` | Action 构造 / **MODEL** | 在一个 Graph 中记录固定 denoise loop，并使用 FP16 Euler state。相对 host loop 会改变数值。 |
| `RPU_PI05_EMBED_PREFIX_PATCH` | **PB(true)** | Adapter class-patch 安装 / **MODEL** | 启用支持的 prefix-embedding patch。class hook 是进程状态，因此只能在新进程中改变。 |
| `RPU_PI05_EULER_FP16` | **PB(false)** | 每次 action 调用 / **CALL**；重建 prepared Graph 配置 | 使用 FP16 而不是 host-FP32 Euler state。它会直接改变 action 数值。 |
| `RPU_PI05_FUSED_DENOISE` | **PB(true)** | Action 构造 / **MODEL** | 使用 fused denoise subsystem。`0` 选择 legacy per-step 路径，并改变拓扑/性能。 |
| `RPU_PI05_GEMMA_GRAPH` | **PB(true)** | VLM 构造 / **MODEL** | 启用 Gemma GraphCache 路径。需要稳定 prefix/cache owner。 |
| `RPU_PI05_KEEP_CPU` | 只接受未设置/空；任意非空值（包括 `0`）都会引发错误 | Pi0.5 安装 / **MODEL** | retired-path tripwire。刻意不支持通过该变量保留 CPU 权重。 |
| `RPU_PI05_KVINSERT_PAD16` | **B01(false)** | Expert 构造 / **MODEL** | 对 KV row 做 padding/mask。它会改变 cache layout 和 Graph signature。 |
| `RPU_PI05_PREFIX_MASK_CACHE` | **PB(true)** | 每次 action 调用 / **CALL** | 复用 one-entry host mask cache。`0` 会重新计算；正确性仍要求稳定 Graph input。 |
| `RPU_PI05_PREFIX_PAD16` | **PB(true)** | Prefix 规划 / **CALL**；新 plan 需重建 Graph | 为允许的 shape 增加 masked prefix padding。它改变 execution length/signature，不改变 logical prefix length。 |
| `RPU_PI05_SIGLIP_BATCH` | **PB(true)** | SigLIP 构造 / **MODEL** | 打包兼容 camera input。禁用会改变 Graph shape 和 host/设备 traffic。 |
| `RPU_PI05_SIGLIP_GRAPH` | **PB(true)** | SigLIP 构造 / **MODEL** | 启用 SigLIP GraphCache 路径。`0` 是 timing 不同的诊断 fallback。 |

<a id="qwen3-vl-shared-vision-profile"></a>
## Qwen3-VL 共享 Vision 配置

这些变量由 Qwen3-VL Vision 以及嵌入该 tower 的 VLA facade 消费。facade 可以在模型
构造前设置精确默认值。

精确普通 dense FP16 Qwen3-VL-2B、4B 和 8B 配置默认在 设备 执行 patch 和
merger GEMM，merger LayerNorm 保持 CPU FP32。安装前设置
`RPU_QWEN3VL_VISION_HOST_FP32_PATCH=1` 可保留 CPU FP32 patch projection 和
CPU FP32 merger。该选择在安装时固定；之后修改环境变量需要重新创建模型。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_QWEN3VL_VISION_BATCH` | **PB(false)** | Vision forward / **CALL**；重建 Graph | 打包兼容 image/view。它会改变 Graph signature 和内存；facade 约束仍会限制 shape。 |
| `RPU_QWEN3VL_VISION_BATCH_CAP` | `3`；十进制整数，至少收窄为 `1` | Vision forward / **CALL**；重建 Graph | 每个 packed group 的最大兼容 image 数。更大值可能超过配置内存 envelope。 |
| `RPU_QWEN3VL_VISION_FUSED_MERGER` | 原始默认关闭；**B01(false)** | Vision 构造 / **MODEL** | 将 merger 折叠进 Vision Graph。只有字面值 `0`/`1` 能保持 Python 与原生 reader 一致。 |
| `RPU_QWEN3VL_VISION_HOST_FP32_PATCH` | **PB(false)** | Vision 安装 / **MODEL**；在 `to_rpu()` 前设置，修改需新建模型 | 固定 CPU FP32 patch 配置；精确普通 dense FP16 VL2/VL4/VL8 同时保留 CPU FP32 merger。它会改变数值和 transfer cost。 |
| `RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE` | `0`；**E1** | Vision 构造 / **MODEL** | 将 patch embedding 移到 设备。它会改变已安装权重、数值和 Graph 拓扑。 |
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
| `RPU_RHINOVLA_VISION_RPU_MERGER_NORM` | **PB(false)** | Vision 安装 / **MODEL** | 在 设备 上运行 merger normalization。它会改变 boundary 和数值配置。 |
| `RPU_RHINOVLA_VISION_RPU_MERGER_OUTPUT_RPU` | **PB(false)** | Vision 安装 / **MODEL** | 将 merger output 保留在 设备 上。consumer 必须接受 设备-resident output 和稳定 ownership。 |
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

固定 v3 的 `rtc=True` 模式接受相同的 FP16 与 W8 精度范围。统一入口选择对应
`rtc_*.toml`，独立配置则保持原 TOML 并使用 `run --rtc --delay 6`。AdaRMS 驻留、
专家融合、HIGH、vector norm 与 partial RoPE 沿用所选设置；RTC 按该配方准备各
delay 的条件参数。精度与 RTC 模式在构造时固定，请求中的 delay 可在 `0..6` 内
变化，详见 [RTC 输入契约](../examples/configs/rhinovla/v3/README.md#rtc)。
启用 AdaRMS 驻留时，RTC 每个去噪步将该步的 state、已确定 prefix、剩余 suffix
三组条件表装入 SPM，并在该步的 18 层专家中复用，为 Vision/Text 保留缓冲区留出
空间；non-RTC 仍使用十步条件表驻留。

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

构造时绑定 `camera_names`，每次请求按名称提供对应图像。Vision 分块通过
`rpu_execution.vision` 选择；packed Vision 在 `RPU_WALL_OSS_BATCH_MAX_SEQ` 容量内合并
兼容的图像 grid。RTC 构造器绑定固定相机/动作配置，拒绝组件 `linear_acc32=true`；
普通配置使用各 stage 显式设置。

| 变量 | 未设置时的默认值与接受值 | 读取 / 更改 | 作用域、效果与风险 |
|---|---|---|---|
| `RPU_WALL_OSS_ACTION_FP32_TAIL` | **PB(false)** | 每次 denoise 调用 / **CALL**；重建 prepared Graph 配置 | Euler tail 使用 host FP32。它会改变 action 数值，并与 fused/unrolled 执行冲突。 |
| `RPU_WALL_OSS_ATTN_TP8` | 原始为 **PB(false)**；facade 为 `1` | 权重安装 / **MODEL** | 为 eight-core attention 复制 KV head。重建权重、cache 和 Graph。 |
| `RPU_WALL_OSS_BATCH_MAX_SEQ` | `768`；十进制整数 | Vision grouping / **CALL**；重建 Graph | 限制 packed vision sequence length。过大可能超出内存；所选公开配置可能进一步约束。 |
| `RPU_WALL_OSS_BATCH_VISION` | **PB(true)** | Vision forward / **CALL**；重建 Graph | 在配置容量内合并 equal-grid image；`0` 使用逐图执行。 |
| `RPU_WALL_OSS_DENOISE_UNROLL` | **PB(true)** | Action 构造和调用 / **MODEL** | 使用 fused denoise loop。`0` 选择 host loop，并改变拓扑和 timing。 |
| `RPU_WALL_OSS_DEVICE_PATCH_EMBED` | **PB(true)** | Vision 构造 / **MODEL** | 在 设备 上运行 folded patch embedding。`0` 使用 CPU projection，并改变 boundary/性能。 |
| `RPU_WALL_OSS_EXPERT0_DOWN_INT4` | **PB(true)** | 量化 LLM 权重安装 / **MODEL** | 使用允许的 compact expert-0 down projection。部分 W4 配置强制关闭；不匹配会校验失败。 |
| `RPU_WALL_OSS_FAST_IMGPROC` | **PB(true)** | Policy 构造 / **MODEL** | 首次调用 parity 检查后使用 direct image-preprocessing 路径。`0` 强制使用 HF processor。 |
| `RPU_WALL_OSS_FAST_PROCESSOR` | **PB(true)** | Policy 构造 / **MODEL** | 首次调用 parity 检查后绕过 combined processor wrapper。`0` 保留 wrapper。 |
| `RPU_WALL_OSS_FAST_REPLAY` | 原始默认关闭；不以 `0` 或 `f` 开头的非空值；facade 为 `1` | Fused handle 构造 / **MODEL** | replay 时跳过已审计的 layer-body emission。错误使用可能 replay 过期 dynamic state。 |
| `RPU_WALL_OSS_FUSED_ASSEMBLE` | 原始为 **PB(false)**；facade 为 `1` | Vision 构造 / **MODEL** | 将 Vision reorder 与 input-embedding 构造合并。需要 on-设备 embedding 路径。 |
| `RPU_WALL_OSS_FUSED_DENOISE` | Facade 为 **PB(true)** | Action 构造 / **MODEL** | 使用 fused action subsystem。`0` 选择 legacy per-step 路径，并改变拓扑/timing。 |
| `RPU_WALL_OSS_GENERIC_PROLOGUE` | **PB(false)** | Prompt 构造 / **CALL** | 选择 generic prompt schema。与其他 prologue selector 互斥，并改变 token。 |
| `RPU_WALL_OSS_HOST_CACHE` | **PB(true)** | Component 构造 / **MODEL** | 启用 host layout/prompt memoization 和 stable owner。`0` 增加重复 host 工作。 |
| `RPU_WALL_OSS_KVINSERT_PAD16` | **PB(true)** | LLM 构造 / **MODEL** | 对 prefill KV row 做 padding/mask。它会改变 execution length、cache layout 和 Graph signature。 |
| `RPU_WALL_OSS_MULTISUITE_PROLOGUE` | **PB(false)** | Prompt 构造 / **CALL** | 选择 multisuite prompt schema；与 generic/short mode 互斥。改变 token。 |
| `RPU_WALL_OSS_ONDEVICE_EMBED` | **PB(true)** | Policy 构造 / **MODEL** | 在 设备 上构造 token/vision embedding。`0` 使用 CPU fallback，并禁用 fused construction 路径。 |
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
| `RPU_WALL_OSS_VISION_DEVICE_MERGED` | **PB(false)** | Vision 构造 / **MODEL** | 将 eager merged result 保留在 设备 上，作为普通 fused 路径的诊断替代。改变 ownership/topology。 |
| `WALL_OSS_PROFILE_PREFIX` | 未设置；十进制整数 target length | 每次 prediction / **CALL**；prepared production Graph 禁止使用 | 保留 image token 后截断尾部 text，用于 profiling shape。所得 action 没有意义。 |

## 非环境变量 Runtime 控制

应优先使用这些公共 API，而不是增加新的环境开关。

### 按 handle 生效的 `rpu_execution`

公共 loader 和 policy 接受不可变的 `rpu_execution` mapping。stage 表为
`prefill`、`vision` 和 `action`；支持该能力的入口另有 `model` 控制表；每个入口都会公布其支持的 subset，并在加载权重前拒绝
其他所有 stage 或字段。

| 字段 | 接受值 | 含义与生命周期 |
|---|---|---|
| `model.num_cores` | 整数 `4/6/8` | 仅限入口声明并由模型 profile 准入的冷态计算预算。 |
| `<stage>.linear_acc32` | 布尔值 | 上述组件的 ACC16（`false`，默认）或 ACC32（`true`）；不改变量化权重格式。 |
| `prefill.fast_replay` | 布尔值 | 仅 Qwen3-VL；普通 API 默认 `false`，适用的 FP16/量化示例显式启用。 |
| `chunk_size` | `"auto"` 或 16 的正整数倍 | 在模型准入范围内自动选择或指定精确 chunk；支持该能力的 owner 可通过执行 API 更新。 |
| `padding_rows` | `"auto"` 或非负整数 | 指定 stage 的精确/自动 execution padding。精确整数与 `padding_budget` 互斥。 |
| `padding_budget` | 非负整数 | stage planner 考虑的最大可选 padding。不能与精确整数 `padding_rows` 同时使用。 |

初始 mapping 在权重安装和首次 Graph BUILD 前传给 loader/factory。绑定了执行会话的 owner，可在请求之间用 [`reconfigure_rpu_execution`](api_reference.zh.md#执行配置与运行中重配) 更新支持热修改的 stage。传入 stage 整体替换原 override，并使受影响的 Graph 失效。核数、累加精度、重放策略及其他安装设置仍为冷态；无会话的 owner 需新实例，所有更新均须满足该 owner 的能力范围。

### Graph 快照策略

`GraphRuntimePolicy.batch_snapshot_mode` 按实例冷绑定：`"auto"` 在 SDK 和算子资产支持时启用快照重放；`"off"` 使用普通 prepared Graph。两者均保留 Graph 捕获、重放和模型已有的 fast replay 设置。

```python
from dataclasses import replace
from rpu_backend.graph import GraphCache, GraphRuntimePolicy

policy = GraphRuntimePolicy.from_environment(batch_snapshot_mode="auto")
cache = GraphCache(runtime_policy=policy)
control_policy = replace(policy, batch_snapshot_mode="off")
```

在权重安装前，将 `graph_runtime_policy=policy` 传给 `Pi05Policy.from_pretrained`、`WallOssPolicy.from_pretrained` 或 `build_lingbot_vla_v2`。Wall 的各组件构建器接受同一参数。Pi 保留原计算核数，至少需要五个 Graph arena；镜像组件复用首个 owner 的冷策略。传入策略为完整的冻结配置，仅改变快照模式时使用 `replace`。

AUTO 校验 Launch 能力以及已加载资产的哈希和程序身份。未知资产、旧 SDK 或配置被拒绝时回退到普通 Graph；模型环境开关不能越过这些检查。Graph 统计公开 `batch_snapshot_mode`、`batch_snapshot_available`、`batch_snapshot_effective_contracts` 和 `batch_snapshot_fallback_reason`。回退原因包括 `policy_off`、`sdk_symbols_unavailable`、`sdk_capability_unavailable`、`untrusted_program:<name>` 和 `sdk_configuration_rejected`。

### 内存与 coherency 控制

- `torch.rpu.set_caching_allocator(bool)` 选择进程 tensor allocator（默认 direct/关闭）。
  第一次显式选择或非空 设备 tensor allocation 会冻结该选择；同值重复调用是幂等的，
  冲突值会报错。切换模式必须使用新进程。`torch.rpu.empty_cache()` 释放已缓存、未使用的
  block；无法释放 live tensor 或 Graph-owned storage。需要保留大量 tensor 的 adapter
  会在首次 设备 allocation 前选择同一个通用 caching policy。
- `torch.rpu.memory_stats()` 和 `torch.rpu.get_memory_stats()` 返回不含地址的
  allocator counter。`caching_allocator_policy_frozen` 表示选择是否已经不可更改。
  当 `caching_allocator_enabled` 为 true 时，
  `caching_allocator_mapping` 只统计该 allocator 持有的 HostDDR segment，
  `cached_idle_mapping` 统计其中完全空闲的 segment；它们不包含 Launch、SPM 或
  direct allocation，因此不是进程级 driver mapping 总数。
  `reset_peak_memory_stats()` 重置 peak counter，`reset_accumulated_memory_stats()` 重置
  累计 allocate/free counter。重置 counter 不会释放内存。
- `torch.rpu.set_ddr_flush(bool)` 控制内部 设备-to-设备 flush point（默认关闭）。
  RhinoVLA 保留调用方的进程设置，不再自动开启跨组件的冗余刷新。
- `torch.rpu.set_ddr_flush_force(bool)` 控制 CPU/设备 boundary coherency（默认开启）。
  普通推理应保持启用；除隔离 microbenchmark 外，禁用是不安全的。对应 `get_*` 函数
  报告当前进程状态。

缓存同步应位于实际 CPU 数据访问处：CPU 写入后、设备读取前发布数据，CPU 读取
设备输出前完成同步。设备 tensor view、静态表绑定及设备间直接交接不需要额外刷新。
上传和 `.cpu()` 拷贝负责对应边界；通过主机 `memcpy` 实现的 设备-to-设备 `copy_`
仍需要读、写两侧的同步，不能只根据 tensor 的设备标签判断。

官方 Qwen3.5-27B W8/W4 在取得模型所有权后选择共享缓存分配器，并在上传权重前预留四个 Graph arena。原始权重留在 CPU，逐层上传最终布局；分配器冲突会在修改权重前释放 owner 并报错。Text prefill 为有界单次 Graph，decode 独立保留 Graph；W8 支持 4/6/8 核，W4 group32 支持 8 核。详见 [27B 配置](../examples/configs/qwen3_5/README.md)。

同步按实际张量字节范围执行，包含 view offset。复用输入上传时仍检查本次内容和归属，原地修改会刷新；固定存储地址不表示内容不变。

### Profiling 机制

| 机制 | 测量内容 | 启用时机 | 输出/风险 |
|---|---|---|---|
| `torch.rpu.set_profile(True)` | Backend wrapper 和累计 timing counter | 如果只需要 steady-state counter，在 warmup 后启用；测量窗口前调用 `reset_profile_accumulators()` | 轻量 console/counter 诊断；它不是 PyTorch operation trace。 |
| `torch.profiler.profile(...)` | CPU 和 `PrivateUse1` operation scope，包括 Graph capture/replay range | 只包围要测量的调用；按 benchmark protocol 单独包含 warmup | 生成 operation-level trace/table，并增加 profiler 开销。 |
| `torch.rpu.hw_perf_trace(output_dir, max_dumps=32)` | 匹配 sanitized Release Launch 的硬件 trace | 在模型准备和 Graph BUILD 前启用；Qwen3 runner 提供 `--hwperf` / `run.hwperf=true` | 输出 Chrome 硬件 trace，限制 dump 数且影响执行耗时；保存在源码和发布资产之外。 |

不同 runner 支持范围不同；Qwen3 命令及拒绝跟踪的延迟入口见[模型诊断](model_testing.zh.md#诊断调用)。

## 清单维护

表格为每个当前具体环境/配置名称保留一行。新增 reader 或 facade preset 时，必须在正确
tier 增加一行；删除最后一个 reader/preset 时，必须删除对应行。自动化文档检查应将行
集合与源码 reader 双向比较，并将诊断变量保持在稳定的 `Diagnostic-only inventory`
heading 之后。

## 模型与诊断补充设置

模型开关在构建实例前设置，修改后须重新创建模型。下表默认值指未设置时的行为，
模板可显式选择其他取值。模型及其[配置](../examples/configs/CONFIGURATIONS.md)
会检查支持的输入形状。诊断输出来自当前调用，保存在调用方指定的输出目录。

| 变量 | 未设置时的默认值 | 接受值 | 作用 |
|---|---|---|---|
| `HALO_CHUNK_DIAG` | 未设置 | 未设置关闭；只要设置即开启 | 记录共享规划器的候选 SPM 占用和最终分块，每次规划读取。 |
| `HALO_NODE_HIST` | 未设置 | 未设置关闭；只要设置即开启 | 提交批次前记录捕获的节点和算子数量，每批读取。 |
| `HALO_PROFILE` | 0 | 仅字面值 `1` 开启 | 用 torch.profiler 记录 HALO forward，每次进入分析上下文读取。 |
| `HALO_PROFILE_DIR` | output/halo_profile | 目录路径 | HALO_PROFILE 开启时，表格和 Chrome trace 的输出目录。 |
| `QWEN3_5_VISION_DBG_Q` | 0 | `1` 或 `true` 开启 | 开启张量导出时捕获各层旋转后的 Q，首次使用读取。 |
| `QWEN3_VL_DELIVERY_ALLOW_NUMERIC_BLOCKED` | 0 | 仅字面值 `1` 开启 | 允许固定 Qwen3-VL delivery 配置通过入口限制；仍检查权重与模型形状。 |
| `RHINO_CHUNK` | 0 | 整数 | 未传 rpu_execution 时指定 RhinoVLA prefill 分块；正数请求固定分块，非正数保持自动规划。 |
| `RHINO_SMOKE_PREFIX_PADDING_ROWS` | 0 | 整数 | 诊断用前缀可见范围截断；官方降序 flow 流程要求为 0。 |
| `RHINO_SPM` | 未设置 | 未设置或空值关闭；其他值开启 | 在 RhinoVLA 流程检查点打印 SPM 分配摘要，每次调用读取。 |
| `RHINO_TRACE` | 未设置 | 未设置或空值关闭；其他值开启 | 打印 RhinoVLA 流程阶段日志，每次调用读取。 |
| `RPU_GR00T_CROSS_KV_REUSE` | 0 | `0`/`1`；空值关闭，拒绝其他值 | 在普通 B1/Q41、TP8、四步或十步降噪中跨步骤复用交叉注意力 K/V；每次推理的首步骤刷新，仍检查模型形状、缓存容量和完整 SPM 需求。 |
| `RPU_GR00T_CROSS_V_TRANSPOSE_REUSE` | 0 | `0`/`1`；空值关闭，拒绝其他值 | 依赖交叉 K/V 复用及对应 SPM 路径；首步骤转置 V，后续步骤复用。 |
| `RPU_GR00T_FF1_RESIDENT_W8` | 0 | `0`/`1`；空值关闭，拒绝其他值 | 在支持的普通 B1/Q41、TP8 降噪循环中将第 0 层 FF1 W8 权重驻留 SPM；要求对应扩展算子和完整 SPM 检查，源权重修改会使副本失效。 |
| `RPU_GR00T_KV_SPM_ONLY` | 0 | `0`/`1`；空值关闭，拒绝其他值 | 普通四步或十步 TP8 计划实际使用 SPM K/V 时省略无用 DDR 镜像；DDR、单步及 Z2 路径保留镜像。 |
| `RPU_GR00T_NATIVE_POSITION_CACHE` | 0 | `0`/`1`；空值关闭，拒绝其他值 | 为普通未补齐的 VL 调用缓存按内容和版本识别的 int32 位置表；补齐及 Z1 路径保持原位置处理。 |
| `RPU_GR00T_PRECOMPUTE_ADALN` | 0 | `0`/`1`；空值关闭，拒绝其他值 | 权重安装后，为普通 B1/Q41、TP8 展开循环预计算四步或十步 AdaLN 表；要求源权重未改变且尚未 forward 或准备 Z2。 |
| `RPU_GR00T_PRECOMPUTE_TIME_GLUE` | 0 | `0`/`1`；空值关闭，拒绝其他值 | 为相同普通 TP8 降噪配置预计算固定时间步和输出调制表，与 AdaLN 预计算独立。 |
| `RPU_GR00T_QWEN3VL_SPM_Z1` | 0 | `1` 或 `true` 开启 | 选择固定 FP16 单图像 Vision→Text SPM 路径；不接受 W8、部分旋转和主机 FP32 patch 计算。 |
| `RPU_GR00T_RING_XOR3` | 0 | `0`/`1`；空值关闭，拒绝其他值 | 为支持的普通 B1/Q41、TP8、FP16 或 W8 O/FF2 投影选择可选残差规约路径；保留 FP32 累加及最终 FP16 舍入。 |
| `RPU_GR00T_SELF_QKV_MULTIBASE` | 0 | `0`/`1`；空值关闭，拒绝其他值 | 模型形状和扩展算子匹配时合并普通 W8、TP8 自注意力 Q/K/V 投影；交叉注意力仍分别投影。 |
| `RPU_GR00T_VL_KV_SPM_ONLY` | 0 | `0`/`1`；空值关闭，拒绝其他值 | 支持的普通 B1、TP8 VL 注意力采用 SPM 路径时省略无用 DDR K/V 镜像；DDR 和已准备的 Z2 路径保留镜像。 |
| `RPU_GRAPH_HCB_CHECKSUM` | 0 | 未设置、空值或首字符 `0` 关闭；其他值开启 | 在 aten::_to_copy 回调前后记录形状、类型、求和及小张量值；首次执行回调时读取，不输出地址。 |
| `RPU_GRAPH_TO_COPY_LOG` | 0 | 未设置、空值或首字符 `0` 关闭；其他值开启 | 记录稳定输出登记和接收返回值的字节数；登记路径首次读取，捕获路径每次读取。 |
| `RPU_INTERNVLA_NEXTDIT_FOLD_NORM_MOD` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 在融合展开循环中将逐步 AdaRMS 调制与固定归一化权重预合并；逐步诊断执行保持原计算。 |
| `RPU_INTERNVLA_NEXTDIT_FUSED_QKQ` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 合并 NextDiT 每层的 self-Q、self-K 和 cross-Q 投影，要求匹配支持的 flow 形状。 |
| `RPU_INTERNVLA_NEXTDIT_STEP_PREP_RPU` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 依赖归一化调制预合并，将时间步投影、bias/tanh、合并及输入准备放入设备降噪图；SiLU 保留在 CPU。 |
| `RPU_L2_CAPTURE_GATE` | 0 | 仅字面值 `1` 开启 | 捕获 LingBot2 gate 和累加器张量，安装权重时增加捕获缓冲区。 |
| `RPU_L2_CAPTURE_INNORM` | 0 | 仅字面值 `1` 开启 | 捕获 LingBot2 第 0 层 AdaRMS 后的输入，安装权重时分配。 |
| `RPU_L2_CAPTURE_L0` | 0 | 仅字面值 `1` 开启 | 捕获 LingBot2 第 0 层输出，安装权重时分配。 |
| `RPU_L2_CAP_RESID` | 0 | 仅字面值 `1` 开启 | 捕获 LingBot2 各层输入和注意力残差，安装权重时分配。 |
| `RPU_L2_DBG_PACKED` | 0 | 仅字面值 `1` 开启 | 启用 LingBot2 打包权重诊断读取接口，每次读取时检查。 |
| `RPU_L2_STAGES` | 0 | 仅字面值 `1` 开启 | 捕获 LingBot2 分组中间阶段张量，安装权重时分配。 |
| `RPU_LINGBOT2_DEBUG_DUMP_ROUTER_H` | 0 | 仅字面值 `1` 开启 | 捕获 LingBot2 路由器输入张量，安装权重时分配。 |
| `RPU_LINGBOT2_MULTIVIEW_SPM_Z2` | 0 | 仅字面值 `1` 开启 | 选择固定三相机 256×256 Z2 配置；要求 RPU_LINGBOT2_ALLOW_UNVALIDATED=1 及对应模板开关。 |
| `RPU_LINGBOT_DENOISE_UNROLL` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 将 LingBot 十步 Euler 降噪循环捕获为一张图。 |
| `RPU_LINGBOT_ENCODER_1THREAD` | 1 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 降噪循环内使用一个 CPU 线程，结束后恢复原线程数；视觉阶段保持原线程设置。 |
| `RPU_LINGBOT_EXPERT_REPLAY` | 1 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 通过稳定输入和刷新的调制量跨降噪步骤复用动作专家图；0 表示每步重建。 |
| `RPU_LINGBOT_FUSED_ENCODER` | 1 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 预合并动作/时间编码器投影及逐步时间偏置；0 保留独立投影路径。 |
| `RPU_LINGBOT_ONDEVICE_ENCODER` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 在设备上执行融合后缀编码器、输出投影和 Euler 更新。 |
| `RPU_LINGBOT_VLM_REPLAY` | 1 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 使用稳定输入缓冲区，在前缀形状相同的调用间复用 VLM prefill 图。 |
| `RPU_P7_SPLIT_KERNELS` | 空 | 逗号分隔的名称 | 将匹配的算子名称隔离为独立 Graph 分段用于诊断，准备分段时读取。 |
| `RPU_PI05_DENOISE_GATEUP_RESIDENT_LAYERS` | 0 | 字面值 `0`、`1` 或 `5` | 为固定 P800/M50、TP8、十步循环将首层或前五层动作 W8 Gate/Up 权重组成驻留包；要求已准备 AdaRMS 及 body streaming，独立 GateUp 驻留开关须关闭。 |
| `RPU_PI05_PREFILL_KV_ONLY` | 0 | `0`/`1`；空值关闭，拒绝其他值 | 在支持的配对 TP8 Pi0.5 prefill 配置中，末层只生成 K/V，省略未使用的 hidden 输出；要求专用 KV-only 入口及 ACC16 prefill，部分模板显式开启。 |
| `RPU_PI05_PREFILL_MASK_FP16` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 在 CPU 上以 FP16 准备并缓存前缀加性 mask，类型纳入缓存键；部分配对 TP8 模板显式开启。 |
| `RPU_PI05_PREFILL_ROPE_CACHE` | 0 | `0`/`1`；空值关闭，拒绝其他值 | 按源张量身份、版本和位置缓存一次 RoPE 上传；原生稳定槽仍刷新，配对 Down、KV-direct 和 KV-only 配置要求关闭。 |
| `RPU_PI05_PROBE_DIR` | 未设置 | 目录路径 | 将具名 Pi0.5 中间张量和 K/V 快照写入指定目录；每次导出读取，未设置则不写文件。 |
| `RPU_RHINOVLA_ADARMS_RESIDENT` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 将固定 v3 专家的预计算 AdaRMS 表装入 SPM：non-RTC 驻留十步表；RTC 逐步驻留 state/prefix/suffix 三组表并在该步 18 层复用。要求预计算表，并拒绝 SKIP_ADARMS_GEMV。 |
| `RPU_RHINOVLA_ALIGNED_KV` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 补齐固定 v3 专家的物理前缀与 K/V 写入，保留逻辑 mask 和 RoPE 位置；检查容量并清零补齐部分。 |
| `RPU_RHINOVLA_EXPERT_DEPTH_LIMIT` | 未设置 | 不超过权重层数的正整数 | 诊断用专家层数，范围为 1 到权重层数，并调整前缀层选择；官方及固定 W8 配置拒绝缩减层数。 |
| `RPU_RHINOVLA_EXPERT_FUSIONS` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 在固定 v3 专家中融合 norm/shift 及门控残差；要求预计算 AdaRMS、GATED_NO_SUB 及两个对应算子。 |
| `RPU_RHINOVLA_EXPERT_MLP_DIM_LIMIT` | 未设置 | 不超过权重宽度的正整数 | 诊断用专家 MLP 宽度，范围为 1 到权重宽度；官方及固定 W8 配置拒绝缩减宽度。 |
| `RPU_RHINOVLA_EXPERT_W8A16` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 在 non-RTC 或 RTC 模式下量化固定 v3 专家的 Q/K/V/O 和 MLP 权重，使用逐通道 FP16 scale；拒绝缩减形状及 PACKED_QKV，全流程 W8 同时启用专家 W8。 |
| `RPU_RHINOVLA_HIGH_PRECISION` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 为带驻留预计算 AdaRMS 的固定全 W8 v3 选择 Newton RMSNorm 和高精度激活；捕获前要求对应可选算子齐备。 |
| `RPU_RHINOVLA_HIGH_PRECISION_FUSIONS` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 依赖 HIGH_PRECISION，使用对应高精度算子融合动作 norm/shift 和文本/动作 SiLU×up。 |
| `RPU_RHINOVLA_PACKED_QKV` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 为固定 FP16 v3 专家合并不可变 Q/K/V 权重，保留原输出分区；修改后须重新安装权重。 |
| `RPU_RHINOVLA_PARTIAL_ROPE` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 固定全 W8 v3 开启 HIGH_PRECISION 时，使用现有 partial-RoPE 路径旋转完整 head，保持补齐行不变。 |
| `RPU_RHINOVLA_PRECOMPUTE_GATE_TANH` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 全 W8 v3 使用驻留预计算 AdaRMS 时，在构造时间步表期间计算固定 gate TANH，回放时复用。 |
| `RPU_RHINOVLA_VECTOR_K_NORM` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 固定全 W8 v3 使用向量 K 归一化；要求 HIGH_PRECISION 和 ALIGNED_KV，归一化前清零尾部。 |
| `RPU_RHINOVLA_VECTOR_Q_NORM` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 固定全 W8 v3 开启 HIGH_PRECISION 时，将 Q 归一化补齐到向量形状，并在每层/每次回放清零新增行。 |
| `RPU_RHINOVLA_VISION_INPUT_CACHE` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 复用视觉类型转换、patch 输入和带版本的位置缓冲区，每次调用刷新像素；绑定已安装视觉实例，最多保留四条缓存。 |
| `RPU_WALL_OSS_ACTION_KV_OWNER_FANOUT` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 固定非驻留动作路径只生成一次 K/V 并经 SPM 分发；M32 不支持此路径。 |
| `RPU_WALL_OSS_ACTION_WEIGHT_RESIDENCY` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 每次执行图时预装支持的动作权重：W8 第 1–4 层或 FP16 第 1–2 层；base H32 要求 sorting SiLU，FP16 还要求 transfer bit4。 |
| `RPU_WALL_OSS_FP16_MLP_M224` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 将 FP16 M640 Gate/Up 拆为 224+224+192 行；M320 边界调用保留原路径。 |
| `RPU_WALL_OSS_FP16_TRANSFER_MASK` | 0 | 十进制 `0..7` | 位 1/2/4 选择固定 FP16 Vision/Prefill/Action 调度，权重保持 FP16。 |
| `RPU_WALL_OSS_PREFILL_MIXED_PRECISION` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 为固定 TP8、H2048/I11008/L36 RTC prefill 选择 FP16/BF16 边界，须配合视觉混合精度；RTC 模板显式设置取值。 |
| `RPU_WALL_OSS_PREFILL_OWNER_DIRECT_KV` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 固定混合精度、分阶段 P640 owner-QKV 路径在部分 RoPE 后直接写 K/V。 |
| `RPU_WALL_OSS_PREFILL_STAGED_MLP` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 选择层优先的 W8 P640/C320×2 MLP 调度；要求混合 prefill、TP8、因果位置 0 及固定 RTC 权重，P672 保留自身调度。 |
| `RPU_WALL_OSS_RTC_ALLOW_NUMERIC_BLOCKED` | 0 | 仅字面值 `1` 开启 | 允许具名 RTC FP16/W8A16 配置通过入口限制；仍拒绝 W4 和不匹配配置，不会自动设置其他开关。 |
| `RPU_WALL_OSS_RTC_EMBODIMENT` | X2 ex001_desktop | 非空字符串 | RTC 提示词中的机器人名称；非 RTC 调用使用数据集名称。应填写调用方机器人映射中的对应名称。 |
| `RPU_WALL_OSS_SORTING_SILU_MUL` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 为支持的 sorting/base 路径融合 SiLU×Mul；固定 FP16 路径还要求对应 transfer 位，配对小图像视觉路径可跨层保留稠密 mask。 |
| `RPU_WALL_OSS_VISION_LAYER_GROUP` | 0 | `0` 或 `32` | 固定三图像、每图 576 patch 的批量视觉配置中，让每幅图连续经过全部视觉层；与 weight-outer 调度互斥，构建时在进程内固定。 |
| `RPU_WALL_OSS_VISION_MIXED_PRECISION` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 在融合 merger、每图 576 patch 的 RTC 视觉配置中选择 FP16/BF16 边界，须配合 prefill 混合精度；RTC 模板显式设置取值。 |
| `RPU_WALL_OSS_VISION_NATIVE_WINDOW_SDPA` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 在支持的 W8 weight-outer 配置或开启行复用和 SiLU 的 base 双/三 224 图像配置中使用原生窗口注意力；FP16 还要求 transfer bit1。 |
| `RPU_WALL_OSS_VISION_PAIR_252` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 固定三幅 252 图像时配对前两幅视觉输入，第三幅独立处理；要求通用窗口视觉、融合 merger、sorting SiLU 及 batch 容量至少 672。 |
| `RPU_WALL_OSS_VISION_ROW_WEIGHT_OUTER` | 0 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 在支持的紧凑/base 形状中跨视觉行块复用完整本地权重；FP16 要求 transfer bit1，小图像路径关闭混合精度。 |
| `RPU_WALL_OSS_VISION_WEIGHT_OUTER` | 0；RTC W8 默认 1 | `0`/`1`、`false`/`true`、`off`/`on`，忽略大小写 | 选择固定三图像、每图 576 patch 的批量视觉、混合精度和融合 merger 调度；FP16/W4 RTC 强制关闭，与 layer-group32 互斥，进程内固定。 |
| `RPU_WALL_OSS_W16_NEXT3_MASK` | 0 | `0` 或 `1` | 为 prefill O 投影选择 M160×4/N96 行调度。 |
| `RPU_WALL_OSS_W16_NEXT4_MASK` | 0 | 十进制 `0..3` | 位 1 选择 W8 M640 Gate/Up 行调度，位 2 选择配对 ring/postnorm。 |
| `RPU_WALL_OSS_W16_REUSE_MASK` | 0 | `0` 或 `8` | 为驻留动作 Gate/Up 启用 A16 复用，其他位组合会被拒绝。 |
