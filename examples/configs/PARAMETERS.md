# 通用配置参数

[模型目录](README.md) · [权重准备](WEIGHTS.md) · [命名规则](NAMING.md) · [日志与结果](../RESULTS.md)

复制所选模型的完整 TOML，再填写路径和输入。字段是否可用及其取值范围取决于模型；省略字段时由运行器或环境提供默认值，不等于关闭该选项。

## 配置身份与权重

每个模板由 `[model]`、`[input]`、`[run]` 三张表组成，按需加 `[cpu]`、`[rpu_execution]`、`[logging]`。`[model]` 写明交给哪个运行器、用什么精度和哪份权重；`[input]` 只放请求数据。

```toml
[model]
target = "pi05"                      # 运行器
precision = "fp16"                   # 本文件的权重/计算配方
checkpoint = "pi05-libero-finetuned" # 权重路径或登记别名

[input]
batch = "/path/to/pi05_3cam_t32_batch.pt"
cameras = 3
```

| 字段 | 用途 | 设置方法 |
|---|---|---|
| `model.target` | 选择运行器 | 必填，使用模板已有值 |
| `model.profile_id` | 选择登记的模型结构、精度和执行路径 | 绑定登记 profile 的入口才写；Pi0.5、G0.5、HyVLA、LingBot v2、Wall-OSS Base/Sorting、Qwen3 ASR/TTS 等专用入口不写。部分配方通过它区分输出头量化、核数或加载方式 |
| `model.precision` | 本文件的权重/计算配方 | 与 profile、权重格式一致；具体模块精度见模型页 |
| `model.registry_alias` | 按登记名定位缓存中的模型资产 | 与 profile 配套，目录由模型登记表解析；设置 `RPU_MODEL_CACHE` 指定缓存根目录 |
| `model.checkpoint` 等资产字段 | 显式指定权重或配套文件 | 路径或该入口接受的权重别名。Wall-OSS 分别用 `fp16_checkpoint`、`w8_checkpoint`；GR00T 的骨干用 `qwen3vl_registry_alias`；RhinoVLA v3 源码入口另有 `source_dir` |
| `model.architecture` | 模型架构名 | 使用模板已有值 |
| `model.opt_in` | 冷启动开关 | 须与登记 profile 完全一致（目前只有 RhinoVLA v3 和 LingBot exact-Z2 使用）；修改会被拒绝 |
| 其他 `model.*` | 模型专用的配方字段 | 见下方“模型专用参数” |

不同模型使用的资产字段不同。Qwen 系列通过 `registry_alias` 定位目录；Pi0.5 等用 `checkpoint` 填写路径或别名。多数入口填写实际路径；LingBot v2 等支持环境变量展开的入口按其模板设置。TOML 本身不执行 Shell 替换。

<a id="legacy-layout"></a>

### 旧格式

2026-10-20 之前的模板用 `[example]` 或以模型命名的表（如 `[pi05]`、`[g05]`），权重字段写在 `[input]`。旧文件仍可直接运行，加载时按下表转换；同一文件不能同时使用 `[model]` 和旧表。

| 旧写法 | 新写法 |
|---|---|
| `[example]`，其中 `dtype` | `[model]`，其中 `precision` |
| `[pi05]`、`[g05]`、`[hy_vla]`、`[lingbot2]`、`[wall_oss]`、`[wall_oss_sorting]`、`[qwen3_asr]`、`[qwen3_tts]` | `[model]` 加 `target = "<原表名>"` |
| `[input]` 中的 `checkpoint`、`fp16_checkpoint`、`w8_checkpoint`、`action_w4_checkpoint`、`release_root`、`training_config`、`runtime_factory`、`factory_config`、`asset_manifest`、`fp16_registry_alias`、`qwen3vl_registry_alias` | 移到 `[model]` |
| G0.5 `linear_accumulation = "acc16"` / `"acc32"` | `linear_acc32 = false` / `true` |
| RhinoVLA v3 验证片段的 `[env]` | `[model.opt_in]`，并补齐 `[model]`、`[input]`、`[run]` 后由 `run.py` 运行 |

实际加载由 TOML 内容决定。切换配方时使用对应的完整模板，保持权重、profile、精度和组件设置匹配。

## 输入

| 字段 | 含义 | 关联要求 |
|---|---|---|
| `input.prompt/images` | 提示词和图片路径 | 图片数量、分辨率和 processor 与模型配套 |
| `input.prefill_tokens` | 目标 prefill token 数 P | 文本的补齐方式、图文 token 计算方法由模型页说明 |
| `input.decode_steps` | prefill 后的 decode 调用上限 D | prefill 已产生第一个输出 token，执行满 D 次时输出 D+1 个 |
| `input.stop_on_eos` | 生成 EOS 后是否提前结束 | 布尔值，只用于文本与图文生成入口。`true` 时生成 EOS 即停止，decode 次数可能少于 D；`false` 时固定执行 D 次。省略时 Qwen3、Llama、Gemma4、Qwen3.5 文本与 Qwen3-VL 为 `true`，Qwen3.5 图文为 `false`；所有模板都写 `false`，以固定负载。Qwen3.5 MoE 固定文本和 `legacy_dense_vl` 计时要求 `false`。CPU 参考按同一设置生成。实际 decode 次数记录在 `report.json` 的 `generation_runs[].decode_calls` |
| `input.batch` | 保存预处理 tensor 字典的文件路径 | 匹配相机槽位、图像 shape、状态、token 和 mask |
| `input.cameras` | 相机槽位数 | 与输入 batch 和模型槽位定义匹配 |
| `input.text_tokens` | 模型定义的文本长度 | Pi0.5 等使用 token tensor 宽度，GR00T 使用 instruction 有效 token 数，范围见模型页 |
| `input.noise/num_steps` | 动作模型的初始噪声和去噪步数 | 与动作 shape、schedule 及预计算表配套 |

并非每个模型都接受以上全部字段。完整输入结构见所选模型页。配置检查核对结构和登记关系，资产和 tensor 形状在运行时检查。

<a id="cpu-threads"></a>

## 执行与采样

| 字段 | 用途 | 设置方法 |
|---|---|---|
| `cpu.threads` | CPU intra-op 线程数，与 RPU 核数无关 | 正整数。省略时：G0.5、Pi0.5、Wall-OSS Base 为 8，HyVLA、HY Embodied 为 12，LingBot v2 为 1，Qwen3 ASR 为 4，Qwen3 TTS 为 16；其余入口不修改线程设置，PyTorch 使用环境变量 `OMP_NUM_THREADS`，未设置该变量时使用全部 CPU 核（参考板 24）。实际值记录在结果 `report.json` 的 `cpu_runtime.torch_num_threads`。2026-10-19 性能报告中未写该字段的配置，测量环境预设了 `OMP_NUM_THREADS=16`；复现时写 `threads = 16` |
| `rpu_execution.model.num_cores` | RPU 核数预算 | Qwen3、Qwen3.5（含 MoE）、Qwen3-VL、Qwen3 ASR/TTS 可选 4 / 6 / 8，默认 8；Llama 3.2-1B FP16 可选 4 / 6 / 8，默认 8（6 核时 MLP 宽度 8192 一次性补零到 8256）；Pi0.5 只有 FP16/W8、T32（2cam 或 3cam）可选 4/6；DINOv3 ViT-B FP16 可选 4 / 6 / 8；Gemma4 E4B FP16 可选 4 / 6 / 8（注意力始终 2 核，MLP/逐层嵌入为 4/4、6/4，6 核时 MLP 宽度补零到 10272）；其余入口固定 8 核，可以显式写 8，写 4/6 时配置检查报出原因。模板未写该字段时为 8。6 核时文本组件为 Attention 4、MLP 6、输出头 4；Vision、音频编码器和 codec Transformer 在 4/6 核下用 4 核。每个配置的支持情况、不支持的原因和实际分配见[核数支持矩阵](../../docs/planner_core_topology.zh.md)与模型页的“4 核 / 6 核 / 8 核”列，修改后需重新加载 |
| `rpu_execution.*.linear_acc32` | 对应组件 Linear 的累加精度 | `false` 为 ACC16，`true` 为 ACC32；不改变权重位宽。省略时多数入口为 `false`；Qwen3 ASR 为 `true`；G0.5 写在 `model.linear_acc32`，模板中 fp16 为 `true`、w8a16 为 `false`。Qwen3.5 MoE INT4 与 Wall-OSS RTC 只支持 `false` |
| `rpu_execution.*.chunk_size` | 对应组件执行分块行数 | `"auto"`（默认）或整数。auto 由共享 planner 在不超过本模型上限的候选中，先选分块最少、再选补齐最少的一种；输入行数不超过上限时通常只分一块。上限与对齐：Qwen3 0.6B/1.7B 512、4B 256、8B 128（只接受 16/32/64/128）、14B 32、32B 64；Qwen3-VL 2B 320、4B 256、8B 128、32B 64（16 对齐）；Qwen3.5 0.8B 512、2B 640、4B 448、9B 256、27B 128、MoE 512（64 对齐；MoE 开启 `prefill.moe_resident_prefill` 后上限放宽到 8192，仍须 64 对齐）。decode 固定 16，不可配置。部分 VLA 入口自行决定分块（如 Pi0.5 8 核为 (256×相机数+T)/2），每个 TOML 顶部注释写有实测值；实际值记录在 `report.json` 的执行计划中 |
| `rpu_execution.*.padding_rows` | 物理执行补齐行数 | 省略时由 planner 与分块一起选择（规则同 `chunk_size` 的 auto，例如 Wall-OSS Base 超过 384 行时平均分成两块），实际值记录在 `report.json` 的执行计划中；写成整数时按该行数补齐，只适合输入长度固定的场景，换输入后须重新计算。补齐行不增加有效 token。可与 `padding_budget`（允许补齐的上限）二选一 |
| `run.warmup` | 预热次数 | 公共登记入口省略为 `2`；HyVLA 至少 `3`，LingBot v2 至少 `1`，其余按模型页要求；预热不计入正式统计 |
| `run.runs` | 正式运行次数 | 公共登记入口省略为 `1`，取正整数；专用入口和模板的次数见模型页 |
| `run.seed` | 随机种子 | 公共登记入口省略为 `0`，取非负整数；各入口按其规则生成输入或噪声 |
| `run.inference_mode` | Torch 推理上下文 | 公共入口省略时为 `true`；`false` 使用 `no_grad`，专用入口按模型页说明 |
| `run.output_dir` | 报告和产物目录 | 公共登记入口省略为 `perf_results/examples`；专用入口可能要求填写，子目录和文件名见模型页 |
| `run.profile` | 额外分析调用 | 布尔值，省略为 `false`；命令行 `--profile` 等同 `true`。开启后在正式运行之后另做 1 次推理，用 PyTorch profiler 记录 CPU 活动（以及可用时的 RPU 活动），写出 `profile.trace.json`；除 G0.5 与 HyVLA/HY-Embodied 外还写出 `profile.txt`（按 self CPU 时间排序的前 100 项）。这次调用不计入耗时样本。Wall-OSS Base 与 Sorting 不接受；Pi0.5 开启时须填写 `run.output_dir` |
| `run.hwperf` | 硬件跟踪采集 | 布尔值，省略为 `false`；命令行 `--hwperf` 等同 `true`。跟踪写入 `<run.output_dir>/hwperf/`，汇总在 `report.json` 的 `hwperf`；请求了却没有生成可读跟踪时运行报错。采集窗口：G0.5 在正式计时之后另做 2 次回放、LingBot v2 另做 1 次调用，最多 16 份 dump，不影响耗时样本；其余入口（文本、图文、Vision、Pi0.5、HyVLA/HY-Embodied、GR00T、InternVLA-N1、RhinoVLA、Wall-OSS legacy/RTC）从加载模型（Graph BUILD 之前）一直跟踪到正式运行结束，最多 32 份 dump，因此这时的正式耗时包含跟踪开销，不能作为性能数据。Wall-OSS Base、Sorting 与 Qwen3 ASR/TTS 不接受 |
| `run.timing_mode` | 文本生成的 prefill/decode 计时边界 | 只用于文本与图文生成入口。`"per_call"`（省略时）：每次调用计入输入传输、forward 和 argmax，prefill 含 FP32 CPU 最后一位 logits，decode 为各次调用之和。`"forward_argmax"`：只用于并且必须用于 Qwen3.5 MoE 固定文本性能配置（`prompt_file` + 两项 SHA），计入输入传输、`forward(logits_to_keep=1)` 和 greedy argmax，不含 reset、快照和检查。`"legacy_dense_vl"`：只用于 Dense FP16 Qwen3.5 图文 0.8B–9B，复现历史 Dense VL 报告，计入 forward、Torch argmax 和 token 上传/回读，decode 循环含首末 logits 检查，不含 prompt 上传；要求固定 P/D、`stop_on_eos = false`、`inference_mode = true` 和完整 decode 预热。当前模板只写 `"per_call"`（Qwen3.5 的 12 个量化模板）或不写。实际边界记录在 `report.json` 的 `generation_runs[].timing_scope`。完整调用耗时（速览表）不受此项影响 |
| `run.summary_statistic` | prefill/decode 摘要的统计方式 | `"mean"`（省略时）或 `"median"`，只用于文本生成入口（Qwen3、Llama、Gemma4、Qwen3.5、Qwen3-VL 文本与图文；Qwen3-VL Vision 不接受）。`mean`：耗时取平均，吞吐为全部 token 除以全部时间；`median`：耗时和每次运行的吞吐都取中位数（历史 Dense VL 报告的用法）。模板都未写，即 `mean`。完整调用耗时总是取中位数，不受此项影响 |

表中的 `*` 表示模板已有的组件位置，不是可直接填写的 TOML 字段。多数 LLM 使用 `prefill`、`vision`；RhinoVLA factory 使用 `components.language_model.prefill` 等路径，完整执行配置见[运行时配置参考](../../docs/runtime_config.md)。

CPU 线程与 RPU 核数独立设置。权重位宽、算子累加精度和非线性算法也各有开关，例如 W8 权重仍可搭配 ACC32。权重、核数、累加和安装开关变更后，用新进程重新加载。

## 模型专用参数

以下字段只出现在对应模型的模板中；未列出的模型专用字段按模板保留即可。

| 字段 | 模型 | 作用与取值 |
|---|---|---|
| `rpu_execution.prefill.fast_replay`、`vision.fast_replay` | Qwen3-VL、RhinoVLA | Graph 快速回放，冷启动选定，与累加精度无关；`true` 开启（模板默认），省略为 `false` |
| `rpu_execution.prefill.moe_resident_prefill` | Qwen3.5 MoE 35B-A3B | 路由专家常驻 prefill 数据流，冷启动选定。`false`（省略时的默认值）或 `true`；三个 35B-A3B 文本模板自 2026-10-29 起取 `true`。`true` 要求 `num_cores = 8`、UINT4 或 E4M3 路由权重、ACC16，且请求是 prefill 起点为 0 的因果 B1；任一条不满足时在安装或该次 forward 前报错，不会静默退回普通数据流。开启后 `chunk_size` 上限由 512 放宽到 8192（仍须 64 对齐），`"auto"` 改由 native 联合 SPM/mixer 规划选外窗，2026-10-26 实测最大 768。实际值记录在 `report.json` 的执行计划中，实测结果见性能页的 TP8 正文 prefill |
| `model.vision_window`、`vision_pair`、`vision_row_reuse` | Wall-OSS Base | Vision 执行布局：224 图使用 window，256 图使用 pair；随相机和图像尺寸模板给定，换尺寸时用对应模板 |
| `input.text_length_scope` | Wall-OSS Base | `"instruction"` 只统计指令 token；`"non_image"` 统计提示中的全部非图像 token；`text_tokens` 必须等于该范围的实际 token 数 |
| `input.inference_delay`、`input.rtc_prefix_actions` | Wall-OSS RTC | `inference_delay` 取 0–10 的整数，省略为 0（不使用前缀）。非零时必须给 `rtc_prefix_actions`：上一段已执行的物理动作，形状 `[L,26]`（或 `[1,L,26]`），`inference_delay ≤ L ≤ 20`，只使用前 `inference_delay` 行；为 0 时不能给出。两者都计入 CPU 参考的查找身份。Sorting 的前缀在 batch 的 `prefix_actions` 中，见 [Sorting 输入文件](wall_oss/sorting/README.md#input-files) |
| `model.rmsnorm_vwarp` | HyVLA | VLM 部分 RMSNorm 的实现。`"auto"`（省略时与模板）：按行数选择，能被 32 整除用 V32，能被 16 整除用 V16，否则用 BASE；`"16"`：固定请求 V16，所需 payload 不存在时在加载权重前报错。实际选择记录在执行描述符中 |
| `model.resident_fp8_scale_layers` | HyVLA | 把前 N 层 Gate/Up 的 FP8 scale 常驻在核内存中。取 `0`（省略时与模板）、`4`、`8`；非零只适用于 Action NVFP4（`w8a16_action_nvfp4.toml`）。4 层和 8 层都测过但未采用：8 层在 P240 第二次预热时内存不足，4 层的 denoise 与 DDR scale 持平。保持 `0`，详情见[运行时配置](../../docs/runtime_config.md) |
| `model.instruction_residency`、`rpu_execution.prefill.padding_budget` | LingBot v2 | 指令常驻策略（`"auto"`）与 prefill 允许的补齐行数（非负整数，省略为 15，模板为 64；prefix 为 2cam 132+T、3cam 198+T 行，先对齐到 16 的倍数，再最多补该行数。2026-10-23 板测对齐后 144/160/176/192/208/224/240/256/288/320/400 行可运行，272/304/336/352/368/384 行 NO_FEASIBLE，取 64 时 T1–160 都能补到可行长度；2026-10-30 在 native 997a369c 上复验 3cam 336 行仍 NO_FEASIBLE，该表未变，见模板注释）；补齐行不增加有效 token |
| `input.component` | Qwen3-VL | 运行哪一部分：`"multimodal"`（省略时）为图像加文本生成；`"text"` 为纯文本生成，不读取图片；`"vision"` 只运行 Vision 编码器，不做文本生成，`decode_steps` 与 `summary_statistic` 不适用。必须与模板登记的 profile 一致（profile id 以 `.multimodal`/`.text`/`.vision` 结尾），换组件时改用对应模板 |
| `input.action_horizon` | Pi0.5 | 动作序列长度，取 10 或 50，省略为 50；是权重的属性（checkpoint 的 `chunk_size`），写在这里只说明期望哪一份权重，与权重不一致时报错，不会改写权重。H10 用 `pi05-libero-h10` 系列权重，模板在 `2cam/h10/`、`3cam/h10/` |
| `input.embodiment_id`、`num_steps`、`action_horizon` | GR00T | `embodiment_id`：整数，动作头按这个下标选取各 embodiment 专用的权重，省略与模板均为 20，只测过 20；须是 checkpoint 中存在的下标，observation 若带 `embodiment_id` 须与它相同，它也计入 CPU 参考的查找身份。`num_steps` 取 4 或 10（模板 4），`action_horizon` 只取 40 |
| `rpu_execution.components.*` 布尔开关 | RhinoVLA v3 | Vision、prefill、action 三段的冷启动执行开关（`rpu_patch_embed`、`fused_merger`、`prefix_alias` 等），模板全部为 `true`，省略为 `false`；含义、依赖关系和是否改变数值见 [v3 执行开关](rhinovla/v3/README.md#execution-switches) |
| `input.expert_w8a16`、`input.full_w8a16` | RhinoVLA v3（factory 模板） | 声明本次运行的 expert W8 量化范围，只用于带 `runtime_factory` 的 factory 模板。两项都是布尔值，省略为 `false`。`expert_w8a16` 把 action expert 的七个主投影换成 INT8 权重加逐输出通道 FP16 scale；`full_w8a16` 在此之上再量化 expert 的 AdaRMS 条件投影，**不**改变 vision、text 和动作 IO 的精度——整条流程的 W8 由 `[model.opt_in]` 的 `RPU_RHINOVLA_FULL_W8A16` 决定，两个 `w8a16_pipeline*` 模板把它与 `RPU_RHINOVLA_EXPERT_W8A16` 一起置 `"1"`。因此 `full_w8a16 = true` 要求 `expert_w8a16 = true`。两项须与 `[model].precision` 一致：`precision = "w8a16"` 时必须**显式**写出两项且 `expert_w8a16 = true`（省略会被拒绝，不会默认推断），`precision = "fp16"` 时两项都不写。模板取值：`w8a16_action.toml` 为 `true`/`false`，`w8a16_action_adarms.toml` 与两个 `w8a16_pipeline*` 模板为 `true`/`true`。取值会写入 factory 的 `factory_config`（与其中同名键冲突即报错），加载后再按实际安装的 expert scale 与 AdaRMS owner 复核，factory 忽略该精度会报错而不是静默换配方。`controlled/` 下的 source 模板不写这两个键（多余的 `[input]` 键会被拒绝），其精度由上面两个 opt-in 开关决定 |
| `[libraries]` 的 `native`、`main`、`expansion` | G0.5、HyVLA | backend 构建与两个算子库 REF 的路径，三项须一起填写；main/expansion 的内容须与 `release/runtime-dependencies.json` 发布的 SHA-256 一致，路径可以不同。省略整个表时依次使用环境变量 `RPU_BACKEND_SO`、`RPU_KERNEL_LIB_PATH`、`RPU_KERNEL_LIB_PATH_EXPANSION`，未设置的 REF 取仓库 `src/`（发行包取已安装 wheel）中的当前文件，backend 取已安装的构建。模板中的 `/path/to/...` 是占位，`--check-config` 接受，运行时检查文件是否存在 |
| `model.math_precision`、`vision_camera_batch`、`linear_acc32` | G0.5 | `base` / `high` 非线性精度；Vision 批处理（FP16 只支持 B1）；`false` 为 ACC16，`true` 为 ACC32 |
| `model.codec_backend` | Qwen3 TTS | `"rpu"`（模板默认）或 `"cpu"`：codec 在 RPU 或 CPU 上运行；`fp16_cpu_codec.toml` 使用 `"cpu"` |


## 命令行参数

| 参数 | 作用 |
|---|---|
| `--config <TOML>` | 必填，运行的配置 |
| `--check-config` | 只检查字段与登记关系，不导入 Torch 或后端，也不读取权重和输入文件 |
| `--hwperf` / `--profile` | 额外采集硬件跟踪 / profiler；Wall-OSS Base 与 Sorting 不接受 |
| `--model <路径>` | 仅 Qwen3.5 MoE：替换 checkpoint 路径，profile 校验不变 |

## 日志与配置检查

`logging.level` 默认为 `ERROR`，`logging.summary` 默认为 `true`。取值、终端显示和日志文件见[日志与结果](../RESULTS.md)。

```bash
python examples/run.py --config /path/to/filled-config.toml --check-config
python examples/run.py --config /path/to/filled-config.toml
```

第一条命令检查字段和登记关系，第二条命令加载资产并运行。旧格式的 RhinoVLA `[env]` 片段不能用 `run.py`，只能交给 `validate_v3.py`。
