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
| `model.precision` | 本文件的精度配方 | 必填，写 [manifests/precision_recipes.v1.json](../../manifests/precision_recipes.v1.json) 中该模型族登记的配方名，即文件名或其开头（见[配置命名](NAMING.md)）。加载时展开为运行器的精度写法；绑定 profile 的入口还须是该 profile 登记的配方，checkpoint 限制（安装时量化、W4 分组、Qwen3-VL 量化包格式）与 profile 一致，否则报错。权重目录带[配方记录](WEIGHTS.md#recipe-record)时，记录的配方须是本配方在该字段加载的那份产物；没有记录的权重不做这项检查。Qwen3 / Qwen3-VL 安装时做的量化由配方给出。VLA 与语音运行器按配方展开后的逐模块精度选择实现路径（如 Pi0.5 的 SigLIP 与 AdaRMS W8A16、Prefill Gate/Up W8A8 开关，Wall-OSS、G0.5、HY-VLA、LingBot v2、GR00T 的 W8 / W4 / NVFP4 开关，Qwen3 语音的精度），只看各模块精度，不看配方名；运行器没有实现的组合在加载时报错。切换精度时改用对应模板，权重也随之更换；各模块精度见目录的“量化范围”一列 |
| `model.registry_alias` | 按登记名定位缓存中的模型资产 | 与 profile 配套，目录由模型登记表解析；设置 `RPU_MODEL_CACHE` 指定缓存根目录 |
| `model.checkpoint` 等资产字段 | 显式指定权重或配套文件 | 路径或该入口接受的权重别名。Wall-OSS 分别用 `fp16_checkpoint`、`w8_checkpoint`；GR00T 的骨干用 `qwen3vl_registry_alias`；RhinoVLA v3 源码入口另有 `source_dir`。Wall RTC 同时写 `registry_alias` 和 `checkpoint` 时，权重从 `checkpoint` 读取（W8A16 的 FP16 源取 `fp16_checkpoint`）；`registry_alias` 只用于匹配 profile，须保留且与 profile 一致。删去 `checkpoint` 会改读缓存中的上游 `wall-oss-0.5`，不满足 RTC 的 v728 要求 |
| `model.architecture` | 模型架构名 | 使用模板已有值 |
| `model.opt_in` | 冷启动开关 | 须与登记 profile 完全一致（目前只有 RhinoVLA v3 和 LingBot exact-Z2 使用）；修改会被拒绝。RhinoVLA v3 的 `RPU_RHINOVLA_FULL_W8A16` / `RPU_RHINOVLA_EXPERT_W8A16` 由精度配方设置，TOML 写出即报错 |
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
| `input.stop_on_eos` | 生成 EOS 后是否提前结束 | 布尔值，只用于文本与图文生成入口。`true` 时生成 EOS 即停止，decode 次数可能少于 D；`false` 时固定执行 D 次。省略时 Qwen3、Llama、Gemma4、Qwen3.5 文本与 Qwen3-VL 为 `true`，Qwen3.5 图文为 `false`。模板不一定写这一项：写了的都是 `false`，没写的按上面的默认值，所以比较性能时要看实际 decode 次数；[输入预设](#input-presets)都写 `false`，以固定负载。Qwen3.5 MoE 固定文本和 `legacy_dense_vl` 计时要求 `false`。CPU 参考按同一设置生成。实际 decode 次数记录在 `report.json` 的 `generation_runs[].decode_calls` |
| `input.batch` | 保存预处理 tensor 字典的文件路径 | 匹配相机槽位、图像 shape、状态、token 和 mask |
| `input.cameras` | 相机槽位数 | 与输入 batch 和模型槽位定义匹配 |
| `input.text_tokens` | 模型定义的文本长度 | Pi0.5 等使用 token tensor 宽度，GR00T 使用 instruction 有效 token 数，范围见模型页 |
| `input.noise/num_steps` | 动作模型的初始噪声和去噪步数 | 与动作 shape、schedule 及预计算表配套 |

并非每个模型都接受以上全部字段。完整输入结构见所选模型页。配置检查核对结构和登记关系，资产和 tensor 形状在运行时检查。

<a id="cpu-threads"></a>

## 执行与采样

| 字段 | 用途 | 设置方法 |
|---|---|---|
| `cpu.threads` | CPU intra-op 线程数，不是 RPU 核数 | 正整数。省略时：G0.5、Pi0.5、Wall-OSS Base 为 8，HyVLA、HY Embodied 为 12，LingBot v2 在 8 核为 1、4/6 核为 12（4/6 核时 Vision 的 patch embed 与 merger 在 CPU 上逐图计算），Qwen3 ASR 为 4，Qwen3 TTS 为 16；其余入口不修改线程设置，PyTorch 使用环境变量 `OMP_NUM_THREADS`，未设置该变量时使用全部 CPU 核（参考板 24）。实际值记录在结果 `report.json` 的 `cpu_runtime.torch_num_threads`。2026-10-19 性能报告中未写该字段的配置，测量环境预设了 `OMP_NUM_THREADS=16`；复现时写 `threads = 16` |
| `rpu_execution.model.num_cores` | RPU 核数预算 | Qwen3、Qwen3.5（含 MoE）、Qwen3-VL、Qwen3 ASR/TTS 可选 4 / 6 / 8，默认 8；Llama 3.2-1B FP16 可选 4 / 6 / 8，默认 8（6 核时 MLP 宽度 8192 一次性补零到 8256）；Pi0.5 全部 29 份模板（含 LIBERO H10/H50 的 FP16、W8A16、Action NVFP4、Prefill W8A8 + Action NVFP4、Vision W4 + Action NVFP4，2cam 或 3cam，T32–T160；以及 Kuavo openpi 原始观测模板的 FP16、W8A16，3cam，T64–T192）均可选 4 / 6 / 8，默认 8；DINOv3 ViT-B FP16 可选 4 / 6 / 8；Gemma4 E4B FP16 可选 4 / 6 / 8（注意力始终 2 核，MLP/逐层嵌入为 4/4、6/4，6 核时 MLP 宽度补零到 10272）；其余入口固定 8 核，可以显式写 8，写 4/6 时配置检查报出原因。模板未写该字段时为 8。6 核时文本组件为 Attention 4、MLP 6、输出头 4；Vision、音频编码器和 codec Transformer 在 4/6 核下用 4 核。每个配置的支持情况、不支持的原因和实际分配见[核数支持矩阵](../../docs/planner_core_topology.zh.md)与模型页的“4 核 / 6 核 / 8 核”列，修改后需重新加载 |
| `rpu_execution.*.linear_acc32` | 对应组件 Linear 的累加精度 | `false` 为 ACC16，`true` 为 ACC32；不改变权重位宽。省略时多数入口为 `false`；Qwen3 ASR 为 `true`；G0.5 写在 `model.linear_acc32`，模板中 fp16 为 `true`、w8a16 为 `false`。Qwen3.5 MoE INT4 与 Wall-OSS RTC 只支持 `false` |
| `rpu_execution.*.chunk_size` | 对应组件执行分块行数 | `"auto"`（默认）或整数。auto 由共享 planner 在不超过本模型上限的候选中，先选分块最少、再选补齐最少的一种；输入行数不超过上限时通常只分一块。上限与对齐：Qwen3 0.6B/1.7B 512、4B 256、8B 128（只接受 16/32/64/128）、14B 32、32B 64；Qwen3-VL 2B 320、4B 256、8B 128、32B 64（16 对齐）；Qwen3.5 0.8B 512、2B 640、4B 448、9B 256、27B 128、MoE 512（64 对齐；MoE 开启 `prefill.moe_resident_prefill` 后上限放宽到 8192，仍须 64 对齐）。decode 固定 16，不可配置。部分 VLA 入口自行决定分块（如 Pi0.5 8 核为 (256×相机数+T)/2，3 相机 T192 即为 480 行的配对 C480），每个 TOML 顶部注释写有实测值；G0.5 prefill 取 64/128/192/256，auto 按选定表取值，见 [G0.5](g05/README.md#auto-values)。Qwen3-VL 文本组件（`text/` 下的配置）不运行 Vision，`vision.chunk_size` 只在加载时校验（16 的倍数，不超过 4096），不参与执行。实际值记录在 `report.json` 的执行计划中 |
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
| `model.rmsnorm_vwarp` | HyVLA | VLM 部分 RMSNorm 的实现。`"auto"`（省略时与模板）：按行数选择，能被 32 整除用 V32，能被 16 整除用 V16，否则用 BASE；`"16"`：固定请求 V16，所需 payload 不存在时在加载权重前报错。各输入下的实际取值见 [HyVLA](hy_vla/README.md#auto-values)，实际选择记录在执行描述符中 |
| `input.prefix_len` | HyVLA | VLM prefix 行数。`"auto"`（省略时与模板）取 ceil16(2+58×相机数+text_tokens)，模板 T32 为 160（2cam）/ 208（3cam）；显式值须 16 对齐，从 auto 值到 ceil16(2+58×相机数+64)（2cam 192，3cam 240）。取值表见 [HyVLA](hy_vla/README.md#auto-values) |
| `model.resident_fp8_scale_layers` | HyVLA | 把前 N 层 Gate/Up 的 FP8 scale 常驻在核内存中。取 `0`（省略时与模板）、`4`、`8`；非零只适用于 Action NVFP4（`w8a16_action_nvfp4.toml`）。4 层和 8 层都测过但未采用：8 层在 P240 第二次预热时内存不足，4 层的 denoise 与 DDR scale 持平。保持 `0`，详情见[运行时配置](../../docs/runtime_config.md) |
| `model.instruction_residency` | LingBot v2 | Graph 指令常驻（batch snapshot replay）策略，取 `"auto"`（省略时与模板）或 `"off"`，其他值报错。`"auto"`：每个 Graph 冷启动时判断，Launch SDK 提供 snapshot 符号与能力、且该 Graph 全部程序都有经审核的契约时启用，否则退回普通 prepared replay，计算内容不变；与相机数、指令长度、精度无关。`"off"`：始终用普通 replay，Graph 捕获与 replay 照常。报告中每个 Graph 的 `instruction_residency` 记录 `batch_snapshot_available`、`batch_snapshot_effective_contracts`（3 表示只读指令区与位置无关程序两类契约都生效）与 `batch_snapshot_fallback_reason`（如 `policy_off`、`sdk_symbols_unavailable`、`untrusted_program:<名称>`）；2026-10-09 与 2026-10-23 的记录中三个 Graph 都是 available、契约 3。只能在 TOML 中修改（`--set` 不接受 `model.*`）。与入口固定的 `LKN_BATCH_INSTR_REUSE=0` 无关。详见[运行时配置](../../docs/runtime_config.md) |
| `rpu_execution.prefill.padding_budget` | LingBot v2 | prefill 允许的补齐行数（非负整数，省略为 15，模板为 64；prefix 为 2cam 132+T、3cam 198+T 行（`training_config` 不含 `align_params` 时；含时各加 16 行，见 [LingBot v2 训练配置](lingbot/v2/README.md#training-config)），先对齐到 16 的倍数，再最多补该行数。2026-10-23 板测对齐后 144/160/176/192/208/224/240/256/288/320/400 行可运行，272/304/336/352/368/384 行 NO_FEASIBLE，取 64 时 T1–160 都能补到可行长度；2026-10-30 在 native 997a369c 上复验 3cam 336 行仍 NO_FEASIBLE，该表未变，见模板注释）；补齐行不增加有效 token |
| `input.component` | Qwen3-VL | 运行哪一部分：`"multimodal"`（省略时）为图像加文本生成；`"text"` 为纯文本生成，不读取图片；`"vision"` 只运行 Vision 编码器，不做文本生成，`decode_steps` 与 `summary_statistic` 不适用。必须与模板登记的 profile 一致（profile id 以 `.multimodal`/`.text`/`.vision` 结尾），换组件时改用对应模板 |
| `input.action_horizon` | Pi0.5 | 动作序列长度，取 10 或 50，省略为 50；是权重的属性（checkpoint 的 `chunk_size`），写在这里只说明期望哪一份权重，与权重不一致时报错，不会改写权重。H10 用 `pi05-libero-h10` 系列权重，模板在 `2cam/h10/`、`3cam/h10/` |
| `input.embodiment_id`、`num_steps`、`action_horizon` | GR00T | `embodiment_id`：整数，动作头按这个下标选取各 embodiment 专用的权重，省略与模板均为 20，只测过 20；须是 checkpoint 中存在的下标，observation 若带 `embodiment_id` 须与它相同，它也计入 CPU 参考的查找身份。`num_steps` 取 4 或 10（模板 4），`action_horizon` 只取 40 |
| `rpu_execution.components.*` 布尔开关 | RhinoVLA v3 | Vision、prefill、action 三段的冷启动执行开关（`rpu_patch_embed`、`fused_merger`、`prefix_alias` 等），模板全部为 `true`，省略为 `false`；含义、依赖关系和是否改变数值见 [v3 执行开关](rhinovla/v3/README.md#execution-switches) |
| `input.expert_w8a16`、`input.full_w8a16` | RhinoVLA v3（factory 模板） | 声明本次运行的 expert W8 量化范围，只用于带 `runtime_factory` 的 factory 模板。两项都是布尔值，省略为 `false`。`expert_w8a16` 把 action expert 的七个主投影换成 INT8 权重加逐输出通道 FP16 scale；`full_w8a16` 在此之上再量化 expert 的 AdaRMS 条件投影，**不**改变 vision、text 和动作 IO 的精度——整条流程的 W8 由 `[model.opt_in]` 的 `RPU_RHINOVLA_FULL_W8A16` 决定，两个 `w8a16_pipeline*` 模板把它与 `RPU_RHINOVLA_EXPERT_W8A16` 一起置 `"1"`。因此 `full_w8a16 = true` 要求 `expert_w8a16 = true`。两项由 `[model].precision` 的配方设置，TOML 不写（写出即报错）：`w8a16_action` 为 `true`/`false`，`w8a16_action_adarms` 与 `w8a16_pipeline` 为 `true`/`true`，`fp16` 两项都不设。运行器仍要求 W8A16 运行显式收到两项且 `expert_w8a16 = true`，不会默认推断。取值会写入 factory 的 `factory_config`（与其中同名键冲突即报错），加载后再按实际安装的 expert scale 与 AdaRMS owner 复核，factory 忽略该精度会报错而不是静默换配方。`controlled/` 下的 source 模板没有这两个键（多余的 `[input]` 键会被拒绝），其精度由配方设置的两个 opt-in 开关决定 |
| `[libraries]` 的 `native`、`main`、`expansion` | G0.5、HyVLA | backend 构建与两个算子库 REF 的路径，三项须一起填写；main/expansion 的内容须与 `release/runtime-dependencies.json` 发布的 SHA-256 一致，路径可以不同。省略整个表时依次使用环境变量 `RPU_BACKEND_SO`、`RPU_KERNEL_LIB_PATH`、`RPU_KERNEL_LIB_PATH_EXPANSION`，未设置的 REF 取仓库 `src/`（发行包取已安装 wheel）中的当前文件，backend 取已安装的构建。模板中的 `/path/to/...` 是占位，`--check-config` 接受，运行时检查文件是否存在 |
| `model.math_precision`、`vision_camera_batch`、`linear_acc32` | G0.5 | `base` / `high` 非线性精度；Vision 批处理（`"auto"`、1 或相机数，省略时为 1，FP16 只支持 1）；`false` 为 ACC16，`true` 为 ACC32。Vision 批和 prefill chunk 的 auto 按 `selected_profiles.json` 取值，见 [G0.5](g05/README.md#auto-values) |
| `input.image_size` | Qwen3.5 VL、Qwen3-VL 图文与 Vision、GR00T | Qwen3.5 / Qwen3-VL：正整数，读入的每张图先缩放成 `image_size × image_size` 再交给 processor；省略时按图片原尺寸。Qwen3-VL 每张图不超过 4096 patch（≤ 1024），超出在运行时报错。需要 `input.images`，Qwen3-VL 文本组件和预处理输入不接受。GR00T 的取值见其模板（224 或 256） |
| `input.audio_seconds` | Qwen3 ASR | 取 `input.audio` 开头的这么多秒，0.5–8（整数或小数）；省略时用整段录音（须为 0.5–8 秒）。不补静音，录音短于该值时运行时报错 |

## 命令行参数

| 参数 | 作用 |
|---|---|
| `--config <TOML>` | 必填，运行的配置；与 `--recipe` 同用时也可以写模型目录 |
| `--recipe <配方名>` | 选同一模型目录下的 `<配方名>.toml`，配方名见[配置命名](NAMING.md)。不写时运行 `--config` 指定的文件；`--config` 写目录时必须给出，名字不对时报错并列出该目录的配方。只选已有模板，不生成新组合 |
| `--check-config` | 只检查字段与登记关系，不导入 Torch 或后端，也不读取权重和输入文件 |
| `--hwperf` / `--profile` | 额外采集硬件跟踪 / profiler；Wall-OSS Base 与 Sorting 不接受 |
| `--model <路径>` | 仅 Qwen3.5 MoE：替换 checkpoint 路径，profile 校验不变 |
| `--cores 4\|6\|8` | 覆盖 `rpu_execution.model.num_cores`；不写时用 TOML 中的值，TOML 也未写时为 8。配置不支持的核数在配置检查时报错，原因与[核数支持矩阵](../../docs/planner_core_topology.zh.md)表下所列相同 |
| `--input <预设名>` | 套用本模型族的一组输入字段，`canonical` 取该族的参考输入；不写时用 TOML 中的输入。预设见下文[输入预设](#input-presets) |
| `--set <表>.<键>=<值>` | 覆盖一个字段，可重复；不写时全部取 TOML 中的值。可改的表和字段见下文[命令行覆盖](#cli-overrides) |
| `--emit-toml <路径>` | 把覆盖后的完整配置写成 TOML，用 `--config` 运行该文件得到同一配置；不能与 `--model` 同用 |

<a id="cli-overrides"></a>

### 命令行覆盖

`--recipe` 先选定模板；`--cores`、`--input` 和 `--set` 修改加载后的配置，之后按 TOML 字段的同一套规则检查，不支持的取值报出的原因与直接写进 TOML 时相同。`<值>` 按 TOML 字面量解析（`128`、`true`、`"带空格的文本"`），未加引号的单词按字符串处理（`auto` 即 `"auto"`）。同一字段出现两次时报错，`--cores` 与 `--set rpu_execution.model.num_cores=…` 视为同一字段。

| 表 | 可覆盖的字段 |
|---|---|
| `[input]` | 对应模型页列出的输入字段与范围；选择权重的键（`checkpoint`、`*_checkpoint`、`registry_alias` 等）除外 |
| `[run]`、`[cpu]` | 本页列出的运行与 CPU 字段 |
| `[rpu_execution]` | `model.num_cores`：4、6 或 8。`prefill.chunk_size`：模板已写该字段时可改为 `"auto"` 或 16 的正整数倍，auto 的实际取值规则见[执行与采样](#执行与采样)。各阶段的 `linear_acc32`：`true` 或 `false`，阶段须已在模板中出现（如 `prefill`、`vision`、`action`）。其余执行字段只能改 TOML；G0.5 的 chunk_size 位于 `components.language_model.prefill`，不能用 `--set` 修改 |
| `[g05]` | 仅 G0.5：`linear_accumulation`（`"acc16"` / `"acc32"`）、`math_precision`（`"base"` / `"high"`） |

`[model]` / `[example]`、`[pi05]` 等专用运行器表和 `[libraries]` 不能覆盖：它们决定运行哪个 profile 和哪套运行库，需要换用对应的 TOML。

```bash
python examples/run.py --config examples/configs/qwen3/text/4b/fp16.toml --cores 6 \
    --set rpu_execution.prefill.chunk_size=128 --set input.decode_steps=32 \
    --emit-toml qwen3-4b-6core.toml --check-config
```

<a id="input-presets"></a>

### 输入预设

`--input <预设名>` 从 [`examples/inputs/`](../inputs/) 中找到列出本配置模型族的文件（`applies_to`），把该预设的字段写入 `[input]`，再按 `--set input.*` 的规则检查。`--input canonical` 取文件里 `canonical` 指定的预设。`--set input.<键>` 与预设的同名字段同时出现时，以 `--set` 为准。没有预设的模型族会报错，并列出已有预设的模型族。

| 文件 | 适用 | 预设 |
|---|---|---|
| [text_generation.toml](../inputs/text_generation.toml) | Qwen3、Qwen3.5 dense 文本、Qwen3-VL 文本、Gemma4 | `p128_d32`、`p128_d128`、`p1024_d128`（canonical）、`p2048_d32`、`p4096_d32`：P = `prefill_tokens`，D = `decode_steps`，均为 `stop_on_eos = false` |
| [llama.toml](../inputs/llama.toml) | Llama 3.2 1B | `p64_d32`、`p64_d128`（canonical）：prefill 上限 64 |
| [vla.toml](../inputs/vla.toml) | Pi0.5、HyVLA、GR00T、Wall-OSS Base | `2cam_t32`（canonical）、`2cam_t64`、`2cam_t128`、`3cam_t32`、`3cam_t64`、`3cam_t128`：`cameras` 个相机，`text_tokens` 个指令 token |
| [vla_g05.toml](../inputs/vla_g05.toml) | G0.5 | 同上六个，另写对应的 `camera_names` |
| [vla_lingbot2.toml](../inputs/vla_lingbot2.toml) | LingBot v2 | 同上六个，指令长度写在 `expected_language_tokens` |
| [image_text.toml](../inputs/image_text.toml) | Qwen3-VL 图文、Qwen3.5 VL（dense 与 MoE） | `i448_p256_d32`、`i896_p1024_d32`（canonical）、`i1024_p1280_d32`：图像缩放到 `image_size` 见方，P = `prefill_tokens`，D = `decode_steps`，均为 `stop_on_eos = false` |
| [image_vision.toml](../inputs/image_vision.toml) | Qwen3-VL Vision | `i448`、`i896`（canonical）、`i1024`：只写 `image_size` |
| [asr.toml](../inputs/asr.toml) | Qwen3 ASR | `2s`、`6s`（canonical）、`8s`：取录音开头 `audio_seconds` 秒，`max_new_tokens = 64` |
| [tts.toml](../inputs/tts.toml) | Qwen3 TTS | `zh_short`、`zh_sentence`（canonical）、`zh_long`、`en_sentence`：`text` 与 `language`，`max_new_tokens = 256` |

预设只写长度字段，prompt 仍取所选 TOML 的值：`prefill_tokens` 把 prompt 的 token 截取或重复到 P 个，decode 固定调用 D 次。长度上限由运行时检查，`--check-config` 不检查 `--set` 写的长度：Qwen3-14B 的多 token prefill 上限为 192，只能用 `p128_d32` 和 `p128_d128`，其余预设配置检查直接拒绝（写在预设文件的 `[[limit]]`）；Qwen3 其他尺寸为 4096，Qwen3.5 dense 为 8192，Qwen3-VL 文本为 4176。Qwen3.5 MoE 文本的性能输入用 `prompt_file` 绑定原文和 token SHA，配置检查会拒绝这些预设。Llama 3.2 1B 当前 prefill 上限为 64，没有可用的预设。

图文预设的图片和 prompt 仍取所选 TOML：每张图先缩放成 `image_size` 见方再交给 processor，`prefill_tokens` 在最后一条用户消息末尾重复 prompt 的 token，把图文输入补到 P 个，不截断。单张图、模板的一句描述在 448、896、1024 时 processor 输出约 210–215、800–805、1040–1045 个 token，均在预设的 P 以内；多张图或更长的 prompt 超过 P 时运行时报错，用 `--set input.prefill_tokens` 改大。Qwen3-VL 每张图不超过 4096 patch（1024×1024），2026-10-21/24 板上 896 测过 4/6/8 核、1024 只测过 8 核；Qwen3.5 VL 测过 448、896，1024 未上板。读取预处理输入（`prepared_input_safetensors`）的模板没有 `images`，不接受预设。DINOv3 的 processor 固定缩放到 224×224，没有尺寸预设。

音频预设的录音仍取所选 TOML。ASR 的 `audio_seconds` 取 `audio` 开头的 N 秒，不补静音，录音短于 N 秒时运行时报错，所以用一段至少 8 秒的录音即可套用三个预设；截取后的波形是 CPU 参考的输入，也记在 `report.json` 的 `input_identity.used_frames`。2026-10-21 板测用的是 2、6、8 秒的独立录音，不是截取。TTS 预设的三段中文与 2026-10-21 板测文本相同（12、25、228 字符），`zh_long` 超过 128 字符，按 tokenizer 分段合成；`en_sentence` 未上板。参考音频仍取 TOML。

VLA 预设只写相机数和指令长度，图像尺寸、动作长度、去噪步数和权重仍取所选 TOML。Pi0.5、G0.5、HyVLA、LingBot v2 和 GR00T 的模板默认 3 相机；原来只差这些输入字段的 2 相机模板已并入同名模板，2 相机输入改用 `2cam_*` 预设（对照见 [config_names.json](config_names.json) 的 `merged`）。Wall-OSS Base 仍分 2 相机、3 相机模板，两者的 `[cpu].threads` 按相机数分别调过；套用另一相机数的预设时，CPU 线程数仍取所选 TOML。`2cam_t32` 是各 VLA 族共用的参考输入；GR00T 原 2 相机模板的默认输入是 `2cam_t128`。超出范围时会报出原因：HyVLA 的 `text_tokens` 上限为 64，`*_t128` 不可用；LingBot v2 的上限是 `[input].max_language_tokens`（省略时 72，最大 160），`*_t128` 需同时写 `--set input.max_language_tokens=128`；Pi0.5 只有 T32 支持 4/6 核。Wall-OSS 的 `--check-config` 只检查语法，长度在加载时检查。

Pi0.5 的 `batch`、GR00T 和 LingBot v2 的 `observation` 是已经固定了相机数和文本长度的输入文件。预设改变了形状时，须同时用 `--set input.<键>=<路径>` 给出该形状的文件，否则配置检查会拒绝；加载时还会核对文件内容。Wall-OSS 的 `batch_template` 按 `cameras`、`image_size`、`text_tokens` 展开成文件名；HyVLA 和 G0.5 的 `source = "synthetic"` 由真实 processor 构造输入，不需要另给文件。输入布局固定的模板（G0.5、Wall-OSS、LingBot v2 的 legacy 模板和 HY-Embodied）没有这些字段，不接受预设。

```bash
python examples/run.py --config examples/configs/qwen3/text/8b/fp16.toml --input p4096_d32 --cores 6 \
    --emit-toml qwen3-8b-p4096-6core.toml --check-config
python examples/run.py --config examples/configs/pi05/h50/w8a16.toml --input 2cam_t64 \
    --set input.batch=/path/to/pi05_2cam_t64_batch.pt --emit-toml pi05-2cam-t64.toml --check-config
python examples/run.py --config examples/configs/qwen3_vl/vl/4b/fp16.toml --input i1024_p1280_d32 --check-config
python examples/run.py --config examples/configs/qwen3_asr/1_7b/w8a16.toml --input 2s \
    --set input.audio=/path/to/speech-8s.wav --check-config
```

终端 `[MODEL]` 摘要中的 `config_sha256=…` 是覆盖后完整配置的 SHA-256（键排序后的紧凑 JSON）。通用 LLM / VLM 入口的 `report.json` 在 `provenance.config_sha256` 记录同一值；两次运行的该值相同，表示用的是同一配置。

## 日志与配置检查

`logging.level` 默认为 `ERROR`，`logging.summary` 默认为 `true`。取值、终端显示和日志文件见[日志与结果](../RESULTS.md)。

```bash
python examples/run.py --config /path/to/filled-config.toml --check-config
python examples/run.py --config /path/to/filled-config.toml
```

第一条命令检查字段和登记关系，第二条命令加载资产并运行。旧格式的 RhinoVLA `[env]` 片段不能用 `run.py`，只能交给 `scripts/validate_rhinovla_v3.py`。
