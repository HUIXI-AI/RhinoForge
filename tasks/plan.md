# Whisper base RPU 推理实施计划

## 目标

在 `feat/whisper-asr-rpu` 分支上，为 `csukuangfj/sherpa-onnx-whisper-base` 提供可用的 RhinoForge 离线转写入口：单条、单声道、最长 30 秒、自动语言识别、贪心解码、最多 448 token。CPU 只做音频前后处理和 token 选择，encoder、decoder、KV cache 与 logits 在 RPU 完成，不允许静默回退 CPU。

设计依据：`docs/superpowers/specs/2026-10-21-whisper-asr-rpu-design.md`。

## 范围与影响面

- 公共调用：新增 `WhisperASRModel` 与转写结果类型，并从 `rpu_backend.api` 导出。
- Python 适配：资产锁定、输入校验、音频前处理、tokenizer、RPU 生命周期与生成循环。
- Native：新增 Whisper encoder/decoder fused owner、调度 schema、声明与构建入口。
- 配置与示例：增加固定 Whisper base FP16 配置和最小运行器。
- 测试与夹具：host-only 预检、source contract、参考数值、KV/Graph 生命周期、R1 端到端与性能报告。
- 文档：只有 R1 数值和语义验收通过后，才登记模型支持状态。

不会改动现有 Qwen3、NavDP、CausalLM 的公共行为；不会提交模型权重、音频大文件或本地未跟踪的 `lib/`。

## 执行原则

1. 先验证运行时资产与动态 KV 写入能力，再投入完整 fused owner 开发。
2. 每一步保留一个可独立运行的探针，先做第 0 层，再扩展至 6 层。
3. 复用现有 ASR 输入校验、FusedModelBase 生命周期、Graph BUILD/REPLAY 和报告格式，不引入新依赖。
4. 缺少精确算子时直接形成编译器/运行时团队最小复现，不用数学近似或 CPU fallback 绕过。

## 任务

### T1：锁定模型 profile 与参考夹具

**工作内容**

- 从目标模型目录读取配置、encoder/decoder ONNX、tokenizer 和词表，生成完整 SHA256 manifest。
- 建立固定短音频与 30 秒边界音频的 CPU FP32、CPU FP16、ONNX 参考元数据；大文件只保存在测试机数据目录，不提交仓库。
- 记录语言 token 集合、特殊 token、输入输出名、shape 和 dtype。

**验收标准**

- 同一资产重复生成 manifest 完全一致；任一文件变化都会被检测。
- 参考输出均为有限值，且能复现语言、逐 token 输出、EOS/length 结果。
- 仓库中只提交哈希、shape、token 与期望摘要，不提交模型和音频本体。

**验证命令/证据**

- `pytest -q tests/test_whisper_assets.py`
- 保存参考生成命令、模型来源和 SHA256 清单。

**依赖**：无。

**预计文件**：`python/rpu_backend/adapters/whisper/assets.py`、`tests/test_whisper_assets.py`、`tests/fixtures/whisper_base/manifest.json`、`tools/whisper_reference.py`。

### T2：运行时能力闸门

**工作内容**

- 对照 Whisper 的卷积、affine LayerNorm、ERF GELU、self/cross attention、线性层和动态 KV 追加需求，检查 v1.2.0 运行时/Launch/算子资产。
- 在 R1 上先运行现有 decoder 第 0 层 prefill3/decode1 探针，确认动态位置和 cache 写入语义。
- 若能力缺失，输出最小输入、期望、实测、版本和精确缺口并暂停后续 native 开发。

**验收标准**

- 每个必需能力都有“已有资产 + 可执行证据”，不能只凭符号或文件名判断。
- 第 0 层 hidden、self-K/V 余弦相似度均不低于 0.9999，cache 历史前缀不变。
- 闸门失败时能给编译器/运行时团队直接复现，且公共 API 仍不可用。

**验证命令/证据**

- R1 探针日志、运行时版本、资产清单和余弦相似度报告。

**依赖**：T1。

**预计文件**：`tests/test_whisper_runtime_capability.py`、`docs/whisper-runtime-capability.md`；优先复用 `primebot_deploy` 中已有探针，不复制大文件。

### T3：公共 API 的失败前置与输入边界

**工作内容**

- 新增 `WhisperASRModel.from_pretrained()`、转写结果类型和 context manager 生命周期骨架。
- 复用现有 ASR 音频校验/重采样路径；在访问设备和加载完整权重前校验 profile、哈希、采样率、时长、空值、静音及 NaN/Inf。
- Native 尚未接通时明确报告“不支持/资产不匹配”，不回退 ONNX Runtime。

**验收标准**

- 无效资产和音频均在 RPU 初始化前失败，错误指出具体文件或输入条件。
- 最大 30 秒、batch 1、单声道边界固定；超过边界不自动截断。
- 现有 ASR API 测试不回归。

**验证命令/证据**

- `pytest -q tests/test_whisper_asr.py tests/test_qwen3_asr.py`

**依赖**：T1、T2 通过。

**预计文件**：`python/rpu_backend/api/whisper_asr.py`、`python/rpu_backend/api/__init__.py`、`python/rpu_backend/adapters/whisper/__init__.py`、`tests/test_whisper_asr.py`。

### T4：Whisper encoder native 最小闭环

**工作内容**

- 注册独立 Whisper encoder owner，接通构建、调度和销毁生命周期。
- 先完成卷积前端与单层 encoder 探针，再扩展到固定 6 层和最终 LayerNorm。
- 固定输入 `[1, 80, 3000]`、输出 `[1, 1500, 512]`，短音频使用参考一致的 padding/mask。

**验收标准**

- 非 Whisper profile、错误 shape/dtype、重复 owner 和部分安装失败均安全拒绝或清理。
- R1 encoder 输出有限，与同精度 ONNX 基准余弦相似度不低于 0.9999。
- 相同 Graph 签名只 BUILD 一次，后续稳定 REPLAY。

**验证命令/证据**

- `pytest -q tests/test_whisper_cpp_source_contract.py`
- R1 encoder 数值与 Graph 报告。

**依赖**：T2、T3。

**预计文件**：`src/fused/rpu_whisper_encoder.cpp`、`src/core/rpu_kernel_decls.h`、`src/core/rpu_dispatch_registrations.inc`、`src/CMakeLists.txt`、`tests/test_whisper_cpp_source_contract.py`。

### T5：encoder Python 安装与执行适配

**工作内容**

- 将 manifest 权重按 encoder owner 要求安装，转换一次并保持明确所有权。
- 实现 encoder Graph 的首次 BUILD、后续 REPLAY、输出持有和关闭清理。
- 暴露仅供模型 API 使用的 encoder 调用，不新增通用抽象层。

**验收标准**

- 同一实例重复 encoder 输入会刷新结果，Python 可见输出不共享可变存储。
- 安装中途失败不可重试，关闭幂等且无句柄/设备缓存泄漏。
- host mock 生命周期测试和 R1 数值测试均通过。

**验证命令/证据**

- `pytest -q tests/test_whisper_runtime.py -k encoder`
- R1 重复输入与关闭重开报告。

**依赖**：T4。

**预计文件**：`python/rpu_backend/adapters/whisper/runtime.py`、`python/rpu_backend/adapters/whisper/__init__.py`、`tests/test_whisper_runtime.py`。

### T6：decoder 第 0 层 prefill/decode 闭环

**工作内容**

- 新增 decoder owner，精确实现第 0 层 self-attention、cross-attention、FFN 和三段 pre-norm residual。
- 支持固定 prompt prefill 与单 token decode；当前位置作为受校验的 mutable 输入。
- 建立 self-K/V 追加和 cross-K/V 一次性准备的最小闭环。

**验收标准**

- prefill 与连续 decode 的 hidden、self-K/V、cross-K/V 余弦相似度均不低于 0.9999。
- decode 只写当前位置，历史 cache 前缀逐位不变；位置倒退、越界和长度不一致立即失败。
- 同一 decode Graph 不因 token/position 变化重新 BUILD。

**验证命令/证据**

- `pytest -q tests/test_whisper_decoder_contract.py`
- R1 第 0 层 prefill/decode 数值与 cache 差分报告。

**依赖**：T2、T5。

**预计文件**：`src/fused/rpu_whisper_decoder.cpp`、`src/core/rpu_kernel_decls.h`、`src/core/rpu_dispatch_registrations.inc`、`src/CMakeLists.txt`、`tests/test_whisper_decoder_contract.py`。

### T7：完整 6 层 decoder、KV 与 vocabulary head

**工作内容**

- 将已验证的第 0 层路径扩展到 6 层，不引入另一套执行路径。
- 每条音频只生成一次 6 层 cross-K/V；self-K/V 容量固定 448 token。
- 接入 token/position embedding、最终 LayerNorm 和 vocabulary head，RPU 返回 logits。

**验收标准**

- 完整 decoder hidden、各层 K/V 与 logits 均有限，关键张量余弦相似度不低于 0.9999。
- 新音频开始时旧 self/cross cache 被清空或销毁；不得跨音频复用。
- warmup 0/1/3 的 logits 和 token 一致，Graph cache 大小不增长。

**验证命令/证据**

- `pytest -q tests/test_whisper_runtime.py -k decoder`
- R1 6 层、448 容量、跨音频和 Graph invariant 报告。

**依赖**：T6。

**预计文件**：`src/fused/rpu_whisper_decoder.cpp`、`python/rpu_backend/adapters/whisper/runtime.py`、`tests/test_whisper_runtime.py`、`tests/test_whisper_decoder_numeric.py`。

### T8：自动语言识别与完整贪心转写

**工作内容**

- position 0 只在 tokenizer 定义的合法语言 token 集合内 argmax。
- 追加 language、transcribe、no-timestamps prompt，再以单 token decode 生成至 EOS 或 448。
- 返回 `text`、`language`、`token_ids`、`finish_reason`，CPU 只做 argmax/EOS/tokenizer。

**验收标准**

- 固定音频的语言、每一步 token、结束原因和文本与 ONNX 参考完全一致。
- 正常 EOS 与达到 448 上限都返回已有文本，后者标记 `length`。
- 生成阶段无 ONNX Runtime 或其他 CPU 模型计算调用。

**验证命令/证据**

- `pytest -q tests/test_whisper_generation.py tests/test_whisper_asr.py`
- R1 固定短音频逐 token 对照报告。

**依赖**：T7。

**预计文件**：`python/rpu_backend/api/whisper_asr.py`、`python/rpu_backend/adapters/whisper/runtime.py`、`tests/test_whisper_generation.py`、`tests/test_whisper_asr.py`。

### T9：示例、固定配置与性能报告

**工作内容**

- 增加唯一的 Whisper base FP16 固定配置和最小命令行示例。
- 复用现有 ASR `report.json` 风格，记录预处理、encoder、语言识别、prefill、decode P50/P95、token/s、总耗时、RTF、BUILD/REPLAY。
- 固定 2 次 warmup、10 次测量；性能仅报告，不作为首期通过门槛。

**验收标准**

- 用户只需提供模型目录和音频即可运行；默认配置不能悄悄切换 CPU。
- 报告字段完整、单位明确、重复运行不会混入 warmup。
- 示例 contract 测试与现有示例测试通过。

**验证命令/证据**

- `pytest -q tests/test_whisper_audio_examples.py tests/test_qwen3_audio_examples.py`
- R1 `report.json`。

**依赖**：T8。

**预计文件**：`examples/runners/whisper_audio/runner.py`、`examples/configs/whisper_base/fp16.toml`、`examples/configs/whisper_base/README.md`、`tests/test_whisper_audio_examples.py`。

### T10：R1 回归、支持登记与交付

**工作内容**

- 执行短音频、30 秒音频、EOS、448 上限、重复输入、warmup、BUILD/REPLAY 和错误边界矩阵。
- 运行相关 host 回归与 R1 数值/语义测试，保存环境、版本、命令和结果。
- 只有全部硬门槛通过后才更新模型支持表、移植文档和使用说明。

**验收标准**

- 设计文档 7.1、7.2 的所有硬门槛有可追溯证据；7.3 性能数据完整。
- 现有 Qwen3、NavDP 相关测试无回归。
- 支持状态与实际验证范围一致，不把未验证功能写成支持。

**验证命令/证据**

- 运行 Whisper、Qwen3 ASR、NavDP 相关 pytest 集合。
- R1 最终验收报告和干净的 `git status --short`（用户自有 `lib/` 除外）。

**依赖**：T9。

**预计文件**：`docs/model_support.md`、`docs/model_porting.md`、`examples/configs/whisper_base/README.md`、`docs/reports/whisper-base-r1.md`。

## 依赖关系与检查点

```text
T1 -> T2 --能力通过--> T3 -> T4 -> T5 -> T6 -> T7 -> T8 -> T9 -> T10
          \\能力缺失--> 最小复现与缺口报告，暂停 native 实现
```

- 检查点 A（T2 后）：确认精确算子与动态 KV 能力齐全，才进入 API/native 实现。
- 检查点 B（T5 后）：encoder 数值和生命周期通过，才扩展 decoder。
- 检查点 C（T6 后）：第 0 层 prefill/decode 与 cache 语义通过，才扩展 6 层。
- 检查点 D（T8 后）：端到端 token/text 一致，才补示例、性能和公开支持状态。

## 完成定义

- 公共 API 可在 R1 完整转写目标模型，自动语言、token、文本和结束原因与参考一致。
- encoder、decoder 与 KV 数值达到设计阈值，且 Graph/cache 生命周期通过。
- 缺失或错误资产、非法音频和 cache 状态都能在安全边界明确失败。
- 相关 host/R1 回归通过，文档、配置、示例和证据与代码同步。
- 工作区不包含模型权重、音频大文件，也不触碰用户自有 `lib/`。

## 已知风险

- v1.2.0 的 Launch/算子资产可能没有覆盖 Whisper 精确数学或动态 KV 地址更新；T2 是硬闸门。
- 完整 6 层 cross-K/V 约占 18 MiB，需以 R1 实测确认 DDR 分配和生命周期，不以估算代替。
- Whisper tokenizer/特殊 token 必须与目标 checkpoint 完全匹配，不能拿其他 base 模型替换。

## 待用户确认

批准本计划后从 T1 开始实施。若 T2 发现资产缺口，将先交付可复现报告并等待相应运行时/编译器资产，不使用近似方案继续。
