# Whisper base 的 RhinoForge RPU 推理设计

日期：2026-10-21

实现分支：`feat/whisper-asr-rpu`

基础分支：`compat/runtime-v1.2.0`

## 1. 目标与范围

首期支持 `csukuangfj/sherpa-onnx-whisper-base` 的完整离线转写：输入一条不超过 30 秒的单声道音频，由模型自动判断语言，使用贪心解码输出文本。

固定范围如下：

- batch 为 1；
- 输入统一重采样到 16 kHz；
- Whisper encoder 音频上下文为 1500；
- decoder 为 6 层，hidden size 512、8 个 attention head、head dim 64、FFN 2048；
- 最多生成 448 token；
- 使用 FP16 RPU 计算；
- 不支持流式识别、时间戳、beam search 和批量推理。

CPU 只负责音频读取、重采样、log-mel 特征、贪心 token 选择和 tokenizer 文本解码。Whisper encoder、decoder、cross-attention、KV cache 和 logits 均在 RPU 执行，不允许静默回退 CPU。

## 2. 能力评审结论

```text
outcome:       New fused subsystem
certification: uncertified
```

已有证据表明 encoder 静态图可在 R1 运行，固定 448-token decoder 静态图的 logits 余弦相似度为 0.99998048；真实 decoder 第 0 层的固定 3-token prefill 和 1-token decode 探针也已通过，RPU self-K/V 可连续传递。

这些证据不足以登记完整支持，因为 RhinoForge 当前没有 Whisper 适配器、6 层 encoder-decoder owner、完整 logits 入口和对应 Graph 生命周期。正式支持还需要与源码匹配的 Launch 和算子资产，并完成本文定义的端到端验收。

## 3. 公共接口

新增独立的语音入口，不复用 Qwen3、NavDP 或通用 CausalLM 的模型身份：

```python
from rpu_backend.api import WhisperASRModel

model = WhisperASRModel.from_pretrained(checkpoint_dir)
result = model.transcribe(audio, sample_rate=sample_rate)
```

结果包含：

- `text`：最终转写文本；
- `language`：模型检测出的语言；
- `token_ids`：完整生成 token；
- `finish_reason`：`eos` 或 `length`。

达到 448 token 时返回已经生成的文本并标记 `length`，不静默丢弃结果。

## 4. 数据流

```text
音频
  -> CPU 重采样与 80 维 log-mel，补齐到 [1, 80, 3000]
  -> RPU encoder，生成 [1, 1500, 512] 音频特征
  -> RPU decoder position 0，使用起始 token 生成语言 logits
  -> CPU 仅在合法语言 token 集合内执行 argmax
  -> RPU prompt prefill，写入语言、transcribe、no-timestamps token
  -> RPU 单 token decode，逐步追加 self-K/V 并生成 logits
  -> CPU argmax、EOS 判断与 tokenizer 文本解码
```

自动语言识别保持 Whisper 原始语义：第一次 decoder 输出只在 tokenizer 声明的语言 token 集合中选择，不用通用词表 argmax 代替。

## 5. 模块与所有权

### 5.1 Python API 与资产加载

`WhisperASRModel` 负责输入验证、生命周期和公共返回值。Whisper 资产加载器校验配置、ONNX、tokenizer 和词表的固定 SHA256，并在读取完整权重前拒绝不匹配的 profile。

当前已验证 decoder ONNX 的 SHA256 为 `8a12c3f6ad65bb5b86d7e6eccc302378f20f9fb2df6cb10747c62895da7ac194`。其余文件的哈希必须由首个资产锁定改动一次性写入 manifest；在完整 manifest 合入前，公共入口保持不可用，不接受近似文件或同名替换。

音频前处理复用现有 ASR 音频校验和重采样能力。超长、空、全静音或包含 NaN/Inf 的输入直接失败，不自动截断。

### 5.2 RPU encoder owner

新增专用 Whisper encoder fused owner，覆盖卷积前端、绝对位置编码、6 层双向 self-attention、affine LayerNorm、ERF GELU、FFN 和最终 LayerNorm。

encoder 使用固定 30 秒 Graph。较短音频通过特征 padding 和有效长度 mask 保持参考语义。输出存储独立于 Graph 内部临时缓冲，允许 decoder 在整个生成周期持有。

### 5.3 RPU decoder owner

新增专用 Whisper decoder fused owner，保持每层三段 pre-norm residual：

1. causal self-attention；
2. 对 encoder memory 的 cross-attention；
3. ERF GELU FFN。

每层 self-K/V 随 token 位置追加，最大 448 行。历史 cache 前缀不可被后续 decode 修改。每层 cross-K/V 对当前音频只计算一次并保存在设备 DDR；按 FP16、1500×512、K/V 各一份计算，6 层合计约 18 MiB，不尝试全部驻留单核 SPM。

decoder owner 同时绑定 token embedding、绝对位置 embedding、最终 LayerNorm 和 vocabulary head。权重只转换一次，部分安装失败后的模型实例不可重试。

### 5.4 Graph 与可变输入

使用三个执行阶段：

- encoder 固定图；
- prompt prefill 固定图；
- 单 token decode 重放图。

Graph 签名包含模型 profile、精度、阶段、batch、几何和 cache 容量。token、当前位置和当前输出地址通过受校验的 mutable 输入更新；cache 写入位置必须由运行时参数驱动，不为 448 个位置分别保留 Graph。

同一签名第一次调用只能 BUILD 一次，后续调用必须稳定 REPLAY。新音频开始时销毁或清空旧 self-K/V 和 cross-K/V 所有权，禁止跨音频复用。

## 6. 错误处理

- 模型、tokenizer、配置、Launch 或算子资产不匹配时，在完整权重加载和模型所有权声明前失败。
- 音频长度、dtype、shape 或数值非法时，在访问 RPU 前失败。
- 缺少 Whisper 所需的 RPU 运算或动态 cache 写入能力时明确报错，不使用 CPU fallback 掩盖缺口。
- encoder 或 decoder 初始化中途失败时，按相反顺序销毁 native handle、Graph 和设备缓存。
- 一个进程只允许一个完整 Whisper RPU 模型实例；关闭模型后才释放所有权。
- decode 超过 cache 容量、位置倒退或 cache 长度不一致时立即失败，不修正调用方状态。

## 7. 验收标准

### 7.1 数值与语义

- encoder 最终输出有限，与同精度 ONNX 基准的余弦相似度不低于 0.9999；
- decoder hidden、self-K/V 和 cross-K/V 与同精度基准的余弦相似度不低于 0.9999；
- 固定测试音频的自动语言结果、每一步贪心 token、结束原因和最终文本与 ONNX 基准一致；
- cache 更新只追加当前 token，所有历史前缀保持不变；
- CPU FP32 anchor 单独记录误差，不用一个全仓库通用阈值替代上述任务语义门槛。

### 7.2 生命周期

- `warmup=0`、`1` 和 `3` 的 token 与文本一致；
- 同一 Graph 签名一次 BUILD 后稳定 REPLAY，cache size 不增长且 `cache_invariant_ok()` 保持为真；
- 相同 shape、不同音频和 token 的输入必须刷新，不能复用旧值；
- 多次转写返回的 Python 可见输出不得共享可变存储；
- 覆盖短音频、30 秒音频、正常 EOS 和达到 448-token 上限。

### 7.3 性能记录

首期不设业务性能硬门槛。固定使用 2 次 warmup 和 10 次测量，记录音频前处理、encoder、语言识别、prompt prefill、decode P50/P95、token/s、总耗时、RTF，以及 Graph BUILD 与 REPLAY 的单独耗时。

性能数据只有在数值、语义和 Graph 生命周期全部通过后才有效。

## 8. 实施顺序

1. 锁定完整模型资产，建立 CPU FP32、CPU FP16 和 ONNX 参考夹具。
2. 实现资产预检、公共配置和 host-only 失败测试。
3. 实现 6 层 encoder owner，并完成 R1 数值验证。
4. 复现现有 decoder 第 0 层 prefill/decode 探针，再扩展到 6 层。
5. 加入 cross-K/V 一次性准备、448-token self-K/V 和 vocabulary head。
6. 接入自动语言识别、贪心生成、tokenizer 和公共 `transcribe` API。
7. 完成最大范围、Graph、重复输入、错误边界和端到端 R1 验收。
8. 验收全部通过后再更新公开模型支持表和使用文档。

## 9. 非目标

首期不实现流式识别、时间戳、beam search、batch 大于 1、量化模型、训练、动态超过 30 秒的输入，也不为了复用现有 owner 而近似 Whisper 数学。
