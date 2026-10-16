# LightNav-0 运行与缓存

简体中文 | [English](lightnav_0_r1.md)

本适配面向公开模型 `LightOriginsHQ/LightNav-0`，固定使用 Hugging Face revision
`826dc5fbfa37afa8293d2e336d329b6ffc0bfb64` 和上游 LightNav 源码 commit
`3015508b70fb6e30bbd66701e94c9b916dcb8df8`。模型每次固定输出 5 个 token：
2 个指向 token 和 3 个 RVQ 动作 token。

RhinoForge 复用现有 Qwen3-VL 4B 文本、视觉和 DeepStack 路径。LightNav 的
156048 行词表不能使用要求输出维度按 128 对齐的融合列切分输出头，因此 decoder
返回 hidden states 后，通过 TP8 行切分输出头计算 logits。该分支只允许精确的
LightNav 配置，不放宽其他扩展词表模型。

视觉缓存沿用上游按 2 帧 tubelet 建立的 key。每个 miss 只运行一次 ViT，并把
pooler output 与 3 层 DeepStack 特征保存到 CPU；同一会话后续窗口直接复用。
缓存继续保存池化前特征，并额外按“精确 tubelet 序列 + SlowFast pooling factor”
缓存整段 post-ViT pooling 结果；历史组成或倍率变化时 key 不匹配，会重新池化。
滑动窗口位置会变化，所以每次完整 scatter 缓存特征，不启用 prefix scatter 复用。

## 运行

先准备 RhinoForge、模型目录和 LightNav 上游源码。模型目录必须包含权重、
`eval_config.json`、processor/tokenizer 文件和 `action_tokenizer/`。

```bash
python examples/lightnav_benchmark.py \
  --checkpoint /path/to/LightNav-0 \
  --lightnav-source /path/to/LightNav-0-source \
  --output benchmark_results/lightnav_0_r1.json \
  --check-config
```

R1 运行时移除 `--check-config`。12/24/36/48/64 表示过去的历史帧数，每档额外
加入 1 帧当前画面，所以实际输入 13/25/37/49/65 帧。两个场景各测 5 次取中位数，
每次固定输出 5 个 token。输入使用确定性的 256×448 合成帧，因此结果用于比较
推理性能和缓存收益，不代表导航任务准确率。

若业务已经采用上游 `VLN_EVAL_TRAJ_TOP1` action allowlist，可增加
`--constrained-head`。它在冷装载时只保留 4369 个候选 token 的输出头，物理补齐到
4480 行后使用 column-parallel Linear。该选项改变默认全词表解码范围，必须单独验证
真实导航质量。

连续导航请求可增加 `--history-draft`。脚本把上一请求的 5 个 token 作为候选，将
前 4 个候选追加到当前 prompt，一次 prefill 同时复验 5 个位置；全部一致时跳过
串行 decode，任一不一致时自动回退原有的 5-token 贪心解码。首个请求仍用串行解码
产生候选，不需要额外的 MTP 权重。

完整性能测试命令如下：

```bash
python examples/lightnav_benchmark.py \
  --checkpoint /path/to/LightNav-0 \
  --lightnav-source /path/to/LightNav-0-source \
  --history-frames 12,24,36,48,64 \
  --repeats 5 \
  --constrained-head \
  --history-draft \
  --output benchmark_results/lightnav_0_r1_history_draft.json
```

## 历史候选复验

上一请求的输出只是候选，不会直接作为本轮结果。连续导航的相邻画面和动作通常变化
较小，因此上一轮动作有较高概率继续成立；当前请求仍使用自己的视觉输入、prompt 和
模型权重完成验证。候选不一致时回退标准解码，所以候选质量只影响性能，不改变标准
贪心解码结果。

对于当前 prompt `P` 和候选 `[d1, d2, d3, d4, d5]`，自回归位置关系为：

| 输入位置 | 该位置的 logits 用于预测 |
|---|---|
| `P` 的最后一个 token | `d1` |
| `d1` | `d2` |
| `d2` | `d3` |
| `d3` | `d4` |
| `d4` | `d5` |

因此验证输入是 `P + [d1, d2, d3, d4]`。一次 prefill 保留最后 5 个位置的 logits，
分别取贪心 token `[y1, y2, y3, y4, y5]`，再与候选逐项比较。`d5` 不追加，因为它
对应的 logits 将预测第 6 个 token，而 LightNav-0 固定只输出 5 个。

执行流程为：

1. 首个请求按标准方式执行一次 prefill 和 4 次单 token decode，产生初始候选；
2. 后续请求追加候选前 4 个 token，在一次 prefill 中计算 5 个预测位置；
3. 五个位置全部一致时直接返回候选，`decode_5_ms` 记为 0；
4. 任一位置不一致时清理验证 Graph、复位 KV cache，并执行标准串行解码；
5. 本轮最终输出成为下一请求的候选。

这属于带完整目标模型复验的时序投机解码。它利用连续请求的时间相关性，不需要
checkpoint 提供 MTP head；接受时把 4 次单 token decode 合并进一次稍长的 prefill。

### R1 v1.2.0 实测

测试使用确定性 256×448 合成输入、4369-token action allowlist、每种历史长度和缓存
场景测 5 次取中位数，Rhino Graph 预热时间不计入。全历史缓存命中时结果如下：

| 历史帧 | 准备 | 视觉 | Prefill/复验 | Decode | 总耗时 | 优化前总耗时 | 降幅 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 23.8 ms | 72.2 ms | 150.7 ms | 0 ms | 247.1 ms | 458.4 ms | 46.1% |
| 24 | 19.3 ms | 70.6 ms | 221.4 ms | 0 ms | 311.4 ms | 526.6 ms | 40.9% |
| 36 | 23.4 ms | 74.1 ms | 292.2 ms | 0 ms | 390.0 ms | 605.8 ms | 35.6% |
| 48 | 28.4 ms | 71.3 ms | 300.9 ms | 0 ms | 402.2 ms | 615.5 ms | 34.7% |
| 64 | 29.8 ms | 75.2 ms | 314.7 ms | 0 ms | 420.1 ms | 631.7 ms | 33.5% |

全未命中场景的优化后总耗时依次为 598.0/968.8/1311.2/1415.9/1584.0 ms，较优化前
下降 26.3%/18.2%/11.6%/13.8%/11.9%。50 个正式计时请求均接受候选。该接受率来自
重复的确定性输入，只用于证明优化路径和性能上限；真实连续导航需另外统计接受率和
任务准确率。

每个历史帧档位先预热完整窗口和仅当前 tubelet 两种 Rhino Graph，建图耗时不计入
结果；档位内部始终复用同一 KV cache 地址。档位结束后释放 Graph，再进入下一档，
避免多种 prefill 形状同时占满 Graph arena。

每档结果包括：

- `all_miss_median`：每次使用空视觉缓存，所有被 SlowFast 选中的 tubelet 都重新计算；
- `all_history_hit_median`：预先缓存所有不含当前帧的 tubelet，只有包含当前帧的
  1 个 tubelet 重新计算；
- `cache_hits/cache_misses`：tubelet 级命中和缺失；
- `prepare_sample_ms/prepare_processor_ms`：样本构建和 processor 耗时；
- `vision_lookup_ms/vision_vit_ms/vision_pool_ms`：视觉缓存查找、ViT 和 pooling 耗时；
- `prefill_ms/decode_5_ms/total_ms`：文本预填充、5-token 解码和端到端耗时。
- `draft_attempts/draft_accepts`：历史候选复验次数和全部 5 token 接受次数。

启用 `--history-draft` 后，`prefill_ms` 包含候选复验；候选接受时
`decode_5_ms` 为 0。应同时比较 `total_ms`，避免把阶段迁移误认为端到端优化。

当前 SlowFast 配置下，12/24/36/48/64 帧历史加当前帧分别产生约
288/456/568/624/708 个池化后视觉 token。脚本同时记录实际完整输入 token 数，
运行时仍以 Qwen3-VL 4B 的 P4176 准入上限为准。

脚本对缓存和重算的 pooler/DeepStack 特征使用 `1e-3` FP16 绝对容差检查。

## 执行说明

processor 只处理当前未缓存的 tubelet；精确 segment pooling 复用已池化历史，
空缓存仍运行完整窗口的视觉路径。视觉 token 和 M-RoPE 位置变化会使旧 KV 前缀
失效，因此每次执行完整 prefill。默认模式下五个 token 按自回归顺序生成，使用融合
decoder 和设备 argmax；历史候选模式在一次 prefill 中复验全部 5 个 token。

约束输出头沿用上游 action allowlist，候选范围缩小可能改变 token 选择，应与应用的
解码约定一致。历史候选的收益取决于连续请求的接受率；checkpoint 不包含多 token
预测头或权重。
