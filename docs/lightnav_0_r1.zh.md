# LightNav-0 R1 迁移与评测

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
post-ViT spatial pooling 仍按当前 SlowFast segment 的 factor 在 CPU 执行，缓存中
保存池化前特征，避免错误复用不同 pool factor 的结果。滑动窗口位置会变化，所以
每次完整 scatter 缓存特征，不启用 prefix scatter 复用。

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

R1 运行时移除 `--check-config`。脚本固定 12/24/36/48/64 帧、6 次连续滑窗预热、
5 次稳态测量和 5 个输出 token。输入使用确定性的 256×448 合成帧，因此结果用于
比较推理性能和缓存收益，不代表导航任务准确率。6 次预热用于覆盖 48/64 帧下
SlowFast burst segment 的配对周期，使稳态请求只剩最新 tubelet miss。
每个历史帧档位结束后释放该档 RPU Graph，再预热下一档，避免多种 prefill
形状同时占满 Graph arena；档位内部始终复用同一 KV cache 地址。

每档结果包括：

- `cold`：RPU Graph 已预热、视觉缓存为空的首次完整请求；图构建不计入结果；
- `cached_repeat`：同一请求的全命中上限，并逐元素检查 pooler 与 3 层 DeepStack 特征；
- `steady_median`：连续单帧滑窗预热后的 5 次中位数；
- `sliding_fresh`：用空缓存重算首个稳态滑窗，检查视觉特征最大差异不超过 `1e-3`；
- `cache_hits/cache_misses`：tubelet 级命中和缺失；
- `prepare_ms/vision_ms/prefill_ms/decode_5_ms/total_ms`：各阶段及端到端耗时。

当前 SlowFast 配置下，12/24/36/48/64 帧分别产生约
252/420/568/624/708 个池化后视觉 token。脚本同时记录实际完整输入 token 数，
运行时仍以 Qwen3-VL 4B 的 P4176 准入上限为准。

## R1 实测结果（2026-10-10）

使用确定性合成帧、6 次预热和 5 次测量。`cold` 是 Graph 已预热但视觉缓存为空的
单次请求；`steady` 是连续滑窗中位数。

| 历史帧 | 输入 token | cold (ms) | 全命中 (ms) | steady (ms) | steady 命中/缺失 | 相对 cold 降幅 |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 378 | 854.0 | 443.3 | 517.0 | 5 / 1 | 39.5% |
| 24 | 594 | 1262.5 | 546.2 | 607.1 | 11 / 1 | 51.9% |
| 36 | 790 | 1603.6 | 634.0 | 710.1 | 17 / 1 | 55.7% |
| 48 | 864 | 1782.2 | 654.1 | 733.4 | 19 / 1 | 58.8% |
| 64 | 975 | 1932.1 | 680.4 | 770.7 | 22 / 1 | 60.1% |

同窗缓存特征逐元素一致；滑窗缓存与 fresh 重算的最大绝对差异为
`0.00048828125`，不超过一个 FP16 舍入步长。完整原始结果见
`benchmark_results/lightnav_0_r1_20261010.json`。
