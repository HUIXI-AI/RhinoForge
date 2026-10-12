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

R1 运行时移除 `--check-config`。12/24/36/48/64 表示过去的历史帧数，每档额外
加入 1 帧当前画面，所以实际输入 13/25/37/49/65 帧。两个场景各测 5 次取中位数，
每次固定输出 5 个 token。输入使用确定性的 256×448 合成帧，因此结果用于比较
推理性能和缓存收益，不代表导航任务准确率。

每个历史帧档位先预热完整窗口和仅当前 tubelet 两种 RPU Graph，建图耗时不计入
结果；档位内部始终复用同一 KV cache 地址。档位结束后释放 Graph，再进入下一档，
避免多种 prefill 形状同时占满 Graph arena。

每档结果包括：

- `all_miss_median`：每次使用空视觉缓存，所有被 SlowFast 选中的 tubelet 都重新计算；
- `all_history_hit_median`：预先缓存所有不含当前帧的 tubelet，只有包含当前帧的
  1 个 tubelet 重新计算；
- `cache_hits/cache_misses`：tubelet 级命中和缺失；
- `prepare_ms/vision_ms/prefill_ms/decode_5_ms/total_ms`：各阶段及端到端耗时。

当前 SlowFast 配置下，12/24/36/48/64 帧历史加当前帧分别产生约
288/456/568/624/708 个池化后视觉 token。脚本同时记录实际完整输入 token 数，
运行时仍以 Qwen3-VL 4B 的 P4176 准入上限为准。

## R1 实测结果（2026-10-12）

两个场景均排除 Graph 构建，并各运行 5 次取中位数。

| 历史 + 当前帧 | 输入 token | 全未命中 (ms) | 历史全命中 (ms) | tubelet 命中/缺失 | 端到端降幅 |
|---:|---:|---:|---:|---:|---:|
| 12 + 1 | 414 | 897.6 | 532.2 | 6 / 1 | 40.7% |
| 24 + 1 | 630 | 1288.8 | 616.5 | 12 / 1 | 52.2% |
| 36 + 1 | 790 | 1643.7 | 718.7 | 17 / 1 | 56.3% |
| 48 + 1 | 864 | 1763.3 | 738.1 | 19 / 1 | 58.1% |
| 64 + 1 | 975 | 1934.2 | 762.7 | 22 / 1 | 60.6% |

缓存历史与空缓存重算的视觉特征最大绝对差异为 `0.00048828125`，通过 `1e-3`
FP16 容差检查。完整原始结果见
`benchmark_results/lightnav_0_r1_20261012.json`。
