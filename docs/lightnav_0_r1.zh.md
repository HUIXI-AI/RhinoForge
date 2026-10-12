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

每个历史帧档位先预热完整窗口和仅当前 tubelet 两种 RPU Graph，建图耗时不计入
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

当前 SlowFast 配置下，12/24/36/48/64 帧历史加当前帧分别产生约
288/456/568/624/708 个池化后视觉 token。脚本同时记录实际完整输入 token 数，
运行时仍以 Qwen3-VL 4B 的 P4176 准入上限为准。

## 阶段分析与优化

- **prepare**：历史命中后为 21～28 ms，其中 processor 约 15～21 ms。processor 已只
  patchify 当前 miss tubelet；继续缓存完整 prompt 会受时间戳和 SlowFast 布局变化影响，
  本次不改变。
- **vision**：64 帧 profiling 中，优化前约 174 ms，其中一次当前 tubelet ViT 约
  77 ms，重复 pooling/拼接约 87 ms。现已增加精确 segment pooling LRU；pooling 降至
  约 3 ms，视觉阶段稳定在 78～82 ms。空缓存仍使用整段批量 ViT 和 pooling。
- **prefill**：随完整输入从约 156 ms 增至 315 ms。视觉 token、M-RoPE 位置和当前帧
  会共同改变 KV，直接保留旧 KV 不能保证正确，本次保持完整 prefill。
- **decode**：固定 5 token 为约 242～246 ms；每个 token 依赖前一个 token，且当前
  已使用 RPU fused decoder 和设备 argmax，因此不增加改变生成语义的并行或量化路径。

pooling 缓存优化前后，历史全命中端到端耗时如下：

| 历史帧 | 优化前 (ms) | 优化后 (ms) | 进一步降低 |
|---:|---:|---:|---:|
| 12 | 532.2 | 500.1 | 6.0% |
| 24 | 616.5 | 572.4 | 7.1% |
| 36 | 718.7 | 647.3 | 9.9% |
| 48 | 738.1 | 652.2 | 11.6% |
| 64 | 762.7 | 668.9 | 12.3% |

## RhinoVLA 优化手段与解码评估

RhinoVLA 可迁移的原则是稳定 Graph、持久缓存、缩小重复计算范围和使用匹配几何的
并行 Linear。LightNav 已使用稳定 KV 地址、Graph replay 和视觉缓存；本轮把默认
156048 行输出投影缩小为上游 action allowlist 的 4369 个逻辑候选，并补齐到 4480 行
启用 column-parallel 路径。该模式为冷态显式 opt-in，默认全词表路径不分配候选权重。

投机解码探针尝试用上一请求的 5 个 token 作为 draft，并用一次 4-token target
continuation 验证。R1 当前 planner 在 `position=414, seq_len=4` 返回 `NO_FEASIBLE`；
退化为 4 次单 token 验证没有性能收益，因此未保留该实现。MTP 还需要 checkpoint
提供 multi-token prediction heads 和训练权重，当前 LightNav-0 均不包含，不能只靠
运行时开关启用。

action allowlist 输出头的历史全命中结果如下；全词表基线来自上表相同测试口径：

| 历史帧 | prepare | vision | prefill | decode 5 | total | 相对全词表 total |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 22.5 | 77.3 | 149.9 | 214.6 | 468.6 | -6.3% |
| 24 | 23.0 | 80.1 | 220.6 | 216.4 | 541.4 | -5.4% |
| 36 | 25.9 | 77.2 | 290.7 | 217.6 | 612.3 | -5.4% |
| 48 | 24.8 | 81.3 | 294.6 | 217.4 | 618.5 | -5.2% |
| 64 | 28.3 | 78.5 | 311.1 | 218.1 | 637.7 | -4.7% |

5-token decode 相对全词表降低约 11%。候选范围和 FP16 并行布局会改变 token 选择，
该性能结果不构成任务准确率结论。完整结果见
`benchmark_results/lightnav_0_r1_constrained_20261012.json`。

## R1 实测结果（2026-10-12）

两个场景均排除 Graph 构建，并各运行 5 次取中位数。

| 历史 + 当前帧 | 输入 token | 全未命中 (ms) | 历史全命中 (ms) | tubelet 命中/缺失 | 端到端降幅 |
|---:|---:|---:|---:|---:|---:|
| 12 + 1 | 414 | 886.8 | 500.1 | 6 / 1 | 43.6% |
| 24 + 1 | 630 | 1283.4 | 572.4 | 12 / 1 | 55.4% |
| 36 + 1 | 790 | 1639.9 | 647.3 | 17 / 1 | 60.5% |
| 48 + 1 | 864 | 1744.6 | 652.2 | 19 / 1 | 62.6% |
| 64 + 1 | 975 | 1922.8 | 668.9 | 22 / 1 | 65.2% |

历史全命中场景的阶段中位数：

| 历史帧 | prepare | 当前 ViT | pooling | vision | prefill | decode 5 | total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 23.0 | 74.6 | 2.7 | 77.5 | 155.5 | 242.0 | 500.1 |
| 24 | 21.5 | 77.6 | 3.3 | 81.2 | 225.7 | 243.4 | 572.4 |
| 36 | 25.9 | 78.3 | 2.6 | 80.9 | 295.4 | 244.4 | 647.3 |
| 48 | 27.0 | 78.0 | 3.1 | 81.6 | 299.1 | 244.9 | 652.2 |
| 64 | 28.2 | 76.5 | 2.7 | 79.5 | 314.8 | 246.2 | 668.9 |

缓存历史与空缓存重算的视觉特征最大绝对差异为 `0.00048828125`，通过 `1e-3`
FP16 容差检查。完整原始结果见
`benchmark_results/lightnav_0_r1_20261012.json`。
