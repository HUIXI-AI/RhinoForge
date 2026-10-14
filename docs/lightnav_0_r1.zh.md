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

当前 SlowFast 配置下，12/24/36/48/64 帧历史加当前帧分别产生约
288/456/568/624/708 个池化后视觉 token。脚本同时记录实际完整输入 token 数，
运行时仍以 Qwen3-VL 4B 的 P4176 准入上限为准。

脚本对缓存和重算的 pooler/DeepStack 特征使用 `1e-3` FP16 绝对容差检查。

## 执行说明

processor 只处理当前未缓存的 tubelet；精确 segment pooling 复用已池化历史，
空缓存仍运行完整窗口的视觉路径。视觉 token 和 M-RoPE 位置变化会使旧 KV 前缀
失效，因此每次执行完整 prefill。五个 token 按自回归顺序生成，使用融合 decoder
和设备 argmax。

约束输出头沿用上游 action allowlist，候选范围缩小可能改变 token 选择，应与应用的
解码约定一致。当前不提供时序投机解码，checkpoint 也不包含多 token 预测头或权重。
