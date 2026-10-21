# Whisper base RhinoForge 运行能力门禁

## 结论

`csukuangfj/sherpa-onnx-whisper-base` 已完成 RhinoForge 端到端实现与 R1 验收。
固定 16 kHz 和 8 kHz 测试音频的语言、逐 token、结束原因和文本均与 ONNX 一致；
短音频、30 秒单段上限、不同输入刷新和重复执行均通过。CPU 分段层进一步支持
最长 30 分钟离线输入和每秒更新的流式会话。

## 锁定范围

- 上游 revision：`bb53ee204431c90d314c1cc08d28d23e5b7927cc`
- 资产：`base-encoder.onnx`、`base-decoder.onnx`、`base-tokens.txt`
- 固定 RPU profile：batch 1、FP16、16 kHz 单声道、单段最长 30 秒、448 text context
- 公共 API：单次最长 30 分钟；连续静音 1 秒确认句子，连续语音 28 秒技术分段
- 资产大小、SHA256、模型结构和参考转录见
  `tests/fixtures/whisper_base/manifest.json`

## R1 实测

测试环境：R1，RhinoForge runtime v1.2.0。第 0 层固定 prefill3/decode1
产物在 2026-10-21 重新运行，结果如下：

- RhinoForge 源码：commit `d85a63a`
- Rhino Launch：v1.2.0，commit `3dd1b41`，共享库 SHA256
  `cd11d3a33e4c37da9434026d8f3298f9cde02bf404ea331f2e4e494d38e9333b`
- 主算子资产：commit `63ff35f902e3f34d7bac00afc04970b3115cd50b`，SHA256
  `306bd54a56bcf4106ca029adceabebb7446592c333805daf67528b77bcb46417`
- 扩展算子资产：commit `f8d6b94b67d1be9416e35e79b5f8f94b03e7fab2`，SHA256
  `192cf67ccd8ebf88f38c1bbc295a961bad21068afce77de3fcd236b3361e84e8`
- 固定配置 SHA256：
  `9732facd9b67f93867ae8e4aa973f31190741e38f80d1fdcf0c2859802e2873d`

| 路径 | hidden cosine | K cosine | V cosine |
| --- | ---: | ---: | ---: |
| prefill3 | 0.999999645 | 0.999999837 | 0.999999815 |
| decode1 | 0.999999691 | 0.999999939 | 0.999999937 |

单算子探针最低结果：LayerNorm 0.999999523、ERF GELU 0.999999881、
Linear 0.999998391、cross attention 0.999999821、self attention 1.0、
Conv1d 0.999996781，均通过 0.9999 门槛。

完整 encoder 最终输出余弦相似度为 1.000066；完整 decoder 的 logits 最低为
0.9999933，self-K/V 最低为 0.9999976，cross-K/V 最低为 1.0006158。三个常驻
Graph（encoder、prompt、decode）均保持单一 entry，后续请求稳定 REPLAY 且 invariant
为真。

固定协议执行 2 次 warmup 和 10 次测量：总耗时中位数 108.507 ms、RTF 中位数
0.016378、token/s 中位数 221.185。组件 P50/P95（ms）分别为：预处理
2.503/2.614、encoder 9.589/9.648、cross-K/V 21.560/22.336、语言识别
2.482/2.577、prompt 6.048/6.206、decode 总计 48.150/48.277、单步 decode
2.000/2.042。Graph BUILD（ms）分别为 encoder 12.948、prompt 3.696、decode
4.294；REPLAY P50/P95（ms）分别为 8.927/8.976、1.816/1.872、
1.655/1.699。这些数字仅作记录，不是业务性能门槛。

### 30 分钟与流式扩展

2026-10-21 在 `agi@10.10.66.42` 使用实现提交 `3d80847` 和上述 runtime 资产完成
扩展验证。将已通过逐 token 验证的 6.625 秒英文音频重复 5 次，并在每次后加入
1 秒静音，得到 38.125 秒输入：

- 离线长音频输出 120 个 token，等于基准 24 个 token 精确重复 5 次；总耗时
  1.075 秒，RTF 为 0.0282；
- 以 0.25 秒音频块输入流式会话，共返回 38 次整秒更新；最终文字和语言与离线
  长音频一致；产生更新的 `accept_audio()` 调用耗时中位数 151.153 ms，最大
  224.539 ms；
- 30 分钟纯静音会话返回 1800 次空更新且未增加 RPU replay，超过 1 个采样点会在
  推理前失败；这项只验证时长、状态和越界，不代表 30 分钟连续语音性能；
- 完成全部长音频和流式调用后，encoder、prompt、decode Graph 仍各有 1 个 entry，
  replay 分别为 85、85、4013，全部 invariant 为真。

连续语音技术分段另用 39.75 秒重复语音验证：离线与流式结果完全一致，说明
28 秒窗口和 2 秒重叠路径稳定；但该合成输入出现明显重复转写。因此无 1 秒停顿的
超长连续讲话只保证流程可运行，不作为文本质量验收通过项。需要提升这一场景时，
应改用 Whisper 时间戳窗口或更强的重叠对齐，不能把精确 token 去重当作质量保证。

## 实现约束

通用 `F.conv1d` Graph 虽然没有 CPU callback，且 BUILD/REPLAY 计数稳定，但
A/B 输入刷新测试失败。因此 Whisper encoder 不复用该 Graph 封装，必须由独立
native owner 管理持久 buffer、输入 DMA、执行和释放，并增加 A/B/A 刷新回归测试。

2026-10-21 又在 R1 检查了复用 DINO encoder owner 的可能性。Whisper 的
`6×H512×8-head×1500-token` 几何可以生成完整 native descriptor，但普通和
16-bit attention 路径都会被同一条 SDPA 同步约束拒绝：`grid_dim_x=12` 既不
小于等于 8，也不是 8 的倍数。不能通过补齐音频 token 规避，因为无 mask 的
额外 K/V 会改变 softmax。专用 owner 保持完整 1500 行 K/V，并将 query 切成满足
同步约束的小块；`Q≤3,K=1500` 探针和完整 encoder 端到端数值验证均已通过。

固定同一个输入 tensor 再原位更新也不能修复通用 Conv1d Graph：其内部布局
转换产生的缓冲仍保留首轮内容。专用卷积前端必须直接管理布局后的稳定输入缓冲，
并在每次 REPLAY 前显式 DMA 新内容，不能只固定 Python tensor 地址。
