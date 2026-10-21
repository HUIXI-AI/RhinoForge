# Whisper base RPU 推理检查清单

## 当前状态

- [x] 设计已确认并提交：`a144458`
- [x] 实施计划已拆分
- [x] 用户批准实施计划

## 阶段 A：先消除不可落地风险

- [x] T1 锁定模型 profile、完整资产哈希和参考夹具
- [x] T1 验证 manifest 可重复、参考输出有限且可复现
- [x] T2 清点 Whisper 所需精确算子、Launch 与动态 KV 能力
- [x] T2 在 R1 复跑第 0 层 prefill3/decode1 探针
- [x] 检查点 A：算子资产齐全；encoder 使用独立 owner，规避通用 Conv1d Graph 输入刷新问题

## 阶段 B：encoder 闭环

- [x] T3 新增公共 API 骨架、资产预检和音频边界测试
- [x] T4 接通 encoder native owner、6 层计算和 Graph 生命周期
- [x] T4 R1 encoder 余弦相似度 ≥ 0.9999（实测 1.000066，BUILD 1 次、REPLAY 完全一致）
- [x] T5 接通 encoder Python 安装、执行和清理
- [x] 检查点 B：R1 余弦 1.000066；重复执行逐元素一致、输出不别名、BUILD 1 次并稳定 REPLAY

## 阶段 C：decoder 闭环

- [x] T6 实现第 0 层 prompt prefill 和单 token decode
- [x] T6 验证 hidden/self-KV/cross-KV 余弦相似度 ≥ 0.9999（R1 实测最低 0.9999995）
- [x] T6 验证 cache 只追加、历史前缀不变、位置错误明确失败
- [x] 检查点 C：第 0 层数值、cache 与 Graph 复用全部通过
- [x] T7 扩展完整 6 层、cross-K/V、448 self-K/V 与 vocabulary head
- [x] T7 验证 warmup 0/1/3、跨音频 cache 清理和 Graph invariant（logits 最低余弦 0.9999933）

## 阶段 D：端到端能力

- [x] T8 实现合法语言 token 集合内的自动语言识别
- [x] T8 实现 prompt、贪心 decode、EOS/length 和 tokenizer 文本输出
- [x] T8 固定音频语言、逐 token、结束原因和文本与 ONNX 完全一致
- [x] 检查点 D：R1 两次输出均与 ONNX 完全一致，进入示例、性能和支持登记

## 阶段 E：交付

- [x] T9 增加固定配置、最小示例和 `report.json`
- [x] T9 完成 2 次 warmup + 10 次测量，只报告性能不设硬门槛
- [x] T10 覆盖短音频、30 秒、EOS、448、重复输入和错误边界
- [x] T10 运行 Whisper、Qwen3 ASR、NavDP 相关回归（179 项通过；可选 diffusers 用例未收集）
- [x] T10 仅在硬门槛通过后更新模型支持文档
- [x] 检查最终提交不含权重/音频，也未包含用户自有 `lib/`
