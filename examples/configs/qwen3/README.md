# Qwen3 文本配置

本目录统一使用 `text/<size>/<precision>.toml`，共 24 份配置。模板采用 `[example]`、`[input]`、`[run]` 和可选 `[rpu_execution]`；profile 在 [配置约束表](../profiles.json) 中绑定模型、精度和安装 recipe。图文模型使用 [Qwen3-VL](../qwen3_vl/README.md)，混合注意力模型使用 [Qwen3.5](../qwen3_5/README.md)。

| 目录 | 配置 |
|---|---|
| `text/0_6b`、`text/4b` | FP16、FP16 Instruct、W8A16、W4/G32、W4/G32 + LM head |
| `text/1_7b`、`text/8b` | FP16、W8A16、W4/G32、W4/G32 + LM head |
| `text/14b`、`text/32b` | W8A16、W4/G32、W4/G32 + LM head |

## 准备与运行

准备完整本地 checkpoint、tokenizer 及全部权重分片，安装匹配的 [运行时资产](../../../docs/runtime_assets.md)。alias 在 `RPU_MODEL_CACHE` 下解析，默认根目录为 `~/.cache/rhinoforge/models`，不会自动下载模型。

```sh
export RPU_MODEL_CACHE=/path/to/models
python examples/run_model.py --config examples/configs/qwen3/text/0_6b/fp16.toml --check-config
python examples/run_model.py --config examples/configs/qwen3/text/0_6b/fp16.toml
```

从仓库根目录运行。需要自己的 checkpoint 时复制选定配置并设置 `input.checkpoint`；相对路径以启动目录为基准。`--check-config` 不加载权重或访问板卡，只检查配置和 profile 约束；checkpoint 格式、真实 token 长度与可行分块在运行时检查。

实际执行在 tokenizer 和权重加载前核对 checkpoint 元数据的模型尺寸和精度范围。
FP16 配置拒绝量化 checkpoint，W8 配置要求七类 decoder 投影与独立 LM head
同时量化、embedding 保持浮点；兼容 checkpoint 的目录名称不受限制。

## 精度与权重范围

- **FP16** 使用对应尺寸的无量化浮点 checkpoint，安装为 FP16。Instruct 是独立 alias，不靠修改提示词冒充另一套权重。
- **W8A16** 使用离线七类 decoder 投影和独立 LM head 的对称逐输出通道 INT8 checkpoint，scale 为 FP16，embedding、activation 和 KV cache 保持 FP16。14B/32B 要求各自精确的量化结构。
- **W4/G32** 使用对应尺寸的原始 dense 浮点 checkpoint。`w4a16.toml` 在安装时转换七类 decoder 投影，head 保持 FP16；`w4a16_lm_head.toml` 另将独立 head 转为 W4/G32。recipe 来自 profile 的 `on_install_quantization`，不能仅修改 `example.dtype`，也不能以 AWQ、GPTQ 或离线 W8 产物代替源模型。

14B/32B 的显式 W4 安装 recipe 不开放普通 FP16 入口。量化、checkpoint 与核数检查仍然生效；alias 和模板不构成任务质量承诺。W8 离线转换示例：

```sh
python -m rpu_backend.quant.convert_qwen3 \
  --src /path/to/Qwen3-8B \
  --dst /path/to/Qwen3-8B-W8A16 \
  --quant-lm-head
```

输出必须是新目录，保留源 checkpoint 和 tokenizer；将 `input.checkpoint` 指向转换产物，再使用 [8B W8 配置](text/8b/w8a16.toml)。不要对 14B 跳过 decoder 投影或改用 embedding 量化来构造未经准入的组合。详见 [量化说明](../../../docs/quantization.md)。

## 输入、计时与生成

| 字段 | 含义 |
|---|---|
| `example.profile_id` / `registry_alias` / `dtype` | 绑定已定义的模型、checkpoint 路径与精度，不是任意可组合标签。 |
| `input.prompt` | 纯文本请求；单个请求的 batch 为 1。 |
| `input.prefill_tokens` | runner 构造的目标 token 长度，使用模板指定策略；不是字符数。 |
| `input.decode_steps` | prefill 首 token 之后的 decode 调用数，可产生总计 `D+1` 个 token。 |
| `input.stop_on_eos` | 是否允许 EOS 提前结束；固定工作量配置保持 `false`。 |
| `run.warmup` / `runs` / `seed` | 预热次数、正式样本数与随机种子；加载不计入正式推理样本。 |
| `run.warmup_decode_steps` | 模板显式设置时控制预热 decode 工作量，不改变正式样本。 |
| `run.profile` | 诊断采集开关，正常计时保持关闭；`hwperf` 不能用于公开示例采集硬件 trace。 |

不要把 `decode_steps` 当作总生成 token 上限，也不要把不同 P、D、EOS 或预热设置的报告当作相同测量。

## Prefill、精度和生命周期

保留 `rpu_execution.prefill.chunk_size="auto"`，由 planner 检查对齐、尾块、当前位置及 SPM 容量。以下是部分已有入口的多 token prefill 边界，不是任意精度组合的通用保证：

| 入口 | 累计物理 prefill 行数上限 | chunk 约束 |
|---|---:|---|
| 0.6B/1.7B FP16 | 4096 | 正 16 倍数，不超过 512 |
| 4B FP16 | 4096 | 正 16 倍数，不超过 256 |
| 8B FP16 | 4096 | 16 / 32 / 64 / 128 |
| 14B 精确 W8 | 192 | 16 / 32 |

物理行数含 planner padding，不能只看逻辑 prompt 长度。`prefill.padding_budget` 默认 64；显式 `padding_rows` 整数与 budget 互斥，且不能绕过容量限制。padding 不作为真实 token 返回。decode 还受 KV cache、RoPE 和算子条件约束；增加 cache 不扩大多 token prefill 上限。

`rpu_execution.model.num_cores` 和 `prefill.linear_acc32` 都是冷态选择。模板保留各自核数约束；默认 ACC16，`linear_acc32=true` 选择 ACC32。量化配置不得由浮点低核实现推导新核数支持。模板没有通用动态 batch 开关。

一个进程只保留一个活跃 RPU 模型 owner。服务中复用已安装模型和合法的 Graph 签名，正确重置每个请求的 KV 状态；不要逐 token 重载权重或清空 Graph。安装失败后重新加载干净模型，改变核数、精度或冷态环境项时使用新实例。CPU 线程和 Launch 缓冲属于进程部署设置，不能修复模型几何或 SPM 超限。参见 [运行时配置](../../../docs/runtime_config.md) 和 [API 参考](../../../docs/api_reference.md)。
