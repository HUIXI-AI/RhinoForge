# Pi0.5 配置

本目录仅提供当前八核优化配置：`2cam/` 与 `3cam/` 各四份，共 8 份。统一使用 `[pi05]`、`[input]`、`[run]` 和可选 `[rpu_execution]`。

| 配置文件 | 精度范围 |
|---|---|
| `fp16.toml` | 原始浮点 checkpoint；不启用 Vision 和 AdaRMS Dense 的 W8 转换。 |
| `w8a16.toml` | VLM 与 Action decoder 七投影离线 W8；SigLIP encoder Linear 与 AdaRMS Dense 在安装时转 W8。 |
| `w8a16_action_nvfp4.toml` | Action Q/O/Gate/Up/Down 为 NVFP4，Action K/V 与 VLM 为 W8；Vision/AdaRMS 按 W8 profile 安装。 |
| `w8a16_prefill_w8a8_action_nvfp4.toml` | 相同 Action NVFP4 权重，加指定 Prefill GateUp/GeGLU A8 activation 路径；KV cache 仍为 FP16。 |

量化名称描述指定部分，不表示全模型或所有中间值使用同一种精度。当前目录不发布四核、六核或 INT4/G32 兼容模板，不能通过改文件名、核数或 precision 标签得到等价配置。

## 权重与运行

准备 checkpoint 及匹配预处理的 CPU tensor batch，安装 [运行时资产](../../../docs/runtime_assets.md)，从仓库根目录执行：

```sh
python examples/run_model.py --config examples/configs/pi05/3cam/w8a16_prefill_w8a8_action_nvfp4.toml --check-config
python examples/run_model.py --config examples/configs/pi05/3cam/w8a16_prefill_w8a8_action_nvfp4.toml
```

`--check-config` 不加载权重或访问板卡。运行前设置 `input.checkpoint` 和 `input.batch`，前者是注册的本地 alias 或完整目录，后者是 tensor 字典文件；相对路径以启动目录为基准。模板不包含权重、真实观测或私有 fixture。

从同一份未量化 Pi0.5 源 checkpoint 分别生成 W8 与 NVFP4 目录，目标必须尚不存在：

```sh
python -m rpu_backend.quant.convert_pi05 \
  --src /path/to/pi05-source --dst /path/to/pi05-w8a16
python -m rpu_backend.quant.convert_pi05 \
  --src /path/to/pi05-source --dst /path/to/pi05-action-nvfp4 \
  --action-w4 --action-w4-format nvfp4
```

NVFP4 是 E2M1 浮点编码，使用 block16 的 FP8 scale 和 FP32 tensor scale，元数据为 `method=nvfp4a16`、`nvfp4_abi=striped_v2`。INT4/G32 是整数分组格式，不能代替 NVFP4 权重或匹配资产；更改配置不会转换格式。Prefill A8 使用相同 NVFP4 checkpoint，由 profile 选择激活路径。详见 [量化说明](../../../docs/quantization.md)。

## 配置与输入合同

| 字段 | 含义 |
|---|---|
| `pi05.precision` | 选择上述四种精度，绑定对应的公开 `optimized_profile`。 |
| `input.checkpoint` | alias 或本地模型目录；必须匹配量化元数据。 |
| `input.batch` | CPU tensor 字典，包含 policy 要求的图像、相机 mask、状态、token 和文本 mask。 |
| `input.noise` | 可选的 CPU float32 tensor 文件，形状 `[1,50,32]`，作为明确输入传给准备与预测；不设置时保留默认噪声采样。 |
| `input.cameras` | 2 或 3 个相机槽位；与目录和真实 batch 一致。 |
| `input.text_tokens` | 模板使用 32，固定文本容量；不是任意 prompt 字符长度。 |
| `input.num_steps` | 模板固定 10 个 denoise step。 |
| `run.warmup` / `runs` / `seed` | 预热、正式样本次数和种子；加载与 Graph 准备位于正式样本之外。 |
| `run.output_dir` | 报告目录。`run.profile` 正常保持关闭；公开示例不采集硬件 trace。 |

`w8a16_action_nvfp4` 对应 API 的 `w8_action_nvfp4`，`w8a16_prefill_w8a8_action_nvfp4` 对应 `w8_prefill_a8_action_nvfp4`。runner 应用优化 profile 和准备 Graph，不需要逐项复制融合环境开关。

真实 batch 必须使用 checkpoint 对应的预处理、状态归一化、相机槽位和 mask 约定，不能靠增加空图或复制 token 满足容量。固定噪声对照可使用 `input.noise`；runner 仍拒绝通过诊断环境变量替换输入。相同 shape 不表示输入内容或语义可以忽略；任务质量需使用应用真实观测判断。

## 冷态配置与复用

`rpu_execution.model.num_cores` 保持 8。`vision/prefill/action.linear_acc32` 分别控制组件累加精度，默认 `false` 为 ACC16；更改后重新加载 policy。chunk 保留 `"auto"`，runner 按两/三相机与 T32 绑定 paired prefill；优化 profile 要求零额外 prefill padding，不能扩大 `padding_budget` 或随意修改 chunk 来绕过几何检查。

服务中复用同一个 `Pi05Policy`，安装后准备 Graph，再重复预测，结束时 `close()`。新请求保持已准备的相机、文本容量、mask 和 action 范围；变更精度、相机槽位或规划设置时创建新 policy。一个进程只保留一个 RPU policy owner，不要每次请求重载权重。

CPU 线程与 allocator 是进程部署设置，不是 RPU 核数。正常计时关闭 profiler 和详细 trace；配置本身不是机器人任务质量认证。参见 [运行时配置](../../../docs/runtime_config.md)、[API 参考](../../../docs/api_reference.md) 和 [Pi profile 实现](../../../python/rpu_backend/adapters/pi05/optimized.py)。

标准配置的两个示例入口均在导入 Torch 前绑定 `MIMALLOC_PURGE_DELAY=1000` 和
`MIMALLOC_ARENA_PURGE_MULT=0`，避免遗漏优化 profile 的 CPU 分配器设置。
这不改变模型精度或数值校验；应用程序调用库 API 时需自行在新进程启动前配置。

## 数值验证

首轮 fused 与 Python 路径对照保留相同权重、输入和噪声，并检查输出 shape、
dtype、finite、prefix KV 与 Graph 生命周期。MSE、最大绝对误差、relative-L2、
逐行分布及单侧零行计数仅作诊断，不再采用固定的跨模型浮点误差门限。
运行通过不等于任务质量认证；量化实现应对照相同量化权重，原始浮点参考的
动作差异与真实任务质量另行评估。READY 表示 Graph 生命周期准备完成。
