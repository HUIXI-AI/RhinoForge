# API 参考

## 张量诊断

在构造模型、捕获 Graph 前调用 `torch.rpu.set_debug_export(True)`，可开启支持路径的中间张量导出。
用 `list_debug_tensors()` 列出名称，`get_debug_tensor(name)` 读取，`get_debug_export()` 查询开关。
返回值共享运行时存储，跨重放保存数值时使用 `.cpu().clone()`。
`clear_debug_tensors()` 清空注册项，关闭导出也会清空。NextDiT 在整个实例生命周期内固定此开关；
Qwen3.5 使用独立诊断 prefill Graph，冻结后不能临时创建缺失的诊断图。
Pi0.5 还可通过 `RPU_PI05_PROBE_DIR` 指定本地输出目录，将命名输入和中间张量保存为 `.pt` 文件。

简体中文 | [English](api_reference.md)

发行包名为 `rhinoforge`，Python package 仍是 `rpu_backend`。导入后把 PyTorch
`PrivateUse1` 设备注册为 `rpu`，并创建 `torch.rpu` namespace。

支持范围由 checkpoint、配置和执行入口共同决定，见[示例](model_support.zh.md)。

## 顶层 package

`rpu_backend.__all__` 只包含四个名称：

| 名称 | 契约 |
|---|---|
| `__version__` | 已安装 RhinoForge distribution 版本 |
| `RPUCache` | fused causal attention 的 FP16 KV cache |
| `RPUModelForCausalLM` | 已注册 decoder-only CausalLM 配置 loader |
| `reset_graph_cache()` | 清理进程 Graph cache；无 native backend 时为 no-op |

API compatibility level 位于 `rpu_backend.api.API_VERSION`，当前为 `5.0.0`，与
distribution version 独立。

顶层包还按需提供 `set_debug_level(0..5)`、`get_debug_level()` 和 `TRACE`，
`torch.rpu` 提供同名入口。这些名称不在 `__all__` 中；其他诊断使用 `torch.rpu`。
级别依次为 SILENT、ERROR、WARN、INFO、DEBUG、TRACE，默认 INFO。

## Causal language model

```python
import torch
from rpu_backend import RPUCache, RPUModelForCausalLM

model = RPUModelForCausalLM.from_pretrained(
    checkpoint,
    dtype=torch.float16,
    device="rpu",
    rpu_execution={
        "prefill": {
            "chunk_size": "auto",
            "padding_rows": "auto",
            "padding_budget": 64,
        }
    },
)
cache = RPUCache.from_model(model, input_ids=input_ids, max_new_tokens=32)
```

### `RPUModelForCausalLM.from_pretrained`

```python
RPUModelForCausalLM.from_pretrained(
    hf_repo_or_path,
    *,
    dtype=torch.float16,
    device=None,
    rpu_execution=None,
    **hf_kwargs,
)
```

Loader 在完整权重加载前读取 Hugging Face 配置，按 `config.architectures[0]` 选择
adapter，并提前拒绝 unsupported profile。当前入口接收 built-in Qwen3、Llama CausalLM
architecture，只接受 `torch.float16`；量化 checkpoint 自带受检查 metadata。

`rpu_execution` 是 read-only per-handle mapping。CausalLM 的 `prefill` 接受：

- `chunk_size`：`"auto"` 或正的 16 倍数；
- `padding_rows`：`"auto"` 或非负准确行数；
- `padding_budget`：非负整数，与精确 `padding_rows` 互斥。

Adapter 按所选 checkpoint 配方进一步限定字段与核数。`prefill.linear_acc32` 使用
布尔值选择累加方式；`model.num_cores` 接受整数 `4/6/8`。核数、精度等冷态字段需要
新模型，支持热修改的规划字段通过下述执行 API 更新。

`trust_remote_code` 默认为 `False`，会传给 Hugging Face loader。
`Pi05Policy.from_pretrained` 在启用时发出警告。Adapter 的模型与 checkpoint 检查仍然适用。

### `RPUCache`

优先使用 factory：

```python
RPUCache.from_model(
    model,
    *,
    input_ids=None,
    max_new_tokens=...,
    max_seq_len=None,
    batch_size=1,
    device="rpu",
)
```

必须且只能指定一种 sizing：`input_ids + max_new_tokens`，或正的 `max_seq_len`。
Cache 提供 `reset()`、`reset_to_position(pos)`、`get_seq_length()`、
`get_max_length()`、`get_cache(layer_idx)` 和 `to_dynamic_cache(device="cpu")`。
Batch>1 只限明确允许的 Qwen3 路径，并要求各行真实 token 长度相等。

`rpu_backend.api.Qwen3_5Cache.from_config(text_config, max_seq_len, ...)` 增加
Qwen3.5 text 的 recurrent/convolution state。它可完整 reset 到 position 0，但无法通过
rewind 重建非零 recurrent state。
已加载模型，尤其是缩核配置，优先使用
`Qwen3_5Cache.from_model(model, max_seq_len=...)`，让 KV 和 convolution storage
匹配实际拓扑。直接从 config 创建 cache 时须传入与模型相同的核数。

### Greedy token 选择

```python
from rpu_backend.api import greedy_token_ids

out = model(**inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
next_ids = greedy_token_ids(out.logits)
```

`greedy_token_ids` 接受 `[batch, vocab]` 或 `[batch, sequence, vocab]`，后者取最后一个
sequence 行；返回独立的 CPU `int64 [batch, 1]`，不修改 logits。CPU
FP16/BF16/FP32/FP64 保持原精度，设备 logits 必须为 FP16。相等最大值取首个索引，
含 NaN 的行取首个 NaN，与 `torch.argmax` 一致。允许空 batch；空 sequence/vocab、
其他维数或 dtype 会被拒绝。Cache position、EOS 与采样策略由调用方管理；这个函数
不改变 Hugging Face `generate()`。

Qwen3 的 RoPE 容量来自已安装的位置表或模型声明的 `max_position_embeddings`。
请求须同时落在 cache 和位置表范围内；仅增大 cache 不会延长模型上下文。

## 执行配置与运行中重配

`rpu_execution` 接受 Python mapping，也可从 TOML/JSON 读取。component override 优先于顶层 stage，顶层 stage 优先于 profile 默认值；模型保存不可变副本。整数 `chunk_size` 要求精确尺寸，`"auto"` 在允许范围内选择；整数 `padding_rows` 固定补齐量，不能同时指定 `padding_budget`。

已绑定执行会话的模型通过统一 API 查询状态、修改支持热更新的配置：

```python
from rpu_backend.api import reconfigure_rpu_execution, rpu_execution_stats

before = rpu_execution_stats(model)
reconfigure_rpu_execution(model, {
    "prefill": {"chunk_size": "auto", "padding_budget": 64},
})
after = rpu_execution_stats(model)
```

传入 stage 的热参数整体替换，`components` 下同样处理；未出现的 stage 保持不变。`{"prefill": {}}` 清除规划 override，保留已绑定的 `linear_acc32` 等冷参数。配置不变时不重建，也不增加 `generation`；修改提交后才增加 generation、退休受影响的 Graph，并在下次执行时重建。调用须在请求之间进行；无会话、正在 forward、固定或不支持的字段、已关闭或失败的会话都会拒绝修改。核数、精度等冷态参数需新模型；退休或回滚失败后需新进程。见[配置生命周期](../knowledge/concepts/runtime-profiles.md#safe-profile-changes)。

## Image-text model

`rpu_backend.api.RPUModelForConditionalGeneration` 是 Qwen3-VL、Qwen3.5 Dense
及精确 Qwen3.5 MoE 配方的 Hugging Face 风格 loader。Qwen3-VL 示例：

```python
from rpu_backend.api import RPUModelForConditionalGeneration

model = RPUModelForConditionalGeneration.from_pretrained(
    checkpoint,
    dtype=torch.float16,
    device="rpu",
    rpu_execution={
        "prefill": {"chunk_size": "auto"},
        "vision": {"chunk_size": "auto"},
    },
)
```

它与 CausalLM loader 一样 CPU-first、profile fail-fast。`vision` 接受 `chunk_size`，
`prefill` 接受前述字段。Class 存在不自动扩展视频或
[示例](model_support.zh.md)之外的配置。Qwen3.5 Dense 的纯文本与图文请求都可使用
这个 loader，由 checkpoint 元数据选择 adapter；Gemma4 使用独立 adapter 入口。
核数与 cache 的对应关系见[计算拓扑](planner_core_topology.zh.md)。

### Qwen3.5 27B

官方 27B 提供 4/6/8 核 W8 Text/VL 和 8 核 W4 group32。root6 的 MLP 使用 6 核，attention/GDN/head/Vision 使用 4 核。启用既有量化 opt-in；图像输入另需 Vision opt-in。安装使用缓存分配器、四个冷 Graph arena、CPU 权重最终布局上传及物理 KV24；prefill 为有界单次 Graph，decode 保留独立 Graph。暂不提供 FP16 运行入口。详见[配置与权重](../examples/configs/qwen3_5/README.md)。

### Qwen3.5 MoE

35B-A3B 的 RTN/GPTQ 路径使用 conditional loader、MoE adapter 和混合 cache。[配置与权重](../examples/configs/qwen3_5/README.md)须匹配，RTN 与 GPTQ 元数据不能互换。

```python
import torch
from rpu_backend.api import RPUModelForConditionalGeneration, Qwen3_5MoeCache
from rpu_backend.adapters.qwen3_5_moe import Qwen3_5MoeAdapter

model = RPUModelForConditionalGeneration.from_pretrained(
    checkpoint, dtype=torch.float16, local_files_only=True,
    rpu_execution={"model": {"num_cores": 8}, "prefill": {"chunk_size": 64}},
)
model = Qwen3_5MoeAdapter(model).to_rpu(max_seq_len=1024)
cache = Qwen3_5MoeCache.from_model(model, max_seq_len=1024, dtype=torch.float16)
with torch.no_grad():
    output = model(input_ids=input_ids, past_key_values=cache,
                   use_cache=True, return_dict=True, logits_to_keep=0)
cache.reset()
```

`input_ids` 为 CPU `int64`、形状 `[1, T]`，这组配置使用八核。模型安装一次，decode 复用 cache；新请求前调用 `reset()`。prefill 使用有界单次 Graph，decode 保留独立 Graph；按请求长度设置 chunk size 和 cache 容量。

图像请求在 reset 后的首次调用同时传入 processor 的 `input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw`；随后复用 cache 的调用只传文本 token。返回 logits 为 CPU FP32。

独立 mixed-E4M3 配方使用
[`convert_qwen3_5_moe_fp8.py`](../python/rpu_backend/quant/convert_qwen3_5_moe_fp8.py)
从官方未量化 `Qwen/Qwen3.5-35B-A3B` 在 CPU 离线生成，无需校准数据。
按[下载、转换与运行命令](../examples/configs/qwen3_5/README.md#cmd-moe-mixed)
写入 `$RPU_MODEL_CACHE/Qwen3.5-35B-A3B-rpu-fp8-mixed`，将输出目录传给
`from_pretrained` 或示例的 `--model`。官方 block-FP8 包不能作为转换输入，
也不能替代本配方的逐输出通道 E4M3 格式。选择匹配的 mixed-E4M3 Text/VL
及核数模板；上述 RTN/GPTQ 限制保持独立。

## Policy API

`rpu_backend.api` 导出：

| Class | 主要构造与推理方法 |
|---|---|
| `Pi05Policy` | `from_pretrained(...)` / `from_lerobot_policy(...)`；`to("rpu")`；`prepare_graphs(...)`；`predict_action_chunk(...)`；`select_action(...)` |
| `RhinoVLAPolicy` | `from_runtime(...)` / `from_factory(...)` / `from_pretrained(..., runtime_factory=...)`；`prepare_graphs(...)`；`predict(...)`；`predict_action_chunk(...)` |
| `WallOssPolicy` | `from_checkpoint(...)` / `from_pretrained(...)`；`to("rpu")`；`prepare_graphs(...)`；`infer(...)`；`predict_action_chunk(...)` |
| `Lingbot2Policy` | `from_checkpoint(...)`；`to("rpu")`；`prepare_graphs(...)`；`infer(...)`；`predict_action_chunk(...)`；`close()` |
| `HyEmbodiedPolicy` | `from_checkpoint(...)`；`to("rpu")`；`infer(...)`；`predict_action_chunk(...)`；`close()` |

Pi0.5 [模板](../examples/configs/pi05/README.md)指定精度、相机数量和语言长度。
Runner 在加载权重前绑定冷配置，并向 `Pi05Policy.from_pretrained(...)` 传入
`rpu_execution`。Checkpoint 元数据须匹配所选精度；通过
`prepare_graphs(..., precompute_adarms=...)` 显式选择 AdaRMS 预计算。

`Pi05Policy.prepare_graphs(batch, noise=...)` 和
`Pi05Policy.predict_action_chunk(batch, noise=...)` 接受可选的初始噪声张量。
张量必须位于 CPU、类型为 `float32`、所有值有限，形状为已加载 policy 配置定义的
`[batch_size, chunk_size, max_action_dim]`。比较固定输入时，Graph 准备和推理应使用
同一张量；不传此参数时保留原有的种子噪声生成行为。Pi0.5 示例 TOML 通过
`input.noise` 指定 `torch.save` 保存的张量文件，优化示例要求形状为 `[1, 50, 32]`。

Pi0.5 `prepare_graphs()` 返回 Graph 就绪状态与重放诊断。shape、dtype、
finite 和 Graph 不变量须通过检查。
`replay_parity.numerical_status` 为 `DIAGNOSTIC`，`task_quality` 为
`NOT_EVALUATED`。

对于发布绑定的 Pi0.5 组件 profile，传入 `profile="pi0.5-base"` 或
`profile="pi0.5-libero-v044"`，并同时传入首个预处理后的 `admission_batch`。
Loader 会在加载权重前校验精确公开 payload 和请求范围，后续 Graph 与推理请求继续按
同一 profile 校验。

可在应用中保留已加载的 `HyEmbodiedPolicy` 或 `WallOssPolicy`，为连续请求重复调用 `infer(...)`。同一实例串行使用，每次更新请求输入；结束时调用 `HyEmbodiedPolicy.close()`。

`WallOssActionOutput`、`Lingbot2ActionOutput`、`HyEmbodiedActionOutput` 是对应
结构化结果。带物理单位的 action 字段应用配置的 normalization。

`WallOssPolicy.from_checkpoint(...)` 绑定 checkpoint、相机顺序、归一化与精度，
`.to("rpu")` 安装 runtime。调用 `infer(...)` 时，图像顺序必须与 `camera_names`
一致；policy 按已装载模型检查实际请求的图像几何和状态。可用 `prepare_graphs(...)`
为代表输入绑定明确的有限前缀范围，仅在请求满足该 policy 的准入条件时复用。

`RhinoVLAPolicy` 把 checkpoint 组合与 preprocessing 委托给显式 model-repository
runtime factory。Runtime 必须声明 execution capability 和实际消费的 `rpu_execution`；
facade 拒绝静默 mismatch。

字符串 `runtime_factory="module:callable"` 会 import 并执行该 module 中的 Python code。
只使用已安装、已审查的模型集成，不从不可信 TOML/request 复制该值；已导入 callable
同样需要信任。

`HyEmbodiedPolicy.from_checkpoint` 默认不读取 `norm_stats.pkl`。解码物理 action 需要
`norm_stats_path`、显式 `trust_norm_stats_pickle=True` 和准确文件的
`norm_stats_sha256`；同一字节先验证 hash，再 deserialize。
固定三相机 facade 的 `prefix_len` 仅接受 `{192, 208, 224, 240}`。请求中的图像、
noise、state embedding/state 和末端执行器 pose 必须为有限值；非法值会在 policy
执行前被拒绝。

按各 profile 的说明选择 constructor、input schema 和 precision。

### LingBot v2 runtime Graph

当前 FP16/W8A16 入口使用 `rpu_backend.adapters.lingbot_vla_v2.runtime.build_lingbot_vla_v2(..., graph_runtime_policy=...)` 和 `get_action`。[输入指南](../examples/configs/lingbot/v2/README.md)说明两/三相机、训练 YAML、语言存储和归一化 `[1,50,55]` 输出。旧 `Lingbot2Policy.from_checkpoint` facade 接受 `rpu_execution`，保留独立 exact-Z2 配置。

## Error hierarchy

用户可见错误在 `rpu_backend.api`：

```text
RPUBackendError
├── RPUConfigError
├── PlannerRejectError
├── UnsupportedModelError
├── RPUUnsupportedDtypeError
├── RPUSingleHandleError
├── SPMExhaustionError
└── WeightShapeMismatchError
```

通用捕获 `RPUBackendError`；调用方有特定恢复动作时捕获 subclass。
`PlannerRejectError` 通过 `.code`（`EXACT_MISMATCH`、`CAPABILITY`、`NO_FEASIBLE`、
`SEARCH_LIMIT`）和 `.stage` 描述被拒绝的执行计划，也可用 `RuntimeError` 捕获。
非法 Python 参数仍使用标准 `TypeError`、`ValueError`。

## Graph API

`GraphRuntimePolicy` 接受冷配置 `batch_snapshot_mode="auto"` 或 `"off"`。`GraphCache` 及 Pi0.5、Wall-OSS、LingBot v2 构建器在安装权重前接收策略。AUTO 按 SDK/资产能力启用快照或回退普通重放；OFF 保留 Graph 执行。见[快照策略](runtime_config.zh.md#graph-快照策略)。

公开 namespace 为 `rpu_backend.graph`：

```python
from rpu_backend.graph import GraphCache, GraphSignature

cache = GraphCache(max_entries=4)
sig = GraphSignature(
    op_id="my_model_forward",
    shapes=(seq_len, hidden_size),
    dyn_dims=(num_layers,),
    dtypes=(torch.float16,),
)

with cache.capture(sig):
    output = torch.ops.rpu.my_model_forward(...)
```

`GraphCache` 提供 `capture`、`begin_warmup`、`freeze`、`is_frozen`、`lookup`、
`evict`、`clear`、`size`、`max_entries`、`snapshot`、`cache_invariant_ok`。Frozen cache
只允许 lookup，拒绝 online BUILD。应用通常让 adapter 管理 Graph cache；手工 Graph
主要用于 porting。

公开诊断包括 `debug_bucket_counts()`、`debug_branch_counts()`、
`dump_signature_tree()`、`explain_miss(signature)`；`Graph.debug_stats()` 返回最近提交的
聚合 counter。它们解释生命周期，不是数值或支持证据。`Graph.dump_replay_plan()` 返回
pointer-free node/segment 摘要，`Graph.dump_tree()` 按 segment 分组。
`Graph.dump_resources()` 列出资源所有者、字节数、生命周期与 pool slot。
`Graph.dump_kernel_args_jsonl()` 返回 Release runtime 的算子容量元数据，原始状态不可用。
`Graph.debug_nodes()`、`Graph.debug_segments()` 返回字典列表，
`Graph.debug_segments_jsonl()` 返回 segment JSONL。记录包含节点类型与依赖、
逻辑张量信息、传输长度和分段结构；这些视图均省略原始地址及 launch 参数。

`Graph.runtime_policy` 和 `GraphCache.runtime_policy` 是构造时配置的只读视图，
抽取的子图继承父图配置。capture 失败后，scope 会释放借用的 graph 引用，
因此保留异常 traceback 不会额外阻止 `evict()` 或 `clear()`；调用方显式
持有的 graph 引用仍会阻止回收。

`Graph` 是低层 capture object；`get_default_graph_cache()` 返回 thread-local cache。
新 adapter 通常使用显式 per-model `GraphCache`，便于 ownership 与 teardown。

## Adapter registry 与 plugin

Registry 位于 `rpu_backend.runtime.registry`：

```python
from rpu_backend.runtime.registry import get_adapter, list_adapters, register_adapter
```

Built-in binding 在 `rpu_backend.adapters._manifest.BUILTIN_ADAPTERS`，当前包含
`Qwen3ForCausalLM`、`LlamaForCausalLM`、`DINOv3ViTModel`、
`Gemma4ForConditionalGeneration`、`Qwen3_5ForConditionalGeneration`、
`Qwen3_5MoeForConditionalGeneration`、`Qwen3VLForConditionalGeneration`。

外部分发包可声明 `rpu_backend.plugins` entry point：

```toml
[project.entry-points."rpu_backend.plugins"]
my_adapter = "my_package.rpu_plugin:register"
```

```python
def register(api):
    api.register_adapter("MyModelForCausalLM", MyModelAdapter)
```

Plugin 在 unknown-architecture lookup 后延迟加载，或由 `discover_plugins()` 显式发现；
不能覆盖 built-in/已注册 binding。Entry point 会执行安装包代码，只启用受信任 plugin。
Registry 是进程全局且无公开 unregister；改变 plugin set 后启动新进程。

## 模型路径 registry

```python
from rpu_backend.model_registry import MODELS, cache_root, model_path
path = model_path("qwen3-0.6b")
```

`cache_root()` 优先使用 `RPU_MODEL_CACHE`，否则为 `~/.cache/rhinoforge/models`。
`model_path(name)` 拼接受检 relative path，unknown alias 抛 `KeyError`。Alias 只是路径
映射，不是支持或资产可用声明。

## Adapter 作者的 weight-layout API

标准 helper 位于 `rpu_backend.runtime.weights`：

- `get_linear_partition(in_features, out_features, dwidth=2, num_cores=8)`；
- `swizzle_linear(weight, partition, num_cores=8)`；
- `swizzle_model_inplace(model, ..., skip_names=...)`。

Swizzle 原地改变 parameter storage，必须只运行一次。Fused subsystem 自有专用 Linear
应进入 `skip_names`，由该 subsystem 的一个权威转换路径处理。

## `torch.rpu` 设备控制

常用函数：

- `is_available()`、`device_count()`、`current_device()`；
- `set_caching_allocator(bool)`、`empty_cache()`、`memory_stats()`；
- `set_ddr_flush(bool)`、`set_ddr_flush_force(bool)`；
- `set_debug_level(0..5)`、`get_debug_level()`；
- `set_cross_layer_batch_prefill(bool)`、`set_cross_layer_batch_size(int)`；
- `shutdown()`。

开发诊断：

- `set_debug` / `get_debug` / `set_profile` / `get_profile` /
  `reset_profile_accumulators`；
- `set_spm_debug` / `get_spm_debug` / `spm_alloc_dump`。

这些诊断只暴露日志和聚合计数，不暴露模型输入、中间 activation、raw address
或 Graph register/resource payload。

`set_caching_allocator(bool)` 是进程级冷态 tensor storage 选择。第一次显式调用，
或尚未选择时第一次分配非空 设备 tensor，都会把策略冻结到进程结束。同值重复调用
是幂等的；请求相反值会报错，必须使用新进程。应在任何 设备 tensor 物化前调用。
`empty_cache()` 只释放未使用的 cached block，不能释放 live tensor 或 Graph-owned
storage。`memory_stats()` 通过 `caching_allocator_enabled` 返回所选模式，并通过
`caching_allocator_policy_frozen` 返回策略是否已经冻结。

CPU/设备 boundary flush 默认打开，正常推理不要关闭。Chunk size 属于 per-handle
`rpu_execution`，不是进程全局 `torch.rpu` setter。内部 `torch.ops.rpu.*` 是 adapter
使用的 host wrapper interface；未在本页或 model example 中记录的函数视为实现细节。

<a id="bounded-speech-apis"></a>
## 语音 API

```python
from rpu_backend.api import Qwen3ASRModel, Qwen3TTSModel, WhisperASRModel
```

| API | 输入与返回值 |
|---|---|
| `Qwen3ASRModel.from_pretrained(checkpoint, dtype=torch.float16)` | 本地固定 Qwen3-ASR-0.6B checkpoint |
| `model.transcribe(audio, sample_rate=16000, max_new_tokens=64)` | 一维 CPU 有限浮点波形，单声道 16 kHz、0.5–8 秒；生成上限 1–64。返回 `ASRTranscription(text, raw_text, token_ids)` |
| `WhisperASRModel.from_pretrained(checkpoint, dtype=torch.float16)` | 本地固定 `sherpa-onnx-whisper-base` checkpoint，含 encoder、decoder 和 token 表 |
| `model.transcribe(audio, sample_rate=16000, max_new_tokens=448)` | 一维 CPU 有限非静音浮点波形，最长 30 分钟；自动重采样、切分长音频和识别语言。返回 `WhisperTranscription(text, language, token_ids, finish_reason)`；每段最多 444 个内容 token |
| `model.create_stream(sample_rate=16000, max_new_tokens=448)` | 创建最长 30 分钟的流式会话。`accept_audio(chunk)` 每形成完整 1 秒音频返回一项不可变更新，`finish()` 确认尾部；更新包含已确认/临时文字、语言、音频时长和结束状态 |
| `Qwen3TTSModel.from_pretrained(checkpoint, source=prepared_package, dtype=torch.float16, codec_backend="rpu")` | 本地固定 Qwen3-TTS-12Hz-0.6B-Base checkpoint 和准备后的 `qwen_tts/` 源码目录 |
| `model.synthesize(text, reference_audio=local_file, language="Chinese", max_new_tokens=256)` | 文本 1–1024 字符；语言 `Chinese` / `English`；参考音频为有限、非全零的单声道 24 kHz、0.5–20 秒；每段生成上限 2–256。返回独立 float32 单声道波形及采样率 24000 |

Qwen3 ASR 使用 greedy，不提供 context、时间戳或流式接口。Whisper 使用 greedy，
不提供时间戳；流式入口在低于 -40 dBFS 达到 1 秒后确认，连续语音使用 28 秒窗口和
2 秒精确 token 重叠。TTS 使用非流式 speaker
embedding 克隆，按固定 checkpoint 配置采样：top-k 50、temperature 0.9、top-p 1.0。
在加载后、请求前调用 `torch.manual_seed`，可在同一执行配置下复现采样。每段 Talker
prompt 加生成上限须在 512 个位置内；达到上限仍无 EOS 时，在波形解码前报错。
空、非有限值或全零波形会被拒绝。

不超过 128 字符且 32 tokenizer token 的文本保持单段。长文本按标点分段，原始
片段最多 128 字符，目标 12 token；短句尾可与前一分句合并至 16 token。段尾补
句号可能改变 token 数、停顿和语调。每段从本次请求初始 CPU 随机状态开始，后续分段恢复第一段留下的
调用方随机状态。各段独立复制，全部成功后拼接返回；失败不返回部分音频。下载、
源码准备和 API 示例见[语音指南](qwen3_audio.zh.md)。

三类 API 均支持 `with`、幂等 `close()` 和 `execution_stats()`。请求、统计及关闭
串行调用，一个进程只保留一个存活设备模型/策略。加载器在转换前检查资产，并在
设备分配前选择 caching allocator；TTS 还声明 owner 专属六个 arena。
进程已冻结为不兼容 allocator 或 arena 计划时，启动新进程。

ASR 需要重编译的 native 提供 `siglip_set_audio_mode`、`siglip_set_audio_projection`
和 `assemble_inputs_embeds`。TTS codec 显式调用 `causal_decoder_set_tts_codec_mode`；
缺少 setter 时前段保留 CPU 路径。有该接口时，1–63 帧使用设备，64–255 帧使用
原始 CPU FP32 前段及其 72 帧滑窗注意力；其他执行错误直接报告。

`execution_stats()` 返回调用及缓存状态：

- ASR 为当前音频 token 长度保留一张音频图，为当前 prompt 长度保留最多一对
  prefill/decode 图。`audio_graph_size`、`audio_graph_replays`、
  `audio_graph_invariant_ok` 描述音频缓存；`graph_size`、`graph_replays`、
  `graph_invariant_ok` 描述文本缓存。替换缓存签名后重放计数重新累计；
  `audio_linear_calls` 统计逻辑投影调用。
- Whisper 返回 encoder、prompt prefill 和 decode 三张常驻图的大小、重放次数、
  invariant、一次性 `build_ms` 和当前请求的 `replay_ms`；每条音频只准备一次
  cross-K/V。
- TTS 返回 `generation.steps`、`generation.eos` 和 `generation.segment_steps`；
  `codec_frontend.device`、`fallback_reason` 描述最后一段。还返回输出头调用计数
  以及各组件的 calls、tokens 和 Graph 状态。`components.predictor.fused_head_calls`
  统计从主体与头融合图消费的 logits 次数；`head_calls` 仍统计逻辑输出头调用。
  Talker 在 prompt 长度变化时替换 prefill 图并保留
  decode；Predictor 保留双 token 条件 prefill 图和一张 decode 图；codec 在帧数变化时替换自己的图。

每个新序列重置逻辑 KV 位置并保留私有存储。TTS 按解码后的音频内容只缓存最新
speaker embedding，每次请求仍读取和校验参考文件。输出存储独立于重放；释放时
先清 Graph，再释放 native handle 和权重。Predictor 的最后一行设备 hidden 只借给
紧接着的输出头；请求 hidden states 时仍返回独立 CPU 存储。加载前通过
`torch.set_num_threads` 设置 CPU 线程数，API 不修改此值。

默认 `codec_backend="rpu"` 实验性卸载全部 31 层解码 Conv1d；匹配 native 时，
12 组卷积还将中间 SnakeBeta 和首层 bias 留在 RPU 上，以 FP16 执行。
显式设置 `codec_backend="cpu"` 恢复 CPU FP32 卷积和激活。
`execution_stats().codec_convolutions` 返回 backend、已安装层数、累计卷积调用次数、
`transport`（`shared_ddr` 或旧版 `staged`）、`residual_pairs` 和成功的 `paired_calls`。
每组成对执行也计入两次卷积调用。剩余 CPU 工作和精度边界见[语音指南](qwen3_audio.zh.md#实验性-rpu-声码器卷积)。
