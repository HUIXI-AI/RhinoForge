# API 参考

简体中文 | [English](api_reference.md)

发行包名为 `rhinoforge`，Python package 仍是 `rpu_backend`。导入后把 PyTorch
`PrivateUse1` 设备注册为 `rpu`，并创建 `torch.rpu` namespace。

支持按配置判定。Class、adapter、registry alias 或 native schema 可导入并不代表模型
受到支持；选择入口前查[模型支持](model_support.zh.md)。

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

Adapter 可以按配置进一步缩窄。Execution setting 是冷配置：RPU 安装后改变它，需要
创建新模型。

通用 CausalLM、image-text loader 和 `Pi05Policy.from_pretrained` 只支持 built-in model
code；`trust_remote_code` 必须为 `False`，custom Hugging Face code 在配置/权重加载前拒绝。

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

## Image-text model

`rpu_backend.api.RPUModelForConditionalGeneration` 是 built-in Qwen3-VL 路径的
Hugging Face 风格 loader：

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
[模型支持](model_support.zh.md)之外的配置。Qwen3.5 text/vision 与 Gemma4 使用各自
adapter。

## Policy API

`rpu_backend.api` 导出：

| Class | 主要构造与推理方法 |
|---|---|
| `Pi05Policy` | `from_pretrained(...)` / `from_lerobot_policy(...)`；`to("rpu")`；`prepare_graphs(...)`；`predict_action_chunk(...)`；`select_action(...)` |
| `RhinoVLAPolicy` | `from_runtime(...)` / `from_factory(...)` / `from_pretrained(..., runtime_factory=...)`；`prepare_graphs(...)`；`predict(...)`；`predict_action_chunk(...)` |
| `WallOssPolicy` | `from_checkpoint(...)` / `from_pretrained(...)`；`to("rpu")`；`prepare_graphs(...)`；`infer(...)`；`predict_action_chunk(...)` |
| `Lingbot2Policy` | `from_checkpoint(...)`；`to("rpu")`；`prepare_graphs(...)`；`infer(...)`；`predict_action_chunk(...)`；`close()` |
| `HyEmbodiedPolicy` | `from_checkpoint(...)`；`to("rpu")`；`infer(...)`；`predict_action_chunk(...)`；`close()` |

`WallOssActionOutput`、`Lingbot2ActionOutput`、`HyEmbodiedActionOutput` 是对应
结构化结果。带物理单位的 action 字段只表示已应用配置 normalization，不认证机器人
安全或坐标系。

`RhinoVLAPolicy` 把 checkpoint 组合与 preprocessing 委托给显式 model-repository
runtime factory。Runtime 必须声明 execution capability 和实际消费的 `rpu_execution`；
facade 拒绝静默 mismatch。

字符串 `runtime_factory="module:callable"` 会 import 并执行该 module 中的 Python code。
只使用已安装、已审查的模型集成，不从不可信 TOML/request 复制该值；已导入 callable
同样需要信任。

`HyEmbodiedPolicy.from_checkpoint` 默认不读取 `norm_stats.pkl`。解码物理 action 需要
`norm_stats_path`、显式 `trust_norm_stats_pickle=True` 和准确文件的
`norm_stats_sha256`；同一字节先验证 hash，再 deserialize。

多个 policy 配置仍是 Experimental、Component-only 或 Source-only。只使用 release
指定的 constructor、input schema 和 precision，不能从 method 存在推断相邻配置。

## Error hierarchy

用户可见错误在 `rpu_backend.api`：

```text
RPUBackendError
├── RPUConfigError
├── UnsupportedModelError
├── RPUUnsupportedDtypeError
├── RPUSingleHandleError
├── SPMExhaustionError
└── WeightShapeMismatchError
```

通用捕获 `RPUBackendError`；调用方有特定恢复动作时捕获 subclass。非法 Python 参数仍
使用标准 `TypeError`、`ValueError`。

## Graph API

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
pointer-free node/segment 摘要，`Graph.dump_tree()` 按 segment 分组；均不包含 launch
argument word、设备程序或原始 address。

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
`Qwen3VLForConditionalGeneration`。

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
- `set_debug_export` / `get_debug_export` / `list_debug_tensors` /
  `get_debug_tensor` / `clear_debug_tensors`；
- `set_spm_debug` / `get_spm_debug` / `spm_alloc_dump`。

Debug tensor 可能含模型输入或 activation。把这些视为敏感应用产物，不要附到公开
issue；API 不暴露 raw Graph register/resource/plan payload。

CPU/RPU boundary flush 默认打开，正常推理不要关闭。Chunk size 属于 per-handle
`rpu_execution`，不是进程全局 `torch.rpu` setter。内部 `torch.ops.rpu.*` 是 adapter
使用的 host wrapper interface；未在本页或 model example 中记录的函数视为实现细节。
