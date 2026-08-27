# 架构

简体中文 | [English](architecture.md)

RhinoForge 是面向 Rhino Processing Unit 推理的 PyTorch `PrivateUse1` backend。
导入 `rpu_backend` 会注册设备名 `rpu`，在板卡可访问时加载 native extension，安装
`torch.rpu` namespace，并延迟注册模型 adapter。

```text
Hugging Face model 或 policy API
              |
        Python adapter
 profile guard、weight layout、
 cache 与 graph orchestration
              |
       torch.ops.rpu.*
              |
 C++ operators / FusedModelBase
 SPM plan、多核 schedule、
 graph、batch 与 DMA nodes
              |
       Rhino Launch 1.0.0
              |
             RPU
```

合并算子资产由 host launch interface 加载。对 RhinoForge 它是 opaque 的：backend
选择具名 operation 并提供公开 host ABI 参数，不解析资产格式或解释设备程序。kernel
可用性来自相邻的版本化 `.kernels` 发行 manifest，其中只包含资产字节数和
kernel 名称。

## Runtime 分层

### Python API 与 adapter

公共 loader 和 policy facade 位于 `rpu_backend.api`。Loader 先读模型配置，解析精确
Hugging Face architecture，再从 adapter registry 取得实现。Adapter 随后：

1. 在加载或修改权重前拒绝不支持的配置；
2. 把支持的权重转换为 RPU operation 消费的 layout；
3. 把 tensor 移到 `rpu`；
4. 创建 native model handle 与 cache；
5. 安装模型专用 forward；
6. 管理 Graph signature 和生命周期。

权重转换是不可逆原地操作。不要对同一模型实例转换两次，也不要把转换后实例移回 CPU。
主要 fused CausalLM 路径还限制每个进程只有一个 live RPU-resident model 或 policy。

Built-in adapter 来自静态 manifest，仅在选中时导入。第三方 adapter 使用同一 registry，
不能覆盖 built-in architecture。见 [API 参考](api_reference.zh.md)。

### Operation 与 CPU fallback

Native operation 为 PyTorch `PrivateUse1` dispatch key 注册实现，并在 `torch.ops.rpu`
注册模型 schema。支持的 tensor operation 会直接 launch RPU operation，或向 active
Graph 添加 node。没有 RPU 实现的 operation 可以走受控 CPU fallback；所有调用能返回
并不等于模型配置受到支持。

公开 operation 边界是 host wrapper：源码可表达 tensor shape、dtype、address、length、
core selection、同步与参数打包。设备程序源码、设备程序语义和算子资产内部不属于本仓库。

## FusedModelBase 契约

`v3::FusedModelBase` 是 fused model subsystem 的共享 C++ 框架。它把重复 decoder 或
encoder 中间量保留在每核 SPM，并统一 chunk planning、allocation、Graph construction、
DMA 和 model-state invalidation。

新的 fused subclass 实现四个方法：

- `declare_buffers(const LayoutContext&)` 返回 SPM buffer manifest 和生命周期。Planner
  可能为多个候选 chunk 调用它，因此必须 deterministic 且无 launch side effect。
- `static_config()` 声明模型级 layer/traversal 配置。
- `dynamic_config(const ChunkPlan&)` 为 resolved plan 选择执行顺序和跨层 storage 模式。
- `build_layer_subgraph(layer_idx, chunk)` 通过现有 launch wrapper 发射一个 layer/chunk
  的 operation 顺序。

Subclass 改变 bound weight 或任何影响 layout/Graph 的状态后，setter 必须以
`invalidate_model_state()` 结束。框架自有、可跨 forward 存活的 output 使用
`allocate_tracked_output`；Python 调用方可能累计的 output 必须有独立 storage。

普通 operation 需要绝对 SPM address 时使用 `addr()`、`layer_addr()`。只有 host 契约
明确需要 offset 的 operation（如 attention、KV-cache insertion）才使用 offset access。

### Planning 与 invalidation 生命周期

Fused handle 分开维护 model-state invalidation、SPM allocation identity 和 Graph
identity。Setter 绑定 weight/config 并调用 `invalidate_model_state()`。下一次允许的
forward 解析 chunk plan，评估确定性 buffer manifest，分配匹配 layout，并向
adapter-owned Graph BUILD 添加 operation。后续调用只有在 model state、allocation
identity、Graph signature 和 live DMA input 都满足同一契约时才能 REPLAY。

`invalidate_model_state()` 把模型自有 weight/preload 标为 dirty，不会自动产生新的
allocation identity。若 `declare_buffers()` 依据 `LayoutContext` 和标准模型参数之外的
subclass state（如 image grid、action horizon、expert count）决定尺寸，subclass 必须将
其加入 `subclass_layout_hash()`，否则新 shape 可能错误复用旧 SPM layout。

用 `subclass_chunk_size_cap()` 限制自动 chunk 搜索，用
`subclass_chunk_size_valid()` 表达每个自动或显式候选都要满足的约束。与
`declare_buffers()` 一样，两者必须 pure/deterministic；拒绝未证明候选优于发射不安全
Graph。

### 三阶段 chunk 执行

Packed 或多组件输入可能在 input load、Q/K/V 生成和剩余 layer compute 使用不同 chunk
边界。`FusedModelBase` 用一个三阶段 plan（`input`、`qkv`、`compute`）加 semantic spans
表达，保证每行的源 stream 边界。共享 executor 支持每阶段一个或多个 chunk；subclass
提供 plan 和 layer operation，不复制执行循环。

Physical prepare 绑定准确 plan fingerprint。Chunk boundary、semantic span、traversal
mode、layout 和 resolved attention storage 因而属于同一个 Graph identity，而非 ambient
mutable state。

Attention storage 默认 `AUTO`。只有精确配置已认证且联合 SPM layout 可容纳时，才解析为
model-certified SPM-resident K/V；否则使用 DDR-cache。显式要求 SPM 但任一条件不满足时
fail closed。

## SPM 设计

当前 runtime 面向八核、每核 8 MiB SPM，保守规划预算约 7.5 MiB/核。FP16 是主要
execution dtype，很多 operation 要求 16-element 对齐。

`BufferDecl` 表达 storage 和生命周期：

- `Temp`、`TempPerLayer` 可在声明 phase 结束后复用；
- `Persistent`、`PersistentPerLayer` 跨 Graph REPLAY 存活；
- `alias_of` 只有 producer/consumer 生命周期不重叠时才能显式复用。

Chunk planner 使用与 allocator 相同的 alias-aware first-fit plan。SPM 容量是硬
admission constraint。在可行的 16-aligned 候选中，通用 planner 先最小化 chunk 数，
再最小化最后 chunk 缺口。允许的模型专用逻辑 padding 留在 adapter；返回调用方前从
output 和 cache position 中移除。

REPLAY 后仍需的数据必须进入 manifest，必要时带 preload callback。Subsystem 边界重置
temporary SPM，并由对应 handle 在下个 subsystem 开始前释放 compute ownership。

## Graph 与 batch 执行

`GraphSignature` 标识一种允许的执行 shape 和语义模式。`GraphCache.capture(signature)`
把首次调用记录为 BUILD，后续同 signature 调用复用 retained Graph 为 REPLAY。每个
retained cache entry 拥有自己的 launch queue。生产 adapter 通常用此 scope 包裹顶层
native forward。

重复 signature 只有在下列条件全部成立时有效：

- 首次只 BUILD 一次；
- 后续 REPLAY 不再 BUILD；
- replay counter 增长而 cache size 固定；
- `cache_invariant_ok()` 保持 true。

某些变 shape 路径可定义 bounded one-shot Graph。它必须记录 non-trivial Graph、执行
一次、返回 passthrough 状态，并保持 retained-cache size 不变。这是显式模型契约例外，
不是默认 porting pattern。

Batch 把 kernel、DMA 和同步 node 组合成 launch segment。一核和八核 operation 可以共存
于一个 batch，每个 node 保留自己的 core-count 契约。Immediate 多核 launch 必须打开
broadcast mode，使所有参与 core 收到 host 参数；batch/Graph 路径统一处理。

标准 residual reduction 使用共享生成式八核 ring，wrapper 根据已验证 geometry 自动
选择 schedule；partial producer 先清 inactive shard。Linear 的 FP16/W8/W4-pgrp 路径
同样由共享 generated planner 自动选 tile，不再接受模型级 route/tile selector。见
[自动 residual reduction](../knowledge/concepts/allreduce-routing.md)和
[Linear 自动分块](../knowledge/concepts/linear-autotiling.md)。

## DMA ownership

RhinoForge 提供显式 DDR→SPM broadcast 和 SPM→DDR copy wrapper。Replay 行为属于
Graph correctness：

- fixed wrapper 在 BUILD 捕获 DDR source/destination address，只适用于跨 forward 地址
  稳定的 storage；
- mutable wrapper 在 REPLAY 前更新 live DDR address，适用于 caller input 或新分配
  output；
- immediate wrapper 在 Graph capture 外做 one-shot transfer。

Mutable DMA 只改变 replay address，不消除物理搬运。SPM 直接写最终 DDR 只去掉 host
staging，不是 zero-copy。延迟执行还必须持有 source/destination tensor 引用，见
[延迟 DMA 生命周期](../knowledge/concepts/deferred-dma-lifetime.md)。

## 源码目录

| 路径 | 职责 |
|---|---|
| `python/rpu_backend/api/` | 公开 model、cache 和 policy interface |
| `python/rpu_backend/adapters/` | 模型配置 guard 和 orchestration |
| `python/rpu_backend/graph/` | Graph signature、cache 和 capture scope |
| `python/rpu_backend/runtime/` | 共享 device、registry、weight 和 decoder helper |
| `python/rpu_backend/quant/` | 离线量化和 checkpoint loading |
| `src/core/` | Backend state、memory、launch cache 和 FusedModelBase |
| `src/graph/` | Capture、segmentation、replay 和 fallback |
| `src/ops/` | Tensor operation 的公开 host wrapper |
| `src/fused/` | 模型专用 fused subsystem |

支持行为由精确模型配置和 release asset set 定义，不由源码文件存在定义。见
[模型支持](model_support.zh.md)和[模型移植](model_porting.zh.md)。
