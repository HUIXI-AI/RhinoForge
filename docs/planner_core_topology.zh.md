# 计算核拓扑

**简体中文** | [English](planner_core_topology.md)

加载权重前设置 `rpu_execution.model.num_cores`，只接受整数 `4/6/8`，默认 `8`；
具体模型配置会进一步限制范围。布尔值不能作为核数。修改核数需重新创建模型、缓存和 Graph。

核数选择整组组件拓扑，不改变物理八核存储布局，也不是逐算子的线程参数。

| Dense decoder 核数 | Attention | MLP | 输出 head |
|---|---:|---:|---:|
| 4 | 4 | 4 | 4 |
| 6 | 4 | 6 | 4 |
| 8 | 8 | 8 | 8 |

子组件可以少于所属模型的核数，例如六核模型内的四核组件。安装时由 adapter 确定组合。

## 模型配置

| 模型 | 缩核范围 |
|---|---|
| Qwen3 / Llama / Gemma4 | Qwen3 0.6B–32B（按配方支持 FP16、W8A16、W4A16 G32、NVFP4）、Llama 3.2-1B（6 核 MLP 补齐 `8192 → 8256`）及 Gemma4 E4B（2 核 attention，6 核 MLP 补齐 `10240 → 10272`）支持 4/6/8 核。 |
| Qwen3-VL / DINOv3 | Qwen3-VL 2B–32B（含缩核下四核 Vision W8/W4/NVFP4）及 DINOv3 ViT-B（FP16、W8A16、W4A16 G32）支持 4/6/8 核。 |
| Qwen3.5 Dense & 35B-A3B | Dense 0.8B–27B（按模板配方准入；27B W4 group32 固定 8 核）以及 35B-A3B mixed-E4M3、RTN-W8A16 支持 4/6/8 核。 |
| Qwen3 ASR / TTS | 0.6B 与 1.7B 的 FP16、W8A16、W4A16 G32、NVFP4 支持 4/6/8 核；4/6 核时音频编码器和 codec Transformer 使用四核。 |
| Pi0.5 与 VLA 族 | Pi0.5 全部 29 份 LIBERO/Kuavo 模板（完整前缀 W8A8 除外）、G0.5 K1、GR00T-N1.7、HyVLA、HY-Embodied、InternVLA-N1、LingBot v2、RhinoVLA v3 及 Wall-OSS Base/RTC/Sorting（Sorting Action W4 G32 除外）支持 4/6/8 核；各族 Legacy 入口与 HALO 保持 8 核。 |

各入口的 checkpoint、精度和输入限制见[配置索引](../examples/configs/CONFIGURATIONS.md)。
官方 Qwen3.5-27B 的 W8 接受 4/6/8，W4 group32 接受 8；相同架构名不能替代 checkpoint 元数据检查。

## 各模型的具体配置

Qwen3 FP16 缩核入口在转换权重前检查配置与实际模块。词表为 151936，head dimension
为 128，物理 KV 保留八条分片。

| 结构 | 层数 | Hidden | 逻辑 MLP width | Q / KV heads | 六核 MLP 物理 width |
|---|---:|---:|---:|---|---:|
| 0.6B | 28 | 1024 | 3072 | 16 / 8 | 3072 |
| 1.7B | 28 | 2048 | 6144 | 16 / 8 | 6144 |
| 4B | 36 | 2560 | 9728 | 32 / 8 | 9792 |
| 8B | 36 | 4096 | 12288 | 32 / 8 | 12288 |

六核时词表和 attention 仍分为四份，MLP 分为六份。量化、滑动注意力、M-RoPE、AdaRMS、
混合累加、自定义 projection 和批量输入不沿用此 FP16 配置。精确 8B adapter 先校验模型，
上传 embedding/head，再逐层转换、上传 Linear 并释放对应 CPU 存储；保留共享权重和转换标记。
部分转换失败后须重新加载模型。

Qwen3-VL 缩核保留交错 M-RoPE `[24,20,20]`、DeepStack 消费层 `[0,1,2]` 和句柄级
ACC32。Patch embedding 与位置插值使用 CPU FP16，最终及 DeepStack merger 使用 CPU FP32。
修改任一 tower 前先检查两者。量化、packed-delivery、32B、VLA 和批量配置不沿用此入口。
文本 cache 使用 `RPUCache.from_model(model.model.language_model, ...)` 创建。

Qwen3.5 0.8B/2B/4B/9B 的文本 MLP width 分别为 3584/6144/9216/12288，六核时只有
0.8B 需要补齐。精确 2B `w8_main_w8_gdn` 也接受四核、六核：main 与 GDN q/k/v/z/out
projection 使用 signed INT8 和 FP16 per-channel scale；GDN b/a、卷积、归一化、embedding
及 LM head 保持 FP16，文本累加为 ACC16。9B Vision 的物理 head/MLP width 为 80/4352。
视频和其他量化结构仍按各自入口检查。

Qwen3.5-35B-A3B mixed-E4M3 使用独立拓扑。权重按
[官方源下载与 CPU 转换命令](../examples/configs/WEIGHTS.md#cmd-qwen35-moe)准备。
三种核数共用转换产物，核数对应的补齐和打包在冷态安装时完成：

| 核数 | Attention / GDN / router / head / Vision | Routing / routed / shared | Expert 逻辑 / 物理 width |
|---|---|---|---|
| 4 | 4 / 4 / 4 / 4 / 4 | 4 / 4 / 4 | 512 / 512 |
| 6 | 4 / 4 / 4 / 4 / 4 | 6 / 6 / 6 | 512 / 576 |
| 8 | 8 / 8 / 8 / 8 / 8 | 8 / 8 / 8 | 512 / 512 |

原生布局元组为 `(2, root, attention, GDN, router, routing, routed, shared, head, physical_KV)`，
六核对应 `(2,6,4,4,4,6,6,6,4,8)`。新旧布局 getter 都要求权重已经安装；adapter、cache 和
Vision bridge 检查完整元组。Gate/up 先拆分逻辑两半再补齐；routed E4M3 使用零 code 和单位
scale，shared FP16 projection 同步补零。Router 结果先进入模型持有的固定缓冲区，再由六核
routing 读取；replay 不重新打包权重。入口见 [mixed-E4M3 模板](../examples/configs/qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn.toml)（通过 `rpu_execution.model.num_cores` 选择 `4`、`6` 或 `8` 核）。

Pi0.5 FP16/W8A16 缩核使用 Vision4、attention4、conditioning4，以及 MLP4 或 MLP6。
各相机分别以 256 行执行，物理 KV 保留八条分片。其他模型按自身入口检查；HALO 的 `attn_tp`
和 `mlp_cores` 用于核对固定布局，不能自由选择新拓扑。

## 逻辑形状与物理补齐

保留原模型配置和输入形状，由 adapter 一次完成权重补齐与转换，再绑定所属模型：

- 六核 MLP 的 intermediate width 可能为 `3584 → 3648`、`9728 → 9792` 或 `17408 → 17472`。
- Qwen3-VL 8B 缩核 Vision 的 head width 为 `72 → 80`，MLP width 为 `4304 → 4352`。
- mixed-E4M3 MoE 六核 expert 的 intermediate width 为 `512 → 576`。

这些都是存储尺寸，不是新的模型结构。不要预先修改 checkpoint 或 Hugging Face 配置，避免二次转换。

## Cache 与 Graph 所有权

确定拓扑后，通过 `from_model` 创建 `RPUCache` 或 `Qwen3_5Cache`。
Qwen3.5 还包含卷积和递归状态；只从配置构造 cache 时，必须显式匹配模型核数，默认仍为八核。
`reset()` 同时清空递归状态和逻辑位置；回退到非零位置不能重建当时的递归状态。

Attention 使用四核计算时，KV 存储仍保留八条物理分片。计算核数、存储布局和逻辑序列长度各自独立。
Graph 子组件不能超过所属模型的核数，队列和数据传输使用同一所有者。
各模型在安装权重前预留 Graph arena；冻结后增加组件不能扩大已有预留。
Qwen3.5 9B 图文路径须在安装前设置模板的 Vision 启用项；仅文本安装的模型不能原地增加 Vision。
精确 dense FP16 9B 在权重上传前固定预留四个同尺寸 Graph arena，仅文本安装也会预留。
Decode 和 Vision 分别持有长期 lease，有限 prefill 与诊断 Graph 临时借用 lease；安装后池大小
不能增长。关闭模型归还活动 lease，池本身保留到进程退出；`rpu_execution` 不开放 arena 数量选项。

[执行 API](api_reference.zh.md#执行配置与运行中重配)允许在请求之间更新受支持的 chunk 和 padding，
这类更新不会改变冷态拓扑。

Decoder layout hash 和相关算子清单包含完整冷态拓扑；共享 FMB 路由绑定 schema、消费核数
和生产核数。Chunk 仍由各核实际 SPM 容量决定。

Attention TP4 与物理八分片对应有效 KV slots `8*8/4=16`，batch 为一时 K/V 形状为
`[1, blocks, 2, 8, 8, 16, 16]`。Python 与原生 forward 检查真实形状和容量，不能只修改
旧 TP8 cache 的元数据。

同一 graph segment 内的 kernel 可以使用不同核数：segment queue 使用最大的核域，
每个 kernel 保留自己的 `core_ids`。Graph kernel 使用 `[0,N)` 内连续核区间，queue prefix 覆盖该区间；DMA-first/only segment
使用 owner 核数，DMA engine ID 必须在 `0..N-1`。DDR-to-DDR 复制使用独立同步队列并在
完成后释放 batch；它与 immediate、retained 队列各自管理生命周期。普通 Graph 提交在四核、
六核时最多 4096 entries，八核时最多 8192，同时不得超过 SDK 配置上限的一半；这些限制不改变
planner chunk 或算子核数。

Residual ring 按消费核数归约；只有部分核生产数据时，先清空所有消费分片再写入有效前缀。
Pacing 使用每消费核的对齐字节数和 11 KiB 分界。Local-SPM、mixed residual、fixed-delivery
collective 和标准 SPM pipeline ingress 仍限定八核。

源码：[拓扑解析](../python/rpu_backend/runtime/topology.py)、
[Qwen3.5 adapter](../python/rpu_backend/adapters/qwen3_5/__init__.py)；参数见[运行时配置](runtime_config.zh.md)。
