# JanusVLN Extra 受控适配

这是已通过 R1 固定输入整模型联调验收的受控入口，不注册到公共模型加载器；尚未验证真实导航质量或全部输入包络。
对应源码 `8cfe3129f7af4ca276bd5e38f1fae7e5367a09ec`，checkpoint
`misstl/JanusVLN_Extra@8631eee38a4c688256473ec6a59917fb5eec58ff`。

## 执行分工

- `policy.build_policy(model_class, config, checkpoint)`：连接上游视觉前端、空间融合、共享语言解码器和贪心生成；实际配置重新从 checkpoint 构造。
- `decoder.py`：逐张量加载 BF16 权重并转换为 FP16；复用已有 Qwen2.5 融合解码器。注意力 TP4、MLP TP8，保留 QKV bias、无 Q/K norm、分段 M-RoPE `[16,24,24]`。
- `attention.py`：VGGT frame/global attention 的 QKV、输出投影在 RPU，FP32 LayerNorm/RoPE/全局 softmax 在 CPU；Q128/K1024 分块，FP16 原始 K/V 常驻主存。
- 其余视觉、VGGT MLP、空间 merger、embedding 和 lm_head 在 CPU。它们使用 FP16，不能视为与原生 BF16/autocast 路径等价。

每次导航请求只追加一帧空间记忆。第 57 帧先完整参与注意力，再逐层保留最早 8 帧和最近 48 帧。每生成一个 token 重算文本 prefill，暂不使用 native decode 快路径。保留一份 Graph，逻辑长度变化时在前次 CPU 回读完成后释放旧图。位置/RoPE 源张量必须保留到新源分配和上传完成，之后仅保留当前物理长度的缓存，避免内存地址复用造成更新遗漏。generate/reset/close 串行执行。

## 输入与环境

只接收 batch=1、VGGT 当前帧 `[1,3,490,644]`，Qwen grid `[1,34,46]`，最多 9 张语义图、32 个输出 token。文本容量 4096，预留最多 64 行 padding。输入是上游 processor 准备的 CPU 张量；不接收视频、采样或有 padding mask 的请求。

需要冻结上游源码与 Transformers 4.50.0。R1 已有环境是 Transformers 5.3.0，整入口执行须使用隔离的兼容依赖层，不能覆盖现有服务环境。模型实例必须显式 `close()`；失败的导航步不能继续使用原实例。

冷启动会在首次设备分配前启用已有 caching allocator，以复用分块投影的临时设备内存。若进程此前已冻结为 direct allocator，必须换新进程。直接分配模式已在 R1 复现驱动4096映射上限错误，主存有余量也不能避免该错误；不能用增大主存预算代替分配复用。

## 内存与验证边界

`memory.py` 只读 safetensors 头进行配置、形状数据区和预算检查。加载前比较 `/proc/meminfo` 的 MemAvailable、CmaFree，计划保存在 `policy.memory_plan`。CMA检查不能保证连续分配；SPM 分块由 native planner 最终决定。512MiB 图/设备工作区、2GiB 前端激活及4GiB系统余量是预算，不是实测峰值。

`checkpoint.py` 支持本地文件，或通过固定回环地址的 SSH 隧道逐张量加载；流式模式要求 `stream_manifest.json` 提供冻结 revision、config/index 摘要、分片头和逐张量 SHA256。源端必须先对照官方清单校验文件，再生成此清单。接收端验证长度和摘要后才转换权重，不在 R1 缓存整份 checkpoint。该方式省去约17.35GiB的内存盘文件副本，规划总预算从52.81GiB降至35.46GiB；不降低权重精度、图像尺寸或56帧记忆长度。隧道仅在加载期间需要保持连接。

已验证：独立分块 softmax、原始缓存和 57→56 裁剪、生成接线/图切换、失败保护、并发关闭、预算拒绝及截断文件拒绝；R1 首层真实 attention 两种历史各三次，以及 57 帧合成历史。

2026-11-17 整模型验收通过：完整 frontend 与28层融合 decoder 加载，两步动作与原生逐词一致，实际 Graph REPLAY 逐值一致，重置后全部24层 K/V 逐值复现，全部24层合成历史57→56裁剪正确。进程正常退出，实测 RSS 高水位29.84GiB，无设备 OOM/分配重试。两步含检查耗时398.48/554.40秒，满历史请求814.01秒。

待验证：真实导航质量、9张语义图/4096文本容量的组合上限、长时间服务及实时性能。此次固定合成输入验收不能推广成任意输入或生产环境认证。
