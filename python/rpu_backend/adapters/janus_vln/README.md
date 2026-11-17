# JanusVLN Extra 受控适配

这是 RPU 优先的受控入口，不注册到公共模型加载器；不以与原生模型逐词一致作为当前验收门槛，尚未验证真实导航质量或全部输入包络。
对应源码 `8cfe3129f7af4ca276bd5e38f1fae7e5367a09ec`，checkpoint
`misstl/JanusVLN_Extra@8631eee38a4c688256473ec6a59917fb5eec58ff`。

## 执行分工

- `policy.build_policy(model_class, config, checkpoint)`：连接上游视觉前端、空间融合、共享语言解码器和贪心生成；实际配置重新从 checkpoint 构造。
- `decoder.py`：逐张量加载 BF16 权重并转换为 FP16；复用已有 Qwen2.5 融合解码器。注意力 TP4、MLP TP8，保留 QKV bias、无 Q/K norm、分段 M-RoPE `[16,24,24]`。
- `attention.py`：DINO 与 VGGT frame/global attention 的投影及 QK、PV 矩阵乘法在 RPU；FP32 LayerNorm/RoPE/在线 softmax 在 CPU。注意力按单头、Q128/K1024 分块，FP16 原始 K/V 常驻主存。
- `vision.py`：Qwen 视觉注意力按上游图像／窗口边界独立计算，避免窗口之间串扰；视觉与 DINO patch 卷积展开后交给 RPU 投影。
- 视觉、DINO、VGGT 的全部 MLP，空间 merger、视觉 merger 和 lm_head 均使用 RPU FP16 线性层。输入宽度补齐到 16、输出宽度补齐到 128，避免不对齐形状触发 CPU fallback；输出会裁回原尺寸。
- embedding 查表、归一化、位置编码、softmax、激活函数与张量整理留在 CPU。该路径不能视为与原生 BF16/autocast 路径等价。

注意力把每块 K/V 转为设备线性投影权重，通过共享 SPM arena 的 TP8 Linear 执行 QK/PV；不调用会回退 CPU 的 eager SDPA/matmul。独立 eager `bmm` 使用另一套 SPM 分配器，实机整模型共存测试发生过分配冲突，因此已弃用。完整融合的 Qwen/DINO 编码器在本模型尺寸下被原生 planner 拒绝，不会绕过容量检查。在线 softmax 仍覆盖全部历史，不裁掉尾部或拆成独立局部 softmax。

每次导航请求只追加一帧空间记忆。第 57 帧先完整参与注意力，再逐层保留最早 8 帧和最近 48 帧。每生成一个 token 重算文本 prefill，暂不使用 native decode 快路径。保留一份 Graph，逻辑长度变化时在前次 CPU 回读完成后释放旧图。位置/RoPE 源张量必须保留到新源分配和上传完成，之后仅保留当前物理长度的缓存，避免内存地址复用造成更新遗漏。generate/reset/close 串行执行。

## 输入与环境

只接收 batch=1、VGGT 当前帧 `[1,3,490,644]`，Qwen grid `[1,34,46]`，最多 9 张语义图、32 个输出 token。文本容量 4096，预留最多 64 行 padding。输入是上游 processor 准备的 CPU 张量；不接收视频、采样或有 padding mask 的请求。

需要冻结上游源码与 Transformers 4.50.0。R1 已有环境是 Transformers 5.3.0，整入口执行须使用隔离的兼容依赖层，不能覆盖现有服务环境。模型实例必须显式 `close()`；失败的导航步不能继续使用原实例。

冷启动会在首次设备分配前启用已有 caching allocator，以复用分块投影的临时设备内存。若进程此前已冻结为 direct allocator，必须换新进程。直接分配模式已在 R1 复现驱动4096映射上限错误，主存有余量也不能避免该错误；不能用增大主存预算代替分配复用。

## 内存与验证边界

`memory.py` 只读 safetensors 头进行配置、形状数据区和预算检查。加载前比较 `/proc/meminfo` 的 MemAvailable、CmaFree，计划保存在 `policy.memory_plan`。CMA检查不能保证连续分配；SPM 分块由 native planner 最终决定。设备预算按全部 checkpoint FP16 权重加文本 KV、工作区与64MiB权重补齐余量保守估计。512MiB 图/设备工作区、2GiB 前端激活及4GiB系统余量是预算，不是实测峰值。

`checkpoint.py` 支持本地文件，或通过固定回环地址的 SSH 隧道逐张量加载；流式模式要求 `stream_manifest.json` 提供冻结 revision、config/index 摘要、分片头和逐张量 SHA256。源端必须先对照官方清单校验文件，再生成此清单。接收端验证长度和摘要后才转换权重，不在 R1 缓存整份 checkpoint。该方式省去约17.35GiB的内存盘文件副本，原混合基线规划总预算从52.81GiB降至35.46GiB；当前额外预留64MiB补齐空间；不降低权重精度、图像尺寸或56帧记忆长度。隧道仅在加载期间需要保持连接。

已验证：独立分块 softmax、原始缓存和 57→56 裁剪、生成接线/图切换、失败保护、并发关闭、预算拒绝及截断文件拒绝；R1 首层真实 attention 两种历史各三次，以及 57 帧合成历史。

旧混合基线（提交 `5c0bec07`）在2026-11-17 整模型验收通过，以下数据不代表当前 RPU 优先版本：完整 frontend 与28层融合 decoder 加载，两步动作与原生逐词一致，实际 Graph REPLAY 逐值一致，重置后全部24层 K/V 逐值复现，全部24层合成历史57→56裁剪正确。进程正常退出，实测 RSS 高水位29.84GiB，无设备 OOM/分配重试。两步含检查耗时398.48/554.40秒，满历史请求814.01秒。

待验证：真实导航质量、9张语义图/4096文本容量的组合上限、长时间服务及实时性能。此次固定合成输入验收不能推广成任意输入或生产环境认证。


RPU 算子回归：在持有设备锁的 R1 环境显式运行 `python tests/janus_rpu_ops_check.py`，覆盖卷积／MLP非对齐宽度、词表投影、单帧／两帧注意力、80维视觉头和下溢历史块。此检查不会自动并入主机测试；主机测试为 `python tests/test_janus_memory_attention_host.py`。


## RPU 优先版本实测（2026-11-17）

R1 固定输入整模型测试通过：455 个前端线性投影驻留 RPU，104 个视觉／空间注意力模块的 QK/PV 使用设备 Linear，28 层语言解码器继续使用原生融合路径。连续两步耗时271.90／409.87秒，重置后首步复测272.76秒，输出 token 与首次一致。此次三次请求均最多生成3个 token；未启用原生逐词精度门槛。

冷启动1351.77秒，进程 RSS 高水位18.93GiB，设备 allocator 预留峰值17.05GiB，OOM与分配重试均为0，进程退出码0。主机11项测试及显式RPU算子检查通过；额外验证了融合解码器→设备注意力→解码器Graph重放的共存关系。

旧基线398.48／554.40秒包含更多诊断检查，不能把差值全部归因于计算迁移。当前仍是分钟级推理；未验证导航精度、完整56帧历史的整模型耗时或长期运行稳定性。退出阶段现有运行库仍会打印一次 `WallOssLLM.__del__` 的 Python shutdown 提示，设备正常释放且退出码为0。
