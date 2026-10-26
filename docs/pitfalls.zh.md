# 模型移植常见陷阱

简体中文 | [English](pitfalls.md)

当 port 能运行但数值错误、warmup 后变化，或只在重复/多输入推理失败时使用本页。
增加诊断前先检查 P1–P5，每项都给出一个小型区分检查。

下列契约基于[架构](architecture.zh.md)和公开 adapter/`FusedModelBase` 源码，不能替代
[模型移植](model_porting.zh.md)的完整门禁。

## P1 — 重复 swizzle

**现象：** 转换后权重 norm 看似合理，但 projection 方向错误；干净模型第一次工作，
第二次转换或安装后失败。

**原因：** `swizzle_model_inplace` 两次处理同一权重，或通用 walker 与 fused subsystem
都转换了同一个专用 Linear。转换是不可逆原地操作，不能总靠 tensor shape 识别重复。
Layout 还依赖元素宽度；先按 FP32 转换再 cast FP16，不等价于先 cast FP16 再转换。

**修复：** 干净 CPU 模型在首次转换前进入最终 dtype；每个权重只有一个权威转换路径；
专用 Linear 进入 `SKIP_LINEAR_NAMES`；首次 mutation 前设置 conversion-started 标记；
class patch 幂等；部分失败后丢弃并重载模型。

**门禁：** 干净模型安装一次；确认已完成实例的第二次 `.to("rpu")` 不再次转换；模拟
部分安装后必须拒绝重试。

## P2 — Manifest 之外的 SPM 状态

**现象：** 首次 forward 正确，后续 REPLAY 或带 warmup 时损坏。

**原因：** REPLAY 间需要的数据放在 temporary SPM，或创建在确定性的
`declare_buffers()` manifest 之外；reset/reuse 破坏隐藏生命周期假设。

**修复：** 把 replay 持久数据声明为 `Persistent` 或 `PersistentPerLayer`，需要在
allocation state 变化后恢复时使用 `preload_callback`。`declare_buffers()` 不得 launch，
alias 只能用于生命周期不重叠的 storage。

**门禁：** 比较 `RPU_WARMUP=0/1/3`，再重复一个 signature 直到稳定 REPLAY；值必须
bit-identical，`cache_invariant_ok()` 始终为 true。

## P3 — 绝对地址与 offset 混用

**现象：** shape、weight、SPM capacity 都正确，但输出稳定错误。

**原因：** host wrapper 需要绝对 SPM address 却收到 offset，或反过来。

**修复：** 普通 operation 使用 `addr()`、`layer_addr()`。只有 wrapper 明确要求 typed
`SpmOffset`（如 attention 或 KV-cache insertion）时才使用 `addr_offset()`、
`layer_addr_offset()`；不得 cast 来绕过类型检查。

**门禁：** 按 `src/core/rpu_kernel_decls.h` 检查每个 SPM argument，先与 CPU 参考比较
受影响 operation，再跑端到端。

## P4 — 复用输出 storage

**现象：** 单次调用通过，但调用方保留结果、多输入或累计 image output 时，早期值变化。

**原因：** 两次 forward 返回同一 DDR tensor 的 view，或 Graph REPLAY 写入 BUILD 时捕获
的旧 output address。

**修复：** 每次 forward 为 Python 可见输出分配独立 storage。`FusedModelBase` 可用
`allocate_tracked_output`，其他路径用新的 `at::empty(...)`。新 destination 在 BUILD 与
REPLAY 间变化时使用 mutable DMA。

**门禁：** 保留第一输出并 clone bytes，运行不同第二输入，确认第一输出不变且不与第二
输出共享 storage。

## P5 — 冷 per-handle 执行设置

**现象：** 直接 adapter 与公共 loader 选择不同 chunk plan；首次 forward 后改设置无效，
或复用了不兼容 Graph。

**原因：** chunk/padding 通过已淘汰的进程全局路径设置、在模型安装后改变，或未传入
native handle。

**修复：** 在 `.to("rpu")` 和首次 BUILD 前，通过 loader 的不可变 `rpu_execution`
传入公开 chunk/padding。只有 adapter 明确说明时才把 `_rpu_chunk_size` 当兼容属性；
新集成使用 `rpu_execution`。改变冷设置需要重建模型。

**门禁：** 非法设置在 weight mutation 前失败；每种允许设置在新进程报告并执行预期
resolved plan。

## Pi0.5 v1.2.0 denoise 已知故障（更新于 2026-10-26）

本节记录 v1.2.0 客户现场排查确认的两个关联故障。它们分别影响默认 Graph loop 和
关闭 Graph 后的 host loop。当前 RhinoForge 源码已在 host 与 native extension 层完成兼容修复，不改变
v1.2.0 Runtime/Launch 包或算子资产；未包含该源码补丁的 v1.2.0 构建仍受影响。

### 故障一：不同逻辑前缀持续 BUILD，耗尽 Graph arena

**现象：** `pi05_denoise_loop` 随控制帧连续 BUILD；保留 Graph 数达到冷态 arena 数后报
`graph arena pool capacity exhausted`。这不是同一帧内部每个 denoise step 都重建，而是
每个顶层 action 调用保留一张新的完整 loop Graph。

**根因：** [`_set_denoise_rope_position`](../python/rpu_backend/adapters/pi05/runtime.py)
用 prefix mask 的有效 token 数计算逻辑 RoPE 起点；状态进入 tokenized prompt 后，该值可
随控制帧变化。RoPE 起点会在 BUILD 时写入 kernel，因此
[`_run_denoise_loop`](../python/rpu_backend/adapters/pi05/runtime.py) 正确地把它和物理执行计划
放入 `GraphSignature`。不同值产生 cache miss，而
[`GraphCache`](../python/rpu_backend/graph/_runtime.py) 与 native
[`RpuGraphCache::get_or_create`](../src/graph/graph_infra.cpp) 只做容量检查，不自动淘汰；旧
Graph 继续持有有限的物理 arena，最终耗尽。这里的“泄漏”是 retained Graph 的生命周期
未受界，并非普通 host heap 泄漏。

**解决方案：**

- 有界 cache 修复继续作为兜底：denoise cache 只常驻当前 signature；创建不同 signature
  前通过 `GraphCache.evict()` 安全淘汰旧 Graph。
- 正式性能修复把逻辑 RoPE 起点改为 replay-mutable：从 Python Graph signature 和 native
  plan identity 中移除该值。已验证的 TP8/T160/10-step 路径在 REPLAY 前只更新
  `PARTIAL_MROPE` 和 KV1 direct-cache 节点中的两个 32 位位置寄存器，并校验命中的节点数；
  其他路径保留重发 host layer body 的安全回退。物理 route 仍使用规范化的 padded prefix
  length 校验；Vision 和 Prefill 继续使用原有完整 fast replay。
- 不可用方案：单纯增大 `graph_arena_count` 只会推迟失败；直接从 signature 删除 RoPE
  起点会让 Graph 重放旧语义，造成静默数值错误。

**修复门禁：** 用有效 token 数持续变化的多帧输入验证 cache size 与 arena 占用有界；
长期方案还必须证明一张 Graph 只 BUILD 一次、随后稳定 REPLAY，`cache_invariant_ok()`
始终为 true，并逐位置对比数值一致性。淘汰不得绕过 active/use-count 保护。

**R1 实测：** 客户提供的 6 帧 T160 数据（有效 token 为 149--154），2 次预热后测量 6 次。
denoise Graph 保持 1 张，累计 10 次 REPLAY、0 次 recapture，cache invariant 正常。W8A16
端到端平均 168.27 ms（中位 168.54 ms，P95 169.72 ms），相对修复前动态多帧 209.14 ms
下降 19.54%；denoise 阶段从 101.94 ms 降至 60.01 ms。逐帧对比关闭 Graph 的参考结果，
动作 RMSE 为 0.00049--0.00055，最大绝对误差不超过 0.00340，余弦相似度不低于
0.99999988。v1.2.0 Runtime/Launch 包和算子资产均未修改。

在同一批数据上进一步将精确寄存器更新和 4 个 CPU 线程用于 W8A16 后，端到端平均为
153.69 ms（中位 153.93 ms，P95 154.25 ms），denoise 平均 50.91 ms；六帧动作与上述正确的
整层重发实现逐位一致。Action NVFP4 + Prefill W8A8 的正式 T160 配置平均为 142.97 ms
（中位 142.76 ms，P95 143.93 ms），其中 Vision/Prefill/denoise 分别为
22.36/67.69/44.85 ms。4 线程和 8 线程输出逐位一致，且 4 线程比 8 线程的 145.88 ms 更快，
因此 T160 NVFP4 模板将 CPU 线程限制为 4，避免与同机应用争抢。Prefill W8A8 会引入额外
量化差异，仍需用客户任务成功率验收，不作为无条件默认替换。

### 故障二：关闭 denoise Graph 后稳态动作发散

**现象：** `RPU_PI05_DENOISE_GRAPH=0` 后不再耗尽 arena，但首帧
`[Pi05Fused] runtime checks PASS` 后，固定输入和固定 noise 的后续输出仍发生大幅变化，
真机动作数秒内发散。

**根因：** [`_run_denoise_fused`](../python/rpu_backend/adapters/pi05/runtime.py) 已经计算出完整
`action_plan`，Graph 分支传入所选 `physical_descriptor`，host 分支却传入空列表。native
[`restore_planned_route_profile`](../src/fused/rpu_pi05_denoise_step_model.cpp) 因此进入没有
Graph proof 的 legacy immediate 路径。Graph 外的
[`FusedModelBase::run_preload_callbacks_`](../src/core/fused_model_base.cpp) 可在 generation 看似
未变化且 dirty flag 为 false 时跳过 preload；其间 SigLIP/Gemma Graph 会复用并改写物理
SPM，于是 denoise 的 `action_in_proj_b_spm`、`action_out_proj_b_spm` 等 Persistent bias 可能
读取到被覆盖的数据。首帧在正确 preload 后可通过，而 `_rpu_fuse_validated` 只做一次安全
检查，不能证明后续稳态正确。

固定输入、固定 noise 的三次诊断中，原路径首帧到第二帧 RMSE 为 `0.8649037`、最大差
`2.6068361`；仅保留并传入同一 COMPLETE descriptor 后，三次输出 bit-identical，RMSE
和最大差均为 `0`。这排除了 host FP32 Euler 本身，定位到 descriptor 丢失及 preload/route
authority 不完整。

**解决方案：**

- 当前正式修复：host 分支也传入已选择的 COMPLETE `physical_descriptor`，让 immediate 调用按
  当前物理 manifest 执行并完成所需 preload。
- Graph-only 的 optimized descriptor 在 Graph 关闭时必须提前、明确拒绝，不能静默退回
  空 descriptor 的 legacy 路径。
- 不建议全局改成“所有 legacy forward 都无条件 preload”；该改动影响全部
  `FusedModelBase` 调用方，范围远大于使用本次已经计算好的精确 descriptor。

**修复门禁：** 固定输入与 noise 连续运行至少三次，并在两次 action 之间执行
SigLIP/Gemma Graph replay，输出必须保持一致；再比较 host loop 与参考/Graph loop 的多帧
数值。Graph-only profile 在 `RPU_PI05_DENOISE_GRAPH=0` 时必须 fail closed。首帧一次性的
`runtime checks PASS` 不计作稳态门禁。

### v1.2.0 处置建议

未包含 RhinoForge 源码补丁的 v1.2.0 构建，不要通过增加 arena 或关闭 denoise Graph
规避问题；使用已验证的 v1.0 组合，或重新构建包含两项 adapter 修复的 RhinoForge 包。
代表性 W8A16/R1 验证中，10 帧变化前缀下 denoise cache 始终为 1，最后相同 signature
进入 REPLAY；Graph 关闭后固定输入/noise 连续三次输出 bit-identical、最大差为 0。客户
原始 checkpoint 仍需在交付前执行同样的针对性验收。arena 失败后的 prefix composition
fallback 和进程退出时的 `sys.meta_path is None` 清理异常是后果/次要问题，不是本次两个
故障的根因。

## 其他高价值检查

| 检查 | 失败模式 | 必需修正 |
|---|---|---|
| 顶层 capture | 每次调用都不进入 retained Graph | 除明确受审查的 bounded one-shot 外，生产 native forward 使用 adapter-owned `GraphCache.capture(signature)` |
| Tensor view | 跨 inference-mode 上下文创建 view 后丢失 base 状态或报 version-counter 错误 | 使用 PyTorch 原生 view 构造，保留 base 的 inference 状态、别名、storage offset、边界检查和版本计数 |
| 完整 signature | 同 shape、不同 mask/grid/lookup 数据复用旧结果 | 加入 signature、保持经证明的稳定地址，或用 mutable DMA 更新 |
| DMA 地址 ownership | BUILD 通过，REPLAY 读写早期 tensor | fixed 仅用于稳定 model storage；caller input/新 output 用 mutable |
| 多核 broadcast | immediate 多核 operation 只更新一核 | 调用 `set_broadcast_mode(true)`；batch/Graph 已统一处理 |
| Model invalidation | 新 weight/layout 设置复用旧 plan | 每个影响 layout、weight 或 Graph 的 setter 以 `invalidate_model_state()` 结束 |
| CPU fallback | E2E 完成但必需组件没在 设备 跑 | 明确列出 fallback；预期执行契约未验证前不标 Supported |
| 配置继承 | 一个尺寸/短输入通过就接受相邻配置 | 在 weight load 前 guard 精确配置 tuple 和最大范围 |
| 进程 ownership | 第二个 live fused model 或搬动已转换模型导致冲突 | 每进程一个 live 设备 fused model/policy；另一 placement 重载干净 CPU 实例 |

## 快速定位顺序

1. 比较首次调用与第一次重复调用。只在 repeat 失败，先查 P2、signature 完整性、DMA
   ownership。
2. 保留 output 1 后运行另一输入；发生 mutation，查 P4。
3. 用干净 CPU 模型重跑并定位第一个错误 projection；先查 P1，再查 P3。
4. 对比 requested/resolved execution plan；不一致指向 P5 或 profile preflight。
5. 顶层一致性或任务证据仍失败时，才用精确同 dtype 参考按 model phase 二分；CPU FP32
   只作为精度 anchor。

最后运行[模型验证策略](validation_policy.zh.md)中的完整语义、一致性、任务、warmup、
Graph lifecycle、多输入和最大范围门禁。
