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

## 其他高价值检查

| 检查 | 失败模式 | 必需修正 |
|---|---|---|
| 顶层 capture | 每次调用都不进入 retained Graph | 除明确受审查的 bounded one-shot 外，生产 native forward 使用 adapter-owned `GraphCache.capture(signature)` |
| 完整 signature | 同 shape、不同 mask/grid/lookup 数据复用旧结果 | 加入 signature、保持经证明的稳定地址，或用 mutable DMA 更新 |
| DMA 地址 ownership | BUILD 通过，REPLAY 读写早期 tensor | fixed 仅用于稳定 model storage；caller input/新 output 用 mutable |
| 多核 broadcast | immediate 多核 operation 只更新一核 | 调用 `set_broadcast_mode(true)`；batch/Graph 已统一处理 |
| Model invalidation | 新 weight/layout 设置复用旧 plan | 每个影响 layout、weight 或 Graph 的 setter 以 `invalidate_model_state()` 结束 |
| CPU fallback | E2E 完成但必需组件没在 RPU 跑 | 明确列出 fallback；预期执行契约未验证前不标 Supported |
| 配置继承 | 一个尺寸/短输入通过就接受相邻配置 | 在 weight load 前 guard 精确配置 tuple 和最大范围 |
| 进程 ownership | 第二个 live fused model 或搬动已转换模型导致冲突 | 每进程一个 live RPU fused model/policy；另一 placement 重载干净 CPU 实例 |

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
