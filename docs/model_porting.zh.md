# 模型移植

简体中文 | [English](model_porting.md)

本指南覆盖只使用 RhinoForge 公开 adapter、operation、Graph、SPM 和 host-launch 接口的
推理 port。Port 只对精确模型配置完成：architecture、checkpoint revision、precision、
input envelope、execution setting、Rhino Launch 和算子资产版本。复用源码或匹配同一
Hugging Face architecture name 不是支持声明。

改代码前先对照[模型支持](model_support.zh.md)，写下要接受的精确配置。不支持配置必须在
完整 weight load 或任何原地转换前失败。

AI 辅助工作使用仓库内的 [`rhinoforge-port` skill](../.agents/skills/rhinoforge-port/SKILL.md)。
先用 assess 模式，审核 outcome 与 certification；只有 accepted、certified、非 Blocked
时才进入 port 模式。

## 1. 评估能力缺口

从输入预处理到返回结果追踪一次完整 reference forward。记录每项 operation 的 shape、
dtype、state update、cache 行为和数值参考，再映射到现有公开 RhinoForge 路径。

选择最小适用 outcome：

| Outcome | 使用条件 |
|---|---|
| Existing path | 已有支持 adapter 和精确配置完整覆盖 |
| Adapter-only | 现有 operation 能表达模型，只需调整配置、weight、cache 或 forward orchestration |
| Runtime extension | 需要新的 host-side operation wrapper 或共享 runtime 行为 |
| New fused subsystem | 需要新的重复 fused execution path 和 SPM plan |
| Blocked | 必需 operation asset、shape、precision、memory envelope 或数值结果不可用 |

不得为“让 forward 跑完”而近似缺失数学。受限算子资产缺少 operation 时，按数学定义、
tensor shape/dtype 和 CPU reference 请求 release 匹配的资产更新；设备程序实现细节不在
公开 porting workflow 内。

## 2. 从 adapter 模板开始

Decoder-only CausalLM 复制：

```text
python/rpu_backend/adapters/_template/minimal_causal_lm.py
```

需要更多注解结构时用同目录 `adapter.py`。注册前填写：

- `HF_ARCH`：精确等于 `config.architectures[0]`；
- `DECODER_ARCH`：只有数学匹配时才选择已有 decoder branch；
- `SUPPORTED_PROFILES`：精确配置 tuple，不是家族范围；
- `RMSNORM_CLASS`、`ROTARY_CLASS`：目标 Hugging Face class；
- `SKIP_LINEAR_NAMES`：由另一条权威转换路径处理的专用 Linear；
- 配置允许的 chunk envelope。

Adapter 的 Python 职责包括配置/依赖 preflight、CPU-first checkpoint loading、一次权重
转换、小型 class/layout boundary patch、cache/native handle ownership、Graph signature、
forward orchestration、teardown 和进程 ownership。只有现有公开 operation 无法表达数学
或生命周期时才使用 C++。

## 3. 数学匹配时复用 causal decoder

保持最小模板顺序：

1. 验证精确配置和冷 `rpu_execution`；
2. 声明 live model instance；
3. 标记不可逆转换已开始；
4. 每个 weight 只转换一次；
5. 把 parameter/buffer 移到 `rpu`；
6. 创建并配置 native handle；
7. 绑定 weight，安装 forward；
8. 只有此前全部成功才标记 runtime 完成。

不要把转换后模型移回 CPU，也不要在同一实例重试部分失败的转换；重新加载干净实例。
调用方可能保留或拼接的 Python output 每次 forward 需要新 storage。

## 4. 只在必要时添加 fused subsystem

新的重复 fused 路径继承 `v3::FusedModelBase`，实现：

```cpp
std::vector<BufferDecl> declare_buffers(const LayoutContext&);
ModelStaticConfig static_config();
ModelDynamicConfig dynamic_config(const ChunkPlan&);
void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk);
```

`declare_buffers` 必须 deterministic、无 launch side effect。Replay 持久 SPM state 用
`Persistent`/`PersistentPerLayer`，不能从 temporary path 分配。普通 host 契约需要
绝对地址时用 `addr()`/`layer_addr()`；只有明确要求 offset 时才用 offset。

Manifest 若依赖 `LayoutContext` 和标准模型参数以外的 subclass sizing state，应按
[planning 生命周期](architecture.zh.md#planning-与-invalidation-生命周期)实现
`subclass_layout_hash()`。每个影响 layout、planning 或 Graph 的 setter 以
`invalidate_model_state()` 结束。

Immediate 多核 launch 必须 broadcast；优先使用框架 batch/Graph 统一处理。DDR/SPM
transfer 明确选择：稳定地址用 fixed DMA，caller input/新 output 用 mutable DMA，capture
外 one-shot 用 immediate DMA。

## 5. 新增或复用公开 operation

现有 operation 足够时调用其公共 host wrapper，不重复实现。新增 host-side operation
通常需要：

1. 在 `src/core/rpu_kernel_decls.h` 声明 wrapper；
2. 在 `src/ops/` 实现参数验证和 launch orchestration；
3. 通过 `src/core/rpu_dispatch_registrations.inc` 添加 schema 和 `PrivateUse1` 注册；
4. 在 `src/CMakeLists.txt` 加入源码；
5. 只有公开配置承诺 `torch.compile` 时才添加 FakeTensor 实现。

`rpu_dispatch_registrations.inc` 由主 backend translation unit include，不作为独立源码
编译。Wrapper 可公开选择具名 operation 所需的 host ABI：tensor metadata、memory
location、length、core selection、sync 和参数打包。受限资产留在仓库外，只使用 release
公开 lookup interface。

标准 residual reduction 与 Linear tile 已由共享 generated planner 自动处理。不要在
新 adapter 添加退役的 all-reduce route 或 Linear tile selector；分别见
[自动归约](../knowledge/concepts/allreduce-routing.md)和
[Linear 自动分块](../knowledge/concepts/linear-autotiling.md)。

## 6. 注册 adapter

In-tree adapter 在 `python/rpu_backend/adapters/_manifest.py` 加 architecture/module/class
tuple；模块也调用 `register_adapter(HF_ARCH, AdapterClass)`，保证 direct import 与 lazy
manifest load 一致。

外部包可通过 `rpu_backend.plugins` entry-point group 注册同一契约，不改 in-tree 源码。
Plugin 不能替换 built-in 或已注册 architecture。见
[API 参考](api_reference.zh.md#adapter-registry-与-plugin)。

## 7. Graph 与 cache 不变式

生产顶层 forward 通常在 `GraphCache.capture(signature)` 中运行。重复 signature 应证明
一次 BUILD 后稳定 REPLAY：replay counter 增长、cache size 固定、无第二次 BUILD，且
`cache_invariant_ok()` 为 true。

Bounded one-shot 是变 shape 路径的显式例外：必须录制并执行 non-trivial Graph 一次，
返回 passthrough，且 retained `GraphCache` size 不变。

所有影响 output 的语义输入应进入 signature，或在稳定契约点由 mutable transfer 更新。
同 signature 不同 value 用例必须与 uncaptured path 一致。

## 8. 验证门禁

先跑无板卡检查：unknown architecture 和 unsupported profile 在 weight load/transform 前
拒绝；registry/plugin discovery 确定；execution config 拒绝非法或超范围值；cache sizing
和 output shape 符合参考；源码 build 和 package import 干净完成。

随后在 RPU 板卡跑精确 release 配置：

- 按[验证策略](validation_policy.zh.md)运行硬语义、同 dtype 一致性、CPU FP32 anchor
  与任务质量门禁；
- 验证 `RPU_WARMUP=0/1/3` 默认输出 bit-identical；若精确配置有已提交的有界
  repeatability 合同，则按该合同验证；
- 证明适用 Graph 生命周期；
- 多输入，vision 时多图，捕获 reused output storage；
- 最大 sequence、batch、image 和 cache 范围；
- 用精确 checkpoint、TOML、Launch 和算子资产，在新进程运行公开 E2E example。

缺必需 operation、memory plan、profile asset 或硬语义门禁时标为 Blocked。代表性任务
证据不完整可以保持 Experimental/Source-only，但不能晋级 Supported/Limited。小输入
通过不能扩大模型表。

把 checkpoint revision/hash、配置 hash、软件版本、算子资产 hash、输入范围、精度和
结果写入 release evidence，之后才可修改[模型支持](model_support.zh.md)。
