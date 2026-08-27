# 新增 decoder 模型：逐步指南

简体中文 | [English](new_model_step_by_step.md)

这是 decoder-only Hugging Face CausalLM 的最短支持路径，前提是能力评审 outcome 为
**Adapter-only**。它复用共享 causal decoder，不定义新的 fused subsystem。

先完成[模型移植能力评审](model_porting_capability_review.zh.md)。目标数学不匹配现有
`llama` 或 `qwen3` decoder 分支时，应停止本流程并使用[模型移植](model_porting.zh.md)。

## 1. 固定精确配置和数值参考

记录 checkpoint revision、配置 hash、精度、输入范围和期望输出契约。固定精确同 dtype
参考用于实现一致性，另用 CPU FP32 模型作为高精度 anchor。固定参考 backend、依赖
版本、累加与融合设置，调用 `eval()`、关闭 gradient，并保存下列 reference 输入输出：

- 多 token prefill；
- 至少一个 teacher-forced decode step；
- 最大允许 cache 长度；
- shape 相同的两个不同输入。

CPU 参考实例保持独立。RPU 权重转换是原地操作，不能安全逆转。

**Done：** 不使用 RhinoForge 时，固定输入可重复产生两个 reference 契约。

## 2. 证明共享 decoder 数学完全匹配

把目标 forward 与 Qwen3、Llama adapter 对比，明确验证：

- norm 位置与 epsilon；
- 是否存在 Q/K normalization；
- attention head、KV head、head dimension、mask 与 cache 行为；
- rotary position 约定；
- projection bias 与 tied weight 行为；
- MLP projection 与 activation；
- logits 与 cache 返回语义。

只有 Q/K-normalized 路径匹配时用 `DECODER_ARCH = "qwen3"`；只有非 Q/K-normalized
路径匹配时用 `"llama"`。相似类名不是证据。

**Done：** 目标每个 operation 都无遗漏、无近似地映射到选择的 decoder 分支。

## 3. 复制并填写最小 adapter

```bash
cp python/rpu_backend/adapters/_template/minimal_causal_lm.py \
   python/rpu_backend/adapters/<model_name>.py
```

填写模板配置：

- `HF_ARCH` 精确等于 `config.architectures[0]`；
- `DECODER_ARCH` 是步骤 2 证明的分支；
- `SUPPORTED_PROFILES` 只含允许的五字段 geometry tuple；
- `RMSNORM_CLASS`、`ROTARY_CLASS` 来自已安装 upstream 实现；
- `SKIP_LINEAR_NAMES` 使用 leaf child name，只列由另一个权威路径转换的专用 Linear，
  并在每层匹配同一 child name；
- `_CHUNK_ENVELOPE` 只包含实际验证安全的配置行，不保留模板示例。

五字段 tuple 不是完整模型契约。扩展 `preflight()`，在加载/转换权重前拒绝所有会改变
数学或 layout 的步骤 2 字段，如 norm epsilon、rotary 约定、bias、activation、tied
weight 设置。

不要新增 base class、factory 或第二个权重 walker。模板已经提供 geometry preflight、
一次转换、模型 ownership、runtime 安装和失败清理；新 adapter 只补齐精确配置 guard。

**Done：** 模块导入除一次 adapter registration 外无副作用；任一不支持 geometry 或
数学配置都在 `preflight()` 失败。

## 4. 连接一个 registry route

In-tree adapter 在 `python/rpu_backend/adapters/_manifest.py` 添加一项，并把
architecture 加入 `RPUModelForCausalLM._SUPPORTED_BUILTIN_ARCHITECTURES`。模块
`register()` 与 manifest 必须使用相同 architecture 和 class。

外部分发包不改 manifest，按 [API 参考](api_reference.zh.md)注册一个受信任的
`rpu_backend.plugins` entry point。Plugin 不能覆盖 built-in binding。

添加一个只读取配置的无板卡检查，证明：

1. 精确配置解析到新 adapter；
2. 相邻尺寸，以及 geometry 相同但步骤 2 某数学字段变化的配置，都在完整 checkpoint
   加载前失败；
3. 非法 `rpu_execution` 在模型 mutation 前被拒绝。

**Done：** `list_adapters()` 只报告一个确定 binding，unsupported profile 不能进入
weight loader。

## 5. 保持安装顺序

保留模板 `to_rpu()` 顺序：

1. 验证硬件属性并声明 live instance；
2. 标记不可逆转换已开始；
3. 安装幂等 class swap；
4. 调用一次 `swizzle_model_inplace(..., skip_names=SKIP_LINEAR_NAMES)`；
5. 把转换后模型移到 `rpu`；
6. 用冷 `rpu_execution` 安装共享 causal decoder；
7. 验证安装后的模型；
8. 只有此前全部成功才标记 ready。

同一实例转换失败后不得重试，应重新加载干净 CPU 模型。不要添加进程全局 chunk setter；
公共 loader 通过 `rpu_execution` 绑定 chunk/padding。

**Done：** 完整安装后的第二次 `.to("rpu")` 只是安全 no-op；部分安装和把已转换权重
移回 CPU 会清晰失败。

## 6. 验证 Graph、cache 与 storage ownership

共享 decoder 已拥有 per-model `GraphCache`，不要再加一层 capture。确认 signature
覆盖所有会改变 emitted forward 的 shape 和语义模式。未进入 signature 的同 shape
语义输入必须通过现有 mutable-transfer 契约刷新。

重复相同允许 signature 并检查：首次 BUILD，后续 REPLAY；cache size 固定、replay
count 增长；`cache_invariant_ok()` 保持 true；不同语义值与 uncaptured 路径逐字节一致；
调用方保留的输出 storage 独立。

**Done：** warmup 不改变值、不重复 BUILD 同一 signature，也不覆盖之前返回的 output。

## 7. 添加一个公开 example 及检查

TOML schema 已覆盖目标时复用 `examples/causal_lm.py`。只为精确配置新增
`examples/configs/` 文件；只有输入/输出契约确实不同才新建 script。

Example 必须在不导入 native backend 时通过配置校验：

```bash
python examples/causal_lm.py --config examples/configs/<profile>.toml --check-config
```

在[模型资产](model_assets.zh.md)更新来源、revision、destination 和 hash；只有入口表
变化时才更新[模型运行与性能分析](model_testing.zh.md)。Release-only 环境值不得放入
TOML；以[运行时配置](runtime_config.zh.md)为 allowlist。

运行现有无板卡检查：

```bash
python -m pytest \
  tests/test_public_runtime.py \
  tests/test_configuration.py \
  tests/test_kernel_contracts.py
```

**Done：** 无板卡时配置、import、registry 和现有公开契约全部通过。

## 8. 运行配置门禁并晋级

在新进程中使用 release 匹配依赖和资产构建，然后在 RPU 板卡运行精确 TOML。证据包括：

- [验证策略](validation_policy.zh.md)固定的硬语义、同 dtype 实现一致性、CPU FP32
  anchor 和代表性任务门禁；
- `RPU_WARMUP=0`、`1`、`3` 默认输出 bit-identical；精确配置有已提交的有界
  repeatability 合同时按该合同验证；
- 一次 BUILD 后稳定 REPLAY；
- 同 shape 不同 value 覆盖；
- 最大 batch、sequence、cache 范围；
- 新进程从固定 checkpoint 完成公开 example。

把源码、依赖、checkpoint、配置、资产、输入和输出 hash 一起记录。缺 operation、不安全
resource plan 或硬语义失败会阻塞配置。一致性或任务证据不完整时保持 Experimental 或
Source-only；不能仅靠缩小范围晋级，除非这个更窄配置通过全部适用门禁。

**Done：** 只有所有门禁通过后，才可把精确配置加入
[模型支持](model_support.zh.md)可运行表。
