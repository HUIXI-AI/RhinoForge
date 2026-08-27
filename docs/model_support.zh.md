# 模型支持

[English](model_support.md) | 简体中文

模型支持针对精确 profile 定义，而不是只按模型家族定义。checkpoint revision、
输入范围、精度、执行配置和 runtime 资产组合共同构成一个 profile。状态不会在
不同模型尺寸或量化格式之间自动继承。

> **v1.0.0 发布：** 下列状态是当前发布声明，仅适用于
> [`release/public-models-v1.0.0.json`](../release/public-models-v1.0.0.json)
> 中的不可变公开身份及配套 v1.0.0 runtime set。生成的派生 checkpoint
> 或相邻配置不继承状态。

## 状态定义

| 状态 | 含义 |
|---|---|
| Supported | 精确公开 profile 通过 checkpoint/输入、数值、warmup、Graph 生命周期和端到端验证 |
| Limited | 通过与 Supported 相同的质量门，但只承诺明确命名的更窄范围；相邻 profile 不继承结论 |
| Experimental | 硬语义、必要 finite-output、声明的重复性和安全拒绝门通过，但实现对齐或代表性任务证据仍可能不完整 |
| Component-only | 只提供指定组件 API 和组件验证，不声明完整产品模型 |
| Source-only | 包含源码，但不承诺可运行支持；已知不支持的配置会在模型修改前拒绝 |
| Unsupported | 不注册可运行 profile，也没有专用公开 runtime；适用时共享 loader 会提前拒绝 |

独立的硬门、实现对齐、任务质量和 runtime 生命周期规则见
[模型验证策略](validation_policy.zh.md)。`Limited` 不代表可以降低数值质量，
`Experimental` 也不等于模型一定存在数值错误。

## Supported 与 Limited profile

| 模型/profile | 公开入口 | 状态 | 精确范围 |
|---|---|---|---|
| Qwen3 0.6B / 1.7B / 4B / 8B | `rpu_backend.RPUModelForCausalLM` | Supported | FP16 推理；batch、prefill 和 decode 范围由发布版本固定 |
| Llama-3.2-1B | `rpu_backend.RPUModelForCausalLM` | Supported | 因果语言模型推理 |
| Qwen3.5 text 2B / 9B | `rpu_backend.adapters.qwen3_5.Qwen3_5Adapter` 与 `rpu_backend.api.Qwen3_5Cache` | Supported | 仅 dense FP16 文本；不支持 `torch.compile` |
| Qwen3-VL 2B | `rpu_backend.api.RPUModelForConditionalGeneration` | Supported | 图像和文本；首发范围不包含视频 |
| Qwen3-VL 4B | `rpu_backend.api.RPUModelForConditionalGeneration` | Supported | 仅限已验证的单图、多图、prefill、decode 和 chunk 范围 |
| Qwen3-VL 8B | `rpu_backend.api.RPUModelForConditionalGeneration` | Limited | 精确 dense-FP16 padded profile：4/4 numeric/task/Graph 已通过，包含 dual-full；base 5/5、long-text 9/9 token 精确一致；相邻长度、视频和量化路径不继承 |
| Pi0.5 Libero FP16 | `rpu_backend.api.Pi05Policy` | Supported | 精确 Libero FP16 profile：9/9 numeric/task 与 27/27 execution-plan record 已通过；独立 fresh-process Graph 生命周期门也已通过 |
| Hy-Embodied-0.5-VLA | `rpu_backend.api.HyEmbodiedPolicy` | Limited | 仅精确 FP16/W16 输入 profile；数值推理支持不代表机器人就绪认证 |
| DINOv3 ViT-B | `rpu_backend.adapters.dinov3.DINOv3Adapter` | Supported | 仅 ViT-B |

## Experimental、Component-only 与 Source-only profile

| 模型/profile | 公开入口 | 状态 | 精确范围 |
|---|---|---|---|
| Qwen3 W8A16（0.6B / 1.7B / 4B / 8B / 14B） | `rpu_backend.RPUModelForCausalLM` | Source-only | 包含 converter 和 loader 源码；v1.0.0 未绑定公开不可变派生 checkpoint 身份或 hash，alias 仅解析本地输出 |
| Qwen3 32B | `rpu_backend.RPUModelForCausalLM` | Source-only | 已知不支持；preflight 保持 fail-fast |
| Qwen3.5 text 0.8B / 4B | `rpu_backend.adapters.qwen3_5.Qwen3_5Adapter` 与 `rpu_backend.api.Qwen3_5Cache` | Experimental | 本次 192-token 验证的 last-prefill full-vocabulary relative-L2 门超过 `0.01`；这些 profile 属于 numeric-blocked，仅供受控评估 |
| Qwen3.5 Vision 2B / 4B | `rpu_backend.adapters.qwen3_5.Qwen3_5Adapter` | Experimental | 两个尺寸的官方真实图 standalone 门均失败；2B 真实图端到端也失败，4B 真实图端到端尚未运行。默认 fail-closed；精确 `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1` 仅允许受控评估 |
| Qwen3-VL 32B W8A16 | `rpu_backend.api.RPUModelForConditionalGeneration` | Source-only | retained Graph replay 已知从第三个 decode step 起产生全零 logits。默认保持 fail-closed；显式 opt-in 仅用于受控诊断，不代表通过 Graph/生命周期硬门 |
| Pi0.5 Libero W8A16 / W4A16 | `rpu_backend.api.Pi05Policy` | Source-only | 包含 converter 和 runtime 源码；v1.0.0 未绑定公开不可变派生 checkpoint 身份或 hash |
| Pi0.5 reduced-step / closed-loop | `rpu_backend.api.Pi05Policy` | Source-only | 仅保留策略源码；验证仍待完成，这些执行 profile 不继承精确 FP16 或量化路径的证据与状态 |
| Wall-OSS-0.5 | `rpu_backend.api.WallOssPolicy` | Source-only | 仅公开 `wall-x` 接入；固定的公开 checkpoint 尚未完成发布验证门禁 |
| Wall-OSS 量化配置 | `rpu_backend.api.WallOssPolicy` | Source-only | 包含公开 converter；每个派生资产仍需 converter provenance、输出身份和重新验证 |
| SigLIP | `rpu_backend.adapters.siglip.patch_siglip_model_for_rpu_all_layers_once` | Component-only | 仅视觉编码器组件 |
| GR00T-N1.7-3B | `rpu_backend.adapters.gr00t.build_gr00t_vla` | Source-only | 公开 loader、checkpoint、TOML 和端到端资产流程尚未完整 |
| Gemma4-E4B text | `rpu_backend.adapters.gemma4.Gemma4Adapter` | Source-only | 依赖兼容已解决；checkpoint、runtime 资产和精确 profile 验证仍待完成 |
| RhinoVLA | `rpu_backend.api.RhinoVLAPolicy` | Source-only | 集成候选；checkpoint 组成、预处理和 live-input 合同由独立模型仓库负责 |
| LingBot-VLA-V2 | `rpu_backend.api.Lingbot2Policy` | Source-only | 公开上游接入；固定的公开 checkpoint 尚未完成发布验证门禁 |
| Galaxea G0.5 continuous | `rpu_backend.adapters.g05.patch_g05_policy_for_rpu` | Source-only | 仅接入公开 `G05PolicyQwen35` continuous policy；其他上游 policy 模式在权重变换前拒绝 |
| InternVLA-N1 + NavDP | `rpu_backend.adapters.internvla_n1` 与 `rpu_backend.adapters.navdp` | Source-only | 公开上游接入；精确公开资产闭包和验证仍待完成 |
| Hy-Embodied W8/W4 | `rpu_backend.api.HyEmbodiedPolicy` | Source-only | 包含 runtime 转换源码；v1.0.0 未绑定公开不可变派生 checkpoint 身份或 hash，FP16/W16 证据不转移 |

## 分模型验证合同

验证分为三层独立门禁：硬语义与 Graph/runtime 生命周期、同语义实现对齐，以及
针对实际消费输出的任务或端到端质量。较高的聚合 cosine 不能豁免硬门失败，
仓库也不会给所有模型套用一个统一 cosine 阈值。量化 profile 先与相同量化语义
的 reference 比较，再与 FP16 或 FP32 task-quality anchor 比较。

v1.0.0 release record 已将每个声明 profile 绑定到 reference identity、输入、
指标和阈值。当前合同如下：

| Profile | 硬语义与生命周期 | 实现对齐 | 任务或发布质量证据 |
|---|---|---|---|
| Qwen3 FP16 | 有效 position、cache slot、warmup `0/1/3`、长上下文边界，以及 BUILD 后稳定 REPLAY | CPU-FP32 与 RPU 逐位置 cosine `>=0.99995`；所有非精确 greedy near-tie 都必须报告 | 各命名尺寸和范围的固定 perplexity/任务准确率与固定 prompt 生成 |
| Qwen3-VL 2B / 4B | 图文顺序、修正后的 DeepStack 路由、单图/多图输入，以及稳定 Vision/prefill/decode Graph 生命周期 | 消费的文本行在单图场景 mean cosine `>0.99`、双图场景 `>0.98`，final argmax 和有序 top-5 精确一致；image-token 指标单独作为诊断 | 只覆盖命名 prompt、decode 和 chunk 范围的固定图像条件与多图任务 |
| Qwen3-VL 8B dense FP16 | 与 2B/4B 相同的语义和生命周期门，另加精确 padded profile 的 shape 与 physical-padding preflight；4/4 命名用例均通过 Graph 生命周期门 | 4/4 命名用例均通过校准后的 numeric 与 task 门，包含 dual-full；base 5/5、long-text 9/9 token 精确一致 | Limited 只覆盖该精确 padded profile；相邻长度、视频和量化路径不继承 |
| Qwen3.5 text 2B / 9B | prefill 使用声明的 bounded one-shot，decode 使用稳定 REPLAY | text/decode cosine `>=0.999`，last-prefill full-vocabulary relative L2 `<=0.01`；接受的相同/相邻 FP16-bin near-tie 标记 `greedy_exact=false` | 指定尺寸的固定 dense-FP16 文本用例已通过 |
| Qwen3.5 text 0.8B / 4B | 执行相同的 bounded-one-shot prefill 和稳定 REPLAY decode 生命周期 | 本次 192-token 验证的 last-prefill full-vocabulary relative L2 超过 `0.01` | Numeric-blocked；受控评估不构成发布支持 |
| Qwen3.5 Vision 2B / 4B | 默认在 Vision handle 创建或权重转换前 fail closed；受控评估使用有界 retained cache 且仅接受图像 | 声明的 row-mean cosine `>=0.999`、loose minimum `>=0.90`、maximum relative error `<0.40` 门未被任一尺寸的官方真实图满足 | 尚无认证的生产安全正常图像范围；受控结果不构成图像或视频支持 |
| Pi0.5 Libero FP16 | 精确 action shape/dtype、finite output、多相机与长度行为，以及 27/27 action/prefill/vision execution-plan record；独立 fresh process 证明 READY 后稳定 Graph replay，且无 recapture 或 invariant 失败 | 9/9 same-noise numeric 用例均通过源码 MSE 上限 `0.0072` | 精确 9/9 action/task matrix 与独立 Graph cell 均通过；量化和 reduced-step profile 不继承 |
| Hy-Embodied FP16/W16 | 精确 profile admission、finite output、四次调用 repeatability spread `<=1e-4`、稳定 Graph 生命周期和 A/B/A 输入刷新 | final action cosine `>=0.999`、MSE `<=0.01`；命名中间阶段分别使用冻结的 `0.99995` 或 `0.999` 门 | checkpoint 自有离线 action 证据；机器人安全和 W8/W4 不在结论内 |

reference identity、near-zero row、量化 reference 顺序、task metric retention
和状态决策见[模型验证策略](validation_policy.zh.md)。未运行的门禁保持 pending，
不会从其他模型尺寸或精度推断为通过。

## 不支持和排除的 profile

以下是排除列表，不是可运行模型矩阵。这些 profile 不会出现在可运行 alias 和
example 中：

| Profile | 状态 | 公开入口 |
|---|---|---|
| Gemma2 与 PaliGemma2 | Unsupported | 无 |
| DINOv3 ViT-S 与 ViT-L | Unsupported | 无；DINOv3 ViT-B 仍在范围内 |
| LingBot-VLA v1 | Unsupported | 无；LingBot-VLA-V2 仅保留上述范围 |
| 训练、微调、数据集分发和 checkpoint 再分发 | Unsupported | 无 |

## Registry alias 分类

registry 只把稳定名称映射到本地 cache 目录。每个保留 alias 仍必须归入一个
release profile，避免把路径 alias 误解为额外支持声明。

| Registry alias | 家族/profile | 状态 | 分类 |
|---|---|---|---|
| `qwen3-0.6b`、`qwen3-1.7b`、`qwen3-4b`、`qwen3-8b` | Qwen3 FP16 | Supported | 对应精确尺寸的资产 alias |
| `qwen3-14b-w8a16-lmhead-int8` | Qwen3 14B W8A16 | Source-only | 本地转换输出 alias；v1.0.0 未绑定公开不可变派生 checkpoint 身份或 hash |
| `qwen3_5-2b`、`qwen3_5-9b` | Qwen3.5 text FP16 | Supported | 对应精确尺寸的资产 alias |
| `qwen3_5-0.8b`、`qwen3_5-4b` | Qwen3.5 text FP16 | Experimental | 对应精确 numeric-blocked 尺寸的资产 alias；alias 解析不代表发布支持 |
| `qwen3-vl-2b` | Qwen3-VL 2B Instruct | Supported | 资产 alias |
| `qwen3-vl-4b` | Qwen3-VL 4B Instruct | Supported | 仅已验证图像、prefill、decode 和 chunk 范围的资产 alias |
| `qwen3-vl-8b` | Qwen3-VL 8B Instruct | Limited | 精确 dense-FP16 padded profile 的资产 alias；4/4 numeric/task/Graph 包含 dual-full；相邻长度、视频和量化路径不继承 |
| `llama-3.2-1b` | Llama-3.2-1B | Supported | 资产 alias |
| `pi05-libero-finetuned` | Pi0.5 Libero FP16 | Supported | 资产 alias |
| `pi05-libero-finetuned-w8a16-vlm-expert` | Pi0.5 Libero W8A16 | Source-only | 本地转换输出 alias；未绑定公开不可变派生 checkpoint 身份或 hash |
| `pi05-libero-finetuned-w4real-kvint8` | Pi0.5 Libero W4A16 | Source-only | 本地转换输出 alias；未绑定公开不可变派生 checkpoint 身份或 hash |
| `pi05-base` | Pi0.5 base | Source-only | 没有独立任务 profile 支持声明的资产 alias |
| `wall-oss-0.5`、`wall-oss-0.5-w8a16`、`wall-oss-0.5-w4a16`、`wall-oss-0.5-w4a16-pgrp` | Wall-OSS-0.5 | Source-only | 公开源码/转换路径；alias 不建立派生资产身份或支持声明 |
| `dinov3-vit-b` | DINOv3 ViT-B | Supported | 资产 alias；ViT-S/L 保持缺省 |
| `hy-embodied-0.5-vla-umi` | Hy-Embodied-0.5-VLA | Limited | 公开 FP16/W16 资产 alias；runtime 派生 W8/W4 路径仍为 Source-only |
| `g05-base` | Galaxea G0.5 | Source-only | 公开上游资产 alias；访问和非商业条款仍适用 |
| `lingbot-vla-v2-6b` | LingBot-VLA-V2 | Source-only | 公开上游资产 alias |
| `internvla-n1-navdp` | InternVLA-N1 + NavDP | Source-only | 公开上游资产 alias |
| `gr00t-n1d7-3b` | GR00T-N1.7-3B | Source-only | 资产 alias |
| `gemma4-e4b` | Gemma4-E4B text | Source-only | 资产 alias |

原有 `qwen3-0.6b-instruct` 和 `qwen3-4b-instruct` 本地 alias 不发布，因为它们
没有独立固定的公开资产记录。未来只有在提供精确 source、revision、hash 和
profile 分类后才能加入。

共享 loader 如果识别到排除配置，必须在加载或修改模型权重前拒绝。

## 通用 runtime 限制

- RhinoForge 是推理后端，不支持训练和微调。
- FP16 是主要执行 dtype；量化支持按每个精确 profile 单独说明。
- registry alias 只解析 checkpoint 路径，不构成支持声明。
- 输入 shape、batch、context、图像数量、warmup 和 Graph 行为都是 release
  profile 的一部分；未命名的值不继承状态。
- VLA 数值输出验证不认证坐标系、执行器限制、安全策略或机器人就绪状态。

checkpoint 与配置记录见[模型资产](model_assets.zh.md)，源码构建流程见
[入门指南](getting_started.zh.md)。
