# Qwen3.5 配置

本目录统一提供 78 份 `[example/input/run]` 配置：Dense Text 48、Dense VL 19、35B-A3B MoE Text/VL 8，以及 legacy 27B Text 3。所有模板按组件、尺寸与精度组织在 `text/`、`vl/` 和 `legacy/` 下。

| 目录 | 配置 |
|---|---|
| `text/{0_8b,2b,4b,9b}` | FP16、W8、W4；FP16 TP4/6；显式 P64/D4096、P4096/D4、P4096/D4096 |
| `vl/{0_8b,2b,4b}` | FP16 图文及长输入；Text W8/W4 + 浮点 Vision 的短配置 |
| `vl/9b` | FP16 图文及长输入；八核 uniform Text W8 + 浮点 Vision 的短配置 |
| `text/35b_a3b`、`vl/35b_a3b` | 精确 mixed-E4M3，默认核数、TP4/6 和长输入 |
| `legacy/27b` | 预 v1 量化元数据的精确 text-only W8 bundle，默认核数和 TP4/6 |

```sh
python examples/run_model.py --config examples/configs/qwen3_5/text/2b/w8a16.toml --check-config
python examples/run_model.py --config examples/configs/qwen3_5/text/2b/w8a16.toml
python examples/run_model.py --config examples/configs/qwen3_5/vl/2b/w8a16.toml --check-config
python examples/run_model.py --config examples/configs/qwen3_5/vl/2b/w8a16.toml
```

别名定位到 `RPU_MODEL_CACHE` 中的完整 checkpoint。W8/W4 保留量化元数据和 projection scope，不从 dtype 猜测；35B-A3B mixed-E4M3 与 Dense W8/W4 是不同格式。模板保留显式 opt-in，实际准入由加载器检查，文件存在不代表模型质量验证。

## Dense 图文量化

0.8B、2B、4B 的 `vl/<size>/w8a16.toml` 与 `w4a16.toml`，以及 9B 的 `vl/9b/w8a16.toml`，使用完整图文 checkpoint。量化范围是 **Text 主干的声明投影**，不表示 Vision 也使用 W8/W4。Vision 保持未量化的 FP16 安装及既有浮点辅助路径；embedding、LM head、norm、GDN conv1d 不在转换量化范围内，activation 与 KV cache 保持浮点。

默认离线转换覆盖每层 MLP 的 gate/up/down，full-attention 层的 Q/K/V/O，以及 GDN 层的 `in_proj_qkv`、`in_proj_z`、`in_proj_b`、`in_proj_a`、`out_proj`。W8 使用逐输出通道 INT8 和 FP16 scale；W4 使用沿 K 方向的分组整数与 FP16 scale，下面示例为 G32。若 checkpoint 使用逐投影混合声明，加载器按 `quant_config` 执行，不能仅凭文件名或 `example.dtype` 推断所有投影同精度。

```sh
python -m rpu_backend.quant.convert_qwen3_5 \
  --src /path/to/Qwen3.5-2B --dst /path/to/Qwen3.5-2B-w8a16 \
  --method w8a16
python -m rpu_backend.quant.convert_qwen3_5 \
  --src /path/to/Qwen3.5-2B --dst /path/to/Qwen3.5-2B-w4a16 \
  --method w4a16_pgrp --group-size 32
```

源必须是未量化的对应 Dense checkpoint，输出目录必须尚不存在。转换保留完整 Vision、processor 和 tokenizer 文件；运行时将 `input.checkpoint` 指向输出，或放入 `example.registry_alias` 对应的本地缓存目录。请使用公开 `RPUModelForConditionalGeneration` 加载入口；专用 loader 保留整数权重和 scale，普通 HF 加载造成的隐式浮点转换会被拒绝。

这些配置同时显式设置 `QWEN3_5_QUANT_ALLOW_UNCERTIFIED="1"` 与 `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED="1"`，分别确认量化未认证与 Vision 数值受控状态；两个 gate 不可互相替代。默认八核、预热 1 次、正式 3 次、`decode_steps=32` 且不因 EOS 提前停止，最多生成首 token 加 32 次 decode 的输出。配置文件本身不构成模型质量或性能认证。

9B W8 图文仅接受八核、现代 uniform W8 Text 投影与浮点 Vision；所有 Text 投影须为 INT8，并有逐输出通道的正有限 FP16 scale，非量化权重须保持 FP16。它沿用图文的资源和 Graph 生命周期，不把纯文本的执行方式套到图文；混合投影、低核和 W4 图文不属于这一入口。35B-A3B MoE 继续使用自己的 mixed-E4M3 格式，不能套用 Dense 转换命令或改成 W8。

`model.num_cores` 是冷态选择，TP4/6 只覆盖相应模板的精确结构和精度。2B `fp16_p4096_d4_c640.toml` 保留 C640，其他 `auto` 由 planner 选择。Text、VL、MoE、legacy 27B 各自保持已有 Graph 和内存合同。

`input.prefill_tokens` 与 `decode_steps` 分别为构造后的 token 数和 prefill 首 token 后的 decode 调用数。固定长度模板保留 EOS、预热 decode 与计时模式；MoE 的固定输入策略也应按配置执行。`assets/example.ppm` 是公开示意图片，不是历史测量输入。长 P/D 配置不会自动触发全矩阵板测。

`legacy/27b` 的 `qwen3_8-27b` 是历史 alias/profile 名，其几何属于 Qwen3.5 27B；它要求专用的预 v1 W8 主投影、浮点 GDN、text-only 量化声明。普通浮点 27B 或现代 uniform W8 不能冒充该格式。本目录不据该 alias 声明普通 27B FP16/VL 支持。参见 [量化说明](../../../docs/quantization.md)。
