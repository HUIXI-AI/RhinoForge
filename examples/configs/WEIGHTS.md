# 权重准备

[模型目录](README.md) · [通用参数](PARAMETERS.md) · [命名规则](NAMING.md)

每份 TOML 在 `[model]` 中用 `registry_alias` 或 `checkpoint` 等字段指向权重。按下面四步准备：

1. 设置缓存根目录 `RPU_MODEL_CACHE`（未设置时默认为 `~/.cache/rhinoforge/models`）。登记名解析为 `$RPU_MODEL_CACHE/<目录>`，映射见下表，也可在 [model_registry.py](../../python/rpu_backend/model_registry.py) 中查看。
2. 按表中的仓库和 revision 下载上游权重，放到表中的目录。
3. “获取方式”为“转换”的资产，用对应小节的命令从上游权重生成，输出到表中的目录。
4. 准备输入文件（batch、observation、fixture 等），见各模型页的“输入文件准备”。

表中的仓库、revision 和许可取自本项目资产库的下载与转换记录。许可由各模型发布方规定，使用前请自行确认；部分仓库需要先在模型页同意许可，再执行 `hf auth login`。

<a id="download"></a>

## 下载与转换的通用写法

```bash
export RPU_MODEL_CACHE=/path/to/models
# Hugging Face：仓库、revision 和目录取自下表
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-0.6B"
# ModelScope：表中标为 ModelScope 的仓库
modelscope download --model QuantTrio/Qwen3.5-2B-AWQ --local_dir "$RPU_MODEL_CACHE/Qwen3.5-2B-AWQ"
```

转换命令在安装了本仓库的环境中运行（`python -m pip install -e . --no-build-isolation`），只用 CPU，不需要 RPU；输出目录须不存在。按命令从上游重新生成的是同一配方。资产库中 Qwen3 W8（14B 为直接交付，无转换记录）、Qwen3-VL-32B W8、Wall-OSS W8 与 W4 G32 副本所记录的转换器，与当前仓库的文件完全相同；其余转换器在生成后有修改或没有记录，重新生成的文件字节可能与现有副本不同。TOML 文件名不决定权重目录，设置登记名也不会自动下载。

<a id="quantization"></a>
<a id="sources"></a>
<a id="offline"></a>
<a id="runtime-quantization"></a>

### 三种获取方式

| 获取方式 | 含义 |
|---|---|
| 下载 | 直接使用上游发布的权重（浮点或已量化的包）；W8/W4 配置若指向浮点目录，则在加载时按 TOML 量化，不需要单独转换 |
| 转换 | 从上游浮点权重离线生成量化或导出产物，命令见对应小节 |
| 配套交付 | 没有公开来源的客户或内部资产（checkpoint、源码、fixture），需向资产提供方获取；格式要求见模型页 |

精度、量化和累加的含义见[量化说明](../../docs/quantization.zh.md)与[累加模式](../../docs/runtime_config.zh.md)。`W8A16` 表示所选投影的权重为 8 位、激活为 16 位；`G32/G128` 是量化分组大小。

<a id="cmd-qwen3"></a>

## Qwen3

| 登记名 | 目录（`$RPU_MODEL_CACHE/` 下） | 获取方式 | 来源与版本 |
|---|---|---|---|
| `qwen3-0.6b` | `Qwen3-0.6B` | 下载 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) `c1899de289a04d12100db370d81485cdf75e47ca`；许可 Apache-2.0 |
| `qwen3-1.7b` | `Qwen3-1.7B` | 下载 | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；许可 Apache-2.0 |
| `qwen3-4b` | `Qwen3-4B` | 下载 | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) `1cfa9a7208912126459214e8b04321603b3df60c`；许可 Apache-2.0 |
| `qwen3-8b` | `Qwen3-8B` | 下载 | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) `b968826d9c46dd6066d109eabc6255188de91218`；许可 Apache-2.0 |
| `qwen3-14b` | `Qwen3-14B` | 下载 | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) `40c069824f4251a91eefaf281ebe4c544efd3e18`；许可 Apache-2.0 |
| `qwen3-32b` | `Qwen3-32B` | 下载 | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) `9216db5781bf21249d130ec9da846c4624c16137`；许可 Apache-2.0 |
| `qwen3-4b-instruct` | `qwen3-4b-instruct` | 下载 | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) `cdbee75f17c01a7cc42f958dc650907174af0554`；许可 Apache-2.0 |
| `qwen3-0.6b-instruct` | `qwen3-0.6b-instruct` | 配套交付 | 资产库中没有公开来源记录，需向资产维护方获取 |
| `qwen3-0.6b-w8a16-lmhead-int8` | `qwen3_lmhead_int8_ckpts/qwen3-0.6b-w8a16-lmhead-int8` | 转换 | 由 `qwen3-0.6b` 生成，见下方命令 |
| `qwen3-1.7b-w8a16-lmhead-int8` | `qwen3_lmhead_int8_ckpts/qwen3-1.7b-w8a16-lmhead-int8` | 转换 | 由 `qwen3-1.7b` 生成，见下方命令 |
| `qwen3-4b-w8a16-lmhead-int8` | `qwen3_lmhead_int8_ckpts/qwen3-4b-w8a16-lmhead-int8` | 转换 | 由 `qwen3-4b` 生成，见下方命令 |
| `qwen3-8b-w8a16-lmhead-int8` | `qwen3_lmhead_int8_ckpts/qwen3-8b-w8a16-lmhead-int8` | 转换 | 由 `qwen3-8b` 生成，见下方命令 |
| `qwen3-14b-w8a16-lmhead-int8` | `qwen3_lmhead_int8_ckpts/qwen3-14b-w8a16-lmhead-int8` | 转换 | 由 `qwen3-14b` 生成，见下方命令 |
| `qwen3-32b-w8a16-lmhead-int8` | `Qwen3-32B-w8a16` | 转换 | 由 `qwen3-32b` 生成，见下方命令 |

W8A16 配置使用 `*-w8a16-lmhead-int8` 产物（七个 decoder 投影和输出头为逐通道 INT8）。W4A16 配置直接指向浮点目录，在加载时量化，不需要转换。

```bash
python -m rpu_backend.quant.convert_qwen3 --src "$RPU_MODEL_CACHE/Qwen3-0.6B" \
  --dst "$RPU_MODEL_CACHE/qwen3_lmhead_int8_ckpts/qwen3-0.6b-w8a16-lmhead-int8" --quant-lm-head
# 其他尺寸同理；32B 的输出目录为 $RPU_MODEL_CACHE/Qwen3-32B-w8a16
```

<a id="cmd-qwen3-vl"></a>

## Qwen3-VL

| 登记名 | 目录（`$RPU_MODEL_CACHE/` 下） | 获取方式 | 来源与版本 |
|---|---|---|---|
| `qwen3-vl-2b` | `Qwen3-VL-2B-Instruct` | 下载 | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) `89644892e4d85e24eaac8bacfd4f463576704203`；许可 Apache-2.0 |
| `qwen3-vl-4b` | `Qwen3-VL-4B-Instruct` | 下载 | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) `ebb281ec70b05090aa6165b016eac8ec08e71b17`；许可 Apache-2.0 |
| `qwen3-vl-8b` | `Qwen3-VL-8B-Instruct` | 下载 | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；许可 Apache-2.0 |
| `qwen3-vl-2b-text-awq-w4a16` | `Qwen3-VL-2B-Instruct-AWQ` | 下载 | [cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit) `db40a251bdba88fafabf8f3176e7488ed523ab51`；许可 Apache-2.0（模型卡声明） |
| `qwen3-vl-4b-text-awq-w4a16` | `Qwen3-VL-4B-Instruct-AWQ` | 下载 | [cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit) `b6a6f1cc360d21feee7465f8656b3855d61e767a`；许可 Apache-2.0（模型卡声明） |
| `qwen3-vl-2b-runtime-w8a16` | `Qwen3-VL-2B-Instruct-runtime-w8` | 转换 | 由同尺寸上游浮点权重生成（`--bits 8`） |
| `qwen3-vl-4b-runtime-w8a16` | `Qwen3-VL-4B-Instruct-runtime-w8` | 转换 | 由同尺寸上游浮点权重生成（`--bits 8`） |
| `qwen3-vl-8b-runtime-w8a16` | `Qwen3-VL-8B-Instruct-runtime-w8` | 转换 | 由同尺寸上游浮点权重生成（`--bits 8`） |
| `qwen3-vl-32b-runtime-w8a16` | `Qwen3-VL-32B-Instruct-runtime-w8` | 转换 | 由同尺寸上游浮点权重生成（`--bits 8`） |
| `qwen3-vl-2b-runtime-w4a16` | `Qwen3-VL-2B-Instruct-runtime-w4-awq` | 转换 | 由同尺寸 cyankiwi AWQ 包生成（`--bits 4`，保留 AWQ 文本投影） |
| `qwen3-vl-4b-runtime-w4a16` | `Qwen3-VL-4B-Instruct-runtime-w4-awq` | 转换 | 由同尺寸 cyankiwi AWQ 包生成（`--bits 4`，保留 AWQ 文本投影） |
| `qwen3-vl-8b-runtime-w4a16` | `Qwen3-VL-8B-Instruct-runtime-w4-awq` | 转换 | 由同尺寸 cyankiwi AWQ 包生成（`--bits 4`，保留 AWQ 文本投影） |
| `qwen3-vl-32b-runtime-w4a16` | `Qwen3-VL-32B-Instruct-runtime-w4` | 转换 | 由 32B 上游浮点权重生成（`--bits 4`，RTN G32） |
| `qwen3-vl-2b-text-w8a16` | `Qwen3-VL-2B-Instruct-w8a16-text` | 转换 | 由 `qwen3-vl-2b` 生成，只量化文本投影 |
| `qwen3-vl-32b-w8a16` | `Qwen3-VL-32B-Instruct-w8a16-full-20260816` | 转换 | 由 32B 上游浮点权重生成 |

32B 没有直接使用的浮点登记名，先把 [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) `0cfaf48183f594c314753d30a4c4974bc75f3ccb` 下载到任意工作目录（下称 `$QWEN3_VL_32B`）。8B 的 W4 源包为 [cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit) `87196f7771efdd7022a8d8f094ac6e063bf87f5c`，同样下载到工作目录（下称 `$QWEN3_VL_8B_AWQ`）。

```bash
# runtime W8：文本投影与输出头 INT8，Vision 在安装时量化
python -m rpu_backend.quant.convert_qwen3_vl --src "$RPU_MODEL_CACHE/Qwen3-VL-2B-Instruct" \
  --dst "$RPU_MODEL_CACHE/Qwen3-VL-2B-Instruct-runtime-w8" --bits 8
# runtime W4（2B/4B/8B）：输入为同尺寸 AWQ 包，保留 AWQ 文本投影，输出头 RTN W4
python -m rpu_backend.quant.convert_qwen3_vl --src "$RPU_MODEL_CACHE/Qwen3-VL-2B-Instruct-AWQ" \
  --dst "$RPU_MODEL_CACHE/Qwen3-VL-2B-Instruct-runtime-w4-awq" --bits 4
python -m rpu_backend.quant.convert_qwen3_vl --src "$QWEN3_VL_8B_AWQ" \
  --dst "$RPU_MODEL_CACHE/Qwen3-VL-8B-Instruct-runtime-w4-awq" --bits 4
# runtime W8/W4（32B）：输入为浮点权重，W4 为 RTN G32
python -m rpu_backend.quant.convert_qwen3_vl --src "$QWEN3_VL_32B" \
  --dst "$RPU_MODEL_CACHE/Qwen3-VL-32B-Instruct-runtime-w8" --bits 8
python -m rpu_backend.quant.convert_qwen3_vl --src "$QWEN3_VL_32B" \
  --dst "$RPU_MODEL_CACHE/Qwen3-VL-32B-Instruct-runtime-w4" --bits 4
# 2B 仅文本 W8：Vision 和输出头保持浮点
python -m rpu_backend.quant.convert_qwen3 --src "$RPU_MODEL_CACHE/Qwen3-VL-2B-Instruct" \
  --dst "$RPU_MODEL_CACHE/Qwen3-VL-2B-Instruct-w8a16-text" --skip-modules model.visual lm_head
# 32B W8：输出头保持浮点
python -m rpu_backend.quant.convert_qwen3 --src "$QWEN3_VL_32B" \
  --dst "$RPU_MODEL_CACHE/Qwen3-VL-32B-Instruct-w8a16-full-20260816"
```

4B 的 runtime 产物把上面命令中的 `2B` 换成 `4B`；8B 的 runtime W8 用 `$RPU_MODEL_CACHE/Qwen3-VL-8B-Instruct` 作输入。

<a id="cmd-qwen35-dense"></a>
<a id="cmd-qwen35-text-w4"></a>

## Qwen3.5 Dense（0.8B–27B）

| 登记名 | 目录（`$RPU_MODEL_CACHE/` 下） | 获取方式 | 来源与版本 |
|---|---|---|---|
| `qwen3_5-0.8b` | `qwen3_5-0.8b` | 下载 | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) `2fc06364715b967f1860aea9cf38778875588b17`；许可 Apache-2.0 |
| `qwen3_5-2b` | `Qwen3.5-2B` | 下载 | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) `15852e8c16360a2fea060d615a32b45270f8a8fc`；许可 Apache-2.0 |
| `qwen3_5-4b` | `Qwen3.5-4B` | 下载 | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`；许可 Apache-2.0 |
| `qwen3_5-9b` | `Qwen3.5-9B` | 下载 | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) `c202236235762e1c871ad0ccb60c8ee5ba337b9a`；许可 Apache-2.0 |
| `qwen3_5-27b` | `Qwen3.5-27B` | 下载 | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) `fc05daec18b0a78c049392ed2e771dde82bdf654`；许可 Apache-2.0 |
| `qwen3_5-2b-quanttrio-awq-g128` | `Qwen3.5-2B-AWQ` | 下载 | [QuantTrio/Qwen3.5-2B-AWQ](https://modelscope.cn/models/QuantTrio/Qwen3.5-2B-AWQ)（ModelScope）；资产库只记录了文件清单哈希，没有 revision |
| `qwen3_5-4b-quanttrio-awq-g128` | `Qwen3.5-4B-AWQ` | 下载 | [QuantTrio/Qwen3.5-4B-AWQ](https://modelscope.cn/models/QuantTrio/Qwen3.5-4B-AWQ)（ModelScope）；资产库只记录了文件清单哈希，没有 revision |
| `qwen3_5-9b-quanttrio-awq-g128` | `Qwen3.5-9B-AWQ` | 下载 | [QuantTrio/Qwen3.5-9B-AWQ](https://modelscope.cn/models/QuantTrio/Qwen3.5-9B-AWQ)（ModelScope）；资产库只记录了文件清单哈希，没有 revision |
| `qwen3_5-0.8b-w8a16` | `qwen3_5_quant_ckpts/qwen3_5-0.8b-w8a16` | 转换 | `--method w8a16`，由 `qwen3_5-0.8b` 生成 |
| `qwen3_5-2b-w8a16` | `qwen3_5_quant_ckpts/qwen3_5-2b-w8a16` | 转换 | `--method w8a16`，由 `qwen3_5-2b` 生成 |
| `qwen3_5-4b-w8a16` | `qwen3_5_quant_ckpts/qwen3_5-4b-w8a16` | 转换 | `--method w8a16`，由 `qwen3_5-4b` 生成 |
| `qwen3_5-9b-w8a16` | `qwen3_5_quant_ckpts/qwen3_5-9b-w8a16` | 转换 | `--method w8a16`，由 `qwen3_5-9b` 生成 |
| `qwen3_5-27b-w8a16` | `qwen3_5_quant_ckpts/qwen3_5-27b-w8a16` | 转换 | `--method w8a16`，由 `qwen3_5-27b` 生成 |
| `qwen3_5-0.8b-w4a16` | `qwen3_5_quant_ckpts/qwen3_5-0.8b-w4a16` | 转换 | `--method w4a16_pgrp --group-size 32`，由 `qwen3_5-0.8b` 生成 |
| `qwen3_5-2b-w4a16` | `qwen3_5_quant_ckpts/qwen3_5-2b-w4a16` | 转换 | `--method w4a16_pgrp --group-size 32`，由 `qwen3_5-2b` 生成 |
| `qwen3_5-4b-w4a16` | `qwen3_5_quant_ckpts/qwen3_5-4b-w4a16` | 转换 | `--method w4a16_pgrp --group-size 32`，由 `qwen3_5-4b` 生成 |
| `qwen3_5-9b-w4a16` | `qwen3_5_quant_ckpts/qwen3_5-9b-w4a16` | 转换 | `--method w4a16_pgrp --group-size 32`，由 `qwen3_5-9b` 生成 |
| `qwen3_5-27b-w4a16` | `qwen3_5_quant_ckpts/qwen3_5-27b-w4a16` | 转换 | `--method w4a16_pgrp --group-size 32`，由 `qwen3_5-27b` 生成 |
| `qwen3_5-27b-fp16` | `Qwen3.5-27B-FP16` | 转换 | `--method fp16`，由 `qwen3_5-27b` 导出，减少加载时的转换内存 |
| `qwen3_5-2b-w4-mlp-only-g32` | `qwen3_5-2b-w4a16-g32-mlp-only` | 转换 | `--profile w4_mlp_only` |
| `qwen3_5-27b-w4-mlp-only-g32` | `qwen3_5_quant_ckpts/qwen3_5-27b-w4a16-g32-mlp-only` | 转换 | `--profile w4_mlp_only` |
| `qwen3_5-2b-w4-mlp-w8-attn-gdn-g32` | `qwen3_5-2b-w4a16-g32-mlp-w8-attn-gdn` | 转换 | `--profile w4_mlp_w8_attn_gdn` |
| `qwen3_5-27b-w4-mlp-w8-attn-gdn-g32` | `qwen3_5_quant_ckpts/qwen3_5-27b-w4a16-g32-mlp-w8-attn-gdn` | 转换 | `--profile w4_mlp_w8_attn_gdn` |
| `qwen3_5-27b-activation-awq-g128` | `qwen3_5_quant_ckpts/qwen3_5-27b-activation-awq-g128` | 转换 | 激活感知 AWQ G128，需要校准提示词，见下方命令 |

```bash
SRC="$RPU_MODEL_CACHE/Qwen3.5-2B"; OUT="$RPU_MODEL_CACHE/qwen3_5_quant_ckpts"
python -m rpu_backend.quant.convert_qwen3_5 --src "$SRC" --dst "$OUT/qwen3_5-2b-w8a16" --method w8a16
python -m rpu_backend.quant.convert_qwen3_5 --src "$SRC" --dst "$OUT/qwen3_5-2b-w4a16" --method w4a16_pgrp --group-size 32
python -m rpu_backend.quant.convert_qwen3_5 --src "$SRC" --dst "$RPU_MODEL_CACHE/qwen3_5-2b-w4a16-g32-mlp-only" --profile w4_mlp_only
python -m rpu_backend.quant.convert_qwen3_5 --src "$SRC" --dst "$RPU_MODEL_CACHE/qwen3_5-2b-w4a16-g32-mlp-w8-attn-gdn" --profile w4_mlp_w8_attn_gdn
# 27B：先导出 FP16，再按需要生成各量化配方（输出目录见上表）
python -m rpu_backend.quant.convert_qwen3_5 --src "$RPU_MODEL_CACHE/Qwen3.5-27B" --dst "$RPU_MODEL_CACHE/Qwen3.5-27B-FP16" --method fp16
# 27B 激活感知 AWQ：先用校准提示词收集激活（每行一条纯文本、JSON 字符串或 {"text": "..."}），再转换
python -m rpu_backend.quant.convert_qwen3_5_awq calibrate --src "$RPU_MODEL_CACHE/Qwen3.5-27B" \
  --prompts calibration_prompts.txt --output 27b_awq_calibration.safetensors
python -m rpu_backend.quant.convert_qwen3_5_awq convert --src "$RPU_MODEL_CACHE/Qwen3.5-27B" \
  --dst "$RPU_MODEL_CACHE/qwen3_5_quant_ckpts/qwen3_5-27b-activation-awq-g128" --calibration 27b_awq_calibration.safetensors
```

0.8B/4B/9B 的命令把尺寸和目录换掉即可。`--method` 和 `--profile` 二选一；量化范围（哪些投影是 W4、哪些是 W8）见 [Qwen3.5 配置页](qwen3_5/README.md#quantization)。

<a id="cmd-qwen35-moe"></a>

## Qwen3.5 35B-A3B MoE

| 登记名 | 目录（`$RPU_MODEL_CACHE/` 下） | 获取方式 | 来源与版本 |
|---|---|---|---|
| `qwen3_5-35b-a3b-gptq-w4a16` | `Qwen3.5-35B-A3B-GPTQ-Int4` | 下载 | [Qwen/Qwen3.5-35B-A3B-GPTQ-Int4](https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4)（ModelScope） 文件 revision `7baa396cda1da0d759cfdaadd1b7fbd6003738f9`；许可 Apache-2.0 |
| `qwen3_5-35b-a3b-rtn-w4a16` | `Qwen3.5-35B-A3B-RTN-W4A16-G128` | 转换 | 路由专家 RTN W4 G128，由上游浮点权重生成 |
| `qwen3_5-35b-a3b-mixed` | `Qwen3.5-35B-A3B-rpu-fp8-mixed` | 转换 | 路由专家与文本混合层为逐行 FP8 E4M3，由上游浮点权重生成 |

上游浮点权重为 [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) `59d61f3ce65a6d9863b86d2e96597125219dc754`（许可 Apache-2.0），下载到任意工作目录（下称 `$QWEN35_MOE`）。两个转换器都需要大量内存，请在 CPU 主机上运行，并设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python -m rpu_backend.quant.convert_qwen3_5_moe_rtn --src "$QWEN35_MOE" \
  --dst "$RPU_MODEL_CACHE/Qwen3.5-35B-A3B-RTN-W4A16-G128" --threads 8
python -m rpu_backend.quant.convert_qwen3_5_moe_fp8 --src "$QWEN35_MOE" \
  --dst "$RPU_MODEL_CACHE/Qwen3.5-35B-A3B-rpu-fp8-mixed" --threads 8
```

MoE 配置也可以用 `examples/run.py --model <目录>` 指定权重目录，profile 校验不变。配方细节见[量化说明](../../docs/quantization.zh.md)。

## Llama、Gemma4、DINOv3

| 登记名 | 目录（`$RPU_MODEL_CACHE/` 下） | 获取方式 | 来源与版本 |
|---|---|---|---|
| `llama-3.2-1b` | `Llama-3.2-1B` | 下载 | [unsloth/Llama-3.2-1B](https://huggingface.co/unsloth/Llama-3.2-1B) `9535bd9b1d1dea6acafbdc4813b728796aeb28da`；许可 Llama 3.2 社区许可（Meta 原仓库需申请访问，这里使用 unsloth 镜像） |
| `gemma4-e4b` | `gemma-4-e4b-it` | 下载 | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) `ee0ef6023621cff504d758262d4e04895a5af4a2`；许可 Apache-2.0 |
| `dinov3-vit-b` | `dinov3-vitb16-pretrain-lvd1689m` | 下载 | [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) `5931719e67bbdb9737e363e781fb0c67687896bc`；许可 DINOv3 License |

三者都是浮点权重，加载时转为 FP16，不需要转换。

## Qwen3 ASR / TTS

语音配置用 `[model].checkpoint` 填写目录，不使用登记名。仓库与固定 revision、下载命令和 TTS 源码准备见[语音模型的安装与权重](../../docs/qwen3_audio.zh.md#2-下载权重并准备-tts-源码)。量化配方（W8A16、W4A16 G32、NVFP4）在加载时由浮点权重生成，不需要离线转换。

## 视觉动作模型（VLA）

VLA 除了权重，通常还需要配套源码、processor 和预处理过的输入文件。下表只列权重；输入文件的格式和生成方法见各模型页的“输入文件准备”。

<a id="cmd-pi05"></a>

### Pi0.5

| 登记名 | 目录（`$RPU_MODEL_CACHE/` 下） | 获取方式 | 来源与版本 |
|---|---|---|---|
| `pi05-libero-finetuned` | `pi05_libero_finetuned` | 下载 | [lerobot/pi05_libero_finetuned_v044](https://huggingface.co/lerobot/pi05_libero_finetuned_v044) `dbf8a3f794a9c4297b44f40b752712f50073d945`；许可 Gemma 使用条款 |
| `pi05-libero-finetuned-w8a16-vlm-expert` | `pi05_libero_finetuned-w8a16-vlm-expert` | 转换 | VLM 与动作专家投影 INT8，由 `pi05-libero-finetuned` 生成 |
| `pi05-libero-h10` | `pi05_libero_h10` | 下载后改 `config.json` | [lerobot/pi05_libero_base](https://huggingface.co/lerobot/pi05_libero_base) `5fcf06fc42b6717e22ab885fd381a096574b12c7`，权重原样（FP32，`model.safetensors` sha256 `21b8711787c4a75861b02cff6aa81675a3a943d32b435a68262ac4461e476ba4`）；`chunk_size` / `n_action_steps` 由 50 / 10 改为 10 / 10；许可 Gemma 使用条款 |
| `pi05-libero-h10-w8a16-vlm-expert` | `pi05_libero_h10-w8a16-vlm-expert` | 转换 | VLM 与动作专家投影 INT8，由 `pi05-libero-h10` 生成 |

action horizon 是 checkpoint 的属性（`config.json` 的 `chunk_size`），`[input].action_horizon` 只声明配置期望哪一份权重，与权重不符时拒绝运行。H50 用 `pi05-libero-finetuned` 系列（在 `chunk_size=50` 上微调）。H10 用 `lerobot/pi05_libero_base`：它是 openpi `pi05_libero` 的 PyTorch 转换，该模型按 `Pi0Config(pi05=True, action_horizon=10)` 训练，仓库 `config.json` 里的 `chunk_size=50` 是 LeRobot 转换时的默认值，所以下载后改为 10：

```bash
python -c 'import json, sys; p = sys.argv[1]; c = json.load(open(p)); \
assert (c["chunk_size"], c["n_action_steps"]) == (50, 10); \
c.update(chunk_size=10, n_action_steps=10); json.dump(c, open(p, "w"), indent=4)' \
  "$RPU_MODEL_CACHE/pi05_libero_h10/config.json"
```

动作 NVFP4 配置（`w8a16_action_nvfp4`、`w8a16_prefill_w8a8_action_nvfp4`）用 `[model].checkpoint` 填写转换产物的路径，同一 horizon 的两者共用同一份产物：

```bash
python -m rpu_backend.quant.convert_pi05 --src "$RPU_MODEL_CACHE/pi05_libero_finetuned" \
  --dst "$RPU_MODEL_CACHE/pi05_libero_finetuned-w8a16-vlm-expert"
python -m rpu_backend.quant.convert_pi05 --src "$RPU_MODEL_CACHE/pi05_libero_finetuned" \
  --dst /path/to/pi05_action_nvfp4 --action-w4
# H10：源换成 pi05_libero_h10
python -m rpu_backend.quant.convert_pi05 --src "$RPU_MODEL_CACHE/pi05_libero_h10" \
  --dst "$RPU_MODEL_CACHE/pi05_libero_h10-w8a16-vlm-expert"
python -m rpu_backend.quant.convert_pi05 --src "$RPU_MODEL_CACHE/pi05_libero_h10" \
  --dst /path/to/pi05_h10_action_nvfp4 --action-w4
```

生成 batch 还需要 tokenizer `unsloth/gemma-2-2b` 和 LIBERO 数据集，见 [Pi0.5 输入文件准备](pi05/README.md#input-files)。转换选项见 Pi0.5 权重准备。

<a id="cmd-pi05-kuavo"></a>

#### Kuavo（openpi pi05）

`pi05/kuavo/` 配置用 `[model].checkpoint` 填写目录，不使用登记名。公开权重是 [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) `22c30e5b`（许可 Gemma 使用条款），存储为 FP32，三路相机特征均为 `[3,224,224]`，action/state 宽度 32。FP16 配置直接读取这份 FP32 权重，加载时转为 FP16；W8A16 配置先转换：

```bash
python python/rpu_backend/quant/convert_pi05.py --src /path/to/pi05_base --dst /path/to/pi05_base-w8a16
```

每种机器人构型再建一个 checkpoint 视图，把 `config.json` 的 action/state 宽度改为 8（单臂）、16（双臂）或 20（全身）；权重文件以符号链接共享，不复制。模型仍计算 32 维，入口取前 `action_dim` 维，等同于在该构型上微调过的 checkpoint：

```bash
for d in 8 16 20; do
  python examples/runners/pi05/openpi_inputs.py checkpoint-view --src /path/to/pi05_base --dst /path/to/pi05_base_kuavo_a$d --action-dim $d
  python examples/runners/pi05/openpi_inputs.py checkpoint-view --src /path/to/pi05_base-w8a16 --dst /path/to/pi05_base_kuavo_a$d-w8a16 --action-dim $d
done
```

源目录中已有的 `model_remapped.safetensors`（首次加载时生成的键名映射缓存）不比权重旧时一并链接，否则由视图首次加载时重新生成。pi05_base 未在 Kuavo 数据上训练，输出动作只用于运行时检查；任务效果需要客户微调的 checkpoint 和对应 norm stats。其余输入文件见 [Kuavo 输入准备](pi05/README.md#kuavo)。

<a id="cmd-wall"></a>

### Wall-OSS

| 登记名 | 目录（`$RPU_MODEL_CACHE/` 下） | 获取方式 | 来源与版本 |
|---|---|---|---|
| `wall-oss-0.5` | `wall-oss-0.5` | 下载 | [x-square-robot/wall-oss-0.5](https://huggingface.co/x-square-robot/wall-oss-0.5) `b7dc23a82dbf70a65859c9fdd596703234597028`；许可 仓库未声明 |
| `wall-oss-0.5-w8a16` | `wall-oss-0.5-w8a16` | 转换 | 除 `embed_tokens` 外的全部二维 Linear 权重 INT8 |
| `wall-oss-0.5-w4a16-pgrp` | `wall-oss-0.5-w4a16-pgrp` | 转换 | decoder 专家投影 W4 G32，expert-0 down 保持 INT8 |
| `wall-oss-0.5-w4a16` | `wall-oss-0.5-w4a16` | 转换 | decoder 专家投影逐通道 W4（legacy 配置使用） |

```bash
W="$RPU_MODEL_CACHE/wall-oss-0.5"
python -m rpu_backend.quant.convert_wall_oss_w8a16 --src "$W" --dst "$RPU_MODEL_CACHE/wall-oss-0.5-w8a16"
# W4 G32 且 expert-0 down 保持 INT8：命令行不接受 --keep-int8，用 Python 接口生成
python - "$W" "$RPU_MODEL_CACHE/wall-oss-0.5-w4a16-pgrp" <<'PY'
import sys
from rpu_backend.quant.convert_wall_oss import convert_checkpoint
convert_checkpoint(sys.argv[1], sys.argv[2], group_size=32,
                   extra_int8_suffixes=("moe.experts.0.down_proj.weight",))
PY
python -m rpu_backend.quant.convert_wall_oss --src "$W" --dst "$RPU_MODEL_CACHE/wall-oss-0.5-w4a16" --group-size 0
```

Base 配置用 `[model].fp16_checkpoint` 和 `w8_checkpoint` 填写上面两个目录的路径，不使用登记名。RTC 与分拣配置使用客户的 `wrc_sf_v728` 系列权重（配套交付），见 [RTC](wall_oss/rtc/README.md) 和[分拣](wall_oss/sorting/README.md)。


### 其他 VLA

| 登记名 | 目录（`$RPU_MODEL_CACHE/` 下） | 获取方式 | 来源与版本 |
|---|---|---|---|
| `gr00t-n1d7-3b` | `GR00T-N1.7-3B` | 下载 | [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) `2fc962b973bccdd5d8ce4f67cc63b264d6886495`；许可 NVIDIA Open Model License；另需 `qwen3-vl-2b` 的 config 与 tokenizer |
| `hy-embodied-0.5-vla-umi` | `Hy-Embodied-0.5-VLA-UMI` | 下载 | [tencent/Hy-Embodied-0.5-VLA-UMI](https://huggingface.co/tencent/Hy-Embodied-0.5-VLA-UMI) `3f53d1f8d2bc587c523cfdc9f1041ceee42c2524`；许可 Apache-2.0；HyVLA 与 Hy-Embodied 共用 |
| `lingbot-vla-v2-6b` | `hf/robbyant/lingbot-vla-v2-6b` | 下载 | [robbyant/lingbot-vla-v2-6b](https://huggingface.co/robbyant/lingbot-vla-v2-6b) `11c703bf6a5c1f45b3b69168482da11fdbba53d7`；许可 Apache-2.0（模型卡声明）；模板用 `${RPU_MODEL_CACHE}/lingbot-vla-v2-6b` 作为 checkpoint，按实际下载位置填写 |
| `g05-base` | `G05/g05-base` | 下载 | [OpenGalaxea/G05](https://huggingface.co/OpenGalaxea/G05) `e312be81e90c56a55bcb26b57429bd39a335b449`；许可 G0.5 社区许可（非商用），子目录 `g05-base`；需先申请访问 |
| `internvla-n1-controlled` | `internvla_n1-controlled` | 配套交付 | 本地组装的资产：上游 [InternRobotics/InternVLA-N1-w-NavDP](https://huggingface.co/InternRobotics/InternVLA-N1-w-NavDP) `6d48df24da45e8b31fdcef9572bbcbf68c0baf64`；许可 CC-BY-NC-SA-4.0（模型卡声明，非商用），加上 System2 W8 产物；`asset_manifest` 固定各文件哈希，见 [InternVLA](internvla/README.md#input-files) |
| `rhinovla-controlled` | `rhinovla-controlled` | 配套交付 | 固定版本的源码、checkpoint 与 CPU fixture 组成的资产包；上游 [HuixiAI/RhinoVLA](https://huggingface.co/HuixiAI/RhinoVLA) `7f3708650e302f395e7c63607b48b56fb96b8076`；许可 Apache-2.0，组成见 [RhinoVLA v1](rhinovla/v1/README.md) |
| `lingbot-vla-4b` | `lingbot-vla-4b` | 下载 | [robbyant/lingbot-vla-4b](https://huggingface.co/robbyant/lingbot-vla-4b) `87712223fb463f0f6a6565f8feba64c35418e393`；许可 Apache-2.0（模型卡声明）（LingBot v1：延期，受限支持，尚未在板上运行） |
| `halo_action_expert` | `HALO/pytorch_weights_current` | 配套交付 | HALO CoT EMA 权重（延期：受限支持，尚未在板上运行），见 [HALO](halo/README.md) |

GR00T、HyVLA、LingBot v2 和 G0.5 的 W8 及 NVFP4 配置指向同一份浮点权重，在加载时量化，不需要离线转换；InternVLA 的 System2 W8 已包含在资产包中。下列入口用路径字段而不是登记名：

| 配置 | 字段 | 资产 |
|---|---|---|
| G0.5 K1（`g05/2cam`、`g05/3cam`） | `checkpoint`、`release_root` | 客户 RTC release（配套交付），见 [G0.5](g05/README.md#input-files) |
| LingBot v2 | `checkpoint`、`training_config` | 上表的 `lingbot-vla-v2-6b` 与训练配置文件，见 [LingBot v2](lingbot/v2/README.md#input-files) |
| RhinoVLA v3 | `source_dir`、`checkpoint`（`controlled/`）或 `runtime_factory`、`factory_config`（工厂模板） | 客户 v3 源码与 checkpoint（配套交付），见 [RhinoVLA v3](rhinovla/v3/README.md) |
| Wall-OSS RTC / 分拣 | `checkpoint`、`fp16_checkpoint`、`w8_checkpoint`、`action_w4_checkpoint` | 客户 `wrc_sf_v728` 系列权重（配套交付） |

<a id="no-local-converter"></a>

## 历史与归档

历史 27B 配套权重已归档，不属于当前配置。

