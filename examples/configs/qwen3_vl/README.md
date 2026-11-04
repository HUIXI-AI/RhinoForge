# Qwen3-VL 配置

[配置索引](../CONFIGURATIONS.md) · [配置与权重](#weight-recipes) · [运行命令](#quantization)

`text/` 只运行文本生成，`vl/` 运行完整图文生成，`vision/` 只运行视觉编码器。同尺寸、同精度的三类入口共用完整模型权重。图文基础示例可从 [2B FP16](vl/2b/fp16.toml) 开始。

## 配置说明

<a id="选择配置"></a>

<a id="runtime-w8a16--w4a16"></a>

<a id="weight-recipes"></a>

<a id="配置与权重对应"></a>

<a id="基准与配置增量"></a>

### 配置、精度与权重

2/4/8B 以同尺寸、同组件的 `fp16.toml` 为基准：官方 BF16 权重加载为 FP16。32B 没有 FP16 模板，以同组件 W8 作为比较起点，浮点权重用于转换和参考计算。

**Text7** 指每层 Attention 的 Q/K/V/O 和 MLP 的 gate/up/down。下表列出改变精度的模块；embedding、norm、bias 始终保持 FP16，激活为 A16。W8 为逐输出通道对称 INT8，W4 为每 32 个输入值一组的对称 INT4（G32）。

| 配方与配置 | 相对 FP16 的改动；32B 相对 W8 | 权重获取步骤 |
|---|---|---|
| FP16：2B [文本](text/2b/fp16.toml) / [图文](vl/2b/fp16.toml) / [视觉](vision/2b/fp16.toml)；4B [文本](text/4b/fp16.toml) / [图文](vl/4b/fp16.toml) / [视觉](vision/4b/fp16.toml)；8B [文本](text/8b/fp16.toml) / [图文](vl/8b/fp16.toml) / [视觉](vision/8b/fp16.toml) | 实际执行的 Text7、输出头、Vision 均为 FP16 | [下载官方 BF16](#cmd-fp16) |
| Text7 W8：2B [文本](text/2b/w8a16_text.toml) / [图文](vl/2b/w8a16_text.toml) | 仅 Text7 改为 W8；head、Vision FP16；切到 `text-w8a16` alias/profile | [离线转换 Text7](#text-only-w82b只量化text7headvision保留fp16) |
| Text7 + head W8、Vision W8：2B [文本](text/2b/w8a16_text_head_vision.toml) / [图文](vl/2b/w8a16_text_head_vision.toml)；4B [文本](text/4b/w8a16_text_head_vision.toml) / [图文](vl/4b/w8a16_text_head_vision.toml)；8B [文本](text/8b/w8a16_text_head_vision.toml) / [图文](vl/8b/w8a16_text_head_vision.toml)；32B [文本](text/32b/w8a16_text_head_vision.toml) / [图文](vl/32b/w8a16_text_head_vision.toml) | Text7/head 离线转 W8，Vision 加载时转 W8；切到 `runtime-w8a16` alias/profile | [从官方浮点包转换](#runtime-w8官方浮点父--text7head-w8) |
| Text7 AWQ W4：2B [文本](text/2b/w4a16_text_awq_g32.toml) / [图文](vl/2b/w4a16_text_awq_g32.toml)；4B [文本](text/4b/w4a16_text_awq_g32.toml) / [图文](vl/4b/w4a16_text_awq_g32.toml) | Text7 使用 AWQ W4/G32；head、Vision FP16；切到 `text-awq-w4a16` alias/profile | [直接下载 cyankiwi AWQ](#text-only-awq24b直接下载不再转换) |
| Text7 AWQ W4 + head RTN W4、Vision W8：2B [文本](text/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) / [图文](vl/2b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml)；4B [文本](text/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) / [图文](vl/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml)；8B [文本](text/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) / [图文](vl/8b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) | 保留 AWQ Text7；输出头另做 RTN W4/G32，Vision 加载时转 W8；切到 `runtime-w4a16` alias/profile | [从 cyankiwi AWQ 完整包转换](#runtime-w4g32248b-awq--runtime-w4) |
| Text7 + head RTN W4、Vision W8：32B [文本](text/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml) / [图文](vl/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml) | 相对 W8，将 Text7/head 改为 RTN W4/G32；Vision 仍为 W8；使用独立 W4 目录、alias/profile | [从官方 32B 浮点包转换](#runtime-w4g3232b-官方浮点父--runtime-w4) |
| 32B Text7 W8 兼容配方：[图文](vl/32b/w8a16_text.toml) | head、Vision 保留 FP16；使用独立 alias/profile 及模板内 opt-in；物理 prefill≤336、chunk≤64 | [配套包或同格式转换](#32b-legacy-text-w8原包复现--新的同格式转换) |

切换配方时，`dtype/registry_alias/profile_id` 一起选择。表中“只量化文本”指量化范围，图文入口同样可用；纯文本入口不执行 Vision。范围差异用于比较计算变化，各配方的获取步骤使用表内指定源包。

完整 W8/W4 产物中的 Vision 离线仍存 FP16，安装时将 patch embedding、各 block 的 Q/K/V/O/MLP、末端 merger 和 DeepStack merger 转为 W8；norm、bias、位置参数仍为 FP16。仅 Text7 量化的配方不启用这一步。

2/4/8B 的 Text7 AWQ 来自 cyankiwi 发布的已量化完整包，加载或转换时保留整数码和 scale；RTN 是在本仓库对浮点权重直接舍入。32B W4 的 Text7/head 均用 RTN。运行产物用 UINT8 打包两个 4 bit 值及 FP16 scale；AWQ 源包用 compressed-tensors 的 I32 packed 格式，两者按各自入口读取。

### 输入与执行变体

VL 先编码图像，再把视觉特征送入文本 prefill 和 DeepStack，随后使用 KV cache 逐 token decode。基准 Text/VL 为自动 chunk、`prefill.fast_replay=true`、默认 TP8 和文本/Vision ACC16、自然长度、D4、EOS 停止、W2/R3。Vision 模板只返回视觉输出，D 不用于生成。

<a id="输入与执行变体"></a>

W2/R3 表示预热 2 次、正式测量 3 次，TP 表示配置的根核数。P 是输入 token 数，D 是 prefill 后的 decode 次数，固定 D 时共产生 D+1 个输出。VL 的 P 包含视觉和 chat 模板 token；输入超过目标 P 时会报错，避免截断图像或 assistant 结构。

| 配置组 | 相对相同组件 FP16 基础模板的输入和执行变化 |
|---|---|
| Text 的 `w8a16_*.toml` / `w4a16_*.toml` | prompt 改为编译器问题，固定 P512/D31，关闭 EOS；W2/R5、no_grad、显式 TP8、文本/Vision ACC16 |
| VL 量化配置（下列 32B Text7-W8 除外） | 改为单句图像描述、P430/D31，图片占位路径为 640×480 输入，关闭 EOS；W2/R5、no_grad、显式 TP8、文本/Vision ACC16 |
| Text `fp16_p192_d192.toml` | 改用单句描述提示词，P192/D192、关闭 EOS、no_grad、W2/R1 |
| VL `fp16_p256_d192.toml` | 同一单句描述提示词，P256/D192、关闭 EOS、no_grad、W2/R1；需填写图片 |
| Text/VL `fp16_p4096_d4096.toml` | 单句描述提示词，P4096/D4096、关闭 EOS、no_grad、W2/R1；4/8B 还显式写出 TP8，2B 使用默认 TP8 |
| 32B `w8a16_text.toml` | 自然图文长度、D4、EOS 停止、W2/R3；未开启 `prefill.fast_replay` |

量化模板比 FP16 基础模板生成更长，运行次数也不同。对比配方时先统一输入、P/D、EOS、线程和测量设置；表中的参数变化与量化范围分别计算影响。

## 准备与运行

<a id="准备权重并运行"></a>

<a id="quantization"></a>

下载和离线转换可在 CPU 机器完成，`examples/run.py` 在设备板卡执行。以下命令从仓库根目录运行，使用已安装项目依赖的 Python 环境。

先设置缓存根目录：

```bash
export RPU_MODEL_CACHE=/absolute/path/to/model_cache
```

离线转换时，`SRC` 填该配方所需的完整输入 checkpoint，`DST` 按下面命令写入尚不存在的登记目录。运行时只按 `example.registry_alias` 查找缓存目录，不读取 Shell 的 `SRC/DST`；此族不支持 `--model` 或 `input.checkpoint`。

已有权重可跳过对应下载，但需绑定到下表的缓存目录。例如，已有 4B 官方浮点权重时，在目标路径尚不存在的情况下建立链接：

```bash
mkdir -p "$RPU_MODEL_CACHE"
ln -s /absolute/path/to/existing-Qwen3-VL-4B-Instruct "$RPU_MODEL_CACHE/Qwen3-VL-4B-Instruct"
```

FP16 直接读取上述目录；直接加载 AWQ 或已有转换产物时，改用下表中该配方的缓存目录。离线转换的 `SRC` 可另选源路径，`DST` 仍须匹配登记目录。同尺寸、同配方的 Text/VL 共用权重，VL 另填 `input.images`。

<a id="weights"></a>

<a id="weight-sources"></a>

### 权重来源

官方权重以 BF16 保存，FP16 是加载后的计算精度。路径均相对于 `$RPU_MODEL_CACHE`，需完整下载 config、tokenizer、processor 和全部分片。

| 权重 | 磁盘存储精度 | 发布者与下载来源 | 缓存目录 |
|---|---|---|---|
| Qwen3-VL-2B-Instruct | BF16 | Qwen 官方 [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct) | `Qwen3-VL-2B-Instruct` |
| Qwen3-VL-4B-Instruct | BF16 | Qwen 官方 [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct) | `Qwen3-VL-4B-Instruct` |
| Qwen3-VL-8B-Instruct | BF16 | Qwen 官方 [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-VL-8B-Instruct) | `Qwen3-VL-8B-Instruct` |
| Qwen3-VL-32B-Instruct | BF16 | Qwen 官方 [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) · [精度元数据](https://huggingface.co/api/models/Qwen/Qwen3-VL-32B-Instruct) | `Qwen3-VL-32B-Instruct` |
| 2B AWQ | Text7 为 **INT4/G32**，I32 packed；scale 与未量化 companion 为 BF16；shape 为 I64 | **第三方 cyankiwi** [Qwen3-VL-2B-Instruct-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit) · [量化声明](https://huggingface.co/cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit/blob/main/config.json) | `Qwen3-VL-2B-Instruct-AWQ` |
| 4B AWQ | Text7 为 **INT4/G32**，I32 packed；scale 与未量化 companion 为 BF16；shape 为 I64 | **第三方 cyankiwi** [Qwen3-VL-4B-Instruct-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit) · [量化声明](https://huggingface.co/cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit/blob/main/config.json) | `Qwen3-VL-4B-Instruct-AWQ` |
| 8B AWQ | Text7 为 **INT4/G32**，I32 packed；scale 与未量化 companion 为 BF16；shape 为 I64 | **第三方 cyankiwi** [Qwen3-VL-8B-Instruct-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit) · [量化声明](https://huggingface.co/cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit/blob/main/config.json) | `Qwen3-VL-8B-Instruct-AWQ` |
| 本仓库运行时 W8/W4 派生包 | Text7/head：INT8 或 packed INT4，FP16 scale；其余 FP16 | 本仓库离线转换，不是 Qwen 官方或 cyankiwi 原包 | `Qwen3-VL-{SIZE}B-Instruct-runtime-w8`；2/4/8B 的 `-runtime-w4-awq`；32B 的 `-runtime-w4` |
| 2B Text-W8 派生包 | Text7 INT8 + FP16 scale；其余 FP16 | 本仓库离线转换 | `Qwen3-VL-2B-Instruct-w8a16-text` |
| 32B Text-W8 配套包 | Text7 INT8 + FP16 scale；head/Vision 等 FP16 | 配套包；也可用下方命令从官方父转换 | `Qwen3-VL-32B-Instruct-w8a16-full-20260816` |

32B 的 BF16 父只用于转换；没有32B FP16运行模板。直接消费 AWQ 的 `w4a16_text_awq_g32.toml` 仅有2/4B，8B AWQ须先转换为运行时 W4。AWQ 包的未量化参数、tokenizer 和 processor 一起使用，保持同一来源。

<a id="cmd-fp16"></a>

### 官方浮点源：FP16 运行与量化准备

```bash
SIZE=4  # 可选：2、4、8、32；32B仅用于量化
hf download "Qwen/Qwen3-VL-${SIZE}B-Instruct" \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-VL-${SIZE}B-Instruct"
```

2/4/8B 图文/纯视觉分别使用 `vl/${SIZE}b/fp16.toml` 或 `vision/${SIZE}b/fp16.toml`，先填写该模板 `input.images`，无需另一份权重。

<a id="text-only-awq24b直接下载不再转换"></a>

### AWQ 源权重：2/4/8B 下载一次

2/4B 的 `w4a16_text_awq_g32.toml` 直接读取此目录，head/Vision 保留 FP16；2/4/8B 的完整 W4 也从此包转换。只改 `SIZE`：

```bash
SIZE=2  # 可选：2、4、8；8B仅作完整W4转换源
DST="$RPU_MODEL_CACHE/Qwen3-VL-${SIZE}B-Instruct-AWQ"
hf download "cyankiwi/Qwen3-VL-${SIZE}B-Instruct-AWQ-4bit" --local-dir "$DST"
```

<a id="runtime-w8官方浮点父--text7head-w8"></a>

### 运行时 W8：官方浮点父 → Text7/输出头 W8

先按[官方浮点下载步骤](#cmd-fp16)准备对应尺寸的源目录。

只改 `SIZE` 选择2/4/8/32B。转换器必须使用 `python -m` 调用；Vision权重在加载时转W8，不需要第二个离线目录。

```bash
SIZE=4  # 可选：2、4、8、32
SRC="$RPU_MODEL_CACHE/Qwen3-VL-${SIZE}B-Instruct"
DST="$RPU_MODEL_CACHE/Qwen3-VL-${SIZE}B-Instruct-runtime-w8"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=python python3 -m rpu_backend.quant.convert_qwen3_vl \
  --src "$SRC" --dst "$DST" --bits 8
```

例如4B输出供 [text/4b/w8a16_text_head_vision.toml](text/4b/w8a16_text_head_vision.toml) / [vl/4b/w8a16_text_head_vision.toml](vl/4b/w8a16_text_head_vision.toml) 使用。

<a id="runtime-w4g32248b-awq--runtime-w4"></a>

### 运行时 W4/G32：2/4/8B AWQ → 运行时 W4

先按[AWQ 下载步骤](#text-only-awq24b直接下载不再转换)准备对应尺寸的源目录。

**输入必须是下列 cyankiwi AWQ 完整包，不是官方浮点目录。** 保留该包Text7 AWQ及浮点 companion，转换器将head转RTN W4；Vision加载时转W8。只改 `SIZE`：

```bash
SIZE=4  # 可选：2、4、8
SRC="$RPU_MODEL_CACHE/Qwen3-VL-${SIZE}B-Instruct-AWQ"
DST="$RPU_MODEL_CACHE/Qwen3-VL-${SIZE}B-Instruct-runtime-w4-awq"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=python python3 -m rpu_backend.quant.convert_qwen3_vl \
  --src "$SRC" --dst "$DST" --bits 4
```

例如4B输出供 [text/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](text/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) / [vl/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml](vl/4b/w4a16_text_awq_head_rtn_g32_vision_w8a16.toml) 使用。

<a id="runtime-w4g3232b-官方浮点父--runtime-w4"></a>

### 运行时 W4/G32：32B 官方浮点父 → 运行时 W4

先按[官方浮点下载步骤](#cmd-fp16)准备对应尺寸的源目录。

32B使用浮点父，Text7/head均为RTN W4；不要套用上面的AWQ源命令。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3-VL-32B-Instruct"
DST="$RPU_MODEL_CACHE/Qwen3-VL-32B-Instruct-runtime-w4"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=python python3 -m rpu_backend.quant.convert_qwen3_vl \
  --src "$SRC" --dst "$DST" --bits 4
```

输出也供 [vl/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml](vl/32b/w4a16_text_head_rtn_g32_vision_w8a16.toml) 使用。以上三类离线入口源码为 [python/rpu_backend/quant/convert_qwen3_vl.py](../../../python/rpu_backend/quant/convert_qwen3_vl.py)。

<a id="text-only-w82b只量化text7headvision保留fp16"></a>

### 仅文本量化 W8：2B只量化Text7，输出头/视觉保留FP16

先按[官方浮点下载步骤](#cmd-fp16)准备对应尺寸的源目录。

这里的“Text-only”指**量化范围**；产物同时供 text / VL 的 `w8a16_text.toml` 使用，与 runtime W8 目录不同。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3-VL-2B-Instruct"
DST="$RPU_MODEL_CACHE/Qwen3-VL-2B-Instruct-w8a16-text"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 python/rpu_backend/quant/convert_qwen3.py \
  --src "$SRC" --dst "$DST" --skip-modules model.visual lm_head
```

入口为 [python/rpu_backend/quant/convert_qwen3.py](../../../python/rpu_backend/quant/convert_qwen3.py)；不加 `--quant-lm-head`。

<a id="32b-legacy-text-w8原包复现--新的同格式转换"></a>

### 32B Text-W8：准备兼容配方权重

先按[官方浮点下载步骤](#cmd-fp16)准备对应尺寸的源目录。

[vl/32b/w8a16_text.toml](vl/32b/w8a16_text.toml) 使用 `Qwen3-VL-32B-Instruct-w8a16-full-20260816` 目录。已有配套包可直接使用；以下命令从官方浮点父转换同格式权重，`DST` 应为新目录。运行前填写 `input.images`。

```bash
SRC="$RPU_MODEL_CACHE/Qwen3-VL-32B-Instruct"
DST="$RPU_MODEL_CACHE/Qwen3-VL-32B-Instruct-w8a16-full-20260816"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 python/rpu_backend/quant/convert_qwen3.py \
  --src "$SRC" --dst "$DST" --skip-modules lm_head
```

### 关键配置项

通用语义见[参数说明](../PARAMETERS.md)；本模型的模板值、有效省略值及关联如下。

| 层次 / 字段 | 模板值与省略后的有效值 | 范围与关联 |
|---|---|---|
| 配方：`example.dtype/registry_alias/profile_id` | 按上表选择，三项必填 | 同尺寸 Text/VL 和长度变体共用相同配方目录；仅改 dtype 不会增加量化范围 |
| 入口：`input.component` | 模板为 `text`、`multimodal` 或 `vision`；省略为 `multimodal` | 与 profile 组件一致；Vision 没有文本生成 |
| 输入：`input.prompt/images` | 文本为模板提示词；图文/视觉需填写图片路径 | Text 用裸 token；VL 由 processor 生成 chat 模板；图像尺寸由 processor 处理 |
| 长度：`input.prefill_tokens/decode_steps/stop_on_eos` | 基础自然 P/D4/`true`；量化 Text P512、VL P430，均 D31/`false`；省略值为自然 P、D4、`true` | P≥1、D≥0；VL 的 P 包括视觉与 chat token，不能截断已有输入。P4096/D4096 对应 8192 位置缓存 |
| 拓扑：`rpu_execution.model.num_cores` | 基础默认 8，量化模板显式 8；省略为 8 | 量化模板使用 TP8；FP16 的 4/6/8 核须选已登记尺寸与 profile |
| 累加：`rpu_execution.prefill/vision.linear_acc32` | 量化模板显式 `false`；省略均为 ACC16 | 两组件独立；`true` 为 ACC32。文本项同时控制 prefill/decode Linear，权重不变 |
| 分块：`rpu_execution.prefill.chunk_size` | 模板与省略值为 `auto` | 显式值是 16 的正整数倍；2/4/8B 的 chunk 上限分别为 320/256/128；32B 兼容配方≤64。总 prefill 长度由实际安装容量、物理 padding 和算子/SPM 约束决定 |
| 重放：`rpu_execution.prefill.fast_replay` | 普通 Text/VL 显式 `true`，32B 兼容模板省略；省略为 `false` | 复用符合条件的 prefill 执行记录；Vision 没有此字段 |
| 测量：`cpu.threads`、`run.*` | CPU 省略时沿用环境；轮次见上表；省略 W/R 为 2/1、`inference_mode=true` | 普通量化和长输入模板显式 no_grad；`warmup≥0`、`runs≥1`，对比时统一 |

完整尺寸与组件范围见[模型矩阵](../../../docs/model_support.md)。改变精度、核数和 ACC 等安装参数后用新进程运行。

公开 loader 可通过
`RPUModelForConditionalGeneration.from_pretrained(..., max_seq_len=L)`
绑定文本安装容量，再创建容量不超过 `L` 的 cache。默认容量和示例长度不是硬上限；
更大申请由实际内存分配结果决定，位置表、算子及坐标位宽、输入形状和 SPM 约束仍然
适用。修改安装容量须重新创建模型和 cache，申请成功不扩大现有硬件验证范围。

<a id="使用"></a>

### 运行自己的输入

准备权重后复制 TOML，填写提示词、图片和 `run.output_dir`：

```bash
CONFIG=examples/configs/qwen3_vl/text/4b/w8a16_text_head_vision.toml
python3 examples/run.py --config "$CONFIG" --check-config
python3 examples/run.py --config "$CONFIG"
```

`--check-config` 检查配置字段和登记关系；正式加载再检查权重与模型形状。精度、核数和 ACC 等安装参数改变后用新进程运行。

### 重放保存输入

VL可以重放带完整SHA256的预处理safetensors。替换整个 `[input]`，不保留 `prompt/images/prefill_tokens` 等其他输入字段：

```toml
[input]
component = "multimodal"
prepared_input_safetensors = "/path/to/input.safetensors"
prepared_input_sha256 = "填写该文件的64位SHA256"
decode_steps = 31
stop_on_eos = false
```

文件含INT64的 `input_ids/attention_mask/image_grid_thw/position_ids` 和FP32的 `pixel_values`，仅支持B1、单图单帧、全1 mask。入口在模型安装前校验文件、五张量和正常M-RoPE位置；不添加chat或补齐，正常decode delta保持。相对路径按配置文件目录解析，输出的 `generation.prepared_input` 记录全部哈希。

转换实现：[完整 W8/W4](../../../python/rpu_backend/quant/convert_qwen3_vl.py)、[Text-W8](../../../python/rpu_backend/quant/convert_qwen3.py)、[Text-AWQ](../../../python/rpu_backend/quant/load_qwen3_vl_awq.py)、[Vision 安装](../../../python/rpu_backend/adapters/qwen3_vl/vision.py)。运行库见[运行资产](../../../docs/runtime_assets.md)。
