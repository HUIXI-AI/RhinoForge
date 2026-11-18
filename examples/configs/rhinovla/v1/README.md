# RhinoVLA v1 官方 non-RTC 配置

<a id="rhinovla-v1-配置"></a>

[模型页](../README.md) · [模型配置表](../README.md#configurations) · [权重准备](../../WEIGHTS.md)

[FP16 模板](fp16.toml) 加载 `rhinovla-controlled` 的官方 non-RTC 资产，固定使用 10 步去噪。该配置已在板上通过运行时检查，状态见[模型配置表](../README.md#configurations)。

## 准备与运行

资产目录需包含 `source/`、`weights/rhinovla_pretrain.ckpt` 和 `fixture/cpu_fixture.pt`。运行时会核对源码版本、checkpoint 身份及 fixture 的 non-RTC 标记，不能将其他版本的输入混入该目录。模板的 `trust_model_code = true` 用于加载配套模型源码。

从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/rhinovla/v1/fp16.toml --check-config
python examples/run.py --config examples/configs/rhinovla/v1/fp16.toml
```

配置检查不读取这些资产，也不构成硬件验证；输入约束与核心范围见[模型配置表](../README.md#configurations)。

`fixture/cpu_fixture.pt` 可用仓库脚本在 CPU 上生成。需要固定提交的 RhinoVLA 源码、对应 checkpoint 和 LIBERO parquet（脚本会核对 sha256）：

```bash
python scripts/ci/prepare_rhinovla_official_assets.py --source <RhinoVLA 源码> \
  --checkpoint <checkpoint> --libero-parquet <LIBERO parquet> --output-dir <新目录>
```

脚本把 `cpu_fixture.pt` 写在 `<新目录>` 根下，需要手动移到资产目录的 `fixture/cpu_fixture.pt`，与 `source/`、`weights/rhinovla_pretrain.ckpt` 并列。
