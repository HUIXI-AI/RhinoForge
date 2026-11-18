# InternVLA NavDP 配置

<a id="internvla-配置"></a>

<a id="配置矩阵"></a>

[配置总览](../README.md) · [全部配置](#configurations) · [权重准备](../WEIGHTS.md) · [参数说明](../PARAMETERS.md)

[System2 + RGBD + NavDP 配置](navdp/w8a16_system2.toml) 用于固定 `output_ids` 的 System2/RGBD/NavDP 链路，不执行文本生成，需要配套 fixture 和资产清单。文件名中的 W8A16 对应 System2 backbone；登记 profile 的 dtype 为 `fp16`，不能据此将整条链路理解为统一 W8A16。

<a id="运行"></a>

## 准备与运行

准备 `internvla-n1-controlled` 资产目录，填写 `input.fixture` 和 `model.asset_manifest`。fixture 提供导航图像、目标等输入；`sample_num` 必须与其中 `init_noise` 的样本数一致，模板默认值为 4。资产身份和输入形状在运行时校验。

从仓库根目录执行：

```bash
python examples/run.py --config examples/configs/internvla/navdp/w8a16_system2.toml --check-config
python examples/run.py --config examples/configs/internvla/navdp/w8a16_system2.toml
```

`--check-config` 通过只说明配置匹配，不能替代配套资产检查。配置表中的“支持”表示该配置在板上通过运行时检查，不代表导航任务质量已评估。

<a id="input-files"></a>

## 输入文件准备

`input.fixture`（导航图像、RGBD、固定 `output_ids` 和初始噪声）与 `model.asset_manifest`（各文件的 sha256）可用仓库脚本一次生成。脚本从 Hugging Face 下载 `InternRobotics/InternVLA-N1-w-NavDP` 的固定版本，需要 InternNav 源码和 transformers 4.51.0、diffusers 0.33.1 的 CPU 环境：

```bash
python scripts/ci/prepare_internvla_full_chain_assets.py --root <新目录> \
  --hf-cache <HF 缓存目录> --internnav-root <InternNav 源码目录>
```

脚本在 `<新目录>/internvla_n1-controlled/` 下写出 `full_chain_fixture.pt` 和 `asset_manifest.json`，把这两个路径填入 TOML。`sample_num` 必须等于 fixture 中 `init_noise` 的第一维（脚本生成的是 4）。

| 参数 | 取值与默认 | 说明 |
|---|---|---|
| `--root` | 必填 | 输出根目录。资产写到 `<root>/internvla_n1-controlled/`（NavDP 头、System2 前端与 FP16 ViT、System2 W8 decoder、fixture、`provenance.json`、`asset_manifest.json`）；`extracted.json`、`system2_fixture.pt` 写在 `<root>` |
| `--hf-cache` | 必填 | Hugging Face 缓存目录；下载固定 revision `6d48df24da45e8b31fdcef9572bbcbf68c0baf64` |
| `--internnav-root` | 必填 | InternNav 源码的 git 目录；HEAD 须为 `7a5c62400ac45b313d9b709c740b64191556a242`，工作区不能有改动 |
| `--prepare-only` | 开关，默认关 | 只抽取权重并写清单，不生成 fixture，也不检查 transformers / diffusers 版本；产物不能直接运行 |
| `--modelscope-weights` | 开关，默认关 | safetensors 改从 ModelScope revision `f0458b17d5cf475728c73d3dea8fc5da5a66c9ea` 获取，其余文件仍取自 Hugging Face |

- **放到哪里**：权重按登记名 `internvla-n1-controlled` 从 `$RPU_MODEL_CACHE/internvla_n1-controlled` 读取，入口不接受 `checkpoint` 字段。生成后把整个 `internvla_n1-controlled/` 目录移到 `$RPU_MODEL_CACHE/` 下；清单里的相对路径按清单所在目录解析，整体移动后仍有效。
- **重跑**：同一 `<root>` 重跑时复用已抽取的文件，文件或脚本有变化会报错，此时换新目录；fixture 每次重新生成。

<!-- model-configurations-start -->

<a id="configurations"></a>
## 全部配置与输入范围

每组先列各配置不同的项，再列本组配置相同的项；修改相机、文本长度、精度或核数时，同时满足联动限制。4/6/8 核列：● 已在板上运行通过（4/6 核写出各组件实际核数），○ 支持但未在板上运行，✕ 不支持（原因列在表下）。支持状态和 CPU 线程数的读法见配置表说明；实测耗时见性能报告：InternVLA。

### 固定导航系统fixture

| 项目 | 值 |
|---|---|
| 配置 | [navdp/w8a16_system2.toml](navdp/w8a16_system2.toml) |
| 支持状态 | 支持（`SUPPORTED`） |
| 精度 | w8a16 |
| 量化范围 | System2 backbone W8A16 |
| 输入类型 | System2 + RGBD + NavDP tensor |
| 文本可调范围 | output_ids[1,S]；图像token恰196个 |
| 图像/音频可调范围 | grid[[1,28,28]]；pixels首维784；RGB[1,2,224,224,3]；depth[1,2,224,224,1] |
| 生成可调范围 | sample_num=4 |
| 其他输入形状 | noise[4,32,3]；position_ids[S+4,3]或[3,1,S+4] |
| 联动限制 | 资产清单绑定；token ID在词表内；4个latent rows位置完整。属于固定系统链fixture，不接受任意原始图文。 |
| 默认核数 | 8 |
| 4 核 | ● 已实测 |
| 6 核 | ● 已实测 |
| 8 核 | ● 已实测 |
| CPU线程数 | 未设置（沿用环境） |
| 权重 | `internvla-n1-controlled` |
| 入口 | [run.py](../../run.py) |

输入校验来源：[runner.py](../../../examples/runners/internvla/runner.py) · [backbone.py](../../../python/rpu_backend/adapters/internvla_n1/backbone.py)。

<!-- model-configurations-end -->
