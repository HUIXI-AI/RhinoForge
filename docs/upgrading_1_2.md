# Upgrade to v1.2.0

[简体中文](upgrading_1_2.zh.md) | English

The current release is `1.2.0`. See the [changelog](../CHANGELOG.md) for
changes. The distribution is `rhinoforge`; Python uses `rpu_backend` and `torch.rpu`.

## Install

1. Follow [Getting started](getting_started.md) in a fresh Python 3.12 Conda environment.
   [pyproject.toml](../pyproject.toml) pins PyTorch 2.10.0 and Transformers 5.3.0;
   select the `vision`, `vla` or `audio` extra for the models you use.
2. Obtain `RhinoForge-runtime-v1.2.0.tar.gz` and its checksum from the
   [asset request entry](runtime_assets.md#request-runtime-assets). Verify the
   archive and keep both main and DDR-expansion assets with their sidecars.
   Rhino Launch uses the same `1.2.0` version. Exact component revisions are in
   [runtime-dependencies.json](../release/runtime-dependencies.json).
3. Update `RHINO_LAUNCH_ROOT`, `RPU_KERNEL_LIB_PATH` and
   `RPU_KERNEL_LIB_PATH_EXPANSION`, rebuild the extension, and start a new process.
   Keep the previous environment and assets together for rollback.
4. Run `python -m pip check`, validate the chosen TOML with the command below,
   then run it once on an idle device with its required checkpoint and inputs.

## Select a configuration

Choose a current template from the [model index](../examples/configs/README.md).
Copy it to a local file and fill in the model guide's asset and input paths:

```bash
cp examples/configs/qwen3/text/0_6b/fp16.toml qwen3.local.toml
python -m examples.run --config qwen3.local.toml --check-config
python -m examples.run --config qwen3.local.toml
```

Configuration names describe the model, component, precision and workload;
see the [naming guide](../examples/configs/NAMING.md).

| Model | Required inputs |
|---|---|
| [GR00T](../examples/configs/gr00t/README.md) | Matching checkpoint and camera/state observation tensors; set `input.observation`, `input.seed` and `input.num_steps`. |
| [InternVLA / NavDP](../examples/configs/internvla/README.md) | Complete-chain `input.fixture` and `input.asset_manifest`, with `input.sample_num` and `input.seed`. |
| [Pi0.5](../examples/configs/pi05/README.md) | Complete policy batch and matching checkpoint. Standalone SigLIP callers use its component adapter. |
| [LingBot v2](../examples/configs/lingbot/v2/README.md) | Select camera count and language length; provide the checkpoint, training YAML and matching case, observation or image/state inputs. Preserve camera order and action normalization. |
| [Qwen3.5 MoE](../examples/configs/qwen3_5/README.md) | Long-prompt templates use `prompt_file`, `prompt_sha256`, `input_ids_sha256` and explicit prefill/decode lengths. Use the template's `run.timing_mode`. |
| [RhinoVLA](../examples/configs/rhinovla/README.md) | v1 uses the pinned source, checkpoint and fixture. v3 uses the documented runtime factory and paired prepared inputs. |

For templates that require `input_ids_sha256`, hash the first `prefill_tokens`
contiguous INT64 token-ID bytes produced by the matching tokenizer with
`add_special_tokens=False`. The text must contain enough tokens and the sequence
must exclude EOS/PAD. See [input construction](../examples/common/generation.py).

## Logging and conversion

To display generated text and ordinary progress, add this top-level table:

```toml
[logging]
level = "INFO"
summary = true
```

The default log level is `ERROR`; `summary` independently controls the model and
runtime summaries. See [logs and output](../examples/RESULTS.md).

Wall-OSS W4 conversion defaults to `--group-size 32` and converts both decoder
experts, including expert-0 Down, to INT4. Use `--group-size 0` for per-channel
scales; non-empty `--keep-int8` is rejected. Generate a new directory with the
[conversion commands](../examples/configs/WEIGHTS.md#cmd-wall). The sorting
recipe loads only the action expert from that W4 directory.

Use a new process after changing runtime assets, model precision or cold Graph
settings. After a failed or partial weight transformation, reload clean original
weights. See [runtime configuration](runtime_config.md) for setting lifetimes.
