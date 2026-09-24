# Examples and execution scope

[简体中文](model_support.zh.md) | English

The [`examples/`](../examples/README.md) directory provides the unified
`run_model.py` entry point and model-specific scripts. TOML configurations are
organized by model, component, precision, and workload under
[`examples/configs/`](../examples/configs/README.md).

Choose a configuration, supply its checkpoint and input assets, and run
`python examples/run_model.py --config PATH --check-config` before execution.
This checks the configuration without loading weights or accessing a board;
it does not validate the checkpoint or certify model quality. See
[model execution](model_testing.md), [API reference](api_reference.md), and
[quantization](quantization.md) for the corresponding runtime contracts.

## General runtime limits

Runtime admission checks remain authoritative for each checkpoint and request.
A configuration file does not imply support for other model sizes, precisions,
core counts, image counts, or input lengths. Source-only and controlled examples
retain their exact input restrictions and opt-in gates. Historical validation
records do not describe the current runtime or establish a new certification.

Model weights, request data, Rhino Launch, and operator assets are supplied
separately. Follow [model assets](model_assets.md) and
[runtime assets](runtime_assets.md) for their format, provenance, and licensing
requirements. The public Wall-OSS base adaptation is retained;
application-specific models, private data, and deployment customizations are
excluded from the public package.
