# Compute-core topology

[简体中文](planner_core_topology.zh.md) | English

Set `rpu_execution.model.num_cores` before loading weights. It accepts the
integers `4`, `6` and `8`, with `8` as the default; the selected model recipe
further restricts admission. A boolean is not a core count. Changing the budget
requires a new model, cache and Graph set.

The budget selects a complete component topology. It does not change the
physical eight-core storage layout or act as a per-operation thread setting.

| Dense decoder budget | Attention | MLP | Output head |
|---|---:|---:|---:|
| 4 | 4 | 4 | 4 |
| 6 | 4 | 6 | 4 |
| 8 | 8 | 8 | 8 |

An admitted component can use fewer cores than its owner. In particular, a
six-core owner can contain a four-core component. The adapter fixes these
choices during installation; callers do not construct their own combinations.

## Model recipes

| Family | Reduced-core configuration |
|---|---|
| Qwen3 / Llama / Gemma4 | Budgets 4/6/8 across Qwen3 0.6B–32B (FP16, W8A16, W4A16 G32, NVFP4 per recipe), Llama 3.2-1B (6-core MLP padded `8192 → 8256`), and Gemma4 E4B (2-core attention, 4/6-core MLP with `10240 → 10272` at 6 cores). |
| Qwen3-VL / DINOv3 | Budgets 4/6/8 across Qwen3-VL 2B–32B (including Vision W8/W4/NVFP4 on four Vision lanes at reduced budgets) and DINOv3 ViT-B (FP16, W8A16, W4A16 G32). |
| Qwen3.5 dense & 35B-A3B | Budgets 4/6/8 for dense 0.8B–27B (per template recipe; 27B W4 group32 stays 8-core) and 35B-A3B mixed-E4M3 / RTN-W8A16 (`(4,6,4)` expert split at budget 6). |
| Qwen3 ASR / TTS | Budgets 4/6/8 for 0.6B and 1.7B across FP16, W8A16, W4A16 G32 and NVFP4; audio encoder and codec Transformer use four cores at budgets 4/6. |
| Pi0.5 & VLA families | Budgets 4/6/8 for all 29 Pi0.5 LIBERO/Kuavo templates (except T32 full-prefix W8A8), G0.5 K1, GR00T-N1.7, HyVLA, HY-Embodied, InternVLA-N1, LingBot v2, RhinoVLA v3, and Wall-OSS Base/RTC/Sorting (except Sorting Action W4 G32). Legacy fixtures and HALO remain 8-core. |

Use the [configuration catalog](../examples/configs/CONFIGURATIONS.md) for the
checkpoint, precision and input limits of each entry. For official Qwen3.5-27B,
W8 accepts 4/6/8; W4 group32 accepts 8. The selected checkpoint metadata remains
part of admission even when the architecture name matches.

## Profile details

For reduced Qwen3 FP16, admission checks the configuration and actual modules
before weight conversion. The vocabulary is 151936, head dimension is 128 and
physical KV storage has eight stripes.

| Geometry | Layers | Hidden | Logical MLP width | Q / KV heads | MLP width at budget 6 |
|---|---:|---:|---:|---|---:|
| 0.6B | 28 | 1024 | 3072 | 16 / 8 | 3072 |
| 1.7B | 28 | 2048 | 6144 | 16 / 8 | 6144 |
| 4B | 36 | 2560 | 9728 | 32 / 8 | 9792 |
| 8B | 36 | 4096 | 12288 | 32 / 8 | 12288 |

At budget six, vocabulary and attention heads still use four partitions; MLP
uses six. Quantization, sliding attention, M-RoPE, AdaRMS, mixed accumulation,
custom projections and batched requests do not inherit this plain FP16 profile.
The exact 8B adapter validates the model first, transfers the embedding/head,
then converts and transfers each layer Linear while releasing its CPU storage.
Tied weights and conversion tags remain tracked; failed partial conversion
requires a fresh model.

Reduced Qwen3-VL keeps interleaved M-RoPE `[24,20,20]`, DeepStack consumers
`[0,1,2]` and per-handle ACC32. Patch embedding and position interpolation use
CPU FP16; final and DeepStack mergers use CPU FP32. Admission checks both towers
before either is modified. Quantized, packed-delivery, 32B, VLA and batched
profiles do not inherit this reduced-core recipe. Create its text cache with
`RPUCache.from_model(model.model.language_model, ...)`.

Qwen3.5 0.8B/2B/4B/9B text MLP widths are 3584/6144/9216/12288. Only 0.8B
needs padding for six MLP cores. The exact 2B `w8_main_w8_gdn` recipe also
accepts budgets four and six: main and GDN q/k/v/z/out projections use signed
INT8 with FP16 per-channel scales; GDN b/a, convolution, normalization,
embedding and LM head stay FP16. Text accumulation is ACC16. The 9B image tower uses physical head/MLP widths
80/4352. Video and other quantized geometries retain their own admission.

The Qwen3.5-35B-A3B mixed-E4M3 profile has a separate topology. Prepare its
checkpoint with the [official-source download and CPU conversion commands](../examples/configs/qwen3_5/README.md).
All three core budgets share the converted weights; core-specific padding and
packing happen during cold installation:

| Budget | Attention / GDN / router / head / Vision | Routing / routed / shared | Expert logical / physical width |
|---|---|---|---|
| 4 | 4 / 4 / 4 / 4 / 4 | 4 / 4 / 4 | 512 / 512 |
| 6 | 4 / 4 / 4 / 4 / 4 | 6 / 6 / 6 | 512 / 576 |
| 8 | 8 / 8 / 8 / 8 / 8 | 8 / 8 / 8 | 512 / 512 |

Its native tuple is `(2, root, attention, GDN, router, routing, routed, shared,
head, physical_KV)`; budget six binds `(2,6,4,4,4,6,6,6,4,8)`. Both this and the
legacy layout getter require installed weights. Adapter, cache and Vision
bridge check the complete tuple. Gate/up logical halves are split before zero
padding; routed E4M3 padding uses zero codes and unit scales, and shared FP16
projections receive matching zeros. Router output uses a fixed model-owned
buffer before the six-core routing stage. No weights are repacked during replay.
Configure `rpu_execution.model.num_cores` (`4`, `6`, or `8`) in the [mixed-E4M3 template](../examples/configs/qwen3_5/vl/35b_a3b/fp8_e4m3_experts_attn_gdn.toml).

Pi0.5 reduced FP16/W8A16 uses Vision4, attention4 and conditioning4, with MLP4
or MLP6. Cameras run separately at 256 rows and physical KV keeps eight stripes.
Other owners, including HALO, retain their own admission. HALO's `attn_tp` and
`mlp_cores` validate its fixed layout; they do not select arbitrary topologies.

## Logical shapes and physical padding

Keep the original model configuration and input shapes. The adapter pads and
converts weights once, then binds the resulting layouts to the owner:

- Six-core MLP layouts may pad intermediate widths: `3584 → 3648`,
  `9728 → 9792` or `17408 → 17472`.
- The Qwen3-VL 8B reduced vision layout pads head width `72 → 80` and MLP width
  `4304 → 4352`; these are storage dimensions, not new model geometry.
- The mixed-E4M3 MoE six-core expert layout pads its intermediate width
  `512 → 576`.

Do not pre-pad the checkpoint or update Hugging Face configuration to these
physical sizes. A second conversion would change the layout again.

## Cache and Graph ownership

Create `RPUCache` or `Qwen3_5Cache` through `from_model` after selecting the
topology. Qwen3.5 also binds convolution and recurrent state; a cache constructed
from configuration alone must receive the same core budget. Its default remains
eight. `reset()` clears the recurrent state as well as the logical position;
rewinding to a nonzero position cannot reconstruct that state.

KV storage retains eight physical stripes even when attention computes on four
cores. Compute budget, physical storage and logical sequence length are distinct.
Graph children must fit their owner's budget, and queues and transfers use that
same owner. Model-specific arena reservations happen before weight installation;
creating another component later cannot extend a frozen reservation. For the
Qwen3.5 9B image/text path, enable the template's vision opt-in before
installation. A model installed for text only cannot add vision in place.
The exact dense FP16 9B profile reserves four uniform Graph arenas before
weight upload, including a text-only install. Decode and Vision retain their
own leases; bounded prefill and diagnostic Graphs borrow transient leases.
The pool cannot grow after installation. Closing a model returns its active
leases; the pool remains reserved until process shutdown. Arena count is not
an `rpu_execution` option.

Chunk and padding changes supported by the [execution API](api_reference.md#execution-configuration-and-updates)
replace plans between requests. They do not change this cold topology.

The decoder layout hash and relevant operator manifests include the complete
cold topology. Shared FMB routes bind schema, consumer budget and producer
count. Chunk selection still uses actual per-core SPM capacity.

For attention TP4 with eight physical stripes, effective KV slots are
`8*8/4=16`; the batch-one K/V cache shape is `[1, blocks, 2, 8, 8, 16, 16]`.
Python and native forward check real geometry and capacity, so changing metadata
on an old TP8 cache is insufficient.

Different kernels in one graph segment may use different core counts: its queue uses the
largest core domain, while each kernel retains its own `core_ids`.
Graph kernels use contiguous spans within `[0,N)`. Queue prefixes cover the
span; DMA-first/only segments use the owner's budget. DMA engine IDs must be in
`0..N-1`. DDR-to-DDR copies use a separate synchronous queue and release their
completed batch. This queue, immediate queues and retained queues have separate
lifetimes. Ordinary Graph submissions use the SDK batch envelope across four,
six, and eight cores (up to 65536 entries and the configured KD/instruction
buffer limits by default).
These limits do not change planner chunks or operator cores.

Residual rings reduce over the consumer count; partial producers clear all
consumer shards before writing their prefix. Pacing uses aligned bytes per
consumer and the 11 KiB crossover. Local-SPM, mixed residual, fixed-delivery
collectives and canonical SPM pipeline ingress retain their eight-core bounds.

Sources: [topology resolver](../python/rpu_backend/runtime/topology.py),
[Qwen3.5 adapter](../python/rpu_backend/adapters/qwen3_5/__init__.py),
[runtime configuration](runtime_config.md).
