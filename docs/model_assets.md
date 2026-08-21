# Model assets

RhinoForge does not redistribute model checkpoints. Obtain each checkpoint from
the model owner or through an authorized asset channel, accept its separate
terms, and verify the exact release manifest before use.

This page describes the asset contract for the source candidate. Exact
checkpoint revisions, checkpoint hashes, and TOML hashes are intentionally not
invented here: they must be populated from the immutable release candidate.
Until a row has those values in a published release manifest, its planned
`Supported` or `Limited` status is validation pending.

## Required record for a runnable profile

Every runnable profile must publish all of these fields together:

- model family and exact profile;
- source identifier and immutable revision;
- model license and access conditions;
- source or quantized checkpoint format;
- conversion command and converter version, when conversion is required;
- checkpoint SHA256;
- matching example TOML and its SHA256;
- compatible RhinoForge version, Rhino Launch version, and combined operator
  asset SHA256.

Do not combine values from different release manifests.

## Candidate asset sources

| Model/profile | Model source | Access and license | Format / conversion | Candidate manifest |
|---|---|---|---|---|
| Qwen3 0.6B / 1.7B / 4B / 8B FP16 | `Qwen/Qwen3-{0.6B,1.7B,4B,8B}` | Model repository terms; checkpoint not redistributed here | Original Hugging Face checkpoint | Revision, checkpoint SHA256, and TOML SHA256 pending |
| Qwen3 0.6B / 1.7B / 4B / 8B W8A16 | Corresponding Qwen source checkpoint | Model repository terms; checkpoint not redistributed here | Source-only converter output; no runnable release manifest | Not applicable until an exact quantized profile is promoted |
| Qwen3 14B W8A16 | `Qwen/Qwen3-14B` | Model repository terms; checkpoint not redistributed here | Convert with `rpu_backend.quant.convert_qwen3 --quant-lm-head`; embeddings remain FP16 | Source revision, converted-checkpoint SHA256, and TOML SHA256 pending |
| Qwen3 32B | `Qwen/Qwen3-32B` | Model repository terms | Source only; no runnable asset | Not applicable until a supported profile exists |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B` | Model-owner license and access approval required | Original Hugging Face checkpoint | Revision, checkpoint SHA256, and TOML SHA256 pending |
| Qwen3.5 text and vision | Model-owner official distribution | Exact source identifier and terms must be fixed by the release | Original checkpoint unless a release row states otherwise | Source identifier, revision, hashes, and TOML pending |
| Qwen3-VL 2B / 4B | `Qwen/Qwen3-VL-{2B,4B}-Instruct` | Model repository terms; checkpoint not redistributed here | Original Hugging Face checkpoint | Revision, checkpoint SHA256, and TOML SHA256 pending |
| Qwen3-VL 32B W8A16 | Model-owner checkpoint plus release-specific converted asset | Controlled asset terms | Release-specific W8A16 asset; no public conversion command is declared | Exact source, conversion provenance, hashes, and TOML pending |
| Gemma4-E4B text | Model-owner official distribution | Exact repository terms must be fixed by a later release | Source only | Compatible runtime dependency set, source identifier, revision, and hashes pending |
| Pi0.5 Libero | Model-owner official distribution or authorized supplied asset | Model and dataset terms apply separately | FP16 original; W8A16 via `rpu_backend.quant.convert_pi05` | Exact source, revision, hashes, and TOML pending |
| Wall-OSS-0.5 | Model-owner official distribution or authorized supplied asset | Model terms apply separately | FP16 original; W8A16 via `rpu_backend.quant.convert_wall_oss_w8a16`; W4 via `rpu_backend.quant.convert_wall_oss` | Exact source, revision, hashes, and TOML pending |
| Hy-Embodied-0.5-VLA | Authorized supplied asset | Asset terms apply separately | Source checkpoint; supported precision is selected by the exact profile | Checkpoint and TOML hashes pending; any `norm_stats.pkl` requires an exact caller-supplied SHA-256 and explicit pickle trust opt-in |
| DINOv3 ViT-B | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Model repository terms; checkpoint not redistributed here | Original Hugging Face checkpoint | Revision, checkpoint SHA256, and TOML SHA256 pending |
| SigLIP | Model-owner source selected by the consuming model profile | Source-model terms apply | Component asset; format follows the consuming profile | Exact source and hashes pending |
| GR00T-N1.7-3B | Authorized supplied asset | Asset terms apply separately | Source only | Public asset record not yet available |
| RhinoVLA | [HUIXI-AI/RhinoVLA](https://github.com/HUIXI-AI/RhinoVLA) plus its model assets | Repository and checkpoint terms apply separately | Source only; runtime-factory and checkpoint-schema alignment pending | Public end-to-end asset record not yet available |
| LingBot-VLA-V2 exact-Z2 W8A16 | `robbyant/lingbot-vla-v2-6b` plus the exact converted asset | Model repository and converted-asset terms apply | Controlled W8A16 profile | Revision, conversion provenance, hashes, and TOML pending |
| Galaxea G0.5 exact K1 RTC/FM W0 | Authorized supplied asset | Asset terms apply separately | Controlled exact-profile asset | Checkpoint and TOML hashes pending |
| InternVLA-N1 + NavDP | Caller-provided assets with a pinned manifest | Asset terms apply separately | Controlled exact asset set | Full caller-pinned manifest required |

`Source-only`, `Experimental`, and `Component-only` entries may legitimately have
no generally downloadable end-to-end checkpoint. In that case the release must
say `source only` or `authorized supplied asset`; it must not present a generic
download command as a runnable path.

## Download pattern

For a release row that names a Hugging Face source and immutable revision:

```bash
export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
hf download SOURCE_ID \
  --revision REVISION \
  --local-dir "$RPU_MODEL_CACHE/LOCAL_MODEL_DIRECTORY"
```

Use the source identifier, revision, directory name, and expected SHA256 from
the same release row. Access to a private or gated model must be granted by the
model owner.

## Existing offline converters

Run converters only for a profile that explicitly names the corresponding
output format.

```bash
# Qwen3 14B release-profile candidate
python -m rpu_backend.quant.convert_qwen3 \
  --src SOURCE_DIR --dst OUTPUT_DIR --quant-lm-head

# Pi0.5 W8A16
python -m rpu_backend.quant.convert_pi05 --src SOURCE_DIR --dst OUTPUT_DIR

# Pi0.5 packed W4 evaluation profile
python -m rpu_backend.quant.convert_pi05 \
  --src SOURCE_DIR --dst OUTPUT_DIR --fake-w4 --real-w4

# Wall-OSS W8A16
python -m rpu_backend.quant.convert_wall_oss_w8a16 \
  --src SOURCE_DIR --dst OUTPUT_DIR

# Wall-OSS group-wise W4 evaluation profile
python -m rpu_backend.quant.convert_wall_oss \
  --src SOURCE_DIR --dst OUTPUT_DIR --group-size 32
```

Converters create a new destination and reject an existing output directory.
After conversion, hash the complete checkpoint artifact as specified by the
release manifest and point the matching TOML at that output.

## Runtime assets are separate

Model assets do not replace the two restricted runtime prerequisites:

- Rhino Launch 1.0.0 binary development package.
- One combined `rhinoOpLib_current.ref`, selected through
  `RPU_KERNEL_LIB_PATH`.

Neither asset is stored in this repository, a source archive, or a Python
package. Obtain both through the authorized distribution channel and use the
versions and SHA256 values listed for the same RhinoForge release. Follow
[Restricted runtime assets](runtime_assets.md) for authorization, compatibility,
signature, checksum, installation, and revocation handling.
