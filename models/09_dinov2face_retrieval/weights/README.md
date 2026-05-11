# DINOv2-Face weights

This directory holds an optional **DINOv2-Face** state_dict that overlays the
base DINOv2 ViT-B/14 in Model 09. The file is gitignored because it would be
~330-600 MB.

## Loader contract

`DINOv2FaceEncoder` first instantiates `vit_base_patch14_dinov2.lvd142m` via
timm (downloads the base DINOv2 weights from HuggingFace's `timm` repo on
first use), then if `--dinov2_weights /path/to/file.pth` is provided, it
loads the file and `load_state_dict(..., strict=False)` on top.

Accepted formats:

- Raw state_dict: keys directly match timm's internal names —
  `cls_token`, `pos_embed`, `patch_embed.proj.weight`,
  `blocks.0.norm1.weight`, ..., `norm.weight`, ...
- `{"state_dict": {...}}` — common Lightning / HuggingFace export.
- `{"model": {...}}` — common DINO / DINOv2 export.
- Prefixes `module.`, `backbone.`, `vit.`, `encoder.`, `teacher.`,
  `student.` are stripped automatically.

Mismatched keys are printed but tolerated. Add `--strict_load` to fail on
any missing / unexpected keys.

## How to obtain (status: 2026-05-10)

A targeted HuggingFace search for face-specialized DINOv2 weights was
performed using the following queries:

| Query | Result |
|---|---|
| `dinov2 face` | Only `facebook/dinov2-*` and `facebook/dpt-dinov2-*` (no face variant) |
| `dinov2 vggface` | (no results) |
| `DINOv2-Face` | (no results) |
| `face dinov2` | Same as `dinov2 face` |
| `FRoundation` | (no results) |
| `facedino` | (no results) |
| `face foundation` | (no results) |
| `face self-supervised` | (no results) |
| `face encoder ssl` | (no results) |
| `selfsup face` | (no results) |
| `face pretrained vit` | (no results) |
| `dinov2 fine-tune face` | (no results) |
| `vit face ssl` | (no results) |
| `dino face pretrained` | Only base `facebook/dino-*` (no face variant) |
| `MAE face` | Only generic ViT-MAE |

**No public DINOv2-Face weight matching the loader contract was found** as
of the date above. The paper "FRoundation: A face foundation model with
self-supervised learning" (Liu et al. 2024) describes such a model, but its
release on HuggingFace under common names was not located by these queries.

### Backup options surveyed

These face encoders were discovered but **do not match the timm DINOv2
state_dict layout** — they would require custom loaders, not just an
overlay:

| Repo | Architecture | Notes |
|---|---|---|
| `Green-Sky/FaRL-Base-Patch16-LAIONFace20M-ep64` | CLIP ViT-B/16 (image + text) | Multi-modal, key names follow `transformers` CLIP, not timm DINOv2 |
| `kartiknarayan/facexformer` | FaceXFormer (multi-task) | Domain-specific (parsing, landmark, etc.); transformers AutoModel |
| `minchul/cvlface_adaface_vit_base_kprpe_webface4m` | AdaFace ViT-B + KPRPE on WebFace-4M | Margin-loss trained → anti-kinship (same failure mode as M08) |
| `minchul/cvlface_adaface_vit_base_webface4m` | AdaFace ViT-B on WebFace-4M | Same as above |
| `kunkunlin1221/face-recognition_vit-l-pfc0.3-cosface-web42m` | CosFace ViT-L on WebFace-42M | Margin-loss → anti-kinship |

Of those, **FaRL** is the only one trained without explicit identity
margin loss (it uses CLIP image-text contrastive on faces). It is therefore
the most natural backup if DINOv2-Face proves unavailable. However, plugging
it into Model 09's loader requires converting the CLIP vision-encoder
state_dict to timm's ViT-B/16 naming — not a one-line drop-in.

### How to add weights if you find them

```bash
cd models/09_dinov2face_retrieval/weights/

# example placeholder — replace with your actual download command
huggingface-cli download <some-org>/dinov2-face-base dinov2_face_base.pth \
    --local-dir . --local-dir-use-symlinks=False
mv dinov2_face_base.pth dinov2_face.pth

# verify it loads:
python -c "
import sys; sys.path.insert(0, 'models/09_dinov2face_retrieval')
from model import DINOv2FaceEncoder
enc = DINOv2FaceEncoder(weights_path='models/09_dinov2face_retrieval/weights/dinov2_face.pth')
print('OK')
"
```

Then run:

```bash
DINOV2_WEIGHTS=models/09_dinov2face_retrieval/weights/dinov2_face.pth \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
SKIP_INSTALL=1 \
bash models/09_dinov2face_retrieval/AMD/run_pipeline.sh
```

## Fallback: base DINOv2

If no DINOv2-Face overlay is provided (`DINOV2_WEIGHTS=""`, the default),
the pipeline uses the base `vit_base_patch14_dinov2.lvd142m` weights from
timm / HuggingFace directly. These will be downloaded automatically on
first use (~340 MB into the timm/HF cache).

That fallback is essentially **M06 R002 (DINOv2 frozen)** plugged into the
M08 retrieval architecture — useful as a clean comparison datapoint.

## Variants comparison (face encoders surveyed)

| Encoder | Pretrain set | Objective | Anti-kinship? | Suitable for M09? |
|---|---|---|---|---|
| base DINOv2 ViT-B/14 (`vit_base_patch14_dinov2.lvd142m`) | LVD-142M (natural images, no faces) | DINO SSL | No | Yes (fallback) |
| **DINOv2-Face** (target) | DINOv2 → face fine-tune via DINO/contrastive | DINO SSL on faces | **No** | **Yes (preferred)** |
| FaRL (CLIP-based) | LAION-Face-20M | CLIP image-text contrastive | No | Possible (requires custom adapter) |
| ArcFace (M08) | MS1MV2/MV3, Glint360K | Angular margin (identity sep.) | **YES — confirmed by M08** | No |
| AdaFace, CosFace (face SOTA) | WebFace-4M/12M/42M | Angular margin | **YES** | No |
| ViT-MSN, ViT-MAE | ImageNet (not faces) | SSL on natural images | No | Possible — but not face-specialized |

## License

Base DINOv2 weights: Apache 2.0 / CC-BY-NC (per Facebook AI Research's
release on HuggingFace). Any overlay should ship with its own license — for
academic TCC use, single-machine training within Brazilian university
research is generally permitted under both DINOv2 license clauses, but
verify with the upstream source before redistribution.
