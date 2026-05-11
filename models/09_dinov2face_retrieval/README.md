# Model 09 — DINOv2-Face + Retrieval-Augmented Kinship

## Overview

Model 09 tests whether a face encoder that was specialized through a
**self-supervised** objective preserves kinship signal when used frozen,
in contrast to Model 08 which used the **identity-discriminative** ArcFace
encoder and produced the worst Test AUC of the project (0.693).

The intuition (formalized in Model 08's run-review): kinship verification
depends on two signals,

- **(a) facial similarity** — "do these two faces share visible traits?"
- **(b) absence of identity** — "these are not the same person."

ArcFace is state-of-the-art at (a) but explicitly trained against (b) — its
margin loss pushes images of different people (which include parents,
children, siblings) into different regions of the embedding space. When
ArcFace is used frozen, the head cannot undo this anti-kinship pressure.

DINOv2 (Liu et al. 2024) was pretrained without identity labels, on a
mixture of natural images via a self-supervised teacher-student contrastive
objective. **DINOv2-Face** refers to variants of DINOv2 further trained on
face data using the same self-supervised pretext (no identity discrimination).
Such an encoder retains face specialization without inheriting the anti-kinship
inductive bias.

## Status of the DINOv2-Face checkpoint

A targeted HuggingFace search (2026-05-10, terms: `dinov2 face`,
`dinov2 vggface`, `FRoundation`, `facedino`, `DINOv2-Face`,
`face foundation`, `face self-supervised`, `face encoder ssl`,
`face pretrained vit`) did **not** surface a public weight matching the
naming convention. The original FRoundation paper (Liu et al. 2024) does
not yet have a public release on HuggingFace under the expected names.

We therefore **default to base DINOv2** (`vit_base_patch14_dinov2.lvd142m`
via timm) and provide a clean overlay mechanism for the moment a DINOv2-Face
state_dict becomes available. See `weights/README.md` for the candidates
that were inspected and the loader contract.

Without the face fine-tune, Model 09 effectively reproduces M06 R002 (DINOv2
frozen, Test AUC ≈ 0.77-0.78) with the M06/M08 retrieval architecture. That
is still a useful comparison datapoint — it isolates the "DINOv2 vs ArcFace"
contrast inside an otherwise-identical retrieval pipeline.

## Key changes vs Model 08

| Component | Model 08 | Model 09 |
|---|---|---|
| **Backbone** | IResNet-100 (ArcFace) | **DINOv2 ViT-B/14 (optional face overlay)** |
| Backbone pretrain | MS1MV2 ArcFace loss | **DINOv2 SSL on LVD-142M (+ face overlay)** |
| Backbone params | 65 M | **86 M** |
| **Input resolution** | 112×112 | **224×224** (DINOv2 native) |
| **Normalization** | [-1, 1] ArcFace | **ImageNet mean/std** |
| Face alignment | MTCNN-aligned 112×112 (required) | Works with raw FIW or pre-aligned 224×224 |
| Backbone state | Frozen | Frozen |
| Pair embedding | 512-d ArcFace native | **512-d projection of 768-d CLS token** |
| Retrieval architecture | Identical | Identical |
| Cross-attention | Identical | Identical |
| Heads (binary + relation) | Identical | Identical |

Everything in the retrieval-augmented head (gallery building, top-K retrieval,
support encoding, cross-attention layers, binary + relation heads,
optimization recipe) is copied verbatim from M08 / M06 R001. This isolates
the encoder swap as the single intervention.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│              Model 09 — DINOv2-Face + Retrieval Kinship                │
│                                                                        │
│   Input pair (Face A, Face B), 224×224, ImageNet-normalized            │
│                       │                                                │
│                       ▼                                                │
│   ┌───────────────────────────────────────────┐                        │
│   │ DINOv2 ViT-B/14 (FROZEN)                  │                        │
│   │  base: timm vit_base_patch14_dinov2.lvd142m │                      │
│   │  optional overlay: DINOv2-Face state_dict  │                       │
│   │  CLS token (768-d)                         │                       │
│   └───────────────────┬───────────────────────┘                        │
│                       │ f ∈ ℝ^768                                      │
│                       ▼                                                │
│   Projection: 768 → 512, L2-normalised → e_a, e_b ∈ ℝ^512              │
│                       │                                                │
│                       ▼                                                │
│   pair signature s_q = norm([e_a + e_b, |e_a - e_b|, e_a * e_b])       │
│                       │                                                │
│                       │ cosine top-K from gallery (33k positive pairs) │
│                       ▼                                                │
│   K support tokens = sig_to_token(s_i) + relation_embed(r_i)           │
│                       │                                                │
│                       ▼                                                │
│   Cross-attention (2 layers, 4 heads): query ↔ supports                │
│                       │                                                │
│                       ▼                                                │
│   Binary head (kin) + Relation head (11-class auxiliary)               │
└────────────────────────────────────────────────────────────────────────┘
```

Trainable parts:
- 768→512 projection (~1.0 M)
- `sig_to_token` linear (~1.0 M)
- relation embedding (12 × 512)
- retrieval cross-attention (~5 M)
- binary head + relation head (~0.2 M)
- Total: **~7.2 M trainable** (vs 86 M frozen backbone)

## Setup

### 1. (Optional) Download DINOv2-Face weights

If a DINOv2-Face checkpoint becomes available, put it under
`weights/dinov2_face.pth` and set `DINOV2_WEIGHTS=...` when launching
`run_pipeline.sh`. See `weights/README.md` for the loader's expected key
naming convention and the candidate repos that were surveyed.

If you don't have face-specific weights, leave `DINOV2_WEIGHTS=""` — the
pipeline will use base DINOv2 directly from timm.

### 2. (Optional) Pre-aligned face crops

The project already provides `datasets/FIW_aligned` (224×224 MTCNN-aligned
crops). Use it to skip on-the-fly resizing:

```bash
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
  bash models/09_dinov2face_retrieval/AMD/run_pipeline.sh
```

DINOv2's patch size 14 requires `img_size` divisible by 14 (224, 238, 252...).
The default 224 matches the aligned crops natively.

### 3. Train

```bash
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
  SKIP_INSTALL=1 EPOCHS=20 \
  bash models/09_dinov2face_retrieval/AMD/run_pipeline.sh
```

Defaults mirror Model 06 / 08 (LR 1e-4, 20 epochs, retrieval K=32, BCE +
0.3 contrastive + 0.15 relation-CE). Override via env vars listed in the
script header.

## Hardware

| Setting | Value |
|---|---|
| GPU | AMD Radeon RX 6750 XT (12 GB VRAM) |
| Frozen backbone params | 86 M (DINOv2 ViT-B/14) |
| Trainable params | ~7.2 M |
| Peak VRAM (batch 8) | ~7-8 GB (frozen → no Adam state on backbone) |
| Time per epoch (FIW) | ~15-20 min (similar to M06 R002) |

## Files

```
09_dinov2face_retrieval/
├── README.md                     (this file)
├── model.py                      (DINOv2FaceEncoder + retrieval head)
├── weights/
│   ├── README.md                 (DINOv2-Face sources & loader contract)
│   └── .gitignore                (ignore .pth/.pt/.bin)
├── data/                         (gitignored — optional preprocessed cache)
├── AMD/
│   ├── train.py
│   ├── test.py
│   ├── evaluate.py
│   └── run_pipeline.sh
├── Nvidia/                       (CUDA mirror, optional)
└── run-review/
    ├── overview.md               (run summary)
    └── RUN_LOG.md                (template)
```

## Expected results

| Metric | Model 08 (ArcFace) | Model 06 R002 (DINOv2) | **Model 09 hypothesis** |
|---|---:|---:|---:|
| Test AUC | 0.693 | ~0.77 | **0.78-0.83** |
| Test Accuracy | 60.8% | ~69% | **70-74%** |
| TAR@FAR=0.01 | 0.100 | ~0.06 | **0.10-0.15** |

If Model 09 (base DINOv2, no face overlay) lands at ~0.77, it confirms that
the encoder choice — not identity-pressure — was the M06 ceiling. If a
DINOv2-Face overlay pushes it above 0.80, it confirms that *self-supervised
face specialization* yields the productive middle ground that ArcFace
frozen does not.

If Model 09 plateaus at or below M06 R001 (0.776) even with the face
overlay, the retrieval bottleneck is the architecture itself, not the
encoder family.

Either outcome is informative for the TCC.

## License

DINOv2 base weights are released by Facebook AI Research under the
Apache 2.0 / CC-BY-NC license (per the timm card). Any DINOv2-Face overlay
should carry its own license; check `weights/README.md` and the source
repo before redistribution.
