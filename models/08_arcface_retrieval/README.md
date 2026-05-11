# Model 08 — ArcFace + Retrieval-Augmented Kinship

## Overview

Model 08 rebuilds the architectural contribution of Model 06 (non-parametric
gallery + cross-attention over retrieved supports) with a **face-specific
backbone** instead of a general-purpose self-supervised encoder.

Both Model 06's previous variants (R001 with ImageNet ViT-B/16, R002 with
DINOv2 ViT-B/14) capped at Test AUC ≈ 0.77-0.78. The hypothesis tested here:
**the retrieval bottleneck was the encoder, not the architecture.** Model 02's
results (0.85 AUC with ImageNet ViT + full fine-tune) and the face recognition
literature (ArcFace IResNet-100 → 99.83% LFW, 96.0% IJB-C) together suggest
that swapping the encoder for ArcFace — which is trained specifically to
discriminate identity — should improve retrieval quality because the gallery
recall will return pairs that *actually share genealogical features* rather
than pairs with similar lighting/pose/expression.

## Key changes vs Model 06

| Component | Model 06 | Model 08 |
|---|---|---|
| **Backbone** | ViT-B/16 (ImageNet) or DINOv2 ViT-B/14 | **IResNet-100 (ArcFace)** |
| Backbone pretrain | ImageNet-1K supervised / DINOv2 SSL on 142M | **MS1MV2 ArcFace loss on 5.8M faces / 85K identities** |
| Backbone params | 86M | 65M |
| Input resolution | 224×224 (timm) | 112×112 (ArcFace native) |
| **Face alignment** | None (raw FIW crops) | **MTCNN-aligned 112×112** (preprocessing once, cached on disk) |
| Backbone state | Frozen | Frozen |
| Retrieval architecture | Identical | Identical |
| Cross-attention | Identical | Identical |
| Heads (binary + relation) | Identical | Identical |

Everything else (gallery building, top-K retrieval, support encoding,
cross-attention layers, binary + relation heads, optimization recipe) is
copied verbatim from Model 06 R001. This isolates the encoder swap as the
single intervention.

## Why this is not a FaCoRNet copy

FaCoRNet (Wang et al., 2023) is **ArcFace + bidirectional cross-attention
between face A patches and face B patches**. It uses cross-attention as the
sole pair-interaction mechanism. Model 02 in this project replicates
FaCoRNet's design.

Model 08 uses cross-attention **between the query pair and K retrieved
training pairs**, not between face A and face B directly. The query pair's
representation is influenced by retrieving similar known-kin examples from a
gallery — a non-parametric memory mechanism that FaCoRNet does not have.

To the knowledge of this project's literature survey, the combination of
ArcFace + retrieval-augmented kinship verification is unpublished.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                Model 08 — ArcFace + Retrieval Kinship                  │
│                                                                        │
│   Input pair (Face A, Face B), 112×112 aligned via MTCNN               │
│                       │                                                │
│                       ▼                                                │
│   ┌───────────────────────────────────────────┐                        │
│   │ ArcFace IResNet-100 (FROZEN)              │                        │
│   │  pretrained MS1MV2                        │                        │
│   │  L2-normalized 512-dim embeddings         │                        │
│   └───────────────────┬───────────────────────┘                        │
│                       │ e_a, e_b ∈ ℝ^512                                │
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

Trainable parts (same as Model 06 R001):
- `sig_to_token` linear (1 M)
- relation embedding (12 × 512)
- retrieval cross-attention (~5 M)
- binary head + relation head (~0.2 M)
- Total: **~6.3 M trainable** (vs 65 M frozen backbone)

## Setup

### 1. Download ArcFace weights

The pretrained IResNet-100 weights from InsightFace (MS1MV2-trained) are
required. They are not committed to this repo (size ~250 MB).

```bash
# From the model root:
cd models/08_arcface_retrieval/weights/

# Option A: InsightFace official (MS1MV3, R100)
wget https://github.com/deepinsight/insightface/releases/download/v0.7/ms1mv3_arcface_r100_fp16.pth \
     -O arcface_r100.pth

# Option B: HuggingFace mirror (Glint360K, R100)
# huggingface-cli download deepinsight/insightface arcface_glint360k_r100.pth --local-dir .

# Verify:
ls -lh arcface_r100.pth   # expect ~250-500 MB depending on variant
```

The model loader expects the state dict to live at
`models/08_arcface_retrieval/weights/arcface_r100.pth`. Override with
`ARCFACE_WEIGHTS` env var in `run_pipeline.sh`.

### 2. Preprocess FIW with MTCNN

Run **once** before training to produce 112×112 aligned crops:

```bash
bash models/08_arcface_retrieval/AMD/run_preprocessing.sh
```

This builds a parallel `data/` directory under the model root, mirroring the
FIW structure but with each image replaced by its MTCNN-aligned 112×112 crop.
Expected runtime: ~2-3 hours single-GPU (or ~6h CPU). Cached on disk; runs
once, all subsequent training runs read from here.

### 3. Train

```bash
bash models/08_arcface_retrieval/AMD/run_pipeline.sh
```

Defaults mirror Model 06 R001 (LR 1e-4, 20 epochs, retrieval K=32). Override
via env vars same as Model 06.

## Hardware

| Setting | Value |
|---|---|
| GPU | AMD Radeon RX 6750 XT (12 GB VRAM) |
| Frozen backbone params | 65 M (IResNet-100) |
| Trainable params | ~6.3 M |
| Peak VRAM (batch 8) | ~6-7 GB (frozen → no Adam state on backbone) |
| Time per epoch (FIW) | ~12-15 min (faster than M06 — smaller input + smaller backbone) |

## Files

```
08_arcface_retrieval/
├── README.md                     (this file)
├── iresnet.py                    (IResNet architecture, from InsightFace MIT)
├── model.py                      (encoder + retrieval-augmented head)
├── RUN_LOG.md
├── weights/
│   └── arcface_r100.pth          (gitignored)
├── preprocessing/
│   ├── align_faces.py            (MTCNN preprocessing script)
│   └── transforms.py             (ArcFace input transforms)
├── data/                         (gitignored — aligned crops)
├── AMD/
│   ├── train.py
│   ├── test.py
│   ├── evaluate.py
│   ├── run_pipeline.sh
│   └── run_preprocessing.sh
├── Nvidia/                       (CUDA mirror, optional)
└── run-review/
    └── overview.md
```

## Expected results

| Metric | Model 06 R001 | Model 08 hypothesis |
|---|---:|---:|
| Test AUC | 0.776 | **0.83-0.86** |
| Test Accuracy | 69.8% | **74-76%** |
| TAR@FAR=0.01 | 0.062 | **0.15-0.20** |
| Per-relation min | 61.2% (gmgs) | **70-80%** (gmgs) |

If Model 08 matches or beats Model 02's 0.85 AUC, it confirms two things:
1. The retrieval architecture is competitive with FaCoR cross-attention when
   backbone is face-specific.
2. Model 06's previous ceiling at 0.77 was the encoder, not the architecture.

If Model 08 plateaus at 0.80-0.82, it suggests that retrieval-augmented
kinship has an intrinsic ceiling below FaCoRNet, regardless of backbone.

Either outcome is informative for the TCC.

## License

ArcFace weights and IResNet architecture are MIT-licensed (Deng et al.,
InsightFace project). MTCNN preprocessing uses `facenet-pytorch` (MIT, Tim
Esler).
