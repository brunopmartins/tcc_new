# Model 10 — AdaFace + FaCoR Cross-Attention

## Hypothesis

Model 02 (ViT-B/16 ImageNet + bidirectional FaCoR cross-attention + supervised
cosine contrastive loss, end-to-end fine-tune) is the strongest model in the
project so far (Test ROC-AUC = 0.850 on FIW). Its main architectural levers
were:

- A patch-token backbone, so cross-attention can operate at face-region
  granularity.
- A fully trainable backbone — kinship-specific features emerge during
  fine-tuning rather than being locked at the ImageNet pretrain.
- A face-aware cross-attention design that explicitly compares the two faces
  before pooling.

Model 08 (ArcFace IR-100 retrieval-augmented, frozen) gave a glimpse of what
face-discriminative pretraining can offer but kept the backbone frozen and
sometimes drifted into an "anti-kinship" regime (looking for *non*-kin
because that's what ArcFace was trained to do).

**M10 replaces M02's ImageNet ViT with AdaFace IR-101 (WebFace4M)** and keeps
everything else identical — including the end-to-end fine-tune. The bet is
that:

1. Starting from face-aware features should beat starting from ImageNet
   features.
2. Letting the backbone train (unlike M08) should let those features adapt
   away from the ArcFace/AdaFace verification objective and toward kinship.
3. The FaCoR cross-attention should focus the trainable budget on the
   regions that matter for kinship rather than identity.

## Architecture

```
   Face A (112×112)            Face B (112×112)
        ↓                            ↓
┌────────────────┐          ┌────────────────┐
│ AdaFace IR-101 │          │ AdaFace IR-101 │
│ (shared, ~65M, │          │ (shared, ~65M, │
│  trainable)    │          │  trainable)    │
└──────┬─────────┘          └─────┬──────────┘
       │  spatial map (B,512,7,7)│
       └─────────┬───────────────┘
                 ↓
   flatten + 49-token learnable pos-embed
                 ↓
        49 tokens × 512 dim per face
                 ↓
   ┌──────────────────────────────┐
   │ FaCoR Cross-Attention (×2)   │
   │  A → B  and  B → A           │
   └──────────────────────────────┘
                 ↓                    ┌───────────────────┐
       mean pool + Channel-SE         │ AdaFace pooled    │
                 │                    │ embedding (B,512) │
                 └──────── + ─────────┤ from output_layer │
                          (CLS-like)  └───────────────────┘
                          ↓
         Projection: Linear(512→512) + LN + ReLU + Drop + Linear
                          ↓
                    L2-normalised
                  emb1 (B,512), emb2 (B,512)
                          ↓
               cosine contrastive loss (m=0.3)
```

### Token extraction

AdaFace IR-101 normally returns a single 512-dim pooled embedding via its
`output_layer = BN → Dropout → Flatten → Linear(512·7·7 → 512) → BN1d`. M02's
FaCoR cross-attention expects per-region tokens. M10 therefore taps the
backbone at the *output of `self.body`* (i.e. just before `output_layer`):
the feature map there has shape `(B, 512, 7, 7)`. We flatten to
`(B, 49, 512)` row-major and add a learnable positional embedding to
preserve spatial layout, then feed into FaCoR.

The original 512-dim pooled embedding is still computed (it's cheap and the
weights are already trained) and summed into the per-face feature before
projection, acting as a CLS-token analogue.

### Key differences from M02

| Aspect            | M02 (ViT-B/16)                       | M10 (AdaFace IR-101)               |
|-------------------|--------------------------------------|------------------------------------|
| Backbone params   | ~86M                                 | ~65M                               |
| Pretrain set      | ImageNet-21K                         | WebFace4M (4M faces)               |
| Pretrain loss     | Classification (softmax)             | AdaFace margin loss                |
| Input             | 224×224, ImageNet mean/std            | 112×112, [-1, 1] (mean=std=0.5)    |
| Tokens            | 196 patch tokens (14×14) × 768       | 49 spatial tokens (7×7) × 512      |
| Global feature    | CLS token                            | AdaFace pooled `output_layer`(x)   |
| Positional info   | Built into ViT                       | Learnable 49-token PE (this model) |

### Key differences from M08

| Aspect          | M08 (ArcFace, frozen)                | M10 (AdaFace, full FT)             |
|-----------------|--------------------------------------|------------------------------------|
| Backbone        | InsightFace iresnet100 (BasicBlock)  | cvlface AdaFace IR-101 (BasicBlockIR) |
| Pretrain        | MS1MV2/Glint360K (ArcFace)           | WebFace4M (AdaFace)                |
| Training        | Backbone frozen                      | End-to-end                          |
| Auxiliary       | Retrieval gallery + cross-attn       | None — pure FaCoR pair attention   |
| Loss            | BCE + (optional) cosine contrastive  | Cosine contrastive (M02 recipe)    |

## Files

```
10_adaface_facor/
├── README.md                       # This file
├── adaface_iresnet.py              # Vendored AdaFace IR-101 backbone
├── model.py                        # M10 main model (FaCoR + AdaFace)
├── AMD/
│   ├── train.py
│   ├── test.py
│   ├── evaluate.py
│   └── run_pipeline.sh             # End-to-end pipeline runner
├── weights/                        # AdaFace .pth (gitignored)
│   ├── README.md
│   └── adaface_ir101_webface4m.pth
├── data/                           # Optional per-model staging (gitignored)
└── run-review/                     # Per-run analysis notes (manual)
```

## Quick start

```bash
# 1. Make sure weights are downloaded (see weights/README.md)
ls models/10_adaface_facor/weights/adaface_ir101_webface4m.pth

# 2. (Optional) use existing MTCNN-aligned FIW crops at 224x224 — train.py
#    resizes to 112×112 automatically.
export ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned

# 3. Run the pipeline
bash models/10_adaface_facor/AMD/run_pipeline.sh
```

Default hyperparameters mirror M02's best recipe:

| Knob              | Value                |
|-------------------|----------------------|
| `LEARNING_RATE`   | 5e-6 (peak)          |
| `WARMUP_EPOCHS`   | 5                    |
| `DROPOUT`         | 0.2                  |
| `LOSS`            | cosine_contrastive   |
| `TEMPERATURE`     | 0.3                  |
| `MARGIN`          | 0.3                  |
| `BATCH_SIZE`      | 8                    |
| `GRAD_ACCUM`      | 4 (effective 32)     |

## Smoke test

From the model root:

```bash
python model.py
```

Expected output:

```
emb1 shape:     (2, 512)
emb2 shape:     (2, 512)
attn_map shape: (2, 8, 49, 49)
params total/trainable: 72,039,872 / 72,039,872 (100.00%)
classifier logits shape: (2, 1)
```

## VRAM budget (AMD RX 6750 XT, 12 GB)

| Component             | Size            |
|-----------------------|-----------------|
| AdaFace IR-101 params | 65.2M           |
| FaCoR + heads         | ~6.9M           |
| Total trainable       | ~72.0M          |
| Adam state            | 2× weights      |
| Activations (b=8)     | ~3 GB           |

Total fit at batch=8 + AMP is around ~10 GB. Use `GRAD_ACCUM=4` to reach
effective batch 32, matching M02's tuned recipe.
