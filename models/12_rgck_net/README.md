# Model 12 — RGCK-Net (Region-Guided Cross Kinship Network)

Implementation of the proposal in
[`proposta_rgck_net_kinship.md`](../../proposta_rgck_net_kinship.md).

## Architecture

```
aligned 224×224 face
    │
    ├── crop 5 anatomical regions (global/eyes/nose/mouth/jaw)
    │   each region → resize 112×112 → AdaFace IR-101 (shared, frozen)
    │
    └─► per face: K=5 tokens × 512-d

(B, K, 512) tokens_A         (B, K, 512) tokens_B
       │                            │
       └──── Cross-Region Adapter ──┘     (bidirectional MultiHeadAttention,
              ↓        ↓                   1 layer × 4 heads × 512-d)
       contextualised  contextualised
        tokens_A_ctx    tokens_B_ctx
              │             │
              ├── per-region cosine similarities (B, K)
              └── Regional Gate (sigmoid weights, B, K)
                                │
                       weighted regional score (B, 1)
                                │
        Fusion: [gA, gB, |gA-gB|, gA*gB, sims, weights, score]
                                │
                       MLP classifier → kinship logit
```

Phase 1 recipe (per proposal section 30, 37, 38):

| Component | Value |
|---|---|
| Backbone | AdaFace IR-101 (WebFace4M), **frozen** |
| Input | 224×224 aligned (cropped/resized inside the model) |
| Regions | 5 fixed boxes: global, eyes, nose, mouth, jaw |
| Cross-attn | 1 bidirectional layer, 4 heads, 512-d |
| Regional gate | MLP over `[rA, rB, |rA-rB|, rA*rB]` → sigmoid |
| Classifier head | 3-layer MLP with BatchNorm + GELU + dropout |
| Loss | BCE on classifier logit (Phase 1, no aux losses yet) |
| LR | 1e-4 (head-only, backbone frozen) |
| Scheduler | cosine, warmup 5, min_lr 1e-6 |
| Optimizer | AdamW, weight_decay 1e-4 |
| Batch | 8 × grad-accum 4 (eff 32) |
| Dropout | 0.2 |

Trainable params: ~5.6 M / 70.7 M total (7.9 %). Backbone frozen
means only the cross-region adapter + gate + classifier are updated.

## What's NEW vs M09/M10/M11

- **Region tokens with anatomical structure** — not patch tokens (49 or
  196 from spatial grid). 5-7 tokens per face, each tied to a face
  component (eyes, nose, etc.).
- **Regional gating with per-region weights** — interpretable; can
  inspect which regions matter for each pair / each relation type.
- **Lightweight cross-attention** — K² = 25 attention scores per layer
  vs 49² or 196² in M09/M10/M11. ~40-100× cheaper at the cross-attn
  step.
- **Frozen backbone by default** — avoids the AdaFace identity-cluster
  fine-tune instabilities (M10 R002) and family-memorisation risks
  (M09 R001's val→test gap -0.094).

## Quick start

### R001 — Phase 1 (frozen backbone, BCE only)

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 GRAD_ACCUM=4 \
NUM_WORKERS=4 SEED=42 \
bash models/12_rgck_net/AMD/run_pipeline.sh
```

This is the baseline. If it lands at Test AUC > 0.78 the architecture
is validated. If Test per-relation accuracies on grandparent classes
land above 50 % (vs M09 R001's 31-37 %), the region-token approach is
working.

### Future runs (proposal phases 2-6)

- **Phase 2** (`UNFREEZE_BACKBONE=1` — TBD): unfreeze last stage of IR-101
- **Phase 3**: full fine-tune
- **Phase 4**: + supervised contrastive loss (λ=0.05)
- **Phase 5**: + relation-type auxiliary head
- **Phase 6**: hard negatives (`TRAIN_NEGATIVE_STRATEGY=relation_matched`)

Most of these are env-var changes; relation-aux head needs a model.py
extension.

## Comparison with prior models

| Model | Backbone | Cross-attn over | Trainable params |
|---|---|---|---:|
| M02 R031 | ViT-B/16 ImageNet (full FT) | 196 patches | 86 M |
| M09 R001 | AdaFace IR-101 (full FT) | 196 + 49 patches (multistage) | 70.9 M |
| M10 R003 | AdaFace IR-101 (full FT) | 49 patches (top-only) | 72.6 M |
| M11 R001 v4 | AdaFace IR-101 (full FT) | 196 + 49 (multistage) | 70.9 M |
| **M12 R001** | **AdaFace IR-101 (frozen)** | **5 region tokens** | **5.6 M (7.9 %)** |

M12 is the only model where the backbone is **frozen** by default. All
prior models needed full-FT to achieve their best numbers (M09 R001
Test AUC 0.7982, M02 R031 Test AUC 0.850).

## Files

```
12_rgck_net/
├── README.md
├── model.py                  # RGCKNet, FixedPartitionRegionTokenizer, CrossRegionAdapter, RegionalGate
├── adaface_iresnet.py        # Symlink to M10's
├── AMD/
│   ├── train.py
│   ├── test.py
│   └── run_pipeline.sh
├── weights/                  # Symlinked AdaFace .pth
├── data/                     # (gitignored)
└── run-review/               # Per-run analyses
```

## Reference

- Proposal: `proposta_rgck_net_kinship.md` (project root)
- AdaFace: Kim et al., CVPR 2022, "AdaFace: Quality Adaptive Margin for Face Recognition"
- FaCoRNet: Su et al., ICCV Workshops 2023, "Kinship Representation Learning with Face Componential Relation"
- CI³Former: IEEE TCSVT 2025, "Cross-Image Information Interaction Network for Kinship Verification"
