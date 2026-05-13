# Model 11 — AdaFace IR-101 + FaCoRNet Recipe (full)

## Motivation

M10 and M02 already implement the **FaCoR cross-attention module**
(bidirectional Q1·K2 / Q2·K1 with shared FFN) on top of their respective
backbones. But the FaCoRNet paper bundles that module with a *specific
training recipe* — a contrastive loss whose temperature is dynamically
modulated by the cross-attention map, plus hard-negative mining via
relation-matched sampling, plus a face-discriminative backbone (AdaFace
IR-101).

M11 isolates **the rest of the FaCoRNet recipe** on top of M10's
architecture, so we can attribute future Test-AUC differences to the
loss + sampling stack rather than to the cross-attention module itself.

The hypothesis being tested:

> **Does the full FaCoRNet recipe (relation-guided loss + relation-matched
> negatives) on M10's architecture produce a different val→test
> generalisation profile than M10 R003's BCE classifier head?**

M10 R003 used a BCE classifier head and got Val AUC 0.8875 / Test AUC
0.7478 (-0.140 gap). M11 R001 is set up to answer whether the
attention-driven contrastive loss + hard negatives close that gap.

## Recipe (what changes vs M10)

| Component                    | M10 R003                | **M11 R001**                                      |
|------------------------------|-------------------------|---------------------------------------------------|
| Backbone                     | AdaFace IR-101 full FT  | **Same**                                          |
| Cross-attention              | FaCoR, top-only, 2L×8H  | **Same**                                          |
| Positional embedding         | Learnable 49 tokens     | **Same**                                          |
| Global embedding             | AdaFace pool            | **Same**                                          |
| **Loss**                     | bce (classifier head)   | **`relation_guided`** (attention-driven contrastive) |
| **Base temperature**         | 0.3                     | **0.07** (FaCoRNet base)                          |
| **Train neg. sampling**      | random                  | **`relation_matched`** (hard negatives)           |
| **Eval neg. sampling**       | random                  | **`relation_matched`**                            |
| Classifier head              | enabled (MLP on diff/prod) | **disabled** (embeddings consumed directly)    |
| LR (peak)                    | 5e-6                    | Same                                              |
| Warmup                       | 5 epochs                | Same                                              |
| Dropout                      | 0.2                     | Same                                              |
| Batch                        | 8 × grad-accum 4 (32)   | Same                                              |

### What `relation_guided` loss does

`RelationGuidedContrastiveLoss` is implemented in
`models/shared/losses.py`. Given:

- L2-normalised embeddings `(emb1, emb2)` for each pair,
- The cross-attention map `attn_map` of shape `(B, heads, T, T)`,

it computes a supervised contrastive loss where the temperature for each
sample is `base_temperature × (1 + α × attn_intensity)`, with
`attn_intensity = attn_map.mean(dim=[heads, T_q, T_k])`.

In effect, **high-attention pairs (the model "focused" on something) use
a higher temperature** → smoother gradients, less aggressive pulling
together of positives. **Low-attention pairs use a lower temperature** →
sharper, more aggressive separation when the model "didn't engage". This
echoes FaCoRNet's claim that attention-driven dynamic temperature helps
the model calibrate its confidence per-pair.

### What `relation_matched` negative sampling does

For a positive pair of relation type R (e.g. `fs`), the negative for
that batch position is constructed from a *different fs pair* — so the
two faces look kinship-like in the same role (father/son) but are not
genuinely related. This is the canonical hard-negative strategy used
in kinship verification literature.

This is implemented in `models/shared/dataset.py` via
`_sample_fiw_relation_matched_negatives()` and exposed through
`--train_negative_strategy relation_matched`.

## Differences from M09

M09 also uses AdaFace IR-101, but **modifies the architecture** by
injecting cross-attention after stages 3 and 4 of the backbone (SAI-style
inside-the-backbone interaction). M11 leaves the architecture alone (top-
only) and **modifies the training recipe** instead. They test orthogonal
hypotheses and together with M10 R003 form a three-way comparison:

| Model       | Architecture                | Training recipe          |
|-------------|-----------------------------|--------------------------|
| M10 R003    | FaCoR top-only              | BCE + random negatives   |
| M09 R001    | **Multi-stage cross-attn**  | BCE + random negatives   |
| **M11 R001**| FaCoR top-only              | **FaCoRNet (relation_guided + relation_matched)** |

## Two run variants planned

M11 supports **two architecture variants** sharing the FaCoRNet recipe.
The variant is selected at launch time via `USE_MULTISTAGE`; the
checkpoint records the choice so `test.py` / `evaluate.py` rebuild the
right model automatically.

### R001 — top-only (default, M10 architecture)

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=8 GRAD_ACCUM=4 \
NUM_WORKERS=4 SEED=42 \
bash models/11_facornet_adaface/AMD/run_pipeline.sh
```

Isolates **the recipe** (loss + sampling + temperature) on top of M10's
architecture. Direct ablation against M10 R003.

### R002 — multi-stage (M09 architecture)

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=4 GRAD_ACCUM=8 \
USE_MULTISTAGE=1 \
NUM_WORKERS=4 SEED=42 \
bash models/11_facornet_adaface/AMD/run_pipeline.sh
```

Combines M09's inside-the-backbone interaction with the FaCoRNet recipe.
Direct ablation against M09 R001. M11/AMD/train.py imports
`build_adaface_multistage_model` from `models/09_adaface_multistage/`
when this flag is set — no code duplication.

### Other ablations

```bash
# Same recipe but with random negatives instead of relation_matched
TRAIN_NEGATIVE_STRATEGY=random EVAL_NEGATIVE_STRATEGY=random \
    bash models/11_facornet_adaface/AMD/run_pipeline.sh

# Higher temperature (M10's base 0.3)
TEMPERATURE=0.3 \
    bash models/11_facornet_adaface/AMD/run_pipeline.sh

# Cosine_contrastive instead of relation_guided (isolate sampling effect)
LOSS=cosine_contrastive \
    bash models/11_facornet_adaface/AMD/run_pipeline.sh
```

## 2×2 comparison the project will produce

| Architecture \ Recipe       | BCE / classifier head (no recipe) | FaCoRNet recipe (relation_guided + relation_matched) |
|-----------------------------|-----------------------------------|------------------------------------------------------|
| FaCoR top-only              | **M10 R003** (Test AUC 0.7478)    | **M11 R001**                                         |
| Multi-stage (stages 3 + 4)  | **M09 R001** (Test AUC 0.7982)    | **M11 R002**                                         |

Differences M11 R001 → M11 R002 isolate the architecture given the
FaCoRNet recipe is held constant. Differences M10/M09 → M11 R001/R002
isolate the recipe given the architecture is held constant. Together
they give a clean 2×2 ablation.

## Reference

The FaCoRNet recipe in this model is inspired by published kinship
verification work using AdaFace + FaCoR-style attention + dynamic
temperature contrastive losses. The exact `RelationGuidedContrastiveLoss`
implementation in `models/shared/losses.py` is a project-specific
formulation derived from the published descriptions, not a verbatim port.

## Files

```
11_facornet_adaface/
├── README.md                # This file
├── model.py                 # Same `AdaFaceFaCoR*` classes as M10 (no changes)
├── adaface_iresnet.py       # AdaFace IR-101 backbone (copied from M10)
├── age_dataset.py           # Inert — SAM age augmentation not used by FaCoRNet
├── AMD/
│   ├── train.py
│   ├── test.py
│   ├── evaluate.py
│   └── run_pipeline.sh      # Defaults: LOSS=relation_guided, TEMP=0.07, neg=relation_matched
├── weights/                 # Symlinked to M10's AdaFace .pth
├── data/                    # (gitignored)
└── run-review/              # Per-run analyses (manual)
```
