# Model 11 — AdaFace IR-101 + FaCoRNet treatments (label-aware)

## Important context (2026-05-13)

The initial "FaCoRNet recipe" framing assumed the project's existing
`relation_guided` and `cosine_contrastive` losses were usable for our
mixed kin/non-kin batches. **They are not.** A pre-run audit
([discussao.md](discussao.md)) showed that both losses ignore the
batch labels — they treat every `(emb1_i, emb2_i)` correspondence as a
positive pair in the InfoNCE numerator, regardless of whether the dataset
labelled it kin (1) or non-kin (0). Combined with `relation_matched`
hard negatives, this trained the model to *pull non-kin pairs closer*,
producing `predict-all-kin` degenerate behaviour (Val Acc 50 %,
per-relation 1.000 uniform).

This is documented in:

- `models/shared/losses.py:74` — comment "Optional labels (not used in
  standard InfoNCE)" inside `CosineContrastiveLoss.forward`
- `models/shared/losses.py:258` — `RelationGuidedContrastiveLoss.forward`
  signature accepts only `emb1, emb2, attention_map` (no labels)

The original FaCoRNet paper (Su et al., ICCV Workshops 2023) constructs
its batches differently: positives come from the correspondence index;
negatives are *implicit* via cross-pair terms in the InfoNCE denominator
(`(x_i, x_j)` and `(x_i, y_j)`). Our project's dataset emits *explicit*
`label ∈ {0,1}` pairs, so the losses need to be label-aware to use them.

**M11 has been re-scoped accordingly.** See "Current scope" below.

## Current scope (post bug discovery)

M11 tests the **single FaCoRNet treatment that can be applied to our
existing pipeline without breaking the loss semantics**: `relation_matched`
hard negative sampling. The architecture and loss stack stay aligned with
M09 R001 (multi-stage cross-attention + BCE classifier head, the
combination that achieved Test AUC 0.7982).

> **Does adding `relation_matched` hard negatives (FaCoRNet's negative
> sampling strategy) to the M09 R001 stack improve over M09 R001's random
> negatives?**

If this works, it isolates the value of hard-negative mining. A future
**M11 R002** can implement a faithful, label-aware Rel-Guide loss (Caminho A
or Caminho B from `discussao.md`) and test the loss-side of the FaCoRNet
recipe separately.

## Recipe (post bug discovery)

M11 R001 isolates the single FaCoRNet-inspired knob that works with our
existing dataset/loss contract:

| Component                    | M09 R001                | **M11 R001 (current)**                            |
|------------------------------|-------------------------|---------------------------------------------------|
| Backbone                     | AdaFace IR-101 full FT  | **Same**                                          |
| Cross-attention              | Multi-stage (3+4) FaCoR | **Same**                                          |
| Positional embedding         | Learnable per-stage     | **Same**                                          |
| Global embedding             | AdaFace pool            | **Same**                                          |
| Classifier head              | enabled (MLP on diff/prod) | **Same**                                       |
| Loss                         | bce on classifier output | **Same (bce)**                                   |
| **Train neg. sampling**      | random                  | **`relation_matched`** (hard negatives)           |
| **Eval neg. sampling**       | random                  | **`relation_matched`**                            |
| LR (peak), warmup, dropout   | 5e-6, 5, 0.2            | Same                                              |
| Batch                        | 4 × grad-accum 8 (32)   | Same                                              |

The only knob changed from M09 R001 to M11 R001 is the negative
sampling strategy. Everything else is identical.

### What the failed FaCoRNet-recipe attempts revealed

Three M11 R001 attempts were killed early (each at ep 1-2) before the
bug was diagnosed:

| Attempt | Config                                  | Outcome                                       |
|---------|-----------------------------------------|-----------------------------------------------|
| v1      | `relation_guided`, temp 0.07, multistage | Train loss collapsed 0.80→0.02 in 1 epoch; per-rel uniform 1.000; Val Acc 50 % |
| v2      | `relation_guided`, temp 0.3, multistage  | Slower collapse but same degenerate state     |
| v3      | `cosine_contrastive`, temp 0.3, multistage | Same degenerate state                       |

Common signature: `predict-all-kin` regime, Val Acc stuck at 50 %, Val
AUC 0.60-0.69 (barely above random). Root cause: both losses ignore the
batch labels and treat every (emb1_i, emb2_i) correspondence as a
positive in the InfoNCE numerator. The dataset's `label=0` pairs were
trained to be pulled together along with the `label=1` pairs.

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

| Model       | Architecture                | Recipe                                                  |
|-------------|-----------------------------|---------------------------------------------------------|
| M10 R003    | FaCoR top-only              | BCE + classifier head + **random** negatives            |
| M09 R001    | Multi-stage (stages 3+4)    | BCE + classifier head + **random** negatives            |
| **M11 R001**| Multi-stage (stages 3+4)    | BCE + classifier head + **`relation_matched`** negatives|
| M11 R002 (planned) | Multi-stage          | label-aware Rel-Guide (faithful port) + `relation_matched` |

## Launch

### R001 — multi-stage + BCE classifier head + relation_matched negatives (current)

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
BATCH_SIZE=4 GRAD_ACCUM=8 \
USE_MULTISTAGE=1 \
USE_CLASSIFIER_HEAD=1 \
LOSS=bce \
NUM_WORKERS=4 SEED=42 \
bash models/11_facornet_adaface/AMD/run_pipeline.sh
```

(Defaults TRAIN_NEGATIVE_STRATEGY and EVAL_NEGATIVE_STRATEGY are both
`relation_matched` in the pipeline script — pass them explicitly if you
want random for a control run.)

### R002 (planned) — label-aware Rel-Guide

R002 will exercise Caminho B from [discussao.md](discussao.md):
implement a label-aware version of the FaCoR-style attention-driven
contrastive loss (using the dataset's `label` to choose pull vs push per
pair) and run with multistage + relation_matched negatives. This is
where the "FaCoR loss" angle of the recipe actually gets tested.
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

| Configuration               | BCE + classifier head (M09 stack) | label-aware Rel-Guide (planned)     |
|-----------------------------|-----------------------------------|--------------------------------------|
| random negatives            | **M09 R001** (Test AUC 0.7982)    | not planned                          |
| **`relation_matched` negs** | **M11 R001** (current run)        | **M11 R002 (planned)**               |

M11 R001 vs M09 R001 isolates the `relation_matched` negative sampling
contribution. M11 R002 vs M11 R001 would isolate the loss-side
contribution (faithful FaCoR Rel-Guide vs BCE).

## Reference

- Su et al., *Kinship Representation Learning with Face Componential Relation*, ICCV Workshops 2023 — the FaCoRNet paper this model is inspired by.
- Official code: `wtnthu/FaCoR` — particularly `losses.py` and `train_p.py`.
- [discussao.md](discussao.md) — local audit identifying the
  label-vs-loss contract bug in our `RelationGuidedContrastiveLoss` and
  `CosineContrastiveLoss` implementations.

The project's `RelationGuidedContrastiveLoss` (in `models/shared/losses.py`)
is *inspired* by FaCoR's Rel-Guide concept (attention-driven dynamic
temperature) but is not a faithful port. Differences:

- Paper: `psi = M(beta) / s` with `M = global sum pool`, `s = 500`
- Local: `temperature = base_temperature * (1 + alpha * mean(attention_map))`

The paper's formulation gives a much wider temperature range. Ours
collapses to ~constant `base_temperature` because `mean(softmaxed_attention)`
is approximately a constant.

## Files

```
11_facornet_adaface/
├── README.md                # This file
├── discussao.md             # Bug audit + path forward
├── model.py                 # Same `AdaFaceFaCoR*` classes as M10 (no changes)
├── adaface_iresnet.py       # AdaFace IR-101 backbone (copied from M10)
├── age_dataset.py           # Inert — SAM age augmentation not used
├── AMD/
│   ├── train.py             # Supports --use_multistage flag (imports M09's build via importlib.util)
│   ├── test.py
│   ├── evaluate.py
│   └── run_pipeline.sh      # Defaults: USE_MULTISTAGE=0, LOSS=relation_guided (DO NOT USE w/o head); pass USE_CLASSIFIER_HEAD=1 LOSS=bce for the safe R001 recipe
├── weights/                 # Symlinked to M10's AdaFace .pth
├── data/                    # (gitignored)
└── run-review/              # Per-run analyses (manual)
```
