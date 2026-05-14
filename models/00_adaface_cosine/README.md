# Baseline B0 — AdaFace Cosine Frozen

Implementation of **§5 of `baselines_rgck_net_tcc.md`**: the minimal
no-adaptation reference for the kinship verification experiments.

## What this is

- AdaFace IR-101 (WebFace4M) backbone, **fully frozen**
- For each pair `(img_a, img_b)`: extract L2-normalised 512-d embeddings,
  compute cosine similarity, score the pair
- F1-optimal threshold is selected on the validation split, applied to test

**No training.** This is pure inference — the script runs once and
emits the validation-tuned test metrics.

## Why this baseline matters

Per the proposal, B0 answers:

> What kinship verification performance can you get from an
> off-the-shelf face recognition model with no kinship-specific
> adaptation?

If RGCK-Net (the proposed model) substantially exceeds B0, that's
direct evidence the kinship task requires more than face identification.

## Quick start

```bash
SKIP_INSTALL=1 \
ALIGNED_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
models/00_adaface_cosine/.venv/bin/python models/00_adaface_cosine/evaluate.py \
  --dataset fiw \
  --data_root /home/bruno/Desktop/tcc_new/datasets/FIW \
  --aligned_root /home/bruno/Desktop/tcc_new/datasets/FIW_aligned \
  --batch_size 32 \
  --num_workers 4
```

To run on CPU when the GPU is busy with another model:

```bash
... evaluate.py --cpu ...
```

## Outputs

- `output/test_metrics_rocm.txt` — full val + test metrics, per-relation breakdown
- `output/scores.npz` — raw cosine similarities, labels, relations for both splits + selected threshold

## Conventions (matched against other project models)

- **Image preprocessing**: FIW_aligned 224×224 → resize 112×112 → normalize `[-1, 1]` (mean 0.5, std 0.5) — identical to M09/M10/M11/M12
- **Split seed**: 42 (FIW Track-I default)
- **Threshold selection**: F1-optimal on validation
- **Cosine sim → score**: `(cos + 1) / 2` to map `[-1, 1]` to `[0, 1]` so the threshold-finder and per-relation metrics behave the same way as for the trained models
- **Per-relation metrics**: computed at the val-chosen threshold

## What this isn't

- Not a model in the project's `train.py` / `test.py` /
  `run_pipeline.sh` sense. There's no training step.
- Not a contribution. It's a reference number.

## Files

```
00_adaface_cosine/
├── README.md           # This file
├── evaluate.py         # Single script: extract embeddings → cos sim → thresholded metrics
├── adaface_iresnet.py  # Symlink to M10's
├── weights/            # Symlinked AdaFace .pth
└── run-review/         # Result analyses (manual)
```

## Reference

`baselines_rgck_net_tcc.md` §5 — Baseline B0 specification.
