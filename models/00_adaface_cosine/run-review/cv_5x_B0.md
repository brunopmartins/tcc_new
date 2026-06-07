# Baseline B0 — 5-fold Cross-Validation (deterministic)

**Period:** 2026-06-01 (≈10 min, no training — inference only)
**Hardware:** ROCm 5.7, single AMD RX 6750 XT (used only for embedding extraction).
**Folds:** family-disjoint 5-fold over RFIW Track-I `train-pairs.csv` families,
official RFIW Track-I test set (13,425 face-level pairs) preserved unchanged.
Same splitter as M02/M12: [shared/dataset.py `create_fiw_5fold_train_val_loaders`](../../shared/dataset.py).
Evaluator: [cv_eval.py](../cv_eval.py).

## Motivation

B0 (AdaFace IR-101 frozen + cosine similarity, no training) is the
no-adaptation reference in the TCC comparison. The other models were
CV'd, so B0 needs a CV band for the comparison table to be uniform.

The key property: **B0 has zero trainable parameters.** For a frozen
model on a fixed test set, the test scores are identical every time.
Threshold-invariant metrics (AUC, AP, TAR@FAR) are therefore
**deterministic** — `std = 0` by construction, not by luck. Only the
val-selected F1-optimal threshold can vary per fold, and that only
affects threshold-dependent metrics (Accuracy, F1, Precision, Recall).

## B0 recipe (replicated exactly)

| Knob | Value |
|---|---|
| Backbone | AdaFace IR-101 (WebFace4M), **fully frozen** |
| Adaptation | none — pure inference |
| Pair score | cosine similarity of L2-normalised 512-d embeddings, mapped from [-1,1] to [0,1] |
| Threshold | F1-optimal on the per-fold val split |
| Image size | 112×112 |
| Test set | RFIW Track-I test, 13,425 pairs (fixed) |
| Seed | 42 |

## How the CV works for a frozen model

Per fold:

1. The fold's val split (different families per fold) is scored.
2. An F1-optimal threshold is picked on that val split.
3. The **same cached test scores** are evaluated at that threshold.

Test embeddings/similarities are computed **once** (deterministic) and
reused for all 5 folds; only the threshold-pick step differs.

## Per-fold results

| Fold | Val threshold | Test AUC | AP | Acc | F1 | TAR@.001 | TAR@.01 | TAR@.1 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.5000 | 0.7991 | 0.8093 | 0.6660 | 0.7121 | 0.0706 | 0.2178 | 0.5238 |
| 1 | 0.5000 | 0.7991 | 0.8093 | 0.6660 | 0.7121 | 0.0706 | 0.2178 | 0.5238 |
| 2 | 0.5000 | 0.7991 | 0.8093 | 0.6660 | 0.7121 | 0.0706 | 0.2178 | 0.5238 |
| 3 | 0.5000 | 0.7991 | 0.8093 | 0.6660 | 0.7121 | 0.0706 | 0.2178 | 0.5238 |
| 4 | 0.5000 | 0.7991 | 0.8093 | 0.6660 | 0.7121 | 0.0706 | 0.2178 | 0.5238 |

All 5 folds picked the **same** val threshold (0.5000), so even the
threshold-dependent metrics came out identical. The per-fold rows are
literally the same numbers.

## Aggregate (mean ± std, n=5)

| Metric | mean | std |
|---|---:|---:|
| Test ROC AUC | 0.7991 | 0.0000 |
| Test Average Precision | 0.8093 | 0.0000 |
| Test Accuracy | 0.6660 | 0.0000 |
| Test Balanced Accuracy | 0.6739 | 0.0000 |
| Test Precision | 0.6065 | 0.0000 |
| Test Recall | 0.8623 | 0.0000 |
| Test F1 | 0.7121 | 0.0000 |
| TAR @ FAR=0.001 | 0.0706 | 0.0000 |
| TAR @ FAR=0.01 | 0.2178 | 0.0000 |
| TAR @ FAR=0.1 | 0.5238 | 0.0000 |

## Single-run vs CV

The single-run B0 reported in the TCC (Test AUC 0.799, AP 0.809,
TAR@FAR=0.001 0.071, TAR@FAR=0.01 0.218, TAR@FAR=0.1 0.524) is **exactly
equal** to the CV mean. There is no favorable-draw gap because there is
no randomness: B0 is a fixed function of the test set.

This means the † marker on B0 in the Resultados table ("single-run") is
structurally equivalent to "CV-validated" for the threshold-invariant
metrics — the single run *is* the CV mean.

## Notable: B0 is competitive at low-FAR

A no-training frozen face embedding lands at:

- TAR@FAR=0.01 = 21.78 % — basically tied with M12 R011 CV ensemble
  (21.72 %) and above M12 R011 CV mean (20.69 %).
- TAR@FAR=0.001 = 7.06 % — above M12 R011 CV mean (6.77 %), close to the
  R011 CV ensemble (8.01 %).

The AdaFace identity embedding already separates a meaningful fraction
of kin pairs at strict operating points. Part of the supervised models'
job is to *reach and surpass* this floor, not build from scratch. This
is one of the more interesting framing points for the discussion: the
gain of the trained models over B0 is in aggregate ranking (AUC/AP) and
mid-FAR, not uniformly across all operating points.

## Implications for the TCC narrative

1. **Quote B0 as 0.7991 (no ± needed, or ± 0.000).** It is the same
   under single-run and CV. The † footnote can note "deterministic —
   frozen model, fixed test set".
2. **B0's strict-FAR competitiveness is a real finding,** not noise.
   It should be stated explicitly when discussing why TAR@FAR is not a
   metric the supervised models dominate as cleanly as AUC.
3. **B0 is the floor the VLM zero-shot baselines fail to reach** — the
   Claude Sonnet binary VLM (AUC 0.7888) lands *below* B0's 0.7991,
   which strengthens the "general multimodal knowledge ≠ kinship
   discrimination" argument.

## Artifacts

- Metrics + per-fold raw: `output/cv/cv_metrics.txt`
- Continuous scores + labels + thresholds: `output/cv/cv_probs.npz`
- Evaluator: [cv_eval.py](../cv_eval.py)

(Both `output/` artifacts are git-ignored per the project `.gitignore`
rule `**/output/*`; reproduce with the command below.)

## Reproducing

```bash
PYTHONPATH=models:models/shared \
models/12_rgck_net/.venv/bin/python models/00_adaface_cosine/cv_eval.py \
  --data_root datasets/FIW \
  --aligned_root datasets/FIW_aligned \
  --batch_size 32 --num_workers 4
```
