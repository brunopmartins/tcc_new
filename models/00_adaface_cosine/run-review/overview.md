# Overview — Baseline B0: AdaFace Cosine Frozen

**Baseline:** B0 — implementation of `baselines_rgck_net_tcc.md` §5
**Purpose:** No-adaptation reference for the RGCK-Net comparison
**GPU:** AMD Radeon RX 6750 XT (inference only)
**Dataset:** FIW Track-I (split_seed=42)

This is **not a model** in the project's train/test/wrap-up sense —
there's no training step. It's a single-script evaluation that
measures what an off-the-shelf face recognition model can do on
kinship verification without any kinship-specific adaptation.

---

## Hypothesis

> What kinship verification performance is achievable from an off-the-shelf
> face recognition model (AdaFace IR-101) with no kinship-specific
> training?

Per `baselines_rgck_net_tcc.md` §5.5: if the proposed RGCK-Net
substantially exceeds B0, that's evidence the kinship task requires
more than face identification.

---

## Configuration

| Knob | Value |
|---|---|
| Backbone | AdaFace IR-101 (WebFace4M weights, all frozen) |
| Input | 112×112 (FIW_aligned 224 resized at load), `[-1, 1]` norm |
| Pair score | `(cosine(emb_A, emb_B) + 1) / 2` ∈ `[0, 1]` |
| Threshold | F1-optimal on validation |
| Training | None |
| Time | ~7 min on GPU (24,815 pair forwards) |

---

## Run table

| | **Run 001** |
|---|---|
| Date | 2026-05-14 |
| Val F1-optimal threshold | 0.5000 (cosine sim = 0.0) |
| Val ROC AUC | 0.8234 |
| **Test ROC AUC** | **0.7991** |
| Test Accuracy | 66.6 % |
| Test F1 | 0.7121 |
| Avg Precision | 0.8093 |
| TAR @ FAR=0.001 | **7.06 %** (highest in project) |
| TAR @ FAR=0.01 | **21.78 %** (highest in project) |
| TAR @ FAR=0.1 | 52.38 % |
| **Val→test AUC gap** | **-0.024** (smallest in project) |

---

## Comparison with project models

| Model | Test ROC AUC | Test Accuracy | Val→test gap | Trainable params |
|---|---:|---:|---:|---:|
| **M12 R002** (project headline) | **0.8564** | 76.79 % | -0.076 | 31.6 M |
| M02 R031 | 0.850 | 74.4 % | -0.031 | 86 M (full FT) |
| M12 R003 | 0.8510 | 76.4 % | -0.080 | 31.6 M |
| M05 R007 | 0.810 | — | — | — |
| M09 R001 | 0.7982 | 71.9 % | -0.094 | 70.9 M |
| **B0 AdaFace Cosine** | **0.7991** | 66.6 % | **-0.024** | **0** |
| M09 R002 | 0.7824 | 71.6 % | -0.107 | 70.9 M |
| M06 R001 | 0.776 | 69.8 % | — | — |
| M11 R001 v4 | 0.7707 | 70.6 % | -0.128 | 70.9 M |
| M10 R003 | 0.7478 | 70.6 % | -0.140 | 72.6 M |
| M12 R001 | 0.7464 | 68.0 % | -0.089 | 5.6 M |
| M08 R001 | 0.693 | 60.8 % | — | — |

---

## Key observations

1. **B0 beats M09 R001 on Test AUC** (0.7991 vs 0.7982) despite zero
   training. Four trained AdaFace-based models (M09 R001/R002, M11 v4,
   M10 R003, M12 R001) are at or below B0's Test AUC. This is a strong
   signal that *most of the kinship-relevant information is already
   encoded in AdaFace embeddings*.

2. **B0 has the highest TAR@FAR=0.001 of any model** (7.06 %, vs M12
   R002's 4.18 % and M02 R031's 2.5 %). At very low false-acceptance
   rates — the strict-verification regime — raw AdaFace cosine is the
   strongest discriminator. This is the regime AdaFace was trained for
   (identity verification at low FAR).

3. **B0 has the smallest val→test gap** (-0.024). No training = no
   memorisation. This is the true generalisation ceiling for AdaFace
   embeddings on FIW.

4. **M12 R002 beats B0 by +0.057 Test AUC** (0.8564 vs 0.7991). This
   measures the architectural contribution of RGCK-Net (region tokens,
   cross-region attention, sigmoid gating, partial backbone unfreeze).
   It's a real and meaningful gain — about 7 % relative over baseline
   — but smaller than naive expectations would suggest.

---

## Per-relation accuracy (B0 at threshold 0.500)

| Relation | N | acc |
|----------|--:|----:|
| sibs | 234 | 90.6 % |
| bb | 860 | 89.9 % |
| ss | 731 | 89.9 % |
| fs | 1135 | 89.9 % |
| md | 1038 | 89.4 % |
| ms | 1036 | 86.6 % |
| fd | 918 | 80.1 % |
| gmgd | 123 | 75.6 % |
| gfgd | 138 | 72.5 % |
| gfgs | 98 | 68.4 % |
| gmgs | 121 | 52.9 % |
| non-kin | 6993 | 48.6 % |

At the F1-optimal threshold, B0 is in a high-recall/low-precision
regime — it captures most kin pairs (kin accuracy 53-91 %) but
misclassifies more than half of non-kin pairs (non-kin accuracy 48.6 %).

This is consistent with cosine similarity on AdaFace being a *noisy*
kinship signal — useful but not cleanly separable from non-kin.

---

## Conclusion

B0 establishes the **no-adaptation floor** for the project. It
surprisingly outperforms several trained AdaFace-based models on
Test AUC, demonstrating that:

1. AdaFace embeddings inherently encode kinship-relevant information
   because family members share facial features that AdaFace's
   identity-discriminative training preserves.
2. Naive cosine thresholding on AdaFace already captures ~80 % of the
   ROC AUC achievable with trained architectures on FIW.
3. The proposed RGCK-Net's contribution (+0.057 over B0) is real and
   measurable, but the architectural gain is moderate, not dramatic.

For the TCC, B0 is exactly the comparison the proposal §5.5 specifies.
The honest comparison is **+0.057 Test AUC for M12 R002 over B0**
(7 % relative improvement), with proportionate gains across most
threshold-invariant metrics — except TAR@FAR=0.001 where B0 actually
wins (AdaFace's identity training shines at very low FAR).

The next baselines (B1 = AdaFace Global-MLP Partial FT, B2 = AdaFace
Global-MLP Frozen) are essential to isolate the architectural
contributions more precisely.
