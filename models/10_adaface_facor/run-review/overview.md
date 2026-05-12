# Overview — Model 10: AdaFace + FaCoR Cross-Attention

**Model:** 10 — AdaFace IR-101 (WebFace4M) + FaCoR Cross-Attention + supervised
cosine contrastive loss
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Dataset:** FIW (FIW_aligned 224×224 → resized to 112 at load time)

This file aggregates per-run results. Individual run details land in
`run-NNN.md` files alongside this overview.

---

## Hypothesis recap

M10 takes M02's best-performing recipe (Test ROC-AUC 0.850 with ViT-B/16
end-to-end + bidirectional FaCoR cross-attention + cosine contrastive m=0.3)
and swaps the ImageNet ViT backbone for **AdaFace IR-101 pretrained on
WebFace4M**. The expectation is that face-discriminative pretraining +
end-to-end fine-tuning will let the FaCoR cross-attention pick up on
kinship-specific cues that ImageNet features cannot represent natively, while
avoiding the "anti-kinship trap" that frozen ArcFace exhibits in M08.

An optional axis added in R001 was **SAM-based age augmentation** —
preprocessing each face into child / young-adult / elderly variants and
ensembling at the token level for generational invariance. R001 rejected
this design (see below). Future runs should focus on the clean M10 design
first and reattempt age augmentation, if at all, as per-sample input
augmentation rather than feature-level ensemble.

## Configuration baseline (defaults from run_pipeline.sh)

| Knob               | Default                  |
|--------------------|--------------------------|
| Backbone           | AdaFace IR-101 WebFace4M |
| Input              | 112×112, [-1, 1]         |
| Tokens             | 49 (7×7), 512-d          |
| Pos embed          | learnable (49 tokens)    |
| Global embed       | enabled (AdaFace pool)   |
| Cross-attn layers  | 2                        |
| Cross-attn heads   | 8                        |
| Dropout            | 0.2                      |
| Loss               | cosine_contrastive       |
| Temperature        | 0.3                      |
| Margin             | 0.3                      |
| Backbone trainable | yes (end-to-end)         |
| LR (peak)          | 5e-6                     |
| Scheduler          | cosine, warmup 5         |
| Min LR             | 1e-7                     |
| Batch              | 8 × grad-accum 4 (eff 32)|
| Patience           | 50                       |
| Epochs (max)       | 100                      |

## Run table

| | Run 001 | Run 002 | Run 003 |
|---|---|---|---|
| **Date** | 2026-05-11 | 2026-05-11 | 2026-05-12 |
| **Purpose** | M10 + SAM age ensemble (4× variants, w0=0.5) | M10 clean baseline (M02 recipe, cosine_contrastive) | M10 + BCE classifier head |
| **Status** | Aborted at iter 110/16604 of epoch 5 | Aborted at iter 148/8302 of epoch 6 | **Stopped at ep 18** for early eval |
| **Best Val AUC** | 0.6685 (ep 1) — peak; then monotonic decline | 0.7099 (ep 3) — peak; then collapse to 0.57 | **0.8875 (ep 17)** |
| **Test ROC-AUC** | not computed | not computed | **0.7478** |
| **Test Accuracy** | not computed | not computed | 70.55 % |
| **Notes** | Token-level age ensemble erases cross-generational signal. See [run-001.md](run-001.md). | cosine_contrastive + AdaFace full-FT unstable; threshold flips. See [run-002.md](run-002.md). | First stable M10 run, but val→test gap -0.140 indicates overfit to val family pool. See [run-003.md](run-003.md). |

---

## Test metrics side-by-side

| Metric | Run 001 | Run 002 | **Run 003** |
|---|---:|---:|---:|
| **Test ROC-AUC** | — | — | **0.7478** |
| Test Accuracy | — | — | 70.55 % |
| Test F1 | — | — | 0.6262 |
| Test Precision | — | — | 79.92 % |
| Test Recall | — | — | 51.48 % |
| Avg Precision | — | — | 0.7505 |
| TAR@FAR=0.01 | — | — | 9.31 % |
| TAR@FAR=0.1 | — | — | 46.55 % |
| Best Val ROC-AUC | 0.6685 (ep1) | 0.7099 (ep3) | **0.8875 (ep17)** |
| Best Val Accuracy | 51.4 % (ep1) | 54.6 % (ep1) | 81.6 % (ep14) |
| Val→Test AUC gap | — | — | **-0.140** |

---

## Issues Log

| # | Severity | Status | Title | Notes |
|---|----------|--------|-------|-------|
| I-01 | High   | Open (closed for the current SAM design) | Token-level age ensemble destroys kinship signal | R001 monotonic Val-AUC decline (0.6685→0.5718 in 4 epochs) with train loss collapsing 6× faster than M02. Reattempt only as per-sample-per-epoch input augmentation, not feature-level mean. |
| I-02 | High   | Closed in R003 | cosine_contrastive + AdaFace IR-101 full-FT is unstable | R002 oscillated 0.57–0.71 across warmup. Pivoting to BCE classifier head (R003) gave a clean monotonic warmup → unlock → plateau curve. |
| I-03 | High   | Open | M10 R003 overfits to val family pool (-0.140 val→test gap) | Val 0.8875 → Test 0.7478. M02 R031 had -0.030 gap. AdaFace's identity-cluster pretrain memorises seen families. Next experiments: strong reg (R004), partial freeze (R005), or smaller LR (R006). |
| I-04 | High   | Open | Grandparent classes collapse on M10 R003 | gfgd 26.8 %, gmgd 21.1 %, gfgs 28.6 %, gmgs 33.9 % accuracy on test — well below random for two of them. M02 R031 maintained 88-96 % across all relations. R003 has no relation-conditional discrimination. |
| I-05 | Info   | Workaround applied | Checkpoint missing `model_config` when training is killed mid-pipeline | `train.py` writes `model_config` to best.pt only after the full pipeline completes. R003 best.pt had to be patched manually before test.py could rebuild the `AdaFaceFaCoRClassifier` wrapper. Fix: write `model_config` in the trainer's `save_checkpoint`, not in train.py's finaliser. |

---

## Comparison with other models (FIW, Test ROC-AUC)

| Model | Test ROC-AUC | Test Acc | Backbone | Cross-pair signal | Notes |
|---|---:|---:|---|---|---|
| M02 R031 | **0.850** | 74.4 % | ViT-B/16 (ImageNet, full FT) | FaCoR top-only | best in project |
| M06 R001 | 0.776 | 69.8 % | ViT-B/16 (frozen) + retrieval | retrieval + cross-attn | best frozen-encoder |
| **M09 R001** | **0.7982** | 71.9 % | AdaFace IR-101 (full FT) + **multi-stage cross-attn (stages 3+4)** | inside-backbone pair interaction | **best AdaFace-based, val→test gap -0.094** |
| M10 R003 | 0.7478 | 70.6 % | AdaFace IR-101 (full FT) + BCE classifier head | FaCoR top-only + MLP on diff/product | stable convergence, val→test gap -0.140 |
| M08 R001 | 0.693 | 60.8 % | ArcFace IR-100 (frozen) + retrieval | retrieval + cross-attn | anti-kinship trap |
| M10 R001 | (aborted) | — | AdaFace IR-101 + SAM age ensemble | FaCoR top-only over age-averaged tokens | feature collapse |
| M10 R002 | (aborted) | — | AdaFace IR-101 (full FT) + cosine_contrastive | FaCoR top-only | warmup oscillation 0.57–0.71 |

---

## Conclusion (as of R003)

R001, R002 and R003 together rule out three M10 design choices:

1. **R001** rejects *feature-level* SAM age ensembling (token average
   across age variants destroys cross-generational kinship signal).
2. **R002** rejects *cosine_contrastive on AdaFace IR-101 full-FT* — the
   loss doesn't pin embedding orientation, the backbone's identity
   cluster prior resists, and the optimisation oscillates rather than
   converges.
3. **R003** is the first stable M10 run (Val AUC 0.8875) but **fails
   the project headline metric**: Test AUC 0.7478 << M02 R031's 0.850.
   The val→test gap is -0.140 vs M02's -0.030, indicating the model
   memorised the validation family pool rather than learning
   generalisable kinship features.

R003 reveals a structural problem: AdaFace IR-101's identity-
discriminative pretrain is a *liability* for kinship verification on
held-out families. The same backbone that gave M08 its anti-kinship
trap (frozen mode) gives M10 family-memorisation (full-FT mode).

**Per-relation finding (R003):** all kin relations collapse to 21–58 %
accuracy vs M02 R031's 88–96 %. Grandparent classes (gfgd/gmgd/gfgs/gmgs)
are worst — *worse than random* for two of them. The model has no
relation-conditional discrimination, suggesting it learned a global "this
family or not" rule instead of generalisable kinship signatures.

**Next directions:**
- M09 (multi-stage cross-attention) — tests whether interleaving pair
  processing with feature extraction recovers the kinship signal.
- M10 R004 with strong regularisation (`DROPOUT=0.4`,
  `WEIGHT_DECAY=1e-3`) to suppress the memorisation behaviour.
- M10 R005 with partial backbone freeze (only fine-tune stages 3+4) to
  keep the pretrained features more intact.
- M10 R006 with much smaller LR (1e-6 or 2e-6) for gentler fine-tune.
